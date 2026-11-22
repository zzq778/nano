#!/usr/bin/env python3
"""Termux GitHub bridge: safe-by-default remote actions for this Android phone.

Requests: private repo phone_requests/<id>.json
Results:  private repo phone_results/<id>.json
No inbound port. Safe defaults; opt-in `--enable-dev` and `--enable-adb`
or `--enable-root` expand capabilities. No bootloader unlock or rooting.
"""
import argparse
import base64
import datetime as dt
import getpass
import io
import shlex
import json
import os
import pathlib
import platform
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

VERSION = "2.0"
REPO = "zzq778/jetson"
BRANCH = "main"
CONFIG_DIR = pathlib.Path.home() / ".config" / "chat-mobile-agent"
TOKEN_FILE = CONFIG_DIR / "token"
WORKSPACE = pathlib.Path.home() / "chat_mobile_workspace"
POLL_SECONDS = 30
MAX_RESULT_CHARS = 12000
MAX_WRITE_CHARS = 40000
ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
SAFE_ACTIONS = {"ping", "device_info", "list_files", "read_file", "write_file"}
ADB_ACTIONS = {"adb_status", "android_tap", "android_swipe", "android_keyevent", "android_ui_dump", "android_screenshot", "android_input_text", "android_launch_app", "android_open_url", "android_batch", "android_status"}
DEV_COMMANDS = {"git", "python", "python3", "pytest", "gcc", "g++", "clang", "clang++", "cmake", "make", "ninja"}


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")


class Github:
    def __init__(self, token):
        self.token = token

    def api(self, method, path, data=None):
        url = "https://api.github.com/" + path.lstrip("/")
        headers = {
            "Authorization": "Bearer " + self.token,
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "chat-mobile-termux-agent/1.0",
        }
        body = json_bytes(data) if data is not None else None
        if body is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                raw = response.read()
                return json.loads(raw.decode("utf-8")) if raw else None
        except urllib.error.HTTPError as exc:
            # Do not print the Authorization header or token.
            raise RuntimeError("GitHub HTTP {}: {}".format(exc.code, exc.read(240).decode("utf-8", "replace"))) from None

    def names(self, directory):
        items = self.api("GET", "repos/{}/contents/{}?ref={}".format(REPO, directory, BRANCH))
        if not isinstance(items, list) or len(items) >= 1000:
            raise RuntimeError("Invalid/full queue directory: " + directory)
        return {item.get("name", "")[:-5] for item in items
                if item.get("type") == "file" and item.get("name", "").endswith(".json")
                and ID_RE.fullmatch(item.get("name", "")[:-5])}

    def pending_ids(self):
        return sorted(self.names("phone_requests") - self.names("phone_results"))[:10]

    def get_request(self, job_id):
        obj = self.api("GET", "repos/{}/contents/phone_requests/{}.json?ref={}".format(REPO, job_id, BRANCH))
        if obj.get("encoding") != "base64":
            raise RuntimeError("Not a base64 request")
        raw = base64.b64decode(obj["content"], validate=False)
        if len(raw) > 64 * 1024:
            raise RuntimeError("Request exceeds 64 KiB")
        return json.loads(raw.decode("utf-8"))

    def put_result(self, job_id, result):
        payload = {
            "message": "Phone agent response: " + job_id,
            "branch": BRANCH,
            "content": base64.b64encode(json_bytes(result)).decode("ascii"),
        }
        return self.api("PUT", "repos/{}/contents/phone_results/{}.json".format(REPO, job_id), payload)


def path_in_workspace(relative_path):
    if not isinstance(relative_path, str) or len(relative_path) > 400 or "\0" in relative_path:
        raise ValueError("Invalid relative path")
    candidate = pathlib.Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("Absolute paths and '..' are forbidden")
    root = WORKSPACE.resolve()
    resolved = (root / candidate).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError("Path escapes workspace (including symlink)")
    return resolved


def capture(argv, timeout=15, cwd=None):
    # Keep unbounded process output off RAM; only the first 12 KiB enter GitHub.
    try:
        with tempfile.TemporaryFile() as stream:
            proc = subprocess.Popen(argv, cwd=str(cwd) if cwd else None,
                                    stdout=stream, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            timed_out = False
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                timed_out = True
            stream.seek(0)
            output = stream.read(MAX_RESULT_CHARS + 1)
            return {"returncode": proc.returncode, "timed_out": timed_out,
                    "output": output[:MAX_RESULT_CHARS].decode("utf-8", "replace"),
                    "truncated": len(output) > MAX_RESULT_CHARS}
    except OSError as exc:
        return {"returncode": 127, "timed_out": False,
                "output": "{}: {}".format(type(exc).__name__, exc), "truncated": False}


def adb(args, timeout=20):
    if not shutil.which("adb"):
        raise RuntimeError("adb not installed: pkg install android-tools")
    return capture(["adb"] + args, timeout=timeout)


def root_check():
    """Read-only check. Never attempt to obtain root or change the bootloader."""
    if not shutil.which("su"):
        return False
    probe = capture(["su", "-c", "id"], timeout=4)
    return probe["returncode"] == 0 and "uid=0" in probe["output"]


def ui_backend(argv, enable_adb=False, enable_root=False, timeout=20):
    """Use the shell privilege of a previously paired ADB or installed su."""
    if enable_adb:
        # All argv values are constructed by validated UI functions below.
        return adb(["shell"] + argv, timeout=timeout)
    if enable_root and root_check():
        return capture(["su", "-c", shlex.join(argv)], timeout=timeout)
    raise RuntimeError("Neither paired ADB nor existing root is available")


def ui_binary(argv, enable_adb=False, enable_root=False, timeout=30):
    if enable_adb:
        command = ["adb", "exec-out"] + argv
    elif enable_root and root_check():
        command = ["su", "-c", shlex.join(argv)]
    else:
        raise RuntimeError("Neither paired ADB nor existing root is available")
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        output, error = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        raise RuntimeError("Screenshot timed out") from None
    if proc.returncode != 0:
        raise RuntimeError("Screenshot failed: " + error[:300].decode("utf-8", "replace"))
    if len(output) > 12 * 1024 * 1024:
        raise RuntimeError("Screenshot exceeds 12 MiB")
    return output


def android_action(job, enable_adb=False, enable_root=False):
    action = job["action"]
    def cmd(parts, timeout=20):
        return ui_backend(parts, enable_adb=enable_adb, enable_root=enable_root, timeout=timeout)
    if action == "adb_status":
        if enable_adb:
            return adb(["devices", "-l"])
        return {"returncode": 0, "output": "ADB disabled; root available: {}".format(root_check())}
    if action == "android_status":
        result = {"adb_installed": bool(shutil.which("adb")), "adb_enabled": enable_adb,
                  "root_enabled": enable_root, "su_present": bool(shutil.which("su"))}
        if enable_adb and shutil.which("adb"):
            result["adb_devices"] = adb(["devices", "-l"], timeout=8)
        if enable_root:
            result["root_available"] = root_check()
        return result
    if action == "android_tap":
        x, y = int(job["x"]), int(job["y"])
        if not 0 <= x <= 10000 or not 0 <= y <= 10000:
            raise ValueError("Invalid coordinates")
        return cmd(["input", "tap", str(x), str(y)])
    if action == "android_swipe":
        coords = [int(job[k]) for k in ("x1", "y1", "x2", "y2")]
        millis = int(job.get("duration_ms", 400))
        if any(not 0 <= n <= 10000 for n in coords) or not 100 <= millis <= 5000:
            raise ValueError("Invalid swipe parameters")
        return cmd(["input", "swipe"] + list(map(str, coords)) + [str(millis)])
    if action == "android_keyevent":
        code = int(job["keycode"])
        if code not in (3, 4, 19, 20, 21, 22, 23, 24, 25, 61, 62, 66, 67, 82, 84, 187, 279):
            raise ValueError("Keycode not in allowlist")
        return cmd(["input", "keyevent", str(code)])
    if action == "android_input_text":
        text = job.get("text")
        if not isinstance(text, str) or not (1 <= len(text) <= 150):
            raise ValueError("text must contain 1..150 characters")
        # Android 'input text' is not a general Unicode input method.
        if not re.fullmatch(r"[A-Za-z0-9 .,:;@_+\-=/!?()]+", text):
            raise ValueError("ADB input text supports a limited ASCII set; Unicode IME not installed")
        return cmd(["input", "text", text.replace(" ", "%s")])
    if action == "android_launch_app":
        package = job.get("package")
        if not isinstance(package, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)+", package) or len(package)>160:
            raise ValueError("Invalid Android package name")
        return cmd(["monkey", "-p", package, "-c", "android.intent.category.LAUNCHER", "1"], timeout=25)
    if action == "android_open_url":
        url = job.get("url")
        if not isinstance(url, str) or len(url)>1200 or not re.fullmatch(r"https?://[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%\-]+", url):
            raise ValueError("URL must be HTTP(S) without spaces or control characters")
        return cmd(["am", "start", "-a", "android.intent.action.VIEW", "-d", url])
    if action == "android_ui_dump":
        first = cmd(["uiautomator", "dump", "/sdcard/window.xml"], timeout=35)
        if first["returncode"] != 0:
            return first
        if enable_adb:
            return adb(["exec-out", "cat", "/sdcard/window.xml"], timeout=20)
        return cmd(["cat", "/sdcard/window.xml"], timeout=20)
    if action == "android_screenshot":
        raw = ui_binary(["screencap", "-p"], enable_adb=enable_adb, enable_root=enable_root)
        try:
            from PIL import Image
        except ImportError:
            raise RuntimeError("Install Pillow to resize screenshots: pkg install python-pillow") from None
        if not raw.startswith(b"\x89PNG"):
            raise RuntimeError("ADB/Root returned non-PNG screenshot")
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        original_size = list(im.size)
        im.thumbnail((900, 1600))
        buffer = io.BytesIO()
        im.save(buffer, format="JPEG", quality=62, optimize=True)
        jpg = buffer.getvalue()
        if len(jpg) > 500000:
            raise RuntimeError("Compressed screenshot too large")
        return {"mime_type": "image/jpeg", "encoding": "base64", "width": im.width,
                "height": im.height, "original_size": original_size,
                "image_base64": base64.b64encode(jpg).decode("ascii")}
    if action == "android_batch":
        steps = job.get("steps")
        if not isinstance(steps, list) or not 1 <= len(steps) <= 12:
            raise ValueError("Batch requires 1..12 steps")
        results = []
        for idx, step in enumerate(steps):
            if not isinstance(step, dict) or step.get("action") not in {
                "android_tap", "android_swipe", "android_keyevent", "android_input_text",
                "android_launch_app", "android_open_url"}:
                raise ValueError("Unsupported batch step {}".format(idx))
            outcome = android_action(step, enable_adb=enable_adb, enable_root=enable_root)
            results.append({"step": idx, "action": step["action"], "detail": outcome})
            if outcome.get("returncode", 0) != 0 or outcome.get("timed_out", False):
                break
            # Small configurable pause to allow UI transitions; no background automation.
            pause = step.get("after_ms", 0)
            if not isinstance(pause, int) or not 0 <= pause <= 3000:
                raise ValueError("after_ms must be 0..3000")
            if pause:
                time.sleep(pause / 1000)
        completed = len(results) == len(steps) and all(
            s["detail"].get("returncode", 0) == 0 and not s["detail"].get("timed_out", False)
            for s in results)
        return {"steps": results, "completed": completed, "returncode": 0 if completed else 1}
    raise ValueError("Unsupported Android action: " + action)


def execute(job, enable_dev=False, enable_adb=False, enable_root=False):
    if not isinstance(job, dict):
        raise ValueError("Request must be a JSON object")
    action = job.get("action")
    if not isinstance(action, str):
        raise ValueError("Missing action")
    if action not in SAFE_ACTIONS | ADB_ACTIONS | {"dev_exec"}:
        raise ValueError("Action not supported: " + action[:60])
    if action in ADB_ACTIONS and not (enable_adb or enable_root):
        raise PermissionError("Android control disabled; use --enable-adb or --enable-root")
    if action == "dev_exec" and not enable_dev:
        raise PermissionError("Developer command execution disabled; start with --enable-dev")
    WORKSPACE.mkdir(mode=0o700, parents=True, exist_ok=True)
    if action == "ping":
        return {"message": "pong", "version": VERSION, "device": "Android/Termux", "utc": utc_now()}
    if action == "device_info":
        info = {"system": platform.platform(), "architecture": platform.machine(),
                "python": platform.python_version(), "workspace": str(WORKSPACE),
                "adb_installed": bool(shutil.which("adb")), "dev_enabled": enable_dev,
                "adb_enabled": enable_adb, "root_enabled": enable_root, "agent_version": VERSION}
        getprop = shutil.which("getprop")
        if getprop:
            for key in ("ro.product.manufacturer", "ro.product.model", "ro.build.version.release", "ro.build.version.sdk"):
                result = capture([getprop, key], timeout=5)
                info[key] = result["output"].strip()[:100]
        return info
    if action == "list_files":
        folder = path_in_workspace(job.get("path", "."))
        if not folder.is_dir():
            raise ValueError("Not a directory")
        entries = sorted(folder.iterdir(), key=lambda p: p.name)[:100]
        return {"path": str(folder.relative_to(WORKSPACE.resolve())),
                "entries": [{"name": p.name, "kind": "dir" if p.is_dir() else "file"} for p in entries],
                "capped": len(list(folder.iterdir())) > 100}
    if action == "read_file":
        path = path_in_workspace(job.get("path"))
        if not path.is_file():
            raise ValueError("Not a file")
        with path.open("rb") as stream:
            raw = stream.read(MAX_RESULT_CHARS + 1)
        if b"\x00" in raw:
            raise ValueError("Binary file not supported")
        return {"path": str(path.relative_to(WORKSPACE.resolve())),
                "text": raw[:MAX_RESULT_CHARS].decode("utf-8", "replace"),
                "truncated": len(raw) > MAX_RESULT_CHARS}
    if action == "write_file":
        path = path_in_workspace(job.get("path"))
        content = job.get("content")
        if not isinstance(content, str) or len(content) > MAX_WRITE_CHARS:
            raise ValueError("content missing or too long")
        if path == WORKSPACE.resolve() or path.exists() and not path.is_file():
            raise ValueError("Target is not a regular file")
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".chatwrite-", delete=False) as tmp:
            tmp.write(content)
            temp_path = pathlib.Path(tmp.name)
        os.chmod(temp_path, 0o600)
        os.replace(temp_path, path)
        return {"path": str(path.relative_to(WORKSPACE.resolve())), "bytes": len(content.encode("utf-8"))}
    if action == "dev_exec":
        argv = job.get("argv")
        if not isinstance(argv, list) or not (1 <= len(argv) <= 24) or any(not isinstance(x, str) or len(x) > 1000 for x in argv):
            raise ValueError("argv must be an array of 1..24 short strings")
        if argv[0] not in DEV_COMMANDS:
            raise PermissionError("Developer executable not permitted")
        cwd = path_in_workspace(job.get("cwd", "."))
        if not cwd.is_dir():
            raise ValueError("cwd is not a directory")
        return capture(argv, timeout=min(120, max(1, int(job.get("timeout", 30)))), cwd=cwd)
    if action in ADB_ACTIONS:
        return android_action(job, enable_adb=enable_adb, enable_root=enable_root)
    raise AssertionError("unreachable")


def parse_deadline(value):
    if not isinstance(value, str):
        raise ValueError("Every request needs expires_at, e.g. 2026-11-29T00:00:00Z")
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Expiry must contain UTC offset")
    return parsed.astimezone(dt.timezone.utc)


def process(client, enable_dev=False, enable_adb=False, enable_root=False):
    count = 0
    for job_id in client.pending_ids():
        response = {"id": job_id, "started_utc": utc_now()}
        try:
            request = client.get_request(job_id)
            if not isinstance(request, dict) or request.get("id") != job_id:
                raise ValueError("Request ID mismatch")
            if dt.datetime.now(dt.timezone.utc) > parse_deadline(request.get("expires_at")):
                raise ValueError("Request expired")
            response["action"] = request.get("action")
            response["detail"] = execute(request, enable_dev=enable_dev, enable_adb=enable_adb, enable_root=enable_root)
            detail = response["detail"]
            response["ok"] = not (isinstance(detail, dict) and
                                   (detail.get("returncode", 0) != 0 or detail.get("timed_out", False)))
        except Exception as exc:
            response["ok"] = False
            response["error"] = "{}: {}".format(type(exc).__name__, exc)
        response["finished_utc"] = utc_now()
        client.put_result(job_id, response)
        count += 1
        print("{} {}".format(job_id, "ok" if response["ok"] else "failed"), flush=True)
    return count


def read_token():
    if not TOKEN_FILE.is_file():
        raise RuntimeError("No token file: run setup first")
    if TOKEN_FILE.stat().st_mode & 0o077:
        raise RuntimeError("Token permissions unsafe; run chmod 600 " + str(TOKEN_FILE))
    token = TOKEN_FILE.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError("Empty token")
    return token


def setup():
    CONFIG_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(CONFIG_DIR, 0o700)
    print("Use a fine-grained PAT: ONLY zzq778/jetson, Contents read/write.")
    print("Never send that token in ChatGPT or place it in a GitHub repo.")
    token = getpass.getpass("GitHub token (hidden): ").strip()
    if not token:
        raise ValueError("Token is empty")
    Github(token).api("GET", "repos/{}".format(REPO))
    fd = os.open(TOKEN_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(token + "\n")
    os.chmod(TOKEN_FILE, 0o600)
    print("Token validated. Stored privately on this phone.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("setup", "once", "loop"))
    parser.add_argument("--enable-dev", action="store_true", help="Allow developer tools to run code in workspace (not sandboxed)")
    parser.add_argument("--enable-adb", action="store_true", help="Allow ADB UI interactions after explicit pairing")
    parser.add_argument("--enable-root", action="store_true", help="Use already installed su for limited UI actions (never installs root)")
    args = parser.parse_args(argv)
    if args.mode == "setup":
        setup()
        return 0
    client = Github(read_token())
    print("Connected to {} ({}); workspace {}".format(REPO, args.mode, WORKSPACE), flush=True)
    while True:
        try:
            num = process(client, enable_dev=args.enable_dev, enable_adb=args.enable_adb, enable_root=args.enable_root)
            if args.mode == "once":
                print("Processed {} jobs".format(num))
                return 0
        except Exception as exc:
            print("Queue failure: {}: {}".format(type(exc).__name__, exc), file=sys.stderr, flush=True)
            if args.mode == "once":
                return 1
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())

