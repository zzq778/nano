#!/usr/bin/env python3
"""Small, allowlisted GitHub relay for Jetson Nano (Ubuntu 18.04 / Python 3.6).

No SSH port, sudo, arbitrary shell commands, user-provided paths, or model uploads.
Requests and results are kept in a private GitHub repository. A fine-grained token
is stored only on the Jetson under ~/.config/jetson-remote-bridge/token (0600).
"""

import base64
import datetime
import fcntl
import getpass
import json
import os
import pathlib
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

REPO = 'zzq778/jetson'
SOURCE_ROOT = pathlib.Path(__file__).resolve().parent.parent
CONF_DIR = pathlib.Path.home() / '.config' / 'jetson-remote-bridge'
TOKEN_FILE = CONF_DIR / 'token'
LOCK_FILE = CONF_DIR / 'agent.lock'
POLL_SECONDS = 30
MAX_OUTPUT_BYTES = 16000
JOB_ID = re.compile(r'^[a-zA-Z0-9_-]{1,64}$')
ALLOWED_ACTIONS = ('health', 'nms_smoke', 'python_tests', 'native_build')


def utc_now():
    return datetime.datetime.utcnow().isoformat() + 'Z'


def read_token():
    if not TOKEN_FILE.is_file():
        raise RuntimeError('Token missing. First run: python3 tools/remote_bridge.py setup')
    if TOKEN_FILE.stat().st_mode & 0o077:
        raise RuntimeError('Token file is not private. Fix with: chmod 600 ' + str(TOKEN_FILE))
    token = TOKEN_FILE.read_text(encoding='utf-8').strip()
    if not token:
        raise RuntimeError('Token file is empty')
    return token


class Github(object):
    def __init__(self, token):
        self.token = token

    def api(self, method, path, payload=None):
        url = 'https://api.github.com/' + path.lstrip('/')
        headers = {
            'Authorization': 'Bearer ' + self.token,
            'Accept': 'application/vnd.github+json',
            'X-GitHub-Api-Version': '2022-11-28',
            'User-Agent': 'jetson-nano-limited-agent/1.0',
        }
        data = None
        if payload is not None:
            headers['Content-Type'] = 'application/json'
            data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=20) as response:
                body = response.read()
                return json.loads(body.decode('utf-8')) if body else None
        except urllib.error.HTTPError as error:
            # Avoid logging request headers/token, and cap returned error details.
            reason = error.read(500).decode('utf-8', 'replace')
            raise RuntimeError('GitHub HTTP {}: {}'.format(error.code, reason))

    def list_jobs(self):
        items = self.api('GET', 'repos/{}/contents/requests?ref=main'.format(REPO))
        if not isinstance(items, list):
            raise RuntimeError('requests directory is not accessible')
        jobs = []
        for item in items:
            name = item.get('name', '')
            if name.endswith('.json') and JOB_ID.fullmatch(name[:-5]):
                jobs.append(name[:-5])
        return sorted(jobs)[:10]

    def get_json(self, directory, job_id):
        path = 'repos/{}/contents/{}/{}.json?ref=main'.format(REPO, directory, job_id)
        try:
            file = self.api('GET', path)
        except RuntimeError as error:
            if 'GitHub HTTP 404:' in str(error):
                return None
            raise
        if file.get('encoding') != 'base64':
            raise RuntimeError('Expected base64 file from GitHub')
        raw = base64.b64decode(file['content']).decode('utf-8')
        return json.loads(raw)

    def write_result(self, job_id, result):
        path = 'repos/{}/contents/results/{}.json'.format(REPO, job_id)
        content = base64.b64encode(json.dumps(result, ensure_ascii=False,
                                              indent=2, sort_keys=True).encode('utf-8')).decode('ascii')
        return self.api('PUT', path, {
            'message': 'Jetson agent result: ' + job_id,
            'content': content,
            'branch': 'main',
        })


def capture_command(argv, timeout):
    # All argv lists are hardcoded below; requests cannot specify commands.
    with tempfile.TemporaryFile() as capture:
        proc = subprocess.Popen(argv, cwd=str(SOURCE_ROOT), stdout=capture,
                                stderr=subprocess.STDOUT, start_new_session=True)
        timed_out = False
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        capture.seek(0)
        data = capture.read(MAX_OUTPUT_BYTES + 1)
        truncated = len(data) > MAX_OUTPUT_BYTES
        return {
            'argv': argv,
            'returncode': proc.returncode,
            'timed_out': timed_out,
            'output': data[:MAX_OUTPUT_BYTES].decode('utf-8', 'replace'),
            'output_truncated': truncated,
        }


def execute_action(action):
    if action == 'health':
        return {
            'host': platform.node(),
            'arch': platform.machine(),
            'system': platform.platform(),
            'python': platform.python_version(),
            'source_root': str(SOURCE_ROOT),
            'tools': {name: bool(shutil.which(name)) for name in
                      ('python3', 'g++', 'cmake', 'nvcc', 'git')},
            'git': capture_command(['git', 'rev-parse', '--short', 'HEAD'], 10)
                   if shutil.which('git') else None,
        }
    commands = {
        'nms_smoke': (['bash', 'scripts/phone_nms_smoke.sh'], 180),
        'python_tests': (['python3', '-m', 'pytest', '-q', 'tests'], 300),
        'native_build': (['bash', 'scripts/build_native_on_jetson.sh'], 900),
    }
    argv, timeout = commands[action]
    return capture_command(argv, timeout)


def process_jobs(client):
    processed = 0
    for job_id in client.list_jobs():
        if client.get_json('results', job_id) is not None:
            continue
        job = client.get_json('requests', job_id)
        if not isinstance(job, dict):
            continue
        action = job.get('action')
        started = utc_now()
        start_time = time.monotonic()
        response = {'id': job_id, 'action': action, 'started_utc': started}
        try:
            if job.get('id') != job_id or action not in ALLOWED_ACTIONS:
                raise ValueError('Rejected request: invalid ID or action')
            response['detail'] = execute_action(action)
            detail = response['detail']
            response['ok'] = (detail.get('returncode', 0) == 0 and
                              not detail.get('timed_out', False))
        except Exception as error:
            response['ok'] = False
            response['error'] = '{}: {}'.format(type(error).__name__, error)
        response['ended_utc'] = utc_now()
        response['duration_seconds'] = round(time.monotonic() - start_time, 3)
        client.write_result(job_id, response)
        print('Processed job {} ({}) -> {}'.format(job_id, action,
                                                  'ok' if response['ok'] else 'error'), flush=True)
        processed += 1
    return processed


def setup():
    CONF_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(str(CONF_DIR), 0o700)
    print('Create a fine-grained GitHub PAT: ONLY repo zzq778/jetson; Contents Read and write.')
    print('The token stays on this Jetson. Do NOT paste it into ChatGPT.')
    token = getpass.getpass('GitHub token (hidden): ').strip()
    if not token:
        raise RuntimeError('No token supplied')
    # Validate before writing anything to the filesystem.
    Github(token).api('GET', 'repos/{}'.format(REPO))
    fd = os.open(str(TOKEN_FILE), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as file:
        file.write(token + '\n')
    os.chmod(str(TOKEN_FILE), 0o600)
    print('Token validated and stored with mode 0600 in {}'.format(TOKEN_FILE))


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ('setup', 'once', 'loop'):
        print('Usage: python3 tools/remote_bridge.py {setup|once|loop}')
        return 2
    command = sys.argv[1]
    if command == 'setup':
        setup()
        return 0
    CONF_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock = open(str(LOCK_FILE), 'a+')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, IOError):
        print('Agent already running. Refusing a second instance.')
        return 2
    client = Github(read_token())
    print('Connected to private request queue {}. Mode: {}'.format(REPO, command), flush=True)
    while True:
        try:
            count = process_jobs(client)
            if command == 'once':
                print('Completed {} new job(s)'.format(count))
                return 0
        except Exception as error:
            print('{} Poll failure: {}: {}'.format(utc_now(), type(error).__name__, error), flush=True)
            if command == 'once':
                return 1
        time.sleep(POLL_SECONDS)


if __name__ == '__main__':
    sys.exit(main())