#!/usr/bin/env python3
"""Workspace-scoped developer actions for the Jetson GitHub bridge.

All requests go through a fixed action allowlist, never through a caller-provided
shell command. This is a *separate checkout*, not a security sandbox: tests and
build scripts still execute with the privileges of the Jetson login account.
"""

import os
import pathlib
import re
import shutil

PUBLIC_SOURCE_URL = 'https://github.com/zzq778/nano.git'
WORK_ROOT = pathlib.Path.home() / '.local' / 'share' / 'jetson-chat-dev'
BRANCH_PATTERN = re.compile(r'^chat/[a-z0-9][a-z0-9-]{0,39}$')
SHA_PATTERN = re.compile(r'^[0-9a-f]{40}$')
FILE_EXTENSIONS = frozenset(('.py', '.c', '.cpp', '.cc', '.h', '.hpp',
                             '.md', '.json', '.yml', '.yaml', '.txt',
                             '.sh', '.cmake', '.toml', '.ini'))
MAX_FILE_BYTES = 128 * 1024
MAX_RESPONSE_CHARS = 12000
VALID_ACTIONS = ('sync_code', 'repo_status', 'list_code', 'read_code',
                 'search_code', 'run_check', 'device_info')
CHECKS = ('pytest', 'nms_smoke', 'compileall', 'native_build',
          'cmake_configure', 'cmake_build', 'diff_check',
          'r41_readiness', 'r41_preflight_quick')


def require_keys(job, allowed):
    unsupported = set(job).difference(set(allowed).union(('id', 'action', 'description')))
    if unsupported:
        raise ValueError('Unsupported request fields: ' + ', '.join(sorted(unsupported)))


def require_work_root():
    if WORK_ROOT.is_symlink():
        raise RuntimeError('Development workspace may not be a symlink')
    if not WORK_ROOT.is_dir() or not (WORK_ROOT / '.git').exists():
        raise RuntimeError('Development workspace missing; first run sync_code')
    return WORK_ROOT


def run_checked(capture, argv, seconds=30, cwd=None):
    result = capture(argv, seconds, cwd=cwd)
    if result['timed_out'] or result['returncode'] != 0:
        raise RuntimeError('Command failed: {} (exit {}, timeout {}): {}'.format(
            argv[0], result['returncode'], result['timed_out'],
            result['output'][:1000]))
    return result


def repo_command(capture, *args):
    root = require_work_root()
    return run_checked(capture, ['git', '-C', str(root)] + list(args))['output'].strip()


def tree_files(capture, root, ref):
    # Read tracked paths from a verified Git tree. A --no-checkout clone
    # may have no valid HEAD and no entries in its index.
    listing = run_checked(capture, ['git', '-C', str(root), 'ls-tree',
                                    '-r', '--name-only', '-z', ref])
    if listing['output_truncated']:
        raise RuntimeError('Git tree file list exceeds bounded output')
    return [name for name in listing['output'].split('\x00') if name]


def has_checkout_collision(root, paths):
    # Do not overwrite existing files or follow symlinks when deciding
    # whether a force checkout of a truly empty worktree is safe.
    for relative in paths:
        current = root
        parts = pathlib.PurePosixPath(relative).parts
        for index, part in enumerate(parts):
            current = current / part
            if (current.is_file() or current.is_symlink() or
                    (index == len(parts) - 1 and current.is_dir())):
                return True
    return False


def require_checkout_files(root):
    # compileall exits zero on nonexistent directories. Check real files.
    expected = ('README.md', 'tools/dev_agent.py',
                'tests/test_dev_agent.py', 'python/rescuevision/pipeline.py')
    if not all((root / path).is_file() for path in expected):
        raise RuntimeError('Development checkout is empty or incomplete; first run sync_code')


def sync_code(job, capture):
    require_keys(job, ('branch', 'expected_sha'))
    branch = job.get('branch')
    expected_sha = job.get('expected_sha')
    if not isinstance(branch, str) or not BRANCH_PATTERN.fullmatch(branch):
        raise ValueError('Branch must match chat/<lowercase-letters-numbers-hyphens>')
    if not isinstance(expected_sha, str) or not SHA_PATTERN.fullmatch(expected_sha):
        raise ValueError('expected_sha must be the reviewed 40-character commit SHA')
    if not WORK_ROOT.exists():
        WORK_ROOT.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        run_checked(capture, ['git', '-c', 'http.version=HTTP/1.1',
                              'clone', '--no-checkout',
                              PUBLIC_SOURCE_URL, str(WORK_ROOT)], 180)
    root = require_work_root()
    origin = repo_command(capture, 'remote', 'get-url', 'origin')
    if origin != PUBLIC_SOURCE_URL:
        raise RuntimeError('Unexpected remote origin; development workspace unchanged')
    # Fetch and verify the exact reviewed commit before writing files.
    ref = 'refs/remotes/origin/' + branch
    run_checked(capture, ['git', '-c', 'http.version=HTTP/1.1',
                          '-C', str(root), 'fetch', '--no-tags',
                          'origin', 'refs/heads/' + branch + ':' + ref], 180)
    actual_sha = repo_command(capture, 'rev-parse', ref)
    if actual_sha != expected_sha:
        raise RuntimeError('Branch moved after review; refusing to execute unverified code')

    target_paths = tree_files(capture, root, ref)
    head_result = capture(['git', '-C', str(root), 'rev-parse',
                           '--verify', 'HEAD'], 10)
    head_valid = head_result['returncode'] == 0
    previous_paths = tree_files(capture, root, 'HEAD') if head_valid else []
    candidates = set(target_paths).union(previous_paths)
    empty_checkout = bool(candidates) and not has_checkout_collision(root, candidates)
    if empty_checkout:
        # No tracked file is on disk. Only here is force checkout safe,
        # even when HEAD is unborn or the index holds staged deletions.
        run_checked(capture, ['git', '-C', str(root), 'checkout',
                              '--force', '--detach', ref], 90)
    else:
        if (not head_valid or
                repo_command(capture, 'status', '--porcelain',
                             '--untracked-files=no')):
            raise RuntimeError('Tracked changes exist in development workspace; refusing checkout')
        run_checked(capture, ['git', '-C', str(root), 'checkout',
                              '--detach', ref], 90)
    return {'branch': branch, 'head': repo_command(capture, 'rev-parse', 'HEAD'),
            'source': PUBLIC_SOURCE_URL, 'workspace': str(root)}


def allowed_relative_path(relative):
    if not isinstance(relative, str) or not relative or len(relative) > 240:
        raise ValueError('A short relative path is required')
    if '\\' in relative or '\x00' in relative or relative.startswith('/'):
        raise ValueError('Invalid path')
    parts = pathlib.PurePosixPath(relative).parts
    if not parts or any(part in ('.', '..') or part.startswith('.') for part in parts):
        raise ValueError('Hidden paths and traversal are not permitted')
    filename = parts[-1]
    if filename not in ('CMakeLists.txt', 'Makefile') and pathlib.PurePosixPath(filename).suffix.lower() not in FILE_EXTENSIONS:
        raise ValueError('Only source and documentation files can be read')
    return parts


def safe_file(relative):
    parts = allowed_relative_path(relative)
    root = require_work_root()
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise ValueError('Symlink traversal is not permitted')
    if not current.is_file():
        raise FileNotFoundError('Requested source file not found')
    if current.stat().st_size > MAX_FILE_BYTES:
        raise ValueError('Source file too large for remote read')
    return current


def tracked_sources(capture):
    result = run_checked(capture, ['git', '-C', str(require_work_root()),
                                   'ls-files', '-z'])
    if result['output_truncated']:
        raise RuntimeError('Source file list exceeds response limit')
    raw = result['output']
    # Git filenames are returned NUL-delimited, and always relative to WORK_ROOT.
    return [name for name in raw.split('\x00') if name]


def list_code(job, capture):
    require_keys(job, ('prefix',))
    prefix = job.get('prefix', '')
    if not isinstance(prefix, str) or len(prefix) > 100:
        raise ValueError('Invalid prefix')
    if prefix and ('..' in prefix or prefix.startswith('.') or '\\' in prefix or prefix.startswith('/')):
        raise ValueError('Invalid prefix')
    paths = []
    count = 0
    for name in tracked_sources(capture):
        if not name.startswith(prefix):
            continue
        try:
            allowed_relative_path(name)
        except ValueError:
            continue
        count += 1
        if len(paths) < 150:
            paths.append(name)
    return {'prefix': prefix, 'matched': count, 'files': paths,
            'truncated': count > len(paths)}


def read_code(job, capture):
    require_keys(job, ('path', 'start_line', 'max_lines'))
    path = job.get('path')
    start = job.get('start_line', 1)
    max_lines = job.get('max_lines', 100)
    if type(start) is not int or start < 1 or start > 100000:
        raise ValueError('start_line must be a positive integer')
    if type(max_lines) is not int or not 1 <= max_lines <= 180:
        raise ValueError('max_lines must be 1..180')
    file_path = safe_file(path)
    with file_path.open('r', encoding='utf-8', errors='replace') as handle:
        lines = handle.readlines()
    selected = lines[start-1:start-1+max_lines]
    content = ''.join('{}: {}'.format(i, line) for i, line in enumerate(selected, start))
    return {'path': path, 'start_line': start, 'total_lines': len(lines),
            'content': content[:MAX_RESPONSE_CHARS],
            'truncated': len(content) > MAX_RESPONSE_CHARS or start-1+max_lines < len(lines)}


def search_code(job, capture):
    require_keys(job, ('query',))
    needle = job.get('query')
    if not isinstance(needle, str) or not 2 <= len(needle) <= 120 or '\n' in needle:
        raise ValueError('Search query must be 2..120 characters on one line')
    matches = []
    total = 0
    for name in tracked_sources(capture):
        try:
            target = safe_file(name)
        except (ValueError, FileNotFoundError, OSError):
            continue
        with target.open('r', encoding='utf-8', errors='replace') as file:
            for number, line in enumerate(file, 1):
                if needle.casefold() not in line.casefold():
                    continue
                total += 1
                if len(matches) < 60:
                    matches.append('{}:{}: {}'.format(name, number, line.strip()[:170]))
    output = '\n'.join(matches)
    return {'query': needle, 'matches': matches, 'total': total,
            'truncated': total > len(matches) or len(output) > MAX_RESPONSE_CHARS}


def run_check(job, capture):
    require_keys(job, ('check', 'test_file'))
    name = job.get('check')
    if name not in CHECKS:
        raise ValueError('Unsupported check')
    root = require_work_root()
    require_checkout_files(root)
    if name in ('native_build', 'cmake_configure', 'cmake_build'):
        output_dir = root / 'build_chat_verify'
        if output_dir.is_symlink() or (output_dir.exists() and not output_dir.is_dir()):
            raise RuntimeError('Refusing unsafe native build directory')
    checks = {
        'pytest': (['python3', '-m', 'pytest', '-q', 'tests'], 420),
        'nms_smoke': (['bash', 'scripts/phone_nms_smoke.sh'], 180),
        'compileall': (['python3', '-m', 'compileall', '-q', 'python', 'tests', 'tools'], 120),
        'native_build': (['bash', 'scripts/build_native_on_jetson.sh'], 900),
        'cmake_configure': (['cmake', '-S', '.', '-B', 'build_chat_verify',
                             '-DWITH_TENSORRT=ON', '-DCMAKE_BUILD_TYPE=Release'], 180),
        'cmake_build': (['cmake', '--build', 'build_chat_verify', '--parallel', '2'], 900),
        'diff_check': (['git', 'diff', '--check'], 30),
        # Fixed, parameter-free diagnostics: only sources from the reviewed
        # isolated checkout are executed. Neither action opens a camera/UART.
        'r41_readiness': (['python3', 'tools/r41_device_readiness.py'], 30),
        'r41_preflight_quick': (['python3', 'tools/r41_preflight.py',
                                 '--tier', 'quick'], 600),
    }
    command, timeout = checks[name]
    if name == 'pytest' and job.get('test_file') is not None:
        test_file = job['test_file']
        if not isinstance(test_file, str) or not re.fullmatch(r'tests/test_[a-zA-Z0-9_]+\.py', test_file):
            raise ValueError('test_file must be tests/test_<name>.py')
        command = ['python3', '-m', 'pytest', '-q', test_file]
    elif job.get('test_file') is not None:
        raise ValueError('test_file can be used only with pytest')
    return capture(command, timeout, cwd=root)


def repo_status(job, capture):
    require_keys(job, ())
    root = require_work_root()
    return {'head': repo_command(capture, 'rev-parse', 'HEAD'),
            'status': repo_command(capture, 'status', '--short')[:4000],
            'workspace': str(root),
            'origin': repo_command(capture, 'remote', 'get-url', 'origin')}


def device_info(job, capture):
    require_keys(job, ())
    return {'memory': capture(['free', '-m'], 10),
            'disk': capture(['df', '-h', str(pathlib.Path.home())], 10),
            'uptime': capture(['uptime'], 10),
            'tools': {name: bool(shutil.which(name)) for name in
                      ('git', 'cmake', 'g++', 'python3', 'nvcc')}}


def execute(job, capture):
    action = job['action']
    if action == 'sync_code':
        return sync_code(job, capture)
    if action == 'repo_status':
        return repo_status(job, capture)
    if action == 'list_code':
        return list_code(job, capture)
    if action == 'read_code':
        return read_code(job, capture)
    if action == 'search_code':
        return search_code(job, capture)
    if action == 'run_check':
        return run_check(job, capture)
    if action == 'device_info':
        return device_info(job, capture)
    raise ValueError('Action is not allowed')