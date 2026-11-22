"""Regression tests for the separate Git-based development workspace."""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
import dev_agent

spec = importlib.util.spec_from_file_location('remote_bridge', str(TOOLS / 'remote_bridge.py'))
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


def run(args, cwd=None):
    try:
        return subprocess.check_output(args, cwd=str(cwd) if cwd else None,
                                       stderr=subprocess.STDOUT).decode('utf-8', 'replace').strip()
    except subprocess.CalledProcessError as error:
        output = (error.output or b'').decode('utf-8', 'replace')
        raise AssertionError('Git fixture command failed: {}\\n{}'.format(args, output)) from error


class SafeWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / 'dev'
        self.root.mkdir()
        run(['git', 'init', '-q'], self.root)
        self.data = self.root / 'src'
        self.data.mkdir()
        (self.data / 'example.py').write_text('alpha = 1\nbeta = 2\nalpha = 3\n')
        (self.root / 'README.md').write_text('hello\nworld\n')
        (self.root / '.env').write_text('PASSWORD=secret\n')
        run(['git', 'add', '.'], self.root)
        self.patch = mock.patch.object(dev_agent, 'WORK_ROOT', self.root)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def test_read_line_range(self):
        result = dev_agent.read_code({'path': 'src/example.py', 'start_line': 2,
                                      'max_lines': 1, 'action': 'read_code'}, bridge.capture_command)
        self.assertEqual(result['content'], '2: beta = 2\n')
        self.assertEqual(result['total_lines'], 3)

    def test_no_traversal_hidden_or_absolute_paths(self):
        for p in ('../.env', '/etc/passwd', '.env', 'src/../../.env',
                  'src\\example.py', 'src/.hidden.py'):
            with self.subTest(path=p), self.assertRaises(ValueError):
                dev_agent.allowed_relative_path(p)

    def test_symlink_escape_is_rejected(self):
        (self.data / 'link.py').symlink_to(self.root / 'README.md')
        with self.assertRaises(ValueError):
            dev_agent.safe_file('src/link.py')

    def test_only_source_files_are_remotely_readable(self):
        (self.root / 'secret.pem').write_text('not a real key')
        with self.assertRaises(ValueError):
            dev_agent.safe_file('secret.pem')

    def test_search_is_bounded_and_finds_code(self):
        result = dev_agent.search_code({'query': 'alpha', 'action': 'search_code'}, bridge.capture_command)
        self.assertEqual(result['total'], 2)
        self.assertEqual(len(result['matches']), 2)
        self.assertTrue(all('src/example.py:' in line for line in result['matches']))
        self.assertNotIn('PASSWORD', repr(result))

    def test_list_code_uses_tracked_files(self):
        (self.data / 'private.py').write_text('not tracked')
        result = dev_agent.list_code({'prefix': 'src/', 'action': 'list_code'}, bridge.capture_command)
        self.assertEqual(result['files'], ['src/example.py'])

    def test_invalid_check_cannot_execute(self):
        with mock.patch.object(bridge, 'capture_command') as capture:
            with self.assertRaises(ValueError):
                dev_agent.run_check({'action': 'run_check', 'check': 'sudo bash'}, capture)
            capture.assert_not_called()

    def test_pytest_arguments_are_fixed_and_scoped(self):
        with mock.patch.object(bridge, 'capture_command', return_value={'returncode': 0}) as capture:
            dev_agent.run_check({'action': 'run_check', 'check': 'pytest',
                                 'test_file': 'tests/test_example.py'}, capture)
            self.assertEqual(capture.call_args[0][0],
                             ['python3', '-m', 'pytest', '-q', 'tests/test_example.py'])
            self.assertEqual(capture.call_args[1]['cwd'], self.root)
        with self.assertRaises(ValueError):
            dev_agent.run_check({'action': 'run_check', 'check': 'pytest',
                                 'test_file': '../outside.py'}, bridge.capture_command)

    def test_rejects_unexpected_parameters(self):
        with self.assertRaises(ValueError):
            dev_agent.read_code({'action': 'read_code', 'path': 'README.md',
                                 'command': 'rm -rf /'}, bridge.capture_command)

    def test_no_missing_workspace_auto_creation_on_read(self):
        with mock.patch.object(dev_agent, 'WORK_ROOT', Path(self.tmp.name) / 'missing'):
            with self.assertRaises(RuntimeError):
                dev_agent.repo_status({'action': 'repo_status'}, bridge.capture_command)


@unittest.skipUnless(shutil.which('git'), 'git is required')
class GitSyncIntegration(unittest.TestCase):
    def test_sync_development_branch_and_refuse_dirty_checkout(self):
        with tempfile.TemporaryDirectory() as temp:
            src = Path(temp) / 'src'
            bare = Path(temp) / 'origin.git'
            workspace = Path(temp) / 'workspace'
            src.mkdir()
            run(['git', 'init', '-q'], src)
            run(['git', 'config', 'user.email', 'test@example.com'], src)
            run(['git', 'config', 'user.name', 'Test'], src)
            (src / 'README.md').write_text('first\n')
            run(['git', 'add', 'README.md'], src)
            run(['git', 'commit', '-qm', 'first'], src)
            run(['git', 'checkout', '-qb', 'chat/test-1'], src)
            run(['git', 'init', '--bare', '-q', str(bare)])
            run(['git', 'remote', 'add', 'origin', str(bare)], src)
            run(['git', 'push', '-q', '-u', 'origin', 'chat/test-1'], src)
            sha = run(['git', 'rev-parse', 'chat/test-1'], src)
            with mock.patch.object(dev_agent, 'WORK_ROOT', workspace), \
                    mock.patch.object(dev_agent, 'PUBLIC_SOURCE_URL', str(bare)):
                result = dev_agent.sync_code({'action': 'sync_code',
                                              'branch': 'chat/test-1',
                                              'expected_sha': sha}, bridge.capture_command)
                self.assertEqual(result['branch'], 'chat/test-1')
                self.assertEqual((workspace / 'README.md').read_text(), 'first\n')
                (workspace / 'README.md').write_text('local changes')
                with self.assertRaisesRegex(RuntimeError, 'Tracked changes'):
                    dev_agent.sync_code({'action': 'sync_code',
                                         'branch': 'chat/test-1',
                                         'expected_sha': sha}, bridge.capture_command)

    def test_sync_recovers_empty_no_checkout_clone_from_main(self):
        # Real repositories whose bare HEAD points to main report every tracked
        # file as "D" after clone --no-checkout; this is not user modification.
        with tempfile.TemporaryDirectory() as temp:
            src = Path(temp) / 'src'
            bare = Path(temp) / 'origin.git'
            workspace = Path(temp) / 'workspace'
            src.mkdir()
            run(['git', 'init', '-q'], src)
            run(['git', 'config', 'user.email', 'test@example.com'], src)
            run(['git', 'config', 'user.name', 'Test'], src)
            run(['git', 'checkout', '-qb', 'main'], src)
            (src / 'README.md').write_text('from-main\n')
            run(['git', 'add', 'README.md'], src)
            run(['git', 'commit', '-qm', 'initial'], src)
            run(['git', 'checkout', '-qb', 'chat/test-1'], src)
            # Clone the two local branches into a bare repository. Unlike
            # pushing both refspecs, this requires no local receive-pack
            # hooks/config and behaves consistently on Jetson's older Git.
            run(['git', 'clone', '--bare', '-q', str(src), str(bare)])
            run(['git', '--git-dir', str(bare), 'symbolic-ref', 'HEAD',
                 'refs/heads/main'])
            sha = run(['git', 'rev-parse', 'chat/test-1'], src)
            run(['git', 'clone', '--no-checkout', '-q', str(bare), str(workspace)])
            self.assertEqual({p.name for p in workspace.iterdir()}, {'.git'})
            self.assertIn('D  README.md', run(['git', 'status', '--porcelain'], workspace))
            with mock.patch.object(dev_agent, 'WORK_ROOT', workspace), \
                    mock.patch.object(dev_agent, 'PUBLIC_SOURCE_URL', str(bare)):
                result = dev_agent.sync_code(
                    {'action': 'sync_code', 'branch': 'chat/test-1',
                     'expected_sha': sha}, bridge.capture_command)
                self.assertEqual(result['head'], sha)
                self.assertEqual((workspace / 'README.md').read_text(), 'from-main\n')
                self.assertFalse(run(['git', 'status', '--porcelain'], workspace))

    def test_empty_checkout_refuses_checks(self):
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp) / 'workspace'
            workspace.mkdir()
            (workspace / '.git').mkdir()
            with mock.patch.object(dev_agent, 'WORK_ROOT', workspace), \
                    mock.patch.object(bridge, 'capture_command') as capture:
                with self.assertRaisesRegex(RuntimeError, 'checkout is empty'):
                    dev_agent.run_check({'action': 'run_check', 'check': 'compileall'},
                                        capture)
                capture.assert_not_called()

    def test_ref_and_branch_injection_rejected(self):
        with mock.patch.object(dev_agent, 'WORK_ROOT', Path('/nonexistent')):
            for branch in ('main', 'chat/../../etc', 'chat/x;rm', 'chat/x --help',
                           'chat/UPPER', 'chat/'):
                with self.subTest(branch=branch), self.assertRaises(ValueError):
                    dev_agent.sync_code({'action': 'sync_code', 'branch': branch,
                                         'expected_sha': '0'*40}, bridge.capture_command)


class QueueTests(unittest.TestCase):
    def test_completed_jobs_are_filtered_before_batch_cap(self):
        client = bridge.Github('dummy')
        req = [{'type': 'file', 'name': 'job{:02d}.json'.format(i)} for i in range(15)]
        done = req[:13]
        with mock.patch.object(client, 'api', side_effect=[req, done]):
            self.assertEqual(client.list_jobs(), ['job13', 'job14'])

    def test_new_developer_action_routes_to_safe_handler(self):
        class Client(object):
            def list_jobs(self):
                return ['task1']
            def get_json(self, directory, id):
                if directory == 'results':
                    return None
                return {'id': id, 'action': 'repo_status'}
            def write_result(self, id, response):
                self.response = response
        client = Client()
        with mock.patch.object(bridge.dev_agent, 'execute',
                               return_value={'head': 'abc'}) as execute:
            self.assertEqual(bridge.process_jobs(client), 1)
            self.assertEqual(client.response['detail']['head'], 'abc')
            self.assertTrue(client.response['ok'])
            execute.assert_called_once()


if __name__ == '__main__':
    unittest.main()