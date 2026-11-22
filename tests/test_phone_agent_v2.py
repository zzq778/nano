import datetime as dt
import pathlib
import tempfile
import unittest
from unittest.mock import patch

import phone_agent as a


class FakeGithub:
    def __init__(self, request):
        self.request = request
        self.results = {}

    def pending_ids(self):
        return [self.request['id']]

    def get_request(self, job_id):
        return self.request

    def put_result(self, job_id, result):
        self.results[job_id] = result


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.workspace_patch = patch.object(a, 'WORKSPACE', pathlib.Path(self.temp.name) / 'workspace')
        self.workspace_patch.start()

    def tearDown(self):
        self.workspace_patch.stop()
        self.temp.cleanup()

    def test_ping(self):
        self.assertEqual(a.execute({'action': 'ping'})['message'], 'pong')

    def test_file_roundtrip(self):
        a.execute({'action': 'write_file', 'path': 'src/hello.txt', 'content': 'hello 你好'})
        self.assertEqual(a.execute({'action': 'read_file', 'path': 'src/hello.txt'})['text'], 'hello 你好')
        self.assertEqual(a.execute({'action': 'list_files', 'path': 'src'})['entries'][0]['name'], 'hello.txt')

    def test_block_traversal(self):
        with self.assertRaises(ValueError):
            a.execute({'action': 'read_file', 'path': '../secret'})
        with self.assertRaises(ValueError):
            a.execute({'action': 'write_file', 'path': '/tmp/x', 'content': 'x'})

    def test_block_symlink_escape(self):
        a.WORKSPACE.mkdir(parents=True)
        (a.WORKSPACE / 'out').symlink_to(pathlib.Path(self.temp.name))
        with self.assertRaises(ValueError):
            a.execute({'action': 'read_file', 'path': 'out/secret'})

    def test_dev_and_adb_off_by_default(self):
        with self.assertRaises(PermissionError):
            a.execute({'action': 'dev_exec', 'argv': ['git', 'status']})
        with self.assertRaises(PermissionError):
            a.execute({'action': 'android_tap', 'x': 2, 'y': 3})

    def test_dev_denies_unlisted(self):
        with self.assertRaises(PermissionError):
            a.execute({'action': 'dev_exec', 'argv': ['bash', '-c', 'whoami']}, enable_dev=True)

    def test_process_and_expiry(self):
        good = {'id': 'hello_01', 'action': 'ping', 'expires_at': '2099-01-01T00:00:00Z'}
        fake = FakeGithub(good)
        self.assertEqual(a.process(fake), 1)
        self.assertTrue(fake.results['hello_01']['ok'])
        bad = FakeGithub({'id': 'late', 'action': 'ping', 'expires_at': '2020-01-01T00:00:00Z'})
        a.process(bad)
        self.assertFalse(bad.results['late']['ok'])

    def test_id_mismatch_rejected(self):
        fake = FakeGithub({'id': 'hello', 'action': 'ping', 'expires_at': '2099-01-01T00:00:00Z'})
        fake.get_request = lambda job_id: {'id': 'different', 'action': 'ping', 'expires_at': '2099-01-01T00:00:00Z'}
        a.process(fake)
        self.assertFalse(fake.results['hello']['ok'])


if __name__ == '__main__':
    unittest.main()

class PhoneV2Tests(unittest.TestCase):
    def test_full_mode_requires_optin(self):
        with self.assertRaises(PermissionError):
            a.execute({'action': 'android_screenshot'})
        with self.assertRaises(PermissionError):
            a.execute({'action': 'android_launch_app', 'package': 'com.android.settings'})

    def test_adb_ui_tap(self):
        with patch.object(a, 'adb', return_value={'returncode': 0, 'output': ''}) as mock_adb:
            r = a.execute({'action': 'android_tap', 'x': 25, 'y': 44}, enable_adb=True)
        self.assertEqual(r['returncode'], 0)
        mock_adb.assert_called_with(['shell', 'input', 'tap', '25', '44'], timeout=20)

    def test_app_package_injection_rejected(self):
        with self.assertRaises(ValueError):
            a.execute({'action': 'android_launch_app', 'package': 'com.android.settings;rm -rf /'}, enable_adb=True)

    def test_input_text_limit(self):
        with self.assertRaises(ValueError):
            a.execute({'action': 'android_input_text', 'text': '你好'}, enable_adb=True)
        with patch.object(a, 'adb', return_value={'returncode': 0, 'output': ''}) as mock_adb:
            a.execute({'action': 'android_input_text', 'text': 'hello world'}, enable_adb=True)
        mock_adb.assert_called_with(['shell', 'input', 'text', 'hello%sworld'], timeout=20)

    def test_batch_aborts_on_step_failure(self):
        def fake_adb(argv, timeout=20):
            return {'returncode': 1 if 'tap' in argv else 0, 'output': ''}
        with patch.object(a, 'adb', side_effect=fake_adb):
            result = a.execute({'action': 'android_batch', 'steps': [
                {'action': 'android_tap', 'x': 1, 'y': 2},
                {'action': 'android_keyevent', 'keycode': 3}]}, enable_adb=True)
        self.assertFalse(result['completed'])
        self.assertEqual(result['returncode'], 1)
        self.assertEqual(len(result['steps']), 1)

    def test_bad_batch_rejected(self):
        with self.assertRaises(ValueError):
            a.execute({'action': 'android_batch', 'steps': [{'action': 'dev_exec', 'argv': ['python']} ]}, enable_adb=True)

    def test_root_backend_uses_su_not_adb(self):
        with patch.object(a, 'root_check', return_value=True), \
             patch.object(a, 'capture', return_value={'returncode': 0, 'output': ''}) as mock_capture:
            a.execute({'action': 'android_keyevent', 'keycode': 3}, enable_root=True)
        self.assertEqual(mock_capture.call_args.args[0][0:2], ['su', '-c'])

    def test_no_implicit_root(self):
        with self.assertRaises(PermissionError):
            a.execute({'action': 'android_keyevent', 'keycode': 3}, enable_root=False)

