"""No network, GitHub PAT, Jetson hardware or root access required."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

MODULE = Path(__file__).resolve().parents[1] / 'tools' / 'remote_bridge.py'
spec = importlib.util.spec_from_file_location('remote_bridge', str(MODULE))
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


class FakeGithub(object):
    def __init__(self, requests, responses=None):
        self.requests = requests
        self.responses = dict(responses or {})
        self.writes = {}

    def list_jobs(self):
        return sorted(self.requests)

    def get_json(self, directory, job_id):
        return self.responses.get(job_id) if directory == 'results' else self.requests.get(job_id)

    def write_result(self, job_id, response):
        self.writes[job_id] = response


class BridgeTests(unittest.TestCase):
    def test_allowlist_rejects_arbitrary_command(self):
        malicious = FakeGithub({'job1': {'id': 'job1', 'action': 'rm -rf /'}})
        with mock.patch.object(bridge, 'execute_action') as run:
            self.assertEqual(bridge.process_jobs(malicious), 1)
            run.assert_not_called()
        self.assertFalse(malicious.writes['job1']['ok'])

    def test_job_id_mismatch_rejected(self):
        client = FakeGithub({'expected': {'id': '../outside', 'action': 'health'}})
        with mock.patch.object(bridge, 'execute_action') as run:
            bridge.process_jobs(client)
            run.assert_not_called()
        self.assertFalse(client.writes['expected']['ok'])

    def test_completed_job_is_not_repeated(self):
        client = FakeGithub({'done': {'id': 'done', 'action': 'health'}}, {'done': {'ok': True}})
        self.assertEqual(bridge.process_jobs(client), 0)
        self.assertEqual(client.writes, {})

    def test_valid_health_job(self):
        client = FakeGithub({'ping_1': {'id': 'ping_1', 'action': 'health'}})
        with mock.patch.object(bridge, 'execute_action', return_value={'arch': 'aarch64'}) as run:
            self.assertEqual(bridge.process_jobs(client), 1)
            run.assert_called_once_with('health')
        self.assertTrue(client.writes['ping_1']['ok'])
        self.assertEqual(client.writes['ping_1']['detail']['arch'], 'aarch64')

    def test_output_bounded(self):
        result = bridge.capture_command([sys.executable, '-c', 'print("x"*20000)'], 4)
        self.assertTrue(result['output_truncated'])
        self.assertLessEqual(len(result['output'].encode()), bridge.MAX_OUTPUT_BYTES)

    def test_timeout_stops_process(self):
        result = bridge.capture_command([sys.executable, '-c', 'import time;time.sleep(10)'], 0.05)
        self.assertTrue(result['timed_out'])
        self.assertNotEqual(result['returncode'], 0)

    def test_rejects_group_readable_token(self):
        with tempfile.TemporaryDirectory() as directory:
            token = Path(directory) / 'token'
            token.write_text('dummy\n')
            os.chmod(str(token), 0o640)
            with mock.patch.object(bridge, 'TOKEN_FILE', token):
                with self.assertRaises(RuntimeError):
                    bridge.read_token()
            os.chmod(str(token), 0o600)
            with mock.patch.object(bridge, 'TOKEN_FILE', token):
                self.assertEqual(bridge.read_token(), 'dummy')

    def test_id_validation(self):
        self.assertIsNotNone(bridge.JOB_ID.fullmatch('health_2026-11-21'))
        self.assertIsNone(bridge.JOB_ID.fullmatch('../malicious'))
        self.assertIsNone(bridge.JOB_ID.fullmatch('a'*65))


if __name__ == '__main__':
    unittest.main()