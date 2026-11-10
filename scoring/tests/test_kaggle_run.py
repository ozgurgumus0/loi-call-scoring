"""Offline regression checks for submission and queued-run handling."""
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src import kaggle_run as runner


class KaggleRunTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        sample = Path(directory.name) / "synthetic.jsonl"
        sample.write_text("{}\n", encoding="utf-8")
        jsonl_patch = patch.object(runner, "JSONL", sample)
        jsonl_patch.start()
        self.addCleanup(jsonl_patch.stop)

    def test_status_formats(self):
        for output, expected in [
            ('has status "KernelWorkerStatus.QUEUED"', 'QUEUED'),
            ('has status "complete"', 'COMPLETE'),
            ('{"status": "CANCEL_ACKNOWLEDGED"}', 'CANCEL_ACKNOWLEDGED'),
            ('ConnectionError: status request failed', None),
        ]:
            self.assertEqual(runner._parse_status(output), expected)

    def test_wait_timeout_is_not_completion(self):
        with patch.object(runner, '_sh', return_value=(0, 'has status "queued"')), \
             patch.object(runner.time, 'monotonic', side_effect=[0, 0, 1, 2]), \
             patch.object(runner.time, 'sleep'), redirect_stdout(io.StringIO()):
            self.assertFalse(runner._wait_for_completion('owner/kernel', 1, 1))

    def test_transient_network_failure_then_completion(self):
        with patch.object(runner, '_sh', side_effect=[
                (1, 'ConnectionError'), (0, 'has status "complete"')]), \
             patch.object(runner.time, 'sleep'), redirect_stdout(io.StringIO()):
            self.assertTrue(runner._wait_for_completion('owner/kernel', 1, 60))

    def test_cancel_does_not_download(self):
        with patch.object(runner, '_sh', return_value=(0, 'has status "cancel_acknowledged"')), \
             redirect_stdout(io.StringIO()):
            self.assertFalse(runner._wait_for_completion('owner/kernel', 1, 60))

    def test_active_run_cannot_be_submitted_over(self):
        with patch.object(runner, '_preflight', return_value=None), \
             patch.object(runner, '_slugs', return_value=('owner/data', 'owner/kernel')), \
             patch.object(runner, '_sh', return_value=(0, 'has status "queued"')) as command, \
             patch.object(runner, '_write_metadata') as stage, \
             redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(['--go']), 2)
            self.assertEqual(command.call_count, 1)
            stage.assert_not_called()

    def test_resume_timeout_never_uploads_or_downloads(self):
        with patch.object(runner, '_preflight', return_value=None), \
             patch.object(runner, '_slugs', return_value=('owner/data', 'owner/kernel')), \
             patch.object(runner, '_wait_for_completion', return_value=False), \
             patch.object(runner, '_sh') as command, \
             patch.object(runner, '_write_metadata') as stage, \
             redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(['--go', '--resume']), 1)
            command.assert_not_called()
            stage.assert_not_called()


if __name__ == '__main__':
    unittest.main()
