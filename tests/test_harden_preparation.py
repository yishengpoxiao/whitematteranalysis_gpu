"""Exercise the input-preparation gate without launching Slicer."""
import importlib.util
import signal
import subprocess
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'wm_harden_transform_preparation', ROOT / 'bin/wm_harden_transform.py')
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class HardenPreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='wma preparation gate ')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.inputdir = self.root / 'input files'
        self.inputdir.mkdir()
        self.status = self.root / 'logs/input.status'
        self.status.parent.mkdir()
        self.expected = {'cluster_00001.vtp', 'cluster_00002.vtk'}
        self.started = [threading.Event(), threading.Event()]
        self.finished = []
        self.outputs = []

    def workers(self):
        def run(index):
            self.started[index].set()
            try:
                while not self.status.is_file():
                    time.sleep(.01)
                if self.status.read_text() != 'ready':
                    raise RuntimeError('Input preparation failed.')
                self.outputs.append(index)
            except RuntimeError:
                pass
            finally:
                self.finished.append(index)
        return [lambda index=index: run(index) for index in range(2)]

    def wait_for_workers(self):
        self.assertTrue(all(event.wait(5) for event in self.started))

    def write_expected_inputs(self):
        for name in self.expected:
            (self.inputdir / name).write_bytes(b'fixture')

    def test_workers_wait_until_producer_completes_and_catalogue_matches(self):
        def produce(command, check):
            self.assertEqual(command, ['produce fixtures'])
            self.assertTrue(check)
            self.wait_for_workers()
            self.assertFalse(self.status.exists())
            self.assertEqual(self.finished, [])
            self.write_expected_inputs()

        with patch.object(launcher.subprocess, 'run', side_effect=produce):
            elapsed = launcher._run_prepared(self.workers(), 2, ['produce fixtures'],
                self.status, self.inputdir, self.expected)
        self.assertGreaterEqual(elapsed, 0.)
        self.assertEqual(self.status.read_text(), 'ready')
        self.assertEqual(sorted(self.outputs), [0, 1])
        self.assertEqual(sorted(self.finished), [0, 1])

    def test_producer_failure_is_preserved_and_releases_all_workers(self):
        error = subprocess.CalledProcessError(17, ['prepare'], 'producer failed')

        def fail(command, check):
            self.wait_for_workers()
            raise error

        with patch.object(launcher.subprocess, 'run', side_effect=fail):
            with self.assertRaises(subprocess.CalledProcessError) as raised:
                launcher._run_prepared(self.workers(), 2, ['prepare'], self.status,
                                       self.inputdir, self.expected)
        self.assertIs(raised.exception, error)
        self.assertEqual(self.status.read_text(), 'failed')
        self.assertEqual(sorted(self.finished), [0, 1])
        self.assertEqual(self.outputs, [])
        self.assertEqual(list(self.inputdir.iterdir()), [])

    def test_successful_producer_with_wrong_catalogue_fails_gate(self):
        def produce(command, check):
            self.wait_for_workers()
            (self.inputdir / 'cluster_00001.vtp').write_bytes(b'fixture')

        with patch.object(launcher.subprocess, 'run', side_effect=produce):
            with self.assertRaisesRegex(RuntimeError, 'filenames do not match'):
                launcher._run_prepared(self.workers(), 2, ['prepare'], self.status,
                                       self.inputdir, self.expected)
        self.assertEqual(self.status.read_text(), 'failed')
        self.assertEqual(sorted(self.finished), [0, 1])
        self.assertEqual(self.outputs, [])

    def test_sigterm_releases_workers_and_restores_previous_handler(self):
        previous = signal.getsignal(signal.SIGTERM)

        def terminate(command, check):
            self.wait_for_workers()
            signal.raise_signal(signal.SIGTERM)

        with patch.object(launcher.subprocess, 'run', side_effect=terminate):
            with self.assertRaises(SystemExit) as raised:
                launcher._run_prepared(self.workers(), 2, ['prepare'], self.status,
                                       self.inputdir, self.expected)
        self.assertEqual(raised.exception.code, 128 + signal.SIGTERM)
        self.assertEqual(self.status.read_text(), 'failed')
        self.assertEqual(sorted(self.finished), [0, 1])
        self.assertIs(signal.getsignal(signal.SIGTERM), previous)

    def test_keyboard_interrupt_releases_workers_and_restores_previous_handler(self):
        previous = signal.getsignal(signal.SIGTERM)

        def interrupt(command, check):
            self.wait_for_workers()
            raise KeyboardInterrupt()

        with patch.object(launcher.subprocess, 'run', side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                launcher._run_prepared(self.workers(), 2, ['prepare'], self.status,
                                       self.inputdir, self.expected)
        self.assertEqual(self.status.read_text(), 'failed')
        self.assertEqual(sorted(self.finished), [0, 1])
        self.assertIs(signal.getsignal(signal.SIGTERM), previous)


if __name__ == '__main__':
    unittest.main()
