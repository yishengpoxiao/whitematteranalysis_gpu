"""Geometry allocation policy restores state across failures and worker overlap."""
import os
from threading import Event, Thread
import unittest
from unittest.mock import Mock, patch
from joblib.externals import cloudpickle

from whitematteranalysis import _memory


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        os.environ.pop('NUMPY_MADVISE_HUGEPAGE', None)
        self.platform = patch.object(_memory.sys, 'platform', 'linux')
        self.platform.start()
        self.addCleanup(self.platform.stop)

    def test_nested_scopes_and_failure_restore_original_policy(self):
        for original in (False, True):
            with self.subTest(original=original):
                setter = Mock(return_value=original)
                with patch.object(_memory, '_set_hugepage', setter):
                    with self.assertRaisesRegex(ValueError, 'failed'):
                        with _memory.SmallPages():
                            with _memory.SmallPages():
                                self.assertEqual(setter.call_count, 1)
                                raise ValueError('failed')
                self.assertEqual([c.args for c in setter.call_args_list],
                                 [(False,), (original,)])

    def test_overlapping_workers_restore_only_after_last_exit(self):
        entered, release = Event(), Event()
        setter = Mock(return_value=True)

        def worker():
            with _memory.SmallPages():
                entered.set()
                release.wait(timeout=5)

        with patch.object(_memory, '_set_hugepage', setter):
            with _memory.SmallPages():
                thread = Thread(target=worker)
                thread.start()
                self.assertTrue(entered.wait(timeout=5))
            self.assertEqual([c.args for c in setter.call_args_list], [(False,)])
            release.set()
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
        self.assertEqual([c.args for c in setter.call_args_list], [(False,), (True,)])

    def test_explicit_hint_and_unsupported_platform_are_unchanged(self):
        with patch.object(_memory, '_set_hugepage') as setter:
            with patch.dict(os.environ, {'NUMPY_MADVISE_HUGEPAGE': '1'}):
                with _memory.SmallPages():
                    pass
            with patch.object(_memory.sys, 'platform', 'win32'):
                with _memory.SmallPages():
                    pass
            setter.assert_not_called()
        with patch.object(_memory, '_set_hugepage', None):
            with _memory.SmallPages():
                pass

    def test_decorated_cli_tasks_can_be_pickled(self):
        @_memory.SmallPages()
        def task(value):
            return value + 1

        restored = cloudpickle.loads(cloudpickle.dumps(task))
        self.assertEqual(restored(4), 5)


if __name__ == '__main__':
    unittest.main()
