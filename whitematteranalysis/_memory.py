"""Keep geometry scratch allocations from triggering Linux huge-page compaction."""
from contextlib import ContextDecorator
import os
import sys
from threading import Lock
import numpy

_core = getattr(numpy, '_core', None)
if _core is None:
    _core = numpy.core
_set_hugepage = getattr(_core.multiarray, '_set_madvise_hugepage', None)

_lock = Lock()
_users = 0
_previous = None


class SmallPages(ContextDecorator):
    """Scope NumPy's allocation hint; overlapping geometry calls share the scope.

    The hint is process-wide until the last preparation scope exits, including
    allocations in other threads; the caller's policy is restored afterward.
    NUMPY_MADVISE_HUGEPAGE=1 explicitly restores NumPy's usual allocation hint.
    """
    def _recreate_cm(self):
        return type(self)()

    def __enter__(self):
        global _users, _previous
        self.enabled = (sys.platform == 'linux' and _set_hugepage is not None and
                        os.environ.get('NUMPY_MADVISE_HUGEPAGE') != '1')
        if self.enabled:
            with _lock:
                if _users == 0:
                    _previous = _set_hugepage(False)
                _users += 1
        return self

    def __exit__(self, *exception):
        global _users
        if self.enabled:
            with _lock:
                _users -= 1
                if _users == 0:
                    _set_hugepage(_previous)
