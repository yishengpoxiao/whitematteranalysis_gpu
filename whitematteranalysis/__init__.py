"""WhiteMatterAnalysis, with computation modules loaded when first used."""

from importlib import import_module as _import_module

__all__ = (
    'io', 'similarity', 'fibers', 'filter', 'laterality', 'render', 'cluster',
    'relative_distance', 'mrml', 'congeal_multisubject', 'register_two_subjects',
    'register_two_subjects_nonrigid', 'register_two_subjects_nonrigid_bsplines',
    'congeal_to_atlas', 'tract_measurement', 'utils', 'gpu',
)


def __getattr__(name):
    if name in __all__:
        module = _import_module('.' + name, __name__)
        globals()[name] = module
        return module
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


def __dir__():
    return sorted(set(globals()) | set(__all__))

