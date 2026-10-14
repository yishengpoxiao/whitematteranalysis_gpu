"""Check cold imports in fresh interpreters so prior tests cannot hide eager dependencies."""

from pathlib import Path
import os
import subprocess
import sys
import unittest


class StartupTests(unittest.TestCase):
    def test_lazy_imports_and_cpu_without_torch(self):
        root = Path(__file__).resolve().parents[3]
        script = """
import sys
before = set(sys.modules)
import whitematteranalysis as wma
assert not {'numpy', 'vtk', 'scipy', 'matplotlib', 'torch'} & (set(sys.modules) - before)
assert 'cluster' in dir(wma) and 'whitematteranalysis.cluster' not in sys.modules
assert hasattr(wma, 'gpu')
assert 'distance' in dir(wma.gpu) and 'whitematteranalysis.gpu.distance' not in sys.modules
assert not {'numpy', 'vtk', 'scipy', 'matplotlib', 'torch'} & (set(sys.modules) - before)
cpu = wma.gpu.cpu_backend
assert cpu is wma.gpu.cpu_backend
assert not {'scipy', 'matplotlib', 'torch'} & (set(sys.modules) - before)
from whitematteranalysis import io, filter, fibers
assert io is wma.io and filter is wma.filter and fibers is wma.fibers
assert not {'scipy', 'matplotlib', 'torch'} & (set(sys.modules) - before)
import numpy as np
registration = wma.register_two_subjects.RegisterTractography()
registration.fixed = np.zeros((3, 4, 3))
registration.moving = np.ones((3, 4, 3))
registration.use_gpu, registration.maxfun = False, 2
registration.compute()
assert 'torch' not in sys.modules
for module in (wma, wma.gpu):
    try:
        module.unknown_wma_attribute
    except AttributeError:
        pass
    else:
        raise AssertionError('Missing attributes must raise AttributeError')
"""
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(root))
        subprocess.run([sys.executable, '-c', script], cwd=root, env=environment, check=True)


if __name__ == '__main__':
    unittest.main()
