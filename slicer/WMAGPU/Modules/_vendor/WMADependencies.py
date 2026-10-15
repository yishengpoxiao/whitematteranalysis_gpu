"""Install missing Slicer workflow dependencies without replacing NumPy or VTK."""
import argparse
from datetime import date
import importlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile

REQUIRED = {'scipy': ('scipy.optimize', 'scipy.cluster.vq', 'scipy.cluster.hierarchy'),
            'matplotlib': ('matplotlib.pyplot',), 'threadpoolctl': ('threadpoolctl',)}
REQUIREMENTS = {'scipy': 'scipy>=1.4', 'matplotlib': 'matplotlib>=3.5',
                'threadpoolctl': 'threadpoolctl>=3.1'}


def missing_packages():
    missing = []
    for package, modules in REQUIRED.items():
        try:
            for module in modules:
                loaded = importlib.import_module(module)
                if package == 'threadpoolctl':
                    getattr(loaded, 'ThreadpoolController')
        except (ImportError, AttributeError):
            missing.append(REQUIREMENTS[package])
    return missing


def install_missing():
    import numpy
    import vtk
    needed = missing_packages()
    print('Slicer Python:', sys.version, flush=True)
    print('Platform:', sys.platform, 'NumPy:', numpy.__version__,
          'VTK:', vtk.vtkVersion.GetVTKVersion(), flush=True)
    if needed:
        with tempfile.TemporaryDirectory(prefix='wma-dependencies-') as directory:
            constraints = Path(directory) / ('constraints-%s.txt' % date.today().isoformat())
            constraints.write_text('numpy==%s\nvtk==%s\n' % (
                numpy.__version__, vtk.vtkVersion.GetVTKVersion()))
            # PythonSlicer is the launcher; sys.executable may be SlicerApp-real.
            executable = os.environ['WMA_PYTHON_SLICER']
            subprocess.check_call([executable, '-m', 'pip', 'install',
                '--disable-pip-version-check', '--no-input', '--progress-bar', 'off',
                '--only-binary=:all:', '--constraint', str(constraints)] + needed)
    else:
        print('Required scientific dependencies are already available.', flush=True)


def verify():
    missing = missing_packages()
    if missing:
        raise RuntimeError('Missing scientific dependencies: ' + ', '.join(missing))
    import joblib
    import xlrd
    import whitematteranalysis
    from whitematteranalysis.gpu import cpu_backend
    print('WMA_DEPENDENCIES_OK:', whitematteranalysis.__file__, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    os.environ.setdefault('MPLBACKEND', 'Agg')
    if args.install:
        install_missing()
    else:
        verify()
