"""Installer selection and package-layout tests without changing Slicer Python."""
import base64
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Modules/_vendor'))
import WMADependencies as dependencies
import WMATorchInstaller as torch_setup


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


packager = module_at('wma_packager', ROOT / 'tools/package_extension.py')
installer = module_at('wma_installer', ROOT / 'tools/installer_template.py')


class InstallerTests(unittest.TestCase):
    def test_blas_dependency_feature_probe(self):
        available = types.SimpleNamespace(ThreadpoolController=object, __version__='3.6.0')
        with patch.object(dependencies.importlib, 'import_module', return_value=available):
            self.assertEqual(dependencies.missing_packages(), [])
        with patch.object(dependencies.importlib, 'import_module', return_value=object()):
            self.assertEqual(dependencies.missing_packages(), ['threadpoolctl>=3.1,<3.7'])
        available.__version__ = '3.7.0'
        with patch.object(dependencies.importlib, 'import_module', return_value=available):
            self.assertEqual(dependencies.missing_packages(), ['threadpoolctl>=3.1,<3.7'])

    def test_version_detection(self):
        for version in ((5, 2), (5, 4), (5, 8), (5, 10), (5, 12), (6, 0)):
            self.assertEqual(installer.runtime_version(*version, (3, 12)), '%d.%d' % version)
        with self.assertRaisesRegex(RuntimeError, 'Slicer 5.2'):
            installer.runtime_version(4, 11, (3, 9))
        with self.assertRaisesRegex(RuntimeError, 'Python 3.8'):
            installer.runtime_version(5, 2, (3, 7))

    def test_universal_archive_adapts_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = packager.package_universal(Path(directory) / 'universal.zip')
            with zipfile.ZipFile(bundle) as archive:
                script = archive.read('WMAGPU-Installer/InstallWMAGPU.py').decode('utf-8')
            namespace = {'__name__': 'test_generated_installer'}
            exec(compile(script, 'InstallWMAGPU.py', 'exec'), namespace)
            payload = base64.b64decode(namespace['PAYLOAD'])
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                self.assertIn('Modules/_vendor/WMADependencies.py', archive.namelist())
                self.assertFalse(any('WMALabelFromAtlas' in n or '__pycache__' in n for n in archive.namelist()))
            for version in ('5.2', '5.10', '6.0'):
                output = Path(directory) / ('version-%s.zip' % version)
                namespace['make_archive'](payload, output, version)
                with zipfile.ZipFile(output) as archive:
                    prefix = 'WMAGPU/lib/Slicer-%s/qt-scripted-modules/' % version
                    self.assertIn(prefix + 'WMAPipeline.py', archive.namelist())
                    scripts = [n for n in archive.namelist()
                               if n.startswith(prefix) and n.endswith('.py') and '/_vendor/' not in n]
                    self.assertEqual(len(scripts), 5)
                    self.assertIn('WMAGPU/share/Slicer-%s/WMAGPU.s4ext' % version, archive.namelist())

    def test_dependency_install_preserves_numpy_and_vtk(self):
        import numpy
        import vtk
        calls = []

        def run(arguments):
            constraint = Path(arguments[arguments.index('--constraint') + 1])
            self.assertIn('numpy==' + numpy.__version__, constraint.read_text())
            self.assertIn('vtk==' + vtk.vtkVersion.GetVTKVersion(), constraint.read_text())
            self.assertRegex(constraint.name, r'constraints-\d{4}-\d{2}-\d{2}\.txt')
            calls.append(arguments)

        with patch.object(dependencies, 'missing_packages', return_value=['scipy>=1.4']), patch.dict(
                dependencies.os.environ, WMA_PYTHON_SLICER='PythonSlicer.exe'), patch.object(
                dependencies.subprocess, 'check_call', side_effect=run):
            dependencies.install_missing()
        self.assertEqual(calls[0][0], 'PythonSlicer.exe')
        self.assertIn('--only-binary=:all:', calls[0])
        self.assertEqual(calls[0][-1], 'scipy>=1.4')
        with patch.object(dependencies, 'missing_packages', return_value=[]), patch.object(
                dependencies.subprocess, 'check_call') as pip:
            dependencies.install_missing()
            pip.assert_not_called()

    def test_cuda_profiles(self):
        for operating_system, machine in (('win32', 'AMD64'), ('linux', 'x86_64'), ('linux2', 'x86_64')):
            with patch.object(torch_setup.sys, 'platform', operating_system), patch.object(
                    torch_setup.platform, 'machine', return_value=machine):
                with patch.object(torch_setup.sys, 'version_info', (3, 9)):
                    self.assertIn('torch==2.4.1+cu121', torch_setup.installation_arguments())
                with patch.object(torch_setup.sys, 'version_info', (3, 12)):
                    self.assertIn('torch==2.10.0', torch_setup.installation_arguments())
                    self.assertIn('https://download.pytorch.org/whl/cu126', torch_setup.installation_arguments())
        for operating_system, machine in (('darwin', 'arm64'), ('darwin', 'x86_64'), ('linux', 'aarch64'), ('win32', 'ARM64')):
            with patch.object(torch_setup.sys, 'platform', operating_system), patch.object(
                    torch_setup.platform, 'machine', return_value=machine):
                with self.assertRaisesRegex(RuntimeError, 'CPU accelerated'):
                    torch_setup.installation_arguments()


if __name__ == '__main__':
    unittest.main()
