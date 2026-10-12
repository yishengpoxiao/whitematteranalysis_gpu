"""Run the universal installer UI in Slicer with isolated extension settings."""
import os
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch
import zipfile
import qt
import slicer

try:
    manager = slicer.app.extensionsManagerModel()
    manager.interactive = False
    manager.extensionsSettingsFilePath = os.environ['WMA_EXTENSION_SETTINGS']
    settings = qt.QSettings(os.environ['WMA_EXTENSION_SETTINGS'], qt.QSettings.IniFormat)
    settings.setValue('Extensions/InstallPath', os.environ['WMA_EXTENSION_INSTALL_ROOT'])
    settings.sync()
    with tempfile.TemporaryDirectory() as directory:
        with zipfile.ZipFile(os.environ['WMA_INSTALLER_ARCHIVE']) as archive:
            archive.extractall(directory)
        script = Path(directory) / 'WMAGPU-Installer/InstallWMAGPU.py'
        namespace = {'__name__': 'test_universal_installer'}
        exec(compile(script.read_text(), str(script), 'exec'), namespace)
        installer = namespace['show_installer']()
        assert installer.version == '%d.%d' % (slicer.app.majorVersion, slicer.app.minorVersion)
        assert not installer.gpu.checked

        def wait():
            deadline = time.monotonic() + 60
            while (installer.phase is not None or (
                    installer.gpu_installer is not None and installer.gpu_installer.phase is not None)):
                assert time.monotonic() < deadline, installer.log.toPlainText()
                slicer.app.processEvents()
                time.sleep(.01)
            assert installer.button.enabled

        def fail_launch(phase, arguments):
            installer.phase = phase
            installer.process.start(installer.executable, ['-c', 'raise SystemExit(7)'])

        with patch.object(installer, 'launch', side_effect=fail_launch):
            installer.start()
            wait()
        assert 'Setup failed' in installer.log.toPlainText()
        assert not manager.isExtensionInstalled('WMAGPU')
        installer.gpu.checked = True
        helper = sys.modules['_WMAGPUInstallTorch']
        # Optional CUDA path must never download packages in this test.
        with patch.object(helper, 'PROBE', "print('CUDA available: True')"), patch.object(
                helper, 'installation_arguments', return_value=['-c', 'raise SystemExit(91)']):
            installer.start()
            wait()
        log = installer.log.toPlainText()
        assert 'WMA_DEPENDENCIES_OK' in log, log
        assert 'WMAGPU installed successfully' in log, log
        assert 'CUDA PyTorch is already available' in log, log
        assert manager.isExtensionInstalled('WMAGPU')
        modules = Path(manager.extensionInstallPath('WMAGPU')) / (
            'lib/Slicer-%s/qt-scripted-modules' % installer.version)
        assert len(list(modules.glob('WMA*.py'))) == 5
        assert (modules / '_vendor/WMADependencies.py').is_file()
        assert not (modules / 'WMALabelFromAtlas.py').exists()
        assert any(str(modules) in path for path in manager.extensionModulePaths('WMAGPU'))
        installer.start()
        assert 'already installed' in installer.log.toPlainText()
        installer.dialog.close()
        print('UNIVERSAL_INSTALLER_VERIFIED:', modules, flush=True)
except Exception:
    import traceback
    traceback.print_exc()
    sys.exit(1)
sys.exit(0)
