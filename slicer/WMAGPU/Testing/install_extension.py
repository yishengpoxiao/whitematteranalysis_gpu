"""Test Install from file inside Slicer with isolated extension settings."""
import os
import sys
import slicer

try:
    manager = slicer.app.extensionsManagerModel()
    manager.interactive = False
    import qt
    settings_path = os.environ['WMA_EXTENSION_SETTINGS']
    manager.extensionsSettingsFilePath = settings_path
    settings = qt.QSettings(settings_path, qt.QSettings.IniFormat)
    settings.setValue('Extensions/InstallPath', os.environ['WMA_EXTENSION_INSTALL_ROOT'])
    settings.sync()
    archive = os.environ['WMA_EXTENSION_ARCHIVE']
    assert manager.installExtension(archive, False, True), 'Extension Manager rejected the archive'
    assert manager.isExtensionInstalled('WMAGPU')
    print('EXTENSION_INSTALL_OK:', manager.extensionInstallPath('WMAGPU'), flush=True)
except Exception:
    import traceback
    traceback.print_exc()
    sys.exit(1)
sys.exit(0)
