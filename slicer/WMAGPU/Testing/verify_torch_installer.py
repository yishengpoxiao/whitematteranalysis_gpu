"""Run in Slicer: test real CUDA probe and simulated pip results without changes."""
import sys
import time
from unittest.mock import patch
import qt
import slicer

try:
    widgets = []
    for name in ('WMAPipeline', 'WMARegisterToAtlas', 'WMAGroupRegister',
                 'WMAAtlasBuilder'):
        widget = slicer.util.getModule(name).widgetRepresentation().self()
        installer = widget.torchInstaller
        assert not installer.allow.checked and not installer.button.enabled
        assert not installer.profile.enabled and installer.profile.count >= 4
        installer.start()
        assert installer.phase is None
        widgets.append(widget)
    import WMATorchInstaller as setup
    instance = widgets[0].torchInstaller

    def wait():
        deadline = time.monotonic() + 45
        while instance.phase is not None and time.monotonic() < deadline:
            slicer.app.processEvents()
            time.sleep(.01)
        assert instance.phase is None, 'Installer timeout'
        assert setup.ACTIVE_INSTALLER is None
        assert instance.allow.enabled and instance.button.enabled

    instance.allow.checked = True
    assert instance.button.enabled
    instance.profile.currentIndex = instance.profile.findData('cu124')
    assert instance.profile.enabled
    # Existing GPU PyTorch must be left intact, with a real fresh-process probe.
    with patch.object(setup, 'installation_arguments', side_effect=AssertionError('Existing CUDA must be reused')):
        instance.start()
        assert not instance.allow.enabled, instance.log.toPlainText()
        wait()
    assert 'already available' in instance.log.toPlainText(), instance.log.toPlainText()
    # Simulate absent torch before pip, then a successful post-install probe.
    real_launch = instance.launch
    launches = []

    def launch(phase, arguments):
        launches.append((phase, arguments))
        if phase == 'check':
            arguments = ['-c', 'raise SystemExit(1)']
        elif phase == 'verify':
            arguments = ['-c', "print('CUDA available: True')"]
        real_launch(phase, arguments)

    with patch.object(instance, 'launch', side_effect=launch), patch.object(
            setup, 'installation_arguments', return_value=['-c', "print('SIMULATED_PIP_LOG')"]) as arguments:
        instance.start()
        other = widgets[1].torchInstaller
        other.allow.checked = True
        other.start()
        assert 'Another WMA' in other.log.toPlainText()
        wait()
        arguments.assert_called_once_with('cu124')
    assert instance.profile.enabled
    assert [phase for phase, _ in launches] == ['check', 'install', 'verify']
    assert 'SIMULATED_PIP_LOG' in instance.log.toPlainText()
    assert 'Restart Slicer before using GPU' in instance.log.toPlainText()
    with patch.object(instance, 'launch', side_effect=launch), patch.object(
            setup, 'installation_arguments', return_value=['-c', 'raise SystemExit(7)']):
        instance.start()
        wait()
    assert 'installation failed (exit 7)' in instance.log.toPlainText()
    def no_cuda(phase, arguments):
        if phase in ('check', 'verify'):
            arguments = ['-c', 'raise SystemExit(2)']
        real_launch(phase, arguments)

    with patch.object(instance, 'launch', side_effect=no_cuda), patch.object(
            setup, 'installation_arguments', return_value=['-c', "print('SIMULATED_PIP_LOG')"]):
        instance.start()
        wait()
    assert 'CUDA verification failed' in instance.log.toPlainText()
    with patch.object(setup, 'python_slicer', return_value='/nonexistent/PythonSlicer'):
        instance.start()
        wait()
    assert 'Unable to start PythonSlicer' in instance.log.toPlainText()
    with patch.object(setup.sys, 'platform', 'darwin'):
        try:
            setup.installation_arguments()
        except RuntimeError:
            pass
        else:
            raise AssertionError('Unsupported platform accepted')
    assert '--index-url' in setup.installation_arguments()
    print('TORCH_INSTALLER_VERIFIED', flush=True)
except Exception:
    import traceback
    traceback.print_exc()
    sys.exit(1)
sys.exit(0)
