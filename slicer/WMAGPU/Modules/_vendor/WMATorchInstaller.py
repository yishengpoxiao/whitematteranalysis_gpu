"""Opt-in CUDA PyTorch installation using Slicer's own Python interpreter."""
import os
import platform
import sys

PROBE = ("import torch; print('PyTorch:', torch.__version__); "
         "print('CUDA available:', torch.cuda.is_available()); "
         "raise SystemExit(0 if torch.cuda.is_available() else 2)")
ACTIVE_INSTALLER = None


def cuda_profiles():
    profiles = {}
    if (3, 8) <= sys.version_info[:2] <= (3, 12):
        profiles.update(dict.fromkeys(('cu118', 'cu121', 'cu124'), '2.4.1'))
    if (3, 10) <= sys.version_info[:2] <= (3, 14):
        profiles.update(dict.fromkeys(('cu126', 'cu128', 'cu130'), '2.10.0'))
    return profiles


def installation_arguments(cuda='auto'):
    if not (sys.platform.startswith('linux') or sys.platform == 'win32') or platform.machine().lower() not in ('x86_64', 'amd64'):
        raise RuntimeError('Automatic CUDA installation supports Linux/Windows x86_64 only. Use CPU accelerated.')
    if not (3, 8) <= sys.version_info[:2] <= (3, 14):
        raise RuntimeError('This CUDA PyTorch installer supports Slicer Python 3.8–3.14. Use CPU accelerated.')
    if cuda == 'auto':
        cuda = 'cu126' if sys.version_info[:2] >= (3, 10) else 'cu121'
    profiles = cuda_profiles()
    if cuda not in profiles:
        raise RuntimeError('The selected CUDA build has no installation profile for this Slicer Python.')
    package, index = 'torch==' + profiles[cuda], 'https://download.pytorch.org/whl/' + cuda
    return ['-m', 'pip', 'install', '--disable-pip-version-check', '--no-input',
            '--progress-bar', 'off', '--only-binary=:all:', package, '--index-url', index]


def cuda_profile_selector():
    import qt
    selector = qt.QComboBox()
    selector.addItem('Auto (Slicer Python default)', 'auto')
    for profile in cuda_profiles():
        selector.addItem('CUDA %s.%s' % (profile[2:-1], profile[-1]), profile)
    selector.toolTip = 'Select a PyTorch CUDA build supported by your NVIDIA driver. Existing working CUDA PyTorch is reused.'
    return selector


def python_slicer():
    import slicer
    executable = os.path.join(slicer.app.slicerHome, 'bin',
                              'PythonSlicer.exe' if sys.platform == 'win32' else 'PythonSlicer')
    if not os.path.isfile(executable):
        raise RuntimeError('PythonSlicer was not found: ' + executable)
    return executable


class TorchInstaller:
    def __init__(self, allow, button, log, profile):
        import qt
        import slicer
        self.allow, self.button, self.log = allow, button, log
        self.profile = profile
        self.profile.enabled = allow.checked
        self.process = qt.QProcess(slicer.app)
        self.process.setProcessChannelMode(qt.QProcess.MergedChannels)
        self.process.connect('readyReadStandardOutput()', self.read_output)
        self.process.connect('finished(int,QProcess::ExitStatus)', self.finished)
        self.process.connect('errorOccurred(QProcess::ProcessError)', self.failed)
        self.phase = None
        allow.connect('toggled(bool)', self.toggle)
        button.connect('clicked(bool)', self.start)

    def toggle(self, checked):
        self.button.enabled = checked and self.phase is None
        self.profile.enabled = checked and self.phase is None

    def read_output(self):
        output = self.process.readAllStandardOutput().data().decode('utf-8', errors='replace')
        if output:
            self.log.appendPlainText(output.rstrip())

    def launch(self, phase, arguments):
        self.phase = phase
        self.process.start(self.executable, arguments)

    def start(self, *args):
        global ACTIVE_INSTALLER
        if not self.allow.checked:
            return
        if ACTIVE_INSTALLER is not None:
            self.log.appendPlainText('Another WMA PyTorch installation is running. Wait for it to finish.')
            return
        try:
            self.executable = python_slicer()
        except RuntimeError as exc:
            self.log.appendPlainText(str(exc))
            return
        ACTIVE_INSTALLER = self
        self.allow.enabled = self.button.enabled = False
        self.profile.enabled = False
        self.log.appendPlainText('Checking CUDA in Slicer Python: ' + self.executable)
        self.launch('check', ['-c', PROBE])

    def finished(self, exit_code, exit_status):
        import qt
        self.read_output()
        if exit_status != qt.QProcess.NormalExit:
            self.complete('Installer process crashed. Review the log and retry.')
        elif self.phase == 'check':
            if exit_code == 0:
                self.complete('CUDA PyTorch is already available. No packages changed.')
            else:
                try:
                    self.arguments = installation_arguments(self.profile.itemData(self.profile.currentIndex))
                except RuntimeError as exc:
                    self.complete(str(exc))
                    return
                self.log.appendPlainText('Installing CUDA PyTorch. Download may take several minutes.\n' +
                                         ' '.join(self.arguments))
                self.launch('install', self.arguments)
        elif self.phase == 'install':
            if exit_code:
                self.complete('PyTorch installation failed (exit %d). Review the log, check network/disk permissions, then retry.' % exit_code)
            else:
                self.launch('verify', ['-c', PROBE])
        elif self.phase == 'verify':
            if exit_code == 0:
                self.complete('CUDA PyTorch installation verified. Restart Slicer before using GPU.')
            else:
                self.complete('pip finished, but CUDA verification failed. Review the log and NVIDIA driver/GPU availability. Restart Slicer; CPU accelerated remains available.')

    def failed(self, error):
        import qt
        if error == qt.QProcess.FailedToStart:
            self.complete('Unable to start PythonSlicer: ' + self.process.errorString())

    def complete(self, message):
        global ACTIVE_INSTALLER
        self.log.appendPlainText(message)
        self.phase = None
        ACTIVE_INSTALLER = None
        self.allow.enabled = True
        self.button.enabled = self.allow.checked
        self.profile.enabled = self.allow.checked


def add_install_row(form, widget):
    import qt
    allow = qt.QCheckBox('Allow downloading and installing CUDA PyTorch into Slicer')
    allow.toolTip = 'Optional. Downloads several GB; NVIDIA GPU/driver and writable Slicer Python are required. Restart after installation.'
    button = qt.QPushButton('Check / Install PyTorch')
    button.enabled = False
    log = qt.QPlainTextEdit()
    log.readOnly = True
    log.maximumHeight = 120
    log.setPlainText('Optional GPU setup: enable the checkbox, then click Check / Install. CPU needs no PyTorch.')
    profile = cuda_profile_selector()
    form.addRow('', allow)
    form.addRow('CUDA build:', profile)
    form.addRow('', button)
    form.addRow('PyTorch setup:', log)
    widget.torchInstaller = TorchInstaller(allow, button, log, profile)
