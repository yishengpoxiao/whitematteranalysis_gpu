"""Opt-in CUDA PyTorch installation using Slicer's own Python interpreter."""
import os
import platform
import sys

TORCH_INDEX = 'https://download.pytorch.org/whl/cu121'
TORCH_PACKAGE = 'torch==2.4.1+cu121'
PROBE = ("import torch; print('PyTorch:', torch.__version__); "
         "print('CUDA available:', torch.cuda.is_available()); "
         "raise SystemExit(0 if torch.cuda.is_available() else 2)")
ACTIVE_INSTALLER = None


def installation_arguments():
    if not (sys.platform.startswith('linux') or sys.platform == 'win32') or platform.machine().lower() not in ('x86_64', 'amd64'):
        raise RuntimeError('Automatic CUDA installation supports Linux/Windows x86_64 only. Use CPU accelerated.')
    if not (3, 8) <= sys.version_info[:2] <= (3, 12):
        raise RuntimeError('This CUDA PyTorch installer requires Slicer Python 3.8–3.12. Use CPU accelerated.')
    return ['-m', 'pip', 'install', '--disable-pip-version-check', '--no-input',
            '--progress-bar', 'off', TORCH_PACKAGE, '--index-url', TORCH_INDEX]


def python_slicer():
    import slicer
    executable = os.path.join(slicer.app.slicerHome, 'bin',
                              'PythonSlicer.exe' if sys.platform == 'win32' else 'PythonSlicer')
    if not os.path.isfile(executable):
        raise RuntimeError('PythonSlicer was not found: ' + executable)
    return executable


class TorchInstaller:
    def __init__(self, allow, button, log):
        import qt
        import slicer
        self.allow, self.button, self.log = allow, button, log
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
            self.arguments = installation_arguments()
            self.executable = python_slicer()
        except RuntimeError as exc:
            self.log.appendPlainText(str(exc))
            return
        ACTIVE_INSTALLER = self
        self.allow.enabled = self.button.enabled = False
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
                self.log.appendPlainText('Installing %s from %s. Download may take several minutes.' %
                                         (TORCH_PACKAGE, TORCH_INDEX))
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
    form.addRow('', allow)
    form.addRow('', button)
    form.addRow('PyTorch setup:', log)
    widget.torchInstaller = TorchInstaller(allow, button, log)
