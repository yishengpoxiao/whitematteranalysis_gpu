"""Explicit CUDA device selection and unused-memory cleanup without startup probes."""

import gc
import json
import sys

ACTIVE_WORKFLOW = None
PROBE = """
import json, torch
devices = []
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        devices.append(dict(index=index, name=properties.name, total=properties.total_memory))
print('WMA_DEVICES=' + json.dumps(devices))
"""


def release_unused_memory(index):
    """Release inactive PyTorch allocations on the selected device, preserving live tensors."""
    torch = sys.modules.get('torch')
    if index is None or torch is None or not torch.cuda.is_initialized():
        return None
    if not torch.cuda.memory_reserved(index):
        return dict(device=index, released=0, allocated=torch.cuda.memory_allocated(index), reserved=0)
    gc.collect()
    with torch.cuda.device(index):
        torch.cuda.synchronize(index)
        before = torch.cuda.memory_reserved(index)
        torch.cuda.empty_cache()
        return dict(device=index, released=before - torch.cuda.memory_reserved(index),
                    allocated=torch.cuda.memory_allocated(index), reserved=torch.cuda.memory_reserved(index))


class GPUControls:
    def __init__(self, form, widget):
        import qt
        import slicer
        self.widget = widget
        self.device = qt.QComboBox()
        self.device.setEditable(True)
        self.device.addItem('0: Default visible GPU')
        self.device.toolTip = 'Choose a CUDA device or enter its visible index. This controls WMA computation, not Slicer rendering.'
        self.refresh = qt.QPushButton('Detect GPUs')
        row = qt.QWidget()
        layout = qt.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.device)
        layout.addWidget(self.refresh)
        form.addRow('Compute GPU:', row)
        self.autoRelease = qt.QCheckBox('Release unused GPU memory after each run')
        self.autoRelease.checked = True
        form.addRow(self.autoRelease)
        self.release = qt.QPushButton('Release unused GPU memory now')
        form.addRow(self.release)
        self.process = qt.QProcess(slicer.app)
        self.process.setProcessChannelMode(qt.QProcess.MergedChannels)
        self.process.connect('finished(int,QProcess::ExitStatus)', self.detected)
        self.process.connect('errorOccurred(QProcess::ProcessError)', self.detectFailed)
        self.refresh.connect('clicked(bool)', self.detect)
        self.release.connect('clicked(bool)', self.releaseNow)

    def selected_device(self):
        try:
            index = int(self.device.currentText.split(':', 1)[0].strip())
        except ValueError:
            raise ValueError('Choose a GPU or enter a nonnegative CUDA device index.')
        if index < 0:
            raise ValueError('CUDA device index must be nonnegative.')
        return index

    def log(self, text):
        if hasattr(self.widget, 'statusLabel'):
            self.widget.statusLabel.appendPlainText(text)

    def detect(self):
        from WMATorchInstaller import python_slicer
        self.refresh.enabled = False
        try:
            executable = python_slicer()
            self.process.start(executable, ['-c', PROBE])
        except Exception as exc:
            self.refresh.enabled = True
            self.log(str(exc))

    def detectFailed(self, error):
        import qt
        if error == qt.QProcess.FailedToStart:
            self.refresh.enabled = True
            self.log('GPU detection could not start: ' + self.process.errorString())

    def detected(self, code, status):
        import qt
        output = self.process.readAllStandardOutput().data().decode('utf-8', errors='replace')
        self.refresh.enabled = True
        try:
            if code or status != qt.QProcess.NormalExit:
                raise RuntimeError(output.strip() or 'CUDA PyTorch is unavailable. Use CPU or install PyTorch.')
            devices = json.loads(next(line[12:] for line in output.splitlines() if line.startswith('WMA_DEVICES=')))
            selected = self.selected_device()
            if not devices:
                self.log('No CUDA GPUs detected. Use CPU accelerated or install CUDA PyTorch.')
                return
            self.device.clear()
            for device in devices:
                self.device.addItem('%d: %s (%.1f GB)' % (device['index'], device['name'], device['total'] / 2**30))
            self.device.currentIndex = selected if selected < len(devices) else 0
            self.log('Detected %d CUDA GPU(s). IDs refer to CUDA-visible devices.' % len(devices))
        except Exception as exc:
            self.log('GPU detection: ' + str(exc))

    def report_release(self, report):
        if report is None:
            self.log('No initialized CUDA allocator to release.')
        else:
            self.log('GPU %d: released %.1f MiB; live tensors %.1f MiB; reserved %.1f MiB. '
                     'CUDA context and Slicer rendering memory can remain.' % (
                         report['device'], report['released'] / 2**20,
                         report['allocated'] / 2**20, report['reserved'] / 2**20))

    def releaseNow(self):
        if ACTIVE_WORKFLOW is not None:
            self.log('Wait for the active WMA computation before releasing GPU memory.')
            return
        try:
            self.report_release(release_unused_memory(self.selected_device()))
        except Exception as exc:
            self.log('GPU cleanup: ' + str(exc))

    def cleanup(self):
        if self.process.state():
            self.process.terminate()


def start_computation(widget):
    global ACTIVE_WORKFLOW
    if ACTIVE_WORKFLOW is not None:
        raise RuntimeError('Another WMA workflow is running. Wait for it to finish.')
    ACTIVE_WORKFLOW = widget
    widget._activeGPU = None
    widget._gpuMemoryReleased = False
    import whitematteranalysis as wma
    index = widget.gpuControls.selected_device()
    wma.gpu.set_devices([index])
    if widget.backendComboBox.currentIndex < 2 and wma.gpu.available():
        import torch
        if index >= torch.cuda.device_count():
            raise ValueError('CUDA GPU %d is unavailable. Click Detect GPUs and choose an available device.' % index)
        torch.cuda.set_device(index)
        widget._activeGPU = index
        widget.gpuControls.log('WMA compute device: GPU %d (%s)' % (index, torch.cuda.get_device_name(index)))
    widget.gpuControls.device.enabled = False
    widget.gpuControls.refresh.enabled = False
    widget.backendComboBox.enabled = False


def release_workflow_memory(widget):
    if widget.gpuControls.autoRelease.checked and widget._activeGPU is not None:
        try:
            widget.gpuControls.report_release(release_unused_memory(widget._activeGPU))
            widget._gpuMemoryReleased = True
        except Exception as exc:
            widget.gpuControls.log('GPU cleanup: ' + str(exc))


def finish_computation(widget):
    global ACTIVE_WORKFLOW
    if ACTIVE_WORKFLOW is not widget:
        return
    try:
        if not widget._gpuMemoryReleased:
            release_workflow_memory(widget)
    finally:
        ACTIVE_WORKFLOW = None
        widget.gpuControls.device.enabled = True
        widget.gpuControls.refresh.enabled = True
        widget.backendComboBox.enabled = True
