"""Run in Slicer: device routing, allocator cleanup, threaded loading, and line display."""

from pathlib import Path
import sys
import tempfile
import threading
import time
from unittest.mock import patch

import numpy as np
import qt
import slicer
import vtk


def wait_until(done, message):
    deadline = time.monotonic() + 40
    while not done():
        if time.monotonic() > deadline:
            raise AssertionError(message)
        slicer.app.processEvents()
        time.sleep(.01)


try:
    sys.path.append(str(Path(__file__).resolve().parent))
    from test_registration import tractography
    import WMAGuiCommon as common
    import WMAGPUControls as controls
    import WMAtlasLabelGPUCore as core
    import whitematteranalysis as wma
    widgets = [slicer.util.getModule(name).widgetRepresentation().self() for name in
               ('WMARegisterToAtlas', 'WMAPipeline', 'WMAGroupRegister', 'WMAAtlasBuilder')]
    register, pipeline, group, builder = widgets
    for widget in widgets:
        assert widget.gpuControls.autoRelease.checked
        assert widget.gpuControls.process.state() == qt.QProcess.NotRunning
        assert widget.gpuControls.selected_device() == 0
    pd = tractography()
    with tempfile.TemporaryDirectory() as directory:
        paths = [str(Path(directory) / ('input' + extension)) for extension in ('.vtp', '.vtk')]
        for path in paths:
            wma.io.write_polydata(pd, path)
        read = wma.io.read_polydata
        threads, heartbeats = [], []

        def slow_read(*args, **kwargs):
            threads.append(threading.get_ident())
            time.sleep(.3)
            return read(*args, **kwargs)

        heartbeat = qt.QTimer(slicer.app)
        heartbeat.setInterval(10)
        heartbeat.connect('timeout()', lambda: heartbeats.append(True))
        heartbeat.start()
        with patch.object(wma.io, 'read_polydata', side_effect=slow_read):
            register.inputLoader.load(paths[0])
            wait_until(lambda: register.inputLoader.future is None, 'Background read did not finish')
        heartbeat.stop()
        assert threads == [threads[0]] and threads[0] != threading.get_ident()
        assert len(heartbeats) > 5, 'GUI event loop blocked during reading'
        moving = register.inputSelector.currentNode()
        assert moving.GetClassName() == 'vtkMRMLModelNode'
        assert moving.GetPolyData().GetNumberOfLines() == 8
        assert not moving.GetDisplayNode().GetVisibility()
        register.fixedLoader.load(paths[1])
        wait_until(lambda: register.fixedLoader.future is None, 'Fixed read did not finish')
        fixed = register.fixedSelector.currentNode()
        assert fixed != moving
        np.testing.assert_array_equal(vtk.util.numpy_support.vtk_to_numpy(fixed.GetPolyData().GetPoints().GetData()),
                                     vtk.util.numpy_support.vtk_to_numpy(pd.GetPoints().GetData()))
        borrowed = common.world_polydata(moving)
        assert borrowed is not moving.GetPolyData()
        assert borrowed.GetPoints() == moving.GetPolyData().GetPoints(), 'Identity transform copied geometry'

        register.gpuControls.device.setEditText('invalid')
        try:
            common.start_computation(register)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid GPU identifier accepted')
        finally:
            common.finish_computation(register)
        assert not common.workflow_running()
        register.gpuControls.device.setEditText('0')
        register.gpuControls.detect()
        wait_until(lambda: register.gpuControls.process.state() == qt.QProcess.NotRunning,
                   'Asynchronous GPU detection did not finish')
        slicer.app.processEvents()
        assert register.gpuControls.refresh.enabled

        import torch
        if torch.cuda.is_available():
            selected = min(1, torch.cuda.device_count() - 1)
            register.gpuControls.device.setEditText(str(selected))
            register.backendComboBox.currentIndex = 1
            register.fiberLengthSpinBox.value = 0.
            register.saveCheckBox.checked = False
            options = register.registrationOptions
            options.method.setCurrentText('custom')
            options.table.setRowCount(1)
            for column, value in enumerate(['Rigid', 20, 1, 2, 4, 4, 1, .1, 3]):
                options.table.setItem(0, column, qt.QTableWidgetItem(str(value)))
            options.points.value = 3
            register.statusLabel.clear()
            register.onApply()
            assert 'ERROR:' not in register.statusLabel.toPlainText(), register.statusLabel.toPlainText()
            assert wma.gpu.get_device().index == selected
            assert register._activeGPU == selected
            assert torch.cuda.memory_allocated(selected) == 0
            assert torch.cuda.memory_reserved(selected) == 0
            assert register.outputNode.GetClassName() == 'vtkMRMLModelNode'

            common.start_computation(register)
            pipeline.onApply()
            register.onApply()
            assert controls.ACTIVE_WORKFLOW is register, 'Reentrant action released another workflow'
            common.finish_computation(register)

            def fail(*args, **kwargs):
                temporary = torch.ones(8 << 20, device=wma.gpu.get_device())
                raise RuntimeError('Intentional GPU computation failure')

            with patch.object(core, 'register_subject_to_atlas', side_effect=fail):
                register.onApply()
            assert 'Intentional GPU computation failure' in register.statusLabel.toPlainText()
            assert not common.workflow_running()
            assert register.applyButton.enabled
            assert torch.cuda.memory_allocated(selected) == 0
            assert torch.cuda.memory_reserved(selected) == 0
        else:
            print('CUDA device-routing checks skipped: CUDA unavailable', flush=True)

        nodes = common.show_clusters_in_scene([pd] * 74, base_name='anatomical_loading_test',
                    names=['T_test_%02d' % i for i in range(74)], node_class='vtkMRMLModelNode')
        assert len(nodes) == 74 and all(node.GetClassName() == 'vtkMRMLModelNode' for node in nodes)
        assert all(node.GetPolyData().GetNumberOfLines() == 8 for node in nodes)
        common.show_clusters_in_scene([], base_name='anatomical_loading_test')
    for widget in widgets:
        widget.cleanup()
    print('GPU_AND_LOADING_VERIFIED', flush=True)
except Exception:
    import traceback
    traceback.print_exc()
    sys.exit(1)
sys.exit(0)
