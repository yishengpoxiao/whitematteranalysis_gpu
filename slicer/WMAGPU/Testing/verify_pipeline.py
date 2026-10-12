"""Run in Slicer: exercise pipeline visualization and saving with bounded data."""
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch
import numpy as np
import vtk
import slicer

try:
    names = ('WMAPipeline', 'WMARegisterToAtlas', 'WMAGroupRegister',
             'WMAAtlasBuilder', 'WMAMergeMRML')
    widgets = [slicer.util.getModule(name).widgetRepresentation().self() for name in names]
    for name, widget in zip(names, widgets):
        assert 'whitematteranalysis: OK' in widget.statusLabel.toPlainText(), name
    assert 'WMALabelFromAtlas' not in slicer.app.moduleManager().factoryManager().registeredModuleNames()
    import WMAGuiCommon as common
    import WMAtlasLabelGPUCore as core
    import whitematteranalysis as wma
    pd = vtk.vtkPolyData()
    points, lines = vtk.vtkPoints(), vtk.vtkCellArray()
    lines.InsertNextCell(10)
    for j in range(10):
        lines.InsertCellPoint(points.InsertNextPoint(-50., j * 8., 0.))
    pd.SetPoints(points)
    pd.SetLines(lines)
    subject = common.add_tractography_node('test_subject')
    subject.SetAndObservePolyData(pd)
    unrelated = common.add_tractography_node('cluster_user_data')
    unrelated_id = unrelated.GetID()
    widget = widgets[0]
    widget.inputSelector.setCurrentNode(subject)
    widget.outlierCheckBox.checked = False
    assert widget.outlierWorkersSpinBox.value == 0
    widget.fiberLengthSpinBox.value = 0.
    transform = vtk.vtkTransform()
    transform.Translate(100., 0., 0.)

    def label(data, *args, **kwargs):
        return dict(output_polydata=data, cluster_idx=np.zeros(1, dtype=int),
                    color=np.ones((1, 3)), cluster_count=1, n_fibers=1, backend_name='CPU')

    def results():
        return [node for node in slicer.util.getNodesByClass('vtkMRMLModelNode')
                if node.GetAttribute('WMAGPU.ResultGroup') == 'pipeline']

    def run():
        widget.onApply()
        log = widget.statusLabel.toPlainText()
        assert 'ERROR:' not in log, log
        assert 'PIPELINE DONE' in log, log
        assert widget.applyButton.enabled

    with tempfile.TemporaryDirectory() as directory:
        atlas = Path(directory) / 'atlas'
        atlas.mkdir()
        (atlas / 'atlas.p').touch()
        (atlas / 'registration_atlas.vtk').touch()
        (atlas / 'T_AF.mrml').write_text('<FiberBundleStorage fileName="cluster_00001.vtp"/>\n')
        widget.atlasDirectory = str(atlas)
        widget.outputDirectory = str(Path(directory) / 'output')
        widget.saveCheckBox.checked = True
        with patch.object(core, 'run_atlas_labeling', side_effect=label), patch.object(
                core, 'register_subject_to_atlas', return_value=dict(transform=transform, iterations=1)), patch.object(
                wma.io, 'read_polydata', return_value=pd):
            run()
        nodes = results()
        assert len(nodes) == 1 and nodes[0].GetName() == 'T_AF_right'
        assert nodes[0].GetPolyData().GetPoint(0)[0] == -50.
        assert nodes[0].GetDisplayNode().GetVisibility()
        saved = wma.io.read_polydata(str(Path(widget.outputDirectory) / 'anatomical_tracts/T_AF_right.vtp'))
        assert saved.GetPoint(0)[0] == -50.
        assert not list(Path(widget.outputDirectory).rglob('*.mrml'))
        widget.registerCheckBox.checked = False
        widget.anatomicalCheckBox.checked = False
        widget.saveCheckBox.checked = False
        with patch.object(core, 'run_atlas_labeling', side_effect=label):
            run()
            assert len(results()) == 1 and results()[0].GetName() == 'WMA cluster overview'
            widget.visualizationComboBox.currentIndex = 1
            widget.outlierCheckBox.checked = True
            widget.outlierWorkersSpinBox.value = 4
            run()
            assert len(results()) == 1 and results()[0].GetName() == 'cluster_00001'
            assert not results()[0].GetDisplayNode().GetVisibility()
            before = results()[0].GetID()
            widget.visualizationComboBox.currentIndex = 2
            run()
            assert results()[0].GetID() == before  # Save-only leaves the scene intact.
    assert slicer.mrmlScene.GetNodeByID(unrelated_id) is not None
    assert not slicer.mrmlScene.IsBatchProcessing()
    empty = vtk.vtkPolyData()
    start = time.monotonic()
    nodes = common.show_clusters_in_scene([pd] * 800 + [empty], base_name='batch_test', visible=False)
    assert len(nodes) == 800 and not any(n.GetDisplayNode().GetVisibility() for n in nodes)
    common.show_clusters_in_scene([], base_name='batch_test')
    assert not slicer.mrmlScene.IsBatchProcessing()
    print('BATCH_800_SECONDS:', time.monotonic() - start, flush=True)
    with patch.object(common, 'add_tractography_node', side_effect=RuntimeError('test failure')):
        try:
            common.show_clusters_in_scene([pd], base_name='failure_test')
        except RuntimeError:
            pass
        else:
            raise AssertionError('Expected node creation failure')
    assert not slicer.mrmlScene.IsBatchProcessing()
    import unittest
    sys.path.append(str(Path(__file__).resolve().parent))
    import test_outlier_parallel
    assert test_outlier_parallel.core is core
    report = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(test_outlier_parallel))
    assert report.wasSuccessful(), 'Parallel outlier tests failed'
    print('PIPELINE_VISUALIZATION_VERIFIED', flush=True)
except Exception:
    import traceback
    traceback.print_exc()
    sys.exit(1)
sys.exit(0)
