"""Run inside Slicer: registration controls, transform round-trips, and checked-node merging."""

from pathlib import Path
import os
import sys
import tempfile
import time
from unittest.mock import patch

import numpy as np
import qt
import slicer
import vtk


def short_custom(options, mode):
    options.method.setCurrentText('nonrigid' if mode == 'Nonrigid' else 'rigid')
    options.method.setCurrentText('custom')
    options.table.setRowCount(1)
    for column, value in enumerate([mode, 20, 1, 2, 4, 4, 1, .1, 3]):
        options.table.setItem(0, column, qt.QTableWidgetItem(str(value)))
    options.points.value = 3


try:
    sys.path.append(str(Path(__file__).resolve().parent))
    from test_registration import tractography
    import WMAGuiCommon as common
    import WMAtlasLabelGPUCore as core
    import whitematteranalysis as wma
    from WMAMergeMRML import WMAMergeMRMLLogic

    names = ('WMARegisterToAtlas', 'WMAGroupRegister', 'WMAPipeline', 'WMAMergeMRML')
    widgets = [slicer.util.getModule(name).widgetRepresentation().self() for name in names]
    for name, widget in zip(names, widgets):
        assert hasattr(widget, 'statusLabel'), name + ' UI setup failed'
        assert 'whitematteranalysis: OK' in widget.statusLabel.toPlainText(), name
    if os.environ.get('WMA_EXTENSION_INSTALL_ROOT'):
        installed = Path(os.environ['WMA_EXTENSION_INSTALL_ROOT']) / 'WMAGPU'
        assert installed in Path(core.__file__).parents, core.__file__
        assert installed in Path(wma.__file__).parents, wma.__file__
    register, group, pipeline, merge = widgets
    assert len(merge._sceneObservers) == 3, 'Merge page setup did not complete'
    assert register.registrationOptions.method.findText('rigid_affine_fast') >= 0
    assert register.registrationOptions.method.findText('nonrigid_10') >= 0
    assert group.registrationOptions.method.findText('affine_fast_cheap') >= 0
    assert pipeline.registrationOptions.method.findText('nonrigid_neonate') >= 0
    pipeline.registerCheckBox.checked = False
    assert not pipeline.registrationOptions.method.enabled
    pipeline.registerCheckBox.checked = True
    assert pipeline.registrationOptions.method.enabled

    moving = common.add_tractography_node('moving_subject')
    moving.SetAndObservePolyData(tractography())
    fixed = common.add_tractography_node('fixed_subject')
    fixed.SetAndObservePolyData(tractography())
    register.inputSelector.setCurrentNode(moving)
    register.fixedSelector.setCurrentNode(fixed)
    register.saveCheckBox.checked = False
    register.fiberLengthSpinBox.value = 0.
    register.backendComboBox.currentIndex = 2
    short_custom(register.registrationOptions, 'Rigid')
    register.onApply()
    assert 'ERROR:' not in register.statusLabel.toPlainText(), register.statusLabel.toPlainText()
    assert register.outputNode.GetParentTransformNode() == register.transformNode

    parent = vtk.vtkTransform()
    parent.Translate(12., 0., 0.)
    moving_parent = common.add_registration_transform('existing_moving_transform', parent)
    moving.SetAndObserveTransformNodeID(moving_parent.GetID())
    parent2 = vtk.vtkTransform()
    parent2.Translate(-3., 0., 0.)
    fixed_parent = common.add_registration_transform('existing_fixed_transform', parent2)
    fixed.SetAndObserveTransformNodeID(fixed_parent.GetID())

    with tempfile.TemporaryDirectory() as directory:
        for nonrigid in (False, True):
            if nonrigid:
                vectors = np.zeros(3 * 3 * 3 * 3)
                vectors[::3] = 2.
                transform = wma.register_two_subjects_nonrigid_bsplines.convert_transform_to_vtk(vectors)
            else:
                transform = vtk.vtkTransform()
                transform.Translate(5., 1., 0.)
            register.saveCheckBox.checked = True
            register.outputDirectory = str(Path(directory) / ('nonrigid' if nonrigid else 'affine'))
            short_custom(register.registrationOptions, 'Nonrigid' if nonrigid else 'Affine')
            calls = []

            def registration(moving_pd, fixed_pd, **options):
                np.testing.assert_allclose(moving_pd.GetPoint(0), np.array(moving.GetPolyData().GetPoint(0)) + [12., 0., 0.])
                np.testing.assert_allclose(fixed_pd.GetPoint(0), np.array(fixed.GetPolyData().GetPoint(0)) - [3., 0., 0.])
                calls.append(options)
                return dict(transform=transform, iterations=1, backend_name='test')

            with patch.object(core, 'register_subject_to_atlas', side_effect=registration):
                register.onApply()
            assert 'ERROR:' not in register.statusLabel.toPlainText(), register.statusLabel.toPlainText()
            assert calls[0]['mode'] == 'custom'
            assert calls[0]['params']['modes'] == ['Nonrigid' if nonrigid else 'Affine']
            assert moving.GetParentTransformNode() == moving_parent
            assert fixed.GetParentTransformNode() == fixed_parent
            registered = common.world_polydata(register.outputNode)
            expected = core.apply_transform_to_polydata(common.world_polydata(moving), transform)
            np.testing.assert_allclose(registered.GetPoint(0), expected.GetPoint(0), atol=1e-5)
            files = list(Path(register.outputDirectory).glob('*.h5'))
            assert len(files) == 1, files
            reloaded = slicer.util.loadTransform(str(files[0]))
            np.testing.assert_allclose(reloaded.GetTransformToParent().TransformPoint(10., 2., 1.),
                                       transform.TransformPoint(10., 2., 1.), atol=1e-5)
            saved = wma.io.read_polydata(str(next(Path(register.outputDirectory).glob('*.vtp'))))
            np.testing.assert_allclose(saved.GetPoint(0), expected.GetPoint(0), atol=1e-5)

        source = Path(directory) / 'group_inputs'
        source.mkdir()
        for i in (1, 2):
            wma.io.write_polydata(tractography(), str(source / ('subject%d.vtp' % i)))
        for mode in ('Rigid', 'Nonrigid'):
            group.subjectsDirectory = str(source)
            group.outputDirectory = str(Path(directory) / ('group_' + mode))
            group.fiberLengthSpinBox.value = 0.
            group.backendComboBox.currentIndex = 2
            short_custom(group.registrationOptions, mode)
            group.onApply()
            assert 'ERROR:' not in group.statusLabel.toPlainText(), group.statusLabel.toPlainText()
            assert len(group.transformNodes) == 2
            assert len(list(Path(group.outputDirectory).glob('*.h5'))) == 2
            assert len(list((Path(group.outputDirectory) / 'output_tractography').glob('*.vtk'))) == 2

        atlas = Path(directory) / 'atlas'
        atlas.mkdir()
        (atlas / 'atlas.p').touch()
        (atlas / 'registration_atlas.vtk').touch()
        pipeline.inputSelector.setCurrentNode(moving)
        pipeline.atlasDirectory = str(atlas)
        pipeline.outputDirectory = str(Path(directory) / 'pipeline')
        pipeline.outlierCheckBox.checked = False
        pipeline.anatomicalCheckBox.checked = False
        pipeline.fiberLengthSpinBox.value = 0.
        pipeline.registrationOptions.method.setCurrentText('rigid')
        transform = vtk.vtkTransform()
        transform.Translate(5., 0., 0.)

        def label(data, *args, **kwargs):
            return dict(output_polydata=data, cluster_idx=np.zeros(8, dtype=int),
                        color=np.ones((8, 3)), cluster_count=1, n_fibers=8, backend_name='test')

        with patch.object(core, 'register_subject_to_atlas', return_value=dict(transform=transform, iterations=1)) as run, patch.object(
                wma.io, 'read_polydata', return_value=tractography()), patch.object(core, 'run_atlas_labeling', side_effect=label):
            pipeline.onApply()
        assert 'ERROR:' not in pipeline.statusLabel.toPlainText(), pipeline.statusLabel.toPlainText()
        assert run.call_args.kwargs['mode'] == 'rigid'
        assert pipeline.transformNode.GetScene() == slicer.mrmlScene
        assert len(list(Path(pipeline.outputDirectory).glob('*.h5'))) == 1
        assert not list(Path(pipeline.outputDirectory).rglob('*.mrml'))

    excluded = common.add_tractography_node('excluded_subject')
    excluded.SetAndObservePolyData(tractography())
    with patch.object(common, 'tractography_node_types', return_value=['vtkMRMLModelNode'] * 2):
        merge.refreshNodes()
    ids = [merge.nodeTable.item(row, 0).data(qt.Qt.UserRole) for row in range(merge.nodeTable.rowCount)]
    assert len(ids) == len(set(ids)), 'Overlapping node classes produced duplicate checklist rows'
    chosen = {moving.GetID(), fixed.GetID()}
    for row in range(merge.nodeTable.rowCount):
        item = merge.nodeTable.item(row, 0)
        item.setCheckState(qt.Qt.Checked if item.data(qt.Qt.UserRole) in chosen else qt.Qt.Unchecked)
    assert {node.GetID() for node in merge.selectedNodes()} == chosen
    merge.outputNameEdit.text = 'Selected tract group'
    merge.mergeMethodComboBox.currentIndex = 0
    merge.onMergeNodes()
    hierarchy = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
    folder = merge.mergeResult
    assert hierarchy.GetItemName(folder) == 'Selected tract group'
    assert hierarchy.GetItemParent(hierarchy.GetItemByDataNode(moving)) == folder
    assert hierarchy.GetItemParent(hierarchy.GetItemByDataNode(fixed)) == folder
    assert hierarchy.GetItemParent(hierarchy.GetItemByDataNode(excluded)) != folder
    assert moving.GetParentTransformNode() == moving_parent
    merge.outputNameEdit.text = 'Selected combined tract'
    merge.mergeMethodComboBox.currentIndex = 1
    merge.onMergeNodes()
    combined = merge.mergeResult
    assert combined.GetName() == 'Selected combined tract'
    assert combined.GetPolyData().GetNumberOfLines() == 16
    assert combined.GetParentTransformNode() is None
    np.testing.assert_allclose(combined.GetPolyData().GetPoint(0), common.world_polydata(moving).GetPoint(0))
    assert moving.GetPolyData().GetPointData().GetArray('cluster_idx') is None
    assert fixed.GetPolyData().GetPointData().GetArray('cluster_idx') is None
    assert combined.GetPolyData().GetPointData().GetArray('cluster_idx') is not None
    assert {node.GetID() for node in merge.selectedNodes()} == chosen
    slicer.mrmlScene.RemoveNode(moving)
    slicer.app.processEvents()
    assert {node.GetID() for node in merge.selectedNodes()} == {fixed.GetID()}
    merge.checkAll(False)
    assert not merge.selectedNodes()
    for nodes, name in (([], 'name'), ([fixed], '')):
        try:
            WMAMergeMRMLLogic().merge(nodes, name)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid merge was accepted')
    slicer.app.processEvents()
    refreshes = []
    merge._refreshTimer.connect('timeout()', lambda: refreshes.append(True))
    start = time.perf_counter()

    def during_batch(text):
        slicer.app.processEvents()
        assert not merge._refreshTimer.isActive()
        assert not refreshes, 'Checklist refreshed inside a scene batch'

    batch = common.show_clusters_in_scene([tractography()] * 800,
        base_name='merge_batch_test', visible=False, progress_callback=during_batch)
    slicer.app.processEvents()
    assert len(refreshes) == 1, len(refreshes)
    refreshes.clear()
    folder = WMAMergeMRMLLogic().merge(batch, '800 selected tracts')
    assert not slicer.mrmlScene.IsBatchProcessing()
    assert all(hierarchy.GetItemParent(hierarchy.GetItemByDataNode(node)) == folder for node in batch)
    slicer.app.processEvents()
    assert len(refreshes) == 1, len(refreshes)
    with patch.object(core, 'append_polydatas', side_effect=RuntimeError('merge test failure')):
        try:
            WMAMergeMRMLLogic().merge([fixed], 'failed merge', combine=True)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Merge failure did not propagate')
    assert not slicer.mrmlScene.IsBatchProcessing(), 'Merge failure left scene batch open'
    assert merge.nodeTable.updatesEnabled
    print('MERGE_800_SECONDS:', time.perf_counter() - start, flush=True)
    common.show_clusters_in_scene([], base_name='merge_batch_test')
    merge.cleanup()
    print('REGISTRATION_AND_MERGE_VERIFIED', flush=True)
except Exception:
    import traceback
    traceback.print_exc()
    sys.exit(1)
sys.exit(0)
