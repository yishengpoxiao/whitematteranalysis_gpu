"""Verify installed modules in a fresh Slicer process using their installed path."""
import os
from pathlib import Path
import sys
import tempfile
import numpy as np
import slicer

try:
    root = Path(os.environ['WMA_EXTENSION_INSTALL_ROOT'])/'WMAGPU'
    names = ['WMAPipeline', 'WMARegisterToAtlas', 'WMAGroupRegister',
             'WMAAtlasBuilder', 'WMAMergeMRML']
    widgets = []
    for name in names:
        module = slicer.util.getModule(name)
        assert module is not None, name + ' was not discovered after restart'
        widget = module.widgetRepresentation().self()
        assert hasattr(widget, 'statusLabel'), name + ' setup failed'
        assert 'whitematteranalysis: OK' in widget.statusLabel.toPlainText(), widget.statusLabel.toPlainText()
        widgets.append(widget)
        print('MODULE_OK:', name, flush=True)
    import whitematteranalysis as wma
    assert root in Path(wma.__file__).parents, 'Loaded external WMA: ' + wma.__file__
    from whitematteranalysis.gpu import distance_kernel, bspline
    print('BUNDLED_WMA:', wma.__file__, flush=True)
    import WMAGuiCommon as common
    import WMAtlasLabelGPUCore as core
    # A model node must work even without SlicerDMRI; verify selector signals.
    import vtk
    pd = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    for i in range(20):
        lines.InsertNextCell(10)
        for j in range(10):
            lines.InsertCellPoint(points.InsertNextPoint(j*8., i*.2, i*.1))
    pd.SetPoints(points)
    pd.SetLines(lines)
    node = common.add_tractography_node('WMA_extension_test')
    node.SetAndObservePolyData(pd)
    for widget in widgets:
        if hasattr(widget, 'inputSelector'):
            widget.inputSelector.setCurrentNode(node)
            assert widget.applyButton.enabled, type(widget).__name__ + ' node signal failed'
    # Synthetic atlas keeps CPU/GPU coverage independent of external datasets.
    class Atlas:
        pass
    atlas = Atlas()
    rng = np.random.RandomState(42)
    atlas.nystrom_polydata = pd
    atlas.threshold, atlas.sigma, atlas.distance_method, atlas.bilateral = 0., 20., 'Mean', False
    atlas.number_of_eigenvectors = 3
    atlas.row_sum_1 = np.ones(20)
    atlas.row_sum_matrix = rng.uniform(0, .1, 20)
    atlas.e_vec = rng.uniform(.1, 1, (20, 20))
    atlas.e_val, atlas.e_vec_norm = np.arange(1., 21.), np.ones(20)
    atlas.centroids = rng.uniform(0, 2, (4, 3))
    outputs = []
    backends = ['cpu'] + (['gpu'] if wma.gpu.available() else [])
    for backend in backends:
        core.resolve_backend(backend)
        data = vtk.vtkPolyData()
        data.DeepCopy(pd)
        result = wma.cluster.spectral_atlas_label(data, atlas)
        outputs.append(result)
        print('LABEL_OK:', backend, flush=True)
    if len(outputs) == 2:
        np.testing.assert_array_equal(outputs[0][1], outputs[1][1])
        np.testing.assert_allclose(outputs[0][3], outputs[1][3], rtol=1e-12, atol=1e-12)
    clusters = core.extract_clusters(outputs[-1][0], outputs[-1][1], 4)
    with tempfile.TemporaryDirectory() as directory:
        common.save_clusters_parallel(clusters, directory)
        assert len(list(Path(directory).glob('cluster_*.vtp'))) == 4
        scene = core.create_mrml_file(directory)
        assert Path(scene).is_file()
    # Reduced affine optimization exercises the GUI's actual registration core.
    result = core.register_subject_to_atlas(pd, pd, fiber_length_mm=0., backend=backends[-1],
        params=dict(points_per_fiber=10, sigma=[20.], initial_step=[1.], final_step=[.1],
                    maxfun=[2], mean_brain_size=[10], subject_brain_size=[10], iterations=[1]))
    assert core.apply_transform_to_polydata(pd, result['transform']).GetNumberOfLines() == 20
    print('REGISTRATION_OK', flush=True)
    # Run atlas persistence and groupwise output checks against installed code.
    import unittest
    sys.path.append(str(Path(__file__).resolve().parent))
    import test_core
    assert root in Path(test_core.core.__file__).parents
    report = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_core))
    assert report.wasSuccessful(), 'Installed core functional tests failed'
    print('EXTENSION_VERIFIED', flush=True)
except Exception:
    import traceback
    traceback.print_exc()
    sys.exit(1)
sys.exit(0)
