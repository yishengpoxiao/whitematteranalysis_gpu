"""Bounded functional tests of the extension core, without Slicer or CUDA."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'Modules/_vendor'))
import numpy as np
import vtk
import whitematteranalysis as wma
import WMAtlasLabelGPUCore as core


def tractography(seed):
    rng = np.random.RandomState(seed)
    pd = vtk.vtkPolyData()
    points, lines = vtk.vtkPoints(), vtk.vtkCellArray()
    for i in range(150):
        base = rng.normal(size=3)*8.
        # A subject-only population should be rejected by leave-one-out.
        if seed == 1 and i >= 100:
            base[1] += 150.
        lines.InsertNextCell(10)
        for j in range(10):
            xyz = base + [j*8., 2*np.sin(j + i), 2*np.cos(j + i)]
            lines.InsertCellPoint(points.InsertNextPoint(*xyz))
    pd.SetPoints(points)
    pd.SetLines(lines)
    return pd


class CoreTests(unittest.TestCase):
    def test_labeling_respects_cpu_worker_setting(self):
        import types
        rng = np.random.RandomState(88)
        reference = tractography(1)
        atlas = types.SimpleNamespace(number_of_eigenvectors=3,
            nystrom_polydata=reference, threshold=0., sigma=20.,
            distance_method='Mean', bilateral=False,
            row_sum_1=np.ones(150), row_sum_matrix=np.zeros(150),
            e_vec=rng.uniform(.1, 1, (150, 4)), e_val=np.arange(1., 5.),
            e_vec_norm=np.ones(4), centroids=rng.uniform(0, 2, (7, 3)))
        cpu = wma.gpu.cpu_backend
        with patch.object(core, 'load_atlas', return_value=atlas), \
                patch.object(wma.gpu.config, 'cpu_threads', 4), \
                patch.object(cpu, '_distance_matrix', wraps=cpu._distance_matrix) as matrix:
            parallel = core.run_atlas_labeling(tractography(2), '.', 0., backend='cpu')
            self.assertTrue(matrix.call_args_list)
            self.assertTrue(all(call.kwargs['number_of_jobs'] == 4 for call in matrix.call_args_list))
            matrix.reset_mock()
            serial = core.run_atlas_labeling(tractography(2), '.', 0., backend='cpu', number_of_jobs=1)
            self.assertTrue(all(call.kwargs['number_of_jobs'] == 1 for call in matrix.call_args_list))
        np.testing.assert_array_equal(parallel['cluster_idx'], serial['cluster_idx'])
        for index in range(parallel['output_polydata'].GetCellData().GetNumberOfArrays()):
            first = parallel['output_polydata'].GetCellData().GetArray(index)
            second = serial['output_polydata'].GetCellData().GetArray(first.GetName())
            np.testing.assert_array_equal(vtk.util.numpy_support.vtk_to_numpy(first),
                                          vtk.util.numpy_support.vtk_to_numpy(second))

    def tearDown(self):
        core.resolve_backend('auto')

    def test_explicit_gpu_requires_cuda(self):
        with patch.object(wma.gpu, 'available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'CUDA'):
                core.resolve_backend('gpu')
        self.assertIn('CPU accelerated', core.resolve_backend('cpu'))

    def test_single_file_mrml(self):
        with tempfile.TemporaryDirectory() as directory:
            wma.io.write_polydata(tractography(1), str(Path(directory)/'cluster_00001.vtp'))
            root = ET.parse(core.create_mrml_file(directory)).getroot()
            self.assertEqual(root.tag, 'MRML')
            self.assertEqual(len(root.findall('FiberBundleStorage')), 1)

    def test_build_reload_label(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory)/'subjects', Path(directory)/'atlas'
            source.mkdir()
            for i in (1, 2):
                wma.io.write_polydata(tractography(i), str(source/('%d.vtp' % i)))
            result = core.build_atlas(str(source), str(output), fibers_per_subject=150,
                fiber_length_mm=0., number_of_clusters=3, nystrom_sample=100, iterations=1,
                cluster_outlier_std=100., subject_percent=.4, random_seed=42, backend='cpu')
            atlas = core.load_atlas(result['atlas_directory'])
            self.assertEqual(atlas.centroids.shape[0], result['cluster_count'])
            clusters = sorted(Path(result['atlas_directory']).glob('cluster_*.vtp'))
            self.assertEqual(len(clusters), result['cluster_count'])
            for i, path in enumerate(clusters):
                pd = wma.io.read_polydata(str(path))
                array = vtk.util.numpy_support.vtk_to_numpy(pd.GetCellData().GetArray('ClusterNumber'))
                np.testing.assert_array_equal(array, np.full(len(array), i))
            labeled = core.run_atlas_labeling(tractography(2), result['atlas_directory'],
                                              fiber_length_mm=0., backend='cpu')
            self.assertEqual(labeled['n_fibers'], 150)
            self.assertTrue(np.all(labeled['cluster_idx'] < result['cluster_count']))

    def test_groupwise_registration(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory)/'subjects', Path(directory)/'registered'
            source.mkdir()
            for i in (1, 2):
                wma.io.write_polydata(tractography(i), str(source/('%d.vtp' % i)))
            result = core.register_groupwise(str(source), str(output), fibers_per_subject=20,
                fiber_length_mm=0., backend='cpu', random_seed=42, parallel_jobs=1,
                params=dict(points_per_fiber=10, sigma=[20.], initial_step=[1.], final_step=[.1],
                            maxfun=[2], mean_brain_size=[10], subject_brain_size=[10], iterations=[1]))
            self.assertEqual(result['n_subjects'], 2)
            self.assertTrue(list(output.rglob('registration_atlas.vtk')))


if __name__ == '__main__':
    unittest.main()
