"""CPU streaming must preserve atlas labels, metadata and serialized geometry."""
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import vtk
import whitematteranalysis as wma
from test_atlas_blocks import polydata


class CPUAtlasTests(unittest.TestCase):
    def test_explicit_cpu_similarity_does_not_use_global_cuda(self):
        data = polydata(11)
        with patch.object(wma.gpu.config, 'requested', True), \
                patch.object(wma.gpu.config, 'cpu_fast_enabled', return_value=False), \
                patch.object(wma.gpu.distance, 'rectangular_distance_matrix', side_effect=AssertionError), \
                patch.object(wma.gpu.distance, 'pairwise_distance_matrix', side_effect=AssertionError):
            rectangular = wma.cluster._rectangular_similarity_matrix(
                data, data, 0., 20., number_of_jobs=1, use_gpu=False)
            pairwise = wma.cluster._pairwise_similarity_matrix(
                data, 0., 20., number_of_jobs=1, use_gpu=False)
        np.testing.assert_allclose(rectangular, pairwise, rtol=1e-12, atol=1e-12)

    def test_streamed_matches_full_matrix(self):
        rng = np.random.RandomState(42)
        atlas = types.SimpleNamespace(number_of_eigenvectors=3,
            row_sum_1=np.ones(11), row_sum_matrix=rng.uniform(0, .1, 11),
            e_vec=rng.uniform(.1, 1, (11, 11)), e_val=np.arange(1., 12.),
            e_vec_norm=np.linspace(.8, 1.2, 11), centroids=rng.uniform(0, 2, (5, 3)),
            nystrom_polydata=polydata(11), threshold=.1, sigma=20.)
        subject = polydata(513)  # Includes a partial final block.
        for method in ('Mean', 'MeanSquared', 'Hausdorff', 'StrictSimilarity'):
            for bilateral in (False, True):
                atlas.distance_method, atlas.bilateral = method, bilateral
                original_pd = vtk.vtkPolyData()
                original_pd.DeepCopy(subject)
                with patch.object(wma.gpu.config, 'requested', False), \
                        patch.object(wma.gpu.config, 'cpu_fast_enabled', return_value=False):
                    expected = wma.cluster.spectral_atlas_label(
                        original_pd, atlas, number_of_jobs=1, use_gpu=False)
                for workers in (1, 4):
                    with self.subTest(method=method, bilateral=bilateral, workers=workers):
                        actual_pd = vtk.vtkPolyData()
                        actual_pd.DeepCopy(subject)
                        with patch.object(wma.gpu.config, 'cpu_fast_enabled', return_value=True):
                            actual = wma.cluster.spectral_atlas_label(
                                actual_pd, atlas, number_of_jobs=workers, use_gpu=False)
                        np.testing.assert_array_equal(actual[1], expected[1])
                        np.testing.assert_allclose(actual[2], expected[2], rtol=1e-12, atol=1e-12)
                        np.testing.assert_allclose(actual[3], expected[3], rtol=1e-12, atol=1e-12)
                        with tempfile.TemporaryDirectory() as directory:
                            paths = [Path(directory) / name for name in ('original.vtp', 'streamed.vtp')]
                            for data, path in zip((original_pd, actual_pd), paths):
                                wma.io.write_polydata(data, str(path))
                            self.assertEqual(paths[0].read_bytes(), paths[1].read_bytes())

    def test_empty_subject(self):
        atlas = types.SimpleNamespace(number_of_eigenvectors=3,
            row_sum_1=np.ones(11), e_vec=np.eye(11), e_val=np.arange(1., 12.),
            e_vec_norm=np.ones(11), nystrom_polydata=polydata(11))
        result = wma.gpu.cpu_backend.atlas_label(polydata(0), atlas, number_of_jobs=4)
        self.assertEqual(result[0].shape, (0, 3))
        self.assertEqual(result[1].size, 0)
        np.testing.assert_array_equal(result[3], atlas.row_sum_1)


if __name__ == '__main__':
    unittest.main()
