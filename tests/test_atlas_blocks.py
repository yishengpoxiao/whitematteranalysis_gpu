"""Streamed atlas similarities and embeddings against the previous full matrix."""
import types
import unittest
from unittest.mock import patch

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk
import whitematteranalysis as wma
from whitematteranalysis.gpu import distance, linalg
from test_output_pipeline import baseline_module


def polydata(count):
    pd = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(np.random.RandomState(count).normal(size=(count * 7, 3)), deep=True))
    lines = vtk.vtkCellArray()
    for i in range(count):
        lines.InsertNextCell(7)
        for j in range(7):
            lines.InsertCellPoint(i * 7 + j)
    pd.SetPoints(points)
    pd.SetLines(lines)
    return pd


@unittest.skipUnless(wma.gpu.available(), 'CUDA is required')
class AtlasTests(unittest.TestCase):
    def test_negative_rows_and_input_preservation(self):
        import torch
        original = baseline_module('whitematteranalysis.gpu._packedbaseline',
                                   'gpu/linalg.py', 'cffed63')
        rng = np.random.RandomState(43)
        atlas = types.SimpleNamespace(number_of_eigenvectors=3,
            row_sum_1=np.ones(11), row_sum_matrix=-2. * np.ones(11),
            e_vec=rng.uniform(.1, 1, (11, 11)), e_val=np.arange(1., 12.),
            e_vec_norm=np.ones(11), centroids=rng.uniform(0, 2, (5, 3)))
        similarities = torch.full((11, 7), .5, dtype=torch.float64, device='cuda')
        before = similarities.clone()
        expected = original.atlas_label_embedding(similarities, atlas)
        actual = linalg.atlas_label_embedding(similarities, atlas)
        self.assertTrue(torch.equal(similarities, before))
        for a, b in zip(expected, actual):
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(actual[1], expected[1])
        np.testing.assert_array_equal(actual[3][11:], .1)
        atlas.centroids = np.stack((actual[0][0], actual[0][0], actual[0][0] + 1.))
        tied = linalg.atlas_label_embedding(similarities, atlas)
        self.assertEqual(tied[1][0], 0)
        self.assertEqual(tied[2][0], 0.)
        atlas.centroids[0, 0] = np.nan
        for implementation in (original, linalg):
            with self.assertRaises(ValueError):
                implementation.atlas_label_embedding(similarities, atlas)
        atlas.centroids[0, 0] = 0.
        atlas.row_sum_matrix[:] = -1.
        for implementation in (original, linalg):
            with self.assertRaises(ValueError):
                implementation.atlas_label_embedding(similarities, atlas)

    def test_streamed_results(self):
        import torch
        original = baseline_module('whitematteranalysis.gpu._oldlinalg', 'gpu/linalg.py')
        # linalg uses .config, rather than the top-level package.
        rng = np.random.RandomState(42)
        atlas = types.SimpleNamespace(number_of_eigenvectors=3,
            row_sum_1=np.ones(11), row_sum_matrix=rng.uniform(0, .1, 11),
            e_vec=rng.uniform(.1, 1, (11, 11)), e_val=np.arange(1., 12.),
            e_vec_norm=np.ones(11), centroids=rng.uniform(0, 2, (5, 3)))
        queries, fixed = polydata(13), polydata(11)
        for method in ('Mean', 'MeanSquared', 'Hausdorff', 'StrictSimilarity'):
            for bilateral in (False, True):
                for fp32 in (False, True):
                    with self.subTest(method=method, bilateral=bilateral, fp32=fp32):
                        args = (queries, fixed, .1, 20., method, bilateral)
                        full = distance.rectangular_similarity_tensor(*args, gpu_float32=fp32, tile_rows=4)
                        blocks = list(distance.iter_rectangular_similarity(*args, gpu_float32=fp32, tile_rows=4))
                        self.assertEqual([i for i, _ in blocks], [0, 4, 8, 12])
                        self.assertTrue(torch.equal(torch.cat([b for _, b in blocks], dim=1), full))
                        expected = original.atlas_label_embedding(full, atlas)
                        actual = linalg.atlas_label_embedding_blocks(iter(blocks), atlas, 13)
                        for a, b in zip(expected, actual):
                            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
                        np.testing.assert_array_equal(expected[1], actual[1])
        empty = linalg.atlas_label_embedding_blocks(iter(()), atlas, 0)
        self.assertEqual(empty[0].shape, (0, 3))
        np.testing.assert_array_equal(empty[3], atlas.row_sum_1)

    def test_out_of_memory_retries(self):
        import torch
        original = distance._similarity_tensor_block
        sizes = []
        def limited(queries, *args):
            sizes.append(len(queries))
            if len(queries) > 2:
                raise torch.cuda.OutOfMemoryError('tile')
            return original(queries, *args)
        args = (polydata(7), polydata(3), 0., 20.)
        with patch.object(distance, '_similarity_tensor_block', side_effect=limited):
            blocks = list(distance.iter_rectangular_similarity(*args, tile_rows=7))
        self.assertEqual(sizes[:3], [7, 3, 1])
        self.assertEqual(sum(b.shape[1] for _, b in blocks), 7)
        with patch.object(distance, '_similarity_tensor_block', side_effect=torch.cuda.OutOfMemoryError):
            with self.assertRaises(torch.cuda.OutOfMemoryError):
                list(distance.iter_rectangular_similarity(*args, tile_rows=1))
