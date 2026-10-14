"""Laterality reduces bounded row tiles without changing hemisphere arithmetic."""
import unittest
from unittest.mock import patch

import numpy as np
import whitematteranalysis as wma
from whitematteranalysis.gpu import cpu_backend as cpu, distance
from test_atlas_blocks import polydata
from test_cli_file_stages import tractography, assert_equal
from test_cpu_tiling import fiber_array
from test_output_pipeline import baseline_module


class LateralityTests(unittest.TestCase):
    def test_hemisphere_defaults_and_custom_boundaries(self):
        fibers = wma.fibers.FiberArray()
        self.assertEqual(fibers.hemisphere_percent_threshold, .95)
        fibers.number_of_fibers, fibers.points_per_fiber = 6, 10
        fibers.fiber_array_r = np.ones((6, 10))
        fibers.fiber_array_a = np.zeros((6, 10))
        fibers.fiber_array_s = np.zeros((6, 10))
        for i, count in enumerate((10, 9, 8, 2, 1, 0)):
            fibers.fiber_array_r[i, :count] = -1.
        fibers.hemisphere_percent_threshold = .8
        fibers.calculate_hemispheres()
        np.testing.assert_array_equal(fibers.fiber_hemisphere, [-1., -1., 0., 0., 1., 1.])
        self.assertEqual(fibers.get_fibers([0, 5]).hemisphere_percent_threshold, .8)

    def test_index_masks(self):
        left, right = np.array([0., 2., 0., 0., 4.]), np.array([1., 2., 0., 3., 0.])
        expected = np.array([1., 0., 0., 1., -1.])
        np.testing.assert_array_equal(wma.laterality.compute_laterality_index(left, right), expected)
        np.testing.assert_array_equal(wma.laterality.compute_laterality_index(
            left, right, np.array([0, 1, 3, 4])), expected)
        np.testing.assert_array_equal(wma.laterality.compute_laterality_index(np.zeros(3), np.zeros(3)), 0.)

    def test_cpu_totals_match_original(self):
        rng = np.random.RandomState(46)
        for points in (5, 15, 20, 50):
            queries, fixed = rng.normal(size=(37, points, 3)), rng.normal(size=(257, points, 3))
            reflected = np.arange(37) % 2 == 0
            q, f = fiber_array(queries), fiber_array(fixed)
            expected = np.array([wma.similarity.total_similarity_for_laterality(
                q.get_fiber(i), f, reflected[i], .1, 400.) for i in range(len(queries))])
            before = queries.copy()
            for rows in (1, 16, 37):
                matrix = cpu.total_similarity_matrix_strict(queries, fixed, reflected, .1, 400., tile_rows=rows)
                actual = cpu.total_similarity_matrix_strict(queries, fixed, reflected, .1, 400.,
                                                            tile_rows=rows, row_sum=True)
                np.testing.assert_array_equal(actual, matrix.sum(1))
                np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(queries, before)

    def test_empty_dimensions(self):
        for n, m in ((0, 3), (3, 0), (0, 0)):
            actual = cpu.total_similarity_matrix_strict(np.zeros((n, 15, 3)), np.zeros((m, 15, 3)),
                np.zeros(n, bool), 0., 400., row_sum=True)
            np.testing.assert_array_equal(actual, np.zeros(n))

    def test_cpu_output_allocation_is_bounded(self):
        queries, fixed = np.zeros((37, 15, 3)), np.zeros((257, 15, 3))
        original = np.empty
        def bounded(shape, *args, **kwargs):
            self.assertNotEqual(shape, (37, 257), 'Allocated a full result matrix')
            return original(shape, *args, **kwargs)
        with patch.object(np, 'empty', side_effect=bounded):
            actual = cpu.total_similarity_matrix_strict(queries, fixed, np.zeros(37, bool), 0., 400.,
                                                       tile_rows=16, row_sum=True)
        np.testing.assert_array_equal(actual, np.full(37, 257.))

    def compare_pipeline(self, gpu):
        original = baseline_module('whitematteranalysis._lateralitybaseline', 'laterality.py', 'ff8bacb')
        # The preceding NumPy-index None check was invalid; use the corrected
        # index formula in both implementations to isolate matrix streaming.
        original.compute_laterality_index = wma.laterality.compute_laterality_index
        for equal in (False, True):
            outputs = []
            for factory in (original.WhiteMatterLaterality, wma.laterality.WhiteMatterLaterality):
                analysis = factory()
                analysis.parallel_jobs, analysis.verbose, analysis.use_equal_fibers = 1, False, equal
                with patch.object(wma.gpu.config, 'requested', gpu), patch.object(
                        wma.gpu.config, 'cpu_fast_enabled', return_value=True):
                    outputs.append(analysis.compute(tractography(61)))
            for attribute in ('laterality_index', 'left_hem_similarity', 'right_hem_similarity', 'hemisphere'):
                np.testing.assert_array_equal(getattr(outputs[0], attribute), getattr(outputs[1], attribute))
            assert_equal(self, outputs[0].polydata, outputs[1].polydata)
            if not gpu:
                with patch.object(wma.gpu.config, 'requested', False), patch.object(
                        wma.gpu.config, 'cpu_fast_enabled', return_value=False):
                    serial = wma.laterality.WhiteMatterLaterality()
                    serial.parallel_jobs, serial.verbose, serial.use_equal_fibers = 1, False, equal
                    result = serial.compute(tractography(61))
                np.testing.assert_array_equal(result.laterality_index, outputs[1].laterality_index)

    def test_cpu_pipeline_matches_full_and_serial(self):
        self.compare_pipeline(False)


@unittest.skipUnless(wma.gpu.available(), 'CUDA is required')
class GPULateralityTests(unittest.TestCase):
    compare_pipeline = LateralityTests.compare_pipeline

    def test_gpu_pipeline_matches_full(self):
        self.compare_pipeline(True)

    def test_gpu_totals_and_inputs(self):
        rng = np.random.RandomState(47)
        for n, m in ((17, 37), (0, 3), (3, 0)):
            queries, fixed = rng.normal(size=(n, 20, 3)), rng.normal(size=(m, 20, 3))
            reflected = np.arange(n) % 2 == 0
            before = queries.copy()
            for float32 in (False, True):
                for rows in (1, 5):
                    matrix = distance.total_similarity_matrix_strict(queries, fixed, reflected, .1, 400.,
                        gpu_float32=float32, tile_rows=rows)
                    actual = distance.total_similarity_matrix_strict(queries, fixed, reflected, .1, 400.,
                        gpu_float32=float32, tile_rows=rows, row_sum=True)
                    np.testing.assert_array_equal(actual, matrix.sum(1))
                    self.assertEqual(actual.dtype, np.float32 if float32 else np.float64)
                    if not float32:
                        expected = cpu.total_similarity_matrix_strict(queries, fixed, reflected, .1, 400., row_sum=True)
                        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
            np.testing.assert_array_equal(queries, before)

    def test_gpu_retry_and_termination(self):
        import torch
        queries, fixed = np.zeros((5, 15, 3)), np.zeros((3, 15, 3))
        functions = (
            lambda rows: distance.distance_matrix_from_arrays(queries, fixed, 0., tile_rows=rows),
            lambda rows: distance.rectangular_similarity_tensor(polydata(5), polydata(3), 0., 20., tile_rows=rows),
            lambda rows: distance.total_similarity_matrix_strict(queries, fixed, np.array([1, 0, 1, 0, 1], bool),
                                                                 0., 400., tile_rows=rows, row_sum=True))
        original = distance._fiber_block_distances
        def array(value):
            return value.cpu().numpy() if isinstance(value, torch.Tensor) else value
        for function in functions:
            expected = array(function(2))
            sizes = []
            def constrained(q, *args, **kwargs):
                sizes.append(len(q))
                if len(q) > 2:
                    raise torch.cuda.OutOfMemoryError('test tile too large')
                return original(q, *args, **kwargs)
            for requested in (5, 8192):
                sizes.clear()
                with patch.object(distance, '_fiber_block_distances', side_effect=constrained):
                    np.testing.assert_array_equal(array(function(requested)), expected)
                self.assertEqual(sizes, [5, 2, 2, 1])
            sizes.clear()
            def exhausted(q, *args, **kwargs):
                sizes.append(len(q))
                if len(sizes) > 3:
                    raise AssertionError('Retry did not terminate at one row')
                raise torch.cuda.OutOfMemoryError('test exhausted device')
            with patch.object(distance, '_fiber_block_distances', side_effect=exhausted):
                with self.assertRaises(torch.cuda.OutOfMemoryError):
                    function(8192)
            self.assertEqual(sizes, [5, 2, 1])


if __name__ == '__main__':
    unittest.main()
