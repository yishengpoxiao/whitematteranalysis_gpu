"""Accelerated outlier filtering matches the original joblib path."""
import unittest
from unittest.mock import patch

import numpy as np
import vtk

import whitematteranalysis as wma
from test_output_pipeline import baseline_module


def fibers_at(x_positions):
    polydata = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    for x in x_positions:
        lines.InsertNextCell(3)
        for y in (0., 1., 2.):
            lines.InsertCellPoint(points.InsertNextPoint(x, y, 0.))
    polydata.SetPoints(points)
    polydata.SetLines(lines)
    return polydata


class RemoveOutliersTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = baseline_module(
            'whitematteranalysis._oldfilter_outliers', 'filter.py')

    def run_original(self, data, threshold, method='Mean'):
        with patch.object(wma.gpu, 'use_gpu', return_value=False), \
                patch.object(wma.gpu.config, 'cpu_fast_enabled', return_value=False):
            return self.original.remove_outliers(
                data, threshold, n_jobs=1, distance_method=method)

    def run_cpu_fast(self, data, threshold, method='Mean'):
        with patch.object(wma.gpu, 'use_gpu', return_value=False), \
                patch.object(wma.gpu.config, 'cpu_fast_enabled', return_value=True):
            return wma.filter.remove_outliers(
                data, threshold, n_jobs=1, distance_method=method)

    def run_current_joblib(self, data, threshold):
        with patch.object(wma.gpu, 'use_gpu', return_value=False), \
                patch.object(wma.gpu.config, 'cpu_fast_enabled', return_value=False):
            return wma.filter.remove_outliers(data, threshold, n_jobs=1)

    def assert_same_result(self, actual, expected):
        np.testing.assert_array_equal(actual[1], expected[1])
        self.assertEqual(actual[0].GetNumberOfLines(), expected[0].GetNumberOfLines())
        self.assertEqual(actual[2].GetNumberOfLines(), expected[2].GetNumberOfLines())

    def test_cpu_fast_matches_frozen_baseline_for_supported_methods(self):
        data = fibers_at((0., 1., 2., 3., 100.))
        for method in wma.gpu.fiber_tensors.SUPPORTED_DISTANCE_METHODS:
            with self.subTest(method=method):
                expected = self.run_original(data, 10., method)
                self.assert_same_result(self.run_cpu_fast(data, 10., method), expected)

    def test_cpu_fast_honors_worker_count(self):
        matrix = wma.gpu.cpu_backend._distance_matrix
        with patch.object(wma.gpu.cpu_backend, '_distance_matrix', wraps=matrix) as compute:
            self.run_cpu_fast(fibers_at((0., 1., 2., 3.)), 10.)
        self.assertEqual(compute.call_args.kwargs['number_of_jobs'], 1)

    def test_near_threshold_masks_match_frozen_baseline(self):
        data = fibers_at((0., 1., 2., 3., 100.))
        cutoff = np.sqrt(14. / 3.)
        masks = []
        for threshold in (cutoff * (1. - 1e-8), cutoff * (1. + 1e-8)):
            expected = self.run_original(data, threshold)
            actual = self.run_cpu_fast(data, threshold)
            self.assert_same_result(actual, expected)
            masks.append(actual[1])
        self.assertFalse(masks[0][0])
        self.assertTrue(masks[1][0])

    def test_empty_input_matches_frozen_baseline(self):
        data = fibers_at(())
        expected = self.run_original(data, 10.)
        self.assert_same_result(self.run_cpu_fast(data, 10.), expected)
        self.assert_same_result(self.run_current_joblib(data, 10.), expected)
        self.assertEqual(len(expected[1]), 0)

    @unittest.skipUnless(wma.gpu.available(), 'CUDA is unavailable')
    def test_cuda_matches_frozen_baseline_for_supported_methods(self):
        data = fibers_at((0., 1., 2., 3., 100.))
        for method in wma.gpu.fiber_tensors.SUPPORTED_DISTANCE_METHODS:
            with self.subTest(method=method):
                expected = self.run_original(data, 10., method)
                with patch.object(wma.gpu, 'use_gpu', return_value=True):
                    actual = wma.filter.remove_outliers(
                        data, 10., n_jobs=1, distance_method=method)
                self.assert_same_result(actual, expected)

    def test_small_fiber_counts_use_available_neighbors(self):
        for count, expected_mask in ((1, [False]),
                                     (2, [True, True]),
                                     (3, [True, True, True])):
            data = fibers_at(tuple(float(i) for i in range(count)))
            reference = self.run_current_joblib(data, 2.)
            accelerated = self.run_cpu_fast(data, 2.)
            np.testing.assert_array_equal(reference[1], expected_mask)
            np.testing.assert_array_equal(accelerated[1], expected_mask)


if __name__ == '__main__':
    unittest.main()
