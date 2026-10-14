"""CPU distance work must honor tile bounds at every worker count."""

import unittest
import warnings
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np
import whitematteranalysis as wma
from whitematteranalysis.gpu import cpu_backend as cpu


def fiber_array(values):
    result = wma.fibers.FiberArray()
    result.number_of_fibers, result.points_per_fiber = values.shape[:2]
    result.fiber_array_r, result.fiber_array_a, result.fiber_array_s = values.transpose(2, 0, 1)
    return result


class CPUTilingTests(unittest.TestCase):
    def test_zero_threshold_mean_matches_original(self):
        rng = np.random.RandomState(24)
        with np.errstate(all='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            for points in (1, 3, 10, 15, 20, 50):
                for scale in (1e-150, 1., 1e150, 1e160):
                    queries = rng.normal(size=(17, points, 3)) * scale
                    fixed = rng.normal(size=(257, points, 3)) * scale
                    queries[-1, 0, 0] = np.nan
                    reference, moving = fiber_array(fixed), fiber_array(queries)
                    for bilateral in (False, True):
                        expected = np.array([wma.similarity.fiber_distance(moving.get_fiber(i), reference,
                            threshold=0., distance_method='Mean', bilateral=bilateral) for i in range(17)])
                        for workers in (1, 4):
                            # Includes a compiled full tile and a small NumPy tail.
                            actual = cpu._distance_matrix(queries, fixed, 0., 'Mean', bilateral,
                                tile_rows=16, number_of_jobs=workers)
                            np.testing.assert_array_equal(actual, expected)

    def test_compiled_and_small_tiles_match_original(self):
        rng = np.random.RandomState(23)
        for points in (1, 2, 3, 7, 10, 15, 20, 31, 50):
            queries, fixed = rng.normal(size=(37, points, 3)), rng.normal(size=(257, points, 3))
            reference, moving = fiber_array(fixed), fiber_array(queries)
            for method in ('Mean', 'MeanSquared', 'Hausdorff', 'StrictSimilarity'):
                for bilateral in (False, True):
                    expected = np.array([wma.similarity.fiber_distance(moving.get_fiber(i), reference,
                        threshold=.1, distance_method=method, sigmasq=6400, bilateral=bilateral)
                        for i in range(37)])
                    for workers in (1, 4):
                        actual = cpu._distance_matrix(queries, fixed, .1, method, bilateral,
                            tile_rows=16, number_of_jobs=workers)
                        np.testing.assert_array_equal(actual, expected)

    def test_joblib_worker_counts(self):
        from joblib import effective_n_jobs
        for requested in (1, 4, -1, -2):
            self.assertEqual(cpu.cpu_threads(requested), effective_n_jobs(requested))

    def test_parallel_objective_matches_original(self):
        rng = np.random.RandomState(22)
        for points in (1, 7, 10, 15, 20):
            for n_fixed, n_moving, rows in ((31, 17, 3), (257, 37, 16)):
                fixed = rng.normal(size=(3, n_fixed, points))
                moving = rng.normal(size=(3, n_moving, points))
                for sigma in (1e-5, 100., 1e6):
                    expected = wma.register_two_subjects.inner_loop_objective(fixed, moving, sigma)
                    for workers in (1, 4):
                        with ThreadPoolExecutor(max_workers=workers) as pool:
                            actual = cpu.inner_loop_objective(fixed, moving, sigma,
                                chunk_rows=rows, threads=workers, executor=pool)
                        self.assertEqual(actual, expected)

    def test_tiled_distance_matches_original_and_limits_temporaries(self):
        rng = np.random.RandomState(21)
        queries, fixed = rng.normal(size=(13, 5, 3)), rng.normal(size=(7, 5, 3))
        reference, moving = fiber_array(fixed), fiber_array(queries)
        block = cpu._fiber_block_distances
        for method in ('Mean', 'MeanSquared', 'Hausdorff', 'StrictSimilarity'):
            for bilateral in (False, True):
                expected = np.array([wma.similarity.fiber_distance(moving.get_fiber(i), reference,
                    threshold=.1, distance_method=method, sigmasq=6400, bilateral=bilateral)
                    for i in range(13)])
                for workers in (1, 4):
                    sizes = []
                    def record(q, *args):
                        sizes.append(len(q))
                        return block(q, *args)
                    with patch.object(cpu, '_fiber_block_distances', side_effect=record):
                        actual = cpu._distance_matrix(queries, fixed, .1, method, bilateral,
                                                      tile_rows=2, number_of_jobs=workers)
                    self.assertLessEqual(max(sizes), 2)
                    np.testing.assert_array_equal(actual, expected)

    def test_automatic_tile_can_be_smaller_than_eight_rows(self):
        self.assertEqual(cpu._tile_rows(1_000_000, 20, 8, 4), 1)


if __name__ == '__main__':
    unittest.main()
