"""CPU distance work must honor tile bounds at every worker count."""

import unittest
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
