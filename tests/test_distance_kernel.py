"""Fused distance arithmetic retains eager PyTorch values and fallback paths."""
import unittest
from itertools import product
from unittest.mock import patch
import numpy as np
import whitematteranalysis as wma
from whitematteranalysis.gpu import distance, distance_kernel


@unittest.skipUnless(wma.gpu.available(), 'CUDA is required')
class DistanceKernelTests(unittest.TestCase):
    def test_strict_probability_scales(self):
        import torch
        rng = np.random.RandomState(48)
        for points in (1, 3, 15, 20, 32, 50):
            fixed = torch.tensor(rng.normal(size=(37, points, 3)), dtype=torch.float64, device='cuda')
            moving = fixed[:5].clone()
            for sigma in (1e-5, 100., 1e6, 0., -1., float('inf'), float('nan')):
                for bilateral in (False, True):
                    actual = distance._fiber_block_distances(moving, fixed, .1,
                        'StrictSimilarity', sigma, bilateral)
                    with patch.object(distance_kernel, 'triton', None):
                        expected = distance._fiber_block_distances(moving, fixed, .1,
                            'StrictSimilarity', sigma, bilateral)
                    torch.testing.assert_close(actual, expected, rtol=0., atol=0., equal_nan=True)

    def test_eager_equivalence(self):
        import torch
        rng = np.random.RandomState(42)
        for points in (1, 2, 3, 7, 10, 15, 16, 20, 31, 32, 50):
            for dtype in (torch.float64, torch.float32):
                fixed = torch.tensor(rng.normal(size=(37, points, 3)) * 40., dtype=dtype, device='cuda')
                moving = torch.tensor(rng.normal(size=(5, points, 3)) * 40., dtype=dtype, device='cuda')
                before = moving.clone()
                for method, bilateral in product(distance.SUPPORTED_DISTANCE_METHODS, (False, True)):
                    for threshold in (0., .1, 2., 100.):
                        with self.subTest(points=points, dtype=dtype, method=method, threshold=threshold, bilateral=bilateral):
                            actual = distance._fiber_block_distances(moving, fixed, threshold, method, 6400., bilateral)
                            with patch.object(distance_kernel, 'triton', None):
                                expected = distance._fiber_block_distances(moving, fixed, threshold, method, 6400., bilateral)
                            self.assertTrue(torch.equal(actual, expected))
                self.assertTrue(torch.equal(moving, before))

    def test_special_values_and_strides(self):
        import torch
        for strided in (False, True):
            fixed = torch.ones((7, 10, 3), dtype=torch.float64, device='cuda')
            fixed[0, 0, 0], fixed[1, 1, 1] = float('nan'), float('inf')
            moving = torch.zeros((3, 10, 3), dtype=torch.float64, device='cuda')
            if strided:
                fixed, moving = fixed[:, ::2], moving[:, ::2]
            for method, bilateral in product(distance.SUPPORTED_DISTANCE_METHODS, (False, True)):
                actual = distance._fiber_block_distances(moving, fixed, .1, method, 6400., bilateral)
                with patch.object(distance_kernel, 'triton', None):
                    expected = distance._fiber_block_distances(moving, fixed, .1, method, 6400., bilateral)
                torch.testing.assert_close(actual, expected, rtol=0., atol=0., equal_nan=True)

    def test_selected_device(self):
        import torch
        if distance_kernel.triton is None or torch.cuda.device_count() < 2:
            self.skipTest('Triton and two GPUs are required')
        previous = torch.cuda.current_device()
        selected = (previous + 1) % torch.cuda.device_count()
        fixed = torch.ones((7, 10, 3), dtype=torch.float64, device=selected)
        moving = torch.zeros((3, 10, 3), dtype=torch.float64, device=selected)
        actual = distance._fiber_block_distances(moving, fixed, .1, 'Mean', 6400., bilateral=True)
        with patch.object(distance_kernel, 'triton', None):
            expected = distance._fiber_block_distances(moving, fixed, .1, 'Mean', 6400., bilateral=True)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(torch.cuda.current_device(), previous)
