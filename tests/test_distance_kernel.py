"""Fused distance arithmetic retains eager PyTorch values and fallback paths."""
import unittest
from itertools import product
from unittest.mock import patch
import numpy as np
import whitematteranalysis as wma
from whitematteranalysis.gpu import distance, distance_kernel


@unittest.skipUnless(wma.gpu.available(), 'CUDA is required')
class DistanceKernelTests(unittest.TestCase):
    def test_empty_public_matrices(self):
        from test_atlas_blocks import polydata
        empty, populated = polydata(0), polydata(3)
        for method in distance.SUPPORTED_DISTANCE_METHODS:
            result = distance.pairwise_distance_matrix(empty, 0., distance_method=method)
            self.assertEqual(result.shape, (0, 0))
            for queries, fixed, shape in ((empty, populated, (0, 3)),
                                          (populated, empty, (3, 0))):
                result = distance.rectangular_distance_matrix(
                    queries, fixed, 0., distance_method=method)
                self.assertEqual(result.shape, shape)

    def test_float64_preserves_small_differences_at_large_offsets(self):
        queries = np.full((1, 15, 3), 1e8)
        fixed = queries.copy()
        fixed[:, :, 0] += 1.
        for method in ('Mean', 'MeanSquared', 'Hausdorff'):
            precise = distance.distance_matrix_from_arrays(
                queries, fixed, 0., distance_method=method, gpu_float32=False)
            np.testing.assert_array_equal(precise, [[1.]])
            approximate = distance.distance_matrix_from_arrays(
                queries, fixed, 0., distance_method=method, gpu_float32=True)
            np.testing.assert_array_equal(approximate, [[0.]])

    def test_gaussian_similarities_match_frozen_backend(self):
        import torch
        from test_output_pipeline import baseline_module
        original = baseline_module('whitematteranalysis.gpu._gaussianbaseline',
                                   'gpu/distance.py', '8f83f6f')
        torch.manual_seed(83)
        for points, dtype in product((10, 15, 50), (torch.float64, torch.float32)):
            fixed = torch.randn((37, points, 3), dtype=dtype, device='cuda')
            queries = fixed[:5].clone()
            fixed[0, 0, 0], fixed[1, 0, 0] = float('nan'), float('inf')
            before = queries.clone()
            for method, bilateral in product(distance.SUPPORTED_DISTANCE_METHODS, (False, True)):
                for sigma in (1e-5, 20., 1e6, 0., float('inf'), float('nan')):
                    expected = original._similarity_tensor_block(queries, fixed, .1, sigma, method, bilateral, 6400.)
                    actual = distance._similarity_tensor_block(queries, fixed, .1, sigma, method, bilateral, 6400.)
                    torch.testing.assert_close(actual, expected, rtol=0., atol=0., equal_nan=True)
            self.assertTrue(torch.equal(queries, before))

    def test_eager_scratch_matches_frozen_backend(self):
        import torch
        from test_output_pipeline import baseline_module
        original = baseline_module('whitematteranalysis.gpu._scratchbaseline',
                                   'gpu/distance.py', '97bc764')
        rng = np.random.RandomState(78)
        with patch.object(distance_kernel, 'triton', None):
            for points, dtype, strided in product((10, 50), (torch.float32, torch.float64), (False, True)):
                fixed = torch.tensor(rng.normal(size=(37, points, 3)), dtype=dtype, device='cuda')
                moving = fixed[:5].clone()
                fixed[0, 0, 0], fixed[1, 1, 1] = float('nan'), float('inf')
                if strided:
                    fixed, moving = fixed[:, ::2], moving[:, ::2]
                before_fixed, before_moving = fixed.clone(), moving.clone()
                for method, bilateral, threshold in product(distance.SUPPORTED_DISTANCE_METHODS,
                                                            (False, True), (0., .1, float('inf'), float('nan'))):
                    sigmas = (1e-5, 100., 1e6, 0., -1., float('inf'), float('nan')) if method == 'StrictSimilarity' else (6400.,)
                    for sigma in sigmas:
                        expected = original._fiber_block_distances(moving, fixed, threshold, method, sigma, bilateral)
                        actual = distance._fiber_block_distances(moving, fixed, threshold, method, sigma, bilateral)
                        torch.testing.assert_close(actual, expected, rtol=0., atol=0., equal_nan=True)
                torch.testing.assert_close(fixed, before_fixed, rtol=0., atol=0., equal_nan=True)
                torch.testing.assert_close(moving, before_moving, rtol=0., atol=0., equal_nan=True)

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
        for similarity_sigmasq in (None, 400.):
            actual = distance._fiber_block_distances(moving, fixed, .1, 'Mean', 6400.,
                bilateral=True, similarity_sigmasq=similarity_sigmasq)
            with patch.object(distance_kernel, 'triton', None):
                expected = distance._fiber_block_distances(moving, fixed, .1, 'Mean', 6400.,
                    bilateral=True, similarity_sigmasq=similarity_sigmasq)
            self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(torch.cuda.current_device(), previous)
