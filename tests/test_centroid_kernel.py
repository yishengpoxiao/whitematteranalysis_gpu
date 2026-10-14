"""Short-vector CUDA distances preserve direct cdist values, ties and fallback."""
import unittest
from unittest.mock import patch

import whitematteranalysis as wma
from whitematteranalysis.gpu import distance_kernel, linalg


@unittest.skipUnless(wma.gpu.available(), 'CUDA is required')
class CentroidKernelTests(unittest.TestCase):
    def test_does_not_retain_tensors(self):
        import gc
        import torch
        gc.collect()
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        query = torch.ones((257, 10), dtype=torch.float64, device='cuda')
        centroids = torch.zeros((37, 10), dtype=torch.float64, device='cuda')
        result = linalg._nearest_centroids(query, centroids)
        del result, query, centroids
        gc.collect()
        torch.cuda.synchronize()
        self.assertEqual(torch.cuda.memory_allocated(), baseline)

    def test_exact_direct_distances(self):
        import torch
        if distance_kernel.triton is None:
            self.skipTest('Triton is unavailable')
        torch.manual_seed(42)
        for dimensions in (1, 2, 3, 4, 5, 7, 10, 15, 16, 20, 31, 32):
            for scale in (1e-100, 1., 1e100, 1e160):
                query = torch.randn((37, dimensions), dtype=torch.float64, device='cuda') * scale
                centroids = torch.randn((257, dimensions), dtype=torch.float64, device='cuda') * scale
                actual = distance_kernel.centroid_distances(query, centroids)
                expected = torch.cdist(query, centroids, compute_mode='donot_use_mm_for_euclid_dist')
                self.assertTrue(torch.equal(actual, expected), (dimensions, scale))

    def test_ties_special_values_strides_and_fallback(self):
        import torch
        for dimensions in (0, 1, 10, 32, 33):
            for dtype in (torch.float32, torch.float64):
                query = torch.zeros((7, dimensions), dtype=dtype, device='cuda')
                centroids = torch.ones((5, dimensions), dtype=dtype, device='cuda')
                query[1:] = 1.
                for inputs in ((query, centroids), (query[:, ::2], centroids[:, ::2]),
                               (query[:0], centroids), (query.cpu(), centroids.cpu())):
                    actual = linalg._nearest_centroids(*inputs)
                    with patch.object(distance_kernel, 'triton', None):
                        expected = linalg._nearest_centroids(*inputs)
                    for a, b in zip(actual, expected):
                        self.assertTrue(torch.equal(a, b))
                    self.assertTrue(torch.equal(actual[1], torch.zeros_like(actual[1])))
        query = torch.zeros((3, 10), dtype=torch.float64, device='cuda')
        centroids = torch.ones((7, 10), dtype=torch.float64, device='cuda')
        query[0, 0], query[1, 1] = float('nan'), float('inf')
        actual = linalg._nearest_centroids(query, centroids)
        with patch.object(distance_kernel, 'triton', None):
            expected = linalg._nearest_centroids(query, centroids)
        torch.testing.assert_close(actual[0], expected[0], rtol=0., atol=0., equal_nan=True)
        self.assertTrue(torch.equal(actual[1], expected[1]))

    def test_selected_device(self):
        import torch
        if torch.cuda.device_count() < 2:
            self.skipTest('Two GPUs are required')
        previous = torch.cuda.current_device()
        selected = (previous + 1) % torch.cuda.device_count()
        query = torch.ones((7, 10), dtype=torch.float64, device=selected)
        centroids = torch.zeros((257, 10), dtype=torch.float64, device=selected)
        actual = linalg._nearest_centroids(query, centroids)
        with patch.object(distance_kernel, 'triton', None):
            expected = linalg._nearest_centroids(query, centroids)
        self.assertEqual(torch.cuda.current_device(), previous)
        for a, b in zip(actual, expected):
            self.assertTrue(torch.equal(a, b))


if __name__ == '__main__':
    unittest.main()
