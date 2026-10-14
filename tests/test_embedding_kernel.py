"""Fused normalization preserves matrix values, layout and caller storage."""
import unittest
from unittest.mock import patch
import numpy as np
import whitematteranalysis as wma
from whitematteranalysis.gpu import embedding_kernel as kernel, linalg


@unittest.skipUnless(wma.gpu.available(), 'CUDA is required')
class EmbeddingKernelTests(unittest.TestCase):
    def test_values_layout_and_inputs(self):
        import torch
        torch.manual_seed(81)
        for dtype in (torch.float64, torch.float32):
            base = torch.randn((4099, 257), dtype=dtype, device='cuda')
            base[0, :4] = torch.tensor([0., -0., float('inf'), float('nan')], dtype=dtype, device='cuda')
            for B in (base, base.t(), base[:, ::2], base.t()[::2], base[:7], base[:0]):
                a = torch.randn(B.shape[0] * 2, dtype=dtype, device='cuda')[::2]
                c = torch.randn(B.shape[1] * 2, dtype=dtype, device='cuda')[::2]
                before = B.clone()
                actual = kernel.normalized_similarities(B, a, c)
                with patch.object(kernel, 'triton', None):
                    expected = kernel.normalized_similarities(B, a, c)
                torch.testing.assert_close(actual, expected, rtol=0., atol=0., equal_nan=True)
                self.assertEqual(actual.stride(), expected.stride())
                torch.testing.assert_close(B, before, rtol=0., atol=0., equal_nan=True)

    def test_large_embedding_matches_frozen_backend(self):
        import torch
        from test_output_pipeline import baseline_module
        original = baseline_module('whitematteranalysis.gpu._normalizationbaseline',
                                   'gpu/linalg.py', '8f83f6f')
        import types
        rng = np.random.RandomState(81)
        atlas = types.SimpleNamespace(number_of_eigenvectors=3, row_sum_1=np.ones(257),
            row_sum_matrix=rng.uniform(-.01, 0., 257), e_vec=rng.uniform(.1, 1, (257, 4)),
            e_val=np.arange(1., 5.), e_vec_norm=np.linspace(.8, 1.2, 4),
            centroids=rng.uniform(0, 2, (7, 3)))
        prepared = linalg._prepare_atlas(atlas)
        for transposed in (False, True):
            B = torch.rand((257, 4099), dtype=torch.float64, device='cuda')
            if transposed:
                B = B.t().contiguous().t()
            before = B.clone()
            expected = original._atlas_embedding_block(B, atlas, prepared)
            actual = linalg._atlas_embedding_block(B, atlas, prepared)
            for a, b in zip(actual, expected):
                np.testing.assert_array_equal(a, b)
            self.assertTrue(torch.equal(B, before))

    def test_selected_device_and_release(self):
        import gc
        import torch
        if torch.cuda.device_count() < 2:
            self.skipTest('Two GPUs are required')
        previous = torch.cuda.current_device()
        selected = (previous + 1) % torch.cuda.device_count()
        gc.collect()
        allocated = torch.cuda.memory_allocated(selected)
        B = torch.ones((4099, 257), dtype=torch.float64, device=selected)
        a, c = B[:, 0].contiguous(), B[0].contiguous()
        actual = kernel.normalized_similarities(B, a, c)
        self.assertTrue(torch.equal(actual, B))
        self.assertEqual(torch.cuda.current_device(), previous)
        del actual, B, a, c
        gc.collect()
        torch.cuda.synchronize(selected)
        self.assertEqual(torch.cuda.memory_allocated(selected), allocated)


if __name__ == '__main__':
    unittest.main()
