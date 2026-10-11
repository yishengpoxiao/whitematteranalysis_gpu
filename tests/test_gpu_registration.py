"""Compare the fused registration path with the original VTK/NumPy code.

Run: CUDA_VISIBLE_DEVICES=0 python -m unittest discover -s tests
"""
import unittest
from unittest.mock import patch

import numpy as np
import whitematteranalysis as wma
from whitematteranalysis.gpu import registration as gpu
from whitematteranalysis.register_two_subjects import (
    RegisterTractography, inner_loop_objective, transform_fiber_array_numpy)


@unittest.skipUnless(wma.gpu.available(), 'CUDA is required')
class RegistrationTests(unittest.TestCase):
    def test_fused_objective(self):
        rng = np.random.RandomState(42)
        fixed = rng.normal(size=(3, 31, 10)) * 40
        moving = rng.normal(size=(3, 17, 10)) * 40
        transform = np.array([5., -3., 2., 4., -2., 1., 1.02, .98, 1.01,
                              1., -1., .5, -.5, 2., -2.])
        for mode in ([1]*4, [1, 0, 0, 0], [0, 1, 0, 0]):
            with self.subTest(mode=mode):
                original = transform_fiber_array_numpy(moving, transform, mode)
                tensor = gpu._transform_fiber_tensor(moving, transform, mode, None)
                np.testing.assert_array_equal(original, tensor.cpu().numpy())
                expected = inner_loop_objective(fixed, original, 400.)
                for rows in (1, 7, 17):
                    value = gpu.inner_loop_objective(fixed, tensor, 400., chunk_rows=rows)
                    np.testing.assert_allclose(value, expected, rtol=1e-12)
                # The fused objective must not call the NumPy-returning wrapper.
                with patch.object(gpu, 'transform_fiber_array', side_effect=AssertionError):
                    value = gpu.objective_value(fixed, moving, transform, mode, 20.)
                split = gpu.inner_loop_objective(fixed, original, 400.)
                self.assertEqual(value, split)
                reg = RegisterTractography()
                reg.fixed, reg.moving = fixed, moving
                reg.use_gpu, reg.mode, reg.sigma = True, mode, 20.
                self.assertEqual(reg.objective_function(transform), value)
                self.assertEqual(reg.objective_function_values, [value])


if __name__ == '__main__':
    unittest.main()
