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

    def test_optimizer_reuses_data_and_refreshes_samples(self):
        rng = np.random.RandomState(123)
        reg = RegisterTractography()
        reg.fixed = rng.normal(size=(3, 31, 10)) * 20
        reg.moving = rng.normal(size=(3, 17, 10)) * 20
        reg.use_gpu = True
        reg.maxfun = 25
        reg.mode = [1, 1, 0, 0]
        for optimizer in ('Cobyla', 'BFGS', 'Powell'):
            reg.optimizer = optimizer
            for scale in range(2):
                with self.subTest(optimizer=optimizer, scale=scale):
                    # Same objects, new samples: an identity-only cache is unsafe.
                    reg.fixed += 2.
                    reg.moving *= .95
                    reg.sigma = 10. + scale * 5.
                    fixed_before, moving_before = reg.fixed.copy(), reg.moving.copy()
                    reg.objective_function_values.clear()
                    with patch.object(gpu, 'AffineObjective', wraps=gpu.AffineObjective) as factory:
                        actual = reg.compute().copy()
                    factory.assert_called_once()
                    self.assertIsNone(reg._gpu_objective)
                    actual_values = reg.objective_function_values.copy()
                    # Run the same optimizer with the preceding stateless GPU path.
                    def stateless(fixed, moving):
                        return lambda x, mode, sigma: gpu.objective_value(
                            fixed, moving, x, mode, sigma)
                    reg.objective_function_values.clear()
                    with patch.object(gpu, 'AffineObjective', side_effect=stateless):
                        expected = reg.compute()
                    np.testing.assert_array_equal(actual, expected)
                    np.testing.assert_array_equal(actual_values, reg.objective_function_values)
                    np.testing.assert_array_equal(reg.fixed, fixed_before)
                    np.testing.assert_array_equal(reg.moving, moving_before)

    def test_optimizer_failure_releases_device_data(self):
        reg = RegisterTractography()
        reg.fixed = np.zeros((3, 3, 10))
        reg.moving = np.ones((3, 2, 10))
        reg.use_gpu = True

        def fail(objective, *args, **kwargs):
            objective(reg.initial_transform * reg.transform_scaling)
            raise RuntimeError('optimizer failed')

        with patch('scipy.optimize.fmin_cobyla', side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, 'optimizer failed'):
                reg.compute()
        self.assertIsNone(reg._gpu_objective)
        # A direct call outside compute must read current arrays.
        reg.fixed += 4.
        x = reg.initial_transform * reg.transform_scaling
        self.assertEqual(reg.objective_function(x), gpu.objective_value(
            reg.fixed, reg.moving, x, reg.mode, reg.sigma))


class CPURegistrationTests(unittest.TestCase):
    def test_cpu_optimizer_never_prepares_cuda_data(self):
        rng = np.random.RandomState(321)
        reg = RegisterTractography()
        reg.fixed = rng.normal(size=(3, 7, 10)) * 20
        reg.moving = rng.normal(size=(3, 5, 10)) * 20
        reg.use_gpu, reg.maxfun = False, 20
        with patch.object(gpu, 'AffineObjective', side_effect=AssertionError):
            actual = reg.compute().copy()
        values = reg.objective_function_values.copy()
        reg.objective_function_values.clear()
        with patch.object(reg, '_using_cpu_fast', return_value=False):
            expected = reg.compute()
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(values, reg.objective_function_values)
        self.assertIsNone(reg._gpu_objective)


if __name__ == '__main__':
    unittest.main()
