"""B-spline coordinates and optimizer results against VTK's point API."""
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import whitematteranalysis as wma
from whitematteranalysis.gpu import bspline, registration, registration_kernel
from whitematteranalysis.register_two_subjects_nonrigid_bsplines import (
    RegisterTractographyNonrigid, convert_transform_to_vtk)


def reference(fibers, field):
    transform = convert_transform_to_vtk(field)
    out = np.empty(fibers.shape, dtype=np.float64)
    for i in range(fibers.shape[1]):
        for j in range(fibers.shape[2]):
            out[:, i, j] = transform.TransformPoint(*fibers[:, i, j])
    return out


def samples(resolution):
    rng = np.random.RandomState(resolution)
    fibers = rng.uniform(-300, 300, (3, 53, 10))
    # Grid knots and one-ULP neighborhoods include all Zero-border cases.
    knots = -100. + np.arange(-3, resolution + 3) * (200. / (resolution - 1))
    positions = np.concatenate((knots, np.nextafter(knots, -np.inf),
                                np.nextafter(knots, np.inf)))
    fibers.reshape(3, -1)[:, :len(positions)] = positions
    field = rng.normal(size=resolution**3 * 3) * 10.
    return fibers, field


class BatchVTKTests(unittest.TestCase):
    def test_reused_coefficients(self):
        from vtk.util.numpy_support import vtk_to_numpy
        for resolution in (2, 3, 4, 6, 10):
            _, field = samples(resolution)
            cache = bspline._CoefficientGrid(resolution)
            for value in (field, -field, field * 0, field * 3):
                transform = convert_transform_to_vtk(value)
                expected = vtk_to_numpy(transform.GetCoefficientData().GetPointData().GetScalars())
                np.testing.assert_array_equal(cache(value), expected)

    def test_coordinates_bit_identical(self):
        for resolution in (2, 3, 4, 6, 10):
            fibers, field = samples(resolution)
            with self.subTest(resolution=resolution):
                np.testing.assert_array_equal(
                    bspline.transform_fiber_array_numpy(fibers, field), reference(fibers, field))


@unittest.skipUnless(wma.gpu.available(), 'CUDA is required')
class GPUBSplineTests(unittest.TestCase):
    def test_coordinates_and_entropy(self):
        for resolution in (2, 3, 4, 6, 10):
            fibers, field = samples(resolution)
            fixed = fibers[:, :7].copy()
            with self.subTest(resolution=resolution):
                original = reference(fibers, field)
                objective = bspline.NonrigidObjective(fixed, fibers, resolution)
                actual = objective.transform(field).cpu().numpy().transpose(2, 0, 1)
                np.testing.assert_array_equal(actual, original)
                self.assertEqual(objective(field, 20.),
                                 registration.inner_loop_objective(fixed, original, 400.))
                with patch.object(registration_kernel, 'triton', None):
                    fallback = bspline.NonrigidObjective(fixed, fibers, resolution)
                    np.testing.assert_array_equal(
                        fallback.transform(field).cpu().numpy().transpose(2, 0, 1), original)

    def test_complete_optimization_and_release(self):
        rng = np.random.RandomState(42)
        reg = RegisterTractographyNonrigid()
        reg.nonrigid_grid_resolution = 3
        reg.initialize_nonrigid_grid()
        reg.initial_transform = reg.displacement_field_numpy.copy()
        reg.fixed = rng.uniform(-120, 120, (3, 13, 10))
        reg.moving = rng.uniform(-120, 120, (3, 7, 10))
        reg.maxfun, reg.optimizer, reg.use_gpu = 100, 'Cobyla', True
        reg.initial_step, reg.final_step = 5., 1.
        with tempfile.TemporaryDirectory() as directory:
            reg.output_directory = directory
            actual = reg.compute().copy()
            values = reg.objective_function_values.copy()
            self.assertIsNone(reg._gpu_objective)
            reg.objective_function_values.clear()
            reg.objective_computations = 0
            reg.minimum_objective = np.inf
            with patch.object(registration_kernel, 'triton', None):
                expected = reg.compute().copy()
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(values, reg.objective_function_values)
            with patch('scipy.optimize.fmin_cobyla', side_effect=RuntimeError('failed')):
                with self.assertRaisesRegex(RuntimeError, 'failed'):
                    reg.compute()
            self.assertIsNone(reg._gpu_objective)
