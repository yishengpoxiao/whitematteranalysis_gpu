"""Registration sampling keeps the previous filter, RNG and VTK semantics."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk

import whitematteranalysis as wma
from test_output_pipeline import baseline_module


def fibers_polydata(point_dtype=np.float64):
    lengths = (8, 7, 6, 5, 4, 3, 2, 8)
    coords = []
    lines = vtk.vtkCellArray()
    for fiber, count in enumerate(lengths):
        start = len(coords)
        for point in range(count):
            z = -60. if fiber == 2 else 0.
            y = float(point)
            if fiber == 4 and point == count - 1:
                y = 0.
            coords.append((float(fiber * 10), y, z))
        ids = vtk.vtkIdList()
        for point_id in range(start, len(coords)):
            ids.InsertNextId(point_id)
        lines.InsertNextCell(ids)

    polydata = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(np.asarray(coords, dtype=point_dtype), deep=True))
    polydata.SetPoints(points)
    polydata.SetLines(lines)
    return polydata


def rng_state():
    state = np.random.get_state()
    return state[0], state[1].copy(), state[2], state[3], state[4]


class RegistrationPreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = baseline_module(
            'whitematteranalysis._oldfilter_registration_preprocess',
            'filter.py', '95ef896')

    def assert_polydata_bytes_equal(self, expected, actual):
        with tempfile.TemporaryDirectory() as directory:
            old_path = Path(directory) / 'old.vtp'
            new_path = Path(directory) / 'new.vtp'
            wma.io.write_polydata(expected, str(old_path))
            wma.io.write_polydata(actual, str(new_path))
            self.assertEqual(old_path.read_bytes(), new_path.read_bytes())

    def test_indices_and_preprocess_output_match_previous_implementation(self):
        options = (
            {'min_length_mm': 3., 'max_length_mm': 7.},
            {'min_length_mm': 3., 'max_length_mm': 7., 'remove_u': True,
             'remove_u_endpoint_dist': 2.},
            {'min_length_mm': 3., 'max_length_mm': 7., 'remove_brainstem': True},
            {'min_length_mm': 3., 'max_length_mm': 7., 'remove_u': True,
             'remove_u_endpoint_dist': 2., 'remove_brainstem': True},
        )
        for dtype in (np.float32, np.float64):
            for kwargs in options:
                with self.subTest(dtype=dtype, kwargs=kwargs):
                    source = fibers_polydata(dtype)
                    expected, expected_indices, expected_lengths, expected_step = (
                        self.original.preprocess(
                            source, return_indices=True, return_lengths=True,
                            preserve_point_data=False, preserve_cell_data=False,
                            verbose=False, **kwargs))
                    actual_indices, actual_lengths, actual_step = (
                        wma.filter._preprocess_indices(
                            source, verbose=False, **kwargs))
                    np.testing.assert_array_equal(actual_indices, expected_indices)
                    np.testing.assert_array_equal(actual_lengths, expected_lengths)
                    self.assertEqual(actual_step, expected_step)
                    actual, returned_indices, lengths, step = wma.filter.preprocess(
                        source, return_indices=True, return_lengths=True,
                        preserve_point_data=False, preserve_cell_data=False,
                        verbose=False, **kwargs)
                    np.testing.assert_array_equal(returned_indices, expected_indices)
                    np.testing.assert_array_equal(lengths, expected_lengths)
                    self.assertEqual(step, expected_step)
                    self.assert_polydata_bytes_equal(expected, actual)

    def test_index_and_mask_fallback_matches_fast_path_baseline(self):
        source = fibers_polydata(np.float64)
        kwargs = {'min_length_mm': 3., 'max_length_mm': 7.,
                  'remove_u': True, 'remove_u_endpoint_dist': 2.,
                  'remove_brainstem': True}
        expected, expected_indices = self.original.preprocess(
            source, return_indices=True, preserve_point_data=False,
            preserve_cell_data=False, verbose=False, **kwargs)
        with patch.object(wma.filter, '_lines_offsets_connectivity', return_value=None):
            actual_indices, _, _ = wma.filter._preprocess_indices(
                source, verbose=False, **kwargs)
            actual = wma.filter.preprocess(
                source, preserve_point_data=False, preserve_cell_data=False,
                verbose=False, **kwargs)
        np.testing.assert_array_equal(actual_indices, expected_indices)
        self.assert_polydata_bytes_equal(expected, actual)

    def legacy_sample(self, source, sample_count, seed, **kwargs):
        filtered = self.original.preprocess(
            source, preserve_point_data=False, preserve_cell_data=False,
            verbose=False, **kwargs)
        return self.original.downsample(
            filtered, sample_count, preserve_point_data=False,
            preserve_cell_data=False, verbose=False, random_seed=seed)

    def test_direct_sampling_matches_legacy_across_iterations_and_rng_states(self):
        for dtype in (np.float32, np.float64):
            for sample_count in (2, 4, 7):
                with self.subTest(dtype=dtype, sample_count=sample_count):
                    source = fibers_polydata(dtype)
                    kwargs = {'min_length_mm': 3., 'max_length_mm': 7.}
                    registration = wma.congeal_to_atlas.SubjectToAtlasRegistration()
                    registration.fiber_length = kwargs['min_length_mm']
                    registration.fiber_length_max = kwargs['max_length_mm']
                    for seed in (1001, 1002, 1003):
                        np.random.seed(77)
                        expected = self.legacy_sample(source, sample_count, seed, **kwargs)
                        expected_state = rng_state()
                        np.random.seed(77)
                        actual = registration._sample_preprocessed_polydata(
                            source, 'subject', sample_count, seed)
                        actual_state = rng_state()
                        np.testing.assert_array_equal(actual_state[1], expected_state[1])
                        self.assertEqual(actual_state[0], expected_state[0])
                        self.assertEqual(actual_state[2:], expected_state[2:])
                        self.assert_polydata_bytes_equal(expected, actual)

                        expected_fibers = wma.fibers.FiberArray()
                        expected_fibers.convert_from_polydata(expected, 15)
                        actual_fibers = wma.fibers.FiberArray()
                        actual_fibers.convert_from_polydata(actual, 15)
                        np.testing.assert_array_equal(
                            actual_fibers.fiber_array_r, expected_fibers.fiber_array_r)
                        np.testing.assert_array_equal(
                            actual_fibers.fiber_array_a, expected_fibers.fiber_array_a)
                        np.testing.assert_array_equal(
                            actual_fibers.fiber_array_s, expected_fibers.fiber_array_s)

    def test_preprocessed_index_cache_invalidates_on_input_or_threshold_change(self):
        source = fibers_polydata()
        registration = wma.congeal_to_atlas.SubjectToAtlasRegistration()
        registration.fiber_length = 3.
        registration.fiber_length_max = 7.
        first = registration._preprocessed_indices(source, 'atlas')
        self.assertIs(first, registration._preprocessed_indices(source, 'atlas'))

        registration.fiber_length_max = 5.
        narrower = registration._preprocessed_indices(source, 'atlas')
        self.assertIsNot(first, narrower)
        np.testing.assert_array_equal(narrower, [3, 4])

        source.GetPoints().SetPoint(0, 0., 0., -60.)
        source.GetPoints().Modified()
        source.Modified()
        changed = registration._preprocessed_indices(source, 'atlas')
        self.assertIsNot(narrower, changed)

    def test_iterate_passes_legacy_sampled_arrays_to_optimizer(self):
        source = fibers_polydata(np.float64)
        registration = wma.congeal_to_atlas.SubjectToAtlasRegistration()
        registration.save_iteration_transforms = False
        registration.mean_brain_size = 2
        registration.subject_brain_size = 4
        registration.fiber_length = 3.
        registration.fiber_length_max = 7.
        registration.set_atlas(source, 'atlas')
        registration.set_subject(source, 'subject')
        captured = {}

        def capture_optimizer(fixed, moving, *args, **kwargs):
            captured['fixed'] = fixed.copy()
            captured['moving'] = moving.copy()
            return args[0], [], 0.

        expected_arrays = []
        for sample_count in (registration.mean_brain_size,
                             registration.subject_brain_size):
            sampled = self.legacy_sample(
                source, sample_count, registration.random_seed + 1,
                min_length_mm=registration.fiber_length,
                max_length_mm=registration.fiber_length_max)
            fibers = wma.fibers.FiberArray()
            fibers.convert_from_polydata(sampled, registration.points_per_fiber)
            expected_arrays.append(np.array([
                fibers.fiber_array_r, fibers.fiber_array_a, fibers.fiber_array_s]))

        with tempfile.TemporaryDirectory() as directory:
            registration.output_directory = directory
            with patch.object(
                    wma.congeal_multisubject, 'congeal_multisubject_inner_loop',
                    side_effect=capture_optimizer), patch.object(
                        wma.gpu, 'use_gpu', return_value=False):
                registration.iterate()

        np.testing.assert_array_equal(captured['fixed'], expected_arrays[0])
        np.testing.assert_array_equal(captured['moving'], expected_arrays[1])

    def test_no_eligible_and_empty_inputs_keep_legacy_short_circuit_rng(self):
        registration = wma.congeal_to_atlas.SubjectToAtlasRegistration()
        registration.fiber_length = 1000.
        registration.fiber_length_max = 2000.
        no_eligible = fibers_polydata()
        np.random.seed(31)
        expected = self.legacy_sample(
            no_eligible, 4, 3001, min_length_mm=1000., max_length_mm=2000.)
        expected_state = rng_state()
        np.random.seed(31)
        actual = registration._sample_preprocessed_polydata(
            no_eligible, 'subject', 4, 3001)
        actual_state = rng_state()
        self.assert_polydata_bytes_equal(expected, actual)
        np.testing.assert_array_equal(actual_state[1], expected_state[1])
        self.assertEqual(actual_state[2:], expected_state[2:])

        empty = vtk.vtkPolyData()
        np.random.seed(32)
        expected_state = rng_state()
        actual_empty = registration._sample_preprocessed_polydata(
            empty, 'atlas', 4, 3002)
        self.assertIs(actual_empty, empty)
        self.assertEqual(rng_state()[2:], expected_state[2:])


if __name__ == '__main__':
    unittest.main()
