"""Bounded registration checks: CLI schedules, transform types, and real outputs."""

import ast
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import vtk

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT / 'slicer/WMAGPU/Modules/_vendor'))
import whitematteranalysis as wma
import WMAtlasLabelGPUCore as core
from WMARegistration import PAIRWISE_PRESETS, GROUPWISE_PRESETS, registration_parameters


def tractography():
    data = vtk.vtkPolyData()
    points, lines = vtk.vtkPoints(), vtk.vtkCellArray()
    for i in range(8):
        lines.InsertNextCell(6)
        for j in range(6):
            lines.InsertCellPoint(points.InsertNextPoint(-20. + j * 8., i * 2., np.sin(i + j)))
    data.SetPoints(points)
    data.SetLines(lines)
    return data


def small_schedule(nonrigid=False):
    return dict(points_per_fiber=3, sigma=[20.], iterations=[1], maxfun=[2],
                mean_brain_size=[4], subject_brain_size=[4], initial_step=[1.],
                final_step=[.1], grid_resolution=[3],
                modes=['Nonrigid' if nonrigid else 'Rigid'])


class RegistrationTests(unittest.TestCase):
    def tearDown(self):
        core.resolve_backend('auto')

    def test_every_cli_preset_matches(self):
        for filename, presets in (('wm_register_to_atlas_new.py', PAIRWISE_PRESETS),
                                  ('wm_register_multisubject_faster.py', GROUPWISE_PRESETS)):
            tree = ast.parse((ROOT / 'bin' / filename).read_text())
            names = set()
            for node in ast.walk(tree):
                if not (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                        and isinstance(node.test.left, ast.Name) and node.test.left.id == 'mode'
                        and isinstance(node.test.comparators[0], ast.Constant)):
                    continue
                mode = node.test.comparators[0].value
                values = {}
                for statement in node.body:
                    if isinstance(statement, ast.Assign) and isinstance(statement.targets[0], ast.Name):
                        try:
                            values[statement.targets[0].id] = ast.literal_eval(statement.value)
                        except ValueError:
                            pass
                if 'sigma_per_scale' not in values:
                    continue
                names.add(mode)
                expected = presets[mode]
                count = len(expected['sigma'])
                for field in ('sigma', 'iterations', 'maxfun', 'mean_brain_size',
                              'subject_brain_size', 'initial_step', 'final_step', 'grid_resolution'):
                    if field + '_per_scale' in values:
                        self.assertEqual(expected[field], values[field + '_per_scale'][:count], mode)
                self.assertEqual(expected['points_per_fiber'], values['points_per_fiber'])
                modes = ['Nonrigid' if values['nonrigid'] else 'Affine'] * count
                if 'rigid_scale' in values:
                    modes = ['Rigid' if value else 'Affine' for value in values['rigid_scale']]
                self.assertEqual(expected['modes'], modes)
                registration_parameters(mode, groupwise=filename.startswith('wm_register_multisubject'))
            self.assertEqual(names | {'rigid'}, set(presets))

    def test_invalid_custom_schedules(self):
        cases = [dict(iterations=[1.5]), dict(sigma=[]), dict(sigma=[float('nan')]),
                 dict(grid_resolution=[2], modes=['Nonrigid']), dict(points_per_fiber=1)]
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                registration_parameters('custom', dict(small_schedule(), **values))
        mixed = {key: value * 2 if isinstance(value, list) else value
                 for key, value in small_schedule().items()}
        mixed['modes'] = ['Affine', 'Nonrigid']
        with self.assertRaisesRegex(ValueError, 'first'):
            registration_parameters('custom', mixed)

    def test_pairwise_transform_types_and_inverse(self):
        pd = tractography()
        for backend in ['cpu'] + (['gpu'] if wma.gpu.available() else []):
            for mode in ('affine', 'rigid', 'nonrigid'):
                with self.subTest(backend=backend, mode=mode), tempfile.TemporaryDirectory() as directory:
                    p = small_schedule(mode == 'nonrigid')
                    p['modes'] = [mode.title()]
                    result = core.register_subject_to_atlas(pd, pd, fiber_length_mm=0.,
                        mode=mode, params=p, backend=backend, output_directory=directory)
                    transform = result['transform']
                    self.assertEqual(transform.IsA('vtkBSplineTransform'), mode == 'nonrigid')
                    output = core.apply_transform_to_polydata(pd, transform)
                    inverse = core.apply_transform_to_polydata(output, transform, inverse=True)
                    self.assertEqual(output.GetNumberOfLines(), pd.GetNumberOfLines())
                    np.testing.assert_allclose(vtk.util.numpy_support.vtk_to_numpy(inverse.GetPoints().GetData()),
                        vtk.util.numpy_support.vtk_to_numpy(pd.GetPoints().GetData()), atol=.01)
                    if mode == 'rigid':
                        matrix = np.array([[transform.GetMatrix().GetElement(i, j) for j in range(3)] for i in range(3)])
                        np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-10)

    def test_groupwise_rigid_and_nonrigid(self):
        for mode in ('rigid', 'nonrigid'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / 'subjects'
                source.mkdir()
                pd = tractography()
                for index in (1, 2):
                    wma.io.write_polydata(pd, str(source / ('subject%d.vtp' % index)))
                output = Path(directory) / 'output'
                result = core.register_groupwise(str(source), str(output), fibers_per_subject=8,
                    fiber_length_mm=0., mode=mode, params=small_schedule(mode == 'nonrigid'),
                    backend='cpu', random_seed=42, save_transforms=False)
                self.assertEqual(len(result['transforms']), 2)
                self.assertEqual(len(result['subject_ids']), 2)
                self.assertEqual(len(list((output / 'output_tractography').glob('*.vtk'))), 2)
                self.assertTrue((output / 'registration_atlas.vtk').is_file())
                for transform in result['transforms']:
                    self.assertEqual(transform.IsA('vtkBSplineTransform'), mode == 'nonrigid')

    def test_preprocessing_reuse_preserves_samples_and_invalidates(self):
        registration = wma.congeal_to_atlas.SubjectToAtlasRegistration()
        registration.set_subject(tractography(), 'moving')
        registration.set_atlas(tractography(), 'fixed')
        registration.fiber_length = 0.
        registration.save_iteration_transforms = False

        def reference_sample(pd, count, seed):
            filtered = wma.filter.preprocess(pd, registration.fiber_length,
                max_length_mm=registration.fiber_length_max, return_indices=False,
                preserve_point_data=False, preserve_cell_data=False, verbose=False)
            sampled = wma.filter.downsample(filtered, count, verbose=False, random_seed=seed)
            fibers = wma.fibers.FiberArray()
            fibers.convert_from_polydata(sampled, registration.points_per_fiber)
            return np.array([fibers.fiber_array_r, fibers.fiber_array_a, fibers.fiber_array_s])

        with tempfile.TemporaryDirectory() as directory, patch.object(
                wma.congeal_multisubject, 'congeal_multisubject_inner_loop',
                side_effect=lambda fixed, moving, initial, *a, **kw: (initial, [0., 0.], 0.)) as loop:
            registration.output_directory = directory
            cases = [('initial', 2), ('samples', 0), ('minimum', 2), ('maximum', 2),
                     ('new_subject', 1), ('atlas_points', 1)]
            for change, expected_calls in cases:
                if change == 'samples':
                    registration.mean_brain_size, registration.subject_brain_size = 8, 4
                    registration.points_per_fiber = 5
                elif change == 'minimum':
                    registration.fiber_length = 1.
                elif change == 'maximum':
                    registration.fiber_length_max = 200.
                elif change == 'new_subject':
                    registration.set_subject(tractography(), 'replacement')
                    registration.mean_brain_size, registration.subject_brain_size = 4, 100
                elif change == 'atlas_points':
                    points = registration.atlas_polydata.GetPoints()
                    points.SetPoint(0, -19., 0., 0.)
                    points.Modified()
                seed = registration.random_seed + registration.total_iterations + 1
                np.random.seed(101)
                fixed = reference_sample(registration.atlas_polydata, registration.mean_brain_size, seed)
                moving = reference_sample(registration.subject_polydata, registration.subject_brain_size, seed)
                expected_state = np.random.get_state()
                np.random.seed(101)
                with patch.object(wma.filter, 'preprocess', wraps=wma.filter.preprocess) as preprocess:
                    registration.iterate()
                self.assertEqual(preprocess.call_count, expected_calls, change)
                np.testing.assert_array_equal(loop.call_args.args[0], fixed)
                np.testing.assert_array_equal(loop.call_args.args[1], moving)
                state = np.random.get_state()
                self.assertEqual(state[0], expected_state[0])
                np.testing.assert_array_equal(state[1], expected_state[1])
                self.assertEqual(state[2:], expected_state[2:])


if __name__ == '__main__':
    unittest.main()
