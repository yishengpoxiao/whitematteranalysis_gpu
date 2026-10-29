"""Hemisphere labels stay identical when added during outlier output."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

import whitematteranalysis as wma
from whitematteranalysis.gpu import atlas_cache
from whitematteranalysis._hemisphere import assess_polydata, load_location_policy

ROOT = Path(__file__).resolve().parents[1]


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'bin' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


outlier_cli = load_script('hemisphere_outlier_cli', 'wm_cluster_remove_outliers.py')
assessment_cli = load_script('hemisphere_assessment_cli',
                             'wm_assess_cluster_location_by_hemisphere.py')


def polydata():
    data = vtk.vtkPolyData()
    points, lines = vtk.vtkPoints(), vtk.vtkCellArray()
    points.SetDataTypeToDouble()
    for fiber in range(3):
        lines.InsertNextCell(7)
        for point in range(7):
            x = (-10., 10., (point - 3.) * 5.)[fiber]
            lines.InsertCellPoint(points.InsertNextPoint(x, point, fiber))
    data.SetPoints(points)
    data.SetLines(lines)
    return data


class HemisphereFusionTests(unittest.TestCase):
    def test_two_character_location_labels_are_not_truncated(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'locations.tsv'
            path.write_text('Cluster Index\tLocation Label\ncluster_00001\tng\n')
            policy = load_location_policy(path)
        self.assertEqual(policy[0][1], b'ng')
        _, labels, _ = assess_polydata(polydata(), 0, policy)
        np.testing.assert_array_equal(labels, [3., 3., 3.])

    def test_policy_labels_preserve_h_tie_order_and_empty_geometry(self):
        data = polydata()

        class Fibers:
            def __init__(self):
                self.index_left_hem = np.array([0])
                self.index_right_hem = np.array([1])
                self.index_commissure = np.array([2])

            def convert_from_polydata(self, _):
                pass

        policy = np.array([(b'cluster_00001', b'h')],
                          dtype=[('Cluster Index', 'S17'), ('Location Label', 'S1')])
        with patch.object(wma.fibers, 'FiberArray', Fibers):
            data, labels, changed = assess_polydata(data, 0, policy)
        self.assertTrue(changed)
        np.testing.assert_array_equal(labels, [1., 2., 1.])
        stored = vtk_to_numpy(data.GetPointData().GetArray('HemisphereLocation'))
        np.testing.assert_array_equal(stored, np.repeat(labels, 7))

        empty = vtk.vtkPolyData()
        empty.SetPoints(vtk.vtkPoints())
        empty.SetLines(vtk.vtkCellArray())
        empty, labels, changed = assess_polydata(empty, 0, policy)
        self.assertTrue(changed)
        self.assertEqual(len(labels), 0)
        self.assertEqual(empty.GetPointData().GetArray('HemisphereLocation').GetNumberOfTuples(), 0)

    def test_fused_bytes_match_outlier_then_standalone_assessment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            atlas_dir, input_dir, output = (root / n for n in ('atlas', 'subject', 'output'))
            atlas_dir.mkdir()
            input_dir.mkdir()
            for directory in (atlas_dir, input_dir):
                (directory / 'cluster_00001.vtp').touch()
            policy_file = root / 'locations.tsv'
            policy_file.write_text('Cluster Index\tLocation Label\ncluster_00001\th\n')
            arrays = np.zeros((3, 15, 3), dtype=np.float64)
            atlas_arrays = np.zeros((3, 15, 3), dtype=np.float64)
            reference = (atlas_arrays, np.array([.3, .5, .7]))
            reduction = (np.array([.3, 1.5, 3.]), np.array([3., 6., 9.]))

            def run_outlier(destination, fused):
                args = [str(ROOT / 'bin/wm_cluster_remove_outliers.py'), str(input_dir),
                        str(atlas_dir), str(destination), '-gpu', 'on']
                if fused:
                    args += ['--hemisphere-location-file', str(policy_file)]
                with patch.object(sys, 'argv', args), \
                        patch.object(wma.gpu, 'apply_gpu_arguments'), \
                        patch.object(wma.gpu, 'use_gpu', return_value=True), \
                        patch.object(wma.cluster, 'load_atlas',
                                     return_value=types.SimpleNamespace(bilateral=False)), \
                        patch.object(atlas_cache, 'load_reference', return_value=reference), \
                        patch.object(wma.gpu.fiber_tensors, 'polydata_to_numpy', return_value=arrays), \
                        patch.object(wma.gpu.distance, 'similarity_row_sums_from_arrays',
                                     return_value=reduction), \
                        patch.object(wma.io, 'read_polydata', return_value=polydata()):
                    outlier_cli.main()
                return destination / 'subject_outlier_removed' / 'cluster_00001.vtp'

            fused_file = run_outlier(output / 'fused', True)
            sequential_file = run_outlier(output / 'sequential', False)
            self.assertTrue((fused_file.parent / 'cluster_location_by_hemisphere.log').exists())

            with patch.object(sys, 'argv', [str(ROOT / 'bin/wm_assess_cluster_location_by_hemisphere.py'),
                                            str(sequential_file.parent),
                                            '-clusterLocationFile', str(policy_file), '-j', '1']):
                assessment_cli.main()
            self.assertEqual(fused_file.read_bytes(), sequential_file.read_bytes())

    def test_failed_worker_removes_stale_completion_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            atlas_dir, input_dir, output = (root / n for n in ('atlas', 'subject', 'output'))
            atlas_dir.mkdir()
            input_dir.mkdir()
            for directory in (atlas_dir, input_dir):
                (directory / 'cluster_00001.vtp').touch()
            policy_file = root / 'locations.tsv'
            policy_file.write_text('Cluster Index\tLocation Label\ncluster_00001\tc\n')
            result_dir = output / 'subject_outlier_removed'
            result_dir.mkdir(parents=True)
            marker = result_dir / 'cluster_location_by_hemisphere.log'
            marker.write_text('stale')
            args = [str(ROOT / 'bin/wm_cluster_remove_outliers.py'), str(input_dir),
                    str(atlas_dir), str(output), '--hemisphere-location-file', str(policy_file)]
            with patch.object(sys, 'argv', args), \
                    patch.object(wma.gpu, 'apply_gpu_arguments'), \
                    patch.object(wma.gpu, 'use_gpu', return_value=False), \
                    patch.object(wma.cluster, 'load_atlas',
                                 return_value=types.SimpleNamespace(bilateral=False)), \
                    patch.object(outlier_cli, 'Parallel', side_effect=RuntimeError('worker failed')):
                with self.assertRaisesRegex(RuntimeError, 'worker failed'):
                    outlier_cli.main()
            self.assertFalse(marker.exists())

    def test_fused_empty_subject_and_empty_atlas_outputs_are_labeled(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            atlas_dir, input_dir, output = (root / n for n in ('atlas', 'subject', 'output'))
            atlas_dir.mkdir()
            input_dir.mkdir()
            for index in (1, 2):
                for directory in (atlas_dir, input_dir):
                    (directory / ('cluster_%05d.vtp' % index)).touch()
            policy_file = root / 'locations.tsv'
            policy_file.write_text('Cluster Index\tLocation Label\ncluster_00001\tc\ncluster_00002\tc\n')
            empty = vtk.vtkPolyData()
            empty.SetPoints(vtk.vtkPoints())
            empty.SetLines(vtk.vtkCellArray())
            subject_data = polydata()
            args = [str(ROOT / 'bin/wm_cluster_remove_outliers.py'), str(input_dir),
                    str(atlas_dir), str(output), '-gpu', 'on',
                    '--hemisphere-location-file', str(policy_file)]
            with patch.object(sys, 'argv', args), \
                    patch.object(wma.gpu, 'apply_gpu_arguments'), \
                    patch.object(wma.gpu, 'use_gpu', return_value=True), \
                    patch.object(wma.cluster, 'load_atlas',
                                 return_value=types.SimpleNamespace(bilateral=False)), \
                    patch.object(atlas_cache, 'load_reference',
                                 return_value=(np.empty((0, 15, 3)), np.empty(0))), \
                    patch.object(wma.io, 'read_polydata', side_effect=[empty, subject_data]), \
                    patch.object(wma.io, 'write_polydata', wraps=wma.io.write_polydata):
                outlier_cli.main()

            result_dir = output / 'subject_outlier_removed'
            self.assertEqual((result_dir / 'cluster_location_by_hemisphere.log').read_text(),
                             '<wm_assess_cluster_location_by_hemisphere.py> Done!!!')
            for index, expected_count in ((1, 0), (2, 3)):
                result = wma.io.read_polydata(str(result_dir / ('cluster_%05d.vtp' % index)))
                values = vtk_to_numpy(result.GetPointData().GetArray('HemisphereLocation'))
                self.assertEqual(len(values), expected_count * 7)
                np.testing.assert_array_equal(values, np.full(expected_count * 7, 3.))


if __name__ == '__main__':
    unittest.main()
