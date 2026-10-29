"""The GPU outlier CLI consumes per-subject reductions and preserves logs."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import vtk
import whitematteranalysis as wma
from whitematteranalysis.gpu import atlas_cache


SCRIPT = Path(__file__).resolve().parents[1] / 'bin/wm_cluster_remove_outliers.py'
SPEC = importlib.util.spec_from_file_location('outlier_reduction_cli', SCRIPT)
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


def subject_polydata():
    data = vtk.vtkPolyData()
    points, lines = vtk.vtkPoints(), vtk.vtkCellArray()
    for fiber in range(3):
        lines.InsertNextCell(3)
        for point in range(3):
            lines.InsertCellPoint(points.InsertNextPoint(fiber, point, 0.))
    data.SetPoints(points)
    data.SetLines(lines)
    return data


class CLIOutlierReductionTests(unittest.TestCase):
    def test_gpu_uses_reductions_without_a_full_matrix_and_logs_same_stats(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            atlas_dir, input_dir, output_dir = (root / name for name in
                                                ('atlas', 'subject', 'output'))
            atlas_dir.mkdir()
            input_dir.mkdir()
            (atlas_dir / 'cluster_00001.vtp').touch()
            (input_dir / 'cluster_00001.vtp').touch()
            polydata = subject_polydata()
            arrays = np.zeros((3, 15, 3), dtype=np.float64)
            atlas_arrays = np.zeros((3, 15, 3), dtype=np.float64)
            saved = []
            reduction = patch.object(
                wma.gpu.distance, 'similarity_row_sums_from_arrays',
                return_value=(np.array([.3, 1.5, 3.]), np.array([3., 6., 9.])))
            no_matrix = patch.object(
                wma.gpu.distance, 'distance_matrix_from_arrays',
                side_effect=AssertionError('full pairwise matrix must not be built'))
            arguments = [str(SCRIPT), str(input_dir), str(atlas_dir), str(output_dir),
                         '-gpu', 'on']
            with patch.object(sys, 'argv', arguments), \
                    patch.object(wma.gpu, 'apply_gpu_arguments'), \
                    patch.object(wma.gpu, 'use_gpu', return_value=True), \
                    patch.object(wma.cluster, 'load_atlas', return_value=types.SimpleNamespace(bilateral=False)), \
                    patch.object(atlas_cache, 'load_reference',
                                 return_value=(atlas_arrays, np.array([.3, .5, .7]))), \
                    patch.object(wma.gpu.fiber_tensors, 'polydata_to_numpy', return_value=arrays), \
                    patch.object(wma.io, 'read_polydata', return_value=polydata), \
                    patch.object(wma.io, 'write_polydata', side_effect=lambda data, path: saved.append(data)), \
                    reduction as reduce_rows, no_matrix:
                cli.main()

            reduce_rows.assert_called_once()
            self.assertEqual(reduce_rows.call_args.args[0].shape, (3, 15, 3))
            self.assertTrue(reduce_rows.call_args.kwargs['sqrt_distance_sums'])
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved[0].GetNumberOfLines(), 2)
            log = (output_dir / 'subject_outlier_removed' / 'cluster_log.txt').read_text().splitlines()
            fields = log[1].split()
            self.assertEqual(fields[:4], ['1', '3', '2', '1'])
            np.testing.assert_allclose(np.asarray(fields[5:], dtype=float),
                                       [2., 2.5, .5333333333333333, .75])


if __name__ == '__main__':
    unittest.main()
