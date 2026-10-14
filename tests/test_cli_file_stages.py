"""File workers preserve the original CLI geometry, labels, order and metadata."""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
import whitematteranalysis as wma

ROOT = Path(__file__).resolve().parents[1]
BASELINE = 'd80c014'


def tractography(count=9):
    data = vtk.vtkPolyData()
    points, lines = vtk.vtkPoints(), vtk.vtkCellArray()
    points.SetDataTypeToDouble()
    for fiber in range(count):
        lines.InsertNextCell(7)
        for point in range(7):
            x = (-10., 10., (point-3.)*5.)[fiber % 3]
            lines.InsertCellPoint(points.InsertNextPoint(x, point, fiber))
    data.SetPoints(points)
    data.SetLines(lines)
    for attributes, size in ((data.GetPointData(), 7*count), (data.GetCellData(), count)):
        for name, components, dtype in (('scalar', 1, np.float64), ('rgb', 3, np.uint8), ('tensor', 9, np.float32)):
            values = np.arange(size*components, dtype=dtype).reshape(size, components)
            array = numpy_to_vtk(values, deep=True)
            array.SetName(name)
            attributes.AddArray(array)
        attributes.SetActiveScalars('scalar')
        attributes.SetActiveTensors('tensor')
    text = vtk.vtkStringArray()
    text.SetName('description')
    text.InsertNextValue('Preserve tract metadata')
    data.GetFieldData().AddArray(text)
    return data


def assert_equal(test, before, after):
    np.testing.assert_array_equal(vtk_to_numpy(before.GetPoints().GetData()), vtk_to_numpy(after.GetPoints().GetData()))
    for name in ('GetOffsetsArray', 'GetConnectivityArray'):
        np.testing.assert_array_equal(vtk_to_numpy(getattr(before.GetLines(), name)()),
                                      vtk_to_numpy(getattr(after.GetLines(), name)()))
    for location in ('GetPointData', 'GetCellData', 'GetFieldData'):
        original, actual = getattr(before, location)(), getattr(after, location)()
        test.assertEqual(original.GetNumberOfArrays(), actual.GetNumberOfArrays())
        for index in range(original.GetNumberOfArrays()):
            a = original.GetAbstractArray(index)
            b = actual.GetAbstractArray(index)
            test.assertEqual(a.GetName(), b.GetName())
            test.assertEqual(a.GetDataType(), b.GetDataType())
            test.assertEqual(a.GetNumberOfComponents(), b.GetNumberOfComponents())
            if isinstance(a, vtk.vtkDataArray):
                np.testing.assert_array_equal(vtk_to_numpy(a), vtk_to_numpy(b))
            else:
                test.assertEqual([a.GetVariantValue(i).ToString() for i in range(a.GetNumberOfValues())],
                                 [b.GetVariantValue(i).ToString() for i in range(b.GetNumberOfValues())])
        if location != 'GetFieldData':
            for role in ('GetScalars', 'GetTensors', 'GetVectors', 'GetNormals'):
                a, b = getattr(original, role)(), getattr(actual, role)()
                test.assertEqual(a.GetName() if a else None, b.GetName() if b else None)


class FileStageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='wma file stages ')
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(ROOT),
            WMA_GPU='off', CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1')

    def run_cli(self, name, arguments, baseline=False, check=True):
        script = ROOT/'bin'/name
        if baseline:
            script = self.directory/('baseline_'+name)
            script.write_bytes(subprocess.check_output(['git', 'show', BASELINE+':bin/'+name], cwd=ROOT))
        result = subprocess.run([sys.executable, str(script)]+list(map(str, arguments)),
            cwd=ROOT, env=self.environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if check:
            self.assertEqual(result.returncode, 0, result.stdout)
        return result

    def compare_directories(self, before, after):
        expected = sorted(p.relative_to(before) for p in before.rglob('*.vtp'))
        self.assertEqual(expected, sorted(p.relative_to(after) for p in after.rglob('*.vtp')))
        for relative in expected:
            self.assertEqual((before/relative).read_bytes(), (after/relative).read_bytes(), str(relative))

    def test_lossless_vtp_roundtrip(self):
        for count in (0, 9):
            data = tractography(count)
            path = self.directory/('roundtrip%d.vtp' % count)
            wma.io.write_polydata(data, str(path))
            assert_equal(self, data, wma.io.read_polydata(str(path)))

    def test_hemisphere_workers_match_original(self):
        original = self.directory/'original'
        original.mkdir()
        for index, count in enumerate((9, 0, 6), 1):
            wma.io.write_polydata(tractography(count), str(original/('cluster_%05d.vtp' % index)))
        source = self.directory/'source'
        shutil.copytree(original, source)
        self.run_cli('wm_assess_cluster_location_by_hemisphere.py', [original], baseline=True)
        expected = self.directory/'expected separated'
        self.run_cli('wm_separate_clusters_by_hemisphere.py', [original, expected], baseline=True)
        for workers in (1, 2):
            current = self.directory/('workers%d' % workers)
            shutil.copytree(source, current)
            assessed = self.directory/('assessed%d' % workers)
            self.run_cli('wm_assess_cluster_location_by_hemisphere.py',
                [current, '-j', workers, '-outputDirectory', assessed])
            self.compare_directories(original, current)
            self.compare_directories(expected, assessed)
            separated = self.directory/('separated%d' % workers)
            self.run_cli('wm_separate_clusters_by_hemisphere.py', [current, separated, '-j', workers])
            self.compare_directories(expected, separated)
            self.assertTrue((separated/'hemisphere_separation_complete.log').is_file())

    def test_point_labels_use_last_point_and_preserve_empty_lines(self):
        data = tractography(3)
        lines = vtk.vtkCellArray()
        for ids in ([4, 1, 6], [], [0, 13]):
            lines.InsertNextCell(len(ids))
            for point in ids:
                lines.InsertCellPoint(point)
        data.SetLines(lines)
        values = np.column_stack((np.arange(21), -np.arange(21))).astype(np.float64)
        array = numpy_to_vtk(values, deep=True)
        array.SetName('HemisphereLocation')
        data.GetPointData().AddArray(array)
        for name in ('wm_assess_cluster_location_by_hemisphere.py', 'wm_separate_clusters_by_hemisphere.py'):
            spec = importlib.util.spec_from_file_location('_point_labels', ROOT/'bin'/name)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            found, labels = module.read_mask_location_from_vtk(data)
            self.assertTrue(found)
            np.testing.assert_array_equal(labels, [6., 0., 13.])

    def test_parallel_worker_errors_fail_the_command(self):
        source = self.directory/'missing labels'
        source.mkdir()
        for index in (1, 2):
            wma.io.write_polydata(tractography(3), str(source/('cluster_%05d.vtp' % index)))
        output = self.directory/'failed output'
        output.mkdir()
        marker = output/'hemisphere_separation_complete.log'
        marker.write_text('Previous completion\n')
        result = self.run_cli('wm_separate_clusters_by_hemisphere.py', [source, output, '-j', 2], check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Hemisphere labels are missing or incomplete', result.stdout)
        self.assertFalse(marker.exists())

    def test_slicer_process_failures_preserve_logs_and_unrelated_files(self):
        source = self.directory/'harden input'
        source.mkdir()
        for index in (1, 2):
            wma.io.write_polydata(tractography(3), str(source/('cluster_%05d.vtp' % index)))
        transform = self.directory/'unused transform.tfm'
        transform.write_text('This launcher test does not evaluate a transform.\n')
        output = self.directory/'harden output'
        output.mkdir()
        unrelated = output/'log.txt'
        unrelated.write_text('Keep this existing log.\n')
        marker = output/'transform_complete.log'
        marker.write_text('Previous completion\n')
        # Python rejects Slicer's flags and exits nonzero without running the helper.
        result = self.run_cli('wm_harden_transform.py',
            [source, output, sys.executable, '-t', transform, '-j', 2], check=False)
        self.assertNotEqual(result.returncode, 0)
        logs = list(output.glob('wma-harden-*/worker_*.log'))
        self.assertTrue(logs)
        self.assertTrue(any(path.stat().st_size for path in logs))
        self.assertEqual(unrelated.read_text(), 'Keep this existing log.\n')
        self.assertFalse(marker.exists())

    def test_anatomical_workers_match_original(self):
        source = self.directory/'separated input'
        for side in ('left_hemisphere', 'right_hemisphere', 'commissural'):
            directory = source/('tracts_'+side)
            directory.mkdir(parents=True)
            for index, count in enumerate((9, 3), 1):
                wma.io.write_polydata(tractography(count), str(directory/('cluster_%05d.vtp' % index)))
        import ast
        tree = ast.parse((ROOT/'bin/wm_append_clusters_to_anatomical_tracts.py').read_text())
        catalogue = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and
                    target.id in ('hemispheric_tracts', 'commissural_tracts') for target in node.targets):
                catalogue.extend(ast.literal_eval(node.value))
        atlas = self.directory/'atlas'
        atlas.mkdir()
        for name in catalogue:
            (atlas/(name+'.mrml')).write_text(''.join('<FiberBundleStorage fileName="cluster_%05d.vtp"/>\n' % index
                                                    for index in (2, 1, 2)))
        expected = self.directory/'original tracts'
        self.run_cli('wm_append_clusters_to_anatomical_tracts.py', [source, atlas, expected], baseline=True)
        self.assertEqual(len(list(expected.glob('*.vtp'))), 74)
        for workers in (1, 2):
            current = self.directory/('tracts%d' % workers)
            self.run_cli('wm_append_clusters_to_anatomical_tracts.py', [source, atlas, current, '-j', workers])
            self.compare_directories(expected, current)
            self.assertTrue((current/'anatomical_tracts_complete.log').is_file())


if __name__ == '__main__':
    unittest.main()
