"""Geometry-only reference reads preserve topology and independent ownership."""
from concurrent.futures import ThreadPoolExecutor
import gc
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
import whitematteranalysis as wma
from test_cli_file_stages import tractography, assert_equal
from test_output_pipeline import baseline_module


def assert_geometry(test, expected, actual):
    np.testing.assert_array_equal(vtk_to_numpy(expected.GetPoints().GetData()),
                                  vtk_to_numpy(actual.GetPoints().GetData()))
    for cells in ('GetVerts', 'GetLines', 'GetPolys', 'GetStrips'):
        for array in ('GetOffsetsArray', 'GetConnectivityArray'):
            np.testing.assert_array_equal(vtk_to_numpy(getattr(getattr(expected, cells)(), array)()),
                                          vtk_to_numpy(getattr(getattr(actual, cells)(), array)()))
    for attributes in ('GetPointData', 'GetCellData', 'GetFieldData'):
        test.assertEqual(getattr(actual, attributes)().GetNumberOfArrays(), 0)


class GeometryReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = baseline_module('whitematteranalysis._readerbaseline', 'io.py', 'ff8bacb')

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_default_and_geometry_reads(self):
        for extension in ('.vtk', '.vtp'):
            for count in (0, 9):
                data = tractography(count)
                path = str(self.root / ('input%d%s' % (count, extension)))
                wma.io.write_polydata(data, path)
                original = self.original.read_polydata(path)
                assert_equal(self, original, wma.io.read_polydata(path))
                assert_geometry(self, original, wma.io.read_polydata(path, geometry_only=True))
                assert_equal(self, original, wma.io.read_polydata(path))

    def test_repeated_changed_files_and_callback_lifetime(self):
        path = str(self.root / 'reference.vtp')
        first = tractography(9)
        wma.io.write_polydata(first, path)
        progress = []
        retained = wma.io.read_polydata(path, progress.append, geometry_only=True)
        assert_geometry(self, first, retained)
        self.assertTrue(progress)
        completed = len(progress)
        assert_geometry(self, first, wma.io.read_polydata(path, geometry_only=True))
        self.assertEqual(len(progress), completed)
        changed = tractography(12)
        values = vtk_to_numpy(changed.GetPoints().GetData())
        values += 1000.
        changed.GetPointData().GetArray(0).SetName('new schema')
        wma.io.write_polydata(changed, path)
        assert_geometry(self, changed, wma.io.read_polydata(path, geometry_only=True))
        gc.collect()
        assert_geometry(self, first, retained)
        assert_equal(self, changed, wma.io.read_polydata(path))

    def test_legacy_geometry_read_skips_typed_arrays(self):
        reader = Mock()
        reader.GetOutput.return_value = tractography(3)
        with patch.object(wma.io.vtk, 'vtkPolyDataReader', return_value=reader):
            output = wma.io.read_polydata('reference.vtk', geometry_only=True)
        self.assertEqual(output.GetNumberOfLines(), 3)
        for array_type in ('Scalars', 'Vectors', 'Tensors', 'Normals', 'TCoords'):
            getattr(reader, 'ReadAll' + array_type + 'Off').assert_called_once_with()
            getattr(reader, 'Set' + array_type + 'Name').assert_called_once_with(
                '__wma_geometry_only__')
        reader.ReadAllColorScalarsOff.assert_called_once_with()
        reader.ReadAllFieldsOff.assert_called_once_with()

    def test_concurrent_readers_and_owned_outputs(self):
        paths, data = [], []
        for count in (3, 6, 9, 12):
            item = tractography(count)
            path = str(self.root / ('cluster%d.vtp' % count))
            wma.io.write_polydata(item, path)
            data.append(item)
            paths.append(path)
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda path: wma.io.read_polydata(path, geometry_only=True), paths * 4))
        for original, actual in zip(data * 4, results):
            assert_geometry(self, original, actual)


if __name__ == '__main__':
    unittest.main()
