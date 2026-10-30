"""Geometry-only reference reads preserve topology and independent ownership."""
from concurrent.futures import ThreadPoolExecutor
import gc
import os
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

    def test_prefetch_preserves_full_payload_and_chunks_advice(self):
        if not hasattr(os, 'POSIX_FADV_WILLNEED'):
            self.skipTest('POSIX_FADV_WILLNEED is unavailable')
        chunk = 1024 * 1024
        advised_size = 64 * chunk + chunk + 17
        for extension in ('.vtk', '.vtp'):
            with self.subTest(extension=extension):
                path = str(self.root / ('prefetch' + extension))
                data = tractography(9)
                wma.io.write_polydata(data, path)
                expected = self.original.read_polydata(path)
                advice = Mock()
                with patch.dict(os.environ, {'WMA_IO_PREFETCH': '1'}), \
                     patch.object(wma.io.os.path, 'getsize', return_value=advised_size), \
                     patch.object(wma.io.os, 'posix_fadvise', advice, create=True):
                    actual = wma.io.read_polydata(path)
                assert_equal(self, expected, actual)
                self.assertEqual(advice.call_count, 66)
                self.assertEqual(advice.call_args_list[0].args[1:],
                                 (0, chunk, os.POSIX_FADV_WILLNEED))
                self.assertEqual(advice.call_args_list[-1].args[1:],
                                 (65 * chunk, 17, os.POSIX_FADV_WILLNEED))
                with self.assertRaises(OSError):
                    os.fstat(advice.call_args.args[0])
                # Geometry-only reads skip attributes and must not prefetch them.
                with patch.object(wma.io, '_prefetch_file') as prefetch:
                    geometry = wma.io.read_polydata(path, geometry_only=True)
                prefetch.assert_not_called()
                assert_geometry(self, expected, geometry)

    def test_prefetch_skips_small_or_disabled_files(self):
        advice = Mock()
        getsize = Mock(return_value=64 * 1024 * 1024)
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(wma.io.os, 'posix_fadvise', advice, create=True), \
             patch.object(wma.io.os.path, 'getsize', getsize):
            wma.io._prefetch_file('default.vtk')
        getsize.assert_not_called()
        advice.assert_not_called()

        with patch.dict(os.environ, {'WMA_IO_PREFETCH': '1'}), \
             patch.object(wma.io.os, 'posix_fadvise', advice, create=True), \
             patch.object(wma.io.os.path, 'getsize', return_value=64 * 1024 * 1024 - 1):
            wma.io._prefetch_file('small.vtk')
        with patch.dict(os.environ, {'WMA_IO_PREFETCH': '0'}), \
             patch.object(wma.io.os, 'posix_fadvise', advice, create=True), \
             patch.object(wma.io.os.path, 'getsize', return_value=64 * 1024 * 1024):
            wma.io._prefetch_file('disabled.vtk')
        advice.assert_not_called()

    def test_prefetch_unavailable_or_failing_falls_back_to_vtk_read(self):
        path = str(self.root / 'fallback.vtk')
        data = tractography(9)
        wma.io.write_polydata(data, path)
        expected = self.original.read_polydata(path)
        with patch.dict(os.environ, {'WMA_IO_PREFETCH': '1'}), \
             patch.object(wma.io.os.path, 'getsize', return_value=64 * 1024 * 1024):
            with patch.object(wma.io.os, 'POSIX_FADV_WILLNEED', 0, create=True), \
                 patch.object(wma.io.os, 'posix_fadvise', create=True) as advice:
                advice.side_effect = OSError('advice unsupported')
                actual = wma.io.read_polydata(path)
                assert_equal(self, expected, actual)
                with self.assertRaises(OSError):
                    os.fstat(advice.call_args.args[0])
            if hasattr(os, 'posix_fadvise'):
                with patch.object(wma.io.os, 'posix_fadvise'):
                    del os.posix_fadvise
                    actual = wma.io.read_polydata(path)
                    assert_equal(self, expected, actual)
            else:
                actual = wma.io.read_polydata(path)
                assert_equal(self, expected, actual)

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
