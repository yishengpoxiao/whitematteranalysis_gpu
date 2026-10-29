"""Cluster ordering, metadata and file bytes survive vectorization/process writing."""
from pathlib import Path
import multiprocessing
import subprocess
import linecache
import sys
import tempfile
import types
import unittest
from threading import Barrier
from unittest.mock import patch

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk
import whitematteranalysis as wma


def baseline_module(name, path, revision="7a5e5b8"):
    module = types.ModuleType(name)
    module.__package__ = name.rsplit(".", 1)[0]
    module.__file__ = path
    source = subprocess.check_output(['git', 'show', revision + ':whitematteranalysis/' + path], text=True)
    sys.modules[name] = module
    filename = '<baseline ' + revision + ' ' + path + '>'
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    exec(compile(source, filename, 'exec'), module.__dict__)
    return module


def fixture():
    pd = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(np.random.RandomState(42).normal(size=(12, 3)), deep=True))
    lines = vtk.vtkCellArray()
    for ids in ([0, 2, 5], [5, 3], [4, 10], [9, 8, 2, 1], [7, 6]):
        line = vtk.vtkIdList()
        for i in ids:
            line.InsertNextId(i)
        lines.InsertNextCell(line)
    pd.SetPoints(points)
    pd.SetLines(lines)
    for attributes, count in ((pd.GetPointData(), 12), (pd.GetCellData(), 5)):
        for name, dtype, components in (('scalar', np.float64, 1), ('rgb', np.uint8, 3),
                                         ('integer', np.int32, 1), ('tensors', np.float32, 9)):
            values = np.arange(count*components, dtype=dtype).reshape(count, components)
            array = numpy_to_vtk(values, deep=True)
            array.SetName(name)
            attributes.AddArray(array)
        attributes.SetActiveScalars('scalar')
        attributes.SetActiveTensors('tensors')
    text = vtk.vtkStringArray()
    text.SetName('description')
    text.InsertNextValue('preserve field data')
    pd.GetFieldData().AddArray(text)
    return pd


class OutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = baseline_module('whitematteranalysis._oldcluster', 'cluster.py')

    def test_split_matches_original_bytes(self):
        labels = np.array([2, -1, 0, 2, 1])
        for point_data in (False, True):
            for cell_data in (False, True):
                with self.subTest(point_data=point_data, cell_data=cell_data):
                    pd = fixture()
                    expected = self.original.mask_all_clusters(
                        pd, labels, 4, preserve_point_data=point_data,
                        preserve_cell_data=cell_data, verbose=False)
                    actual = wma.cluster.mask_all_clusters(
                        pd, labels, 4, preserve_point_data=point_data,
                        preserve_cell_data=cell_data, verbose=False)
                    with tempfile.TemporaryDirectory() as directory:
                        for i, (old, new) in enumerate(zip(expected, actual)):
                            a, b = Path(directory)/'old.vtp', Path(directory)/'new.vtp'
                            wma.io.write_polydata(old, str(a))
                            wma.io.write_polydata(new, str(b))
                            self.assertEqual(a.read_bytes(), b.read_bytes(), 'cluster %d' % i)

    def test_parallel_writes_preserve_serial_files(self):
        polydatas = [fixture() for _ in range(8)]
        for extension in ('.vtk', '.vtp'):
            with self.subTest(extension=extension), tempfile.TemporaryDirectory() as directory:
                serial = [str(Path(directory)/('serial%d%s' % (i, extension))) for i in range(8)]
                parallel = [str(Path(directory)/('parallel%d%s' % (i, extension))) for i in range(8)]
                for pd, path in zip(polydatas, serial):
                    wma.io.write_polydata(pd, path)
                wma.io.write_polydatas_parallel(polydatas, parallel, jobs=3)
                for a, b in zip(serial, parallel):
                    self.assertEqual(Path(a).read_bytes(), Path(b).read_bytes())

    def test_parallel_writes_without_fork_preserve_bytes_and_overlap(self):
        # Exercise the Windows branch on any host, with real concurrent VTK
        # writers. The barrier would fail if the fallback became serial.
        for extension in ('.vtk', '.vtp'):
            with self.subTest(extension=extension), tempfile.TemporaryDirectory() as directory:
                polydatas = [fixture(), fixture()]
                serial = [str(Path(directory)/('serial%d%s' % (i, extension))) for i in range(2)]
                parallel = [str(Path(directory)/('parallel%d%s' % (i, extension))) for i in range(2)]
                write = wma.io.write_polydata
                for pd, filename in zip(polydatas, serial):
                    write(pd, filename)
                barrier = Barrier(2)

                def concurrent_write(pd, filename):
                    barrier.wait(timeout=10)
                    write(pd, filename)

                with patch.object(multiprocessing, 'get_all_start_methods', return_value=['spawn']), \
                        patch.object(multiprocessing, 'get_context') as context, \
                        patch.object(wma.io, 'write_polydata', side_effect=concurrent_write):
                    wma.io.write_polydatas_parallel(polydatas, parallel, jobs=2)
                context.assert_not_called()
                for a, b in zip(serial, parallel):
                    self.assertEqual(Path(a).read_bytes(), Path(b).read_bytes())

    def test_thread_writer_propagates_failure(self):
        with patch.object(multiprocessing, 'get_all_start_methods', return_value=['spawn']), \
                patch.object(wma.io, 'write_polydata', side_effect=IOError('write failed')):
            with self.assertRaisesRegex(IOError, 'write failed'):
                wma.io.write_polydatas_parallel([fixture(), fixture()], ['a.vtp', 'b.vtp'], jobs=2)

    def test_colors_match_colorsys(self):
        rng = np.random.RandomState(2026)
        embed = rng.normal(size=(10000, 10))
        embed[:9, :3] = [[0, 0, 0], [1, 1, 1], [-1, -1, -1],
                         [1, 0, 0], [0, 1, 0], [0, 0, 1],
                         [1, 1, 0], [1, 0, 1], [0, 1, 1]]
        embed[9, 0] = np.nan
        np.testing.assert_array_equal(wma.cluster._embed_to_rgb(embed.copy()),
                                      self.original._embed_to_rgb(embed.copy()))
        np.testing.assert_array_equal(wma.cluster._embed_to_rgb(embed[:0].copy()),
                                      self.original._embed_to_rgb(embed[:0].copy()))

    def test_invalid_cluster_is_reported(self):
        with self.assertRaises(IndexError):
            wma.cluster.mask_all_clusters(fixture(), [0, 1, 2, 3, 4], 4, verbose=False)
