"""Avoid full geometry copies while retaining double arithmetic and output bytes."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import whitematteranalysis as wma
from test_atlas_blocks import polydata
from test_output_pipeline import baseline_module


class GeometryViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_filter = baseline_module('whitematteranalysis._geometry_filter', 'filter.py', '710f7d4')
        cls.old_fibers = baseline_module('whitematteranalysis._geometry_fibers', 'fibers.py', '710f7d4')

    def test_views_and_detached_mask_output(self):
        data = polydata(11)
        original = vtk_to_numpy(data.GetPoints().GetData()).copy()
        offsets, connectivity, points = wma.filter._lines_offsets_connectivity(data)
        self.assertTrue(np.shares_memory(points, vtk_to_numpy(data.GetPoints().GetData())))
        self.assertTrue(np.shares_memory(offsets, vtk_to_numpy(data.GetLines().GetOffsetsArray())))
        self.assertTrue(np.shares_memory(connectivity, vtk_to_numpy(data.GetLines().GetConnectivityArray())))
        output = wma.filter.mask(data, np.ones(11), verbose=False)
        output.GetPoints().SetPoint(0, 1000., 2000., 3000.)
        np.testing.assert_array_equal(vtk_to_numpy(data.GetPoints().GetData()), original)

    def test_precision_filters_and_resampling_match_baseline(self):
        for dtype in (np.float32, np.float64):
            data = polydata(37)
            values = vtk_to_numpy(data.GetPoints().GetData()).astype(dtype)
            # Float32 addition would round this endpoint mean to exactly -40.
            low = np.nextafter(np.float32(-40.), np.float32(-np.inf))
            values[0, 2] = np.nextafter(low, np.float32(-np.inf))
            values[6, 2] = np.nextafter(np.float32(-40.), np.float32(np.inf))
            data.GetPoints().SetData(numpy_to_vtk(values, deep=True))
            expected = self.old_filter.compute_lengths(data)
            actual = wma.filter.compute_lengths(data)
            np.testing.assert_array_equal(actual[0], expected[0])
            self.assertEqual(actual[1], expected[1])
            distance = np.sqrt(np.sum((values[0].astype(np.float64) - values[6]) ** 2))
            for options in ({}, dict(remove_u=True, remove_u_endpoint_dist=2.),
                            dict(remove_brainstem=True), dict(max_length_mm=8.),
                            dict(remove_u=True, remove_u_endpoint_dist=np.nextafter(distance, -np.inf)),
                            dict(remove_u=True, remove_u_endpoint_dist=np.nextafter(distance, np.inf))):
                expected = self.old_filter.preprocess(data, 0., verbose=False, **options)
                actual = wma.filter.preprocess(data, 0., verbose=False, **options)
                with tempfile.TemporaryDirectory() as directory:
                    a, b = Path(directory)/'a.vtp', Path(directory)/'b.vtp'
                    wma.io.write_polydata(expected, str(a))
                    wma.io.write_polydata(actual, str(b))
                    self.assertEqual(a.read_bytes(), b.read_bytes())
            for points in (2, 3, 10, 20):
                a, b = self.old_fibers.FiberArray(), wma.fibers.FiberArray()
                a.convert_from_polydata(data, points)
                b.convert_from_polydata(data, points)
                for name in ('fiber_array_r', 'fiber_array_a', 'fiber_array_s'):
                    np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
                    self.assertTrue(getattr(b, name).flags.c_contiguous)
            np.testing.assert_array_equal(vtk_to_numpy(data.GetPoints().GetData()), values)

    def test_large_mask_batches_preserve_order_types_and_ownership(self):
        lengths = np.full(4200, 67, dtype=np.int64)
        lengths[::100] = 2
        offsets = np.r_[0, lengths.cumsum()]
        count = int(offsets[-1])
        values = np.arange(count * 3, dtype=np.float64).reshape(count, 3) / 7.
        points = vtk.vtkPoints()
        points.SetData(numpy_to_vtk(values, deep=True))
        lines = vtk.vtkCellArray()
        lines.SetData(numpy_to_vtk(offsets, deep=True),
                      numpy_to_vtk(np.arange(count, dtype=np.int64)[::-1].copy(), deep=True))
        data = vtk.vtkPolyData()
        data.SetPoints(points)
        data.SetLines(lines)
        attribute = numpy_to_vtk(values[:, 0].copy(), deep=True)
        attribute.SetName('scalar')
        data.GetPointData().SetScalars(attribute)
        colors = np.arange(4200 * 3).reshape(4200, 3) % 256
        mask = np.ones(4200)
        expected = self.old_filter.mask(data, mask, colors, preserve_point_data=True, verbose=False)
        actual = wma.filter.mask(data, mask, colors, preserve_point_data=True, verbose=False)
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / 'expected.vtp', Path(directory) / 'actual.vtp'
            wma.io.write_polydata(expected, str(a))
            wma.io.write_polydata(actual, str(b))
            self.assertEqual(a.read_bytes(), b.read_bytes())
        actual.GetPointData().GetArray('scalar').SetTuple1(0, -100.)
        np.testing.assert_array_equal(vtk_to_numpy(data.GetPointData().GetArray('scalar')), values[:, 0])

        # Empty connectivity ranges need no VTK cell construction.
        offsets = np.array([0, 0, 3, 3, 5], dtype=np.int64)
        connectivity = np.array([7, 4, 7, 2, 9], dtype=np.int64)
        selected = np.array([2, 3, 0, 1], dtype=np.int64)
        expected = self.old_filter._gather_conn(connectivity, offsets, selected)
        actual = wma.filter._gather_conn(connectivity, offsets, selected)
        for a, b in zip(expected, actual):
            np.testing.assert_array_equal(a, b)


if __name__ == '__main__':
    unittest.main()
