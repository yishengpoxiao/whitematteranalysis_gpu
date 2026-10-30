"""CPU fallback outlier decisions depend only on atlas geometry."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import vtk
from vtk.util.numpy_support import numpy_to_vtk
import whitematteranalysis as wma
from test_cli_file_stages import tractography

VENDOR = Path(__file__).resolve().parents[1] / 'slicer/WMAGPU/Modules/_vendor'
sys.path.insert(0, str(VENDOR))
# The Slicer-bundled core is the production implementation under test.
from WMAtlasLabelGPUCore import remove_cluster_outliers


class GeometryOnlyCpuFallbackTests(unittest.TestCase):
    def test_full_and_geometry_only_atlas_reads_give_same_membership(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            atlas_dir = root / 'atlas'
            atlas_dir.mkdir()
            atlas = tractography(9)
            array = numpy_to_vtk([float(i) for i in range(9 * 7)], deep=True)
            array.SetName('atlas_attribute')
            atlas.GetPointData().AddArray(array)
            metadata = vtk.vtkStringArray()
            metadata.SetName('large_metadata')
            metadata.InsertNextValue('unused by distances')
            atlas.GetFieldData().AddArray(metadata)
            atlas_path = atlas_dir / 'cluster_00001.vtp'
            wma.io.write_polydata(atlas, str(atlas_path))
            subject = tractography(9)
            for point_id in range(8 * 7, 9 * 7):
                point = subject.GetPoint(point_id)
                subject.GetPoints().SetPoint(point_id, *(value + 1000. for value in point))
            subject.GetPoints().Modified()
            labels = [0] * subject.GetNumberOfLines()
            read_polydata = wma.io.read_polydata

            def run(geometry_only):
                calls = []

                def read(path, **kwargs):
                    calls.append(kwargs.get('geometry_only', False))
                    atlas_data = read_polydata(path, geometry_only=geometry_only)
                    if geometry_only:
                        self.assertEqual(atlas_data.GetPointData().GetNumberOfArrays(), 0)
                        self.assertEqual(atlas_data.GetFieldData().GetNumberOfArrays(), 0)
                    else:
                        self.assertGreater(atlas_data.GetPointData().GetNumberOfArrays(), 0)
                        self.assertGreater(atlas_data.GetFieldData().GetNumberOfArrays(), 0)
                    return atlas_data

                with patch.object(wma.gpu, 'use_gpu', return_value=False), \
                        patch.object(wma.gpu.config, 'cpu_fast_enabled', return_value=False), \
                        patch.object(wma.io, 'read_polydata', side_effect=read):
                    result = remove_cluster_outliers(
                        subject, labels, 1, str(atlas_dir), backend='cpu',
                        max_workers=1, std_threshold=1.5)
                self.assertEqual(calls, [True])
                return result

            expected = run(False)
            actual = run(True)
            self.assertGreater(sum(expected[0]), 0)
            self.assertLess(sum(expected[0]), len(labels))
            self.assertEqual(expected[1], actual[1])
            self.assertEqual(expected[0].tolist(), actual[0].tolist())


if __name__ == '__main__':
    unittest.main()
