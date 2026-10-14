"""Anatomical selection parity, bounded atlas reuse, and CUDA allocator cleanup."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk

sys.path.append(str(Path(__file__).resolve().parents[1] / 'Modules/_vendor'))
import whitematteranalysis as wma
import WMAtlasLabelGPUCore as core
from WMAGPUControls import release_unused_memory


def anatomical_fixture():
    pd = vtk.vtkPolyData()
    points, lines = vtk.vtkPoints(), vtk.vtkCellArray()
    points.SetDataTypeToDouble()
    for fiber in range(36):
        count = (2, 3, 5, 20, 21, 31)[fiber % 6]
        lines.InsertNextCell(count)
        for point in range(count):
            x = -10. if fiber % 3 == 0 else (10. if fiber % 3 == 1 else (-10. if point < count / 2 else 10.))
            if fiber == 19:
                x = -1e-46  # float32 rounding must precede hemisphere classification.
            lines.InsertCellPoint(points.InsertNextPoint(x, point, fiber))
    pd.SetPoints(points)
    pd.SetLines(lines)
    for attributes, count in ((pd.GetPointData(), pd.GetNumberOfPoints()), (pd.GetCellData(), 36)):
        for name, dtype, components in (('scalar', np.float64, 1), ('rgb', np.uint8, 3), ('tensors', np.float32, 9)):
            array = numpy_to_vtk(np.arange(count * components, dtype=dtype).reshape(count, components), deep=True)
            array.SetName(name)
            attributes.AddArray(array)
        attributes.SetActiveTensors('tensors')
    return pd, np.array([2, 0, 1, 3, 2, 0] * 6)


class LoadingTests(unittest.TestCase):
    def test_anatomical_output_matches_separate_and_append(self):
        pd, labels = anatomical_fixture()
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for name, members in (('T_AF', [4, 1, 3, 3, 9]), ('T_CC1', [1, 2, 3]), ('T_MCP', [2, 4]), ('T_empty', [99])):
                path = Path(directory) / (name + '.mrml')
                path.write_text(''.join('<FiberBundleStorage fileName="cluster_%05d.vtp"/>\n' % n for n in members))
                paths.append(str(path))
            clusters = core.extract_clusters(pd, labels, 5)
            separated = core.separate_clusters_by_hemisphere(clusters)
            expected = core.append_clusters_to_anatomical_tracts(clusters, paths, separated)
            actual = core.extract_anatomical_tracts(pd, labels, paths)
            self.assertEqual([name for name, _ in expected], [name for name, _ in actual])
            for (_, old), (_, new) in zip(expected, actual):
                a, b = Path(directory) / 'before.vtp', Path(directory) / 'after.vtp'
                wma.io.write_polydata(old, str(a))
                wma.io.write_polydata(new, str(b))
                self.assertEqual(a.read_bytes(), b.read_bytes())
            with patch.object(core, 'assess_hemisphere_location', side_effect=AssertionError('repeated resampling')):
                self.assertTrue(core.extract_anatomical_tracts(pd, labels, paths))

    def test_atlas_cache_invalidates_and_keeps_one_entry(self):
        core._ATLAS_CACHE = None
        with tempfile.TemporaryDirectory() as directory, patch.object(wma.cluster, 'load_atlas', side_effect=lambda *a: object()) as load:
            root = Path(directory)
            for name in ('a', 'b'):
                (root / name).mkdir()
                (root / name / 'atlas.p').touch()
                (root / name / 'atlas.vtp').touch()
            first = core.load_atlas(str(root / 'a'))
            self.assertIs(core.load_atlas(str(root / 'a')), first)
            self.assertEqual(load.call_count, 1)
            (root / 'a/atlas.vtp').write_bytes(b'changed')
            self.assertIsNot(core.load_atlas(str(root / 'a')), first)
            core.load_atlas(str(root / 'b'))
            core.load_atlas(str(root / 'a'))
            self.assertEqual(load.call_count, 4)
        core._ATLAS_CACHE = None

    @unittest.skipUnless(wma.gpu.available(), 'CUDA is unavailable')
    def test_release_keeps_live_tensors_and_clears_inactive_cache(self):
        import torch
        device = 0
        baseline = release_unused_memory(device)
        live = torch.ones(1024, device='cuda:0')
        temporary = torch.ones(8 << 20, device='cuda:0')
        del temporary
        before = torch.cuda.memory_reserved(device)
        report = release_unused_memory(device)
        self.assertGreater(report['released'], 0)
        self.assertLess(report['reserved'], before)
        self.assertEqual(live.sum().item(), 1024)
        self.assertGreater(report['allocated'], 0)
        del live
        report = release_unused_memory(device)
        # Prior GEMMs may retain a cuBLAS workspace owned by PyTorch.
        self.assertEqual(report['allocated'], baseline['allocated'])
        self.assertEqual(report['reserved'], baseline['reserved'])


if __name__ == '__main__':
    unittest.main()
