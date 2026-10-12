"""Atlas path resolution tests; no atlas computation or Slicer required."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Modules/_vendor'))
import WMAtlasLabelGPUCore as core


class AtlasPathTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def file(self, relative):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path

    def test_org_parent_and_files(self):
        atlas = self.file('ORG-800FC-100HCP/atlas.p').parent
        self.file('ORG-800FC-100HCP/atlas.vtp')
        cluster = self.file('ORG-800FC-100HCP/cluster_00001.vtp')
        mrml = self.file('ORG-800FC-100HCP/T_AF.mrml')
        registration = self.file('ORG-RegAtlas-100HCP/registration_atlas.vtk')
        self.file('Clusters/cluster_00001.vtp')
        self.file('Anatomical/T_wrong.mrml')
        for selection in (self.root, atlas, atlas / 'atlas.p'):
            self.assertEqual(core.resolve_atlas_directory(selection), str(atlas))
            self.assertEqual(core.find_registration_atlas(selection), str(registration))
            self.assertEqual(core.find_atlas_cluster_directory(selection), str(cluster.parent))
            self.assertEqual(core.find_anatomical_tracts_mrml(selection), [str(mrml)])
        self.assertEqual(core.find_registration_atlas(registration), str(registration))
        with patch.dict(os.environ, WMA_TEST_ATLAS=str(self.root)):
            self.assertEqual(core.resolve_atlas_directory('$WMA_TEST_ATLAS'), str(atlas))

    def test_build_root_latest_iteration_and_explicit_iteration(self):
        first = self.file('iteration_2/initial_clusters/atlas.p').parent
        latest = self.file('iteration_10/initial_clusters/atlas.p').parent
        self.file('iteration_11/incomplete.txt')
        tractography = self.file('iteration_10/initial_clusters/atlas.vtp')
        self.assertEqual(core.resolve_atlas_directory(self.root), str(latest))
        self.assertEqual(core.resolve_atlas_directory(first.parent), str(first))
        self.assertEqual(core.find_registration_atlas(self.root), str(tractography))
        with patch.object(core, '_atlas_path', return_value=str(self.root)):
            self.assertEqual(core.resolve_atlas_directory('~/atlas'), str(latest))

    def test_wrapper_direct_priority_and_ambiguity(self):
        atlas = self.file('wrapper/custom/atlas.p').parent
        self.assertEqual(core.resolve_atlas_directory(self.root), str(atlas))
        self.file('other/atlas.p')
        with self.assertRaisesRegex(IOError, 'Multiple atlas matches'):
            core.resolve_atlas_directory(self.root)
        self.file('atlas.p')
        self.assertEqual(core.resolve_atlas_directory(self.root), str(self.root))

    def test_missing_atlas(self):
        with self.assertRaisesRegex(IOError, 'atlas.p not found'):
            core.resolve_atlas_directory(self.root)
        with self.assertRaisesRegex(IOError, 'does not exist'):
            core.resolve_atlas_directory(self.root / 'missing')


if __name__ == '__main__':
    unittest.main()
