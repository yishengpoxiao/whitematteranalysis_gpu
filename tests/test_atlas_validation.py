"""Reject ambiguous atlas settings and mismatched cluster references."""
import contextlib
import importlib.util
import io
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import whitematteranalysis as wma
from test_cli_file_stages import tractography


class AtlasValidationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_bilateral_is_a_boolean_without_implicit_defaults(self):
        atlas = wma.cluster.ClusterAtlas()
        atlas.nystrom_polydata = tractography(4)
        atlas.pinv_A = np.eye(4)
        atlas.number_of_eigenvectors = 2
        for bilateral in (False, True, np.bool_(False)):
            atlas.bilateral = bilateral
            atlas.save(str(self.root), 'atlas')
            loaded = wma.cluster.load_atlas(str(self.root), 'atlas')
            self.assertEqual(loaded.bilateral, bilateral)
        atlas.nystrom_polydata = None
        for value in ('False', 'True', None, 'missing'):
            if value == 'missing':
                del atlas.bilateral
            else:
                atlas.bilateral = value
            with (self.root / 'atlas.p').open('wb') as output:
                pickle.dump(atlas, output)
            with self.assertRaisesRegex(ValueError, 'Boolean bilateral'):
                wma.cluster.load_atlas(str(self.root), 'atlas')

    def test_cluster_names_are_checked_before_output_or_atlas_reads(self):
        path = Path(__file__).resolve().parents[1] / 'bin/wm_cluster_remove_outliers.py'
        spec = importlib.util.spec_from_file_location('outlier_cli_validation', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        atlas, subject, output = (self.root / name for name in ('atlas', 'subject', 'output'))
        atlas.mkdir()
        subject.mkdir()
        arguments = [str(path), str(subject), str(atlas), str(output), '-gpu', 'off']
        cases = (((), (), 'No atlas'),
                 (('00001', '00002'), ('00001', '00003'), 'filename mismatch'),
                 (('00001', '00002'), ('00001',), 'filename mismatch'))
        for reference, queries, message in cases:
            for directory, names in ((atlas, reference), (subject, queries)):
                for old in directory.glob('*.vtp'):
                    old.unlink()
                for name in names:
                    (directory / ('cluster_' + name + '.vtp')).touch()
            errors = io.StringIO()
            with patch.object(sys, 'argv', arguments), \
                    patch.object(wma.gpu, 'apply_gpu_arguments'), \
                    patch.object(wma.cluster, 'load_atlas') as loader, \
                    contextlib.redirect_stderr(errors), self.assertRaises(SystemExit) as raised:
                module.main()
            self.assertEqual(raised.exception.code, 2)
            self.assertIn(message, errors.getvalue())
            self.assertFalse(output.exists())
            loader.assert_not_called()


if __name__ == '__main__':
    unittest.main()
