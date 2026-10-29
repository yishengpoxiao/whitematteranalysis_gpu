"""Atlas caching preserves probabilities and invalidates changed references."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import whitematteranalysis as wma
from whitematteranalysis.gpu import atlas_cache
from whitematteranalysis.gpu.config import float32_enabled
from test_cli_file_stages import tractography


class AtlasCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'atlas.vtp'
        self.cache = self.root / 'cache'
        environment = patch.dict(os.environ, WMA_ATLAS_CACHE=str(self.cache))
        environment.start()
        self.addCleanup(environment.stop)
        self.data = tractography(80)
        wma.io.write_polydata(self.data, str(self.source))

    def check_reference(self, backend, method='Mean', bilateral=True, sigmasq=400.):
        load = lambda: atlas_cache.load_reference(self.source, sigmasq, method, bilateral, backend)
        arrays, probabilities = load()
        distances = wma.cluster._pairwise_distance_matrix(self.data, 0., number_of_jobs=1,
            distance_method=method, bilateral=bilateral, sigmasq=sigmasq, use_gpu=backend == 'gpu')
        expected = (np.exp(-distances / sigmasq).sum(axis=1) - 1.) / len(arrays)
        if backend == 'gpu':
            tolerance = 2e-6 if float32_enabled() else 2e-14
            np.testing.assert_allclose(probabilities, expected, rtol=tolerance,
                                       atol=tolerance * 0.1)
        else:
            np.testing.assert_array_equal(probabilities, expected)
        with patch.object(wma.io, 'read_polydata', side_effect=AssertionError('cache miss')):
            cached_arrays, cached_probabilities = load()
        np.testing.assert_array_equal(cached_arrays, arrays)
        np.testing.assert_array_equal(cached_probabilities, probabilities)

    def test_cpu_probabilities_are_exact(self):
        for method in ('Mean', 'MeanSquared', 'Hausdorff', 'StrictSimilarity'):
            for bilateral in (False, True):
                self.check_reference('cpu', method, bilateral)

    @unittest.skipUnless(wma.gpu.available(), 'CUDA is unavailable')
    def test_gpu_probabilities_and_precision_are_exact(self):
        for float32 in (False, True):
            with patch.object(wma.gpu.config, 'float32', float32):
                for method in ('Mean', 'MeanSquared', 'Hausdorff', 'StrictSimilarity'):
                    self.check_reference('gpu', method)
        self.assertEqual(len(list(self.cache.glob('*.npz'))), 8)

    def test_parameters_reference_and_implementation_invalidate(self):
        load = lambda **kwargs: atlas_cache.load_reference(self.source, backend='cpu', **kwargs)
        load(sigmasq=400.)
        read = wma.io.read_polydata
        with patch.object(wma.io, 'read_polydata', wraps=read) as reader:
            load(sigmasq=401.)
            load(sigmasq=400., bilateral=False)
            load(sigmasq=400., method='MeanSquared')
            stat = self.source.stat()
            os.utime(str(self.source), ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
            load(sigmasq=400.)
            with patch.object(atlas_cache, '_implementation', return_value=['changed implementation']):
                load(sigmasq=400.)
            self.assertEqual(reader.call_count, 5)
        self.assertEqual(len(list(self.cache.glob('*.npz'))), 6)

    def test_corrupt_and_unavailable_cache_recompute(self):
        load = lambda: atlas_cache.load_reference(self.source, 400., backend='cpu')
        expected = load()
        path, = self.cache.glob('*.npz')
        for damaged in (b'corrupt file', b'PK\x03\x04'):
            path.write_bytes(damaged)
            for actual, reference in zip(load(), expected):
                np.testing.assert_array_equal(actual, reference)
        np.savez(str(path), coordinates=np.zeros((3, 4)), probabilities=np.zeros(3))
        np.testing.assert_array_equal(load()[1], expected[1])
        with path.open('wb') as output:
            np.save(output, np.zeros(3))
        np.testing.assert_array_equal(load()[1], expected[1])
        blocked = self.root / 'not a directory'
        blocked.write_text('occupied')
        with patch.dict(os.environ, WMA_ATLAS_CACHE=str(blocked / 'cache')):
            np.testing.assert_array_equal(load()[1], expected[1])
        with patch.dict(os.environ, WMA_ATLAS_CACHE='off'), patch.object(
                wma.io, 'read_polydata', wraps=wma.io.read_polydata) as reader:
            load()
            load()
            self.assertEqual(reader.call_count, 2)

    def test_empty_and_concurrent_writers(self):
        wma.io.write_polydata(tractography(0), str(self.source))
        load = lambda _: atlas_cache.load_reference(self.source, 400., backend='cpu')
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(load, range(8)))
        for arrays, probabilities in results:
            self.assertEqual(arrays.shape, (0, 15, 3))
            self.assertEqual(probabilities.shape, (0,))
        self.assertEqual(len(list(self.cache.iterdir())), 1)
        with patch.object(wma.io, 'read_polydata', side_effect=AssertionError('cache miss')):
            load(0)


if __name__ == '__main__':
    unittest.main()
