"""Parallel outlier masks must agree with the original per-fiber algorithm."""
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Modules/_vendor'))
import numpy as np
import vtk
import whitematteranalysis as wma
import WMAtlasLabelGPUCore as core


def fibers(offsets):
    pd = vtk.vtkPolyData()
    points, lines = vtk.vtkPoints(), vtk.vtkCellArray()
    for offset in offsets:
        lines.InsertNextCell(10)
        for j in range(10):
            lines.InsertCellPoint(points.InsertNextPoint(offset, j * 8., np.sin(j)))
    pd.SetPoints(points)
    pd.SetLines(lines)
    return pd


class OutlierParallelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = self.temp.name
        cache = patch.dict('os.environ', WMA_ATLAS_CACHE=str(Path(self.directory) / 'cache'))
        cache.start()
        self.addCleanup(cache.stop)
        for c in range(3):
            wma.io.write_polydata(fibers(np.arange(7) + c * 20.),
                                 str(Path(self.directory) / ('cluster_%05d.vtp' % (c + 1))))
        wma.io.write_polydata(fibers([]), str(Path(self.directory) / 'cluster_00005.vtp'))
        # Interleaved labels catch incorrect ordering when futures complete out of order.
        self.labels = np.tile(np.arange(5), 5)
        self.subject = fibers([c * 20. + (100. if i >= 15 else i // 5)
                               for i, c in enumerate(self.labels)])
        self.addCleanup(core.resolve_backend, 'auto')

    def run_outliers(self, backend, workers, **kwargs):
        return core.remove_cluster_outliers(self.subject, self.labels, 6, self.directory,
                                             backend=backend, max_workers=workers, **kwargs)

    def test_cpu_parallel_matches_original(self):
        reference, rejected = self.run_outliers('original', 1)
        self.assertGreater(rejected, 0)
        for workers in (1, 4, 0):
            messages = []
            mask, count = self.run_outliers('cpu', workers,
                progress_callback=lambda text, pct: messages.append(text))
            np.testing.assert_array_equal(mask, reference)
            self.assertEqual(count, rejected)
            self.assertIn('atlas defines 6 clusters; subject has 5 nonempty, 1 empty', messages[0])
            self.assertEqual(messages[1], 'Atlas reference cluster files: 4/6')
            self.assertTrue(any('cluster 4: missing atlas reference' in text for text in messages))
            self.assertTrue(any('cluster 5: empty atlas reference' in text for text in messages))
            self.assertIn('5/5 nonempty subject clusters done (atlas total 6)', messages[-1])
        # Missing and empty reference clusters retain their subject fibers.
        np.testing.assert_array_equal(reference[np.isin(self.labels, [3, 4])], 1)

    @unittest.skipUnless(wma.gpu.available(), 'CUDA is unavailable')
    def test_gpu_streams_match_original(self):
        import torch
        reference, rejected = self.run_outliers('original', 1)
        streams = set()
        real_matrix = wma.gpu.distance.distance_matrix_from_arrays

        def matrix(*args, **kwargs):
            streams.add(torch.cuda.current_stream().cuda_stream)
            return real_matrix(*args, **kwargs)

        with patch.object(wma.gpu.distance, 'distance_matrix_from_arrays', side_effect=matrix):
            for workers in (1, 4, 0):
                mask, count = self.run_outliers('gpu', workers)
                np.testing.assert_array_equal(mask, reference)
                self.assertEqual(count, rejected)
        self.assertGreater(len(streams), 1)
        messages = []
        with patch.object(torch.cuda, 'mem_get_info', return_value=(512 << 20, 1024 << 20)):
            mask, count = self.run_outliers('gpu', 800,
                progress_callback=lambda text, pct: messages.append(text))
        self.assertIn('1 parallel workers', messages[0])
        np.testing.assert_array_equal(mask, reference)
        self.assertEqual(count, rejected)

    def test_workers_overlap_and_callbacks_stay_on_caller(self):
        caller = threading.get_ident()
        callback_threads, event_threads = [], []
        active = 0
        peak = 0
        lock = threading.Lock()
        real_read = wma.io.read_polydata

        def read(path, **kwargs):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                time.sleep(.05)
                return real_read(path, **kwargs)
            finally:
                with lock:
                    active -= 1

        with patch.object(wma.io, 'read_polydata', side_effect=read):
            self.run_outliers('cpu', 4,
                progress_callback=lambda *args: callback_threads.append(threading.get_ident()),
                event_callback=lambda: event_threads.append(threading.get_ident()))
        self.assertGreater(peak, 1)
        self.assertEqual(set(callback_threads), {caller})
        self.assertEqual(set(event_threads), {caller})

    def test_empty_and_worker_failure(self):
        mask, count = core.remove_cluster_outliers(fibers([]), np.array([], dtype=int),
            6, self.directory, backend='cpu')
        self.assertEqual(len(mask), 0)
        self.assertEqual(count, 0)
        before = wma.gpu.config.cpu_threads
        with patch.object(wma.io, 'read_polydata', side_effect=RuntimeError('test read failure')):
            with self.assertRaisesRegex(RuntimeError, 'test read failure'):
                self.run_outliers('cpu', 4)
        self.assertEqual(wma.gpu.config.cpu_threads, before)


if __name__ == '__main__':
    unittest.main()
