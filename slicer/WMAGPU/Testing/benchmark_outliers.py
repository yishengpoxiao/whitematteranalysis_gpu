"""Benchmark real atlas clusters with a small derived subject, not a clinical run."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Modules/_vendor'))
import numpy as np
import vtk
import whitematteranalysis as wma
import WMAtlasLabelGPUCore as core

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('atlas')
parser.add_argument('--clusters', type=int, default=32)
parser.add_argument('--workers', type=int, nargs='+', default=[1, 4, 16, 0])
parser.add_argument('--backends', nargs='+', default=['cpu', 'gpu'])
args = parser.parse_args()
directory = core.resolve_atlas_directory(args.atlas)
append = vtk.vtkAppendPolyData()
labels = []
for c in range(args.clusters):
    atlas = wma.io.read_polydata(str(Path(directory) / ('cluster_%05d.vtp' % (c + 1))))
    n = min(8, atlas.GetNumberOfLines())
    if not n:
        continue
    select = np.zeros(atlas.GetNumberOfLines())
    select[:n] = 1
    pd = wma.filter.mask(atlas, select, verbose=False)
    # Last sampled fiber is deliberately far from the atlas.
    cell = pd.GetCell(n - 1)
    for j in range(cell.GetNumberOfPoints()):
        idx = cell.GetPointId(j)
        xyz = pd.GetPoint(idx)
        pd.GetPoints().SetPoint(idx, xyz[0], xyz[1] + 1000., xyz[2])
    append.AddInputData(pd)
    labels.extend([c] * n)
append.Update()
subject = append.GetOutput()
labels = np.asarray(labels)
reference = None
for backend in args.backends:
    if backend == 'gpu' and not wma.gpu.available():
        continue
    # Warm up kernels and file caches outside the timings.
    core.remove_cluster_outliers(subject, labels, args.clusters, directory,
                                 backend=backend, max_workers=1)
    for workers in args.workers:
        messages = []
        if backend == 'gpu':
            import torch
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        mask, rejected = core.remove_cluster_outliers(subject, labels, args.clusters, directory,
            backend=backend, max_workers=workers,
            progress_callback=lambda text, pct: messages.append(text))
        elapsed = time.perf_counter() - start
        if reference is None:
            reference = mask
        np.testing.assert_array_equal(mask, reference)
        result = dict(backend=backend, requested_workers=workers, plan=messages[0],
                      clusters=args.clusters, subject_fibers=len(labels),
                      seconds=elapsed, rejected=rejected, mask_matches=True)
        if backend == 'gpu':
            result['peak_allocated_mib'] = torch.cuda.max_memory_allocated() / 2**20
        print(json.dumps(result), flush=True)
