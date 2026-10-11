"""Real-data benchmark for B-spline, atlas streaming, cluster splitting and I/O.

Run from the repository: CUDA_VISIBLE_DEVICES=0 python tests/bench_remaining_pipeline.py
Use --full to additionally label every input fiber with the streamed path.
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import statistics
import tempfile
import time

import numpy as np
import torch
import whitematteranalysis as wma
from whitematteranalysis.gpu import bspline, distance, linalg
from test_bspline import reference
from test_output_pipeline import baseline_module


def timed(function, repeats=3):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return result, statistics.median(times)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='/data/yijie/TractData/val_dsi_studio/250932.vtk')
    parser.add_argument('--atlas', default='/data/software/ORG-Atlases/ORG-800FC-100HCP')
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()
    pd = wma.io.read_polydata(args.input)
    sample = wma.filter.downsample(pd, 10000, random_seed=42, preserve_point_data=True, verbose=False)
    fibers = wma.fibers.FiberArray()
    fibers.points_per_fiber = 10
    fibers.convert_from_polydata(sample)
    moving = np.array([fibers.fiber_array_r[:1000], fibers.fiber_array_a[:1000], fibers.fiber_array_s[:1000]])
    field = np.random.RandomState(42).normal(size=6**3 * 3) * 5.
    objective = bspline.NonrigidObjective(moving[:, :100], moving, 6)
    objective.transform(field)
    coordinates = []
    for name, function in (
        ('VTK point loop', lambda: reference(moving, field)),
        ('VTK batch', lambda: bspline.transform_fiber_array_numpy(moving, field)),
        ('GPU B-spline', lambda: objective.transform(field).cpu().numpy().transpose(2, 0, 1))):
        result, seconds = timed(function)
        coordinates.append(result)
        print(name, 'median seconds', seconds, flush=True)
    for result in coordinates[1:]:
        np.testing.assert_array_equal(result, coordinates[0])
    oldcluster = baseline_module('whitematteranalysis._oldcluster', 'cluster.py')
    labels = np.random.RandomState(42).randint(-1, 20, sample.GetNumberOfLines())
    split = []
    for module in (oldcluster, wma.cluster):
        result, seconds = timed(lambda: module.mask_all_clusters(sample, labels, 22,
            preserve_point_data=True, preserve_cell_data=True, verbose=False))
        split.append(result)
        print('split', module.__name__, seconds, flush=True)
    oldio = baseline_module('whitematteranalysis._oldio', 'io.py')
    with tempfile.TemporaryDirectory() as directory:
        files = [str(Path(directory)/('%d.vtp' % i)) for i in range(22)]
        for name, function in (
            ('serial', lambda: [wma.io.write_polydata(p, f) for p, f in zip(split[1], files)]),
            ('old process writer', lambda: oldio.write_polydatas_parallel(split[1], files, jobs=8)),
            ('inherited-data writer', lambda: wma.io.write_polydatas_parallel(split[1], files, jobs=8))):
            _, seconds = timed(function)
            print(name, seconds, flush=True)
        for i, (a, b) in enumerate(zip(*split)):
            wma.io.write_polydata(a, str(Path(directory)/'old.vtp'))
            wma.io.write_polydata(b, str(Path(directory)/'new.vtp'))
            assert (Path(directory)/'old.vtp').read_bytes() == (Path(directory)/'new.vtp').read_bytes(), i
    atlas = wma.cluster.load_atlas(args.atlas, 'atlas')
    oldlinalg = baseline_module('whitematteranalysis.gpu._oldlinalg', 'gpu/linalg.py')
    outputs = []
    for streamed in (False, True):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        start = time.perf_counter()
        if streamed:
            blocks = distance.iter_rectangular_similarity(sample, atlas.nystrom_polydata,
                atlas.threshold, atlas.sigma, atlas.distance_method, atlas.bilateral)
            result = linalg.atlas_label_embedding_blocks(blocks, atlas, sample.GetNumberOfLines())
        else:
            B = distance.rectangular_similarity_tensor(sample, atlas.nystrom_polydata,
                atlas.threshold, atlas.sigma, atlas.distance_method, atlas.bilateral)
            result = oldlinalg.atlas_label_embedding(B, atlas)
            del B
        print('atlas streamed', streamed, 'seconds', time.perf_counter()-start,
            'peak extra MiB', (torch.cuda.max_memory_allocated()-base)/1024**2, flush=True)
        outputs.append(result)
    np.testing.assert_array_equal(outputs[0][1], outputs[1][1])
    for a, b in zip(*outputs):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
    # Check the final float32 VTK payload, not just labels.
    with tempfile.TemporaryDirectory() as directory:
        for i, result in enumerate(outputs):
            embed, labels, _, rows, columns = result
            output = vtk_copy(sample)
            color = wma.cluster._embed_to_rgb(embed)
            wma.cluster._format_output_polydata(output, labels, color, embed, rows, columns)
            wma.io.write_polydata(output, str(Path(directory)/('%d.vtp' % i)))
        assert (Path(directory)/'0.vtp').read_bytes() == (Path(directory)/'1.vtp').read_bytes()
    print('Coordinates, labels and output bytes match baseline.', flush=True)
    if args.full:
        start = time.perf_counter()
        result = wma.cluster.spectral_atlas_label(pd, atlas, use_gpu=True)
        print('Full atlas', pd.GetNumberOfLines(), 'fibers seconds', time.perf_counter()-start, flush=True)


def vtk_copy(pd):
    import vtk
    out = vtk.vtkPolyData()
    out.DeepCopy(pd)
    return out


if __name__ == '__main__':
    main()
