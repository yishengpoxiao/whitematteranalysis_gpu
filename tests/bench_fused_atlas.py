"""Compare fused distances, colors, coefficient reuse and atlas overlap with b2ce685."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import statistics
import tempfile
import time
from unittest.mock import patch

import numpy as np
import torch
import vtk
import whitematteranalysis as wma
from whitematteranalysis.gpu import distance, linalg, bspline, registration
from test_output_pipeline import baseline_module


def measure(function, repeats=5):
    function()
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = function()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return output, statistics.median(times)


def clone(pd):
    out = vtk.vtkPolyData()
    out.DeepCopy(pd)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='/data/yijie/TractData/val_dsi_studio/250932.vtk')
    parser.add_argument('--atlas', default='/data/software/ORG-Atlases/ORG-800FC-100HCP')
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()
    olddistance = baseline_module('whitematteranalysis.gpu._olddistance', 'gpu/distance.py', 'b2ce685')
    oldlinalg = baseline_module('whitematteranalysis.gpu._oldlinalg', 'gpu/linalg.py', 'b2ce685')
    oldbspline = baseline_module('whitematteranalysis.gpu._oldbspline', 'gpu/bspline.py', 'b2ce685')
    oldregistration = baseline_module('whitematteranalysis.gpu._oldregistration', 'gpu/registration.py', 'b2ce685')
    oldcluster = baseline_module('whitematteranalysis._oldcluster', 'cluster.py', 'b2ce685')
    rng = np.random.RandomState(42)
    f = torch.tensor(rng.normal(size=(2500, 15, 3))*40, device='cuda')
    q = torch.tensor(rng.normal(size=(2048, 15, 3))*40, device='cuda')
    for method in ('Mean', 'MeanSquared', 'Hausdorff'):
        outputs = []
        for module in (olddistance, distance):
            function = lambda: module._fiber_block_distances(q, f, 2., method, 6400.)
            output, seconds = measure(function)
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            check = function()
            torch.cuda.synchronize()
            peak = (torch.cuda.max_memory_allocated()-base)/1024**2
            del check
            outputs.append(output)
            print('distance', method, module.__name__, seconds, 'peak extra MiB', peak, flush=True)
        assert torch.equal(*outputs)
    embed = rng.normal(size=(100000, 10))
    for module in (oldcluster, wma.cluster):
        _, seconds = measure(lambda: module._embed_to_rgb(embed.copy()))
        print('RGB', module.__name__, seconds, flush=True)
    moving = rng.normal(size=(3, 1000, 10))*40
    field = rng.normal(size=6**3*3)*5
    outputs = []
    for module in (oldbspline, bspline):
        objective = module.NonrigidObjective(moving[:, :100], moving, 6)
        output, seconds = measure(lambda: objective.transform(field).cpu().numpy())
        outputs.append(output)
        print('B-spline', module.__name__, seconds, flush=True)
    np.testing.assert_array_equal(*outputs)
    fixed = registration._prepare_fixed(moving, 'cuda', torch.float64)
    for module in (oldregistration, registration):
        output, seconds = measure(lambda: module._inner_loop_objective(fixed, fixed, 400., 'cuda', 64))
        print('multi-tile entropy', module.__name__, output, seconds, flush=True)
    pd = wma.io.read_polydata(args.input)
    if not args.full:
        pd = wma.filter.downsample(pd, 10000, random_seed=42, preserve_point_data=True, verbose=False)
    atlas = wma.cluster.load_atlas(args.atlas, 'atlas')
    overlap_sample = wma.filter.downsample(pd, 10000, random_seed=42, verbose=False)
    overlap_outputs = []
    for module in (oldlinalg, linalg):
        def embedding():
            blocks = distance.iter_rectangular_similarity(overlap_sample, atlas.nystrom_polydata,
                atlas.threshold, atlas.sigma, atlas.distance_method, atlas.bilateral)
            return module.atlas_label_embedding_blocks(blocks, atlas, overlap_sample.GetNumberOfLines())
        output, seconds = measure(embedding, repeats=5)
        overlap_outputs.append(output)
        print('classification overlap', module.__name__, seconds, flush=True)
    for a, b in zip(*overlap_outputs):
        np.testing.assert_array_equal(a, b)
    outputs = []
    for baseline in (True, False):
        def run():
            data = clone(pd)
            if baseline:
                with patch.object(distance, 'iter_rectangular_similarity', olddistance.iter_rectangular_similarity), \
                     patch.object(linalg, 'atlas_label_embedding_blocks', oldlinalg.atlas_label_embedding_blocks):
                    return oldcluster.spectral_atlas_label(data, atlas, use_gpu=True)
            return wma.cluster.spectral_atlas_label(data, atlas, use_gpu=True)
        output, seconds = measure(run, repeats=3)
        outputs.append(output)
        print('atlas baseline', baseline, pd.GetNumberOfLines(), 'fibers', seconds, flush=True)
    np.testing.assert_array_equal(outputs[0][1], outputs[1][1])
    np.testing.assert_array_equal(outputs[0][2], outputs[1][2])
    np.testing.assert_array_equal(outputs[0][3], outputs[1][3])
    with tempfile.TemporaryDirectory() as directory:
        for i, result in enumerate(outputs):
            wma.io.write_polydata(result[0], str(Path(directory)/('%d.vtp' % i)))
        assert (Path(directory)/'0.vtp').read_bytes() == (Path(directory)/'1.vtp').read_bytes()
    print('Full atlas labels, colors, embedding and VTP bytes match.', flush=True)


if __name__ == '__main__':
    main()
