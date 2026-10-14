"""Compare CUDA distance launch settings and optional real-subject labeling."""
import argparse
import json
import inspect
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import numpy as np
import torch
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import whitematteranalysis as wma
from whitematteranalysis.gpu import distance
from test_output_pipeline import baseline_module


def compare(functions, repeats):
    outputs = [function() for function in functions]
    samples = [[], []]
    peaks = [0, 0]
    device = wma.gpu.get_device()
    for repeat in range(repeats):
        for index in (range(2) if repeat % 2 == 0 else reversed(range(2))):
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            allocated = torch.cuda.memory_allocated(device)
            start = time.perf_counter()
            outputs[index] = functions[index]()
            torch.cuda.synchronize(device)
            samples[index].append(time.perf_counter()-start)
            peaks[index] = max(peaks[index], torch.cuda.max_memory_allocated(device) - allocated)
    return outputs, [dict(seconds=s, median_seconds=statistics.median(s),
                          peak_extra_tensor_bytes=peaks[i]) for i, s in enumerate(samples)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default='710f7d4')
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--bilateral', action='store_true')
    parser.add_argument('--input')
    parser.add_argument('--atlas')
    parser.add_argument('--fibers', type=int, default=100000)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    original = baseline_module('whitematteranalysis.gpu._launchbaseline',
                               'gpu/distance.py', args.baseline)
    original.distance_kernel = baseline_module('whitematteranalysis.gpu._launchkernelbaseline',
                                               'gpu/distance_kernel.py', args.baseline)
    bilateral_supported = 'bilateral' in inspect.signature(original._fiber_block_distances).parameters
    report = dict(baseline=args.baseline, bilateral=args.bilateral, distance={})
    torch.manual_seed(42)
    for points in (10, 15, 20):
        fixed = torch.randn((2500, points, 3), dtype=torch.float64, device=wma.gpu.get_device())*40
        queries = torch.randn((2048, points, 3), dtype=torch.float64, device=wma.gpu.get_device())*40
        for method in ('Mean', 'MeanSquared', 'Hausdorff', 'StrictSimilarity'):
            def previous():
                if bilateral_supported:
                    return original._fiber_block_distances(queries, fixed, 2., method, 6400., args.bilateral)
                result = original._fiber_block_distances(queries, fixed, 2., method, 6400.)
                if args.bilateral:
                    reflected = queries.clone()
                    reflected[:, :, 0] *= -1
                    combine = torch.maximum if method == 'StrictSimilarity' else torch.minimum
                    result = combine(result, original._fiber_block_distances(
                        reflected, fixed, 2., method, 6400.))
                return result
            outputs, timings = compare([
                previous, lambda: distance._fiber_block_distances(
                    queries, fixed, 2., method, 6400., args.bilateral)], args.repeat)
            assert torch.equal(*outputs)
            name = '%s_P%d' % (method, points)
            report['distance'][name] = timings
            print(name, json.dumps(timings), flush=True)
    del fixed, queries, outputs
    if args.input:
        assert args.atlas, '--atlas is required with --input'
        data = wma.filter.downsample(wma.io.read_polydata(args.input), args.fibers,
                                    random_seed=42, verbose=False)
        atlas = wma.cluster.load_atlas(args.atlas, 'atlas')
        def classify(module):
            subject = vtk.vtkPolyData()
            subject.ShallowCopy(data)
            with patch.object(wma.gpu, 'distance', module):
                return wma.cluster.spectral_atlas_label(subject, atlas, use_gpu=True)
        outputs, timings = compare([lambda: classify(original),
                                    lambda: classify(distance)], args.repeat)
        np.testing.assert_array_equal(outputs[0][1], outputs[1][1])
        np.testing.assert_array_equal(outputs[0][3], outputs[1][3])
        for index in range(outputs[0][0].GetCellData().GetNumberOfArrays()):
            reference = outputs[0][0].GetCellData().GetArray(index)
            actual = outputs[1][0].GetCellData().GetArray(reference.GetName())
            np.testing.assert_array_equal(vtk_to_numpy(reference), vtk_to_numpy(actual))
        report['classification'] = dict(fibers=data.GetNumberOfLines(), timings=timings)
        print('CLASSIFICATION', json.dumps(report['classification']), flush=True)
    report['values_match_exactly'] = True
    Path(args.output).write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
