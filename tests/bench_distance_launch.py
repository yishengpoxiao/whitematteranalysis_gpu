"""Compare CUDA distance launch settings and optional real-subject labeling."""
import argparse
import json
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import numpy as np
import torch
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import whitematteranalysis as wma
from whitematteranalysis.gpu import distance, distance_kernel
from test_output_pipeline import baseline_module


def compare(functions, repeats):
    outputs = [function() for function in functions]
    samples = [[], []]
    for repeat in range(repeats):
        for index in (range(2) if repeat % 2 == 0 else reversed(range(2))):
            torch.cuda.synchronize()
            start = time.perf_counter()
            outputs[index] = functions[index]()
            torch.cuda.synchronize()
            samples[index].append(time.perf_counter()-start)
    return outputs, [dict(seconds=s, median_seconds=statistics.median(s)) for s in samples]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default='710f7d4')
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--input')
    parser.add_argument('--atlas')
    parser.add_argument('--fibers', type=int, default=100000)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    original = baseline_module('whitematteranalysis.gpu._launchbaseline',
                               'gpu/distance_kernel.py', args.baseline)
    report = dict(baseline=args.baseline, distance={})
    torch.manual_seed(42)
    for points in (10, 15, 20):
        fixed = torch.randn((2500, points, 3), dtype=torch.float64, device=wma.gpu.get_device())*40
        queries = torch.randn((2048, points, 3), dtype=torch.float64, device=wma.gpu.get_device())*40
        for method in ('Mean', 'MeanSquared', 'Hausdorff'):
            outputs, timings = compare([
                lambda: original.fiber_distances(queries, fixed, 2., method),
                lambda: distance_kernel.fiber_distances(queries, fixed, 2., method)], args.repeat)
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
            with patch.object(distance, 'distance_kernel', module):
                return wma.cluster.spectral_atlas_label(subject, atlas, use_gpu=True)
        outputs, timings = compare([lambda: classify(original),
                                    lambda: classify(distance_kernel)], args.repeat)
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
