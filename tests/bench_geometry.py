"""Compare geometry preparation against a committed source on a real tractogram."""
import argparse
import json
from pathlib import Path
import tempfile
import statistics
import time
from unittest.mock import patch

import numpy as np
import vtk
import whitematteranalysis as wma
from test_output_pipeline import baseline_module


def compare(functions, repeats):
    outputs = [function() for function in functions]
    samples = [[], []]
    for repeat in range(repeats):
        for index in (range(2) if repeat % 2 == 0 else reversed(range(2))):
            start = time.perf_counter()
            outputs[index] = functions[index]()
            samples[index].append(time.perf_counter() - start)
    return outputs, [dict(seconds=values, median_seconds=statistics.median(values)) for values in samples]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input')
    parser.add_argument('--atlas')
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--baseline', default='710f7d4')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    old_filter = baseline_module('whitematteranalysis._geometryfilter', 'filter.py', args.baseline)
    old_fibers = baseline_module('whitematteranalysis._geometryfibers', 'fibers.py', args.baseline)
    current_fibers = wma.fibers
    data = wma.io.read_polydata(args.input)
    report = dict(baseline=args.baseline, fibers=data.GetNumberOfLines(), points=data.GetNumberOfPoints())
    results, report['access'] = compare([
        lambda: old_filter._lines_offsets_connectivity(data),
        lambda: wma.filter._lines_offsets_connectivity(data)], args.repeat)
    for a, b in zip(*results):
        np.testing.assert_array_equal(a, b)
    del results
    results, report['downsample'] = compare([
        lambda: old_filter.downsample(data, 2000, random_seed=42, verbose=False),
        lambda: wma.filter.downsample(data, 2000, random_seed=42, verbose=False)], args.repeat)
    with tempfile.TemporaryDirectory() as directory:
        paths = [Path(directory)/name for name in ('old.vtp','current.vtp')]
        for result, path in zip(results, paths):
            wma.io.write_polydata(result, str(path))
        assert paths[0].read_bytes() == paths[1].read_bytes()
    del results
    def resample(module):
        fibers = module.FiberArray()
        fibers.convert_from_polydata(data, 15)
        return fibers
    results, report['resample'] = compare([
        lambda: resample(old_fibers), lambda: resample(current_fibers)], args.repeat)
    for name in ('fiber_array_r', 'fiber_array_a', 'fiber_array_s'):
        np.testing.assert_array_equal(getattr(results[0], name), getattr(results[1], name))
    del results
    if args.atlas:
        assert wma.gpu.available(), 'CUDA is required for the GPU labeling comparison'
        atlas = wma.cluster.load_atlas(args.atlas, 'atlas')
        def label(module):
            pd = vtk.vtkPolyData()
            pd.ShallowCopy(data)
            with patch.object(wma, 'fibers', module):
                return wma.cluster.spectral_atlas_label(pd, atlas, use_gpu=True)
        results, report['gpu_labeling'] = compare([
            lambda: label(old_fibers), lambda: label(current_fibers)], args.repeat)
        np.testing.assert_array_equal(results[0][1], results[1][1])
        np.testing.assert_array_equal(results[0][3], results[1][3])
    report['values_and_downsample_bytes_match'] = True
    Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
