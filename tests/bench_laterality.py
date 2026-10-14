"""Paired laterality timing against the corrected full-matrix implementation."""
import argparse
import json
from pathlib import Path
import statistics
import time
import unittest
from unittest.mock import patch

import numpy as np
import vtk
import whitematteranalysis as wma
from test_cli_file_stages import assert_equal, tractography
from test_output_pipeline import baseline_module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', help='Optional real VTK/VTP tractography')
    parser.add_argument('--fibers', type=int, default=2048)
    parser.add_argument('--backend', choices=('cpu', 'gpu'), default='gpu')
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--baseline', default='ff8bacb')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.input:
        subject = wma.filter.downsample(wma.io.read_polydata(args.input), args.fibers,
                                       random_seed=42, verbose=False)
    else:
        subject = tractography(args.fibers)
    original = baseline_module('whitematteranalysis._lateralitybenchmark', 'laterality.py', args.baseline)
    original.compute_laterality_index = wma.laterality.compute_laterality_index
    gpu = args.backend == 'gpu'
    if gpu:
        import torch
        from whitematteranalysis.gpu import distance_kernel
    def run(index):
        analysis = (original.WhiteMatterLaterality, wma.laterality.WhiteMatterLaterality)[index]()
        analysis.verbose, analysis.parallel_jobs = False, 1
        data = vtk.vtkPolyData()
        data.ShallowCopy(subject)
        with patch.object(wma.gpu.config, 'requested', gpu), patch.object(
                wma.gpu.config, 'cpu_fast_enabled', return_value=True), patch.object(
                wma.gpu.config, 'float32', False):
            if gpu and not index:
                # The original GPU dtype was invalid. Use the corrected matrix
                # helper without Triton to isolate its intended eager algorithm.
                with patch.object(distance_kernel, 'triton', None):
                    return analysis.compute(data)
            return analysis.compute(data)
    expected = run(0)
    test = unittest.TestCase()
    samples, peaks = [[], []], [0, 0]
    actual = run(1)  # Compile before timing.
    for repeat in range(args.repeat):
        for index in (range(2) if repeat % 2 == 0 else reversed(range(2))):
            if gpu:
                device = wma.gpu.get_device()
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                allocated = torch.cuda.memory_allocated(device)
            start = time.perf_counter()
            actual = run(index)
            if gpu:
                torch.cuda.synchronize(device)
                peaks[index] = max(peaks[index], torch.cuda.max_memory_allocated(device) - allocated)
            samples[index].append(time.perf_counter() - start)
            for name in ('laterality_index', 'left_hem_similarity', 'right_hem_similarity', 'hemisphere'):
                np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
            assert_equal(test, expected.polydata, actual.polydata)
    report = dict(backend=args.backend, baseline=args.baseline,
        input_fibers=subject.GetNumberOfLines(), output_fibers=actual.polydata.GetNumberOfLines(),
        median_seconds=[statistics.median(s) for s in samples], seconds=samples,
        peak_extra_tensor_bytes=peaks if gpu else None, all_outputs_exact=True,
        baseline_initialization_and_index_fixes=True, baseline_gpu_dtype_corrected=gpu)
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
