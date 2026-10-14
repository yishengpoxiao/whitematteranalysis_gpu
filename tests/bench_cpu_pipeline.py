"""Bounded, warm-cache CPU/GPU classification and CPU registration comparison.

Run with PYTHONPATH=. and OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
OMP_NUM_THREADS=1 WMA_CPU_THREADS=4. Use taskset for a fixed CPU budget.
Inputs and generated geometry are never included in the JSON report.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import numpy as np
import vtk
import whitematteranalysis as wma
from test_output_pipeline import baseline_module


def measure(function, repeats):
    function()  # Imports, CUDA kernels and worker startup are outside timing.
    seconds = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        seconds.append(time.perf_counter() - start)
    return result, dict(seconds=seconds, median_seconds=statistics.median(seconds))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input')
    parser.add_argument('atlas')
    parser.add_argument('--fibers', type=int, default=2048)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--baseline', default='60c7b2d')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cpu = wma.gpu.cpu_backend
    oldcpu = baseline_module('whitematteranalysis.gpu._benchmarkcpu', 'gpu/cpu_backend.py', args.baseline)
    oldcluster = baseline_module('whitematteranalysis._benchmarkcluster', 'cluster.py', args.baseline)
    subject = wma.filter.downsample(wma.io.read_polydata(args.input), args.fibers,
                                   random_seed=1000, verbose=False)
    atlas = wma.cluster.load_atlas(args.atlas, 'atlas')
    report = dict(baseline=args.baseline, fibers=subject.GetNumberOfLines(),
                  atlas_sample=atlas.nystrom_polydata.GetNumberOfLines(),
                  embedding_columns=atlas.number_of_eigenvectors + 1,
                  classification={}, registration={})
    expected = None
    profiles = [('previous_cpu_matrix_4', oldcluster, oldcpu, True, False),
                ('original_joblib_4', oldcluster, oldcpu, False, False),
                ('streamed_cpu_4', wma.cluster, cpu, True, False)]
    if wma.gpu.available():
        profiles.append(('gpu', wma.cluster, cpu, True, True))
    for name, cluster, backend, fast, gpu in profiles:
        def classify():
            pd = vtk.vtkPolyData()
            pd.ShallowCopy(subject)
            with patch.object(wma.gpu, 'cpu_backend', backend), \
                    patch.object(wma.gpu.config, 'requested', gpu), \
                    patch.object(wma.gpu.config, 'cpu_fast_enabled', return_value=fast):
                return cluster.spectral_atlas_label(pd, atlas, number_of_jobs=4, use_gpu=gpu)
        result, timing = measure(classify, args.repeats)
        if expected is None:
            expected = result
        np.testing.assert_array_equal(result[1], expected[1])
        np.testing.assert_allclose(result[3], expected[3], rtol=1e-11, atol=1e-11)
        # Serialized attributes are float32; compare their values and topology.
        for index in range(expected[0].GetCellData().GetNumberOfArrays()):
            from vtk.util.numpy_support import vtk_to_numpy
            reference = expected[0].GetCellData().GetArray(index)
            actual = result[0].GetCellData().GetArray(reference.GetName())
            np.testing.assert_array_equal(vtk_to_numpy(actual), vtk_to_numpy(reference))
        report['classification'][name] = timing
        print(name, json.dumps(timing), flush=True)
    rng = np.random.RandomState(1000)
    fixed, moving = rng.normal(size=(3, 2500, 10)) * 40., rng.normal(size=(3, 1000, 10)) * 40.
    expected_objective, timing = measure(
        lambda: wma.register_two_subjects.inner_loop_objective(fixed, moving, 400.), args.repeats)
    report['registration']['original_serial'] = timing
    print('original_serial', json.dumps(timing), flush=True)
    for workers in (1, 4):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            value, timing = measure(lambda: cpu.inner_loop_objective(
                fixed, moving, 400., threads=workers, executor=pool), args.repeats)
        assert value == expected_objective
        name = 'cpu_%d_workers' % workers
        report['registration'][name] = timing
        print(name, json.dumps(timing), flush=True)
    report['labels_and_serialized_attributes_match'] = True
    report['registration_objectives_match_exactly'] = True
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
