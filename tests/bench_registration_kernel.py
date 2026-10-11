"""Compare the current kernel with registration from a committed baseline.

Run: python tests/bench_registration_kernel.py --input brain.vtk
"""
import argparse
import subprocess
import time
import types
from unittest.mock import patch

import numpy as np
import torch
import whitematteranalysis as wma
from whitematteranalysis.gpu import registration as current
from whitematteranalysis.register_two_subjects import RegisterTractography


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', help='Optional real tractography vtk/vtp file')
    parser.add_argument('--baseline', default='406c585', help='Git revision to compare')
    parser.add_argument('--fixed', type=int, default=3000)
    parser.add_argument('--moving', type=int, default=1000)
    parser.add_argument('--repeat', type=int, default=20)
    parser.add_argument('--maxfun', type=int, default=30)
    args = parser.parse_args()
    source = subprocess.check_output([
        'git', 'show', args.baseline + ':whitematteranalysis/gpu/registration.py'], text=True)
    baseline = types.ModuleType('whitematteranalysis.gpu._benchmark_baseline')
    exec(compile(source, '<baseline registration>', 'exec'), baseline.__dict__)
    if args.input:
        pd = wma.io.read_polydata(args.input)
        pd = wma.filter.downsample(pd, args.fixed + args.moving,
                                   random_seed=42, verbose=False)
        if pd.GetNumberOfLines() < args.fixed + args.moving:
            parser.error('input has fewer fibers than --fixed + --moving')
        fa = wma.fibers.FiberArray()
        fa.convert_from_polydata(pd, points_per_fiber=10)
        fibers = np.array([fa.fiber_array_r, fa.fiber_array_a, fa.fiber_array_s])
    else:
        fibers = np.random.RandomState(42).normal(
            size=(3, args.fixed + args.moving, 10)) * 40
    fixed, moving = fibers[:, :args.fixed], fibers[:, args.fixed:]
    x = np.array([5., -3., 2., 4., -2., 1., 1.02, .98, 1.01,
                  1., -1., .5, -.5, 2., -2.])
    calls = [mod.AffineObjective(fixed, moving) for mod in (baseline, current)]
    values, peaks = [], []
    for fn in calls:
        torch.cuda.reset_peak_memory_stats()
        allocated = torch.cuda.memory_allocated()
        start = time.perf_counter()
        values.append(fn(x, [1]*4, 10.))
        print('Process first evaluation seconds:', time.perf_counter() - start)
        peaks.append((torch.cuda.max_memory_allocated() - allocated)/1024**2)
    assert values[0] == values[1]
    timings = [[], []]
    for i in range(args.repeat):
        for j in (range(2) if i % 2 == 0 else reversed(range(2))):
            start = time.perf_counter()
            calls[j](x, [1]*4, 10.)
            timings[j].append(time.perf_counter() - start)
    print('Objective median ms (baseline, current):',
          *[np.median(t)*1000 for t in timings])
    print('Peak extra tensor MiB (baseline, current):', *peaks)

    timings, results = [[], []], [None, None]
    for i in range(3):
        for j in (range(2) if i % 2 == 0 else reversed(range(2))):
            factory = (baseline.AffineObjective, type(calls[1]))[j]
            reg = RegisterTractography()
            reg.fixed, reg.moving = fixed, moving
            reg.use_gpu, reg.maxfun = True, args.maxfun
            with patch.object(current, 'AffineObjective', side_effect=factory):
                start = time.perf_counter()
                transform = reg.compute().copy()
                timings[j].append(time.perf_counter() - start)
            results[j] = (transform, reg.objective_function_values)
            if results[0] is not None and results[1] is not None:
                np.testing.assert_array_equal(results[0][0], results[1][0])
                np.testing.assert_array_equal(results[0][1], results[1][1])
    print('Cobyla budget %d median seconds including setup (baseline, current):' % args.maxfun,
          *[np.median(t) for t in timings])
    print('Actual evaluations:', len(results[0][1]))
    print('Objective sequence and final transform: bit-identical')


if __name__ == '__main__':
    main()
