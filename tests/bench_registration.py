"""Measure stateless vs resident GPU registration, including optimizer setup.

Run from the repository: python tests/bench_registration.py [--input brain.vtk]
"""
import argparse
import time
from unittest.mock import patch

import numpy as np
import whitematteranalysis as wma
from whitematteranalysis.gpu import registration as gpu
from whitematteranalysis.register_two_subjects import RegisterTractography


def stateless(fixed, moving):
    return lambda x, mode, sigma: gpu.objective_value(fixed, moving, x, mode, sigma)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', help='Optional real tractography vtk/vtp file')
    parser.add_argument('--fixed', type=int, default=3000)
    parser.add_argument('--moving', type=int, default=1000)
    parser.add_argument('--repeat', type=int, default=20)
    args = parser.parse_args()
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
    resident = gpu.AffineObjective(fixed, moving)
    calls = [stateless(fixed, moving), resident]
    expected = calls[0](x, [1]*4, 10.)
    assert calls[1](x, [1]*4, 10.) == expected
    timings = [[], []]
    for i in range(args.repeat):
        for j in (range(2) if i % 2 == 0 else reversed(range(2))):
            start = time.perf_counter()
            calls[j](x, [1]*4, 10.)
            timings[j].append(time.perf_counter() - start)
    print('Objective median ms (stateless, resident):',
          *[np.median(t)*1000 for t in timings])

    results, elapsed = [], []
    factory = gpu.AffineObjective
    for backend in (stateless, factory):
        reg = RegisterTractography()
        reg.fixed, reg.moving = fixed, moving
        reg.use_gpu, reg.maxfun = True, 30
        start = time.perf_counter()
        with patch.object(gpu, 'AffineObjective', side_effect=backend):
            transform = reg.compute()
        elapsed.append(time.perf_counter() - start)
        results.append((transform, reg.objective_function_values))
    np.testing.assert_array_equal(results[0][0], results[1][0])
    np.testing.assert_array_equal(results[0][1], results[1][1])
    print('Cobyla 30 evaluations seconds including setup (stateless, resident):', *elapsed)
    print('Objective sequence and final transform: bit-identical')


if __name__ == '__main__':
    main()
