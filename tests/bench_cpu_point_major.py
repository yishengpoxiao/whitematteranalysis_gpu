"""Compare direct point-plane reductions with the preceding CPU matrices."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import time

import numpy as np
from whitematteranalysis.gpu import cpu_backend as current
from test_output_pipeline import baseline_module


def compare(functions, repeats):
    expected = functions[0]()
    np.testing.assert_array_equal(functions[1](), expected)
    samples = [[], []]
    for repeat in range(repeats):
        for index in (range(2) if repeat % 2 == 0 else reversed(range(2))):
            start = time.perf_counter()
            result = functions[index]()
            samples[index].append(time.perf_counter() - start)
            np.testing.assert_array_equal(result, expected)
    return dict(median_seconds=[statistics.median(s) for s in samples], seconds=samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default='592eefa')
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--queries', type=int, default=512)
    parser.add_argument('--workers', type=int, nargs='+', default=[1, 4])
    parser.add_argument('--points', type=int, nargs='+', default=[10, 15, 20])
    parser.add_argument('--methods', nargs='+', choices=('Mean', 'MeanSquared', 'Hausdorff', 'StrictSimilarity'),
                        default=['Mean', 'MeanSquared', 'Hausdorff', 'StrictSimilarity'])
    parser.add_argument('--seed', type=int, default=54)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    original = baseline_module('whitematteranalysis.gpu._pointmajor_baseline',
                               'gpu/cpu_backend.py', args.baseline)
    rng = np.random.RandomState(args.seed)
    records = []
    for points in args.points:
        queries = rng.normal(size=(args.queries, points, 3)) * 40.
        fixed = rng.normal(size=(2500, points, 3)) * 40.
        for method in args.methods:
            for bilateral in (False, True):
                for workers in args.workers:
                    def run(module):
                        return module._distance_matrix(queries, fixed, 2., method, bilateral,
                                                       number_of_jobs=workers)
                    record = dict(queries=args.queries, fixed=len(fixed), points=points,
                        method=method, bilateral=bilateral, workers=workers,
                        **compare([lambda: run(original), lambda: run(current)], args.repeat))
                    records.append(record)
                    print(json.dumps(record), flush=True)
    objectives = []
    for n_fixed, n_moving in ((2500, 1000), (5000, 2000)):
        fixed, moving = rng.normal(size=(3, n_fixed, 10))*40., rng.normal(size=(3, n_moving, 10))*40.
        with ThreadPoolExecutor(max_workers=4) as pool:
            def run(module):
                return module.inner_loop_objective(fixed, moving, 400., threads=4, executor=pool)
            record = dict(fixed=n_fixed, moving=n_moving,
                **compare([lambda: run(original), lambda: run(current)], args.repeat))
        objectives.append(record)
        print('REGISTRATION', json.dumps(record), flush=True)
    Path(args.output).write_text(json.dumps(dict(baseline=args.baseline,
        all_values_exact=True, distance=records, registration=objectives), indent=2)+'\n')


if __name__ == '__main__':
    main()
