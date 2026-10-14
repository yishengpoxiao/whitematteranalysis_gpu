"""Compare zero-threshold Mean distance with the preceding CPU implementation."""
import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np
from whitematteranalysis.gpu import cpu_backend as current
from test_output_pipeline import baseline_module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default='dfe1ca5')
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    original = baseline_module('whitematteranalysis.gpu._mean_baseline',
                               'gpu/cpu_backend.py', args.baseline)
    rng = np.random.RandomState(42)
    records = []
    for points in (10, 15, 20):
        queries = rng.normal(size=(512, points, 3)) * 40.
        fixed = rng.normal(size=(2500, points, 3)) * 40.
        for bilateral in (False, True):
            for jobs in (1, 4):
                def run(module):
                    return module._distance_matrix(queries, fixed, 0., 'Mean', bilateral,
                                                   number_of_jobs=jobs)
                expected = run(original)
                np.testing.assert_array_equal(run(current), expected)
                samples = [[], []]
                for repeat in range(args.repeat):
                    for index in (range(2) if repeat % 2 == 0 else reversed(range(2))):
                        start = time.perf_counter()
                        result = run((original, current)[index])
                        samples[index].append(time.perf_counter() - start)
                        np.testing.assert_array_equal(result, expected)
                record = dict(points=points, bilateral=bilateral, jobs=jobs,
                    median_seconds=[statistics.median(s) for s in samples], seconds=samples)
                records.append(record)
                print(json.dumps(record), flush=True)
    Path(args.output).write_text(json.dumps(dict(baseline=args.baseline,
        values_match_exactly=True, measurements=records), indent=2) + '\n')


if __name__ == '__main__':
    main()
