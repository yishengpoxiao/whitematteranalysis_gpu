"""Measure exact eager CUDA distances with reused point-distance scratch."""
import argparse
import json
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import torch
from whitematteranalysis.gpu import distance, distance_kernel
from test_output_pipeline import baseline_module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default='97bc764')
    parser.add_argument('--repeat', type=int, default=7)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    original = baseline_module('whitematteranalysis.gpu._scratchbenchbaseline',
                               'gpu/distance.py', args.baseline)
    torch.manual_seed(78)
    records = []
    # Force the same eager arithmetic used when Triton is unavailable.
    with patch.object(distance_kernel, 'triton', None):
        for dtype, points in ((torch.float32, 15), (torch.float64, 15), (torch.float64, 50)):
            queries = torch.randn((512, points, 3), dtype=dtype, device='cuda') * 40.
            fixed = torch.randn((2500, points, 3), dtype=dtype, device='cuda') * 40.
            for method in ('Mean', 'MeanSquared', 'Hausdorff', 'StrictSimilarity'):
                def run(module):
                    return module._fiber_block_distances(queries, fixed, 2., method, 6400.)
                expected = run(original)
                torch.testing.assert_close(run(distance), expected, rtol=0., atol=0., equal_nan=True)
                samples, peaks = [[], []], [0, 0]
                for repeat in range(args.repeat):
                    for index in (range(2) if repeat % 2 == 0 else reversed(range(2))):
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        allocated = torch.cuda.memory_allocated()
                        start = time.perf_counter()
                        actual = run((original, distance)[index])
                        torch.cuda.synchronize()
                        samples[index].append(time.perf_counter() - start)
                        peaks[index] = max(peaks[index], torch.cuda.max_memory_allocated() - allocated)
                        torch.testing.assert_close(actual, expected, rtol=0., atol=0., equal_nan=True)
                        del actual
                record = dict(dtype=str(dtype), points=points, method=method,
                    median_seconds=[statistics.median(s) for s in samples],
                    peak_extra_bytes=peaks, seconds=samples)
                records.append(record)
                print(json.dumps(record), flush=True)
            del expected, queries, fixed
    Path(args.output).write_text(json.dumps(dict(baseline=args.baseline,
        device=torch.cuda.get_device_name(), values_match_exactly=True,
        distance=records), indent=2) + '\n')


if __name__ == '__main__':
    main()
