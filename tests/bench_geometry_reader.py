"""Compare atlas reference geometry reads with the preceding full-data reader."""
import argparse
import json
from pathlib import Path
import statistics
import time
import unittest

import whitematteranalysis as wma
from test_geometry_reader import assert_geometry
from test_output_pipeline import baseline_module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--atlas', required=True)
    parser.add_argument('--stride', type=int, default=25)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--baseline', default='ff8bacb')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    original = baseline_module('whitematteranalysis._readbenchmark', 'io.py', args.baseline)
    paths = sorted(Path(args.atlas).glob('cluster_*.vtp'))[::args.stride]
    if not paths:
        parser.error('No reference cluster VTP files found')
    test = unittest.TestCase()
    memory = [0, 0]
    for path in paths:
        before = original.read_polydata(str(path))
        after = wma.io.read_polydata(str(path), geometry_only=True)
        assert_geometry(test, before, after)
        memory[0] += before.GetActualMemorySize()
        memory[1] += after.GetActualMemorySize()
    samples = [[], []]
    for repeat in range(args.repeat):
        for index in (range(2) if repeat % 2 == 0 else reversed(range(2))):
            start = time.perf_counter()
            for path in paths:
                if index:
                    wma.io.read_polydata(str(path), geometry_only=True)
                else:
                    original.read_polydata(str(path))
            samples[index].append(time.perf_counter() - start)
    report = dict(baseline=args.baseline, files=len(paths), geometry_exact=True,
        median_seconds=[statistics.median(s) for s in samples], seconds=samples,
        returned_polydata_kib=memory)
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
