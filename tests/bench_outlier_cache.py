"""Benchmark cold/warm atlas caching and compare every cluster file and log."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Existing subject cluster directory')
    parser.add_argument('--atlas', required=True, help='Atlas directory containing atlas.p and clusters')
    parser.add_argument('--output', required=True, help='New benchmark directory')
    parser.add_argument('--backend', choices=('cpu', 'gpu'), default='gpu')
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--baseline', default='d68372a')
    parser.add_argument('--repeat', type=int, default=2)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    original = output / 'baseline.py'
    original.write_bytes(subprocess.check_output(
        ['git', 'show', args.baseline + ':bin/wm_cluster_remove_outliers.py'], cwd=ROOT))
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(ROOT),
        WMA_ATLAS_CACHE=str(output / 'cache'), WMA_CPU_FAST='1')
    subject = Path(args.input).resolve().name + '_outlier_removed'
    records = []
    for mode in ['baseline', 'cold'] + ['warm%d' % i for i in range(1, args.repeat + 1)]:
        script = original if mode == 'baseline' else ROOT / 'bin/wm_cluster_remove_outliers.py'
        command = [sys.executable, str(script), args.input, args.atlas, str(output / mode),
                   '-j', str(args.jobs), '-gpu', 'on' if args.backend == 'gpu' else 'off']
        start = time.perf_counter()
        with (output / (mode + '.log')).open('w') as log:
            subprocess.run(command, cwd=ROOT, env=environment, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        record = dict(mode=mode, seconds=time.perf_counter() - start)
        expected, actual = output / 'baseline' / subject, output / mode / subject
        filenames = sorted(p.name for p in expected.glob('cluster_*.vtp'))
        assert filenames == sorted(p.name for p in actual.glob('cluster_*.vtp'))
        for name in filenames + ['cluster_log.txt']:
            assert (expected / name).read_bytes() == (actual / name).read_bytes(), name
        record.update(clusters=len(filenames), files_and_log_exact=True,
            cache_bytes=sum(p.stat().st_size for p in (output / 'cache').glob('*.npz')))
        records.append(record)
        (output / 'timings.json').write_text(json.dumps(records, indent=2) + '\n')
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
