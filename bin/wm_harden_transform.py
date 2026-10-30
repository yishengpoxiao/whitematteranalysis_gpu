#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time
from joblib import Parallel, delayed, effective_n_jobs

import whitematteranalysis as wma


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Harden transforms with independent Slicer processes.",
                                     epilog="Original script by Fan Zhang, fzhang@bwh.harvard.edu.")
    parser.add_argument('-v', '--version', action='version', version='1.0')
    parser.add_argument('inputDirectory', help='Directory of input VTK/VTP files.')
    parser.add_argument('outputDirectory', help='Directory of transformed results.')
    parser.add_argument('Slicer', help='Path of 3D Slicer.')
    transforms = parser.add_mutually_exclusive_group(required=True)
    transforms.add_argument('-t', dest='transform_file', help='One transform for all input files.')
    transforms.add_argument('-ts', dest='transform_folder', help='Sorted transforms matching sorted input files.')
    parser.add_argument('-i', action='store_true', dest='inverse_transform', help='Invert the transform.')
    parser.add_argument('-j', dest='numberOfJobs', type=int, default=1,
                        help='Number of Slicer workers. A shared transform uses one batch per worker.')
    parser.add_argument('--split-output-directory', default=None,
                        help='Also write hemisphere subsets from each hardened file.')
    parser.add_argument('--profile', action='store_true',
                        help='Keep worker logs and JSON timings after successful processing.')
    parser.add_argument('--input-catalogue',
                        help='Directory listing expected input filenames before preparation.')
    parser.add_argument('--prepare-command', nargs=argparse.REMAINDER,
                        help='Prepare inputs while Slicer starts; place this command last.')
    return parser


def _run_slicer(polydata, transform, inverse, slicer_path, helper, outdir, partition,
                partitions, log, split_output_dir=None, profile_output=None, input_status=None):
    command = [str(slicer_path), '--no-main-window', '--disable-modules',
               '--python-script', helper,
               str(polydata), str(transform), str(int(inverse)), str(outdir),
               str(partition), str(partitions)]
    if split_output_dir is not None:
        command.extend(['--split-output-directory', str(split_output_dir)])
    if profile_output is not None:
        command.extend(['--profile-output', str(profile_output)])
    if input_status is not None:
        command.extend(['--input-status', str(input_status)])
    command.extend(['--python-code', 'slicer.app.quit()'])
    environment = os.environ.copy()
    environment.pop('PYTHONPATH', None)
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'):
        environment[name] = '1'
    if split_output_dir is not None:
        environment.setdefault('WMA_VTP_COMPRESSION', 'lz4')
    if profile_output is not None:
        environment['WMA_HARDEN_LAUNCH_TIME'] = str(time.monotonic())
    with log.open('w') as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True,
                       env=environment)


def _publish_status(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(value)
    temporary.replace(path)


def _cancel_preparation(signum, frame):
    raise SystemExit(128 + signum)


def _run_prepared(runs, workers, command, status, inputdir, expected):
    previous = signal.signal(signal.SIGTERM, _cancel_preparation)
    released = False
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            try:
                futures = [pool.submit(run) for run in runs]
                started = time.perf_counter()
                subprocess.run(command, check=True)
                prepare_seconds = time.perf_counter() - started
                actual = {Path(path).name for path in wma.io.list_vtk_files(str(inputdir))}
                if actual != expected:
                    raise RuntimeError('Prepared input filenames do not match the catalogue.')
                _publish_status(status, 'ready')
                released = True
            finally:
                if not released:
                    _publish_status(status, 'failed')
            for future in futures:
                future.result()
    finally:
        signal.signal(signal.SIGTERM, previous)
    return prepare_seconds


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    inputdir, outdir, slicer_path = map(Path, (args.inputDirectory, args.outputDirectory, args.Slicer))
    inputdir, outdir, slicer_path = inputdir.resolve(), outdir.resolve(), slicer_path.resolve()
    split_output_dir = (Path(args.split_output_directory).resolve()
                        if args.split_output_directory else None)
    if bool(args.input_catalogue) != bool(args.prepare_command):
        parser.error('--input-catalogue and --prepare-command must be used together.')
    catalogue = Path(args.input_catalogue).resolve() if args.input_catalogue else inputdir
    if not catalogue.is_dir():
        parser.error('Input catalogue does not exist: ' + str(catalogue))
    if not args.prepare_command and not inputdir.is_dir():
        parser.error('Input directory does not exist: ' + str(inputdir))
    if not slicer_path.is_file():
        parser.error('Slicer executable does not exist: ' + str(slicer_path))
    outdir.mkdir(parents=True, exist_ok=True)
    if split_output_dir is not None:
        for folder in ('tracts_left_hemisphere', 'tracts_right_hemisphere', 'tracts_commissural'):
            (split_output_dir / folder).mkdir(parents=True, exist_ok=True)
    inputs = [str(inputdir / Path(path).name) for path in wma.io.list_vtk_files(str(catalogue))]
    if args.prepare_command and not inputs:
        parser.error('The preparation catalogue must contain input VTK/VTP files.')
    workers = min(effective_n_jobs(args.numberOfJobs), max(1, len(inputs)))
    helper = str(Path(__file__).resolve().with_name('harden_transform_with_slicer.py'))
    if args.transform_folder:
        transform_path = Path(args.transform_folder).resolve()
        if not transform_path.is_dir():
            parser.error('Transform directory does not exist: ' + str(transform_path))
        transforms = wma.io.list_transform_files(str(transform_path))
        if len(inputs) != len(transforms):
            parser.error('The input and transform file counts must match.')
        tasks = [(polydata, transform, 0, 1) for polydata, transform in zip(inputs, transforms)]
    else:
        transform_path = Path(args.transform_file).resolve()
        if not transform_path.is_file():
            parser.error('Transform file does not exist: ' + str(transform_path))
        tasks = [(inputdir, transform_path, index, workers) for index in range(workers)] if inputs else []
    print('Harden %d files with %d Slicer worker(s).' % (len(inputs), workers), flush=True)
    if not tasks:
        return
    marker = outdir/'transform_complete.log'
    if marker.exists():
        marker.unlink()
    logs = Path(tempfile.mkdtemp(prefix='wma-harden-', dir=str(outdir)))
    print('Slicer worker logs: ' + str(logs), flush=True)
    status = logs / 'input.status' if args.prepare_command else None
    runs = [partial(_run_slicer, polydata, transform, args.inverse_transform, slicer_path, helper,
            outdir, index, count, logs/('worker_%04d.log' % task), split_output_dir,
            logs/('worker_%04d.json' % task) if args.profile else None, status)
        for task, (polydata, transform, index, count) in enumerate(tasks)]
    expected = {Path(path).name for path in inputs}
    started = time.perf_counter()
    if args.prepare_command:
        prepare_seconds = _run_prepared(runs, workers, args.prepare_command, status, inputdir, expected)
        if args.profile:
            (logs / 'preparation.json').write_text(json.dumps({
                'prepare_s': prepare_seconds, 'combined_s': time.perf_counter() - started,
                'command': args.prepare_command}, indent=2) + '\n')
    else:
        Parallel(n_jobs=workers, prefer='threads')(delayed(run)() for run in runs)
    actual = {Path(path).name for path in wma.io.list_vtk_files(str(outdir))}
    if not expected.issubset(actual):
        raise RuntimeError('Transformed files are missing; inspect Slicer logs: ' + str(logs))
    if split_output_dir is not None:
        for folder in ('tracts_left_hemisphere', 'tracts_right_hemisphere', 'tracts_commissural'):
            split_files = {Path(path).name for path in wma.io.list_vtk_files(str(split_output_dir / folder))}
            if not expected.issubset(split_files):
                raise RuntimeError('Hemisphere outputs are missing from %s; inspect Slicer logs: %s'
                                   % (folder, logs))
    if not args.profile:
        shutil.rmtree(logs)
    marker.write_text('Transforms completed for %d files.\n' % len(inputs))
    print('Transforms completed for %d files.' % len(inputs), flush=True)


if __name__ == '__main__':
    main()
