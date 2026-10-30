"""Exercise the Slicer launcher and completion bookkeeping without Slicer."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'bin' / 'wm_harden_transform.py'
SPEC = importlib.util.spec_from_file_location('wm_harden_transform', SCRIPT)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class HardenWorkerTests(unittest.TestCase):
    def test_run_slicer_builds_structured_command_and_isolates_environment(self):
        thread_limits = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                         'NUMEXPR_NUM_THREADS', 'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS')
        with tempfile.TemporaryDirectory(prefix='wma slicer launcher ') as temporary:
            root = Path(temporary)
            log = root / 'worker log.txt'
            profile = root / 'worker profile.json'
            command_paths = [root / name for name in ('input file.vtp', 'transform file.tfm',
                           'Slicer app', 'helper script.py', 'output files', 'split output')]
            slicer, helper, outdir, split = command_paths[2], command_paths[3], command_paths[4], command_paths[5]
            with patch.dict(os.environ, {'PYTHONPATH': 'parent path', 'WMA_VTP_COMPRESSION': 'zlib'}, clear=True), \
                    patch.object(launcher.subprocess, 'run') as run:
                launcher._run_slicer(command_paths[0], command_paths[1], True, slicer,
                                     helper, outdir, 2, 4, log, split, profile)
                command = run.call_args.args[0]
                env = run.call_args.kwargs['env']
                self.assertEqual(list(map(str, command)), [str(slicer), '--no-main-window',
                    '--disable-cli-modules', '--disable-scripted-loadable-modules',
                    '--python-script', str(helper), str(command_paths[0]), str(command_paths[1]),
                    '1', str(outdir), '2', '4', '--split-output-directory', str(split),
                    '--profile-output', str(profile), '--python-code', 'slicer.app.quit()'])
                self.assertEqual(run.call_args.kwargs['stdout'].name, str(log))
                self.assertIs(run.call_args.kwargs['stderr'], launcher.subprocess.STDOUT)
                self.assertTrue(run.call_args.kwargs['check'])
                self.assertNotIn('PYTHONPATH', env)
                self.assertEqual(env['WMA_VTP_COMPRESSION'], 'zlib')
                self.assertEqual({name: env[name] for name in thread_limits},
                                 {name: '1' for name in thread_limits})
                self.assertIn('WMA_HARDEN_LAUNCH_TIME', env)
                self.assertEqual(os.environ.get('PYTHONPATH'), 'parent path')
                self.assertEqual(os.environ.get('WMA_VTP_COMPRESSION'), 'zlib')
            with patch.dict(os.environ, {}, clear=True), patch.object(launcher.subprocess, 'run') as run:
                launcher._run_slicer(command_paths[0], command_paths[1], False, slicer,
                                     helper, outdir, 0, 1, log, split)
                self.assertEqual(run.call_args.kwargs['env']['WMA_VTP_COMPRESSION'], 'lz4')

    def test_main_keeps_profile_artifacts_only_when_requested(self):
        with tempfile.TemporaryDirectory(prefix='wma harden main ') as temporary:
            root = Path(temporary)
            source, output = root / 'input files', root / 'output files'
            source.mkdir()
            output.mkdir()
            (source / 'cluster_00001.vtp').write_bytes(b'input')
            (source / 'cluster_00002.vtp').write_bytes(b'input')
            transform = root / 'transform file.tfm'
            transform.write_text('unused by fake workers')
            slicer = root / 'Slicer app'
            slicer.touch()
            split = root / 'split output'

            def fake_worker(polydata, transform, inverse, slicer_path, helper, outdir,
                            partition, partitions, log, split_output_dir=None,
                            profile_output=None):
                names = sorted(path.name for path in source.glob('*.vtp'))[partition::partitions]
                for name in names:
                    (outdir / name).write_bytes(b'hardened')
                    if split_output_dir is not None:
                        for folder in ('tracts_left_hemisphere', 'tracts_right_hemisphere',
                                       'tracts_commissural'):
                            (split_output_dir / folder / name).write_bytes(b'split')
                log.write_text('fake worker log')
                if profile_output is not None:
                    Path(profile_output).write_text(json.dumps({'files': 1, 'success': True}))

            args = [str(source), str(output), str(slicer), '-t', str(transform), '-j', '2',
                    '--split-output-directory', str(split)]
            with patch.object(launcher, '_run_slicer', side_effect=fake_worker) as worker:
                with patch('sys.argv', ['wm_harden_transform.py'] + args + ['--profile']):
                    launcher.main()
                self.assertEqual(worker.call_count, 2)
                self.assertTrue((output / 'transform_complete.log').is_file())
                profiles = list(output.glob('wma-harden-*/*.json'))
                self.assertEqual(len(profiles), 2)
                for profile in profiles:
                    self.assertEqual(json.loads(profile.read_text()), {'files': 1, 'success': True})
                self.assertTrue(all((split / folder / name).is_file()
                    for folder in ('tracts_left_hemisphere', 'tracts_right_hemisphere',
                                   'tracts_commissural')
                    for name in ('cluster_00001.vtp', 'cluster_00002.vtp')))

            for path in output.glob('wma-harden-*'):
                if path.is_dir():
                    import shutil
                    shutil.rmtree(path)
            (output / 'transform_complete.log').unlink()
            with patch.object(launcher, '_run_slicer', side_effect=fake_worker) as worker:
                with patch('sys.argv', ['wm_harden_transform.py'] + args):
                    launcher.main()
                self.assertEqual(worker.call_count, 2)
                self.assertTrue((output / 'transform_complete.log').is_file())
                self.assertFalse(list(output.glob('wma-harden-*')))


if __name__ == '__main__':
    unittest.main()
