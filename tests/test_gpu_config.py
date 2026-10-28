"""Backend selection honors explicit GPU requests."""
import argparse
import importlib
import os
import unittest
from unittest.mock import patch

gpu_config = importlib.import_module('whitematteranalysis.gpu.config')


class GPUConfigTests(unittest.TestCase):
    def setUp(self):
        self.settings = {name: getattr(gpu_config.config, name)
                         for name in ('requested', 'float32', 'devices', 'tile_rows')}

    def tearDown(self):
        for name, value in self.settings.items():
            setattr(gpu_config.config, name, value)

    def test_explicit_requests_fail_clearly_without_cuda(self):
        with patch.object(gpu_config.config, 'cuda_available', return_value=False) as available:
            for request in (True, 'on'):
                with self.subTest(request=request), self.assertRaisesRegex(
                        RuntimeError, 'explicitly requested.*CUDA is unavailable'):
                    gpu_config.use_gpu(request)
            gpu_config.set_gpu('on')
            with self.assertRaisesRegex(RuntimeError, 'CUDA is unavailable'):
                gpu_config.use_gpu()
            self.assertEqual(available.call_count, 3)

    def test_auto_falls_back_or_uses_available_cuda(self):
        gpu_config.set_gpu('auto')
        with patch.object(gpu_config.config, 'cuda_available', return_value=False) as available:
            self.assertFalse(gpu_config.use_gpu())
            available.assert_called_once_with()
        with patch.object(gpu_config.config, 'cuda_available', return_value=True):
            self.assertTrue(gpu_config.use_gpu())

    def test_off_does_not_probe_cuda(self):
        with patch.object(gpu_config.config, 'cuda_available', side_effect=AssertionError('probed')) as available:
            self.assertFalse(gpu_config.use_gpu(False))
            gpu_config.set_gpu('off')
            self.assertFalse(gpu_config.use_gpu())
            available.assert_not_called()

    def test_environment_and_cli_modes(self):
        with patch.dict(os.environ, {'WMA_GPU': 'on'}):
            gpu_config._from_env()
        self.assertIs(gpu_config.config.requested, True)

        with patch.dict(os.environ):
            os.environ.pop('WMA_GPU', None)
            gpu_config._from_env()
        self.assertIsNone(gpu_config.config.requested)

        parser = argparse.ArgumentParser()
        gpu_config.add_gpu_arguments(parser)
        self.assertEqual(parser.parse_args([]).gpuMode, 'auto')
        with patch.object(gpu_config.config, 'cuda_available', return_value=True):
            for mode, expected in (('on', True), ('off', False)):
                args = parser.parse_args(['-gpu', mode])
                gpu_config.apply_gpu_arguments(args)
                self.assertIs(gpu_config.config.requested, expected)
        with patch.object(gpu_config.config, 'cuda_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'CUDA is unavailable'):
                gpu_config.apply_gpu_arguments(parser.parse_args(['-gpu', 'on']))


if __name__ == '__main__':
    unittest.main()
