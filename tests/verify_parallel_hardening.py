"""Run outside Slicer: compare native serial/parallel SlicerDMRI hardening outputs."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import whitematteranalysis as wma
from test_cli_file_stages import tractography


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--slicer', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root/'bin'))
    import wm_assess_cluster_location_by_hemisphere as assessment
    environment = dict(os.environ, PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE='1',
        OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1',
        WMA_VTP_COMPRESSION='lz4')
    with tempfile.TemporaryDirectory(prefix='wma native batch ') as temporary:
        directory = Path(temporary)
        environment['XDG_CONFIG_HOME'] = str(directory/'slicer-config')
        source = directory/'input bundles'
        source.mkdir()
        for index in range(9):
            data = tractography(0 if index == 3 else 9)
            labels = [index % 3 + 1] * data.GetNumberOfLines()
            assessment.write_mask_location_to_vtk(data, labels)
            wma.io.write_polydata(data, str(source/('cluster_%05d.vtp' % index)))
        transform = directory/'transform with spaces.tfm'
        transform.write_text('#Insight Transform File V1.0\n#Transform 0\n'
            'Transform: AffineTransform_double_3_3\nParameters: 1 0 0 0 1 0 0 0 1 2 -3 4\nFixedParameters: 0 0 0\n')

        def run(arguments):
            result = subprocess.run([sys.executable, str(root/'bin/wm_harden_transform.py')]+arguments,
                cwd=root, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            assert result.returncode == 0, result.stdout
            print(result.stdout, flush=True)

        serial, parallel = directory/'serial outputs', directory/'parallel outputs'
        run(['-i', '-t', str(transform), '-j', '1', str(source), str(serial), args.slicer])
        fused_splits = directory/'fused splits'
        run(['-i', '-t', str(transform), '-j', '2', str(source), str(parallel), args.slicer,
             '--split-output-directory', str(fused_splits)])
        names = {path.name for path in source.glob('*.vtp')}
        assert {path.name for path in serial.glob('*.vtp')} == names
        assert {path.name for path in parallel.glob('*.vtp')} == names
        for name in names:
            assert (serial/name).read_bytes() == (parallel/name).read_bytes(), name
        assert not list(parallel.glob('wma-harden-*'))
        separate = directory/'separate outputs'
        result = subprocess.run([sys.executable, str(root/'bin/wm_separate_clusters_by_hemisphere.py'),
            str(serial), str(separate), '-j', '2'], cwd=root, env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert result.returncode == 0, result.stdout
        for folder in ('tracts_left_hemisphere', 'tracts_right_hemisphere', 'tracts_commissural'):
            expected = {path.name for path in (separate/folder).glob('*.vtp')}
            actual = {path.name for path in (fused_splits/folder).glob('*.vtp')}
            assert actual == expected == names, (folder, actual, expected)
            for name in names:
                assert ((fused_splits/folder/name).read_bytes() ==
                        (separate/folder/name).read_bytes()), (folder, name)

        # A fresh Slicer process must recover a missing subset with no inherited
        # PYTHONPATH; the helper bootstraps the repository modules itself.
        missing_split = fused_splits/'tracts_left_hemisphere'/'cluster_00000.vtp'
        missing_split.unlink()
        run(['-i', '-t', str(transform), '-j', '1', str(source), str(parallel), args.slicer,
             '--split-output-directory', str(fused_splits)])
        for folder in ('tracts_left_hemisphere', 'tracts_right_hemisphere', 'tracts_commissural'):
            for name in names:
                assert ((fused_splits/folder/name).read_bytes() ==
                        (separate/folder/name).read_bytes()), (folder, name)

        multiple = directory/'individual transforms'
        multiple.mkdir()
        for index in range(9):
            (multiple/('%05d.tfm' % index)).write_text(transform.read_text().replace('1 2 -3 4', '1 %d -3 4' % index))
        output = directory/'multiple outputs'
        run(['-ts', str(multiple), '-j', '2', str(source), str(output), args.slicer])
        assert {path.name for path in output.glob('*.vtp')} == names
        # Different transforms must produce different geometry, not reuse one matrix.
        a = wma.io.read_polydata(str(output/'cluster_00000.vtp'))
        b = wma.io.read_polydata(str(output/'cluster_00001.vtp'))
        assert a.GetPoint(0) != b.GetPoint(0)
        print('NATIVE_PARALLEL_HARDENING_VERIFIED', flush=True)


if __name__ == '__main__':
    main()
