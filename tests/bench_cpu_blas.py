"""Compare scoped CPU BLAS limits with the preceding atlas-labeling backend."""
import argparse
import json
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import whitematteranalysis as wma
from threadpoolctl import ThreadpoolController
from test_output_pipeline import baseline_module


def verify(actual, expected):
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_allclose(actual[3], expected[3], rtol=1e-12, atol=1e-12)
    before, after = expected[0], actual[0]
    np.testing.assert_array_equal(vtk_to_numpy(before.GetPoints().GetData()),
                                  vtk_to_numpy(after.GetPoints().GetData()))
    for method in ('GetOffsetsArray', 'GetConnectivityArray'):
        np.testing.assert_array_equal(vtk_to_numpy(getattr(before.GetLines(), method)()),
                                      vtk_to_numpy(getattr(after.GetLines(), method)()))
    changed = {}
    for index in range(before.GetCellData().GetNumberOfArrays()):
        reference = before.GetCellData().GetArray(index)
        observed = after.GetCellData().GetArray(reference.GetName())
        a, b = vtk_to_numpy(reference), vtk_to_numpy(observed)
        assert a.dtype == b.dtype and a.shape == b.shape, reference.GetName()
        if reference.GetName() in ('EmbeddingCoordinate', 'TotalFiberSimilarity', 'MeasuredFiberSimilarity'):
            np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-12)
            if a.dtype == np.float32:
                x, y = a.view(np.int32).astype(np.int64), b.view(np.int32).astype(np.int64)
                x, y = np.where(x < 0, -2147483648-x, x), np.where(y < 0, -2147483648-y, y)
                ulps = int(np.max(np.abs(x-y))) if x.size else 0
                assert ulps <= 1, (reference.GetName(), ulps)
            if not np.array_equal(a, b):
                changed[reference.GetName()] = int(np.count_nonzero(a != b))
        else:
            np.testing.assert_array_equal(a, b)
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input')
    parser.add_argument('atlas')
    parser.add_argument('--baseline', default='3891846')
    parser.add_argument('--fibers', type=int, default=2048)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    original = baseline_module('whitematteranalysis.gpu._blasbaseline', 'gpu/cpu_backend.py', args.baseline)
    current = wma.gpu.cpu_backend
    atlas = wma.cluster.load_atlas(args.atlas, 'atlas')
    subject = wma.filter.downsample(wma.io.read_polydata(args.input), args.fibers,
                                    random_seed=1000, verbose=False)
    controller = ThreadpoolController()
    before = controller.info()

    def classify(backend):
        data = vtk.vtkPolyData()
        data.ShallowCopy(subject)
        with patch.object(wma.gpu, 'cpu_backend', backend):
            return wma.cluster.spectral_atlas_label(data, atlas, args.workers, use_gpu=False)

    expected = classify(original)
    changed = verify(classify(current), expected)
    samples = [[], []]
    for repeat in range(args.repeat):
        for index in (range(2) if repeat % 2 == 0 else reversed(range(2))):
            start = time.perf_counter()
            actual = classify((original, current)[index])
            samples[index].append(time.perf_counter()-start)
            differences = verify(actual, expected)
            if differences:
                changed = differences
            assert controller.info() == before, 'BLAS settings were not restored'
    report = dict(baseline=args.baseline, fibers=subject.GetNumberOfLines(), workers=args.workers,
        median_seconds=[statistics.median(s) for s in samples], seconds=samples,
        labels_geometry_and_colors_exact=True, changed_float32_attributes=changed,
        embeddings_rtol=1e-12, blas_settings_restored=True,
        original_blas=[dict(prefix=p['prefix'], threads=p['num_threads']) for p in before if p['user_api']=='blas'])
    Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
