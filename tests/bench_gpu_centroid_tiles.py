"""Compare CUDA centroid tiles and optional real tractography classification."""
import argparse
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import torch
import vtk
import whitematteranalysis as wma
from whitematteranalysis.gpu import distance_kernel
from bench_gpu_embedding import compare
from test_output_pipeline import baseline_module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default='cef12ca')
    parser.add_argument('--repeat', type=int, default=31)
    parser.add_argument('--input')
    parser.add_argument('--atlas')
    parser.add_argument('--fibers', type=int, default=100000)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.input and not args.atlas:
        parser.error('--input requires --atlas')
    old = baseline_module('whitematteranalysis.gpu._centroidtilebaseline',
                          'gpu/distance_kernel.py', args.baseline)
    functions = (old.centroid_distances, distance_kernel.centroid_distances)
    torch.manual_seed(83)
    report = dict(baseline=args.baseline, micro=[])
    for queries, centroids, dimensions in ((2048, 800, 10), (8192, 800, 10),
                                          (2048, 800, 20), (257, 37, 10)):
        q = torch.randn((queries, dimensions), dtype=torch.float64, device=wma.gpu.get_device())
        c = torch.randn((centroids, dimensions), dtype=torch.float64, device=q.device)
        outputs, timings = compare([lambda f=f: f(q, c).min(1) for f in functions], args.repeat)
        for a, b in zip(*outputs):
            torch.testing.assert_close(a, b, rtol=0., atol=0.)
        report['micro'].append(dict(queries=queries, centroids=centroids,
                                    dimensions=dimensions, timings=timings))
    if args.input:
        atlas = wma.cluster.load_atlas(args.atlas, 'atlas')
        subject = wma.filter.downsample(wma.io.read_polydata(args.input), args.fibers,
                                        random_seed=1000, verbose=False)

        def classify(function):
            data = vtk.vtkPolyData()
            data.ShallowCopy(subject)
            with patch.object(distance_kernel, 'centroid_distances', function):
                return wma.cluster.spectral_atlas_label(data, atlas, use_gpu=True)

        outputs, timings = compare([lambda f=f: classify(f) for f in functions], args.repeat)
        for index in (1, 2, 3):
            np.testing.assert_array_equal(outputs[0][index], outputs[1][index])
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / ('%d.vtp' % i) for i in range(2)]
            for output, path in zip(outputs, paths):
                wma.io.write_polydata(output[0], str(path))
            assert paths[0].read_bytes() == paths[1].read_bytes()
        report['classification'] = dict(fibers=subject.GetNumberOfLines(), timings=timings,
                                         labels_embeddings_and_vtp_bytes_exact=True)
    report['centroid_values_and_indices_exact'] = True
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
