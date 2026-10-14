"""Measure GPU embedding transfers and optional real-subject classification."""
import argparse
import json
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import numpy as np
import torch
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import whitematteranalysis as wma
from test_output_pipeline import baseline_module


def compare(functions, repeats):
    outputs = [function() for function in functions]
    samples = [[], []]
    for repeat in range(repeats):
        for index in (range(2) if repeat % 2 == 0 else reversed(range(2))):
            start = time.perf_counter()
            outputs[index] = functions[index]()
            samples[index].append(time.perf_counter() - start)
    return outputs, [dict(seconds=values, median_seconds=statistics.median(values)) for values in samples]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('atlas')
    parser.add_argument('--input')
    parser.add_argument('--fibers', type=int, default=8192)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--baseline', default='cffed63')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    original = baseline_module('whitematteranalysis.gpu._embeddingbaseline',
                               'gpu/linalg.py', args.baseline)
    current = wma.gpu.linalg
    atlas = wma.cluster.load_atlas(args.atlas, 'atlas')
    torch.manual_seed(42)
    similarities = torch.rand((args.fibers, atlas.nystrom_polydata.GetNumberOfLines()),
                              dtype=torch.float64, device=wma.gpu.get_device())
    def embedding(module):
        blocks = ((start, similarities[start:start+2048].t())
                  for start in range(0, args.fibers, 2048))
        return module.atlas_label_embedding_blocks(blocks, atlas, args.fibers)
    outputs, timings = compare([lambda: embedding(original), lambda: embedding(current)], args.repeat)
    for a, b in zip(*outputs):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(outputs[0][1], outputs[1][1])
    report = dict(baseline=args.baseline, fibers=args.fibers, embedding=timings)
    print('EMBEDDING', json.dumps(timings), flush=True)
    del similarities, outputs
    if args.input:
        subject = wma.filter.downsample(wma.io.read_polydata(args.input), args.fibers,
                                       random_seed=42, verbose=False)
        def classify(module):
            data = vtk.vtkPolyData()
            data.ShallowCopy(subject)
            with patch.object(wma.gpu, 'linalg', module):
                return wma.cluster.spectral_atlas_label(data, atlas, use_gpu=True)
        outputs, timings = compare([lambda: classify(original), lambda: classify(current)], args.repeat)
        np.testing.assert_array_equal(outputs[0][1], outputs[1][1])
        np.testing.assert_allclose(outputs[0][3], outputs[1][3], rtol=1e-12, atol=1e-12)
        for index in range(outputs[0][0].GetCellData().GetNumberOfArrays()):
            reference = outputs[0][0].GetCellData().GetArray(index)
            result = outputs[1][0].GetCellData().GetArray(reference.GetName())
            np.testing.assert_array_equal(vtk_to_numpy(reference), vtk_to_numpy(result))
        report['classification'] = timings
        print('CLASSIFICATION', json.dumps(timings), flush=True)
    report['labels_and_attributes_match'] = True
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
