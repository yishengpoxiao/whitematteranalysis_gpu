"""Bounded, deterministic geometry-extraction timings; not a full pipeline benchmark."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

sys.path.append(str(Path(__file__).resolve().parents[1] / 'Modules/_vendor'))
import whitematteranalysis as wma
import WMAtlasLabelGPUCore as core


def fixture(count):
    points = np.zeros((count, 20, 3), dtype=np.float64)
    points[:, :, 0] = np.where(np.arange(count)[:, None] % 3 == 0, -20., 20.)
    points[np.arange(count) % 3 == 2, 10:, 0] = -20.
    points[:, :, 1] = np.arange(20)
    points[:, :, 2] = (np.arange(count) % 100)[:, None]
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(points.reshape(-1, 3), deep=True))
    lines = vtk.vtkCellArray()
    lines.SetData(numpy_to_vtkIdTypeArray(np.arange(0, count * 20 + 1, 20), deep=True),
                  numpy_to_vtkIdTypeArray(np.arange(count * 20), deep=True))
    pd = vtk.vtkPolyData()
    pd.SetPoints(vtk_points)
    pd.SetLines(lines)
    scalar = numpy_to_vtk(np.arange(count * 20, dtype=np.float64), deep=True)
    scalar.SetName('measure')
    pd.GetPointData().AddArray(scalar)
    labels = np.arange(count) % 800
    clusters = numpy_to_vtk(labels, deep=True)
    clusters.SetName('ClusterNumber')
    pd.GetCellData().AddArray(clusters)
    return pd, labels


def digest(items):
    result = hashlib.sha256()
    for name, pd in items:
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetInputData(pd)
        writer.WriteToOutputStringOn()
        writer.Write()
        result.update(name.encode())
        result.update(writer.GetOutputString().encode('utf-8'))
    return result.hexdigest()


def timed(call):
    samples = []
    for _ in range(3):
        start = time.perf_counter()
        result = call()
        samples.append(time.perf_counter() - start)
    return result, statistics.median(samples)


def benchmark(count):
    # Exclude lazy module imports from geometry-operation timings on both paths.
    wma.cluster.mask_all_clusters
    pd, labels = fixture(count)
    old, old_seconds = timed(lambda: [wma.filter.mask(pd, labels == c,
        preserve_point_data=True, preserve_cell_data=True, verbose=False) for c in range(800)])
    new, new_seconds = timed(lambda: core.extract_clusters(pd, labels, 800))
    assert digest((str(i), pd) for i, pd in enumerate(old)) == digest((str(i), pd) for i, pd in enumerate(new))
    del new
    with tempfile.TemporaryDirectory() as directory:
        paths = []
        for i in range(40):
            path = Path(directory) / ('T_CC%d.mrml' % i if i < 8 else 'T_tract%d.mrml' % i)
            path.write_text(''.join('<FiberBundleStorage fileName="cluster_%05d.vtp"/>\n' % n
                                    for n in range(i * 20 + 1, i * 20 + 21)))
            paths.append(str(path))
        def legacy():
            separated = core.separate_clusters_by_hemisphere(old)
            return core.append_clusters_to_anatomical_tracts(old, paths, separated)
        expected, append_seconds = timed(legacy)
        actual, direct_seconds = timed(lambda: core.extract_anatomical_tracts(pd, labels, paths))
        assert digest(expected) == digest(actual)
    return dict(fibers=count, points_per_fiber=20, clusters=800, anatomical_tracts=len(actual), repeats=3,
                cluster_split_seconds=dict(previous=old_seconds, shared_geometry=new_seconds),
                anatomical_seconds=dict(previous_including_split=old_seconds + append_seconds,
                                        direct_selection=direct_seconds), byte_identical=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fibers', type=int, default=40000)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report = json.dumps(benchmark(args.fibers), indent=2)
    if args.report:
        args.report.write_text(report + '\n')
    print(report)
