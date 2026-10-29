"""Run in native SlicerDMRI: compare CLI hardening output and MRML lifetime."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types
from unittest.mock import patch

import numpy as np
import slicer
import vtk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
sys.path.insert(0, str(ROOT / 'bin'))
import whitematteranalysis as wma
import wm_separate_clusters_by_hemisphere as separation
from test_output_pipeline import fixture


def main():
    previous = types.ModuleType('previous_hardening')
    previous.__file__ = str(ROOT / 'bin/harden_transform_with_slicer.py')
    source = subprocess.check_output(['git', '-C', str(ROOT), 'show',
        '60c7b2d:bin/harden_transform_with_slicer.py'], text=True)
    exec(compile(source, previous.__file__, 'exec'), previous.__dict__)
    spec = importlib.util.spec_from_file_location('current_hardening', previous.__file__)
    current = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(current)
    results = []
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        inputs = root / 'input'
        inputs.mkdir()
        data = fixture()
        for index in range(64):
            wma.io.write_polydata(data, str(inputs / ('cluster_%05d.vtp' % index)))
        empty = vtk.vtkPolyData()
        empty.SetPoints(vtk.vtkPoints())
        wma.io.write_polydata(empty, str(inputs / 'cluster_00800.vtp'))
        for nonrigid in (False, True):
            if nonrigid:
                coefficients = np.zeros(3 * 3 * 3 * 3)
                coefficients[::3] = 2.
                raw = wma.register_two_subjects_nonrigid_bsplines.convert_transform_to_vtk(coefficients)
                transform = slicer.vtkOrientedBSplineTransform()
                transform.SetCoefficientData(raw.GetCoefficientData())
                transform.SetBorderMode(raw.GetBorderMode())
            else:
                transform = vtk.vtkTransform()
                transform.Translate(2., -3., 4.)
                transform.RotateY(12.)
            node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLTransformNode')
            node.SetAndObserveTransformToParent(transform)
            node.GetTransformFromParent().Update()
            transform_path = root / ('nonrigid.h5' if nonrigid else 'affine.tfm')
            assert slicer.util.saveNode(node, str(transform_path))
            for inverse in ('0', '1'):
                outputs = []
                for label, module in (('previous', previous), ('current', current)):
                    slicer.mrmlScene.Clear(0)
                    sentinel = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode', 'unrelated')
                    before = slicer.mrmlScene.GetNumberOfNodes()
                    output = root / ('%s_%s_%s' % (label, nonrigid, inverse))
                    output.mkdir()
                    args = argparse.Namespace(polydata=str(inputs), transform=str(transform_path),
                                              inverse=inverse, outdir=str(output),
                                              partition_index=0, partition_count=1,
                                              split_output_directory=None)
                    with patch.object(module, '_parse_args', return_value=args), \
                            patch.object(slicer.util, 'loadTransform', wraps=slicer.util.loadTransform) as load:
                        start = time.perf_counter()
                        module.main()
                        seconds = time.perf_counter() - start
                    after = slicer.mrmlScene.GetNumberOfNodes()
                    if label == 'current':
                        assert load.call_count == 1, load.call_count
                        assert after == before, (before, after)
                        assert sentinel.GetScene() == slicer.mrmlScene
                    results.append(dict(implementation=label, nonrigid=nonrigid,
                        inverse=inverse, seconds=seconds, transform_loads=load.call_count,
                        scene_nodes_added=after - before))
                    outputs.append(output)
                for path in inputs.glob('*.vtp'):
                    assert (outputs[0] / path.name).read_bytes() == (outputs[1] / path.name).read_bytes(), path.name

        split_inputs = root / 'split_input'
        split_inputs.mkdir()
        labeled = fixture()
        point_labels = vtk.vtkDoubleArray()
        point_labels.SetName('HemisphereLocation')
        point_labels.SetNumberOfTuples(labeled.GetNumberOfPoints())
        point_labels.Fill(0.)
        line_labels = (1., 2., 3., 1., 2.)
        labeled.GetLines().InitTraversal()
        ids = vtk.vtkIdList()
        for index, label in enumerate(line_labels):
            labeled.GetLines().GetNextCell(ids)
            point_labels.SetValue(ids.GetId(ids.GetNumberOfIds() - 1), label)
        labeled.GetPointData().AddArray(point_labels)
        wma.io.write_polydata(labeled, str(split_inputs / 'cluster_00001.vtp'))
        empty = vtk.vtkPolyData()
        empty.SetPoints(vtk.vtkPoints())
        empty.SetLines(vtk.vtkCellArray())
        empty_labels = vtk.vtkDoubleArray()
        empty_labels.SetName('HemisphereLocation')
        empty.GetPointData().AddArray(empty_labels)
        wma.io.write_polydata(empty, str(split_inputs / 'cluster_00002.vtp'))

        split_transform = root / 'split_transform.tfm'
        transform_node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLTransformNode')
        transform = vtk.vtkTransform()
        transform.Translate(2., -3., 4.)
        transform_node.SetAndObserveTransformToParent(transform)
        transform_node.GetTransformFromParent().Update()
        assert slicer.util.saveNode(transform_node, str(split_transform))
        slicer.mrmlScene.RemoveNode(transform_node)

        split_output = root / 'split_transformed'
        split_actual = root / 'split_actual'
        split_expected = root / 'split_expected'
        split_args = argparse.Namespace(polydata=str(split_inputs), transform=str(split_transform),
            inverse='0', outdir=str(split_output), partition_index=0, partition_count=1,
            split_output_directory=str(split_actual))
        for folder in ('tracts_left_hemisphere', 'tracts_right_hemisphere', 'tracts_commissural'):
            (split_actual / folder).mkdir(parents=True)
            (split_expected / folder).mkdir(parents=True)
        with patch.object(current, '_parse_args', return_value=split_args):
            current.main()
        for transformed in split_output.glob('*.vtp'):
            pd = wma.io.read_polydata(str(transformed))
            found, labels = separation.read_mask_location_from_vtk(pd)
            assert found, transformed
            separation.split_polydata(pd, labels, transformed.name, str(split_expected))
            for folder in ('tracts_left_hemisphere', 'tracts_right_hemisphere', 'tracts_commissural'):
                actual = split_actual / folder / transformed.name
                expected = split_expected / folder / transformed.name
                assert actual.read_bytes() == expected.read_bytes(), (folder, transformed.name)
    print('HARDENING_RESULTS', json.dumps(results), flush=True)
    print('SLICER_HARDENING_VERIFIED', flush=True)


try:
    main()
except Exception:
    import traceback
    traceback.print_exc()
    sys.exit(1)
sys.exit(0)
