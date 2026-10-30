"""Run in Slicer: compare current hardening with helper at commit 3f358f1."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import slicer
import vtk
from vtk.util.numpy_support import numpy_to_vtk

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'bin'))
sys.path.insert(0, str(ROOT / 'tests'))
import whitematteranalysis as wma
import wm_separate_clusters_by_hemisphere as separation
from test_cli_file_stages import assert_equal


def load_helper(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(labels, space):
    data = vtk.vtkPolyData()
    points, lines = vtk.vtkPoints(), vtk.vtkCellArray()
    points.SetDataTypeToDouble()
    for fiber, label in enumerate(labels):
        lines.InsertNextCell(3)
        for point in range(3):
            lines.InsertCellPoint(points.InsertNextPoint(10. + fiber, point, fiber - point))
    data.SetPoints(points)
    data.SetLines(lines)
    for name, values in (
            ('scalar', np.arange(points.GetNumberOfPoints(), dtype=np.float64)),
            ('rgb', np.arange(points.GetNumberOfPoints() * 3, dtype=np.uint8).reshape(-1, 3)),
            ('tensor', np.arange(points.GetNumberOfPoints() * 9, dtype=np.float32).reshape(-1, 9)),
            ('HemisphereLocation', np.repeat(labels, 3).astype(np.float32))):
        array = numpy_to_vtk(values, deep=True)
        array.SetName(name)
        data.GetPointData().AddArray(array)
    data.GetPointData().SetActiveScalars('scalar')
    data.GetPointData().SetActiveTensors('tensor')
    if space is not None:
        metadata = vtk.vtkStringArray()
        metadata.SetName('SPACE')
        metadata.InsertNextValue(space)
        data.GetFieldData().AddArray(metadata)
    note = vtk.vtkStringArray()
    note.SetName('fixture')
    note.InsertNextValue('preserve metadata')
    data.GetFieldData().AddArray(note)
    return data


def run(module, source, transform, output, split=None, inverse='1'):
    argv = ['harden_transform_with_slicer.py', str(source), str(transform), inverse, str(output)]
    if split is not None:
        argv += ['--split-output-directory', str(split)]
    with patch.object(sys, 'argv', argv):
        module.main()


def same_directory(left, right, names):
    for name in names:
        assert_equal(unittest.TestCase(), wma.io.read_polydata(str(left / name)),
                     wma.io.read_polydata(str(right / name)))


def save_nonlinear_transform(path):
    image = vtk.vtkImageData()
    image.SetDimensions(5, 5, 5)
    image.SetOrigin(-10., -10., -10.)
    image.SetSpacing(10., 10., 10.)
    displacement = vtk.vtkFloatArray()
    displacement.SetNumberOfComponents(3)
    displacement.SetNumberOfTuples(5 * 5 * 5)
    for z in range(5):
        for y in range(5):
            for x in range(5):
                point = (x * 10. - 10., y * 10. - 10., z * 10. - 10.)
                displacement.SetTuple3(x + 5 * (y + 5 * z),
                    .01 * point[0] * point[1], .01 * point[1] * point[2],
                    .01 * point[2] * point[0])
    image.GetPointData().SetScalars(displacement)
    grid = slicer.vtkOrientedGridTransform()
    grid.SetDisplacementGridData(image)
    grid.SetInterpolationModeToLinear()
    samples = [(0., 0., 0.), (10., 10., 0.), (20., 20., 0.)]
    displacements = [np.subtract(grid.TransformPoint(point), point) for point in samples]
    assert not np.allclose(displacements[1] - displacements[0],
                           displacements[2] - displacements[1])
    node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLTransformNode')
    node.SetAndObserveTransformToParent(grid)
    assert slicer.util.saveNode(node, str(path))
    storage = node.GetStorageNode()
    slicer.mrmlScene.RemoveNode(node)
    if storage is not None and storage.GetScene() == slicer.mrmlScene:
        slicer.mrmlScene.RemoveNode(storage)


def main():
    current = load_helper('current_harden_helper', ROOT / 'bin/harden_transform_with_slicer.py')
    with tempfile.TemporaryDirectory(prefix='wma harden regression ') as temporary:
        root = Path(temporary)
        frozen = root / 'original_harden_transform_with_slicer.py'
        frozen.write_bytes(subprocess.check_output(
            ['git', 'show', '3f358f1:bin/harden_transform_with_slicer.py'], cwd=ROOT))
        original = load_helper('original_harden_helper', frozen)
        original._prepare_split_modules = lambda: (wma, separation)
        current._prepare_split_modules = lambda: (wma, separation)

        source = root / 'input files'
        source.mkdir()
        wma.io.write_polydata(fixture([1, 2, 3], 'LPS'), str(source / 'cluster_00001.vtk'))
        wma.io.write_polydata(fixture([], 'RAS'), str(source / 'cluster_00002.vtp'))
        wma.io.write_polydata(fixture([0, 1, 2, 3], 'LPS'), str(source / 'cluster_00003.vtp'))
        wma.io.write_polydata(fixture([1, 2, 3], None), str(source / 'cluster_00004.vtp'))
        wma.io.write_polydata(fixture([1, 2, 3], 'RAS'), str(source / 'cluster_00005.vtp'))

        transform = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLinearTransformNode')
        vtk_transform = vtk.vtkTransform()
        vtk_transform.Translate(3., -2., 5.)
        transform.SetAndObserveTransformToParent(vtk_transform)
        transform_path = root / 'inverse transform.tfm'
        assert slicer.util.saveNode(transform, str(transform_path))
        storage = transform.GetStorageNode()
        slicer.mrmlScene.RemoveNode(transform)
        slicer.mrmlScene.RemoveNode(storage)
        names = sorted(path.name for path in source.iterdir())
        before = {node.GetID() for node in slicer.util.getNodesByClass('vtkMRMLNode')}

        for label, module in (('original', original), ('current', current)):
            no_split = root / (label + ' no split')
            no_split.mkdir()
            run(module, source, transform_path, no_split)
            split = root / (label + ' split')
            split.mkdir()
            (split / 'hardened').mkdir()
            for folder in ('tracts_left_hemisphere', 'tracts_right_hemisphere', 'tracts_commissural'):
                (split / folder).mkdir()
            run(module, source, transform_path, split / 'hardened', split)
            assert sorted(path.name for path in (split / 'hardened').iterdir()) == names
            if label == 'original':
                expected_no_split, expected_split = no_split, split
            else:
                same_directory(expected_no_split, no_split, names)
                same_directory(expected_split / 'hardened', split / 'hardened', names)
                for folder in ('tracts_left_hemisphere', 'tracts_right_hemisphere', 'tracts_commissural'):
                    same_directory(expected_split / folder, split / folder, names)
                missing = split / 'tracts_right_hemisphere' / names[-1]
                missing.unlink()
                run(current, source, transform_path, split / 'hardened', split)
                assert_equal(unittest.TestCase(), wma.io.read_polydata(str(expected_split / 'tracts_right_hemisphere' / names[-1])),
                             wma.io.read_polydata(str(missing)))

        # Preserve native tensor/normal handling for nonlinear transforms too.
        landmarks, targets = vtk.vtkPoints(), vtk.vtkPoints()
        for point, displacement in (((0, 0, 0), (0, 0, 0)), ((20, 0, 0), (1, 0, 0)),
                                    ((0, 20, 0), (0, 2, 0)), ((0, 0, 20), (0, 0, 1))):
            landmarks.InsertNextPoint(point)
            targets.InsertNextPoint([p + d for p, d in zip(point, displacement)])
        warp = vtk.vtkThinPlateSplineTransform()
        warp.SetSourceLandmarks(landmarks)
        warp.SetTargetLandmarks(targets)
        nonlinear = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLTransformNode')
        nonlinear.SetAndObserveTransformToParent(warp)
        fiber = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLFiberBundleNode')
        storage = fiber.CreateDefaultStorageNode()
        slicer.mrmlScene.AddNode(storage)
        fiber.SetAndObserveStorageNodeID(storage.GetID())
        old_warp, new_warp = root / 'original warp', root / 'current warp'
        old_warp.mkdir()
        new_warp.mkdir()
        logic = slicer.vtkSlicerTransformLogic()
        for name in names:
            original.harden_transform(str(source / name), nonlinear, logic, str(old_warp))
            current.harden_transform(str(source / name), nonlinear, logic, fiber, storage,
                                     str(new_warp))
        same_directory(old_warp, new_warp, names)
        current._remove_fiber_node(fiber)
        slicer.mrmlScene.RemoveNode(nonlinear)

        nonlinear_path = root / 'nonlinear transform.h5'
        save_nonlinear_transform(nonlinear_path)
        for inverse in ('0', '1'):
            for label, module in (('original nonlinear', original),
                                  ('current nonlinear', current)):
                run_name = '%s nonlinear inverse%s' % (label, inverse)
                no_split = root / (run_name + ' no split')
                no_split.mkdir()
                run(module, source, nonlinear_path, no_split, inverse=inverse)
                split = root / (run_name + ' split')
                split.mkdir()
                (split / 'hardened').mkdir()
                for folder in ('tracts_left_hemisphere', 'tracts_right_hemisphere', 'tracts_commissural'):
                    (split / folder).mkdir()
                run(module, source, nonlinear_path, split / 'hardened', split, inverse=inverse)
                assert sorted(path.name for path in (split / 'hardened').iterdir()) == names
                if label == 'original nonlinear':
                    expected_nonlinear_no_split = no_split
                    expected_nonlinear_split = split
                else:
                    same_directory(expected_nonlinear_no_split, no_split, names)
                    same_directory(expected_nonlinear_split / 'hardened', split / 'hardened', names)
                    for folder in ('tracts_left_hemisphere', 'tracts_right_hemisphere', 'tracts_commissural'):
                        same_directory(expected_nonlinear_split / folder, split / folder, names)

        after = {node.GetID() for node in slicer.util.getNodesByClass('vtkMRMLNode')}
        assert after == before, (before, after)
        assert not slicer.mrmlScene.IsBatchProcessing()
    print('HARDENING_REGRESSION_VERIFIED', flush=True)


try:
    main()
except Exception:
    import traceback
    traceback.print_exc()
    sys.exit(1)
sys.exit(0)
