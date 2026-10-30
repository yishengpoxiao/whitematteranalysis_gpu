#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import time

_ENTRY_TIME = time.perf_counter()
_ENTRY_MONOTONIC = time.monotonic()

import slicer
import vtk
import os
import glob
import json
import shutil
import sys
from pathlib import Path

_WMA = None
_SEPARATION = None
_TIMINGS = {}
_PROFILE_OUTPUT = None
_FILE_COUNT = 0


def _timed(name, function, *args):
    started = time.perf_counter()
    try:
        return function(*args)
    finally:
        item = _TIMINGS.setdefault(name, {'seconds': 0., 'calls': 0})
        item['seconds'] += time.perf_counter() - started
        item['calls'] += 1


def _prepare_split_modules():
    global _WMA, _SEPARATION
    if _WMA is None:
        repo_root = str(Path(__file__).resolve().parents[1])
        for path in (repo_root, str(Path(__file__).resolve().parent)):
            if path not in sys.path:
                sys.path.insert(0, path)
        import whitematteranalysis as wma
        import wm_separate_clusters_by_hemisphere as separation
        _WMA, _SEPARATION = wma, separation
    return _WMA, _SEPARATION


def _prepare_native_modules():
    # The launcher skips GUI module discovery. Use the paths supplied by Slicer
    # and its installed extensions rather than version/platform-specific paths.
    for path in slicer.app.moduleManager().factoryManager().searchPaths:
        if path not in sys.path:
            sys.path.append(path)
    from vtkSlicerTractographyDisplayModuleLogicPython import vtkSlicerFiberBundleLogic
    from vtkSlicerTransformsModuleLogicPython import vtkSlicerTransformLogic
    fiber_logic = vtkSlicerFiberBundleLogic()
    fiber_logic.SetMRMLScene(slicer.mrmlScene)
    return fiber_logic, vtkSlicerTransformLogic()


def _load_transform(filename):
    # Use the same MRML storage reader without constructing the GUI I/O module.
    node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLTransformNode')
    storage = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLTransformStorageNode')
    node.SetAndObserveStorageNodeID(storage.GetID())
    storage.SetFileName(filename)
    if not storage.ReadData(node):
        slicer.mrmlScene.RemoveNode(node)
        slicer.mrmlScene.RemoveNode(storage)
        raise RuntimeError('Could not load transform file: ' + filename)
    return node


def _wait_for_input(status):
    # The producer publishes atomically after it has closed every input file.
    while not os.path.isfile(status):
        time.sleep(.05)
    if Path(status).read_text() != 'ready':
        raise RuntimeError('Input preparation failed; no files were processed.')


def split_polydata(polydata, polydata_path, split_output_dir):
    _, separation = _prepare_split_modules()
    found, labels = separation.read_mask_location_from_vtk(polydata)
    if not found:
        raise RuntimeError('HemisphereLocation is missing from transformed file: ' + str(polydata_path))
    if (labels == 0).sum() > 1:
        raise RuntimeError('Hemisphere labels are incomplete: ' + str(polydata_path))
    separation.split_polydata(polydata, labels, os.path.basename(polydata_path), split_output_dir)


def harden_transform(polydata, transform_node, logic, fiber_node, storage_node,
                     outdir, split_output_dir=None):
    
    output_name = os.path.join(outdir, os.path.basename(polydata))
    
    if os.path.exists(output_name):
        if split_output_dir is None:
            return
        split_files = [os.path.join(split_output_dir, folder, os.path.basename(polydata))
                       for folder in ('tracts_left_hemisphere', 'tracts_right_hemisphere',
                                      'tracts_commissural')]
        if all(os.path.isfile(path) for path in split_files):
            return
        wma, _ = _prepare_split_modules()
        _timed('split', split_polydata, wma.io.read_polydata(output_name),
               output_name, split_output_dir)
        return
    
    # Storage reads reuse the fiber/display nodes rather than recreating them
    # through the GUI I/O manager for every cluster. Clear the previous input.
    fiber_node.SetAndObservePolyData(None)
    storage_node.SetCoordinateSystem(storage_node.CoordinateSystemRAS)
    storage_node.SetFileName(str(polydata))
    if not _timed('read', storage_node.ReadData, fiber_node):
        raise RuntimeError('Could not load polydata file: ' + polydata)
    data = fiber_node.GetPolyData()
    if data.GetNumberOfCells() == 0:
        _timed('write', shutil.copyfile, polydata, output_name)
    else:
        fiber_node.SetAndObserveTransformNodeID(transform_node.GetID())
        if not _timed('harden', logic.hardenTransform, fiber_node):
            raise RuntimeError('Could not harden transform: ' + polydata)
        data = fiber_node.GetPolyData()
        if split_output_dir is None or os.path.splitext(output_name)[1] != '.vtp':
            storage_node.SetFileName(output_name)
            # The GUI file writer selects RAS even if the input was LPS.
            storage_node.SetCoordinateSystem(storage_node.CoordinateSystemRAS)
            saved = _timed('write', storage_node.WriteData, fiber_node)
            if not saved:
                raise RuntimeError('Could not save transformed polydata: ' + output_name)
        else:
            # Match Slicer's RAS output metadata while using the same fast,
            # lossless writer as the split files. Tensor hardening stays native.
            space = vtk.vtkStringArray()
            space.SetName('SPACE')
            space.InsertNextValue('RAS')
            data.GetFieldData().AddArray(space)
            wma, _ = _prepare_split_modules()
            _timed('write', wma.io.write_polydata, data, output_name)
    if split_output_dir is not None:
        _timed('split', split_polydata, data, output_name, split_output_dir)


def _remove_fiber_node(fiber_node):
    # Storage reads create display/properties nodes once. Release all of them.
    owned = [fiber_node.GetNthDisplayNode(i)
             for i in range(fiber_node.GetNumberOfDisplayNodes())]
    owned.extend([display.GetDiffusionTensorDisplayPropertiesNode()
                  for display in list(owned)])
    owned.append(fiber_node.GetStorageNode())
    slicer.mrmlScene.RemoveNode(fiber_node)
    for node in owned:
        if node is not None and node.GetScene() == slicer.mrmlScene:
            slicer.mrmlScene.RemoveNode(node)


def _build_arg_parser():

    parser = argparse.ArgumentParser(
        description="Harden transform with Slicer.",
        epilog="Written by Fan Zhang, fzhang@bwh.harvard.edu")
    parser.add_argument("-v", "--version",
        action="version", default=argparse.SUPPRESS,
        version='1.0',
        help="Show program's version number and exit")
    parser.add_argument(
        'polydata',
        help='')
    parser.add_argument(
        'transform',
        help='')
    parser.add_argument(
        'inverse',
        help='')
    parser.add_argument(
        'outdir',
        help='')
    parser.add_argument('partition_index', type=int, nargs='?', default=0)
    parser.add_argument('partition_count', type=int, nargs='?', default=1)
    parser.add_argument('--split-output-directory', default=None)
    parser.add_argument('--profile-output', default=None)
    parser.add_argument('--input-status', default=None)

    return parser


def _parse_args(parser):

    return parser.parse_args()


def main():
    global _PROFILE_OUTPUT, _FILE_COUNT
    parser = _build_arg_parser()
    args = _parse_args(parser)
    _PROFILE_OUTPUT = args.profile_output
    # Each process handles independent files; avoid nested native thread pools.
    vtk.vtkMultiThreader.SetGlobalMaximumNumberOfThreads(1)
    vtk.vtkMultiThreader.SetGlobalDefaultNumberOfThreads(1)
    fiber_logic, logic = _timed('native_modules', _prepare_native_modules)
    if args.split_output_directory is not None:
        _timed('imports', _prepare_split_modules)
    if args.input_status is not None:
        _timed('input_wait', _wait_for_input, args.input_status)

    if os.path.isfile(args.polydata):
        input_polydatas = [args.polydata]
    elif os.path.isdir(args.polydata):
        input_mask = os.path.join(args.polydata, "*.vtk")
        input_mask2 = os.path.join(args.polydata, "*.vtp")
        input_pd_fnames = glob.glob(input_mask) + glob.glob(input_mask2)
        input_polydatas = sorted(input_pd_fnames)
    
    else:
        raise ValueError('Input is not a file or directory: ' + args.polydata)

    input_polydatas = input_polydatas[args.partition_index::args.partition_count]
    _FILE_COUNT = len(input_polydatas)

    transform_node = _timed('transform_load', _load_transform, str(args.transform))
    slicer.mrmlScene.StartState(slicer.mrmlScene.BatchProcessState)
    fiber_node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLFiberBundleNode')
    storage_node = fiber_node.CreateDefaultStorageNode()
    slicer.mrmlScene.AddNode(storage_node)
    fiber_node.SetAndObserveStorageNodeID(storage_node.GetID())
    try:
        if args.inverse == '1':
            transform_node.Inverse()
        for polydata in input_polydatas:
            harden_transform(polydata, transform_node, logic, fiber_node, storage_node,
                             args.outdir, args.split_output_directory)
    finally:
        _remove_fiber_node(fiber_node)
        storage = transform_node.GetStorageNode()
        slicer.mrmlScene.RemoveNode(transform_node)
        if storage is not None and storage.GetScene() == slicer.mrmlScene:
            slicer.mrmlScene.RemoveNode(storage)
        slicer.mrmlScene.EndState(slicer.mrmlScene.BatchProcessState)

if __name__ == '__main__':
    success = False
    try:
        main()
        success = True
    except Exception:
        import sys
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if _PROFILE_OUTPUT is not None:
            launched = float(os.environ.get('WMA_HARDEN_LAUNCH_TIME', _ENTRY_MONOTONIC))
            with open(_PROFILE_OUTPUT, 'w') as stream:
                json.dump({'success': success, 'files': _FILE_COUNT,
                           'startup_s': _ENTRY_MONOTONIC - launched,
                           'work_s': time.perf_counter() - _ENTRY_TIME,
                           'phases': _TIMINGS}, stream, indent=2)
