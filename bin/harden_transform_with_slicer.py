#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import slicer
import os
import glob
import shutil
import sys
from pathlib import Path

_WMA = None
_SEPARATION = None


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


def split_polydata(polydata, polydata_path, split_output_dir):
    _, separation = _prepare_split_modules()
    found, labels = separation.read_mask_location_from_vtk(polydata)
    if not found:
        raise RuntimeError('HemisphereLocation is missing from transformed file: ' + str(polydata_path))
    if (labels == 0).sum() > 1:
        raise RuntimeError('Hemisphere labels are incomplete: ' + str(polydata_path))
    separation.split_polydata(polydata, labels, os.path.basename(polydata_path), split_output_dir)


def harden_transform(polydata, transform_node, logic, outdir, split_output_dir=None):
    
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
        split_polydata(wma.io.read_polydata(output_name), output_name, split_output_dir)
        return
    
    check_load, polydata_node = slicer.util.loadFiberBundle(str(polydata), 1)
    if not check_load:
        raise RuntimeError('Could not load polydata file: ' + polydata)

    try:
        if polydata_node.GetPolyData().GetNumberOfCells() == 0:
            print('Empty cluster:', polydata)
            shutil.copyfile(polydata, output_name)
            if split_output_dir is not None:
                split_polydata(polydata_node.GetPolyData(), output_name, split_output_dir)
            return
        polydata_node.SetAndObserveTransformNodeID(transform_node.GetID())
        if not logic.hardenTransform(polydata_node):
            raise RuntimeError('Could not harden transform: ' + polydata)
        if not slicer.util.saveNode(polydata_node, output_name):
            raise RuntimeError('Could not save transformed polydata: ' + output_name)
        if split_output_dir is not None:
            # Split the same in-memory polydata just serialized by Slicer.
            split_polydata(polydata_node.GetPolyData(), output_name, split_output_dir)
    finally:
        # Fiber bundles own several display nodes and a storage node. Releasing
        # all of them keeps an 800-cluster run from growing the MRML scene.
        owned = [polydata_node.GetNthDisplayNode(i)
                 for i in range(polydata_node.GetNumberOfDisplayNodes())]
        owned.extend([display.GetDiffusionTensorDisplayPropertiesNode()
                      for display in list(owned)])
        owned.append(polydata_node.GetStorageNode())
        slicer.mrmlScene.RemoveNode(polydata_node)
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

    return parser


def _parse_args(parser):

    return parser.parse_args()


def main():

    parser = _build_arg_parser()
    args = _parse_args(parser)
    if args.split_output_directory is not None:
        _prepare_split_modules()

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

    check_load, transform_node = slicer.util.loadTransform(str(args.transform), 1)
    if not check_load:
        raise RuntimeError('Could not load transform file: ' + args.transform)
    slicer.mrmlScene.StartState(slicer.mrmlScene.BatchProcessState)
    try:
        if args.inverse == '1':
            transform_node.Inverse()
        logic = slicer.vtkSlicerTransformLogic()
        for polydata in input_polydatas:
            harden_transform(polydata, transform_node, logic, args.outdir,
                             args.split_output_directory)
    finally:
        storage = transform_node.GetStorageNode()
        slicer.mrmlScene.RemoveNode(transform_node)
        if storage is not None and storage.GetScene() == slicer.mrmlScene:
            slicer.mrmlScene.RemoveNode(storage)
        slicer.mrmlScene.EndState(slicer.mrmlScene.BatchProcessState)

if __name__ == '__main__':
    try:
        main()
    except Exception:
        import sys
        import traceback
        traceback.print_exc()
        sys.exit(1)
