#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import slicer
import os
import glob
import shutil

def harden_transform(polydata, transform_node, logic, outdir):
    
    output_name = os.path.join(outdir, os.path.basename(polydata))
    
    if os.path.exists(output_name):
        return
    
    check_load, polydata_node = slicer.util.loadFiberBundle(str(polydata), 1)
    if not check_load:
        raise RuntimeError('Could not load polydata file: ' + polydata)

    try:
        if polydata_node.GetPolyData().GetNumberOfCells() == 0:
            print('Empty cluster:', polydata)
            shutil.copyfile(polydata, output_name)
            return
        polydata_node.SetAndObserveTransformNodeID(transform_node.GetID())
        if not logic.hardenTransform(polydata_node):
            raise RuntimeError('Could not harden transform: ' + polydata)
        if not slicer.util.saveNode(polydata_node, output_name):
            raise RuntimeError('Could not save transformed polydata: ' + output_name)
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

    return parser


def _parse_args(parser):

    return parser.parse_args()


def main():

    parser = _build_arg_parser()
    args = _parse_args(parser)

    if os.path.isfile(args.polydata):
        input_polydatas = [args.polydata]
    elif os.path.isdir(args.polydata):
        input_mask = os.path.join(args.polydata, "*.vtk")
        input_mask2 = os.path.join(args.polydata, "*.vtp")
        input_pd_fnames = glob.glob(input_mask) + glob.glob(input_mask2)
        input_polydatas = sorted(input_pd_fnames)
    
    else:
        raise ValueError('Input is not a file or directory: ' + args.polydata)

    check_load, transform_node = slicer.util.loadTransform(str(args.transform), 1)
    if not check_load:
        raise RuntimeError('Could not load transform file: ' + args.transform)
    slicer.mrmlScene.StartState(slicer.mrmlScene.BatchProcessState)
    try:
        if args.inverse == '1':
            transform_node.Inverse()
        logic = slicer.vtkSlicerTransformLogic()
        for polydata in input_polydatas:
            harden_transform(polydata, transform_node, logic, args.outdir)
    finally:
        storage = transform_node.GetStorageNode()
        slicer.mrmlScene.RemoveNode(transform_node)
        if storage is not None and storage.GetScene() == slicer.mrmlScene:
            slicer.mrmlScene.RemoveNode(storage)
        slicer.mrmlScene.EndState(slicer.mrmlScene.BatchProcessState)

if __name__ == '__main__':
    main()
