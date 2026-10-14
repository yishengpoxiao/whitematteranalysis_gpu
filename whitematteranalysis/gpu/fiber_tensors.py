# -*- coding: utf-8 -*-

""" gpu.fiber_tensors

Conversions between whitematteranalysis fiber representations and
compact (N, P, 3) numpy arrays that the GPU kernels consume.

The last axis holds the R, A, S coordinates, matching the three
separate float64 arrays stored in fibers.FiberArray. No coordinate
transforms are applied here: the GPU kernels must operate on exactly
the same numbers as the CPU code.

"""

import numpy


SUPPORTED_DISTANCE_METHODS = ('Mean', 'MeanSquared', 'Hausdorff', 'StrictSimilarity')


def distance_supported(distance_method, landmarks=None, landmarks_n=None, landmarks_m=None):
    """Check accelerated distance support without importing a CUDA backend."""
    return (distance_method in SUPPORTED_DISTANCE_METHODS and landmarks is None
            and landmarks_n is None and landmarks_m is None)


def fiber_array_to_numpy(fiber_array, coordinate_major=False):
    """Return (N, P, 3); optionally keep each coordinate contiguous for CPUs."""

    values = numpy.stack((fiber_array.fiber_array_r,
                          fiber_array.fiber_array_a,
                          fiber_array.fiber_array_s), axis=0 if coordinate_major else -1)
    return values.transpose(1, 2, 0) if coordinate_major else values


def polydata_to_numpy(input_vtk_polydata, points_per_fiber, coordinate_major=False):
    """vtkPolyData -> (N, P, 3) float64 numpy array.

    Uses the (vectorized) fixed-length resampling in fibers.py, so the
    output is bit-identical to what the CPU distance code consumes.
    """

    from .. import fibers

    fiber_array = fibers.FiberArray()
    fiber_array.convert_from_polydata(input_vtk_polydata,
                                      points_per_fiber=points_per_fiber)
    return fiber_array_to_numpy(fiber_array, coordinate_major=coordinate_major)
