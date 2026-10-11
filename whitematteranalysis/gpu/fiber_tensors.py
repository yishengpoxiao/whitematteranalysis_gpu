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


def fiber_array_to_numpy(fiber_array):
    """Fibers.FiberArray -> (N, P, 3) float64 numpy array."""

    return numpy.stack((fiber_array.fiber_array_r,
                        fiber_array.fiber_array_a,
                        fiber_array.fiber_array_s), axis=-1)


def polydata_to_numpy(input_vtk_polydata, points_per_fiber):
    """vtkPolyData -> (N, P, 3) float64 numpy array.

    Uses the (vectorized) fixed-length resampling in fibers.py, so the
    output is bit-identical to what the CPU distance code consumes.
    """

    from .. import fibers

    fiber_array = fibers.FiberArray()
    fiber_array.convert_from_polydata(input_vtk_polydata,
                                      points_per_fiber=points_per_fiber)
    return fiber_array_to_numpy(fiber_array)
