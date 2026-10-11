"""Batched WMA B-spline transforms, matching VTK's cubic/Zero-border rules.

Coefficient filtering stays in VTK; fixed sampling geometry is prepared once.
Reference: Kitware/VTK v9.1.0 Filters/Hybrid/vtkBSplineTransform.cxx.
"""
import numpy

from .config import get_device
from . import registration, registration_kernel

try:
    import torch
except ImportError:
    torch = None


def transform_fiber_array_numpy(fibers, displacement):
    """Apply VTK's double-precision transform in C++, without Python point loops."""
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
    from ..register_two_subjects_nonrigid_bsplines import convert_transform_to_vtk

    coordinates = numpy.ascontiguousarray(fibers.transpose(1, 2, 0).reshape(-1, 3), dtype=numpy.float64)
    points, output = vtk.vtkPoints(), vtk.vtkPoints()
    points.SetData(numpy_to_vtk(coordinates))
    output.SetDataTypeToDouble()
    convert_transform_to_vtk(displacement).TransformPoints(points, output)
    return vtk_to_numpy(output.GetData()).reshape(fibers.shape[1], fibers.shape[2], 3).transpose(2, 0, 1).copy()


def _sampling_geometry(fibers, resolution):
    coordinates = numpy.ascontiguousarray(
        fibers.transpose(1, 2, 0).reshape(-1, 3), dtype=numpy.float64)
    grid = (coordinates + 100.) / (200. / (resolution - 1))
    integer = numpy.floor(grid).astype(numpy.int64)
    fraction = grid - integer
    square = fraction * fraction
    w3 = square * fraction * (1. / 6.)
    w0 = (square - fraction) * .5 - w3 + (1. / 6.)
    w2 = fraction + w0 - w3 * 2
    w1 = 1 - w0 - w2 - w3
    weights = numpy.stack((w0, w1, w2, w3), axis=-1)
    offsets = integer[..., None] + numpy.arange(-1, 3)
    weights[offsets < 0] = 0.
    weights[offsets >= resolution] = 0.
    invalid = numpy.any((integer < -2) | (integer > resolution), axis=1)
    offsets = offsets.clip(0, resolution - 1).astype(numpy.int32)
    return coordinates, offsets, weights, invalid


if registration_kernel.triton is not None:
    triton = registration_kernel.triton
    import triton.language as tl

    @triton.jit
    def _transform(Points, Indices, Weights, Invalid, Coefficients, Output,
                   COUNT, RES: tl.constexpr, BLOCK: tl.constexpr):
        scalar = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        point, component = scalar // 3, scalar % 3
        mask = scalar < COUNT
        invalid = tl.load(Invalid + point, mask, other=1)
        delta = tl.full((BLOCK,), 0., tl.float64)
        for z in tl.static_range(4):
            iz = tl.load(Indices + point * 12 + 8 + z, mask, other=0)
            wz = tl.load(Weights + point * 12 + 8 + z, mask, other=0.)
            vz = tl.full((BLOCK,), 0., tl.float64)
            for y in tl.static_range(4):
                iy = tl.load(Indices + point * 12 + 4 + y, mask, other=0)
                wy = tl.load(Weights + point * 12 + 4 + y, mask, other=0.)
                vy = tl.full((BLOCK,), 0., tl.float64)
                for x in tl.static_range(4):
                    ix = tl.load(Indices + point * 12 + x, mask, other=0)
                    wx = tl.load(Weights + point * 12 + x, mask, other=0.)
                    index = ((iz * RES + iy) * RES + ix) * 3 + component
                    value = tl.load(Coefficients + index, mask, other=0.).to(tl.float64)
                    vy += value * wx
                vz += vy * wy
            delta += vz * wz
        original = tl.load(Points + scalar, mask, other=0.)
        tl.store(Output + scalar, original + tl.where(invalid, 0., delta), mask)


class _CoefficientGrid:
    """Reuse VTK's grid and coefficient filter for one optimization."""
    def __init__(self, resolution):
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk

        self.displacement = numpy.zeros((resolution**3, 3), dtype=numpy.float32)
        self.scalars = numpy_to_vtk(self.displacement)
        self.scalars.SetName('DisplacementField')
        self.grid = vtk.vtkImageData()
        self.grid.SetOrigin(-100., -100., -100.)
        spacing = 200. / (resolution - 1)
        self.grid.SetSpacing(spacing, spacing, spacing)
        self.grid.SetDimensions(resolution, resolution, resolution)
        self.grid.GetPointData().SetScalars(self.scalars)
        self.filter = vtk.vtkImageBSplineCoefficients()
        self.filter.SetInputData(self.grid)

    def __call__(self, displacement):
        from vtk.util.numpy_support import vtk_to_numpy

        numpy.copyto(self.displacement, numpy.asarray(displacement).reshape(-1, 3))
        self.scalars.Modified()
        self.grid.Modified()
        self.filter.Update()
        self.filter.UpdateWholeExtent()
        return vtk_to_numpy(self.filter.GetOutput().GetPointData().GetScalars())


class NonrigidObjective:
    """Fixed geometry and GPU inputs owned by one nonrigid optimization."""
    def __init__(self, fixed, moving, resolution, device=None):
        self.device = device if device is not None else get_device()
        self.fixed = registration._prepare_fixed(fixed, self.device, torch.float64)
        self.resolution = resolution
        self.shape = moving.shape
        if registration_kernel.triton is not None:
            self.coefficients = _CoefficientGrid(resolution)
            geometry = _sampling_geometry(moving, resolution)
            self.points, self.indices, self.weights, self.invalid = (
                torch.as_tensor(a, device=self.device) for a in geometry)
        else:
            self.moving = moving

    def transform(self, displacement):
        if registration_kernel.triton is None:
            moved = transform_fiber_array_numpy(self.moving, displacement)
            return registration._prepare_fixed(moved, self.device, torch.float64)
        coefficients = torch.as_tensor(self.coefficients(displacement), device=self.device)
        output = torch.empty_like(self.points)
        with torch.cuda.device(self.device):
            _transform[(triton.cdiv(output.numel(), 128),)](
                self.points, self.indices, self.weights, self.invalid, coefficients,
                output, output.numel(), RES=self.resolution, BLOCK=128,
                enable_fp_fusion=False)
        return output.reshape(self.shape[1], self.shape[2], 3)

    def __call__(self, displacement, sigma):
        moved = self.transform(displacement)
        return registration._inner_loop_objective(self.fixed, moved, sigma * sigma, self.device)
