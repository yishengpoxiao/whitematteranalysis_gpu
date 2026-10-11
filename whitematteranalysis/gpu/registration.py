# -*- coding: utf-8 -*-

""" gpu.registration

GPU (torch) implementations of the fiber registration objective,
numerically equivalent to register_two_subjects.py:

    transform_fiber_array_numpy   (register_two_subjects.py:386)
    inner_loop_objective          (register_two_subjects.py:305)

The 15 parameter transform is still built with
register_two_subjects.convert_transform_to_vtk, but its final 4x4
matrix is applied to all fiber points as one batched computation with
the same accumulation order as vtkMatrix4x4::MultiplyPoint, so with
float64 the transformed points are identical to the CPU code.

The objective differs from the CPU only in the summation order over
fibers (~1e-15 relative difference in float64).

"""

import numpy

from .config import get_device
from . import registration_kernel

try:
    import torch
except ImportError:
    torch = None


def transform_matrix_from_vtk(transform, mode=[1, 1, 1, 1]):
    """Build the 4x4 matrix of the 15 parameter transform exactly as
    vtk would apply it (register_two_subjects.convert_transform_to_vtk
    with scaled=True)."""

    from ..register_two_subjects import convert_transform_to_vtk

    vtktrans = convert_transform_to_vtk(transform, scaled=True, mode=mode)
    m4 = vtktrans.GetMatrix()
    M = numpy.array([m4.GetElement(i, j) for i in range(4) for j in range(4)],
                    dtype=numpy.float64)
    return M.reshape(4, 4)


def transform_fiber_array(in_array, transform, mode=[1, 1, 1, 1], device=None):
    """Batched transform_fiber_array_numpy.

    in_array: (3, N, P) float64 numpy array (R, A, S on the first axis).
    Returns a (3, N, P) float64 numpy array, the same layout.
    """

    return _transform_fiber_tensor(in_array, transform, mode, device).cpu().numpy()


def _transform_fiber_tensor(in_array, transform, mode, device):
    """Apply the affine transform and retain its float64 result on device."""
    if torch is None:
        raise RuntimeError("torch is not available, cannot use the GPU registration backend.")

    M = transform_matrix_from_vtk(transform, mode)
    dev = device if device is not None else get_device()

    t = torch.as_tensor(in_array, dtype=torch.float64, device=dev)
    x, y, z = t.unbind(0)
    Mt = torch.from_numpy(M).to(dev)

    # same accumulation order as vtkMatrix4x4::MultiplyPoint
    out_x = Mt[0, 0] * x + Mt[0, 1] * y + Mt[0, 2] * z + Mt[0, 3]
    out_y = Mt[1, 0] * x + Mt[1, 1] * y + Mt[1, 2] * z + Mt[1, 3]
    out_z = Mt[2, 0] * x + Mt[2, 1] * y + Mt[2, 2] * z + Mt[2, 3]

    return torch.stack((out_x, out_y, out_z), dim=0)


def _prepare_fixed(fixed, device, dtype):
    """(3, Nf, P) NumPy array -> contiguous (Nf, P, 3) device tensor."""

    F = numpy.ascontiguousarray(numpy.transpose(fixed, (1, 2, 0)))
    return torch.as_tensor(F, dtype=dtype, device=device)


def inner_loop_objective(fixed, moving, sigmasq, device=None, chunk_rows=None):
    """Batched inner_loop_objective.

    fixed: (3, N, P) numpy array.
    moving: (3, N, P) numpy array or device tensor.
    Returns a python float, matching register_two_subjects.py:305-330.
    """

    if torch is None:
        raise RuntimeError("torch is not available, cannot use the GPU registration backend.")

    dev = device if device is not None else get_device()
    # the registration objective is MeanSquared in float64 on CPU; keep
    # float64 here regardless of the WMA_GPU_FLOAT32 fast mode so the
    # optimization trajectory is not perturbed more than necessary
    dtype = torch.float64

    F = _prepare_fixed(fixed, dev, dtype)

    if isinstance(moving, torch.Tensor):
        M = moving.permute(1, 2, 0).contiguous().to(device=dev, dtype=dtype)
    else:
        M = numpy.ascontiguousarray(numpy.transpose(moving, (1, 2, 0)))
        M = torch.as_tensor(M, dtype=dtype)
    return _inner_loop_objective(F, M, sigmasq, dev, chunk_rows)


def _inner_loop_objective(F, M, sigmasq, dev, chunk_rows=None):
    """Entropy for prepared (N, P, 3) tensors; retain the CPU reduction."""
    n_fixed, points_per_fiber = F.shape[:2]
    n_moving = M.shape[0]

    if chunk_rows is None:
        itemsize = 8
        bytes_per_row = max(1, n_fixed * points_per_fiber * itemsize * 6)
        try:
            free_bytes, _total = torch.cuda.mem_get_info(dev)
        except Exception:
            free_bytes = 4 << 30
        chunk_rows = max(1, min(n_moving, int(int(free_bytes * 0.5) // bytes_per_row),
                                8192))

    probability = torch.empty(n_moving, dtype=torch.float64, device=dev)

    s = 0
    while s < n_moving:
        e = min(s + chunk_rows, n_moving)
        try:
            q = M[s:e].to(dev)
            prob = _pairwise_probability(F, q, sigmasq)
        except torch.cuda.OutOfMemoryError:
            if chunk_rows == 1:
                raise
            torch.cuda.empty_cache()
            chunk_rows = max(1, chunk_rows // 2)
            continue

        probability[s:e] = prob
        s = e

    # register_two_subjects.py:315-329: floor before dividing by n_fixed
    probability = probability.cpu()
    probability = probability + 1e-20
    probability = probability / float(n_fixed)
    entropy = torch.sum(-torch.log(probability))
    return float(entropy)


def _squared_point_distances(F, q):
    """Eager reference for environments without Triton."""
    dx = F[None, :, :, 0] - q[:, None, :, 0]
    dy = F[None, :, :, 1] - q[:, None, :, 1]
    dz = F[None, :, :, 2] - q[:, None, :, 2]
    dx = dx.mul_(dx)
    dy = dy.mul_(dy)
    dz = dz.mul_(dz)
    return dx.add_(dy).add_(dz)


def _pairwise_probability(F, q, sigmasq):
    """Fuse point arithmetic, preserving PyTorch's point/fiber reductions."""
    if registration_kernel.triton is not None and F.is_cuda:
        dist_forward, dist_reverse = registration_kernel.mean_squared_distances(F, q)
    else:
        points = float(F.shape[1])
        dist_forward = _squared_point_distances(F, q).sum(-1) / points
        dist_reverse = _squared_point_distances(F, q.flip(1)).sum(-1) / points
    dist = torch.minimum(dist_forward, dist_reverse)
    return torch.exp(-dist / sigmasq).sum(-1)


def objective_value(fixed, moving, transform, mode, sigma, device=None):
    """Convenience: transform the moving brain then compute the entropy,
    equivalent to RegisterTractography.objective_function."""

    moved = _transform_fiber_tensor(moving, transform, mode, device)
    return inner_loop_objective(fixed, moved, sigma * sigma, device=device)


class AffineObjective:
    """Device data shared by all evaluations of one optimizer invocation.

    Create a fresh instance when the sampled fixed/moving fibers change.
    No source NumPy arrays are retained or modified.
    """

    def __init__(self, fixed, moving, device=None):
        self.device = device if device is not None else get_device()
        self.fixed = _prepare_fixed(fixed, self.device, torch.float64)
        self.moving = torch.as_tensor(moving, dtype=torch.float64, device=self.device)

    def __call__(self, transform, mode, sigma):
        moved = _transform_fiber_tensor(self.moving, transform, mode, self.device)
        moved = moved.permute(1, 2, 0).contiguous()
        return _inner_loop_objective(self.fixed, moved, sigma * sigma, self.device)
