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

from .config import get_device, float32_enabled

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

    if torch is None:
        raise RuntimeError("torch is not available, cannot use the GPU registration backend.")

    M = transform_matrix_from_vtk(transform, mode)
    dev = device if device is not None else get_device()

    t = torch.as_tensor(in_array, dtype=torch.float64)
    x = t[0].to(dev)
    y = t[1].to(dev)
    z = t[2].to(dev)
    Mt = torch.from_numpy(M).to(dev)

    # same accumulation order as vtkMatrix4x4::MultiplyPoint
    out_x = Mt[0, 0] * x + Mt[0, 1] * y + Mt[0, 2] * z + Mt[0, 3]
    out_y = Mt[1, 0] * x + Mt[1, 1] * y + Mt[1, 2] * z + Mt[1, 3]
    out_z = Mt[2, 0] * x + Mt[2, 1] * y + Mt[2, 2] * z + Mt[2, 3]

    out = torch.stack((out_x, out_y, out_z), dim=0)
    return out.cpu().numpy()


def _prepare_fixed(fixed, device, dtype):
    """(3, Nf, P) numpy -> (Nf, P, 3) device tensor plus component views."""

    F = numpy.ascontiguousarray(numpy.transpose(fixed, (1, 2, 0)))
    F = torch.as_tensor(F, dtype=dtype, device=device)
    return F, F[:, :, 0], F[:, :, 1], F[:, :, 2]


def inner_loop_objective(fixed, moving, sigmasq, device=None, chunk_rows=None):
    """Batched inner_loop_objective.

    fixed, moving: (3, N, P) float64 numpy arrays.
    Returns a python float, matching register_two_subjects.py:305-330.
    """

    if torch is None:
        raise RuntimeError("torch is not available, cannot use the GPU registration backend.")

    dev = device if device is not None else get_device()
    # the registration objective is MeanSquared in float64 on CPU; keep
    # float64 here regardless of the WMA_GPU_FLOAT32 fast mode so the
    # optimization trajectory is not perturbed more than necessary
    dtype = torch.float64

    F, fx, fy, fz = _prepare_fixed(fixed, dev, dtype)
    n_fixed = F.shape[0]
    points_per_fiber = F.shape[1]

    M = numpy.ascontiguousarray(numpy.transpose(moving, (1, 2, 0)))
    M = torch.as_tensor(M, dtype=dtype)
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

    probability = torch.empty(n_moving, dtype=torch.float64)

    s = 0
    while s < n_moving:
        e = min(s + chunk_rows, n_moving)
        try:
            q = M[s:e].to(dev)
            qx = q[:, :, 0].unsqueeze(1)
            qy = q[:, :, 1].unsqueeze(1)
            qz = q[:, :, 2].unsqueeze(1)

            # forward point order (sign of the difference does not matter
            # after squaring, matches register_two_subjects.py:366-375)
            dx = fx.unsqueeze(0) - qx
            dx = dx.mul_(dx)
            dy = fy.unsqueeze(0) - qy
            dy = dy.mul_(dy)
            dz = fz.unsqueeze(0) - qz
            dz = dz.mul_(dz)
            d2 = dx.add_(dy)
            d2 = d2.add_(dz)
            dist_fwd = d2.sum(-1) / float(points_per_fiber)

            # reversed point order of the moving fiber (:361-364)
            dx = fx.unsqueeze(0) - qx.flip(2)
            dx = dx.mul_(dx)
            dy = fy.unsqueeze(0) - qy.flip(2)
            dy = dy.mul_(dy)
            dz = fz.unsqueeze(0) - qz.flip(2)
            dz = dz.mul_(dz)
            d2 = dx.add_(dy)
            d2 = d2.add_(dz)
            dist_rev = d2.sum(-1) / float(points_per_fiber)

            dist = torch.minimum(dist_fwd, dist_rev)
            prob = torch.exp(-dist / sigmasq)
            prob = prob.sum(-1)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            chunk_rows = max(1, chunk_rows // 2)
            continue

        probability[s:e] = prob.cpu()
        s = e

    # register_two_subjects.py:315-329: floor before dividing by n_fixed
    probability = probability + 1e-20
    probability = probability / float(n_fixed)
    entropy = torch.sum(-torch.log(probability))
    return float(entropy)


def objective_value(fixed, moving, transform, mode, sigma, device=None):
    """Convenience: transform the moving brain then compute the entropy,
    equivalent to RegisterTractography.objective_function."""

    moved = transform_fiber_array(moving, transform, mode, device=device)
    return inner_loop_objective(fixed, moved, sigma * sigma, device=device)
