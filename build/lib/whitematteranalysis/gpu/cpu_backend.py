# -*- coding: utf-8 -*-

""" gpu.cpu_backend

Vectorized, multi-threaded CPU backend for the whitematteranalysis
fiber distance computations. This is the "CPU accelerated" option that
runs everywhere numpy runs (no torch needed, e.g. inside 3D Slicer).

It computes exactly the same numbers as the original per-fiber joblib
code in similarity.py/cluster.py - every element is produced by the
same numpy operations (elementwise diff/square, pairwise reduction over
the fiber points, threshold clamp, forward/reversed minimum, bilateral
minimum) - only the loop structure is tiled so whole blocks of query
fibers are computed at once. Row tiles are independent, so results do
not depend on the number of threads.

"""

import os

import numpy

from .config import get_device, float32_enabled, config

try:
    import torch
except ImportError:
    torch = None

from . import fiber_tensors


def cpu_threads(requested=None):
    """Number of worker threads for the CPU backend (WMA_CPU_THREADS)."""

    if requested:
        return int(requested)
    if config.cpu_threads:
        return int(config.cpu_threads)
    return min(16, os.cpu_count() or 4)


def _tile_rows(m, points_per_fiber, dtype_bytes, threads):
    """Rows per tile targeting ~250 MB of temporaries per thread."""

    bytes_per_row = m * points_per_fiber * dtype_bytes * 6
    rows = int((250 * 1024 * 1024) // max(1, bytes_per_row))
    return max(8, rows)


def _pointwise_squared_distance(q, f):
    """(b, P, 3) queries vs (m, P, 3) fixed -> (b, m, P) squared
    distances, with the exact operation order of
    similarity._fiber_distance_internal_use."""

    qx = q[:, :, 0][:, None, :]
    qy = q[:, :, 1][:, None, :]
    qz = q[:, :, 2][:, None, :]
    fx = f[:, :, 0][None, :, :]
    fy = f[:, :, 1][None, :, :]
    fz = f[:, :, 2][None, :, :]

    dx = fx - qx
    numpy.square(dx, out=dx)
    dy = fy - qy
    numpy.square(dy, out=dy)
    dz = fz - qz
    numpy.square(dz, out=dz)

    d2 = dx + dy
    d2 += dz
    return d2


def _reduce_points(d2, distance_method, threshold, sigmasq, points_per_fiber):
    """(b, m, P) -> (b, m), mirroring similarity.py:115-147."""

    if threshold:
        d2 -= threshold * threshold
        numpy.clip(d2, 0.0, None, out=d2)

    if distance_method == 'Mean':
        d = numpy.sqrt(d2)
        d = d.sum(-1)
        d = d / float(points_per_fiber)
        d = d * d
    elif distance_method == 'Hausdorff':
        d = d2.max(-1)
    elif distance_method == 'MeanSquared':
        d = d2.sum(-1)
        d = d / float(points_per_fiber)
    elif distance_method == 'StrictSimilarity':
        d = numpy.exp(-d2 / sigmasq)
        d = d.prod(-1)
    else:
        raise ValueError(f"Unsupported distance method: {distance_method}")
    return d


def _fiber_block_distances(q, f, threshold, distance_method, sigmasq):
    """One tile of queries vs all fixed fibers, both point orders."""

    points_per_fiber = q.shape[1]
    fwd = _reduce_points(_pointwise_squared_distance(q, f),
                         distance_method, threshold, sigmasq, points_per_fiber)
    q_rev = numpy.ascontiguousarray(numpy.flip(q, axis=1))
    rev = _reduce_points(_pointwise_squared_distance(q_rev, f),
                         distance_method, threshold, sigmasq, points_per_fiber)

    if distance_method == 'StrictSimilarity':
        return numpy.maximum(fwd, rev)
    return numpy.minimum(fwd, rev)


def _distance_matrix(Xn, Xm, threshold, distance_method='Hausdorff',
                     bilateral=False, sigmasq=6400, gpu_float32=None,
                     tile_rows=None):
    """(n, P, 3) queries x (m, P, 3) fixed -> (n, m) float64 distance
    matrix (row i = query fiber i vs all fixed fibers), computed on
    tiled blocks over threads. Results are independent of threading."""

    dtype_bytes = 8
    threads = cpu_threads()
    tr = tile_rows or _tile_rows(Xm.shape[0], Xn.shape[1], dtype_bytes, threads)
    n = Xn.shape[0]

    def compute(s, e):
        q = Xn[s:e]
        d = _fiber_block_distances(q, Xm, threshold, distance_method, sigmasq)
        if bilateral:
            q_reflect = q.copy()
            q_reflect[:, :, 0] = -q_reflect[:, :, 0]
            d_reflect = _fiber_block_distances(q_reflect, Xm, threshold,
                                               distance_method, sigmasq)
            if distance_method == 'StrictSimilarity':
                numpy.maximum(d, d_reflect, out=d)
            else:
                numpy.minimum(d, d_reflect, out=d)
        return d

    if threads <= 1 or n <= tr:
        return compute(0, n)

    from concurrent.futures import ThreadPoolExecutor
    bounds = [int(x) for x in numpy.linspace(0, n, min(threads, (n + tr - 1) // tr) + 1)]
    blocks = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)
              if bounds[i + 1] > bounds[i]]
    out = numpy.empty((n, Xm.shape[0]), dtype=numpy.float64)
    with ThreadPoolExecutor(max_workers=len(blocks)) as pool:
        futures = {pool.submit(compute, s, e): (s, e) for (s, e) in blocks}
        for fut in futures:
            s, e = futures[fut]
            out[s:e] = fut.result()
    return out


def pairwise_distance_matrix(input_polydata, threshold,
                             distance_method='Hausdorff', bilateral=False,
                             sigmasq=6400, points_per_fiber=15,
                             gpu_float32=None, tile_rows=None):
    """CPU-fast equivalent of cluster._pairwise_distance_matrix."""

    X = fiber_tensors.polydata_to_numpy(input_polydata, points_per_fiber)
    return _distance_matrix(X, X, threshold, distance_method, bilateral,
                            sigmasq, gpu_float32, tile_rows)


def rectangular_distance_matrix(input_polydata_n, input_polydata_m, threshold,
                                distance_method='Hausdorff', bilateral=False,
                                sigmasq=6400, points_per_fiber=15,
                                gpu_float32=None, tile_rows=None):
    """CPU-fast equivalent of cluster._rectangular_distance_matrix;
    returns the (n, m) matrix BEFORE the transpose applied by cluster.py."""

    Xn = fiber_tensors.polydata_to_numpy(input_polydata_n, points_per_fiber)
    Xm = fiber_tensors.polydata_to_numpy(input_polydata_m, points_per_fiber)
    return _distance_matrix(Xn, Xm, threshold, distance_method, bilateral,
                            sigmasq, gpu_float32, tile_rows)


def pairwise_distance_matrix_from_fiber_array(fiber_array, threshold,
                                              distance_method='Hausdorff',
                                              bilateral=False, sigmasq=6400,
                                              gpu_float32=None, tile_rows=None):
    """Pairwise matrix from an already converted fibers.FiberArray."""

    X = fiber_tensors.fiber_array_to_numpy(fiber_array)
    return _distance_matrix(X, X, threshold, distance_method, bilateral,
                            sigmasq, gpu_float32, tile_rows)


def total_similarity_matrix_strict(X_queries, X_fixed, reflect_mask, threshold,
                                   sigmasq, gpu_float32=None, tile_rows=None):
    """StrictSimilarity matrix for laterality (see gpu.distance for the
    GPU version): row i is the StrictSimilarity of query fiber i
    (x-reflected first when reflect_mask[i] is True) to every fiber of
    X_fixed."""

    q_all = numpy.asarray(X_queries, dtype=numpy.float64)
    reflect_mask = numpy.asarray(reflect_mask, dtype=bool)
    q_all = q_all.copy()
    if reflect_mask.any():
        q_all[reflect_mask, :, 0] = -q_all[reflect_mask, :, 0]
    return _distance_matrix(q_all, numpy.asarray(X_fixed, dtype=numpy.float64),
                            threshold, 'StrictSimilarity', False, sigmasq,
                            gpu_float32, tile_rows)


def transform_fiber_array(in_array, transform, mode=(1, 1, 1, 1)):
    """CPU fast transform_fiber_array_numpy: the same 4x4 batched
    application used on the GPU (bit-identical to the per-point
    vtkTransform loop, see gpu.registration)."""

    from .registration import transform_matrix_from_vtk
    M = transform_matrix_from_vtk(transform, list(mode))
    x = in_array[0]
    y = in_array[1]
    z = in_array[2]
    out = numpy.empty_like(in_array)
    out[0] = M[0, 0] * x + M[0, 1] * y + M[0, 2] * z + M[0, 3]
    out[1] = M[1, 0] * x + M[1, 1] * y + M[1, 2] * z + M[1, 3]
    out[2] = M[2, 0] * x + M[2, 1] * y + M[2, 2] * z + M[2, 3]
    return out


def inner_loop_objective(fixed, moving, sigmasq, chunk_rows=None, threads=None):
    """CPU fast inner_loop_objective (register_two_subjects.py:305-330):
    batched over moving fibers with the same per-element operations and
    summation structures as the per-fiber original."""

    n_fixed = fixed.shape[1]
    points_per_fiber = fixed.shape[2]
    n_moving = moving.shape[1]

    if chunk_rows is None:
        chunk_rows = max(1, int(2.5e8 // max(1, n_fixed * points_per_fiber * 8 * 5)))
    threads = cpu_threads(threads)

    fx = fixed[0]
    fy = fixed[1]
    fz = fixed[2]
    probability = numpy.empty(n_moving, dtype=numpy.float64)

    s = 0
    while s < n_moving:
        e = min(s + chunk_rows, n_moving)
        qx = moving[0, s:e]
        qy = moving[1, s:e]
        qz = moving[2, s:e]

        # forward point order (register_two_subjects.py:366-375):
        # squared componentwise, summed dx+dy+dz, mean over points
        dx = fx[None, :, :] - qx[:, None, :]
        numpy.square(dx, out=dx)
        dy = fy[None, :, :] - qy[:, None, :]
        numpy.square(dy, out=dy)
        dz = fz[None, :, :] - qz[:, None, :]
        numpy.square(dz, out=dz)
        d2 = dx + dy
        d2 += dz
        dist_fwd = d2.mean(-1)

        # reversed point order of the moving fiber (:361-364)
        dx = fx[None, :, :] - qx[:, None, ::-1]
        numpy.square(dx, out=dx)
        dy = fy[None, :, :] - qy[:, None, ::-1]
        numpy.square(dy, out=dy)
        dz = fz[None, :, :] - qz[:, None, ::-1]
        numpy.square(dz, out=dz)
        d2 = dx + dy
        d2 += dz
        dist_rev = d2.mean(-1)

        dist = numpy.minimum(dist_fwd, dist_rev)
        probability[s:e] = numpy.exp(-dist / sigmasq).sum(1)
        s = e

    # register_two_subjects.py:315-329: floor before dividing by n_fixed
    probability = probability + 1e-20
    probability = probability / float(n_fixed)
    return float(numpy.sum(-numpy.log(probability)))
