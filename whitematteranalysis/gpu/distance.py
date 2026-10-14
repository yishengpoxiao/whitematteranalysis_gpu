# -*- coding: utf-8 -*-

""" gpu.distance

GPU (torch) implementations of the whitematteranalysis fiber distance
and similarity matrices, numerically equivalent to the CPU code in
similarity.py and cluster.py.

The distance between one query fiber and all fibers of an array
(similarity.fiber_distance) is computed for a whole block of query
fibers at once as a broadcasted tensor operation:

    d2[b, j, p] = (fx_jp - qx_bp)^2 + (fy_jp - qy_bp)^2 + (fz_jp - qz_bp)^2

followed by the per-point threshold of similarity.py:115-119 and one of
the four point-axis reductions of similarity.py:121-147. The forward /
reversed fiber variants and the bilateral (x mirrored) variants are
combined with element-wise minimum (maximum for StrictSimilarity),
matching similarity.py:34-56.

With the default float64 precision the results match the CPU within
~1e-12 relative error (the only difference is floating point summation
order over the fiber points). float32 can be selected with
WMA_GPU_FLOAT32=1 as a fast, lower precision mode.

"""

import numpy

from .config import get_device, float32_enabled, config

try:
    import torch
except ImportError:  # allow import of this module without torch
    torch = None

from . import fiber_tensors, distance_kernel

# Methods with a GPU implementation. Landmarks/Frechet/Mean_shape stay
# on the CPU path (see gpu.distance_supported).
SUPPORTED_DISTANCE_METHODS = fiber_tensors.SUPPORTED_DISTANCE_METHODS
distance_supported = fiber_tensors.distance_supported


def _pointwise_squared_distance(q, f):
    """(b, P, 3) queries vs (m, P, 3) fixed -> (b, m, P) squared distances.

    Follows the component-wise operation order of
    similarity.py:_fiber_distance_internal_use so that float64 results
    agree bit-for-bit up to the reduction over points.
    """

    qx = q[:, :, 0].unsqueeze(1)
    qy = q[:, :, 1].unsqueeze(1)
    qz = q[:, :, 2].unsqueeze(1)
    fx = f[:, :, 0].unsqueeze(0)
    fy = f[:, :, 1].unsqueeze(0)
    fz = f[:, :, 2].unsqueeze(0)

    dx = fx - qx
    dx = dx.mul_(dx)
    dy = fy - qy
    dy = dy.mul_(dy)
    dz = fz - qz
    dz = dz.mul_(dz)

    d2 = dx.add_(dy)
    d2 = d2.add_(dz)
    return d2


def _reduce_points(d2, distance_method, threshold, sigmasq, points_per_fiber):
    """Reduce (b, m, P) squared pointwise distances to (b, m) fiber
    distances, exactly mirroring similarity.py:115-147."""

    if threshold:
        # threshold is applied to the squared pointwise distances,
        # before the reduction over points (similarity.py:115-119).
        d2 = d2 - threshold * threshold
        d2 = d2.clamp_min_(0.0)

    if distance_method == 'Mean':
        # mean euclidean point distance, squared for consistency
        d = torch.sqrt(d2)
        d = d.sum(-1)
        d = d / float(points_per_fiber)
        d = d * d
    elif distance_method == 'Hausdorff':
        # max of SQUARED pointwise distances (as in the CPU code)
        d = d2.amax(-1)
    elif distance_method == 'MeanSquared':
        d = d2.sum(-1)
        d = d / float(points_per_fiber)
    elif distance_method == 'StrictSimilarity':
        # product of per-point similarities (similarity.distance_to_similarity)
        d = torch.exp(-d2 / sigmasq)
        d = d.prod(-1)
    else:
        raise ValueError(f"Unsupported distance method for GPU: {distance_method}")

    return d


def _fiber_block_distances(q, f, threshold, distance_method, sigmasq, bilateral=False):
    """One block of query fibers vs all fixed fibers, both directions,
    (b, m) result. Matches similarity.fiber_distance, including bilateral mode."""

    points_per_fiber = q.shape[1]
    if (distance_kernel.triton is not None and q.is_cuda and q.dtype == f.dtype == torch.float64
            and q.is_contiguous() and f.is_contiguous()
            and points_per_fiber <= 32
            and distance_method in ('Mean', 'MeanSquared', 'Hausdorff')):
        return distance_kernel.fiber_distances(q, f, threshold, distance_method, bilateral)
    fwd = _reduce_points(_pointwise_squared_distance(q, f),
                         distance_method, threshold, sigmasq, points_per_fiber)
    rev = _reduce_points(_pointwise_squared_distance(q.flip(1), f),
                         distance_method, threshold, sigmasq, points_per_fiber)

    combine = torch.maximum if distance_method == 'StrictSimilarity' else torch.minimum
    distances = combine(fwd, rev)
    del fwd, rev
    if bilateral:
        reflected = q.clone()
        reflected[:, :, 0] = -reflected[:, :, 0]
        other = _fiber_block_distances(reflected, f, threshold, distance_method, sigmasq)
        distances = combine(distances, other)
    return distances


def _choose_tile_rows(n, m, points_per_fiber, dtype, device, override=None):
    """Pick the number of query fibers per tile from the free VRAM."""

    if override:
        return max(1, int(override))
    override = config.tile_rows
    if override:
        return max(1, int(override))

    itemsize = 8 if dtype == torch.float64 else 4
    # a handful of (b, m, P) temporaries are alive at once
    bytes_per_query_fiber = m * points_per_fiber * itemsize * 6
    try:
        free_bytes, _total = torch.cuda.mem_get_info(device)
    except Exception:
        free_bytes = 4 << 30
    budget = max(256 << 20, int(free_bytes * 0.55))
    tile_rows = int(budget // max(1, bytes_per_query_fiber))
    return max(16, min(tile_rows, 16384, n))


def _distance_matrix(Xn, Xm, threshold, distance_method='Hausdorff',
                     bilateral=False, sigmasq=6400, gpu_float32=None,
                     tile_rows=None, verbose=False):
    """(n, P, 3) queries x (m, P, 3) fixed -> (n, m) distance matrix.

    Row i holds the distances from query fiber i to all fixed fibers,
    identical in layout to stacking similarity.fiber_distance rows.
    Runs on a single CUDA device (device 0 of the visible set, or the
    device bound by WMA_GPU_DEVICES).
    """

    if torch is None:
        raise RuntimeError("torch is not available, cannot use the GPU distance backend.")

    device = get_device()
    dtype = torch.float32 if float32_enabled(gpu_float32) else torch.float64
    out_dtype = numpy.float32 if dtype == torch.float32 else numpy.float64

    f = torch.as_tensor(Xm, dtype=dtype, device=device)
    n = Xn.shape[0]
    m = Xm.shape[0]
    out = numpy.empty((n, m), dtype=out_dtype)

    tr = _choose_tile_rows(n, m, Xn.shape[1], dtype, device, override=tile_rows)
    if verbose:
        print(f"<{__name__}> GPU distance matrix {n} x {m}, "
              f"tile {tr}, dtype {dtype}, device {device}")

    s = 0
    while s < n:
        e = min(s + tr, n)
        try:
            q = torch.as_tensor(Xn[s:e], dtype=dtype, device=device)
            d = _fiber_block_distances(q, f, threshold, distance_method, sigmasq, bilateral)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            tr = max(1, tr // 2)
            print(f"<{__name__}> GPU out of memory, retrying with tile size {tr}")
            continue

        out[s:e] = d.cpu().numpy()
        s = e

    del f
    return out


def pairwise_distance_matrix(input_polydata, threshold,
                             distance_method='Hausdorff', bilateral=False,
                             sigmasq=6400, points_per_fiber=15,
                             gpu_float32=None, tile_rows=None):
    """GPU equivalent of cluster._pairwise_distance_matrix (non-Frechet)."""

    X = fiber_tensors.polydata_to_numpy(input_polydata, points_per_fiber)
    return _distance_matrix(X, X, threshold, distance_method, bilateral,
                            sigmasq, gpu_float32, tile_rows)


def rectangular_distance_matrix(input_polydata_n, input_polydata_m, threshold,
                                distance_method='Hausdorff', bilateral=False,
                                sigmasq=6400, points_per_fiber=15,
                                gpu_float32=None, tile_rows=None):
    """GPU equivalent of cluster._rectangular_distance_matrix (non-Frechet).

    Returns the (n, m) matrix BEFORE the transpose that
    cluster._rectangular_distance_matrix applies to its result.
    """

    Xn = fiber_tensors.polydata_to_numpy(input_polydata_n, points_per_fiber)
    Xm = fiber_tensors.polydata_to_numpy(input_polydata_m, points_per_fiber)
    return _distance_matrix(Xn, Xm, threshold, distance_method, bilateral,
                            sigmasq, gpu_float32, tile_rows)


def pairwise_distance_matrix_from_fiber_array(fiber_array, threshold,
                                              distance_method='Hausdorff',
                                              bilateral=False, sigmasq=6400,
                                              gpu_float32=None, tile_rows=None):
    """Pairwise matrix from an already converted fibers.FiberArray
    (used by filter.smooth / anisotropic_smooth / laplacian_of_gaussian
    which need custom points_per_fiber)."""

    X = fiber_tensors.fiber_array_to_numpy(fiber_array)
    return _distance_matrix(X, X, threshold, distance_method, bilateral,
                            sigmasq, gpu_float32, tile_rows)


def distance_matrix_from_arrays(Xn, Xm, threshold, distance_method='Hausdorff',
                                bilateral=False, sigmasq=6400,
                                gpu_float32=None, tile_rows=None):
    """Pairwise matrix from raw (n, P, 3) float numpy arrays."""

    return _distance_matrix(Xn, Xm, threshold, distance_method, bilateral,
                            sigmasq, gpu_float32, tile_rows)


def pairwise_similarity_matrix(input_polydata, threshold, sigma,
                               distance_method='Hausdorff', bilateral=False,
                               sigmasq=6400, points_per_fiber=15,
                               gpu_float32=None, tile_rows=None):
    """GPU equivalent of cluster._pairwise_similarity_matrix (non-Frechet).

    For StrictSimilarity the "similarities" are the distances themselves,
    exactly as in the CPU implementation.
    """

    distances = pairwise_distance_matrix(input_polydata, threshold,
                                         distance_method, bilateral,
                                         sigmasq, points_per_fiber,
                                         gpu_float32, tile_rows)
    if distance_method == 'StrictSimilarity':
        return distances
    return numpy.exp(-distances / (sigma * sigma))


def rectangular_similarity_matrix(input_polydata_n, input_polydata_m, threshold,
                                  sigma, distance_method='Hausdorff',
                                  bilateral=False, gpu_float32=None,
                                  tile_rows=None):
    """GPU equivalent of cluster._rectangular_similarity_matrix
    (non-Frechet). Returns the (n, m) matrix BEFORE the transpose that
    cluster._rectangular_similarity_matrix applies to its result.
    """

    distances = rectangular_distance_matrix(input_polydata_n, input_polydata_m,
                                            threshold, distance_method,
                                            bilateral, gpu_float32=gpu_float32,
                                            tile_rows=tile_rows)
    if distance_method == 'StrictSimilarity':
        return distances
    return numpy.exp(-distances / (sigma * sigma))


def rectangular_similarity_tensor(input_polydata_n, input_polydata_m, threshold,
                                  sigma, distance_method='Hausdorff',
                                  bilateral=False, sigmasq=6400,
                                  points_per_fiber=15,
                                  gpu_float32=None, tile_rows=None):
    """(m, n) similarity matrix kept ON THE GPU as a float64 torch
    tensor (transposed view of the (n, m) kernel output). The Gaussian
    kernel exp(-d/sigma^2) is applied on the GPU, so the potentially
    large matrix never crosses PCIe. Used by the fused atlas-labeling
    path in gpu.linalg.atlas_label_embedding."""

    if torch is None:
        raise RuntimeError("torch is not available, cannot use the GPU distance backend.")

    device = get_device()
    dtype = torch.float32 if float32_enabled(gpu_float32) else torch.float64

    Xn = fiber_tensors.polydata_to_numpy(input_polydata_n, points_per_fiber)
    Xm = fiber_tensors.polydata_to_numpy(input_polydata_m, points_per_fiber)
    n = Xn.shape[0]

    f = torch.as_tensor(Xm, dtype=dtype, device=device)
    out = torch.empty((n, Xm.shape[0]), dtype=dtype, device=device)

    tr = _choose_tile_rows(n, Xm.shape[0], Xn.shape[1], dtype, device,
                           override=tile_rows)
    s = 0
    while s < n:
        e = min(s + tr, n)
        try:
            q = torch.as_tensor(Xn[s:e], dtype=dtype, device=device)
            d = _fiber_block_distances(q, f, threshold, distance_method, sigmasq, bilateral)
            if distance_method != 'StrictSimilarity':
                d = torch.exp(-d / (sigma * sigma))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            tr = max(1, tr // 2)
            print(f"<{__name__}> GPU out of memory, retrying with tile size {tr}")
            continue
        out[s:e] = d
        s = e

    del f
    return out.t()


def iter_rectangular_similarity(input_polydata_n, input_polydata_m, threshold,
                               sigma, distance_method='Hausdorff', bilateral=False,
                               sigmasq=6400, points_per_fiber=15,
                               gpu_float32=None, tile_rows=None):
    """Yield (start, (atlas_sample, input_block)) GPU similarity blocks."""
    device = get_device()
    dtype = torch.float32 if float32_enabled(gpu_float32) else torch.float64
    queries = fiber_tensors.polydata_to_numpy(input_polydata_n, points_per_fiber)
    fixed = fiber_tensors.polydata_to_numpy(input_polydata_m, points_per_fiber)
    fixed = torch.as_tensor(fixed, dtype=dtype, device=device)
    requested = tile_rows or config.tile_rows or 2048
    rows = _choose_tile_rows(len(queries), len(fixed), points_per_fiber,
                             dtype, device, override=requested)
    start = 0
    while start < len(queries):
        end = min(start + rows, len(queries))
        try:
            block = _similarity_tensor_block(queries[start:end], fixed, threshold,
                                             sigma, distance_method, bilateral, sigmasq)
        except torch.cuda.OutOfMemoryError:
            if rows == 1:
                raise
            rows = max(1, rows // 2)
            torch.cuda.empty_cache()
            continue
        yield start, block.t()
        del block
        start = end


def _similarity_tensor_block(queries, fixed, threshold, sigma, method, bilateral, sigmasq):
    queries = torch.as_tensor(queries, dtype=fixed.dtype, device=fixed.device)
    distances = _fiber_block_distances(queries, fixed, threshold, method, sigmasq, bilateral)
    if method != 'StrictSimilarity':
        distances = torch.exp(-distances / (sigma * sigma))
    return distances


def total_similarity_matrix_strict(X_queries, X_fixed, reflect_mask, threshold,
                                   sigmasq, gpu_float32=None, tile_rows=None):
    """StrictSimilarity matrix used by laterality.

    Row i is the StrictSimilarity of query fiber i (x-reflected first when
    reflect_mask[i] is True) to every fiber of X_fixed. Mirrors
    similarity.total_similarity_for_laterality.
    """

    if torch is None:
        raise RuntimeError("torch is not available, cannot use the GPU distance backend.")

    device = get_device()
    dtype = torch.float32 if float32_enabled(gpu_float32) else torch.float64
    out_dtype = numpy.float32 if dtype == torch.float32 else numpy.float64

    f = torch.as_tensor(X_fixed, dtype=dtype, device=device)
    q_all = torch.as_tensor(X_queries, dtype=numpy.float64)
    reflect_mask = numpy.asarray(reflect_mask, dtype=bool)

    out = numpy.empty((X_queries.shape[0], X_fixed.shape[0]), dtype=out_dtype)
    n = X_queries.shape[0]
    tr = _choose_tile_rows(n, X_fixed.shape[0], X_queries.shape[1], dtype,
                           device, override=tile_rows)

    s = 0
    while s < n:
        e = min(s + tr, n)
        try:
            q = q_all[s:e].to(device=device, dtype=dtype)
            if reflect_mask[s:e].any():
                idx = numpy.nonzero(reflect_mask[s:e])[0]
                q_reflect = q[idx]
                q_reflect[:, :, 0] = -q_reflect[:, :, 0]
                q[idx] = q_reflect
            sim = _fiber_block_distances(q, f, threshold, 'StrictSimilarity',
                                         sigmasq)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            tr = max(1, tr // 2)
            continue
        out[s:e] = sim.cpu().numpy()
        s = e

    return out
