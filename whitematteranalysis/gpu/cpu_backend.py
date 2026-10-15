# -*- coding: utf-8 -*-

""" gpu.cpu_backend

Vectorized, multi-threaded CPU backend for the whitematteranalysis
fiber distance computations. This is the "CPU accelerated" option that
runs everywhere numpy runs (no torch needed, e.g. inside 3D Slicer).

It preserves the original per-fiber arithmetic and point reduction order
in similarity.py/cluster.py. Large float64 tiles use SciPy's compiled
three-coordinate distances and reduce point planes directly; smaller tiles use
NumPy. Mean reductions retain NumPy's eight-accumulator summation order. Threshold
clamping, orientation/bilateral comparisons and point reductions retain
their original operations. Row tiles are independent, so results do not
depend on the number of threads.

"""

import os
from contextlib import contextmanager
from threading import RLock

import numpy

from .config import config

from . import fiber_tensors


def cpu_threads(requested=None):
    """Number of worker threads for the CPU backend (WMA_CPU_THREADS)."""

    if requested:
        if int(requested) < 0:
            from joblib import effective_n_jobs
            return effective_n_jobs(int(requested))
        return int(requested)
    if config.cpu_threads:
        return int(config.cpu_threads)
    from joblib import cpu_count
    return min(16, cpu_count())


def _tile_rows(m, points_per_fiber, dtype_bytes, threads, distance_method):
    """Keep distance temporaries small enough for CPU cache reuse."""

    # Larger short-fiber tiles reduce dispatch contention in large pools.
    # Compiled float64 planes are reused; float32 fallback buffers are smaller.
    buffers = 2 if (threads >= 16 and 8 <= points_per_fiber <= 32 and dtype_bytes == 8
                    and distance_method != 'StrictSimilarity') else 3
    bytes_per_row = m * points_per_fiber * dtype_bytes * buffers
    budget = (32 if threads == 1 else 16) * 1024 * 1024
    return max(1, int(budget // max(1, bytes_per_row)))


def _pointwise_distances(q, f, euclidean=False, point_major=False):
    """(b, P, 3) queries vs (m, P, 3) fixed -> (b, m, P) point
    distances (squared by default), optionally (P, b, m), with the operation order of
    similarity._fiber_distance_internal_use."""

    if (q.dtype == f.dtype == numpy.float64 and len(q) * len(f) >= 4096):
        # Compiled 3D distances fuse component differences/squares/additions.
        # A contiguous point axis is available for callers that require it.
        from scipy.spatial.distance import cdist
        points = numpy.empty((q.shape[1], len(q), len(f)), dtype=numpy.float64)
        for point in range(q.shape[1]):
            cdist(q[:, point], f[:, point],
                  metric='euclidean' if euclidean else 'sqeuclidean', out=points[point])
        return points if point_major else points.transpose(1, 2, 0).copy()

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
    dx += dy
    numpy.subtract(fz, qz, out=dy)
    numpy.square(dy, out=dy)
    dx += dy
    # A packed fixed layout must not change the NumPy point reduction order.
    dx = numpy.ascontiguousarray(dx)
    if euclidean:
        numpy.sqrt(dx, out=dx)
    return dx.transpose(2, 0, 1) if point_major else dx


def _sum_point_major(points):
    """Sum owned float64 scratch in NumPy's eight-accumulator point order."""
    count = len(points)
    if count < 8:
        return points.sum(0)
    if count > 128:
        middle = (count // 2) // 8 * 8
        return _sum_point_major(points[:middle]) + _sum_point_major(points[middle:])
    end = count - count % 8
    groups = points[:8]
    for offset in range(8, end, 8):
        groups += points[offset:offset + 8]
    total = ((groups[0] + groups[1]) + (groups[2] + groups[3])) + \
            ((groups[4] + groups[5]) + (groups[6] + groups[7]))
    for offset in range(end, count):
        total += points[offset]
    return total


def _reduce_points(d2, distance_method, threshold, sigmasq, points_per_fiber,
                   euclidean=False, axis=-1):
    """(b, m, P) -> (b, m), mirroring similarity.py:115-147."""

    if distance_method == 'Hausdorff':
        # Clipping is monotone, so the same maximum can be clipped afterward.
        d = d2.max(axis)
        if threshold:
            d -= threshold * threshold
            numpy.clip(d, 0.0, None, out=d)
        return d

    if threshold:
        d2 -= threshold * threshold
        numpy.clip(d2, 0.0, None, out=d2)

    if distance_method == 'Mean':
        if not euclidean:
            numpy.sqrt(d2, out=d2)
        d = _sum_point_major(d2) if axis == 0 else d2.sum(-1)
        d /= float(points_per_fiber)
        numpy.square(d, out=d)
    elif distance_method == 'MeanSquared':
        d = _sum_point_major(d2) if axis == 0 else d2.sum(-1)
        d = d / float(points_per_fiber)
    elif distance_method == 'StrictSimilarity':
        if axis == 0:
            numpy.negative(d2, out=d2)
            d2 /= sigmasq
            numpy.exp(d2, out=d2)
            d = d2.prod(0)
        else:
            d = numpy.exp(-d2 / sigmasq).prod(-1)
    else:
        raise ValueError(f"Unsupported distance method: {distance_method}")
    return d


def _fiber_block_distances(q, f, threshold, distance_method, sigmasq):
    """One tile of queries vs all fixed fibers, both point orders."""

    points_per_fiber = q.shape[1]
    # At zero threshold SciPy can fuse sqrt into its compiled 3D distances.
    euclidean = distance_method == 'Mean' and not threshold
    # Reduce compiled point planes directly, preserving each metric's order.
    point_major = q.dtype == f.dtype == numpy.float64 and len(q) * len(f) >= 4096
    axis = 0 if point_major else -1
    fwd = _reduce_points(_pointwise_distances(q, f, euclidean, point_major),
                         distance_method, threshold, sigmasq, points_per_fiber, euclidean, axis)
    q_rev = q[:, ::-1]
    rev = _reduce_points(_pointwise_distances(q_rev, f, euclidean, point_major),
                         distance_method, threshold, sigmasq, points_per_fiber, euclidean, axis)

    if distance_method == 'StrictSimilarity':
        return numpy.maximum(fwd, rev)
    return numpy.minimum(fwd, rev)


def _distance_block(q, fixed, threshold, distance_method, bilateral, sigmasq):
    d = _fiber_block_distances(q, fixed, threshold, distance_method, sigmasq)
    if bilateral:
        reflected = q.copy()
        reflected[:, :, 0] *= -1
        other = _fiber_block_distances(reflected, fixed, threshold, distance_method, sigmasq)
        if distance_method == 'StrictSimilarity':
            numpy.maximum(d, other, out=d)
        else:
            numpy.minimum(d, other, out=d)
    return d


def _distance_matrix(Xn, Xm, threshold, distance_method='Hausdorff',
                     bilateral=False, sigmasq=6400, gpu_float32=None,
                     tile_rows=None, number_of_jobs=None, executor=None, row_sum=False):
    """(n, P, 3) queries x (m, P, 3) fixed -> (n, m) float64 distance
    matrix (row i = query fiber i vs all fixed fibers), computed on
    tiled blocks over threads. Results are independent of threading.
    number_of_jobs overrides the thread count without changing global settings.
    row_sum=True returns per-query sums without allocating the full result.
    """

    dtype_bytes = 8
    threads = cpu_threads(number_of_jobs)
    tr = tile_rows or _tile_rows(Xm.shape[0], Xn.shape[1], dtype_bytes, threads, distance_method)
    n = Xn.shape[0]
    # Each compiled point-distance call reads contiguous three-coordinate rows.
    if Xn.dtype == Xm.dtype == numpy.float64 and min(n, tr) * len(Xm) >= 4096:
        Xm = numpy.ascontiguousarray(Xm.transpose(1, 0, 2)).transpose(1, 0, 2)
        if (Xn.strides[-1] != Xn.itemsize
                and (threads < 16 or distance_method != 'StrictSimilarity')):
            Xn = numpy.ascontiguousarray(Xn.transpose(1, 0, 2)).transpose(1, 0, 2)

    def compute(s, e):
        values = _distance_block(Xn[s:e], Xm, threshold, distance_method, bilateral, sigmasq)
        return values.sum(1) if row_sum else values

    if n <= tr:
        return compute(0, n)

    out = numpy.empty(n if row_sum else (n, Xm.shape[0]), dtype=numpy.float64)
    workers = min(threads, (n + tr - 1) // tr)

    def fill(worker):
        for start in range(worker * tr, n, workers * tr):
            end = min(start + tr, n)
            out[start:end] = compute(start, end)

    if workers == 1:
        fill(0)
    elif executor is not None:
        list(executor.map(fill, range(workers)))
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(fill, range(workers)))
    return out


_BLAS_LOCK = RLock()
_BLAS_USERS = 0
_BLAS_CAP = 0
_BLAS_CONTROLLER = None
_BLAS_LIMITERS = []


@contextmanager
def _atlas_blas_scope(columns, workers):
    """Share BLAS caps across concurrent fits; restore when the last fit exits."""
    global _BLAS_USERS, _BLAS_CAP, _BLAS_CONTROLLER
    from joblib import cpu_count
    automatic = min(4, max(1, cpu_count() - workers + 1)) if columns <= 32 else 0
    threads = int(os.environ.get('WMA_CPU_BLAS_THREADS', str(automatic)))
    if threads < 1:
        yield
        return
    try:
        from threadpoolctl import ThreadpoolController, __version__
        # Shared restoration requires the process-scoped API used in 3.1–3.6.
        if not (3, 1) <= tuple(map(int, __version__.split('.')[:2])) < (3, 7):
            raise ImportError('Unsupported BLAS controller scope')
    except ImportError:
        yield
        return
    with _BLAS_LOCK:
        if not _BLAS_USERS:
            _BLAS_CONTROLLER = ThreadpoolController().select(user_api='blas')
        if not _BLAS_USERS or threads < _BLAS_CAP:
            limits = {}
            for library in _BLAS_CONTROLLER.info():
                if library['num_threads']:
                    prefix = library['prefix']
                    limits[prefix] = min(threads, library['num_threads'], limits.get(prefix, threads))
            _BLAS_LIMITERS.append(_BLAS_CONTROLLER.limit(limits=limits))
            _BLAS_CAP = threads
        _BLAS_USERS += 1
    try:
        yield
    finally:
        with _BLAS_LOCK:
            _BLAS_USERS -= 1
            if not _BLAS_USERS:
                for limiter in reversed(_BLAS_LIMITERS):
                    limiter.restore_original_limits()
                _BLAS_LIMITERS.clear()
                _BLAS_CONTROLLER = None
                _BLAS_CAP = 0


def atlas_label(input_polydata, atlas, number_of_jobs=None):
    """Stream CPU distances and project only the required embedding columns.

    Retain O(N * embedding_columns) results, not the full O(N * atlas_size)
    similarity/projection matrices. Reuse distance workers across blocks.
    """
    from concurrent.futures import ThreadPoolExecutor
    import scipy.cluster.vq

    queries = fiber_tensors.polydata_to_numpy(input_polydata, 15, coordinate_major=True)
    fixed = fiber_tensors.polydata_to_numpy(atlas.nystrom_polydata, 15, coordinate_major=True)
    count = atlas.number_of_eigenvectors + 1
    vectors = numpy.ascontiguousarray(atlas.e_vec[:, -count:])
    reciprocal = numpy.divide(1., atlas.e_val[-count:])
    norm = atlas.e_vec_norm[-count:]
    dhat_a = numpy.sqrt(numpy.divide(1., atlas.row_sum_1))
    n = len(queries)
    embed = numpy.empty((n, atlas.number_of_eigenvectors))
    rows = numpy.empty(n)
    columns = numpy.empty(n)
    labels = numpy.empty(n, dtype=numpy.int32)
    distances = numpy.empty(n)
    # Target 8 MiB for the similarity block, independently of subject size.
    block_rows = max(1, min(512, (8 * 1024 * 1024) // (8 * max(1, len(fixed)))))
    threads = cpu_threads(number_of_jobs)
    with _atlas_blas_scope(count, threads), ThreadPoolExecutor(max_workers=threads) as pool:
        for start in range(0, n, block_rows):
            stop = min(n, start + block_rows)
            B = _distance_matrix(queries[start:stop], fixed, atlas.threshold,
                atlas.distance_method, atlas.bilateral,
                number_of_jobs=threads, executor=pool).T
            if atlas.distance_method != 'StrictSimilarity':
                numpy.negative(B, out=B)
                B /= atlas.sigma * atlas.sigma
                numpy.exp(B, out=B)
            row_sum = B.sum(0) + numpy.dot(atlas.row_sum_matrix, B)
            if numpy.any(row_sum <= 0):
                print(f"<{__name__}> Warning: negative row sum approximations:",
                      numpy.count_nonzero(row_sum <= 0))
                row_sum[row_sum < 0] = .1
            rows[start:stop] = row_sum
            B *= dhat_a[:, None] * numpy.sqrt(1. / row_sum)[None, :]
            columns[start:stop] = B.sum(0)
            V = numpy.dot(B.T, vectors)
            V *= reciprocal
            V /= norm
            embed[start:stop] = V[:, -2::-1] / V[:, -1:]
            labels[start:stop], distances[start:stop] = scipy.cluster.vq.vq(
                embed[start:stop], atlas.centroids)
    return embed, labels, distances, numpy.concatenate((atlas.row_sum_1, rows)), columns


def pairwise_distance_matrix(input_polydata, threshold,
                             distance_method='Hausdorff', bilateral=False,
                             sigmasq=6400, points_per_fiber=15,
                             gpu_float32=None, tile_rows=None):
    """CPU-fast equivalent of cluster._pairwise_distance_matrix."""

    X = fiber_tensors.polydata_to_numpy(input_polydata, points_per_fiber, coordinate_major=True)
    return _distance_matrix(X, X, threshold, distance_method, bilateral,
                            sigmasq, gpu_float32, tile_rows)


def rectangular_distance_matrix(input_polydata_n, input_polydata_m, threshold,
                                distance_method='Hausdorff', bilateral=False,
                                sigmasq=6400, points_per_fiber=15,
                                gpu_float32=None, tile_rows=None):
    """CPU-fast equivalent of cluster._rectangular_distance_matrix;
    returns the (n, m) matrix BEFORE the transpose applied by cluster.py."""

    Xn = fiber_tensors.polydata_to_numpy(input_polydata_n, points_per_fiber, coordinate_major=True)
    Xm = fiber_tensors.polydata_to_numpy(input_polydata_m, points_per_fiber, coordinate_major=True)
    return _distance_matrix(Xn, Xm, threshold, distance_method, bilateral,
                            sigmasq, gpu_float32, tile_rows)


def pairwise_distance_matrix_from_fiber_array(fiber_array, threshold,
                                              distance_method='Hausdorff',
                                              bilateral=False, sigmasq=6400,
                                              gpu_float32=None, tile_rows=None):
    """Pairwise matrix from an already converted fibers.FiberArray."""

    X = fiber_tensors.fiber_array_to_numpy(fiber_array, coordinate_major=True)
    return _distance_matrix(X, X, threshold, distance_method, bilateral,
                            sigmasq, gpu_float32, tile_rows)


def total_similarity_matrix_strict(X_queries, X_fixed, reflect_mask, threshold,
                                   sigmasq, gpu_float32=None, tile_rows=None, row_sum=False):
    """StrictSimilarity matrix for laterality (see gpu.distance for the
    GPU version): row i is the StrictSimilarity of query fiber i
    (x-reflected first when reflect_mask[i] is True) to every fiber of
    X_fixed. row_sum=True returns only per-query totals, preserving the
    original NumPy summation order while bounding the result memory."""

    q_all = numpy.asarray(X_queries, dtype=numpy.float64)
    reflect_mask = numpy.asarray(reflect_mask, dtype=bool)
    q_all = q_all.copy()
    if reflect_mask.any():
        q_all[reflect_mask, :, 0] = -q_all[reflect_mask, :, 0]
    return _distance_matrix(q_all, numpy.asarray(X_fixed, dtype=numpy.float64),
                            threshold, 'StrictSimilarity', False, sigmasq,
                            gpu_float32, tile_rows, row_sum=row_sum)


def transform_fiber_array(in_array, transform, mode=(1, 1, 1, 1)):
    """CPU fast transform_fiber_array_numpy: the same 4x4 batched
    application used on the GPU (bit-identical to the per-point
    vtkTransform loop, see gpu.registration)."""

    from ..register_two_subjects import convert_transform_to_vtk
    matrix = convert_transform_to_vtk(transform, scaled=True, mode=list(mode)).GetMatrix()
    M = numpy.array([[matrix.GetElement(i, j) for j in range(4)] for i in range(4)])
    x = in_array[0]
    y = in_array[1]
    z = in_array[2]
    out = numpy.empty_like(in_array)
    out[0] = M[0, 0] * x + M[0, 1] * y + M[0, 2] * z + M[0, 3]
    out[1] = M[1, 0] * x + M[1, 1] * y + M[1, 2] * z + M[1, 3]
    out[2] = M[2, 0] * x + M[2, 1] * y + M[2, 2] * z + M[2, 3]
    return out


def inner_loop_objective(fixed, moving, sigmasq, chunk_rows=None, threads=None, executor=None):
    """CPU fast inner_loop_objective (register_two_subjects.py:305-330):
    batched over moving fibers with the same per-element operations and
    summation structures as the per-fiber original."""

    n_fixed = fixed.shape[1]
    points_per_fiber = fixed.shape[2]
    n_moving = moving.shape[1]

    threads = cpu_threads(threads)
    if chunk_rows is None:
        chunk_rows = _tile_rows(n_fixed, points_per_fiber, 8, threads, 'MeanSquared')

    fixed = fixed.transpose(1, 2, 0)
    moving = moving.transpose(1, 2, 0)
    if fixed.dtype == moving.dtype == numpy.float64 and min(n_moving, chunk_rows) * n_fixed >= 4096:
        fixed = numpy.ascontiguousarray(fixed.transpose(1, 0, 2)).transpose(1, 0, 2)
        if moving.strides[-1] != moving.itemsize:
            moving = numpy.ascontiguousarray(moving.transpose(1, 0, 2)).transpose(1, 0, 2)
    probability = numpy.empty(n_moving, dtype=numpy.float64)

    workers = min(threads, (n_moving + chunk_rows - 1) // chunk_rows)

    def fill(worker):
        for s in range(worker * chunk_rows, n_moving, workers * chunk_rows):
            e = min(s + chunk_rows, n_moving)
            dist = _fiber_block_distances(moving[s:e], fixed, 0., 'MeanSquared', sigmasq)
            probability[s:e] = numpy.exp(-dist / sigmasq).sum(1)

    if workers <= 1:
        if workers:
            fill(0)
    elif executor is not None:
        list(executor.map(fill, range(workers)))
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(fill, range(workers)))

    # register_two_subjects.py:315-329: floor before dividing by n_fixed
    probability = probability + 1e-20
    probability = probability / float(n_fixed)
    return float(numpy.sum(-numpy.log(probability)))
