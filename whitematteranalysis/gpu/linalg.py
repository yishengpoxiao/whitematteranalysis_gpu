# -*- coding: utf-8 -*-

""" gpu.linalg

GPU (torch) versions of the linear algebra steps that surround the
distance matrices in spectral clustering:

  * the Nystrom row-sum estimates and spectral embedding GEMMs
    (cluster.spectral and cluster.spectral_atlas_label), and
  * (optional, off by default) numpy.linalg.eigh on the GPU.

The GEMM results differ from CPU BLAS only in floating point
summation order (~1e-15 relative in float64), which validation shows
is far below the level that affects cluster labels.

"""

import os

import numpy

from .config import get_device, config

try:
    import torch
except ImportError:
    torch = None


def eigh_enabled():
    """GPU eigendecomposition is opt-in (WMA_GPU_EIGH=1): eigh feeds the
    spectral embedding directly, so it stays on the CPU LAPACK path by
    default to keep atlas creation as close to the CPU results as
    possible."""

    return os.environ.get('WMA_GPU_EIGH', '0') in ('1', 'true', 'yes', 'on')


def eigh(M):
    """eigh for a symmetric (m, m) float64 matrix.

    Returns (e_val, e_vec) as float64 numpy arrays. Uses LAPACK unless
    the WMA_GPU_EIGH flag is set and a GPU is available.
    """

    if eigh_enabled() and config.cuda_available():
        dev = get_device()
        Mt = torch.as_tensor(M, dtype=torch.float64, device=dev)
        e_val, e_vec = torch.linalg.eigh(Mt)
        return e_val.cpu().numpy(), e_vec.cpu().numpy()
    return numpy.linalg.eigh(M)


def _as_gpu(np_array, dev):
    return torch.as_tensor(numpy.ascontiguousarray(np_array),
                           dtype=torch.float64, device=dev)


def nystrom_row_sum_2(B, row_sum_matrix):
    """GPU computation of cluster.spectral's B-part row sum estimate:

        row_sum_2 = B.sum(0) + row_sum_matrix @ B

    B: (m, n) float64. Returns float64 numpy (n,).
    """

    dev = get_device()
    Bg = _as_gpu(B, dev)
    rm = _as_gpu(row_sum_matrix, dev)
    return (Bg.sum(0) + rm @ Bg).cpu().numpy()


def nystrom_embed_B(B, e_vec, e_val):
    """GPU computation of cluster.spectral's Nystrom embedding block:

        V2 = (B.T @ e_vec) @ diag(1/e_val)

    B: (m, n) float64. Returns V2 as float64 numpy (n, m).
    """

    dev = get_device()
    Bg = _as_gpu(B, dev)
    Eg = _as_gpu(e_vec, dev)
    recip = numpy.divide(1.0, e_val)
    V2 = (Bg.t() @ Eg) * _as_gpu(recip, dev)[None, :]
    return V2.cpu().numpy()


def atlas_label_embedding(B, atlas):
    """GPU equivalent of cluster.spectral_atlas_label steps 2-5.

    B: (m, n) float64 similarity matrix (rows = atlas sample, columns =
    input fibers), as produced by _rectangular_similarity_matrix. May be
    a numpy array (uploaded here) or a float torch CUDA tensor already
    on the device (the fused path - avoids a round trip over PCIe).
    Returns (embed, cluster_idx, dist, row_sum, B_col_sums) where
    row_sum and B_col_sums feed _format_output_polydata.
    """

    return _atlas_embedding_block(B, atlas, _prepare_atlas(atlas))


def _prepare_atlas(atlas):
    dev = get_device()
    count = atlas.number_of_eigenvectors + 1
    return tuple(_as_gpu(a, dev) for a in (
        atlas.row_sum_matrix, atlas.e_vec[:, -count:],
        numpy.divide(1., atlas.e_val[-count:]), atlas.e_vec_norm[-count:],
        numpy.sqrt(numpy.divide(1., atlas.row_sum_1))))


def _atlas_embedding_block(B, atlas, prepared):
    import scipy.cluster.vq

    dev = get_device()
    rm, Eg, recip, norm, dhat_a = prepared
    Bg = B.to(device=dev, dtype=torch.float64) if torch.is_tensor(B) else _as_gpu(B, dev)
    row_sum_2 = (Bg.sum(0) + rm @ Bg).cpu().numpy()
    if numpy.any(row_sum_2 <= 0):
        print(f"<{__name__}> Warning: negative row sum approximations:",
              numpy.count_nonzero(row_sum_2 <= 0))
        row_sum_2[row_sum_2 < 0] = 0.1
    dhat_b = _as_gpu(numpy.sqrt(numpy.divide(1., row_sum_2)), dev)
    Bg = Bg * (dhat_a[:, None] * dhat_b[None, :])
    b_col_sums = Bg.sum(0)
    V = ((Bg.t() @ Eg) * recip[None, :]) / norm[None, :]
    embed = torch.stack([V[:, -(i + 2)] / V[:, -1]
                         for i in range(atlas.number_of_eigenvectors)], dim=1).cpu().numpy()
    cluster_idx, dist = scipy.cluster.vq.vq(embed, atlas.centroids)
    return (embed, cluster_idx, dist,
            numpy.concatenate((atlas.row_sum_1, row_sum_2)), b_col_sums.cpu().numpy())


def atlas_label_embedding_blocks(blocks, atlas, number_fibers):
    """Embed each input block without retaining the full similarity or V matrices."""
    prepared = _prepare_atlas(atlas)
    embed = numpy.empty((number_fibers, atlas.number_of_eigenvectors))
    labels = numpy.empty(number_fibers, dtype=numpy.int32)
    distances = numpy.empty(number_fibers)
    rows = numpy.empty(number_fibers)
    columns = numpy.empty(number_fibers)
    sz = len(atlas.row_sum_1)
    for start, block in blocks:
        end = start + block.shape[1]
        e, c, d, r, b = _atlas_embedding_block(block, atlas, prepared)
        embed[start:end], labels[start:end], distances[start:end] = e, c, d
        rows[start:end], columns[start:end] = r[sz:], b
        del block
    return embed, labels, distances, numpy.concatenate((atlas.row_sum_1, rows)), columns
