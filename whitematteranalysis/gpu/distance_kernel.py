"""Optional float64 distance kernels for short fibers and embedding vectors."""
from . import registration_kernel

triton = registration_kernel.triton
if triton is not None:
    import triton.language as tl
    from triton.language.extra.cuda import libdevice

    @triton.jit
    def _values(F, Q, fi, qi, p, mask, P: tl.constexpr,
                REVERSE: tl.constexpr, THRESHOLD2: tl.constexpr, METHOD: tl.constexpr):
        d = registration_kernel._point_distance(F, Q, fi, qi, p, mask, P, REVERSE)
        if THRESHOLD2 != 0:
            d = d - tl.full((), THRESHOLD2, tl.float64)
            d = tl.where(d < 0., 0., d)
        if METHOD == 'Mean':
            d = libdevice.sqrt_rn(d)
        return d

    @triton.jit
    def _distances(F, Q, Output, NF, COUNT,
                   P: tl.constexpr, WIDTH: tl.constexpr, STEPS: tl.constexpr,
                   THRESHOLD2: tl.constexpr, METHOD: tl.constexpr, BLOCK: tl.constexpr):
        pair = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        p = tl.arange(0, WIDTH)[None, :]
        fi, qi = pair[:, None] % NF, pair[:, None] // NF
        mask = pair[:, None] < COUNT
        a = _values(F, Q, fi, qi, p, mask, P, False, THRESHOLD2, METHOD)
        b = _values(F, Q, fi, qi, p, mask, P, True, THRESHOLD2, METHOD)
        if P > WIDTH:
            mask2 = mask & (p + WIDTH < P)
            a2 = _values(F, Q, fi, qi, p + WIDTH, mask2, P, False, THRESHOLD2, METHOD)
            b2 = _values(F, Q, fi, qi, p + WIDTH, mask2, P, True, THRESHOLD2, METHOD)
            if METHOD == 'Hausdorff':
                a, b = tl.maximum(a, a2, propagate_nan=tl.PropagateNan.ALL), tl.maximum(b, b2, propagate_nan=tl.PropagateNan.ALL)
            else:
                a, b = a + a2, b + b2
        for step in tl.static_range(STEPS):
            al, ar = tl.split(tl.reshape(a, (BLOCK, WIDTH // (1 << (step + 1)), 2)))
            bl, br = tl.split(tl.reshape(b, (BLOCK, WIDTH // (1 << (step + 1)), 2)))
            if METHOD == 'Hausdorff':
                a, b = tl.maximum(al, ar, propagate_nan=tl.PropagateNan.ALL), tl.maximum(bl, br, propagate_nan=tl.PropagateNan.ALL)
            else:
                a, b = al + ar, bl + br
        a, b = tl.reshape(a, (BLOCK,)), tl.reshape(b, (BLOCK,))
        if METHOD != 'Hausdorff':
            inv = tl.full((), 1. / P, tl.float64)
            a, b = a * inv, b * inv
        if METHOD == 'Mean':
            a, b = a * a, b * b
        result = tl.minimum(a, b, propagate_nan=tl.PropagateNan.ALL)
        tl.store(Output + pair, result, pair < COUNT)

    @triton.jit
    def _centroid_distances(Q, C, Output, NC, COUNT,
                            D: tl.constexpr, WIDTH: tl.constexpr,
                            STEPS: tl.constexpr, BLOCK: tl.constexpr):
        pair = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        dimension = tl.arange(0, WIDTH)[None, :]
        mask = (pair[:, None] < COUNT) & (dimension < D)
        query = tl.load(Q + (pair[:, None] // NC) * D + dimension, mask, other=0.)
        centroid = tl.load(C + (pair[:, None] % NC) * D + dimension, mask, other=0.)
        diff = query - centroid
        values = diff * diff
        # Match cdist's descending warp-shuffle reduction, including sqrt rounding.
        for step in tl.static_range(STEPS):
            a, b = tl.split(tl.trans(tl.reshape(values,
                (BLOCK, 2, WIDTH // (1 << (step + 1)))), 0, 2, 1))
            values = a + b
        result = libdevice.sqrt_rn(tl.reshape(values, (BLOCK,)))
        tl.store(Output + pair, result, pair < COUNT)


def centroid_distances(queries, centroids):
    """Direct Euclidean distances for contiguous float64 CUDA vectors, 1<=D<=32."""
    import torch

    output = torch.empty((len(queries), len(centroids)), dtype=queries.dtype, device=queries.device)
    dimensions = queries.shape[1]
    block = 128 if dimensions <= 16 else 32
    if output.numel():
        with torch.cuda.device(queries.device):
            _centroid_distances[(triton.cdiv(output.numel(), block),)](
                queries, centroids, output, len(centroids), output.numel(),
                D=dimensions, WIDTH=triton.next_power_of_2(dimensions),
                STEPS=(dimensions - 1).bit_length(), BLOCK=block,
                num_warps=1, enable_fp_fusion=False)
    return output


def fiber_distances(queries, fixed, threshold, method):
    import torch

    shape = (queries.shape[0], fixed.shape[0])
    output = torch.empty(shape, dtype=fixed.dtype, device=fixed.device)
    points = fixed.shape[1]
    steps = points.bit_length() - 1
    block = 64 if points <= 16 else 32
    with torch.cuda.device(fixed.device):
        _distances[(triton.cdiv(output.numel(), block),)](
            fixed, queries, output, shape[1], output.numel(),
            P=points, WIDTH=1 << steps, STEPS=steps, THRESHOLD2=threshold * threshold,
            METHOD=method, BLOCK=block, num_warps=1, enable_fp_fusion=False)
    return output
