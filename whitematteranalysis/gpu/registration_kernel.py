"""Optional fused registration distances/probabilities; PyTorch reduces fibers."""

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
except ImportError:
    triton = None


if triton is not None:
    @triton.jit
    def _affine_points(Moving, Matrix, Output, COUNT, BLOCK: tl.constexpr):
        point = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = point < COUNT
        x = tl.load(Moving + point, mask, other=0.)
        y = tl.load(Moving + COUNT + point, mask, other=0.)
        z = tl.load(Moving + 2 * COUNT + point, mask, other=0.)
        for axis in tl.static_range(3):
            value = (tl.load(Matrix + 4 * axis) * x + tl.load(Matrix + 4 * axis + 1) * y)
            value = value + tl.load(Matrix + 4 * axis + 2) * z
            value = value + tl.load(Matrix + 4 * axis + 3)
            tl.store(Output + 3 * point + axis, value, mask)


def affine_transform(moving, matrix, output):
    """Write contiguous (N, P, 3) points from (3, N, P) coordinates in one launch."""
    import torch

    count = moving.shape[1] * moving.shape[2]
    if count == 0:
        return
    with torch.cuda.device(moving.device):
        _affine_points[(triton.cdiv(count, 128),)](
            moving, matrix, output, count, BLOCK=128, enable_fp_fusion=False)


if triton is not None:
    @triton.jit
    def _squared_distances(F, M, Forward, Reverse, NF,
                           P: tl.constexpr, BLOCK: tl.constexpr):
        tiles = tl.cdiv(NF * P, BLOCK)
        moving = tl.program_id(0) // tiles
        offsets = (tl.program_id(0) % tiles) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < NF * P
        point = offsets % P
        fixed = offsets * 3
        direct = (moving * P + point) * 3
        reverse = (moving * P + P - 1 - point) * 3
        x = tl.load(F + fixed, mask, other=0.)
        y = tl.load(F + fixed + 1, mask, other=0.)
        z = tl.load(F + fixed + 2, mask, other=0.)
        dx = x - tl.load(M + direct, mask, other=0.)
        dy = y - tl.load(M + direct + 1, mask, other=0.)
        dz = z - tl.load(M + direct + 2, mask, other=0.)
        rx = x - tl.load(M + reverse, mask, other=0.)
        ry = y - tl.load(M + reverse + 1, mask, other=0.)
        rz = z - tl.load(M + reverse + 2, mask, other=0.)
        output = moving * NF * P + offsets
        tl.store(Forward + output, (dx * dx + dy * dy) + dz * dz, mask)
        tl.store(Reverse + output, (rx * rx + ry * ry) + rz * rz, mask)


def squared_distances(fixed, moving):
    """Return forward/reverse (Nm, Nf, P) squared point distances.

    Inputs are contiguous float64 CUDA tensors with layout (N, P, 3).
    Disable FMA to preserve the separate multiply/add rounding of PyTorch.
    """
    import torch

    shape = (moving.shape[0], fixed.shape[0], fixed.shape[1])
    forward = torch.empty(shape, dtype=torch.float64, device=fixed.device)
    reverse = torch.empty_like(forward)
    grid = (triton.cdiv(shape[1] * shape[2], 256) * shape[0],)
    with torch.cuda.device(fixed.device):
        _squared_distances[grid](fixed, moving, forward, reverse, shape[1],
                                 P=shape[2], BLOCK=256, enable_fp_fusion=False)
    return forward, reverse


if triton is not None:
    @triton.jit
    def _point_distance(F, M, fixed, moving, point, mask,
                        P: tl.constexpr, REVERSE: tl.constexpr):
        fi = (fixed * P + point) * 3
        order = P - 1 - point if REVERSE else point
        mi = (moving * P + order) * 3
        dx = tl.load(F + fi, mask, other=0.) - tl.load(M + mi, mask, other=0.)
        dy = tl.load(F + fi + 1, mask, other=0.) - tl.load(M + mi + 1, mask, other=0.)
        dz = tl.load(F + fi + 2, mask, other=0.) - tl.load(M + mi + 2, mask, other=0.)
        return (dx * dx + dy * dy) + dz * dz

    @triton.jit
    def _mean_squared(F, M, Forward, Reverse, NF, COUNT,
                      P: tl.constexpr, WIDTH: tl.constexpr, STEPS: tl.constexpr,
                      BLOCK: tl.constexpr, PROBABILITY: tl.constexpr,
                      INV_SIGMA: tl.constexpr):
        pair = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        lane = tl.arange(0, WIDTH)
        fi = pair[:, None] % NF
        mi = pair[:, None] // NF
        point = lane[None, :]
        mask = pair[:, None] < COUNT
        # Match Reduce.cuh's lane accumulation then ascending shuffle tree.
        forward = _point_distance(F, M, fi, mi, point, mask, P, False)
        reverse = _point_distance(F, M, fi, mi, point, mask, P, True)
        if P > WIDTH:
            mask2 = mask & (point + WIDTH < P)
            forward += _point_distance(F, M, fi, mi, point + WIDTH, mask2, P, False)
            reverse += _point_distance(F, M, fi, mi, point + WIDTH, mask2, P, True)
        for step in tl.static_range(0, STEPS):
            af, bf = tl.split(tl.reshape(forward, (BLOCK, WIDTH // (1 << (step + 1)), 2)))
            ar, br = tl.split(tl.reshape(reverse, (BLOCK, WIDTH // (1 << (step + 1)), 2)))
            forward = af + bf
            reverse = ar + br
        inv = tl.full((), 1.0 / P, tl.float64)
        forward = tl.reshape(forward, (BLOCK,)) * inv
        reverse = tl.reshape(reverse, (BLOCK,)) * inv
        if PROBABILITY:
            distance = tl.minimum(forward, reverse, propagate_nan=tl.PropagateNan.ALL)
            values = libdevice.exp(-distance * tl.full((), INV_SIGMA, tl.float64))
            tl.store(Forward + pair, values, pair < COUNT)
        else:
            tl.store(Forward + pair, forward, pair < COUNT)
            tl.store(Reverse + pair, reverse, pair < COUNT)


def mean_squared_distances(fixed, moving):
    """Fuse point reduction for P<=32, matching PyTorch's CUDA sum order."""
    import torch

    points = fixed.shape[1]
    if points > 32:
        forward, reverse = squared_distances(fixed, moving)
        return forward.sum(-1) / float(points), reverse.sum(-1) / float(points)
    shape = (moving.shape[0], fixed.shape[0])
    forward = torch.empty(shape, dtype=torch.float64, device=fixed.device)
    reverse = torch.empty_like(forward)
    steps = points.bit_length() - 1
    count = forward.numel()
    with torch.cuda.device(fixed.device):
        _mean_squared[(triton.cdiv(count, 32),)](
            fixed, moving, forward, reverse, shape[1], count,
            P=points, WIDTH=1 << steps, STEPS=steps, BLOCK=32,
            PROBABILITY=False, INV_SIGMA=1., enable_fp_fusion=False)
    return forward, reverse


def pairwise_probabilities(fixed, moving, sigmasq):
    """For P<=32, fuse distance, minimum and exp; PyTorch sums over fibers."""
    import torch

    points = fixed.shape[1]
    output = torch.empty((moving.shape[0], fixed.shape[0]),
                         dtype=torch.float64, device=fixed.device)
    steps = points.bit_length() - 1
    with torch.cuda.device(fixed.device):
        # One warp keeps exp lanes occupied after the point reduction.
        _mean_squared[(triton.cdiv(output.numel(), 32),)](
            fixed, moving, output, output, fixed.shape[0], output.numel(),
            P=points, WIDTH=1 << steps, STEPS=steps, BLOCK=32,
            PROBABILITY=True, INV_SIGMA=1. / sigmasq,
            num_warps=1, enable_fp_fusion=False)
    return output
