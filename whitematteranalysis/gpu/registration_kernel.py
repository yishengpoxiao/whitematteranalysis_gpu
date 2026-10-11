"""Optional fused point distances; PyTorch retains all reduction operations."""

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


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
