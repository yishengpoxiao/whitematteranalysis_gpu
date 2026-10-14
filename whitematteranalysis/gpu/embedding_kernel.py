"""Normalize large similarity blocks without materializing a weight matrix."""
from .registration_kernel import triton

if triton is not None:
    import triton.language as tl

    @triton.jit
    def _normalize(B, A, C, Output, COUNT, N,
                   AS: tl.constexpr, CS: tl.constexpr, BLOCK: tl.constexpr):
        offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offset < COUNT
        row, column = offset // N, offset % N
        values = tl.load(B + offset, mask, other=0.)
        a = tl.load(A + row * AS, mask, other=0.)
        c = tl.load(C + column * CS, mask, other=0.)
        # Retain the original multiplication grouping and separate rounding.
        tl.store(Output + offset, values * (a * c), mask)


def normalized_similarities(B, a, c):
    """Return B * (a[:, None] * c[None, :]), preserving input values/layout."""
    import torch

    if (triton is None or not B.is_cuda or B.dtype != torch.float64
            or B.numel() < 1 << 20
            or not (B.is_contiguous() or B.t().is_contiguous())):
        return B * (a[:, None] * c[None, :])
    transposed = not B.is_contiguous()
    if transposed:
        B, a, c = B.t(), c, a
    output = torch.empty_like(B)
    with torch.cuda.device(B.device):
        _normalize[(triton.cdiv(B.numel(), 256),)](
            B, a, c, output, B.numel(), B.shape[1],
            AS=a.stride(0), CS=c.stride(0), BLOCK=256,
            enable_fp_fusion=False)
    return output.t() if transposed else output
