# Short-vector CUDA centroid distances — 2026-10-15

## Changes

Profiling 16,384 real fibers after the geometry/CPU batch assigns approximately
61% of CUDA execution time to fiber distances and 33% to direct centroid
distances. Host copy timings include waiting for preceding CUDA work and should
not be interpreted as PCIe transfer time. Normalization/GEMM/reduction stages are
comparatively small in this profile.

Add an optional Triton kernel for float64 Euclidean distances between contiguous
CUDA embedding vectors and atlas centroids with 1–32 dimensions. Evaluate many
short-vector pairs per program rather than using a large CUDA block for each
individual pair. Keep direct coordinate differences, separate multiply/add
rounding, the descending reduction tree, and correctly rounded square roots.
Retain PyTorch's minimum operation and first-centroid tie behavior. The reduction
follows the [PyTorch 2.4.1 CUDA distance implementation](https://github.com/pytorch/pytorch/blob/v2.4.1/aten/src/ATen/native/cuda/DistanceKernel.cu).

Use the existing direct `torch.cdist` path when Triton is unavailable, vectors are
outside the supported dimensions/layout/dtype, or tensors use the CPU. Invalid
embeddings and centroids retain the existing validation. No dependencies,
configuration, precision defaults, or CPU algorithms change in this batch.

## Measurements

Compare with commit `28f254d` on one RTX 4090, Xeon Gold 6330, four pinned physical
CPU cores, one BLAS thread, PyTorch 2.4.1+cu121 and Triton 3.0.0. All operations use
float64. Alternate baseline/current runs and report medians of five warm runs.

| Operation | Previous | Current |
| --- | ---: | ---: |
| Embedding/labeling only, 100,000 random similarity rows | 0.31901 s | 0.06515 s |
| Full GPU classification, 100,000 real fibers | 0.95475 s | 0.70373 s |

The real-subject classification improves approximately 1.36x, reducing elapsed
time by 26%. Labels and serialized cell attributes match exactly. Floating
embedding comparisons pass at 1e-12 tolerance. The ORG atlas has 2500 reference
fibers, 800 centroids, and 10 embedding dimensions. Classification includes
resampling, similarities, normalization, projection, labels, and output attributes;
file reading, atlas loading, registration, saving, scene loading/rendering, and
first imports/JIT compilation are excluded. These are stage timings, not a new
full CLI pipeline benchmark. No native Windows/macOS speedup was measured.

## Validation

Targeted CUDA tests compare full distance matrices with direct PyTorch cdist
using exact equality for 1/2/3/4/5/7/10/15/16/20/31/32 dimensions, partial blocks,
and coordinate scales from 1e-100 to 1e160, including overflow. Additional checks
cover first-centroid ties, zero-dimensional/empty queries, CPU/float32/strided and
33-dimensional fallback, NaN/Inf propagation, selected-device execution, and
preserving the caller's current CUDA device. Existing atlas checks cover all
fiber metrics, both precision options, bilateral comparisons, invalid inputs,
input preservation, allocation retries, and empty streamed output.

Memory checks verify that centroid tensors return to the previous allocation
baseline. Update the existing cleanup test to compare with its pre-test baseline,
since preceding GEMMs can retain a PyTorch-owned cuBLAS workspace. This preserves
the zero-allocation expectation in an otherwise idle process while also testing
cleanup after prior atlas labeling. No device reset or private workspace-clearing
API is introduced.

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests
export WMA_CPU_THREADS=4 CUDA_VISIBLE_DEVICES=1
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1

taskset -c 0-3 python tests/bench_gpu_embedding.py "$LABELING_ATLAS_DIRECTORY" \
  --input "$INPUT_TRACTOGRAPHY" --fibers 100000 --repeat 5 --baseline 28f254d \
  --output /tmp/wma-centroid-classification.json

CUDA_VISIBLE_DEVICES=0,1 python -m unittest test_centroid_kernel test_atlas_blocks -v
```
