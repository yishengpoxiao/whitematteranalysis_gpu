# Reused eager GPU distance scratch

The eager PyTorch distance path reuses its temporary Y-distance buffer for
Z-distance arithmetic, retaining the original separate subtract, square and
add rounding. Point reductions reuse the owned squared-distance buffer for
threshold clipping, square roots and StrictSimilarity exponentials. Hausdorff
reduces the squared maximum before monotone threshold clipping. Forward/reverse
and bilateral comparisons, point reduction order and precision stay unchanged.
Input tensors are never modified.

This path applies when Triton is unavailable, to float32 distances, to
noncontiguous tensors and to fibers exceeding the fused kernel's 32-point
limit. The existing float64 short-fiber fused kernel is unchanged. Thus this
patch does not claim a speedup for the default Linux/Triton ORG pipeline.
No new dependencies or configuration changes are required.

## Paired measurements

Seven alternating repeats compare the frozen `97bc764` backend with the new
backend on an NVIDIA RTX 4090, using 512 queries, 2,500 reference fibers,
threshold 2 mm and sigma-squared 6,400. Triton is disabled for both sides.
CUDA is synchronized around timings. The peak is extra live PyTorch tensor
memory above the input/reference allocations, not total reserved VRAM. Every
returned matrix in every timed repeat matches exactly.

| Precision / points | Metric | Baseline / new | Extra tensor peak, baseline / new |
| --- | --- | ---: | ---: |
| float32 / 15 | Mean | 3.462 / 3.173 ms | 231.8 / 153.4 MiB |
| float32 / 15 | MeanSquared | 3.074 / 2.944 ms | 227.0 / 153.0 MiB |
| float32 / 15 | Hausdorff | 3.070 / 2.469 ms | 227.0 / 153.0 MiB |
| float32 / 15 | StrictSimilarity | 4.105 / 3.589 ms | 301.4 / 153.4 MiB |
| float64 / 15 | Mean | 7.535 / 7.529 ms | 459.5 / 302.9 MiB |
| float64 / 15 | MeanSquared | 6.841 / 6.839 ms | 449.4 / 303.4 MiB |
| float64 / 15 | Hausdorff | 6.928 / 5.647 ms | 449.4 / 302.9 MiB |
| float64 / 15 | StrictSimilarity | 9.117 / 9.104 ms | 595.7 / 302.9 MiB |
| float64 / 50 | Mean | 24.988 / 24.970 ms | 1,484.4 / 986.9 MiB |
| float64 / 50 | MeanSquared | 22.727 / 22.724 ms | 1,475.2 / 986.9 MiB |
| float64 / 50 | Hausdorff | 23.836 / 19.431 ms | 1,475.2 / 986.9 MiB |
| float64 / 50 | StrictSimilarity | 30.390 / 30.389 ms | 1,962.9 / 986.9 MiB |

Memory peaks fall by approximately 32–50%. Hausdorff time falls by 18–20%;
float32 Mean, MeanSquared and StrictSimilarity fall by 8%, 4% and 13%.
Other float64 times are effectively unchanged. These are distance-block
measurements; no whole-pipeline speed ratio is inferred.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests \
WMA_GPU=on CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python tests/bench_gpu_scratch.py --baseline 97bc764 --repeat 7 \
  --output "$NEW_GPU_REPORT"
```

## Numerical and integration validation

The frozen eager backend is compared against the new eager backend with both
precisions, 10/50-point fibers, contiguous/strided inputs, both bilateral modes,
zero/finite/infinite/NaN thresholds, very small/large/zero/negative/infinite/NaN
sigma-squared, exact self distances and NaN/infinite coordinates. Original
input tensors remain identical. Five CUDA distance tests also compare the
unchanged fused kernel against the new eager reference and verify selected
CUDA devices. Python 3.8 and PyTorch 2.4.1+cu121 are used for source checks.
