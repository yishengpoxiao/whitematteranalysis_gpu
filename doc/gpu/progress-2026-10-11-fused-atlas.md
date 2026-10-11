# Fused atlas distances and pipelining (2026-10-11)

This round extends `b2ce685`, retaining float64 defaults and CPU random sampling.

## Changes

- Added optional `gpu/distance_kernel.py`. For contiguous float64 CUDA tensors
  with 1–32 points, fuse forward/reversed coordinate distances, thresholding,
  point reduction, and minimum-direction selection into a two-dimensional
  distance matrix. Supports Mean/MeanSquared/Hausdorff. Mean uses correctly
  rounded double sqrt; threshold constants are explicitly float64, FMA is
  disabled, and reduction order matches PyTorch. Hausdorff preserves NaN propagation.
  Float32, StrictSimilarity, longer fibers, noncontiguous inputs, and environments
  without Triton retain the PyTorch path. Bilateral logic is unchanged.
- Replaced two per-fiber colorsys loops in `_embed_to_rgb` with NumPy operations,
  preserving branch precedence, staged operation order, NaN handling, and input
  mutation conventions. CPU execution also benefits.
- Each B-spline optimization retains a VTK grid, float32 displacement array, and
  coefficient filter. Evaluations update only the array and VTK modification time.
  Coefficient filtering remains on CPU, preserving format and rounding rather
  than reimplementing the filter.
- A single CPU worker performs the original scipy.vq atlas classification while
  the main thread computes the next GPU block. At most one classification task
  is pending; GPU blocks do not accumulate. Worker exceptions propagate. NumPy
  sqrt for row sums remains on CPU to preserve results.
- Registration stores each tile's probabilities in one GPU vector and transfers
  them once after all tiles, reducing synchronization. Floor, normalization, log,
  and entropy summation stay on CPU to preserve optimizer traces. Additional
  resident GPU memory is eight bytes per moving fiber.

## Validation

- **All 20 repository unit tests passed.** Distance comparisons cover
  1/2/3/7/10/15/16/20/31/32/50 points, float64/float32, four methods, thresholds
  0/0.1/2/100, NaN, Inf, noncontiguous inputs, and nondefault CUDA devices.
- Atlas comparisons passed for four methods, bilateral mode, both precisions,
  and partial tiles. Existing OOM retry/termination checks passed. Labels and
  classification distances use the original CPU scipy.vq implementation.
- Color tests cover gray, tied RGB components, six branches, NaN, and empty input;
  results match the previous colorsys implementation bitwise.
- Cached VTK coefficients were updated with positive, negative, zero, and subsequent
  nonzero displacements on grids 2/3/4/6/10, matching rebuilt coefficients bitwise.
  Coordinates, nonrigid/affine optimizer traces, and exception cleanup tests passed.
- Without Triton: 17 passed, three conditionally skipped. Without torch/Triton:
  seven CPU tests passed and 13 GPU tests skipped.
- Existing GPU consistency and CUDA-hidden CPU-fast checks passed; sources compiled.
- All six modules loaded in real Slicer 5.2.2, and the 20-test in-process suite
  passed with two device-switching tests skipped in the single-GPU environment;
  exit code was 0. The stale widget stub still imports a deleted module and was
  neither repaired nor counted as a pass.
- With **100000 real input fibers**, labels, colors, double embeddings, and final
  VTP files match `b2ce685` **bitwise/bytewise**. This applies to tested inputs and
  does not guarantee arbitrary-data bitwise agreement with CPU BLAS rounding.

## Single RTX 4090 benchmarks

`tests/bench_fused_atlas.py --full` loads `b2ce685` source directly for comparison.
Distance, color, B-spline, probability reduction, and classification overlap use
medians of five warm runs; full atlas labeling uses three warm runs. GPU timings
synchronize before and after measurement. Default precision is float64.

| Operation | b2ce685 | This round | Ratio |
|---|---:|---:|---:|
| Mean distance block | 29.84 ms | 5.87 ms | 5.08× |
| MeanSquared distance block | 26.95 ms | 3.71 ms | 7.26× |
| Hausdorff distance block | 27.54 ms | 4.86 ms | 5.67× |
| Peak additional distance tensor memory | 1798–1837 MiB | 39–40 MiB | About 97.8% lower |
| Colors for 100000 fibers | 452.4 ms | 29.7 ms | 15.2× |
| B-spline transform, coefficient preparation and return included | 1.958 ms | 1.558 ms | 1.26× |
| Registration probability reduction and entropy, 16 tiles | 2.792 ms | 2.527 ms | 1.10× |
| Classification overlap only, both using the new distance kernel | 117.4 ms | 107.1 ms | 1.10× |
| Full atlas labeling, 100000 fibers | 4.056 s | 1.226 s | 3.31× |

Distance blocks: 2048 queries × 2500 fixed × 15 points, threshold 2.
B-spline: 1000 fibers × 10 points, 6³ grid. Registration: 1000×1000 fibers,
64 moving fibers per tile. Classification overlap: 10000 fibers, official ORG
atlas. Full labeling input: `/data/yijie/TractData/val_dsi_studio/250932.vtk`;
atlas: `/data/software/ORG-Atlases/ORG-800FC-100HCP`. Full timing includes input
DeepCopy, distances, embedding, CPU classification, colors, and output polydata
construction. It excludes disk reads, registration, outlier removal, and cluster
writing, so it is not a complete clinical pipeline timing.

Optional Triton has initial JIT overhead. Small tasks and different point counts
or thresholds require accounting for compilation costs; benefits depend on size
and machine load. CPU sqrt synchronization remains; classification overlap reduces
its impact without introducing lower precision.

Reproduce:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES=0 python tests/bench_fused_atlas.py --full
```

Python: `/data/yijie/miniconda3/envs/wma/bin/python`.
Logs: `/tmp/wma-next-{final-tests,final-bench,final-slicer,consistency,cpu,no-triton,no-torch}.log`.
At this checkpoint, local Slicer uses the new source through a symlink; the existing
self-contained `.sx` archive was not rebuilt.
