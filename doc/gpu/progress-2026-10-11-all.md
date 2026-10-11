# Four remaining optimizations delivered (2026-10-11)

Baseline: previous commit `7a5e5b8`. Original algorithms are referenced from
`/data/software/whitematteranalysis`. Float64 defaults, random sampling, optimizers,
and atlas format retain the existing conventions.

## Implementation

1. **B-spline transforms:** Added `gpu/bspline.py`. Each optimization caches fixed
   fibers and moving-fiber grid indices/cubic weights. Triton applies displacement
   fields on GPU, preserving VTK Zero-border behavior, float32 coefficients, and
   float64 operation order with FMA disabled. Coefficient filtering remains on
   CPU through VTK; only the small coefficient grid is uploaded each evaluation.
   Without Triton, VTK's bulk C++ `TransformPoints` replaces Python point loops,
   also benefiting CPU execution. See the [VTK 9.1 implementation](https://github.com/Kitware/VTK/blob/v9.1.0/Filters/Hybrid/vtkBSplineTransform.cxx).
   Fixed nonrigid Cobyla progress logging that referenced an unimported `resource`
   module, and avoided modulo-by-zero for budgets below 10. Caches are released
   on success or exception.
2. **Fused registration distance reduction:** For 1–32 points, emit forward and
   reversed mean squared distances per fiber pair directly, avoiding the
   `(moving,fixed,points)` intermediate tensor. Lane accumulation and adjacent
   reduction match local PyTorch 2.4.1 `Reduce.cuh`; division uses a float64
   reciprocal multiplication as in [PyTorch CUDA scalar division](https://github.com/pytorch/pytorch/blob/v2.4.1/aten/src/ATen/native/cuda/BinaryDivTrueKernel.cu).
   Above 32 points, retain the previous kernel and PyTorch reduction. Without
   Triton, use PyTorch. Direction selection, probabilities, and entropy are unchanged.
3. **Streaming atlas embedding:** Normalize, embed, and classify each similarity
   block immediately; CPU retains final results only. The default block contains
   2048 input fibers, overridable by the existing tile setting. OOM retries stop
   if one fiber still fails. Upload atlas constants once and compute only the
   final `number_of_eigenvectors+1` eigenvector columns rather than all 2500.
   Preserve row-sum/output conventions and propagate explicit `gpu_float32` correctly.
4. **Cluster splitting and writing:** Reuse vectorized `filter.mask`, preserving
   fiber order, attribute conversions, and empty clusters. Retain a CPU-only fork
   process pool: children inherit read-only VTK objects, and queues carry only
   indices and filenames. Remove NumPy serialization and VTK reconstruction,
   preserving field data, topology, and active attributes. Children never access
   CUDA; inputs must remain unmodified until return. Threads measured slower and
   were not adopted. Temporarily disable and then restore stdin during pool
   creation because Slicer PythonQt stdin lacks `close()`, which otherwise breaks
   child startup.

## Validation

- Fifteen tests passed across `tests/test_gpu_registration.py`, `test_bspline.py`,
  `test_atlas_blocks.py`, and `test_output_pipeline.py`, covering nondefault CUDA
  devices, OOM retry/termination, full optimizer traces, exception cleanup, VTK
  file bytes, attributes, and empty clusters.
- B-spline resolutions 2/3/4/6/10 cover grid nodes, ULP neighbors, and out-of-bounds
  coordinates. GPU and bulk CPU coordinates match pointwise VTK bitwise. Nonrigid
  Cobyla with budget 100 matches the no-Triton evaluation sequence and final transform.
- Distance reductions cover 1/2/3/7/8/10/15/16/20/31/32/50 points. Forward/reversed
  means, probabilities, and entropy match PyTorch bitwise; affine Cobyla/BFGS/Powell
  trace comparisons passed.
- Atlas comparisons passed for four methods, bilateral mode, float32/float64,
  partial tiles, and empty results. Labels and final VTP files for 10000 real
  input fibers with an ORG 800 atlas match the previous version. Double embedding
  values differ by approximately 2e-14 from GEMM rounding; arbitrary-data bitwise
  equivalence is not claimed.
- Existing GPU consistency and CPU-fast checks passed; sources compiled.
- Without Triton: 13 of 15 passed, two skipped. Without torch/Triton: CPU checks
  passed and GPU tests skipped according to availability.
- All six modules and in-process tests passed in real Slicer 5.2.2; device-switching
  tests conditionally skipped with one visible GPU. The old widget stub imports
  a deleted module and was not counted as a pass.

## Single RTX 4090 performance

Input: `/data/yijie/TractData/val_dsi_studio/250932.vtk`, float64, seed 42.
B-spline: 1000 fibers × 10 points, 6³ grid. Splitting/writing: 10000 fibers,
22 clusters. Atlas: 10000 fibers, 2500 Nystrom samples from ORG-800FC-100HCP,
10-dimensional embedding. CPU and B-spline timings are medians of three runs;
atlas is a single run including conversion, embedding, and classification.

| Operation | Previous | This round | Improvement |
|---|---:|---:|---:|
| B-spline transform, coefficient preparation and GPU return included | 39.15 ms | 1.81 ms | 21.6× |
| Bulk VTK transform without Triton | 39.15 ms | 4.06 ms | 9.65× |
| GPU objective, 3000 fixed × 1000 moving × 10 points | 2.672 ms | 2.373 ms | 1.13× |
| Peak additional registration tensor memory | 572.43 MiB | 114.67 MiB | 80.0% lower |
| Cobyla, 30 evaluations including initialization | 77.36 ms | 68.08 ms | 1.14× |
| Atlas similarity through embedding/classification | 0.615 s | 0.396 s | 1.55× |
| Peak additional atlas tensor memory | 7124.59 MiB | 1879.58 MiB | 73.6% lower |
| Cluster splitting with point/cell data | 0.469 s | 0.086 s | 5.44× |
| VTP writing with 8 processes | 0.430 s | 0.377 s | 1.14× |

Atlas labeling alone for the full **100000 input fibers**, producing in-memory
output polydata, took **3.95 s**. This excludes registration, preprocessing,
outlier removal, and writing 800 cluster files. It cannot be compared directly
with the earlier 21 s full pipeline result for 83348 fibers. These are component
benchmarks; initial JIT adds overhead, and load, disk caches, and input size affect gains.

Reproduce:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES=0 python tests/bench_remaining_pipeline.py --full
CUDA_VISIBLE_DEVICES=0 python tests/bench_registration_kernel.py --baseline 7a5e5b8
```

Python: `/data/yijie/miniconda3/envs/wma/bin/python`.
Logs: `/tmp/wma-all-{final-tests,final-slicer-clean,consistency,cpu,no-triton,no-torch,pipeline-bench,reduction-bench}.log`.
This checkpoint updates repository sources and tests. Local Slicer uses the new
source through a symlink; the existing self-contained `.sx` archive was not rebuilt
and requires separate packaging before distribution.
