# Fused pointwise registration distance kernel (2026-10-11)

## Implementation

The optional Triton kernel in `gpu/registration_kernel.py` computes forward and
reversed pointwise squared distances together, reducing temporary dx/dy/dz tensors
and separate operator launches. FMA is explicitly disabled to preserve the staged
rounding of `(dx*dx + dy*dy) + dz*dz`. Point reduction, direction selection, exp,
fixed-fiber summation, and entropy retain the existing PyTorch operations.
Float64 remains the default.

Indexing and masking follow the [official Triton kernel tutorial](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html).
In the local Triton NVIDIA compiler, `enable_fp_fusion=False` controls floating-point
fusion in both LLVM and ptxas. The comparisons below validate actual rounding.

A one-dimensional launch grid avoids the moving-row limit of a two-dimensional
grid. Launches select the input tensor's CUDA device and restore the caller's
current device. Without Triton, the PyTorch distance path is used automatically;
no new mandatory dependency is required, and CPU behavior and random sampling
remain unchanged.

Temporary tile allocations now live in a separate function so references are
released when a tile fails. OOM retries also stop and rethrow the original exception
if a single moving fiber still fails, preventing infinite retries. The previous
conservative tile memory estimate is retained.

## Validation

- All seven unit tests passed with `CUDA_VISIBLE_DEVICES=0,1`.
- Eleven point counts were covered: 1/2/3/7/8/10/15/16/20/31/50. Forward/reversed
  pointwise results, probabilities, and tiled entropy match PyTorch bitwise,
  including partial tiles and probability underflow.
- Cobyla/BFGS/Powell evaluation sequences and final transforms match the no-Triton
  path bitwise. Input refresh, normal/exception cleanup, nondefault device
  selection, and OOM tile reduction/termination checks passed.
- Simulated Triton import failure: five passed and two conditionally skipped.
  Simulated torch and Triton import failure: the CPU test passed and six GPU
  tests conditionally skipped.
- Existing GPU consistency and CPU-fast comparisons passed; relevant sources compiled.
- All six modules loaded in real Slicer 5.2.2. Triton was available, and 30 Cobyla
  evaluations produced bitwise-identical sequences and final transforms with old
  and new kernels.

## Real-data performance

Baseline commit `406c585`, single RTX 4090, float64, input
`/data/yijie/TractData/val_dsi_studio/250932.vtk`. Seed 42 selects 4000 fibers:
3000 fixed × 1000 moving × 10 points. Objective paths were alternated for 20 runs
each; optimizers were run three times each. Medians are reported. CPU results are
returned before timing stops, so synchronization is included.

| Operation | Previous | This round | Improvement |
|---|---:|---:|---:|
| Warm complete GPU objective | 9.414 ms | 2.627 ms | 3.58× |
| Peak additional tensor memory, including transform and distance | 1167.60 MiB | 572.43 MiB | 51.0% lower |
| Warm Cobyla, 30 evaluations including input initialization | 0.2843 s | 0.0774 s | 3.67× |
| Warm Cobyla, default budget 300, actual 34 evaluations | 0.3212 s | 0.0869 s | 3.70× |

The first objective call in this process took 0.131 → 0.626 s, including runtime
initialization and loading existing disk caches; this is not uncached compilation
time. Initial JIT overhead may erase the benefits for short, one-off tasks. These
are local registration benchmarks, not equivalent full ORG pipeline speedups.

Reproduce (the script loads baseline source directly):

```bash
CUDA_VISIBLE_DEVICES=0 /data/yijie/miniconda3/envs/wma/bin/python \
  tests/bench_registration_kernel.py \
  --input /data/yijie/TractData/val_dsi_studio/250932.vtk --maxfun 300
```

Logs: `/tmp/wma-kernel-{tests,consistency,cpu,no-torch,no-triton,slicer,real-bench,real-300-bench}-2026-10-11.log`.
The stale Slicer stub still imports a deleted module and was neither fixed nor
counted as a pass; real Slicer validation passed independently. At this checkpoint,
the `.sx` archive was not rebuilt, and the local Slicer source symlink used the
new implementation. Nonrigid B-spline transform application still ran on CPU;
this kernel also accelerates its subsequent entropy objective calculation.
