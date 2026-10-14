# Registration workspaces and zero-threshold CPU Mean distances

## Changes

GPU affine and nonrigid optimization now own a pinned host probability buffer
for their lifetime. The initial tile budget is queried once per optimization;
an out-of-memory retry halves it and retains the smaller budget for subsequent
evaluations. Each new optimization/sample set gets a fresh budget and workspace.
A single successful tile uses its probability output directly, avoiding the
extra device-to-device copy into an assembly buffer. Multiple tiles still
assemble probabilities in the original moving-fiber order.

The probability vector is copied asynchronously to the pinned buffer and the
current stream on the selected device is synchronized before reading it.
Flooring, division, logarithm, negation and the final sum remain CPU PyTorch
operations in the same order, now using the owned buffer in place. The
stateless objective remains available. There is no CUDA graph, global tensor
cache, precision change, device reset or new dependency. Workspaces are owned
by the existing optimizer objects and released with them.

For CPU Mean distances at zero threshold, the compiled 3D point-distance call
uses SciPy's Euclidean metric, fusing the square root into that calculation.
The existing contiguous point-axis reduction, division, squaring and
forward/reversed/bilateral combination remain unchanged. Nonzero thresholds
retain squared point distances so clipping still precedes the square root.
Other distance metrics retain their previous path. The small/non-double NumPy
path keeps its original elementwise arithmetic.

## GPU measurements

The baseline is source `dfe1ca5`. Alternating repeated evaluations and three
paired COBYLA runs use identical fixed/moving data and transforms. Every
objective sequence and final optimized transform is checked for exact equality.
All checks pass. Timings use Linux, one RTX 4090, four pinned CPU cores,
single-threaded BLAS and PyTorch 2.4.1 + CUDA 12.1, with float64 registration.
Compilation and first-use startup are outside the repeated objective timing;
optimizer timing includes its setup. Workloads are synthetic 10-point fibers.

| Fixed / moving fibers | Backend | Objective baseline / new | 100-evaluation optimizer baseline / new |
| --- | --- | ---: | ---: |
| 1,000 / 250 | Triton | 0.4455 / 0.3893 ms | 0.05213 / 0.04324 s |
| 5,000 / 2,000 | Triton | 3.9998 / 3.9360 ms | 0.40681 / 0.39838 s |
| 1,000 / 250 | Torch CUDA without Triton | 0.9929 / 0.9244 ms | 0.10031 / 0.09413 s |

The small Triton optimizer is 17.1% faster; the large optimizer improves only
2.1%, as distance computation dominates. Extra peak device-tensor allocation
is unchanged in these measurements (1.912 / 76.325 MiB for the two Triton
sizes). Reused host buffers add 8 bytes per moving fiber.

Profiling 1,000 preceding small evaluations attributes approximately 35 ms to
repeated free-memory queries and 87 ms to CPU probability transfers. These are
profiling observations, not independently additive speedups.

Reproduce from the repository root:

```bash
export CUDA_VISIBLE_DEVICES=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests
export WMA_CPU_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1

taskset -c 0-3 python tests/bench_registration_kernel.py \
  --baseline dfe1ca5 --fixed 1000 --moving 250 --repeat 100 --maxfun 100
```

Use `--fixed 5000 --moving 2000 --repeat 50` for the larger case, or
`--no-triton` for the Torch fallback. These are optimizer measurements, not
complete pipeline times.

## CPU measurements and full output checks

Alternating five repeats compare the complete tiled CPU matrix with `dfe1ca5`,
including its worker setup. Inputs have 512 query fibers and 2,500 fixed fibers;
every returned distance is checked for exact equality.

| Points | Bilateral | Workers | Baseline / new |
| --- | --- | ---: | ---: |
| 10 | No | 1 | 176.55 / 161.65 ms |
| 15 | No | 1 | 245.13 / 222.01 ms |
| 15 | Yes | 1 | 486.65 / 440.83 ms |
| 15 | Yes | 4 | 141.03 / 128.84 ms |
| 20 | Yes | 4 | 172.59 / 160.54 ms |

A separate paired test selects 16 real ORG reference clusters evenly across
sorted fiber counts: 14, 532, 639, 702, 776, 840, 887, 946, 999, 1,063, 1,120,
1,175, 1,256, 1,354, 1,488 and 2,617. Four independent workers each use one
distance thread, with already resampled reference geometry and no file I/O
inside timing. Five alternating repeats give 3.7865 s before and 3.4655 s after
(8.5% faster), with every full pairwise matrix exactly equal.

Reproduce the bounded matrix comparison:

```bash
WMA_GPU=off taskset -c 0-3 python tests/bench_cpu_mean.py \
  --baseline dfe1ca5 --repeat 5 --output "$NEW_JSON_REPORT"
```

The complete clinical CPU outlier stage was also rerun with four processes:
800 categories, 245,946 input fibers, 130,737 retained fibers. All 800 encoded
VTP files and the complete statistics log are byte-identical to the preceding
implementation, for both an empty and a populated atlas cache. Wall times were
144.96 s cold and 42.06 s warm. The preceding separately timed release recorded
135.15 s cold and 44.27 s warm. These shared-workstation runs include reads,
cache writes, masking and serialization and are not paired timing evidence:
**a cold full-stage speedup was not established**. The controlled compute
measurements above support the kernel improvement; they do not imply an
equivalent whole-pipeline gain.

## Validation

Thirty-four targeted source tests pass. They cover all supported CPU distance
metrics and bilateral modes, one/four workers and partial tiles; the new Mean
path at zero threshold with 1–50 points, NaNs and extreme scales; streamed atlas
outputs; affine/nonrigid coordinates, complete optimizer sequences and failure
cleanup; eager Torch and Triton probability agreement; two selected CUDA devices;
nondefault-stream pinned transfer and workspace reuse; persistent smaller tiles
after simulated out-of-memory retries; and atlas caching/parallel outlier masks.

The whole-pipeline objective remains open: the controlled kernel improvements
above do not establish that every remaining optimization opportunity has been
exhausted. File I/O and large GPU distance calculations remain material costs.

The universal installer was rebuilt from source `7b8a14d` and installed through
its actual dialog in a fresh Slicer 5.2.2 Linux profile. All 100 packaged source
files are byte-identical between the repository, installer payload and installed
extension. All 34 targeted tests also pass against those installed modules under
Slicer's own Python/NumPy/SciPy/PyTorch environment, including the compiled Mean
path and two visible CUDA devices. No Windows/macOS runtime was available.
A separate fresh Slicer process also discovers and opens all five installed
module pages, checks CPU/GPU labeling and registration, and passes all four
installed core functional tests.
