# StrictSimilarity, laterality streaming and atlas reference I/O

## Scope and changes

The optional Triton distance kernel now supports float64 StrictSimilarity for
contiguous CUDA fibers with at most 32 points. It fuses component differences,
threshold clipping, per-point exponentials, point products, orientation maximum
and optional bilateral maximum. Product order matches the eager Torch path;
floating-point fusion remains disabled. Float32, longer/strided fibers,
nonpositive/NaN sigma-squared and environments without Triton retain the eager
Torch path. Empty results return without launching a kernel.

CPU/GPU laterality now requests row totals from bounded similarity tiles.
NumPy still performs each full fixed-fiber row sum, preserving that backend's
summation order. The matrix-returning APIs remain available by default.
Result storage changes from two whole-brain matrices to vectors plus tile
storage. CPU hemisphere counts and VTK scalar insertion are also vectorized.
No precision reduction, new dependency, configuration-file modification or
MRML export was added.

Actual previously failing cases were repaired: GPU laterality supplied a NumPy
type to `torch.as_tensor`; `FiberArray` lacked the documented 0.95 hemisphere
threshold; and laterality tested a NumPy index array with `== None`. The
optional-index calculation now selects nonzero denominators rather than
bitwise-combining two lists of indices. Custom hemisphere thresholds survive
fiber subsets.

Distance, similarity, strict-laterality and registration out-of-memory recovery
now halves the effective attempted rows, including an oversized requested tile,
and raises when one actual row cannot fit. It no longer keeps retrying a
one-row distance tile indefinitely. Nonpositive explicit registration chunks
raise a clear error. Existing optimizer workspaces retain reduced budgets.

Atlas outlier references use `io.read_polydata(..., geometry_only=True)`.
Each thread reuses its own XML reader, disables known point/cell arrays before
subsequent reads, drops attributes and returns independent copied geometry.
The reader's output is released after copying, callback observers are removed,
and each read refreshes file information, including overwritten filenames.
Unknown arrays in a changed schema may be decoded once before being discarded.
Default reads preserve the previous metadata behavior. Legacy `.vtk` geometry
reads discard attributes after reading; the measured I/O optimization is for
XML reference VTPs. Atlas caches invalidate automatically after this source
change. Subject reads and reference files themselves are unchanged.

## Controlled GPU kernels

Five alternating repeats compare 2,048 query fibers with 2,500 fixed fibers,
threshold 2 mm and sigma-squared 6,400. Compilation and first use are outside
timing. Every returned value is checked for exact equality with `ff8bacb`.
The workload is synthetic, float64, one RTX 4090, Linux, Python 3.8,
PyTorch 2.4.1 + CUDA 12.1 and Triton 3.0.0, four pinned CPU cores and
single-threaded BLAS.

| Points | StrictSimilarity baseline / new | Speed ratio | Extra peak tensor allocation baseline / new |
| --- | ---: | ---: | ---: |
| 10 | 24.861 / 6.540 ms | 3.80x | 1602.5 / 40.0 MiB |
| 15 | 36.524 / 8.228 ms | 4.44x | 2384.0 / 40.0 MiB |
| 20 | 48.924 / 11.619 ms | 4.21x | 3168.0 / 40.0 MiB |

These ratios apply to StrictSimilarity, not the default ORG pipeline as a whole.
Mean, MeanSquared and Hausdorff remain effectively unchanged in the same paired
comparison. A real 100,000-fiber atlas-labeling case gives 0.6375 s before and
0.6359 s after, with labels, embeddings and serialized cell attributes exactly
equal. This supports absence of a regression, not a material labeling speedup.
The benchmark now uses the baseline's native bilateral API where available and
synchronizes/records memory on the selected device.

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests
export CUDA_VISIBLE_DEVICES=1 WMA_CPU_THREADS=4
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1

taskset -c 0-3 python tests/bench_distance_launch.py \
  --baseline ff8bacb --repeat 5 \
  --input "$TRACTOGRAPHY" --atlas "$ATLAS_DIRECTORY" --fibers 100000 \
  --output "$NEW_GPU_REPORT"
```

## Complete laterality computation

A real DSI tractography is downsampled deterministically to 2,048 input fibers;
equal hemisphere selection retains 1,508. Five alternating repeats compare the
full-matrix implementation with streamed totals and the fused kernel. All
indices, left/right totals, hemisphere labels, geometry, attribute values,
array types and active roles are exactly equal within each backend.

The preceding default pipeline did not complete because of the bugs above.
This comparison supplies the corrected initialization/index logic to both
implementations and corrects the GPU dtype. The GPU baseline uses the intended
eager Torch algorithm without Triton. It is a corrected full-matrix reference,
not a claim that the released preceding GPU laterality path worked.

| Backend | Corrected full-matrix baseline / new | Extra peak device tensors baseline / new |
| --- | ---: | ---: |
| GPU | 44.480 / 20.098 ms | 706.06 / 10.06 MiB |
| CPU matrix, four workers | 283.917 / 279.391 ms | Not measured |

GPU complete laterality is 2.21x faster on this sample; CPU wall time is
essentially unchanged. Both backends avoid retaining the quadratic result
matrices. These exact comparisons are against each backend's own reference;
CPU and GPU floating-point outputs are not claimed bit-identical to each other.

```bash
taskset -c 0-3 python tests/bench_laterality.py \
  --input "$TRACTOGRAPHY" --fibers 2048 --backend gpu --repeat 5 \
  --output "$NEW_LATERALITY_REPORT"
```

Use `--backend cpu` for the CPU matrix comparison. Full fixed-fiber rows are
still compared; there is no neighbor approximation or scientific subsampling
inside the calculation beyond the explicitly requested input sampling.

## Reference reading and complete ORG outlier checks

A paired warm-file-cache test reads 32 actual ORG cluster VTPs five times in
alternating order. Full-data reading takes 0.4600 s; geometry-only reading takes
0.4067 s (11.6% faster). Returned VTK payload estimates decrease from 49,724 to
30,097 KiB (39.5%); these are returned-object sizes, not process peak RSS.
Coordinates and all connectivity/offset arrays are exactly equal.

A fresh-reader geometry-only variant was also measured and rejected: explicitly
reading information for every new reader took 0.5208 s versus 0.4522 s for full
reading. Reusing readers is necessary for the delivered gain.

```bash
WMA_GPU=off taskset -c 0-3 python tests/bench_geometry_reader.py \
  --atlas "$ATLAS_DIRECTORY" --stride 25 --repeat 5 \
  --baseline ff8bacb --output "$NEW_READ_REPORT"
```

The complete clinical GPU outlier stage is rerun with four processes and an
empty/populated reference cache: 800 categories, 245,946 input fibers and
130,737 retained fibers. All 800 encoded VTP files and the complete statistics
log are byte-identical to the preceding baseline in both cases. Cold/warm wall
times are 29.68 / 23.92 s. Separately measured preceding cache runs recorded
29.11 / 23.65–23.89 s. **A full outlier-stage speedup was not established.**
Shared-workstation I/O, process/CUDA startup and cache writing remain material;
the controlled reference-reader gain must not be presented as an equivalent
whole-stage or complete-pipeline improvement.

## Validation

All 63 targeted source tests pass, covering matrix metrics and precisions,
fused/eager equality, probability scales and special values, selected devices,
empty dimensions, row-sum equivalence, bounded result allocation, complete
laterality outputs, hemisphere boundaries/defaults/subsets, reader ownership,
schema/file changes, callback cleanup, concurrent readers, atlas cache
invalidation, parallel outlier masks, effective-row OOM retries/termination,
registration, geometry preparation and CLI file-stage metadata/outputs.
