# Direct CPU point-plane reductions

## Implementation and numerical contract

Compiled float64 distance tiles retain SciPy's `(points, queries, fixed)` layout
through reduction. The preceding implementation transposed and copied the full
point-distance tile to make its point axis contiguous. Small and non-double
tiles retain their NumPy path; output matrices and public distance metrics keep
their existing layout and meaning.

Mean and MeanSquared reproduce NumPy's eight-accumulator point summation order,
including the remainder and aligned recursive split above 128 points. Owned
scratch planes hold the accumulators. Mean still clips before its square root,
divides by the point count and squares; its existing zero-threshold compiled
Euclidean path remains active.

The grouping follows the pairwise reduction in the pinned
[NumPy 1.20.3 implementation](https://github.com/numpy/numpy/blob/v1.20.3/numpy/core/src/umath/loops.c.src)
and [NumPy 1.23.4 implementation](https://github.com/numpy/numpy/blob/v1.23.4/numpy/core/src/umath/loops_utils.h.src).
The former starts its short sum at zero and the latter at negative zero;
distance planes are nonnegative, and short reductions use the running NumPy
implementation. The goal is the same result as the existing contiguous point
sum, not a different sequential or approximate reduction.

Hausdorff reduces the maximum before threshold subtraction/clipping. Clipping
is monotone, so the resulting distance is identical, while threshold operations
apply to the reduced matrix rather than every point. StrictSimilarity performs
negation, division and exponential in its owned float64 point buffer and reduces
products on the point axis, avoiding the full transpose and extra point buffers.
Direction and bilateral combination are unchanged. There is no new dependency,
configuration-file change or precision reduction.

## Paired compute benchmarks

Baseline: `592eefa`. Five alternating repeats use 512 query fibers, 2,500 fixed
fibers, threshold 2 mm, sigma-squared 6,400, float64, four pinned physical CPU
cores and single-threaded BLAS on Linux. All full returned matrices are checked
for exact equality. The table selects 15-point, four-worker cases from the full
10/15/20-point, one/four-worker, unilateral/bilateral benchmark.

| Metric | Bilateral | Baseline / new | Time reduction |
| --- | --- | ---: | ---: |
| Mean | No | 99.057 / 74.758 ms | 24.5% |
| Mean | Yes | 182.786 / 136.841 ms | 25.1% |
| MeanSquared | No | 85.495 / 62.197 ms | 27.2% |
| MeanSquared | Yes | 162.773 / 115.126 ms | 29.3% |
| Hausdorff | No | 106.116 / 44.453 ms | 58.1% |
| Hausdorff | Yes | 204.475 / 81.546 ms | 60.1% |
| StrictSimilarity | No | 124.264 / 87.882 ms | 29.3% |
| StrictSimilarity | Yes | 241.341 / 176.607 ms | 26.8% |

CPU registration objectives use 10-point fibers and a reused four-worker pool.
Every scalar objective is exactly equal to the preceding implementation.

| Fixed / moving fibers | Baseline / new objective | Time reduction |
| --- | ---: | ---: |
| 2,500 / 1,000 | 90.806 / 53.260 ms | 41.3% |
| 5,000 / 2,000 | 349.324 / 205.008 ms | 41.3% |

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests
export WMA_GPU=off WMA_CPU_THREADS=4 CUDA_VISIBLE_DEVICES=
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

taskset -c 0-3 python tests/bench_cpu_point_major.py \
  --baseline 592eefa --repeat 5 --output "$NEW_CPU_REPORT"
```

These are controlled compute measurements, not whole-pipeline speed ratios.
Timings include distance worker setup; registration objectives reuse a pool.

## Real atlas labeling

A real DSI tractography is sampled deterministically to 2,048 fibers and labeled
against the ORG atlas's 2,500-fiber reference and 11 projection columns. Five
repeats give the following median wall times:

| Backend | Labeling time |
| --- | ---: |
| Preceding CPU matrix, four workers | 0.8883 s |
| Original joblib, four workers | 7.2003 s |
| New CPU matrix, four workers | 0.6630 s |

The new matrix path reduces labeling time by 25.4% against the preceding matrix
path. Labels and serialized cell attributes match exactly; embeddings are also
checked against the per-fiber reference. Input reading and sampling are outside
timing. Python 3.8, NumPy 1.20.3 and SciPy 1.10.1 were used.

```bash
taskset -c 0-3 python tests/bench_cpu_pipeline.py \
  "$TRACTOGRAPHY" "$ATLAS_DIRECTORY" --baseline 592eefa \
  --fibers 2048 --repeats 5 --output "$NEW_LABEL_REPORT"
```

## Validation scope

Targeted CPU tests cover all four metrics, bilateral modes, one/four workers,
compiled tiles and small NumPy tails, float32 fallback, NaNs/infinities,
underflow/overflow scales, threshold/sigma special values, input immutability,
streamed atlas outputs and exact original registration objectives. Direct sum
checks cover every point count from 1 through 129, plus 257, 511, 512, 513 and
1,024, including the recursive block boundary. Atlas-cache invalidation and
parallel outlier output checks also pass.

## Full clinical pipeline

A fresh four-worker CPU run of `wm_apply_ORG_atlas_to_subject.sh` uses the same
3.56 GB clinical tractography, rigid-affine-fast registration, ORG atlas and
native Slicer transform hardening as the preceding frozen full CPU case. Atlas
probability cache is empty at startup. The input size and modification time
remain unchanged. All seven instrumented stages exit successfully.

| Stage | Preceding full CPU case | Current full CPU case |
| --- | ---: | ---: |
| Registration | 84.92 s | 71.25 s |
| Atlas classification | 132.98 s | 105.57 s |
| Outlier removal | 157.19 s | 78.52 s |
| Hemisphere assessment | 13.59 s | 16.17 s |
| Transform hardening | 34.15 s | 29.12 s |
| Hemisphere separation | 9.04 s | 8.79 s |
| Anatomical tract assembly | 14.85 s | 16.85 s |
| **Whole shell pipeline** | **448.67 s** | **327.97 s** |

Whole-pipeline wall time falls by 26.9%, with peak process RSS 9.84 GiB. This is
one end-to-end run per case. The preceding full case is frozen at `c9c622c`, so
the comparison includes the subsequent cache, geometry reader, hemisphere and
zero-threshold Euclidean improvements, as well as the point-plane reductions.
It does not isolate this patch. The paired compute and sampled labeling results
above use the immediately preceding `592eefa` backend and isolate this change.
Filesystem caches are not reset; a cold atlas probability cache does not imply
cold operating-system file caches. Hemisphere assessment and anatomical assembly
show run-to-run variation rather than a measured improvement.

All 2,474 output VTP files are byte-identical to the preceding full CPU case:
800 initial clusters (245,946 fibers), 800 outlier-removed clusters (130,737),
800 transformed clusters (130,737) and 74 anatomical tracts (93,766). All 11
saved transform files have exactly identical transform types, parameters and
fixed parameters. Completion markers and output catalogues are checked.
