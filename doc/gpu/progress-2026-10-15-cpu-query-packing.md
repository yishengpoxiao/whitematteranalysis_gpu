# Packed CPU queries and effective concurrency

## Implementation

For compiled float64 distances, separated query coordinates are packed once per
matrix invocation into point-major rows of three coordinates. Existing contiguous
coordinate rows retain their storage. Registration packs transformed moving
coordinates in the same way. This preserves values, coordinate order, point
summation order, caller storage and explicit row overrides.

SciPy consumes strided inputs directly. Its
[SciPy 1.10.1 distance loop](https://github.com/scipy/scipy/blob/v1.10.1/scipy/spatial/src/distance_metrics.h)
and [Slicer-compatible SciPy 1.9.2 loop](https://github.com/scipy/scipy/blob/v1.9.2/scipy/spatial/src/distance_metrics.h)
have a specialized branch when both coordinate strides are one element. Packing
separated queries enables this branch alongside the already packed references.
The arithmetic still iterates the three coordinates in their original order.

For pools of at least 16 workers and 8–32-point fibers, non-StrictSimilarity
metrics use a two-buffer tile estimate with the existing 16 MiB multi-worker
budget. Compiled double point planes are reused; float32 fallback buffers are
smaller. Other shapes and smaller pools keep their three-buffer estimate.
StrictSimilarity retains its existing smaller tiles and query layout in large
pools because the larger-tile/packing experiments regressed that workload.

Automatic worker selection now uses
[joblib.cpu_count](https://joblib.readthedocs.io/en/stable/generated/joblib.cpu_count.html),
which observes CPU affinity and scheduler quotas where supported, capped at 16.
Explicit job counts and positive `WMA_CPU_THREADS` values retain precedence.
There are no dependency or configuration-file changes.

The Slicer Pipeline's `run_atlas_labeling` previously defaulted to one job even
when accelerated CPU execution was selected. Its default is now `None`, allowing
the CPU backend's configured/automatic parallelism to apply. Explicit one-job
calls still run serially. GPU classification is unaffected; original joblib
mode retains joblib's default handling of `None`.

## Paired CPU compute measurements

Baseline: `bf9fd0e`. Seven alternating warm repeats on Xeon Gold 6330 use 16
pinned physical CPU cores, single-threaded BLAS, NumPy 1.20.3 and SciPy 1.10.1.
Each matrix has 512 separated-coordinate queries, 2,500 references, 15 points
and zero threshold. Full matrices match exactly in every repetition.

| Workers | Metric | Bilateral | Previous / current |
| --- | --- | --- | ---: |
| 4 | Mean | No | 63.82 / 52.76 ms |
| 4 | MeanSquared | No | 56.73 / 47.77 ms |
| 4 | Hausdorff | No | 52.85 / 44.15 ms |
| 16 | Mean | No | 47.99 / 41.78 ms |
| 16 | MeanSquared | No | 45.56 / 40.15 ms |
| 16 | Hausdorff | No | 33.40 / 32.01 ms |
| 16 | Mean | Yes | 78.51 / 62.74 ms |
| 16 | MeanSquared | Yes | 83.81 / 67.40 ms |

Four-worker registration objectives reuse their pool and retain exact scalar
values: 54.20 / 41.45 ms for 2,500 fixed / 1,000 moving fibers, and
197.60 / 153.87 ms for 5,000 / 2,000 (approximately 22–24% lower time).
A complete 30-evaluation CPU COBYLA run has an exactly identical objective
history and final transform, and leaves its fixed/moving input arrays unchanged.

## Real ORG atlas labeling

The same real DSI tractography is deterministically sampled with seed 1,000.
Seven alternating repeats include resampling, distances, normalization,
projection and centroid labels; reading and atlas loading are outside timing.
All five returned arrays match exactly, including embeddings, labels,
distances and row/column sums.

| Fibers | Workers | Previous / current | Time reduction |
| --- | --- | ---: | ---: |
| 2,048 | 4 | 0.7675 / 0.7321 s | 4.6% |
| 2,048 | 16 | 0.4575 / 0.4060 s | 11.3% |
| 8,192 | 4 | 3.0285 / 2.8956 s | 4.4% |
| 8,192 | 16 | 1.8772 / 1.5459 s | 17.6% |

A separate automatic-mode comparison restricts the process to four physical
CPUs with `WMA_CPU_THREADS=0`. The old backend selects 16 workers; the current
backend selects four. Labeling 2,048 fibers takes 0.8593 / 0.6205 s (27.8% lower),
with all five output arrays identical. This includes both affinity-aware worker
selection and query packing, rather than isolating either change.

An initial unconditional query-packing prototype regressed the 16-worker
8,192-fiber case by about 5%. The paired results above use the final larger
non-StrictSimilarity tiles. Larger StrictSimilarity tiles regressed a bilateral
matrix by about 26%, so that policy was rejected. No uniform gain across all
metrics, thread counts, machines or operating systems is claimed.

A worker scan of real labeling and registration showed that 16 workers still
outperformed eight on these workloads. The automatic 16-worker cap is retained.
The result from the preceding Hausdorff-only experiment is insufficient to lower
that cap globally. No complete shell pipeline or GPU speed ratio is inferred.

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests
export WMA_GPU=off WMA_CPU_THREADS=4 CUDA_VISIBLE_DEVICES=
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1

taskset -c 12-27 python tests/bench_cpu_point_major.py \
  --baseline bf9fd0e --repeat 7 --queries 512 --workers 4 16 --points 15 \
  --layout coordinate --threshold 0 --seed 86 --output "$NEW_CPU_REPORT"
```

The benchmark now accepts coordinate layout and threshold to reproduce the
real CPU backend layout. Timing on a shared workstation can vary; these are
paired local measurements rather than universal latency guarantees.

## Validation

Sixteen source CPU tests pass, with targeted layout-policy checks rerun after
the StrictSimilarity policy decision. Coverage includes read-only separated,
contiguous and negative-stride queries; 1/15/129-point fibers; full compiled tiles
and NumPy tails; all metrics and bilateral modes; zero/nonzero thresholds;
NaN/infinite coordinates; float32/float64; one/four/eight/sixteen workers;
original point-sum order and objective values; and optimizer pool cleanup.
Five extension core tests pass, including a new check that default CPU labeling
uses the configured worker count, explicit serial execution remains serial and
both produce exactly equal labels and cell attribute values.

The actual extension labeling API is also measured with the current CPU kernel:
forcing its prior one-job setting versus its new automatic 16-worker setting
labels 2,048 real fibers in 1.8135 / 0.4252 s (five alternating warm repeats,
4.26x). This includes preprocessing and output attribute formatting. Labels,
colors and every cell attribute array match exactly. This isolates the serial
job override using the current kernel, rather than timing a complete earlier
Slicer release or a whole pipeline.
