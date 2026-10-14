# GPU similarity fusion, normalization and CPU tile budgets

## Changes

The optional float64 short-fiber CUDA kernel converts the final distance to a
Gaussian similarity before storing it. Point arithmetic, thresholding,
forward/reverse and bilateral comparisons retain their original order. The
Gaussian retains the original separate negation/reciprocal multiplication and
libdevice exponential, with FMA disabled. StrictSimilarity keeps its existing
product metric. Tensor similarity callers use this path; NumPy-returning APIs
retain their existing host exponential. The eager tensor path applies the same
negation/division/exponential in its owned distance buffer. Nonpositive/NaN
Gaussian sigma-squared retains eager PyTorch semantics, including division by
zero; inputs remain unchanged.

A new optional `embedding_kernel.py` normalizes a large similarity block as
`B * (a[:, None] * c[None, :])` in one launch. It eliminates the full weight
matrix while preserving multiplication grouping, separate rounding and the
output's physical layout. Normal C-contiguous and transposed-contiguous float64
CUDA blocks are supported, including strided weight vectors. Blocks smaller
than 1,048,576 elements, other layouts/precisions and absent Triton retain the
PyTorch expression. The threshold avoids the measured overhead on small blocks.
No caller's similarity matrix or weights are mutated.

CPU tile sizing accounts for three point buffers rather than six, retaining the
existing 32 MiB single-worker / 16 MiB multi-worker budget. Compiled point planes
are reused and the NumPy fallback needs at most three point buffers. This reduces
small-tile dispatch overhead without changing arithmetic, reduction order or
public explicit tile overrides. Thread counts, precision and configuration files
are unchanged. Peak usage of other pipeline stages is not covered by this budget.

## GPU measurements

Baseline: `8f83f6f`. Seven alternating warm repeats use an RTX 4090, four pinned
physical CPU cores, single-threaded BLAS, PyTorch 2.4.1+cu121 and Triton 3.0.0.
CUDA is synchronized around timings. Peaks measure additional live PyTorch tensor
allocations above the inputs, not total reserved/device VRAM. First imports and
JIT compilation are outside timing.

| Operation | Previous / current time | Previous / current extra tensor peak |
| --- | ---: | ---: |
| Embedding 100,000 synthetic similarity rows, 2,500 atlas references | 65.138 / 61.847 ms | 80.34 / 40.84 MiB |
| Classifying all 100,000 real DSI fibers against ORG | 654.419 / 652.029 ms | 120.26 / 81.70 MiB |

The embedding computation is 5.1% faster with a 49.2% smaller peak. Actual
classification has a 32.1% smaller peak, while its time changes by less than 1%:
this does not establish a meaningful whole-classification timing improvement.
Full embeddings, labels, distances, row/column sums and serialized cell attribute
values match exactly. Real classification includes resampling, similarities,
embedding, nearest-centroid labels and output attributes, excluding file reading,
atlas loading, registration, saving and scene creation.

An earlier normalization-only probe measured approximately 13% faster embedding
on an 8,192-row transposed block, but this is not a whole-classification speedup.
The complete fused measurement above is the final result used for delivery.
No native Windows/macOS or full clinical shell pipeline speedup is inferred.

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 WMA_CPU_THREADS=4
WMA_GPU=on CUDA_VISIBLE_DEVICES=0 taskset -c 0-3 \
  python tests/bench_gpu_embedding.py "$ATLAS_DIRECTORY" --input "$TRACTOGRAPHY" \
    --fibers 100000 --repeat 7 --baseline 8f83f6f --exact \
    --output "$NEW_GPU_REPORT"
```

## CPU measurements

Five alternating repeats compare the same frozen backend on four physical CPU
cores. Full matrices are exactly equal in all 48 profiles: 512 queries, 2,500
references, 10/15/20 points, all four metrics, bilateral on/off and one/four
workers. The table selects 20-point unilateral four-worker cases.

| Metric | Previous / current time | Time reduction |
| --- | ---: | ---: |
| Mean | 95.84 / 86.99 ms | 9.2% |
| MeanSquared | 82.11 / 72.17 ms | 12.1% |
| Hausdorff | 58.75 / 47.23 ms | 19.6% |
| StrictSimilarity | 123.30 / 113.96 ms | 7.6% |

Not every configuration improves. Ten-point Mean is approximately 1% slower and
15-point StrictSimilarity is unchanged in the selected four-worker unilateral
profiles. Registration objectives with reused workers are 52.87 / 53.54 ms at
2,500 fixed / 1,000 moving fibers, and 205.13 / 197.08 ms at 5,000 / 2,000;
there is no consistent registration timing gain. Every scalar remains exact.

Real ORG labeling with four CPU workers gives 0.6597 / 0.6422 s for 2,048 fibers
and 2.6480 / 2.6057 s for 8,192 fibers (medians of five paired repeats). All five
returned arrays match exactly, including embeddings, labels, distances and
row/column sums. These modest 2.7% / 1.6% timing reductions are shared-workstation
measurements, not universal speed guarantees. Python 3.8, NumPy 1.20.3 and
SciPy 1.10.1 are used for CPU measurements.

```bash
WMA_GPU=off CUDA_VISIBLE_DEVICES= taskset -c 0-3 \
  python tests/bench_cpu_point_major.py --baseline 8f83f6f --repeat 5 \
    --output "$NEW_CPU_REPORT"
```

## Numerical validation

Thirteen source CPU tests pass for explicit/automatic tiles, one/four workers,
small and compiled tiles, float64/float32, all metrics and bilateral modes,
NaNs/infinities, extreme scales, point reduction boundaries, original registration
objectives, original atlas outputs and empty subjects. Gaussian CUDA checks
compare the frozen backend across all metrics, both precisions, short/long fibers,
bilateral modes, tiny/large/zero/infinite/NaN sigma and nonfinite coordinates.
Normalization checks compare both matrix layouts, strided weights, small/empty
fallbacks, input immutability, the complete large embedding, selected CUDA devices
and live tensor cleanup.

Eight/sixteen-worker follow-up comparisons pin 16 physical cores and use 4,096
queries, 2,500 references, 15 points, unilateral distances, threshold 2 mm and
`RandomState(84)`. Five alternating repeats again return exactly equal matrices.

| Workers | Metric | Previous / current time |
| --- | --- | ---: |
| 8 | Mean | 344.51 / 309.51 ms |
| 8 | Hausdorff | 326.46 / 187.20 ms |
| 16 | Mean | 780.14 / 295.91 ms |
| 16 | Hausdorff | 569.12 / 307.17 ms |

The small previous tiles show substantial overhead at 16 workers in this case;
reducing tile dispatches lowers Mean time by 62.1% and Hausdorff by 46.0%.
More workers still do not guarantee better throughput: eight workers outperform
sixteen for the current Hausdorff case. Thread defaults are not changed based on
this limited matrix workload. The CPU benchmark now accepts query count, worker
counts, point counts, metric selection and seed to reproduce these profiles.

```bash
WMA_GPU=off CUDA_VISIBLE_DEVICES= taskset -c 12-27 \
  python tests/bench_cpu_point_major.py --baseline 8f83f6f --repeat 5 \
    --queries 4096 --workers 8 16 --points 15 --methods Mean Hausdorff --seed 84 \
    --output "$NEW_HIGH_THREAD_REPORT"
```

## Installed delivery

All 12 source CUDA distance, Gaussian, normalization and atlas embedding checks
pass, alongside the 13 source CPU checks described above. Their first CUDA run
includes JIT compilation of the parameter combinations.

The universal installer is built from `82fdc3f` and installed through its actual
UI in an isolated native Slicer 5.2.2 environment. All 40 targeted installed
checks pass without skips: 26 CPU and 14 CUDA tests, including streamed atlas
outputs, cache invalidation, parallel outlier removal, laterality and OOM retry
termination. Native versions are NumPy 1.23.4, SciPy 1.9.2, PyTorch 2.4.1+cu121
and VTK 9.1.0. A fresh Slicer process loads all five pages, compares CPU/CUDA
labeling, runs registration and passes four atlas/groupwise core tests. All 101
packaged and actually installed library/module source files match repository
bytes. The tested date-free installer is the delivered archive.
