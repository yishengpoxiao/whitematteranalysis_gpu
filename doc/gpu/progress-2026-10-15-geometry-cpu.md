# Geometry preparation and compiled CPU distances — 2026-10-15

## Changes

- Read VTK connectivity and native point coordinates as NumPy views rather than
  copying every point and connection for each sampling/masking operation. Cast
  only the selected coordinates used in length and endpoint calculations to
  float64, retaining the original double arithmetic and threshold decisions.
- Gather lengths only for selected lines. Avoid redundant dtype copies after
  indexing output geometry and attributes. Keep deep VTK output copies so output
  nodes remain independent of the source geometry and its lifetime.
- Fill the existing float64 R/A/S resampling arrays directly, removing an
  intermediate three-component float64 array and subsequent component copies.
- For float64 CPU distance tiles containing at least 4096 fiber pairs, use
  SciPy's compiled three-coordinate squared-Euclidean distance implementation.
  Restore a contiguous point axis before the original NumPy point reduction.
  Smaller tiles and other dtypes retain the NumPy path. SciPy is already required;
  no dependency or configuration changes are needed.
- Retune CPU tile budgets to 32 MiB with one worker and 16 MiB per worker with
  parallel execution. These are conservative working-set estimates, not process
  memory limits. Very large reference sets can still require single-row tiles.
- Retune the optional CUDA distance kernel to one warp, with 64 fiber pairs per
  program for up to 16 points and 32 pairs for longer supported fibers. Point
  arithmetic/reduction order remains unchanged. The existing PyTorch fallback
  still handles unsupported layouts, precision options, and unavailable Triton.

## Measurements

Compare with commit `710f7d4` on a Xeon Gold 6330, four pinned physical CPU cores,
one BLAS thread, RTX 4090, NumPy 1.20.3, SciPy 1.10.1 and PyTorch 2.4.1+cu121.
Report medians of five warm runs. Geometry input contains 100,000 real fibers and
7,773,080 points. GPU classification uses the ORG atlas with 2500 reference fibers
and 800 clusters. All calculations retain float64 precision.

| Operation | Previous | Current |
| --- | ---: | ---: |
| Downsample to 2000 fibers | 0.07435 s | 0.00605 s |
| Resample all fibers to 15 points | 0.10376 s | 0.04572 s |
| GPU classification, all 100,000 fibers | 1.05417 s | 1.01369 s |
| CPU matrix classification, 2048 fibers, four workers | 1.17591 s | 0.97005 s |

In the CPU comparison, original joblib classification with four workers takes
7.21368 s and GPU classification takes 0.02499 s. CPU matrix classification is
approximately 1.21x faster than the preceding optimized version and 7.44x faster
than original joblib for this sample. Geometry preparation improves GPU
classification by about 4%; the much larger sampling-only ratios do not represent
whole-pipeline acceleration.

The CPU registration objective with 2500 fixed and 1000 moving synthetic fibers,
10 points per fiber, takes 0.49338 s using the original serial algorithm,
0.36186 s with the current single worker, and 0.09723 s with four workers. These
objective values match exactly.

A separate alternating comparison isolates CUDA launch settings, with current
geometry preparation in both cases. Medians of seven runs give:

| CUDA operation | Previous launch | Current launch |
| --- | ---: | ---: |
| Mean distance, 2048 × 2500 fibers, 15 points | 5.849 ms | 5.214 ms |
| MeanSquared distance, same dimensions | 3.565 ms | 2.973 ms |
| Hausdorff distance, same dimensions | 4.720 ms | 3.461 ms |
| Full classification, all 100,000 real fibers | 1.01970 s | 0.97009 s |

Distance tests at 10/15/20 points improve approximately 1.1–1.6x. Full
classification gains about 5% beyond geometry preparation; labels, embedding,
and serialized output attributes match exactly. These launch choices were
measured on RTX 4090 and do not establish the optimum for other CUDA hardware.

Timing excludes file reading, atlas loading, writing, Slicer rendering, and first
imports/JIT compilation. Classification includes resampling and output attributes.
No full clinical CLI pipeline or native Windows/macOS performance was measured
for this batch. Shared-workstation load can affect timings.

## Validation and reproduction

CPU distance checks compare all four metrics, both bilateral modes, thresholds,
1–50 points, one/four workers, and mixed compiled/NumPy partial tiles against the
original per-fiber code. Classification labels and serialized attributes match.
Geometry checks compare float32/float64 inputs, exact length values, endpoint
threshold boundary decisions, resampled coordinates, and serialized filtered
geometry against the preceding implementation. Outputs remain detached from
input arrays. The complete real-subject GPU labels and embedding match exactly.
CUDA regression checks also cover short/long fibers, float32 fallback, strided
inputs, NaN/Inf, partial blocks, allocation retries, and selected-device execution.

The rebuilt universal installer bundles source revision `c0e81a7`. An isolated
Linux Slicer 5.2.2 installation verifies all five pages, CPU/GPU labeling,
registration, core output tests, GPU selection, allocator cleanup, loader behavior,
and workflow locking. The compiled/NumPy distance parity test also passes with
Slicer's SciPy 1.9.2. All 28 bundled and installed WMA Python files match the
committed source bytes. No optional PyTorch download was performed during tests.

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests
export WMA_CPU_THREADS=4 CUDA_VISIBLE_DEVICES=1
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1

taskset -c 0-3 python tests/bench_geometry.py "$INPUT_TRACTOGRAPHY" \
  --atlas "$LABELING_ATLAS_DIRECTORY" --baseline 710f7d4 --repeat 5 \
  --output /tmp/wma-geometry.json

taskset -c 0-3 python tests/bench_cpu_pipeline.py "$INPUT_TRACTOGRAPHY" \
  "$LABELING_ATLAS_DIRECTORY" --baseline 710f7d4 --fibers 2048 --repeats 5 \
  --output /tmp/wma-cpu-compiled.json

taskset -c 0-3 python tests/bench_distance_launch.py --input "$INPUT_TRACTOGRAPHY" \
  --atlas "$LABELING_ATLAS_DIRECTORY" --baseline 710f7d4 --repeat 7 \
  --output /tmp/wma-distance-launch.json

PYTHONPATH=.:tests:slicer/WMAGPU/Testing python -m unittest \
  test_cpu_tiling test_geometry_views test_output_pipeline \
  test_registration test_loading test_outlier_parallel -v

python -m unittest test_distance_kernel test_atlas_blocks -v
```
