# Fused bilateral CUDA distances and packed CPU references — 2026-10-15

## Changes

- Compute direct/reversed and left-right reflected fiber comparisons in one
  optional Triton launch. Share coordinate loads and the squared Y/Z differences
  between the original and reflected comparisons. Preserve the original ordered
  additions, threshold/square-root operations, point reduction, and minima.
- Centralize bilateral handling in the GPU distance-block helper, removing
  repeated reflection/reduction code from matrix and streamed labeling callers.
  The ordinary PyTorch path still handles float32, long/strided fibers,
  StrictSimilarity, and absent Triton. Release forward/reverse temporaries before
  computing reflected fallback distances to retain the preceding memory behavior.
- Pack fixed CPU coordinates by point when a tile uses the compiled SciPy kernel.
  Each point-distance call reads contiguous three-coordinate reference rows.
  Retain a contiguous point axis in the NumPy fallback as well: changing reference
  layout without this step changes point-sum rounding in small partial tiles.
- Update the distance benchmark to load both historical distance modules and
  kernels, retaining reproducible baseline measurements after adding bilateral
  kernel support. No precision, dependency, or configuration changes.

## Measurements

Compare with commit `4db1b46` on RTX 4090, Xeon Gold 6330, four pinned physical
CPU cores, one BLAS thread, PyTorch 2.4.1+cu121, Triton 3.0.0 and SciPy 1.10.1.
GPU timings alternate baseline/current implementations, with medians of seven
warm runs. CPU classification uses medians of five warm runs.

| Operation | Previous | Current |
| --- | ---: | ---: |
| GPU classification, all 100,000 real fibers | 0.70104 s | 0.64298 s |
| CPU matrix classification, 2048 real fibers, four workers | 0.98617 s | 0.89812 s |
| CPU objective, 1000 fixed/250 moving, four workers | 0.01595 s | 0.01348 s |
| CPU objective, 2500 fixed/1000 moving, four workers | 0.09827 s | 0.09119 s |
| CPU objective, 5000 fixed/2000 moving, four workers | 0.39953 s | 0.35367 s |

GPU classification improves approximately 1.09x, reducing time by 8.3%. Labels,
embedding coordinates and serialized cell attributes match exactly. CPU
classification improves approximately 1.10x; labels and serialized attributes
match exactly and floating embedding comparisons pass at 1e-11 tolerance.
Original joblib with four workers takes 7.26699 s for the same CPU sample, while
the current GPU path takes 0.01649 s. The ORG atlas contains 2500 reference fibers,
800 centroids, and 10 embedding dimensions.

Objective comparisons reuse the same worker pool and alternate baseline/current
calls, taking medians of seven warm runs. Use `RandomState(42)` normal coordinates
scaled by 40, 10 points per fiber, and sigma squared 400. Objective values match
exactly; improvements vary with reference/sample sizes. Single-worker improvements
range from about 2% to 15%, so packing does not guarantee a large speedup for every
registration scale.

Classification includes resampling, similarities, normalization, projection,
labels and output attributes. File reading, atlas loading, registration, saving,
Slicer scene work and first imports/JIT compilation are excluded. Shared-workstation
load can affect timing. These measurements do not establish full-pipeline or
native Windows/macOS speedups.

## Validation and reproduction

CUDA tests compare all four fiber metrics, both bilateral modes, float64/float32,
1–50 points, zero/nonzero thresholds and clipped distances, NaN/Inf, strided
inputs, selected-device execution, and input preservation. Atlas tests compare
full/streamed outputs and allocation retries; outlier tests compare serial and
parallel CPU/GPU masks. CPU tests compare all metrics, bilateral modes, one/four
workers, and mixed compiled/NumPy partial tiles against original per-fiber values,
including large registration objectives and several sigma values.

The rebuilt universal installer bundles source revision `befdd51`. An isolated
Linux Slicer 5.2.2 installation verifies the five pages, CPU/CUDA labeling,
registration, core outputs, selected GPU execution, loader behavior, locking, and
allocator cleanup. All 12 installed CPU-tiling, CUDA-distance and parallel-outlier
tests pass, with Triton enabled and Slicer's SciPy 1.9.2. All 28 bundled/installed
WMA Python files match committed source bytes. No optional PyTorch download is
performed during tests.

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests
export WMA_CPU_THREADS=4 CUDA_VISIBLE_DEVICES=1
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1

taskset -c 0-3 python tests/bench_distance_launch.py --baseline 4db1b46 --bilateral \
  --input "$INPUT_TRACTOGRAPHY" --atlas "$LABELING_ATLAS_DIRECTORY" --repeat 7 \
  --output /tmp/wma-bilateral.json

taskset -c 0-3 python tests/bench_cpu_pipeline.py "$INPUT_TRACTOGRAPHY" \
  "$LABELING_ATLAS_DIRECTORY" --baseline 4db1b46 --fibers 2048 --repeats 5 \
  --output /tmp/wma-packed-cpu.json

PYTHONPATH=.:tests:slicer/WMAGPU/Testing CUDA_VISIBLE_DEVICES=0,1 python -m unittest \
  test_distance_kernel test_atlas_blocks test_outlier_parallel test_cpu_tiling test_cpu_atlas -v
```
