# Atlas reference caching for outlier removal

## Scope and behavior

CLI and Slicer CPU/GPU outlier removal now reuse the atlas's 15-point fiber
coordinates and leave-one-out reference probabilities across subjects and
process restarts. Subject coordinates, subject probabilities, masks, cutoff
application and output writing are recomputed every time. The original joblib
backend remains uncached.

A fresh entry uses the existing distance implementation and NumPy probability
calculation without changing precision, reduction order or rejection rules.
Only NumPy arrays are stored; the cache retains no CUDA tensors or MRML nodes.
CLI accelerated workers also reuse the already resampled atlas for subject
comparison and use one CPU distance thread per cluster process, avoiding nested
worker oversubscription.

Keys include the resolved reference VTP path, size, modification/change timestamps,
resampling length, sigma, metric, bilateral setting, backend, precision,
NumPy/SciPy/PyTorch/Triton versions as applicable, CPU/platform information,
CUDA device architecture and the relevant implementation sources. Changing the
standard-deviation rejection threshold can reuse probabilities because this
threshold is applied afterward.

Each entry is a NumPy archive loaded with `allow_pickle=False`. Writes use an
owned temporary file followed by atomic replacement. Concurrent writers are
safe. Missing, damaged or inaccessible cache files trigger normal computation;
cache storage failure does not fail outlier removal. Empty references retain
the previous behavior. Reference VTPs and `atlas.p` are never modified.

## Storage and controls

The default directory is:

- Linux/macOS: `${XDG_CACHE_HOME:-~/.cache}/whitematteranalysis/atlas-outliers`.
- Windows: `%LOCALAPPDATA%/whitematteranalysis/atlas-outliers`, with a home-cache
  fallback when `LOCALAPPDATA` is unset.

Set `WMA_ATLAS_CACHE=/path/to/cache` before starting the CLI or Slicer to override
the location, or `WMA_ATLAS_CACHE=off` to disable caching. No configuration file
is modified. Cache files can be removed while WMA is idle and will be rebuilt.
Different numerical/backend implementations use separate entries; old entries
are not automatically purged.

The tested ORG atlas has 800 reference clusters, 793,990 reference fibers and
877,227,996 within-cluster comparison pairs. One complete backend/precision
variant occupies 292,615,520 bytes (279.1 MiB). The first use still computes the
reference statistics and writes these files; acceleration from reuse applies to
subsequent subjects or reruns with matching keys.

## Reproduction

Use an existing initial-cluster directory so registration and labeling are
excluded from the measurement. Each benchmark creates a new output directory,
runs the preceding CLI revision, then performs one cold and repeated warm runs.
It checks every output cluster and the complete statistics log byte for byte.

```bash
export CUDA_VISIBLE_DEVICES=1
export WMA_CPU_THREADS=4
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export WMA_VTP_COMPRESSION=lz4
export PYTHONDONTWRITEBYTECODE=1

taskset -c 0-3 python tests/bench_outlier_cache.py \
  --input "$SUBJECT_CLUSTERS" --atlas "$ATLAS_DIRECTORY" \
  --backend gpu --jobs 4 --baseline d68372a --repeat 2 \
  --output "$NEW_GPU_BENCHMARK_DIRECTORY"
```

Use `--backend cpu` and a distinct output directory for the CPU matrix case.
The Linux affinity command is optional on other operating systems. These are
stage measurements on a shared workstation; they are not full-pipeline speedups
or Windows measurements.

## Measurements and correctness

The bounded stage test uses the same initial clusters as the preceding complete
ORG case: 245,946 input fibers and 130,737 retained fibers across all 800
categories. Both backend variants use their own preceding output, avoiding
CPU-versus-GPU floating-point comparisons. Measurements use four pinned physical
CPU cores, four cluster processes, single-threaded BLAS, float64, LZ4 output,
Python 3.8 / NumPy 1.20.3 / SciPy 1.10.1 / PyTorch 2.4.1 + CUDA 12.1 and one
RTX 4090 on Linux. Runs are sequential on a shared workstation.

| Outlier stage | Previous CLI | New, empty cache | New, cache hit |
| --- | ---: | ---: | ---: |
| GPU, preliminary repeat | 30.78 s | 29.46 s | 23.77 / 24.26 s |
| GPU, final source and reproducible benchmark | 27.07 s | 29.11 s | 23.65 / 23.89 s |
| CPU matrix, four processes | 148.58 s | 135.15 s | 44.27 s |

Every run checks the exact catalogue of 800 clusters, every encoded VTP byte and
the complete `cluster_log.txt`, including mean distances and probabilities.
All comparisons pass. The final GPU cache-hit median is 23.77 s, 12.2% below
its 27.07 s preceding CLI baseline; its cold run is 7.5% slower because it builds
the cache. The CPU cache-hit run is 70.2% below its baseline. These figures
describe this stage only. Cache-hit results
must not be presented as cold-start or complete-pipeline measurements.

Profiling the preceding GPU CLI across all occupied clusters attributes 55.12
aggregate worker-seconds to reference/subject reads, 18.28 to atlas distances and
probabilities, 3.24 to subject comparison/probabilities and 14.54 to masking,
writing and statistics. Worker sums overlap and are not stage wall times.
A proposed shared-context thread variant was abandoned after it progressed
substantially more slowly than the process baseline; Python/VTK reading did not
scale under those threads. Process parallelism is retained.

Nine targeted source tests pass: exact reference probabilities for supported
metrics, bilateral modes and GPU precisions; cold/warm equality; parameter,
reference timestamp and implementation invalidation; damaged/unavailable and
disabled caches; empty references; concurrent atomic writers; and the existing
Slicer parallel mask, stream, callback and failure contracts.

The universal installer was rebuilt from source commit `76a3f1b` and installed
through its actual dialog in a fresh Slicer 5.2.2 Linux profile. All 100 packaged
source files match the installer payload and installed files byte for byte.
The nine cache/outlier tests also pass against the installed extension's WMA
and core modules under Slicer's own NumPy/SciPy/PyTorch environment.
A separate fresh Slicer process discovers and opens all five installed module
pages, verifies CPU/GPU labeling and registration, and passes the four installed
core functional tests. Windows/macOS execution was not available for validation.
