# Acceleration and performance

## Execution model

WMA supports the original per-fiber joblib implementation, an accelerated CPU
matrix backend, and CUDA. Select the backend through the Slicer controls or the
CLI/environment settings in the [main guide](../README.md). CPU mode requires no
PyTorch. CUDA uses optional Triton kernels and retains a PyTorch fallback.

The CPU backend computes bounded distance tiles, packs coordinates for compiled
SciPy calculations, and reuses worker pools during registration. Atlas labeling
streams similarities and projects only the required embedding columns. Automatic
worker selection observes CPU affinity/quotas where joblib supports them, with
a default cap of 16; explicit worker settings take precedence.

CUDA keeps atlas normalization, embedding and nearest-centroid lookup on device.
Supported Triton paths fuse distance/probability calculations, bilateral
comparisons, Gaussian similarities, normalization and registration transforms.
Registration owns reusable probability buffers and reduces tile sizes after
out-of-memory retries. Laterality processes row tiles rather than retaining
whole-brain similarity matrices. Precision remains float64 by default.

## CPU BLAS threads

The current CPU dependency profile and Slicer installer select
`threadpoolctl>=3.1,<3.7`. CPU atlas labeling caps small BLAS work to avoid
competing with distance workers. For at most 32 projection columns, the automatic
ceiling is `min(4, max(1, available_cpus - distance_workers + 1))`.
Existing smaller BLAS limits are preserved; wider projections retain their
settings unless an explicit ceiling is requested.

Set `WMA_CPU_BLAS_THREADS=0` to disable the cap, or a positive integer to select
it. Missing or unsupported threadpoolctl versions leave BLAS untouched.
Concurrent WMA fits share their lowest requested ceiling; settings are restored
after the last enabled fit exits, including failures. These settings affect
process-wide BLAS users while active. Changing BLAS threads can change float64
roundoff; validation uses the existing 1e-12 embedding tolerance and checks
discrete outputs separately.

## Outliers, cache and file operations

Outlier workers process independent nonempty subject clusters. Empty categories
need no calculation, so the progress denominator may be smaller than the atlas's
800 categories. CPU workers use one distance thread each; GPU workers use CUDA
streams with VRAM-limited concurrency. More workers do not always improve GPU
throughput. Slicer's outlier stage currently uses one selected GPU.

Atlas reference coordinates/probabilities are cached across subjects and process
restarts. Keys include reference identity, numerical parameters, backend,
precision, libraries and implementation. The cache contains no subject fibers
or CUDA tensors. The default is `~/.cache/whitematteranalysis/atlas-outliers`, or
`%LOCALAPPDATA%/whitematteranalysis/atlas-outliers` on Windows; `XDG_CACHE_HOME` is
honored. `WMA_ATLAS_CACHE` overrides the directory; `off` disables caching.
The ORG cache uses approximately 279 MiB per numerical/backend variant. It can
be removed while WMA is idle; unavailable storage falls back to computation.

Geometry readers avoid unnecessary attribute loading and full tractogram copies.
Independent CLI file stages follow the pipeline worker budget. VTP writing is
lossless: library/Slicer exports default to compact zlib; the ORG shell pipeline
uses LZ4 intermediates and compact final anatomical exports.
`WMA_VTP_COMPRESSION=zlib` selects compact intermediates; `lz4` selects LZ4
explicitly for library writes. Decoded arrays are preserved although encoded
file bytes can differ between compression profiles.

## Representative measurements

These Linux measurements use a Xeon Gold 6330 and RTX 4090 on a shared
workstation. Snapshots, worker counts and timing scopes differ between rows;
do not combine them into a single current-release speed ratio. Private inputs,
raw logs and tractography outputs are excluded from Git.

| Workload | Conditions | Wall time |
| --- | --- | ---: |
| Complete ORG CPU matrix pipeline | Four physical cores; source `ba41935` | 327.97 s |
| Complete ORG CUDA pipeline | Four physical cores, one GPU; source `c9c622c` | 166.21 s |
| CPU classification, before / after BLAS control | 100,000 fibers, 16 workers; baseline `3891846`; one warm timing pair | 39.5922 / 18.6382 s |
| CPU classification, before / after BLAS control | 2,048 fibers, four workers; seven alternating repeats | 0.9884 / 0.7560 s |
| GPU centroid lookup, before / after tile tuning | 2,048 queries, 800 centroids, ten dimensions; baseline `cef12ca`; eleven repeats | 0.2669 / 0.2500 ms |
| GPU classification, before / after tile tuning | 100,000 fibers; eleven alternating repeats | 0.6456 / 0.6406 s |

The complete ORG input contains 328,228 fibers and 31,717,748 points (3.56 GB).
Runs include registration, labeling, outliers, hemisphere processing, Slicer
inverse-transform hardening and anatomical assembly, using
`-n 4 -r rig -x 1 -c 0`, ORG-800FC-100HCP and ORG-RegAtlas-100HCP. Here `rig`
selects rigid-affine-fast registration. BLAS/OpenMP use one thread. Each full
pipeline time is a single run including file I/O and startup; filesystem caches
and workstation load are uncontrolled. Interactive display and diffusion-measure
export are excluded.

Output checks cover 800 initial clusters (245,946 fibers), 800 outlier-removed
and 800 transformed clusters (130,737 fibers each), and 74 anatomical tracts
(93,766 fibers). The full CPU case has byte-identical VTP outputs and identical
saved transforms against its preceding CPU reference. The complete CPU/GPU
comparison checks topology, counts, metadata and array types exactly; floating
coordinates use `rtol=1e-6, atol=1e-5` and attributes `rtol=1e-6, atol=1e-7`.

Classification timings exclude file reading, atlas loading and import/JIT
warmup. The CPU BLAS cases preserve labels, geometry, colors and stored attributes
exactly, with embeddings within 1e-12 tolerance. GPU tile tuning preserves labels,
embeddings and serialized VTP bytes exactly. Its small whole-classification
difference is within timing variation and is not a robust end-to-end gain.

## Validation and reproduction

The computational payload has passed 31 native Slicer CPU tests, seven CUDA
centroid/atlas tests and five extension core tests, plus actual installer UI,
five-page initialization, CPU/GPU labeling and affine registration checks.
Tests cover numerical equivalence, cache invalidation, parallel outliers,
selected devices, empty inputs, failures and memory cleanup. Native application
validation is Linux Slicer 5.2.2 (NumPy 1.23.4, SciPy 1.9.2, VTK 9.1.0).
Windows/macOS and other Slicer releases require native checks.

Benchmark scripts remain under `tests/` and `slicer/WMAGPU/Testing/`. From the
repository root, supply your own input/atlas paths and fresh report paths:

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests

# CPU BLAS comparison; restrict this process to 16 CPUs for the quoted setup.
unset OPENBLAS_NUM_THREADS MKL_NUM_THREADS OMP_NUM_THREADS WMA_CPU_BLAS_THREADS
WMA_GPU=off CUDA_VISIBLE_DEVICES= taskset -c 12-27 \
  python tests/bench_cpu_blas.py "$TRACTOGRAPHY" "$ATLAS_DIRECTORY" \
  --baseline 3891846 --fibers 100000 --workers 16 --repeat 1 \
  --output "$CPU_BLAS_REPORT"

# GPU centroid and classification comparison.
WMA_GPU=on CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python tests/bench_gpu_centroid_tiles.py --baseline cef12ca --repeat 11 \
  --input "$TRACTOGRAPHY" --atlas "$ATLAS_DIRECTORY" --fibers 100000 \
  --output "$GPU_CENTROID_REPORT"
```

Linux `taskset` core indices must be adapted to the machine.
CPU/GPU benchmarks are separate measurements;
their reported gains do not accumulate across stages or configurations.
