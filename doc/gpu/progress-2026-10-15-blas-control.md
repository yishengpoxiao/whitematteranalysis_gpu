# Scoped CPU BLAS control and CUDA centroid tile tuning

## CPU implementation

CPU atlas labeling now uses optional `threadpoolctl` to bound small BLAS work.
For at most 32 projection columns, the automatic ceiling is
`min(4, max(1, available_cpus - distance_workers + 1))`, using joblib's effective
CPU count. Existing smaller BLAS limits are never increased. Wider projections
keep their original settings unless an explicit positive ceiling is requested.
`WMA_CPU_BLAS_THREADS=0` disables this optimization; a positive integer chooses
the ceiling. Without threadpoolctl the existing CPU path continues to work.
Distance point arithmetic, thread counts, tile sizes and precision are unchanged.

The limit spans the CPU atlas-labeling calculation. NumPy projection GEMMs and
SciPy's vector quantization both use BLAS: the
[SciPy 1.10.1 implementation](https://github.com/scipy/scipy/blob/v1.10.1/scipy/cluster/_vq.pyx)
uses GEMM when the feature count is at least five. Limiting only the projection
expressions leaves the centroid-assignment GEMM unrestricted.

The supported [threadpoolctl 3.1–3.6 API](https://github.com/joblib/threadpoolctl/blob/3.6.0/threadpoolctl.py)
uses process-scoped BLAS settings. Later versions are excluded because upstream
development changes MKL to thread-local setters; unsupported versions leave BLAS
untouched, and the Slicer installer selects the supported version range. Concurrent WMA CPU fits share their lowest
requested ceiling without serializing their distance work. A lock protects
entry/exit bookkeeping; the original BLAS settings are restored after the last
enabled fit exits, including failures. Descending limits are retained for the
active group and restored in reverse order. OpenMP libraries outside the BLAS
selection are not changed. As with other threadpoolctl contexts, concurrent
non-WMA BLAS users can observe the temporary process-level limit.

No NumPy or VTK replacement is required. The Slicer installer installs
`threadpoolctl>=3.1,<3.7` when missing, including an API feature probe for older
packages. Standalone users can install the new dated CPU/GPU requirement
profiles; the original requirement files and setup.py are preserved.

## Numerical contract

Changing native BLAS thread counts can change floating summation roundoff.
The distance matrices and original per-point order remain unchanged; embeddings
retain the existing 1e-12 float64 comparison tolerance. Labels, geometry, colors
and stored attributes are checked separately. Intermediate row sums can differ
by approximately 1e-15 relatively; full float64 bitwise equality is not claimed.
The benchmark permits at most one ULP in projected float32 attributes and requires
exact original/discrete attributes. No stored attribute differences occurred in
the real cases below. Disable the cap to retain the caller's BLAS setup.

## CPU measurements

Baseline: `3891846`. Xeon Gold 6330, 16 pinned physical cores, Python 3.8,
NumPy 1.20.3, SciPy 1.10.1 and threadpoolctl 3.5.0. BLAS thread environment
variables are unset: both original OpenBLAS libraries report 16 threads.
The same real DSI tractography and ORG atlas with 2,500 references, 800 centroids
and ten embedding dimensions are used. Reading, atlas loading and first imports
are outside timing; labeling includes resampling, distances, normalization,
projection, nearest centroids and output attribute formatting.

| Fibers | Distance workers | Previous / current | Comparison |
| --- | ---: | ---: | --- |
| 2,048 | 4 | 0.9884 / 0.7560 s | Seven alternating repeats; medians |
| 100,000 | 16 | 39.5922 / 18.6382 s | One timed run per backend after warmup |

The sample median falls by 23.5%. The full case falls by 52.9% (2.12x), but its
single timing pair is not a repeated end-to-end estimate. The full case uses a
one-thread BLAS ceiling, which the final automatic policy selects when all 16
available CPUs are assigned to distance workers. All labels, point coordinates,
line topology, colors and cell attribute values match exactly; embeddings pass
at 1e-12 tolerance. Original BLAS settings are verified after every call.
These are classification measurements, not complete shell-pipeline, Slicer scene
loading or cross-platform speed guarantees.

An initial four-worker scan measured 1.0316 s with default BLAS, 0.7344 s with
one BLAS thread, 0.7001 s with four, and 1.0419 s with sixteen. Exact labels
matched; intermediate float64 arrays showed the expected tiny BLAS roundoff.
Limiting each projection separately produced only 39.5480 / 38.4434 s in the
full case, so the final shared whole-fit scope is used instead.

```bash
python -m pip install -r requirements_cpu-2026-10-15-1.txt
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests
export WMA_GPU=off CUDA_VISIBLE_DEVICES=
unset OPENBLAS_NUM_THREADS MKL_NUM_THREADS OMP_NUM_THREADS

taskset -c 12-27 python tests/bench_cpu_blas.py "$TRACTOGRAPHY" "$ATLAS_DIRECTORY" \
  --baseline 3891846 --fibers 2048 --workers 4 --repeat 7 \
  --output "$NEW_BLAS_SAMPLE_REPORT"
taskset -c 12-27 python tests/bench_cpu_blas.py "$TRACTOGRAPHY" "$ATLAS_DIRECTORY" \
  --baseline 3891846 --fibers 100000 --workers 16 --repeat 1 \
  --output "$NEW_BLAS_FULL_REPORT"
```

## CUDA nearest-centroid audit

Two Triton prototypes attempted to avoid the full query-by-centroid distance
matrix, retaining the existing float64 coordinate reduction and exact observed
values/indices. Nine interleaved repeats on RTX 4090 gave:

| Shape: queries / centroids / dimensions | Existing | Two-stage, one warp | Single-stage, four warps |
| --- | ---: | ---: | ---: |
| 2,048 / 800 / 10 | 0.320–0.342 ms | 0.347 ms | 0.365 ms |
| 8,192 / 800 / 10 | 0.843 ms | 0.946 ms | 1.101 ms |
| 2,048 / 800 / 20 | 0.423–0.462 ms | 0.517 ms | 6.173 ms |

Tensor allocations decreased substantially, but common 800-centroid workloads
became slower. These prototypes are rejected. The tensor-allocation figures do
not account for CUDA local spill memory.

## Accepted CUDA tile tuning

The existing direct distance kernel remains in use. For at most 16 embedding
dimensions its block size decreases from 128 to 64 pairs; wider embeddings keep
32. Arithmetic, float64 precision, dimensional reduction order, square root and
the subsequent minimum operation are unchanged. This avoids the extra reduction
and launch overhead of the rejected prototypes.

Against `cef12ca`, eleven alternating repeats with the committed benchmark on
an otherwise idle RTX 4090 give:

| Queries / centroids / dimensions | Previous / current nearest-centroid time |
| --- | ---: |
| 2,048 / 800 / 10 | 0.2669 / 0.2500 ms |
| 8,192 / 800 / 10 | 0.8179 / 0.7943 ms |
| 2,048 / 800 / 20 | 0.3973 / 0.3969 ms; unchanged block policy |
| 257 / 37 / 10 | 0.0974 / 0.0984 ms; essentially flat |

The common ten-dimensional centroid step improves by 6.3% and 2.9% respectively.
A separate 31-repeat dimension scan covers 1–32 dimensions and confirms exact
values/indices; unchanged wide-dimensional cases also show timing variability.
For the real 100,000-fiber classification, the eleven-repeat median changes from
0.6456 to 0.6406 s (0.8%). An earlier independent eleven-repeat run gives
0.6471 / 0.6410 s. This small whole-classification difference is within the
scale of timing variation and is not a robust end-to-end speedup claim.
Labels, embeddings, colors, stored attributes and serialized VTP bytes match
exactly. Peak extra tensor allocation remains 85,669,376 bytes. File reading,
atlas loading, import/JIT warmup and serialization are outside timing.

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests
export WMA_GPU=on CUDA_VISIBLE_DEVICES=0
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
python tests/bench_gpu_centroid_tiles.py --baseline cef12ca --repeat 11 \
  --input "$TRACTOGRAPHY" --atlas "$ATLAS_DIRECTORY" --fibers 100000 \
  --output "$NEW_CENTROID_TILE_REPORT"
```

## Validation

Source checks cover scoped limits, explicit disable/override, preserving smaller
caller limits, wide projections, absent optional dependencies, failures during
projection, concurrent callers, nested descending ceilings and final restoration.
Five CPU atlas tests, five installer tests and five extension core tests pass.
Installer probes verify the ThreadpoolController API and continue to pin Slicer's
NumPy/VTK versions. Original numerical/thread/tile tests also pass under their
existing one-thread BLAS environment, where the new limit does not change BLAS
arithmetic. Seven CUDA centroid/atlas tests also pass, covering dimensions,
extreme magnitudes, ties, NaNs/infinities, strides, fallbacks, empty queries,
selected devices, tensor release and streamed atlas output.

## Native Slicer checks

The universal installer built from `cef12ca` passes the real installer UI test,
including a failed launch, retry, dependency verification and optional CUDA probe.
Slicer Python supplies NumPy 1.23.4, SciPy 1.9.2, joblib 1.4.2, VTK 9.1.0 and
the automatically selected threadpoolctl 3.6.0. The installed CPU backend passes
31 tests without skips: tiling, atlas labeling/BLAS scopes, registration,
laterality, atlas probability caching and parallel outlier removal.
All five installed workflow pages initialize, CPU/GPU labeling and affine
registration succeed, and five installed core functional tests pass. The final
installer containing CUDA tile tuning is verified separately below.

The final universal installer embeds source `ac55f9f` and passes the real
installer UI again in a fresh extension directory. Its installed code passes
all seven centroid/streamed-atlas tests without skips in Slicer Python on two
visible GPUs. All five workflow pages, CPU/GPU classification, affine registration
and five core tests pass in another fresh Slicer process. The 31 CPU tests above
exercise identical CPU sources and dependencies. All 101 packaged source/license
files match the repository, embedded payload and final installed directory.
The tested archive replaces `slicer/WMAGPU/dist/WMAGPU-Installer.zip`; its name
remains independent of date and Slicer version. Windows/macOS native execution
and complete shell-pipeline timings are outside this validation.
