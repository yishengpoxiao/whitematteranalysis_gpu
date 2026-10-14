# CPU streaming and fused GPU registration — 2026-10-14

## Changes

- CPU atlas labeling streams similarity blocks instead of retaining the full
  input-by-atlas matrix. Project onto the required embedding columns only and
  scale columns directly, avoiding both the discarded projections and a dense
  diagonal-matrix multiplication. ORG uses 11 of the 2500 stored eigenvectors.
  The 245,946-fiber case previously needed approximately 4.58 GiB for each full
  float64 similarity/projection matrix, before normalization copies.
- Target 8 MiB similarity blocks and small distance temporaries: 8 MiB per
  parallel worker or 1 MiB for serial work. Reuse two arithmetic buffers and keep
  CPU coordinates contiguous by component. Preserve thresholds, forward/reversed
  order, bilateral comparison, and the original StrictSimilarity sigma convention.
  Input geometry, resampled coordinates, and final embeddings still occupy host
  memory; this is not a constant-memory implementation of the entire pipeline.
- Affine and nonrigid CPU registration now use the tiled parallel objective.
  Reuse one worker pool throughout an optimizer invocation and close it on
  success or failure. Respect CPU backend selection and explicit labeling worker
  counts, including joblib's negative worker-count conventions.
- GPU registration fuses squared point distances, orientation minimum, and
  exponentiation for fibers with up to 32 sample points. One warp keeps the
  exponentiation stage occupied after point reduction. Retain PyTorch's fiber
  reduction and CPU entropy reduction, float64, and separate multiply/add
  rounding. Longer fibers and installations without Triton retain the existing
  PyTorch implementation. This also accelerates nonrigid objectives, which call
  the same probability calculation.
- The PyTorch CUDA fallback also reuses two point-distance buffers and evaluates
  minimum, negation, division, and exponentiation in place. It needs no Triton
  installation and preserves the original tensor values.
- Reuse the full subject already held by subject-to-atlas registration when
  writing transformed geometry. Avoid rereading the original tractography.
- CLI Slicer hardening loads/inverts the transform once for the entire directory,
  batches scene updates, and releases each temporary bundle, display, diffusion
  tensor display properties, and storage node after saving. Keep unrelated nodes.
- Explicit CPU similarity calls no longer fall through to global CUDA settings.
  CPU registration and support checks avoid importing PyTorch when GPU is disabled.

No dependency or configuration-file changes are required. The universal installer
bundles the updated WMA source and retains its date-free filename.

## Correctness checks

- Numerical tests pass, including real two-device CUDA selection,
  four distance methods, bilateral mode, partial/empty CPU labeling blocks,
  CPU worker counts, CPU optimizer cleanup, GPU OOM retry, optional Triton
  fallback, affine/nonrigid optimizer sequences, and B-spline sampling.
- CPU labeling preserves labels and serialized VTP bytes in bounded synthetic
  fixtures. Floating embedding/color values agree within 1e-12; exact float64
  equality is not claimed for the reduced-column BLAS projection.
- Parallel CPU objectives match the original objective exactly across several
  point counts, sigmas, and worker counts. The GPU probability fusion preserves
  the tested PyTorch values, objective sequences, and final transforms exactly.
- Five Slicer-core registration tests, four outlier parallel tests, and a fresh
  CPU-only startup test pass. CPU-only registration imports no PyTorch.
- Native Linux Slicer 5.2.2/SlicerDMRI hardening matches the previous output bytes
  for affine/nonrigid, forward/inverse transforms, point/cell/field attributes,
  and empty clusters. Each scenario uses 64 nonempty files plus one empty file.
  Transform loads decrease from 65 to one; added scene nodes decrease from 649
  to zero. Small affine hardening timings are essentially unchanged, so these
  checks establish bounded scene lifetime rather than a full-stage speedup.

## CPU measurements

Warm classification of 2048 deterministically sampled fibers against the actual
ORG atlas (2500 reference fibers, 11 embedding columns), with four workers and
one BLAS thread. Medians of three runs, excluding input/atlas loading and file
saving. CPU affinity is four physical cores. Compare with commit `60c7b2d`.

| Classification path | Median |
| --- | ---: |
| Previous CPU matrix, four workers | 8.431 s |
| Original joblib, four workers | 7.197 s |
| Current streamed CPU, four workers | 1.204 s |
| GPU classification reference | 0.035 s |

The current CPU path is approximately 7.0x faster than the previous matrix path
and 6.0x faster than joblib in this bounded case. Labels and serialized cell
attributes match across all four profiles. GPU classification itself is unchanged
by this registration optimization.

For a deterministic CPU registration objective with 2500 fixed fibers, 1000
moving fibers, and 10 points per fiber, medians are 0.479 s for the original
serial loop, 0.429 s for the current single-worker implementation, and 0.142 s
with four workers. Objectives match exactly; four workers give approximately
3.37x acceleration over the original loop.

## GPU registration measurements

Xeon Gold 6330, four pinned physical CPU cores, one RTX 4090, PyTorch 2.4.1+cu121,
Triton 3.0.0, float64. Compare with commit `60c7b2d` using deterministic synthetic
samples: 5000 fixed fibers, 2000 moving fibers, and 10 points per fiber.

| Operation | Previous | Current |
| --- | ---: | ---: |
| Warm objective, median of 30 calls | 6.811 ms | 4.640 ms |
| Temporary allocated GPU memory during evaluation | 381.94 MiB | 76.78 MiB |
| 30-evaluation COBYLA, median of three runs including setup | 0.192 s | 0.131 s |

The bounded objective and optimizer are approximately 1.47x faster, with about
80% less temporary tensor memory. Objective sequences and final transforms match
exactly. Initial JIT compilation is outside warm timings. GPU programs/context and
allocator reservations are not included in the temporary allocated-memory metric.

With Triton disabled in the same Linux CUDA environment, the PyTorch fallback's
warm objective is essentially unchanged (30.288 ms to 30.199 ms), while temporary
allocated memory decreases from 2366.04 MiB to 1603.10 MiB, approximately 32%.
The 30-evaluation optimizer changes from 0.914 s to 0.910 s, with identical
objectives and transforms. This establishes a memory improvement for the
no-Triton path, not a meaningful speedup or native Windows validation.

These stage measurements do not replace the earlier
[complete ORG CLI comparison](benchmark-org-2026-10-14.md). The full clinical
pipeline was not rerun for this change. File reading/writing, VTP compression,
VTK preprocessing, and Slicer launch remain outside the fused GPU computation.
The workstation is shared; medians reduce noise but do not isolate other users'
CPU, disk, or memory traffic. Native Windows/macOS validation remains outstanding.

## Reproduction

Use an existing WMA environment. Set the input and atlas variables locally;
private paths and generated geometry are excluded from repository records.

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
export WMA_CPU_THREADS=4
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=1

taskset -c 0-3 python tests/bench_cpu_pipeline.py \
  "$INPUT_TRACTOGRAPHY" "$LABELING_ATLAS_DIRECTORY" \
  --fibers 2048 --repeats 3 --output /tmp/wma-cpu-benchmark.json

taskset -c 0-3 python tests/bench_registration_kernel.py \
  --baseline 60c7b2d --fixed 5000 --moving 2000 --repeat 30 --maxfun 30

# Add --no-triton to measure the PyTorch CUDA fallback.

python -m unittest discover -s tests -p 'test_cpu*.py' -v
```

The native hardening check is `tests/verify_slicer_hardening.py`, run using
Slicer with SlicerDMRI, disabled user settings, and an isolated display.
