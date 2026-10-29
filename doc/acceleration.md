# Acceleration overview

WMA accelerates the expensive tractography comparisons in registration, atlas
labeling and outlier removal. The original per-fiber joblib backend remains
available for comparison. See the [README](../README.md#full-pipeline-performance)
for measured end-to-end times and output validation, and the
[command-line reference](command-line-reference.md#shared-acceleration-options)
for backend selection.

## Registration

The optimizer, sampling schedule and registration presets stay the same. The
repeated objective evaluation is faster because:

- **CPU matrix:** NumPy processes blocks of fiber pairs rather than scheduling
  one Python/joblib task per fiber. Contiguous coordinate layouts improve cache
  access, and a reusable worker pool avoids repeated worker startup.
- **GPU:** fixed/moving samples and scratch buffers stay on the selected CUDA
  device across objective evaluations. Batched affine transforms and optional
  Triton kernels compute forward/reversed fiber distances and probabilities
  with fewer intermediate tensors and kernel launches. PyTorch provides the
  fallback when Triton is unavailable.
- **Nonrigid workflows:** B-spline evaluation has batch CPU and CUDA paths,
  avoiding a Python call to VTK for every point. These paths support nonrigid
  registration; the README's `rig` benchmark measures rigid plus affine only.

Source: [CPU backend](../whitematteranalysis/gpu/cpu_backend.py),
[GPU registration](../whitematteranalysis/gpu/registration.py),
[registration kernels](../whitematteranalysis/gpu/registration_kernel.py),
[B-splines](../whitematteranalysis/gpu/bspline.py).

## Atlas labeling and atlas construction

- **CPU matrix:** distance/similarity blocks feed the Nyström projection directly.
  Only the required eigenvector columns and compact per-fiber results are kept;
  the full subject-by-atlas similarity matrix is not retained. Scoped BLAS
  thread limits avoid competing thread pools on small embedding matrices.
- **GPU:** streamed similarity blocks remain on CUDA through normalization,
  embedding and nearest-centroid assignment. Optional fused distance,
  normalization and centroid kernels reduce temporary allocations; only compact
  labels, embedding coordinates and statistics return to the CPU.
- **Shared geometry/I/O:** bulk VTK connectivity/coordinate access replaces
  per-point Python calls. Cluster splitting preserves fiber order and data
  arrays while avoiding repeated scans of the full tractogram. Cluster counts
  and color means reuse stable label groups instead of rescanning all labels for
  each cluster. On Linux, parallel VTK writers use fork workers with inherited
  read-only polydata; Windows and macOS use one writer per thread. Lossless LZ4
  intermediate files reduce output overhead.
- Atlas construction reuses the accelerated distance/similarity operations.
  Sampling and clustering settings are kept consistent across backends.

Source: [CPU labeling](../whitematteranalysis/gpu/cpu_backend.py),
[GPU distances](../whitematteranalysis/gpu/distance.py),
[GPU embedding](../whitematteranalysis/gpu/linalg.py),
[cluster output](../whitematteranalysis/cluster.py),
[VTK I/O](../whitematteranalysis/io.py).

## Outlier removal

- Independent clusters run concurrently, with the worker budget shared across
  cluster tasks instead of nesting large distance worker pools.
- CPU/GPU blocks replace per-fiber distance loops while keeping the same
  probability calculation and rejection rule. Empty subject clusters skip
  unnecessary reference comparisons.
- GPU outlier scoring reduces similarity sums by query row, avoiding a full
  subject-by-atlas distance matrix. In float64 mode, scores near the rejection
  cutoff are recomputed with the CPU metric and NumPy reductions to preserve
  cutoff-sensitive decisions. Explicit float32 mode remains approximate and
  skips this refinement.
- Atlas coordinates and leave-one-out reference probabilities can be cached
  across subjects. Cache identity includes atlas file metadata, numerical
  settings, backend/precision, library versions and implementation identity.
  Subject-dependent comparisons are always recomputed.
- The generic `filter.remove_outliers` API also uses CPU/GPU distance matrices.
  Partial selection finds the three nearest neighbors without sorting every
  distance, while preserving the original summation order and rejection cutoff.
  For two or three fibers it uses the available neighbors; a singleton is rejected.

The atlas-based CLI requires identical subject/atlas cluster filenames before
writing outputs. Atlas loading requires a Boolean `bilateral` field; missing or
string-valued settings are rejected instead of silently changing reflection.

The README comparison starts with empty accelerated reference caches. Later
subjects may benefit from a warm cache; that benefit is not added to the reported
cold-cache speedup.

Source: [outlier command](../bin/wm_cluster_remove_outliers.py),
[reference cache](../whitematteranalysis/gpu/atlas_cache.py).

## The four final stages

Hemisphere assessment, inverse transformation with Slicer, hemisphere separation
and anatomical tract assembly run on the **CPU/Slicer in all three backend
profiles**. Changing `-g` does not select another implementation for these stages.
Their improvements come from vectorized geometry/label access, independent
file processing, lossless intermediate compression and batching many files into
Slicer workers. They do not contain the large distance matrices that benefit
from CUDA.

The ORG shell pipeline now splits hemispheres inside each inverse-transform
worker while the hardened polydata is still in memory. It retains the complete
transformed files and avoids reading them again for separation. The following
separation command verifies the three output catalogues and repairs incomplete
sets. Its timing is therefore verification time; the split work is included in
the inverse-transform stage.

Small differences between the three full-run columns do not establish a backend
regression: they also include process startup, filesystem cache state and shared
workstation load. Replacing one column with another run's faster times would not
be a measured pipeline result. Execution strategies must be compared on identical
inputs, and any updated total must come from a complete timed run.

## Numerical behavior

Float64 is the default. Fiber orientation handling, bilateral comparisons,
thresholds and cluster rejection rules are preserved. GPU centroid assignment
uses direct coordinate differences and selects the first centroid on a tie.
The README records the actual comparison against joblib, including small GPU
coordinate/transform differences; it does not claim universal byte identity.
`-gpu_float32` is an explicit speed/accuracy tradeoff and can change near-tie
labels. Filesystem and rendering work remain CPU operations.

The correctness audit uses seed `20261028` to select 2,048 fibers from the saved
C028 joblib registered tractogram. With the trained ORG atlas parameters, both
accelerated label arrays match joblib exactly. Maximum embedding differences
are `3.11e-15` (CPU) and `1.78e-14` (GPU). Estimated row sums differ by at most
`2.98e-8` and `5.54e-8`, respectively; these are large intermediate sums, with
errors relative to their maximum magnitude below `5.12e-14`. All checked
continuous outputs pass `rtol=1e-10, atol=1e-10`.

Independent comparisons against the upstream `similarity.py` use 128 query by
64 reference fibers, 15 points per fiber, all four supported distance methods,
both bilateral settings and thresholds 0 and 2. CPU distances are exact; GPU
maximum absolute error is `7.28e-12`. An affine objective with 128 moving and
64 fixed fibers matches the upstream value exactly on both backends. These
sampled checks supplement the earlier full-output comparison; they do not
establish accuracy for every input or for optional float32 mode.

Regression tests additionally exercise cutoff-sensitive masks, empty inputs,
one-to-three-fiber neighborhoods, atlas validation, explicit GPU failure, fused
and eager kernels, B-splines, file metadata and Slicer workflows. Float64 retains
unit coordinate differences even around `1e8`; float32 can erase those differences
and remains opt-in. Explicit `-gpu on` now requires usable CUDA rather than
silently falling back to CPU.
