# GPU device selection and tractography loading — 2026-10-14

## Changes

- Add a CUDA device selector to all four Slicer computation pages. GPU discovery
  runs in a separate PythonSlicer process after an explicit click. Page setup
  starts no CUDA probe. Selected IDs follow CUDA-visible device numbering and
  apply to the entire workflow; Slicer's rendering device is independent.
- Release unused PyTorch allocations on the selected device after success or
  failure, enabled by default. Pipeline also releases them before CPU anatomy,
  saving, and display. A manual action reports released bytes, allocated bytes,
  and reserved bytes. It preserves live tensors and does not reset CUDA contexts.
  A shared workflow lock prevents event processing from starting overlapping WMA
  computations with conflicting backend/device settings.
- Read pairwise and Pipeline VTK/VTP inputs on a background thread, report VTK
  reader progress, and create MRML nodes on the GUI thread. Inputs are hidden
  unless the user requests display. Keep every fiber and data array.
- Share geometry during input preparation when no parent transform exists.
  Actual parent transforms still produce transformed scene-coordinate geometry.
- Cache one labeling atlas in host memory, keyed by directory, file size, and
  modification time of both atlas files. Replacing the atlas replaces the entry.
- Split clusters using one connectivity/geometry conversion and one stable label
  grouping. Select final anatomical tracts directly, preserving the previous
  nearest-point hemisphere classification, float32 casts, member/fiber order,
  duplicate-member behavior, provenance, and metadata. Skip intermediate cluster
  construction when neither saving nor individual-cluster display requires it.
- Write anatomical VTP files through the existing parallel writer. Default to
  Model line display for Pipeline results and pairwise output. Pipeline can still
  select SlicerDMRI FiberBundle display; geometry is not reduced in either mode.
- Add explicit `-g auto|on|off` forwarding to the complete ORG shell pipeline.
  This prevents child argparse defaults from overriding CPU-only environment
  settings. Respect `-n` for outlier workers instead of hardcoding 80 workers.
  Normalize numeric `WMA_GPU=0/1` values to `off/on` as accepted by the backend.
- Correct CPU distance tiling: the previous implementation calculated a tile
  size but submitted entire thread partitions. Workers now fill disjoint output
  slices using bounded tiles, including with one worker. Permit one-row tiles
  when eight rows would exceed the temporary-memory target. Scalar arithmetic
  and result order are unchanged.
- Clarify outlier progress: report atlas-defined category count, available
  reference-file count, nonempty subject count, and empty subject count separately.
  Count completed nonempty clusters in progress, and warn when a missing/empty
  reference prevents an outlier check. Preserve the previous behavior of keeping
  those fibers. Tests distinguish six defined categories, five occupied subject
  clusters, four reference files, and missing versus empty references.

## Bounded measurements

Linux, shared workstation, Slicer 5.2.2 and VTK 9.1. These measurements isolate
removed overhead; they do not predict a full clinical pipeline's acceleration.

`slicer/WMAGPU/Testing/benchmark_loading.py` uses 40,000 deterministic fibers,
20 points per fiber, 800 clusters, and 72 nonempty synthetic anatomical tracts.
The table reports medians of three operations after importing computation modules.
Serialized VTP output matches byte for byte between paths, including metadata.

| Operation | Previous | Current |
| --- | ---: | ---: |
| Split 800 clusters | 1.919 s | 0.252 s |
| Anatomical selection from labeled geometry, without cluster saving | 2.868 s | 0.101 s |

The previous anatomical path includes cluster splitting, hemisphere separation,
and appending. When cluster files are also saved, the current path additionally
performs the 0.252 s split; disk writes are excluded from these timings. Timings
vary under shared memory/CPU contention. No FiberBundle-versus-Model render timing
is claimed, and native Windows/macOS validation remains outstanding.

Fresh initialization of all five Slicer pages takes 0.024 s in the startup check;
no new torch, scipy, or matplotlib imports occur. This excludes Slicer launch,
data reading, and rendering.

A real 3,555,014,103-byte legacy VTK input contains 328,228 fibers and 31,717,748 points.
The hidden Model loader retains all of them. A single Slicer read took 31.00 s;
this is a current-path measurement, not a before/after file-reading speedup.
Identity input preparation decreased from 2.738 s to 0.000302 s by sharing geometry.
The large-file read exposed a limitation: VTK 9.1 holds the Python interpreter
lock for long intervals, so Python GUI callbacks can still stall despite the
worker thread. Yielding in progress callbacks did not resolve that limitation
and was not retained. VTK 9.4 introduced wrapper hints to release the lock for
pipeline updates; see [VTK concurrency notes](https://docs.vtk.org/en/v9.6-final/release_details/9.4/python-hint-unblock.html).
No claim of fully responsive large-file reading is made for older Slicer builds.

The [complete ORG CLI comparison](benchmark-org-2026-10-14.md) records successful
full-input joblib, CPU-matrix, and GPU runs, stage timings, memory, and independent
output validation. It is separate from the bounded Slicer display/loading checks.

A separate native Slicer check used a 100,000-fiber input with complete ORG
reference files (800/800). With the default 40 mm length filter, labeling without
registration retained 82,824 fibers in 658 nonempty categories. Default affine
registration followed by labeling retained 87,407 fibers in 699 nonempty
categories, leaving 101 empty categories. Affine scaling changes the lengths
tested by the filter. These counts describe this check, not a reconstruction of
the user's earlier settings. Empty fine-grained categories do not directly imply
that an anatomical tract, which aggregates several categories, is absent.

The real labeling check also distinguishes residual library memory from a WMA
tensor leak. After allocator cleanup, PyTorch 2.4 retained 8.125 MiB allocated and
40 MiB reserved. In a separate isolated diagnostic process, clearing cuBLAS
workspaces reduced both to zero, confirming workspace retention. Production
cleanup preserves PyTorch library state and uses allocator cleanup; the diagnostic
private API is not called by the extension. See the version-specific
[cuBLAS workspace documentation](https://docs.pytorch.org/docs/2.4/notes/cuda.html#cublas-workspaces).

## Validation

- Four cluster-output tests validate original metadata types, array order,
  negative labels, and serialized output.
- Two CPU tiling tests compare all four supported distance methods, bilateral
  mode, and one/four workers against the original per-fiber algorithm. Every
  temporary block obeys the requested two-row test bound; results match exactly.
- Three loading tests validate anatomical byte parity, atlas invalidation and
  bounded retention, and real CUDA cleanup while preserving live tensors.
- Five registration tests cover presets, schedules, CPU/CUDA transforms, group
  output, and cached preprocessing.
- Native Slicer tests validate asynchronous VTK/VTP reading, GUI heartbeats,
  hidden full-size inputs, shared identity geometry, actual selected-device
  registration, failed-run cleanup, and overlapping-run rejection. With
  `CUDA_VISIBLE_DEVICES=0,4`, selecting logical GPU 1 routes computation to physical
  GPU 4. Successful and failed bounded runs both finish with zero live/reserved
  PyTorch tensor memory on that device.
- Pipeline validation covers inverse-transform/anatomical ordering, file output,
  all display modes, unrelated-node preservation, 800-node batching, and four
  sequential/parallel outlier tests. Registration/merge checks also pass.

PyTorch's caching allocator can account for idle VRAM after a computation.
`empty_cache()` releases unused blocks, not live tensors. CUDA context memory and
Slicer graphics resources can remain visible even when PyTorch reports zero
allocated/reserved memory. Sources: [CUDA memory management](https://docs.pytorch.org/docs/main/notes/cuda.html#memory-management),
[empty_cache](https://docs.pytorch.org/docs/main/generated/torch.cuda.memory.empty_cache.html),
[allocated memory accounting](https://docs.pytorch.org/docs/main/generated/torch.cuda.memory.memory_allocated.html).
