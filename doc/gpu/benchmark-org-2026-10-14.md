# Complete ORG pipeline benchmark — 2026-10-14

## Scope and setup

The requested local tractography file was processed through
`bin/wm_apply_ORG_atlas_to_subject.sh`, including registration, labeling, outlier
removal, hemisphere assessment, inverse transformation with SlicerDMRI,
hemisphere separation, and anatomical tract appending. The input contains
328,228 fibers and 31,717,748 points (3,555,014,103 bytes). No reduced input or
precomputed registration substitutes were used in the successful profiles.
Private input paths, logs, and tractography outputs remain outside Git.

- Hardware: Intel Xeon Gold 6330; four physical CPU cores pinned as 0–3;
  NVIDIA RTX 4090, 24 GB, physical device 1 for GPU computation.
- Environment: Python 3.8, PyTorch 2.4.1+cu121, VTK 9.1, Linux Slicer 5.2.2
  with SlicerDMRI; driver 535.230.02.
- Atlas: ORG-800FC-100HCP with ORG-RegAtlas-100HCP as the registration target.
- Common shell options: `-n 4 -r rig -x 1 -c 0`. Here `rig` selects the
  `rigid_affine_fast` preset, including rigid and affine scales; it is not a
  rigid-only benchmark. Optional diffusion-measure export was not requested.
  Intermediate files were retained in separate output directories.
  Registration uses standard preset sampling with seed 1000; labeling processes
  the full input after the default 40 mm minimum-length filter.
- CPU budget: `WMA_CPU_THREADS=4`; OpenBLAS, MKL, OpenMP, and NumExpr each
  limited to one thread. The original registration objective is serial;
  `-n 4` controls the stages that support joblib/worker parallelism.
- GPU: `CUDA_VISIBLE_DEVICES=1`, `WMA_GPU_DEVICES=0`, default float64 precision.
  The three modes use the same current output/I/O implementation. “Original
  joblib” refers to the original mathematical backend, not a separate untouched
  upstream checkout.

| Mode | Shell GPU option | WMA_CPU_FAST |
| --- | --- | --- |
| Original joblib, four workers | `-g off` | `0` |
| CPU matrix, four-worker budget | `-g off` | `1` |
| GPU, one RTX 4090 | `-g on` | `1` |

The GPU profile completed first. An initial CPU-matrix attempt exposed unbounded
distance temporaries: a calculated tile size was ignored when work was submitted
to threads. That attempt was stopped at about 92 GiB maximum resident memory and
excluded from successful-run timings. The corrected CPU profile restarts the
entire pipeline. Its frozen source differs only in CPU distance tiling; GPU and
original-joblib execution paths are unchanged. Numerical tests match the original
per-fiber distance algorithm exactly for all four supported metrics, bilateral
mode, and one/four workers.

The benchmark directory retains the original source snapshot, its Git base
revision and patch, a second snapshot with the CPU-only repair, per-command
start/end records, full logs, and `/usr/bin/time -v` resource measurements.

## Measurement limits

These are sequential, single full-case measurements on a shared workstation,
including file reading/writing and Slicer startup. Disk and page-cache conditions
and other workloads are not controlled. During the CPU retry, `/data` storage
reached 99.6% utilization
with an approximately 43-request queue; initial reading took substantially longer
than in the GPU run. Total-time ratios therefore combine backend differences with
changing I/O contention. They are not isolated kernel speedups or a clinical
throughput guarantee. Slicer interactive result display is outside this CLI test.

Output validation checks 800 initial, 800 outlier-removed, and 800 transformed
cluster files plus the complete 74-file anatomical catalogue: 33 bilateral tracts
and eight commissural tracts. The initial harness incorrectly expected 73;
independent catalogue validation corrects that harness error without rerunning a
successful GPU pipeline. Command exit status and full output filenames are both
checked; a count alone is insufficient.

## Results

| Mode | Total wall time (hh:mm:ss) | Maximum host RSS (GiB) |
| --- | ---: | ---: |
| Original joblib, four workers | 00:31:24 | 25.06 |
| CPU matrix, four-worker budget | 01:06:21 | 20.88 |
| GPU, one RTX 4090 | 00:14:39 | 14.58 |

In this run, GPU was 2.14x faster end to end than joblib and 4.53x faster than
CPU matrix. CPU matrix took 2.11 times as long as joblib. These are observed
whole-pipeline ratios under the shared-workstation conditions described above.
Maximum RSS is the GNU time host-memory metric, not GPU VRAM.

| Stage (seconds) | Joblib 4 | CPU matrix | GPU |
| --- | ---: | ---: | ---: |
| Registration | 397.5 | 1247.8 | 289.1 |
| Atlas labeling and cluster saving | 915.2 | 1910.0 | 88.0 |
| Outlier removal | 212.5 | 438.9 | 65.9 |
| Hemisphere assessment | 87.1 | 87.2 | 95.8 |
| Inverse transform with Slicer | 122.9 | 123.1 | 184.4 |
| Hemisphere separation | 85.1 | 85.2 | 86.1 |
| Anatomical tract appending | 61.4 | 62.9 | 65.1 |

The Slicer inverse-transform row includes its nested hardening script; that nested
measurement is not counted twice. Total time also includes shell/interpreter
startup and directory checks outside the stage wrappers.

The CPU matrix mode did not accelerate this case. Both CPU branches retain the
full similarity matrix and project onto all 2500 stored atlas eigenvectors before
selecting the required embedding columns; the fused GPU path projects only onto
the 11 required columns. The matrix implementation, CPU/cache contention, and
large-file I/O all contribute to these full-run timings. This comparison does not
isolate a CPU distance-kernel speedup from those other costs.

## Output validation

All three complete profiles returned zero and produced exactly the expected
800 numbered files in each of the initial, outlier-removed, and transformed
cluster directories, and all 74 named anatomical files. VTP headers are readable;
no files are zero bytes. Counts match across modes:

| Fiber count | All three modes |
| --- | ---: |
| Initial labeled clusters after preprocessing | 245,946 |
| After outlier removal | 130,737 |
| After inverse transformation | 130,737 |
| Named anatomical outputs | 93,766 |

Inverse transformation preserves every retained fiber. The maximum absolute
difference between the final affine parameters of either CPU mode and GPU is
7.55e-14. These checks establish successful output generation and count/transform
agreement; no byte-identical full-case output claim is made.

The `Mean of empty slice` warnings in outlier QC occur when a cluster has no
retained fibers; its post-removal mean is undefined. Empty cluster files remain
part of the 800-file catalogue. All output and stage-exit checks above passed.

## Reproduction

Use a WMA Python environment, put its Python and the repository's `bin` directory
on `PATH`, and ensure SlicerDMRI is available to the Slicer executable. Set
`INPUT_TRACTOGRAPHY`, `ATLAS_ROOT`, `SLICER_WITH_DMRI`, and `RUN_ROOT` locally;
private paths are intentionally omitted here. `SLICER_WITH_DMRI` may be a wrapper
that adds the installed SlicerDMRI module path and disables user settings.
The recorded run used NumPy 1.20.3, SciPy 1.10.1, joblib 1.4.2, and Triton 3.0.0
in addition to the versions above. Backend source changes are recorded in commit
`5a454b7`; later GUI, logging, packaging, and numeric shell-mode aliases do not
change the executed CLI paths for the explicit `on`/`off` flags used here.

```bash
export WMA_CPU_THREADS=4
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=1 WMA_GPU_DEVICES=0

# Original per-fiber joblib backend, four workers.
WMA_CPU_FAST=0 /usr/bin/time -v taskset -c 0,1,2,3 bash bin/wm_apply_ORG_atlas_to_subject.sh \
  -i "$INPUT_TRACTOGRAPHY" -o "$RUN_ROOT/joblib_4" -a "$ATLAS_ROOT" \
  -s "$SLICER_WITH_DMRI" -n 4 -g off -r rig -x 1 -c 0

# CPU matrix backend with the repaired tile bounds.
WMA_CPU_FAST=1 /usr/bin/time -v taskset -c 0,1,2,3 bash bin/wm_apply_ORG_atlas_to_subject.sh \
  -i "$INPUT_TRACTOGRAPHY" -o "$RUN_ROOT/cpu_matrix_4" -a "$ATLAS_ROOT" \
  -s "$SLICER_WITH_DMRI" -n 4 -g off -r rig -x 1 -c 0

# One visible RTX 4090.
WMA_CPU_FAST=1 /usr/bin/time -v taskset -c 0,1,2,3 bash bin/wm_apply_ORG_atlas_to_subject.sh \
  -i "$INPUT_TRACTOGRAPHY" -o "$RUN_ROOT/gpu" -a "$ATLAS_ROOT" \
  -s "$SLICER_WITH_DMRI" -n 4 -g on -r rig -x 1 -c 0
```

Use new empty output directories: the shell pipeline skips stages with existing
outputs. The commands list modes for comparison; the actual recorded order was
GPU, CPU matrix (interrupted attempt, then complete retry), and original joblib.
