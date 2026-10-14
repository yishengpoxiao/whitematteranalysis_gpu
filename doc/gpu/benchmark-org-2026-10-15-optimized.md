# Complete ORG GPU pipeline after compute/geometry optimization — 2026-10-15

## Scope

Run the same complete clinical case and shell workflow as the
[2026-10-14 benchmark](benchmark-org-2026-10-14.md), using frozen source `befdd51`:
`bin/wm_apply_ORG_atlas_to_subject.sh -n 4 -r rig -x 1 -c 0 -g on`.
The input contains 328,228 fibers and 31,717,748 points, with 3,555,014,103 input
bytes. Its size and modification timestamp remain unchanged after the run.
Registration uses the `rigid_affine_fast` preset, including rigid and affine
scales. Labeling uses the full subject after the standard 40 mm length filter.
No reduced case or precomputed registration replaces any pipeline stage.

Use the same Xeon Gold 6330, four physical cores pinned as 0–3, physical RTX 4090
device 1, default float64, Python 3.8, PyTorch 2.4.1+cu121, VTK 9.1, and Linux
Slicer 5.2.2 with SlicerDMRI. OpenBLAS/MKL/OpenMP/NumExpr use one thread each;
`WMA_CPU_THREADS=4`, `CUDA_VISIBLE_DEVICES=1`, `WMA_GPU_DEVICES=0`. The ORG atlas
and registration target remain unchanged. Intermediate outputs are retained in
a separate directory, with isolated Slicer settings and per-command timing.
Private filenames, logs, geometry, and the frozen source snapshot remain outside
Git.

## Observed results

| Metric | Earlier complete GPU case | Compute/geometry optimized case |
| --- | ---: | ---: |
| Total wall time | 879.43 s (14:39) | 458.22 s (7:38) |
| Maximum host RSS | 14.58 GiB | 10.48 GiB |

The observed complete run is approximately 1.92x faster. This is a single
historical/current comparison on a shared workstation, including actual file
I/O, subprocess imports and Slicer startup. Page-cache, CPU frequency, storage
load and other workloads are not controlled. The ratio is not an isolated CUDA
kernel speedup or a throughput guarantee. Earlier CPU/joblib runs are not rerun
for this table.

| Stage | Earlier GPU | Optimized GPU |
| --- | ---: | ---: |
| Registration | 289.14 s | 31.14 s |
| Atlas labeling and cluster saving | 88.02 s | 57.26 s |
| Outlier removal | 65.89 s | 41.25 s |
| Hemisphere assessment | 95.80 s | 88.04 s |
| Inverse transform with Slicer | 184.40 s | 91.23 s |
| Hemisphere separation | 86.11 s | 85.03 s |
| Anatomical tract appending | 65.07 s | 61.53 s |

The Slicer row includes its child hardening command; nested timings are not
counted twice. Total time includes other shell/interpreter overhead. The final
four stages account for about 71% of the optimized total and primarily perform
CPU/file/scene work. This motivates the subsequent independent-file and lossless
serialization optimization; those later changes are excluded from this run.

## Output validation

Every expected filename and command exit code is checked. All 2474 final stage
files are byte-identical to the earlier complete GPU case:

| Output | Files | Total fibers | Byte-identical files |
| --- | ---: | ---: | ---: |
| Initial clusters | 800 | 245,946 | 800 |
| Outlier-removed clusters | 800 | 130,737 | 800 |
| Transformed clusters | 800 | 130,737 | 800 |
| Anatomical tracts | 74 | 93,766 | 74 |

The 12 saved affine transform parameters also match exactly, with maximum
absolute difference zero. Catalogue validation covers all 33 bilateral tracts
and eight commissural tracts, rather than checking only output counts. No
diffusion-measure export or interactive Slicer scene rendering is included.

```bash
export PYTHONDONTWRITEBYTECODE=1 WMA_GPU=on WMA_CPU_FAST=1 WMA_CPU_THREADS=4
export CUDA_VISIBLE_DEVICES=1 WMA_GPU_DEVICES=0
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

taskset -c 0-3 bash bin/wm_apply_ORG_atlas_to_subject.sh \
  -i "$INPUT_TRACTOGRAPHY" -o "$NEW_OUTPUT_DIRECTORY" -a "$ORG_ATLAS_ROOT" \
  -s "$SLICER_EXECUTABLE_WITH_DMRI" -n 4 -g on -r rig -x 1 -c 0
```
