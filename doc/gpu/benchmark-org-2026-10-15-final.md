# Complete optimized ORG CPU/GPU comparison — 2026-10-15

## Setup

Run the requested complete subject through
`bin/wm_apply_ORG_atlas_to_subject.sh`, using frozen source `c9c622c`. The input
contains 328,228 fibers, 31,717,748 points and 3,555,014,103 bytes. The original
input size and modification timestamp remain unchanged. Use the ORG-800FC-100HCP
atlas, ORG-RegAtlas-100HCP registration target, standard seed 1000 and the full
subject after the standard 40 mm length filter. No reduced data or precomputed
registration replaces a pipeline stage.

Use Xeon Gold 6330, four physical cores pinned as 0–3, RTX 4090 physical device 1,
Python 3.8, PyTorch 2.4.1+cu121, VTK 9.1 and Linux Slicer 5.2.2/SlicerDMRI.
OpenBLAS/MKL/OpenMP/NumExpr each use one thread. Both profiles use
`WMA_CPU_THREADS=4`, `WMA_CPU_FAST=1`, `-n 4 -r rig -x 1 -c 0`; GPU uses `-g on`,
CPU matrix uses `-g off`. `rig` selects `rigid_affine_fast`, including rigid and
affine scales. Default numerical precision remains float64.

GPU and CPU profiles run sequentially from the same source snapshot, with
separate new output directories and isolated Slicer settings. The shell selects
LZ4 intermediates and compact zlib anatomical export. Full logs, command timings,
resource measurements, validation scripts and geometry remain outside Git.

## Results

| Current mode | Complete wall time | Maximum host RSS |
| --- | ---: | ---: |
| CPU matrix, four-worker budget | 448.67 s (7:29) | 9.85 GiB |
| GPU, one RTX 4090 | 166.21 s (2:46) | 10.48 GiB |

The current GPU profile is approximately 2.70x faster than the current CPU matrix
profile. The [earlier full-case comparison](benchmark-org-2026-10-14.md) measured
879.43 s for GPU and approximately 3981 s for CPU matrix. The observed reductions
are about 5.29x and 8.87x respectively. Original joblib's earlier 31:24 measurement
is retained as historical context; it was not rerun with the new common I/O stages.

These are single full-case measurements on a shared workstation, including
actual reading/writing, Python imports, first-use CUDA work and Slicer startup.
CPU frequency, storage activity, other workloads and page-cache conditions are
not controlled. The earlier CPU run encountered substantial storage contention.
Historical ratios combine algorithm/memory/I/O changes and changing machine
conditions; they are not isolated kernel speedups or throughput guarantees.

| Stage | Current CPU matrix | Current GPU |
| --- | ---: | ---: |
| Registration | 84.92 s | 30.24 s |
| Atlas labeling and cluster saving | 132.98 s | 26.20 s |
| Outlier removal | 157.19 s | 26.81 s |
| Hemisphere assessment | 13.59 s | 26.47 s |
| Inverse transform with Slicer | 34.15 s | 29.81 s |
| Hemisphere separation | 9.04 s | 8.85 s |
| Anatomical tract appending | 14.85 s | 14.90 s |

The Slicer row measures the parent command, including concurrent Slicer workers.
Other shell overhead explains the difference between stage sums and total wall
time. GPU/CPU labels describe the mathematical backend; file/scene stages remain
CPU operations. Their variation illustrates why each stage does not necessarily
improve between two shared-machine runs.

## Scientific output validation

Both profiles exit successfully, with every expected filename and completion
marker present. Compare every output against the original complete GPU case:

| Output | Files per profile | Total fibers per profile |
| --- | ---: | ---: |
| Initial clusters | 800 | 245,946 |
| Outlier-removed clusters | 800 | 130,737 |
| Transformed clusters | 800 | 130,737 |
| Anatomical tracts | 74 | 93,766 |

The complete 74-tract catalogue covers 33 bilateral and eight commissural tracts.
Validation checks ordered connectivity, point counts, coordinates, point/cell
arrays, array names/types/components, and string field metadata. Integer arrays
match exactly. Compression changes encoded VTP bytes while preserving their data.

For GPU, decoded coordinates and numeric arrays match exactly across all 2474
files; the saved 12 affine parameters have maximum absolute difference zero.
For CPU matrix, coordinate differences from the GPU reference are at most
1.90735e-6 mm and transform-parameter differences at most 7.54952e-14. Floating
coordinates pass `rtol=1e-6, atol=1e-5`; floating attributes pass
`rtol=1e-6, atol=1e-7`. Per-cluster fiber counts, ordered topology, array types and
string fields match. No diffusion-measure export, nonrigid full-case test or interactive scene display
is included.

## Reproduction

Use separate fresh output directories. Provide a Slicer executable or wrapper
that discovers SlicerDMRI. The Linux benchmark uses an isolated settings directory
and disables personal Slicer startup scripts.

```bash
export PYTHONDONTWRITEBYTECODE=1 WMA_CPU_FAST=1 WMA_CPU_THREADS=4
export CUDA_VISIBLE_DEVICES=1 WMA_GPU_DEVICES=0
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

taskset -c 0-3 bash bin/wm_apply_ORG_atlas_to_subject.sh \
  -i "$INPUT_TRACTOGRAPHY" -o "$GPU_OUTPUT_DIRECTORY" -a "$ORG_ATLAS_ROOT" \
  -s "$SLICER_EXECUTABLE_WITH_DMRI" -n 4 -g on -r rig -x 1 -c 0

taskset -c 0-3 bash bin/wm_apply_ORG_atlas_to_subject.sh \
  -i "$INPUT_TRACTOGRAPHY" -o "$CPU_OUTPUT_DIRECTORY" -a "$ORG_ATLAS_ROOT" \
  -s "$SLICER_EXECUTABLE_WITH_DMRI" -n 4 -g off -r rig -x 1 -c 0
```
