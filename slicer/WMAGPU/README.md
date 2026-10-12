# WMA GPU Slicer Extension

Six workflow modules bundle the current repository's GPU/CPU backend, joblib 1.4.2,
and xlrd. No separate WMA installation is required; atlas data must be supplied.

Sources are in `slicer/WMAGPU/Modules/`, release archives in `slicer/WMAGPU/dist/`,
packaging tools in `tools/`, and validation scripts in `Testing/`.
The current English-language release is
`dist/WMAGPU-Slicer-5.2-2026-10-12-parallel.zip`, including optional PyTorch setup.

The local legacy path `/data/yijie/wma-GPU/slicer` is a symlink to `Modules/` for
existing Slicer Additional module paths. Obsolete source copies, `.sx` archives,
and installation scripts were removed. Distribute the ZIP to other computers;
it does not depend on this local symlink.

## Installation and use

1. In Slicer 5.2, open **Extensions Manager → Install from file**, select the
   `WMAGPU-Slicer-5.2-*.zip` archive, and restart Slicer.
2. Select a workflow under **Diffusion → WMA GPU**.
3. Load `.vtk/.vtp` tractography using Add Data. Select the input node, atlas
   directory, output directory, and backend, then run the workflow.

| Module | Function |
|---|---|
| WMA Pipeline | Affine registration → atlas labeling → outlier removal → return to subject space → anatomical tracts |
| WMA Register to Atlas | Affine subject-to-atlas registration |
| WMA Groupwise Registration | Population affine registration and registration atlas output |
| WMA Atlas Builder | Atlas construction through multisubject spectral clustering |
| WMA Merge & MRML | Cluster merging and scene.mrml export |

Outlier removal computes independent occupied clusters in parallel. **Outlier
workers** defaults to **Auto**, using available CPU cores and limiting GPU
concurrency by free VRAM. Set **1** for sequential execution or choose a larger
limit explicitly (up to the number of occupied clusters, with the GPU memory
bound still applied). CPU workers use one distance thread each; GPU workers use
separate CUDA streams with per-worker tile budgets. The subject is resampled once,
and progress, event processing, and final mask assembly stay on Slicer's main
thread. More workers can increase memory use or reduce throughput; benchmark on
your atlas and hardware. This currently uses one GPU. Auto honors
`WMA_CPU_THREADS` as the CPU budget when it is set.

A bounded ORG benchmark (32 real reference clusters, 256 derived subject fibers,
float64, one timing per setting after warm-up) produced identical masks and 39
rejections at every worker count:

| Workers | CPU seconds | GPU seconds |
| --- | ---: | ---: |
| 1 | 45.07 | 0.89 |
| 4 | 14.51 | 1.21 |
| 16 | 9.75 | 1.26 |
| Auto (32 occupied clusters) | 13.52 | 1.31 |

These shared-workstation measurements are not a full-pipeline speed guarantee.
CPU benefited from cluster parallelism; this small GPU workload was faster with
one worker. Auto chooses high concurrency, not a measured optimum. Choose 1 to
avoid GPU scheduling overhead when it dominates.

For labeling already registered tractography, use **WMA Pipeline** and uncheck
**Register subject to atlas (affine)**. Outlier removal and anatomical tract mapping
remain independently optional. Results display as anatomically named tracts when
mappings are available, otherwise as one overview node. **Individual clusters**
loads nonempty clusters hidden so you can enable selected tracts in Data;
**Save only** avoids scene loading. Scene updates are batched, and saving finishes
before visualization. Pipeline saves VTP files only; it does not export MRML scenes.
Hemisphere mapping runs in atlas coordinates before returning results to subject
space. The separate WMA Label From Atlas page has been removed.

Select the ORG atlas parent directory (for example `/data/software/ORG-Atlases`)
or its `ORG-800FC-100HCP` labeling folder. Registration automatically uses
`ORG-RegAtlas-100HCP/registration_atlas.vtk` under the same parent, following
`wm_apply_ORG_atlas_to_subject.sh`.

Built atlases accept the run root (latest completed numbered iteration), an
`iteration_XXXXXX` directory, or `initial_clusters`. Other parent directories
are accepted when they contain exactly one `atlas.p`; ambiguous matches require
a more specific selection. Core functions also accept an `atlas.p` file path,
expand `~` and environment variables, and registration accepts `.vtk`/`.vtp`
files directly. Cluster references and anatomical mappings come from the resolved
labeling atlas, so unrelated sibling directories are excluded.

Official ORG atlases and atlases built with this tool are supported. The complete Pipeline additionally
requires a registration atlas, reference cluster files, and `T_*.mrml` anatomical
mappings. The interface reports missing optional mappings. Without SlicerDMRI,
fibers use Model nodes; with SlicerDMRI, they use FiberBundle nodes.

Auto selects GPU when CUDA is available, otherwise CPU accelerated. GPU requires
CUDA PyTorch compatible with **Slicer's own Python**. Triton provides optional
additional acceleration. The archive does not bundle torch/CUDA and downloads
nothing by default. CPU accelerated needs no torch; Original uses the original
joblib algorithm. Explicit GPU selection reports an error if CUDA is unavailable.
Float64 is the default; random sampling and classification remain on CPU.

## Optional PyTorch installation

Below Backend on the four computation pages, enable
**Allow downloading and installing CUDA PyTorch into Slicer**, then click
**Check / Install PyTorch**. The extension checks CUDA in a background process
using Slicer's PythonSlicer. Existing working CUDA PyTorch is left intact;
otherwise it installs `torch==2.4.1+cu121` and dependencies from the
[official CUDA 12.1 index](https://docs.pytorch.org/get-started/previous-versions/#v241),
then verifies CUDA in a fresh process. This matches the version validated for
this project. Automatic installation supports Linux/Windows x86_64 with Python
3.8–3.12; other environments receive a CPU-mode suggestion. Windows is not yet tested.

Downloads may total several GB and require internet access, a writable Slicer
Python environment, and a working NVIDIA driver. Logs appear in the page; after
failure, address the reported issue and retry. Restart Slicer after installation,
then select Auto/GPU. Successful pip installation with unavailable CUDA produces
a verification failure and driver-check guidance. Opt-in is session-only: it
changes no Slicer configuration and triggers no installation at startup. Triton
is not installed separately; torch's platform dependencies determine its installation.

Workflows report stage progress. Long computations still run on Slicer's main
thread and can reduce interface responsiveness. Registration pages currently expose
affine mode; nonrigid backend APIs are available through scripts. The old local
directory was migrated as described above; existing Slicer settings remain unchanged.
Remove old WMA Additional module paths when installing the extension to avoid
duplicate module names. Remove an installed older WMAGPU release before installing
the new archive.

## Development and packaging

`Modules/` contains the module entry points. Shared computation and GUI helpers
live in `_vendor/` so Slicer does not scan them as standalone modules. Pure Python
packaging requires no Slicer SDK:

```bash
python slicer/WMAGPU/tools/package_extension.py \
  --slicer-version 5.2 --output /tmp/WMAGPU-Slicer-5.2.zip
```

The packager always uses the repository's current `whitematteranalysis/`, excludes
build/lib and old `.sx` backends, and refuses to overwrite an existing archive.
The layout targets the specified Slicer major.minor version. Slicer 5.2.2 on Linux
was tested; other versions require repackaging with their version and validation.

A standard `CMakeLists.txt` extension project is also provided. Configure with
`Slicer_DIR` from a Slicer development build and run
`cmake --build ... --target package`. See the
[official extension development guide](https://slicer.readthedocs.io/en/latest/developer_guide/extensions.html).
This extension supports local Install from file and is not listed in the public
Extensions Index.

## Validation

`Testing/test_outlier_parallel.py` compares sequential/parallel CPU and CUDA masks
against the original per-fiber algorithm, including unequal subject/reference
cluster sizes, interleaved labels, missing/empty reference clusters, concurrent
worker execution, main-thread callbacks, and worker failures.
`Testing/benchmark_outliers.py` measures worker counts on a real atlas with a small
derived subject; it does not measure a full clinical pipeline.

`Testing/verify_pipeline.py` exercises the five module widgets, labeling without
registration, anatomical mapping before the inverse transform, output files without
MRML export, result replacement without deleting unrelated nodes, all display
modes, empty cluster omission, and batched creation/removal of 800 hidden nodes.
It uses a bounded synthetic subject and substitutes registration and labeling;
hemisphere mapping, transforms, file writing, and Slicer scene operations run normally.


Slicer 5.2.2 on Linux was used to verify Extension Manager installation, the five current
modules, node selection without SlicerDMRI, CPU/GPU labeling agreement, affine
registration, parallel writing, and MRML export. Synthetic tests also cover
cluster numbering through atlas construction/reload/labeling, groupwise output,
single-file MRML, and explicit GPU failure when CUDA is unavailable. The full ORG
anatomical Pipeline still requires validation with actual atlas data; these checks
do not replace real-case validation.

`Testing/verify_torch_installer.py` checks default opt-out, a real CUDA subprocess
probe, skipping an existing working installation, and simulated pip success,
pip failure, CUDA verification failure, process startup failure, and cross-page
concurrency limits. It downloads or installs nothing; a first full download still
depends on network conditions.

Run the bounded core tests in an existing WMA Python environment:

```bash
PYTHONPATH=. python -m unittest discover -s slicer/WMAGPU/Testing -p test_core.py -v
```

`Testing/install_extension.py` uses `WMA_EXTENSION_ARCHIVE`,
`WMA_EXTENSION_INSTALL_ROOT`, and `WMA_EXTENSION_SETTINGS` for the archive,
isolated install directory, and isolated extension settings. After installation,
start a fresh Slicer process with
`--disable-settings --additional-module-path <install-root>/WMAGPU/lib/Slicer-5.2/qt-scripted-modules`
and `--python-script Testing/verify_extension.py` to verify installed code. Keep
`WMA_EXTENSION_INSTALL_ROOT` set. Tests use `--no-main-window --ignore-slicerrc`;
without a display server, start through `xvfb-run -a`.
