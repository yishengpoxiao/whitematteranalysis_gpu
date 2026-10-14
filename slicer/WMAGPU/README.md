# WMA GPU Slicer Extension

Five workflow modules bundle the current repository's GPU/CPU backend, joblib 1.4.2,
and xlrd. No separate WMA installation is required; atlas data must be supplied.

Sources are in `slicer/WMAGPU/Modules/`, release archives in `slicer/WMAGPU/dist/`,
packaging tools in `tools/`, and validation scripts in `Testing/`.
The current release is `dist/WMAGPU-Installer.zip`, a universal installer
that detects the running Slicer version, Python, operating system, and architecture.
It contains pure Python sources and builds the matching extension layout locally.

Distribute the ZIP to other computers; it contains the maintained sources and
does not depend on a separate WMA checkout or a legacy local module path.

## Installation and use

1. Extract `WMAGPU-Installer.zip` on the target computer. Remove an
   older WMAGPU release in Extensions Manager and restart Slicer before installing.
2. Open the target Slicer, then **View → Python Interactor**, and run:

   ```python
   p = r"C:\path\to\WMAGPU-Installer\InstallWMAGPU.py"  # Windows example
   exec(compile(open(p, encoding="utf-8").read(), p, "exec"), {"__name__": "__main__"})
   ```

   On Linux/macOS, use the extracted script's absolute path instead.
3. Click **Install WMAGPU and required dependencies**. Missing SciPy/Matplotlib
   and their dependencies are installed automatically through this Slicer's
   `PythonSlicer`, using compatible binary wheels. Working packages are reused;
   Slicer's NumPy and VTK versions are constrained to their existing versions.
   WMA, joblib, and xlrd are bundled. Full command-line WMA requirements are not
   installed because the five Slicer workflows do not require them.
4. Optionally select **Also check / install CUDA PyTorch** before installing.
   Leave it unchecked for CPU mode. Restart after installation, then open
   **Diffusion → WMA GPU** and choose a workflow.
5. In Pipeline or Tractography Registration, use **Load tractography file...**
   to read `.vtk/.vtp` in the background. Inputs remain hidden unless **Show after
   loading** is checked. Existing nodes loaded through Add Data are also accepted.
   Select the input node, atlas directory, output directory, and backend, then run.

The universal ZIP is **not** an Extensions Manager "Install from file" package.
Run its installer inside Slicer; it creates and installs that Slicer's matching
package automatically. It requires Slicer 5.2+ and Python 3.8+. Layout adaptation
is tested for 5.2, 5.4, 5.8, 5.10, 5.12, and 6.0; actual application validation is
currently Linux Slicer 5.2.2 only. Windows and macOS need native validation.

CPU installation supports Windows, Linux, and macOS, including ARM when compatible
scientific-package wheels exist for the bundled Python/NumPy. CUDA setup supports
NVIDIA on Linux/Windows x86_64; macOS and ARM use CPU mode. Missing wheels, network
failures, and read-only Slicer Python directories produce a visible installation
error. The installer verifies imports in a fresh process before installing the
extension. NVIDIA drivers and CUDA hardware are not installed by this package.
Internet is required only when dependencies need downloading.

| Module | Function |
|---|---|
| WMA Pipeline | Selectable registration → atlas labeling → outlier removal → return to subject space → anatomical tracts |
| WMA Tractography Registration | Moving-to-fixed or atlas registration; retained scene transforms |
| WMA Groupwise Registration | Rigid, affine, or nonrigid population registration and registration atlas output |
| WMA Atlas Builder | Atlas construction through multisubject spectral clustering |
| WMA Merge & MRML | Checkbox node selection, named folders or combined fibers, and directory MRML export |

Opening a workflow page uses lazy computation imports and a shared environment
check. PyTorch, SciPy, and Matplotlib load when the requested operation needs them;
page setup performs no CUDA probe, dependency download, or package installation.
The first computation can therefore take longer than subsequent computations.
In a bounded Linux Slicer 5.2.2 measurement, initializing all five pages decreased
from 3.24 s to 0.035 s. This excludes Slicer startup, viewport rendering, and data
loading; Windows still requires native timing. See
[measurement details](https://github.com/yishengpoxiao/whitematteranalysis_gpu/blob/main/doc/gpu/progress-2026-10-14.md).

### Registration methods and transforms

**WMA Tractography Registration** accepts a **Moving tractography** node and a
**Fixed tractography** node. The resulting transform maps moving to fixed.
Alternatively choose **Atlas directory** as the fixed target source; the existing
ORG/built-atlas discovery rules still apply. Existing parent transforms on both
input nodes are included using scene RAS coordinates. Inputs remain unchanged;
the registered output keeps its geometry under the new Transform node, so the
transform can be inspected, detached, or applied to other data in Slicer.

The pairwise page and Pipeline expose every `wm_register_to_atlas_new.py` preset:
`affine`, `affine_fast`, `rigid_affine_fast`, `affine_neonate`, `nonrigid`,
`nonrigid_neonate`, `nonrigid_10`, `affineTEST`, and `nonrigidTEST`, plus `rigid`
and `custom`. Groupwise registration exposes its own CLI presets, including
`affine_fast_cheap` and `nonrigid_lowsample`, plus `rigid` and `custom`.
The default remains `affine` on all three pages.

Expand **Registration settings / custom schedule** to set the maximum fiber
length, random seed, intermediate output saving, and GPU float32.
The minimum length and backend controls are on the main form. Selecting `custom`
starts with the currently selected preset's schedule and enables editing: transform
type, sigma, iterations, maximum optimizer evaluations, fixed/mean and moving fiber
sample counts, initial/final steps, grid resolution, and points per fiber. Add or
remove scales as needed. Use `Affine` and/or `Rigid` for a linear schedule, or
`Nonrigid` for a separate deformable run. As with the CLI, run rigid/affine first
and use the aligned result as input for nonrigid registration. Nonrigid grids use
the upstream 200 mm cube centered at RAS zero; inputs must fit that atlas space.
`TEST` presets check execution and are not production-quality settings.
Registration uses the GPU selected in **Compute GPU** on the main form.

### GPU selection and memory

All four computation pages offer **Compute GPU**. Click **Detect GPUs** to list
CUDA devices asynchronously, or enter a nonnegative device index directly.
Indices refer to devices visible to Slicer's Python, including any
`CUDA_VISIBLE_DEVICES` remapping. Selection applies to registration, labeling,
outlier removal, group registration, and atlas building. It does not select the
GPU used by Slicer's OpenGL renderer. Opening a page does not start detection.
Only one WMA computation can run at a time within a Slicer process.

**Release unused GPU memory after each run** is enabled by default, including
failed runs. Pipeline also releases unused allocations before anatomical mapping,
saving, and display. **Release unused GPU memory now** performs the same cleanup
on the selected device when WMA is idle. The status log reports released memory,
allocated PyTorch memory, and remaining allocator reservations.

PyTorch keeps freed allocations in its caching allocator. Cleanup releases unused
blocks while preserving live tensors. CUDA context and Slicer rendering memory
can remain visible in `nvidia-smi`, even with zero PyTorch allocated/reserved bytes;
this is expected and does not imply that WMA retains a tensor. See the official
[CUDA memory management notes](https://docs.pytorch.org/docs/main/notes/cuda.html#memory-management)
and [allocated-memory accounting](https://docs.pytorch.org/docs/main/generated/torch.cuda.memory.memory_allocated.html).
In PyTorch 2.4, cuBLAS workspaces can also remain allocated after matrix operations;
see [workspace behavior](https://docs.pytorch.org/docs/2.4/notes/cuda.html#cublas-workspaces).

Pairwise and Pipeline registration reuse length-filtered moving/fixed inputs
across iterations. Input replacement, VTK geometry modification times, and length
limit changes invalidate the cache. Random sampling seeds, sample order, and
resampling remain unchanged. The run retains up to two filtered polydata copies
in host memory; these are released with the registration object.

All three workflows retain their final transforms as Slicer **Transform** nodes.
Pairwise registration and Pipeline save them as `.h5` files when file output is
checked; Groupwise always saves one `.h5` per subject alongside its outputs.
Groupwise transforms map each subject to the common group space. Linear and
B-spline transforms are written through Slicer's transform storage, preserving the
forward transform for reuse. Registered VTP/VTK files contain transformed points;
do not apply the same transform again when loading those files. Pipeline's
**Show clusters in original subject space** uses the inverse transform for its
tractography output; its retained Transform still maps subject to atlas.

### Selecting nodes to merge

In **WMA Merge & MRML**, check the desired nodes in **Merge cluster nodes in the
scene**, enter **Output name**, and choose:

- **Group separate nodes in a named folder**: puts checked nodes under a named
  Subject Hierarchy folder, keeping their separate geometry, colors, and transforms.
- **Combine all fibers into one tractography node**: appends checked tractography
  in scene coordinates into one named node. Parent transforms are included once;
  `cluster_idx` identifies the selected source of each point. Source data is not
  modified. The combined node uses a single display configuration.

All tractography nodes can be selected regardless of their names. New and removed
nodes update the checklist; **Refresh nodes** also picks up geometry or name changes.
**Select all** and **Clear selection** operate on the checklist. At least one checked
node and a nonempty output name are required. Slicer may suffix a node name if it
already exists. Directory file appending and optional `scene.mrml` export remain
separate actions.

Scene grouping and combining pause rendering and batch MRML updates. The node
checklist refreshes once after a scene batch, rather than scanning the scene as
each cluster is added. Table painting is suspended while rows are populated;
overlapping node classes produce only one checkbox per node. Batch state and
rendering are restored even when merging fails.

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

The log distinguishes atlas-defined categories (800 for ORG), available reference
cluster files, and nonempty clusters actually present in this subject. Progress
counts only nonempty subject clusters, so its denominator can be below 800.
Empty categories require no outlier calculation; saving cluster files still
writes every atlas category. Missing or empty reference files produce a warning
and retain that cluster's fibers without an outlier check.

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
**Register subject to atlas**. Outlier removal and anatomical tract mapping
remain independently optional. Results display as anatomically named tracts when
mappings are available, otherwise as one overview node. **Individual clusters**
loads nonempty clusters hidden so you can enable selected tracts in Data;
**Save only** avoids loading tractography results; a computed registration transform is still retained. Scene updates are batched, and saving finishes
before visualization. Pipeline saves VTP files and, when registration runs, a reusable `.h5` transform; it does not export MRML scenes.
Hemisphere mapping runs in atlas coordinates before returning results to subject
space. The separate WMA Label From Atlas page has been removed.

**Fast line display (Model nodes)** is enabled by default in Pipeline; registration
outputs also use Model nodes. This retains every fiber and its metadata while
avoiding FiberBundle tube/glyph setup. Uncheck it in Pipeline to use FiberBundle
nodes when SlicerDMRI is installed. Rendering very large visible results still
depends on graphics hardware; **Individual clusters** and **Save only** remain
available to limit scene work.

The background loader keeps VTK file reading off the GUI thread and creates MRML
nodes on the GUI thread. Hidden inputs avoid whole-brain rendering during loading.
Older VTK Python wrappers can still hold the interpreter lock during a large
read, delaying Python GUI callbacks; background reading does not guarantee a
fully responsive interface on every Slicer version.
Inputs without a parent transform share read-only geometry during preparation,
avoiding an unnecessary full copy. The labeling atlas is cached as one host-memory
entry, invalidated when its directory, file size, or modification time changes.
Selecting another atlas replaces that entry; no CUDA tensors are cached there.
Cluster extraction shares connectivity and geometry across clusters. Anatomical
mapping selects final tracts directly, avoiding intermediate per-cluster hemisphere
copies; saved anatomical files use the existing parallel writer.

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
mappings. The interface reports missing optional mappings. Model nodes work
without SlicerDMRI; FiberBundle display requires SlicerDMRI.

Auto selects GPU when CUDA is available, otherwise CPU accelerated. GPU requires
CUDA PyTorch compatible with **Slicer's own Python**. Triton provides optional
additional acceleration. The installer does not bundle torch/CUDA. Required scientific dependencies
are downloaded only when missing; CUDA downloads require the optional checkbox. CPU accelerated needs no torch; Original uses the original
joblib algorithm. Explicit GPU selection reports an error if CUDA is unavailable.
Float64 is the default; random sampling and classification remain on CPU.

## Optional PyTorch installation

Below Backend on the four computation pages, enable
**Allow downloading and installing CUDA PyTorch into Slicer**, then click
**Check / Install PyTorch**. The extension checks CUDA in a background process
using Slicer's PythonSlicer. Existing working CUDA PyTorch is left intact;
otherwise it selects a profile for Slicer's Python: `torch==2.4.1+cu121` for
Python 3.8–3.9, or `torch==2.10.0` from the CUDA 12.6 index for Python 3.10–3.14,
then verifies CUDA in a fresh process. These are official
[PyTorch wheel profiles](https://pytorch.org/get-started/previous-versions/).
The 2.4.1 profile is validated here; the newer profile requires native validation.
Automatic CUDA installation supports Linux/Windows x86_64; other environments
receive a CPU-mode suggestion. Existing working CUDA PyTorch is reused.

Downloads may total several GB and require internet access, a writable Slicer
Python environment, and a working NVIDIA driver. Logs appear in the page; after
failure, address the reported issue and retry. Restart Slicer after installation,
then select Auto/GPU. Successful pip installation with unavailable CUDA produces
a verification failure and driver-check guidance. Opt-in is session-only: it
triggers no installation at startup. Required dependency installation happens when you run the universal installer. Triton
is not installed separately; torch's platform dependencies determine its installation.

Workflows report stage progress. Long computations still run on Slicer's main
thread and can reduce interface responsiveness. The old local
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
  --universal --output /tmp/WMAGPU-Installer.zip
```

The packager always uses the repository's current `whitematteranalysis/`, excludes
build/lib and old `.sx` backends, and refuses to overwrite an existing archive.
The universal installer chooses the running Slicer major.minor version automatically.
For a version-specific offline extension ZIP, omit `--universal` and use
`--slicer-version 5.2`; that ZIP requires dependencies to be available already.

A standard `CMakeLists.txt` extension project is also provided. Configure with
`Slicer_DIR` from a Slicer development build and run
`cmake --build ... --target package`. See the
[official extension development guide](https://slicer.readthedocs.io/en/latest/developer_guide/extensions.html).
This extension supports local Install from file and is not listed in the public
Extensions Index.

## Validation

`Testing/test_startup.py` verifies lazy imports and CPU helpers without importing
PyTorch in a fresh interpreter. `Testing/verify_startup.py` measures all five page
initializations in a fresh Slicer process and checks that heavy computation packages
are not newly imported. The registration tests also compare sampled arrays and RNG
state with the previous preprocessing path and check cache invalidation. The Slicer
registration/merge test includes 800-node batching, one checklist refresh per batch,
duplicate node-class handling, and cleanup after a failed merge.

`Testing/test_registration.py` verifies every CLI preset schedule and bounded
CPU/CUDA affine, rigid, and nonrigid registration, inverse transforms, rigid
orthogonality, and groupwise rigid/nonrigid output. `Testing/verify_registration_merge.py`
checks the actual Slicer controls, parent transform handling, affine/B-spline HDF5
save/reload, retained pairwise/group/pipeline transforms, and both merge methods
with checked and excluded nodes. The optimizer tests use deliberately small samples
and evaluation limits; they establish execution, not registration quality.

`Testing/test_installer.py` verifies generated layouts, minimum versions, dependency
constraints, existing-package reuse, and CUDA/CPU platform selection.
`Testing/verify_universal_installer.py` exercises the real installer dialog,
asynchronous dependency checks, failure reporting, Extension Manager installation,
and optional CUDA setup with isolated settings.


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

`Testing/test_loading.py` checks byte-identical anatomical VTP output against the
previous separate-and-append path, single-entry atlas invalidation, and real CUDA
cleanup with live tensors. `Testing/verify_gpu_loading.py` checks threaded VTK/VTP
reading, GUI responsiveness, full geometry preservation, selected-device execution,
successful/failed-run cleanup, workflow locking, and Model display in native Slicer.
`Testing/benchmark_loading.py` measures cluster and anatomical extraction with
deterministic geometry and checks serialized output equality. It excludes disk
reading, saving, rendering, and lazy imports from the operation timings.

The bundled WMA also streams CPU atlas labeling, reuses CPU registration workers,
and fuses GPU registration probabilities without changing float64 precision.
See the [CPU/GPU optimization record](../../doc/gpu/progress-2026-10-14-streaming.md)
for correctness checks and bounded timings. No additional dependencies are needed.

GPU labeling also performs normalization and nearest-centroid lookup on CUDA,
including without Triton. With Triton, affine registration reuses a fused transform
and matrix/coordinate buffers. See the [GPU continuation record](../../doc/gpu/progress-2026-10-15-gpu.md)
for real-subject classification and bounded registration comparisons.

Geometry preparation also avoids full tractogram copies, and large CPU distance
tiles use SciPy's compiled kernel. See the [geometry/CPU record](../../doc/gpu/progress-2026-10-15-geometry-cpu.md)
for numerical validation and timings.


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
