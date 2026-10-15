# WMA GPU Slicer Extension

Five workflows bundle this repository's WMA backend, joblib and xlrd. Atlas data
must be supplied separately. The release is `dist/WMAGPU-Installer.zip`; it
adapts its extension layout to the running Slicer version and platform.

## Installation and use

1. Extract `WMAGPU-Installer.zip`. Remove an older WMAGPU release in Extensions
   Manager and restart Slicer before installing.
2. Open **View → Python Interactor** in the target Slicer and run:

   ```python
   p = r"C:\path\to\WMAGPU-Installer\InstallWMAGPU.py"
   exec(compile(open(p, encoding="utf-8").read(), p, "exec"), {"__name__": "__main__"})
   ```

   On Linux/macOS, substitute the extracted script's absolute path.
3. Click **Install WMAGPU and required dependencies**. Missing scientific
   dependencies are installed through Slicer's `PythonSlicer`, preserving its
   NumPy and VTK versions. WMA, joblib and xlrd are bundled.
4. Optionally enable **Also check / install CUDA PyTorch**; leave it unchecked
   for CPU mode. Restart after installation, then open **Diffusion → WMA GPU**.
5. Load `.vtk/.vtp` through **Load tractography file...**, or select an existing
   scene node. Inputs stay hidden unless **Show after loading** is checked.
   Select the workflow inputs, output directory and backend, then run.

The universal ZIP must be run through its installer inside Slicer. It is not an
Extensions Manager **Install from file** archive. Remove obsolete WMA Additional
module paths to avoid duplicate module names.

The installer requires Slicer 5.2+ and Python 3.8+. Layout tests cover Slicer
5.2–6.0; native application validation is Linux Slicer 5.2.2. Windows/macOS and
other releases require native checks. CPU installation supports platforms with
compatible scientific wheels, including ARM. Automatic CUDA setup supports
Linux/Windows x86_64 with an NVIDIA driver; macOS/ARM use CPU mode. Downloads need
internet access and writable Slicer Python storage. The installer verifies
imports in a fresh process and reports missing wheels or installation failures.

## Workflows

| Module | Function |
| --- | --- |
| WMA Pipeline | Registration, atlas labeling, optional outliers and anatomical tracts |
| WMA Tractography Registration | Moving-to-fixed or atlas registration |
| WMA Groupwise Registration | Population registration and registration atlas output |
| WMA Atlas Builder | Multisubject spectral clustering and atlas construction |
| WMA Merge & MRML | Selected-node grouping/combining and directory MRML export |

### Pipeline and atlas selection

Uncheck **Register subject to atlas** for already registered inputs. Outlier
removal and anatomical mapping are independently optional. The pipeline saves
VTP files and a reusable `.h5` transform when registration runs; it does not
export MRML scenes. Anatomical mapping runs in atlas coordinates before results
are returned to subject space.

Choose anatomical display, an overview node, **Individual clusters** or
**Save only**. Individual clusters load hidden for selective display; Save only
avoids loading tractography while retaining a computed registration Transform.
**Fast line display (Model nodes)** preserves fibers and metadata without
FiberBundle tube/glyph setup. FiberBundle display requires SlicerDMRI.
Large visible results still depend on rendering hardware.

Select the ORG atlas parent directory or `ORG-800FC-100HCP`. Registration uses
`ORG-RegAtlas-100HCP/registration_atlas.vtk` under the same parent. Built atlases
accept a run root, `iteration_XXXXXX` or `initial_clusters`; a run root resolves
the latest completed numbered iteration. Other parents are accepted when they
contain exactly one `atlas.p`. Ambiguous matches require a more specific folder.
Core functions also accept `atlas.p` paths, expand environment variables and `~`,
and accept `.vtk/.vtp` registration targets directly.

Full anatomical processing requires reference cluster files and `T_*.mrml`
mappings from the resolved labeling atlas. Missing optional mappings are reported.
The log distinguishes atlas categories, available reference files and nonempty
subject clusters. Outlier progress counts only nonempty subject clusters, which
can be fewer than 800. Saving still includes all atlas categories; missing/empty
references produce a warning and retain the affected fibers without a check.

**Outlier workers** defaults to Auto, using available CPUs and free VRAM. Set 1
for sequential execution or choose a larger limit. CPU workers use one distance
thread each; GPU workers use separate streams on one selected GPU. More workers
can reduce GPU throughput. Auto honors `WMA_CPU_THREADS` when set.
Atlas reference statistics are cached across subjects and sessions.

### Registration and transforms

Tractography Registration maps **Moving tractography** to **Fixed tractography**,
or to an atlas target. Existing parent transforms are included in scene RAS
coordinates. Input nodes are preserved; the registered scene output keeps its
geometry under the resulting Transform node.

Pairwise registration and Pipeline offer all `wm_register_to_atlas_new.py`
presets: `affine`, `affine_fast`, `rigid_affine_fast`, `affine_neonate`,
`nonrigid`, `nonrigid_neonate`, `nonrigid_10`, `affineTEST` and `nonrigidTEST`,
plus `rigid` and `custom`. Groupwise registration offers its CLI presets,
including `affine_fast_cheap` and `nonrigid_lowsample`, plus `rigid` and `custom`.
The default is `affine`.

Expand **Registration settings / custom schedule** for length limits, seed,
intermediate saving and float32. Custom schedules edit transform type, sigma,
iterations, evaluation limits, sampling counts, steps, grid resolution and points
per fiber. Run rigid/affine alignment before a separate nonrigid run. Nonrigid
grids use the upstream 200 mm cube centered at RAS zero; inputs must fit this
space. TEST presets verify execution rather than registration quality.

All registration workflows retain final Slicer **Transform** nodes. Pairwise and
Pipeline save `.h5` files when file output is enabled; Groupwise saves one per
subject. Groupwise transforms map subjects to the common group space. Registered
files already contain transformed points, so applying the transform again would
transform them twice. Pipeline's **Show clusters in original subject space**
uses the inverse for tractography; its retained Transform still maps to atlas.

### Merging nodes

Check the desired nodes, enter **Output name**, then choose:

- **Group separate nodes in a named folder**: keeps each node's geometry, colors
  and transforms under one Subject Hierarchy folder.
- **Combine all fibers into one tractography node**: appends selected geometry
  in scene coordinates, including parent transforms once. Source data is
  preserved; `cluster_idx` identifies sources. The result has one display setup.

Use **Select all**, **Clear selection** or **Refresh nodes** to manage the list.
Slicer may suffix an existing output name. Directory appending and optional
`scene.mrml` export are separate actions.

## GPU selection and dependencies

Auto uses CUDA when available, otherwise accelerated CPU. Original selects the
per-fiber joblib algorithm. Explicit GPU selection fails if CUDA is unavailable.
Float64 is the default; float32 is optional. Workflow pages defer heavy imports
and GPU detection until needed; the first computation includes that setup.

All four computation pages provide **Compute GPU**. Use **Detect GPUs**, or
enter a nonnegative index. Indices follow devices visible to Slicer's Python,
including `CUDA_VISIBLE_DEVICES` remapping. This selects the WMA computation GPU,
not Slicer's OpenGL renderer. Only one WMA computation runs at a time per Slicer
process.

**Release unused GPU memory after each run** is enabled by default, including
failures. **Release unused GPU memory now** performs cleanup while idle. Live
tensors are preserved. CUDA context, rendering memory and library workspaces can
remain visible in `nvidia-smi` even when PyTorch tensor allocations are zero.

### Optional CUDA PyTorch installation

Enable **Allow downloading and installing CUDA PyTorch into Slicer**, then click
**Check / Install PyTorch**. A background subprocess checks Slicer's own Python;
Conda PyTorch does not enable Slicer CUDA. Existing working CUDA PyTorch is reused.
Otherwise the installer chooses `torch==2.4.1+cu121` for Python 3.8–3.9, or
`torch==2.10.0` from the CUDA 12.6 index for Python 3.10–3.14, then verifies CUDA
in a fresh process. The 2.4.1 profile is validated; the newer profile requires
native validation. Triton is optional and is not installed separately.

Downloads may total several GB. After installation, restart Slicer and select
Auto/GPU. Installation failures appear in the log and can be retried after the
reported issue is resolved. Opt-in is session-only; opening a page triggers no
installation. Required scientific dependencies are installed by the universal
installer, independently of the optional CUDA checkbox.

## Development and validation

Module entry points live in `Modules/`; shared helpers and bundled dependencies
live in `Modules/_vendor/`. The packager reads the repository's current WMA
sources, excludes generated bytecode, and refuses to overwrite an archive:

```bash
python slicer/WMAGPU/tools/package_extension.py \
  --universal --output /tmp/WMAGPU-Installer.zip
```

For a version-specific offline ZIP, omit `--universal` and specify
`--slicer-version 5.2`; dependencies must already be installed. A CMake extension
project is also provided for Slicer development builds. This extension is
locally distributed and is not listed in the public Extensions Index.

Validation scripts remain in `Testing/`: unit tests cover core computations,
registration presets/transforms, installer constraints, parallel outliers,
lazy startup and loading. Native `verify_*.py` scripts exercise installation,
widgets, transforms, merging, result display and cleanup. Synthetic registration
checks establish execution, not clinical registration quality.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  python -m unittest discover -s slicer/WMAGPU/Testing -p test_core.py -v
```

For installed-code verification, set `WMA_EXTENSION_INSTALL_ROOT`, use a fresh
Slicer process with `--additional-module-paths` pointing to
`<install-root>/WMAGPU/lib/Slicer-<major.minor>/qt-scripted-modules`, and run
`Testing/verify_extension.py` through `--python-script`. Installer tests also use
`WMA_EXTENSION_ARCHIVE`, `WMA_EXTENSION_SETTINGS` and `WMA_INSTALLER_ARCHIVE`.
Use isolated settings; headless Linux checks run through `xvfb-run -a`.
