# WhiteMatterAnalysis

WhiteMatterAnalysis (WMA) provides fiber clustering and tractography analysis tools.
This fork adds CUDA GPU acceleration, a vectorized CPU backend, and a five-module
3D Slicer extension. It is based on the [upstream WMA project](https://github.com/SlicerDMRI/whitematteranalysis).

## Full-pipeline performance

On **2026-10-28**, source revision `a698f51` processed C028 (approximately
**330,000 streamlines, 3.6 GB**) using `wm_apply_ORG_atlas_to_subject.sh`,
ORG-800FC-100HCP and ORG-RegAtlas-100HCP. All modes used four physical CPU cores,
`-n 4 -r rig -x 1 -c 0`, seed 1000 and float64. Hardware: Intel Xeon Gold 6330
and one RTX 4090; Python 3.8, PyTorch 2.4.1/CUDA 12.1, Slicer 5.2.2 + SlicerDMRI.
CUDA 12.1 is the benchmark environment, not a requirement.

| Step | Joblib (4 workers), s | CPU matrix, s | CPU speedup | GPU, s | GPU speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Registration (rigid + affine) | 371.4 | 106.4 | 3.49x | 25.7 | 14.46x |
| Atlas labeling and cluster saving | 1001.2 | 95.6 | 10.47x | 25.5 | 39.19x |
| Outlier removal | 201.6 | 82.2 | 2.45x | 28.5 | 7.08x |
| Hemisphere assessment | 15.6 | 14.4 | 1.09x | 24.3 | 0.64x |
| Inverse transform with Slicer | 32.7 | 30.1 | 1.09x | 47.4 | 0.69x |
| Hemisphere separation | 9.0 | 8.8 | 1.03x | 8.8 | 1.03x |
| Anatomical tract assembly | 15.0 | 14.9 | 1.01x | 15.0 | 1.00x |
| **Total** | **1648.7** | **354.2** | **4.65x** | **178.6** | **9.23x** |

Totals: **27.48 min** (joblib), **5.90 min** (CPU matrix), **2.98 min** (GPU).
Speedups are relative to joblib's original per-fiber mathematical backend within
the current pipeline; shared I/O improvements apply to all modes. These were
single sequential runs with fresh outputs and empty atlas caches, four WMA
workers and one thread per BLAS/OpenMP library. Totals include I/O, startup and
Slicer processing; filesystem caches and workstation load were not reset.
The final four stages use identical CPU/Slicer code across modes.

All modes produced 800 initial, outlier-removed and transformed clusters plus
74 anatomical tracts (245,946 initial fibers, 130,737 retained, 93,766 anatomical).
Validation covered all **4,874 VTPs**, the full registered tractogram and
**11 transforms**:

- **CPU matrix:** byte-identical VTP/VTK outputs and exact transform parameters.
- **GPU:** identical membership, connectivity and attributes; maximum coordinate
  difference **1.91e-6 mm**, maximum transform-parameter difference **3.30e-13**.
  All passed coordinate tolerances `rtol=1e-6, atol=1e-5` and transform tolerances
  `rtol=1e-10, atol=1e-10`.
- The new joblib outputs were also byte-identical to the previous `e5abd52` run.

The CPU total retains its measured **58.1 s** input read. Later read-only tests
reproduced disk/page-cache waits in joblib too; cached reads took **2.3–2.5 s**
across all modes. Large-file OS read-ahead reduced partial-cache waits in separate
controls (`WMA_IO_PREFETCH=0` disables it). The full-pipeline table has not been
adjusted using isolated read tests. See the [acceleration overview](doc/acceleration.md)
for implementation and numerical validation details.

# GPU and 3D Slicer quick start

## Python / command line

Install this fork from a local checkout:

```shell
git clone https://github.com/yishengpoxiao/whitematteranalysis_gpu.git
cd whitematteranalysis_gpu
python -m pip install .
```

Choose one dependency profile in the same Python environment:

```shell
# CPU: base dependencies, no PyTorch required
python -m pip install -r requirements_cpu.txt

# GPU: install a CUDA-enabled PyTorch build first, then the remaining dependencies
python -m pip install -r requirements_gpu.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`python -m pip install .` installs the current CPU profile, including
`threadpoolctl` for scoped BLAS control. The GPU profile includes the CPU profile
and `torch>=2.4.1`, without fixing a CUDA wheel index or exact PyTorch release.
Existing compatible PyTorch installations are reused.

Install the CUDA-enabled PyTorch build matching your Python, GPU and NVIDIA
driver using the [official PyTorch selector](https://pytorch.org/get-started/locally/)
or [version-specific commands](https://pytorch.org/get-started/previous-versions/).
WMA is not restricted to CUDA 12.1. A CPU-only PyTorch build cannot run CUDA;
check `torch.cuda.is_available()` after installation. Triton kernels are used when
Triton is available; the torch CUDA path remains available without Triton.
Linux / Python 3.8 and Slicer 5.2.2 / Python 3.9 were tested; Windows has not been tested.

| Setting | Behavior |
|---|---|
| `WMA_GPU=auto` (default) | Use available CUDA; otherwise use accelerated CPU |
| `WMA_GPU=on` | Require CUDA; raise an error if unavailable |
| `WMA_GPU=off` | Use CPU |
| `WMA_GPU=off`, `WMA_CPU_FAST=0` | Use the original per-fiber CPU implementation |
| `WMA_GPU_DEVICES=0,1` | Select visible CUDA device indices for supported multi-GPU operations |
| `WMA_GPU_FLOAT32=1` | Opt into float32; default is float64 |
| `WMA_ATLAS_CACHE=/path/to/cache` | Override the automatic outlier atlas cache; `off` disables it |

These environment variables apply to the Python/CLI backend. Slicer pages provide
their own Backend selection. Some CLI tools also expose `-gpu auto|on|off`,
`-gpu_float32`, and `-gpu_devices`; check the individual tool's `--help`.
For those tools, use `-gpu on` explicitly: their default `-gpu auto` overrides
`WMA_GPU`. An explicit GPU request fails before processing if CUDA is unavailable.

The accelerated CPU backend streams distance/similarity tiles and honors CPU
affinity. CUDA performs distance, probability, embedding and centroid operations
on the selected device, using optional Triton kernels with a PyTorch fallback.
Float64 is the default; float32 is an explicit accuracy/performance tradeoff.
Outlier removal shares cached atlas reference statistics across subjects.

| Additional setting | Behavior |
| --- | --- |
| `WMA_CPU_THREADS=4` | Set the accelerated CPU worker budget; otherwise use an affinity-aware default |
| `WMA_CPU_BLAS_THREADS=0` | Disable automatic atlas-labeling BLAS caps; a positive value sets a ceiling |
| `WMA_VTP_COMPRESSION=lz4` | Select faster lossless VTP writing; library/Slicer defaults use compact zlib |

## 3D Slicer extension

Use the [universal installer](slicer/WMAGPU/dist/WMAGPU-Installer.zip).
Extract it, run `InstallWMAGPU.py` inside the target Slicer's Python Interactor,
and click **Install WMAGPU and required dependencies**. The installer detects
Slicer's version and platform, installs missing scientific dependencies through
Slicer's Python while preserving its NumPy/VTK versions, and installs the matching
extension. This ZIP is not an Extensions Manager "Install from file" archive.
Remove an older WMAGPU release and restart before installing.
See [installation instructions](slicer/WMAGPU/README.md#installation-and-use).

The five workflows cover Pipeline, moving-to-fixed/atlas registration, groupwise
registration, atlas building, and selected-node merging. CPU mode needs no
PyTorch; optional CUDA installation targets Slicer's own Python. Registration
retains Transform nodes, and Pipeline can display anatomical tracts or save
results without loading them.

The installer adapts to Slicer's version and platform. Native validation is
Linux Slicer 5.2.2; Windows/macOS and newer Slicer versions require native checks.
macOS/ARM use CPU mode; automatic CUDA setup targets Linux/Windows x86_64.
See the [extension guide](slicer/WMAGPU/README.md) for installation, GPU selection,
registration methods, output visualization, dependencies and packaging.

# Documentation and command-line use

The [command-line reference](doc/command-line-reference.md) covers all **28 scripts
in `bin/`**, with their purpose, every argument and default, input/output layouts,
and example commands. It includes the full ORG pipeline, registration presets,
atlas building and labeling, preprocessing, merging, measurements and quality
control, plus CPU/joblib/GPU backend selection and the internal Slicer helper.

Run `wm_quality_control_tractography.py --help` to check the CLI installation.

The upstream `whitematteranalysis` documentation can be found at
https://whitematteranalysis.readthedocs.io/en/latest/.
Use the guides above for acceleration and Slicer behavior specific to this fork.

* A master shell script `wm_apply_ORG_atlas_to_subject.sh` (see code [here](bin/wm_apply_ORG_atlas_to_subject.sh)) is provided to apply an anatomically curated white matter atlas ([the ORG atlas](https://dmri.slicer.org/atlases/)), along with the computation tools provided in whitematteranalysis, to perform subject-specific tractography parcellation.

```shell
$ wm_apply_ORG_atlas_to_subject.sh \
  -i input_tractography.vtk \
  -o output_dir \
  -a path_to_atlas/ORG-Atlases-v1.x \
  -s /Applications/Slicer5.2.2.app/Contents/MacOS/Slicer \
  -d 1 \
  -m /Applications/Slicer5.2.2.app/Contents/Extensions-31382/SlicerDMRI/lib/Slicer-5.2/cli-modules/FiberTractMeasurements
```

* A step-by-step tutorial to explain the detailed computational process within the pipeline is provided [here](doc/subject-specific-tractography-parcellation.md).

* The names of the anatomical bundles WMA extracts can be found in the [bundles](doc/bundles.md) page.

# References

**Please cite the following papers for using the code and/or the training data :**
 
    Zhang, F., Wu, Y., Norton, I., Rathi, Y., Makris, N., O'Donnell, LJ. 
    An anatomically curated fiber clustering white matter atlas for consistent white matter tract parcellation across the lifespan. 
    NeuroImage, 2018 (179): 429-447

    O'Donnell LJ, Wells III WM, Golby AJ, Westin CF. 
    Unbiased groupwise registration of white matter tractography.
    In MICCAI, 2012, pp. 123-130.

    O'Donnell, LJ., and Westin, CF. Automatic tractography segmentation
    using a high-dimensional white matter atlas. Medical Imaging,
    IEEE Transactions on 26.11 (2007): 1562-1575.

**For projects using Slicer and SlicerDMRI please also include the following text (or similar) and citations:**

* How to cite the [Slicer platform](https://slicer.readthedocs.io/en/latest/user_guide/about.html#how-to-cite)
* An example of how to cite SlicerDMRI (modify the first part of the sentence according to your use case):

    "We performed diffusion MRI tractography and/or analysis and/or visualization in 3D Slicer (https://www.slicer.org/) via the SlicerDMRI project (https://dmri.slicer.org/) (Norton et al. 2017)."
    
    Fan Zhang, Thomas Noh, Parikshit Juvekar, Sarah F Frisken, Laura Rigolo, Isaiah Norton, Tina Kapur, Sonia Pujol, William Wells III, Alex Yarmarkovich, Gordon Kindlmann, Demian Wassermann, Raul San Jose Estepar, Yogesh Rathi, Ron Kikinis, Hans J Johnson, Carl-Fredrik Westin, Steve Pieper, Alexandra J Golby, Lauren J O’Donnell. 
    SlicerDMRI: Diffusion MRI and Tractography Research Software for Brain Cancer Surgery Planning and Visualization. 
    JCO Clinical Cancer Informatics 4, e299-309, 2020.

    Isaiah Norton, Walid Ibn Essayed, Fan Zhang, Sonia Pujol, Alex Yarmarkovich, Alexandra J. Golby, Gordon Kindlmann, Demian Wassermann, Raul San Jose Estepar, Yogesh Rathi, Steve Pieper, Ron Kikinis, Hans J. Johnson, Carl-Fredrik Westin and Lauren J. O'Donnell. 
    SlicerDMRI: Open Source Diffusion MRI Software for Brain Cancer Research. Cancer Research 77(21), e101-e103, 2017.
