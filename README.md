whitematteranalysis
===================

# Synopsis

WhiteMatterAnalysis (WMA) provides fiber clustering and tractography analysis tools.
This fork adds CUDA GPU acceleration, a vectorized CPU backend, and a five-module
3D Slicer extension. It is based on the [upstream WMA project](https://github.com/SlicerDMRI/whitematteranalysis).

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

# GPU: all CPU dependencies plus CUDA 12.1 / PyTorch 2.4.1
python -m pip install -r requirements_gpu.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`setup.py` reads `requirements_cpu.txt`, so `python -m pip install .` installs
the CPU/base profile by default. The GPU profile includes it using `-r` and adds
the official CUDA wheel index alongside the base package index. Use the two
profiles above for installation.

The GPU profile targets Linux/Windows x86_64 and Python 3.8–3.12 and uses the
[official PyTorch CUDA 12.1 index](https://docs.pytorch.org/get-started/previous-versions/#v241).
GPU execution requires a working NVIDIA GPU/driver. Triton kernels are used when
Triton is available; the torch CUDA path remains available without Triton.
Linux / Python 3.8 and Slicer 5.2.2 / Python 3.9 were tested; Windows has not been tested.

| Setting | Behavior |
|---|---|
| `WMA_GPU=auto` (default) | Use available CUDA; otherwise use accelerated CPU |
| `WMA_GPU=off` | Use CPU |
| `WMA_GPU=off`, `WMA_CPU_FAST=0` | Use the original per-fiber CPU implementation |
| `WMA_GPU_DEVICES=0,1` | Select visible CUDA device indices for supported multi-GPU operations |
| `WMA_GPU_FLOAT32=1` | Opt into float32; default is float64 |

These environment variables apply to the Python/CLI backend. Slicer pages provide
their own Backend selection. Some CLI tools also expose `-gpu auto|on|off`,
`-gpu_float32`, and `-gpu_devices`; check the individual tool's `--help`.
See [GPU development and validation records](doc/gpu/) for tested behavior and measurements.

CPU atlas labeling streams similarity blocks and computes only the required
embedding columns. CPU registration reuses tiled worker pools; GPU registration
fuses distance and probability calculations when Triton is available. See the
[CPU/GPU optimization measurements](doc/gpu/progress-2026-10-14-streaming.md)
for numerical checks, stage timings, and their scope.

GPU labeling keeps normalization and nearest-centroid selection on CUDA, and
affine registration reuses a fused transform when Triton is available. See the
[latest GPU measurements](doc/gpu/progress-2026-10-15-gpu.md).

Geometry preparation avoids full tractogram copies, and CPU distance tiles use
SciPy's compiled kernel where beneficial. See the
[geometry and CPU measurements](doc/gpu/progress-2026-10-15-geometry-cpu.md).

An optional short-vector CUDA kernel also accelerates nearest-centroid distances
without changing float64 results. See the
[centroid optimization measurements](doc/gpu/progress-2026-10-15-centroids.md).

GPU bilateral comparisons share one fused launch; compiled CPU distances read
packed reference coordinates. See the
[bilateral and layout measurements](doc/gpu/progress-2026-10-15-bilateral-layout.md).

The [complete ORG GPU case](doc/gpu/benchmark-org-2026-10-15-optimized.md)
also verifies all cluster/anatomical files against the preceding run.
Independent CLI file stages now honor the pipeline worker count, and VTP writing
uses fast lossless compression. See the [file-stage record](doc/gpu/progress-2026-10-15-file-stages.md).

## 3D Slicer extension

Use the [universal installer](slicer/WMAGPU/dist/WMAGPU-Installer.zip).
Extract it, run `InstallWMAGPU.py` inside the target Slicer's Python Interactor,
and click **Install WMAGPU and required dependencies**. The installer detects
Slicer's version and platform, installs missing scientific dependencies through
Slicer's Python while preserving its NumPy/VTK versions, and installs the matching
extension. This ZIP is not an Extensions Manager "Install from file" archive.
Remove an older WMAGPU release and restart before installing.
See [installation instructions](slicer/WMAGPU/README.md#installation-and-use).

The extension provides Pipeline, moving-to-fixed/atlas registration, groupwise registration, atlas
building, and cluster merge/MRML export. Pipeline includes atlas labeling; uncheck
registration when the input tractography is already registered. It bundles WMA and its pure
Python helpers; CPU mode requires no separate WMA or PyTorch installation.
GPU setup is optional: enable **Allow downloading and installing CUDA PyTorch into Slicer**,
click **Check / Install PyTorch**, and restart after completion. This installs into
Slicer's Python; installing PyTorch in Conda does not enable it in Slicer.

The installer adapts the layout to the detected Slicer version. Actual application
validation is currently Linux Slicer 5.2.2; Windows/macOS and newer Slicer versions
require native validation. macOS/ARM use CPU mode; CUDA setup targets Linux/Windows x86_64. GUI registration supports the CLI presets, rigid, affine, nonrigid, and custom
multiscale schedules, retaining final Slicer Transform nodes. Merge tools offer
checkbox selection with named folders or one combined tractography node. Full ORG
anatomical Pipeline validation requires actual atlas data.
See [extension installation, dependencies, packaging, and tests](slicer/WMAGPU/README.md).

Workflow pages load computation libraries on demand, so opening the extension
does not initialize PyTorch, SciPy, or Matplotlib. Successful environment checks
are reused across pages. Registration reuses unchanged length-filtered inputs,
and scene merging batches hierarchy updates and checklist refreshes. See the
[startup and processing measurements](doc/gpu/progress-2026-10-14.md).

# WMA Installation and Usage
## 1. Install Python 3:

Miniconda is a nice option because it includes pip, setuptools, and all library dependencies (such as VTK and scipy).

  - Download and install Miniconda from https://conda.io/miniconda.html

## 2. Install whitematteranalysis with pip:

The following command will use pip to install whitematteranalysis from this source repository and all library dependencies:

```shell
$ python -m pip install git+https://github.com/yishengpoxiao/whitematteranalysis_gpu.git
```

Note: On macOS, to be able to use `pip`, `X-code` needs to be installed using `$ xcode-select --install`.

Run `$ wm_quality_control_tractography.py --help` to test if the installation is successful.

## 3. Documentation

The upstream `whitematteranalysis` documentation can be found at
https://whitematteranalysis.readthedocs.io/en/latest/.
GPU and Slicer extension behavior specific to this fork is documented above.

* A master shell script `wm_apply_ORG_atlas_to_subject.sh` (see code [here](https://github.com/SlicerDMRI/whitematteranalysis/blob/73a7948751f49d9fda8ec84fb5caeecaeeb92621/bin/wm_apply_ORG_atlas_to_subject.sh#L1-L40)) is provided to apply an anatomically curated white matter atlas ([the ORG atlas](https://dmri.slicer.org/atlases/)), along with the computation tools provided in whitematteranalysis, to perform subject-specific tractography parcellation.

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
