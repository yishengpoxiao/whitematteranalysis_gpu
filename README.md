# WhiteMatterAnalysis

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
| `WMA_GPU=off` | Use CPU |
| `WMA_GPU=off`, `WMA_CPU_FAST=0` | Use the original per-fiber CPU implementation |
| `WMA_GPU_DEVICES=0,1` | Select visible CUDA device indices for supported multi-GPU operations |
| `WMA_GPU_FLOAT32=1` | Opt into float32; default is float64 |
| `WMA_ATLAS_CACHE=/path/to/cache` | Override the automatic outlier atlas cache; `off` disables it |

These environment variables apply to the Python/CLI backend. Slicer pages provide
their own Backend selection. Some CLI tools also expose `-gpu auto|on|off`,
`-gpu_float32`, and `-gpu_devices`; check the individual tool's `--help`.

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
