# WhiteMatterAnalysis

WhiteMatterAnalysis (WMA) provides fiber clustering and tractography analysis tools.
This fork adds CUDA GPU acceleration, a vectorized CPU backend, and a five-module
3D Slicer extension. It is based on the [upstream WMA project](https://github.com/SlicerDMRI/whitematteranalysis).

## Full-pipeline performance

On **2026-10-28**, the C028 whole-brain input of approximately
**330,000 streamlines (3.6 GB)** was processed using source revision `e5abd52`
through `wm_apply_ORG_atlas_to_subject.sh` with ORG-800FC-100HCP and
ORG-RegAtlas-100HCP. All three modes use the same updated source, four physical
CPU cores, `-n 4 -r rig -x 1 -c 0`, standard sampling seed 1000, and float64.
Hardware: Intel Xeon Gold 6330 and one NVIDIA RTX 4090; Linux, Python 3.8,
PyTorch 2.4.1/CUDA 12.1 and Slicer 5.2.2 with SlicerDMRI. CUDA 12.1 is the
benchmark environment, not a WMA requirement.

Total runtime: original joblib **31.15 min**, CPU matrix **6.08 min** (**5.12x** speedup), and GPU **6.11 min** (**5.09x** speedup).

| Step | Joblib (4 workers), s | CPU matrix, s | CPU speedup | GPU, s | GPU speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Registration (rigid + affine) | 391.5 | 94.5 | 4.14x | 163.8 | 2.39x |
| Atlas labeling and cluster saving | 1159.3 | 96.5 | 12.01x | 73.2 | 15.84x |
| Outlier removal | 223.7 | 95.7 | 2.34x | 31.9 | 7.01x |
| Hemisphere assessment | 15.7 | 18.8 | 0.84x | 14.1 | 1.11x |
| Inverse transform with Slicer | 51.8 | 33.4 | 1.55x | 55.0 | 0.94x |
| Hemisphere separation | 9.0 | 9.2 | 0.98x | 9.0 | 1.01x |
| Anatomical tract assembly | 15.7 | 14.9 | 1.06x | 16.5 | 0.95x |
| **Total** | **1868.9** | **364.7** | **5.12x** | **366.9** | **5.09x** |

Speedup is joblib time divided by the corresponding accelerated time. Original
joblib means the original per-fiber mathematical backend within the current
pipeline; common file/I/O improvements are shared by all modes. Four workers
apply where the original algorithm supports parallelism. Each mode starts in a
fresh output directory; accelerated atlas-reference caches start empty. Runs
were sequential: joblib, CPU matrix, then GPU. CPU affinity was cores 0–3,
with four WMA workers and one thread per BLAS/OpenMP library; the GPU profile
used physical device 4 with no competing CUDA process observed on that device.

These are single complete runs on a shared workstation, including reading,
writing, process startup and Slicer transform hardening. Filesystem/JIT caches
and workstation load are not reset; file/scene stages remain CPU operations.
Total time includes shell overhead and can differ slightly from the stage sum.
The final four stages use the same CPU/Slicer implementations in every profile;
their per-run timing differences are not measurements of different mathematical
backends. See the [acceleration overview](doc/acceleration.md) for the changes
behind the registration, labeling and outlier-removal speedups.
Interactive display and optional diffusion-measure export are excluded.
All runs produce 800 initial, outlier-removed and transformed cluster files plus
74 anatomical tracts, with identical counts of 245,946 initial fibers, 130,737
retained fibers and 93,766 fibers across the anatomical outputs.
Validation compares every cluster/tract VTP, all hemisphere
splits, the full registered tractogram and all 11 saved ITK transforms against
this joblib run:

- **CPU matrix:** all 4,874 VTP files and the registered VTK are byte-identical;
  all 11 transform parameter sets match exactly.
- **GPU:** cluster membership, counts, topology, active data attributes and all
  point/cell/field data values match exactly. Of 4,874 VTPs, 4,794 are
  byte-identical; the remaining 80 differ only in coordinates. Maximum coordinate
  difference is **1.91e-6 mm**, and maximum saved transform-parameter difference
  is **3.30e-13**. The full registered VTK also preserves connectivity and
  original attribute bytes exactly, with a maximum coordinate difference of
  **4.77e-7 mm**. These floating differences pass
  `rtol=1e-6, atol=1e-5` for coordinates and `rtol=1e-10, atol=1e-10` for transforms.

The GPU registration and labeling stages were slower in this complete run than
in the October 15 run (39.1 s and 30.7 s). Subsequent instrumented runs of the
same current code measured registration at **55.9 s and 36.2 s** and labeling at
**32.9 s and 32.8 s**, using the pipeline's LZ4 output compression. Labeling
computation itself took **3.8 s and 3.7 s**; the rest includes imports, input
reading, filtering, splitting and writing 800 clusters. A separate repeated-read
control spent **125.4 s versus 2.5 s** reading the same registered VTK. Registration
profiling also showed atlas CPU preprocessing varying from **19.4 s to 5.0 s**,
while the ten GPU optimizer calls totaled about 3–4 s. Delaying the CUDA availability
check did not consistently improve registration time.

These controls found substantial host-side I/O and preparation variability, with
no repeatable GPU computation regression. The original complete run did not log
per-function durations, so its extra time cannot be assigned precisely after
the fact. The table retains the actual complete-run measurements; isolated
repeats are not substituted into stage times or totals. In this run CPU matrix
and GPU total times were effectively tied.

An additional correctness audit sampled **2,048 C028 fibers** using the current
code and the trained ORG atlas settings. CPU matrix and GPU float64 labels both
matched joblib exactly; maximum embedding differences were `3.11e-15` and
`1.78e-14`, respectively. Across 16 distance-method/bilateral/threshold settings
on real fibers, CPU distances matched upstream exactly and the maximum GPU
absolute difference was `7.28e-12`. The sampled affine objective matched upstream
exactly for both backends. These are focused checks, not new full-pipeline runs;
see [numerical behavior](doc/acceleration.md#numerical-behavior) for validation
scope and tolerances.

### Final-stage scheduling check

An earlier, separate control reused identical saved C028 intermediates (800 clusters,
about 131,000 retained fibers) to compare one worker with four workers. Values
below are the median of two runs per setting, with execution order reversed,
the same four-core affinity and fresh outputs. Hemisphere labels were removed
from input copies before assessment so that the calculation ran again. Input
preparation and output validation are outside the timed commands; process
startup and stage I/O are included. Filesystem caches were not flushed.

| Step | One worker, s | Four workers, s |
| --- | ---: | ---: |
| Hemisphere assessment | 37.4 | 14.5 |
| Inverse transform with Slicer | 86.9 | 29.8 |
| Hemisphere separation | 31.6 | 9.1 |
| Anatomical tract assembly | 48.8 | 15.2 |

Four workers were faster in every stage, so the current parallel implementation
is retained. A four-thread control was also slower for the three Python file
stages; inverse transformation already launches independent Slicer processes.
All VTP outputs in the 22 stage runs were byte-identical to the corresponding saved
joblib results. The one-worker control uses the current geometry/I/O functions
with serial scheduling; it is not a benchmark of an unmodified upstream checkout.
These are isolated-stage measurements, not replacement full-pipeline totals.

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
