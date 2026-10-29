# Command-line reference

This reference covers all **28 top-level scripts in `bin/`**, including the
internal Slicer transform helper. `bin/tests/` contains tests, not user commands.
Each entry describes the current implementation, its arguments, output and an
example. Defaults below are effective runtime defaults, including defaults set
after argument parsing. Flag spelling and capitalization matter.

## Before running commands

Install this checkout with `python -m pip install .`; see the
[README](../README.md#python--command-line) for CPU/GPU dependencies. Examples
below run from the repository root and use placeholder paths. Replace these with
your files. Installed Python scripts can also be invoked directly, for example
`wm_cluster_from_atlas.py --help`. The shell pipeline calls these installed
commands through `PATH`, so installation is required even when running it from
the checkout.

- Tractography inputs are VTK PolyData `.vtk` or `.vtp`, in **RAS coordinates,
  millimeters**. Most directory arguments scan files directly inside the named
  directory; cohort layouts are identified explicitly below.
- Positional arguments appear in the order shown in each table and example.
  Optional Boolean flags are off unless specified. All Python scripts accept
  `-h` / `--help`. Some expose `-v` / `--version` to print the program version;
  these are listed per command. The downloader's `--version` selects an atlas
  release instead.
- Create parent directories for output files such as CSVs and NIfTI images.
  Commands with output-directory arguments generally create those directories.
  Use fresh output directories when changing inputs/settings: the pipeline and
  transform helper can reuse existing outputs.
- Commands that render images need a working VTK/OpenGL display. Use `-norender`
  where available, or run under `xvfb-run -a` on a suitably configured headless
  Linux system. The shell pipeline's `-x 1` selects its virtual-display path.
- Shell examples use POSIX syntax. The `.sh` pipeline requires Bash and its shell
  utilities; these examples are not native PowerShell commands. Python scripts
  use the same arguments on Windows, with Windows paths and an appropriate
  Python/Slicer installation. See the [Slicer guide](../slicer/WMAGPU/README.md)
  for the extension's platform support.

### Shared acceleration options

Only these five Python commands accept the following flags:
`wm_cluster_atlas.py`, `wm_cluster_from_atlas.py`,
`wm_cluster_remove_outliers.py`, `wm_register_to_atlas_new.py`, and
`wm_register_multisubject_faster.py`. Each relevant entry links here.

| Option | Default | Meaning |
| --- | --- | --- |
| `-gpu {auto,on,off}` | `auto` | `auto`: CUDA when available, otherwise CPU matrix acceleration. `on`: require CUDA. `off`: CPU. |
| `-gpu_devices IDS` | All visible devices, or `WMA_GPU_DEVICES` | Comma-separated visible CUDA indices, for example `0` or `0,1`. Multi-GPU use depends on the operation; single-device operations use the first selected device. |
| `-gpu_float32` | Off (float64) | Opt into float32 GPU distances. This can change near-tie cluster assignments; retain float64 for equivalence comparisons. |

For these Python commands, the parser's `-gpu auto` default overrides `WMA_GPU`,
and the absence of `-gpu_float32` selects float64 even if `WMA_GPU_FLOAT32` is set.
Use explicit command-line flags to select the backend and precision. CUDA indices
are relative to `CUDA_VISIBLE_DEVICES` if that variable restricts visibility.

For example, atlas labeling with the three backends:

```shell
# Original per-fiber joblib backend, four workers
WMA_CPU_FAST=0 python bin/wm_cluster_from_atlas.py \
  data/subject_registered.vtk atlases/ORG-800FC-100HCP work/joblib \
  -j 4 -gpu off -norender

# CPU matrix backend, four-worker budget
WMA_CPU_FAST=1 WMA_CPU_THREADS=4 python bin/wm_cluster_from_atlas.py \
  data/subject_registered.vtk atlases/ORG-800FC-100HCP work/cpu \
  -j 4 -gpu off -norender

# CUDA, first visible GPU, default float64
python bin/wm_cluster_from_atlas.py \
  data/subject_registered.vtk atlases/ORG-800FC-100HCP work/gpu \
  -j 4 -gpu on -gpu_devices 0 -norender
```

`-j` / `--jobs` controls parallelism where a command exposes it; it does not
select a GPU. `WMA_CPU_THREADS` sets the accelerated CPU budget, and
`WMA_ATLAS_CACHE` controls the reusable outlier reference cache. See the
[README settings](../README.md#python--command-line) for other environment options.

## Command index

| Workflow | Scripts |
| --- | --- |
| Full pipeline and atlas download | [wm_apply_ORG_atlas_to_subject.sh](#wm_apply_org_atlas_to_subjectsh), [wm_download_anatomically_curated_atlas.py](#wm_download_anatomically_curated_atlaspy) |
| Registration and transforms | [wm_register_to_atlas_new.py](#wm_register_to_atlas_newpy), [wm_register_multisubject_faster.py](#wm_register_multisubject_fasterpy), [wm_harden_transform.py](#wm_harden_transformpy), [harden_transform_with_slicer.py](#harden_transform_with_slicerpy) |
| Atlas construction and parcellation | [wm_cluster_atlas.py](#wm_cluster_atlaspy), [wm_cluster_from_atlas.py](#wm_cluster_from_atlaspy), [wm_cluster_remove_outliers.py](#wm_cluster_remove_outlierspy), [wm_assess_cluster_location_by_hemisphere.py](#wm_assess_cluster_location_by_hemispherepy), [wm_separate_clusters_by_hemisphere.py](#wm_separate_clusters_by_hemispherepy), [wm_append_clusters_to_anatomical_tracts.py](#wm_append_clusters_to_anatomical_tractspy) |
| Preprocessing, merging and formats | [wm_preprocess_all.py](#wm_preprocess_allpy), [wm_remove_data_along_tracts.py](#wm_remove_data_along_tractspy), [wm_append_clusters.py](#wm_append_clusterspy), [wm_vtp2vtk.py](#wm_vtp2vtkpy), [wm_create_mrml_file.py](#wm_create_mrml_filepy), [wm_change_nrrd_dir.py](#wm_change_nrrd_dirpy) |
| Measurements and volume conversion | [wm_diffusion_measurements.py](#wm_diffusion_measurementspy), [wm_append_diffusion_measures_across_subjects.py](#wm_append_diffusion_measures_across_subjectspy), [wm_average_tract_measures.py](#wm_average_tract_measurespy), [wm_tract_to_volume.py](#wm_tract_to_volumepy), [wm_cluster_volumetric_measurements.py](#wm_cluster_volumetric_measurementspy) |
| Quality control and comparison | [wm_quality_control_tractography.py](#wm_quality_control_tractographypy), [wm_quality_control_after_clustering.py](#wm_quality_control_after_clusteringpy), [wm_quality_control_cluster_measurements.py](#wm_quality_control_cluster_measurementspy), [wm_quality_control_tract_overlap.py](#wm_quality_control_tract_overlappy), [wm_compare_vtks.py](#wm_compare_vtkspy) |

## Full pipeline and atlas download

### wm_apply_ORG_atlas_to_subject.sh

[Source](../bin/wm_apply_ORG_atlas_to_subject.sh). Apply the ORG atlas to one
whole-brain tractogram: registration, labeling, outlier removal, hemisphere
assessment, inverse transformation to subject space, hemisphere separation and
anatomical tract assembly. Diffusion-measure export is optional. Requires
installed WMA commands and Slicer with **SlicerDMRI**.

| Argument | Default / requirement | Meaning |
| --- | --- | --- |
| `-i FILE` | Required | Input subject tractography. |
| `-o DIR` | Required | Output root; a subject subdirectory is derived from the input filename. |
| `-a DIR` | Required | Atlas parent containing `ORG-RegAtlas-100HCP/` and `ORG-800FC-100HCP/`. |
| `-s FILE` | Required | Slicer executable, not its containing directory. |
| `-t FILE` | None | Initial transform into atlas-compatible adult brain size; its inverse is included when returning results to subject space. |
| `-r {rig,nonrig}` | `rig` | `rig` runs the `rigid_affine_fast` preset (rigid **and affine**). `nonrig` runs affine followed by nonrigid registration. |
| `-n N` | `1` | Worker count for supported CPU/file-processing stages. |
| `-g {auto,on,off}` | `WMA_GPU`, otherwise `auto` | Backend selection; also accepts `1` / `0` for on / off. Unlike the Python tools, this shell option reads `WMA_GPU` as its default. |
| `-x {0,1}` | `0` | Use a virtual X display for Slicer/rendering when `1`. Requires the corresponding Xvfb utilities. |
| `-d {0,1}` | `0` | Export diffusion measurements when `1`; requires `-m`. |
| `-m FILE` | None | SlicerDMRI `FiberTractMeasurements` CLI executable for `-d 1`. |
| `-c {0,1,2}` | `0` | Cleanup: `0` retains intermediates; `1` removes initial and transformed bilateral clusters; `2` additionally removes registration temporaries and outlier-removed bilateral clusters. |
| `-h` | Help | Print usage (currently exits with status 1). |

```shell
WMA_GPU_DEVICES=0 bash bin/wm_apply_ORG_atlas_to_subject.sh \
  -i data/subject.vtk -o work/pipeline_gpu \
  -a atlases/ORG-Atlases-1.2 -s /path/to/Slicer \
  -n 4 -g on -r rig -x 1 -c 0

# CPU matrix: use a separate output directory
WMA_CPU_FAST=1 WMA_CPU_THREADS=4 bash bin/wm_apply_ORG_atlas_to_subject.sh \
  -i data/subject.vtk -o work/pipeline_cpu \
  -a atlases/ORG-Atlases-1.2 -s /path/to/Slicer \
  -n 4 -g off -r rig -x 1 -c 0

# Original joblib: same pipeline, separate output directory
WMA_CPU_FAST=0 bash bin/wm_apply_ORG_atlas_to_subject.sh \
  -i data/subject.vtk -o work/pipeline_joblib \
  -a atlases/ORG-Atlases-1.2 -s /path/to/Slicer \
  -n 4 -g off -r rig -x 1 -c 0
```

Outputs under `<output>/<subject>/` include `TractRegistration/`,
`FiberClustering/{InitialClusters,OutlierRemovedClusters,TransformedClusters,SeparatedClusters}/`
and `AnatomicalTracts/`. The current ORG-800 catalogue produces 74 anatomical
tract files. Completion markers can skip completed stages on reruns. Intermediate
VTPs use lossless LZ4 by default; final outputs use zlib. Set
`WMA_VTP_COMPRESSION=zlib` to override intermediate compression.

### wm_download_anatomically_curated_atlas.py

[Source](../bin/wm_download_anatomically_curated_atlas.py). Download and extract a
supported ORG clustering atlas, the registration atlas, population images and
release metadata from the configured Zenodo release. Requires network access.

| Argument | Default / requirement | Meaning |
| --- | --- | --- |
| `outputDirectory` | Required positional | Download root. |
| `-atlas NAME` | Required in practice | `ORG-800FC-100HCP` (anatomically curated) or `ORG-2000FC-100HCP` (finer, uncurated). Supply this explicitly. |
| `--version RELEASE` | `V1_2` | One of `V1_1`, `V1_1_1`, `V1_2`, `V1_3_A`, `V1_3_B`, `V1_4`; these are atlas release identifiers. |

```shell
python bin/wm_download_anatomically_curated_atlas.py atlases \
  -atlas ORG-800FC-100HCP --version V1_2
```

This example creates `atlases/ORG-Atlases-1.2/`. Existing atlas directories and
population files can be skipped; use a fresh destination for a fresh download.

## Registration and transforms

### wm_register_to_atlas_new.py

[Source](../bin/wm_register_to_atlas_new.py). Register **moving tractography to
fixed tractography**. The fixed file can be an atlas or another subject.

| Argument | Default / requirement | Meaning |
| --- | --- | --- |
| `inputSubject` | Required positional | Moving `.vtk` / `.vtp` file. |
| `inputAtlas` | Required positional | Fixed `.vtk` / `.vtp` file, not a directory containing `atlas.p`. |
| `outputDirectory` | Required positional | Registration output root. |
| `-mode MODE` | `affine` | Registration preset; see the complete list below. |
| `-l MM` | `40` | Minimum sampled fiber length. |
| `-lmax MM` | `260` | Maximum sampled fiber length. |
| `-verbose` | Off | Save additional intermediate files and images. |
| Shared GPU flags | See [acceleration options](#shared-acceleration-options) | `-gpu`, `-gpu_devices`, `-gpu_float32`. |

Supported mode names are `affine`, `affine_fast`, `rigid_affine_fast`,
`affine_neonate`, `nonrigid`, `nonrigid_neonate`, `nonrigid_10`, `affineTEST`, and
`nonrigidTEST`. `affine_fast` uses a shorter affine schedule;
`rigid_affine_fast` starts with rigid alignment then refines affine alignment.
Neonate presets use specialized schedules; `nonrigid_10` uses a finer nonrigid
grid schedule. `*TEST` presets are intended for development. Perform linear
alignment before a separate nonrigid run. There is no standalone `rigid` or
`custom` mode in this CLI; those choices belong to the Slicer extension.

```shell
python bin/wm_register_to_atlas_new.py \
  data/moving.vtk data/fixed.vtk work/registration \
  -mode rigid_affine_fast -l 40 -lmax 260 -gpu on -gpu_devices 0
```

Outputs include the registered tractogram under
`work/registration/moving/output_tractography/`, saved ITK/VTK transforms,
iteration directories and `progress.txt`. Keep the transform files to apply the
same mapping to clusters or load a Transform in Slicer.

### wm_register_multisubject_faster.py

[Source](../bin/wm_register_multisubject_faster.py). Build an unbiased group
registration from a directory of subjects; no fixed subject is required.

| Argument | Default / requirement | Meaning |
| --- | --- | --- |
| `inputDirectory` | Required positional | Flat directory of subject tractograms. |
| `outputDirectory` | Required positional | Group registration output directory. |
| `-mode MODE` | `affine` | Preset from the list below. |
| `-f N` | `10000` | Fiber pool per subject; iterations sample from this pool. |
| `-l MM` | `20` | Minimum fiber length. |
| `-lmax MM` | `260` | Maximum fiber length. |
| `-j N` | `1` | CPU worker count. |
| `-verbose` | Off | Save additional intermediate files/images. |
| `-norender` | Off | Disable rendering. |
| `-midsag_symmetric` | Off | Include reflected subject copies to make registration midsagittally symmetric. |
| `-advanced_only_random_seed N` | Unset | Fix the sampling seed for reproducibility. |
| Shared GPU flags | See [acceleration options](#shared-acceleration-options) | `-gpu`, `-gpu_devices`, `-gpu_float32`. |

Supported modes are `affine`, `affine_fast_cheap`, `affine_neonate`, `nonrigid`,
`nonrigid_neonate`, `nonrigid_lowsample`, `affineTEST`, and `nonrigidTEST`.
The fast/low-sample presets reduce sampling or optimization work; neonate presets
use specialized schedules. Run affine before nonrigid. This CLI does not expose
standalone `rigid` or `custom` presets.

```shell
python bin/wm_register_multisubject_faster.py data/subjects work/group \
  -mode affine -f 10000 -j 4 -norender \
  -advanced_only_random_seed 1000 -gpu on -gpu_devices 0
```

Outputs include `registration_atlas.vtk`, `output_tractography/` with registered
subjects and transforms, iteration directories, `registration_performance.txt`,
`input_subjects.txt` and `progress.txt`.

### wm_harden_transform.py

[Source](../bin/wm_harden_transform.py). Apply a saved transform to every
tractography file in a directory using independent Slicer processes. Requires
Slicer with SlicerDMRI; use a working display or virtual display.

| Argument | Default / requirement | Meaning |
| --- | --- | --- |
| `inputDirectory` | Required positional | Flat directory of tractograms/clusters. |
| `outputDirectory` | Required positional | Transformed output directory. |
| `Slicer` | Required positional | Slicer executable. |
| `-t FILE` | Exactly one of `-t` / `-ts` required | One transform for all inputs. |
| `-ts DIR` | Alternative to `-t` | Sorted transform files paired with sorted input tractograms; counts must match. |
| `-i` | Off | Invert the transform before applying it. |
| `-j N` | `1` | Number of independent Slicer workers; a shared transform is batched across workers. |
| `--split-output-directory DIR` | Off | Also write left, right and commissural subsets under this directory, using the same hardened polydata while retaining the complete transformed files in `outputDirectory`. |
| `-v` / `--version` | Version action | Print the program version and exit. |

```shell
python bin/wm_harden_transform.py \
  work/cleaned/subject_registered_outlier_removed work/native /path/to/Slicer \
  -t work/subject_to_atlas.tfm -i -j 4 \
  --split-output-directory work/separated
```

Output filenames match the inputs. Successful runs write
`transform_complete.log`; failed worker logs remain for inspection. Existing
transformed files are reused. With `--split-output-directory`, missing split
files are repaired from an existing transformed file; all three split folders
are checked before the completion marker is written. Omit the option to produce
only the complete transformed files.

### harden_transform_with_slicer.py

[Source](../bin/harden_transform_with_slicer.py). **Internal Slicer Python
helper** used by `wm_harden_transform.py`. It loads fiber bundles, hardens a
transform, writes results and removes its temporary scene nodes. Run inside
Slicer with SlicerDMRI, not ordinary Python. Prefer the wrapper above for normal
batch use.

| Argument | Default / requirement | Meaning |
| --- | --- | --- |
| `polydata` | Required positional | One tractogram file or a flat directory of files. |
| `transform` | Required positional | Transform file. |
| `inverse` | Required positional | String `1` inverts the transform; use `0` for forward application. |
| `outdir` | Required positional | Existing output directory; create it first. |
| `partition_index` | Optional positional, `0` | Zero-based partition index. |
| `partition_count` | Optional positional, `1` | Total partitions; processes sorted files at `[partition_index::partition_count]`. |
| `--split-output-directory DIR` | Off | Also write left, right and commissural subsets from the hardened polydata; create the three subdirectories first. |
| `-v` / `--version` | Version action | Print the program version and exit. |

```shell
mkdir -p work/hardened
mkdir -p work/separated/tracts_left_hemisphere \
  work/separated/tracts_right_hemisphere work/separated/tracts_commissural
/path/to/Slicer --no-main-window \
  --python-script bin/harden_transform_with_slicer.py \
  data/subject.vtk work/subject_to_atlas.tfm 0 work/hardened 0 1 \
  --split-output-directory work/separated \
  --python-code 'slicer.app.quit()'
```

Writes the original filename under `outdir`; existing transformed files are
reused. When split output is requested, missing subsets are generated from the
saved transformed file.

## Atlas construction and parcellation

### wm_cluster_atlas.py

[Source](../bin/wm_cluster_atlas.py). Construct a spectral-clustering atlas from
**already registered** subjects, iterating clustering and outlier removal.

| Argument | Default / requirement | Meaning |
| --- | --- | --- |
| `inputDirectory` | Required positional | Flat directory of aligned whole-brain subject tractograms. |
| `outputDirectory` | Required positional | Atlas output root. |
| `-f N` | `2000` | Fibers sampled per subject; fewer subjects may require a larger sample. |
| `-l MM` | `60` | Minimum fiber length. |
| `-j N` | `1` | CPU worker count. |
| `-k N` | `250` | Number of clusters. |
| `-thresh MM` | `2` | Ignore point separations below this distance in fiber comparisons. |
| `-nystrom_sample N` | `2000` | Nyström reference sample size; the help recommends more than 1500. Must fit within the available sample; larger values use more memory. |
| `-sigma MM` | `60` | Similarity kernel scale. |
| `-mrml_fibers N` | `10000` | Approximate display fiber budget for the subsampled MRML scene; does not limit saved cluster geometry. |
| `-bilateral_off` | Off | Disable bilateral clustering (bilateral clustering is the default). |
| `-cluster_outlier_std X` | `2.0` | Within-cluster outlier rejection threshold in standard deviations; smaller values reject more strictly. |
| `-iter N` | `3` | Clustering/outlier-removal iterations. |
| `-subject_percent X` | `0.4` | Minimum fraction of subjects represented in a retained cluster. |
| `-verbose` | Off | Extra console output. |
| `-norender` | Off | Disable rendered images. |
| `-advanced_only_testing_distance METHOD` | `Mean` | Experimental distance method; help lists `Mean`, `Hausdorff` and `StrictSimilarity`. |
| `-advanced_only_debug_nystrom_off` | Off | Compute a full pairwise matrix for debugging small datasets; no output atlas is created. |
| `-advanced_only_testing_on` | Off | Save additional development/test output. |
| `-advanced_only_random_seed N` | Unset | Fix the clustering seed. |
| `-advanced_only_force_pos_def_off` | Off | Disable replacement of the affinity matrix with a nearby positive-definite matrix. |
| `-advanced_only_outlier_std X` | `4.0` | Global probability-based outlier threshold before clustering. |
| `-advanced_only_outlier_sigma MM` | `20.0` | Local similarity scale for cluster outlier probabilities. |
| `-v` / `--version` | Version action | Print the program version and exit. |
| Shared GPU flags | See [acceleration options](#shared-acceleration-options) | `-gpu`, `-gpu_devices`, `-gpu_float32`. |

```shell
python bin/wm_cluster_atlas.py data/registered_subjects work/atlas \
  -f 10000 -l 60 -k 800 -nystrom_sample 2000 -iter 3 \
  -j 4 -norender -advanced_only_random_seed 1000 -gpu on -gpu_devices 0
```

Each `iteration_XXXXX/initial_clusters/` contains the reusable `atlas.p`,
`atlas.vtp` and numbered cluster files. Use that **initial_clusters directory**
for labeling and outlier removal, not the atlas output root or a directory of
outlier-removed reference clusters.

### wm_cluster_from_atlas.py

[Source](../bin/wm_cluster_from_atlas.py). Label a subject's fibers using an
existing clustering atlas. Register the subject to the atlas coordinate system
first. This step assigns clusters; outlier removal is a separate command.

| Argument | Default / requirement | Meaning |
| --- | --- | --- |
| `inputFile` | Required positional | Registered subject tractography. |
| `atlasDirectory` | Required positional | Exact directory containing `atlas.p` and `atlas.vtp`, e.g. `ORG-800FC-100HCP/` or an atlas iteration's `initial_clusters/`. |
| `outputDirectory` | Required positional | Output root; a subject subdirectory is added. |
| `-f N` | All fibers | Optional fiber subsample. |
| `-l MM` | `40` | Minimum fiber length. The legacy help text says 60, but the code defaults to 40. |
| `-j N` | `1` | CPU worker count. |
| `-mrml_fibers N` | `10000` | Approximate display budget for the MRML scene, not saved cluster size. |
| `-verbose` | Off | Extra console output. |
| `-reg` | Off; unsupported | Legacy atlas-to-subject registration switch. The current implementation prints a warning and exits; register separately and omit this flag. |
| `-norender` | Off | Disable rendered cluster images. |
| `-v` / `--version` | Version action | Print the program version and exit. |
| Shared GPU flags | See [acceleration options](#shared-acceleration-options) | `-gpu`, `-gpu_devices`, `-gpu_float32`. |

```shell
python bin/wm_cluster_from_atlas.py data/subject_registered.vtk \
  atlases/ORG-800FC-100HCP work/clusters \
  -l 40 -j 4 -norender -gpu on -gpu_devices 0
```

Outputs under `work/clusters/subject_registered/` include numbered
`cluster_00001.vtp`, etc., MRML scene files and clustering information. Atlas
labels determine the file count; some subject clusters may contain no fibers.

### wm_cluster_remove_outliers.py

[Source](../bin/wm_cluster_remove_outliers.py). Remove low-probability subject
fibers using the matching atlas cluster distributions. Independent clusters can
be processed concurrently; accelerated atlas statistics are cached for reuse.
Atlas and subject `cluster_*.vtp` filename sets must match exactly. Missing,
unexpected or absent atlas clusters cause an error before any output is written;
matching file counts alone are insufficient.

| Argument | Default / requirement | Meaning |
| --- | --- | --- |
| `inputDirectory` | Required positional | One subject's flat cluster directory; avoid a trailing path separator. |
| `atlasDirectory` | Required positional | Matching reference atlas with `atlas.p`, `atlas.vtp` and nonempty reference cluster VTPs; use `initial_clusters/` for a locally built atlas. |
| `outputDirectory` | Required positional | Output root. |
| `-cluster_outlier_std X` | `2.0` | Rejection threshold in standard deviations; smaller values reject more strictly. |
| `-advanced_only_outlier_sigma MM` | `20.0` | Local probability kernel scale. |
| `-j N` | `1` | Parallel worker budget for cluster processing. |
| `-verbose` | Off | Extra console output. |
| `-v` / `--version` | Version action | Print the program version and exit. |
| Shared GPU flags | See [acceleration options](#shared-acceleration-options) | `-gpu`, `-gpu_devices`, `-gpu_float32`. |

```shell
python bin/wm_cluster_remove_outliers.py work/clusters/subject_registered \
  atlases/ORG-800FC-100HCP work/cleaned \
  -cluster_outlier_std 2 -j 4 -gpu on -gpu_devices 0
```

Outputs are under `work/cleaned/subject_registered_outlier_removed/`, retaining
cluster filenames, with `cluster_log.txt` and copied MRML files. Empty subject
clusters do not require distance computation; a progress count of processed
nonempty clusters can therefore be below the atlas's 800 cluster files.

### wm_assess_cluster_location_by_hemisphere.py

[Source](../bin/wm_assess_cluster_location_by_hemisphere.py). Assign per-fiber
left, right or commissural labels and store them as point data
`HemisphereLocation` (1 = left, 2 = right, 3 = commissural). Run **in atlas space**,
where the midsagittal plane is at RAS x = 0, before inverse-transforming clusters.

| Argument | Default / requirement | Meaning |
| --- | --- | --- |
| `inputDirectory` | Required positional | One subject's flat `cluster_*` directory in atlas space. |
| `-pthresh X` | `0.6` | Fraction of a fiber required in one hemisphere; valid range is `0.5 < X <= 1`. Superseded by atlas location annotations when supplied. |
| `-clusterLocationFile FILE` | None | Atlas cluster hemisphere annotations, e.g. ORG `cluster_hemisphere_location.txt`; adapts the threshold by cluster. |
| `-outputDirectory DIR` | No separate outputs | Optionally also write hemisphere-separated clusters into three subdirectories here. **Input cluster files are updated in place whether or not this option is given.** |
| `-j N` / `--jobs N` | `1` | Independent cluster-file workers. |
| `-v` / `--version` | Version action | Print the program version and exit. |

```shell
python bin/wm_assess_cluster_location_by_hemisphere.py \
  work/cleaned/subject_registered_outlier_removed \
  -clusterLocationFile atlases/ORG-800FC-100HCP/cluster_hemisphere_location.txt \
  -j 4
```

Existing `HemisphereLocation` or legacy `HemisphereLocataion` data are retained.
Copy the input cluster directory first if you need to preserve unlabeled files.
Adding `-outputDirectory work/separated_atlas` also writes
`tracts_left_hemisphere/`, `tracts_right_hemisphere/` and `tracts_commissural/`
in atlas space. After labeling, inverse-transform the updated input files to
subject space if needed; the saved labels allow hemisphere separation there
without recomputing the midline.

### wm_separate_clusters_by_hemisphere.py

[Source](../bin/wm_separate_clusters_by_hemisphere.py). Split labeled clusters
into left, right and commissural outputs. Requires `HemisphereLocation` (or the
legacy spelling `HemisphereLocataion`); this command does not infer those labels.

| Argument | Default / requirement | Meaning |
| --- | --- | --- |
| `inputDirectory` | Required positional | Flat directory of hemisphere-labeled clusters. |
| `outputDirectory` | Required positional | Root for the three separated directories. |
| `-j N` / `--jobs N` | `1` | Independent cluster-file workers. |
| `--reuse-complete` | Off | If every expected split filename already exists in all three folders, write the completion marker and skip recomputation. |
| `-v` / `--version` | Version action | Print the program version and exit. |

```shell
python bin/wm_separate_clusters_by_hemisphere.py work/native work/separated -j 4 --reuse-complete
```

Creates `tracts_left_hemisphere/`, `tracts_right_hemisphere/` and
`tracts_commissural/`, retaining indexed cluster files, including empty splits.
Use `--reuse-complete` to reuse a full set of existing split files or to repair
the completion marker after verifying them; incomplete sets are recomputed.

### wm_append_clusters_to_anatomical_tracts.py

[Source](../bin/wm_append_clusters_to_anatomical_tracts.py). Assemble the ORG
anatomical tract definitions from hemisphere-separated cluster files.

| Argument | Default / requirement | Meaning |
| --- | --- | --- |
| `inputDirectory` | Required positional | Parent containing `tracts_left_hemisphere/`, `tracts_right_hemisphere/` and `tracts_commissural/`. |
| `atlasDirectory` | Required positional | ORG-800 atlas directory containing anatomical `T_*.mrml` definitions. |
| `outputDirectory` | Required positional | Anatomical tract output directory. |
| `-j N` / `--jobs N` | `1` | Independent anatomical-tract workers. |

```shell
python bin/wm_append_clusters_to_anatomical_tracts.py work/separated \
  atlases/ORG-800FC-100HCP work/anatomical -j 4
```

The current catalogue yields 74 tract VTPs: 33 left/right pairs and 8 commissural
tracts. See the [bundle catalogue](bundles.md) for anatomical names.

## Preprocessing, merging and formats

### wm_preprocess_all.py

[Source](../bin/wm_preprocess_all.py). Filter fiber lengths and/or downsample each
tractogram in a flat input directory.

| Argument | Default / requirement | Meaning |
| --- | --- | --- |
| `inputDirectory` | Required positional | Subject tractograms. |
| `outputDirectory` | Required positional | Preprocessed output directory. |
| `-f N` | No downsampling | Maximum fibers retained per subject. |
| `-l MM` | No lower filter | Minimum fiber length. |
| `-lmax MM` | No upper filter | Maximum fiber length. |
| `-j N` | All detected CPU cores | Joblib worker count; set explicitly to limit process/memory use. |
| `-retaindata` | Off | Retain point/cell data during filtering and downsampling; these operations otherwise discard it. |
| `--nonidentical` | Off | Use nonidentical random sampling across runs; otherwise downsampling uses seed 1234. |

```shell
python bin/wm_preprocess_all.py data/subjects work/preprocessed \
  -f 10000 -l 40 -lmax 260 -j 4 -retaindata
```

Outputs `<subject>_pp.vtk` or `<subject>_pp.vtp`, retaining the input extension.

### wm_remove_data_along_tracts.py

[Source](../bin/wm_remove_data_along_tracts.py). Remove point and cell data arrays
from tractography while keeping all fibers and their geometry. Despite legacy
help wording, this command does not expose length filtering or downsampling.

| Argument | Default / requirement | Meaning |
| --- | --- | --- |
| `inputDirectory` | Required positional | Flat directory of tractography files. |
| `outputDirectory` | Required positional | Geometry-only output directory. |
| `-j N` | All detected CPU cores | Joblib worker count. |

```shell
python bin/wm_remove_data_along_tracts.py data/subjects work/geometry_only -j 4
```

Writes the original filenames under the output directory. Preserve the original
files if downstream diffusion measurements need their scalar/tensor arrays.

### wm_append_clusters.py

[Source](../bin/wm_append_clusters.py). Merge selected cluster geometries into
one tractography file, using either cluster numbers or an MRML definition.

| Argument | Default / requirement | Meaning |
| --- | --- | --- |
| `inputDirectory` | Required positional | Flat directory of input cluster files. |
| `outputDirectory` | Required positional | Merged output directory. |
| `-appendedTractName NAME` | `appended_tract` | Output basename without extension; writes `.vtp`. |
| `-clusterList N [N ...]` | All files if neither selector is given | One-based cluster file indices, e.g. `1 2 3` for `cluster_00001.vtp`, etc. |
| `-tractMRML FILE` | None | Select clusters referenced by this MRML file. Cannot be combined with `-clusterList`. |

```shell
python bin/wm_append_clusters.py work/native work/merged \
  -clusterList 1 2 3 -appendedTractName selected_bundle
```

Writes `work/merged/selected_bundle.vtp`. Added point data `cluster_idx` records
the zero-based position in the selected input list, not the original atlas label.
This merges geometry; for grouping existing Slicer nodes in a named folder, use
the extension's merge workflow.

### wm_vtp2vtk.py

[Source](../bin/wm_vtp2vtk.py). Convert VTP tractograms in a flat directory to
legacy VTK files.

| Argument | Requirement | Meaning |
| --- | --- | --- |
| `inputDirectory` | Required positional | Directory of `.vtp` files. |
| `outputDirectory` | Required positional | Output root. |

```shell
python bin/wm_vtp2vtk.py work/anatomical work/legacy
```

Writes `work/legacy/anatomical_vtk/<tract>.vtk`. No additional options beyond
`-h` / `--help`.

### wm_create_mrml_file.py

[Source](../bin/wm_create_mrml_file.py). Create a color-coded Slicer scene
referencing all tractography files in one directory; it does not merge geometry.

| Argument | Requirement | Meaning |
| --- | --- | --- |
| `inputDirectory` | Required positional | Flat tractography directory; the current color-generation code requires **at least two files**. |

```shell
python bin/wm_create_mrml_file.py work/anatomical
```

Writes `work/anatomical/scene.mrml`. Keep the referenced tractography files
available when opening the scene in Slicer. No additional options beyond help.

### wm_change_nrrd_dir.py

[Source](../bin/wm_change_nrrd_dir.py). Flip one component of each diffusion
gradient direction in a detached NRRD header. This changes `DWMRI_gradient_*`
header entries, not voxel data or the image's spatial affine.

| Argument | Default / requirement | Meaning |
| --- | --- | --- |
| `inputnrrd` | Required positional | Input `.nhdr` header. |
| `outputnrrd` | Required positional | New `.nhdr` header. |
| `-d AXIS` | Supply explicitly | `x`, `y` or `z`: gradient component to negate. |

```shell
python bin/wm_change_nrrd_dir.py data/dwi.nhdr data/dwi_flip_z.nhdr -d z
```

Other header lines are copied. If the header references a relative raw-data
filename, keep the new header where that reference still resolves.

## Measurements and volume conversion

### wm_diffusion_measurements.py

[Source](../bin/wm_diffusion_measurements.py). Export measurements from scalar or
tensor arrays already stored on tractography using SlicerDMRI's
`FiberTractMeasurements` CLI. It does not estimate diffusion measures from raw DWI.

| Argument | Requirement | Meaning |
| --- | --- | --- |
| `inputDirectory` | Required positional | Flat directory of fiber files to measure. |
| `outputCSV` | Required positional | Output CSV **file path**, despite the legacy help describing a directory. |
| `Slicer` | Required positional | Complete command string launching `FiberTractMeasurements`, despite the legacy help describing only the Slicer path. |
| `-v` / `--version` | Version action | Print the program version and exit. |

```shell
python bin/wm_diffusion_measurements.py work/anatomical \
  work/measures/anatomical.csv \
  "/path/to/Slicer --launch /path/to/FiberTractMeasurements"
```

The wrapper requests `Fibers_File_Folder`, `Column_Hierarchy` and comma-separated
output. Its legacy shell-string construction requires command/input/output paths
without spaces. For automatic pipeline exports, prefer the full pipeline's
`-d 1 -m /path/to/FiberTractMeasurements` options.

### wm_append_diffusion_measures_across_subjects.py

[Source](../bin/wm_append_diffusion_measures_across_subjects.py). Aggregate
per-subject ORG pipeline measurements into wide cohort CSV tables.

| Argument | Requirement | Meaning |
| --- | --- | --- |
| `inputfolder` | Required positional | Cohort root with one pipeline output directory per subject. |
| `appenedmeasurefile` | Required positional | Output base filename ending in `.csv`; spelling reflects the current parser. |

Each subject directory must contain
`AnatomicalTracts/diffusion_measurements_anatomical_tracts.csv` and the three
`FiberClustering/SeparatedClusters/diffusion_measurements_{left_hemisphere,right_hemisphere,commissural}.csv`
files produced by pipeline measurement export. Inputs must have matching tract
order/columns and the expected Slicer comma-separated format. The aggregator is
specific to the ORG catalogue, not an arbitrary CSV concatenation tool.

```shell
python bin/wm_append_diffusion_measures_across_subjects.py \
  work/cohort work/measures/cohort.csv
```

Writes `cohort_anatomical_tracts.csv` and `cohort_fiber_clusters.csv` in
`work/measures/`, with one row per subject. No additional options beyond help.

### wm_average_tract_measures.py

[Source](../bin/wm_average_tract_measures.py). Combine named tract measurements
in a wide cohort table. Sums fiber/point counts, weights mean length by fiber
count and other mean measures by point count.

| Argument | Default / requirement | Meaning |
| --- | --- | --- |
| `inputmeasure` | Required positional | Wide comma-separated cohort CSV, such as the anatomical output of the aggregator above. |
| `outoutmeasure` | Required positional | Output CSV file; spelling reflects the current parser. |
| `-tractList NAME [NAME ...]` | Required in practice | Tract prefixes matching input columns such as `<tract>.Num_Points` and `<tract>.Num_Fibers`. |
| `-appendedTractName NAME` | Empty string | Prefix for combined measurement columns; provide a descriptive name. |

```shell
python bin/wm_average_tract_measures.py \
  work/measures/cohort_anatomical_tracts.csv work/measures/AF_bilateral.csv \
  -tractList T_AF_left T_AF_right -appendedTractName AF_bilateral
```

Match tract prefixes to your actual CSV headers. The current implementation
combines count/mean fields, not standard deviations or extrema. It expects
`-tractList` even though argparse marks that option as optional.

### wm_tract_to_volume.py

[Source](../bin/wm_tract_to_volume.py). Rasterize tractography into a reference
NIfTI grid, either as a density-like point-count map or as mean point-scalar values.
Tractography and the reference image must share physical coordinates.

| Argument | Default / requirement | Meaning |
| --- | --- | --- |
| `inputVTK` | Required positional | Input `.vtk` / `.vtp` tract or cluster. |
| `refvolume` | Required positional | Reference NIfTI image supplying voxel grid and affine. |
| `outputVol` | Required positional | Output NIfTI filename, e.g. `.nii.gz`. |
| `-m NAME` | None | Point-data array to average in each visited voxel; the first component is used. Without it, count stored tract points per voxel. |
| `-v` / `--version` | Version action | Print the program version and exit. |

```shell
python bin/wm_tract_to_volume.py work/anatomical/T_AF_left.vtp \
  data/FA.nii.gz work/AF_left_point_density.nii.gz

python bin/wm_tract_to_volume.py work/anatomical/T_AF_left.vtp \
  data/FA.nii.gz work/AF_left_mean_FA.nii.gz -m FA
```

Points map to the nearest voxel using the inverse reference affine; points outside
the volume are skipped. The default counts stored points, **not unique fibers**
per voxel, so values depend on streamline point sampling. The script's separate
resampling helper is not used by its main command.

### wm_cluster_volumetric_measurements.py

[Source](../bin/wm_cluster_volumetric_measurements.py). For each tract/cluster,
measure occupied voxels, physical volume and statistics from a scalar NIfTI map.
Inputs must share physical coordinates, with tract points inside the image grid.

| Argument | Default / requirement | Meaning |
| --- | --- | --- |
| `inputDirectory` | Required positional | Flat directory of tract/cluster files. |
| `inputVolumetricMap` | Required positional | Scalar `.nii` / `.nii.gz` map, e.g. FA. |
| `outputDirectory` | Required positional | Measurement output directory. |
| `outputStatFile` | Required positional | Statistics filename within that directory. |
| `-sampleSize MM` | No resampling | Spatial distance for tract resampling before voxelization; this is not a fiber count. |
| `-outputLabelmap` | Off | Also save an occupied-voxel labelmap per tract/cluster. |
| `-v` / `--version` | Version action | Print the program version and exit. |

```shell
python bin/wm_cluster_volumetric_measurements.py work/anatomical \
  data/FA.nii.gz work/volumetric tract_stats.csv \
  -sampleSize 0.5 -outputLabelmap
```

Writes `tract_stats.csv` with `Name`, `Num_Voxels`, `Volume`, `Mean`, `Median`,
`Min`, `Max` and `Variance`, plus optional labelmaps. This legacy utility still
calls NiBabel `get_data()`; it requires a NiBabel installation supporting that API.
Its help can be inspected independently of running the volume calculations.

## Quality control and comparison

### wm_quality_control_tractography.py

[Source](../bin/wm_quality_control_tractography.py). Inspect whole-brain
tractography: fiber counts/lengths, step sizes, spatial extents, stored arrays and
rendered views. Requires a working rendering display.

| Argument | Requirement | Meaning |
| --- | --- | --- |
| `inputDirectory` | Required positional | Flat directory of subject whole-brain tractograms. |
| `outputDirectory` | Required positional | QC report directory. |

```shell
python bin/wm_quality_control_tractography.py data/subjects work/qc_tractography
```

Outputs include `index.html`, `README.txt`, `quality_control_data.txt`, spatial
summaries, histograms and per-subject rendered views. No worker/backend or
`-norender` option is exposed; rendering is part of this command.

### wm_quality_control_after_clustering.py

[Source](../bin/wm_quality_control_after_clustering.py). Compare cluster counts
and occupancy across a cohort of clustered subjects.

| Argument | Requirement | Meaning |
| --- | --- | --- |
| `inputDirectory` | Required positional | Cohort directory whose direct subdirectories each contain one subject's indexed cluster files. |
| `outputDirectory` | Required positional | QC output directory. |
| `-v` / `--version` | Version action | Print the program version and exit. |

```shell
python bin/wm_quality_control_after_clustering.py \
  work/cohort_clusters work/qc_clusters
```

Writes `cluster_quality_control.txt` and `subjects_per_cluster_hist.pdf`. This
command uses a noninteractive plotting backend; it does not render tractography.

### wm_quality_control_cluster_measurements.py

[Source](../bin/wm_quality_control_cluster_measurements.py). Check per-subject
cluster measurement tables for matching structure and flag unusual fiber counts
or numbers of empty clusters.

| Argument | Default / requirement | Meaning |
| --- | --- | --- |
| `inputDirectory` | Required positional | Directory of matching per-subject measurement tables. The current loader uses Slicer `Column` hierarchy with **tab** separators. |
| `-outlier_std X` | `2.0` | Standard-deviation threshold for subject outlier detection. |

```shell
python bin/wm_quality_control_cluster_measurements.py \
  data/cluster_measurements_tab -outlier_std 2
```

Prints QC findings to the terminal; redirect stdout to retain them. Despite help
mentioning `.csv` / `.txt`, this is not a loader for the wide comma-separated
cohort table from `wm_append_diffusion_measures_across_subjects.py`.

### wm_quality_control_tract_overlap.py

[Source](../bin/wm_quality_control_tract_overlap.py). Render two tracts together
in different colors for visual overlap inspection. They must already share a
coordinate system. Requires a working rendering display.

| Argument | Requirement | Meaning |
| --- | --- | --- |
| `inputTract1` | Required positional | First tract `.vtk` / `.vtp`. |
| `inputTract2` | Required positional | Second tract `.vtk` / `.vtp`. |
| `outputDirectory` | Required positional | Rendered overlap output directory. |

```shell
python bin/wm_quality_control_tract_overlap.py \
  data/tract_reference.vtp data/tract_candidate.vtp work/qc_overlap
```

Produces six anatomical views, after internal length filtering/downsampling for
visualization. It does not compute a numerical overlap score. No additional
options beyond help.

### wm_compare_vtks.py

[Source](../bin/wm_compare_vtks.py). Print coordinate displacement statistics
between corresponding fibers, resampling each fiber to 30 points.

| Argument | Requirement | Meaning |
| --- | --- | --- |
| `inputFile1` | Required positional | Reference tractogram. |
| `inputFile2` | Required positional | Comparison tractogram, with matching fiber count and order. |

```shell
python bin/wm_compare_vtks.py data/reference.vtk data/candidate.vtk
```

Prints minimum, maximum, mean, standard deviation and median displacement, plus
percentages below several distance thresholds. This is a resampled-coordinate
comparison, not a byte, topology, attribute or cluster-membership equivalence
check. No additional options beyond help.
