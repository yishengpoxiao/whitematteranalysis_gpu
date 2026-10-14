# GPU-resident labeling and affine transforms — 2026-10-15

## Changes

- Keep row normalization and nearest-centroid classification on CUDA. Use
  float64 `torch.cdist` with direct coordinate differences, explicitly disabling
  the squared-norm GEMM shortcut. Choose the first centroid on equal distances,
  as SciPy vq does. Upload atlas centroids once per labeling invocation.
- Replace per-column embedding divisions with one broadcast operation. Transfer
  embedding coordinates, row/column sums, labels, and distances to the host in
  one compact copy per block. Remove the row-sum upload and CPU classification
  worker. Release the normalized similarity block before centroid distances.
- Preserve negative-row warnings and clamping. Validate finite centroids and
  embeddings, retaining SciPy's rejection of NaN/Inf input instead of silently
  returning a cluster number. Input similarity tensors remain unmodified.
- Apply affine transforms in one optional Triton kernel, writing the contiguous
  coordinate layout consumed by the registration objective. Preserve VTK matrix
  construction and separate float64 multiply/add rounding. Reuse a pinned host
  matrix, its CUDA copy, and transformed coordinates for one optimizer call.
  Evaluations are sequential and return host scalars before reusing buffers.
- Keep the existing PyTorch affine transform when Triton is unavailable or an
  explicit CPU device is used. GPU classification uses ordinary PyTorch CUDA
  operations and does not require Triton. No dependencies/configuration changes.

## Measurements

Compare with the preceding optimization, commit `cffed63`, on one RTX 4090,
Xeon Gold 6330, four pinned physical CPU cores, one BLAS thread, PyTorch
2.4.1+cu121 and Triton 3.0.0. Precision remains float64.

| Operation | Previous | Current |
| --- | ---: | ---: |
| Classification, 8192 real fibers, median of seven | 0.1250 s | 0.0913 s |
| Classification, all 100,000 real fibers, median of three | 1.5228 s | 1.0619 s |
| Registration objective, 1000 fixed/250 moving, median of 100 | 0.7208 ms | 0.4431 ms |
| Registration objective, 5000 fixed/2000 moving, median of 50 | 4.2776 ms | 4.0158 ms |
| 30-evaluation COBYLA, 1000/250, median of three including setup | 0.02318 s | 0.01539 s |
| 30-evaluation COBYLA, 5000/2000, median of three including setup | 0.13125 s | 0.12240 s |

Classification uses the actual ORG atlas with 2500 reference fibers, 800
centroids, and 11 projection columns. It includes resampling, fiber similarities,
normalization, embedding, and labeling. Input/atlas file reading, registration,
length filtering, saving, and Slicer scene display are excluded. The full-subject
classification is approximately 1.43x faster than the preceding optimized path.
Labels and serialized cell attributes match across both implementations for all
100,000 fibers. Floating embedding values agree within 1e-12.

Registration uses deterministic synthetic fibers with 10 points per fiber.
Small objectives are approximately 1.63x faster, while large objectives improve
by about 6.5%. The 30-evaluation optimizer improves by approximately 1.51x and
1.07x respectively. Objective sequences and final transforms match exactly.

A separate 100,000-row random-similarity embedding-only test changes from
0.3324 s to 0.3230 s; moving stages to CUDA is not itself a guarantee of a large
speedup. Warm timings exclude JIT compilation. Workstation load and caches are
shared. These measurements do not establish a full ORG-pipeline speedup or native
Windows/macOS performance. The complete clinical CLI pipeline was not rerun.

## Validation and reproduction

Targeted tests compare all four distance methods and both precision options,
bilateral comparisons, partial/empty blocks, negative row handling, duplicate
centroid ties, NaN/Inf rejection, input preservation, CUDA device selection,
optimizer refresh/failure cleanup, and optional Triton fallback. The fused affine
transform matches the original VTK coordinates exactly for several point counts,
strided inputs, rigid/affine/shear modes, and repeated matrix updates.

Use local input variables; private paths and generated geometry are not stored
in this repository:

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
export WMA_CPU_THREADS=4 CUDA_VISIBLE_DEVICES=1
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1

taskset -c 0-3 python tests/bench_gpu_embedding.py "$LABELING_ATLAS_DIRECTORY" \
  --input "$INPUT_TRACTOGRAPHY" --fibers 100000 --repeat 3 \
  --output /tmp/wma-gpu-labeling.json

taskset -c 0-3 python tests/bench_registration_kernel.py --baseline cffed63 \
  --fixed 5000 --moving 2000 --repeat 50 --maxfun 30

PYTHONPATH=.:tests python -m unittest test_gpu_registration test_atlas_blocks -v
```
