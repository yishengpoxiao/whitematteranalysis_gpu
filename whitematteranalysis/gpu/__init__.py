# -*- coding: utf-8 -*-

""" gpu

Acceleration backends for whitematteranalysis computations.

Two accelerated backends are provided, both drop-in numerical
equivalents of the original numpy/joblib implementations:

  * GPU backend (torch): distance/similarity matrices, spectral
    embedding GEMMs and the registration objective on one CUDA device.
  * CPU accelerated backend (vectorized numpy, multi-threaded): the
    same computations without torch, e.g. inside 3D Slicer.

With the default float64 precision both match the original results to
~1e-12 relative error and all discrete outputs (cluster labels, output
fiber files) are identical. All randomness (downsampling seeds, Nystrom
sampling, k-means initialization) stays on the CPU side by design.

Backend selection (also available as -gpu in the bin/ scripts):

    -gpu auto   GPU if torch+CUDA available, else CPU accelerated
    -gpu on     force GPU
    -gpu off    force the CPU accelerated backend (WMA_CPU_FAST=0
                restores the original per-fiber joblib loops)

Modules:
    config        global settings and availability checks
    fiber_tensors conversion between FiberArray/polydata and (N,P,3) tensors
    distance      pairwise/rectangular fiber distance and similarity matrices (GPU)
    cpu_backend   vectorized multi-threaded CPU backend (no torch needed)
    registration  batched registration objective and affine transforms
    linalg        spectral embedding GEMMs and optional GPU eigh

"""

from .config import (config, set_gpu, set_float32, set_devices, set_tile_rows,
                     use_gpu, float32_enabled, get_device, visible_device_count,
                     add_gpu_arguments, apply_gpu_arguments)

# import the backends so that `wma.gpu.distance` etc. always resolve.
# None of them import torch unconditionally at their module level, so
# this stays safe in environments without torch.
from . import fiber_tensors, distance, registration, linalg, cpu_backend


def available():
    """True if GPU acceleration can be used in this process."""
    return config.cuda_available()
