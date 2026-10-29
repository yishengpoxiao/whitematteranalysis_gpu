"""Reusable atlas coordinates and outlier probabilities; never cache subjects."""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import zipfile

import numpy as np

from . import fiber_tensors
from .config import float32_enabled, get_device


@lru_cache(maxsize=4)
def _implementation(backend, float32, device_index):
    files = ('fibers.py', 'filter.py', 'io.py', 'similarity.py', 'gpu/fiber_tensors.py',
             'gpu/cpu_backend.py', 'gpu/distance.py', 'gpu/distance_kernel.py',
             'gpu/registration_kernel.py', 'gpu/atlas_cache.py')
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for name in files:
        digest.update((root / name).read_bytes())
    features = sorted(name for name, enabled in np.core._multiarray_umath.__cpu_features__.items() if enabled)
    versions = [sys.platform, platform.machine(), platform.libc_ver(), features,
                np.__version__, backend, float32, digest.hexdigest()]
    if backend == 'gpu':
        import torch
        from . import distance_kernel
        versions += [torch.__version__, torch.version.cuda,
                     getattr(distance_kernel.triton, '__version__', None),
                     torch.cuda.get_device_name(device_index),
                     torch.cuda.get_device_capability(device_index)]
    else:
        import scipy
        versions.append(scipy.__version__)
    return versions


def _cache_file(filename, sigmasq, method, bilateral, backend):
    directory = os.environ.get('WMA_ATLAS_CACHE')
    if directory and directory.lower() in ('off', '0', 'false'):
        return None
    if not directory:
        base = os.environ.get('LOCALAPPDATA') if os.name == 'nt' else os.environ.get('XDG_CACHE_HOME')
        directory = str(Path(base or Path.home() / '.cache') / 'whitematteranalysis' / 'atlas-outliers')
    source = Path(filename).resolve()
    stat = source.stat()
    identity = [str(source), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns,
                15, float(sigmasq), str(method), bool(bilateral),
                _implementation(backend, float32_enabled() if backend == 'gpu' else False,
                                get_device().index if backend == 'gpu' else None)]
    key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
    return Path(directory).expanduser() / (key + '.npz')


def load_reference(filename, sigmasq, method='Mean', bilateral=True,
                   backend='gpu', tile_bytes=None):
    """Return (N,15,3) atlas coordinates and exact leave-one-out probabilities.

    Keys include reference identity, numerical parameters, precision, backend,
    library versions and implementation. Cache writes are atomic; an unavailable
    or damaged optional cache falls back to computing the reference normally.
    """
    from .. import io, similarity

    probability_dtype = np.dtype(np.float32 if backend == 'gpu' and float32_enabled() else np.float64)
    path = _cache_file(filename, sigmasq, method, bilateral, backend)
    if path is not None:
        try:
            with path.open('rb') as source, np.load(source, allow_pickle=False) as saved:
                coordinates, probabilities = saved['coordinates'], saved['probabilities']
            if (coordinates.ndim == 3 and coordinates.shape[1:] == (15, 3)
                    and coordinates.dtype == np.float64
                    and probabilities.dtype == probability_dtype
                    and probabilities.shape == (len(coordinates),)):
                return coordinates, probabilities
        except (OSError, ValueError, TypeError, AttributeError, EOFError, KeyError, zipfile.BadZipFile):
            pass
    coordinates = fiber_tensors.polydata_to_numpy(io.read_polydata(str(filename), geometry_only=True), 15)
    probabilities = np.empty(0, dtype=probability_dtype)
    if len(coordinates):
        if backend == 'gpu':
            from .distance import similarity_row_sums_from_arrays
            tile_rows = (max(1, tile_bytes // (len(coordinates) * 15 * 8 * 12))
                         if tile_bytes is not None else None)
            similarity_sums = similarity_row_sums_from_arrays(
                coordinates, coordinates, 0., distance_method=method,
                bilateral=bilateral, sigmasq=sigmasq, tile_rows=tile_rows)
            probabilities = (similarity_sums - 1.) / len(coordinates)
        else:
            from .cpu_backend import _distance_matrix as matrix
            kwargs = dict(number_of_jobs=1)
            if tile_bytes is not None:
                kwargs['tile_rows'] = max(1, tile_bytes // (len(coordinates) * 15 * 8 * 12))
            distances = matrix(coordinates, coordinates, 0., distance_method=method,
                               bilateral=bilateral, sigmasq=sigmasq, **kwargs)
            similarities = similarity.distance_to_similarity(distances, sigmasq)
            probabilities = (np.sum(similarities, axis=1) - 1.) / len(coordinates)
    if path is not None:
        temporary = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=str(path.parent), suffix='.npz', delete=False) as output:
                temporary = output.name
                np.savez(output, coordinates=coordinates, probabilities=probabilities)
            os.replace(temporary, str(path))
        except OSError:
            pass
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)
    return coordinates, probabilities
