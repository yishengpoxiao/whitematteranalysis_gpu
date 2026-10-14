# -*- coding: utf-8 -*-

"""WMAtlasLabelGPUCore

Pure-python core of the WhiteMatterAnalysis atlas-labeling pipeline,
shared by the WMA Slicer workflows and headless use.
No Slicer imports here: everything can run in a plain python
environment for testing.

Given a subject tractography (vtkPolyData) and a WhiteMatterAnalysis
atlas directory (as produced by wm_cluster_atlas.py, containing
atlas.p), this labels every fiber with an anatomical cluster id using
the GPU-accelerated or the vectorized-CPU backend of the
whitematteranalysis package.
"""

import os
import vtk

from WMARegistration import registration_parameters, configure_scale

_ATLAS_CACHE = None


def resolve_backend(backend):
    """Map a friendly backend name to wma.gpu settings.

    backend: 'auto'      - GPU if torch+CUDA available, else CPU accelerated
             'gpu'       - force GPU (requires torch with CUDA)
             'cpu'       - vectorized multi-threaded CPU backend (no torch)
             'original'  - the unmodified per-fiber joblib implementation
    """

    import whitematteranalysis as wma

    backend = (backend or 'auto').lower()
    if backend == 'gpu':
        if not wma.gpu.available():
            raise RuntimeError('GPU backend requires CUDA-enabled PyTorch in Slicer. Choose Auto or CPU accelerated.')
        wma.gpu.set_gpu('on')
    elif backend == 'cpu':
        wma.gpu.set_gpu('off')
        wma.gpu.config.set_cpu_fast(True)
    elif backend == 'original':
        wma.gpu.set_gpu('off')
        wma.gpu.config.set_cpu_fast(False)
    else:  # auto
        wma.gpu.set_gpu('auto')
        wma.gpu.config.set_cpu_fast(True)
    return describe_backend(backend)


def describe_backend(backend):
    import whitematteranalysis as wma
    if wma.gpu.use_gpu():
        return "GPU (torch, single device)"
    if wma.gpu.config.cpu_fast_enabled():
        return "CPU accelerated (vectorized numpy, multi-threaded)"
    return "CPU original (per-fiber joblib)"


def _atlas_path(path):
    return os.path.abspath(os.path.expanduser(os.path.expandvars(os.fspath(path))))


def _find_atlas_file(directory, names):
    """Prefer direct files; reject ambiguous recursive matches."""
    for name in names:
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            return path
    hits = []
    for root, dirs, files in os.walk(directory):
        dirs[:] = sorted(d for d in dirs if not d.startswith('.') and d != '__MACOSX')
        for name in names:
            if name in files:
                hits.append(os.path.join(root, name))
                break
    if len(hits) > 1:
        raise IOError("Multiple atlas matches in %s; select a specific atlas directory: %s"
                      % (directory, ', '.join(hits)))
    return hits[0] if hits else None


def resolve_atlas_directory(atlas_directory):
    """Accept ORG roots, atlas.p, build roots, iterations, or atlas folders.

    Build roots select the latest numbered iteration containing atlas.p.
    Other parent layouts must contain exactly one labeling atlas.
    """
    directory = _atlas_path(atlas_directory)
    if os.path.isfile(directory):
        directory = os.path.dirname(directory)
    if not os.path.isdir(directory):
        raise IOError("Atlas directory does not exist: %s" % directory)
    if os.path.isfile(os.path.join(directory, 'atlas.p')):
        return directory
    for child in ('ORG-800FC-100HCP', 'initial_clusters'):
        candidate = os.path.join(directory, child)
        if os.path.isfile(os.path.join(candidate, 'atlas.p')):
            return candidate
    iterations = []
    for name in os.listdir(directory):
        if name.startswith('iteration_') and name[10:].isdigit():
            candidate = os.path.join(directory, name, 'initial_clusters')
            if os.path.isfile(os.path.join(candidate, 'atlas.p')):
                iterations.append((int(name[10:]), candidate))
    if iterations:
        return max(iterations)[1]
    hit = _find_atlas_file(directory, ('atlas.p',))
    if hit:
        return os.path.dirname(hit)
    raise IOError("atlas.p not found in %s. Select an ORG atlas parent, "
                  "ORG-800FC-100HCP, or a built atlas run/iteration/initial_clusters directory."
                  % directory)


def load_atlas(atlas_directory):
    """Load the labeling atlas resolved from a supported file or directory."""
    import whitematteranalysis as wma
    global _ATLAS_CACHE
    directory = resolve_atlas_directory(atlas_directory)
    files = [os.path.join(directory, 'atlas' + extension) for extension in ('.p', '.vtp')]
    key = tuple((path, os.stat(path).st_mtime_ns, os.stat(path).st_size) for path in files)
    if _ATLAS_CACHE is None or _ATLAS_CACHE[0] != key:
        _ATLAS_CACHE = (key, wma.cluster.load_atlas(directory, 'atlas'))
    return _ATLAS_CACHE[1]


def run_atlas_labeling(input_polydata, atlas_directory, fiber_length_mm=40.0,
                       backend='auto', number_of_jobs=1, progress_callback=None):
    """Label every fiber of `input_polydata` (vtkPolyData) with the
    clusters of a WMA atlas.

    Returns a dict with:
        output_polydata : the (preprocessed) polydata with ClusterNumber,
                          EmbeddingColor, EmbeddingCoordinate,
                          TotalFiberSimilarity and MeasuredFiberSimilarity
                          cell arrays attached
        cluster_idx     : (n_fibers,) int cluster id per fiber
        color           : (n_fibers, 3) float embedding colors
        cluster_count   : number of clusters in the atlas
        n_fibers        : number of fibers after preprocessing
        backend_name    : human readable description of the backend used
    """

    import whitematteranalysis as wma

    backend_name = resolve_backend(backend)
    if progress_callback:
        progress_callback("Loading atlas...", 5)

    atlas = load_atlas(atlas_directory)

    if progress_callback:
        progress_callback("Preprocessing fibers (length filter)...", 15)
    pd2 = wma.filter.preprocess(input_polydata, fiber_length_mm,
                                verbose=False)

    if progress_callback:
        progress_callback("Labeling fibers (%s)..." % backend_name, 30)
    output_polydata, cluster_idx, color, embed = wma.cluster.spectral_atlas_label(
        pd2, atlas, number_of_jobs)

    cluster_count = atlas.centroids.shape[0]
    if progress_callback:
        progress_callback("Done: %d fibers in %d clusters." % (
            pd2.GetNumberOfLines(), cluster_count), 100)

    return dict(output_polydata=output_polydata,
                cluster_idx=cluster_idx,
                color=color,
                embed=embed,
                cluster_count=cluster_count,
                n_fibers=pd2.GetNumberOfLines(),
                backend_name=backend_name)


def extract_clusters(output_polydata, cluster_idx, cluster_count):
    """Split the labeled polydata into per-cluster vtkPolyData objects
    (one per atlas cluster, order = cluster id)."""

    import whitematteranalysis as wma
    pd_c_list = wma.cluster.mask_all_clusters(
        output_polydata, cluster_idx, cluster_count,
        preserve_point_data=True, preserve_cell_data=True, verbose=False)
    return pd_c_list


# ---------------------------------------------------------------------------
# Registration (subject -> atlas), following bin/wm_register_to_atlas_new.py
# ---------------------------------------------------------------------------

def find_atlas_tractography(atlas_directory):
    """Locate the atlas tractography polydata file (atlas.vtp/vtk)
    inside an atlas directory, searching common layouts."""

    directory = resolve_atlas_directory(atlas_directory)
    hit = _find_atlas_file(directory, ('atlas.vtp', 'atlas.vtk'))
    if hit:
        return hit
    raise IOError("No atlas tractography (atlas.vtp / atlas.vtk) found in %s" % directory)


def register_subject_to_atlas(subject_polydata, atlas_polydata,
                              fiber_length_mm=40.0, fiber_length_max_mm=260.0,
                              backend='auto', random_seed=1000, params=None,
                              progress_callback=None, mode='affine',
                              output_directory=None, verbose=False):
    """Register moving tractography to fixed tractography with a CLI preset
    or a custom multiscale schedule. Returns the forward RAS transform.
    Nonrigid registration expects inputs already aligned with rigid/affine.
    """

    import whitematteranalysis as wma

    import tempfile

    backend_name = resolve_backend(backend)
    p = registration_parameters(mode, params)

    register = wma.congeal_to_atlas.SubjectToAtlasRegistration()
    register.output_directory = output_directory or tempfile.mkdtemp(prefix='wma_register_')
    register.mode = p['modes'][0]
    register.save_iteration_transforms = verbose
    register.fiber_length = fiber_length_mm
    register.fiber_length_max = fiber_length_max_mm
    register.set_subject(subject_polydata, 'subject')
    register.set_atlas(atlas_polydata, 'atlas')
    register.points_per_fiber = p['points_per_fiber']
    register.random_seed = random_seed

    n_scales = len(p['sigma'])
    total_iterations = sum(p['iterations'])
    iteration = 0
    for scale in range(n_scales):
        configure_scale(register, p, scale)
        for _it in range(p['iterations'][scale]):
            if progress_callback:
                progress_callback(
                    "Registration scale %d/%d, iteration %d/%d" % (
                        scale + 1, n_scales, iteration + 1, total_iterations),
                    40 * iteration // max(1, total_iterations))
            register.iterate()
            if verbose:
                intermediate = os.path.join(register.output_directory, 'iteration_%05d' % (iteration + 1))
                os.makedirs(intermediate, exist_ok=True)
                wma.io.write_polydata(apply_transform_to_polydata(subject_polydata, register.transform),
                                      os.path.join(intermediate, 'moving_registered.vtp'))
            iteration += 1

    # note: the upstream iterate() writes per-iteration objective values
    # to log files under its output_directory and keeps only the final
    # transform on the object, so the objectives series is not returned
    # here; labeling quality can be compared via TotalFiberSimilarity.
    return dict(transform=register.transform,
                transform_array=register.transform_as_array,
                iterations=iteration,
                output_directory=register.output_directory,
                backend_name=backend_name)


def apply_transform_to_polydata(polydata, transform, inverse=False):
    """Apply a vtkTransform to a polydata (optionally inverted), i.e.
    harden the transform, like congeal_to_atlas's save_transformed_
    polydata_to_disk but in memory."""

    import vtk
    use_transform = transform.GetInverse() if inverse else transform
    transformer = vtk.vtkTransformPolyDataFilter()
    transformer.SetInputData(polydata)
    transformer.SetTransform(use_transform)
    transformer.Update()
    return transformer.GetOutput()


# ---------------------------------------------------------------------------
# Atlas creation from multiple subjects, following bin/wm_cluster_atlas.py
# ---------------------------------------------------------------------------

def build_atlas(subject_directory, output_directory,
                fibers_per_subject=500, fiber_length_mm=60.0,
                number_of_clusters=20, nystrom_sample=600, iterations=2,
                cluster_outlier_std=2.0, subject_percent=0.4,
                outlier_sigma=20.0, distance_method='Mean', bilateral=True,
                random_seed=None, backend='auto', progress_callback=None):
    """Build a WMA tract atlas from a directory of (registered) subject
    tractography files, following the wm_cluster_atlas.py core loop:
    spectral clustering with the Nystrom method plus per-iteration
    leave-one-out cluster outlier rejection and whole-cluster rejection
    when too few subjects are present.

    Differences from the CLI: QC histograms, MRML scenes and rendering
    are not produced (the atlas.p / atlas.vtp / cluster files, which
    define the atlas, are identical in content).

    Returns dict(atlas_directory=directory containing atlas.p for later
    labeling, cluster_count, n_fibers, iterations_run).
    """

    import numpy
    import whitematteranalysis as wma

    backend_name = resolve_backend(backend)

    if random_seed is not None:
        numpy.random.seed(seed=random_seed)

    fnames = wma.io.list_vtk_files(subject_directory)
    if len(fnames) < 2:
        raise IOError("Need at least 2 subject files in %s" % subject_directory)
    n_subjects = len(fnames)
    if progress_callback:
        progress_callback("Reading and preprocessing %d subjects..." % n_subjects, 5)

    input_pds = list()
    for sidx, fname in enumerate(fnames):
        pd = wma.io.read_polydata(fname)
        pd2 = wma.filter.preprocess(pd, fiber_length_mm, verbose=False)
        pd3 = wma.filter.downsample(pd2, fibers_per_subject, verbose=False,
                                    random_seed=random_seed)
        if pd3.GetNumberOfLines() != fibers_per_subject:
            raise IOError("Subject %s has too few fibers over the length "
                          "threshold (%d found, %d requested)" % (
                              fname, pd3.GetNumberOfLines(), fibers_per_subject))
        input_pds.append(pd3)

    import vtk
    appender = vtk.vtkAppendPolyData()
    for pd in input_pds:
        appender.AddInputData(pd)
    appender.Update()
    input_data = appender.GetOutput()

    subject_fiber_list = numpy.repeat(numpy.arange(n_subjects), fibers_per_subject)

    outdir = os.path.abspath(output_directory)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    supported = wma.gpu.distance.distance_supported(distance_method)
    use_gpu = wma.gpu.use_gpu() and supported
    cpu_fast = wma.gpu.config.cpu_fast_enabled() and supported
    cluster_local_sigmasq = outlier_sigma * outlier_sigma
    subjects_threshold = subject_percent * n_subjects

    result = None
    for iteration in range(iterations):
        if progress_callback:
            progress_callback("Atlas iteration %d/%d: spectral clustering..." % (
                iteration + 1, iterations), 10 + 60 * iteration // iterations)

        nystrom_mask = numpy.random.permutation(
            input_data.GetNumberOfLines()) < nystrom_sample
        (output_polydata_s, cluster_numbers_s, color, embed, _distortion,
         atlas, reject_idx) = wma.cluster.spectral(
            input_data, number_of_clusters=number_of_clusters,
            number_of_jobs=1, use_nystrom=True, nystrom_mask=nystrom_mask,
            number_of_eigenvectors=10, sigma=60.0,
            distance_method=distance_method, normalized_cuts=True,
            threshold=2.0, outlier_std_threshold=2.0, pos_def_approx=True,
            bilateral=bilateral)

        # drop the fibers spectral rejected, keep bookkeeping consistent
        subject_fiber_list = numpy.delete(subject_fiber_list, reject_idx)
        input_data = output_polydata_s
        cluster_numbers_s = numpy.asarray(cluster_numbers_s)
        cluster_indices = numpy.unique(cluster_numbers_s[cluster_numbers_s >= 0])

        if progress_callback:
            progress_callback("Atlas iteration %d/%d: cluster outlier rejection..." % (
                iteration + 1, iterations), 10 + 60 * (iteration + 1) // iterations)

        # per-cluster leave-one-out outlier rejection (bin:503-595)
        reject_idx = list()
        fiber_mean_sim = numpy.zeros(len(cluster_numbers_s))
        for c in cluster_indices:
            mask = cluster_numbers_s == c
            fiber_indices = numpy.nonzero(mask)[0]
            number_fibers_in_cluster = int(numpy.sum(mask))
            subject_ID_per_fiber = subject_fiber_list[mask]
            subjects_per_cluster = len(set(subject_ID_per_fiber))

            pd_c = wma.filter.mask(input_data, mask, verbose=False)
            if use_gpu:
                cluster_distances = wma.cluster._pairwise_distance_matrix(
                    pd_c, 0.0, number_of_jobs=1, bilateral=bilateral,
                    distance_method=distance_method, use_gpu=True)
            elif cpu_fast:
                cluster_distances = wma.gpu.cpu_backend.pairwise_distance_matrix(
                    pd_c, 0.0, distance_method=distance_method,
                    bilateral=bilateral)
            else:
                cluster_distances = wma.cluster._pairwise_distance_matrix(
                    pd_c, 0.0, number_of_jobs=1, bilateral=bilateral,
                    distance_method=distance_method)
            if distance_method == 'StrictSimilarity':
                cluster_similarity = cluster_distances
            else:
                cluster_similarity = wma.similarity.distance_to_similarity(
                    cluster_distances, cluster_local_sigmasq)

            # leave-one-out total similarity: only fibers from other
            # subjects count (bin:541-552)
            total_similarity = numpy.zeros(number_fibers_in_cluster)
            for j, sidx in enumerate(subject_ID_per_fiber):
                loo_mask = subject_ID_per_fiber != sidx
                n_loo = int(numpy.sum(loo_mask))
                if n_loo == 0:
                    total_similarity[j] = numpy.nan
                else:
                    total_similarity[j] = numpy.sum(
                        cluster_similarity[j, loo_mask]) / n_loo

            mean_sim = numpy.nanmean(total_similarity)
            cluster_std = numpy.nanstd(total_similarity)
            cutoff = mean_sim - cluster_outlier_std * cluster_std
            for j, fidx in enumerate(fiber_indices):
                fiber_mean_sim[fidx] = total_similarity[j]
                sim = total_similarity[j]
                if numpy.isnan(sim) or sim < cutoff or \
                        subjects_per_cluster < subjects_threshold:
                    cluster_numbers_s[fidx] = -c - 1
                    reject_idx.append(fidx)

        # second pass: prune fibers whose mean similarity is low compared
        # to the whole brain (bin:597-612)
        brain_mean_sim = numpy.mean(fiber_mean_sim)
        brain_std_sim = numpy.std(fiber_mean_sim)
        for fidx in range(len(cluster_numbers_s)):
            if fiber_mean_sim[fidx] < brain_mean_sim - cluster_outlier_std * brain_std_sim:
                if cluster_numbers_s[fidx] >= 0:
                    cluster_numbers_s[fidx] = -cluster_numbers_s[fidx] - 1
                    reject_idx.append(fidx)

        # keep only the retained fibers for the next iteration / output
        keep = cluster_numbers_s >= 0
        input_data = wma.filter.mask(input_data, keep, verbose=False)
        cluster_numbers_s = cluster_numbers_s[keep]
        subject_fiber_list = subject_fiber_list[keep]
        kept_cluster_ids = numpy.unique(cluster_numbers_s)
        if len(kept_cluster_ids) == 0:
            raise ValueError('Atlas construction rejected every cluster; adjust input data or outlier thresholds.')
        renumber = numpy.zeros(cluster_indices.max() + 1, dtype=numpy.int64)
        for new_id, old_id in enumerate(kept_cluster_ids):
            renumber[old_id] = new_id
        cluster_numbers_s = renumber[cluster_numbers_s]
        atlas.centroids = atlas.centroids[kept_cluster_ids]
        cluster_array = vtk.util.numpy_support.numpy_to_vtk(
            cluster_numbers_s.astype(numpy.int32), deep=True)
        cluster_array.SetName('ClusterNumber')
        input_data.GetCellData().AddArray(cluster_array)

        # save this iteration's atlas + cluster files (atlas.p + atlas.vtp)
        iter_dir = os.path.join(outdir, 'iteration_%05d' % (iteration + 1))
        clusters_dir = os.path.join(iter_dir, 'initial_clusters')
        if not os.path.isdir(clusters_dir):
            os.makedirs(clusters_dir)
        atlas.save(clusters_dir, 'atlas')
        pd_c_list = wma.cluster.mask_all_clusters(
            input_data, cluster_numbers_s, len(kept_cluster_ids),
            preserve_point_data=True, preserve_cell_data=True, verbose=False)
        for c, pd_c in enumerate(pd_c_list):
            wma.io.write_polydata(
                pd_c, os.path.join(clusters_dir, 'cluster_%05d.vtp' % (c + 1)))

        result = dict(atlas_directory=clusters_dir,
                      cluster_count=len(kept_cluster_ids),
                      n_fibers=int(len(cluster_numbers_s)),
                      iterations_run=iteration + 1)
        if progress_callback:
            progress_callback("Iteration %d done: %d fibers, %d clusters." % (
                iteration + 1, len(cluster_numbers_s), len(kept_cluster_ids)),
                10 + 60 * (iteration + 1) // iterations)

    result['backend_name'] = backend_name
    result['n_subjects'] = n_subjects
    return result

def find_atlas_cluster_directory(atlas_directory):
    """Locate a directory containing the atlas cluster files
    (cluster_00001.vtp ...) needed for outlier removal."""

    directory = resolve_atlas_directory(atlas_directory)
    hit = _find_atlas_file(directory, ('cluster_00001.vtp',))
    return os.path.dirname(hit) if hit else None


# ---------------------------------------------------------------------------
# Atlas-based outlier removal, following bin/wm_cluster_remove_outliers.py
# ---------------------------------------------------------------------------

def remove_cluster_outliers(subject_polydata, cluster_idx, cluster_count,
                            atlas_cluster_directory, outlier_sigma=20.0,
                            std_threshold=2.0, distance_method='Mean',
                            bilateral=True, backend='auto',
                            progress_callback=None, max_workers=0,
                            event_callback=None):
    """Reject outliers in independent clusters using bounded parallel workers.

    max_workers=0 uses available CPU cores (and a GPU memory bound), 1 is
    sequential. GPU workers use separate CUDA streams. CPU workers each use
    one distance thread. Progress/events and mask assembly stay on the caller.
    Returns (fiber_mask, rejected_count), in original subject fiber order.
    """
    from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
    from contextlib import nullcontext
    from threading import local
    import numpy
    import whitematteranalysis as wma
    from whitematteranalysis.gpu import fiber_tensors

    resolve_backend(backend)
    supported = wma.gpu.distance.distance_supported(distance_method)
    use_gpu = wma.gpu.use_gpu() and supported
    cpu_fast = wma.gpu.config.cpu_fast_enabled() and supported
    cluster_local_sigmasq = outlier_sigma * outlier_sigma
    labels = numpy.asarray(cluster_idx)
    order = numpy.argsort(labels, kind='stable')
    ids, starts = numpy.unique(labels[order], return_index=True)
    groups = {int(c): indices for c, indices in zip(ids, numpy.split(order, starts[1:]))
              if 0 <= c < cluster_count}
    fiber_mask = numpy.zeros(subject_polydata.GetNumberOfLines())
    if not groups:
        return fiber_mask, 0
    if max_workers < 0:
        raise ValueError("max_workers must be nonnegative")
    cores = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    if not use_gpu and wma.gpu.config.cpu_threads:
        cores = min(cores, wma.gpu.config.cpu_threads)
    workers = min(len(groups), max_workers or cores)
    worker_state = local()
    gpu_budget = None
    if use_gpu:
        import torch
        from whitematteranalysis.gpu.config import get_device
        device = get_device()
        free_bytes, _ = torch.cuda.mem_get_info(device)
        # Leave half of free VRAM for outputs, CUDA context, and other Slicer work.
        workers = min(workers, max(1, free_bytes // (512 << 20)))
        gpu_budget = max(1, int(free_bytes * 0.5) // workers)
    if progress_callback:
        progress_callback("Outlier removal: atlas defines %d clusters; subject has %d nonempty, %d empty; "
                          "%d parallel workers (%s)" % (
            cluster_count, len(groups), cluster_count - len(groups), workers,
            'GPU streams' if use_gpu else 'CPU'), 60)
        reference_names = {entry.name for entry in os.scandir(atlas_cluster_directory) if entry.is_file()}
        reference_count = sum('cluster_%05d.vtp' % (c + 1) in reference_names for c in range(cluster_count))
        progress_callback("Atlas reference cluster files: %d/%d" % (reference_count, cluster_count), 60)

    # Resample the subject once. Workers receive disjoint NumPy slices rather
    # than repeatedly masking and reading the shared Slicer VTK input.
    if use_gpu or cpu_fast:
        subject_arrays = fiber_tensors.polydata_to_numpy(subject_polydata, 15)
    else:
        subject_clusters = {c: wma.filter.mask(subject_polydata, labels == c, verbose=False)
                            for c in groups}

    def compute(c):
        indices = groups[c]
        filename = os.path.join(atlas_cluster_directory, 'cluster_%05d.vtp' % (c + 1))
        if not os.path.isfile(filename):
            return c, indices, 0, 'missing'
        kwargs = dict(bilateral=bilateral, distance_method=distance_method)
        if use_gpu or cpu_fast:
            subject_array = subject_arrays[indices]
            # Bound each worker's temporary distance tensors, including bilateral
            # and forward/reversed comparisons. Full result matrices live on CPU.
            budget = gpu_budget if use_gpu else (16 << 20)
            if use_gpu:
                if not hasattr(worker_state, 'stream'):
                    worker_state.stream = torch.cuda.Stream(device=device)
                stream_context = torch.cuda.stream(worker_state.stream)
            else:
                stream_context = nullcontext()
            with stream_context:
                from whitematteranalysis.gpu.atlas_cache import load_reference
                atlas_array, total_atlas = load_reference(filename, cluster_local_sigmasq,
                    distance_method, bilateral, 'gpu' if use_gpu else 'cpu', tile_bytes=budget)
                n_atlas = len(atlas_array)
                if not n_atlas:
                    return c, indices, 0, 'empty'
                tile_rows = max(1, budget // (n_atlas * 15 * 8 * 12))
                if use_gpu:
                    matrix = wma.gpu.distance.distance_matrix_from_arrays
                    parallel = {}
                else:
                    matrix = wma.gpu.cpu_backend._distance_matrix
                    parallel = dict(number_of_jobs=1)
                distances_subject = matrix(subject_array, atlas_array, 0.,
                    tile_rows=tile_rows, **parallel, **kwargs).T
        else:
            atlas_pd = wma.io.read_polydata(filename)
            n_atlas = atlas_pd.GetNumberOfLines()
            if not n_atlas:
                return c, indices, 0, 'empty'
            distances_atlas = wma.cluster._pairwise_distance_matrix(
                atlas_pd, 0., number_of_jobs=1, **kwargs)
            distances_subject = wma.cluster._rectangular_distance_matrix(
                subject_clusters[c], atlas_pd, 0., number_of_jobs=1, **kwargs)
            similarity_atlas = wma.similarity.distance_to_similarity(distances_atlas, cluster_local_sigmasq)
            total_atlas = (numpy.sum(similarity_atlas, axis=1) - 1.) / n_atlas
        similarity_subject = wma.similarity.distance_to_similarity(distances_subject, cluster_local_sigmasq)
        total_subject = numpy.sum(similarity_subject, axis=0) / n_atlas
        cutoff = numpy.mean(total_atlas) - std_threshold * numpy.std(total_atlas)
        keep = total_subject >= cutoff
        return c, indices[keep], int((~keep).sum()), None

    rejected_total = 0
    completed = 0

    def collect(result):
        nonlocal rejected_total, completed
        c, kept_indices, rejected, skipped = result
        fiber_mask[kept_indices] = 1
        rejected_total += rejected
        completed += 1
        if progress_callback:
            if skipped:
                progress_callback("WARNING: cluster %d: %s atlas reference; outlier check skipped, "
                                  "%d subject fibers retained." % (c + 1, skipped, len(kept_indices)),
                                  60 + 35 * completed // len(groups))
            progress_callback("Outlier removal: %d/%d nonempty subject clusters done (atlas total %d); "
                              "cluster %d rejected %d/%d fibers" % (
                completed, len(groups), cluster_count, c + 1, rejected, len(groups[c])),
                60 + 35 * completed // len(groups))

    if workers == 1:
        for c in groups:
            collect(compute(c))
            if event_callback:
                event_callback()
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {pool.submit(compute, c) for c in groups}
            while pending:
                done, pending = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
                for future in done:
                    collect(future.result())
                if event_callback:
                    event_callback()
    return fiber_mask, rejected_total


# ---------------------------------------------------------------------------
# Hemisphere separation + anatomical tracts, following
# bin/wm_assess_cluster_location_by_hemisphere.py,
# bin/wm_separate_clusters_by_hemisphere.py and
# bin/wm_append_clusters_to_anatomical_tracts.py
# ---------------------------------------------------------------------------

def assess_hemisphere_location(cluster_polydata, hemisphere_percent_threshold=0.95):
    """Per-fiber hemisphere location of one cluster polydata:
    1 = left, 2 = right, 3 = commissural (RAS coordinates; a fiber is
    lateralized when >threshold of its points are on one side)."""

    import numpy
    import whitematteranalysis as wma

    fibers = wma.fibers.FiberArray()
    fibers.points_per_fiber = 20
    fibers.hemisphere_percent_threshold = hemisphere_percent_threshold
    fibers.hemispheres = True
    fibers.convert_from_polydata(cluster_polydata)

    mask_location = numpy.zeros(fibers.number_of_fibers)
    mask_location[fibers.index_left_hem] = 1
    mask_location[fibers.index_right_hem] = 2
    mask_location[fibers.index_commissure] = 3
    return mask_location


def separate_clusters_by_hemisphere(pd_c_list, hemisphere_percent_threshold=0.95,
                                    output_directory=None, progress_callback=None):
    """Split every cluster into its left / right / commissural parts
    (wm_separate_clusters_by_hemisphere.py, in memory).

    Returns dict(tracts_left_hemisphere=[...], tracts_right_hemisphere=[...],
                 tracts_commissural=[...]) with one polydata per cluster
    per side (empty when the cluster has no fibers on that side). If
    output_directory is given, the three standard subdirectories
    (tracts_left_hemisphere etc.) are written for CLI compatibility.
    """

    import os
    import numpy
    import whitematteranalysis as wma

    outdirs = {}
    if output_directory:
        for side in ('tracts_left_hemisphere', 'tracts_right_hemisphere',
                     'tracts_commissural'):
            d = os.path.join(output_directory, side)
            if not os.path.isdir(d):
                os.makedirs(d)
            outdirs[side] = d

    result = {'tracts_left_hemisphere': [], 'tracts_right_hemisphere': [],
              'tracts_commissural': []}
    for c, pd_c in enumerate(pd_c_list):
        if progress_callback and c % 25 == 0:
            progress_callback("Separating hemispheres: cluster %d/%d" % (c + 1, len(pd_c_list)))
        location = assess_hemisphere_location(pd_c, hemisphere_percent_threshold)
        fname_base = 'cluster_%05d.vtp' % (c + 1)
        for side, value in (('tracts_left_hemisphere', 1),
                            ('tracts_right_hemisphere', 2),
                            ('tracts_commissural', 3)):
            part_mask = (location == value).astype(numpy.float64)
            pd_part = wma.filter.mask(pd_c, part_mask, preserve_point_data=True,
                                      preserve_cell_data=True, verbose=False)
            result[side].append(pd_part)
            if output_directory:
                wma.io.write_polydata(pd_part, os.path.join(outdirs[side], fname_base))
    return result


def find_anatomical_tracts_mrml(atlas_directory):
    """Locate the ORG anatomical tract MRML files (T_*.mrml) inside an
    atlas directory. Returns the list of file paths, or None when the
    atlas does not provide anatomical tracts."""

    hits = []
    for root, _dirs, files in os.walk(resolve_atlas_directory(atlas_directory)):
        for f in files:
            if f.startswith('T_') and f.endswith('.mrml'):
                hits.append(os.path.join(root, f))
    hits.sort()
    return hits if hits else None


def append_clusters_to_anatomical_tracts(pd_c_list, atlas_mrml_files,
                                         separated, output_directory=None,
                                         progress_callback=None):
    """Merge the subject's (hemisphere-separated) clusters into the
    anatomically named tracts of the ORG atlas
    (wm_append_clusters_to_anatomical_tracts.py, in memory).

    pd_c_list: per-cluster polydatas (cluster id order)
    atlas_mrml_files: T_*.mrml paths from find_anatomical_tracts_mrml
    separated: dict from separate_clusters_by_hemisphere

    Each output tract polydata carries a per-point 'cluster_idx' array.
    Returns [(tract_name, polydata), ...] with names like
    'T_AF_left', 'T_CC1'; if output_directory is given the tracts are
    also written there as <name>.vtp.
    """

    import numpy
    import vtk
    from vtk.util import numpy_support
    import whitematteranalysis as wma

    def cluster_number(fname_base):
        # 'cluster_00007.vtp' -> 7
        return int(fname_base[8:13])

    tracts = []
    for i, mrml_path in enumerate(atlas_mrml_files):
        if progress_callback:
            progress_callback("Mapping anatomical tract %d/%d: %s" % (
                i + 1, len(atlas_mrml_files), os.path.basename(mrml_path)))
        tract = os.path.splitext(os.path.basename(mrml_path))[0]
        member_numbers = []
        with open(mrml_path) as f:
            for line in f:
                idx = line.find('.vtp')
                if idx > 0:
                    member_numbers.append(cluster_number(line[idx - 13:idx + 4]))
        if not member_numbers:
            continue

        is_commissural = tract.startswith('T_CC') or tract.startswith('T_MCP')
        # mrml groups define the cluster membership of each anatomical
        # tract; the side is decided by tract family (left/right tracts
        # share one mrml group listing their clusters)
        groups = {}
        if is_commissural:
            groups[tract] = ('tracts_commissural', member_numbers)
        else:
            groups[tract + '_left'] = ('tracts_left_hemisphere', member_numbers)
            groups[tract + '_right'] = ('tracts_right_hemisphere', member_numbers)

        for tract_name, (side, numbers) in groups.items():
            appender = vtk.vtkAppendPolyData()
            used = False
            # the CLI writes the position of the cluster within this
            # tract's member list (0-based) as the per-point cluster_idx
            for cluster_seq, number in enumerate(sorted(numbers)):
                if number - 1 >= len(pd_c_list):
                    continue
                pd_part = separated[side][number - 1]
                if pd_part.GetNumberOfLines() == 0:
                    continue
                values = numpy.full(pd_part.GetPoints().GetNumberOfPoints(),
                                    cluster_seq, dtype=numpy.int32)
                arr = numpy_support.numpy_to_vtk(values, deep=True)
                arr.SetName('cluster_idx')
                pd_part.GetPointData().AddArray(arr)
                appender.AddInputData(pd_part)
                used = True
            if not used:
                continue
            appender.Update()
            pd_tract = appender.GetOutput()
            tracts.append((tract_name, pd_tract))
            if output_directory:
                if not os.path.isdir(output_directory):
                    os.makedirs(output_directory)
                wma.io.write_polydata(
                    pd_tract, os.path.join(output_directory, tract_name + '.vtp'))
    return tracts


def extract_anatomical_tracts(polydata, cluster_idx, atlas_mrml_files,
                              hemisphere_percent_threshold=0.95, progress_callback=None):
    """Select final tracts directly, sharing geometry and hemisphere assessment.

    Preserve the legacy cluster/member order, metadata casts, and cluster_idx
    provenance without constructing 800 clusters and 2,400 hemisphere copies.
    """
    import numpy
    from vtk.util import numpy_support
    import whitematteranalysis as wma

    count = polydata.GetNumberOfLines()
    if not count:
        return []
    geometry = wma.filter._lines_offsets_connectivity(polydata)
    if geometry is None:
        raise RuntimeError('Direct anatomical extraction requires VTK 9 connectivity arrays.')
    offsets, connectivity, points = geometry
    labels = numpy.asarray(cluster_idx)
    if len(labels) != count:
        raise ValueError('Cluster labels must match the number of tractography lines.')
    lengths = numpy.diff(offsets)
    location = numpy.full(count, 3, dtype=numpy.uint8)
    for start in range(0, count, 50000):
        end = min(start + 50000, count)
        positions = numpy.rint(numpy.arange(20) * ((lengths[start:end] - 1.) / 19.)[:, None]).astype(numpy.int64)
        ids = connectivity[offsets[start:end, None] + positions]
        # Legacy cluster extraction casts points to float32 before assessment.
        fraction_left = (points[ids, 0].astype(numpy.float32) < 0).sum(axis=1) / 20.
        location[start:end][fraction_left > hemisphere_percent_threshold] = 1
        location[start:end][fraction_left < 1 - hemisphere_percent_threshold] = 2

    order = numpy.argsort(labels, kind='stable')
    ids, starts = numpy.unique(labels[order], return_index=True)
    groups = {int(c): indices for c, indices in zip(ids, numpy.split(order, starts[1:]))}
    empty = numpy.empty(0, dtype=numpy.int64)
    tracts = []
    for i, path in enumerate(atlas_mrml_files):
        if progress_callback:
            progress_callback('Mapping anatomical tract %d/%d: %s' % (i + 1, len(atlas_mrml_files), os.path.basename(path)))
        name = os.path.splitext(os.path.basename(path))[0]
        members = sorted(int(filename[8:13]) for filename in _parse_tract_mrml_members(path))
        member_positions = {number: position for position, number in enumerate(members)}
        sides = [(name, 3)] if name.startswith(('T_CC', 'T_MCP')) else [(name + '_left', 1), (name + '_right', 2)]
        for tract_name, side in sides:
            selected, provenance = [], []
            for number in members:
                indices = groups.get(number - 1, empty)
                indices = indices[location[indices] == side]
                if len(indices):
                    selected.append(indices)
                    # Repeated members share their last assigned index in the legacy appender.
                    provenance.append(numpy.full(len(indices), member_positions[number], dtype=numpy.int32))
            if not selected:
                continue
            indices = numpy.concatenate(selected)
            tract = wma.filter.mask(polydata, None, preserve_point_data=True, preserve_cell_data=True,
                                    verbose=False, _geometry=geometry, _line_indices=indices)
            values = numpy.repeat(numpy.concatenate(provenance), lengths[indices])
            array = numpy_support.numpy_to_vtk(values, deep=True)
            array.SetName('cluster_idx')
            tract.GetPointData().AddArray(array)
            if len(selected) > 1:
                # vtkAppendPolyData orders merged attribute fields by name.
                for attributes in (tract.GetPointData(), tract.GetCellData()):
                    arrays = [attributes.GetAbstractArray(index) for index in range(attributes.GetNumberOfArrays())]
                    tensors = attributes.GetTensors()
                    tensor_name = tensors.GetName() if tensors is not None else None
                    attributes.Initialize()
                    for item in sorted(arrays, key=lambda item: item.GetName() or ''):
                        attributes.AddArray(item)
                    if tensor_name:
                        attributes.SetActiveTensors(tensor_name)
            tracts.append((tract_name, tract))
    return tracts


def find_registration_atlas(atlas_directory):
    """Accept a target file, ORG parent/folder, or a generated atlas.

    Prefer the official ORG registration target over labeling tractography.
    Only the known ORG sibling is searched outside the selected directory.
    """
    directory = _atlas_path(atlas_directory)
    if os.path.isfile(directory):
        if os.path.splitext(directory)[1].lower() in ('.vtk', '.vtp'):
            return directory
        directory = os.path.dirname(directory)
    targets = ('registration_atlas.vtk', 'registration_atlas.vtp')
    for candidate in (directory,
                      os.path.join(directory, 'ORG-RegAtlas-100HCP'),
                      os.path.join(os.path.dirname(directory), 'ORG-RegAtlas-100HCP')):
        for name in targets:
            path = os.path.join(candidate, name)
            if os.path.isfile(path):
                return path
    hit = _find_atlas_file(directory, targets)
    return hit if hit else find_atlas_tractography(directory)


# ---------------------------------------------------------------------------
# Cluster merging + MRML creation, following
# bin/wm_append_clusters.py and bin/wm_create_mrml_file.py
# ---------------------------------------------------------------------------

def append_polydatas(pd_list):
    """Merge polydatas into one, adding a per-point 'cluster_idx' array
    holding the position of each input within the list (0-based), the
    same convention as wm_append_clusters.py."""

    import vtk
    from vtk.util import numpy_support
    import numpy

    appender = vtk.vtkAppendPolyData()
    for c_idx, pd in enumerate(pd_list):
        source = pd
        pd = vtk.vtkPolyData()
        pd.ShallowCopy(source)
        values = numpy.full(pd.GetPoints().GetNumberOfPoints(), c_idx,
                            dtype=numpy.int32)
        arr = numpy_support.numpy_to_vtk(values, deep=True)
        arr.SetName('cluster_idx')
        pd.GetPointData().AddArray(arr)
        appender.AddInputData(pd)
    appender.Update()
    return appender.GetOutput()


def _list_cluster_files(input_directory):
    import glob
    files = sorted(glob.glob(os.path.join(input_directory, 'cluster_*.vtk')) +
                   glob.glob(os.path.join(input_directory, 'cluster_*.vtp')))
    return files


def _parse_tract_mrml_members(mrml_filename):
    """cluster file names listed in a tract MRML file (T_*.mrml)."""

    members = []
    with open(mrml_filename) as f:
        for line in f:
            idx = line.find('.vtp')
            if idx > 0:
                members.append(line[idx - 13:idx + 4])
    return members


def append_cluster_files(input_directory, output_directory,
                         appended_name='appended_tract', cluster_list=None,
                         tract_mrml=None):
    """Merge cluster files of a directory into one fiber tract
    (wm_append_clusters.py, in memory).

    Member selection, in the same priority as the CLI:
      cluster_list (1-based cluster numbers) XOR tract_mrml (T_*.mrml
      listing the members); when neither is given, all cluster files of
      the directory are appended.

    Returns (merged_polydata, output_path, member_count).
    """

    import glob
    import whitematteranalysis as wma

    if cluster_list is not None and tract_mrml is not None:
        raise ValueError("Only one of cluster_list and tract_mrml can be provided.")

    if cluster_list is not None:
        members = ['cluster_%05d.vtp' % c for c in cluster_list]
    elif tract_mrml is not None:
        members = _parse_tract_mrml_members(tract_mrml)
    else:
        members = [os.path.basename(f) for f in
                   sorted(glob.glob(os.path.join(input_directory, '*.vtk')) +
                          glob.glob(os.path.join(input_directory, '*.vtp')))]

    pd_list = []
    used_members = []
    for member in members:
        fname = os.path.join(input_directory, member)
        if not os.path.isfile(fname):
            continue
        pd_list.append(wma.io.read_polydata(fname))
        used_members.append(member)
    if not pd_list:
        raise IOError("No cluster files found to append in %s" % input_directory)

    merged = append_polydatas(pd_list)

    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
    output_path = os.path.join(output_directory, appended_name + '.vtp')
    wma.io.write_polydata(merged, output_path)
    return merged, output_path, len(used_members)


def create_mrml_file(input_directory, mrml_name='scene.mrml'):
    """Create a scene MRML file referencing every vtk/vtp tractography
    in a directory, with the same jet-like per-file colormap as
    wm_create_mrml_file.py. Returns the MRML file path."""

    import glob
    import numpy
    import whitematteranalysis as wma

    files = sorted(glob.glob(os.path.join(input_directory, '*.vtk')) +
                   glob.glob(os.path.join(input_directory, '*.vtp')))
    number_of_files = len(files)
    if number_of_files == 0:
        raise IOError("No vtk/vtp files found in %s" % input_directory)

    if number_of_files == 1:
        colors = [[0., 254., 255.]]
    else:
        # the exact 'hacked colormap' of wm_create_mrml_file.py
        step = int(100 * 255.0 / (number_of_files - 1))
        R = numpy.array(list(range(0, 100 * 255 + 1, step))) / 100.0
        G = numpy.abs(list(range(100 * -127, 100 * 128 + 1, step))) * 2.0 / 100.0
        B = numpy.array(list(range(100 * 255 + 1, 0, -step))) / 100.0
        colors = [[R[i], G[i], B[i]] for i in range(number_of_files)]

    mrml_path = os.path.join(input_directory, mrml_name)
    wma.mrml.write(files, numpy.array(colors), mrml_path, ratio=1.0)
    return mrml_path


# ---------------------------------------------------------------------------
# Groupwise (population) registration, following
# bin/wm_register_multisubject_faster.py
# ---------------------------------------------------------------------------

def register_groupwise(subject_directory, output_directory,
                       fibers_per_subject=10000, fiber_length_mm=20.0,
                       fiber_length_max_mm=260.0, backend='auto',
                       random_seed=None, parallel_jobs=1, params=None,
                       progress_callback=None, mode='affine', verbose=False,
                       save_transforms=True):
    """Unbiased groupwise registration of a directory of subject
    tractographies (congeal_multisubject.MultiSubjectRegistration),
    using the wm_register_multisubject_faster.py presets or custom settings.

    Returns dict(output_directory, n_subjects, iterations_run,
    backend_name). Outputs are written by the upstream class:
    output_tractography/*_reg.vtk, transforms and
    registration_atlas.vtk under `output_directory`.
    """

    import whitematteranalysis as wma

    backend_name = resolve_backend(backend)
    os.makedirs(output_directory, exist_ok=True)
    p = registration_parameters(mode, params, groupwise=True)

    register = wma.congeal_multisubject.MultiSubjectRegistration()
    register.output_directory = output_directory
    register.input_directory = subject_directory
    register.verbose = verbose
    register.parallel_jobs = parallel_jobs
    register.render = False
    register.mode = p['modes'][0]
    register.fiber_length = fiber_length_mm
    register.fiber_length_max = fiber_length_max_mm
    register.random_seed = random_seed
    register.points_per_fiber = p['points_per_fiber']

    input_pds, subject_ids = wma.io.read_and_preprocess_polydata_directory(
        subject_directory, fiber_length_mm, fibers_per_subject,
        random_seed, fiber_length_max_mm)
    n_subjects = len(input_pds)
    if progress_callback:
        progress_callback("Registered %d subjects for optimization" % n_subjects, 10)
    for pd, sid in zip(input_pds, subject_ids):
        register.add_polydata(pd, sid)

    n_scales = len(p['sigma'])
    total_iterations = sum(p['iterations'])
    iteration = 0
    for scale in range(n_scales):
        configure_scale(register, p, scale)
        for _it in range(p['iterations'][scale]):
            if progress_callback:
                progress_callback(
                    "Groupwise registration: scale %d/%d, iteration %d/%d" % (
                        scale + 1, n_scales, iteration + 1, total_iterations),
                    10 + 80 * iteration // max(1, total_iterations))
            register.iterate()
            iteration += 1

    if progress_callback:
        progress_callback("Saving registered tractographies...", 95)
    register.save_transformed_polydatas(save_transforms=save_transforms)

    return dict(output_directory=output_directory,
                n_subjects=n_subjects,
                iterations_run=iteration,
                transforms=register.transforms, subject_ids=subject_ids,
                backend_name=backend_name)
