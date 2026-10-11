# -*- coding: utf-8 -*-

"""WMAtlasLabelGPUCore

Pure-python core of the WhiteMatterAnalysis atlas-labeling pipeline,
shared by the 3D Slicer module (WMAtlasLabelGPU.py) and headless use.
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


def load_atlas(atlas_directory):
    """Load a WMA atlas. `atlas_directory` must contain atlas.p
    (e.g. <run>/iteration_00001/initial_clusters/)."""

    import whitematteranalysis as wma
    atlas_dir = os.path.abspath(atlas_directory)
    if not os.path.isfile(os.path.join(atlas_dir, 'atlas.p')):
        raise IOError(
            "atlas.p not found in %s - point to the iteration_XXXXXX/"
            "initial_clusters directory of an atlas built with "
            "wm_cluster_atlas.py" % atlas_dir)
    return wma.cluster.load_atlas(atlas_dir, 'atlas')


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

AFFINE_DEFAULTS = {
    'sigma': [20, 10, 10, 5, 5],
    'iterations': [3, 3, 3, 3, 3],
    'maxfun': [50, 80, 80, 80, 100],
    'mean_brain_size': [2000, 3000, 3000, 3500, 4000],
    'subject_brain_size': [1000, 1500, 1500, 1500, 1500],
    'initial_step': [10, 8, 5, 3, 2],
    'final_step': [2, 2, 1, 1, 1],
    'points_per_fiber': 10,
}


def find_atlas_tractography(atlas_directory):
    """Locate the atlas tractography polydata file (atlas.vtp/vtk)
    inside an atlas directory, searching common layouts."""

    import whitematteranalysis as wma
    atlas_directory = os.path.abspath(atlas_directory)

    candidates = [os.path.join(atlas_directory, 'atlas.vtp'),
                  os.path.join(atlas_directory, 'atlas.vtk')]
    for root, _dirs, files in os.walk(atlas_directory):
        if 'atlas.vtp' in files:
            candidates.insert(0, os.path.join(root, 'atlas.vtp'))
            break
        if 'atlas.vtk' in files:
            candidates.insert(0, os.path.join(root, 'atlas.vtk'))
            break
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise IOError("No atlas tractography (atlas.vtp / atlas.vtk) found in %s"
                  % atlas_directory)


def register_subject_to_atlas(subject_polydata, atlas_polydata,
                              fiber_length_mm=40.0, fiber_length_max_mm=260.0,
                              backend='auto', random_seed=1000, params=None,
                              progress_callback=None):
    """Affine registration of a subject tractography to an atlas
    tractography (congeal_to_atlas.SubjectToAtlasRegistration), with
    the wm_register_to_atlas_new.py affine parameter defaults.

    Returns dict(transform=vtkTransform mapping subject->atlas,
                 transform_array=the 15 parameter vector,
                 objectives=per-iteration objective values).
    """

    import whitematteranalysis as wma

    import tempfile

    backend_name = resolve_backend(backend)
    p = dict(AFFINE_DEFAULTS)
    if params:
        p.update(params)

    register = wma.congeal_to_atlas.SubjectToAtlasRegistration()
    register.output_directory = tempfile.mkdtemp(prefix='wma_register_')
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
        register.sigma = p['sigma'][scale]
        register.initial_step = p['initial_step'][scale]
        register.final_step = p['final_step'][scale]
        register.maxfun = p['maxfun'][scale]
        register.mean_brain_size = p['mean_brain_size'][scale]
        register.subject_brain_size = p['subject_brain_size'][scale]
        for _it in range(p['iterations'][scale]):
            if progress_callback:
                progress_callback(
                    "Registration scale %d/%d, iteration %d/%d" % (
                        scale + 1, n_scales, iteration + 1, total_iterations),
                    40 * iteration // max(1, total_iterations))
            register.iterate()
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

    use_gpu = wma.gpu.use_gpu()
    cpu_fast = wma.gpu.config.cpu_fast_enabled()
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

    atlas_directory = os.path.abspath(atlas_directory)
    best = None
    for root, _dirs, files in os.walk(atlas_directory):
        if 'cluster_00001.vtp' in files:
            n = sum(1 for f in files if f.startswith('cluster_'))
            if best is None or n > best[1]:
                best = (root, n)
    if best is None:
        return None
    return best[0]


# ---------------------------------------------------------------------------
# Atlas-based outlier removal, following bin/wm_cluster_remove_outliers.py
# ---------------------------------------------------------------------------

def remove_cluster_outliers(subject_polydata, cluster_idx, cluster_count,
                            atlas_cluster_directory, outlier_sigma=20.0,
                            std_threshold=2.0, distance_method='Mean',
                            bilateral=True, backend='auto',
                            progress_callback=None):
    """Reject subject fibers whose similarity to their own cluster is
    more than `std_threshold` standard deviations below the cluster
    mean, following bin/wm_cluster_remove_outliers.py.

    Returns (fiber_mask, rejected_count): fiber_mask is 1 for kept
    fibers, in input fiber order.
    """

    import numpy
    import whitematteranalysis as wma

    backend_name = resolve_backend(backend)
    use_gpu = wma.gpu.use_gpu()
    cpu_fast = wma.gpu.config.cpu_fast_enabled()
    cluster_local_sigmasq = outlier_sigma * outlier_sigma

    n_fibers = subject_polydata.GetNumberOfLines()
    fiber_mask = numpy.zeros(n_fibers)
    rejected_total = 0

    for c in range(cluster_count):
        if progress_callback:
            progress_callback("Outlier removal: cluster %d/%d" % (c + 1, cluster_count),
                              60 + 35 * c // max(1, cluster_count))
        atlas_fname = os.path.join(atlas_cluster_directory,
                                   'cluster_%05d.vtp' % (c + 1))
        if not os.path.isfile(atlas_fname):
            fiber_mask[cluster_idx == c] = 1
            continue
        pd_atlas = wma.io.read_polydata(atlas_fname)
        number_fibers_in_atlas_cluster = pd_atlas.GetNumberOfLines()
        subject_sel = (cluster_idx == c)
        if number_fibers_in_atlas_cluster == 0 or not subject_sel.any():
            fiber_mask[subject_sel] = 1
            continue

        pd_subject_c = wma.filter.mask(subject_polydata, subject_sel,
                                       verbose=False)

        kwargs = dict(bilateral=bilateral, distance_method=distance_method)
        if use_gpu:
            cluster_distances = wma.cluster._pairwise_distance_matrix(
                pd_atlas, 0.0, number_of_jobs=1,
                sigmasq=cluster_local_sigmasq, use_gpu=True, **kwargs)
        elif cpu_fast:
            cluster_distances = wma.gpu.cpu_backend.pairwise_distance_matrix(
                pd_atlas, 0.0, distance_method=distance_method,
                bilateral=bilateral, sigmasq=cluster_local_sigmasq)
        else:
            cluster_distances = wma.cluster._pairwise_distance_matrix(
                pd_atlas, 0.0, number_of_jobs=1, **kwargs)
        cluster_similarity_atlas = wma.similarity.distance_to_similarity(
            cluster_distances, cluster_local_sigmasq)

        if use_gpu:
            cluster_distances_sub = wma.cluster._rectangular_distance_matrix(
                pd_subject_c, pd_atlas, 0.0, number_of_jobs=1,
                bilateral=bilateral, distance_method=distance_method,
                use_gpu=True)
        elif cpu_fast:
            cluster_distances_sub = wma.gpu.cpu_backend.rectangular_distance_matrix(
                pd_subject_c, pd_atlas, 0.0, distance_method=distance_method,
                bilateral=bilateral)
        else:
            cluster_distances_sub = wma.cluster._rectangular_distance_matrix(
                pd_subject_c, pd_atlas, 0.0, number_of_jobs=1,
                bilateral=bilateral, distance_method=distance_method)
        cluster_similarity_subject = wma.similarity.distance_to_similarity(
            cluster_distances_sub, cluster_local_sigmasq)

        # p(f1) = sum over atlas f2 of p(f1|f2), normalized by cluster size
        total_similarity_atlas = (numpy.sum(cluster_similarity_atlas, axis=1)
                                  - 1.0) / number_fibers_in_atlas_cluster
        total_similarity_subject = numpy.sum(cluster_similarity_subject,
                                             axis=0) / number_fibers_in_atlas_cluster

        mean_sim_atlas = numpy.mean(total_similarity_atlas)
        cluster_std_atlas = numpy.std(total_similarity_atlas)
        cutoff = mean_sim_atlas - std_threshold * cluster_std_atlas

        keep_c = total_similarity_subject >= cutoff
        sel_idx = numpy.nonzero(subject_sel)[0]
        fiber_mask[sel_idx[keep_c]] = 1
        rejected_total += int((~keep_c).sum())

        if progress_callback:
            progress_callback("  cluster %d: rejecting %d/%d fibers" % (
                c + 1, int((~keep_c).sum()), int(subject_sel.sum())), None)

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
                                    output_directory=None):
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

    import glob
    hits = []
    for root, _dirs, files in os.walk(os.path.abspath(atlas_directory)):
        for f in files:
            if f.startswith('T_') and f.endswith('.mrml'):
                hits.append(os.path.join(root, f))
    hits.sort()
    return hits if hits else None


def append_clusters_to_anatomical_tracts(pd_c_list, atlas_mrml_files,
                                         separated, output_directory=None):
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
    for mrml_path in atlas_mrml_files:
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


def find_registration_atlas(atlas_directory):
    """Locate the registration target tractography. The official ORG
    atlases register subjects to ORG-RegAtlas-100HCP/
    registration_atlas.vtk, which is a SIBLING directory of the
    labeling atlas folder (ORG-800FC-100HCP), NOT the labeling
    atlas.vtp. Search order:

      1. registration_atlas.vtk/.vtp inside the atlas directory tree
      2. registration_atlas.vtk/.vtp under the parent directory
         (covers the official AtlasBaseFolder/ORG-RegAtlas-100HCP layout)
      3. fallback: the labeling atlas tractography (atlas.vtp/.vtk),
         for self-built atlases that ship no RegAtlas
    """

    def _find_in(root, names):
        for walk_root, _dirs, files in os.walk(root):
            for name in names:
                if name in files:
                    return os.path.join(walk_root, name)
        return None

    atlas_directory = os.path.abspath(atlas_directory)
    targets = ('registration_atlas.vtk', 'registration_atlas.vtp')

    hit = _find_in(atlas_directory, targets)
    if hit:
        return hit

    parent = os.path.dirname(atlas_directory)
    hit = _find_in(parent, targets)
    if hit:
        return hit

    return find_atlas_tractography(atlas_directory)


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

GROUPWISE_AFFINE_DEFAULTS = {
    'sigma': [20, 10, 10, 5, 5],
    'iterations': [3, 3, 3, 3, 3],
    'maxfun': [50, 80, 80, 80, 100],
    'mean_brain_size': [2000, 3000, 3000, 3500, 4000],
    'subject_brain_size': [1000, 1500, 1500, 1500, 1500],
    'initial_step': [10, 8, 5, 3, 2],
    'final_step': [2, 2, 1, 1, 1],
    'points_per_fiber': 10,
}


def register_groupwise(subject_directory, output_directory,
                       fibers_per_subject=10000, fiber_length_mm=20.0,
                       fiber_length_max_mm=260.0, backend='auto',
                       random_seed=None, parallel_jobs=1, params=None,
                       progress_callback=None):
    """Unbiased groupwise registration of a directory of subject
    tractographies (congeal_multisubject.MultiSubjectRegistration),
    affine mode with the wm_register_multisubject_faster.py defaults.

    Returns dict(output_directory, n_subjects, iterations_run,
    backend_name). Outputs are written by the upstream class:
    output_tractography/*_reg.vtk, transforms and
    registration_atlas.vtk under `output_directory`.
    """

    import whitematteranalysis as wma

    backend_name = resolve_backend(backend)
    os.makedirs(output_directory, exist_ok=True)
    p = dict(GROUPWISE_AFFINE_DEFAULTS)
    if params:
        p.update(params)

    register = wma.congeal_multisubject.MultiSubjectRegistration()
    register.output_directory = output_directory
    register.input_directory = subject_directory
    register.verbose = False
    register.parallel_jobs = parallel_jobs
    register.render = False
    register.mode = "Affine"
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
        register.sigma = p['sigma'][scale]
        register.initial_step = p['initial_step'][scale]
        register.final_step = p['final_step'][scale]
        register.maxfun = p['maxfun'][scale]
        register.mean_brain_size = p['mean_brain_size'][scale]
        register.subject_brain_size = p['subject_brain_size'][scale]
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
    register.save_transformed_polydatas()

    return dict(output_directory=output_directory,
                n_subjects=n_subjects,
                iterations_run=iteration,
                backend_name=backend_name)
