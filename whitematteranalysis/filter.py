# -*- coding: utf-8 -*-

""" filter.py

Swiss army knife of fiber polydata processing.

This module provides filtering functions that operate on vtkPolyData
objects containing whole-brain tractography.

Preprocess by removing fibers based on length, or endpoint
distance. Downsample. Mask.  Create symmetrized test data.

preprocess
downsample
mask
symmetrize
remove_hemisphere
remove_outliers

"""

import os

import vtk
import numpy

try:
    from joblib import Parallel, delayed
    USE_PARALLEL = 1
except ImportError:
    USE_PARALLEL = 0
    print(f"<{os.path.basename(__file__)}> Failed to import joblib, cannot multiprocess.")
    print(f"<{os.path.basename(__file__)}> Please install joblib for this functionality.")

from . import fibers, similarity, gpu

verbose = 0


def _lines_offsets_connectivity(inpd):
    """Fast access to the line structure of a polydata (VTK >= 9).

    Returns (offsets, connectivity, native points) views for reading, or None
    when the VTK connectivity API is unavailable.
    """

    try:
        from vtk.util import numpy_support
        lines = inpd.GetLines()
        offsets = numpy_support.vtk_to_numpy(lines.GetOffsetsArray()).astype(numpy.int64, copy=False)
        conn = numpy_support.vtk_to_numpy(lines.GetConnectivityArray()).astype(numpy.int64, copy=False)
        # Arithmetic callers upcast only the coordinates they actually use.
        points = numpy_support.vtk_to_numpy(inpd.GetPoints().GetData())
        return offsets, conn, points
    except (AttributeError, ImportError):
        return None


def _gather_conn(conn, offsets, keep_indices):
    """Connectivity array of the selected lines, preserving order."""

    starts = offsets[:-1][keep_indices]
    lengths = offsets[1:][keep_indices] - starts
    total = int(lengths.sum())
    if total == 0:
        return numpy.zeros(0, dtype=numpy.int64), numpy.zeros(keep_indices.size + 1, dtype=numpy.int64)
    out_offsets = numpy.zeros(keep_indices.size + 1, dtype=numpy.int64)
    numpy.cumsum(lengths, out=out_offsets[1:])
    # Build source IDs in line batches instead of allocating two full-size
    # temporary index arrays for a whole-brain tractogram.
    selected = numpy.empty(total, dtype=numpy.int64)
    for first in range(0, keep_indices.size, 4096):
        last = min(first + 4096, keep_indices.size)
        begin, end = out_offsets[first], out_offsets[last]
        idx = numpy.arange(end - begin, dtype=numpy.int64)
        idx += numpy.repeat(starts[first:last] - out_offsets[first:last] + begin,
                            lengths[first:last])
        selected[begin:end] = conn[idx]
    return selected, out_offsets


def _select_vtk_array(values, indices, dtype):
    """Gather into VTK-owned storage with bounded temporary NumPy arrays."""
    from vtk.util import numpy_support

    output = vtk.vtkDataArray.CreateDataArray(numpy_support.get_vtk_array_type(dtype))
    output.SetNumberOfComponents(values.shape[1] if values.ndim == 2 else 1)
    output.SetNumberOfTuples(indices.size)
    target = numpy_support.vtk_to_numpy(output).reshape((indices.size,) + values.shape[1:])
    for first in range(0, indices.size, 262144):
        last = min(first + 262144, indices.size)
        target[first:last] = values[indices[first:last]]
    return output


def _mean_nearest_neighbor_distance(distances):
    """Mean of the three smallest non-self distances in each row.

    With fewer than four fibers, use all available non-self neighbors. A
    lone fiber has no neighbors and is treated as an outlier.
    """

    distances = numpy.asarray(distances)
    number_of_fibers = distances.shape[-1]
    output_shape = distances.shape[:-1]
    if number_of_fibers == 0:
        return numpy.empty(0, dtype=numpy.float64)
    if number_of_fibers == 1:
        return numpy.full(output_shape, numpy.inf, dtype=numpy.float64)

    count = min(3, number_of_fibers - 1)
    nearest = numpy.partition(distances, count, axis=-1)[..., :count + 1]
    nearest.sort(axis=-1)
    neighbors = nearest[..., 1:]
    if count == 3:
        # Keep the original left-to-right addition order for exact cutoff parity.
        total = (neighbors[..., 0] + neighbors[..., 1]) + neighbors[..., 2]
        return numpy.asarray(total, dtype=numpy.float64) / 3.0
    return neighbors.mean(axis=-1, dtype=numpy.float64)


def add_point_data_array(inpd, data, array_name):
    """Make a vtk point data array from input data array and add to inpd.

    Input data must have dimensions of the number of lines in the
    input polydata. The output array will be added to the polydata and
    will be point data, so the per-line values will be duplicated to
    become per-point values (each point on the line) for visualization.
    """

    fast = _lines_offsets_connectivity(inpd)
    if fast is not None:
        offsets, conn, points = fast
        line_lengths = offsets[1:] - offsets[:-1]
        from vtk.util import numpy_support
        values = numpy.repeat(numpy.asarray(data), line_lengths).astype(numpy.float32)
        outarray = numpy_support.numpy_to_vtk(values, deep=True)
        outarray.SetName(array_name)
        inpd.GetPointData().AddArray(outarray)
        return

    ptids = vtk.vtkIdList()
    inpd.GetLines().InitTraversal()
    outarray = vtk.vtkFloatArray()
    outarray.SetName(array_name)
    for lidx in range(0, inpd.GetNumberOfLines()):
        inpd.GetLines().GetNextCell(ptids)
        for pidx in range(0, ptids.GetNumberOfIds()):
            outarray.InsertNextTuple1(data[lidx])
    inpd.GetPointData().AddArray(outarray)

def flatten_length_distribution(inpd, min_length_mm=None, max_length_mm=None, num_bins=10, fibers_per_bin=1000, verbose=True):
    """In order to cluster all structures without removing more
    prevalent shorter ones, sample equal numbers across length ranges.

    This should enable clustering of smaller u-fibers than previously
    possible, while incorporating information about longer fibers they are
    near for stability.
    """

    # try to get N fibers in each length range ("bin")
    # First calculate the ranges using input or measured max/min lengths
    if (max_length_mm is None) or (min_length_mm is None):
        # call preprocess just to get the lengths measured
        inpd2, lengths, step_size = preprocess(inpd, 0.0, max_length_mm=max_length_mm, verbose=False, return_lengths=True)
        if max_length_mm is None:
            max_length_mm = numpy.max(lengths)
        if min_length_mm is None:
            min_length_mm = numpy.min(lengths)

    increment = (max_length_mm - min_length_mm) / (num_bins - 1)
    bin_ends = list()
    max_l = min_length_mm
    while max_l <= max_length_mm:
        bin_ends.append(max_l)
        max_l += increment
    if verbose:
        print("Bins/length ranges:", bin_ends)

    print(bin_ends[0:-1], bin_ends[1:])

    # append the sampled fibers together into a new polydata
    appender = vtk.vtkAppendPolyData()
    for (bin_low, bin_hi) in zip (bin_ends[0:-1], bin_ends[1:]):
        pd = preprocess(inpd, bin_low, max_length_mm=bin_hi, verbose=False)
        pd2 = downsample(pd, fibers_per_bin,verbose=False)
        if verbose:
            print(pd2.GetNumberOfLines(), "fibers in length range:", [bin_low, bin_hi])
        if (vtk.vtkVersion().GetVTKMajorVersion() >= 6.0):
            appender.AddInputData(pd2)
        else:
            appender.AddInput(pd2)
    appender.Update()
    return(appender.GetOutput())

def compute_lengths(inpd):
    """Compute length of each fiber in polydata. Returns lengths and step size.

    Step size is estimated using points in the middle of a fiber with over 15 points.

    """

    # Make sure we have lines and points.
    if (inpd.GetNumberOfLines() == 0) or (inpd.GetNumberOfPoints() == 0):
        print(f"<{os.path.basename(__file__)}> No fibers found in input polydata.")
        return 0, 0
    
    # measure step size (using first line that has >=5 points)
    fast = _lines_offsets_connectivity(inpd)
    if fast is not None:
        offsets, conn, points = fast
        line_lengths = offsets[1:] - offsets[:-1]
        candidates = numpy.nonzero(line_lengths >= 5)[0]
        if candidates.size == 0:
            return 0, 0
        c = candidates[0]
        # interior segments (ptidx, ptidx+1) for ptidx in 1..n-2, as in the
        # original loop; (p0-p1)**2 is identical to (p1-p0)**2
        seg0 = points[conn[offsets[c] + 1:offsets[c + 1] - 1]].astype(numpy.float64, copy=False)
        seg1 = points[conn[offsets[c] + 2:offsets[c + 1]]].astype(numpy.float64, copy=False)
        seg_sq = numpy.power(numpy.subtract(seg0, seg1), 2)
        seg_dists = numpy.sqrt(numpy.sum(seg_sq, axis=1))
        # sequential accumulation (cumsum) matches the += loop bitwise
        step_size = seg_dists.cumsum()[-1] / float(seg_dists.size)
        fiber_lengths = line_lengths.astype(numpy.float64) * step_size
        return fiber_lengths, step_size

    # measure step size (using first line that has >=5 points)
    cell_idx = 0
    ptids = vtk.vtkIdList()
    inpoints = inpd.GetPoints()
    inpd.GetLines().InitTraversal()
    while (ptids.GetNumberOfIds() < 5) & (cell_idx < inpd.GetNumberOfLines()):
        inpd.GetLines().GetNextCell(ptids)
        ##    inpd.GetLines().GetCell(cell_idx, ptids)
        ## the GetCell function is not wrapped in Canopy python-vtk
        cell_idx += 1
    # make sure we have some points along this fiber
    # In case all fibers in the brain are really short, treat it the same as no fibers.
    if ptids.GetNumberOfIds() < 5:
        return 0, 0
    
    # Use points from the middle of the fiber to estimate step length.
    # This is because the step size may vary near endpoints (in order to include
    # endpoints when downsampling the fiber to reduce file size).
    step_size = 0.0
    count = 0.0
    for ptidx in range(1, ptids.GetNumberOfIds()-1):
        point0 = inpoints.GetPoint(ptids.GetId(ptidx))
        point1 = inpoints.GetPoint(ptids.GetId(ptidx + 1))
        step_size += numpy.sqrt(numpy.sum(numpy.power(numpy.subtract(point0, point1), 2)))
        count += 1
    step_size = step_size / count

    fiber_lengths = list()
    # loop over lines
    inpd.GetLines().InitTraversal()
    num_lines = inpd.GetNumberOfLines()
    for lidx in range(0, num_lines):
        inpd.GetLines().GetNextCell(ptids)
        # save length
        fiber_lengths.append(ptids.GetNumberOfIds() * step_size)

    return numpy.array(fiber_lengths), step_size

def preprocess(inpd, min_length_mm,
               remove_u=False,
               remove_u_endpoint_dist=40,
               remove_brainstem=False,
               return_indices=False,
               return_lengths=False,
               preserve_point_data=False,
               preserve_cell_data=False,
               verbose=True, max_length_mm=None):
    """Remove fibers below a length threshold and using other criteria (optional).

    Based on fiber length, and optionally on distance between
    endpoints (u-shape has low distance), and inferior location
    (likely in brainstem).

    """

    # Make sure we have lines and points.
    if (inpd.GetNumberOfLines() == 0) or (inpd.GetNumberOfPoints() == 0):
        print(f"<{os.path.basename(__file__)}> No fibers found in input polydata.")
        if return_indices:
            if return_lengths:
                return inpd, 0, 0, 0
            else:
                return inpd, 0
        else:
            if return_lengths:
                return inpd, 0, 0
            else:
                return inpd

    
    line_indices, fiber_lengths, step_size = _preprocess_indices(
        inpd, min_length_mm, remove_u=remove_u,
        remove_u_endpoint_dist=remove_u_endpoint_dist,
        remove_brainstem=remove_brainstem, max_length_mm=max_length_mm,
        verbose=verbose)
    num_lines = inpd.GetNumberOfLines()
    fiber_mask = numpy.zeros(num_lines)
    fiber_mask[line_indices] = 1
    outpd = mask(inpd, fiber_mask, preserve_point_data=preserve_point_data, preserve_cell_data=preserve_cell_data, verbose=verbose)

    if return_indices:
        if return_lengths:
            return outpd, numpy.array(line_indices), fiber_lengths, step_size
        else:
            return outpd, numpy.array(line_indices)
    else:
        if return_lengths:
            return outpd, fiber_lengths, step_size
        else:
            return outpd


def _preprocess_indices(inpd, min_length_mm, remove_u=False,
                        remove_u_endpoint_dist=40, remove_brainstem=False,
                        max_length_mm=None, verbose=False):
    """Return the indices and lengths selected by :func:`preprocess`."""
    if inpd.GetNumberOfLines() == 0 or inpd.GetNumberOfPoints() == 0:
        return numpy.empty(0, dtype=numpy.int64), 0, 0

    fiber_lengths, step_size = compute_lengths(inpd)
    min_length_pts = round(min_length_mm / float(step_size))
    if verbose:
        print(f"<{os.path.basename(__file__)}> Minimum length", min_length_mm, \
            "mm. Tractography step size * minimum number of points =", step_size, "*", min_length_pts, ")")

    fast = _lines_offsets_connectivity(inpd)
    if fast is not None:
        offsets, conn, points = fast
        line_lengths = offsets[1:] - offsets[:-1]
        keep = line_lengths > min_length_pts
        if max_length_mm is not None:
            keep &= ~(line_lengths.astype(numpy.float64) * step_size > max_length_mm)
        if remove_u | remove_brainstem:
            point0 = points[conn[offsets[:-1]]].astype(numpy.float64, copy=False)
            point1 = points[conn[offsets[1:] - 1]].astype(numpy.float64, copy=False)
            if remove_u:
                endpoint_dist = numpy.sqrt(
                    numpy.sum(numpy.power(numpy.subtract(point0, point1), 2), axis=1))
                keep &= ~(endpoint_dist < remove_u_endpoint_dist)
            if remove_brainstem:
                mean_sup_inf = (point0[:, 2] + point1[:, 2]) / 2
                keep &= ~(mean_sup_inf < -40)
        return numpy.nonzero(keep)[0], fiber_lengths, step_size

    ptids = vtk.vtkIdList()
    inpoints = inpd.GetPoints()
    inpd.GetLines().InitTraversal()
    line_indices = []
    for lidx in range(inpd.GetNumberOfLines()):
        inpd.GetLines().GetNextCell(ptids)
        keep_curr_fiber = ptids.GetNumberOfIds() > min_length_pts
        if keep_curr_fiber and max_length_mm is not None:
            keep_curr_fiber = not (ptids.GetNumberOfIds() * step_size > max_length_mm)
        if keep_curr_fiber and (remove_u or remove_brainstem):
            point0 = inpoints.GetPoint(ptids.GetId(0))
            point1 = inpoints.GetPoint(ptids.GetId(ptids.GetNumberOfIds() - 1))
            if remove_u:
                endpoint_dist = numpy.sqrt(numpy.sum(numpy.power(
                    numpy.subtract(point0, point1), 2)))
                if endpoint_dist < remove_u_endpoint_dist:
                    keep_curr_fiber = False
            if remove_brainstem:
                mean_sup_inf = (point0[2] + point1[2]) / 2
                if mean_sup_inf < -40:
                    keep_curr_fiber = False
        if keep_curr_fiber:
            line_indices.append(lidx)
    return numpy.asarray(line_indices, dtype=numpy.int64), fiber_lengths, step_size

def downsample(inpd, output_number_of_lines, return_indices=False, preserve_point_data=False, preserve_cell_data=True, initial_indices=None, verbose=True, random_seed=1234):
    """ Random (down)sampling of fibers without replacement. """

    if initial_indices is None:
        num_lines = inpd.GetNumberOfLines()
    else:
        num_lines = len(initial_indices)

    if num_lines < output_number_of_lines:
        # fewer fibers than requested: return the input unchanged (also
        # honoring return_indices, which the early return used to ignore)
        if return_indices:
            if initial_indices is None:
                return inpd, numpy.arange(num_lines)
            return inpd, numpy.sort(numpy.asarray(initial_indices))
        return inpd

    # use the input random seed every time for code testing experiments
    if random_seed is not None:
        if verbose:
            print(f"<{os.path.basename(__file__)}> Setting random seed to", random_seed)
        numpy.random.seed(seed=random_seed)

    # randomly pick the lines that we will keep
    line_indices = numpy.random.permutation(num_lines)
    if initial_indices is None:
        line_indices = line_indices[0:output_number_of_lines]
        fiber_mask = numpy.zeros(num_lines)
        fiber_mask[line_indices] = 1
    else:
        line_indices = initial_indices[line_indices[0:output_number_of_lines]]
        fiber_mask = numpy.zeros(inpd.GetNumberOfLines())
        fiber_mask[line_indices] = 1

    # don't color by line index by default, preserve whatever was there
    #outpd = mask(inpd, fiber_mask, fiber_mask)
    outpd = mask(inpd, fiber_mask, preserve_point_data=preserve_point_data, preserve_cell_data=preserve_cell_data, verbose=verbose)

    # final line count
    #print "<filter.py> Number of lines selected:", outpd.GetNumberOfLines()
    if return_indices:
        # return sorted indices, this is the line ordering of output
        # polydata (because we mask rather than changing input line order)
        return outpd, numpy.sort(line_indices)
    else:
        return outpd


def mask(inpd, fiber_mask, color=None, preserve_point_data=False, preserve_cell_data=True, verbose=True,
         _geometry=None, _line_indices=None):
    """ Keep lines and their points where fiber_mask == 1.

     Unlike vtkMaskPolyData that samples every nth cell, this function
     uses an actual mask, and also gets rid of points that
     are not used by any cell, reducing the size of the polydata file.

     This code also sets scalar cell data to input color data.  This
     input data is expected to be 1 or 3 components.

     If there is no input cell scalar color data, existing cell
     scalars that we have created (EmbeddingColor, ClusterNumber,
     EmbeddingCoordinate) are looked for and masked as well.

     """

    inpoints = inpd.GetPoints()
    inpointdata = inpd.GetPointData()
    incelldata = inpd.GetCellData()
    
    # output and temporary objects
    ptids = vtk.vtkIdList()
    outpd = vtk.vtkPolyData()
    outlines = vtk.vtkCellArray()
    outpoints = vtk.vtkPoints()
    outcolors = None
    outpointdata = outpd.GetPointData()
    outcelldata = outpd.GetCellData()
    tensor_names = []

    if color is not None:
        # if input is RGB
        if len(color.shape) == 2:
            if color.shape[1] == 3:
                outcolors = vtk.vtkUnsignedCharArray()
                outcolors.SetNumberOfComponents(3)

        # otherwise output floats as colors
        if outcolors == None:
            outcolors = vtk.vtkFloatArray()

    # check for cell data arrays to keep
    if preserve_cell_data:
        if incelldata.GetNumberOfArrays() > 0:
            cell_data_array_indices = list(range(incelldata.GetNumberOfArrays()))            
            for idx in cell_data_array_indices:
                array = incelldata.GetArray(idx)
                dtype = array.GetDataType()
                if dtype == 10:
                    out_array = vtk.vtkFloatArray()
                elif dtype == 6:
                    out_array = vtk.vtkIntArray()
                elif dtype == 3:
                    out_array = vtk.vtkUnsignedCharArray()
                else:
                    out_array = vtk.vtkFloatArray()
                out_array.SetNumberOfComponents(array.GetNumberOfComponents())
                out_array.SetName(array.GetName())
                if verbose:
                    print("Cell data array found:", array.GetName(), array.GetNumberOfComponents())
                outcelldata.AddArray(out_array)
                # make sure some scalars are active so rendering works
                #outpd.GetCellData().SetActiveScalars(array.GetName())
                
            #if inpd.GetCellData().GetArray('ClusterNumber'):
            #    # this will be active unless we have embedding colors
            #    outpd.GetCellData().SetActiveScalars('ClusterNumber')
            #if inpd.GetCellData().GetArray('EmbeddingColor'):
            #    # Note Slicer can't display these cell scalars (all is red)
            #    outpd.GetCellData().SetActiveScalars('EmbeddingColor')

        else:
            preserve_cell_data = False

    # check for point data arrays to keep
    if preserve_point_data:
        if inpointdata.GetNumberOfArrays() > 0:
            point_data_array_indices = list(range(inpointdata.GetNumberOfArrays()))            
            for idx in point_data_array_indices:
                array = inpointdata.GetArray(idx)
                out_array = vtk.vtkFloatArray()
                out_array.SetNumberOfComponents(array.GetNumberOfComponents())
                out_array.SetName(array.GetName())
                if verbose:
                    print("Point data array found:", array.GetName(), array.GetNumberOfComponents())
                outpointdata.AddArray(out_array)
                # make sure some scalars are active so rendering works
                #outpd.GetPointData().SetActiveScalars(array.GetName())
                # keep track of tensors to choose which is active
                if array.GetNumberOfComponents() == 9:
                    tensor_names.append(array.GetName())
        else:
            preserve_point_data = False

    # Set up scalars and tensors attributes for correct visualization in Slicer.
    # Slicer works with point data and does not handle cell data well.
    # This set of changes breaks old internal wma default visualization of cell scalars. 
    # Changes must be propagated through wma so that render is called with the name of the field to visualize.
    # the new way in wma is like this line, below.
    #ren = wma.render.render(output_polydata_s, 1000, data_mode="Cell", data_name='EmbeddingColor')

    # For Slicer: First set one of the expected tensor arrays as default for vis
    tensors_labeled = False
    for name in tensor_names:
        if name == "tensors":
            outpd.GetPointData().SetTensors(outpd.GetPointData().GetArray("tensors"))
            tensors_labeled = True
        if name == "Tensors":
            outpd.GetPointData().SetTensors(outpd.GetPointData().GetArray("Tensors"))
            tensors_labeled = True
        if name == "tensor1":
            outpd.GetPointData().SetTensors(outpd.GetPointData().GetArray("tensor1"))
            tensors_labeled = True
        if name == "Tensor1":
            outpd.GetPointData().SetTensors(outpd.GetPointData().GetArray("Tensor1"))
            tensors_labeled = True
    if not tensors_labeled:
        if len(tensor_names) > 0:
            print("Data has unexpected tensor name(s). Unable to set active for visualization:", tensor_names)
    # now set cell data visualization inactive.
    outpd.GetCellData().SetActiveScalars(None)

    # fast path: vectorized line selection and array rebuild (VTK >= 9).
    # Produces the same output polydata as the per-line loop below:
    # points are re-inserted in kept-line order and cast to float32
    # (vtkPoints default), tuples are cast to the same output array types.
    fast = _geometry if _geometry is not None else _lines_offsets_connectivity(inpd)
    if fast is not None:
        from vtk.util import numpy_support
        offsets, conn, in_points = fast
        keep_indices = (numpy.asarray(_line_indices, dtype=numpy.int64) if _line_indices is not None
                        else numpy.nonzero(numpy.asarray(fiber_mask) != 0)[0])

        # the original loop re-inserts every point of every kept line,
        # so output points are consecutive: the output connectivity is
        # 0..total-1, and src_conn maps each output point to its
        # original point id (used to gather coordinates and point data)
        src_conn, out_offsets = _gather_conn(conn, offsets, keep_indices)
        total = int(out_offsets[-1]) if out_offsets.size else 0
        # points: only those referenced by kept lines, in output order
        outpoints.SetData(_select_vtk_array(in_points, src_conn, numpy.float32))

        # lines
        out_lines = vtk.vtkCellArray()
        offsets_vtk = numpy_support.numpy_to_vtk(
            out_offsets, deep=True, array_type=vtk.vtkIdTypeArray().GetDataType())
        conn_vtk = vtk.vtkIdTypeArray()
        conn_vtk.SetNumberOfValues(total)
        out_conn = numpy_support.vtk_to_numpy(conn_vtk)
        for first in range(0, total, 262144):
            last = min(first + 262144, total)
            out_conn[first:last] = numpy.arange(first, last, dtype=numpy.int64)
        out_lines.SetData(offsets_vtk, conn_vtk)
        outlines = out_lines

        # cell data arrays, cast to the same output types as the loop
        if preserve_cell_data:
            for idx in range(incelldata.GetNumberOfArrays()):
                array = incelldata.GetArray(idx)
                full = numpy_support.vtk_to_numpy(array)
                dtype = array.GetDataType()
                if dtype == 10:
                    output_dtype = numpy.float32
                elif dtype == 6:
                    output_dtype = numpy.int32
                elif dtype == 3:
                    output_dtype = numpy.uint8
                else:
                    output_dtype = numpy.float32
                out_array = _select_vtk_array(full, keep_indices, output_dtype)
                out_array.SetName(array.GetName())
                outcelldata.AddArray(out_array)

        # point data arrays are per output point (always float, as the loop)
        if preserve_point_data:
            for idx in range(inpointdata.GetNumberOfArrays()):
                array = inpointdata.GetArray(idx)
                full = numpy_support.vtk_to_numpy(array)
                out_array = _select_vtk_array(full, src_conn, numpy.float32)
                out_array.SetName(array.GetName())
                outpointdata.AddArray(out_array)

        # color scalars: same array types as the loop, set as active scalars
        if color is not None:
            values = numpy.asarray(color)
            if outcolors.GetNumberOfComponents() == 3:
                color_arr = _select_vtk_array(values, keep_indices, numpy.uint8)
                color_arr.SetNumberOfComponents(3)
            else:
                color_arr = _select_vtk_array(values, keep_indices, numpy.float32)
            outpd.GetCellData().SetScalars(color_arr)

        outpd.SetPoints(outpoints)
        outpd.SetLines(outlines)
        if verbose:
            print(f"<{os.path.basename(__file__)}> Fibers sampled:", outpd.GetNumberOfLines(), "/", inpd.GetNumberOfLines())
        return outpd

    if _line_indices is not None:
        raise RuntimeError('Indexed fiber selection requires VTK 9 connectivity arrays.')

    # loop over lines
    inpd.GetLines().InitTraversal()
    outlines.InitTraversal()

    for lidx in range(0, inpd.GetNumberOfLines()):
        inpd.GetLines().GetNextCell(ptids)

        if fiber_mask[lidx]:

            if verbose:
                if lidx % 100 == 0:
                    print(f"<{os.path.basename(__file__)}> Line:", lidx, "/", inpd.GetNumberOfLines())

            # get points for each ptid and add to output polydata
            cellptids = vtk.vtkIdList()

            for pidx in range(0, ptids.GetNumberOfIds()):
                point = inpoints.GetPoint(ptids.GetId(pidx))
                idx = outpoints.InsertNextPoint(point)
                cellptids.InsertNextId(idx)
                if preserve_point_data:
                    for idx in point_data_array_indices:
                        array = inpointdata.GetArray(idx)
                        out_array = outpointdata.GetArray(idx)
                        out_array.InsertNextTuple(array.GetTuple(ptids.GetId(pidx)))

            outlines.InsertNextCell(cellptids)                    

            if color is not None:
                # this code works with either 3 or 1 component only
                if outcolors.GetNumberOfComponents() == 3:
                    outcolors.InsertNextTuple3(color[lidx,0], color[lidx,1], color[lidx,2])
                else:
                    outcolors.InsertNextTuple1(color[lidx])

            if preserve_cell_data:
                for idx in cell_data_array_indices:
                    array = incelldata.GetArray(idx)
                    out_array = outcelldata.GetArray(idx)
                    out_array.InsertNextTuple(array.GetTuple(lidx))

    # put data into output polydata
    outpd.SetLines(outlines)
    outpd.SetPoints(outpoints)

    # if we had an input color requested during masking, set that to be the default scalar for vis
    if color is not None:
        outpd.GetCellData().SetScalars(outcolors)

    if verbose:
        print(f"<{os.path.basename(__file__)}> Fibers sampled:", outpd.GetNumberOfLines(), "/", inpd.GetNumberOfLines())

    return outpd


def symmetrize(inpd):
    """Generate symmetric polydata by reflecting.

    Output polydata has twice as many lines as input.

    """

    # output and temporary objects
    ptids = vtk.vtkIdList()
    points = inpd.GetPoints()
    outpd = vtk.vtkPolyData()
    outlines = vtk.vtkCellArray()
    outpoints = vtk.vtkPoints()
    outpoints.DeepCopy(points)

    # loop over lines
    inpd.GetLines().InitTraversal()
    outlines.InitTraversal()

    # set scalar cell data to 1 for orig, -1 for reflect, for vis
    outcolors = vtk.vtkFloatArray()

    # index into end of point array
    lastidx = outpoints.GetNumberOfPoints()
    print(f"<{os.path.basename(__file__)}> Input number of points: ", lastidx)

    # loop over all lines, insert line and reflected copy into output pd
    for lidx in range(0, inpd.GetNumberOfLines()):
        # progress
        if verbose:
            if lidx % 100 == 0:
                print(f"<{os.path.basename(__file__)}> Line:", lidx, "/", inpd.GetNumberOfLines())

        inpd.GetLines().GetNextCell(ptids)

        num_points = ptids.GetNumberOfIds()

        # insert fiber (ptids are same since new points go at the end)
        outlines.InsertNextCell(ptids)
        outcolors.InsertNextTuple1(1)

        # insert reflection into END of point array and into line array
        refptids = vtk.vtkIdList()
        for pidx in range(0, num_points):

            point = points.GetPoint(ptids.GetId(pidx))

            # reflect (RAS -> reflect first value)
            refpoint = (-point[0], point[1], point[2])
            idx = outpoints.InsertNextPoint(refpoint)
            refptids.InsertNextId(idx)

        outlines.InsertNextCell(refptids)
        outcolors.InsertNextTuple1(-1)

    # put data into output polydata
    outpd.SetLines(outlines)
    outpd.SetPoints(outpoints)
    outpd.GetCellData().SetScalars(outcolors)

    return outpd


def remove_hemisphere(inpd, hemisphere=-1):
    """ Remove left (-1) or right (+1) hemisphere points. """

    # output and temporary objects
    ptids = vtk.vtkIdList()
    outpd = vtk.vtkPolyData()
    outlines = vtk.vtkCellArray()
    outlines.InitTraversal()
    outpoints = vtk.vtkPoints()

    # loop over lines
    inpd.GetLines().InitTraversal()

    # loop over all lines, inserting into output only
    # the part in the correct hemisphere
    for lidx in range(0, inpd.GetNumberOfLines()):
        # progress
        if verbose:
            if lidx % 100 == 0:
                print(f"<{os.path.basename(__file__)}> Line:", lidx, "/", inpd.GetNumberOfLines())

        inpd.GetLines().GetNextCell(ptids)

        num_points = ptids.GetNumberOfIds()

        # insert kept points into point array and into line array
        keptptids = vtk.vtkIdList()
        for pidx in range(0, num_points):

            point = inpd.GetPoints().GetPoint(ptids.GetId(pidx))

            # if we keep this point (if not in removed hemisphere)
            if ((point[0] < 0) &
                (hemisphere == 1)) | ((point[0] > 0) &
                                      (hemisphere == -1)):

                idx = outpoints.InsertNextPoint(point)
                keptptids.InsertNextIdx(idx)

        outlines.InsertNextCell(keptptids)

    # put data into output polydata
    outpd.SetLines(outlines)
    outpd.SetPoints(outpoints)

    return outpd


def remove_outliers(inpd, min_fiber_distance, n_jobs=0, distance_method ='Mean'):
    """ Remove fibers that have no other nearby fibers, i.e. outliers.

    The pairwise fiber distance matrix is computed, then fibers
    are rejected if their average neighbor distance (using closest 3
    neighbors) is higher than min_fiber_distance.

    """

    fiber_array = fibers.FiberArray()
    #fiber_array.points_per_fiber = 5
    fiber_array.points_per_fiber = 10
    fiber_array.convert_from_polydata(inpd)

    fiber_indices = list(range(0, fiber_array.number_of_fibers))

    # squared distances are computed
    min_fiber_distance = min_fiber_distance * min_fiber_distance
    
    # pairwise distance matrix
    distances_matrix = None
    if gpu.use_gpu() and gpu.distance.distance_supported(distance_method):
        print(f"<{os.path.basename(__file__)}> Computing pairwise distances on GPU.")
        distances_matrix = gpu.distance.pairwise_distance_matrix_from_fiber_array(
            fiber_array, 0, distance_method=distance_method)
    elif gpu.config.cpu_fast_enabled() and gpu.fiber_tensors.distance_supported(distance_method):
        print(f"<{os.path.basename(__file__)}> Computing pairwise distances with the vectorized CPU backend.")
        coordinates = gpu.fiber_tensors.fiber_array_to_numpy(
            fiber_array, coordinate_major=True)
        distances_matrix = gpu.cpu_backend._distance_matrix(
            coordinates, coordinates, 0, distance_method=distance_method,
            number_of_jobs=n_jobs if n_jobs > 0 else 1)
    elif USE_PARALLEL and n_jobs > 0:
        distances_matrix = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(similarity.fiber_distance)(
                fiber_array.get_fiber(lidx),
                fiber_array,
                threshold = 0,
                distance_method = distance_method)
            for lidx in fiber_indices)

        distances_matrix = numpy.array(distances_matrix)

    else:
        # do this in a loop to use less memory. then parallelization can 
        # happen over the number of subjects.
        mindist = numpy.zeros(fiber_array.number_of_fibers)
        for lidx in fiber_indices:
            distances = similarity.fiber_distance(fiber_array.get_fiber(lidx), fiber_array, 0,  distance_method = distance_method)
            mindist[lidx] = _mean_nearest_neighbor_distance(distances)

    if distances_matrix is not None:
        mindist = _mean_nearest_neighbor_distance(distances_matrix)
            
    # keep only fibers who have nearby similar fibers
    fiber_mask = mindist < min_fiber_distance

    if True:
        num_fibers = len(numpy.nonzero(fiber_mask)[0]), "/", len(fiber_mask)
        print(f"<{os.path.basename(__file__)}> Number retained after outlier removal: ", num_fibers)

    outpd = mask(inpd, fiber_mask, mindist)
    outpd_reject = mask(inpd, ~fiber_mask, mindist)

    return outpd, fiber_mask, outpd_reject

def smooth(inpd, fiber_distance_sigma = 25, points_per_fiber=30, n_jobs=2, upper_thresh=30):
    """ Average nearby fibers.
    
    The pairwise fiber distance matrix is computed, then fibers
    are averaged with their neighbors using Gaussian weighting.

    The "local density" or soft neighbor count is also output.
    """

    sigmasq = fiber_distance_sigma * fiber_distance_sigma
    
    # polydata to array conversion, fixed-length fiber representation
    current_fiber_array = fibers.FiberArray()
    current_fiber_array.points_per_fiber = points_per_fiber
    current_fiber_array.convert_from_polydata(inpd)

    # fiber list data structure initialization for easy fiber averaging
    curr_count = list()
    next_fibers = list()
    next_weights = list()
    for lidx in range(0, current_fiber_array.number_of_fibers):
        curr_count.append(1)

    fiber_indices = list(range(0, current_fiber_array.number_of_fibers))

    # compare squared distances to squared distance threshold
    upper_thresh = upper_thresh*upper_thresh
    
    print(f"<{os.path.basename(__file__)}> Computing pairwise distances...")
    
    # pairwise distance matrix
    if gpu.use_gpu() and gpu.distance.distance_supported('Hausdorff'):
        print(f"<{os.path.basename(__file__)}> Computing pairwise distances on GPU.")
        distances = gpu.distance.pairwise_distance_matrix_from_fiber_array(
            current_fiber_array, 0, distance_method='Hausdorff')
    elif gpu.config.cpu_fast_enabled() and gpu.distance.distance_supported('Hausdorff'):
        print(f"<{os.path.basename(__file__)}> Computing pairwise distances with the vectorized CPU backend.")
        distances = gpu.cpu_backend.pairwise_distance_matrix_from_fiber_array(
            current_fiber_array, 0, distance_method='Hausdorff')
    elif USE_PARALLEL:
        distances = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(similarity.fiber_distance)(
            current_fiber_array.get_fiber(lidx),
            current_fiber_array,
            0, 'Hausdorff')
            for lidx in fiber_indices)
        distances = numpy.array(distances)
    else:
        distances = \
            numpy.zeros(
            (current_fiber_array.number_of_fibers,
             current_fiber_array.number_of_fibers))
        for lidx in fiber_indices:
            distances[lidx, :] = \
                similarity.fiber_distance(
                    current_fiber_array.get_fiber(lidx),
                    current_fiber_array, 0)
   
    # gaussian smooth all fibers using local neighborhood.
    # Array-based reimplementation of the original Fiber-object loop:
    # same arithmetic order (scale each neighbor, match_order against the
    # evolving weighted accumulator, sequential accumulation and division).
    number_of_fibers = current_fiber_array.number_of_fibers
    fr = current_fiber_array.fiber_array_r
    fa = current_fiber_array.fiber_array_a
    fs = current_fiber_array.fiber_array_s
    out_r = numpy.zeros_like(fr)
    out_a = numpy.zeros_like(fa)
    out_s = numpy.zeros_like(fs)
    for fidx in fiber_indices:
        if (fidx % 100) == 0:
            print(fidx, '/', number_of_fibers)

        # find indices of all nearby fibers
        indices = numpy.nonzero(distances[fidx] < upper_thresh)[0]

        # local_fibers.append(curr_fibers[idx] * weight), then start the
        # accumulator with the first neighbor and add the rest in order
        w0 = numpy.exp(-distances[fidx][indices[0]] / sigmasq)
        j0 = indices[0]
        acc_r = fr[j0] * w0
        acc_a = fa[j0] * w0
        acc_s = fs[j0] * w0
        out_weights = w0
        for idx in indices[1:]:
            wj = numpy.exp(-distances[fidx][idx] / sigmasq)
            sr = fr[idx] * wj
            sa = fa[idx] * wj
            ss = fs[idx] * wj
            # Fiber.match_order(acc, scaled neighbor)
            corr = numpy.sum(acc_r * sr + acc_a * sa + acc_s * ss)
            corr2 = numpy.sum(acc_r * sr[::-1] + acc_a * sa[::-1] + acc_s * ss[::-1])
            if corr > corr2:
                acc_r = acc_r + sr
                acc_a = acc_a + sa
                acc_s = acc_s + ss
            else:
                acc_r = acc_r + sr[::-1]
                acc_a = acc_a + sa[::-1]
                acc_s = acc_s + ss[::-1]
            out_weights = out_weights + wj
        out_r[fidx] = acc_r / out_weights
        out_a[fidx] = acc_a / out_weights
        out_s[fidx] = acc_s / out_weights
        next_weights.append(out_weights)

    # set up array for output
    output_fiber_array = fibers.FiberArray()
    output_fiber_array.number_of_fibers = number_of_fibers
    output_fiber_array.points_per_fiber = points_per_fiber
    output_fiber_array.fiber_array_r = out_r
    output_fiber_array.fiber_array_a = out_a
    output_fiber_array.fiber_array_s = out_s
            
    # convert output to polydata
    outpd = output_fiber_array.convert_to_polydata()
    
    # color by the weights or "local density"
    # color output by the number of fibers that each output fiber corresponds to
    outcolors = vtk.vtkFloatArray()
    outcolors.SetName('KernelDensity')
    for weight in next_weights:
        outcolors.InsertNextTuple1(weight)
    #outpd.GetCellData().SetScalars(outcolors)
    outpd.GetCellData().AddArray(outcolors)
    outpd.GetCellData().SetActiveScalars('KernelDensity')

    return outpd, numpy.array(next_weights)
    
def anisotropic_smooth(inpd, fiber_distance_threshold, points_per_fiber=30, n_jobs=2, cluster_max = 10):
    """ Average nearby fibers.
    
    The pairwise fiber distance matrix is computed, then fibers
    are averaged with their neighbors until an edge (>max_fiber_distance) is encountered.

    """

    # polydata to array conversion, fixed-length fiber representation
    current_fiber_array = fibers.FiberArray()
    current_fiber_array.points_per_fiber = points_per_fiber
    current_fiber_array.convert_from_polydata(inpd)
    original_number_of_fibers = current_fiber_array.number_of_fibers
    
    # fiber list data structure initialization for easy fiber averaging
    curr_count = list()
    curr_fibers = list()
    curr_indices = list()
    for lidx in range(0, current_fiber_array.number_of_fibers):
        curr_fibers.append(current_fiber_array.get_fiber(lidx))
        curr_count.append(1)
        curr_indices.append(list([lidx]))
        
    converged = False
    iteration_count = 0
    
    while not converged:
        print(f"<{os.path.basename(__file__)}> ITERATION:", iteration_count, "SUM FIBER COUNTS:", numpy.sum(numpy.array(curr_count)))
        print(f"<{os.path.basename(__file__)}> number indices", len(curr_indices))
        
        # fiber data structures for output of this iteration
        next_fibers = list()
        next_count = list()
        next_indices = list()
        
        # information for this iteration
        done = numpy.zeros(current_fiber_array.number_of_fibers)
        fiber_indices = list(range(0, current_fiber_array.number_of_fibers))

        # if the maximum number of fibers have been combined, stop averaging this fiber
        done[numpy.nonzero(numpy.array(curr_count) >= cluster_max)] = 1
        
        # pairwise distance matrix
        if gpu.use_gpu() and gpu.distance.distance_supported('Hausdorff'):
            print(f"<{os.path.basename(__file__)}> Computing pairwise distances on GPU.")
            distances = gpu.distance.pairwise_distance_matrix_from_fiber_array(
                current_fiber_array, 0, distance_method='Hausdorff')
        elif gpu.config.cpu_fast_enabled() and gpu.distance.distance_supported('Hausdorff'):
            print(f"<{os.path.basename(__file__)}> Computing pairwise distances with the vectorized CPU backend.")
            distances = gpu.cpu_backend.pairwise_distance_matrix_from_fiber_array(
                current_fiber_array, 0, distance_method='Hausdorff')
        elif USE_PARALLEL:
            distances = Parallel(n_jobs=n_jobs, verbose=1)(
                delayed(similarity.fiber_distance)(
                current_fiber_array.get_fiber(lidx),
                current_fiber_array,
                0, 'Hausdorff')
                for lidx in fiber_indices)
            distances = numpy.array(distances)
        else:
            distances = \
                numpy.zeros(
                (current_fiber_array.number_of_fibers,
                 current_fiber_array.number_of_fibers))
            for lidx in fiber_indices:
                distances[lidx, :] = \
                    similarity.fiber_distance(
                        current_fiber_array.get_fiber(lidx),
                        current_fiber_array, 0, 'Hausdorff')

        # distances to self are not of interest
        for lidx in fiber_indices:
            distances[lidx,lidx] = numpy.inf
        
        # sort the pairwise distances. 
        distances_flat = distances.flatten()
        pair_order = numpy.argsort(distances_flat)

        print(f"<{os.path.basename(__file__)}> DISTANCE MIN:", distances_flat[pair_order[0]], \
            "DISTANCE COUNT:", distances.shape)

        # if the smallest distance is greater or equal to the
        # threshold, we have converged
        if distances_flat[pair_order[0]] >= fiber_distance_threshold:
            converged = True
            print(f"<{os.path.basename(__file__)}> CONVERGED")
            break
        else:
            print(f"<{os.path.basename(__file__)}> NOT CONVERGED")
            
        # loop variables
        idx = 0
        pair_idx = pair_order[idx]
        number_of_fibers = distances.shape[0]
        number_averages = 0
        
        # combine nearest neighbors unless done, until hit threshold
        while distances_flat[pair_idx] < fiber_distance_threshold:
            # find the fiber indices corresponding to this pairwise distance
            # use div and mod
            f_row = pair_idx / number_of_fibers
            f_col = pair_idx % number_of_fibers

            # check if this neighbor pair can be combined
            combine = (not done[f_row]) and (not done[f_col])
            if combine :
                done[f_row] += 1
                done[f_col] += 1
                # weighted average of the fibers (depending on how many each one represents)
                next_fibers.append(
                    (curr_fibers[f_row] * curr_count[f_row] + \
                     curr_fibers[f_col] *curr_count[f_col]) / \
                    (curr_count[f_row] + curr_count[f_col]))
                # this was the regular average
                #next_fibers.append((curr_fibers[f_row] + curr_fibers[f_col])/2)
                next_count.append(curr_count[f_row] + curr_count[f_col])
                number_averages += 1
                #next_indices.append(list([curr_indices[f_row], curr_indices[f_col]]))
                next_indices.append(list(curr_indices[f_row] + curr_indices[f_col]))
                
            # increment for the loop
            idx += 1
            pair_idx = pair_order[idx]

        # copy through any unvisited (already converged) fibers
        unvisited = numpy.nonzero(done==0)[0]
        for fidx in unvisited:
            next_fibers.append(curr_fibers[fidx])
            next_count.append(curr_count[fidx])
            next_indices.append(curr_indices[fidx])
            
        # set up for next iteration
        curr_fibers = next_fibers
        curr_count = next_count
        curr_indices = next_indices
        iteration_count += 1

        # set up array for next iteration distance computation
        current_fiber_array = fibers.FiberArray()    
        current_fiber_array.number_of_fibers = len(curr_fibers)
        current_fiber_array.points_per_fiber = points_per_fiber
        dims = [current_fiber_array.number_of_fibers, current_fiber_array.points_per_fiber]
        # fiber data
        current_fiber_array.fiber_array_r = numpy.zeros(dims)
        current_fiber_array.fiber_array_a = numpy.zeros(dims)
        current_fiber_array.fiber_array_s = numpy.zeros(dims)
        curr_fidx = 0
        for curr_fib in curr_fibers:
            current_fiber_array.fiber_array_r[curr_fidx] = curr_fib.r
            current_fiber_array.fiber_array_a[curr_fidx] = curr_fib.a
            current_fiber_array.fiber_array_s[curr_fidx] = curr_fib.s
            curr_fidx += 1

        print(f"<{os.path.basename(__file__)}> SUM FIBER COUNTS:", numpy.sum(numpy.array(curr_count)), "SUM DONE FIBERS:", numpy.sum(done))
        print(f"<{os.path.basename(__file__)}> MAX COUNT:" , numpy.max(numpy.array(curr_count)), "AVGS THIS ITER:", number_averages)

    # when converged, convert output to polydata
    outpd = current_fiber_array.convert_to_polydata()

    # color output by the number of fibers that each output fiber corresponds to
    outcolors = vtk.vtkFloatArray()
    outcolors.SetName('FiberTotal')
    for count in curr_count:
        outcolors.InsertNextTuple1(count)
    outpd.GetCellData().SetScalars(outcolors)

    # also color the input pd by output cluster number
    cluster_numbers = numpy.zeros(original_number_of_fibers)
    cluster_count = numpy.zeros(original_number_of_fibers)
    cluster_idx = 0
    for index_list in curr_indices:
        indices = numpy.array(index_list).astype(int)
        cluster_numbers[indices] = cluster_idx
        cluster_count[indices] = curr_count[cluster_idx]
        cluster_idx += 1
    outclusters =  vtk.vtkFloatArray()
    outclusters.SetName('ClusterNumber')
    for cluster in cluster_numbers:
        outclusters.InsertNextTuple1(cluster)
    inpd.GetCellData().AddArray(outclusters)
    inpd.GetCellData().SetActiveScalars('ClusterNumber')

    return outpd, numpy.array(curr_count), inpd, cluster_numbers, cluster_count
    

    
def laplacian_of_gaussian(inpd, fiber_distance_sigma = 25, points_per_fiber=30, n_jobs=2, upper_thresh=30):
    """ Filter nearby fibers, using LoG weights.
    
    The pairwise fiber distance matrix is computed, then fibers are
    averaged with their neighbors using LoG weighting.  This is
    essentially a fiber subtraction operation, giving vectors pointing
    from the center fiber under the kernel, to all nearby fibers. Thus
    the output of this operation is not a fiber, but we compute
    properties of the output that might be interesting and related to
    fibers. We summarize the result using the average vector at each
    fiber point (output is its magnitude, similar to edge
    strength). The covariance of the vectors is also
    investigated. This matrix would be spherical in an isotropic
    region such as a tract center (tube/line detector), or planar in a
    sheetlike tract (sheet detector).

    The equation is: (1-d^2/sigma^2) exp(-d^2/(2*sigma^2)), and
    weights are normalized in the neighborhood (weighted averaging).
    """

    sigmasq = fiber_distance_sigma * fiber_distance_sigma
    
    # polydata to array conversion, fixed-length fiber representation
    fiber_array = fibers.FiberArray()
    fiber_array.points_per_fiber = points_per_fiber
    fiber_array.convert_from_polydata(inpd)

    fiber_indices = list(range(0, fiber_array.number_of_fibers))

    # pairwise distance matrix
    if gpu.use_gpu() and gpu.distance.distance_supported('Hausdorff'):
        print(f"<{os.path.basename(__file__)}> Computing pairwise distances on GPU.")
        distances = gpu.distance.pairwise_distance_matrix_from_fiber_array(
            fiber_array, 0, distance_method='Hausdorff')
    elif gpu.config.cpu_fast_enabled() and gpu.distance.distance_supported('Hausdorff'):
        print(f"<{os.path.basename(__file__)}> Computing pairwise distances with the vectorized CPU backend.")
        distances = gpu.cpu_backend.pairwise_distance_matrix_from_fiber_array(
            fiber_array, 0, distance_method='Hausdorff')
    elif USE_PARALLEL:
        distances = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(similarity.fiber_distance)(
            fiber_array.get_fiber(lidx),
            fiber_array,
            0, 'Hausdorff')
            for lidx in fiber_indices)
        distances = numpy.array(distances)
    else:
        distances = \
            numpy.zeros(
            (fiber_array.number_of_fibers,
             fiber_array.number_of_fibers))
        for lidx in fiber_indices:
            distances[lidx, :] = \
                similarity.fiber_distance(
                    fiber_array.get_fiber(lidx),
                    fiber_array, 0)

    # Array-based reimplementation of the original Fiber-object loop.
    # match_order decisions are computed against the fixed current fiber
    # with the same combined-array summation structure; the per-neighbor
    # accumulation order is preserved.
    number_of_fibers = fiber_array.number_of_fibers
    fr = fiber_array.fiber_array_r
    fa = fiber_array.fiber_array_a
    fs = fiber_array.fiber_array_s

    filter_vectors_r = numpy.zeros((number_of_fibers, points_per_fiber))
    filter_vectors_a = numpy.zeros((number_of_fibers, points_per_fiber))
    filter_vectors_s = numpy.zeros((number_of_fibers, points_per_fiber))
    filter_vector_magnitudes = list()
    filter_confidences = list()

    # gaussian smooth all fibers using local neighborhood
    for fidx in fiber_indices:
        if (fidx % 100) == 0:
            print(fidx, '/', number_of_fibers)

        cur_r = fr[fidx]
        cur_a = fa[fidx]
        cur_s = fs[fidx]

        # find indices of all nearby fibers
        # this includes the center fiber under the kernel
        indices = numpy.nonzero(distances[fidx] < upper_thresh)[0]

        # compute filter kernel weights (dist*dist), as in the loop
        d = distances[fidx][indices]
        weights = numpy.exp(-(d * d) / sigmasq)

        # Fiber.match_order against the fixed current fiber, for all
        # neighbors at once; the summation structure matches the original
        nr = fr[indices]
        na = fa[indices]
        ns = fs[indices]
        corr = (cur_r * nr + cur_a * na + cur_s * ns).sum(1)
        corr2 = (cur_r * nr[:, ::-1] + cur_a * na[:, ::-1] + cur_s * ns[:, ::-1]).sum(1)
        flip = ~(corr > corr2)
        mr = numpy.where(flip[:, numpy.newaxis], nr[:, ::-1], nr)
        ma = numpy.where(flip[:, numpy.newaxis], na[:, ::-1], na)
        ms = numpy.where(flip[:, numpy.newaxis], ns[:, ::-1], ns)

        # sequential accumulation, matching the original += loop
        out_r = numpy.zeros(points_per_fiber)
        out_a = numpy.zeros(points_per_fiber)
        out_s = numpy.zeros(points_per_fiber)
        for j in range(indices.size):
            out_r += (cur_r - mr[j]) * weights[j]
            out_a += (cur_a - ma[j]) * weights[j]
            out_s += (cur_s - ms[j]) * weights[j]

        total_weights = weights.sum()
        out_r = out_r / total_weights
        out_a = out_a / total_weights
        out_s = out_s / total_weights

        filter_vectors_r[fidx] = out_r
        filter_vectors_a[fidx] = out_a
        filter_vectors_s[fidx] = out_s
        filter_confidences.append(total_weights)

        filter_vector_magnitudes.append(numpy.sqrt(\
                numpy.multiply(out_r, out_r) + \
                    numpy.multiply(out_a, out_a) + \
                    numpy.multiply(out_s, out_s)))


    # output a new pd!!!!
    # with fixed length fibers. and the new vector field.
    # output the vectors from the filtering
    outpd = fiber_array.convert_to_polydata()
    vectors = vtk.vtkFloatArray()
    vectors.SetName('FiberDifferenceVectors')
    vectors.SetNumberOfComponents(3)
    for fidx in fiber_indices:
        for idx in range(points_per_fiber):
            vectors.InsertNextTuple3(filter_vectors_r[fidx, idx],
                                     filter_vectors_a[fidx, idx],
                                     filter_vectors_s[fidx, idx])
    magnitudes = vtk.vtkFloatArray()
    magnitudes.SetName('FiberDifferenceMagnitudes')
    magnitudes.SetNumberOfComponents(1)
    for mag in filter_vector_magnitudes:
        for idx in range(points_per_fiber):
            magnitudes.InsertNextTuple1(mag[idx])
    confidences = vtk.vtkFloatArray()
    confidences.SetName('FiberDifferenceConfidences')
    confidences.SetNumberOfComponents(1)
    for mag in filter_confidences:
        for idx in range(points_per_fiber):
            confidences.InsertNextTuple1(mag)
         
    outpd.GetPointData().AddArray(vectors)
    outpd.GetPointData().SetActiveVectors('FiberDifferenceVectors')

    outpd.GetPointData().AddArray(confidences)
    outpd.GetPointData().SetActiveScalars('FiberDifferenceConfidences')

    outpd.GetPointData().AddArray(magnitudes)
    outpd.GetPointData().SetActiveScalars('FiberDifferenceMagnitudes')

    # color by the weights or "local density"
    # color output by the number of fibers that each output fiber corresponds to
    #outcolors = vtk.vtkFloatArray()
    #outcolors.SetName('KernelDensity')
    #for weight in next_weights:
    #    outcolors.InsertNextTuple1(weight)
    #inpd.GetCellData().AddArray(outcolors)
    #inpd.GetCellData().SetActiveScalars('KernelDensity')
    #outcolors = vtk.vtkFloatArray()
    #outcolors.SetName('EdgeMagnitude')
    #for magnitude in filter_vector_magnitudes:
    #    outcolors.InsertNextTuple1(magnitude)
    #inpd.GetCellData().AddArray(outcolors)
    #inpd.GetCellData().SetActiveScalars('EdgeMagnitude')

    return outpd, numpy.array(filter_vector_magnitudes)

def pd_to_array(inpd, dims=225):
    count_vol = numpy.ndarray([dims,dims,dims])
    ptids = vtk.vtkIdList()
    points = inpd.GetPoints()
    data_vol = []    
    # check for cell data
    cell_data = inpd.GetCellData().GetScalars()
    if cell_data:
        data_vol = numpy.ndarray([dims,dims,dims])
    # loop over lines
    inpd.GetLines().InitTraversal()
    print(f"<{os.path.basename(__file__)}> Input number of points: ",\
        points.GetNumberOfPoints(),\
        "lines:", inpd.GetNumberOfLines()) 
    # loop over all lines
    for lidx in range(0, inpd.GetNumberOfLines()):
        # progress
        #if verbose:
        #    if lidx % 1 == 0:
        #        print "<filter.py> Line:", lidx, "/", inpd.GetNumberOfLines()
        inpd.GetLines().GetNextCell(ptids)
        num_points = ptids.GetNumberOfIds()
        for pidx in range(0, num_points):
            point = points.GetPoint(ptids.GetId(pidx))
            # center so that 0,0,0 moves to 100,100,100
            point = numpy.round(numpy.array(point) + 110)         
            count_vol[point[0], point[1], point[2]] += 1
            if cell_data:
                data_vol[point[0], point[1], point[2]] += cell_data.GetTuple(lidx)[0]
    return count_vol, data_vol
    
def array_to_vtk(inarray, name='from_numpy'):
    vol = vtk.vtkImageData()
    dims = inarray.shape
    vol.SetDimensions(dims[0], dims[1], dims[2])
    vol.SetOrigin(0,0,0)
    #vol.SetSpacing(gridSpacing,gridSpacing,gridSpacing)
    sc = vtk.vtkShortArray()
    sc.SetNumberOfValues(dims[0] * dims[1] * dims[2])
    sc.SetNumberOfComponents(1)
    sc.SetName(name)
    for ii,tmp in enumerate(inarray.flatten()):
        sc.SetValue(ii,round((numpy.abs(tmp))*100))
    vol.GetPointData().SetScalars(sc)
    return vol
    

def measure_line_lengths(inpd):
    ptids = vtk.vtkIdList()
    points = inpd.GetPoints()
    output_lengths = numpy.zeros(inpd.GetNumberOfLines())
    # loop over lines
    inpd.GetLines().InitTraversal()
    print(f"<{os.path.basename(__file__)}> Input number of points: ",\
        points.GetNumberOfPoints(),\
        "lines:", inpd.GetNumberOfLines()) 
    # loop over all lines
    for lidx in range(0, inpd.GetNumberOfLines()):
        # progress
        #if verbose:
        #    if lidx % 1 == 0:
        #        print "<filter.py> Line:", lidx, "/", inpd.GetNumberOfLines()
        inpd.GetLines().GetNextCell(ptids)
        output_lengths[lidx] = ptids.GetNumberOfIds()
    return(output_lengths)
