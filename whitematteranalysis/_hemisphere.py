"""Shared hemisphere-label logic for standalone and fused pipeline stages."""
import warnings

import numpy
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

import whitematteranalysis as wma


def load_location_policy(filename):
    """Load the assessment table and apply the historical CPC overrides."""
    policy = numpy.atleast_1d(numpy.loadtxt(
        filename,
        dtype={'names': ('Cluster Index', 'Location Label'), 'formats': ('S17', 'S2')},
        delimiter="\t", skiprows=1))
    for index in (145, 159, 557, 677, 770):
        if index < len(policy):
            policy[index][1] = b'h'
    return policy


def read_mask_location_from_vtk(polydata):
    labels = numpy.zeros(polydata.GetNumberOfLines())
    point_data = polydata.GetPointData()
    for index in range(point_data.GetNumberOfArrays()):
        array = point_data.GetArray(index)
        if array.GetName() not in ('HemisphereLocataion', 'HemisphereLocation'):
            continue
        if array.GetName() == 'HemisphereLocataion':
            warnings.warn(wma.utils.hemisphere_loc_name_typo_warn_msg,
                          PendingDeprecationWarning)
        geometry = wma.filter._lines_offsets_connectivity(polydata)
        if geometry is not None:
            offsets, connectivity, _ = geometry
            nonempty = numpy.diff(offsets) > 0
            values = vtk_to_numpy(array).reshape(-1, array.GetNumberOfComponents())[:, 0]
            labels[nonempty] = values[connectivity[offsets[1:][nonempty] - 1]]
        else:
            polydata.GetLines().InitTraversal()
            for line_index in range(polydata.GetNumberOfLines()):
                point_ids = vtk.vtkIdList()
                polydata.GetLines().GetNextCell(point_ids)
                for point_index in range(point_ids.GetNumberOfIds()):
                    labels[line_index] = array.GetTuple(point_ids.GetId(point_index))[0]
        return True, labels
    return False, labels


def write_mask_location_to_vtk(polydata, labels):
    point_data = polydata.GetPointData()
    for index in range(point_data.GetNumberOfArrays()):
        array = point_data.GetArray(index)
        if array.GetName() == 'HemisphereLocataion':
            warnings.warn(wma.utils.hemisphere_loc_name_typo_warn_msg,
                          PendingDeprecationWarning)
            print('  -- HemisphereLocataion is in the input data: skip updating the vtk file.')
            return polydata
        if array.GetName() == 'HemisphereLocation':
            print('  -- HemisphereLocation is in the input data: skip updating the vtk file.')
            return polydata

    geometry = wma.filter._lines_offsets_connectivity(polydata)
    if geometry is not None:
        values = numpy.repeat(numpy.asarray(labels, dtype=numpy.float64),
                              numpy.diff(geometry[0]))
        array = numpy_to_vtk(values, deep=True)
        array.SetName('HemisphereLocation')
    else:
        array = vtk.vtkDoubleArray()
        array.SetName('HemisphereLocation')
        polydata.GetLines().InitTraversal()
        for line_index in range(polydata.GetNumberOfLines()):
            point_ids = vtk.vtkIdList()
            polydata.GetLines().GetNextCell(point_ids)
            for _ in range(point_ids.GetNumberOfIds()):
                array.InsertNextTuple1(labels[line_index])
    point_data.AddArray(array)
    return polydata


def assess_polydata(polydata, cluster_index, location_data=None,
                    hemisphere_percent_threshold=0.6):
    """Ensure labels exist, returning (polydata, labels, whether it changed)."""
    has_labels, labels = read_mask_location_from_vtk(polydata)
    if has_labels:
        return polydata, labels, False

    if location_data is None:
        label_group = None
    else:
        label_group = location_data[cluster_index][1]

    if len(labels) == 0:
        pass
    elif label_group in (b'c', b'ng'):
        labels[:] = 3
    elif label_group == b'h':
        fibers = wma.fibers.FiberArray()
        fibers.points_per_fiber = 40
        fibers.hemisphere_percent_threshold = 0.5001
        fibers.hemispheres = True
        fibers.convert_from_polydata(polydata)
        labels[fibers.index_left_hem] = 1
        labels[fibers.index_right_hem] = 2
        if len(fibers.index_commissure) > 0:
            side = 1 if len(fibers.index_left_hem) <= len(fibers.index_right_hem) else 2
            labels[fibers.index_commissure] = side
    elif label_group is None:
        fibers = wma.fibers.FiberArray()
        fibers.points_per_fiber = 40
        fibers.hemisphere_percent_threshold = hemisphere_percent_threshold
        fibers.hemispheres = True
        fibers.convert_from_polydata(polydata)
        labels[fibers.index_left_hem] = 1
        labels[fibers.index_right_hem] = 2
        labels[fibers.index_commissure] = 3

    write_mask_location_to_vtk(polydata, labels)
    return polydata, labels, True
