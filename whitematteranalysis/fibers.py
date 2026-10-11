# -*- coding: utf-8 -*-

""" fibers.py

This module contains code for representation of tractography using a
fixed-length parameterization.

class FiberArray

"""

import os

import numpy
import vtk
import time


class Fiber:
    """A class for fiber tractography data, represented with a fixed length"""

    def __init__(self):
        self.r = None
        self.a = None
        self.s = None
        self.points_per_fiber = None
        self.hemisphere_percent_threshold = 0.95
        
    def get_equivalent_fiber(self):
        """ Get the reverse order of current line (trajectory), as the
        fiber can be equivalently represented in either order."""
        
        fiber = Fiber()

        fiber.r = self.r[::-1]
        fiber.a = self.a[::-1]
        fiber.s = self.s[::-1]

        fiber.points_per_fiber = self.points_per_fiber

        return fiber

    def get_reflected_fiber(self):
        """ Returns reflected version of current fiber by reflecting
        fiber across midsagittal plane. Just sets output R coordinate to -R."""
 
        fiber = Fiber()

        fiber.r = - self.r
        fiber.a = self.a
        fiber.s = self.s

        fiber.points_per_fiber = self.points_per_fiber

        return fiber

    def match_order(self, other):
        """ Reverse order of fiber to match this one if needed """
        # compute correlation
        corr = numpy.multiply(self.r, other.r) + \
            numpy.multiply(self.a, other.a) + \
            numpy.multiply(self.s, other.s)

        other2 = other.get_equivalent_fiber()
        corr2 = numpy.multiply(self.r, other2.r) + \
            numpy.multiply(self.a, other2.a) + \
            numpy.multiply(self.s, other2.s)
        
        if numpy.sum(corr) > numpy.sum(corr2):
            return other
        else:
            return other2
        
    def __add__(self, other):
        """This is the + operator for fibers"""
        other_matched = self.match_order(other)
        fiber = Fiber()
        fiber.r = self.r + other_matched.r
        fiber.a = self.a + other_matched.a
        fiber.s = self.s + other_matched.s
        return fiber
    
    def __div__(self, other):
        """ This is to divide a fiber by a number"""
        fiber = Fiber()
        fiber.r = numpy.divide(self.r, other)
        fiber.a = numpy.divide(self.a, other)
        fiber.s = numpy.divide(self.s, other)
        return fiber

    def __mul__(self, other):
        """ This is to multiply a fiber by a number"""
        fiber = Fiber()
        fiber.r = numpy.multiply(self.r, other)
        fiber.a = numpy.multiply(self.a, other)
        fiber.s = numpy.multiply(self.s, other)
        return fiber
    
    def __subtract__(self, other):
        """This is the - operator for fibers"""
        other_matched = self.match_order(other)
        fiber = Fiber()
        fiber.r = self.r - other_matched.r
        fiber.a = self.a - other_matched.a
        fiber.s = self.s - other_matched.s
        #fiber.r = self.r + other_matched.r
        #fiber.a = self.a + other_matched.a
        #fiber.s = self.s + other_matched.s
        return fiber
    
class FiberArray:

    """A class for arrays of fiber tractography data, represented with
    a fixed length"""

    def __init__(self):
        # parameters
        self.points_per_fiber = 10
        self.verbose = 0

        # fiber data
        self.fiber_array_r = None
        self.fiber_array_a = None
        self.fiber_array_s = None

        # output arrays indicating hemisphere/callosal (L,C,R= -1, 0, 1)
        self.fiber_hemisphere = None
        self.hemispheres = False
        
        # output boolean arrays for each hemisphere and callosal fibers
        self.is_left_hem = None
        self.is_right_hem = None
        self.is_commissure = None

        # output indices of each type above
        self.index_left_hem = None
        self.index_right_hem = None
        self.index_commissure = None
        self.index_hem = None

        # output totals of each type also
        self.number_of_fibers = 0
        self.number_left_hem = None
        self.number_right_hem = None
        self.number_commissure = None

    def __str__(self):
        output = "\n points_per_fiber\t" + str(self.points_per_fiber) \
            + "\n number_of_fibers\t\t" + str(self.number_of_fibers) \
            + "\n fiber_hemisphere\t\t" + str(self.fiber_hemisphere) \
            + "\n verbose\t" + str(self.verbose)

        return output

    def _calculate_line_indices(self, input_line_length, output_line_length):
        """ Figure out indices for downsampling of polyline data.

        The indices include the first and last points on the line,
        plus evenly spaced points along the line.  This code figures
        out which indices we actually want from a line based on its
        length (in number of points) and the desired length.

        """

        # this is the increment between output points
        step = (input_line_length - 1.0) / (output_line_length - 1.0)

        # these are the output point indices (0-based)
        ptlist = []
        for ptidx in range(0, output_line_length):
        #print(ptidx*step)
            ptlist.append(ptidx * step)

        # test
        if __debug__:
            # this tests we output the last point on the line
            #test = ((output_line_length - 1) * step == input_line_length - 1)
            test = (round(ptidx*step) == input_line_length-1)
            if not test:
                print(f"<{os.path.basename(__file__)}> ERROR: fiber numbers don't add up.")
                print(step)
                print(input_line_length)
                print(output_line_length)
                print(test)
                raise AssertionError

        return ptlist

    def get_fiber(self, fiber_index):
        """ Return fiber number fiber_index. Return value is class
        Fiber."""

        fiber = Fiber()
        fiber.r = self.fiber_array_r[fiber_index, :]
        fiber.a = self.fiber_array_a[fiber_index, :]
        fiber.s = self.fiber_array_s[fiber_index, :]

        fiber.points_per_fiber = self.points_per_fiber

        return fiber

    def get_equivalent_fiber(self, fiber_index):
        """ Return equivalent version of fiber number
        fiber_index. Return value is class Fiber. Gets the reverse
        order of line (trajectory), as the fiber can be equivalently
        represented in either order."""
  
        fiber = self.get_fiber(fiber_index)

        return fiber.get_equivalent_fiber()

    def get_fibers(self, fiber_indices):
        """ Return FiberArray containing subset of data corresponding
        to fiber_indices"""
        
        fibers = FiberArray()

        fibers.number_of_fibers = len(fiber_indices)

        # parameters
        fibers.points_per_fiber = self.points_per_fiber
        fibers.verbose = self.verbose

        # fiber data
        fibers.fiber_array_r = self.fiber_array_r[fiber_indices]
        fibers.fiber_array_a = self.fiber_array_a[fiber_indices]
        fibers.fiber_array_s = self.fiber_array_s[fiber_indices]

        if self.fiber_hemisphere is not None:
            # Output arrays indicating hemisphere/callosal (L,C,R= -1, 0, 1)
            fibers.fiber_hemisphere = self.fiber_hemisphere[fiber_indices]

            # output boolean arrays for each hemisphere and callosal fibers
            fibers.is_left_hem = self.is_left_hem[fiber_indices]
            fibers.is_right_hem = self.is_right_hem[fiber_indices]
            fibers.is_commissure = self.is_commissure[fiber_indices]

            # calculate indices of each type above
            fibers.index_left_hem = numpy.nonzero(fibers.is_left_hem)[0]
            fibers.index_right_hem = numpy.nonzero(fibers.is_right_hem)[0]
            fibers.index_commissure = numpy.nonzero(fibers.is_commissure)[0]
            fibers.index_hem = \
                numpy.nonzero(fibers.is_left_hem | fibers.is_right_hem)[0]

            # output totals of each type also
            fibers.number_left_hem = len(fibers.index_left_hem)
            fibers.number_right_hem = len(fibers.index_right_hem)
            fibers.number_commissure = len(fibers.index_commissure)

            # test
            if __debug__:
                test = fibers.number_of_fibers == \
                    (fibers.number_left_hem + fibers.number_right_hem \
                         + fibers.number_commissure)
                if not test:
                    print(f"<{os.path.basename(__file__)}> ERROR: fiber numbers don't add up.")
                    raise AssertionError

        return fibers

    def get_oriented_fibers(self, fiber_indices, order):
        """Return FiberArray containing subset of data corresponding to
        fiber_indices. Order fibers according to the array (where 0 is no

        change, and 1 means to reverse the order and return the
        equivalent fiber)
        """

        fibers = FiberArray()

        fibers.number_of_fibers = len(fiber_indices)

        # parameters
        fibers.points_per_fiber = self.points_per_fiber
        fibers.verbose = self.verbose

        # fiber data
        fibers.fiber_array_r = self.fiber_array_r[fiber_indices]
        fibers.fiber_array_a = self.fiber_array_a[fiber_indices]
        fibers.fiber_array_s = self.fiber_array_s[fiber_indices]

        # swap orientation as requested
        for (ord, fidx) in zip(order, range(fibers.number_of_fibers)):
            if ord == 1:
                f2 = fibers.get_equivalent_fiber(fidx)
                # replace it in the array
                fibers.fiber_array_r[fidx,:] = f2.r
                fibers.fiber_array_a[fidx,:] = f2.a
                fibers.fiber_array_s[fidx,:] = f2.s

        if self.fiber_hemisphere is not None:
            # Output arrays indicating hemisphere/callosal (L,C,R= -1, 0, 1)
            fibers.fiber_hemisphere = self.fiber_hemisphere[fiber_indices]

            # output boolean arrays for each hemisphere and callosal fibers
            fibers.is_left_hem = self.is_left_hem[fiber_indices]
            fibers.is_right_hem = self.is_right_hem[fiber_indices]
            fibers.is_commissure = self.is_commissure[fiber_indices]

            # calculate indices of each type above
            fibers.index_left_hem = numpy.nonzero(fibers.is_left_hem)[0]
            fibers.index_right_hem = numpy.nonzero(fibers.is_right_hem)[0]
            fibers.index_commissure = numpy.nonzero(fibers.is_commissure)[0]
            fibers.index_hem = \
                numpy.nonzero(fibers.is_left_hem | fibers.is_right_hem)[0]

            # output totals of each type also
            fibers.number_left_hem = len(fibers.index_left_hem)
            fibers.number_right_hem = len(fibers.index_right_hem)
            fibers.number_commissure = len(fibers.index_commissure)

            # test
            if __debug__:
                test = fibers.number_of_fibers == \
                    (fibers.number_left_hem + fibers.number_right_hem \
                         + fibers.number_commissure)
                if not test:
                    print(f"<{os.path.basename(__file__)}> ERROR: fiber numbers don't add up.")
                    raise AssertionError

        return fibers

    def convert_from_polydata(self, input_vtk_polydata, points_per_fiber=None):

        """Convert input vtkPolyData to the fixed length fiber
        representation of this class.

        The polydata should contain the output of tractography.

        The output is downsampled fibers in array format and
        hemisphere info is also calculated.

        """

        # points used in discretization of each trajectory
        if points_per_fiber is not None:
            self.points_per_fiber = points_per_fiber

        # line count. Assume all input lines are from tractography.
        self.number_of_fibers = input_vtk_polydata.GetNumberOfLines()

        if self.verbose:
            print(f"<{os.path.basename(__file__)}> Converting polydata to array representation. Lines:", self.number_of_fibers)

        # allocate array number of lines by line length
        self.fiber_array_r = numpy.zeros((self.number_of_fibers,
                                          self.points_per_fiber))
        self.fiber_array_a = numpy.zeros((self.number_of_fibers,
                                          self.points_per_fiber))
        self.fiber_array_s = numpy.zeros((self.number_of_fibers,
                                          self.points_per_fiber))

        # loop over lines
        input_vtk_polydata.GetLines().InitTraversal()
        line_ptids = vtk.vtkIdList()
        inpoints = input_vtk_polydata.GetPoints()

        # fast path: vectorized gather (bit-identical to the loop below)
        if self._convert_from_polydata_vectorized(input_vtk_polydata):
            if self.hemispheres:
                self.calculate_hemispheres()
            return

        for lidx in range(0, self.number_of_fibers):

            input_vtk_polydata.GetLines().GetNextCell(line_ptids)
            line_length = line_ptids.GetNumberOfIds()

            if self.verbose:
                if lidx % 100 == 0:
                    print(f"<{os.path.basename(__file__)}> Line:", lidx, "/", self.number_of_fibers)
                    print(f"<{os.path.basename(__file__)}> number of points:", line_length)

            # loop over the indices that we want and get those points
            pidx = 0
            for line_index in self._calculate_line_indices(line_length,
                                                 self.points_per_fiber):

                # do nearest neighbor interpolation: round index
                ptidx = line_ptids.GetId(int(round(line_index)))

                point = inpoints.GetPoint(ptidx)

                self.fiber_array_r[lidx, pidx] = point[0]
                self.fiber_array_a[lidx, pidx] = point[1]
                self.fiber_array_s[lidx, pidx] = point[2]
                pidx = pidx + 1

        # initialize hemisphere info
        if self.hemispheres:
            self.calculate_hemispheres()

    def _convert_from_polydata_vectorized(self, input_vtk_polydata):
        """Vectorized replacement for the per-line resampling loop.

        Returns False (and does nothing) if the VTK 9 connectivity API
        needed for it is unavailable, in which case the caller falls
        back to the original loop. The computation reproduces the loop
        exactly: for each line, step = (L-1.0)/(P-1.0), output point
        indices int(round(ptidx * step)) for ptidx in range(P) (Python
        round and numpy.rint share round-half-to-even), then nearest
        neighbor point lookup. Points are stored float32 by VTK and
        upcast to float64, as GetPoint() does.

        """

        try:
            from vtk.util import numpy_support
            offsets_array = input_vtk_polydata.GetLines().GetOffsetsArray()
            conn_array = input_vtk_polydata.GetLines().GetConnectivityArray()
        except AttributeError:
            return False

        offsets = numpy_support.vtk_to_numpy(offsets_array).astype(numpy.int64)
        conn = numpy_support.vtk_to_numpy(conn_array).astype(numpy.int64)
        points = numpy_support.vtk_to_numpy(input_vtk_polydata.GetPoints().GetData())

        number_of_lines = len(offsets) - 1
        if number_of_lines != self.number_of_fibers:
            return False

        P = self.points_per_fiber
        line_lengths = offsets[1:] - offsets[:-1]

        # step = (input_line_length - 1.0) / (output_line_length - 1.0)
        steps = (line_lengths.astype(numpy.float64) - 1.0) / (P - 1.0)
        # ptlist.append(ptidx * step) for ptidx in range(P)
        positions = numpy.arange(P, dtype=numpy.float64) * steps[:, numpy.newaxis]
        # int(round(...)): round-half-to-even in both Python and numpy
        local_ids = numpy.rint(positions).astype(numpy.int64)

        # same sanity check as _calculate_line_indices: last point is included
        if __debug__ and self.number_of_fibers > 0:
            test = bool(numpy.all(local_ids[:, -1] == (line_lengths - 1)))
            if not test:
                print(f"<{os.path.basename(__file__)}> ERROR: fiber numbers don't add up.")
                raise AssertionError

        # nearest neighbor lookup of the original point ids
        global_ids = conn[offsets[:-1][:, numpy.newaxis] + local_ids]

        pts = points[global_ids]  # (n, P, 3) float32, exact upcast below
        coords = numpy.zeros((self.number_of_fibers, P, 3), dtype=numpy.float64)
        coords[:, :, 0] = pts[:, :, 0]
        coords[:, :, 1] = pts[:, :, 1]
        coords[:, :, 2] = pts[:, :, 2]

        self.fiber_array_r = numpy.ascontiguousarray(coords[:, :, 0])
        self.fiber_array_a = numpy.ascontiguousarray(coords[:, :, 1])
        self.fiber_array_s = numpy.ascontiguousarray(coords[:, :, 2])
        return True
            
    def calculate_hemispheres(self):

        """ For each fiber assign a hemisphere using the first (R)
        coordinates.

        This part assumes we are in RAS so the first coordinate is
        positive to the RIGHT and negative to the LEFT.  The fiber
        must be more than 95% within 1 hemisphere.  This excludes
        corpus but can retain errant cingulum. We also want to
        identify likely commissural fibers.

        """

        # Figure out hemisphere of each line
        self.fiber_hemisphere = numpy.zeros(self.number_of_fibers)
        # percentage in left hemisphere
        test = sum(self.fiber_array_r.T < 0) / float(self.points_per_fiber)
        thresh = self.hemisphere_percent_threshold
        self.fiber_hemisphere[numpy.nonzero(test > thresh)] = -1
        self.fiber_hemisphere[numpy.nonzero(test < 1 - thresh)] = 1
        # previous code left for clarity below, concrete example of threshold:
        #self.fiber_hemisphere[numpy.nonzero(test > 0.95)] = -1
        #self.fiber_hemisphere[numpy.nonzero(test < 0.05)] = 1
        # otherwise hem stays 0 for commissural

        # output boolean arrays for each hemisphere and callosal fibers
        self.is_left_hem = (self.fiber_hemisphere == -1)
        self.is_right_hem = (self.fiber_hemisphere == 1)
        self.is_commissure = (self.fiber_hemisphere == 0)

        # output indices of each type above
        self.index_left_hem = numpy.nonzero(self.is_left_hem)[0]
        self.index_right_hem = numpy.nonzero(self.is_right_hem)[0]
        self.index_commissure = numpy.nonzero(self.is_commissure)[0]
        self.index_hem = \
            numpy.nonzero(self.is_left_hem | self.is_right_hem)[0]

        # output totals of each type also
        self.number_left_hem = len(self.index_left_hem)
        self.number_right_hem = len(self.index_right_hem)
        self.number_commissure = len(self.index_commissure)

        # test
        if __debug__:
            test = self.number_of_fibers == \
                (self.number_left_hem + self.number_right_hem \
                     + self.number_commissure)
            if not test:
                print(f"<{os.path.basename(__file__)}> ERROR: fiber numbers don't add up.")
                raise AssertionError

    def convert_to_polydata(self):
        """Convert fiber array to vtkPolyData object."""

        outpd = vtk.vtkPolyData()
        outpoints = vtk.vtkPoints()
        outlines = vtk.vtkCellArray()
        
        outlines.InitTraversal()

        for lidx in range(0, self.number_of_fibers):
            cellptids = vtk.vtkIdList()
            
            for pidx in range(0, self.points_per_fiber):

                idx = outpoints.InsertNextPoint(self.fiber_array_r[lidx, pidx],
                                                self.fiber_array_a[lidx, pidx],
                                                self.fiber_array_s[lidx, pidx])

                cellptids.InsertNextId(idx)
            
            outlines.InsertNextCell(cellptids)
            
        # put data into output polydata
        outpd.SetLines(outlines)
        outpd.SetPoints(outpoints)

        return outpd
