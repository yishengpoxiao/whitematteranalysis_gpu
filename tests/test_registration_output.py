"""Registration output reuses the full subject and preserves VTK attributes."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import vtk
import whitematteranalysis as wma
from test_output_pipeline import fixture


class RegistrationOutputTests(unittest.TestCase):
    def test_save_does_not_reread_or_modify_input(self):
        subject = fixture()
        register = wma.congeal_to_atlas.SubjectToAtlasRegistration()
        register.set_subject(subject, 'subject')
        register.input_polydata_filename = '/unused/original.vtk'
        register.transform.Translate(3., -4., 5.)
        register.transform.RotateY(12.)
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / 'original.vtk'
            wma.io.write_polydata(subject, str(original))
            original_bytes = original.read_bytes()
            transformer = vtk.vtkTransformPolyDataFilter()
            transformer.SetInputData(subject)
            transformer.SetTransform(register.transform)
            transformer.Update()
            expected = Path(directory) / 'expected.vtk'
            wma.io.write_polydata(transformer.GetOutput(), str(expected))
            with patch.object(wma.io, 'read_polydata', side_effect=AssertionError('Unexpected disk read')):
                register.save_transformed_polydata_to_disk(directory)
            self.assertEqual(expected.read_bytes(), (Path(directory) / 'subject_reg.vtk').read_bytes())
            wma.io.write_polydata(subject, str(original))
            self.assertEqual(original.read_bytes(), original_bytes)


if __name__ == '__main__':
    unittest.main()
