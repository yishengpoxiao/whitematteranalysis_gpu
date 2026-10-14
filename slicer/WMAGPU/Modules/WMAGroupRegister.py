# -*- coding: utf-8 -*-

"""WMA Groupwise Registration

Unbiased groupwise registration of a directory of subject
tractographies (wm_register_multisubject_faster.py presets,
GPU/CPU accelerated objective). Outputs: registered subjects,
transforms and registration_atlas.vtk in the chosen output directory.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(HERE, '_vendor')
if VENDOR not in sys.path:
    sys.path.insert(0, VENDOR)

import qt

import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule, ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic, ScriptedLoadableModuleTest)

from WMARegistrationUI import RegistrationOptions
import WMAGuiCommon as common
import WMAtlasLabelGPUCore as _CORE


class WMAGroupRegister(ScriptedLoadableModule):

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "WMA Groupwise Registration"
        self.parent.categories = [common.CATEGORY]
        self.parent.dependencies = []
        self.parent.contributors = "WhiteMatterAnalysis GPU acceleration project"
        self.parent.helpText = """
Unbiased groupwise (population) rigid, affine, or nonrigid registration of all whole-brain
tractography files in a directory (GPU/CPU accelerated objective).
Produces registered subjects, transforms and registration_atlas.vtk.
"""


class WMAGroupRegisterLogic(ScriptedLoadableModuleLogic):

    def run(self, subject_directory, output_directory, fibers_per_subject,
            fiber_length_mm, backend, parallel_jobs, progress_callback=None, **options):
        return _CORE.register_groupwise(
            subject_directory, output_directory,
            fibers_per_subject=fibers_per_subject,
            fiber_length_mm=fiber_length_mm,
            backend=backend, parallel_jobs=parallel_jobs,
            progress_callback=progress_callback, save_transforms=False, **options)


class WMAGroupRegisterWidget(ScriptedLoadableModuleWidget):

    def setup(self):
        try:
            self._setup_ui()
        except Exception:
            import traceback
            info = qt.QPlainTextEdit()
            info.readOnly = True
            info.setPlainText("Failed to build the interface:\n\n"
                              + traceback.format_exc())
            self.layout.addWidget(info)

    def _setup_ui(self):
        ScriptedLoadableModuleWidget.setup(self)
        self._CORE, wma_ok, wma_detail = common.ensure_environment()

        formBox = qt.QGroupBox("Groupwise registration of subject tractographies")
        form = qt.QFormLayout(formBox)

        self.subjectsDirectoryButton = qt.QPushButton("Select subject directory...")
        self.subjectsDirectory = ""
        self.subjectsDirectoryButton.connect('clicked(bool)', self.onPickSubjects)
        form.addRow("Subject directory:", self.subjectsDirectoryButton)

        self.outputDirectoryButton = qt.QPushButton("Select output directory...")
        self.outputDirectory = ""
        self.outputDirectoryButton.connect('clicked(bool)', self.onPickOutput)
        form.addRow("Output directory:", self.outputDirectoryButton)

        self.fibersPerSubjectSpin = qt.QSpinBox()
        self.fibersPerSubjectSpin.value = 10000
        self.fibersPerSubjectSpin.minimum = 100
        self.fibersPerSubjectSpin.maximum = 50000
        self.fibersPerSubjectSpin.singleStep = 100
        form.addRow("Fibers per subject:", self.fibersPerSubjectSpin)

        self.fiberLengthSpinBox = qt.QDoubleSpinBox()
        self.fiberLengthSpinBox.value = 20.0
        self.fiberLengthSpinBox.minimum = 0.0
        self.fiberLengthSpinBox.maximum = 200.0
        self.fiberLengthSpinBox.suffix = " mm"
        form.addRow("Minimum fiber length:", self.fiberLengthSpinBox)

        self.registrationOptions = RegistrationOptions(form, groupwise=True)
        common.add_backend_row(form, self)

        self.jobsSpin = qt.QSpinBox()
        self.jobsSpin.value = 1
        self.jobsSpin.minimum = 1
        self.jobsSpin.maximum = 64
        form.addRow("Parallel jobs:", self.jobsSpin)

        self.applyButton = qt.QPushButton("Run groupwise registration")
        self.applyButton.connect('clicked(bool)', self.onApply)
        form.addRow("", self.applyButton)

        self.layout.addWidget(formBox)
        common.add_status_area(self, self.layout)
        self.statusLabel.setPlainText(
            "whitematteranalysis: OK" if wma_ok else wma_detail)

    def onPickSubjects(self):
        path = common.pick_directory(slicer.util.mainWindow(),
                                     "Select subject directory", self.subjectsDirectory)
        if path:
            self.subjectsDirectory = path
            self.subjectsDirectoryButton.text = path

    def onPickOutput(self):
        path = common.pick_directory(slicer.util.mainWindow(),
                                     "Select output directory", self.outputDirectory)
        if path:
            self.outputDirectory = path
            self.outputDirectoryButton.text = path

    def cleanup(self):
        common.cleanup_controls(self)

    def onApply(self):
        if common.workflow_running():
            common.status(self, 'Another WMA workflow is running. Wait for it to finish.')
            return
        if not self.subjectsDirectory or not self.outputDirectory:
            slicer.util.errorDisplay("Select the subject and output directories first.")
            return
        backend = common.backend_key(self.backendComboBox.currentIndex)
        logic = WMAGroupRegisterLogic()
        self.applyButton.enabled = False
        try:
            common.start_computation(self)
            options = self.registrationOptions.arguments(self.fiberLengthSpinBox.value)
            result = logic.run(self.subjectsDirectory, self.outputDirectory,
                               self.fibersPerSubjectSpin.value,
                               self.fiberLengthSpinBox.value, backend,
                               self.jobsSpin.value,
                               progress_callback=lambda t, p: common.status(self, t, p), **options)
            self.transformNodes = [common.add_registration_transform(
                sid + '_to_group', transform, self.outputDirectory)
                for sid, transform in zip(result['subject_ids'], result['transforms'])]
            common.status(self,
                          "Done: %d subjects registered. Outputs in %s"
                          % (result['n_subjects'], result['output_directory']), 100)
        except Exception:
            import traceback
            common.status(self, "ERROR:\n" + traceback.format_exc())
        finally:
            common.finish_computation(self)
            self.applyButton.enabled = True


class WMAGroupRegisterTest(ScriptedLoadableModuleTest):

    def runTest(self):
        pass
