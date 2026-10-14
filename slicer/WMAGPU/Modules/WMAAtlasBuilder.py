# -*- coding: utf-8 -*-

"""WMA Atlas Builder

Build a WMA tract atlas from a directory of (registered) subject
tractographies: Nystrom spectral clustering plus iterative
leave-one-out cluster outlier rejection (wm_cluster_atlas.py core
loop, GPU/CPU accelerated).
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

import WMAGuiCommon as common
import WMAtlasLabelGPUCore as _CORE


class WMAAtlasBuilder(ScriptedLoadableModule):

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "WMA Atlas Builder"
        self.parent.categories = [common.CATEGORY]
        self.parent.dependencies = []
        self.parent.contributors = "WhiteMatterAnalysis GPU acceleration project"
        self.parent.helpText = """
Build a WMA tract atlas from a directory of registered subject
tractographies (GPU/CPU accelerated spectral clustering with
outlier rejection). The resulting directory (containing atlas.p,
atlas.vtp and cluster files) can be used by WMA Pipeline.
"""


class WMAAtlasBuilderLogic(ScriptedLoadableModuleLogic):

    def run(self, subject_directory, output_directory, fibers_per_subject,
            number_of_clusters, iterations, backend, progress_callback=None):
        return _CORE.build_atlas(
            subject_directory, output_directory,
            fibers_per_subject=fibers_per_subject,
            number_of_clusters=number_of_clusters,
            iterations=iterations, backend=backend,
            progress_callback=progress_callback)


class WMAAtlasBuilderWidget(ScriptedLoadableModuleWidget):

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

        formBox = qt.QGroupBox("Build atlas from registered subjects")
        form = qt.QFormLayout(formBox)

        self.subjectsDirectoryButton = qt.QPushButton("Select subject directory...")
        self.subjectsDirectory = ""
        self.subjectsDirectoryButton.connect('clicked(bool)', self.onPickSubjects)
        form.addRow("Subject directory:", self.subjectsDirectoryButton)

        self.outputDirectoryButton = qt.QPushButton("Select atlas output directory...")
        self.outputDirectory = ""
        self.outputDirectoryButton.connect('clicked(bool)', self.onPickOutput)
        form.addRow("Atlas output:", self.outputDirectoryButton)

        self.fibersPerSubjectSpin = qt.QSpinBox()
        self.fibersPerSubjectSpin.value = 500
        self.fibersPerSubjectSpin.minimum = 100
        self.fibersPerSubjectSpin.maximum = 20000
        self.fibersPerSubjectSpin.singleStep = 100
        form.addRow("Fibers per subject:", self.fibersPerSubjectSpin)

        self.clustersSpin = qt.QSpinBox()
        self.clustersSpin.value = 20
        self.clustersSpin.minimum = 2
        self.clustersSpin.maximum = 1000
        form.addRow("Number of clusters:", self.clustersSpin)

        self.iterationsSpin = qt.QSpinBox()
        self.iterationsSpin.value = 2
        self.iterationsSpin.minimum = 1
        self.iterationsSpin.maximum = 10
        form.addRow("Iterations:", self.iterationsSpin)

        common.add_backend_row(form, self)

        self.applyButton = qt.QPushButton("Build atlas")
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
                                     "Select atlas output directory", self.outputDirectory)
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
        logic = WMAAtlasBuilderLogic()
        self.applyButton.enabled = False
        try:
            common.start_computation(self)
            result = logic.run(self.subjectsDirectory, self.outputDirectory,
                               self.fibersPerSubjectSpin.value,
                               self.clustersSpin.value,
                               self.iterationsSpin.value, backend,
                               progress_callback=lambda t, p: common.status(self, t, p))
            common.status(self,
                          "Atlas built: %d subjects, %d clusters, %d fibers.\n"
                          "Use this atlas directory in WMA Pipeline:\n%s"
                          % (result['n_subjects'], result['cluster_count'],
                             result['n_fibers'], result['atlas_directory']), 100)
        except Exception:
            import traceback
            common.status(self, "ERROR:\n" + traceback.format_exc())
        finally:
            common.finish_computation(self)
            self.applyButton.enabled = True


class WMAAtlasBuilderTest(ScriptedLoadableModuleTest):

    def runTest(self):
        pass
