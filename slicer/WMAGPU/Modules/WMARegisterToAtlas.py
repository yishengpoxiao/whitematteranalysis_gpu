# -*- coding: utf-8 -*-

"""WMA Register to Atlas

Affine registration of a subject whole-brain tractography to a WMA
atlas (wm_register_to_atlas_new.py affine defaults, GPU/CPU
accelerated objective). The registration target is found
automatically: official atlases use ORG-RegAtlas-100HCP/
registration_atlas.vtk; self-built atlases fall back to atlas.vtp.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(HERE, '_vendor')
for p in (VENDOR,):
    if p not in sys.path:
        sys.path.insert(0, p)

import qt
import vtk

import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule, ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic, ScriptedLoadableModuleTest)

import WMAGuiCommon as common
import WMAtlasLabelGPUCore as _CORE


class WMARegisterToAtlas(ScriptedLoadableModule):

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "WMA Register to Atlas"
        self.parent.categories = [common.CATEGORY]
        self.parent.dependencies = []
        self.parent.contributors = "WhiteMatterAnalysis GPU acceleration project"
        self.parent.helpText = """
Affine registration of a subject whole-brain tractography to a WMA
atlas (GPU/CPU accelerated). Provide the atlas directory; the
registration target (ORG-RegAtlas registration_atlas.vtk, or atlas.vtp
for self-built atlases) is located automatically.
"""


class WMARegisterToAtlasLogic(ScriptedLoadableModuleLogic):

    def register(self, subject_polydata, atlas_polydata, fiber_length_mm,
                 backend, progress_callback=None):
        return _CORE.register_subject_to_atlas(
            subject_polydata, atlas_polydata,
            fiber_length_mm=fiber_length_mm, backend=backend,
            progress_callback=progress_callback)

    @staticmethod
    def apply_transform(polydata, transform):
        return _CORE.apply_transform_to_polydata(polydata, transform)


class WMARegisterToAtlasWidget(ScriptedLoadableModuleWidget):

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
        _CORE = self._CORE

        formBox = qt.QGroupBox("Register subject to atlas (affine)")
        form = qt.QFormLayout(formBox)

        self.inputSelector = slicer.qMRMLNodeComboBox()
        self.inputSelector.nodeTypes = common.tractography_node_types()
        self.inputSelector.noneEnabled = False
        self.inputSelector.setMRMLScene(slicer.mrmlScene)
        form.addRow("Input tractography:", self.inputSelector)

        self.atlasDirectoryButton = qt.QPushButton("Select atlas directory...")
        self.atlasDirectory = ""
        self.atlasDirectoryButton.connect('clicked(bool)', self.onPickAtlas)
        form.addRow("Atlas directory:", self.atlasDirectoryButton)

        self.fiberLengthSpinBox = qt.QDoubleSpinBox()
        self.fiberLengthSpinBox.value = 40.0
        self.fiberLengthSpinBox.minimum = 0.0
        self.fiberLengthSpinBox.maximum = 200.0
        self.fiberLengthSpinBox.suffix = " mm"
        form.addRow("Minimum fiber length:", self.fiberLengthSpinBox)

        common.add_backend_row(form, self)

        self.saveCheckBox = qt.QCheckBox("Save registered tractography to file")
        self.saveCheckBox.checked = True
        form.addRow("", self.saveCheckBox)

        self.outputDirectoryButton = qt.QPushButton("Select output directory...")
        self.outputDirectory = ""
        self.outputDirectoryButton.connect('clicked(bool)', self.onPickOutput)
        form.addRow("Output directory:", self.outputDirectoryButton)

        self.applyButton = qt.QPushButton("Register")
        self.applyButton.connect('clicked(bool)', self.onApply)
        form.addRow("", self.applyButton)

        self.layout.addWidget(formBox)
        common.add_status_area(self, self.layout)

        self.applyButton.enabled = wma_ok and self.inputSelector.currentNode() is not None
        self.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onInputChanged)
        self.statusLabel.setPlainText(
            "whitematteranalysis: OK" if wma_ok else wma_detail)

    def onInputChanged(self, *args):
        self.applyButton.enabled = (
            self._CORE is not None and self.inputSelector.currentNode() is not None)

    def onPickAtlas(self):
        path = common.pick_directory(slicer.util.mainWindow(),
                                     "Select atlas directory", self.atlasDirectory)
        if path:
            self.atlasDirectory = path
            self.atlasDirectoryButton.text = path

    def onPickOutput(self):
        path = common.pick_directory(slicer.util.mainWindow(),
                                     "Select output directory", self.outputDirectory)
        if path:
            self.outputDirectory = path
            self.outputDirectoryButton.text = path

    def onApply(self):
        node = self.inputSelector.currentNode()
        if node is None or not self.atlasDirectory:
            return
        polydata = node.GetPolyData()
        if polydata is None or polydata.GetNumberOfLines() == 0:
            slicer.util.errorDisplay("The selected node has no tractography lines.")
            return

        backend = common.backend_key(self.backendComboBox.currentIndex)
        logic = WMARegisterToAtlasLogic()
        self.applyButton.enabled = False
        try:
            import whitematteranalysis as wma
            reg_atlas = _CORE.find_registration_atlas(self.atlasDirectory)
            common.status(self, "Registration target: %s" % reg_atlas, 5)
            atlas_pd = wma.io.read_polydata(reg_atlas)

            reg = logic.register(polydata, atlas_pd,
                                 self.fiberLengthSpinBox.value, backend,
                                 progress_callback=lambda t, p: common.status(self, t, p))
            registered = logic.apply_transform(polydata, reg['transform'])

            out_node = common.add_tractography_node(node.GetName() + '_registered')
            out_node.SetAndObservePolyData(registered)
            out_node.CreateDefaultDisplayNodes()

            if self.saveCheckBox.checked and self.outputDirectory:
                fname = os.path.join(self.outputDirectory,
                                     os.path.basename(node.GetName()) + '_reg.vtp')
                import whitematteranalysis as wma
                wma.io.write_polydata(registered, fname)
                common.status(self, "Saved: %s" % fname)

            common.status(self, "Registration done (%d iterations, backend: %s)." % (
                reg['iterations'], reg['backend_name']), 100)
        except Exception:
            import traceback
            common.status(self, "ERROR:\n" + traceback.format_exc())
        finally:
            self.applyButton.enabled = True


class WMARegisterToAtlasTest(ScriptedLoadableModuleTest):

    def runTest(self):
        pass
