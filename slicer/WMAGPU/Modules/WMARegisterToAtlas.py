# -*- coding: utf-8 -*-

"""Moving-to-fixed tractography registration, including atlas targets."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(HERE, '_vendor')
for p in (VENDOR,):
    if p not in sys.path:
        sys.path.insert(0, p)

import qt

import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule, ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic, ScriptedLoadableModuleTest)

from WMARegistrationUI import RegistrationOptions
import WMAGuiCommon as common
import WMAtlasLabelGPUCore as _CORE


class WMARegisterToAtlas(ScriptedLoadableModule):

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "WMA Tractography Registration"
        self.parent.categories = [common.CATEGORY]
        self.parent.dependencies = []
        self.parent.contributors = "WhiteMatterAnalysis GPU acceleration project"
        self.parent.helpText = """
Register a moving tractography node to a fixed tractography node or an atlas.
Choose a WMA CLI preset, rigid registration, or a custom multiscale schedule.
The final transform remains in the scene and is saved when file output is enabled.
Nonrigid registration expects inputs aligned with rigid or affine first.
"""


class WMARegisterToAtlasLogic(ScriptedLoadableModuleLogic):

    def register(self, subject_polydata, atlas_polydata, fiber_length_mm,
                 backend, progress_callback=None, **options):
        return _CORE.register_subject_to_atlas(
            subject_polydata, atlas_polydata,
            fiber_length_mm=fiber_length_mm, backend=backend,
            progress_callback=progress_callback, **options)

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

        formBox = qt.QGroupBox("Register moving tractography to fixed tractography")
        form = qt.QFormLayout(formBox)

        self.inputSelector = slicer.qMRMLNodeComboBox()
        self.inputSelector.nodeTypes = common.tractography_node_types()
        self.inputSelector.noneEnabled = False
        self.inputSelector.setMRMLScene(slicer.mrmlScene)
        form.addRow("Moving tractography:", self.inputSelector)
        self.inputLoader = common.add_tractography_loader(form, self, self.inputSelector)

        self.targetComboBox = qt.QComboBox()
        self.targetComboBox.addItems(['Fixed tractography node', 'Atlas directory'])
        form.addRow('Fixed target source:', self.targetComboBox)
        self.fixedSelector = slicer.qMRMLNodeComboBox()
        self.fixedSelector.nodeTypes = common.tractography_node_types()
        self.fixedSelector.noneEnabled = True
        self.fixedSelector.setMRMLScene(slicer.mrmlScene)
        form.addRow('Fixed tractography:', self.fixedSelector)
        self.fixedLoader = common.add_tractography_loader(form, self, self.fixedSelector, 'Load fixed tractography file...')

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

        self.registrationOptions = RegistrationOptions(form)
        common.add_backend_row(form, self)
        self.targetComboBox.connect('currentIndexChanged(int)', self.onTargetChanged)
        self.onTargetChanged()

        self.saveCheckBox = qt.QCheckBox("Save registered tractography and transform")
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
            self._CORE is not None and self.inputSelector.currentNode() is not None
            and not common.workflow_running() and getattr(self, '_loadingCount', 0) == 0)

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

    def onTargetChanged(self, *args):
        atlas = self.targetComboBox.currentIndex == 1
        self.fixedSelector.enabled = not atlas
        self.fixedLoader.button.enabled = not atlas
        self.atlasDirectoryButton.enabled = atlas

    def cleanup(self):
        common.cleanup_controls(self)

    def onApply(self):
        if common.workflow_running():
            common.status(self, 'Another WMA workflow is running. Wait for it to finish.')
            return
        node = self.inputSelector.currentNode()
        fixed = self.fixedSelector.currentNode()
        use_atlas = self.targetComboBox.currentIndex == 1
        if node is None or (use_atlas and not self.atlasDirectory) or (not use_atlas and fixed is None):
            slicer.util.errorDisplay('Select moving tractography and a fixed target first.')
            return
        if not use_atlas and node == fixed:
            slicer.util.errorDisplay('Choose different moving and fixed tractography nodes.')
            return
        if self.saveCheckBox.checked and not self.outputDirectory:
            slicer.util.errorDisplay('Select an output directory or uncheck saving.')
            return
        self.applyButton.enabled = False
        try:
            common.start_computation(self)
            import whitematteranalysis as wma
            options = self.registrationOptions.arguments(self.fiberLengthSpinBox.value)
            polydata = common.world_polydata(node)
            if use_atlas:
                target = _CORE.find_registration_atlas(self.atlasDirectory)
                common.status(self, 'Registration target: %s' % target, 5)
                fixed_pd = wma.io.read_polydata(target)
                target_name = 'atlas'
            else:
                fixed_pd = common.world_polydata(fixed)
                target_name = fixed.GetName()
            if polydata.GetNumberOfLines() == 0 or fixed_pd.GetNumberOfLines() == 0:
                raise ValueError('Moving and fixed nodes must contain tractography lines.')
            output = self.outputDirectory if self.saveCheckBox.checked else None
            if options['verbose'] and not output:
                raise ValueError('Select an output directory and enable saving for intermediate outputs.')
            reg = WMARegisterToAtlasLogic().register(
                polydata, fixed_pd, self.fiberLengthSpinBox.value,
                common.backend_key(self.backendComboBox.currentIndex),
                progress_callback=lambda t, p: common.status(self, t, p),
                output_directory=output, **options)
            name = node.GetName() + '_to_' + target_name
            self.transformNode = common.add_registration_transform(name, reg['transform'], output)
            self.outputNode = common.add_tractography_node(node.GetName() + '_registered', 'vtkMRMLModelNode')
            self.outputNode.SetAndObservePolyData(polydata)
            self.outputNode.SetAndObserveTransformNodeID(self.transformNode.GetID())
            if output:
                registered = _CORE.apply_transform_to_polydata(polydata, reg['transform'])
                filename = os.path.join(output, os.path.basename(node.GetName()) + '_reg.vtp')
                wma.io.write_polydata(registered, filename)
                common.status(self, 'Saved: %s' % filename)
            common.status(self, 'Registration done (%d iterations, backend: %s). Transform: %s' % (
                reg['iterations'], reg['backend_name'], self.transformNode.GetName()), 100)
        except Exception:
            import traceback
            common.status(self, 'ERROR:\n' + traceback.format_exc())
        finally:
            common.finish_computation(self)
            self.applyButton.enabled = True


class WMARegisterToAtlasTest(ScriptedLoadableModuleTest):

    def runTest(self):
        pass
