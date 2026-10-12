# -*- coding: utf-8 -*-

"""WMA Label From Atlas

Label every fiber of a subject whole-brain tractography with the
anatomical clusters of a WMA atlas (wm_cluster_from_atlas.py core,
GPU/CPU accelerated), with optional atlas-based outlier removal and
anatomical tract mapping.
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


class WMALabelFromAtlas(ScriptedLoadableModule):

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "WMA Label From Atlas"
        self.parent.categories = [common.CATEGORY]
        self.parent.dependencies = []
        self.parent.contributors = "WhiteMatterAnalysis GPU acceleration project"
        self.parent.helpText = """
Label the fibers of a subject whole-brain tractography with the
anatomical clusters of a WMA tract atlas (GPU/CPU accelerated).
Optional: atlas-based outlier removal, mapping to anatomical tracts,
and saving cluster files.
"""


class WMALabelFromAtlasLogic(ScriptedLoadableModuleLogic):

    def run(self, input_polydata, atlas_directory, fiber_length_mm,
            backend, progress_callback=None):
        return _CORE.run_atlas_labeling(
            input_polydata, atlas_directory, fiber_length_mm=fiber_length_mm,
            backend=backend, number_of_jobs=1,
            progress_callback=progress_callback)


class WMALabelFromAtlasWidget(ScriptedLoadableModuleWidget):

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

        formBox = qt.QGroupBox("Label subject fibers with a WMA atlas")
        form = qt.QFormLayout(formBox)

        self.inputSelector = slicer.qMRMLNodeComboBox()
        self.inputSelector.nodeTypes = common.tractography_node_types()
        self.inputSelector.noneEnabled = False
        self.inputSelector.setMRMLScene(slicer.mrmlScene)
        form.addRow("Input tractography:", self.inputSelector)

        self.atlasDirectoryButton = qt.QPushButton("Select atlas directory...")
        self.atlasDirectoryButton.toolTip = "Select an ORG parent, ORG-800FC-100HCP, or atlas build/iteration directory."
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

        self.outlierCheckBox = qt.QCheckBox("Remove cluster outliers (atlas-based)")
        form.addRow("", self.outlierCheckBox)

        self.anatomicalCheckBox = qt.QCheckBox("Map clusters to anatomical tracts")
        self.anatomicalCheckBox.toolTip = (
            "Needs an ORG-style atlas providing T_*.mrml files "
            "(auto-detected in the atlas directory).")
        form.addRow("", self.anatomicalCheckBox)

        self.saveCheckBox = qt.QCheckBox("Save cluster files (parallel)")
        self.saveCheckBox.checked = True
        form.addRow("", self.saveCheckBox)

        self.outputDirectoryButton = qt.QPushButton("Select output directory...")
        self.outputDirectory = ""
        self.outputDirectoryButton.connect('clicked(bool)', self.onPickOutput)
        form.addRow("Output directory:", self.outputDirectoryButton)

        self.applyButton = qt.QPushButton("Label fibers")
        self.applyButton.connect('clicked(bool)', self.onApply)
        form.addRow("", self.applyButton)

        self.layout.addWidget(formBox)
        common.add_status_area(self, self.layout)
        self.statusLabel.setPlainText(
            "whitematteranalysis: OK" if wma_ok else wma_detail)

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
        logic = WMALabelFromAtlasLogic()
        self.applyButton.enabled = False
        try:
            result = logic.run(polydata, self.atlasDirectory,
                               self.fiberLengthSpinBox.value, backend,
                               progress_callback=lambda t, p: common.status(self, t, p))

            if self.outlierCheckBox.checked:
                cluster_dir = _CORE.find_atlas_cluster_directory(self.atlasDirectory)
                if cluster_dir is None:
                    common.status(self, "Outlier removal skipped: no atlas cluster files.")
                else:
                    fmask, rejected = _CORE.remove_cluster_outliers(
                        result['output_polydata'], result['cluster_idx'],
                        result['cluster_count'], cluster_dir, backend=backend,
                        progress_callback=lambda t, p: common.status(self, t, None))
                    import numpy
                    keep = numpy.asarray(fmask) == 1
                    import whitematteranalysis as wma
                    result['output_polydata'] = wma.filter.mask(
                        result['output_polydata'], fmask, verbose=False)
                    result['cluster_idx'] = result['cluster_idx'][keep]
                    result['color'] = result['color'][keep]
                    common.status(self, "Outlier removal: rejected %d fibers."
                                  % rejected)

            pd_c_list = _CORE.extract_clusters(
                result['output_polydata'], result['cluster_idx'],
                result['cluster_count'])
            common.show_clusters_in_scene(pd_c_list)

            if self.anatomicalCheckBox.checked:
                mrml_files = _CORE.find_anatomical_tracts_mrml(self.atlasDirectory)
                if mrml_files is None:
                    common.status(self, "Anatomical tracts skipped: no T_*.mrml files.")
                else:
                    anat_dir = (os.path.join(self.outputDirectory, 'anatomical_tracts')
                                if self.saveCheckBox.checked and self.outputDirectory
                                else None)
                    sep = _CORE.separate_clusters_by_hemisphere(
                        pd_c_list, output_directory=anat_dir)
                    tracts = _CORE.append_clusters_to_anatomical_tracts(
                        pd_c_list, mrml_files, sep, output_directory=anat_dir)
                    common.status(self, "Mapped into %d anatomical tracts."
                                  % len(tracts))

            if self.saveCheckBox.checked and self.outputDirectory:
                fnames = common.save_clusters_parallel(pd_c_list,
                                                       self.outputDirectory)
                common.status(self, "Saved %d cluster files." % len(fnames))

            common.status(self, "Done: %d fibers, %d clusters." % (
                result['n_fibers'], result['cluster_count']), 100)
        except Exception:
            import traceback
            common.status(self, "ERROR:\n" + traceback.format_exc())
        finally:
            self.applyButton.enabled = True


class WMALabelFromAtlasTest(ScriptedLoadableModuleTest):

    def runTest(self):
        pass
