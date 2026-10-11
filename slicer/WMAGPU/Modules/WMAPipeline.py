# -*- coding: utf-8 -*-

"""WMA Pipeline

One-click full ORG pipeline for a subject tractography:
register to atlas -> label fibers -> remove outliers -> transform
clusters back to subject space -> map to anatomical tracts.
Every step runs on the GPU or the vectorized CPU backend.
"""

import os
import sys

import numpy

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


class WMAPipeline(ScriptedLoadableModule):

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "WMA Pipeline"
        self.parent.categories = [common.CATEGORY]
        self.parent.dependencies = []
        self.parent.contributors = "WhiteMatterAnalysis GPU acceleration project"
        self.parent.helpText = """
One-click full pipeline (equivalent to wm_apply_ORG_atlas_to_subject.sh):
affine registration to the atlas, fiber labeling, atlas-based outlier
removal, mapping back to subject space and to the anatomically named
tracts of the atlas. Just provide the subject tractography and the
atlas directory.
"""


class WMAPipelineLogic(ScriptedLoadableModuleLogic):

    pass


class WMAPipelineWidget(ScriptedLoadableModuleWidget):

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

        formBox = qt.QGroupBox("Full pipeline (register -> label -> outliers -> anatomical tracts)")
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

        self.registerCheckBox = qt.QCheckBox("Register subject to atlas (affine)")
        self.registerCheckBox.checked = True
        form.addRow("", self.registerCheckBox)

        self.outlierCheckBox = qt.QCheckBox("Remove cluster outliers (atlas-based)")
        self.outlierCheckBox.checked = True
        form.addRow("", self.outlierCheckBox)

        self.anatomicalCheckBox = qt.QCheckBox("Map clusters to anatomical tracts")
        self.anatomicalCheckBox.checked = True
        form.addRow("", self.anatomicalCheckBox)

        self.nativeSpaceCheckBox = qt.QCheckBox("Show clusters in original subject space")
        self.nativeSpaceCheckBox.checked = True
        form.addRow("", self.nativeSpaceCheckBox)

        self.saveCheckBox = qt.QCheckBox("Save cluster files and registered tractography")
        self.saveCheckBox.checked = True
        form.addRow("", self.saveCheckBox)

        self.outputDirectoryButton = qt.QPushButton("Select output directory...")
        self.outputDirectory = ""
        self.outputDirectoryButton.connect('clicked(bool)', self.onPickOutput)
        form.addRow("Output directory:", self.outputDirectoryButton)

        self.applyButton = qt.QPushButton("Run full pipeline")
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
        if self.saveCheckBox.checked and not self.outputDirectory:
            slicer.util.errorDisplay(
                "Select an output directory (or uncheck saving).")
            return

        backend = common.backend_key(self.backendComboBox.currentIndex)
        do_register = self.registerCheckBox.checked
        do_outliers = self.outlierCheckBox.checked
        do_anatomical = self.anatomicalCheckBox.checked
        native_space = self.nativeSpaceCheckBox.checked
        do_save = self.saveCheckBox.checked
        fiber_length = self.fiberLengthSpinBox.value

        self.applyButton.enabled = False
        try:
            import whitematteranalysis as wma

            transform = None
            subject_pd = polydata

            if do_register:
                reg_atlas = _CORE.find_registration_atlas(self.atlasDirectory)
                common.status(self, "Registration target: %s" % reg_atlas, 3)
                atlas_pd = wma.io.read_polydata(reg_atlas)
                reg = _CORE.register_subject_to_atlas(
                    polydata, atlas_pd, fiber_length_mm=fiber_length,
                    backend=backend,
                    progress_callback=lambda t, p: common.status(self, t, p))
                transform = reg['transform']
                subject_pd = _CORE.apply_transform_to_polydata(polydata, transform)
                common.status(self, "Registration done (%d iterations)."
                              % reg['iterations'])

            result = _CORE.run_atlas_labeling(
                subject_pd, self.atlasDirectory, fiber_length_mm=fiber_length,
                backend=backend,
                progress_callback=lambda t, p: common.status(self, t, 30 + int((p or 0) * 0.4)))

            if do_outliers:
                cluster_dir = _CORE.find_atlas_cluster_directory(self.atlasDirectory)
                if cluster_dir is None:
                    common.status(self, "Outlier removal skipped: no atlas cluster files.")
                else:
                    fmask, rejected = _CORE.remove_cluster_outliers(
                        result['output_polydata'], result['cluster_idx'],
                        result['cluster_count'], cluster_dir, backend=backend,
                        progress_callback=lambda t, p: common.status(self, t, None))
                    keep = numpy.asarray(fmask) == 1
                    result['output_polydata'] = wma.filter.mask(
                        result['output_polydata'], fmask, verbose=False)
                    result['cluster_idx'] = result['cluster_idx'][keep]
                    result['color'] = result['color'][keep]
                    common.status(self, "Outlier removal: rejected %d fibers."
                                  % rejected)

            pd_c_list = _CORE.extract_clusters(
                result['output_polydata'], result['cluster_idx'],
                result['cluster_count'])

            anatomical_tracts = None
            if transform is not None and native_space:
                pd_c_list = [_CORE.apply_transform_to_polydata(
                    p, transform, inverse=True) for p in pd_c_list]

            if do_anatomical:
                mrml_files = _CORE.find_anatomical_tracts_mrml(self.atlasDirectory)
                if mrml_files is None:
                    common.status(self, "Anatomical tracts skipped: no T_*.mrml files.")
                else:
                    anat_dir = (os.path.join(self.outputDirectory, 'anatomical_tracts')
                                if do_save else None)
                    sep = _CORE.separate_clusters_by_hemisphere(
                        pd_c_list, output_directory=anat_dir)
                    anatomical_tracts = _CORE.append_clusters_to_anatomical_tracts(
                        pd_c_list, mrml_files, sep, output_directory=anat_dir)
                    common.status(self, "Mapped into %d anatomical tracts."
                                  % len(anatomical_tracts))

            common.show_clusters_in_scene(pd_c_list)

            if do_save:
                outdir = self.outputDirectory
                fnames = common.save_clusters_parallel(pd_c_list, outdir)
                wma.io.write_polydata(
                    subject_pd, os.path.join(outdir, 'registered_subject.vtp'))
                if anatomical_tracts:
                    anat_dir = os.path.join(outdir, 'anatomical_tracts')
                    if not os.path.isdir(anat_dir):
                        os.makedirs(anat_dir)
                    for tract_name, pd_tract in anatomical_tracts:
                        wma.io.write_polydata(
                            pd_tract, os.path.join(anat_dir, tract_name + '.vtp'))
                common.status(self, "Saved %d cluster files and registered "
                              "tractography." % len(fnames))

            common.status(self, "PIPELINE DONE: %d fibers, %d clusters, backend %s."
                          % (result['n_fibers'], result['cluster_count'],
                             result['backend_name']), 100)
        except Exception:
            import traceback
            common.status(self, "ERROR:\n" + traceback.format_exc())
        finally:
            self.applyButton.enabled = True


class WMAPipelineTest(ScriptedLoadableModuleTest):

    def runTest(self):
        pass
