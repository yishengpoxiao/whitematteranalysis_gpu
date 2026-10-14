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

from WMARegistrationUI import RegistrationOptions
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
selected registration to the atlas, fiber labeling, atlas-based outlier
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
        self.inputLoader = common.add_tractography_loader(form, self, self.inputSelector)

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

        self.registerCheckBox = qt.QCheckBox("Register subject to atlas")
        self.registerCheckBox.checked = True
        form.addRow("", self.registerCheckBox)
        self.registrationOptions = RegistrationOptions(form)
        self.registerCheckBox.connect('toggled(bool)', self.onRegistrationToggled)

        self.outlierCheckBox = qt.QCheckBox("Remove cluster outliers (atlas-based)")
        self.outlierCheckBox.checked = True
        form.addRow("", self.outlierCheckBox)

        self.outlierWorkersSpinBox = qt.QSpinBox()
        self.outlierWorkersSpinBox.setRange(0, 800)
        self.outlierWorkersSpinBox.specialValueText = "Auto (available cores / GPU memory)"
        self.outlierWorkersSpinBox.toolTip = "0: automatic parallelism; 1: sequential. GPU concurrency is bounded by free VRAM."
        form.addRow("Outlier workers:", self.outlierWorkersSpinBox)

        self.anatomicalCheckBox = qt.QCheckBox("Map clusters to anatomical tracts")
        self.anatomicalCheckBox.checked = True
        form.addRow("", self.anatomicalCheckBox)

        self.nativeSpaceCheckBox = qt.QCheckBox("Show clusters in original subject space")
        self.nativeSpaceCheckBox.checked = True
        form.addRow("", self.nativeSpaceCheckBox)

        self.visualizationComboBox = qt.QComboBox()
        self.visualizationComboBox.addItems([
            "Anatomical tracts (otherwise one cluster overview)",
            "Individual clusters (initially hidden)",
            "Save only (no tractography nodes)"])
        form.addRow("Result display:", self.visualizationComboBox)

        self.fastDisplayCheckBox = qt.QCheckBox('Fast line display (Model nodes)')
        self.fastDisplayCheckBox.checked = True
        self.fastDisplayCheckBox.toolTip = ('Keeps all fibers and metadata, avoiding FiberBundle tube/glyph pipelines. '
                                          'Uncheck to use FiberBundle nodes when SlicerDMRI is installed.')
        form.addRow(self.fastDisplayCheckBox)

        self.saveCheckBox = qt.QCheckBox("Save cluster files and tractography")
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

    def onRegistrationToggled(self, enabled):
        self.registrationOptions.method.enabled = enabled
        self.registrationOptions.advanced.enabled = enabled

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

    def cleanup(self):
        common.cleanup_controls(self)

    def onApply(self):
        if common.workflow_running():
            common.status(self, 'Another WMA workflow is running. Wait for it to finish.')
            return
        node = self.inputSelector.currentNode()
        if node is None or not self.atlasDirectory:
            return
        polydata = common.world_polydata(node)
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
            common.start_computation(self)
            import whitematteranalysis as wma

            transform = None
            subject_pd = polydata

            if do_register:
                options = self.registrationOptions.arguments(fiber_length)
                if options['verbose'] and not (do_save and self.outputDirectory):
                    raise ValueError('Enable saving and select an output directory for intermediate outputs.')
                reg_atlas = _CORE.find_registration_atlas(self.atlasDirectory)
                common.status(self, "Registration target: %s" % reg_atlas, 3)
                atlas_pd = wma.io.read_polydata(reg_atlas)
                reg = _CORE.register_subject_to_atlas(
                    polydata, atlas_pd, fiber_length_mm=fiber_length,
                    backend=backend,
                    progress_callback=lambda t, p: common.status(self, t, p),
                    output_directory=os.path.join(self.outputDirectory, 'registration') if do_save else None,
                    **options)
                transform = reg['transform']
                self.transformNode = common.add_registration_transform(
                    node.GetName() + '_to_atlas', transform,
                    self.outputDirectory if do_save else None)
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
                        max_workers=self.outlierWorkersSpinBox.value,
                        event_callback=slicer.app.processEvents,
                        progress_callback=lambda t, p: common.status(self, t, None))
                    keep = numpy.asarray(fmask) == 1
                    result['output_polydata'] = wma.filter.mask(
                        result['output_polydata'], fmask, verbose=False)
                    result['cluster_idx'] = result['cluster_idx'][keep]
                    result['color'] = result['color'][keep]
                    common.status(self, "Outlier removal: rejected %d fibers."
                                  % rejected)

            common.release_workflow_memory(self)
            display_mode = self.visualizationComboBox.currentIndex
            pd_c_list = (_CORE.extract_clusters(result['output_polydata'], result['cluster_idx'], result['cluster_count'])
                         if do_save or display_mode == 1 else [])

            anatomical_tracts = None
            if do_anatomical:
                mrml_files = _CORE.find_anatomical_tracts_mrml(self.atlasDirectory)
                if mrml_files is None:
                    common.status(self, "Anatomical tracts skipped: no T_*.mrml files.")
                else:
                    common.status(self, "Mapping anatomical tracts...", 75)
                    callback = lambda text: common.status(self, text)
                    anatomical_tracts = _CORE.extract_anatomical_tracts(
                        result['output_polydata'], result['cluster_idx'], mrml_files, progress_callback=callback)
                    common.status(self, "Mapped into %d anatomical tracts."
                                  % len(anatomical_tracts))

            # Hemisphere mapping uses atlas coordinates; transform outputs afterward.
            if transform is not None and native_space:
                common.status(self, "Returning results to subject space...", 80)
                pd_c_list = [_CORE.apply_transform_to_polydata(
                    p, transform, inverse=True) for p in pd_c_list]
                if anatomical_tracts:
                    anatomical_tracts = [(name, _CORE.apply_transform_to_polydata(
                        pd, transform, inverse=True)) for name, pd in anatomical_tracts]

            if do_save:
                common.status(self, "Saving cluster files...", 85)
                outdir = self.outputDirectory
                fnames = common.save_clusters_parallel(pd_c_list, outdir)
                wma.io.write_polydata(
                    subject_pd, os.path.join(outdir, 'registered_subject.vtp'))
                if anatomical_tracts:
                    common.status(self, "Saving anatomical tracts...", 90)
                    anat_dir = os.path.join(outdir, 'anatomical_tracts')
                    if not os.path.isdir(anat_dir):
                        os.makedirs(anat_dir)
                    wma.io.write_polydatas_parallel(
                        [pd for name, pd in anatomical_tracts],
                        [os.path.join(anat_dir, name + '.vtp') for name, pd in anatomical_tracts])
                common.status(self, "Saved %d cluster files and registered "
                              "tractography." % len(fnames))

            display_mode = self.visualizationComboBox.currentIndex
            node_class = 'vtkMRMLModelNode' if self.fastDisplayCheckBox.checked else None
            if display_mode != 2:
                common.status(self, "Preparing result visualization...", 95)
                callback = lambda text: common.status(self, text)
                if display_mode == 1:
                    common.show_clusters_in_scene(
                        pd_c_list, base_name="pipeline", visible=False,
                        names=["cluster_%05d" % (i + 1) for i in range(len(pd_c_list))],
                                                  progress_callback=callback, node_class=node_class)
                    common.status(self, "Clusters loaded hidden; enable selected tracts in Data.")
                elif anatomical_tracts:
                    common.show_clusters_in_scene(
                        [pd for name, pd in anatomical_tracts], base_name="pipeline",
                        names=[name for name, pd in anatomical_tracts],
                        progress_callback=callback, node_class=node_class)
                else:
                    overview = result['output_polydata']
                    if transform is not None and native_space:
                        overview = _CORE.apply_transform_to_polydata(overview, transform, inverse=True)
                    common.show_clusters_in_scene([overview], base_name="pipeline",
                                                  names=["WMA cluster overview"], node_class=node_class)
                slicer.util.resetThreeDViews()

            common.status(self, "PIPELINE DONE: %d fibers, %d clusters, backend %s."
                          % (result['output_polydata'].GetNumberOfLines(), result['cluster_count'],
                             result['backend_name']), 100)
        except Exception:
            import traceback
            common.status(self, "ERROR:\n" + traceback.format_exc())
        finally:
            common.finish_computation(self)
            self.applyButton.enabled = True


class WMAPipelineTest(ScriptedLoadableModuleTest):

    def runTest(self):
        pass
