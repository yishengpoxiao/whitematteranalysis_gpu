# -*- coding: utf-8 -*-

"""WMA Merge & MRML Tools

Merge cluster nodes or cluster files into single tracts and create
scene MRML files (wm_append_clusters.py / wm_create_mrml_file.py,
GPU/CPU accelerated where applicable).
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


class WMAMergeMRML(ScriptedLoadableModule):

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "WMA Merge & MRML"
        self.parent.categories = [common.CATEGORY]
        self.parent.dependencies = []
        self.parent.contributors = "WhiteMatterAnalysis GPU acceleration project"
        self.parent.helpText = """
Merge cluster nodes or cluster files into single tracts
(wm_append_clusters.py) and create scene MRML files
(wm_create_mrml_file.py).
"""


class WMAMergeMRMLLogic(ScriptedLoadableModuleLogic):

    pass


class WMAMergeMRMLWidget(ScriptedLoadableModuleWidget):

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

        mergeBox = qt.QGroupBox("Merge cluster nodes in the scene")
        mergeForm = qt.QFormLayout(mergeBox)
        self.mergeNodesButton = qt.QPushButton(
            "Merge cluster_* nodes into one tract node")
        self.mergeNodesButton.toolTip = (
            "Appends every scene node named 'cluster_*' into a single "
            "tract node (wm_append_clusters.py, cluster_idx per node)")
        self.mergeNodesButton.connect('clicked(bool)', self.onMergeNodes)
        mergeForm.addRow("", self.mergeNodesButton)
        self.layout.addWidget(mergeBox)

        filesBox = qt.QGroupBox("Cluster files directory tools")
        filesForm = qt.QFormLayout(filesBox)

        self.filesDirectoryButton = qt.QPushButton("Select cluster files directory...")
        self.filesDirectory = ""
        self.filesDirectoryButton.connect('clicked(bool)', self.onPickFiles)
        filesForm.addRow("Cluster files directory:", self.filesDirectoryButton)

        self.clusterListEdit = qt.QLineEdit()
        self.clusterListEdit.toolTip = (
            "Optional 1-based cluster numbers to append, e.g. '1 2 3' "
            "(-clusterList). Empty = all files.")
        filesForm.addRow("Cluster list (optional):", self.clusterListEdit)

        self.appendFilesButton = qt.QPushButton("Append cluster files -> merged tract")
        self.appendFilesButton.enabled = False
        self.appendFilesButton.connect('clicked(bool)', self.onAppendFiles)
        filesForm.addRow("", self.appendFilesButton)

        self.createMrmlButton = qt.QPushButton("Create scene.mrml for this directory")
        self.createMrmlButton.enabled = False
        self.createMrmlButton.connect('clicked(bool)', self.onCreateMrml)
        filesForm.addRow("", self.createMrmlButton)

        self.layout.addWidget(filesBox)
        common.add_status_area(self, self.layout)
        self.statusLabel.setPlainText(
            "whitematteranalysis: OK" if wma_ok else wma_detail)

    def onPickFiles(self):
        path = common.pick_directory(slicer.util.mainWindow(),
                                     "Select cluster files directory",
                                     self.filesDirectory)
        if path:
            self.filesDirectory = path
            self.filesDirectoryButton.text = path
        self.appendFilesButton.enabled = bool(path)
        self.createMrmlButton.enabled = bool(path)

    def onMergeNodes(self):
        nodes = [n for n in slicer.util.getNodesByClass('vtkMRMLModelNode')
                 if n.GetName().startswith('cluster_')]
        try:
            nodes += [n for n in slicer.util.getNodesByClass('vtkMRMLFiberBundleNode')
                      if n.GetName().startswith('cluster_')]
        except Exception:
            pass
        if len(nodes) < 2:
            slicer.util.infoDisplay(
                "Need at least two scene nodes named 'cluster_*' to merge.")
            return
        merged = _CORE.append_polydatas([n.GetPolyData() for n in nodes])
        node = common.add_tractography_node('appended_tract')
        node.SetAndObservePolyData(merged)
        node.CreateDefaultDisplayNodes()
        slicer.util.infoDisplay(
            "Merged %d cluster nodes (%d fibers) into 'appended_tract'." % (
                len(nodes), merged.GetNumberOfLines()))

    def onAppendFiles(self):
        cluster_list = None
        text = self.clusterListEdit.text.strip()
        if text:
            try:
                cluster_list = [int(x) for x in text.replace(',', ' ').split()]
            except ValueError:
                slicer.util.errorDisplay("Cluster list must be integers, e.g. '1 2 3'.")
                return
        try:
            merged, out_path, count = _CORE.append_cluster_files(
                self.filesDirectory, self.filesDirectory,
                cluster_list=cluster_list)
            slicer.util.infoDisplay(
                "Appended %d cluster files (%d fibers) to:\n%s" % (
                    count, merged.GetNumberOfLines(), out_path))
        except Exception as exc:
            slicer.util.errorDisplay(str(exc))

    def onCreateMrml(self):
        try:
            mrml_path = _CORE.create_mrml_file(self.filesDirectory)
            slicer.util.infoDisplay("scene.mrml written to:\n%s" % mrml_path)
        except Exception as exc:
            slicer.util.errorDisplay(str(exc))


class WMAMergeMRMLTest(ScriptedLoadableModuleTest):

    def runTest(self):
        pass
