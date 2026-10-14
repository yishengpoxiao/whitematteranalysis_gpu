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
Check tractography nodes and group them in a named folder or combine
their fibers into one named node. Directory tools append cluster files
(wm_append_clusters.py) or create scene MRML files (wm_create_mrml_file.py).
"""


class WMAMergeMRMLLogic(ScriptedLoadableModuleLogic):

    def merge(self, nodes, name, combine=False):
        if not nodes:
            raise ValueError('Check at least one tractography node.')
        if not name.strip():
            raise ValueError('Specify an output name.')
        scene = slicer.mrmlScene
        slicer.app.pauseRender()
        scene.StartState(scene.BatchProcessState)
        try:
            if combine:
                merged = _CORE.append_polydatas([common.world_polydata(node) for node in nodes])
                result = common.add_tractography_node(name.strip())
                result.SetAndObservePolyData(merged)
                return result
            hierarchy = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(scene)
            folder = hierarchy.CreateFolderItem(hierarchy.GetSceneItemID(), name.strip())
            for node in nodes:
                hierarchy.SetItemParent(hierarchy.GetItemByDataNode(node), folder)
            return folder
        finally:
            scene.EndState(scene.BatchProcessState)
            slicer.app.resumeRender()


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
        self.nodeTable = qt.QTableWidget()
        self.nodeTable.setColumnCount(2)
        self.nodeTable.setHorizontalHeaderLabels(['Tractography node', 'Fibers'])
        self.nodeTable.minimumHeight = 200
        self.nodeTable.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        mergeForm.addRow(self.nodeTable)
        buttons = qt.QWidget()
        row = qt.QHBoxLayout(buttons)
        self.refreshButton = qt.QPushButton('Refresh nodes')
        self.selectAllButton = qt.QPushButton('Select all')
        self.selectNoneButton = qt.QPushButton('Clear selection')
        for button in (self.refreshButton, self.selectAllButton, self.selectNoneButton):
            row.addWidget(button)
        self.refreshButton.connect('clicked(bool)', self.refreshNodes)
        self.selectAllButton.connect('clicked(bool)', lambda: self.checkAll(True))
        self.selectNoneButton.connect('clicked(bool)', lambda: self.checkAll(False))
        mergeForm.addRow(buttons)
        self.outputNameEdit = qt.QLineEdit('merged_tract')
        mergeForm.addRow('Output name:', self.outputNameEdit)
        self.mergeMethodComboBox = qt.QComboBox()
        self.mergeMethodComboBox.addItems([
            'Group separate nodes in a named folder', 'Combine all fibers into one tractography node'])
        self.mergeMethodComboBox.toolTip = (
            'Folder: keeps geometry, colors and transforms of each checked node. '
            'Combined node: appends geometry in scene coordinates, retaining cluster_idx provenance.')
        mergeForm.addRow('Merge method:', self.mergeMethodComboBox)
        self.mergeNodesButton = qt.QPushButton('Merge checked nodes')
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
        self._refreshTimer = qt.QTimer()
        self._refreshTimer.setSingleShot(True)
        self._refreshTimer.connect('timeout()', self.refreshNodes)
        self._sceneObservers = [slicer.mrmlScene.AddObserver(event, self.scheduleRefresh)
                                for event in (slicer.mrmlScene.NodeAddedEvent, slicer.mrmlScene.NodeRemovedEvent)]
        self._sceneObservers.append(slicer.mrmlScene.AddObserver(
            slicer.mrmlScene.EndBatchProcessEvent, lambda *args: self._refreshTimer.start(0)))
        self.refreshNodes()

    def cleanup(self):
        self._refreshTimer.stop()
        for observer in getattr(self, '_sceneObservers', []):
            slicer.mrmlScene.RemoveObserver(observer)
        self._sceneObservers = []

    def enter(self):
        self.refreshNodes()

    def scheduleRefresh(self, *args):
        if not slicer.mrmlScene.IsBatchProcessing():
            self._refreshTimer.start(0)

    def selectedNodes(self):
        nodes = []
        for row in range(self.nodeTable.rowCount):
            item = self.nodeTable.item(row, 0)
            if item.checkState() == qt.Qt.Checked:
                node = slicer.mrmlScene.GetNodeByID(item.data(qt.Qt.UserRole))
                if node is not None:
                    nodes.append(node)
        return nodes

    def refreshNodes(self):
        self._refreshTimer.stop()
        if slicer.mrmlScene.IsBatchProcessing():
            return
        selected = {node.GetID() for node in self.selectedNodes()}
        nodes = list({node.GetID(): node for cls in common.tractography_node_types()
                      for node in slicer.util.getNodesByClass(cls)
                      if node.GetPolyData() is not None and node.GetPolyData().GetNumberOfLines() > 0}.values())
        self.nodeTable.setUpdatesEnabled(False)
        try:
            self.nodeTable.setRowCount(len(nodes))
            for row, node in enumerate(nodes):
                item = qt.QTableWidgetItem(node.GetName())
                item.setFlags(qt.Qt.ItemIsEnabled | qt.Qt.ItemIsSelectable | qt.Qt.ItemIsUserCheckable)
                item.setData(qt.Qt.UserRole, node.GetID())
                item.setCheckState(qt.Qt.Checked if node.GetID() in selected else qt.Qt.Unchecked)
                self.nodeTable.setItem(row, 0, item)
                fibers = qt.QTableWidgetItem(str(node.GetPolyData().GetNumberOfLines()))
                fibers.setFlags(qt.Qt.ItemIsEnabled | qt.Qt.ItemIsSelectable)
                self.nodeTable.setItem(row, 1, fibers)
            self.nodeTable.resizeColumnsToContents()
        finally:
            self.nodeTable.setUpdatesEnabled(True)

    def checkAll(self, checked):
        for row in range(self.nodeTable.rowCount):
            self.nodeTable.item(row, 0).setCheckState(qt.Qt.Checked if checked else qt.Qt.Unchecked)

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
        try:
            nodes = self.selectedNodes()
            self.mergeResult = WMAMergeMRMLLogic().merge(
                nodes, self.outputNameEdit.text, self.mergeMethodComboBox.currentIndex == 1)
            common.status(self, 'Merged %d selected nodes as %s.' % (len(nodes), self.outputNameEdit.text), 100)
            self.refreshNodes()
        except Exception as exc:
            slicer.util.errorDisplay(str(exc))

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
