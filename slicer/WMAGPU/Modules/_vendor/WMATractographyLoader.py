"""Read tractography off the GUI thread; create scene nodes only on the GUI thread."""

import os


class TractographyLoader:
    def __init__(self, form, widget, selector, label):
        import qt
        import slicer
        self.widget, self.selector = widget, selector
        self.button = qt.QPushButton(label)
        self.show = qt.QCheckBox('Show after loading')
        self.show.toolTip = 'Leave unchecked to avoid rendering a large whole-brain input during processing.'
        row = qt.QWidget()
        layout = qt.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.button)
        layout.addWidget(self.show)
        form.addRow(row)
        self.timer = qt.QTimer(slicer.app)
        self.timer.setInterval(100)
        self.timer.connect('timeout()', self.poll)
        self.button.connect('clicked(bool)', self.pick)
        self.future = self.pool = None
        self.progress = 0.

    def pick(self):
        import qt
        import slicer
        filename = qt.QFileDialog.getOpenFileName(slicer.util.mainWindow(), 'Load tractography', '', 'Tractography (*.vtk *.vtp)')
        if filename:
            self.load(filename)

    def load(self, filename):
        from concurrent.futures import ThreadPoolExecutor
        from WMAGPUControls import ACTIVE_WORKFLOW
        import WMAGuiCommon as common
        if self.future is not None or ACTIVE_WORKFLOW is not None:
            common.status(self.widget, 'Wait for the active load or WMA computation to finish.')
            return
        self.filename = filename
        self.progress = 0.
        self.button.enabled = False
        self.widget.applyButton.enabled = False
        self.widget._loadingCount = getattr(self.widget, '_loadingCount', 0) + 1
        common.status(self.widget, 'Reading tractography: ' + filename, 0)
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.future = self.pool.submit(self.read, filename)
        self.timer.start()

    def read(self, filename):
        import whitematteranalysis as wma
        return wma.io.read_polydata(filename, progress_callback=self.setProgress)

    def setProgress(self, value):
        self.progress = value

    def poll(self):
        import WMAGuiCommon as common
        from WMAGPUControls import ACTIVE_WORKFLOW
        if self.future is None:
            return
        self.widget.progressBar.value = int(self.progress * 100)
        if not self.future.done():
            return
        self.timer.stop()
        try:
            polydata = self.future.result()
            if polydata is None or polydata.GetNumberOfLines() == 0:
                raise ValueError('The file contains no tractography lines.')
            node = common.add_tractography_node(os.path.splitext(os.path.basename(self.filename))[0], 'vtkMRMLModelNode')
            node.SetDisplayVisibility(False)
            node.SetAndObservePolyData(polydata)
            node.GetDisplayNode().SetScalarVisibility(False)
            node.SetDisplayVisibility(self.show.checked)
            self.selector.setCurrentNode(node)
            common.status(self.widget, 'Loaded %d fibers: %s' % (polydata.GetNumberOfLines(), self.filename), 100)
        except Exception as exc:
            common.status(self.widget, 'ERROR loading tractography: ' + str(exc))
        finally:
            self.pool.shutdown(wait=False)
            self.pool = self.future = None
            self.button.enabled = True
            self.widget._loadingCount -= 1
            self.widget.applyButton.enabled = self.widget._loadingCount == 0 and ACTIVE_WORKFLOW is None
            if hasattr(self.widget, 'onInputChanged'):
                self.widget.onInputChanged()

    def cleanup(self):
        self.timer.stop()
        if self.pool is not None:
            self.future.cancel()
            self.pool.shutdown(wait=False)
            self.pool = self.future = None
