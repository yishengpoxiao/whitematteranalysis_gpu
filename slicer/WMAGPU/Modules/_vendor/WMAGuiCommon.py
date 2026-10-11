# -*- coding: utf-8 -*-

"""WMAGuiCommon

Shared helpers for the WMA GPU Slicer modules: backend selection,
status area, directory pickers, and reliable whitematteranalysis
discovery. Lives in _vendor/ so Slicer does not scan it as a module.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

CORE = None
CORE_ERROR = None
WMA_OK = False
WMA_DETAIL = ""

BACKENDS = ['Auto (GPU if available)', 'GPU (torch CUDA)',
            'CPU accelerated (no torch needed)', 'Original (joblib)']
BACKEND_KEYS = ['auto', 'gpu', 'cpu', 'original']

CATEGORY = "Diffusion.WMA GPU"


def ensure_environment():
    """Put the vendor dir on sys.path and import the core. Safe to call
    repeatedly. Returns (CORE_or_None, wma_ok, wma_detail)."""

    global CORE, CORE_ERROR, WMA_OK, WMA_DETAIL

    # bundled pure-python deps first
    if HERE not in sys.path:
        sys.path.insert(0, HERE)

    try:
        import whitematteranalysis as wma  # noqa: F401
    except Exception as exc:
        WMA_OK = False
        WMA_DETAIL = "Bundled WMA failed to load: %s. Reinstall WMAGPU and restart Slicer." % exc
        return None, WMA_OK, WMA_DETAIL

    global CORE
    try:
        import WMAtlasLabelGPUCore
        CORE = WMAtlasLabelGPUCore
        CORE_ERROR = None
    except Exception:
        import traceback
        CORE_ERROR = traceback.format_exc()
        CORE = None
        return None, WMA_OK, ("module core failed to import:\n" + CORE_ERROR)

    gpu_now = hasattr(wma, 'gpu')
    WMA_OK = True
    WMA_DETAIL = wma.__file__
    if not gpu_now:
        WMA_DETAIL = ("warning: the imported whitematteranalysis is a "
                      "pre-GPU copy without wma.gpu (from %s). "
                      "Reinstall the WMAGPU extension and restart to activate the "
                      "GPU version." % wma.__file__)
    return CORE, WMA_OK, WMA_DETAIL


def backend_key(index):
    return BACKEND_KEYS[index] if index is not None and 0 <= index < len(BACKEND_KEYS) \
        else 'auto'


def add_backend_row(form, current_widget):
    """Append the standard backend combo below a Qt form layout."""

    import qt
    combo = qt.QComboBox()
    for name in BACKENDS:
        combo.addItem(name)
    combo.toolTip = (
        "Default float64 preserves validated labels and output files. GPU needs torch with CUDA; "
        "the CPU accelerated backend is vectorized numpy (no torch).")
    form.addRow("Backend:", combo)
    current_widget.backendComboBox = combo
    from WMATorchInstaller import add_install_row
    add_install_row(form, current_widget)
    return combo


def add_status_area(widget, layout):
    """Append a status text + progress bar to a layout; returns them."""

    import qt
    widget.statusLabel = qt.QPlainTextEdit()
    widget.statusLabel.readOnly = True
    widget.statusLabel.maximumHeight = 110
    layout.addWidget(widget.statusLabel)
    widget.progressBar = qt.QProgressBar()
    layout.addWidget(widget.progressBar)


def status(widget, text, pct=None):
    if hasattr(widget, 'statusLabel'):
        widget.statusLabel.appendPlainText(text)
    if pct is not None and hasattr(widget, 'progressBar'):
        widget.progressBar.value = pct
    try:
        import slicer
        slicer.app.processEvents()
    except Exception:
        pass


def pick_directory(parent_window, title, current=""):
    import qt
    import slicer
    return qt.QFileDialog.getExistingDirectory(
        slicer.util.mainWindow(), title, current or "") or ""


def set_dir_button(button, path):
    if path:
        button.text = path


def tractography_node_types():
    import slicer
    if slicer.mrmlScene.GetTagByClassName('vtkMRMLFiberBundleNode'):
        return ['vtkMRMLFiberBundleNode', 'vtkMRMLModelNode']
    return ['vtkMRMLModelNode']


def add_tractography_node(name):
    import slicer
    node = slicer.mrmlScene.AddNewNodeByClass(tractography_node_types()[0], name)
    node.CreateDefaultDisplayNodes()
    return node


def show_clusters_in_scene(pd_c_list, base_name="cluster",
                           clear_previous=True):
    """Add one colored tractography/model node per cluster to the
    Slicer scene (FiberBundle when SlicerDMRI provides it)."""

    import colorsys

    import slicer

    if clear_previous:
        for cls in ('vtkMRMLModelNode', 'vtkMRMLFiberBundleNode'):
            try:
                for node in slicer.util.getNodesByClass(cls):
                    if node.GetName().startswith(base_name + "_"):
                        slicer.mrmlScene.RemoveNode(node)
            except Exception:
                pass

    nodes = []
    for c, pd_c in enumerate(pd_c_list):
        name = "%s_%05d" % (base_name, c + 1)
        node = add_tractography_node(name)
        node.SetAndObservePolyData(pd_c)
        node.CreateDefaultDisplayNodes()
        display = node.GetDisplayNode()
        if display is not None:
            h = (c * 0.61803398875) % 1.0
            r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
            display.SetColor(r, g, b)
        nodes.append(node)
    return nodes


def save_clusters_parallel(pd_c_list, output_directory):
    """Write cluster_<id>.vtp files (parallel when supported). Returns
    the list of file names."""

    import whitematteranalysis as wma
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
    fnames = [os.path.join(output_directory, 'cluster_%05d.vtp' % (c + 1))
              for c in range(len(pd_c_list))]
    if hasattr(wma.io, 'write_polydatas_parallel'):
        wma.io.write_polydatas_parallel(pd_c_list, fnames)
    else:
        for pd_c, fname in zip(pd_c_list, fnames):
            wma.io.write_polydata(pd_c, fname)
    return fnames
