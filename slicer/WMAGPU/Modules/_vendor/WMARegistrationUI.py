"""Shared registration controls for moving/fixed, groupwise, and pipeline tools."""

import qt
import ctk

from WMARegistration import PAIRWISE_PRESETS, GROUPWISE_PRESETS, registration_parameters


class RegistrationOptions:
    columns = ('modes', 'sigma', 'iterations', 'maxfun', 'mean_brain_size',
               'subject_brain_size', 'initial_step', 'final_step', 'grid_resolution')

    def __init__(self, form, groupwise=False):
        self.groupwise = groupwise
        self.presets = GROUPWISE_PRESETS if groupwise else PAIRWISE_PRESETS
        self.method = qt.QComboBox()
        self.method.addItems(list(self.presets) + ['custom'])
        self.method.toolTip = ('Nonrigid expects rigid/affine-aligned tractography in the upstream '
                               '200 mm grid centered at RAS zero. TEST presets check execution only.')
        form.addRow('Registration method:', self.method)
        self.advanced = ctk.ctkCollapsibleButton()
        self.advanced.text = 'Registration settings / custom schedule'
        self.advanced.collapsed = True
        layout = qt.QFormLayout(self.advanced)
        self.maximumLength = qt.QDoubleSpinBox()
        self.maximumLength.setRange(1., 10000.)
        self.maximumLength.value = 260.
        self.maximumLength.suffix = ' mm'
        layout.addRow('Maximum fiber length:', self.maximumLength)
        self.seed = qt.QSpinBox()
        self.seed.setRange(0, 2147483647)
        self.seed.value = 1000
        layout.addRow('Random seed:', self.seed)
        self.points = qt.QSpinBox()
        self.points.setRange(2, 1000)
        layout.addRow('Points per fiber (custom):', self.points)
        self.table = qt.QTableWidget()
        self.table.setColumnCount(len(self.columns))
        self.table.setHorizontalHeaderLabels([
            'Transform', 'Sigma (mm)', 'Iterations', 'Max evaluations',
            'Fixed / mean fibers', 'Moving fibers', 'Initial step', 'Final step', 'Grid'])
        self.table.minimumHeight = 190
        self.table.toolTip = ('Choose custom to edit. Use Affine/Rigid in one schedule, or Nonrigid '
                              'in a separate run after affine alignment. Grid is used only for Nonrigid.')
        layout.addRow(self.table)
        buttons = qt.QWidget()
        buttonLayout = qt.QHBoxLayout(buttons)
        self.add = qt.QPushButton('Add scale')
        self.remove = qt.QPushButton('Remove selected scale')
        buttonLayout.addWidget(self.add)
        buttonLayout.addWidget(self.remove)
        layout.addRow(buttons)
        self.verbose = qt.QCheckBox('Save intermediate registration outputs')
        layout.addRow(self.verbose)
        self.float32 = qt.QCheckBox('GPU float32 (may change near-tie results)')
        layout.addRow(self.float32)
        self.devices = qt.QLineEdit()
        self.devices.placeholderText = 'Default visible devices; e.g. 0 or 0,1'
        self.devices.toolTip = 'Registration uses one CUDA device; the first selected device is used.'
        layout.addRow('CUDA device IDs:', self.devices)
        form.addRow(self.advanced)
        self.method.connect('currentIndexChanged(int)', self.onMethodChanged)
        self.add.connect('clicked(bool)', self.addScale)
        self.remove.connect('clicked(bool)', self.removeScale)
        self.onMethodChanged()

    def onMethodChanged(self, *args):
        custom = self.method.currentText == 'custom'
        self.table.enabled = self.points.enabled = self.add.enabled = self.remove.enabled = custom
        if custom:
            self.advanced.collapsed = False
            return
        p = self.presets[self.method.currentText]
        self.points.value = p['points_per_fiber']
        self.table.setRowCount(len(p['sigma']))
        for row in range(self.table.rowCount):
            for column, field in enumerate(self.columns):
                value = p[field][row] if field in p else 4
                self.table.setItem(row, column, qt.QTableWidgetItem(str(value)))

    def addScale(self):
        row = self.table.rowCount
        self.table.insertRow(row)
        for column in range(len(self.columns)):
            previous = self.table.item(row - 1, column) if row else None
            value = previous.text() if previous else ('Affine' if column == 0 else '1')
            self.table.setItem(row, column, qt.QTableWidgetItem(value))

    def removeScale(self):
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)

    def arguments(self, minimum_length):
        if self.maximumLength.value <= minimum_length:
            raise ValueError('Maximum fiber length must exceed minimum fiber length.')
        method = self.method.currentText
        params = None
        if method == 'custom':
            params = dict(points_per_fiber=self.points.value)
            for column, field in enumerate(self.columns):
                values = [self.table.item(row, column).text().strip()
                          for row in range(self.table.rowCount)]
                try:
                    params[field] = values if field == 'modes' else [float(v) for v in values]
                except ValueError:
                    raise ValueError('Invalid numbers in %s.' % field)
            for field in ('iterations', 'maxfun', 'mean_brain_size', 'subject_brain_size', 'grid_resolution'):
                if any(int(v) != v for v in params[field]):
                    raise ValueError('%s must contain integers.' % field)
                params[field] = [int(v) for v in params[field]]
        registration_parameters(method, params, self.groupwise)
        text = self.devices.text.strip()
        try:
            devices = [int(v) for v in text.replace(',', ' ').split()] if text else None
        except ValueError:
            raise ValueError('CUDA device IDs must be nonnegative integers separated by commas.')
        if devices and any(v < 0 for v in devices):
            raise ValueError('CUDA device IDs must be nonnegative integers.')
        import whitematteranalysis as wma
        wma.gpu.set_float32(self.float32.checked)
        wma.gpu.set_devices(devices)
        return dict(mode=method, params=params, fiber_length_max_mm=self.maximumLength.value,
                    random_seed=self.seed.value, verbose=self.verbose.checked)
