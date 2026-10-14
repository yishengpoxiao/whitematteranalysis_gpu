"""Measure the first opening of all workflow pages inside a fresh Slicer process."""

import json
import os
from pathlib import Path
import sys
import time

import slicer

try:
    before = set(sys.modules)
    timings = {}
    for name in ('WMAPipeline', 'WMARegisterToAtlas', 'WMAGroupRegister',
                 'WMAAtlasBuilder', 'WMAMergeMRML'):
        start = time.perf_counter()
        widget = slicer.util.getModule(name).widgetRepresentation().self()
        assert hasattr(widget, 'statusLabel'), name
        assert 'whitematteranalysis: OK' in widget.statusLabel.toPlainText(), name
        if name == 'WMAMergeMRML':
            assert widget._sceneObservers, 'Merge page setup did not complete'
        timings[name] = time.perf_counter() - start
    heavy = sorted({'torch', 'scipy', 'matplotlib'} & (set(sys.modules) - before))
    if not os.environ.get('WMA_STARTUP_BASELINE'):
        assert not heavy, 'Opening pages imported: ' + ', '.join(heavy)
    import whitematteranalysis as wma
    report = dict(page_seconds=timings, total_seconds=sum(timings.values()),
                  imported_heavy_packages=heavy, wma_path=wma.__file__)
    if os.environ.get('WMA_STARTUP_REPORT'):
        Path(os.environ['WMA_STARTUP_REPORT']).write_text(json.dumps(report, indent=2) + '\n')
    print('STARTUP_VERIFIED:', json.dumps(report), flush=True)
except Exception:
    import traceback
    traceback.print_exc()
    sys.exit(1)
sys.exit(0)
