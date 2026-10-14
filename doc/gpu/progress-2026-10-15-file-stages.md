# Lossless VTP writing and independent file workers — 2026-10-15

## Changes

- Use lossless zlib compression level 1 for Python-written VTP files. Keep
  binary encoding, geometry, numeric types, metadata and active attributes.
  Compressed file bytes and sizes can change; decoded data retain their values.
  Legacy binary VTK writing remains unchanged.
- Give hemisphere assessment, hemisphere separation and anatomical tract
  appending independent process workers through `-j/--jobs`. The ORG shell
  workflow forwards its `-n` worker count to these stages. Standalone tools
  default to one worker. Workers read/compute/write their own files; VTK objects
  are not serialized between processes. Input/member order within each output
  remains unchanged.
- Vectorize hemisphere point-label insertion, last-point label lookup, and
  anatomical `cluster_idx` provenance. Preserve float64 hemisphere labels,
  int32 provenance, empty-line behavior, legacy label spelling and the original
  per-file classification thresholds. Retain the existing VTK connectivity
  fallback.
- Make shared-transform hardening honor `-j`: each Slicer process receives one
  round-robin partition and loads/inverts its transform once for that batch.
  Each process owns an independent MRML scene. Multiple-transform mode keeps
  sorted input/transform correspondence. Invoke the matching adjacent helper
  using checked subprocess argument lists, supporting paths with spaces.
- Keep failed-worker logs in an owned temporary directory and propagate command
  failures. Check expected output filenames before deleting successful-worker
  logs. Existing unrelated files remain available.
- Write completion markers after all workers succeed and expected files exist.
  Replace stale markers when an explicitly invoked stage starts. The shell uses
  these markers for resuming parallel stages, propagates failed commands, returns
  nonzero for failed output checks, and checks the complete 74-tract catalogue.
  Checked VTK writes also raise on serialization failures.

## Measurements and scope

The preceding complete GPU pipeline takes 458.22 s, with about 71% in the final
four CPU/file stages; see the [complete-case record](benchmark-org-2026-10-15-optimized.md).
Those results predate the file-worker and compression changes described here.

A representative real transformed cluster contains 1177 fibers and 87,939
points, including tensor/scalar/color arrays and string field metadata. On one
Xeon Gold 6330 worker with VTK 9.1, alternating seven warm writes gives:

| VTP compression | Median write time | File bytes |
| --- | ---: | ---: |
| Previous zlib level 5 | 0.42271 s | 9,444,308 |
| Current zlib level 1 | 0.32438 s | 9,495,184 |

Writing is approximately 1.30x faster, reducing time by 23%, with about 0.54%
additional storage for this sample. Round-trip coordinates, numeric arrays,
array types and string metadata match exactly. This is a serialization sample,
not a full-pipeline speedup or a file-size bound for other datasets.

The complete GPU case is also rerun with frozen source `1e6172a`, four pinned CPU
cores and the same full clinical input. Total wall time is 205.09 s (3:25), down
from the preceding 458.22 s (7:38); maximum host RSS is 10.48 GiB. Registration,
labeling/saving and outlier removal take 31.04/50.64/37.60 s. Hemisphere assessment,
Slicer inverse hardening, hemisphere separation and anatomical appending take
19.52/29.78/17.98/15.54 s. These are single shared-workstation runs, including
imports and I/O, with changing cache/load conditions.

All 800 initial, 800 outlier-removed, 800 transformed and 74 anatomical files
pass exact decoded coordinate, numeric-array, type, name and string-field
comparisons with the original complete GPU case. Total fiber counts remain
245,946/130,737/130,737/93,766. Saved transform parameters match exactly. Compressed
bytes change with the codec; all 800 Slicer-native transformed files remain
byte-identical. Completion markers and every expected filename are checked.
This run uses zlib intermediates and predates the subsequent LZ4 optimization.

## Validation

Targeted CLI comparisons run the old scripts and current one/two-worker scripts
on deterministic tractography with scalar/color/tensor/field arrays, empty
clusters, repeated anatomical memberships and paths with spaces. All resulting
VTP files match byte-for-byte when written with the same codec. All 74 anatomical
outputs are checked. Tests also cover last-point semantics for non-contiguous
connectivity, empty lines, multiple point components, lossless round trips,
failed file workers, checked Slicer process failures, and log preservation.

Native Linux Slicer 5.2.2/SlicerDMRI tests compare serial versus two-worker inverse
hardening of nine bundles, including an empty bundle and tensor metadata. Outputs
match byte-for-byte. A separate nine-transform run checks sorted correspondence
and distinct transforms. Successful worker logs are removed.

The rebuilt universal installer bundles source revision `1e6172a`. An isolated
Linux Slicer 5.2.2 installation verifies all five pages, CPU/CUDA labeling,
registration, core output/persistence checks, selected-device execution, loader
behavior, locking and allocator cleanup. All 28 bundled/installed WMA Python
files match committed source bytes. Optional PyTorch downloads are not performed
by these tests.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests python -m unittest test_cli_file_stages -v

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests xvfb-run -a python \
  tests/verify_parallel_hardening.py --slicer "$SLICER_EXECUTABLE_WITH_DMRI"
```
