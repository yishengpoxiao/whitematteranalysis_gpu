# Fast intermediate VTP compression — 2026-10-15

## Changes

Expose `WMA_VTP_COMPRESSION=zlib|lz4` for lossless Python VTP writing. The default
library/Slicer export setting remains zlib level 1. Both codecs are provided by
the required VTK runtime; no Python dependency changes are needed.

The complete ORG shell workflow selects LZ4 for initial clustering, outlier
removal, hemisphere assessment and hemisphere separation intermediates. Final
anatomical exports retain the caller's setting, defaulting to compact zlib.
Explicit `WMA_VTP_COMPRESSION=zlib` preserves compact intermediate writing.
The existing `-c` cleanup options remove intermediate outputs when requested.
Slicer-native hardening retains its storage writer and reads either codec.

The initial/outlier stage resume checks also verify every expected cluster
filename. A partially completed parallel write can create cluster 800 before
other clusters; its presence alone no longer marks either stage complete.
Failures in these commands propagate to the shell caller.

## Serialization sample

Compare seven alternating warm writes of the same representative transformed
cluster: 1177 fibers, 87,939 points, tensor/scalar/color arrays and string field
metadata. Use VTK 9.1 on one Xeon Gold 6330 core.

| Codec | Median write time | File bytes |
| --- | ---: | ---: |
| zlib level 1 | 0.33788 s | 9,495,184 |
| LZ4 level 1 | 0.10067 s | 12,811,427 |

LZ4 writing is approximately 3.36x faster, with about 35% additional intermediate
storage in this sample. This motivates limiting automatic selection to
intermediates. File-size and speed ratios depend on the dataset and storage.
Round-trip numeric values, coordinates, array types and string fields match
exactly. Legacy VTK files retain their existing binary encoding.

## Validation

Source tests verify both codecs with empty/nonempty geometry, active scalar and
tensor attributes, string metadata, and invalid codec reporting. Native Linux
Slicer 5.2.2/SlicerDMRI reads LZ4 bundles and verifies byte-identical transformed
outputs between serial and two-worker hardening. Shared inverse transforms and
nine distinct transforms both pass. A shell catalogue test removes cluster 42
while retaining cluster 800 and confirms that resume detection rejects the
incomplete set.

The universal installer bundles source revision `c9c622c`. An isolated Linux
Slicer 5.2.2 installation verifies the five pages, CPU/CUDA labeling,
registration and core outputs. Both codec round-trip tests and invalid-setting
checks pass against installed WMA code. All 28 bundled/installed WMA Python
files match committed source bytes. No optional PyTorch download is performed.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests WMA_VTP_COMPRESSION=lz4 \
  python -m unittest test_cli_file_stages -v

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests WMA_VTP_COMPRESSION=lz4 \
  xvfb-run -a python tests/verify_parallel_hardening.py \
  --slicer "$SLICER_EXECUTABLE_WITH_DMRI"
```
