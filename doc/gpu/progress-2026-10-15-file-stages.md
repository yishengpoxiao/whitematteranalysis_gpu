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

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests python -m unittest test_cli_file_stages -v

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests xvfb-run -a python \
  tests/verify_parallel_hardening.py --slicer "$SLICER_EXECUTABLE_WITH_DMRI"
```
