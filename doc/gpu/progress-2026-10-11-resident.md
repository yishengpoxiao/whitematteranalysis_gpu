# Reusing GPU registration inputs (2026-10-11)

## Implementation

`gpu.registration.AffineObjective` retains float64 fixed and moving fiber tensors
for one optimization. `RegisterTractography.compute()` uploads them before the
optimizer starts, reuses them through `objective_function`, and clears references
in `finally`. Each new `compute()` recreates the inputs, supporting multiple
scales, different subjects, and inputs modified in place. Direct calls to
`objective_function()` still read the current NumPy inputs.

Distances, probabilities, entropy, VTK matrix construction, CPU optimizers, and
the random sampling sequence retain the previous algorithm. The CPU backend does
not create this device object. For 3000+1000 fibers with 10 points each, resident
input tensors occupy approximately 0.96 MB; the registration object releases its
references after successful or failed optimization.

## Validation

- Four GPU registration tests passed, covering original VTK coordinates/objectives,
  bitwise equality of full evaluation sequences and final transforms for three
  optimizers, input refresh after in-place NumPy mutation, release on exceptions,
  and bitwise agreement of CPU optimization with the original path.
- With CUDA hidden, the CPU test passed and three GPU tests explicitly skipped.
- Existing `test_gpu_consistency.py` and `test_cpu_fast.py` checks all passed.
- Compilation checks passed; all six modules reported SETUP OK in real Slicer
  5.2.2, with exit code 0.
- The previously identified stale module import in the stub harness was unchanged
  and was not counted as a pass.

Logs: `/tmp/wma-resident-{tests,no-cuda,consistency,cpu,slicer,real-bench}-2026-10-11.log`.
Tests deliberately use small optimizer evaluation budgets; budget warnings do not
indicate convergence.

## Performance

Single RTX 4090, float64, 10 points per fiber. Synthetic inputs use seed 42. After
warmup, paths were alternated for 30 runs each; medians include synchronization
before the objective returns.

| fixed × moving | Previous stateless GPU | Resident-input GPU | Time reduction |
|---|---:|---:|---:|
| 100 × 100 | 0.997 ms | 0.883 ms | 11.4% |
| 3000 × 1000 | 9.907 ms | 9.480 ms | 4.3% |
| 10000 × 100 | 4.472 ms | 3.527 ms | 21.1% |

Real fibers: `/data/yijie/TractData/val_dsi_studio/250932.vtk`. Seed 42 selects
4000 fibers, split into 3000 fixed and 1000 moving. Median evaluation time fell
from 9.872 to 9.414 ms (4.6%). One Cobyla run with 30 evaluations, including device
initialization, fell from 0.297 to 0.287 s (3.3%). Full objective sequences and
final transforms match bitwise. A single optimization timing is illustrative and
does not establish a speedup for the complete multiscale pipeline.

Reproduce:

```bash
CUDA_VISIBLE_DEVICES=0 /data/yijie/miniconda3/envs/wma/bin/python \
  tests/bench_registration.py \
  --input /data/yijie/TractData/val_dsi_studio/250932.vtk
```

At this checkpoint, nonrigid B-spline transform application still ran on CPU and
the Slicer `.sx` archive was not rebuilt. The local Slicer source symlink uses the
new implementation.
