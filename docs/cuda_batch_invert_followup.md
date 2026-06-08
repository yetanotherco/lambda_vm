# Follow-up: GPU batch inverse for R3 OOD and R4 DEEP denominators

PR #648 (R4 DEEP composition + FRI commit on GPU) originally included a Blelloch
chunk-scan parallel batch-inverse subsystem in `crypto/math-cuda` (`inverse.cu`
+ `inverse.rs` + the `batch_inverse.rs` parity test). That code was removed
before merging because it was tested but never wired into the prover — R3 OOD
and R4 DEEP both still invert their denominator arrays on the CPU and H2D the
inverted result to the device, paying a non-trivial PCIe cost on large traces.

This document captures what the next PR should do to land that path properly,
so we don't lose the work. Refer to the commit history of PR #648 for the
original kernel and orchestrator code (search `feat/cuda-pr4` for files
`crypto/math-cuda/kernels/inverse.cu`, `crypto/math-cuda/src/inverse.rs`,
`crypto/math-cuda/tests/batch_inverse.rs`).

## Current CPU cost we want to eliminate

`compute_deep_composition_poly_evaluations` in `crypto/stark/src/prover.rs`
builds and inverts the denominator array on the CPU:

```rust
let num_denoms = lde_size * (1 + num_eval_points);
let mut denoms: Vec<FieldElement<FieldExtension>> = Vec::with_capacity(num_denoms);
// ... fill H-term and trace-term denoms ...
FieldElement::inplace_batch_inverse(&mut denoms).expect("...");
```

Then `try_deep_composition_gpu` H2Ds the inverted slice
(`lde_size * (1 + num_eval_points) * 24` bytes — tens of MB for the larger
benches like `fib_iterative_4M`) inside `deep_composition_ext3_impl`.

The R3 OOD barycentric path has the same shape: CPU computes inv_denoms, GPU
reads them as a host slice that gets H2D'd.

## Target architecture

1. **Compute denoms on device.** Restore `compute_denoms_ext3` (an ext3 kernel
   that builds `denoms[k * n + i] = x[i * stride] - z_scalars[k]` for all
   `(k, i)`). Inputs: a base-field LDE coset slice and the ext3 `z_scalars`
   array. Output: an `lde_size * (1 + num_eval_points)`-element ext3 device
   buffer.
2. **Invert in-place on device.** Restore the six-kernel Blelloch pipeline
   (`chunk_prefix_scan_ext3`, `exclusive_scan_of_totals_ext3`,
   `apply_scan_offsets_ext3`, the matching suffix variants, and
   `batch_inverse_combine_ext3`), driven by `batch_inverse_pipeline` in
   `inverse.rs`.
3. **Return a `CudaSlice` handle, not a `Vec`.** The previous incarnation
   exposed `pub fn compute_and_invert_denoms_ext3(...) -> Result<Vec<u64>>`
   which D2H'd the result. The next PR's version must return a device handle
   so the inverted slice never crosses PCIe.
4. **Thread the device handle through to DEEP.** `deep_composition_ext3_impl`
   currently takes `inv_h: &[u64]` and `inv_t: &[u64]` (host slices that get
   H2D'd internally). Add a `_with_dev_inv_denoms` variant — or refactor the
   existing one to accept `Option<&CudaSlice<u64>>` for each — and pull from
   the device buffer when present.
5. **Same on R3 OOD.** `try_barycentric_base_on_handle` /
   `try_barycentric_ext3_on_handle` take `inv_denoms: &[FieldElement<E>]`. Add
   `_dev` variants that accept a `CudaSlice` handle.

## Performance notes for the next PR

- **Multi-block scan, not grid=1.** The PR-4 version launched
  `grid=1, block=K=256`, with one thread doing `O(C) = O(n/K)` serial work.
  For `n = 2^22`, each thread did ~16k serial ext3 multiplies while ~170 SMs
  on a 5090 sat idle. Use a proper multi-block scan: phase-1 per-block scans
  in shared memory, phase-2 inter-block carry, phase-3 application. CUB-style
  layout works; OneSweep is overkill.
- **`alloc` not `alloc_zeros`.** Scratch buffers (`prefix_dev`, `suffix_dev`,
  `chunk_totals`, `chunk_offsets`, `out_dev`) are fully overwritten by the
  scan kernels — no need to pay for `cuMemsetD8` on tens of MB. Use
  `unsafe { stream.alloc::<u64>(n) }?` and a SAFETY note that the kernel
  writes every slot before any read.
- **`gl_sub` footgun.** The host helper `gl_sub(a, b)` (used by
  `invert_ext3_host` for the one-element total inversion) underflows in u128
  when `b >= GOLDILOCKS_P`. Inputs are currently safe by construction, but
  the next PR's wiring may feed it raw GPU outputs (non-canonical). Add an
  upfront `b %= GOLDILOCKS_P` or a `debug_assert!(b < GOLDILOCKS_P as u64)`.
- **n=1 path correctness.** PR-4 audit found a latent bug where `n=1`
  silently returned `[0; 3]` instead of inverting the single element. The
  fix landed as `if n == 1 { return Ok(invert_ext3_host(...).to_vec()); }`
  before the GPU pipeline. Keep this when restoring `batch_inverse_ext3`.

## Verification once wired

The `cuda_path_integration::gpu_path_fires_end_to_end` test already asserts
the full GPU pipeline fires and the proof verifies. Once batch-inverse is
wired:

1. Add `GPU_BATCH_INVERT_CALLS` counter alongside the other six.
2. Assert it fires in `gpu_path_fires_end_to_end`.
3. Re-run the prove-fib_4M benchmark; expected savings are the
   `(1 + num_eval_points) * lde_size * 24` bytes of H2D per table, plus
   the CPU `inplace_batch_inverse` time on a `~lde_size * 4` ext3 array.

## Out of scope for the next PR

- Replacing the *evaluator-side* inversions (e.g., in `lde.rs` or `frame.rs`)
  unless a clear bottleneck profile shows otherwise. The R3/R4 inverse path
  is the only confirmed perf win.
- Generalising to non-ext3 inputs. Ext3 covers both confirmed call sites.
