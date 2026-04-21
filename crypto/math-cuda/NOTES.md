# math-cuda — performance notes

Running log of attempts, analysis, and what's left. Intended to survive
context loss between sessions. Update as you go.

## Current state (as of this commit)

`math-cuda` has a batched Goldilocks coset-LDE:

- Kernels: `bit_reverse_permute_batched`, `ntt_dit_level_batched`,
  `ntt_dit_8_levels_batched` (shmem fusion of the first 8 DIT levels),
  `pointwise_mul_batched`, `scalar_mul_batched`.
- Backend (`device.rs`): CUDA context, pool of 32 streams, single shared
  pinned host staging buffer (non-WC, allocated lazily and grown, reused
  across calls), twiddle cache per `log_n`. Event tracking is
  disabled globally — it adds ~2 CUDA API calls per slice allocation
  and serialised concurrent callers on the driver's context lock.
- Public entry points:
  - `lde::coset_lde_batch_base(columns: &[&[u64]], blowup, weights) -> Vec<Vec<u64>>`
  - `lde::coset_lde_batch_base_into(columns, blowup, weights, outputs: &mut [&mut [u64]])`
  - `ntt::forward/inverse` for single-column base-field NTT.
- Parity-tested against CPU (`tests/ntt.rs`, `tests/lde.rs`, `tests/lde_batch.rs`)
  up to `log_n = 20`.
- Hooked into stark via `crypto/stark/src/gpu_lde.rs` and
  `expand_columns_to_lde` at `crypto/stark/src/prover.rs:479`. Feature
  flag: `cuda` on `stark` and `lambda-vm-prover`.

## Microbench results (RTX 5090, 46-core host, blowup=4, warm)

| Size | CPU rayon | GPU batched | Ratio |
|---|---|---|---|
| 64 cols, log_n=16 (LDE 2^18) | ~75–100 ms | ~15–20 ms | **5–12×** (high variance) |
| 20 cols, log_n=20 (LDE 2^22, prover-scale) | ~470 ms | ~220 ms | **~2.0–2.3×** |

End-to-end 1M-fib fibonacci proof: CPU ~17 s, CUDA ~16.5–17 s — **tied**.
The microbench win doesn't translate to end-to-end because LDE is only
~20% of proof wall time (Round 1 LDE) and the per-call timings inside
the prover incur initial warmup and mutex serialisation on the shared
pinned staging.

## Where the time goes at prover scale (single LDE call, log_n=20, 20 cols)

Phase timings (enable with `MATH_CUDA_PHASE_TIMING=1`):

| Phase | Time |
|---|---|
| host pack into pinned (rayon) | ~8 ms |
| device alloc_zeros (async) | ~0.5 ms |
| H2D (pinned → device) | ~9 ms |
| iNTT body (22 levels total) | ~3 ms |
| pointwise + bit-reverse LDE | ~2 ms |
| forward NTT body (22 levels) | ~13 ms |
| D2H (device → pinned) | ~28 ms |
| copy out (pinned → caller Vecs, rayon) | ~65 ms |
| **total** | **~130 ms** |

**Compute is only ~15% of GPU wall time.** The other 85% is PCIe and
pageable host memcpy / page faults. No amount of kernel optimisation
alone closes this gap.

## Things tried and their outcomes

### ✅ Kept

1. **Fused 8-level DIT kernel** (`ntt_dit_8_levels_batched`): first 8
   butterfly levels in shared memory. 7× reduction in launches for
   levels 0–7; ~8× less DRAM traffic there.
2. **Column batching via `gridDim.y = M`**: single kernel launch handles
   all columns at a level instead of M separate launches.
3. **Reusable shared pinned staging buffer** (`PinnedStaging` in
   `device.rs`): `cuMemHostAlloc` with flags=0 (portable, non-WC). One
   allocation grows as needed; locked on call-entry for exclusive use.
4. **Rayon-parallel host pack**: 27 ms → 8 ms at prover scale.
5. **Median-of-10 microbench** for stable measurement.

### ❌ Tried and reverted

1. **4-col register tile in fused 8-level kernel (A1).** Clean port of
   Zisk's `br_ntt_8_steps` inner loop — 256 threads × 4 columns each in
   a 1024-entry shmem tile. Neutral at prover scale (1.81× vs 1.88×
   without); regressed small-n microbench (shmem pressure lowered
   occupancy). The fused kernel handles only the first 8 of 22 levels at
   prover scale, so even a 2× win there is ~2 ms of the ~20 ms compute
   budget.
2. **Per-caller-Vec pinning via `cuMemHostRegister`.** Fast when
   isolated (~1.7× on 64-col microbench) but the driver serialises pin
   calls globally; under rayon-parallel table dispatch in the prover
   this turned GPU slower than CPU.
3. **Per-stream pinned staging (32 buffers).** Each slot paid the
   ~1 second `cuMemHostAlloc` cost on first large-table use. Replaced
   with a single shared staging buffer.
4. **Pre-fault output Vec pages overlapped with D2H.** Saved ~40 ms of
   copy-out, but the prefault itself cost ~60 ms on a parallel rayon
   sweep (mm_struct rwsem serialisation). Net neutral.
5. **A lot of single-trial microbenches.** CPU rayon time is 20–50%
   noisy; needed median-of-10 to stop chasing phantoms.

## Why we're stuck at ~2× and the 10× ceiling

Amdahl: at 1M-fib scale only ~20% of proof wall time is LDE, and inside
the LDE call itself only ~15% is GPU compute. The remaining 85% of a
per-call GPU budget is:

| Cost | Size @ prover scale | Why it's there |
|---|---|---|
| PCIe D2H (pinned) | 28 ms | LDE result has to come back for Merkle |
| Pinned → pageable Vec copy | 65 ms | Caller expects `Vec<FieldElement<F>>` for Round 2-4 cache; fresh-alloc pages fault on first write, fault path serialises on mm_struct rwsem |
| PCIe H2D (pinned) | 9 ms | Input columns from CPU |
| host pack | 8 ms | Pageable trace Vec → pinned staging |

Other projects don't pay this because they **keep data GPU-resident
across Rounds 1–4**. Zisk (`pil2-stark/src/goldilocks/src/ntt_goldilocks.cu`)
chains trace → NTT → Merkle → constraint eval → FRI on device;
Airbender (`zksync-airbender/gpu_prover/`) uses a 5-stage on-device
pipeline. In both, host transfer is roughly "witness in, proof out",
nothing in between.

## The 10× path

Ranked by expected wall-time impact on 1M-fib (CPU baseline ~17 s):

1. **C1: GPU Keccak256 + LDE stays on GPU through Merkle commit.**
   Addresses the 28 ms D2H + 65 ms copy-out. ~4–6 s saved end-to-end.
   Needs: (a) Goldilocks-input Keccak256 kernel (no reference in the
   repos we explored — Airbender uses Blake2s, Zisk uses Poseidon2),
   (b) a batched "commit over GPU-resident columns" kernel that reads
   LDE directly from device memory and produces the 32-byte root, (c)
   refactoring `commit_columns_bit_reversed` in stark to accept a GPU
   handle instead of `&[Vec<FieldElement<E>>]`. Estimated 1-2 days of
   focused work.

2. **B1: keep LDE buffer on GPU across rounds.** Round 2–4 currently
   re-read the cached LDE from host memory (populated by Round 1).
   Holding it on device instead avoids repeat H2D. Needs: refactoring
   `Round1<F, E>` to hold either a GPU handle OR the host Vecs, plus a
   GPU constraint-eval and/or FFT path for Round 2's `extend_half_to_lde`
   (`prover.rs:834`). Estimated 2-3 days.

3. **D: ext3 NTT via component decomposition.** A single ext3 column is
   `[a, b, c]` per element; butterflies use a base-field twiddle
   multiplication, and `base × ext3` is componentwise. So NTT over M
   ext3 columns = NTT over 3M base columns with the same twiddles and
   weights. No new kernels needed — just a de-interleave at pack time
   and re-interleave at unpack. This unlocks:
   - Aux trace LDE (`expand_columns_to_lde` on ext3, 2.9 s aggregate)
   - `extend_half_to_lde` (Round 2 decompose, 6.1 s aggregate, biggest
     single FFT chunk in the proof). Needs different weights —
     `g^(-k) / N` rather than `g^k / N`. Easy.

4. **A2: warp-shuffle butterflies for stages 0–5.** Saves maybe 3 ms of
   compute. Low priority after (1)–(3).

5. **A3: vectorised `uint2` `__ldg` loads in per-level kernels.** Saves
   maybe 5 ms. Low priority.

## Key files

- `crypto/math-cuda/kernels/{goldilocks.cuh,ntt.cu,arith.cu}`
- `crypto/math-cuda/src/{device.rs,ntt.rs,lde.rs,lib.rs}`
- `crypto/math-cuda/tests/{goldilocks.rs,ntt.rs,lde.rs,lde_batch.rs,bench_quick.rs}`
- `crypto/stark/src/gpu_lde.rs` — the stark-level dispatch wrapper
- `crypto/stark/src/prover.rs:479` — `expand_columns_to_lde` call site
- `crypto/stark/src/prover.rs:834` — `extend_half_to_lde`, **not yet
  GPU-enabled** (Round 2 quotient extension FFTs)
- `crypto/stark/src/prover.rs:368` — `commit_columns_bit_reversed`, the
  Merkle commit that C1 would replace

## References

- `/workspace/references/pil2-proofman/pil2-stark/src/goldilocks/src/ntt_goldilocks.cu`
  — Zisk's NTT, especially `br_ntt_8_steps:674` (4-col register tile pattern)
- `/workspace/references/zksync-airbender/gpu_prover/native/ntt/`
  — Airbender's NTT with warp-shuffle butterflies and `uint2` loads
- `/workspace/references/zksync-airbender/gpu_prover/native/blake2s.cu`
  — Template for GPU tree hashing (but Blake2s, not Keccak)
- Research summary in earlier session — see conversation history or the
  `vast-squishing-crayon` plan file at `/root/.claude/plans/` if it still
  exists.

## Useful commands

```sh
# Build with GPU feature
cargo check -p stark --features cuda

# Parity tests
cargo test -p math-cuda

# Microbenches (median-of-10)
cargo test -p math-cuda --test bench_quick --release bench_lde_batched -- --ignored --nocapture

# Per-phase timing within a batched call
MATH_CUDA_PHASE_TIMING=1 cargo test -p math-cuda --test bench_quick --release bench_lde_batched_prover_scale -- --ignored --nocapture

# End-to-end prove bench
cargo test -p lambda-vm-prover --release --test bench_gpu bench_prove_fib_1m -- --ignored --nocapture
cargo test -p lambda-vm-prover --release --features cuda --test bench_gpu bench_prove_fib_1m -- --ignored --nocapture
cargo test -p lambda-vm-prover --release --features instruments,cuda --test bench_gpu bench_prove_fib_1m -- --ignored --nocapture  # adds phase breakdown

# Threshold override
LAMBDA_VM_GPU_LDE_THRESHOLD=$((1<<18)) cargo test ...
```
