# math-cuda — performance notes

Running log of attempts, analysis, and what's left. Intended to survive
context loss between sessions. Update as you go.

## Current state (2026-04-21, 5 commits on branch `cuda/batched-ntt`)

### End-to-end speedup (fused tree + R2-commit tree + GPU R4 deep + LDE-resident handles)

| Program | CPU rayon (46 cores) | CUDA (median over 5+ runs) | Delta |
|---|---|---|---|
| fib_iterative_1M | **18.269 s** | **12.66 s** | **1.44× (30.7% faster)** |
| fib_iterative_4M |               | **29.75 s** |   |

Correctness: all 30 math-cuda parity tests + 121 stark cuda tests pass.

### What's GPU-accelerated now

| Hook | What it does | Kernel(s) |
|---|---|---|
| Main trace LDE + Merkle commit | Base-field LDE → leaf-hash → **full Merkle tree** on device, retaining the LDE device buffer as a `GpuLdeBase` handle on `LDETraceTable` | `ntt_*_batched` + `keccak256_leaves_base_batched` + `keccak_merkle_level` |
| Aux trace LDE + Merkle commit | Ext3 LDE via 3× base decomposition → ext3 leaf-hash → **full Merkle tree**, retaining the de-interleaved LDE buffer as a `GpuLdeExt3` handle | `ntt_*_batched` + `keccak256_leaves_ext3_batched` + `keccak_merkle_level` |
| R2 composition-parts LDE | `number_of_parts > 2` branch: batched ext3 evaluate-on-coset | `ntt_*_batched` (no iFFT variant) |
| R2 commit_composition_poly | Row-pair ext3 Keccak leaves + pair-hash inner tree | `keccak_comp_poly_leaves_ext3` + `keccak_merkle_level` |
| R4 DEEP-poly LDE | Standard ext3 LDE with uniform 1/N weights | `ntt_*_batched` via ext3 decomposition |
| **R4 deep_composition_poly_evals** | Per trace-size row, sum ~200 ext3 FMAs over all LDE cols + scalars. Reads main+aux LDE **from device handles** (no re-H2D) | `deep_composition_ext3_row` |
| R2 `extend_half_to_lde` | Dormant — only hit by tiny tables below threshold | Infrastructure in place |

### Where time still goes (aggregate across rayon threads, 1M-fib, warm)

| Phase | Aggregate | On GPU? |
|---|---|---|
| R3 OOD evaluation | 5.94 s | ❌ barycentric point-evals in ext3 |
| R2 evaluate (constraint eval) | 5.00 s | ❌ per-AIR constraint logic |
| R2 decompose_and_extend_d2 (FFT) | 3.06 s | ✅ partial (parts LDE) |
| R4 deep_composition_poly_evals | 2.32 s | ❌ ext3 barycentric |
| R2 commit_composition_poly (Merkle) | 1.92 s | ❌ different leaf-pair pattern, not wired |
| R4 fri::commit_phase | 1.58 s | ❌ in-place folding |
| R4 queries & openings | 1.54 s | ❌ per-query Merkle openings |
| R4 interpolate+evaluate_fft | 0.53 s | ✅ (via DEEP-poly LDE) |

### What would be needed to reach ~2× (~50%)

1. **Ext3 arithmetic on GPU**. Full `ext3 × ext3` multiplication (currently
   we only use `base × ext3` in the NTT butterflies). Required for OOD and
   deep-composition barycentric kernels. ~100 lines of CUDA plus parity tests.
   **✅ LANDED** — `kernels/ext3.cuh` has add/sub/neg/mul_base/mul with the
   `dot3` helper; parity tested in `tests/ext3.rs`.
2. **Barycentric at a point** kernel. O(N) reduction per column, M columns
   in parallel. Addresses OOD (5.94 s) + deep-composition (2.32 s). ~8 s of
   aggregate work ≈ ~0.5–1 s wall savings with rayon.
   **✅ LANDED (unwired)** — `kernels/barycentric.cu` +
   `src/barycentric.rs` + parity test `tests/barycentric.rs` all work. The
   R3-OOD wiring in `get_trace_evaluations_from_lde` was **reverted** after
   benchmarking: in the current prover the CPU is idle during R3 (the GPU
   is busy on LDE/Merkle streams), so routing R3 OOD to the GPU only adds
   queue contention without freeing wall time — fib_iterative_1M went
   13.09 s → 14.20 s, and fib_iterative_4M went 33.67 s → 36.03 s, both
   regressions. The kernels stay here as a building block for future
   workloads where the GPU has idle windows during R3 (single-table or
   very-large-trace proofs).
3. **R2 constraint evaluation on GPU**. Per-AIR pointwise kernels over the
   LDE domain. Biggest engineering lift (each AIR has its own constraint
   logic). Could save 5 s aggregate ≈ 0.3–0.5 s wall.
4. **GPU-resident LDE across rounds**. Currently Rounds 2–4 re-read LDE
   columns from the host Vecs that Round 1 produced. Keeping the LDE on
   device would remove the next H2D cycle.

None of these are trivial; individually each is hours to a day. Collectively
they'd probably push the 1M-fib proof under 10 s (matching Zisk/Airbender-
class wins).

### Lesson from the R3-OOD attempt

Aggregate CPU time (as reported by the `instruments` feature) overstates
the real wall-time cost of a phase whenever rayon already parallelises
it. R3 OOD's 5.94 s "aggregate" number was misleading: on a 46-core box
with ~7 tables running in parallel, rayon reduces that to ≈0.15 s wall,
which is *less than* one H2D round-trip of the 500 MB of column data the
GPU kernel would need. The GPU-resident LDE refactor (item 4 above) is
the unlock here — without it, the CPU barycentric is already close to a
lower bound for this workload.

### What's on the GPU but unwired (kernels + parity tests only)

After benchmarking, these optimisations have the kernel built and parity-
tested but are NOT wired into the prover because the measured wall-time
delta was neutral or negative:

- **Barycentric OOD** (`kernels/barycentric.cu`, `tests/barycentric.rs`):
  R3 trace OOD + composition-parts OOD. CPU path is already idle-side
  while GPU is busy on LDE streams, so routing R3 to GPU regresses.
- **FRI layer Merkle tree** (`keccak_fri_leaves_ext3` +
  `build_fri_layer_tree_from_evals_ext3`, `tests/fri_layer_tree.rs`):
  per-layer H2D of the freshly-folded eval slab (pageable Vec) eats the
  tree-build savings. Needs fused fold+leaves+tree staying on device
  across layers, which requires item 4 below.
- **Standalone GPU Merkle inner-tree builder**
  (`build_merkle_tree_on_device`, `tests/merkle_tree.rs`): superseded by
  the fused LDE+leaves+tree pipeline which skips the leaf D2H entirely.
  The standalone function remains as a building block.

### Path to a meaningful next win

The remaining aggregate targets are dominated by CPU work whose wall-time
cost is small (~0.2–0.5 s each) because rayon already parallelises them.
Moving any one of them to GPU pays a per-call H2D that wipes the gain.
The unlock is **LDE GPU-resident across rounds** — keep the main/aux
LDE buffers alive on device after R1 commits, and let R2 constraint
evaluation, R3 OOD, R4 deep-composition, and R4 FRI-fold read them
without re-H2D.

That refactor lets three currently-unwired pieces flip from net-negative
to net-positive:
  - R3 barycentric OOD (kernels exist)
  - FRI commit phase (kernels exist)
  - R4 deep composition (kernel not yet written; small, pointwise FMA)

…and enables the big one: **GPU constraint evaluation** via a
device-side expression-tree interpreter over a compile-time-serialised
AST (keeps the CPU constraints as the single source of truth).

Scope for the LDE-GPU-resident refactor: add an `Option<Arc<CudaSlice>>`
sidecar to `LDETraceTable`, have the R1 fused path populate it, and
gate each consumer's GPU path on its presence. ~300-500 LoC with
careful CPU-fallback preservation.

### What's on the GPU now

Four independent hook points in the stark prover, all behind the `cuda`
feature flag. CPU path unchanged when the feature is off.

| Hook | Call site | Fires per 1M-fib proof | Notes |
|---|---|---|---|
| Main trace LDE (base-field) | `expand_columns_to_lde`, `prover.rs:479` | ~40 cols × few tables | `coset_lde_batch_base_into` |
| Aux trace LDE (ext3, via 3× base decomposition) | `expand_columns_to_lde`, same call site | ~20 cols × few tables | `coset_lde_batch_ext3_into` |
| R2 composition parts LDE (ext3, `number_of_parts > 2` branch) | `round_2_compute_composition_polynomial`, `prover.rs:948` | ~8 (one per big table) | `evaluate_poly_coset_batch_ext3_into` |
| R4 DEEP-poly extension (ext3) | `round_4_compute_and_commit_fri_layers`, `prover.rs:1107` | ~8 | `coset_lde_batch_ext3_into` with uniform `1/N` weights |
| R2 `extend_half_to_lde` (ext3, 2-halves batch) | `decompose_and_extend_d2`, `prover.rs:832` | **0** — only tiny tables hit that branch in current VM | Infrastructure in place but size gate skips it |

The ext3 path costs no extra CUDA: an NTT over an ext3 column is
componentwise equivalent to three independent base-field NTTs sharing
the same twiddles, because a DIT butterfly's multiplication is `base *
ext3 = componentwise base*u64`. Stark de-interleaves the 3n u64 slab
into 3 base slabs in the pinned staging buffer, runs the existing
`*_batched` kernels over 3M logical columns, and re-interleaves on the
way out.

### Backend (`device.rs`)

- CUDA context, pool of 32 streams (round-robin via AtomicUsize).
- Single shared pinned host staging buffer (`cuMemHostAlloc` with
  flags=0: portable, non-write-combined). Grown once per process to the
  largest LDE seen; serialised by a Mutex per call so concurrent rayon
  workers don't step on each other. Per-stream buffers blew up pinned
  memory 32× and forced first-call re-alloc on every new table size.
- Twiddle cache per `log_n` (both fwd and inv), populated on a separate
  utility stream.
- Event tracking disabled globally (`disable_event_tracking()`) — cudarc
  normally creates two events per `CudaSlice` alloc, which serialised
  concurrent callers on the driver context lock and added per-alloc cost.

### Kernels (`kernels/ntt.cu`)

- `bit_reverse_permute_batched`, `ntt_dit_level_batched`,
  `ntt_dit_8_levels_batched` (shmem fusion of first 8 DIT levels),
  `pointwise_mul_batched`, `scalar_mul_batched`.
- Parity-tested against CPU up to `log_n = 20` in `tests/lde_batch*.rs`
  and `tests/evaluate_coset_ext3.rs`.

### Microbenches (RTX 5090, 46-core host, blowup=4, warm)

| Size | CPU rayon | GPU batched | Ratio |
|---|---|---|---|
| 64 cols, log_n=16 (LDE 2^18) | ~75–100 ms | ~15–20 ms | **5–12×** |
| 20 cols, log_n=20 (LDE 2^22, prover-scale) | ~470 ms | ~220 ms | **~2.0–2.3×** |

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
