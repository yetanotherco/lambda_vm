# GPU Trace-Generation Design (v2 — fresh)

**Date:** 2026-07-08
**Branch context:** `perf/tracegen-cpu-optimizations` (current CPU trace-gen), targeting the CUDA prover path.
**Goal (user, 2026-07-08):** **100% of trace generation runs on the GPU — no CPU/GPU split.** Everything that is part of trace generation moves to the device, *including* the keccak / elliptic-curve precompile crypto. A permanent "half and half" is explicitly rejected (it's what made v1 slower).
**Scope decision:** **prover-only** — changes confined to `prover/src/tables/*`, `crypto/math-cuda/*`, and the `crypto/stark/*` LDE seam. The executor is **not** touched.
**Out of scope (already done elsewhere):** the heavy proving math (LDE / Merkle / FRI) is already on GPU; constraint evaluation is being ported in a separate PR. This effort is *only* trace generation.
**Status:** design + plan + architecture options (§8) + decided solutions (§9). **Implementation underway on branch `tracegen-gpu`** — see §10 for progress.

---

## 10. Implementation progress (branch `tracegen-gpu`)

**P0 — instrumentation + kill-switch** ✅ built & verified.
- `LAMBDA_VM_CPU_TRACE=1` kill-switch (`prover/src/tables/gpu_trace.rs`).
- Transfer counters (`instruments.rs`) surfaced as `[transfer] main-trace H2D: X MiB over N tables; device-resident builds: M`.

**P1a — device→LDE seam** ✅ built & validated on RTX 5090.
- `coset_lde_row_major_with_merkle_tree_keep_dev` (base-field device-input LDE), `gpu_lde.rs` wrapper, `TraceTable.main_input_dev` handle, `commit_main_trace` device branch (host fallback preserved).
- Seam parity test (`crypto/math-cuda/tests/lde_dev_parity.rs`): `_keep_dev` produces a **byte-identical Merkle root + LDE** vs the host path — both tests pass on the 5090.

**P1b — `trace_cpu.cu` CPU-table kernel** ✅ built (nvcc, compute_120) & bit-correct.
- One thread/row, row-major, pure bit-slicing (no field reduction); packed-op input at stride 11 u64/op.

**P1c — wire + go/no-go** ✅ **PASSED (2026-07-08, RTX 5090, ethrex 5-tx).**
- CPU table built entirely on device (host table left zeroed), fed to the LDE with no upload. `[transfer]` shows **8 CPU chunks device-resident**; the other 18 table commits still upload (~1.6 GiB).
- **`Verification succeeded!`** — the device-built CPU trace yields a valid proof, so the kernel is bit-correct AND nothing reads the host trace-domain table (aux uses the resident snapshot, queries the device tree, R2 the LDE). This is the go/no-go gate — **passed.**
- A/B sign test (GPU trace vs `LAMBDA_VM_CPU_TRACE=1`, ethrex 5-tx, RTX 5090):

  | mode | trace_build (median) | total (median) |
  |---|--:|--:|
  | GPU (CPU table resident) | ~4.48s | ~30.0s |
  | CPU (kill-switch) | ~4.57s | ~30.8s |

  **Break-even-to-slightly-positive with a single table on device — the sign is NOT negative.** This is the decisive contrast with v1 (which was ~1.5s *slower* from D2H+rebuild). The device-resident seam removes the CPU table's H2D (8 chunks resident; the other 18 commits still upload ~1.6 GiB) with no regression. The full speedup arrives as the high-volume tables (memw_register ~15M rows, bitwise, memw) move to device in P2/P3 — that removes their CPU fills *and* their H2D together. **Go/no-go: GO.**

**P2 (device memory-walk) — register slice: core primitive ✅ validated on RTX 5090 (2026-07-08).**
- `crypto/math-cuda/kernels/trace_walk.cu` + `src/trace_walk.rs::gpu_walk_registers`: stable group-by-register (per-block histogram → global stable offsets → stable scatter) + per-register carry scan recovering `(old_value, old_ts)` — the last write before each access, seeded by the register init.
- Confirmed semantics: a register cell holds the *last write's* `(value, ts)`; reads return it unchanged, writes update it — so the carry advances only on writes and every access reads the carry-before-it. Matches `RegisterState` exactly.
- Parity test `crypto/math-cuda/tests/walk_registers.rs` (device vs sequential last-write reference): **both cases pass**, incl. multi-block (100k, 500k accesses > the 4096/block tile).
- **Perf: parallelized ✅.** The first carry scan was one-thread-per-register and serialized on hot registers (measured 871 ms at 12M accesses / 1.23M hot group). Replaced with a fully parallel scan: gather to contiguous sorted order, then `old = last-write-before-p` via two non-segmented **inclusive prefix-max** scans (last-write-position and segment-start), clamped so the write must lie in p's own register group; else the init seed. No sequential carry.
  - Result at 12M accesses (1.23M hot group), RTX 5090: **871 ms → 654 ms (gather) → 141.6 ms (parallel scan)**, still bit-exact (all parity cases pass). That is **~3× faster than the CPU register walk (~400–500 ms)** and scales with the GPU — and the 141 ms still includes ~250 MB of H2D+D2H that disappears once accesses are born on-device and `old` feeds device fills.
  - Kernels (`trace_walk.cu`): `walk_gather`, `walk_make_scan_inputs`, `pmax_block`/`pmax_offsets`/`pmax_add` (generic u32 inclusive prefix-max), `walk_combine`, `walk_unscatter`.

**P2c — register-slice decomposition (CPU side) ✅ built & validated locally (2026-07-08, no GPU needed).**
- Confirmed the walk model in code: **read-old + write-new** (`trace_builder.rs:830`) — *every* access (read or write) advances the cell timestamp, because reads write back the same value. So `old_ts` is the **previous access's** ts (not the last write's) and `old_value` is the last write's value. ⇒ the device walk must be fed **`is_write = 1` for all register accesses**; the kernel is correct as-is (its `walk_registers.rs` parity test uses a "last-write only" reference with random `is_write`, which validates the general primitive but not this usage).
- New state-free decomposition in `trace_builder.rs` (all `#[allow(dead_code)]` until wired): `RegAccess` + `emit_register_accesses` (pure per-op emit of M1/M3/M5 + the implicit per-instruction PC write, no register-state threading), `walk_register_accesses` (CPU reference = **the swap point** for `gpu_walk_registers`), and `collect_register_ops_parallel` (emit → walk → reconstruct via the shared `MemwSink::push_reg_access`, so routing + push order match the sequential path). `RegRow`/`MemwOperation` gained `PartialEq`/`Eq`.
- Byte-parity tests `walk_decomp_tests` (3, all pass, `cargo test -p lambda-vm-prover --lib walk_decomp`): the load-bearing one is `read_between_writes_uses_previous_access_ts` (a read between two writes ⇒ `old_ts` = the intervening read's ts, which a "last write" walk gets wrong), plus PC/`rs1==255` + implicit-PC-write chaining, and a 200-op mixed sequence. Consistency note: the sequential path sets a read row's `old_value` to the read value `op.rvX` (valid — a read returns the current value); the walk recovers it from the previous access, so they agree only on a *consistent* register trace (always true in real execution; the mixed test threads a shadow register file to honor it).

**P2c — GPU walk wired & proven on device ✅ (2026-07-08, RTX 5090).** `walk_register_accesses_gpu` (cuda) marshals the emitted `RegAccess` stream into the SoA `gpu_walk_registers` consumes — `is_write = 1` for every access, `nbins = 512`, per-register `init_value` from the seed `RegisterState` — and `collect_register_ops_parallel` routes through it (CPU walk as fallback / kill-switch). A cuda-only test `gpu_walk_matches_cpu_reference` calls the device walk directly and `.expect()`s it (so a silent fallback panics rather than passes), asserting the device result equals the CPU reference on ~240k accesses (60k ops, multi-block); all four `walk_decomp` tests pass on the box (`cargo test -p lambda-vm-prover --lib --features cuda walk_decomp`). The device kernel + the `is_write = 1` usage are therefore bit-correct.

**P3 — MEMW_R device fill (the biggest table) ✅ built & validated on device (2026-07-08, RTX 5090).** The MEMW_R table (~15M rows on ethrex — the dominant volume) is now built **entirely on device**: the walk's recovered `(old_value, old_ts)` stay resident in VRAM and feed a fill kernel in place (no D2H). `walk_core` was extracted from `gpu_walk_registers` to return the device-resident `old_*` slices; `memw_register_fill` (in `trace_walk.cu`) writes the 10 MEMW_R columns row-major `[row*10+col]`; `gpu_build_memw_register_trace` returns a residency-ready `CudaSlice<u64>` (mirrors `gpu_build_cpu_trace`). To avoid a device-side stream-compaction, each access's compacted MEMW_R **row index is computed on the host from `emits_row`** (known at emit time, no `old_ts` needed) and passed in; the walk output is consumed by the fill with no round-trip. Validated by `gpu_memw_register_fill_matches_cpu`: the device row-major buffer is **byte-identical** to the CPU `generate_memw_register_trace_from_rows` (60k ops; all 5 `walk_decomp` tests green on the box). Scoped to no-fallback inputs (every register access routes to MEMW_R); the rare `ts`-delta > 2¹⁶ fallback is a TODO.

**P3 — MEMW_R wired into commit & prove+verify on ethrex ✅ (2026-07-08, RTX 5090).** The MEMW_R table now builds on device and feeds the `_keep_dev` LDE with no full-column upload — validated **end-to-end on the real ethrex 5-tx block (precompiles included): "Verification succeeded!"**. To be program-agnostic (correct *with* precompiles), the columns are filled on device from the sequential walk's `RegRow`s via `gpu_fill_memw_register` (reusing the `memw_register_fill` kernel; `RegRow::fill_soa` marshals the SoA), attached as `main_input_dev` by `gpu_build_memw_register_tables`, and dispatched by a GPU-first `gen_memw_registers` hook mirroring `gen_cpus`. Only the compact `RegRow` SoA is uploaded, not the column matrix. **A/B (ethrex 5-tx):** main-trace H2D dropped **2802 MiB → 866 MiB (~69%, ~1.9 GiB removed)** and device-resident builds 0 → 17 (CPU table + MEMW_R chunks); trace-build 4.47 s → 4.09 s; total ~flat (FFT-dominated at this scale — the transfer isn't yet the critical path, but the reduction compounds toward "log in, roots out"). Both paths verify; the `LAMBDA_VM_CPU_TRACE` kill-switch fallback is confirmed (0 resident).

**P3 — all memory tables now device-resident ✅ (2026-07-08, RTX 5090).** MEMW_A (29 cols), LOAD (18 cols), and STORE (16 cols) joined MEMW_R + the CPU table on device (per-row-map fills from the walked ops, `_keep_dev` LDE, GPU-first hooks mirroring `gen_cpus`; byte-parity unit tests + prove+verify on ethrex all green). Cumulative main-trace H2D on ethrex 5-tx:

| stage | H2D | resident builds |
|---|--:|--:|
| all-CPU baseline | 2802 MiB | 0 |
| + CPU table + MEMW_R | 866 MiB | 17 |
| + MEMW_A | 634 MiB | 19 |
| + LOAD + STORE | **498 MiB** | **21** |

**H2D down 82%** (2802 → 498 MiB), all verify. Total time stays ~30–32 s throughout (FFT-dominated; the transfer is not yet on the critical path — this is residency progress toward "log in, roots out", not yet a wall-clock win). A per-commit dims diagnostic (`LAMBDA_VM_H2D_DIMS=1`) breaks the aggregate down by table (column count is the fingerprint — note 29 = MEMW_A *and* SHIFT, 26 = bytewise *and* mul). Remaining 498 MiB: SHIFT 232 + LT 204 (ALU dedup), bytewise/mul 52, PAGE 10.

**P3 — SHIFT + LT device fills ✅ (2026-07-13, RTX 5090) + latent logup bug fixed.** SHIFT (29 cols, a per-row *compute* kernel replicating `compute_aux`/`compute_shifted`; byte-parity) and LT (17 cols, host per-chunk HashMap dedup → device fill; multiset-parity) now build on device. Cumulative main-trace H2D on ethrex 5-tx is now **2802 MiB → 62 MiB (~98%)** with **24 device-resident commits**, and the full proof verifies. Wiring SHIFT surfaced a **pre-existing latent bug**: the launch guard in `crypto/math-cuda/src/inverse.rs` (`batch_inverse_ext3_dev` + `compute_and_invert_denoms`) and `logup.rs` used `n > u32::MAX / BLOCK_SIZE` — that's the grid-*y/z* cap (65535), but grid-*x*'s limit is 2³¹−1, so it was 256× too strict. SHIFT's logup aux batch-invert (18 interactions × 2²⁰ rows = 18.9M > `u32::MAX/256` ≈ 16.77M) tripped it → `logup_aux_resident` returned `Err` → the aux **silently fell back to the zeroed host placeholder** (device-resident tables keep a zeroed host trace) → wrong aux → R2-4 verify failed. Host-uploaded tables tripped the same guard but fell back to *real* host data (correct, just slower), which is why it lay dormant. **Fix: guard on `> u32::MAX`** (the cast/index bound) at the three sites. Soundness-adjacent (silent wrong fallback) — flag to the team. Remaining uploaders: bytewise/mul (~52 MiB), PAGE (~10 MiB).

**Remaining for P2/P3:** (1) **finish the ALU/dedup tables** (eq/bytewise/mul/dvrm/branch — the ~52 MiB bytewise/mul) — host per-chunk HashMap dedup (unchanged) → device fill from the deduped op SoA → resident LDE; order-independent (LogUp bus) so validated by prove+verify. (2) emit **precompile register accesses** (keccak/ecsm/commit, ts-ordered, host-computed values for now) so a real ethrex block byte-parities, then wire the on-device WALK into `collect_ops_from_cpu` so MEMW_R is FULLY resident (walk → fill → LDE, no `RegRow` H2D) and drop the decomposition's `#[allow(dead_code)]`s; handle MEMW_R fallback rows (`ts`-delta > 2¹⁶). PAGE keeps its preprocessed/multiplicity split (commit-side). Memory-access slice (radix sort by u64 address) is the same stable-group primitive but low leverage (~1M accesses).

> This supersedes the earlier table-by-table GPU port design, which was a net loss.
> See §2 for exactly why, so we don't repeat it.

---

## 1. The current boundary (measured against current code)

### Input
- Trace-gen's entire input is the execution log: `Log` = 5 × `u64` = **40 B/instruction, POD, fixed-stride** (`executor/src/vm/logs.rs:15`), plus a static per-program decode table (KB).
- For 10-tx ethrex (~6.8M instructions) the log is **~272 MB**. That is the theoretical minimum we must move to the device.

### Output & handoff
- `Traces` (`prover/src/tables/trace_builder.rs:2649`) is **~26 separate `TraceTable`s** (`cpus`, `memws`, `memw_registers`, `loads`, `lts`, `bitwise`, `page`, `branch`, `decode`, precompile tables, …), several of them chunked into `Vec<TraceTable>`.
- Each table is committed **independently** via `commit_main_trace` (`crypto/stark/src/prover.rs:764`).
- The GPU commit path calls `trace.main_data_row_major()` — a **host, row-major, `u64`** buffer (`crypto/stark/src/trace.rs:300`) — and passes it to `try_expand_leaf_and_tree_row_major_keep` (`crypto/stark/src/gpu_lde.rs:447`), which **H2D-uploads** the raw slice (`gpu_lde.rs:477` → `math_cuda::lde::coset_lde_row_major_with_merkle_tree_keep`).
- Summed across all tables, this upload is **several GB per proof** (the CPU table alone ≈ 1.79M rows × 74 cols × 8 B ≈ **1.06 GB** at 5-tx-ish sizes; MEMW_R ≈ 3.88M × 10 × 8 ≈ 310 MB; bitwise 2²⁰ × 21 × 8 ≈ 176 MB; etc.).

### Field representation (enables zero-copy reinterpret)
- Goldilocks is **canonical `u64`, no Montgomery**; `FieldElement` is `repr(transparent)` over `u64`. The trace `Table` stores data as effectively `Vec<u64>`, row-major. The existing GPU path already `transmute`s host `&[FieldElement]` ↔ `&[u64]` (`gpu_lde.rs:477,495`).

### What already exists on our side (do not rebuild)
1. **A device-resident input path into the main LDE already exists.** `coset_lde_row_major_inner` (`crypto/math-cuda/src/lde.rs:410`) accepts `InnerInput::Host(&[u64])` **or** `Dev(&CudaSlice<u64>)` (`lde.rs:386,448`). The `Host` arm does the `memcpy_htod`; the `Dev` arm skips it. This is the seam a device-born trace feeds into with **no upload**. (#762's aux path already uses the `_dev` variant.)
2. **Post-LDE residency (#748/#762).** `TraceTable` carries `main_trace_dev: Option<ResidentMainTrace>` (column-major `[col*rows+row]`, `crypto/stark/src/trace.rs:47`) and `aux_resident` — but these retain the trace **after** the LDE for the aux fingerprint kernel. The **generation → LDE** upload is still host→device. That upload is the target.

### Where the CPU time actually is (measured, ethrex 10-tx)
| phase | share | nature |
|---|---|---|
| p2a sequential memory-model walk | **~47% and rising** | sequential (last-writer dep), bandwidth-bound |
| p4 bitwise collect | ~18–27% | histogram, parallelizable |
| p5 table fills | ~17% (shrinking) | **already parallel** (rayon scope + `into_par_iter`) |
| p0/p1 decode + cpu-ops | small | — |

The fills are already the small, already-parallel part.

---

## 2. Why the previous design failed (and the reframe)

The prior design ported **table fills (p5)** to the GPU, one table at a time. It was wrong twice:

1. **It attacked the wrong phase.** Fills are ~17% and already parallel; the real cost is the sequential walk (~47%), which was left entirely on the CPU.
2. **It moved the same bytes twice.** With the walk on the CPU, per-op data lives on the host, so a GPU fill must *upload the ops* (H2D) and then either *download the result* (D2H) or keep it resident. Measured on ethrex_simple_tx (RTX 5090): forward marshal+H2D ≈ **1.4 s**, backward D2H+rebuild ≈ **0.98 s** → **net ≈ 1.5 s slower** end-to-end. GPU compute was negligible; **data movement ate the entire budget.**

**Reframe (the load-bearing principle):**

> Transfer only disappears if the data is **born where it is consumed.** The LDE runs on the device, so trace generation must run on the device — **including the walk.** Any split (walk on CPU, fill on GPU) merely relocates the same gigabytes and cannot win.

This is the single idea the v1 design violated. Everything below follows from it.

---

## 3. The GPU-native pipeline

Rethink the pipeline around its one input (the log) and its one consumer (the LDE). The walk's dominant cost is recovering, per memory access, its predecessor `(old_value, old_ts)`. That is a *previous-occurrence-per-key* problem = **sort by `(addr, ts)` + segmented scan** — a textbook data-parallel GPU primitive. (RISC0 / ZisK recover predecessors exactly this way via an address-sort + permutation argument; our sequential re-walk is the outlier.)

```
 TODAY  — several GB of H2D scattered across the proof
   CPU:  logs ─walk(sequential)→ ops ─fills(parallel)→ ~26 host row-major matrices
   GPU:                                     └─ per table: H2D (GBs) → coset_lde → Merkle

 PROPOSED — "log in, roots out"
   HOST → ONE H2D:  logs (~272 MB @10tx) + decode table (KB)
   ┌──────────────────────── GPU, resident end-to-end ────────────────────────┐
   │  1. emit access records   (1 thread/instr: reg M1/M3/M5 + mem + ecall)     │  parallel
   │  2. radix sort by (addr, ts)                                               │  parallel
   │  3. segmented scan → (old_ts, old_value) per access                        │  parallel
   │  4. per-table fill kernels → row-major u64 buffers  (fused compute)        │  parallel
   │  5. coset_lde  **Dev-input path**  → Merkle tree       (consumed in place) │  NO H2D
   │  6. aux build reads resident main (#762) → aux LDE → DEEP/FRI              │
   └───────────────────────────────────────────────────────────────────────────┘
   D2H: only Merkle roots + query openings (KB).
```

**Transfer budget: ~GB of table matrices → ~272 MB (the log), once.**

Design invariants:
- **Row-major from birth.** Fill kernels write `buf[row*cols + col]` so the buffer *is* exactly what the existing LDE `Dev` input arm expects (`lde.rs:448` copies `input[0..n*cols]` row-major); no transpose, no host round-trip. (The LDE *output* / retained snapshot is column-major — that's the LDE's job, not ours.)
- **One thread per row** (including padding rows), fused with each table's per-row compute (the same arithmetic the CPU `generate_*_trace` does).
- **Parity is the contract.** Deterministic tables (CPU/memory/cpu32/keccak) validate by byte-parity vs the CPU builder; order-independent tables (ALU/LT/BITWISE, which ride the permutation-invariant LogUp bus) validate by prove+verify or multiset equality.

---

## 4. The one hard blocker: precompiles

Confirmed at `collect_keccak_memw_ops` (`prover/src/tables/trace_builder.rs:1354`) and the ecsm/commit collectors: precompiles thread `&mut memory_state` / `&mut register_state`, **read live memory** for `old_ts` (`memory_state.read_byte(...)`), and mutate state — all **interleaved in timestamp order** with normal accesses. So you cannot carve "non-precompile accesses" onto the GPU in isolation: a normal access's predecessor may be a keccak-lane write.

Because the goal is **100% on GPU**, the precompiles go on the device too — both their memory accesses *and* their crypto compute. Two sub-problems:

1. **Predecessor recovery for precompile accesses** — handled for free by the same `(addr, ts)` sort + segmented scan as every other access, *provided* the precompile access records (keccak lanes, EC memory I/O) are emitted into the stream on device. Their timestamps are positional/known, so the records can be produced in parallel.
2. **Precompile crypto compute on GPU** — the genuinely hard, novel kernels:
   - **keccak-f1600** round function → the KECCAK / KECCAK_RND tables (KECCAK_RND is 1480 cols, 24 rows/call). A dedicated permutation kernel. Row-volume is small (~80–100 calls/block) so it's about *correctness*, not throughput.
   - **secp256k1 EC witness** (`crypto/ecsm/src/witness.rs::compute_witness`, double-and-add) → ECSM / EC_SCALAR / ECDAS tables. The hardest single kernel (256-bit modular field/point arithmetic). Rare (0–4 calls/block) but must be exact.

Note: the existing `crypto/math-cuda/kernels/keccak.cu` hashes for Merkle/transcript — it is **not** the keccak-f1600 *trace* round function, so it doesn't directly transfer.

⇒ These precompile kernels are the **last and hardest phase**, but they are **in scope** — nothing about trace generation stays on the CPU when this is done. Still prover-only (no executor change).

---

## 5. Phased plan (prover-only, each phase independently shippable)

Every phase sits behind a `LAMBDA_VM_*` kill-switch with a CPU fallback, and is validated by prove+verify (order-independent tables) or byte-parity (deterministic tables).

### P0 — Instrument & kill-switch *(no behavior change)*
- Add per-table + boundary **transfer timing** on the cuda commit path (H2D bytes + µs per table) and an A/B kill-switch env var.
- Baseline the real H2D bytes/time on GPU (ethrex 5-tx) so every later phase has a number to beat.

### P1 — Prove the seam on ONE big table *(the go/no-go)*
- Take the single largest pure per-row-map table (**CPU**). Generate its **column-major `u64`** buffer directly in VRAM and feed the **existing `Dev` LDE path** (`lde.rs` `InnerInput::Dev`) — no H2D of the matrix, no D2H, no host rebuild.
- Validate: prove+verify green **and** Merkle-root byte-parity vs the host path; confirm the **sign flips** (net faster on this table, not slower). This empirically tests the §2 thesis before we build the walk.

### P2 — The walk on device
- Emit access records in parallel (1 thread/instr: register M1/M3/M5 + memory + ecall lanes). Precompile values are host-computed *for now* so their records join the stream (this is temporary scaffolding — P4 moves that crypto on-device; it does not become a permanent CPU dependency).
- **Radix sort by `(addr, ts)`** → **segmented scan** → `(old_ts, old_value)` per access.
- Validate the recovered predecessors **byte-parity** vs the sequential `collect_ops_from_cpu` walk.

### P3 — Device fills feeding the LDE (the payoff)
- Fill the high-volume tables directly into column-major device buffers from the scanned records, each consumed **in place** by the LDE `Dev` path: **CPU, MEMW_R, MEMW/MEMW_A, LOAD, LT, and the bitwise histogram**.
- At this point the **log is the only upload**; the trace never lands on the host.

### P4 — Precompiles on GPU (last & hardest, but in scope)
- **keccak-f1600** kernel → KECCAK / KECCAK_RND / KECCAK_RC tables (byte-parity vs CPU; small volume).
- **secp256k1 EC witness** kernel → ECSM / EC_SCALAR / ECDAS (the hard one; byte-parity).
- Their access records already join the P2 sort+scan stream. After this, **nothing in trace generation runs on the CPU** — the goal is met.

---

## 6. Validation contract (unchanged from CPU work)
- Order-independent tables (ALU/LT/BITWISE ride permutation-invariant LogUp) → **prove+verify or multiset equality**, not byte-parity.
- Deterministic fills (CPU/memory/cpu32/keccak) → **byte-parity** ok.
- Keep the old path behind an env kill-switch for one-flag A/B.
- Bench methodology: ABBA, medians, ethrex 5-tx, on the GPU box; report total prove, trace-gen, trace-gen%.

## 7. Open risks
- **Radix sort at scale in VRAM.** ~20M+ access records at 10-tx; need a sort that fits the VRAM budget alongside the LDE peak (chunking may be required).
- **Precompile record emission correctness.** The host must emit precompile accesses into the stream in exactly the timestamp order the sequential walk assumed — soundness-sensitive; guarded by byte-parity in P2.
- **Chunking interaction.** Tables are chunked (`max_rows`); the device pipeline must preserve per-chunk padding/pow2 geometry so bus multiplicities match.
- **Sign confirmation gate.** If P1 does **not** flip the sign on the CPU table, stop and re-examine before building the walk — that would falsify the core thesis.

---

## 8. Architecture options (the forks)

The real design forks in *how* to build the on-device pipeline. Each lists options, trade-offs, and a lean recommendation. **§9 records the decided solution for each, after deep code investigation (2026-07-08).**

### Q1 — Predecessor recovery `(old_ts, old_value)` (the crux / "the walk")
Every memory & register access needs its previous-access-to-the-same-address. That's a *previous-occurrence-per-key* problem.
- **A. One global sort by `(addr, ts)` + segmented scan.** Sort *all* ~20M+ accesses; predecessor = prior element in the same-address run. Uniform, one primitive, precompiles fold in naturally. This is the RISC0/ZisK approach. Cost: ~32 B/record ⇒ ~640 MB at 20M records + sort scratch.
- **B. Split by class (hybrid).** Registers have a tiny keyspace (≤ ~512 word addresses) and are the *dominant* volume (~16M of ~20M) ⇒ **counting/bucket sort is near-free**. Only the smaller memory+precompile slice (large address space) needs a real radix sort. Fastest, but two code paths.
- **C. Hash group-by by address + per-group sort by ts.** Avoids a global sort but per-group sorts are awkward and group sizes vary wildly on GPU.
- **Recommendation:** **A first** (simplest, one proven primitive, correctness baseline), then **optimize registers to B** if profiling shows the sort dominates. Timestamps are positional (`4·i+…`), so no state is needed to produce the sort keys — the whole thing is data-parallel.

### Q2 — Sort implementation: hand-written vs library
- **A. Hand-written LSD radix sort** (nvcc → PTX → cudarc), consistent with the repo's runtime-dlopen, no-linked-libs model. More code, but a standard kernel.
- **B. CUB / thrust device sort.** Fast & battle-tested, but needs device-link/static libs — breaks the pure-PTX cudarc pattern the whole `math-cuda` crate relies on.
- **Recommendation:** **A** — stay within the existing build model (`build.rs` compiles `.cu` → PTX; `Backend` loads modules). Don't introduce a linker dependency.

### Q3 — Dataflow: intermediate op arrays vs fused
- **A. Mirror the CPU** — emit device arrays of the same op structs, then per-table fill kernels read them. Easy 1:1 port of `generate_*_trace`; more VRAM (intermediate arrays).
- **B. Fused** — go straight from scanned access records to column-major table buffers, one kernel per table, no intermediate op array. Less VRAM, fewer passes.
- **Recommendation:** **B for direct-map tables** (CPU, memory tables — pure per-row maps), **A for tables needing dedup** (ALU: LT/EQ/MUL/DVRM group-by unique operands first, then fill). Pragmatic mix.

### Q4 — The main-trace → LDE device seam
- The aux path already has a device-input LDE (`coset_lde_ext3_..._keep_dev`). The main path needs the analogous variant.
- **Plan (not really a fork):** add a `main_input_dev`-style handle to `TraceTable` (mirror `aux_resident` / `main_trace_dev`), add a `try_expand_leaf_and_tree_row_major_keep_dev` that calls the existing `InnerInput::Dev` arm (`lde.rs:448`), and branch `commit_main_trace` to the device path when a table carries a device buffer (fall back to host upload otherwise — the same pattern already used).

### Q5 — VRAM budget & chunking
Must co-reside: log + access records + sort scratch + a table's column buffers + the LDE peak. Tables are already chunked by `max_rows`.
- **A. Stream per table/chunk.** Run the shared walk (sort+scan) once; then for each table/chunk: fill → LDE → commit → free. Only one table's buffers + the LDE peak live at once. Matches the existing per-table commit loop.
- **B. Keep everything resident.** Simpler dataflow, risks OOM on big blocks.
- **Recommendation:** **A** — the walk output (per-access `old_ts/old_value`) is the one shared, briefly-resident structure; table fills stream and free, exactly like the current commit loop.

### Q6 — Column layout & field encoding (settled, listed for completeness)
- Write **column-major** `buf[col*rows + row]`, canonical Goldilocks `u64` (no Montgomery), so the buffer *is* the LDE input and reinterprets zero-copy to `FieldElement` — same `transmute` the current GPU path already uses.

---

## 9. Decided solutions (deep, code-verified 2026-07-08)

Four parallel code investigations confirmed the facts below (file:line inline). Decisions per question:

### Q1 — Predecessor recovery → **register/memory split, and neither needs a comparison sort of the hot path**
Facts: accesses are emitted per-instruction in strict timestamp order (`ts = 4·i+4`, `trace_builder.rs:348`; M1@ts+0, M3@ts+1, M5@ts+2, PC@ts+1, load/store@ts+4, `collect_register_ops_from_cpu:950`). Register word-addresses are a tiny fixed set (x0–x31→2·idx, x254→508, x255→510/511; **max 511**, `register.rs:443`), ~15M of ~16M accesses. Memory addresses are `rs1+imm`, full u64, ~1M accesses. `(addr, ts)` is unique per byte-access.

- **Key realization:** because accesses are emitted in global ts order, the subsequence for any single address is *already* ts-sorted. So we never need to sort *by ts* — only to **group by address, stably**.
- **Registers → stable counting sort** over ≤512 buckets (histogram → exclusive-scan → scatter). No comparison sort. Predecessor = previous element in the bucket; the first element seeds from the register init (regs `(0, ts=1)`, sp=`STACK_TOP`, pc=`(entry,1)`; `RegisterState::new:167`).
- **Memory → stable LSD radix sort by address** only (~1M elements, ~6–8 digit passes). Within an address, emission order = ts order is preserved. First element seeds from the image value (`ts=0`) or 0.
- **Precompiles** emit their access records into these same two streams (keccak: 1 reg read + 25 lane mem ops @ same ts, distinct addresses, `collect_keccak_memw_ops:1354`; ecsm 47 ops over T/T+1/T+2 `:834`; commit 4 reg + N byte reads `:1196`). Because they're in the sorted stream, their `old_ts` falls out of the same scan — no special handling. Values host-computed until P4.
- **Segmented scan** is then uniform over both grouped streams: `old = prev-in-group`, seeded per group.

### Q1-bonus — **one sort feeds two things (predecessor *and* final state)**
The same grouped-by-address streams give **final state for free**: the *last* element per address group is that cell's end value+timestamp. That's exactly what the **PAGE** and **REGISTER** tables (classified (d) final-state scan) need. So one grouping primitive feeds the memory-table fills (segmented-first → `old`) *and* PAGE/REGISTER (segmented-last → `FINI`). No separate final-state pass.

### Q2 — Sort implementation → **hand-written in the existing PTX/cudarc model; introduce `atomicAdd` for histograms (contained)**
Facts: `build.rs`+`device.rs` add-a-kernel checklist is mechanical; reusable Hillis-Steele additive/multiplicative scans exist (`inverse.cu`, `logup.cu`, 3-phase, 256-blocks); **no atomics anywhere today**; field helpers in `goldilocks.cuh` (`reduce128`).
- Hand-write `trace_sort.cu` (counting sort, radix pass, segmented scan). Stay in the nvcc→PTX→cudarc model — **do not** pull in CUB/thrust (would break the no-linked-libs pattern the whole crate relies on).
- Use **privatized per-block histograms in shared memory + `atomicAdd` to a small global histogram** (512 buckets for registers; 256/pass for radix). This is a deliberate, contained departure from the scan-only house style — trivial contention on a 5090, far simpler than scan-based counting. Reuse the existing additive scan for the exclusive-scan of bucket offsets.

### Q3 — Dataflow → **fused for maps, hash group-by for dedup, atomic scatter for histograms, sort-tail for final-state**
Per the classification (§ investigation): 
- **(a) 10 pure-map tables** (cpu 38, memw 49, memw_a 29, memw_r 10, load 18, store 16, cpu32 38, commit 19, halt 4, decode 6) → **fused** record→row-major-columns, one thread/row, no intermediate op array.
- **(b) 7 dedup tables** (lt, eq, bytewise, shift, mul, dvrm, branch) → **hash group-by** on the operand key (keys captured per table; e.g. lt `(lhs,rhs,signed)`, mul `(lhs,ls,rhs,rs)` with dual `mu_lo/mu_hi`), summed multiplicity, then fused fill. Order-independent (LogUp bus) ⇒ grouping suffices, no sort; validate by prove+verify.
- **(c) 2 histogram tables** (bitwise 21, keccak_rc 10) → **atomic scatter-add** into the MU columns; preprocessed columns stay static. bitwise MU is the big one (~40–141M lookups) → privatized histograms + merge.
- **(d) 2 final-state tables** (page 5, register 5) → **reuse the Q1 sort tail** (segmented-last), no separate scan.
- **(e) 5 precompile tables** → P4 kernels.

### Q4 — LDE seam → **add the missing base-field `_keep_dev`; branch `commit_main_trace` on a device handle**
Facts: `InnerInput::Host|Dev` exists (`lde.rs:386,448`); ext3 has `coset_lde_ext3_row_major_with_merkle_tree_keep_dev` (`:625`) but **the base-field `_keep_dev` does not exist** — small addition. Dev arm expects **row-major** input; retained snapshot is column-major; `commit_main_trace:764` currently always uploads the host row-major slice.
- Add `coset_lde_row_major_with_merkle_tree_keep_dev(input_dev, n, m, blowup, weights)` → `coset_lde_row_major_inner(InnerInput::Dev(..), retain_trace=true)` → returns `GpuLdeBase` with `trace_dev` retained (so #762's aux path still works unchanged).
- Add a `main_input_dev: Option<Arc<CudaSlice<u64>>>` handle to `TraceTable` (mirrors `aux_resident`); trace-gen fills it row-major. `commit_main_trace` branches: handle present → `_keep_dev` (no H2D); else the current host path (fallback preserved).
- Preprocessed-split tables (bitwise/decode/page/keccak_rc/register commit precomputed cols separately) keep their split commit path; wire the device buffer per-table there.

### Q5 — VRAM & chunking → **one shared compact walk buffer, then stream per table/chunk (matches today's loop)**
Facts: the commit loop already **streams one chunk at a time**, parallel within chunk (`prover.rs:2146`), and frees device snapshots right after the aux build (`clear_main_trace_dev`, `:2269`). VRAM budget = 80% of total, async pool (`device.rs:239,275`). Largest single matrix ≈ MEMW chunk 2¹⁹×49×8 ≈ **210 MB**.
- Keep the walk output **compact** (a ~32 B/access record like `RegRow`, not the 168 B `MemwOperation`): ~16M accesses ≈ **~512 MB**, resident once.
- Order: walk (sort+scan) → fill memory tables + PAGE/REGISTER (consume walk) → **free walk** → cpu/cpu32/ALU/bitwise (from log/cpu-ops, no walk) → precompiles. Each table: fill → `_keep_dev` LDE → commit → free, exactly the current per-chunk streaming shape.
- Budget: walk (~0.5–1 GB) + one table chunk (~0.2 GB) + its LDE peak + Merkle nodes ≪ 80% of a 32 GB 5090. Chunk the walk only if a huge block ever pressures a smaller card.

### Q6 — Layout & encoding → **row-major `[row*cols+col]`, canonical Goldilocks u64**
Confirmed against the seam: emit row-major to feed the `Dev` arm with zero transpose; canonical `u64` reinterprets zero-copy to `FieldElement` (the same `transmute` the current path uses, `gpu_lde.rs:477`).

### Precompiles (P4) effort read
keccak-f1600 (KECCAK/KECCAK_RND 1480 cols, ~tens of rows/block) — **medium**: theta/rho/pi/chi/iota is data-movement-heavy but mechanical, tiny volume, byte-parity testable. secp256k1 EC witness (`crypto/ecsm/src/witness.rs:275`, double-and-add, 256-bit modular arith; ECSM 427 / ECDAS 521 cols; 0–4 calls/block) — **hard**: the one real crypto kernel. Both tiny-volume ⇒ correctness-driven, not throughput-driven; they come last.
