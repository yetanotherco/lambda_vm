# PR-3 plan — memory-model walk (trace-gen stage 3) on GPU

Goal: move the register memory-model walk onto the GPU so the walked ops feed the
already-device-resident MEMW_R fill with **no host round-trip** — removing the
biggest remaining H2D (MEMW_R ≈ 15M rows on ethrex) and the dominant sequential
CPU walk cost.

## ⚠️ Correction to prior notes
`reports/tracegen/GPU-TRACEGEN-DESIGN-V2.md` §10 claims a device register walk
(`trace_walk.cu`, `src/trace_walk.rs`, `walk_registers.rs`, CPU decomposition
`emit_register_accesses`/`walk_register_accesses`) is **built and validated**.
**None of it exists in the current tree** (verified). PR-3 writes the walk from
scratch; §9 Q1/Q2 of that doc is a sound *spec* only. What actually shipped is the
fill-from-host-walked-rows path.

## Current state (verified)
- The walk is a **single sequential CPU pass**, `collect_ops_from_cpu`
  (`prover/src/tables/trace_builder.rs:537`), interleaving stage 2 (op collection),
  stage 3 (walk via `&mut MemoryState`/`&mut RegisterState`), stage 4 (derive).
- `RegisterState` (`trace_builder.rs:156`): `[(value, last_ts); 32]` + x254/x255.
  `MemoryState` (`:83`): per-byte `(u8, u64)` paged store.
- **Load-bearing semantic** (`collect_register_ops_from_cpu:953`): read-old +
  write-new where **every access advances the cell timestamp** — a *read* writes
  the cell back at the new ts. So `old_ts` = the *previous access's* ts at that
  address (not the previous *write's*). Reads-between-writes is the case a naive
  "chain on writes only" walk gets wrong.
- **Seam is already device-ready:** `memw_register_fill`
  (`crypto/math-cuda/kernels/trace_cpu.cu:960`) takes SoA `reg_addr, ts, value,
  is_read, old_value, old_ts, row_index`. `old_value`/`old_ts` are exactly what the
  walk produces — today host-computed + uploaded; the device walk produces them in
  VRAM.
- Register keyspace is tiny: word addresses 0–63 (x0–x31), 508 (x254), 510/511
  (x255) → **≤512 buckets** → counting sort, no comparison/radix sort.
- Build model: `nvcc → PTX → cudarc`, no CUB/thrust, additive scans already exist
  (`inverse.cu`/`logup.cu`).

## Scope: register walk (MEMW_R) only, first
Biggest table + simplest address space + seam already there. Defer the full memory
walk (radix sort over u64 addresses for MEMW/MEMW_A/LOAD/STORE) and on-device
precompile *value* computation to later PRs.

## Device algorithm (registers)
Accesses are emitted in **strict global ts order** (`ts = 4·i+4`; M1@ts, M3@ts+1,
M5@ts+2, PC@ts+1), so per-register subsequences are already ts-sorted — no ts sort.
1. **Emit** (1 thread/instr, stateless map): `reg_addr, ts, value, is_read` per
   access. Precompile/commit/halt register accesses pre-emitted into the same
   stream (values host-computed for now; addresses/timestamps positional).
2. **Stable group-by-register**: privatized shared-mem histogram over ≤512 buckets
   → exclusive scan (reuse existing) → stable scatter into contiguous per-register
   runs.
3. **Predecessor link** (fully parallel): within a run, `old_ts[p]=ts[p-1]`,
   `old_value[p]=value[p-1]`; first element seeds from the per-epoch register init
   (`ts=1`). **Feed `is_write=1` for every access into the recurrence** (reads
   advance the chain). No sequential carry — predecessor is just the prior compacted
   element.
4. **Row placement**: host-computes each access's MEMW_R `row_index` at emit time
   (known without `old_ts`); the fill scatters by it.
- MEMW_R routing (`is_register_op`/`reg_ts_delta_in_range`, `:1561`/`:1582`) stays
  on host for PR-1 (cheap per-access u64 compare); device-side partition later.

## Integration seam
- Walk outputs device-resident `old_value_d`/`old_ts_d`. Add
  `fill_memw_register_from_dev` (device-input variant of `fill_memw_register_on`,
  `trace_cpu.rs:809`) that takes `&CudaSlice<u64>` and skips the H2D of the
  walk-produced arrays — mirroring the LDE `InnerInput::Host|Dev` seam.
- Result attaches via `set_main_input_dev` (as `build_memw_register_chunk`,
  `gpu_trace.rs:166`) → existing `commit_main_trace` device branch. No downstream
  change.

## Correctness (parity oracle, mirrors `prover/src/tests/gpu_fill_tests.rs`)
1. **CPU decomposition == sequential walk**, byte-for-byte (no GPU). Must cover:
   **read-between-writes** (asserts `old_ts` = intervening read's ts), x255 PC
   chaining, x0 suppression, continuation seeding (`from_init`).
2. **Device walk == CPU walk** (cuda, `.expect()` so a silent fallback panics),
   multi-block inputs, hot/skewed registers.
3. **Device walk → device fill byte-parity** (extend `gpu_memw_register_fill_matches_cpu`).
4. **e2e prove+verify** on ethrex 5-tx (precompiles) vs `LAMBDA_VM_CPU_TRACE=1` A/B.
   MEMW_R rides a permutation-invariant bus → a verifying proof is the multiset
   guarantee; byte-parity (1–3) is the stronger local contract.

## Risks
- Read-old/write-new semantics (highest) — mitigated by test 1's read-between-writes.
- Continuation carry (x254 commit-index / x255 PC seeding across epochs).
- Precompile-inclusive register accesses (must emit in ts order).
- Determinism: same-address accesses never share a ts (M1/M3/M5 = ts/+1/+2), so
  stability is automatic; bus is permutation-invariant anyway.

## PR sizing
- **PR-3a (CPU, no GPU, no behavior change):** add `RegAccess`,
  `emit_register_accesses`, `walk_register_accesses` (reference), behind a
  kill-switch with the sequential path as default+oracle; `PartialEq` on
  `RegRow`/`MemwOperation`; parity test 1. Isolates the risky semantics off-GPU,
  CI-testable without a GPU.
- **PR-3b (GPU, kill-switch):** `trace_walk.cu` + `src/trace_walk.rs::gpu_walk_registers`
  (histogram/scan/scatter/link) + `fill_memw_register_from_dev`; wire
  `gen_memw_registers` (`trace_builder.rs:3283`) GPU-first with CPU fallback; tests
  2–4. Independently validatable on a GPU box via kill-switch A/B.
