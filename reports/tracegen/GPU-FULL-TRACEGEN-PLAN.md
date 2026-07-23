# Plan: move ALL trace generation onto the GPU

Goal: the **entire** trace-generation pipeline runs on the GPU, end to end, with no CPU
excursion between stages. Scope is trace generation only (logs → filled + committed-ready
trace tables); the proving phase (LDE/Merkle/FRI) is out of scope here. Perf is not the
objective — *completeness* is: every step device-resident.

## End state (target architecture)

```
executor logs + decoded instructions
      │  (one upload — the ONLY host→device transfer in)
      ▼
[GPU] build CpuOperation SoA  ──►  resident op stream
      ▼
[GPU] register walk ─┐
[GPU] memory walk  ──┼─►  resident MEMW_R / MEMW_A / MEMW rows
[GPU] chip-op gen  ──┘     (LOAD/STORE/LT/SHIFT/MUL/DVRM/EQ/BYTEWISE/BRANCH/CPU32 inputs)
      ▼
[GPU] all table fills (per-op + preprocessed + precompile)  ──►  resident matrices
[GPU] bitwise histogram (all sources) ──► resident BITWISE MU columns
      ▼
resident trace tables ──► LDE (proving)     — zero CPU trace-gen work
```

Everything between the single input upload and the resident matrices stays in VRAM.

## Current state (what's already on GPU)

- ✅ **14 per-op table fills** device-resident (CPU, MEMW, MEMW_A/R, LOAD, STORE, LT, SHIFT,
  EQ, BYTEWISE, MUL, DVRM, BRANCH, CPU32) — but they consume **host-packed** op data.
- ✅ **Register walk** on GPU (built, opt-in `LAMBDA_VM_GPU_REGISTERS`).
- ✅ **Bitwise histogram** partial (in_walk + memw_reg sources, opt-in `LAMBDA_VM_GPU_BITWISE`).
- ✅ Main + preprocessed **LDE/Merkle** on GPU (proving side; fills feed it resident).

Everything else in trace-gen is still CPU: decode, CpuOperation build, memory walk, chip-op
generation, remaining histogram sources, preprocessed + precompile table fills, and the
host-side packing that feeds the GPU fills.

## ✅ Resident cpu_ops seam (2026-07-17) — the "one upload → device cpu_ops → chips read in place"
`DeviceCpuOpsResident` (trace_ops.rs) holds the Phase-0 device cpu_op fields (packed/imm/pc/rv1/
rv2/arg2/res/rvd/flags) ON DEVICE. `gpu_build_cpu_ops_resident` uploads the per-cycle Log SoA +
decode SoA ONCE, runs `build_cpu_ops` keeping outputs resident. `gpu_build_cpu32_resident_from_
devops` fills CPU32 reading those device buffers with NO re-upload. Test `gpu_cpu_ops_resident_
seam_cpu32` → byte-identical to the host-input path (4,096 rows). This eliminates the per-chip
full-SoA re-uploads (the overhead that made a mid-transition timing measurement misleading — see
the timing note). ✅ Proven for **3 chips from ONE upload**: `gpu_build_{cpu32,load,store}_resident_from_devops` all
read the shared `DeviceCpuOpsResident` — test `gpu_cpu_ops_resident_seam_cpu32` → CPU32 (4,096) +
LOAD (524,288) + STORE (524,288) byte-identical to host-input, **zero per-chip re-uploads**.
Remaining to fully realize: `*_from_devops` for SHIFT + the deduped chips (EQ/BYTEWISE/MUL/DVRM/
BRANCH — refactor the generic helpers to read device-buffer refs) + wire p5 to build the device
cpu_ops once and share it across all chip dispatches. This is the seam that makes an eventual
timing measurement reflect the design, not the scaffolding.

## Findings — structural realities (updated 2026-07-16)

Deep read of `trace_builder.rs` / `trace_walk.*` / `device.rs` revealed the true dependency
shape, which reshapes the sequencing:

1. **Ecall entanglement — every *stateful* phase bottoms out in Phase 6.** On the mandated
   workload (ethrex_5tx) the register/memory walks are NOT independent of precompiles:
   `device_registers_eligible` (trace_builder.rs:1225) *bails* if any op is
   `ecall_commit|keccak|ecsm`, because COMMIT/KECCAK/ECSM generate register+memory accesses
   (`collect_commit_memw_ops` / `collect_keccak_memw_ops` / `collect_ecsm_ops`) the device
   walk path doesn't feed. ethrex_5tx has all three. ⇒ The device register walk (Phase 1)
   currently only runs on ecall-free programs (fibonacci). To flip Phase 1 on for the real
   workload, the ecall-generated accesses must also flow through the device walk — i.e.
   Phase 1/2 completion **requires** Phase 6. Clean phase independence holds only for
   fibonacci-like guests.

2. **Memory walk needs a NEW device sort.** The register walk uses a *direct-indexed*
   histogram (`walk_seg_hist`: `sh[nbins]`, `atomicAdd(&sh[key[i]])`), valid only for small
   bounded keys (register word address < ~512). Memory addresses are sparse 64-bit ⇒ the
   counting-sort machinery does NOT generalize; Phase 2 needs a real radix sort over the
   (address, ts) key (or an address→dense-id compaction pass first). The route/compact/scan
   kernels (`memw_route_flags`, `scan_*`, `excl_scan`) ARE general and reusable.

3. **State-free ALU chips are the only clean Phase-3 slice for ethrex_5tx.** LT/SHIFT/EQ/
   MUL/DVRM/CPU32 instruction-driven ops derive purely from resident cpu_op fields (no
   `memory_state`/`register_state`), so they validate byte-parity on the real guest without
   touching Phase 2/6. LOAD/STORE/MEMW and the memw-derived LT/bitwise are Phase-2-gated.

**Revised sequencing:** state-free chip-ops (Phase 3a, done for LT/SHIFT) → device
memory-walk radix sort (Phase 2) → ecall accesses through the walks (Phase 6 seam) → flip
walks on (Phase 1/2) → stateful chip-ops (Phase 3b) → remaining histogram sources (Phase 4).

### Progress
- ✅ **Phase 0** — device `CpuOperation` builder (`gpu_build_cpu_ops`), byte-parity over
  4,036,972 ethrex_5tx cycles.
- ✅ **Phase 3a+3b** — device state-free chip-op extract for **8 chips** (all pure per-cycle
  cpu_ops projections). Kernels (trace_ops.cu): `chipop_alu_route` (6 ALU chips: LT/SHIFT/EQ/
  BYTEWISE/MUL/DVRM, shared `(rv1,arg2,alu_flags)` gather via `chipop_gather`),
  `chipop_branch_store_route` + `chipop_gather4` (BRANCH `(pc,imm,rv1,packed)` / STORE
  `(res,ts,rv2,packed)`). Wrappers `gpu_extract_alu_chipops`→`DeviceAluChips` and
  `gpu_extract_branch_store`→`(DeviceGather4,DeviceGather4)` (trace_ops.rs; route→
  `excl_scan`→gather, all on device). Test `gpu_extract_alu_chipops_matches_collect` → parity
  OK over 4,036,972 cycles: LT 338597 / SHIFT 555827 / EQ 107065 / BYTEWISE 467791 / MUL
  330693 / DVRM 4 / BRANCH 257535 / STORE 425132, on RTX 5090 box.
- ✅ **Phase 3c — CPU32** (word `*W`): device builder `build_cpu32_ops` (+ `cpu32_route`) in
  trace_ops.cu emits the 8-u64 `pack_cpu32_op` row per word cycle, compacted, with **`res`
  computed on device** — compute_aux (32-bit operand sign-ext) + cpu32_res, the SHIFT/MUL/DVRM
  result math ported bit-for-bit from the validated shift_fill/mul_fill/dvrm_fill kernels.
  `gpu_build_cpu32_ops` (trace_ops.rs) → `(flat rows*8, rows)` (feeds cpu32_fill directly). Test
  `gpu_build_cpu32_ops_matches_build_cpu32_op` → parity OK over 4,036,972 cycles (3,938 CPU32
  rows), on box. Proves chip ARITHMETIC ports correctly, not just operand gather.
- ✅ **Phase 3d — LOAD**: device builder `build_load_ops` (+ `load_route`) in trace_ops.cu emits
  the 7-u64 `pack_load_op` row per is_load cycle, compacted, with the sign/zero-extended
  `res_bytes` computed on device. `gpu_build_load_ops` (trace_ops.rs). Test
  `gpu_build_load_ops_matches_collect` → parity OK over 4,036,972 cycles (492,350 LOAD rows),
  on box. (The LOAD chip TABLE is state-free — only its MEMW read row's old_ts is Phase-2.)
  → **All 10 instruction-driven chip-op tables now generate on device** (LT/SHIFT/EQ/BYTEWISE/
  MUL/DVRM/BRANCH/STORE/CPU32/LOAD).
  **Remaining Phase 3:** MEMW row routing (Phase-2 walk integration), derived LT-from-dvrm /
  LT-from-memw / bitwise-from-*.
- ✅ **RESIDENT SEAM PROVEN (CPU32)** — `gpu_build_cpu32_resident` (trace_ops.rs) runs the full
  device→device chain: cpu_op fields → device op-build (`build_cpu32_ops`, ops stay resident) →
  device fill (`cpu32_fill` reads the resident buffer, no re-upload) → filled trace matrix.
  Test `gpu_cpu32_resident_matches_host_path` → **byte-identical to the host path** (3,938 rows,
  4,096 padded, 155,648 cells) on box. This is the first chip run *entirely* on-device from op
  fields through the filled matrix — the pattern all other chips follow to become resident. The
  only host transfers are the input upload (logs are CPU-side) and the final download (replaced
  by feeding the LDE resident in the full pipeline).
- ✅ **RESIDENT LOAD + STORE** (`gpu_build_load_resident` / `gpu_build_store_resident`, per-row
  device→device chains) → byte-identical to host path (492,350 LOAD + 425,132 STORE rows), box.
  → **3 chips run fully on-device from op fields → filled matrix** (CPU32, LOAD, STORE).
- ✅ **DEVICE DEDUP** (`gpu_dedup3`, trace_walk.rs) — the resident enabler for the 7 deduped
  chips (LT/SHIFT/EQ/BYTEWISE/MUL/DVRM/BRANCH). Multi-word LSD radix sort of a permutation by
  the full 3-word op key (reuses the mem-walk radix via `radix_sort_perm`) → `dedup_seg_start`
  marks equal-key runs → excl-scan → `dedup_emit` collapses to unique rows with summed
  multiplicity. Test `gpu_dedup3_matches_hashmap_lt` → 338,597 LT ops → 154,094 unique rows,
  **multiset-identical to the host HashMap**, on box. (Output is sorted, not HashMap order —
  fine, the deduped-chip buses are order-independent LogUp.) Every algorithmic primitive for
  on-device chip-op generation now exists; making the deduped chips resident is now pattern-
  application (extract key → gpu_dedup3 → pack → fill), like the per-row resident chains.
- ✅ **RESIDENT LT (deduped-chip capstone)** — `gpu_build_lt_resident` runs the full deduped
  chain entirely on device: cpu_op fields → `chipop_alu_route` → `lt_key_gather` → `dedup3_core`
  (device-buffer core of the dedup) → `lt_pack` → `lt_fill` → matrix, no host round-trip. Test
  `gpu_lt_resident_matches_host_path` → **multiset-identical to the host path** (154,094 unique
  rows, 262,144 padded, 4,456,448 cells), on box. → **4 chips run fully resident** (CPU32, LOAD,
  STORE per-row; LT deduped). **Both resident patterns are proven.** The other 6 deduped chips
  (SHIFT/EQ/BYTEWISE/MUL/DVRM/BRANCH) follow the LT template exactly — each = a key-gather
  kernel + a pack kernel + a wrapper (route flags already exist in `chipop_alu_route` /
  `chipop_branch_store_route`; `dedup3_core` + the fills are shared). Pure pattern-application.
- ✅ **RESIDENT EQ + BYTEWISE** via a **generic deduped-chip helper** `resident_alu_dedup_chip`
  (trace_ops.rs): one call per chip with its (key_gather, pack, fill, stride, ncols). EQ key =
  `invert` only (`eq_key_gather`); BYTEWISE key = `alu_op` (`bytewise_key_gather`); both use the
  generic `dedup_pack_abf`. Tests (gpu_lt_resident_parity.rs) → multiset-identical to host path:
  EQ 4,282 rows, BYTEWISE 165,421 rows. → **6 chips have validated resident chains** (CPU32/
  LOAD/STORE per-row; LT/EQ/BYTEWISE deduped). Chunk-builder audit: EQ/BYTEWISE/MUL/DVRM/LT/
  BRANCH dedup; SHIFT/CPU32/LOAD/STORE are per-row.
- ⚠️ **SCOPE NOTE (honest):** each resident chain reproduces the chip's **instruction-driven op
  source** (from `collect_ops_from_cpu` / the p5 cpu_ops projections) — the dominant source —
  validated device→device against exactly that source. The full production tables for LT/SHIFT/
  MUL/DVRM additionally MERGE ops from other sources: LT-from-dvrm + LT-from-memw(-aligned),
  MUL/SHIFT/DVRM-from-cpu32, MUL-from-dvrm, etc. `gpu_dedup3` accepts a merged key stream, so
  completing a full table = concatenating all its source op-streams (each device-derived) before
  the dedup. Those derived streams are the remaining Phase-3 "derived-source" work. Remaining
  deduped chips still to wire: SHIFT (per-row + cpu32 source), MUL/DVRM (dual multiplicity
  mu_lo/mu_hi & mu_q/mu_r + dvrm/cpu32 sources), BRANCH (4-field key → needs a dedup4).
- ✅ **ALL 10 CHIPS NOW HAVE RESIDENT CHAINS** (instruction-driven source). Added: dual-mult
  dedup (`dedup_emit2` + `dedup3_core2`) → `gpu_build_mul_resident` (MUL, mu_lo/mu_hi) +
  `gpu_build_dvrm_resident` (DVRM, mu_q/mu_r); 4-key dedup (`dedup_seg_start4`/`dedup_emit4` +
  `dedup4_core`) → `gpu_build_branch_resident` (BRANCH); per-row `build_shift_ops` →
  `gpu_build_shift_resident` (SHIFT). Test `gpu_dedup2_resident_all` → all multiset-identical to
  host path: MUL 130,676 / DVRM 4 / SHIFT 555,827 / BRANCH 47,191 unique rows, on box.
  **Full resident chip roster (10/10, all device→device→matrix, validated):** per-row CPU32/
  LOAD/STORE/SHIFT; single-mult LT/EQ/BYTEWISE; dual-mult MUL/DVRM; 4-key BRANCH. Generic
  helpers: `resident_alu_dedup_chip` (single-mult), `resident_alu_dedup2_chip` (dual-mult).
  **Remaining for the WHOLE pipeline:** (1) derived-source merges into the full tables
  (LT-from-dvrm/memw, MUL/SHIFT/DVRM-from-cpu32, MUL-from-dvrm) — concat device op-streams
  before dedup; (2) MEMW row routing from the walk; (3) wire the resident chains into
  `trace_builder` p5 behind a flag (single full-prove e2e); (4) Phase 6 precompiles; (5) bitwise.
- ✅ **MULTI-SOURCE MERGE validated** — `gpu_build_lt_instr_dvrm_resident` merges two op sources
  on device (instruction-driven LT ⊕ **dvrm-derived** LT = `LtOperation::new(abs_r, abs_d,
  false)` per is_divrem cycle, with abs_r/abs_d computed on device via `dvrm_lt_key_gather` →
  `cpu32_dvrm`), by gathering both key streams into ONE buffer (dvrm appended after instruction)
  and running a single `dedup3_core`. Test `gpu_lt_instr_dvrm_resident_matches_host_path` →
  154,095 unique rows, multiset-identical to the host instruction+dvrm merge, on box. This is the
  general recipe for completing every production chip table from its multiple sources: derive each
  source's key stream on device, concat, dedup once. (Fixed `lt_key_gather` k0 to the CANONICAL
  discriminator `signed|invert<<1` — not the raw alu_flags byte — so cross-source merge is
  correct; LT/EQ/BYTEWISE resident tests still pass.)
  ✅ Extended to a DUAL-mult chip: `gpu_build_mul_instr_dvrm_resident` merges instruction MUL ⊕
  dvrm-derived MUL (`mul_dvrm_key_gather`: each is_divrem cycle → `MulOperation::new(d, d_signed,
  q, q_signed)` contributing to both mu_lo & mu_hi, 2 entries/cycle) via `dedup3_core2`. Test →
  130,679 unique rows, multiset-identical to host. **Merge mechanism proven for single-mult AND
  dual-mult.**
  ✅ Also proven for a PER-ROW chip + a cpu32-derived source: `gpu_build_shift_full_resident`
  merges instruction SHIFT (word=0) ⊕ cpu32-derived SHIFT (`cpu32_shift_route` +
  `cpu32_shift_ops`, reusing compute_aux; word=1). Test `gpu_shift_full_resident` → 557,683 rows
  (555,827 + 1,856), multiset-identical to host. **Merge recipe now validated across all 3 chip
  patterns (single/dual/per-row) × both derived-source types (dvrm, cpu32).**
  ✅ **First COMPLETE production chip table on GPU: DVRM** — `gpu_build_dvrm_full_resident` merges
  DVRM's only two sources (instruction ⊕ cpu32-derived, via `cpu32_dvrm_route`/`cpu32_dvrm_ops`)
  → the full production DVRM table, validated multiset-identical to host (4 unique rows). Also
  `gpu_build_mul_full_resident` merges 3 MUL sources (instruction ⊕ instruction-dvrm-derived ⊕
  cpu32), validated (130,679 rows) — honestly still a subset (missing the dvrm→mul C13/C14
  contribution from *cpu32-derived* dvrm ops, an intertwined 4th source). **3-way merge proven.**
  Remaining to complete each full table: MUL (+ cpu32-dvrm→mul), LT/bitwise (+ memw-derived,
  gated on MEMW routing).
- ✅ **MEMW routing — GPU-VALIDATED (2026-07-17, box 42011)**: parity OK over 917,482 LOAD/STORE
  ops / 5,999,929 byte-accesses; classify 901,268 aligned + 16,214 general, matching the reference.
  `memw_gather` kernel
  (trace_walk.cu) scatters the per-byte walk `(old_ts, old_value)` into per-op MEMW rows via an
  `(op_row, byte_off)` mapping; `gpu_mem_walk_memw` (trace_walk.rs) = walk + gather →
  `(old_ts_per_op, old_value_per_op)` [num_ops*8]. Test `gpu_memw_routing_matches_reference`
  (prover) emits the ethrex LOAD/STORE byte-accesses (image-seeded), walks+gathers on device, and
  checks per-op `old_ts[0..width]`/`old_value[0..width]` + the aligned/general classification
  against a sequential HashMap reference. Compiles clean (cuda+non-cuda). **BLOCKED on GPU
  validation: both vast.ai boxes (21060 @201.165.125.8, 41143 @81.167.235.66) are `connection
  refused` (stopped/expired).** Resume: bring up a GPU box (driver ≥ CUDA 13.0), reinstall
  cuda-nvcc-13-0, `CUDA_HOME=/usr/local/cuda-13.0`, sync the files, run the test. Follow-ons
  after validation: LOAD old_value=own-value override + width-8 old_ts, device-side byte-access
  expansion (general prefix-sum) for a fully-resident MEMW build, then MEMW_A/MEMW fills.
- ✅ **MEMW TABLE FILL — GPU-VALIDATED (2026-07-17, box 42011)**: `memw_classify` + `memw_pack`
  kernels (trace_walk.cu) + `gpu_build_memw_ls` (trace_walk.rs) assemble the MEMW_A (stride 12)
  and MEMW (stride 19) packed rows on device — walk → gather → classify aligned/general →
  compact each bucket (excl_scan) → pack value[8]/old[8]/old_ts (LOAD own-value + sign-ext;
  STORE all-8 value + walk old). Test `gpu_memw_fill_matches_reference` → 917,482 LOAD/STORE ops
  → **901,268 MEMW_A + 16,214 MEMW rows, packed rows byte-identical to reference**. Unconstrained
  [width,8) old positions = 0 on both sides (valid trace; production's read-8 there is zero-
  multiplicity, doesn't affect the proof; constrained [0,width) validated here + by gpu_memw_routing).
  → **The LOAD/STORE memory side is fully on GPU: walk → routing → MEMW_A/MEMW tables.** Remaining
  MEMW: ecall-generated memory accesses (Phase 6), device-side byte-access expansion (host-prepped
  in the test now) for full residency.
- ✅ **PIPELINE INTEGRATION — first resident chip wired into production p5 (2026-07-17, box 42011)**:
  new flag `LAMBDA_VM_GPU_RESIDENT_CHIPS=1` (`gpu_trace::gpu_resident_chips_enabled`). p5's CPU32
  dispatch (trace_builder.rs) now, under the flag, calls `build_cpu32_resident_tables(cpu_ops_ref,
  max_rows.cpu32)` — which feeds the resident device chain (`gpu_build_cpu32_resident_dev`, returns
  the filled CudaSlice) straight from the resident `cpu_ops` (no host CPU32-op build/pack) and wraps
  it in a `TraceTable` via `set_main_input_dev` — falling through to the host-op device path / CPU on
  None. Test `gpu_cpu32_pipeline_resident_matches_host_op_path` → the p5 resident CPU32 table is
  **byte-identical to `gpu_build_cpu32_tables`** (3938 rows, 155,648 cells), on box.
  **Integration pattern proven.** Extending to the other complete chips (SHIFT/EQ/BYTEWISE/BRANCH/
  DVRM/LOAD/STORE) is mechanical (device-buffer resident builder + `build_*_resident_tables` wrapper
  + p5 flag dispatch). Caveats: single-chunk only (returns None if rows>max_rows → CPU fallback);
  full residency still uploads the cpu_op SoA per chip (host cpu_ops not yet device-resident); a
  *working full-GPU ethrex proof* remains gated on the incomplete tables (MUL 4th source, LT-from-
  memw, MEMW ecall, bitwise) + Phase-6 precompiles.
  Refactored the deduped helper to a device-buffer core `resident_alu_dedup_chip_dev` (+ `dtoh`
  download wrapper) and added `gpu_build_{eq,bytewise}_resident_dev` — ready for their p5 wrappers.
  Re-validated: 11 resident + pipeline tests all pass after the refactor (incl. gpu_cpu32_pipeline,
  gpu_eq_resident, gpu_bytewise_resident). **Next mechanical steps:** per-row LOAD/STORE/SHIFT p5
  wrappers (byte-exact, fixed height = host-known count.next_pow2, direct CPU32 analogs); deduped
  EQ/BYTEWISE/DVRM p5 wrappers need dynamic table height (unique count is device-computed) +
  multiset (not byte-exact) validation since resident emits sorted vs the host HashMap order (both
  valid — LogUp bus is order-independent).
- ✅ **LOAD + STORE wired into production p5 (2026-07-17, box 42011)**: `gpu_build_{load,store}_
  resident_dev` (device-buffer variants) + `build_{load,store}_resident_tables` (gpu_trace.rs) +
  p5 flag dispatch. Test `gpu_load_store_pipeline_resident_matches_host_op_path` → both
  **byte-identical to `gpu_build_{load,store}_tables`** (LOAD 492,350 rows, STORE 425,132 rows).
  → 3 chips run resident in production p5 (CPU32, LOAD, STORE), all byte-exact validated.
- ✅ **SHIFT wired into production p5 (2026-07-17)**: `gpu_build_shift_full_resident_dev` +
  `build_shift_resident_tables` (merges instruction + cpu32 shifts on device) + p5 flag dispatch.
  Test `gpu_shift_pipeline_resident_matches_host_op_path` → byte-identical to `gpu_build_shift_
  tables` (557,683 rows). → **ALL 4 per-row chips now run resident in production p5 (CPU32, LOAD,
  STORE, SHIFT)**, byte-exact validated. Per-row pipeline integration COMPLETE.
- ✅ **EQ + BYTEWISE wired into production p5 (2026-07-17)** — the deduped-chip pattern. Refactored
  `resident_alu_dedup_chip_dev` to **auto-size** the table from the device unique count and return
  `(buffer, num_rows)`; `gpu_build_{eq,bytewise}_resident_dev` → `(CudaSlice, num_rows)`;
  `build_{eq,bytewise}_resident_tables` (gpu_trace.rs; single-chunk guard: `None` if raw op count
  > max_rows, since global dedup can't match the CPU's per-chunk dedup) + p5 flag dispatch. Test
  `gpu_eq_bytewise_pipeline_resident_matches_host_op_path` → **multiset-identical** to
  `gpu_build_{eq,bytewise}_tables` (EQ 107,065 / BYTEWISE 467,791 raw ops). (Resident emits sorted
  vs host HashMap order — both valid, LogUp order-independent; a permuted table still proves.)
  → **6 chips now run resident in production p5: CPU32/LOAD/STORE/SHIFT (byte-exact) + EQ/BYTEWISE
  (multiset).** Both integration patterns proven.
- ✅ **DVRM + BRANCH wired into production p5 (2026-07-17)** — the last complete deduped chips.
  Auto-size `_dev` variants: `gpu_build_dvrm_full_resident_dev` (dual-mult, instruction⊕cpu32),
  `gpu_build_branch_resident_dev` (dedup4); `build_{dvrm,branch}_resident_tables` (gpu_trace.rs,
  single-chunk guard) + p5 flag dispatch. Test `gpu_dvrm_branch_pipeline_resident_matches_host_op_
  path` → multiset-identical (DVRM 4, BRANCH 257,535 raw ops). → **8 of 10 chips now run resident
  in production p5: CPU32/LOAD/STORE/SHIFT (byte-exact) + EQ/BYTEWISE/DVRM/BRANCH (multiset).**
  **Every chip whose table is complete-on-device is now wired.**
- ✅ **MUL completed + wired into production p5 (2026-07-17)** — added the 4th source
  (`cpu32-dvrm→mul`, kernel `cpu32_dvrm_mul_key_gather`). `gpu_build_mul_full_resident_dev` now
  merges ALL FOUR sources (instruction ⊕ instruction-dvrm→mul ⊕ cpu32 ⊕ cpu32-dvrm→mul), auto-
  sized; `build_mul_resident_tables` + p5 flag dispatch. Tests: `gpu_mul_pipeline` → multiset-
  identical to `gpu_build_mul_tables` (330,701 raw mul ops); `gpu_dvrm_full_and_mul_resident` →
  MUL FULL 130,679 unique rows. → **9 of 10 chips now run resident in production p5** (CPU32/LOAD/
  STORE/SHIFT byte-exact + EQ/BYTEWISE/DVRM/BRANCH/MUL multiset). Every chip except LT is wired,
  and MUL is a complete production table.
  **Remaining chip: LT — genuinely GATED on Phase 6 (finding 2026-07-17).** LT's full source set
  is instruction ⊕ dvrm→lt(ALL dvrm ops incl cpu32) ⊕ **LT-from-memw** (`collect_lt_from_memw` +
  `_aligned`: one `LtOperation::new(old_ts[i], ts, false)` per constrained byte position of every
  general memw op, and one per aligned memw op). Those memw buckets (`memw_ops`/`memw_aligned_ops`
  in p5, line 3470) include the **ecall-generated** memory ops (COMMIT/KECCAK/ECSM) — so a correct
  LT table requires the complete memw stream, which needs the Phase-6 ecall memory ops. Wiring a
  partial LT (LOAD/STORE memw only) would produce an INCORRECT table → invalid proof, so it is NOT
  wired. LT and the full-GPU proof share the same gate. Buildable component (deferred): a device
  memw→lt derivation from the `gpu_build_memw_ls` packed rows (aligned = 1 LT/row; general = width
  LTs/row via an 8-slot valid-mask + `excl_scan` compaction) + cpu32-dvrm→lt — completes LT once
  the ecall memw ops exist. **Chip pipeline integration: 9/10 wired; LT awaits Phase 6.**
- ✅ **Phase 2 core** — device memory memory-model walk: `radix_iota` / `radix_seg_hist` /
  `radix_seg_scatter` (+ reused `walk_seg_offsets`) / `mem_link` in trace_walk.cu — a stable
  8-pass LSD radix sort of a permutation by 64-bit byte address, then a predecessor link keyed
  on address change. `gpu_mem_walk` (trace_walk.rs). Test `gpu_mem_walk_matches_reference` →
  parity OK over **5,999,929 byte-accesses / 294,899 distinct addresses** (ethrex_5tx LOAD+STORE
  stream) vs a sequential HashMap read-old/write-new reference, on RTX 5090. The hard algorithm
  (sparse 64-bit-key sort) is proven. ✅ **Initial-image seeding validated** too: the test now
  seeds `init_value[i]` from the real `build_initial_image` (image size 2,804,965 bytes) and the
  reference pre-seeds the same at ts 0 → parity holds, 7,503 first-accesses correctly read a
  nonzero image byte. **Remaining Phase 2:** ecall (COMMIT/KECCAK/ECSM) memory accesses in the
  stream (Phase 6 seam), and routing walked accesses → MEMW_A/MEMW rows + the LOAD fill.

## Migration phases (by dependency)

### Phase 0 — The device data seam (foundation; do FIRST)
Nothing downstream is truly resident until the op stream is on-device.
- Upload `logs` + decoded `instructions` map to device once.
- Kernel: build the **`CpuOperation` SoA on device** (one thread per cycle: instruction decode
  fields + per-cycle register/memory values → the op record). This is `collect_cpu_ops` on GPU.
- Output: resident CpuOperation SoA that every later stage reads. Removes the per-stage host
  re-packing that the current GPU fills still pay.
- Decode itself (`instructions_from_elf`) can stay host (one-time, tiny) — just upload the map.

### Phase 1 — Register walk → default (mostly DONE)
Flip the existing device register walk on in the resident path (consume the resident op
stream instead of host emit). Byte-parity already proven.

### Phase 2 — Memory walk on device (radix sort)
Byte-granular read-old/write-new over the 64-bit address space → MEMW / MEMW_A rows resident.
New primitive: a **device stable radix sort** over the byte address (details in
`GPU-MEMORY-WALK-SCOPE.md`), then the predecessor-link + route/fill machinery reused from the
register walk. Hardest of the "core" stages.

### Phase 3 — Chip-op generation on device
The per-op inputs to LOAD/STORE/LT/SHIFT/MUL/DVRM/EQ/BYTEWISE/BRANCH/CPU32 (built today by
`collect_*`/`build_cpu32_op`/`LtOperation::new`/… on CPU) generated on-device from the resident
op stream — one small kernel per chip. Then the existing 14 fills consume **device** op records
(no host packing). Mostly mechanical field-extraction kernels.

### Phase 4 — Bitwise histogram: all sources on device
Extend the histogram (in_walk + memw_reg done) with the remaining sources — page, lt, mul,
dvrm, shift, branch, eq, bytewise, store, cpu32, memw_aligned, padding — each a device key-gen
from the resident op records/tables, scattered into the one device histogram → fill the BITWISE
MU columns **on device** (no 80 MiB download).

### Phase 5 — Preprocessed table fills on device
PAGE, DECODE, REGISTER, GLOBAL_MEMORY, KECCAK_RC filled on device from the resident memory
image / final-state / decode data. Per-cell fills (PAGE-like) — straightforward kernels; their
LDE is already on GPU, so this closes those seams.

### Phase 6 — Precompiles on device (hardest, needed for ethrex)
COMMIT / KECCAK / ECSM / ECDAS do register+memory I/O threaded through state plus complex table
generation. On device this means: (a) emit their register/memory accesses into the walk streams
(values computed on device — keccak permutation, EC ops), and (b) device fills for the KECCAK /
ECSM / ECDAS / COMMIT / HALT tables. Largest, most involved chunk; can be staged per precompile.

## Cross-cutting

- **Validation:** byte-parity per stage (device output == current CPU output, as done for PAGE
  and the register walk) + e2e prove/verify on **ethrex_5tx** after each phase.
- **Gating:** each stage behind a flag during bring-up; the resident path becomes default only
  once all stages are byte-parity-clean, so partial states never ship.
- **Order:** Phase 0 first (unblocks everything), then 1/3/4/5 (independent, incremental), then
  2 (radix sort) and 6 (precompiles) as the two hard long-poles.
- **Guest program:** always ethrex_5tx for validation/measurement, never fib.

## What "done" means

`Traces::from_elf_and_logs` performs exactly one host→device transfer (logs + instructions) and
returns resident trace-table matrices, with every fill / walk / histogram / op-derivation having
run on the GPU. No `collect_*` / histogram / fill executes on the CPU in the default path.

See [[gpu-tracegen-v2-progress]] for the current gated code + prior byte-parity harnesses.
