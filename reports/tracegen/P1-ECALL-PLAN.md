# P1-ecall — feeding COMMIT/KECCAK/ECSM accesses into the device walks

**Status (2026-07-17):** designed + architecture fully mapped; implementation in contained,
GPU-validated increments. This is THE correctness gate for a full resident ethrex proof (P4b): the
resident register + memory walks currently omit ecall-generated accesses, so on ethrex (which has all
three ecalls) `memw_reg` / `memw_aligned` / `page-fini` / `lt-from-memw` would undercount → invalid proof.

## Why it's needed (not optional)

In the sequential path (`collect_ops_from_cpu`, trace_builder.rs ~585-700) the ecall collectors
`collect_{commit,keccak,ecsm}_memw_ops` thread `register_state`/`memory_state`. So an **ecall register
write advances the register cell timeline**: a later *regular* M1/M3/M5 read of that register gets the
ecall's ts as `old_ts`. The device register walk reconstructs `old_ts` from its access stream — if that
stream omits ecall accesses, the regular rows' `old_ts`/`old_value` are wrong. Same for memory
(keccak lanes / commit bytes / ecsm dwords vs regular LOAD/STORE). ⇒ ecall accesses MUST enter the
device walk streams as timeline events.

## Architecture map (verified 2026-07-17)

- **MemwOperation** (memw.rs:104-175): `{is_register, base_address, value[8], timestamp, width, is_read,
  old[8], old_timestamp[8]}`. `new(is_register, base_address, value, timestamp, width, is_read)` +
  `.with_old(old, old_ts)`.
- **MemwBuckets** (trace_builder.rs:403-432) = the collector sink: `register_rows: Vec<RegRow>` +
  `aligned/general: Vec<MemwOperation>`. Every op routes via `classify_memw` (369-387):
  `is_register_op` (1967: `is_register && width==2 && old_ts[0]==old_ts[1] && reg_ts_delta_in_range`)
  → **RegRow (MEMW_R)**; else `is_aligned_op` (1898: byte-aligned + shared old_ts) → **MEMW_A**; else
  **MEMW (general)**.
- **Two row-production paths feed the register table** (KEY subtlety):
  1. Regular M1/M3/M5: `emit_register_accesses`→`RegAccess`→`walk_register_accesses`→`RegRow` (always
     a register row; the device analog is `gpu_walk_route_memw_register_chunked`, which mirrors
     `is_register_op` and returns an aligned/general FALLBACK for out-of-range register accesses).
  2. Ecall register ops (x10/11/12/254): `MemwOperation` → `classify_memw` (MEMW_R **or** fallback).
- **Ecall accesses (verified):** COMMIT = x10(rw)/x11(r)/x12(r)/x254(rw) + `count` memory byte-reads
  (buf_addr+i). KECCAK = x10(r) + 25 lanes × width-8 memory rw (lane_addr = state_addr + 8*i);
  values from `keccak_f1600` run during collection. ECSM = x10/x11/x12 reads (t/t+1/t+2) + 12 width-8
  memory ops (4 xG read @t, 4 k read @t+1, 4 xR write @t+2); values EC-witness-derived.
- **Device register walk**: `emit_register_accesses_dev` (trace_walk.rs) emits (reg_addr, ts, value,
  is_read, row_index) on-device from cpu_op fields; `gpu_walk_fill_memw_register_resident` (trace_cpu.rs:982)
  chains emit→walk_core→memw_register_fill. NO ecalls today. `row_index<0` = non-emitting (PC write).
- **Device memory walk**: `gpu_build_memw_ls_resident` (trace_walk.rs:1392) emits LOAD/STORE
  byte-accesses + op-meta, `image_lookup` init, radix walk, gather, classify aligned/general, pack
  MEMW_A/MEMW. NO ecalls today.
- The resident walks are validated (gpu_reg_emit_parity) but **not yet wired into production
  build_traces** — that's P4b. So P1-ecall makes them *able to incorporate* ecall accesses.

## Design — "Option Z" (inject ecall accesses as timeline events; keep tiny ecall ROWS on CPU)

Ecall counts are tiny (keccak ~135, ecsm ~20, commit ~161 ops on ethrex_5tx), and their values need
the crypto (run on CPU during collection). So:

1. **Capture** ecall accesses during collection (they're already produced as `MemwOperation`s at
   trace_builder.rs 631/667/680). Record, in per-op emission order, raw access events:
   - register-addressed → `(reg_addr, ts, value_u64, is_read)`
   - memory-addressed → expand to `width` byte-accesses `(addr+b, ts, byte, is_read)` + op-meta.
2. **Inject** into the resident walk streams as **non-emitting timeline events** (`row_index=-1` for
   register; excluded from the "ops to emit rows for" set in memw_ls). They advance the walk timeline
   so REGULAR rows get correct `old_ts`/`old_value`, but the device does NOT emit their rows.
   Mechanism: upload the (small) ecall access arrays, concat after the device-emitted accesses BEFORE
   `walk_core` / the radix sort. **Ordering:** append per-op AFTER that op's regular accesses so the
   STABLE sort reproduces the sequential (regular-before-ecall) tie-break at equal (addr, ts).
3. **Ecall rows stay on CPU**: `classify_memw` on the ecall `MemwOperation`s (already carry correct
   `old` from state threading) → their MEMW_R/MEMW_A/MEMW rows + their (tiny) histogram bumps computed
   on CPU. This keeps the DEVICE walk correct (the hard part) while the negligible ecall rows ride the
   CPU (acceptable: they add to the "~8k EC bumps stay on CPU" residual for P4b's empty-CPU goal).

Why Option Z over "fully device ecall rows": (a) values need the crypto (CPU) anyway; (b) avoids
reconciling the dual register-row paths (RegAccess-walk vs MemwOperation-classify) on device; (c) the
device-walk *correctness* (regular rows accounting for the ecall timeline) is the only thing that
actually gates a valid proof, and Option Z delivers exactly that.

## ⚠️ CRITICAL FINDING (2026-07-17, GPU-verified) — the walk is INPUT-ORDER-dependent

The register walk's `walk_link` (trace_walk.cu) sets `old[perm[p]] = access at perm[p-1]`, where `perm`
comes from `walk_seg_scatter` — a **STABLE counting-sort by BIN that preserves INPUT-ARRAY ORDER**. It
does NOT sort by `ts` within a bin (`ts` is only stored as the old_ts VALUE). ⇒ the walk assumes the
input access array is already in timeline order per register. The emit produces accesses in op order
(= timeline order), so that holds for the regular stream.

CONSEQUENCE for injection: ecall accesses must be **INTERLEAVED at their op's timeline position**, NOT
appended. GPU-verified: appending 2000 synthetic accesses at the end of the array changed **0** of the
9.3M emitting rows (they landed last within each bin → never a predecessor). Test
`gpu_reg_walk_injected_matches_host` still confirms the CONCAT MECHANISM (device == host walk over the
same combined stream), but append-ordering is operationally wrong.

REVISED injection design (register walk): extend `emit_register_accesses_dev` to interleave the ecall
accesses during emit — add per-op ecall counts to the per-op access counts before the excl_scan, and
have `reg_access_scatter` write op i's [regular accesses, then ecall accesses] into op i's slot. Then
input order = timeline order within every bin (ops in order; regular-before-ecall within an op matches
the sequential `collect_register_ops_from_cpu`→ecall-collector order). Upload: ecall (op_index, reg_addr,
ts, value, is_read) — tiny. The append-based `gpu_walk_fill_memw_register_resident_injected` validated
the plumbing but is superseded by this interleaving emit. (Same input-order caveat likely applies to the
memory walk's radix path — verify whether radix_sort_perm tie-breaks by ts or input order.)

## KEY SIMPLIFICATION (found 2026-07-17 reading the walk primitives)

The device walk's **non-emitting timeline-event mechanism is ALREADY validated**: the per-instruction
implicit PC write is emitted with `row_index=-1` (`emit_register_accesses`, trace_builder.rs:1138 /
device `reg_access_scatter`) — it advances x255's timeline so the next read gets the right `old_ts`,
but emits no MEMW_R row, and `memw_register_fill` / `walk_link` already handle exactly that. Ecall
accesses injected as `row_index=-1` events are the SAME shape. So **no new walk semantics are needed**
— the walk already links non-emitting predecessors correctly (proven by `gpu_reg_walk_fill_resident`).

Consequences:
- Register injection into the HOST path is trivial: concat the extracted ecall `(addr,ts,value,is_read,
  row_index=-1)` onto the arrays fed to `gpu_walk_and_fill_memw_register_host` (which already takes raw
  access arrays). The resident (device-emit) path just needs a small dtod-concat
  (`memcpy_dtod`/`memcpy_htod` + `.slice`) after `emit_register_accesses_dev`, before `walk_core`.
- Memory injection: feed extra non-emitting byte-accesses to the walk in `gpu_build_memw_ls_resident`
  (they participate in the radix sort/link; their ops are excluded from the emit set).
- ⇒ **The real, novel, risky work is the CPU-side EXTRACTION** (which ecall accesses, in what order,
  with crypto-derived values) — NOT the walk plumbing. Validate the extraction by feeding
  [emit ⊕ extracted-ecall] to the (validated) `gpu_walk_and_fill_memw_register_host` and comparing to
  the sequential path's register table on ethrex_5tx.

## ⚠️ P4b input — register fallback rate (measured 2026-07-20, ethrex_5tx, CPU diagnostic)

`register_fallback_rate_ethrex` (walk_decomp_tests.rs): **202 of 9,339,363 emitting register accesses
(0.0022%) fall back** to MEMW_A/MEMW (reg_ts_delta > 2^16 — a register unaccessed for >~16k cycles).
The resident register walk (`gpu_walk_fill_memw_register_resident`) does NOT model this fallback (it
emits ALL emitting accesses as MEMW_R). So for an EXACTLY correct resident proof, P4b must reconcile
these 202 rows: cleanest is the same Option-Z pattern — the resident walk emits the 9.34M regular
MEMW_R rows, and the ~202 fallback rows (+ ecall rows) are added on the CPU side (they're already part
of the "small CPU remainder"). Tiny, tractable, but MUST be handled or the LogUp multiplicities mismatch.

## Increment plan (each GPU-validated on ethrex_5tx)

1. **Device register injection — ✅ DONE + validated via INTERLEAVING (2026-07-20, box 44902).**
   - Append primitive (`gpu_walk_fill_memw_register_resident_injected`, 2026-07-17) validated the concat
     plumbing but is superseded: appending gave 0-diff (the input-order finding).
   - CORRECT primitive: CPU capture tags each ecall access with its op index (`EcallAccesses.reg_op_index`,
     emits_row=false); three CUDA kernels (`reg_ecall_op_counts`, `reg_access_counts_ecall`,
     `reg_access_scatter_ecall` in trace_walk.cu) INTERLEAVE each op's ecall accesses right after its
     regular accesses + PC write during emit; wrappers `emit_register_accesses_with_ecall_dev`
     (trace_walk.rs) + `gpu_walk_fill_memw_register_resident_ecall[_host]` (trace_cpu.rs). Per-op ecall
     counts computed on device (scatter-add), only tiny ecall arrays uploaded.
   - Test `gpu_reg_walk_ecall_interleaved_matches_host` (gpu_reg_emit_parity.rs): 9.3M regular rows +
     2000 synthetic ecall accesses at real op positions on hot bins → device BYTE-IDENTICAL to host walk
     over the correctly-interleaved stream, AND 4358 cells differ vs no-injection (interleaved events
     re-link old_ts; append gave 0). Interleaving mechanism proven.
   REMAINING in this increment:
   - Wire the REAL captured `EcallAccesses` (from collect_ops_from_cpu) into
     `gpu_walk_fill_memw_register_resident_ecall`; validate the regular MEMW_R rows vs the SEQUENTIAL
     path on ethrex_5tx (handle Option-Z accounting: sequential emits ecall rows too + reg_ts_delta
     fallback — compare the regular-row subset, or the full multiset once ecall rows are added on CPU).
   - **Memory walk injection — ✅ DONE + GPU-validated (2026-07-20, box 44902).** Kernels
     `mem_ecall_byte_counts`, `memacc_counts_ecall`, `memacc_emit_ecall` (trace_walk.cu) + wrapper
     `gpu_build_memw_ls_resident_ecall` (trace_walk.rs) + capture extended to ecall MEMORY accesses
     (byte-expanded, `EcallAccesses.mem_*`). Dump-row (op_row=num_ops) approach: touches no existing
     validated kernel; out_old sized (num_ops+1)*8. Test `gpu_memw_ls_ecall_interleaved`
     (gpu_reg_emit_parity.rs): empty-ecall BYTE-IDENTICAL to base (901268 MEMW_A + 16214 MEMW), and 300
     interleaved ecall accesses re-link old_ts → aligned/general split shifts (224 ops flip
     aligned→general as bytes stop sharing old_ts) with TOTAL 917482 conserved (no spurious rows). →
     BOTH walk mechanisms (register + memory) can now incorporate interleaved ecall accesses.
   - (superseded note) VERIFIED the radix path has the SAME input-order dependence (trace_walk.cu comment: stable LSD
     radix by address, "within an address the ts order is preserved") → ecall memory accesses MUST be
     interleaved in timeline order too (append fails, same as registers).
     PRECISE DESIGN (dump-row, touches no existing validated kernel):
     1. Extend the CPU capture: also record ecall MEMORY accesses (the `!is_register` MemwOperations)
        as byte-level events `(addr = base+b, ts, byte = value[b], op_index)` for b in 0..width — in
        per-op order. Add to `EcallAccesses` (e.g. `mem_addr/mem_ts/mem_val/mem_op_index: Vec<..>`).
     2. Reserve per-op ecall byte slots: new `memacc_counts_ecall` adds `ecall_byte_cnt[i]` (per-op
        scatter-add from mem_op_index, like `reg_ecall_op_counts`) to `byte_cnt[i]` BEFORE the
        `byte_base` excl_scan. `memacc_emit` writes op i's regular bytes at `byte_base[i]` (unchanged,
        op_row=r/byte_off=j). A new `memacc_emit_ecall` writes op i's ecall bytes at
        `byte_base[i] + regular_width` with `op_row = num_ops` (DUMP row) + byte_off=0.
     3. Allocate `out_old_{ts,value}` as `(num_ops + 1) * 8`; `memw_gather` UNCHANGED (ecall bytes land
        in the dump row num_ops, which `memw_classify`/`memw_pack` ignore). The walk (`radix_sort_perm`
        + `mem_link`) now sees ecall bytes interleaved in timeline order → regular LOAD/STORE old_ts is
        correct across the ecall interleaving. init_value for ecall bytes via `image_lookup_dev` (same).
     4. Validate like the register test: device == host walk over the correctly-interleaved byte stream,
        AND differs from no-injection. Then wire real captured `EcallAccesses.mem*`.
     NOTE: ecall memory accesses use width up to 8 (keccak lanes are width-8; commit bytes width-1;
     ecsm dwords width-8). Their MEMW_A/MEMW rows stay on CPU (Option Z).
2. **CPU extraction (the crux):** capture the ecall accesses INSIDE `collect_ops_from_cpu` (it already
   threads state) — NOT a pure cpu_ops function, because ECSM register reads take their values from
   `register_state.read(10/11/12)` (the program-loaded addresses; trace_builder.rs:869-871) and the
   COMMIT x254 value needs the running commit index. The ecall `MemwOperation`s at lines 631/667/680
   ALREADY carry correct `(is_register, base_address, value[8], timestamp, width, is_read)` from the
   state threading, so the capture is a pure converter over them (no collector changes):
   - register op (is_register) → `RegAccess { reg_addr=base_address, ts, value=value[0]|value[1]<<32,
     is_read, emits_row=false }`  (Option Z: non-emitting timeline event).
   - memory op (!is_register) → `width` byte-accesses `(base_address+b, ts, value[b], is_read)`, emits_row=false.
   Add an `EcallAccesses { reg: Vec<RegAccess>, mem_addr/ts/val/is_read/width: Vec<..> }` as an extra
   return of `collect_ops_from_cpu` (3 call sites: 2488 `let _`, 5309, 5393). Validate: feed
   [emit_register_accesses ⊕ captured ecall reg] to `gpu_walk_and_fill_memw_register_host` and compare
   the regular rows to the sequential path (accounting for Option Z: sequential emits ecall rows too, so
   compare either the regular-row subset, or set ecall emits_row=true to reproduce the full table modulo
   the reg_ts_delta fallback).
3. **Wire + validate:** feed the captured ecall accesses into the resident walks; validate the
   resident MEMW_R + MEMW_A/MEMW rows (regular, device) + CPU ecall rows == the sequential path's
   full tables on ethrex_5tx.
4. Then P4b can assemble the resident histogram with complete walk-derived sources.

## Gotchas

- STABLE sort tie-break: ecall accesses must be appended AFTER the op's regular accesses (sequential
  order: LOAD/STORE → M1/M3/M5 → ecall). The radix sort is stable (handoff §6) → order preserved.
- Do ecall ops emit regular M1/M3/M5? An ECALL instruction's decode likely has
  read_register1/2/write_register = false (register I/O is via the collectors) — VERIFY before wiring
  (if true, the only regular access for an ecall op is the non-emitting PC write @ts+1).
- `reg_ts_delta_in_range` fallback: an ecall register access whose ts-delta exceeds 2^16 routes to
  MEMW_A/MEMW, not MEMW_R. Handled on CPU in Option Z (classify_memw), so no device concern.
- value_u64 for a register op = `MemwOperation.value` packed (word0 | word1<<32).
