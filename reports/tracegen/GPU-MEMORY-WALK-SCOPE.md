# GPU memory-model walk — scope

Retarget the validated register-walk machinery (PR-3) at the **general memory model**,
where the CPU walk is plausibly a real bottleneck (unlike registers, measured at ~8 ms →
net loss). Prover-only, no caching ([[no-tracegen-caching]]).

## Premise (why memory, not registers)

The register walk is CORRECT + LIVE but a measured ~+1 % net loss: the CPU register walk is
only ~8 ms (tiny keyspace, cheap single pass), so GPU launch+transfer overhead exceeds the
saving. Memory is the opposite bet: byte-granular accesses over a huge sparse address space,
potentially tens of millions of bytes on memory-heavy programs — where the CPU walk should
actually hurt. **But that is a hypothesis, not a fact.** Phase 0 below MUST confirm it before
any kernel work. Do not build GPU machinery for a cheap CPU operation twice.

## How the CPU memory model works today (verified)

- **State**: `MemoryState` = per-BYTE `(value, timestamp)` in a dense `PagedMem`
  (`trace_builder.rs:83`). `read_bytes(base, 8)` / `write_bytes` loop per byte.
- **Walk**: the collect loop threads `memory_state`. `collect_load_op_from_cpu` /
  `collect_store_op_from_cpu` do read-old (`read_bytes` → `old_values[8]`,
  `old_timestamps[8]`) then write-new — identical read-old/write-new semantics to the
  register walk ([[tracegen-walk-semantics]]), just byte-granular over 64-bit addresses.
- **Rows**: a `MemwOperation` carries `base_address` (64-bit), `value[8]`, `width` (1/2/4/8),
  per-byte `old[8]` + `old_timestamps[8]`, `is_register`, `is_read`. Routed by `classify_memw`
  into **MEMW (general, width-8)** vs **MEMW_A (aligned)** buckets (register fast-path is the
  third route, now on GPU).
- **Ordering**: NO global table sort. Timestamp ordering is per-row via `collect_lt_from_memw`
  / `collect_lt_from_memw_aligned` → **LT** lookups (old_ts < ts), plus IS_HALF/ARE_BYTES
  address range checks. Memory consistency is the permutation-invariant LogUp multiset. So —
  like registers — no sort of the TABLE is needed; the walk just recovers per-byte `old_*`.

## What's reusable vs new

Reusable from PR-3 (all validated): the **predecessor link** (`walk_link`), the
**route + 2-level exclusive scan + localize + resident chunked fill** seam, and the
**build_traces wiring pattern** (null-sink emit → device build → merge histogram/fallbacks →
resident tables → LDE, behind a kill-switch with CPU fallback).

New, memory-specific:
1. **Device stable group-by over 64-bit byte addresses.** Registers used counting-sort
   (≤512 buckets); memory needs a **stable radix sort** over the address (LSD, ~8×8-bit or
   4×16-bit passes reusing the existing `excl_scan`), keeping ts order within an address so the
   predecessor link works unchanged. This is the one genuinely hard new kernel.
2. **Per-byte emit + per-byte→row aggregation.** Emit one byte-access per accessed byte
   (`width` per op); walk per byte; gather the 8 per-byte `old_*` back into each MEMW row.
   Volume ≈ total bytes accessed (bigger than register accesses).
3. **Dual output + LT/range lookups.** Rows split MEMW vs MEMW_A (route by `classify_memw`);
   the walk output also feeds **LT ops** (timestamp ordering) and **BITWISE** address range
   checks — more downstream consumers than registers (which only fed IS_HALF).
4. **Precompile memory I/O.** COMMIT/KECCAK/ECSM read+write `memory_state` inline, so — as with
   registers — the device walk is only clean for programs whose memory traffic is all
   load/store. MORE precompiles touch memory than registers, so the precompile-free gate is
   narrower; emitting precompile memory accesses (values host-computed) is a bigger follow-up.

## Phases

- **Phase 0 — MEASURE (gate).** Add dedicated `instruments` spans around the memory collect
  (load/store → memory_state) and MEMW/MEMW_A generation. Prove `memory.elf` (pure memory,
  likely precompile-free) and `ethrex.elf` with `--features cuda,instruments`; read the memory
  walk vs fill split and MEMW/MEMW_A row counts. **Proceed only if the walk is a real
  bottleneck (100s of ms), not ~8 ms.** If cheap, stop — keep memory on CPU (register lesson).
- **Phase 1 — Device radix sort.** Stable LSD radix over the 64-bit byte address (composite
  reuse of `excl_scan`), standalone parity test vs a CPU sort. The load-bearing new primitive.
- **Phase 2 — Byte-granular walk → MEMW rows.** Per-byte emit → radix group-by → `walk_link` →
  aggregate 8 per-byte `old_*` into rows. Byte-parity vs the CPU `collect_load/store` output.
- **Phase 3 — Route + fill + lookups.** Split MEMW/MEMW_A (device route+compact+chunked fill,
  reusing PR-3), produce LT ops / range-check multiplicities. Byte-parity vs CPU tables.
- **Phase 4 — Wire + e2e.** `build_traces` behind `LAMBDA_VM_CPU_MEMORY` (mirrors
  `LAMBDA_VM_CPU_REGISTERS`), memory-precompile-free gate, CPU fallback; prove+verify A/B on
  `memory.elf`; measure the win.

## Risks

- **Radix sort correctness + cost** — the hard part; a slow/incorrect sort sinks the win.
- **VRAM** — byte-granular over millions of ops × up to 8 bytes; may need chunked/streamed sort.
- **Precompile memory emission** — needed before ethrex (precompile-heavy) benefits; big.
- **Repeating the register outcome** — Phase 0 is the guard. Measure first.

Foundation: register walk in [[gpu-tracegen-v2-progress]] (kernels, `build_traces` wiring,
`LAMBDA_VM_CPU_REGISTERS` switch). Same box workflow.
