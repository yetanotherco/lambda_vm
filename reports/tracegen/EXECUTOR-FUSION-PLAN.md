# Plan: enrich the executor so trace-gen doesn't rebuild memory

**Goal.** Delete the trace-gen memory-model walk (`p2a` / `collect_ops_from_cpu` — ~43% of
`trace_build`, ~1.0 s at 10-tx, bandwidth-bound). The executor already threads live memory +
registers as it runs; have it **emit each access's predecessor `(old_value, old_ts)`** so the
prover consumes it directly instead of re-deriving it by replaying the whole log.

**Constraint (user).** Executor and prover stay **separate components**. Only the executor's
**output contract** changes (the log / a new access stream). Redundant log data may be dropped.

## Why this is the right lever
- The walk is memory-bandwidth-bound (materializing ~20 M `MemwOperation` structs); parallelism
  can't fix it (measured — parallelizing p2b regressed).
- The executor has everything for free: `old_value` = the cell it's about to overwrite; the
  only missing piece is a **last-write timestamp per location**, and timestamps are *positional*
  (`ts = 4·cycle + sub_offset`), so the executor can compute them from its cycle counter
  (`logs.len()`).
- The walk even **recomputes** `keccak_f1600` and secp256k1 `compute_witness` that the executor
  already computed — recording precompile I/O removes that too.
- Execution is **40–50× cheaper** than trace_build (30–80 ms vs 1.5–3.3 s), so adding this work
  to the executor is nearly free relative to what it deletes.

## Key facts (grounded)
- Executor: `executor/src/vm/execution.rs` — loop at `:98`, `instruction.run(&mut pc, &mut
  registers, &mut memory)` (`:109`) returns a `Log`, pushed to `self.logs`. Cycle index =
  `logs.len()`. `CHUNK_SIZE=100_000` streaming; `run_epochs` for continuations.
- `Memory` = `U64HashMap<[u8;4]>` **word-granular, no timestamps** (`memory.rs:57`).
  `Registers([u64;31])` **no timestamps** (`registers.rs:8`).
- `Log` (40 B) = `{current_pc, next_pc, src1_val, src2_val, dst_val}` (`logs.rs:15`); ECALL
  repurposes the value fields to carry syscall addresses only (precompile I/O discarded).
- Trace-gen memory model is **byte-granular** `(val, ts)` per byte; register accesses are
  width-2 with a shared `old_ts`. Sub-timestamps: M1 read rs1 @ `ts+0`, M3 read rs2 @ `ts+1`,
  M5 write rd @ `ts+2`, PC write @ `ts+1` (`trace_builder.rs:collect_register_ops_from_cpu`).
- The prover already has `MemwBuckets` (from the walk-fusion change) that routes
  `MemwOperation`s into register/aligned/general buckets — so executor-emitted records drop
  straight into it.

## The interface change
Executor emits, alongside (a possibly slimmed) `Log` stream, a **memory-access record stream**
— one record per real access, in program+timestamp order:

```
struct MemAccessRecord {         // ~ the executor-side twin of MemwOperation
    kind: Reg | Mem | Pc,
    addr: u64,
    value: [u8; W] / packed,     // new value written (or value read)
    old_value: ...,              // predecessor value  (executor: the overwritten cell)
    old_ts: ...,                 // predecessor timestamp (from the ts shadow)
    timestamp: u64,              // 4·cycle + sub_offset
    width: u8, is_read: bool,
}
```
Prover side: `collect_ops_from_cpu`'s memory/register work is replaced by **routing these
records into `MemwBuckets`** (reuse E4). The rest of the walk (ALU dispatch, cpu bitwise) stays.

To produce it, the executor gains a **timestamp shadow**: `last_ts` per memory word (a second
`U64HashMap<u64>` or fold into the cell) and per register (`[u64; 32]` + PC + x254). At each
access: emit `old_ts = shadow[addr]` (0 if unset → genesis), then `shadow[addr] = current_ts`.

## Phased implementation (each phase parity-gated, walk kept as fallback for the rest)

**Phase 0 — timestamp shadow, validated in isolation.** Add the ts shadow to the executor;
compute `ts = 4·(cycle) + sub_offset` matching the prover's positional scheme. Don't change the
output yet. Add a debug/test that, for a run, the shadow's `(addr → last_ts)` matches what the
walk computes. *Deliverable: the executor can name every access's timestamp; zero prover change.*

**Phase 1 — REGISTER accesses (the bulk: M1/M3/M5, ~3.9 M+/1-tx).** Emit register access
records with `old_ts`/`old_value` from the register shadow (32 keys + PC + index — tiny,
cheap). Prover: a feature/flag path routes executor register records into `MemwBuckets.register`
instead of calling the register part of the walk; the walk still handles memory + precompiles.
*Biggest single win; smallest state (registers), so lowest risk.* Gate: byte-parity of
`memw_register_ops` vs the walk, then prove+verify.

**Phase 2 — MEMORY load/store accesses (byte-granular).** Emit per-byte (or per-word, matching
the AIR's memory-argument granularity) access records with `old_value`/`old_ts` from the memory
ts shadow. Prover routes them into `aligned`/`general` buckets. Careful: byte vs word
granularity, unaligned splits, `old_value` for stores. Gate: byte-parity of memw/memw_aligned +
load/store ops, then prove+verify.

**Phase 3 — PRECOMPILES (keccak / ecsm / commit).** Executor records the I/O it already
computes (keccak 25+25 lanes, ecsm operands+result, commit bytes) + their memory accesses.
Removes the walk's redundant `keccak_f1600` / `compute_witness` recompute and its live-memory
reads. Gate: byte-parity of keccak/ecsm/commit ops + prove+verify.

**Phase 4 — delete the walk.** With all accesses executor-emitted, `collect_ops_from_cpu` loses
its memory-model half; only ALU-chip dispatch + cpu bitwise remain (both cheap, order-free).
Remove the ts double-maintenance. Final prove+verify across the ELF suite.

## "Redundant data" opportunity (log slimming)
Once the executor emits access records, revisit the `Log`:
- Register values (`src1_val/src2_val/dst_val`) may be recoverable from the access records for
  some paths → candidate to drop or not duplicate.
- The access stream is larger than the log (3–4×/instr); stream it per `CHUNK_SIZE` epoch (the
  mechanism exists) to bound memory. Net bytes may still fall if the walk's transient
  materialization is removed.
- Keep only what the CPU table genuinely needs from the log; everything memory-consistency
  moves to the access stream.

## Testing contract
- **Per phase:** byte-parity of the affected op vectors (executor-emitted vs walk) on ethrex
  1/5/10-tx — deterministic, so exact equality — then **prove+verify** (the bus balances iff
  every access + predecessor is correct).
- Keep the walk behind `LAMBDA_VM_LEGACY_TRACEGEN` for A/B until Phase 4.
- Continuations: the ts shadow must reset/reseed per epoch (positional ts resets); test
  `run_epochs` parity.
- Perf gate: `count-elements` per-stage timing; expect `p2a` to collapse toward the ALU/bitwise
  residue as phases land.

## Risks
- **Granularity match** (byte vs word memory ts) must exactly mirror the AIR's memory argument —
  the #1 correctness trap.
- **Timestamp sub-offsets** (M1/M3/M5, PC, memory) must match the prover's scheme bit-for-bit.
- **Continuation epoch boundaries** (ts reset, carried register/memory snapshot).
- Executor memory grows by the ts shadow (~same size the walk's `MemoryState` used — net neutral,
  just moved upstream).

## Recommended start
**Phase 0 + Phase 1 (registers).** Registers are the bulk of the walk's volume yet the smallest,
simplest state (34 keys) — highest impact-per-risk, and it proves the whole approach end-to-end
before touching byte-granular memory or precompiles.
