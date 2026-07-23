# Resident Trace-Gen Pipeline — plan for the remaining work

## Where we are (done)
- Per-step data (Phase 0) on GPU.
- **All trace TABLES on GPU + verified**: 9 resident chips, memory tables, all precompiles
  (commit / keccak / keccak_rnd / ecdsa / ecsm). Full prove+verify passes with everything on.
- Building blocks that EXIST + are validated but not wired as the default path: device memory walk
  (radix sort + predecessor link), device register walk, resident cpu_ops (`DeviceCpuOpsResident`),
  chip op-routing on device, histogram kernels (in_walk / memw_reg / page), MEMW fills.

## What's left = the walk / collect / histogram MACHINERY (not tables)
All of it is already correct on CPU. Moving it to GPU is a **speed** project, and every *partial*
move has lost (measured 5×) to upload cost + CPU-pool imbalance. It only wins as ONE resident
pipeline: **data born and kept on GPU (no uploads), CPU histogram left empty.** This is that plan.

## Guiding constraints (learned from measurements)
1. **No uploads** in the moved phases — data must be device-resident end to end.
2. **Empty the CPU** in the moved phases — no leftover unbalanced remainder.
3. Validate each phase byte/multiset-identical **and** e2e prove+verify.
4. Stay behind a flag until the final flip; keep `LAMBDA_VM_CPU_TRACE` kill-switch.

## Phases (ordered)

### P1 — Device access emission (foundation; biggest new piece)
Emit ALL register + memory accesses on device from the resident cpu_ops:
- **Register**: M1 rs1@ts, M3 rs2@ts+1, M5 rd@ts+2, implicit PC write — from the resident packed
  decode + rv1/rv2/rvd (all already on device).
- **Memory**: load/store per-byte accesses (addr, ts, value, is_read) from res/rvd/rv2 + mem_flags.
- **Ecall accesses** (keccak state r/w, commit buffer, EC I/O): a SMALL bounded set the executor
  already computes — upload it (few KB–MB). This is what resolves "the walk bails on ecall guests"
  WITHOUT needing EC crypto on device.
- Output: resident access streams (addr[], ts[], value[], is_read[]).
- Validate: emitted set == CPU `collect_register_ops` + `collect_load/store` + ecall memw (multiset).

### P2 — Wire the device walks (reuse existing, validated)
Point the existing device memory + register walks at P1's resident streams (instead of
CPU-collected + uploaded). Output: `old_value[]`, `old_ts[]` resident. Kills the walk uploads that
made the register walk a measured loss.
- Validate: reuse the walk parity tests, re-pointed at device inputs.

### P3 — Resident fills + memw_reg histogram from the walk
- MEMW_A / MEMW / MEMW_R / LOAD fills read P2's resident walk output (fills exist; re-point — no
  re-collect, no upload).
- memw_reg histogram source: scatter from the resident walk ts/old_ts — **no upload** (the exact
  cost that made it lose before is now gone).
- Validate: fills byte-identical; memw_reg histogram bin-identical.

### P4 — Full resident histogram (the p4 win)
Every feeder resident so the CPU histogram is EMPTY:
- in_walk (devops ✓), memw_reg (P3 ✓), page (device page data), op-vec (build the ~8 decomposition
  kernels: lt / mul / shift / branch / memw_aligned / eq / bytewise / cpu32 from resident routing).
- Leave only the ~8k tiny EC/precompile bumps on CPU (free — no imbalance).
- Validate: full histogram bin-identical; **MEASURE p4 — expect the first real win** (no uploads,
  empty CPU pool).

### P5 — Drop host `p2a_collect`
With chip tables + walks + histogram all reading resident data, `collect_ops_from_cpu` is redundant
for them. Remove it (keep only ecall-access prep). Reclaims ~592ms.
- Validate: e2e prove+verify; **MEASURE trace_build — expect the big drop.**

### P6 — Default + broaden
Flip the resident pipeline on by default (drop the opt-in gate; keep the CPU kill-switch). e2e across
more guests. Final measurement.

## Dependencies
P1 → P2 → P3 → P4 → P5 → P6. P4's op-vec decomposition kernels can be built in parallel with P1–P3.

## Measurement checkpoints (where the win should appear)
- **After P4**: p4 histogram time (first expected win).
- **After P5**: full trace_build (the big drop, once host collect is gone).

## Explicitly NOT needed
- No new modular / EC CUDA — the executor computes EC/keccak; trace-gen only formats (done).
- No new tables — all done.

## Honest scale + risk
This is the multi-week resident rearchitecture the earlier notes flagged. The biggest new piece is
**P1 (device access emission)**; P2 reuses validated walks; P4's op-vec fan-out is voluminous but
mechanical. The payoff is bounded: trace-gen is ~9% of the whole proof, so even a fully-resident
trace-gen moves total prove time by a small fraction — but it IS the only path that makes trace-gen
itself faster, and it completes "the whole process on GPU."
