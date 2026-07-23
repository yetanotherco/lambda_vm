# GPU resident trace-gen pipeline — scope

The honest next move for "all trace-gen on GPU". Grounded in this session's measurements;
leads with the cheap experiment that validates or kills the thesis before any large build.

## Why we're here (the measured finding)

Incrementally porting individual trace-gen phases to GPU **loses** to 8-core CPU:

| phase → GPU | result |
|---|---|
| register walk | loss (CPU walk ~8-40ms; GPU upload+kernels+download > that) |
| memory walk | shelved (CPU ~65ms; too cheap to beat) |
| bitwise histogram (naive atomics) | loss: p4 972ms → 1.36s |
| bitwise histogram (32× replicated) | loss: 1.37s (== naive → contention was NOT the cost) |
| **PAGE dense-read (CPU fix)** | **win ~10% — but not a GPU port** |

Root cause (proven by replication changing nothing): the loss is **structural, not kernel**.
Peeling one phase to GPU pays host-SoA-build + upload + a **sequential** kernel + 80MiB
download + 80MiB CPU merge, AND forfeits the CPU's 8-core parallelism over the whole
histogram. The overhead exceeds the compute saved.

## The thesis (UNPROVEN)

The overhead that sank per-phase porting is all **excursion cost** — data crossing the PCIe
boundary and syncing per phase. If the *entire* trace-gen ran device-resident (logs/cpu_ops
uploaded once → all derivations, histogram, fills, LDE/Merkle/FRI on device → download only
the final proof), each phase consumes VRAM-resident inputs with **no per-phase transfer, no
host SoA, no CPU/GPU ping-pong**. Then the histogram (and the walks) could win — the compute
is genuinely GPU-suited (fills + LDE + Merkle + FRI + constraint-eval already are). This is
plausible but NOT demonstrated. Partial residency does NOT help — we proved a single peeled
phase loses — so it is **all-or-nothing**, which makes the validation gate essential.

## Phase 0 — validate the thesis cheaply (GATE; do this first)

Before any rearchitecture, prove that *residency removes the overhead* on the one phase we
already have a kernel for — the bitwise histogram:
1. Upload the cpu-op fields **once** into a resident buffer (not per-call host SoA).
2. Generate keys + histogram on device (kernel exists: `bitwise_hist.cu`).
3. Keep the MU counts **on device**; skip the 80MiB download + CPU `add_raw_counts` merge —
   fill the BITWISE table's MU columns from the device counts directly (needs the BITWISE
   MU-column fill on device — small, or prototype by timing just steps 1-2 vs the CPU p4).
Measure vs the 972ms CPU p4. **If the resident histogram still loses, the thesis is false —
stop; per-phase and pipeline GPU trace-gen are both dead here.** If it wins big, the
rearchitecture is justified.

## Architecture (only if Phase 0 passes)

Resident dataflow, no CPU excursion between stages:
- **Stage A — cpu_ops on device.** Upload once (or generate from logs on device). ~4M ops ×
  compact fields.
- **Stage B — device op-derivation.** Per cpu_op → the per-table op records + the range-check
  lookup keys, on device. Big kernel set; the *fills* already exist (pt2, 14 tables resident)
  — this adds the op-derivation that currently happens in p1/p2a (~1.2s CPU).
- **Stage C — device histogram.** Consumes Stage-B keys resident; atomic scatter (replication
  if needed) → device MU counts. No download until the MU fill.
- **Stage D — device MU fills + tables.** BITWISE/preprocessed MU columns filled from device
  counts; per-op tables filled (pt2, done).
- **Stage E — LDE / Merkle / FRI / constraint-eval / logup-aux.** Already on device (#798 + the
  landed stack).
Net remaining new work = Stages A-D (cpu_ops residency, op-derivation, histogram, MU fills).

## Effort & risk (be honest)

- **Large** — Stage B (device op-derivation for all tables' range checks + op records) is the
  bulk; it re-implements p1/p2a/p4 on device. Weeks, not days.
- **All-or-nothing** — measured: partial residency doesn't pay. No incremental de-risking
  except the Phase-0 gate.
- **Payoff bound** — trace-build is ~3.5s (ethrex); if the resident pipeline halves it, that's
  the ceiling. Compare against the effort before committing.
- **Correctness** — every stage feeds the proof; each needs byte-parity like PAGE/walk did.

## Recommendation

Do **Phase 0 only** next (a few hours: resident-input histogram timing). It is the single
measurement that decides whether the multi-week rearchitecture is worth starting. Do NOT begin
Stages A-D until Phase 0 shows residency flips the histogram to a win. Meanwhile bank the PAGE
fix (landed, validated). See [[gpu-tracegen-v2-progress]] for the full scoreboard + gated code
(`LAMBDA_VM_GPU_BITWISE`, `LAMBDA_VM_CPU_REGISTERS`).
