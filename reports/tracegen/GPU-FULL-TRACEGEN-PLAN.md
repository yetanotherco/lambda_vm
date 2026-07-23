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
