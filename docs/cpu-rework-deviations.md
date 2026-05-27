# CPU rework (`shrink-cpu`) — deviations from the spec

This document records where the implementation of the `spec/shrink-cpu` CPU
rework in this repository intentionally diverges from the spec
(`~/Documents/specs_vm`, branch `spec/shrink-cpu`), and why. It mirrors the
intent of the keccak spec-deviation notes: make every divergence explicit and
reviewable.

The rework shrinks the CPU table (~76 → ~39 columns), introduces a new dense
`packed_decode`, collapses the bitwise ops onto a single `BYTE_ALU` lookup, and
unifies the per-chip ALU buses (`Lt`/`Mul`/`Dvrm`/`Shift`) and the load/store
path onto two buses: `ALU[out; in1, in2, flags]` and
`MEMORY[out; timestamp, address, value, flags]`. Word (`*W`) instructions are
delegated to a dedicated `CPU32` table.

## Spec-author questions (resolved with deviations)

### Q3 — conditional branches set `BRANCH = 1 ∧ ALU = 1`
`decode.typ` does not set `BRANCH` on `BEQ/BNE/BLT/BGE`, but the `arg2`
multiplex (`cpu.toml`) only routes `arg2 = rv2` when `BRANCH ∧ ¬JALR`. Without
`BRANCH` the comparison operand would be wrong. We therefore decode conditional
branches with both `BRANCH = 1` and `ALU = 1` (the EQ/LT comparison is dispatched
on the `ALU` bus; `branch_cond = BRANCH·(JALR + (1−JALR)·res[0])`). Reported to
the spec authors.

### Q7 — `STORE` `MEMORY` flags include the `memory_op` bit
`store.toml`'s `MEMORY` receiver omitted the `memory_op` bit from its flags, so
the bus could not balance against the CPU sender (which sets `memory_op = 1` for
stores). The STORE chip reconstructs `mem_flags = 1 + 4·write2 + 8·write4 +
16·write8` (the `+1` is `memory_op`).

### Q9 — `CPU32` register reads cast `DWordWHH → DWordWL`
CPU32 stores `rv1`/`rv2` as `DWordWHH` but the register `MEMW` accesses use
`DWordWL`; the chip casts the halves to words on the bus
(`lo = rv[0] + 2¹⁶·rv[1]`, `hi = rv[2]`), matching the main CPU's register
access fingerprint.

## Implementation deviations

### D-PAD — `non_padding` column for the inline-PC multiplicity
`cpu.toml` gives the inline-PC `memory` token interactions a constant
multiplicity of `1`, which assumes padding rows also chain the PC cell. Our
memory argument instead leaves the final PC write as the terminal state and
requires padding rows to be inert. We add a `non_padding` (Bit) CPU column,
`1` on real instruction rows and `0` on padding, and use it as the inline-PC
multiplicity. Soundness is preserved by the memory bus itself (a real row that
sets `non_padding = 0` fails to consume its predecessor's PC token → imbalance;
a padding row that sets `non_padding = 1` emits tokens at the unreachable
`pc = 1` → imbalance).

### `memw`/`memw_aligned` keep the dedicated `LT`/`MUL` buses
The spec unifies *every* less-than / multiply lookup onto the `ALU` bus
(`memw` sends `ALU[old_ts, ts, opsel(LT)]`, `dvrm` sends `ALU` for its internal
`d·q`). We instead keep the existing `Lt`/`Mul` buses for those internal
consumers and make the `LT`/`MUL` chips **dual-receive**: they keep their `Lt`
/`Mul` receivers (for `memw`/`memw_aligned`/`dvrm`) and add an `ALU` receiver
(for the CPU/CPU32 dispatch), with separate multiplicity columns. This avoids
reworking `memw`/`memw_aligned` and is bus-equivalent; it costs the `LT`/`MUL`
chips one extra multiplicity column each (`MU_ALU`, `MU_ALU_LO`/`MU_ALU_HI`).
The `LT` chip also gains an `invert` column + `out = lt ⊕ invert` for `BGE[U]`.

### SHIFT input widening
The spec widens the SHIFT `shift` input to `DWordWHBB`. We keep the existing
effective-shift byte `SHIFT_AMOUNT` (used by the constrained computation, which
reduces mod 32/64) and add `SHIFT_B1` (byte) + `SHIFT_H1/H2/H3` (halves) holding
the rest of the full `arg2`, so the `ALU` receiver can present
`in2 = arg2`. These are range-checked (`ARE_BYTES`/`IS_HALF`), which makes the
decomposition unique and forces `SHIFT_AMOUNT = arg2 & 0xFF`.

### `BYTE_ALU` as three receivers
`bitwise.typ` notes a single `2²⁰` column as an optimization. We implement the
`BYTE_ALU[opsel, X, Y] → out` lookup as three receivers (one per `opsel`
`AND`/`OR`/`XOR`) reusing the precomputed `AND`/`OR`/`XOR` columns, with three
`MU_BYTE_ALU_*` multiplicity columns. `NUM_PRECOMPUTED_COLS` is unchanged so the
preprocessed commitment root is byte-identical.

### Retained-but-unused `BusId`s
`BusId::Shift`, `BusId::Load`, and `BusId::Dvrm` have no remaining senders or
receivers (their chips were migrated to `Alu`/`MemoryOp`). They are kept as enum
variants for now: `From<BusId> for u64` uses `id as u64`, so deleting middle
variants would renumber all later bus IDs and desync the explicit `TryFrom`
arms. They cause no dead-code warnings (still matched in `name()`/`TryFrom`).

### Disabled old-design test modules
`prover/src/tests/{decode_tests, cpu_tests, constraints_tests}.rs` test the
pre-rework 76-column / one-hot layout and are disabled (`#[cfg(any())]`) pending
a rewrite against the new layout. Coverage of the new design is currently
provided by `decode_layout_tests`, the per-chip tests, and the `prove_elfs`
end-to-end prove/verify tests.
