# formal-riscv: the Clean model against the Sail RISC-V spec

Proves, instruction by instruction, that what the AIR's Clean model (`../formal`, PR #1027)
computes is what the official RISC-V spec says. The spec is the Sail model
([sail-riscv](https://github.com/riscv/sail-riscv)), translated to Lean by Sail's Lean
backend and published as
[`opencompl/sail-riscv-lean`](https://github.com/opencompl/sail-riscv-lean).
The approach follows [zisk-fv](https://github.com/eth-act/zisk-fv) (`ZiskFv/SailSpec/`).

## Build

```bash
cd formal-riscv
lake exe cache get   # Mathlib oleans
lake build
```

The first build compiles Clean, the Clean model and the Sail model (the largest Sail module,
`InstsEnd`, takes minutes).

`../formal` is a path dependency, but this project has its own `lake-manifest.json`: when
`formal/` updates Mathlib or Clean, run `lake update` here too so both use the same versions.
This folder is not built by CI (`.github/workflows/formal.yml` only covers `formal/**`).

## Structure

Each instruction is three steps:

```
Sail execute_RTYPE … SLL ──(1)── sllPure ──(2)── Alu.shiftResult ──(3)── ALU bus message
        (state)                 (values)          (Clean model)          (what the CPU learns)
```

1. **Sail** (`RiscvEquiv/Sll/Sail.lean`): the instruction's computation is taken out of Sail
   and written as a pure function, `sllPure`, with no state. `execute_SLL` proves that what
   Sail does to the registers is exactly: read `rs1` and `rs2`, write `sllPure x[rs1] x[rs2]`
   to `rd`.
2. **Bridge** (`RiscvEquiv/Sll/Bridge.lean`): Sail works with 64-bit vectors (`BitVec 64`), the
   Clean model with naturals. `sllPure_toNat` proves that `sllPure`, read as a natural with
   `.toNat`, is the Clean model's function at the flags of `SLL` (`Alu.shiftResult`; for `SLT`
   and `SLTU` it is `Alu.ltBit`).
3. **Alu** (`RiscvEquiv/Sll/Alu.lean`): `sll_alu_matches_sail` proves that if the CPU pulls an
   ALU bus message with the flags of `SLL`, its result is `sllPure a b`. It uses the bus
   guarantees (`Alu.Guarantees`) that PR #1027 already proves for each table.

In one line: Sail = `*Pure`, `*Pure` = Clean function, Clean function = what arrives on the bus.

`AluBus.lean` is shared: one lemma per clause of `Alu.Guarantees` (SHIFT, SHIFT with
`word = 1`, LT, EQ) that reads a message's result as a natural. Each `Alu.lean` instantiates
one of them at its flags, so it is one line per instruction.

The immediate versions have no `Bridge.lean`: they are the register version with the
immediate as the second operand, so they reuse its bridge and its `Alu` theorem.

Done so far:

| Instructions | Folder | Table |
|---|---|---|
| `SLL`, `SRL`, `SRA` | `Sll/`, `Srl/`, `Sra/` | SHIFT |
| `SLLI`, `SRLI`, `SRAI` | `ShiftImm/` | SHIFT |
| `SLLW`, `SRLW`, `SRAW`, `SLLIW`, `SRLIW`, `SRAIW` (low 32 bits) | `ShiftW/` | SHIFT |
| `SLT`, `SLTU` | `Slt/` | LT |
| `SLTI`, `SLTIU` | `SltImm/` | LT |
| `BEQ`, `BNE` (condition only) | `Beq/` | EQ |
| `BLT`, `BGE`, `BLTU`, `BGEU` (condition only) | `Blt/` | LT |

## Trust and scope

- That the Sail model matches the RISC-V manual is assumed (it is the official model).
- That the Clean model matches the Rust circuit is checked by `prover/src/tests/lean_diff.rs`,
  a randomized test, not a proof.
- The PC and the register file are not modeled on the circuit side yet: that needs the CPU in
  Clean. Steps 1 to 3 connect values only.
- Word shifts: only the 32-bit result (`outLo`) is connected. The sign extension to 64 bits
  is done by the CPU32 chip, not modeled yet. The first operand is assumed to be what CPU32
  sends (`spec/src/cpu32.toml`): the low word of `rs1`, zero-extended (`SLLW`, `SRLW`) or
  sign-extended (`SRAW`).
- Branches: only the condition is connected to the circuit. Sail's jump to `PC + imm` needs the
  PC on the circuit side (the CPU in Clean).
- The Sail model declares some platform settings as `axiom`s (`LeanRV64D/RiscvExtras.lean`).
  The branch theorems mention `jump_to`, which depends on `sys_enable_experimental_extensions`,
  so `#print axioms` lists it; the proofs themselves use only Lean's standard axioms.
- The `Alu` theorems assume (hypotheses `ha`, `hb`) that the CPU sends `x[rs1]` and the second
  operand (`x[rs2]`, `shamt`, or the immediate sign-extended to 64 bits) on the bus. Proving
  that also needs the CPU in Clean; it will be a fourth step per instruction on top of these
  three.
