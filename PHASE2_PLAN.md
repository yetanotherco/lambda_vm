# Phase 2 — Move ADD-style and bytewise ops out of CPU

## Goal

Bring Lambda's CPU table from **76 cols → ~58 cols** by (a) replacing the
24 per-byte operand columns with 6 word-pair columns and (b) removing the
ADD/SUB/AND/OR/XOR carry/byte constraints from CPU. The work moves to two
new dedicated AIRs (`BinaryAdd`, `Binary`). Closes most of Lambda's
`-1.4B element` gap with ZisK on `fib_iterative_8M`.

This document is the contract for Phase 2 — concrete column tables, bus
shapes, and per-step test gates. Subsequent commits implement one step at
a time and must keep the prover suite green at every step.

A separate Phase 1 (selector compression, 16 → ~5 cols via bit-decomp at
blowup=4) can run *after* Phase 2 lands; together they target 76 → ~47 cols.

## Where the savings come from

| Bucket | Cols today | Cols after Phase 2 |
|---|---:|---:|
| `arg1` (byte-decomp `[u8; 8]`) | 8 | 2 (`u32` lo/hi) |
| `arg2` (byte-decomp) | 8 | 2 |
| `res` (byte-decomp) | 8 | 2 |
| `RV1_EXT_BIT`, `RV2_EXT_BIT`, `RES_EXT_BIT` | 3 | 0 (folded into receivers) |
| **CPU operand block** | **27** | **6** |
| Other (decode / state / selectors / modifiers) | 49 | 49 (untouched in Phase 2) |
| **CPU total** | **76** | **55–58** |

Plus: removed transition constraints — 4 ADD-carry, 2 STORE-add carry,
2 SUB-carry, 2 JALR-carry, 7 SLT-res-zero, the `IS_EQUAL` definition,
all per-byte AND/OR/XOR sends to BITWISE.

Cost added:
- New `BinaryAdd` AIR — `~12` cols × `~N_add` rows, where `N_add` ≤
  number of ADD/LOAD/STORE/SUB/BEQ/JALR ops (deduped by operand pair).
- New `Binary` AIR — `~14` cols × `~N_bitwise` rows for AND/OR/XOR.
- Existing `BITWISE` AIR keeps IS_BYTE/IS_HALF/range-check work (no
  byte-level AND/OR/XOR sends from CPU anymore — those move to `Binary`).

For fib (mostly ADDs, very few unique pairs but every iteration is a
new operand): `N_add` ≈ `N_cpu_rows`. The 18-col-per-row CPU saving
dominates the per-add row growth in `BinaryAdd` (which has fewer cols).

---

## End-state CPU column layout (post-Phase 2)

| Range | Cols | Description | Same as today? |
|---|---:|---|---|
| `TIMESTAMP` | 1 | Cycle counter | yes |
| `PC[0..2]` | 2 | Program counter (DWordWL) | yes |
| `RS1, RS2, RD` | 3 | Register indices | yes |
| `READ_REG1/2, WRITE_REG, MEM_2/4/8B, C_TYPE` | 7 | Decode flags | yes |
| `IMM[0..2]` | 2 | Immediate (DWordWL) | yes |
| `SIGNED, MP_SELECTOR, MULDIV_SELECTOR, WORD_INSTR` | 4 | Op modifiers | yes |
| ALU selectors `ADD..EBREAK` | 16 | One-hot | yes (Phase 1 will compress later) |
| `NEXT_PC[0..2]` | 2 | Next PC | yes |
| `RVD[0..2]` | 2 | Register write value | yes |
| `RV1[0..3], RV2[0..3]` | 6 | Register read values (DWordWHH) | yes |
| `ARG1[0..2]` | 2 | Sign-extended rv1 (DWordWL) | **NEW shape** (was 8 bytes + ext bit) |
| `ARG2[0..2]` | 2 | Sign-extended rv2/imm (DWordWL) | **NEW shape** |
| `RES[0..2]` | 2 | ALU result (DWordWL) | **NEW shape** |
| `IS_EQUAL, BRANCH_COND` | 2 | Branch helpers | yes |
| `PREV_PC_TIMESTAMP_BORROW, PC_DOUBLE_READ` | 2 | PC inline borrow | yes |
| **NUM_COLUMNS** | **55** | | |

Diff from current: 76 → 55 (-21 cols).

The 3 `*_EXT_BIT` cols disappear because sign extension is no longer done
inline in CPU — the receiver AIRs compute it from word-form operands and
the existing `SIGNED` flag.

---

## New AIRs

### `BinaryAdd` AIR

Validates `lhs + rhs = sum (mod 2^64)` for ADD/SUB/LOAD/STORE/JALR/BEQ
dispatches.

**Columns** (12 total):

| Col | Width | Description |
|---|---|---|
| `LHS_LO, LHS_HI` | 2 | lhs as DWordWL |
| `RHS_LO, RHS_HI` | 2 | rhs as DWordWL |
| `SUM_LO, SUM_HI` | 2 | result as DWordWL |
| `CARRY_0, CARRY_1` | 2 | bit (carry between word boundaries) |
| `MU_ADD, MU_SUB` | 2 | multiplicities for the two receiver flavours |
| `_padding` | 0–2 | round to power of 2 if helpful |

**Constraints** (degree 3 max, fits blowup=2):
- `CARRY_0 * (1 - CARRY_0) = 0`
- `CARRY_1 * (1 - CARRY_1) = 0`
- `LHS_LO + RHS_LO - SUM_LO - 2^32 * CARRY_0 = 0`
- `LHS_HI + RHS_HI + CARRY_0 - SUM_HI - 2^32 * CARRY_1 = 0`

**Bus interactions:**
- Senders: 6 IS_HALFWORD lookups (range-check `LHS_LO/HI`, `RHS_LO/HI`,
  `SUM_LO/HI` × 2 = each is a u32 = 2 IS_HALF per word). Same pattern as
  today's MUL chip.
- Receivers (BinaryAdd's contract): two flavours on `BusId::BinaryAdd`:
  - `Multiplicity::Column(MU_ADD)`: `(lhs, rhs, sum)` triple — used by
    ADD/LOAD/STORE/JALR (forward addition).
  - `Multiplicity::Column(MU_SUB)`: `(lhs, rhs, sum)` — sender supplies
    `(arg2, res, arg1)` for SUB/BEQ (proves `arg2 + res = arg1`).

  CPU senders pick which flavour by which selector fires; the receiver
  doesn't distinguish the operand mapping.

### `Binary` AIR

Validates whole-64-bit AND / OR / XOR for AND/OR/XOR dispatches.

**Columns** (~14 total):

| Col | Width | Description |
|---|---|---|
| `LHS[0..8]` | 8 | lhs as DWordBL — byte cols stay here, where the work happens |
| `RHS[0..8]` | 8 | rhs as DWordBL |
| `RES[0..8]` | 8 | result as DWordBL |
| `OP_AND, OP_OR, OP_XOR` | 3 | one-hot per row (which op this row is) |
| `MU` | 1 | bus receiver multiplicity |

Wait — that's 28 cols, more than CPU's saving. **Refine: pack lhs/rhs/res
as DWordWL in `Binary` too**, do byte work via per-row IS_BYTE + AND/OR/XOR
BITWISE sends. That trades fewer cols for more bus interactions.

Actually looking at zisk's design: their `Binary` uses byte cols and BITWISE
sends, but only ~150 cols total are paid per Binary row vs CPU paying them
*every row*. The win is the row count: `N_binary` (only AND/OR/XOR ops) ≪
`N_cpu`.

For fib specifically: AND/OR/XOR are rare. The Binary AIR is small.

**Final Binary layout TBD** — depends on whether the byte-level work fits
better in Binary or stays in BITWISE-as-receiver. Decide during step 2.

**Bus interactions:**
- Senders (Binary → others): per-byte BITWISE sends for AND/OR/XOR (8 per
  row, like CPU does today) + per-byte IS_BYTE range checks.
- Receiver: `BusId::Binary` with `Multiplicity::Column(MU)`. CPU sends
  `(op, lhs, rhs, res)` where `op ∈ {AND, OR, XOR}` is a small constant.

---

## New `BusId` entries

Add two values to `prover/src/tables/types.rs::BusId`:

- `BinaryAdd` — for ADD-style 64-bit additions.
- `Binary` — for whole-64-bit AND/OR/XOR.

(Optional alternative: a single `Operation` bus carrying `op + operands`,
zisk-style. Defer that decision to step 1 if the migration logic gets
simpler with separate buses.)

---

## Migration steps (one PR per step)

Each step must keep all existing prover tests green — full prover suite
(`cargo test --release -p lambda-vm-prover`) plus stark suite. No
intermediate commit may have lower test coverage than `main`.

### Step 1 — Add `BusId` entries + skeleton AIRs (read-only)

- Add `BusId::BinaryAdd`, `BusId::Binary` to types.rs.
- Add `prover/src/tables/binary_add.rs` and `prover/src/tables/binary.rs`
  with column layouts, witness gen for empty traces, and bus-interactions
  fns returning empty `Vec`. No senders, no receivers — the AIRs
  exist but absorb nothing.
- Wire into `Traces`, `VmAirs`, `test_utils`, `lib.rs` as new tables that
  always have ≥4 padding rows.

**Test gate:** existing tests pass; both new AIRs prove and verify with
empty multiplicity.

### Step 2 — `BinaryAdd` receives ADD/LOAD ops

- Implement `BinaryAdd` constraints (carry chain) and the
  `(lhs, rhs, sum)` receiver with `Multiplicity::Column(MU_ADD)`.
- In `cpu.rs::bus_interactions`: add a sender on `BusId::BinaryAdd` for
  ADD and LOAD rows. Multiplicity = `Sum(ADD, LOAD)` (or two separate
  interactions).
- In `constraints/cpu.rs::create_add_constraints`: **keep** for now —
  CPU still has byte cols and the existing carry constraints fire
  redundantly. We'll drop them in step 4 once all add-style ops are
  migrated.
- Trace builder: collect `BinaryAdd` ops from CPU operations, build the
  `BinaryAdd` trace, fill `MU_ADD` from per-row multiplicities.

**Test gate:** prover tests pass; `BinaryAdd` AIR proves and verifies for
real workloads; bus balance holds.

### Step 3 — `BinaryAdd` receives STORE / SUB / BEQ / JALR ops

- Add CPU senders for STORE (lhs=arg1, rhs=imm, sum=res), SUB
  (lhs=arg2, rhs=res, sum=arg1), BEQ (same operand mapping as SUB),
  JALR (lhs=pc, rhs=instr_size, sum=res). Use `MU_ADD` for ADD/LOAD/
  STORE/JALR (forward) and `MU_SUB` for SUB/BEQ.
- Trace builder collects all six op families.

**Test gate:** prover tests pass; `BinaryAdd` row count matches expected
ADD-style op count.

### Step 4 — Drop ADD/SUB/STORE/JALR carry constraints from CPU

- Remove `create_add_constraints`, `create_sub_constraints`,
  `create_jalr_constraints`. Update `create_all_cpu_constraints` to
  return one fewer set.
- The carry validity is now enforced by `BinaryAdd`'s internal carry
  constraints + bus balance.
- This removes 8 transition constraints from CPU. No column change yet.

**Test gate:** prover tests pass.

### Step 5 — `Binary` receives AND/OR/XOR

- Implement `Binary` AIR with byte-level AND/OR/XOR senders to BITWISE
  and `BusId::Binary` receiver.
- CPU sender on `BusId::Binary` for AND/OR/XOR rows: send `(op, lhs,
  rhs, res)`. Multiplicity = `Sum(AND, OR, XOR)`.
- Remove CPU's per-byte AND/OR/XOR sends to BITWISE — those move to
  `Binary`'s senders.
- Trace builder collects AND/OR/XOR ops.

**Test gate:** prover tests pass; `Binary` AIR proves and verifies.

### Step 6 — Drop byte cols from CPU

- Replace `cols::ARG1[0..8]` with `cols::ARG1_LO, cols::ARG1_HI` (2 u32
  cols). Same for ARG2 and RES.
- Update CPU witness generation: write 2 u32 values per operand instead
  of 8 bytes.
- Update CPU senders to use `Packing::DWordWL` on 2 cols instead of
  `DWordBL` on 8 — same bus values, smaller storage.
- Remove `RV1_EXT_BIT`, `RV2_EXT_BIT`, `RES_EXT_BIT`. Sign extension
  moves into `BinaryAdd`/`Binary` receivers, computed from `SIGNED` and
  the high u32.
- Update IS_BYTE range checks: send IS_HALF on each u32 instead of
  IS_BYTE on each byte pair (or move range checks entirely into
  receivers — design decision in step 5).
- Update branch_cond, slt_res_zero, JALR (now via BinaryAdd) constraints.
- `cols::NUM_COLUMNS = 55` (was 76).

**Test gate:** full prover suite passes; element count benchmark shows
the expected drop on `fib_iterative_8M`.

### Step 7 (optional) — element count check

- Run `bench_vs/run_elements.sh --no-sp1 -n 8000000`. Expect Lambda's
  main element count to drop by `~18 × cycles ≈ 580M` (vs the +8.4M
  regression we measured before this work). Document in the PR.

---

## Per-step test invariants

Every step must satisfy:

- `make lint` passes.
- `cargo test --release -p stark` passes.
- `cargo test --release -p lambda-vm-prover` passes (the 11 known
  pre-existing failures from missing ELF artifacts on this machine
  remain — no new regressions).
- The constraint-analyzer agent reports no new Critical/High findings.
- Bus balance holds for every interaction (trace builder must build
  consistent senders/receivers).

A step is not done until it ships green on a fresh checkout of
`compress-cpu`.

---

## Open questions to resolve before step 1

1. **Single `Operation` bus or two separate buses?** **Decided: two
   separate buses** (`BinaryAdd`, `Binary`), matching Lambda's current
   style of one bus per ALU AIR family (`Mul`, `Dvrm`, `Lt`, `Shift`,
   …). No unified `Operation` bus refactor planned.

2. **Where do byte-level work and IS_BYTE range checks live in
   `Binary`?** Two designs:
   - (a) `Binary` keeps byte cols (8 × 3 = 24) and sends per-byte
     AND/OR/XOR + IS_BYTE to BITWISE — same as CPU does today, just
     fewer rows.
   - (b) `Binary` uses 2-word ops + does byte breakdown via internal
     virtual cols, sends one 64-bit AND/OR/XOR lookup to a new
     whole-word table.

   (a) is simpler, smaller per-row footprint than CPU pays today,
   compatible with existing BITWISE. (b) saves more cols on `Binary`
   but adds a new lookup table. **Recommendation: (a)**.

3. **Trace builder phase ordering with two new AIRs.** `BinaryAdd`
   needs to collect ADD-style ops (deduped). `Binary` needs AND/OR/XOR
   ops. Both need to run before the bitwise multiplicity update phase.
   Likely an extra phase between current Phase 2 (CPU op collection)
   and Phase 4 (bitwise). Confirm during step 1.

4. **`Multiplicity` for the BinaryAdd receivers.** Two flavours
   (`MU_ADD`, `MU_SUB`). Forward-add multiplicity uses
   `Multiplicity::Linear` over `[ADD, LOAD, STORE, JALR]` —
   confirmed: CPU's existing `non_pad_mult` at `cpu.rs:1746-1811`
   already uses `Multiplicity::Linear` over all 16 selectors, so the
   framework supports arbitrary-length column sums and the LogUp
   constraints stay at degree 1 / 2. SUB/BEQ multiplicity uses
   `Multiplicity::Sum(SUB, BEQ)` (just 2 cols).

---

## What this plan does NOT cover

- **Phase 1 selector compression** (16 → 4 bit cols + op byte). Runs
  separately, *after* Phase 2 reduces complexity. Targets a further
  ~10-col saving and requires a `blowup_factor=4` framework change.
- **FrequentOps integration** with the new buses. Once `BinaryAdd` and
  `Binary` exist, FrequentOps can absorb hits on them — that's where
  the column savings translate to actual element-count savings on
  arith-heavy workloads. Wire up in a follow-up PR after Phase 2 lands.
- **STORE arg2/rv2 ambiguity.** Today STORE uses arg2 to hold rv2 (the
  store value), and the address-add uses imm directly. Step 3 must
  preserve this — the BinaryAdd sender for STORE uses `imm` as rhs, not
  `arg2`. Confirm the bus shape matches CPU's existing semantics.

---

## Estimated effort

- Step 1: 1 day (skeleton AIRs + wiring).
- Step 2: 2-3 days (BinaryAdd impl + ADD/LOAD migration).
- Step 3: 2 days (other op families).
- Step 4: 1 day (constraint cleanup).
- Step 5: 3-4 days (Binary AIR + AND/OR/XOR migration).
- Step 6: 3-4 days (byte-col removal — touches every CPU bus interaction).
- Step 7: 0.5 days (measurement).

**Total: ~2-3 weeks calendar, sequential**. Steps 1-4 (BinaryAdd path)
can ship before steps 5-6 (Binary path) — they're independent ops.
