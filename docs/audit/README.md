# Prover Spec Audit

Verifying the prover implementation matches the spec exactly.

## Spec Source

- Source of truth: `spec/main` branch, `/spec/` directory (Typst/TOML format)
- Readable version: `md_spec` branch, `docs/spec/` directory (.md files)
- Conversion: `scripts/extract_and_convert_spec.sh` + `scripts/typst_to_md.py`
- Before auditing, verify sync: `git checkout md_spec && bash scripts/extract_and_convert_spec.sh origin/spec/main docs/spec && git diff docs/spec/`
- Last verified in sync: 2026-03-11

## Audit Order

| # | Table | Constraints | Status | Notes |
|---|-------|-------------|--------|-------|
| 1 | Bitwise | 11 (all bus interactions) | PASS | No polynomial constraints, receiver only |
| 2 | Branch | 10 (4 poly + 6 bus) | PASS | ADD template x2, IS_BYTE, AND_BYTE, IS_HALF x3, BRANCH receiver |
| 3 | LT | 12 (3 poly + 9 bus) | PASS | carry IS_BIT x2, LT formula, MSB16 x2, IS_HALF x6, LT receiver |
| 4 | Load | 13 (8 poly + 5 bus) | PASS | Extension constraints, MEMW sender, MSB8 x3, LOAD receiver |
| 5 | MUL | 22 (6 poly + 16 bus) | PASS | SIGN x2, raw_product x4, IS_HALF x8, IS_B20 x4, MUL receiver x2 |
| 6 | DVRM | 34 (19 poly + 26 bus + 8 extra) | PASS | SIGN x3, NEG x2, 8 extra IS_HALF for input assumptions |
| 7 | Shift | 15 | | |
| 8 | MEMW | 25 | | Memory argument core |
| 9 | HALT | 33 (0 poly + 33 bus) | PASS | 1 ECALL receiver + 32 MEMW senders (register finalization) |
| 10 | COMMIT | ~12 | | Part of ecall.md |
| 11 | Decode | 1 (0 poly + 1 bus) | PASS | Preprocessed lookup table, packed_decode 51-bit layout verified |
| 12 | CPU | 56 poly + ~60 bus | **FINDINGS** | 4 CRITICAL missing constraints, 2 MODERATE deviations |

## Cross-Cutting Audits

| Audit | Status | Notes |
|-------|--------|-------|
| Bus Balance (all bus IDs) | | After all tables done |
| Signature Consistency | | |
| Memory Argument | | |
| Decode Instructions | | |
| Variable Types | | |

---

## Table Audit Reports

### 1. Bitwise

**Spec**: `docs/spec/bitwise.md` on `md_spec` branch
**Code**: `prover/src/tables/bitwise.rs`
**Tests**: `prover/src/tests/bitwise_tests.rs`, `prover/src/tests/bitwise_bus_tests.rs`

#### Columns

| Check | Status | Details |
|-------|--------|---------|
| Input columns match spec | PASS | X (Byte, col 0), Y (Byte, col 1), Z (B4, col 2) |
| Output columns match spec | PASS | AND (col 3), OR (col 4), XOR (col 5), MSB8 (col 6), MSB16 (col 7), ZERO (col 8), SLL (col 9), SLLC (col 10) |
| Multiplicity columns match spec | PASS | 11 mu columns (cols 11-21), one per bus interaction |
| Column count | PASS | NUM_COLUMNS = 22 (3 input + 8 output + 11 multiplicity) |
| Column ordering contiguous | PASS | 0-21, no gaps |
| No extra columns | PASS | |
| No missing columns | PASS | |
| Type correctness | PASS | All types match: Byte=1 FE, B4=1 FE, Bit=1 FE, Half=1 FE, BaseField=1 FE |
| Table size | PASS | NUM_ROWS = 2^20 = 256*256*16, matches spec |
| Preprocessed | PASS | Input+output columns are precomputed, matches spec |

#### Output Value Formulas

| Column | Spec Formula | Code (generate_bitwise_row) | Match |
|--------|-------------|---------------------------|-------|
| AND | binary AND of X and Y | `x & y` | PASS |
| OR | binary OR of X and Y | `x \| y` | PASS |
| XOR | binary XOR of X and Y | `x ^ y` | PASS |
| MSB8 | most significant bit of X | `(x >> 7) & 1` | PASS |
| MSB16 | most significant bit of Y | `(halfword >> 15) & 1` where halfword = X + 256*Y | PASS |
| ZERO | X=0, Y=0, Z=0 | `if x==0 && y==0 && z==0 {1} else {0}` | PASS |
| SLL | (X+256Y) << Z mod 2^16 | `(halfword << z) & 0xFFFF` (z=0: halfword) | PASS |
| SLLC | (X+256Y) >> (16-Z) | `halfword >> (16-z)` (z=0: 0) | PASS |

Note on SLLC when Z=0: spec says `(X+256Y) >> 16` which is always 0 for 16-bit input. Code returns 0. Consistent.

#### Row Indexing

| Check | Status | Details |
|-------|--------|---------|
| Index formula | PASS | `x + y*256 + z*65536` matches spec decomposition |
| generate_bitwise_row | PASS | Uses `index & 0xFF`, `(index>>8) & 0xFF`, `(index>>16) & 0xF` |
| generate_bitwise_trace | PASS | Same formula: `x + y*256 + z*256*256` (256*256 = 65536) |
| row_index() helper | PASS | Same formula |

#### Polynomial Constraints

| Check | Status | Details |
|-------|--------|---------|
| No polynomial constraints in spec | PASS | Spec has no polynomial constraint section (only bus interactions) |
| No polynomial constraints in code | PASS | No `DvrmConstraint`-style evaluator exists for bitwise |

#### Bus Interactions (BITWISE-C1 through BITWISE-C11)

All interactions are **receiver** (negative multiplicity in spec = receiver in code). Checked against signatures.toml.

| Tag | Spec | Code BusId | Direction | Multiplicity | Values | Status |
|-----|------|-----------|-----------|--------------|--------|--------|
| BITWISE-C1 | `AND_BYTE[AND; X, Y]` | AndByte | receiver | Column(MU_AND) | [X, Y, AND] | PASS |
| BITWISE-C2 | `OR_BYTE[OR; X, Y]` | OrByte | receiver | Column(MU_OR) | [X, Y, OR] | PASS |
| BITWISE-C3 | `XOR_BYTE[XOR; X, Y]` | XorByte | receiver | Column(MU_XOR) | [X, Y, XOR] | PASS |
| BITWISE-C4 | `MSB8[MSB8; X]` | Msb8 | receiver | Column(MU_MSB8) | [X, MSB8] | PASS |
| BITWISE-C5 | `MSB16[MSB16; X+256*Y]` | Msb16 | receiver | Column(MU_MSB16) | [linear(X+256*Y), MSB16] | PASS |
| BITWISE-C6 | `ZERO[ZERO; X+256*Y+65536*Z]` | Zero | receiver | Column(MU_ZERO) | [linear(X+256*Y+65536*Z), ZERO] | PASS |
| BITWISE-C7 | `IS_BYTE[X]` | IsByte | receiver | Column(MU_IS_BYTE) | [X] | PASS |
| BITWISE-C8 | `IS_HALF[X+256*Y]` | IsHalfword | receiver | Column(MU_IS_HALF) | [linear(X+256*Y)] | PASS |
| BITWISE-C9 | `IS_B20[X+256*Y+65536*Z]` | IsB20 | receiver | Column(MU_IS_B20) | [linear(X+256*Y+65536*Z)] | PASS |
| BITWISE-C10 | `HWSL[SLL; X+256*Y, Z]` | Hwsl | receiver | Column(MU_HWSL) | [linear(X+256*Y), Z, SLL] | PASS |
| BITWISE-C11 | `HWSLC[SLLC; X+256*Y, Z]` | Hwslc | receiver | Column(MU_HWSLC) | [linear(X+256*Y), Z, SLLC] | PASS |

##### Signature element count verification

| Interaction | Spec Signature | Expected bus size | Code values count | Match |
|-------------|---------------|-------------------|-------------------|-------|
| AND_BYTE | `[res; X, Y]` → Byte, Byte, Byte | 3 | 3 (X, Y, AND) | PASS |
| OR_BYTE | `[res; X, Y]` → Byte, Byte, Byte | 3 | 3 (X, Y, OR) | PASS |
| XOR_BYTE | `[res; X, Y]` → Byte, Byte, Byte | 3 | 3 (X, Y, XOR) | PASS |
| MSB8 | `[msb; X]` → Byte, Bit | 2 | 2 (X, MSB8) | PASS |
| MSB16 | `[msb; X]` → Half, Bit | 2 | 2 (X+256Y, MSB16) | PASS |
| ZERO | `[is_zero; X]` → B20, Bit | 2 | 2 (X+256Y+65536Z, ZERO) | PASS |
| IS_BYTE | `[X]` → Byte | 1 | 1 (X) | PASS |
| IS_HALF | `[X]` → Half | 1 | 1 (X+256Y) | PASS |
| IS_B20 | `[X]` → B20 | 1 | 1 (X+256Y+65536Z) | PASS |
| HWSL | `[res; X, shift]` → Half, B4, Half | 3 | 3 (X+256Y, Z, SLL) | PASS |
| HWSLC | `[res; X, shift]` → Half, B4, Half | 3 | 3 (X+256Y, Z, SLLC) | PASS |

##### Value ordering verification

For interactions with output (`;` in signature), spec convention is: output comes first in the signature name, but in the bus values the inputs come first followed by the output. Checking each:

| Interaction | Spec signature | Spec implies order | Code order | Match |
|-------------|---------------|-------------------|------------|-------|
| AND_BYTE[AND; X, Y] | output=AND, input=[X,Y] | X, Y, AND | X, Y, AND | PASS |
| OR_BYTE[OR; X, Y] | output=OR, input=[X,Y] | X, Y, OR | X, Y, OR | PASS |
| XOR_BYTE[XOR; X, Y] | output=XOR, input=[X,Y] | X, Y, XOR | X, Y, XOR | PASS |
| MSB8[MSB8; X] | output=MSB8, input=[X] | X, MSB8 | X, MSB8 | PASS |
| MSB16[MSB16; X+256Y] | output=MSB16, input=[X+256Y] | X+256Y, MSB16 | X+256Y, MSB16 | PASS |
| ZERO[ZERO; X+256Y+65536Z] | output=ZERO, input=[X+256Y+65536Z] | X+256Y+65536Z, ZERO | X+256Y+65536Z, ZERO | PASS |
| IS_BYTE[X] | no output, input=[X] | X | X | PASS |
| IS_HALF[X+256Y] | no output, input=[X+256Y] | X+256Y | X+256Y | PASS |
| IS_B20[X+256Y+65536Z] | no output | X+256Y+65536Z | X+256Y+65536Z | PASS |
| HWSL[SLL; X+256Y, Z] | output=SLL, input=[X+256Y, Z] | X+256Y, Z, SLL | X+256Y, Z, SLL | PASS |
| HWSLC[SLLC; X+256Y, Z] | output=SLLC, input=[X+256Y, Z] | X+256Y, Z, SLLC | X+256Y, Z, SLLC | PASS |

#### Padding

| Check | Status | Details |
|-------|--------|---------|
| Table is fixed size 2^20 | PASS | No padding needed — table always has exactly 2^20 rows |
| Unused rows have mu=0 | PASS | Multiplicity columns init to zero, only incremented by update_multiplicities |

#### Trace Generation

| Check | Status | Details |
|-------|--------|---------|
| generate_bitwise_trace fills all input/output cols | PASS | Triple loop over x,y,z fills all 11 precomputed columns |
| Multiplicity columns init to zero | PASS | vec![FE::zero(); ...] and not explicitly set |
| update_multiplicities increments correctly | PASS | Maps op type to mu column, increments by 1 |
| Row indexing consistent | PASS | Both trace gen and row_index use same formula |
| const fn matches runtime | PASS | generate_bitwise_row uses same formulas as trace gen |

#### No extra/missing interactions

| Check | Status | Details |
|-------|--------|---------|
| Code has 11 bus interactions | PASS | Matches spec's 11 constraints (C1-C11) |
| No extra interactions in code | PASS | |
| No missing interactions from spec | PASS | |

#### Spec tag comments in code

| Check | Status | Details |
|-------|--------|---------|
| Constraint tags in comments | NOTE | Code comments don't reference BITWISE-C1..C11 tags explicitly. Not a correctness issue but would aid traceability. |

#### Summary

**BITWISE TABLE: PASS**

All columns, bus interactions, signatures, value formulas, multiplicities, and trace generation match the spec exactly. No findings.

---

### 2. Branch

**Spec**: `docs/spec/branch.md` on `md_spec` branch
**Code**: `prover/src/tables/branch.rs`
**Tests**: `prover/src/tests/branch_bus_tests.rs`

#### Columns

| Check | Status | Details |
|-------|--------|---------|
| Input columns match spec | PASS | pc: DWordWL (cols 0-1), offset: DWordWL (cols 2-3), register: DWordWL (cols 4-5), JALR: Bit (col 6) |
| Output columns match spec | PASS | next_pc_high: Half[3] (cols 7-9), next_pc_low: Byte[2] (cols 10-11) |
| Auxiliary columns match spec | PASS | unmasked_low_byte: Byte (col 12) |
| Multiplicity column match spec | PASS | mu: Bit (col 13) |
| Virtual columns defined correctly | PASS | next_pc_unmasked and next_pc formulas match spec (see below) |
| Column count | PASS | NUM_COLUMNS = 14 (7 input + 5 output + 1 aux + 1 mu) |
| Column ordering contiguous | PASS | 0-13, no gaps |
| No extra columns | PASS | |
| No missing columns | PASS | |
| Type correctness | PASS | DWordWL=2 Words, Half=1 FE, Byte=1 FE, Bit=1 FE |

#### Virtual Columns

| Virtual | Spec Formula | Code Formula | Match |
|---------|-------------|-------------|-------|
| next_pc_unmasked[0] | `2^16 * next_pc_high[0] + 2^8 * next_pc_low[1] + unmasked_low_byte` | `unmasked_low_byte + next_pc_low_1 * 256 + next_pc_high_0 * 65536` | PASS |
| next_pc_unmasked[1] | `2^16 * next_pc_high[2] + next_pc_high[1]` | `next_pc_high_1 + next_pc_high_2 * 65536` | PASS |
| next_pc[0] | `2^16 * next_pc_high[0] + 2^8 * next_pc_low[1] + next_pc_low[0]` | `next_pc_low_0 + 256 * next_pc_low_1 + 65536 * next_pc_high_0` | PASS |
| next_pc[1] | `2^16 * next_pc_high[2] + next_pc_high[1]` | `next_pc_high_1 + 65536 * next_pc_high_2` | PASS |

#### Polynomial Constraints (ADD Template)

Spec BRANCH-C1: `(1 - JALR) => ADD<next_pc_unmasked; pc, offset>` expands via ADD-C1.i to:
- `(1-JALR) * carry[0] * (1 - carry[0]) = 0`
- `(1-JALR) * carry[1] * (1 - carry[1]) = 0`

Spec BRANCH-C2: `JALR => ADD<next_pc_unmasked; register, offset>` expands to:
- `JALR * carry[0] * (1 - carry[0]) = 0`
- `JALR * carry[1] * (1 - carry[1]) = 0`

| Constraint | Spec | Code (BranchConstraintKind) | Match |
|-----------|------|---------------------------|-------|
| C1 carry[0] | `(1-JALR) * carry_0_pc * (1-carry_0_pc) = 0` | PcCarry0IsBit | PASS |
| C1 carry[1] | `(1-JALR) * carry_1_pc * (1-carry_1_pc) = 0` | PcCarry1IsBit | PASS |
| C2 carry[0] | `JALR * carry_0_reg * (1-carry_0_reg) = 0` | RegCarry0IsBit | PASS |
| C2 carry[1] | `JALR * carry_1_reg * (1-carry_1_reg) = 0` | RegCarry1IsBit | PASS |

Carry formulas:
| Formula | Spec | Code | Match |
|---------|------|------|-------|
| carry[0] | `2^{-32} * (lhs[0] + rhs[0] - sum[0])` | `(base_0 + offset_0 - unmasked_0) * inv_2_32` | PASS |
| carry[1] | `2^{-32} * (lhs[1] + rhs[1] + carry[0] - sum[1])` | `(base_1 + offset_1 + carry_0 - unmasked_1) * inv_2_32` | PASS |

Constraint degree: spec requires degree 3 (`cond * carry * (1-carry)`). Code returns `degree() = 3`. **PASS**

#### Bus Interactions

| Tag | Spec | Code BusId | Direction | Multiplicity | Values | Status |
|-----|------|-----------|-----------|--------------|--------|--------|
| BRANCH-C3 | `IS_BYTE[next_pc_low[1]]` | IsByte | sender | Column(MU) | [NEXT_PC_LOW_1] | PASS |
| BRANCH-C4 | `AND_BYTE[next_pc_low[0]; unmasked_low_byte, 254]` | AndByte | sender | Column(MU) | [UNMASKED_LOW_BYTE, const(254), NEXT_PC_LOW_0] | PASS |
| BRANCH-C5.0 | `IS_HALF[next_pc_high[0]]` | IsHalfword | sender | Column(MU) | [NEXT_PC_HIGH_0] | PASS |
| BRANCH-C5.1 | `IS_HALF[next_pc_high[1]]` | IsHalfword | sender | Column(MU) | [NEXT_PC_HIGH_1] | PASS |
| BRANCH-C5.2 | `IS_HALF[next_pc_high[2]]` | IsHalfword | sender | Column(MU) | [NEXT_PC_HIGH_2] | PASS |
| BRANCH-C6 | `BRANCH[next_pc; pc, offset, register, JALR]` | Branch | receiver | Column(MU) | [next_pc(2), pc(2), offset(2), register(2), JALR(1)] = 9 FEs | PASS |

##### Signature verification (BRANCH bus)

| Element | Spec type | Code representation | Match |
|---------|----------|-------------------|-------|
| next_pc | DWordWL (2 Words) | 2 linear combinations packing Half/Byte to Word | PASS |
| pc | DWordWL (2 Words) | 2 Direct columns (PC_0, PC_1) | PASS |
| offset | DWordWL (2 Words) | 2 Direct columns (OFFSET_0, OFFSET_1) | PASS |
| register | DWordWL (2 Words) | 2 Direct columns (REGISTER_0, REGISTER_1) | PASS |
| JALR | Bit (1 FE) | 1 Direct column (JALR) | PASS |
| **Total bus size** | **9** | **9** | **PASS** |

#### Assumptions

| Tag | Spec | Verified by | Status |
|-----|------|------------|--------|
| BRANCH-A1.i | IS_WORD[pc[i]] | CPU table ensures pc is valid Word pair | PASS (cross-table) |
| BRANCH-A2 | IS_WORD[offset] | CPU table provides sign-extended offset as DWordWL | PASS (cross-table) |
| BRANCH-A3.i | IS_WORD[register[i]] | CPU table ensures register is valid | PASS (cross-table) |
| BRANCH-A4 | IS_BIT<JALR> | CPU sends JALR as 0 or 1 | PASS (cross-table) |

#### Padding

| Check | Status | Details |
|-------|--------|---------|
| Padding with zeros | PASS | `vec![FE::zero(); ...]` initializes all columns to 0 |
| Padding rows don't fire bus interactions | PASS | MU=0 on padding rows, all interactions use Multiplicity::Column(MU) |
| Padding rows satisfy polynomial constraints | PASS | All zeros: cond=0 (both 1-JALR=1 path has carry=0, and IS_BIT(0)=0) |
| Power-of-2 sizing | PASS | `next_power_of_two().max(4)` |

#### Trace Generation

| Check | Status | Details |
|-------|--------|---------|
| All columns filled correctly | PASS | pc, offset, register split to DWordWL; next_pc decomposed to high/low |
| Deduplication | PASS | HashMap merges identical ops, sums multiplicities |
| LSB masking | PASS | `unmasked & !1u64` correctly masks bit 0 |
| next_pc_low[0] | PASS | Extracted from masked next_pc, not unmasked |
| unmasked_low_byte | PASS | Extracted from unmasked next_pc |

#### Summary

**BRANCH TABLE: PASS**

All 14 columns, 4 polynomial constraints (ADD template), 6 bus interactions, virtual column definitions, carry formulas, and trace generation match the spec exactly. No findings.

---

### 3. LT

**Spec**: `docs/spec/lt.md` on `md_spec` branch
**Code**: `prover/src/tables/lt.rs`
**Tests**: `prover/src/tests/lt_tests.rs`, `prover/src/tests/lt_bus_tests.rs`

#### Columns

| Check | Status | Details |
|-------|--------|---------|
| Input columns match spec | PASS | lhs: DWordHHW [Word(0), Half(1), Half(2)], rhs: DWordHHW [Word(3), Half(4), Half(5)], signed: Bit(6) |
| Output columns match spec | PASS | lt: Bit (col 7) |
| Auxiliary columns match spec | PASS | lhs_sub_rhs: DWordHL [Half(8), Half(9), Half(10), Half(11)], lhs_msb: Bit(12), rhs_msb: Bit(13) |
| Multiplicity column match spec | PASS | mu (col 14) |
| Virtual columns defined correctly | PASS | carry[0], carry[1], unsigned_lt = carry[1] — all computed inline |
| Column count | PASS | NUM_COLUMNS = 15 (7 input + 1 output + 6 aux + 1 mu) |
| Column ordering contiguous | PASS | 0-14, no gaps |
| No extra columns | PASS | |
| No missing columns | PASS | |
| Type correctness | PASS | DWordHHW = [Word, Half, Half] = 3 cols, DWordHL = [Half×4] = 4 cols |

#### Virtual Columns

| Virtual | Spec Formula | Code Formula | Match |
|---------|-------------|-------------|-------|
| carry[0] | `2^{-32} * (rhs[0] + (lhs_sub_rhs::DWordWL)[0] - lhs[0])` | `(rhs_0 + (sub_0 + sub_1*2^16) - lhs_0) * inv_2_32` | PASS |
| carry[1] | `2^{-32} * ((rhs::DWordWL)[1] + (lhs_sub_rhs::DWordWL)[1] + carry[0] - (lhs::DWordWL)[1])` | `(rhs_hi + sub_hi + carry_0 - lhs_hi) * inv_2_32` | PASS |
| unsigned_lt | `carry[1]` | `c = self.compute_carry_1(step)` | PASS |

Type cast verification:
- `lhs_sub_rhs::DWordWL[0]` = `sub[0] + 2^16 * sub[1]` (DWordHL→DWordWL low word) — code: `sub_0 + sub_1 * shift_16` PASS
- `lhs_sub_rhs::DWordWL[1]` = `sub[2] + 2^16 * sub[3]` — code: `sub_2 + sub_3 * shift_16` PASS
- `rhs::DWordWL[1]` = `rhs[1] + 2^16 * rhs[2]` (DWordHHW→DWordWL high word) — code: `rhs_1 + rhs_2 * shift_16` PASS
- `lhs::DWordWL[1]` = `lhs[1] + 2^16 * lhs[2]` — code: `lhs_1 + lhs_2 * shift_16` PASS

#### Polynomial Constraints

| Tag | Spec | Code (LtConstraintKind) | Degree | Match |
|-----|------|------------------------|--------|-------|
| LT-C6.0 | `IS_BIT<carry[0]>`: `carry[0] * (1 - carry[0]) = 0` | Carry0IsBit | 2 | PASS |
| LT-C6.1 | `IS_BIT<carry[1]>`: `carry[1] * (1 - carry[1]) = 0` | Carry1IsBit | 2 | PASS |
| LT-C3 | `lt - signed*(A*(1-B) + A*C + (1-B)*C) - (1-signed)*unsigned_lt = 0` | LtFormula | 3 | PASS |

LT formula detailed check:
- Spec: `lt - signed * (lhs_msb * (1 - rhs_msb) + lhs_msb * carry[1] + (1 - rhs_msb) * carry[1]) - (1 - signed) * unsigned_lt = 0`
- Code: `signed_lt = a * (1-b) + a * c + (1-b) * c; expected = signed * signed_lt + (1-signed) * c; return lt - expected`
- Where A=lhs_msb, B=rhs_msb, C=carry[1], unsigned_lt=carry[1]
- Matches exactly (Q(A,B,C) = A(1-B) + AC + (1-B)C, with proof in spec that Q≡P for valid inputs)

#### Bus Interactions

| Tag | Spec | Code BusId | Direction | Multiplicity | Values | Status |
|-----|------|-----------|-----------|--------------|--------|--------|
| LT-C1 | `MSB16[lhs_msb; lhs[2]]` | Msb16 | sender | Column(MU) | [LHS_2, LHS_MSB] | PASS |
| LT-C2 | `MSB16[rhs_msb; rhs[2]]` | Msb16 | sender | Column(MU) | [RHS_2, RHS_MSB] | PASS |
| LT-C4 | `IS_HALF[lhs[1]]` | IsHalfword | sender | Column(MU) | [LHS_1] | PASS |
| LT-C5 | `IS_HALF[rhs[1]]` | IsHalfword | sender | Column(MU) | [RHS_1] | PASS |
| LT-C7.0 | `IS_HALF[lhs_sub_rhs[0]]` | IsHalfword | sender | Column(MU) | [LHS_SUB_RHS_0] | PASS |
| LT-C7.1 | `IS_HALF[lhs_sub_rhs[1]]` | IsHalfword | sender | Column(MU) | [LHS_SUB_RHS_1] | PASS |
| LT-C7.2 | `IS_HALF[lhs_sub_rhs[2]]` | IsHalfword | sender | Column(MU) | [LHS_SUB_RHS_2] | PASS |
| LT-C7.3 | `IS_HALF[lhs_sub_rhs[3]]` | IsHalfword | sender | Column(MU) | [LHS_SUB_RHS_3] | PASS |
| LT-C8 | `LT[lt; lhs::DWordWL, rhs::DWordWL, signed]` | Lt | receiver | Column(MU) | [lhs(DWordHHW→2), rhs(DWordHHW→2), signed(1), lt(1)] = 6 FEs | PASS |

##### LT bus signature verification

Spec signature: `LT[lt; lhs, rhs, signed]` with input=[DWordWL, DWordWL, Bit], output=Bit → 2+2+1+1 = 6 FEs

Code uses Packing::DWordHHW which reads [Word, Half, Half] and produces 2 elements: [col[0], col[1]+2^16*col[2]]. This produces the same 2-Word representation as DWordWL, so senders using DWordWL packing will match. **PASS**

#### Assumptions

| Tag | Spec | Verified by | Status |
|-----|------|------------|--------|
| LT-A1 | IS_WORD[lhs[0]] | CPU sender ensures lhs[0] is valid Word | PASS (cross-table) |
| LT-A2 | IS_WORD[rhs[0]] | CPU sender ensures rhs[0] is valid Word | PASS (cross-table) |
| LT-A3 | IS_BIT<signed> | CPU sender ensures signed is 0 or 1 | PASS (cross-table) |

#### Padding

| Check | Status | Details |
|-------|--------|---------|
| Padding with zeros | PASS | `vec![FE::zero(); ...]` initializes all columns to 0 |
| Padding rows don't fire bus interactions | PASS | MU=0 on padding rows |
| Padding rows satisfy carry IS_BIT | PASS | carry[0]=carry[1]=0 from all-zero inputs, IS_BIT(0)=0 |
| Padding rows satisfy LT formula | PASS | lt=0, signed=0, (1-signed)*carry[1] = 1*0 = 0 |
| Power-of-2 sizing | PASS | `next_power_of_two().max(4)` |

#### Trace Generation

| Check | Status | Details |
|-------|--------|---------|
| lhs decomposition | PASS | DWordHHW: [bits 0-31, bits 32-47, bits 48-63] |
| rhs decomposition | PASS | Same DWordHHW format |
| lhs_sub_rhs computation | PASS | `lhs.wrapping_sub(rhs)` then split to 4 halfwords |
| MSB extraction | PASS | `(value >> 63) & 1` correctly gets bit 63 |
| lt computation | PASS | Signed: cast to i64 and compare. Unsigned: direct u64 compare |
| Deduplication | PASS | HashMap merges identical (lhs, rhs, signed) tuples |

#### Summary

**LT TABLE: PASS**

All 15 columns, 3 polynomial constraints (2 carry IS_BIT + LT formula), 9 bus interactions (2 MSB16 + 6 IS_HALF + 1 LT receiver), virtual column definitions with DWordWL casts, and trace generation match the spec exactly. No findings.

---

### 4. Load

**Spec**: `docs/spec/load.md` on `md_spec` branch
**Code**: `prover/src/tables/load.rs`
**Tests**: `prover/src/tables/load.rs::tests` (inline)

#### Columns

| Check | Status | Details |
|-------|--------|---------|
| Input columns match spec | PASS | base_address: DWordWL (cols 0-1), timestamp: DWordWL (cols 2-3), read2: Bit (4), read4: Bit (5), read8: Bit (6), signed: Bit (7) |
| Output columns match spec | PASS | res: DWordBL = Byte[8] (cols 8-15) |
| Auxiliary columns match spec | PASS | sign_bit: Bit (col 16) |
| Multiplicity column match spec | PASS | mu: Bit (col 17) |
| Virtual columns defined correctly | PASS | read1 = mu - read2 - read4 - read8 (computed inline in MSB8 multiplicity) |
| Column count | PASS | NUM_COLUMNS = 18 (8 input + 8 output + 1 aux + 1 mu) |
| Column ordering contiguous | PASS | 0-17, no gaps |
| No extra columns | PASS | |
| No missing columns | PASS | |
| Type correctness | PASS | DWordWL=2 Words, DWordBL=8 Bytes, Bit=1 FE |

#### Polynomial Constraints

| Tag | Spec | Code (LoadConstraintKind) | Degree | Match |
|-----|------|--------------------------|--------|-------|
| LOAD-C1 | `(read2 + read4 + read8) * (1 - mu) = 0` | ReadImpliesMu | 2 | PASS |
| LOAD-C6.4 | `(1 - read8) * (res[4] - signed * sign_bit * 255) = 0` | ExtensionHigh(4) | 3 | PASS |
| LOAD-C6.5 | `(1 - read8) * (res[5] - signed * sign_bit * 255) = 0` | ExtensionHigh(5) | 3 | PASS |
| LOAD-C6.6 | `(1 - read8) * (res[6] - signed * sign_bit * 255) = 0` | ExtensionHigh(6) | 3 | PASS |
| LOAD-C6.7 | `(1 - read8) * (res[7] - signed * sign_bit * 255) = 0` | ExtensionHigh(7) | 3 | PASS |
| LOAD-C7.2 | `(1 - read4 - read8) * (res[2] - signed * sign_bit * 255) = 0` | ExtensionMid(2) | 3 | PASS |
| LOAD-C7.3 | `(1 - read4 - read8) * (res[3] - signed * sign_bit * 255) = 0` | ExtensionMid(3) | 3 | PASS |
| LOAD-C8 | `(1 - read2 - read4 - read8) * (res[1] - signed * sign_bit * 255) = 0` | ExtensionLow | 3 | PASS |

Extension logic:
- Not reading 8 bytes → bytes 4-7 must be sign-extended (0x00 or 0xFF)
- Not reading 4+ bytes → bytes 2-3 must also be sign-extended
- Not reading 2+ bytes → byte 1 must also be sign-extended
- Byte 0 is always real data (the minimum read width is 1 byte)

#### Bus Interactions

| Tag | Spec | Code BusId | Direction | Multiplicity | Values | Status |
|-----|------|-----------|-----------|--------------|--------|--------|
| LOAD-C2 | `MEMW[res; 0, base_address, res, timestamp, read2, read4, read8]` | Memw | sender | Column(MU) | [old=res(8), is_reg=0(1), base_addr(2), value=res(8), timestamp(2), r2(1), r4(1), r8(1)] = 24 FEs | PASS |
| LOAD-C3 | `MSB8[sign_bit; res[0]]` | Msb8 | sender | Linear(MU-READ2-READ4-READ8) = read1 | [RES[0], SIGN_BIT] | PASS |
| LOAD-C4 | `MSB8[sign_bit; res[1]]` | Msb8 | sender | Column(READ2) | [RES[1], SIGN_BIT] | PASS |
| LOAD-C5 | `MSB8[sign_bit; res[3]]` | Msb8 | sender | Column(READ4) | [RES[3], SIGN_BIT] | PASS |
| LOAD-C9 | `LOAD[res::DWordWL; base_address, timestamp, read2, read4, read8, signed]` | Load | receiver | Column(MU) | [res(DWordBL→2), base_addr(DWordWL→2), timestamp(DWordWL→2), r2(1), r4(1), r8(1), signed(1)] = 10 FEs | PASS |

##### MEMW interaction cross-check

Verified LOAD sender ordering matches MEMW read receiver (memw.rs line 715):
- Both: old[0..7], is_register, base_address[0..1], value[0..7], timestamp[0..1], write2, write4, write8
- For reads: old = value = res (LOAD correctly sends RES columns for both)
- is_register = constant(0) (memory, not register access)

##### LOAD bus signature verification

Spec: `LOAD[res; base_address, timestamp, read2, read4, read8, signed]`
Signature: input=[DWordWL, DWordWL, Bit, Bit, Bit, Bit], output=DWordWL → 2+2+1+1+1+1+2 = 10 FEs
Code: DWordBL packing (8 bytes → 2 words) + DWordWL (2) + DWordWL (2) + 4 Direct = 10 FEs. **PASS**

##### MSB8 multiplicity verification

| Width | read1 | read2 | read4 | MSB8 source | Multiplicity |
|-------|-------|-------|-------|-------------|-------------|
| 1 byte | 1 | 0 | 0 | res[0] | read1 = mu-0-0-0 = 1 |
| 2 bytes | 0 | 1 | 0 | res[1] | read2 = 1 |
| 4 bytes | 0 | 0 | 1 | res[3] | read4 = 1 |
| 8 bytes | 0 | 0 | 0 | none | no MSB8 lookup (all 3 mults = 0) |

All correct — exactly one MSB8 lookup fires per active row (except read8 where no sign extension needed).

#### Assumptions

| Tag | Spec | Verified by | Status |
|-----|------|------------|--------|
| LOAD-A1.i | IS_WORD[base_address[i]] | CPU sender | PASS (cross-table) |
| LOAD-A2 | IS_BIT<signed> | CPU sender | PASS (cross-table) |
| LOAD-A3 | IS_BIT<read2> | CPU sender | PASS (cross-table) |
| LOAD-A4 | IS_BIT<read4> | CPU sender | PASS (cross-table) |
| LOAD-A5 | IS_BIT<read8> | CPU sender | PASS (cross-table) |
| LOAD-A6 | IS_BIT<read2+read4+read8> | CPU sends at most one flag set | PASS (cross-table) |
| LOAD-A7.i | IS_WORD[timestamp[i]] | CPU sender | PASS (cross-table) |

#### Padding

| Check | Status | Details |
|-------|--------|---------|
| Padding with zeros | PASS | `vec![FE::zero(); ...]` initializes all to 0 |
| Padding rows don't fire bus interactions | PASS | MU=0, read1=0, READ2=0, READ4=0 — all multiplicities zero |
| Padding rows satisfy C1 | PASS | (0+0+0)*(1-0) = 0 |
| Padding rows satisfy C6-C8 | PASS | All res[i]=0, signed=0, sign_bit=0 → (1-0)*(0-0)=0 |
| Power-of-2 sizing | PASS | `next_power_of_two().max(4)` |

#### Trace Generation

| Check | Status | Details |
|-------|--------|---------|
| base_address split to DWordWL | PASS | Low 32 bits, high 32 bits |
| timestamp split to DWordWL | PASS | Same |
| Read flags encoding | PASS | Width 1→(0,0,0), 2→(1,0,0), 4→(0,1,0), 8→(0,0,1) — "exactly N" semantics |
| res[8] stored correctly | PASS | 8 individual byte columns |
| sign_bit computation | PASS | MSB of byte at index [0,1,3,7] depending on width |
| No deduplication | NOTE | Unlike Branch/LT, LOAD doesn't deduplicate (each op is unique row). Acceptable since loads at different timestamps are always distinct. |

#### Summary

**LOAD TABLE: PASS**

All 18 columns, 8 polynomial constraints (1 ReadImpliesMu + 4 ExtensionHigh + 2 ExtensionMid + 1 ExtensionLow), 5 bus interactions (1 MEMW sender + 3 MSB8 senders + 1 LOAD receiver), and trace generation match the spec exactly. MEMW interaction ordering verified against MEMW read receiver. No findings.

---

### 5. MUL

**Spec**: `docs/spec/mul.md` on `md_spec` branch
**Code**: `prover/src/tables/mul.rs`
**Tests**: `prover/src/tests/mul_tests.rs`, `prover/src/tests/mul_bus_tests.rs`

#### Columns

| Check | Status | Details |
|-------|--------|---------|
| Input columns match spec | PASS | lhs: DWordHL (cols 0-3), lhs_signed: Bit (4), rhs: DWordHL (cols 5-8), rhs_signed: Bit (9) |
| Output columns match spec | PASS | lo: DWordHL (cols 10-13), hi: DWordHL (cols 14-17) |
| Auxiliary columns match spec | PASS | lhs_is_negative: Bit (18), rhs_is_negative: Bit (19), raw_product: B51[4] (cols 20-23) |
| Multiplicity columns match spec | PASS | mu_lo: BaseField (24), mu_hi: BaseField (25) |
| Virtual columns defined correctly | PASS | lhs_ext, rhs_ext (sign-extended 8 halves), res (QuadWL), carry (B20[4]), mu_sum — all computed inline |
| Column count | PASS | NUM_COLUMNS = 26 (10 input + 8 output + 6 aux + 2 mu) |
| No extra/missing columns | PASS | |
| Type correctness | PASS | DWordHL=4 Halves, B51=1 FE, Bit=1 FE |

#### Virtual Columns

| Virtual | Spec | Code | Match |
|---------|------|------|-------|
| lhs_ext[0..3] | lhs[i] | lhs[0..3] (halfwords) | PASS |
| lhs_ext[4..7] | 65535 * lhs_is_negative | `SIGN_FILL * lhs_is_neg` where SIGN_FILL=0xFFFF | PASS |
| rhs_ext[0..3] | rhs[i] | rhs[0..3] | PASS |
| rhs_ext[4..7] | 65535 * rhs_is_negative | `SIGN_FILL * rhs_is_neg` | PASS |
| res[0] | lo::DWordWL[0] = lo[0]+2^16*lo[1] | Implicit in carry formula | PASS |
| res[1] | lo::DWordWL[1] = lo[2]+2^16*lo[3] | Implicit in carry formula | PASS |
| res[2] | hi::DWordWL[0] = hi[0]+2^16*hi[1] | Implicit in carry formula | PASS |
| res[3] | hi::DWordWL[1] = hi[2]+2^16*hi[3] | Implicit in carry formula | PASS |
| mu_sum | mu_lo + mu_hi | Multiplicity::Sum(MU_LO, MU_HI) | PASS |

#### Polynomial Constraints

| Tag | Spec | Code (MulConstraintKind) | Degree | Match |
|-----|------|--------------------------|--------|-------|
| MUL-C1 (SIGN-C2) | `(1 - lhs_signed) * lhs_is_negative = 0` | LhsSign | 2 | PASS |
| MUL-C2 (SIGN-C2) | `(1 - rhs_signed) * rhs_is_negative = 0` | RhsSign | 2 | PASS |
| MUL-C6.0 | raw_product[0] = convolution at i=0 | RawProduct(0) | 2 | PASS |
| MUL-C6.1 | raw_product[1] = convolution at i=1 | RawProduct(1) | 2 | PASS |
| MUL-C6.2 | raw_product[2] = convolution at i=2 | RawProduct(2) | 2 | PASS |
| MUL-C6.3 | raw_product[3] = convolution at i=3 | RawProduct(3) | 2 | PASS |

Raw product formula: `raw_product[i] = Σ_{k=0}^{1} 2^{16k} × Σ_{j=0}^{2i+k} lhs_ext[j] × rhs_ext[2i+k-j]`
Code computes this with nested loops matching the exact formula. **PASS**

#### Bus Interactions

| Tag | Spec | Code BusId | Direction | Multiplicity | Values | Status |
|-----|------|-----------|-----------|--------------|--------|--------|
| MUL-C1 (SIGN-C1) | `MSB16[lhs_is_negative; lhs[3]]` | Msb16 | sender | Column(LHS_SIGNED) | [LHS_3, LHS_IS_NEGATIVE] | PASS |
| MUL-C2 (SIGN-C1) | `MSB16[rhs_is_negative; rhs[3]]` | Msb16 | sender | Column(RHS_SIGNED) | [RHS_3, RHS_IS_NEGATIVE] | PASS |
| MUL-C3.0-3 | `IS_HALF[lo[i]]` for i∈[0,3] | IsHalfword | sender | Sum(MU_LO, MU_HI) | [LO_i] | PASS |
| MUL-C4.0-3 | `IS_HALF[hi[i]]` for i∈[0,3] | IsHalfword | sender | Sum(MU_LO, MU_HI) | [HI_i] | PASS |
| MUL-C5.0 | `IS_B20[carry[0]]` | IsB20 | sender | Sum(MU_LO, MU_HI) | [linear: 2^{-32}*rp[0] - 2^{-32}*lo[0] - 2^{-16}*lo[1]] | PASS |
| MUL-C5.1 | `IS_B20[carry[1]]` | IsB20 | sender | Sum(MU_LO, MU_HI) | [linear: 6 terms with rp[0..1], lo[0..3]] | PASS |
| MUL-C5.2 | `IS_B20[carry[2]]` | IsB20 | sender | Sum(MU_LO, MU_HI) | [linear: 9 terms with rp[0..2], lo[0..3], hi[0..1]] | PASS |
| MUL-C5.3 | `IS_B20[carry[3]]` | IsB20 | sender | Sum(MU_LO, MU_HI) | [linear: 12 terms with rp[0..3], lo[0..3], hi[0..3]] | PASS |
| MUL-C7 | `MUL[lo::DWordWL; lhs, lhs_signed, rhs, rhs_signed, 0]` | Mul | receiver | Column(MU_LO) | [lhs(DWordHL→2), lhs_signed(1), rhs(DWordHL→2), rhs_signed(1), lo(DWordHL→2), const(0)] = 9 | PASS |
| MUL-C8 | `MUL[hi::DWordWL; lhs, lhs_signed, rhs, rhs_signed, 1]` | Mul | receiver | Column(MU_HI) | [lhs(DWordHL→2), lhs_signed(1), rhs(DWordHL→2), rhs_signed(1), hi(DWordHL→2), const(1)] = 9 | PASS |

##### Carry formula expansion verification

Each carry[i] is sent as a linear combination of columns to the IS_B20 bus. The formula is recursively expanded so carry[i] depends only on raw_product and lo/hi columns (no intermediate carry columns):

| carry | Expanded formula | Code terms | Match |
|-------|-----------------|------------|-------|
| carry[0] | `2^{-32}*rp[0] - 2^{-32}*lo[0] - 2^{-16}*lo[1]` | 3 terms | PASS |
| carry[1] | `2^{-32}*rp[1] + 2^{-64}*rp[0] - 2^{-64}*lo[0] - 2^{-48}*lo[1] - 2^{-32}*lo[2] - 2^{-16}*lo[3]` | 6 terms | PASS |
| carry[2] | `2^{-32}*rp[2] + 2^{-64}*rp[1] + 2^{-96}*rp[0] - 2^{-96}*lo[0] - 2^{-80}*lo[1] - 2^{-64}*lo[2] - 2^{-48}*lo[3] - 2^{-32}*hi[0] - 2^{-16}*hi[1]` | 9 terms | PASS |
| carry[3] | `2^{-32}*rp[3] + 2^{-64}*rp[2] + 2^{-96}*rp[1] + 2^{-128}*rp[0] - 2^{-128}*lo[0] - 2^{-112}*lo[1] - 2^{-96}*lo[2] - 2^{-80}*lo[3] - 2^{-64}*hi[0] - 2^{-48}*hi[1] - 2^{-32}*hi[2] - 2^{-16}*hi[3]` | 12 terms | PASS |

##### MUL bus signature verification

Spec: `MUL[lo/hi; lhs, lhs_signed, rhs, rhs_signed, 0/1]`
Signature: input=[DWordHL, Bit, DWordHL, Bit, Bit], output=DWordWL
Bus size: 2+1+2+1+1+2 = 9 FEs
Code uses DWordHL packing (4 halves → 2 words) for lhs, rhs, lo/hi — produces same 2-word representation as DWordWL. **PASS**

#### SIGN Template Verification

| Check | MUL-C1 (lhs) | MUL-C2 (rhs) |
|-------|-------------|-------------|
| SIGN-C1: MSB16 lookup | Msb16 sender, mult=LHS_SIGNED, [LHS_3, LHS_IS_NEGATIVE] | Msb16 sender, mult=RHS_SIGNED, [RHS_3, RHS_IS_NEGATIVE] |
| SIGN-C2: unsigned zero | (1-lhs_signed)*lhs_is_negative=0 | (1-rhs_signed)*rhs_is_negative=0 |
| Spec: X=lhs[3]/rhs[3], signed=lhs_signed/rhs_signed, sign=lhs_is_neg/rhs_is_neg | PASS | PASS |

#### Assumptions

| Tag | Spec | Verified by | Status |
|-----|------|------------|--------|
| MUL-A1.i | IS_HALF[lhs[i]] for i∈[0,3] | CPU sender ensures lhs halfwords valid | PASS (cross-table) |
| MUL-A2.i | IS_HALF[rhs[i]] for i∈[0,3] | CPU sender ensures rhs halfwords valid | PASS (cross-table) |

#### Padding

| Check | Status | Details |
|-------|--------|---------|
| Padding with zeros | PASS | All zeros: SIGN constraints satisfied (0*0=0), raw_product constraints satisfied (0=0) |
| Padding bus interactions | PASS | MU_LO=MU_HI=0 so mu_sum=0 and all multiplicities zero |
| Power-of-2 sizing | PASS | `next_power_of_two().max(4)` |

#### Trace Generation

| Check | Status | Details |
|-------|--------|---------|
| lhs/rhs decomposition | PASS | DWordHL: 4 halfwords via `& 0xFFFF` and shifts |
| Product computation | PASS | Uses i128 arithmetic with correct sign extension |
| lo/hi decomposition | PASS | DWordHL from 64-bit product halves |
| raw_product computation | PASS | Convolution formula with sign-extended arrays |
| Deduplication | PASS | HashMap on (lhs, lhs_signed, rhs, rhs_signed), separate mu_lo/mu_hi |

#### Constraint Completeness

| Check | Status | Details |
|-------|--------|---------|
| All spec constraints have code | PASS | C1-C8 all implemented (6 poly + 16 bus) |
| No extra code constraints | PASS | 6 polynomial kinds + 16 bus interactions, all mapped to spec tags |

#### Summary

**MUL TABLE: PASS**

All 26 columns, 6 polynomial constraints (2 SIGN template + 4 raw_product convolution), 16 bus interactions (2 MSB16 + 8 IS_HALF + 4 IS_B20 carry + 2 MUL receivers), carry formula expansions, and trace generation match the spec exactly. No findings.

---

### 6. DVRM

**Spec**: `docs/spec/dvrm.md` on `md_spec` branch
**Code**: `prover/src/tables/dvrm.rs`
**Templates used**: SIGN (sign.md), NEG (neg.md), IS_BIT (is_bit.md)

#### Columns

| Check | Status | Details |
|-------|--------|---------|
| Input columns match spec | PASS | n: DWordHL (N_0..N_3, cols 0-3), d: DWordHL (D_0..D_3, cols 4-7), signed: Bit (col 8) |
| Output columns match spec | PASS | q: DWordHL (Q_0..Q_3, cols 9-12), r: DWordHL (R_0..R_3, cols 13-16) |
| Auxiliary columns match spec | PASS | div_by_zero (17), overflow (18), abs_r DWordWL (19-20), abs_d DWordWL (21-22), n_sub_r DWordHL (23-26), sign_n_sub_r (27), sign_n (28), sign_d (29), sign_q (30), sign_r (31) |
| Multiplicity columns match spec | PASS | μ_q (32), μ_r (33) |
| Virtual columns | PASS | extended_n, extended_r, extension_n_sub_r, extended_n_sub_r (QuadHL→QuadWL), carry[0..3]: all computed inline in CarryIsBit constraint |
| Column count | PASS | NUM_COLUMNS = 34 |
| Column ordering contiguous | PASS | 0-33, no gaps |
| No extra columns | PASS | |
| No missing columns | PASS | |
| Type correctness | PASS | DWordHL=4 Half cols, DWordWL=2 Word cols, Bit=1 col, BaseField=1 col |

#### Assumptions

| Tag | Spec | Code | Status | Notes |
|-----|------|------|--------|-------|
| DVRM-A1.i | IS_HALF[n[i]] | IS_HALF sender for N_0..N_3 (μ_sum) | PASS* | Code enforces as bus interaction; spec treats as assumption. Overconstrained but sound. |
| DVRM-A2.i | IS_HALF[d[i]] | IS_HALF sender for D_0..D_3 (μ_sum) | PASS* | Same as A1. |
| DVRM-A3 | IS_BIT\<signed\> | SignedIsBit: signed*(1-signed)=0 | PASS | IS_BIT template expanded to polynomial |

*Note: The code sends 8 extra IS_HALF interactions (4 for n, 4 for d) that the spec lists only as assumptions (to be enforced by the CPU sender). This is redundant but sound — it means the DVRM table self-enforces its input range checks rather than relying on the sender.

#### Polynomial Constraints

| Spec Tag | Code Kind | Polynomial | Status |
|----------|-----------|-----------|--------|
| DVRM-A3 | SignedIsBit | `signed * (1-signed) = 0` | PASS |
| DVRM-C1 | RemainderSignMatchesNumerator | `(r[0]+r[1]+r[2]+r[3]) * (sign_r - sign_n) = 0` | PASS |
| DVRM-C4.0 | AbsRFormula(0) | `(1-sign_r) * (abs_r[0] - (r[0]+r[1]*2^16)) = 0` | PASS |
| DVRM-C4.1 | AbsRFormula(1) | `(1-sign_r) * (abs_r[1] - (r[2]+r[3]*2^16)) = 0` | PASS |
| DVRM-C6.0 | AbsDFormula(0) | `(1-sign_d) * (abs_d[0] - (d[0]+d[1]*2^16)) = 0` | PASS |
| DVRM-C6.1 | AbsDFormula(1) | `(1-sign_d) * (abs_d[1] - (d[2]+d[3]*2^16)) = 0` | PASS |
| DVRM-C7 | SignQFormula | `signed * (1-overflow) - sign_q = 0` | PASS |
| DVRM-C12.0 | CarryIsBit(0) | `carry[0] * (1-carry[0]) = 0` where carry[0] = 2^-32 * (ext_nsr[0]+ext_r[0]-ext_n[0]) | PASS |
| DVRM-C12.1 | CarryIsBit(1) | `carry[1] * (1-carry[1]) = 0` where carry[1] recursive | PASS |
| DVRM-C12.2 | CarryIsBit(2) | `carry[2] * (1-carry[2]) = 0` | PASS |
| DVRM-C12.3 | CarryIsBit(3) | `carry[3] * (1-carry[3]) = 0` | PASS |
| DVRM-C15 | SignNSubRIsBit | `sign_n_sub_r * (1-sign_n_sub_r) = 0` | PASS |
| DVRM-C18 (SIGN-C2) | UnsignedSignN | `(1-signed) * sign_n = 0` | PASS |
| DVRM-C19 (SIGN-C2) | UnsignedSignR | `(1-signed) * sign_r = 0` | PASS |
| DVRM-C20 (SIGN-C2) | UnsignedSignD | `(1-signed) * sign_d = 0` | PASS |
| DVRM-C16.0-3 | DivByZeroQ(0..3) | `div_by_zero * (q[i]-65535) = 0` | PASS |

Total: 19 polynomial constraints. All degree 2. All match spec exactly.

**Carry computation verification**: Virtual carries use sign-extended QuadWL representation via `build_extended_quad`. Extended values: lower 2 words = halfword pairs packed to words, upper 2 words = sign * 0xFFFFFFFF. Carry formula matches spec definition exactly: `carry[0] = 2^-32 * ((ext_nsr::QuadWL)[0] + (ext_r::QuadWL)[0] - (ext_n::QuadWL)[0])`, recursive for carry[1..3].

#### Bus Interactions

| Spec Tag | Code | Bus ID | Direction | Mult | Values | Status |
|----------|------|--------|-----------|------|--------|--------|
| DVRM-C2 | LT sender | Lt | Sender | μ_sum | [abs_r(DWordWL), abs_d(DWordWL), 0, 1-div_by_zero] | PASS |
| DVRM-C3 (NEG-C1) | ZERO carry0 | Zero | Sender | sign_r | [r[0]+r[1], 1-carry[0] expanded] | PASS |
| DVRM-C3 (NEG-C2) | ZERO carry1 | Zero | Sender | sign_r | [r[0]+r[1]+r[2]+r[3], 1-carry[1] expanded] | PASS |
| DVRM-C5 (NEG-C1) | ZERO carry0 | Zero | Sender | sign_d | [d[0]+d[1], 1-carry[0] expanded] | PASS |
| DVRM-C5 (NEG-C2) | ZERO carry1 | Zero | Sender | sign_d | [d[0]+d[1]+d[2]+d[3], 1-carry[1] expanded] | PASS |
| DVRM-C8 | ZERO overflow | Zero | Sender | μ_sum | [overflow_sum, overflow] | PASS |
| DVRM-C9 | MUL lo | Mul | Sender | μ_sum | [d(DWordHL), signed, q(DWordHL), sign_q, n_sub_r(DWordHL), 0] | PASS |
| DVRM-C10 | MUL hi | Mul | Sender | μ_sum | [d(DWordHL), signed, q(DWordHL), sign_q, ext_nsr(linear), 1] | PASS |
| DVRM-C11.i | IS_HALF q[i] | IsHalfword | Sender | μ_sum | [Q_i] ×4 | PASS |
| DVRM-C13.i | IS_HALF r[i] | IsHalfword | Sender | μ_sum | [R_i] ×4 | PASS |
| DVRM-C14.i | IS_HALF n_sub_r[i] | IsHalfword | Sender | μ_sum | [N_SUB_R_i] ×4 | PASS |
| DVRM-C17 | ZERO div_by_zero | Zero | Sender | μ_sum | [d[0]+d[1]+d[2]+d[3], div_by_zero] | PASS |
| DVRM-C18 (SIGN-C1) | MSB16 sign_n | Msb16 | Sender | signed | [N_3, SIGN_N] | PASS |
| DVRM-C19 (SIGN-C1) | MSB16 sign_r | Msb16 | Sender | signed | [R_3, SIGN_R] | PASS |
| DVRM-C20 (SIGN-C1) | MSB16 sign_d | Msb16 | Sender | signed | [D_3, SIGN_D] | PASS |
| DVRM-C21 | DVRM q | Dvrm | Receiver | μ_q | [n(DWordHL), d(DWordHL), signed, q(DWordHL), 0] | PASS |
| DVRM-C22 | DVRM r | Dvrm | Receiver | μ_r | [n(DWordHL), d(DWordHL), signed, r(DWordHL), 1] | PASS |

Total spec interactions: 26. Code: 34 (26 + 8 extra IS_HALF for inputs).

**Cross-check: MUL sender ordering** matches MUL receiver (mul.rs lines 600-670): [lhs, lhs_signed, rhs, rhs_signed, result, selector]. Verified consistent.

**C8 overflow_sum verification**: `n[0]+n[1]+n[2]+n[3]-(2^15+1)*sign_n+(1+4*65535)-d[0]-d[1]-d[2]-d[3]`. Code coefficient: -32769 for SIGN_N, constant 262141 = 1+4*65535. When overflow holds: n=0x8000_0000_0000_0000, d=0xFFFF_FFFF_FFFF_FFFF, signed=1 → sum=0. ZERO lookup correctly constrains overflow bit.

**C10 extension_n_sub_r**: Each word = `sign_n_sub_r * (65535 + 65535*2^16)` = `sign_n_sub_r * 0xFFFFFFFF`. Matches spec: `extension_n_sub_r[i] = 65535 * sign_n_sub_r` as DWordHL→DWordWL.

**NEG carry expansion verification**: C3a: `1-carry[0] = 1 - 2^-32*abs_r[0] - 2^-32*r[0] - 2^-16*r[1]`. Matches `1 - 2^-32*((r::DWordWL)[0] + abs_r[0])` where `(r::DWordWL)[0] = r[0]+r[1]*2^16`. C3b: recursive expansion with carry[0] substituted inline. All coefficients verified: NEG_INV_2_32, NEG_INV_2_16, NEG_INV_2_48, NEG_INV_2_64.

#### Padding

| Check | Status | Details |
|-------|--------|---------|
| Padding values | PASS | All zeros (n=0, d=0, signed=0, q=0, r=0, all aux=0, μ_q=μ_r=0) |
| Polynomial constraints hold | PASS | All constraints evaluate to 0 on all-zero rows |
| Bus interactions silent | PASS | All multiplicities are 0 on padding (μ_q=μ_r=0, signed=0, sign_*=0) |

#### Trace Generation

| Check | Status | Details |
|-------|--------|---------|
| Column filling | PASS | All halfword/word decompositions correct for n, d, q, r, n_sub_r, abs_r, abs_d |
| Computation | PASS | compute_quotient/compute_remainder handle div_by_zero, overflow, signed/unsigned |
| sign_q formula | PASS | `signed && !is_overflow()` matches spec C7 |
| Deduplication | PASS | HashMap by (n, d, signed), separate μ_q and μ_r |
| Padding | PASS | Remaining rows zero-filled, power-of-2 sizing |

#### Completeness

| Check | Status | Details |
|-------|--------|---------|
| All spec constraints have code | PASS | C1-C22 all implemented (19 poly + 26 bus) |
| No extra code constraints | PASS | 8 extra IS_HALF for inputs (sound but not in spec) |

#### Tag Numbering Note

Code comments use different C-tag numbers than spec (e.g., code "C9" for carry IS_BIT, spec C12). This is a documentation-only mismatch. All constraints are functionally correct regardless of tag labels.

#### Summary

**DVRM TABLE: PASS**

All 34 columns, 19 polynomial constraints (1 IS_BIT signed, 1 remainder sign, 2+2 abs formulas, 1 sign_q, 4 carry IS_BIT, 1 sign_n_sub_r IS_BIT, 3 unsigned sign, 4 div_by_zero q), and 26 bus interactions (1 LT, 4 ZERO NEG carry, 1 ZERO overflow, 1 ZERO div_by_zero, 2 MUL, 12 IS_HALF, 3 MSB16, 2 DVRM receivers) match the spec. 8 additional IS_HALF interactions for input range checks (spec assumptions A1, A2) are overconstrained but sound. Virtual carry computation for 128-bit sign-extended addition verified. No findings.

---

### 7. HALT

**Spec**: `docs/spec/ecall.md` (HALT section) on `md_spec` branch
**Code**: `prover/src/tables/halt.rs`

#### Columns

| Check | Status | Details |
|-------|--------|---------|
| Input columns match spec | PASS | timestamp: DWordWL (TIMESTAMP_0 col 0, TIMESTAMP_1 col 1) |
| Column count | PASS | NUM_COLUMNS = 2 |
| No extra columns | PASS | |
| No missing columns | PASS | |
| Type correctness | PASS | DWordWL = 2 Word columns |

#### Assumptions

| Tag | Spec | Code | Status | Notes |
|-----|------|------|--------|-------|
| HALT-A1.i | IS_WORD[timestamp[i]] | Not enforced locally | PASS | Assumption — enforced by CPU sender |

#### Polynomial Constraints

None in spec, none in code. PASS.

#### Bus Interactions

| Spec Tag | Code | Bus ID | Direction | Mult | Values | Status |
|----------|------|--------|-----------|------|--------|--------|
| HALT-C1.i (i∈[1,9]) | x1-x9 write | Memw | Sender | 1 | MEMW write format: [is_reg=1, addr=2*i, val=0, ts=2^64-1, w2=1, w4=0, w8=0] (16 elems) | PASS |
| HALT-C2 | x10 read | Memw | Sender | 1 | MEMW read format: [old=0, is_reg=1, addr=20, val=0, ts=2^64-1, w2=1, w4=0, w8=0] (24 elems) | PASS |
| HALT-C3.i (i∈[11,31]) | x11-x31 write | Memw | Sender | 1 | Same write format, addr=2*i, val=0 (16 elems) | PASS |
| HALT-C4 | x255 write | Memw | Sender | 1 | Write format: [is_reg=1, addr=510, val=[1,0..0], ts=2^64-1, w2=1, w4=0, w8=0] (16 elems) | PASS |
| HALT-C5 | ECALL receiver | Ecall | Receiver | -1 | [timestamp[0], timestamp[1], 93, 0] (4 elems) | PASS |

Total: 33 interactions (9 + 1 + 21 + 1 MEMW senders + 1 ECALL receiver). Code `Vec::with_capacity(33)` matches.

**MEMW format verification**:
- Write format (CO25, 16 elems): `[is_register(1), base_addr(2), value(8), timestamp(2), write2(1), write4(1), write8(1)]`
- Read format (CO24, 24 elems): `[old(8), is_register(1), base_addr(2), value(8), timestamp(2), write2(1), write4(1), write8(1)]`
- C2 (x10 read) uses old=0 to enforce exit_code=0: if x10≠0 at halt, bus imbalance → proof failure. ✓
- C4 (x255) uses value[0]=1 (PC halted sentinel). ✓
- All use ts=0xFFFFFFFF_FFFFFFFF (2^64-1, maximum timestamp preventing further operations). ✓

**ECALL signature verification**: `ECALL[timestamp, syscall_number]` where input=[DWordWL, DWordWL]. Code: [TIMESTAMP_0, TIMESTAMP_1, 93, 0]. Syscall 93 = sys_exit. ✓

#### Padding

| Check | Status | Details |
|-------|--------|---------|
| Single-row table | PASS | 2^0 = 1 row, no padding needed per spec |
| Trace generation | PASS | `generate_halt_trace` produces 1 row with timestamp split into lo/hi words |

#### Trace Generation

| Check | Status | Details |
|-------|--------|---------|
| Column filling | PASS | timestamp_lo = ts & 0xFFFF_FFFF, timestamp_hi = ts >> 32 |
| Table size | PASS | TraceTable::new_main with 2 cols, 1 row |
| All values constant | PASS | No computation — all MEMW values are hardcoded constants in bus_interactions() |

#### Completeness

| Check | Status | Details |
|-------|--------|---------|
| All spec constraints have code | PASS | C1-C5 all implemented |
| No extra code constraints | PASS | 33 bus interactions, all mapped to spec |

#### Summary

**HALT TABLE: PASS**

All 2 columns, 0 polynomial constraints, and 33 bus interactions (9 write-zero x1-x9, 1 read-zero x10 exit code, 21 write-zero x11-x31, 1 write-1 x255 PC sentinel, 1 ECALL receiver for syscall 93) match the spec exactly. Register addresses use `2*i` convention. All MEMW interactions use timestamp 2^64-1 (maximum). Exit code enforcement via read-with-old=0 is correct. No findings.

---

### 8. Decode

**Spec**: `docs/spec/decode.md` on `md_spec` branch
**Code**: `prover/src/tables/decode.rs`, `prover/src/tables/types.rs` (DecodeEntry, packed_decode)

This is a preprocessed lookup table — no polynomial constraints. The audit focuses on column layout, packed_decode bit format, instruction-to-flags mapping, bus interaction, and padding.

#### Columns

| Check | Status | Details |
|-------|--------|---------|
| Output columns match spec | PASS | pc: DWordWL (PC_0 col 0, PC_1 col 1), packed_decode: BaseField (col 2), imm: DWordWL (IMM_0 col 3, IMM_1 col 4) |
| Multiplicity column | PASS | μ (MU col 5) |
| Column count | PASS | NUM_COLUMNS = 6, NUM_PRECOMPUTED_COLS = 5 (excludes MU) |
| No extra columns | PASS | |
| No missing columns | PASS | |

#### packed_decode Bit Layout (51 bits)

| Bit(s) | Spec | Code (packed_decode module) | Status |
|--------|------|---------------------------|--------|
| [0] | read_register1 | READ_REG1 = 0 | PASS |
| [1] | read_register2 | READ_REG2 = 1 | PASS |
| [2] | write_register | WRITE_REG = 2 | PASS |
| [3] | memory_2bytes | MEMORY_2BYTES = 3 | PASS |
| [4] | memory_4bytes | MEMORY_4BYTES = 4 | PASS |
| [5] | memory_8bytes | MEMORY_8BYTES = 5 | PASS |
| [6] | c_type | C_TYPE = 6 | PASS |
| [7] | signed | SIGNED = 7 | PASS |
| [8] | mp_selector | MP_SELECTOR = 8 | PASS |
| [9] | muldiv_selector | MULDIV_SELECTOR = 9 | PASS |
| [10] | word_instr | WORD_INSTR = 10 | PASS |
| [11] | ADD | OP_ADD = 11 | PASS |
| [12] | SUB | OP_SUB = 12 | PASS |
| [13] | SLT | OP_SLT = 13 | PASS |
| [14] | AND | OP_AND = 14 | PASS |
| [15] | OR | OP_OR = 15 | PASS |
| [16] | XOR | OP_XOR = 16 | PASS |
| [17] | SHIFT | OP_SHIFT = 17 | PASS |
| [18] | JALR | OP_JALR = 18 | PASS |
| [19] | BEQ | OP_BEQ = 19 | PASS |
| [20] | BLT | OP_BLT = 20 | PASS |
| [21] | LOAD | OP_LOAD = 21 | PASS |
| [22] | STORE | OP_STORE = 22 | PASS |
| [23] | MUL | OP_MUL = 23 | PASS |
| [24] | DIVREM | OP_DIVREM = 24 | PASS |
| [25] | ECALL | OP_ECALL = 25 | PASS |
| [26] | EBREAK | OP_EBREAK = 26 | PASS |
| [27:35] | rs1 (8 bits) | RS1 = 27 | PASS |
| [35:43] | rs2 (8 bits) | RS2 = 35 | PASS |
| [43:51] | rd (8 bits) | RD = 43 | PASS |

All 51 bit positions match spec exactly.

#### Instruction Decoding (from_instruction)

| Instruction | op-flag | w_instr | signed | other | Status |
|-------------|---------|---------|--------|-------|--------|
| ADDI[W] | ADD | [W] | 0 | | PASS |
| SLTI | SLT | 0 | 1 | | PASS |
| SLTIU | SLT | 0 | 0 | | PASS |
| ANDI | AND | 0 | 0 | | PASS |
| ORI | OR | 0 | 0 | | PASS |
| XORI | XOR | 0 | 0 | | PASS |
| SLLI[W] | SHIFT | [W] | 0 | | PASS |
| SRLI[W] | SHIFT | [W] | 0 | mp_selector | PASS |
| SRAI[W] | SHIFT | [W] | 1 | mp_selector | PASS |
| ADD[W] | ADD | [W] | 0 | | PASS |
| SUB[W] | SUB | [W] | 0 | | PASS |
| SLT[U] | SLT | 0 | !U | | PASS |
| AND/OR/XOR | flags | 0 | 0 | | PASS |
| SLL[W]/SRL[W]/SRA[W] | SHIFT | [W] | SRA:1 | SRL/SRA:mp_sel | PASS |
| MUL | MUL | 0 | 1 | mp_selector | PASS |
| MULW | MUL | 1 | **0** | mp_selector | **NOTE** |
| MULH | MUL | 0 | 1 | mp_sel+muldiv_sel | PASS |
| MULHU | MUL | 0 | 0 | muldiv_sel | PASS |
| MULHSU | MUL | 0 | 1 | muldiv_sel | PASS |
| DIV[U][W] | DIVREM | [W] | !U | | PASS |
| REM[U][W] | DIVREM | [W] | !U | muldiv_sel | PASS |
| LUI | ADD | 0 | 0 | rs1=0 | PASS |
| AUIPC | ADD | 0 | 0 | rs1=x255 | PASS |
| JAL | JALR | 0 | 0 | rs1=x255 | PASS |
| JALR | JALR | 0 | 0 | | PASS |
| BEQ/BNE | BEQ | 0 | 0 | BNE:mp_sel | PASS |
| BLT[U]/BGE[U] | BLT | 0 | !U | BGE:mp_sel | PASS |
| LD | LOAD | 0 | 0 | mem_8B | PASS |
| LW[U] | LOAD | 0 | !U | mem_4B | PASS |
| LH[U] | LOAD | 0 | !U | mem_2B | PASS |
| LB[U] | LOAD | 0 | !U | | PASS |
| SD/SW/SH/SB | STORE | 0 | 0 | mem flags | PASS |
| ECALL | ECALL | 0 | 0 | rs1=x17 | **NOTE** |
| FENCE | ADD | 0 | 0 | (no-op) | PASS |

**MULW note**: Spec says `MUL[W]` has `signed=1` always. Code sets `signed=false` for MULW (`if !is_word { signed=true }`). For 32-bit multiplication, the lower 32 bits are identical regardless of signed/unsigned interpretation, so this does not affect correctness. Still, it is a spec deviation.

**ECALL note**: Code sets `rs2=10` (a0) and `rd=10` (a0) for ECALL, but `read_register2=false` and `write_register=false`. The spec defaults rs2/rd to 0 when not specified. This changes the packed_decode value (bits 35-50). However, this is consistent with the CPU table which constructs the same packed_decode. Not a soundness issue but a spec deviation.

**read_register1 and x255**: Code excludes `rs1=255` from `read_register1` (types.rs:444, cpu.rs:750, trace_builder.rs:479). The spec text says `read_register1=1` when `rs1≠0`, which would include x255. Code deviates because x255 (PC) is not a physical register requiring a MEMW read — the CPU already has the PC value. This is consistent across all tables.

**c_type (compressed instructions)**: The c_type bit is defined but never set to `true` in actual decoding (`from_instruction`). RV64C support is not yet implemented. The bit position is reserved and correctly placed at position 6 per spec.

#### Bus Interactions

| Check | Status | Details |
|-------|--------|---------|
| DECODE receiver | PASS | BusId::Decode, Multiplicity::Column(MU), values: [pc(DWordWL), imm(DWordWL), packed_decode(Direct)] = 5 elements |
| Signature match | PASS | Spec: `DECODE[pc, imm, packed_decode]`, input=[DWordWL, DWordWL, BaseField] = 2+2+1 = 5 elements |
| No extra interactions | PASS | |

#### Padding

| Check | Status | Details |
|-------|--------|---------|
| Padding pattern | PASS | pc=7, EBREAK=1, all else 0. Per spec: "smallest odd number > 1+4" |
| CPU padding entry | PASS | pc=1 (`CPU_PADDING_PC`), all flags=0. Per spec: "pc=1 and every other variable set to 0" |
| Padding μ=0 | PASS | Bus interaction silent on padding rows |
| EBREAK trap safety | PASS | CPU asserts EBREAK=0, so padding rows are unprovable if referenced |

#### Precomputed Commitment

| Check | Status | Details |
|-------|--------|---------|
| Commitment computation | PASS | LDE of 5 precomputed cols → Merkle tree. Verifier recomputes from ELF. |
| Columns committed | PASS | PC_0, PC_1, PACKED_DECODE, IMM_0, IMM_1 (excludes MU) |

#### Summary

**DECODE TABLE: PASS**

All 6 columns, 51-bit packed_decode layout, 1 bus interaction (DECODE receiver from CPU), instruction-to-flags mapping for all RV64IM instructions, padding (pc=7, EBREAK=1), and CPU padding entry (pc=1, all zeros) match the spec. Precomputed commitment mechanism is sound (verifier recomputes from ELF).

Minor spec deviations (all internally consistent, not soundness issues):
- MULW: `signed=0` in code vs `signed=1` in spec (lower 32-bit multiplication unaffected)
- ECALL: `rs2=10, rd=10` in code vs `rs2=0, rd=0` in spec (consistent with CPU table)
- `read_register1` excludes `rs1=255` (consistent across CPU, DECODE, trace_builder)
- `c_type` (RV64C) not yet implemented (bit reserved at correct position)

---

### 12. CPU

**Spec**: `docs/spec/cpu.md` on `md_spec` branch
**Code**: `prover/src/tables/cpu.rs`, `prover/src/constraints/cpu.rs`

#### Columns

| Check | Status | Details |
|-------|--------|---------|
| Input columns match spec | PASS | timestamp(1), pc(DWordWL=2), rs1/rs2/rd(Byte=3), read_register1/2(Bit=2), write_register(Bit=1), memory_Xbytes(Bit=3), c_type(Bit=1), imm(DWordWL=2), signed/mp_selector/muldiv_selector/word_instr(Bit=4), 16 ALU flags(Bit=16) |
| Output columns match spec | PASS | next_pc(DWordWL=2), rvd(DWordWL=2) |
| Auxiliary columns match spec | PASS | rv1(DWordWHH=3), rv2(DWordWHH=3), rv1_sign_bit(1), arg1(DWordBL=8), arg2_sign_bit(1), arg2(DWordBL=8), res_sign_bit(1), res(DWordBL=8), is_equal(1), branch_cond(1) |
| Virtual columns match spec | PASS | packed_decode (linear combination of decode columns), pad (1 - sum of ALU flags) |
| Column count | PASS | NUM_COLUMNS = 74, matches spec total |
| Column ordering contiguous | PASS | 0-73, no gaps |
| No extra columns | PASS | |
| No missing columns | PASS | |

#### Polynomial Constraints

| Check | Status | Details |
|-------|--------|---------|
| CR2: IS_BIT\<read_register1\> | **FAIL** | **MISSING** — not in BIT_FLAG_COLUMNS list. See DEV-7. |
| CR3: IS_BIT\<read_register2\> | **FAIL** | **MISSING** — not in BIT_FLAG_COLUMNS list. See DEV-8. |
| CR4-CR28: IS_BIT for all other flags | PASS | 25 flags in BIT_FLAG_COLUMNS: write_register, memory flags(3), c_type, signed, mp_selector, muldiv_selector, word_instr, 16 ALU flags |
| Extra IS_BIT (not in spec) | NOTE | 5 extra IS_BIT for rv1_sign_bit, arg2_sign_bit, res_sign_bit, is_equal, branch_cond — overconstrained, harmless |
| CA35: ADD+LOAD → ADD\<res; arg1, arg2\> | PASS | `create_add_constraints` with cond=[ADD,LOAD], 2 carry constraints |
| CA36: STORE → ADD\<res; arg1, imm\> | PASS | STORE ADD with cond=[STORE], lhs=ARG1, rhs=IMM, sum=RES |
| CA37: SUB+BEQ → SUB\<res; arg1, arg2\> | PASS | `create_sub_constraints` verifies arg2+res=arg1, cond=[SUB,BEQ] |
| CA39.i: (SLT+BLT)\*res[i]=0 for i∈[1,7] | PASS | `SltResZeroConstraint` for bytes 1-7 |
| CA44: JALR → ADD\<res; pc, instr_size\> | PASS | `create_jalr_constraints` with instr_size = 4 - 2\*c_type |
| CE57: sign bits zero when word_instr=0 | PASS | `SignBitZeroConstraint`: (sum of sign bits)\*(1-word_instr)=0 |
| CE59: arg1[:4] = rv1[:2] | PASS | `Arg1LowerConstraint`: arg1_lo = rv1_0 + rv1_1\*2^16 |
| CE60: arg1[4:] = rv1[2]\*(1-word_instr) + (2^32-1)\*rv1_sign_bit\*signed | PASS | `Arg1UpperConstraint` matches spec formula |
| CE62: arg2[:4] = (1-LOAD)\*rv2[:2] + (1-BEQ-BLT-STORE)\*imm[0] | PASS | `Arg2LowerConstraint` matches spec formula |
| CE63: arg2[4:] = (1-LOAD)\*((1-word_instr)\*rv2[2] + signed\*arg2_sign_bit\*(2^32-1)) + (1-BEQ-BLT-STORE)\*imm[1] | PASS | `Arg2UpperConstraint` matches spec formula |
| CE65: (1-LOAD)\*(rvd[0]-res[:4])=0 | PASS | `RvdLowerConstraint` |
| CE66: (1-LOAD)\*(rvd[1]-(1-word_instr)\*res[4:]-res_sign_bit\*(2^32-1))=0 | PASS | `RvdUpperConstraint` |
| CM48: (1-read_register1)\*rv1[i]=0 for i∈[0,2] | **FAIL** | **MISSING** — no polynomial constraining rv1 to zero when not reading rs1. See DEV-9. |
| CM50: (1-read_register2)\*rv2[i]=0 for i∈[0,2] | **FAIL** | **MISSING** — no polynomial constraining rv2 to zero when not reading rs2. See DEV-10. |
| CS55: EBREAK = 0 | PASS | `EbreakConstraint` correctly enforces EBREAK=0. Note: spec says "1-EBREAK=0" which would mean EBREAK=1 — likely spec typo. Code follows intent. |
| CO68: branch_cond formula | PASS | `BranchCondConstraint`: JALR + BLT\*(res[0] XOR mp_selector) + BEQ\*(is_equal XOR mp_selector) |
| CO70: ADD\<next_pc; pc, instr_size\> | PASS | `NextPcAddConstraint` with condition (1-branch_cond). Spec doesn't show explicit condition but code is correct — unconditional would fail on branching rows. |
| Constraint count | PASS | NUM_CPU_CONSTRAINTS = 56 (30 IS_BIT + 8 ADD/SUB carries + 1 branch_cond + 1 EBREAK + 2 arg1 + 2 arg2 + 2 rvd + 7 SLT zero + 1 sign_bit_zero + 2 next_pc) |

#### Bus Interactions

| Check | Status | Details |
|-------|--------|---------|
| C1: DECODE[pc, imm, packed_decode] \| 1 | PASS | Multiplicity::One, packed_decode as linear combination of all decode columns using correct bit positions |
| CR29: IS_BYTE[rs1] \| 1 | PASS | Multiplicity::One |
| CR30: IS_BYTE[rs2] \| 1 | PASS | Multiplicity::One |
| CR31: IS_BYTE[rd] \| 1 | PASS | Multiplicity::One |
| CR32.i: IS_BYTE[arg1[i]] \| 1 (×8) | PASS | 8 interactions, Multiplicity::One |
| CR33.i: IS_BYTE[arg2[i]] \| 1 (×8) | PASS | 8 interactions, Multiplicity::One |
| CR34.i: IS_BYTE[res[i]] \| 1 (×8) | PASS | 8 interactions, Multiplicity::One |
| CA38: LT[res[0]; arg1, arg2, signed] \| SLT+BLT | PASS | BusId::Lt, Multiplicity::Sum(SLT,BLT), arg1/arg2 as DWordBL (compatible with LT receiver DWordHHW: both → 2 words) |
| CA40.i: AND_BYTE[res[i]; arg1[i], arg2[i]] \| AND (×8) | PASS | BusId::AndByte, Multiplicity::Column(AND) |
| CA41.i: OR_BYTE[res[i]; arg1[i], arg2[i]] \| OR (×8) | PASS | BusId::OrByte, Multiplicity::Column(OR) |
| CA42.i: XOR_BYTE[res[i]; arg1[i], arg2[i]] \| XOR (×8) | PASS | BusId::XorByte, Multiplicity::Column(XOR) |
| CA43: SHIFT[res; arg1, arg2[0], mp_selector, signed, word_instr] \| SHIFT | PASS | BusId::Shift, Multiplicity::Column(SHIFT), 6 values matching signature |
| CA45: MUL[res; arg1, signed, arg2, mp_selector, muldiv_selector] \| MUL | **DEV** | BusId::Mul, Multiplicity::Column(MUL). **Code sends rvd instead of res.** See DEV-11. |
| CA46: DVRM[res; arg1, arg2, signed, muldiv_selector] \| DIVREM | **DEV** | BusId::Dvrm, Multiplicity::Column(DIVREM). **Code sends rvd instead of res.** See DEV-12. |
| CM47: MEMW read rs1 (24 elems) \| read_register1 | PASS | old=rv1 as WL+6zeros, is_reg=1, addr=2\*rs1, val=rv1, ts=timestamp+0, write2=1 |
| CM49: MEMW read rs2 (24 elems) \| read_register2 | PASS | old=rv2 as WL+6zeros, is_reg=1, addr=2\*rs2, val=rv2, ts=timestamp+1, write2=1 |
| CM51: MEMW write rd (16 elems) \| write_register | PASS | is_reg=1, addr=2\*rd, val=rvd as WL+6zeros, ts=timestamp+2, write2=1 |
| CM52: LOAD[rvd; res, timestamp, memory_flags, signed] \| LOAD | PASS | BusId::Load, 10 elements matching LOAD signature |
| CM53: MEMW write memory (16 elems) \| STORE | PASS | is_reg=0, addr=res::DWordBL, val=arg2 (8 bytes), ts=timestamp+1, write flags from decode |
| CM54: MEMW PC register (24 elems) \| 1-pad | PASS | old=pc, is_reg=1, addr=510 (2\*255), val=next_pc, ts=timestamp+1, write2=1. Multiplicity = Linear(sum of all 16 ALU flags) = 1-pad. |
| CS56: ECALL[timestamp, rv1] \| ECALL | PASS | BusId::Ecall, [timestamp::DWordWL, rv1::DWordWL], Multiplicity::Column(ECALL) |
| CE58: MSB16[rv1_sign_bit; rv1[1]] \| word_instr | PASS | BusId::Msb16, Multiplicity::Column(WORD_INSTR) |
| CE61: MSB16[arg2_sign_bit; rv2[1]] \| word_instr | PASS | BusId::Msb16, Multiplicity::Column(WORD_INSTR) |
| CE64: MSB8[res_sign_bit; res[3]] \| word_instr | PASS | BusId::Msb8, Multiplicity::Column(WORD_INSTR) |
| CO67: ZERO[is_equal; sum(res[0..7])] \| BEQ | PASS | BusId::Zero, linear sum of 8 res bytes, Multiplicity::Column(BEQ) |
| CO69: BRANCH[next_pc; pc, imm, arg1, JALR] \| branch_cond | PASS | BusId::Branch, 9 elements, arg1 repacked from bytes to words via Linear, Multiplicity::Column(BRANCH_COND) |
| Bus interaction count | PASS | 1 DECODE + 27 IS_BYTE + 1 LT + 24 AND/OR/XOR + 1 SHIFT + 1 MUL + 1 DVRM + 5 MEMW + 1 LOAD + 1 ECALL + 3 MSB/ZERO + 1 BRANCH = ~67 interactions |

#### Padding

| Check | Status | Details |
|-------|--------|---------|
| Padding PC | PASS | CPU_PADDING_PC = 1 (odd address, unreachable) |
| Padding next_pc | PASS | next_pc = 5 = 1 + 4. NextPcAdd carry = (1+4-5)/2^32 = 0. ✓ |
| All flags = 0 on padding | PASS | pad = 1, no ALU interactions fire |
| IS_BYTE still fires on padding | PASS | Multiplicity::One, so 27 IS_BYTE lookups fire for zero-valued columns |
| DECODE fires on padding | PASS | Multiplicity::One, DECODE table has entry at pc=1 |
| CM54 (PC MEMW) does NOT fire | PASS | Multiplicity = sum(ALU flags) = 0 on padding rows |
| No constraints break on padding | PASS | All polynomial constraints hold: arg1/arg2/res/rvd all zero, sign bits zero, branch_cond=0 |

#### Trace Generation

| Check | Status | Details |
|-------|--------|---------|
| Column filling | PASS | All 74 columns correctly filled from CpuOperation fields |
| rv1 as DWordWHH | PASS | [rv1&0xFFFF, (rv1>>16)&0xFFFF, rv1>>32] |
| rv2 as DWordWHH | PASS | Same pattern |
| arg1 computation | PASS | `compute_arg1()`: pass-through for 64-bit, sign/zero-extend for word_instr |
| arg2 computation | PASS | `compute_arg2()`: LOAD→imm, STORE→rv2, BEQ/BLT→rv2, else→imm or rv2 with word extension |
| res computation | PASS | `compute_res()`: ADD/LOAD→wrapping_add, STORE→arg1+imm, SUB→wrapping_sub, SHIFT→raw shift, else→executor result |
| rvd computation | PASS | LOAD→executor value, else→`compute_rvd()` (res with word sign extension) |
| read_register1 excludes x0 and x255 | NOTE | Matches DEV-5 (spec says rs1≠0, code also excludes rs1=255) |
| Timestamp stride | PASS | 4 slots per CPU row (timestamp = i * 4) |
| ECALL next_pc override | PASS | Forces next_pc = pc + 4 even though executor sets next_pc=0 |
| AUIPC/JAL rv1 = pc | PASS | When rs1=255, rv1 = current_pc from executor log |
| Padding rows | PASS | pc=1, next_pc=5, all other columns zero |

#### Summary

**CPU TABLE: FINDINGS — 4 CRITICAL missing constraints, 2 MODERATE deviations**

The CPU table has 74 columns (matching spec), 56 polynomial constraints, and ~67 bus interactions. Column layout, ALU dispatch, memory interactions, extension constraints, branch condition, and next_pc computation all match the spec.

**CRITICAL findings** (potential soundness issues):
- **DEV-7/8**: Missing IS_BIT for `read_register1` and `read_register2`. These multiplicity columns are unconstrained to binary values. The packed_decode provides only one equation for two unknowns, leaving a degree of freedom exploitable by a dishonest prover.
- **DEV-9/10**: Missing rv1/rv2 zero-forcing polynomials (`CM48`, `CM50`). When `read_register1=0`, rv1 is completely unconstrained — no MEMW read and no zero polynomial. The prover can forge rv1 for instructions using x0 (zero register) or x255 (virtual PC). Same for rv2 when `read_register2=0`.

**MODERATE deviations** (functional difference, needs cross-table verification):
- **DEV-11/12**: MUL and DVRM interactions send `rvd` (sign-extended for word instructions) instead of `res` (raw result). For word instructions (MULW, DIVW, REMW), rvd differs from the chip's computed result in the upper 32 bits.

---

## Spec Deviations

See **[DEVIATIONS.md](DEVIATIONS.md)** for the detailed report. 12 deviations found across DVRM (2), Decode (4), and CPU (6). CPU has 4 CRITICAL and 2 MODERATE findings.
