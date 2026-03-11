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
| 6 | DVRM | 22 | | Uses SIGN + NEG |
| 7 | Shift | 15 | | |
| 8 | MEMW | 25 | | Memory argument core |
| 9 | HALT | ~10 | | Part of ecall.md |
| 10 | COMMIT | ~12 | | Part of ecall.md |
| 11 | Decode | special | | Preprocessed, bit layout |
| 12 | CPU | ~70 | | Largest table, audit last |

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
