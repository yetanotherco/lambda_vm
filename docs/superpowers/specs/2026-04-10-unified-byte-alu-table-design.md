# Unified Byte-ALU Table (ZisK-style BinaryTable)

## Problem

Every CPU row sends 27 IS_BYTE range-check lookups (3 register indices +
8 arg1 bytes + 8 arg2 bytes + 8 res bytes). For bitwise ops, 8 additional
AND_BYTE/OR_BYTE/XOR_BYTE lookups are sent. These are redundant — the byte-op
lookup already proves the operands are bytes.

Total per AND instruction: 27 IS_BYTE + 8 AND_BYTE = 35 bus interactions.
With LogUp batching: ~18 CPU aux columns.

## Goal

Replace IS_BYTE + AND_BYTE + OR_BYTE + XOR_BYTE with a single unified
byte-ALU lookup table (ZisK's BinaryTable approach). Each lookup:
- Range-checks both operands (implicit from table construction)
- Verifies the byte-level operation result
- Provides carry/flag outputs for ADD/SUB

Target: reduce CPU bus interactions from ~35 to ~11 per instruction.

## Design

### Unified Byte-ALU Table

Extend the existing Bitwise table (256 × 256 × 16 = 2^20 rows, indexed by
X, Y, Z) to serve as a unified byte-ALU table. Current columns:

```
X, Y, Z, AND, OR, XOR, MSB8, MSB16, ZERO, SLL, SLLC
+ multiplicity columns MU_AND, MU_OR, MU_XOR, MU_MSB8, ...
```

Add new columns for ADD/SUB:
```
ADD_LO    = (X + Y) & 0xFF          // ADD result low byte (no carry-in)
ADD_CIN_LO = (X + Y + 1) & 0xFF    // ADD result low byte (carry-in = 1)
ADD_CARRY  = (X + Y) >> 8           // carry-out (0 or 1, no carry-in)
ADD_CIN_CARRY = (X + Y + 1) >> 8   // carry-out (carry-in = 1)
SUB_LO    = (X - Y) & 0xFF          // SUB result (mod 256)
SUB_BORROW = ((X as i16 - Y as i16) < 0) as u8  // borrow flag
```

The Z column (shift amount, 0-15) is irrelevant for byte-ALU ops but doesn't
interfere — the same (X, Y) pair appears for all Z values, and the ALU
columns are the same across all Z.

### New Bus: ByteAlu

Replace `BusId::IsBytes` with `BusId::ByteAlu`. The message format:

```
ByteAlu[X, Y, op_type, result]
```

Where:
- op_type = 0 (AND), 1 (OR), 2 (XOR), 3 (ADD no carry), 4 (ADD with carry),
  5 (SUB), 6 (range-check only — result = X, used for rs1/rs2/rd)
- result = the operation result byte

For op_type 3-4 (ADD): result also encodes the carry-out as a separate field:
```
ByteAlu_Add[X, Y, carry_in, result_lo, carry_out]
```

### CPU Changes

**Eliminate 24 IS_BYTE lookups** for arg1[0..7], arg2[0..7], res[0..7].
These are replaced by the byte-ALU lookups which implicitly range-check.

**For AND/OR/XOR instructions (currently 8 AND_BYTE + 24 IS_BYTE = 32):**
Replace with 8 ByteAlu[arg1[i], arg2[i], op, res[i]] = 8 lookups total.

**For ADD/SUB instructions (currently 24 IS_BYTE + inline carry constraints):**
Replace with 8 ByteAlu_Add[arg1[i], arg2[i], carry_in[i], res[i], carry_out[i]]
= 8 lookups total. Carry chain: carry_in[0] = 0, carry_in[i] = carry_out[i-1].
The carry values are either committed as 7 columns or constrained inline.

**For other instructions (SLT, branch, load, store):**
Replace 24 IS_BYTE with 8 ByteAlu[arg1[i], arg2[i], RANGE_CHECK, res[i]]
= 8 lookups. These prove all three bytes are in range without a separate
operation check.

**Register index range checks (rs1, rs2, rd):**
Keep 3 IS_BYTE lookups for register indices (5-bit values, IS_BYTE is
sufficient). Alternatively, use the byte-ALU table with a RANGE_CHECK op_type.

### Net Bus Interaction Reduction

| Instruction | Before | After | Savings |
|-------------|--------|-------|---------|
| AND/OR/XOR | 27 IS_BYTE + 8 byte-op = 35 | 8 ByteAlu + 3 IS_BYTE = 11 | -24 |
| ADD/SUB | 27 IS_BYTE = 27 | 8 ByteAlu_Add + 3 IS_BYTE = 11 | -16 |
| SLT/Branch | 27 IS_BYTE = 27 | 8 ByteAlu + 3 IS_BYTE = 11 | -16 |
| Load/Store | 27 IS_BYTE = 27 | 8 ByteAlu + 3 IS_BYTE = 11 | -16 |

### Impact on Aux Columns

With batching (pairs of 2):
- Before: ~(35/2 + 1) ≈ 18-20 aux columns for bitwise instructions
- After: ~(11/2 + 1) ≈ 6-7 aux columns
- **Saves ~12 aux columns** → 12 fewer extension-field FFTs + commits

### Bitwise Table Changes

Current: 21 columns (11 precomputed + 10 multiplicity)
After: ~27 columns (17 precomputed + 10 multiplicity)

New precomputed columns: ADD_LO, ADD_CIN_LO, ADD_CARRY, ADD_CIN_CARRY,
SUB_LO, SUB_BORROW = 6 new columns.

New multiplicity columns: MU_BYTE_ALU replaces separate MU_AND, MU_OR,
MU_XOR for the merged ByteAlu bus (using op_type discriminant, same as
the BitwiseByte merge from feat/merged-bitwise-bus).

## Files Changed

| File | Change |
|------|--------|
| `prover/src/tables/bitwise.rs` | Add ADD/SUB columns, ByteAlu receiver |
| `prover/src/tables/cpu.rs` | Replace 24 IS_BYTE + 8 byte-op with 8 ByteAlu |
| `prover/src/tables/types.rs` | Add ByteAlu BusId |
| `prover/src/tables/trace_builder.rs` | Update bitwise collection |
| Other tables using IS_BYTE | Route through ByteAlu where applicable |

## Implementation Note

This can be combined with the BitwiseByte merge (from feat/merged-bitwise-bus)
which already merged AND_BYTE/OR_BYTE/XOR_BYTE into one bus with op_type. The
ByteAlu bus extends that merge to also cover ADD/SUB/range-check operations.
