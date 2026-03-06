# Unified BitwiseByte Bus Design

## Problem

The CPU table currently uses 24 separate bus interactions for byte-wise bitwise operations: 8 for AND, 8 for OR, and 8 for XOR. Each interaction has a 3-element signature `(X, Y, result)` and maps to a separate bus ID (`AndByte`, `OrByte`, `XorByte`). These three buses share identical structure — only the operation differs.

With LogUp batching (⌈N/2⌉ formula), the CPU's 40 total interactions produce 20 auxiliary extension columns. Each extension column is 3 Goldilocks field elements wide, making bitwise operations the dominant contributor to CPU trace width.

## Solution

Replace `AndByte`, `OrByte`, `XorByte` (3 bus IDs) with a single `BitwiseByte` bus. The fingerprint signature becomes `(opcode, X, Y, result)` — 4 elements instead of 3, but 8 interactions total instead of 24.

The opcode encoding reuses the BITWISE table's existing Z column:
- Z = 0 → AND
- Z = 1 → OR
- Z = 2 → XOR
- Z ≥ 3 → reserved for shift operations (unchanged)

This is inspired by ZisK's Binary SM which uses a single bus with a mode selector.

## Detailed Changes

### 1. BusId Enum (`prover/src/tables/types.rs`)

Remove `AndByte = 3`, `OrByte = 4`, `XorByte = 5`. Add `BitwiseByte = 3`. Other IDs remain unchanged (keep gaps for simplicity).

### 2. BITWISE Table (`prover/src/tables/bitwise.rs`)

#### Column Layout (22 → 21 columns)

Add precomputed `BITWISE_RESULT` column (index 11):
- Z=0: BITWISE_RESULT = AND(X, Y)
- Z=1: BITWISE_RESULT = OR(X, Y)
- Z=2: BITWISE_RESULT = XOR(X, Y)
- Z≥3: BITWISE_RESULT = 0 (unused for this bus)

Remove multiplicity columns `MU_OR` (was 12) and `MU_XOR` (was 13). Rename `MU_AND` → `MU_BITWISE_BYTE`. Renumber remaining multiplicity columns.

New column layout:
| Index | Column | Description |
|-------|--------|-------------|
| 0-10 | X, Y, Z, AND, OR, XOR, MSB8, MSB16, ZERO, SLL, SLLC | Precomputed (unchanged) |
| 11 | BITWISE_RESULT | AND/OR/XOR result selected by Z |
| 12 | MU_BITWISE_BYTE | Unified multiplicity for BitwiseByte |
| 13 | MU_MSB8 | (was 14) |
| 14 | MU_MSB16 | (was 15) |
| 15 | MU_ZERO | (was 16) |
| 16 | MU_IS_BYTE | (was 17) |
| 17 | MU_IS_HALF | (was 18) |
| 18 | MU_IS_B20 | (was 19) |
| 19 | MU_HWSL | (was 20) |
| 20 | MU_HWSLC | (was 21) |

Total: 21 columns (was 22). `NUM_PRECOMPUTED_COLS` = 12 (was 11).

#### Bus Interactions (11 → 9)

Remove 3 receivers (`AndByte`, `OrByte`, `XorByte`). Add 1 receiver:

```
BitwiseByte receiver:
  bus_id: BitwiseByte
  multiplicity: Column(MU_BITWISE_BYTE)
  values: [Z (Direct), X (Direct), Y (Direct), BITWISE_RESULT (Direct)]
```

#### generate_bitwise_row

Add BITWISE_RESULT to the precomputed output array. When Z=0 return AND, Z=1 return OR, Z=2 return XOR, else 0.

#### update_multiplicities

`BitwiseOperationType::BitwiseByte` operations increment `MU_BITWISE_BYTE` at the standard row index `x + y*256 + opcode*65536` where opcode ∈ {0, 1, 2}.

#### Hardcoded Commitment

Must be regenerated after column layout change.

### 3. CPU Table (`prover/src/tables/cpu.rs`)

#### Bus Interactions (40 → 24)

Remove 24 interactions (8 AndByte + 8 OrByte + 8 XorByte). Add 8 BitwiseByte interactions:

```
for i in 0..8:
  BitwiseByte sender:
    bus_id: BitwiseByte
    multiplicity: Linear([AND*1, OR*1, XOR*1])  // = AND+OR+XOR, at most one is 1
    values: [
      Linear([OR*1, XOR*2]),   // opcode: AND→0, OR→1, XOR→2
      ARG1[i] (Direct),
      ARG2[i] (Direct),
      RES[i] (Direct),
    ]
```

The opcode is encoded as `OR + 2*XOR` using a `BusValue::Linear` — when AND=1, both OR and XOR are 0, so opcode=0; when OR=1, opcode=1; when XOR=1, opcode=2.

#### collect_bitwise_ops

Merge all three AND/OR/XOR branches into one emitting `BitwiseOperationType::BitwiseByte` with the appropriate Z value (0, 1, or 2).

### 4. BitwiseOperationType Enum

Remove `AndByte`, `OrByte`, `XorByte`. Add `BitwiseByte`. The `byte_op` constructor takes the opcode as the Z value.

### 5. Other Tables

No other tables send AND/OR/XOR lookups — only CPU does. Other tables using IsHalfword, IsByte, MSB8, MSB16, Zero, etc. are unaffected.

### 6. Tests

Update bitwise bus tests to use `BitwiseByte` bus ID. Verify all three operations (AND, OR, XOR) go through the unified bus. Existing program-level tests should pass unchanged.

## Savings

| Table | Main cols | Aux cols (extension) | Goldilocks cols saved |
|-------|-----------|---------------------|-----------------------|
| CPU | 0 | -8 (20→12) | 24 |
| BITWISE | -1 (22→21) | -1 (6→5) | 4 |
| **Total** | **-1** | **-9** | **28** |

CPU effective width: 194 → 170 (-12.4%)

## Risks

- **Hardcoded commitment regeneration**: The BITWISE preprocessed commitment changes. Must update the cached value.
- **Fingerprint width**: BitwiseByte fingerprint has 4 elements (was 3). This slightly increases fingerprint computation cost per interaction, but we have 8 interactions instead of 24 — net win.
- **Opcode correctness**: Must verify that AND rows consistently have Z=0, OR have Z=1, XOR have Z=2 in the BITWISE table. The precomputed table generation ensures this by construction.
