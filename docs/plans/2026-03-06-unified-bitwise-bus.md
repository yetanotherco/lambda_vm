# Unified BitwiseByte Bus Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Merge AND/OR/XOR into a single BitwiseByte bus, reducing CPU trace width by 12.4% (28 fewer Goldilocks columns).

**Architecture:** Replace 3 separate bus IDs (`AndByte`, `OrByte`, `XorByte`) with 1 (`BitwiseByte`) using an opcode field encoded in the existing Z column (0=AND, 1=OR, 2=XOR). CPU sends 8 interactions instead of 24. BITWISE adds a `BITWISE_RESULT` precomputed column and merges 3 multiplicity columns into 1.

**Tech Stack:** Rust, Goldilocks field, LogUp lookup arguments, STARK prover

**Design doc:** `docs/plans/2026-03-06-unified-bitwise-bus-design.md`

---

### Task 1: Update BusId Enum

**Files:**
- Modify: `prover/src/tables/types.rs:42-168`

**Step 1: Update the BusId enum**

Replace `AndByte`, `OrByte`, `XorByte` with `BitwiseByte`:

```rust
// In the enum definition (line 53-61):
// Replace:
    /// Bitwise AND of two bytes: AND_BYTE[X, Y] -> X & Y
    AndByte,
    /// Bitwise OR of two bytes: OR_BYTE[X, Y] -> X | Y
    OrByte,
    /// Bitwise XOR of two bytes: XOR_BYTE[X, Y] -> X ^ Y
    XorByte,

// With:
    /// Unified bitwise byte operation: BITWISE_BYTE[opcode, X, Y] -> result
    /// Opcode: 0=AND, 1=OR, 2=XOR (encoded in BITWISE table's Z column)
    BitwiseByte,
    // IDs 4 and 5 are unused (gap from removed OrByte/XorByte)
```

**Step 2: Update the `name()` method**

```rust
// Replace the three match arms (lines 118-120):
    BusId::AndByte => "AndByte",
    BusId::OrByte => "OrByte",
    BusId::XorByte => "XorByte",

// With:
    BusId::BitwiseByte => "BitwiseByte",
```

**Step 3: Update `TryFrom<u64>`**

```rust
// Replace lines 148-150:
    3 => Ok(BusId::AndByte),
    4 => Ok(BusId::OrByte),
    5 => Ok(BusId::XorByte),

// With:
    3 => Ok(BusId::BitwiseByte),
    // 4 and 5 are unused gaps
```

**Step 4: Fix the discriminant gap**

Since `BitwiseByte` takes value 3 and `Msb8` follows, `Msb8` would become 4 by default. But we need `Msb8 = 6` to maintain the same bus IDs for everything else. Add explicit discriminant:

```rust
    BitwiseByte,
    /// Most significant bit of a byte: MSB8[X] -> (X >> 7) & 1
    Msb8 = 6,
```

This ensures `Msb8 = 6, Msb16 = 7, Zero = 8, ...` are unchanged.

**Step 5: Run compilation check**

Run: `cargo check -p prover 2>&1 | head -50`

Expected: Compilation errors in files that reference `AndByte`, `OrByte`, `XorByte` — this is expected and will be fixed in subsequent tasks.

**Step 6: Commit**

```
git add prover/src/tables/types.rs
git commit -m "refactor: replace AndByte/OrByte/XorByte with unified BitwiseByte bus ID"
```

---

### Task 2: Update BITWISE Table Columns and Row Generation

**Files:**
- Modify: `prover/src/tables/bitwise.rs:50-161` (cols module, generate_bitwise_row, NUM_PRECOMPUTED_COLS)

**Step 1: Update column layout**

Replace the `cols` module (lines 50-101):

```rust
pub mod cols {
    /// X: Byte input (0-255)
    pub const X: usize = 0;
    /// Y: Byte input (0-255)
    pub const Y: usize = 1;
    /// Z: 4-bit input (0-15) for shift amount / bitwise opcode
    pub const Z: usize = 2;

    /// AND result: X & Y
    pub const AND: usize = 3;
    /// OR result: X | Y
    pub const OR: usize = 4;
    /// XOR result: X ^ Y
    pub const XOR: usize = 5;
    /// MSB of byte X: (X >> 7) & 1
    pub const MSB8: usize = 6;
    /// MSB of halfword (X + 256*Y): ((X + 256*Y) >> 15) & 1
    pub const MSB16: usize = 7;
    /// Zero check: (X == 0 && Y == 0) ? 1 : 0
    pub const ZERO: usize = 8;
    /// Shift left result: ((X + 256*Y) << Z) & 0xFFFF
    pub const SLL: usize = 9;
    /// Shift left carry: (X + 256*Y) >> (16 - Z)
    pub const SLLC: usize = 10;

    /// Bitwise result selected by Z: AND when Z=0, OR when Z=1, XOR when Z=2
    pub const BITWISE_RESULT: usize = 11;

    // Multiplicity columns for each lookup type
    /// Multiplicity for unified BitwiseByte lookups (AND/OR/XOR)
    pub const MU_BITWISE_BYTE: usize = 12;
    /// Multiplicity for MSB8 lookups
    pub const MU_MSB8: usize = 13;
    /// Multiplicity for MSB16 lookups
    pub const MU_MSB16: usize = 14;
    /// Multiplicity for ZERO lookups
    pub const MU_ZERO: usize = 15;
    /// Multiplicity for IS_BYTE lookups
    pub const MU_IS_BYTE: usize = 16;
    /// Multiplicity for IS_HALF lookups
    pub const MU_IS_HALF: usize = 17;
    /// Multiplicity for IS_B20 lookups
    pub const MU_IS_B20: usize = 18;
    /// Multiplicity for HWSL lookups
    pub const MU_HWSL: usize = 19;
    /// Multiplicity for HWSLC lookups
    pub const MU_HWSLC: usize = 20;

    /// Total number of columns
    pub const NUM_COLUMNS: usize = 21;
}
```

**Step 2: Update `NUM_PRECOMPUTED_COLS`**

```rust
// Change line 107:
pub const NUM_PRECOMPUTED_COLS: usize = 12; // was 11
```

**Step 3: Update `generate_bitwise_row`**

Add the BITWISE_RESULT computation and update the return array. Change the doc comment, the array type, and the return value:

```rust
/// Returns the 12 precomputed columns: [X, Y, Z, AND, OR, XOR, MSB8, MSB16, ZERO, SLL, SLLC, BITWISE_RESULT]
#[inline]
pub const fn generate_bitwise_row(index: usize) -> [u64; NUM_PRECOMPUTED_COLS] {
    // ... existing computation unchanged ...

    // Bitwise result selected by opcode (Z)
    let bitwise_result = if z == 0 {
        and_val
    } else if z == 1 {
        or_val
    } else if z == 2 {
        xor_val
    } else {
        0 // unused for shift operations
    };

    [
        x as u64,              // X
        y as u64,              // Y
        z as u64,              // Z
        and_val as u64,        // AND
        or_val as u64,         // OR
        xor_val as u64,        // XOR
        msb8 as u64,           // MSB8
        msb16 as u64,          // MSB16
        is_zero as u64,        // ZERO
        sll as u64,            // SLL
        sllc as u64,           // SLLC
        bitwise_result as u64, // BITWISE_RESULT
    ]
}
```

**Step 4: Update `generate_bitwise_trace`**

Add the BITWISE_RESULT line after the SLLC line (around line 346):

```rust
                data[base + cols::SLLC] = FE::from(sllc as u64);

                // Bitwise result selected by opcode (Z)
                let bitwise_result = match z {
                    0 => x & y,
                    1 => x | y,
                    2 => x ^ y,
                    _ => 0,
                };
                data[base + cols::BITWISE_RESULT] = FE::from(bitwise_result as u64);
```

**Step 5: Compile check**

Run: `cargo check -p prover 2>&1 | head -50`

Expected: Should compile (with warnings about unused `MU_AND`/`MU_OR`/`MU_XOR` being gone — these are handled in the next tasks).

**Step 6: Commit**

```
git add prover/src/tables/bitwise.rs
git commit -m "feat: add BITWISE_RESULT column and update column layout for unified bus"
```

---

### Task 3: Update BitwiseOperationType and Multiplicities

**Files:**
- Modify: `prover/src/tables/bitwise.rs:371-541` (update_multiplicities, BitwiseOperationType, BitwiseOperation)

**Step 1: Update `BitwiseOperationType` enum**

Replace `AndByte, OrByte, XorByte` with `BitwiseByte`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BitwiseOperationType {
    BitwiseByte,
    Msb8,
    Msb16,
    Zero,
    IsByte,
    IsHalf,
    IsB20,
    Hwsl,
    Hwslc,
}
```

**Step 2: Update `update_multiplicities`**

Replace the match arms for `AndByte`/`OrByte`/`XorByte` with `BitwiseByte`:

```rust
pub fn update_multiplicities(
    trace: &mut TraceTable<GoldilocksField, GoldilocksExtension>,
    ops: &[BitwiseOperation],
) {
    for op in ops {
        let row = row_index(op.x, op.y, op.z);
        let mu_col = match op.lookup_type {
            BitwiseOperationType::BitwiseByte => cols::MU_BITWISE_BYTE,
            BitwiseOperationType::Msb8 => cols::MU_MSB8,
            BitwiseOperationType::Msb16 => cols::MU_MSB16,
            BitwiseOperationType::Zero => cols::MU_ZERO,
            BitwiseOperationType::IsByte => cols::MU_IS_BYTE,
            BitwiseOperationType::IsHalf => cols::MU_IS_HALF,
            BitwiseOperationType::IsB20 => cols::MU_IS_B20,
            BitwiseOperationType::Hwsl => cols::MU_HWSL,
            BitwiseOperationType::Hwslc => cols::MU_HWSLC,
        };

        let current = trace.main_table.get_row(row)[mu_col];
        trace.set_main(row, mu_col, current + FE::one());
    }
}
```

**Step 3: Update `BitwiseOperation::byte_op`**

The key change: `byte_op` now uses the Z field for the opcode. For AND (opcode=0), OR (opcode=1), XOR (opcode=2):

```rust
    /// Create an operation for byte ops (AND, OR, XOR).
    /// The opcode is encoded in the Z field: 0=AND, 1=OR, 2=XOR.
    pub fn byte_op(opcode: u8, x: u8, y: u8) -> Self {
        debug_assert!(opcode <= 2, "Bitwise opcode must be 0 (AND), 1 (OR), or 2 (XOR)");
        Self::new(BitwiseOperationType::BitwiseByte, x, y, opcode)
    }
```

Note: The old signature was `byte_op(lookup_type: BitwiseOperationType, x: u8, y: u8)` with `z=0`. Now it takes `opcode: u8` directly since the operation type is always `BitwiseByte`.

**Step 4: Add opcode constants**

Add at the top of the file (after `pub const NUM_PRECOMPUTED_COLS`):

```rust
/// Opcode for AND in the unified BitwiseByte bus
pub const BITWISE_OP_AND: u8 = 0;
/// Opcode for OR in the unified BitwiseByte bus
pub const BITWISE_OP_OR: u8 = 1;
/// Opcode for XOR in the unified BitwiseByte bus
pub const BITWISE_OP_XOR: u8 = 2;
```

**Step 5: Update `trim_zero_rows`**

Update the range check for multiplicity columns (was `MU_AND..=MU_HWSLC`, now `MU_BITWISE_BYTE..=MU_HWSLC`):

```rust
            (cols::MU_BITWISE_BYTE..=cols::MU_HWSLC).any(|col| row_data[col] != FE::zero())
```

**Step 6: Commit**

```
git add prover/src/tables/bitwise.rs
git commit -m "refactor: unify AND/OR/XOR into BitwiseByte operation type with opcode encoding"
```

---

### Task 4: Update BITWISE Bus Interactions

**Files:**
- Modify: `prover/src/tables/bitwise.rs:552-769` (bus_interactions function)

**Step 1: Replace 3 receiver interactions with 1**

Replace the `AndByte`, `OrByte`, `XorByte` receivers (lines 553-610) with a single `BitwiseByte` receiver:

```rust
pub fn bus_interactions() -> Vec<BusInteraction> {
    vec![
        // BitwiseByte[Z, X, Y] -> BITWISE_RESULT
        // Z encodes opcode: 0=AND, 1=OR, 2=XOR
        BusInteraction::receiver(
            BusId::BitwiseByte,
            Multiplicity::Column(cols::MU_BITWISE_BYTE),
            vec![
                BusValue::Packed {
                    start_column: cols::Z,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::X,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::Y,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::BITWISE_RESULT,
                    packing: Packing::Direct,
                },
            ],
        ),
        // MSB8[X] -> MSB8  (unchanged)
        // ... rest of interactions unchanged ...
    ]
}
```

Keep all other interactions (MSB8, MSB16, ZERO, IsByte, IsHalfword, IsB20, HWSL, HWSLC) exactly the same. Only their multiplicity column indices have shifted due to the renumbering in Task 2 — but since they reference `cols::MU_MSB8` etc. by name (not by number), they auto-adjust.

**Step 2: Compile check**

Run: `cargo check -p prover 2>&1 | head -50`

**Step 3: Commit**

```
git add prover/src/tables/bitwise.rs
git commit -m "feat: replace 3 AND/OR/XOR receivers with single BitwiseByte receiver"
```

---

### Task 5: Update CPU Table Bus Interactions and Trace Building

**Files:**
- Modify: `prover/src/tables/cpu.rs:510-1003` (collect_bitwise_ops, bus_interactions)

**Step 1: Update `collect_bitwise_ops`**

Replace the three separate AND/OR/XOR branches (lines 514-548) with a unified branch:

```rust
        // AND/OR/XOR lookups (×8 each for each byte, unified via BitwiseByte bus)
        use super::bitwise::{BITWISE_OP_AND, BITWISE_OP_OR, BITWISE_OP_XOR};

        let opcode = if self.decode.op_and {
            Some(BITWISE_OP_AND)
        } else if self.decode.op_or {
            Some(BITWISE_OP_OR)
        } else if self.decode.op_xor {
            Some(BITWISE_OP_XOR)
        } else {
            None
        };

        if let Some(opcode) = opcode {
            for i in 0..8 {
                let a = ((arg1 >> (i * 8)) & 0xFF) as u8;
                let b = ((arg2 >> (i * 8)) & 0xFF) as u8;
                lookups.push(BitwiseOperation::byte_op(opcode, a, b));
            }
        }
```

**Step 2: Update CPU `bus_interactions` — remove 24, add 8**

Replace the three loops (lines 936-1003) with a single loop:

```rust
    // -------------------------------------------------------------------------
    // BitwiseByte interactions (×8 for each byte, unified AND/OR/XOR)
    // -------------------------------------------------------------------------
    for i in 0..8 {
        interactions.push(BusInteraction::sender(
            BusId::BitwiseByte,
            // Multiplicity: AND + OR + XOR (at most one is 1 per row)
            Multiplicity::Linear(vec![
                stark::lookup::LinearTerm::Column {
                    coefficient: 1,
                    column: cols::AND,
                },
                stark::lookup::LinearTerm::Column {
                    coefficient: 1,
                    column: cols::OR,
                },
                stark::lookup::LinearTerm::Column {
                    coefficient: 1,
                    column: cols::XOR,
                },
            ]),
            vec![
                // Opcode: 0*AND + 1*OR + 2*XOR = OR + 2*XOR
                BusValue::linear(vec![
                    stark::lookup::LinearTerm::Column {
                        coefficient: 1,
                        column: cols::OR,
                    },
                    stark::lookup::LinearTerm::Column {
                        coefficient: 2,
                        column: cols::XOR,
                    },
                ]),
                BusValue::Packed {
                    start_column: cols::ARG1[i],
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::ARG2[i],
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::RES[i],
                    packing: Packing::Direct,
                },
            ],
        ));
    }
```

**Step 3: Compile check**

Run: `cargo check -p prover 2>&1 | head -50`

Expected: Clean compilation.

**Step 4: Commit**

```
git add prover/src/tables/cpu.rs
git commit -m "feat: unify CPU AND/OR/XOR into 8 BitwiseByte sender interactions"
```

---

### Task 6: Update Remaining References

**Files:**
- Modify: `prover/src/tables/trace_builder.rs` (references to `BitwiseOperationType::AndByte`)
- Modify: `prover/src/tables/load.rs` (if any bitwise op type references)
- Any other files referencing the old enum variants

**Step 1: Find all remaining references**

Run: `cargo check -p prover 2>&1 | grep "error"` to find any remaining compile errors from old enum variant names.

Common locations based on grep results:
- `trace_builder.rs:895` — uses `BitwiseOperationType::AndByte` for BRANCH table's AND lookup
- `load.rs` — uses `BitwiseOperationType::Msb8` (unchanged, no action needed)

**Step 2: Fix trace_builder.rs**

The BRANCH table at line 895 uses `BitwiseOperationType::AndByte` for a single-byte AND lookup. Update to use `BitwiseOperation::byte_op(BITWISE_OP_AND, ...)`.

Find the exact call and replace:
```rust
// Old:
BitwiseOperation::byte_op(BitwiseOperationType::AndByte, lo, hi)
// New:
BitwiseOperation::byte_op(bitwise::BITWISE_OP_AND, lo, hi)
```

**Step 3: Compile until clean**

Run: `cargo check -p prover 2>&1 | head -50`

Iterate until no errors remain.

**Step 4: Commit**

```
git add -A
git commit -m "fix: update all remaining references to old AND/OR/XOR enum variants"
```

---

### Task 7: Update Tests

**Files:**
- Modify: `prover/src/tests/bitwise_bus_tests.rs`
- Modify: `prover/src/tests/bitwise_tests.rs`

**Step 1: Update `bitwise_bus_tests.rs`**

The test uses `BusId::AndByte` for both sender and receiver AIRs. Update to `BusId::BitwiseByte` and add the opcode element to both sides.

Sender side — add an OPCODE column (col 4) and include it in the bus value:
```rust
mod sender_cols {
    pub const X: usize = 0;
    pub const Y: usize = 1;
    pub const RESULT: usize = 2;
    pub const FLAG: usize = 3;     // 1 when active
    pub const OPCODE: usize = 4;   // 0=AND, 1=OR, 2=XOR
    pub const NUM_COLUMNS: usize = 5;
}
```

Receiver side — add OPCODE and BITWISE_RESULT columns:
```rust
mod receiver_cols {
    pub const X: usize = 0;
    pub const Y: usize = 1;
    pub const AND: usize = 2;
    pub const OPCODE: usize = 3;         // Z value (0=AND, 1=OR, 2=XOR)
    pub const BITWISE_RESULT: usize = 4; // result selected by opcode
    pub const MU: usize = 5;             // unified multiplicity
    pub const NUM_COLUMNS: usize = 6;
}
```

Update `new_sender_air` and `new_receiver_air` to use `BusId::BitwiseByte` with 4-element signature `[OPCODE, X, Y, RESULT]`.

Update `create_sender_trace` to set opcode=0 (AND) for existing tests.

Update `create_receiver_trace` to include OPCODE=0, BITWISE_RESULT = x & y.

Add new test: `test_completeness_or_byte` with opcode=1, `test_completeness_xor_byte` with opcode=2.

**Step 2: Update `bitwise_tests.rs`**

Update any references to old column constants (`MU_AND`, `MU_OR`, `MU_XOR`).

The test at line 4 imports `cols` — since column names changed, verify the test still references valid column names. Key changes:
- `cols::MU_AND` → `cols::MU_BITWISE_BYTE`
- `cols::MU_OR` removed
- `cols::MU_XOR` removed
- Add tests for `cols::BITWISE_RESULT`

**Step 3: Run tests**

Run: `cargo test -p prover -- bitwise 2>&1 | tail -20`

Expected: All tests pass.

**Step 4: Commit**

```
git add prover/src/tests/
git commit -m "test: update bitwise tests for unified BitwiseByte bus"
```

---

### Task 8: Run Full Test Suite and Fix Regressions

**Files:**
- Any files that need fixing based on test failures

**Step 1: Run the full prover test suite**

Run: `cargo test -p prover 2>&1 | tail -30`

**Step 2: Run the integration tests (end-to-end program proofs)**

Run: `cargo test -p prover -- --ignored 2>&1 | tail -30`

These tests prove and verify actual RISC-V programs. If the BitwiseByte bus is correct, all AND/OR/XOR instructions will go through the unified bus and verification should succeed.

**Step 3: Fix any regressions**

Common issues to watch for:
- The preprocessed commitment will be different (column layout changed). The `OnceLock` caching means it auto-recomputes, but any hardcoded expected commitment values in tests will need updating.
- If tests assert specific column counts, update them (22 → 21 for BITWISE, interaction counts for CPU).

**Step 4: Commit**

```
git add -A
git commit -m "fix: resolve regressions from unified BitwiseByte bus migration"
```

---

### Task 9: Verify Savings

**Step 1: Add or update a test that checks interaction/column counts**

Verify CPU now has 24 interactions (was 40) and BITWISE has 9 (was 11).

```rust
#[test]
fn test_cpu_interaction_count() {
    let interactions = cpu::bus_interactions();
    // 8 BitwiseByte + 2 MSB16 + 1 MSB8 + 1 ZERO + 4 MEMW + ... = 24
    assert_eq!(interactions.len(), 24);
}

#[test]
fn test_bitwise_interaction_count() {
    let interactions = bitwise::bus_interactions();
    // 1 BitwiseByte + 1 MSB8 + 1 MSB16 + 1 ZERO + 1 IsByte + 1 IsHalfword + 1 IsB20 + 1 HWSL + 1 HWSLC = 9
    assert_eq!(interactions.len(), 9);
}
```

**Step 2: Verify the column savings match the design**

```rust
#[test]
fn test_bitwise_column_count() {
    assert_eq!(bitwise::cols::NUM_COLUMNS, 21); // was 22
}
```

**Step 3: Run the verification tests**

Run: `cargo test -p prover -- test_cpu_interaction_count test_bitwise_interaction_count test_bitwise_column_count`

**Step 4: Commit**

```
git add -A
git commit -m "test: verify interaction and column count savings from unified BitwiseByte bus"
```
