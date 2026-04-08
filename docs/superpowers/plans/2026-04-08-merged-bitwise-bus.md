# Merged Bitwise Bus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge AndByte/OrByte/XorByte buses into a single BitwiseByte bus with op_type discriminant, reducing CPU interactions from 24 to 8.

**Architecture:** Replace 3 separate bus IDs with 1. Add a 4th fingerprint field (op_type: 0=AND, 1=OR, 2=XOR). CPU uses `Multiplicity::Sum3(AND, OR, XOR)` and a virtual op_type column. SHIFT and MEMW_A add op_type=0 constant. Bitwise table receivers get op_type constants.

**Tech Stack:** Rust, lambdaworks STARK framework

**Spec:** `docs/superpowers/specs/2026-04-08-merged-bitwise-bus-design.md`

---

### Task 1: Replace BusId variants in types.rs

**Files:**
- Modify: `prover/src/tables/types.rs`

- [ ] **Step 1: Find the BusId enum and replace variants**

Replace `AndByte`, `OrByte`, `XorByte` with `BitwiseByte`. Keep the same
numeric discriminant for `BitwiseByte` (use `AndByte`'s old value = 3).
Renumber any subsequent variants if needed.

```rust
// Before:
// AndByte = 3,
// OrByte = 4,
// XorByte = 5,

// After:
BitwiseByte = 3,  // Merged AND/OR/XOR byte bus (replaces AndByte, OrByte, XorByte)
```

Adjust subsequent discriminant values to fill the gap (or leave gaps — they're
just identifiers).

- [ ] **Step 2: Fix all compilation errors from removed variants**

Run `cargo check -p lambda-vm-prover 2>&1 | head -40` to find all references
to the old variant names. Don't fix the logic yet — just note which files
need changes. They should be: `cpu.rs`, `bitwise.rs`, `shift.rs`, and
possibly `memw_aligned.rs`.

- [ ] **Step 3: Commit**

```bash
git add prover/src/tables/types.rs
git commit -m "refactor: replace AndByte/OrByte/XorByte BusIds with BitwiseByte"
```

(This commit intentionally breaks compilation — subsequent tasks fix it.)

---

### Task 2: Update bitwise table receivers

**Files:**
- Modify: `prover/src/tables/bitwise.rs`

- [ ] **Step 1: Update the 3 receiver interactions in `bus_interactions()`**

Find the 3 `BusInteraction::receiver` calls for `BusId::AndByte`,
`BusId::OrByte`, `BusId::XorByte` (around line 547-605). Replace each with
`BusId::BitwiseByte` and add a 4th bus value for op_type:

```rust
// AND: [X, Y, AND_result, op_type=0]
BusInteraction::receiver(
    BusId::BitwiseByte,
    Multiplicity::Column(cols::MU_AND),
    vec![
        BusValue::Packed { start_column: cols::X, packing: Packing::Direct },
        BusValue::Packed { start_column: cols::Y, packing: Packing::Direct },
        BusValue::Packed { start_column: cols::AND, packing: Packing::Direct },
        BusValue::Linear(vec![LinearTerm::Constant(0)]),
    ],
),
// OR: [X, Y, OR_result, op_type=1]
BusInteraction::receiver(
    BusId::BitwiseByte,
    Multiplicity::Column(cols::MU_OR),
    vec![
        BusValue::Packed { start_column: cols::X, packing: Packing::Direct },
        BusValue::Packed { start_column: cols::Y, packing: Packing::Direct },
        BusValue::Packed { start_column: cols::OR, packing: Packing::Direct },
        BusValue::Linear(vec![LinearTerm::Constant(1)]),
    ],
),
// XOR: [X, Y, XOR_result, op_type=2]
BusInteraction::receiver(
    BusId::BitwiseByte,
    Multiplicity::Column(cols::MU_XOR),
    vec![
        BusValue::Packed { start_column: cols::X, packing: Packing::Direct },
        BusValue::Packed { start_column: cols::Y, packing: Packing::Direct },
        BusValue::Packed { start_column: cols::XOR, packing: Packing::Direct },
        BusValue::Linear(vec![LinearTerm::Constant(2)]),
    ],
),
```

- [ ] **Step 2: Commit**

```bash
git add prover/src/tables/bitwise.rs
git commit -m "feat(bitwise): update receivers to merged BitwiseByte bus with op_type"
```

---

### Task 3: Update CPU table senders (the main optimization)

**Files:**
- Modify: `prover/src/tables/cpu.rs`

- [ ] **Step 1: Replace the three bitwise loops with one merged loop**

Find the three `for i in 0..8` loops for AND_BYTE, OR_BYTE, XOR_BYTE
interactions (around lines 1026-1096). Replace all three with:

```rust
    // -------------------------------------------------------------------------
    // BITWISE_BYTE interactions (×8, merged AND/OR/XOR)
    // -------------------------------------------------------------------------
    // AND, OR, XOR are mutually exclusive per CPU cycle (enforced by DECODE).
    // Merged into a single bus with op_type discriminant:
    //   op_type = 0*AND + 1*OR + 2*XOR
    //   multiplicity = AND + OR + XOR (at most 1)
    for i in 0..8 {
        interactions.push(BusInteraction::sender(
            BusId::BitwiseByte,
            Multiplicity::Sum3(cols::AND, cols::OR, cols::XOR),
            vec![
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
                // op_type: 0=AND, 1=OR, 2=XOR
                BusValue::Linear(vec![
                    LinearTerm::Column { coefficient: 1, column: cols::OR },
                    LinearTerm::Column { coefficient: 2, column: cols::XOR },
                ]),
            ],
        ));
    }
```

- [ ] **Step 2: Update interaction count in any doc comments**

Search for comments mentioning the old count of AND/OR/XOR interactions and
update them.

- [ ] **Step 3: Commit**

```bash
git add prover/src/tables/cpu.rs
git commit -m "feat(cpu): merge AND/OR/XOR into 8 BitwiseByte interactions (was 24)"
```

---

### Task 4: Update SHIFT table senders

**Files:**
- Modify: `prover/src/tables/shift.rs`

- [ ] **Step 1: Find all AndByte references in shift.rs**

There are 3 `BusId::AndByte` sender interactions (around lines 396, 417, 521).
For each, change `BusId::AndByte` → `BusId::BitwiseByte` and add the 4th
bus value:

```rust
BusValue::Linear(vec![LinearTerm::Constant(0)]),  // op_type=0 (AND)
```

- [ ] **Step 2: Update trace builder if needed**

Check `shift.rs` for any `BitwiseOperationType::AndByte` references in trace
generation. These should still work unchanged since the operation type enum
is separate from the BusId.

- [ ] **Step 3: Commit**

```bash
git add prover/src/tables/shift.rs
git commit -m "feat(shift): migrate AndByte senders to BitwiseByte with op_type=0"
```

---

### Task 5: Update MEMW_A table if needed

**Files:**
- Modify: `prover/src/tables/memw_aligned.rs` (if it still uses AndByte)

- [ ] **Step 1: Check if MEMW_A still uses AndByte on this branch**

```bash
grep -n "AndByte" prover/src/tables/memw_aligned.rs
```

If yes: change `BusId::AndByte` → `BusId::BitwiseByte` and add op_type=0.
If no (already migrated to IsHalfword): skip this task.

- [ ] **Step 2: Commit (if changes made)**

```bash
git add prover/src/tables/memw_aligned.rs
git commit -m "feat(memw_a): migrate AndByte sender to BitwiseByte with op_type=0"
```

---

### Task 6: Remove old BusId variants and verify compilation

**Files:**
- Modify: `prover/src/tables/types.rs` (already done in Task 1)

- [ ] **Step 1: Verify no remaining references to old variants**

```bash
cargo check -p lambda-vm-prover 2>&1
```

Expected: compiles with no errors. If there are remaining references,
fix them.

- [ ] **Step 2: Regenerate precomputed bitwise commitment**

The bitwise table's precomputed commitment is a hardcoded hash. Since the
fingerprints changed (4 fields instead of 3), the commitment must be
regenerated. Find where the commitment is hardcoded and run the commitment
generation tool/test.

Search for the hardcoded commitment:
```bash
grep -rn "precomputed_commitment\|PRECOMPUTED_COMMITMENT" prover/src/tables/bitwise.rs
```

Run the commitment generation (likely a test or script that prints the new
hash).

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "fix(bitwise): regenerate precomputed commitment for merged bus fingerprints"
```

---

### Task 7: Verify correctness

**Files:** No changes — verification only.

- [ ] **Step 1: Run stark crate tests**

```bash
cargo test -p stark -- --test-threads=1
```

Expected: all tests pass

- [ ] **Step 2: Run prover tests (if any are working)**

```bash
cargo test -p lambda-vm-prover -- --test-threads=1
```

Note: some tests may have pre-existing failures (UnknownSyscall). Focus on
tests that were passing before this change.

- [ ] **Step 3: Verify CPU aux column count decreased**

Check the CPU's `num_auxiliary_rap_columns()` or count the bus interactions.
With the merged bus, the CPU should have 16 fewer interactions. After batching
(pairs of 2), this saves ~8 aux columns.
