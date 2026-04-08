# Merged Bitwise Bus: AND/OR/XOR Interaction Consolidation

## Problem

The CPU table sends 24 bus interactions for bitwise operations: 8 per byte for
each of AND, OR, XOR on separate buses (AndByte, OrByte, XorByte). Since these
operations are mutually exclusive per CPU cycle (enforced via DECODE), 16 of
these 24 interactions always have zero multiplicity. Each interaction adds a
LogUp term column to the CPU's auxiliary trace, inflating the aux width by ~8
columns (after batching).

## Goals

- Merge 3 bitwise buses (AndByte, OrByte, XorByte) into 1 (BitwiseByte)
- Reduce CPU interactions from 24 to 8 for these operations
- Save ~8 CPU aux columns (from ~42 to ~34)
- Keep CPU main columns unchanged (AND/OR/XOR selectors stay)
- Keep bitwise table row count unchanged (2^20)

## Non-Goals

- Eliminating AND/OR/XOR selector columns from the CPU (used by DECODE)
- Restructuring the bitwise table layout
- Merging other bitwise bus types (IsHalfword, IsByte, MSB8, etc.)

## Design

### Replace Three Buses with One: BitwiseByte

Replace `BusId::AndByte`, `BusId::OrByte`, and `BusId::XorByte` with a single
`BusId::BitwiseByte`. The message adds an `op_type` discriminant:

```
BitwiseByte[X, Y, result, op_type]
```

Where `op_type` = 0 (AND), 1 (OR), 2 (XOR).

**Important:** `AndByte`, `OrByte`, and `XorByte` are used by multiple tables,
not just the CPU:

| Bus | Senders |
|-----|---------|
| AndByte | CPU (×8), SHIFT (×3), MEMW_A (×1) |
| OrByte | CPU (×8) |
| XorByte | CPU (×8) |

All senders must migrate to `BitwiseByte` with the appropriate `op_type`.
OrByte and XorByte are CPU-only, so only the CPU changes for those. AndByte
requires updating SHIFT and MEMW_A as well — they add `op_type=0` to their
existing `[X, Y, result]` messages. This is a straightforward addition of one
constant field per interaction.

### CPU Table Changes

Replace the three `for i in 0..8` loops (24 interactions) with one loop (8):

```rust
for i in 0..8 {
    interactions.push(BusInteraction::sender(
        BusId::BitwiseByte,
        Multiplicity::Sum3(cols::AND, cols::OR, cols::XOR),
        vec![
            BusValue::Packed { start_column: cols::ARG1[i], packing: Packing::Direct },
            BusValue::Packed { start_column: cols::ARG2[i], packing: Packing::Direct },
            BusValue::Packed { start_column: cols::RES[i],  packing: Packing::Direct },
            BusValue::linear(vec![
                LinearTerm::Column { coefficient: 1, column: cols::OR },
                LinearTerm::Column { coefficient: 2, column: cols::XOR },
            ]),
        ],
    ));
}
```

- `Multiplicity::Sum3(AND, OR, XOR)` is 0 or 1 per row (mutual exclusivity)
- `op_type` = `0*AND + 1*OR + 2*XOR` (virtual linear column)

### SHIFT Table Changes

Replace `BusId::AndByte` with `BusId::BitwiseByte` in all 3 SHIFT interactions.
Add `BusValue::Linear(vec![LinearTerm::Constant(0)])` as the 4th field
(op_type=0 for AND). Multiplicity and other fields stay the same.

### MEMW_A Table Changes

Replace `BusId::AndByte` with `BusId::BitwiseByte` in the 1 MEMW_A interaction.
Add `BusValue::Linear(vec![LinearTerm::Constant(0)])` as the 4th field.

Note: on `feat/memw_r`, MEMW_A may have already replaced AndByte with
IsHalfword (per PR #472). If so, this table needs no changes. Verify at
implementation time.

### Bitwise Table Changes

Replace 3 receivers (one per old bus) with 3 receivers on `BitwiseByte`,
distinguished by the constant `op_type` field:

```rust
// AND: [X, Y, AND_result, op_type=0]
BusInteraction::receiver(BusId::BitwiseByte, Multiplicity::Column(cols::MU_AND),
    vec![X, Y, AND, BusValue::Linear(vec![LinearTerm::Constant(0)])]),

// OR: [X, Y, OR_result, op_type=1]
BusInteraction::receiver(BusId::BitwiseByte, Multiplicity::Column(cols::MU_OR),
    vec![X, Y, OR, BusValue::Linear(vec![LinearTerm::Constant(1)])]),

// XOR: [X, Y, XOR_result, op_type=2]
BusInteraction::receiver(BusId::BitwiseByte, Multiplicity::Column(cols::MU_XOR),
    vec![X, Y, XOR, BusValue::Linear(vec![LinearTerm::Constant(2)])]),
```

Each receiver keeps its own multiplicity column (MU_AND, MU_OR, MU_XOR).
The `op_type` constant in the fingerprint ensures AND lookups only match
AND receivers.

### Trace Builder Changes

The trace builder emits `BitwiseOperation` entries tagged with
`BitwiseOperationType::{AndByte, OrByte, XorByte}`. These types still
map to the same multiplicity columns (MU_AND, MU_OR, MU_XOR) in the bitwise
table. No changes needed to the operation types or multiplicity routing —
only the `BusId` on the constraint side changes.

### Remove Old BusIds

After migrating all senders and receivers, remove `AndByte`, `OrByte`, and
`XorByte` from the `BusId` enum.

## Files Changed

| File | Change |
|------|--------|
| `prover/src/tables/types.rs` | Add `BitwiseByte`, remove `AndByte`/`OrByte`/`XorByte` |
| `prover/src/tables/cpu.rs` | Replace 24 interactions with 8 |
| `prover/src/tables/bitwise.rs` | Replace 3 receivers with 3 on merged bus + op_type |
| `prover/src/tables/shift.rs` | Update 3 AndByte senders → BitwiseByte + op_type=0 |
| `prover/src/tables/memw_aligned.rs` | Update 1 AndByte sender if still present |

## Soundness Argument

1. The `op_type` discriminant (0/1/2) in the fingerprint makes AND/OR/XOR
   lookups cryptographically distinct. A send with op_type=0 cannot match a
   receiver with op_type=1.

2. `Multiplicity::Sum3(AND, OR, XOR)` produces 0 or 1 because the flags are
   mutually exclusive (enforced by DECODE).

3. SHIFT and MEMW_A always use op_type=0 (AND) with their existing multiplicity
   columns — no soundness change for those tables.

4. The bitwise table receivers still validate `result = X op Y` for each
   op_type. The fingerprint binding ensures no cross-operation forgery.

## Expected Impact

- CPU bus interactions: 24 → 8 (saves 16)
- CPU aux columns: ~42 → ~34 (saves ~8 after LogUp batching)
- CPU effective width: ~114 → ~106
- SHIFT: +1 field per interaction (3 interactions, minor fingerprint cost)
- Bitwise table: same columns, same row count, same 3 receiver count
- Precomputed bitwise commitment: must be regenerated (fingerprints change)

## Testing

- All existing prove/verify tests cover correctness
- Verify bus balance with merged interactions
- Verify precomputed bitwise commitment is updated
