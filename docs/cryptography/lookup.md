# Lookup arguments

Lambda VM uses **LogUp** lookup arguments to connect its trace tables. Each table generates and consumes "tokens" on one or more **named buses**; the system is sound when, across every bus, the total sender contribution equals the total receiver contribution.

Lookups are how the prover proves cross-table relations without duplicating constraints. For example, the CPU table proves it dispatched a bitwise AND instruction by sending a token `(AndByte, x, y, x & y)` on the `AndByte` bus; the BITWISE table proves it has a row matching that token by sending a receiver token on the same bus. If both sides match, the bus balances. If a sender has no matching receiver (or vice versa), the bus does not balance and verification fails.

The implementation lives in [`crypto/stark/src/lookup.rs`](../../crypto/stark/src/lookup.rs).

## The `BusInteraction` struct

A single lookup contribution is a `BusInteraction`:

```rust
pub struct BusInteraction {
    pub bus_id: u64,
    pub multiplicity: Multiplicity,
    pub values: Vec<BusValue>,
    pub is_sender: bool,
}
```

| Field | Role |
|---|---|
| `bus_id` | Names the bus. Senders and receivers must use the same `bus_id` for their tokens to match. Different buses use different IDs so that, e.g., an `And` token doesn't accidentally cancel an `Xor` token. |
| `multiplicity` | How many times this row contributes (see below). |
| `values` | The token payload — the data being looked up. |
| `is_sender` | Carries the sign. Senders add to the bus sum, receivers subtract. The balance check is `Σ sender − Σ receiver = 0`. |

Build them with `BusInteraction::sender(bus_id, mul, values)` or `::receiver(bus_id, mul, values)`.

## Named buses (`BusId`)

Bus IDs are declared in [`prover/src/tables/types.rs`](../../prover/src/tables/types.rs) as a `#[repr(u64)]` enum:

```rust
#[repr(u64)]
pub enum BusId {
    IsByte = 0,
    IsHalfword,
    IsB20,
    AndByte,
    OrByte,
    XorByte,
    Msb8,
    Msb16,
    Zero,
    Hwsl,
    Lt,
    Mul,
    Dvrm,
    Shift,
    Memw,
    Load,
    Memory,
    // ...
}
```

Each value is a `u64` discriminant, auto-incremented from 0. `BusInteraction::new` takes `impl Into<u64>` so you pass `BusId::AndByte` directly.

## Multiplicity

How many copies of the token a row contributes. Most rows are `One`, but tables that deduplicate or have flag-gated participation use richer forms:

```rust
pub enum Multiplicity {
    One,                              // 1
    Column(usize),                    // col[i]
    Sum(usize, usize),                // col[a] + col[b]
    Negated(usize),                   // 1 - col[i]    (col must be a bit)
    Diff(usize, usize),               // col[a] - col[b]
    Sum3(usize, usize, usize),        // col[a] + col[b] + col[c]
    Linear(Vec<LinearTerm>),          // arbitrary signed combination
}
```

`Linear` is the escape hatch — it supports signed coefficients and large unsigned coefficients (e.g. `2^{-32} mod p`), and is how interactions like `μ − read2 − read4 − read8` are expressed.

## Bus values (the token payload)

Each entry in `BusInteraction.values` is a `BusValue`:

```rust
pub enum BusValue {
    Packed { start_column: usize, packing: Packing },
    Linear(Vec<LinearTerm>),
}
```

- `Packed` reads consecutive trace columns and combines them via a `Packing` formula (powers of 2). For example, `Packing::Word2L` at `start_column = 4` reads columns 4 and 5 and computes `c₄ + 2¹⁶·c₅`, producing one bus element representing a 32-bit word.
- `Linear` is an arbitrary signed linear combination over columns and constants — used when the value is a flag, a constant tag, or a derived expression that doesn't fit a `Packing`.

The `Packing` enum supports primitive shapes (`Direct`, `Word2L`, `Word4L`) and compound shapes (`DWordHL`, `DWordBL`, `QuadHL`, …) that produce multiple bus elements. A 64-bit double-word stored as 4 half-words is one `BusValue::Packed { packing: DWordHL, .. }` that yields two bus elements.

## Two-stage value combination

A token's contribution to the bus is computed in **two stages**:

1. **Limb packing.** Within each `BusValue`, columns are combined using powers of 2 according to the chosen `Packing`. This is how multi-limb values are formed from their column-level representation (e.g. assembling a 32-bit word from four 8-bit byte columns).

2. **Bus fingerprint.** All bus elements from the interaction — starting with the `bus_id`, then the elements produced by each `BusValue` — are folded together using powers of a single challenge α:

   ```
   fingerprint = z − (bus_id + α·v₁ + α²·v₂ + … + α^(k−1)·v_{k−1})
   ```

   The interaction's contribution at this row is `± multiplicity / fingerprint`, with the sign coming from `is_sender`.

The `bus_id` is the first bus element. This is what makes tokens on different buses non-interfering: two interactions on `BusId::Mul` and `BusId::Lt` have different fingerprints even when all the data values match.

## Challenges

LogUp uses two challenges sampled from the transcript after the main trace is committed:

- `z` — read as `challenges[0]`; the subtractor in the fingerprint denominator.
- `α` — read as `challenges[LOGUP_CHALLENGE_ALPHA]` where `LOGUP_CHALLENGE_ALPHA = 1`; the base for the powers-of-α combination of bus elements.

The total challenge count is `LOGUP_NUM_CHALLENGES = 2`. There is no separate `LOGUP_CHALLENGE_Z` constant — `z` is just the first challenge by convention.

## Bus balance

For every bus to be sound, across all tables and all rows:

```
Σ (over senders)    multiplicity / fingerprint
−
Σ (over receivers)  multiplicity / fingerprint
= 0
```

In code this becomes: every table contributes a per-bus running sum (the "table contribution") to its auxiliary trace; the verifier checks that the sum of table contributions equals the expected bus balance. For most buses the expected balance is zero. The `Commit` bus is an exception: its expected balance is recomputed by the verifier from the public output bytes (see `compute_commit_bus_offset` in [`prover/src/lib.rs`](../../prover/src/lib.rs)) so that tampering with the proof's public output is caught.

When `--features debug-checks` is on, [`crypto/stark/src/bus_debug.rs`](../../crypto/stark/src/bus_debug.rs) prints per-bus sender vs. receiver sums to help diagnose imbalances during development.

## Implementation pointers

- Interaction shape, packing, balance: [`crypto/stark/src/lookup.rs`](../../crypto/stark/src/lookup.rs)
- Named bus IDs used by the VM: [`prover/src/tables/types.rs`](../../prover/src/tables/types.rs)
- Per-table interactions: [`prover/src/tables/`](../../prover/src/tables/) (one file per table)
- Verifier-side bus offset for the COMMIT bus: [`prover/src/lib.rs`](../../prover/src/lib.rs) (`compute_commit_bus_offset`)
- Debug-checks bus diagnostics: [`crypto/stark/src/bus_debug.rs`](../../crypto/stark/src/bus_debug.rs)
