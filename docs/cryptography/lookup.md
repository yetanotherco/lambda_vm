# Lookup Arguments

Lookup arguments are a cryptographic technique that allows a prover to demonstrate that values in one table appear in another table, without revealing the actual values. They are essential for building efficient virtual machines where different components (CPU, memory, ALU) need to verify consistency with each other.

## Why Lookup Arguments?

In a virtual machine proof, different execution tables need to communicate:

- The **CPU table** performs operations and accesses memory
- The **Memory table** stores and retrieves values
- The **ALU table** computes arithmetic operations

Without lookups, verifying that "the CPU read value X from address Y" would require expensive polynomial constraints. Lookup arguments provide an efficient way to prove these cross-table relationships.

## The LogUp Protocol

We use the **LogUp** (Logarithmic Derivative Lookup) protocol, which is based on a key mathematical insight: two multisets are equal if and only if their logarithmic derivatives are equal.

### Fingerprints

Each row in a table is compressed into a single field element called a **fingerprint**:

```
fingerprint = 1 / (z - (v₀ + v₁·α + v₂·α² + ...))
```

Where:
- `z` and `α` are random challenges sampled via Fiat-Shamir
- `v₀, v₁, v₂, ...` are the column values in that row

The linear combination `v₀ + v₁·α + v₂·α² + ...` compresses multiple columns into one value, and `z` shifts it to enable the logarithmic derivative form.

### Running Sum

For each table interaction, we build an auxiliary column that accumulates fingerprints:

```
s[i+1] = s[i] + multiplicity[i] / (z - linear_combination[i])
```

Where `multiplicity` indicates how many times this row participates in the lookup:
- **Positive** for rows being "looked up" (proving side)
- **Negative** for rows doing the "looking" (assuming side)

### Bus Balancing

The key property: if all lookups are valid, the sum of all fingerprints across all tables equals zero. This is because every "send" (negative multiplicity) has a matching "receive" (positive multiplicity).

```
Σ (sends) + Σ (receives) = 0
```

This is verified by checking that the final values of all running-sum columns sum to zero.

## Multi-Table Challenge Sharing

For the bus to balance correctly, **all tables must use the same random challenges** `(z, α)`. This is critical for security and correctness.

### Protocol Flow

1. **Commit all main traces**: Each table commits its main execution trace to the transcript
2. **Sample shared challenges**: After ALL main traces are committed, sample `z` and `α` once
3. **Build auxiliary traces**: Each table builds its running-sum columns using the shared challenges
4. **Commit auxiliary traces**: Each table commits its auxiliary trace
5. **Continue STARK protocol**: Proceed with composition polynomial, FRI, etc.

### Why Share Challenges?

If tables used different challenges:
- Table A computes fingerprints with `(z₁, α₁)`
- Table B computes fingerprints with `(z₂, α₂)`
- The fingerprints don't match even for identical values
- The bus cannot balance, and valid proofs become impossible

By sharing challenges, fingerprints for the same values are identical across tables, enabling the bus to balance.

## Implementation

### Challenge Constants

```rust
// Index of the `z` challenge - evaluation point for fingerprints
pub const LOGUP_CHALLENGE_Z: usize = 0;

// Index of the `α` challenge - base for linear combination
pub const LOGUP_CHALLENGE_ALPHA: usize = 1;

// Total number of LogUp challenges
pub const LOGUP_NUM_CHALLENGES: usize = 2;
```

### Table Interactions

Each AIR defines its lookup interactions via `TableInteraction`:

```rust
pub struct TableInteraction {
    pub flag_columns: Vec<usize>,   // Columns indicating participation (multiplicity)
    pub value_columns: Vec<usize>,  // Columns containing the looked-up values
}
```

### Auxiliary Trace

The auxiliary trace contains one running-sum column per interaction, plus optionally a grand-sum column that aggregates all interactions.

## Security Considerations

1. **Challenge derivation**: Challenges must be sampled via Fiat-Shamir after all main traces are committed to prevent manipulation
2. **Shared challenges**: All tables in a multi-table proof MUST use identical challenges
3. **Field size**: The field must be large enough that random challenges don't accidentally cause fingerprint collisions
