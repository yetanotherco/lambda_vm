<!-- Provenance: mapped 2026-07-31 by a read-only scout agent commissioned by the
reg-tree worker for the FRI folding leg; every section is marked with the scout's
own verification status and file:line citations against feat/lfm @ edfc5a81.
The sim/24 cycle figure in §5 is campaign memory (guest-side RV32), not re-measured
here. Committed by team-lead so the spec survives agent context loss. -->

# Production FRI VERIFY path — implementation spec for LFM emission

Worktree: `/private/tmp/claude-501/-Users-maurofab-workspace-lambda-vm-3/0cdd934d-c82f-4724-bc05-01b1924f85f0/scratchpad/wt-reg-tree`

Note: this worktree contains **only the unbatched FRI**. `grep -rl batched crypto/stark/src/` returns no FRI-verify file — the batched-FRI verifier (#768) is not on this branch. Everything below is the single production verify path.

---

## 1. The verify-side query loop ✓ VERIFIED

**Entry:** `step_3_verify_fri`, `crypto/stark/src/verifier.rs:387-483`.
**Per-query core:** `verify_query_and_sym_openings`, `crypto/stark/src/verifier.rs:660-748`.

Driver (`verifier.rs:469-482`) — one call per query index, no cross-query state:

```rust
(0..challenges.iotas.len())
    .zip(evaluation_point_inverse)
    .all(|(i, eval)| {
        Self::verify_query_and_sym_openings(
            proof, &challenges.zetas, challenges.iotas[i], proof.query(i),
            eval, &deep_poly_evaluations[i], &deep_poly_evaluations_sym[i],
            &terminal_codeword,
        )
    })
```

**Exact sequence for ONE query** (`iota`):

1. Take `p₀(υ)`, `p₀(−υ)` from the DEEP reconstruction (these are *not* Merkle-checked here; step 4 authenticates the underlying trace/composition leaves).
2. **Fold 0** (unauthenticated, no layer): `v ← (p₀+p₀ˢ) + υ⁻¹·ζ₀·(p₀−p₀ˢ)`. `index ← iota`.
3. **For i = 0 .. num_committed−1**: authenticate the leaf `{v, evaluation_sym[i]}` against `fri_layers_merkle_roots[i]` at Merkle position `index>>1`; then fold `v ← (v+sym) + υ^(−2^(i+1))·ζ_{i+1}·(v−sym)`; then `index >>= 1`.
4. **Terminal**: `terminal_codeword[index] == v`.

So per query: `num_committed` Merkle authentications, `num_committed + 1` folds, 1 array lookup + equality. Note the **asymmetry**: folds = layers + 1, because the first fold consumes DEEP values rather than a committed layer.

Verbatim core (`verifier.rs:692-747`):

```rust
        let evaluation_point_vec: Vec<FieldElement<Field>> =
            core::iter::successors(Some(evaluation_point_inv.square()), |evaluation_point| {
                Some(evaluation_point.square())
            })
            .take(fri_layers_merkle_roots.len())
            .collect();

        // Reconstruct p₁(𝜐²)
        let mut v =
            (p0_eval + p0_eval_sym) + evaluation_point_inv * &zetas[0] * (p0_eval - p0_eval_sym);
        let mut index = iota;

        let openings_ok = fri_layers_merkle_roots
            .iter()
            .zip(fri_decommitment.layers_evaluations_sym())
            .zip(evaluation_point_vec)
            .enumerate()
            .fold(
                true,
                |result, (i, ((merkle_root, evaluation_sym), evaluation_point_inv))| {
                    let openings_ok = Self::verify_fri_layer_openings(
                        merkle_root,
                        fri_decommitment.layer_auth_path(i),
                        &v,
                        evaluation_sym,
                        index,
                    );

                    // Update `v` with next value pᵢ₊₁(𝜐^(2ⁱ⁺¹)).
                    v = (&v + evaluation_sym)
                        + evaluation_point_inv * &zetas[i + 1] * (&v - evaluation_sym);

                    index >>= 1;

                    result & openings_ok
                },
            );

        let terminal_ok = terminal_codeword.get(index).is_some_and(|t| &v == t);
        openings_ok & terminal_ok
```

**Degenerate branch you must emit** (`verifier.rs:683-690`): when `zetas.is_empty()` (`total_folds == 0`, clamp case) the terminal codeword *is* p₀, and the check is `terminal[2·iota] == p₀ ∧ terminal[2·iota+1] == p₀ˢ`. Not reachable under production presets (see §7) but present.

**Structural pre-checks that must precede the loop** (`verifier.rs:426-448`) — all three are soundness-critical and reject rather than panic:
- `fri_layers_merkle_roots().len() == num_committed`
- `fri_final_poly_coeffs().len() == 1 << effective_k`
- every query's `layers_auth_paths_len() == num_committed` **and** `layers_evaluations_sym().len() == num_committed`. The comment at 434-441 is explicit: these vecs are *not* Fiat-Shamir-bound, so this length check is the only thing pinning them.

---

## 2. Layer commitments and the stop condition ✓ VERIFIED

Single source of truth: `FriFoldLayout::new`, `crypto/stark/src/fri/terminal.rs:45-54`:

```rust
    pub(crate) fn new(lde_log: u32, blowup_log: u32, k: u32) -> Self {
        let terminal_log = (blowup_log + k).min(lde_log);
        let total_folds = lde_log - terminal_log;
        Self {
            total_folds,
            num_committed: total_folds.saturating_sub(1) as usize,
            terminal_len: 1usize << terminal_log,
            effective_k: terminal_log - blowup_log,
        }
    }
```

Verifier binding (`verifier.rs:375-382`): `k = air.options().fri_final_poly_log_degree`, `blowup_log = (lde_length/trace_length).trailing_zeros()`, `lde_log = lde_length.trailing_zeros()`.

With `n = log₂(lde_length)`, `b = log₂(blowup)`, `k = 7`:

| quantity | value |
|---|---|
| `terminal_log` | `min(b+k, n)` |
| `total_folds` | `n − b − k` |
| `num_committed` (= Merkle roots = auth paths per query) | `n − b − k − 1` |
| `terminal_len` | `2^(b+k)` |
| `effective_k` | `k` (unclamped) |
| `zetas.len()` | `num_committed + 1` |

**Yes, there is an "early stop at k=7", and it is universal.** `DEFAULT_FRI_FINAL_POLY_LOG_DEGREE: u8 = 7` (`crypto/stark/src/proof/options.rs:93`) is written into every constructor: `default_test_options` (:72), `GoldilocksCubicProofOptions::with_params` (:132), and `MIN_PROOF_OPTIONS` (`prover/src/recursion.rs:44`). Folding stops at codeword length `2^(b+7)` and the prover ships `2^7 = 128` coefficients instead of folding to a constant.

`.min(lde_log)` is the tiny-trace clamp: only when `n ≤ b+7`, i.e. trace_bits ≤ 7. Then `effective_k = n − b < k`, `total_folds = 0`, no zetas, no layers.

Prover/verifier symmetry: `commit_phase_from_evaluations` (`crypto/stark/src/fri/mod.rs:76-118`) runs `num_committed` commit iterations then **one extra unconditional final fold** if `total_folds > 0` — that final fold is never Merkle-committed. The verifier mirrors this in the transcript replay (`verifier.rs:1463-1483`): one zeta per root, then `if total_folds > 0 { zetas.push(sample) }`.

---

## 3. The fold ✓ VERIFIED

`crypto/stark/src/fri/fri_functions.rs:8-59`, in full:

```rust
/// Evaluation-form FRI fold: given evaluations in bit-reversed order where
/// consecutive pairs (2j, 2j+1) are conjugates (p(x_j), p(-x_j)), compute
/// the folded evaluations: (lo + hi) + inv_twiddle[j] * zeta * (lo - hi)
/// = 2 * (p_even(x_j²) + zeta * p_odd(x_j²))
pub(crate) fn fold_evaluations_in_place<F: IsSubFieldOf<E>, E: IsField>(
    evals: &mut Vec<FieldElement<E>>,
    zeta: &FieldElement<E>,
    inv_twiddles: &[FieldElement<F>],
) {
    let half = evals.len() / 2;
    for j in 0..half {
        let lo = &evals[2 * j];
        let hi = &evals[2 * j + 1];
        let sum = lo + hi;
        let diff = lo - hi;
        evals[j] = &sum + &(&inv_twiddles[j] * &(zeta * &diff));
    }
    evals.truncate(half);
}

pub(crate) fn compute_coset_twiddles_inv<F: IsFFTField>(
    coset_offset: &FieldElement<F>,
    domain_size: usize,
) -> Vec<FieldElement<F>> {
    let half = domain_size / 2;
    let order = domain_size.trailing_zeros() as u64;
    let mut points = get_powers_of_primitive_root_coset(order, half, coset_offset).unwrap();
    in_place_bit_reverse_permute(&mut points);
    FieldElement::inplace_batch_inverse(&mut points).unwrap();
    points
}

pub(crate) fn update_twiddles_in_place<F: IsField>(twiddles: &mut Vec<FieldElement<F>>) {
    let new_len = twiddles.len() / 2;
    for j in 0..new_len {
        twiddles[j] = twiddles[2 * j].square();
    }
    twiddles.truncate(new_len);
}
```

**Formula: `f(j) = (lo + hi) + x⁻¹·ζ·(lo − hi)`.**

- **UNNORMALIZED.** No division by 2. The result is `2·(p_even(x²) + ζ·p_odd(x²))` — the factor 2^i accumulates across layers and is absorbed identically on both sides. Do not "fix" this; the terminal comparison is against the prover's own accumulated scaling.
- The point enters as its **inverse**, multiplied into the odd part. Association in the verifier is `(x⁻¹ · ζ) · diff` (`verifier.rs:701, 729`) — a base×ext mul followed by an ext×ext mul. Match this exactly if you care about bit-exactness of intermediate representations; the field result is associative but your chip decomposition may not be.
- Verifier form is `v ← (v+sym) + x⁻¹·ζ·(v−sym)` — same shape, with `v`/`sym` in place of `lo`/`hi`.

### ⚠ The parity/sign compensation — critical, and non-obvious ✓ VERIFIED

The prover's `inv_twiddles[j]` is the inverse of the point at the **even** slot `2j`. The verifier's `evaluation_point_vec[i] = υ^(−2^(i+1))` is the inverse of the point at the **query's own** position `iota>>(i+1)`, which is the odd slot whenever the relevant index bit is 1.

I traced this. With `x_j = offset·ω_N^{br_m(j)}` (`m = log₂N − 1`) and `br_m(2j) = br_{m−1}(j)`, the two differ by exactly `(−1)^{bit}`. But when the query sits in the odd slot, `(v, sym) = (hi, lo)`, so `(v − sym) = −(lo − hi)`. The two sign flips cancel:

```
(hi + lo) + (−x⁻¹)·ζ·(hi − lo)  =  (lo + hi) + x⁻¹·ζ·(lo − hi)
```

**Consequence for LFM: the fold arithmetic requires NO parity branch.** You derive `υ⁻¹` once and square repeatedly. Parity is consulted *only* for leaf ordering in the Merkle check (§4).

---

## 4. Per-layer Merkle authentication ✓ VERIFIED

`verify_fri_layer_openings`, `crypto/stark/src/verifier.rs:626-649`:

```rust
        let evaluations = if iota % 2 == 1 {
            vec![evaluation_sym.clone(), evaluation.clone()]
        } else {
            vec![evaluation.clone(), evaluation_sym.clone()]
        };

        verify_merkle_path::<BatchedMerkleTreeBackend<FieldExtension>>(
            auth_path_sym,
            merkle_root,
            iota >> 1,
            &evaluations,
        )
```

- **Leaf** = the conjugate pair `{p_i(υ^(2^i)), p_i(−υ^(2^i))}` ordered so the **even codeword slot comes first**. `iota` here is the running `index`, not the original query challenge.
- **Byte layout** = 48 bytes: two `Degree3GoldilocksExtensionField` elements, each 24 bytes = three Goldilocks limbs in **component order 0,1,2**, each 8 bytes **big-endian** from `canonical_u64()`. Cited: `crypto/math/src/field/extensions_goldilocks.rs:497-503` (`write_bytes_be`), `:567-571` (`stream_bytes` → same 24 bytes), `crypto/math/src/field/goldilocks.rs:493-495`. One keccak-256 absorb of 48 bytes = **1 permutation** (rate 136).
- **Index** = `index >> 1`; **evolution** = `index >>= 1` after each layer (`verifier.rs:735`), starting at `index = iota`.
- **Tree** = one independent tree per layer, `2^(n−i−2)` leaves, root at `fri_layers_merkle_roots[i]`.

### Which backend — the answer is "both, and they are byte-identical"

This is the sharp edge you flagged, and the two sides genuinely use **different types**:

| side | type | citation |
|---|---|---|
| prover commit | `FriLayerMerkleTree = MerkleTree<PairKeccak256Backend<E>>` | `crypto/stark/src/config.rs:23-24`, used at `crypto/stark/src/fri/mod.rs:100` |
| verifier | `BatchedMerkleTreeBackend<FieldExtension>` = `BatchKeccak256Backend` = `FieldElementVectorBackend` | `crypto/stark/src/config.rs:19-20`, used at `verifier.rs:643` |

They agree because both leaf hashes stream the same bytes into one fresh keccak:

- `FieldElementPairBackend::hash_data` (`crypto/crypto/src/merkle_tree/backends/field_element_vector.rs:122-127`) streams `input[0]` then `input[1]`.
- `FieldElementVectorBackend::hash_data` (`:193-198`) delegates to `hash_data_from_slices(input, &[])` (`:173-179`), which streams every element of `a` then `b`.
- `hash_new_parent` is literally the same function in both (`:129-131` and `:200-202` both call `hash_new_parent_bytes`).

Both are `FieldElement*Backend<F, PlatformKeccak256, 32>` (`crypto/crypto/src/merkle_tree/backends/types.rs:12,15`).

**It is NOT the trace's `commit_bit_reversed` + `ROWS_PER_LEAF=2` scheme.** The distinction is real and you must emit them differently:

- Trace/composition leaves (`crypto/stark/src/commitment.rs:81-91`) apply `reverse_index(rows_per_leaf*leaf_idx + k, num_rows)` **inside** the leaf builder, and concatenate **column-by-column across all columns** for two rows. Leaf size = `2 · num_cols · byte_len`.
- FRI layer leaves (`crypto/stark/src/fri/mod.rs:96-99`) take `evals.chunks_exact(2)` of an **already bit-reversed single codeword** — no permutation applied at commit time, exactly one column. Leaf size = 48 bytes, always.

Path verification fold is shared (`crypto/crypto/src/merkle_tree/proof.rs:31-51`): `index % 2 == 0 ? H(acc‖sib) : H(sib‖acc)`, `index >>= 1`, compare to root. Path length = `log₂(num_leaves)` — no length field, no domain separation, no leaf-index in the hash.

---

## 5. The terminal polynomial ✓ VERIFIED

`crypto/stark/src/fri/terminal.rs` has both directions. The **verify** side is `terminal_codeword_from_coeffs` (`:125-156`), called once per proof at `verifier.rs:450-456`:

```rust
        let terminal_offset = domain.coset_offset.pow(1u64 << layout.total_folds);
        let terminal_codeword =
            crate::fri::terminal::terminal_codeword_from_coeffs::<Field, FieldExtension>(
                proof.fri_final_poly_coeffs(),
                &terminal_offset,
                layout.terminal_len,
            );
```

and its body (`terminal.rs:134-155`):

```rust
    assert!(
        !coeffs.is_empty()
            && coeffs.len().is_power_of_two()
            && codeword_len.is_power_of_two()
            && coeffs.len() <= codeword_len
            && codeword_len.is_multiple_of(coeffs.len()),
        ...
    );

    let poly = Polynomial::new(coeffs);
    let blowup = codeword_len / coeffs.len();

    // Step 1: coset FFT to get natural-order evaluations.
    let mut natural =
        Polynomial::evaluate_offset_fft::<F>(&poly, blowup, Some(coeffs.len()), terminal_offset)
            .expect("terminal coset size must be a power of two within the field's two-adicity");

    // Step 2: convert natural order to bit-reversed (FRI) order.
    in_place_bit_reverse_permute(&mut natural);
    natural
```

**It is an FFT, not a per-point polynomial evaluation and not a coefficient comparison.** The final check is `terminal_codeword.get(index).is_some_and(|t| &v == t)` (`verifier.rs:746`) — a single array lookup and extension-field equality, done once per query against a codeword materialized once per proof.

Cost breakdown: `evaluate_offset_fft` = `poly.scale(offset)` then `evaluate_fft` (`crypto/math/src/polynomial.rs:325-326`), i.e. `2^k` ext×base scalings plus a `terminal_len`-point extension-field FFT, plus a `terminal_len` bit-reverse permute. `terminal_offset` is a base-field `pow` with exponent `2^total_folds` ≈ `total_folds` squarings (square-and-multiply, `crypto/math/src/field/traits.rs:122-142`).

⚠ Design note for LFM: this is the point where my earlier sim/24 measurement applies — replacing this FFT with per-point Horner **regressed +20M cycles**. Emit the FFT.

The assert at `:134` is unreachable in the verifier flow because `verifier.rs:431` length-checks `coeffs` first — but if your emitter reorders those, you convert a rejection into a panic.

---

## 6. The evaluation point per layer ✓ VERIFIED

`query_challenge_to_evaluation_point`, `verifier.rs:489-496`:

```rust
        let raw = iota * 2 + if sym { 1 } else { 0 };
        domain.lde_coset_element(reverse_index(raw, domain.lde_length as u64))
```

with `lde_coset_element(i) = coset_offset · lde_primitive_root^i` (`crypto/stark/src/domain.rs:116-118`) and `reverse_index(i, size) = i.reverse_bits() >> (usize::BITS − size.trailing_zeros())` (`crypto/math/src/fft/bit_reversing.rs:15-21`).

**Yes — this is exactly the `υ = offset · g^{br(2·iota)}` convention documented in `prover/src/lfm/sub_proof.rs:47-52`.** Identical function, identical bit-reversal width (`lde_length`). Your `pow_bits` construction is faithful to production.

The symmetric point: `br(2·iota+1) = br(2·iota) + L/2` and `g^{L/2} = −1`, so `−υ`. Confirmed by `sym: bool` selecting `raw = 2·iota+1`, and used with `sym=true` on the DEEP side only.

**How it changes across layers — this is the part that saves you work.** The verifier never re-derives a point. It computes `υ⁻¹` once (batch-inverted across all queries, `verifier.rs:459-467`) and then produces the whole chain by repeated squaring (`verifier.rs:692-697`):

```
evaluation_point_vec[i] = υ^(−2^(i+1)),  i = 0..num_committed−1
```

So layer `i`'s point is `υ^(2^(i+1))` — **no bit-reversal, no domain lookup, no coset offset, past the first point**. One base-field squaring per layer. Combined with §3's sign result, the entire per-layer point derivation is: one squaring, and nothing else.

The base-field batch inverse (`verifier.rs:459-467`) is over all `Q` queries at once and **fails closed**: `if inplace_batch_inverse(...).is_err() { return false }` — a zero evaluation point (malformed index) rejects rather than panics.

---

## 7. Degenerate parameters — what is actually constant ✓ VERIFIED

`ProofOptions` fields consumed by the FRI verify path (`crypto/stark/src/proof/options.rs:52-61`):

| parameter | production values | constant? |
|---|---|---|
| `fri_final_poly_log_degree` (k) | **7** — always | ✅ **CONSTANT across every config in the repo** |
| `coset_offset` | **3** — always | ✅ **CONSTANT** |
| `blowup_factor` | 2, 4, 8 (min uses 2) | ❌ varies (3 values) |
| `fri_number_of_queries` | 219 / 110 / 73 (min: 1) | ❌ varies, but **fully determined by blowup** |
| `grinding_factor` | 20 (min: 1) | ✅ effectively constant at 20 in all secure presets |
| layer count `num_committed` | `n − b − 8` | ❌ varies with trace size |

Sources: `MIN_PROOF_OPTIONS` (`prover/src/recursion.rs:39-45`), `Preset` (`prover/src/recursion.rs:53-86`), `GoldilocksCubicProofOptions::with_params` (`crypto/stark/src/proof/options.rs:106-134`), `DEFAULT_FRI_FINAL_POLY_LOG_DEGREE` (`:93`), `DEFAULT_GRINDING = 20` (`:96`).

Query counts are computed, not stored — I recomputed the JBR formula (`:121-125`) and it reproduces the doc comments exactly: blowup 2 → 219, blowup 4 → 110, blowup 8 → 73.

Derived per-preset FRI shape:

| preset | b | terminal_log | terminal_len | coeffs | queries | num_committed |
|---|---|---|---|---|---|---|
| Min | 1 | 8 | 256 | 128 | 1 | trace_bits − 8 |
| Blowup2 | 1 | 8 | 256 | 128 | 219 | trace_bits − 8 |
| Blowup4 | 2 | 9 | 512 | 128 | 110 | trace_bits − 8 |
| Blowup8 | 3 | 10 | 1024 | 128 | 73 | trace_bits − 8 |

Note the invariant: **`num_committed = trace_bits − 8` for every preset**, since `n = trace_bits + b` cancels `b`.

### ⚠ What a differential over real proofs CANNOT distinguish

This is the answer you actually need. Because `k = 7` and `coset_offset = 3` are **hardcoded constants with no production variation**, a differential test over real proofs is blind to:

1. **Any k-dependent logic.** An implementation that hardcodes `terminal_log = b + 7`, hardcodes 128 coefficients, or hardcodes `terminal_len ∈ {256,512,1024}` is indistinguishable from one that reads `k` from the AIR. Only `crypto/stark/src/tests/small_trace_tests.rs:177` (k=0), `:215` (k=63), `:720` (k=6) exercise other values.
2. **The clamp path** (`.min(lde_log)`, `terminal.rs:46`). Requires trace_bits ≤ 7. Never reached in production.
3. **The `zetas.is_empty()` no-fold branch** (`verifier.rs:683-690`). Same condition. Dead in production.
4. **`effective_k ≠ k`.** Only occurs under the clamp. Production always has `effective_k == 7`, so an implementation that conflates the two passes everything.
5. **Any `coset_offset ≠ 3` handling**, including the `evaluate_offset_fft` offset path and `terminal_offset = 3^(2^total_folds)`.
6. **Grinding-factor variation.** Only 20 and 1 appear.

Recommendation: build the differential over **synthetic proofs at k ∈ {0, 6, 7, 63} and trace_bits ≤ 7**, using the fixtures already in `small_trace_tests.rs`, or accept that those branches are unexercised and pin them with structural assertions instead.

---

## 8. Counts for sizing — DERIVED ✓ VERIFIED

Let `n = log₂(lde_length)`, `b = log₂(blowup)`, `k = 7`, `Q` = query count, `C = num_committed = n − b − k − 1`.

### Merkle path steps

Layer `i` codeword length = `2^(n−i−1)`; leaves = `2^(n−i−2)`. Path length = `log₂(leaves)`:

$$\text{pathlen}(i) = n - i - 2$$

Derived from `build_merkle_path` (`crypto/crypto/src/merkle_tree/merkle.rs:271-288`) walking `pos → parent_index(pos)` until `ROOT`, over a tree with `2·leaves − 1` nodes (`:199-200`); the leaf count is already a power of two so `complete_until_power_of_two` (`:194`) is a no-op.

**Per query, total path steps:**

$$\sum_{i=0}^{C-1}(n-i-2) \;=\; C(n-2) \;-\; \frac{C(C-1)}{2}$$

Last layer's path length is `n − C − 1 = b + k` — consistent with its `2^(b+k)` leaves. ✓

**Keccak permutations per query** = `C` leaf hashes (48 B → 1 perm each) + path parents (64 B → 1 perm each):

$$\text{perms/query} \;=\; C \;+\; C(n-2) - \tfrac{C(C-1)}{2}$$

Worked example, trace_bits = 20, Blowup2 (`n=21, C=12`): path steps = `12·19 − 66 = 162`; perms/query = `174`; × 219 queries = **38,106 keccak permutations** for the FRI leg alone.

### Field operations

**Per query, per fold** (`verifier.rs:701` and `:729-730`, identical shape):
- 2 ext additions, 1 ext subtraction
- 1 base×ext multiplication (`x⁻¹ · ζ`)
- 1 ext×ext multiplication

There are `C + 1` folds. Total per query:

$$(C+1)\times(2\ \text{ext-add} + 1\ \text{ext-sub} + 1\ \text{base}\!\times\!\text{ext} + 1\ \text{ext}\!\times\!\text{ext})$$

**Per query, point chain** (`verifier.rs:692-697`): `C` base-field squarings.

**Per query, initial point** (`verifier.rs:462, 495`): one `reverse_index` (bit ops), one base-field `pow` with an `n`-bit exponent ≈ `n` squarings + ≤ `n` muls (square-and-multiply, `crypto/math/src/field/traits.rs:134-142`), one base mul by `coset_offset`.

**Amortized per query** (`verifier.rs:465`, `crypto/math/src/field/element.rs:90-108`): batch inverse over `Q` base elements = `3(Q−1)` muls + 1 inversion → ~3 base muls/query.

**Once per proof:** `terminal_offset` pow ≈ `total_folds` base squarings; `Polynomial::new` + `scale` = `2^k` ext×base muls + `2^k` base muls for the geometric offset powers; one `terminal_len`-point extension FFT ≈ `(terminal_len/2)·log₂(terminal_len)` butterflies; one `terminal_len` bit-reverse permute. For Blowup2: 1024 butterflies over Ext3. For Blowup4: 2304.

**Terminal check per query:** 1 bounds-checked index + 1 ext equality (3 base comparisons).

---

## Emission checklist (things that will silently break bit-exactness)

1. Fold is **unnormalized** — no `/2`. §3.
2. Mul association is `(x⁻¹ · ζ) · diff`, base×ext then ext×ext. §3.
3. **No parity branch in the fold**; parity branch **only** in leaf ordering. §3, §4.
4. Leaf = 48 bytes, Ext3 components 0,1,2, each 8 B big-endian, even codeword slot first. §4.
5. FRI leaves are pairs of an **already-bit-reversed** codeword — do not re-apply `reverse_index` the way the trace commitment does. §4.
6. Terminal is an **FFT**, not Horner. §5.
7. `index` starts at `iota` (not `2·iota`) and the Merkle position is `index >> 1`. §1, §4.
8. Folds = layers **+ 1**; the first fold has no Merkle check. §1.
9. The three structural length checks must run **before** the query loop. §1.
10. `zetas[i+1]` in the loop, `zetas[0]` for the first fold — off-by-one here verifies nothing. §1.

---

# Addendum — the fri leg's own measurements and decisions

Appended 2026-07-31 by the fri leg. The spec above is the scout's; this section
is FIRST-HAND from this worktree and is where the two disagree or the spec is
silent.

## ★ The fixture folds NOTHING — the leg's instrument problem

**Measured, not inferred**, off the real proof: `fri_layers_merkle_roots = 0`,
`fri_final_poly_coeffs = 4`, 219 query decommitments. Pinned by
`join_tests::the_fixture_carries_no_fri_layers_so_it_cannot_witness_the_fold`.

The fixture is the `min` preset over a `2^4`-step epoch, so its sub-proof has
`log2(lde) = 3`; §2's arithmetic gives `terminal_log = min(1+7, 3) = 3`,
`total_folds = 0`, `num_committed = 0`, and `query_phase` takes its
empty-decommitment branch.

This is §7's blindness taken one step further. §7 says a differential over real
proofs cannot distinguish implementations that differ only off `k = 7` /
`coset_offset = 3`. On the fixture specifically it is worse: **the production
instance exercises none of the mechanism at all** — no fold, no walk, no
terminal lookup. An emitter differentialled only against it would fold nothing
and pass everything.

Consequence: the primary instrument is synthetic codewords driven through
production's own `commit_phase_from_evaluations` + `query_phase`, differentialled
against the verifier's own check, with `num_committed` swept. Only the INPUT is
synthetic; it remains a differential against production code.

## Correction to §4 — and to what I first reported

I told the team lead the FRI layer leaf is committed under
`PairKeccak256Backend` and **"not the trace's `BatchedMerkleTreeBackend`"**.
That is true prover-side and **wrong as a statement about the verify path**,
which is what the machine emits. §4's "both, and they are byte-identical" is the
correct account: `verify_fri_layer_openings` (`verifier.rs:643`) calls
`verify_merkle_path::<BatchedMerkleTreeBackend<FieldExtension>>` over a
two-element vector. The emitted leaf bytes are unaffected — both stream the two
elements into one fresh keccak — but the claim as I stated it was wrong.

The verify-side reading also surfaces something the prover-side reading hides:
**the leaf ordering is parity-dependent** (`verifier.rs:637-641`,
`if iota % 2 == 1 { [sym, v] } else { [v, sym] }`). The verifier holds the
folded `v` and receives `sym`, so the machine must SELECT the order on the low
index bit. Reading only the prover's `chunks_exact(2)` would have missed it.

## Predictions, written BEFORE measuring

Per §8, with `n = trace_bits + b`, `C = num_committed = trace_bits − 8` at
`k = 7`, `steps = C(n−2) − C(C−1)/2`, `perms/query = C + steps`:

| blowup | b | n  | C  | Q   | steps/q | perms/q | FRI perms/epoch-table |
|--------|---|----|----|-----|---------|---------|-----------------------|
| 2      | 1 | 21 | 12 | 219 | 162     | 174     | **38,106**            |
| 4      | 2 | 22 | 12 | 110 | 174     | 186     | **20,460**            |
| 8      | 3 | 23 | 12 |  73 | 186     | 198     | **14,454**            |

at `trace_bits = 20`. The blowup-2 row reproduces §8's worked example exactly,
which is the check that the formula is being read as written rather than
re-derived by guess.

⚠ This also corrects an arithmetic slip in my first report to the team lead: I
quoted "180 path steps, ≈192 perms/query" for blowup 8. The correct figures are
**186 and 198** — I mis-summed `Σ_{i=0}^{11}(21−i)`.

**Standalone result worth carrying out of this leg:** FRI is **2.6× cheaper at
blowup 8 than at blowup 2** (14,454 vs 38,106), because the query count falls
3× while per-query cost rises only 14%. The blowup-8 decision was made on DEEP
and on the keccak bill; this is an independent third leg pointing the same way.

## Decision on §7's dead branches — deferral with its argument

§7 lists the clamp path, the `zetas.is_empty()` no-fold branch, and
`effective_k != k` as unreachable under production presets. The team-lead
charter requires either synthetic coverage or a structural pin. **Neither
option is quite right for this machine, and the reason is worth stating.**

In LFM, shape is COMPILE-TIME. `num_committed`, `terminal_len` and
`effective_k` are program constants, so none of these is emitted control flow —
there is no branch in the program to leave dead. What exists instead is:

1. **Host-side shape arithmetic** in the emitter (the `FriFoldLayout`
   computation, clamp included). This is ordinary Rust running at program-build
   time, so it is covered by ordinary unit tests over synthetic
   `(trace_bits, blowup, k)` — including `k ∈ {0, 6, 63}` and `trace_bits ≤ 7` —
   at no proving cost. This is where §7's requirement is discharged.
2. **Two program SHAPES**: `num_committed = 0` and `num_committed > 0`. Both are
   emitted and both are differentialled. The zero case is not a dead branch to
   pin — it is the fixture's own shape and a real production path for small
   tables.

So: no dead program text is emitted, and nothing is left unexercised. The one
thing genuinely NOT covered is a proof whose `coset_offset ≠ 3`, because no
production configuration produces one and the LDE domain constants are baked
into the program; that is a deferral, and its safety argument is that a wrong
coset offset changes every domain point and therefore every leaf, so it cannot
produce a passing proof — it can only fail. Stated rather than assumed.
