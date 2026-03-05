# LogUp-GKR Design: Eliminating Auxiliary Traces

**Date:** 2026-03-05
**Branch:** `feat/logup-gkr`
**Status:** Design

## 1. Motivation

Lambda VM currently uses committed auxiliary trace columns for LogUp lookup arguments. After batching and absorption optimizations, 55 auxiliary columns remain across 12 tables. Each auxiliary column requires:

- LDE computation (FFT from trace domain to evaluation domain)
- Merkle tree commitment (Keccak256 hashing)
- Constraint evaluation at every LDE domain point
- FRI query openings

LogUp-GKR replaces committed auxiliary columns with an interactive GKR proof that verifies the LogUp sum directly. The auxiliary trace is eliminated entirely (except for a single Lagrange kernel column per table for the MLE bridge).

**Expected gains:**
- 55 aux columns → 1 per table with interactions (≤12 total)
- Eliminate all LogUp constraint evaluation (67 constraints across all tables)
- Fewer Merkle tree commitments and FRI query openings
- Proof size reduction (no aux column data at query points)

**What does NOT change:**
- Max constraint degree stays at 3 (table constraints like `cond * carry * (1-carry)` already have degree 3)
- Composition polynomial still has 3 parts
- Main trace structure unchanged
- BusInteraction definitions unchanged (tables still declare their interactions the same way)
- Bus balance check: Σ table_contribution = 0

## 2. Background

### 2.1 Current LogUp Architecture

Each table declares bus interactions via `bus_interactions() -> Vec<BusInteraction>`. After the prover commits main traces and samples challenges (z, α), the auxiliary trace is built:

1. **Fingerprints:** `fp_k(ω^i) = z - (bus_id·α⁰ + v₀·α¹ + v₁·α² + ...)`
2. **Batched terms:** Pairs of interactions share one aux column via cleared-denominator identity
3. **Absorbed interactions:** Last 1-2 interactions folded into accumulated constraint (virtual, not committed)
4. **Accumulated column:** Running sum with circular constraint (acc[N-1] = 0)
5. **Constraints:** `LookupBatchedTermConstraint` (degree 3), `LookupAccumulatedConstraint` (degree 2-3)

### 2.2 GKR Protocol

The GKR (Goldwasser-Kalai-Rothblum) protocol proves claims about layered arithmetic circuits. For a circuit with L layers, GKR reduces a claim about the output to claims about the input through L rounds of sumcheck, each reducing the claim by one layer.

**Key properties:**
- Prover work: O(N) per layer (linear scan of current layer)
- Total prover work: O(N·L) for L layers
- Verifier work: O(L²) field operations (very fast)
- Proof size: O(L·d) field elements per layer, where d is the sumcheck degree

### 2.3 LogUp-GKR (Haböck, ePrint 2023/1284)

LogUp-GKR applies GKR to prove the LogUp sum identity:

```
Σ_{i=0}^{N-1} Σ_{k=1}^{K} sign_k · m_k(ω^i) / fp_k(ω^i) = L
```

Instead of committing auxiliary columns for the partial-fraction terms, the prover runs a GKR proof on a binary summation tree where each leaf holds the combined rational value for row i, and internal nodes sum fractions via cross-multiplication:

```
a/b + c/d = (a·d + b·c) / (b·d)
```

### 2.4 Univariate-to-Multilinear Bridge

GKR operates on multilinear extensions (MLEs) over the Boolean hypercube {0,1}^n. Lambda VM uses univariate polynomials over the multiplicative group H = <ω> of order N = 2^n.

The **squaring-tower bijection** maps:
```
ω^i ↔ bits(i) = (b₀, b₁, ..., b_{n-1}) ∈ {0,1}^n
```

where i = b₀ + 2·b₁ + ... + 2^{n-1}·b_{n-1}. This establishes a one-to-one correspondence between evaluations on H and evaluations on {0,1}^n.

For a trace column t with values t(ω^i), the MLE t̃ is the unique multilinear polynomial satisfying:
```
t̃(bits(i)) = t(ω^i) for all i ∈ [N]
```

After GKR completes, the verifier holds claims of the form t̃(r) = c for random r ∈ F^n. The bridge proves these claims are consistent with the committed univariate trace polynomial.

## 3. Protocol Design

### 3.1 High-Level Protocol Flow

```
ROUND 1:
  Phase A: Commit main traces for all 12 tables
           → Merkle roots into transcript

  Phase B: Sample z, α from transcript (LogUp fingerprint challenges)

  Phase B' [NEW]: GKR sub-protocol
    For each table with interactions (sequentially or in parallel):
      1. Compute leaf values: h(i) = Σ_k sign_k · m_k(i) / fp_k(i)
      2. Run fractional-sum GKR over depth-n binary tree
      3. GKR produces random point r_j and MLE evaluation claims
      4. Append GKR transcript to main transcript
    Verify bus balance: Σ L_j = 0

  Phase C [MODIFIED]: Build Lagrange kernel auxiliary trace
    For each table with interactions:
      1. Compute s(ω^i) = eq(bits(i), r_j) via butterfly
      2. Commit s-column (1 column per table, vs 55 total previously)

ROUND 2: Composition polynomial
  → Standard AIR constraints (degree ≤ 3, unchanged)
  → Lagrange kernel transition constraints (degree 2)
  → Bridge inner-product constraints (degree 2)
  → 3 composition parts (unchanged, since table constraints already have degree 3)

ROUND 3: OOD evaluation
  → Main trace columns at OOD point (unchanged)
  → 1 aux column (s-column) per table (vs ~55 aux columns previously)

ROUND 4: FRI + queries
  → Same structure, but far fewer aux column openings
```

### 3.2 GKR Circuit for Fractional Summation

For a table with K interactions and N = 2^n rows:

**Input layer (N leaves):**

Each leaf i computes the combined fraction for all K interactions at row i. To avoid computing K separate fractions, we batch interactions:

For K interactions, compute per row i:
- Numerator: N(i) = Σ_k sign_k · m_k(i) · Π_{j≠k} fp_j(i)
- Denominator: D(i) = Π_k fp_k(i)

The leaf value is the pair (N(i), D(i)), representing the fraction N(i)/D(i).

**Internal layers (depth 1 to n):**

Each node combines two child fractions:
```
(n_left, d_left) + (n_right, d_right) = (n_left·d_right + n_right·d_left, d_left·d_right)
```

**Output (root):**

Single fraction (N_total, D_total). The claim is N_total/D_total = L (table contribution).

**GKR reduction per layer:**

At layer l, the GKR verifier holds a claim about V_l(r_l) where V_l is the MLE of layer-l values. The sumcheck reduces this to claims about V_{l+1} at specific points.

For fraction-addition gates, the sumcheck polynomial has degree ≤ 3 per variable (products of two children's numerators/denominators). Each sumcheck round produces a degree-3 univariate polynomial, specified by 4 field elements.

**Total GKR proof size per table:**

n layers × variable rounds per layer. For the standard binary-tree GKR:
- Layer l has n-l variables
- Each round: 4 extension field elements (degree-3 polynomial)
- Total: Σ_{l=0}^{n-1} (n-l) × 4 = 4 · n·(n+1)/2 = 2n² + 2n field elements
- For n=20: ~840 extension field elements (~20 KB with Goldilocks cubic extension)

### 3.3 GKR Implementation Details

#### 3.3.1 Bookkeeping Tables

The prover maintains a "bookkeeping table" for each layer: the current MLE values on the remaining hypercube {0,1}^{vars} where vars decreases by 1 per sumcheck round within a layer.

**Memory:** O(N) per table (one bookkeeping table at a time, halved per layer).

**Computation per sumcheck round:**

For each point x in the remaining hypercube (size 2^{vars-1}):
1. Evaluate the gate polynomial at x with the current variable set to 0, 1, 2, 3 (degree 3 requires 4 evaluations)
2. Interpolate to get the round polynomial coefficients

Total work per layer: O(2^{n-l}) for layer l.
Total work across all layers: O(N) (geometric series).

#### 3.3.2 Batching GKR Across Interactions

Rather than running separate GKR instances per interaction, we batch all K interactions into a single GKR circuit:

**Option 1 (per-row reduction first):** Sum all K fractions per row into one fraction, then run GKR on N single fractions. This is what stwo does.

**Option 2 (interleaved):** Run GKR on K·N leaves where each interaction contributes N leaves. Use random linear combination to batch.

We choose **Option 1** as it produces a cleaner circuit with smaller bookkeeping tables and matches the stwo approach.

**Per-row fraction computation (prover):**

For each row i, compute (N(i), D(i)) where:
```
D(i) = Π_{k=1}^{K} fp_k(i)
N(i) = Σ_{k=1}^{K} sign_k · m_k(i) · Π_{j≠k} fp_j(i)
```

This costs O(K²) field operations per row (or O(K log K) with product trees), O(N·K²) total per table. For the largest tables (K ≈ 20), this is ~400·N ≈ 400M operations for N = 2^20.

**Optimization:** Reuse the existing packing/fingerprint computation from BusInteraction. The fingerprints fp_k(i) are linear functions of trace columns and can be computed incrementally.

#### 3.3.3 Random Point and MLE Claims

After the GKR protocol, the verifier holds:
1. **Table contribution:** L_j (verified by GKR)
2. **Random point:** r_j = (r_1, ..., r_{n_j}) sampled via Fiat-Shamir during GKR
3. **Input-layer MLE claims:** N_tilde(r_j) = v_N and D_tilde(r_j) = v_D

The verifier checks that v_N / v_D = L_j (or equivalently v_N = L_j · v_D).

**Reducing to trace column claims:**

The denominator D(i) = Π_k fp_k(i) and numerator N(i) = Σ_k sign_k · m_k(i) · Π_{j≠k} fp_j(i) are polynomial functions of the trace columns. Their MLEs at r_j are determined by the MLEs of the individual trace columns at r_j.

Specifically, fp_k(i) = z - (bus_id_k · α⁰ + v_{k,0}(i) · α¹ + ...) is a linear function of trace column values. So:

```
fp_k_tilde(r) = z - (bus_id_k · α⁰ + v_{k,0}_tilde(r) · α¹ + ...)
```

The prover provides the MLE evaluations col_tilde(r_j) for each distinct trace column used in interactions. The verifier reconstructs fp_k_tilde(r_j) and checks the numerator/denominator relationship.

**Number of distinct column claims per table:**

Each bus interaction references several columns via BusValue::Packed or BusValue::Linear, plus a multiplicity column. The number of distinct columns is bounded by the main column count but in practice much smaller (many interactions share columns). Estimated: 15-40 distinct column claims per table.

### 3.4 Lagrange Kernel Column (s-column)

The Lagrange kernel column bridges MLE evaluation claims back to committed univariate traces.

#### 3.4.1 Construction

For table j with GKR random point r_j = (r_1, ..., r_n):

```
s(ω^i) = eq(bits(i), r_j) = Π_{l=0}^{n-1} [r_l · b_l(i) + (1 - r_l) · (1 - b_l(i))]
```

where b_l(i) = (i >> l) & 1 is the l-th bit of i.

**Butterfly computation (O(N)):**
```rust
fn compute_lagrange_kernel(r: &[FE], n: usize) -> Vec<FE> {
    let len = 1 << n;
    let mut s = vec![FE::one(); len];
    for j in 0..n {
        let rj = &r[j];
        let one_minus_rj = FE::one() - rj;
        for i in 0..len {
            if (i >> j) & 1 == 1 {
                s[i] *= rj;
            } else {
                s[i] *= &one_minus_rj;
            }
        }
    }
    s
}
```

#### 3.4.2 Transition Constraint

The s-column satisfies a degree-2 transition constraint based on the product structure of eq. The constraint verifies that consecutive s values differ by the correct ratio determined by the bit changes.

**Approach (Winterfell/Miden):** Define the Lagrange kernel constraint using the recursive structure:

```
eq(b, r) = Π_{l=0}^{n-1} (r_l · b_l + (1-r_l) · (1-b_l))
```

The key property: when only bit b_j changes (from 0 to 1), the ratio is:
```
eq(..., b_j=1, ...) / eq(..., b_j=0, ...) = r_j / (1 - r_j)
```

For the ordering ω^0, ω^1, ..., ω^{N-1}, consecutive indices i and i+1 differ in a specific bit pattern. The transition from i to i+1 flips bit 0 and potentially carries to higher bits.

**Simplified constraint:** The transition constraint checks:
```
s(ω^{i+1}) = s(ω^i) · factor(i)
```
where factor(i) depends on which bits change from i to i+1. This can be expressed as a degree-2 constraint.

Specifically, define the "frame ratio":
```
s[i+1] / s[i] = Π_{l: b_l changes} (r_l · b_l(i+1) + (1-r_l)(1-b_l(i+1))) / (r_l · b_l(i) + (1-r_l)(1-b_l(i)))
```

For the standard bit-reversal ordering (which matches the squaring tower), only one bit changes per step in certain orderings. But in natural ordering, multiple bits can change.

**Alternative: Direct verification.** Instead of a transition constraint on consecutive rows, verify the s-column values directly via the product structure. The prover commits s, and the verifier checks:
1. A single random evaluation s(ζ) at the OOD point
2. The product-structure constraint: s(ζ) should equal Π_{l=0}^{n-1} [r_l · φ_l(ζ) + (1-r_l)(1-φ_l(ζ))] where φ_l(ζ) are the bit-extraction functions evaluated at ζ

**Chosen approach: Logarithmic decomposition.**

Define n "partial product" columns p_0, ..., p_{n-1} where:
```
p_0(ω^i) = r_0 · b_0(i) + (1-r_0) · (1-b_0(i))
p_1(ω^i) = p_0(ω^i) · [r_1 · b_1(i) + (1-r_1) · (1-b_1(i))]
...
p_{n-1}(ω^i) = s(ω^i) = eq(bits(i), r)
```

But this requires n auxiliary columns, which is worse than 1.

**Final approach: Single constraint using the "next-row" trick.**

Since i and i+1 differ only in the trailing bits (binary increment), and b_0 always flips:

When i is even (b_0 = 0): i+1 has b_0 = 1, all other bits same.
```
s[i+1] / s[i] = r_0 / (1 - r_0)
```

When i is odd (b_0 = 1): i+1 carries. The exact bit pattern depends on how many trailing 1s there are.

**Practical constraint:** We use a "Lagrange kernel transition" approach inspired by Miden VM:

The s-column satisfies: for all i ∈ [0, N-2]:
```
s[i+1] · ((1-r_0)(1-b_0(i)) + r_0·b_0(i))  =  s[i] · ((1-r_0)·b_0(i) + r_0·(1-b_0(i))) · R(i)
```

where R(i) accounts for higher-bit transitions. However, R(i) is complex.

**Simplest sound approach:** Commit s, verify at OOD point using the product formula. The verifier computes eq(bits, r) at ζ using:

```
s_expected(ζ) = Σ_{i=0}^{N-1} eq(bits(i), r) · L_i(ζ)
```

where L_i(ζ) are Lagrange basis evaluations. This is O(N) for the verifier using barycentric interpolation with known weights eq(bits(i), r). This is exactly the bridge: the verifier checks s(ζ) against the expected value computed from r.

**This means the s-column needs only a boundary/evaluation constraint, not a complex transition constraint.** The constraint is:

```
s(ζ) = Σ_{i=0}^{N-1} eq(bits(i), r) · L_i(ζ)
```

The prover commits s, and the DEEP protocol checks s at the OOD point. The verifier recomputes the expected value from r using barycentric interpolation.

#### 3.4.3 Bridge: MLE Evaluation Proof

For each trace column col with claim col_tilde(r) = c:

```
col_tilde(r) = Σ_{i=0}^{N-1} col(ω^i) · eq(bits(i), r)
             = Σ_{i=0}^{N-1} col(ω^i) · s(ω^i)
             = <col, s>_H
```

This inner product is proven as a composition polynomial quotient:

**Define:** p(X) = col(X) · s(X). Then:
```
Σ_{a ∈ H} p(a) = c
```

This is equivalent to: (p(X) - c/N) vanishes as a sum over H, i.e.:
```
Σ_{a ∈ H} [col(a) · s(a) - c/N] = 0
```

**Quotient form:** The polynomial col(X)·s(X) - c/N has degree < 2N. Its sum over H being zero means there exists q(X) of degree < N such that:

```
col(X) · s(X) - c/N = q(X) · (X^N - 1) + remainder terms
```

Wait — the sum being zero does not directly give a quotient. Let's use the standard univariate sumcheck approach.

**Univariate sumcheck for inner product:**

Claim: Σ_{a ∈ H} col(a) · s(a) = c.

Define f(X) = col(X) · s(X). Then Σ_{a ∈ H} f(a) = c.

The polynomial f(X) - c/N has the property that Σ_{a ∈ H} (f(a) - c/N) = 0. By the sumcheck lemma, if g(X) satisfies Σ_{a ∈ H} g(a) = 0 and deg(g) < 2N, then there exists q(X) with deg(q) < N such that:

```
g(X) = q(X) · v_H(X)
```

where v_H(X) = (X^N - 1)/N · X^{-1} ... no, this is not right for general sums.

**Correct approach:** For the multiplicative subgroup H of order N, if Σ_{a ∈ H} g(a) = 0 then Z_H(X) = X^N - 1 divides the "sum-adjusted" polynomial.

Actually, the standard trick is: Σ_{a ∈ H} f(a) = N · f_0 where f_0 is the constant term of f(X) mod (X^N - 1). The claim Σ f(a) = c is equivalent to f_0 = c/N.

For DEEP-style verification: the verifier queries f(ζ) = col(ζ) · s(ζ) at the OOD point. Both col(ζ) and s(ζ) are available from the OOD evaluation. The verifier has both values. But this alone doesn't prove the sum.

**Simplification: Running sum approach.**

Instead of a univariate sumcheck, introduce a running-sum column σ:
```
σ(ω^0) = col(ω^0) · s(ω^0)
σ(ω^{i+1}) = σ(ω^i) + col(ω^{i+1}) · s(ω^{i+1})
```

Then σ(ω^{N-1}) = c (the total inner product).

**Constraint:** σ(ω·X) - σ(X) - col(ω·X) · s(ω·X) = 0 (degree 2 in aux columns)
**Boundary:** σ(ω^{N-1}) = c

But this requires another committed column σ, bringing us to 2 aux columns per table. Still much better than 55 total.

**Better: Fold all claims into one running sum.**

Since all bridge claims share the same s-column, define a single σ column that accumulates all inner products via random linear combination:

Sample bridge challenge γ after committing s. Define:
```
σ(ω^i) = Σ_{j=1}^{M} γ^j · Σ_{l=0}^{i} col_j(ω^l) · s(ω^l)
```

where col_1, ..., col_M are the distinct columns with MLE claims.

This requires only 1 additional column σ (total: 2 aux columns, s and σ).

**Transition constraint:** σ(ω·X) - σ(X) = s(ω·X) · Σ_j γ^j · col_j(ω·X) (degree 2)
**Boundary:** σ(ω^{N-1}) = Σ_j γ^j · c_j

**However**, this couples the s-column commitment with a second challenge round (for γ), complicating the protocol flow.

**Simplest viable approach: Direct OOD quotient.**

After committing s, the bridge claims can be verified at the OOD point without a running sum, IF we add the bridge as part of the composition polynomial.

For each column claim col_tilde(r) = c, the DEEP quotient is:

```
bridge_j(X) = [col(X) · s(X) · N - c] / (X^N - 1)
```

Since col(X) has degree < N and s(X) has degree < N, the numerator has degree < 2N, and the quotient has degree < N. This fits within the existing composition polynomial framework.

The prover evaluates bridge_j on the LDE domain and adds it (with random coefficient) to the composition polynomial.

The verifier checks at ζ:
```
bridge_j(ζ) = [col(ζ) · s(ζ) · N - c] / (ζ^N - 1)
```

All values are available: col(ζ) from main trace OOD, s(ζ) from aux trace OOD, c from the GKR proof, N and ζ^N - 1 are computed.

**This adds zero extra committed columns beyond s.** Each bridge quotient is just another term in the composition polynomial.

**Number of bridge quotient terms:** Equal to the number of distinct trace columns claimed. Across all tables, this is bounded by the total number of distinct main columns used in interactions. Estimated: 100-150 terms across all 12 tables. Each adds one degree-2 quotient evaluation per LDE point.

#### 3.4.4 Final Architecture: 1 Aux Column Per Table

**Committed auxiliary:** Only the s-column (Lagrange kernel) per table.
**Bridge:** Direct OOD quotient terms in the composition polynomial.
**No running-sum column needed.**

Total aux columns: ≤12 (one per table with interactions, some tables like PAGE and HALT have 0 interactions).

### 3.5 Verifier Protocol

The verifier performs:

1. **Replay GKR** for each table:
   - Read the GKR proof (sumcheck round polynomials)
   - Verify each sumcheck round (evaluate round polynomial at 0 and 1, check sum)
   - Reduce layer by layer to input-layer claims
   - Extract random point r_j and MLE claims (N_tilde(r_j), D_tilde(r_j))
   - Verify N_tilde(r_j) = L_j · D_tilde(r_j)

2. **Reduce MLE claims to trace column claims:**
   - From N_tilde and D_tilde, compute individual col_tilde(r_j) = c for each trace column
   - These are linear operations on the column MLE values

3. **Check bus balance:** Σ L_j = 0

4. **Verify s-column (Lagrange kernel):**
   - At OOD point ζ: compute expected s(ζ) = Σ_{i=0}^{N-1} eq(bits(i), r) · L_i(ζ)
   - This uses barycentric interpolation: O(N) field operations
   - Check s(ζ) from the proof matches

5. **Verify bridge quotients:**
   - For each column claim col_tilde(r) = c:
     - Check: bridge(ζ) = [col(ζ) · s(ζ) · N - c] / (ζ^N - 1)
   - These are folded into the composition polynomial check

6. **Standard STARK verification** (Rounds 2-4 unchanged structurally)

**Verifier complexity:**
- GKR verification: O(n²) per table (n sumcheck rounds per layer, n layers)
- Lagrange kernel check: O(N) per table (barycentric interpolation)
- Bridge quotients: O(1) per claim (at OOD point)
- Total verifier overhead: O(N) dominated by the Lagrange kernel check

**Note:** The O(N) Lagrange kernel verification can be made O(n) by using the FRI-as-PCS trick (Approach C), but we accept O(N) verifier cost for implementation simplicity. For N = 2^20, this is ~1M field operations — fast enough in practice.

### 3.6 GKR Proof Structure

```rust
/// Proof of the LogUp-GKR sub-protocol for one table.
pub struct LogUpGkrProof<E: IsField> {
    /// The claimed table contribution (Σ of all LogUp terms).
    pub table_contribution: FieldElement<E>,

    /// GKR layer proofs, from output (root) to input (leaves).
    pub layer_proofs: Vec<GkrLayerProof<E>>,

    /// MLE evaluation claims for trace columns at the GKR random point.
    /// Maps column index → claimed evaluation value.
    pub column_claims: Vec<(usize, FieldElement<E>)>,
}

/// Proof for one GKR layer (sumcheck instance).
pub struct GkrLayerProof<E: IsField> {
    /// Round polynomials for the sumcheck in this layer.
    /// Each polynomial is represented by its evaluations at 0, 1, 2, 3.
    pub round_polys: Vec<[FieldElement<E>; 4]>,
}
```

### 3.7 Modified Prover Flow

```rust
fn multi_prove(...) {
    // Phase A: Commit main traces (UNCHANGED)
    let main_commits = commit_main_traces(&tables, &transcript);

    // Phase B: Sample LogUp challenges (UNCHANGED)
    let (z, alpha) = sample_logup_challenges(&mut transcript);

    // Phase B': NEW — Run GKR sub-protocol
    let gkr_results: Vec<LogUpGkrResult> = tables
        .iter()
        .map(|table| {
            if table.has_bus_interactions() {
                run_logup_gkr(table, &main_traces, z, alpha, &mut transcript)
            } else {
                LogUpGkrResult::empty()
            }
        })
        .collect();

    // Verify bus balance
    assert_eq!(
        gkr_results.iter().map(|r| &r.table_contribution).sum(),
        FieldElement::zero()
    );

    // Phase C: MODIFIED — Build and commit Lagrange kernel columns
    let aux_traces: Vec<Option<TraceTable>> = tables
        .iter()
        .zip(&gkr_results)
        .map(|(table, gkr)| {
            if gkr.has_claims() {
                let s = compute_lagrange_kernel(&gkr.random_point, table.trace_length_log2());
                Some(TraceTable::from_columns(vec![s]))
            } else {
                None
            }
        })
        .collect();

    // Commit aux traces (1 column each, vs 55 previously)
    let aux_commits = commit_aux_traces(&aux_traces, &mut transcript);

    // Rounds 2-4: MODIFIED
    // - No LogUp constraints in composition polynomial
    // - Add bridge quotient terms for MLE claims
    // - Add Lagrange kernel verification terms
    prove_rounds_2_to_4(main_commits, aux_commits, gkr_results, ...);
}
```

### 3.8 Constraint Changes

**Removed constraints:**
- `LookupBatchedTermConstraint` (all instances)
- `LookupAccumulatedConstraint` (all instances)

**Added constraints (per table with interactions):**
- **Bridge quotient terms** in composition polynomial: one per distinct trace column claim
  - Degree 2: col(X) · s(X) is degree < 2N, quotient is degree < N
- **Lagrange kernel verification** at OOD point (verifier-side only, not a transition constraint)

**Net constraint reduction:**
- Removed: 67 transition constraints across all tables (55 batched term + 12 accumulated)
- Added: ~100-150 bridge quotient terms in composition polynomial (but these are simple degree-2 evaluations, not transition constraints with zerofier overhead)

### 3.9 Sequential Per-Table Proving Compatibility

The current `multi_prove` uses sequential per-table proving (P3.1) to minimize peak memory. LogUp-GKR is fully compatible:

- **Phase A:** Main trace commits unchanged
- **Phase B':** GKR runs sequentially per table, storing only the `LogUpGkrProof` (compact: ~1-2 KB per table)
- **Phase C:** Lagrange kernel column is 1 column per table (tiny compared to 55 aux columns)
- **Rounds 2-4:** `reconstruct_round1` recomputes LDE as before. Bridge quotients are computed during composition polynomial evaluation. No additional memory pressure.

Peak memory is actually **reduced** because the aux trace is 1 column instead of ⌈K/2⌉ columns per table.

## 4. Data Flow

```
┌─────────────────────────────────────────────────────┐
│ Main Trace (committed)                               │
│ CPU: 74 cols, MEMW: 70 cols, ... (unchanged)        │
└───────────────┬─────────────────────────────────────┘
                │
        sample z, α
                │
                ▼
┌─────────────────────────────────────────────────────┐
│ GKR Sub-protocol (per table)                         │
│                                                      │
│  Input: main trace values + fingerprints (z, α)     │
│  Compute: h(i) = Σ_k sign_k · m_k(i) / fp_k(i)    │
│  Build: binary summation tree of fractions           │
│  Run: layered sumcheck (depth n)                     │
│  Output: L_j, r_j, column_claims                    │
│                                                      │
│  Proof: round polynomials per layer (~1-2 KB/table) │
└───────────────┬─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────┐
│ Bus Balance: Σ L_j = 0                               │
└───────────────┬─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────┐
│ Lagrange Kernel (1 aux column per table)             │
│                                                      │
│  s(ω^i) = eq(bits(i), r_j)                          │
│  Computed via O(N) butterfly                         │
│  Committed via LDE + Merkle tree                    │
└───────────────┬─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────┐
│ Composition Polynomial (Round 2)                     │
│                                                      │
│  Standard AIR constraints (unchanged)               │
│  + Bridge quotients: [col·s·N - c] / Z_H           │
│  No LogUp transition constraints                    │
│                                                      │
│  Still 3 parts (table constraints have degree 3)    │
└───────────────┬─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────┐
│ DEEP + FRI (Rounds 3-4)                              │
│  1 aux column opening per query (vs ~55 previously) │
└─────────────────────────────────────────────────────┘
```

## 5. Implementation Plan

### Phase 1: GKR Core (crypto/stark)

1. **Sumcheck module** (`crypto/stark/src/sumcheck.rs`)
   - Univariate round polynomial representation
   - Sumcheck prover: generate round polynomials from evaluations
   - Sumcheck verifier: check round polynomial consistency

2. **GKR module** (`crypto/stark/src/gkr.rs`)
   - Layer representation (bookkeeping table)
   - GKR prover: reduce from output to input layer
   - GKR verifier: replay and verify layer proofs
   - Fractional-sum gate type (numerator/denominator pairs)

3. **Proof types** (`crypto/stark/src/proof.rs`)
   - `LogUpGkrProof`, `GkrLayerProof` structs
   - Serialization/deserialization

### Phase 2: Lagrange Kernel & Bridge (crypto/stark)

4. **Lagrange kernel** (`crypto/stark/src/lagrange_kernel.rs`)
   - `compute_lagrange_kernel(r, n)` → eq weight vector
   - Bridge quotient evaluation helper

5. **Modified lookup.rs** (`crypto/stark/src/lookup.rs`)
   - `run_logup_gkr()` function replacing `build_auxiliary_trace()`
   - Compute leaf values (per-row fraction combining)
   - Build GKR circuit and run prover
   - Extract column claims from GKR output
   - Keep: `BusInteraction`, `BusValue`, `Multiplicity`, fingerprint computation
   - Remove: `LookupBatchedTermConstraint`, `LookupAccumulatedConstraint`, batch term column computation, accumulated column construction

### Phase 3: Prover Integration

6. **Modified prover.rs** (`crypto/stark/src/prover.rs`)
   - Add Phase B' (GKR sub-protocol) in `multi_prove`
   - Modify Phase C to build only Lagrange kernel columns
   - Add bridge quotient evaluation in composition polynomial
   - Update `reconstruct_round1` for simplified aux trace

7. **Modified evaluator.rs** (`crypto/stark/src/constraints/evaluator.rs`)
   - Remove LogUp constraint evaluation from `evaluate_transitions`
   - Remove `logup_table_offset` and `logup_alpha_powers` from `TransitionEvaluationContext`
   - Add bridge quotient terms to composition polynomial evaluation

### Phase 4: Verifier Integration

8. **Modified verifier.rs** (`crypto/stark/src/verifier.rs`)
   - Add GKR verification step
   - Replace bus balance check (now uses GKR-proven contributions)
   - Add Lagrange kernel verification (barycentric check at OOD point)
   - Add bridge quotient verification at OOD point
   - Remove aux column constraint verification for LogUp

### Phase 5: Table Updates

9. **Table AIR modifications** (all tables in `prover/src/tables/`)
   - Remove `num_auxiliary_columns()` LogUp component
   - Keep `bus_interactions()` definitions unchanged
   - Update `transition_constraints()` to not include LogUp constraints
   - Update tests

### Phase 6: Testing & Benchmarks

10. **Tests**
    - Unit tests for sumcheck prover/verifier
    - Unit tests for GKR prover/verifier
    - Unit tests for Lagrange kernel computation
    - Integration tests: full prove/verify cycle with GKR
    - Regression tests: verify same programs produce valid proofs

11. **Benchmarks**
    - Compare proving time before/after
    - Memory usage comparison
    - Proof size comparison

## 6. Security Analysis

### 6.1 Soundness

The LogUp-GKR protocol has three components, each with independent soundness:

1. **GKR soundness:** Each layer's sumcheck has error ≤ d/|F| per round, where d is the polynomial degree (≤ 3). With n rounds per layer and n layers, total error ≤ 3n²/|F|. For n = 20, |F| = 2^64: error ≤ 1200/2^64 ≈ 2^{-53}. Well within security margin.

2. **Lagrange kernel commitment:** The s-column is a degree < N polynomial committed via Merkle tree. FRI ensures the committed polynomial has correct degree. The OOD evaluation check verifies consistency with r.

3. **Bridge quotients:** Each quotient [col·s·N - c] / Z_H is verified at the OOD point. If the inner product claim is false, the quotient has a pole on H, which FRI will catch with high probability.

### 6.2 Transcript Binding

All GKR messages are bound to the Fiat-Shamir transcript:
- GKR round polynomials are hashed into the transcript
- The random point r is derived from the transcript
- Column claims are bound to the transcript before Phase C
- The Lagrange kernel commitment (Phase C) is bound before Round 2

This ensures the prover cannot adaptively choose values after seeing challenges.

### 6.3 Extension Field

GKR operates over the cubic extension F_{p^3} (same as current LogUp). All fingerprints, MLE evaluations, and sumcheck polynomials use extension field arithmetic. The Lagrange kernel column s uses extension field values (since r ∈ (F_{p^3})^n).

## 7. Performance Estimates

### 7.1 Prover Cost Comparison (N = 2^20, largest table)

| Operation | Current LogUp | LogUp-GKR |
|-----------|--------------|-----------|
| Aux trace build (fingerprints, batch inv) | ~120M ext-ops | 0 |
| Aux LDE (55 FFTs of size 2^21) | ~1.2B base-ops | 1 FFT |
| Aux Merkle (55 columns) | ~110M Keccak | 1 column |
| LogUp constraint eval (67 constraints) | ~1.4B ext-ops | 0 |
| GKR prover | 0 | ~60M ext-ops |
| Per-row fraction combining | 0 | ~400·N ≈ 420M ext-ops |
| Lagrange kernel (butterfly) | 0 | ~20·N ≈ 20M ext-ops |
| Bridge quotients in composition poly | 0 | ~150·blowup·N |

**Net estimate:** Current LogUp overhead ≈ 2.8B operations. LogUp-GKR overhead ≈ 800M operations. **~3.5x reduction** in LogUp-related work.

### 7.2 Proof Size Comparison

| Component | Current | LogUp-GKR | Savings |
|-----------|---------|-----------|---------|
| Aux Merkle roots | 12 roots | ≤12 roots (s-column) | ~0 |
| Aux column openings per query | 55 values | ≤12 values | ~43 values/query |
| GKR proof | 0 | ~2K ext elements | +2K elements |
| Column claims | 0 | ~150 elements | +150 elements |
| Composition poly parts | 3 | 3 | 0 |
| Total per query | ~55 + main | ~12 + main | ~43 values saved |

With ~30 FRI queries: 43 × 30 = 1290 ext field elements saved per proof. GKR adds ~2150 elements. **Net: roughly neutral proof size** (slight increase from GKR, offset by fewer query openings).

### 7.3 Verifier Cost

| Operation | Current | LogUp-GKR |
|-----------|---------|-----------|
| Constraint eval at OOD | 67 LogUp constraints | 0 |
| GKR verification | 0 | O(n²) per table |
| Lagrange kernel check | 0 | O(N) per table |
| Bridge quotient checks | 0 | O(#claims) per table |
| Merkle path verification | 55 aux paths | ≤12 aux paths |

Verifier is faster overall due to fewer Merkle path verifications, despite the O(N) Lagrange kernel check.

## 8. References

1. Haböck, U. "Multivariate lookups based on logarithmic derivatives." ePrint 2022/1530.
2. Haböck, U. "LogUp-GKR: A GKR-based protocol for LogUp." ePrint 2023/1284.
3. Goldwasser, S., Kalai, Y.T., Rothblum, G.N. "Delegating computation: interactive proofs for muggles." STOC 2008.
4. Thaler, J. "Proofs, arguments, and zero-knowledge." Chapter on GKR protocol.
5. Polygon Miden VM: Production LogUp-GKR implementation over Goldilocks.
6. StarkWare stwo: LogUp-GKR over M31 with circle STARKs.
7. Succinct SP1: LogUp-GKR over KoalaBear with hypercube STARKs.
