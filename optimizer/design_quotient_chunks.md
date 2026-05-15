# Design Doc — Lambda STARK migration to quotient chunks

**Status**: DRAFT (not approved, not implemented)
**Author**: optimizer workstream, 2026-05-15
**Scope**: Lambda VM's STARK prover + verifier
**Estimated effort**: 2-4 weeks implementation + 1 week tests + soundness review

---

## 1. Problem

Lambda STARK uses a **single composition polynomial H(x) committed over the
LDE 2N**. Plonky3 (and SP1, OpenVM) uses **quotient chunks committed
separately**. Two attempts to close the constraint-eval gap within Lambda's
single-H architecture failed:

- **Attempt 1** (`experiment/eval-d1-domain-n`): evaluate constraints on N
  instead of 2N for d_max=1 AIRs, extend via FFT. **Failed** because FFT
  extension in Fp3 cost 600-900 ms (~9× more than estimated).
- **Attempt 2** (`experiment/batched-lin-combination`): explicit 8-way ILP
  accumulator. **Failed** because LLVM already auto-ILP-s the scalar fold;
  no change.

The Lambda vs Plonky3 gap at log=21:

| cols | Lambda total | P3 total | Ratio |
|---:|---:|---:|---:|
|  32 | 1.877 s | 0.964 s | 1.95× |
|  64 | 2.741 s | 1.155 s | 2.46× |
| 128 | 4.728 s | 1.626 s | **2.93×** |

The gap is **structural**, intrinsic to the single-H architecture. Closing
it requires migrating to the quotient-chunks architecture used by P3.

## 2. Goal

Migrate Lambda STARK from "single composition polynomial committed over
LDE 2N" to "quotient evaluations committed as `next_pow2(d_max)` chunks,
each LDE'd independently to size 2N". This is a **protocol-level change**
(verifier, proof struct, FRI all change). Scope expansion authorized
2026-05-15 (see `project_optimizer_protocol_scope` memory).

**Success criteria**:
- All 124 stark tests pass under the new protocol
- `bench_vs_plonky3` at log=21, 32 cols shows ≥ 20% reduction in Lambda
  prove time vs baseline
- At log=21, 128 cols: ≥ 35% reduction (the worst-case gap currently)
- No regression on any test, no soundness regression (verified by audit
  + soundness argument below)

## 3. Architectural change

### 3.1 Current architecture (single H)

```
Round 2 (round_2_compute_composition_polynomial):
  1. constraint_evals: Vec<E>          # size 2N (eval on full LDE)
  2. Decide number_of_parts = d_max:
     - d_max=1:   1 part  (constraint_evals is the LDE directly)
     - d_max=2:   2 parts via decompose_and_extend_d2 (algebraic trick)
     - d_max=3+:  k parts via iFFT(LDE) + break_in_parts + LDE-per-part
  3. lde_composition_poly_parts_evaluations: Vec<Vec<E>>   # k vectors of size 2N
  4. composition_poly_merkle_tree: 1 Merkle over the concatenated parts
  5. composition_poly_root: 1 Commitment

Round 3 (OOD): evaluate each part at z^k (z raised to num_parts), sum
Round 4 (DEEP, FRI): single DEEP composition, single FRI run over LDE 2N

Verifier:
  - composition_poly_root: 1 Commitment
  - composition_poly_parts_ood_evaluation: Vec<E>
  - DEEP openings include composition_poly path through 1 Merkle
```

### 3.2 New architecture (quotient chunks)

```
Round 2 (round_2_compute_composition_polynomial — rewritten):
  1. Choose quotient_domain_size = next_pow2(d_max) * N
     - d_max=1: N
     - d_max=2: 2N
     - d_max=3: 4N
  2. constraint_evals: Vec<E>          # size quotient_domain_size
  3. Split into num_chunks = next_pow2(d_max) chunks of size N each
     - Split is by indices (coset decomposition), not by FFT
  4. For each chunk (in parallel):
       chunk_lde: Vec<E> = LDE(chunk, blowup=2)  # size 2N per chunk
       chunk_merkle = MerkleTree over chunk_lde
  5. chunk_roots: Vec<Commitment>      # one per chunk

Round 3 (OOD): for each chunk, evaluate at z (no z^k power needed)
Round 4 (DEEP, FRI):
  - DEEP composition uses N chunks
  - FRI accepts Vec of (LDE, commitment, opening_points) instead of 1

Verifier:
  - quotient_chunk_roots: Vec<Commitment>
  - quotient_chunk_ood_evaluations: Vec<E>
  - DEEP openings include num_chunks Merkle paths
```

**Key win**: no full-poly FFT. The expensive `iFFT(LDE) + break_in_parts +
LDE-per-part` step disappears. Each chunk has degree < N (by design), so
LDE-per-chunk is a single small FFT — cheaper than the full FFT extension.

## 4. Changes per file

### 4.1 `crypto/stark/src/domain.rs`

- Add `pub struct QuotientDomain<F>` with size `next_pow2(d_max) * N`,
  offset, generator. Or extend `Domain` with `quotient_size` field.
- Add helper `quotient_domain.split_into_chunks(num_chunks) -> Vec<...>`.
- Add helper `chunk_domain(chunk_idx) -> sub_coset_offset`.

### 4.2 `crypto/stark/src/constraints/evaluator.rs`

- `ConstraintEvaluator::evaluate` — change loop bounds: iterate
  `0..quotient_domain_size` instead of `0..lde_size`.
- The stride logic from Attempt 1 (revert it) — not needed; instead the
  domain itself is sized to match.
- Stride access still needed for boundary_zerofiers etc. (they're computed
  over LDE 2N, but we iterate quotient_domain — index map: for chunk_idx,
  trace_idx → lde_idx via the disjoint coset map).

### 4.3 `crypto/stark/src/prover.rs`

- **`round_2_compute_composition_polynomial`**: rewrite the second half
  (the part after `evaluator.evaluate(...)`). Replace
  `if number_of_parts == 1 { ... } else if == 2 { ... } else { ... }`
  branching with:
  1. Split constraint_evals into chunks (no FFT).
  2. For each chunk: LDE via `evaluate_polynomial_on_lde_domain`.
  3. Each chunk → its own Merkle tree.
- Return type changes: `Round2 { quotient_chunk_evaluations: Vec<Vec<E>>,
  quotient_chunk_merkle_trees: Vec<MerkleTree>, quotient_chunk_roots:
  Vec<Commitment> }`.
- **`round_3_evaluate_polynomials_in_out_of_domain_element`**: iterate
  chunks, evaluate each at z (no z^k power). Returns
  `Vec<FieldElement<E>>` (one per chunk).
- **`round_4_deep_compose`**: include each chunk as a separate polynomial
  in the DEEP composition. The DEEP gamma randomness now consumes
  num_chunks slots.
- **`open_deep_composition_poly`**: opens each chunk's Merkle tree at the
  query indices.

### 4.4 `crypto/stark/src/verifier.rs`

- Parse `quotient_chunk_roots: Vec<Commitment>` instead of single root.
- For each chunk: append its root to the transcript before sampling z.
- Verify OOD evaluations chunk-by-chunk.
- DEEP composition verification: parallel paths through num_chunks Merkles.
- FRI verification: pass num_chunks commitments to the FRI verifier.

### 4.5 `crypto/stark/src/proof/`

- Field rename / split:
  - Out: `composition_poly_root: Commitment`
  - In: `quotient_chunk_roots: Vec<Commitment>`
  - Out: `composition_poly_parts_ood_evaluation: Vec<E>`
  - In: `quotient_chunk_ood_evaluations: Vec<E>`
- Update `DeepPolyOpenings` to carry per-chunk authentication paths.
- Bump proof version / format ID (we lose backward compatibility).

### 4.6 `crypto/stark/src/fri/`

- FRI prover signature changes: takes
  `Vec<(LDE, MerkleTree, opening_points)>` instead of single LDE.
- Folding logic: similar to P3's `prove_fri` — fold across the union of
  all chunk LDEs at each query.
- FRI verifier: idem, parse multiple commitments.

### 4.7 Tests

- `crypto/stark/src/tests/`: ~124 tests. Most are correctness tests over
  small AIRs. They should pass with no AIR change — only the prover output
  shape changes.
- Update test helpers that construct or inspect `Proof` (e.g., serialization
  roundtrip tests).
- Add new tests:
  - End-to-end roundtrip with `d_max=1` AIR (fib_pair) — 1 chunk path
  - End-to-end with `d_max=3` (read_only_memory_logup) — multi-chunk path
  - Soundness regression: ensure modified verifier rejects proofs with
    tampered chunk roots, tampered OOD evals, etc.

### 4.8 `bench_vs_plonky3/src/`

- No changes required. The Lambda AIR is unchanged; only the protocol
  underneath changes.

### 4.9 Production AIRs (Keccak, CPU chip, etc.)

- **No changes**. AIR definitions stay the same. Only `d_max` and
  `composition_poly_degree_bound` are consulted — those already exist.

## 5. Soundness argument

The Lambda protocol today proves: given commitment `R = MT(H_lde)` of a
polynomial H of degree < d_max·N evaluated on LDE of size 2N, and a FRI
proof of low-degree-ness, the verifier accepts iff H(x) = 0 for all x in
the trace domain (modulo zerofier).

**Claim**: The chunks protocol proves the same property with the same
soundness (and uses the same primitives — Merkle + FRI — same Fiat-Shamir
transcript shape).

**Argument**:
1. **Functional equivalence**: Splitting `H(x)` into `next_pow2(d_max)`
   chunks `Q_0, Q_1, ...` such that `H(x) = Q_0(x^k) + x·Q_1(x^k) + ...`
   (where k = num_chunks) is a polynomial identity. The verifier can
   reconstruct `H(z)` from `(z, Q_i(z^k) for all i)`. Identical
   information content.
2. **Merkle commitment soundness** is per-commitment. Multiple Merkle
   commitments → soundness is the AND over all of them (collision-
   resistance bound). No degradation as long as transcript samples are
   refreshed between commitments.
3. **FRI soundness** over multiple committed polynomials is well-studied
   (it's literally how PCS-style FRI works). Each chunk is a degree-<N
   polynomial; FRI proves all of them low-degree simultaneously by folding
   across the union.
4. **Fiat-Shamir transcript**: must append every chunk root in
   deterministic order before sampling z, then sample DEEP gammas
   per-chunk in deterministic order, then sample FRI alphas. The order is
   protocol-specified.

**This matches P3's `prove_fri` in `/uni-stark/src/prover.rs` lines 43-136**
and the soundness analysis from the Stark by Hand / RISC Zero / SP1 design
docs. We are not inventing new cryptography.

**Verification under scope expansion** (per
`project_optimizer_protocol_scope` memory):
- Build prover + verifier on the experiment branch
- Verify proofs end-to-end on that branch
- Sanity check: pre-change proofs still verify with pre-change verifier
  (keeps the existing protocol intact during exploration)
- Document soundness argument in each commit message

## 6. Implementation plan (incremental phases)

Goal: keep something verifiable at every phase to avoid a multi-week branch
that can't be tested.

### Phase 1 — Add chunks code path in parallel (no protocol change yet)

- Add `round_2_compute_composition_polynomial_chunks` as a NEW function
  alongside the existing one (don't replace).
- Add `QuotientDomain` to `domain.rs`.
- The new function returns `Round2Chunks { ... }` with the chunk shape.
- Unit test: compare H(z) recovered from chunks at random z vs H(z)
  computed by the original `round_2_compute_composition_polynomial`.
  They must match for the same constraint values.
- **Output**: a flag/feature that toggles new vs old prover path. Old
  protocol is the default; new path is opt-in via env var or feature flag.

### Phase 2 — Verifier-side chunks support

- Add `verify_chunks` path in `verifier.rs` that consumes `Vec<Commitment>`.
- New `ProofChunks` struct in `proof/`.
- Unit test: round-trip serialization, deserialization.
- **Output**: a CLI flag or feature to verify chunk-shape proofs.

### Phase 3 — FRI over multiple commitments

- Extend FRI prover to accept `Vec<LDE>` (currently takes single LDE).
- Extend FRI verifier idem.
- Unit test: FRI commit phase on 1 LDE → same result as FRI on 1-element
  Vec<LDE> (regression check).
- Unit test: FRI commit phase on 4 LDEs (simulating d_max=4) produces a
  proof that verifies.
- **Output**: FRI primitives ready for the chunks protocol.

### Phase 4 — Wire end-to-end (flag-gated)

- Add `--protocol=chunks` flag (or env var) that selects which path to
  run end-to-end.
- Default remains `--protocol=single-h` (existing behavior).
- Tests: run all 124 stark tests under both flags. Expectation: both pass.
- Bench: run `bench_vs_plonky3` under both flags. Expectation: chunks
  matches or beats single-h.

### Phase 5 — Benchmark + tune

- Profile chunks path on `vm-benchmarks-1` at log=21 × {32, 64, 128} cols
  + breakdown.
- Compare against baseline `bench_vs_p3_20260513_2033_upstream/`.
- Iterate if any subphase regresses.
- **Stop criterion**: success criteria from §2 are met, OR the gap to P3
  is within 1.2× (vs 1.95-2.93× today).

### Phase 6 — Decide default + cleanup

- If chunks consistently wins and tests are clean, flip default to chunks.
- Deprecate the single-H path (or keep both behind a feature flag for
  compat).
- Update all docs (METHODOLOGY.md, lambda_vs_p3_port.md, etc.) with the
  new architecture.
- This phase is OUT of the optimizer workstream — it's a project decision.

## 7. Risks

| Risk | Mitigation |
|---|---|
| Soundness bug (catastrophic) | Phase-gated rollout, soundness argument per commit, independent audit before flipping default |
| Performance regression on some AIR (e.g., logup tables with d_max=3 might pay more in commit phase) | Phase 5 benchmarks ALL representative AIRs, not just fib_pair |
| Increased verifier complexity → harder to audit | Document protocol shape change clearly, write expanded test suite for verifier |
| Proof size grows (we have N Merkle roots vs 1) | Measure; if too large, batch Merkle trees Plonky3-MMCS style (out of scope phase 1-6) |
| Migration churn — breaks pinned dependencies | Bump proof version / format ID, keep old verifier available for legacy proofs |
| FRI changes affect security parameter (queries / Johnson Bound) | None — same FRI primitives, same number of queries; only the input list size changes |

## 8. Estimation

| Phase | Effort | Notes |
|---|---|---|
| 1. Parallel chunks code path | 3-5 days | Domain + new round_2 fn + unit tests |
| 2. Verifier chunks support | 3-5 days | Proof struct + verifier path + serialization |
| 3. FRI multi-commit | 5-7 days | The trickiest part; folding logic + tests |
| 4. End-to-end flag | 2-3 days | Wiring + test sweep |
| 5. Benchmark + tune | 3-5 days | Server bench runs + analysis |
| **Total** | **3-4 weeks** | Plus 1 week buffer for soundness review, edge cases |

If the gain is confirmed in Phase 5, an additional 1-2 weeks for Phase 6
(deprecate single-H, cleanup, docs).

## 9. Open questions

1. **Should we keep both paths long-term**? Or fully migrate? Argument for
   keeping: backward compat with already-issued proofs. Argument against:
   maintenance burden.
2. **Single-Merkle multi-column commit (MMCS) as further optimization**?
   P3 uses MMCS to commit multiple columns in one tree (saves proof size).
   We could do that in a follow-up.
3. **ZK extension**: P3 supports ZK via additional chunks (`is_zk()` flag).
   Lambda doesn't have ZK today; we could add the placeholder bits but
   actual ZK is a separate project.
4. **Keccak chip behavior** (d_max=3): we expect 4 chunks. Need to
   confirm Phase 5 that the per-chunk overhead doesn't exceed the saving.
   If it does, we may need to dial down to 2 chunks (forcing degree=2 by
   constraint normalization) — but that's an AIR change.

## 10. Out of scope (for now)

- SIMD / PackedField in chunks code path (`feedback_no_simd_for_now` memory)
- ZK rounds (per Q3)
- MMCS multi-column-per-tree optimization (per Q2)
- Changes to existing AIRs (Keccak, CPU, BITWISE, ROM, etc.) — they stay
  as-is, only the prover/verifier underneath migrates

## 11. Decision point for the user

Before starting Phase 1, the user should explicitly confirm:
- Time commitment (3-4 weeks) is acceptable
- Risk of catastrophic soundness bug is understood
- Audit / review process is in place
- Production deployment (if any) accounts for proof format change

If any of those is NO, the recommended fallback is **Pattern 2
(batched column-major LDE)** — a smaller refactor that doesn't touch
protocol, costs ~1-2 weeks, attacks `r1_main_lde+merkle` (28% of prove
total) and is expected to give 30-50% in that phase.

---

**Status today**: design doc written, not approved, no code written.
Next action: user decision on whether to proceed with Phase 1 or pivot
to Pattern 2.
