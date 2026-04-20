# MMCS Batched Commitment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Replace per-table independent Merkle trees with shared batched trees (one per commitment phase), enabling shared FRI and eval-form quotient commitment.

**Architecture:** All tables' columns for a given phase (main, aux, composition) are concatenated into a single wide Merkle tree. With uniform table sizing (all tables capped at 2^20 LDE rows), this is pure column concatenation — no jagged-height insertion needed. Each query opens one row from the shared tree, which contains columns from ALL tables. A BatchedLayout maps column ranges to tables. The composition polynomial parts from decompose_d2 are committed at N points (half-size) in their own shared tree, enabling eval-form quotient savings.

**Tech Stack:** Rust, Goldilocks field, Keccak256 Merkle trees, existing BatchedMerkleTree and commit_columns_bit_reversed infrastructure.

**Branch:** feat/eval-form-quotient (continues from the decompose_d2 refactoring).

---

## Task 1: BatchedLayout — Column offset tracking

**Files:**
- Create: crypto/stark/src/batched_layout.rs
- Modify: crypto/stark/src/lib.rs (add module)

Tracks where each table's columns live in the concatenated Merkle tree row. new(column_counts, lde_size) builds ranges. extract_table(idx, row) slices a table's columns from an opened row.

- [ ] Step 1: Define BatchedLayout struct with new() and extract_table()
- [ ] Step 2: Add unit tests
- [ ] Step 3: Register module in lib.rs
- [ ] Step 4: Run tests, commit

---

## Task 2: BatchedProof — New proof structure with shared roots

**Files:**
- Modify: crypto/stark/src/proof/stark.rs

New BatchedProof struct: shared main/aux/composition Merkle roots, shared FRI data, per-table OOD data in Vec<TableProofData>, per-query BatchedQueryOpening with openings from all three shared trees. Old MultiProof kept for backward compatibility.

- [ ] Step 1: Define BatchedProof, TableProofData, BatchedQueryOpening structs
- [ ] Step 2: Verify existing tests still pass (no regressions)
- [ ] Step 3: Commit

---

## Task 3: Batched main trace commitment

**Files:**
- Modify: crypto/stark/src/prover.rs — add commit_main_traces_batched
- Create: crypto/stark/src/tests/batched_tests.rs

Build a single Merkle tree from all tables' main trace LDE columns concatenated. Flatten per-table columns into one Vec, call commit_columns_bit_reversed. Return tree + root + BatchedLayout.

- [ ] Step 1: Write commit_main_traces_batched function
- [ ] Step 2: Write test for batched commit/open roundtrip
- [ ] Step 3: Run tests, commit

---

## Task 4: Batched composition commitment with eval-form quotient

**Files:**
- Modify: crypto/stark/src/prover.rs

All tables' composition parts (H0, H1 from decompose_d2) committed at N squared-coset points in a shared tree. No iFFT+FFT extension needed — the N-point evaluations go directly into the batched tree. This is where the FFT savings land.

- [ ] Step 1: Write commit_composition_polys_batched (N-point tree)
- [ ] Step 2: Update round_2 to return N-point evaluations for batched path
- [ ] Step 3: Test OOD evaluation from N-point squared-coset matches extended path
- [ ] Step 4: Commit

---

## Task 5: Shared FRI with alpha-batching

**Files:**
- Modify: crypto/stark/src/prover.rs
- Modify: crypto/stark/src/fri/mod.rs

All tables' trace-only DEEP polynomials are alpha-batched into one 2N-point vector. Composition DEEP values (N-point, from the batched composition tree) are alpha-batched and injected after the first FRI fold via commit_phase_from_evaluations_with_injection. One FRI instance for all tables.

- [ ] Step 1: Implement alpha_batch_deep_polynomials
- [ ] Step 2: Implement composition DEEP injection for shared FRI
- [ ] Step 3: Update multi_prove to use shared Round 4
- [ ] Step 4: Write multi-table batched prove roundtrip test
- [ ] Step 5: Commit

---

## Task 6: Batched verifier

**Files:**
- Modify: crypto/stark/src/verifier.rs

Verifier replays shared transcript, opens shared trees at query indices, extracts per-table values via BatchedLayout, reconstructs alpha-batched DEEP + composition injection, verifies shared FRI.

- [ ] Step 1: Implement multi_verify_batched
- [ ] Step 2: Run full test suite (old + new paths)
- [ ] Step 3: Commit

---

## Task 7: Integration — Wire into multi_prove/multi_verify

**Files:**
- Modify: crypto/stark/src/prover.rs
- Modify: crypto/stark/src/verifier.rs
- Modify: prover/src/lib.rs (VM prover entry point)

New multi_prove_batched and multi_verify_batched entry points alongside existing functions. Run VM prover benchmarks.

- [ ] Step 1: Add multi_prove_batched entry point
- [ ] Step 2: Add multi_verify_batched entry point
- [ ] Step 3: Run VM prover benchmarks, compare against baseline
- [ ] Step 4: Commit + push

---

## Task 8: Cleanup — Drop extension FFTs in batched mode

**Files:**
- Modify: crypto/stark/src/prover.rs

In the batched path, decompose_d2 output is committed directly at N points. No iFFT+FFT extension in Round 2. The DEEP uses coset-shifted values (iFFT(N,g2) + FFT(N,g)) or verifier-side barycentric interpolation from query openings.

- [ ] Step 1: Drop extension in batched Round 2
- [ ] Step 2: Verify all tests pass
- [ ] Step 3: Commit

---

## Expected Savings

| Metric | Current | After MMCS + Shared FRI |
|--------|---------|------------------------|
| Merkle trees per proof | ~32 | 3 |
| FRI instances | ~12 | 1 |
| FRI layer trees | ~228 | ~19 |
| Composition FFTs (Round 2) | 2 iFFT(N) + 2 FFT(2N) per table | 0 (N-point commit) |
| Proof size | ~2.5 MB | ~0.8 MB (est.) |
