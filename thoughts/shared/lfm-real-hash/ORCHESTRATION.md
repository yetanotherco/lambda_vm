# BLAKE3 real-hash implementation — orchestration tracker

Goal: make the LFM machine's role-2 hash (LFM_HASH) cryptographically real with BLAKE3,
round-parameterized (7-round baseline / 6-round perf), bound into the digest/registry.
Full plan: PLAN.md. Decisions locked: Route A (behind LFM_HASH, no digest move);
Phase-4 mapping = **Option A** (truncate 256→128 into the 1-cell digest + domain separation).

Worktree: /Users/maurofab/workspace/lambda_vm-blake3-impl (branch `blake3-real-hash` off pr915).
Agents work there; heavy builds SERIALIZE (one cargo build at a time). Agents checkpoint to
this dir; lead reviews verdicts, commits per phase, does not read full file dumps.

## Dependency order & waves
- WAVE 1 (parallel — only one builder):
  - [x] P1  Phase 1 DONE (green): reference now TWO-source (python oracle + upstream BLAKE3 portable C,
        round-parameterized) — all 10 6r vectors reproduced byte-for-byte, 7r matches official vectors,
        neg controls pass. Socket KATs (Option A) generated. Rust cross-check deferred to a build phase.
  - [x] P3  Phase 3 DONE + COMMITTED (2d236786): hasher bound into program_id/registry/lfm_verify.
        SOUNDNESS PROPERTY HOLDS + tested (different hasher => distinct program_id). Registry regenerated
        (6 program_ids moved, no root moved). lint/fmt green. 19 lfm:: failures = pre-existing stub-ELF.
  - [ ] DOC Phase 6 A6R sign-off doc (folded into P1). NO build.
  - [ ] ORACLE (spawned ahead, no build): build the human-owned z3 gate ORACLE for BLAKE3 BEFORE the chip —
        reference f (6/7r), the Option-A socket reference + KATs, the COLUMN-ROLE MAP (which is ALSO the
        Phase-2 chip spec), the chip-contract library, and the z3 gate framework with mandatory negative
        controls + width audit. Output: thoughts/shared/lfm-real-hash/gate-oracle/. FEEDS: Phase 2 (chip
        conforms to the column-role map) + the z3 gate (plug the real chip constraints into the seam).
- WAVE 2 (after P1 + P3, verified):
  - [ ] P2  Phase 2: HasherKind::Blake3, round-parameterized (BLAKE3_ROUNDS knob), Route A behind LFM_HASH,
            wire executor/trace/AIR, hasher-dependent bus_interactions, measure at 6/7. BUILD. THE BULK.
- WAVE 3 (after P2):
  - [ ] P5  Phase 5: prove+verify a wrap under BLAKE3 (swap TestPermutation). BUILD.
- VERIFY GATES: adversarial check after P3 (binding sound?), after P2 (proves+verifies+KAT match?), after P5.
- Z3 FORMAL GATE (after P2, since it needs the round-parameterized chip + socket): extend the existing
  blake3 z3/QF-BV gate (thoughts/blake3/blake3-chip/z3_blake_verify.py, restored by P1) to cover the chip
  at BOTH 6 and 7 rounds + the Option-A socket (2-to-1 compress + truncate 256->128 + domain tag + byte
  enforcement). Mandatory rigor: negative controls (drop an AreBytes/BITWISE contract -> SAT; wrong
  truncation window -> SAT; missing domain tag -> SAT) + a non-vacuity positive control + the width audit
  (every field-lifted width cites a real BITWISE/AreBytes contract). z3/QF-BV fits because BLAKE3 is a
  byte computation; native-field (Poseidon) would need cvc5-FF/Lean and is out of scope. NOT a z3 target:
  the Phase-3 hasher binding (domain-separation argument -> pinned by the "distinct program_id" test).

## Status log
- 2026-08-10: Phase-1 anchor pre-confirmed GREEN by lead (oracle vs official vectors + Plonky3, 6r derivative). PLAN.md written.
- WAVE 1 spawned.
- 2026-08-10 (late): ORACLE complete (ORACLE.md + gate board PASS; chip_model.py = the Phase-2 spec).
  A6R decision recorded by user: 7-round instantiated baseline, 6 behind the blake3-6round feature.
  O5 RATIFIED by user: future leaf hashing uses the reserved "LFML" tag.
- 2026-08-11: **P2 DONE + COMMITTED `b693eece`** (+ O5 docs `cece4a0b`): Blake3 arm behind LFM_HASH,
  compress-only (MODE_P=0 pinned; permute = task #8 and it BLOCKS P5's full wrap), 7r default,
  KATs 15/15 both counts + direct crate anchor, 4,741/5,509 cell-equiv (6r/7r). Adversarial review
  closed both directions: 0 soundness, 0 regression (phase2-verify.md); executor compress_out fix
  is the one cross-hasher touch (latent inlining bug, defaults preserve Test/Poseidon).
  **Z3 CHIP GATE DONE: CHIP-GATE.md VERDICT PASS 75/75 on the chip AS BUILT** (seam transcription,
  argued algebra ledger AR1-AR4, census == built chip to the unit at both round counts);
  artifact_pin v2 --check confirms the verdict applies to the committed file (semantic regions +
  resolved framing unchanged; drift is comments/docs only). Remaining oracle-side: D7b (ORACLE.md
  §3.2 census refresh), D10 (chip_model.py:152-156 docstring), D6 durable note, commit-SHA pin update.
- 2026-08-11: oracle closed all four doc items + re-ran the board (PASS, 23:06). Reviewer's exit audit
  flagged O5's safety claim as unaudited; lead verified in code: "no leaf-hashing path" was FALSE
  (FriToyV0 compresses raw rows into leaves under the LFMC tag, programs.rs:577/585/625); safety rests
  on fixed-depth static circuits ALONE. Corrected in commit 2957c3f9 + ORACLE.md §7 + CHIP-GATE.md.
  Phase-2 agents shut down after independently re-verifying the commit hashes. Branch head: 2957c3f9.
  NEXT: task #8 permute-socket spec (Phase-5 blocker; mapping decision goes to the user first).
- 2026-08-11: permute-socket options paper delivered (A: LFMP socket / B: compress-based sponge /
  C: mixed Poseidon). Corrected map inside: WRAP IS HASH-NEUTRAL (epoch emits no Instr::Hash) — only
  TrivialV0/FriToyV0 gate on this; TrivialV0 calls b.permute directly. **USER RATIFIED OPTION B (B1):
  compress-based FS chain for all hashers, no permute socket ever, MODE_P=0 permanent, no assumption
  beyond A6R.** Oracle assigned the transcript spec (TAG_LFMT, reference+KATs incl. end-to-end
  FriToyV0-preamble vector, gate two-tag framing, TrivialV0-fate rec) → transcript-spec/. Build agent
  spawns on the spec, not before.
- 2026-08-11: oracle re-gate on post-B1 chip: CHIP-GATE PASS 79/79, re-pinned (pin extended: TAG_LFMT
  resolved + tags-distinct check), census unit-exact incl. program totals; M8 standing control added
  (idx-4-alone forgery SAT / one-hot UNSAT / both real tags reachable); TRANSCRIPT.md §3.3 + both §2.2
  framing rows corrected BEFORE transcription; two own-instrument bugs caught (§4.6.3).
  Leaf-convention options note delivered (leaf-convention-options.md): NEW option C found — in-socket
  felt mode reusing O1's lane machinery + Z/GINV canonicity (p−1 = 0xFFFFFFFF_00000000 ⇒ "hi maximal ⇒
  lo zero"; 2 cols + 4 constraints/felt, no new sends). FriToyV0 @7r: C = 502,047 (+36%) vs A
  (felt_be_halves) = 585,039 (+58.5%, a floor) vs B = off-BLAKE3 forever.
  **USER RATIFIED OPTION C + LFML** (leaves hash under the ratified LFML tag via MODE_L): cheapest by
  14%, reuses the machine's own canonicity idiom, retires O5's fixed-depth crutch; cost = one more
  re-gate (new pin + canonicity width-audit pair; M8 extends unchanged). §6 open point resolved by
  decision: MODE_L implies felt-input (contiguous one-hot span). SEQUENCE: commit B1 first (b1-verify
  pending), then MODE_L spec-first (oracle), then build.
