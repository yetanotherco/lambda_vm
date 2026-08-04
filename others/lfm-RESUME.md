# RESUME HERE — Phase R, keccak recursion

Written 2026-07-30 as a compaction-survival doc. If you are picking this up
with no memory of the session, read this file first, then
`lfm-standing-decisions.md`, then `lfm-target-shape.md`. Everything else is
reference.

## The goal, in one paragraph

Make the LFM (Lambda Field Machine — a straight-line, field-native recursion
machine living in `prover/src/lfm/`) verify a real Lambda VM **continuation
epoch proof**, using **keccak** as the hash. Keccak is explicitly the hash we
do NOT expect to ship; it is first because it needs zero changes to the inner
prover (post-#841 the verify path is keccak-only). Once the e2e works, the
same e2e becomes a **hash test matrix**: blake (most probable final choice)
and Poseidon behind the same socket, giving measured cells-per-verify per
candidate as input to the ecosystem hash decision.

## Where the code is

- Branch **`feat/lfm`**, worktree `/Users/maurofab/workspace/lambda_vm_3-lfm`.
  Never pushed. Based on `origin/main` (includes #841).
- Side branches, both merged into `feat/lfm` as of this writing:
  `feat/lfm-chunking`, `feat/phase0-constraint-ir`, `feat/lfm-constraint-emitter`.
- ⚠ **One worktree per agent, always.** Merging into a worktree an agent is
  live in produced a merge commit wearing that agent's message (`af5ea7c4`).
  Both agents were down when the latest consolidation happened; that is the
  only safe time.

## What works today (all committed, all green)

The machine proves and verifies, end to end, through the registry:

1. **Keccak family hosted unchanged** — production `KECCAK_RND`/`KECCAK_RC`/
   `BITWISE` AIRs driven by an LFM adapter chip speaking their bus contract.
   `KECCAK_RND` is chunkable (chunk count is program shape, in the digest).
2. **`keccak256` bit-exact** vs `PlatformKeccak256` at eight boundary lengths.
3. **Transcript replay** bit-exact vs the real post-#841 `DefaultTranscript`
   (squeeze buffer, absorb invalidation, canonicity guard, zero-rejection).
4. **Continuation-epoch statement + Phase A** → publishes the real `(z, α)`,
   verified against production's own `absorb_statement_with_digest`.
5. **A real Merkle opening authenticated under production keccak** — real
   proof, real committed root, real path.
6. **Constraint evaluation** — a serialized `ConstraintArtifact` lowered to
   machine instructions, all 28 AIRs vs the production evaluator, plus a
   real-proof composition check. 57,252 instructions/epoch, 9.7% under budget.
7. **DEEP slice 1** — composition-polynomial reconstruction at a query point,
   vs production's own reconstruction.
8. **Chaining (ii) and (iii)** — cross-epoch L2G root binding, and the
   attestation's `program_id` fold bit-exact vs production.

## What is left, in order (updated 2026-07-31)

1. ~~**Chaining (i)**: REGISTER preprocessed commitment from `reg_fini`~~ —
   **DONE** (reg-tree, merged at 69b4a915). Prediction confirmed exactly:
   255/511/1023 permutations, 0.0182%/0.2224% noise. No second hashing
   gadget — `keccak_hash_pair` unwelded from the walk sufficed. ⚠ Left an
   OPEN assembly obligation: the `reg_fini` felt-width gap — see
   `lfm-assembly-obligations.md`, which is now the ledger every leg's
   deferral goes into.
2. ~~**DEEP across a full sub-proof**, wired to R1f's Merkle
   authentication~~ — **DONE** (deep-join, merged at 703f742b). Join is
   structural: same arena cells, index bits bound (a hinted point is the
   same gap one level over), control programs run the denied attacks.
   Cost inversion found: authentication is 99.0% of the leg, DEEP 1.0%;
   213,744 permutations/epoch at blowup 8 for openings (~46% of the
   predicted epoch keccak bill). Shared-commitment lever measured at 48%
   collapse (111,471) — parked, see
   `lfm-team-lead-shared-commitment-ruling.md`.
3. ~~**FRI folding leg**~~ — **DONE, leg CLOSED** (fri-emitter, merged at
   5a246ba5; spec now carries Addendum 2). Emitter (per-layer walk + fold
   chain + terminal check) differentialled entirely against REAL
   production proofs that fold: the leg's blindness premise was FALSE —
   the L2G fixture's trace sizes with boundary count, so 512/1024/2048
   boundaries give real proofs with 1/2/3 committed layers in under a
   second. Measured = predicted on all six pinned numbers (174/186/198
   perms/query; 38,106/20,460/14,454 per sub-proof at blowup 2/4/8).
   Proves+verifies end to end. Two approved deviations: the terminal
   check EVALUATES at υ^(2^total_folds) instead of emitting the FFT (a
   codeword lookup is a 1,023-wide Select tree on this machine — the
   guest's economics do not transfer; equivalence checked at 876
   index/shape points, and the zero-fold branch unifies), and
   `fri_fold`'s mul association differs from production's while the
   field element does not. The OWED leaf-gadget byte check discharged
   executably vs BOTH production backends. Left ledger entries 4 (FRI
   challenges from the transcript in production's interleaved order;
   coefficients+roots are proof DATA the transcript must absorb —
   nothing leg-side can catch this) and 5 (informational:
   isolation-driver index width).
4. ~~**LogUp closure**~~ — **DONE, leg CLOSED** (deep-join, 7 slices,
   merged at 1145041a; handoff `lfm-logup-handoff.md`). Closure built
   against production's own oracles; found and closed THREE soundness
   gaps (L two-consumer split, hinted alpha powers — instance 3, worse in
   degree — and the earlier DEEP/auth parallel-copy class), witnessed
   per-chunk accumulation with a ≥2-chunk fixture, resolved
   has_trace_interaction by reading. The one unknown it left is now
   **SETTLED** (zerorow, 2026-08-03): a zero-row fixed table reports
   `Some(zero)`, measured on a real accepted epoch — five of them
   (KECCAK, KECCAK_RND, KECCAK_RC, ECSM, ECDAS) — and stripping the field
   makes the proof fail, so `Some` is forced. Same test closes the
   table-set-LENGTH gap (closure run over a real epoch's 24
   contributions) and found that three of the five have NON-blank traces
   with every multiplicity column zero: "unused" ≠ "blank".
5. **Assembly** into one epoch-verifier program — **SPINE + LEGS DONE; the
   whole verifier runs on a real epoch** (assembly waves 4 and 5, branch
   `feat/lfm-assembly`).
   - **DONE**: the Fiat-Shamir spine RUNS on a real 24-sub-proof continuation
     epoch that production accepts. `prover/src/lfm/epoch.rs` replays the
     fork, Phase C, and rounds 2-4 in production's order;
     `epoch_tests::the_epoch_challenge_spine_matches_production` matches
     production's own `replay_rounds_after_round_1` on all 111 challenges
     (shared z/α, then per table β, z, γ, every ζ, every query index), and the
     LogUp closure on top reaches production's COMMIT-bus target. Ledger
     entries 4, 5 and 6 DISCHARGED; 2 half (the cell is right, the derivation
     is not built); 3 partially (one cell + two views is now a construction,
     but the second consumers are not wired).
   - **LEGS DONE** (assembly-w5, `feat/lfm-assembly` @ a1f32859): the whole
     verifier RUNS. `prover/src/lfm/epoch_verify.rs` is the seam — per sub-proof
     it rebuilds the OOD grid from the two pruned blocks the transcript
     absorbed, runs the constraint evaluation and quotient check at the spine's
     own `z`/`beta`, and takes each query's `iota_bits` straight into the Merkle
     walk, the DEEP fold and the FRI chain. `epoch_challenge_program` became
     `epoch_program(e, with_legs)` so ONE spine emitter serves both programs and
     the leg program cannot drift from the one the 111-challenge differential
     covers. Ledger entry 3 DISCHARGED (the absolute hinted-once count now runs
     over a program that HAS both consumers of every value, with a positive
     control that it declares strictly more arena words than the spine); 21
     tamper vectors rejected.
     * MEASURED, min preset (blowup 2, 1 query/table, grinding 1, 24
       sub-proofs): **spine 1,095,553 instructions / 1,211 permutations / 5,716
       arena words -> ASSEMBLED 2,184,360 / 2,616 / 16,478.** Legs alone
       1,088,807 / 1,405 / 10,762, so the verifier is ~50/50 Fiat-Shamir and
       verification at this preset.
     * Leg permutations match a closed form over the shapes EXACTLY (927 leaves
       + 304 Merkle levels + 174 FRI = 1,405). Constraint lowering reproduces
       the design's 54,358 ALU rows to the digit (63,393 unfused likewise);
       recombination 2,431 against 2,894, the gap being zerofier squarings at
       this epoch's real trace lengths. FRI at blowup 8 lands on the pinned
       14,454 exactly.
     * WARNING: the composed OPENING predictions assumed a UNIFORM 2^20 per
       sub-proof. A real intermediate epoch is `[2 x14, 3, 4 x4, 5 x3, 7, 20]` —
       one big table and 23 tiny ones — so its openings are 1.88x cheaper than
       the uniform model (100,959 against 189,727 at blowup 8). 213,744 stands
       as a model of a PRODUCTION-sized epoch, not of this one; ledger entry 10.
   - **NEW ledger entries** 7 (the preprocessed commitments are hinted — and
     wave 5 CORRECTED its taxonomy: DECODE is ELF-dependent like PAGE, so the
     split is 2 constants + 2 ELF-dependent + 1 derived, with a proposed
     resolution awaiting a ruling), 8 (the OOD absorb ORDER has no production
     witness: every OOD block of all 24 sub-proofs is one row tall), 9 (the
     constraint leg's frame-STEP view of the grid is invisible at
     `step_size = 1` — same witness as 8) and 10 (per-epoch numbers must name
     their epoch shape).
   - **STILL NOT DONE**: entry 7's wiring (intern BITWISE + KECCAK_RC, call
     `programs::emit_register_commitment` from Phase A, rule on DECODE/PAGE) and
     therefore entry 2's derivation; entry 8's synthetic AIR.
6. ~~**The wrap run**~~ — **DONE, and the machine PROVES its own epoch verifier**
   (assembly-w7, `feat/lfm-assembly`). Run LOCALLY, not on the box: the box was
   occupied by an ethrex continuation campaign at both check points (18:43 and
   18:52 UTC+2, load 22-29 on 32 cores, two different `cli prove` invocations), so
   per the brief nothing was started on it. Every number names the epoch profile
   `[2 x14, 3, 4 x4, 5 x3, 7, 20]`; the full table is in
   `lfm-assembly-obligations.md` entry 10, now SATISFIED.
   - **Slice 0** (min preset): the assembled verifier proves in 19.5 s and
     verifies in 0.09 s, 30,707,816-byte proof, 14 sub-proofs, 16.2 GiB peak.
     220,107,920 main + 87,073,068 aux ext cells.
   - **Slice 1a** (inner blowup 8, 1 query — the GEOMETRY, 2^23 LDE, 22 Merkle
     levels, 12 committed FRI layers): proves in 23.3 s, verifies in 0.09 s,
     31,147,664-byte proof, 15.5 GiB peak.
   - **Slice 1b** (inner blowup 8, 73 queries — the PRODUCTION SHAPE, emitted and
     censused, not proved): 76,501,118 instructions / 118,080 permutations /
     817,101 arena words / 6 KECCAK_RND chunks and **5,077,422,224 main +
     2,029,461,548 aux ext = 11,165,806,868 base-field-equivalent cells per epoch
     verify**. Openings **100,959** and FRI **14,454** — both pinned predictions
     hit exactly.
   - ★ **84.0% of the cells are the keccak family**, 36,256 main + 13,912 aux per
     permutation. The hash matrix's other columns therefore decide the machine's
     SIZE; its structure is already settled.
   - ⚠ **The production-shaped wrap is not provable at 124 GiB**: 350.6 GiB
     projected peak from a coefficient measured twice (33.7 bytes per
     base-field-equivalent cell; a 15.9 GiB projection came in at 15.5 GiB). The
     three ways out are a cheaper hash, disk spill, or splitting the wrap — a
     decision, not a debt.
   - Falsified both directions, end to end: a tampered inner proof makes the wrap
     **unbuildable** (execution dies at `DivByZero` in the root compare — a false
     assert has no witness), while a moved claimed public word or a moved program
     digest makes an honest proof **unverifiable**.

## Decisions already made — do not relitigate

- **Prove the inner proof at BLOWUP 8.** Three independent legs point
  there: DEEP scales with query count (73 vs 219 ⇒ ~3×), the keccak bill
  does too (~460k vs ~1.4M permutations), and FRI is 2.6× cheaper (14,454
  vs 38,106 permutations — query count falls 3× while per-query cost rises
  only 14%; reg-tree, FRI slice 0, derived from the verified spec).
- **The REGISTER derivation IS the binding.** `VmAirs::new`'s
  `register_preprocessed` parameter looks like unfinished plumbing; it must
  stay unwired. Computing the commitment from `reg_fini` is what ties the
  values to it.
- **Shape-static values are program constants, never arena reads** — and
  next-row pruning likewise, because the verifier reconstructs an undeclared
  column as ZERO.
- **The uniform promotion (`epoch_label`/`page_base`) is PARKED** — off the
  critical path, design complete in `lfm-page-base-uniform-proposal.md`.
- **Zero-rejection transcript** is forced by straight-line shape, not a
  choice; completeness cost < 1e-6/proof.

## Open items needing the USER, not an agent

- **The prover determinism fix.** Root-caused: six dedup tables assign row
  indices by std `HashMap` iteration order (`lt.rs:163/168/177` and five
  siblings), plus grinding's `find_any`. Fix is a contained ~7-file change
  (insertion-ordered index map; `find_first`), no soundness risk, and it
  would restore byte-reproducible proofs and enable a decisive experiment on
  the long-standing ±100k recursion-bench noise. **Offered, not started.**
- **The `check_attestation` gap.** The consumer-side recompute that binds
  supplied roots to a trusted ELF has ZERO production call sites; there is a
  committed PoC (`prover/src/tests/recursion_soundness_gap_poc.rs`). Working
  as designed ("not self-enforcing"), but the design assumes a consumer who
  performs the ritual and nothing in the CLI does.

## How to restart the work

(Updated 2026-08-03 after wave 2.) Every wave so far ended the same way —
worker agents hit session limits, so a restart is always a cold start:
nothing to resume, only to re-spawn against the committed briefs. What
worked, twice now:

- **One agent per leg, one worktree per agent.** Create the worktree off
  `feat/lfm` first (`git worktree add <path> -b <branch> feat/lfm`, then
  symlink `executor/program_artifacts` from the main checkout, or prover
  tests fail on missing fixtures).
- **Brief with pointers, not content**: this file, then
  `lfm-standing-decisions.md` (binding), then the leg's own section above.
  Tell the agent to verify ground truth (`cargo test -p lambda-vm-prover
  --lib lfm`) before writing anything.
- **Have them merge `feat/lfm` INTO their branch** as it moves, never the
  other direction, and consolidate only when no agent is live.
- **Ask for the report format** the phase used: headline, what landed, tests
  verbatim, measurements vs prediction, deviations with reasoning,
  surprises, falsification runs. The measurements-vs-prediction line is what
  caught most of the errors.
- Agents append to `lfm-agent-status.log` at slice boundaries; that log is
  the history if a mailbox message is lost, which happened repeatedly.

Wave 3 CLOSED 2026-08-03 (both legs same day, both agents stood down
cleanly — first wave that did not end at a session limit). 188 green,
lint 0.

Wave 4 (assembly) SPAWNED and ABORTED same day: the agent hit the
session token limit ~25 minutes in (reset 16:40 America/Buenos_Aires),
branch untouched. Its worktree is ALIVE and clean — reuse it, do not
create another:
`/private/tmp/claude-501/-Users-maurofab-workspace-lambda-vm-3/0f390d07-adf0-4a3e-a1b5-d6a58e444fae/scratchpad/wt-assembly`,
branch `feat/lfm-assembly` @ 35845e4c (artifacts symlink in place).
One deliverable survived and is COMMITTED:
`lfm-team-lead-start-index-research.md` answers ledger entry 2 —
production binds `start_index` (x254, reg slot 64) by REBUILDING epoch
N's REGISTER preprocessed commitment from epoch N−1's FINI vector; no
arithmetic start+len check exists anywhere; the LFM analogue is binding
the arena word to `reg_fini[64]`, which the reg leg already handles.
Bonus: FINI's u32 commitment forces `start_index < 2^32`, which bears
on ledger entry 1 (may upgrade the REG-C2 argument route over the
range check).

Wave 4 (assembly) RAN 2026-08-03 on `feat/lfm-assembly` (3 commits off
35845e4c). Suite 195 green (188 + 7), `make lint` exit 0. See item 5 above for what
landed. `lfm-team-lead-start-index-research.md` was originally committed as
a raw 518 KB JSONL session transcript under a `.md` name; the team lead
replaced it (post-wave-4) with the research agent's final report extracted
verbatim from that transcript. The raw session survives in git history at
e105dea2 if ever needed; findings are also summarised in ledger entry 2.

Wave 6 CLOSED 2026-08-04 (`feat/lfm-assembly`, 3 commits off 3766214a; suite
208 green / 1 ignored, `make lint` exit 0). **The assembly ledger is now empty
of debts**: entries 1, 2, 7, 8 and 9 all discharged, leaving only entry 10,
which is the wrap run's own reporting rule and not a debt.

- **Entry 7 + 2**: every preprocessed root now comes from the source its
  provenance admits, chosen by a classifier that recomputes production's
  candidate functions (so an unknown provenance PANICS instead of being hinted
  unbound). Options-only roots intern as program text; REGISTER is derived in
  Phase A from the register boundary, which is what binds `start_index`; DECODE
  stays an arena cell bound by the attestation join, with the `program_id` fold
  emitted on the same cell and differentialled against production. The join is
  denied structurally by a hinted-once guard PLUS an exact arena schema, and
  falsified with a coherent forgery — a split-cell control program runs the
  substitution and attests to another program's id.
- **Entries 8 + 9**: witnessed by TWO fixtures, not one. The brief's single AIR
  is unbuildable — `AirWithBuses` hardcodes two transition offsets, and
  `step_size > 1` is unprovable (a framework ceiling, measured). Entry 8 needed
  no synthetic AIR at all: `FibonacciMultiColumnAIR` already has three offsets,
  giving a 3×2 next-row block. Entry 9 needed no proof: production's own
  `into_frame` is the oracle for the grid→frame-step mapping.
- **Entry 1** discharged by its own stated default (emit the range check), which
  slice 1 triggered by making the boundary vectors live arena data.
- **Cost**: assembled verifier 2,184,360 → 2,244,094 instructions, 2,616 →
  2,872 permutations at the min preset. The +256 is exactly 255 (REGISTER tree at
  blowup 2) + 1 (the `program_id` fold).

⚠ TWO THINGS FOR THE USER, both always-stop items:
1. **The `step_size > 1` framework ceiling.** `RowFrame::from_lde` asserts
   single-row steps (`frame.rs:38`, reached from `evaluator.rs:72`). From reading
   only, the assert looks over-strict for the access pattern that exists — the
   general `Frame::read_from_lde` already handles multi-row steps and constraint
   bodies only ever read row 0 of a step — so it is plausibly a one-line
   relaxation in `crypto/**`. Lifting it would let entry 9 have an end-to-end
   witness.
2. **PAGE's preprocessed roots are the GLOBAL proof's, not an epoch's.** No
   continuation epoch of any guest carries a PAGE sub-proof (`prove_epoch`
   rejects one). This overturns the entry-7 ruling's condition (b), which asked
   for a witness epoch that cannot exist; the obligation migrates to a
   global-proof verifier.

Wave 7 CLOSED 2026-08-04 (`feat/lfm-assembly`; suite 209 green / 5 ignored,
`make lint` exit 0). **The wrap run happened: the LFM prover proves the assembled
epoch verifier and the LFM verifier accepts it.** Item 6 above has the numbers and
entry 10 has the table. The three things wave 8 inherits:

1. **The hash matrix, which is now the whole remaining question.** Keccak is 84.0%
   of cells per verify, so blake and Poseidon behind the same socket are not a
   refinement of the number — they ARE the number. The e2e that measures them
   exists and is one function call parameterised by options
   (`lfm::wrap_tests::wrap_run`).
2. **A resource ceiling, measured**: the production-shaped wrap (inner blowup 8,
   73 queries) is 11.17 billion cells and needs a projected 350.6 GiB. Nothing
   about the machine blocks it; a box or a cheaper hash does.
3. **The box was never used.** It was busy both times it was checked. A run there
   buys a bigger provable RUNG (4 queries ≈ 70 GiB projected), not the headline.

Superseded — the wave-6 hand-off line, kept for the record: "Ready to start next
(wave 7): the wrap run, whose numbers must state their epoch's trace-length
profile (entry 10)."

Superseded — the wave-6 order of work, kept for the record:
- **Ledger entry 7's wiring, and entry 2 with it.** `programs::emit_register_
  commitment` now exists (extracted in wave 5); Phase A must call it on the
  register-boundary arena the spine already declares, so REGISTER's root is
  COMPUTED and `start_index` is bound. BITWISE and KECCAK_RC intern as program
  constants. DECODE and PAGE need a RULING — wave 5 found DECODE is
  ELF-dependent, so the entry's own taxonomy was wrong; the proposal is in
  ledger entry 7 and it touches program identity, which is an always-stop item.
- **Ledger entries 8 and 9 together** — one synthetic AIR, proved by the
  PRODUCTION prover, with three transition offsets AND `step_size > 1`. Entry 8
  is the OOD absorb ORDER (column- vs row-major), entry 9 is the constraint
  leg's frame-STEP view of the grid; a witness built for one does not close the
  other unless it exercises both.
- **Then the wrap run**, whose numbers must state their epoch's trace-length
  profile (ledger entry 10).

DONE in wave 5 (kept for the record):
- ~~**Assembly, part 2 — hang the legs off the spine.**~~ The seam already
  exists: `epoch::TableAbsorbs` carries every proof-carried cell and
  `epoch::TableChallenges` every derived challenge, per table. What is
  needed is, per sub-proof: reconstruct the full OOD grid from the two
  pruned blocks with program-constant zeros, run the constraint
  evaluation and quotient check at the spine's `z` and `β` powers, then
  per query take `TableChallenges::iota_bits` straight into
  `sub_proof::emit_query_with_bits` and `fri::emit_query_fri`. Only then
  do the composed per-epoch numbers become measurements. Start from
  `epoch_tests::epoch_challenge_program`, which is the assembled program
  minus exactly these legs.
- **Ledger entries 7 + 2, which close together**: intern the three
  constant preprocessed commitments, wire reg-tree's derivation into
  Phase A so REGISTER's root is computed from the register-boundary
  arena the spine already declares (which is what binds `start_index`),
  and decide what to do about PAGE's ELF-dependent commitment.
- **Ledger entry 8** needs a synthetic AIR with three transition offsets
  (or `step_size > 1`), proved by the production prover, or the OOD
  absorb order stays unwitnessed.

After that: the wrap run on the box.

## How to work here

`lfm-standing-decisions.md` is binding: six method rules, the
pre-authorization list, and the always-stop list. The rules exist because
each one caught something. The highest-yield pattern of the phase, stated
generally:

> When all production instances share a degenerate parameter value, a
> differential over production data cannot distinguish implementations that
> differ only off that value. The synthetic case is the only witness.

Three members so far: next-row pruning, the DEEP coefficient stride, and the
`step_size = 1` collapse. Expect more — but check the premise first: the
FRI leg's "no real proof can witness the fold" turned out to be a claim
about the FIXTURES ON HAND, not about the prover, and fell to a
one-parameter change (boundary count) that made real folding proofs in
under a second. "All production instances share the value" and "all
fixtures we happen to have share the value" are different claims; only
the first forces a synthetic witness.

Second-highest: **falsify your own test guards, not just the mechanism.**
Three separate agents found real holes that way — including a tamper suite
whose every vector hit byte 0, so a digest's second word was never checked.
