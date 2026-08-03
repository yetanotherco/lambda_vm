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
5. **Assembly** into one epoch-verifier program. ⚠ Every per-epoch number so
   far is a COMPOSITION of per-AIR measurements, not a run. Assembly is what
   confirms or falsifies them. Discharge `lfm-assembly-obligations.md` —
   six OPEN entries as of 2026-08-03 (reg_fini width, start_index binding,
   the five two-consumer unifications, the FRI transcript obligations, the
   index-as-bits requirement, the challenges guard); the ledger IS the
   assembly spec's skeleton.
6. **The wrap run** on the box (see `[[scaleway-box-idp]]` in memory:
   195.154.218.198, 124 GB, warm-built).

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

Ready to start immediately (wave 4 respawn):
- **Assembly** — compose the legs into ONE epoch-verifier program and
  run it. The brief is `lfm-assembly-obligations.md` (the six OPEN
  entries are the spec's skeleton) plus the leg handoffs
  (`lfm-logup-handoff.md`, `lfm-fri-verify-spec.md` Addendum 2, the
  join/reg-tree sections above) — and point the agent at
  `lfm-team-lead-start-index-research.md` so entry 2's reading is not
  re-derived. The composed per-epoch numbers are predictions to confirm
  or falsify, not facts.

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
