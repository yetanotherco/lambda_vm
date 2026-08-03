# Standing decisions — Phase R agents

Read this before stopping to ask. If your question is answered here, proceed.
Last updated 2026-08-03 by team-lead (rule-7 refinement from the FRI leg).

## Pre-authorized — do NOT ask

- **Merging `origin/main` into your branch** when your premise depends on
  upstream state, using: stash tracked-dirty files → `git merge --ff-only`
  (or a real merge if the branch has commits) → pop → full suite + drift
  tests → regenerate the registry if digests moved. Report what happened.
- **Regenerating `LFM_REGISTRY`** via `cargo run --bin compute_lfm_registry
  --release` and pasting the block, whenever a program or a layout changes.
  Always re-run the drift tests after, and report moved-vs-survived.
- **Adding tests beyond the spec**, including tests that assert a hazard
  still exists (see the guard-test map in `lfm-agent-handoff.md` §8).
- **Refactoring for correctness or clarity inside `prover/src/lfm/`** —
  extracting shared helpers, renaming, splitting files — provided the full
  suite stays green and the public surface other slices depend on is either
  unchanged or reported.
- **Overriding my spec when it is wrong.** You have done this three times
  and been right three times. Implement the correct thing, flag it loudly in
  the report as a deviation with the derivation. Do not implement something
  you believe is wrong because I wrote it.
- **Committing your own slice** to the branch you were given, signed
  (`git -c user.name="Mauro Toscano" -c user.email="maurotoscano2@gmail.com"`),
  no AI attribution or co-author trailers, once the suite is green and lint
  is clean. Never commit red. Never force-push. Never rewrite history.
- **Deciding the internal design** of anything the spec describes by
  behaviour rather than by construction.

## Always stop and ask

- Anything touching `crypto/**` or `prover/src/tables/**` beyond additive
  `BusId` variants — those are production paths shared with the VM.
- Pushing to a remote, opening or merging a PR, or any GitHub write.
- Deleting or rewriting another agent's work, or reverting a guard test.
- A framework ceiling (interaction counts, capture limits, aux widths):
  report it as a finding rather than working around it silently.
- Anything that would make program identity proof-dependent, add a runtime
  off-switch to the registry check, or weaken a soundness obligation to make
  a test pass.

## Method (non-negotiable — these caught every real bug this phase)

1. **Falsify every new mechanism.** Break it deliberately, watch the right
   test fail, revert. If nothing fails, the TEST is wrong, not the mechanism.
2. **Execute-only tests prove nothing about chips.** Where the executor
   mirrors a computation the chip also does, only a prove+verify test sees
   the chip.
3. **Scrutinise the oracle** as hard as the thing under test. A wrong oracle
   looks exactly like a wrong implementation.
4. **Soundness claims need coherent forgeries**, not trace tampering — build
   the attack so every bus balances and every claimed value is consistent,
   then show the one constraint that rejects it.
5. **A deferral's safety argument is itself a claim needing evidence.**
   Deferring work behind a loud assert is fine. Deferring it because you
   believe it is cosmetic, without checking, is not — the check is what tells
   you whether the thing you postponed was a convenience or a soundness
   obligation. Two instances this phase: a trailing-half mask that looked
   cosmetic actually pinned arena bytes past a length prefix (without it a
   prover rewrites the absorbed string while the prefix claims otherwise), and
   a "public output is surely 4-byte aligned" recollection that was simply
   false — public output is one byte per COMMIT op with no alignment
   guarantee. Verify the premise, then defer.
6. **Mark provenance; never assert past your evidence.** In any document,
   separate what you verified first-hand from what you took from someone
   else's report — and give every instrument a "what this cannot see" note
   naming the questions it is structurally unable to answer. Not bookkeeping:
   the one claim this phase that was flat wrong ("the constraint leg is
   workload-shaped, a no-EC epoch drops 65%") was the single sentence its
   author wrote without marking provenance. It was asserted from a node
   census, which cannot see how sub-proofs are ASSEMBLED — and ten tables
   contribute a sub-proof each regardless of whether they have any rows. It
   reached the team lead's durable notes before the author's own
   re-measurement pulled it back. Both halves of this rule answer the same
   question: *how do I stop myself asserting past my evidence?*

7. **A relative test dies the moment its two sides unify.** A differential
   between two code paths is worthless once one is implemented in terms of
   the other — the refactor that makes an API additive-by-delegation is
   exactly what kills any test comparing the two forms, silently, at that
   moment. When you delegate, replace the comparison with an ABSOLUTE
   property of the output. Demonstrated (reg-tree, bits exposure): the
   guard comparing `emit_sub_proof` against `emit_sub_proof_with_bits`
   stayed green with the exact denied defect injected, because both sides
   were the same program by construction; the replacement asserts every
   returned bit is consumed by some `Select`, which the walk's own bits
   satisfy and a fresh copy cannot. This is rule 1 applied to a guard:
   falsify the guard itself, especially right after a delegation refactor.

   REFINEMENT (fri-emitter, 2026-08-03): **a difference of two counts taken
   from our own emitter is still a relative test**, however much it looks
   like an absolute count. `selects(joined) − selects(trace_only)` stayed
   green with the exact denied defect injected, because the defect lived in
   the function BOTH sides call and the difference never moved. The
   marginal-cost idiom (`marginal()`, `marginal_fri`) is safe only when the
   result is compared against a number that did not come from the emitter —
   a pinned prediction, or a closed form over the shapes. Corollary from the
   same session, rule 3 applied to instruments: a falsification harness that
   parses `cargo test -q` for per-test FAILED lines reports every breakage
   as "nothing failed" (failures are named only in the trailing summary
   block) — check the instrument against a known breakage before believing
   "my mutation changed nothing".

## Coordination

- Append one line to `others/lfm-agent-status.log` at every slice boundary.
- Report via SendMessage to `main`; plain text output does not reach me.
- Answers to blocking questions arrive as `others/lfm-team-lead-*.md` files
  as well as mailbox messages — check the directory if a reply seems overdue.
- If your context runs thin, checkpoint and write a handoff file rather than
  delivering a half-built slice. Quality over completion.
