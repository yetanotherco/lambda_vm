# Team-lead call for keccak-probe — R1d blocker resolution

Written 2026-07-29 ~19:30Z as a filesystem fallback because two mailbox
authorizations apparently did not reach you (msgs 3007f24b, db4f6d79).
This file is the operative instruction; it answers your (A)/(B) question.

## THE CALL: (A) — MERGE. Authorized.

Protocol (stops are hard stops — report and wait):

1. `git stash push prover/src/lib.rs prover/src/tables/types.rs` — the only
   dirty TRACKED files. NOTE: `git stash list` already shows an unrelated
   pre-existing stash (`bench-keccak-vs-leanvm WIP`) — leave it alone; your
   push becomes stash@{0}, pop takes it back off, the old one stays put.
   Verify `git status` shows a clean tracked tree (untracked lfm/ + bin +
   others/ remain, untouched by merge mechanics).
2. `git merge origin/main` — feat/lfm has zero local commits, so this must
   be a clean FAST-FORWARD to 5fd961a0. Anything else: STOP.
3. `git stash pop` (your stash@{0}) — our lib.rs hunk is ~line 21, theirs
   ~209–270; types.rs BusId arms are ours alone. Conflict: STOP.
4. Full lfm suite + drift tests. If digests moved (crypto/math changes could
   perturb the commit pipeline): regenerate via
   `cargo run --bin compute_lfm_registry --release`, paste, re-run, and
   report MOVED vs SURVIVED as a finding either way.
5. Confirm `out_buf`/`out_pos` exist in
   crypto/crypto/src/fiat_shamir/default_transcript.rs, then finish R1d in
   one pass per the original spec, with your corrections folded in:
   ext3 = 3 independently rejection-sampled candidates (≈3× per-draw
   completeness loss — state the per-proof bound at real draw counts), and
   sample_u64 at pow-2 bounds = low nbits of the BE u64, no rejection.

Your interim work is ACCEPTED: the sample() replay + reversed-coefficient
Linear design, the vacuous-test catch (keep both tests; the lesson is noted
and your R1b/R1c audit of it is appreciated), and building the
version-independent half without waiting was the right judgment call.

Going forward: check this directory for `lfm-team-lead-*.md` whenever a
blocker answer seems overdue — I will use files for anything
authorization-shaped from now on, with mailbox pings as notification.
