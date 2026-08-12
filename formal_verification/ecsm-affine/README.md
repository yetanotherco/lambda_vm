# ECSM affine selector — oracle + z3-proved soundness gate

Formal-verification campaign for **PR #879** (`perf/ecsm-affine-selector`), which adds an
affine variant of the ECSM ecall: input `xG‖yG` (64 B), output `xR‖yR` (64 B), with an
`IS_AFFINE` selector column so one prover serves both ABIs.

Same shape as the two earlier campaigns — an independent Python model of the function,
anchored against third parties, plus a z3/sympy gate over the constraints, plus a
transcription audit of the gate against the Rust:

- `thoughts/blake3/` on branch `feat/blake3-accelerator` (PR #903) — the gate-then-implement
  pattern this follows, and the source of the "audit the transcription" lesson.
- `thoughts/ec-recover-opt/` on branch `feat/ec-lincomb2` — the ORIGINAL ECSM/ECDAS board.
  **This campaign imports its lemmas rather than repeating them.**

Neither of those paths exists on `main`: `thoughts/` is gitignored, and both campaigns live
only on their own unmerged branches. Citations to them name the branch and commit for that
reason.

### Why this lives in `formal_verification/`, not `thoughts/` or `docs/`

`formal_verification/<chip>/` is the layout PR **#923** establishes as "the canonical, reusable
template" for machine-checked chip verification, and `tooling/loc` (extended in that PR) does
`read_dir("formal_verification")` and reports **every subdirectory** as one verified gate. So
placing a campaign here gets it counted automatically; nothing else has to be wired up, and
this PR touches no shared file.

`thoughts/` was the other candidate and is wrong: `d83b4d9e` (PR #863, 2026-07-31) added it to
`.gitignore` under the comment *"Profiling outputs … and working notes"* and in the same commit
**deleted** the two files that had leaked in. Nothing under `thoughts/` is tracked on `main`
today — `git ls-tree -r --name-only origin/main -- thoughts/` returns 0 files. PR #903
force-adds back into it and argues the case in its own README, but #903 is unmerged, so that is
one PR relitigating a merged decision rather than precedent.

The file layout here is nested (`oracle/`, `gate/`, `harness/`) where #923's keccak instance is
flat. #923's own instruction is to clone the **method**, not the filenames, and its "Mandatory
discipline" — negative controls of two kinds, *changed* and *removed* constraints, because an
over-constrained model hides a missing one — is satisfied here by the mutation-tested audit
premises and the drop-the-constraint controls (A1e, A1f, A2g, A3d). At 28 files against
keccak's 9, subdirectories earn their keep, and the LOC tooling counts the whole tree either
way.

## Status

**Board fully green.** See [`gate/RESULTS.md`](gate/RESULTS.md) for the lemma table and the
soundness theorem.

```
audit:   20/20 premises read from source, 20/20 mutation controls bite, 0 failures
oracle:  ORACLE STATUS: VALIDATED   (9 pass, 0 skip, 0 fail)
anchor:  32 real witnesses from ecsm::compute_witness{,_with_y}, every field re-derived
A1 selector    : IS_BIT + µ-gating PROVED, Ecall pinning PROVED, 3 forgery controls
A2 YrLtP       : lift + strict chain PROVED, forgery instantiated on a real y = 1 point
A3 parity      : forgery instantiated (2 full witnesses), the yG read PROVED to close it
A3g            : yG canonicality is UNCHECKED in spec AND impl — Medium, VM-parity gap
A4 addressing   : LT bound == executor's predicate PROVED, u64-wrap control FORGES
```

Seven distinct attacks (11 `SAT` results; several are exhibited from more than one
angle). **Every new check in PR #879 has a control showing it is
load-bearing** — the `yG` read, `YrLtP`, `IS_AFFINE`'s bit constraint and its µ-gate, the
`Alu` LT senders and the `u128` widening each admit a concrete attack when removed.

## What is being verified, and what is imported

PR #879 does not change the double-and-add chain, the curve relations, or the ECDAS buses.
Those were taken to a green board by the earlier EC campaign, whose **L1–L7 and contracts
C1–C7 are hypotheses here**. What is new, and what this board covers:

| Added by the PR | Lemma |
|---|---|
| the `IS_AFFINE` column, `IS_BIT` (idx 421) and `AffineZeroOnPadding` (idx 422) | A1a, A1b, A1e |
| the `Ecall` receiver's `xonly + IS_AFFINE·(affine − xonly)` syscall words | A1c |
| the `IS_AFFINE`-gated `yG` read (4 dwords at `addr_xG + 32 + 8i`, `ts`) | A3 |
| the `IS_AFFINE`-gated `yR` write (4 dwords at `addr_xR + 32 + 8i`, `ts+3`) | A3c, A4f |
| `OverflowKind::YrLtP` — the `yR < p` chain (idx 413..420) + 16 halfword columns | A2 |
| the `Alu` LT address-limb senders and their mode-dependent bound | A4 |
| the executor's 64-byte spans and `u128` overlap guard | A4e |

Two of the imported lemmas are stressed in a way their own statements did not anticipate, so
they are **re-examined rather than imported blind**:

- **L7** concluded `xR = x(k·P)` "for both `yG` sign classes", *because* the parity was
  unobservable. Publishing `yR` retires that premise — A3 is the whole subject.
- **C4** listed `YR` as inheriting byte-ness "from tuple equality with ECDAS's byte-checked
  `yR` (or `YG` for k=1)". `YrLtP` now *consumes* that, so it is written down as **C4-YR** and
  its provenance checked (A2d, audit P17) instead of used silently.

## The two findings a reader should not skip

**The `y = 1` point.** The PR's soundness section argues the `YrLtP` band is populated
("constructible: `3 | p−1` makes cubing 3-to-1"). Checked, and it understates the case — the
*first* candidate works. There is a real secp256k1 point with `y = 1`:

```
x = 0x1fe1e5ef3fceb5c135ab7741333ce5a6e80d68167653f6b2b24bcbcfaaaff507,  y = 1
```

so the attack instance sits at the very bottom of a `2^32 + 977`-wide band, and `crypto/ecsm`
itself returns `y_r = 1` for it. `gate/a2_yr_lt_p.py` carries the `yR + p` / `q2 − 1` forgery
all the way through the ECDAS `Yr` relation *and its carry window* — a forgery its windows
reject is not a forgery.

**The parity gap is instantiated, not argued.** `gate/a3_parity_binding.py` builds two
complete witnesses over the same `(xG, k)` — one per root of `xG³ + b` — evaluates the entire
in-table constraint set on each, and shows both are valid with the same `xR` and different
`yR`. `gate/a6_real_witness.py` then reproduces the same pair 9 times straight out of
`ecsm::compute_witness_with_y`, so the gap does not depend on the model being right.

## Contents

| path | what it is |
|---|---|
| `oracle/ecsm_affine_ref.py` | independent secp256k1 + the two ecall semantics + the ABI predicates. No `k256`, no repo code |
| `oracle/test_oracle.py` | 9 independent anchors; a missing fixture SKIPs only itself |
| `oracle/small_y_point.py` | constructs the `y = 1` attack instance via cube roots mod `p` |
| `oracle/small_y_point.json` | the instance, consumed by A2 |
| `gate/affine_common.py` | the transcribed model of the new AIR surface, with citations |
| `gate/a1_selector.py` | A1 — `IS_AFFINE` is a bit, dead on padding, and pinned |
| `gate/a2_yr_lt_p.py` | A2 — `YrLtP`: lift, strict chain, width, C4-YR, the forgery |
| `gate/a3_parity_binding.py` | A3 — the parity forgery and the read that closes it |
| `gate/a4_addressing.py` | A4 — address bounds, the `+32…+63` span, the overlap guard |
| `gate/a6_real_witness.py` | the real-witness (column) anchor |
| `gate/audit_transcription.py` | A5 — 20 premises read from source + mutation controls |
| `gate/RESULTS.md` | lemma board, soundness theorem, contracts, findings, method notes |
| `gate/TRANSCRIPTION-AUDIT.md` | the audit's prose half: premise table and what it cannot see |
| `gate/logs/` | run logs, and the real-witness dump |
| `harness/` | tiny Rust binary dumping real `EcsmWitness` values as JSON |
| `run_gate.sh` | runs everything in dependency order |

## Running it

```bash
python3 -m venv .venv && ./.venv/bin/pip install z3-solver sympy ecdsa
./run_gate.sh                 # everything, logs to gate/logs/
./run_gate.sh --quick         # reuse the witness dump, skip the cargo build
```

A few seconds plus one `cargo build`. The harness depends only on `crypto/ecsm`, deliberately:
a harness that needs the whole prover does not get run. Without a `.venv` the runner falls
back to `python3` from `PATH`.

Committing is a plain `git add formal_verification/ecsm-affine` — no `-f`, no pathspecs. That
is the practical reason this directory is not under `thoughts/`: `-f` would be required
there, and **`-f` overrides the nested `.gitignore` too**, so it stages `.venv/` and
`harness/target/` as well (5108 files instead of 28). Outside an ignored parent, the nested
`.gitignore` below does its job:

```
$ git add -nf thoughts/camp    ->  .gitignore, .venv/junk, gate.py, harness/target/junk.o
$ git add -n  formal_verification/camp  ->  .gitignore, gate.py
```

So **keep `.gitignore`** — it is what excludes the venv and the build output. Sanity-check
before committing:

```bash
git add -n formal_verification/ecsm-affine | wc -l    # ~28, not 5108
```

`ecdsa` is optional — without it the oracle reports `PARTIALLY VALIDATED` and **names the
anchor it is not anchored on**, rather than printing a green banner that outlives its evidence.
(That banner defect is one BLAKE3's harness shipped with; see `thoughts/blake3/README.md`
"Harness defects", on branch `feat/blake3-accelerator`.)

## Method notes worth knowing before extending this

- **`x·(1−x) ≡ 0 (mod p_g)` is not a z3 query.** In lifted integer form with a free quotient
  it does not terminate at this modulus, and neither does 160-bit bit-blasting. These are
  root-of-a-polynomial-over-a-field statements, discharged by factoring over `GF(q)` and
  checking the split is **complete**. z3 keeps the bounded/linear work: the carry lifts,
  the predicate-equivalence sweeps, the quantified interval statement.
- **Two fields, and it matters.** The AIR is over `GF(p_g)`; the curve is over `GF(p)`.
  `field_roots` takes the modulus explicitly because an early A3a factored the curve
  polynomial over Goldilocks and correctly reported FAIL.
- **Every control was seen to go red.** Two of them started out vacuous — A2c recomputed its
  witness for the perturbed constant (so any constant passed), and audit premise P18 matched a
  *comment* about the timestamp stride rather than the stride. Both are written up in
  RESULTS.md Findings 5–6. A green control is worth nothing until you have watched it fail.

## What this gate cannot see

Bus wiring and lookup coverage — the same boundary both earlier gates recorded about
themselves. A1c's pinning, A3c's "at most one witness matches the caller's buffer", and C4-YR
all reduce to contract C5 (LogUp balance) plus the imported L6. An arithmetic gate cannot
catch a mis-wired receiver; the e2e `prove + verify` tests in `prover/src/tests/ecsm_tests.rs`
and `prove_elfs_tests.rs` are what cover that, and they are green on this branch.
