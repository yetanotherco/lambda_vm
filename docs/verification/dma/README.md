# DMA memcpy — oracle + gate-proved chip

Verification artifacts for the DMA memcpy accelerator (PR #874). Same shape as
the BLAKE3 campaign (PR #903, branch `feat/blake3-accelerator`): an independent
Python reference model with external anchors, a z3 gate over the AIR's
constraints, a specification with a soundness ledger, and an executable
transcription audit tying all three to the Rust.

**Scope of the branch.** No shipped constraint, column, bus interaction or
executor path changes. Outside this directory it touches three files:
`prover/src/tests/dma_tests.rs` (two tests driving the real trace-builder
decomposition against the oracle's emitted vectors), a `#[cfg(test)]` accessor in
`prover/src/tables/trace_builder.rs` that exposes that decomposition to the test
(following the `epoch_touched_cells` pattern already in that file), and a
`verify-dma` target in the `Makefile`. Diffed against its base branch
`feat/dma-memcpy`; against `main` you will also see all of PR #874, which is not
merged yet.

## Why these are here and not in `thoughts/`

PR #903 keeps its equivalent artifacts under `thoughts/blake3/`, force-added past
a `.gitignore` rule. That rule is deliberate: **PR #863 added `thoughts/` to
`.gitignore` and deleted the working-note files that had leaked into it** — commit
`f2def578`, subject *"chore: keep working notes out of the tree"*, plus more
deletions in the squash `d83b4d9e` (at least four files across the two).
`thoughts/` is where a coding agent dumps scratch, and the maintainers decided
scratch does not belong in the repo. They were right.

These files are not scratch, and the distinction is testable rather than
rhetorical: all three scripts run unattended, exit nonzero on failure, need one
`pip install`, and the oracle's emitted row table is `include_str!`-ed by a Rust
test that drives the real trace-builder decomposition — so a regenerated oracle is
a compile-time input, not a note. `make verify-dma` runs all three. So they live in
`docs/`, which is tracked on purpose, under a name that says what they are.

Honest caveat: **no CI workflow runs them yet.** The branch establishes
runnability and wires the fixture into `cargo test`; scheduling the Python side in
CI is a separate call for whoever owns the workflow budget.

The reason to commit them at all is the one #903 states: two earlier campaigns in
this repo wrote their verification work to a session scratchpad under
`/private/tmp/...` and lost it, and the BLAKE3 files had to be reconstructed by
replaying tool calls out of subagent transcripts. **A gate nobody can rerun is a
claim, not evidence.** Everything here runs from a clean checkout.

## Contents

| file | what it is |
|---|---|
| `dma-oracle/dma_ref.py` | the reference model: byte semantics, row decomposition, MEMW multiset, guest chunking |
| `dma-oracle/test_oracle.py` | five-anchor validation harness; emits the canonical vectors |
| `dma-oracle/canonical_dma_rows.txt` | line-oriented row table, `include_str!`-ed by the Rust test |
| `dma-oracle/ORACLE.md` | anchor results, the chip-contract map, open questions |
| `dma-oracle/canonical_dma_vectors.json` | 10 pinned vectors with full column expansions |
| `dma-chip/DESIGN.md` | the constraint system + §7 soundness ledger |
| `dma-chip/IMPLEMENTATION.md` | what ships, what was verified, what is still open |
| `dma-chip/z3_dma_verify.py` | the soundness gate |
| `TRANSCRIPTION-AUDIT.md` | does the gate assert what the Rust says? |
| `audit_gate_transcription.py` | the executable half of that audit (100 claims) |

## Results, 2026-08-11

```
oracle:  [1] libc memmove                  PASS  3855 cases x overlap/alignment
         [2] CPython slice assignment       PASS  3855 cases
         [3] row/bus level <-> byte level   PASS  257 lengths x 15 overlaps
         [4] guest stub chunking            PASS  1100 lengths
         [5] mutation sweep                 PASS  8/8 mutants caught
         VALIDATION STATUS: VALIDATED

gate:    layer 1 (row semantics)            PASS  6/6 UNSAT
         layer 2 (chain structure)          PASS  4 integer + 2 field-exact UNSAT
         layer 2 controls                   PASS  4 positive + 3 negative
         negative controls                  PASS  10/10 SAT
         width audit (bound necessity)      PASS  6/6
         completeness sweep                 PASS  5153 honest + 257 padding rows
         OVERALL: PASS                            (~96 s on z3 5.0.0)

audit:   100 claims, 0 findings; mutation-tested against 6 source mutants, 6 caught

rust:    cargo test -p lambda-vm-prover --lib dma   18 passed
```

`make verify-dma` runs all three. Full gate transcript in `dma-chip/DESIGN.md` §9.

## What the gate proves, in one paragraph

Given the modelled lookup contracts and given that bus balance means multiset
equality: every satisfying assignment of one DMA row does what the oracle says
(`tail = count < 8`, `end = count == 0`, `src_incr = src + width` without
wrapping `2^64`, `count_decr = count − width` wrapping only on the terminal row);
among groups containing exactly one head row, the only bus-balanced multi-row
structure at depth ≤ 5 is a single chain whose data rows tile `[src, src+n)`
exactly once with the greedy widths; each of the **ten** range checks and lookups
involved is individually necessary, each with a named forgery; and the AIR accepts
every honest trace for every length `0..256`.

The gate is honest about four things it cannot see — bus wiring, the memory
consistency argument (hence overlap ordering), LogUp soundness, and trace length.
The first is what `audit_gate_transcription.py` exists for.

## No open soundness gap — and a retracted finding worth reading about

The board is clean: no residual, no known hole in the chip. An independent
security scan, deliberately blinded to these artifacts, reached the same
conclusion and independently re-derived all ten items of `DESIGN.md` §7.

An earlier version of this campaign reported one — "RESIDUAL R1", that `count`'s
limb split was unconstrained on non-head rows — and published it as the headline
result across five documents. **It was wrong**, and the story is the most useful
thing here. The gate modelled the `DmaNext` hop as one equation on a packed
64-bit value; the bus actually binds **two 32-bit elements** with separate alpha
powers, so the limbs are pinned and the alias the gate exhibited is unreachable.
`DESIGN.md` §7 carries the full account.

Three transferable lessons:

* **The direction of a modelling error decides its cost.** Weaker than the AIR ⇒
  false alarms, never false proofs; every UNSAT survived the correction. Stronger
  than the AIR ⇒ false proofs that no positive anchor can catch. Classify every
  gap before trusting any result.
* **A phantom finding induces real damage.** Working around R1 led to asserting
  `count ≤ 256` on *every* row of the field-exact chain check, when the AIR bounds
  only the head — a genuine over-strong assumption, in the dangerous direction.
* **A proposed fix that is a no-op means the gap isn't there.** R1's second fix
  was "receive `count` as `DWordHL`", which changes nothing under the real
  semantics.

## Method notes worth reusing

**Model the receiving table's constraints, not its advertised contract.** The
gate models `Alu[a,b,LT] → o` as `lt.rs`'s own columns and carries rather than as
`o = (a < b)` — `lt.rs` range-checks `lhs[1]` and `lhs[2]` but not the bare
`LHS_0` word, so the contract form would hide which limbs are actually pinned.
**But apply the same rule to the bus itself, which is what the retracted R1 got
wrong:** "how many field elements does this value cross the bus as?" is a premise
like any other, and must be read from `num_bus_elements()` rather than assumed.
The audit's §G now asserts it.

**Negative controls must be paired with the check they can actually break.**
Three of the original eight reported UNSAT because they dropped a premise and
re-ran a check whose reference said nothing about it — a control that cannot fail.
And a multi-row check needs *its own* controls: Layer 2 shipped with neither a
positive (is the premise set even satisfiable?) nor a negative one. Both are on
the board now. `TRANSCRIPTION-AUDIT.md` §4.

**Do not negate a modular equality that carries a witness quotient.** The gate's
first run reported a bogus SAT on its main check for exactly this reason:
`Not(a − b == k·2^64)` is satisfiable by picking a nonzero `k`. Under negation,
spell the claim out witness-free. (Encoding note in `FieldRow`.)

**Field-exact, over integers, linear.** Every column is an `Int` in `[0, p)`,
modular equalities carry explicit quotients, and `x·(1−x) = 0` becomes
`x ∈ {0,1}` (exact for `x < p` prime). A bit-vector model cannot answer a
"is a range check missing?" question at all, since it bounds the unconstrained
column for free — but the naive `%p` encoding is nonlinear and the first version
of this gate timed out on its own main check. The rewrite is what made the
5410-row completeness sweep affordable.

**Mutation-test the audit, not just the model.** Five source mutants; one was
initially missed, because the check asserted an error variant was *mentioned*
rather than that the guard existed — a `if false` guard passed. That is precisely
the defect class the audit exists to catch, found in the audit itself.

## Where to send the next reviewer

1. **The `Memw` ordering argument for unaligned 8-byte accesses.** A misaligned
   DMA copy generates one on nearly every row, and the snapshot/overlap story
   rests entirely on `T+1` reads preceding `T+2` writes per address. Nobody has
   checked it. Largest remaining gap around this feature, and not DMA's to fix.
2. **Assumptions A1–A4** (`DESIGN.md` §Assumptions), centrally rather than
   per-chip. `IS_WORD` appears across ~10 spec chapters *exclusively* inside
   `[[assumptions]]`, with no interaction, no template and no 2³² table — so the
   spec asserts a range obligation for nearly every address, register value and
   timestamp in the VM without naming a discharger. That vacuum is what an earlier
   draft of `DESIGN.md` filled by inventing labels for it.
3. **`spec/memw.typ`'s `value` obligation**, which it assigns to "the system as a
   whole", i.e. to nobody. DMA is a concrete sender for which no chip discharges it.
4. **For PR #874, not this branch:** `end·(1 − tail) = 0` would close the
   `count = 7` seven-byte-truncation hole inside the AIR instead of leaving it
   entirely to the `Alu` bus (defense-in-depth — the pin is sound today); DMA is
   the only high-volume table with no `max_rows`/chunking; and there is no
   `spec/dma.typ`.
