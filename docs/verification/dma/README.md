# DMA memcpy — oracle + gate-proved chip

Verification artifacts for the DMA memcpy accelerator (PR #874). Same shape as
the BLAKE3 campaign (PR #903, branch `feat/blake3-accelerator`): an independent
Python reference model with external anchors, a z3 gate over the AIR's
constraints, a specification with a soundness ledger, and an executable
transcription audit tying all three to the Rust.

Nothing here changes the chip. The only file outside this directory that the
branch touches is `prover/src/tests/dma_tests.rs`, which gains two tests pinning
the trace builder's row decomposition against the oracle's vectors.

## Why these are here and not in `thoughts/`

PR #903 keeps its equivalent artifacts under `thoughts/blake3/`, force-added past
a `.gitignore` rule. That rule is deliberate: **PR #863 added `thoughts/` to
`.gitignore` and deleted the two files that had leaked into it**, under the
commit subjects "keep working notes out of the tree" and "untrack the working
notes". `thoughts/` is where a coding agent dumps scratch, and the maintainers
decided scratch does not belong in the repo. They were right.

These files are not scratch, and the distinction is testable rather than
rhetorical: all three scripts exit nonzero on failure, take no arguments, need
one `pip install`, and are wired to a Rust test that fails if the oracle and the
trace builder disagree. That is a CI job, not a note. So they live in `docs/`,
which is tracked on purpose, under a name that says what they are.

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
| `dma-oracle/ORACLE.md` | anchor results, the chip-contract map, open questions |
| `dma-oracle/canonical_dma_vectors.json` | 10 pinned vectors with full column expansions |
| `dma-chip/DESIGN.md` | the constraint system + §7 soundness ledger |
| `dma-chip/IMPLEMENTATION.md` | what ships, what was verified, what is still open |
| `dma-chip/z3_dma_verify.py` | the soundness gate |
| `TRANSCRIPTION-AUDIT.md` | does the gate assert what the Rust says? |
| `audit_gate_transcription.py` | the executable half of that audit (83 claims) |

## Results, 2026-08-11

```
oracle:  [1] libc memmove                  PASS  3855 cases x overlap/alignment
         [2] CPython slice assignment       PASS  3855 cases
         [3] row/bus level <-> byte level   PASS  257 lengths x 15 overlaps
         [4] guest stub chunking            PASS  1100 lengths
         [5] mutation sweep                 PASS  6/6 mutants caught
         VALIDATION STATUS: VALIDATED

gate:    layer 1 (row semantics)            PASS  6 UNSAT + 1 deliberate SAT (R1)
         layer 2 (chain structure)          PASS  4 integer + 2 field-exact, all UNSAT
         negative controls                  PASS  8/8 SAT
         width audit (bound necessity)      PASS  6/6
         completeness sweep                 PASS  5410 honest rows, 257 lengths
         OVERALL: PASS

audit:   83 claims, 0 findings; mutation-tested against 5 source mutants, 5 caught

rust:    cargo test -p lambda-vm-prover --lib dma   18 passed
```

## What the gate proves, in one paragraph

Given the modelled lookup contracts and given that bus balance means multiset
equality: every satisfying assignment of one DMA row does what the oracle says
(`tail = count < 8`, `end = count == 0`, `src_incr = src + width` without
wrapping `2^64`, `count_decr = count − width` wrapping only on the terminal row);
the only bus-balanced multi-row structure at depth ≤ 5 is a single chain whose
data rows tile `[src, src+n)` exactly once with the greedy widths; each of the
eight range checks and lookups involved is individually necessary, each with a
named forgery; and the AIR accepts every honest trace for every length `0..256`.

The gate is honest about four things it cannot see — bus wiring, the memory
consistency argument (hence overlap ordering), LogUp soundness, and trace length.
The first is what `audit_gate_transcription.py` exists for.

## The one finding: R1

`count`'s limb split is unconstrained on non-`first` rows. `DmaNext` compares
**packed** field elements, so a Goldilocks alias (`COUNT_1 = 2^32 − 1`,
`COUNT_0 = V + 1`) passes the chain while `lt.rs` sees an integer near `2^64` and
returns `tail = 0` — a row claiming an eight-byte width where fewer than eight
bytes remained.

Not exploitable today: the aliased row's `count_decr` is then near `2^64`, and a
chain must descend to zero to terminate, needing ~2^61 rows. **The obstacle is
trace length, not a constraint**, which is the category of argument that quietly
breaks when a parameter changes. Two one-line fixes in `DESIGN.md` §7 R1.

The gate *exhibits* this rather than assuming it away: MAIN 3 proves the honest
disjunction (`successor honest OR count > 256`) and MAIN 3b confirms the second
branch is reachable, so a reader can tell it is a real gap in the range checks
and not a modelling artefact.

## Method notes worth reusing

**Model the receiving table's constraints, not its advertised contract.** The
gate models `Alu[a,b,LT] → o` as `lt.rs`'s own columns and carries rather than as
`o = (a < b)`. R1 is only visible because of that choice: `lt.rs` range-checks
`lhs[1]` and `lhs[2]` but not the bare `LHS_0` word, and a gate that assumed the
contract would have reported a clean UNSAT.

**Negative controls must be paired with the check they can actually break.**
Three of eight controls initially reported UNSAT because they were dropping a
premise and re-running a check whose reference said nothing about it — a control
that cannot fail. Recorded in `TRANSCRIPTION-AUDIT.md` §4.

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
2. **MEMW-ADDR32 and REG-32**, centrally. Both are assumed by this gate and by
   every other table that sends an address or a register value.
3. **R1**, if the fix is wanted before the trace-length argument is load-bearing.
