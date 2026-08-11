# DMA memcpy chip — implementation notes

Companion to `DESIGN.md`: what the shipped Rust does, where the verification
artifacts touch it, and what is still open. The order of events here is the
reverse of the BLAKE3 campaign's — the Rust (PR #874) came first and this
directory was added on top (PR: `feat/dma-memcpy-formal-verification`), so this
file records **what was verified about existing code**, not deltas from a design.

## What ships in PR #874

| piece | file |
|---|---|
| the ecall | `executor/src/vm/instruction/execution.rs`, `SyscallNumbers::DmaMemcpy` |
| the table | `prover/src/tables/dma.rs` (32 columns, 18 constraints, 23 bus interactions) |
| the trace replay | `prover/src/tables/trace_builder.rs`, `collect_dma_memcpy_ops` |
| the no-overflow template | `prover/src/constraints/templates.rs`, `emit_add_pair_no_overflow` |
| the guest stub | `syscalls/src/syscalls.rs`, the strong `memcpy` symbol |

Syscall number `u64::MAX - 2`; ABI `memcpy(dst = x10, src = x11, n = x12)` with
`n ≤ 256`; `FIXED_TABLE_COUNT` **11 → 12** (PR #876's hint table landed on
`main` in between, so the 10 → 11 in #874's own PR body is now stale).

## What this branch adds

| piece | file |
|---|---|
| the reference model (four levels) | `../dma-oracle/dma_ref.py` |
| the validation harness (five anchors) | `../dma-oracle/test_oracle.py` |
| 10 pinned vectors with full column expansions | `../dma-oracle/canonical_dma_vectors.json` |
| the recovered specification + soundness ledger | `DESIGN.md` |
| the z3 gate | `z3_dma_verify.py` |
| the transcription audit (100 claims) | `../audit_gate_transcription.py` |
| the audit's findings and residuals | `../TRANSCRIPTION-AUDIT.md` |
| two Rust tests driving the real decomposition | `prover/src/tests/dma_tests.rs` |
| a `#[cfg(test)]` accessor for that decomposition | `prover/src/tables/trace_builder.rs` |
| a `verify-dma` target | `Makefile` |

No change to any shipped constraint, column, bus interaction or executor path.
The only non-test Rust is a `#[cfg(test)]` function in `trace_builder.rs`, which
compiles out of the library entirely; it exists because the row decomposition and
its `MemoryState`/`RegisterState` operands are module-private, and testing the
public `generate_dma_trace` instead is what made the first version of these tests
vacuous (that function only formats an already-decomposed op list into columns).
The file already used this pattern for `epoch_touched_cells`.

## Gates run

**Oracle → external anchors.** `python3 ../dma-oracle/test_oracle.py`, full:
3855 cases against libc `memmove`, 3855 against CPython slice assignment, the
row-level/byte-level replay equivalence over all 257 lengths × 15 overlap
configurations, chunking over 1100 lengths, and a 6-mutant sensitivity sweep.
`VALIDATED`.

**Gate → the design.** `python3 z3_dma_verify.py`, full board, z3 5.0.0, ~96 s:
six Layer-1 checks UNSAT, six Layer-2 chain checks UNSAT (four integer, two
field-exact) plus four positive and three negative Layer-2 controls, 10/10
premise controls SAT, 6/6 width-audit rows as expected, and a completeness sweep
of 5153 honest rows + 257 padding rows over every length `0..256`.
`OVERALL: PASS`. Verbatim transcript in `DESIGN.md` §9.

The solver is **not** pinned, and that is a real caveat: the queries are
version-independent in meaning, but older solvers are far slower on the
field-exact chain (`CHAIN-F 2` measured 0.45 s on 5.0.0 against 7.60 s on 4.12.2,
17×), so on 4.12.2 the whole board takes ~1210 s and two queries blow their
budgets and report `unknown`. **`unknown` is scored as failure everywhere, never
as success**, so an old solver produces a false alarm and never a false proof —
and the gate now prints its solver version, warns when it is older than the
validated one, and prints a legend saying `unknown` means a timeout rather than a
soundness problem.

**Audit → the Rust.** `python3 ../audit_gate_transcription.py`: **100 claims, 0
findings**, per-section counts printed by the script rather than documented by
hand. Mutation-tested against six source mutants, all six caught — including the
one that matters most, `num_bus_elements(DWordHL) 2 → 1`, whose absence from the
audit is what let the retracted R1 through. Two of the six needed the check
strengthened before they were caught (`../TRANSCRIPTION-AUDIT.md` §1). Source is
whitespace-normalised before literal matching, so a `rustfmt` reflow no longer
produces a spurious finding, and the audit no longer imports the solver (~80 of
its claims need no z3).

**Rust → the oracle.** `cargo test -p lambda-vm-prover --lib dma`: 18 tests
pass, including the two new ones
(`dma_trace_matches_oracle_row_decomposition` over seven pinned structural cases,
and `dma_maximum_chunk_is_thirty_three_rows_with_no_tail`).

## The retracted finding

An earlier version of this campaign reported "RESIDUAL R1" — that `count`'s limb
split was unconstrained on non-head rows — and made it the headline result across
five documents. **It does not exist.** `DmaNext` binds each 64-bit value as two
32-bit bus elements with separate alpha powers, not as one packed field element,
so the limbs are pinned by the predecessor's `IsHalfword`-checked halfwords. The
gate's weaker model manufactured an alias the real bus rejects. Full account and
the transferable lessons: `DESIGN.md` §7.

What it cost, beyond the retraction: working around the phantom led to asserting
`count ≤ 256` on *every* row of the field-exact chain check when the AIR bounds
only the head — a genuine over-strong assumption, which is the direction that
yields false proofs. Both are fixed; the bound is now derived from the head row
through the limb-wise link, as MAIN 3 proves.

## Premises the gate assumes and this campaign does not discharge

These are **caller obligations the spec states**, not checks the receiver
performs — `spec/src/memw.toml` and `spec/src/memw_register.toml` both carry them
as `[[assumptions]] IS_WORD[...]`. An earlier draft of `DESIGN.md` had the
direction backwards and invented the labels "MEMW-ADDR32"/"REG-32" for them;
they are now A1–A4 in `DESIGN.md` §Assumptions.

They bind the **head row only** — every other row's limbs are derived through the
`DmaNext` link. The gate's `drop_reg32` control shows what A2 buys: without it,
`Alu[count, 257, LT] → 1` caps only a residue class, not the count.

None is DMA-specific; every table that sends an address, a register value or a
timestamp leans on them, and `IS_WORD` has no named discharger anywhere in the
spec. That argues for settling it once, centrally.

## Known limits of the verification, restated plainly

* The gate cannot see bus **wiring**, timestamps, or LogUp soundness. The first is
  covered by the audit script's §D and §G, the second partly by the oracle's
  `write_before_read`/`interleaved` mutants. §G is new, and its absence is what
  let the retracted R1 through.
* The chain checks run at depth ≤ 5 (integer) and ≤ 3 (field-exact), and prove
  the tiling **among groups containing exactly one head row**. `ChainRow` carries
  no timestamp, so the multi-call case is out of model rather than covered; the
  `ts` in both `DmaNext` tuples is what separates two calls, and it is guarded
  textually by the audit. The general depth case rests on MAIN 2's wrap lemma plus
  the strict decrease of `count`.
* **Nobody has checked the `Memw` ordering argument for unaligned 8-byte
  accesses**, which is what a misaligned DMA copy generates on nearly every row.
  That is the largest remaining gap around this feature and it is not DMA's to
  fix. `../TRANSCRIPTION-AUDIT.md` §7 item 1.
* `DESIGN.md` was recovered from the implementation, so it cannot independently
  disagree with it. The independent notion of "correct" is the oracle's, and the
  audit script is what keeps the three artifacts pinned together.

## Reproducing

```sh
pip install z3-solver                                   # validated on 5.0.0
python3 docs/verification/dma/dma-oracle/test_oracle.py          # anchors, emits the vectors
python3 docs/verification/dma/dma-chip/z3_dma_verify.py          # the gate  (--quick to shorten)
python3 docs/verification/dma/audit_gate_transcription.py        # 83 transcription claims
cargo test -p lambda-vm-prover --lib dma                # the Rust side
```

`make verify-dma` runs all three. Each exits nonzero on failure; the oracle exits
**2** when it ran but an anchor skipped, so a degraded run is distinguishable from
a clean one. No CI workflow schedules them yet — that is a separate call.
