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
`n ≤ 256`; `FIXED_TABLE_COUNT` 10 → 11.

## What this branch adds

| piece | file |
|---|---|
| the reference model (four levels) | `../dma-oracle/dma_ref.py` |
| the validation harness (five anchors) | `../dma-oracle/test_oracle.py` |
| 10 pinned vectors with full column expansions | `../dma-oracle/canonical_dma_vectors.json` |
| the recovered specification + soundness ledger | `DESIGN.md` |
| the z3 gate | `z3_dma_verify.py` |
| the transcription audit (83 claims) | `../audit_gate_transcription.py` |
| the audit's findings and residuals | `../TRANSCRIPTION-AUDIT.md` |
| two Rust tests pinning the oracle's decomposition | `prover/src/tests/dma_tests.rs` |

No change to any shipped constraint, column, bus interaction or executor path.
The only Rust touched is the test module.

## Gates run

**Oracle → external anchors.** `python3 ../dma-oracle/test_oracle.py`, full:
3855 cases against libc `memmove`, 3855 against CPython slice assignment, the
row-level/byte-level replay equivalence over all 257 lengths × 15 overlap
configurations, chunking over 1100 lengths, and a 6-mutant sensitivity sweep.
`VALIDATED`.

**Gate → the design.** `python3 z3_dma_verify.py`, full board, z3 5.0.0, ~3 min:
seven Layer-1 checks (six UNSAT, one deliberately SAT — the R1 witness), six
Layer-2 chain checks UNSAT (four integer, two field-exact), 8/8 negative controls
SAT, 6/6 width-audit rows as expected, and a 5410-row completeness sweep over
every length `0..256`. `OVERALL: PASS`.

**Audit → the Rust.** `python3 ../audit_gate_transcription.py`: 83 claims, 0
findings. Mutation-tested against five source mutants; all five caught (one only
after strengthening the check that missed it — recorded in
`../TRANSCRIPTION-AUDIT.md` §1).

**Rust → the oracle.** `cargo test -p lambda-vm-prover --lib dma`: 18 tests
pass, including the two new ones
(`dma_trace_matches_oracle_row_decomposition` over seven pinned structural cases,
and `dma_maximum_chunk_is_thirty_three_rows_with_no_tail`).

## The one finding: R1

`count`'s limb split is unconstrained on non-`first` rows, so a Goldilocks alias
(`COUNT_1 = 2^32 − 1`, `COUNT_0 = V + 1`) passes `DmaNext` — which compares packed
field elements — while the `LT` lookup sees an integer near `2^64` and returns
`tail = 0`. Such a row claims an eight-byte width where fewer than eight bytes
remained.

Not exploitable as the code stands: the aliased row's own `count_decr` is then
near `2^64`, and a chain must descend to zero to terminate, needing ~2^61 rows.
**But the obstacle is trace length, not a constraint.** Full construction, the
exhibited witness and two one-line fixes: `DESIGN.md` §7 R1 and
`../TRANSCRIPTION-AUDIT.md` §6.

The honest severity: not a blocker on its own, and not a bug anyone can trigger
today. It is a soundness argument resting on a resource bound, which is the
category that quietly breaks when a trace-length parameter changes.

## Premises the gate assumes and this campaign does not discharge

Both are "the receiving table range-checks what it receives" — the same shape as
the BLAKE3 gate's F1 finding, so they are named rather than assumed:

* **MEMW-ADDR32** — the `src`/`dst` limbs are 32-bit, from the `Memw` table's
  address decomposition. If false, `src`'s limb split would be as unpinned as
  `count`'s is in R1.
* **REG-32** — the three argument registers' limbs are 32-bit, from the register
  file via `memw_register.rs`. The gate's `drop_reg32` control shows what it
  buys: without it, `Alu[count, 257, LT] → 1` caps only a residue class, not the
  count.

Neither is DMA-specific; every table that sends an address or a register value
leans on them. That argues for verifying them once, centrally.

## Known limits of the verification, restated plainly

* The gate cannot see bus **wiring**, timestamps, LogUp soundness, or trace
  length. The first is covered by the audit script's §D, the second partly by the
  oracle's `write_before_read`/`interleaved` mutants, the fourth by R1 being
  reported instead of assumed away.
* The chain checks run at depth ≤ 5 (integer) and ≤ 3 (field-exact). The general
  case rests on MAIN 2's wrap lemma plus the strict decrease of `count`, argued
  in `DESIGN.md` §7 and confirmed mechanically at those depths — not proved for
  arbitrary depth.
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

All three scripts exit nonzero on failure and are safe to wire into CI.
