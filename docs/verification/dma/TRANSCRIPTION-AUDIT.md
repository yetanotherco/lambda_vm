# Transcription audit — does the DMA gate assert what the Rust actually says?

The gate (`dma-chip/z3_dma_verify.py`) proves things about a **model**. Every
UNSAT it reports is worthless if the model and `prover/src/tables/dma.rs` have
drifted, and the dangerous drift direction is a model **stronger** than the
object it models: it yields UNSAT where the real table is forgeable, and no
positive anchor can catch it, because honest inputs satisfy a correct model and
an over-strong one equally well.

That is not hypothetical. The EC campaign's equivalent audit
(`thoughts/ec-recover-opt/gate/TRANSCRIPTION-AUDIT.md`, referenced by PR #903) found three premises its gate
asserted about the chip and never read, one of them hiding a working forgery,
and the BLAKE3 campaign's found that its "free range check" was declared rather
than derived (PR #903's `thoughts/blake3/GATE-TRANSCRIPTION-AUDIT.md` F1). Both were found by
reading the source against the model, not by running the model.

## Verdict

**No drift.** 83 textual and structural claims, 0 findings
(`python3 audit_gate_transcription.py`, run against the working tree at
`ef9e7526` + the changes in this branch).

**One residual, R1**, which is a property of the AIR and not a transcription
error: `count`'s limb split is unconstrained on non-`first` rows. The gate
reports it rather than assuming it away; details in `dma-chip/DESIGN.md` §7 R1
and §6 below.

**One structural asymmetry worth naming.** Unlike BLAKE3, this campaign's
`DESIGN.md` was written *after* the Rust, so it cannot independently disagree
with it. The independent notion of "correct" comes entirely from the oracle
(`dma-oracle/`, anchored on libc `memmove` and CPython, neither of which knows
this repo exists) and from the audit script keeping the design, the gate and the
Rust textually pinned to each other. Anyone re-reviewing should treat
`DESIGN.md` as *evidence about the gate*, not as evidence about the chip.

## §1 — What was checked, and how hard

`audit_gate_transcription.py` is deliberately **textual** — regex over the Rust
source — rather than a Rust test. The point is to catch a change in `dma.rs` that
nobody reflected in this directory, and a Rust test would be edited in the same
commit as the code it guards.

| section | claims | what it pins |
|---|---|---|
| A. constants | 11 | every number the oracle and gate hard-code |
| B. columns | 28 | the full `dma::cols` layout, `NUM_COLUMNS`, layout density |
| C. constraints | 9 | each index, its template, its operands, the degree bound, and that no index exists the gate does not model |
| D. buses | 21 | 23 interactions, bus mix, every multiplicity, and the five wiring facts the gate explicitly cannot see |
| E. executor | 6 | the ecall validates what the oracle validates, in that order |
| F. generator | 8 | the padding row is the row the oracle describes |

### Sensitivity — the audit was mutation-tested

An audit that cannot fail is not an audit. Five mutants were applied to copies of
the source and the script re-run; all five are caught:

| mutant | findings |
|---|---|
| `timestamp_with_offset(2)` → `(1)` on the write tuple | 1 |
| the write tuple's `value_columns()` → eight zero constants | 1 |
| one `halfword(cols::COUNT_DECR_0)` send deleted | 3 |
| `DMA_MEMCPY_MAX_BYTES + 1` → `+ 2` in the bound lookup | 1 |
| the executor's `if n > DMA_MEMCPY_MAX_BYTES` guard → `if false` | 1 |

The fifth needed a strengthened check to catch: the original version asserted
only that the `DmaMemcpyChunkTooLarge` error variant appeared before the
`checked_add` calls, which a guard rewritten to `if false` satisfies. It now
requires the literal predicate. **That is the class of defect this whole file
exists to find, and the audit found one in itself.**

## §2 — Per-premise table (gate ↔ design ↔ Rust)

| gate premise | modelled as | discharged by | verified where |
|---|---|---|---|
| `IsHalfword[h]` ⟹ `h < 2^16` | hard range bound under `mu == 1` | preprocessed table; the contract *is* the range | audit D: all 12 sends present, on the right columns, at multiplicity `mu` |
| `Zero[v] → z` | `v = x + 256y + 65536z`, x,y bytes, z<16, `z = (v == 0)` | `bitwise.rs` `generate_bitwise_row` | audit A (domain), D (the send's shape, the four `-1` coefficients) |
| Zero **domain** `v < 2^20` | asserted at gate import | halfword bounds put the argument in `[0, 262140]` | gate module assert; audit A |
| `Alu[a,b,LT] → o` | `lt.rs`'s own columns and carries, **not** `o = (a<b)` | `lt.rs` constraints 0–2 | audit D (both lookups, their constants, their multiplicities) |
| MEMW-ADDR32 | address limbs `< 2^32` when `mu − end == 1` | the `Memw` table's address decomposition | **not verified here** — see §5 |
| REG-32 | all three argument limbs `< 2^32` when `first == 1` | the register file / `memw_register.rs` | **not verified here** — see §5 |
| constraint set = 18, degree 2 | 18 modelled constraints, all quadratic | `DmaConstraints` | audit C + the Rust test `dma_constraints_count_and_indices` |
| bus balance = multiset equality | assumed | LogUp | out of scope, stated in the gate docstring |

## §3 — Reference independence

The gate's notion of "correct" is `dma_ref.row_columns`, which is imported from
`dma-oracle/dma_ref.py` and used to *pin* the completeness sweep. So the gate's
positive controls are oracle-driven, not self-consistent.

The oracle's own independence is what `dma-oracle/ORACLE.md` §1 records: libc
`memmove` (3855 cases) and CPython slice assignment (3855 cases) are two
implementations that share no code with `dma_ref.py` or with each other, and the
row-level/byte-level replay equivalence (257 lengths × 15 overlap
configurations) is the bridge between the two levels the AIR straddles.

One thing the reference is **not** independent of: the row decomposition itself.
The greedy `8-while-≥8-then-1` rule is a design decision, and the oracle
transcribes it from the same place the AIR gets it. What the oracle proves is
that this decomposition *implements the byte copy*; it cannot tell you the
decomposition is the best one, and it would not catch a design where both the
AIR and the oracle chunked differently but consistently. That is why the Rust
test `dma_trace_matches_oracle_row_decomposition` exists: it pins the trace
builder's decomposition against the pinned vectors rather than against a
recomputation.

## §4 — Gate sensitivity: the shipped controls

Eight negative controls, each dropping exactly one premise and re-running the
check that premise is load-bearing for. All eight are SAT.

**Pairing matters, and getting it wrong is silent.** Three of the eight were
initially paired with the wrong check and reported UNSAT — a control that cannot
fail:

* `drop_tail_lane_zero` was re-running MAIN 0, whose reference says nothing about
  the value lanes. Now paired with `check_tail_lanes`.
* `drop_lt_bound` was re-running the chain check, whose tiling claim does not
  depend on the per-call bound at depth 3. Now paired with `check_row_budget`.
* `drop_reg32` was re-running MAIN 0, which *assumes* `well_formed()` as a
  hypothesis — so removing the lookup that supplies it changed nothing. Now
  paired with `check_row_budget`, which deliberately does not assume it.

A fourth defect was in the width audit rather than the controls: the truncation
forgery was written at `count = 3`, and it is not reachable there. `end` needs
`count_decr` all-`0xFFFF`, i.e. `count = step − 1`, so a free `tail` buys exactly
`count = 7` and no other value. Corrected to 7 — which makes the forgery worse
(seven bytes lost, not three) and, more usefully, shows the two constraints
compose to leave exactly one hole.

**The most valuable single result.** The four shipped forgery tests in
`prover/src/tests/prove_elfs_tests.rs` each map onto a gate check, and the gate
says *which mechanism blocks each* — which the tests themselves do not, since
they only observe that verification fails:

| shipped Rust forgery test | what it perturbs | the mechanism that rejects it | gate check |
|---|---|---|---|
| `forged_early_end_rejected` | `END := 1` on a data row | the `Zero` lookup: the sum no longer reads zero, so the `Zero` bus unbalances | MAIN 1, and its `drop_zero_end` / `drop_halfword_count_decr` controls |
| `forged_wide_tail_rejected` | `TAIL := 1` on a wide row | the `Alu` width pin: `tail` must equal `count < 8` | MAIN 0, and the width audit's `count = 7` case (which is the *opposite* direction — `tail := 0` on a narrow row) |
| `forged_intermediate_source_rejected` | `SRC_0` **and** `SRC_INCR_0` shifted together | **nothing row-local** — the row's own ADD stays satisfied. The predecessor's `DmaNext` tuple no longer matches, and the source read no longer matches memory | CHAIN / CHAIN-F, which is exactly the check that treats `DmaNext` as a free bijection rather than an assumed chain |
| `forged_value_rejected` | `VALUE[0]` | **not the copy relation** — read and write still agree with each other, because they are one set of columns. What rejects it is the `Memw` read no longer matching memory | none; audit §D pins the one-set-of-columns wiring |

Two of these are worth dwelling on. `forged_intermediate_source_rejected` is the
case where per-row soundness is genuinely insufficient and the chain argument is
doing the work — which is why the gate builds the bijection model rather than
assuming rows are chained. And `forged_value_rejected` passes for a reason no
solver query establishes: the only thing standing behind it is a textual fact
about how two bus tuples are constructed. That asymmetry is the reason the gate
and the audit script are separate artifacts.

## §5 — Premises NOT discharged here (and where they should be)

**MEMW-ADDR32** and **REG-32** are asserted by the gate and verified by nothing
in this directory. Both are "the receiving table range-checks what it receives",
which is the exact shape of BLAKE3's F1 ("the free range check is declared, not
derived"), so they deserve naming rather than assuming:

* **MEMW-ADDR32** — `src0`/`src1`/`dst0`/`dst1` cross the bus as
  `base_address` lo/hi on the two data `Memw` sends, and `memw.rs` decomposes the
  address into range-checked pieces. Chased far enough to be confident, not far
  enough to be a claim. If it were false, `src`'s limb split would be as
  unpinned as `count`'s is in R1.
* **REG-32** — the three register reads bind the limbs to `memw_register.rs`'s
  `VAL_0`/`VAL_1`, which are `Word` columns whose range comes from the memory-bus
  tokens chaining back to the register file. Not a local fact. Note the gate's
  `drop_reg32` control shows what it buys: without it, the `count < 257` bound
  lookup does not cap `count`, because a bound lookup on a residue class caps
  only the residue.

Neither is DMA-specific — every table sending an address or a register value
depends on them — which is an argument for verifying them once, centrally, not
an argument for not verifying them.

## §6 — R1, in the form a reviewer should act on

`DmaNext` equates **packed** field elements. The receiver's
`COUNT_0 + 2^32·COUNT_1` is matched against the sender's four `count_decr`
halfwords as one value; nothing says the successor split its limbs the same way.
`count`'s limbs are bounded only on `first` rows, and `lt.rs` bounds `lhs[1]` and
`lhs[2]` (hence `COUNT_1`) but not the bare `LHS_0` word (hence not `COUNT_0`).

The gate exhibits the alias (MAIN 3b, SAT). Concrete witness, for a row whose
honest count is 3:

```
COUNT_1 = 0xFFFF_FFFF        COUNT_0 = 4
packed  = 4 + 2^32·(2^32−1) = 2^64 − 2^32 + 4  ≡  3   (mod p)   <- passes DmaNext
lt.rs sees the integer 18446744069414584324, which is not < 8   <- tail = 0
```

So the row claims an **eight-byte width where three bytes remained**: up to five
bytes written past the destination end, with every bus balanced. What stops it is
that the row's own `count_decr` is then near `2^64`, and a chain must descend to
`count = 0` to terminate — roughly 2^61 rows. **The obstacle is trace length,
not a constraint.**

Recommended fix, either one:

1. add `IsHalfword` sends on `COUNT_0`/`COUNT_1` at multiplicity `mu` (they are
   `Word` columns, so this needs the same treatment `count_decr` already gets:
   receive `count` as `DWordHL`); or
2. simply receive `count` as `DWordHL` on the `DmaNext` tuple, reusing the
   existing halfword sends.

Either collapses MAIN 3's disjunction to its first branch and makes
`count ≤ 256` an invariant of every row rather than of the head row.

Severity, stated honestly: **not exploitable as the code stands**, and not a
reason to block the PR on its own. It is a soundness argument resting on a
resource bound, which is the category that quietly breaks when someone raises a
trace-length parameter.

## §7 — Still open (report-only, outside this audit's scope)

1. **The memory-consistency argument.** The whole snapshot/overlap story rests on
   `T+1` reads strictly preceding `T+2` writes per address. The gate cannot see
   timestamps; the audit script checks the constants are `+1`/`+2` and that the
   offset only touches the low limb; the oracle's `write_before_read` and
   `interleaved` mutants cover the model side. Nobody has checked the `Memw`
   table's ordering argument for *unaligned 8-byte* accesses, which is what a
   misaligned DMA copy generates on nearly every row.
2. **`count_table_lengths`** — the disk-spill sizing pass. The PR's own
   `count_table_lengths_drift_tests.rs` covers DMA; not re-derived here.
3. **The `n = 0` ecall.** One row, both `first` and `end`, no `DmaNext` traffic,
   no memory operations. Pinned by the completeness sweep and by
   `empty_dma_call_is_a_single_first_and_terminal_row`, but it is the row shape
   most likely to be broken by a future multiplicity change, because every
   multiplicity on it is zero.
4. **Two ecalls at one timestamp.** Ruled out by CPU timestamps being strictly
   increasing per instruction. Asserted, not verified here, and it is what the
   `DmaNext` timestamp binding relies on.

## §8 — Reproducing

```sh
python3 docs/verification/dma/dma-oracle/test_oracle.py        # anchors + emit vectors
python3 docs/verification/dma/dma-chip/z3_dma_verify.py        # the gate (add --quick to shorten)
python3 docs/verification/dma/audit_gate_transcription.py      # this audit
cargo test -p lambda-vm-prover --lib dma              # the Rust side, incl. the vector test
```

`z3-solver` is the only dependency (`pip install z3-solver`; validated on 5.0.0).
All three scripts exit nonzero on failure and are safe to wire into CI; the gate
takes about three minutes for the full board, dominated by the 5410-row
completeness sweep.
