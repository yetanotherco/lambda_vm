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

**No drift.** 100 textual and structural claims, 0 findings
(`python3 audit_gate_transcription.py`, against the working tree at `ef9e7526`
plus this branch). Per-section counts are printed by the script; §1 quotes that
output rather than restating it.

**No residual.** An earlier version of this audit carried one — "R1", that
`count`'s limb split was unconstrained on non-head rows — and it was **wrong**:
`DmaNext` binds each 64-bit value as two 32-bit bus elements, not one packed
field element, so the limbs are pinned. §6 below is now the account of that error
rather than a finding, and §1 records the audit gap that let it through: the
audit checked that the packing *names* appeared and never that the element counts
aligned. Section **G** now does.

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

Counts below are the script's own output, not prose. An earlier version stated
them by hand and got **five of six wrong** — apportioned to sum to the real total
of 83 rather than measured, which is exactly the "declared, not derived" defect
this file exists to catch. The script now prints them.

```
  A. constants          10 claims   every number the oracle and gate hard-code
  B. columns            28 claims   the full dma::cols layout, NUM_COLUMNS, density
  C. constraints        10 claims   each index, template, operands, the degree bound,
                                    and that no index exists the gate does not model
  D. buses              23 claims   23 interactions, bus mix, every multiplicity, and
                                    the four wiring facts the gate cannot see
  E. executor            5 claims   the ecall validates what the oracle validates,
                                    in that order
  F. generator           7 claims   the padding row is the row the oracle describes
  G. bus packing        14 claims   element counts per Packing and DmaNext tuple
                                    alignment -- the section whose absence hid R1
  H. fixture pinning     3 claims   the Rust test consumes the oracle's output
```

### Sensitivity — the audit was mutation-tested

An audit that cannot fail is not an audit. Six semantic mutants and one
must-not-fire control were applied to copies of the source and the script re-run:

| mutant | findings | notes |
|---|---|---|
| `timestamp_with_offset(2)` → `(1)` on the write tuple | 1 | |
| the write tuple's `value_columns()` → eight zero constants | 1 | |
| one `halfword(cols::COUNT_DECR_0)` send deleted | 3 | |
| `DMA_MEMCPY_MAX_BYTES + 1` → `+ 2` in the bound lookup | 1 | |
| the executor's `if n > DMA_MEMCPY_MAX_BYTES` guard → `if false` | 1 | needed a strengthened check |
| **`num_bus_elements(DWordHL)` `2 → 1`** | 1 | **the mutant that was not caught at all** |
| a `rustfmt`-style reflow of `if tail { 1 } else { 8 }` | **0** | must NOT fire — see below |

Two of these are the point of the section.

**The `num_bus_elements` mutant was not caught before**, because §D checked only
that the strings `DWordWL`/`DWordHL` appeared in the two tuples and never that
their element counts aligned. That is the gap the retracted R1 came through, and
it is a textbook instance of the failure this file is written to prevent: the gate
asserted a premise about the bus that nothing ever read from the source. §G now
asserts the element count of every `Packing` variant, that no variant folds a
64-bit value into one element, that `DWordHL` accumulates two halves at
consecutive alpha powers, and that both `DmaNext` tuples carry 8 elements.

**The reflow mutant must produce zero findings, and used to produce two.** The
literal checks match fragments like `if tail { 1 } else { 8 }`, and `rustfmt`
breaks those across lines the moment one grows past `max_width`. The original
guard, `src.replace("\n", " ")`, collapsed the newline but left the indentation,
so it could never match a reflowed form — dead code. `read()` now
whitespace-normalises. This matters because the script is meant to run
unattended: a spurious red is how a check gets deleted rather than fixed.

The executor mutant also needed strengthening: the original asserted only that the
`DmaMemcpyChunkTooLarge` variant appeared before the `checked_add` calls, which a
guard rewritten to `if false` satisfies. It now requires the literal predicate.
**Three of the seven were initially missed, in a file whose entire job is catching
exactly this** — and a fourth was found later still: §G's "no variant folds a
64-bit value into one element" searched for an ASCII `2x` where the source writes
`2×` (U+00D7), so the regex could never match and the claim was dead code. The
`2 → 1` mutation now produces two findings rather than one.

## §2 — Per-premise table (gate ↔ design ↔ Rust)

| gate premise | modelled as | discharged by | verified where |
|---|---|---|---|
| `IsHalfword[h]` ⟹ `h < 2^16` | hard range bound under `mu == 1` | preprocessed table; the contract *is* the range | audit D: all 12 sends present, on the right columns, at multiplicity `mu` |
| `Zero[v] → z` | `v = x + 256y + 65536z`, x,y bytes, z<16, `z = (v == 0)` | `bitwise.rs` `generate_bitwise_row` | audit A (domain), D (the send's shape, the four `-1` coefficients) |
| Zero **domain** `v < 2^20` | asserted at gate import | halfword bounds put the argument in `[0, 262140]` | gate module assert; audit A |
| `Alu[a,b,LT] → o` | `lt.rs`'s own columns and carries, **not** `o = (a<b)` | `lt.rs` constraints 0–2 | audit D (both lookups, their constants, their multiplicities) |
| **DmaNext per-limb binding** | 2 bus elements per 64-bit value, aligned pairwise | `lookup.rs` `num_bus_elements` + `accumulate_fingerprint_with` | audit **§G** (14 claims). Was assumed and unverified; that is how R1 happened |
| A1 (was "MEMW-ADDR32") | address limbs `< 2^32` wherever a `Memw` data op fires (`mu − end == 1`) — this is what the *gate* asserts; the head row is where it is load-bearing, since non-head limbs are also derived through the link | `spec/src/memw.toml` `[[assumptions]] IS_WORD[base_address[i]]` — a **caller obligation**, not a receiver check | **not discharged** — see §5 |
| A2 (was "REG-32") | the **head row's** count limbs are 32-bit | `spec/src/memw_register.toml` `[[assumptions]] IS_WORD[val[i]]` | **not discharged** — see §5 |
| constraint set = 18, degree 2 | 18 modelled constraints, all quadratic | `DmaConstraints` | audit C + the Rust test `dma_constraints_count_and_indices` |
| the Rust fixture is the oracle's output | `include_str!` of the emitted row table | the emitter in `test_oracle.py` | audit **§H** |
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

Ten negative controls, each dropping exactly one premise and re-running the check
that premise is load-bearing for. All ten are SAT. (`Premises.NAMES` has eleven
entries; `memw_addr32` has no control **deliberately** — see §5.)

**Pairing matters, and getting it wrong is silent.** Three of the original eight
were initially paired with the wrong check and reported UNSAT — a control that
cannot fail:

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
| `forged_early_end_rejected` | `END := 1` on a data row | the `Zero` lookup (the sum no longer reads zero) **and** the three sends gated on `mu − end`, which vanish — so `DmaNext` and both `Memw` buses unbalance too. Not the `Zero` bus alone, as an earlier version of this table said | MAIN 1, and its `drop_zero_end` / `drop_halfword_count_decr` controls |
| `forged_wide_tail_rejected` | `TAIL := 1` on a wide row | **overdetermined — at least five independent mechanisms reject it.** Row-locally: `step = 8 − 7·tail` breaks the *ungated* idx-9 `emit_add_pair` on `count` (`count_decr + 1 ≠ count` when the trace holds `count − 8`), idx 5-6 and 7-8 fail identically, and idx 11-17 (`tail·value[i] = 0`) fail whenever the eight copied bytes are not all zero. On the buses: the `Alu` width pin (bus 20, multiplicity `mu`) sends `[count, 8, 0, LT, TAIL, 0]`, so with `TAIL = 1` on a `count ≥ 8` row it asks `lt.rs` for output 1 where that table holds 0 — no matching row, `Alu` unbalances; and `w8 = 1 − tail` changes the `Memw` width. | MAIN 0 |
| `forged_intermediate_source_rejected` | `SRC_0` **and** `SRC_INCR_0` shifted together | **nothing row-local** — the row's own ADD stays satisfied. The predecessor's `DmaNext` tuple no longer matches, and the source read no longer matches memory | CHAIN / CHAIN-F, which is exactly the check that treats `DmaNext` as a free bijection rather than an assumed chain |
| `forged_value_rejected` | `VALUE[0]` | **not the copy relation** — read and write still agree with each other, because they are one set of columns. What rejects it is the `Memw` read no longer matching memory | none; audit §D pins the one-set-of-columns wiring |

**A correction worth keeping, because it is instructive.** An earlier version of
this table credited the `Alu` width pin alone; a review called that incomplete,
and the replacement over-corrected into *"**Not** the `Alu` width pin"* — which is
false. The pin does reject it, by the argument now in the cell. The chain was:
a finder wrote "the Alu lookup is not what blocks it", that was accepted without
checking, and it was then sharpened into an explicit negation. **An overstatement
became a falsehood by being propagated.** The lesson generalises past this cell:
when a mechanism is overdetermined, "X rejects it" and "Y rejects it" are both
true, and the tempting edit — replacing one with the other — is the one that
introduces an error. Prefer "at least these", never "not that".

Two rows are worth dwelling on. `forged_intermediate_source_rejected` is the
case where per-row soundness is genuinely insufficient and the chain argument is
doing the work — which is why the gate builds the bijection model rather than
assuming rows are chained. And `forged_value_rejected` passes for a reason no
solver query establishes: the only thing standing behind it is a textual fact
about how two bus tuples are constructed. That asymmetry is the reason the gate
and the audit script are separate artifacts.

## §5 — Premises NOT discharged here (and where they should be)

**A1** and **A2** (`dma-chip/DESIGN.md` §Assumptions) are asserted by the gate and
verified by nothing in this directory. Their control coverage is **asymmetric**, and
saying so matters: `drop_reg32` flips `check_row_budget` to SAT, so A2 is
demonstrably load-bearing, but **A1 has no control on purpose** — dropping
`memw_addr32` leaves every check on the board unchanged, because the limb-wise
`DmaNext` link derives well-formedness from the sender's `IsHalfword` checks
instead. A control that cannot fail is worse than no control, so none was added;
the gate's docstring previously claimed "every negative control shows what breaks
without them", which was false for exactly this premise. Both are **obligations on the caller** that
the spec states as such — `spec/src/memw.toml` and `spec/src/memw_register.toml`
carry them as `[[assumptions]] IS_WORD[...]`, and `memw_register.rs`'s only
range-check interaction is on the timestamp delta.

An earlier version of this campaign had the **direction backwards**, presenting
them in `DESIGN.md`'s column table as what `src`/`dst`/`count` are "range-checked
by", under invented labels "MEMW-ADDR32" and "REG-32". `memw.rs:257-262` is
explicit that its own bound holds because *the CPU table* splits addresses with
both halves in `[0, 2^32)` — and DMA is a non-CPU sender, so that argument does
not extend to it.

They bind the **head row only.** Every other row's limbs are derived through the
`DmaNext` per-limb binding (§2, audit §G). Worth recording as a lesson in where to
look: the retracted R1 reported a gap on non-head rows, where the bus in fact pins
the limbs, while the genuine obligation sits on the head row — the one row with no
`DmaNext` receive at all. The phantom and the real premise were exact mirror
images, and the phantom is the one that got written up.

Neither is DMA-specific, and the underlying problem is the spec's:

> **`IS_WORD` has no discharger.** It appears across ~10 chip TOMLs
> **exclusively** inside `[[assumptions]]` blocks — never as `kind = "interaction"`
> or `kind = "template"` — and `spec/bitwise.typ` offers only MSB8/MSB16/ZERO/
> ARE_BYTES/IS_HALF/IS_B20, a 2^32 table being infeasible. So the spec asserts a
> range obligation for essentially every address, register value and timestamp in
> the VM without saying how it is met or which chip owns it per sender. **That
> vacuum is what this campaign filled by inventing two label names, and the next
> chip author will fill it the same way.** It wants either a
> `= Range-check provenance` chapter or a per-sender naming requirement.

Related and equally ownerless: `spec/memw.typ:42-45` concedes that `value` range
checks "are necessary for the consistency of the system as a whole" and documents
the types "as a reading help". DMA is a concrete sender for which no chip
discharges it. An obligation owned by everyone is owned by nobody.

## §6 — The retracted finding, and what it cost

This section used to be a finding. It is now the account of an error, kept because
the error is more instructive than the finding would have been.

**What was claimed.** That `DmaNext` "equates packed field elements", so `count`'s
limb split was unconstrained on non-head rows, and a Goldilocks alias
(`COUNT_1 = 2^32−1`, `COUNT_0 = V+1`, packed ≡ V mod p) would pass the chain while
`lt.rs` saw an integer near `2^64` and returned `tail = 0` — a row claiming an
eight-byte width where fewer than eight bytes remained, blocked only by trace
length. Two fixes were recommended.

**Why it was wrong.** `Packing::num_bus_elements()` returns **2** for both
`DWordWL` ("2× Direct") and `DWordHL` ("2× Word2L"), and
`accumulate_fingerprint_with` gives each element its own alpha power. No `Packing`
variant contains a `2^32` shift, so a 64-bit value is never one bus element
anywhere in this codebase. Both `DmaNext` tuples are `1+1+2+2+2 = 8` elements and
align pairwise, so balance forces `COUNT_0 = cd₀ + 2^16·cd₁` **and**
`COUNT_1 = cd₂ + 2^16·cd₃`. With the sender's four halfwords `IsHalfword`-checked,
the receiver's limbs are 32-bit for free. Under the corrected link the alias is
UNSAT and the successor's count is pinned exactly.

**Four things to take from it.**

1. **Classify the direction of every modelling gap.** A model *weaker* than the
   AIR yields false alarms but never false proofs — every UNSAT on the board
   survived the correction, having been proven under weaker hypotheses than
   reality supplies. A model *stronger* than the AIR is the one that yields a
   false proof no positive anchor can catch.
2. **A phantom finding causes real damage.** Working around R1 led to asserting
   `count ≤ 256` on *every* row of the field-exact chain check when the AIR bounds
   only the head — the one genuinely over-strong assertion in the gate, i.e. the
   dangerous direction, introduced to accommodate something that did not exist.
3. **A proposed fix that is a no-op means the gap is not there.** R1's second fix
   was "receive `count` as `DWordHL`", which changes nothing under the real
   semantics. That should have stopped the write-up.
4. **The audit's coverage gap is the root cause, not the gate's model.** §D pinned
   the tuples' column names and multiplicities and never their packing semantics,
   so "83 claims, 0 findings" never tested the one fact R1 rested on. §G exists
   now, and the `num_bus_elements(DWordHL) 2 → 1` mutant is in the regression set.

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
3. **The multi-call case.** Layer 2 proves the tiling among groups containing
   exactly one head row. `ChainRow` carries no timestamp, so two DMA calls in one
   trace are out of model; the `ts` carried in both `DmaNext` tuples is what
   separates them, and it is guarded only textually (audit §D).
4. **The `n = 0` ecall.** One row, both `first` and `end`, no `DmaNext` traffic,
   no memory operations. Pinned by the completeness sweep and by
   `empty_dma_call_is_a_single_first_and_terminal_row`, but it is the row shape
   most likely to be broken by a future multiplicity change, because every
   multiplicity on it is zero.
5. **Two ecalls at one timestamp.** Ruled out by CPU timestamps being strictly
   increasing per instruction. Asserted, not verified here, and it is what the
   `DmaNext` timestamp binding relies on.

## §8 — Reproducing

```sh
python3 docs/verification/dma/dma-oracle/test_oracle.py        # anchors + emit vectors
python3 docs/verification/dma/dma-chip/z3_dma_verify.py        # the gate (add --quick to shorten)
python3 docs/verification/dma/audit_gate_transcription.py      # this audit
cargo test -p lambda-vm-prover --lib dma              # the Rust side, incl. the vector test
```

`make verify-dma` runs all three (it tolerates the oracle's exit 2, which means
"ran but degraded", and aborts only on exit 1). `z3-solver` is the only dependency
(`pip install z3-solver`; validated on 5.0.0 — the audit alone needs no solver).
The gate takes ~96 s for the full board on 5.0.0, dominated by the completeness
sweep; on 4.12.2 it takes ~1210 s and two queries blow their budgets and report
`unknown`, which is scored as **failure** everywhere. The oracle exits 2 when it
ran but an anchor skipped, so a degraded run is distinguishable from a clean one.
No CI workflow schedules any of this yet.
