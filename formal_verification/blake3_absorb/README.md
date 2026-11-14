# Formal verification of the BLAKE3 chained-absorb mode

Machine-checks the row-local constraints of the absorb mode added to
`prover/src/tables/blake3.rs` by commit `9cf6c352` — the second syscall the
BLAKE3 table answers, in which one ecall folds `num_blocks` 64-byte blocks and
occupies `num_blocks + 1` rows.

Method, file layout and the Mandatory-discipline checklist come from
`formal_verification/keccak/README.md`, which is the canonical template. The
compression core's own gate is the earlier
`thoughts/blake3/blake3-chip/z3_blake_verify.py` (#903); this directory is its
successor for the absorb mode and re-runs its G-level check so the board here
stands alone.

```
python3 test_ref.py            # anchor the reference outside this repo
python3 z3_absorb_verify.py    # the gate — 60 queries, ~25 s
make test-blake3-absorb-fv     # both, from the repo root
```

Only dependency is z3's Python bindings (`pip install z3-solver`). No cargo, no
repo build.

---

## ⚠ The binding scope rule: one round at a time

**The 6 rounds are never composed in one query, because that query does not
close.** This is not a preference, it is measured. On 2026-08-06 the
predecessor gate's `--full` board ran 145 minutes and returned `unknown` — z3's
timeout answer — for all four of its composed queries (one round, rounds=2,
rounds=6, rounds=7). Its verdict line tested `== unsat`, so four timeouts
scored as four failures and the board printed `FAIL`. Commit `89aeeb8c`
recorded the outcome as ATTEMPTED-INCONCLUSIVE.

**Re-examined here: that was a scope artifact, not a finding.** ✓ VERIFIED by
reading `z3_blake_verify.py:309-344` (the four checks return `check()` directly
after `s.set("timeout", …)`) against `89aeeb8c`'s own message ("all four
monolithic queries returned `unknown`, not `sat`… The four budgets sum to 140
min against ~145 min wall"). Nothing was disproven; each check simply spent its
whole allowance. `89aeeb8c` itself names the remedy this directory implements:
"restructure the monolithic query as round-by-round induction rather than one
flat bit-vector problem."

So each round is factored into two queries that close in under a second:

| level | query | what it decides |
|---|---|---|
| **P1b** | the byte-level G circuit vs the spec G, **free inputs** | that a G computes BLAKE3's G. One query covers all 48 instances — the circuit text of a G is identical for every one, so 48 copies would re-decide the same formula. |
| **P1a** | round *r*'s **wiring**, with G an uninterpreted function | that round *r* hands each G the state slots `G_INDICES` names and the message words `permute^r` names. Six queries, `r = 0..5`, each with that round's concrete schedule indices. |

### What is a structural argument, and therefore NOT SMT-proved

Two compositions are stated here and carried as wiring arguments. Both are
checkable by reading; neither is claimed as a solver result.

1. **G-internal ∘ round-wiring.** The wiring query treats G as a black box; the
   G query proves the black box is the spec's G *for arbitrary inputs*.
   Substituting the second into the first yields the round. Sound precisely
   because the G query quantifies over free inputs, so it holds at every
   instantiation.
2. **Round *r* → round *r+1*.** `run_flow` (`blake3.rs:342-373`) is one loop
   whose body writes `v[ia] = a2; v[ib] = b2; v[ic] = c2; v[id] = vd2` back
   into the array the next iteration reads. Round *r*'s output columns *are*
   round *r+1*'s input columns — there is no committed handoff to constrain and
   nothing for a prover to choose. ✓ VERIFIED by reading.

**If a single-round query ever fails to close, report the bound and stop. Do
not weaken the encoding to make it green.**

---

## Two theories, deliberately

The compression core is boolean/byte algebra, which **QF-BV** models exactly
(P1). Byte columns are 8-bit bitvectors — that *is* the `AreBytes`/`ByteAlu`
range-check contract, expressed as a sort.

The absorb mode's own constraints are not byte algebra. They are field
arithmetic over selector columns, a countdown, and packed limbs, and every
attack they defend against is field-level: `256`, `2^16` and `2^32` are
**invertible** mod the Goldilocks prime while they are zero divisors mod `2^n`.
A bitvector model of those constraints would report UNSAT for free and bless a
chip that is forgeable in the field. **P2–P5 therefore run in integer
arithmetic mod p**, every congruence linearized by a bounded quotient. This is
the same split the predecessor gate's `WIDTH AUDIT` section made, and
discipline #3 of the keccak template.

Two modeling steps are supplied to the solver rather than derived, and are
recorded here because they are the model's trust boundary:

- **`IS_BIT` as a disjunction.** The circuit emits `x·(1−x) = 0`. Since p is
  prime and committed values lie in `[0,p)`, that is *equivalent* to
  `x ∈ {0,1}`, and the model asserts the disjunction. z3 cannot derive this —
  it does not know p is prime. (Proof: `p | x(1−x) ⟹ p|x or p|(1−x)`.)
- **Quotient bounds.** `expr ≡ 0 (mod p)` is encoded as `expr = k·p` with `k`
  bounded. ⚠ **A bound computed for the contract-present case will silently
  re-impose the contract when that contract is dropped**, turning a negative
  control falsely UNSAT. `pointer_add` carries an explicit comment about this;
  it is the reason its bound is `±2^17·p` and not `±2·p`.

---

## The board

60 queries, ~25 s wall on an M-series laptop, z3 4.15.4. `P*-neg` rows are
negative controls and **must** be SAT.

| # | property | verdict |
|---|---|---|
| **P1a** | round *r* wiring, `r = 0..5` (6 queries) | PROVED |
| **P1b** | G circuit vs spec G, free inputs | PROVED |
| **P1c** | input feed under the absorb framing: `IV \| t=0 \| block_len=64 \| flags` | PROVED |
| **P1d** | feed-forward, and that the chain payload is the low 8 words | PROVED |
| **P2.1-2** | an interior row's counter bytes are zero and `block_len` bytes are exactly `(64,0,0,0)` | PROVED |
| **P2.3** | ★ an interior row cannot carry **any** flag byte — the forged-shorter-message class | PROVED |
| **P2.4** | a FIRST row's flags word is `< 2^32` | PROVED |
| **P3.1** | exactly **five** row shapes exist; there is no sixth | PROVED |
| **P3.2** | every derived multiplicity — `MU_C`, `MU − END`, `MU_A − FIRST`, `MU_S + FIRST` — lies in `{0,1}` | PROVED |
| **P3.3-4** | no cross-mode bleed, in both directions | PROVED |
| **P3.5** | `MU_S·MU_A = 0` is **implied** by `IS_BIT(MU)` + the partition (informational) | PROVED |
| **P4.1** | a FIRST row's `REMAINING` is in `1..=1024` — the cap, in circuit | PROVED |
| **P4.2-3** | END is neither early nor late: `END = 1 ⟺ REMAINING = 0` | PROVED |
| **P4.4-5** | the END row is inert (`MU_C = 0`); a compressing row decrements by one | PROVED |
| **P4.6** | ★ the cap cannot be wrapped mod p | PROVED |
| **P5.1** | the END row's `cv_out` bytes are **unique** given the chain-delivered words | PROVED |
| **P5.2** | `M_BASE_INCR` is a function of `M_BASE` | PROVED |
| **P6.1-2** | a FIRST row's `M_BASE[0] ≡ 0 (mod 8)` — x11 is 8-aligned — with the model shown consistent | PROVED |
| **P6.3** | a NON-final block's successor address cannot leave the address space | PROVED |
| **P6.4** | …but the FINAL block's may, so a message ending at `2^64` stays provable | SAT (intended) |

No counterexample was found against any shipped constraint. Every negative
control flipped to SAT.

**P6 was added by the F1 fix pass**, after the adversarial review demonstrated
two absorbs the chip accepted and the executor rejects: an unaligned `x11`, and
a message region wrapping the address space mid-group.

### ★ The witness that changed the fix

`P6-neg: the SCALED form IsB20[Q·2^7] admits an odd base`. The review sketched
the alignment check as `M_BASE[0] = 8·Q` bounded by `IsB20[Q · 2^7]`, mirroring
the block cap's scaling trick. It is **vacuous**: with `Q` bounded only through
the scaled product a prover takes `Q = M_BASE[0] · 8⁻¹ mod p`, and then
`Q · 2^7 = M_BASE[0] · 16`, under `2^20` for ANY halfword base, aligned or not.
The shipped chip bounds `Q` directly with `IsHalfword[Q]` instead, which forces
the congruence to be an integer equation and so forces `8 | M_BASE[0]`.

The block cap survives the identical attack only because `REMAINING`'s domain
bound arrives first from the ZERO lookup — the same reason P4.6 needs
`zero_domain` present. Modelling the alignment equation as an integer equality
rather than a field congruence hides all of this: no integer `Q` divides an odd
base, so the vacuous variant reports UNSAT and looks safe. That trap is why
`msg_addressing` uses `cong0`.

### Two witnesses worth reading

**`drop zero_domain` (P4-neg).** Dropping only the *domain* half of the
`Zero[REMAINING]` contract — keeping `END = (REMAINING == 0)` — lets a FIRST
row claim `REMAINING = 18014398505288705` and still satisfy the `IsB20` cap,
because `REM_DECR · 2^10 ≡ 2^20 − 1 (mod p)`. The cap lookup alone does not
bound the block count; it bounds it **only in combination with** the `Zero`
lookup's `< 2^20` domain. The chip has both, and `blake3.rs:1459-1462` says so;
this makes the dependency machine-checked rather than asserted.

**`drop byte_range_framing` (P5-neg).** With the `block_len` bytes' range check
removed, `block_len_b1 = 13835058052060938241` satisfies
`MU_C·(word − 64) = 0` while the byte fed to the mixing core is nowhere near
`64`. The word-level constraint means what it says *only* because those four
bytes are `ByteAlu` XOR operands.

---

## Width census — "the chip assumes there are bytes"

Every column an absorb row consumes at a width, and the contract that pins that
width on the row shape where it is consumed. ✓ VERIFIED by reading each cited
construct.

| columns | consumed on | pinned by | site |
|---|---|---|---|
| `h`/`cv_in` bytes (32) | compressing rows | `ByteAlu[XOR]` operands (mixing core, gated `MU − END`) | `blake3.rs:1196-1209` |
| `h` bytes | **END row** | `AreBytes`, END-gated — the one row whose core is off | `blake3.rs:1245-1251` |
| `m` bytes (64) | compressing rows | `AreBytes` (m is never XORed) | `blake3.rs:1226-1237` |
| `t_lo`,`t_hi`,`block_len`,`flags` (16) | compressing rows | `ByteAlu[XOR]` operands: they are `v[12..16]`, the `d` of round 0's G-calls 0..3 | `blake3.rs:334, 350` |
| `OUT` bytes (64) | compressing rows | `ByteAlu[XOR]` **outputs** (feed-forward) | `blake3.rs:1196-1209` |
| `OLD_OUT` (64), `ADDR` (8) | `MU` | `AreBytes` (+ AND alignment on `ADDR`) | `blake3.rs:1256-1302` |
| `PTR[0..8]` halfwords | `MU` | `IsHalfword` | `blake3.rs:1308-1320` |
| `M_BASE` halfwords | `MU_A`, incl. END (it rides the chain) | `IsHalfword` | `blake3.rs:1486-1492` |
| `M_BASE_INCR`, `msg_ptr[0..8]` halfwords | `MU_C` | `IsHalfword` | `blake3.rs:1493-1508` |
| `REMAINING` | `MU_A` | `Zero` domain `< 2^20`; x12's `hi32 = 0` constant caps the high half | `blake3.rs:1463-1467, 1385` |
| `REM_DECR` | FIRST | `IsB20[REM_DECR·2^10]` ⟹ `< 2^10`; on later rows pinned by the countdown | `blake3.rs:1474-1481` |
| `MU_S`,`MU_A`,`FIRST`,`END` | all | `IS_BIT` | `blake3.rs:1730-1733` |
| `MU_C` | all | **derived**, `MU_C = MU_A − END`; that it is a bit is theorem **P3.2**, not a constraint | `blake3.rs:1755-1759` |

**Free-but-unconsumed on the END row — audited, not gaps.** The END row leaves
`m`, `t_lo`/`t_hi`/`block_len`/`flags`, and `OUT` unconstrained, because their
gates (`MU_C`, `MU − END`, `MU_C − FIRST`) are all zero there. None rides a bus
on that row: the message reads are `MU_C`-gated, the x13 read is `FIRST`-gated
and `FIRST·END = 0`, the chain send is `MU_C`-gated, and `cv_out` writes the
`h` columns — not `OUT` (`blake3.rs:1414-1424`). A free column that no
interaction reads cannot forge anything.

---

## Typed contract library (assume-guarantee)

Each helper lookup is modeled by its contract, never its implementation
(`bitwise.rs:756-830`). Each is a separately verified, fully enumerated
preprocessed table.

| contract | guarantee modeled |
|---|---|
| `ByteAlu(op,a,b,c)` | `a,b ∈ [0,256)` and `c = a op b`. Operands are byte range-checked **by** the lookup — the table has only byte rows. |
| `AreBytes(a,b)` | `a,b ∈ [0,256)`. |
| `Zero(v) -> z` | `v ∈ [0,2^20)` **and** `z = 1 iff v = 0`. ★ Both halves are load-bearing; see the P4 witness. |
| `IsB20(v)` | `v ∈ [0,2^20)`. |
| `IsHalfword(v)` | `v ∈ [0,2^16)`. |
| Memw register read | the value's `lo32` limb is a 32-bit limb; the `hi32` slot is the **constant** `0` in the tuple (`blake3.rs:1385,1394`). |

---

## Out of scope — do not read this gate as covering these

- **Bus telescoping / multiset balance of the `Blake3Absorb` chain**: that a
  group's rows form one chain, that every send has a receiver, that two groups
  cannot interleave. Every claim above is **row-local**. Owned by the in-tree
  falsification suite (`blake3.rs` tests `a_tampered_chained_cv_unbalances_the_chain`,
  `a_group_without_its_end_row_leaves_a_dangling_send`,
  `an_early_end_sends_a_zero_tuple_that_does_not_exist`) and by the `rev-absorb`
  review lane.
- **MEMW ordering / the memory argument** — that a read returns what was written.
- **Fiat-Shamir, the transcript, and everything above the AIR.**
- **The helper chips themselves** — assumed via the contract table.
- **The 6-round variant's collision resistance**, which is a named assumption
  (`thoughts/blake3/blake3-chip/DESIGN.md`), not a theorem.

## What a green board still does not show

Carried forward from the keccak template's discipline #1, and true here:

- **The model is a hand transcription.** z3 never sees the Rust. Faithfulness
  is a human obligation; the long-term fix is to generate the model from the
  constraint IR (`prover/src/bin/compute_constraint_artifacts.rs`) instead.
  Today that binary emits `.bin` artifacts for `production_airs` only, and
  **BLAKE3 is not in that set** (it is a chip-group table), so there is nothing
  to diff against yet. The transcription is pinned instead by the in-repo shape
  test `the_tables_shape_is_pinned` (3,266 columns / 1,473 interactions / 848
  constraints) and by the ledger comment at `blake3.rs:1517-1544`, which is the
  authoritative index map the model follows.
- **A constraint carried by a variable's sort is outside the falsifiable set.**
  In P1 the byte range checks are the 8-bit BV sort, so they cannot be dropped
  from the model — deleting them from the Rust would leave P1 green. That is
  why the same bounds are re-checked *removably* in the P5 field model, where
  dropping them does flip the board.
- **Fail-open is the dangerous mode.** A gate that is green for the wrong
  reason silently blesses an unsound chip. One such bug was found and fixed
  during this gate's development: two "independent" models in the uniqueness
  queries were built from identically-named z3 variables, which alias to the
  same variable and made the query trivially UNSAT. The `distinct_names` guard
  now fails the run rather than printing green.

---

## Files

- `z3_absorb_verify.py` — the gate: the field model of the absorb constraints,
  the QF-BV model of the compression core, all five properties, the negative
  controls, and the positive controls.
- `blake3_ref.py` — the independent reference, written from the spec over two
  backends (concrete ints for anchoring, z3 bitvectors for the gate).
- `test_ref.py` — anchors the reference outside this repo: the IV against
  `frac(sqrt(p))` for the first 8 primes, the 6-round compression against the
  10 recorded canonical vectors, and `absorb()` against chained `compress`.
