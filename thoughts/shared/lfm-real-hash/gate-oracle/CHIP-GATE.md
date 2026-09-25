# Gating the BLAKE3 socket chip AS BUILT

> ## ⟳ RE-GATED FOR THE D1 FIX — 2026-08-11. **VERDICT: PASS, 86/86.**
>
> Fourth re-gate. **The pin caught this one against an explicit assertion that it
> was out of scope** — the D1 fix was believed to be `chips.rs`-only, but the fix
> is a *shared* `emit_unread_input_pins` and the BLAKE3 arm calls it too, so
> `eval` and `framing_consts` both drifted and the 84/84 verdict did not carry.
> A verdict nearly shipped on a stale board; the instrument is what stopped it.
>
> **What moved:** the unread-input pins went from 4 (one cell) to **8 (both
> cells)**, `LEAF_IDX` shifted 30 → **34**, `CORE_IDX` → 50, `NUM_CONSTRAINTS`
> 942 → **946** @7r. **No cell counts moved** — 5,509 @7r / 4,741 @6r unchanged,
> so `program_id`s stay byte-identical, as the build reported.
>
> | region | sha256 | |
> |---|---|---|
> | `eval` | `240619f1580493b3998ac9fbc86aec61352132219ce5ca87c642e6ec2099a6be` | moved |
> | `framing_consts` | `21ab1892612cfd3b815c3d3c983a4baedd842de9f4d1b9268bdbc78843a26c24` | moved |
> | `bitwise_interactions` | `c880036158518796d36e097f3a676a197c5a707aec94ea1284d196dca3a46fa4` | unchanged |
> | `cols` | `f370814ae32795fe6366dbba7956f4e38bfb33681d810c0000bd3e5c799edf44` | unchanged |
>
> whole file `9d358f7bb3e2457065a473d478542aa7d218826e6ce81d96c2652d846f7a4cf6`.
>
> **Two new rows — §4.8.** One is an audit; the other is a *blindness*, and the
> blindness is the more important of the two.


**Status:** the seam is closed. The gate now certifies the chip that exists, not
the pre-Phase-2 model. **Date:** 2026-08-10. **VERDICT: PASS, 75/75.**

Companion to `ORACLE.md`, which describes the oracle itself (reference `f`, the
Option-A socket, the contract library, the column-role map). This document
records what changed when the real constraint bodies were transcribed in, and
what the board says about them.

No cargo was run. Everything here is python + z3 4.15.4.

---

## 1. WHICH ARTIFACT WAS GATED

### 1.1 ⚠ My first pin had a fail-open, and it fired

`artifact_pin.py` v1 hashed three regions (`eval`, `bitwise_interactions`,
`cols`) and recorded constants as their **expression text**. Asked to re-verify
after the implementer's wave, it answered **"artifact matches the pin"** — a
**FALSE PASS**, for two reasons:

1. `SOCKET_ROUNDS` changed definition (it is now an alias of `BLAKE3_ROUNDS`).
   v1 recorded `NUM_G = "SOCKET_ROUNDS * 8"`, which is stable under exactly that
   change. **Hashing an expression is not hashing a value.**
2. Worse: `SOCKET_ROUNDS`, `TAG_LFMC`, `FLAGS_LFMC`, `BLOCK_LEN_LFMC`,
   `COUNTER_LFMC` and `OUT_WINDOW` are top-level constants living in **none** of
   the three hashed regions. They are precisely the framing degrees of freedom
   the negative-control board tests. **A change of `FLAGS_LFMC` from `0x0B` to
   anything else — a live control (`flags_parent`) — would have passed silently.**

The change that exposed this was benign. The hole was not. A drift detector that
answers PASS without looking at the thing that matters is worse than none,
because it is trusted. This is the same fail-open class the whole gate is built
to prevent, and I had reproduced it in my own instrument.

**Fixed in v2**, which now (a) hashes a fourth region — the top-level constant
block — and (b) **resolves the framing constants and checks them against
`socket_ref.py`'s specification**, so the pin answers *"does the chip still
compute the socket the oracle specifies?"* rather than *"has this text changed?"*.
Every extraction is mandatory: a constant it cannot find is a hard failure, never
a silent skip. `artifact_pin.py` refuses to pin at all if conformance fails.

### 1.2 What the file actually is, and when

**COMMIT ANCHORS.** Phase 2 is now committed to `blake3-real-hash`:

| commit | what |
|---|---|
| **`b693eece`** | `feat(lfm): BLAKE3 as a first-class LFM_HASH hasher (compress socket, 7-round default)` — **the gated code** |
| **`cece4a0b`** | `docs(lfm): record the O5 decision — future leaf hashing uses the LFML tag` — docs only |

✓ EXECUTED **2026-08-11**: `python3 artifact_pin.py --check` against the
committed file → *"artifact matches the pin AND its framing still equals the
oracle spec; the gate verdict applies"*, noting the whole-file hash moved
`9d91954d…` → `b3d2755d…` **outside the four hashed regions only**. So the
board's **PASS 75/75 covers the committed artifact**, not merely the worktree
copy I transcribed from.

The narrative below is retained because it is how the pin came to be trusted;
it describes the same content, before it was committed. Base commit at
transcription time was `65025095`, file `prover/src/lfm/blake3_socket.rs`.

**Timeline, checked rather than assumed.** The file's mtime is `21:09:50`; I
pinned at `21:18` and ran the board at `21:25`. The wave the lead flagged — the
O5 module-doc note and the `SOCKET_ROUNDS` alias — is therefore the
`a03211d9… → 9d91954d…` change that landed **before** the pin and which the
gated transcription already reflects. I re-read the constraint bodies from the
file as it now is, per the instruction, and re-derived the pin from scratch.

**The file then moved a third time, during the board run** —
`9d91954d…` → `fd19f4c5…`. The v2 pin's verdict on that:

```
artifact matches the pin AND its framing still equals the oracle spec;
the gate verdict applies
  (whole-file hash moved 9d91954dd243 -> fd19f4c55d4b, but only outside
   the four hashed regions -- i.e. in comments/docs)
```

That is the instrument working as intended, and a live demonstration of why v2
was needed: it distinguishes *"the prose moved"* from *"the semantics moved"*
and says which. v1 would have answered PASS here too — but for the wrong reason,
having never looked at the framing constants at all. Concurrent editing of an
uncommitted file is the normal condition for this task, not an anomaly, so the
pin has to be the thing that carries the claim.

| region (normalized: comments + whitespace stripped) | sha256 |
|---|---|
| `pub fn eval<B: ConstraintBuilder…>` | `0441de9b71229ef5000c4a19f7d50273eb9abb826f38e7e8b3a22ec3ca1a2650` |
| `pub fn bitwise_interactions()` | `b49ff66c374161b7acd3742d03ba2fc969f1fa2f26efdc9659b4e8d7a81ba94a` |
| `pub mod cols` | `fac9bd6634cfc29e607b37a4c1f49e9b89c556431214c2afd6710e32b0f56b70` |
| **framing constants (v2, new)** | `64e9babf31afc7e2a9850324d4fa0a680a0d53adf6fc978412fe12ee2aa972b9` |

### 1.3 Framing conformance — resolved values vs the oracle spec

✓ EXECUTED. The chip's constants are resolved from source and compared against
`socket_ref.py`. This is the check v1 lacked entirely:

| chip constant | resolved | oracle spec | |
|---|---|---|:--:|
| `TAG_LFMC` | `"LFMC"` = `0x434D464C` | `0x434D464C` | ✓ |
| `FLAGS_LFMC` | `0x0B` | `CHUNK_START\|CHUNK_END\|ROOT` | ✓ |
| `BLOCK_LEN_LFMC` | `36` | `36` | ✓ |
| `COUNTER_LFMC` | `0` | `0` | ✓ |
| `OUT_WINDOW` | `HASH_DIGEST_FELTS` | low 4 of 16 words | ✓ |
| `NUM_LANES` | `8` | 2 cells × 4 lanes | ✓ |
| `G_SIZE` | `60` | the gated per-G cell count | ✓ |
| `FLOW.full_output` | `false` | requirement R3 | ✓ |
| `NUM_G` | `SOCKET_ROUNDS * 8` | 8 G-calls per round | ✓ |
| `SOCKET_ROUNDS` | `BLAKE3_ROUNDS` → **7** default, **6** under `blake3-6round` | the gated pair {6,7} | ✓ |

The round-count alias is the one semantic item in the wave. It is benign **for
this gate specifically** because the board covers *both* reachable values; the
chip compiles to exactly one. The pin now fails if that pair ever stops being
`{6, 7}`, since the board would then be certifying a round count the chip does
not use.

> **If any region hash or framing value changes, this verdict does not carry
> over.** `python3 artifact_pin.py --check` is the one-command test.

## 2. What changed in the seam

### 2.1 `emit_add2` — the deviation, closed

The pre-Phase-2 model witnessed the add2 carry as a **column** and constrained
it twice (sum identity, degree 2; booleanity, degree 3). The chip derives it as
an **expression**, `carry := (A + B − s)·2^{−32}`, and emits **one** constraint:

```
MU · carry · (1 − carry) = 0                    blake3_socket.rs, add2 loop
```

The model now does the same. It is the same statement — the model's pair asserts
*∃ carry ∈ {0,1} with A + B = s + 2^32·carry*, the chip eliminates an existential
whose witness is determined — but the gate must certify **the chip that exists,
not a stronger cousin**, so the model follows the chip.

**Modelling note, and the reason WA7 exists.** `2^{−32}` is a *field* inverse
with no faithful BV counterpart. The BV domain therefore encodes the
*post-audit* statement — the difference lies in `{0, 2^32}` — and the side
condition that those are the only reachable roots is discharged in the field by
WA7. Encoding that disjunction in BV *without* the audit would be assuming
precisely the thing that makes the form sound.

### 2.2 BLOCK 0 — the four framing constraints the model did not cover

All four are over felts and mode selectors, not bytes. **Decision: they go to the
FIELD/structural ledger, not BV** — a BV model has no faithful representation of
a Goldilocks mode selector, and pretending otherwise would be the fail-open this
whole split exists to prevent. All four are now *checked* in the field, both
ways, rather than merely asserted:

| chip idx | constraint | where it is checked |
|---|---|---|
| 0–3 | `S_k − (MODE_P·IN_{8+k} + MODE_C·IV_k)` | **B0a**, field |
| 4 | `mode_sum·(1 − mode_sum)`, `mode_sum = MODE_C + MODE_P` | **B0b**, field |
| 5 | `MODE_P = 0` | **B0a/B0b**, field |
| 14–21 | `OUT_{4+j} = 0`, j ∈ 0..8 | **B0c**, field |

All four are **ungated** (no `MU` factor), which is correct: they must hold on
padding rows too, and padding is all-zero.

**The finding worth having.** idx 0–3 pin nothing on their own — they only
constrain the capacity prefix because **idx 5 kills the `MODE_P` term**. Drop
idx 5 and `MODE_P` is free, so the prefix becomes a prover-chosen copy of
`IN_{8+k}`. ✓ EXECUTED both ways (B0a): with the pin → UNSAT, without → SAT.
`MODE_P` being *preprocessed* is the deeper defence, but idx 5 is what makes the
capacity family mean anything, and the two should not be confused.

Also: the model previously deferred MU booleanity to "structural, not a BV
theorem". The chip emits it as a real constraint, so it is now checked (B0b) —
`ORACLE.md`'s claim that it is not checkable is superseded.

### 2.3 Census prefix

13 → **28**, the frozen shared prefix as built (12 `IN` + 4 `S` + 12 `OUT`).
`MU = MODE_C` is *preprocessed*, so it is outside the main-column census
entirely — and, more importantly, a prover cannot choose it.

---

## 3. Census reconciliation — exact, to the unit

✓ EXECUTED. The model's census is derived from the gated constraints, so this is
a real cross-check, not a restatement:

| | model | built chip | |
|---|---:|---:|:--:|
| main columns, 6r | 2,956 | 2,956 | ✓ |
| main columns, 7r | 3,436 | 3,436 | ✓ |
| sends, 6r | 1,190 | 1,190 | ✓ |
| sends, 7r | 1,382 | 1,382 | ✓ |
| cell-equiv, 6r | 4,741 | 4,741 | ✓ |
| cell-equiv, 7r | 5,509 | 5,509 | ✓ |

Both deltas the report predicted are accounted for exactly: **−96 (6r) / −112
(7r)** from dropping the add2 carry column (2 per G × `NUM_G`), and **+15** from
the 28-column prefix replacing the model's 13. Per-G block is now **60 cells**,
matching `cols::G_SIZE = 60`.

7r blocks: `rotr_shift` 1,344 · `xor_out` 912 · `add3` 672 · `add2` 448 ·
`lane_bytes` 32 · `frozen_socket_prefix` 28.

---

## 4. THE BOARD — 75 checks, VERDICT PASS

`python3 gate.py`, ~13 min. Full output in `run-chip-gate.log`.

| section | checks | all as wanted |
|---|---:|:--:|
| main theorems (symbolic BV) + per-theorem discrimination controls | 12 | ✓ |
| T4 full pipeline, concrete, vs anchored KATs (6r and 7r, both directions) | 4 | ✓ |
| negative controls (**both round counts**) | 36 | ✓ |
| documented BV blindness | 1 | ✓ |
| optional tail truncation | 2 | ✓ |
| non-vacuity | 1 | ✓ |
| width audit (field) | 13 | ✓ |
| BLOCK-0 framing audit (field) | 6 | ✓ |

### 4.1 Negative controls — re-measured against the transcribed bodies

**A transcription that accidentally strengthens is as wrong as one that weakens,
and only the controls can tell the difference.** Every control was therefore
re-run against the new bodies, at **both** round counts (the chip ships 7r by
default and 6r behind `blake3-6round`), against the **full concrete pipeline**:

`swap_a_b`, `tag_changed`, `tag_omitted`, `truncate_high_half`, `flags_parent`,
`flags_no_root`, `block_len_64`, `block_len_32`, `counter_one`, `cv_zero`,
`lanes_big_endian`, `tag_slot_moved`, `msg_perm_swapped`, round-count confusion
(`rounds_6_not_7` at 7r, `rounds_7_not_6` at 6r), `drop_ff_xor`,
`swap_g_operand`, `drop_add2_carry` — **all SAT at both round counts**, plus the
two symbolic G-level controls and nine per-theorem discrimination controls.

**New control for the new form: `drop_add2_carry`** — removes the add2
constraint outright. Under the expression-carry form there is no carry column
left to un-boolean, so the whole constraint *is* the booleanity, and unlike the
add3 case it is **BV-visible**. SAT at both round counts. Without this control
the new `emit_add2` would have had no test that its single constraint is
load-bearing at all.

### 4.2 The documented BV blindness, now sharper

`drop_carry_bool` (which un-booleans add3's carry **columns**) remains **UNSAT in
BV** — correct, and recorded. In BV a carry column is an 8-bit variable, so
removing its booleanity leaves it bounded and `s` stays pinned; the same bug is a
live forgery in the field (WA4 → SAT).

The distinction now matters more than before, because the two adds are
constrained differently: **add3's carries are columns (field-only bug), add2's
carry is an expression (BV-visible bug)**. Same chip, two bug classes, two
domains. A gate running only BV would report the add3 class as absent.

### 4.3 Width audit — 13 items, including the new WA7

| item | present | dropped |
|---|---|---|
| WA1 lane decomposition (obligation O1) | UNSAT | **SAT** |
| WA2 lane `< 2^32` forced | UNSAT | **SAT** |
| WA3 shift `SLL` tight bound | UNSAT | **SAT** |
| WA4 add3 carry booleanity | UNSAT | **SAT** |
| WA5 tail case: word value pinned / bytes not | UNSAT | **SAT** |
| WA6 no-wrap side condition (worst `2^34 ≪ p`) | ok | — |
| **WA7 add2 expression-carry pins `s`** | **UNSAT** | **SAT** |

**WA7** is the companion the expression-carry form needs. With `A`, `B`, `s`
byte-bounded below `2^32`, are `0` and `2^32` the only reachable roots — can a
*negative* difference alias `2^32 mod p`?

It cannot. If `A + B − s ≥ 0` it lies in `[0, 2^33)` and `2^33 ≪ p`, so the only
residues are the honest two. If `A + B − s < 0` it lies in `(−2^32, 0)`, i.e. the
field element sits in `(p − 2^32, p)`; that equals `0` only for a zero difference,
and equals `2^32` only if the difference were `2^32 − p ≈ −2^64`, far below
`−2^32`. Hence `s` is pinned to `(A + B) mod 2^32`. Dropping the byte bound on
`s` makes it a free field element and the add forgeable — **SAT**, the same class
as WA4 and equally invisible to BV.

### 4.4 The argued ledger — new, and deliberately visible

z3 4.15.4 has **no finite-field sort** (✓ VERIFIED — `FiniteFieldSort` does not
exist in this build), and the `Int`+`mod` encodings of the quadratic field facts
are nonlinear and intractable: the first attempt at WA7 and B0b hung the solver.

So four steps are discharged by **algebra, not by a solver**, and the board now
prints them rather than baking them in silently — an unstated assumption is
exactly how a fail-open happens:

| | fact | relied on by |
|---|---|---|
| **AR1** | `F_p` has no zero divisors, so `x·(1−x) = 0` has root set exactly `{0,1}` | WA4, B0b |
| **AR2** | `2^{−32}` is a unit, so `d·2^{−32} ∈ {0,1}` iff `d ∈ {0, 2^32}` | WA7 |
| **AR3** | `2^16` is invertible mod `p` | WA3 |
| **AR4** | every field-lifted expression stays below `2^34 ≪ p` | WA6 |

This is the posture the audit already took for WA4 (whose "present" case encodes
booleanity as a root set rather than asking z3 to derive it). Making it explicit
is the change; the solver is left on the questions it can actually decide.

---

## 4.5 ⚠ STANDING NOTE (D6) — the last-round diagonal-G `Y` columns are
underconstrained-but-unread. **Harmless as built. Do not "tighten" without re-gating.**

Independently found by F9 in review; it is the same surface as WA5 and O-TAIL,
recorded here because it is a **live constraint on future edits**, not a defect.

**What it is.** The chip does *not* take the tail-truncation option: the last
round emits all eight G-calls in full, including `X4` and the `rotr7` that
produces `B2` = `v[b]`. In the last round nothing reads those `v[b]` values —
the feed-forward reads `v[0..4]` and `v[8..12]`, and the diagonal group's
`b`-positions are `{5,6,7,4}`. So `B2`'s four `Y` byte columns are:

* **constrained** as a word — the two recombine identities pin
  `Y0 + 256·Y1` and `Y2 + 256·Y3`, hence `Σ Yₖ·2^{8k}`;
* **not** pinned per byte — nothing forces the split between `Y0` and `Y1`
  (an extra 8 bits of prover freedom per halfword);
* **read by nothing.**

**Why it is harmless.** An unread column cannot influence the digest. ✓ EXECUTED
in the field, both directions (WA5): the rotation's **word value** is still
pinned (UNSAT) while its **individual bytes** are not (SAT). Those two results
are exactly this note.

**The constraint on future work.** The safety rests on *unread*, not on
*constrained*. Two ways a later PR breaks it, both plausible-looking cleanups:

1. **Giving the columns a reader.** Any consumer that reads `Y`'s bytes
   individually — a byte relabel, a `ByteAlu` operand, any sub-combination
   rather than the full linear form — is **unsound** without an added
   `AreBytes`. The surviving reader in the non-last rounds (`add3`) is safe only
   because it reads `Σ Yₖ·2^{8k}`, which regroups exactly into the two
   constrained halfword sums.
2. **Deleting them as dead** (the tail-truncation optimisation). Legal, and
   ✓ EXECUTED as correct — but worth only **112 cell-equiv of 5,509 (2.0%)**,
   and it must drop `X4` **only** for the last round's *diagonal* group
   (`gi ≥ 4`). A column G's `v[b]` is consumed by the diagonal group that
   follows it in the same round; dropping that one is a bug, not an
   optimisation. My own first draft of the option made exactly that mistake and
   it was caught only because the option was exercised rather than described.

**If you change this surface, re-run `gate.py` and `artifact_pin.py --check`.**
Neither the region hashes nor the framing values would catch a *reader* being
added to a previously-unread column, because it is a change inside `eval` — the
region hash moves, which is the signal to re-transcribe and re-gate.

---

## 4.6 POST-B1 — the new audits, and the claim of mine they falsified

### 4.6.1 `m[8]` is no longer a constant, and transcribing it as one was the trap

The post-B1 chip computes `m[8] = MODE_C·TAG_LFMC + MODE_T·TAG_LFMT`
(`WordRef::ModeSelected`, evaluated `Σ col·tag`). Two documents I own —
`ORACLE.md` §2.1/§2.2 and `SOCKET.md` §2.2 — still described it as the constant
`0x434D464C`, and `ORACLE.md` justified its zero cost *because* it was constant.

**Those framing tables are exactly what this gate transcribes.** Transcribed as
written, the z3 model would carry a constant where the chip has a linear form —
a model that no longer checks the chip **and still reports PASS**. That is the
same fail-open class as the pin's v1, on the transcription side instead of the
identification side, and it would have been mine. Both rows are now corrected,
with the reason spelled out: `m[8]` is still free and still prover-unchosen, but
because the selectors are **preprocessed**, not because the value is constant.

Caught by the builder and relayed; recorded here so the lesson has a home.

### 4.6.2 ⚠ M8 — idx 4 does NOT make the tag one-hot. My spec said it did.

`TRANSCRIPT.md` §3.3 asserted *"idx 4 forces the mode sum to a bit, so at most
one tag is selected."* **The clause after "so" does not follow**, and the
consequence is not academic: a refactor trusting that sentence could delete the
registrar's one-hot check as redundant, and every constraint would still pass.

Over a prime field `mode_sum ∈ {0,1}` pins the SUM, not the selectors:
`MODE_C = x`, `MODE_T = 1 − x` satisfies idx 4 for any `x`, and since the tags
differ, `x = (T − TAG_LFMT)/(TAG_LFMC − TAG_LFMT)` reaches **any** target `T`.

✓ EXECUTED independently, twice — the builder's Rust M5/M6 run forges the tag
`"XXXX"` by a fractional split with zero constraint violations, and this board
reproduces it in the field model:

| | check | result |
|---|---|---|
| M8 | forged tag reachable with **idx 4 alone** | **SAT** — `MODE_C = 4387334679741772800`, `MODE_T = 14059409389672811522`, sum ≡ 1, `m[8] = 0x58585858` |
| M8 | forged tag **excluded** once one-hot is present | UNSAT |
| M8 | honest leg: `TAG_LFMC` still reachable | SAT |
| M8 | honest leg: `TAG_LFMT` still reachable | SAT |

Both honest legs are there deliberately: a "fix" that rejected everything would
pass the attack leg on its own.

**What actually closes it:** the selectors being **preprocessed** (the prover
cannot choose them at all), plus the **registrar's exactly-one-of check**. Idx 4
buys only the exclusion of the both-set case. `TRANSCRIPT.md` §3.3 is corrected
and M8 is now a standing control so the mistake cannot be re-made silently.

Related, and ✓ VERIFIED from the layout: this is also why `MODE_T` sits at index
**8**, inside the selector run, rather than after the multiplicities — the
admission validator reads the selectors as a contiguous span, so a selector
parked past the mults would sit outside the one-hot check and be silently
unchecked.

### 4.6.3 Two modelling bugs the board caught in its own audits

Recorded because they are the argument for keeping two-sided controls on
everything, including the audits themselves.

1. **B0b went vacuous.** My first widened version added the registrar's one-hot
   unconditionally, which forces `MU = 1` outright — so the `dropped` leg came
   back UNSAT and the audit was testing nothing. Removed: idx 4 *does* give MU
   booleanity (MU **is** the mode sum), and that is what B0b checks; one-hotness
   is M8's job. The division of labour is sharper than the original claim.
2. **A residue bug.** `MU` is a *sum* of felts, so as a z3 `Int` it can exceed
   `p`; comparing the raw value against 0/1 let `mu = p + 1` count as "not 1"
   and reported SAT for a sound chip. Now compared by residue.

Neither would have been visible without the `present`/`dropped` pair on each
audit. An audit with only one leg is an audit that can quietly stop testing.

---

## 4.7 POST-MODE_L — the gating split, and the audit it demanded

### 4.7.1 ⚠ WA9 — O1 is TWO obligations and only ONE of them narrowed

The review target worth the attention it was given. When `MODE_L` landed, the
lane identity `idx 6-13` was **narrowed to the digest modes**
(`DIGEST_MODE_COLUMNS = MODE_C + MODE_T`). That is correct and necessary: on a
leaf row the eight lanes are four felts' *halves*, so `IN_lane` and `m[lane]` are
deliberately different field elements, and gating on the full `MU` would make
every leaf row unprovable.

**But O1 was never one obligation.** It is a *lane identity* plus an *AreBytes
range bound*, and the leaf block depends on the second, not the first:
canonicity **assumes** `lo, hi < 2^32` and does not establish it.

✓ VERIFIED from the source, and this is what makes the design sound: the lane
`AreBytes` sends carry `Multiplicity::Sum3(MODE_C, MODE_T, MODE_L)` — the **full**
mu — so all 32 lane byte columns stay bounded on leaf rows. **The identity
narrowed; the range bound did not.**

WA9 turns that from an observation into a control, because the plausible future
refactor is *"tidy the multiplicities so they match"*:

| | check | result |
|---|---|---|
| WA9 | `AreBytes` still covers leaf rows (as built) → felt→halves map injective | **UNSAT** |
| WA9 | `AreBytes` **narrowed** to the digest modes → a felt gets a second half-pair | **SAT** |

The second row is the finding: with the bound gone, `lo` and `hi` become full
field elements, a second encoding of the same felt satisfies binding *and*
canonicity, and **the canonicity gate is still there but VACUOUS**. That is
precisely the trap `LEAF.md` §2.2 warned about, now executable.

### 4.7.2 WA8 — leaf canonicity

| | check | result |
|---|---|---|
| WA8 | canonicity present → a non-canonical half-pair is unprovable | **UNSAT** |
| WA8 | canonicity dropped → a felt acquires a second half-pair | **SAT** |

`p − 1 = 0xFFFFFFFF_00000000`, so every pair with `hi` maximal and `lo ≥ 1`
encodes a field element that *also* has an ordinary encoding — one felt, two leaf
digests, which is the collision a Merkle tree must not have. The honest leg is
covered Rust-side by the build's own tests; the "dropped ⇒ SAT" leg needs the
gate, since it means editing the constraint set.

### 4.7.3 M8 over four selectors

A third tag does not weaken the M8 finding and does not strengthen idx 4: the
mode sum is still only a *sum*, so a fractional split still reaches any target
tag. Verified with `MODE_L` in the span — forged target **SAT** under idx 4
alone, **UNSAT** under the four-way one-hot, and all three real tags still
reachable (the honest legs).

---

## 4.8 THE D1 FIX — one audit, and one documented blindness

### 4.8.1 No honest row is over-constrained

`emit_unread_input_pins` derives slot `k`'s selector as the sum of modes with
`num_input_cells() <= k`. ✓ VERIFIED against `instr.rs:104-110`
(Compress/Transcript 2, Leaf 1, Permute 3), the resulting pin matrix is:

| mode | reads | slot 1 (`IN4..8`) | slot 2 (`IN8..12`) |
|---|---:|---|---|
| Leaf | 1 cell | **pinned** | **pinned** |
| Compress | 2 cells | free | **pinned** |
| Transcript | 2 cells | free | **pinned** |
| Permute | 3 cells | free | free |

**No mode is ever pinned on a cell it reads** — UNSAT, and that is the property
a soundness fix most easily breaks, because over-constraining makes honest rows
unprovable rather than making dishonest ones provable, so the tests that would
catch it are the *honest-path* ones.

### 4.8.2 ⚠ DOCUMENTED BLINDNESS — the pins are inert on BLAKE3

**This gate cannot show the D1 pins are necessary, and it is important that this
is written down rather than inferred from a green board.**

On the BLAKE3 arm the two unread cells are read by nothing: cell 1 is read only
by the lane identity `idx 6-13`, which is gated on the digest modes and therefore
zero on the one row (leaf) where cell 1 is unread; cell 2 is read only through
`idx 0-3`'s `MODE_P · IN` term, and `idx 5` pins `MODE_P` to zero permanently.
So dropping the BLAKE3 pins cannot change a BLAKE3 digest, and any "wrong
output" question this gate asks about them returns UNSAT.

| | what | verdict |
|---|---|---|
| **this gate certifies** | the pins are **inert** on BLAKE3 (hygiene) | UNSAT |
| **this gate cannot show** | their **necessity** on `Test`/`Poseidon`, where those cells *are* read — D1's actual defect | out of model |
| **what carries that instead** | the builder's Rust junk-rejection controls, in the WA9 shape (drop the pins → SAT) | Rust side |

**"The gate said UNSAT" is exactly how a fix gets dropped as redundant**, which
is why this is a labelled blindness row on the board and not an omission. It is
the same discipline as the `drop_carry_bool` BV blindness in §4.2.

The general lesson, worth keeping: **hygiene in one arm was soundness in
another.** The BLAKE3 arm pinned its unread cell and called it hygiene — correctly
— and the identical omission in `eval_test`/`eval_poseidon` was a HIGH soundness
defect. A property's importance is not a property of the constraint; it is a
property of the constraint *plus the arm it sits in*.

---

## 5. Conformance verdict, row by row

Against the Phase-2 report's §3 table, re-derived from the source rather than
taken on trust. All rows conformant. The one flagged deviation (row 6, `add2`) is
**closed** — the model now matches the chip. Specifically re-verified by reading
`eval`: the framing indices (0–3, 4, 5, 6–13, 14–21, 22–25), `word_expr` as
`Σ byte·2^{8k}` little-endian, `half_expr` as `b0 + 256·b1`, add3 as sum identity
plus two booleanities, the rotation's four identities, and the send shapes
(`ByteAlu[XOR]` 4 per XOR word; `AreBytes` 4 per rotation; `AreBytes` 2 per lane
× 8 lanes = 16), every send `Multiplicity::Column(MU)`.

**Max degree 3**, unchanged: reached by the µ-gated carry booleanities, including
add2's, whose expression carry is a linear form so the product stays degree 3.

---

## 6. What this gate does and does not establish

| claim | status |
|---|---|
| the transcribed bodies compute the anchored socket reference at 6r and 7r | ✓ EXECUTED (T1–T4) |
| every framing/wiring bug class is still caught after transcription, both round counts | ✓ EXECUTED (36 controls) |
| the add2 expression-carry form pins `s`, and its bound is necessary | ✓ EXECUTED (WA7) |
| the four BLOCK-0 framing constraints do what they claim, and idx 5 is load-bearing for idx 0–3 | ✓ EXECUTED (B0a–B0c) |
| model census == built chip census, both round counts | ✓ EXECUTED (exact) |
| the gated artifact is the one on disk | ✓ EXECUTED (`artifact_pin.py --check`, v2) |
| the chip's framing constants EQUAL the oracle spec (tag/flags/block_len/counter/window/rounds) | ✓ EXECUTED (§1.3) -- **new in v2; v1 could not see this** |
| AR1–AR4 | ✗ argued, not solved — stated in §4.4 |
| monolithic symbolic `rounds = 1,2` UNSAT | ✗ bonus, still not completed (see `ORACLE.md` §8) |
| the socket identity against the Rust `blake3` crate | ✗ deferred — needs cargo |
| **the `permute` socket** | ✗ **OPEN** — not specified, not built; chip pins `MODE_P = 0` so a program using it is unprovable rather than silently wrong. Good failure mode, still a gap |
| **O5 leaf/parent domain separation** | ✓ **DECIDED 2026-08-10 — no longer open.** Ratified by the user: any future leaf-hashing path MUST use the reserved `"LFML"` tag (the RFC 6962 leaf/parent split expressed in the tag scheme, keeping both domains direct `blake3::hash` KATs). Recorded in `ORACLE.md` §7 and in the chip's module docs (`cece4a0b`, justification corrected in `2957c3f9`). Nothing implements `"LFML"` yet, and the safety argument is **fixed depth alone** — NOT absence of leaf hashing: FriToyV0 already compresses raw data rows into leaf digests under the same `"LFMC"` tag (`programs.rs:577/585/625`), safe only because every current tree is a fixed-depth static circuit (eDSL shape fixed at build time; hints supply values, never structure). **The obligation binds review, not code**: a change adding variable-depth trees, or leaf hashing meant to coexist with them, without `"LFML"` is rejected on O5 |

Two things I did **not** do, deliberately: I did not run cargo (the reviewer is
using the worktree), so the chip's own Rust tests are not part of this verdict;
and I did not re-derive the reference or the KATs, which are unchanged from
`ORACLE.md` and remain externally anchored.

---

## 7. Files touched

| file | change |
|---|---|
| `chip_model.py` | `emit_add2` → expression-carry (no carry column, 4 cells); BLOCK-0 documented as built with its four constraints; census prefix 13 → 28; `drop_add2_carry` bug hook |
| `gate.py` | WA7 + B0a/B0b/B0c field audits; argued ledger AR1–AR4; control sweep at both round counts; blindness row sharpened |
| `artifact_pin.py`, `artifact_pin.json` | **v2** — four hashed regions (the fourth being the framing constants v1 missed) plus resolved-value conformance against `socket_ref.py`; refuses to pin if the chip's framing diverges from the oracle. v1's fail-open is described in §1.1 |
| `run-chip-gate.log` | **the board of record** — 75 checks, both round counts |
| `run-gate.log` → `run-gate-PRE-PHASE2-SUPERSEDED.log` | renamed with a DO-NOT-CITE header; its census figures are the superseded model's and contradict §3 |
| `ORACLE.md` | unchanged; superseded only where noted in §2.2 (MU booleanity) and §2.1 (add2) |
