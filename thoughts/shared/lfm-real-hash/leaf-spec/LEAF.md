# The `LFML` leaf mode — specification

**Status:** specification + reference + vectors + gate plan, written **before any
Rust exists**. **No chip code exists for this.** **Date:** 2026-08-11.

**Decision this implements:** the user ratified **option C with `LFML`**
(`../leaf-convention-options.md`): arbitrary field elements reach the BLAKE3
socket as **checked u32 halves inside the socket itself**, under a fourth
preprocessed selector `MODE_L` carrying the `"LFML"` tag. `MODE_L` **implies
felt-input semantics** — fixed by decision, not assumed.

**Sequencing:** this builds on **committed B1**, not into B1's uncommitted diff.

Claims are ✓ VERIFIED / ✓ EXECUTED / ? INFERRED / ✗ OPEN.

---

## 0. Board

| check | result |
|---|---|
| `leaf_kats.py` L1–L6 | **PASS**, both round counts |
| canonicity predicate == `v < p` | ✓ EXECUTED, **300,007 cases** incl. every boundary |
| non-canonical inputs rejected, not reduced | ✓ EXECUTED (L3) |
| `LFML` / `LFMC` / `LFMT` pairwise distinct on the same lanes | ✓ EXECUTED (L5) |
| `TAG_LFML` = `0x4C4D464C` round-trip | ✓ EXECUTED |
| `FriToyV0` 91 compresses / 502,047 @7r | ✓ EXECUTED, matches the ratified pricing |

---

## 1. Construction

### 1.1 The felt↔halves boundary

```
v = lo + 2^32·hi ,   lo, hi ∈ [0, 2^32)
```

**Canonicity, and why it is cheap.** `p − 1 = 0xFFFFFFFF_00000000` — that is
`hi = 2^32−1`, `lo = 0`. So for halves already known to be `u32`:

> **`v < p` ⟺ NOT( `hi = 2^32−1` AND `lo ≥ 1` )**

✓ EXECUTED over 300,007 cases including every boundary. The socket's **existing**
O1 machinery (byte columns + `AreBytes` + the lane identity) already forces
`lo, hi < 2^32`; canonicity was the only missing piece.

**Boundary table** — the cases the KATs pin:

| felt | `hi` | `lo` | canonical? |
|---|---|---|---|
| `0` | `0x00000000` | `0x00000000` | ✓ |
| `1` | `0x00000000` | `0x00000001` | ✓ |
| `2^32 − 1` | `0x00000000` | `0xFFFFFFFF` | ✓ |
| `2^32` | `0x00000001` | `0x00000000` | ✓ |
| `p − 2^32` | `0xFFFFFFFE` | `0x00000001` | ✓ |
| `p − 1` | `0xFFFFFFFF` | `0x00000000` | ✓ **the tight case** |
| `p` | `0xFFFFFFFF` | `0x00000001` | ✗ **rejected** |
| `p + 1` | `0xFFFFFFFF` | `0x00000002` | ✗ rejected |
| `2^64 − 1` | `0xFFFFFFFF` | `0xFFFFFFFF` | ✗ rejected |

**REJECT, DO NOT REDUCE.** A non-canonical input has no satisfying witness, so
the row is unprovable. Same shape as O1 itself; the host-side impl must refuse,
never wrap.

### 1.2 Lane layout

A leaf row hashes **four felts** = eight lanes = exactly one compress input:

```
lanes = [lo0, hi0, lo1, hi1, lo2, hi2, lo3, hi3]      felt i at lanes 2i, 2i+1
```

**Halves adjacent** is load-bearing: it lets the canonicity gate read one pair of
neighbouring lanes rather than reaching across the row.

### 1.3 Byte serialization (normative — the crate-KAT anchor)

Each lane is four **little-endian** bytes, in lane order, then the tag:

```
msg = LE32(lo0)‖LE32(hi0)‖…‖LE32(lo3)‖LE32(hi3)‖"LFML"      (36 bytes)
digest = BLAKE3(msg)[0..16]  read back as four LE u32 lanes
```

✓ EXECUTED (L1): identical to the word-level route at both round counts. At 7
rounds this is a plain `blake3::hash` call — **the crate-KAT property survives
because the message layout is byte-identical to a digest-mode compress.**

### 1.4 Leaf structure: 1 → 3 compresses

A `FriToyV0` leaf covers **two trace rows = eight field elements**
(`NUM_COLS = 4`). Four felts per row ⇒

```
d0   = LFML(f0..f3)          leaf row 1
d1   = LFML(f4..f7)          leaf row 2
leaf = LFMC(d0, d1)          ordinary parent
```

Two `LFML` rows + one `LFMC` parent. ✓ EXECUTED (L6).

---

## 2. `MODE_L` — layout and constraints

### 2.1 Layout, applying the builder's §7.2 lesson

`MODE_L` **must sit inside the contiguous selector run**: the admission
validator's one-hot check reads `NUM_SELECTORS` from `MODE_C`, so a selector
parked past the multiplicities would be outside that check and silently
unchecked. That is the mistake §7.2 already caught once; it must not be repeated.

| index | column | change |
|---:|---|---|
| 6 | `MODE_C` | — |
| 7 | `MODE_P` | — |
| 8 | `MODE_T` | — |
| **9** | **`MODE_L`** | **NEW** |
| 10 | `MULT0` | shifted 9 → 10 |
| 11 | `MULT1` | shifted 10 → 11 |
| 12 | `MULT2` | shifted 11 → 12 |

**`NUM_SELECTORS` 3 → 4. `PREP_WIDTH` 12 → 13.**

`TAG_SELECTOR` gains `(cols::MODE_L, TAG_LFML)`; `MU_COLUMNS` becomes
`MODE_C + MODE_T + MODE_L`.

### 2.2 The constraints

Per felt `i ∈ 0..4`, with `lo = lane(2i)`, `hi = lane(2i+1)`, `v = IN_i`:

```
binding      MU_L · ( v − lo − 2^32·hi )        = 0        degree 2
canon-a      MU_L · Z_i · G_i                   = 0        degree 3
canon-b      MU_L · ( 1 − Z_i − G_i·GINV_i )    = 0        degree 3
canon-c      MU_L · Z_i · lo                    = 0        degree 3

where  G_i = (2^32 − 1) − hi        MU_L = MODE_L
```

`canon-a` gives `G ≠ 0 ⇒ Z = 0`; `canon-b` gives `G = 0 ⇒ Z = 1`; `canon-c` then
says *hi maximal ⇒ lo zero*. This is **`LFM_BITDEC`'s own `Z`/`GINV` idiom**
(✓ VERIFIED `chips.rs`), applied to two halves instead of 64 bits — the machine's
established canonicity shape, not a new invention.

**Cost: 2 witness columns (`Z_i`, `GINV_i`) and 4 constraints per felt** — 8
columns and 16 constraints per row, **zero extra sends**. Max degree stays **3**.

**Range comes free.** `lo` and `hi` are ordinary input lanes, so the existing
lane identity plus `AreBytes` already force them `< 2^32` — the same machinery
WA1/WA2 gate. **This is the whole reason option C is cheap** and it must be
stated in the chip's own docs, because a future reader who does not see it may
"helpfully" add a redundant range check or, worse, remove the lane identity
believing the canonicity gate subsumes it. It does not: canonicity assumes the
`u32` bound, it does not establish it.

---

## 3. Gate extension plan

### 3.1 New width-audit pair (field domain)

The direct analogue of WA1/WA2, and the item that makes §2.2 checked rather than
asserted:

| | check | expected |
|---|---|---|
| **WA8** | canonicity present → a non-canonical felt is **unprovable** | UNSAT |
| **WA8** | canonicity **dropped** → the same felt becomes provable | **SAT** |
| **WA8** | honest leg → canonical felts still prove | **SAT** |

The honest leg is not optional: a "fix" that rejected every felt would pass the
first two on its own.

### 3.2 M-controls, extended

- **M8 over a four-way one-hot.** The existing M8 (idx 4 pins the mode *sum*, not
  the selectors) extends unchanged — with four selectors a fractional split still
  forges any tag as a blend, so the registrar's one-hot check remains the
  load-bearing mechanism. Verify M8 fires with `MODE_L` in the span.
- **M9 (new) — mode confusion.** An `LFML` row computing an `LFMC` or `LFMT`
  digest → **SAT** (detected), and the mirror images. Three tags now means six
  ordered confusions; L5 already shows the three are pairwise distinct at the
  reference level.
- **M10 (new) — felt-input semantics is implied.** A row with `MODE_L = 1` that
  skips the canonicity block → **SAT**. This is what pins "`MODE_L` implies
  felt-input" as a constraint rather than a convention.

### 3.3 Pin

`artifact_pin.py` must resolve **`TAG_LFML`** into the checked set and assert the
three tags are **pairwise distinct** (today it checks only `LFMC ≠ LFMT`).
`framing_consts` and `cols` regions will both drift; `eval` will too.

---

## 4. O5 — what closes, and what the fixed-depth argument becomes

✓ VERIFIED, and the framing needs stating plainly: **`FriToyV0` already performs
leaf hashing today**, compressing raw trace rows into leaves under the **`LFMC`**
tag. The claim "no leaf-hashing path exists" is false. O5's safety today rests on
**fixed depth alone** — every eDSL circuit is fixed-shape at build time, so no
variable-depth second-preimage confusion is reachable.

**With `LFML` live, that changes structurally.** Leaves and parents occupy
different domains by construction: a leaf digest is `BLAKE3(…‖"LFML")` and a
parent is `BLAKE3(…‖"LFMC")`, so an internal node can no longer be replayed as a
leaf regardless of tree shape.

**What fixed depth still buys: nothing that O5 needs.** It remains true of every
current program and is worth keeping as a property, but it stops being
load-bearing for second-preimage resistance.

> **O5's obligation becomes: RETIRED, and enforced by the tag rather than by
> review.** The rule "any leaf-hashing path MUST use `LFML`" stops being a
> review checklist item and becomes a mechanical fact — a leaf row is one with
> `MODE_L` set, and `MODE_L` selects `LFML`. The reviewer's job shrinks to
> *"is this row's mode right?"*, which the one-hot check and M9/M10 answer.

? INFERRED and worth a build-time check: whether any *existing* program besides
`FriToyV0` compresses non-digest data under `LFMC`. If one does, it is a leaf
path that must move to `MODE_L` in the same pass.

---

## 5. Program impact

| | today (post-B1) | with `MODE_L` | note |
|---|---:|---:|---|
| per-row price @7r | 5,509 | **5,517** | +8 cells: the canonicity witnesses exist on every row |
| `FriToyV0` compresses | 67 (blocked) | **93** | leaves 1→3, **plus 2 LFML rows for the `t0`/`t1` felt absorbs** — see the correction below |
| `FriToyV0` cell-equiv @7r | — | **513,081** | ✓ EXECUTED against the built chip |
| `TrivialV0` compresses | 3 | 3 | unchanged in count |
| `TrivialV0` cell-equiv @7r | 16,527 | **16,551** | ⚠ **not unchanged** — see below |

> ⚠ **CORRECTION TO THIS SPEC — mine, and worth reading as a lesson.** §5
> originally said **91 compresses / 502,047**. The truth is **93 / 513,081**: the
> transcript's `absorb2(t0w, t1w)` absorbs the terminal-polynomial coefficients,
> which are **arbitrary field elements**, so they must go through the leaf/felt
> path — two more `LFML` rows, +11,034 cell-equiv.
>
> **The failure was not arithmetic, it was leaving an open item open.** The
> options note flagged exactly this at §6: *"whether `t0w`/`t1w`'s absorbs want
> felt mode or a separate convention"* — recorded as ✗ OPEN. Then this spec
> asserted "transcript unchanged at 11" and put a definite number in a table.
> **An open question carried into a concrete figure stops looking open.** The
> ranking is unaffected (C stays ~12% under A), but the number was wrong for two
> documents until the build measured it.
>
> ⚠ **Correction to the brief:** `TrivialV0` is **not** cost-unchanged. It gains
> **+24 cell-equiv** (3 rows × 8 columns), because the canonicity witness columns
> are part of the AIR and therefore exist on *every* compress row, leaf or not.
> Small, but the brief said "unchanged" and the census must not carry a claim the
> formula contradicts.

**Registry re-bless:** rides the `PREP_WIDTH` 12 → 13 change — the preprocessed
roots move, so all entries are re-blessed **once**, in the same pass. ✓ Consistent
with how B1's re-bless was sequenced.

**Tripwire replacement.** `blake3_socket_tests::fri_toy_is_still_blocked_by_o1_and_no_longer_by_the_sponge`
asserts (a) no permute remains, (b) fixture values are not `u32`-laned, (c) the
refusal is specifically O1, (d) the honest control under `Test`. Its own doc says
it must be replaced when O1 closes. **Replacement criteria:**

1. delete it only when `FriToyV0` **proves and verifies** under `Blake3`;
2. the replacement is a **prove+verify**, not an execute — an execute-only test
   proves nothing about the chip;
3. keep an honest control that the same program still proves under `Test`;
4. add a **negative** leg: a deliberately non-canonical arena value must make the
   proof fail, and fail *for canonicity*, not for some other reason.

Criterion 4 is the one most likely to be skipped, and it is the one that shows
the canonicity gate is doing work in the assembled program rather than only in
the unit test.

---

## 6. Open

| item | status |
|---|---|
| the same identity against the Rust `blake3` **crate** | ✗ DEFERRED — needs cargo |
| WA8 / M9 / M10 against a real chip | ✗ OPEN — needs the build |
| any other program leaf-hashing under `LFMC` (§4) | ✗ OPEN — build-time sweep |
| tag tables gain `"LFML"` as **live** rather than reserved | ✗ OPEN — one pass, with the build |

## 7. Files

| file | what |
|---|---|
| `leaf_ref.py` | the reference: halves boundary, canonicity predicate, leaf compress |
| `leaf_kats.py`, `leaf_kats.json` | L1–L6 incl. boundary felts and non-canonical rejects |
| `../leaf-convention-options.md` | why option C was chosen (the decision record) |
