# The LFM-native commitment layer — specification

> # ⚠ DRAFT — PENDING MAURO RATIFICATION
>
> **The decision points are closed; the construction is not.** D1–D6 were ruled
> on by Mauro on 2026-08-12 and are recorded with provenance in §7. What still
> needs his read is **the S1 wide-leaf construction itself** (§1) — the part
> nobody has ratified because nobody had specified it before this document.
>
> **★★ Read §1.4.1 first.** The leaf **RATE** (`LFML_FELTS_PER_ROW = 4`) is the
> single most consequential number here: leaf absorption is 69.8% of a tower
> node's bill, and this parameter decides whether the recursion tower fits on
> real hardware. Gate D1 was projected to **FAIL at 124 GiB** against a ~93 GiB
> budget at the old rate; at `RATE = 4` it lands at **≈81 GiB**. It is **✗ OPEN
> (D8)** and it is a chip change, so it wants a deliberate yes/no.
>
> > **★★ NEW — D9: rate 4 or rate 5? An OPTIMIZATION call, not a fit call.**
> > A refute pass on 2026-08-12 (§1.4.2a) found that the argument retiring
> > `RATE = 5` — *"a hash row reads whole cells, so felts per row must be a
> > multiple of 4"* — is a **non-sequitur**: a receive binds all four felts of a
> > cell to memory, so a row may read three cells and use nine felts soundly. The
> > true ceiling is **5**, not 4, and three message words sit dead at rate 4.
> > `RATE = 4` **stays the adopted working default** and Gate D1 already fits at
> > it with ~13% margin, so nothing is blocked. But rate 5 is ? ~19% cheaper on
> > an **unpriced** sketch and someone should decide whether to price it.
> > **✗ OPEN (D9)**, §7.
>
> **⚠ Before anyone writes the chip change, read §1.4.4** — nine verified
> implementation hazards. **H1 is a silent one:** at `NUM_LANES = 12` the lane
> identities collide with the output pins, the constraint *count* does not move,
> and the only assert that would catch it is disabled in release builds.
>
> This is step 1 of the D0 change list (`../D0-DESIGN.md` §6), written **before
> any Rust exists**, in the same discipline as `lfm-real-hash/leaf-spec/LEAF.md`
> and `lfm-real-hash/transcript-spec/TRANSCRIPT.md`. Three sub-questions remain
> **✗ OPEN** (**D8**, **D9** and **D6a**, §7); nothing has been silently
> defaulted.

**Date:** 2026-08-12. **Depends on:** ratified `LFMC` (Merkle parent), `LFML`
(leaf/felt mode, LEAF.md) and `LFMT` (B1 transcript, TRANSCRIPT.md).
**Allocates no new socket tag.**

**What it covers** — the three things no ratified doc covers when the LFM
machine's own proof moves to the machine's native hashing scheme:

1. the **wide leaf**: an arbitrary-width row pair → a chained `LFML` sequence,
   with the shape bound inside the construction (D0 §7 **S1**, the gating item,
   and **S3**) — and its **RATE** (§1.4.1), the parameter that decides whether
   the recursion tower fits on real hardware;
2. the **byte→cell absorb** encoding for the B1 transcript (D0 §3 item 4);
3. the **node codec** — `pack_digest` into `[u8;32]` plus a strict decode
   (**S2**) — and the tree's arity/padding rule (**S6**);
4. **grinding under B1** (§4.1) — added after the D3 ruling, since B1 has no
   `state() -> [u8;32]` and cannot express the keccak PoW it replaces.

Claims are ✓ EXECUTED (ran it, output in `run-kats.log`) / ✓ VERIFIED (read the
code, cited) / ? INFERRED / ✗ OPEN.

---

## 0. Board

✓ EXECUTED, `python3 commit_kats.py`, full log in `run-kats.log`.

| id | check | result |
|---|---|---|
| **C1** | wide leaf over a BASE matrix, both round counts, cost formula | **PASS 4/4** |
| **C2** | wide leaf over an EXT3 matrix; same felt count, different kind ⇒ different leaf | **PASS 4/4** |
| **C3** | ★ width binding: the recorded live break, plus the honest leg | **PASS 4/4** |
| **C4** | ★ padding is unambiguous *because* the header binds the count | **PASS 3/3** |
| **C5** | byte→cell encoding: O1 automatic, injective under zero-pad | **PASS 6/6** |
| **C6** | ★ node codec: round-trip + four rejection flavours + honest leg | **PASS 8/8** |
| **C7** | tree arity/padding: power-of-two asserted, not padded | **PASS 3/3** |
| **C8** | the 96-bit question, both options costed in compressions | **PASS 4/4** |
| **C9** | ★ the crate anchor survives — `LFML` rows are still plain `blake3` @7r | **PASS 2/2** |
| **C10** | ★ the header is load-bearing (construction-level domain separation) | **PASS 3/3** |
| **C11** | ★ B1 grinding: honest mine, factor/seed/marker binding, both cross-domain directions, range discipline, **and the absorb identity §4.1.3 depends on** | **PASS 14/14** |
| **C12** | ★★ the leaf RATE: 4 felts/compression (a whole machine cell), anchor intact, header properties survive, per-query cost 6,048 → 3,024 | **PASS 11/11** |
| **PIN** | all 19 vectors match `commit_kats.json` | **PASS 19/19** |
| | **TOTAL** | **85/85 PASS** |

> ⚠ **Two pinned vectors were re-blessed on 2026-08-12** — `C12.per_query_old`
> 6,062 → **6,048** and `C12.per_query_new` 3,031 → **3,024** — when §1.5's
> `LFM_HASH` census row was corrected from `NUM_COLUMNS` (3,457) to main columns
> (**3,444**), the preprocessed prefix being committed in the precomputed tree
> rather than the main tree. `commit_kats.py:457`'s width list carries the
> correction and its reason. **No cryptographic vector moved**: the re-pin diff is
> exactly those two integers, and the other 17 digests are byte-identical.
> Recorded here rather than absorbed silently, because re-pinning a KAT to match
> a new belief is how a regression gets laundered — this one is a scope fix with
> a stated reason and a checkable diff.

Run order: `python3 commit_kats.py --write` once, then `python3 commit_kats.py`
to check. Plain `python3`, no cargo, no third-party packages.

---

## 1. The wide leaf (S1 — the gating item)

### 1.1 The problem, stated from the code

Production hashes a leaf as `evaluations ‖ evaluations_sym`, streamed **with no
length prefix and no separator**. ✓ VERIFIED — `verifier.rs:204-206` says it in
those words, and `verify_opening_pair` (`verifier.rs:569-594`) is the single
generic implementation, instantiated at `Field` for the main and precomputed
trees and at `FieldExtension` for the aux and composition trees.

The consequence was a **live break**, and the code records it rather than
alluding to it. ✓ VERIFIED `verifier.rs:633-639`:

> *"This authenticates the opening against the aux root; it does NOT constrain
> how many columns that opening has. Nothing here did, and that was a live
> break: the aux root is absorbed only after the shared LogUp challenges, so a
> prover that moved main columns into the aux tree got to choose them after
> seeing `z`/`alpha` (`tests::aux_opening_width_tests`). The width is pinned
> upstream by `trace_opening_widths_well_formed`; do not re-derive it from the
> proof."*

So the hazard is not "a wrong width" in the abstract — it is **moving columns
between trees that are absorbed at different times**, buying the prover a choice
after a challenge that should precede it. Today that is closed by an *external*
check (I3, `trace_opening_widths_well_formed`), not by the hash.

Rebuilding the leaf under `LFML`/`LFMC` is the moment to decide whether the hash
carries its own shape. **It should.**

### 1.2 The construction

```
RATE  = LFML_FELTS_PER_ROW = 4                    ★ the spec parameter, §1.4.1
H     = [ LEAF_MARK, num_cols, kind, ROWS_PER_LEAF ]        one header cell
F     = serialize(evaluations) ‖ serialize(evaluations_sym)
F'    = F ‖ 0^r                  r = (−|F|) mod RATE         zero-pad to RATE
acc   = H
        for each chunk c of RATE felts:  acc = LFML_row(acc, c)
leaf  = acc

LFML_row(acc, c) = BLAKE3( LE32(acc[0..4])
                         ‖ LE32(lo_i)‖LE32(hi_i) for each felt in c
                         ‖ "LFML" )[0..16]              52 bytes, ONE block
```

**The accumulator rides in the message, so there is no separate fold** — each
row absorbs `RATE` felts *and* chains, in one compression. That is the whole of
§1.4.1, and it is the parameter that decides whether the tower fits.

- `kind` ∈ {1 = base, 3 = ext3} — the felts-per-element count doubles as the
  kind tag: injective over the kinds that exist, and the number the serializer
  needs anyway.
- `serialize` writes a row column by column; an ext3 element contributes its
  three components in order `(c0, c1, c2)`, mirroring `write_bytes_be`
  (✓ VERIFIED `sub_proof.rs:234-236`).
- `ROWS_PER_LEAF = 2` (✓ VERIFIED `commitment.rs:42`). It is **not** a parameter
  of the function — the two-slice signature *is* the row pair — but it is bound
  in the header so a future layout could not collide with this one.

### 1.3 Why a header cell, and why these fields

**The header binds `num_cols` AND `kind`.** Binding the width alone would not
close §1.1: 6 base columns and 2 ext3 columns serialize to the **same twelve
felts**, so under a width-only header those two openings still share a preimage
— which is the main↔aux confusion in miniature. ✓ EXECUTED (**C3**): the two
produce different leaves under this construction, and (**C2**) the same holds
for the 18-felt pair.

**The verifier must build the header from the AIR, never from the opening.**
This is the whole load-bearing condition and it is the exact analogue of the
instruction already in the code at `verifier.rs:639` — *"do not re-derive it
from the proof."* A verifier that set `num_cols = len(evaluations)` would
reproduce the prover's own choice and bind nothing at all. The reference
enforces this shape by taking `num_cols` as an argument and *checking* the data
against it (✓ EXECUTED, **C3**: a disagreeing width is refused).

**Zero-padding is safe here, and only here.** Two felt streams that agree after
padding must have differed in `(num_cols, kind)`, which the header separates.
✓ EXECUTED (**C4**) on a constructed collision: `m=1` padded and `m=2` unpadded
share the felt stream `[7, 9, 0, 0]` and produce different leaves. Without the
header, that collision is real.

**The fold is a sequential chain, not a balanced tree.** A balanced tree over
`k` chunk digests costs `k−1` compressions against the chain's `k` — but needs
`k` padded to a power of two, reintroducing exactly the shape ambiguity the
header exists to remove. One compression is not worth a second padding rule, and
the chain binds chunk order for free.

### 1.4 Cost

```
compressions = ceil( 2 · num_cols · kind / RATE )          RATE = 4
```

So **`0.5 · num_cols` compressions per base leaf** and `1.5 · num_cols` per ext3
leaf. ✓ EXECUTED (**C1**, **C2**, **C12**).

### 1.4.1 ★★ The leaf RATE — the parameter Mauro must ratify

> **This single number decides whether the recursion tower fits on real
> hardware.** Leaf absorption is **69.8%** of a tower node's bill (Gate D1
> census), so the rate scales ~70% of the cost linearly. The Gate D1 node was
> projected at **124 GiB against a ~93 GiB budget — a 1.3× FAIL** at the old
> rate.

**What the chip actually supports** — ✓ VERIFIED, not taken on faith.
`message_word_ref` (`blake3_socket.rs:725-731`) maps `m[0..8]` to the eight
input lanes' byte columns, `m[8]` to the mode-selected tag, and **`m[9..16]` to
`WordRef::Const(0)`**. Seven of BLAKE3's sixteen message words are dead; the
socket uses nine (`BLOCK_LEN_LFMC = 36`, `blake3_socket.rs:261`).

There is headroom to spend — but not as much as the block alone suggests:

> ### ⚠⚠ The binding constraint is the machine's CELL structure, not the block
>
> ✓ VERIFIED `instr.rs:99-110`: `HashMode::num_input_cells` is **2** for
> Compress/Transcript, **1** for Leaf, 3 for Permute — and the doc is explicit
> that *"the `LFM_HASH` bus receives are gated by exactly this"*. A hash row
> reads whole **cells** from memory, and a cell is **four felts**
> (`LfmWord`, `word.rs:15`).
>
> **So the felts per row must be a multiple of 4.** An earlier revision of this
> section set `RATE = 5` from block headroom alone — accumulator cell plus five
> felts. That is 1.25 cells of felt input and is **unbuildable**: the machine
> cannot read it. The error was reasoning from BLAKE3's block size while
> ignoring the machine's word size, and it is the reason this section is now
> written from `instr.rs` rather than from byte counts.
>
> ---
>
> > ### ⛔ SUPERSEDED by §1.4.2a (1) — the rule above is a NON-SEQUITUR
> >
> > **Kept in place because it is what the RATE-4 adoption was reasoned from, and
> > a reader who meets `RATE = 4` elsewhere needs to find the retraction here.**
> >
> > The quoted text is accurate: `instr.rs:100-103` does say the receives are
> > gated by `num_input_cells`. But read what it constrains — a mode must not
> > **receive** a cell it does not **read**. Nothing says a row must **use** every
> > felt of a cell it does receive, and nothing could: all four felts of a
> > received cell are bound to memory by the `LfmMem` receive
> > (✓ VERIFIED `chips.rs:628-642`), so ignoring three of them is sound, not
> > underconstrained. A 5-felt row is buildable as a **3-cell read** — accumulator
> > cell plus two felt cells, 12 felts received and 9 used. The third receive
> > already exists (`chips.rs:638-642`, multiplicity `Column(MODE_P)`); it would
> > need `MODE_L` added, which is the **same edit** the adopted RATE-4 construction
> > already needs on the second receive (see the next banner).
> >
> > **The real constraint is the compile-time lane map, and it is a stronger
> > argument.** `leaf_lo_lane(i) = 2i` / `leaf_hi_lane(i) = 2i+1`
> > (✓ VERIFIED `blake3_socket.rs:680-687`) are `const fn`s, identical on every
> > row. A rate that does not divide the 4-felt cell puts each row's felts at a
> > *different offset* inside the cells it reads, and the AIR has exactly one
> > mapping. That forces either a rotating per-row lane map or a **re-packed felt
> > stream** — and re-packing exists: `Instr::Unpack { input, outs: [Addr; 4] }`
> > and `Instr::Pack { lanes: [Addr; 4], out }` (✓ VERIFIED `instr.rs:229-241`)
> > are a felt-granular scatter/gather, running on `LFM_LANES` at `PREP_WIDTH + 4`
> > columns (`chips.rs:1267-1271`) against `LFM_HASH`'s 3,460.
> >
> > **So RATE = 5 is a COST question, not an impossibility.** It is now **D9**
> > (§7). `RATE = 4` remains the adopted working default — this banner does not
> > change it.

Enumerating what actually fits, given both constraints:

| | construction | words | rate | vs old | verdict |
|---|---|---:|---:|---:|---|
| A | 7 felts + keep the `LFMC` fold | 15 | 3.5 | 1.75× | ✗ 7 is not a multiple of 4 |
| ~~B~~ | accumulator + 5 felts, no fold | 15 | 5.0 | 2.5× | ⛔ verdict RETRACTED — see below |
| **★ C** | **accumulator cell + ONE felt cell (4 felts), no fold** | **13** | **4.0** | **2.0×** | ✓ **ADOPTED (working default)** |
| — | accumulator + two felt cells | 21 | 8.0 | 4× | ✗ > 16 words |
| — | two felt cells, keep the fold | 17 | 4.0 | 2× | ✗ > 16 words |
| — | 6 felts + keep the fold | 13 | 3.0 | 1.5× | ✗ dominated by C |

> ### ⛔ SUPERSEDED by §1.4.2a (2) — "4 is the maximum" and the headroom claim
>
> **The block ceiling is 5, not 4.** 16 message words − 1 tag − 4 accumulator
> lanes = 11 words ⇒ **5 half-pairs, with one word spare.** Row ~~B~~ above *is*
> the ceiling; it was struck only by the multiple-of-4 rule the previous banner
> retracts. At the adopted `RATE = 4` the socket uses 13 of 16 words and **three
> are dead**, so the statement below that 4 "is the whole of the available
> headroom" is false as written.
>
> **Two ways out that do NOT exist**, checked and closed so nobody re-opens them:
> the tag cannot stop consuming a word — moving it to `flags`/`t`/`h` breaks the
> crate anchor (✓ VERIFIED `blake3_socket.rs:35-41`) and it is what mechanically
> discharges O5 (`:127-142`); and the accumulator cannot overlap it — a 3-lane
> (96-bit) accumulator frees a word but drops the chain to 48-bit collision
> resistance against the socket's recorded 128-bit/64-bit posture (`:150-154`),
> while folding `acc[3]` into the tag word is *expressible* (`ModeSelected` is a
> linear form; `word_expr` would take `ModeSelected + Cols` at degree 1) but stops
> the message being a plain byte string, killing the C9 anchor — the same
> objection §1.4.1 uses against `h`-chaining. Note a *free* tag reaches only 6
> felts, so under the retracted rule it would have bought nothing either.
>
> **✗ OPEN as D9** (§7): nobody has priced RATE 5 end to end. See §1.4.2a (2) for
> the ? INFERRED ~19% sketch and, more importantly, for what it does **not**
> cover.

**`RATE = 4` is the working default, and it has one property the rate-5 route
does not:** its felt input is a whole machine cell, so the leaf program reads the
opening stream in its natural 4-per-cell layout with no re-packing pass at all.
Canonicity witnesses stay at 4 felts — **no change** — because the accumulator
lanes are a previous digest, hence `u32` by construction: they need byte
decomposition but no canonicity gate. (⚠ That last clause holds **only** if the
lanes-0–3 identity is gated on the full `mu`; see §1.4.4 hazard **H6**.)

> ### ⛔ SUPERSEDED by §1.4.2a (3) — "the frozen bus arity does not move at all"
>
> The retracted sentence read: *"it lands on the **existing** two-cells-in /
> one-cell-out bus contract (`num_input_cells == 2`, the same arity Compress and
> Transcript already use), so the frozen `LFM_HASH` bus arity does not move at
> all."*
>
> **The arity does not move. The MULTIPLICITY does.** ✓ VERIFIED `chips.rs:626`:
> the second input cell's receive is
> `reads_two() = Multiplicity::Sum3(cols::MODE_C, cols::MODE_T, cols::MODE_P)` —
> **`MODE_L` is deliberately absent**, and `chips.rs:620-622` says so in those
> words, because today a leaf row reads one cell. Under construction C a leaf row
> **must** receive cell 1, so that multiplicity gains `MODE_L`; and
> `Multiplicity::Sum3` is exactly three columns
> (✓ VERIFIED `crypto/stark/src/lookup.rs:1458`), so it must become the four-way
> `selector_sum(MODE_C, NUM_SELECTORS)` (`chips.rs:52-61`) the *first* receive
> already uses.
>
> **And raising `num_input_cells(Leaf)` to 2 panics AIR construction as the code
> stands.** ✓ VERIFIED `chips.rs:722-733`: `emit_unread_input_pins`' `slot = 1`
> pass filters modes with `num_input_cells() <= 1`; that set becomes **empty**,
> the fold returns `None`, and
> `.expect("some mode reads fewer than three input cells")` fires.
>
> This is a real edit to the frozen contract, not a no-op. It is tracked as
> hazards **H2** and **H3** in §1.4.4.

**✓ The crate-KAT anchor survives.** A full row is `16 + 32 + 4 = 52` bytes —
still **one** BLAKE3 block, so `block_len` moves 36 → 52 and nothing else about
the framing does. For any input
under 64 bytes `blake3::hash` is exactly one compression with `h = IV`, `t = 0`,
`block_len = len`, `flags = CHUNK_START|CHUNK_END|ROOT`, so a 52-byte row is a
plain library call just as the 36-byte row was. ✓ EXECUTED (**C12**), asserted
against `blake3_oracle` directly. Carrying the accumulator in the chaining value
`h` instead (the earlier D7 sketch, now **superseded**) would have made the row a
chunk *continuation* and split that anchor for the **same** rate of 4.0 — strictly
worse, since it also costs `h`-as-witness and an O3 revision.

**Chip cost of the widening** — ✓ VERIFIED against the chip, see §1.4.2 for the
audit. `NUM_LANES` appears in exactly **four** non-test places, all generic:

- byte columns `4 × NUM_LANES` (`blake3_socket.rs:618-622`): 8 → **12** lanes, **+16**
- `AreBytes` sends, 2 per lane (`blake3_socket.rs:947-957`): **+8**
- canonicity witnesses: still 4 felts, **+0**
- **the mixing core does not move**: `NUM_G = rounds × 8` G-blocks of `G_SIZE = 60`
  (3,360 cells at 7r) is driven by the round count, not by how many message
  words are live — a `Const(0)` word still feeds an `add3`.

So ≈ **+16 columns on a 3,457-column chip (+0.5%) for a 2.0× cut in ~70% of the
tower's cost.** That ratio is why this is worth a chip change at all.

**Measured effect** — ✓ EXECUTED (**C12**), at the real widths of §1.5:

| | per-query main-tree leaf compressions | ×219 q | ×110 q |
|---|---:|---:|---:|
| old rate (2 felts/compression) | 6,048 | 1,324,512 | 665,280 |
| **RATE = 4** | **3,024** | **662,256** | **332,640** |

> ⚠ These figures moved by −14 / −7 on 2026-08-12 when §1.5's `LFM_HASH` row was
> corrected from 3,457 to **3,444** (the preprocessed prefix is committed in the
> *precomputed* tree, not the main tree — §1.5 note **(4)**). The superseded
> figures were 6,062 / 3,031. `commit_kats.py`'s width list was corrected with
> them and the two `C12.per_query_*` vectors re-pinned; **no cryptographic vector
> moved** and the board stayed 85/85.

Against the Gate D1 sensitivity (×2 → 81 GiB fits; ×4 → 59 GiB), **2.0× lands
the node at ≈81 GiB — inside the ~93 GiB budget with ~13% margin.** The census
correction is −0.23% and does not move that number visibly.

> ⛔ The sentence that stood here — *"That is the whole of the available
> headroom: the enumeration above shows 4 is the ceiling, so if 13% proves too
> tight the next lever is not the leaf rate"* — is **SUPERSEDED by §1.4.2a (2)**.
> The block ceiling is 5, three message words are dead at `RATE = 4`, and a
> ? INFERRED sketch puts RATE 5 ~19% cheaper again. **If 13% proves too tight,
> the leaf rate IS still a lever — it is D9.** What is true, and worth keeping,
> is that Gate D1 already **fits** at `RATE = 4`, so D9 is an optimization
> decision and not a fit decision.

> **✗ OPEN (D8) — FOR MAURO'S RATIFICATION READ.** `RATE = 4` is a **chip**
> change (`NUM_LANES` 8 → 12, `block_len` 36 → 52, `MODE_L` semantics widened).
> It moves every `LFML` digest, so all vectors and all six registry entries
> re-bless — but that re-bless is already happening under D0, so the marginal
> protocol cost is zero **provided it is sequenced into the same pass**. That
> sequencing is the decision: taking it later costs a second re-bless.

### 1.4.2 ✓ The chip audit — does `NUM_LANES` 8 → 12 constrain cleanly?

Read of `blake3_socket.rs` (the `LFM_HASH` arm) and `blake3_chip.rs`.
**Verdict: mechanical except for one real constraint change, named below.**

> ⛔ **That verdict is too optimistic — SUPERSEDED by §1.4.2a and §1.4.4.** This
> section is a *site survey*: it enumerates where `NUM_LANES` appears, and on
> that it is correct and complete. But three of the breaks are not greppable —
> they are arithmetic on constraint **indices**, a bus **multiplicity**, and a
> mode's **cell count** — so a survey cannot see them. The count is **nine**
> hazards, not one. §1.4.4 is the register; read it instead of this line.

**`NUM_LANES` is used in exactly four non-test places, all generic:**

| site | use | generic? |
|---|---|---|
| `blake3_socket.rs:618` | `pub const NUM_LANES: usize = 8` | the definition |
| `:622` | `G = LANES + 4 * NUM_LANES` — byte-column base | ✓ arithmetic |
| `:913` | `Vec::with_capacity(… + 2 * NUM_LANES)` | ✓ capacity hint only |
| `:947` | `for lane in 0..NUM_LANES` — the `AreBytes` sends | ✓ loop |
| `:1304` | `for lane in 0..NUM_LANES` — the lane/message identity | ✓ loop, but see below |

`lane_byte(lane, b) = LANES + 4·lane + b` (`:649-651`) is generic in `lane`.

**Hardcoded 8s that must move** — mechanical, but they are real edits:
`message_word_ref`'s `0..=7` arm (`:726`), and the `[u32; 8]` lane arrays in
`socket_values` (`:864`), `bitwise_ops_for` (`:970`), `lanes_from_row` (`:1084`)
and `fill_canonicity_witness` (`:1100`).

> **★ The one substantive change, and it is NOT mechanical.** The lane/message
> identity at `:1304` is gated on `digest_mu` — the *digest* modes — and the
> comment at `:1299-1303` says why: *"On a LEAF row the eight message lanes are
> four felts' halves, so `IN_lane` and `m[lane]` are deliberately NOT the same
> field element … Gating this on mu instead would make every leaf row
> unprovable."*
>
> Under `RATE = 4` a leaf row's twelve lanes are **mixed**: lanes 0–3 are the
> accumulator (a digest — the identity *should* hold) and lanes 4–11 are the
> four felts' halves (it must *not*). So the gate stops being per-mode and
> becomes **per-lane-range**. That is a genuine constraint change with a
> soundness face: get the split wrong in the permissive direction and the
> accumulator lanes go unconstrained. It needs its own control in the chip's
> gate suite, in the style of WA1/WA2.

**The `with_capacity(1_259)` figure is not the socket's.** ✓ VERIFIED it lives in
`blake3_chip.rs:913` and is pinned by `blake3_probe.rs:351`
(`predicted_interactions(6) == 1_259`) — the standalone chip, not the `LFM_HASH`
arm. The socket sizes its own vector at `blake3_socket.rs:913` and that
expression is already generic in `NUM_LANES`. **No unasserted constant blocks
the widening.**

✓ Both halves of that are confirmed by the end-to-end read.
`blake3_chip::bus_interactions` emits exactly `107 + 24·NUM_G`
(7 receivers + 4 senders + `4·(4·NUM_G + 16)` `ByteAlu` + `4·(2·NUM_G)`
`AreBytes` + 32 message `AreBytes`), which is 1,259 at `NUM_G = 48` and 1,451 at
`NUM_G = 56`; `blake3_probe.rs:365-366` asserts the *built* length against the
formula at the compiled round count, so it is pinned, not merely predicted. It
is `NUM_LANES`-independent because the standalone chip has no lanes — all
sixteen of its message words are `Cols` and always draw 32 `AreBytes`
(`blake3_chip.rs:968-979`). ⚠ The literal is nonetheless the **6-round** one in a
file that compiles at 7 by default, so at the default it under-allocates by 192;
capacity hint only, one realloc, no correctness effect. Same for
`bitwise_ops_for`'s `1_248` (`blake3_chip.rs:991`) `= 24·NUM_G + 96`.

#### 1.4.2a ★ The end-to-end read — three breaks a site survey cannot see

The survey above enumerates where `NUM_LANES` *appears*. Reading
`blake3_socket::eval` and `blake3_chip::run_flow` end to end finds three further
breaks, none of which appears in any `NUM_LANES` grep: they are arithmetic on
constraint **indices**, a bus **multiplicity**, and a mode's **cell count**.

**✓ First, the shared dataflow itself generalises cleanly — question (ii)
answered.** A `Const(0)` message word and a `Cols` one flow through the identical
path, differing only in the operand source, and this is structural rather than
incidental: **message words reach `add3` and nothing else.** ✓ VERIFIED — the
schedule indices `mx`/`my` are consumed at exactly two call sites,
`blake3_chip.rs:327` and `:333`, both `f.add3(…)`. `word_expr`
(`blake3_chip.rs:1032-1048`) handles `Cols`, `Const` and `ModeSelected`
uniformly at degree ≤ 1, so the `add3` sum identity stays degree 2 under the
mu gate and the chip's max degree of 3 does not move. `Add3Wire.m` is already
typed `WordRef` for precisely this reason (`blake3_chip.rs:439-446`). The
`unreachable!`s in `WordRef::byte` and `WordRef::rotr_bytes`
(`blake3_chip.rs:395-424`) are never reachable from a message word. **The G-block
wiring is fully index-agnostic; the mixing core's sends, constraints and degree
are functions of the round count alone.**

> **★ Break 1 — the lane identities collide with the output pins, and in a
> release build the collision is SILENT.** ✓ VERIFIED.
>
> `eval` numbers its framing constraints by hand: the lane identities are
> `b.emit_base(6 + lane, …)` over `0..NUM_LANES` (`blake3_socket.rs:1304-1309`),
> then the unused-output pins are `b.emit_base(14 + j, …)` for `j ∈ 0..8`
> (`:1315-1318`), the digest recompositions `22 + i` (`:1325-1330`), and
> `UNREAD_IDX = 26` (`:1223`). At `NUM_LANES = 12` the lane block runs 6..17 and
> **overlaps the output pins at 14..17**.
>
> `EmitTracker::mark` asserts `"constraint {idx} emitted twice"` — but only under
> `#[cfg(debug_assertions)]` (`crypto/stark/src/constraints/builder.rs:492-504`),
> and this workspace declares no `[profile.release]` override, so under the house
> convention `cargo test --release` the tracker is a no-op and the second write
> simply overwrites the first (`builder.rs:614-617`). The lane loop runs first,
> so **lanes 8–11 lose their identity entirely and nothing fails.**
> `assert_complete` does not catch it either: every index in `0..NUM_CONSTRAINTS`
> is still written, and `NUM_CONSTRAINTS` (`:1214`) does not reference
> `NUM_LANES`, so the declared count never moves.
>
> The failure mode is exactly the soundness hole below, which makes this the most
> dangerous item in the change: the four constraints that go missing are the four
> that matter. **Fix: derive the framing indices from `NUM_LANES` instead of
> writing 14/22/26 as literals, and add a test that the emitted index set is
> `0..NUM_CONSTRAINTS` without repeats** — the debug tracker is not enough,
> because the suite runs in release.

> **★ Break 2 — raising `HashMode::Leaf` to two input cells panics the pin
> emitter.** ✓ VERIFIED. Construction C reads an accumulator cell *and* a felt
> cell, so `num_input_cells` for `Leaf` goes 1 → 2 (`instr.rs:104-110`). Then in
> `emit_unread_input_pins` the `slot = 1` iteration filters modes with
> `num_input_cells() <= 1` — **which becomes empty**, the fold returns `None`, and
> `.expect("some mode reads fewer than three input cells")` fires
> (`chips.rs:722-733`). AIR construction panics.
>
> Consequences, all mechanical once seen: the loop must skip slots no mode
> under-reads; `NUM_UNREAD_INPUT_PINS` goes 8 → 4 (`chips.rs:678`), which moves
> `UNREAD_IDX`, `LEAF_IDX`, `CORE_IDX` and `NUM_CONSTRAINTS`; and the leaf felts
> move from cell 0 to cell 1, so `leaf_lo_lane(i) = 2i` / `leaf_hi_lane(i) = 2i+1`
> (`:680-687`) become `4 + 2i` / `4 + 2i + 1` and the felt source `IN0 + i` at
> `:1369` becomes `IN0 + 4 + i`.

> **★ Break 3 — the second `LfmMem` receive excludes `MODE_L`, so the felt cell
> would never be read.** ✓ VERIFIED `chips.rs:626`: the second input cell's
> multiplicity is `reads_two() = Multiplicity::Sum3(MODE_C, MODE_T, MODE_P)` —
> `MODE_L` is deliberately absent, because today a leaf row reads one cell
> (`chips.rs:620-622` says so in those words). Under construction C a leaf row
> **must** receive cell 1, so that multiplicity has to include `MODE_L`.
> `Multiplicity::Sum3` is exactly three columns (`crypto/stark/src/lookup.rs:1458`),
> so this becomes the four-way `selector_sum(MODE_C, NUM_SELECTORS)`
> (`chips.rs:52-61`) that the first receive already uses.
>
> ⚠ **This is a correction to §1.4.1's claim that "the frozen `LFM_HASH` bus arity
> does not move at all."** The *arity* does not — still three receives, three
> sends. The *multiplicity* of the second receive does. That is a smaller change
> than a new tuple, but it is a change to the frozen contract and it must be
> stated as one, because a reader who takes "does not move at all" literally will
> not look at `lfm_mem_interactions`.

**★ A free win the survey also misses: the four new lanes are pinned for you.**
Lane bytes reach only two kinds of constraint — the identity at `6 + lane`, gated
on `digest_mu = MODE_C + MODE_T` (`:1304-1309`), and the leaf halves binding,
gated `mode_l` (`:1360-1388`) — plus the `AreBytes` sends (`:947-957`), which
bound each byte below `2^8` but say nothing about its value. So on a Compress or
Transcript row, four *unconstrained* lanes would hand the prover `m[9..13]`
outright and the parent digest would stop being a function of `(a, b)`: Merkle
parents forge. **At `NUM_LANES = 12` the existing code already closes this**, and
by luck rather than design: `b.main(0, cols::IN0 + lane)` for `lane ∈ 8..12`
lands on the **third input cell**, `IN8..IN12` (`IN0 = PREP_WIDTH = 13`,
`S8 = PREP_WIDTH + 12`, `chips.rs:482-488`) — which `emit_unread_input_pins`
pins to zero on every digest row. The identity then reads `0 = Σ bytes·2^{8k}`,
and with the `AreBytes` bound in hand that forces all sixteen bytes to zero.
✓ So the required pin is free **provided Break 1 is fixed**; if it is not, those
are exactly the four identities that get silently overwritten. Note also that 12
is the last lane count for which this holds: at 13 lanes `IN0 + 12` is `S8`, and
the identity would start reading the capacity-state columns as input felts.

**Question (iii) — `block_len` 36 → 52 flows through as a plain framing constant,
with one hard caveat.** ✓ VERIFIED: one definition (`blake3_socket.rs:261`) feeds
three consumers — the host reference (`:304`), the wire interpretation
`input_v12` (`:752`) and the value interpretation (`:873`) — so changing the
constant moves all three together and they cannot desynchronise. `36` is assumed
nowhere else load-bearing: the only other occurrences are one test assertion
(`leaf_tests.rs:179`, `assert_eq!(msg.len(), 36)`) and doc headers in the KAT
tables. No canonicity gate and no mode selection reads it.
⚠ **But it cannot be made mode-dependent.** `block_len` is `v[14]`, which
`G_INDICES[2] = (2,6,10,14)` makes the `vd` operand of round-0 G #2, and `vd`
goes straight into `f.xor(g, 0, vd, a1)` (`blake3_chip.rs:328`) — an XOR, whose
byte extraction `WordRef::byte` panics on `ModeSelected` (`:395-404`). So all
three domains move to 52 together: **compress and transcript digests re-bless
too**, and their messages gain sixteen zero bytes — which is what makes the pin
above load-bearing rather than cosmetic.

**Question (iv) — the verified arithmetic.** From the layout constants
(`PREP_WIDTH = 13`, `layout.rs::hash`; `SHARED_VALUE_COLUMNS = 28`,
`chips.rs:494`; `G_SIZE = 60`; `OUT_WINDOW = HASH_DIGEST_FELTS = 4`;
`NUM_G = 8·rounds`), the socket's width is

```
NUM_COLUMNS = PREP_WIDTH + SHARED_VALUE_COLUMNS + 4·NUM_LANES
            + 60·NUM_G + 4·OUT_WINDOW + 2·FELTS_PER_LEAF
```

| | 7r, 8 lanes | 7r, 12 lanes | Δ |
|---|---:|---:|---:|
| lane bytes | 32 | 48 | **+16** |
| canonicity witnesses | 8 | 8 | 0 |
| mixing core | 3,360 | 3,360 | 0 |
| `NUM_COLUMNS` | 3,457 | 3,473 | **+16** |
| main (census) columns | 3,444 | 3,460 | **+16** |
| `AreBytes` sends (`2·NUM_LANES`) | 16 | 24 | **+8** |
| bus interactions | 1,382 | 1,390 | +8 |
| census cells (`main + 3·⌈n/2⌉`) | 5,517 | 5,545 | +28 |

✓ The **+16 columns / +8 sends** in §1.4.1 are exact. What the estimate omits is
the constraint delta — the framing block is renumbered and grows (Breaks 1–3),
at **zero column cost**, since every fix is a constraint or a multiplicity.

**⚠ One correction to §1.5's census table — ✓ APPLIED 2026-08-12.** It listed
`LFM_HASH` at **3,457**, which is `cols::NUM_COLUMNS` *including* the 13
preprocessed columns. Those are committed in the precomputed tree, not the main
tree, so the main-tree row is **3,444** — the figure §1.5's own prose already
named. At `RATE = 4` that is `⌈2·3444/4⌉ = 1,722` compressions rather than 1,729,
i.e. **−7 per query**; the per-query totals move 6,062 → **6,048** and
3,031 → **3,024**. Propagated to §1.4.1's measured-effect table, §1.5's table and
totals, §0's board line, and `commit_kats.py:457`'s width list (two
`C12.per_query_*` vectors re-pinned; no cryptographic vector moved; board still
85/85). ⚠ **This is a re-attribution, not a saving** — the 13 columns are still
absorbed, in the precomputed tree, which this census does not count at all
(§1.5 note **(4)**).

### 1.4.3 ✓ Reconciled: 2,964 vs 3,056 are two DIFFERENT chips

The census track's 3,056 and this document's 2,964 are both correct and measure
different tables — ✓ VERIFIED:

| | file | `NUM_COLUMNS` | `PREP_WIDTH` | main | in the LFM AIR set? |
|---|---|---:|---:|---:|---|
| `LFM_HASH` (Blake3 arm) | `blake3_socket.rs:638` | 2,977 @6r / 3,457 @7r | 13 | **2,964 / 3,444** | **yes** (`chips.rs:592`) |
| standalone BLAKE3 chip | `blake3_chip.rs:162` | 3,072 | 16 (`:151`) | **3,056** | **no** — `airs.rs` never references it |

They are not variants of one chip: the standalone one takes `h`, `t`,
`block_len` and `flags` as *inputs* (`blake3_chip.rs:157`, seven input machine
words), which is the general compression function; the socket pins all four and
reads two cells. **The tower pays the socket's width, so 2,964 @6r is the figure
for every tower number in this document.** ✗ RESOLVED — nothing left open here.

*(Aside, now moot: the standalone chip already carries `h` as a variable input,
so the D7 sketch was buildable — just against a 3,056-column chip reading seven
words, for the same rate 4 the socket reaches with 12 lanes.)*

### 1.4.4 ⚠ Implementation hazard register — read before writing the chip change

Every item is ✓ VERIFIED against `lambda_vm-blake3-impl@blake3-real-hash`. They
apply to the **adopted `RATE = 4` / `NUM_LANES = 12`** construction; D9 moving to
5 would add to this list, not shorten it. **H1 is the one that ships broken.**

| id | hazard | where | fails how |
|---|---|---|---|
| **H1** | ★★ constraint-index collision | `blake3_socket.rs:1304-1318` | **SILENT in release** |
| **H2** | `num_input_cells(Leaf)` = 2 panics the pin emitter | `chips.rs:722-733` | loud panic |
| **H3** | 2nd `LfmMem` receive excludes `MODE_L` | `chips.rs:626` | leaf never reads its felts |
| **H4** | leaf lane/felt offsets shift by one cell | `blake3_socket.rs:680-687`, `:1369` | binds the wrong felts |
| **H5** | `lanes_from_row` is all-or-nothing | `blake3_socket.rs:1084-1093` | witness ≠ AIR |
| **H6** | lanes 0–3 gate must be `mu`, not `digest_mu` | `blake3_socket.rs:1304-1309` | **accumulator unconstrained** |
| **H7** | `admits`' Leaf arm inspects the wrong cell | `blake3_socket.rs:529-537` | prover panic, not rejection |
| **H8** | `LfmHasher::leaf` signature ripples to Test/Poseidon | `hash.rs:109-114` | silent semantic change |
| **H9** | `block_len` 52 is a **tri-domain** re-bless | `blake3_socket.rs:261` | scope under-counted |

> ### ★★ H1 — the lane identities collide with the output pins, and every guard is blind
>
> `eval` numbers its framing constraints by hand: the lane identities are
> `b.emit_base(6 + lane, …)` over `0..NUM_LANES`
> (✓ VERIFIED `blake3_socket.rs:1304-1309`), then the unused-output pins are
> `b.emit_base(14 + j, …)` for `j ∈ 0..8` (`:1315-1318`), the digest
> recompositions `22 + i` (`:1325-1330`), and `UNREAD_IDX = 26` (`:1223`).
> **At `NUM_LANES = 12` the lane block runs 6..17 and overlaps the output pins at
> 14..17.**
>
> `EmitTracker::mark` asserts `"constraint {idx} emitted twice"` — but only under
> `#[cfg(debug_assertions)]`
> (✓ VERIFIED `crypto/stark/src/constraints/builder.rs:492-504`), and the
> workspace declares **no `[profile.release]` override**, so under the house
> convention `cargo test --release` the tracker is a no-op and the second write
> silently overwrites the first (`builder.rs:614-617`). The lane loop runs first,
> so **lanes 8–11 lose their identity entirely and nothing fails.**
>
> **Why every existing guard misses it.** The constraint *count* does not move:
> lane identities go 8 → 12 (+4) while the unread pins go 8 → 4 (−4, per **H2**),
> so `NUM_CONSTRAINTS` (`:1214`), `CORE_IDX` and
> `predicted_constraints(rounds) = 50 + 16·(8·rounds)`
> (`blake3_socket_tests.rs:104-106`) all still hold. `assert_complete` sees no
> gap either, because every index in `0..NUM_CONSTRAINTS` is still written by
> *something*. **The only thing that would have caught it is a debug-only assert
> the release suite disables.**
>
> **And the four constraints lost are exactly the four that matter** — see
> **H6**: lanes 8–11's identity is what pins `m[9..13]` to zero on digest rows.
> Losing it hands the prover four free message words in a Merkle parent.
>
> ⛔ INDEX CORRECTION (2026-08-13, at implementation): the NORMATIVE layout is
> §1.2 / `commit_ref.py` — lanes at `m[0..12]`, **tag LAST at `m[12]`** — so the
> words this aside calls `m[9..13]` are `m[8..12]` in the implemented layout
> (this aside and two other mentions predate the resolution; the substantive
> argument is unchanged — the free pin comes from the lane→COLUMN map, not the
> message index). Resolved toward §1.2 by the RATE-4 implementation.
>
> **Required:** derive the framing indices from `NUM_LANES` instead of the 14/22/26
> literals, **and** add a release-visible test that the emitted index multiset is
> exactly `0..NUM_CONSTRAINTS` with no repeats. The debug tracker is not
> sufficient, because the suite that would run it does not.

**H2 / H3 — the frozen bus contract does move.** Both are stated in full in the
third supersession banner in §1.4.1. In short: `reads_two()` must gain `MODE_L`
and outgrow `Multiplicity::Sum3`, and `emit_unread_input_pins`' `slot = 1` pass
must stop assuming some mode reads fewer than two cells.

**H4 — the felts move from cell 0 to cell 1.** `leaf_lo_lane(i) = 2i` /
`leaf_hi_lane(i) = 2i+1` (`:680-687`) become `4 + 2i` / `4 + 2i + 1`, and the
felt source `b.main(0, cols::IN0 + i)` (`:1369`) becomes `IN0 + 4 + i`.
`fill_canonicity_witness` (`:1100-1115`) reads through the same helpers, so it
follows automatically — which is the trap: fix the helpers and the filler moves
with them, fix `:1369` alone and it does not.

**H5 — the row is a HYBRID and no current code can express one.**
`lanes_from_row` (`:1084-1093`) branches on `MODE_L` and applies **one** reading
to all eight lanes. Construction C needs cell 0 through `lanes_of` (u32 lanes)
and cell 1 through `leaf_lanes` (felt halves) **on the same row**. The same
all-or-nothing shape is in the constraint gating, which is **H6**.

> **★ H6 — the gate re-cut, and the one direction that is a soundness break.**
> §1.4.2's blockquote has this right; here is the exact split and why the
> "+0 canonicity witnesses" claim depends on it.
>
> - **lanes 0–3 → gate on full `mu`.** On a leaf row they are the accumulator; on
>   a digest row they are `a` = `IN0..IN4`. **The same identity is correct for
>   both readings**, which is why one gate serves. This is also the *only* thing
>   that range-checks the accumulator: identity + `AreBytes` forces
>   `IN_lane < 2^32`, which is what "the accumulator lanes are a previous digest,
>   hence `u32` by construction" cashes out to. **Gate these on `digest_mu` and a
>   leaf row's accumulator lanes carry no identity at all — the prover picks the
>   chain's message words freely and the whole leaf chain unbinds.**
> - **lanes 4–11 → gate on `digest_mu`.** On a leaf row they are felt halves and
>   the identity must NOT hold; the halves binding covers them instead.
>
> **A free win worth not throwing away:** at exactly 12 lanes,
> `b.main(0, cols::IN0 + lane)` for `lane ∈ 8..12` lands on the **third input
> cell**, which `emit_unread_input_pins` pins to zero on every digest row. The
> identity then reads `0 = Σ bytes·2^{8k}`, and with the `AreBytes` bound in hand
> that forces all sixteen bytes to zero — so the pin that keeps `m[9..13]` out of
> the prover's hands costs nothing. ⚠ **12 is the last lane count for which this
> works:** at 13, `IN0 + 12` is `S8`
> (✓ VERIFIED `chips.rs:482-488`) and the identity would start reading the
> capacity-state columns as input felts. A D9 move to 14 lanes must supply these
> pins explicitly.

**H7 — `admits` would inspect the accumulator and call it the felts.** The Leaf
arm checks `leaf_lanes` over `state[0..4]` (`:529-537`), which under construction
C is the **accumulator cell**, not the felts. Left as is, a non-canonical felt
passes execution and fails later in the filler or the AIR — a prover panic where
the house rule wants a clean rejection ("reject, never reduce"). It needs to
check `lanes_of(acc)` **and** `leaf_lanes(felts)`.

**H8 — the trait change is not local to BLAKE3.** `LfmHasher::leaf(&self, felts:
&LfmWord)` (`hash.rs:114`) takes one cell; construction C needs `(acc, felts)`.
The default `leaf_out` delegates to `compress_out(felts, &[zero; 4])`
(`hash.rs:109-110`), so **the `Test` and `Poseidon` arms' leaf semantics change
too** — silently, since they compile either way. Both already carry the recorded
weakening that they do not domain-separate leaves from parents; this widens it.

**H9 — `block_len` 52 re-blesses THREE domains, not one.** ✓ VERIFIED it cannot
be made mode-dependent: `block_len` is `v[14]`, which
`G_INDICES[2] = (2,6,10,14)` makes the `vd` operand of round-0 G #2, and `vd`
goes straight into `f.xor(g, 0, vd, a1)` (`blake3_chip.rs:328`) — an XOR, whose
byte extraction `WordRef::byte` panics on `ModeSelected` (`:395-404`). So
`LFMC` and `LFMT` move to 52 with `LFML`: **every Merkle parent and every
transcript step re-blesses, and their messages gain 16 zero bytes** — which is
what makes H6's pin load-bearing rather than cosmetic. Scope this into D8's
re-bless pass, not just the leaf vectors. On the credit side it flows from one
constant (`:261`) into the host reference (`:304`), the wire interpretation
(`:752`) and the value interpretation (`:873`), so the three cannot
desynchronise; and 52 < 64 keeps every row a single block, so the C9 crate anchor
survives for all three domains.

**✓ What is NOT a hazard: the shared dataflow.** Message words reach `add3` and
nothing else — the schedule indices are consumed at exactly two call sites,
`blake3_chip.rs:327` and `:333`, both `f.add3(…)`. `word_expr` (`:1032-1048`)
handles `Cols`, `Const` and `ModeSelected` uniformly at degree ≤ 1, so the sum
identity stays degree 2 under the mu gate and the chip's max degree of 3 does not
move. `Add3Wire.m` is already typed `WordRef` for exactly this reason
(`:439-446`). **The G-block wiring is fully index-agnostic; the mixing core's
sends, constraints and degree are functions of the round count alone.**

### 1.5 ★ Chain-depth census — and the rate problem it exposes

The build-time question §6 flagged, answered. Main-tree widths ✓ VERIFIED from
source (`chips.rs` + `layout.rs`; `keccak_rnd.rs:95`, `keccak_rc.rs:36`,
`bitwise.rs:94`). `LFM_HASH` under Blake3 has `cols::NUM_COLUMNS = 3457`
(`blake3_socket.rs:616-638` with `SHARED_VALUE_COLUMNS = 28`, `NUM_G = 56` at 7
rounds), of which **3,444 are main-tree value columns** — the figure that
reproduces the leaf-impl-report exactly, and the one this table uses.

Per leaf = one row pair, so `felts = 2·num_cols·kind`, `chunks = ceil(felts/4)`,
**chain depth = chunks**, `compressions = 2·chunks` (§1.4).

| chip (main tree) | cols | felts | chain depth | compressions |
|---|---:|---:|---:|---:|
| **`LFM_HASH` (Blake3, 7r)** | **3444** | 6888 | **1722** | **3444** |
| `KECCAK_RND` (per chunk) | 1480 | 2960 | 740 | 1480 |
| `LFM_KECCAK` | 792 | 1584 | 396 | 792 |
| `LFM_BITDEC` | 196 | 392 | 98 | 196 |
| `LFM_SELECT` / `LFM_XALU` / `BITWISE` | 25 / 23 / 21 | | 13 / 12 / 11 | 26 / 24 / 22 |
| `LFM_LANES` / `LFM_BALU` / `KECCAK_RC` | 16 / 14 / 10 | | 8 / 7 / 5 | 16 / 14 / 10 |
| `LFM_CONST` / `LFM_PUBLIC` / `LFM_HINT` / `LFM_RANGE` | 7 / 7 / 6 / 2 | | 4 / 4 / 3 / 1 | 8 / 8 / 6 / 2 |
| **total, main trees, `C = 1`** | | | | **6,048** |

**Four findings, in order of how much they matter.**

**(1) ★ The widest table is the BLAKE3 chip itself, not `KECCAK_RND`.** At 3,444
main columns it is 2.3× `KECCAK_RND` and **56.9%** of the per-query main-tree leaf
cost. The tower spends most of its leaf budget re-absorbing the trace of the
hash chip that made the proof cheap. This inverts the natural assumption and is
a property of the *tower*, not of the base layer — where `KECCAK_RND` still
dominates at 92.5% of cells.

**(2) ★★ The cost formula was not a depth problem, it was a RATE problem —
and §1.4.1 fixes it.** At the old construction, for a base tree
**compressions per leaf ≈ `num_cols`**, because 2 rows × `m` felts cost
`2·ceil(2m/4) ≈ m`. That was **2 felts per compression** — four felts per `LFML`
row, halved by the `LFMC` fold. Keccak absorbs **17 felts per permutation**
(rate 136 B ÷ 8 B), which is why the native scheme was only ~1.7× better on leaf
absorption despite ~14.7× on Merkle parents.

**`RATE = 4` (§1.4.1) takes it to 4 felts per compression, a 2.0× cut.** The
table above is the OLD-rate census, kept because it is what located the problem;
the columns are unchanged, so the new per-chip cost is `0.5 × cols` (e.g.
`LFM_HASH` 3,444 → **1,722**), and the per-query total is **6,048 → 3,024**.

Chain *depth* is fine — 1,722 sequential steps is nothing for a fully-unrolled
straight-line program, and depth carries no soundness cost since the header
binds the shape (§1.3). The compression *count* is the whole story:

| | per query, main trees, `C = 1` | ×219 queries | ×110 queries |
|---|---:|---:|---:|
| leaf-absorption compressions | 6,048 | **1,324,512** | **665,280** |

**(3) It is the Gate D1 lever, and §1.4.1 pulls it.** The Gate D1 verdict
(PLAN.md) independently measures leaf absorption at **69.8% of the tower node
bill** and finds the node fails at 124 GiB against ~93 GiB, with **×2 → fits**.
`RATE = 4` delivers **2.0×** → ≈81 GiB. ⚠ And it is **not** the last turn of this
lever — see D9 (§7) and the second supersession banner in §1.4.1.

**(4) ⚠ This census counts MAIN trees only, and that is a real scope limit.**
The row above is 3,444 rather than `NUM_COLUMNS`'s 3,457 because the 13
preprocessed columns are committed in the **precomputed** tree. They are still
absorbed — this table just does not count them, nor the aux or composition trees
(`verifier.rs:605-650` confirms three separate trees). ✓ The correction was
applied 2026-08-12 (6,062 → 6,048 old, 3,031 → 3,024 new; −0.23%, invisible at
Gate D1's ≈81 GiB). **Do not read the total as the whole tower leaf bill** — it
is the main-tree component of it, which is what §1.4's formula is scoped to.

> ⚠ **The earlier D7 sketch in this section — carry the accumulator in the
> chaining value `h` — is SUPERSEDED and should not be built.** It reached only
> 4 felts/compression, required `h` to become a witness, forced a revisit of
> obligation **O3**, and split the crate-KAT anchor. §1.4.1's in-message
> accumulator is strictly better on all four counts. The premise that made D7
> look necessary — "the socket's eight lanes are full, so an accumulator cannot
> ride in the message" — was **wrong**: it counted the *lanes* the socket
> currently reads, not the *message words* BLAKE3 has, and seven of those are
> `WordRef::Const(0)`.

### 1.5.1 Two build traps this spec must state

**(a) `blake3-6round` is OFF by default.** ✓ VERIFIED `SOCKET_ROUNDS =
BLAKE3_ROUNDS` (`blake3_socket.rs:202`) and `BLAKE3_ROUNDS =
BLAKE3_STANDARD_ROUNDS` unless the `blake3-6round` feature is set
(`blake3.rs:83-85`). Every number in this document is quoted at **7 rounds**,
which is the compiling default and **+16%** on every tower figure. The campaign
intends 6 rounds. **The spec therefore states a build requirement: tower
proving builds must enable `blake3-6round`, and any census that does not must
say so.** At 6 rounds `NUM_G` is 48 rather than 56, so the `LFM_HASH` arm is 2,977 columns
total / **2,964 main** (2,993 total with the §1.4.1 widening) against 3,457 /
3,444 at 7r. ✓ The 3,056 figure circulating on the census track is the
**standalone** BLAKE3 chip, a different table — §1.4.3 reconciles them.

**(b) The `LFM_HASH` chip is the widest table under D0.** Finding (1) above:
3,444 main columns at 7r (2,964 at 6r), 2.3× `KECCAK_RND`, **56.9%** of the
old-rate per-query leaf bill. Each tower layer pays to re-absorb it. `RATE = 4`
cuts the absolute cost 2.0× but does **not** change the share — the chip is still
the widest table, and any future widening of the socket lands on the tower with
that ~57% multiplier. Worth remembering before adding socket columns for anything
else — including the **+16** this very construction adds (§1.4.1), which is
+0.5% on the widest table and therefore ~+0.3% on the whole per-query leaf bill.

### 1.6 What the I3 width check still guards afterward

? INFERRED, and stated conservatively on purpose.

With the header built from the AIR, an opening of the wrong width produces a
different leaf digest and fails authentication — so for the leaf path the check
becomes **defence in depth rather than the primary mechanism**.

It is still needed, for two reasons. First, it runs *before* any opening is
indexed (✓ VERIFIED `verifier.rs:213-215`: "Runs once per table, before any
opening is read"), so it is what stops a malformed opening from being indexed at
all. Second, it pins widths for things no leaf covers — the OOD tables.

> **⚠ Do not delete it as redundant.** This is the M8 lesson from
> TRANSCRIPT.md §3.3, where a paragraph that *looked* like it made the
> registrar's one-hot check redundant was wrong, and a reader who trusted it
> could have removed the load-bearing check while every constraint still passed.
> The same trap is available here. **DECIDED (D2): the check stays** — §7.

---

## 2. The byte→cell absorb encoding

### 2.1 The problem

`DefaultTranscript` absorbs **bytes** (`append_bytes`); B1 absorbs **cells** of
four u32 lanes. `absorb_lfm_statement` feeds raw byte strings — a tag, a
`program_id`, little-endian integers (✓ VERIFIED `statement.rs:79-89`). An
encoding is required, and it must be injective.

### 2.2 The construction

```
header = [ BYTES_MARK, len & 0xFFFFFFFF, len >> 32, 0 ]
body   = data zero-padded to a multiple of 16, each 16 bytes read as
         four LITTLE-ENDIAN u32 lanes
absorb = absorb(header) then absorb(each body cell)
cost   = 1 + ceil(len / 16) compressions
```

**O1 compliance is automatic, and that is the point.** Every lane is exactly
four bytes, hence `< 2^32` by construction — no canonicity gate, no rejection,
no `MODE_L` row. A byte block is already digest-shaped. This is precisely why
bytes take *this* path while field elements take `absorb_felts` (`LFML`), where
the canonicity gate lives. ✓ EXECUTED (**C5**).

**The length prefix is what makes it injective**: without it `b"\x01"` and
`b"\x01\x00"` absorb identically. ✓ EXECUTED (**C5**).

Little-endian to match `word_of`'s stated convention — ✓ VERIFIED
`blake3_socket.rs:441-443`: *"one felt = one u32 = four little-endian bytes"*.
Note this is **not** `pack_digest`'s eight-bytes-per-lane serialization; the two
conventions coexist in the codebase and the same doc comment already warns about
it.

---

## 3. Node codec and tree shape

### 3.1 The embedding

`pack_digest` (✓ VERIFIED `word.rs:44-50`) writes four canonical u64 lanes
little-endian: 32 bytes. Under BLAKE3 every lane is `< 2^32` (✓ VERIFIED
`word_of`, `blake3_socket.rs:443`), so **the high four bytes of each 8-byte
chunk are zero** — sixteen bytes of padding.

That padding is what lets a 128-bit digest ride inside the existing 32-byte
`Commitment` without moving the proof format: `StarkProof`'s commitment fields
and the rkyv derives stay byte-identical (D0 §2). Parents hash **cells**, never
the padded bytes, so nothing enters a preimage that the guest must re-pad.

### 3.2 ⚠ The strict decode (S2)

`unpack_digest` (✓ VERIFIED `word.rs:52-61`) reads each chunk as a u64 and
**reduces mod p**. Many distinct 32-byte strings therefore decode to one node —
any lane may be offset by a multiple of `p`, and far more cheaply, any of the
sixteen padding bytes may be set. Node-level malleability inside a Merkle path
is a proof-format forgery surface.

**The rule: every lane must be `< 2^32`; reject otherwise.** `< 2^32` implies
`< p`, so one test covers both. This mirrors `lanes_of` (✓ VERIFIED
`blake3_socket.rs:431-438`), which already rejects rather than reduces on the
host.

✓ EXECUTED (**C6**), four rejection flavours plus two honest legs: a set high
byte in lane 0, a set top byte in lane 3, a lane congruent to 1 mod p, and a
short commitment all reject; the round trip and an all-zero digest still decode.
The honest legs are not optional — a decoder that rejected everything would pass
a rejection-only suite.

### 3.3 Tree arity and padding (S6)

**Binary, and assert the leaf count is a power of two — do not pad.**

The leaf count is always `lde_size / 2` and `lde_size` is always a power of two
(✓ VERIFIED the prover debug-asserts exactly this, `commitment.rs:67-70`), so
the assertion is always satisfiable on the honest path and costs nothing.
Padding would add a duplicate-leaf second-preimage surface for a case that does
not arise — an unreachable branch that weakens the tree. `HostTree::build`
already asserts the same (✓ VERIFIED `fixture.rs:163-175`). ✓ EXECUTED (**C7**).

---

## 4. Scope-outs, stated in the spec rather than assumed

### 4.1 Grinding under B1 (S7) — **DECIDED: grinding STAYS**

> **Ruling (Mauro, 2026-08-12), verbatim:** *"Grinding should help you, we need
> 128 security for sure."*
>
> This **reverses** an earlier recommendation in this document to set
> `grinding_factor: 0` and scope grinding out. §7.1/§7.2 show that
> recommendation was wrong on the numbers: dropping grinding costs +41 queries
> at blowup 2 — **+222,794 tower permutations per wrap verify, forever**, in
> exactly the cost centre D0 exists to shrink. Grinding is not overhead here; it
> is the cheapest 20 bits in the protocol.

So B1 needs a PoW it can express. This section specifies one.

#### 4.1.1 What it replaces

✓ VERIFIED `grinding.rs:67-89` — **two** keccak256 hashes over byte buffers:

```
inner = Keccak256( PREFIX(8) ‖ seed(32) ‖ factor(1) )                41 bytes
valid = u64_be( Keccak256( inner(32) ‖ nonce_be(8) )[..8] ) < 2^(64−factor)
```

Neither layer is a 2-to-1 compress, and both run through the hosted keccak
family. The seed is `transcript.state()` (`prover.rs:2093`,
`verifier.rs:1587`) — a `[u8;32]` B1 does not have.

#### 4.1.2 The construction

```
GRIND_MARK = "GRD0" as a little-endian u32
N(nonce, factor) = [ nonce_lo, nonce_hi, GRIND_MARK, factor ]        one cell
W                = compress_T( state, N(nonce, factor) )             ONE compress
valid            ⟺  ( W[0] + 2^32·W[1] )  mod  2^factor  ==  0
```

- **One cell, one compression.** The whole PoW is a single `compress_T`, which
  is the design target: verification cost is O(1) in the difficulty.
- **The difficulty is in the preimage.** Without `factor` in the operand a
  prover mines once at factor 1 and presents the result at factor 20.
  ✓ EXECUTED (**C11**): the same nonce at factors 12 and 13 gives different
  digests.
- **The seed is the transcript state cell**, not a 32-byte digest — the B1-shaped
  analogue of today's `transcript.state()` seed. ✓ EXECUTED (**C11**).
- **The difficulty predicate reads `W[0] ‖ W[1]` as one 64-bit value**, covering
  the whole documented `1..=64` range (`grinding.rs:22`) under one rule. For a
  realistic `factor ≤ 32` it touches lane 0 only. The alternative — a lane-0
  rule with a second rule bolted on above 32 — is two cases where one will do.

#### 4.1.3 ⚠ Domain separation — read before changing the tag

The construction **reuses `LFMT`** and allocates no fourth domain. The argument
is the one B1 already relies on for absorb-vs-squeeze, quoting TRANSCRIPT.md
§1.1: the operation sequence is a compile-time constant of the program, so *"a
prover cannot perform a squeeze where the program says absorb"* — and equally
cannot present a PoW evaluation where the program says absorb. `GRIND_MARK` sits
on exactly the same footing as `SQUEEZE_MARK`, which that section is explicit is
**defence in depth, not the load-bearing argument**.

Sharing the tag costs nothing cryptographically: to satisfy the difficulty a
prover must still search operands at a state it does not control, and no
transcript step computed elsewhere helps. It saves a tag, a fifth preprocessed
selector (`MODE_G`), `PREP_WIDTH` 13 → 14, and a registry re-bless.

✓ EXECUTED (**C11**), both cross-domain directions plus the marker: a PoW step
equals neither an `LFMC` parent nor an `LFML` leaf of the same cells, and
dropping `GRIND_MARK` changes the digest.

> ### ⚠ The one separation the hash does NOT provide
>
> **A PoW step and an ABSORB of its operand cell are the same function** — both
> are `compress_T(state, cell)`. Against a Merkle parent and a leaf the tag
> separates them; against a transcript absorb **nothing in the hash does**, and
> `GRIND_MARK` only means an *honest* absorb is unlikely to collide, not that a
> chosen one cannot.
>
> That separation is carried entirely by the program's compile-time operation
> sequence — the same mechanism B1 already accepts for absorb-vs-squeeze. It is
> asserted as an **identity** in **C11** rather than left in prose, so the
> reliance is visible on the board: if D6a is ever taken, that leg flips and says
> so. A reader who wants the separation to hold without the fixed-sequence
> premise wants D6a.

> **✗ OPEN (D6a)** — whether to spend the tag + selector + re-bless anyway, for a
> separation that does not lean on the fixed-sequence argument. Recommendation:
> no, on consistency grounds — if the fixed sequence is good enough for
> absorb/squeeze it is good enough here, and a fifth selector is not free.

#### 4.1.4 The payoff, stated honestly

The guest verifies PoW with **one blake3 compression plus one `LFM_BITDEC` row**
(to expose the low bits), against **two keccak sponge invocations** through the
hosted keccak family.

⚠ **That saving is O(1) per proof and therefore small in absolute terms.** It is
not the reason grinding stays. The reason is §7.1: grinding buys back 41 queries
— ~222,794 tower permutations per wrap verify — for a one-off mining cost the
*prover* pays once. Quoting the compression saving as the justification would
overstate a real but minor effect and understate the actual argument.

### 4.2 Challenge entropy (S8)

> **DECIDED: squeeze-twice (~192-bit).** ⚠ **Decided *by implication* of the
> 128-bit total-security requirement, not by an explicit ruling on this
> question** — 96-bit challenges are below target, so the upgrade follows.
> **Flagged for explicit confirmation** rather than recorded as settled, because
> a decision nobody consciously made is the kind that gets silently reversed.
> Priced in §7.1 at **254 permutations** per tower wrap verify — about two orders
> of magnitude below D3. The analysis below stands as written; only the "not
> recommended either way" framing is resolved.

The ratified B1 `squeeze_ext` takes lanes 0–2 of one squeezed cell. Each lane is
a u32, so an extension challenge carries **96 bits**, against the ~192 that
`DefaultTranscript` delivers (three near-full Goldilocks coordinates,
✓ VERIFIED `extensions_goldilocks.rs:575-581`). TRANSCRIPT.md §4.1 bounds the
*state* (128 bits, ~64-bit collision) but does not analyse *per-challenge*
entropy at production query counts.

**The alternative, costed** — `squeeze_ext_2` in the reference:

```
c0, c1 = squeeze(), squeeze()
coef_i = ( lanes[2i] + 2^32 · lanes[2i+1] ) mod p      i ∈ 0..3
```

- **Cost: a flat +1 compression per extension challenge** (2 instead of 1).
  ✓ EXECUTED (**C8**).
- **Query-index sampling is unaffected**: `squeeze_bits` reads lane 0 only, so
  the query loop — the dominant squeeze run — pays nothing.
- **No rejection loop, deliberately.** A uniform 64-bit value reduced mod p is
  biased by ~2^-32, negligible for a Fiat–Shamir challenge. An exact rejection
  loop is *unimplementable* in the fully-unrolled eDSL ("nothing loop-shaped
  reaches the machine", TRANSCRIPT.md §1.1 citing `edsl.rs:1-4`). The bias is
  the right trade and the reason is structural.

> **DECIDED (D4): squeeze-twice.** 96 bits is below the 128-bit total-security
> requirement, so the upgrade follows by implication — see the banner above for
> why that is flagged for confirmation rather than filed as settled.

---

## 5. What this construction does *not* change

- **No new socket tag.** The wide leaf is `LFML` rows folded by `LFMC` parents —
  the two domains LEAF.md already ratified. `LEAF_MARK`/`BYTES_MARK` are **lane
  constants inside a header cell**, not `m[8]` tags, so no new hash domain is
  created and no new domain analysis is owed. (**DECIDED D1**, §7.) The B1 PoW
  (§4.1) follows the same rule, reusing `LFMT` with a `GRIND_MARK` lane constant.
- ⛔ ~~**No change to the chip.** Nothing here needs a constraint that does not
  already exist; the wide leaf is a *program shape* built from existing rows.~~
  **RETRACTED — this was true only of the pre-§1.4.1 construction.** Adopting a
  `RATE > 2` puts the accumulator in the message, and that **is** a chip change:
  `NUM_LANES` 8 → 12, `block_len` 36 → 52, the lane/message identity re-cut per
  lane range, `reads_two()` gaining `MODE_L`, and `emit_unread_input_pins`
  restructured. Nine verified hazards in §1.4.4. The bullet is kept because it is
  what §5 promised before D8 existed, and a reader who takes §5 as the change
  budget would under-scope the work by an order of magnitude.
- **The crate anchor survives.** ✓ EXECUTED (**C9**): the `LFML` rows are still
  byte-identical to a plain `blake3::hash` call at 7 rounds — now of a 52-byte
  message rather than 36. ⚠ There is no longer a *fold*: the accumulator rides in
  the message (§1.4.1), so an `LFML` chain is a sequence of rows and not rows
  plus `LFMC` parents. The `LFMC` socket is unchanged as a *function*, but its
  `block_len` moves to 52 with everything else (§1.4.4 **H9**), so Merkle-parent
  and transcript digests re-bless too.

---

## 6. Open items that are *build-time*, not decisions

| item | status |
|---|---|
| the same vectors against the Rust `blake3` crate | ✗ DEFERRED — needs cargo |
| the exact production grouping of precomputed/main/aux trees per AIR | ✓ VERIFIED as three separate trees (`verifier.rs:605-650`); the per-AIR widths still need reading off `trace_layout` at build time |
| whether any LFM AIR has `num_cols` large enough to make the chain depth a cost concern | ✓ **DONE — §1.5.** Depth is a non-issue; the **rate** was, and §1.4.1 fixes it (**✗ OPEN D8**, and how far to push it is **✗ OPEN D9**) |
| the end-to-end cost of a `RATE = 5` leaf: extra `LfmMem` traffic, the ~60%-larger padded opening buffer, and the re-packing `Pack`/`Unpack` rows | ✗ **OPEN — this is what D9 needs.** §1.4.2a's ~19% is a ? INFERRED sketch over trace columns only; the leaf program does not exist yet, so nothing here is measured |
| a release-visible test that the emitted constraint-index set is exactly `0..NUM_CONSTRAINTS` with no repeats | ✗ **REQUIRED by §1.4.4 H1.** The existing `EmitTracker` duplicate check is `#[cfg(debug_assertions)]` and the house convention runs `cargo test --release` |

---

## 7. Decision points

> **Status 2026-08-12 — D1–D6 are RULED ON by Mauro** (D4 by implication, see
> its row). **D8, D9 and D6a remain open.** The construction itself (§1, the S1
> wide leaf) stays DRAFT pending his read. The quantification in §7.1 was
> produced *after* the D3/D4 rulings and is recorded because it **supports**
> them — and because it falsifies this document's own earlier recommendation
> on D3.
>
> **D9 is new (2026-08-12) and it exists because this document was wrong twice
> about the same number.** The first RATE draft said 5 from block headroom while
> ignoring the machine's cell structure; the correction said 4 and justified it
> with a cell argument that does not follow. Both retractions are marked in place
> in §1.4.1 rather than edited away, because the pattern — *an open item carried
> into a recommendation stops looking open* — is the one §7.2 already names, and
> this is its second instance in the same section.

| id | decision | **ruling** | provenance / note |
|---|---|---|---|
| **D1** | `LEAF_MARK`/`BYTES_MARK` as lane constants, or genuine `m[8]` socket tags? | **lane constants** | Mauro: *"do whatever feels simpler."* Matches the recommendation (§5). No new hash domain, no new domain analysis owed. |
| **D2** | Does the I3 `trace_opening_widths_well_formed` check stay after the header lands? | **stays** | Uncontested. As recommended (§1.6). The M8 lesson from TRANSCRIPT.md §3.3 holds: a mechanism that *looks* like it subsumes a check is how a load-bearing check gets deleted. |
| **D3** | Grinding: `grinding_factor: 0`, or compensate with more queries? | **grinding STAYS** | Mauro, verbatim: *"Grinding should help you, we need 128 security for sure."* ⚠ **Reverses this document's own earlier recommendation**; §7.2 records why it was wrong. Consequence: the B1 PoW is now **specified** in §4.1, not scoped out. |
| **D4** | Keep 96-bit `squeeze_ext`, or pay +1 compression for ~192-bit? | **squeeze-twice (~192-bit)** | ⚠ Decided *by implication* of the 128-bit requirement, **not** by an explicit ruling — flagged for confirmation (§4.2). §7.1 prices it ~2 orders of magnitude below D3. |
| **D5** | Zero-pad the felt stream to 4, or forbid non-multiple-of-2 widths? | **zero-pad** | Mauro: no opinion → the recommendation stands (§1.3). |
| **D6** | Grinding needs a B1-expressible PoW | **specified** (§4.1), KATs **C11** | Raised by this document after the D3 ruling; discharged in the same pass. |
| **D8** | ★★ Adopt `RATE = LFML_FELTS_PER_ROW = 4` (§1.4.1)? | ✗ **OPEN — the consequential one** | Decides whether the tower fits: 2.0× on 69.8% of the node bill, Gate D1 124 GiB → ≈81 GiB. A **chip** change (`NUM_LANES` 8 → 12, `block_len` 36 → 52) costing ≈+16 columns on 3,444 main (+0.5%). ⛔ "4 is the CEILING" is **RETRACTED** — the ceiling is 5, see **D9**. Moves every `LFML` digest → re-bless — and ⚠ **`LFMC` and `LFMT` too**, since `block_len` cannot be made mode-dependent (§1.4.4 **H9**). D0 is re-blessing anyway **if sequenced into the same pass**. Supersedes the D7 `h`-chaining sketch (same rate, breaks the C9 anchor). **Before implementing, read §1.4.4** — 9 hazards, H1 silent in release. |
| **D9** | ★★ Stay at `RATE = 4`, or price and take `RATE = 5`? | ✗ **OPEN — optimization, NOT fit** | **Nothing is blocked either way: Gate D1 already fits at rate 4 (≈81 GiB vs ~93 GiB, ~13% margin).** This is about whether to chase ~19% more. **Rate 4 (adopted default):** buildable today, fully priced (+16 cols, +8 sends, all costs in §1.4.1/§1.4.4), felt input is one whole machine cell so the leaf program reads the opening in its natural layout with **no re-packing pass**. Needs the multiplicity change (H3), the pin-emitter fix (H2), the per-lane-range gate (H6) and the rest of §1.4.4. **Rate 5:** the true block ceiling (16 words − tag − 4 accumulator lanes = 5 half-pairs); ? INFERRED **~19% cheaper** per §1.4.2a (2). ⚠ **UNPRICED, and the gaps are not small:** the extra `LfmMem` traffic, the ~60%-larger padded opening buffer, and the re-packing program's own `Pack`/`Unpack` rows. Needs the **third** receive to admit `MODE_L` plus either a rotating per-row lane map or a repacked felt stream, and §1.4.4 **H6**'s free digest-row pin stops working at >12 lanes. **Recommendation: ratify 4 now, and treat 5 as a follow-up only if Gate D1's 13% margin proves too thin** — taking it later costs a second re-bless, which is the same sequencing argument D8 makes. |
| **D6a** | Give the PoW its own `LFMG` tag + `MODE_G` selector? | ✗ **OPEN** | Recommendation: no (§4.1.3). Costs `PREP_WIDTH` 13 → 14 and a re-bless for a separation the fixed-sequence argument already carries — the same argument B1 accepts for absorb/squeeze. **But note what it buys:** a PoW step *is* an absorb of its operand cell (C11's identity leg), so PoW-vs-absorb separation is the one direction the hash does not give you. Take D6a if that premise should not be load-bearing. |

### 7.1 The D3 / D4 arithmetic

✓ VERIFIED formula — `options.rs:121-125`:

```
rate           = 1 / blowup
proximity      = 1 − sqrt(rate) − 1/300
bits_per_query = −log2(1 − proximity)
queries        = ceil( (security_bits − grinding_factor) / bits_per_query )
```

✓ EXECUTED, and the formula reproduces the recorded presets exactly (219 at
blowup 2, 110 at blowup 4 — `prover/src/recursion.rs`'s `Blowup2`/`Blowup4`),
which is what makes it safe to extrapolate:

| blowup | bits/query | q @ grinding 20 | q @ grinding 0 | Δ |
|---|---|---|---|---|
| **2** | 0.493215 | **219** | **260** | **+41 (+18.7%)** |
| 4 | 0.990414 | 110 | 130 | +20 (+18.2%) |

**Is `grinding_factor: 0` admissible?** ✓ VERIFIED **yes**, on both counts:
`security_bits <= grinding_factor` is `128 <= 0` = false, so `with_params`
returns `Ok` (`options.rs:114-119`); and every grinding call site is gated on
`security_bits > 0` (`prover.rs:2092`, `verifier.rs:1584`, `verifier.rs:1666`),
so `grinding.rs:22`'s `debug_assert!((1..=64).contains(..))` is never reached.
Admissible — just expensive.

**The cost of dropping grinding, three ways** (blowup 2, 128-bit):

| axis | effect |
|---|---|
| **(a) proof size** | +18.7%. Every query contributes, per tree, `evaluations ‖ evaluations_sym` plus a Merkle path; all of it scales linearly in query count. |
| **(b) prover work** | +18.7% on the query phase (path gathers, FRI openings). Total prover time grows by *less*, since LDE and commit are query-independent. |
| **(c) ★ tower verifier permutations** | **+222,794 per wrap verify** — 41 extra queries × **5,434 leg permutations per query** (✓ measured, `CENSUS.md` §2b, closed-form checked against `epoch_verify::query_permutations`). |

For scale: `CENSUS.md` §3 records that the 93 GiB box budget holds ~**40.8k
permutations in total**. The grinding-0 delta *alone* is ~**5.5× the entire box
budget** — and the tower pays it on every layer, for every proof, forever.

> ⚠ **Caveat on the absolute figure.** 5,434 perms/query is the *epoch*
> verifier (keccak inner) at 2^21/blowup2. The tower's LFM-proof verifier has
> not been censused (Gate D1). The **+18.7% is exact and hash-independent**; the
> 222,794 is an order-of-magnitude anchor from the nearest measured shape.

**The cost of D4's squeeze-twice.** ✓ VERIFIED the extension challenges a verify
actually samples:

- per table — `beta` (`verifier.rs:1470`), `z` (`:1501`), `gamma` (`:1535`), and
  `zetas` = one per committed FRI root (`:1557`) plus one final-fold challenge
  (`:1572`), i.e. `total_folds` of them;
- once per multi-proof — `LOGUP_NUM_CHALLENGES = 2` (`lookup.rs:105`,
  `verifier.rs:1312`).

Everything else is free: the boundary, transition, trace-term and DEEP
coefficients are **powers** of `beta`/`gamma` (`verifier.rs:1538-1541`), not
squeezes, and query indices go through `sample_u64` → `squeeze_bits`, which
reads lane 0 only and needs no extra entropy.

`total_folds = lde_log − min(blowup_log + k, lde_log)` with `k = 7`
(`fri/terminal.rs:45-55`), so a 2^22-row table at blowup 2 gives
`num_committed = 14` — ✓ consistent with `CENSUS.md` §3's observed "14 FRI
layers" at that shape.

| shape | ext challenges | **Δ permutations (+1 each)** |
|---|---|---|
| tower: 14 LFM tables, 2^22 rows | 14 × 18 + 2 | **254** |
| epoch verifier: 64 sub-proofs, 2^22 rows | 64 × 18 + 2 | **1,154** |

### 7.2 The conclusion, and a correction

**D3 costs ~200–900× what D4 costs** (222,794 against 254–1,154 permutations per
wrap verify). Both rulings take the cheap-per-bit option: keep the security
grinding buys for a flat one-time PoW, and buy the entropy `squeeze_ext` lacks
for ~254 permutations.

> **§4.1's recommendation ("`grinding_factor: 0`; grinding out of scope") was
> wrong, and this section is why.** It reasoned qualitatively — *a guest
> recomputing keccak PoW defeats the purpose* — which is true but answers the
> wrong question. The right comparison is one PoW recomputation per wrap verify
> against 222,794 extra permutations per wrap verify, forever, in exactly the
> cost centre D0 exists to shrink. The correct move is the one Mauro took: keep
> grinding and **specify a blake3 PoW**, so the guest recomputes a cheap PoW
> instead of paying 41 extra queries.
>
> The lesson is the one this campaign keeps relearning: an open item carried
> into a recommendation stops looking open. §4.1 recorded the compensating-query
> cost as "not quantified here" and then recommended anyway.

**D6 — discharged in this pass.** The B1 PoW the D3 ruling requires is specified
in **§4.1**, with a domain-separation argument (§4.1.3) and thirteen KAT legs
(**C11**), including both cross-domain directions and the factor/seed/marker
bindings. One sub-question stays open, **D6a**: whether to give the PoW its own
tag and selector rather than lean on the fixed-sequence argument.

---

## 8. Files

| file | what |
|---|---|
| `commit_ref.py` | the reference — wide leaf, byte absorb, node codec, tree, both squeeze options |
| `commit_kats.py`, `commit_kats.json` | C1–C10 + 13 pinned vectors, both round counts |
| `run-kats.log` | the executed board, 54/54 |

Imports resolve relatively to `../../lfm-real-hash/{gate-oracle,leaf-spec,transcript-spec}`;
no absolute paths, no worktree assumptions.
