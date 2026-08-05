# Transcription audit — does the BLAKE3 gate assert more than the design delivers?

Adversarial, one-directional audit of `blake3-chip/z3_blake_verify.py` against
(a) the oracle `blake3-oracle/blake3_ref.py` and (b) the constraint system
`blake3-chip/DESIGN.md` specifies. Branch `spike/blake3-recovered`.

Only one direction is dangerous. A model **weaker** than the object yields a
spurious SAT — a false alarm. A model **stronger** yields UNSAT on an object
that is genuinely forgeable — false assurance — and no positive control can see
it, because an honest witness satisfies a correct model and an over-strong model
equally well. The three positive controls and the 35-vector oracle anchor are
blind to exactly this.

Reproduce: `python3 audit_gate_transcription.py` (~4 min, 73 checks) or
`--slow` (+ the gate's own BV UNSATs, ~8 min). Every check is paired with a
tamper that must make it fail; a check that does not bite is itself reported as
a failure. No file outside this audit was modified — tampers are applied to
in-memory copies and reverted.

---

## Verdict

**One over-strong assertion, and it is the load-bearing one: the gate declares
byte-ness rather than deriving it.** Every committed cell in the model is a
`BitVec(…, 8)`, so the range check that DESIGN §4.3/§4.4/§5/§7.4 make the whole
soundness argument rest on is an *axiom of the model*, not something the model
can observe being present or absent. The gate proves the same UNSAT for the
designed chip and for a chip with **no range checks at all**.

I checked whether that gap is live. It is not, today: I traced every SSA value
in `build_g` and `build_compress` mechanically and the design's premise holds —
all 288 add/shift outputs of a 6-round compression are consumed by a `ByteAlu`
XOR, for ROUNDS ∈ {1,2,6,7}. So the gate's UNSAT is *correct*, for a reason the
gate does not model. It would stay UNSAT under three specific deviations, one of
them invited by DESIGN §1.1 itself.

Transcription (a) — `bref_*` vs the oracle — is **exact**, mechanically, on every
element the brief named.

| # | severity | what |
|---|---|---|
| **F1** | **high** | **Model stronger.** Byte-ness is the `BitVec(…,8)` declaration. The gate cannot distinguish a chip that range-checks an add output from one that does not, nor a chip with §4.7's 32 message `AreBytes` sends from one without them. The premise holds today; nothing in the gate would notice it becoming false. |
| **F2** | medium | **Model weaker / evidence over-stated.** The width audit's `DROP SLL bound → SAT` control is run on a *single-halfword fragment*. Composed with the second identity, the recombine and the downstream byte check, dropping one `SLL` bound is **UNSAT** and dropping both is forgeable at **exactly one input**, `X = 0xFFFFFFFF`. §7.3's stated mechanism is also backwards. |
| **F3** | medium | The "covers every G, hence every round" chaining argument (§9, MAIN 0) **is** the free-range-check argument in disguise, and the gate models neither. It is prose on both counts. |
| F4 | low | μ, padding rows, the degree ledger, and the bus layer are not carried at all. Scope, not error — but §7.1's "μ-gate every eval constraint" and §4.8's degree table are cited as gate-backed and are not. |
| F5 | cosmetic | Three documentation defects (§7.3's mechanism, §4.8's recombine degree row, §3-vs-§4.3 on whether add2 commits a carry column). |

Along the way, two results the gate did not have:

* The rotation is pinned to `rotr12` / `rotr7` **for all 2³² inputs in the
  field**, not just in BV and not just at one concrete halfword.
* The necessary *and sufficient* `AreBytes` set for a rotation is **one of
  `{SLL_lo, SLL_hi}`**. The two `SLLC` bounds and one of the two `SLL` bounds
  are not load-bearing — 3 of the 4 sends per rotation, i.e. 288 of the design's
  1,250 sends per compression.

---

## F1 — the free range check is declared, not derived

This is the claim DESIGN §7.4 flags as soundness-critical and the one the gate is
cited to discharge:

> §4.3 `s`'s bytes are range-checked **free** by the next XOR that consumes `s`
> §5   Every committed add/shift word is an operand of a later XOR ⇒ its bytes
>      are byte-range-checked for free by that `ByteAlu` lookup.
> §7.4 **Every add/shift output must actually feed a downstream XOR** (its only
>      range check). If a future refactor reorders so an add output is *last*
>      with no XOR consumer, add an explicit AreBytes or the carry argument is
>      unsound.

The model's answer to "where does `s ∈ [0,2^32)` come from" is
`z3_blake_verify.py:133-134`:

```python
def fresh_word(self):
    return [self._fresh(8) for _ in range(4)]
```

and `:128-129`, `self._fresh(w=8)` → `BitVec(f"{tag}_v{n}", w)`. `add2`
(`:176-184`) and `add3` (`:186-197`) both begin `s = self.fresh_word()`. So `s`
is four 8-bit bitvectors *by construction*. The audit confirms it mechanically:
every cell the model creates is width 8, and every constraint it emits is `=`
(xor / sum / shift / recombine) or `or` (carry booleanity) — 1,408 and 288
respectively for a 6-round compression. **There is no range-check object in the
model to be present or absent.**

**In fairness, the gate says it is doing this.** The `Circuit` class header
(`:118-119`) reads *"A `word` is a list of 4 free 8-bit BVs … Byte width == the
ByteAlu/AreBytes range-check contract"*, and the module docstring (`:18-21`)
lists `ByteAlu[XOR]` and `AreBytes` under "Chip contracts assumed". So this is a
**documented assumption, not a hidden one** — unlike the EC audit's F2, nothing
here is detected from a comment. Inside the class, "AreBytes" occurs only in
those two comments (`:119`, `:212`); the only place the gate *models* an
`AreBytes` bound is `field_shift_bound` (`:388`), and only for the isolated
fragment F2 is about.

The problem is that the equation "byte width == the range-check contract" is
applied **uniformly to every column**, including the ones for which the design
must separately arrange the contract (§4.2's and §4.7's explicit sends, §5's
downstream-XOR argument) — and DESIGN §9 then cites the gate's UNSAT as
establishing what the comment assumed. The gate's own docstring (`:5`) states
the method as "Every committed column of the designed chip is a FREE bitvector";
it is a free *byte*, and byte-ness is the property under proof.

### Which gate rows this invalidates

| DESIGN §9 row | as written | what it actually establishes |
|---|---|---|
| MAIN 0, one G, free inputs, **UNSAT** | "the quarter-round … is **correctly & tightly constrained**" | correctly & tightly constrained **given that its inputs and its add outputs are byte-range-checked** — which is §7.4/§7.5's obligation on the Rust, not a result |
| MAIN 1, rounds=0, **UNSAT** | "`v` layout … and the feed-forward are correct" | same caveat for `h`, `t_lo/t_hi`, `block_len`, `flags` |
| §9 "Proven (symbolic, all inputs): the G quarter-round … hence, by the chaining argument, the full N-round compression" | unconditional | conditional on the same premise — see F3 |

### Where the missing range check actually bites

The gate declares byte-ness on *every* cell, so I checked, per SSA class,
whether the range check is load-bearing at all. Three of the six classes do not
need one; the model's blanket declaration hides that distinction too.

| SSA class | per G | model's source of byte-ness | chip's source (DESIGN) | is it load-bearing? |
|---|---:|---|---|---|
| `add2`/`add3` output `s` (A1, C1, A2, C2) | 4 words | `BitVec(…,8)` | downstream `ByteAlu` (§4.3/§4.4/§5) | **YES** — without it the chip may commit the **unreduced** sum |
| `ByteAlu` XOR output (X1..X4) | 4 words | `BitVec(…,8)` | the lookup itself | n/a — pinned by contract |
| `SLL_lo`, `SLL_hi` | 2 halfwords/rot | `BitVec(…,8)` | explicit `AreBytes` (§4.2) | **YES, but only one of the two** (F2) |
| `SLLC_lo`, `SLLC_hi` | 2 halfwords/rot | `BitVec(…,8)` | explicit `AreBytes` (§4.2) | **no** — proved |
| rotation output `Y` (B1, B2) | 1 word/rot | `BitVec(…,8)` | downstream `ByteAlu` (§5) | **no** — the two recombine identities pin its *value* even with free field cells |
| message `m` | 16 words | `BitVec(…,8)` | explicit `AreBytes`, §4.7 | **YES** for cell-binding |
| `h`, `t_lo/t_hi/block_len/flags` | 12 words | `BitVec(…,8)` | XOR consumers (§4.7) | for cell-binding |

Each row is an executable check in `audit_gate_transcription.py` §B/§C.

### The forgery it hides — construction

**(1) An add output without its ByteAlu consumer.** This is verbatim the §7.4
deviation. Model the design's add in the field instead of in BV: `s` is four
Goldilocks cells, `carry ∈ {0,1}`, and the only constraint is §4.3's sum
identity.

```
a = b = 0x80000000
honest :  carry = 1,  s = 0                cells [0, 0, 0, 0]
forged :  carry = 0,  s = 2^32             cells [2^32, 0, 0, 0]
```

Both satisfy `a + b ≡ s + 2^32·carry (mod p)` and the booleanity. The add no
longer computes mod 2^32; the chip's `v[c]` is off by exactly 2^32, and the
error propagates through every downstream add. `audit_gate_transcription.py` §C
runs it and reports both `add2` and `add3` FORGEABLE with the range check
removed, PINNED with it, symbolically over all operands. The BV model reports
UNSAT in both worlds because there is nothing to remove.

**(2) §4.7's message range check — the one explicit input `AreBytes` in the whole
design (32 sends).** The gate models `m = [cir.fresh_word() for _ in range(16)]`
(`:332`), i.e. 64 declared bytes. Drop the sends and the 64 cells are free field
elements entering only through `wval(M) = Σ m_i·2^{8i}`, so the chip binds the
*value*, not the bytes:

```
m0 honest : [0x9A, 0x00, 0x13, 0x7F]
m0 forged : [0x19A, p−1, 0x13, 0x7F]        same value mod p
```

Every constraint in the circuit is satisfied identically and the compression
output is bit-identical. The chip proves "the compression of these 64 cells" and
there are `p^3` cell-vectors per message word. Whether this becomes a Merkle
collision depends on the caller — for the §1.1 internal bus the receive tuple
carries the cells, so a byte-constrained counterparty pins them, and for the
§1.2 memory variant MEMW does. **The two gaps compose**: the gate models neither
the `AreBytes` nor the bus, so the obligation is discharged by nothing.

**(3) The CV-only feed-forward trim — the deviation the design invites.**
`h[0..4]` land in round-0 `a` slots, and G uses `v[a]` only as an **add
operand**; it is never XORed. Their sole `ByteAlu` consumer is the *upper*
feed-forward half `out[i+8] = v[i+8] ⊕ h[i]`. The audit measures this: consumer
counts `[1,1,1,1,2,2,2,2]`, and for `h[0]` that single consumer is op #577 of 592
— i.e. the feed-forward, not a round XOR.

DESIGN §1.1 says: *"CV-only call sites read `out[0:8]`; the chip always produces
all 16 (the XOF root needs them)"*. For the internal 2-to-1 Merkle variant — the
primary target — an implementer dropping `out[8:16]` saves 32 committed cells and
32 sends (≈80 cell-equiv, 1.6%) and removes `h[0..4]`'s only range check. §7.4
does not cover it, because it speaks of *add/shift outputs* and `h` is an input.
The gate reports the same UNSAT.

### Fix

Give the model a range-check object. Concretely: have `Circuit` allocate cells
that are *not* byte-bounded by default and add an explicit `are_bytes(word)` /
`byte_alu_xor()` that imposes the bound, so that "this word has no consumer" is
representable and shows up as SAT. That is a rewrite of the model into the field
(z3 `Int` mod p, as `audit_gate_transcription.py` §C does per-op) — but a cheaper
90% is available: keep BV and add the **structural** check this audit implements,
asserting that every `add`/`rotr` output word appears as a `xor` operand and that
`m` is explicitly range-checked. Twenty lines, and it is the check the design's
§7.4 actually asks for.

---

## F2 — the width audit's shift control is run on a fragment

`field_shift_bound` (`:380-395`) models **one halfword identity in isolation**:
`in_hw·2^r ≡ SLLC·2^16 + SLL (mod p)`, `SLLC ∈ [0,2^16)`, `SLL` unbounded, at one
concrete `in_hw = 0x9C3A` and one `r = 9`. It reports SAT and §9 renders that as

> audit: **DROP `SLL` bound** (field neg ctrl) | **SAT** | without it the
> rotation is forgeable

The chip does not contain that fragment. It contains **two** shift identities,
**two** recombine identities, and `Y` byte-range-checked by the downstream XOR.
Composing them (`audit_gate_transcription.py` §C2, symbolic over all 2³² inputs)
gives a different picture:

| bounds kept | r=4 (rotr12) | r=9 (rotr7) |
|---|---|---|
| all four | PINNED | PINNED |
| `SLL_lo` only | **PINNED** | **PINNED** |
| `SLL_hi` only | **PINNED** | **PINNED** |
| `SLLC_lo` and/or `SLLC_hi`, no `SLL` | FORGEABLE | FORGEABLE |
| none | FORGEABLE | FORGEABLE |

All 32 configurations were also checked non-vacuous (the honest witness
satisfies each). So:

* **dropping one `SLL` bound is not exploitable at all** — the gate's control
  claims it is;
* **dropping both** is exploitable at **exactly one input**. Enumerated
  exhaustively, for both `r`:

```
X = 0xFFFFFFFF     honest Y = 0xFFFFFFFF     forged Y = 0x00000000
SLL_lo = SLL_hi = p − 2^r   (i.e. honest − 2^16 as a field element)
SLLC_lo = SLLC_hi = 2^r     (i.e. honest + 1)
```

Every other `X` is UNSAT. The defence is still necessary — a prover can grind an
intermediate XOR output to `0xFFFFFFFF` cheaply, and there are 96 rotation slots
per compression — but "forgeable" at one point is not what the control shows, and
the control shows it for a chip that does not exist.

**§7.3's mechanism is backwards.** It says *"dropping it makes the rotation
forgeable (a wrong `SLL` admits a **large field SLLC**)"*. `SLLC` is bounded to
`[0,2^16)` by its own `AreBytes` and stays small in the forgery (`2^r`); it is
`SLL` that goes large (`p − 2^r`). The witness above is the counterexample to the
prose, not to the conclusion.

**Cost consequence, flagged not pursued:** §4.2 spends 4 `AreBytes` sends per
rotation. One suffices in this composed model. At 96 rotations that is 288 of
the design's ~1,250 sends per compression → ≈432 aux cells ≈ **8.6% of the 5,030
cell-equiv budget**. Before acting on that, note it depends on `Y` being
byte-checked by its consumer — i.e. on F1's premise — and on `2^{-16} mod p`
being large, which is exactly the kind of implicit structural fact this audit
exists to distrust. It wants its own gate row, not a code change.

By contrast `field_add_carry` (`:398-413`) **is** faithful: dropping the carry
booleanity is forgeable even with `s` byte-range-checked (verified composed), and
its UNSAT direction holds for all `(a,b,m)`, not just the one concrete triple it
tests. Both width-audit positives were re-derived symbolically and hold
universally.

---

## F3 — "covers every G, hence every round" is the same argument as the free range check

§9's MAIN 0 row carries the whole default run:

> **covers every G, hence every round** (a round is a fixed composition of 8
> G-calls).

MAIN 0 proves: *for all byte-valued `v[a],v[b],v[c],v[d],mx,my`, the G's four
outputs equal `bref_g`*. Composing it needs each G's **inputs** to be
byte-valued, which holds because they are the previous G's outputs, which are
byte-valued because of the downstream-XOR range check. **The chaining argument
and the free-range-check argument are one argument.** The gate models neither:
the chaining is prose in §9, and the range check is the `BitVec(…,8)`
declaration.

What *is* checkable, and what this audit checks mechanically because the gate
does not:

* `build_round` calls `build_g` on the 8 quadruples of `G_CALLS`, in order, all
  7 rounds — compared against the quadruples recovered by instrumenting the
  **oracle's** `round_fn`, not against the gate's own constant.
* `build_compress` feeds every G the original message column under `permute^r`
  — all 7 rounds × 8 calls = **56 index pairs**, compared against the oracle's
  permutation composition. Both tampers (a swapped `MSG_PERMUTATION` entry, a
  swapped `mx/my` in `G_CALLS[5]`) are detected.
* Every G quadruple has four distinct state indices, the 8 calls touch each of
  the 16 slots exactly twice and consume each message index exactly once — so
  MAIN 0's `a,b,c,d = 0,1,2,3` instance really is general.

README finding 2 ("in Rust, 48 G instances are emitted separately and a wrong
column index in instance #37 is not covered") stands and is out of scope: there
is no Rust.

---

## F4 — what the model does not carry at all

Not errors; scope. Listed because DESIGN cites the gate for some of them.

| DESIGN claim | modelled? | consequence |
|---|---|---|
| §4.5 / §7.1 "every eval constraint is μ-gated; padding rows all-zero" | **no** — no μ variable exists; the docstring says "here mu=1 (a real row), so mu drops out" | exact for a live row. The gate says nothing about padding rows, so §7.1 is unbacked. Note the *ungated* system is strictly **stronger** as a system over all rows — it is only the single-live-row scope that makes this safe. |
| §4.8 degree ledger; the O1 (a)-vs-(c) decision | **no** — the model has no degree notion | `check_g` would be equally UNSAT for the rejected ternary-carry option (a). (a) is rejected for degree, not soundness, so this is harmless — but §4.8 is not gate-backed. |
| §1.1 `Blake3` bus, `Multiplicity::Column(MU)`, `TIMESTAMP_0/1` | **no** — confirmed absent: no `Multiplicity`, `TIMESTAMP`, `receive`/`send` anywhere in the file | README finding 1 (the missing input↔output timestamp binding) is invisible to the gate. Confirmed, not re-derived. |
| §4.1 "operands may be linear combos (sum ≤ 255)" | unexercised | `rotr16`/`rotr8` are pure index relabels, so every modelled operand is a single cell. Consistent with the design's actual use. |
| `block_len ∈ [0,64]`, `flags ∈ [0,128)` | modelled as free 32-bit words | model **weaker** — safe, and DESIGN specifies no such constraint either. |

The BITWISE contracts the gate *assumes* were cross-checked and are real:
`prover/src/tables/bitwise.rs:351-364` enumerates `x,y ∈ [0,256)` and sets
`cols::XOR = x ^ y`; the `ByteAlu` receiver is at `:903-920` and the `AreBytes`
receiver at `:781-796`, both over that domain. DESIGN's cites (`:903`, `:783`)
are accurate.

---

## The assertion tables

Verdicts: **match** = model = object; **stronger** = model asserts more;
**weaker** = model omits; **not modelled** = outside the model entirely.
`file:line` refers to `blake3-chip/z3_blake_verify.py` unless noted.

### (a) `blake3_ref.py` → `bref_*`

Checked mechanically, not by eye: the G schedule and rotation amounts are
**recovered from the oracle by instrumentation** and compared, and every function
is differentially tested.

| element | oracle | gate | verdict |
|---|---|---|---|
| `IV`, 8 words | `blake3_ref.py:29-32` | `:50-51` | match, element-wise |
| `MSG_PERMUTATION`, 16 indices | `:37` | `:52` | match, element-wise; and is a permutation of 0..15 |
| `MASK32` | `:54` | `:53` | match |
| G body: add order, XOR order, rotation amounts **16,12,8,7 in that order** | `:96-103` | `bref_g` `:72-80` | match — amounts recovered by patching `rotr`/`RotateRight` on both sides |
| G argument order (`v[a]+v[b]+mx` first, `my` second half) | `:96-100` | `:73,77` | match (differential, 300 random + 18 edge inputs) |
| `G_CALLS` 8 quadruples + message indices, **including the 4 diagonals** | `round_fn` `:113-121` | `:58-67` | **match, recovered from `round_fn` by instrumenting `g`** |
| `bref_round` iterates `G_CALLS` in order | `:113-121` | `:83-85` | match |
| `bref_permute` | `permute` `:126` | `:88-89` | match (index-identical) |
| initial `v`: `h[0..8]`, `IV[0..4]`, `t_lo`→v[12], `t_hi`→v[13], `block_len`→v[14], `flags`→v[15] | `compress` `:167-172` | `bref_compress` `:101-104` | match — probed slot by slot at rounds=0, and across 7 counters incl. `2^32−1`, `2^32` |
| counter split `t_lo = t mod 2^32`, `t_hi = t >> 32` | `:164-165` | `:509` (caller) | match |
| permutation applied `r < rounds−1`, i.e. **rounds−1 times** | `:181-182` | `:106-109` | match — counted by instrumentation for rounds 0..8 on both sides: `0,0,1,2,3,4,5,6,7` |
| feed-forward `out[i]=v[i]^v[i+8]`, `out[i+8]=v[i+8]^h[i]` with `h` the **original** CV | `:186-188` | `:110-114` | match |
| rounds parameterisation (6 vs 7 is the loop bound only) | `:176-182` | `:106-109` | match — differential over rounds {0,1,2,5,6,7,8} × 25 vectors |

No discrepancy. Tampers on `IV`, `G_CALLS` and `MSG_PERMUTATION` are all
detected by the differentials.

### (b) `DESIGN.md` → `build_g` / `build_round` / `build_compress`

| DESIGN element | design | model | verdict |
|---|---|---|---|
| §4.1 XOR = 4 per-byte `ByteAlu[XOR]` sends, output pinned + operands range-checked | §4.1 | `xor` `:161-166`, 4 equalities | match (contract verified against `bitwise.rs:351-364`) |
| §4.2 `rotr16` = byte relabel `[b2,b3,b0,b1]`, free | §4.2/§7.6 | `:168-170` | **match — an actual index permutation of the source XOR's byte objects**, not a BV rotate; commits 0 columns, emits 0 constraints; and value-equal to `RotateRight(...,16)` over all 2³² |
| §4.2 `rotr8` = `[b1,b2,b3,b0]`, free | §4.2/§7.6 | `:172-174` | match, same evidence |
| §4.2 `rotr12 = rotl16∘rotl4` (r=4), `rotr7 = rotl16∘rotl9` (r=9) | §4.2 | `:207` `{12:4, 7:9}` | match |
| §4.2 two shift identities `hw·2^r = SLLC·2^16 + SLL` | §4.2 | `:222-223` | match |
| §4.2 recombine `Ylo = SLL_hi + SLLC_lo`, `Yhi = SLL_lo + SLLC_hi` | §4.2 | `:226-227` | match |
| §4.2 `SLL_*`, `SLLC_*` are 16-bit (2 bytes each) | §4.2 | `fresh_word()[:2]` `:213-216` | match |
| §4.2 4 `AreBytes` sends/rotation | §4.2 | **not modelled** | **stronger** (F1); and 3 of the 4 are not load-bearing (F2) |
| §4.3 2-op add: `a+b = s + 2^32·carry`, carry boolean | §4.3 | `add2` `:176-184` | match — the design's *derived* carry and the model's *committed* boolean carry proved equivalent over F_p |
| §4.4 3-op add: `a+b+m = s + 2^32·(c1+c2)`, `c1,c2` boolean | §4.4 | `add3` `:186-197` | match |
| §4.3/§4.4/§5 `s` byte-range-checked free by the next XOR | §4.3/§4.4/§5/§7.4 | `s = self.fresh_word()` (declared bytes) | **stronger** (F1) |
| §4.6 feed-forward, 16 XORs, `out[i+8] = v[i+8] ⊕ h[i]` | §4.6 | `:274-277` | match |
| §4.7 `m` needs explicit `AreBytes`; `h`,`t`,`bl`,`fl` are free | §4.7 | **not modelled** | **stronger** (F1). The *dataflow* half of the claim is verified here mechanically: `h`,`t_lo`,`t_hi`,`bl`,`fl` each feed an XOR, `m` does not |
| §5 every add/shift output feeds a downstream XOR | §5/§7.4 | **not modelled** | **stronger** (F1) — premise verified true here for ROUNDS ∈ {1,2,6,7}: 288 add/shift outputs, 0 unchecked |
| §3 per-G budget 56 byte-cells + 6 carry bits | §3 | counted from the model | match (56 / 6) |
| §2 per-G op mix: 2 add3, 2 add2, 4 xor, 2 shift-rotations, 1 rotr16, 1 rotr8 | §2/§5 | counted from the model | match |
| §7.7 `permute^r` wired from the original `M` columns | §7.7 | `:266-272` | match — 56 index pairs vs the oracle's composition |
| §7.8 IV inlined as constants at `v[8..12]` | §1.1/§7.8 | `const_word` `:258-261` | match |
| §7.9 all field expressions `< 2^35 ≪ p` | §7.9 | `WIDE = 48` | match — the ℤ identity and the mod-p identity proved equivalent under the byte bounds |
| §4.5 μ-gating, all-zero padding | §4.5/§7.1 | **not modelled** | not modelled (F4) |
| §4.8 degree ≤ 3 | §4.8 | **not modelled** | not modelled (F4) |
| §1.1 `Blake3` bus, μ multiplicity, timestamps | §1.1/§3 | **not modelled** | not modelled (F4; README finding 1) |

### Gate hygiene

| check | result |
|---|---|
| the G circuit's constraints are satisfiable on their own — MAIN 0's UNSAT is not vacuous | pass |
| the 6-round circuit's constraints are satisfiable on their own | pass |
| constraints emitted per op: xor 4, add2 2, add3 3, rotr12 4, rotr16 0 | pass |
| all 10 canonical 6-round fixtures reproduce from the live oracle (the positive controls are not anchored to a stale file) | pass |
| …and none of them equals the 7-round compression of the same input | pass |
| `gen_7round_vector` returns the oracle's own 7-round output | pass |
| the assumed `ByteAlu[XOR]` / `AreBytes` contracts match `prover/src/tables/bitwise.rs` | pass |
| `check_g()` UNSAT, `check_compress(0)` UNSAT, `check_g(swap_g_operand)` SAT (`--slow`) | pass |

---

## Documentation defects (F5)

Not soundness; a reader following the citations is misled.

* **§7.3's mechanism is backwards.** "a wrong `SLL` admits a large field `SLLC`"
  — the forgery keeps `SLLC` small (`2^r`, inside its own bound) and makes `SLL`
  large (`p − 2^r`). Witness above.
* **§4.8's recombine row over-states its degree.** `μ·(Ylo − SLL_hi − SLLC_lo)`
  is linear in committed columns → body degree 1, ×μ = 2. The table says 2 → 3.
  Safe-side wrong; the "no constraint exceeds 3" verdict is unaffected.
* **§3 and §4.3 disagree on whether `add2` commits a carry column.** §3's per-G
  table counts 1 carry bit per `add2` (6 per G); §4.3 makes it a *derived linear
  expression* `(a+b−s)·INV_SHIFT_32` with no column. Semantically equivalent
  (proved), but 96 cells per compression hang on the reading, in a design whose
  headline is a cell count.
* The gate's docstring "Every committed column … is a FREE bitvector" should say
  "a free **byte**" — the distinction is F1.

---

## Could not determine

Stated so the boundary is explicit rather than implied.

1. **Anything about a Rust chip.** There is none. README finding 2 (48 G
   instances emitted separately; a wrong column index in instance #37) is
   unauditable until it exists, and F1's forgeries are all statements about what
   a future implementation must not do.
2. **The bus layer.** Not modelled by the gate, not audited here. F1's message
   and `h` constructions become live or benign depending on it; README finding 1
   (the missing input↔output timestamp binding) sits in the same place.
   Confirmed absent from the gate, per the brief — not re-derived.
3. **The `--full` monolithic UNSATs** (`check_round`, `check_compress(2/6/7)`)
   were **not run** — 30-40 min timeouts each. I audited the model they run on,
   not their verdicts. Note that they inherit F1 in full: a monolithic 6-round
   UNSAT is still an UNSAT about a model in which every cell is a declared byte.
4. **The `HWSL` inline soundness proof** the design defers to
   (`../keccak-verify/hwsl_inline_test.py` Part 2) — that directory is not in
   this artifact. §C's composed field model re-derives the shift-identity result
   independently, so the conclusion does not rest on the missing file, but the
   cited proof was not read.
5. **Completeness.** Every result here is about soundness (can a wrong witness
   pass). Whether an honest trace generator can *produce* the witnesses — the
   `AreBytes` send layout, the carry values — is unchecked; a mismatch there is
   an unprovable honest witness, not a forgery.
6. **Whether the recovered artifact is byte-identical to the 2026-07-23
   original** (README's own open item). Unchanged by this audit.

---

## Regression suite

`audit_gate_transcription.py`, 73 checks (76 with `--slow`), all passing, every
one paired with a tamper that must break it:

```
A  reference transcription (a): constants element-wise; G_CALLS and rotation
   amounts RECOVERED from the oracle by instrumentation; differential g /
   round / permute / compress over rounds {0,1,2,5,6,7,8}; counter split
   across 2^32; permute-application count 0,0,1,2,3,4,5,6,7; v-layout probed
   slot by slot.  Tampers: IV, G_CALLS, MSG_PERMUTATION — all detected.
B  circuit transcription (b): rotr16/rotr8 are index relabels by object
   identity, commit no columns, value-equal to RotateRight; per-G cell and op
   census; SSA range-check provenance over one G and over ROUNDS 1/2/6/7;
   h[0..4]'s single feed-forward consumer; message indexing under permute^r,
   56 pairs; what the model does not represent (no range object, no mu, no
   bus).  Tampers: a wrong relabel, a G whose add output loses its XOR
   consumer, a swapped MSG_PERMUTATION entry, a swapped G_CALLS message pair
   — all detected.
C  the dangerous direction, in the field: add2/add3 pinned with the range
   check and FORGEABLE without it (concrete witness a=b=0x80000000 -> s=2^32);
   the rotation output needs no range check of its own; the 32-configuration
   bound lattice with non-vacuity; the composed forgery at X=0xFFFFFFFF,
   enumerated exhaustively; both width-audit positives re-derived symbolically
   for all inputs; the message-cell collision.
D  hygiene: non-vacuity of MAIN 0 and the 6-round model; per-op constraint
   counts; derived-vs-committed carry equivalence; canonical fixtures
   reproduce from the live oracle; Z-vs-F_p equivalence of the WIDE=48 model;
   the BITWISE contracts checked against prover/src/tables/bitwise.rs;
   (--slow) the gate's own BV verdicts.
```
