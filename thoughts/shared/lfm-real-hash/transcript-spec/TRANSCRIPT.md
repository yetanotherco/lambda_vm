# The LFM compress-chain transcript — specification

**Status:** specification + reference + vectors + gate extension, written
**before any Rust exists**, same discipline as Phase 2. **No chip code exists
for this.** **Date:** 2026-08-11.

**Decision this implements:** the user ratified **option B, form B1**
(`permute-socket-options.md`): the Fiat–Shamir sponge becomes a compress-based
chain **for all hashers**, **no permute socket is ever built**, and `MODE_P`
stays pinned to 0 permanently. The deciding argument was that B needs no
assumption beyond A6R.

Claims are ✓ VERIFIED (read the code, cited), ✓ EXECUTED (ran it), ? INFERRED,
or ✗ OPEN.

---

## 0. Board

| check | result |
|---|---|
| `transcript_kats.py` — K1–K6 | **PASS**, both round counts; the end-to-end vector tracks `FriToyV0`'s CURRENT preamble (`absorb_felts` ×2) |
| `transcript_gate.py` — G1–G5 (executable today) | **PASS 6/6** |
| `squeeze_run_analysis.py` — entropy loss vs run length | executed, §4 |
| M1–M7 against the built chip | **ALL FIRED** (builder, Rust side) |
| **M8** — the eighth control, added after M5/M6 falsified §3.3 | **PASS**, 4 legs, model side |
| post-B1 CHIP-GATE re-gate + re-pin | **PASS 79/79**, `gate-oracle/CHIP-GATE.md` |

Run order: `python3 transcript_kats.py --write` → `python3 transcript_gate.py`
→ `python3 squeeze_run_analysis.py`. Plain `python3` + `z3`; no cargo.

---

## 1. The construction

**State: one cell** — 4 lanes × u32 = 128 bits, initially all-zero (mirroring
`SpongeVar::new`, which starts from three zero cells). Down from three cells,
because a chain needs no rate/capacity split.

Every operation is one ordinary **frozen-socket compress** under the transcript
tag, written `compress_T`:

| op | definition | compresses |
|---|---|---:|
| `absorb(c)` | `state ← compress_T(state, c)` | 1 |
| `absorb2(c0, c1)` | `state ← compress_T(compress_T(state, c0), c1)` | 2 |
| `squeeze()` | `out = state` **then** `state ← compress_T(state, SQ(i))` | 1 |

`squeeze` outputs **before** advancing, mirroring `SpongeVar::squeeze_cell`
(`out = state[0]; state = permute(state)`), so the two constructions stay
structurally parallel and the eventual diff is reviewable.

`squeeze_ext` takes lanes 0–2 of the squeezed cell and `squeeze_bits(n)` the low
`n` bits of lane 0 — unchanged from today, so **`programs.rs` needs no edit**:
`fri_toy_program_source` calls only `absorb`/`absorb2`/`squeeze_ext`/
`squeeze_bits`, all of which keep their signatures.

### 1.1 `SQ(i)` — the squeeze counter, and why it is free

The advance operand is the **constant cell** `SQ(i) = [SQUEEZE_MARK, i, 0, 0]`,
where `i` is the squeeze index and `SQUEEZE_MARK = "SQZ0"` as a little-endian
u32.

**It costs nothing.** ✓ VERIFIED `edsl.rs:1-4` — the eDSL fully unrolls,
"nothing loop-shaped reaches the machine", so `i` is a compile-time constant and
the operand is a program constant either way, pinned by `program_id`. A constant
cell was going to be emitted regardless; this one just carries a counter.

**What it buys** is §4's FSE-2014 lesson written into the construction. Without
it, a run of consecutive squeezes iterates **one fixed public non-injective
map**, whose functional graph an adversary can precompute — precisely the
structure the GLUON-64 T-sponge attacks exploit. With it, each step is a
different map and no single functional graph exists to analyse.

**Absorb/squeeze separation** rests primarily on the operation sequence being a
compile-time constant of the program: a prover cannot perform a squeeze where
the program says absorb, because the sequence is fixed at emission and bound by
`program_id`. `SQUEEZE_MARK` is defence in depth, not the load-bearing argument.

### 1.2 Framing — identical to the Merkle socket but for one constant

| input to `f` | value |
|---|---|
| `h` | `IV[0..8]` |
| `m[0..4]` | `state` |
| `m[4..8]` | the operand (absorbed cell, or `SQ(i)`) |
| `m[8]` | **`TAG_LFMT`** ← *the only thing that differs* |
| `m[9..16]`, `t` | `0` |
| `block_len` | `36` |
| `flags` | `0x0B` |
| output | `out[0..4]` |

**Consequence, and it is the whole point of option B:** the transcript inherits
the compress socket's external anchor unchanged. ✓ EXECUTED (K1): at 7 rounds
every step equals `BLAKE3(LE32(state) ‖ LE32(operand) ‖ "LFMT")[0..16]`,
computed by two separate routes and asserted equal. The implementer must
re-assert this as a one-line `blake3::hash` call.

---

## 2. Tag allocation

| tag | u32 (LE) | use | status |
|---|---|---|---|
| `"LFMC"` | `0x434D464C` | 2-to-1 compress / Merkle parent | **built** |
| **`"LFMT"`** | **`0x544D464C`** | **transcript step (this document)** | **specified here** |
| `"LFMP"` | `0x504D464C` | ~~permute socket~~ — **retired unused** | never built (B1) |
| `"LFML"` | `0x4C4D464C` | leaf domain | reserved, O5-ratified |

A tag is never reused for a second purpose. `"LFMP"` is now **permanently
unused**: B1 means no permute socket will ever exist, so the reservation should
be marked retired rather than deleted — deleting it would let a future
allocation reuse the value and silently create a domain nobody analysed.

**✓ DONE 2026-08-11.** Both tag tables updated: `gate-oracle/ORACLE.md` §2.3
and `thoughts/blake3/socket-kats/SOCKET.md` §2.4 now carry `"LFMT"`, mark
`"LFMP"` **RETIRED UNUSED** with the reuse-hazard note, and record O5's
ratification on `"LFML"`. `SOCKET.md` §7 — the rejected permute sketch — also
got a superseded banner, so a reader landing there directly cannot mistake it
for a plan.

---

## 3. The `m[8]` mechanism — exact constraint change

Today `m[8]` is `WordRef::Const(TAG_LFMC)` — a compile-time constant, hence zero
columns and zero range checks. Two tags need `m[8]` to depend on the row's mode
**without becoming prover-chosen**.

### 3.1 `MODE_T`: a new preprocessed column, not a reuse

**Recommendation: add a fresh preprocessed `MODE_T`; do NOT repurpose `MODE_P`.**

Reusing `MODE_P` is tempting — B1 pins it to 0 for BLAKE3, so it looks dead. It
is not: `MODE_P` is in the **shared** preprocessed prefix and the `Test` and
`Poseidon` arms still use it for their permute (and `TrivialV0` still calls
`b.permute` directly — §6). Repurposing it would make one preprocessed column
mean different things under different hashers, which is worse than the column it
saves.

**Cost:** `PREP_WIDTH` 11 → 12, so the preprocessed roots move and all six
registry entries are re-blessed. ? INFERRED but well-supported: that re-bless is
**already happening** — B1 changes the sponge for every hasher, so every
`program_id` moves regardless. The new column rides along at no marginal
protocol cost, provided it is sequenced into the same re-bless.

### 3.2 The constraints, before and after

> **Scope note:** this table records the **B1** change. `MODE_L` has since landed
> and widened the same constraints again — `MU` and the capacity selector are now
> `MODE_C + MODE_T + MODE_L`, `NUM_SELECTORS` is 4 and `PREP_WIDTH` is 13. For
> the current state read `../leaf-spec/LEAF.md` §2 and `../gate-oracle/CHIP-GATE.md`
> §4.7; the "AFTER" column below is B1's after, not today's.

```
idx 0-3   BEFORE:  S_k − (MODE_P·IN_{8+k} + MODE_C·IV_k)
          AFTER:   S_k − (MODE_P·IN_{8+k} + (MODE_C + MODE_T)·IV_k)
```
A transcript row is still a compress, so its capacity prefix is still the IV;
only the selector widens.

```
idx 4     BEFORE:  mode_sum·(1 − mode_sum),  mode_sum = MODE_C + MODE_P
          AFTER:   mode_sum·(1 − mode_sum),  mode_sum = MODE_C + MODE_T + MODE_P
```
Exactly-one-of stays the registrar's job; this pins the sum to a bit. It is what
excludes `MODE_C = MODE_T = 1` (which would give `mode_sum = 2`, and
`2·(1−2) = −2 ≠ 0`).

```
idx 5     BEFORE:  MODE_P          (pin to zero — no permute socket)
          AFTER:   MODE_P          (unchanged, and now PERMANENT under B1)
```

```
MU        BEFORE:  MODE_C
          AFTER:   MODE_C + MODE_T
```

```
m[8]      BEFORE:  WordRef::Const(TAG_LFMC)
          AFTER:   MODE_C·TAG_LFMC + MODE_T·TAG_LFMT      (a new WordRef variant)
```

### 3.3 Soundness argument for the tag

**The tag stays prover-unchosen because `MODE_C` and `MODE_T` are preprocessed.**
A preprocessed column is fixed by the row's position in the preprocessed trace,
which is bound by the preprocessed commitment, which is folded into
`lfm_program_id`. The prover chooses neither. This is the same argument that
already makes `MU = MODE_C` trustworthy — the existing arm's doc calls it out:
*"a prover cannot choose it"*.

> **⚠ CORRECTED 2026-08-11 — an earlier revision of this paragraph drew an
> inference that does not hold, and the correction matters more than the
> original claim did.**
>
> It said: *"idx 4 forces the mode sum to a bit, so at most one tag is
> selected."* **The clause after "so" is a non-sequitur.** Over a prime field
> `mode_sum ∈ {0,1}` pins the SUM, not the selectors: `MODE_C = x`,
> `MODE_T = 1 − x` satisfies idx 4 for *any* `x`, and since the two tags are
> distinct, `x = (T − TAG_LFMT)/(TAG_LFMC − TAG_LFMT)` solves for **any** target
> tag `T`. So idx 4 contributes nothing to one-hotness.
>
> ✓ EXECUTED twice, independently: the builder's Rust M5/M6 run forges the tag
> `"XXXX"` by a fractional split and the eval set accepts with zero violations;
> I reproduced it in the gate's own field model —
> `idx 4 alone → sat` with `MODE_C = 4387334679741772800`,
> `MODE_T = 14059409389672811522` (sum ≡ 1, `m[8] = 0x58585858`), and
> `idx 4 + one-hot → unsat`, with both honest tags still reachable under one-hot.

**What actually closes it, stated correctly.** Two independent mechanisms, and
**idx 4 is neither of them**:

1. **`MODE_C`/`MODE_T` are preprocessed.** The prover cannot choose them at all —
   a row's mode is fixed by its position in the committed instruction group. This
   is the primary closure.
2. **The registrar's exactly-one-of check.** This, not idx 4, is what makes the
   selectors one-hot. ✓ VERIFIED it is also why `MODE_T` sits at layout index 8
   rather than after the multiplicities: the admission validator reads the
   selectors as a **contiguous span** (`NUM_SELECTORS` from `MODE_C` — 3 at B1, **4 since `MODE_L`**), so a
   selector parked past the mults would be outside the one-hot check and
   silently unchecked.

**What idx 4 does buy:** it excludes the both-set case `MODE_C = MODE_T = 1`
(sum 2, and `2·(1−2) ≠ 0`). Useful, but strictly weaker than one-hotness.

**Do not delete the registrar's one-hot check as redundant.** idx 4 would not
save it. **M8 in §5.3 is the control that enforces this**, and it exists because
this paragraph was wrong: a reader who trusted the original sentence could have
removed the load-bearing check while every constraint still passed. M5 and M6
prove the other two dependencies are real.

### 3.4 Degree and cost impact: none

- **Degree unchanged.** `m[8]` was degree 0 (a constant); it becomes degree 1 (a
  linear form over preprocessed columns). It appears only as an `add3` operand,
  whose body `a + b + m − s − 2^32(c1+c2)` is degree 1 either way; × `MU` = 2.
  Max degree stays **3** (the carry booleanities). ✓ VERIFIED against the
  committed arm's structure.
- **Zero columns, zero sends.** `m[8]` is used as a whole word value
  (`word_expr`), never byte-decomposed, so it needs no byte columns and no
  `AreBytes` — exactly as the constant did.
- **One preprocessed column** (`MODE_T`), which is not a main column and does not
  enter the census.

---

## 4. Security

### 4.1 The argument, in one paragraph

The transcript is a hash chain over a collision-resistant compression function,
domain-separated from Merkle parents by a tag the prover cannot choose. Fiat–
Shamir needs each challenge to be a random-oracle function of everything
committed before it, so that a prover cannot predict or grind it before
committing; it does **not** need a secret capacity, because the protocol is
public-coin — every absorbed value is a public commitment and every squeezed
value a public challenge (✓ VERIFIED against `programs.rs:547-567`). That is
exactly what A6R already asserts: *"suitable as a 2-to-1 compression for Merkle
hashing **and as a PRF for Fiat–Shamir**"*.

> **New named assumption required: NONE beyond A6R.** This is why option B was
> chosen. Option A would have needed A-TSP (a T-sponge instantiation) on top.

**Bound:** the state is one cell = 128 bits, so ~**64-bit collision resistance**
by the birthday bound — the same number as the digest's, from the same
`HASH_DIGEST_FELTS = 4` cause, not introduced by this construction.

### 4.2 ⚠ The squeeze-run analysis — the FSE-2014 lesson, applied to B itself

A **squeeze run** is a maximal sequence of consecutive squeezes with no absorb
between. Within a run the state advances by repeatedly applying a non-injective
map, so the reachable set shrinks. Option A was not the only construction
exposed to this — **B is too**, and the same rigour demanded of A-TSP's
iteration bound is owed here.

**(a) The bound.** Model `compress_T(·, operand)` as a random map on `2^128`
points. Image fraction after a run of `k` follows `α_{j+1} = 1 − e^{−α_j}`,
`α_0 = 1`, with `α_k ~ 2/k` (Flajolet–Odlyzko); loss is `−log₂ α_k` bits.
✓ EXECUTED, and the asymptotic verified (`k = 65536`: `α = 3.052e-05`,
`2/k = 3.052e-05`):

| run `k` | 1 | 4 | 16 | 64 | 256 | 1024 | 65536 |
|---|---:|---:|---:|---:|---:|---:|---:|
| loss (bits) | 0.66 | 1.68 | 3.23 | 5.07 | 7.02 | 9.01 | 15.00 |
| state left | 127.3 | 126.3 | 124.8 | 122.9 | 121.0 | 119.0 | 113.0 |

**The counter does not change these numbers** — composing distinct random maps
obeys the same recursion. It removes the *attack structure* (one precomputable
functional graph), which is the part that matters.

**(b) Run lengths as they exist. ✓ VERIFIED** `programs.rs:549-567`:
`FriToyV0`'s runs are **[2, 1, 4]**, so **max run = 4** → **1.68 bits** of 128.
`TrivialV0` has no sponge at all.

> **⚠ The max run IS `NUM_QUERIES`.** The query loop squeezes once per query with
> no absorb in the body, so the run length **scales with the query count**.
> `NUM_QUERIES = 4` today (`fixture.rs:37`); a production FRI at 100–200 queries
> would have a run that long — 6–7 bits. Still fine, and this is exactly why the
> regime has to be written down rather than left to the current toy shape.

**(c) Regime and guidance bound.**

> Runs up to `k ≈ 16` cost under 4 bits; up to `k ≈ 256`, under 8 bits. The
> analysis holds while `k ≪ 2^64`, at which point the birthday bound on the
> 128-bit state dominates anyway. **A program whose squeeze runs exceed
> `k = 2^16` (15 bits of loss) must revisit this section**; below that, the loss
> is dominated by the 64-bit collision bound of §4.1 and changes nothing.

**(d) Recommendation: keep the counter, do NOT mandate absorb-interleaving.**
Argued rather than asserted. Interleaving a counter-absorb every `K` squeezes
would cost one extra compress per `K` squeezes to buy a *bit-counting* benefit
that is already negligible — at the current `k = 4` it would save 1.68 bits of
128, and even at a production `k = 256` only 7. The counter, by contrast, costs
**zero** (§1.1) and removes the *structural* exposure, which is the part with a
cryptanalytic track record. Paying compressions for the negligible half while
skipping the free fix would be the wrong trade. **Documented bound + free
counter, with the `k = 2^16` revisit trigger, is the right shaping.**

### 4.3 Exposure profile — the design-level reason this construction stands

Folded in from `permute-socket-options.md` §8.5 at the lead's request, because it
is construction rationale and belongs in the signed record rather than only in a
decision paper. ✓ EXECUTED (200 and 2000 random inputs respectively, exact).

With `h = IV`, BLAKE3's output is `out[i] = v[i] ⊕ v[i+8]` and
`out[i+8] = v[i+8] ⊕ IV[i]`. **How many output words you publish therefore
decides how much of the final internal state a reader can reconstruct.**

**This construction publishes four of sixteen.** `out[0..4] = v[0..4] ⊕
v[8..12]`, and with no second output block to cross-XOR against the two summands
**cannot be separated** (✓ EXECUTED, 2000/2000 samples). Twelve words stay
unpublished.

**The rejected option A would have published twelve of sixteen**, which exposes
both halves of a cross-relation:

```
out[i] ⊕ out[i+8] == v_final[i]   ⊕ IV[i]     (i in 0..8)
out[8+i]          == v_final[8+i] ⊕ IV[i]     (i in 0..4)
```

— so one permute output would have revealed **8 of the 16 final state words**
directly, by XOR with public constants.

**This is not an attack and none is claimed.** The final state is a pseudorandom
function of the input, so recovering it from the output is not obviously
exploitable. It is a *structural* property, of the kind that belongs in a
security argument a reviewer signs — and §4.2's GLUON-64 line began with
structure rather than with a break. It is recorded here because it is the one
argument for this construction that is about the cryptography rather than about
process, reversibility or cost.

A related note, now moot but worth preserving: option A would have made XOF words
part of the *chaining state*. Standard BLAKE3 chains on `out[0..8]` and uses
`out[8..16]` only as extended output — a role its designers analyse as output,
not as state. This construction stays inside `out[0..8]`, using only `out[0..4]`.

---

## 5. Gate extension

### 5.1 Executable today — ✓ EXECUTED, PASS 6/6

The transcript step is the frozen socket with a different `m[8]` constant, and
`Framing.tag_word` already parameterises exactly that, so the existing theorems
apply **before any Rust exists**. `transcript_gate.py` imports `../gate-oracle/`
rather than editing it — that model is **pinned** to the committed chip and a
spec exercise must not move a pinned instrument.

| | check | result |
|---|---|---|
| G1 | message schedule under `LFMT`, symbolic, all 7 rounds | UNSAT |
| G2 | full 7-round pipeline == transcript KAT | SAT |
| G3 | the same pipeline **excludes** a wrong digest | UNSAT |
| G4a | Merkle tag used for a transcript step | **SAT** |
| G4b | transcript tag used for a Merkle parent | **SAT** |
| G5 | squeeze counter `i=1` cannot produce squeeze `i=0` | **SAT** |

G4a/G4b are the domain-separation controls in both directions; G5 makes the
counter load-bearing at the gate level, not only in the vectors.

### 5.2 What the build must add

`chip_model.py` needs a `WordRef`-equivalent for the mode-selected tag and a
`MODE_T` role in BLOCK 0; `gate.py`'s BLOCK-0 field audit (`B0a`/`B0b`) needs
its mode-sum widened to `MODE_C + MODE_T + MODE_P` (and again to include
`MODE_L`). Both are small, and the
census is unaffected (§3.4).

### 5.3 The `MODE_T` controls — ✓ ALL EXECUTED

> **Status, 2026-08-11:** these were written before the chip existed, as a
> checklist the build would inherit rather than invent. **M1–M7 have since all
> fired against the built chip** (builder, Rust side) and **M8 is executed
> model-side** in the CHIP-GATE board (4 legs, `gate-oracle/CHIP-GATE.md` §4.6.2).
> Kept in full, in the original pre-commitment wording, because a control list
> written *after* seeing the implementation is worth much less than one written
> before it — and because they are now the standing regression set.

| | control | expected |
|---|---|---|
| M1 | `m[8]` pinned to `TAG_LFMC` while `MODE_T = 1` | **SAT** — a transcript row computing the Merkle tag |
| M2 | `m[8]` pinned to `TAG_LFMT` while `MODE_C = 1` | **SAT** — the mirror image |
| M3 | `MODE_C` and `MODE_T` both 1 on one row | UNSAT — excluded by idx 4 (this **is** what idx 4 buys; see M8 for what it does *not*) |
| M4 | `MODE_C = MODE_T = 0` with `MU = 1` | UNSAT — `MU` *is* their sum |
| M5 | drop the mode-sum booleanity | **SAT** — modes become arbitrary felts, so `m[8]` becomes a prover-chosen combination of both tags and the domain separation evaporates |
| M6 | `MODE_T` as a MAIN (prover-chosen) column | **SAT** — this is the control that proves the preprocessed dependency of §3.3 is real |
| M7 | generalised capacity form idx 0–3 | UNSAT present / SAT dropped |
| **M8** | **idx 4 present, registrar one-hot ABSENT** | **SAT** — a forged `m[8]` is reachable as a fractional blend of the two tags. The control that stops a refactor deleting the one-hot check as redundant. Pair with the honest-path leg: both real tags must stay reachable, or a fix that rejects everything would pass |

**M5, M6 and M8 are the three that matter.** They are what turn §3.3 from an assertion
into a checked claim, exactly as WA1/WA2 did for obligation O1.

---

## 6. `TrivialV0`'s fate — recommendation

✓ VERIFIED: `TrivialV0` calls `b.permute` **directly** (`programs.rs`, the
`trivial_program_source` body: two `compress`es then
`b.permute([d1.as_cell(), h[3], d0.as_cell()])`), not through `SpongeVar`. So it
is blocked by B1 independently of the sponge rewrite, and B1 does not touch it.

**Recommendation: drop the raw `permute` from `TrivialV0` and replace it with a
third `compress`, making the program run under every hasher — and add a
permute-coverage fixture that is NOT a registry entry.**

Reasoning. Keeping `TrivialV0` as a Test/Poseidon-only fixture would leave the
registry with an entry that cannot run under the machine's real hash, which is
the F3.4 situation in miniature — a registered program whose cryptographic
meaning depends on a placeholder. The registry's six entries should all be
provable under the production hasher. Against that, permute mode does not
disappear: `Test` and `Poseidon` keep it, and it needs *some* test coverage or
the arms rot. But coverage does not require a **registry** entry — a
`#[cfg(test)]` permute fixture exercises the arms without claiming a program
identity, and the cost of the swap is one compress (16,527 vs 16,635 cell-equiv,
✓ EXECUTED — the compress version is marginally *cheaper*).

? INFERRED and worth checking during the build: whether any test asserts
`TrivialV0`'s public output shape, which the swap would move. Its `program_id`
moves anyway in the B1 re-bless.

---

## 7. What is executed, and what is open

| claim | status |
|---|---|
| every step == `blake3::hash(state‖operand‖"LFMT")[..16]` at 7 rounds | ✓ EXECUTED (K1, two routes) |
| end-to-end `FriToyV0`-shaped transcript, op by op, both round counts | ✓ EXECUTED (K2) |
| transcript step ≠ Merkle parent on the same cells | ✓ EXECUTED (K3) |
| the squeeze counter is load-bearing | ✓ EXECUTED (K4, G5) |
| absorb order is load-bearing | ✓ EXECUTED (K5) |
| `FriToyV0` transcript costs **13** compressions (11 + 2 leaf rows) | ✓ EXECUTED (K6), re-pointed at the current preamble |
| the frozen socket computes the transcript step correctly under `LFMT` | ✓ EXECUTED (G1–G3) |
| domain-separation controls fire both directions | ✓ EXECUTED (G4a/G4b) |
| squeeze-run entropy bound + the programs' actual runs | ✓ EXECUTED (§4.2) |
| the same identity against the Rust `blake3` **crate** | ✗ DEFERRED — needs cargo |
| `MODE_T` mechanism (M1–M7) | ✗ OPEN — needs the chip |
| tag tables in `ORACLE.md` §2.3 / `SOCKET.md` §2.4 updated | ✗ OPEN — §2, one pass |
| `PREP_WIDTH` 11 → 12 sequenced into the B1 re-bless | ✗ OPEN — build |

---

## 8. Files

| file | what |
|---|---|
| `transcript_ref.py` | the reference — the future `HostSponge` mirror |
| `transcript_kats.py`, `transcript_kats.json` | K1–K6 + the end-to-end `FriToyV0` vector |
| `squeeze_run_analysis.py` | §4.2's entropy-loss numbers |
| `transcript_gate.py` | G1–G5 executable now + M1–M7 pre-committed |
