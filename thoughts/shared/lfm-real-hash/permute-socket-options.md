# The `LFM_HASH` permute socket — options paper

**This is a decision paper for the user. It is NOT an implementation and NOT a
unilateral pick.** §7 carries a recommendation, clearly marked as mine.

> **DECIDED — 2026-08-11: the user ratified OPTION B in its B1 form** (the
> Fiat–Shamir sponge becomes a compress-based chain for ALL hashers; no permute
> socket is ever built; `MODE_P` stays pinned to 0 permanently). Decision made
> on this paper's presentation with §6's unsettled items disclosed. Next steps
> per §7: TAG_LFMT allocation, `SpongeVar`+`HostSponge` rewritten together,
> `TrivialV0`'s raw permute resolved, registry re-bless sequenced once, gate
> re-run with the two-tag framing.
>
> **Post-decision update (§8, same day):** the A-TSP research completed AFTER
> ratification and was reported to the user: A-TSP is citable (not novel), so
> §7's reason 1 was overstated at decision time; and the T-sponge entropy-loss
> caveat applies to BOTH options' squeeze runs (§8.3), so that axis separates
> nothing. The decision was reaffirmed on reasons 2–5, and B's transcript spec
> must now carry its own squeeze-run iteration bound (assigned).

**Date:** 2026-08-11. **Scope:** what to do about `LFM_HASH`'s `permute` mode
under BLAKE3 — the thing that keeps the F3.4 disclosure half-retired.
**Ground:** worktree `lambda_vm-blake3-impl`, branch `blake3-real-hash`, head
`2957c3f9`. No cargo run; costs are priced from the gated census, not estimated.

Claims are ✓ VERIFIED (read the code, cited), ✓ EXECUTED (ran it), ? INFERRED
(derived, reasoning shown), or ✗ OPEN.

---

## 1. The situation, verified rather than assumed

**The sponge.** ✓ VERIFIED `edsl.rs:16-60`. `SpongeVar` is an overwrite-rate
duplex: state is **3 cells** (rate = cells 0–1, capacity = cell 2).

```rust
absorb2(c0, c1):  state = permute([c0, c1, state[2]])   // OVERWRITE the rate
squeeze_cell():   out = state[0]; state = permute(state)
```

Absorb **overwrites** the rate rather than XOR-ing into it, and the capacity
cell is carried unchanged. `squeeze_ext` takes lanes 0–2 of a squeezed cell;
`squeeze_bits` takes the bit decomposition of lane 0.

**⚠ The single most important fact in this paper, and it changes the shape of
the decision.** ✓ VERIFIED `edsl.rs:6-10`, quoted in full:

> *"The duplex sponge here is the machine side of the test transcript and is
> mirrored bit-exactly by `fixture::HostSponge`. Like `TestPermutation` itself it
> is **NOT a production construction** — the real transcript lands with the
> ecosystem hash decision; this one exists so the protocol loop can be built and
> measured now."*

The sponge is **already scheduled for replacement**, by the same decision this
paper serves. Redesigning it is therefore *the planned work*, not a detour — and
that removes most of the usual objection to option B.

**Who actually needs `permute`.** ✓ VERIFIED by grepping the whole module:

| user | how | ops |
|---|---|---|
| `FriToyV0` (`programs.rs:524-640`) | via `SpongeVar` | **10 permutes**, 56 compresses |
| `TrivialV0` (`programs.rs:17-40`) | a **raw `b.permute(...)` call**, not the sponge | 1 permute, 2 compresses |

Nothing else. The 10 permutes are 6 in the preamble (`absorb`, `squeeze_ext`×2,
`absorb`, `squeeze_ext`, `absorb2`) plus 1 per query × `NUM_QUERIES = 4`
(`squeeze_bits`). The 56 compresses are 4 queries × 14 (leaf + 4-level walk,
twice, plus an L1 leaf + 3-level walk). ✓ VERIFIED against `fixture.rs:26-40`
for the shape constants.

Note `TrivialV0` calls `permute` **directly**, so it is blocked by this decision
independently of whatever happens to the sponge. Any option that removes the
permute socket must say what happens to that call.

**What the wrap needs: nothing.** ✓ VERIFIED (F3.4, `F3-hash-chips.md:184-205`):
the epoch verifier hashes with keccak throughout and emits no `Instr::Hash` at
all, so its `LFM_HASH` group is empty. **The wrap is not blocked by this
decision.** Only the two registered `LFM_HASH` programs are.

**Today's behaviour under BLAKE3 is loud, not silent.** ✓ VERIFIED: the AIR pins
`MODE_P = 0` (constraint idx 5), `admits()` rejects a `Permute` row naming why,
and `Blake3Permutation::permute` panics rather than returning a value the chip
does not prove. So the status quo is *safe*; it is merely incomplete. There is
no soundness fire here, which means this decision can be made on design merit
rather than under pressure.

**A6R already covers the transcript.** ✓ VERIFIED `A6R-signoff.md:56-66` — the
signed statement reads *"...suitable as a 2-to-1 compression for Merkle hashing
**and as a PRF for Fiat–Shamir**"*, and the sheet says outright that it "covers
the transcript sponge as well as the Merkle compress". So using BLAKE3 for the
transcript invokes **no assumption that is not already signed**. This matters:
it means options A and B differ in *construction* risk, not in *primitive* risk.

**What Fiat–Shamir actually needs here.** ? INFERRED, and it is the crux. The
protocol is public-coin: every absorbed value (`main_root`, `l1_root`, `t0`,
`t1`) is a public commitment, and every squeezed value (`alpha`, `zeta0`,
`zeta1`, query bits) is a public challenge. The requirement is that a challenge
be a random-oracle function of everything committed before it, so the prover
cannot grind or predict it before committing. **Secrecy of the capacity is not
required** — there is no secret in the transcript. That observation is what makes
option B's much simpler construction legitimate; a sponge's capacity buys
security against an adversary who sees only the rate, which is not the threat
model here.

---

## 2. Option A — a compress-derived transform under the reserved `"LFMP"` tag

The `SOCKET.md` §7 direction (which that document is careful to label *"a sketch,
not a decision — unreviewed"*).

**Framing.** `h = IV`; `m[0..12] = state`; `m[12] = "LFMP"`; `m[13..16] = 0`;
`t = 0`; `block_len = 52`; `flags = 0x0B`; new state = `out[0..12]`.

**Security property required.** The state-update map must behave as a random
transformation on 12 words. Note precisely what it is *not*: `out[0..12]` is 12
of the 16 output words of a compression function, so it is **not a permutation**
— it is non-invertible. The standard sponge proof is for a random *permutation*;
this needs the random-*transformation* variant (the "T-sponge" of
Bertoni–Daemen–Peeters–Van Assche), which gives essentially the same bound. That
is a defensible but **different theorem**, and it is a construction assumption
that does not exist today.

> **Named assumption this option would add — it must be signable, like A6R:**
> **A-TSP.** *The overwrite-rate duplex with rate 2 cells and capacity 1 cell,
> instantiated with `T(state) = BLAKE3-compress(IV, state‖"LFMP"‖0, t=0,
> block_len=52, flags=0x0B)[0..12]`, is indifferentiable from a random oracle up
> to ~2^64 queries.* At 7 rounds this rests on BLAKE3 plus the T-sponge theorem;
> at 6 rounds it additionally invokes A6R.

**KAT-ability: ✓ EXECUTED, and it is good news.** `out[0..16]` is exactly the
first 64 bytes of BLAKE3's XOF stream over the 52-byte message, so
`out[0..12]` = XOF bytes 0..48. I ran this against my anchored oracle:

```
msg = LE32(state[0..12]) ‖ "LFMP"        (52 bytes)
XOF 64B == out[0..16]        : True
first 12 (the new state)     : True
32B hash == out[0..8]        : True
```

So at 7 rounds a permute is a direct `blake3::Hasher::finalize_xof()` assertion
against the crate — **the exact property the 7-round decision was bought for is
preserved.** This is option A's strongest point.

**Cost.** ✓ EXECUTED via `permute-socket-cost.py`, whose formula is validated by
reproducing the gated compress census to the unit before it prices anything:

| | main | sends | aux | cell-equiv | vs one compress |
|---|---:|---:|---:|---:|---:|
| compress (as built), 7r | 3,436 | 1,382 | 2,073 | **5,509** | 1.000 |
| **A permute, 7r** | 3,484 | 1,422 | 2,133 | **5,617** | **1.020** |
| compress, 6r | 2,956 | 1,190 | 1,785 | **4,741** | 1.000 |
| **A permute, 6r** | 3,004 | 1,230 | 1,845 | **4,849** | **1.023** |

**One permute ≈ 1.02 compressions** — the mixing core is identical and only the
I/O differs (12 input lanes instead of 8, 12 output words instead of 4).
`FriToyV0` total: **364,674** cell-equiv at 7r.

**Blast radius.** Large — it is a **second socket**. A second mode in
`blake3_socket.rs` with its own column layout (12 lanes, 12 output words), its
own constraint indices, `NUM_CONSTRAINTS` change, deleting the `MODE_P = 0` pin
(itself a currently-gated constraint), an executor arm, a trace filler, the host
`permute` impl replacing its panic, a KAT file, and a registry re-bless.
Roughly the size of the compress arm again.

**Gate extension: HIGH feasibility.** The G-core theorem T1 is untouched — same
mixing core, same contracts. Only the framing theorems (T2/T3) and the KATs
change, and my `Framing` dataclass already parameterises `tag_word`, `tag_slot`
and `out_window`; it needs a lane count and a variable window *width*. Every
negative control transfers. The `MODE_P = 0` audit (B0a) would have to be
re-derived, since idx 5 is exactly what this option deletes.

**F3.4:** fully retired for BLAKE3 programs.

---

## 3. Option B — make the sponge compress-based, so the socket never exists

Keep `LFM_HASH` **compress-only by design**. `MODE_P` stays pinned to 0
permanently. No permute socket is ever built.

**Construction.** State is **1 cell** (128 bits) — the chaining value, which is
BLAKE3's own native shape.

```
absorb(c)        : state = compress_T(state, c)                    1 compress
absorb2(c0, c1)  : state = compress_T(compress_T(state, c0), c1)   2 compresses
squeeze_cell()   : state = compress_T(state, DOMAIN); out = state  1 compress
```

**Domain separation, and the neat part: no new socket is needed for it.** A
transcript step must not be replayable as a Merkle parent, so it needs its own
tag — but the *shape* is unchanged (2 cells in, 1 cell out). Only the constant
`m[8]` differs. Make it a linear form over the **preprocessed** mode columns,
`m[8] = MODE_C·TAG_LFMC + MODE_T·TAG_LFMT`: prover-unchosen, essentially free in
cells, no new layout. ? INFERRED but well-supported — the existing arm already
computes `S_k = MODE_P·IN + MODE_C·IV` in exactly this shape (idx 0–3).

**Security property required — and this is option B's real advantage.** This is
the textbook Fiat–Shamir transcript: a hash chain. What it needs is that the
chain is collision-resistant and the challenge derivation is a random oracle,
which is **precisely and only what A6R already asserts**. There is no T-sponge
theorem, no capacity argument, no overwrite-mode analysis, and (per §1) no need
for a secret capacity in a public-coin protocol.

> **New named assumption required: NONE beyond A6R.** That is the difference
> between B and A, and it is worth more than the 1.2% cost gap between them.

**Security bound.** State is 1 cell = 128 bits → ~64-bit collision resistance,
by the birthday bound. Identical to option A's (whose capacity is also one
128-bit cell) and identical to the digest's. All three options land on the same
number, because it is dictated by `HASH_DIGEST_FELTS = 4`, not by the
construction. Nothing here makes it worse.

**KAT-ability.** Every transcript step is an ordinary socket compress, so it is
KAT-able exactly as the compress socket already is — `blake3::hash(a‖b‖tag)`
truncated. **No new KAT machinery at all.**

**Cost.** ✓ EXECUTED. `FriToyV0`'s 10 permutes become 11 compresses (1 + 1 + 1 +
1 + 1 + 2 + 4, from the op-by-op derivation above):

| program | option A | option B | B/A |
|---|---:|---:|---:|
| `FriToyV0`, 7r | 364,674 | 369,103 | **+1.2%** |
| `FriToyV0`, 6r | 313,986 | 317,647 | +1.2% |
| `TrivialV0`, 7r | 16,635 | **16,527** | **−0.6%** |

**Cost is a tie.** It does not decide this.

**Blast radius.** Moderate, and mostly in code that is already marked
provisional: `edsl.rs`'s `SpongeVar` (~45 lines), `fixture.rs`'s `HostSponge`
mirror (~45 lines), and `TrivialV0`'s raw `b.permute` call. **`programs.rs` need
not change at all** if `SpongeVar` keeps its public method signatures —
`fri_toy_program_source` calls only `absorb`/`absorb2`/`squeeze_ext`/
`squeeze_bits`. Registry re-bless: all `program_id`s move, but Phase 3 moves
them anyway when the hasher tag enters the preimage, so this is close to free at
the protocol level if sequenced with Phase 3.

**Does the eDSL fork per hasher?** The lead asked explicitly, and the answer
matters:

- **B1 — the sponge becomes compress-based for ALL hashers. ★ the right
  version.** `Test` and `Poseidon` both implement `compress`, so nothing breaks.
  One transcript construction, one security argument, one host mirror. It also
  opens the door to dropping `permute` from `LfmHasher` entirely later.
- **B2 — fork per hasher (permute-based for Test/Poseidon, compress-based for
  BLAKE3). ✗ reject.** Two transcript constructions means two security
  arguments, two host mirrors, and a program whose *meaning* depends on which
  hasher verified it. That is a trap, not a compromise.

Costs of B, stated honestly: it changes a shared construction that the
Test/Poseidon paths currently exercise green; it removes `TrivialV0`'s
deliberate permute coverage (that program would need to either drop the call or
be retained as a Test/Poseidon-only fixture); and if the ecosystem later wants a
genuine 12-felt sponge for some other protocol, B does not provide one — though
**A can always be added later on top of B**, which is not true in reverse.

**Gate extension: TRIVIAL.** The gated surface does not change at all. The
75/75 board already covers the compress socket, and the two-tag variant is a
`Framing.tag_word` parameter my model already has. The `MODE_P = 0` audit (B0a)
stays valid *permanently* instead of being deleted.

**F3.4:** fully retired for BLAKE3 programs.

---

## 4. Option C — mixed hasher: BLAKE3 compress, Poseidon permute

**Cost — and this is C's case.** ✓ EXECUTED, derived from
`chips.rs:515-578` + `poseidon.rs:48-53` (30 rounds, 8 full + 22 partial, S-box
on all 12 lanes in full rounds and lane 0 in partial):

Poseidon's arm is **584 appended witness cells + 28 shared prefix = 612 main**,
601 constraints, and — because it is pure field arithmetic with **no BITWISE
traffic at all** — only the 6 `LfmMem` sends, so aux ≈ 9. **≈ 621 cell-equiv per
permute, ~9× cheaper than a BLAKE3 permute (5,617).**

| program | option A | **option C** | C vs A |
|---|---:|---:|---:|
| `FriToyV0`, 7r | 364,674 | **314,714** | **−13.7%** |

That is a real saving and it should not be dismissed.

**What it costs instead.**

- **Two primitives in the trusted base**, two KAT stories, two parameter
  sign-offs. Poseidon-Goldilocks's round counts and MDS would need their own
  review; ✓ VERIFIED the file cites Plonky3 and pins a known-answer vector, but
  "matches Plonky3's vector" establishes *correctness of transcription*, not
  *security of the parameters for this use*.
- **F3.3 blocks it today.** ✓ VERIFIED (recorded in the same findings file): the
  registry verify path cannot reach Poseidon; it is measurement-only. That must
  be fixed first.
- **The registry binds one hasher per entry.** Phase 3 added `hasher:
  HasherKind` as a single field folded into `lfm_program_id`. A mixed machine
  needs either two fields or a composite variant
  (`Blake3CompressPoseidonPermute`) — a wire-format and digest-preimage change,
  and a new way for the binding to be got wrong.
- **The disclosure gets stranger, not simpler.** F3.4 *would* be retired — both
  are real hashes — but it is replaced by a standing note that *this machine's
  Merkle tree and its Fiat–Shamir transcript rest on different primitives*. That
  is an unusual sentence to have to write, and it doubles the cryptanalytic
  surface a reviewer must cover.

**Gate extension.** My gate says nothing about Poseidon and would not; a second
gate for the Poseidon arm is a separate project of comparable size to the BLAKE3
one. Feasible in QF-BV but far less natural — Poseidon is field arithmetic, so
a bit-vector model is the wrong tool and it would want a field-domain gate
throughout.

---

## 5. Comparison

| | **A — `"LFMP"` transform** | **B — compress-based sponge** | **C — mixed hasher** |
|---|---|---|---|
| new named assumption | **A-TSP** (T-sponge instantiation) | **none beyond A6R** | Poseidon parameter sign-off |
| security argument | random-transformation duplex; new theorem | textbook FS hash chain | two independent arguments |
| security bound | ~64-bit (128-bit capacity) | ~64-bit (128-bit state) | ~64-bit / Poseidon-dependent |
| KAT-able vs `blake3` crate | ✓ EXECUTED (XOF 64B = `out[0..16]`) | ✓ same as compress, no new machinery | partially — Poseidon has no published crate KAT |
| cost / permute-equivalent | 5,617 (1.02× compress) | 5,509 (1 compress) | **621 (0.11×)** |
| `FriToyV0` total, 7r | 364,674 | 369,103 (+1.2%) | **314,714 (−13.7%)** |
| blast radius | **large** — a second socket, layout, executor, filler, KATs | moderate — `edsl.rs` + `fixture.rs` + one call site | large — F3.3 fix, registry shape, second gate |
| `MODE_P` | un-pinned; idx 5 deleted | **stays pinned to 0 permanently** | un-pinned (Poseidon uses it) |
| gate extension | high feasibility, real work | **trivial — surface unchanged** | separate project, wrong tool |
| primitives in TCB | 1 | **1** | 2 |
| retires F3.4 | yes | yes | yes |
| reversible? | adds a permanent socket | **yes — A can be added later on top** | registry shape change is sticky |

---

## 6. What I could not settle

- ~~✗ **The T-sponge bound for this exact construction**~~ → **RESOLVED in §8**
  (2026-08-11). It is citable: Eurocrypt 2008 covers random transformations, and
  duplex + overwrite mode compose onto it. **But** transformation-based sponges
  carry a cryptanalytic caveat (entropy loss under iteration) that applies to
  option B as well — see §8.2/§8.3. Read §8 before using §7.
- ✗ **Poseidon's cost is a column count, not a measurement.** 621 cell-equiv is
  derived from the AIR's own `const fn`s (✓ VERIFIED arithmetic) but nothing was
  proved or benched.
- ✗ **Whether the ecosystem's real transcript will want a sponge shape.** If the
  eventual production transcript is specified as a sponge by an external
  standard, B's chain would have to be revisited. Nobody has told me what that
  transcript is; `edsl.rs:6-10` says it "lands with the ecosystem hash decision",
  which is this one.

---

## 7. ★ MY RECOMMENDATION (the decision is the user's)

**Take option B — redesign the sponge as a compress-based chain, in its B1 form
(compress-based for all hashers), and never build a permute socket.**

Five reasons, in the order I weight them:

1. **It needs no new assumption.** A6R already covers PRF-for-Fiat–Shamir, and
   B's construction is the textbook FS transcript. Option A needs A-TSP — a new,
   signable, currently-unwritten construction assumption — and §6 says I could
   not settle its bound. Adding an assumption to a project whose whole thesis is
   "one primitive, externally anchored" is the wrong direction.
2. **It removes a gated surface instead of adding one.** `MODE_P = 0` stays
   pinned permanently, the 75/75 board keeps covering everything, and the gate
   extension is a `tag_word` parameter I already have. A adds a second socket
   that needs its own layout, its own KATs, its own gate pass, and deletes the
   idx-5 audit in the process.
3. **Cost does not decide between A and B** — 1.2% on `FriToyV0`, and B is
   *cheaper* on `TrivialV0`. Anyone choosing A for performance is paying an
   assumption for noise.
4. **The sponge is already slated for replacement.** `edsl.rs:6-10` says so in
   as many words. B is the scheduled work; A builds a permanent socket to
   preserve the shape of a construction that is explicitly provisional.
5. **B is reversible and A is not.** A permute socket, once registered, has a
   `program_id`-bearing footprint forever. If the ecosystem later demands a true
   12-felt sponge, A can be added on top of B; B cannot be recovered after A.

**On option C:** its 13.7% saving is real and it is the only option that would
change my mind on cost grounds — but it puts a second primitive in the trusted
base to save one-seventh of two test programs that the wrap does not even use.
I would revisit C only if a *production* workload turns out to be
transcript-dominated, which today's numbers say it is not (56 compresses to 10
permutes in the one program that has both).

**If the user picks B, the next steps are:** sequence it with Phase 3 so the
`program_id` re-bless happens once; add the `TAG_LFMT` allocation beside
`"LFMC"`/`"LFMP"`/`"LFML"`; decide `TrivialV0`'s fate (drop its raw `permute`, or
keep the program as a Test/Poseidon-only fixture); rewrite `SpongeVar` +
`HostSponge` together so the bit-exact mirror property is preserved; and re-run
the gate with the two-tag framing, which is a parameter change rather than new
gate code.

**If the user picks A instead**, the work is well-understood and my gate extends
cleanly — but A-TSP must be written down and signed *before* the arm is built,
the same way A6R was, and `SOCKET.md` §7's sketch should not be treated as the
spec until it has had the review the compress socket got.

---

## 8. ADDENDUM (2026-08-11) — A-TSP researched: it is citable, with a caveat.
## This UPGRADES option A and I am reporting it against my own recommendation.

§6 listed A-TSP as ✗ OPEN — cited from memory, unsigned. I went and checked.
The result is better for option A than I represented, and worse in one specific
way that nobody had named. Both directions below.

### 8.1 The theorem is real and it composes

- **Sponge indifferentiability holds for a random TRANSFORMATION, not only a
  permutation**, up to the birthday-type bound `O(2^{c/2})` — Bertoni, Daemen,
  Peeters, Van Assche, *On the Indifferentiability of the Sponge Construction*,
  Eurocrypt 2008. This is the load-bearing citation and it directly covers the
  fact that `out[0..12]` is non-invertible.
- **Duplex security reduces to sponge indifferentiability**, same `O(2^{c/2})`
  bound — *Duplexing the Sponge*, SAC 2011.
- **Overwrite-mode absorb is a known, analysed variant**: the XOR at absorb can
  be omitted while maintaining the chosen security level.

With `c` = 1 cell = 128 bits that is `O(2^64)` — the number already stated in
§2, now with a reference under it rather than my recollection.

**So A-TSP is no longer an unwritten assumption; it is a composition of three
published results.** That was my first reason for preferring B, and it is
materially weaker than I wrote. Stated plainly because it argues against me.

### 8.2 The caveat, which is specific to transformation-based sponges

T-sponges have a **dedicated cryptanalytic literature that P-sponges do not**,
and a broken real-world instance:

> *Collision Spectrum, Entropy Loss, T-Sponges, and Cryptanalysis of GLUON-64*
> (FSE 2014). Iterating a permutation loses no entropy; iterating a
> **transformation** does — the image shrinks with each application, collision
> trees grow quadratically, and certain collision-spectrum and rate values yield
> **improved preimage attacks on long messages**. GLUON-64 was broken this way.

Option A's map is a transformation, so it sits in exactly that family. The
attacks bite on *long* iteration counts; our transcript is ~10 applications, at
which the image has shrunk by around a bit. **Quantitatively irrelevant here —
but that is a regime-dependent argument, and it has to be written down and
bounded rather than assumed.** A6R is not regime-dependent; A-TSP would be. A
signer needs to see the iteration-count bound stated as part of the assumption.

### 8.3 ⚠ The same caveat applies to option B, and I did not say so before

Being even-handed: option B's squeeze is `state = compress_T(state, DOMAIN)`
with `DOMAIN` **constant**, so a run of consecutive squeezes iterates a fixed
non-injective map exactly as a T-sponge does. ✓ VERIFIED that such runs exist —
`programs.rs:550-551` squeezes twice back-to-back, and the query loop
(`programs.rs:565-567`) squeezes once per query with no absorb between, so runs
of ~4–5 occur.

The entropy-loss analysis is therefore **the same for A and B**, and equally
negligible at these lengths. B is not immune, and anything above implying it was
should be read as corrected here. **This axis does not separate the options.**

### 8.4 Does the recommendation change? No — but the margin narrows

Reason 1 of §7 ("needs no new assumption") must be restated honestly:

> B needs collision-resistance and RO-behaviour of the compression function —
> already inside A6R. A needs that **plus** the T-sponge + duplex + overwrite
> composition **plus** a written iteration-count bound. Both are defensible; A
> simply has more moving parts, each of which someone must check.

That is a real difference but a smaller one than §7 implied. Reasons 2–5 —
removes gated surface rather than adding it, cost is a tie, the sponge is
already slated for replacement, and B is reversible where A is not — are
untouched by this research and are what the recommendation now mostly rests on.

**Recommendation stands: option B.** If the user prefers A, §8.1 means it can
proceed on citations rather than on a novel assumption — provided A-TSP is
written with the iteration bound of §8.2 in it, and signed, *before* the arm is
built.

**Sources:**
- [On the Indifferentiability of the Sponge Construction (Eurocrypt 2008)](https://keccak.team/files/SpongeIndifferentiability.pdf)
- [Duplexing the Sponge (SAC 2011)](https://link.springer.com/chapter/10.1007/978-3-642-28496-0_19)
- [Collision Spectrum, Entropy Loss, T-Sponges, and Cryptanalysis of GLUON-64](https://link.springer.com/chapter/10.1007/978-3-662-46706-0_5)
- [The sponge and duplex constructions (keccak.team)](https://keccak.team/sponge_duplex.html)

### 8.5 "For this exact construction" — the part the general theorems do not cover

The citations in §8.1 are about *idealised* sponges. They say nothing about the
specific map each option instantiates. Three construction-level notes, the last
of which is a concrete difference between A and B that I had not previously
identified.

**(i) The idealisation step is itself an assumption.** T-sponge results model `T`
as a *random* transformation. Option A's `T` is BLAKE3's compression with a
**fixed** chaining value (`IV`) and a **fixed** tag word — a single public
function, not a random one. Treating it as ideal is the standard move and is the
same move A6R already makes, but it is a step, and A-TSP's text should contain
it rather than leave it implicit.

**(ii) Rate exceeds capacity, which is fine but worth stating.** Option A's state
is 3 cells = 384 bits, split rate 256 / capacity 128. The bound depends only on
the capacity, so `O(2^{c/2}) = O(2^64)`; the wide rate buys throughput, not
weakness. Same number as the digest's, from the same 128-bit-cell cause.

**(iii) ⚠ Option A's 12-word output exposes final-state words that option B's
4-word output does not.** ✓ EXECUTED (200 random inputs, exact):

With `h = IV`, BLAKE3's output is `out[i] = v[i] ^ v[i+8]` and
`out[i+8] = v[i+8] ^ IV[i]`. Taking **twelve** words therefore publishes both
halves of a cross-relation:

```
out[i] ^ out[i+8] == v_final[i] ^ IV[i]        (i in 0..8)   -> v_final[0..4] recoverable
out[8+i]          == v_final[8+i] ^ IV[i]      (i in 0..4)   -> v_final[8..12] recoverable
```

So from one option-A permute output a reader recovers **8 of the 16 final state
words directly** — `v_final[0..4]` and `v_final[8..12]` — by XOR with public
constants.

Option B's socket publishes **four** of the sixteen words, so the same query
gets nothing comparable: `out[0..4] = v[0..4] ^ v[8..12]`, and with no second
output block to cross-XOR against, the two summands cannot be separated
(✓ EXECUTED, 2000/2000 samples). Twelve words stay unpublished.

**Is (iii) an attack? I do not have one, and I am not claiming one.** The final
state is still a pseudorandom function of the input, so recovering it from the
output is not obviously exploitable — this is a *structural observation*, of the
kind that belongs in a security argument a reviewer signs rather than in a
footnote. But it is a real asymmetry, it points the same way as everything else
in §7, and it is the sort of thing that has historically been the first step of
a T-sponge attack (§8.2's GLUON-64 line began with structure, not with a break).

Note also that option A's state includes four words BLAKE3's own chaining value
never propagates: standard BLAKE3 chains on `out[0..8]` and uses `out[8..16]`
only as extended output. Option A would make XOF words part of the *chaining
state* — specified, KAT-able (§2), but a role BLAKE3's designers analyse as
output rather than as state.

**Net effect on the recommendation: unchanged, slightly reinforced.** §8.1 moved
option A's assumption from "unwritten" to "citable"; §8.5(iii) adds a
construction-level reason that points back the other way. Option B remains the
one with fewer moving parts and less exposed structure.
