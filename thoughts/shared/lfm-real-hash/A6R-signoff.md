# A6R — the 6-round BLAKE3 assumption: decision sheet

**For:** the user, to sign or decline. **From:** Phase 1. **Date:** 2026-08-10.
**One question:** *is a named, unratified, non-interoperable assumption worth ~5.5% of the epoch column?*
**Recommendation: no — instantiate 7 rounds. Keep 6 as a measured variant behind an explicit signature.**

---

## 1. The assumption, stated precisely

**A6R:** *the 6-round internal variant of the BLAKE3 compression function is
collision resistant.*

It is named and recorded as unratified at `prover/src/lfm/blake3.rs:40-42`
(✓ VERIFIED), which says outright that the module "exists to price the AIR, not
to endorse the hash". It originates in PR #903's `IMPLEMENTATION.md`.

**Today A6R costs nothing.** `LFM_BLAKE3` is unregistered and unreachable.
✓ VERIFIED independently, not inherited from the plan: `grep -rln blake3
prover/src crypto executor` returns only `lfm/blake3.rs`, `lfm/blake3_chip.rs`,
`lfm/blake3_probe.rs` and `lfm/mod.rs` (the module declarations), and grepping
`LFM_BLAKE3|Blake3` across `airs.rs`, `instr.rs`, `compiler.rs`, `executor.rs`
and `trace.rs` returns **nothing**. **The moment BLAKE3 becomes a
selectable hasher with a registry entry, A6R becomes a live soundness surface
for every program that selects it.** That is what needs a signature — not the
code, the exposure.

### 1.1 What the spec actually says — quoted, not paraphrased

⚠ **Correction to `PLAN.md` §7.** The plan renders the external-review note as
ending *"variants below 6 rounds are out of scope and MUST NOT be
instantiated."* That is a strengthening of the source. The actual text
(`git show 783c5a95:spec/blake3.typ`, ✓ VERIFIED by reading) is:

> *External review (2026-08).* The round-count choice was reviewed with external
> symmetric-cryptography experts consulted by the project: removing *one* round
> (7 → 6) was judged comfortable; removing *two* (7 → 5) was explicitly not.
> Accordingly, 6 rounds is the endorsed floor. Variants below 6 rounds are not
> formally ruled out, but they are not available on the project's own authority:
> adopting one would require the external experts to study the reduced-round
> margin specifically — a dedicated cryptanalytic review, not an engineering or
> configuration decision.

"Not available on the project's own authority" is a procedural bar, not a
prohibition. The distinction matters for how §6's third record item is worded.

The spec also supplies the context that argues *for* A6R, and it belongs on a
fair decision sheet:

> (Precedent: KangarooTwelve's reduced-round Keccak. Best public cryptanalysis of
> BLAKE3 reaches far fewer rounds; the margin removed here is one round of seven.)

and the scope of what the assumption covers, which is wider than the compress
socket alone:

> *A6R.* The BLAKE3 compression function restricted to 6 rounds is
> collision-resistant and suitable as a 2-to-1 compression for Merkle hashing
> **and as a PRF for Fiat–Shamir**, in the same sense the full 7-round function
> is believed to be.

> Any use of BLAKE3 as a Merkle or transcript hash *invokes this assumption*. The
> z3 gate proves the chip computes 6-round BLAKE3 correctly; it neither proves
> nor addresses whether 6 rounds are secure.

So A6R is not reckless. It is *reviewed but unratified*, non-interoperable by
construction, and it covers the transcript sponge as well as the Merkle compress.

### 1.2 ⚠ This sheet's recommendation reverses the spec's recorded default

State this plainly rather than letting it pass. `spec/blake3.typ` records:

> The 6-round variant is the primary internal target per the review above; the
> 7-round variant is the interoperability / zero-assumption fallback. If both are
> instantiated they are distinct chips with distinct ECALL numbers.

§5 below recommends the opposite ordering — 7 primary, 6 as the measured
variant. That is a deliberate disagreement, argued on the reference chain rather
than on cryptanalysis, and **if it is accepted the spec section must be updated
to match**, or the tree will carry two contradictory statements of intent. The
plan (§7) reaches the same recommendation; neither it nor this sheet is a
cryptographic re-assessment of the 6-round margin.

## 2. What 7 rounds buys

Setting the round count to 7 makes the primitive **bit-identical to published
BLAKE3**. Concretely, and this is the argument:

- **The reference problem dissolves.** Today the chain is: official crate
  vectors pin the oracle at 7 rounds → the oracle at 6 rounds emitted ten
  vectors → those vectors pin the Rust port. `blake3.rs:33-38` describes its own
  anchor as "one step removed… weaker than a direct KAT and is recorded as
  such" (✓ VERIFIED). At 7 rounds there is no step removed: the `blake3` crate
  *is* the KAT.
- **The 2-to-1 socket becomes a library call too.** Phase 1 specified the socket
  so that `compress(a, b) = blake3::hash(a ‖ b ‖ "LFMC")[0..16]`
  (`thoughts/blake3/socket-kats/SOCKET.md`). At 7 rounds that identity is
  checkable in one line against the crate — **already executed** against
  upstream BLAKE3's C, which passes the official vectors in all three modes. At
  6 rounds the socket vectors can only ever come from our own two sources.
- **Nothing to ratify, re-litigate, or disclose at audit.** No assumption in
  `SOUNDNESS.md`, no caveat on the registry entry, no "MUST NOT go below 6" rule
  to enforce in perpetuity.
- **Interoperability.** 7-round parent merges are bit-compatible with published
  BLAKE3, so an external verifier can recompute a tree. 6-round merges are
  computed by nothing else in the world.

## 3. What 6 rounds buys: the cost delta

✓ MEASURED, at 6 rounds, standalone prove+verify against the production
`BITWISE` table (`prover/src/lfm/blake3_probe.rs:327-356`, re-read and
confirmed):

```
main columns 3,056 + 3 × aux 630 = 4,946 base-field-equivalent cells / compression
interactions = 11 + 832 + 384 + 32 = 1,259 ;  aux = ceil(1259/2) = 630
```

? INFERRED for 7 rounds — arithmetic over the chip's own parameterised formulas,
shown so it can be rechecked. `NUM_G = BLAKE3_ROUNDS * 8` goes 48 → 56:

```
main columns    3,056 + 8 G-blocks × 60 cells                     = 3,536
BITWISE XOR     (56×4 + 16) × 4        (was (48×4+16)×4 = 832)    =   960
shift halfwords 56 × 2 × 4             (was 384)                  =   448
message bytes   unchanged                                          =    32
LfmMem tokens   unchanged                                          =    11
interactions    960 + 448 + 32 + 11    (was 1,259)                = 1,451
aux             ceil(1451 / 2)         (was 630)                  =   726
base-equiv      3,536 + 3 × 726        (was 4,946)                = 5,714   (+15.5%)

epoch column    2.752 B − 967 M + (195,593 × 5,714 = 1.118 B)     = 2.903 B (+5.5%)
                vs keccak 11.166 B                                 = 3.85×  (was 4.06×)
```

**Cross-check against the spec's independent figure.** `spec/blake3.typ` states
7-round costs "roughly 10–12% more per merge end-to-end", and its cost section
gives "≈7,194 committed cell-equivalents per compression end-to-end (≈5,316
table-only)". So ≈1,878 cell-equivalents per merge are *not* table cells and do
not grow with the round count. Applying +15.5% to the table part alone:
`5,316 × 1.155 = 6,140`, so end-to-end goes `7,194 → 8,018`, i.e. **+11.5%** —
inside the spec's 10–12%. ? INFERRED but it is two independent routes agreeing,
which is real evidence for both.

**So A6R buys ≈ 5.5% of the epoch column.** For scale, the plan's own §6.1 notes
that the felt-absorbing variant is a ~2.5× lever on the same column — an order
of magnitude more leverage than the round count.

## 4. The switch is a constant, not a redesign

✓ VERIFIED, and this materially changes the cost of choosing 7 — the chip is
**already round-parameterised**:

- `BLAKE3_ROUNDS = 6` (`blake3.rs:56`); the primitive's loop permutes the
  schedule when `r < BLAKE3_ROUNDS - 1` (`blake3.rs:106-125`), so setting it to
  7 yields standard BLAKE3's `f` with no other edit.
- `NUM_G: usize = BLAKE3_ROUNDS * 8` (`blake3_chip.rs:98`); the column layout
  derives from `NUM_G` (`blake3_chip.rs:157`), the dataflow loops
  `for r in 0..BLAKE3_ROUNDS` (`blake3_chip.rs:280`), and
  `NUM_CONSTRAINTS = 16 × NUM_G + 1` (`blake3_chip.rs:1042`).

So the plan's "build round-parameterised" recommendation is already satisfied.
Flipping the constant re-derives the layout, the constraints and the census.

**Not literally one line, and the difference matters:** the 6-round *expected
values* are baked into tests as literals — `blake3_probe.rs` asserts `2_880`,
`1_259`, `4_946` and `769`, and `blake3_probe.rs:403-421` checks the chip's `OUT`
columns against `CANONICAL_VECTORS`, which are 6-round-specific. A 7-round
instantiation needs those expectations regenerated (to `3,360` / `1,451` /
`5,714` / `897`) and needs 7-round vectors, which — unlike the 6-round ones —
come straight from the crate. ? INFERRED for the four projected constants; they
are compile-time consts and a build phase can confirm them in one `cargo test`.

## 5. Recommendation

**Instantiate 7 rounds. Do not sign A6R.**

In order of weight:

1. The reference problem dissolves — the crate becomes a direct external KAT for
   both the primitive and the socket framing, satisfying standing rule 9 in its
   intended form rather than one step removed.
2. No assumption to sign, ratify, or defend at audit; no floor rule to police.
3. Bit-compatibility with published BLAKE3 is worth something on its own.
4. 5.5% is inside the noise of the decisions still open above it.
5. The switch costs a constant plus regenerated test expectations (§4), and the
   6-round path stays available as a measured variant.

Keep 6-round behind `BLAKE3_ROUNDS` as the performance variant, and switch to it
**if and when** the 5.5% matters — at which point it needs the signature below
and not before.

## 6. If you sign A6R anyway

The record needs all four of these, and the first is the one usually missed:

1. The assumption named in `prover/src/lfm/SOUNDNESS.md`, not only in
   `blake3.rs`'s module header — a soundness surface belongs in the soundness
   document.
2. The registry entry's doc comment stating which hash its `program_id` rests on.
3. A note that sub-6-round variants are **not available on the project's own
   authority** and would need a dedicated external cryptanalytic review —
   quoting §1.1's actual wording, not the plan's stronger paraphrase.
4. The 6-round socket vectors (`thoughts/blake3/socket-kats/socket_kats.json`,
   `rounds.6`) as the only reference that will ever exist for the socket, with
   `SOCKET.md` §6's deferred-crate-check row struck through as unachievable
   rather than pending.

---

## Sign-off

> **A6R** — the 6-round BLAKE3 internal variant is collision resistant. Reviewed
> by external symmetric-cryptography experts as "comfortable" at one round
> removed; unratified; non-interoperable. Buys ≈ 5.5% of the epoch column.

- [x] **Start with 7 rounds** (recommended path) — decision recorded from the user
  in-session, 2026-08-10: *"6 or 7 rounds is the same, we have the greenlight from
  symmetric cryptographers to use 6. But we can start with 7, as long as it works
  I don't care."*
- [ ] **Sign A6R — instantiate 6 rounds**, and complete §6's four record items

Signed: ______________________  Date: ____________

### Decision record — 2026-08-10

The user confirmed the round-parameterised build (`BLAKE3_ROUNDS` knob, both counts
compiled and swept) with **7 rounds as the instantiated baseline**. The external
greenlight for 6 rounds is acknowledged and consistent with §1.1's quoted review
note; the 6-round variant stays available behind the knob. This is **not** a
signature on A6R: per §5, the signature (and §6's four record items) become due
if and when the default switches to 6 — not before.

Follow-up when Phase 2 lands: update `spec/blake3.typ`'s primary/fallback ordering
(§1.2) so the spec and this sheet agree — 7 primary, 6 as the measured variant.
