# Phase 1 report — a trustworthy 6-round reference and the KATs it pins

**Date:** 2026-08-10. **Worktree:** `/Users/maurofab/workspace/lambda_vm-blake3-impl` (branch `blake3-real-hash`).
**Status:** complete, uncommitted, nothing pushed. **No cargo was run.**
**Headline:** the 6-round vector set is no longer single-source, and the anchor holds.

Claims are marked ✓ VERIFIED / ✓ EXECUTED, ? INFERRED, or ✗ OPEN.

---

## 1. Result in one paragraph

The external anchor is green and the vectors now rest on **two independent
sources**. `thoughts/blake3/`'s reference material was restored from git;
`test_oracle.py` reproduces the official BLAKE3 vectors at 7 rounds (35 cases ×
3 modes) and regenerates `canonical_6round_vectors.json` **byte-identical to the
git blob**. A second source — upstream BLAKE3's own portable **C**
implementation, round-parameterised in a 2-hunk diff — independently reproduces
all ten `CANONICAL_VECTORS` at 6 rounds and matches the Rust constants in
`prover/src/lfm/blake3.rs` directly. The 2-to-1 compress socket is now specified
and pinned with its own vectors and nine negative controls, framed so that at 7
rounds it is literally `blake3::hash(a ‖ b ‖ "LFMC")` truncated. The A6R sheet
is written and recommends declining the assumption.

## 2. Task 1 — restoration and the anchor

All nine artifacts restored into the worktree with `git show`, per plan §9.
✓ EXECUTED.

| file | source commit |
|---|---|
| `blake3-oracle/{blake3_ref.py, ORACLE.md, test_oracle.py}` | `3b9b8137` |
| `blake3-chip/{DESIGN.md, z3_blake_verify.py}` | `3b9b8137` |
| `blake3-oracle/{official_test_vectors.json, canonical_6round_vectors.json}` | `19ed761b` |
| `ground-truth/{Cargo.toml, src/main.rs}` | `19ed761b` |

`python3 test_oracle.py` (Python 3.14.6, ~4 s):

```
[1] Official test_vectors.json : PASS  (35/35 cases x 3 modes)
[2] Official `blake3` PyPI pkg : SKIP  (package not importable)
[3] Plonky3 blake3-air (direct): PASS  (20000 random compressions, flags=0)
[.] Internal self-consistency  : PASS  (1000 checks)
[4] 6-round variant derivation : PASS  (differs from 7r on 2000/2000)
```

**The anchor holds.** ✓ EXECUTED. Anchor 2 (the `blake3` PyPI package) is
unavailable in this environment — noted, not a gap, because Task 2 replaced it
with something stronger (§3).

Two extra checks beyond the brief, both ✓ EXECUTED:

- The regenerated `canonical_6round_vectors.json` is **byte-identical** to the
  git blob. So the oracle is deterministic across the Python version change, and
  the recorded vectors were not edited after generation.
- The ten `CANONICAL_VECTORS` in `prover/src/lfm/blake3.rs:151-342` were parsed
  and compared field by field against that JSON: **`h`, `m`, `t`, `block_len`,
  `flags` and all 16 `out` words match on all ten**. The transcription into Rust
  is exact. (The repo asserts the port *reproduces* the vectors; nothing
  previously asserted the transcribed *constants* match the JSON.)

## 3. Task 2 — the second independent source

**What I used, and why it beats the brief's suggestion.** The brief proposed
vendoring upstream's `reference_impl/reference_impl.rs`. That file is not
shipped in the published `blake3` crate (✓ VERIFIED — the crate's contents are
`src/`, `c/`, `benches/`, `tools/`; no `reference_impl`). But the crate *does*
ship upstream's portable **C** implementation, `c/blake3_portable.c`, together
with the full tree hasher `c/blake3.c`. That is better on every axis that
matters: same authors as the reference impl, a different language from the
Python oracle, and — decisively — **a different construction of the message
schedule**. The C indexes a precomputed `MSG_SCHEDULE[7][16]` table; the Python
oracle and the Rust port iteratively apply one permutation between rounds. A bug
in the iterative composition is exactly the class of error a single source
cannot catch, and this second source catches it.

Vendored at `thoughts/blake3/reference-impl/` (BLAKE3 is CC0/Apache-2.0;
`LICENSE_CC0` copied alongside). `upstream/` holds `blake3.c`,
`blake3_dispatch.c`, `blake3_impl.h`, `blake3.h`, `blake3_portable.c`
**verbatim**. The single modified file is `blake3_portable_paramrounds.c`, and
the entire diff is in `PARAMETERISATION.diff` — two hunks:

1. a `BLAKE3_ROUNDS_PARAM` `#define` defaulting to 7, with an `#error` guard at
   `> 7` (`MSG_SCHEDULE` has exactly 7 rows);
2. the seven literal `round_fn(state, &block_words[0], 0..6)` calls replaced by
   `for (size_t r = 0; r < BLAKE3_ROUNDS_PARAM; r++) round_fn(state, &block_words[0], r);`

At the default the loop issues the identical seven calls in the identical order,
so the parameterisation is **inert by inspection** — and then re-checked
empirically. NEON is disabled and no x86 SIMD applies, so the dispatcher resolves
every compression to the portable path; the round knob therefore governs the
*whole tree hasher*, not just a directly-called compress.

`python3 check.py` (after `./build.sh`, ~2 s), all ✓ EXECUTED:

```
PASS [A] parameterised C @ rounds=7 vs official vectors (35 cases x 3 modes)
PASS [B] rounds=6 differs from rounds=7 on all 8 probe lengths
PASS [C] MSG_SCHEDULE[r] == permute^r(identity) for r in 0..7
PASS [C] MSG_SCHEDULE[1] == the repo's BLAKE3_MSG_PERMUTATION
PASS [D] C @ rounds=6 == canonical_6round_vectors.json (all 10, 16 words)
PASS [D] C @ rounds=6 == Rust CANONICAL_VECTORS in prover/src/lfm/blake3.rs
PASS [D] Rust vector INPUTS == JSON vector inputs
PASS [D] negative control: C @ rounds=7 matches none of the 10 vectors
PASS [E] C vs Python oracle @ rounds=7 (5000 random compressions)
PASS [E] C vs Python oracle @ rounds=6 (5000 random compressions)
```

**Plan §2.2 step 3's acceptance criterion is met: both sources, at rounds = 6,
reproduce all ten `CANONICAL_VECTORS` byte for byte.** Check [C] is the one that
earns the "independent" label — it proves the two *different* schedule
constructions denote the same function.

**Deferred to a build phase (✗ OPEN, needs cargo):** the equivalent check
against the Rust `blake3` crate, i.e. plan §2.2 step 4's direct KAT of `f` at
7 rounds via `blake3::hash`. Low risk — the C that was checked *is* upstream
BLAKE3 and passes the official vectors in three modes — but it should still be
written, because it is the form of the check that survives this directory being
deleted. `thoughts/blake3/ground-truth/` (restored, links the real crate) is the
place for it.

## 4. Task 3 — the socket specification and its KATs

`thoughts/blake3/socket-kats/` — `SOCKET.md` (the spec), `gen_socket_kats.py`
(the generator and its checks), `socket_kats.json` (the vectors).

**The decision, as instructed: Option A + domain separation, 128-bit digest.**
Byte-level normative form:

```
msg = LE32(a0..a3) ‖ LE32(b0..b3) ‖ "LFMC"          (36 bytes)
c   = LE32⁻¹( BLAKE3(msg)[0..16] )                   (4 u32 lanes, 1 cell)
```

Realised as one compression: `h = IV`, `m[0..4] = a`, `m[4..8] = b`,
`m[8] = 0x434D464C`, `m[9..16] = 0`, `t = 0`, `block_len = 36`,
`flags = 0x0B (CHUNK_START|CHUNK_END|ROOT)`, digest = output words `0..4`.

**The one design choice worth surfacing: the domain tag goes in the message, not
in `flags`.** Plan §5 option D says "domain separation in `flags`", which is
where BLAKE3 itself puts domain bits. But any tag in `flags` (or `t`, or `h`)
makes the socket a *nonstandard* invocation that no library computes, so its
KATs could only ever come from our own oracle — at 7 rounds as well as at 6,
which throws away the main reason to prefer 7. Putting the tag in the message
keeps the socket a standard BLAKE3 hash of a domain-separated byte string, at a
cost of 4 bytes in a block that had 28 spare. Domain separation is equally real.

Vectors: 10 inputs × 2 round counts, all inputs written out explicitly (no RNG
dependence), each with **9 negative controls** — `swap_a_b`, `tag_changed`,
`tag_omitted`, `truncate_high_half`, `flags_parent`, `block_len_64`,
`counter_one`, `lanes_big_endian`, `other_round_count`. Three computations must
agree per vector (Python word-level, C word-level, C **whole-tree** byte-level);
the generator fails loudly otherwise. All ✓ EXECUTED, both round counts.

The 7-round cross-check the brief flagged for the build phase is **already
executed** here, against upstream C rather than the Rust crate: every 7-round
vector equals `BLAKE3(a ‖ b ‖ "LFMC")` truncated to 16 bytes. Only the
`blake3`-crate restatement remains ✗ OPEN.

**A control that fired, and what it taught.** `lanes_big_endian` initially
failed on three of the ten vectors — `zeros`, `all_ones`, `nibble_ramp`. Not a
bug: every lane of those inputs is a byte-palindrome (`0x00000000`,
`0xFFFFFFFF`, `0x11111111`, …), so byte-order cannot be observed on them. Rather
than skip it, the generator now declares applicability per control per vector
**and separately asserts every control is discriminated by at least one
vector** — otherwise a framing degree of freedom would sit unpinned behind a
green run. Worth keeping in mind for the chip tests: three of the five obvious
structural inputs cannot detect a byte-order error.

## 5. Task 4 — the A6R sheet

`thoughts/shared/lfm-real-hash/A6R-signoff.md`. One page, with a signature block
offering "decline (recommended)" or "sign, and complete these four record
items". Recommendation: **decline A6R, instantiate 7 rounds.**

Two findings that changed the sheet relative to the plan:

**(a) ⚠ `PLAN.md` §7 misquotes the spec, in the strengthening direction.** The
plan renders the external-review note as ending *"variants below 6 rounds are out
of scope and MUST NOT be instantiated."* The actual text
(`git show 783c5a95:spec/blake3.typ`, ✓ VERIFIED by reading) says variants below
6 rounds are *"not formally ruled out, but they are not available on the
project's own authority"* — a procedural bar requiring dedicated external
cryptanalysis, not a prohibition. The sheet quotes the source. The plan should be
corrected.

Reading the source also surfaced context the plan omits, in both directions: the
precedent argument for A6R (KangarooTwelve; "the margin removed here is one round
of seven"), and the fact that **A6R covers Fiat–Shamir as well as Merkle
compression** — "suitable as a 2-to-1 compression for Merkle hashing *and as a
PRF for Fiat–Shamir*". Both are in the sheet.

**(b) ⚠ The sheet's recommendation reverses the spec's recorded default,** which
says "the 6-round variant is the primary internal target… the 7-round variant is
the interoperability / zero-assumption fallback". That disagreement is now stated
explicitly in the sheet rather than left implicit, with the note that accepting
the recommendation requires updating `spec/blake3.typ` or the tree will carry two
contradictory statements of intent.

**Cost numbers re-derived from source, not taken from the plan.** ✓ VERIFIED
`4,946 = MAIN_COLUMNS 3,056 + 3 × aux 630`, `interactions = 11 + 832 + 384 + 32
= 1,259` (`blake3_probe.rs:327-356`). ? INFERRED for 7 rounds: `5,714` per
compression (+15.5%), epoch column `2.903 B` (+5.5%), `3.85×` keccak. The plan's
arithmetic checks out. I added an independent cross-check the plan asserts but
does not show: applying +15.5% to the spec's own table-only figure (5,316 of
7,194 end-to-end) gives +11.5% end-to-end, inside the spec's independently
stated "10–12%".

**And the finding that most changes the decision's price:** ✓ VERIFIED **the
chip is already round-parameterised.** `NUM_G = BLAKE3_ROUNDS * 8`
(`blake3_chip.rs:98`), the layout derives from `NUM_G` (`:157`), the dataflow
loops `for r in 0..BLAKE3_ROUNDS` (`:280`), and
`NUM_CONSTRAINTS = 16 × NUM_G + 1` (`:1042`). So "build round-parameterised" is
already done; choosing 7 is a constant, plus regenerating four hard-coded test
expectations (`2,880 → 3,360`, `1,259 → 1,451`, `4,946 → 5,714`, `769 → 897`)
and 7-round vectors — which, unlike the 6-round ones, come straight from the
crate.

## 6. Findings for other phases

1. **✗ OPEN — the `permute` socket is unspecified, and Phase 5's E1 claim
   depends on it.** ✓ VERIFIED: `edsl::merkle_walk` calls `b.compress`
   (`edsl.rs:75`) but `edsl::SpongeVar` calls `b.permute` (`edsl.rs:31,43`). So
   `FriToyV0`'s Fiat–Shamir sponge rests on `permute`, not `compress`.
   Specifying the compress socket makes Merkle authentication real; **it does not
   on its own retire the F3.4 disclosure**, which covers the sponge too.
   `SOCKET.md` §7 sketches a mapping (12 lanes = 48 bytes fits one block) and
   marks it explicitly as a sketch with no vectors and no security argument.
2. **Soundness obligation for Phase 2 (`SOCKET.md` O1).** `merkle_walk`'s sibling
   digests are **arena-hinted, i.e. prover-chosen** — the doc comment at
   `edsl.rs:63-64` says so. A lane is a Goldilocks felt over `[0, p)`, `p ≈ 2^64`.
   If the chip derives message bytes by reduction mod 2^32 instead of a checked
   32-bit decomposition, `v` and `v + 2^32` give the same digest: a
   prover-chosen collision, hence a forged Merkle path. Input lanes must be
   range-checked in the chip, and the host impl must **reject** rather than
   silently reduce. This is plan §3.2's failure mode, on the socket's input side.
3. **⚠ Phase 3 is being written concurrently in this same worktree — see §9.**
   `HasherKind` now has explicit `#[repr(u8)]` discriminants and `as_tag()`
   (`hash.rs:107-127` in the *working tree*), which is plan §4 step 1. I first
   recorded this as "already done"; that was wrong. ✓ VERIFIED by
   `git show HEAD:prover/src/lfm/hash.rs`: `as_tag` **does not exist at HEAD**.
   It is another agent's **uncommitted** work, along with edits to
   `statement.rs`, `registry.rs`, `proof.rs` and `compute_lfm_registry.rs` —
   exactly plan §4's file list. **Since resolved:** that work landed as
   `2d236786 feat(lfm): bind the hasher into the program digest and registry`,
   so `as_tag` and the digest binding are now at `HEAD` and their line numbers
   are stable again. The lesson stands — I recorded a concurrent agent's
   in-flight edit as pre-existing repo state, which `git show HEAD:` caught.
4. **`compress_iv()` is dead weight under BLAKE3** (`SOCKET.md` O3). BLAKE3's IV
   enters through `h` (all 8 words), not through state lanes 8–11, so the arm
   overrides `compress` wholesale — explicitly permitted by `hash.rs:25-26`. The
   override must be wired into `HasherKind::compress`'s explicit delegation
   (`hash.rs:146-151`), whose own doc comment warns about precisely this.

## 7. What needs a build phase

| check | why it is deferred |
|---|---|
| `blake3::hash` restatement of the 7-round KAT of `f` (plan §2.2 step 4) | needs cargo; `ground-truth/` is the place |
| `blake3::hash(a ‖ b ‖ "LFMC")` restatement of the socket identity | needs cargo (already executed against upstream C) |
| the four projected 7-round constants (`3,360 / 1,451 / 5,714 / 897`) | compile-time consts; one `cargo test` confirms |
| chip `OUT` columns vs the socket vectors | no chip arm exists yet (Phase 2) |
| plan §2.2 step 6 — a CI job that re-derives the chain | nothing re-derives it today; `check.py` + `gen_socket_kats.py` + `test_oracle.py` are the three commands |

## 8. Files

Under `/Users/maurofab/workspace/lambda_vm-blake3-impl/` (all **uncommitted**;
`thoughts/blake3/` is untracked, no repo source was modified):

- `thoughts/blake3/blake3-oracle/` — restored; `test_oracle.py` is the anchor run
- `thoughts/blake3/blake3-chip/` — restored (`DESIGN.md`, `z3_blake_verify.py`)
- `thoughts/blake3/ground-truth/` — restored; the crate-linked project for the deferred checks
- `thoughts/blake3/reference-impl/` — **new.** `upstream/` verbatim,
  `blake3_portable_paramrounds.c` + `PARAMETERISATION.diff` (the 2-hunk edit),
  `driver.c`, `build.sh`, `check.py`
- `thoughts/blake3/socket-kats/` — **new.** `SOCKET.md`, `gen_socket_kats.py`, `socket_kats.json`

Under `/Users/maurofab/workspace/lambda_vm/thoughts/shared/lfm-real-hash/`:
`A6R-signoff.md`, `phase1-report.md` (this file).

Build products `b3ref6`, `b3ref7` are regenerable (`./build.sh`) and are
`.gitignore`d, along with `__pycache__/`. `PARAMETERISATION.diff` is regenerable
but should be committed — it is the reviewable artifact.

## 9. Two agents shared this worktree — RESOLVED, kept for the lesson

> **Resolution (2026-08-10, after the fact).** This warning is historical; the
> hazard did not fire. Both bodies of work landed as separate clean commits —
> `2d236786 feat(lfm): bind the hasher into the program digest and registry`
> (the other agent's), then `65025095 test(blake3): restore the
> round-parameterized reference and add a second independent source` (mine, 25
> files, all under `thoughts/blake3/`, no `prover/` source swept in). The
> worktree is clean and `as_tag` is now at `HEAD`. ✓ VERIFIED. The section below
> describes the situation as it stood mid-phase; keep it as the record of why
> commits were made by explicit path.

⚠ **As it stood mid-phase — read before committing:**

`/Users/maurofab/workspace/lambda_vm-blake3-impl` contains **uncommitted changes
to 12 tracked source files that are not mine**: `hash.rs`, `statement.rs`,
`registry.rs`, `proof.rs`, `compute_lfm_registry.rs`, `mod.rs` and six test
modules (+337 / −87). That is plan §4's file list — another agent is writing
Phase 3 here concurrently. ✓ VERIFIED by `git diff` and by confirming `as_tag`
is absent from `HEAD`.

**My own changes touch no tracked file.** Everything I produced is untracked and
lives under `thoughts/blake3/` (plus the two docs in the main checkout). So
`git add thoughts/blake3` is safe; `git add -A` or `git commit -a` would sweep up
another agent's half-finished Phase 3 and commit it under a Phase 1 message.

Two consequences worth acting on:

- **Committing per phase does not work while the worktree is shared.** Either
  give Phase 3 its own worktree, or commit Phase 1 by explicit path.
- **Line citations into `prover/src/lfm/*.rs` are unstable right now.** Mine that
  point below `hash.rs:98` (the trait, its default `compress`, the digest
  constants) are unaffected — ✓ VERIFIED, the concurrent diff is entirely at
  line 98 and after. Citations at or after that point, including
  `HasherKind::compress`'s delegation, are working-tree line numbers and will
  move.

**Three commands reproduce everything:**

```
python3 thoughts/blake3/blake3-oracle/test_oracle.py
thoughts/blake3/reference-impl/build.sh && python3 thoughts/blake3/reference-impl/check.py
python3 thoughts/blake3/socket-kats/gen_socket_kats.py
```
