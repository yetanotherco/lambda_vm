# The LFM 2-to-1 BLAKE3 compress socket

**Status:** specification + reference vectors. **No chip code exists for this.**
**Date:** 2026-08-10. **Phase:** 1 (reference layer), per `thoughts/shared/lfm-real-hash/PLAN.md` §2.5 and §5.
**Scope:** the `compress` socket only. The `permute` socket is *not* specified here — see §7.

Claims about the tree are marked ✓ VERIFIED (read the code, cited `file:line`),
? INFERRED (derived, arithmetic shown) or ✗ OPEN.

---

## 1. What this pins, and why `CANONICAL_VECTORS` is not enough

`prover/src/lfm/blake3.rs:151` pins the compression function `f(h, m, t,
block_len, flags)` with ten vectors. ✓ VERIFIED. That is the *primitive*.

It says nothing about how a two-input hash **calls** `f`. Between "we have a
correct `f`" and "we have a correct 2-to-1 compress" sit six independent
choices, every one of which is a way to be wrong while every existing test stays
green:

1. where the two input digest cells land in the 16-word message `m`;
2. what the chaining value `h` is;
3. what the counter `t` is;
4. what `block_len` is;
5. what the `flags` byte is;
6. which 4 of the 16 output words become the digest.

This document fixes all six, and `socket_kats.json` pins them with a vector
table plus one negative control per choice.

## 2. The specification

### 2.1 Byte-level form (normative)

Let `a`, `b` be the two input digest cells, each four lanes, **each lane an
unsigned 32-bit value** (see obligation O1). Write `LE32(x)` for the four-byte
little-endian encoding of `x`.

```
msg  = LE32(a0) ‖ LE32(a1) ‖ LE32(a2) ‖ LE32(a3)      (16 bytes)
     ‖ LE32(b0) ‖ LE32(b1) ‖ LE32(b2) ‖ LE32(b3)      (16 bytes)
     ‖ "LFMC"                                          ( 4 bytes, domain tag)
                                                       ------------------
                                                        36 bytes

digest_bytes = BLAKE3(msg)[0 .. 16]                    (truncate 256 → 128 bits)

c_i = LE32⁻¹( digest_bytes[4i .. 4i+4] )    for i in 0..4
```

`BLAKE3(·)` is the standard default hashing mode — the plain one-argument hash,
no key, no context.

**This is the whole specification.** It is deliberately written as a call to a
library rather than as a compression-function invocation, because that is what
makes it externally checkable: at 7 rounds, `compress(a, b)` is *literally*
`blake3::hash(a ‖ b ‖ "LFMC")` truncated to 16 bytes. There is no oracle in the
chain, and no assumption. §5 records that this equality is already executed.

### 2.2 Word-level form (what the chip proves)

The 36-byte message is one BLAKE3 block, so the byte-level form is exactly one
compression. The chip proves this:

| input to `f` | value |
|---|---|
| `h` (chaining value) | `BLAKE3_IV[0..8]` — all eight words |
| `m[0..4]` | `a[0..4]` |
| `m[4..8]` | `b[0..4]` |
| `m[8]` | **mode-selected on the built chip** — `MODE_C·TAG_LFMC + MODE_T·TAG_LFMT`; `0x434D464C` on a Merkle row (`MODE_C = 1`). See the note below. |
| `m[9..16]` | `0` |
| `t` (counter) | `0` |
| `block_len` | `36` |
| `flags` | `0x0B` = `CHUNK_START | CHUNK_END | ROOT` |

Output: `c_i = f(...)[i]` for `i in 0..4` — the **low four** words of the
16-word output, i.e. the low half of the truncated chaining value.

Everything in that table except `a` and `b` and `m[8]` is a compile-time
constant, and the socket costs the chip no extra columns beyond the compression
it already proves.

> **⚠ UPDATED FOR B1 (2026-08-11) — the conclusion holds, the REASON changed.**
> `m[8]` was a compile-time constant when this document was written, and
> "constant" was why it was free. The built chip selects it from the row's
> domain: `MODE_C·TAG_LFMC + MODE_T·TAG_LFMT`, a linear form over two
> **preprocessed** columns (`WordRef::ModeSelected`, evaluated `Σ col·tag`),
> because the Fiat–Shamir transcript now runs on this same socket under
> `"LFMT"` (option B1 — see §2.4 and §7). It remains free (only ever an `add3`
> operand, read as a whole word, never byte-decomposed) and remains
> prover-unchosen — but now because the selectors are preprocessed, not because
> the value is constant.
>
> **If you transcribe this table into a model, transcribe the linear form.** A
> model carrying a constant where the chip has a linear form still reports PASS
> while checking something the chip does not do.

`gen_socket_kats.py` computes §2.1 and §2.2 by separate routes and asserts they
agree, for every vector, at both round counts. That equality is the framing
check; if the chip is ever re-expressed, it is the property to re-run.

### 2.3 Why the domain tag sits in the message

Plan §5 recommends "option D": a 128-bit digest **plus domain separation**. The
obvious place for a domain tag is the `flags` byte, and that is what BLAKE3
itself does for `PARENT` / `CHUNK_START` / `ROOT`.

**We put it in the message instead, and that choice is load-bearing.** Any tag
in `flags` (or in `t`, or in `h`) makes the socket a *nonstandard* invocation of
`f` that no library computes — so the KATs could only ever come from our own
oracle, at 6 **and** at 7 rounds. Putting the tag in the message keeps the
socket a standard BLAKE3 hash of a domain-separated byte string, which is the
entire reason §2.1 can be a library call. The domain separation is just as real:
distinct tags give distinct 36-byte messages.

Cost of the choice: the message is 36 bytes rather than 32, which is still one
block. Zero extra compressions, zero extra columns. ? INFERRED — `block_len` and
`m[8]` is not a column either — post-B1 it is a linear form over preprocessed
mode columns, which is still zero columns and zero sends (✓ CONFIRMED against
the built arm).

### 2.4 Tag allocation

| tag | bytes | u32 (LE) | use |
|---|---|---|---|
| `"LFMC"` | `4C 46 4D 43` | `0x434D464C` | **this socket** — 2-to-1 compress / Merkle parent |
| `"LFMT"` | `4C 46 4D 54` | `0x544D464C` | **transcript step** — the compress-chain Fiat–Shamir transcript (`thoughts/shared/lfm-real-hash/transcript-spec/TRANSCRIPT.md`) |
| `"LFMP"` | `4C 46 4D 50` | `0x504D464C` | ~~`permute` socket (§7)~~ — **RETIRED UNUSED**, see below |
| `"LFML"` | `4C 46 4D 4C` | `0x4C4D464C` | reserved — leaf domain; **O5 ratified**: any future leaf-hashing path MUST use it |

⚠ **§7's `permute`-socket sketch is superseded and will never be built.** The
user ratified option **B1** on 2026-08-11
(`thoughts/shared/lfm-real-hash/permute-socket-options.md`): the Fiat–Shamir
sponge becomes a **compress-based chain** over *this* socket under the new
`"LFMT"` tag, for all hashers, and `MODE_P` stays pinned to 0 permanently. Read
§7 as a record of a rejected direction, not as a plan.

`"LFMP"` is **retired rather than deleted**, and the distinction is
load-bearing: the value is now permanently unused, but removing the row would
let a future allocation reuse `0x504D464C` and silently create a domain nobody
analysed.

A tag is never reused for a second purpose, for the same reason
`HasherKind::as_tag` never reuses a discriminant. ⚠ `as_tag` is **not yet
committed** — it is another agent's in-flight Phase 3 work in this worktree and
is absent from `HEAD` (✓ VERIFIED via `git show HEAD:prover/src/lfm/hash.rs`).
Cited by symbol, not by line, because its line numbers will move.

## 3. Security consequence, stated plainly

The digest is **128 bits**, so this socket offers **64-bit collision
resistance** by the birthday bound, not 128-bit. That is the honest consequence
of `HASH_DIGEST_FELTS = 4` (`hash.rs:21`) and of `word.rs:1-9`'s declared
"128-bit target", both ✓ VERIFIED — it is not introduced by BLAKE3 or by the
truncation window.

**This is the question the plan (§5) puts to the user and it is not settled
here.** If the target is 128-bit *collision* resistance, the digest must be two
cells (256 bits) and the frozen 1-cell `LFM_HASH` output contract has to be
reopened. If the target is a 128-bit *security level* in the ordinary
preimage sense, this socket meets it. Nothing below depends on which answer
comes back; only the digest width does.

Preimage resistance of the truncated digest is 128 bits. ? INFERRED — standard
for a truncated random oracle; it is not an assumption specific to this design.

## 4. Obligations for the chip arm (Phase 2)

**O1 — input lanes MUST be range-checked to 32 bits. This is a soundness
obligation, not hygiene.** ✓ VERIFIED that it bites: `edsl::merkle_walk`
(`edsl.rs:65-80`) feeds `compress` sibling cells that are **arena-hinted**, i.e.
prover-chosen — the doc comment says so outright: *"Sibling digests come as
(arena-hinted) cells; every hinted value ends up inside a `compress`, which is
what authenticates it."* A lane is a Goldilocks felt, so it ranges over
`[0, p)` with `p ≈ 2^64`. If the chip derives the four message bytes of a lane
by reduction mod 2^32 rather than by a checked decomposition, then lane values
`v` and `v + 2^32` produce the **same** message and hence the same digest — a
free collision, chosen by the prover, and therefore a forged Merkle path. The
host-side `LfmHasher` impl must likewise **reject** an out-of-range lane rather
than silently reduce, or the host and the chip disagree about what was proved
(plan §3.2, same failure mode on the input side).

**O2 — the socket must be closed on its own output.** `c_i` is a `u32` by
construction, so a digest produced by this socket always satisfies O1. Only
*leaf* digests and prover-hinted siblings can violate it, which is exactly where
O1's check must sit.

**O3 — `compress_iv()` does not participate.** The trait's default `compress`
injects `compress_iv()` into state lanes 8–11 (`hash.rs:35-43`, ✓ VERIFIED).
The BLAKE3 arm **overrides** `compress` entirely — the IV enters through `h`,
all eight words, not through the state. Overriding is explicitly sanctioned:
*"a real hash may override it, but the bus contract (2 cells in, 1 cell out) is
frozen"* (`hash.rs:25-26`). Two consequences: `compress_iv()` should return
`BLAKE3_IV[0..4]` as felts so it is meaningful if read, with a doc comment
saying it is not part of the compress framing; and the override must be wired
into `HasherKind::compress`'s explicit delegation, whose own doc comment already
warns that a candidate overriding `compress` must be honoured through that
dispatch. (Cited by symbol: that part of `hash.rs` is being edited concurrently
by the Phase 3 agent, so its line numbers are in motion. The trait definition and
its default `compress` at `hash.rs:19-44` are *not* in the edited region and are
✓ VERIFIED stable against `HEAD`.)

**O4 — the byte order is the `keccak_host` convention, and it is already the
machine's.** One felt carries one `u32` as four little-endian bytes
(`keccak_host.rs:17-32`, ✓ VERIFIED: `FE::from(u64::from(u32::from_le_bytes(half)))`).
This socket reuses it unchanged. Note this is *not* `word::pack_digest`
(`word.rs:44-50`), which serialises each lane as eight bytes; the two are
different serialisations of a cell and must not be confused.

## 5. The vectors

`socket_kats.json`, generated by `gen_socket_kats.py`.

- **10 vectors × 2 round counts.** Five structural inputs (zeros, unit `a`, unit
  `b`, all-ones, a nibble ramp) and five from an explicit formula. All inputs are
  written out in the JSON, so nothing depends on a random-number generator.
- **9 negative controls per vector**, one per framing degree of freedom:
  `swap_a_b`, `tag_changed`, `tag_omitted`, `truncate_high_half`, `flags_parent`,
  `block_len_64`, `counter_one`, `lanes_big_endian`, `other_round_count`. The
  generator asserts each applicable control **changes** the digest, and
  separately asserts every control is discriminated by at least one vector.

  Two controls are declared inapplicable on degenerate inputs rather than
  skipped: `swap_a_b` when `a == b`, and `lanes_big_endian` when every lane is a
  byte-palindrome (`0x00000000`, `0xFFFFFFFF`, `0x11111111`, …). That is a real
  property of those inputs, not a workaround — three of the five structural
  vectors cannot detect a byte-order error, which is precisely why the formula
  vectors are in the table.
- **Three independent computations agree** on every vector: the in-repo Python
  oracle at word level, upstream BLAKE3's C at word level, and upstream BLAKE3's
  **whole tree hasher** over the 36-byte string at byte level.

Worked example (`nibble_ramp`, rounds = 7):

```
a   = 00000000 11111111 22222222 33333333
b   = 44444444 55555555 66666666 77777777
msg = 00000000111111112222222233333333444444445555555566666666777777774c464d43
BLAKE3(msg)      = c03eaa1a295bdd663056a4e9ff74d261051f49096ec2345cde112bda36168bf4
digest (16 bytes)= c03eaa1a295bdd663056a4e9ff74d261
c   = 1aaa3ec0 66dd5b29 e9a45630 61d274ff      (the same 16 bytes as u32 lanes)
```

At rounds = 6 the same inputs give `c = 2ef9ed44 4b4ab3f5 6be64dc6 dabef7b1`.
No library computes that value and no published vector contains it — which is
the whole of the A6R argument, in one line.

## 6. What is executed and what is deferred

| claim | status |
|---|---|
| word-level and byte-level forms agree, both round counts, all 10 vectors | ✓ EXECUTED |
| Python oracle and upstream C agree on every socket vector | ✓ EXECUTED |
| at rounds = 7 the socket equals upstream BLAKE3's whole-hash output, truncated | ✓ EXECUTED (against upstream **C**, which passes the official vectors) |
| all 9 controls discriminate | ✓ EXECUTED |
| the same equality against the Rust **`blake3` crate** | ✗ DEFERRED to a build phase — needs cargo |
| the chip's `OUT` columns match these vectors | ✗ DEFERRED — no chip arm exists yet |

The deferred crate check is a formality rather than a risk: the C that was
checked *is* upstream BLAKE3, and it reproduced the official test vectors in all
three modes. It should still be written, as a one-line `blake3::hash` assertion,
because it is the version of the check that survives this directory being
deleted.

## 7. ~~✗ OPEN: the `permute` socket is not specified here~~
## ⛔ SUPERSEDED — NO PERMUTE SOCKET WILL EVER BE BUILT (option B1, 2026-08-11)

> The user ratified **option B1**: the Fiat–Shamir sponge becomes a
> **compress-based chain** over the socket this document specifies, under the
> new `"LFMT"` tag; `MODE_P` stays pinned to 0 permanently. Spec, reference,
> KATs and gate extension:
> `thoughts/shared/lfm-real-hash/transcript-spec/TRANSCRIPT.md`.
> **Everything below is a record of the rejected direction.** It is kept because
> the options paper's analysis cites it, not because anyone should build it.

The brief asked for the 2-to-1 compress socket and that is what this document
covers. Flagging the gap explicitly, because it changes what Phase 5's E1
milestone can claim:

✓ VERIFIED — `edsl::merkle_walk` compresses (`edsl.rs:75`, `b.compress(...)`),
but `edsl::SpongeVar` **permutes** (`edsl.rs:31` and `edsl.rs:43`,
`b.permute(...)`). `FriToyV0`'s Fiat–Shamir sponge is therefore built on
`permute`, not on `compress`. Specifying this socket makes `merkle_walk`'s
authentication real; it does **not** on its own make the sponge real, so the
F3.4 disclosure is only half retired by it.

The `permute` socket needs its own mapping decision and its own KATs: 12 felts
in, 12 felts out. The natural shape under the u32-lane restriction is one
compression — 12 lanes = 48 bytes fits one 64-byte block, per plan §3.2 option
(i) — taking `h = IV`, `m[0..12] = state`, `m[12] = "LFMP"`, `m[13..16] = 0`,
`t = 0`, `block_len = 52`, `flags = 0x0B`, and `out[0..12]` as the new state.
That is a **sketch, not a decision**: it is unreviewed, has no vectors, and the
security argument for a 12-word permutation built from a truncated compression
output is not the same argument as §3's. It should get the same treatment this
document gave `compress` before any code is written against it.
