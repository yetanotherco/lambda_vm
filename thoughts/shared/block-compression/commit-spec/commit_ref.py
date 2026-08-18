"""
THE LFM-NATIVE COMMITMENT LAYER — reference implementation.

DRAFT — PENDING MAURO RATIFICATION. Nothing here is ratified; every open
decision is marked `OPEN:` in the docstrings and listed in COMMIT.md §7.

This is the reference for the three things no ratified doc covers when the LFM
machine's OWN proof moves to the machine's native hashing scheme:

  1. WIDE LEAF   — an arbitrary-width row pair -> an LFML chain folded by LFMC,
                   with the width bound INSIDE the construction.
  2. BYTE ABSORB — a canonical byte-string -> cell encoding for the B1
                   transcript, since `DefaultTranscript` absorbs bytes and B1
                   absorbs 4xu32 cells.
  3. NODE CODEC  — `pack_digest` into [u8;32] plus a STRICT decode that rejects
                   the non-canonical encodings `unpack_digest` would silently
                   reduce.

It builds on the ratified LFMC / LFML / LFMT sockets and adds no new tag: the
wide leaf is LFML rows folded by LFMC parents, exactly the two domains
`leaf-spec/LEAF.md` already ratified.

Runnable with plain python3; no cargo, no third-party packages.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_LFM = os.path.join(_HERE, "..", "..", "lfm-real-hash")
for _p in ("gate-oracle", "leaf-spec", "transcript-spec"):
    sys.path.insert(0, os.path.join(_LFM, _p))

import blake3_oracle as ora          # noqa: E402
import socket_ref as sk              # noqa: E402
import leaf_ref as lr                # noqa: E402
import transcript_ref as tr          # noqa: E402

P = 2**64 - 2**32 + 1                # Goldilocks
MASK32 = 0xFFFFFFFF
LANES = sk.DIGEST_LANES              # 4
FELTS_PER_LFML_ROW = lr.FELTS_PER_LEAF_ROW   # 4

# The production row-pair leaf: leaf i covers bit-reversed rows 2i, 2i+1.
# `stark::commitment::ROWS_PER_LEAF = 2`.
ROWS_PER_LEAF = 2

# --- Element kinds ----------------------------------------------------------
# A committed matrix is uniform in element type: `verify_opening_pair<E>` is
# instantiated at `Field` for the main and precomputed trees and at
# `FieldExtension` for the aux and composition trees (verifier.rs:605-650).
# The felts-per-element count doubles as the kind tag: it is injective over the
# kinds that exist and is the number the serializer actually needs.
KIND_BASE = 1                        # one Goldilocks felt per element
KIND_EXT3 = 3                        # three components per element

# --- Domain markers ---------------------------------------------------------
# These are LANE CONSTANTS inside a header cell, not new socket tags. No new
# tag is allocated: `LFMW`/`LFMB` would be new hash domains needing their own
# analysis, and the construction does not need one — the header cell is an
# ordinary LFMC operand at a program-fixed chain position.
#
# OPEN (D1): whether these should instead be genuine `m[8]` tags. Argued
# against in COMMIT.md §2.4; the counter-argument is that a lane constant is
# only as good as the chain position that carries it.
LEAF_MARK = int.from_bytes(b"LFW0", "little")    # wide-leaf header
BYTES_MARK = int.from_bytes(b"LFB0", "little")   # byte-absorb header


# ===========================================================================
# 1. THE WIDE LEAF
# ===========================================================================

def serialize_elements(elements: list, kind: int) -> list[int]:
    """One row's elements -> a flat felt sequence.

    Base elements contribute one felt; ext3 elements contribute their three
    components in order (c0, c1, c2). This mirrors the existing byte layout,
    where `write_bytes_be` writes an extension element's components 0, 1, 2 in
    that order (`sub_proof.rs:234-236`).
    """
    out: list[int] = []
    for e in elements:
        if kind == KIND_BASE:
            assert isinstance(e, int), "a base element is one felt"
            out.append(e)
        elif kind == KIND_EXT3:
            assert len(e) == 3, "an ext3 element is three components"
            out.extend(int(c) for c in e)
        else:
            raise ValueError(f"unknown element kind {kind}")
    return out


def leaf_header(num_cols: int, kind: int,
                rows_per_leaf: int = ROWS_PER_LEAF) -> list[int]:
    """The header cell that binds the leaf's SHAPE.

    H = [ LEAF_MARK, num_cols, kind, rows_per_leaf ]

    THIS IS THE WHOLE POINT OF THE CONSTRUCTION. The keccak leaf it replaces
    streams `evaluations ‖ evaluations_sym` with no length prefix and no
    separator (`verifier.rs:204-206`), which is why a prover could move columns
    between the main and aux trees and choose them AFTER the LogUp challenges
    the aux root is absorbed behind (`verifier.rs:633-639`, the recorded live
    break, `tests::aux_opening_width_tests`).

    ⚠ The header only closes that break if the VERIFIER BUILDS IT FROM THE AIR,
    never from the opening it received. That is the exact analogue of the
    existing instruction at `verifier.rs:639` — "The width is pinned upstream by
    `trace_opening_widths_well_formed`; do not re-derive it from the proof."
    A verifier that read `num_cols` off `len(evaluations)` would reproduce the
    prover's own choice and bind nothing.
    """
    assert 0 <= num_cols <= MASK32
    assert kind in (KIND_BASE, KIND_EXT3)
    assert 0 <= rows_per_leaf <= MASK32
    return [LEAF_MARK, num_cols, kind, rows_per_leaf]


# ===========================================================================
# ★ THE LEAF RATE — the single most consequential parameter in this spec
# ===========================================================================
#
# It decides whether the recursion tower fits on real hardware: leaf absorption
# is 69.8% of a tower node's bill (Gate D1 census), so the rate scales ~70% of
# the cost linearly.
#
# ✓ VERIFIED from the chip, not assumed. `message_word_ref`
# (`blake3_socket.rs:725-731`) maps m[0..8] -> the eight input lanes' byte
# columns, m[8] -> the mode-selected tag, and **m[9..16] -> `WordRef::Const(0)`**.
# Seven message words are dead. BLAKE3's block is 16 words / 64 bytes; the
# socket uses 9 (block_len 36).
#
# Two ways to spend the headroom:
#
# ⚠⚠ THE BINDING CONSTRAINT IS THE MACHINE'S CELL STRUCTURE, NOT THE BLOCK.
# ✓ VERIFIED `instr.rs:99-110`: `HashMode::num_input_cells` is 2 for
# Compress/Transcript, 1 for Leaf, 3 for Permute, and the doc is explicit that
# "the LFM_HASH bus receives are gated by exactly this". A hash row reads whole
# CELLS from memory, and a cell is FOUR felts (`LfmWord`, `word.rs:15`).
#
# So the felt count per row must be a MULTIPLE OF 4. An earlier draft of this
# spec set the rate to 5 (accumulator cell + 5 felts, 14 lanes) purely from
# block headroom. That is 1.25 cells of felt input and is UNBUILDABLE — the
# machine cannot read it. Enumerating what actually fits:
#
#   accumulator cell (4 lanes) + ONE felt cell (4 felts = 8 lanes) + tag
#                    = 13 of 16 words  ->  4 felts/compression   ✓ 2.0x  ADOPTED
#   accumulator cell + TWO felt cells  = 4 + 16 + 1 = 21 words   ✗ > 16
#   no accumulator, two felt cells     = 16 + 1    = 17 words    ✗ > 16
#
# 4 IS THE MAXIMUM. And it lands on the EXISTING two-cells-in/one-cell-out bus
# contract (`num_input_cells == 2`, same as Compress/Transcript), so the frozen
# LFM_HASH bus arity does not move at all — a better outcome than the rate-5
# draft, which would have needed a bus contract the machine does not have.
LFML_FELTS_PER_ROW = 4           # ★ the spec parameter — a MULTIPLE OF 4 by construction
LFML_ACC_LANES = 4               # the chained accumulator, one digest cell


def _u32le(x: int) -> bytes:
    return int(x).to_bytes(4, "little")


def lfml_chain_row(acc: list[int], felts: list[int],
                   rounds: int = 7) -> list[int]:
    """One widened LFML row: absorb `felts` AND chain `acc`, in one compression.

        msg    = LE32(acc[0..4]) ‖ LE32(lo_i)‖LE32(hi_i) for each felt ‖ "LFML"
        digest = BLAKE3(msg)[0..16] as four LE u32 lanes

    At `LFML_FELTS_PER_ROW = 4` that is 16 + 32 + 4 = **52 bytes** — still ONE
    BLAKE3 block (64), so `block_len` moves 36 -> 52 and nothing else about the
    framing does.

    ✓ THE CRATE-KAT ANCHOR SURVIVES, which is the reason to prefer this over
    carrying the accumulator in the chaining value `h` (the D7 sketch). For any
    input under 64 bytes `blake3::hash` is exactly one compression with `h = IV`,
    `t = 0`, `block_len = len`, `flags = CHUNK_START|CHUNK_END|ROOT` — so a
    60-byte row is a plain library call just as the 36-byte row was. Moving the
    accumulator into `h` would have made the row a chunk *continuation* and split
    the anchor (C9).

    The accumulator lanes need byte decomposition (they are message words) but
    NOT canonicity: they are a previous digest, hence u32 by construction. That
    is why B costs fewer witness columns than A despite the higher rate.
    """
    assert len(acc) == LFML_ACC_LANES
    assert 1 <= len(felts) <= LFML_FELTS_PER_ROW
    msg = b"".join(_u32le(x) for x in acc)
    for v in felts:
        lo, hi = lr.felt_halves(v)          # REJECTS non-canonical, never reduces
        msg += _u32le(lo) + _u32le(hi)
    msg += lr.TAG_LFML_ASCII
    assert len(msg) == 4 * LFML_ACC_LANES + 8 * len(felts) + 4
    full = ora.hash_bytes(msg, 32, rounds=rounds)
    return [int.from_bytes(full[4 * i:4 * i + 4], "little") for i in range(4)]


def wide_leaf(evaluations: list, evaluations_sym: list, kind: int,
              num_cols: int, rounds: int = 7) -> list[int]:
    """The wide-leaf digest: an LFML chain folded into an LFMC chain.

        H     = [LEAF_MARK, num_cols, kind, rows_per_leaf]
        F     = serialize(evaluations) ‖ serialize(evaluations_sym)
        F'    = F ‖ 0^r          r = (-len F) mod 4     (zero-pad to 4 felts)
        d_j   = LFML(F'[4j : 4j+4])
        acc   = H ; for each j:  acc = LFMC(acc, d_j)
        leaf  = acc

    `num_cols` is passed in rather than read off the inputs, and the lengths are
    CHECKED against it — a reference that derived the width from the data would
    encode the very bug this construction exists to remove.

    ZERO-PADDING IS SAFE HERE, and only because the header binds the exact
    element count: two different felt sequences that agree after padding must
    have had different (num_cols, kind, rows_per_leaf), which the header
    separates. Without the header, zero-padding is ambiguous.

    THE FOLD IS A SEQUENTIAL CHAIN, not a balanced tree. A balanced tree over k
    chunk digests costs k-1 compressions against the chain's k, but needs the
    chunk count padded to a power of two — reintroducing exactly the
    shape-ambiguity the header was added to remove. One compression is not worth
    a second padding rule. The chain also binds chunk ORDER for free.

    `rows_per_leaf` is NOT a parameter: the two-slice signature mirrors
    production's `hash_data_from_slices(evaluations, evaluations_sym)`
    (`verifier.rs:583`), which structurally IS the row pair. It is still bound in
    the header, as the constant it is, so a future layout that changed it could
    not collide with this one.
    """
    assert len(evaluations) == num_cols, (
        f"evaluations has {len(evaluations)} columns, AIR pins {num_cols}")
    assert len(evaluations_sym) == num_cols, (
        f"evaluations_sym has {len(evaluations_sym)} columns, AIR pins {num_cols}")

    felts = (serialize_elements(evaluations, kind)
             + serialize_elements(evaluations_sym, kind))
    expected = ROWS_PER_LEAF * num_cols * kind
    assert len(felts) == expected, f"{len(felts)} felts, shape implies {expected}"

    if not felts:
        return leaf_header(num_cols, kind)

    pad = (-len(felts)) % LFML_FELTS_PER_ROW
    felts = felts + [0] * pad

    acc = leaf_header(num_cols, kind)
    for j in range(0, len(felts), LFML_FELTS_PER_ROW):
        acc = lfml_chain_row(acc, felts[j:j + LFML_FELTS_PER_ROW], rounds)
    return acc


def wide_leaf_compressions(num_cols: int, kind: int,
                           rows_per_leaf: int = ROWS_PER_LEAF) -> int:
    """Compressions one wide leaf costs: ONE per `LFML_FELTS_PER_ROW` felts.

    The fold is gone — each row absorbs and chains in the same compression — so
    this is `ceil(felts / rate)`, not `2 * ceil(felts / 4)`.
    """
    felts = rows_per_leaf * num_cols * kind
    return -(-felts // LFML_FELTS_PER_ROW)             # ceil


def wide_leaf_v0_folded(evaluations: list, evaluations_sym: list, kind: int,
                        num_cols: int, rounds: int = 7) -> list[int]:
    """The SUPERSEDED 4-felt + LFMC-fold construction, kept for the rate KAT.

    2 felts/compression. Retained only so C12 can measure the improvement
    against something executable rather than against a remembered number.
    """
    felts = (serialize_elements(evaluations, kind)
             + serialize_elements(evaluations_sym, kind))
    pad = (-len(felts)) % FELTS_PER_LFML_ROW
    felts = felts + [0] * pad
    acc = leaf_header(num_cols, kind)
    fr = sk.Framing(rounds=rounds)
    for j in range(0, len(felts), FELTS_PER_LFML_ROW):
        d = lr.leaf_compress(felts[j:j + FELTS_PER_LFML_ROW], rounds)
        acc = sk.socket_digest_wordlevel(acc, d, fr)
    return acc


def wide_leaf_v0_compressions(num_cols: int, kind: int) -> int:
    felts = ROWS_PER_LEAF * num_cols * kind
    return 2 * (-(-felts // FELTS_PER_LFML_ROW))


# ===========================================================================
# 2. THE BYTE -> CELL ABSORB ENCODING
# ===========================================================================

def bytes_to_cells(data: bytes) -> list[list[int]]:
    """A byte string -> a length-prefixed cell sequence, for B1 absorb.

        header = [BYTES_MARK, len & 0xFFFFFFFF, len >> 32, 0]
        body   = data zero-padded to a multiple of 16, each 16 bytes read as
                 four LITTLE-ENDIAN u32 lanes

    O1 COMPLIANCE IS AUTOMATIC AND THAT IS THE POINT: every lane is exactly four
    bytes, so every lane is `< 2^32` by construction. No canonicity gate, no
    rejection, no `MODE_L` row — a byte block is already digest-shaped. This is
    why bytes go through THIS path and field elements go through `absorb_felts`
    (`LFML`), which is where the canonicity gate lives.

    The length prefix is what makes the encoding injective under zero-padding:
    without it `b"\\x01"` and `b"\\x01\\x00"` would absorb identically.

    Little-endian to match `word_of`'s convention (`blake3_socket.rs:441-443`:
    "one felt = one u32 = four little-endian bytes").
    """
    n = len(data)
    assert n < 2**64, "byte strings are length-prefixed with 64 bits"
    header = [BYTES_MARK, n & MASK32, (n >> 32) & MASK32, 0]

    pad = (-n) % 16
    padded = data + b"\x00" * pad
    body = []
    for i in range(0, len(padded), 16):
        block = padded[i:i + 16]
        body.append([int.from_bytes(block[4 * k:4 * k + 4], "little")
                     for k in range(4)])
    return [header] + body


def absorb_bytes(t: "tr.Transcript", data: bytes) -> "tr.Transcript":
    """Absorb a byte string into a B1 transcript under the encoding above.

    Costs `1 + ceil(len/16)` compressions — the header cell plus one per block.
    """
    for cell in bytes_to_cells(data):
        t.absorb(cell)
    return t


def absorb_bytes_compressions(nbytes: int) -> int:
    return 1 + (-(-nbytes // 16))


# ===========================================================================
# 3. NODE EMBEDDING AND STRICT DECODE
# ===========================================================================

def pack_digest(word: list[int]) -> bytes:
    """LfmWord -> the 32-byte `Commitment`, mirroring `word.rs:44-50`.

    Four canonical u64 lanes, little-endian, in lane order. Under BLAKE3 every
    lane is `< 2^32` (`word_of`, `blake3_socket.rs:443`), so bytes 4..8 of each
    8-byte chunk are ZERO — the padding that lets a 128-bit digest ride inside
    the existing 32-byte proof format without moving the rkyv wire layout.
    """
    assert len(word) == LANES
    out = b""
    for lane in word:
        assert 0 <= lane < P, "a digest lane must be a canonical felt"
        out += int(lane).to_bytes(8, "little")
    return out


def strict_unpack_digest(b: bytes) -> list[int]:
    """[u8;32] -> LfmWord, REJECTING everything `unpack_digest` would reduce.

    ⚠ THIS IS THE MALLEABILITY FIX (COMMIT.md S2). `word.rs:52-61`'s
    `unpack_digest` reads each 8-byte chunk as a u64 and reduces mod p, so MANY
    distinct 32-byte strings decode to ONE node: any lane may be offset by a
    multiple of p, and — more cheaply — any of the sixteen zero padding bytes may
    be set to anything below the reduction boundary. Node-level malleability in a
    Merkle path is a proof-format forgery surface, not a cosmetic issue.

    The strict rule mirrors `lanes_of` (`blake3_socket.rs:431-438`), which
    already rejects rather than reduces on the host: EVERY lane must be `< 2^32`,
    i.e. the high four bytes of every chunk must be zero. `< 2^32` implies
    `< p`, so one test covers both.
    """
    if len(b) != 32:
        raise ValueError(f"a commitment is 32 bytes, got {len(b)}")
    word = []
    for i in range(LANES):
        chunk = b[8 * i:8 * i + 8]
        if chunk[4:] != b"\x00\x00\x00\x00":
            raise ValueError(
                f"lane {i} has non-zero high bytes {chunk[4:].hex()}: a BLAKE3 "
                f"digest lane is a u32 (reject, never reduce)")
        word.append(int.from_bytes(chunk[:4], "little"))
    return word


# ===========================================================================
# 4. THE MERKLE TREE — arity and padding
# ===========================================================================

def merkle_root(leaves: list[list[int]], rounds: int = 7) -> list[int]:
    """Binary LFMC tree over wide-leaf digests. ASSERTS a power-of-two count.

    ARITY 2, NO PADDING, BY DECISION. The leaf count is always `lde_size / 2`
    and `lde_size` is always a power of two (the prover debug-asserts exactly
    this at `commitment.rs:67-70`), so the assertion costs nothing and is
    ALWAYS satisfiable on the honest path. Padding to a power of two would add a
    duplicate-leaf second-preimage surface for a case that does not arise, which
    is the wrong trade: an unreachable branch that weakens the tree.

    Mirrors `fixture.rs:163-175`'s `HostTree::build`, which asserts the same.
    """
    assert leaves, "a tree needs at least one leaf"
    n = len(leaves)
    assert n & (n - 1) == 0, (
        f"{n} leaves is not a power of two; the wide-leaf tree asserts rather "
        f"than pads (COMMIT.md S6)")
    fr = sk.Framing(rounds=rounds)
    level = [list(x) for x in leaves]
    while len(level) > 1:
        level = [sk.socket_digest_wordlevel(level[i], level[i + 1], fr)
                 for i in range(0, len(level), 2)]
    return level[0]


# ===========================================================================
# 5. CHALLENGE SAMPLING — the 96-bit question, both options
# ===========================================================================

def squeeze_ext_1(t: "tr.Transcript") -> list[int]:
    """The ratified B1 shape: lanes 0-2 of ONE squeezed cell. 1 compression.

    Each lane is a u32, so each coordinate is `< 2^32`: the extension challenge
    carries 96 bits, not the ~192 `DefaultTranscript` delivers. TRANSCRIPT.md
    §4.1 bounds the STATE (128 bits, ~64-bit collision) but does not analyse
    per-challenge entropy at production query counts.
    """
    return t.squeeze()[0:3]


def squeeze_ext_2_DECIDED(t: "tr.Transcript") -> list[int]:
    """Alias marking the ratified choice. See `squeeze_ext_2`."""
    return squeeze_ext_2(t)


def squeeze_ext_2(t: "tr.Transcript") -> list[int]:
    """The alternative: TWO squeezed cells -> three ~64-bit coordinates.

        c0, c1 = squeeze(), squeeze()
        lanes  = c0 ‖ c1                       (8 lanes)
        coef_i = (lanes[2i] + 2^32 * lanes[2i+1]) mod p      for i in 0..3

    COST: 2 compressions per extension challenge instead of 1 — a flat +1.
    Query-index sampling is UNAFFECTED: `squeeze_bits` reads lane 0 only and
    needs no extra entropy, so the query loop (the dominant squeeze run) does
    not pay.

    NO REJECTION LOOP, deliberately. A uniform 64-bit value reduced mod p is
    biased by about 2^-32 towards the low `2^32 - 1` residues, which is
    negligible for a Fiat-Shamir challenge; a rejection loop would be exact but
    is UNIMPLEMENTABLE in the fully-unrolled eDSL ("nothing loop-shaped reaches
    the machine", TRANSCRIPT.md §1.1 citing `edsl.rs:1-4`). Bias is the right
    trade here and the reason is structural, not lazy.

    OPEN (D4): whether 96 bits is in fact insufficient. This function exists so
    the cost of the answer is known before the question is decided.
    """
    lanes = t.squeeze() + t.squeeze()
    return [(lanes[2 * i] + (lanes[2 * i + 1] << 32)) % P for i in range(3)]


# ===========================================================================
# 6. GRINDING UNDER B1 — the proof-of-work construction (D3, D6)
# ===========================================================================
#
# DECIDED (Mauro, 2026-08-12): "Grinding should help you, we need 128 security
# for sure." Grinding STAYS, so B1 needs a PoW it can express. The keccak PoW it
# replaces is TWO keccak256 hashes over byte buffers (`grinding.rs:67-89`):
#
#     inner = Keccak256( PREFIX(8) ‖ seed(32) ‖ factor(1) )          41 bytes
#     valid = u64_be( Keccak256( inner(32) ‖ nonce_be(8) )[..8] ) < 2^(64-factor)
#
# Neither layer is expressible as a 2-to-1 compress, and both run through the
# hosted keccak family — the chips D0 exists to stop paying.

GRIND_MARK = int.from_bytes(b"GRD0", "little")


def grind_operand(nonce: int, factor: int) -> list[int]:
    """The PoW operand cell: `[nonce_lo, nonce_hi, GRIND_MARK, factor]`.

    ONE cell, so the whole PoW is ONE `compress_T` — that is the design target.
    Both the nonce AND the difficulty live in the operand, so a nonce found at
    one difficulty is worthless at another (KAT C11.d): without `factor` in the
    preimage a prover could mine once at factor 1 and present the result at
    factor 20.
    """
    if not 0 <= nonce < 2**64:
        raise ValueError("the nonce is a u64")
    if not 1 <= factor <= 64:
        raise ValueError("grinding_factor is in 1..=64 (`grinding.rs:22`)")
    return [nonce & MASK32, (nonce >> 32) & MASK32, GRIND_MARK, factor]


def pow_digest(state: list[int], nonce: int, factor: int,
               rounds: int = 7) -> list[int]:
    """`W = compress_T(state, [nonce_lo, nonce_hi, GRIND_MARK, factor])`.

    ⚠ DOMAIN SEPARATION — read this before changing the tag.

    This reuses the TRANSCRIPT tag `LFMT`; it does NOT allocate a fourth domain.
    The separation argument is exactly the one B1 already relies on for
    absorb-vs-squeeze, quoting TRANSCRIPT.md §1.1: the operation sequence is a
    compile-time constant of the program, so "a prover cannot perform a squeeze
    where the program says absorb", and equally cannot present a PoW evaluation
    where the program says absorb. `GRIND_MARK` is defence in depth on the same
    footing as `SQUEEZE_MARK` — which that section is explicit is NOT the
    load-bearing argument.

    Sharing the tag costs nothing cryptographically here: to satisfy the
    difficulty a prover must still search operands at a state it does not
    control, and no transcript step it computes elsewhere helps. It saves a tag,
    a fourth preprocessed selector (`MODE_G`), `PREP_WIDTH` 13 -> 14 and a
    registry re-bless.

    OPEN (D6a): whether to spend those anyway for an unconditional separation.
    Costed in COMMIT.md §4.1.
    """
    return tr.compress_t(state, grind_operand(nonce, factor), rounds)


def pow_is_valid(state: list[int], nonce: int, factor: int,
                 rounds: int = 7) -> bool:
    """The difficulty predicate: the low `factor` bits of `W[0] ‖ W[1]` are zero.

    Reading the two lanes as one 64-bit value `W[0] + 2^32·W[1]` covers the whole
    documented range `1..=64` under ONE rule, and for the realistic `factor <= 32`
    it touches lane 0 only. The alternative — a rule on lane 0 with a second rule
    bolted on above 32 — is two cases where one will do.

    GUEST COST: one `compress_T` plus one `LFM_BITDEC` row to expose the low
    bits. Against the keccak PoW's two sponge invocations through the hosted
    keccak family.
    """
    w = pow_digest(state, nonce, factor, rounds)
    combined = w[0] + (w[1] << 32)
    return combined % (1 << factor) == 0


def find_nonce(state: list[int], factor: int, rounds: int = 7,
               limit: int = 1 << 24) -> int | None:
    """Mine a nonce. Expected 2^factor trials — the honest prover's cost."""
    for nonce in range(limit):
        if pow_is_valid(state, nonce, factor, rounds):
            return nonce
    return None


def pow_verify_compressions() -> int:
    """PoW verification is ONE compression, independent of the difficulty.

    ⚠ Read the honest framing in COMMIT.md §4.1: this saving is O(1) per proof
    and therefore small in absolute terms. The reason grinding stays is NOT this
    compression — it is the 41 queries (222,794 tower permutations) that
    grinding buys back, §7.1.
    """
    return 1
