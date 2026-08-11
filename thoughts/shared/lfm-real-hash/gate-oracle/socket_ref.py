"""
LAYER 2: the OPTION-A `LFM_HASH` 2-to-1 compress socket, as a reference function.

This is the thing the chip must compute.  `blake3_oracle.compress` is the
primitive; this file is the *framing* -- the six independent choices that sit
between "we have a correct f" and "we have a correct 2-to-1 compress", every one
of which is a way to be wrong while every primitive test stays green:

  1. where a and b land in the 16 message words,
  2. what the chaining value h is,
  3. what the counter t is,
  4. what block_len is,
  5. what the flags byte is,
  6. which 4 of the 16 output words become the digest (the truncation window),

plus, for the LFM socket specifically and *not* present in a syscall-shaped chip:

  7. how a Goldilocks felt becomes four message bytes (the lane boundary).

THE SPECIFICATION
-----------------
Byte-level (normative, and deliberately expressible as a library call):

    msg = LE32(a0)‖LE32(a1)‖LE32(a2)‖LE32(a3)      (16 bytes)
        ‖ LE32(b0)‖LE32(b1)‖LE32(b2)‖LE32(b3)      (16 bytes)
        ‖ "LFMC"                                    ( 4 bytes)   = 36 bytes

    digest_bytes = BLAKE3(msg)[0..16]
    c_i          = LE32^-1(digest_bytes[4i .. 4i+4])   for i in 0..4

Word-level (what the chip proves) -- 36 bytes is one block, so this is exactly
one compression:

    h         = IV[0..8]                 (all eight words; the unkeyed default)
    m[0..4]   = a,  m[4..8] = b
    m[8]      = 0x434D464C               ("LFMC" read as one little-endian u32)
    m[9..16]  = 0
    t         = 0
    block_len = 36
    flags     = CHUNK_START|CHUNK_END|ROOT = 0x0B
    digest    = out[0..4]                (the LOW four of the 16 output words)

The two routes are computed here by separate code paths and asserted equal.
At rounds = 7 the byte-level route is literally `blake3::hash(a‖b‖"LFMC")[..16]`,
so the socket has an external anchor and needs no oracle in the chain.  At
rounds = 6 no library computes it: that is assumption A6R, and it is the reason
the tag lives in the MESSAGE and not in `flags`/`t`/`h` -- any tag outside the
message would make even the 7-round socket a nonstandard invocation of f that no
library computes, throwing away the anchor for nothing.

WHY THE DOMAIN TAG IS IN THE MESSAGE (independently re-derived, agrees with
`thoughts/blake3/socket-kats/SOCKET.md`).  The message is fixed-length (36 bytes)
with the tag at a fixed offset, and `block_len` is itself an input to f, so the
encoding is unambiguous: distinct tags give distinct messages, and no length
ambiguity exists.  Cost is 4 extra message bytes inside the same single block --
zero extra compressions, and (see COLUMN_ROLE_MAP) zero extra columns, because
m[8] is a constant.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import blake3_oracle as ora
from blake3_oracle import IV, MASK32

# Domain tags.  A tag is never reused for a second purpose.
TAG_LFMC_ASCII = b"LFMC"          # this socket: 2-to-1 compress / Merkle parent
TAG_LFMP_ASCII = b"LFMP"          # reserved: the `permute` socket (NOT specified)
TAG_LFML_ASCII = b"LFML"          # reserved: leaf domain (see obligation O5)

TAG_LFMC = int.from_bytes(TAG_LFMC_ASCII, "little")   # 0x434D464C
TAG_LFMP = int.from_bytes(TAG_LFMP_ASCII, "little")
TAG_LFML = int.from_bytes(TAG_LFML_ASCII, "little")

FLAGS_LFMC = ora.CHUNK_START | ora.CHUNK_END | ora.ROOT   # 0x0B
BLOCK_LEN_LFMC = 36
DIGEST_LANES = 4


@dataclass(frozen=True)
class Framing:
    """Every framing degree of freedom, in one object.

    The honest socket is `HONEST`.  A negative control is a `replace(HONEST, ...)`
    applied to the CHIP side while the reference keeps `HONEST` -- which is what
    makes the control suite systematic instead of ad hoc, and what lets the gate
    and the reference share one definition of "the framing".
    """
    rounds: int = 7
    cv: tuple[int, ...] = tuple(IV)          # h[0..8]
    tag_word: int = TAG_LFMC
    counter: int = 0
    block_len: int = BLOCK_LEN_LFMC
    flags: int = FLAGS_LFMC
    a_slot: int = 0                          # m[a_slot .. a_slot+4] = a
    b_slot: int = 4                          # m[b_slot .. b_slot+4] = b
    tag_slot: int = 8                        # m[tag_slot] = tag_word
    out_window: int = 0                      # digest = out[out_window .. +4]
    lane_le: bool = True                     # lane -> 4 bytes, little-endian
    msg_permutation: tuple[int, ...] = tuple(ora.MSG_PERMUTATION)


HONEST_7 = Framing(rounds=7)
HONEST_6 = Framing(rounds=6)


def honest(rounds: int) -> Framing:
    return Framing(rounds=rounds)


# ---------------------------------------------------------------------------
# Lane boundary (choice 7).  A digest cell is 4 lanes; a lane is a felt that
# MUST carry a u32.  `keccak_host`'s convention: one felt = one u32 = four
# little-endian bytes.  This is NOT `word::pack_digest` (8 bytes per lane).
# ---------------------------------------------------------------------------

def lane_to_bytes(lane: int, le: bool = True) -> bytes:
    if not 0 <= lane <= MASK32:
        raise ValueError(f"lane {lane:#x} is not a u32 -- obligation O1 violated")
    return int(lane).to_bytes(4, "little" if le else "big")


def bytes_to_lane(b: bytes, le: bool = True) -> int:
    return int.from_bytes(b, "little" if le else "big")


def message_bytes(a: list[int], b: list[int], fr: Framing = HONEST_7) -> bytes:
    """The normative 36-byte message."""
    assert len(a) == len(b) == DIGEST_LANES
    out = b"".join(lane_to_bytes(x, fr.lane_le) for x in a)
    out += b"".join(lane_to_bytes(x, fr.lane_le) for x in b)
    out += int(fr.tag_word & MASK32).to_bytes(4, "little")
    return out


# ---------------------------------------------------------------------------
# Route 1 -- byte level.  At rounds = 7 this is a plain BLAKE3 hash.
# ---------------------------------------------------------------------------

def socket_digest_bytelevel(a: list[int], b: list[int],
                            fr: Framing = HONEST_7) -> list[int]:
    msg = message_bytes(a, b, fr)
    full = ora.hash_bytes(msg, 32, rounds=fr.rounds)
    window = full[4 * fr.out_window: 4 * fr.out_window + 16]
    return [bytes_to_lane(window[4 * i:4 * i + 4]) for i in range(DIGEST_LANES)]


# ---------------------------------------------------------------------------
# Route 2 -- word level.  This is what the chip proves.
# ---------------------------------------------------------------------------

def socket_message_words(a: list[int], b: list[int],
                         fr: Framing = HONEST_7) -> list[int]:
    m = [0] * 16
    for i in range(DIGEST_LANES):
        m[fr.a_slot + i] = a[i] & MASK32
        m[fr.b_slot + i] = b[i] & MASK32
    m[fr.tag_slot] = fr.tag_word & MASK32
    if not fr.lane_le:
        # A big-endian lane serialisation changes the message WORDS, because a
        # word is read little-endian from the byte string.
        for i in range(DIGEST_LANES):
            m[fr.a_slot + i] = int.from_bytes(lane_to_bytes(a[i], False), "little")
            m[fr.b_slot + i] = int.from_bytes(lane_to_bytes(b[i], False), "little")
    return m


def socket_digest_wordlevel(a: list[int], b: list[int],
                            fr: Framing = HONEST_7) -> list[int]:
    saved = list(ora.MSG_PERMUTATION)
    ora.MSG_PERMUTATION[:] = list(fr.msg_permutation)
    try:
        out = ora.compress(list(fr.cv), socket_message_words(a, b, fr),
                           fr.counter, fr.block_len, fr.flags, rounds=fr.rounds)
    finally:
        ora.MSG_PERMUTATION[:] = saved
    return out[fr.out_window: fr.out_window + DIGEST_LANES]


def socket_digest(a: list[int], b: list[int], fr: Framing = HONEST_7) -> list[int]:
    """THE reference the gate checks the chip against.  Both routes, asserted equal."""
    w = socket_digest_wordlevel(a, b, fr)
    if fr.counter == 0 and fr.block_len == BLOCK_LEN_LFMC and \
            fr.flags == FLAGS_LFMC and tuple(fr.cv) == tuple(IV) and \
            (fr.a_slot, fr.b_slot, fr.tag_slot) == (0, 4, 8) and \
            tuple(fr.msg_permutation) == tuple(ora.MSG_PERMUTATION):
        # The byte-level route only *exists* for the honest framing -- it is a
        # call to the tree hasher, which fixes h/t/block_len/flags itself.
        bl = socket_digest_bytelevel(a, b, fr)
        assert w == bl, (
            "FRAMING CHECK FAILED: word-level and byte-level routes disagree\n"
            f"  word={[hex(x) for x in w]}\n  byte={[hex(x) for x in bl]}")
    return w


# ---------------------------------------------------------------------------
# The negative-control catalogue, as framing perturbations.
# ---------------------------------------------------------------------------

CONTROLS: dict[str, Framing] = {
    "swap_a_b":            replace(HONEST_7, a_slot=4, b_slot=0),
    "tag_changed":         replace(HONEST_7, tag_word=TAG_LFMP),
    "tag_omitted":         replace(HONEST_7, tag_word=0),
    "truncate_high_half":  replace(HONEST_7, out_window=4),
    "flags_parent":        replace(HONEST_7, flags=ora.PARENT),
    "flags_no_root":       replace(HONEST_7, flags=ora.CHUNK_START | ora.CHUNK_END),
    "block_len_64":        replace(HONEST_7, block_len=64),
    "block_len_32":        replace(HONEST_7, block_len=32),
    "counter_one":         replace(HONEST_7, counter=1),
    "cv_zero":             replace(HONEST_7, cv=tuple([0] * 8)),
    "lanes_big_endian":    replace(HONEST_7, lane_le=False),
    "tag_slot_moved":      replace(HONEST_7, tag_slot=9),
    "msg_perm_swapped":    replace(
        HONEST_7,
        msg_permutation=tuple([ora.MSG_PERMUTATION[1], ora.MSG_PERMUTATION[0]]
                              + list(ora.MSG_PERMUTATION[2:]))),
    "rounds_6_not_7":      replace(HONEST_7, rounds=6),
}


def effective_trace(a: list[int], b: list[int], fr: Framing):
    """Everything f actually sees: the initial state, the message schedule at
    every round, and the output window.

    Two framings with identical traces compute identical digests, necessarily.
    So a control whose trace equals the honest trace on some input is genuinely
    INAPPLICABLE on that input -- not undetected.  Deriving applicability this
    way rather than hand-listing it is deliberate: a hand-list silently grows
    stale as controls are added, and a stale entry is a control that looks
    covered and is not.
    """
    m = socket_message_words(a, b, fr)
    sched = list(m)
    scheds = []
    for r in range(fr.rounds):
        scheds.append(tuple(sched))
        if r < fr.rounds - 1:
            sched = [sched[fr.msg_permutation[i]] for i in range(16)]
    init = (tuple(fr.cv), fr.counter & MASK32, (fr.counter >> 32) & MASK32,
            fr.block_len & MASK32, fr.flags & MASK32)
    return (fr.rounds, init, tuple(scheds), fr.out_window)


def control_applicable(a: list[int], b: list[int],
                       cfr: Framing, honest_fr: Framing) -> bool:
    return effective_trace(a, b, cfr) != effective_trace(a, b, honest_fr)
