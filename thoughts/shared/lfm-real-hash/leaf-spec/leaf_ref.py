"""
THE LFML LEAF MODE — reference implementation (option C, ratified 2026-08-11).

A leaf row hashes FOUR arbitrary Goldilocks field elements. Each felt occupies
TWO lanes as checked u32 halves, so four felts fill exactly the socket's eight
input lanes — the message layout is byte-identical to a digest-mode compress and
the crate-KAT property survives untouched.

    v = lo + 2^32 * hi,     lo, hi in [0, 2^32)

CANONICITY, and why it is cheap. `p - 1 = 0xFFFFFFFF_00000000`, i.e. hi = 2^32-1
and lo = 0. So for lo, hi already known to be u32:

    v < p   <==>   NOT( hi == 2^32-1  AND  lo >= 1 )

which is just "if hi is maximal then lo is zero". The socket's EXISTING O1
machinery (byte columns + AreBytes + the lane identity) already forces lo and hi
to be u32; canonicity was the only missing piece, and it costs two witness
columns per felt rather than a 64-bit decomposition.

THE DECOMPOSITION IS CHECKED, NOT REDUCING. A non-canonical input has no
satisfying witness, so the row is unprovable — the same reject-don't-reduce shape
as O1 itself. `felt_halves` raises rather than wrapping, mirroring that.
"""

from __future__ import annotations

import os
import sys

_GATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gate-oracle")
sys.path.insert(0, _GATE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "transcript-spec"))

import blake3_oracle as ora          # noqa: E402
import socket_ref as sk              # noqa: E402

P = 2**64 - 2**32 + 1                # Goldilocks
MASK32 = 0xFFFFFFFF
MAX_HI = 0xFFFFFFFF

TAG_LFML_ASCII = b"LFML"
TAG_LFML = int.from_bytes(TAG_LFML_ASCII, "little")     # 0x4C4D464C

FELTS_PER_LEAF_ROW = 4               # 4 felts = 8 lanes = one compress input


# ---------------------------------------------------------------------------
# The felt <-> halves boundary
# ---------------------------------------------------------------------------

def is_canonical(lo: int, hi: int) -> bool:
    """The chip's canonicity predicate, stated exactly as the constraints do."""
    if not (0 <= lo <= MASK32 and 0 <= hi <= MASK32):
        return False
    return not (hi == MAX_HI and lo >= 1)


def felt_halves(v: int) -> tuple[int, int]:
    """v -> (lo, hi). REJECTS rather than reduces, matching the AIR."""
    if not 0 <= v < P:
        raise ValueError(
            f"{v:#x} is not a canonical Goldilocks element; the leaf mode "
            f"REJECTS it (reject-don't-reduce, obligation O1)")
    lo, hi = v & MASK32, (v >> 32) & MASK32
    assert is_canonical(lo, hi), "canonical felt must pass the chip predicate"
    return lo, hi


def halves_felt(lo: int, hi: int) -> int:
    if not is_canonical(lo, hi):
        raise ValueError(f"({lo:#x}, {hi:#x}) is not a canonical half-pair")
    return lo + (hi << 32)


def leaf_lanes(felts: list[int]) -> list[int]:
    """Four felts -> eight lanes, LOW half first within each felt.

    Lane order is `[lo0, hi0, lo1, hi1, lo2, hi2, lo3, hi3]`, so felt `i`
    occupies lanes `2i` and `2i+1`. Keeping a felt's two halves ADJACENT is what
    lets the canonicity gate read one pair of neighbouring lanes.
    """
    assert len(felts) == FELTS_PER_LEAF_ROW
    lanes: list[int] = []
    for v in felts:
        lo, hi = felt_halves(v)
        lanes += [lo, hi]
    return lanes


# ---------------------------------------------------------------------------
# The leaf compress
# ---------------------------------------------------------------------------

def leaf_compress(felts: list[int], rounds: int = 7) -> list[int]:
    """One LFML row: four felts -> one digest cell.

    Framing is the socket's, with `m[8] = TAG_LFML`; the eight lanes are the
    felts' halves rather than eight u32s.
    """
    lanes = leaf_lanes(felts)
    fr = sk.Framing(rounds=rounds, tag_word=TAG_LFML)
    return sk.socket_digest_wordlevel(lanes[0:4], lanes[4:8], fr)


def leaf_compress_bytelevel(felts: list[int], rounds: int = 7) -> list[int]:
    """The library-shaped route — the external anchor.

    BYTE SERIALIZATION, stated exactly: each of the eight lanes is written as
    four LITTLE-ENDIAN bytes in lane order, then the four tag bytes `"LFML"`.
    So a felt contributes its low half's 4 bytes then its high half's 4 bytes:

        msg = LE32(lo0)‖LE32(hi0)‖…‖LE32(lo3)‖LE32(hi3)‖"LFML"   (36 bytes)
        digest = BLAKE3(msg)[0..16], read back as four LE u32 lanes

    At 7 rounds this is a plain `blake3::hash` call.
    """
    lanes = leaf_lanes(felts)
    msg = b"".join(int(x).to_bytes(4, "little") for x in lanes) + TAG_LFML_ASCII
    assert len(msg) == 36
    full = ora.hash_bytes(msg, 32, rounds=rounds)
    return [int.from_bytes(full[4 * i:4 * i + 4], "little") for i in range(4)]


def leaf_over_8_felts(felts: list[int], rounds: int = 7) -> list[int]:
    """A FriToyV0 leaf covers TWO trace rows = EIGHT field elements.

    Three compresses, per the ratified pricing: two LFML rows (4 felts each)
    and one ordinary LFMC parent combining them.
    """
    assert len(felts) == 8
    d0 = leaf_compress(felts[0:4], rounds)
    d1 = leaf_compress(felts[4:8], rounds)
    return sk.socket_digest_wordlevel(d0, d1, sk.Framing(rounds=rounds))


# Boundary felts the KATs must pin, including the non-canonical rejects.
BOUNDARY_FELTS = [
    ("zero", 0),
    ("one", 1),
    ("u32_max", 2**32 - 1),
    ("two_pow_32", 2**32),
    ("p_minus_2_32", P - 2**32),
    ("p_minus_1", P - 1),
]
NON_CANONICAL = [
    ("p", P),
    ("p_plus_1", P + 1),
    ("two_pow_64_minus_1", 2**64 - 1),
]
