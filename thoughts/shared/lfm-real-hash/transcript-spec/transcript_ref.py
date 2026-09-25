"""
THE COMPRESS-CHAIN TRANSCRIPT — reference implementation (option B1, ratified).

This is the future `fixture::HostSponge` mirror and the thing the chip's
transcript rows must reproduce. It replaces `edsl::SpongeVar`'s permute-driven
overwrite-rate duplex with a chain over the FROZEN `LFM_HASH` compress socket,
so no permute socket is ever built and `MODE_P` stays pinned to 0.

STATE: one cell (4 lanes x u32 = 128 bits), initially all-zero — mirroring
`SpongeVar::new`, which starts from three zero cells.

OPERATIONS (each `compress_T` is one ordinary socket compress under the
TRANSCRIPT tag `"LFMT"`):

    absorb(c)        state <- compress_T(state, c)                  1 compress
    absorb2(c0, c1)  state <- compress_T(compress_T(state, c0), c1) 2 compresses
    squeeze()        out = state ; state <- compress_T(state, SQ(i))
                                                                    1 compress

`squeeze` outputs BEFORE advancing, mirroring `SpongeVar::squeeze_cell`
(`out = state[0]; state = permute(state)`) so the two constructions stay
structurally parallel and the diff is reviewable.

SQ(i) — THE SQUEEZE COUNTER, AND WHY IT IS FREE.  The advance operand is the
constant cell `[SQUEEZE_MARK, i, 0, 0]`, where `i` is the squeeze index. It costs
NOTHING: the eDSL fully unrolls (`edsl.rs:1-4` — "nothing loop-shaped reaches the
machine"), so `i` is a compile-time constant and the operand is a program
constant either way, pinned by `program_id`.

What it buys is §8.2's FSE-2014 lesson, written into the construction: without
it, a run of consecutive squeezes iterates ONE fixed public non-injective map,
whose functional graph an attacker can precompute — the structure the GLUON-64
T-sponge attacks exploit. With it, each step is a different map and no single
functional graph exists to analyse. See `squeeze_run_analysis.py` for the
quantitative side, which is negligible either way; this is about removing the
attack *structure*, not the bit-counting.

ABSORB/SQUEEZE SEPARATION rests primarily on the operation sequence being a
compile-time constant of the program (so a prover cannot perform a squeeze where
the program says absorb), with `SQUEEZE_MARK` as defence in depth.
"""

from __future__ import annotations

import os
import sys

_GATE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "gate-oracle")
sys.path.insert(0, _GATE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "leaf-spec"))

import blake3_oracle as ora          # noqa: E402
import socket_ref as sk              # noqa: E402

MASK32 = 0xFFFFFFFF
LANES = sk.DIGEST_LANES              # 4 lanes per cell

# --- Tag allocation ---------------------------------------------------------
# A tag is never reused for a second purpose.
TAG_LFMT_ASCII = b"LFMT"             # transcript step (this construction)
TAG_LFMT = int.from_bytes(TAG_LFMT_ASCII, "little")

# The squeeze-advance marker. Distinguishes an advance operand from an absorbed
# digest as defence in depth; the load-bearing separation is the fixed sequence.
SQUEEZE_MARK = int.from_bytes(b"SQZ0", "little")

ZERO_CELL = [0, 0, 0, 0]


def squeeze_operand(i: int) -> list[int]:
    """SQ(i) — a compile-time constant cell, hence free."""
    return [SQUEEZE_MARK, i & MASK32, 0, 0]


def compress_t(state: list[int], operand: list[int],
               rounds: int = 7) -> list[int]:
    """One transcript step: the FROZEN compress socket under the LFMT tag.

    Identical framing to the Merkle socket in every respect except `m[8]`:
    h = IV, m[0..4] = state, m[4..8] = operand, m[8] = TAG_LFMT, m[9..16] = 0,
    t = 0, block_len = 36, flags = 0x0B, digest = out[0..4].
    """
    fr = sk.Framing(rounds=rounds, tag_word=TAG_LFMT)
    return sk.socket_digest_wordlevel(state, operand, fr)


class Transcript:
    """The reference. Mirrors the eventual `HostSponge` bit for bit."""

    def __init__(self, rounds: int = 7):
        self.state = list(ZERO_CELL)
        self.rounds = rounds
        self.squeeze_index = 0
        self.compressions = 0
        self.trace: list[tuple[str, list[int]]] = []

    def absorb(self, c: list[int]) -> "Transcript":
        assert len(c) == LANES
        self.state = compress_t(self.state, c, self.rounds)
        self.compressions += 1
        self.trace.append(("absorb", list(self.state)))
        return self

    def absorb2(self, c0: list[int], c1: list[int]) -> "Transcript":
        self.absorb(c0)
        self.absorb(c1)
        self.trace[-2] = ("absorb2.0", self.trace[-2][1])
        self.trace[-1] = ("absorb2.1", self.trace[-1][1])
        return self

    def absorb_felts(self, felts: list[int]) -> "Transcript":
        """Absorb a cell of ARBITRARY field elements: leaf-hash it, then absorb
        the resulting digest. ✓ VERIFIED `edsl.rs`: `absorb_felts` is
        `let d = b.leaf(c); self.absorb(d)`.

        TWO compresses — one `LFML` leaf row plus one `LFMT` chain step — because
        a felt cell cannot enter the socket directly (obligation O1). This is the
        step the spec's original 91-row figure missed."""
        import leaf_ref as lr  # noqa: PLC0415 (kept local: leaf-spec is optional)
        d = lr.leaf_compress(felts, self.rounds)
        self.compressions += 1                      # the LFML leaf row
        self.trace.append(("absorb_felts.leaf", list(d)))
        self.absorb(d)                              # the LFMT chain step
        self.trace[-1] = ("absorb_felts.absorb", self.trace[-1][1])
        return self

    def squeeze(self) -> list[int]:
        """out = state (pre-advance), then advance with SQ(i)."""
        out = list(self.state)
        self.state = compress_t(self.state, squeeze_operand(self.squeeze_index),
                                self.rounds)
        self.squeeze_index += 1
        self.compressions += 1
        self.trace.append(("squeeze", list(out)))
        return out

    # -- the shapes the eDSL exposes ------------------------------------
    def squeeze_ext(self) -> list[int]:
        """`SpongeVar::squeeze_ext`: lanes 0-2 of a squeezed cell."""
        return self.squeeze()[0:3]

    def squeeze_bits(self, nbits: int) -> list[int]:
        """`SpongeVar::squeeze_bits`: the low `nbits` of lane 0, LSB first."""
        lane0 = self.squeeze()[0]
        return [(lane0 >> k) & 1 for k in range(nbits)]


# ---------------------------------------------------------------------------
# The byte-level (library-shaped) form — the external anchor.
# ---------------------------------------------------------------------------

def compress_t_bytelevel(state: list[int], operand: list[int],
                         rounds: int = 7) -> list[int]:
    """`BLAKE3(LE32(state) ‖ LE32(operand) ‖ "LFMT")[0..16]`, as four u32 lanes.

    At rounds = 7 this is a plain `blake3::hash` call — the property the
    7-round decision was bought for, inherited unchanged because the tag lives
    in the message and nothing else about the framing moved.
    """
    msg = (b"".join(int(x).to_bytes(4, "little") for x in state)
           + b"".join(int(x).to_bytes(4, "little") for x in operand)
           + TAG_LFMT_ASCII)
    assert len(msg) == 36
    full = ora.hash_bytes(msg, 32, rounds=rounds)
    return [int.from_bytes(full[4 * i:4 * i + 4], "little") for i in range(4)]
