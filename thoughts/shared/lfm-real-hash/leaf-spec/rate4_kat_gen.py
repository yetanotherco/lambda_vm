"""
RATE-4 KAT GENERATOR — renders the Rust vector tables for the widened socket.

The socket's message grew from 8 lanes to TWELVE and its `block_len` from 36 to
52 when the leaf gained a chaining accumulator in the message (COMMIT.md §1.2,
the `RATE = 4` construction). That moves EVERY digest in all THREE domains —
`LFML`, `LFMC` and `LFMT` — because `block_len` is `v[14]` and cannot be made
mode-dependent (COMMIT.md §1.4.4 H9). So every pinned vector re-blesses, and
this is the script that re-pins them.

    msg = LE32(lane0..lane11) ‖ tag                                (52 bytes)

  LFML   lanes = acc[0..4] ‖ (lo_i ‖ hi_i for each of four felts)
  LFMC   lanes = a[0..4] ‖ b[0..4] ‖ 0^4      (the third input cell, pinned)
  LFMT   lanes = state ‖ operand ‖ 0^4

★ THE VECTORS COME FROM THE ORACLE, NOT FROM THE RUST. Every digest below is
computed by `blake3_oracle.hash_bytes` — a from-scratch Python BLAKE3 written
before any of this Rust existed — over a message this script serialises itself.
Nothing here reads the implementation under test, which is the only thing that
makes the tables a specification rather than a recording. 52 < 64 keeps a row a
single block, so at 7 rounds each vector is also a plain `blake3::hash` call and
the crate anchor survives the widening.

The INPUTS are carried over unchanged: the socket and transcript tables are
rewritten digest-line by digest-line out of the existing Rust, so their diff is
exactly the digests and a reviewer can see that nothing structural moved. The
leaf table is re-rendered whole, because the leaf row genuinely gained an input.

Run: python3 rate4_kat_gen.py [--check]
"""

from __future__ import annotations

import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "gate-oracle"))

import blake3_oracle as ora          # noqa: E402

P = 2**64 - 2**32 + 1                # Goldilocks
MASK32 = 0xFFFFFFFF

NUM_LANES = 12                       # 4 accumulator lanes + 4 felts' halves
ACC_LANES = 4
FELTS_PER_LEAF = 4
BLOCK_LEN = 4 * (NUM_LANES + 1)      # 52

TAG_LFMC = int.from_bytes(b"LFMC", "little")
TAG_LFML = int.from_bytes(b"LFML", "little")
TAG_LFMT = int.from_bytes(b"LFMT", "little")

RUST = os.path.join(_HERE, "..", "..", "..", "..", "prover", "src", "lfm")


# ---------------------------------------------------------------------------
# The construction
# ---------------------------------------------------------------------------

def _u32le(x: int) -> bytes:
    if not 0 <= x <= MASK32:
        raise ValueError(f"{x:#x} is not a u32 — obligation O1 (reject, never reduce)")
    return int(x).to_bytes(4, "little")


def socket_digest(lanes: list[int], tag: int, rounds: int) -> list[int]:
    """One row: twelve lanes and a tag -> four digest lanes.

    Serialised as a plain byte string and hashed by the oracle's `hash_bytes`,
    which for any input under 64 bytes is exactly one compression with h = IV,
    t = 0, block_len = len, flags = CHUNK_START|CHUNK_END|ROOT. That IS the
    socket's framing, which is why the anchor holds.
    """
    assert len(lanes) == NUM_LANES
    msg = b"".join(_u32le(x) for x in lanes) + int(tag).to_bytes(4, "little")
    assert len(msg) == BLOCK_LEN
    full = ora.hash_bytes(msg, 32, rounds=rounds)
    return [int.from_bytes(full[4 * i:4 * i + 4], "little") for i in range(4)]


def digest_lanes(a: list[int], b: list[int]) -> list[int]:
    """A digest row's lanes: the two cells it reads, then the pinned zeros."""
    return list(a) + list(b) + [0] * (NUM_LANES - 8)


def felt_halves(v: int) -> tuple[int, int]:
    """v -> (lo, hi). REJECTS rather than reduces, matching the AIR."""
    if not 0 <= v < P:
        raise ValueError(f"{v:#x} is not a canonical Goldilocks element")
    lo, hi = v & MASK32, (v >> 32) & MASK32
    assert not (hi == MASK32 and lo >= 1), "canonical felt must pass the predicate"
    return lo, hi


def leaf_lanes(acc: list[int], felts: list[int]) -> list[int]:
    """A leaf row's twelve lanes: the accumulator, then the felts' halves.

    Halves stay ADJACENT within a felt and start above the accumulator, so felt
    `i` occupies lanes `4 + 2i` and `4 + 2i + 1`.
    """
    assert len(acc) == ACC_LANES and len(felts) == FELTS_PER_LEAF
    lanes = list(acc)
    for v in felts:
        lo, hi = felt_halves(v)
        lanes += [lo, hi]
    return lanes


def leaf_digest(acc: list[int], felts: list[int], rounds: int) -> list[int]:
    """ONE compression that absorbs four felts AND chains the accumulator."""
    return socket_digest(leaf_lanes(acc, felts), TAG_LFML, rounds)


def leaf_chain(felts: list[int], rounds: int) -> list[int]:
    """A wide leaf: the felts absorbed four at a time into one chain.

    The chain starts at the zero cell here, NOT at COMMIT.md §1.3's shape
    header — these fixture leaves are fixed-shape by the program that builds
    them and have no width to bind. A commitment layer over arbitrary-width
    openings must open the chain at the header instead.
    """
    assert len(felts) % FELTS_PER_LEAF == 0
    acc = [0] * ACC_LANES
    for j in range(0, len(felts), FELTS_PER_LEAF):
        acc = leaf_digest(acc, felts[j:j + FELTS_PER_LEAF], rounds)
    return acc


def leaf_chain_compressions(num_felts: int) -> int:
    """One compression per RATE felts — no fold, so no `2 *`."""
    return -(-num_felts // FELTS_PER_LEAF)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def lanes_rs(v: list[int]) -> str:
    return "[" + ", ".join(f"0x{x:08X}" for x in v) + "]"


def rewrite_digests(path: str, inputs: list[str], digests: dict[str, int],
                    tag: int, check: bool) -> tuple[int, int]:
    """Recompute a table's digest fields IN PLACE from its own input fields.

    The inputs are read back out of the Rust rather than restated here, so this
    cannot quietly re-pin a vector to a different input than the one the table
    claims — and the resulting diff is exactly the digest lines.
    """
    src = open(path).read()
    field = lambda name: rf"{name}: \[((?:0x[0-9A-Fa-f]{{8}}(?:, )?)+)\],"
    cells = [[int(x, 16) for x in m.group(1).split(", ")]
             for m in re.finditer(field(inputs[0]), src)]
    other = [[int(x, 16) for x in m.group(1).split(", ")]
             for m in re.finditer(field(inputs[1]), src)]
    assert len(cells) == len(other), f"{path}: {inputs} counts disagree"

    moved = 0
    for name, rounds in digests.items():
        wanted = [socket_digest(digest_lanes(a, b), tag, rounds)
                  for a, b in zip(cells, other)]
        it = iter(wanted)
        def sub(m, it=it):
            nonlocal moved
            new = lanes_rs(next(it))
            if m.group(0) != f"{name}: {new},":
                moved += 1
            return f"{name}: {new},"
        src = re.sub(field(name), sub, src)
        assert next(it, None) is None, f"{path}: {name} count != input count"

    if check:
        if src != open(path).read():
            raise SystemExit(f"STALE: {path} does not match the oracle")
    else:
        open(path, "w").write(src)
    return len(cells), moved


LEAF_HEADER = '''//! LEAF-mode KATs for the LFM `"LFML"` domain, at 6 and 7 rounds.
//!
//! GENERATED — do not hand-edit. Rendered by
//! `thoughts/shared/lfm-real-hash/leaf-spec/rate4_kat_gen.py` from
//! `gate-oracle/blake3_oracle.py`, a Python BLAKE3 written **before any Rust
//! existed**. These vectors are a specification the implementation is checked
//! against, not a recording of what the implementation happened to do.
//!
//! A leaf row hashes FOUR arbitrary Goldilocks elements AND chains an
//! accumulator, in ONE compression (COMMIT.md §1.2). The accumulator is a digest
//! cell and fills lanes 0–3; each felt occupies two lanes above it as checked
//! `u32` halves, `[lo0, hi0, …, lo3, hi3]`. So the message is
//! `LE32(acc ‖ halves) ‖ "LFML"` — 52 bytes, still one BLAKE3 block, so the
//! crate-KAT anchor survives the widening.

/// One leaf row: the chaining accumulator, four felts, the twelve lanes they
/// become, and the digest at each round count.
pub struct LeafVector {
    pub name: &'static str,
    pub acc: [u32; 4],
    pub felts: [u64; 4],
    pub lanes: [u32; 12],
    /// Digest at 6 rounds (the A6R variant; no library computes it).
    pub digest_6: [u32; 4],
    /// Digest at 7 rounds — `blake3::hash(LE32(lanes) ‖ "LFML")[..16]`.
    pub digest_7: [u32; 4],
}

'''

# The five ratified felt inputs, each now carried by a DIFFERENT accumulator, so
# a row that dropped the accumulator from its preimage could not reproduce the
# table. `acc_ignored_control` is that discrimination made explicit: same felts
# as `zeros`, nonzero accumulator, and the suite asserts the digests differ.
LEAF_CASES = [
    ("zeros", [0, 0, 0, 0], [0, 0, 0, 0]),
    ("boundary_mix", [0x00000000, 0x00000001, 0xFFFFFFFE, 0xFFFFFFFF],
     [0, 1, P - 1, 2**32]),
    ("all_p_minus_1", [0xFFFFFFFF] * 4, [P - 1] * 4),
    ("ramp", [0x01020304, 0x05060708, 0x090A0B0C, 0x0D0E0F10],
     [0x0102030405060708, 0x1112131415161718,
      0x2122232425262728, 0x3132333435363738]),
    ("u32_edges", [0x80000000, 0x7FFFFFFF, 0x00010000, 0x0000FFFF],
     [2**32 - 1, 2**32, P - 2**32, 1]),
    ("acc_ignored_control", [0x11121314, 0x15161718, 0x191A1B1C, 0x1D1E1F20],
     [0, 0, 0, 0]),
]

FRI_FELTS = [P - 1, 0, 1, 2**32, 12345678901234567, 2**32 - 1, P - 2**32, 999]


def render_leaf(path: str, check: bool) -> int:
    out = [LEAF_HEADER]
    out.append(f"pub const LEAF_VECTORS: [LeafVector; {len(LEAF_CASES)}] = [\n")
    for name, acc, felts in LEAF_CASES:
        out.append("    LeafVector {\n")
        out.append(f'        name: "{name}",\n')
        out.append(f"        acc: {lanes_rs(acc)},\n")
        out.append("        felts: [" + ", ".join(f"{v}u64" for v in felts) + "],\n")
        out.append(f"        lanes: {lanes_rs(leaf_lanes(acc, felts))},\n")
        out.append(f"        digest_6: {lanes_rs(leaf_digest(acc, felts, 6))},\n")
        out.append(f"        digest_7: {lanes_rs(leaf_digest(acc, felts, 7))},\n")
        out.append("    },\n")
    out.append("];\n")

    tail = open(os.path.join(RUST, "leaf_kats.rs")).read()
    keep = tail[tail.index("/// A boundary felt and the halves"):]
    keep = keep[:keep.index("/// The eight-felt `FriToyV0` leaf")]
    out.append("\n" + keep)

    out.append('''/// The eight-felt `FriToyV0` leaf: ONE `LFML` chain, two rows, no fold.
pub struct FriLeafVector {
    pub felts: [u64; 8],
    pub digest_6: [u32; 4],
    pub digest_7: [u32; 4],
    /// Compressions the whole leaf costs — 3 before the accumulator moved into
    /// the message, 2 after (COMMIT.md §1.4.1: the RATE, measured).
    pub compresses: usize,
}

pub const FRI_LEAF: FriLeafVector = FriLeafVector {
''')
    out.append("    felts: [\n")
    for v in FRI_FELTS:
        out.append(f"        {v}u64,\n")
    out.append("    ],\n")
    out.append(f"    digest_6: {lanes_rs(leaf_chain(FRI_FELTS, 6))},\n")
    out.append(f"    digest_7: {lanes_rs(leaf_chain(FRI_FELTS, 7))},\n")
    out.append(f"    compresses: {leaf_chain_compressions(len(FRI_FELTS))},\n")
    out.append("};\n")

    src = "".join(out)
    if check:
        if src != open(path).read():
            raise SystemExit(f"STALE: {path} does not match the oracle")
    else:
        open(path, "w").write(src)
    return len(LEAF_CASES)


SQUEEZE_MARK = int.from_bytes(b"SQ00"[:4], "little") if False else 811225427

MAIN_ROOT = [0x01020304, 0x05060708, 0x090A0B0C, 0x0D0E0F10]
L1_ROOT = [0x11121314, 0x15161718, 0x191A1B1C, 0x1D1E1F20]
T0W = [0xDEADBEEF, 0xCAFEBABE, 0x8BADF00D, 0xFEEDFACE]
T1W = [0x0BADC0DE, 0xD15EA5E5, 0xC0FFEE00, 0xBAAAAAAD]
NUM_QUERIES = 4
QUERY_BITS = 4


def fri_toy_transcript(rounds: int) -> tuple[list[list[int]], list[list[int]]]:
    """The `FriToyV0` preamble, op by op — the K2 end-to-end vector.

    `absorb(main_root), squeeze, squeeze, absorb(l1_root), squeeze,
    absorb_felts(t0w), absorb_felts(t1w), 4x squeeze_bits`. The two data absorbs
    go through the LEAF encoding and absorb the digest, which is what makes this
    vector move with the leaf construction and not only with `block_len`.
    """
    state = [0, 0, 0, 0]
    idx = 0
    states, outputs = [], []

    def absorb(operand):
        nonlocal state
        state = socket_digest(digest_lanes(state, operand), TAG_LFMT, rounds)
        states.append(list(state))

    def squeeze():
        nonlocal state, idx
        outputs.append(list(state))
        sq = [SQUEEZE_MARK, idx, 0, 0]
        state = socket_digest(digest_lanes(state, sq), TAG_LFMT, rounds)
        idx += 1
        states.append(list(state))

    absorb(MAIN_ROOT)
    squeeze()
    squeeze()
    absorb(L1_ROOT)
    squeeze()
    for cell in (T0W, T1W):
        # DATA: leaf-hashed from the chain start, then the digest absorbed.
        absorb(leaf_digest([0] * ACC_LANES, cell, rounds))
    for _ in range(NUM_QUERIES):
        squeeze()
    return states, outputs


def render_end_to_end(path: str, check: bool) -> int:
    src = open(path).read()
    moved = 0
    for rounds, const in ((7, "FRI_TOY_7"), (6, "FRI_TOY_6")):
        states, outputs = fri_toy_transcript(rounds)
        body = ["    states: [\n"]
        for s in states:
            body.append(f"        {lanes_rs(s)},\n")
        body.append("    ],\n")
        for name, o in (("alpha", outputs[0]), ("zeta0", outputs[1]),
                        ("zeta1", outputs[2])):
            body.append(f"    {name}: {lanes_rs(o[:3])},\n")
        bits = []
        for q in range(NUM_QUERIES):
            lane0 = outputs[3 + q][0]
            bits.append("[" + ", ".join(str((lane0 >> k) & 1)
                                        for k in range(QUERY_BITS)) + "]")
        body.append("    query_bits: [" + ", ".join(bits) + "],\n")

        pat = re.compile(rf"(pub const {const}: EndToEndVector = EndToEndVector \{{\n).*?(\}};\n)",
                         re.S)
        new = pat.sub(lambda m: m.group(1) + "".join(body) + m.group(2), src)
        if new != src:
            moved += 1
        src = new

    if check:
        if src != open(path).read():
            raise SystemExit(f"STALE: {path} end-to-end vectors")
    else:
        open(path, "w").write(src)
    return moved


def main() -> None:
    check = "--check" in sys.argv
    n, moved = rewrite_digests(os.path.join(RUST, "blake3_socket_kats.rs"),
                               ["a", "b"], {"digest_6": 6, "digest_7": 7},
                               TAG_LFMC, check)
    print(f"socket    : {n} vectors, {moved} digests moved")
    n, moved = rewrite_digests(os.path.join(RUST, "transcript_kats.rs"),
                               ["state", "operand"],
                               {"result_6": 6, "result_7": 7}, TAG_LFMT, check)
    print(f"transcript: {n} vectors, {moved} digests moved")
    moved = render_end_to_end(os.path.join(RUST, "transcript_kats.rs"), check)
    print(f"end-to-end: {moved} FriToyV0 vectors re-pinned")
    n = render_leaf(os.path.join(RUST, "leaf_kats.rs"), check)
    print(f"leaf      : {n} vectors re-rendered (the row gained an input)")


if __name__ == "__main__":
    main()
