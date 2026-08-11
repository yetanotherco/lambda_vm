"""
Validation harness for `dma_ref.py`, and the emitter for the canonical vectors.

Five independent anchors. Each one SKIPs on its own if its dependency is
missing; a missing anchor never cascades into the others and never lets the
banner claim more than actually ran (the two harness defects the BLAKE3
campaign had to fix after the fact -- see ../README.md).

  [1] libc `memmove`         -- an implementation nobody here wrote
  [2] CPython slice assign   -- a second such implementation
  [3] row/bus <-> byte level -- the decomposition really implements the copy
  [4] chunking composition   -- the guest stub really implements a long memcpy
  [5] mutation sweep         -- the anchors above are sensitive, not vacuous

Anchors 1 and 2 pin the *semantics*. Anchor 3 is the one the chip depends on:
it is the only check that the row sequence the AIR proves is the byte copy the
guest asked for. Anchor 5 is what makes 1-4 worth running.

    python3 test_oracle.py           # run everything, emit the vectors
    python3 test_oracle.py --quick   # skip the exhaustive length sweeps
"""

import ctypes
import ctypes.util
import json
import os
import random
import sys

import dma_ref as ref
from dma_ref import DMA_MEMCPY_MAX_BYTES as MAX

HERE = os.path.dirname(os.path.abspath(__file__))
VECTORS = os.path.join(HERE, "canonical_dma_vectors.json")

#: Overlap configurations every sweep runs. `delta = dst - src`.
#: 0 is the aliasing case; +-1/+-7 straddle a wide row; +-8 is exactly one row;
#: +-9/+-64 are the near cases; +-2048 is disjoint.
#: Bounded by REGION/4 so `src_off + delta` and `+ MAX` stay inside the buffer --
#: an out-of-range offset would make the libc anchor read past its own buffer and
#: "pass" on garbage.
DELTAS = [0, 1, -1, 7, -7, 8, -8, 9, -9, 64, -64, 255, -255, 2048, -2048]

BASE = 0x10_0000
REGION = 8192
#: Every sweep copies from here, so both overlap directions have room.
SRC_OFF = REGION // 2
MID = BASE + SRC_OFF

assert all(0 <= SRC_OFF + d and SRC_OFF + d + MAX <= REGION for d in DELTAS), \
    "a delta would put the destination outside the test region"


def _region(seed: int, size: int) -> dict:
    """A deterministic pseudo-random byte region based at `BASE`."""
    rng = random.Random(seed)
    return {BASE + i: rng.randrange(256) for i in range(size)}


# ---------------------------------------------------------------------------
# [1] libc memmove
# ---------------------------------------------------------------------------

def anchor_libc(quick: bool):
    """Differential against the platform C library's own `memmove`.

    Genuinely non-circular: `dma_ref.memcpy_ref` and libc share no code, and
    libc is the definition the guest's compiler-builtin `memcpy` was replacing.
    Overlap is included, which is where `memcpy` and `memmove` diverge and
    where the executor's snapshot buffer is the deciding implementation choice.
    """
    path = ctypes.util.find_library("c")
    if path is None:
        return None, "libc not found"
    libc = ctypes.CDLL(path)
    libc.memmove.restype = ctypes.c_void_p
    libc.memmove.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]

    cases = 0
    lengths = range(0, MAX + 1) if not quick else [0, 1, 7, 8, 9, 15, 16, 255, MAX]
    for n in lengths:
        for delta in DELTAS:
            src_off = SRC_OFF
            dst_off = src_off + delta
            initial = _region(n * 131 + delta, REGION)

            buf = ctypes.create_string_buffer(
                bytes(initial.get(BASE + i, 0) for i in range(REGION)), REGION)
            libc.memmove(ctypes.byref(buf, dst_off), ctypes.byref(buf, src_off), n)
            expected = list(buf.raw[:REGION])

            got = ref.memcpy_ref(initial, BASE + dst_off, BASE + src_off, n)
            actual = [got.get(BASE + i, 0) for i in range(REGION)]
            if actual != expected:
                return False, f"n={n} delta={delta}: disagrees with libc memmove"
            cases += 1
    return True, f"{cases} cases x overlap/alignment, all agree"


# ---------------------------------------------------------------------------
# [2] CPython slice assignment
# ---------------------------------------------------------------------------

def anchor_slice_assign(quick: bool):
    """Differential against `bytearray[a:b] = bytearray[c:d]`.

    CPython materialises the right-hand slice first, so this is a `memmove` too,
    written by yet another set of hands. Cheap, and it catches a snapshot bug
    even on a platform whose libc anchor is unavailable.
    """
    cases = 0
    lengths = range(0, MAX + 1) if not quick else [0, 1, 8, 9, 200, MAX]
    for n in lengths:
        for delta in DELTAS:
            src_off = SRC_OFF
            dst_off = src_off + delta
            initial = _region(n * 977 + delta, REGION)

            buf = bytearray(initial.get(BASE + i, 0) for i in range(REGION))
            buf[dst_off:dst_off + n] = buf[src_off:src_off + n]

            got = ref.memcpy_ref(initial, BASE + dst_off, BASE + src_off, n)
            if [got.get(BASE + i, 0) for i in range(REGION)] != list(buf):
                return False, f"n={n} delta={delta}: disagrees with slice assignment"
            cases += 1
    return True, f"{cases} cases x overlap/alignment, all agree"


# ---------------------------------------------------------------------------
# [3] row/bus level <-> byte level
# ---------------------------------------------------------------------------

def anchor_row_level(quick: bool, widths=None, ops=None):
    """The decomposition the AIR proves implements the copy the guest asked for.

    Four claims, all over every length 0..MAX x every overlap configuration:
      (a) replaying the MEMW multiset reproduces `memcpy_ref` byte for byte;
      (b) the row widths sum to `n` and the row `src`/`dst`/`count` sequence is
          exactly `src + prefix`, `dst + prefix`, `n - prefix`;
      (c) there is exactly one `first` row and exactly one `end` row, the `end`
          row has `count == 0`, and no other row does;
      (d) the greedy width loop equals the closed form `[8]*(n//8) + [1]*(n%8)`.

    `widths`/`ops` are injection points for the mutation sweep.
    """
    widths = widths or ref.row_widths
    ops = ops or ref.memw_ops

    lengths = range(0, MAX + 1) if not quick else [0, 1, 7, 8, 9, 16, 27, 255, MAX]
    for n in lengths:
        if widths(n) != [8] * (n // 8) + [1] * (n % 8):
            return False, f"n={n}: greedy widths disagree with the closed form"
        if sum(widths(n)) != n:
            return False, f"n={n}: widths sum to {sum(widths(n))}"

        for delta in DELTAS:
            src = MID
            dst = MID + delta
            initial = _region(n * 31 + delta, REGION)

            replayed = ref.replay_memw(ops(1000, dst, src, n, initial), initial)
            expected = ref.memcpy_ref(initial, dst, src, n)
            if replayed != expected:
                return False, f"n={n} delta={delta}: MEMW replay != memcpy_ref"

            rows = ref.row_decomposition(1000, dst, src, n, initial)
            if sum(1 for r in rows if r.first) != 1:
                return False, f"n={n}: not exactly one first row"
            if sum(1 for r in rows if r.end) != 1:
                return False, f"n={n}: not exactly one end row"
            if not rows[-1].end or rows[-1].count != 0:
                return False, f"n={n}: last row is not the terminal row"
            if any(r.count == 0 for r in rows[:-1]):
                return False, f"n={n}: a data row has count == 0"

            offset = 0
            for row, width in zip(rows[:-1], widths(n)):
                if (row.src, row.dst, row.count, row.width) != (
                        src + offset, dst + offset, n - offset, width):
                    return False, f"n={n} delta={delta}: row at offset {offset} is wrong"
                if row.value[width:] != [0] * (8 - width):
                    return False, f"n={n}: unused value lanes are not zero"
                offset += width
    return True, f"{len(list(lengths))} lengths x {len(DELTAS)} overlaps, replay == memcpy_ref"


# ---------------------------------------------------------------------------
# [4] the guest stub's chunking
# ---------------------------------------------------------------------------

def anchor_chunking(quick: bool, chunk=None):
    """`chunk_ecalls` composed over the reference is a `memcpy` of any length.

    Three claims: no chunk exceeds the bound (an oversized chunk is what the
    executor rejects); the chunk count is `ceil(n / MAX)`; and for
    non-overlapping ranges the composition equals a single `memcpy_ref`.
    Overlap is deliberately excluded here -- see `guest_memcpy`'s docstring and
    ORACLE.md O2.
    """
    chunk = chunk or ref.chunk_ecalls
    lengths = list(range(0, 1100)) if not quick else [0, 1, 255, MAX, 257, 512, 1000]
    for n in lengths:
        calls = chunk(0x2_0000, 0x1_0000, n)
        if any(c > MAX for (_, _, c) in calls):
            return False, f"n={n}: a chunk exceeds {MAX} bytes"
        if len(calls) != (n + MAX - 1) // MAX:
            return False, f"n={n}: {len(calls)} chunks, expected {(n + MAX - 1) // MAX}"
        if sum(c for (_, _, c) in calls) != n:
            return False, f"n={n}: chunks cover {sum(c for (_, _, c) in calls)} bytes"

        initial = _region(n, REGION)
        composed = dict(initial)
        for cdst, csrc, cn in calls:
            composed = ref.memcpy_ref(composed, cdst, csrc, cn)
        # The whole-length expectation cannot come from `memcpy_ref` -- that
        # models ONE ecall and rejects n > MAX. Spell the copy out instead.
        expected = dict(initial)
        for i in range(n):
            expected[0x2_0000 + i] = initial.get(0x1_0000 + i, 0)
        if composed != expected:
            return False, f"n={n}: chunked copy != a plain byte-by-byte copy"

        effect, returned = ref.guest_memcpy(initial, 0x2_0000, 0x1_0000, n)
        if returned != 0x2_0000:
            return False, f"n={n}: memcpy must return dst"
        if effect != expected:
            return False, f"n={n}: guest_memcpy disagrees with a plain copy"
    return True, f"{len(lengths)} lengths, chunk count and composition both exact"


# ---------------------------------------------------------------------------
# [5] mutation sweep -- are the anchors above sensitive?
# ---------------------------------------------------------------------------

def _mutant_all_ones(n):
    return [1] * n


def _mutant_always_wide(n):
    return [8] * ((n + 7) // 8)


def _mutant_off_by_one_tail(n):
    widths, remaining = [], n
    while remaining != 0:
        width = 8 if remaining > 8 else 1      # `>` instead of `>=`
        widths.append(min(width, remaining))
        remaining -= widths[-1]
    return widths


def _mutant_write_before_read(timestamp, dst, src, n, memory):
    """Reads at T+2, writes at T+1: the copy stops being a snapshot."""
    ops = ref.memw_ops(timestamp, dst, src, n, memory)
    return [
        op if op.is_register else
        type(op)(op.is_register, op.address,
                 timestamp + (1 if op.is_write else 2),
                 op.width, op.value, op.is_write)
        for op in ops
    ]


def _mutant_interleaved(timestamp, dst, src, n, memory):
    """Each chunk written immediately after it is read (per-chunk timestamps)."""
    out = [op for op in ref.memw_ops(timestamp, dst, src, n, memory) if op.is_register]
    offset = 0
    for i, width in enumerate(ref.row_widths(n)):
        chunk = tuple(memory.get(src + offset + j, 0) for j in range(width))
        out.append(ref.MemwOp(False, src + offset, timestamp + 1 + 2 * i, width, chunk, False))
        out.append(ref.MemwOp(False, dst + offset, timestamp + 2 + 2 * i, width, chunk, True))
        offset += width
    return out


def _mutant_chunk_257(dst, src, n):
    calls, offset, remaining = [], 0, n
    while remaining != 0:
        c = min(remaining, MAX + 1)            # one byte over the executor's bound
        calls.append((dst + offset, src + offset, c))
        offset += c
        remaining -= c
    return calls


def anchor_mutations(quick: bool):
    """Every mutant must be caught by the anchor it targets."""
    mutants = [
        ("row_widths = all ones", lambda: anchor_row_level(quick, widths=_mutant_all_ones)),
        ("row_widths = always wide", lambda: anchor_row_level(quick, widths=_mutant_always_wide)),
        ("row_widths tail off by one", lambda: anchor_row_level(quick, widths=_mutant_off_by_one_tail)),
        ("memw write before read", lambda: anchor_row_level(quick, ops=_mutant_write_before_read)),
        ("memw read/write interleaved", lambda: anchor_row_level(quick, ops=_mutant_interleaved)),
        ("chunk_ecalls at MAX+1", lambda: anchor_chunking(quick, chunk=_mutant_chunk_257)),
    ]
    survivors = []
    for name, run in mutants:
        try:
            ok, _ = run()
        except (AssertionError, ref.DmaRejected):
            ok = False              # replay_memw or the executor bound caught it
        if ok:
            survivors.append(name)
        print(f"      mutant {name:32s} -> {'SURVIVED (bad)' if ok else 'caught'}")
    if survivors:
        return False, f"{len(survivors)} mutant(s) survived: {', '.join(survivors)}"
    return True, f"all {len(mutants)} mutants caught"


# ---------------------------------------------------------------------------
# Canonical vectors
# ---------------------------------------------------------------------------

#: Hand-picked so every structural case is covered exactly once: empty, a lone
#: tail byte, a full wide row, wide+tail, the widest tail (7), an unaligned
#: unaligned-overlapping copy, both overlap directions, a page-crossing copy,
#: and the maximum chunk (which has no tail row at all).
CANONICAL_CASES = [
    ("empty", 0x1000, 0x2000, 0),
    ("single byte", 0x1000, 0x2000, 1),
    ("one wide row", 0x1000, 0x2000, 8),
    ("wide plus tail", 0x1000, 0x2000, 9),
    ("widest tail", 0x1000, 0x2000, 7),
    ("unaligned body and tail", 0x2005, 0x1003, 27),
    ("forward overlap", 0x3004, 0x3000, 24),
    ("backward overlap", 0x3000, 0x3004, 24),
    ("page crossing", 0x0FFC, 0x1FFC, 16),
    ("maximum chunk", 0x1000, 0x2000, MAX),
]


def emit_vectors():
    """Write `canonical_dma_vectors.json`: the pinned cases with their full
    row-and-column expansion, so the Rust side can be checked against this
    model without re-deriving it."""
    vectors = []
    for name, dst, src, n in CANONICAL_CASES:
        memory = {src + i: (i * 7 + 3) & 0xFF for i in range(n)}
        rows = ref.row_decomposition(0x30, dst, src, n, memory)
        vectors.append({
            "name": name,
            "timestamp": 0x30,
            "dst": dst,
            "src": src,
            "count": n,
            "widths": ref.row_widths(n),
            "data_rows": len(rows) - 1,
            "rows": [
                {
                    "src": r.src, "dst": r.dst, "count": r.count,
                    "first": r.first, "end": r.end, "tail": r.tail,
                    "width": r.width, "value": r.value,
                    "columns": ref.row_columns(r),
                }
                for r in rows
            ],
            "memw": [
                {
                    "is_register": o.is_register, "address": o.address,
                    "timestamp": o.timestamp, "width": o.width,
                    "value": list(o.value), "is_write": o.is_write,
                }
                for o in ref.memw_ops(0x30, dst, src, n, memory)
            ],
        })
    with open(VECTORS, "w") as f:
        json.dump(vectors, f, indent=1)
        f.write("\n")
    return vectors


# ---------------------------------------------------------------------------

def main():
    quick = "--quick" in sys.argv
    print("=" * 72)
    print("DMA memcpy oracle -- validation harness" + ("  (--quick)" if quick else ""))
    print("=" * 72)

    anchors = [
        ("[1] libc memmove", anchor_libc),
        ("[2] CPython slice assignment", anchor_slice_assign),
        ("[3] row/bus level <-> byte level", anchor_row_level),
        ("[4] guest stub chunking", anchor_chunking),
        ("[5] mutation sweep", anchor_mutations),
    ]
    results = {}
    for name, run in anchors:
        print(f"\n  {name}")
        ok, detail = run(quick)
        results[name] = ok
        label = {True: "PASS", False: "FAIL", None: "SKIP"}[ok]
        print(f"      {label}  {detail}")

    print("\n" + "=" * 72)
    ran = [n for n, ok in results.items() if ok is not None]
    failed = [n for n, ok in results.items() if ok is False]
    skipped = [n for n, ok in results.items() if ok is None]
    if failed:
        status = "NOT VALIDATED"
    elif not ran:
        status = "NOT VALIDATED"
    elif skipped:
        status = "PARTIALLY VALIDATED"
    else:
        status = "VALIDATED"
    print(f"VALIDATION STATUS: {status}")
    print(f"  anchored on : {', '.join(n for n in ran if results[n]) or 'nothing'}")
    if skipped:
        print(f"  NOT anchored on: {', '.join(skipped)}")
    if failed:
        print(f"  FAILING     : {', '.join(failed)}")
    if quick:
        print("  NOTE: --quick skipped the exhaustive 0..256 length sweeps.")

    if not failed:
        vectors = emit_vectors()
        print(f"\n  emitted {len(vectors)} canonical vectors -> {os.path.basename(VECTORS)}")

    print("=" * 72)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
