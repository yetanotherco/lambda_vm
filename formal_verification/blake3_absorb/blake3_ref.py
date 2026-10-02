"""
Independent BLAKE3 reference, written from the spec, usable over two backends.

The same round/compression text runs on concrete `int`s (for anchoring against
the recorded oracle vectors) and on z3 32-bit bitvectors (for the QF-BV gate).
Nothing here is derived from `prover/src/tables/blake3.rs` — that is the point:
a reference copied from the circuit proves only that the circuit equals itself.

Anchors (see `test_ref.py`):
  * the constants below against the BLAKE3 spec's IV (= SHA-256's) and the
    published message permutation;
  * `compress(..., rounds=7)` against the official BLAKE3 test vectors;
  * `compress(..., rounds=6)` against `thoughts/blake3/blake3-oracle/`'s
    recorded canonical vectors, which the executor's own primitive reproduces.

⚠ The chip implements the 6-ROUND internal variant, not standard 7-round
BLAKE3. `rounds` is a parameter here so both can be exercised.
"""

MASK32 = 0xFFFFFFFF

# BLAKE3 spec §2: IV = SHA-256's initial hash value.
IV = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
]

# BLAKE3 spec §2.2, the message word permutation applied between rounds.
MSG_PERMUTATION = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8]

# BLAKE3 spec §2.1: the 8 G-calls of a round — 4 column mixes, then 4 diagonals.
# (a, b, c, d) state indices; G-call j consumes message positions 2j and 2j+1.
G_INDICES = [
    (0, 4, 8, 12),
    (1, 5, 9, 13),
    (2, 6, 10, 14),
    (3, 7, 11, 15),
    (0, 5, 10, 15),
    (1, 6, 11, 12),
    (2, 7, 8, 13),
    (3, 4, 9, 14),
]

# The 64-byte block length every absorbed block is framed with, and the counter
# value the absorb mode fixes. Both are ABI facts of the absorb ecall, not spec
# facts: `executor::…::blake3_absorb_step_6round` calls the compression with
# `t = 0` and `block_len = BLAKE3_BLOCK_BYTES`.
ABSORB_BLOCK_LEN = 64
ABSORB_COUNTER = 0


class IntOps:
    """Concrete backend: Python ints reduced mod 2^32."""

    @staticmethod
    def add(a, b):
        return (a + b) & MASK32

    @staticmethod
    def xor(a, b):
        return a ^ b

    @staticmethod
    def rotr(x, n):
        return ((x >> n) | (x << (32 - n))) & MASK32

    @staticmethod
    def const(v):
        return v & MASK32


class BvOps:
    """z3 backend: 32-bit bitvector terms."""

    @staticmethod
    def add(a, b):
        return a + b

    @staticmethod
    def xor(a, b):
        return a ^ b

    @staticmethod
    def rotr(x, n):
        from z3 import RotateRight

        return RotateRight(x, n)

    @staticmethod
    def const(v):
        from z3 import BitVecVal

        return BitVecVal(v & MASK32, 32)


def g(ops, v, a, b, c, d, mx, my):
    """The BLAKE3 quarter-round, spec §2.1, in place on `v`."""
    v[a] = ops.add(ops.add(v[a], v[b]), mx)
    v[d] = ops.rotr(ops.xor(v[d], v[a]), 16)
    v[c] = ops.add(v[c], v[d])
    v[b] = ops.rotr(ops.xor(v[b], v[c]), 12)
    v[a] = ops.add(ops.add(v[a], v[b]), my)
    v[d] = ops.rotr(ops.xor(v[d], v[a]), 8)
    v[c] = ops.add(v[c], v[d])
    v[b] = ops.rotr(ops.xor(v[b], v[c]), 7)


def round_fn(ops, v, m):
    """One round: the 8 G-calls, consuming `m` in position order."""
    for j, (a, b, c, d) in enumerate(G_INDICES):
        g(ops, v, a, b, c, d, m[2 * j], m[2 * j + 1])


def permute(m):
    """m'[i] = m[MSG_PERMUTATION[i]]."""
    return [m[MSG_PERMUTATION[i]] for i in range(16)]


def schedule_indices(r):
    """Indices into the ORIGINAL message consumed at each position of round r.

    Round 0 consumes m[i] at position i; each later round permutes, so
    sched_{r+1}[i] = sched_r[P[i]]. Returned as a list of 16 indices.
    """
    sched = list(range(16))
    for _ in range(r):
        prev = sched
        sched = [prev[MSG_PERMUTATION[i]] for i in range(16)]
    return sched


def initial_state(ops, h, tlo, thi, block_len, flags):
    """The 16-word compression state before round 0 (spec §2.3)."""
    return [
        h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7],
        ops.const(IV[0]), ops.const(IV[1]), ops.const(IV[2]), ops.const(IV[3]),
        tlo, thi, block_len, flags,
    ]


def feed_forward(ops, v, h):
    """The 16-word output: out[i] = v[i]^v[i+8], out[i+8] = v[i+8]^h[i]."""
    out = [None] * 16
    for i in range(8):
        out[i] = ops.xor(v[i], v[i + 8])
        out[i + 8] = ops.xor(v[i + 8], h[i])
    return out


def compress(ops, h, m, tlo, thi, block_len, flags, rounds):
    """Full compression: init, `rounds` rounds with the permuting schedule,
    then feed-forward. Returns 16 words."""
    v = initial_state(ops, h, tlo, thi, block_len, flags)
    sched = list(m)
    for r in range(rounds):
        round_fn(ops, v, sched)
        if r < rounds - 1:
            sched = permute(sched)
    return feed_forward(ops, v, h)


def absorb_step(ops, cv, m, flags, rounds):
    """One absorbed block: compress under the absorb framing, keep 8 words.

    This is the pure function `executor::…::blake3_absorb_step_6round` names,
    restated from the spec side.
    """
    out = compress(
        ops,
        cv,
        m,
        ops.const(ABSORB_COUNTER),
        ops.const(ABSORB_COUNTER),
        ops.const(ABSORB_BLOCK_LEN),
        flags,
        rounds,
    )
    return out[:8]


def absorb(ops, cv, blocks, first_flags, rounds):
    """A whole absorb ecall: `first_flags` on block 0, zero on every later one."""
    for i, m in enumerate(blocks):
        cv = absorb_step(ops, cv, m, first_flags if i == 0 else ops.const(0), rounds)
    return cv


# ---------------------------------------------------------------------------
# Concrete helpers (int backend only)
# ---------------------------------------------------------------------------

def compress_int(h, m, t, block_len, flags, rounds):
    """Concrete compression with the 64-bit counter split as the chip splits it."""
    return compress(
        IntOps, list(h), list(m),
        t & MASK32, (t >> 32) & MASK32, block_len, flags, rounds,
    )
