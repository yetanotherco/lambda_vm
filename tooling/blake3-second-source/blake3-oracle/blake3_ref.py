"""
BLAKE3 compression-function ORACLE (reference implementation).

This is the TRUST ANCHOR for a future BLAKE3 accelerator chip in the Lambda VM
STARK prover. It is written directly from the BLAKE3 specification / reference
design, NOT copied from any implementation, and then validated externally in
`test_oracle.py` against:
  - the official BLAKE3 team's `test_vectors.json`,
  - the official `blake3` PyPI package (the reference Rust implementation),
  - Plonky3's independent `blake3-air` compression implementation.

Spec sources used while writing this file (all public):
  - BLAKE3 paper / spec, section 2.1-2.2 (compression function, G, round).
  - The reference message-permutation schedule and IV constants, which also
    appear verbatim in the vendored Plonky3 `blake3-air/src/constants.rs`
    (IV, MSG_PERMUTATION) — used here only as a cross-check of the constants,
    the mixing logic is written from the spec's G-function definition.

Everything operates on 32-bit unsigned words, little-endian, exactly as BLAKE3
specifies.
"""

# ---------------------------------------------------------------------------
# Constants (BLAKE3 spec, section 2.1)
# ---------------------------------------------------------------------------

# Initialisation vector: the first 8 words of the SHA-256 IV (fractional parts
# of the square roots of the first 8 primes). Identical to SHA-256 / BLAKE2s.
IV = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
]

# Message word permutation applied between successive rounds. After each round
# the 16 message words are permuted by this index map; round r therefore mixes
# the original message under permutation^r.  (BLAKE3 spec / reference schedule.)
MSG_PERMUTATION = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8]

# Domain-separation flags (BLAKE3 spec, table of flags).
CHUNK_START         = 1 << 0   # 0x01
CHUNK_END           = 1 << 1   # 0x02
PARENT              = 1 << 2   # 0x04
ROOT                = 1 << 3   # 0x08
KEYED_HASH          = 1 << 4   # 0x10
DERIVE_KEY_CONTEXT  = 1 << 5   # 0x20
DERIVE_KEY_MATERIAL = 1 << 6   # 0x40

# Structural sizes.
BLOCK_LEN = 64      # bytes per compression input block (16 words * 4 bytes)
CHUNK_LEN = 1024    # bytes per chunk (16 blocks)
KEY_LEN   = 32      # bytes in a key / chaining value (8 words * 4 bytes)
OUT_LEN   = 32      # default output length in bytes

MASK32 = 0xFFFFFFFF

# Standard round count for BLAKE3. Variant B is the same function with ROUNDS=6.
DEFAULT_ROUNDS = 7


# ---------------------------------------------------------------------------
# 32-bit word primitives (BLAKE3 spec, section 2.1 "G function")
# ---------------------------------------------------------------------------

def add32(a, b):
    """Addition modulo 2^32 (wrapping)."""
    return (a + b) & MASK32


def rotr(x, n):
    """Rotate the 32-bit word `x` RIGHT by `n` bits.

    BLAKE3's G uses rotation amounts 16, 12, 8, 7. Rotations by 16 and 8 are
    byte-aligned (multiples of 8); 12 and 7 are not. The chip-contract reuse
    map in ORACLE.md analyses each of these against the HWSL lookup table.
    """
    x &= MASK32
    return ((x >> n) | (x << (32 - n))) & MASK32


def g(state, a, b, c, d, mx, my):
    """The BLAKE3 quarter-round mixing function G (spec section 2.1).

    Mixes two message words `mx`, `my` into four state words at indices
    a, b, c, d of the 16-word working state. Two "half rounds" of the form
    add / xor+rotate:

        v[a] = v[a] + v[b] + mx
        v[d] = (v[d] ^ v[a]) >>> 16
        v[c] = v[c] + v[d]
        v[b] = (v[b] ^ v[c]) >>> 12
        v[a] = v[a] + v[b] + my
        v[d] = (v[d] ^ v[a]) >>>  8
        v[c] = v[c] + v[d]
        v[b] = (v[b] ^ v[c]) >>>  7
    """
    state[a] = add32(add32(state[a], state[b]), mx)
    state[d] = rotr(state[d] ^ state[a], 16)
    state[c] = add32(state[c], state[d])
    state[b] = rotr(state[b] ^ state[c], 12)
    state[a] = add32(add32(state[a], state[b]), my)
    state[d] = rotr(state[d] ^ state[a], 8)
    state[c] = add32(state[c], state[d])
    state[b] = rotr(state[b] ^ state[c], 7)


def round_fn(state, m):
    """One BLAKE3 round: 4 column mixes then 4 diagonal mixes (spec 2.1).

    `m` is the (already-permuted for this round) 16-word message schedule.
    The G calls consume message words m[0..16] in order.
    """
    # Mix the columns.
    g(state, 0, 4,  8, 12, m[0],  m[1])
    g(state, 1, 5,  9, 13, m[2],  m[3])
    g(state, 2, 6, 10, 14, m[4],  m[5])
    g(state, 3, 7, 11, 15, m[6],  m[7])
    # Mix the diagonals.
    g(state, 0, 5, 10, 15, m[8],  m[9])
    g(state, 1, 6, 11, 12, m[10], m[11])
    g(state, 2, 7,  8, 13, m[12], m[13])
    g(state, 3, 4,  9, 14, m[14], m[15])


def permute(m):
    """Apply MSG_PERMUTATION to a 16-word message list, returning a new list."""
    return [m[MSG_PERMUTATION[i]] for i in range(16)]


# ---------------------------------------------------------------------------
# The compression function `f` (BLAKE3 spec, section 2.2)
# ---------------------------------------------------------------------------

def compress(chaining_value, block_words, counter, block_len, flags,
             rounds=DEFAULT_ROUNDS):
    """BLAKE3 compression function.

    Inputs:
      chaining_value : list of 8 u32 words (h[0..8])
      block_words    : list of 16 u32 words (m[0..16])
      counter        : u64 block counter t
      block_len      : u32 number of input bytes in this block (0..64)
      flags          : u32 domain-separation flags
      rounds         : number of rounds (7 = standard, 6 = variant B)

    Returns a list of 16 u32 words: the full compression output. The truncated
    8-word chaining value used elsewhere in the tree is `output[0:8]`.

    The 16-word initial working state v is:
        v[0..8]  = chaining_value[0..8]
        v[8..12] = IV[0..4]
        v[12]    = counter mod 2^32   (low  32 bits of t)
        v[13]    = counter >> 32      (high 32 bits of t)
        v[14]    = block_len
        v[15]    = flags
    Then `rounds` rounds are applied, permuting the message schedule between
    rounds. Finally the feed-forward XOR produces the 16-word output:
        output[i]     = v[i] ^ v[i+8]          for i in 0..8
        output[i+8]   = v[i+8] ^ chaining_value[i]   for i in 0..8
    """
    assert len(chaining_value) == 8
    assert len(block_words) == 16
    assert 0 <= counter < (1 << 64)

    counter_low = counter & MASK32
    counter_high = (counter >> 32) & MASK32

    state = [
        chaining_value[0], chaining_value[1], chaining_value[2], chaining_value[3],
        chaining_value[4], chaining_value[5], chaining_value[6], chaining_value[7],
        IV[0], IV[1], IV[2], IV[3],
        counter_low & MASK32, counter_high & MASK32, block_len & MASK32, flags & MASK32,
    ]

    # Local copy of the message schedule; permuted between rounds.
    m = list(block_words)
    for r in range(rounds):
        round_fn(state, m)
        # Permute between rounds. The permutation after the final round is
        # never consumed, so applying it only for r < rounds-1 is equivalent;
        # we permute between rounds to keep the loop structure obvious.
        if r < rounds - 1:
            m = permute(m)

    # Feed-forward XOR producing the full 16-word output.
    output = [0] * 16
    for i in range(8):
        output[i] = state[i] ^ state[i + 8]
        output[i + 8] = state[i + 8] ^ chaining_value[i]
    return output


def compress_cv(chaining_value, block_words, counter, block_len, flags,
                rounds=DEFAULT_ROUNDS):
    """The truncated 8-word chaining value: first 8 words of `compress`."""
    return compress(chaining_value, block_words, counter, block_len, flags,
                    rounds)[:8]


# ===========================================================================
# Variant B: 6-round BLAKE3 compression.
#
# This is EXACTLY `compress(..., rounds=6)`. It is a NONSTANDARD function with
# no external test vectors; ORACLE.md documents its canonical vectors. The only
# difference from the validated 7-round function is the loop bound `rounds`.
# ===========================================================================

def compress_6round(chaining_value, block_words, counter, block_len, flags):
    """6-round variant of the BLAKE3 compression function (variant B).

    Rounds 0..5 are applied with message permutations 0..5 (i.e. round r mixes
    permute^r(block_words)), then the identical feed-forward XOR finalisation.
    Everything else — IV, initial state layout, G function, feed-forward — is
    bit-for-bit identical to the 7-round function.
    """
    return compress(chaining_value, block_words, counter, block_len, flags,
                    rounds=6)


# ===========================================================================
# Full BLAKE3 tree hash, built ON TOP of `compress`.
#
# This exists ONLY so the compression function can be validated against the
# official whole-hash test vectors (which exercise `compress` under every flag
# combination and many counter values). The chip does NOT implement the tree;
# it implements `compress`. Written from the spec's tree/chunk structure.
# ===========================================================================

def words_from_le_bytes(b):
    """Convert a bytes object (len multiple of 4) into a list of u32 words."""
    assert len(b) % 4 == 0
    return [int.from_bytes(b[i:i + 4], "little") for i in range(0, len(b), 4)]


def le_bytes_from_words(words):
    return b"".join((w & MASK32).to_bytes(4, "little") for w in words)


class _Output:
    """A not-yet-finalised node (chunk or parent). Can emit a chaining value
    or an extendable root output (spec section 2.3, XOF)."""

    def __init__(self, input_cv, block_words, counter, block_len, flags, rounds):
        self.input_cv = input_cv
        self.block_words = block_words
        self.counter = counter
        self.block_len = block_len
        self.flags = flags
        self.rounds = rounds

    def chaining_value(self):
        return compress(self.input_cv, self.block_words, self.counter,
                        self.block_len, self.flags, self.rounds)[:8]

    def root_output_bytes(self, out_len):
        out = bytearray()
        counter = 0
        while len(out) < out_len:
            words = compress(self.input_cv, self.block_words, counter,
                             self.block_len, self.flags | ROOT, self.rounds)
            # The ROOT output uses ALL 16 output words (this is why compress
            # returns 16 words rather than the truncated 8).
            out += le_bytes_from_words(words)
            counter += 1
        return bytes(out[:out_len])


class _ChunkState:
    def __init__(self, key_words, chunk_counter, flags, rounds):
        self.cv = list(key_words)
        self.chunk_counter = chunk_counter
        self.block = b""
        self.blocks_compressed = 0
        self.flags = flags
        self.rounds = rounds

    def _start_flag(self):
        return CHUNK_START if self.blocks_compressed == 0 else 0

    def update(self, data):
        while data:
            if len(self.block) == BLOCK_LEN:
                block_words = words_from_le_bytes(self.block)
                self.cv = compress(self.cv, block_words, self.chunk_counter,
                                   BLOCK_LEN, self.flags | self._start_flag(),
                                   self.rounds)[:8]
                self.blocks_compressed += 1
                self.block = b""
            take = min(BLOCK_LEN - len(self.block), len(data))
            self.block += data[:take]
            data = data[take:]

    def output(self):
        block_words = words_from_le_bytes(self.block + b"\x00" * (BLOCK_LEN - len(self.block)))
        return _Output(self.cv, block_words, self.chunk_counter, len(self.block),
                       self.flags | self._start_flag() | CHUNK_END, self.rounds)


def _parent_output(left_cv, right_cv, key_words, flags, rounds):
    block_words = left_cv + right_cv  # 16 words
    return _Output(list(key_words), block_words, 0, BLOCK_LEN, flags | PARENT, rounds)


class Blake3Hasher:
    """Minimal BLAKE3 tree hasher over the reference `compress`.

    Supports the three official modes (default hash, keyed hash, derive-key)
    and extendable output, so it can be checked against `test_vectors.json`.
    """

    def __init__(self, key_words, flags, rounds=DEFAULT_ROUNDS):
        self.key_words = list(key_words)
        self.flags = flags
        self.rounds = rounds
        self.chunk_state = _ChunkState(self.key_words, 0, flags, rounds)
        self.cv_stack = []  # list of 8-word chaining values

    @classmethod
    def default(cls, rounds=DEFAULT_ROUNDS):
        return cls(IV, 0, rounds)

    @classmethod
    def keyed(cls, key32, rounds=DEFAULT_ROUNDS):
        assert len(key32) == KEY_LEN
        return cls(words_from_le_bytes(key32), KEYED_HASH, rounds)

    @classmethod
    def derive_key(cls, context_string, rounds=DEFAULT_ROUNDS):
        # Phase 1: hash the context string in DERIVE_KEY_CONTEXT mode to get a
        # 32-byte context key; Phase 2: keyed-hash the material with that key
        # under DERIVE_KEY_MATERIAL.
        ctx_hasher = cls(IV, DERIVE_KEY_CONTEXT, rounds)
        ctx_hasher.update(context_string.encode("utf-8") if isinstance(context_string, str) else context_string)
        context_key = ctx_hasher.finalize(KEY_LEN)
        return cls(words_from_le_bytes(context_key), DERIVE_KEY_MATERIAL, rounds)

    def _add_chunk_cv(self, new_cv, total_chunks):
        # Merge the CV stack following the binary-tree structure. A completed
        # subtree is merged whenever the total chunk count is even at that level.
        while total_chunks & 1 == 0:
            left = self.cv_stack.pop()
            new_cv = _parent_output(left, new_cv, self.key_words, self.flags,
                                    self.rounds).chaining_value()
            total_chunks >>= 1
        self.cv_stack.append(new_cv)

    def update(self, data):
        data = bytes(data)
        while data:
            if len(self.chunk_state.block) == BLOCK_LEN and \
                    self.chunk_state.blocks_compressed == CHUNK_LEN // BLOCK_LEN - 1:
                # current chunk is full: finalise it and start a new one.
                chunk_cv = self.chunk_state.output().chaining_value()
                total_chunks = self.chunk_state.chunk_counter + 1
                self._add_chunk_cv(chunk_cv, total_chunks)
                self.chunk_state = _ChunkState(self.key_words, total_chunks,
                                               self.flags, self.rounds)
            # How many bytes still fit in the current chunk.
            want = CHUNK_LEN - self._chunk_len()
            take = min(want, len(data))
            self.chunk_state.update(data[:take])
            data = data[take:]

    def _chunk_len(self):
        return self.chunk_state.blocks_compressed * BLOCK_LEN + len(self.chunk_state.block)

    def finalize(self, out_len=OUT_LEN):
        # Walk the current chunk's output up the CV stack, XORing/parenting all
        # the way to the root, and emit the root output.
        output = self.chunk_state.output()
        parent_nodes_remaining = len(self.cv_stack)
        while parent_nodes_remaining > 0:
            parent_nodes_remaining -= 1
            left = self.cv_stack[parent_nodes_remaining]
            output = _parent_output(left, output.chaining_value(),
                                    self.key_words, self.flags, self.rounds)
        return output.root_output_bytes(out_len)


def blake3_hash(data, out_len=OUT_LEN, rounds=DEFAULT_ROUNDS):
    h = Blake3Hasher.default(rounds)
    h.update(data)
    return h.finalize(out_len)


def blake3_keyed_hash(key32, data, out_len=OUT_LEN, rounds=DEFAULT_ROUNDS):
    h = Blake3Hasher.keyed(key32, rounds)
    h.update(data)
    return h.finalize(out_len)


def blake3_derive_key(context_string, key_material, out_len=OUT_LEN, rounds=DEFAULT_ROUNDS):
    h = Blake3Hasher.derive_key(context_string, rounds)
    h.update(key_material)
    return h.finalize(out_len)


if __name__ == "__main__":
    # Tiny smoke test: empty-input default hash (compare to test_oracle.py).
    print("blake3('') =", blake3_hash(b"").hex())
