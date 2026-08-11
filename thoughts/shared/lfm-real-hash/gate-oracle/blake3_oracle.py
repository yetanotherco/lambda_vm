"""
BLAKE3 reference, round-parameterised.  THE ORACLE'S LAYER 1.

Written from the BLAKE3 specification (the `reference_impl` algorithm: compression
function, chunk state, CV stack, output node, XOF), NOT transcribed from any
in-repo file.  Independence is the point: `thoughts/blake3/blake3-oracle/blake3_ref.py`
and `prover/src/lfm/blake3.rs` are cross-checks, not sources.

WHAT MAKES THIS TRUSTWORTHY (the provenance chain, in order of strength):

  1. At `rounds = 7` this is standard BLAKE3, so it is checked against the
     OFFICIAL BLAKE3 test vectors (`official_test_vectors.json`, upstream's
     `test_vectors.json`) in all three modes -- hash, keyed_hash, derive_key --
     across 35 input lengths and the full extended-output (XOF) window.  That
     anchor is external to this repo and to this project.
  2. Differentially checked against the independently-written in-repo reference
     `thoughts/blake3/blake3-oracle/blake3_ref.py` (two agreeing sources).
  3. At `rounds = 6` NO external anchor exists -- no library computes it and no
     published vector contains it.  6-round values in this file are therefore
     the *definition* of the 6-round variant, defensible only as "the same code
     path with the loop bound changed".  That is assumption A6R and it is why
     the parameterisation is a single integer with no other edit: the 7-round
     anchor is what certifies the code path, and the 6-round instantiation
     inherits nothing but the code.

The round loop permutes the message schedule when `r < rounds - 1`, so
`rounds = 7` is bit-for-bit standard BLAKE3 with no other change.
"""

from __future__ import annotations

MASK32 = 0xFFFFFFFF

IV = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
]

MSG_PERMUTATION = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8]

# Flag bits.
CHUNK_START = 1 << 0
CHUNK_END = 1 << 1
PARENT = 1 << 2
ROOT = 1 << 3
KEYED_HASH = 1 << 4
DERIVE_KEY_CONTEXT = 1 << 5
DERIVE_KEY_MATERIAL = 1 << 6

BLOCK_LEN = 64
CHUNK_LEN = 1024

STANDARD_ROUNDS = 7

# The eight G-calls of one round: (a, b, c, d, mx_index, my_index).
G_CALLS = [
    (0, 4, 8, 12, 0, 1),
    (1, 5, 9, 13, 2, 3),
    (2, 6, 10, 14, 4, 5),
    (3, 7, 11, 15, 6, 7),
    (0, 5, 10, 15, 8, 9),
    (1, 6, 11, 12, 10, 11),
    (2, 7, 8, 13, 12, 13),
    (3, 4, 9, 14, 14, 15),
]


def rotr32(x: int, n: int) -> int:
    x &= MASK32
    return ((x >> n) | (x << (32 - n))) & MASK32


def g(v: list[int], a: int, b: int, c: int, d: int, mx: int, my: int) -> None:
    v[a] = (v[a] + v[b] + mx) & MASK32
    v[d] = rotr32(v[d] ^ v[a], 16)
    v[c] = (v[c] + v[d]) & MASK32
    v[b] = rotr32(v[b] ^ v[c], 12)
    v[a] = (v[a] + v[b] + my) & MASK32
    v[d] = rotr32(v[d] ^ v[a], 8)
    v[c] = (v[c] + v[d]) & MASK32
    v[b] = rotr32(v[b] ^ v[c], 7)


def round_fn(v: list[int], m: list[int]) -> None:
    for (a, b, c, d, ix, iy) in G_CALLS:
        g(v, a, b, c, d, m[ix], m[iy])


def permute(m: list[int]) -> list[int]:
    return [m[MSG_PERMUTATION[i]] for i in range(16)]


def compress(
    chaining_value: list[int],
    block_words: list[int],
    counter: int,
    block_len: int,
    flags: int,
    rounds: int = STANDARD_ROUNDS,
) -> list[int]:
    """The compression function f.  Returns all 16 output words.

    `rounds = 7` is standard BLAKE3.  Any other value is the LFM variant and has
    no external anchor (see the module docstring, assumption A6R).
    """
    assert len(chaining_value) == 8 and len(block_words) == 16
    state = [
        chaining_value[0], chaining_value[1], chaining_value[2], chaining_value[3],
        chaining_value[4], chaining_value[5], chaining_value[6], chaining_value[7],
        IV[0], IV[1], IV[2], IV[3],
        counter & MASK32,
        (counter >> 32) & MASK32,
        block_len & MASK32,
        flags & MASK32,
    ]
    schedule = list(block_words)
    for r in range(rounds):
        round_fn(state, schedule)
        if r < rounds - 1:
            schedule = permute(schedule)

    out = [0] * 16
    for i in range(8):
        out[i] = state[i] ^ state[i + 8]
        out[i + 8] = state[i + 8] ^ chaining_value[i]
    return out


# ---------------------------------------------------------------------------
# Tree hashing -- needed ONLY so the official vectors can anchor `compress`.
# The LFM socket never uses more than one block, but the anchor does.
# ---------------------------------------------------------------------------

def words_from_le_bytes(b: bytes) -> list[int]:
    assert len(b) % 4 == 0
    return [int.from_bytes(b[i:i + 4], "little") for i in range(0, len(b), 4)]


def le_bytes_from_words(w: list[int]) -> bytes:
    return b"".join(int(x & MASK32).to_bytes(4, "little") for x in w)


class _Output:
    """A not-yet-finalised node: the inputs to one last compression."""

    __slots__ = ("cv", "block_words", "counter", "block_len", "flags", "rounds")

    def __init__(self, cv, block_words, counter, block_len, flags, rounds):
        self.cv = cv
        self.block_words = block_words
        self.counter = counter
        self.block_len = block_len
        self.flags = flags
        self.rounds = rounds

    def chaining_value(self) -> list[int]:
        return compress(self.cv, self.block_words, self.counter,
                        self.block_len, self.flags, self.rounds)[:8]

    def root_output_bytes(self, length: int) -> bytes:
        out = bytearray()
        block_counter = 0
        while len(out) < length:
            words = compress(self.cv, self.block_words, block_counter,
                             self.block_len, self.flags | ROOT, self.rounds)
            out += le_bytes_from_words(words)
            block_counter += 1
        return bytes(out[:length])


class _ChunkState:
    def __init__(self, key_words, chunk_counter, flags, rounds):
        self.cv = list(key_words)
        self.chunk_counter = chunk_counter
        self.block = bytearray()
        self.blocks_compressed = 0
        self.flags = flags
        self.rounds = rounds

    def length(self) -> int:
        return BLOCK_LEN * self.blocks_compressed + len(self.block)

    def start_flag(self) -> int:
        return CHUNK_START if self.blocks_compressed == 0 else 0

    def update(self, data: bytes) -> None:
        while data:
            if len(self.block) == BLOCK_LEN:
                block_words = words_from_le_bytes(bytes(self.block))
                self.cv = compress(self.cv, block_words, self.chunk_counter,
                                   BLOCK_LEN, self.flags | self.start_flag(),
                                   self.rounds)[:8]
                self.blocks_compressed += 1
                self.block = bytearray()
            take = min(BLOCK_LEN - len(self.block), len(data))
            self.block += data[:take]
            data = data[take:]

    def output(self) -> _Output:
        padded = bytes(self.block) + b"\x00" * (BLOCK_LEN - len(self.block))
        return _Output(self.cv, words_from_le_bytes(padded), self.chunk_counter,
                       len(self.block), self.flags | self.start_flag() | CHUNK_END,
                       self.rounds)


def _parent_output(left_cv, right_cv, key_words, flags, rounds) -> _Output:
    return _Output(list(key_words), left_cv + right_cv, 0, BLOCK_LEN,
                   PARENT | flags, rounds)


class Hasher:
    """Full BLAKE3 tree hasher.  Exists to run the official-vector anchor."""

    def __init__(self, key_words=None, flags=0, rounds: int = STANDARD_ROUNDS):
        self.key_words = list(key_words) if key_words is not None else list(IV)
        self.flags = flags
        self.rounds = rounds
        self.chunk_state = _ChunkState(self.key_words, 0, flags, rounds)
        self.cv_stack: list[list[int]] = []

    @classmethod
    def new_keyed(cls, key: bytes, rounds: int = STANDARD_ROUNDS) -> "Hasher":
        assert len(key) == 32
        return cls(words_from_le_bytes(key), KEYED_HASH, rounds)

    @classmethod
    def new_derive_key(cls, context: str, rounds: int = STANDARD_ROUNDS) -> "Hasher":
        ctx = cls(list(IV), DERIVE_KEY_CONTEXT, rounds)
        ctx.update(context.encode("utf-8"))
        ctx_key = ctx.finalize(32)
        return cls(words_from_le_bytes(ctx_key), DERIVE_KEY_MATERIAL, rounds)

    def _add_chunk_cv(self, new_cv: list[int], total_chunks: int) -> None:
        while total_chunks & 1 == 0:
            left = self.cv_stack.pop()
            new_cv = _parent_output(left, new_cv, self.key_words,
                                    self.flags, self.rounds).chaining_value()
            total_chunks >>= 1
        self.cv_stack.append(new_cv)

    def update(self, data: bytes) -> "Hasher":
        while data:
            if self.chunk_state.length() == CHUNK_LEN:
                cv = self.chunk_state.output().chaining_value()
                counter = self.chunk_state.chunk_counter
                self._add_chunk_cv(cv, counter + 1)
                self.chunk_state = _ChunkState(self.key_words, counter + 1,
                                               self.flags, self.rounds)
            take = min(CHUNK_LEN - self.chunk_state.length(), len(data))
            self.chunk_state.update(data[:take])
            data = data[take:]
        return self

    def finalize(self, length: int = 32) -> bytes:
        output = self.chunk_state.output()
        for cv in reversed(self.cv_stack):
            output = _parent_output(cv, output.chaining_value(), self.key_words,
                                    self.flags, self.rounds)
        return output.root_output_bytes(length)


def hash_bytes(data: bytes, length: int = 32,
               rounds: int = STANDARD_ROUNDS) -> bytes:
    return Hasher(rounds=rounds).update(data).finalize(length)
