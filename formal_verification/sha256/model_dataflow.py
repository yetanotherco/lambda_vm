"""
Concrete mirror of the SHA256ROUND circuit's contract dataflow.

Every equation here corresponds 1:1 to a bus interaction / eval constraint in
prover/src/tables/sha256_round.rs, evaluated FORWARD with concrete ints. Its
sole job is to validate that the wiring hand-encoded into z3 actually
reproduces the FIPS 180-4 reference round — a wholesale wrong model is
something a symbolic UNSAT could not reveal.

The `bug` flag lets each negative control be confirmed to genuinely perturb the
output. Line citations are to sha256_round.rs on the branch this gate was
authored against; `python3 check_citations.py` re-checks them against the file.

COLUMN-ROLE MAP (sha256_round.rs:28-55)
  TS      0..2    timestamp, not part of the round function
  INDEX   2       round index t, keys the ShaK and ShaM lookups
  A       3..35   `a`, 32 bit columns, least-significant first
  B       35..39  `b`, 4 byte columns, least-significant first
  C       39..43  `c`, 4 bytes
  D       43      `d`, one field element (only ever added)
  E_      44..76  `e`, 32 bit columns
  FF      76..80  `f`, 4 bytes
  G       80..84  `g`, 4 bytes
  H       84      `h`, one field element
  OUT_A   85..87  new `a` as WordHL: lo + 2^16*hi
  OUT_E   87..89  new `e` as WordHL
  A_AND_B        89..93   BYTE_ALU AND(a, b)      per byte
  A_XOR_B        93..97   BYTE_ALU XOR(a, b)
  C_AND_A_XOR_B  97..101  BYTE_ALU AND(c, a^b)
  E_AND_F        101..105 BYTE_ALU AND(e, f)
  NOT_E_AND_G    105..109 BYTE_ALU AND(255-e, g)
  K       109     round constant, from the ShaK lookup
  W       110     schedule word, from the ShaM lookup
  CARRY_A 111     carry of the out_a addition, byte-range-checked
  CARRY_E 112     carry of the out_e addition
  MU      113     multiplicity / row-is-real flag

`a` and `e` are bits because the circuit rotates them: Sigma0 and Sigma1 are
expressions over those bit columns, not lookups. `b, c, f, g` stay bytes
because they only ever feed BYTE_ALU. `d, h` stay single elements because they
only ever appear in an addition.
"""
from sha256_ref import MASK32

# ---------------------------------------------------------------- helpers ----
def word_to_bits(v):
    """The 32 bit columns at A / E_ (sha256_round.rs:79,84 (put_bits) via put_bits)."""
    return [(v >> i) & 1 for i in range(32)]


def word_to_bytes(v):
    """The 4 byte columns at B / C / FF / G (sha256_round.rs:74-78 put_bytes)."""
    return [(v >> (8 * j)) & 0xFF for j in range(4)]


def bits_to_word(bits):
    """`bits(c, 32)` in sha256_common.rs: sum of bit_i * 2^i."""
    return sum(int(b) << i for i, b in enumerate(bits))


def bytes_to_word(bs):
    """`word(b, c)` in the Constraints closure, sha256_round.rs:208-212."""
    return sum(int(x) << (8 * j) for j, x in enumerate(bs))


def byte_of_bits(bits, j, complement=False):
    """`byte_bits(c, j, complement)` in sha256_common.rs — byte j of a
    bit-held word as a BYTE_ALU operand, or its 255-complement."""
    v = sum(int(bits[8 * j + i]) << i for i in range(8))
    return (255 - v) if complement else v


def halves(v):
    """The WordHL pair at OUT_A / OUT_E (sha256_round.rs:112-116)."""
    return [v & 0xFFFF, (v >> 16) & 0xFFFF]


def from_halves(hl):
    """`half(b, c)` in the Constraints closure, sha256_round.rs:213."""
    return int(hl[0]) + 65536 * int(hl[1])


# ------------------------------------------------------------- the model ----
def round_dataflow(state, k_t, w_t, bug=None):
    """Forward-evaluate one SHA256ROUND row through the circuit's own equations.

    state: (a,b,c,d,e,f,g,h) as 8 u32. Returns the next 8-word state, built the
    way the circuit builds it: out_a and out_e from the two addition
    constraints, and b,c,d / f,g,h by the shift the ShaRound bus performs.
    """
    a, b, c, d, e, f, g, h = state

    # --- the trace columns ---------------------------------------------------
    A = word_to_bits(a)
    E = word_to_bits(e)
    B, C, FF, G = word_to_bytes(b), word_to_bytes(c), word_to_bytes(f), word_to_bytes(g)

    # --- the five BYTE_ALU sends, sha256_round.rs:136-153 --------------------
    # Contract: ByteAlu(op, x, y, z) guarantees x,y,z are bytes and z = x op y.
    A_AND_B = [byte_of_bits(A, j) & B[j] for j in range(4)]
    A_XOR_B = [byte_of_bits(A, j) ^ B[j] for j in range(4)]
    if bug == "maj_uses_b_not_c":
        C_AND_A_XOR_B = [B[j] & A_XOR_B[j] for j in range(4)]
    else:
        C_AND_A_XOR_B = [C[j] & A_XOR_B[j] for j in range(4)]
    E_AND_F = [byte_of_bits(E, j) & FF[j] for j in range(4)]
    if bug == "ch_no_not":
        NOT_E_AND_G = [byte_of_bits(E, j) & G[j] for j in range(4)]
    else:
        NOT_E_AND_G = [byte_of_bits(E, j, complement=True) & G[j] for j in range(4)]

    # --- ch and maj as the circuit sums them, sha256_round.rs:222-223 --------
    # Addition, not XOR: (e&f) and (!e&g) are bit-disjoint, and so are (a&b) and
    # (c&(a^b)) — if a&b is set then a=b so a^b is clear. Equality of + and ^
    # here is a property of SHA-256, and test_dataflow.py checks it exhaustively
    # over random words rather than taking it on faith.
    ch = bytes_to_word(E_AND_F) + bytes_to_word(NOT_E_AND_G)
    maj = bytes_to_word(A_AND_B) + bytes_to_word(C_AND_A_XOR_B)

    # --- Sigma0 / Sigma1 as expressions over the bit columns -----------------
    # sha256_common.rs `sigma(b, c, kind)`: kind 2 = Sigma0 (rotr 2,13,22),
    # kind 3 = Sigma1 (rotr 6,11,25). A rotation is an index permutation of the
    # bits; the three-way XOR is x+y+z-2(xy+xz+yz)+4xyz.
    if bug == "sigma0_off_by_one":
        S0 = sigma_from_bits(A, 2, rot_delta=1)   # rotr 3,13,22 instead of 2,13,22
    else:
        S0 = sigma_from_bits(A, 2)
    if bug == "sigma1_dropped":
        S1 = 0
    elif bug == "sigma1_is_sigma0":
        S1 = sigma_from_bits(E, 2)                # Sigma0's rotations on e
    else:
        S1 = sigma_from_bits(E, 3)

    # --- the two addition constraints, sha256_round.rs:227-235 ---------------
    temp1 = int(h) + S1 + ch + int(k_t) + int(w_t)
    temp2 = S0 + maj
    if bug == "drop_h_from_temp1":
        temp1 = S1 + ch + int(k_t) + int(w_t)

    total_a = temp1 + temp2
    OUT_A, CARRY_A = halves(total_a & MASK32), total_a >> 32
    total_e = int(d) + temp1
    OUT_E, CARRY_E = halves(total_e & MASK32), total_e >> 32
    if bug == "carry_dropped":
        CARRY_A = 0

    # The AIR reads the outputs back through `half(...)`, so mirror that.
    out_a = from_halves(OUT_A)
    out_e = from_halves(OUT_E)

    # --- the ShaRound send, sha256_round.rs:192-193 --------------------------
    # The next row's state is (out_a, a, b, c, out_e, e, f, g).
    return [out_a, a, b, c, out_e, e, f, g], {
        "carry_a": CARRY_A, "carry_e": CARRY_E, "ch": ch, "maj": maj,
        "S0": S0, "S1": S1, "temp1": temp1, "temp2": temp2,
    }


def sigma_from_bits(bits, kind, rot_delta=0):
    """Sigma computed the way the AIR computes it: a rotation is a re-indexing
    of `bits`, and each output bit is the three-way XOR polynomial. Mirrors
    sha256_common.rs `sigma`, including the degree-2 collapse where a shift
    contributes a zero bit."""
    ra, rb, rc, is_shift = {0: (7, 18, 3, 1), 1: (17, 19, 10, 1),
                            2: (2, 13, 22, 0), 3: (6, 11, 25, 0)}[kind]
    ra += rot_delta
    acc = 0
    for i in range(32):
        x = bits[(i + ra) % 32]
        y = bits[(i + rb) % 32]
        if is_shift and i + rc >= 32:
            bit = x + y - 2 * x * y
        else:
            z = bits[i + rc] if is_shift else bits[(i + rc) % 32]
            bit = x + y + z - 2 * (x * y + x * z + y * z) + 4 * x * y * z
        acc += bit << i
    return acc
