"""
Independent Keccak-f[1600] reference, built from the FIPS-202 spec ALGORITHMS
(not by copying the circuit or the repo's constant tables).

  - RHO offsets generated from the FIPS-202 (x,y) walk with triangular offsets.
  - RC round constants generated from the FIPS-202 LFSR (rc(t)).
  - theta / rho / pi / chi / iota implemented per FIPS-202.

Validation anchors (see run at bottom / test_ref.py):
  - The permutation is wired into a SHA3-256 sponge and checked against
    Python's hashlib (an independent NIST implementation).
  - RHO/RC are separately cross-checked against the repo's KECCAK_RHO/KECCAK_RC.

Lane indexing matches the circuit: state[x + 5*y], x = column, y = row.
"""

MASK64 = (1 << 64) - 1


def rotl64(v, r):
    r &= 63
    if r == 0:
        return v & MASK64
    return ((v << r) | (v >> (64 - r))) & MASK64


# --- RHO offsets from the FIPS-202 recurrence (Algorithm 2, rho) ---------------
# Start at (x,y) = (1,0); for t = 0..23 the offset is (t+1)(t+2)/2 mod 64,
# then (x,y) <- (y, (2x+3y) mod 5). (0,0) keeps offset 0.
def gen_rho():
    rho = [[0] * 5 for _ in range(5)]  # rho[x][y]
    x, y = 1, 0
    for t in range(24):
        rho[x][y] = ((t + 1) * (t + 2) // 2) % 64
        x, y = y, (2 * x + 3 * y) % 5
    return rho


RHO = gen_rho()


# --- RC round constants from the FIPS-202 LFSR (Algorithm 5, rc) ---------------
def _rc_bit(t):
    t %= 255
    if t == 0:
        return 1
    R = 0b10000000  # register holding r0..r7, r0 = MSB per our shifting below
    # Use the standard byte-register formulation.
    R = 0x01
    for _ in range(t):
        R <<= 1
        if R & 0x100:
            R ^= 0x71  # x^8 + x^6 + x^5 + x^4 + 1  -> low byte feedback 0x71
        R &= 0xFF
    return R & 1


def gen_rc():
    rc = []
    for ir in range(24):
        w = 0
        for j in range(7):  # j = 0..6 -> bit positions 2^j - 1
            if _rc_bit(j + 7 * ir):
                w |= 1 << ((1 << j) - 1)
        rc.append(w & MASK64)
    return rc


RC = gen_rc()


# --- The permutation, per FIPS-202 -------------------------------------------
def keccak_round(state, rc):
    """One round of Keccak-f[1600]. `state` is list[25] of u64, state[x+5y]."""
    a = list(state)

    # theta
    C = [a[x] ^ a[x + 5] ^ a[x + 10] ^ a[x + 15] ^ a[x + 20] for x in range(5)]
    D = [C[(x + 4) % 5] ^ rotl64(C[(x + 1) % 5], 1) for x in range(5)]
    for x in range(5):
        for y in range(5):
            a[x + 5 * y] ^= D[x]

    # rho + pi:  B[X][Y] = rotl(A[(X+3Y)%5][X], RHO[(X+3Y)%5][X])
    B = [0] * 25
    for X in range(5):
        for Y in range(5):
            sx = (X + 3 * Y) % 5
            sy = X
            B[X + 5 * Y] = rotl64(a[sx + 5 * sy], RHO[sx][sy])

    # chi
    out = [0] * 25
    for x in range(5):
        for y in range(5):
            out[x + 5 * y] = B[x + 5 * y] ^ ((~B[(x + 1) % 5 + 5 * y] & MASK64) & B[(x + 2) % 5 + 5 * y])

    # iota
    out[0] ^= rc
    return out


def keccak_f1600(state):
    s = list(state)
    for r in range(24):
        s = keccak_round(s, RC[r])
    return s


# --- SHA3-256 sponge on top of the permutation (for external validation) ------
def sha3_256(msg: bytes) -> bytes:
    rate = 136  # bytes (1088 bits)
    # pad10*1 with SHA-3 domain separation 0x06
    m = bytearray(msg)
    m.append(0x06)
    while len(m) % rate != 0:
        m.append(0x00)
    m[-1] ^= 0x80

    state = [0] * 25
    for off in range(0, len(m), rate):
        block = m[off:off + rate]
        for i in range(rate // 8):
            lane = int.from_bytes(block[i * 8:i * 8 + 8], "little")
            state[i] ^= lane
        state = keccak_f1600(state)

    out = bytearray()
    while len(out) < 32:
        for i in range(rate // 8):
            out += state[i].to_bytes(8, "little")
            if len(out) >= 32:
                break
        if len(out) < 32:
            state = keccak_f1600(state)
    return bytes(out[:32])
