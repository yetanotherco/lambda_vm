"""
Independent SHA-256 reference, built from the FIPS 180-4 spec DEFINITIONS
(not by copying the circuit or the repo's constant tables).

  - K round constants generated from their definition: the first 32 bits of the
    fractional parts of the CUBE roots of the first 64 primes (FIPS 180-4 §4.2.2).
  - H initial hash generated from its definition: the first 32 bits of the
    fractional parts of the SQUARE roots of the first 8 primes (§5.3.3).
  - Ch, Maj, Sigma0/1, sigma0/1, the message schedule and the compression loop
    implemented per §4.1.2 and §6.2.

Validation anchors (see test_ref.py):
  - The full hash is checked against Python's hashlib (an independent
    implementation) on the NIST vectors and on random/structured inputs.
  - K and H are separately cross-checked against the repo's
    executor/src/sha256.rs `K` and `IV`.

Word indexing matches the circuit: state[0..8] = (a, b, c, d, e, f, g, h),
big-endian bytes on the syscall boundary, little-endian bit order inside the
chips (bit i of the column group is bit i of the word).
"""

MASK32 = (1 << 32) - 1


def rotr32(v, r):
    r &= 31
    if r == 0:
        return v & MASK32
    return ((v >> r) | (v << (32 - r))) & MASK32


def shr32(v, r):
    return (v & MASK32) >> r


# --- constants from their FIPS 180-4 definitions ------------------------------
def _primes(n):
    out, cand = [], 2
    while len(out) < n:
        if all(cand % p for p in out if p * p <= cand):
            out.append(cand)
        cand += 1
    return out


def _frac_bits(x, root):
    """First 32 bits of the fractional part of x**(1/root), by integer arithmetic
    so the result does not depend on float precision."""
    # find integer part of x**(1/root)
    i = 1
    while (i + 1) ** root <= x:
        i += 1
    # frac = x**(1/root) - i ; we want floor(frac * 2**32).
    # Equivalently the largest f with (i + f/2**32)**root <= x, found by bisection.
    lo, hi = 0, 1 << 32
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if ((i << 32) + mid) ** root <= x << (32 * root):
            lo = mid
        else:
            hi = mid
    return lo


def gen_k():
    return [_frac_bits(p, 3) for p in _primes(64)]


def gen_h():
    return [_frac_bits(p, 2) for p in _primes(8)]


K = gen_k()
H0 = gen_h()


# --- the round functions, FIPS 180-4 §4.1.2 -----------------------------------
def ch(x, y, z):
    return ((x & y) ^ ((~x & MASK32) & z)) & MASK32


def maj(x, y, z):
    return (x & y) ^ (x & z) ^ (y & z)


def big_sigma0(x):
    return rotr32(x, 2) ^ rotr32(x, 13) ^ rotr32(x, 22)


def big_sigma1(x):
    return rotr32(x, 6) ^ rotr32(x, 11) ^ rotr32(x, 25)


def small_sigma0(x):
    return rotr32(x, 7) ^ rotr32(x, 18) ^ shr32(x, 3)


def small_sigma1(x):
    return rotr32(x, 17) ^ rotr32(x, 19) ^ shr32(x, 10)


# --- message schedule, §6.2.2 step 1 ------------------------------------------
def schedule(block):
    """block: 64 bytes -> w[0..64]"""
    assert len(block) == 64
    w = [int.from_bytes(block[4 * i:4 * i + 4], "big") for i in range(16)]
    for i in range(16, 64):
        w.append((small_sigma1(w[i - 2]) + w[i - 7] + small_sigma0(w[i - 15]) + w[i - 16]) & MASK32)
    return w


# --- one round of the compression loop, §6.2.2 step 3 -------------------------
def round_fn(state, w_t, k_t):
    """state = (a,b,c,d,e,f,g,h) -> next state"""
    a, b, c, d, e, f, g, h = state
    t1 = (h + big_sigma1(e) + ch(e, f, g) + k_t + w_t) & MASK32
    t2 = (big_sigma0(a) + maj(a, b, c)) & MASK32
    return [(t1 + t2) & MASK32, a, b, c, (d + t1) & MASK32, e, f, g]


# --- the compression function (what the accelerator's ECALL computes) ---------
def compress(state, block):
    """state: 8 words in, 8 words out. The feed-forward is included, matching
    the syscall: the ECALL takes the state and one 64-byte chunk and returns the
    updated state."""
    w = schedule(block)
    s = list(state)
    for t in range(64):
        s = round_fn(s, w[t], K[t])
    return [(x + y) & MASK32 for x, y in zip(state, s)]


# --- full SHA-256 with padding, only to anchor against hashlib ----------------
def sha256(msg):
    bit_len = len(msg) * 8
    padded = msg + b"\x80"
    while len(padded) % 64 != 56:
        padded += b"\x00"
    padded += bit_len.to_bytes(8, "big")
    state = list(H0)
    for i in range(0, len(padded), 64):
        state = compress(state, padded[i:i + 64])
    return b"".join(x.to_bytes(4, "big") for x in state)


if __name__ == "__main__":
    import hashlib
    print("K[0..4] :", [hex(x) for x in K[:4]])
    print("H0      :", [hex(x) for x in H0])
    print("abc     :", sha256(b"abc").hex())
    print("hashlib :", hashlib.sha256(b"abc").hexdigest())
