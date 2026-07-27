"""Independent secp256k1 reference for the ECSM/ECDAS oracle.

Written from the SEC2 curve definition and textbook affine Weierstrass
formulas ONLY. No k256, no transcription from lambda_vm repo code — the whole
point is independent lineage. Constants below are the published SEC2 values
(Certicom SEC2 v2.0, section 2.4.1).

Curve: y^2 = x^3 + 7 over GF(p), p = 2^256 - 2^32 - 977.
"""

# ── SEC2 published constants (from the standard, not from the repo) ─────────
P = 2**256 - 2**32 - 977
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
B = 7
GX = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
GY = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8

assert P % 4 == 3  # sqrt(a) = a^((p+1)/4) is valid


class EcError(Exception):
    """Mirrors the accelerator's rejection classes."""

    def __init__(self, kind):
        self.kind = kind  # 'ScalarIsZero' | 'ScalarOutOfRange' | 'CoordinateOutOfRange' | 'NotOnCurve'
        super().__init__(kind)


# ── field helpers ────────────────────────────────────────────────────────────

def finv(a):
    return pow(a, P - 2, P)


def fsqrt_even(a):
    """The EVEN square root of a mod p, or None if a is a non-residue."""
    r = pow(a, (P + 1) // 4, P)
    if (r * r) % P != a % P:
        return None
    return r if r % 2 == 0 else P - r


def recover_even_y(x):
    """Canonical (even) y with y^2 = x^3 + B, or None if x is not on the curve."""
    if not (0 <= x < P):
        return None
    return fsqrt_even((pow(x, 3, P) + B) % P)


# ── affine group law (never the point at infinity; callers guarantee it) ────

def pt_double(p1):
    x1, y1 = p1
    assert y1 != 0, "double of a 2-torsion point (impossible on secp256k1)"
    lam = (3 * x1 * x1 * finv(2 * y1)) % P
    x3 = (lam * lam - 2 * x1) % P
    y3 = (lam * (x1 - x3) - y1) % P
    return (x3, y3)


def pt_add(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    assert x1 != x2, "add with equal x (doubling or inverse) must not reach here"
    lam = ((y2 - y1) * finv(x2 - x1)) % P
    x3 = (lam * lam - x1 - x2) % P
    y3 = (lam * (x1 - x3) - y1) % P
    return (x3, y3)


def pt_neg(p1):
    x1, y1 = p1
    return (x1, (P - y1) % P)


def scalar_mul(k, pt):
    """k·pt by plain MSB-first double-and-add over affine points.

    Requires 1 <= k < N and pt on curve, not infinity. Because the group has
    prime order N and 1 <= k < N, no intermediate ever hits infinity and the
    add never sees equal-x operands (acc = ±pt would need k' ≡ ±1 with more
    bits pending, which the MSB-first schedule only allows transiently — the
    assertion in pt_add would fire if the claim were wrong, making this
    self-checking rather than silently incorrect).
    """
    assert 1 <= k < N
    bits = bin(k)[2:]
    acc = pt
    for b in bits[1:]:
        acc = pt_double(acc)
        if b == "1":
            acc = pt_add(acc, pt)
    return acc


# ── the precompile ABI mirror ────────────────────────────────────────────────

def x_only_mul_ints(x, k):
    """Integer-level core: x(k·P) for P = (x, even-y). Raises EcError exactly
    when the accelerator's contract rejects. Check order mirrors the contract:
    scalar checks, then coordinate range, then curve membership."""
    if k == 0:
        raise EcError("ScalarIsZero")
    if k >= N:
        raise EcError("ScalarOutOfRange")
    if x >= P:
        raise EcError("CoordinateOutOfRange")
    y = recover_even_y(x)
    if y is None:
        raise EcError("NotOnCurve")
    return scalar_mul(k, (x, y))[0]


def x_only_mul(x_le: bytes, k_le: bytes) -> bytes:
    """Byte-level ABI mirror of ecsm_mul: 32-byte little-endian in/out."""
    assert len(x_le) == 32 and len(k_le) == 32
    x = int.from_bytes(x_le, "little")
    k = int.from_bytes(k_le, "little")
    xr = x_only_mul_ints(x, k)
    return xr.to_bytes(32, "little")


# ── documented ECDAS schedule semantics (MSB-first double-and-add rows) ──────

def expected_schedule(k):
    """The row list ((round, op, next_op) per row) the ECDAS design documents:
    one double row per bit below the MSB, plus an add row when that bit is set.
    `round` is the bit index, op 0=double 1=add, next_op = op of the following
    row (0 for the last row). Derived here purely from the bit pattern of k —
    an independent statement of the documented algorithm, for comparison
    against the repo's schedule/replay.
    """
    assert 1 <= k < N
    bits = bin(k)[2:]
    m = len(bits) - 1  # msb position
    rows = []  # (round, op)
    for i, b in enumerate(bits[1:]):
        rnd = m - 1 - i
        rows.append((rnd, 0))
        if b == "1":
            rows.append((rnd, 1))
    out = []
    for j, (rnd, op) in enumerate(rows):
        nxt = rows[j + 1][1] if j + 1 < len(rows) else 0
        out.append((rnd, op, nxt))
    return out


def replay_schedule(k, g):
    """Execute expected_schedule step-by-step, returning per-row
    (round, op, next_op, a, lambda, r) with a=incoming accumulator, r=result.
    Lambda is the tangent/chord slope of that row's operation."""
    sched = expected_schedule(k)
    acc = g
    steps = []
    for (rnd, op, nxt) in sched:
        if op == 0:
            lam = (3 * acc[0] * acc[0] * finv(2 * acc[1])) % P
            r = pt_double(acc)
        else:
            lam = ((g[1] - acc[1]) * finv(g[0] - acc[0])) % P
            r = pt_add(acc, g)
        steps.append((rnd, op, nxt, acc, lam, r))
        acc = r
    return steps, acc


# ── ecrecover on top of the reference (for the end-to-end differential) ─────

def ninv(a):
    return pow(a, N - 2, N)


def ecrecover(msg_hash: bytes, v: int, r: int, s: int):
    """Recover the uncompressed pubkey (x, y) or None. v in {0,1} = parity of
    R.y. Implements pk = r^-1 (s·R - z·G) from the ECDSA definition. Rejects
    r,s outside [1, N) and unrecoverable R. Does not handle recid >= 2."""
    if not (1 <= r < N and 1 <= s < N):
        return None
    if r >= P:  # r is used directly as R.x here (recid>=2 not supported)
        return None
    y_even = recover_even_y(r)
    if y_even is None:
        return None
    ry = y_even if v == 0 else (P - y_even) % P
    R = (r, ry)
    z = int.from_bytes(msg_hash, "big") % N
    rinv = ninv(r)
    u1 = (-(rinv * z)) % N
    u2 = (rinv * s) % N
    # pk = u1·G + u2·R, handling the (never-for-valid-sigs) degenerate cases.
    if u1 == 0 and u2 == 0:
        return None
    if u1 == 0:
        return scalar_mul(u2, R)
    if u2 == 0:
        return scalar_mul(u1, (GX, GY))
    A = scalar_mul(u1, (GX, GY))
    Bp = scalar_mul(u2, R)
    if A[0] == Bp[0]:
        if A[1] == Bp[1]:
            return pt_double(A)
        return None  # A = -B: infinity
    return pt_add(A, Bp)
