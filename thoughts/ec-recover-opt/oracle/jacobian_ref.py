"""Second independent secp256k1 implementation — Jacobian coordinates.

Deliberately a DIFFERENT CODE PATH from `ec_ref.py`, not a refactor of it:

| | `ec_ref.py` | this file |
|---|---|---|
| coordinates | affine, one inversion per step | Jacobian `(X:Y:Z)`, one inversion at the very end |
| infinity | cannot be represented (asserts) | represented as `Z = 0`, handled by the formulas |
| doubling | `lam = 3x^2/(2y)` | `dbl-2009-l` (a = 0), inversion-free |
| addition | `lam = (y2-y1)/(x2-x1)` | `add-2007-bl`, inversion-free, equal-`x` handled |
| scalar mul | MSB-first double-and-add | **LSB-first** (right-to-left) double-and-add |
| lincomb | n/a | two independent scalar muls, then one add |

The only thing shared with `ec_ref.py` is the SEC2 curve constant `p` (and the
group order `N` for input validation) — both re-stated here from the standard,
so the two files agree on the curve but on nothing else. Formulas are the
standard EFD (Explicit-Formulas Database) short-Weierstrass `a = 0` ones.

Used by `lincomb2_anchors.py` as the independent side of the lincomb2
differential, and by `anchor_a_wycheproof.py`'s ECDSA-verify anchor.
"""

# ── SEC2 published constants (restated, not imported) ───────────────────────
P = 2**256 - 2**32 - 977
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
B = 7
GX = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
GY = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8

# The point at infinity in Jacobian form. Any Z = 0 triple is infinity; this is
# the canonical one produced by the formulas.
INF = (0, 0, 0)


def is_inf(pt):
    return pt[2] % P == 0


def to_jac(affine):
    """(x, y) -> (X:Y:1). `None` means infinity."""
    if affine is None:
        return INF
    x, y = affine
    return (x % P, y % P, 1)


def to_affine(pt):
    """(X:Y:Z) -> (x, y), or None for infinity. One inversion, at the end."""
    if is_inf(pt):
        return None
    x, y, z = pt
    zi = pow(z, P - 2, P)
    zi2 = (zi * zi) % P
    return ((x * zi2) % P, (y * zi2 % P * zi) % P)


def jac_double(pt):
    """EFD `dbl-2009-l` for a = 0. Returns infinity for y = 0 and for infinity."""
    x, y, z = pt
    if z % P == 0 or y % P == 0:
        return INF
    a = (x * x) % P
    b = (y * y) % P
    c = (b * b) % P
    d = (2 * (((x + b) * (x + b) - a - c) % P)) % P
    e = (3 * a) % P
    f = (e * e) % P
    x3 = (f - 2 * d) % P
    y3 = (e * (d - x3) - 8 * c) % P
    z3 = (2 * y * z) % P
    return (x3, y3, z3)


def jac_add(p1, p2):
    """EFD `add-2007-bl`, with the equal-input cases handled explicitly.

    Complete for our purposes: returns infinity when the inputs are inverses,
    and delegates to `jac_double` when they are the same point.
    """
    if is_inf(p1):
        return p2
    if is_inf(p2):
        return p1
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    z1z1 = (z1 * z1) % P
    z2z2 = (z2 * z2) % P
    u1 = (x1 * z2z2) % P
    u2 = (x2 * z1z1) % P
    s1 = (y1 * z2 % P * z2z2) % P
    s2 = (y2 * z1 % P * z1z1) % P
    if u1 == u2:
        if s1 != s2:
            return INF  # P2 = -P1
        return jac_double(p1)  # P2 = P1
    h = (u2 - u1) % P
    i = (4 * h * h) % P
    j = (h * i) % P
    r = (2 * (s2 - s1)) % P
    v = (u1 * i) % P
    x3 = (r * r - j - 2 * v) % P
    y3 = (r * (v - x3) - 2 * s1 * j) % P
    z3 = ((((z1 + z2) * (z1 + z2)) % P - z1z1 - z2z2) * h) % P
    return (x3, y3, z3)


def jac_neg(pt):
    x, y, z = pt
    return (x, (-y) % P, z)


def jac_mul(k, pt):
    """k·pt, LSB-first (right-to-left) double-and-add.

    Opposite scan direction to `ec_ref.scalar_mul`, and it accumulates into a
    running "addend doubling" register rather than into the point itself, so a
    bug in either loop cannot cancel against the other. Accepts k = 0 (returns
    infinity) and any k >= 0; no range assertion, so out-of-range scalars are
    the caller's business.
    """
    acc = INF
    addend = pt
    while k:
        if k & 1:
            acc = jac_add(acc, addend)
        addend = jac_double(addend)
        k >>= 1
    return acc


def on_curve_affine(affine):
    if affine is None:
        return True
    x, y = affine
    if not (0 <= x < P and 0 <= y < P):
        return False
    return (y * y - x * x % P * x - B) % P == 0


def lincomb2(u1, pt1, u2, pt2):
    """Q = u1·pt1 + u2·pt2 over affine inputs; returns affine Q or None (∞).

    Two independent scalar muls plus one group add — structurally as far from
    the joint/interleaved Shamir–Straus chain the chip proves as it gets, which
    is exactly the point of a differential.
    """
    q = jac_add(jac_mul(u1, to_jac(pt1)), jac_mul(u2, to_jac(pt2)))
    return to_affine(q)


def ecdsa_verify(pubkey, z, r, s):
    """Textbook ECDSA verification, on the Jacobian path.

    `pubkey` is affine (x, y), `z` the already-truncated message integer.
    Returns True iff the signature is valid. This is the `u1·G + u2·PK` lincomb
    shape the Wycheproof ECDSA vectors exercise.
    """
    if not (1 <= r < N and 1 <= s < N):
        return False
    if not on_curve_affine(pubkey) or pubkey is None:
        return False
    w = pow(s, N - 2, N)
    u1 = (z * w) % N
    u2 = (r * w) % N
    q = jac_add(jac_mul(u1, to_jac((GX, GY))), jac_mul(u2, to_jac(pubkey)))
    if is_inf(q):
        return False
    x, _ = to_affine(q)
    return (x % N) == r
