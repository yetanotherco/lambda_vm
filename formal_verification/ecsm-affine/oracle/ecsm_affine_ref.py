"""Independent reference for the ECSM **affine** ecall (`ECSM_AFFINE_SYSCALL_NUMBER`).

Written from the curve definition, not from the repo: no `k256`, no `ecsm` crate, no
`num_bigint`. Group law is plain textbook chord/tangent over `F_p` with Python ints, so a
transcription error in `crypto/ecsm` cannot hide behind a shared implementation.

What it defines (the contract the gate's UNSATs are *about*):

  x_only_mul(k, xG)          → x(k·P), P = the canonical EVEN lift of xG   (pre-existing ecall)
  affine_mul(k, xG, yG)      → (x, y) of k·(xG, yG)                        (the NEW ecall)

and the ABI/validation predicates the executor applies before either
(`executor/src/vm/instruction/execution.rs`, `SyscallNumbers::EcsmAffine` arm):

  addr_limb_ok(addr, span)   → the low-limb no-straddle test
  operands_disjoint(...)     → the xG‖yG vs k overlap guard
  validate_affine(...)       → 0 < k < N, xG < p, yG < p, (xG,yG) on curve

The affine result is **root-dependent** by construction — that is the whole point of the
variant and the reason the AIR has to pin `yG` to the caller's buffer:

    affine_mul(k, x, p - y) == (X, p - Y)  where (X, Y) == affine_mul(k, x, y)

so publishing `yR` makes the input parity observable, while `x_only_mul` cannot see it.

Citations to the code being modelled are inline, `file:line` against the branch
`verify/ecsm-affine-selector` (head of PR #879 plus this campaign).
"""

# ── secp256k1 (SEC 2 v2 §2.4.1). Recomputed here, cross-checked against
#    crypto/ecsm/src/lib.rs by test_oracle.py's anchor A0. ──

P = 2**256 - 2**32 - 977
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
A = 0
B = 7
GX = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
GY = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
G = (GX, GY)

# `2^256 - p` — the width of the non-canonical band a 32-byte y can occupy, i.e. the
# largest `y` for which `y + p` is still 32-byte representable. This is the number the
# `YrLtP` range check exists to exclude; see oracle/small_y_point.py, which constructs a
# real curve point inside the band.
NONCANONICAL_BAND = 2**256 - P  # == 2**32 + 977


# ── field ──

def inv(a):
    """1/a mod p. Fermat; p is prime (SEC2 / A-PRIME)."""
    a %= P
    if a == 0:
        raise ZeroDivisionError("no inverse of 0 mod p")
    return pow(a, P - 2, P)


def is_square(a):
    return a == 0 or pow(a % P, (P - 1) // 2, P) == 1


def sqrt_mod_p(a):
    """A square root of `a` mod p, or None. p ≡ 3 (mod 4), so it is a^((p+1)/4)."""
    a %= P
    if not is_square(a):
        return None
    r = pow(a, (P + 1) // 4, P)
    assert (r * r - a) % P == 0
    return r


# ── group law (affine chord/tangent; O is None) ──

def is_on_curve(pt):
    if pt is None:
        return True
    x, y = pt
    if not (0 <= x < P and 0 <= y < P):
        return False
    return (y * y - (x * x % P * x + A * x + B)) % P == 0


def neg(pt):
    if pt is None:
        return None
    x, y = pt
    return (x, (-y) % P)


def add(p1, p2):
    if p1 is None:
        return p2
    if p2 is None:
        return p1
    x1, y1 = p1
    x2, y2 = p2
    if x1 == x2:
        if (y1 + y2) % P == 0:
            return None  # P + (−P) = O
        lam = (3 * x1 * x1 + A) * inv(2 * y1) % P  # tangent
    else:
        lam = (y2 - y1) * inv(x2 - x1) % P  # chord
    x3 = (lam * lam - x1 - x2) % P
    y3 = (lam * (x1 - x3) - y1) % P
    return (x3, y3)


def mul(k, pt):
    """k·pt by double-and-add from the MSB down — the same schedule the chip proves
    (`crypto/ecsm/src/curve.rs::schedule`), so a schedule bug shows up as a mismatch
    rather than being reproduced."""
    if k % N == 0 or pt is None:
        return None
    k %= N
    acc = None
    for bit in reversed(range(k.bit_length())):
        acc = add(acc, acc)
        if (k >> bit) & 1:
            acc = add(acc, pt)
    return acc


# ── lifts ──

def recover_y_canonical(x):
    """The EVEN root of x³+b, or None. Mirrors `crypto/ecsm/src/curve.rs:recover_y_canonical`
    (`curve.rs:18-60` on this branch). Used only by the x-only path."""
    if not 0 <= x < P:
        return None
    y = sqrt_mod_p((x * x % P * x + B) % P)
    if y is None:
        return None
    return y if y % 2 == 0 else P - y


# ── the two ecall semantics ──

class EcsmError(Exception):
    """Mirrors `ecsm::EcsmError` (crypto/ecsm/src/lib.rs:70-100)."""


def validate_scalar(k):
    if k == 0:
        raise EcsmError("ScalarIsZero")
    if k >= N:
        raise EcsmError("ScalarOutOfRange")


def x_only_mul(k, xg):
    """Pre-existing ecall (`ECSM_SYSCALL_NUMBER`): x(k·P) with P the even lift of xG.

    Root-independent: x(k·P) == x(k·(−P)) for every k, which is exactly why the AIR is
    allowed to leave yG's parity free on this path (`curve.rs:18-31` comment)."""
    validate_scalar(k)
    if xg >= P:
        raise EcsmError("CoordinateOutOfRange")
    yg = recover_y_canonical(xg)
    if yg is None:
        raise EcsmError("NotOnCurve")
    r = mul(k, (xg, yg))
    assert r is not None, "k·P = O impossible for 0<k<N on a prime-order curve"
    return r[0]


def affine_mul(k, xg, yg):
    """NEW ecall (`ECSM_AFFINE_SYSCALL_NUMBER`): both coordinates of k·(xG, yG).

    Mirrors `ecsm::scalar_mul_xy_with_y` / `prepare_with_y`
    (crypto/ecsm/src/lib.rs:136-178). `yG` is the caller's own value — validated on
    curve and canonical, but NOT canonicalised to a parity."""
    validate_scalar(k)
    if xg >= P or yg >= P:
        raise EcsmError("CoordinateOutOfRange")
    if not is_on_curve((xg, yg)):
        raise EcsmError("NotOnCurve")
    r = mul(k, (xg, yg))
    assert r is not None
    return r


# ── executor ABI predicates (execution.rs, `SyscallNumbers::EcsmAffine` arm) ──

MASK64 = 2**64 - 1


def addr_limb_ok(addr, span):
    """`executor/src/vm/instruction/execution.rs::addr_limb_ok`: the operand's low 32-bit
    limb plus its span must not reach 2^32, so a multi-byte access cannot straddle the
    limb boundary while the AIR reuses the high limb unchanged."""
    return (addr % 2**32) + span < 2**32


def operands_disjoint(addr_xg, addr_k, point_bytes=64, scalar_bytes=32):
    """The affine overlap guard (`execution.rs`, EcsmAffine arm): the point buffer
    [addr_xg, +64) and the scalar [addr_k, +32) must not intersect.

    Computed in unbounded ints, which is what the branch's `u128` widening buys: the
    naive u64 form `addr_k < addr_xg + 64` wraps at `addr_xg = 2^64 - 64`, making the
    clause vacuously false and skipping the guard. That address PASSES
    `addr_limb_ok(·, 63)`, so the wrap is reachable — see gate/a4_addressing.py N-WRAP."""
    return not (addr_k < addr_xg + point_bytes and addr_xg < addr_k + scalar_bytes)


def operands_disjoint_u64_buggy(addr_xg, addr_k):
    """The pre-fix wrapping form, kept as the negative control's oracle."""
    lhs = (addr_xg + 64) & MASK64
    rhs = (addr_k + 32) & MASK64
    return not (addr_k < lhs and addr_xg < rhs)


# ── little-endian 32-byte codec (the ABI's wire form) ──

def to_le32(v):
    assert 0 <= v < 2**256
    return list(v.to_bytes(32, "little"))


def from_le32(bs):
    assert len(bs) == 32
    return int.from_bytes(bytes(bs), "little")
