"""Shared model for the ECSM/ECDAS z3 gate.

Ground truth: prover/src/tables/ecsm.rs and prover/src/tables/ecdas.rs (constraint
bodies + bus sends). The SAME S_i builders below serve three purposes:
  1. exact interval bounds (L2 width audit)      — Interval leaves
  2. z3 constraint models (L1/L8/byte-level)     — z3 Int leaves
  3. concrete evaluation of real Rust witnesses  — int leaves (positive controls)
so a transcription error would be caught by the real-witness evaluation (purpose 3)
before any UNSAT is trusted.

Citations per relation are in each builder's docstring.
"""

import json
from fractions import Fraction

# ── Constants (independently recomputed; cross-checked vs crypto/ecsm/src/lib.rs
#    by the oracle's check_constants.py — see thoughts/ec-recover-opt/oracle/) ──

P = 2**256 - 2**32 - 977  # secp256k1 base field prime
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141  # curve order
PG = 2**64 - 2**32 + 1  # Goldilocks prime (constraints are enforced mod PG)
R3P = 3 * P  # the µ-gated offset in all three ECDAS relations (R_BYTES == 3p)
B = 7

P_BYTES = list(P.to_bytes(32, "little"))
N_BYTES = list(N.to_bytes(32, "little"))
R_BYTES = list(R3P.to_bytes(33, "little"))

# Carry offsets: ecsm.rs:27-28, ecdas.rs:24-26.
OFF = {
    "ecsm_x2": 8160,
    "ecsm_yg": 16319,
    "ecdas_lambda": 32636,
    "ecdas_xr": 8161,
    "ecdas_yr": 16320,
}

GEN_X = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
GEN_Y = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8


def compose(bytes_seq):
    """Value of a little-endian byte/limb sequence: Σ 256^j · b_j."""
    v = 0
    for j, b in enumerate(reversed(list(bytes_seq))):
        v = v * 256 + b
    return v


def decompose(v, n):
    """n little-endian byte limbs of v (must fit)."""
    assert 0 <= v < 256**n
    return [(v >> (8 * j)) & 0xFF for j in range(n)]


# ── Interval arithmetic (exact for the multilinear-plus-signed-square forms
#    of every S_i: all extremes are attained at byte corners 0/255, and one
#    consistent corner assignment attains every monomial extreme at once —
#    see RESULTS.md L2 method note) ──


class Iv:
    __slots__ = ("lo", "hi")

    def __init__(self, lo, hi=None):
        self.lo = lo
        self.hi = lo if hi is None else hi
        assert self.lo <= self.hi

    @staticmethod
    def of(x):
        return x if isinstance(x, Iv) else Iv(x, x)

    def __add__(self, o):
        o = Iv.of(o)
        return Iv(self.lo + o.lo, self.hi + o.hi)

    __radd__ = __add__

    def __neg__(self):
        return Iv(-self.hi, -self.lo)

    def __sub__(self, o):
        return self + (-Iv.of(o))

    def __rsub__(self, o):
        return Iv.of(o) + (-self)

    def __mul__(self, o):
        o = Iv.of(o)
        cs = [self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi]
        return Iv(min(cs), max(cs))

    __rmul__ = __mul__

    def __repr__(self):
        return f"[{self.lo}, {self.hi}]"


BYTE = Iv(0, 255)
BIT = Iv(0, 1)


# ── S_i builders. `v` maps operand name -> indexable of leaves (int / z3 Int / Iv).
#    Leaves must support +, -, * with python ints. Zero-padding past each
#    operand's length mirrors byte_at (ecsm.rs:710-721, ecdas.rs:302-314). ──


def _at(arr, length, j):
    return arr[j] if j < length else 0


def s_ecsm_x2(v, i):
    """ECSM X2 relation S_i (ecsm.rs:731-739): Σ xG_j·xG_{i-j} − x2_i − Σ q0_j·P_{i-j}."""
    s = 0
    for j in range(i + 1):
        s = s + _at(v["xg"], 32, j) * _at(v["xg"], 32, i - j)
        s = s - _at(v["q0"], 32, j) * (P_BYTES[i - j] if i - j < 32 else 0)
    s = s - _at(v["x2"], 32, i)
    return s


def s_ecsm_yg(v, i, mu=1):
    """ECSM Yg relation S_i (ecsm.rs:740-759):
    Σ yG_j·yG_{i-j} + µ·Σ P_j·P_{i-j} − Σ x2_j·xG_{i-j} − Σ q1_j·P_{i-j} − µ·B·[i=0].
    q1 is 33 bytes (byte 32 is IS_BIT-constrained, ecsm.rs:876-879)."""
    s = 0
    p2 = 0
    for j in range(i + 1):
        pj = P_BYTES[j] if j < 32 else 0
        pij = P_BYTES[i - j] if i - j < 32 else 0
        s = s + _at(v["yg"], 32, j) * _at(v["yg"], 32, i - j)
        p2 = p2 + pj * pij
        s = s - _at(v["x2"], 32, j) * _at(v["xg"], 32, i - j)
        s = s - _at(v["q1"], 33, j) * pij
    s = s + mu * p2
    if i == 0:
        s = s - mu * B
    return s


def _rq(v, i, qname, mu=1):
    """µ·Σ R_j·P_{i-j} − Σ q_j·P_{i-j} (ecdas.rs:321-334). R = 3p, 33 const bytes."""
    rp = 0
    qp = 0
    for j in range(i + 1):
        pij = P_BYTES[i - j] if i - j < 32 else 0
        rp += (R_BYTES[j] if j < 33 else 0) * pij
        qp = qp + _at(v[qname], 33, j) * pij
    return mu * rp - qp


def s_ecdas_lambda(v, i, op, mu=1, tamper=None):
    """ECDAS Lambda relation S_i (ecdas.rs:352-368):
    op·(Σ λ_j(xG−xA)_{i-j} + yA_i − yG_i) + (1−op)·(Σ 2λ_j·yA_{i-j} − 3xA_j·xA_{i-j}) + rq(Q0)."""
    ob = _at(v["ya"], 32, i) - _at(v["yg"], 32, i)
    for j in range(i + 1):
        ob = ob + _at(v["lam"], 32, j) * (_at(v["xg"], 32, i - j) - _at(v["xa"], 32, i - j))
    nb = 0
    for j in range(i + 1):
        nb = nb + 2 * _at(v["lam"], 32, j) * _at(v["ya"], 32, i - j)
        nb = nb - 3 * _at(v["xa"], 32, j) * _at(v["xa"], 32, i - j)
    return op * ob + (1 - op) * nb + _rq(v, i, "q0", mu)


def s_ecdas_xr(v, i, op, mu=1, tamper=None):
    """ECDAS Xr relation S_i (ecdas.rs:369-376):
    Σ λ_j·λ_{i-j} − xA_i − xG_i − xR_i − (1−op)(xA_i − xG_i) + rq(Q1)."""
    s = 0
    for j in range(i + 1):
        s = s + _at(v["lam"], 32, j) * _at(v["lam"], 32, i - j)
    s = s - _at(v["xa"], 32, i) - _at(v["xg"], 32, i) - _at(v["xr"], 32, i)
    s = s - (1 - op) * (_at(v["xa"], 32, i) - _at(v["xg"], 32, i))
    return s + _rq(v, i, "q1", mu)


def s_ecdas_yr(v, i, op=None, mu=1, tamper=None):
    """ECDAS Yr relation S_i (ecdas.rs:377-384):
    Σ λ_j(xA−xR)_{i-j} − yA_i − yR_i + rq(Q2).
    tamper='swap_xa_xg' replaces xA by xG (transcription-sensitivity control)."""
    xa = v["xg"] if tamper == "swap_xa_xg" else v["xa"]
    s = 0
    for j in range(i + 1):
        s = s + _at(v["lam"], 32, j) * (_at(xa, 32, i - j) - _at(v["xr"], 32, i - j))
    s = s - _at(v["ya"], 32, i) - _at(v["yr"], 32, i)
    return s + _rq(v, i, "q2", mu)


def conv_carry(c, s_i, i):
    """256·c_i − c_{i-1} − S_i (ecsm.rs:764-781, ecdas.rs:388-407); c_{-1}=0."""
    prev = c[i - 1] if i > 0 else 0
    return 256 * c[i] - prev - s_i


# ── Value-level integer relations (what L1+L2+L3 prove the byte constraints
#    equivalent to, µ=1 rows). All are EXACT integer equations. ──


def val_relations_ecdas(op, lam, xa, ya, xg, yg, xr, yr, q0, q1, q2):
    """The three ECDAS step identities over ℤ (µ=1)."""
    lam_rel = (
        op * (lam * (xg - xa) + ya - yg)
        + (1 - op) * (2 * lam * ya - 3 * xa * xa)
        + R3P * P
        - q0 * P
    )
    xr_rel = lam * lam - xa - xg - xr - (1 - op) * (xa - xg) + R3P * P - q1 * P
    yr_rel = lam * (xa - xr) - ya - yr + R3P * P - q2 * P
    return lam_rel, xr_rel, yr_rel


def val_relations_ecsm(xg, yg, x2, q0, q1):
    """ECSM curve-membership identities over ℤ (µ=1)."""
    x2_rel = xg * xg - x2 - q0 * P
    yg_rel = yg * yg + P * P - x2 * xg - B - q1 * P
    return x2_rel, yg_rel


# ── Real-witness loading (harness `witness` command output) ──


def load_witness_json(line):
    """Parses one `witness_json {...}` line into ints/lists (bytes stay as lists)."""
    assert line.startswith("witness_json ")
    w = json.loads(line[len("witness_json "):])

    def bl(h):
        return list(bytes.fromhex(h))

    for k in ["x_g", "y_g", "k", "x2", "q0", "q1", "x_g_sub_p", "k_sub_n",
              "x_r_sub_p", "x_r", "y_r"]:
        w[k] = bl(w[k])
    for st in w["steps"]:
        for k in ["x_a", "y_a", "x_g", "y_g", "lambda", "x_r", "y_r", "q0", "q1", "q2"]:
            st[k] = bl(st[k])
    return w


# ── Reference step (chord/tangent over F_p) — used for expected values only;
#    the independent reference remains the oracle's ec_ref.py. ──


def ref_step(op, xa, ya, xg, yg):
    if op == 1:
        lam = ((yg - ya) * pow(xg - xa, P - 2, P)) % P
    else:
        lam = (3 * xa * xa * pow(2 * ya, P - 2, P)) % P
    xr = (lam * lam - xa - xg - (1 - op) * (xa - xg)) % P
    # NB (1-op)(xa-xg) mirrors the chip's Xr relation: op=0 → λ²−2xA, op=1 → λ²−xA−xG.
    yr = (lam * (xa - xr) - ya) % P
    return lam, xr, yr
