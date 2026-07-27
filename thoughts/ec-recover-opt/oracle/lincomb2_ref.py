"""Independent lincomb2 + NUMS-T0 reference, layered on ec_ref.py.

Written from the group law only (no k256, no repo transcription). Used to
validate the Rust `ecsm::lincomb2_witness` and to pin the T0 blinding point.
"""

import hashlib
from ec_ref import P, N, B, GX, GY, pt_add, pt_double, pt_neg, scalar_mul, recover_even_y

T0_TAG = b"lambdavm/ecsm/lincomb2/T0/v1"


def t0_ref():
    """NUMS blinding point via try-and-increment:
    x = SHA-256(tag || counter_be32) as a big-endian int; accept the first
    counter giving x < P that is a valid curve x; pick the EVEN y.
    Returns ((x, y), counter)."""
    counter = 0
    while True:
        h = hashlib.sha256(T0_TAG + counter.to_bytes(4, "big")).digest()
        x = int.from_bytes(h, "big")
        if x < P:
            y = recover_even_y(x)
            if y is not None:
                return (x, y), counter
        counter += 1


def lincomb2(u1, P1, u2, P2):
    """Q = u1*P1 + u2*P2 by independent scalar-mul + add. Returns Q or None
    (None = Q is the point at infinity: the degenerate output case)."""
    assert 1 <= u1 < N and 1 <= u2 < N
    A = scalar_mul(u1, P1)
    Bp = scalar_mul(u2, P2)
    if A[0] == Bp[0]:
        if A[1] == Bp[1]:
            return pt_double(A)
        return None  # A = -B -> infinity
    return pt_add(A, Bp)


def lincomb2_blinded_trace(u1, P1, u2, P2, T0):
    """Mirror of the witness algorithm: NUMS-blinded joint Shamir/Straus.
    Returns (Q, len, n_double_rows, n_add_rows) and asserts the blinded
    accumulator equals Q + 2^len*T0 before correction. Independent check that
    the row schedule the Rust builder emits actually computes Q."""
    assert 1 <= u1 < N and 1 <= u2 < N
    P12 = None if P1[0] == P2[0] else pt_add(P1, P2)  # None => degenerate P1=+-P2
    m = max(u1.bit_length(), u2.bit_length())  # = max_msb + 1
    length = m
    acc = T0
    n_dbl = 0
    n_add = 0
    for r in range(length - 1, -1, -1):
        acc = pt_double(acc)
        n_dbl += 1
        d1 = (u1 >> r) & 1
        d2 = (u2 >> r) & 1
        if d1 and d2:
            addend = P12
        elif d1:
            addend = P1
        elif d2:
            addend = P2
        else:
            addend = None
        if addend is not None:
            acc = pt_add(acc, addend)
            n_add += 1
    # 2^len * T0
    tpow = T0
    for _ in range(length):
        tpow = pt_double(tpow)
    Q = lincomb2(u1, P1, u2, P2)
    if Q is not None:
        # acc should equal Q + 2^len*T0
        assert acc == pt_add(Q, tpow), "blinded accumulator != Q + 2^len*T0"
        # correction: acc + (-tpow) = Q
        assert pt_add(acc, pt_neg(tpow)) == Q, "correction did not land on Q"
    return Q, length, n_dbl, n_add


# ── full row emission, mirroring the Rust `lincomb2_witness` row list ────────

# Row roles, matching `witness.rs::JointSel` by name so gate/chip work can key
# off the same strings.
SEL_DOUBLE = "Double"
SEL_ADD_P1 = "AddP1"
SEL_ADD_P2 = "AddP2"
SEL_ADD_P12 = "AddP12"
SEL_PRECOMPUTE = "Precompute"
SEL_CORRECTION = "Correction"


def _slope_add(a, g):
    return ((g[1] - a[1]) * pow((g[0] - a[0]) % P, P - 2, P)) % P


def _slope_dbl(a):
    return (3 * a[0] * a[0] * pow((2 * a[1]) % P, P - 2, P)) % P


def lincomb2_rows(u1, P1, u2, P2, T0):
    """Emit the FULL joint-chain row list, in the order and with the field
    values `ecsm::lincomb2_witness` emits.

    Returns `(Q, length, rows)` where each row is a dict
    `{sel, round, op, d1, d2, nb, a, addend, r, lam}` with affine `(x, y)`
    tuples. Raises `ValueError` for the degenerate cases the Rust side reports
    as `Lincomb2Error` (`SumDegenerate`, `ResultInfinity`).

    `d1`/`d2` are the two scalars' bits at the row's round, carried on BOTH the
    double and the add of that round (the double needs them for its per-stream
    `Bit` sends and to derive `nb`; the add needs them to select the addend),
    and zero on the precompute/correction rows, which belong to no round.

    `nb = d1 | d2` on a double row and 0 everywhere else: "an add follows me at
    this same round". It is what makes the successor round `round - 1 + nb` a
    function of the row's own columns, so a prover cannot stall `round` by
    inserting or dropping doublings.

    Row order (matches `witness.rs`):
      1 x Precompute (a = P1, addend = P2, OFF the accumulator line)
      then for round = length-1 .. 0: one Double, plus one Add iff the joint
      digit is nonzero
      1 x Correction (a = last accumulator, addend = -2^length * T0)

    This is the ground truth the phase-E L7 unrollings compare against.
    """
    assert 1 <= u1 < N and 1 <= u2 < N
    if P1[0] == P2[0]:
        raise ValueError("SumDegenerate")
    rows = []

    P12 = pt_add(P1, P2)
    rows.append(dict(sel=SEL_PRECOMPUTE, round=0, op=1, d1=0, d2=0, nb=0,
                     a=P1, addend=P2, r=P12, lam=_slope_add(P1, P2)))

    length = max(u1.bit_length(), u2.bit_length())
    acc = T0
    for r in range(length - 1, -1, -1):
        d1 = (u1 >> r) & 1
        d2 = (u2 >> r) & 1
        lam = _slope_dbl(acc)
        nxt = pt_double(acc)
        rows.append(dict(sel=SEL_DOUBLE, round=r, op=0, d1=d1, d2=d2,
                         nb=d1 | d2, a=acc, addend=(0, 0), r=nxt, lam=lam))
        acc = nxt

        if not (d1 or d2):
            continue
        if d1 and d2:
            addend, sel = P12, SEL_ADD_P12
        elif d1:
            addend, sel = P1, SEL_ADD_P1
        else:
            addend, sel = P2, SEL_ADD_P2
        if acc[0] == addend[0]:
            raise ValueError("ResultInfinity")  # dlog event under blinding
        lam = _slope_add(acc, addend)
        nxt = pt_add(acc, addend)
        rows.append(dict(sel=sel, round=r, op=1, d1=d1, d2=d2, nb=0,
                         a=acc, addend=addend, r=nxt, lam=lam))
        acc = nxt

    tpow = T0
    for _ in range(length):
        tpow = pt_double(tpow)
    neg_tpow = pt_neg(tpow)
    if acc[0] == neg_tpow[0]:
        raise ValueError("ResultInfinity")
    lam = _slope_add(acc, neg_tpow)
    Q = pt_add(acc, neg_tpow)
    rows.append(dict(sel=SEL_CORRECTION, round=0, op=1, d1=0, d2=0, nb=0,
                     a=acc, addend=neg_tpow, r=Q, lam=lam))
    return Q, length, rows
