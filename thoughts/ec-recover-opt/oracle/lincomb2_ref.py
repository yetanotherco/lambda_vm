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
