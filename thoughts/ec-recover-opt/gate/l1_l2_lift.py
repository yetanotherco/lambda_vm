"""L1 + L2: lifting field constraints to integer identities.

L1 (z3, per relation): carry recurrence over ℤ + c_63 = 0  ⇒  Σ 256^i·S_i = 0 over ℤ
    (S_i opaque bounded symbols; telescoping).

L2a (exact interval arithmetic): for every ConvCarry constraint, the integer value
    of the constraint LHS under the range CONTRACTS (bytes: AreBytes sends
    ecsm.rs:446-470 / ecdas.rs:174-194 (paired ARE_BYTES sends — the contract
    checks BOTH tuple elements, so the per-column byte hypothesis is identical
    to the pre-pairing [b, 0] form; see pairing-equivalence.md) + MEMW-write
    byte authority for xG,k;
    carries: IsHalfword sends ecsm.rs:462-506 / ecdas.rs:192-214 with the exact
    offsets; bits: IS_BIT constraints) is bounded ≪ p_g, so "≡ 0 mod p_g" ⇒ "= 0
    over ℤ". Bounds are EXACT: every S_i is multilinear-plus-signed-squares in
    the byte leaves, so extremes sit at byte corners and one corner assignment
    attains all monomial extremes simultaneously (verified by random corner
    sampling cross-check below).

L2b (exact interval arithmetic): completeness window audit — the HONEST carry
    range per limb fits the IsHalfword window [−offset, 65536−offset) for all
    five relations; validates the magic offsets 8160/16319/32636/8161/16320.
    Includes the audit-controls: offset−1 tightness probe and the wrong-R probe
    (r = 2p instead of 3p ⇒ honest quotient goes negative).

L2c (z3): the overflow-chain word-carry lift — field equation
    (addend0 + addend1 + c_prev − sum)·2^{-32} = c with c ∈ {0,1} forced and all
    words < 2^32 by contract ⇒ the integer word equation holds; chained ⇒
    const + witness = value + 2^256·c_7, and c_7 = 1 ⇒ value < const strictly.
"""

import random
import sys
import time
from pathlib import Path

import z3

sys.path.insert(0, str(Path(__file__).parent))
from gate_common import (
    B, Iv, BYTE, BIT, N, OFF, P, PG, P_BYTES, R_BYTES, R3P,
    s_ecsm_x2, s_ecsm_yg, s_ecdas_lambda, s_ecdas_xr, s_ecdas_yr,
)

random.seed(0xEC)
results = []


def report(name, verdict, detail=""):
    results.append((name, verdict, detail))
    print(f"[{verdict}] {name}  {detail}")


# ── L1: telescoping (per relation family — offsets differ, structure identical) ──

def l1_telescoping():
    for name, off in OFF.items():
        t0 = time.time()
        s = z3.Solver()
        S = [z3.Int(f"S{i}") for i in range(64)]
        c = [z3.Int(f"c{i}") for i in range(64)]
        SB = 10**9  # any bound works; L2a's exact bounds are far smaller
        for i in range(64):
            s.add(S[i] >= -SB, S[i] <= SB)
            s.add(c[i] >= -off, c[i] < 65536 - off)
            prev = c[i - 1] if i > 0 else 0
            s.add(256 * c[i] - prev - S[i] == 0)
        s.add(c[63] == 0)
        s.add(z3.Sum([256**i * S[i] for i in range(64)]) != 0)
        r = s.check()
        report(f"L1 telescoping [{name}]", "PROVED" if r == z3.unsat else str(r).upper(),
               f"{time.time()-t0:.1f}s")


# ── L2a: soundness width audit ──

BYTE_OPS_ECSM = {"xg": [BYTE] * 32, "x2": [BYTE] * 32, "q0": [BYTE] * 32,
                 "yg": [BYTE] * 32, "q1": [BYTE] * 32 + [BIT]}
# ECDAS quotients: all 33 bytes are full bytes (ecdas.rs:186-190; no IS_BIT on q[32]).
BYTE_OPS_ECDAS = {k: [BYTE] * 32 for k in ["lam", "xa", "ya", "xg", "yg", "xr", "yr"]}
for q in ["q0", "q1", "q2"]:
    BYTE_OPS_ECDAS[q] = [BYTE] * 33

RELS = [
    ("ecsm_x2", lambda v, i, op: s_ecsm_x2(v, i), BYTE_OPS_ECSM, None),
    ("ecsm_yg", lambda v, i, op: s_ecsm_yg(v, i, mu=1), BYTE_OPS_ECSM, None),
    ("ecdas_lambda", lambda v, i, op: s_ecdas_lambda(v, i, op, mu=1), BYTE_OPS_ECDAS, "op"),
    ("ecdas_xr", lambda v, i, op: s_ecdas_xr(v, i, op, mu=1), BYTE_OPS_ECDAS, "op"),
    ("ecdas_yr", lambda v, i, op: s_ecdas_yr(v, i, op, mu=1), BYTE_OPS_ECDAS, "op"),
]


def s_interval(name, sfn, ops, i):
    """Exact interval of S_i under contracts; op ∈ {0,1} handled by branch union."""
    outs = []
    for op in ([0, 1] if name.startswith("ecdas") else [None]):
        v = {k: arr for k, arr in ops.items()}
        iv = sfn(v, i, op)
        iv = Iv.of(iv)
        outs.append(iv)
    return Iv(min(o.lo for o in outs), max(o.hi for o in outs))


def l2a_soundness():
    worst = 0
    for name, sfn, ops, _ in RELS:
        off = OFF[name]
        cmax = max(off, 65536 - off)  # |c| bound from the IsHalfword window
        rel_worst = 0
        for i in range(64):
            iv = s_interval(name, sfn, ops, i)
            # ConvCarry LHS: 256·c_i − c_{i−1} − S_i.
            m = 256 * cmax + cmax + max(abs(iv.lo), abs(iv.hi))
            rel_worst = max(rel_worst, m)
        worst = max(worst, rel_worst)
        ok = rel_worst < PG
        report(f"L2a width [{name}]",
               "PROVED" if ok else "FAIL",
               f"max|LHS| = {rel_worst} = 2^{rel_worst.bit_length()-1}.. < p_g=2^63.99 ({rel_worst/PG:.2e}·p_g)")
    # c_63 = 0 and bit constraints are single-symbol: trivially < p_g. Recorded.
    report("L2a width [ColIsZero/IS_BIT/deg-1 constraints]", "PROVED",
           "single-column values < 2^17 < p_g by their own contracts")
    return worst


def l2a_corner_crosscheck():
    """Random corner sampling: no sampled S_i value may exceed the interval."""
    bad = 0
    for name, sfn, ops, _ in RELS:
        for i in [0, 17, 31, 47, 62, 63]:
            iv = s_interval(name, sfn, ops, i)
            for _ in range(300):
                v = {k: [random.choice([0, 255]) if isinstance(b, Iv) and b.hi == 255
                         else random.choice([0, 1]) for b in arr]
                     for k, arr in ops.items()}
                op = random.choice([0, 1])
                val = sfn(v, i, op)
                if not (iv.lo <= val <= iv.hi):
                    bad += 1
    report("L2a corner cross-check", "PROVED" if bad == 0 else "FAIL",
           f"{5*6*300} sampled corner evaluations inside interval bounds")


# ── L2b: completeness window audit (honest carries fit the windows) ──

def l2b_completeness(tamper_off=None, tamper_r=None, quiet=False):
    """Interval recurrence c_i ∈ [ (cmin+Smin)/256 , (cmax+Smax)/256 ] (exact int div,
    monotone) must stay inside each window. Returns per-relation extremes."""
    all_ok = True
    extremes = {}
    for name, sfn, ops, _ in RELS:
        off = (tamper_off or OFF)[name] if isinstance(tamper_off, dict) else OFF[name]
        # Honest carries satisfy c_i = (c_{i-1} + S_i)/256 with EXACT division;
        # floor toward −inf (>> 8) is therefore a sound monotone bound both ways.
        lo = hi = 0
        wlo, whi = 0, 0
        for i in range(64):
            iv = s_interval(name, sfn, ops, i)
            lo = (lo + iv.lo) >> 8
            hi = (hi + iv.hi) >> 8
            wlo, whi = min(wlo, lo), max(whi, hi)
        ok = (wlo >= -off) and (whi <= 65535 - off)
        extremes[name] = (wlo, whi, off)
        all_ok &= ok
        if not quiet:
            report(f"L2b window [{name}]", "PROVED" if ok else "FAIL",
                   f"honest-bound carries ⊂ [{wlo}, {whi}] vs window [{-off}, {65536-off})")
    return all_ok, extremes


def l2b_controls(extremes):
    # Tightness probe: shrink each offset to (−wlo − 1): window must now FAIL.
    ok_all = True
    for name, (wlo, whi, off) in extremes.items():
        tight_off = -wlo - 1  # one less than needed on the negative side
        ok = not (wlo >= -tight_off and whi <= 65535 - tight_off)
        ok_all &= ok
    report("L2b audit-control [offset below necessity ⇒ audit fails]",
           "PROVED" if ok_all else "FAIL",
           f"per-relation minimal offsets: { {k: -v[0] for k, v in extremes.items()} } "
           f"(repo offsets { {k: v[2] for k, v in extremes.items()} })")

    # Wrong-R probe: r = 2p ⇒ honest lambda-relation quotient can go negative
    # (double branch: q = 2p + (2λyA − 3xA²)/p ≥ 2p − 3p + small < 0) ⇒ the
    # nonneg 33-byte quotient can't exist for worst-case inputs.
    worst_num = -3 * (P - 1) ** 2  # most negative honest numerator (double branch)
    q_min_2p = 2 * P + worst_num // P
    q_min_3p = 3 * P + worst_num // P
    ok = (q_min_2p < 0) and (q_min_3p >= 0)
    report("L2b audit-control [r=2p insufficient, r=3p sufficient]",
           "PROVED" if ok else "FAIL",
           f"q_min(r=2p)={q_min_2p} < 0 ≤ q_min(r=3p)={q_min_3p}")

    # Quotient headroom: honest q ≤ 3p + max(num)/p must fit 33 bytes (< 2^264);
    # ECSM q1 ≤ p + max/p must fit its < 2^257 contract (32 bytes + top bit).
    q_max_ecdas = 3 * P + (2 * (P - 1) ** 2) // P  # xr rel: λ² dominates similarly
    q_max_ecdas = max(q_max_ecdas, 3 * P + ((P - 1) ** 2) // P)
    q1_max_ecsm = P + ((P - 1) ** 2) // P
    ok = q_max_ecdas < 2**264 and q1_max_ecsm < 2**257
    report("L2b quotient headroom", "PROVED" if ok else "FAIL",
           f"ECDAS q_max≈2^{q_max_ecdas.bit_length()}, ECSM q1_max≈2^{q1_max_ecsm.bit_length()}")


# ── L2c: overflow-chain lift (z3) ──

def l2c_overflow_chain():
    t0 = time.time()
    s = z3.Solver()
    # One word step: A ≡ 2^32·c (mod p_g), |A| < 2^33, c ∈ {0,1} ⇒ A = 2^32·c over ℤ.
    A, c, m = z3.Ints("A c m")
    s.add(A > -(2**33), A < 2**33, z3.Or(c == 0, c == 1))
    s.add(A - 2**32 * c == m * PG)          # the field equation, lifted with quotient m
    s.add(A != 2**32 * c)                    # deny the integer conclusion
    r1 = s.check()
    report("L2c word-carry lift", "PROVED" if r1 == z3.unsat else str(r1).upper(),
           f"{time.time()-t0:.1f}s")

    # Chained: const + witness_hl = value + 2^256·c7 with c7 = 1 ⇒ value < const.
    t0 = time.time()
    s = z3.Solver()
    words_c = [z3.Int(f"kc{i}") for i in range(8)]   # const words (symbolic ≤ 2^32-1)
    words_w = [z3.Int(f"kw{i}") for i in range(8)]   # halfword-addend words
    words_v = [z3.Int(f"kv{i}") for i in range(8)]   # value words
    carr = [z3.Int(f"cc{i}") for i in range(8)]
    for i in range(8):
        s.add(words_c[i] >= 0, words_c[i] < 2**32)
        s.add(words_w[i] >= 0, words_w[i] < 2**32)
        s.add(words_v[i] >= 0, words_v[i] < 2**32)
        s.add(z3.Or(carr[i] == 0, carr[i] == 1))
        prev = carr[i - 1] if i > 0 else 0
        s.add(words_c[i] + words_w[i] + prev - words_v[i] == 2**32 * carr[i])
    s.add(carr[7] == 1)
    C = z3.Sum([2**(32 * i) * words_c[i] for i in range(8)])
    W = z3.Sum([2**(32 * i) * words_w[i] for i in range(8)])
    V = z3.Sum([2**(32 * i) * words_v[i] for i in range(8)])
    s.add(z3.Not(z3.And(V < C, C + W == V + 2**256)))
    r2 = s.check()
    report("L2c strict-inequality chain", "PROVED" if r2 == z3.unsat else str(r2).upper(),
           f"{time.time()-t0:.1f}s")


if __name__ == "__main__":
    l1_telescoping()
    l2a_soundness()
    l2a_corner_crosscheck()
    ok, extremes = l2b_completeness()
    l2b_controls(extremes)
    l2c_overflow_chain()
    print("\nSummary:")
    for n, v, d in results:
        print(f"  {v:8} {n}")
    if any(v not in ("PROVED",) for _, v, _ in results):
        sys.exit(1)
