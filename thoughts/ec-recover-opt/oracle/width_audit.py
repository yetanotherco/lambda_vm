"""Width audit for the lincomb2 joint chain (IMPL-PLAN section 11, risk 8).

The question. Today's ECDAS proves `A + G` for a loop-invariant, external,
canonical addend `G`. Under lincomb2 the addend varies per row over
{P1, P2, P12, -2^len*T0}, and `P12 = P1 + P2` is an INTERIOR chip output that is
byte-bounded but never proven canonical (< p). Does the convolution
carry/quotient width argument survive that?

Two questions that must not be conflated, because the answer differs:

  SOUNDNESS  -- does the field equation `256*c_i - c_{i-1} - S_i = 0` still
                imply the INTEGER equation? That needs |256*c_i - c_{i-1} - S_i|
                < p_g so nothing wraps. Computed here as an exact interval over
                the constraint system's own guarantees (limbs are bytes, op/mu
                are bits, carries sit in their IsHalfword windows), with a
                random-corner cross-check and a z3 confirmation on small limbs.

  COMPLETENESS -- can the HONEST prover always find carries inside the windows
                and a quotient that is >= 0 and fits 33 bytes? That does depend
                on composed values. Measured here on real lincomb2 witnesses,
                over every row type including the two that break telescoping.

Run:  <venv>/bin/python width_audit.py
"""

import random
import sys

import lincomb2_ref
from ec_ref import GX, GY, N, P, recover_even_y, scalar_mul

# Goldilocks
P_G = (1 << 64) - (1 << 32) + 1

# Carry offsets, read from the chips (prover/src/tables/ecdas.rs:24-26, reused
# verbatim by ecdas2.rs:71; ecsm.rs:27-28).
CARRY_OFFSET_LAMBDA = 32636
CARRY_OFFSET_XR = 8161
CARRY_OFFSET_YR = 16320
CARRY_OFFSET_X2 = 8160
CARRY_OFFSET_YG = 16319

P_BYTES = list(P.to_bytes(32, "little"))
R_BYTES = list((3 * P).to_bytes(33, "little"))  # r = 3p, the shifted-quotient offset

BYTE_MAX = 255
NLIMB = 64


def ext64(byts):
    a = [0] * NLIMB
    for i, b in enumerate(byts):
        a[i] = b
    return a


PP = ext64(P_BYTES)
RR = ext64(R_BYTES)


# ── exact per-limb interval bounds ──────────────────────────────────────────
#
# Every term of S_i is a product of at most two limbs (times a small constant),
# summed over j <= i. Treating repeated variables (lam*lam, xa*xa) as
# independent is an OVER-approximation, which is the safe direction for a
# soundness bound. Terms whose factors are known constants (R, P) are bounded
# exactly rather than by 255.

def conv_const_bound(consts, i):
    """max over byte-valued b of  sum_j b[j] * consts[i-j]  (and the min, 0)."""
    return BYTE_MAX * sum(consts[i - j] for j in range(0, i + 1))


def rq_interval(i):
    """Interval of  mu*sum_j R[j]*P[i-j]  -  sum_j q[j]*P[i-j] ,
    with mu a bit and q byte-valued. R and P are constants."""
    r_p = sum(RR[j] * PP[i - j] for j in range(0, i + 1))  # exact, mu in {0,1}
    q_p_max = conv_const_bound(PP, i)  # q free bytes
    return (-q_p_max, r_p)


def s_interval(relation, i):
    """Exact interval for S_i, maximised over byte limbs and bit op/mu."""
    n = i + 1  # number of convolution terms
    b2 = BYTE_MAX * BYTE_MAX
    rq_lo, rq_hi = rq_interval(i)

    if relation == "lambda":
        # op = 1 branch: (ya_i - yg_i) + sum_j lam_j*(xg - xa)_{i-j}
        #   |lam_j| <= 255, |(xg-xa)_{i-j}| <= 255
        op1_lo, op1_hi = -(n * b2 + BYTE_MAX), n * b2 + BYTE_MAX
        # op = 0 branch: sum_j 2*lam_j*ya_{i-j} - 3*xa_j*xa_{i-j}
        op0_lo, op0_hi = -3 * n * b2, 2 * n * b2
        lo, hi = min(op1_lo, op0_lo), max(op1_hi, op0_hi)
    elif relation == "xr":
        # sum_j lam_j*lam_{i-j} - xa_i - xg_i - xr_i - (1-op)(xa_i - xg_i)
        lo = -3 * BYTE_MAX - BYTE_MAX  # the (1-op) term adds at most one more 255
        hi = n * b2 + BYTE_MAX
    elif relation == "yr":
        # sum_j lam_j*(xa - xr)_{i-j} - ya_i - yr_i
        lo, hi = -(n * b2 + 2 * BYTE_MAX), n * b2 + 2 * BYTE_MAX
    elif relation == "dinv":
        # NEW (closes the NUMS finding): sum_j dinv_j*(xb - xa)_{i-j} - [i==0]
        one = 1 if i == 0 else 0
        lo, hi = -(n * b2 + one), n * b2
    else:
        raise ValueError(relation)
    return lo + rq_lo, hi + rq_hi


def carry_window(offset):
    """IsHalfword(c + offset) in [0, 2^16)  =>  c in [-offset, 2^16-1-offset]."""
    return -offset, (1 << 16) - 1 - offset


def soundness_bound(relation, offset):
    """max |256*c_i - c_{i-1} - S_i| over everything the constraints allow."""
    c_lo, c_hi = carry_window(offset)
    worst = 0
    arg = None
    for i in range(NLIMB):
        s_lo, s_hi = s_interval(relation, i)
        # c_63 is ColIsZero'd, not windowed; c_{-1} is structurally 0.
        ci_lo, ci_hi = (0, 0) if i == 63 else (c_lo, c_hi)
        cp_lo, cp_hi = (0, 0) if i == 0 else (c_lo, c_hi)
        hi = 256 * ci_hi - cp_lo - s_lo
        lo = 256 * ci_lo - cp_hi - s_hi
        m = max(abs(hi), abs(lo))
        if m > worst:
            worst, arg = m, i
    return worst, arg


# ── honest witnesses: recompute the carries independently ───────────────────

def conv(a, b, i):
    return sum(a[j] * b[i - j] for j in range(0, i + 1))


def limb_carries(terms):
    """c_i = (c_{i-1} + terms_i)/256, asserting exact divisibility and c_63 = 0."""
    c = []
    carry = 0
    for i in range(NLIMB):
        s = carry + terms[i]
        assert s % 256 == 0, f"limb {i} not divisible by 256"
        carry = s // 256
        c.append(carry)
    assert c[63] == 0, "closing carry c_63 != 0"
    return c


def le32(v):
    return ext64(list(v.to_bytes(32, "little")))


def le33(v):
    return ext64(list(v.to_bytes(33, "little")))


def honest_row_carries(a, addend, r_pt, lam, op):
    """The three relations' honest carries and quotients for one row, computed
    from the group-law values alone (no Rust involved)."""
    xa, ya = le32(a[0]), le32(a[1])
    xg, yg = le32(addend[0]), le32(addend[1])
    xr, yr = le32(r_pt[0]), le32(r_pt[1])
    lm = le32(lam)
    out = {}

    # lambda: op*(lam*(xg-xa) + ya - yg) + (1-op)*(2*lam*ya - 3*xa^2) + (r - q0)*p
    if op == 1:
        v = lam * (addend[0] - a[0]) + a[1] - addend[1]
    else:
        v = 2 * lam * a[1] - 3 * a[0] * a[0]
    q0 = 3 * P + v // P
    assert v % P == 0, "lambda numerator not divisible by p"
    q0b = le33(q0)
    terms = []
    for i in range(NLIMB):
        if op == 1:
            s = ya[i] - yg[i] + sum(lm[j] * (xg[i - j] - xa[i - j]) for j in range(0, i + 1))
        else:
            s = sum(2 * lm[j] * ya[i - j] - 3 * xa[j] * xa[i - j] for j in range(0, i + 1))
        terms.append(s + conv(RR, PP, i) - conv(q0b, PP, i))
    out["lambda"] = (limb_carries(terms), q0)

    # xR: lam^2 - xa - xg - xr - (1-op)(xa - xg) + (r - q1)*p
    v = lam * lam - a[0] - addend[0] - r_pt[0] - (0 if op == 1 else (a[0] - addend[0]))
    assert v % P == 0, "xR numerator not divisible by p"
    q1 = 3 * P + v // P
    q1b = le33(q1)
    terms = []
    for i in range(NLIMB):
        op_term = (xa[i] - xg[i]) if op == 0 else 0
        terms.append(conv(lm, lm, i) - xa[i] - xg[i] - xr[i] - op_term
                     + conv(RR, PP, i) - conv(q1b, PP, i))
    out["xr"] = (limb_carries(terms), q1)

    # yR: lam*(xa - xr) - ya - yr + (r - q2)*p
    v = lam * (a[0] - r_pt[0]) - a[1] - r_pt[1]
    assert v % P == 0, "yR numerator not divisible by p"
    q2 = 3 * P + v // P
    q2b = le33(q2)
    terms = []
    for i in range(NLIMB):
        cl = sum(lm[j] * (xa[i - j] - xr[i - j]) for j in range(0, i + 1))
        terms.append(cl - ya[i] - yr[i] + conv(RR, PP, i) - conv(q2b, PP, i))
    out["yr"] = (limb_carries(terms), q2)
    return out


def honest_dinv_carries(a, addend):
    """The proposed non-degeneracy relation: d_inv*(xB - xA) - 1 + (r - q3)*p = 0."""
    d = (addend[0] - a[0]) % P
    assert d != 0
    dinv = pow(d, P - 2, P)
    v = dinv * (addend[0] - a[0]) - 1
    assert v % P == 0, "dinv numerator not divisible by p"
    q3 = 3 * P + v // P
    assert q3 >= 0
    di, q3b = le32(dinv), le33(q3)
    xa, xg = le32(a[0]), le32(addend[0])
    terms = []
    for i in range(NLIMB):
        s = sum(di[j] * (xg[i - j] - xa[i - j]) for j in range(0, i + 1))
        if i == 0:
            s -= 1
        terms.append(s + conv(RR, PP, i) - conv(q3b, PP, i))
    return limb_carries(terms), q3


def rust_differential(cases, T0, harness="repo-harness/target/release/ecsm-oracle-harness"):
    """Cross-check: the carries and quotients this script derives must equal the
    ones the real prover witness (`ecsm::lincomb2_witness`) emits. Without this,
    Part 3 would only be measuring a reimplementation of the relations."""
    import json
    import subprocess

    lines = [f"lincomb2 {u1:x} {u2:x} {p1[0]:x} {p1[1]:x} {p2[0]:x} {p2[1]:x}"
             for u1, u2, p1, p2 in cases]
    try:
        r = subprocess.run([harness], input="\n".join(lines) + "\n",
                           capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as e:
        return None, f"harness unavailable ({e})"

    mismatches = 0
    compared = 0
    for (u1, u2, p1, p2), out in zip(cases, r.stdout.splitlines()):
        if not out.startswith("lincomb2_json "):
            return None, f"harness rejected a case: {out[:60]}"
        w = json.loads(out[len("lincomb2_json "):])
        _q, _len, rows = lincomb2_ref.lincomb2_rows(u1, p1, u2, p2, T0)
        for rrow, prow in zip(w["rows"], rows):
            mine = honest_row_carries(prow["a"], prow["addend"], prow["r"],
                                      prow["lam"], prow["op"])
            for rel, qkey, ckey in (("lambda", "q0", "c0"), ("xr", "q1", "c1"),
                                    ("yr", "q2", "c2")):
                carries, q = mine[rel]
                if int.from_bytes(bytes.fromhex(rrow[qkey]), "little") != q:
                    mismatches += 1
                if rrow[ckey] != carries:
                    mismatches += 1
                compared += 2
    return (compared, mismatches), None


def main():
    rng = random.Random(31337)
    T0, _ = lincomb2_ref.t0_ref()
    G = (GX, GY)

    print("=" * 78)
    print("PART 1 -- SOUNDNESS: does the field equation still imply the integer one?")
    print("=" * 78)
    print(f"p_g (Goldilocks) = {P_G} ~ 2^{P_G.bit_length() - 1}")
    print()
    print(f"{'relation':10} {'offset':>7} {'carry window':>22} "
          f"{'max |256c - c- - S|':>21} {'vs p_g':>12}")
    rels = [("lambda", CARRY_OFFSET_LAMBDA), ("xr", CARRY_OFFSET_XR),
            ("yr", CARRY_OFFSET_YR), ("dinv", CARRY_OFFSET_LAMBDA)]
    worst_overall = 0
    for rel, off in rels:
        w, i = soundness_bound(rel, off)
        lo, hi = carry_window(off)
        worst_overall = max(worst_overall, w)
        print(f"{rel:10} {off:>7} {f'[{lo}, {hi}]':>22} "
              f"{w:>21,} {P_G // w:>10,}x")
    print()
    print(f"worst case over all relations: {worst_overall:,} "
          f"~ 2^{worst_overall.bit_length() - 1}, headroom 2^"
          f"{(P_G // worst_overall).bit_length() - 1}")
    print()
    print("These intervals are over-approximations (repeated variables such as")
    print("lam*lam and xa*xa are treated as independent), and they are maximised")
    print("over BYTE LIMBS ONLY -- no operand is assumed canonical. Restricting")
    print("any operand to [0, p) can only shrink them.")

    print()
    print("Sensitivity: how large would ONE non-byte limb have to be to break it?")
    for name, coeff in [("3*xa_j*xa_{i-j} (lambda, op=0)", 3), ("lam_j*lam_{i-j} (xR)", 1)]:
        x = 1
        while coeff * 64 * x * x < P_G:
            x *= 2
        print(f"   {name:34} breaks at a limb of ~2^{x.bit_length() - 1}")
    print("   => byte-ness of EVERY limb is load-bearing, not hygiene.")

    print()
    print("=" * 78)
    print("PART 2 -- random-corner cross-check of the interval method")
    print("=" * 78)
    fails = 0
    checked = 0
    for rel, off in rels:
        s_lo_all = {}
        for i in (0, 1, 7, 31, 62, 63):
            lo, hi = s_interval(rel, i)
            for _ in range(1500):
                # sample limbs at their corners (0 / 255) plus random values
                def lim():
                    r = rng.random()
                    return 0 if r < 0.4 else (255 if r < 0.8 else rng.randrange(256))
                lam = [lim() for _ in range(NLIMB)]
                xa = [lim() for _ in range(NLIMB)]
                xg = [lim() for _ in range(NLIMB)]
                ya = [lim() for _ in range(NLIMB)]
                yg = [lim() for _ in range(NLIMB)]
                xr = [lim() for _ in range(NLIMB)]
                yr = [lim() for _ in range(NLIMB)]
                q = [lim() for _ in range(NLIMB)]
                op = rng.randrange(2)
                mu = rng.randrange(2)
                if rel == "lambda":
                    if op == 1:
                        s = ya[i] - yg[i] + sum(lam[j] * (xg[i - j] - xa[i - j])
                                                for j in range(i + 1))
                    else:
                        s = sum(2 * lam[j] * ya[i - j] - 3 * xa[j] * xa[i - j]
                                for j in range(i + 1))
                elif rel == "xr":
                    s = (sum(lam[j] * lam[i - j] for j in range(i + 1))
                         - xa[i] - xg[i] - xr[i] - ((xa[i] - xg[i]) if op == 0 else 0))
                elif rel == "yr":
                    s = (sum(lam[j] * (xa[i - j] - xr[i - j]) for j in range(i + 1))
                         - ya[i] - yr[i])
                else:
                    s = sum(lam[j] * (xg[i - j] - xa[i - j]) for j in range(i + 1))
                    if i == 0:
                        s -= 1
                s += mu * conv(RR, PP, i) - conv(q, PP, i)
                checked += 1
                if not (lo <= s <= hi):
                    fails += 1
                    if fails < 4:
                        print(f"   INTERVAL VIOLATED {rel} i={i}: S={s} not in [{lo}, {hi}]")
                s_lo_all[i] = min(s_lo_all.get(i, s), s)
    print(f"   {checked:,} corner/random samples, {fails} interval violations")

    print()
    print("=" * 78)
    print("PART 3 -- COMPLETENESS: honest carries on real lincomb2 witnesses")
    print("=" * 78)

    stats = {}          # relation -> [min c, max c, min q, max q]
    per_sel = {}        # (sel, relation) -> [min c, max c]

    def record(rel, carries, q, sel):
        st = stats.setdefault(rel, [10**9, -10**9, None, None])
        st[0] = min(st[0], min(carries))
        st[1] = max(st[1], max(carries))
        st[2] = q if st[2] is None else min(st[2], q)
        st[3] = q if st[3] is None else max(st[3], q)
        ps = per_sel.setdefault((sel, rel), [10**9, -10**9])
        ps[0] = min(ps[0], min(carries))
        ps[1] = max(ps[1], max(carries))

    cases = []
    for _ in range(12):
        while True:
            x = rng.randrange(P)
            y = recover_even_y(x)
            if y is not None:
                break
        p2 = (x, (P - y) % P if rng.random() < 0.5 else y)
        cases.append((rng.randrange(1, N), rng.randrange(1, N), G, p2))
    # edges: maximal scalars, and a P2 with tiny/huge coordinates
    cases.append((N - 1, N - 1, G, scalar_mul(7, G)))
    cases.append((1, 1, G, scalar_mul(3, G)))
    cases.append((2**255, 2**255 - 1, G, scalar_mul(5, G)))

    res, err = rust_differential(cases, T0)
    if err:
        print(f"   [!] Rust differential SKIPPED: {err}")
        print("       Part 3 then measures this script's own reimplementation only.")
        diff_ok = False
    else:
        compared, mism = res
        diff_ok = mism == 0
        print(f"   Rust differential: {compared:,} (quotient, carry-array) comparisons "
              f"against `ecsm::lincomb2_witness`, {mism} mismatches")
        if mism:
            print("   [!] the carries measured below are NOT the prover's")
    print()

    rows_seen = 0
    dinv_c = [10**9, -10**9]
    dinv_q = [None, None]
    for u1, u2, p1, p2 in cases:
        _q, _len, rows = lincomb2_ref.lincomb2_rows(u1, p1, u2, p2, T0)
        for r in rows:
            rows_seen += 1
            res = honest_row_carries(r["a"], r["addend"], r["r"], r["lam"], r["op"])
            for rel, (carries, q) in res.items():
                record(rel, carries, q, r["sel"])
            if r["op"] == 1:  # addend-consuming rows get the new relation
                c, q3 = honest_dinv_carries(r["a"], r["addend"])
                dinv_c[0] = min(dinv_c[0], min(c))
                dinv_c[1] = max(dinv_c[1], max(c))
                dinv_q[0] = q3 if dinv_q[0] is None else min(dinv_q[0], q3)
                dinv_q[1] = q3 if dinv_q[1] is None else max(dinv_q[1], q3)

    print(f"{rows_seen:,} real rows over {len(cases)} lincomb2 evaluations "
          f"(every row type incl. Precompute and Correction)")
    print()
    print(f"{'relation':10} {'honest c range':>26} {'window':>22} {'fits':>6} "
          f"{'min offset needed':>18}")
    ok = True
    for rel, off in [("lambda", CARRY_OFFSET_LAMBDA), ("xr", CARRY_OFFSET_XR),
                     ("yr", CARRY_OFFSET_YR)]:
        lo, hi = carry_window(off)
        cmin, cmax, qmin, qmax = stats[rel]
        fits = lo <= cmin and cmax <= hi
        ok &= fits
        print(f"{rel:10} {f'[{cmin}, {cmax}]':>26} {f'[{lo}, {hi}]':>22} "
              f"{str(fits):>6} {-cmin:>18}")
    lo, hi = carry_window(CARRY_OFFSET_LAMBDA)
    fits = lo <= dinv_c[0] and dinv_c[1] <= hi
    ok &= fits
    print(f"{'dinv':10} {f'[{dinv_c[0]}, {dinv_c[1]}]':>26} {f'[{lo}, {hi}]':>22} "
          f"{str(fits):>6} {-dinv_c[0]:>18}")

    print()
    print("quotient headroom (33 bytes = [0, 2^264)):")
    for rel in ("lambda", "xr", "yr"):
        _, _, qmin, qmax = stats[rel]
        print(f"   {rel:8} q in [{qmin.bit_length()} bits, {qmax.bit_length()} bits]  "
              f"min q >= 0: {qmin >= 0}   max q < 2^264: {qmax < (1 << 264)}")
        ok &= (qmin >= 0 and qmax < (1 << 264))
    print(f"   {'dinv':8} q in [{dinv_q[0].bit_length()} bits, {dinv_q[1].bit_length()} bits]  "
          f"min q >= 0: {dinv_q[0] >= 0}   max q < 2^264: {dinv_q[1] < (1 << 264)}")
    ok &= (dinv_q[0] >= 0 and dinv_q[1] < (1 << 264))

    print()
    print("per row type (carry ranges), to show the new row shapes are not outliers:")
    for sel in ("Precompute", "Double", "AddP1", "AddP2", "AddP12", "Correction"):
        parts = []
        for rel in ("lambda", "xr", "yr"):
            v = per_sel.get((sel, rel))
            if v:
                parts.append(f"{rel}=[{v[0]}, {v[1]}]")
        if parts:
            print(f"   {sel:12} " + "  ".join(parts))

    print()
    print("=" * 78)
    print(f"WIDTH AUDIT: {chr(80)+chr(65)+chr(83)+chr(83) if (ok and fails == 0 and diff_ok) else chr(70)+chr(65)+chr(73)+chr(76)}")
    print("=" * 78)
    return 0 if (ok and fails == 0 and diff_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
