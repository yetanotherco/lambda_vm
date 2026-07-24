"""L5: discharging the step lemma's side conditions.

L5a (computation, certified): secp256k1 has NO point with y ≡ 0: such a point
    would have order 2, impossible since |E(F_p)| = N is odd (cofactor 1).
    Equivalently x³+7 has no root in F_p ⟺ −7 is a non-cube (p ≡ 1 mod 3).
    Both routes computed and cross-checked. Covers the op=0 side condition
    p∤2yA for ANY byte-bounded yA with A on-curve mod p (yA ≡ 0 mod p would be
    a y=0 curve point).

L5b (z3): the incomplete-addition edge is unreachable — at any add row the
    incoming accumulator is c·G with c = 2t even, 3 ≤ u = 2t+1 = ⌊k/2^r⌋ (bit r
    of k set), k < N ⇒ c ∈ [2, N−2] ⇒ c ≢ ±1 (mod N) ⇒ A ≠ ±G in the group.
    (The identification A = c·G and u = ⌊k/2^r⌋ is the L6 chain induction; this
    lemma discharges its arithmetic core.)

L5c (z3): x-equality ⇒ same-or-negated point (needed to turn "A ≠ ±G" into the
    algebraic side condition p∤(xG−xA)): if both points satisfy the curve
    equation mod p and xA ≡ xG, then yA ≡ ±yG. Easy divisibility direction plus
    ONE more A-PRIME Euclid instance for (yA−yG)(yA+yG) ≡ 0.
"""

import sys
import time
from pathlib import Path

import sympy as sp
import z3

sys.path.insert(0, str(Path(__file__).parent))
from gate_common import B, N, P

results = []


def report(name, verdict, detail=""):
    results.append((name, verdict, detail))
    print(f"[{verdict}] {name}  {detail}", flush=True)


def l5a():
    assert P % 3 == 1
    cube_test = pow((-B) % P, (P - 1) // 3, P)
    no_root = cube_test != 1  # −7 is a cube ⟺ x³+7 has a root
    n_odd = N % 2 == 1
    # Independent cross-check via sympy's nthroot_mod on a few candidates is
    # overkill; the two facts must agree: no y=0 point ⟺ no root ⟸ N odd.
    report("L5a no-2-torsion", "PROVED" if (no_root and n_odd) else "FAIL",
           f"(−7)^((p−1)/3) mod p ≠ 1: {no_root}; |E|=N odd: {n_odd}")


def l5b():
    t0 = time.time()
    s = z3.Solver()
    s.set("timeout", 120000)
    k, u, ss, rem, t, c = z3.Ints("k u s rem t c")
    s.add(k >= 1, k < N)
    s.add(ss >= 1)                       # s = 2^r ≥ 1 (r ≥ 0)
    s.add(k == u * ss + rem, rem >= 0, rem < ss)  # u = ⌊k/2^r⌋
    s.add(u == 2 * t + 1)                # bit r of k is set (add row, Bit balance)
    s.add(t >= 1)                        # not the seed: at least one prior double
    s.add(c == 2 * t)                    # incoming accumulator multiplier
    s.add(z3.Or(c % N == 1, c % N == N - 1))  # A = ±G
    r = s.check()
    report("L5b incomplete-addition unreachable", "PROVED" if r == z3.unsat else str(r).upper(),
           f"{time.time()-t0:.0f}s")


def l5c():
    t0 = time.time()
    # (i) ring: curve equations + xA ≡ xG ⇒ (yA−yG)(yA+yG) = T·p explicit.
    xa, ya, xg, yg, qa, qg, dq, T = z3.Ints("xa ya xg yg qa qg dq T")
    s = z3.Solver()
    s.set("timeout", 120000)
    s.add(ya * ya - (xa * xa * xa + B) == qa * P)
    s.add(yg * yg - (xg * xg * xg + B) == qg * P)
    s.add(xa - xg == dq * P)
    # (yA−yG)(yA+yG) = yA²−yG² = (xa³−xg³) + (qa−qg)p; xa³−xg³ = (xa−xg)(xa²+xa·xg+xg²).
    s.add((ya - yg) * (ya + yg) != (dq * (xa * xa + xa * xg + xg * xg) + qa - qg) * P)
    r1 = s.check()

    # (ii) remainder split: p | (yA−yG)(yA+yG) ∧ p∤(yA−yG) ∧ p∤(yA+yG) ⇒ p | gr·dr.
    g1q, g1r, g2q, g2r, T2 = z3.Ints("g1q g1r g2q g2r T2")
    s = z3.Solver()
    s.set("timeout", 120000)
    s.add((ya - yg) * (ya + yg) == T2 * P)
    s.add(ya - yg == g1q * P + g1r, g1r > 0, g1r < P)
    s.add(ya + yg == g2q * P + g2r, g2r > 0, g2r < P)
    s.add((g1r * g2r) % P != 0)
    r2 = s.check()
    v = "PROVED" if (r1 == z3.unsat and r2 == z3.unsat) else f"(i)={r1},(ii)={r2}"
    report("L5c x-equal ⇒ y = ±y (mod p)", v,
           f"+ A-PRIME closes; {time.time()-t0:.0f}s")


if __name__ == "__main__":
    l5a()
    l5b()
    l5c()
    print("\nSummary:")
    for n, v, d in results:
        print(f"  {v:8} {n}")
    if any(v != "PROVED" for _, v, _ in results):
        sys.exit(1)
