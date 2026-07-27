"""L3 + L4: byte-level ⇒ value-level ⇒ step soundness.

L3a (sympy, exact symbolic expansion): the composition identities — for each of
    the five relations, Σ_i 256^i·S_i(bytes) is IDENTICALLY the intended
    polynomial in the composed values. Together with L1+L2 this turns the chip's
    field constraints into exact integer equations (gate_common.val_relations_*).

L3b (sympy): on-curve preservation — the chord/tangent output of each branch
    satisfies the curve equation, as a rational-function identity modulo
    yA²=xA³+7, yG²=xG³+7.

L4 (z3): value-level pinning. The ONLY lemma resting on primality (contract
    A-PRIME, certified below): λ is pinned mod p by its relation given the side
    condition (op=1: p∤(xG−xA); op=0: p∤2yA). xR/yR pinning then follows by the
    easy divisibility direction (p|a ⇒ p|ab), no primality needed.

    Euclid use is split into machine-checked ring algebra + ONE assumed,
    certified fact (see euclid_note in RESULTS.md):
      (i)  z3: relations ⇒ (λ−λ*)·d = T·p for an explicit T          [UNSAT]
      (ii) z3: (λ−λ*)·d = T·p ∧ λ−λ* = gq·p+gr ∧ d = dq·p+dr
               ⇒ (gr·dr) ≡ 0 mod p                                    [UNSAT]
      (iii) certified assumption A-PRIME: 0<gr,dr<p ⇒ gr·dr ≢ 0 mod p
            (p prime; sympy.isprime + printed certificate).
"""

import sys
import time
from pathlib import Path

import sympy as sp
import z3

sys.path.insert(0, str(Path(__file__).parent))
from gate_common import (
    B, N, P, R3P,
    s_ecsm_x2, s_ecsm_yg, s_ecdas_lambda, s_ecdas_xr, s_ecdas_yr,
)

results = []


def report(name, verdict, detail=""):
    results.append((name, verdict, detail))
    print(f"[{verdict}] {name}  {detail}", flush=True)


# ── L3a: composition identities (sympy) ──

def syms(prefix, n):
    return [sp.Symbol(f"{prefix}{j}", integer=True) for j in range(n)]


def V(arr):
    return sum(sp.Integer(256) ** j * a for j, a in enumerate(arr))


def l3a():
    t0 = time.time()
    op = sp.Symbol("op", integer=True)

    ecsm_ops = {"xg": syms("xg", 32), "yg": syms("yg", 32), "x2": syms("x2", 32),
                "q0": syms("q0", 32), "q1": syms("q1", 33)}
    ecdas_ops = {k: syms(k, 32) for k in ["lam", "xa", "ya", "xg", "yg", "xr", "yr"]}
    ecdas_ops.update({q: syms(q, 33) for q in ["q0", "q1", "q2"]})

    def total(sfn, v, **kw):
        return sp.expand(sum(sp.Integer(256) ** i * sfn(v, i, **kw) for i in range(64)))

    cases = [
        ("ecsm_x2",
         total(lambda v, i: s_ecsm_x2(v, i), ecsm_ops),
         V(ecsm_ops["xg"]) ** 2 - V(ecsm_ops["x2"]) - V(ecsm_ops["q0"]) * P),
        ("ecsm_yg",
         total(lambda v, i: s_ecsm_yg(v, i, mu=1), ecsm_ops),
         V(ecsm_ops["yg"]) ** 2 + P * P - V(ecsm_ops["x2"]) * V(ecsm_ops["xg"])
         - V(ecsm_ops["q1"]) * P - B),
        ("ecdas_lambda",
         total(lambda v, i: s_ecdas_lambda(v, i, op, mu=1), ecdas_ops),
         op * (V(ecdas_ops["lam"]) * (V(ecdas_ops["xg"]) - V(ecdas_ops["xa"]))
               + V(ecdas_ops["ya"]) - V(ecdas_ops["yg"]))
         + (1 - op) * (2 * V(ecdas_ops["lam"]) * V(ecdas_ops["ya"])
                       - 3 * V(ecdas_ops["xa"]) ** 2)
         + R3P * P - V(ecdas_ops["q0"]) * P),
        ("ecdas_xr",
         total(lambda v, i: s_ecdas_xr(v, i, op, mu=1), ecdas_ops),
         V(ecdas_ops["lam"]) ** 2 - V(ecdas_ops["xa"]) - V(ecdas_ops["xg"])
         - V(ecdas_ops["xr"]) - (1 - op) * (V(ecdas_ops["xa"]) - V(ecdas_ops["xg"]))
         + R3P * P - V(ecdas_ops["q1"]) * P),
        ("ecdas_yr",
         total(lambda v, i: s_ecdas_yr(v, i, mu=1), ecdas_ops),
         V(ecdas_ops["lam"]) * (V(ecdas_ops["xa"]) - V(ecdas_ops["xr"]))
         - V(ecdas_ops["ya"]) - V(ecdas_ops["yr"]) + R3P * P - V(ecdas_ops["q2"]) * P),
    ]
    for name, lhs, rhs in cases:
        diff = sp.expand(lhs - rhs)
        report(f"L3a composition [{name}]",
               "PROVED" if diff == 0 else "FAIL",
               f"{time.time()-t0:.0f}s cumulative")


# ── L3b: on-curve preservation (sympy rational identity) ──

def l3b():
    xa, ya, xg, yg = sp.symbols("xa ya xg yg")
    b = sp.Integer(B)

    def reduce_curve(expr):
        """Reduce even powers of ya/yg via ya² = xa³+b, yg² = xg³+b."""
        expr = sp.expand(expr)
        for y, x in [(ya, xa), (yg, xg)]:
            p = sp.Poly(expr, y)
            new = 0
            for (deg,), coef in p.terms():
                q, r = divmod(deg, 2)
                new += coef * (x**3 + b) ** q * y**r
            expr = sp.expand(new)
        return expr

    # Add branch (op=1): λ = (yg−ya)/(xg−xa), xr = λ²−xa−xg, yr = λ(xa−xr)−ya.
    lam = (yg - ya) / (xg - xa)
    xr = lam**2 - xa - xg
    yr = lam * (xa - xr) - ya
    num = sp.together(yr**2 - xr**3 - b)
    num = sp.fraction(sp.cancel(num))[0]
    add_ok = reduce_curve(num) == 0

    # Double branch (op=0): λ = 3xa²/(2ya), xr = λ²−2xa, yr = λ(xa−xr)−ya.
    lam = 3 * xa**2 / (2 * ya)
    xr = lam**2 - 2 * xa
    yr = lam * (xa - xr) - ya
    num = sp.fraction(sp.cancel(sp.together(yr**2 - xr**3 - b)))[0]
    dbl_ok = reduce_curve(sp.expand(num)) == 0

    report("L3b on-curve preservation [add]", "PROVED" if add_ok else "FAIL")
    report("L3b on-curve preservation [double]", "PROVED" if dbl_ok else "FAIL")


# ── A-PRIME certificate ──

def certify_primes():
    okp, okn = sp.isprime(P), sp.isprime(N)
    report("A-PRIME certificate", "PROVED" if (okp and okn) else "FAIL",
           f"isprime(p)={okp}, isprime(N)={okn}, N odd={N % 2 == 1}")


# ── L4a: λ pinned mod p (per branch) — Euclid split ──

def _lambda_pin(op):
    """Returns verdicts for sub-queries (i) and (ii) of the Euclid split."""
    lam, lamstar, q0, mstar = z3.Ints("lam lamstar q0 mstar")
    xa, ya, xg, yg = z3.Ints("xa ya xg yg")
    gq, gr, dq, dr, T = z3.Ints("gq gr dq dr T")

    if op == 1:
        d = xg - xa
        rel = lam * d + ya - yg + R3P * P - q0 * P          # chip relation, ℤ
        ref = lamstar * d + ya - yg - mstar * P              # λ* def: λ*d ≡ yg−ya... sign!
        # chip: λd + ya − yg ≡ 0  ⇒ λd ≡ yg − ya. Reference identical form.
        T_expr = q0 - 3 * P - mstar
    else:
        d = 2 * ya
        rel = 2 * lam * ya - 3 * xa * xa + R3P * P - q0 * P  # 2λyA − 3xA²
        ref = 2 * lamstar * ya - 3 * xa * xa - mstar * P
        T_expr = q0 - 3 * P - mstar

    base = [rel == 0, ref == 0,
            lam >= 0, lam < 2**256, lamstar >= 0, lamstar < P,
            q0 >= 0, q0 < 2**264]

    # (i) ring algebra: (λ−λ*)·d == T·p with T explicit.
    s = z3.Solver()
    s.set("timeout", 120000)
    s.add(base)
    s.add((lam - lamstar) * d != T_expr * P)
    r1 = s.check()

    # (ii) remainder decomposition forces p | gr·dr.
    s = z3.Solver()
    s.set("timeout", 120000)
    s.add((lam - lamstar) * d == T * P)
    s.add(lam - lamstar == gq * P + gr, gr > 0, gr < P)     # negated goal: λ ≢ λ*
    s.add(d == dq * P + dr, dr > 0, dr < P)                 # side condition: p ∤ d
    s.add((gr * dr) % P != 0)
    r2 = s.check()
    return r1, r2


def l4a():
    for op, side in [(1, "p∤(xG−xA)"), (0, "p∤2yA")]:
        t0 = time.time()
        r1, r2 = _lambda_pin(op)
        v1 = "PROVED" if r1 == z3.unsat else str(r1).upper()
        v2 = "PROVED" if r2 == z3.unsat else str(r2).upper()
        verdict = "PROVED" if (v1 == v2 == "PROVED") else f"(i)={v1},(ii)={v2}"
        report(f"L4a λ-pin [op={op}]", verdict,
               f"side: {side}; + A-PRIME closes gr·dr ≡ 0; {time.time()-t0:.0f}s")


# ── L4b/L4c: xR, yR pinned mod p given λ pinned (no primality) ──

def l4bc():
    lam, lamstar, q1, q2, m1, m2, gq = z3.Ints("lam lamstar q1 q2 m1 m2 gq")
    xa, xg, xr, xrstar, yr, yrstar, ya = z3.Ints("xa xg xr xrstar yr yrstar ya")
    hq, hr = z3.Ints("hq hr")
    op = z3.Int("op")

    t0 = time.time()
    s = z3.Solver()
    s.set("timeout", 120000)
    s.add(z3.Or(op == 0, op == 1))
    s.add(lam - lamstar == gq * P)  # λ ≡ λ* (from L4a)
    # chip xR relation (ℤ) and reference congruence:
    s.add(lam * lam - xa - xg - xr - (1 - op) * (xa - xg) + R3P * P - q1 * P == 0)
    s.add(lamstar * lamstar - xa - xg - xrstar - (1 - op) * (xa - xg) - m1 * P == 0)
    s.add(xr - xrstar == hq * P + hr, hr > 0, hr < P)  # negated goal
    r1 = s.check()
    report("L4b xR-pin", "PROVED" if r1 == z3.unsat else str(r1).upper(),
           f"{time.time()-t0:.0f}s")

    t0 = time.time()
    s = z3.Solver()
    s.set("timeout", 120000)
    s.add(lam - lamstar == gq * P)
    s.add(xr - xrstar == m2 * P)  # xR ≡ xR* (from L4b)
    s.add(lam * (xa - xr) - ya - yr + R3P * P - q2 * P == 0)
    s.add(lamstar * (xa - xrstar) - ya - yrstar - m1 * P == 0)
    s.add(yr - yrstar == hq * P + hr, hr > 0, hr < P)
    r2 = s.check()
    report("L4c yR-pin", "PROVED" if r2 == z3.unsat else str(r2).upper(),
           f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    certify_primes()
    l3a()
    l3b()
    l4a()
    l4bc()
    print("\nSummary:")
    for n, v, d in results:
        print(f"  {v:8} {n}")
    if any(v != "PROVED" for _, v, _ in results):
        sys.exit(1)
