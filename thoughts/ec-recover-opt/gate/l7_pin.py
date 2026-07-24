"""L7: end-to-end — the constraint system pins the chip output to the ORACLE's
x(k·P), for concrete (P, k) with symbolic witnesses.

Method (mod-p class propagation; all z3 queries LINEAR):
  Leave every quotient q as a FREE integer (strictly weaker hypothesis than the
  chip's q ∈ [0,2^264) — so the pinning conclusion is strictly stronger, and it
  makes each relation constrain values mod p only). Then a step's output classes
  depend only on the input classes mod p, so the whole chain state is one
  (xa, ya) class pair per seed sign. Per step, three z3 UNSAT queries prove
  λ / xR / yR ≡ the reference values mod p for ANY byte-representable witness
  (the denominators are concrete ⇒ relations linear in the unknown; unknown
  ranges [0,2^256) from the AreBytes contracts).

  Seed: xG is canonical (XG_SUB_P, L2c); yG is pinned by the ECSM curve
  relations only up to sign — BOTH sign classes are propagated and must
  converge to the same drain x (they do: x-only contract).

  Drain: the ECSM row's XR bytes equal the chain's final xR bytes (Ecdas tuple
  equality), and XR_SUB_P forces xR < p (L2c) ⇒ the pinned class has exactly
  one admissible representative, compared against the oracle.

  Schedule shape is the honest one for k (justified by L6's Bit-balance
  argument; l8's N4 probes tampered schedules).

Reference: thoughts/ec-recover-opt/oracle/ec_ref.py (independent lineage).
"""

import sys
import time
from pathlib import Path

import z3

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "oracle"))
import ec_ref  # the oracle's independent reference
from gate_common import B, N, P, R3P, GEN_X, ref_step

results = []
n_queries = 0
n_unsat = 0


def report(name, verdict, detail=""):
    results.append((name, verdict, detail))
    print(f"[{verdict}] {name}  {detail}", flush=True)


def pin(constraint_fn, expected_mod_p, tag, tamper_note=""):
    """UNSAT query: relation(x, q free) ∧ x ∈ [0,2^256) ∧ x ≢ expected (mod P)."""
    global n_queries, n_unsat
    x, q, gq, gr = z3.Ints("x q gq gr")
    s = z3.Solver()
    s.set("timeout", 60000)
    s.add(constraint_fn(x, q))
    s.add(x >= 0, x < 2**256)  # q intentionally FREE (see module docstring)
    s.add(x - expected_mod_p == gq * P + gr, gr > 0, gr < P)
    r = s.check()
    n_queries += 1
    if r == z3.unsat:
        n_unsat += 1
        return True
    print(f"  !! not pinned: {tag} -> {r} {tamper_note}")
    return False


def schedule(k):
    """Honest double/add schedule (round, op, next_op) — mirrors curve.rs:128-148."""
    m = k.bit_length() - 1
    sched = []
    round_, op = m - 1, 0
    while round_ >= 0:
        next_op = (1 if (k >> round_) & 1 else 0) if op == 0 else 0
        sched.append((round_, op, next_op))
        rs = round_ - (1 - next_op)
        if rs < 0:
            break
        round_, op = rs, next_op
    return sched


def run_chain(xg, k, sign, tamper=None):
    """Propagate one seed sign class; returns final xR class mod p (or None)."""
    y0 = ec_ref.recover_even_y(xg)
    yg = y0 if sign == 0 else (-y0) % P
    if k == 1:
        return xg  # drain == seed tuple ⇒ xR = xG bytes (echo)
    xa, ya = xg, yg
    for t, (rnd, op, next_op) in enumerate(schedule(k)):
        lam_s, xr_s, yr_s = ref_step(op, xa, ya, xg, yg)
        if op == 1:
            rel = lambda x, q: x * (xg - xa) + ya - yg + R3P * P - q * P == 0
        else:
            rel = lambda x, q: 2 * x * ya - 3 * xa * xa + R3P * P - q * P == 0
        if not pin(rel, lam_s, f"k={k} t={t} λ"):
            return None
        lam = lam_s
        swap = tamper == ("swap_xa_xg_yr", t)
        relx = lambda x, q: (lam * lam - xa - xg - x
                             - (1 - op) * (xa - xg) + R3P * P - q * P == 0)
        if not pin(relx, xr_s, f"k={k} t={t} xR"):
            return None
        xr = xr_s
        ysrc = xg if swap else xa
        rely = lambda x, q: lam * (ysrc - xr) - ya - x + R3P * P - q * P == 0
        if not pin(rely, yr_s, f"k={k} t={t} yR", "(tampered)" if swap else ""):
            return None
        xa, ya = xr, yr_s
    return xa


def main():
    t0 = time.time()
    ks = [1, 2, 3, 5, 6, 7, 11, 21, 33, 42, 45, 63]
    points = [GEN_X,
              ec_ref.x_only_mul_ints(GEN_X, 2),
              ec_ref.x_only_mul_ints(GEN_X, 7)]
    all_ok = True
    for xg in points:
        for k in ks:
            finals = set()
            for sign in (0, 1):
                f = run_chain(xg, k, sign)
                if f is None:
                    all_ok = False
                else:
                    finals.add(f)
            expect = ec_ref.x_only_mul_ints(xg, k)
            if finals != {expect}:
                all_ok = False
                report(f"L7 pin [xG=…{xg % 10**6} k={k}]", "FAIL",
                       f"classes {sorted(finals)} vs oracle {expect}")
    if all_ok:
        report("L7 end-to-end pinning vs oracle", "PROVED",
               f"{len(points)} points × {len(ks)} scalars × 2 sign classes; "
               f"{n_queries} linear UNSAT queries all unsat; both sign classes "
               f"converge; drain (XR_SUB_P) unique == oracle; {time.time()-t0:.0f}s")
    print(f"queries: {n_queries}, unsat: {n_unsat}")
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
