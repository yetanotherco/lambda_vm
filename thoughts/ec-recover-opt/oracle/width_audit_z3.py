"""z3 confirmation of the width audit's interval step.

`width_audit.py` bounds `S_i` by interval arithmetic and cross-checks with random
corners. Corners can miss; this script machine-checks the step that matters.

Two claims per relation and limb index `i`:

  W1 (interval soundness)  for all byte limbs and bit op/mu, `S_i` lies inside
      the interval `width_audit.s_interval` computes.  Expect UNSAT on the
      negation.

  W2 (no wraparound)       for all byte limbs, bit op/mu, and carries inside
      their IsHalfword windows, `|256*c_i - c_{i-1} - S_i| < p_g`.  Expect UNSAT
      on the negation. This is the claim the whole integer-lifting argument (L1)
      rests on, stated directly rather than through the interval.

Both are checked with the addend limbs FREE BYTES -- no canonicality assumed --
which is precisely the varying-addend question.

W2 is also checked in a TAMPERED form (N-WIDTH): drop the byte constraint from
the addend's limbs, leaving them free field elements. That must go SAT, showing
the check is non-vacuous and that byte-ness of the addend is load-bearing.

Run:  <venv>/bin/python width_audit_z3.py [max_i]
"""

import sys

from z3 import And, Int, Or, Solver, sat, unsat

from width_audit import (CARRY_OFFSET_LAMBDA, CARRY_OFFSET_XR, CARRY_OFFSET_YR,
                         NLIMB, PP, P_G, RR, carry_window, s_interval)

TIMEOUT_MS = 120_000


def build_s(rel, i, lam, xa, ya, xb, yb, xr, yr, q, op, mu):
    """`S_i` as a z3 term, transcribed from `EcdasConstraints::s_i`
    (prover/src/tables/ecdas.rs:348-397) and its ecdas2 twin."""
    rq = mu * sum(RR[j] * PP[i - j] for j in range(i + 1)) \
        - sum(q[j] * PP[i - j] for j in range(i + 1))

    if rel == "lambda":
        op_branch = ya[i] - yb[i] + sum(lam[j] * (xb[i - j] - xa[i - j])
                                        for j in range(i + 1))
        notop = sum(2 * lam[j] * ya[i - j] - 3 * xa[j] * xa[i - j]
                    for j in range(i + 1))
        return op * op_branch + (1 - op) * notop + rq
    if rel == "xr":
        s = sum(lam[j] * lam[i - j] for j in range(i + 1))
        return s - xa[i] - xb[i] - xr[i] - (1 - op) * (xa[i] - xb[i]) + rq
    if rel == "yr":
        s = sum(lam[j] * (xa[i - j] - xr[i - j]) for j in range(i + 1))
        return s - ya[i] - yr[i] + rq
    if rel == "dinv":
        # proposed: d_inv*(xB - xA) - 1 == 0 (mod p); `lam` plays d_inv
        s = sum(lam[j] * (xb[i - j] - xa[i - j]) for j in range(i + 1))
        return (s - 1 if i == 0 else s) + rq
    raise ValueError(rel)


def limbs(name, n):
    return [Int(f"{name}_{j}") for j in range(n)]


def byte_bounds(vs):
    return [And(v >= 0, v <= 255) for v in vs]


def run(rel, i, offset, check, tamper_addend=False):
    lam, xa, ya = limbs("lam", NLIMB), limbs("xa", NLIMB), limbs("ya", NLIMB)
    xb, yb = limbs("xb", NLIMB), limbs("yb", NLIMB)
    xr, yr, q = limbs("xr", NLIMB), limbs("yr", NLIMB), limbs("q", NLIMB)
    op, mu = Int("op"), Int("mu")

    s = Solver()
    s.set("timeout", TIMEOUT_MS)
    for group in (lam, xa, ya, xr, yr, q):
        s.add(byte_bounds(group))
    if tamper_addend:
        # N-WIDTH: the addend's limbs are only field elements, not bytes.
        for v in xb + yb:
            s.add(And(v >= 0, v < P_G))
    else:
        s.add(byte_bounds(xb + yb))
    s.add(Or(op == 0, op == 1), Or(mu == 0, mu == 1))
    # limbs beyond 32 are structurally zero (values are 32 bytes, quotients 33)
    for v in lam[32:] + xa[32:] + ya[32:] + xb[32:] + yb[32:] + xr[32:] + yr[32:]:
        s.add(v == 0)
    for v in q[33:]:
        s.add(v == 0)

    S = build_s(rel, i, lam, xa, ya, xb, yb, xr, yr, q, op, mu)

    if check == "interval":
        lo, hi = s_interval(rel, i)
        s.add(Or(S < lo, S > hi))
    else:  # "wrap"
        c_lo, c_hi = carry_window(offset)
        ci, cp = Int("c_i"), Int("c_prev")
        if i == 63:
            s.add(ci == 0)          # ColIsZero on the closing carry
        else:
            s.add(And(ci >= c_lo, ci <= c_hi))
        if i == 0:
            s.add(cp == 0)
        else:
            s.add(And(cp >= c_lo, cp <= c_hi))
        expr = 256 * ci - cp - S
        s.add(Or(expr >= P_G, expr <= -P_G))

    return s.check()


def main():
    max_i = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    rels = [("lambda", CARRY_OFFSET_LAMBDA), ("xr", CARRY_OFFSET_XR),
            ("yr", CARRY_OFFSET_YR), ("dinv", CARRY_OFFSET_LAMBDA)]
    idxs = [i for i in (0, 1, 2, 3, 7, 15, 31, 47, 63) if i <= max_i] or [0]

    print(f"z3 {'':2}  limb indices checked: {idxs}   (timeout {TIMEOUT_MS//1000}s each)")
    print()
    bad = 0
    unknown = 0
    for check in ("interval", "wrap"):
        print(f"--- W{'1' if check == 'interval' else '2'} "
              f"({'interval soundness' if check == 'interval' else 'no wraparound mod p_g'}) ---")
        for rel, off in rels:
            verdicts = []
            for i in idxs:
                r = run(rel, i, off, check)
                verdicts.append(f"i={i}:{r}")
                if r == sat:
                    bad += 1
                elif r != unsat:
                    unknown += 1
            print(f"   {rel:8} " + "  ".join(verdicts))
        print()

    print("--- N-WIDTH (negative control: addend limbs NOT byte-constrained) ---")
    ctl_sat = 0
    for rel, off in rels:
        verdicts = []
        for i in (31, 63):
            if i > max_i:
                continue
            r = run(rel, i, off, "wrap", tamper_addend=True)
            verdicts.append(f"i={i}:{r}")
            if r == sat:
                ctl_sat += 1
        if verdicts:
            print(f"   {rel:8} " + "  ".join(verdicts))
    print()
    print(f"   {ctl_sat} of the tampered queries are SAT "
          f"(wraparound becomes reachable once the addend is not byte-bounded)")

    print()
    ok = bad == 0 and unknown == 0 and ctl_sat > 0
    print(f"WIDTH AUDIT z3: {'PASS' if ok else 'FAIL'} "
          f"({bad} soundness violations, {unknown} unknown/timeout, "
          f"{ctl_sat} negative controls SAT)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
