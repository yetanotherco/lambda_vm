"""Probe: is the incomplete-addition edge really closed by the NUMS blind?

DESIGN.md section 4 introduces the T0 blind and argues:

    "Every intermediate accumulator is 2^j*T0 + (c1*P1 + c2*P2); a collision now
     implies a known linear relation on dlog_G(T0) -- i.e. the attacker computes
     a discrete log nobody knows."

This script tests that argument constructively, and it does NOT hold: the
prover chooses P2 (for ecrecover, P2 = lift_x(r) for a signature component `r`
it picks freely), so it can set P2 = mu*T0 for a mu it knows. The collision
equation then has its T0 coefficient cancelled against P2's, and is satisfiable
without knowing dlog_G(T0) at all.

Concretely, with P1 = G and P2 = mu*T0, the accumulator entering the add at
round r is

    acc = alpha*T0 + beta1*G + beta2*P2 = (alpha + mu*beta2)*T0 + beta1*G

(alpha, beta1, beta2 are public functions of the schedule and the scalar bits).
Taking u1 < 2^r forces beta1 = e1 = 0, so acc = addend = P2 reduces to the
single scalar equation

    alpha + mu*(beta2 - 1) == 0   (mod N)      ->   mu = -alpha/(beta2 - 1)

which is one modular inversion. Cost of the whole construction: one scalar
multiplication. It is CHEAPER than the ~2^-j-probability attack on the
UNBLINDED chain that the blind was introduced to prevent.

On such a row the chip's lambda relation reads lambda*(xB - xA) + yA - yB = 0
with xA = xB and yA = yB: it degenerates to 0 = 0, lambda is unconstrained, and
the row's xR/yR become a free one-parameter family. That is a soundness break
(the chip proves Q = u1*P1 + u2*P2 for a Q that is not), independent of how far
an attacker can then steer Q.

Note the honest path is fine: `ecsm::lincomb2_witness` and the Python reference
both detect the collision and report `ResultInfinity` -> status != 0 -> the
guest's software fallback. The hole is on the malicious-prover side, where the
row is hand-crafted rather than generated.

Run:  <venv>/bin/python nums_blinding_probe.py
"""

from ec_ref import GX, GY, N, P, pt_add, pt_double, recover_even_y, scalar_mul
import lincomb2_ref

T0, _T0_COUNTER = lincomb2_ref.t0_ref()
G = (GX, GY)


def chain_coeffs(u1, u2, length):
    """Track acc = alpha*T0 + beta1*P1 + beta2*P2 through the blinded schedule.
    Returns {round: (alpha, beta1, beta2, e1, e2)} captured just BEFORE each add
    (i.e. after that round's doubling)."""
    alpha, b1, b2 = 1, 0, 0
    pre_add = {}
    for r in range(length - 1, -1, -1):
        alpha, b1, b2 = 2 * alpha % N, 2 * b1 % N, 2 * b2 % N
        e1, e2 = (u1 >> r) & 1, (u2 >> r) & 1
        if e1 or e2:
            pre_add[r] = (alpha, b1, b2, e1, e2)
            b1, b2 = (b1 + e1) % N, (b2 + e2) % N
    return pre_add


def solve_mu(length, r_target, u1, u2):
    """The mu making the add at `r_target` degenerate, with P1 = G, P2 = mu*T0."""
    assert (u2 >> r_target) & 1 == 1, "target round must have u2's bit set"
    assert u2.bit_length() == length
    alpha, b1, b2, e1, e2 = chain_coeffs(u1, u2, length)[r_target]
    if b1 % N != e1 % N:
        return None, f"G-coefficient does not vanish (beta1={b1}, e1={e1})"
    denom = (b2 - e2) % N
    if denom == 0:
        return None, "P2-coefficient denominator is zero"
    return (-alpha * pow(denom, N - 2, N)) % N, None


def simulate(u1, u2, p1, p2, length):
    """Run the real blinded schedule; report the first degenerate add."""
    p12 = pt_add(p1, p2) if p1[0] != p2[0] else None
    acc = T0
    for r in range(length - 1, -1, -1):
        acc = pt_double(acc)
        e1, e2 = (u1 >> r) & 1, (u2 >> r) & 1
        if not (e1 or e2):
            continue
        addend = p12 if (e1 and e2) else (p1 if e1 else p2)
        if acc[0] == addend[0]:
            same_y = acc[1] == addend[1]
            return r, same_y, acc, addend
        acc = pt_add(acc, addend)
    return None, None, acc, None


def package_ecrecover(u1, u2, p2):
    """Back out the (z, v, r, s) an attacker would submit, and check the guest's
    own decomposition reproduces (u1, u2) and lifts R back to p2."""
    if p2[0] >= N:
        return None  # would need the recid >= 2 branch
    r = p2[0]
    v = p2[1] & 1
    z = (-u1 * r) % N
    s = (u2 * r) % N
    if not (1 <= s < N and 1 <= r < N):
        return None
    rinv = pow(r, N - 2, N)
    gu1, gu2 = (-(rinv * z)) % N, (rinv * s) % N
    ye = recover_even_y(r)
    lifted = None if ye is None else (r, ye if v == 0 else (P - ye) % P)
    return {"z": z, "v": v, "r": r, "s": s,
            "guest_u1": gu1, "guest_u2": gu2, "guest_R": lifted}


CASES = [
    # (length, target round, u2)   -- u1 is any value < 2^r; 1 is used below
    (8, 3, 0b10001000),
    (12, 5, 0b100000100000),
    (16, 9, 0b1000001000000000),
    (32, 17, (1 << 31) | (1 << 17)),
    (256, 128, (1 << 255) | (1 << 128)),
]


def main():
    print(f"T0 = ({T0[0]:#x}, {T0[1]:#x})")
    print()
    forging = 0
    for length, r_target, u2 in CASES:
        u1 = 1
        mu, why = solve_mu(length, r_target, u1, u2)
        if mu is None:
            print(f"len={length:4d} r={r_target:4d}: no solution ({why})")
            continue
        p2 = scalar_mul(mu, T0)
        if p2[0] == GX:
            print(f"len={length:4d} r={r_target:4d}: P2 = +-G, skipped")
            continue

        hit, same_y, acc, addend = simulate(u1, u2, G, p2, length)
        kind = ("acc == +addend  ->  LAMBDA UNCONSTRAINED (forgeable row)"
                if same_y else
                "acc == -addend  ->  row unsatisfiable (rejects, not a forgery)")
        print(f"len={length:4d} r={r_target:4d} u1={u1} u2={u2:#x}")
        print(f"   mu = {mu:#x}")
        print(f"   P2 = mu*T0 = ({p2[0]:#x},")
        print(f"                 {p2[1]:#x})")
        print(f"   degenerate add at round {hit} (target {r_target}): {kind}")
        print(f"   acc == addend as points: {acc == addend}")

        # the honest generator's verdict on the same input
        try:
            lincomb2_ref.lincomb2_rows(u1, G, u2, p2, T0)
            print("   honest reference: ACCEPTED (unexpected!)")
        except ValueError as e:
            print(f"   honest reference: rejects with {e} -> status != 0 -> guest fallback")

        pkg = package_ecrecover(u1, u2, p2)
        if pkg:
            print(f"   ecrecover packaging: z={pkg['z']:#x}")
            print(f"                        r={pkg['r']:#x}")
            print(f"                        s={pkg['s']:#x}  v={pkg['v']}")
            print(f"      guest recomputes u1: {pkg['guest_u1'] == u1}   "
                  f"u2: {pkg['guest_u2'] == u2}   lifts R == P2: {pkg['guest_R'] == p2}")
        print()
        if hit == r_target and same_y:
            forging += 1

    print(f"forgeable (lambda-free) degenerate adds constructed: {forging}/{len(CASES)}")
    print()
    print("READING: the NUMS blind does NOT close the incomplete-addition edge when")
    print("P2 is prover-chosen. The named dlog assumption on T0 is NECESSARY but not")
    print("SUFFICIENT; an explicit non-degeneracy check on add rows (or an equivalent)")
    print("is required. See ../lincomb2/FINDING-nums-blinding.log.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
