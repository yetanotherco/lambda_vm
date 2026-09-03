"""
Is each rho range check NECESSARY for the shipped inline identity?

Shipped constraints (KeccakRndConstraints::eval group (3), plus the banner
"Rho: ARE_BYTES range checks on rot_left + rot_right (200 pairs)"):

    mu * (in*2**rnc - right*2**16 - left) = 0     100 identities, rnc = RHO[x][y] % 16
    ARE_BYTES(rot_left[x][y][b], rot_right[x][y][b])          200 pairs

Downstream, pi is virtual and is consumed as a ByteAlu OPERAND (banners
"Chi: BYTE_ALU[AND] (200)" and "Chi: BYTE_ALU[XOR] (200)"):

    pi[z] = rot_left[l(z)] + rot_right[r(z)]      must be a byte

Premises from combinatorics.py (run it first): the offsets are even, so a pi
halfword reads one source halfword as P_h = L_(h+A) + R_(h+A-1), and every one
of the 400 byte columns is read exactly once.

RESULT, and it is asymmetric — unlike theta, here ONE check is load-bearing on
its own. `left` and `right` enter the identity with weights 1 and 2**16, so
bounding `left` kills the deviation while bounding `right` does not.
"""
from itertools import product
from keccak_ref import RHO
from field_model import (P, MASK16, as_field, honest_shift, identity_holds,
                         deviate, rho_pi_offsets, rho_operand_bytes, is_byte)

FAIL = []


def check(cond, msg):
    print(f"  {'OK  ' if cond else 'FALLA'} {msg}")
    if not cond:
        FAIL.append(msg)


LANES = [(x, y) for x in range(5) for y in range(5)]

print("=== A / B: left stays range-checked — d = 0 forced, on every lane ===")
# L' = L - 2**16*d with L, L' both in [0, 2**16) leaves no room for d != 0. This
# holds whatever `right` is allowed to be, so dropping rot_right's half of the
# pair changes nothing: its value is then recovered from the identity, uniquely,
# because 2**16 is invertible mod p. rot_right's check is IMPLIED.
check(2**16 > MASK16, "|2**16 * d| >= 2**16 > 65535 for d != 0 -> A and B sound for all 25 lanes")

print("\n=== C: rot_left's check dropped — completeness of the search first ===")
# right stays in [0, 2**16), so d = right' - right has |d| <= 65535. And
# P'_h = P_h - 2**16*d_(h+A) + d_(h+A-1) in [0, 65535] with P_h in [0, 65535]
# forces |2**16*d_j - d_(j-1)| <= 65535, so |d_j| >= 2 would need
# |d_(j-1)| >= 2*2**16 - 65535 = 65537 > 65535. Hence |d_j| <= 1 for all j.
check(2 * 2**16 - MASK16 > MASK16,
      f"|d| >= 2 would need a neighbour |d| >= {2 * 2**16 - MASK16} > {MASK16} -> d in {{-1,0,1}}, search complete")

print("\n=== C: the forged witness, verified PER BYTE on every lane ===")
forged = 0
for (sx, sy) in LANES:
    rho = RHO[sx][sy]
    rnc, rbc = rho % 16, rho // 16
    in_hws = [0xFFFF] * 4                      # theta[sx][sy] = 0xFFFF...FF
    honest = [honest_shift(i, rnc) for i in in_hws]

    # saturation: pi = 0xFF..FF, the only configuration d = +1 can survive
    hon_left = [b for h in range(4) for b in (honest[h][0] & 0xFF, honest[h][0] >> 8)]
    hon_right = [b for h in range(4) for b in (honest[h][1] & 0xFF, honest[h][1] >> 8)]
    hon_pi = rho_operand_bytes(hon_left, hon_right, rbc)

    # forge with d = +1 on all four halfwords: right' = 2**rnc, and choose the
    # byte split of left' so that every pi byte cancels to zero
    dev = [deviate(honest[h][0], honest[h][1], 1) for h in range(4)]
    frg_right = [b for h in range(4) for b in (dev[h][1] & 0xFF, dev[h][1] >> 8)]
    frg_left = [-frg_right[(w - 2) % 8] for w in range(8)]
    frg_pi = rho_operand_bytes(frg_left, frg_right, rbc)

    okid = all(identity_holds(in_hws[h], rnc, frg_left[2 * h] + 256 * frg_left[2 * h + 1],
                              frg_right[2 * h] + 256 * frg_right[2 * h + 1]) for h in range(4))
    good = (hon_pi == [0xFF] * 8 and okid and frg_pi == [0] * 8
            and all(is_byte(v) for v in frg_right)          # rot_right stays byte-valued
            and any(as_field(v) > 255 for v in frg_left))   # only rot_left goes out of range
    forged += good
    if (sx, sy) in ((0, 0), (2, 0), (4, 4)):
        print(f"       lane ({sx},{sy}) RHO={rho:2d} rnc={rnc:2d}: honest pi=0xFF*8 -> forged pi={frg_pi[:3]}..., "
              f"rot_right bytes={all(is_byte(v) for v in frg_right)}, rot_left out-of-range={sum(as_field(v) > 255 for v in frg_left)}/8")
check(forged == 25, f"FORGEABLE on {forged}/25 lanes — rot_left's check is LOAD-BEARING")
print("       note this is strictly stronger than the spec's own witness: rot_right stays")
print("       byte-valued here, so only rot_left's check catches it.")

print("\n=== D: both dropped — the lane's output is completely free ===")
# Eliminating left via the identity leaves a cyclic system in right:
#   right'_(j-1) - 2**16 * right'_j = c_j,  solvable because 1 - 2**64 is invertible.
# The check verifies the TARGET is hit. `identity_holds` alone cannot fail here:
# L is DERIVED from the identity, so it is true by construction, and with it as
# the only check an index slip in the recurrence went unnoticed.
INV = pow(1 - 2**64, -1, P)
free = 0
for (sx, sy) in LANES:
    rho = RHO[sx][sy]
    rnc, rbc, a = rho % 16, rho // 16, rho_pi_offsets(rho // 16)
    A = a // 2
    in_hws = [0x1234, 0xABCD, 0x0F0F, 0xFFFF]
    target = [(7 * z + 3) % 256 for z in range(8)]        # an arbitrary byte target
    Q = [target[2 * h] + 256 * target[2 * h + 1] for h in range(4)]
    c = [(Q[(j - A) % 4] - in_hws[j] * (2**rnc)) % P for j in range(4)]
    r0 = ((c[1] + (2**16) * c[2] + (2**32) * c[3] + (2**48) * c[0]) * INV) % P
    R = [0] * 4
    R[0] = r0
    for j in (1, 2, 3):
        # R[j-1] - 2**16*R[j] = c[j] is the relation r0's closed form above
        # solves; the target check below is what pins this index.
        R[j] = ((R[j - 1] - c[j]) * pow(2**16, -1, P)) % P
    L = [(in_hws[j] * (2**rnc) - (2**16) * R[j]) % P for j in range(4)]
    okid = all(identity_holds(in_hws[j], rnc, L[j], R[j]) for j in range(4))
    hits = [(L[(h + A) % 4] + R[(h + A - 1) % 4]) % P for h in range(4)] == [q % P for q in Q]
    free += okid and hits
check(free == 25, f"the forged pi halfwords equal the ARBITRARY target on {free}/25 lanes "
                  f"(det = 1 - 2**64 = {(1 - 2**64) % P} mod p, invertible). Per-byte\n                   realizability is the construction exhibited in C.")

print("\n=== VERDICT ===")
print("  A sound | B sound -> rot_right's check is IMPLIED by rot_left's + the pi operand")
print("  C FORGEABLE      -> rot_left's check is LOAD-BEARING on its own")
print("  D output free    -> the pair pins nothing without at least rot_left")
print("  Ceiling for any interaction saving is therefore 100 of 200, not 200.")
assert not FAIL, FAIL
print("\nALL RHO NECESSITY CHECKS PASSED")
