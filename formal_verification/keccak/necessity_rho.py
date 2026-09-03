"""
Is each rho range check NECESSARY for the shipped inline identity?

Shipped constraints (KeccakRndConstraints::eval group (3), plus the banner
"Rho: ARE_BYTES range checks on rot_left + rot_right (200 pairs)"):

    mu * (in*2**rnc - right*2**16 - left) = 0     100 identities, rnc = RHO[x][y] % 16
    ARE_BYTES(rot_left[x][y][b], rot_right[x][y][b])          200 pairs

Downstream, pi is virtual and is consumed as a ByteAlu OPERAND (banners
"Chi: BYTE_ALU[AND] (200)" and "Chi: BYTE_ALU[XOR] (200)"):

    pi[z] = rot_left[l(z)] + rot_right[r(z)]      must be a byte

so dropping one column's check does not free it: the operand still confines it
(`operand_summand_window`), which is what makes configurations A and B sound and
is the step a bound-vs-bound comparison cannot express. Premises live in
combinatorics.py and are imported below, not left to be run by hand: the
offsets are even, so a pi halfword reads one source halfword as
P_h = L_(h+A) + R_(h+A-1), and every one of the 400 byte columns is read exactly
once -- a column read twice would need the intersection of two windows.

RESULT, and it is asymmetric — unlike theta, here ONE check is load-bearing on
its own. `left` and `right` enter the identity with weights 1 and 2**16, so
bounding `left` kills the deviation while bounding `right` does not.
"""
from keccak_ref import RHO
from combinatorics import premises
from field_model import (P, BYTE, as_field, honest_shift, identity_holds,
                         deviate, difference_form_is_exact, operand_summand_window,
                         packed_pair_bounds, rho_pi_offsets, rho_operand_bytes,
                         surviving_deviation, is_byte)

premises(verbose=False)

FAIL = []


def check(cond, msg):
    print(f"  {'OK  ' if cond else 'FAIL'} {msg}")
    if not cond:
        FAIL.append(msg)


LANES = [(x, y) for x in range(5) for y in range(5)]
RNCS = sorted({RHO[x][y] % 16 for (x, y) in LANES})

# Both halves are byte PAIRS here (unlike theta's single carry column), so each
# window is the packed span of two per-byte windows.
CHECKED = packed_pair_bounds(BYTE, BYTE)
OPERAND_ONLY = packed_pair_bounds(operand_summand_window(BYTE), operand_summand_window(BYTE))

print(f"=== A / B: left stays range-checked — complete sweep, all {len(RNCS)} distinct rotations ===")
# Dropping rot_right's check leaves it the pi operand window; the sweep then
# shows the honest pair is the only one, so rot_right's check is IMPLIED.
for name, left_b, right_b in (("A: both checked", CHECKED, CHECKED),
                              ("B: rot_right's check dropped", CHECKED, OPERAND_ONLY)):
    surv = [(rnc, surviving_deviation(rnc, left_b, right_b)) for rnc in RNCS]
    check(all(difference_form_is_exact(rnc, left_b, right_b) for rnc in RNCS),
          f"{name}: every term < p/2, so the field identity IS the integer one")
    check(all(s is None for _, s in surv),
          f"{name}: left in {left_b}, right in {right_b} -> pinned on all "
          f"{len(RNCS)} rotations x 2**16 inputs"
          f"{'' if all(s is None for _, s in surv) else f' — SURVIVORS {[s for s in surv if s[1]][:2]}'}")

print("\n=== C: rot_left's check dropped — the sweep already says forgeable ===")
surv_c = {rnc: surviving_deviation(rnc, OPERAND_ONLY, CHECKED) for rnc in RNCS}
check(all(s is not None for s in surv_c.values()),
      f"a deviation survives on all {len(RNCS)} rotations, e.g. rnc={RNCS[0]} -> {surv_c[RNCS[0]]}")
check(all(d == 1 for _, d in surv_c.values()),
      "and it is d = +1 every time -> the witness below is the general shape, not a special case")

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
# No per-column window survives (the operand bounds only the SUM of two
# unchecked columns), so eliminating left via the identity leaves a cyclic
# system in right: right'_(j-1) - 2**16 * right'_j = c_j, solvable because
# 1 - 2**64 is invertible. The check below verifies the TARGET is hit, not just
# that the identity holds -- the identity holds by construction, since L is
# derived from it.
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
                  f"(det = 1 - 2**64 = {(1 - 2**64) % P} mod p, invertible). Per-byte\n"
                  f"                   realizability is the construction exhibited in C.")

print("\n=== VERDICT ===")
print("  A sound | B sound -> rot_right's check is IMPLIED by rot_left's + the pi operand")
print("  C FORGEABLE      -> rot_left's check is LOAD-BEARING on its own")
print("  D output free    -> the pair pins nothing without at least rot_left")
print("  Ceiling for any interaction saving is therefore 100 of 200, not 200.")
assert not FAIL, FAIL
print("\nALL RHO NECESSITY CHECKS PASSED")
