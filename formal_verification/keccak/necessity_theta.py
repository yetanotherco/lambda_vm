"""
Is each theta range check NECESSARY for the shipped inline identity?

Shipped constraints (KeccakRndConstraints::eval, groups (1) and (2), plus the
banner "Theta: ARE_BYTES range checks on Cxz_left (20 pairs)"):

    (i)   mu * (in*2 - right*2**16 - left) = 0          20 identities
    (ii)  mu * right*(1 - right) = 0                    20 IS_BIT on the carry
    (iii) ARE_BYTES(Cxz_left[2i], Cxz_left[2i+1])       20 pairs

Downstream, rotated_C is a ByteAlu OPERAND (banner "Theta: Dxz BYTE_ALU[XOR]
(40)"), so it must be a byte:

    rotated_C[2h]   = Cxz_left[2h]   + Cxz_right[(h-1) % 4]      (carry lands here)
    rotated_C[2h+1] = Cxz_left[2h+1]                             (no carry)

HOW A CONFIGURATION IS DECIDED. Dropping a check does not leave its column
free: whichever of the two the operand still reads alongside a bounded summand
keeps a window (`operand_summand_window`, premise section 6 of
combinatorics.py). So each configuration is a pair of intervals, and
`surviving_deviation` sweeps all 2**16 input halfwords against them --
completely, not by sampling. Configuration D is the exception: both columns of
the same operand lose their check, the operand then bounds only their SUM,
there is no per-column window, and the explicit witness below is what decides
it.

Board: A/B/C sound, D forgeable.
"""
from combinatorics import premises
from field_model import (P, THETA_RNC, BYTE, BIT, as_field, honest_shift,
                         identity_holds, deviate, difference_form_is_exact,
                         operand_summand_window, packed_pair_bounds,
                         surviving_deviation, theta_operand_bytes, is_byte)

premises(verbose=False)

FAIL = []


def check(cond, msg):
    print(f"  {'OK  ' if cond else 'FAIL'} {msg}")
    if not cond:
        FAIL.append(msg)


# The window each column occupies per configuration, derived from the contracts:
#   Cxz_left  ARE_BYTES pair, or -- with that gone -- the Dxz operand, whose low
#             byte carries Cxz_right and whose high byte does not.
#   Cxz_right IS_BIT, or -- with that gone -- the Dxz operand alongside a
#             range-checked Cxz_left byte.
LEFT_CHECKED = packed_pair_bounds(BYTE, BYTE)
LEFT_OPERAND_ONLY = packed_pair_bounds(operand_summand_window(BIT), BYTE)
RIGHT_CHECKED = BIT
RIGHT_OPERAND_ONLY = operand_summand_window(BYTE)

CONFIGS = [
    ("A: shipped, both checks", LEFT_CHECKED, RIGHT_CHECKED),
    ("B: ARE_BYTES dropped, IS_BIT kept", LEFT_OPERAND_ONLY, RIGHT_CHECKED),
    ("C: IS_BIT dropped, ARE_BYTES kept", LEFT_CHECKED, RIGHT_OPERAND_ONLY),
]

print("=== the two facts that make theta's windows what they are ===")
# rnc = 1, so left = (in << 1) mod 2**16 is ALWAYS EVEN. This is what kills
# d = +1 in configuration B, and it is specific to a shift by one.
evens = all(honest_shift(i, THETA_RNC)[0] % 2 == 0 for i in range(1 << 16))
check(evens, "left = (in<<1) mod 2**16 is even for all 2**16 inputs")
carries = {honest_shift(i, THETA_RNC)[1] for i in range(1 << 16)}
check(carries <= {0, 1}, f"the honest carry only ever takes {sorted(carries)} -> one bit suffices")

print("\n=== A / B / C: complete sweep over every input halfword ===")
for name, left_b, right_b in CONFIGS:
    check(difference_form_is_exact(THETA_RNC, left_b, right_b),
          f"{name}: every term < p/2, so the field identity IS the integer one")
    surv = surviving_deviation(THETA_RNC, left_b, right_b)
    check(surv is None,
          f"{name}: left in {left_b}, right in {right_b} -> "
          f"{'no deviation survives any of the 2**16 inputs' if surv is None else f'SURVIVOR {surv}'}")
print("       B is the interesting one: d=+1 needs L' = L - 2**16 >= -1, i.e. L = 65535,")
print("       and L is EVEN, so the sweep finds nothing. d=-1 needs L' > 65535.")

print("\n=== D: both dropped — no per-column window exists, so: explicit witness ===")
in_hws = [0xFFFF] * 4                                  # C = 0xFFFF...FF
honest = [honest_shift(i, THETA_RNC) for i in in_hws]
d = (1, 1, 1, 1)
dev = [deviate(honest[h][0], honest[h][1], d[h]) for h in range(4)]

hon_left = [b for h in range(4) for b in (honest[h][0] & 0xFF, honest[h][0] >> 8)]
hon_right = [honest[h][1] for h in range(4)]
frg_left = [b for h in range(4) for b in (dev[h][0], 0)]       # L' = -2 -> (-2, 0)
frg_right = [dev[h][1] for h in range(4)]

hon_out = theta_operand_bytes(hon_left, hon_right)
frg_out = theta_operand_bytes(frg_left, frg_right)

check(all(identity_holds(in_hws[h], THETA_RNC, dev[h][0], dev[h][1]) for h in range(4)),
      "the forged (left, right) satisfies all four shipped identities")
check(all(is_byte(v) for v in frg_out), "every forged rotated_C byte is a byte (ByteAlu accepts it)")
check(any(a != b for a, b in zip(hon_out, frg_out)), "the theta output CHANGES")
check(any(as_field(v) > 255 for v in frg_left), "only Cxz_left holds non-bytes (its check is the one gone)")
check(any(v not in (0, 1) for v in frg_right), "and Cxz_right holds a non-bit (IS_BIT would reject it)")
print(f"       honest  rotated_C = {hon_out}")
print(f"       FORGED  rotated_C = {frg_out}")
print(f"       forged Cxz_left (as field elements) = {[as_field(v) for v in frg_left[:2]]}...")
print(f"       forged Cxz_right = {frg_right}")

# generality: the four carries form a cycle, so an arbitrary target is reachable
det = (2**16) ** 4 - 1
check(det % P != 0, f"det(2**16*I - S) = 2**64-1 = {det % P} mod p is invertible -> ANY target output")

print("\n=== VERDICT ===")
print("  A sound | B sound (ARE_BYTES alone is redundant) | C sound (IS_BIT alone is redundant)")
print("  D FORGEABLE -> the PAIR is load-bearing; neither check is, on its own.")
print("  Note WHAT each half costs: Cxz_left's is 20 ARE_BYTES sends, Cxz_right's is")
print("  20 degree-3 polynomial constraints -- the reason this AIR declares max_degree 3.")
assert not FAIL, FAIL
print("\nALL THETA NECESSITY CHECKS PASSED")
