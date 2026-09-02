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

Four configurations, dropping (ii) and/or (iii). Board: A/B/C sound, D forgeable.
"""
from itertools import product
from field_model import (P, MASK16, THETA_RNC, as_field, honest_shift,
                         identity_holds, deviate, theta_carry_source,
                         theta_operand_bytes, is_byte)

FAIL = []


def check(cond, msg):
    print(f"  {'OK  ' if cond else 'FALLA'} {msg}")
    if not cond:
        FAIL.append(msg)


print("=== premise the whole theta argument rests on ===")
# rnc = 1, so left = (in << 1) mod 2**16 is ALWAYS EVEN. This is what kills
# d = +1 in configuration B, and it is specific to a shift by one.
evens = all(honest_shift(i, THETA_RNC)[0] % 2 == 0 for i in range(1 << 16))
check(evens, "left = (in<<1) mod 2**16 is even for all 2**16 inputs")
# and the carry never needs more than a bit: in*2 < 2**17
carries = {honest_shift(i, THETA_RNC)[1] for i in range(1 << 16)}
check(carries <= {0, 1}, f"the honest carry only ever takes {sorted(carries)} -> one bit suffices")

print("\n=== A: shipped (both checks) — d = 0 forced ===")
# L' = L - 2**16*d with L, L' both in [0, 2**16) forces |2**16 d| <= 65535 < 2**16.
check(2**16 > MASK16, "|2**16 * d| >= 2**16 > 65535 for any d != 0, so ARE_BYTES alone pins d = 0")

print("\n=== C: IS_BIT dropped, ARE_BYTES kept — d = 0 forced, IS_BIT is IMPLIED ===")
check(2**16 > MASK16, "same bound: ARE_BYTES on left pins d = 0, hence right = honest quotient in {0,1}")

print("\n=== B: ARE_BYTES dropped, IS_BIT kept — exhaustive over every reachable d ===")
# IS_BIT keeps right in {0,1}, so d = right' - right lies in {-1,0,1}: the search
# space is finite and the enumeration below is complete. No solver needed.
#
# With no range check on Cxz_left, the only bound left on L' is the operand:
#   L' = lo + 256*hi,  hi in [0,255]        (hi IS the odd operand byte)
#                      lo in [-r', 255-r']  (lo + carry must be a byte)
# so L' ranges over [-r'_prev, 65535 - r'_prev] -- a window of width 2**16.
def theta_operand_window(right_prev):
    """The interval L' may occupy when its own range check is gone."""
    return -right_prev, 65535 - right_prev


def config_b_survivor(in_hws):
    """Return the first d != 0 that satisfies IS_BIT and every operand window."""
    honest = [honest_shift(i, THETA_RNC) for i in in_hws]
    for d in product((-1, 0, 1), repeat=4):
        if not any(d):
            continue
        dev = [deviate(honest[h][0], honest[h][1], d[h]) for h in range(4)]
        if any(not 0 <= right <= 1 for _, right in dev):
            continue                                    # IS_BIT rejects it
        lo_hi = [theta_operand_window(dev[theta_carry_source(h)][1]) for h in range(4)]
        if all(lo <= dev[h][0] <= hi for h, (lo, hi) in enumerate(lo_hi)):
            return in_hws, d, dev
    return None


survivors = [config_b_survivor(c) for c in
             ([0xFFFF] * 4, [0] * 4, [0xAAAA, 0x5555, 0xFFFF, 0x0001], [0x8000] * 4)]
check(not any(survivors),
      "no d != 0 survives IS_BIT + the operand windows")
print("       why: d=+1 needs L' = L - 2**16 >= -r'_prev >= -1, i.e. L = 65535 -- but L is")
print("            EVEN (shift by one), so that is unreachable; d=-1 needs L' > 65535.")

print("\n=== D: both dropped — FORGEABLE, explicit witness ===")
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
print(f"       honest  rotated_C = {hon_out}")
print(f"       FORGED  rotated_C = {frg_out}")
print(f"       forged Cxz_left (as field elements) = {[as_field(v) for v in frg_left[:2]]}...")
print(f"       forged Cxz_right = {frg_right}   (2 is not a bit -> IS_BIT would reject)")

# generality: the four carries form a cycle, so an arbitrary target is reachable
det = (2**16) ** 4 - 1
check(det % P != 0, f"det(2**16*I - S) = 2**64-1 = {det % P} mod p is invertible -> ANY target output")

print("\n=== VERDICT ===")
print("  A sound | B sound (ARE_BYTES alone is redundant) | C sound (IS_BIT alone is redundant)")
print("  D FORGEABLE -> the PAIR is load-bearing; neither check is, on its own.")
assert not FAIL, FAIL
print("\nALL THETA NECESSITY CHECKS PASSED")
