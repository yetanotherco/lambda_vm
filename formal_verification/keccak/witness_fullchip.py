"""
The forgery as a COMPLETE KECCAK_RND row, not a lane in isolation.

necessity_rho.py answers the algebra for one source lane. The fair objection is
that a lane is a fragment: show that a whole row of the chip accepts, with every
one of its 140 constraints and every one of its ByteAlu/AreBytes operands
satisfied, and still emits a state that is not Keccak-f.

That is what this builds, for the tamper "the rho ARE_BYTES pair stops covering
rot_left" — which is a ONE-LINE edit in `bus_interactions()` (the pair's first
BusValue changed from cols::rot_left to cols::rot_right) and therefore leaves
the interaction count, the column count and the constraint count untouched.

Reachability: the input is a real message state — all zeros except one lane at
0xFFFF...FF. That lane comes straight from the absorbed block, so this is
round 0 of a permutation an attacker can request.
"""
from keccak_ref import RHO, RC, keccak_round
from field_model import (P, as_field, honest_shift, identity_holds, deviate,
                         rho_pi_offsets, rho_operand_bytes, theta_carry_source,
                         theta_operand_bytes, is_byte, THETA_RNC)

FAIL = []


def check(cond, msg):
    print(f"  {'OK  ' if cond else 'FALLA'} {msg}")
    if not cond:
        FAIL.append(msg)


ALL_ONES = (1 << 64) - 1
ROUND = 0
state = [0] * 25
state[0] = ALL_ONES                       # one lane of the absorbed block
lanes = [[state[x + 5 * y] for y in range(5)] for x in range(5)]


def to_bytes(v):
    return [(v >> (8 * b)) & 0xFF for b in range(8)]


# ---------------------------------------------------------------- honest row
S = [[to_bytes(lanes[x][y]) for y in range(5)] for x in range(5)]
cxz = [[[0] * 8 for _ in range(4)] for _ in range(5)]
for x in range(5):
    for b in range(8):
        cxz[x][0][b] = S[x][0][b] ^ S[x][1][b]
    for st in range(1, 4):
        for b in range(8):
            cxz[x][st][b] = cxz[x][st - 1][b] ^ S[x][st + 1][b]

cxz_left = [[0] * 8 for _ in range(5)]
cxz_right = [[0] * 4 for _ in range(5)]
for x in range(5):
    for h in range(4):
        inp = cxz[x][3][2 * h] | (cxz[x][3][2 * h + 1] << 8)
        L, R = honest_shift(inp, THETA_RNC)
        cxz_left[x][2 * h], cxz_left[x][2 * h + 1] = L & 0xFF, L >> 8
        cxz_right[x][h] = R

dxz = [[0] * 8 for _ in range(5)]
for x in range(5):
    rc_bytes = theta_operand_bytes(cxz_left[(x + 1) % 5], cxz_right[(x + 1) % 5])
    for b in range(8):
        dxz[x][b] = cxz[(x + 4) % 5][3][b] ^ rc_bytes[b]

theta = [[[S[x][y][b] ^ dxz[x][b] for b in range(8)] for y in range(5)] for x in range(5)]
theta_lane = [[sum(theta[x][y][b] << (8 * b) for b in range(8)) for y in range(5)] for x in range(5)]

rot_left = [[[0] * 8 for _ in range(5)] for _ in range(5)]
rot_right = [[[0] * 8 for _ in range(5)] for _ in range(5)]
for x in range(5):
    for y in range(5):
        rnc = RHO[x][y] % 16
        for h in range(4):
            inp = theta[x][y][2 * h] | (theta[x][y][2 * h + 1] << 8)
            L, R = honest_shift(inp, rnc)
            rot_left[x][y][2 * h], rot_left[x][y][2 * h + 1] = L & 0xFF, L >> 8
            rot_right[x][y][2 * h], rot_right[x][y][2 * h + 1] = R & 0xFF, R >> 8

# --------------------------------------------------- pick a saturated source lane
saturated = [(x, y) for x in range(5) for y in range(5) if theta_lane[x][y] == ALL_ONES]
check(bool(saturated), f"the message state reaches theta = 0xFFFF...FF on {len(saturated)} lanes: {saturated}")
TX, TY = saturated[0]
print(f"       tampering source lane ({TX},{TY}), RHO={RHO[TX][TY]}")

# --------------------------------------------------------------- forge that lane
rnc, rbc = RHO[TX][TY] % 16, RHO[TX][TY] // 16
in_hws = [theta[TX][TY][2 * h] | (theta[TX][TY][2 * h + 1] << 8) for h in range(4)]
dev = [deviate(*honest_shift(in_hws[h], rnc), 1) for h in range(4)]
f_right = [b for h in range(4) for b in (dev[h][1] & 0xFF, dev[h][1] >> 8)]
f_left = [-f_right[(w - 2) % 8] for w in range(8)]
rot_left[TX][TY] = f_left
rot_right[TX][TY] = f_right


def pi(X, Y, z):
    sx, sy = (X + 3 * Y) % 5, X
    a = rho_pi_offsets(RHO[sx][sy] // 16)
    return rot_left[sx][sy][(z + a) % 8] + rot_right[sx][sy][(z + a - 2) % 8]


# --------------------------------------------------- rebuild chi / iota downstream
chi_ands = [[[0] * 8 for _ in range(5)] for _ in range(5)]
chi = [[[0] * 8 for _ in range(5)] for _ in range(5)]
for x in range(5):
    for y in range(5):
        for b in range(8):
            p0, p1, p2 = pi(x, y, b), pi((x + 1) % 5, y, b), pi((x + 2) % 5, y, b)
            chi_ands[x][y][b] = (255 - as_field(p1) % 256) & (as_field(p2) % 256)
            chi[x][y][b] = (as_field(p0) % 256) ^ chi_ands[x][y][b]
iota = [chi[0][0][b] ^ to_bytes(RC[ROUND])[b] for b in range(8)]

# ------------------------------------------------------------------ verify the row
viol = []
for x in range(5):                                          # 20 IS_BIT + 20 theta
    for h in range(4):
        if cxz_right[x][h] not in (0, 1):
            viol.append(f"IS_BIT x={x} h={h}")
        inp = cxz[x][3][2 * h] | (cxz[x][3][2 * h + 1] << 8)
        if not identity_holds(inp, THETA_RNC,
                              cxz_left[x][2 * h] + 256 * cxz_left[x][2 * h + 1],
                              cxz_right[x][h]):
            viol.append(f"theta identity x={x} h={h}")
for x in range(5):                                          # 100 rho identities
    for y in range(5):
        r = RHO[x][y] % 16
        for h in range(4):
            inp = theta[x][y][2 * h] | (theta[x][y][2 * h + 1] << 8)
            if not identity_holds(inp, r,
                                  rot_left[x][y][2 * h] + 256 * rot_left[x][y][2 * h + 1],
                                  rot_right[x][y][2 * h] + 256 * rot_right[x][y][2 * h + 1]):
                viol.append(f"rho identity ({x},{y}) h={h}")
check(not viol, f"all 140 shipped constraints satisfied ({len(viol)} violations)")

opviol = [f"pi({x},{y},{b})" for x in range(5) for y in range(5) for b in range(8)
          if not is_byte(pi(x, y, b))]
opviol += [f"rotated_C({x},{b})" for x in range(5)
           for b, v in enumerate(theta_operand_bytes(cxz_left[x], cxz_right[x])) if not is_byte(v)]
check(not opviol, f"every ByteAlu operand is a byte, so every lookup matches ({len(opviol)} bad)")

kept = [f"({x},{y})" for x in range(5) for y in range(5) for b in range(8)
        if not is_byte(rot_right[x][y][b])]
check(not kept, "rot_right stays byte-valued everywhere — the surviving check accepts it")
oor = sum(1 for x in range(5) for y in range(5) for b in range(8)
          if as_field(rot_left[x][y][b]) > 255)
check(oor > 0, f"{oor} of 200 rot_left columns hold non-bytes — only the DROPPED check would object")

# ------------------------------------------------------------------- vs FIPS-202
ref = keccak_round(state, RC[ROUND])
got = [0] * 25
for x in range(5):
    for y in range(5):
        bs = iota if (x, y) == (0, 0) else chi[x][y]
        got[x + 5 * y] = sum(bs[b] << (8 * b) for b in range(8))
wrong = [(i % 5, i // 5) for i in range(25) if got[i] != ref[i]]
check(bool(wrong), f"{len(wrong)} of 25 output lanes differ from FIPS-202: {wrong}")

print("\n=== VERDICT ===")
print("  A one-line change to the rho ARE_BYTES pair yields a complete, reachable")
print("  KECCAK_RND row with 0 constraint violations, every lookup matching, and a")
print("  wrong permutation output. Interaction/column/constraint counts unchanged.")
assert not FAIL, FAIL
print("\nFULL-CHIP WITNESS VERIFIED")
