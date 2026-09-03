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

WHY THERE IS A POSITIVE CONTROL. "The output differs from FIPS-202" is also what
a bug in the transcription BELOW produces, so on its own it proves nothing —
misreading one Dxz column, or dropping chi's NOT, both make lanes differ. So the
honest and the forged row come out of the SAME builder, and the honest one is
required to reproduce FIPS-202 exactly, cross-checked against the independent
reference and against the mirror test_dataflow.py already validates.

Reachability: the input is a real message state — all zeros except one lane at
0xFFFF...FF. That lane comes straight from the absorbed block, so this is
round 0 of a permutation an attacker can request.
"""
from keccak_ref import RHO, RC, keccak_round
from model_dataflow import round_dataflow
from field_model import (P, as_field, honest_shift, identity_holds, deviate,
                         rho_pi_offsets, theta_operand_bytes, is_byte, THETA_RNC)

FAIL = []


def check(cond, msg):
    print(f"  {'OK  ' if cond else 'FAIL'} {msg}")
    if not cond:
        FAIL.append(msg)


ALL_ONES = (1 << 64) - 1
ROUND = 0
state = [0] * 25
state[0] = ALL_ONES                       # one lane of the absorbed block


def to_bytes(v):
    return [(v >> (8 * b)) & 0xFF for b in range(8)]


def build_row(state, round_idx, tamper=None):
    """One complete KECCAK_RND row. `tamper=(sx,sy)` forges that source lane's
    rho decomposition with d = +1; `tamper=None` is the honest row. Returns
    `(out_lanes, cols)` — every committed column the audit needs."""
    lanes = [[state[x + 5 * y] for y in range(5)] for x in range(5)]
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
            left, right = honest_shift(inp, THETA_RNC)
            cxz_left[x][2 * h], cxz_left[x][2 * h + 1] = left & 0xFF, left >> 8
            cxz_right[x][h] = right

    dxz = [[0] * 8 for _ in range(5)]
    for x in range(5):
        rc_bytes = theta_operand_bytes(cxz_left[(x + 1) % 5], cxz_right[(x + 1) % 5])
        for b in range(8):
            dxz[x][b] = cxz[(x + 4) % 5][3][b] ^ rc_bytes[b]

    theta = [[[S[x][y][b] ^ dxz[x][b] for b in range(8)] for y in range(5)] for x in range(5)]

    rot_left = [[[0] * 8 for _ in range(5)] for _ in range(5)]
    rot_right = [[[0] * 8 for _ in range(5)] for _ in range(5)]
    for x in range(5):
        for y in range(5):
            rnc = RHO[x][y] % 16
            for h in range(4):
                inp = theta[x][y][2 * h] | (theta[x][y][2 * h + 1] << 8)
                left, right = honest_shift(inp, rnc)
                rot_left[x][y][2 * h], rot_left[x][y][2 * h + 1] = left & 0xFF, left >> 8
                rot_right[x][y][2 * h], rot_right[x][y][2 * h + 1] = right & 0xFF, right >> 8

    if tamper is not None:
        # d = +1 on all four halfwords, with the byte split of left' chosen so
        # every pi byte of the reader cancels to zero (necessity_rho.py, C).
        tx, ty = tamper
        rnc = RHO[tx][ty] % 16
        in_hws = [theta[tx][ty][2 * h] | (theta[tx][ty][2 * h + 1] << 8) for h in range(4)]
        dev = [deviate(*honest_shift(in_hws[h], rnc), 1) for h in range(4)]
        f_right = [b for h in range(4) for b in (dev[h][1] & 0xFF, dev[h][1] >> 8)]
        rot_right[tx][ty] = f_right
        rot_left[tx][ty] = [-f_right[(w - 2) % 8] for w in range(8)]

    def pi(X, Y, z):
        sx, sy = (X + 3 * Y) % 5, X
        a = rho_pi_offsets(RHO[sx][sy] // 16)
        return rot_left[sx][sy][(z + a) % 8] + rot_right[sx][sy][(z + a - 2) % 8]

    chi_ands = [[[0] * 8 for _ in range(5)] for _ in range(5)]
    chi = [[[0] * 8 for _ in range(5)] for _ in range(5)]
    for x in range(5):
        for y in range(5):
            for b in range(8):
                p0, p1, p2 = pi(x, y, b), pi((x + 1) % 5, y, b), pi((x + 2) % 5, y, b)
                chi_ands[x][y][b] = (255 - as_field(p1) % 256) & (as_field(p2) % 256)
                chi[x][y][b] = (as_field(p0) % 256) ^ chi_ands[x][y][b]
    iota = [chi[0][0][b] ^ to_bytes(RC[round_idx])[b] for b in range(8)]

    out = [0] * 25
    for x in range(5):
        for y in range(5):
            bs = iota if (x, y) == (0, 0) else chi[x][y]
            out[x + 5 * y] = sum(bs[b] << (8 * b) for b in range(8))
    return out, dict(cxz=cxz, cxz_left=cxz_left, cxz_right=cxz_right, theta=theta,
                     rot_left=rot_left, rot_right=rot_right, pi=pi)


def constraint_violations(c):
    """The 140 shipped constraints: 20 IS_BIT + 20 theta + 100 rho identities."""
    viol = []
    for x in range(5):
        for h in range(4):
            if c["cxz_right"][x][h] not in (0, 1):
                viol.append(f"IS_BIT x={x} h={h}")
            inp = c["cxz"][x][3][2 * h] | (c["cxz"][x][3][2 * h + 1] << 8)
            if not identity_holds(inp, THETA_RNC,
                                  c["cxz_left"][x][2 * h] + 256 * c["cxz_left"][x][2 * h + 1],
                                  c["cxz_right"][x][h]):
                viol.append(f"theta identity x={x} h={h}")
    for x in range(5):
        for y in range(5):
            rnc = RHO[x][y] % 16
            for h in range(4):
                inp = c["theta"][x][y][2 * h] | (c["theta"][x][y][2 * h + 1] << 8)
                if not identity_holds(inp, rnc,
                                      c["rot_left"][x][y][2 * h] + 256 * c["rot_left"][x][y][2 * h + 1],
                                      c["rot_right"][x][y][2 * h] + 256 * c["rot_right"][x][y][2 * h + 1]):
                    viol.append(f"rho identity ({x},{y}) h={h}")
    return viol


def operand_violations(c):
    bad = [f"pi({x},{y},{b})" for x in range(5) for y in range(5) for b in range(8)
           if not is_byte(c["pi"](x, y, b))]
    bad += [f"rotated_C({x},{b})" for x in range(5)
            for b, v in enumerate(theta_operand_bytes(c["cxz_left"][x], c["cxz_right"][x]))
            if not is_byte(v)]
    return bad


def reader_lanes(sx, sy):
    """The output lanes that can move when source lane (sx,sy) is forged.

    pi is a bijection, so exactly one pi lane (X,Y) reads (sx,sy) — X = sy and
    Y = 2*(sx - sy) mod 5, since 3*2 = 1 mod 5 — and chi at (x,y) reads
    pi(x), pi(x+1), pi(x+2), so the movable outputs are (X-k, Y), k in 0..2."""
    X = sy
    Y = (2 * (sx - sy)) % 5
    return {((X - k) % 5, Y) for k in range(3)}


# ------------------------------------------------------------- positive control
ref = keccak_round(state, RC[ROUND])
honest, cols = build_row(state, ROUND)
check(honest == ref, "POSITIVE CONTROL: the untampered row is EXACTLY FIPS-202")
check(honest == round_dataflow(state, ROUND),
      "and equals model_dataflow's mirror, the one test_dataflow.py validates")
check(not constraint_violations(cols) and not operand_violations(cols),
      "the honest row satisfies all 140 constraints and every operand")

# --------------------------------------------------------- the forgeable lanes
theta_lane = [[sum(cols["theta"][x][y][b] << (8 * b) for b in range(8))
               for y in range(5)] for x in range(5)]
saturated = [(x, y) for x in range(5) for y in range(5) if theta_lane[x][y] == ALL_ONES]
check(bool(saturated), f"the message state reaches theta = 0xFFFF...FF on {len(saturated)} lanes")
rotated = [l for l in saturated if RHO[l[0]][l[1]] % 16]
check(bool(rotated), f"{len(rotated)} of them have a NON-ZERO rotation: {rotated}")

print("\n=== every saturated lane, forged in turn ===")
for (tx, ty) in saturated:
    got, c = build_row(state, ROUND, tamper=(tx, ty))
    wrong = {(i % 5, i // 5) for i in range(25) if got[i] != ref[i]}
    ok = (not constraint_violations(c)
          and not operand_violations(c)
          and all(is_byte(v) for x in range(5) for y in range(5) for v in c["rot_right"][x][y])
          and sum(1 for b in c["rot_left"][tx][ty] if as_field(b) > 255) > 0
          and wrong
          and wrong <= reader_lanes(tx, ty))
    check(ok, f"lane ({tx},{ty}) RHO={RHO[tx][ty]:2d}: 0 violations, rot_right all bytes, "
              f"{sum(1 for b in c['rot_left'][tx][ty] if as_field(b) > 255)}/8 rot_left out of range, "
              f"output lanes {sorted(wrong)} wrong (readers {sorted(reader_lanes(tx, ty))})")

print("\n=== VERDICT ===")
print("  A one-line change to the rho ARE_BYTES pair yields a complete, reachable")
print("  KECCAK_RND row with 0 constraint violations, every lookup matching, and a")
print(f"  wrong permutation output — on all {len(saturated)} saturated lanes, {len(rotated)} of them")
print("  with a non-zero rotation. Interaction/column/constraint counts unchanged.")
assert not FAIL, FAIL
print("\nFULL-CHIP WITNESS VERIFIED")
