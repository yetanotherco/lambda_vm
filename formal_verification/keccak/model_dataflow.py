"""
Concrete byte-level mirror of the keccak_rnd circuit's contract dataflow.

Every equation here corresponds 1:1 to a bus interaction / eval constraint in
prover/src/tables/keccak_rnd.rs, evaluated FORWARD with concrete ints. Its sole
job is to validate that the byte-level wiring I will hand-encode into z3
actually reproduces the FIPS-202 reference round (guards against a wholesale
wrong model that a symbolic UNSAT could not reveal).

The `bug` flag lets us confirm each negative control genuinely perturbs output.
Line-number citations (prover/src/tables/keccak_rnd.rs) in comments.
"""
from keccak_ref import RHO, RC


def lane_to_bytes(v):
    return [(v >> (8 * b)) & 0xFF for b in range(8)]


def bytes_to_lane(bs):
    return sum(int(bs[b]) << (8 * b) for b in range(8))


def cxz_right_bit_for_byte(b):
    # keccak_rnd.rs:126-132  -> even b: Some((b/2 + 3)%4); odd: None
    return (b // 2 + 3) % 4 if b % 2 == 0 else None


def pi_src_bytes(X, Y, z):
    # keccak_rnd.rs:161-174 pi_src_cols: (sx,sy)=((X+3Y)%5, X), rbc=RHO[sx][sy]//16
    sx = (X + 3 * Y) % 5
    sy = X
    rbc = RHO[sx][sy] // 16
    if rbc == 0:
        l, r = z, (z + 6) % 8
    elif rbc == 1:
        l, r = (z + 6) % 8, (z + 4) % 8
    elif rbc == 2:
        l, r = (z + 4) % 8, (z + 2) % 8
    else:
        l, r = (z + 2) % 8, z
    return sx, sy, l, r


def round_dataflow(start_lanes, r, bug=None):
    """Forward-evaluate one round via the circuit's contract equations.

    start_lanes: list[25] u64. Returns list[25] u64 (out state)."""
    S = [[lane_to_bytes(start_lanes[x + 5 * y]) for y in range(5)] for x in range(5)]
    # index as S[x][y][b]

    # === theta: Cxz XOR chain === keccak_rnd.rs:539-588
    cxz = [[[0] * 8 for _ in range(4)] for _ in range(5)]
    for x in range(5):
        for b in range(8):
            cxz[x][0][b] = S[x][0][b] ^ S[x][1][b]          # :541-559
        for stage in range(1, 4):
            y = stage + 1
            for b in range(8):
                cxz[x][stage][b] = cxz[x][stage - 1][b] ^ S[x][y][b]  # :567-585

    # === theta: HWSL rotate-C-by-1 === keccak_rnd.rs:593-631
    cxz_left = [[0] * 8 for _ in range(5)]
    cxz_right = [[0] * 4 for _ in range(5)]
    for x in range(5):
        for hw in range(4):
            Chw = cxz[x][3][2 * hw] | (cxz[x][3][2 * hw + 1] << 8)   # :600-609 input hw
            left16 = (Chw << 1) & 0xFFFF                             # :613-622 shifted
            cxz_left[x][2 * hw] = left16 & 0xFF
            cxz_left[x][2 * hw + 1] = (left16 >> 8) & 0xFF
            cxz_right[x][hw] = (Chw >> 15) & 1                       # :624 carry bit

    def rotated_c(xp, b):
        # keccak_rnd.rs:322-329 / 663-672 reconstruction
        contrib = 0
        hw = cxz_right_bit_for_byte(b)
        if hw is not None:
            contrib = cxz_right[xp][hw]
        val = cxz_left[xp][b] + contrib
        assert val <= 255, "rotated_C operand exceeds a byte"
        return val

    # === theta: Dxz XOR === keccak_rnd.rs:661-690
    Dxz = [[0] * 8 for _ in range(5)]
    for x in range(5):
        for b in range(8):
            cm1 = cxz[(x + 4) % 5][3][b]                             # C[(x-1)%5]
            rc1 = rotated_c((x + 1) % 5, b)                          # rot(C[(x+1)%5],1)
            if bug == "theta_no_rot":
                rc1 = cxz[(x + 1) % 5][3][b]                          # drop the rotate
            Dxz[x][b] = cm1 ^ rc1

    # === theta final XOR === keccak_rnd.rs:694-717
    theta = [[[0] * 8 for _ in range(5)] for _ in range(5)]
    for x in range(5):
        for y in range(5):
            for b in range(8):
                theta[x][y][b] = S[x][y][b] ^ Dxz[x][b]

    # === rho: HWSL === keccak_rnd.rs:723-766
    rho_tbl = [[RHO[x][y] for y in range(5)] for x in range(5)]
    if bug == "rho_swap":
        rho_tbl[1][0], rho_tbl[2][0] = rho_tbl[2][0], rho_tbl[1][0]
    rot_left = [[[0] * 8 for _ in range(5)] for _ in range(5)]
    rot_right = [[[0] * 8 for _ in range(5)] for _ in range(5)]
    for x in range(5):
        for y in range(5):
            rnc = rho_tbl[x][y] % 16
            for hw in range(4):
                Thw = theta[x][y][2 * hw] | (theta[x][y][2 * hw + 1] << 8)
                left16 = (Thw << rnc) & 0xFFFF
                right16 = (Thw >> (16 - rnc)) & 0xFFFF if rnc > 0 else 0
                rot_left[x][y][2 * hw] = left16 & 0xFF
                rot_left[x][y][2 * hw + 1] = (left16 >> 8) & 0xFF
                rot_right[x][y][2 * hw] = right16 & 0xFF
                rot_right[x][y][2 * hw + 1] = (right16 >> 8) & 0xFF

    def pi(X, Y, z):
        # keccak_rnd.rs:793-795 virtual pi = rot_left[l] + rot_right[r]
        sx, sy, l, rr = pi_src_bytes(X, Y, z)
        val = rot_left[sx][sy][l] + rot_right[sx][sy][rr]
        assert val <= 255, "pi operand exceeds a byte"
        return val

    # === chi: AND then XOR === keccak_rnd.rs:796-870
    chi = [[[0] * 8 for _ in range(5)] for _ in range(5)]
    for x in range(5):
        for y in range(5):
            for b in range(8):
                p0 = pi(x, y, b)
                p1 = pi((x + 1) % 5, y, b)
                p2 = pi((x + 2) % 5, y, b)
                if bug == "chi_no_not":
                    ands = p1 & p2                       # drop the NOT
                elif bug == "chi_swap":
                    ands = (0xFF - p2) & p1              # swap the two operands
                else:
                    ands = (0xFF - p1) & p2              # (255 - pi[x+1]) AND pi[x+2]
                chi[x][y][b] = p0 ^ ands

    # === iota === keccak_rnd.rs:872-894
    rc_bytes = lane_to_bytes(RC[r])
    iota = [0] * 8
    for b in range(8):
        if bug == "iota_no_rc":
            iota[b] = chi[0][0][b]                        # drop rc XOR
        else:
            iota[b] = chi[0][0][b] ^ rc_bytes[b]

    # === output handoff === keccak_rnd.rs:496-509
    out = [0] * 25
    for x in range(5):
        for y in range(5):
            if x == 0 and y == 0:
                out[0] = bytes_to_lane(iota)
            else:
                out[x + 5 * y] = bytes_to_lane(chi[x][y])
    return out
