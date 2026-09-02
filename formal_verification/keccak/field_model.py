"""
Shared integer-mod-`p` model of the inlined θ/ρ shift identities.

WHY A SECOND MODEL. `z3_verify.py` proves the round wiring correct *given* the
byte bounds, and it structurally cannot ask whether those bounds are NEEDED: it
carries them as the WIDTH of its bitvectors, and in bitvector arithmetic `2**16`
is a zero divisor, so a widened model would wrongly keep the decomposition
pinned. Mod the Goldilocks prime `2**16` is invertible, which is exactly what
makes the question askable — and answerable.

THE SHAPE OF THE QUESTION. Each shift is one identity per halfword,

    mu * (in * 2**rnc  -  right * 2**16  -  left) = 0            (rnc = 1 for theta)

over the field. One equation, two unknowns: for ANY `left` there is exactly one
`right = (in*2**rnc - left) * inv(2**16)`, so the identity alone pins nothing.
What pins it are the range checks on `left`/`right`.

THE DIFFERENCE FORM (why no `% p` appears below). If `(L, R)` satisfies the
identity then so does `(L - 2**16 * d, R + d)` for any `d`, and those are the
ONLY other solutions. So instead of solving over the field we parameterise the
deviation directly by `d` per halfword. Every magnitude then stays under 2**18,
far below `p`, so the field equation and the integer equation coincide and the
whole analysis is exact integer arithmetic. The field enters in exactly one
place: a committed column may hold a NEGATIVE integer (as `p - k`), because
nothing bounds it once its range check is gone. `as_field` marks those.

WHAT BOUNDS THE DEVIATION. Two things, and which one bites is the whole result:
  * the range check on `left` (ARE_BYTES) confines `L' = L - 2**16*d` to
    [0, 2**16), and any `d != 0` moves `L` by at least 2**16 -> `d = 0`;
  * the downstream ByteAlu OPERAND that consumes the shift output. The BITWISE
    table holds only byte rows, so the operand must be a byte, which leaves a
    residual window on `left` even with no range check of its own. Whether
    `d = +/-1` fits inside that window is what decides necessity.
"""

P = 2**64 - 2**32 + 1                    # Goldilocks
MASK16 = 0xFFFF


def as_field(v):
    """A committed column holds `v` as a field element; negatives wrap to p-|v|."""
    return v % P


def pack(lo, hi):
    """The halfword a (low byte, high byte) column pair denotes."""
    return lo + 256 * hi


def honest_shift(in_hw, rnc):
    """The unique (left, right) the identity forces when both are byte-bounded.

    Euclidean division of `in_hw * 2**rnc` by `2**16`: right = quotient,
    left = remainder. Valid for theta (rnc = 1) and every rho lane."""
    assert 0 <= in_hw <= MASK16 and 0 <= rnc < 16
    prod = in_hw << rnc
    return prod & MASK16, prod >> 16


def identity_holds(in_hw, rnc, left, right):
    """The shipped constraint, evaluated over the field."""
    return (in_hw * (2**rnc) - right * (2**16) - left) % P == 0


def deviate(left, right, d):
    """The only other solution family: (L, R) -> (L - 2**16*d, R + d)."""
    return left - (2**16) * d, right + d


# --- theta: rnc = 1, `right` is a single IS_BIT-pinned carry column ----------
# The carry of halfword h lands on the LOW byte of halfword h+1 (cols::
# cxz_right_bit_for_byte: even b -> (b/2 + 3) % 4), so within one x the four
# halfwords form a cycle of length 4. Odd bytes take no carry.
THETA_RNC = 1


def theta_carry_source(h):
    """Which halfword's carry is added to the low byte of halfword `h`."""
    return (h - 1) % 4


def theta_operand_bytes(cxz_left, cxz_right):
    """rotated_C[0..8): the ByteAlu operand the Dxz XOR consumes."""
    out = []
    for b in range(8):
        v = cxz_left[b]
        if b % 2 == 0:
            v += cxz_right[theta_carry_source(b // 2)]
        out.append(v)
    return out


# --- rho: `right` is a byte pair, and pi pairs it with `left` ----------------
def rho_pi_offsets(rbc):
    """cols::pi_src_cols: l(z) = z + a mod 8, r(z) = z + a - 2 mod 8."""
    return [0, 6, 4, 2][rbc]


def rho_operand_bytes(rot_left, rot_right, rbc):
    """pi[0..8) for the output lane that reads this source lane."""
    a = rho_pi_offsets(rbc)
    return [rot_left[(z + a) % 8] + rot_right[(z + a - 2) % 8] for z in range(8)]


def is_byte(v):
    return 0 <= v <= 255
