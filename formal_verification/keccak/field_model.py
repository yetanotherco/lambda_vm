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


# --- the contracts that bound a column, and what survives dropping one -------
#
# Each interval below is the contract of a named construct, so that a change in
# the chip changes the number here rather than leaving a stale comment:
#
#   ARE_BYTES pair   the two columns it carries are bytes            -> BYTE
#   IS_BIT           the theta carry column is a bit                 -> BIT
#   ByteAlu OPERAND  the BITWISE table holds byte rows only, so a virtual
#                    operand `a + b` must land in [0, 255]. That still bounds a
#                    column whose OWN range check is gone -- as long as the
#                    other summand is bounded -- and it is the whole reason the
#                    two "implied" verdicts hold.
BYTE = (0, 255)
BIT = (0, 1)


def operand_summand_window(other):
    """What a ByteAlu operand alone leaves for one summand.

    `this + other` in [0, 255] with `other` in `other`, so `this` is confined to
    [-max(other), 255 - min(other)] -- NOT to [0, 255], but small, which is the
    only property the analysis needs. Requires the read-once premise (each
    column read by exactly ONE operand byte: combinatorics sections 3 and 6) and
    breaks down when BOTH summands are unchecked, since then the operand bounds
    only their sum: that is configuration D, and it has no per-column window at
    all, which is why its output is entirely free.
    """
    return -other[1], 255 - other[0]


def packed_pair_bounds(lo, hi):
    """The interval `lo_col + 256*hi_col` occupies, given per-byte intervals."""
    return lo[0] + 256 * hi[0], lo[1] + 256 * hi[1]


def difference_form_is_exact(rnc, left_bounds, right_bounds):
    """Is the field identity the same statement as the integer identity?

    Everything below parameterises deviations by an INTEGER `d`, which is only
    legitimate while every term stays far below `p`. This is the step the
    difference form silently assumed: widen a bound enough -- a column with no
    bound at all -- and `d` ranges over the whole field, `2**16` is invertible,
    and no sweep over small `d` means anything.
    """
    worst = (
        MASK16 * 2**rnc
        + max(abs(right_bounds[0]), abs(right_bounds[1])) * 2**16
        + max(abs(left_bounds[0]), abs(left_bounds[1]))
    )
    return worst < P // 2


def surviving_deviation(rnc, left_bounds, right_bounds):
    """Complete sweep: does any input halfword admit a second `(left, right)`?

    Returns `None` when all 2**16 inputs are pinned -- the configuration is
    sound -- or `(in_hw, d)` for the first input that admits another solution.

    Complete, not sampled: the identity's solution set is exactly
    `(L - 2**16*d, R + d)` over `d`, `difference_form_is_exact` keeps `d` an
    integer, and `left_bounds` caps `|d|`, so the `d` range below is exhaustive.
    Also asserts the HONEST pair lies inside the bounds, which catches a window
    modelled wrongly (the failure that would make a `None` here meaningless).
    """
    lo_l, hi_l = left_bounds
    lo_r, hi_r = right_bounds
    dmax = (hi_l - lo_l) // 2**16 + 1
    for in_hw in range(1 << 16):
        left, right = honest_shift(in_hw, rnc)
        assert lo_l <= left <= hi_l and lo_r <= right <= hi_r, (
            f"the honest pair for in={in_hw:#06x} falls outside the modelled "
            f"bounds left={left_bounds} right={right_bounds}"
        )
        for d in range(-dmax, dmax + 1):
            if d == 0:
                continue
            dev_left, dev_right = deviate(left, right, d)
            if lo_l <= dev_left <= hi_l and lo_r <= dev_right <= hi_r:
                return in_hw, d
    return None
