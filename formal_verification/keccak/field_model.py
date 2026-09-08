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
deviation directly by `d` per halfword, which is exact integer arithmetic for
as long as every magnitude stays far below `p`: 2**18 for theta with its carry
checked, 2**24 without it, and 2**32.6 for every rho configuration, against a
`p/2` of 2**63. `difference_form_is_exact` checks that per configuration rather
than trusting this sentence. The field enters in exactly one place: a committed column may hold a NEGATIVE integer (as `p - k`), because
nothing bounds it once its range check is gone. `as_field` marks those.

WHAT BOUNDS THE DEVIATION. Two things, and which one bites is the whole result:
  * the range check on `left` (ARE_BYTES) confines `L' = L - 2**16*d` to
    [0, 2**16), and any `d != 0` moves `L` by at least 2**16 -> `d = 0`;
  * the downstream ByteAlu OPERAND that consumes the shift output. The BITWISE
    table holds only byte rows, so the operand must be a byte, which leaves a
    residual window on `left` even with no range check of its own. Whether
    `d = +/-1` fits inside that window is what decides necessity.

THE SECOND AXIS: THE BYTE SPLIT. `d` is not the only freedom, because the chip
commits BYTES and the identity reads a pair only as `lo + 256*hi`. So
`(lo + 256*k, hi - k)` satisfies the identity exactly, leaves the packed value
honest, and no sweep over `d` can see it -- while chi and Dxz read the two bytes
SEPARATELY. A pair whose packed value is pinned therefore still needs its split
pinned, and that is a second question with its own two answers:
`checked_split_is_unique` for a pair that kept its range checks, and
`surviving_byte_split` for one that lost them, whose split is pinned by the
operand byte that reads each half instead.
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
# Not a contract but a wiring fact (cols::cxz_right_bit_for_byte): the odd Dxz
# operand bytes take no carry, so such a byte IS the operand, and the window the
# operand leaves it is the whole byte range.
NO_CARRY = (0, 0)


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


def checked_split_is_unique(bounds=BYTE):
    """Given a PINNED packed value, does a range check pin the two bytes?

    Uniqueness of a split is a property of the CHECK's width, not of the
    identity, which sees only `lo + 256*hi`: complete over every `lo` the check
    admits, `lo + 256*k` has to leave the interval for every `k != 0`; the two
    tested cover all of them, since the deviation grows with `|k|`, so escaping
    at one step escapes at every further one. It holds for a byte check, whose
    256 values are exactly the packing radix -- and it
    stops holding one bit wider, which is the sensitivity control the necessity
    boards run alongside. This is what makes the CHECKED side of a configuration
    honest byte by byte, the premise `surviving_byte_split` then leans on.
    """
    lo_b, hi_b = bounds
    return all(not lo_b <= lo + 256 * k <= hi_b
               for lo in range(lo_b, hi_b + 1) for k in (-1, 1))


def split_form_is_exact(companion):
    """Is `lo + 256*hi = packed` the same statement over the field and over Z?

    `surviving_byte_split` parameterises the split by an INTEGER `k`, and over
    the field the packed value alone allows anything: `256` is invertible, so
    every `lo` has its `hi` and the redistributions are the whole field, not a
    sequence of steps of 256. What collapses them to integer ones is the operand
    window on BOTH bytes of the pair -- each is read by one operand byte, so each
    is small -- which keeps `lo + 256*hi` far below `p`, making the field
    equation the integer equation and `k = hi_honest - hi` an integer.

    This is the split axis's analogue of `difference_form_is_exact`, and it fails
    the same way: a byte with no window at all puts `k` back over the field.
    """
    window = operand_summand_window(companion)
    span = max(abs(window[0]), abs(window[1]))
    return span + 256 * span < P // 2


def surviving_byte_split(companion, companion_moves=(0,)):
    """Is the split of an UNCHECKED pair pinned by the operand bytes reading it?

    The pair's packed value is pinned (`surviving_deviation`), each of its bytes
    is read by exactly ONE ByteAlu operand byte (read-once, combinatorics
    sections 3 and 6) alongside a summand in `companion`, and those windows are
    what make `k` an integer in the first place (`split_form_is_exact`).
    Redistributing the pair
    by `k` moves that operand sum by `256*k`, so the sweep below asks, over every
    (byte, companion) pair an honest row can present -- their sum is a byte,
    since the honest row passes the operand lookup -- whether the moved sum is a
    byte too.

    `companion_moves` is what that summand may do ITSELF, and it is the whole
    reason the answer is configuration-dependent: `(0,)` when the checked side
    pinned it to its honest value (`checked_split_is_unique`), and +/-256 when
    nothing checks it either -- the "both dropped" configuration, where the
    redistribution is absorbed and the split is as free as the packed value.

    Returns None when the split is pinned, else `(byte, companion, k, move)`.
    With `companion_moves = (0,)` the two non-zero `k` are exhaustive: a survivor
    needs both sums inside [0, 255], so it needs `|256*k| <= 255`.
    """
    for byte in range(256):
        for other in range(companion[0], companion[1] + 1):
            if not is_byte(byte + other):
                continue
            for k in (-1, 1):
                for move in companion_moves:
                    if is_byte(byte + 256 * k + other + move):
                        return byte, other, k, move
    return None
