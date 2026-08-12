"""Shared model for the ECSM **affine-selector** z3 gate.

Ground truth: `prover/src/tables/ecsm.rs` on branch `verify/ecsm-affine-selector`
(head of PR #879), plus `executor/src/vm/instruction/execution.rs` for the ABI predicates.
Every builder below carries the `file:line` it was transcribed from; the audit that checks
those citations still say what the builder assumes is `TRANSCRIPTION-AUDIT.md`.

Scope. This board covers ONLY the surface PR #879 adds or changes:

  * the `IS_AFFINE` selector column and its two constraints (idx 421, 422),
  * its `Ecall`-bus pinning (`syscall = xonly + IS_AFFINE·(affine − xonly)`),
  * the `IS_AFFINE`-gated `yG` read and `yR` write, and their `+32 + 8i` offsets,
  * the new `OverflowKind::YrLtP` carry chain (idx 413..420),
  * the `Alu`-LT address-limb senders and their mode-dependent bound.

The pre-existing chip — the curve relations, the ECDAS chain, the double-and-add
induction — is already proved by the earlier board, which this one deliberately does not
re-derive:

    thoughts/ec-recover-opt/gate/RESULTS.md   on branch feat/ec-lincomb2, commit 1d2b4dd7
    (unmerged: that path exists on that branch only, never on main)

Its lemmas L1–L7 and contracts C1–C7 are IMPORTED here as hypotheses and listed as such in
`RESULTS.md`. Two of them the affine change stresses in a new way, so they are re-examined
rather than imported blind:

  * L7 concluded `xR = x(k·P)` "for both yG sign classes", explicitly BECAUSE `yG`'s parity
    was unobservable. Publishing `yR` breaks that premise — see a3_parity_binding.py.
  * C4 (MEMW byte authority) listed `YR` as inheriting byte-ness "from tuple equality with
    ECDAS's byte-checked yR (or YG for k=1)". `YrLtP` now *consumes* that byte contract, so
    it is checked rather than assumed — see a2_yr_lt_p.py's C4-YR probe.

The same builders serve three purposes, as in the earlier board: exact interval bounds,
z3 constraint models, and concrete evaluation of real witnesses — so a transcription error
shows up in the concrete evaluation before any UNSAT is trusted.
"""

import sys
from pathlib import Path


from ecsm_affine_ref import N, P  # noqa: E402  (the oracle is the reference, not the repo)

# ── field / template constants ───────────────────────────────────────────────

PG = 2**64 - 2**32 + 1  # Goldilocks; all AIR constraints are enforced mod PG
SHIFT_32 = 2**32
# prover/src/constraints/templates.rs:26
INV_SHIFT_32 = 18446744065119617026
assert INV_SHIFT_32 * SHIFT_32 % PG == 1, "INV_SHIFT_32 is not 2^-32 mod p_g"

P_BYTES = list(P.to_bytes(32, "little"))
N_BYTES = list(N.to_bytes(32, "little"))

# ── syscall numbers (executor/src/vm/instruction/execution.rs:38, 47) ────────

ECSM_SYSCALL_NUMBER = 2**64 - 1 - 10          # u64::MAX - 10, x-only
ECSM_AFFINE_SYSCALL_NUMBER = 2**64 - 1 - 11   # u64::MAX - 11, affine

# EVERY syscall number the CPU can put on the `Ecall` bus (execution.rs:29, 40, 48, 69).
# A1f needs the whole set, not just the ECSM pair: the receiver's syscall word is LINEAR in
# IS_AFFINE with low-word coefficient −1 and high-word coefficient 0, so as `IS_AFFINE` ranges
# over the field the received low word ranges over the WHOLE field while the high word stays
# fixed — which means every other syscall's tuple is reachable at some value of `IS_AFFINE`.
# `IS_BIT(IS_AFFINE)` is what confines it to {0, 1}. Audit premise P19 keeps this set in sync.
SYSCALL_NUMBERS = {
    "KECCAK": 2**64 - 1 - 1,
    "ECSM": ECSM_SYSCALL_NUMBER,
    "ECSM_AFFINE": ECSM_AFFINE_SYSCALL_NUMBER,
    "HINT": 2**64 - 1 - 30,
}

# ── address-limb bounds (prover/src/tables/ecsm.rs:43, 47) ──────────────────

ADDR_LIMB_BOUND_32B = (1 << 32) - 31
ADDR_LIMB_BOUND_64B = (1 << 32) - 63

# ── the affine memory-op layout (ecsm.rs, the two `for i in 0..4` blocks) ───
#
# yG read : 4 doublewords at ADDR_XG_0 + 32 + 8i, high limb ADDR_XG_1, ts     (mult IS_AFFINE)
# yR write: 4 doublewords at ADDR_XR_0 + 32 + 8i, high limb ADDR_XR_1, ts + 3 (mult IS_AFFINE)
#
# The pre-existing x-only ops, for the offset-collision audit:
# xG read : 4 dwords at ADDR_XG_0 + 8i, ts      ; k read : ADDR_K_0 + 8i, ts + 1
# xR write: 4 dwords at ADDR_XR_0 + 8i, ts + 2
AFFINE_YG_READ_OFFSETS = [32 + 8 * i for i in range(4)]
AFFINE_YR_WRITE_OFFSETS = [32 + 8 * i for i in range(4)]
XONLY_XG_READ_OFFSETS = [8 * i for i in range(4)]
XONLY_XR_WRITE_OFFSETS = [8 * i for i in range(4)]
DWORD = 8

TS_XG_READ, TS_K_READ, TS_XR_WRITE, TS_YR_WRITE = 0, 1, 2, 3
INSTRUCTION_TS_STRIDE = 4  # cpu.rs: one instruction consumes 4 sub-timestamps


# ── the Ecall syscall-word model (ecsm.rs, `syscall_word` closure) ──────────

def syscall_word_lo(is_affine):
    """`xonly_lo + IS_AFFINE·(affine_lo − xonly_lo)`, the low 32-bit word the ECSM row
    puts on the `Ecall` bus (ecsm.rs, ECALL receiver). Linear in IS_AFFINE by design so
    one receiver serves both modes."""
    lo_x = ECSM_SYSCALL_NUMBER & 0xFFFF_FFFF
    lo_a = ECSM_AFFINE_SYSCALL_NUMBER & 0xFFFF_FFFF
    return lo_x + is_affine * (lo_a - lo_x)


def syscall_word_hi(is_affine):
    """Same for the high word. Today both numbers share `0xFFFF_FFFF`, so the IS_AFFINE
    coefficient is ZERO and this word carries no mode information — which is exactly what
    the `const _: () = assert!` in execution.rs:53-58 exists to keep true-by-accident from
    becoming true-by-nobody-noticing."""
    hi_x = ECSM_SYSCALL_NUMBER >> 32
    hi_a = ECSM_AFFINE_SYSCALL_NUMBER >> 32
    return hi_x + is_affine * (hi_a - hi_x)


def addr_bound_by_mode(is_affine):
    """`ADDR_LIMB_BOUND_32B + IS_AFFINE·(BOUND_64B − BOUND_32B)`, the RHS of the `Alu` LT
    senders for `ADDR_XG_0` and `ADDR_XR_0` (ecsm.rs, `addr_bound_by_mode`). `ADDR_K_0`
    uses the flat 32-byte bound in both modes."""
    return ADDR_LIMB_BOUND_32B + is_affine * (ADDR_LIMB_BOUND_64B - ADDR_LIMB_BOUND_32B)


# ── the overflow carry chain (ecsm.rs, `EcsmConstraints::carry_chain`) ──────

OVERFLOW_KINDS = {
    #  kind        const addend   sum stored as
    "XgLtP": (P, "bytes"),
    "KLtN": (N, "bits"),
    "XrLtP": (P, "bytes"),
    "YrLtP": (P, "bytes"),  # NEW in PR #879 (idx 413..420)
}


def const_word(const_value, i):
    """`OverflowKind::const_word(i)`: 32-bit word `i` of the constant addend, assembled
    from its little-endian bytes (ecsm.rs, `const_word`)."""
    return (const_value >> (32 * i)) & 0xFFFF_FFFF


def addend1_word(halfwords, i):
    """`hl[2i] + 2^16·hl[2i+1]` — word `i` of the witnessed halfword addend
    (`XG_SUB_P` / `K_SUB_N` / `XR_SUB_P` / `YR_SUB_P`)."""
    return halfwords[2 * i] + halfwords[2 * i + 1] * (1 << 16)


def sum_word_bytes(byte_cols, i):
    """Word `i` of a byte-stored sum (`XG` / `XR` / `YR`): bytes 4i..4i+3."""
    s = 0
    for byte in range(4):
        s = s + byte_cols[4 * i + byte] * (1 << (8 * byte))
    return s


def sum_word_bits(bit_cols, i):
    """Word `i` of the bit-stored sum (`K`): bits 32i..32i+31."""
    s = 0
    for bit in range(32):
        s = s + bit_cols[32 * i + bit] * (1 << bit)
    return s


def carry_chain(const_value, halfwords, sum_cols, sum_is_bits=False, inv=INV_SHIFT_32):
    """The eight VIRTUAL carries, as expressions in the leaves:

        c_i = (const_word_i + addend1_i + c_{i−1} − sum_i) · 2^{−32}

    Returned in order; `c[7]` is the carry-out the `OverflowRequired` constraint pins to 1.
    `inv` is a parameter only so a negative control can perturb it."""
    c = []
    prev = 0
    for i in range(8):
        a1 = addend1_word(halfwords, i)
        s = sum_word_bits(sum_cols, i) if sum_is_bits else sum_word_bytes(sum_cols, i)
        ci = (const_word(const_value, i) + a1 + prev - s) * inv
        c.append(ci)
        prev = ci
    return c


def overflow_constraints(mu, c):
    """The nine emitted constraints per `OverflowKind`, as expressions that must vanish:
    `µ·c_i·(1−c_i)` for i ∈ 0..6, then `µ·(1−c_7)` (ecsm.rs, the `for kind in [...]` loop).

    Note what is NOT here: there is no `µ·c_7·(1−c_7)`. `c_7` is pinned to the constant 1
    outright, which is stronger, so the missing bit constraint is not a gap."""
    out = [mu * c[i] * (1 - c[i]) for i in range(7)]
    out.append(mu * (1 - c[7]))
    return out


# ── the two new selector constraints (ecsm.rs idx 421, 422) ────────────────

def is_bit_is_affine(is_affine):
    """idx 421 — `IS_AFFINE·(1 − IS_AFFINE)`."""
    return is_affine * (1 - is_affine)


def affine_zero_on_padding(is_affine, mu):
    """idx 422 — `IS_AFFINE·(1 − µ)`."""
    return is_affine * (1 - mu)


# ── honest witness generation (mirrors crypto/ecsm/src/witness.rs) ──────────

def y_sub_p_halfwords(value):
    """`(value − p) mod 2^256` as 16 little-endian halfwords — the honest `YR_SUB_P`
    witness (`witness.rs`: `to_le_32(&((&two_256 + &result.y) - p()))`)."""
    v = (2**256 + value - P) % 2**256
    return [(v >> (16 * j)) & 0xFFFF for j in range(16)]


def le_bytes(value, n=32):
    return [(value >> (8 * j)) & 0xFF for j in range(n)]


def le_bits(value, n=256):
    return [(value >> j) & 1 for j in range(n)]


# ── the two pre-existing ECSM curve relations ───────────────────────────────
#
# IMPORTED, not newly derived: these are the earlier board's `s_ecsm_x2` / `s_ecsm_yg` /
# `conv_carry` (thoughts/ec-recover-opt/gate/gate_common.py, branch feat/ec-lincomb2), and
# they were re-read against `prover/src/tables/ecsm.rs`'s `s_i` / `conv_carry` on THIS branch
# — where the bodies are byte-identical to main, since PR #879 does not touch them.
#
# They are here only because A3 needs to evaluate the FULL in-table constraint set on a
# forged witness. Establishing that these relations pin `yG² ≡ xG³ + b` is the earlier
# board's L3a/L4; this board consumes that.

CARRY_OFFSET_X2 = 8160    # ecsm.rs:37
CARRY_OFFSET_YG = 16319   # ecsm.rs:38
CURVE_B = 7


def _at(arr, length, j):
    """`byte_at`: zero-padding past the operand's length (ecsm.rs, `byte_at`)."""
    return arr[j] if 0 <= j < length else 0


def s_ecsm_x2(xg, q0, x2, i):
    """`Relation::X2` at limb i: `Σ xG_j·xG_{i−j} − x2_i − Σ q0_j·P_{i−j}`."""
    s = 0
    for j in range(i + 1):
        s += _at(xg, 32, j) * _at(xg, 32, i - j)
        s -= _at(q0, 32, j) * _at(P_BYTES, 32, i - j)
    return s - _at(x2, 32, i)


def s_ecsm_yg(yg, x2, xg, q1, i, mu=1):
    """`Relation::Yg` at limb i:
    `Σ(yG_j·yG_{i−j} − x2_j·xG_{i−j} − q1_j·P_{i−j}) + µ·Σ P_j·P_{i−j} − µ·b·[i=0]`.
    `q1` is 33 bytes; its top byte is IS_BIT-constrained."""
    s = 0
    p2 = 0
    for j in range(i + 1):
        s += _at(yg, 32, j) * _at(yg, 32, i - j)
        p2 += _at(P_BYTES, 32, j) * _at(P_BYTES, 32, i - j)
        s -= _at(x2, 32, j) * _at(xg, 32, i - j)
        s -= _at(q1, 33, j) * _at(P_BYTES, 32, i - j)
    s += mu * p2
    if i == 0:
        s -= mu * CURVE_B
    return s


def honest_conv_carries(s_values):
    """The honest carry array for a convolution relation: `c_i = (c_{i−1} + S_i)/256`,
    the exact-division solution of `256·c_i − c_{i−1} − S_i = 0`. Returns
    `(carries, exact)`; `exact` also requires the chain to close at `c_63 = 0`, which is the
    `ColIsZero` constraint the earlier board's N3 found load-bearing."""
    c = []
    prev = 0
    exact = True
    for s in s_values:
        total = prev + s
        if total % 256 != 0:
            exact = False
        prev = total // 256
        c.append(prev)
    return c, exact and prev == 0


# ── field-root machinery ────────────────────────────────────────────────────
#
# The `IS_BIT`-shaped constraints (`x·(1−x) ≡ 0 mod p_g`) are statements about ROOTS OF A
# POLYNOMIAL OVER A FIELD, and that is how they are discharged here: sympy factors the
# polynomial over GF(p_g) and the factorisation is checked to be complete (total degree of
# the linear factors == degree of the polynomial), so no root is missed.
#
# Handing the lifted integer form `x(1−x) = m·p_g` to z3 instead does NOT terminate on
# constants this size — the query is nonlinear integer arithmetic with a free quotient.
# Recorded in RESULTS.md's method note so nobody re-attempts it.

def certify_pg_prime():
    """p_g = 2^64 − 2^32 + 1 is prime. The one assumed-then-certified algebraic fact this
    board needs (the analogue of the earlier board's A-PRIME)."""
    import sympy

    return bool(sympy.isprime(PG))


def field_roots(coeffs, modulus=None):
    """Roots over GF(`modulus`) of the polynomial with the given coefficients, highest degree
    first, together with a completeness flag. `modulus` defaults to the Goldilocks prime
    `p_g`, the field the AIR is enforced over; the curve-side lemmas pass `p` instead.

    Returns `(roots, complete)`. `complete` is True when the linear factors account for the
    full degree, i.e. the polynomial splits and the root list is exhaustive. A degree-`d`
    polynomial over a field has at most `d` roots, so a complete split is a proof that the
    returned set is ALL of them.

    sympy's `modulus=` uses the SYMMETRIC residue range, so factor coefficients come back
    possibly negative; roots are normalised into `[0, modulus)` here."""
    import sympy
    from sympy.abc import x

    q = PG if modulus is None else modulus
    poly = sympy.Poly(sum(c * x**i for i, c in enumerate(reversed(coeffs))), x, modulus=q)
    _, factors = poly.factor_list()
    roots = {}
    linear_degree = 0
    for f, mult in factors:
        if f.degree() == 1:
            a, b = (int(v) for v in f.all_coeffs())   # a·x + b
            roots[(-b * pow(a, -1, q)) % q] = mult
            linear_degree += mult
    return roots, linear_degree == poly.degree()


def eval_overflow_chain_concrete(const_value, value, sum_is_bits=False,
                                 addend_value=None):
    """Evaluate the carry chain on a CONCRETE honest witness, over F_pg, and return
    `(carries, ok)` where ok means: every c_i ∈ {0,1} and c_7 == 1.

    `addend_value` overrides the honest `(value − const) mod 2^256` addend, which is how
    the negative controls inject a forged representation."""
    if addend_value is None:
        addend_value = (2**256 + value - const_value) % 2**256
    hl = [(addend_value >> (16 * j)) & 0xFFFF for j in range(16)]
    cols = le_bits(value) if sum_is_bits else le_bytes(value)
    c = carry_chain(const_value, hl, cols, sum_is_bits)
    c = [ci % PG for ci in c]
    ok = all(ci in (0, 1) for ci in c) and c[7] == 1
    return c, ok
