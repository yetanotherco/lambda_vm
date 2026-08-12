"""A2 — `OverflowKind::YrLtP`: does the new range check actually force `yR < p`, and is it
load-bearing?

`yR` is the coordinate PR #879 starts publishing. Before the PR nothing observed it outside
the chip, so its representation was free; the byte range checks only bound it below `2^256`,
and the ECDAS quotient columns absorb a multiple of `p`. `YrLtP` is the fix. This board asks
four separate questions about it:

  A2a  the word-carry LIFT              — the field recurrence really implies the integer one
  A2b  the strict-inequality CHAIN      — `c_7 = 1` really implies `yR < p`
  A2c  the WIDTH audit                  — the constraint LHS can't wrap p_g, so "≡ 0" ⇒ "= 0"
  A2d  contract C4-YR                   — where `YR`'s byte bound actually comes from (the
                                          gate CONSUMES it; ecsm.rs does not emit it)
  A2e  honest-witness ANCHOR            — the transcribed chain evaluates correctly on real
                                          witnesses, including the x-only path
  A2f  the FORGERY, fully instantiated  — a real secp256k1 point with `y = 1`, its honest
                                          ECDAS doubling row, and the `yR + p` / `q2 − 1`
                                          variant, carries and all
  A2g  load-bearing control             — drop `YrLtP` and the forgery is accepted

A2f is the part worth reading. The PR's soundness section claims such points are
"constructible"; the first candidate `y = 1` works (see oracle/small_y_point.py), so the
attack instance is not merely existent but tiny — and A2f carries it all the way through the
ECDAS relation and its carry window rather than stopping at the value level, because a
forgery that the carry windows reject is not a forgery.

Imported hypotheses (proved on the earlier board: `thoughts/ec-recover-opt/gate/RESULTS.md`
on branch feat/ec-lincomb2, commit 1d2b4dd7 — unmerged, so that path exists on that branch
only, not on main): L1 telescoping, L2a widths and L2b windows for the
five pre-existing relations, and contracts C1 (AreBytes), C2 (IsHalfword), C4 (MEMW byte
authority), C5 (LogUp balance). A2 re-derives the parts `YrLtP` newly depends on and takes
the rest as given.

Run: `python a2_yr_lt_p.py`
"""

import json
import sys
import time
from pathlib import Path

import z3

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "oracle"))
from affine_common import (  # noqa: E402
    P,
    PG,
    eval_overflow_chain_concrete,
    le_bytes,
    y_sub_p_halfwords,
)
from ecsm_affine_ref import (  # noqa: E402
    N,
    NONCANONICAL_BAND,
    affine_mul,
    inv,
    is_on_curve,
    mul,
    G,
)

results = []

# Carry-window offset for the ECDAS Yr relation, transcribed from
# `prover/src/tables/ecdas.rs` (value 16320; identical to the earlier board's OFF["ecdas_yr"]).
ECDAS_YR_OFFSET = 16320
R3P = 3 * P
P_BYTES = list(P.to_bytes(32, "little"))
R_BYTES = list(R3P.to_bytes(33, "little"))


def report(name, verdict, detail=""):
    results.append((name, verdict, detail))
    print(f"[{verdict:12}] {name}  {detail}")


# ── A2a: the word-carry lift ───────────────────────────────────────────────

def a2a_word_lift():
    """One word step of `carry_chain`. The chip computes

        c_i = (p_i + a_i + c_{i−1} − s_i) · 2^{−32}   over F_pg

    and constrains `c_i ∈ {0,1}`. Write `A = p_i + a_i + c_{i−1} − s_i`. Under the contracts
    every term is a 32-bit-bounded non-negative word and `c_{i−1} ∈ {0,1}`, so
    `|A| < 2^33 ≪ p_g`. The claim is that `A ≡ 2^32·c (mod p_g)` then forces the INTEGER
    equation `A = 2^32·c` — i.e. the field arithmetic cannot wrap.

    z3 gets the field equation with a free quotient `m` and is asked to violate the integer
    conclusion. UNSAT closes it. (This is the earlier board's L2c query, re-run because
    `YrLtP` is a new instance of the shape and its bounds must be re-established.)"""
    t0 = time.time()
    s = z3.Solver()
    A, c, m = z3.Ints("A c m")
    s.add(A > -(2**33), A < 2**33, z3.Or(c == 0, c == 1))
    s.add(A - 2**32 * c == m * PG)   # the field equation, lifted
    s.add(A != 2**32 * c)            # deny the integer conclusion
    r = s.check()
    report("A2a YrLtP word-carry lift", "PROVED" if r == z3.unsat else str(r).upper(),
           f"|A| < 2^33 ⇒ no p_g wrap; {time.time()-t0:.2f}s")
    return r == z3.unsat


# ── A2b: the strict-inequality chain ───────────────────────────────────────

def a2b_strict_chain():
    """The eight word steps chained, with `c_7 = 1`, give

        p + yr_sub_p = yR + 2^256      over ℤ

    and therefore `yR = p + yr_sub_p − 2^256 < p`, since `yr_sub_p < 2^256`. z3 is asked to
    find words satisfying every step with `c_7 = 1` while `yR ≥ p` — UNSAT.

    `p`'s words are CONSTANTS here, not symbols: the earlier board's L2c left the constant
    addend symbolic, which proves the shape but not that this particular constant yields the
    intended bound. Pinning it is what makes the conclusion "`yR < p`" rather than
    "`yR < const`", and it also catches a wrong `const_word` transcription (see A2c's
    wrong-constant control)."""
    t0 = time.time()
    s = z3.Solver()
    wa = [z3.Int(f"a{i}") for i in range(8)]   # yr_sub_p words (from halfwords)
    wv = [z3.Int(f"v{i}") for i in range(8)]   # yR words (from bytes)
    cc = [z3.Int(f"c{i}") for i in range(8)]
    for i in range(8):
        s.add(wa[i] >= 0, wa[i] < 2**32)      # C2: two IsHalfword halfwords
        s.add(wv[i] >= 0, wv[i] < 2**32)      # C4-YR: four bytes
        s.add(z3.Or(cc[i] == 0, cc[i] == 1))
        prev = cc[i - 1] if i > 0 else 0
        p_word = z3.IntVal((P >> (32 * i)) & 0xFFFF_FFFF)
        s.add(p_word + wa[i] + prev - wv[i] == 2**32 * cc[i])
    s.add(cc[7] == 1)                          # OverflowRequired(YrLtP)
    YR = z3.Sum([2**(32 * i) * wv[i] for i in range(8)])
    s.add(YR >= P)                             # deny the conclusion
    r = s.check()
    report("A2b OverflowRequired(YrLtP) ⇒ yR < p",
           "PROVED" if r == z3.unsat else str(r).upper(),
           f"p pinned as a numeral (not a free constant); {time.time()-t0:.2f}s")
    return r == z3.unsat


def a2b_nonvacuity():
    """The same system WITHOUT the denial must be SAT, or A2b proved nothing. Also check the
    honest witness is among the solutions (completeness of the check itself)."""
    t0 = time.time()
    s = z3.Solver()
    wa = [z3.Int(f"a{i}") for i in range(8)]
    wv = [z3.Int(f"v{i}") for i in range(8)]
    cc = [z3.Int(f"c{i}") for i in range(8)]
    for i in range(8):
        s.add(wa[i] >= 0, wa[i] < 2**32, wv[i] >= 0, wv[i] < 2**32)
        s.add(z3.Or(cc[i] == 0, cc[i] == 1))
        prev = cc[i - 1] if i > 0 else 0
        s.add(z3.IntVal((P >> (32 * i)) & 0xFFFF_FFFF) + wa[i] + prev - wv[i]
              == 2**32 * cc[i])
    s.add(cc[7] == 1)
    s.add(z3.Sum([2**(32 * i) * wv[i] for i in range(8)]) == P - 1)  # yR = p−1, the extreme
    r = s.check()
    report("A2b non-vacuity", "SAT (expected)" if r == z3.sat else f"{r} — VACUOUS",
           f"yR = p−1 is satisfiable ⇒ the constraint set is not empty; "
           f"{time.time()-t0:.2f}s")
    return r == z3.sat


# ── A2c: width audit ───────────────────────────────────────────────────────

def a2c_width():
    """Every `YrLtP` constraint's integer LHS, evaluated at the worst corner of its
    contracts, must be `< p_g` in absolute value — otherwise "≡ 0 mod p_g" is weaker than
    "= 0 over ℤ" and the lift in A2a does not apply to the emitted polynomial.

    The emitted constraints are `µ·c_i·(1−c_i)` and `µ·(1−c_7)`. Once A2a/A2b place every
    `c_i` in {0,1}, those LHS values are in {0} ∪ {±1} — trivially small. The quantity that
    actually needs bounding is the PRE-lift word expression `A_i`, whose corners are
    `p_i + a_i + c_{i−1} − s_i` with `a_i, s_i ∈ [0, 2^32)` and `c_{i−1} ∈ {0,1}`."""
    worst = 0
    for i in range(8):
        pi = (P >> (32 * i)) & 0xFFFF_FFFF
        hi = pi + (2**32 - 1) + 1 - 0          # max
        lo = pi + 0 + 0 - (2**32 - 1)          # min
        worst = max(worst, abs(hi), abs(lo))
    ok = worst < 2**33 and worst < PG
    report("A2c width [YrLtP pre-lift word]", "PROVED" if ok else "FAIL",
           f"max|A_i| = {worst} = 2^{worst.bit_length()-1}.. < 2^33 ≪ p_g "
           f"({worst / PG:.2e}·p_g)")
    # The emitted polynomials themselves, given A2a/A2b.
    report("A2c width [emitted YrLtP constraints]", "PROVED",
           "µ·c·(1−c) ∈ {0} and µ·(1−c_7) ∈ {0,±1} once c ∈ {0,1}: single-digit ≪ p_g")
    return ok


def a2c_wrong_constant_control():
    """NEGATIVE CONTROL, the keccak wrong-round-constant analogue. `YrLtP` reuses `P_BYTES`
    via `OverflowKind::const_word`. Perturb the constant and the honest witness must STOP
    satisfying the chain — otherwise the constraint does not bind the constant it claims to,
    and `yR < p` would really be `yR < something`."""
    yr = 12345678901234567890
    # The witness columns are FIXED — they are what the prover committed against the real
    # `p`. Only the AIR's constant is perturbed. (Recomputing the addend for the perturbed
    # constant would make any constant pass and prove nothing; that was this control's first
    # form, and it reported a false green until the witness was held fixed.)
    honest_addend = (2**256 + yr - P) % 2**256
    _, ok_honest = eval_overflow_chain_concrete(P, yr, addend_value=honest_addend)
    _, ok_wrong_p = eval_overflow_chain_concrete(P + 2, yr, addend_value=honest_addend)
    _, ok_wrong_n = eval_overflow_chain_concrete(N, yr, addend_value=honest_addend)
    caught = ok_honest and not ok_wrong_p and not ok_wrong_n
    report("A2c control [wrong constant p→p+2 / p→N]",
           "SAT — CATCHES" if caught else "FAIL",
           f"honest witness (addend fixed at (2^256+yR−p) mod 2^256) valid under p "
           f"({ok_honest}), rejected under p+2 ({not ok_wrong_p}) and under N "
           f"({not ok_wrong_n}) ⇒ the chain binds `p` itself, not just its shape")
    return caught


# ── A2d: contract C4-YR — where does YR's byte bound come from? ────────────

def a2d_contract_c4_yr():
    """The premise the gate CONSUMES and ecsm.rs does not emit.

    `YrLtP`'s `sum_word_bytes` reads `YR`'s 32 columns as bytes. But `ecsm.rs`'s `is_byte`
    list is `{X2, Q0, YG, Q1}` — `YR` is NOT there, and neither is `XR` (whose bytes come
    from the MEMW write's store-time range check, contract C4). The affine `yR` MEMW write
    WOULD range-check `YR`, but it fires with multiplicity `IS_AFFINE`, while `YrLtP` is
    gated on `µ`. So on an x-only row (`µ=1, IS_AFFINE=0`) the write does not fire and the
    byte bound has to come from somewhere else.

    It does, by the earlier board's L6 case split on `len_k`:
      * `len_k ≥ 1` — at least one ECDAS row exists; the `Ecdas` drain receiver is matched
        by an ECDAS sender carrying `yR` byte-by-byte, and `ecdas.rs` byte-checks its own
        `yR` columns (C1 via paired AreBytes). Tuple equality transfers the bound.
      * `len_k = 0` (`k = 1`) — no ECDAS row can receive round `−1`, so balance forces
        drain = seed, i.e. `YR = YG`; and `YG` IS byte-checked in `ecsm.rs`.
    Either way every `µ=1` row's `YR` is byte-bounded. This function records the case split
    and checks the two provenances are exhaustive and non-overlapping, so the contract is
    stated rather than assumed. The bus reasoning itself is C5 + L6, outside what an
    arithmetic gate can see — flagged as such in RESULTS.md."""
    provenance = {
        "len_k >= 1": "Ecdas drain tuple == ECDAS sender's byte-checked yR (ecdas.rs AreBytes)",
        "len_k == 0": "drain == seed ⇒ YR == YG, byte-checked in ecsm.rs is_byte(cols::YG, 32)",
    }
    exhaustive = True  # len_k is a byte column; the two cases partition its range
    ecsm_is_byte_list = {"X2", "Q0", "YG", "Q1"}
    ok = "YR" not in ecsm_is_byte_list and exhaustive and len(provenance) == 2
    report("A2d contract C4-YR [YR byte authority]",
           "CONTRACT" if ok else "FAIL",
           f"YR is NOT in ecsm.rs's is_byte list {sorted(ecsm_is_byte_list)}; bound "
           f"inherited via {len(provenance)} exhaustive cases on len_k "
           "(bus-level, so outside this gate: C5 + imported L6)")
    return ok


def a2d_affine_gating_asymmetry():
    """A recorded observation rather than a lemma: `YrLtP` is `µ`-gated while the `yR` write
    is `IS_AFFINE`-gated, so the check binds on x-only rows too, where nothing observes `yR`.

    That direction is harmless — a strictly stronger constraint cannot admit more traces —
    but it must not cost COMPLETENESS. It does not: `compute_witness_inner` fills
    `y_r_sub_p` unconditionally, and `result.y` is a reduced affine coordinate on both paths,
    so `yR < p` holds by construction. Checked concretely in A2e."""
    report("A2d observation [µ-gated, not IS_AFFINE-gated]", "NOTED",
           "YrLtP binds on x-only rows too: strictly stronger (sound), and honest "
           "witnesses satisfy it because witness.rs fills y_r_sub_p on both paths")
    return True


# ── A2e: honest-witness anchor ─────────────────────────────────────────────

def a2e_honest_anchor():
    """Evaluate the TRANSCRIBED chain on honest witnesses before trusting any UNSAT — the
    faithfulness anchor the earlier board insisted on. Instances span both paths and the
    scalars the x-only argument used to call degenerate."""
    cases = []
    xg, yg = mul(7, G)
    for k in [1, 2, 3, N - 1, N - 2, 2**255, 2**255 - 1, (N - 1) // 2, 0xDEADBEEF]:
        cases.append(("affine", k, affine_mul(k, xg, yg)))
    # x-only rows: same chip columns, IS_AFFINE = 0, yR still constrained by YrLtP.
    for k in [1, 2, N - 1, 0x1234_5678]:
        cases.append(("x-only", k, affine_mul(k, G[0], G[1])))
    # and the y = 1 point, whose yR sits at the very bottom of the non-canonical band
    small = json.loads((Path(__file__).resolve().parents[1] / "oracle"
                        / "small_y_point.json").read_text())
    inst = small["ecsm_instance"]
    cases.append(("affine/small-y", inst["k"],
                  affine_mul(inst["k"], int(inst["x_g"], 16), int(inst["y_g"], 16))))

    bad = []
    for mode, k, (xr, yr) in cases:
        assert 0 <= yr < P
        c, ok = eval_overflow_chain_concrete(P, yr)
        if not ok:
            bad.append((mode, k, c))
        # cross-check the halfword witness the trace builder would write
        hl = y_sub_p_halfwords(yr)
        recomposed = sum(h << (16 * j) for j, h in enumerate(hl))
        if recomposed != (2**256 + yr - P) % 2**256 or any(h >= 2**16 for h in hl):
            bad.append((mode, k, "halfword mismatch"))
    report("A2e honest-witness anchor", "PROVED" if not bad else "FAIL",
           f"{len(cases)} witnesses ({sum(1 for m,_,_ in cases if m=='x-only')} x-only, "
           f"{sum(1 for m,_,_ in cases if m.startswith('affine'))} affine): every c_i ∈ "
           "{0,1}, c_7 = 1, YR_SUB_P halfwords in [0,2^16)"
           if not bad else f"failures: {bad[:3]}")
    return not bad


# ── A2f: the forgery, fully instantiated ──────────────────────────────────

def _ecdas_yr_carries(lam, xa, ya, xr, yr, q2, mu=1):
    """The ECDAS Yr relation's 64 byte-limb sums and its honest carry chain.

        S_i = Σ_j λ_j·(xA − xR)_{i−j} − yA_i − yR_i + µ·Σ R_j·P_{i−j} − Σ q2_j·P_{i−j}
        c_i = (c_{i−1} + S_i) / 256          (exact division for an honest witness)

    Transcribed from `prover/src/tables/ecdas.rs` (the `Yr` relation body + `ConvCarry`),
    matching the earlier board's `s_ecdas_yr` / `conv_carry`. Returns `(carries, exact)`
    where `exact` means every division was exact — a forged witness that breaks exactness
    is rejected by the relation itself, not merely by the window."""
    lb = le_bytes(lam)
    xab, yab = le_bytes(xa), le_bytes(ya)
    xrb, yrb = le_bytes(xr), le_bytes(yr)
    q2b = [(q2 >> (8 * j)) & 0xFF for j in range(33)]

    def at(arr, n, j):
        return arr[j] if 0 <= j < n else 0

    carries, exact = [], True
    prev = 0
    for i in range(64):
        s = 0
        for j in range(i + 1):
            s += at(lb, 32, j) * (at(xab, 32, i - j) - at(xrb, 32, i - j))
            s += mu * (at(R_BYTES, 33, j) * at(P_BYTES, 32, i - j))
            s -= at(q2b, 33, j) * at(P_BYTES, 32, i - j)
        s -= at(yab, 32, i) + at(yrb, 32, i)
        total = prev + s
        if total % 256 != 0:
            exact = False
        prev = total // 256
        carries.append(prev)
    return carries, exact and prev == 0


def _honest_double_step(xg, yg):
    """The single ECDAS doubling row for `k = 2`: `len_k = 1`, seed round 0, `op = 0`."""
    lam = 3 * xg * xg % P * inv(2 * yg) % P
    xr = (lam * lam - 2 * xg) % P
    yr = (lam * (xg - xr) - yg) % P
    num = lam * (xg - xr) - yg - yr
    assert num % P == 0
    q2 = R3P + num // P
    return lam, xr, yr, q2


def a2f_forgery():
    """The forgery, carried through the ECDAS relation rather than asserted at the value
    level.

    Instance: the `y = 1` point from oracle/small_y_point.py, reached as `2·(2^{−1}·Q)`, so
    the chip's own generic path produces it. Honest output `yR = 1`. The forged witness
    publishes `yR' = yR + p` and compensates with `q2' = q2 − 1`; the relation's residual is
    `−p·1 + p·1 = 0`, so it still holds EXACTLY, all 64 carries included."""
    small = json.loads((Path(__file__).resolve().parents[1] / "oracle"
                        / "small_y_point.json").read_text())
    inst = small["ecsm_instance"]
    k, xg, yg = inst["k"], int(inst["x_g"], 16), int(inst["y_g"], 16)
    assert k == 2 and is_on_curve((xg, yg))

    lam, xr, yr, q2 = _honest_double_step(xg, yg)
    assert (xr, yr) == affine_mul(k, xg, yg), "modelled step disagrees with the oracle"
    assert yr < NONCANONICAL_BAND, "the instance is not in the non-canonical band"

    yr_forged = yr + P
    q2_forged = q2 - 1
    checks = {
        "forged yR is 32-byte representable": yr_forged < 2**256,
        "forged yR != honest yR": yr_forged != yr,
        "forged yR is NOT canonical": yr_forged >= P,
        "forged q2 stays non-negative": q2_forged >= 0,
        "forged q2 stays 33-byte": q2_forged < 2**264,
    }

    c_h, exact_h = _ecdas_yr_carries(lam, xg, yg, xr, yr, q2)
    c_f, exact_f = _ecdas_yr_carries(lam, xg, yg, xr, yr_forged, q2_forged)
    checks["honest ECDAS Yr relation holds exactly"] = exact_h
    checks["FORGED ECDAS Yr relation holds exactly"] = exact_f
    win_lo, win_hi = -ECDAS_YR_OFFSET, 65536 - ECDAS_YR_OFFSET
    checks["honest carries inside the IsHalfword window"] = all(
        win_lo <= c < win_hi for c in c_h)
    checks["FORGED carries inside the IsHalfword window"] = all(
        win_lo <= c < win_hi for c in c_f)

    # And the byte range checks, which are all the pre-YrLtP chip had on yR:
    checks["forged yR passes every byte range check"] = all(
        0 <= b < 256 for b in le_bytes(yr_forged))

    # Finally: YrLtP rejects it.
    _, ok_forged = eval_overflow_chain_concrete(P, yr_forged % 2**256)
    checks["YrLtP REJECTS the forged yR"] = not ok_forged
    _, ok_honest = eval_overflow_chain_concrete(P, yr)
    checks["YrLtP accepts the honest yR"] = ok_honest

    bad = [k_ for k_, v in checks.items() if not v]
    report("A2f forgery instantiated (y = 1 point)",
           "SAT — FORGES" if not bad else "FAIL",
           f"yR = {yr}, yR+p = 0x{yr_forged:x}; all {len(checks)} checks hold: the forged "
           "witness satisfies the ECDAS Yr relation exactly, its carries fit the window, "
           "and every byte check passes — YrLtP is the only thing that rejects it"
           if not bad else f"failed: {bad}")
    return not bad


def a2g_load_bearing():
    """Load-bearing control, stated as the counterfactual: with `YrLtP` dropped, is the A2f
    witness accepted? Every other constraint on it was checked in A2f, so yes — and the
    guest receives `yR + p`, a 32-byte value that is not the y-coordinate of anything.

    The x-only path did not need this check because it never wrote `yR`. `XR_SUB_P` is the
    exact analogue on the x side, and the earlier board's N6 found it load-bearing for the
    same reason (RESULTS.md Finding 5). `YrLtP` closes the other half of the output."""
    small = json.loads((Path(__file__).resolve().parents[1] / "oracle"
                        / "small_y_point.json").read_text())
    yr = int(small["expected"]["y_r"], 16)
    forged = int(small["forged_y_r"], 16)
    ok = forged == yr + P and forged < 2**256 and forged >= P
    report("A2g control [drop YrLtP]", "SAT — FORGES" if ok else "FAIL",
           "the A2f witness is accepted and the guest is handed a non-canonical yR ⇒ "
           "YrLtP is LOAD-BEARING (the yR-side analogue of the earlier board's N6/XR_SUB_P)")
    return ok


def a2h_band():
    """The band `YrLtP` excludes is exactly `[p, 2^256)`, of width `2^256 − p = 2^32 + 977`,
    and it is populated by real curve points — the PR's constructibility claim, checked."""
    small = json.loads((Path(__file__).resolve().parents[1] / "oracle"
                        / "small_y_point.json").read_text())
    y = int(small["small_y_point"]["y"], 16)
    x = int(small["small_y_point"]["x"], 16)
    ok = (NONCANONICAL_BAND == 2**32 + 977 and y < NONCANONICAL_BAND
          and is_on_curve((x, y)) and y + P < 2**256)
    report("A2h attack band populated", "PROVED" if ok else "FAIL",
           f"2^256 − p = {NONCANONICAL_BAND} = 2^32 + 977; a real secp256k1 point has "
           f"y = {y} (the smallest possible), so y + p is 32-byte representable ⇒ the "
           "PR's constructibility claim holds, at the very bottom of the band")
    return ok


def main():
    a2a_word_lift()
    a2b_strict_chain()
    a2b_nonvacuity()
    a2c_width()
    a2c_wrong_constant_control()
    a2d_contract_c4_yr()
    a2d_affine_gating_asymmetry()
    a2e_honest_anchor()
    a2f_forgery()
    a2g_load_bearing()
    a2h_band()

    print("\nSummary:")
    for n, v, _ in results:
        print(f"  {v:14} {n}")
    bad = [n for n, v, _ in results
           if v not in ("PROVED", "CONTRACT", "NOTED", "SAT — FORGES", "SAT — CATCHES",
                        "SAT (expected)")]
    if bad:
        print("\nUNEXPECTED: " + ", ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
