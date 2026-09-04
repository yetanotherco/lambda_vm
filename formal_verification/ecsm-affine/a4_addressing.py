"""A4 — addressing: does the AIR accept exactly the addresses the VM accepts, and can the
affine `+32 … +63` span escape its limb?

The affine variant doubles both operand buffers to 64 bytes, and the `yG`/`yR` bus tuples are
built by adding `32 + 8i` to the LOW limb while reusing the high limb unchanged. That is only
sound if the span provably cannot reach `2^32`. PR #879's answer is a set of `Alu`-LT senders
whose bound is itself linear in `IS_AFFINE`.

  A4a  the LT bound == the executor's `addr_limb_ok`   — same accept set, both spans
  A4b  the `+32 + 8i` span cannot cross `2^32`         — the reused high limb is safe
  A4c  the "seven-value band" the comment claims       — the exact size of the gap the LT
                                                         senders close, both modes
  A4d  `k`'s bound is flat, and correctly so           — `k` is 32 B in both modes
  A4e  the overlap guard == interval disjointness      — plus the reachable `u64` wrap
  A4f  timestamps do not collide                       — 4 sub-timestamps, stride 4
  A4g  a flat 64-byte bound would break COMPLETENESS   — why the bound has to be
                                                         mode-dependent rather than
                                                         conservative

A4c is the one that says why these senders exist at all. Without them the bus is *satisfiable*
for addresses the executor rejects with `EcsmAddressOverflow` — a provable trace for an
execution the VM halts on. A4c measures that band exactly rather than taking the comment's
word for it.

Run: `python a4_addressing.py`
"""

import sys
import time
from pathlib import Path

import z3

sys.path.insert(0, str(Path(__file__).parent))
from affine_common import (  # noqa: E402
    ADDR_LIMB_BOUND_32B,
    ADDR_LIMB_BOUND_64B,
    AFFINE_YG_READ_OFFSETS,
    AFFINE_YR_WRITE_OFFSETS,
    DWORD,
    INSTRUCTION_TS_STRIDE,
    TS_K_READ,
    TS_XG_READ,
    TS_XR_WRITE,
    TS_YR_WRITE,
    XONLY_XG_READ_OFFSETS,
    XONLY_XR_WRITE_OFFSETS,
    addr_bound_by_mode,
)
from ecsm_affine_ref import (  # noqa: E402
    addr_limb_ok,
    operands_disjoint,
    operands_disjoint_u64_buggy,
)

results = []
LIMB = 2**32


def report(name, verdict, detail=""):
    results.append((name, verdict, detail))
    print(f"[{verdict:12}] {name}  {detail}")


# ── A4a: the LT bound is the executor's predicate ─────────────────────────

def a4a_bound_matches_executor():
    """For each mode, the `Alu` LT sender asserts `addr_lo < bound(IS_AFFINE)` with
    `result = 1` and a literal zero high word, so only the low limb is compared — exactly
    what `addr_limb_ok` does (it ignores the high limb by construction).

    z3 is asked for a low limb where the two predicates DISAGREE, over the whole limb range.
    UNSAT means the AIR's accept set is the executor's accept set — no provable-but-halting
    execution, and no legal execution made unprovable."""
    ok = True
    for is_affine, span in [(0, 31), (1, 63)]:
        t0 = time.time()
        bound = addr_bound_by_mode(is_affine)
        s = z3.Solver()
        lo = z3.Int("lo")
        s.add(lo >= 0, lo < LIMB)
        # executor: lo + span < 2^32 ;  AIR: lo < bound
        s.add((lo + span < LIMB) != (lo < bound))
        r = s.check()
        ok &= r == z3.unsat
        report(f"A4a LT bound == addr_limb_ok [IS_AFFINE={is_affine}, span={span}]",
               "PROVED" if r == z3.unsat else str(r).upper(),
               f"bound = {bound} = 2^32 − {LIMB - bound}; {time.time()-t0:.2f}s")
    # and the linear form really evaluates to those two numbers
    ok &= addr_bound_by_mode(0) == ADDR_LIMB_BOUND_32B
    ok &= addr_bound_by_mode(1) == ADDR_LIMB_BOUND_64B
    report("A4a linear bound interpolates the two constants",
           "PROVED" if ok else "FAIL",
           f"BOUND_32B + a·(BOUND_64B − BOUND_32B) = {ADDR_LIMB_BOUND_32B} at a=0, "
           f"{ADDR_LIMB_BOUND_64B} at a=1")
    return ok


# ── A4b: the affine span cannot cross the limb ────────────────────────────

def a4b_span_safe():
    """Every byte the affine ops touch must have a low-limb address `< 2^32`, or the reused
    high limb would name the wrong page.

    The worst byte is the last of the highest dword: base `addr_lo + 56`, byte `+7`, i.e.
    `addr_lo + 63`. Under `addr_lo < 2^32 − 63` that is `< 2^32`. Proved over the whole
    accepted range for all 4 dwords × 8 bytes of both the `yG` read and the `yR` write, and
    the pre-existing x-only ops are re-checked under the same bound (they moved: their
    address column is now bounded by the affine constant on affine rows)."""
    t0 = time.time()
    s = z3.Solver()
    lo = z3.Int("lo")
    s.add(lo >= 0, lo < addr_bound_by_mode(1))   # accepted affine addresses
    touched = []
    for offs in (AFFINE_YG_READ_OFFSETS, AFFINE_YR_WRITE_OFFSETS,
                 XONLY_XG_READ_OFFSETS, XONLY_XR_WRITE_OFFSETS):
        for off in offs:
            for b in range(DWORD):
                touched.append(off + b)
    s.add(z3.Or([lo + t >= LIMB for t in touched]))   # deny: some byte escapes the limb
    r = s.check()
    ok = r == z3.unsat
    report("A4b affine span stays inside the limb", "PROVED" if ok else str(r).upper(),
           f"{len(touched)} touched byte offsets (max +{max(touched)}) all < 2^32 for every "
           f"accepted addr_lo; {time.time()-t0:.2f}s")
    return ok


# ── A4c: the band the LT senders close ────────────────────────────────────

def a4c_band():
    """The comment's claim, measured. Without the LT sender, what saves the bus is only that
    every dword BASE must be a legal field element under `2^32` for the tuple to be
    satisfiable — the largest base, not the largest byte. So the addresses the bus accepts and
    the executor rejects are

        [2^32 − span − 1 + 1, 2^32 − max_base)   =   [2^32 − (span+1), 2^32 − max_base)

    which for a 32-byte operand is `[2^32−31, 2^32−24)` and for a 64-byte one
    `[2^32−63, 2^32−56)`. Seven values each, as claimed."""
    facts = {}
    for label, offs, span in [
        ("32-byte (x-only xG/xR, k)", XONLY_XG_READ_OFFSETS, 31),
        ("64-byte (affine xG‖yG, xR‖yR)",
         XONLY_XG_READ_OFFSETS + AFFINE_YG_READ_OFFSETS, 63),
    ]:
        max_base = max(offs)
        band = [lo for lo in range(LIMB - span - 8, LIMB)
                if not addr_limb_ok(lo, span) and lo + max_base < LIMB]
        facts[label] = (max_base, len(band), band[0] if band else None,
                        band[-1] if band else None)
    ok = all(v[1] == 7 for v in facts.values())
    detail = "; ".join(
        f"{k}: max dword base +{v[0]}, band size {v[1]} "
        f"([2^32−{LIMB - v[2]}, 2^32−{LIMB - v[3] - 1}])"
        for k, v in facts.items())
    report("A4c seven-value band, both modes", "PROVED" if ok else "FAIL", detail)
    return ok


def a4c_band_is_closed():
    """And the LT senders close it: no address in either band satisfies `lo < bound`."""
    ok = True
    for is_affine, span, offs in [
        (0, 31, XONLY_XG_READ_OFFSETS),
        (1, 63, XONLY_XG_READ_OFFSETS + AFFINE_YG_READ_OFFSETS),
    ]:
        bound = addr_bound_by_mode(is_affine)
        max_base = max(offs)
        band = [lo for lo in range(LIMB - span - 8, LIMB)
                if not addr_limb_ok(lo, span) and lo + max_base < LIMB]
        ok &= all(lo >= bound for lo in band)
    report("A4c band closed by the LT senders", "PROVED" if ok else "FAIL",
           "every band address fails `addr_lo < bound` ⇒ the bus now rejects exactly what "
           "the executor rejects (the gap `hint.rs` closes for the Hint ecall)")
    return ok


# ── A4d: k's bound ────────────────────────────────────────────────────────

def a4d_scalar_bound():
    """`k` is a 32-byte scalar in BOTH modes, so its LT sender uses the flat
    `ADDR_LIMB_BOUND_32B`, with no `IS_AFFINE` term. Checked against the executor, which
    calls `addr_limb_ok(addr_k, 31)` on both arms."""
    t0 = time.time()
    s = z3.Solver()
    lo = z3.Int("lo")
    s.add(lo >= 0, lo < LIMB)
    s.add((lo + 31 < LIMB) != (lo < ADDR_LIMB_BOUND_32B))
    r = s.check()
    ok = r == z3.unsat
    report("A4d k bound flat in both modes", "PROVED" if ok else str(r).upper(),
           f"ADDR_K_0 < {ADDR_LIMB_BOUND_32B} == addr_limb_ok(addr_k, 31); "
           f"{time.time()-t0:.2f}s")
    return ok


# ── A4e: the overlap guard ────────────────────────────────────────────────

def a4e_overlap_guard():
    """The guard must be EXACT interval disjointness. A distance bound would be wrong in a
    way that shows up as an ABI wart rather than a crash: the two operands have different
    sizes, so a scalar placed immediately below the point (`addr_k + 32 == addr_xg`) is
    disjoint at distance 32, and a `< 64` bound would reject it — making the ecall's
    acceptance depend on which operand the guest's compiler laid out first.

    z3 over unbounded integers, so this is about the `u128` form's algebra, not about any
    machine width."""
    t0 = time.time()
    s = z3.Solver()
    xg, k = z3.Ints("xg k")
    s.add(xg >= 0, k >= 0)
    overlaps = z3.And(k < xg + 64, xg < k + 32)
    # the set-theoretic statement: ∃ a byte in both ranges
    a = z3.Int("a")
    truly_overlaps = z3.And(a >= xg, a < xg + 64, a >= k, a < k + 32)
    s.add(z3.Or(z3.And(overlaps, z3.Not(z3.Exists([a], truly_overlaps))),
                z3.And(z3.Not(overlaps), z3.Exists([a], truly_overlaps))))
    r = s.check()
    ok = r == z3.unsat
    report("A4e overlap guard == interval disjointness",
           "PROVED" if ok else str(r).upper(),
           f"the u128 clause `k < xg+64 ∧ xg < k+32` is exactly "
           f"`[xg,xg+64) ∩ [k,k+32) ≠ ∅`; {time.time()-t0:.2f}s")
    return ok


def a4e_wrap_control():
    """NEGATIVE CONTROL for the `u128` widening. The pre-fix `u64` form computes `addr_xg + 64`
    with wrapping, so at `addr_xg = 2^64 − 64` the first clause is `addr_k < 0` — vacuously
    false — and the guard is skipped entirely.

    Reachable: that address PASSES `addr_limb_ok(·, 63)`, whose low limb is `0xFFFFFFC0` and
    `0xFFFFFFC0 + 63 = 0xFFFFFFFF < 2^32`. The executor's checks run in that order, so nothing
    earlier rejects it. Worst case is a total overlap (`addr_k == addr_xg`), where the trace
    builder reads the same address at `ts` and `ts+1` and the MEMW consistency argument cannot
    prove the chain."""
    wrap = 2**64 - 64
    facts = {
        "wrap address passes addr_limb_ok(·, 63)": addr_limb_ok(wrap, 63),
        "u64 form skips the guard (accepts)": operands_disjoint_u64_buggy(wrap, wrap),
        "u128 form catches it (rejects)": not operands_disjoint(wrap, wrap),
        "the overlap is total, not marginal": True,   # addr_k == addr_xg
    }
    # and the whole wrapping band, not just one address
    band = [a for a in range(2**64 - 64, 2**64)
            if addr_limb_ok(a, 63) and operands_disjoint_u64_buggy(a, a)
            and not operands_disjoint(a, a)]
    facts["the band is non-empty"] = len(band) > 0
    bad = [k for k, v in facts.items() if not v]
    report("A4e control [u64 wrap in the overlap guard]",
           "SAT — FORGES" if not bad else "FAIL",
           f"{len(band)} reachable addresses in [2^64−64, 2^64) pass addr_limb_ok, wrap the "
           "pre-fix `+64`, and slip a TOTAL operand overlap past the guard ⇒ the u128 "
           "widening is LOAD-BEARING"
           if not bad else f"failed: {bad}")
    return not bad


# ── A4f: timestamps ──────────────────────────────────────────────────────

def a4f_timestamps():
    """The four sub-timestamps and their stride. `xG` and `yG` are both read at `ts` — legal,
    because they sit at disjoint addresses inside the same buffer, and MEMW only forbids the
    same address at two timestamps. `yR`'s write takes `ts + 3`, the last free slot before the
    next instruction's `ts + 4`."""
    slots = {"xG read": TS_XG_READ, "yG read": TS_XG_READ, "k read": TS_K_READ,
             "xR write": TS_XR_WRITE, "yR write": TS_YR_WRITE}
    yg_addrs = set()
    for off in AFFINE_YG_READ_OFFSETS:
        yg_addrs |= {off + b for b in range(DWORD)}
    xg_addrs = set()
    for off in XONLY_XG_READ_OFFSETS:
        xg_addrs |= {off + b for b in range(DWORD)}
    facts = {
        "all slots < the instruction stride": max(slots.values()) < INSTRUCTION_TS_STRIDE,
        "yR takes the last free slot": TS_YR_WRITE == INSTRUCTION_TS_STRIDE - 1,
        "xG and yG share ts but not an address": not (xg_addrs & yg_addrs),
        "xG‖yG jointly cover [0, 64)": xg_addrs | yg_addrs == set(range(64)),
        "the three distinct read/write times are distinct":
            len({TS_XG_READ, TS_K_READ, TS_XR_WRITE, TS_YR_WRITE}) == 4,
    }
    bad = [k for k, v in facts.items() if not v]
    report("A4f timestamp layout", "PROVED" if not bad else "FAIL",
           f"slots {slots} within stride {INSTRUCTION_TS_STRIDE}; xG@ts and yG@ts are "
           "address-disjoint so the same-timestamp reads are legal"
           if not bad else f"failed: {bad}")
    return not bad


# ── A4g: why the bound must be mode-dependent ────────────────────────────

def a4g_completeness():
    """The bound could have been a flat 64-byte one, which is *sound* — but it would reject
    x-only addresses the executor accepts, i.e. break completeness on a path this PR is not
    supposed to touch. Measured: 32 such addresses per operand.

    Conversely a flat 32-byte bound leaves the affine band open (A4c), so neither constant
    works alone and the `IS_AFFINE` interpolation is doing real work."""
    flat64_rejects_legal_xonly = [
        lo for lo in range(LIMB - 64, LIMB)
        if addr_limb_ok(lo, 31) and not (lo < ADDR_LIMB_BOUND_64B)]
    flat32_leaves_affine_open = [
        lo for lo in range(LIMB - 64, LIMB)
        if not addr_limb_ok(lo, 63) and lo < ADDR_LIMB_BOUND_32B]
    ok = len(flat64_rejects_legal_xonly) > 0 and len(flat32_leaves_affine_open) > 0
    report("A4g mode-dependent bound is necessary", "PROVED" if ok else "FAIL",
           f"a flat 64-byte bound would reject {len(flat64_rejects_legal_xonly)} legal "
           f"x-only addresses (completeness); a flat 32-byte bound would admit "
           f"{len(flat32_leaves_affine_open)} illegal affine ones (soundness) ⇒ the "
           "IS_AFFINE interpolation is load-bearing in BOTH directions")
    return ok


def main():
    a4a_bound_matches_executor()
    a4b_span_safe()
    a4c_band()
    a4c_band_is_closed()
    a4d_scalar_bound()
    a4e_overlap_guard()
    a4e_wrap_control()
    a4f_timestamps()
    a4g_completeness()

    print("\nSummary:")
    for n, v, _ in results:
        print(f"  {v:14} {n}")
    bad = [n for n, v, _ in results if v not in ("PROVED", "SAT — FORGES")]
    if bad:
        print("\nUNEXPECTED: " + ", ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
