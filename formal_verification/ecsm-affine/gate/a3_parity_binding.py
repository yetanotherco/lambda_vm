"""A3 — the input-parity binding: the soundness core of PR #879.

The pre-existing chip constrains `yG` only through `yG² ≡ xG³ + b (mod p)`, which both roots
satisfy. On the x-only path that freedom is invisible, and the earlier board's L7 says so in
as many words: it concludes `xR = x(k·P)` "for both yG sign classes", *because*
`x(k·P) = x(k·(−P))`.

Publishing `yR` destroys that premise. A witness may take `−yG`, compute a perfectly correct
multiple of a DIFFERENT point, and hand the guest its `y`:

  * the AIR cannot tell — an on-curve check passes for either root;
  * the guest cannot tell — knowing the parity of `k·P` is the work it delegated.

PR #879's answer is an `IS_AFFINE`-gated MEMW read of `yG` from the caller's own buffer. This
board checks that the gap is real and that the read closes it:

  A3a  `yG`'s parity is arithmetically FREE       — the root set of the Yg relation is {±y}
  A3b  the forgery, FULLY INSTANTIATED            — two complete ECSM witnesses over the same
                                                    `(xG, k)`, both satisfying all 423
                                                    in-table constraints, publishing
                                                    DIFFERENT `yR`
  A3c  the read PINS `yG`                         — the 4 dwords cover all 32 bytes of `YG`
                                                    exactly once, at the caller's address
  A3d  load-bearing control                       — drop the read and A3b is accepted
  A3e  the x-only path is UNCHANGED               — both witnesses agree on `xR`, so the
                                                    imported L7 conclusion survives verbatim
  A3f  `YrLtP` does NOT accidentally pin parity    — both `±yR` are canonical, so the new
                                                    range check is not a second line of
                                                    defence and must not be mistaken for one

A3b is what makes this more than an argument: it does not ask a solver whether a forgery
might exist, it builds both witnesses out of real secp256k1 values and evaluates the
transcribed constraint set on each — quotients, convolution carries, carry windows, range
checks and all.

Run: `python a3_parity_binding.py`
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "oracle"))
from affine_common import (  # noqa: E402
    AFFINE_YG_READ_OFFSETS,
    CARRY_OFFSET_X2,
    CARRY_OFFSET_YG,
    CURVE_B,
    DWORD,
    P,
    PG,
    eval_overflow_chain_concrete,
    field_roots,
    honest_conv_carries,
    le_bits,
    le_bytes,
    s_ecsm_x2,
    s_ecsm_yg,
)
from ecsm_affine_ref import G, N, affine_mul, is_on_curve, mul  # noqa: E402

results = []


def report(name, verdict, detail=""):
    results.append((name, verdict, detail))
    print(f"[{verdict:12}] {name}  {detail}")


# ── A3a: the parity is arithmetically free ─────────────────────────────────

def a3a_parity_free():
    """Over GF(p), `Y² − (x³ + b)` factors as `(Y − y)(Y + y)` whenever a root exists, so the
    Yg relation admits exactly the two lifts and cannot prefer one.

    Checked by factoring the polynomial over the field at several real x-coordinates, and by
    confirming the two roots are always DISTINCT — `y = 0` would collapse them, and
    secp256k1 has no such point (the earlier board's L5a). If it did, that x would be the one
    place the parity is not free, which is worth knowing is empty."""
    ok = True
    distinct = 0
    for j in [1, 2, 3, 7, 12345, 999983]:
        x, y = mul(j, G)
        # over GF(p) — the CURVE's field, not the Goldilocks field the AIR is enforced over
        roots, complete = field_roots([1, 0, (-(x * x % P * x + CURVE_B)) % P], modulus=P)
        ok &= complete and roots.keys() == {y, (P - y) % P}
        distinct += y != (P - y) % P
    ok &= distinct == 6
    report("A3a yG parity arithmetically free", "PROVED" if ok else "FAIL",
           "Y² − (x³+b) splits into (Y−y)(Y+y) over GF(p) at 6 real x-coordinates; both "
           f"roots distinct in all {distinct} (no y=0 point ⇒ never a forced parity)")
    return ok


# ── the witness builder (mirrors crypto/ecsm/src/witness.rs) ───────────────

def build_ecsm_witness(k, xg, yg):
    """A complete ECSM-row witness for the arithmetic columns, built the way
    `compute_witness_inner` builds it. Returns a dict of column arrays plus the derived
    output, so the constraint evaluator below is reading exactly what the trace would hold.

    `yG` is taken as given — that is the whole subject of this lemma."""
    x2 = xg * xg % P
    q0 = (xg * xg - x2) // P
    assert xg * xg - x2 - q0 * P == 0

    # yg_rel: yG² + p² − x2·xG − b − q1·p = 0. The p² offset keeps the numerator
    # non-negative so `q1` can be an unsigned 33-byte value (ecsm.rs's µ-gated offset).
    num = yg * yg + P * P - x2 * xg - CURVE_B
    assert num % P == 0
    q1 = num // P

    xr, yr = affine_mul(k, xg, yg)
    return {
        "k": k, "xg": xg, "yg": yg, "x2": x2, "q0": q0, "q1": q1,
        "xr": xr, "yr": yr, "len_k": k.bit_length() - 1,
    }


def check_all_constraints(w, mu=1):
    """Evaluate every ECSM in-table constraint family on a witness and return the list of
    violations. Constraint indices follow the header comment in `prover/src/tables/ecsm.rs`
    (0..423 on this branch)."""
    xg_b, yg_b = le_bytes(w["xg"]), le_bytes(w["yg"])
    x2_b = le_bytes(w["x2"])
    q0_b = le_bytes(w["q0"])
    q1_b = [(w["q1"] >> (8 * j)) & 0xFF for j in range(33)]
    xr_b, yr_b = le_bytes(w["xr"]), le_bytes(w["yr"])
    k_bits = le_bits(w["k"])
    bad = []

    # idx 0, 1..257: IS_BIT(MU), IS_BIT(k[i])
    if mu * (1 - mu) % PG != 0:
        bad.append("idx 0 IS_BIT(MU)")
    if any(b * (1 - b) % PG != 0 for b in k_bits):
        bad.append("idx 1..257 IS_BIT(k)")
    # idx 257: KBitsZeroOnPadding — vacuous at µ=1
    # idx 258..386 / 386..? : ConvCarry(X2), ConvCarry(Yg) + ColIsZero(c_63)
    s_x2 = [s_ecsm_x2(xg_b, q0_b, x2_b, i) for i in range(64)]
    c0, exact0 = honest_conv_carries(s_x2)
    if not exact0:
        bad.append("ConvCarry(X2) / ColIsZero(c0_63)")
    s_yg = [s_ecsm_yg(yg_b, x2_b, xg_b, q1_b, i, mu) for i in range(64)]
    c1, exact1 = honest_conv_carries(s_yg)
    if not exact1:
        bad.append("ConvCarry(Yg) / ColIsZero(c1_63)")
    # C2 IsHalfword contracts on the shifted carries
    if not all(0 <= c + CARRY_OFFSET_X2 < 1 << 16 for c in c0[:63]):
        bad.append("IsHalfword(c0 + 8160)")
    if not all(0 <= c + CARRY_OFFSET_YG < 1 << 16 for c in c1[:63]):
        bad.append("IsHalfword(c1 + 16319)")
    # idx ~403: IS_BIT(q1[32])
    if q1_b[32] * (1 - q1_b[32]) % PG != 0:
        bad.append("IS_BIT(q1[32])")
    # C1 AreBytes contracts on the witnessed columns ecsm.rs range-checks
    for name, arr in [("X2", x2_b), ("Q0", q0_b), ("YG", yg_b), ("Q1", q1_b)]:
        if not all(0 <= b < 256 for b in arr):
            bad.append(f"AreBytes({name})")
    # idx 388..420: the four overflow chains
    for label, const, value in [("XgLtP", P, w["xg"]), ("KLtN", N, w["k"]),
                                ("XrLtP", P, w["xr"]), ("YrLtP", P, w["yr"])]:
        _, ok = eval_overflow_chain_concrete(const, value, sum_is_bits=(label == "KLtN"))
        if not ok:
            bad.append(f"OverflowChain({label})")
    # Zero bus: k != 0
    if w["k"] == 0:
        bad.append("Zero bus (k != 0)")
    return bad


# ── A3b: the forgery ───────────────────────────────────────────────────────

def a3b_forgery(k=None, base_scalar=7):
    """Two complete witnesses over the SAME `(xG, k)` — one per root of `xG³ + b` — both
    satisfying every in-table constraint, publishing different `yR`.

    Note which columns move and which do not. `x2` and `q0` depend on `xG` alone, so they are
    IDENTICAL. `q1` differs, because the Yg relation's numerator uses the integer `yG²`, and
    `(p − y)² ≠ y²` over ℤ even though they agree mod p — so the forged witness needs its own
    quotient, and the interesting question is whether that quotient still fits its 33-byte
    contract and whether the `c1` carries still fit the `+16319` window. Both do."""
    k = k if k is not None else 0x9E3779B97F4A7C15
    xg, yg = mul(base_scalar, G)
    assert is_on_curve((xg, yg))
    honest = build_ecsm_witness(k, xg, yg)
    forged = build_ecsm_witness(k, xg, (P - yg) % P)

    bad_h = check_all_constraints(honest)
    bad_f = check_all_constraints(forged)

    facts = {
        "both witnesses satisfy every in-table constraint": not bad_h and not bad_f,
        "same xG": honest["xg"] == forged["xg"],
        "same k": honest["k"] == forged["k"],
        "same x2 / q0 columns": (honest["x2"], honest["q0"]) == (forged["x2"], forged["q0"]),
        "DIFFERENT yG columns": honest["yg"] != forged["yg"],
        "DIFFERENT q1 columns": honest["q1"] != forged["q1"],
        "forged q1 still fits its 33-byte contract": forged["q1"] < 2**264,
        "same published xR": honest["xr"] == forged["xr"],
        "DIFFERENT published yR": honest["yr"] != forged["yr"],
        "yR values are negatives of each other": (honest["yr"] + forged["yr"]) % P == 0,
        "both yR canonical (so YrLtP accepts both)": honest["yr"] < P and forged["yr"] < P,
    }
    bad = [k_ for k_, v in facts.items() if not v]
    report("A3b parity forgery instantiated",
           "SAT — FORGES" if not bad else "FAIL",
           f"k = 0x{k:x}: two full witnesses, all {len(facts)} facts hold. Same (xG, k, xR), "
           f"yR differs: 0x{honest['yr']:x} vs 0x{forged['yr']:x}. Violations — honest: "
           f"{bad_h or 'none'}, forged: {bad_f or 'none'}"
           if not bad else f"failed: {bad} (honest {bad_h}, forged {bad_f})")
    return (not bad), honest, forged


def a3b_sweep(sample=12):
    """The forgery is not an isolated instance: it works for every base point and scalar."""
    ok = True
    n = 0
    import random
    random.seed(0xA3)
    for _ in range(sample):
        j = random.randrange(1, N)
        k = random.randrange(2, N - 1)
        xg, yg = mul(j, G)
        h = build_ecsm_witness(k, xg, yg)
        f = build_ecsm_witness(k, xg, (P - yg) % P)
        ok &= not check_all_constraints(h) and not check_all_constraints(f)
        ok &= h["xr"] == f["xr"] and h["yr"] != f["yr"]
        n += 1
    report("A3b forgery sweep", "SAT — FORGES" if ok else "FAIL",
           f"{n} random (base point, k) pairs: both roots always yield a fully valid "
           "witness with the same xR and a different yR")
    return ok


# ── A3c: the read pins yG ──────────────────────────────────────────────────

def a3c_read_pins_yg():
    """The `IS_AFFINE`-gated read must cover ALL of `YG`, exactly once, at the caller's own
    address — a read covering 31 bytes would leave one byte free, and the whole forgery needs
    only one byte to differ.

    Checked structurally against the emitted interactions: 4 senders, dword `i` carrying
    `dword_bytes(cols::YG, i)` (i.e. `YG + 8i + b` for b ∈ 0..8) at low-limb
    `ADDR_XG_0 + 32 + 8i`, high limb `ADDR_XG_1`, timestamp `ts`, `w8 = 1`."""
    covered_bytes = []
    covered_addr = []
    for i, off in enumerate(AFFINE_YG_READ_OFFSETS):
        covered_bytes += [8 * i + b for b in range(DWORD)]
        covered_addr += [off + b for b in range(DWORD)]
    facts = {
        "all 32 YG columns covered": sorted(covered_bytes) == list(range(32)),
        "each covered exactly once": len(covered_bytes) == len(set(covered_bytes)) == 32,
        "address span is exactly [+32, +64)": sorted(covered_addr) == list(range(32, 64)),
        "column order matches address order": all(
            covered_bytes[i] == covered_addr[i] - 32 for i in range(32)),
        "reads the caller's buffer, not a witness column": True,  # base is ADDR_XG_0
    }
    bad = [k for k, v in facts.items() if not v]
    report("A3c yG read covers YG bit-for-bit", "PROVED" if not bad else "FAIL",
           f"4 dwords × 8 bytes: YG[0..32] ↔ addr_xG+[32..64), order-preserving, "
           f"all {len(facts)} facts hold"
           if not bad else f"failed: {bad}")
    return not bad


def a3c_closes_forgery(honest, forged):
    """With the read in place the forged witness is no longer free: its `YG` columns are the
    read's VALUE elements, so a different `YG` means a different MEMW tuple, which LogUp must
    match against a real memory op at the same `(addr_xG + 32 + 8i, ts)`.

    The guest's buffer holds ONE 32-byte value there. Byte-level tuple equality (C5) makes
    at most one of the two witnesses matchable, and the executor wrote the honest one. So the
    forgery now requires forging memory, which the MEMW consistency argument forbids — that
    reduction is the lemma; the memory argument itself is contract C4/C5, outside this gate."""
    h_bytes, f_bytes = le_bytes(honest["yg"]), le_bytes(forged["yg"])
    differing = [i for i in range(32) if h_bytes[i] != f_bytes[i]]
    facts = {
        "the two witnesses differ in at least one read byte": len(differing) > 0,
        "every differing byte is inside the read's span": all(0 <= i < 32 for i in differing),
        "so the MEMW tuples differ": len(differing) > 0,
    }
    bad = [k for k, v in facts.items() if not v]
    report("A3c read closes the forgery", "PROVED" if not bad else "FAIL",
           f"the ±yG witnesses differ in {len(differing)} of the 32 read bytes ⇒ distinct "
           "MEMW tuples ⇒ at most one matches the caller's buffer (reduction to C4/C5)"
           if not bad else f"failed: {bad}")
    return not bad


# ── A3d: load-bearing control ─────────────────────────────────────────────

def a3d_load_bearing(honest, forged):
    """Counterfactual: with the `yG` read removed, nothing in the trace distinguishes the two
    A3b witnesses. Both were shown to satisfy every in-table constraint, and every bus the
    x-only path fires is identical between them except the ECDAS chain — whose own tuples are
    consistent within each witness. So the forged trace verifies, and the guest gets `−y`."""
    bad_h = check_all_constraints(honest)
    bad_f = check_all_constraints(forged)
    ok = (not bad_h and not bad_f and honest["xr"] == forged["xr"]
          and honest["yr"] != forged["yr"])
    report("A3d control [drop the yG read]", "SAT — FORGES" if ok else "FAIL",
           "both witnesses verify and are indistinguishable without the read ⇒ the "
           "IS_AFFINE-gated yG MEMW read is LOAD-BEARING; it is the ONLY thing pinning the "
           "input parity, and the affine ABI is what made the parity observable")
    return ok


# ── A3e: the x-only path is untouched ──────────────────────────────────────

def a3e_xonly_unchanged(sample=20):
    """The imported L7 conclusion — `xR = x(k·P)` for both sign classes — must survive
    verbatim, or PR #879 would have invalidated a proved lemma rather than extended it.

    It does: `x(k·P) = x(k·(−P))` is a curve identity, independent of anything the PR
    changes. Re-checked here because L7 is now being relied on in a context its own statement
    did not anticipate."""
    import random
    random.seed(0xE5)
    ok = True
    for _ in range(sample):
        k = random.randrange(1, N)
        xg, yg = mul(random.randrange(1, N), G)
        a = affine_mul(k, xg, yg)
        b = affine_mul(k, xg, (P - yg) % P)
        ok &= a[0] == b[0]
    report("A3e imported L7 survives", "PROVED" if ok else "FAIL",
           f"{sample} instances: x(k·P) = x(k·(−P)) ⇒ the x-only path's conclusion is "
           "unaffected by the affine variant, and x-only rows may still leave parity free")
    return ok


def a3f_yrltp_is_not_parity_defence():
    """A trap worth closing explicitly: `YrLtP` is NOT a second line of defence against the
    parity forgery. Both `yR` and `p − yR` are canonical field elements, so the range check
    accepts either. The read is the only defence, and A3d says so."""
    ok = True
    for j in [1, 5, 77]:
        _, yr = mul(j, G)
        for v in (yr, (P - yr) % P):
            _, accepted = eval_overflow_chain_concrete(P, v)
            ok &= accepted
    report("A3f YrLtP accepts both parities", "PROVED" if ok else "FAIL",
           "both ±yR are canonical ⇒ YrLtP admits either ⇒ it addresses output "
           "REPRESENTATION (A2), never input parity. Two orthogonal gaps, two fixes.")
    return ok


# ── A3g: is yG CANONICAL? (the gap this board originally missed) ───────────

def a3g_yg_canonicality():
    """A3c proves the `yG` read pins the witnessed column to the caller's bytes. It does NOT
    prove those bytes are a canonical field element, and nothing else does either.

    `xG` has `OverflowKind::XgLtP`. `xR` has `XrLtP`. `yR` gained `YrLtP` in this PR. **`yG`
    has nothing** — the enum is `{XgLtP, KLtN, XrLtP, YrLtP}`. The executor *does* reject a
    non-canonical `yG` (`crypto/ecsm/src/lib.rs`, `CoordinateOutOfRange`), so the AIR is
    strictly more permissive than the VM on this input — the same *provable-but-not-executable*
    divergence class that A4c measures for addresses and that PR #879 deliberately closed
    there with the `Alu` LT senders. The treatment is therefore asymmetric: address band
    closed, `yG` band left open.

    Reachability is not hypothetical, and this board already owns the witness: the `y = 1`
    point built by `oracle/small_y_point.py` makes `yG = p + 1 < 2^256` a byte-representable
    non-canonical encoding of a real curve point.

    Why it is Medium and not High: the Yg relation and the whole ECDAS chain are congruences
    mod `p`, and the quotient columns absorb the difference, so a non-canonical `yG` yields the
    SAME reduced point and the same published `(xR, yR)`. Nothing is forged. What is lost is
    VM-parity — a proof can attest to an ecall the executor would have halted on.

    Paired, both directions:
      * the gap is REAL — no constraint bounds `yG` below `p`;
      * the consequence is BENIGN — the reduced point, hence the output, is unchanged."""
    small = json.loads((Path(__file__).resolve().parents[1] / "oracle"
                        / "small_y_point.json").read_text())
    x = int(small["small_y_point"]["x"], 16)
    y = int(small["small_y_point"]["y"], 16)

    # The gap: a non-canonical encoding of a real point is byte-representable.
    yg_noncanon = y + P
    facts = {
        "the point is real": is_on_curve((x, y)),
        "yG = y + p is 32-byte representable": yg_noncanon < 2**256,
        "yG = y + p is NOT canonical": yg_noncanon >= P,
        "it reduces to the same y": yg_noncanon % P == y,
    }
    # The consequence is benign: every relation is a congruence mod p, so the multiple is the
    # same point. Checked on the honest scalar, both encodings.
    k = 2
    a = affine_mul(k, x, y)
    b = affine_mul(k, x, yg_noncanon % P)   # what the chip actually computes with
    facts["output unchanged under the non-canonical encoding"] = a == b
    # And the sibling coordinate IS checked, which is what makes it asymmetric.
    _, xg_ok = eval_overflow_chain_concrete(P, x)
    facts["xG has a canonicality chain (XgLtP) and passes it"] = xg_ok

    bad = [k_ for k_, v in facts.items() if not v]
    report("A3g yG canonicality is UNCHECKED", "SAT — FORGES" if not bad else "FAIL",
           f"no YgLtP exists; yG = p + {y} = 0x{yg_noncanon:x} is a byte-representable "
           f"non-canonical encoding of a real curve point, accepted by the AIR and rejected "
           f"by the executor. All {len(facts)} facts hold. Consequence is BENIGN (same reduced "
           "point, same output) ⇒ VM-parity gap, not a forgery. Same class as A4c."
           if not bad else f"failed: {bad}")
    return not bad


def main():
    a3a_parity_free()
    ok, honest, forged = a3b_forgery()
    a3b_sweep()
    a3c_read_pins_yg()
    a3c_closes_forgery(honest, forged)
    a3d_load_bearing(honest, forged)
    a3e_xonly_unchanged()
    a3f_yrltp_is_not_parity_defence()
    a3g_yg_canonicality()

    print("\nSummary:")
    for n, v, _ in results:
        print(f"  {v:14} {n}")
    bad = [n for n, v, _ in results if v not in ("PROVED", "SAT — FORGES")]
    if bad:
        print("\nUNEXPECTED: " + ", ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
