"""A6 — the real-witness anchor: evaluate the transcribed model on witnesses produced by the
REPO's own generator.

The earlier board's rule, and a good one: no UNSAT is trusted until the transcribed model has
been evaluated on real prover witnesses. A model that is *stronger* than the chip yields UNSAT
where the chip is forgeable, and no amount of solver work notices — but a real witness that
fails the model does.

Two independent things are being anchored:

  * **the FUNCTION** — the Python oracle is a from-scratch reimplementation, so agreement
    means the gate reasons about the right scalar multiplication;
  * **the COLUMNS** — this file, which reads `ecsm::compute_witness{,_with_y}`'s actual output
    and re-derives every witness field the model consumes.

It also carries the campaign's most direct exhibit: the parity forgery, produced by the repo's
own witness generator rather than by the model. For every scalar, `compute_witness_with_y`
accepts BOTH roots of `xG³ + b` and returns two complete, internally consistent witnesses with
the same `x_r` and different `y_r`. Nothing in `crypto/ecsm` objects, and nothing in the AIR's
arithmetic objects either (A3b) — which is exactly why the `IS_AFFINE`-gated `yG` read has to
exist.

Input: `logs/real_witnesses.jsonl`, produced by

    cd ../harness && cargo run --release -- > ../gate/logs/real_witnesses.jsonl

Run: `python a6_real_witness.py`
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "oracle"))
from affine_common import (  # noqa: E402
    CARRY_OFFSET_X2,
    CARRY_OFFSET_YG,
    CURVE_B,
    N,
    P,
    eval_overflow_chain_concrete,
    honest_conv_carries,
    le_bytes,
    s_ecsm_x2,
    s_ecsm_yg,
)
from ecsm_affine_ref import affine_mul, is_on_curve, recover_y_canonical, x_only_mul  # noqa: E402

WITNESSES = Path(__file__).parent / "logs" / "real_witnesses.jsonl"
results = []


def report(name, verdict, detail=""):
    results.append((name, verdict, detail))
    print(f"[{verdict:12}] {name}  {detail}")


def le_hex(h):
    """The harness emits little-endian byte hex; recover the integer."""
    return int.from_bytes(bytes.fromhex(h), "little")


def load():
    if not WITNESSES.exists():
        return None
    out = []
    for line in WITNESSES.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


# ── the per-witness check ──────────────────────────────────────────────────

def check_witness(w):
    """Re-derive every field the gate's model reads. Returns a list of failures."""
    k = le_hex(w["k"])
    xg, yg = le_hex(w["x_g"]), le_hex(w["y_g"])
    xr, yr = le_hex(w["x_r"]), le_hex(w["y_r"])
    x2, q0, q1 = le_hex(w["x2"]), le_hex(w["q0"]), le_hex(w["q1"])
    bad = []

    # 1. the oracle agrees with the repo
    if w["mode"] == "x-only":
        if x_only_mul(k, xg) != xr:
            bad.append("oracle x_only_mul disagrees with x_r")
        if recover_y_canonical(xg) != yg:
            bad.append("x-only y_g is not the canonical even lift")
    else:
        if affine_mul(k, xg, yg) != (xr, yr):
            bad.append("oracle affine_mul disagrees with (x_r, y_r)")
    if not is_on_curve((xg, yg)):
        bad.append("input point not on curve")
    if not is_on_curve((xr, yr)):
        bad.append("output point not on curve")

    # 2. the three pre-existing overflow addends, plus the NEW y_r_sub_p
    for field, const, value in [("x_g_sub_p", P, xg), ("k_sub_n", N, k),
                                ("x_r_sub_p", P, xr), ("y_r_sub_p", P, yr)]:
        want = (2**256 + value - const) % 2**256
        if le_hex(w[field]) != want:
            bad.append(f"{field} != (2^256 + value − const) mod 2^256")
        _, ok = eval_overflow_chain_concrete(const, value,
                                             sum_is_bits=(field == "k_sub_n"))
        if not ok:
            bad.append(f"overflow chain {field}: some c_i ∉ {{0,1}} or c_7 != 1")

    # 3. the two convolution relations and their carry windows
    xg_b, yg_b, x2_b = le_bytes(xg), le_bytes(yg), le_bytes(x2)
    q0_b = le_bytes(q0)
    q1_b = [(q1 >> (8 * j)) & 0xFF for j in range(33)]
    if x2 != xg * xg % P:
        bad.append("x2 != xG² mod p")
    if q0 != (xg * xg - x2) // P:
        bad.append("q0 != (xG² − x2)/p")
    if q1 != (yg * yg + P * P - x2 * xg - CURVE_B) // P:
        bad.append("q1 != (yG² + p² − x2·xG − b)/p")
    c0, exact0 = honest_conv_carries([s_ecsm_x2(xg_b, q0_b, x2_b, i) for i in range(64)])
    c1, exact1 = honest_conv_carries(
        [s_ecsm_yg(yg_b, x2_b, xg_b, q1_b, i, 1) for i in range(64)])
    if not exact0:
        bad.append("X2 relation: inexact carry or c_63 != 0")
    if not exact1:
        bad.append("Yg relation: inexact carry or c_63 != 0")
    if not all(0 <= c + CARRY_OFFSET_X2 < 1 << 16 for c in c0[:63]):
        bad.append("c0 escapes its IsHalfword window")
    if not all(0 <= c + CARRY_OFFSET_YG < 1 << 16 for c in c1[:63]):
        bad.append("c1 escapes its IsHalfword window")
    if q1_b[32] not in (0, 1):
        bad.append("q1[32] is not a bit")

    # 4. len_k
    if w["len_k"] != k.bit_length() - 1:
        bad.append("len_k != MSB(k)")
    # 5. the ECDAS step count: one per double, plus one per set bit below the MSB
    want_steps = 0 if k.bit_length() <= 1 else (
        k.bit_length() - 1 + bin(k)[3:].count("1"))
    if w["steps"] != want_steps:
        bad.append(f"steps {w['steps']} != expected {want_steps}")
    return bad


def a6a_witnesses(rows):
    accepted = [r for r in rows if "error" not in r]
    rejected = [r for r in rows if "error" in r]
    failures = {}
    n_checks = 0
    for r in accepted:
        bad = check_witness(r)
        n_checks += 1
        if bad:
            failures[f"{r['label']}/{r['mode']}"] = bad
    report("A6a real-witness evaluation", "PROVED" if not failures else "FAIL",
           f"{n_checks} witnesses from ecsm::compute_witness{{,_with_y}}: oracle agreement, "
           "all four overflow chains, both convolution relations, carry windows, len_k and "
           "the ECDAS step count all re-derived"
           if not failures else f"failures: {dict(list(failures.items())[:3])}")
    return not failures, accepted, rejected


def a6b_rejections(rejected):
    """The executor's accept/reject set, from the repo's own generator."""
    want = {
        "k=0": "non-zero",
        "k=N": "< N",
        "off-curve yG": "curve",
        "yG=p": "< p",
    }
    got = {r["label"]: r["error"] for r in rejected}
    ok = set(got) == set(want)
    report("A6b rejections match the oracle", "PROVED" if ok else "FAIL",
           f"{len(got)} rejected by crypto/ecsm: {got}" if ok
           else f"expected {sorted(want)}, got {sorted(got)}")
    return ok


def a6c_parity_forgery_from_the_repo(accepted):
    """The campaign's central exhibit, sourced from the repo rather than the model: for every
    scalar, `compute_witness_with_y` accepts BOTH roots and returns two valid witnesses with
    the same `x_r` and a different `y_r`."""
    by_label = {}
    for r in accepted:
        by_label.setdefault(r["label"], {})[r["mode"]] = r
    pairs = [(lab, v["affine/+y"], v["affine/-y"]) for lab, v in by_label.items()
             if "affine/+y" in v and "affine/-y" in v]
    facts = {
        "at least one pair present": len(pairs) > 0,
        "both roots always accepted": all(
            "error" not in a and "error" not in b for _, a, b in pairs),
        "same x_r in every pair": all(a["x_r"] == b["x_r"] for _, a, b in pairs),
        "different y_r in every pair": all(a["y_r"] != b["y_r"] for _, a, b in pairs),
        "y_r values are negatives mod p": all(
            (le_hex(a["y_r"]) + le_hex(b["y_r"])) % P == 0 for _, a, b in pairs),
        "different q1 in every pair": all(a["q1"] != b["q1"] for _, a, b in pairs),
        "identical x2 / q0 (they depend on xG alone)": all(
            a["x2"] == b["x2"] and a["q0"] == b["q0"] for _, a, b in pairs),
        "both y_r canonical (YrLtP accepts either)": all(
            le_hex(a["y_r"]) < P and le_hex(b["y_r"]) < P for _, a, b in pairs),
    }
    bad = [k for k, v in facts.items() if not v]
    report("A6c parity forgery from the repo's generator",
           "SAT — FORGES" if not bad else "FAIL",
           f"{len(pairs)} ±yG witness pairs out of crypto/ecsm itself: every one is valid, "
           f"agrees on x_r, and publishes a different y_r ⇒ the gap A3 proves is not a "
           "modelling artefact"
           if not bad else f"failed: {bad}")
    return not bad


def a6d_xonly_equals_even_lift(accepted):
    """`G`'s y is even, so the x-only witness and the `affine/+y` witness over `G` must be
    IDENTICAL, field for field. If they diverged, the two paths would not be the same chip and
    A3e's "x-only is untouched" would be false."""
    by_label = {}
    for r in accepted:
        by_label.setdefault(r["label"], {})[r["mode"]] = r
    fields = ["x_g", "y_g", "x_r", "y_r", "x_g_sub_p", "k_sub_n", "x_r_sub_p",
              "y_r_sub_p", "x2", "q0", "q1", "len_k", "steps"]
    n = 0
    ok = True
    for lab, v in by_label.items():
        if "x-only" in v and "affine/+y" in v:
            if le_hex(v["x-only"]["x_g"]) != le_hex(v["affine/+y"]["x_g"]):
                continue  # different base point (the small-y instance)
            ok &= all(v["x-only"][f] == v["affine/+y"][f] for f in fields)
            n += 1
    report("A6d x-only == affine with the even lift", "PROVED" if ok and n else "FAIL",
           f"{n} labels over G (whose y is even): all {len(fields)} witness fields identical "
           "⇒ the affine variant is the same chip, not a parallel one")
    return ok and n > 0


def a6e_small_y_instance(accepted):
    """The `y = 1` instance reached through the repo's generator: `y_r` really is 1, and its
    `y_r_sub_p` sits at the extreme of the addend range — the exact witness A2's forgery
    perturbs."""
    hit = [r for r in accepted if r["label"].startswith("small-y")]
    if not hit:
        report("A6e small-y instance", "FAIL", "instance missing from the dump")
        return False
    r = hit[0]
    yr = le_hex(r["y_r"])
    addend = le_hex(r["y_r_sub_p"])
    ok = (yr == 1 and addend == (2**256 + 1 - P) % 2**256
          and yr + P < 2**256)
    report("A6e small-y instance", "PROVED" if ok else "FAIL",
           f"crypto/ecsm returns y_r = {yr} for the constructed point; y_r_sub_p = "
           f"0x{addend:x} = 2^256 + 1 − p, and y_r + p < 2^256 ⇒ the A2 forgery's premise "
           "holds against the real generator")
    return ok


def main():
    rows = load()
    if rows is None:
        report("A6 real-witness anchor", "SKIP",
               f"{WITNESSES.relative_to(Path(__file__).parent)} not found — run "
               "`cd ../harness && cargo run --release -- > ../gate/logs/real_witnesses.jsonl`")
        print("\nA6 SKIPPED: the anchor is NOT established without the harness dump.")
        return 0

    ok, accepted, rejected = a6a_witnesses(rows)
    a6b_rejections(rejected)
    a6c_parity_forgery_from_the_repo(accepted)
    a6d_xonly_equals_even_lift(accepted)
    a6e_small_y_instance(accepted)

    print("\nSummary:")
    for n, v, _ in results:
        print(f"  {v:14} {n}")
    bad = [n for n, v, _ in results if v not in ("PROVED", "SAT — FORGES", "SKIP")]
    if bad:
        print("\nUNEXPECTED: " + ", ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
