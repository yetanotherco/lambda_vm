"""Anchor D: differential of the repo's `ecsm::lincomb2_witness` (phase A)
against the Python lincomb2 oracle, via the repo-harness `lincomb2` command.

This is the lineage crossing phase A's own tests cannot make: those validate
the Rust witness against two references that are themselves in Rust. Here the
Rust witness is compared, field by field and ROW BY ROW, against
`lincomb2_ref.lincomb2_rows` (affine/ec_ref lineage) with `jacobian_ref`
(Jacobian/LSB-first lineage) as the third opinion on Q.

Checks:
  1. Q, `len`, `P12`, `T0` and `2^len·T0` agree with the Python reference.
  2. Every emitted row agrees on (sel, round, op, d1, d2, nb, a, addend, λ, r) —
     including the two places the accumulator does NOT telescope (the
     precompute row's `a` is P1, the correction row's addend is −2^len·T0) —
     and `nb` is mirrored into the reused `EcdasStep::next_op` column.
  3. The canonicalization witnesses (`y_p2_sub_p`, `x_q_sub_p`, `y_q_sub_p`,
     `u1_sub_n`, `u2_sub_n`) really equal `2^256 + v − modulus`.
  4. Error paths: every `Lincomb2Error` variant is produced for the input class
     that should produce it.

Usage:
    <venv>/bin/python anchor_d_lincomb2_repo.py [path-to-harness-binary]

Default harness path is the in-tree build; pass an explicit path to run against
a harness built from a pinned commit (which is how this was first run, so the
differential measured phase A as committed rather than as concurrently edited).
"""

import json
import random
import subprocess
import sys

import jacobian_ref
import lincomb2_ref
from ec_ref import GX, GY, N, P, recover_even_y

HARNESS = "repo-harness/target/release/ecsm-oracle-harness"
rng = random.Random(4242)
T0, _T0_COUNTER = lincomb2_ref.t0_ref()


def run(harness, lines):
    r = subprocess.run([harness], input="\n".join(lines) + "\n",
                       capture_output=True, text=True, check=True)
    return r.stdout.splitlines()


def random_point():
    while True:
        x = rng.randrange(P)
        y = recover_even_y(x)
        if y is not None:
            break
    return (x, (P - y) % P if rng.random() < 0.5 else y)


def cmd(u1, u2, p1, p2):
    return f"lincomb2 {u1:x} {u2:x} {p1[0]:x} {p1[1]:x} {p2[0]:x} {p2[1]:x}"


def sub_witness(v, modulus):
    return (1 << 256) + v - modulus


def main(harness=HARNESS):
    fails = 0

    def check(cond, msg):
        nonlocal fails
        if not cond:
            fails += 1
            print(f"FAIL {msg}")

    # ── 1-3. positive differential ───────────────────────────────────────────
    G = (GX, GY)
    cases = []
    for i in range(150):
        p1 = G if i % 2 == 0 else random_point()
        p2 = random_point()
        if p1[0] == p2[0]:
            continue
        cases.append((rng.randrange(1, N), rng.randrange(1, N), p1, p2))
    for u1, u2 in [(1, 1), (1, 2), (2, 1), (3, 5), (7, 8), (16, 15),
                   (N - 1, N - 1), (N - 1, 1), (1, N - 1),
                   (2**255, 2**255 - 1), (2**128, 2**128 + 1)]:
        cases.append((u1, u2, G, random_point()))

    lines = [cmd(*c) for c in cases]
    outs = run(harness, lines)
    check(len(outs) == len(cases), f"harness returned {len(outs)} lines for {len(cases)} cases")

    rows_compared = 0
    for (u1, u2, p1, p2), out in zip(cases, outs):
        tag = f"u1={u1:x} u2={u2:x} xP2={p2[0]:x}"
        if not out.startswith("lincomb2_json "):
            fails += 1
            print(f"FAIL rust rejected a valid case ({out}) {tag}")
            continue
        w = json.loads(out[len("lincomb2_json "):])

        q_py, length, rows = lincomb2_ref.lincomb2_rows(u1, p1, u2, p2, T0)
        q_jac = jacobian_ref.lincomb2(u1, p1, u2, p2)

        check(int(w["x_q"], 16) == q_py[0] and int(w["y_q"], 16) == q_py[1],
              f"Q mismatch rust vs python {tag}")
        check(q_jac == q_py, f"Q mismatch python vs jacobian {tag}")
        check(w["len"] == length, f"len mismatch rust={w['len']} py={length} {tag}")

        p12 = rows[0]["r"]
        check(int(w["x_p12"], 16) == p12[0] and int(w["y_p12"], 16) == p12[1],
              f"P12 mismatch {tag}")
        check((int(w["x_t0"], 16), int(w["y_t0"], 16)) == T0, f"T0 mismatch {tag}")

        tpow = lincomb2_ref.pt_neg(rows[-1]["addend"])
        check((int(w["x_t0_pow"], 16), int(w["y_t0_pow"], 16)) == tpow,
              f"2^len*T0 mismatch {tag}")

        # canonicalization / range witnesses
        check(int(w["y_p2_sub_p"], 16) == sub_witness(p2[1], P), f"y_p2_sub_p {tag}")
        check(int(w["x_q_sub_p"], 16) == sub_witness(q_py[0], P), f"x_q_sub_p {tag}")
        check(int(w["y_q_sub_p"], 16) == sub_witness(q_py[1], P), f"y_q_sub_p {tag}")
        check(int(w["u1_sub_n"], 16) == sub_witness(u1, N), f"u1_sub_n {tag}")
        check(int(w["u2_sub_n"], 16) == sub_witness(u2, N), f"u2_sub_n {tag}")

        # row-by-row
        rr = w["rows"]
        if len(rr) != len(rows):
            fails += 1
            print(f"FAIL row count rust={len(rr)} py={len(rows)} {tag}")
            continue
        for i, (a, b) in enumerate(zip(rr, rows)):
            same = (
                a["sel"] == b["sel"]
                and a["round"] == b["round"]
                and a["op"] == b["op"]
                and a["d1"] == b["d1"]
                and a["d2"] == b["d2"]
                and a["nb"] == b["nb"]
                # nb is mirrored into the reused next_op column
                and a["next_op"] == b["nb"]
                and (int(a["x_a"], 16), int(a["y_a"], 16)) == b["a"]
                and (int(a["x_b"], 16), int(a["y_b"], 16)) == b["addend"]
                and int(a["lambda"], 16) == b["lam"]
                and (int(a["x_r"], 16), int(a["y_r"], 16)) == b["r"]
            )
            if not same:
                fails += 1
                print(f"FAIL row {i} ({a['sel']}/{b['sel']}) mismatch {tag}")
                break
            rows_compared += 1

    # ── 4. error paths ───────────────────────────────────────────────────────
    p2 = random_point()
    while p2[0] == GX:
        p2 = random_point()
    off_curve = (p2[0], (p2[1] + 1) % P)
    non_canon = (p2[0], p2[1] + P)  # same point mod p, non-canonical bytes
    neg_g = (GX, (P - GY) % P)
    err_cases = [
        ("ScalarIsZero", cmd(0, 5, G, p2)),
        ("ScalarIsZero", cmd(5, 0, G, p2)),
        ("ScalarOutOfRange", cmd(N, 5, G, p2)),
        ("ScalarOutOfRange", cmd(5, N + 1, G, p2)),
        ("PointNotOnCurve", cmd(3, 5, G, off_curve)),
        ("PointNotCanonical", cmd(3, 5, G, non_canon)),
        ("SumDegenerate", cmd(3, 5, G, G)),
        ("SumDegenerate", cmd(3, 5, G, neg_g)),
    ]
    outs = run(harness, [c for _, c in err_cases])
    for (want, c), out in zip(err_cases, outs):
        check(out == f"err {want}", f"error path: got '{out}' want 'err {want}' for `{c[:40]}...`")

    print(f"ANCHOR D (repo lincomb2_witness vs python oracle): "
          f"{'PASS' if fails == 0 else 'FAIL'} "
          f"({len(cases)} witnesses, {rows_compared} rows compared field-by-field, "
          f"{len(err_cases)} error paths, {fails} failures)")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else HARNESS))
