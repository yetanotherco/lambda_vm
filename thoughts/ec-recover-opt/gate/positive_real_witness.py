"""Positive control: REAL prover witnesses (crypto/ecsm compute_witness, dumped by
the extended oracle harness) must satisfy every transcribed constraint mod p_g,
every range contract, and every bus-side chaining relation.

This is the transcription-faithfulness anchor: any sign/index error in
gate_common's S_i builders or in this file's constraint enumeration fails HERE,
on honest data, before any UNSAT verdict is trusted.

Constraint enumeration mirrors:
  ECSM : ecsm.rs eval() idx 0..412 (413 total)   [ecsm.rs:829-899]
  ECDAS: ecdas.rs eval() idx 0..199 (200 total)  [ecdas.rs:416-451]
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gate_common import (
    B, N, P, PG, P_BYTES, N_BYTES, OFF, compose, load_witness_json,
    s_ecsm_x2, s_ecsm_yg, s_ecdas_lambda, s_ecdas_xr, s_ecdas_yr, conv_carry,
    GEN_X,
)

HARNESS = Path(__file__).parent.parent / "oracle/repo-harness/target/release/ecsm-oracle-harness"
INV32 = pow(1 << 32, PG - 2, PG)

checks = 0
failures = []


def ck(cond, what):
    global checks
    checks += 1
    if not cond:
        failures.append(what)


def fe(x):
    """Integer -> Goldilocks field element (negatives wrap)."""
    return x % PG


def eval_ecsm(w):
    """All 413 ECSM constraints (µ=1 live row) + range/consistency contracts."""
    n_constraints = 0
    kbits = [(w["k"][b // 8] >> (b % 8)) & 1 for b in range(256)]
    mu = 1

    # idx 0: IS_BIT(MU); idx 1..256: IS_BIT(k); idx 257: KBitsZeroOnPadding.
    ck(fe(mu * (1 - mu)) == 0, "ecsm idx0")
    n_constraints += 1
    for i in range(256):
        ck(fe(kbits[i] * (1 - kbits[i])) == 0, f"ecsm kbit {i}")
        n_constraints += 1
    ck(fe(sum(kbits) * (1 - mu)) == 0, "ecsm idx257")
    n_constraints += 1

    # X2 convolution: 64 ConvCarry + ColIsZero(c0[63]).
    v = {"xg": w["x_g"], "x2": w["x2"], "q0": w["q0"], "yg": w["y_g"], "q1": w["q1"]}
    for i in range(64):
        ck(fe(conv_carry(w["c0"], s_ecsm_x2(v, i), i)) == 0, f"ecsm x2 carry {i}")
        n_constraints += 1
    ck(w["c0"][63] == 0, "ecsm c0[63]")
    n_constraints += 1

    # Yg convolution: 64 ConvCarry + ColIsZero(c1[63]).
    for i in range(64):
        ck(fe(conv_carry(w["c1"], s_ecsm_yg(v, i, mu=mu), i)) == 0, f"ecsm yg carry {i}")
        n_constraints += 1
    ck(w["c1"][63] == 0, "ecsm c1[63]")
    n_constraints += 1

    # idx 388: IS_BIT(q1[32]).
    ck(fe(w["q1"][32] * (1 - w["q1"][32])) == 0, "ecsm q1[32] bit")
    n_constraints += 1

    # Overflow chains (ecsm.rs:786-820, 883-897): xG<p, k<N, xR<p.
    for kind, const_bytes, hl_bytes, sum_src in [
        ("XgLtP", P_BYTES, w["x_g_sub_p"], ("bytes", w["x_g"])),
        ("KLtN", N_BYTES, w["k_sub_n"], ("bits", kbits)),
        ("XrLtP", P_BYTES, w["x_r_sub_p"], ("bytes", w["x_r"])),
    ]:
        prev = 0
        cbits = []
        for i in range(8):
            addend0 = sum(const_bytes[4 * i + b] << (8 * b) for b in range(4))
            hl0 = hl_bytes[4 * i] + 256 * hl_bytes[4 * i + 1]
            hl1 = hl_bytes[4 * i + 2] + 256 * hl_bytes[4 * i + 3]
            addend1 = hl0 + (1 << 16) * hl1
            if sum_src[0] == "bits":
                s = sum(sum_src[1][32 * i + b] << b for b in range(32))
            else:
                s = sum(sum_src[1][4 * i + b] << (8 * b) for b in range(4))
            ci = fe((addend0 + addend1 + prev - s) * INV32)
            cbits.append(ci)
            prev = ci
        for i in range(7):
            ck(fe(mu * cbits[i] * (1 - cbits[i])) == 0, f"ecsm {kind} carrybit {i}")
            n_constraints += 1
        ck(fe(mu * (1 - cbits[7])) == 0, f"ecsm {kind} overflow-required")
        n_constraints += 1

    ck(n_constraints == 413, f"ECSM constraint count {n_constraints} != 413")

    # Contract memberships (bus sends, ecsm.rs:446-506): carry windows.
    for i in range(63):
        ck(0 <= w["c0"][i] + OFF["ecsm_x2"] < 65536, f"ecsm c0[{i}] window")
        ck(0 <= w["c1"][i] + OFF["ecsm_yg"] < 65536, f"ecsm c1[{i}] window")

    # Value-level facts the chip is supposed to enforce.
    xg, yg, k = compose(w["x_g"]), compose(w["y_g"]), compose(w["k"])
    xr = compose(w["x_r"])
    ck(xg < P and xr < P and 0 < k < N, "ecsm domain")
    ck((yg * yg - xg**3 - B) % P == 0, "ecsm on-curve")
    ck(w["len_k"] == k.bit_length() - 1, "ecsm len_k = MSB")
    ck(kbits[w["len_k"]] == 1, "ecsm len_k bit set")

    # Overflow-chain witnesses recompose: p + sub = value + 2^256.
    ck(P + compose(w["x_g_sub_p"]) == xg + 2**256, "xg_sub_p recompose")
    ck(N + compose(w["k_sub_n"]) == k + 2**256, "k_sub_n recompose")
    ck(P + compose(w["x_r_sub_p"]) == xr + 2**256, "xr_sub_p recompose")
    return kbits


def eval_ecdas_step(st, idx):
    """All 200 ECDAS constraints for one step row (µ=1)."""
    n_constraints = 0
    mu, op, next_op = 1, st["op"], st["next_op"]
    for name, x in [("mu", mu), ("op", op), ("next_op", next_op)]:
        ck(fe(x * (1 - x)) == 0, f"step{idx} bit {name}")
        n_constraints += 1
    ck(fe(op * next_op) == 0, f"step{idx} OP*NEXT_OP")
    n_constraints += 1
    ck(fe(next_op * (1 - mu)) == 0, f"step{idx} NEXT_OP*(1-MU)")
    n_constraints += 1

    v = {"lam": st["lambda"], "xa": st["x_a"], "ya": st["y_a"], "xg": st["x_g"],
         "yg": st["y_g"], "xr": st["x_r"], "yr": st["y_r"],
         "q0": st["q0"], "q1": st["q1"], "q2": st["q2"]}
    for rel, cname, sfn, off in [
        ("lambda", "c0", lambda i: s_ecdas_lambda(v, i, op, mu), OFF["ecdas_lambda"]),
        ("xr", "c1", lambda i: s_ecdas_xr(v, i, op, mu), OFF["ecdas_xr"]),
        ("yr", "c2", lambda i: s_ecdas_yr(v, i, op, mu), OFF["ecdas_yr"]),
    ]:
        c = st[cname]
        for i in range(64):
            ck(fe(conv_carry(c, sfn(i), i)) == 0, f"step{idx} {rel} carry {i}")
            n_constraints += 1
        ck(c[63] == 0, f"step{idx} {rel} c63")
        n_constraints += 1
        for i in range(63):
            ck(0 <= c[i] + off < 65536, f"step{idx} {rel} c[{i}] window")

    ck(n_constraints == 200, f"ECDAS constraint count {n_constraints} != 200")


def eval_chain(w, kbits):
    """Bus-side chaining semantics on honest data (Ecdas telescoping + Bit counting)."""
    steps = w["steps"]
    if not steps:
        ck(w["len_k"] == 0, "k=1 echo: len_k")
        ck(w["x_r"] == w["x_g"] and w["y_r"] == w["y_g"], "k=1 echo: xR=xG")
        return
    ck(steps[0]["x_a"] == w["x_g"] and steps[0]["y_a"] == w["y_g"], "seed acc = G")
    ck(steps[0]["round"] == w["len_k"] - 1 and steps[0]["op"] == 0, "seed round/op")
    for t in range(len(steps) - 1):
        a, b = steps[t], steps[t + 1]
        ck(a["x_r"] == b["x_a"] and a["y_r"] == b["y_a"], f"chain acc {t}")
        ck(b["round"] == a["round"] - 1 + a["next_op"], f"chain round {t}")
        ck(b["op"] == a["next_op"], f"chain op {t}")
        ck(a["x_g"] == w["x_g"] and a["y_g"] == w["y_g"], f"chain gen {t}")
    last = steps[-1]
    # Drain tuple: round' = round − 1 + next_op must be −1 with op' = next_op = 0.
    ck(last["round"] == 0 and last["next_op"] == 0, "drain round/next_op")
    ck(last["x_r"] == w["x_r"] and last["y_r"] == w["y_r"], "drain result")
    # Bit counting: sends (len_k + rows with next_op=1 at their round) == set bits.
    sends = {w["len_k"]: 1}
    for st in steps:
        if st["next_op"] == 1:
            sends[st["round"]] = sends.get(st["round"], 0) + 1
    for i in range(256):
        ck(sends.get(i, 0) == kbits[i], f"bit balance @{i}")


def main():
    ks = [1, 2, 3, 5, 6, 0b110101, 2**128 + 12345, 2**255, 2**255 - 1, N - 1, N - 2,
          0x5A5A5A5A5A5A5A5A5A5A5A5A5A5A5A5A5A5A5A5A5A5A5A5A5A5A5A5A5A5A5A5A]
    xg_hex = format(GEN_X, "x")
    cmds = "".join(f"witness {xg_hex} {format(k, 'x')}\n" for k in ks)
    out = subprocess.run([str(HARNESS)], input=cmds, capture_output=True, text=True)
    lines = [l for l in out.stdout.splitlines() if l.strip()]
    assert len(lines) == len(ks), (len(lines), out.stderr[:500])

    total_steps = 0
    for k, line in zip(ks, lines):
        w = load_witness_json(line)
        kbits = eval_ecsm(w)
        for idx, st in enumerate(w["steps"]):
            eval_ecdas_step(st, idx)
        eval_chain(w, kbits)
        total_steps += len(w["steps"])
        print(f"k=0x{k:x}: ECSM 413 ok, {len(w['steps'])} ECDAS steps ok"
              if not failures else f"k=0x{k:x}: FAILURES (see below)")
        if failures:
            break

    print(f"\ntotal checks: {checks}, ECDAS rows: {total_steps}")
    if failures:
        print("FAILED:")
        for f in failures[:20]:
            print("  -", f)
        sys.exit(1)
    print("POSITIVE CONTROL PASS: real Rust witnesses satisfy the transcribed model.")


if __name__ == "__main__":
    main()
