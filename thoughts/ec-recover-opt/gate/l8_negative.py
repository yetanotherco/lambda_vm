"""L8: negative + sensitivity controls.

Two kinds of entry, per the gate methodology:
  * GENUINE FORGERIES (non-vacuity): dropping a soundness-load-bearing check
    lets a WRONG value pass. Demonstrated CONSTRUCTIVELY (fixed numerals),
    never by full-system search. If z3 appears it only CHECKS a concrete
    assignment. Establishes the gate is non-vacuous — it catches real bugs.
  * REDUNDANCY PROBES: dropping a check does NOT admit a wrong value because
    other checks already pin it (keccak IS_BIT-style finding). Reported honestly
    as UNSAT/REDUNDANT — not a failure, a finding.

The N1/N3 carry-check probes use a LINEAR carry gadget (no convolutions):
S_i are free bounded integers (over-approximation of the real per-limb
convolution values, L2a bound 2^22 for the yR relation), the field carry
recurrence is 256·c_i = c_{i−1} + S_i + p_g·m_i (m_i = wrap count), present
checks constrain the c_i, and the attack goal is a decoded value V = Σ256^i·S_i
with V ≢ 0 (mod p_secp) — i.e. a point wrong mod p. This is exactly the
Goldilocks-wrap threat model. Because S is free, UNSAT here is a STRONG
redundancy result (holds even for the over-approx); SAT means the wrap MECHANISM
survives the remaining checks (the dropped check is load-bearing), with the
realizability-by-actual-bytes caveat noted.
"""

import subprocess
import sys
import time
from pathlib import Path

import z3

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "oracle"))
import ec_ref
from gate_common import (
    N, P, PG, R3P, GEN_X, OFF, P_BYTES, compose, load_witness_json, ref_step,
)

HARNESS = Path(__file__).parent.parent / "oracle/repo-harness/target/release/ecsm-oracle-harness"
S_BOUND = 2**22          # L2a per-limb |S| bound for the yR relation (measured 4.07e6)
OFF_YR = OFF["ecdas_yr"]
results = []


def report(name, verdict, detail=""):
    results.append((name, verdict, detail))
    print(f"[{verdict:16}] {name}  {detail}", flush=True)


def get_witness(k):
    cmd = f"witness {format(GEN_X, 'x')} {format(k, 'x')}\n"
    out = subprocess.run([str(HARNESS)], input=cmd, capture_output=True, text=True)
    return load_witness_json(out.stdout.strip())


# ── Linear carry gadget for N1 / N3 ──────────────────────────────────────────

def carry_gadget(dropped):
    """dropped ∈ {'c40_window', 'c63_zero'}. Present checks: all c_i windowed for
    i in 0..62 and c_63 == 0, MINUS the dropped one. Attack goal: decoded value
    V = Σ256^i S_i with V ≢ 0 mod p_secp."""
    s = z3.Solver()
    s.set("timeout", 90000)
    S = [z3.Int(f"S{i}") for i in range(64)]
    c = [z3.Int(f"c{i}") for i in range(64)]
    m = [z3.Int(f"m{i}") for i in range(64)]
    for i in range(64):
        s.add(S[i] >= -S_BOUND, S[i] <= S_BOUND)
        s.add(m[i] >= -4, m[i] <= 4)                 # bounded wrap count
        prev = c[i - 1] if i > 0 else 0
        s.add(256 * c[i] == prev + S[i] + PG * m[i]) # field recurrence, lifted
    for i in range(63):                              # window checks c_0..c_62
        if dropped == "c40_window" and i == 40:
            continue
        s.add(c[i] >= -OFF_YR, c[i] < 65536 - OFF_YR)
    if dropped != "c63_zero":                        # closing constraint
        s.add(c[63] == 0)
    V = z3.Sum([256**i * S[i] for i in range(64)])
    t, r = z3.Ints("t r")
    s.add(V - P * t == r, r >= 1, r <= P - 1)         # V mod p ≠ 0
    return s.check()


def n1_c40_window():
    t0 = time.time()
    res = carry_gadget("c40_window")
    if res == z3.unsat:
        report("N1 drop IsHalfword(c[40])", "UNSAT/REDUNDANT",
               f"mid carry window individually redundant given c₆₃=0 + neighbours "
               f"(strong: S free); {time.time()-t0:.0f}s")
    elif res == z3.sat:
        report("N1 drop IsHalfword(c[40])", "SAT(FORGES)",
               f"Goldilocks-wrap at limb 40 survives remaining checks; {time.time()-t0:.0f}s")
    else:
        report("N1 drop IsHalfword(c[40])", "OPEN(TIMEOUT)", f"{time.time()-t0:.0f}s")


def n3_c63_zero():
    t0 = time.time()
    res = carry_gadget("c63_zero")
    if res == z3.sat:
        report("N3 drop ColIsZero(c[63])", "SAT(FORGES)",
               f"top overflow unconstrained ⇒ decoded value ≢ 0 mod p; c₆₃=0 LOAD-BEARING; "
               f"{time.time()-t0:.0f}s (realizability caveat: gadget over-approx)")
    elif res == z3.unsat:
        report("N3 drop ColIsZero(c[63])", "UNSAT/REDUNDANT", f"{time.time()-t0:.0f}s")
    else:
        report("N3 drop ColIsZero(c[63])", "OPEN(TIMEOUT)", f"{time.time()-t0:.0f}s")


def gadget_baseline():
    """Sanity: with ALL checks present the gadget is UNSAT (can't decode wrong)."""
    t0 = time.time()
    s = z3.Solver()
    s.set("timeout", 90000)
    S = [z3.Int(f"S{i}") for i in range(64)]
    c = [z3.Int(f"c{i}") for i in range(64)]
    m = [z3.Int(f"m{i}") for i in range(64)]
    for i in range(64):
        s.add(S[i] >= -S_BOUND, S[i] <= S_BOUND, m[i] >= -4, m[i] <= 4)
        prev = c[i - 1] if i > 0 else 0
        s.add(256 * c[i] == prev + S[i] + PG * m[i])
    for i in range(63):
        s.add(c[i] >= -OFF_YR, c[i] < 65536 - OFF_YR)
    s.add(c[63] == 0)
    V = z3.Sum([256**i * S[i] for i in range(64)])
    t, r = z3.Ints("t r")
    s.add(V - P * t == r, r >= 1, r <= P - 1)
    res = s.check()
    report("N1/N3 baseline (all checks ⇒ no wrong decode)",
           "UNSAT(OK)" if res == z3.unsat else f"UNEXPECTED-{res}",
           f"full window set + c₆₃=0 forces V ≡ 0 mod p; {time.time()-t0:.0f}s")


# ── N6: XR_SUB_P drop → non-canonical drain (genuine forgery) ────────────────

def n6_xr_sub_p():
    def block(with_check):
        s = z3.Solver()
        s.set("timeout", 60000)
        v = z3.Int("v")
        words = [z3.Int(f"w{i}") for i in range(8)]
        s.add(v >= 0, v < 2**32)                    # a small drain x-value
        for w in words:
            s.add(w >= 0, w < 2**32)
        X = z3.Sum([2**(32 * i) * words[i] for i in range(8)])
        mq = z3.Int("mq")
        s.add(X - v == mq * P)                      # bus binds xR ≡ chain x (mod p)
        s.add(X >= 0, X < 2**256)
        if with_check:
            hw = [z3.Int(f"h{i}") for i in range(8)]
            carr = [z3.Int(f"cc{i}") for i in range(8)]
            for i in range(8):
                s.add(hw[i] >= 0, hw[i] < 2**32, z3.Or(carr[i] == 0, carr[i] == 1))
                pw = sum(P_BYTES[4 * i + b] << (8 * b) for b in range(4))
                prev = carr[i - 1] if i > 0 else 0
                s.add(pw + hw[i] + prev - words[i] == 2**32 * carr[i])
            s.add(carr[7] == 1)
        s.add(X != v)                               # non-canonical acceptance
        return s.check()
    r_with, r_without = block(True), block(False)
    ok = r_with == z3.unsat and r_without == z3.sat
    report("N6 drop XR_SUB_P", "SAT(FORGES)" if ok else f"UNEXPECTED w={r_with} wo={r_without}",
           "present⇒UNSAT (canonical forced); dropped⇒SAT (xR=v+p accepted, v<2^32+977). LOAD-BEARING")


# ── NSW: transcription tamper xA↔xG in yR relation (genuine forgery) ─────────

def nsw():
    st = get_witness(5)["steps"][1]      # an ADD step where xA = 2G ≠ G
    lam, xa, xg, xr = (compose(st["lambda"]), compose(st["x_a"]),
                       compose(st["x_g"]), compose(st["x_r"]))
    ya, yr = compose(st["y_a"]), compose(st["y_r"])
    x, q, gq, gr = z3.Ints("x q gq gr")
    s = z3.Solver()
    s.set("timeout", 60000)
    s.add(x >= 0, x < 2**256)
    s.add(lam * (xg - xr) - ya - x + R3P * P - q * P == 0, q >= 0, q < 2**264)  # swapped xA→xG
    s.add(x - yr == gq * P + gr, gr > 0, gr < P)   # forged yR ≢ honest (mod p)
    r = s.check()
    if r == z3.sat:
        m = s.model()
        report("NSW transcription tamper (xA↔xG in yR)", "SAT(FORGES)",
               f"tampered relation admits yR=0x{m[x].as_long():x} ≠ honest 0x{yr:x}")
    else:
        report("NSW transcription tamper", f"UNEXPECTED-{r}")


# ── N-CONST: wrong prime rejects the honest witness (genuine, constant-binding) ─

def n_const():
    st = get_witness(5)["steps"][1]
    lam, xa, xr = compose(st["lambda"]), compose(st["x_a"]), compose(st["x_r"])
    ya, yr, q2 = compose(st["y_a"]), compose(st["y_r"]), compose(st["q2"])
    good = (lam * (xa - xr) - ya - yr + R3P * P - q2 * P) % PG
    Pw = P + 2
    bad = (lam * (xa - xr) - ya - yr + (3 * Pw) * Pw - q2 * Pw) % PG
    ok = good == 0 and bad != 0
    report("N-CONST wrong-prime (p→p+2)", "SAT(CATCHES)" if ok else f"UNEXPECTED good={good} bad={bad}",
           "honest witness: value≡0 under p ✓, value≢0 under p+2 ✓ ⇒ constraints bind the constant")


# ── N4: OP·NEXT_OP redundancy (schedule model) ───────────────────────────────

def n4_schedule(k, drop_bit_balance):
    T = 8
    s = z3.Solver()
    s.set("timeout", 120000)
    act = [z3.Bool(f"a{t}") for t in range(T)]
    rnd = [z3.Int(f"r{t}") for t in range(T)]
    op = [z3.Int(f"o{t}") for t in range(T)]
    nop = [z3.Int(f"n{t}") for t in range(T)]
    m = [z3.Int(f"mm{t}") for t in range(T + 1)]
    len_k = z3.Int("len_k")
    s.add(len_k >= 0, len_k <= 7, m[0] == 1, act[0])
    s.add(z3.Implies(act[0], z3.And(op[0] == 0, rnd[0] == len_k - 1)))
    for t in range(T):
        s.add(z3.Or(op[t] == 0, op[t] == 1), z3.Or(nop[t] == 0, nop[t] == 1))
        s.add(rnd[t] >= 0, rnd[t] <= 255)          # ROUND byte contract
        # OP·NEXT_OP == 0 intentionally OMITTED (the tamper).
        out = z3.If(op[t] == 1, m[t] + 1, 2 * m[t])
        s.add(m[t + 1] == z3.If(act[t], out, m[t]))
        if t + 1 < T:
            s.add(z3.Implies(z3.And(act[t], act[t + 1]),
                             z3.And(rnd[t + 1] == rnd[t] - 1 + nop[t], op[t + 1] == nop[t])))
            s.add(z3.Implies(z3.Not(act[t]), z3.Not(act[t + 1])))
            s.add(z3.Implies(z3.And(act[t], z3.Not(act[t + 1])),
                             z3.And(rnd[t] == 0, nop[t] == 0)))
    s.add(z3.Implies(act[T - 1], z3.And(rnd[T - 1] == 0, nop[T - 1] == 0)))
    if not drop_bit_balance:
        for i in range(8):
            sends = z3.Sum([z3.If(z3.And(act[t], nop[t] == 1, rnd[t] == i), 1, 0)
                            for t in range(T)]) + z3.If(len_k == i, 1, 0)
            s.add(sends == ((k >> i) & 1))
    s.add(m[T] != k)
    return s.check()


def n4():
    for k in (6, 11):
        t0 = time.time()
        r1, r2 = n4_schedule(k, False), n4_schedule(k, True)
        if r1 == z3.unsat and r2 == z3.sat:
            report(f"N4 drop OP·NEXT_OP [k={k}]", "UNSAT/REDUNDANT",
                   f"Bit-balance alone blocks AA schedules; +drop balance ⇒ SAT forgery; "
                   f"{time.time()-t0:.0f}s")
        else:
            report(f"N4 drop OP·NEXT_OP [k={k}]", f"UNEXPECTED r1={r1} r2={r2}", f"{time.time()-t0:.0f}s")


# ── N5: IS_BIT(q1[32]) redundancy (ECSM curve relation) ──────────────────────

def n5_q1_bit():
    t0 = time.time()
    xg = GEN_X
    yg, q0m, q1, x2, gq, gr = z3.Ints("yg q0m q1 x2 gq gr")
    s = z3.Solver()
    s.set("timeout", 90000)
    s.add(yg >= 0, yg < 2**256, x2 >= 0, x2 < 2**256)
    s.add(q1 >= 0, q1 < 2**264)                    # FAT quotient (IS_BIT dropped)
    s.add(xg * xg - x2 - q0m * P == 0, q0m >= 0, q0m < 2**256)
    s.add(yg * yg + P * P - x2 * xg - 7 - q1 * P == 0)
    s.add(yg * yg - (pow(xg, 3, P) + 7) == gq * P + gr, gr > 0, gr < P)  # deny on-curve
    r = s.check()
    report("N5 drop IS_BIT(q1[32])",
           "UNSAT/REDUNDANT" if r == z3.unsat else f"UNEXPECTED-{r}",
           f"curve relation still pins yG²≡xG³+7 with q1<2^264; {time.time()-t0:.0f}s")


# ── N7: KBitsZeroOnPadding — rejection-only argument (recorded verdict) ───────

def n7():
    report("N7 drop KBitsZeroOnPadding", "UNSAT/REDUNDANT",
           "rejection-only: padding k_bits (mult∈{0,1}) can only add unmatched Bit-receives "
           "(all padding senders µ/next_op-dead, ecsm.rs:529-540 / ecdas.rs:217-221) ⇒ imbalance; "
           "soundness-redundant, keep as completeness hygiene (argument, not search)")


if __name__ == "__main__":
    gadget_baseline()
    n1_c40_window()
    n3_c63_zero()
    n6_xr_sub_p()
    nsw()
    n_const()
    n4()
    n5_q1_bit()
    n7()
    print("\nSummary:")
    for n_, v, d in results:
        print(f"  {v:16} {n_}")
    bad = [v for _, v, _ in results if v.startswith("UNEXPECTED") or v.startswith("OPEN")]
    forges = [n_ for n_, v, _ in results if "FORGES" in v or "CATCHES" in v]
    redun = [n_ for n_, v, _ in results if "REDUNDANT" in v]
    print(f"\nGenuine forgeries/catches (non-vacuity): {len(forges)} -> {forges}")
    print(f"Redundancy findings: {len(redun)}")
    if bad:
        print("UNEXPECTED/OPEN:", bad)
        sys.exit(1)
