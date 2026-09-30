#!/usr/bin/env python3
"""QUIET-PRODUCER readout: the default (A) against LAMBDA_VM_QUIET_PRODUCER=1 (B: the producer's stages and the
prover's argues never overlap), A B B A, at fix2/1010-quiet-producer on 7c8272701.

usage: python3 qp_readout.py <manifest.tsv>             (on the box)
       python3 qp_readout.py --selftest <A-log>          (laptop: a real 7c8272701 log, and a B copy made from it)

Two uses, both pre-registered in G6-LEDGER.md §11 before the run:
 (a) beta = (Σargue_A − Σargue_B) / (Σoverlap_A − Σoverlap_B): the argue's seconds per second of producer overlap,
     which moves stage 1b's band;
 (b) the quiet producer as a lever: the base's Δ, with the rows that say whether the producer started binding
     (the prover's Σ prep wait, the producer's Σ hand-off wait).
Level 1 is reported apart and flagged bimodal (G6-LEDGER.md §9.3). Standard library only.
"""
import re
import statistics
import sys

F = r"([0-9]+(?:\.[0-9]+)?)"
SPAN = re.compile(r"^BASE EPOCH ([0-9]+): (execute|collect|build) " + F + r"s t=\[" + F + "," + F + r"\]")
PREPP = re.compile(r"^BASE PREP ([0-9]+|global): prep@producer " + F + r"s t=\[" + F + "," + F + r"\]")
PROVE = re.compile(r"^BASE EPOCH ([0-9]+): prove " + F + r"s t=\[" + F + "," + F + r"\]")
SPLIT = re.compile(r"^WHIR PROVE SPLIT #([0-9]+):.*challenge " + F + r" · argue " + F)
PREPW = re.compile(r"^BASE EPOCH ([0-9]+): prep " + F + r"s t=")
HAND = re.compile(r"^BASE EPOCH ([0-9]+): handoff " + F + r"s t=")
QUIET = re.compile(r"^QUIET PRODUCER: argues waited " + F + r"s over ([0-9]+) argue\(s\) · producer stages waited "
                   + F + r"s over ([0-9]+) stage\(s\)", re.M)
IDS = re.compile(r"IDENTITY: program_id")

# Pre-registered (G6-LEDGER.md §11). B − A unless named.
A_ARGUE = (11.45, 11.95)       # job 292's 12 arms at 7c8272701: 11.70 ± 0.08 (sd)
A_OVERLAP = (7.4, 8.3)         # job 292: 7.86 ± 0.10
B_OVERLAP_MAX = 0.3            # the exclusion leaves nothing but timer granularity
B_ARGUES_MIN = 16              # 15 epochs + the global stage
B_STAGES_MIN = 60              # 15 epochs × 4 stages, + the global prep
BETA_NEGLIGIBLE = 0.05
BETA_MATERIAL = 0.15
B_BASE = (3.0, 8.0)            # (b): the producer is expected to bind
B_PREP_WAIT_MIN = 2.0          # (b): the prover waits for the producer
B_HANDOFF_MAX = 2.0            # (b): the producer's slack mostly gone (A: 7.54 ± 0.18 s)
SLOW_L1 = 8.0                  # level 1's slow mode (§9.3)


def read(path):
    lines = open(path, errors="replace").read().splitlines()
    text = "\n".join(lines)
    spans, prove, split, prepw, hand, prepp = [], {}, {}, {}, {}, {}
    for l in lines:
        m = SPAN.match(l)
        if m:
            spans.append((float(m.group(4)), float(m.group(5))))
            continue
        m = PREPP.match(l)
        if m:
            spans.append((float(m.group(3)), float(m.group(4))))
            if m.group(1) != "global":
                prepp[int(m.group(1))] = float(m.group(2))
            continue
        m = PROVE.match(l)
        if m:
            prove[int(m.group(1))] = (float(m.group(3)), float(m.group(4)))
            continue
        m = SPLIT.match(l)
        if m:
            split[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
            continue
        m = PREPW.match(l)
        if m:
            prepw[int(m.group(1))] = float(m.group(2))
            continue
        m = HAND.match(l)
        if m:
            hand[int(m.group(1))] = float(m.group(2))
    g = lambda rx, i=1: (lambda m: float(m.group(i)) if m else None)(re.search(rx, text, re.M))
    r = {"base": g(r"base \(WHIR\): [0-9]+ epochs in " + F + "s"),
         "l1": g(r"level 1 \(wide\): [0-9]+ wide nodes over [0-9]+ epochs in " + F + "s"),
         "whole": g(r"WHOLE RUN: host peak " + F + r" GiB at t=" + F + r", " + F + r"s total", 3)}
    r["argue"] = sum(split[k][1] for k in range(15) if k in split)
    r["epochs"] = sum(1 for k in range(15) if k in split)
    ov = 0.0
    for k in range(15):
        if k in prove and k in split:
            a0 = prove[k][0] + split[k][0]
            a1 = a0 + split[k][1]
            ov += sum(max(0.0, min(a1, s1) - max(a0, s0)) for s0, s1 in spans)
    r["overlap"] = ov
    r["prep_wait"] = sum(v for k, v in prepw.items() if k >= 1)
    r["handoff_wait"] = sum(hand[k] - prepp.get(k, 0.0) for k in hand)
    m = QUIET.search(text)
    r["quiet"] = (float(m.group(1)), int(m.group(2)), float(m.group(3)), int(m.group(4))) if m else None
    r["ids"] = sorted(l.strip() for l in lines if IDS.search(l))
    r["proved"] = "THE BLOCK IS COMPRESSED UNDER WHIR" in text or "PROVED AND VERIFIED" in text
    return r


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def f(x, nd=2):
    return "NA" if x is None else f"{x:.{nd}f}"


def readout(arms):
    reads = [(tag, name, read(log)) for tag, name, log in arms]
    print("| tag | arm | whole | base | level 1 | Σargue | Σoverlap | Σprep wait | Σhand-off wait | quiet: argue wait / "
          "argues · stage wait / stages |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for tag, name, r in reads:
        q = r["quiet"]
        qs = "-" if q is None else f"{q[0]:.2f} / {q[1]} · {q[2]:.2f} / {q[3]}"
        l1 = f(r["l1"], 1) + (" (slow mode)" if r["l1"] is not None and r["l1"] >= SLOW_L1 else "")
        print(f"| {tag} | {name} | {f(r['whole'], 1)} | {f(r['base'], 1)} | {l1} | {f(r['argue'])} | {f(r['overlap'])} "
              f"| {f(r['prep_wait'])} | {f(r['handoff_wait'])} | {qs} |")
    A = [r for _, n, r in reads if n == "A"]
    B = [r for _, n, r in reads if n == "B"]
    rows = []

    def check(label, ok, detail):
        rows.append((label, "PASS" if ok else "FAIL", detail))

    check("arms", bool(A) and bool(B), f"A {len(A)} · B {len(B)}")
    if not (A and B):
        for row in rows:
            print(" · ".join(row))
        return "READOUT: INCOMPLETE — an arm setting is missing"
    check("each arm ran its own path", all(r["quiet"] is None for r in A) and all(
        r["quiet"] is not None and r["quiet"][1] >= B_ARGUES_MIN and r["quiet"][3] >= B_STAGES_MIN for r in B),
          f"A quiet lines {[r['quiet'] for r in A]} · B {[r['quiet'] for r in B]}")
    check("proved, 15 epochs each", all(r["proved"] and r["epochs"] == 15 for r in A + B), "")
    ids = {tuple(r["ids"]) for r in A + B}
    check("program ids identical across settings", len(ids) == 1 and len(next(iter(ids))) == 5, f"{len(ids)} set(s)")
    a_arg, b_arg = mean(r["argue"] for r in A), mean(r["argue"] for r in B)
    a_ov, b_ov = mean(r["overlap"] for r in A), mean(r["overlap"] for r in B)
    check("A's Σargue (control)", A_ARGUE[0] <= a_arg <= A_ARGUE[1], f"{f(a_arg)} s in {A_ARGUE}")
    check("A's Σoverlap (control)", A_OVERLAP[0] <= a_ov <= A_OVERLAP[1], f"{f(a_ov)} s in {A_OVERLAP}")
    check("B's argues ran with no producer overlap", max(r["overlap"] for r in B) <= B_OVERLAP_MAX,
          f"max {f(max(r['overlap'] for r in B))} s ≤ {B_OVERLAP_MAX}")
    beta = (a_arg - b_arg) / (a_ov - b_ov) if a_ov - b_ov > 0.5 else None
    for row in rows:
        print(" · ".join(row))
    if beta is None:
        use_a = "(a) β: NA (the overlap did not fall)"
    elif beta < BETA_NEGLIGIBLE:
        use_a = f"(a) β = {beta:+.3f} < {BETA_NEGLIGIBLE}: absorption negligible, 1b's band −1.0 s [−1.2, −0.5] stands"
    elif beta >= BETA_MATERIAL:
        use_a = (f"(a) β = {beta:+.3f} ≥ {BETA_MATERIAL}: material; 1b's centre moves to (1 − β) × 1.13 = "
                 f"{-(1 - beta) * 1.13:+.2f} s")
    else:
        use_a = (f"(a) β = {beta:+.3f} in [{BETA_NEGLIGIBLE}, {BETA_MATERIAL}): 1b's low end moves to (1 − β) × 1.06 = "
                 f"{-(1 - beta) * 1.06:+.2f} s")
    print(f"Σargue A {f(a_arg)} · B {f(b_arg)} · Δ {f(b_arg - a_arg)} s · Σoverlap A {f(a_ov)} · B {f(b_ov)}")
    print(use_a)
    a_b, b_b = mean(r["base"] for r in A), mean(r["base"] for r in B)
    a_pw, b_pw = mean(r["prep_wait"] for r in A), mean(r["prep_wait"] for r in B)
    a_hw, b_hw = mean(r["handoff_wait"] for r in A), mean(r["handoff_wait"] for r in B)
    d_b = b_b - a_b
    binds = b_pw >= B_PREP_WAIT_MIN or b_hw <= B_HANDOFF_MAX
    if d_b <= -0.3 and not binds:
        use_b = f"(b) LEVER: Δ base {d_b:+.2f} s and the producer never bound"
    elif d_b >= 0.3:
        use_b = f"(b) REGRESSION: Δ base {d_b:+.2f} s" + (" (band held)" if B_BASE[0] <= d_b <= B_BASE[1] else
                                                        " (outside the band)")
    else:
        use_b = f"(b) NO EFFECT: Δ base {d_b:+.2f} s"
    print(f"base A {f(a_b)} · B {f(b_b)} · Δ {f(d_b)} s · Σprep wait A {f(a_pw)} · B {f(b_pw)} · Σhand-off wait A "
          f"{f(a_hw)} · B {f(b_hw)} · the producer {'BINDS' if binds else 'does not bind'}")
    print(use_b)
    a_l1 = [r["l1"] for r in A]
    b_l1 = [r["l1"] for r in B]
    print(f"level 1 (bimodal; not a deciding row): A {a_l1} · B {b_l1}")
    fails = [label for label, v, _ in rows if v == "FAIL"]
    return (f"READOUT: {use_a} || {use_b} || "
            + ("checks PASS" if not fails else "FAIL: " + "; ".join(fails)))


def selftest(a_log):
    """A real 7c8272701 log reads as A. A B copy — its producer spans moved off the argues, its argues 0.3 s
    shorter in total, a QUIET line added, the prep waits raised — reads as B with the right β and a binding
    producer."""
    import os
    import tempfile
    lines = open(a_log, errors="replace").read().splitlines()
    ra = read(a_log)
    out = []
    for l in lines:
        m = SPAN.match(l)
        if m:
            l = re.sub(r"t=\[" + F + "," + F + r"\]", "t=[1.000,1.001]", l)
        m = PREPP.match(l)
        if m:
            l = re.sub(r"t=\[" + F + "," + F + r"\]", "t=[1.000,1.001]", l)
        m = SPLIT.match(l)
        if m:
            k = int(m.group(1))
            a = float(m.group(3))
            if k < 15:
                l = l.replace(f"· argue {m.group(3)}", f"· argue {a - 0.02:.2f}", 1)
        m = PREPW.match(l)
        if m and int(m.group(1)) >= 1:
            l = re.sub(r"prep " + F + "s", "prep 0.30s", l, count=1)
        out.append(l)
    out.append("QUIET PRODUCER: argues waited 2.100s over 16 argue(s) · producer stages waited 9.000s over 61 stage(s)"
               " (the epochs)")
    d = tempfile.mkdtemp()
    pb = os.path.join(d, "b.log")
    open(pb, "w").write("\n".join(out) + "\n")
    rb = read(pb)
    assert rb["quiet"] == (2.1, 16, 9.0, 61), rb["quiet"]
    assert rb["overlap"] < 0.01 and ra["overlap"] > 5, (ra["overlap"], rb["overlap"])
    assert abs((ra["argue"] - rb["argue"]) - 0.30) < 0.02, (ra["argue"], rb["argue"])
    line = readout([("a1", "A", a_log), ("b1", "B", pb), ("b2", "B", pb), ("a2", "A", a_log)])
    print(line)
    assert "β = +0.03" in line or "β = +0.04" in line, line
    print("SELFTEST OK")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--selftest":
        selftest(sys.argv[2])
        sys.exit(0)
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    arms = []
    for line in open(sys.argv[1]):
        if line.startswith("#") or not line.strip():
            continue
        tag, name, _knobs, log = line.rstrip("\n").split("\t")[:4]
        arms.append((tag, name, log))
    print(readout(arms))
