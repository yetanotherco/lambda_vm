#!/usr/bin/env python3
# Read-only: job 292's twelve identical arms at 7c8272701 — the base, Σ argue over the 15 epochs, Σ producer overlap
# with the argues, Σ prep wait (prover waiting for the producer), Σ hand-off wait (producer waiting for the prover).
import re, statistics
F = r"([0-9]+(?:\.[0-9]+)?)"
SPAN = re.compile(r"^BASE EPOCH ([0-9]+): (execute|collect|build) " + F + r"s t=\[" + F + "," + F + r"\]")
PREPP = re.compile(r"^BASE PREP ([0-9]+|global): prep@producer " + F + r"s t=\[" + F + "," + F + r"\]")
PROVE = re.compile(r"^BASE EPOCH ([0-9]+): prove " + F + r"s t=\[" + F + "," + F + r"\]")
SPLIT = re.compile(r"^WHIR PROVE SPLIT #([0-9]+):.*challenge " + F + r" · argue " + F)
PREPW = re.compile(r"^BASE EPOCH ([0-9]+): prep " + F + r"s t=")
HAND = re.compile(r"^BASE EPOCH ([0-9]+): handoff " + F + r"s t=")
rows = []
for n in range(1302, 1314):
    t = open(f"/root/prof/wt{n}-tree.log", errors="replace").read().splitlines()
    spans, prove, split, prepw, hand, prepp = [], {}, {}, {}, {}, {}
    for l in t:
        m = SPAN.match(l)
        if m: spans.append((float(m.group(4)), float(m.group(5)))); continue
        m = PREPP.match(l)
        if m:
            spans.append((float(m.group(3)), float(m.group(4))))
            if m.group(1) != "global": prepp[int(m.group(1))] = float(m.group(2))
            continue
        m = PROVE.match(l)
        if m: prove[int(m.group(1))] = (float(m.group(3)), float(m.group(4))); continue
        m = SPLIT.match(l)
        if m: split[int(m.group(1))] = (float(m.group(2)), float(m.group(3))); continue
        m = PREPW.match(l)
        if m: prepw[int(m.group(1))] = float(m.group(2)); continue
        m = HAND.match(l)
        if m: hand[int(m.group(1))] = float(m.group(2))
    base = float(re.search(r"base \(WHIR\): [0-9]+ epochs in " + F, "\n".join(t)).group(1))
    arg = sum(split[k][1] for k in range(15) if k in split)
    ov = 0.0
    for k in range(15):
        if k in prove and k in split:
            a0 = prove[k][0] + split[k][0]; a1 = a0 + split[k][1]
            ov += sum(max(0.0, min(a1, s1) - max(a0, s0)) for s0, s1 in spans)
    pw = sum(v for k, v in prepw.items() if k >= 1)
    hw = sum(hand[k] - prepp.get(k, 0.0) for k in hand)
    rows.append((n, base, arg, ov, pw, hw))
    print(f"wt{n} base {base:.1f} · Σargue {arg:.2f} · Σoverlap {ov:.2f} · Σprep wait {pw:.2f} · Σhand-off wait {hw:.2f}")
for i, name in ((2, "Σargue"), (3, "Σoverlap"), (5, "Σhand-off wait")):
    v = [r[i] for r in rows]
    print(f"{name}: mean {statistics.mean(v):.3f} · sd {statistics.stdev(v):.3f} · range {min(v):.2f}–{max(v):.2f}")
