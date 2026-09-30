#!/usr/bin/env python3
# Read-only: per WHIR tree log on FAST, the level-1 wide nodes' timings, the arm's wall clock window, its place in
# its harness run, and the gap since the previous arm ended.
import glob, os, re
F = r"([0-9]+(?:\.[0-9]+)?)"
rows = []
for path in sorted(glob.glob("/root/prof/wt1[0-3][0-9][0-9]-tree.log")):
    tag = os.path.basename(path).split("-")[0]
    t = open(path, errors="replace").read()
    m = re.search(r"BASE HEAD \(WHIR\): start t=" + F, t)
    start = float(m.group(1)) if m else None
    m = re.search(r"WHOLE RUN: host peak " + F + r" GiB at t=" + F + r", " + F + r"s total", t)
    whole = float(m.group(3)) if m else None
    m = re.search(r"LEVEL 1 \(wide\) START t=" + F, t)
    l1s = float(m.group(1)) if m else None
    m = re.search(r"level 1 \(wide\): [0-9]+ wide nodes over [0-9]+ epochs in " + F + "s", t)
    l1 = float(m.group(1)) if m else None
    m = re.search(r"base \(WHIR\): [0-9]+ epochs in " + F + "s", t)
    base = float(m.group(1)) if m else None
    nodes = {}
    for m in re.finditer(r"(L1N[0-9]) \(arity [0-9]+\) TIMING: wide, epochs [0-9]+\.\.=[0-9]+ · harvest-epochs " + F +
                         r"s · emit\+arenas " + F + r"s · artifacts " + F + r"s · prove " + F + r"s · harvest " + F +
                         r"s \(verify " + F + r"\) · wall " + F + r"s( ⓘ lead-in)?", t):
        nodes[m.group(1)] = tuple(float(m.group(i)) for i in range(2, 9)) + (bool(m.group(9)),)
    m = re.search(r"ZF FORMAT[^\n]*", t)
    rows.append((tag, start, whole, base, l1s, l1, nodes))
# harness runs: order and knobs from each manifest
order = {}
for man in glob.glob("/root/zf/wt1*-wt1*/manifest.tsv"):
    run = os.path.basename(os.path.dirname(man))
    arms = [l.rstrip("\n").split("\t") for l in open(man) if l.strip() and not l.startswith("#")]
    for i, a in enumerate(arms):
        order[a[0]] = (run, i + 1, len(arms), a[1], a[2])
prev_end = None
print("tag\trun\tpos\tname\tknobs\tstart_utc\tgap_s\twhole\tbase\tl1\tL1N2_harvest\tL1N2_emit\tL1N2_artifacts\tL1N2_prove\tL1N2_wall\tL1N0_prove\tL1N1_prove\tL1N1_wall")
import time
for tag, start, whole, base, l1s, l1, nodes in rows:
    run, pos, n, name, knobs = order.get(tag, ("-", 0, 0, "-", "-"))
    gap = None if prev_end is None or start is None else start - prev_end
    end = None if start is None or whole is None else start + whole
    n2 = nodes.get("L1N2"); n0 = nodes.get("L1N0"); n1 = nodes.get("L1N1")
    f = lambda x, i: "NA" if x is None else f"{x[i]:.2f}"
    print("\t".join([tag, run, f"{pos}/{n}", name, knobs,
                     time.strftime("%m-%dT%H:%M:%S", time.gmtime(start)) if start else "NA",
                     "NA" if gap is None else f"{gap:.0f}", str(whole), str(base), str(l1),
                     f(n2, 0), f(n2, 1), f(n2, 2), f(n2, 3), f(n2, 6), f(n0, 3), f(n1, 3), f(n1, 6)]))
    prev_end = end
