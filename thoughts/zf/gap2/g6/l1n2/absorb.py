#!/usr/bin/env python3
# Read-only. Does the argue slow down when it overlaps the producer's work? Per arm and epoch: the argue window
# (the head of `BASE EPOCH k: prove`, `WHIR PROVE SPLIT #k: argue`) and its overlap with the producer's
# execute/collect/build/prep@producer spans (any epoch). Then argue on overlap, within epoch (epoch means removed).
import collections, glob, os, re, statistics
F = r"([0-9]+(?:\.[0-9]+)?)"
SPAN = re.compile(r"^BASE EPOCH ([0-9]+): (execute|collect|build) " + F + r"s t=\[" + F + "," + F + r"\]")
PREP = re.compile(r"^BASE PREP ([0-9]+): prep@producer " + F + r"s t=\[" + F + "," + F + r"\]")
PROVE = re.compile(r"^BASE EPOCH ([0-9]+): prove " + F + r"s t=\[" + F + "," + F + r"\]")
COMMIT = re.compile(r"^BASE EPOCH ([0-9]+): commit " + F + r"s t=\[" + F + "," + F + r"\]")
SPLIT = re.compile(r"^WHIR PROVE SPLIT #([0-9]+):.*challenge " + F + r" · argue " + F)
tags = [f"wt{n}" for n in list(range(1200, 1204)) + list(range(1210, 1214)) + list(range(1290, 1302)) + list(range(1080, 1088))]
rows = []
for tag in tags:
    path = f"/root/prof/{tag}-tree.log"
    if not os.path.exists(path): continue
    spans, prove, split, commit = [], {}, {}, {}
    for line in open(path, errors="replace"):
        m = SPAN.match(line)
        if m: spans.append((float(m.group(4)), float(m.group(5)))); continue
        m = PREP.match(line)
        if m: spans.append((float(m.group(3)), float(m.group(4)))); continue
        m = PROVE.match(line)
        if m: prove[int(m.group(1))] = (float(m.group(3)), float(m.group(4))); continue
        m = COMMIT.match(line)
        if m: commit[int(m.group(1))] = float(m.group(2)); continue
        m = SPLIT.match(line)
        if m: split[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    for k in sorted(prove):
        if k not in split: continue
        a0 = prove[k][0] + split[k][0]; a1 = a0 + split[k][1]
        ov = sum(max(0.0, min(a1, s1) - max(a0, s0)) for s0, s1 in spans)
        rows.append((tag, k, split[k][1], ov, commit.get(k)))
# within-epoch regression
by_k = collections.defaultdict(list)
for r in rows: by_k[r[1]].append(r)
xs, ys = [], []
for k, rs in by_k.items():
    if len(rs) < 3: continue
    ma = statistics.mean(r[2] for r in rs); mo = statistics.mean(r[3] for r in rs)
    for r in rs: xs.append(r[3] - mo); ys.append(r[2] - ma)
sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
beta = sxy / sxx
res = [y - beta * x for x, y in zip(xs, ys)]
se = (sum(e * e for e in res) / (len(xs) - 1 - len(by_k)) / sxx) ** 0.5
print(f"epochs×arms {len(xs)} · within-epoch slope: argue += {beta:.3f} s per s of producer overlap (SE {se:.3f}, t {beta/se:.1f})")
print(f"overlap per argue (s): mean {statistics.mean(r[3] for r in rows):.3f} · sd within epoch {statistics.pstdev(xs):.3f}")
# per arm totals
arms = collections.OrderedDict()
for r in rows:
    a = arms.setdefault(r[0], [0.0, 0.0, 0.0])
    a[0] += r[2]; a[1] += r[3]; a[2] += r[4] or 0.0
for tag, (arg, ov, com) in arms.items():
    print(f"{tag} Σ argue {arg:6.2f} · Σ overlap with producer {ov:5.2f} · Σ commit {com:5.2f}")
