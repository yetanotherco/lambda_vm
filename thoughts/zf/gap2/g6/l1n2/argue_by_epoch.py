#!/usr/bin/env python3
# Read-only: job 291's argue seconds per epoch, A arms against B arms, with each epoch's spread.
import re, statistics
F = r"([0-9]+(?:\.[0-9]+)?)"
SPLIT = re.compile(r"^WHIR PROVE SPLIT #([0-9]+):.*· argue " + F + r".*open_groups " + F)
A = ["wt1294", "wt1297", "wt1298", "wt1301"]; B = ["wt1295", "wt1296", "wt1299", "wt1300"]
def read(tag):
    d = {}
    for line in open(f"/root/prof/{tag}-tree.log", errors="replace"):
        m = SPLIT.match(line)
        if m: d[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    return d
ra = [read(t) for t in A]; rb = [read(t) for t in B]
tot = 0.0
print("epoch  A argue (4 arms)            B argue (4 arms)            ΔB−A   A open  B open")
for k in range(16):
    a = [r[k][0] for r in ra if k in r]; b = [r[k][0] for r in rb if k in r]
    ao = [r[k][1] for r in ra if k in r]; bo = [r[k][1] for r in rb if k in r]
    if not a or not b: continue
    d = statistics.mean(b) - statistics.mean(a); tot += d
    print(f"{k:5d}  {' '.join(f'{x:.2f}' for x in a):28s} {' '.join(f'{x:.2f}' for x in b):28s} {d:+.3f}  {statistics.mean(ao):.2f}   {statistics.mean(bo):.2f}")
print(f"Σ Δ argue {tot:+.3f} s")
