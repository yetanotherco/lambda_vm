#!/usr/bin/env python3
# Read-only, post hoc (NOT pre-registered): job 292's L1N2 prologue window per arm, CPU-seconds by thread name and
# the number of threads that ran; plus the level-1 nodes' stage stamps, to see what overlapped the prologue.
import bisect, collections, re
F = r"([0-9]+(?:\.[0-9]+)?)"
OUT = "/root/zf/l1n2-wt1302"
arms = [l.rstrip("\n").split("\t") for l in open("/root/zf/wt1302-wt1313/manifest.tsv") if l.strip() and not l.startswith("#")]
T = collections.defaultdict(lambda: ([], []))
N = {}
for line in open(f"{OUT}/g6.tsv"):
    p = line.rstrip("\n").split("\t")
    if p[0] == "T":
        s = T[(p[2], p[3])]; s[0].append(float(p[1])); s[1].append(int(p[4]))
    elif p[0] == "N":
        N[(p[2], p[3])] = p[4] if len(p) > 4 else (p[3] if len(p) > 3 else "?")
# N lines are "N pid tid name"
N = {}
for line in open(f"{OUT}/g6.tsv"):
    p = line.rstrip("\n").split("\t")
    if p[0] == "N" and len(p) >= 4:
        N[(p[1], p[2])] = p[3]
def at(ts, vs, t):
    i = bisect.bisect_left(ts, t)
    if i <= 0: return vs[0]
    if i >= len(ts): return vs[-1]
    return vs[i-1] + (vs[i]-vs[i-1]) * (t-ts[i-1]) / (ts[i]-ts[i-1]) if ts[i] > ts[i-1] else vs[i]
for tag, name, _k, log in arms:
    t = open(log, errors="replace").read()
    l1s = float(re.search(r"LEVEL 1 \(wide\) START t=" + F, t).group(1))
    m = re.search(r"L1N2 \(arity 5\) TIMING: .*harvest-epochs " + F + r"s · emit\+arenas " + F + r"s · artifacts " + F, t)
    pro = float(m.group(1)) + float(m.group(2))
    a, b = l1s, l1s + pro
    per = collections.Counter(); active = collections.Counter()
    for key, (ts, ons) in T.items():
        if not ts or ts[-1] < a or ts[0] > b: continue
        d = (at(ts, ons, b) - at(ts, ons, a)) / 1e9
        nm = N.get(key, "?")
        per[nm] += d
        if d > 0.05: active[nm] += 1
    # stage stamps inside level 1 (device-permit trace lines), when present
    stages = re.findall(r"STAGE (L1N[0-9]) \(arity 5\) ([a-z_ +()-]+?) ?[:(].*?t=\[" + F + "," + F + r"\]", t)
    ov = []
    for node, what, s0, s1 in stages:
        s0, s1 = float(s0), float(s1)
        o = max(0.0, min(b, s1) - max(a, s0))
        if o > 0.05 and node != "L1N2": ov.append(f"{node} {what.strip()} {o:.2f}")
    print(f"{tag} prologue {pro:.2f}s · " + ", ".join(f"{k} {v:.1f}s/{active[k]}thr" for k, v in per.most_common(6)))
    if ov: print(f"      overlapping level-1 stages: {'; '.join(ov)}")
