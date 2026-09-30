#!/usr/bin/env python3
# Read-only: contention in level 1 from G6's raw sampler files (wt1201-wt1203), in two windows:
# the L1N2 prologue [L1 start, + harvest-epochs + emit] and the whole level 1 [L1 start, + level-1 wall].
import bisect, collections, re
F = r"([0-9]+(?:\.[0-9]+)?)"
for tag in ("wt1201", "wt1202", "wt1203"):
    log = open(f"/root/prof/{tag}-tree.log", errors="replace").read()
    l1s = float(re.search(r"LEVEL 1 \(wide\) START t=" + F, log).group(1))
    l1w = float(re.search(r"level 1 \(wide\): [0-9]+ wide nodes over [0-9]+ epochs in " + F + "s", log).group(1))
    m = re.search(r"L1N2 \(arity 5\) TIMING: .*harvest-epochs " + F + r"s · emit\+arenas " + F + "s", log)
    pro = float(m.group(1)) + float(m.group(2))
    C, T, N = [], collections.defaultdict(list), {}
    header = ""
    for line in open(f"/root/zf/g6/sampler/{tag}.tsv"):
        p = line.rstrip("\n").split("\t")
        if p[0].startswith("#"):
            header = line.strip()
        elif p[0] == "C":
            C.append(tuple(float(x) for x in p[1:7]))
        elif p[0] == "T":
            T[(p[2], p[3])].append((float(p[1]), int(p[4]), int(p[5])))
        elif p[0] == "N":
            N[(p[1], p[2])] = p[3] if len(p) > 3 else "?"
    online = int(re.search(r"cpus_online ([0-9]+)", header).group(1))
    def at(series, t, idx):
        ts = [s[0] for s in series]
        i = bisect.bisect_left(ts, t)
        if i <= 0: return series[0][idx]
        if i >= len(series): return series[-1][idx]
        (t0, t1) = (series[i - 1][0], series[i][0]); (v0, v1) = (series[i - 1][idx], series[i][idx])
        return v0 + (v1 - v0) * (t - t0) / (t1 - t0) if t1 > t0 else v1
    for name, (a, b) in (("L1N2 prologue", (l1s, l1s + pro)), ("level 1", (l1s, l1s + l1w))):
        w = b - a
        cg = (at(C, b, 1) - at(C, a, 1)) / 1e9 / w
        busy = (at(C, b, 2) - at(C, a, 2)); tot = (at(C, b, 3) - at(C, a, 3))
        box = busy / tot * online if tot > 0 else float("nan")
        thr_p = at(C, b, 4) - at(C, a, 4); thr_s = (at(C, b, 5) - at(C, a, 5)) / 1e9
        on = wait = 0.0; per = collections.Counter()
        for key, s in T.items():
            if s[-1][0] < a or s[0][0] > b: continue
            d_on = (at(s, b, 1) - at(s, a, 1)) / 1e9; d_w = (at(s, b, 2) - at(s, a, 2)) / 1e9
            on += d_on; wait += d_w; per[N.get(key, "?")] += d_on
        top = ", ".join(f"{k} {v:.1f}" for k, v in per.most_common(5))
        print(f"{tag} {name:14s} {w:5.2f}s · process {on/w:5.2f} cores (runqueue wait {wait:5.2f} s) · cgroup {cg:5.2f} cores · "
              f"box {box:5.2f} of {online} · throttled {thr_p:.0f} periods {thr_s:.3f} s · top threads (cpu-s): {top}")
