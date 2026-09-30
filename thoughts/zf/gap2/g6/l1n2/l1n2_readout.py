#!/usr/bin/env python3
"""L1N2-noise readout: which host signal, if any, separates the arms whose level 1 runs slow from the rest.

usage: python3 l1n2_readout.py --manifest <manifest.tsv> --g6 <g6_sampler tsv> --sys <sys_sampler tsv>
                               [--gpu <nvidia-smi csv>]
       python3 l1n2_readout.py --selftest

Per arm (tree log from the manifest): three windows, the base [BASE HEAD start, LEVEL 1 START], level 1 [LEVEL 1
START, + its wall], and L1N2's prologue [LEVEL 1 START, + harvest-epochs + emit]. Per window: the prover process's
cores and runqueue wait (g6_sampler threads), the cgroup's throttled seconds, host busy cores, other processes' cores
(sys_sampler P lines, every process not named lambda_vm_prove*), CPU MHz, Tctl, PSI cpu/memory/io stall fractions,
major faults, compaction stalls, direct reclaim, available memory, and the GPU's SM clock and temperature.

Pre-registered in G6-LEDGER.md §9 before the run: an arm is SLOW if its level-1 wall >= 8.1 s or its L1N2 prologue
>= 4.95 s; each hypothesis's separation rule is in RULES below. Standard library only; prints a table, one line per
hypothesis, and a final `READOUT:` line.
"""
import argparse
import bisect
import collections
import re
import statistics
import sys

F = r"([0-9]+(?:\.[0-9]+)?)"
SLOW_L1 = 8.1
SLOW_PROLOGUE = 4.95


def series_at(ts, vs, t):
    """A counter's value at t, interpolated between samples (clamped at the ends)."""
    if not ts:
        return None
    i = bisect.bisect_left(ts, t)
    if i <= 0:
        return vs[0]
    if i >= len(ts):
        return vs[-1]
    t0, t1, v0, v1 = ts[i - 1], ts[i], vs[i - 1], vs[i]
    return v0 + (v1 - v0) * (t - t0) / (t1 - t0) if t1 > t0 else v1


def delta(ts, vs, a, b):
    x, y = series_at(ts, vs, a), series_at(ts, vs, b)
    return None if x is None or y is None or x < 0 or y < 0 else y - x


def in_window(ts, vs, a, b):
    return [v for t, v in zip(ts, vs) if a <= t <= b and v >= 0]


def read_arm(path):
    t = open(path, errors="replace").read()
    g = lambda rx, i=1: (lambda m: float(m.group(i)) if m else None)(re.search(rx, t, re.M))
    head = g(r"BASE HEAD \(WHIR\): start t=" + F)
    l1s = g(r"LEVEL 1 \(wide\) START t=" + F)
    l1 = g(r"level 1 \(wide\): [0-9]+ wide nodes over [0-9]+ epochs in " + F + "s")
    whole = g(r"WHOLE RUN: host peak " + F + r" GiB at t=" + F + r", " + F + r"s total", 3)
    m = re.search(r"L1N2 \(arity 5\) TIMING: .*harvest-epochs " + F + r"s · emit\+arenas " + F + r"s · artifacts "
                  + F + r"s · prove " + F + r"s .*wall " + F + "s", t)
    n2 = tuple(float(m.group(i)) for i in range(1, 6)) if m else None
    return {"head": head, "l1s": l1s, "l1": l1, "whole": whole, "n2": n2,
            "prologue": (n2[0] + n2[1]) if n2 else None}


def read_g6(path):
    C = ([], [], [], [], [])   # t, cgroup ns, busy, total, throttled ns
    T = collections.defaultdict(lambda: ([], [], []))
    online = 32
    for line in open(path):
        p = line.rstrip("\n").split("\t")
        if p[0].startswith("#"):
            m = re.search(r"cpus_online ([0-9]+)", line)
            online = int(m.group(1)) if m else online
        elif p[0] == "C" and len(p) >= 7:
            for k, v in enumerate((p[1], p[2], p[3], p[4], p[6])):
                C[k].append(float(v))
        elif p[0] == "T" and len(p) >= 6:
            s = T[(p[2], p[3])]
            s[0].append(float(p[1])); s[1].append(int(p[4])); s[2].append(int(p[5]))
    return C, T, online


def read_sys(path):
    keys, S, P = [], collections.defaultdict(list), []
    for line in open(path):
        p = line.rstrip("\n").split("\t")
        if p[0].startswith("#"):
            keys = line.split("keys ", 1)[1].split()
        elif p[0] == "S":
            S["t"].append(float(p[1]))
            for k, v in zip(keys, p[2:]):
                S[k].append(float(v))
        elif p[0] == "P" and len(p) >= 5:
            P.append((float(p[1]), p[3], int(p[4])))
    return S, P


def read_gpu(path):
    G = collections.defaultdict(list)
    if not path:
        return G
    import datetime
    for line in open(path, errors="replace"):
        p = [x.strip() for x in line.split(",")]
        if len(p) < 4:
            continue
        try:
            ts = datetime.datetime.strptime(p[0], "%Y/%m/%d %H:%M:%S.%f").replace(
                tzinfo=datetime.timezone.utc).timestamp()
            G["t"].append(ts); G["sm"].append(float(p[1].split()[0])); G["temp"].append(float(p[2]))
            G["power"].append(float(p[3].split()[0]))
        except (ValueError, IndexError):
            continue
    return G


def metrics(a, b, g6, sysd, gpu):
    (C, T, online), (S, P) = g6, sysd
    w = b - a
    r = {}
    on = wait = 0.0
    for ts, ons, waits in T.values():
        if not ts or ts[-1] < a or ts[0] > b:
            continue
        d1, d2 = delta(ts, ons, a, b), delta(ts, waits, a, b)
        on += (d1 or 0) / 1e9; wait += (d2 or 0) / 1e9
    r["proc_cores"] = on / w
    r["runq_s"] = wait
    d = delta(C[0], C[4], a, b)
    r["throttled_s"] = None if d is None else d / 1e9
    bu, to = delta(C[0], C[2], a, b), delta(C[0], C[3], a, b)
    r["host_busy"] = None if not to else bu / to * online
    tick = 100.0
    other = sum(j for t, c, j in P if a < t <= b + 1.0 and not c.startswith("lambda_vm_prove")) / tick
    r["other_cores"] = other / (w + 1.0)
    mhz = in_window(S["t"], S["mhz_mean"], a, b)
    r["mhz"] = statistics.mean(mhz) if mhz else None
    tc = in_window(S["t"], S["tctl"], a, b)
    r["tctl_max"] = max(tc) if tc else None
    for key, name in (("psi_cpu_some", "psi_cpu"), ("psi_mem_some", "psi_mem"), ("psi_io_some", "psi_io")):
        d = delta(S["t"], S[key], a, b)
        r[name] = None if d is None else d / 1e6 / w
    for key in ("pgmajfault", "compact_stall", "pgscan_direct"):
        d = delta(S["t"], S[key], a, b)
        r[key] = None if d is None else d
    ma = in_window(S["t"], S["memavailable"], a, b)
    r["memavail_gib"] = min(ma) / 2 ** 20 if ma else None
    sm = in_window(gpu["t"], gpu["sm"], a, b) if gpu else []
    r["gpu_sm"] = statistics.mean(sm) if sm else None
    gt = in_window(gpu["t"], gpu["temp"], a, b) if gpu else []
    r["gpu_temp"] = max(gt) if gt else None
    return r


RULES = [
    ("H-ext: other processes", "other_cores", lambda o, n: min(o) >= max(n) + 0.5),
    ("H-quota: CFS throttling", "throttled_s", lambda o, n: min(o) >= max(0.15, 3 * statistics.median(n))),
    ("H-freq: CPU clock", "mhz", lambda o, n: max(o) <= 0.97 * min(n)),
    ("H-thermal: Tctl", "tctl_max", lambda o, n: min(o) >= 95.0 and max(n) < 95.0),
    ("H-psi: CPU pressure", "psi_cpu", lambda o, n: min(o) >= 1.5 * max(n) and min(o) > 0),
    ("H-mem: faults / reclaim", "pgmajfault", lambda o, n: min(o) >= 50 and max(n) < 50),
    ("H-mem: direct reclaim", "pgscan_direct", lambda o, n: min(o) > 0 and max(n) == 0),
    ("H-internal: runqueue wait", "runq_s", lambda o, n: min(o) >= 1.5 * statistics.median(n)),
]


def readout(arms, g6, sysd, gpu):
    rows = []
    for tag, name, path in arms:
        r = read_arm(path)
        if r["l1s"] is None or r["l1"] is None or r["prologue"] is None:
            continue
        slow = r["l1"] >= SLOW_L1 or r["prologue"] >= SLOW_PROLOGUE
        pro = metrics(r["l1s"], r["l1s"] + r["prologue"], g6, sysd, gpu)
        lev = metrics(r["l1s"], r["l1s"] + r["l1"], g6, sysd, gpu)
        base = metrics(r["head"], r["l1s"], g6, sysd, gpu) if r["head"] else {}
        rows.append((tag, r, slow, pro, lev, base))
    cols = ["proc_cores", "runq_s", "throttled_s", "host_busy", "other_cores", "mhz", "tctl_max", "psi_cpu",
            "psi_mem", "psi_io", "pgmajfault", "compact_stall", "pgscan_direct", "memavail_gib", "gpu_sm", "gpu_temp"]
    f = lambda v: "NA" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))
    print("| tag | whole | level 1 | L1N2 prologue (harvest+emit) | artifacts | prove | SLOW | " + " | ".join(cols) + " |")
    print("|" + "---|" * (7 + len(cols)))
    for tag, r, slow, pro, _lev, _b in rows:
        n2 = r["n2"]
        print(f"| {tag} | {f(r['whole'])} | {f(r['l1'])} | {r['prologue']:.2f} ({n2[0]:.2f}+{n2[1]:.2f}) | {n2[2]:.2f} | "
              f"{n2[3]:.2f} | {'SLOW' if slow else ''} | " + " | ".join(f(pro.get(c)) for c in cols) + " |")
    print("(metrics over each arm's L1N2 prologue window; level-1 and base windows follow)")
    for label, idx in (("level 1", 4), ("base", 5)):
        print(f"-- {label} window: " + " · ".join(
            f"{tag} proc {f(x[idx].get('proc_cores'))} runq {f(x[idx].get('runq_s'))} thr {f(x[idx].get('throttled_s'))} "
            f"other {f(x[idx].get('other_cores'))} mhz {f(x[idx].get('mhz'))} tctl {f(x[idx].get('tctl_max'))}"
            for tag, *_ , in [(r[0],) for r in rows] for x in [next(y for y in rows if y[0] == tag)]))
    O = [x for x in rows if x[2]]
    N = [x for x in rows if not x[2]]
    print(f"arms {len(rows)} · SLOW {len(O)} {[x[0] for x in O]} · normal {len(N)}")
    if len(O) < 1 or len(N) < 3:
        return f"READOUT: NO SEPARATION TEST — {len(O)} slow arm(s), {len(N)} normal (need >= 1 and >= 3)", rows
    named = []
    for label, key, rule in RULES:
        o = [x[3].get(key) for x in O]; n = [x[3].get(key) for x in N]
        if any(v is None for v in o + n):
            print(f"{label}: NA (a value is missing)")
            continue
        ok = rule(o, n)
        print(f"{label}: {'SEPARATES' if ok else 'no'} · slow {[round(v, 3) for v in o]} · normal "
              f"{[round(v, 3) for v in n]}")
        if ok:
            named.append(label)
    ext = [h for h in named if not h.startswith("H-internal")]
    if ext:
        verdict = "HOST SIGNAL: " + "; ".join(ext)
    elif named:
        verdict = "INTERNAL CONTENTION (runqueue wait separates, no host signal does)"
    else:
        verdict = "UNRESOLVED (no signal separates the slow arms)"
    return f"READOUT: {verdict} · slow {len(O)} of {len(rows)} arms", rows


def selftest():
    """Synthetic: three arms, the third slow with other processes busy in its prologue window."""
    import os
    import tempfile
    d = tempfile.mkdtemp()
    arms = []
    for k, (l1, harvest, emit) in enumerate(((7.7, 2.1, 2.4), (7.8, 2.2, 2.3), (7.75, 2.1, 2.4), (8.6, 2.7, 2.7))):
        t0 = 1000.0 + 50 * k
        p = os.path.join(d, f"a{k}.log")
        open(p, "w").write(
            f"BASE HEAD (WHIR): start t={t0:.3f}\n   ★ LEVEL 1 (wide) START t={t0 + 30:.3f} (prove-split clock)\n"
            f"   L1N2 (arity 5) TIMING: wide, epochs 10..=14 · harvest-epochs {harvest:.2f}s · emit+arenas {emit:.2f}s"
            f" · artifacts 0.28s · prove 2.80s · harvest 0.14s (verify 0.13) · wall {l1:.2f}s\n"
            f"   level 1 (wide): 3 wide nodes over 15 epochs in {l1:.1f}s\n"
            f"★★★ WHOLE RUN: host peak 16.0 GiB at t={t0 + 35:.1f}, {31 + l1:.1f}s total\n")
        arms.append((f"wt{k}", "A", p))
    g6 = os.path.join(d, "g6.tsv"); sy = os.path.join(d, "sys.tsv")
    with open(g6, "w") as f:
        f.write("# g6_sampler v2 · cpus_online 32\n")
        for i in range(0, 4000):
            t = 990.0 + i * 0.05
            f.write(f"C\t{t:.3f}\t{int(t * 1e10)}\t{int(t * 1000)}\t{int(t * 3200)}\t0\t0\n")
            f.write(f"T\t{t:.3f}\t1\t1\t{int(t * 8e9)}\t{int(t * 1e8)}\n")
    keys = ["mhz_min", "mhz_mean", "mhz_max", "tctl", "tccd1", "tccd2", "busy", "total", "psi_cpu_some",
            "psi_mem_some", "psi_mem_full", "psi_io_some", "psi_io_full", "load1", "memfree", "memavailable",
            "cached", "dirty", "writeback", "anonhugepages", "shmem", "pgmajfault", "pgfault", "compact_stall",
            "pgscan_direct", "pgscan_kswapd", "thp_fault_alloc", "cg_nr_throttled", "cg_throttled_ns", "cg_usage_ns"]
    with open(sy, "w") as f:
        f.write("# sys_sampler v1 · interval 0.25 · cpus 32 · keys " + " ".join(keys) + "\n")
        for i in range(0, 800):
            t = 990.0 + i * 0.25
            vals = [5000, 5100, 5200, 80, -1, -1, int(t * 1000), int(t * 3200), int(t * 1e5), 0, 0, 0, 0, 5.0,
                    40e6, 50e6, 1e6, 0, 0, 0, 0, 10, 1000, 0, 0, 0, 0, 0, 0, -1]
            f.write("S\t%.3f\t%s\n" % (t, "\t".join(str(v) for v in vals)))
            if i % 4 == 0:
                busy = 400 if 1180 <= t <= 1186 else 2
                f.write(f"P\t{t:.3f}\t77\tintruder\t{busy}\n")
    line, rows = readout(arms, read_g6(g6), read_sys(sy), None)
    print(line)
    assert "H-ext: other processes" in line, line
    assert sum(1 for r in rows if r[2]) == 1
    print("SELFTEST OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest")
    ap.add_argument("--g6")
    ap.add_argument("--sys")
    ap.add_argument("--gpu")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        sys.exit(0)
    arms = []
    for line in open(a.manifest):
        if line.startswith("#") or not line.strip():
            continue
        tag, name, _knobs, log = line.rstrip("\n").split("\t")[:4]
        arms.append((tag, name, log))
    line, _ = readout(arms, read_g6(a.g6), read_sys(a.sys), read_gpu(a.gpu))
    print(line)
