#!/usr/bin/env python3
"""prof4.sh's text tool: per-stage tables from one Nsight Systems export, the ncu pass plan, and the ncu summaries.

Python 3.8+, standard library only. It reads an export's kernel / memcpy / memset activity, StringIds, the session start
and the GPU metrics; never META_DATA_CAPTURE or TARGET_INFO_SYSTEM_ENV (the process environment).

    prof4_summary.py runa   --sqlite F --log L --kind W|S --out DIR [--host F] [--smi F] [--tz +0000]
    prof4_summary.py plan   --sqlite F --log L --kind W|S --out plan.tsv [--max N] [--cover 0.85] [--prefix P]
    prof4_summary.py ncu    --csv F --pass NAME --out launches.tsv
    prof4_summary.py report --dir RESULTS            (writes RESULTS/SUMMARY.md from what the steps wrote)
    prof4_summary.py stamp                           (stdin → stdout, each line prefixed with the unix time it arrived)
    prof4_summary.py selftest

Stages come from the prover's own log lines, each prefixed with the unix time it was printed (the script's stamper):
W = #1014 (W3 PROVE START unix, BLOCK PHASES A and its wait, W3 BASE, W3 LEVEL 0, W3 RECURSION), S = #1013 (BLOCK PHASE
stream / prove, harvest, BLOCK LEVEL 0, the NO-EPOCH BLOCK line). Card busy = the union of kernels, copies and memsets in
a stage; kernel seconds are raw (overlapping kernels both count). GPU metrics are stage means of the samples.
"""
import argparse
import bisect
import csv
import datetime
import os
import re
import sqlite3
import statistics
import sys
import tempfile
from collections import defaultdict

FORBIDDEN = ("META_DATA_CAPTURE", "TARGET_INFO_SYSTEM_ENV")
FAM = [
    ("hash P1 leaves", r"^(p1s_(leaves|fri_group_leaves)|p1w16_zleaves)"),
    ("hash P1 Merkle", r"^p1s_merkle"),
    ("hash P1 grind", r"^(p1s_grind|p1w8_grind)"),
    ("transpose / gather / pack", r"^(p1s_gather|matrix_transpose|transpose|bit_reverse|gather|merkle_gather|pack|unpack|narrow|widen_)"),
    ("hash P1 other", r"^p1(s|w8|w16)_"),
    ("hash RPX grind", r"^rpx_grind"),
    ("hash RPX leaves", r"^rpx_(leaves|comp_poly_leaves|fri_group_leaves|fri_leaves|.*leaves)"),
    ("hash RPX Merkle", r"^rpx_(merkle|tail)"),
    ("hash RPX other", r"^rpx_"),
    ("hash other (keccak / blake3)", r"^(keccak|blake3)"),
    ("NTT / LDE", r"^(ntt_|mobius_|lift_spread|pointwise_mul|scalar_mul|coset|lde_)"),
    ("sumcheck / GKR / fold", r"^(sumcheck_|sum_partials|whir_|eq_expand|eq_seed|fraction_fold|mle_|fri_fold|fri_update|add_scaled|fill_ext3|gkr_|zc_)"),
    ("constraint / LogUp / quotient", r"^(constraint_|ccomp_|si_|program_|interp|factors_from|logup_|decompose|comp_h|quotient|bytecode|compiled_|lfm_constraint)"),
    ("DEEP / barycentric / inversion", r"^(deep_|barycentric|compute_denoms|batch_inverse|invert|block_inclusive_scan|apply_block_offsets|ood_)"),
]
FAM_RX = [(f, re.compile(r)) for f, r in FAM]
# The GPU metrics a stage row shows (names as nsys 2025.3 writes them on an RTX 5090), then every metric in metrics.tsv.
KEY_METRICS = [
    ("GR active %", re.compile(r"^GR Active \[Throughput %\]")),
    ("SMs active %", re.compile(r"^SMs Active \[Throughput %\]")),
    ("SM issue %", re.compile(r"^SM Issue \[Throughput %\]")),
    ("warps in flight %", re.compile(r"^Compute Warps in Flight \[Throughput %\]")),
    ("DRAM read %", re.compile(r"^DRAM Read Bandwidth \[Throughput %\]")),
    ("DRAM write %", re.compile(r"^DRAM Write Bandwidth \[Throughput %\]")),
    ("PCIe RX %", re.compile(r"^PCIe RX Throughput \[Throughput %\]")),
    ("PCIe TX %", re.compile(r"^PCIe TX Throughput \[Throughput %\]")),
]


def family(name):
    for f, rx in FAM_RX:
        if rx.search(name):
            return f
    return "other kernels"


def rd(p, d=""):
    try:
        with open(p, errors="replace") as f:
            return f.read()
    except OSError:
        return d


def wtsv(path, header, rows):
    with open(path, "w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join("" if x is None else str(x) for x in r) + "\n")


def f2(x, n=2):
    return "-" if x is None else f"{x:.{n}f}"


# ------------------------------------------------------------------------------------------------ the log's stages
def stamped(path):
    out = []
    for line in rd(path).splitlines():
        t, _, txt = line.partition(" ")
        try:
            out.append((float(t), txt))
        except ValueError:
            pass
    return out


def first(lt, rx):
    c = re.compile(rx)
    for t, txt in lt:
        m = c.search(txt)
        if m:
            return t, m
    return None, None


def stages(lt, kind):
    """[(name, a, b)] in unix seconds and the parsed phase numbers; [] when the run did not reach its end line.
    The same rules as the box ladder's iprof4_windows.py (I-PROF4), dry-run there on real captures."""
    info = {}
    if kind == "W":
        _, m0 = first(lt, r"^W3 PROVE START: unix ([\d.]+)")
        _, mp = first(lt, r"^BLOCK PHASES: .*?· A ([\d.]+) \(wait ([\d.]+) .*?· B ([\d.]+)")
        _, mb = first(lt, r"^W3 BASE: ([\d.]+)s")
        _, ml = first(lt, r"^W3 LEVEL 0: ([\d.]+)s wall")
        _, mr = first(lt, r"^W3 RECURSION: ([\d.]+)s after the base .*?whole block ([\d.]+)s")
        _, mh = first(lt, r"^W3 BASE HASH: (\w+)")
        _, mg = first(lt, r"^W3 PLAN: (\d+) groups · .*? · (\d+) leaves ")
        info["hash"] = mh.group(1) if mh else "-"
        info["plan"] = f"{mg.group(1)} groups / {mg.group(2)} leaves" if mg else "-"
        if not (m0 and mp and mb and ml and mr):
            return [], info
        t0, A, wait, base, L0 = (float(x) for x in (m0.group(1), mp.group(1), mp.group(2), mb.group(1), ml.group(1)))
        rec, whole = float(mr.group(1)), float(mr.group(2))
        info.update(A=A, B=base - A, base=base, L0=L0, rec=rec, whole=whole, wait=wait)
        return [("A: wait for group 0", t0, t0 + wait), ("A: build + commits", t0 + wait, t0 + A),
                ("B: argue + open", t0 + A, t0 + base), ("L0 (leaves)", t0 + base, t0 + base + L0),
                ("interior (L1..top)", t0 + base + L0, t0 + whole), ("phase A", t0, t0 + A),
                ("recursion", t0 + base, t0 + whole), ("WHOLE", t0, t0 + whole)], info
    te, me = first(lt, r"★★★ NO-EPOCH BLOCK: base ([\d.]+)s · harvest ([\d.]+)s · level 0 ([\d.]+)s .*?recursion ([\d.]+)s · whole ([\d.]+)s")
    tA, _ = first(lt, r"BLOCK PHASE stream [\d.]+s")
    tB, _ = first(lt, r"BLOCK PHASE prove [\d.]+s")
    th, _ = first(lt, r"^\s+harvest: [\d.]+s \(")
    tl, _ = first(lt, r"^\s+BLOCK LEVEL 0: ")
    _, mh = first(lt, r"^BASE HASH: (\w+)")
    _, mi = first(lt, r"harvest: [\d.]+s .*?· (\d+) instances")
    _, ml = first(lt, r"^\s+BLOCK LEVEL 0: (\d+) leaves")
    info["hash"] = mh.group(1) if mh else "-"
    info["plan"] = f"{mi.group(1) if mi else '?'} instances / {ml.group(1) if ml else '?'} leaves"
    if not (te and tA and tB and th and tl):
        return [], info
    base, harvest, L0, rec, whole = (float(me.group(i)) for i in (1, 2, 3, 4, 5))
    t0 = te - whole
    info.update(A=tA - t0, B=tB - tA, base=base, L0=L0, rec=rec, whole=whole, harvest=harvest)
    return [("phase A (execute + build + streamed commits)", t0, tA), ("phase B (setup + prove)", tA, tB),
            ("harvest", tB, th), ("L0 (leaves)", th, tl), ("interior (L1..top)", tl, te), ("recursion", tB, te),
            ("WHOLE", t0, te)], info


# ------------------------------------------------------------------------------------------------ the export
class Export:
    def __init__(self, path):
        self.db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        self.tables = {r[0] for r in self.db.execute("select name from sqlite_master where type='table'")}
        self.t0 = self.q("select utcEpochNs from TARGET_INFO_SESSION_START_TIME")[0][0]
        self.names = {}
        if "StringIds" in self.tables:
            self.names = dict(self.q("select id, value from StringIds"))
        self.k = []   # (start, end, name, grid, block, corr)
        if "CUPTI_ACTIVITY_KIND_KERNEL" in self.tables:
            for s, e, n, gx, gy, gz, bx, by, bz, c in self.q(
                    "select start, end, shortName, gridX, gridY, gridZ, blockX, blockY, blockZ, correlationId "
                    "from CUPTI_ACTIVITY_KIND_KERNEL order by start"):
                self.k.append((s, e, self.names.get(n, f"?{n}"), f"{gx}x{gy}x{gz}", f"{bx}x{by}x{bz}", c))
        self.c = []   # (start, end, kind, bytes)
        if "CUPTI_ACTIVITY_KIND_MEMCPY" in self.tables:
            kinds = {1: "H2D", 2: "D2H", 8: "D2D", 10: "P2P"}
            for s, e, ck, b in self.q("select start, end, copyKind, bytes from CUPTI_ACTIVITY_KIND_MEMCPY order by start"):
                self.c.append((s, e, kinds.get(ck, f"kind{ck}"), b))
        self.m = []
        if "CUPTI_ACTIVITY_KIND_MEMSET" in self.tables:
            self.m = [(s, e) for s, e in self.q("select start, end from CUPTI_ACTIVITY_KIND_MEMSET order by start")]
        self.kstart = [x[0] for x in self.k]
        self.kmax = max((x[1] - x[0] for x in self.k), default=0)
        self.cstart = [x[0] for x in self.c]
        self.cmax = max((x[1] - x[0] for x in self.c), default=0)
        self.mstart = [x[0] for x in self.m]
        self.mmax = max((x[1] - x[0] for x in self.m), default=0)
        self.metric_names = {}
        self.metric_ts = None
        if "GPU_METRICS" in self.tables:
            cols = {r[1] for r in self.db.execute("pragma table_info(GPU_METRICS)")}
            self.metric_ts = next((c for c in ("timestamp", "start") if c in cols), None)
            if "TARGET_INFO_GPU_METRICS" in self.tables:
                icols = {r[1] for r in self.db.execute("pragma table_info(TARGET_INFO_GPU_METRICS)")}
                if {"metricId", "metricName"} <= icols:
                    tsel = "typeId" if "typeId" in icols else "0"
                    for t, mid, n in self.q(f"select distinct {tsel}, metricId, metricName from TARGET_INFO_GPU_METRICS"):
                        self.metric_names[(t, mid)] = n
            self.metric_type = "typeId" if "typeId" in cols else "0"

    def q(self, sql):
        assert not any(f in sql for f in FORBIDDEN)
        return self.db.execute(sql).fetchall()

    def ns(self, unix):
        return int(round(unix * 1e9)) - self.t0

    def span(self):
        ops = [x for x in (self.k[:1] + self.c[:1]) if x]
        if not ops:
            return None
        a = min(x[0] for x in ops)
        b = max(max((x[1] for x in self.k), default=0), max((x[1] for x in self.c), default=0))
        return a, b

    @staticmethod
    def clip(rows, starts, maxdur, a, b):
        i = bisect.bisect_left(starts, a - maxdur)
        j = bisect.bisect_left(starts, b)
        for r in rows[i:j]:
            s, e = max(r[0], a), min(r[1], b)
            if e > s:
                yield r, s, e

    @staticmethod
    def union(iv):
        tot, cur_s, cur_e = 0, None, None
        for s, e in sorted(iv):
            if cur_e is None or s > cur_e:
                if cur_e is not None:
                    tot += cur_e - cur_s
                cur_s, cur_e = s, e
            elif e > cur_e:
                cur_e = e
        if cur_e is not None:
            tot += cur_e - cur_s
        return tot

    def window(self, a, b):
        kiv, fam, kern = [], defaultdict(float), defaultdict(lambda: [0.0, 0, defaultdict(float)])
        for r, s, e in self.clip(self.k, self.kstart, self.kmax, a, b):
            kiv.append((s, e))
            d = (e - s) / 1e9
            fam[family(r[2])] += d
            kk = kern[r[2]]
            kk[0] += d
            if a <= r[0] < b:
                kk[1] += 1
            kk[2][f"{r[3]}/{r[4]}"] += d
        civ, cps = [], defaultdict(lambda: [0, 0.0, 0])
        for r, s, e in self.clip(self.c, self.cstart, self.cmax, a, b):
            civ.append((s, e))
            x = cps[r[2]]
            x[1] += (e - s) / 1e9
            if a <= r[0] < b:
                x[0] += r[3]
                x[2] += 1
        miv = [(s, e) for _, s, e in self.clip(self.m, self.mstart, self.mmax, a, b)]
        return dict(wall=(b - a) / 1e9, kunion=self.union(kiv) / 1e9, busy=self.union(kiv + civ + miv) / 1e9,
                    fam=dict(fam), kern=dict(kern), copies=dict(cps), metrics=self.metrics(a, b))

    def metrics(self, a, b):
        if not self.metric_ts:
            return {}
        out = {}
        for t, mid, avg, n in self.q(
                f"select {self.metric_type}, metricId, avg(case when value >= 0 then value end), count(*) from GPU_METRICS "
                f"where {self.metric_ts} >= {int(a)} and {self.metric_ts} < {int(b)} group by 1, 2"):
            out[self.metric_names.get((t, mid), f"metric {mid}")] = (avg, n)
        return out


def key_metric(ms, label):
    rx = dict(KEY_METRICS)[label]
    for n, (v, _) in ms.items():
        if rx.search(n):
            return v
    return None


# ------------------------------------------------------------------------------------------------ host side
def host_window(path, a, b):
    """(prover cores, prover VmRSS peak GiB, MemAvailable floor GiB) over [a, b] unix, from prof4_sampler.py."""
    rows = []
    for i, line in enumerate(rd(path).splitlines()):
        q = line.split("\t")
        if i == 0 or len(q) < 10:
            continue
        try:
            rows.append((float(q[0]), int(q[5]), int(q[6]), int(q[9])))
        except ValueError:
            pass
    inw = [r for r in rows if a <= r[0] <= b]
    tk = [r for r in inw if r[2] >= 0]
    cores = ((tk[-1][2] - tk[0][2]) / 100.0 / (tk[-1][0] - tk[0][0])) if len(tk) >= 2 and tk[-1][0] > tk[0][0] else None
    rss = max((r[1] for r in inw), default=-1) / 2**20
    avail = min((r[3] for r in inw if r[3] >= 0), default=-1) / 2**20
    return cores, (rss if rss > 0 else None), (avail if avail > 0 else None)


def smi_window(path, tz, a, b):
    off = (1 if tz[0] == "+" else -1) * (int(tz[1:3]) * 3600 + int(tz[3:5]) * 60)
    u, vr, sm = [], 0, []
    for line in rd(path).splitlines():
        q = [x.strip() for x in line.split(",")]
        try:
            t = datetime.datetime.strptime(q[0], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=datetime.timezone.utc).timestamp() - off
        except (ValueError, IndexError):
            continue
        if a <= t <= b:
            try:
                u.append(int(q[2])); vr = max(vr, int(q[1]))
                if int(q[2]) >= 50:
                    sm.append(int(q[3]))
            except (ValueError, IndexError):
                pass
    return (sum(u) / len(u) if u else None), (vr or None), (int(statistics.median(sm)) if sm else None)


# ------------------------------------------------------------------------------------------------ runa
def cmd_runa(a):
    x = Export(a.sqlite)
    lt = stamped(a.log)
    st, info = stages(lt, a.kind)
    if st:
        wins = [(n, x.ns(p), x.ns(q)) for n, p, q in st]
        unix = {n: (p, q) for n, p, q in st}
    else:
        sp = x.span()
        wins = [("WHOLE (capture span: the log has no stage lines)", sp[0], sp[1])] if sp else []
        unix = {wins[0][0]: ((wins[0][1] + x.t0) / 1e9, (wins[0][2] + x.t0) / 1e9)} if wins else {}
    os.makedirs(a.out, exist_ok=True)
    srow, frow, krow, mrow, crow = [], [], [], [], []
    md = [f"# run A · {os.path.basename(a.out)} · {a.kind} · base hash {info.get('hash')} · {info.get('plan')}", "",
          f"Prover's own times (s): whole {f2(info.get('whole'))} · A {f2(info.get('A'))} · B {f2(info.get('B'))} · "
          f"recursion {f2(info.get('rec'))}", "",
          "| stage | wall s | card busy % | kernel union % | GR active % | SMs active % | SM issue % | warps in flight % | "
          "DRAM rd / wr % | PCIe rx / tx % | smi util % | prover cores | RSS peak GiB |",
          "|---" * 13 + "|"]
    fams = [f for f, _ in FAM] + ["other kernels"]
    for name, p, q in wins:
        w = x.window(p, q)
        ua, ub = unix[name]
        cores, rss, avail = host_window(a.host, ua, ub) if a.host else (None, None, None)
        su, vr, smhz = smi_window(a.smi, a.tz, ua, ub) if a.smi else (None, None, None)
        km = {lab: key_metric(w["metrics"], lab) for lab, _ in KEY_METRICS}
        srow.append((name, f"{w['wall']:.3f}", f"{w['busy']:.3f}", f"{w['kunion']:.3f}") + tuple(f2(km[lab]) for lab, _ in KEY_METRICS)
                    + (f2(su), vr, smhz, f2(cores), f2(rss), f2(avail)))
        busy_pct = 100 * w["busy"] / w["wall"] if w["wall"] else None
        md.append(f"| {name} | {w['wall']:.2f} | {f2(busy_pct, 1)} | {f2(100 * w['kunion'] / w['wall'] if w['wall'] else None, 1)} | "
                  f"{f2(km['GR active %'], 1)} | {f2(km['SMs active %'], 1)} | {f2(km['SM issue %'], 1)} | {f2(km['warps in flight %'], 1)} | "
                  f"{f2(km['DRAM read %'], 1)} / {f2(km['DRAM write %'], 1)} | {f2(km['PCIe RX %'], 1)} / {f2(km['PCIe TX %'], 1)} | "
                  f"{f2(su, 1)} | {f2(cores, 1)} | {f2(rss)} |")
        for fm in fams:
            if w["fam"].get(fm):
                frow.append((name, fm, f"{w['fam'][fm]:.3f}"))
        for kn, (s, n, shapes) in sorted(w["kern"].items(), key=lambda kv: -kv[1][0])[:40]:
            main = max(shapes.items(), key=lambda kv: kv[1])[0]
            krow.append((name, kn, f"{s:.4f}", n, f"{1e6 * s / n:.1f}" if n else "-", main, family(kn)))
        for mn, (v, n) in sorted(w["metrics"].items()):
            mrow.append((name, mn, f2(v, 3), n))
        for ck, (by, s, n) in sorted(w["copies"].items()):
            crow.append((name, ck, n, by, f"{s:.3f}", f"{by / s / 1e9:.1f}" if s else "-"))
    wtsv(f"{a.out}/stages.tsv", ["stage", "wall_s", "busy_s", "kernel_union_s"] + [lab for lab, _ in KEY_METRICS]
         + ["smi_util", "vram_mib_max", "sm_mhz_loaded", "prover_cores", "rss_peak_gib", "memavail_min_gib"], srow)
    wtsv(f"{a.out}/families.tsv", ["stage", "family", "kernel_s"], frow)
    wtsv(f"{a.out}/kernels.tsv", ["stage", "kernel", "kernel_s", "launches", "mean_us", "main_shape", "family"], krow)
    wtsv(f"{a.out}/metrics.tsv", ["stage", "metric", "mean", "samples"], mrow)
    wtsv(f"{a.out}/copies.tsv", ["stage", "kind", "count", "bytes", "seconds", "GB_per_s"], crow)
    md += ["", "Kernel seconds by family (raw, overlaps count twice):", "",
           "| stage | " + " | ".join(fams) + " |", "|---" * (1 + len(fams)) + "|"]
    for name, _, _ in wins:
        fs = {f: float(s) for st_, f, s in frow if st_ == name}
        md.append(f"| {name} | " + " | ".join(f2(fs.get(f)) for f in fams) + " |")
    md += ["", "Top kernels over the last stage (WHOLE):", ""]
    for r in [r for r in krow if r[0] == wins[-1][0]][:15] if wins else []:
        md.append(f"- {r[1]}: {r[2]} s, {r[3]} launches, mean {r[4]} µs, main shape {r[5]}")
    with open(f"{a.out}/runa.md", "w") as f:
        f.write("\n".join(md) + "\n")
    wh = next((r for r in srow if r[0].startswith("WHOLE")), None)
    print(f"RUNA {os.path.basename(a.out)}: stages {len(wins)} · whole {f2(info.get('whole'))} s · card busy "
          f"{f2(100 * float(wh[2]) / float(wh[1]) if wh and float(wh[1]) else None, 1)} % · kernels {len(x.k)} · "
          f"GPU-metric names {len(x.metric_names)}")


# ------------------------------------------------------------------------------------------------ plan
def cmd_plan(a):
    x = Export(a.sqlite)
    st, _ = stages(stamped(a.log), a.kind)
    if st:
        wins = [(n, x.ns(p), x.ns(q)) for n, p, q in st if not n.startswith(("WHOLE", "phase A", "recursion"))]
        whole = next((x.ns(p), x.ns(q)) for n, p, q in st if n == "WHOLE")
    else:
        sp = x.span()
        wins, whole = [("capture", sp[0], sp[1])], sp
    tot = defaultdict(float)
    for s, e, n, *_ in x.k:
        if whole[0] <= s < whole[1]:
            tot[n] += (e - s) / 1e9
    allk = sum(tot.values())
    by_corr = defaultdict(list)
    for s, e, n, g, b, c in x.k:
        by_corr[n].append((c, s, e, g, b))
    rows, cum = [], 0.0
    for n, s in sorted(tot.items(), key=lambda kv: -kv[1]):
        if len(rows) >= a.max or cum >= a.cover * allk or s < 0.005 * allk:
            break
        cum += s
        ls = sorted(by_corr[n])   # launch order = API call order (correlation id), as ncu counts launches
        shapes = defaultdict(float)
        for _, s0, e0, g, b in ls:
            if whole[0] <= s0 < whole[1]:
                shapes[(g, b)] += (e0 - s0) / 1e9
        g, b = max(shapes.items(), key=lambda kv: kv[1])[0]
        cand = [(i, e0 - s0, s0) for i, (_, s0, e0, gg, bb) in enumerate(ls) if (gg, bb) == (g, b) and whole[0] <= s0 < whole[1]]
        cand.sort(key=lambda t: t[1])
        idx, dur, s0 = cand[len(cand) // 2]
        stage = next((wn for wn, p, q in wins if p <= s0 < q), "-")
        rows.append((f"{a.prefix}{len(rows) + 1:02d}", n, idx, 1, g, b, f"{100 * s / allk:.1f}", f"{dur / 1e3:.1f}", stage,
                     f"{100 * shapes[(g, b)] / s:.0f}"))
    wtsv(a.out, ["pass", "kernel", "launch_skip", "launch_count", "grid", "block", "pct_of_kernel_time", "launch_us", "stage",
                 "shape_pct_of_kernel"], rows)
    print(f"PLAN {a.out}: {len(rows)} pass(es) cover {100 * cum / allk:.1f} % of {allk:.2f} kernel-s")


# ------------------------------------------------------------------------------------------------ ncu
NCU_COLS = [("duration_ms", "GPU Speed Of Light Throughput", "Duration"), ("sm_ghz", "GPU Speed Of Light Throughput", "SM Frequency"),
            ("sm_pct", "GPU Speed Of Light Throughput", "Compute (SM) Throughput"),
            ("mem_pct", "GPU Speed Of Light Throughput", "Memory Throughput"),
            ("dram_pct", "GPU Speed Of Light Throughput", "DRAM Throughput"),
            ("l2_pct", "GPU Speed Of Light Throughput", "L2 Cache Throughput"),
            ("issue_pct", "Compute Workload Analysis", "Issue Slots Busy"),
            ("occ_achieved", "Occupancy", "Achieved Occupancy"), ("occ_theoretical", "Occupancy", "Theoretical Occupancy"),
            ("regs", "Launch Statistics", "Registers Per Thread"), ("waves", "Launch Statistics", "Waves Per SM")]
PIPES = ["alu", "fma", "fmaheavy", "lsu", "xu", "uniform"]
STALLS = ["long_scoreboard", "short_scoreboard", "wait", "math_pipe_throttle", "lg_throttle", "mio_throttle", "not_selected", "barrier"]


def num(v):
    try:
        return float(str(v).replace(",", ""))
    except ValueError:
        return None


def cmd_ncu(a):
    by = defaultdict(dict)
    meta = {}
    with open(a.csv, newline="", errors="replace") as f:
        for r in csv.DictReader(f):
            i = r.get("ID")
            if i is None:
                continue
            meta.setdefault(i, (r.get("Kernel Name", "?"), r.get("Grid Size", "?").replace(" ", ""), r.get("Block Size", "?").replace(" ", "")))
            v = num(r.get("Metric Value", ""))
            if v is not None:
                by[i][(r.get("Section Name", ""), r.get("Metric Name", ""))] = v
    rows = []
    for i in sorted(by, key=lambda s: int(s) if s.isdigit() else s):
        m = by[i]
        name, grid, block = meta[i]
        vals = [m.get((s, n)) for _, s, n in NCU_COLS]
        cmd = {n: v for (s, n), v in m.items() if s.startswith("Command line profiler metric")}
        pipe = {p: cmd.get(f"sm__pipe_{p}_cycles_active.avg.pct_of_peak_sustained_active",
                           cmd.get(f"sm__inst_executed_pipe_{p}.avg.pct_of_peak_sustained_active")) for p in PIPES}
        top_pipe = max(((p, v) for p, v in pipe.items() if v is not None), key=lambda kv: kv[1], default=(None, None))
        stall = {s: cmd.get(f"smsp__average_warps_issue_stalled_{s}_per_issue_active.ratio") for s in STALLS}
        top_stall = max(((s, v) for s, v in stall.items() if v is not None), key=lambda kv: kv[1], default=(None, None))
        sm, dram, l2 = vals[2], vals[4], vals[5]
        if sm is None:
            bound = "-"
        elif sm >= 80:
            bound = "compute roof"
        elif max(dram or 0, l2 or 0) >= 80:
            bound = "memory roof"
        elif max(sm, dram or 0, l2 or 0) < 60:
            bound = "latency / under-filled"
        else:
            bound = "mid"
        rows.append((a.pass_, i, name, grid, block) + tuple(f2(v) for v in vals)
                    + (top_pipe[0] or "-", f2(top_pipe[1]), top_stall[0] or "-", f2(top_stall[1]), bound))
    wtsv(a.out, ["pass", "id", "kernel", "grid", "block"] + [c for c, _, _ in NCU_COLS]
         + ["top_pipe", "top_pipe_pct", "top_stall", "top_stall_ratio", "bound"], rows)
    print(f"NCU {a.pass_}: {len(rows)} launch(es) profiled" + (f" · {rows[0][2]} {rows[0][3]}/{rows[0][4]} SM {rows[0][7]} % · {rows[-1][-1]}" if rows else ""))


# ------------------------------------------------------------------------------------------------ report
def cmd_report(a):
    d = a.dir
    out = ["# prof4 results", "", rd(f"{d}/facts.txt").strip(), ""]
    for sub in sorted(os.listdir(f"{d}/runa")) if os.path.isdir(f"{d}/runa") else []:
        out += [rd(f"{d}/runa/{sub}/runa.md").strip(), ""]
    if os.path.isdir(f"{d}/ncu"):
        out += ["# run B (Nsight Compute, base clocks: SM ≈ 2.0 GHz, so a compute-bound launch reads ≈ 1.37× its in-block time)", "",
                "| pass | kernel | grid / block | ms | SM % | DRAM % | L2 % | issue % | occ ach / theo % | regs | waves | top pipe % | top stall | bound |",
                "|---" * 14 + "|"]
        for fn in sorted(os.listdir(f"{d}/ncu")):
            if not fn.endswith(".launches.tsv"):
                continue
            with open(f"{d}/ncu/{fn}") as f:
                for r in csv.DictReader(f, delimiter="\t"):
                    out.append(f"| {r['pass']} | {r['kernel']} | {r['grid']} / {r['block']} | {r['duration_ms']} | {r['sm_pct']} | {r['dram_pct']} | "
                               f"{r['l2_pct']} | {r['issue_pct']} | {r['occ_achieved']} / {r['occ_theoretical']} | {r['regs']} | {r['waves']} | "
                               f"{r['top_pipe']} {r['top_pipe_pct']} | {r['top_stall']} {r['top_stall_ratio']} | {r['bound']} |")
    with open(f"{d}/SUMMARY.md", "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"REPORT {d}/SUMMARY.md: {len(out)} lines")


# ------------------------------------------------------------------------------------------------ selftest
def cmd_selftest(_):
    """A synthetic export of the shape nsys 2025.3 writes (the tables this tool reads), a W log and a host file: the
    stage edges, the busy union, the families, the GPU-metric means, the plan's launch choice and the ncu parser."""
    t = tempfile.mkdtemp(prefix="prof4-selftest-")
    db = sqlite3.connect(f"{t}/x.sqlite")
    T0 = 1_800_000_000 * 10**9
    db.executescript("""
        CREATE TABLE TARGET_INFO_SESSION_START_TIME (utcEpochNs INTEGER, utcTime TEXT, localTime TEXT);
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, shortName INTEGER, gridX INTEGER, gridY INTEGER,
            gridZ INTEGER, blockX INTEGER, blockY INTEGER, blockZ INTEGER, correlationId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INTEGER, end INTEGER, copyKind INTEGER, bytes INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET (start INTEGER, end INTEGER);
        CREATE TABLE GPU_METRICS (rawTimestamp INTEGER, timestamp INTEGER, typeId INTEGER, metricId INTEGER, value INTEGER);
        CREATE TABLE TARGET_INFO_GPU_METRICS (typeId INTEGER, sourceId INTEGER, typeName TEXT, metricId INTEGER, metricName TEXT);
        CREATE TABLE TARGET_INFO_SYSTEM_ENV (name TEXT, value TEXT);
    """)
    db.execute("INSERT INTO TARGET_INFO_SESSION_START_TIME VALUES (?, '', '')", (T0,))
    db.execute("INSERT INTO TARGET_INFO_SYSTEM_ENV VALUES ('SECRET_TOKEN', 'must-never-be-read')")
    for i, n in enumerate(["p1w16_zleaves_base_coset_v2", "gkr_round_gruen", "rpx_merkle_level"], 1):
        db.execute("INSERT INTO StringIds VALUES (?, ?)", (i, n))
    S = 10**9
    # stages (trace s): A [1, 5) with wait [1, 2), B [5, 8), L0 [8, 9), interior [9, 10); whole [1, 10)
    # A: P1 leaves 2.0..4.0 (two launches, 1 s each, shapes 64x1x1 / 128x1x1); B: gkr 5.0..6.5; L0: merkle 8.0..8.5
    # copy H2D 4.5..5.5 (straddles A/B); memset 9.0..9.2; idle elsewhere
    ks = [(2 * S, 3 * S, 1, 64, 1), (3 * S, 4 * S, 1, 128, 1), (5 * S, int(6.5 * S), 2, 32, 2), (8 * S, int(8.5 * S), 3, 16, 3)]
    for s, e, n, g, c in ks:
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, 1, 1, 128, 1, 1, ?)", (s, e, n, g, c))
    db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?, ?, 1, 4000000000)", (int(4.5 * S), int(5.5 * S)))
    db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMSET VALUES (?, ?)", (9 * S, int(9.2 * S)))
    db.execute("INSERT INTO TARGET_INFO_GPU_METRICS VALUES (7, 0, 'syn', 1, 'SMs Active [Throughput %]')")
    db.execute("INSERT INTO TARGET_INFO_GPU_METRICS VALUES (7, 0, 'syn', 2, 'GR Active [Throughput %]')")
    for i in range(0, 1100):   # 100 Hz from 0 to 11 s: SMs Active 80 in A, 40 elsewhere; GR Active 100 everywhere
        ts = i * S // 100
        db.execute("INSERT INTO GPU_METRICS VALUES (?, ?, 7, 1, ?)", (ts, ts, 80 if 1 * S <= ts < 5 * S else 40))
        db.execute("INSERT INTO GPU_METRICS VALUES (?, ?, 7, 2, 100)", (ts, ts))
    db.commit()
    db.close()
    u = T0 / 1e9
    with open(f"{t}/out.log", "w") as f:
        f.write(f"{u + 1:.3f} W3 PROVE START: unix {u + 1:.3f}\n")
        f.write(f"{u + 1:.3f} W3 BASE HASH: Poseidon1\n")
        f.write(f"{u + 8:.3f} BLOCK PHASES: execute 1.00 · build 2.00 · prep 3.00 · A 4.00 (wait 1.00 upload 0.10 paid 0 commit 2 retire 0) · B 3.00 (argue 1 open 1 tax 1 = upload 0 + encode 1)\n")
        f.write(f"{u + 8:.3f} W3 BASE: 7.00s · statement at 4.00s · plan + 1 leaves emitted by 4.5s (inside the base)\n")
        f.write(f"{u + 8:.3f} W3 PLAN: 2 groups · costs [1, 1] · 1 leaves · x\n")
        f.write(f"{u + 10:.3f} W3 LEVEL 0: 1.00s wall · per program build+prove 0.1+0.9\n")
        f.write(f"{u + 10:.3f} W3 RECURSION: 2.00s after the base (tree 2.0s) · whole block 9.00s · x\n")
    with open(f"{t}/host.tsv", "w") as f:
        f.write("t\tcg_mem\tcg_anon\tcg_cpu_usec\tpid\trss_kb\tticks\tthreads\tnsys_rss_kb\tmemavail_kb\n")
        for i in range(0, 23):   # 0.5 s steps from u + 0: 6 cores, RSS 1..23 GiB, MemAvailable 50 GiB
            f.write(f"{u + i * 0.5:.3f}\t-1\t-1\t-1\t7\t{(i + 1) * 1048576}\t{i * 300}\t9\t0\t{50 * 1048576}\n")
    ok = []
    ns = argparse.Namespace(sqlite=f"{t}/x.sqlite", log=f"{t}/out.log", kind="W", out=f"{t}/runa", host=f"{t}/host.tsv", smi=None, tz="+0000")
    cmd_runa(ns)
    with open(f"{t}/runa/stages.tsv") as f:
        st = {r["stage"]: r for r in csv.DictReader(f, delimiter="\t")}
    ok.append(("stage A: build + commits wall 3.0, busy 2.5 (kernels 2..4 + copy 4.5..5)", abs(float(st["A: build + commits"]["wall_s"]) - 3.0) < 1e-6
               and abs(float(st["A: build + commits"]["busy_s"]) - 2.5) < 1e-6))
    ok.append(("stage B: busy 1.5 (copy 5..5.5 inside gkr 5..6.5: a union, not a sum)", abs(float(st["B: argue + open"]["busy_s"]) - 1.5) < 1e-6))
    ok.append(("WHOLE busy 4.7 (2 + 2.0 + 0.5 + 0.2), kernel union 4.0", abs(float(st["WHOLE"]["busy_s"]) - 4.7) < 1e-6
               and abs(float(st["WHOLE"]["kernel_union_s"]) - 4.0) < 1e-6))
    ok.append(("GPU metric means: SMs Active 80 in phase A, 40 in B; GR Active 100", st["phase A"]["SMs active %"] == "80.00"
               and st["B: argue + open"]["SMs active %"] == "40.00" and st["WHOLE"]["GR active %"] == "100.00"))
    ok.append(("host: 6 prover cores in phase A", st["phase A"]["prover_cores"] == "6.00"))
    with open(f"{t}/runa/families.tsv") as f:
        fam = {(r["stage"], r["family"]): r["kernel_s"] for r in csv.DictReader(f, delimiter="\t")}
    ok.append(("families: P1 leaves 2.0 s in A, GKR 1.5 s in B, RPX Merkle 0.5 s in L0", fam.get(("phase A", "hash P1 leaves")) == "2.000"
               and fam.get(("B: argue + open", "sumcheck / GKR / fold")) == "1.500" and fam.get(("L0 (leaves)", "hash RPX Merkle")) == "0.500"))
    cmd_plan(argparse.Namespace(sqlite=f"{t}/x.sqlite", log=f"{t}/out.log", kind="W", out=f"{t}/plan.tsv", max=5, cover=0.99, prefix="w"))
    with open(f"{t}/plan.tsv") as f:
        pl = list(csv.DictReader(f, delimiter="\t"))
    ok.append(("plan: P1 leaves first; 2 launches of two shapes, equal time: skip index of the chosen shape's launch", pl[0]["kernel"] == "p1w16_zleaves_base_coset_v2"
               and pl[0]["launch_skip"] in ("0", "1") and pl[0]["stage"] == "A: build + commits"))
    with open(f"{t}/ncu.csv", "w", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["ID", "Kernel Name", "Block Size", "Grid Size", "Section Name", "Metric Name", "Metric Unit", "Metric Value"])
        for s, n, v in [("GPU Speed Of Light Throughput", "Duration", "188.86"), ("GPU Speed Of Light Throughput", "Compute (SM) Throughput", "90.09"),
                        ("GPU Speed Of Light Throughput", "DRAM Throughput", "4.18"), ("Occupancy", "Achieved Occupancy", "74.24"),
                        ("Command line profiler metrics", "sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_active", "90.18"),
                        ("Command line profiler metrics", "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio", "4.26"),
                        ("Command line profiler metrics", "dram__bytes_read.sum", "n/a")]:
            w.writerow(["0", "p1w16_zleaves_base_coset_v2", "(128, 1, 1)", "(65536, 1, 1)", s, n, "", v])
    cmd_ncu(argparse.Namespace(csv=f"{t}/ncu.csv", pass_="w01", out=f"{t}/ncu.tsv"))
    with open(f"{t}/ncu.tsv") as f:
        nr = list(csv.DictReader(f, delimiter="\t"))
    ok.append(("ncu: duration, SM %, top pipe fmaheavy, top stall math_pipe_throttle, compute roof", nr and nr[0]["duration_ms"] == "188.86"
               and nr[0]["top_pipe"] == "fmaheavy" and nr[0]["top_stall"] == "math_pipe_throttle" and nr[0]["bound"] == "compute roof"))
    bad = [n for n, good in ok if not good]
    for n, good in ok:
        print(f"  {'ok ' if good else 'BAD'} {n}")
    print(f"SELFTEST {'GREEN' if not bad else 'RED'} ({len(ok) - len(bad)}/{len(ok)}) in {t}")
    return 0 if not bad else 1


def cmd_stamp(_):
    import time
    while True:
        line = sys.stdin.readline()
        if not line:
            return 0
        sys.stdout.write(f"{time.time():.3f} {line}")
        sys.stdout.flush()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("runa"); s.add_argument("--sqlite", required=True); s.add_argument("--log", required=True)
    s.add_argument("--kind", required=True, choices=["W", "S"]); s.add_argument("--out", required=True)
    s.add_argument("--host"); s.add_argument("--smi"); s.add_argument("--tz", default="+0000")
    s = sp.add_parser("plan"); s.add_argument("--sqlite", required=True); s.add_argument("--log", required=True)
    s.add_argument("--kind", required=True, choices=["W", "S"]); s.add_argument("--out", required=True)
    s.add_argument("--max", type=int, default=10); s.add_argument("--cover", type=float, default=0.85); s.add_argument("--prefix", default="p")
    s = sp.add_parser("ncu"); s.add_argument("--csv", required=True); s.add_argument("--pass", dest="pass_", required=True)
    s.add_argument("--out", required=True)
    s = sp.add_parser("report"); s.add_argument("--dir", required=True)
    sp.add_parser("selftest")
    sp.add_parser("stamp")
    a = p.parse_args()
    return {"runa": cmd_runa, "plan": cmd_plan, "ncu": cmd_ncu, "report": cmd_report, "selftest": cmd_selftest,
            "stamp": cmd_stamp}[a.cmd](a) or 0


if __name__ == "__main__":
    sys.exit(main())
