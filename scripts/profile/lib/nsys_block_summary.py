#!/usr/bin/env python3
"""Summarise one Nsight Systems sqlite export of the WHIR block run.

    nsys_block_summary.py --sqlite run.sqlite --log run.log [--nvsmi nvsmi.csv] --out DIR

Reads the CUPTI kernel / memcpy / memset activity, the GPU-metrics samples when the
trace has them (nsys --gpu-metrics-devices, counters unlocked), the run's own log and,
optionally, the nvidia-smi sampler CSV block_profile.sh records beside the trace.
Writes small CSV and text files only; the trace itself is never modified.

Stage windows come from the run log, not from NVTX (the record build has none): the
harness prints Unix-time stamps (`BASE EPOCH n: stage Xs t=[a,b]`, `CARD HOLD #n kind:
... t=[a,b]`, `PROVE SPLIT #n: ... t=[a,b]`, `MARK AFTER ...: ... t=T`) and nsys stores the
session's UTC start, so a stamp maps onto the trace clock as `t - session_start`. The
phases close on the harness's own summary lines, in order: `base (WHIR): N epochs in`,
`level 0: N WHIR wraps in`, `WHIR INTERIOR COMPOSED`, then the root runs to the end.
A log without those lines still gets the whole-run and per-bucket tables.

Pure standard library (python >= 3.8): sqlite3, csv, re, bisect.
"""
import argparse
import bisect
import csv
import datetime
import os
import re
import sqlite3
import sys
from collections import defaultdict

NS = 1_000_000_000

# Key GPU-metric series, matched against the metric names nsys stores (they differ a
# little between GPU generations, so by pattern, first match wins). Every metric is
# still written to the *_all_* CSVs whatever its name.
KEY_METRICS = [
    ("sm_active", re.compile(r"\bSMs? Active\b", re.I)),
    ("sm_issue", re.compile(r"\bSM Issue\b", re.I)),
    ("dram_read", re.compile(r"\bDRAM Read\b", re.I)),
    ("dram_write", re.compile(r"\bDRAM Write\b", re.I)),
    ("warps_in_flight", re.compile(r"Compute Warps in Flight", re.I)),
    ("unallocated_warps", re.compile(r"Unallocated Warps", re.I)),
    ("gr_active", re.compile(r"\bGR Active\b", re.I)),
    ("pcie_rx", re.compile(r"\bPCIe (RX|Read)", re.I)),
    ("pcie_tx", re.compile(r"\bPCIe (TX|Write)", re.I)),
]

RE_IVL = re.compile(r"\bt=\[(\d+(?:\.\d+)?),(\d+(?:\.\d+)?)\]")
RE_MARK = re.compile(r"^\s*MARK AFTER (.+?):.*\bt=(\d+(?:\.\d+)?)\s*$")
RE_BASE = re.compile(r"^BASE EPOCH (\S+): (\w+) ")
RE_CARD = re.compile(r"^CARD HOLD #\d+ (\w+):")
RE_SPLIT = re.compile(r"^PROVE SPLIT #\d+:")
CLOSERS = [
    ("base", re.compile(r"base \(WHIR\): \d+ epochs in")),
    ("level0", re.compile(r"level 0: \d+ WHIR wraps in")),
    ("interior", re.compile(r"WHIR INTERIOR COMPOSED")),
]
PHASES = ["base", "level0", "interior", "root"]


def connect(path):
    return sqlite3.connect("file:{}?mode=ro".format(os.path.abspath(path)), uri=True)


def tables(db):
    return {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")}


def columns(db, table):
    return [r[1] for r in db.execute("PRAGMA table_info({})".format(table))]


def merge(ivs):
    """Union of (start, end) intervals, returned sorted and disjoint."""
    out = []
    for s, e in sorted(ivs):
        if out and s <= out[-1][1]:
            if e > out[-1][1]:
                out[-1][1] = e
        else:
            out.append([s, e])
    return out


class Cover:
    """A disjoint interval set with O(log n) 'how much of [a, b] is covered'."""

    def __init__(self, merged):
        self.starts = [s for s, _ in merged]
        self.ends = [e for _, e in merged]
        self.prefix = [0]
        for s, e in merged:
            self.prefix.append(self.prefix[-1] + (e - s))

    def covered(self, a, b):
        if b <= a:
            return 0
        i = bisect.bisect_right(self.ends, a)
        j = bisect.bisect_left(self.starts, b)
        if i >= j:
            return 0
        total = self.prefix[j] - self.prefix[i]
        total -= max(0, a - self.starts[i])
        total -= max(0, self.ends[j - 1] - b)
        return total

    def total(self):
        return self.prefix[-1]


class Window:
    """A named set of trace-clock intervals (ns): one interval for a phase, many for a stage."""

    def __init__(self, name, kind, intervals):
        self.name = name
        self.kind = kind
        self.ivs = merge(intervals)
        self.starts = [s for s, _ in self.ivs]
        self.ends = [e for _, e in self.ivs]

    def wall(self):
        return sum(e - s for s, e in self.ivs)

    def overlap(self, s, e):
        i = bisect.bisect_right(self.ends, s)
        got = 0
        while i < len(self.ivs) and self.starts[i] < e:
            got += min(e, self.ends[i]) - max(s, self.starts[i])
            i += 1
        return got

    def contains(self, t):
        i = bisect.bisect_right(self.starts, t) - 1
        return i >= 0 and t < self.ends[i]

    def covered(self, cover):
        return sum(cover.covered(s, e) for s, e in self.ivs)

    def first(self):
        return self.ivs[0][0] if self.ivs else 0

    def last(self):
        return self.ivs[-1][1] if self.ivs else 0


def load_activity(db, tbls):
    """Kernels (start, end, name) and copies (start, end, kind, bytes), both sorted by start."""
    kernels, copies = [], []
    if "CUPTI_ACTIVITY_KIND_KERNEL" in tbls:
        kcols = columns(db, "CUPTI_ACTIVITY_KIND_KERNEL")
        name_col = next((c for c in ("shortName", "demangledName", "mangledName") if c in kcols), None)
        if name_col and "StringIds" in tbls:
            q = ("SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                 "LEFT JOIN StringIds s ON s.id = k.{} ORDER BY k.start".format(name_col))
        else:
            q = "SELECT start, end, '?' FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"
        kernels = [(s, e, n or "?") for s, e, n in db.execute(q)]
    kinds = {}
    if "ENUM_CUDA_MEMCPY_OPER" in tbls:
        try:
            kinds = {i: lbl for i, lbl in db.execute("SELECT id, label FROM ENUM_CUDA_MEMCPY_OPER")}
        except sqlite3.Error:
            kinds = {}
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tbls:
        for s, e, k, b in db.execute("SELECT start, end, copyKind, bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY"):
            copies.append((s, e, kinds.get(k, "memcpy kind {}".format(k)), b or 0))
    if "CUPTI_ACTIVITY_KIND_MEMSET" in tbls:
        for s, e, b in db.execute("SELECT start, end, bytes FROM CUPTI_ACTIVITY_KIND_MEMSET"):
            copies.append((s, e, "Memset", b or 0))
    copies.sort()
    return kernels, copies


def trace_span(db, tbls, kernels, copies):
    end = 0
    if kernels:
        end = max(end, max(e for _, e, _ in kernels))
    if copies:
        end = max(end, max(e for _, e, _, _ in copies))
    if "CUPTI_ACTIVITY_KIND_RUNTIME" in tbls:
        r = db.execute("SELECT max(end) FROM CUPTI_ACTIVITY_KIND_RUNTIME").fetchone()
        if r and r[0]:
            end = max(end, r[0])
    return 0, end


def session_start_ns(db, tbls):
    if "TARGET_INFO_SESSION_START_TIME" in tbls:
        r = db.execute("SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME").fetchone()
        if r and r[0]:
            return int(r[0])
    return None


def gpu_description(db, tbls):
    if "TARGET_INFO_GPU" not in tbls:
        return "?"
    cols = columns(db, "TARGET_INFO_GPU")
    want = [c for c in ("name", "smCount", "totalMemory", "chipName") if c in cols]
    if not want:
        return "?"
    out = []
    for row in db.execute("SELECT {} FROM TARGET_INFO_GPU".format(", ".join(want))):
        d = dict(zip(want, row))
        mem = d.get("totalMemory")
        out.append("{} ({} SMs, {:.1f} GiB{})".format(
            d.get("name", "?"), d.get("smCount", "?"), (mem or 0) / 2**30,
            ", " + d["chipName"] if d.get("chipName") else ""))
    return "; ".join(out)


def parse_log(path, t0_utc_ns):
    """Phase windows and stage windows (trace-clock ns) from the run log's stamps."""
    to_ns = lambda t: int(round(float(t) * NS)) - t0_utc_ns  # noqa: E731
    phase_i = 0
    events = defaultdict(list)  # phase -> [(a, b)]
    stages = defaultdict(list)  # stage name -> [(a, b)]
    facts = {}
    last_stamp = None
    with open(path, errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            phase = PHASES[min(phase_i, len(PHASES) - 1)]
            ivs = [(to_ns(a), to_ns(b)) for a, b in RE_IVL.findall(line)]
            m = RE_MARK.match(line)
            if m:
                t = to_ns(m.group(2))
                ivs.append((t, t))
            for a, b in ivs:
                events[phase].append((a, b))
                last_stamp = b if last_stamp is None else max(last_stamp, b)
            mb = RE_BASE.match(line)
            if mb and ivs:
                which, stage = mb.group(1), mb.group(2)
                if stage in ("prep", "absorb", "commit", "prove"):
                    stages["base/" + stage].extend(ivs)
                if which == "global":
                    stages["base/global"].extend(ivs)
            mc = RE_CARD.match(line)
            if mc and ivs:
                stages["card/" + mc.group(1)].extend(ivs)
            if RE_SPLIT.match(line) and ivs:
                stages["lfm/prove"].extend(ivs)
            for key, pat in (("whole_run", r"WHOLE RUN: host peak ([\d.]+) GiB.*?([\d.]+)s total"),
                             ("base_wall", r"base \(WHIR\): (\d+) epochs in ([\d.]+)s"),
                             ("level0_wall", r"level 0: (\d+) WHIR wraps in ([\d.]+)s"),
                             ("test_result", r"^test result: (.*)$"),
                             ("commit_fallbacks", r"commit fallbacks (\d+)"),
                             ("device_fallbacks", r"device fallbacks (\d+)")):
                mm = re.search(pat, line)
                if mm and key not in facts:
                    facts[key] = mm.groups()
            if line.startswith("\u2605\u2605\u2605 THE BLOCK IS COMPRESSED UNDER WHIR"):
                facts["compressed"] = facts.get("compressed", 0) + 1
            if "TOP OVERLAP: the WHIR GLOBAL child runs as task 0 of level 0" in line:
                facts["top_overlap"] = True
            if phase_i < len(CLOSERS) and CLOSERS[phase_i][1].search(line):
                phase_i += 1
    return events, stages, facts, last_stamp


def build_windows(events, stages, span_end):
    wins = []
    prev_end = None
    first_start = None
    for p in PHASES:
        evs = events.get(p, [])
        if not evs:
            continue
        lo = min(a for a, _ in evs)
        hi = max(b for _, b in evs)
        start = lo if prev_end is None else prev_end
        if first_start is None:
            first_start = lo
        if hi > start:
            wins.append(Window(p, "phase", [(start, hi)]))
            prev_end = hi
    if first_start is not None and first_start > 0:
        wins.insert(0, Window("pre", "phase", [(0, first_start)]))
    if prev_end is not None and span_end > prev_end:
        wins.append(Window("post", "phase", [(prev_end, span_end)]))
    order = ["base/prep", "base/absorb", "base/commit", "base/prove", "base/global",
             "card/build_artifacts", "card/multi_prove", "lfm/prove"]
    names = order + sorted(k for k in stages if k not in order)
    for name in names:
        ivs = [(a, b) for a, b in stages.get(name, []) if b > a]
        if ivs:
            wins.append(Window(name, "stage", ivs))
    wins.append(Window("whole", "whole", [(0, span_end)]))
    return wins


def gpu_metrics(db, tbls, bin_ns):
    """{(typeId, metricId): {bin: (sum, count)}}, names, or (None, reason)."""
    if "GPU_METRICS" not in tbls:
        return None, None, "no GPU_METRICS table (counters were not collected)"
    cols = columns(db, "GPU_METRICS")
    ts = next((c for c in ("timestamp", "start") if c in cols), None)
    if ts is None or "metricId" not in cols or "value" not in cols:
        return None, None, "GPU_METRICS has unrecognised columns {}".format(cols)
    type_col = "typeId" if "typeId" in cols else "0"
    names = {}
    if "TARGET_INFO_GPU_METRICS" in tbls:
        icols = columns(db, "TARGET_INFO_GPU_METRICS")
        if "metricId" in icols and "metricName" in icols:
            tsel = "typeId" if "typeId" in icols else "0"
            tname = "typeName" if "typeName" in icols else "''"
            for t, m, n, tn in db.execute("SELECT DISTINCT {}, metricId, metricName, {} FROM TARGET_INFO_GPU_METRICS"
                                          .format(tsel, tname)):
                names[(t, m)] = (n, tn)
    q = ("SELECT {t}, metricId, CAST({ts} / {b} AS INTEGER) AS bin, SUM(value), COUNT(*) FROM GPU_METRICS "
         "WHERE {ts} IS NOT NULL GROUP BY 1, 2, 3").format(t=type_col, ts=ts, b=int(bin_ns))
    data = defaultdict(dict)
    for t, m, b, s, c in db.execute(q):
        data[(t, m)][b] = (s, c)
    if not data:
        return None, None, "GPU_METRICS is empty"
    for key in data:
        names.setdefault(key, ("metric {}".format(key[1]), ""))
    return data, names, None


def metric_label(names, key, many_types):
    n, tn = names[key]
    return "{} | {}".format(tn, n) if many_types and tn else n


def key_series(names):
    """key metric label -> [(typeId, metricId)] (one per GPU)."""
    out = defaultdict(list)
    for key in sorted(names):
        for label, pat in KEY_METRICS:
            if pat.search(names[key][0]):
                out[label].append(key)
                break
    return out


def nvtx_top(db, tbls, k=12):
    """(ranges, [(name, count, total_ns)]) of the NVTX ranges, or None when the trace has none."""
    if "NVTX_EVENTS" not in tbls:
        return None
    cols = columns(db, "NVTX_EVENTS")
    if "start" not in cols or "end" not in cols:
        return None
    name, join = ("e.text" if "text" in cols else "NULL"), ""
    if "textId" in cols and "StringIds" in tbls:
        name = "coalesce(e.text, s.value)" if "text" in cols else "s.value"
        join = "LEFT JOIN StringIds s ON s.id = e.textId"
    total = db.execute("SELECT count(*) FROM NVTX_EVENTS WHERE end > start").fetchone()[0]
    rows = db.execute("SELECT {n} AS name, count(*), sum(e.end - e.start) FROM NVTX_EVENTS e {j} "
                      "WHERE e.end > e.start GROUP BY name ORDER BY 3 DESC LIMIT {k}".format(n=name, j=join, k=int(k))).fetchall()
    return total, rows


def parse_nvsmi(path, t0_utc_ns):
    """[(trace ns, memory.used MiB, util %)] from `nvidia-smi --query-gpu=timestamp,memory.used,...` in UTC."""
    rows = []
    if not path or not os.path.exists(path):
        return rows
    with open(path, errors="replace") as f:
        for line in f:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                ts = datetime.datetime.strptime(parts[0], "%Y/%m/%d %H:%M:%S.%f")
                ts = ts.replace(tzinfo=datetime.timezone.utc)
                mem = float(parts[1])
            except ValueError:
                continue
            util = None
            if len(parts) > 2:
                try:
                    util = float(parts[2])
                except ValueError:
                    util = None
            rows.append((int(ts.timestamp() * NS) - t0_utc_ns, mem, util))
    return rows


def pct(a, b):
    return 100.0 * a / b if b else 0.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sqlite", required=True)
    ap.add_argument("--log", help="the run's stdout (its t= stamps give the stage windows)")
    ap.add_argument("--nvsmi", help="nvidia-smi sampler CSV (timestamp,memory.used,utilization.gpu,...; UTC)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bucket", type=float, default=5.0, help="busy/idle timeline bucket, seconds (default 5)")
    ap.add_argument("--metrics-bucket", type=float, default=1.0, help="GPU-metrics timeline bucket, seconds (default 1)")
    ap.add_argument("--top", type=int, default=5, help="top kernels listed per stage (default 5)")
    ap.add_argument("--quiet", action="store_true", help="write the files, print nothing")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    db = connect(a.sqlite)
    tbls = tables(db)

    kernels, copies = load_activity(db, tbls)
    span0, span_end = trace_span(db, tbls, kernels, copies)
    t0_utc = session_start_ns(db, tbls)
    kcover = Cover(merge((s, e) for s, e, _ in kernels))
    ccover = Cover(merge((s, e) for s, e, _, _ in copies))
    acover = Cover(merge([(s, e) for s, e, _ in kernels] + [(s, e) for s, e, _, _ in copies]))

    notes = []
    events, stages, facts, last_stamp = ({}, {}, {}, None)
    if a.log and t0_utc is not None and os.path.exists(a.log):
        events, stages, facts, last_stamp = parse_log(a.log, t0_utc)
    elif a.log and t0_utc is None:
        notes.append("no TARGET_INFO_SESSION_START_TIME: log stamps cannot be placed; whole-run tables only")
    wins = build_windows(events, stages, span_end)
    phase_wins = [w for w in wins if w.kind == "phase"]
    if not phase_wins:
        notes.append("no stage stamps found in the log: whole-run and bucket tables only")

    # clock check: the log's last stamp against the last GPU activity (both on the trace clock)
    last_gpu = max(kcover.ends[-1] if kcover.ends else 0, ccover.ends[-1] if ccover.ends else 0)
    clock = None
    if last_stamp is not None and last_gpu:
        clock = (last_stamp - last_gpu) / NS
        if abs(clock) > 5:
            notes.append("CLOCK SUSPECT: the log's last stamp is {:+.3f} s from the last GPU activity".format(clock))

    # ---- per-window kernel attribution (clipped time) ----
    per_win = [defaultdict(int) for _ in wins]
    launches = [0] * len(wins)
    for s, e, n in kernels:
        for i, w in enumerate(wins):
            got = w.overlap(s, e)
            if got:
                per_win[i][n] += got
                launches[i] += 1
    copy_kind = [defaultdict(int) for _ in wins]
    for s, e, k, _ in copies:
        for i, w in enumerate(wins):
            got = w.overlap(s, e)
            if got:
                copy_kind[i][k] += got

    # ---- GPU metrics and the nvidia-smi sampler ----
    mbin = int(0.01 * NS)
    mdata, mnames, mwhy = gpu_metrics(db, tbls, mbin)
    if mwhy:
        notes.append("GPU metrics: " + mwhy)
    many_types = bool(mnames) and len({k[0] for k in mnames}) > 1
    keys = key_series(mnames) if mnames else {}
    nvsmi = parse_nvsmi(a.nvsmi, t0_utc) if t0_utc is not None else []

    def metric_mean(key, win):
        s = c = 0
        for b, (bs, bc) in mdata[key].items():
            if win.contains(b * mbin + mbin // 2):
                s += bs
                c += bc
        return s / c if c else None

    def key_means(win):
        out = {}
        for label, ks in keys.items():
            vals = [v for v in (metric_mean(k, win) for k in ks) if v is not None]
            out[label] = sum(vals) / len(vals) if vals else None
        return out

    def vram_max(win):
        vals = [m for t, m, _ in nvsmi if win.contains(t)]
        return max(vals) if vals else None

    # ---- stages.csv ----
    key_labels = [k for k, _ in KEY_METRICS if k in keys]
    with open(os.path.join(a.out, "stages.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "kind", "intervals", "start_s", "end_s", "wall_s", "busy_any_pct", "busy_kernel_pct",
                    "busy_copy_pct", "idle_pct", "kernel_sum_s", "launches", "vram_max_mib"]
                   + ["{}_pct".format(k) for k in key_labels] + ["top_kernels"])
        for i, win in enumerate(wins):
            wall = win.wall()
            tot = sum(per_win[i].values())
            top = sorted(per_win[i].items(), key=lambda kv: -kv[1])[:a.top]
            km = key_means(win) if mdata else {}
            vm = vram_max(win)
            w.writerow([win.name, win.kind, len(win.ivs), "{:.3f}".format(win.first() / NS),
                        "{:.3f}".format(win.last() / NS), "{:.3f}".format(wall / NS),
                        "{:.1f}".format(pct(win.covered(acover), wall)), "{:.1f}".format(pct(win.covered(kcover), wall)),
                        "{:.1f}".format(pct(win.covered(ccover), wall)), "{:.1f}".format(100 - pct(win.covered(acover), wall)),
                        "{:.3f}".format(tot / NS), launches[i], "" if vm is None else "{:.0f}".format(vm)]
                       + ["" if km.get(k) is None else "{:.1f}".format(km[k]) for k in key_labels]
                       + [" ; ".join("{}:{:.3f}s:{:.1f}%".format(n, t / NS, pct(t, tot)) for n, t in top)])

    # ---- busy per bucket ----
    B = int(a.bucket * NS)
    nb = int(span_end // B) + 1
    bucket_k = [defaultdict(int) for _ in range(nb)]
    bucket_n = [0] * nb
    for s, e, n in kernels:
        b = int(s // B)
        while b < nb and b * B < e:
            got = min(e, (b + 1) * B) - max(s, b * B)
            if got > 0:
                bucket_k[b][n] += got
            if b == int(s // B):
                bucket_n[b] += 1
            b += 1
    busy_rows = []
    with open(os.path.join(a.out, "busy_{:g}s.csv".format(a.bucket)), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t0_s", "t1_s", "phase", "busy_any_pct", "busy_kernel_pct", "busy_copy_pct", "idle_pct",
                    "kernel_launches", "top_kernel", "top_kernel_pct_of_bucket", "vram_max_mib"]
                   + ["{}_pct".format(k) for k in key_labels])
        for b in range(nb):
            lo, hi = b * B, min((b + 1) * B, span_end)
            if hi <= lo:
                continue
            bw = Window("b", "b", [(lo, hi)])
            ph = next((pw.name for pw in phase_wins if pw.contains((lo + hi) // 2)), "")
            anyp = pct(acover.covered(lo, hi), hi - lo)
            top = max(bucket_k[b].items(), key=lambda kv: kv[1]) if bucket_k[b] else ("", 0)
            km = key_means(bw) if mdata else {}
            vm = vram_max(bw)
            row = [round(lo / NS, 1), round(hi / NS, 1), ph, round(anyp, 1),
                   round(pct(kcover.covered(lo, hi), hi - lo), 1), round(pct(ccover.covered(lo, hi), hi - lo), 1),
                   round(100 - anyp, 1), bucket_n[b], top[0], round(pct(top[1], hi - lo), 1),
                   "" if vm is None else round(vm)] + ["" if km.get(k) is None else round(km[k], 1) for k in key_labels]
            w.writerow(row)
            busy_rows.append(row)

    # ---- GPU metrics per bucket (key + all) and per stage (all) ----
    if mdata:
        MB = int(a.metrics_bucket * NS)
        per = int(round(MB / mbin))
        all_keys = sorted(mdata)
        labels = [metric_label(mnames, k, many_types) for k in all_keys]
        nmb = int(span_end // MB) + 1
        agg = {k: defaultdict(lambda: [0, 0]) for k in all_keys}
        for k in all_keys:
            for b, (s, c) in mdata[k].items():
                cell = agg[k][b // per]
                cell[0] += s
                cell[1] += c
        with open(os.path.join(a.out, "gpu_metrics_{:g}s.csv".format(a.metrics_bucket)), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t0_s", "phase"] + ["{}_pct".format(k) for k in key_labels] + ["dram_total_pct"])
            for mb in range(nmb):
                mid = mb * MB + MB // 2
                ph = next((pw.name for pw in phase_wins if pw.contains(mid)), "")
                vals = {}
                for label in key_labels:
                    vs = [agg[k][mb][0] / agg[k][mb][1] for k in keys[label] if agg[k][mb][1]]
                    vals[label] = sum(vs) / len(vs) if vs else None
                dram = None
                if vals.get("dram_read") is not None or vals.get("dram_write") is not None:
                    dram = (vals.get("dram_read") or 0) + (vals.get("dram_write") or 0)
                w.writerow([round(mb * a.metrics_bucket, 3), ph]
                           + ["" if vals[k] is None else round(vals[k], 1) for k in key_labels]
                           + ["" if dram is None else round(dram, 1)])
        with open(os.path.join(a.out, "gpu_metrics_all_{:g}s.csv".format(a.metrics_bucket)), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t0_s"] + labels)
            for mb in range(nmb):
                w.writerow([round(mb * a.metrics_bucket, 3)]
                           + ["" if not agg[k][mb][1] else round(agg[k][mb][0] / agg[k][mb][1], 2) for k in all_keys])
        with open(os.path.join(a.out, "gpu_metrics_stages.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["stage", "kind", "wall_s"] + labels)
            for win in wins:
                w.writerow([win.name, win.kind, round(win.wall() / NS, 3)]
                           + ["" if metric_mean(k, win) is None else round(metric_mean(k, win), 2) for k in all_keys])

    # ---- kernels, whole run, with the phase split ----
    ktot = defaultdict(lambda: [0, 0, 0])  # time, count, max
    for s, e, n in kernels:
        c = ktot[n]
        c[0] += e - s
        c[1] += 1
        c[2] = max(c[2], e - s)
    alltime = sum(c[0] for c in ktot.values())
    pnames = [w.name for w in phase_wins]
    with open(os.path.join(a.out, "kernels.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kernel", "launches", "total_s", "pct_of_kernel_time", "avg_ms", "max_ms"]
                   + ["s_in_{}".format(p) for p in pnames])
        for n, (t, c, mx) in sorted(ktot.items(), key=lambda kv: -kv[1][0]):
            w.writerow([n, c, round(t / NS, 3), round(pct(t, alltime), 2), round(t / c / 1e6, 4), round(mx / 1e6, 3)]
                       + [round(per_win[wins.index(pw)].get(n, 0) / NS, 3) for pw in phase_wins])

    # ---- summary.txt ----
    L = []
    L.append("== WHIR block run under Nsight Systems: summary (nsys_block_summary.py) ==")
    L.append("trace:   {}".format(os.path.abspath(a.sqlite)))
    if t0_utc is not None:
        L.append("session: starts {} UTC; GPU activity spans 0.0 - {:.1f} s".format(
            datetime.datetime.fromtimestamp(t0_utc / NS, datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            span_end / NS))
    L.append("gpu:     {}".format(gpu_description(db, tbls)))
    L.append("counts:  {} kernels, {} memcpy/memset; kernel time {:.1f} s (sum over launches)".format(
        len(kernels), len(copies), alltime / NS))
    if clock is not None:
        L.append("clock:   the log's last stamp sits {:+.3f} s from the last GPU activity (the stage windows are "
                 "only as good as this)".format(clock))
    nv = nvtx_top(db, tbls)
    if nv is None or nv[0] == 0:
        L.append("nvtx:    no NVTX ranges (a build without --features nvtx, or no libnvToolsExt reached)")
    else:
        L.append("nvtx:    {} ranges; by total time (nested ranges overlap): {}".format(nv[0], "; ".join(
            "{} x{} {:.1f} s".format(n or "?", c, t / NS) for n, c, t in nv[1][:8])))
    if facts:
        tr = facts.get("test_result", ("<none>",))[0]
        L.append("run:     test result: {} · compressed line x{} · commit fallbacks {} · device fallbacks {}".format(
            tr, facts.get("compressed", 0), facts.get("commit_fallbacks", ("?",))[0],
            facts.get("device_fallbacks", ("?",))[0]))
        if "whole_run" in facts:
            L.append("         WHOLE RUN {} s, host peak {} GiB (under tracing: never a block time)".format(
                facts["whole_run"][1], facts["whole_run"][0]))
        if "base_wall" in facts:
            L.append("         base: {} epochs in {} s{}".format(facts["base_wall"][0], facts["base_wall"][1],
                     " · level 0: {} wraps in {} s".format(*facts["level0_wall"]) if "level0_wall" in facts else ""))
    if facts.get("top_overlap"):
        notes.append("level0 includes the WHIR GLOBAL child: it runs as task 0 of level 0's pool (TOP OVERLAP default)")
    for n in notes:
        L.append("NOTE:    " + n)
    L.append("")
    L.append("STAGES (windows from the log's t= stamps; phases partition the run, stages are unions of the")
    L.append("harness's own intervals; busy = union of kernels+copies, so concurrency never counts twice)")
    hdr = "{:<22} {:>8} {:>8} {:>8} {:>6} {:>6} {:>6} {:>9}".format(
        "stage", "start_s", "end_s", "wall_s", "busy%", "kern%", "copy%", "kern_sum")
    if key_labels:
        hdr += "".join(" {:>9}".format(k[:9]) for k in key_labels[:4])
    L.append(hdr + "  top kernels (share of the stage's kernel time)")
    for i, win in enumerate(wins):
        wall = win.wall()
        if wall < NS // 1000:
            continue
        tot = sum(per_win[i].values())
        top = sorted(per_win[i].items(), key=lambda kv: -kv[1])[:3]
        row = "{:<22} {:>8.1f} {:>8.1f} {:>8.1f} {:>6.1f} {:>6.1f} {:>6.1f} {:>9.1f}".format(
            ("  " if win.kind == "stage" else "") + win.name, win.first() / NS, win.last() / NS, wall / NS,
            pct(win.covered(acover), wall), pct(win.covered(kcover), wall), pct(win.covered(ccover), wall), tot / NS)
        if key_labels:
            km = key_means(win)
            row += "".join(" {:>9}".format("-" if km.get(k) is None else "{:.1f}".format(km[k])) for k in key_labels[:4])
        L.append(row + "  " + ", ".join("{} {:.0f}%".format(n, pct(t, tot)) for n, t in top))
    L.append("")
    L.append("BUSY/IDLE per {:g} s (kernel + memcpy + memset union; '#' = 5% busy)".format(a.bucket))
    for r in busy_rows:
        bar = "#" * int(round(r[3] / 5.0))
        L.append("  {:>6.1f}-{:<6.1f} {:<9} busy {:>5.1f}%  {:<20} top {} {:.0f}%{}".format(
            r[0], r[1], r[2], r[3], bar, r[8] or "-", r[9],
            "" if r[10] == "" else "  vram {} MiB".format(r[10])))
    L.append("")
    L.append("TOP KERNELS, whole run (kernel time; phase split in kernels.csv)")
    for n, (t, c, mx) in sorted(ktot.items(), key=lambda kv: -kv[1][0])[:15]:
        L.append("  {:<44} {:>8.2f} s {:>5.1f}%  x{:<7} avg {:>8.3f} ms  max {:>8.2f} ms".format(
            n, t / NS, pct(t, alltime), c, t / c / 1e6, mx / 1e6))
    L.append("")
    L.append("COPIES per phase (seconds of copy-engine time by kind)")
    for i, win in enumerate(wins):
        if win.kind != "phase" or not copy_kind[i]:
            continue
        L.append("  {:<10} ".format(win.name) + " · ".join(
            "{} {:.2f}".format(k, t / NS) for k, t in sorted(copy_kind[i].items(), key=lambda kv: -kv[1])))
    if mdata:
        L.append("")
        L.append("GPU METRICS present: {} series; key series matched: {}".format(
            len(mdata), ", ".join("{} <- {}".format(k, "; ".join(mnames[x][0] for x in keys[k])) for k in key_labels)))
        unmatched = [k for k, _ in KEY_METRICS if k not in keys]
        if unmatched:
            L.append("  (no series matched: {} — see gpu_metrics_all_*.csv for every name)".format(", ".join(unmatched)))
    L.append("")
    L.append("FILES: stages.csv · busy_{:g}s.csv · kernels.csv{}".format(
        a.bucket, " · gpu_metrics_{0:g}s.csv · gpu_metrics_all_{0:g}s.csv · gpu_metrics_stages.csv".format(
            a.metrics_bucket) if mdata else ""))
    text = "\n".join(L) + "\n"
    with open(os.path.join(a.out, "summary.txt"), "w") as f:
        f.write(text)
    if not a.quiet:
        sys.stdout.write(text)
    return 0


def selftest():
    """A synthetic trace + log whose answers are known by hand; exercises every table,
    the GPU-metrics path included (in the GPU_METRICS / TARGET_INFO_GPU_METRICS layout
    nsys 2025 exports — a real trace with metrics is the test that settles the schema)."""
    import tempfile
    d = tempfile.mkdtemp(prefix="nsys_summary_selftest.")
    db = sqlite3.connect(os.path.join(d, "t.sqlite"))
    t0 = 1000 * NS
    db.executescript("""
        CREATE TABLE TARGET_INFO_SESSION_START_TIME (utcEpochNs INTEGER, utcTime TEXT, localTime TEXT);
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, shortName INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INTEGER, end INTEGER, copyKind INTEGER, bytes INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET (start INTEGER, end INTEGER, bytes INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER, end INTEGER, correlationId INTEGER);
        CREATE TABLE ENUM_CUDA_MEMCPY_OPER (id INTEGER, name TEXT, label TEXT);
        CREATE TABLE TARGET_INFO_GPU (name TEXT, smCount INTEGER, totalMemory INTEGER, chipName TEXT);
        CREATE TABLE GPU_METRICS (rawTimestamp INTEGER, timestamp INTEGER, typeId INTEGER, metricId INTEGER, value INTEGER);
        CREATE TABLE TARGET_INFO_GPU_METRICS (typeId INTEGER, sourceId INTEGER, typeName TEXT, metricId INTEGER, metricName TEXT);
        CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, eventType INTEGER, text TEXT, textId INTEGER);
    """)
    s = lambda x: int(x * NS)  # noqa: E731
    db.execute("INSERT INTO TARGET_INFO_SESSION_START_TIME VALUES (?, '', '')", (t0,))
    db.executemany("INSERT INTO StringIds VALUES (?, ?)", [(1, "ka"), (2, "kb"), (3, "kc"), (9, "epoch_prove")])
    # two ranges named inline, one through the string table, one point mark (end NULL): not a range
    db.executemany("INSERT INTO NVTX_EVENTS VALUES (?, ?, 59, ?, ?)", [
        (s(1.0), s(5.0), "prove_continuation_total", None), (s(2.0), s(4.0), None, 9),
        (s(4.0), s(5.0), None, 9), (s(3.0), None, "a mark", None)])
    db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?)", [
        (s(1.5), s(2.0), 1), (s(2.0), s(3.0), 2), (s(3.5), s(4.5), 1),  # base: prep, commit, prove
        (s(5.0), s(6.0), 3), (s(5.5), s(6.5), 3),                       # level 0, overlapping
        (s(7.2), s(7.4), 2),                                            # interior
        (s(8.0), s(9.0), 1),                                            # root
        (s(6.9), s(7.1), 3),   # straddles level0 | interior: 0.1 s each (both clips)
        (s(9.8), s(10.2), 1)])  # straddles root | post and the 10 s bucket edge
    db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?, ?, 1, 4096)", (s(4.4), s(4.8)))
    db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, 1)", (s(0.1), s(10.5)))
    db.execute("INSERT INTO ENUM_CUDA_MEMCPY_OPER VALUES (1, 'HTOD', 'Host-to-Device')")
    db.execute("INSERT INTO TARGET_INFO_GPU VALUES ('Synthetic GPU', 10, ?, 'SYN')", (8 * 2**30,))
    db.execute("INSERT INTO TARGET_INFO_GPU_METRICS VALUES (7, 0, 'syn', 1, 'SMs Active [Throughput %]')")
    db.execute("INSERT INTO TARGET_INFO_GPU_METRICS VALUES (7, 0, 'syn', 2, 'DRAM Read Bandwidth [Throughput %]')")
    rows = []
    for i in range(0, 1000):  # 1 kHz over 1-10 s: SM 80 % in the base, 20 % after; DRAM 10 %
        t = s(1.0) + i * s(0.009)
        rows.append((t, t, 7, 1, 80 if t < s(5.0) else 20))
        rows.append((t, t, 7, 2, 10))
    db.executemany("INSERT INTO GPU_METRICS VALUES (?, ?, ?, ?, ?)", rows)
    db.commit()
    db.close()
    log = os.path.join(d, "run.log")
    with open(log, "w") as f:
        f.write("\n".join([
            "BASE EPOCH 0: prep 1.00s t=[1001.000,1002.000]",
            "BASE EPOCH 0: commit 1.00s t=[1002.000,1003.000]",
            "BASE EPOCH 0: prove 2.00s t=[1003.000,1005.000]",
            "   base (WHIR): 1 epochs in 4.0s",
            "   MARK AFTER the WHIR base (x): live Some(1.0) GiB / t=1005.0",
            "CARD HOLD #0 multi_prove: waited 0.000s \u00b7 held 2.000s \u00b7 t=[1005.000,1007.000]",
            "   level 0: 1 WHIR wraps in 2.0s",
            "PROVE SPLIT #1: airs 11 \u00b7 wall 1.00s \u00b7 t=[1007.000,1008.000] \u00b7 prepass 0.00",
            "\u2605\u2605\u2605 WHIR INTERIOR COMPOSED \u2014 1 proof(s) at level 1",
            "PROVE SPLIT #2: airs 11 \u00b7 wall 2.00s \u00b7 t=[1008.000,1010.000] \u00b7 prepass 0.00",
            "   host peak 1.0 GiB at t=1003.0 (never an event: a peak instant)",
            "\u2605\u2605\u2605 THE BLOCK IS COMPRESSED UNDER WHIR \u2014 the block-artifact ROOT PROVED AND VERIFIED",
            "   \u2605\u2605\u2605 THE BLOCK IS COMPRESSED UNDER WHIR (an echo, not counted)",
            "\u2605\u2605\u2605 WHOLE RUN: host peak 1.000 GiB at t=1003.0, 10.0s total",
            "   commit fallbacks 0",
            "   device fallbacks 0",
            "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 9 filtered out; finished in 10.0s",
        ]) + "\n")
    nvsmi = os.path.join(d, "nvsmi.csv")
    with open(nvsmi, "w") as f:
        for i in range(0, 110):  # 10 Hz, 1000.0-1011.0 s UTC; 3000 MiB in the base, 5000 after
            t = datetime.datetime.fromtimestamp(1000 + i * 0.1, datetime.timezone.utc)
            f.write("{}, {}, 50, 2000, 300.0\n".format(t.strftime("%Y/%m/%d %H:%M:%S.%f")[:-3],
                                                         3000 if i < 50 else 5000))
    out = os.path.join(d, "out")
    main(["--sqlite", os.path.join(d, "t.sqlite"), "--log", log, "--nvsmi", nvsmi, "--out", out, "--quiet"])
    with open(os.path.join(out, "stages.csv")) as f:
        st = {r["stage"]: r for r in csv.DictReader(f)}
    with open(os.path.join(out, "busy_5s.csv")) as f:
        bz = list(csv.DictReader(f))
    with open(os.path.join(out, "summary.txt")) as f:
        summ = f.read()
    checks = [
        ("phases", sorted(k for k, r in st.items() if r["kind"] == "phase"),
         sorted(["pre", "base", "level0", "interior", "root", "post"])),
        ("base window", (st["base"]["start_s"], st["base"]["end_s"]), ("1.000", "5.000")),
        ("base busy_any (2.8 of 4 s)", st["base"]["busy_any_pct"], "70.0"),
        ("base busy_kernel (2.5 of 4 s)", st["base"]["busy_kernel_pct"], "62.5"),
        ("base busy_copy (0.4 of 4 s)", st["base"]["busy_copy_pct"], "10.0"),
        ("level0 busy (1.5 + 0.1 clipped, of 2 s)", st["level0"]["busy_any_pct"], "80.0"),
        ("level0 kernel_sum (2.1 s: overlap counted in the sum only)", st["level0"]["kernel_sum_s"], "2.100"),
        ("interior busy (0.1 clipped + 0.2, of 1 s)", st["interior"]["busy_any_pct"], "30.0"),
        ("root busy (1.0 + 0.2 clipped, of 2 s)", st["root"]["busy_any_pct"], "60.0"),
        ("post window", (st["post"]["start_s"], st["post"]["end_s"]), ("10.000", "10.500")),
        ("post busy (0.2 clipped, of 0.5 s)", st["post"]["busy_any_pct"], "40.0"),
        ("base/prove busy (1.3 of 2 s)", st["base/prove"]["busy_any_pct"], "65.0"),
        ("base sm_active", st["base"]["sm_active_pct"], "80.0"),
        ("level0 sm_active", st["level0"]["sm_active_pct"], "20.0"),
        ("base dram_read", st["base"]["dram_read_pct"], "10.0"),
        ("base vram max", st["base"]["vram_max_mib"], "3000"),
        ("root vram max", st["root"]["vram_max_mib"], "5000"),
        ("bucket 0-5 busy (2.8 s)", bz[0]["busy_any_pct"], "56.0"),
        ("bucket 5-10 busy (3.1 s)", bz[1]["busy_any_pct"], "62.0"),
        ("bucket 10-10.5 busy (0.2 s)", bz[2]["busy_any_pct"], "40.0"),
        ("compressed counted once", "compressed line x1" in summ, True),
        ("clock check", "-0.200 s from the last GPU activity" in summ, True),
        ("nvtx: 3 ranges, the string-table name resolved", "nvtx:    3 ranges; by total time (nested ranges overlap): "
         "prove_continuation_total x1 4.0 s; epoch_prove x2 3.0 s" in summ, True),
    ]
    bad = [(n, got, want) for n, got, want in checks if got != want]
    for n, got, want in checks:
        print("{} {:<58} got {!r}".format("ok  " if got == want else "FAIL", n, got))
    for fn in ("gpu_metrics_1s.csv", "gpu_metrics_all_1s.csv", "gpu_metrics_stages.csv", "kernels.csv"):
        print("{} {:<58} {}".format("ok  " if os.path.exists(os.path.join(out, fn)) else "FAIL",
                                    "wrote " + fn, ""))
        if not os.path.exists(os.path.join(out, fn)):
            bad.append((fn, None, "exists"))
    print("SELFTEST {} ({} checks, {} failed) in {}".format("PASS" if not bad else "FAIL", len(checks) + 4, len(bad), d))
    return 0 if not bad else 1


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        sys.exit(selftest())
    sys.exit(main())
