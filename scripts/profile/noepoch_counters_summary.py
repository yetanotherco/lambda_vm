#!/usr/bin/env python3
"""noepoch_counters_summary.py: the text side of noepoch_counters.sh (lane I-PROF, 2026-09-30).

Standard library only (python >= 3.8; `runa` and `dry` also need the sqlite3 module).
noepoch_counters.sh carries a byte-identical copy of this file and writes it into its
work directory; the copy next to the script is the one to read and edit. It is a fork of
mauro_ncu_summary.py (lane G5-NCU, 2026-09-28): the ncu parsing, the scrub and the
self-check are that file's; the stages, the CUDA API tables and the time series are new.

    runa           --sqlite S --log L --workload W --out O [--smi F] [--bin-ms N]
                                                   run A: per stage (head, prepass, main_commit, between,
                                                   fused, tail; recommit as an overlapping row) the card's busy
                                                   time, GPU metrics, VRAM and the kernels; the kernels each NVTX
                                                   label launched; CUDA API time per category, thread and stage;
                                                   a time series; from an nsys trace and the harness log
    summary        --plan P --ncu-dir D --out O    run B: one row per kernel, and epoch against no-epoch
    stages         --log L                         run B: the stage of each profiled launch, from one pass's output
    dry            --plan P --workload W --sqlite S --log L --out O
                                                   what each ncu pass would profile, against run A's trace
    report         --send D                        SUMMARY.md: the text summary of a run, from the files above
    plan-check     --plan P                        the pass plan is well formed
    cargo-artifact (--exe NAME | --outdir PKG)     one path from cargo's JSON messages on stdin
    scrub          --dir D                         literal replacements (stdin: OLD<TAB>NEW lines), and the
                                                   "Host Name" column dropped from ncu CSVs
    check          --dir D --report R              the bundle self-check (stdin: LABEL<TAB>VALUE records,
                                                   NUL-separated): plain text only, no Nsight report or
                                                   database, no credential marker, none of the values
    selftest                                       every subcommand on synthetic inputs

Stages (run A). The prover's instruments spans are NVTX ranges in a --features nvtx build:
r1_prepass, r1_main_commit and rounds_2to4 once per multi_prove on the calling thread, and per
table on the driver threads r1_main_recommit_table (the no-epoch device recommit), r1_aux_build_table,
r1_aux_commit_table and rounds_2to4_table. The disjoint stages are: head = the trace start to the
first prepass; prepass, main_commit, fused = the unions of those ranges; between = the rest of
[first prepass, last fused end] (the absorb, and in the epoch base the waits between epochs);
tail = after the last fused range (the no-epoch arm's verify). recommit = the union of the
recommit ranges, a part of fused, reported as its own row. Without NVTX ranges (no libnvToolsExt)
the stages come from the log's PROVE SPLIT lines (t=[..] plus the phase walls, the `other`
remainder unplaced) and there is no recommit row.

What `summary` reports, per profiled launch and per kernel (duration-weighted over its launches):
  time      ncu's Duration, at the clock ncu held (launches.tsv's sm_ghz; env.txt names the
            --clock-control mode). Under `base` an RTX 5090 runs its SMs near 2.0 GHz where the block
            runs near 2.76 GHz, so a compute-bound kernel's duration here is ~1.4x its in-block time.
            The percentages are against the peak at the clock ncu ran.
  DRAM %    dram throughput, % of peak (Speed Of Light)
  SM %      Compute (SM) throughput, % of peak (Speed Of Light)
  L2 %      L2 throughput, % of peak (Speed Of Light); L2 hit % from Memory Workload Analysis
  occ %     achieved occupancy (theoretical beside it, and the block limit that sets it)
  B/elem    DRAM bytes (read + write) per element: a field element for the column-major NTT (16 per
            thread), a leaf for the leaf kernels, a parent for the Merkle levels (a half-warp per
            parent in rpx_merkle_level_warp), a thread for every other kernel
  roof      "DRAM", "compute" or "L2" roof when its % of peak is >= 60, "mid" between 30 and 60,
            "below" when all three are under 30 (latency, occupancy or launch bound)

ncu's --csv prints base units (nsecond, byte, byte/second, cycle/second) and, in some versions and
columns, thousands separators; every value goes through num() and a unit table, and a value that
does not parse is left empty rather than guessed.
"""
import argparse
import bisect
import csv
import io
import json
import os
import re
import sys
import tempfile
from datetime import datetime

PLAN_COLS = ["pass", "workload", "mode", "skip", "count", "kernels", "family", "note"]
WORKLOADS = ("epoch", "noepoch")
MODES = ("window", "config")

# ---------------------------------------------------------------------------------------------
# small helpers


def num(v):
    """ncu's printed number -> float. Thousands separators ("4,194,304") are dropped; anything
    else that is not a number (n/a, empty) is None."""
    if v is None:
        return None
    v = v.strip()
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", v):
        v = v.replace(",", "")
    try:
        return float(v)
    except ValueError:
        return None


TIME_US = {"ns": 1e-3, "nsecond": 1e-3, "nseconds": 1e-3, "usecond": 1.0, "useconds": 1.0, "us": 1.0,
           "msecond": 1e3, "mseconds": 1e3, "ms": 1e3, "second": 1e6, "seconds": 1e6, "s": 1e6}
BYTE_PREFIX = {"": 1.0, "K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12, "Ki": 1024.0, "Mi": 1024.0 ** 2,
               "Gi": 1024.0 ** 3, "Ti": 1024.0 ** 4}
GHZ = {"hz": 1e-9, "khz": 1e-6, "mhz": 1e-3, "ghz": 1.0, "cycle/second": 1e-9, "cycle/s": 1e-9,
       "cycle/msecond": 1e-6, "cycle/ms": 1e-6, "cycle/usecond": 1e-3, "cycle/us": 1e-3,
       "cycle/nsecond": 1.0, "cycle/ns": 1.0}


def to_us(value, unit):
    f = TIME_US.get((unit or "").strip().lower())
    return None if value is None or f is None else value * f


def to_bytes(value, unit):
    """byte, Kbyte, Mbyte, ... (ncu's prefixes are decimal: a 16,384-byte carveout prints 16.38 Kbyte)."""
    m = re.fullmatch(r"([KMGT]i?)?(bytes?|B)", (unit or "").strip())
    if value is None or not m:
        return None
    return value * BYTE_PREFIX[m.group(1) or ""]


def to_bytes_per_s(value, unit):
    u = (unit or "").strip()
    m = re.fullmatch(r"(.+?)/(s|second)", u)
    return None if not m else to_bytes(value, m.group(1))


def to_ghz(value, unit):
    f = GHZ.get((unit or "").strip().lower())
    return None if value is None or f is None else value * f


def dims(s):
    """"(128, 1, 1)" or "128, 1, 1" -> (128, 1, 1); anything else -> None."""
    m = re.fullmatch(r"\(?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)?", (s or "").strip())
    return tuple(int(x) for x in m.groups()) if m else None


def prod(t):
    p = 1
    for x in t:
        p *= x
    return p


def fmt(x, nd=1):
    if x is None:
        return "-"
    if isinstance(x, float) and abs(x) >= 1e5:
        return f"{x:.3g}"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def shape(grid, block):
    g = "x".join(str(v) for v in grid) if grid else "?"
    b = "x".join(str(v) for v in block) if block else "?"
    return f"{g}/{b}"


def weighted(items, key):
    """Duration-weighted mean of key over launches that have both."""
    num_, den = 0.0, 0.0
    for it in items:
        v, d = it.get(key), it.get("dur_us")
        if v is not None and d:
            num_ += v * d
            den += d
    return num_ / den if den else None


def write_tsv(path, rows):
    with open(path, "w", newline="") as f:
        csv.writer(f, delimiter="\t", lineterminator="\n").writerows(rows)


def read_tsv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


# ---------------------------------------------------------------------------------------------
# the plan


def read_plan(path):
    rows = []
    with open(path, newline="") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if parts[0] == "pass":
                if parts != PLAN_COLS:
                    raise SystemExit(f"plan {path}: header must be {'<TAB>'.join(PLAN_COLS)}")
                continue
            if len(parts) != len(PLAN_COLS):
                raise SystemExit(f"plan {path}:{ln}: {len(parts)} fields, want {len(PLAN_COLS)}")
            r = dict(zip(PLAN_COLS, parts))
            r["line"] = ln
            rows.append(r)
    return rows


def check_plan(rows):
    errs, seen = [], set()
    for r in rows:
        where = f"line {r['line']} ({r['pass']})"
        if not re.fullmatch(r"[a-z0-9_]+", r["pass"]):
            errs.append(f"{where}: pass name must be [a-z0-9_]+")
        if r["pass"] in seen:
            errs.append(f"{where}: duplicate pass name")
        seen.add(r["pass"])
        if r["workload"] not in WORKLOADS:
            errs.append(f"{where}: workload must be one of {WORKLOADS}")
        if r["mode"] not in MODES:
            errs.append(f"{where}: mode must be one of {MODES}")
        if not r["skip"].isdigit():
            errs.append(f"{where}: skip must be an integer >= 0")
        if not r["count"].isdigit() or int(r["count"]) < 1:
            errs.append(f"{where}: count must be an integer >= 1")
        if any(c in r["kernels"] for c in "^$\t "):
            errs.append(f"{where}: kernels is the regex BODY (anchored as ^(...)$ by the tools), no ^ $ or blanks")
        try:
            re.compile(r["kernels"])
        except re.error as e:
            errs.append(f"{where}: kernels does not compile: {e}")
        # ncu's per-launch-config key is grid, block and shared memory, not the kernel: a config
        # pass over two kernels would skip the second one's launches whose shape the first used.
        if r["mode"] == "config" and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", r["kernels"]):
            errs.append(f"{where}: a config pass names exactly one kernel (ncu's per-launch-config key "
                        "does not include the kernel name)")
        for c in ("family", "note"):
            if not r[c].strip():
                errs.append(f"{where}: {c} is empty")
    return errs


def kernel_re(row):
    """The pass's kernel regex; callers fullmatch it, as ncu is handed ^(...)$."""
    return re.compile("(?:" + row["kernels"] + ")")


def matches(row, name):
    return kernel_re(row).fullmatch(name) is not None


# ---------------------------------------------------------------------------------------------
# summary: ncu details CSV -> one row per launch, one row per kernel

# (kernel regex, element name, elements per thread); first match wins, default one thread
ELEMENTS = [
    (r"ntt_cm_di[ft]_k[4-8]", "felt", 16.0),
    (r"rpx_merkle_level_warp", "parent", 1.0 / 16.0),
    (r"rpx_merkle_level", "parent", 1.0),
    (r"rpx_(leaves_[a-z0-9_]+|comp_poly_leaves_ext3|fri_leaves_ext3|fri_group_leaves_ext3)", "leaf", 1.0),
]

FAMILIES = [  # the summary's family of a kernel, by name
    (r"ntt_[a-z0-9_]+|mobius_[a-z_]+|lift_spread|matrix_transpose_strided|bit_reverse_(permute|row_major)[a-z_]*"
     r"|pointwise_mul[a-z_]*|scalar_mul[a-z_]*", "lde"),
    (r"rpx_merkle_[a-z_]+", "merkle"),
    (r"rpx_grind_[a-z_]+", "grind"),
    (r"rpx_[a-z0-9_]*leaves[a-z0-9_]*", "leaves"),
    (r"sumcheck_[a-z0-9_]+|program_map_ext3|eq_[a-z_]+|fraction_fold[a-z_]*|mle_[a-z0-9_]+|factors_from_columns_ext3"
     r"|sum_partials_ext3|add_scaled_ext3|fill_ext3", "sumcheck"),
    (r"constraint_[a-z_]+|ccomp_[0-9a-f]+|comp_h_to_slabs_ext3|decompose_d2_ext3", "quotient"),
    (r"logup_[a-z0-9_]+", "logup"),
    (r"deep_[a-z0-9_]+|bit_reverse_ext3_interleaved|invert_[a-z0-9_]+|compute_denoms_ext3|batch_inverse_[a-z0-9_]+"
     r"|block_inclusive_scan_[a-z0-9_]+|apply_block_offsets_[a-z0-9_]+|barycentric_[a-z0-9_]+|gather_rows_[a-z0-9_]+",
     "deep"),
    (r"fri_[a-z0-9_]+|gather_ext3_at|merkle_gather_paths", "fri"),
    (r"whir_[a-z0-9_]+|gather_cosets", "whir-fold"),
]

STALL_RE = re.compile(r"smsp__average_warps?_issue_stalled_(\w+?)_per_issue_active\.ratio")
PIPE_RE = re.compile(r"sm__inst_executed_pipe_(\w+?)\.avg\.pct_of_peak_sustained_active")
PIPE_CYC_RE = re.compile(r"sm__pipe_(\w+?)_cycles_active\.avg\.pct_of_peak_sustained_active")


def element_of(kernel):
    for pat, name, per in ELEMENTS:
        if re.fullmatch(pat, kernel):
            return name, per
    return "thread", 1.0


def family_of(kernel):
    for pat, fam in FAMILIES:
        if re.fullmatch(pat, kernel):
            return fam
    return "other"


def read_details(path):
    """{id: {"kernel", "grid", "block", "m": {(section, metric): (unit, value)}}}, in ID order."""
    with open(path, newline="", errors="replace") as f:
        lines = [ln for ln in f if not ln.startswith("==")]
    rdr = csv.reader(lines)
    hdr = None
    launches = {}
    for row in rdr:
        if hdr is None:
            if "ID" in row and "Metric Name" in row and "Kernel Name" in row:
                hdr = {c: i for i, c in enumerate(row)}
            continue
        if len(row) < len(hdr) - 6:  # rule columns may be missing on short rows
            continue

        def col(c):
            i = hdr.get(c)
            return row[i] if i is not None and i < len(row) else ""

        lid = col("ID")
        if not lid.isdigit():
            continue
        lid = int(lid)
        rec = launches.setdefault(lid, {"kernel": col("Kernel Name"), "grid": dims(col("Grid Size")),
                                        "block": dims(col("Block Size")), "m": {}})
        name = col("Metric Name")
        if name:
            rec["m"][(col("Section Name"), name)] = (col("Metric Unit"), col("Metric Value"))
    return [dict(v, id=k) for k, v in sorted(launches.items())]


def metric(rec, names):
    """First present (section, metric) pair of `names` -> (value, unit); section None = any."""
    for sec, name in names:
        for (s, n), (u, v) in rec["m"].items():
            if n == name and (sec is None or s == sec):
                x = num(v)
                if x is not None:
                    return x, u
    return None, None


SOL = "GPU Speed Of Light Throughput"
MWA = "Memory Workload Analysis"
CWA = "Compute Workload Analysis"
OCC = "Occupancy"
LST = "Launch Statistics"


def launch_row(rec):
    kernel = rec["kernel"]
    out = {"id": rec["id"], "kernel": kernel, "grid": rec["grid"], "block": rec["block"]}
    grid, block = rec["grid"], rec["block"]
    threads = prod(grid) * prod(block) if grid and block else None
    if threads is None:
        t, _ = metric(rec, [(LST, "Threads")])
        threads = int(t) if t else None
    out["threads"] = threads
    v, u = metric(rec, [(SOL, "Duration"), (None, "gpu__time_duration.sum")])
    out["dur_us"] = to_us(v, u)
    v, u = metric(rec, [(SOL, "SM Frequency")])
    out["sm_ghz"] = to_ghz(v, u)
    out["dram_pct"], _ = metric(rec, [(SOL, "DRAM Throughput"),
                                      (None, "dram__throughput.avg.pct_of_peak_sustained_elapsed")])
    out["sm_pct"], _ = metric(rec, [(SOL, "Compute (SM) Throughput"),
                                    (None, "sm__throughput.avg.pct_of_peak_sustained_elapsed")])
    out["mem_pct"], _ = metric(rec, [(SOL, "Memory Throughput"),
                                     (None, "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed")])
    out["l2_pct"], _ = metric(rec, [(SOL, "L2 Cache Throughput"),
                                    (None, "lts__throughput.avg.pct_of_peak_sustained_elapsed")])
    out["l1_pct"], _ = metric(rec, [(SOL, "L1/TEX Cache Throughput")])
    out["l2_hit"], _ = metric(rec, [(MWA, "L2 Hit Rate"), (None, "lts__t_sector_hit_rate.pct")])
    out["occ"], _ = metric(rec, [(OCC, "Achieved Occupancy"),
                                 (None, "sm__warps_active.avg.pct_of_peak_sustained_active")])
    out["occ_theo"], _ = metric(rec, [(OCC, "Theoretical Occupancy")])
    out["regs"], _ = metric(rec, [(LST, "Registers Per Thread"), (None, "launch__registers_per_thread")])
    out["waves"], _ = metric(rec, [(LST, "Waves Per SM")])
    out["issue_pct"], _ = metric(rec, [(CWA, "Issue Slots Busy"),
                                       (None, "smsp__issue_active.avg.pct_of_peak_sustained_active")])
    out["ipc"], _ = metric(rec, [(CWA, "Executed Ipc Active")])
    limits = {}
    for lim in ("Registers", "Shared Mem", "Warps", "SM", "Barriers"):
        x, _ = metric(rec, [(OCC, "Block Limit " + lim)])
        if x is not None:
            limits[lim] = x
    out["occ_limit"] = min(limits, key=lambda k: limits[k]) if limits else None
    # DRAM bytes: the explicit counters when collected, else the throughput x the duration
    r, ru = metric(rec, [(None, "dram__bytes_read.sum")])
    w, wu = metric(rec, [(None, "dram__bytes_write.sum")])
    rb, wb = to_bytes(r, ru), to_bytes(w, wu)
    if rb is not None and wb is not None:
        out["dram_bytes"], out["bytes_src"] = rb + wb, "counters"
    else:
        v, u = metric(rec, [(MWA, "Memory Throughput")])
        bps = to_bytes_per_s(v, u)
        if bps is not None and out["dur_us"] is not None:
            out["dram_bytes"], out["bytes_src"] = bps * out["dur_us"] * 1e-6, "throughput x duration"
        else:
            out["dram_bytes"], out["bytes_src"] = None, None
    elem, per = element_of(kernel)
    out["elem"] = elem
    out["elems"] = threads * per if threads else None
    out["b_per_elem"] = (out["dram_bytes"] / out["elems"]
                         if out["dram_bytes"] is not None and out["elems"] else None)
    stalls, pipes, cycles = {}, {}, {}
    for (_s, n), (_u, v) in rec["m"].items():
        x = num(v)
        if x is None:
            continue
        for rx, into in ((STALL_RE, stalls), (PIPE_RE, pipes), (PIPE_CYC_RE, cycles)):
            m = rx.fullmatch(n)
            if m:
                into[m.group(1)] = x
    tot = sum(stalls.values())
    out["stalls"] = ", ".join(f"{k} {100 * v / tot:.0f}%" for k, v in
                              sorted(stalls.items(), key=lambda kv: -kv[1])[:3]) if tot > 0 else ""
    out["pipes"] = ", ".join(f"{k} {v:.0f}%" for k, v in sorted(pipes.items(), key=lambda kv: -kv[1])[:2])
    out["pipe_cycles"] = ", ".join(f"{k} {v:.0f}%" for k, v in sorted(cycles.items(), key=lambda kv: -kv[1])[:2])
    out["roof"] = roof(out["dram_pct"], out["sm_pct"], out["l2_pct"])
    out["bound"] = bound_of(out["dram_pct"], out["sm_pct"], out["l2_pct"])
    return out


def roof(dram, sm, l2):
    cand = [(v, n) for v, n in ((dram, "DRAM"), (sm, "compute"), (l2, "L2")) if v is not None]
    if not cand:
        return "?"
    top, name = max(cand)
    if top >= 60:
        return f"{name} roof {top:.0f}%"
    if top >= 30:
        return f"mid ({name} {top:.0f}%)"
    return f"below ({name} {top:.0f}%)"


def bound_of(dram, sm, l2):
    """One word: compute, memory (DRAM or L2) or latency (every roof under 30 %)."""
    cand = [(v, n) for v, n in ((dram, "memory"), (sm, "compute"), (l2, "memory")) if v is not None]
    if not cand:
        return "?"
    top, name = max(cand)
    return name if top >= 30 else "latency"


def read_stages(path):
    st = {}
    if path and os.path.exists(path):
        with open(path) as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if row.get("id", "").isdigit():
                    st[int(row["id"])] = row.get("stage", "?")
    return st


LAUNCH_COLS = ["workload", "pass", "family", "stage", "id", "kernel", "shape", "threads", "regs", "waves",
               "dur_us", "sm_ghz", "dram_pct", "sm_pct", "l2_pct", "mem_pct", "l1_pct", "l2_hit", "occ",
               "occ_theo", "occ_limit", "issue_pct", "ipc", "dram_bytes", "bytes_src", "elem", "elems",
               "b_per_elem", "roof", "bound", "stalls", "pipes", "pipe_cycles"]


def cell(v):
    return fmt(v, 2) if isinstance(v, float) else ("" if v is None else v)


def cmd_summary(a):
    plan = {r["pass"]: r for r in read_plan(a.plan)} if a.plan else {}
    launches = []
    names = sorted(f[:-len(".details.csv")] for f in os.listdir(a.ncu_dir) if f.endswith(".details.csv"))
    for p in names:
        recs = read_details(os.path.join(a.ncu_dir, p + ".details.csv"))
        stages = read_stages(os.path.join(a.ncu_dir, p + ".stages.tsv"))
        pr = plan.get(p, {})
        for rec in recs:
            row = launch_row(rec)
            row["pass"] = p
            row["workload"] = pr.get("workload", "?")
            row["family"] = family_of(row["kernel"])
            row["stage"] = stages.get(rec["id"], "?")
            row["shape"] = shape(row["grid"], row["block"])
            launches.append(row)
    os.makedirs(a.out, exist_ok=True)
    write_tsv(os.path.join(a.out, "launches.tsv"), [LAUNCH_COLS] + [[cell(r.get(c)) for c in LAUNCH_COLS]
                                                                     for r in launches])
    kern = aggregate(launches, ("workload", "kernel"))
    kstage = aggregate(launches, ("workload", "kernel", "stage"))
    fam = aggregate(launches, ("workload", "stage", "family"))
    md = render(kern, kstage, fam, launches, names)
    with open(os.path.join(a.out, "kernels.md"), "w") as f:
        f.write(md)
    cols = ["workload", "kernel", "family", "stages", "launches", "configs", "dur_us_sum", "dram_pct",
            "sm_pct", "l2_pct", "l2_hit", "occ", "occ_theo", "issue_pct", "b_per_elem", "elem", "roof", "bound",
            "largest_shape", "largest_dur_us", "largest_dram_pct", "largest_sm_pct", "largest_occ", "passes"]
    write_tsv(os.path.join(a.out, "kernels.tsv"), [cols] + [[cell(k.get(c)) for c in cols] for k in kern])
    scols = ["workload", "kernel", "stage", "launches", "configs", "dur_us_sum", "dram_pct", "sm_pct", "l2_pct",
             "occ", "roof", "bound", "largest_shape"]
    write_tsv(os.path.join(a.out, "kernels_by_stage.tsv"), [scols] + [[cell(k.get(c)) for c in scols]
                                                                      for k in kstage])
    sys.stdout.write(md)
    if not launches:
        sys.stderr.write("summary: no profiled launch in any details CSV\n")
        return 2
    return 0


def aggregate(launches, key):
    groups = {}
    for r in launches:
        groups.setdefault(tuple(r[k] for k in key), []).append(r)
    out = []
    for gk, rs in groups.items():
        d = dict(zip(key, gk))
        d["launches"] = len(rs)
        d["configs"] = len({(r["kernel"], r["shape"]) for r in rs})
        d["dur_us_sum"] = sum(r["dur_us"] or 0.0 for r in rs)
        for c in ("dram_pct", "sm_pct", "l2_pct", "l2_hit", "occ", "occ_theo", "issue_pct"):
            d[c] = weighted(rs, c)
        byt = [r for r in rs if r.get("dram_bytes") is not None and r.get("elems")]
        d["b_per_elem"] = (sum(r["dram_bytes"] for r in byt) / sum(r["elems"] for r in byt)) if byt else None
        d["elem"] = "/".join(sorted({r["elem"] for r in rs}))
        if "family" not in d:
            d["family"] = rs[0]["family"]
        d["stages"] = "/".join(sorted({r["stage"] for r in rs}))
        d["passes"] = "/".join(sorted({r["pass"] for r in rs}))
        big = max(rs, key=lambda r: ((r["threads"] or 0), (r["dur_us"] or 0.0)))
        d["largest_shape"], d["largest_dur_us"] = big["shape"], big["dur_us"]
        d["largest_dram_pct"], d["largest_sm_pct"], d["largest_occ"] = big["dram_pct"], big["sm_pct"], big["occ"]
        d["largest_threads"] = big["threads"]
        d["roof"] = roof(d["dram_pct"], d["sm_pct"], d["l2_pct"])
        d["bound"] = bound_of(d["dram_pct"], d["sm_pct"], d["l2_pct"])
        out.append(d)
    out.sort(key=lambda d: (d.get("workload", ""), -d["dur_us_sum"]))
    return out


def render(kern, kstage, fam, launches, passes):
    o = io.StringIO()
    o.write("# Nsight Compute summary (noepoch_counters.sh, run B)\n\n")
    o.write(f"{len(launches)} profiled launches from {len(passes)} pass export(s). Duration is ncu's, at the "
            "clock ncu held (the sm_ghz column of launches.tsv); percentages are of peak at that clock. "
            "Values per kernel are duration-weighted over its profiled launches (one per launch shape in a "
            "config pass, the first N in a window pass). B/elem = DRAM bytes (read + write) per element (the "
            "elem column). roof: the highest of DRAM/compute/L2 % of peak, a roof at >= 60 %; bound: compute, "
            "memory, or latency when every roof is under 30 %. Σ time adds the profiled launches only: it "
            "ranks nothing, run A's stage tables do.\n\n")
    by = {}
    for k in kern:
        by.setdefault(k["kernel"], {})[k.get("workload", "?")] = k
    o.write("## epoch base against no-epoch, per kernel\n\n")
    o.write("Largest = the profiled launch with the most threads (the biggest instance each workload ran).\n\n")
    o.write("| kernel | family | workload | configs | SM % | DRAM % | L2 % | occ % (theo) | bound | largest: shape, "
            "µs, SM %, DRAM %, occ % |\n|---|---|---|---|---|---|---|---|---|---|\n")
    for name in sorted(by, key=lambda n: -max(v["dur_us_sum"] for v in by[n].values())):
        for wl in WORKLOADS + tuple(sorted(set(by[name]) - set(WORKLOADS))):
            k = by[name].get(wl)
            if k is None:
                continue
            o.write(f"| {name} | {k['family']} | {wl} | {k['configs']} | {fmt(k['sm_pct'])} | {fmt(k['dram_pct'])} | "
                    f"{fmt(k['l2_pct'])} | {fmt(k['occ'])} ({fmt(k['occ_theo'], 0)}) | {k['bound']} | "
                    f"{k['largest_shape']}, {fmt(k['largest_dur_us'], 0)}, {fmt(k['largest_sm_pct'])}, "
                    f"{fmt(k['largest_dram_pct'])}, {fmt(k['largest_occ'])} |\n")
    o.write("\n")
    for wl in sorted({k.get("workload", "?") for k in kern}):
        o.write(f"## {wl}: one row per kernel\n\n")
        o.write("| kernel | family | stage | launches (configs) | Σ time ms | DRAM % | SM % | L2 % | L2 hit % | "
                "occ % (theo) | issue % | B/elem | roof | largest launch: shape, µs, DRAM %, SM % |\n")
        o.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for k in kern:
            if k.get("workload", "?") != wl:
                continue
            o.write(f"| {k['kernel']} | {k['family']} | {k['stages']} | {k['launches']} ({k['configs']}) | "
                    f"{fmt(k['dur_us_sum'] / 1e3, 2)} | {fmt(k['dram_pct'])} | {fmt(k['sm_pct'])} | {fmt(k['l2_pct'])} | "
                    f"{fmt(k['l2_hit'])} | {fmt(k['occ'])} ({fmt(k['occ_theo'], 0)}) | {fmt(k['issue_pct'])} | "
                    f"{fmt(k['b_per_elem'], 2)} {k['elem']} | {k['roof']} | {k['largest_shape']}, "
                    f"{fmt(k['largest_dur_us'], 0)}, {fmt(k['largest_dram_pct'])}, {fmt(k['largest_sm_pct'])} |\n")
        o.write("\n")
    o.write("## per workload, stage and kernel (a main-commit kernel in the no-epoch fused stage is the recommit)\n\n")
    o.write("| workload | stage | kernel | launches (configs) | SM % | DRAM % | occ % | bound | largest shape |\n"
            "|---|---|---|---|---|---|---|---|---|\n")
    for k in sorted(kstage, key=lambda d: (d["workload"], d["stage"], -d["dur_us_sum"])):
        o.write(f"| {k['workload']} | {k['stage']} | {k['kernel']} | {k['launches']} ({k['configs']}) | "
                f"{fmt(k['sm_pct'])} | {fmt(k['dram_pct'])} | {fmt(k['occ'])} | {k['bound']} | {k['largest_shape']} |\n")
    o.write("\n## per stage and family (duration-weighted over every profiled launch of the family in that stage)\n\n")
    o.write("| workload | stage | family | launches | Σ time ms | DRAM % | SM % | L2 % | occ % | B/elem | roof |\n")
    o.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
    for f in sorted(fam, key=lambda d: (d["workload"], d["stage"], -d["dur_us_sum"])):
        o.write(f"| {f['workload']} | {f['stage']} | {f['family']} | {f['launches']} | {fmt(f['dur_us_sum'] / 1e3, 2)} | "
                f"{fmt(f['dram_pct'])} | {fmt(f['sm_pct'])} | {fmt(f['l2_pct'])} | {fmt(f['occ'])} | "
                f"{fmt(f['b_per_elem'], 2)} {f['elem']} | {f['roof']} |\n")
    return o.getvalue()


# ---------------------------------------------------------------------------------------------
# stages: which stage each profiled launch fell in, from the order of one pass's output lines

PROF_RE = re.compile(r'^==PROF== Profiling "([^"]+)"(?:\s*-\s*(\d+))?:?.*?(?:-\s*(\d+) passes)?\s*$')
R1_WALK_RE = re.compile(r"\[prover\] table walk R1\b")
FUSED_WALK_RE = re.compile(r"\[prover\] table walk rounds 2-4\b")
SPLIT_LINE_RE = re.compile(r"PROVE SPLIT #\d+")


def stages_from_lines(lines):
    """[(id, kernel, stage, passes)]. The k-th `==PROF== Profiling` line is report ID k-1 (ncu
    numbers results in the order it profiles them). The stage is set by the last prover line seen
    before it: `head` until the first `table walk R1` line, `main_commit` from it, `fused` from
    `table walk rounds 2-4`, `between` from the `PROVE SPLIT` line that closes the prove (the next
    prove's R1 line opens `main_commit` again). The prover prints those lines to stderr unbuffered
    and ncu prints its line while the launch is held, into the same file, so the order in the file
    follows the events. ncu cannot see the recommit: its launches read as `fused`."""
    stage, out, n = "head", [], 0
    for line in lines:
        line = line.rstrip("\n")
        m = PROF_RE.match(line)
        if m:
            out.append((n, m.group(1), stage, m.group(3) or ""))
            n += 1
            continue
        if R1_WALK_RE.search(line):
            stage = "main_commit"
        elif FUSED_WALK_RE.search(line):
            stage = "fused"
        elif SPLIT_LINE_RE.search(line):
            stage = "between"
    return out


def cmd_stages(a):
    with open(a.log, errors="replace") as f:
        rows = stages_from_lines(f)
    w = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
    w.writerow(["id", "kernel", "stage", "passes"])
    for r in rows:
        w.writerow(r)
    return 0


# ---------------------------------------------------------------------------------------------
# intervals: sorted, disjoint [start, end) lists in trace nanoseconds


def union(ivs):
    out = []
    for s, e in sorted(ivs):
        if e <= s:
            continue
        if out and s <= out[-1][1]:
            if e > out[-1][1]:
                out[-1][1] = e
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def subtract(a, b):
    """a minus b, both unions."""
    out, j = [], 0
    for s, e in a:
        cur = s
        while j < len(b) and b[j][1] <= cur:
            j += 1
        k = j
        while k < len(b) and b[k][0] < e:
            if b[k][0] > cur:
                out.append((cur, b[k][0]))
            cur = max(cur, b[k][1])
            k += 1
        if cur < e:
            out.append((cur, e))
    return out


def total(ivs):
    return sum(e - s for s, e in ivs)


def covered(merged, a, b):
    """ns of [a, b) covered by the merged intervals (sorted, disjoint)."""
    tot = 0
    i = bisect.bisect_right(merged, (a, float("inf"))) - 1
    i = max(i, 0)
    while i < len(merged):
        s, e = merged[i]
        if s >= b:
            break
        if e > a:
            tot += min(e, b) - max(s, a)
        i += 1
    return tot


def covered_ivs(merged, ivs):
    return sum(covered(merged, s, e) for s, e in ivs)


class StepFn:
    """A piecewise-constant count over time (how many ranges are open), for time averages."""

    def __init__(self, ranges):
        ev = {}
        for s, e in ranges:
            if e > s:
                ev[s] = ev.get(s, 0) + 1
                ev[e] = ev.get(e, 0) - 1
        self.t, self.v, c = [], [], 0
        for t in sorted(ev):
            c += ev[t]
            self.t.append(t)
            self.v.append(c)

    def integral(self, a, b):
        """∫ count dt over [a, b), in count x ns."""
        if not self.t or b <= a:
            return 0.0
        i = bisect.bisect_right(self.t, a) - 1
        tot, cur = 0.0, a
        while cur < b:
            v = self.v[i] if i >= 0 else 0
            nxt = self.t[i + 1] if i + 1 < len(self.t) else b
            nxt = min(nxt, b)
            tot += v * (nxt - cur)
            cur = nxt
            i += 1
        return tot

    def mean(self, ivs):
        w = total(ivs)
        return sum(self.integral(s, e) for s, e in ivs) / w if w else None


# ---------------------------------------------------------------------------------------------
# the trace: tables, kernels, copies, NVTX ranges, API calls, GPU metrics


def tables(db):
    return {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")}


def columns(db, table):
    return [r[1] for r in db.execute(f"PRAGMA table_info({table})")]


def session_start_ns(db, tbls):
    if "TARGET_INFO_SESSION_START_TIME" in tbls and "utcEpochNs" in columns(db, "TARGET_INFO_SESSION_START_TIME"):
        r = db.execute("SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME").fetchone()
        if r and r[0]:
            return int(r[0])
    return None


def open_ro(path):
    import sqlite3
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def load_kernels(db, tbls):
    """[(start, end, name, api_start or None, globalTid or None)] sorted by start: the launch API
    call joined by correlationId gives the launching thread and the launch time."""
    if "CUPTI_ACTIVITY_KIND_KERNEL" not in tbls:
        return []
    kcols = columns(db, "CUPTI_ACTIVITY_KIND_KERNEL")
    name_col = next((c for c in ("shortName", "demangledName", "mangledName") if c in kcols), None)
    name_sql = f"s.value" if name_col and "StringIds" in tbls else "'?'"
    join_s = f"LEFT JOIN StringIds s ON s.id = k.{name_col}" if name_col and "StringIds" in tbls else ""
    if "CUPTI_ACTIVITY_KIND_RUNTIME" in tbls:
        q = (f"SELECT k.start, k.end, {name_sql}, r.start, r.globalTid FROM CUPTI_ACTIVITY_KIND_KERNEL k {join_s} "
             f"LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = k.correlationId")
    else:
        q = f"SELECT k.start, k.end, {name_sql}, NULL, NULL FROM CUPTI_ACTIVITY_KIND_KERNEL k {join_s}"
    return sorted((s, e, n or "?", a, t) for s, e, n, a, t in db.execute(q))


def load_copies(db, tbls):
    out = []
    for t in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
        if t in tbls:
            out += [(s, e) for s, e in db.execute(f"SELECT start, end FROM {t}")]
    return out


def norm_label(lab):
    """An NVTX text to its label: the domain colon and a `[i=3]` instance suffix dropped."""
    lab = (lab or "").strip().lstrip(":")
    return re.sub(r"\[.*\]$", "", lab)


def load_nvtx(db, tbls):
    """[(start, end, label, globalTid)] of the closed NVTX ranges."""
    if "NVTX_EVENTS" not in tbls:
        return []
    cols = columns(db, "NVTX_EVENTS")
    txt = "e.text" if "text" in cols else "NULL"
    tid = "e.globalTid" if "globalTid" in cols else "0"
    if "textId" in cols and "StringIds" in tbls:
        q = (f"SELECT e.start, e.end, COALESCE({txt}, s.value), {tid} FROM NVTX_EVENTS e "
             f"LEFT JOIN StringIds s ON s.id = e.textId WHERE e.end IS NOT NULL AND e.end > e.start")
    else:
        q = f"SELECT e.start, e.end, {txt}, {tid} FROM NVTX_EVENTS e WHERE e.end IS NOT NULL AND e.end > e.start"
    return [(s, e, norm_label(lab), t) for s, e, lab, t in db.execute(q) if lab]


def load_api(db, tbls):
    """[(start, end, name, globalTid)] of every traced CUDA runtime and driver API call."""
    if "CUPTI_ACTIVITY_KIND_RUNTIME" not in tbls or "StringIds" not in tbls:
        return []
    q = ("SELECT r.start, r.end, s.value, r.globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
         "LEFT JOIN StringIds s ON s.id = r.nameId")
    return [(s, e, n or "?", t) for s, e, n, t in db.execute(q)]


def thread_names(db, tbls):
    out = {}
    if "ThreadNames" in tbls and "StringIds" in tbls:
        cols = columns(db, "ThreadNames")
        if "globalTid" in cols and "nameId" in cols:
            for t, n in db.execute("SELECT t.globalTid, s.value FROM ThreadNames t LEFT JOIN StringIds s "
                                   "ON s.id = t.nameId"):
                if n:
                    out[t] = n
    return out


def tid_of(gtid):
    return None if gtid is None else int(gtid) & 0xFFFFFF


# Key GPU-metric series, matched against the names nsys stores (they differ a little between GPU
# generations, so by pattern, first match wins); every series is in the metrics TSV whatever its name.
KEY_METRICS = [
    ("sm_active", re.compile(r"\bSMs? Active\b.*%", re.I)),
    ("sm_issue", re.compile(r"\bSM Issue\b.*%", re.I)),
    ("warps_in_flight", re.compile(r"\bCompute Warps in Flight \[Throughput %\]", re.I)),
    ("dram_read", re.compile(r"\bDRAM Read\b.*%", re.I)),
    ("dram_write", re.compile(r"\bDRAM Write\b.*%", re.I)),
    ("pcie_rx", re.compile(r"\bPCIe (RX|Read)\b.*%", re.I)),
    ("pcie_tx", re.compile(r"\bPCIe (TX|Write)\b.*%", re.I)),
    ("gr_active", re.compile(r"\bGR Active\b.*%", re.I)),
]


class Metrics:
    """Every GPU-metric series, sorted by time, with prefix sums for window means."""

    def __init__(self, db, tbls):
        self.series, self.why = {}, None
        if "GPU_METRICS" not in tbls:
            self.why = "no GPU_METRICS table (no GPU metrics were collected)"
            return
        cols = columns(db, "GPU_METRICS")
        ts = next((c for c in ("timestamp", "start") if c in cols), None)
        if ts is None or "metricId" not in cols or "value" not in cols:
            self.why = f"GPU_METRICS has unrecognised columns {cols}"
            return
        tcol = "typeId" if "typeId" in cols else "0"
        names = {}
        if "TARGET_INFO_GPU_METRICS" in tbls:
            icols = columns(db, "TARGET_INFO_GPU_METRICS")
            if "metricId" in icols and "metricName" in icols:
                tsel = "typeId" if "typeId" in icols else "0"
                for t, m, n in db.execute(f"SELECT DISTINCT {tsel}, metricId, metricName FROM TARGET_INFO_GPU_METRICS"):
                    names[(t, m)] = n
        raw = {}
        for t, m, when, v in db.execute(f"SELECT {tcol}, metricId, {ts}, value FROM GPU_METRICS ORDER BY {ts}"):
            raw.setdefault(names.get((t, m), f"metric {m}"), []).append((when, float(v or 0)))
        for name, pts in raw.items():
            tt = [p[0] for p in pts]
            pre = [0.0]
            for _, v in pts:
                pre.append(pre[-1] + v)
            self.series[name] = (tt, pre)
        if not self.series:
            self.why = "GPU_METRICS is empty"

    def name_of(self, key):
        pat = dict(KEY_METRICS)[key]
        return next((n for n in sorted(self.series) if pat.search(n)), None)

    def mean(self, name, ivs):
        if name not in self.series:
            return None
        tt, pre = self.series[name]
        s = c = 0.0
        for a, b in ivs:
            i, j = bisect.bisect_left(tt, a), bisect.bisect_left(tt, b)
            s += pre[j] - pre[i]
            c += j - i
        return s / c if c else None

    def samples(self, name, ivs):
        if name not in self.series:
            return 0
        tt, _ = self.series[name]
        return sum(bisect.bisect_left(tt, b) - bisect.bisect_left(tt, a) for a, b in ivs)


def read_smi(path, t0_ns):
    """nvidia-smi samples (timestamp, memory.used MiB, utilization.gpu %) -> [(trace ns, mib, util)].
    The timestamp is the local wall clock nvidia-smi prints ("2026/09/30 18:00:00.123"); this
    process runs on the same machine and in the same time zone as the sampler did."""
    out = []
    if not path or not os.path.exists(path) or t0_ns is None:
        return out
    with open(path, errors="replace") as f:
        for line in f:
            p = [x.strip() for x in line.split(",")]
            if len(p) < 3:
                continue
            try:
                ts = datetime.strptime(p[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
                out.append((int(ts * 1e9) - t0_ns, float(p[1]), float(p[2])))
            except ValueError:
                continue
    return sorted(out)


def smi_in(smi, ivs):
    """(max MiB, mean util %) over the samples in the intervals."""
    ts = [x[0] for x in smi]
    mx, us = None, []
    for a, b in ivs:
        for x in smi[bisect.bisect_left(ts, a):bisect.bisect_left(ts, b)]:
            mx = x[1] if mx is None or x[1] > mx else mx
            us.append(x[2])
    return mx, (sum(us) / len(us) if us else None)


# ---------------------------------------------------------------------------------------------
# the harness log


SPLIT_RE = re.compile(r"PROVE SPLIT #(\d+)[^:]*: airs (\d+) · rows (\d+) · wall ([0-9.]+)s · "
                      r"t=\[([0-9.]+),([0-9.]+)\] · prepass ([0-9.]+) · main_commit ([0-9.]+) · absorb ([0-9.]+) · "
                      r"fused ([0-9.]+) · other ([0-9.]+)")
RECOMMIT_SUM_RE = re.compile(r"recommit\[Σ\] ([0-9.]+)")
RESULT_RE = re.compile(r"NOEPOCH RESULT: verified=(\w+) · sub-proofs (\d+) · base ([0-9.]+) s \(execute ([0-9.]+) · "
                       r"build ([0-9.]+) · setup ([0-9.]+) · prove ([0-9.]+)\) · verify ([0-9.]+) s · proof (\d+) B · "
                       r"host peak ([0-9.]+ GiB|unknown) · device recommits (\d+)")
REFERENCE_RE = re.compile(r"NOEPOCH REFERENCE: epoch base ([0-9.]+) s · (\d+) epochs · host peak ([0-9.]+ GiB|unknown)")
TL_RE = re.compile(r"TABLE TL (\S+) idx=(\d+) (.*?) est=([0-9.]+)GiB claim=([0-9.]+) start=([0-9.]+) end=([0-9.]+)")


def read_log(path):
    lg = {"splits": [], "result": None, "reference": None, "tl": [], "recommit_sum": 0.0, "test_result": None,
          "packing": False, "compiled": None}
    if not path or not os.path.exists(path):
        return lg
    with open(path, errors="replace") as f:
        for line in f:
            m = SPLIT_RE.search(line)
            if m:
                g = m.groups()
                lg["splits"].append({"seq": int(g[0]), "airs": int(g[1]), "rows": int(g[2]), "wall": float(g[3]),
                                     "t0": float(g[4]), "t1": float(g[5]), "prepass": float(g[6]),
                                     "main_commit": float(g[7]), "absorb": float(g[8]), "fused": float(g[9]),
                                     "other": float(g[10])})
                r = RECOMMIT_SUM_RE.search(line)
                if r:
                    lg["recommit_sum"] += float(r.group(1))
            m = RESULT_RE.search(line)
            if m and lg["result"] is None:
                g = m.groups()
                lg["result"] = {"verified": g[0], "subs": int(g[1]), "base": float(g[2]), "execute": float(g[3]),
                                "build": float(g[4]), "setup": float(g[5]), "prove": float(g[6]),
                                "verify": float(g[7]), "size": int(g[8]), "peak": g[9], "recommits": int(g[10])}
            m = REFERENCE_RE.search(line)
            if m and lg["reference"] is None:
                lg["reference"] = {"base": float(m.group(1)), "epochs": int(m.group(2)), "peak": m.group(3)}
            m = TL_RE.search(line)
            if m:
                g = m.groups()
                lg["tl"].append({"phase": g[0], "idx": int(g[1]), "label": g[2], "est_gib": float(g[3]),
                                 "claim": float(g[4]), "start": float(g[5]), "end": float(g[6])})
            if line.startswith("test result:"):
                lg["test_result"] = line.strip()
            if "packing admission (LAMBDA_VM_GATE_PACKING=1)" in line:
                lg["packing"] = True
    return lg


def base_line(lg):
    if lg["result"]:
        r = lg["result"]
        return (f"no-epoch base {r['base']:.2f} s (execute {r['execute']:.2f} · build {r['build']:.2f} · setup "
                f"{r['setup']:.2f} · prove {r['prove']:.2f}) · {r['subs']} sub-proofs · verified={r['verified']} · "
                f"verify {r['verify']:.2f} s · host peak {r['peak']} · device recommits {r['recommits']}")
    if lg["reference"]:
        r = lg["reference"]
        return f"epoch base {r['base']:.2f} s · {r['epochs']} epochs · host peak {r['peak']}"
    return "no NOEPOCH RESULT/REFERENCE line in the log"


# ---------------------------------------------------------------------------------------------
# the stage windows

STAGES = ("head", "prepass", "main_commit", "between", "fused", "tail")
RECOMMIT_LABEL = "r1_main_recommit_table"
TASK_LABELS = ("r1_main_recommit_table", "r1_aux_build_table", "r1_aux_commit_table", "rounds_2to4_table")


def trace_windows(nvtx, lg, t0_ns, end_ns):
    """({stage: intervals} over the disjoint STAGES plus `recommit` and `whole`, source note)."""
    by = {}
    for s, e, lab, _t in nvtx:
        by.setdefault(lab, []).append((s, e))
    pre, mc, fu = union(by.get("r1_prepass", [])), union(by.get("r1_main_commit", [])), union(by.get("rounds_2to4", []))
    rc = union(by.get(RECOMMIT_LABEL, []))
    src = "NVTX ranges"
    if not (pre or mc or fu) and lg["splits"] and t0_ns is not None:
        src = "the log's PROVE SPLIT lines (no NVTX ranges; `other` unplaced, no recommit row)"
        for sp in lg["splits"]:
            a = sp["t0"] * 1e9 - t0_ns
            b = sp["t1"] * 1e9 - t0_ns
            pre.append((a, a + sp["prepass"] * 1e9))
            mc.append((a + sp["prepass"] * 1e9, a + (sp["prepass"] + sp["main_commit"]) * 1e9))
            fu.append((b - sp["fused"] * 1e9, b))
        pre, mc, fu = union(pre), union(mc), union(fu)
        rc = []
    if not (pre or mc or fu):
        return {"whole": [(0, end_ns)]}, "no NVTX ranges and no PROVE SPLIT line: one window only"
    allp = union(pre + mc + fu)
    first, last = allp[0][0], max(e for _, e in fu) if fu else allp[-1][1]
    mc = subtract(mc, pre)
    fu = subtract(fu, union(pre + mc))
    w = {"head": [(0, first)] if first > 0 else [], "prepass": pre, "main_commit": mc, "fused": fu,
         "between": subtract([(first, last)], union(pre + mc + fu)),
         "tail": [(last, end_ns)] if end_ns > last else [], "recommit": rc, "whole": [(0, end_ns)]}
    if not rc:
        w.pop("recommit")
    return w, src


def partition(w):
    """Sorted disjoint (start, end, stage) segments of the disjoint stages."""
    segs = sorted((s, e, st) for st in STAGES for s, e in w.get(st, []))
    return segs


def stage_at(segs, starts, t):
    i = bisect.bisect_right(starts, t) - 1
    if i >= 0 and segs[i][0] <= t < segs[i][1]:
        return segs[i][2]
    return "tail" if segs and t >= segs[-1][1] else "head"


# ---------------------------------------------------------------------------------------------
# CUDA API categories (driver and runtime names; first match wins)

API_CATS = [
    ("sync", r"cu(StreamSynchronize|CtxSynchronize|EventSynchronize)(_v\d+)?(_ptsz)?"
             r"|cuda(StreamSynchronize|DeviceSynchronize|EventSynchronize|ThreadSynchronize)(_v\d+)?(_ptsz)?"),
    ("copy_async", r"cu(Memcpy\w*Async|MemsetD\w*Async|MemPrefetchAsync)(_v\d+)?(_ptsz|_ptds)?"
                   r"|cuda(Memcpy\w*Async|Memset\w*Async|MemPrefetchAsync)(_v\d+)?(_ptsz|_ptds)?"),
    ("copy_sync", r"cu(Memcpy(HtoD|DtoH|DtoD|HtoA|AtoH|AtoD|DtoA|AtoA|2D|2DUnaligned|3D|3DPeer|Peer)?"
                  r"|MemsetD(8|16|32|2D8|2D16|2D32))(_v\d+)?(_ptds)?"
                  r"|cuda(Memcpy(2D|3D|Peer|ToSymbol|FromSymbol)?|Memset(2D|3D)?)(_v\d+)?(_ptds)?"),
    ("alloc", r"cu(MemAlloc(Pitch|Managed|Async|FromPoolAsync)?|MemCreate|MemMap|MemAddressReserve"
              r"|MemPoolCreate)(_v\d+)?(_ptsz)?"
              r"|cuda(Malloc(Async|Managed|Pitch|3D|FromPoolAsync)?)(_v\d+)?(_ptsz)?"),
    ("free", r"cu(MemFree(Async)?|MemRelease|MemUnmap|MemAddressFree|MemPoolDestroy|MemPoolTrimTo)(_v\d+)?(_ptsz)?"
             r"|cuda(Free(Async)?|MemPoolTrimTo)(_v\d+)?(_ptsz)?"),
    ("host_pinned", r"cu(MemHostAlloc|MemAllocHost|MemFreeHost|MemHostRegister|MemHostUnregister)(_v\d+)?"
                    r"|cuda(HostAlloc|MallocHost|FreeHost|HostRegister|HostUnregister)(_v\d+)?"),
    ("launch", r"cu(LaunchKernel(Ex)?|LaunchCooperativeKernel)(_ptsz)?"
               r"|cuda(LaunchKernel(ExC)?|LaunchCooperativeKernel)(_v\d+)?(_ptsz)?"),
    ("event_stream", r"cu(Event\w+|Stream\w+)(_v\d+)?(_ptsz)?|cuda(Event\w+|Stream\w+)(_v\d+)?(_ptsz)?"),
    ("module", r"cu(Module\w+|Library\w+|Func\w+|OccupancyMax\w+|Kernel\w+)(_v\d+)?|cuda(Func\w+|Occupancy\w+)"),
    ("context", r"cu(Init|Ctx\w+|DevicePrimaryCtx\w+|Device\w+|MemGetInfo|DriverGetVersion|GetExportTable|"
                r"GetProcAddress|PointerGetAttributes?)(_v\d+)?|cuda(SetDevice|GetDevice\w*|MemGetInfo|DeviceGet\w+)"),
]
API_CAT_RE = [(c, re.compile(p)) for c, p in API_CATS]
API_NOTE = ("sync = stream/context/event synchronize; copy_async = the *Async copies and memsets, which still "
            "block the calling thread when the host side is pageable (the driver stages through its own pinned "
            "buffer); copy_sync = the synchronous copies and memsets; alloc/free = device memory (cuMemAlloc*, "
            "cuMemFree*, pools, VMM); host_pinned = page-locked host memory (cuMemHostAlloc, cuMemAllocHost, "
            "cuMemHostRegister and their frees).")


def api_cat(name):
    for c, rx in API_CAT_RE:
        if rx.fullmatch(name):
            return c
    return "other"


# ---------------------------------------------------------------------------------------------
# runa: run A's tables from an nsys trace


def cmd_runa(a):
    db = open_ro(a.sqlite)
    tbls = tables(db)
    t0 = session_start_ns(db, tbls)
    kernels = load_kernels(db, tbls)
    copies = load_copies(db, tbls)
    nvtx = load_nvtx(db, tbls)
    api = load_api(db, tbls)
    tnames = thread_names(db, tbls)
    mets = Metrics(db, tbls)
    lg = read_log(a.log)
    smi = read_smi(a.smi, t0)
    end = max([e for _, e, *_ in kernels] + [e for _, e in copies] + [e for _, e, *_ in api] + [0])
    w, src = trace_windows(nvtx, lg, t0, end)
    kmerged = union([(s, e) for s, e, *_ in kernels])
    cmerged = union(copies)
    amerged = union([(s, e) for s, e, *_ in kernels] + copies)
    kstarts = [k[0] for k in kernels]
    maxdur = max([e - s for s, e, *_ in kernels] + [0])
    tasks = StepFn([(s, e) for s, e, lab, _ in nvtx if lab in TASK_LABELS])
    recs = StepFn([(s, e) for s, e, lab, _ in nvtx if lab == RECOMMIT_LABEL])
    keys = {k: mets.name_of(k) for k, _ in KEY_METRICS}
    os.makedirs(a.out, exist_ok=True)
    wl = a.workload

    def kernels_in(ivs):
        ksum, nl, kk = 0.0, 0, {}
        for s0, e0 in ivs:
            lo = bisect.bisect_left(kstarts, s0 - maxdur)
            hi = bisect.bisect_left(kstarts, e0)
            for s, e, k, _a, _t in kernels[lo:hi]:
                ov = min(e, e0) - max(s, s0)
                if ov > 0:
                    ksum += ov / 1e9
                    d = kk.setdefault(k, [0, 0.0])
                    d[1] += ov / 1e9
                    if s0 <= s < e0:
                        d[0] += 1
                        nl += 1
        return ksum, nl, kk

    order = [s for s in STAGES if s in w] + [s for s in ("recommit", "whole") if s in w]
    stages, per_kernel = [], {}
    for name in order:
        ivs = w[name]
        wall = total(ivs) / 1e9
        ksum, nl, kk = kernels_in(ivs)
        per_kernel[name] = kk
        vram, util = smi_in(smi, ivs)
        row = {"stage": name, "wall_s": wall, "ranges": len(ivs),
               "busy_pct": 100.0 * covered_ivs(amerged, ivs) / max(total(ivs), 1),
               "kernel_busy_pct": 100.0 * covered_ivs(kmerged, ivs) / max(total(ivs), 1),
               "copy_busy_pct": 100.0 * covered_ivs(cmerged, ivs) / max(total(ivs), 1),
               "kernel_sum_s": ksum, "launches": nl, "tasks_open": tasks.mean(ivs), "recommits_open": recs.mean(ivs),
               "vram_max_mib": vram, "smi_util_pct": util}
        for key, _ in KEY_METRICS:
            row[key] = mets.mean(keys[key], ivs) if keys[key] else None
        top = sorted(kk.items(), key=lambda kv: -kv[1][1])[:5]
        row["top"] = " ; ".join(f"{k} {v[1]:.2f}s ({100 * v[1] / ksum:.0f}%)" for k, v in top) if ksum else ""
        stages.append(row)
    cols = ["stage", "wall_s", "ranges", "busy_pct", "kernel_busy_pct", "copy_busy_pct", "kernel_sum_s", "launches",
            "tasks_open", "recommits_open", "vram_max_mib", "smi_util_pct"] + [k for k, _ in KEY_METRICS] + ["top"]

    def scell(c, v):
        return f"{v:.4f}" if c.endswith("_s") and isinstance(v, float) else cell(v)

    write_tsv(os.path.join(a.out, f"runa-{wl}-stages.tsv"), [cols] + [[scell(c, r.get(c)) for c in cols]
                                                                     for r in stages])
    rows = [["stage", "kernel", "launches", "sum_s", "share_of_stage_kernel_s"]]
    for st in stages:
        for k, (n, s) in sorted(per_kernel.get(st["stage"], {}).items(), key=lambda kv: -kv[1][1]):
            if s >= 0.001:
                rows.append([st["stage"], k, n, f"{s:.4f}",
                             f"{100 * s / st['kernel_sum_s']:.2f}" if st["kernel_sum_s"] else ""])
    write_tsv(os.path.join(a.out, f"runa-{wl}-stage-kernels.tsv"), rows)
    if mets.series:
        mrows = [["stage", "metric", "mean", "samples"]]
        for st in order:
            for n in sorted(mets.series):
                v = mets.mean(n, w[st])
                mrows.append([st, n, "" if v is None else f"{v:.3f}", mets.samples(n, w[st])])
        write_tsv(os.path.join(a.out, f"runa-{wl}-gpu-metrics.tsv"), mrows)

    # the kernels each NVTX label launched: the launch call on the label's thread, inside its range
    by_lab = {}
    for s, e, lab, t in nvtx:
        by_lab.setdefault(lab, {}).setdefault(t, []).append((s, e))
    for lab in by_lab:
        for t in by_lab[lab]:
            by_lab[lab][t].sort()
    attr = {}
    for lab, per_t in by_lab.items():
        d = {"ranges": sum(len(v) for v in per_t.values()), "range_s": sum(total(union(v)) for v in per_t.values()) / 1e9,
             "threads": len(per_t), "launches": 0, "kernel_s": 0.0, "k": {}}
        starts = {t: [r[0] for r in v] for t, v in per_t.items()}
        for s, e, k, api_s, gt in kernels:
            if api_s is None or gt not in per_t:
                continue
            i = bisect.bisect_right(starts[gt], api_s) - 1
            if i >= 0 and per_t[gt][i][0] <= api_s < per_t[gt][i][1]:
                d["launches"] += 1
                d["kernel_s"] += (e - s) / 1e9
                kd = d["k"].setdefault(k, [0, 0.0])
                kd[0] += 1
                kd[1] += (e - s) / 1e9
        attr[lab] = d
    arows = [["label", "ranges", "threads", "range_s_per_thread_union", "launches", "kernel_s", "top_kernels"]]
    krows = [["label", "kernel", "launches", "kernel_s"]]
    for lab, d in sorted(attr.items(), key=lambda kv: -kv[1]["kernel_s"]):
        top = sorted(d["k"].items(), key=lambda kv: -kv[1][1])
        arows.append([lab, d["ranges"], d["threads"], f"{d['range_s']:.3f}", d["launches"], f"{d['kernel_s']:.3f}",
                      " ; ".join(f"{k} {v[1]:.2f}s" for k, v in top[:5])])
        for k, (n, s) in top:
            krows.append([lab, k, n, f"{s:.4f}"])
    write_tsv(os.path.join(a.out, f"runa-{wl}-nvtx-kernels.tsv"), arows)
    write_tsv(os.path.join(a.out, f"runa-{wl}-nvtx-label-kernels.tsv"), krows)

    # CUDA API time per category, thread and stage (by the call's start)
    segs = partition(w)
    sstarts = [s[0] for s in segs]
    cat_st, thr, names = {}, {}, {}
    for s, e, n, gt in api:
        c = api_cat(n)
        st = stage_at(segs, sstarts, s) if segs else "whole"
        dur = (e - s) / 1e9
        d = cat_st.setdefault((c, st), [0, 0.0, 0.0])
        d[0] += 1
        d[1] += dur
        d[2] = max(d[2], dur)
        t = thr.setdefault((gt, c), {"calls": 0, "s": 0.0, "max": 0.0, "st": {}})
        t["calls"] += 1
        t["s"] += dur
        t["max"] = max(t["max"], dur)
        t["st"][st] = t["st"].get(st, 0.0) + dur
        nd = names.setdefault(n, {"cat": c, "calls": 0, "s": 0.0, "max": 0.0, "st": {}})
        nd["calls"] += 1
        nd["s"] += dur
        nd["max"] = max(nd["max"], dur)
        nd["st"][st] = nd["st"].get(st, 0.0) + dur
    stl = [s for s in STAGES if s in w] or ["whole"]
    cats = [c for c, _ in API_CATS] + ["other"]
    write_tsv(os.path.join(a.out, f"api-{wl}-by-category.tsv"),
              [["category", "stage", "calls", "sum_s", "max_ms"]] +
              [[c, st, v[0], f"{v[1]:.4f}", f"{v[2] * 1e3:.3f}"] for (c, st), v in
               sorted(cat_st.items(), key=lambda kv: (cats.index(kv[0][0]), stl.index(kv[0][1])
                                                      if kv[0][1] in stl else 99))])

    def tlabel(gt):
        return f"tid {tid_of(gt)}" + (f" ({tnames[gt]})" if gt in tnames else "")

    write_tsv(os.path.join(a.out, f"api-{wl}-by-thread.tsv"),
              [["thread", "category", "calls", "sum_s", "max_ms"] + [f"{s}_s" for s in stl]] +
              [[tlabel(gt), c, v["calls"], f"{v['s']:.4f}", f"{v['max'] * 1e3:.3f}"] +
               [f"{v['st'].get(s, 0.0):.4f}" for s in stl]
               for (gt, c), v in sorted(thr.items(), key=lambda kv: -kv[1]["s"])])
    write_tsv(os.path.join(a.out, f"api-{wl}-by-call.tsv"),
              [["call", "category", "calls", "sum_s", "max_ms"] + [f"{s}_s" for s in stl]] +
              [[n, v["cat"], v["calls"], f"{v['s']:.4f}", f"{v['max'] * 1e3:.3f}"] +
               [f"{v['st'].get(s, 0.0):.4f}" for s in stl]
               for n, v in sorted(names.items(), key=lambda kv: -kv[1]["s"])])

    # the time series
    binn = max(int(a.bin_ms * 1e6), 1_000_000)
    ts_cols = ["t_s", "stage", "kernel_busy_pct", "copy_busy_pct", "tasks_open", "recommits_open", "vram_mib",
               "smi_util_pct"] + [k for k, _ in KEY_METRICS]
    ts_rows = [ts_cols]
    smi_t = [x[0] for x in smi]
    fused_sec = []
    for b0 in range(0, int(end) + 1, binn):
        b1 = b0 + binn
        iv = [(b0, b1)]
        cnt = {}
        for s, e, st in segs[max(bisect.bisect_right(sstarts, b0) - 1, 0):bisect.bisect_left(sstarts, b1)]:
            ov = min(e, b1) - max(s, b0)
            if ov > 0:
                cnt[st] = cnt.get(st, 0) + ov
        st = max(cnt, key=cnt.get) if cnt else "-"
        j = bisect.bisect_left(smi_t, b1) - 1
        vr = smi[j][1] if j >= 0 and smi else None
        ut = smi[j][2] if j >= 0 and smi else None
        r = [f"{b0 / 1e9:.3f}", st, fmt(100.0 * covered(kmerged, b0, b1) / binn), fmt(100.0 * covered(cmerged, b0, b1) / binn),
             fmt(tasks.mean(iv), 2), fmt(recs.mean(iv), 2), fmt(vr, 0), fmt(ut, 0)]
        r += [fmt(mets.mean(keys[k], iv)) if keys[k] else "-" for k, _ in KEY_METRICS]
        ts_rows.append(r)
    write_tsv(os.path.join(a.out, f"runa-{wl}-timeseries.tsv"), ts_rows)
    if "fused" in w and w["fused"]:
        f0, f1 = w["fused"][0][0], w["fused"][-1][1]
        step = 1_000_000_000
        for b0 in range(int(f0), int(f1), step):
            iv = subtract([(b0, min(b0 + step, f1))], subtract([(f0, f1)], w["fused"]))
            if not total(iv):
                continue
            vr, ut = smi_in(smi, iv)
            fused_sec.append((b0 - f0, iv, vr))
            if len(fused_sec) >= 120:
                break

    # the markdown
    o = io.StringIO()
    o.write(f"# Run A, {wl}: the workload under Nsight Systems\n\n")
    o.write(f"{base_line(lg)}.\n\n")
    o.write(f"trace: {len(kernels)} kernel launches, {len(copies)} copies/memsets, {len(api)} CUDA API calls, "
            f"{len(nvtx)} NVTX ranges, {end / 1e9:.1f} s from the session start; {len(lg['splits'])} PROVE SPLIT "
            f"line(s); TABLE TL lines {len(lg['tl'])}; packing admission {'on' if lg['packing'] else 'off'}. "
            f"Stages from {src}.\n\n")
    o.write("busy % = any kernel or copy on the card; kernel/copy busy % = any kernel / any copy or memset. tasks = "
            "the mean number of fused-task NVTX ranges open (recommit, aux build, aux commit, rounds 2-4, one per "
            "driver thread at a time); recommits = the mean open recommit ranges. VRAM = the nvidia-smi maximum in "
            "the stage (200 ms samples). GPU metrics are nsys's samples averaged over the stage "
            f"({mets.why or 'collected'}): warps = Compute Warps in Flight, % of the card's warp slots (an "
            "occupancy proxy over time). `recommit` overlaps `fused`; `whole` is the run.\n\n")
    o.write("| stage | wall s | busy % | kernel busy % | copy busy % | Σ kernel s | launches | tasks | recommits | "
            "VRAM MiB | SM active % | SM issue % | warps % | DRAM rd % | DRAM wr % | PCIe rx % | PCIe tx % | "
            "top kernels (Σ s in the stage) |\n")
    o.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for r in stages:
        o.write(f"| {r['stage']} | {fmt(r['wall_s'], 2)} | {fmt(r['busy_pct'])} | {fmt(r['kernel_busy_pct'])} | "
                f"{fmt(r['copy_busy_pct'])} | {fmt(r['kernel_sum_s'], 2)} | {r['launches']} | {fmt(r['tasks_open'], 2)} | "
                f"{fmt(r['recommits_open'], 2)} | {fmt(r['vram_max_mib'], 0)} | {fmt(r['sm_active'])} | "
                f"{fmt(r['sm_issue'])} | {fmt(r['warps_in_flight'])} | {fmt(r['dram_read'])} | {fmt(r['dram_write'])} | "
                f"{fmt(r['pcie_rx'])} | {fmt(r['pcie_tx'])} | {r['top']} |\n")
    if fused_sec:
        o.write("\n## the fused stage, second by second\n\n")
        o.write("| s into fused | kernel busy % | tasks | recommits | VRAM MiB | SM active % | warps % | DRAM rd % | "
                "DRAM wr % | PCIe rx % | PCIe tx % |\n|---|---|---|---|---|---|---|---|---|---|---|\n")
        for off, iv, vr in fused_sec:
            o.write(f"| {off / 1e9:.0f} | {fmt(100.0 * covered_ivs(kmerged, iv) / max(total(iv), 1))} | "
                    f"{fmt(tasks.mean(iv), 2)} | {fmt(recs.mean(iv), 2)} | {fmt(vr, 0)} | "
                    f"{fmt(mets.mean(keys['sm_active'], iv)) if keys['sm_active'] else '-'} | "
                    f"{fmt(mets.mean(keys['warps_in_flight'], iv)) if keys['warps_in_flight'] else '-'} | "
                    f"{fmt(mets.mean(keys['dram_read'], iv)) if keys['dram_read'] else '-'} | "
                    f"{fmt(mets.mean(keys['dram_write'], iv)) if keys['dram_write'] else '-'} | "
                    f"{fmt(mets.mean(keys['pcie_rx'], iv)) if keys['pcie_rx'] else '-'} | "
                    f"{fmt(mets.mean(keys['pcie_tx'], iv)) if keys['pcie_tx'] else '-'} |\n")
    if attr:
        o.write("\n## kernels by the NVTX label that launched them\n\n")
        o.write("A kernel belongs to a label when its launch call ran on the label's thread inside one of its "
                "ranges (a launch from another thread, e.g. a rayon worker, is not attributed). range s = the "
                "ranges' time summed over threads.\n\n")
        o.write("| label | ranges | threads | range s | launches | Σ kernel s | top kernels |\n"
                "|---|---|---|---|---|---|---|\n")
        for r in arows[1:]:
            o.write("| " + " | ".join(str(x) for x in r) + " |\n")
        unattr = sum((e - s) for s, e, *_ in kernels) / 1e9 - (attr.get("r1_main_commit", {}).get("kernel_s", 0.0)
                                                               + sum(attr.get(l, {}).get("kernel_s", 0.0)
                                                                     for l in TASK_LABELS))
        o.write(f"\nΣ kernel time not launched from r1_main_commit or a fused-task range: {unattr:.2f} s "
                "(the main commits' own drivers, the head, rayon workers).\n")
    if api:
        o.write("\n## CUDA API time (calls x duration on the calling thread, by the call's start)\n\n")
        o.write(API_NOTE + "\n\n")
        o.write("| category | " + " | ".join(stl) + " | total s | calls | max ms |\n|---|" + "---|" * (len(stl) + 3) + "\n")
        for c in cats:
            vals = [cat_st.get((c, s), [0, 0.0, 0.0]) for s in stl]
            tot_s = sum(v[1] for v in vals)
            if not tot_s:
                continue
            o.write(f"| {c} | " + " | ".join(f"{v[1]:.2f}" for v in vals) +
                    f" | {tot_s:.2f} | {sum(v[0] for v in vals)} | {max(v[2] for v in vals) * 1e3:.1f} |\n")
        o.write("\ntop threads (thread, category), by API seconds:\n\n")
        o.write("| thread | category | calls | total s | max ms | " + " | ".join(stl) + " |\n|---|---|---|---|---|" +
                "---|" * len(stl) + "\n")
        for (gt, c), v in sorted(thr.items(), key=lambda kv: -kv[1]["s"])[:20]:
            if c == "launch" and v["s"] < 0.5:
                continue
            o.write(f"| {tlabel(gt)} | {c} | {v['calls']} | {v['s']:.2f} | {v['max'] * 1e3:.1f} | " +
                    " | ".join(f"{v['st'].get(s, 0.0):.2f}" for s in stl) + " |\n")
        o.write("\ntop calls:\n\n| call | category | calls | total s | max ms | " + " | ".join(stl) +
                " |\n|---|---|---|---|---|" + "---|" * len(stl) + "\n")
        for n, v in sorted(names.items(), key=lambda kv: -kv[1]["s"])[:20]:
            o.write(f"| {n} | {v['cat']} | {v['calls']} | {v['s']:.2f} | {v['max'] * 1e3:.1f} | " +
                    " | ".join(f"{v['st'].get(s, 0.0):.2f}" for s in stl) + " |\n")
    with open(os.path.join(a.out, f"runa-{wl}.md"), "w") as f:
        f.write(o.getvalue())
    sys.stdout.write(o.getvalue())
    nrec = attr.get(RECOMMIT_LABEL, {}).get("ranges", 0)
    print(f"RUNA {wl}: stages from {src}; recommit ranges {nrec}; device recommits "
          f"{lg['result']['recommits'] if lg['result'] else '-'}")
    return 0 if kernels else 2


# ---------------------------------------------------------------------------------------------
# dry: evaluate the plan's passes against an nsys trace of a workload


def load_trace(path):
    """Every kernel launch, in the order ncu sees them (the launch API call's start)."""
    db = open_ro(path)
    q = """SELECT s.value, k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, k.blockZ,
                  k.staticSharedMemory + k.dynamicSharedMemory, k.registersPerThread,
                  k.start, k.end, r.start
           FROM CUPTI_ACTIVITY_KIND_KERNEL k
           JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = k.correlationId
           JOIN StringIds s ON s.id = k.shortName
           ORDER BY r.start, k.correlationId"""
    out = []
    for (name, gx, gy, gz, bx, by, bz, smem, regs, ks, ke, api) in db.execute(q):
        out.append({"kernel": name, "grid": (gx, gy, gz), "block": (bx, by, bz), "smem": smem, "regs": regs,
                    "dur_us": (ke - ks) / 1e3, "api_ns": api, "k_start_ns": ks})
    first = out[0]["api_ns"] if out else 0
    for i, r in enumerate(out):
        r["idx"] = i
        r["t_run_s"] = (r["api_ns"] - first) / 1e9  # seconds since the first launch
    return out


def select(row, launches):
    """(matched, profiled): the launches the pass's regex matches, and the ones ncu would profile.
    window: ncu's default filter: skip `skip` matching launches, profile the next `count` (then
            --kill ends the run).
    config: --filter-mode per-launch-config, whose key is the launch's grid, block and shared
            memory: skip/count per key. A config pass names one kernel, so the key never mixes two."""
    rx = kernel_re(row)
    m = [r for r in launches if rx.fullmatch(r["kernel"])]
    skip, count = int(row["skip"]), int(row["count"])
    if row["mode"] == "window":
        return m, m[skip:skip + count]
    seen, picked = {}, []
    for r in m:
        k = (r["grid"], r["block"], r["smem"])
        n = seen.get(k, 0)
        if skip <= n < skip + count:
            picked.append(r)
        seen[k] = n + 1
    return m, picked


def est_seconds(row, picked, launches, wall_s):
    """A guide for the counters box, not a bound: 15 s to start, the workload up to the last
    profiled launch at 1.3x (window: --kill ends it there; config: the whole run, `wall_s`), and
    per profiled launch 1.5 s plus ~40 replays of its own duration."""
    if not launches:
        return 0.0
    end = wall_s
    if row["mode"] == "window":
        end = picked[-1]["t_run_s"] if picked else launches[-1]["t_run_s"]
    return 15.0 + 1.3 * end + sum(1.5 + 40 * r["dur_us"] * 1e-6 for r in picked)


def cmd_dry(a):
    plan = [r for r in read_plan(a.plan) if r["workload"] == a.workload]
    launches = load_trace(a.sqlite)
    db = open_ro(a.sqlite)
    tbls = tables(db)
    t0 = session_start_ns(db, tbls)
    nvtx = load_nvtx(db, tbls)
    lg = read_log(a.log)
    end = max([r["k_start_ns"] + r["dur_us"] * 1e3 for r in launches] + [0])
    w, src = trace_windows(nvtx, lg, t0, end)
    segs = partition(w)
    sstarts = [s[0] for s in segs]
    rc_ivs = w.get("recommit", [])
    rc_starts = [s for s, _ in rc_ivs]
    for r in launches:
        r["stage"] = stage_at(segs, sstarts, r["api_ns"]) if segs else "whole"
        i = bisect.bisect_right(rc_starts, r["api_ns"]) - 1
        if r["stage"] == "fused" and i >= 0 and rc_ivs[i][0] <= r["api_ns"] < rc_ivs[i][1]:
            r["stage"] = "fused(recommit window)"
    wall = end / 1e9
    os.makedirs(a.out, exist_ok=True)
    tot = sum(r["dur_us"] for r in launches) or 1.0
    per_kernel = {}
    for r in launches:
        k = per_kernel.setdefault(r["kernel"], {"n": 0, "us": 0.0, "cfg": set(), "stages": set()})
        k["n"] += 1
        k["us"] += r["dur_us"]
        k["cfg"].add((r["grid"], r["block"], r["smem"]))
        k["stages"].add(r["stage"].split("(")[0])
    tag = a.tag or a.workload
    o = io.StringIO()
    o.write(f"# The pass plan against a trace: {tag}\n\n")
    o.write(f"trace: {len(launches)} kernel launches, {len(per_kernel)} kernels, Σ kernel time {tot / 1e6:.2f} s, "
            f"{wall:.1f} s of trace; stages from {src}. A launch whose call started inside a recommit range reads "
            "`fused(recommit window)` (other tables' launches in the same window read so too).\n\n")
    o.write("## per pass: what ncu would profile\n\n")
    o.write("| pass | mode | skip/count | matched launches | kernels matched | configs | would profile | "
            "stages of those | shapes of those | est. s | verdict |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|\n")
    tsv = [["pass", "mode", "skip", "count", "matched", "kernels", "configs", "profile", "est_s", "verdict"]]
    detail = io.StringIO()
    rc = 0
    for row in plan:
        m, pa = select(row, launches)
        names = {}
        for r in m:
            names[r["kernel"]] = names.get(r["kernel"], 0) + 1
        cfgs = len({(r["kernel"], r["grid"], r["block"], r["smem"]) for r in m})
        st, shp = {}, {}
        for r in pa:
            st[r["stage"]] = st.get(r["stage"], 0) + 1
            s = shape(r["grid"], r["block"])
            shp[s] = shp.get(s, 0) + 1
        verdict = "ok" if pa else ("NO MATCH" if not m else "NOTHING SELECTED")
        if not pa:
            rc = 1
        est = est_seconds(row, pa, launches, wall)
        shapes_s = ", ".join(f"{k}{' x' + str(v) if v > 1 else ''}" for k, v in
                             sorted(shp.items(), key=lambda kv: -kv[1])[:4]) + (" …" if len(shp) > 4 else "")
        o.write(f"| {row['pass']} | {row['mode']} | {row['skip']}/{row['count']} | {len(m)} | {len(names)} | {cfgs} | "
                f"{len(pa)} | {', '.join(f'{k} {v}' for k, v in sorted(st.items()))} | {shapes_s} | {est:.0f} | "
                f"{verdict} |\n")
        tsv.append([row["pass"], row["mode"], row["skip"], row["count"], len(m), len(names), cfgs, len(pa),
                    f"{est:.0f}", verdict])
        detail.write(f"\n### {row['pass']} ({row['mode']}, kernels `{row['kernels']}`)\n\n")
        detail.write("matched: " + (", ".join(f"{k} x{v}" for k, v in sorted(names.items(), key=lambda kv: -kv[1]))
                                    or "NOTHING") + "\n\n")
        if pa:
            detail.write("| # in pass | launch idx | t s | stage | kernel | grid/block | smem | regs | µs |\n"
                         "|---|---|---|---|---|---|---|---|---|\n")
            for i, r in enumerate(pa[:60]):
                detail.write(f"| {i} | {r['idx']} | {r['t_run_s']:.2f} | {r['stage']} | {r['kernel']} | "
                             f"{shape(r['grid'], r['block'])} | {r['smem']} | {r['regs']} | {r['dur_us']:.1f} |\n")
            if len(pa) > 60:
                detail.write(f"| … | {len(pa) - 60} more | | | | | | | |\n")
    o.write("\nconfigs = distinct (kernel, grid, block, shared memory) among the matched. est. s = a guide for the "
            "counters box (see est_seconds).\n")
    o.write("\n## kernels by time, and which pass covers each\n\n| kernel | launches | configs | Σ s | % | stages | "
            "passes |\n|---|---|---|---|---|---|---|\n")
    uncovered = []
    for name, k in sorted(per_kernel.items(), key=lambda kv: -kv[1]["us"]):
        cov = [row["pass"] for row in plan if matches(row, name)]
        share = 100.0 * k["us"] / tot
        if share >= 1.0 and not cov:
            uncovered.append(f"{name} ({share:.1f} %)")
        if share >= 0.1 or cov:
            o.write(f"| {name} | {k['n']} | {len(k['cfg'])} | {k['us'] / 1e6:.3f} | {share:.1f} | "
                    f"{'/'.join(sorted(k['stages']))} | {', '.join(cov) or '-'} |\n")
    o.write(f"\nkernels with >= 1 % of kernel time that no pass covers: {', '.join(uncovered) or 'none'}\n")
    o.write(detail.getvalue())
    with open(os.path.join(a.out, f"dry-{tag}.md"), "w") as f:
        f.write(o.getvalue())
    write_tsv(os.path.join(a.out, f"dry-{tag}.tsv"), tsv)
    for t in tsv[1:]:
        print(f"DRY {tag} {t[0]}: matched {t[4]} launches ({t[5]} kernels, {t[6]} configs); would profile {t[7]}; "
              f"est {t[8]} s; {t[9]}")
    print(f"DRY {tag}: uncovered kernels >= 1 %: {', '.join(uncovered) or 'none'}")
    return rc


# ---------------------------------------------------------------------------------------------
# report: SUMMARY.md from a send directory


def md_section(path, start, stop="\n## "):
    """The text of a markdown file from the line starting with `start` to the next `stop` heading."""
    if not os.path.exists(path):
        return None
    t = open(path, errors="replace").read()
    i = t.find(start)
    if i < 0:
        return None
    j = t.find(stop, i + len(start))
    return t[i:j if j >= 0 else len(t)].rstrip() + "\n"


def cmd_report(a):
    s = a.send
    o = io.StringIO()
    o.write("# noepoch_counters.sh: summary\n\n")
    env = os.path.join(s, "env.txt")
    if os.path.exists(env):
        keep = ("script=", "repo=", "noepoch_", "gpu_name=", "driver=", "ncu=", "nsys=", "cpu=")
        o.write("```\n" + "".join(l for l in open(env) if l.startswith(keep)) + "```\n\n")
    ref = read_tsv(os.path.join(s, "reference", "runs.tsv"))
    if ref:
        o.write("## reference runs (no profiler)\n\n| workload | rc | seconds | base line | VRAM max MiB |\n"
                "|---|---|---|---|---|\n")
        for r in ref:
            o.write(f"| {r['workload']} | {r['rc']} | {r['seconds']} | {r['base']} | {r['vram_max_mib']} |\n")
        o.write("\n")
    for wl in WORKLOADS:
        p = os.path.join(s, "runa", wl, f"runa-{wl}.md")
        if not os.path.exists(p):
            continue
        t = open(p, errors="replace").read()
        o.write(f"## run A, {wl}\n\n")
        o.write(t.split("\n", 2)[2] if t.count("\n") > 2 else t)
        o.write("\n")
    k = md_section(os.path.join(s, "summary", "kernels.md"), "## epoch base against no-epoch")
    if k:
        o.write("## run B: " + k.split(" ", 1)[1] + "\n")
    passes = read_tsv(os.path.join(s, "passes.tsv"))
    if passes:
        o.write("## run B passes\n\n| pass | workload | mode | profiled | seconds | verdict |\n|---|---|---|---|---|---|\n")
        for r in passes:
            o.write(f"| {r['pass']} | {r['workload']} | {r['mode']} | {r['profiled']} | {r['seconds']} | {r['verdict']} |\n")
        o.write("\n")
    for wl in WORKLOADS:
        d = md_section(os.path.join(s, "runa", wl, f"dry-runa-{wl}.md"), "## per pass")
        if d:
            o.write(f"## the plan against run A's {wl} trace\n\n" + d.split("\n", 1)[1] + "\n")
    with open(os.path.join(s, "SUMMARY.md"), "w") as f:
        f.write(o.getvalue())
    print(f"report: {os.path.join(s, 'SUMMARY.md')} ({len(o.getvalue())} bytes)")
    return 0


# ---------------------------------------------------------------------------------------------
# plan-check, cargo-artifact


def cmd_plan_check(a):
    rows = read_plan(a.plan)
    errs = check_plan(rows)
    if not rows:
        errs.append("the plan has no pass")
    for e in errs:
        print("PLAN ERROR: " + e)
    if not errs:
        by = {}
        for r in rows:
            by[r["workload"]] = by.get(r["workload"], 0) + 1
        print("plan ok: " + ", ".join(f"{k} {v} pass(es)" for k, v in sorted(by.items())))
    return 1 if errs else 0


def cmd_cargo_artifact(a):
    found = None
    pkg = re.compile(r"(^|[/ ])" + re.escape(a.outdir or "\0") + r"([ #@]|$)")
    for line in sys.stdin:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if a.exe and msg.get("reason") == "compiler-artifact":
            tgt = msg.get("target", {})
            if (tgt.get("name") == a.exe and msg.get("executable") and msg.get("profile", {}).get("test")
                    and "lib" in tgt.get("kind", [])):
                found = msg["executable"]
        elif a.outdir and msg.get("reason") == "build-script-executed":
            if pkg.search(msg.get("package_id", "")) and msg.get("out_dir"):
                found = msg["out_dir"]
    if not found:
        sys.stderr.write("cargo-artifact: nothing matching in cargo's messages\n")
        return 1
    print(found)
    return 0


# ---------------------------------------------------------------------------------------------
# scrub and check: what may leave the machine

FORBIDDEN_EXT = (".ncu-rep", ".nsys-rep", ".qdrep", ".qdstrm", ".sqlite", ".sqlite3", ".db", ".arrow", ".parquet")
# case-sensitive credential shapes; a hit refuses the bundle (the report names file:line:marker only)
MARKERS = re.compile(r"API_KEY|APIKEY|SECRET|TOKEN|PASSWORD|PASSWD|PRIVATE KEY|BEGIN [A-Z ]*PRIVATE|"
                     r"ssh-(rsa|ed25519|dss|ecdsa)|ghp_[A-Za-z0-9]|gho_[A-Za-z0-9]|github_pat_|xox[bp]-|"
                     r"LS0tLS1CRUdJT|AKIA[0-9A-Z]{16}")


def text_of(path):
    """The file's text, or None when it is not plain UTF-8 text without NUL bytes."""
    with open(path, "rb") as f:
        data = f.read()
    if b"\0" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def drop_host_column(text):
    rows = list(csv.reader(io.StringIO(text)))
    hdr_i = next((i for i, r in enumerate(rows) if "Host Name" in r and "ID" in r), None)
    if hdr_i is None:
        return text, False
    j = rows[hdr_i].index("Host Name")
    o = io.StringIO()
    w = csv.writer(o, quoting=csv.QUOTE_ALL, lineterminator="\n")
    for i, r in enumerate(rows):
        if i >= hdr_i and len(r) > j:
            r = r[:j] + r[j + 1:]
        w.writerow(r)
    return o.getvalue(), True


def cmd_scrub(a):
    pairs = []
    for line in sys.stdin:
        line = line.rstrip("\n")
        if "\t" in line:
            old, new = line.split("\t", 1)
            if len(old) >= 3:
                pairs.append((old, new))
    pairs.sort(key=lambda p: -len(p[0]))
    counts, dropped = {}, 0
    for root, _dirs, files in os.walk(a.dir):
        for fn in files:
            p = os.path.join(root, fn)
            t = text_of(p)
            if t is None:
                continue
            new = t
            if fn.endswith(".csv"):
                new, d = drop_host_column(new)
                dropped += int(d)
            for old, rep in pairs:
                n = new.count(old)
                if n:
                    counts[rep] = counts.get(rep, 0) + n
                    new = new.replace(old, rep)
            if new != t:
                with open(p, "w", encoding="utf-8") as f:
                    f.write(new)
    print("scrub: " + (", ".join(f"{k} x{v}" for k, v in sorted(counts.items())) or "nothing to replace")
          + f"; Host Name column dropped from {dropped} CSV file(s)")
    return 0


def cmd_check(a):
    raw = sys.stdin.buffer.read().decode("utf-8", "replace")
    values = []
    for rec in raw.split("\0"):
        if "\t" in rec:
            label, val = rec.split("\t", 1)
            if len(val) >= 4:
                values.append((label, val))
    hits, bad, nfiles = [], [], 0
    for root, _dirs, files in os.walk(a.dir):
        for fn in sorted(files):
            p = os.path.join(root, fn)
            rel = os.path.relpath(p, a.dir)
            nfiles += 1
            if fn.lower().endswith(FORBIDDEN_EXT):
                bad.append(f"{rel}: an Nsight report, trace or database")
                continue
            if os.path.getsize(p) > 50 * 1024 * 1024:
                bad.append(f"{rel}: over 50 MB")
                continue
            t = text_of(p)
            if t is None:
                bad.append(f"{rel}: not plain UTF-8 text")
                continue
            if t.startswith("SQLite format 3"):
                bad.append(f"{rel}: a sqlite database")
                continue
            for ln, line in enumerate(t.split("\n"), 1):
                m = MARKERS.search(line)
                if m:
                    hits.append(f"{rel}:{ln}: credential marker '{m.group(0)[:4]}...'")
                for label, val in values:
                    if val in line:
                        hits.append(f"{rel}:{ln}: {label}")
    with open(a.report, "w") as f:
        for x in bad + hits:
            f.write(x + "\n")
    if bad or hits:
        files = sorted({x.split(":", 1)[0] for x in bad + hits})
        print(f"self-check: REFUSED: {len(hits)} hit(s) and {len(bad)} forbidden file(s) in {len(files)} file(s): "
              f"{' '.join(files[:12])}{' …' if len(files) > 12 else ''} (file:line:what in {a.report}; "
              "the matched text itself is never printed)")
        return 1
    print(f"self-check: clean: {nfiles} plain-text files, no Nsight report or database, no credential marker, "
          f"none of {len(values)} machine-specific values (hostname, home path, IP addresses, environment values)")
    return 0


# ---------------------------------------------------------------------------------------------
# selftest


def synth_details_csv(host="127.0.0.1", base_units=False):
    hdr = ["ID", "Process ID", "Process Name", "Host Name", "Kernel Name", "Context", "Stream", "Block Size",
           "Grid Size", "Device", "CC", "Section Name", "Metric Name", "Metric Unit", "Metric Value", "Rule Name",
           "Rule Type", "Rule Description", "Estimated Speedup Type", "Estimated Speedup"]
    rows = []

    def add(i, k, blk, grd, sec, name, unit, val):
        rows.append([str(i), "4242", "lambda_vm_prover-0123", host, k, "1", "13", blk, grd, "0", "12.0", sec, name,
                     unit, val, "", "", "", "", ""])

    # launch 0: a leaf kernel, compute-bound, bytes from the explicit counters
    k0, b0, g0 = "rpx_leaves_base_row_pair_batched", "(128, 1, 1)", "(16384, 1, 1)"
    dur, freq = (("nsecond", "62,360,000"), ("cycle/second", "2,010,000,000")) if base_units else \
        (("ms", "62.36"), ("Ghz", "2.01"))
    rd, wr = (("byte", "1,048,580,000"), ("byte", "67,110,000")) if base_units else \
        (("Mbyte", "1,048.58"), ("Mbyte", "67.11"))
    for sec, n, u, v in [(SOL, "Duration") + dur, (SOL, "SM Frequency") + freq,
                         (SOL, "DRAM Throughput", "%", "2.10"), (SOL, "Compute (SM) Throughput", "%", "92.70"),
                         (SOL, "Memory Throughput", "%", "33.15"), (SOL, "L2 Cache Throughput", "%", "20.00"),
                         (MWA, "L2 Hit Rate", "%", "98.10"), (OCC, "Achieved Occupancy", "%", "72.30"),
                         (OCC, "Theoretical Occupancy", "%", "75.00"), (OCC, "Block Limit Registers", "block", "9"),
                         (OCC, "Block Limit Warps", "block", "12"), (LST, "Threads", "thread", "2,097,152"),
                         (LST, "Registers Per Thread", "register/thread", "56"),
                         ("Command line profiler metrics", "dram__bytes_read.sum") + rd,
                         ("Command line profiler metrics", "dram__bytes_write.sum") + wr,
                         ("Command line profiler metrics",
                          "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio", "inst", "1.2"),
                         ("Command line profiler metrics",
                          "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio", "inst", "0.4"),
                         ("Command line profiler metrics",
                          "sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active", "%", "53.6"),
                         ("Command line profiler metrics",
                          "sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_active", "%", "88.0")]:
        add(0, k0, b0, g0, sec, n, u, v)
    # launch 1: an NTT pass, DRAM-bound, bytes from throughput x duration (no explicit counters)
    k1, b1, g1 = "ntt_cm_dit_k8", "(256, 1, 1)", "(2048, 16, 1)"
    tp = ("byte/second", "1,490,000,000,000") if base_units else ("Tbyte/s", "1.49")
    for sec, n, u, v in [(SOL, "Duration", "us", "140.50"), (SOL, "DRAM Throughput", "%", "84.00"),
                         (SOL, "Compute (SM) Throughput", "%", "20.00"), (SOL, "L2 Cache Throughput", "%", "40.00"),
                         (MWA, "Memory Throughput") + tp, (MWA, "L2 Hit Rate", "%", "45.0"),
                         (OCC, "Achieved Occupancy", "%", "30.0"), (OCC, "Theoretical Occupancy", "%", "33.3")]:
        add(1, k1, b1, g1, sec, n, u, v)
    # launch 2: a second NTT launch of another shape
    for sec, n, u, v in [(SOL, "Duration", "us", "59.50"), (SOL, "DRAM Throughput", "%", "70.00"),
                         (SOL, "Compute (SM) Throughput", "%", "18.00"), (SOL, "L2 Cache Throughput", "%", "30.00"),
                         (MWA, "Memory Throughput", "Gbyte/s", "1,200.00")]:
        add(2, k1, b1, "(1024, 16, 1)", sec, n, u, v)
    rows.append(["2", "4242", "lambda_vm_prover-0123", host, k1, "1", "13", b1, "(1024, 16, 1)", "0", "12.0",
                 "SpeedOfLight", "", "", "", "SOLBottleneck", "OPT", "rule text, no metric", "", ""])
    o = io.StringIO()
    o.write("==PROF== Connected to process 4242\n")
    w = csv.writer(o, quoting=csv.QUOTE_ALL, lineterminator="\n")
    w.writerow(hdr)
    w.writerows(rows)
    return o.getvalue()


T0_SYNTH = 1_800_000_000 * 10 ** 9
MAIN_TID, DRV_TID = (7 << 24) | 100, (7 << 24) | 101


def build_synth_sqlite(path):
    """A small nsys-shaped trace of one no-epoch prove, session start T0_SYNTH (trace seconds):
    head [0, 2), prepass [2, 2.5), main_commit [2.5, 5), absorb [5, 5.2), fused [5.2, 9) with one
    recommit range [5.2, 6) and a rounds_2to4_table range [6.5, 8.5) on a driver thread, tail [9, 10)."""
    import sqlite3
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE TARGET_INFO_SESSION_START_TIME (utcEpochNs INTEGER, utcTime TEXT, localTime TEXT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER, end INTEGER, eventClass INTEGER, globalTid INTEGER,
            correlationId INTEGER, nameId INTEGER, returnValue INTEGER, callchainId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, deviceId INTEGER, correlationId INTEGER,
            shortName INTEGER, gridX INTEGER, gridY INTEGER, gridZ INTEGER, blockX INTEGER, blockY INTEGER,
            blockZ INTEGER, staticSharedMemory INTEGER, dynamicSharedMemory INTEGER, registersPerThread INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INTEGER, end INTEGER, copyKind INTEGER, bytes INTEGER);
        CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, eventType INTEGER, rangeId INTEGER, category INTEGER,
            color INTEGER, text TEXT, globalTid INTEGER, endGlobalTid INTEGER, textId INTEGER, domainId INTEGER);
        CREATE TABLE ThreadNames (nameId INTEGER, priority INTEGER, globalTid INTEGER);
        CREATE TABLE GPU_METRICS (rawTimestamp INTEGER, timestamp INTEGER, typeId INTEGER, metricId INTEGER,
            value INTEGER);
        CREATE TABLE TARGET_INFO_GPU_METRICS (typeId INTEGER, sourceId INTEGER, typeName TEXT, metricId INTEGER,
            metricName TEXT);
    """)
    names = ["ntt_cm_dit_k8", "rpx_merkle_level", "rpx_leaves_base_row_pair_batched", "ccomp_0b8d15837e1e77a3",
             "cuLaunchKernel", "cuStreamSynchronize", "cuMemAlloc_v2", "cuMemcpyDtoHAsync_v2", "cuMemHostAlloc",
             "driver-3", "r1_main_commit"]
    for i, n in enumerate(names, 1):
        db.execute("INSERT INTO StringIds VALUES (?, ?)", (i, n))
    db.execute("INSERT INTO TARGET_INFO_SESSION_START_TIME VALUES (?, 'x', 'x')", (T0_SYNTH,))
    db.execute("INSERT INTO ThreadNames VALUES (10, 0, ?)", (DRV_TID,))
    sec = 10 ** 9

    def rng(a, b, text, tid, text_id=None):
        db.execute("INSERT INTO NVTX_EVENTS VALUES (?, ?, 59, 0, 0, 0, ?, ?, ?, ?, 0)",
                   (int(a * sec), int(b * sec), text, tid, tid, text_id))

    rng(2.0, 2.5, "r1_prepass", MAIN_TID)
    rng(2.5, 5.0, None, MAIN_TID, 11)        # a registered string: the text through StringIds
    rng(5.2, 9.0, "rounds_2to4", MAIN_TID)
    rng(5.2, 6.0, "r1_main_recommit_table", DRV_TID)
    rng(6.5, 8.5, ":rounds_2to4_table", DRV_TID)
    # (kernel, grid, t launch, duration ms, thread): the launch call starts 1 µs before the kernel
    launches = [("ntt_cm_dit_k8", 1024, 3.0, 100.0, MAIN_TID), ("rpx_leaves_base_row_pair_batched", 4096, 3.5, 200.0,
                                                                MAIN_TID),
                ("ntt_cm_dit_k8", 2048, 5.3, 50.0, DRV_TID), ("rpx_leaves_base_row_pair_batched", 8192, 5.5, 300.0,
                                                              DRV_TID),
                ("ccomp_0b8d15837e1e77a3", 512, 7.0, 400.0, DRV_TID), ("rpx_merkle_level", 256, 9.5, 10.0, MAIN_TID)]
    for c, (k, gx, ts, dms, tid) in enumerate(launches, 1):
        s = int(ts * sec)
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, 0, ?, ?, 5, 0, 0)", (s - 1000, s, tid, c))
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, 0, ?, ?, ?, 1, 1, 128, 1, 1, 0, 0, 40)",
                   (s, s + int(dms * 1e6), c, names.index(k) + 1, gx))
    # API calls without kernels: a 0.3 s sync in main_commit, a 0.2 s alloc and a 0.1 s pageable copy in fused
    for c, (nm, ts, dur, tid) in enumerate([("cuStreamSynchronize", 4.0, 0.3, MAIN_TID),
                                            ("cuMemAlloc_v2", 5.25, 0.2, DRV_TID),
                                            ("cuMemcpyDtoHAsync_v2", 7.5, 0.1, DRV_TID),
                                            ("cuMemHostAlloc", 1.0, 0.05, MAIN_TID)], 100):
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, 0, ?, ?, ?, 0, 0)",
                   (int(ts * sec), int((ts + dur) * sec), tid, c, names.index(nm) + 1))
    db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?, ?, 1, 1000)", (1 * sec, int(1.5 * sec)))
    db.execute("INSERT INTO TARGET_INFO_GPU_METRICS VALUES (7, 0, 'syn', 1, 'SMs Active [Throughput %]')")
    db.execute("INSERT INTO TARGET_INFO_GPU_METRICS VALUES (7, 0, 'syn', 2, 'Compute Warps in Flight [Throughput %]')")
    for tenth in range(0, 100):               # 10 Hz: SM active 80 in main_commit, 30 in fused, 0 elsewhere
        ts = tenth * 10 ** 8
        v = 80 if 25 <= tenth < 50 else (30 if 52 <= tenth < 90 else 0)
        db.execute("INSERT INTO GPU_METRICS VALUES (?, ?, 7, 1, ?)", (ts, ts, v))
        db.execute("INSERT INTO GPU_METRICS VALUES (?, ?, 7, 2, ?)", (ts, ts, v // 2))
    db.commit()
    db.close()


def selftest():
    fails = []

    def ok(cond, what):
        if not cond:
            fails.append(what)

    def fnum(s):
        try:
            return float(s)
        except (TypeError, ValueError):
            return float("nan")

    def run(fn, ns, stdin=None):
        so, si = sys.stdout, sys.stdin
        sys.stdout = io.StringIO()
        if stdin is not None:
            sys.stdin = stdin
        try:
            rc = fn(ns)
            return rc, sys.stdout.getvalue()
        finally:
            sys.stdout, sys.stdin = so, si

    ok(num("4,194,304") == 4194304.0 and num("62.36") == 62.36 and num("n/a") is None and num("1,2") is None,
       "num() parses ncu numbers")
    ok(to_bytes(1.5, "Kbyte") == 1500.0 and to_bytes(2, "Gbyte") == 2e9 and to_bytes(3, "%") is None
       and to_bytes(7, "byte") == 7.0, "to_bytes")
    ok(abs(to_bytes_per_s(1.49, "Tbyte/s") - 1.49e12) < 1 and to_bytes_per_s(5, "byte/second") == 5.0
       and to_us(62.36, "ms") == 62360.0 and to_us(62360000, "nsecond") == 62360.0, "rates and times")
    ok(abs(to_ghz(2.01, "Ghz") - 2.01) < 1e-9 and abs(to_ghz(2.01e9, "cycle/second") - 2.01) < 1e-9
       and to_ghz(2, "%") is None, "clock units")
    ok(dims("(128, 1, 1)") == (128, 1, 1) and dims("128, 2, 1") == (128, 2, 1) and dims("n/a") is None, "dims")
    ok(union([(5, 7), (1, 3), (2, 4), (7, 8)]) == [(1, 4), (5, 8)], "union")
    ok(subtract([(0, 10)], [(2, 3), (5, 7)]) == [(0, 2), (3, 5), (7, 10)] and subtract([(0, 4)], [(0, 4)]) == [],
       "subtract")
    sf = StepFn([(0, 10), (5, 15)])
    ok(sf.integral(0, 20) == 20.0 and abs(sf.mean([(5, 10)]) - 2.0) < 1e-9, "the open-range step function")
    ok(api_cat("cuStreamSynchronize") == "sync" and api_cat("cuMemcpyDtoHAsync_v2") == "copy_async"
       and api_cat("cuMemcpyHtoD_v2") == "copy_sync" and api_cat("cuMemAlloc_v2") == "alloc"
       and api_cat("cuMemAllocAsync") == "alloc" and api_cat("cuMemFreeAsync") == "free"
       and api_cat("cudaMallocHost") == "host_pinned" and api_cat("cuMemHostRegister_v2") == "host_pinned"
       and api_cat("cuEventSynchronize") == "sync" and api_cat("cuEventRecord") == "event_stream"
       and api_cat("cuLaunchKernel") == "launch" and api_cat("cuMemcpyBatchAsync") == "copy_async"
       and api_cat("cuModuleLoadData") == "module" and api_cat("cuWeird") == "other", "API categories")
    ok(norm_label(":rounds_2to4_table") == "rounds_2to4_table" and norm_label("epoch_prove[i=3]") == "epoch_prove",
       "NVTX labels")
    with tempfile.TemporaryDirectory() as d:
        plan = os.path.join(d, "plan.tsv")
        with open(plan, "w") as f:
            f.write("\t".join(PLAN_COLS) + "\n")
            f.write("noepoch_rowpair\tnoepoch\tconfig\t0\t1\trpx_leaves_base_row_pair_batched\tleaves\tone per shape\n")
            f.write("noepoch_ntt\tnoepoch\twindow\t1\t1\tntt_cm_di[ft]_k[4-8]\tlde\tncu window\n")
            f.write("noepoch_quot\tnoepoch\twindow\t0\t8\tccomp_[0-9a-f]+|constraint_composition_kernel\tquotient\tq\n")
            f.write("noepoch_none\tnoepoch\twindow\t0\t1\tno_such_kernel\tnone\tmust report NO MATCH\n")
            f.write("epoch_ntt\tepoch\twindow\t0\t4\tntt_cm_di[ft]_k[4-8]\tlde\tother workload\n")
        ok(check_plan(read_plan(plan)) == [], f"the synthetic plan is well formed ({check_plan(read_plan(plan))})")
        badplan = os.path.join(d, "bad.tsv")
        with open(badplan, "w") as f:
            f.write("\t".join(PLAN_COLS) + "\n")
            f.write("x\tepoch\twindow\t0\t0\t^rpx$\tf\tn\n")
            f.write("x\tzisk\tsome\t-1\t1\trpx(\tf\tn\n")
            f.write("y\tnoepoch\tconfig\t0\t1\tfri_fold_ext3|logup_[a-z0-9_]+\tf\tn\n")
        errs = check_plan(read_plan(badplan))
        ok(len(errs) >= 7 and any("exactly one kernel" in e for e in errs),
           f"a malformed plan is refused, a two-kernel config pass included ({len(errs)} errors)")
        # summary, in both unit styles, with an epoch export beside the no-epoch one
        for base_units in (False, True):
            nd = os.path.join(d, f"ncu{int(base_units)}")
            os.makedirs(nd)
            for p in ("noepoch_ntt", "epoch_ntt"):
                with open(os.path.join(nd, f"{p}.details.csv"), "w") as f:
                    f.write(synth_details_csv(base_units=base_units))
                with open(os.path.join(nd, f"{p}.stages.tsv"), "w") as f:
                    f.write("id\tkernel\tstage\tpasses\n0\trpx_leaves_base_row_pair_batched\tfused\t18\n"
                            "1\tntt_cm_dit_k8\tmain_commit\t18\n2\tntt_cm_dit_k8\tfused\t18\n")
            out = os.path.join(d, f"sum{int(base_units)}")
            rc, _ = run(cmd_summary, argparse.Namespace(plan=plan, ncu_dir=nd, out=out))
            tag = "base units" if base_units else "auto units"
            ok(rc == 0, f"summary exits 0 ({tag})")
            rows = [r for r in read_tsv(os.path.join(out, "launches.tsv")) if r["pass"] == "noepoch_ntt"]
            ok(len(rows) == 3, f"three launches ({len(rows)}, {tag})")
            r0 = rows[0]
            ok(r0["kernel"] == "rpx_leaves_base_row_pair_batched" and r0["stage"] == "fused" and r0["elem"] == "leaf",
               f"launch 0 fields ({tag})")
            ok(abs(fnum(r0["dur_us"]) - 62360.0) < 0.5 and abs(fnum(r0["sm_ghz"]) - 2.01) < 1e-6,
               f"duration and SM clock ({r0['dur_us']}, {r0['sm_ghz']}, {tag})")
            ok(r0["family"] == "leaves" and r0["workload"] == "noepoch" and r0["bound"] == "compute",
               "the family comes from the kernel, the workload from the pass")
            ok(abs(fnum(r0["b_per_elem"]) - (1048.58e6 + 67.11e6) / 2097152) < 0.01 and r0["bytes_src"] == "counters",
               f"leaf bytes/elem from the counters ({tag})")
            ok(r0["roof"].startswith("compute roof 93"), f"launch 0 roof ({r0['roof']})")
            ok(r0["occ_limit"] == "Registers" and "math_pipe_throttle 75%" in r0["stalls"] and "alu 54%" in r0["pipes"]
               and "fmaheavy 88%" in r0["pipe_cycles"], "occupancy limiter, stalls, pipes and pipe cycles")
            r1 = rows[1]
            want = 1.49e12 * 140.5e-6 / (2048 * 16 * 256 * 16)
            ok(abs(fnum(r1["b_per_elem"]) - want) < 0.01 and r1["bytes_src"] == "throughput x duration",
               f"NTT bytes/elem from throughput x duration ({r1['b_per_elem']} vs {want:.2f}, {tag})")
            ok(r1["roof"].startswith("DRAM roof 84") and r1["bound"] == "memory", "NTT roof and bound")
            kn = {(k["workload"], k["kernel"]): k for k in read_tsv(os.path.join(out, "kernels.tsv"))}
            ok(set(kn) == {(w, k) for w in ("epoch", "noepoch") for k in ("rpx_leaves_base_row_pair_batched",
                                                                           "ntt_cm_dit_k8")}, "one row per workload and kernel")
            ntt = kn[("noepoch", "ntt_cm_dit_k8")]
            wdram = (84.0 * 140.5 + 70.0 * 59.5) / 200.0
            ok(ntt["launches"] == "2" and ntt["configs"] == "2" and abs(fnum(ntt["dram_pct"]) - wdram) < 0.01,
               f"duration-weighted DRAM % ({ntt['dram_pct']} vs {wdram:.2f})")
            ok(ntt["largest_shape"] == "2048x16x1/256x1x1" and ntt["stages"] == "fused/main_commit",
               "largest launch, stages")
            md = open(os.path.join(out, "kernels.md")).read()
            ok("## epoch base against no-epoch, per kernel" in md and "| ntt_cm_dit_k8 | lde | epoch |" in md
               and "| ntt_cm_dit_k8 | lde | noepoch |" in md and "## noepoch: one row per kernel" in md,
               "kernels.md: the comparison and the per-workload tables")
            ks = read_tsv(os.path.join(out, "kernels_by_stage.tsv"))
            ok(len(ks) == 6 and any(r["stage"] == "fused" and r["kernel"] == "ntt_cm_dit_k8" for r in ks),
               f"kernels by stage ({len(ks)})")
        # stages from a pass's output order
        log = ['==PROF== Connected to process 1 (<W>/bin)', '==PROF== Profiling "ntt_cm_dit_k8": 0%....100% - 18 passes',
               '[prover] table walk R1 (walk weight, largest first): CPU[0] 3.1', '==PROF== Profiling "ntt_cm_dit_k8" - 1: 0%....100% - 17 passes',
               '[prover] table walk rounds 2-4 (walk weight, largest first): x', '==PROF== Profiling "rpx_merkle_level": 0%....100% - 17 passes',
               'PROVE SPLIT #0: airs 134 · rows 1 · wall 30.0s', '==PROF== Profiling "fri_fold_ext3": 0%....100% - 9 passes']
        st = stages_from_lines(log)
        ok([s[2] for s in st] == ["head", "main_commit", "fused", "between"] and st[1][3] == "17" and st[3][0] == 3,
           f"stages from the output order ({st})")
        # runa on the synthetic trace
        db = os.path.join(d, "t.sqlite")
        build_synth_sqlite(db)
        rlog = os.path.join(d, "a.log")
        with open(rlog, "w") as f:
            f.write("[prover] VRAM gate: packing admission (LAMBDA_VM_GATE_PACKING=1)\n"
                    f"PROVE SPLIT #0: airs 134 · rows 99 · wall 7.00s · t=[{T0_SYNTH / 1e9 + 2:.3f},{T0_SYNTH / 1e9 + 9:.3f}]"
                    " · prepass 0.50 · main_commit 2.50 · absorb 0.200 · fused 3.80 · other 0.00 || tables[Σ] aux_build 1"
                    " || recommit[Σ] 0.80\n"
                    "TABLE TL fused idx=3 CPU[0] est=10.50GiB claim=1.000 start=1.100 end=2.000\n"
                    "NOEPOCH RESULT: verified=true · sub-proofs 134 · base 9.00 s (execute 1.00 · build 1.00 · setup "
                    "0.00 · prove 7.00) · verify 1.00 s · proof 73400000 B · host peak 43.90 GiB · device recommits 1\n"
                    "test result: ok. 1 passed; 0 failed\n")
        smi = os.path.join(d, "smi.csv")
        with open(smi, "w") as f:
            for tenth in range(0, 100, 2):
                ts = datetime.fromtimestamp(T0_SYNTH / 1e9 + tenth / 10).strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]
                f.write(f"{ts}, {30000 if 52 <= tenth < 90 else 1000}, 50\n")
        rout = os.path.join(d, "runa")
        rc, printed = run(cmd_runa, argparse.Namespace(sqlite=db, log=rlog, workload="noepoch", out=rout, smi=smi,
                                                       bin_ms=500))
        ok(rc == 0 and "| fused | 3.80 |" in printed and "Stages from NVTX ranges" in printed,
           f"runa exits 0 with a 3.8 s fused stage ({printed[:600]!r})")
        srows = {r["stage"]: r for r in read_tsv(os.path.join(rout, "runa-noepoch-stages.tsv"))}
        ok(set(srows) == {"head", "prepass", "main_commit", "between", "fused", "tail", "recommit", "whole"},
           f"runa's stages ({sorted(srows)})")
        ok(abs(fnum(srows["main_commit"]["wall_s"]) - 2.5) < 1e-6 and abs(fnum(srows["between"]["wall_s"]) - 0.2) < 1e-6
           and abs(fnum(srows["recommit"]["wall_s"]) - 0.8) < 1e-6 and abs(fnum(srows["head"]["wall_s"]) - 2.0) < 1e-6,
           "runa's stage walls")
        ok(abs(fnum(srows["main_commit"]["kernel_sum_s"]) - 0.3) < 1e-6 and srows["fused"]["launches"] == "3",
           f"runa's kernels per stage ({srows['main_commit']['kernel_sum_s']}, {srows['fused']['launches']})")
        ok(abs(fnum(srows["main_commit"]["sm_active"]) - 80.0) < 1e-6 and abs(fnum(srows["fused"]["sm_active"]) - 30.0) < 1e-6
           and abs(fnum(srows["fused"]["warps_in_flight"]) - 15.0) < 1e-6, "runa's GPU metric means per stage")
        ok(abs(fnum(srows["fused"]["vram_max_mib"]) - 30000) < 1e-6 and abs(fnum(srows["head"]["vram_max_mib"]) - 1000) < 1e-6,
           f"runa's VRAM per stage ({srows['fused']['vram_max_mib']})")
        ok(abs(fnum(srows["fused"]["tasks_open"]) - 2.8 / 3.8) < 0.01 and abs(fnum(srows["recommit"]["recommits_open"]) - 1.0) < 1e-6,
           f"runa's open fused tasks ({srows['fused']['tasks_open']})")
        att = {r["label"]: r for r in read_tsv(os.path.join(rout, "runa-noepoch-nvtx-kernels.tsv"))}
        ok(att.get("r1_main_recommit_table", {}).get("launches") == "2"
           and abs(fnum(att["r1_main_recommit_table"]["kernel_s"]) - 0.35) < 1e-6
           and att.get("rounds_2to4_table", {}).get("launches") == "1" and att.get("r1_main_commit", {}).get("launches") == "2",
           f"kernels by launching NVTX label ({att})")
        cat = {(r["category"], r["stage"]): r for r in read_tsv(os.path.join(rout, "api-noepoch-by-category.tsv"))}
        ok(abs(fnum(cat[("sync", "main_commit")]["sum_s"]) - 0.3) < 1e-6 and abs(fnum(cat[("alloc", "fused")]["sum_s"]) - 0.2) < 1e-6
           and abs(fnum(cat[("copy_async", "fused")]["sum_s"]) - 0.1) < 1e-6 and ("host_pinned", "head") in cat
           and cat[("launch", "fused")]["calls"] == "3", f"API seconds per category and stage ({sorted(cat)})")
        thr = read_tsv(os.path.join(rout, "api-noepoch-by-thread.tsv"))
        ok(any(r["thread"] == "tid 101 (driver-3)" and r["category"] == "alloc" for r in thr), "API per named thread")
        ts = read_tsv(os.path.join(rout, "runa-noepoch-timeseries.tsv"))
        ok(len(ts) == 20 and ts[6]["stage"] == "main_commit" and ts[12]["stage"] == "fused", f"the time series ({len(ts)})")
        ok("## the fused stage, second by second" in printed and "RUNA noepoch: stages from NVTX ranges; recommit ranges 1;"
           " device recommits 1" in printed, "runa's fused table and its closing line")
        # the log fallback: no NVTX table
        import sqlite3
        db2 = os.path.join(d, "t2.sqlite")
        build_synth_sqlite(db2)
        c2 = sqlite3.connect(db2)
        c2.execute("DROP TABLE NVTX_EVENTS")
        c2.commit()
        c2.close()
        rc, printed = run(cmd_runa, argparse.Namespace(sqlite=db2, log=rlog, workload="noepoch", out=os.path.join(d, "r2"),
                                                       smi=None, bin_ms=1000))
        s2 = {r["stage"]: r for r in read_tsv(os.path.join(d, "r2", "runa-noepoch-stages.tsv"))}
        ok(rc == 0 and "PROVE SPLIT lines" in printed and "recommit" not in s2
           and abs(fnum(s2["fused"]["wall_s"]) - 3.8) < 1e-3 and abs(fnum(s2["main_commit"]["wall_s"]) - 2.5) < 1e-3,
           f"runa's log fallback ({sorted(s2)})")
        # dry against the synthetic trace
        dout = os.path.join(d, "dry")
        rc, printed = run(cmd_dry, argparse.Namespace(plan=plan, workload="noepoch", sqlite=db, log=rlog, out=dout,
                                                      tag=None))
        ok(rc == 1 and "noepoch_none: matched 0 launches" in printed and "NO MATCH" in printed,
           "dry exits 1 on a pass that matches nothing")
        drows = {r["pass"]: r for r in read_tsv(os.path.join(dout, "dry-noepoch.tsv"))}
        ok(set(drows) == {"noepoch_rowpair", "noepoch_ntt", "noepoch_quot", "noepoch_none"},
           "dry evaluates this workload's passes only")
        ok(drows["noepoch_rowpair"]["profile"] == "2" and drows["noepoch_ntt"]["profile"] == "1"
           and drows["noepoch_quot"]["matched"] == "1", f"dry selections ({drows})")
        dmd = open(os.path.join(dout, "dry-noepoch.md")).read()
        ok("fused(recommit window) 1" in dmd and "main_commit 1" in dmd, "dry stages, the recommit window included")
        # report
        send = os.path.join(d, "send")
        os.makedirs(os.path.join(send, "runa", "noepoch"))
        os.makedirs(os.path.join(send, "summary"))
        os.makedirs(os.path.join(send, "reference"))
        for fn in os.listdir(rout):
            with open(os.path.join(rout, fn)) as src, open(os.path.join(send, "runa", "noepoch", fn), "w") as dst:
                dst.write(src.read())
        with open(os.path.join(send, "summary", "kernels.md"), "w") as f:
            f.write(open(os.path.join(d, "sum0", "kernels.md")).read())
        with open(os.path.join(send, "reference", "runs.tsv"), "w") as f:
            f.write("workload\trc\tseconds\tbase\tvram_max_mib\nnoepoch\t0\t60\tno-epoch base 38.00 s\t32110\n")
        rc, _ = run(cmd_report, argparse.Namespace(send=send))
        rep = open(os.path.join(send, "SUMMARY.md")).read()
        ok(rc == 0 and "## run A, noepoch" in rep and "## run B: epoch base against no-epoch" in rep
           and "## reference runs" in rep and "| fused | 3.80 |" in rep, "report assembles SUMMARY.md")
        # scrub + check
        bd = os.path.join(d, "bundle")
        os.makedirs(os.path.join(bd, "ncu"))
        with open(os.path.join(bd, "ncu", "p.details.csv"), "w") as f:
            f.write(synth_details_csv(host="myhost.example"))
        with open(os.path.join(bd, "log.txt"), "w") as f:
            f.write("==PROF== Report: /home/alice/np-work/runs/x/keep/p.ncu-rep\nhost myhost.example ok\n")
        rc, scrub_out = run(cmd_scrub, argparse.Namespace(dir=bd),
                            stdin=io.StringIO("/home/alice/np-work\t<W>\n/home/alice\t<HOME>\nmyhost.example\t<HOST>\n"))
        txt = open(os.path.join(bd, "log.txt")).read()
        csvt = open(os.path.join(bd, "ncu", "p.details.csv")).read()
        ok("<W>/runs/x" in txt and "alice" not in txt and "<HOST> ok" in txt, "scrub replaces the literals")
        ok("Host Name" not in csvt and "myhost" not in csvt and "Host Name column dropped from 1" in scrub_out,
           "scrub drops the Host Name column")
        ok(len(read_details(os.path.join(bd, "ncu", "p.details.csv"))) == 3, "a scrubbed CSV still parses")

        def run_check(values):
            return run(cmd_check, argparse.Namespace(dir=bd, report=os.path.join(d, "report.txt")),
                       stdin=io.TextIOWrapper(io.BytesIO("\0".join(values).encode())))

        vals = ["hostname\tmyhost.example", "home\t/home/alice", "env:SOMEVAR\tvalue-of-somevar-123"]
        rc, msg = run_check(vals)
        ok(rc == 0 and msg.startswith("self-check: clean"), f"a scrubbed bundle is clean ({msg.strip()})")
        with open(os.path.join(bd, "log.txt"), "a") as f:
            f.write("leak value-of-somevar-123 here\n" + "gh" + "p_" + "A" * 36 + "\n")
        rc, msg = run_check(vals)
        rep = open(os.path.join(d, "report.txt")).read()
        ok(rc == 1 and "log.txt:3: env:SOMEVAR" in rep and "log.txt:4: credential marker" in rep
           and "value-of-somevar" not in msg + rep, "check refuses a planted value and marker, without printing them")
        with open(os.path.join(bd, "x.ncu-rep"), "wb") as f:
            f.write(b"\0\1\2")
        rc, msg = run_check([])
        ok(rc == 1 and "x.ncu-rep: an Nsight report" in open(os.path.join(d, "report.txt")).read(),
           "check refuses an Nsight report")
        # cargo-artifact
        msgs = [json.dumps({"reason": "compiler-artifact", "target": {"name": "lambda_vm_prover", "kind": ["lib"]},
                            "profile": {"test": True}, "executable": "/w/target/release/deps/lambda_vm_prover-1"}),
                json.dumps({"reason": "compiler-artifact", "target": {"name": "lambda_vm_prover", "kind": ["lib"]},
                            "profile": {"test": False}, "executable": None}),
                json.dumps({"reason": "build-script-executed",
                            "package_id": "path+file:///w/crypto/math-cuda#0.1.0", "out_dir": "/w/target/out"})]
        for want, ns in (("/w/target/release/deps/lambda_vm_prover-1", argparse.Namespace(exe="lambda_vm_prover", outdir=None)),
                         ("/w/target/out", argparse.Namespace(exe=None, outdir="math-cuda"))):
            rc, got = run(cmd_cargo_artifact, ns, stdin=io.StringIO("\n".join(msgs) + "\n"))
            ok(rc == 0 and got.strip() == want, f"cargo-artifact {want} ({got.strip()})")
    for f in fails:
        print("SELFTEST FAIL: " + f)
    print("SELFTEST " + ("GREEN" if not fails else f"RED ({len(fails)} failure(s))"))
    return 1 if fails else 0


# ---------------------------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summary")
    s.add_argument("--plan")
    s.add_argument("--ncu-dir", required=True)
    s.add_argument("--out", required=True)
    s = sub.add_parser("stages")
    s.add_argument("--log", required=True)
    s = sub.add_parser("runa")
    s.add_argument("--sqlite", required=True)
    s.add_argument("--log", required=True)
    s.add_argument("--workload", required=True, choices=WORKLOADS)
    s.add_argument("--out", required=True)
    s.add_argument("--smi", help="nvidia-smi samples: timestamp, memory.used, utilization.gpu (csv, noheader)")
    s.add_argument("--bin-ms", type=float, default=100.0, help="the time series' bin (ms, >= 1)")
    s = sub.add_parser("dry")
    s.add_argument("--plan", required=True)
    s.add_argument("--workload", required=True, choices=WORKLOADS)
    s.add_argument("--sqlite", required=True)
    s.add_argument("--log", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--tag", help="names the outputs dry-<tag>.md/.tsv (default: the workload)")
    s = sub.add_parser("report")
    s.add_argument("--send", required=True)
    s = sub.add_parser("plan-check")
    s.add_argument("--plan", required=True)
    s = sub.add_parser("cargo-artifact")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--exe")
    g.add_argument("--outdir")
    s = sub.add_parser("scrub")
    s.add_argument("--dir", required=True)
    s = sub.add_parser("check")
    s.add_argument("--dir", required=True)
    s.add_argument("--report", required=True)
    sub.add_parser("selftest")
    a = ap.parse_args(argv)
    return {"summary": cmd_summary, "stages": cmd_stages, "runa": cmd_runa, "dry": cmd_dry, "report": cmd_report,
            "plan-check": cmd_plan_check, "cargo-artifact": cmd_cargo_artifact, "scrub": cmd_scrub,
            "check": cmd_check, "selftest": lambda _a: selftest()}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
