#!/usr/bin/env python3
"""noepoch_counters_summary.py: the text side of noepoch_counters.sh (lane I-PROF2, 2026-10-01).

Standard library only (python >= 3.8; `runa` and `dry` also need the sqlite3 module).
noepoch_counters.sh carries a byte-identical copy of this file and writes it into its
work directory; the copy next to the script is the one to read and edit. It descends from
mauro_ncu_summary.py (lane G5-NCU, 2026-09-28) and lane I-PROF's version (09-30): the ncu
parsing, the scrub and the self-check are theirs; the workloads, the stage windows, the card
holds, the kernel families per stage, the CPU sampler and the NVTX-filtered passes are new.

    runa           --sqlite S --log L --workload W --out O [--smi F] [--cpu PREFIX] [--bin-ms N]
                                                   run A: per stage the card's busy time, GPU metrics, VRAM,
                                                   the kernels and their families, the host's CPU (cores busy,
                                                   per thread role); every card hold (an LFM proof's device
                                                   phase); the kernels each NVTX label launched; CUDA API time
                                                   per category, thread and stage; a time series
    summary        --plan P --ncu-dir D --out O    run B: one row per launch, per kernel and per stage
    stages         --log L --nvtx N                run B: the stage of each profiled launch (the pass's NVTX
                                                   window, or `run`)
    dry            --plan P --workload W --sqlite S --log L --out O [--modes M]
                                                   what each ncu pass would profile, against run A's trace
    cpu-sample     --exe BIN --out PREFIX [--interval-ms N] [--mem-floor-mib N]
                                                   the profiled process's CPU ticks per thread and the host's
                                                   MemAvailable every N ms, and a memory watchdog that ends the
                                                   process below the floor (PREFIX.watchdog says so)
    cpu-total      --cpu PREFIX                    one line: CPU seconds, mean cores busy, peak RSS of a run
    loghead        --log L --workload W            one line: the run's log gates (ok|bad), test result, card
                                                   holds, headline
    report         --send D                        SUMMARY.md: the text summary of a run, from the files above
    plan-check     --plan P                        the pass plan is well formed
    cargo-artifact (--exe NAME | --outdir PKG)     one path from cargo's JSON messages on stdin
    scrub          --dir D                         literal replacements (stdin: OLD<TAB>NEW lines), and the
                                                   "Host Name" column dropped from ncu CSVs
    check          --dir D --report R              the bundle self-check (stdin: LABEL<TAB>VALUE records,
                                                   NUL-separated): plain text only, no Nsight report or
                                                   database, no credential marker, none of the values
    selftest                                       every subcommand on synthetic inputs

Workloads. `whir`: the no-epoch WHIR block and its tree (lfm::whir_block_tests::
the_whir_block_tree_on_a_real_block, PR #1014). `stark`: the no-epoch STARK block and its tree
(lfm::block_tree_tests::the_block_tree_composes_to_a_top_node, PR #1013).

Stages (run A). The build carries process-wide NVTX ranges (start/end, on no thread's stack) for
the windows, and the prover's instruments spans as push/pop ranges on their threads:
  whir   setup (trace start to the base) · phase_a (blk_phase_a: the streamed build, the uploads and
         the commits) · prepared (blk_prepared) · phase_b (blk_phase_b: the argue and the openings) ·
         base_other (the rest of blk_base: the statement, the absorb, the glue) · recursion (lfm_tree)
         · verify (harness_verify to the end) · other. Overlapping rows: a_upload, a_commit, a_retire,
         b_upload, b_argue, b_encode, b_open (one range per group each) and the card holds.
  stark  setup · base_prepass, base_main_commit, base_fused (the multi_prove spans r1_prepass,
         r1_main_commit, rounds_2to4, inside blk_base) · base_other · harvest (tree_harvest) · level0
         (tree_level0) · interior (tree_interior) · verify · other. Overlapping rows: precommit,
         recommit and the card holds.
A card hold is a range cardhold_<phase> (multi_prove or build_artifacts): one LFM proof's device
phase under the tree's card permit, which admits one holder at a time, so every kernel inside it is
that proof's. Without NVTX ranges there is one window, `whole`.

What `summary` reports, per profiled launch and per kernel (duration-weighted over its launches):
  time      ncu's Duration, at the clock ncu held (launches.tsv's sm_ghz; env.txt names the
            --clock-control mode). Under `base` an RTX 5090 runs its SMs near 2.0 GHz where the block
            runs near 2.76 GHz, so a compute-bound kernel's duration here is ~1.4x its in-block time.
            The percentages are against the peak at the clock ncu ran.
  DRAM %    dram throughput, % of peak (Speed Of Light)
  SM %      Compute (SM) throughput, % of peak (Speed Of Light)
  L2 %      L2 throughput, % of peak (Speed Of Light); L2 hit % from Memory Workload Analysis
  occ %     achieved occupancy (theoretical beside it, and the block limit that sets it)
  waves     waves per SM (under 1: the launch cannot fill the card)
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
import signal
import sys
import tempfile
import time
from datetime import datetime

PLAN_COLS = ["pass", "workload", "modes", "mode", "skip", "count", "nvtx", "kernels", "family", "note"]
WORKLOADS = ("whir", "stark")
MODES = ("window", "config")
RUN_MODES = ("quick", "full")

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
        if not r["modes"] or any(m not in RUN_MODES for m in r["modes"].split(",")):
            errs.append(f"{where}: modes is a comma-separated subset of {RUN_MODES}")
        if r["nvtx"] != "-" and not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", r["nvtx"]):
            errs.append(f"{where}: nvtx is - or one process-wide NVTX range name ([A-Za-z][A-Za-z0-9_]*)")
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
            "sm_pct", "l2_pct", "l2_hit", "occ", "occ_theo", "issue_pct", "waves", "b_per_elem", "elem", "roof", "bound",
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
        for c in ("dram_pct", "sm_pct", "l2_pct", "l2_hit", "occ", "occ_theo", "issue_pct", "waves"):
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
    o.write("## per kernel, both workloads (the roofs)\n\n")
    o.write("Largest = the profiled launch with the most threads (the biggest instance each workload ran). "
            "waves = waves per SM, duration-weighted (under 1: the launch cannot fill the card).\n\n")
    o.write("| kernel | family | workload | configs | SM % | DRAM % | L2 % | occ % (theo) | waves | bound | largest: shape, "
            "µs, SM %, DRAM %, occ % |\n|---|---|---|---|---|---|---|---|---|---|---|\n")
    for name in sorted(by, key=lambda n: -max(v["dur_us_sum"] for v in by[n].values())):
        for wl in WORKLOADS + tuple(sorted(set(by[name]) - set(WORKLOADS))):
            k = by[name].get(wl)
            if k is None:
                continue
            o.write(f"| {name} | {k['family']} | {wl} | {k['configs']} | {fmt(k['sm_pct'])} | {fmt(k['dram_pct'])} | "
                    f"{fmt(k['l2_pct'])} | {fmt(k['occ'])} ({fmt(k['occ_theo'], 0)}) | {fmt(k['waves'], 2)} | {k['bound']} | "
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
    o.write("## per workload, stage and kernel (stage = the pass's NVTX window: blk_phase_a / blk_phase_b on "
            "the WHIR base, cardhold_multi_prove / cardhold_build_artifacts in the STARK tree)\n\n")
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


def stages_from_lines(lines, nvtx="-"):
    """[(id, kernel, stage, passes)]. The k-th `==PROF== Profiling` line is report ID k-1 (ncu
    numbers results in the order it profiles them). A pass filtered by a process-wide NVTX range
    profiles only the launches inside it, so the range is the stage; an unfiltered pass reads `run`."""
    stage = nvtx if nvtx and nvtx != "-" else "run"
    out, n = [], 0
    for line in lines:
        m = PROF_RE.match(line.rstrip("\n"))
        if m:
            out.append((n, m.group(1), stage, m.group(3) or ""))
            n += 1
    return out


def cmd_stages(a):
    with open(a.log, errors="replace") as f:
        rows = stages_from_lines(f, a.nvtx)
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
    """[(start, end, label, globalTid, process_wide)] of the closed NVTX ranges; process_wide is a
    start/end range (eventType 60), which sits on no thread's stack."""
    if "NVTX_EVENTS" not in tbls:
        return []
    cols = columns(db, "NVTX_EVENTS")
    txt = "e.text" if "text" in cols else "NULL"
    tid = "e.globalTid" if "globalTid" in cols else "0"
    et = "e.eventType" if "eventType" in cols else "59"
    if "textId" in cols and "StringIds" in tbls:
        q = (f"SELECT e.start, e.end, COALESCE({txt}, s.value), {tid}, {et} FROM NVTX_EVENTS e "
             f"LEFT JOIN StringIds s ON s.id = e.textId WHERE e.end IS NOT NULL AND e.end > e.start")
    else:
        q = (f"SELECT e.start, e.end, {txt}, {tid}, {et} FROM NVTX_EVENTS e "
             f"WHERE e.end IS NOT NULL AND e.end > e.start")
    return [(s, e, norm_label(lab), t, et_ == 60) for s, e, lab, t, et_ in db.execute(q) if lab]


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
TL_RE = re.compile(r"TABLE TL (\S+) idx=(\d+) (.*?) est=([0-9.]+)GiB claim=([0-9.]+) start=([0-9.]+) end=([0-9.]+)")
HOLD_RE = re.compile(r"CARD HOLD #(\d+) (\S+): waited ([0-9.]+)s · held ([0-9.]+)s · t=\[([0-9.]+),([0-9.]+)\]")
# the whir workload (lfm::whir_block_tests::the_whir_block_tree_on_a_real_block)
W3_BASE_RE = re.compile(r"^W3 BASE: ([0-9.]+)s")
W3_REC_RE = re.compile(r"^W3 RECURSION: ([0-9.]+)s after the base \(tree ([0-9.]+)s\) · whole block ([0-9.]+)s")
W3_VERIFY_RE = re.compile(r"^W3 TREE VERIFY: ([A-Z]+)")
W3_LEVEL_RE = re.compile(r"^W3 LEVEL (\d+): ([0-9.]+)s wall")
PHASES_RE = re.compile(r"^BLOCK PHASES: execute ([0-9.]+) · build ([0-9.]+) · prep ([0-9.]+) · A ([0-9.]+) "
                       r"\(wait ([0-9.]+) upload ([0-9.]+) commit ([0-9.]+) retire ([0-9.]+)\) · B ([0-9.]+) "
                       r"\(argue ([0-9.]+) open ([0-9.]+) tax ([0-9.]+) = upload ([0-9.]+) \+ encode ([0-9.]+)\)")
PHASE_A_END_RE = re.compile(r"phase A ended at ([0-9.]+)s")
# the stark workload (lfm::block_tree_tests::the_block_tree_composes_to_a_top_node)
NOEPOCH_BLOCK_RE = re.compile(r"★★★ NO-EPOCH BLOCK: base ([0-9.]+)s · harvest ([0-9.]+)s · level 0 ([0-9.]+)s "
                              r"\((\d+) leaves\) · interior ([0-9.]+)s \(levels ([^)]*)\) · recursion ([0-9.]+)s · "
                              r"whole ([0-9.]+)s")
STARK_BASE_RE = re.compile(r"^\s*base: (\d+) sub-proofs in ([0-9.]+)s \(execute ([0-9.]+) · build ([0-9.]+) · "
                           r"setup ([0-9.]+) · prove ([0-9.]+)\)")
LFM_PROVE_RE = re.compile(r"(BLOCK L\d+[^:]*?) LFM PROVE: execute ([0-9.]+)s · fill ([0-9.]+)s · "
                          r"multi_prove ([0-9.]+)s")
CHILD_RE = re.compile(r"CHILD VERIFIES beside the timed path: (\d+) accepted")
PEAK_RE = re.compile(r"★★★ WHOLE RUN: host peak ([0-9.]+) GiB")


def read_log(path):
    lg = {"splits": [], "tl": [], "holds": [], "recommit_sum": 0.0, "test_result": None, "packing": False,
          "w3_base": None, "w3_rec": None, "w3_verify": None, "w3_levels": [], "phases": None, "phase_a_end": None,
          "block": None, "stark_base": None, "lfm_proves": [], "child_verifies": None, "peak": None,
          "final_check": False}
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
            m = TL_RE.search(line)
            if m:
                g = m.groups()
                lg["tl"].append({"phase": g[0], "idx": int(g[1]), "label": g[2], "est_gib": float(g[3]),
                                 "claim": float(g[4]), "start": float(g[5]), "end": float(g[6])})
            m = HOLD_RE.search(line)
            if m:
                lg["holds"].append({"seq": int(m.group(1)), "phase": m.group(2), "waited": float(m.group(3)),
                                    "held": float(m.group(4)), "t0": float(m.group(5)), "t1": float(m.group(6))})
            m = W3_BASE_RE.search(line)
            if m:
                lg["w3_base"] = float(m.group(1))
            m = W3_REC_RE.search(line)
            if m:
                lg["w3_rec"] = {"recursion": float(m.group(1)), "tree": float(m.group(2)), "whole": float(m.group(3))}
            m = W3_VERIFY_RE.search(line)
            if m:
                lg["w3_verify"] = m.group(1)
            m = W3_LEVEL_RE.search(line)
            if m:
                lg["w3_levels"].append((int(m.group(1)), float(m.group(2))))
            m = PHASES_RE.search(line)
            if m:
                k = ("execute", "build", "prep", "a", "a_wait", "a_upload", "a_commit", "a_retire", "b", "b_argue",
                     "b_open", "b_tax", "b_upload", "b_encode")
                lg["phases"] = dict(zip(k, (float(x) for x in m.groups())))
            m = PHASE_A_END_RE.search(line)
            if m:
                lg["phase_a_end"] = float(m.group(1))
            m = NOEPOCH_BLOCK_RE.search(line)
            if m:
                g = m.groups()
                lg["block"] = {"base": float(g[0]), "harvest": float(g[1]), "level0": float(g[2]), "leaves": int(g[3]),
                               "interior": float(g[4]), "levels": g[5], "recursion": float(g[6]), "whole": float(g[7])}
            m = STARK_BASE_RE.search(line)
            if m and lg["stark_base"] is None:
                g = m.groups()
                lg["stark_base"] = {"subs": int(g[0]), "base": float(g[1]), "execute": float(g[2]),
                                    "build": float(g[3]), "setup": float(g[4]), "prove": float(g[5])}
            m = LFM_PROVE_RE.search(line)
            if m:
                lg["lfm_proves"].append({"label": m.group(1), "execute": float(m.group(2)), "fill": float(m.group(3)),
                                         "multi_prove": float(m.group(4))})
            m = CHILD_RE.search(line)
            if m:
                lg["child_verifies"] = int(m.group(1))
            m = PEAK_RE.search(line)
            if m:
                lg["peak"] = float(m.group(1))
            if "BLOCK FINAL CHECK (harness)" in line:
                lg["final_check"] = True
            if line.startswith("test result:"):
                lg["test_result"] = line.strip()
            if "packing admission (LAMBDA_VM_GATE_PACKING=1)" in line:
                lg["packing"] = True
    return lg


def base_line(lg):
    """The run's headline from its log, either workload."""
    parts = []
    if lg["w3_base"] is not None:
        parts.append(f"whir base {lg['w3_base']:.2f} s")
        if lg["w3_rec"]:
            r = lg["w3_rec"]
            parts.append(f"recursion {r['recursion']:.2f} s (tree {r['tree']:.2f}) · whole {r['whole']:.2f} s")
        if lg["phases"]:
            p = lg["phases"]
            parts.append(f"phases: execute {p['execute']:.2f} · build {p['build']:.2f} · A {p['a']:.2f} (upload "
                         f"{p['a_upload']:.2f} commit {p['a_commit']:.2f}) · B {p['b']:.2f} (argue {p['b_argue']:.2f} "
                         f"open {p['b_open']:.2f} tax {p['b_tax']:.2f})")
        parts.append(f"tree verify {lg['w3_verify'] or '-'}")
    if lg["block"]:
        b = lg["block"]
        parts.append(f"stark base {b['base']:.2f} s · harvest {b['harvest']:.2f} · level 0 {b['level0']:.2f} "
                     f"({b['leaves']} leaves) · interior {b['interior']:.2f} (levels {b['levels']}) · recursion "
                     f"{b['recursion']:.2f} · whole {b['whole']:.2f} s")
        if lg["lfm_proves"]:
            mp = [x["multi_prove"] for x in lg["lfm_proves"]]
            parts.append(f"{len(mp)} LFM PROVE lines, multi_prove Σ {sum(mp):.2f} s (mean {sum(mp) / len(mp):.2f})")
        parts.append(f"child verifies {lg['child_verifies'] if lg['child_verifies'] is not None else '-'}")
    if lg["holds"]:
        parts.append(f"{len(lg['holds'])} card holds, held Σ {sum(h['held'] for h in lg['holds']):.2f} s")
    if lg["peak"] is not None:
        parts.append(f"host peak {lg['peak']:.1f} GiB")
    return " · ".join(parts) if parts else "no W3 BASE / NO-EPOCH BLOCK line in the log"


# ---------------------------------------------------------------------------------------------
# the stage windows

STAGES_BY = {
    "whir": ("setup", "phase_a", "prepared", "phase_b", "base_other", "recursion", "verify", "other"),
    "stark": ("setup", "base_prepass", "base_main_commit", "base_fused", "base_other", "harvest", "level0",
              "interior", "verify", "other"),
}
OVERLAP_BY = {
    "whir": (("a_upload", "blk_a_upload"), ("a_commit", "blk_a_commit"), ("a_retire", "blk_a_retire"),
             ("b_upload", "blk_b_upload"), ("b_argue", "blk_b_argue"), ("b_encode", "blk_b_encode"),
             ("b_open", "blk_b_open")),
    "stark": (("precommit", "r1_precommit_table"), ("recommit", "r1_main_recommit_table")),
}
HOLD_PREFIX = "cardhold_"
RECOMMIT_LABEL = "r1_main_recommit_table"
TASK_LABELS = ("r1_main_recommit_table", "r1_aux_build_table", "r1_aux_commit_table", "rounds_2to4_table")
# Thread roles: a push/pop range on a thread names what that thread is (the WHIR block's threads).
ROLE_LABELS = ("blk_executor", "blk_walker", "blk_builder", "blk_layout", "blk_b_upload_next")


def intersect(a, b):
    """a and b, both unions."""
    return subtract(a, subtract(a, b))


def trace_windows(nvtx, end_ns, workload):
    """({stage: intervals} over the workload's disjoint stages, its overlapping rows, one row per
    card-hold phase (`hold:<phase>`) and `whole`; source note)."""
    by = {}
    for s, e, lab, *_ in nvtx:
        by.setdefault(lab, []).append((s, e))
    u = {lab: union(v) for lab, v in by.items()}
    whole = [(0, end_ns)]
    base = u.get("blk_base", [])
    if not base:
        return {"whole": whole}, "no blk_base NVTX range (no libnvToolsExt, or another build): one window only"
    claims = []
    if workload == "whir":
        claims = [("phase_a", u.get("blk_phase_a", [])), ("prepared", u.get("blk_prepared", [])),
                  ("phase_b", u.get("blk_phase_b", [])), ("base_other", base)]
    else:
        pre, mc, fu = (intersect(u.get(k, []), base) for k in ("r1_prepass", "r1_main_commit", "rounds_2to4"))
        claims = [("base_prepass", pre), ("base_main_commit", mc), ("base_fused", fu), ("base_other", base),
                  ("harvest", u.get("tree_harvest", [])), ("level0", u.get("tree_level0", [])),
                  ("interior", u.get("tree_interior", []))]
    if workload == "whir":
        claims.append(("recursion", u.get("lfm_tree", [])))
    ver = u.get("harness_verify", [])
    if ver:
        claims.append(("verify", [(ver[0][0], max(end_ns, ver[-1][1]))]))
    claims.append(("setup", [(0, base[0][0])]))
    w, taken = {}, []
    for name, ivs in claims:
        own = subtract(intersect(union(ivs), whole), union(taken))
        if own:
            w[name] = own
            taken = union(taken + own)
    rest = subtract(whole, taken)
    if rest:
        w["other"] = rest
    for name, lab in OVERLAP_BY[workload]:
        if u.get(lab):
            w[name] = u[lab]
    for lab in sorted(u):
        if lab.startswith(HOLD_PREFIX):
            w["hold:" + lab[len(HOLD_PREFIX):]] = u[lab]
    w["whole"] = whole
    return w, "NVTX ranges"


def partition(w, workload):
    """Sorted disjoint (start, end, stage) segments of the workload's disjoint stages."""
    names = STAGES_BY.get(workload, ())
    return sorted((s, e, st) for st in names for s, e in w.get(st, []))


def stage_at(segs, starts, t):
    i = bisect.bisect_right(starts, t) - 1
    if i >= 0 and segs[i][0] <= t < segs[i][1]:
        return segs[i][2]
    return "other"


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
# the host's CPU: the sampler's files (cpu-sample) read back on the trace's clock


def read_cpu(prefix, t0_ns):
    """{"clk", "proc": [(t, ticks, rss_kb, threads, avail_mib)], "tid": {tid: [(t, ticks)]}, "comm": {tid: comm},
    "watchdog"} with t in trace nanoseconds (the sampler stamps the wall clock, as nsys's session start is);
    None without the files."""
    if not prefix or t0_ns is None or not os.path.exists(prefix + ".proc.tsv"):
        return None
    clk, proc, tid, comm = 100, [], {}, {}
    with open(prefix + ".proc.tsv", errors="replace") as f:
        for line in f:
            if line.startswith("# clk_tck "):
                clk = int(line.split()[2])
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 6 or not p[0].isdigit():
                continue
            proc.append((int(p[0]) - t0_ns, int(p[2]), int(p[3]), int(p[4]), int(p[5])))
    if os.path.exists(prefix + ".tid.tsv"):
        with open(prefix + ".tid.tsv", errors="replace") as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) < 4 or not p[0].isdigit() or not p[1].isdigit():
                    continue
                t = int(p[1])
                comm[t] = p[2]
                tid.setdefault(t, []).append((int(p[0]) - t0_ns, int(p[3])))
    wd = None
    if os.path.exists(prefix + ".watchdog"):
        wd = open(prefix + ".watchdog", errors="replace").read().strip()
    return {"clk": clk, "proc": sorted(proc), "tid": tid, "comm": comm, "watchdog": wd}


def cpu_deltas(series):
    """[(t_mid, dt_ns, dticks)] between consecutive cumulative samples."""
    out = []
    for (ta, ka, *_), (tb, kb, *_) in zip(series, series[1:]):
        if tb > ta and kb >= ka:
            out.append(((ta + tb) // 2, tb - ta, kb - ka))
    return out


def comm_family(comm):
    return re.sub(r"[-_ ]?\d+$", "", comm or "?") or "?"


def cpu_by_stage(cpu, segs, starts):
    """({stage: (cpu_s, covered_s)}, {(role, stage): cpu_s}) from the process and per-thread series."""
    st, roles = {}, {}
    if not cpu:
        return st, roles
    for t, dt, dk in cpu_deltas(cpu["proc"]):
        name = stage_at(segs, starts, t) if segs else "whole"
        d = st.setdefault(name, [0.0, 0.0])
        d[0] += dk / cpu["clk"]
        d[1] += dt / 1e9
    for tid, series in cpu["tid"].items():
        role = cpu.get("roles", {}).get(tid) or comm_family(cpu["comm"].get(tid))
        for t, _dt, dk in cpu_deltas(series):
            name = stage_at(segs, starts, t) if segs else "whole"
            roles[(role, name)] = roles.get((role, name), 0.0) + dk / cpu["clk"]
    return st, roles


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
    cpu = read_cpu(a.cpu, t0)
    wl = a.workload
    end = max([e for _, e, *_ in kernels] + [e for _, e in copies] + [e for _, e, *_ in api] + [0])
    w, src = trace_windows(nvtx, end, wl)
    segs = partition(w, wl)
    sstarts = [x[0] for x in segs]
    kmerged = union([(s, e) for s, e, *_ in kernels])
    cmerged = union(copies)
    amerged = union([(s, e) for s, e, *_ in kernels] + copies)
    kstarts = [k[0] for k in kernels]
    maxdur = max([e - s for s, e, *_ in kernels] + [0])
    tasks = StepFn([(s, e) for s, e, lab, *_ in nvtx if lab in TASK_LABELS])
    holdsfn = StepFn([(s, e) for s, e, lab, *_ in nvtx if lab.startswith(HOLD_PREFIX)])
    keys = {k: mets.name_of(k) for k, _ in KEY_METRICS}
    if cpu is not None:
        cpu["roles"] = {}
        for _s, _e, lab, gt, pw in nvtx:
            if not pw and lab in ROLE_LABELS and gt is not None:
                cpu["roles"].setdefault(tid_of(gt), lab[len("blk_"):])
    cpu_st, cpu_roles = cpu_by_stage(cpu, segs, sstarts)
    os.makedirs(a.out, exist_ok=True)

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

    def mmean(key, ivs):
        return mets.mean(keys[key], ivs) if keys[key] else None

    disjoint = [x for x in STAGES_BY[wl] if x in w]
    overlap = [n for n, _ in OVERLAP_BY[wl] if n in w] + sorted(n for n in w if n.startswith("hold:"))
    order = disjoint + overlap + ["whole"] if "whole" in w else disjoint + overlap
    stages, per_kernel, per_family = [], {}, {}
    for name in order:
        ivs = w[name]
        wall = total(ivs) / 1e9
        ksum, nl, kk = kernels_in(ivs)
        per_kernel[name] = kk
        fam = {}
        for k, (_n, ks) in kk.items():
            fam[family_of(k)] = fam.get(family_of(k), 0.0) + ks
        per_family[name] = fam
        vram, util = smi_in(smi, ivs)
        row = {"stage": name, "wall_s": wall, "ranges": len(ivs),
               "busy_pct": 100.0 * covered_ivs(amerged, ivs) / max(total(ivs), 1),
               "kernel_busy_pct": 100.0 * covered_ivs(kmerged, ivs) / max(total(ivs), 1),
               "copy_busy_pct": 100.0 * covered_ivs(cmerged, ivs) / max(total(ivs), 1),
               "kernel_sum_s": ksum, "launches": nl, "tasks_open": tasks.mean(ivs), "holds_open": holdsfn.mean(ivs),
               "vram_max_mib": vram, "smi_util_pct": util}
        c = cpu_st.get(name)
        row["cpu_cores"] = (c[0] / c[1]) if c and c[1] > 0 else None
        row["cpu_s"] = c[0] if c else None
        for key, _ in KEY_METRICS:
            row[key] = mmean(key, ivs)
        top = sorted(kk.items(), key=lambda kv: -kv[1][1])[:5]
        row["top"] = " ; ".join(f"{k} {v[1]:.2f}s ({100 * v[1] / ksum:.0f}%)" for k, v in top) if ksum else ""
        stages.append(row)
    cols = ["stage", "wall_s", "ranges", "busy_pct", "kernel_busy_pct", "copy_busy_pct", "kernel_sum_s", "launches",
            "tasks_open", "holds_open", "vram_max_mib", "smi_util_pct", "cpu_cores", "cpu_s"] + \
        [k for k, _ in KEY_METRICS] + ["top"]

    def scell(c, v):
        return f"{v:.4f}" if c.endswith("_s") and isinstance(v, float) else cell(v)

    write_tsv(os.path.join(a.out, f"runa-{wl}-stages.tsv"), [cols] + [[scell(c, r.get(c)) for c in cols]
                                                                     for r in stages])
    rows = [["stage", "kernel", "family", "launches", "sum_s", "share_of_stage_kernel_s"]]
    for st in stages:
        for k, (n, sm) in sorted(per_kernel.get(st["stage"], {}).items(), key=lambda kv: -kv[1][1]):
            if sm >= 0.001:
                rows.append([st["stage"], k, family_of(k), n, f"{sm:.4f}",
                             f"{100 * sm / st['kernel_sum_s']:.2f}" if st["kernel_sum_s"] else ""])
    write_tsv(os.path.join(a.out, f"runa-{wl}-stage-kernels.tsv"), rows)
    fams = sorted({f for d in per_family.values() for f in d}, key=lambda f: -sum(d.get(f, 0.0) for n, d in
                                                                                  per_family.items() if n in disjoint))
    write_tsv(os.path.join(a.out, f"runa-{wl}-stage-families.tsv"),
              [["stage", "kernel_sum_s"] + [f"{f}_s" for f in fams]] +
              [[st["stage"], f"{st['kernel_sum_s']:.4f}"] + [f"{per_family[st['stage']].get(f, 0.0):.4f}" for f in fams]
               for st in stages])
    if mets.series:
        mrows = [["stage", "metric", "mean", "samples"]]
        for st in order:
            for n in sorted(mets.series):
                v = mets.mean(n, w[st])
                mrows.append([st, n, "" if v is None else f"{v:.3f}", mets.samples(n, w[st])])
        write_tsv(os.path.join(a.out, f"runa-{wl}-gpu-metrics.tsv"), mrows)

    # every card hold: one LFM proof's device phase (the permit admits one holder at a time)
    holds = sorted((s, e, lab[len(HOLD_PREFIX):]) for s, e, lab, *_ in nvtx if lab.startswith(HOLD_PREFIX))
    hrows = [["idx", "phase", "stage", "start_s", "wall_s", "kernel_busy_pct", "kernel_s", "launches", "sm_active",
              "warps_in_flight", "dram_read", "top_kernels"]]
    hsum = {}
    for i, (s0, e0, ph) in enumerate(holds):
        ksum, nl, kk = kernels_in([(s0, e0)])
        busy = 100.0 * covered(kmerged, s0, e0) / max(e0 - s0, 1)
        stg = stage_at(segs, sstarts, s0) if segs else "whole"
        top = sorted(kk.items(), key=lambda kv: -kv[1][1])[:3]
        hrows.append([i, ph, stg, f"{s0 / 1e9:.3f}", f"{(e0 - s0) / 1e9:.4f}", fmt(busy), f"{ksum:.4f}", nl,
                      fmt(mmean("sm_active", [(s0, e0)])), fmt(mmean("warps_in_flight", [(s0, e0)])),
                      fmt(mmean("dram_read", [(s0, e0)])),
                      " ; ".join(f"{k} {v[1] * 1e3:.1f}ms" for k, v in top)])
        d = hsum.setdefault((ph, stg), {"n": 0, "wall": 0.0, "busy_ns": 0.0, "k": 0.0, "busy": [], "kk": {}})
        d["n"] += 1
        d["wall"] += (e0 - s0) / 1e9
        d["busy_ns"] += covered(kmerged, s0, e0)
        d["k"] += ksum
        d["busy"].append(busy)
        for k, (n, sm) in kk.items():
            x = d["kk"].setdefault(k, [0, 0.0])
            x[0] += n
            x[1] += sm
    write_tsv(os.path.join(a.out, f"runa-{wl}-holds.tsv"), hrows)
    hkrows = [["phase", "stage", "kernel", "family", "launches", "kernel_s", "share_of_held_kernel_s"]]
    for (ph, stg), d in sorted(hsum.items()):
        for k, (n, sm) in sorted(d["kk"].items(), key=lambda kv: -kv[1][1]):
            if sm >= 0.0005:
                hkrows.append([ph, stg, k, family_of(k), n, f"{sm:.4f}", f"{100 * sm / d['k']:.2f}" if d["k"] else ""])
    write_tsv(os.path.join(a.out, f"runa-{wl}-hold-kernels.tsv"), hkrows)

    # the kernels each NVTX label launched: push/pop = the launch call on the label's thread inside its
    # range; process-wide = any launch call inside its window
    by_lab, pw_lab = {}, {}
    for s, e, lab, t, pw in nvtx:
        if pw:
            pw_lab.setdefault(lab, []).append((s, e))
        else:
            by_lab.setdefault(lab, {}).setdefault(t, []).append((s, e))
    for lab in by_lab:
        for t in by_lab[lab]:
            by_lab[lab][t].sort()
    attr = {}
    for lab, per_t in by_lab.items():
        d = {"kind": "thread", "ranges": sum(len(v) for v in per_t.values()),
             "range_s": sum(total(union(v)) for v in per_t.values()) / 1e9,
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
    for lab, ivs in pw_lab.items():
        u = union(ivs)
        us = [x[0] for x in u]
        d = {"kind": "process", "ranges": len(ivs), "range_s": total(u) / 1e9, "threads": "-", "launches": 0,
             "kernel_s": 0.0, "k": {}}
        for s, e, k, api_s, _gt in kernels:
            if api_s is None:
                continue
            i = bisect.bisect_right(us, api_s) - 1
            if i >= 0 and u[i][0] <= api_s < u[i][1]:
                d["launches"] += 1
                d["kernel_s"] += (e - s) / 1e9
                kd = d["k"].setdefault(k, [0, 0.0])
                kd[0] += 1
                kd[1] += (e - s) / 1e9
        attr[lab + ("" if lab not in attr else " (process)")] = d
    arows = [["label", "kind", "ranges", "threads", "range_s", "launches", "kernel_s", "top_kernels"]]
    krows = [["label", "kernel", "launches", "kernel_s"]]
    for lab, d in sorted(attr.items(), key=lambda kv: -kv[1]["kernel_s"]):
        top = sorted(d["k"].items(), key=lambda kv: -kv[1][1])
        arows.append([lab, d["kind"], d["ranges"], d["threads"], f"{d['range_s']:.3f}", d["launches"],
                      f"{d['kernel_s']:.3f}", " ; ".join(f"{k} {v[1]:.2f}s" for k, v in top[:5])])
        for k, (n, sm) in top:
            krows.append([lab, k, n, f"{sm:.4f}"])
    write_tsv(os.path.join(a.out, f"runa-{wl}-nvtx-kernels.tsv"), arows)
    write_tsv(os.path.join(a.out, f"runa-{wl}-nvtx-label-kernels.tsv"), krows)

    # CUDA API time per category, thread and stage (by the call's start)
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
    stl = disjoint or ["whole"]
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

    # the host's CPU per stage and thread role
    if cpu is not None:
        role_names = sorted({r for r, _ in cpu_roles}, key=lambda r: -sum(v for (rr, _), v in cpu_roles.items()
                                                                          if rr == r))
        write_tsv(os.path.join(a.out, f"cpu-{wl}-by-stage.tsv"),
                  [["stage", "wall_s", "cpu_s", "cores_busy"] + [f"{r}_s" for r in role_names]] +
                  [[st, f"{total(w[st]) / 1e9:.3f}", f"{cpu_st.get(st, [0.0, 0.0])[0]:.3f}",
                    fmt(cpu_st[st][0] / cpu_st[st][1], 2) if st in cpu_st and cpu_st[st][1] else "-"] +
                   [f"{cpu_roles.get((r, st), 0.0):.3f}" for r in role_names] for st in disjoint])

    # the time series
    binn = max(int(a.bin_ms * 1e6), 1_000_000)
    ts_cols = ["t_s", "stage", "kernel_busy_pct", "copy_busy_pct", "tasks_open", "holds_open", "vram_mib",
               "smi_util_pct", "cpu_cores"] + [k for k, _ in KEY_METRICS]
    ts_rows = [ts_cols]
    smi_t = [x[0] for x in smi]
    cdel = cpu_deltas(cpu["proc"]) if cpu else []
    ct = [x[0] for x in cdel]
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
        cs = cdel[bisect.bisect_left(ct, b0):bisect.bisect_left(ct, b1)]
        cores = (sum(x[2] for x in cs) / cpu["clk"]) / (sum(x[1] for x in cs) / 1e9) if cs and sum(x[1] for x in cs) \
            else None
        r = [f"{b0 / 1e9:.3f}", st, fmt(100.0 * covered(kmerged, b0, b1) / binn), fmt(100.0 * covered(cmerged, b0, b1) / binn),
             fmt(tasks.mean(iv), 2), fmt(holdsfn.mean(iv), 2), fmt(vr, 0), fmt(ut, 0), fmt(cores, 2)]
        r += [fmt(mets.mean(keys[k], iv)) if keys[k] else "-" for k, _ in KEY_METRICS]
        ts_rows.append(r)
    write_tsv(os.path.join(a.out, f"runa-{wl}-timeseries.tsv"), ts_rows)

    # the markdown
    o = io.StringIO()
    o.write(f"# Run A, {wl}: the workload under Nsight Systems\n\n")
    o.write(f"{base_line(lg)}.\n\n")
    o.write(f"trace: {len(kernels)} kernel launches, {len(copies)} copies/memsets, {len(api)} CUDA API calls, "
            f"{len(nvtx)} NVTX ranges, {end / 1e9:.1f} s from the session start; {len(holds)} card holds in the trace, "
            f"{len(lg['holds'])} CARD HOLD lines in the log. Stages from {src}. CPU: "
            f"{'sampled' if cpu else 'no sampler file'}"
            f"{' · WATCHDOG: ' + cpu['watchdog'] if cpu and cpu.get('watchdog') else ''}.\n\n")
    o.write("busy % = any kernel or copy on the card; kernel/copy busy % = any kernel / any copy or memset. holds = "
            "the mean number of card holds open (0 or 1: the permit admits one). cores = the profiled process's CPU "
            "seconds per wall second (the sampler's 100 ms ticks). VRAM = the nvidia-smi maximum in the stage (200 ms "
            f"samples). GPU metrics are nsys's samples averaged over the stage ({mets.why or 'collected'}): warps = "
            "Compute Warps in Flight, % of the card's warp slots (an occupancy proxy over time). The rows after the "
            "disjoint stages overlap them (one range per group, or per hold); `whole` is the run.\n\n")
    o.write("| stage | wall s | busy % | kernel busy % | copy busy % | Σ kernel s | launches | holds | cores | "
            "VRAM MiB | SM active % | SM issue % | warps % | DRAM rd % | DRAM wr % | PCIe rx % | PCIe tx % | "
            "top kernels (Σ s in the stage) |\n")
    o.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for r in stages:
        o.write(f"| {r['stage']} | {fmt(r['wall_s'], 2)} | {fmt(r['busy_pct'])} | {fmt(r['kernel_busy_pct'])} | "
                f"{fmt(r['copy_busy_pct'])} | {fmt(r['kernel_sum_s'], 2)} | {r['launches']} | {fmt(r['holds_open'], 2)} | "
                f"{fmt(r['cpu_cores'], 1)} | {fmt(r['vram_max_mib'], 0)} | {fmt(r['sm_active'])} | "
                f"{fmt(r['sm_issue'])} | {fmt(r['warps_in_flight'])} | {fmt(r['dram_read'])} | {fmt(r['dram_write'])} | "
                f"{fmt(r['pcie_rx'])} | {fmt(r['pcie_tx'])} | {r['top']} |\n")
    if fams:
        o.write("\n## kernel seconds by family and stage\n\n")
        o.write("| stage | Σ kernel s | " + " | ".join(fams) + " |\n|---|---|" + "---|" * len(fams) + "\n")
        for st in stages:
            o.write(f"| {st['stage']} | {st['kernel_sum_s']:.2f} | " +
                    " | ".join(f"{per_family[st['stage']].get(f, 0.0):.2f}" for f in fams) + " |\n")
    if hsum:
        o.write("\n## card holds (each an LFM proof's device phase; one holder at a time)\n\n")
        o.write("busy % = kernel time covered / held time; idle s = held time with no kernel on the card. Per hold "
                f"in runa-{wl}-holds.tsv, its kernels in runa-{wl}-hold-kernels.tsv.\n\n")
        o.write("| phase | stage | holds | held s | Σ kernel s | busy % | idle s | busy % min / median / max | "
                "top kernels (Σ s) |\n|---|---|---|---|---|---|---|---|---|\n")
        for (ph, stg), d in sorted(hsum.items(), key=lambda kv: -kv[1]["wall"]):
            b = sorted(d["busy"])
            top = sorted(d["kk"].items(), key=lambda kv: -kv[1][1])[:4]
            o.write(f"| {ph} | {stg} | {d['n']} | {d['wall']:.2f} | {d['k']:.2f} | "
                    f"{100 * d['busy_ns'] / 1e9 / d['wall'] if d['wall'] else 0:.1f} | "
                    f"{d['wall'] - d['busy_ns'] / 1e9:.2f} | {b[0]:.0f} / {b[len(b) // 2]:.0f} / {b[-1]:.0f} | "
                    + " ; ".join(f"{k} {v[1]:.2f}" for k, v in top) + " |\n")
    if cpu is not None and cpu_st:
        o.write("\n## the host's CPU per stage (cores busy = CPU seconds / wall second) and by thread role\n\n")
        top_roles = sorted({r for r, _ in cpu_roles}, key=lambda r: -sum(v for (rr, _), v in cpu_roles.items()
                                                                         if rr == r))[:8]
        o.write("A role is the thread's NVTX range (executor, walker, builder, layout on the WHIR block) or its "
                "name less a trailing number; an unnamed thread carries its creator's name.\n\n")
        o.write("| stage | wall s | CPU s | cores | " + " | ".join(top_roles) + " |\n|---|---|---|---|" +
                "---|" * len(top_roles) + "\n")
        for st in disjoint:
            c = cpu_st.get(st)
            o.write(f"| {st} | {total(w[st]) / 1e9:.2f} | {c[0] if c else 0:.2f} | "
                    f"{fmt(c[0] / c[1], 2) if c and c[1] else '-'} | " +
                    " | ".join(f"{cpu_roles.get((r, st), 0.0):.2f}" for r in top_roles) + " |\n")
    if attr:
        o.write("\n## kernels by the NVTX label that launched them\n\n")
        o.write("thread: a kernel belongs to a push/pop label when its launch call ran on the label's thread inside "
                "one of its ranges. process: a launch call inside a process-wide range's window, from any thread.\n\n")
        o.write("| label | kind | ranges | threads | range s | launches | Σ kernel s | top kernels |\n"
                "|---|---|---|---|---|---|---|---|\n")
        for r in arows[1:]:
            o.write("| " + " | ".join(str(x) for x in r) + " |\n")
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
    print(f"RUNA {wl}: stages from {src}; card holds {len(holds)} (log {len(lg['holds'])}); recommit ranges {nrec}; "
          f"cpu {'yes' if cpu else 'no'}")
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


def in_window(row, windows, api_ns):
    """The launch call is inside the pass's process-wide NVTX range (ncu --nvtx-include "<name>"); a pass
    without one takes every launch."""
    if row["nvtx"] == "-":
        return True
    u = windows.get(row["nvtx"], [])
    i = bisect.bisect_right(u, (api_ns, float("inf"))) - 1
    return i >= 0 and u[i][0] <= api_ns < u[i][1]


def select(row, launches, windows=None):
    """(matched, profiled): the launches the pass's regex matches inside its NVTX window, and the ones
    ncu would profile.
    window: ncu's default filter: skip `skip` matching launches, profile the next `count` (then
            --kill ends the run).
    config: --filter-mode per-launch-config, whose key is the launch's grid, block and shared
            memory: skip/count per key. A config pass names one kernel, so the key never mixes two."""
    rx = kernel_re(row)
    windows = windows or {}
    m = [r for r in launches if rx.fullmatch(r["kernel"]) and in_window(row, windows, r["api_ns"])]
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


def plan_for(path, workload, modes):
    return [r for r in read_plan(path) if r["workload"] == workload and modes in r["modes"].split(",")]


def cmd_dry(a):
    plan = plan_for(a.plan, a.workload, a.modes)
    launches = load_trace(a.sqlite)
    db = open_ro(a.sqlite)
    tbls = tables(db)
    nvtx = load_nvtx(db, tbls)
    end = max([r["k_start_ns"] + r["dur_us"] * 1e3 for r in launches] + [0])
    w, src = trace_windows(nvtx, end, a.workload)
    segs = partition(w, a.workload)
    sstarts = [s[0] for s in segs]
    windows = {}
    for s, e, lab, _t, pw in nvtx:
        if pw:
            windows.setdefault(lab, []).append((s, e))
    windows = {k: union(v) for k, v in windows.items()}
    for r in launches:
        r["stage"] = stage_at(segs, sstarts, r["api_ns"]) if segs else "whole"
    wall = end / 1e9
    os.makedirs(a.out, exist_ok=True)
    tot = sum(r["dur_us"] for r in launches) or 1.0
    per_kernel = {}
    for r in launches:
        k = per_kernel.setdefault(r["kernel"], {"n": 0, "us": 0.0, "cfg": set(), "stages": {}})
        k["n"] += 1
        k["us"] += r["dur_us"]
        k["cfg"].add((r["grid"], r["block"], r["smem"]))
        k["stages"][r["stage"]] = k["stages"].get(r["stage"], 0.0) + r["dur_us"]
    tag = a.tag or a.workload
    o = io.StringIO()
    o.write(f"# The pass plan against a trace: {tag} ({a.modes} mode)\n\n")
    o.write(f"trace: {len(launches)} kernel launches, {len(per_kernel)} kernels, Σ kernel time {tot / 1e6:.2f} s, "
            f"{wall:.1f} s of trace; stages from {src}; process-wide NVTX ranges: "
            f"{', '.join(f'{k} x{len(v)}' for k, v in sorted(windows.items())) or 'none'}.\n\n")
    o.write("## per pass: what ncu would profile\n\n")
    o.write("| pass | mode | nvtx | skip/count | matched launches | kernels matched | configs | would profile | "
            "stages of those | shapes of those | est. s | verdict |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    tsv = [["pass", "mode", "nvtx", "skip", "count", "matched", "kernels", "configs", "profile", "est_s", "verdict"]]
    detail = io.StringIO()
    rc = 0
    est_total = 0.0
    for row in plan:
        m, pa = select(row, launches, windows)
        names = {}
        for r in m:
            names[r["kernel"]] = names.get(r["kernel"], 0) + 1
        cfgs = len({(r["kernel"], r["grid"], r["block"], r["smem"]) for r in m})
        st, shp = {}, {}
        for r in pa:
            st[r["stage"]] = st.get(r["stage"], 0) + 1
            sh = shape(r["grid"], r["block"])
            shp[sh] = shp.get(sh, 0) + 1
        if row["nvtx"] != "-" and row["nvtx"] not in windows:
            verdict = "NO NVTX RANGE"
        else:
            verdict = "ok" if pa else ("NO MATCH" if not m else "NOTHING SELECTED")
        if verdict != "ok":
            rc = 1
        est = est_seconds(row, pa, launches, wall)
        est_total += est
        shapes_s = ", ".join(f"{k}{' x' + str(v) if v > 1 else ''}" for k, v in
                             sorted(shp.items(), key=lambda kv: -kv[1])[:4]) + (" …" if len(shp) > 4 else "")
        o.write(f"| {row['pass']} | {row['mode']} | {row['nvtx']} | {row['skip']}/{row['count']} | {len(m)} | "
                f"{len(names)} | {cfgs} | {len(pa)} | {', '.join(f'{k} {v}' for k, v in sorted(st.items()))} | "
                f"{shapes_s} | {est:.0f} | {verdict} |\n")
        tsv.append([row["pass"], row["mode"], row["nvtx"], row["skip"], row["count"], len(m), len(names), cfgs,
                    len(pa), f"{est:.0f}", verdict])
        detail.write(f"\n### {row['pass']} ({row['mode']}, nvtx {row['nvtx']}, kernels `{row['kernels']}`)\n\n")
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
    o.write(f"\nconfigs = distinct (kernel, grid, block, shared memory) among the matched. est. s = a guide for the "
            f"counters box (see est_seconds); Σ {est_total:.0f} s for these {len(plan)} passes.\n")
    o.write("\n## kernels by time, and which pass covers each\n\n| kernel | family | launches | configs | Σ s | % | "
            "stages (Σ s) | passes |\n|---|---|---|---|---|---|---|---|\n")
    uncovered = []
    for name, k in sorted(per_kernel.items(), key=lambda kv: -kv[1]["us"]):
        cov = [row["pass"] for row in plan if matches(row, name)]
        share = 100.0 * k["us"] / tot
        if share >= 1.0 and not cov:
            uncovered.append(f"{name} ({share:.1f} %)")
        if share >= 0.1 or cov:
            sts = ", ".join(f"{s} {v / 1e6:.2f}" for s, v in sorted(k["stages"].items(), key=lambda kv: -kv[1])[:4])
            o.write(f"| {name} | {family_of(name)} | {k['n']} | {len(k['cfg'])} | {k['us'] / 1e6:.3f} | {share:.1f} | "
                    f"{sts} | {', '.join(cov) or '-'} |\n")
    o.write(f"\nkernels with >= 1 % of kernel time that no pass covers: {', '.join(uncovered) or 'none'}\n")
    o.write(detail.getvalue())
    with open(os.path.join(a.out, f"dry-{tag}.md"), "w") as f:
        f.write(o.getvalue())
    write_tsv(os.path.join(a.out, f"dry-{tag}.tsv"), tsv)
    for t in tsv[1:]:
        print(f"DRY {tag} {t[0]}: nvtx {t[2]}; matched {t[5]} launches ({t[6]} kernels, {t[7]} configs); would "
              f"profile {t[8]}; est {t[9]} s; {t[10]}")
    print(f"DRY {tag}: {len(plan)} passes, est Σ {est_total:.0f} s; uncovered kernels >= 1 %: "
          f"{', '.join(uncovered) or 'none'}")
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


def log_ok(lg, workload):
    """The pre-registered log gates of one run: the test passed and its headline line is there (the WHIR
    tree's verify accepted it; the STARK tree reached its final check and every child verify)."""
    if not (lg["test_result"] or "").startswith("test result: ok. 1 passed"):
        return False
    if workload == "whir":
        return lg["w3_base"] is not None and lg["w3_rec"] is not None and lg["w3_verify"] == "ACCEPTED"
    return lg["block"] is not None and lg["final_check"]


def cmd_loghead(a):
    """One TSV line for the driver: ok|bad, the test result, CARD HOLD lines, the headline."""
    lg = read_log(a.log)
    print("\t".join([("ok" if log_ok(lg, a.workload) else "bad"), lg["test_result"] or "<none>",
                     str(len(lg["holds"])), base_line(lg)[:600]]))
    return 0


def cmd_report(a):
    s = a.send
    o = io.StringIO()
    o.write("# noepoch_counters.sh: summary\n\n")
    env = os.path.join(s, "env.txt")
    if os.path.exists(env):
        keep = ("script=", "repo=", "whir_", "stark_", "gpu_name=", "driver=", "ncu=", "nsys=", "cpu=", "nvtx_")
        o.write("```\n" + "".join(l for l in open(env) if l.startswith(keep)) + "```\n\n")
    ref = read_tsv(os.path.join(s, "reference", "runs.tsv"))
    if ref:
        o.write("## reference runs (no profiler)\n\n| workload | rc | seconds | headline | CPU | VRAM max MiB |\n"
                "|---|---|---|---|---|---|\n")
        for r in ref:
            o.write(f"| {r['workload']} | {r['rc']} | {r['seconds']} | {r['headline']} | {r.get('cpu', '-')} | "
                    f"{r['vram_max_mib']} |\n")
        o.write("\n")
    for wl in WORKLOADS:
        p = os.path.join(s, "runa", wl, f"runa-{wl}.md")
        if not os.path.exists(p):
            continue
        t = open(p, errors="replace").read()
        o.write(f"## run A, {wl}\n\n")
        o.write(t.split("\n", 2)[2] if t.count("\n") > 2 else t)
        o.write("\n")
    k = md_section(os.path.join(s, "summary", "kernels.md"), "## per kernel, both workloads")
    if k:
        o.write("## run B: " + k.split(" ", 1)[1] + "\n")
    passes = read_tsv(os.path.join(s, "passes.tsv"))
    if passes:
        o.write("## run B passes\n\n| pass | workload | nvtx | mode | profiled | seconds | verdict |\n"
                "|---|---|---|---|---|---|---|\n")
        for r in passes:
            o.write(f"| {r['pass']} | {r['workload']} | {r['nvtx']} | {r['mode']} | {r['profiled']} | {r['seconds']} | "
                    f"{r['verdict']} |\n")
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
# cpu-sample and cpu-total: the profiled process's CPU and the host's memory, beside a run


def proc_stat(path):
    """(comm, utime + stime ticks, threads, rss pages) from a /proc stat file."""
    with open(path) as f:
        t = f.read()
    left, right = t.index("("), t.rindex(")")
    rest = t[right + 2:].split()
    return t[left + 1:right], int(rest[11]) + int(rest[12]), int(rest[17]), int(rest[21])


def mem_avail_mib():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return -1


def find_pid(exe, me):
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) == me:
            continue
        try:
            if os.path.realpath(f"/proc/{d}/exe") == exe:
                return int(d)
        except OSError:
            continue
    return None


def cmd_cpu_sample(a):
    """Waits for the process running --exe (under nsys or ncu it is their child), then every interval
    writes its cumulative CPU ticks, RSS, thread count and the host's MemAvailable (PREFIX.proc.tsv) and
    each thread whose ticks moved (PREFIX.tid.tsv), until it exits. Below --mem-floor-mib of MemAvailable
    it sends the process SIGTERM, then SIGKILL 10 s later, and says so in PREFIX.watchdog: a run that
    would push the machine into swap or the OOM killer ends instead."""
    exe = os.path.realpath(a.exe)
    clk = os.sysconf("SC_CLK_TCK")
    page_kb = os.sysconf("SC_PAGE_SIZE") // 1024
    me = os.getpid()
    deadline = time.time() + a.wait_s
    pid = None
    while pid is None:
        pid = find_pid(exe, me)
        if pid is None:
            if time.time() > deadline:
                sys.stderr.write(f"cpu-sample: no process ran {exe} within {a.wait_s} s\n")
                return 1
            time.sleep(0.05)
    term_at, killed = None, False
    with open(a.out + ".proc.tsv", "w") as fp, open(a.out + ".tid.tsv", "w") as ft:
        fp.write(f"# clk_tck {clk}\nepoch_ns\tpid\tticks\trss_kb\tthreads\tmem_avail_mib\n")
        ft.write("epoch_ns\ttid\tcomm\tticks\n")
        last = {}
        while True:
            now = time.time_ns()
            try:
                _comm, ticks, nth, rss = proc_stat(f"/proc/{pid}/stat")
            except (OSError, ValueError, IndexError):
                break
            avail = mem_avail_mib()
            fp.write(f"{now}\t{pid}\t{ticks}\t{rss * page_kb}\t{nth}\t{avail}\n")
            try:
                tids = os.listdir(f"/proc/{pid}/task")
            except OSError:
                tids = []
            for t in tids:
                try:
                    tc, tt, _, _ = proc_stat(f"/proc/{pid}/task/{t}/stat")
                except (OSError, ValueError, IndexError):
                    continue
                if last.get(t) != tt:
                    ft.write(f"{now}\t{t}\t{tc}\t{tt}\n")
                    last[t] = tt
            fp.flush()
            ft.flush()
            if a.mem_floor_mib and 0 <= avail < a.mem_floor_mib and term_at is None:
                with open(a.out + ".watchdog", "w") as f:
                    f.write(f"MemAvailable {avail} MiB under the floor {a.mem_floor_mib} MiB at epoch_ns {now}: "
                            "the profiled process was sent SIGTERM (SIGKILL 10 s later if still running)\n")
                sys.stderr.write(f"cpu-sample: WATCHDOG: MemAvailable {avail} MiB < {a.mem_floor_mib} MiB; "
                                 "ending the profiled process\n")
                try:
                    os.kill(pid, signal.SIGTERM)
                except OSError:
                    pass
                term_at = time.time()
            elif term_at is not None and not killed and time.time() - term_at > 10:
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
                killed = True
            time.sleep(a.interval_ms / 1000.0)
    return 0


def cpu_total_line(prefix):
    cpu = read_cpu(prefix, 0)
    if not cpu or len(cpu["proc"]) < 2:
        return "no CPU samples"
    p = cpu["proc"]
    secs = (p[-1][1] - p[0][1]) / cpu["clk"]
    wall = (p[-1][0] - p[0][0]) / 1e9
    peak = max(x[2] for x in p) / 1048576
    avail = [x[4] for x in p if x[4] >= 0]
    line = (f"CPU {secs:.1f} s over {wall:.1f} s ({secs / wall if wall else 0:.1f} cores), peak RSS {peak:.1f} GiB, "
            f"min MemAvailable {min(avail) / 1024 if avail else -1:.1f} GiB")
    if cpu.get("watchdog"):
        line += " · WATCHDOG: " + cpu["watchdog"]
    return line


def cmd_cpu_total(a):
    print(cpu_total_line(a.cpu))
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
    """A small nsys-shaped trace of one no-epoch WHIR block and its tree, session start T0_SYNTH (trace
    seconds): setup [0, 1), blk_base [1, 9) with blk_phase_a [1.5, 5), blk_prepared [5, 5.2) and
    blk_phase_b [5.5, 8.5); lfm_tree [9.2, 9.8) holding one cardhold_multi_prove [9.3, 9.6);
    harness_verify [9.8, 10). A blk_walker push/pop range on the driver thread names it."""
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
    names = ["ntt_cm_dit_k8", "rpx_merkle_level", "rpx_leaves_base_coset", "sumcheck_round_ext3",
             "cuLaunchKernel", "cuStreamSynchronize", "cuMemAlloc_v2", "cuMemcpyDtoHAsync_v2", "cuMemHostAlloc",
             "driver-3", "blk_phase_b", "whir_fold_ext3", "ccomp_0b8d15837e1e77a3"]
    for i, n in enumerate(names, 1):
        db.execute("INSERT INTO StringIds VALUES (?, ?)", (i, n))
    db.execute("INSERT INTO TARGET_INFO_SESSION_START_TIME VALUES (?, 'x', 'x')", (T0_SYNTH,))
    db.execute("INSERT INTO ThreadNames VALUES (10, 0, ?)", (DRV_TID,))
    sec = 10 ** 9

    def rng(a, b, text, tid, et=60, text_id=None):
        db.execute("INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, 0, 0, 0, ?, ?, ?, ?, 0)",
                   (int(a * sec), int(b * sec), et, text, tid, tid, text_id))

    rng(1.0, 9.0, "blk_base", MAIN_TID)
    rng(1.5, 5.0, "blk_phase_a", MAIN_TID)
    rng(3.0, 4.5, "blk_a_commit", MAIN_TID)
    rng(5.0, 5.2, "blk_prepared", MAIN_TID)
    rng(5.5, 8.5, None, MAIN_TID, 60, 11)          # a registered string: the text through StringIds
    rng(5.5, 7.0, "blk_b_argue", MAIN_TID)
    rng(9.2, 9.8, "lfm_tree", MAIN_TID)
    rng(9.3, 9.6, "cardhold_multi_prove", DRV_TID)
    rng(9.8, 10.0, "harness_verify", MAIN_TID)
    rng(1.5, 5.0, "blk_walker", DRV_TID, 59)       # a thread's role, push/pop
    # (kernel, grid, t launch, duration ms, thread): the launch call starts 1 µs before the kernel
    launches = [("ntt_cm_dit_k8", 1024, 3.0, 100.0, MAIN_TID), ("rpx_leaves_base_coset", 4096, 3.5, 200.0, MAIN_TID),
                ("sumcheck_round_ext3", 253, 6.0, 300.0, MAIN_TID), ("whir_fold_ext3", 512, 7.0, 400.0, DRV_TID),
                ("ntt_cm_dit_k8", 2048, 7.6, 50.0, MAIN_TID),
                ("ccomp_0b8d15837e1e77a3", 512, 9.4, 100.0, DRV_TID), ("rpx_merkle_level", 256, 9.85, 10.0, MAIN_TID)]
    for c, (k, gx, ts, dms, tid) in enumerate(launches, 1):
        s = int(ts * sec)
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, 0, ?, ?, 5, 0, 0)", (s - 1000, s, tid, c))
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, 0, ?, ?, ?, 1, 1, 128, 1, 1, 0, 0, 40)",
                   (s, s + int(dms * 1e6), c, names.index(k) + 1, gx))
    # API calls without kernels: a 0.3 s sync in phase A, a 0.2 s alloc and a 0.1 s copy in phase B, a pinned
    # alloc in the setup, and the last call at 9.95 s (the trace's end)
    for c, (nm, ts, dur, tid) in enumerate([("cuStreamSynchronize", 4.0, 0.3, MAIN_TID),
                                            ("cuMemAlloc_v2", 5.6, 0.2, DRV_TID),
                                            ("cuMemcpyDtoHAsync_v2", 7.5, 0.1, DRV_TID),
                                            ("cuMemHostAlloc", 0.5, 0.05, MAIN_TID),
                                            ("cuStreamSynchronize", 9.9, 0.05, MAIN_TID)], 100):
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, 0, ?, ?, ?, 0, 0)",
                   (int(ts * sec), int((ts + dur) * sec), tid, c, names.index(nm) + 1))
    db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?, ?, 1, 1000)", (int(1.6 * sec), int(2.1 * sec)))
    db.execute("INSERT INTO TARGET_INFO_GPU_METRICS VALUES (7, 0, 'syn', 1, 'SMs Active [Throughput %]')")
    db.execute("INSERT INTO TARGET_INFO_GPU_METRICS VALUES (7, 0, 'syn', 2, 'Compute Warps in Flight [Throughput %]')")
    for tenth in range(0, 100):               # 10 Hz: SM active 80 in phase A, 30 in phase B, 0 elsewhere
        ts = tenth * 10 ** 8
        v = 80 if 15 <= tenth < 50 else (30 if 55 <= tenth < 85 else 0)
        db.execute("INSERT INTO GPU_METRICS VALUES (?, ?, 7, 1, ?)", (ts, ts, v))
        db.execute("INSERT INTO GPU_METRICS VALUES (?, ?, 7, 2, ?)", (ts, ts, v // 2))
    db.commit()
    db.close()


def write_synth_cpu(prefix):
    """The sampler's files for the synthetic run: every 0.1 s, 20 ticks (2 cores at 100 Hz) inside phase A
    [1.5, 5), 5 ticks elsewhere; the driver thread (tid 101, the walker) 10 ticks a sample in phase A."""
    with open(prefix + ".proc.tsv", "w") as fp, open(prefix + ".tid.tsv", "w") as ft:
        fp.write("# clk_tck 100\nepoch_ns\tpid\tticks\trss_kb\tthreads\tmem_avail_mib\n")
        ft.write("epoch_ns\ttid\tcomm\tticks\n")
        ticks = tt = 0
        for tenth in range(0, 101):
            t = T0_SYNTH + tenth * 10 ** 8
            fp.write(f"{t}\t4242\t{ticks}\t{1048576 * (1 + tenth // 50)}\t40\t30000\n")
            ft.write(f"{t}\t101\tlfm::whir_bloc\t{tt}\n")
            ft.write(f"{t}\t102\telf-beside-3\t{tenth}\n")
            mid = tenth + 0.5
            ticks += 20 if 15 <= mid < 50 else 5
            tt += 10 if 15 <= mid < 50 else 0


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
    ok(intersect([(0, 10)], [(2, 3), (5, 12)]) == [(2, 3), (5, 10)], "intersect")
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
    ok(family_of("sumcheck_round_ext3") == "sumcheck" and family_of("whir_fold_ext3") == "whir-fold"
       and family_of("ccomp_0b8d15837e1e77a3") == "quotient" and family_of("rpx_leaves_base_coset") == "leaves",
       "kernel families")
    ok(comm_family("elf-beside-3") == "elf-beside" and comm_family("rayon-worker-12") == "rayon-worker",
       "thread-name families")
    hl = HOLD_RE.search("CARD HOLD #3 multi_prove: waited 0.120s · held 0.450s · t=[1800000010.000,1800000010.450]")
    ok(hl is not None and hl.group(2) == "multi_prove" and hl.group(4) == "0.450", "the CARD HOLD line")
    ok(LFM_PROVE_RE.search("   BLOCK L1N0 (4 children) LFM PROVE: execute 0.21s · fill 0.30s · multi_prove 0.45s")
       is not None and LFM_PROVE_RE.search("   BLOCK L0 leaf 3 LFM PROVE: execute 0.21s · fill 0.30s · "
                                           "multi_prove 0.45s").group(1) == "BLOCK L0 leaf 3", "the LFM PROVE line")
    with tempfile.TemporaryDirectory() as d:
        plan = os.path.join(d, "plan.tsv")
        with open(plan, "w") as f:
            f.write("\t".join(PLAN_COLS) + "\n")
            f.write("whir_b_sum\twhir\tquick,full\tconfig\t0\t1\tblk_phase_b\tsumcheck_round_ext3\tsumcheck\tone per shape\n")
            f.write("whir_a_ntt\twhir\tfull\twindow\t0\t4\tblk_phase_a\tntt_cm_di[ft]_k[4-8]\tlde\tphase A only\n")
            f.write("whir_all_ntt\twhir\tfull\twindow\t0\t4\t-\tntt_cm_di[ft]_k[4-8]\tlde\tno window\n")
            f.write("whir_b_leaves\twhir\tfull\twindow\t0\t1\tblk_phase_b\trpx_leaves_base_coset\tleaves\tmust match 0\n")
            f.write("whir_none\twhir\tfull\twindow\t0\t1\tno_such_range\tntt_cm_dit_k8\tlde\tmust say NO NVTX RANGE\n")
            f.write("stark_lfm\tstark\tquick,full\twindow\t0\t4\tcardhold_multi_prove\tccomp_[0-9a-f]+\tquotient\tq\n")
        ok(check_plan(read_plan(plan)) == [], f"the synthetic plan is well formed ({check_plan(read_plan(plan))})")
        badplan = os.path.join(d, "bad.tsv")
        with open(badplan, "w") as f:
            f.write("\t".join(PLAN_COLS) + "\n")
            f.write("x\twhir\tfull\twindow\t0\t0\t-\t^rpx$\tf\tn\n")
            f.write("x\tzisk\tsome\tsome\t-1\t1\tbad range!\trpx(\tf\tn\n")
            f.write("y\tstark\tfull\tconfig\t0\t1\t-\tfri_fold_ext3|logup_[a-z0-9_]+\tf\tn\n")
        errs = check_plan(read_plan(badplan))
        ok(len(errs) >= 9 and any("exactly one kernel" in e for e in errs) and any("nvtx is" in e for e in errs)
           and any("modes is" in e for e in errs), f"a malformed plan is refused ({len(errs)} errors: {errs})")
        # summary, in both unit styles, one export per workload
        for base_units in (False, True):
            nd = os.path.join(d, f"ncu{int(base_units)}")
            os.makedirs(nd)
            for p in ("whir_b_sum", "stark_lfm"):
                with open(os.path.join(nd, f"{p}.details.csv"), "w") as f:
                    f.write(synth_details_csv(base_units=base_units))
                with open(os.path.join(nd, f"{p}.stages.tsv"), "w") as f:
                    lab = "blk_phase_b" if p.startswith("whir") else "cardhold_multi_prove"
                    f.write(f"id\tkernel\tstage\tpasses\n0\trpx_leaves_base_row_pair_batched\t{lab}\t18\n"
                            f"1\tntt_cm_dit_k8\t{lab}\t18\n2\tntt_cm_dit_k8\t{lab}\t18\n")
            out = os.path.join(d, f"sum{int(base_units)}")
            rc, _ = run(cmd_summary, argparse.Namespace(plan=plan, ncu_dir=nd, out=out))
            tag = "base units" if base_units else "auto units"
            ok(rc == 0, f"summary exits 0 ({tag})")
            rows = [r for r in read_tsv(os.path.join(out, "launches.tsv")) if r["pass"] == "whir_b_sum"]
            ok(len(rows) == 3, f"three launches ({len(rows)}, {tag})")
            r0 = rows[0]
            ok(r0["kernel"] == "rpx_leaves_base_row_pair_batched" and r0["stage"] == "blk_phase_b"
               and r0["elem"] == "leaf", f"launch 0 fields ({tag})")
            ok(abs(fnum(r0["dur_us"]) - 62360.0) < 0.5 and abs(fnum(r0["sm_ghz"]) - 2.01) < 1e-6,
               f"duration and SM clock ({r0['dur_us']}, {r0['sm_ghz']}, {tag})")
            ok(r0["family"] == "leaves" and r0["workload"] == "whir" and r0["bound"] == "compute",
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
            ok(set(kn) == {(w, k) for w in ("whir", "stark") for k in ("rpx_leaves_base_row_pair_batched",
                                                                        "ntt_cm_dit_k8")}, "one row per workload and kernel")
            ntt = kn[("whir", "ntt_cm_dit_k8")]
            wdram = (84.0 * 140.5 + 70.0 * 59.5) / 200.0
            ok(ntt["launches"] == "2" and ntt["configs"] == "2" and abs(fnum(ntt["dram_pct"]) - wdram) < 0.01,
               f"duration-weighted DRAM % ({ntt['dram_pct']} vs {wdram:.2f})")
            ok(ntt["largest_shape"] == "2048x16x1/256x1x1" and ntt["stages"] == "blk_phase_b", "largest launch, stages")
            md = open(os.path.join(out, "kernels.md")).read()
            ok("## per kernel, both workloads (the roofs)" in md and "| ntt_cm_dit_k8 | lde | whir |" in md
               and "| ntt_cm_dit_k8 | lde | stark |" in md and "## whir: one row per kernel" in md,
               "kernels.md: the comparison and the per-workload tables")
            ks = read_tsv(os.path.join(out, "kernels_by_stage.tsv"))
            ok(len(ks) == 4 and any(r["stage"] == "cardhold_multi_prove" and r["kernel"] == "ntt_cm_dit_k8" for r in ks),
               f"kernels by stage ({len(ks)})")
        # stages of a pass = its NVTX window
        log = ['==PROF== Connected to process 1 (<W>/bin)', '==PROF== Profiling "ntt_cm_dit_k8": 0%....100% - 18 passes',
               'W3 BASE: 18.20s', '==PROF== Profiling "ntt_cm_dit_k8" - 1: 0%....100% - 17 passes']
        st = stages_from_lines(log, "blk_phase_b")
        ok([s[2] for s in st] == ["blk_phase_b", "blk_phase_b"] and st[1][3] == "17" and st[1][0] == 1
           and stages_from_lines(log)[0][2] == "run", f"stages from the pass's window ({st})")
        # runa on the synthetic trace, with the sampler's files
        db = os.path.join(d, "t.sqlite")
        build_synth_sqlite(db)
        rlog = os.path.join(d, "a.log")
        with open(rlog, "w") as f:
            f.write("BLOCK PHASES: execute 3.10 · build 7.60 · prep 1.20 · A 10.42 (wait 2.00 upload 1.00 commit 6.00 "
                    "retire 0.40) · B 7.70 (argue 4.00 open 2.50 tax 1.20 = upload 0.40 + encode 0.80)\n"
                    "W3 BASE: 18.18s · statement at 10.50s · plan + 3 leaves emitted by 12.00s (inside the base)\n"
                    "CARD HOLD #0 multi_prove: waited 0.000s · held 0.300s · t=[1800000009.300,1800000009.600]\n"
                    "W3 LEVEL 0: 2.62s wall · per program build+prove 0.10+0.80\n"
                    "W3 RECURSION: 4.16s after the base (tree 4.10s) · whole block 22.34s\n"
                    "W3 TREE VERIFY: ACCEPTED in 3.00s (derives every program and its artifacts)\n"
                    "test result: ok. 1 passed; 0 failed\n")
        smi = os.path.join(d, "smi.csv")
        with open(smi, "w") as f:
            for tenth in range(0, 100, 2):
                ts = datetime.fromtimestamp(T0_SYNTH / 1e9 + tenth / 10).strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]
                f.write(f"{ts}, {30000 if 55 <= tenth < 85 else 1000}, 50\n")
        cpre = os.path.join(d, "cpu")
        write_synth_cpu(cpre)
        rout = os.path.join(d, "runa")
        rc, printed = run(cmd_runa, argparse.Namespace(sqlite=db, log=rlog, workload="whir", out=rout, smi=smi,
                                                       cpu=cpre, bin_ms=500))
        ok(rc == 0 and "| phase_b | 3.00 |" in printed and "Stages from NVTX ranges" in printed
           and "whir base 18.18 s" in printed, f"runa exits 0 with a 3.0 s phase B ({printed[:700]!r})")
        srows = {r["stage"]: r for r in read_tsv(os.path.join(rout, "runa-whir-stages.tsv"))}
        ok(set(srows) == {"setup", "phase_a", "prepared", "phase_b", "base_other", "recursion", "verify", "other",
                          "a_commit", "b_argue", "hold:multi_prove", "whole"}, f"runa's stages ({sorted(srows)})")
        walls = {k: fnum(v["wall_s"]) for k, v in srows.items()}
        ok(abs(walls["setup"] - 1.0) < 1e-6 and abs(walls["phase_a"] - 3.5) < 1e-6 and abs(walls["prepared"] - 0.2) < 1e-6
           and abs(walls["base_other"] - 1.3) < 1e-6 and abs(walls["recursion"] - 0.6) < 1e-6
           and abs(walls["other"] - 0.2) < 1e-6 and abs(walls["verify"] - 0.15) < 1e-6
           and abs(walls["hold:multi_prove"] - 0.3) < 1e-6, f"runa's stage walls ({walls})")
        ok(abs(fnum(srows["phase_a"]["kernel_sum_s"]) - 0.3) < 1e-6 and srows["phase_b"]["launches"] == "3"
           and abs(fnum(srows["phase_a"]["copy_busy_pct"]) - 100 * 0.5 / 3.5) < 0.01,
           f"runa's kernels per stage ({srows['phase_a']['kernel_sum_s']}, {srows['phase_b']['launches']})")
        ok(abs(fnum(srows["phase_a"]["sm_active"]) - 80.0) < 1e-6 and abs(fnum(srows["phase_b"]["sm_active"]) - 30.0) < 1e-6
           and abs(fnum(srows["phase_b"]["warps_in_flight"]) - 15.0) < 1e-6, "runa's GPU metric means per stage")
        ok(abs(fnum(srows["phase_b"]["vram_max_mib"]) - 30000) < 1e-6 and abs(fnum(srows["setup"]["vram_max_mib"]) - 1000) < 1e-6,
           f"runa's VRAM per stage ({srows['phase_b']['vram_max_mib']})")
        ok(abs(fnum(srows["phase_a"]["cpu_cores"]) - 2.0) < 1e-6 and abs(fnum(srows["setup"]["cpu_cores"]) - 0.5) < 1e-6,
           f"runa's cores per stage ({srows['phase_a']['cpu_cores']}, {srows['setup']['cpu_cores']})")
        fam = {r["stage"]: r for r in read_tsv(os.path.join(rout, "runa-whir-stage-families.tsv"))}
        ok(abs(fnum(fam["phase_b"]["sumcheck_s"]) - 0.3) < 1e-6 and abs(fnum(fam["phase_b"]["whir-fold_s"]) - 0.4) < 1e-6
           and abs(fnum(fam["phase_b"]["lde_s"]) - 0.05) < 1e-6, f"kernel families per stage ({fam.get('phase_b')})")
        hold = read_tsv(os.path.join(rout, "runa-whir-holds.tsv"))
        ok(len(hold) == 1 and hold[0]["phase"] == "multi_prove" and hold[0]["stage"] == "recursion"
           and abs(fnum(hold[0]["kernel_busy_pct"]) - 100 / 3) < 0.1 and hold[0]["launches"] == "1",
           f"the card hold ({hold})")
        ok("## card holds" in printed and "| multi_prove | recursion | 1 | 0.30 | 0.10 | 33.3 | 0.20 |" in printed,
           "the card holds' table")
        cst = {r["stage"]: r for r in read_tsv(os.path.join(rout, "cpu-whir-by-stage.tsv"))}
        ok(abs(fnum(cst["phase_a"]["walker_s"]) - 3.5) < 1e-6 and abs(fnum(cst["phase_a"]["cores_busy"]) - 2.0) < 1e-6
           and "elf-beside_s" in cst["phase_a"], f"CPU by stage and role ({cst.get('phase_a')})")
        att = {r["label"]: r for r in read_tsv(os.path.join(rout, "runa-whir-nvtx-kernels.tsv"))}
        ok(att.get("blk_phase_b", {}).get("launches") == "3" and att["blk_phase_b"]["kind"] == "process"
           and abs(fnum(att["cardhold_multi_prove"]["kernel_s"]) - 0.1) < 1e-6
           and att.get("blk_walker", {}).get("kind") == "thread",
           f"kernels by NVTX label ({att})")
        cat = {(r["category"], r["stage"]): r for r in read_tsv(os.path.join(rout, "api-whir-by-category.tsv"))}
        ok(abs(fnum(cat[("sync", "phase_a")]["sum_s"]) - 0.3) < 1e-6 and abs(fnum(cat[("alloc", "phase_b")]["sum_s"]) - 0.2) < 1e-6
           and abs(fnum(cat[("copy_async", "phase_b")]["sum_s"]) - 0.1) < 1e-6 and ("host_pinned", "setup") in cat
           and cat[("launch", "phase_b")]["calls"] == "3", f"API seconds per category and stage ({sorted(cat)})")
        thr = read_tsv(os.path.join(rout, "api-whir-by-thread.tsv"))
        ok(any(r["thread"] == "tid 101 (driver-3)" and r["category"] == "alloc" for r in thr), "API per named thread")
        ts = read_tsv(os.path.join(rout, "runa-whir-timeseries.tsv"))
        ok(len(ts) == 20 and ts[6]["stage"] == "phase_a" and ts[12]["stage"] == "phase_b"
           and abs(fnum(ts[6]["cpu_cores"]) - 2.0) < 1e-6, f"the time series ({len(ts)}, {ts[6] if len(ts) > 6 else ''})")
        ok("RUNA whir: stages from NVTX ranges; card holds 1 (log 1)" in printed, "runa's closing line")
        rc, lh = run(cmd_loghead, argparse.Namespace(log=rlog, workload="whir"))
        ok(lh.startswith("ok\ttest result: ok. 1 passed; 0 failed\t1\twhir base 18.18 s"), f"loghead ({lh!r})")
        rc, lh = run(cmd_loghead, argparse.Namespace(log=rlog, workload="stark"))
        ok(lh.startswith("bad\t"), "loghead refuses a log without its workload's headline")
        ok(cpu_total_line(cpre).startswith("CPU 10.2 s over 10.0 s (1.0 cores), peak RSS 3.0 GiB"),
           f"cpu-total ({cpu_total_line(cpre)})")
        # no NVTX table: one window
        import sqlite3
        db2 = os.path.join(d, "t2.sqlite")
        build_synth_sqlite(db2)
        c2 = sqlite3.connect(db2)
        c2.execute("DROP TABLE NVTX_EVENTS")
        c2.commit()
        c2.close()
        rc, printed = run(cmd_runa, argparse.Namespace(sqlite=db2, log=rlog, workload="whir", out=os.path.join(d, "r2"),
                                                       smi=None, cpu=None, bin_ms=1000))
        s2 = {r["stage"]: r for r in read_tsv(os.path.join(d, "r2", "runa-whir-stages.tsv"))}
        ok(rc == 0 and "one window only" in printed and set(s2) == {"whole"}, f"runa without NVTX ({sorted(s2)})")
        # dry against the synthetic trace
        dout = os.path.join(d, "dry")
        rc, printed = run(cmd_dry, argparse.Namespace(plan=plan, workload="whir", sqlite=db, log=rlog, out=dout,
                                                      tag=None, modes="full"))
        ok(rc == 1 and "whir_none: nvtx no_such_range; matched 0 launches" in printed and "NO NVTX RANGE" in printed,
           "dry exits 1 on a pass whose range is not in the trace")
        drows = {r["pass"]: r for r in read_tsv(os.path.join(dout, "dry-whir.tsv"))}
        ok(set(drows) == {"whir_b_sum", "whir_a_ntt", "whir_all_ntt", "whir_b_leaves", "whir_none"},
           "dry evaluates this workload's full-mode passes only")
        ok(drows["whir_b_sum"]["profile"] == "1" and drows["whir_a_ntt"]["matched"] == "1"
           and drows["whir_all_ntt"]["matched"] == "2" and drows["whir_b_leaves"]["verdict"] == "NO MATCH",
           f"dry selections inside the NVTX windows ({drows})")
        rc, printed = run(cmd_dry, argparse.Namespace(plan=plan, workload="whir", sqlite=db, log=rlog, out=dout,
                                                      tag="q", modes="quick"))
        ok(rc == 0 and set(r["pass"] for r in read_tsv(os.path.join(dout, "dry-q.tsv"))) == {"whir_b_sum"},
           "dry --modes quick takes the quick passes")
        rc, printed = run(cmd_dry, argparse.Namespace(plan=plan, workload="stark", sqlite=db, log=rlog, out=dout,
                                                      tag="s", modes="quick"))
        ok(rc == 0 and "stark_lfm: nvtx cardhold_multi_prove; matched 1 launches" in printed,
           f"a hold-filtered pass selects the hold's launch ({printed[:300]!r})")
        # report
        send = os.path.join(d, "send")
        os.makedirs(os.path.join(send, "runa", "whir"))
        os.makedirs(os.path.join(send, "summary"))
        os.makedirs(os.path.join(send, "reference"))
        for fn in os.listdir(rout):
            with open(os.path.join(rout, fn)) as src, open(os.path.join(send, "runa", "whir", fn), "w") as dst:
                dst.write(src.read())
        with open(os.path.join(send, "summary", "kernels.md"), "w") as f:
            f.write(open(os.path.join(d, "sum0", "kernels.md")).read())
        with open(os.path.join(send, "reference", "runs.tsv"), "w") as f:
            f.write("workload\trc\tseconds\theadline\tcpu\tvram_max_mib\nwhir\t0\t60\twhir base 18.20 s\tCPU 1 s\t32110\n")
        rc, _ = run(cmd_report, argparse.Namespace(send=send))
        rep = open(os.path.join(send, "SUMMARY.md")).read()
        ok(rc == 0 and "## run A, whir" in rep and "## run B: per kernel, both workloads" in rep
           and "## reference runs" in rep and "| phase_b | 3.00 |" in rep, "report assembles SUMMARY.md")
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
        # cpu-sample on a real short process (Linux only: it reads /proc)
        if os.path.isdir("/proc/self/task"):
            import shutil
            import subprocess
            sl = shutil.which("sleep")
            if sl:
                exe = os.path.join(d, "np-selftest-sleep")
                shutil.copy(sl, exe)
                child = subprocess.Popen([exe, "1.2"])
                pre = os.path.join(d, "live")
                rc, _ = run(cmd_cpu_sample, argparse.Namespace(exe=exe, out=pre, interval_ms=100.0, mem_floor_mib=0,
                                                               wait_s=10.0))
                child.wait()
                live = read_cpu(pre, 0)
                ok(rc == 0 and live is not None and len(live["proc"]) >= 3 and live["tid"],
                   f"cpu-sample follows a live process ({rc}, {len(live['proc']) if live else 0} samples)")
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
    s.add_argument("--nvtx", default="-", help="the pass's process-wide NVTX range, or -")
    s = sub.add_parser("runa")
    s.add_argument("--sqlite", required=True)
    s.add_argument("--log", required=True)
    s.add_argument("--workload", required=True, choices=WORKLOADS)
    s.add_argument("--out", required=True)
    s.add_argument("--smi", help="nvidia-smi samples: timestamp, memory.used, utilization.gpu (csv, noheader)")
    s.add_argument("--cpu", help="the cpu-sample prefix of this run")
    s.add_argument("--bin-ms", type=float, default=100.0, help="the time series' bin (ms, >= 1)")
    s = sub.add_parser("dry")
    s.add_argument("--plan", required=True)
    s.add_argument("--workload", required=True, choices=WORKLOADS)
    s.add_argument("--sqlite", required=True)
    s.add_argument("--log", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--tag", help="names the outputs dry-<tag>.md/.tsv (default: the workload)")
    s.add_argument("--modes", default="full", choices=RUN_MODES, help="the passes of this run mode")
    s = sub.add_parser("cpu-sample")
    s.add_argument("--exe", required=True)
    s.add_argument("--out", required=True, help="the files' prefix: PREFIX.proc.tsv, PREFIX.tid.tsv, PREFIX.watchdog")
    s.add_argument("--interval-ms", type=float, default=100.0)
    s.add_argument("--mem-floor-mib", type=int, default=0, help="0: no watchdog")
    s.add_argument("--wait-s", type=float, default=300.0, help="how long to wait for the process to appear")
    s = sub.add_parser("cpu-total")
    s.add_argument("--cpu", required=True)
    s = sub.add_parser("loghead")
    s.add_argument("--log", required=True)
    s.add_argument("--workload", required=True, choices=WORKLOADS)
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
            "cpu-sample": cmd_cpu_sample, "cpu-total": cmd_cpu_total, "loghead": cmd_loghead,
            "plan-check": cmd_plan_check, "cargo-artifact": cmd_cargo_artifact, "scrub": cmd_scrub,
            "check": cmd_check, "selftest": lambda _a: selftest()}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
