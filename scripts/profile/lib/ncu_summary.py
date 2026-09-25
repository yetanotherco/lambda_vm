#!/usr/bin/env python3
"""Compact per-launch table from Nsight Compute raw-page CSV exports.

    ncu --import K.ncu-rep --page raw --csv > K.raw.csv      # one per profiled kernel
    ncu_summary.py --out DIR K1.raw.csv K2.raw.csv ...        # -> DIR/ncu_summary.{csv,txt}

The raw page is one row per profiled launch, every collected metric a column, with a
units row under the header. This pulls the numbers that decide what bounds a kernel —
speed of light (SM vs memory), issue activity, occupancy achieved vs theoretical and
what limits it, the warp-stall breakdown, the busiest pipes, cache hit rates — into
one row per launch. It reads what is there: a metric a section did not collect is
left blank, never guessed. The authoritative per-kernel text is ncu's own
`--page details` export, which block_profile.sh writes beside this.

Pure standard library (python >= 3.8).
"""
import argparse
import csv
import os
import re
import sys

SCALAR = [  # (column, [metric names, first present wins])
    ("duration_ms", ["gpu__time_duration.sum"]),
    ("sm_throughput_pct", ["sm__throughput.avg.pct_of_peak_sustained_elapsed"]),
    ("mem_throughput_pct", ["gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"]),
    ("dram_throughput_pct", ["gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
                             "dram__throughput.avg.pct_of_peak_sustained_elapsed"]),
    ("issue_active_pct", ["smsp__issue_active.avg.pct_of_peak_sustained_active"]),
    ("ipc_active", ["sm__inst_executed.avg.per_cycle_active", "smsp__inst_executed.avg.per_cycle_active"]),
    ("no_eligible_pct", ["smsp__issue_inst0.avg.pct_of_peak_sustained_active"]),
    ("eligible_warps_per_sched", ["smsp__warps_eligible.avg.per_cycle_active"]),
    ("active_warps_per_sched", ["smsp__warps_active.avg.per_cycle_active"]),
    ("warp_cycles_per_issue", ["smsp__average_warp_latency_per_inst_issued.ratio",
                               "smsp__average_warps_issue_stalled_per_issue_active.ratio"]),
    ("achieved_occupancy_pct", ["sm__warps_active.avg.pct_of_peak_sustained_active"]),
    ("theoretical_occupancy_pct", ["sm__maximum_warps_per_active_cycle_pct"]),
    ("registers_per_thread", ["launch__registers_per_thread"]),
    ("limit_registers", ["launch__occupancy_limit_registers"]),
    ("limit_shared_mem", ["launch__occupancy_limit_shared_mem"]),
    ("limit_warps", ["launch__occupancy_limit_warps"]),
    ("limit_blocks", ["launch__occupancy_limit_blocks"]),
    ("l1_hit_pct", ["l1tex__t_sector_hit_rate.pct"]),
    ("l2_hit_pct", ["lts__t_sector_hit_rate.pct"]),
]
STALL = [re.compile(r"^smsp__average_warps?_issue_stalled_(\w+?)_per_issue_active\.ratio$"),
         re.compile(r"^smsp__average_warp_latency_issue_stalled_(\w+?)\.ratio$")]
PIPE = [re.compile(r"^sm__inst_executed_pipe_(\w+?)\.avg\.pct_of_peak_sustained_active$"),
        re.compile(r"^sm__pipe_(\w+?)_cycles_active\.avg\.pct_of_peak_sustained_active$")]
TIME_UNIT = {"nsecond": 1e-6, "ns": 1e-6, "usecond": 1e-3, "us": 1e-3, "msecond": 1.0, "ms": 1.0,
             "second": 1e3, "s": 1e3}


def num(v):
    if v is None:
        return None
    v = v.strip().replace(",", "")
    if not v or v.lower() in ("n/a", "nan"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def read_raw(path):
    """(rows as dicts, units dict). Tolerates a missing units row and leading ncu banner lines."""
    with open(path, newline="", errors="replace") as f:
        lines = [ln for ln in f if not ln.startswith("==")]
    rdr = list(csv.reader(lines))
    hdr_i = next((i for i, r in enumerate(rdr) if "ID" in r and any(c in r for c in ("Kernel Name", "Function Name"))),
                 None)
    if hdr_i is None:
        return [], {}
    hdr = rdr[hdr_i]
    body = rdr[hdr_i + 1:]
    units = {}
    if body and body[0] and body[0][hdr.index("ID")] == "":
        units = dict(zip(hdr, body[0]))
        body = body[1:]
    return [dict(zip(hdr, r)) for r in body if len(r) >= len(hdr) // 2], units


def summarise(row, units):
    out = {}
    for col, names in SCALAR:
        for n in names:
            if n in row and num(row[n]) is not None:
                v = num(row[n])
                if col == "duration_ms":
                    v *= TIME_UNIT.get(units.get(n, "nsecond").strip(), 1e-6)
                out[col] = v
                break
    stalls, pipes = {}, {}
    for k, v in row.items():
        x = num(v)
        if x is None:
            continue
        for pat in STALL:
            m = pat.match(k)
            if m and m.group(1) not in stalls:
                stalls[m.group(1)] = x
        for pat in PIPE:
            m = pat.match(k)
            if m and m.group(1) not in pipes:
                pipes[m.group(1)] = x
    tot = sum(stalls.values())
    out["top_stalls"] = " ; ".join("{} {:.2f} ({:.0f}%)".format(k, v, 100 * v / tot if tot else 0)
                                   for k, v in sorted(stalls.items(), key=lambda kv: -kv[1])[:5] if v > 0)
    out["top_pipes_pct"] = " ; ".join("{} {:.1f}".format(k, v)
                                      for k, v in sorted(pipes.items(), key=lambda kv: -kv[1])[:4] if v > 0)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("raw", nargs="+", help="ncu --page raw --csv exports")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    cols = ["source", "id", "kernel", "grid", "block"] + [c for c, _ in SCALAR] + ["top_stalls", "top_pipes_pct"]
    rows = []
    for path in a.raw:
        data, units = read_raw(path)
        for r in data:
            s = summarise(r, units)
            s.update({"source": os.path.basename(path), "id": r.get("ID", ""),
                      "kernel": r.get("Kernel Name", r.get("Function Name", "")),
                      "grid": r.get("Grid Size", ""), "block": r.get("Block Size", "")})
            rows.append(s)
    with open(os.path.join(a.out, "ncu_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for s in rows:
            w.writerow({c: ("{:.4g}".format(s[c]) if isinstance(s.get(c), float) else s.get(c, "")) for c in cols})
    fmt = lambda v, p="{:.1f}": "-" if v is None else p.format(v)  # noqa: E731
    L = ["== Nsight Compute: one line per profiled launch (ncu_summary.py) ==",
         "SM% / Mem% / DRAM% = speed of light vs the peak; issue% = issue slots busy; occ = achieved/theoretical",
         "warps; stalls = cycles per issued instruction by reason (share); the kernel's ncu --page details text is",
         "the authority (it has the rule messages).", ""]
    hdr = "{:<42} {:>5} {:>13} {:>11} {:>8} {:>5} {:>5} {:>5} {:>6} {:>11} {:>4}".format(
        "kernel", "id", "grid", "block", "dur_ms", "SM%", "Mem%", "DRAM%", "issue%", "occ", "regs")
    L.append(hdr)
    for s in rows:
        L.append("{:<42} {:>5} {:>13} {:>11} {:>8} {:>5} {:>5} {:>5} {:>6} {:>11} {:>4}".format(
            s["kernel"][:42], s["id"], s["grid"].replace(" ", ""), s["block"].replace(" ", ""),
            fmt(s.get("duration_ms"), "{:.3f}"), fmt(s.get("sm_throughput_pct")), fmt(s.get("mem_throughput_pct")),
            fmt(s.get("dram_throughput_pct")), fmt(s.get("issue_active_pct")),
            "{}/{}".format(fmt(s.get("achieved_occupancy_pct")), fmt(s.get("theoretical_occupancy_pct"))),
            fmt(s.get("registers_per_thread"), "{:.0f}")))
        if s.get("top_stalls"):
            L.append("    stalls: " + s["top_stalls"])
        if s.get("top_pipes_pct"):
            L.append("    pipes:  " + s["top_pipes_pct"])
    if not rows:
        L.append("(no profiled launches in the inputs)")
    text = "\n".join(L) + "\n"
    with open(os.path.join(a.out, "ncu_summary.txt"), "w") as f:
        f.write(text)
    sys.stdout.write(text)
    return 0


def selftest():
    """A raw export in the layout ncu writes (header, units row, one row per launch)."""
    import tempfile
    d = tempfile.mkdtemp(prefix="ncu_summary_selftest.")
    hdr = ["ID", "Process ID", "Process Name", "Host Name", "Kernel Name", "Context", "Stream", "Block Size",
           "Grid Size", "Device", "CC", "gpu__time_duration.sum", "sm__throughput.avg.pct_of_peak_sustained_elapsed",
           "sm__warps_active.avg.pct_of_peak_sustained_active", "launch__registers_per_thread",
           "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
           "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
           "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
           "sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active"]
    units = ["", "", "", "", "", "", "", "", "", "", "", "usecond", "%", "%", "register/thread", "", "", "", "%"]
    row = ["0", "4242", "lambda_vm_prover", "box", "rpx_grind_search", "1", "7", "(128, 1, 1)", "(1024, 1, 1)",
           "0", "12.0", "7,654.32", "91.5", "62.5", "64", "0.5", "6.0", "1.5", "88.8"]
    p = os.path.join(d, "rpx_grind_search.raw.csv")
    with open(p, "w", newline="") as f:
        f.write("==PROF== a banner line ncu may leave in front\n")
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerows([hdr, units, row])
    out = os.path.join(d, "out")
    main(["--out", out, p])
    with open(os.path.join(out, "ncu_summary.csv")) as f:
        r = next(csv.DictReader(f))
    checks = [("kernel", r["kernel"], "rpx_grind_search"), ("duration us -> ms", r["duration_ms"], "7.654"),
              ("sm%", r["sm_throughput_pct"], "91.5"), ("regs", r["registers_per_thread"], "64"),
              ("top stall first", r["top_stalls"].split(" ")[0], "math_pipe_throttle"),
              ("stall share", "math_pipe_throttle 6.00 (75%)" in r["top_stalls"], True),
              ("pipe", r["top_pipes_pct"], "alu 88.8"), ("absent metric blank", r["l2_hit_pct"], "")]
    bad = [c for c in checks if c[1] != c[2]]
    for n, got, want in checks:
        print("{} {:<24} got {!r}".format("ok  " if got == want else "FAIL", n, got))
    print("SELFTEST {} ({} checks, {} failed)".format("PASS" if not bad else "FAIL", len(checks), len(bad)))
    return 0 if not bad else 1


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        sys.exit(selftest())
    sys.exit(main())
