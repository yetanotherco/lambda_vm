#!/usr/bin/env python3
"""Aggregate instruments timeline JSONs into a per-phase table.

Input: one or more files written via LAMBDA_VM_TIMELINE_JSON (schema:
[{"label","depth","wall_ns","order","start_ns"}]), typically one per prove run
of the same workload. Optionally, per-run GPU utilization CSVs from
nvml_sampler.py to attribute average GPU busy% to each phase (aligned via the
spans' start_ns epoch — that field exists precisely for this).

Output: a markdown table in tree order with per-phase median / % of total /
run-to-run spread, plus GPU util columns when samplers were provided.

Usage:
  phase_table.py run1.json run2.json run3.json
  phase_table.py --util run1_util.csv --util run2_util.csv ... run1.json run2.json ...
  (the i-th --util CSV pairs with the i-th timeline JSON)

Stdlib only. Robust to repeated labels (e.g. per-epoch spans): repeats within
one run are summed per tree path and the instance count is reported.
"""

import argparse
import json
import statistics
import sys


def build_paths(spans):
    """Reconstruct tree paths from (order, depth).

    Spans are recorded on close but `order` is assigned on open, so sorting by
    order yields open-order; a span's parent is the most recent span with
    depth-1. Returns a list of (path_tuple, span_dict) in open order.
    """
    spans = sorted(spans, key=lambda s: s["order"])
    stack = []  # labels of currently-open ancestors, by depth
    out = []
    for s in spans:
        d = s["depth"]
        del stack[d:]
        stack.append(s["label"])
        out.append((tuple(stack), s))
    return out


def load_util_csv(path):
    """nvml_sampler.py CSV -> list of (epoch_ns, gpu_util_pct, mem_util_pct, vram_mib)."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("epoch_ns"):
                continue
            parts = line.split(",")
            try:
                rows.append(
                    (int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))
                )
            except (ValueError, IndexError):
                continue
    rows.sort()
    return rows


def util_in_window(samples, start_ns, end_ns):
    """Mean (gpu_util, mem_util, max_vram) of samples inside [start,end]."""
    inside = [s for s in samples if start_ns <= s[0] <= end_ns]
    if not inside:
        return None
    return (
        statistics.fmean(s[1] for s in inside),
        statistics.fmean(s[2] for s in inside),
        max(s[3] for s in inside),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("timelines", nargs="+", help="timeline JSON files (one per run)")
    ap.add_argument(
        "--util",
        action="append",
        default=[],
        help="nvml_sampler CSV paired positionally with the i-th timeline",
    )
    ap.add_argument(
        "--min-pct",
        type=float,
        default=0.0,
        help="hide phases below this %% of total (default: show all)",
    )
    ap.add_argument(
        "--instances",
        action="append",
        default=[],
        help="also print the per-instance table for this repeated label "
        "(depth-1 repeats like continuation `epoch` are included automatically)",
    )
    args = ap.parse_args()

    if args.util and len(args.util) != len(args.timelines):
        sys.exit("--util count must match timeline count (they pair positionally)")

    # per-path data across runs
    # path -> {"wall_ns": [per-run sums], "count": [per-run instance counts],
    #          "util": [per-run (gpu, mem, vram) or None]}
    agg = {}
    order_of_path = {}  # first-seen open order, for stable tree-order output
    totals = []
    # per-(path, instance-index) data for repeated spans (continuation epochs):
    # wall per instance, gap to the next instance, GPU util inside each.
    inst_agg = {}

    for i, tl_path in enumerate(args.timelines):
        with open(tl_path) as f:
            spans = json.load(f)
        if not spans:
            print(f"warning: {tl_path} is empty, skipping", file=sys.stderr)
            continue
        pathed = build_paths(spans)
        totals.append(max(s["wall_ns"] for _, s in pathed))
        samples = load_util_csv(args.util[i]) if args.util else None

        per_run = {}
        for path, s in pathed:
            if path not in order_of_path:
                order_of_path[path] = s["order"]
            e = per_run.setdefault(path, {"wall_ns": 0, "count": 0, "windows": []})
            e["wall_ns"] += s["wall_ns"]
            e["count"] += 1
            e["windows"].append((s["start_ns"], s["start_ns"] + s["wall_ns"]))

        for path, e in per_run.items():
            if len(e["windows"]) < 2:
                continue
            wins = sorted(e["windows"])
            slots = inst_agg.setdefault(path, {})
            for idx, (w0, w1) in enumerate(wins):
                s = slots.setdefault(
                    idx, {"wall": [], "gap": [], "util_span": [], "util_gap": []}
                )
                s["wall"].append(w1 - w0)
                if samples:
                    u = util_in_window(samples, w0, w1)
                    if u:
                        s["util_span"].append(u[0])
                if idx + 1 < len(wins):
                    g0, g1 = w1, wins[idx + 1][0]
                    s["gap"].append(max(g1 - g0, 0))
                    if samples and g1 > g0:
                        u = util_in_window(samples, g0, g1)
                        if u:
                            s["util_gap"].append(u[0])

        for path, e in per_run.items():
            a = agg.setdefault(path, {"wall_ns": [], "count": [], "util": []})
            a["wall_ns"].append(e["wall_ns"])
            a["count"].append(e["count"])
            if samples:
                # weighted mean over this path's windows
                utils = [util_in_window(samples, w0, w1) for (w0, w1) in e["windows"]]
                utils = [u for u in utils if u is not None]
                if utils:
                    a["util"].append(
                        (
                            statistics.fmean(u[0] for u in utils),
                            statistics.fmean(u[1] for u in utils),
                            max(u[2] for u in utils),
                        )
                    )

    if not agg:
        sys.exit("no spans found in any input")

    total_med = statistics.median(totals)
    n_runs = len(totals)
    have_util = bool(args.util)

    hdr = ["phase", "median", "% of total", "cv%", "n/run"]
    if have_util:
        hdr += ["gpu%", "mem%", "vram MiB"]
    print(f"Runs: {n_runs}   total (median): {total_med / 1e9:.3f}s")
    print()
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join("---" for _ in hdr) + "|")

    for path in sorted(agg, key=lambda p: order_of_path[p]):
        a = agg[path]
        med = statistics.median(a["wall_ns"])
        pct = 100.0 * med / total_med
        if pct < args.min_pct:
            continue
        cv = (
            100.0 * statistics.stdev(a["wall_ns"]) / statistics.fmean(a["wall_ns"])
            if len(a["wall_ns"]) > 1 and statistics.fmean(a["wall_ns"]) > 0
            else 0.0
        )
        count = round(statistics.fmean(a["count"]))
        indent = "&nbsp;&nbsp;" * (len(path) - 1)
        row = [
            f"{indent}{path[-1]}",
            f"{med / 1e6:.1f} ms" if med < 10e9 else f"{med / 1e9:.2f} s",
            f"{pct:.1f}%",
            f"{cv:.1f}",
            str(count),
        ]
        if have_util:
            if a["util"]:
                g = statistics.fmean(u[0] for u in a["util"])
                m = statistics.fmean(u[1] for u in a["util"])
                v = max(u[2] for u in a["util"])
                row += [f"{g:.0f}", f"{m:.0f}", f"{v:.0f}"]
            else:
                row += ["-", "-", "-"]
        print("| " + " | ".join(row) + " |")

    # --- per-instance breakdown for repeated spans (continuation epochs) ----
    # The data for parallelization decisions: epoch uniformity (wall spread),
    # inter-instance gaps, and GPU util inside the span vs inside the gap
    # (idle gaps = what pipelining/overlap would recover).
    wanted = {p for p in inst_agg if len(p) <= 2 or p[-1] in set(args.instances)}
    for path in sorted(wanted, key=lambda p: order_of_path[p]):
        slots = inst_agg[path]
        label = " > ".join(path[1:]) or path[0]
        print(f"\n### `{label}` instances ({len(slots)} per run)")
        print()
        hdr2 = ["#", "wall", "gap→next"]
        if have_util:
            hdr2 += ["gpu% span", "gpu% gap"]
        print("| " + " | ".join(hdr2) + " |")
        print("|" + "|".join("---" for _ in hdr2) + "|")
        total_gap = 0
        for idx in sorted(slots):
            s = slots[idx]
            wall = statistics.median(s["wall"])
            gap = statistics.median(s["gap"]) if s["gap"] else None
            total_gap += gap or 0
            row = [
                str(idx),
                f"{wall / 1e6:.1f} ms",
                f"{gap / 1e6:.1f} ms" if gap is not None else "-",
            ]
            if have_util:
                row += [
                    f"{statistics.fmean(s['util_span']):.0f}" if s["util_span"] else "-",
                    f"{statistics.fmean(s['util_gap']):.0f}" if s["util_gap"] else "-",
                ]
            print("| " + " | ".join(row) + " |")
        walls = [statistics.median(s["wall"]) for s in slots.values()]
        print(
            f"\ninstances: {len(slots)}   wall min/median/max: "
            f"{min(walls) / 1e6:.1f} / {statistics.median(walls) / 1e6:.1f} / "
            f"{max(walls) / 1e6:.1f} ms   total inter-instance gap: {total_gap / 1e6:.1f} ms "
            f"({100 * total_gap / total_med:.1f}% of total)"
        )


if __name__ == "__main__":
    main()
