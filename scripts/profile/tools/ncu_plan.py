#!/usr/bin/env python3
"""block_profile.sh's Nsight Compute plan: one representative launch for each named kernel, from one Nsight Systems export.

    ncu_plan.py --sqlite F --out plan.tsv --kernels k1,k2a|k2b,... [--prefix a] [--max N]

For each entry, in the order given (an entry `a|b` takes the first of its names present), that appears in the export's
kernel activity: the launch shape (grid, block) that
carries most of that kernel's time, then the launch of that shape with the median duration. Its launch_skip is its
index among ALL of that kernel's launches in correlation-id order, which is the order ncu counts launches in. Names
absent from the export are reported and skipped. Standard library only; it reads the export through capture_summary's
Export, which never reads the tables that hold the process environment.
"""
import argparse
import os
import sys
from collections import defaultdict

_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_here, os.path.dirname(_here)]   # flat in the run's tools/, or one level up in the branch
import capture_summary as P  # noqa: E402


def plan(sqlite, kernels, prefix, cap):
    x = P.Export(sqlite)
    launches = defaultdict(list)   # name → [(corr, start, end, grid, block)]
    for s, e, n, g, b, c in x.k:
        launches[n].append((c, s, e, g, b))
    total = sum(e - s for s, e, *_ in x.k) or 1
    rows, missing = [], []
    for entry in kernels:
        if len(rows) >= cap:
            break
        name = next((n for n in entry.split("|") if launches.get(n)), None)
        if name is None:
            missing.append(entry)
            continue
        ls = sorted(launches[name])
        shapes = defaultdict(int)
        for _, s, e, g, b in ls:
            shapes[(g, b)] += e - s
        (g, b), shape_ns = max(shapes.items(), key=lambda kv: kv[1])
        cand = sorted(((i, e - s) for i, (_, s, e, gg, bb) in enumerate(ls) if (gg, bb) == (g, b)), key=lambda t: t[1])
        idx, dur = cand[len(cand) // 2]
        kern_ns = sum(e - s for _, s, e, _, _ in ls)
        rows.append((f"{prefix}{len(rows) + 1:02d}", name, idx, 1, g, b, f"{100 * kern_ns / total:.1f}", f"{dur / 1e3:.1f}",
                     len(ls), f"{100 * shape_ns / kern_ns:.0f}"))
    return rows, missing


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sqlite", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--kernels", required=True)
    p.add_argument("--prefix", default="a")
    p.add_argument("--max", type=int, default=12)
    a = p.parse_args()
    names = [k for k in a.kernels.split(",") if k]
    rows, missing = plan(a.sqlite, names, a.prefix, a.max)
    P.wtsv(a.out, ["pass", "kernel", "launch_skip", "launch_count", "grid", "block", "pct_of_kernel_time", "launch_us",
                   "launches", "shape_pct_of_kernel"], rows)
    print(f"NCU PLAN {a.out}: {len(rows)} pass(es)" + (f" · absent from the capture: {', '.join(missing)}" if missing else ""))
    for r in rows:
        print(f"  {r[0]} {r[1]} launch {r[2]} of {r[8]} · {r[4]} / {r[5]} · {r[7]} µs · {r[6]} % of kernel time")
    return 0


if __name__ == "__main__":
    sys.exit(main())
