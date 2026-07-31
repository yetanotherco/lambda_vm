#!/usr/bin/env python3
"""H2D/D2H attribution histogram from an nsys sqlite export.

Groups memcpys by (enclosing phase, innermost NVTX range, size) so the
dominant uploaders inside a phase are identifiable by name + size fingerprint.
Reuses the loaders from nsys_phase_busy.py (same directory).
"""

import os
import sqlite3
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nsys_phase_busy import (
    base_name,
    build_range_lookup,
    load_api_calls,
    load_gpu_rows,
    load_nvtx,
    load_strings,
    tables,
)


def main():
    db = sys.argv[1]
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    tset = tables(con)
    strings = load_strings(con, tset)
    nvtx = load_nvtx(con, tset, strings)
    _, memcpys = load_gpu_rows(con, tset, strings)
    api = load_api_calls(con, tset)
    chain_at = build_range_lookup(nvtx)

    def chain_for(corr, start):
        if corr in api:
            api_start, tid = api[corr]
            c = chain_at(tid, api_start)
            if c:
                return c
        return []

    def coarse_of(chain):
        for name in reversed(chain):
            if "[" not in name:
                return name
        return base_name(chain[0]) if chain else "(none)"

    def innermost(chain):
        return base_name(chain[-1]) if chain else "(none)"

    # (direction, phase, inner, bytes) -> [count, total_bytes, total_ns]
    hist = defaultdict(lambda: [0, 0, 0])
    for start, end, kind, nbytes, corr in memcpys:
        if kind not in ("h2d", "d2h"):
            continue
        chain = chain_for(corr, start)
        key = (kind, coarse_of(chain), innermost(chain), nbytes)
        h = hist[key]
        h[0] += 1
        h[1] += nbytes
        h[2] += end - start

    for direction in ("h2d", "d2h"):
        rows = [(k, v) for k, v in hist.items() if k[0] == direction]
        rows.sort(key=lambda kv: -kv[1][1])
        total_gb = sum(v[1] for _, v in rows) / 2**30
        print(f"\n== {direction.upper()} total {total_gb:.1f} GiB — top 20 by bytes ==")
        print(f"{'phase':<28} {'inner range':<28} {'size MiB':>9} {'count':>6} {'GiB':>7} {'ms':>8}")
        for (_, phase, inner, nbytes), (cnt, tot, ns) in rows[:20]:
            print(
                f"{phase:<28} {inner:<28} {nbytes / 2**20:>9.2f} {cnt:>6} "
                f"{tot / 2**30:>7.2f} {ns / 1e6:>8.1f}"
            )


if __name__ == "__main__":
    main()
