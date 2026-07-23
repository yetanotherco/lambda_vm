#!/usr/bin/env python3
"""Per-phase GPU busy report from an nsys sqlite export.

Answers the question the raw timeline can't answer at a glance: for each
prover phase (NVTX range emitted by the `nvtx` cargo feature), how much GPU
work ran, how busy the GPU actually was, and which kernels dominate — so
host-bound phases (low busy%) are separated from kernel-bound ones (high
busy%) before anyone spends time in Nsight Compute.

Usage:
  nsys export --type sqlite -o report.sqlite report.nsys-rep
  nsys_phase_busy.py report.sqlite

Attribution: kernels/memcpys are mapped to the NVTX range that *launched*
them via CUDA API correlation IDs (correct even for async `_keep`/`_dev`
pipelines where the kernel executes after the range closes). When the export
lacks API-correlation tables it falls back to wall-clock overlap and says so.

Stdlib only (sqlite3). Written against nsys 2025.x exports; table names are
probed defensively — if a table is missing the report degrades rather than
crashes.
"""

import argparse
import bisect
import sqlite3
import sys
from collections import defaultdict

MEMCPY_KIND = {1: "h2d", 2: "d2h", 8: "d2d"}


def tables(con):
    return {
        r[0]
        for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }


def columns(con, table):
    return {r[1] for r in con.execute(f"PRAGMA table_info({table})")}


def load_strings(con, tset):
    if "StringIds" not in tset:
        return {}
    return dict(con.execute("SELECT id, value FROM StringIds"))


def load_nvtx(con, tset, strings):
    """[(start, end, name, tid)] for closed NVTX ranges."""
    if "NVTX_EVENTS" not in tset:
        return []
    cols = columns(con, "NVTX_EVENTS")
    sel_text = "text" if "text" in cols else "NULL"
    sel_tid = "globalTid" if "globalTid" in cols else "NULL"
    sel_textid = "textId" if "textId" in cols else "NULL"
    rows = con.execute(
        f"SELECT start, end, {sel_text}, {sel_textid}, {sel_tid} "
        "FROM NVTX_EVENTS WHERE start IS NOT NULL AND end IS NOT NULL"
    )
    out = []
    for start, end, text, text_id, tid in rows:
        name = text if text else strings.get(text_id)
        if name and end > start:
            out.append((start, end, name, tid))
    return out


def load_gpu_rows(con, tset, strings):
    """kernels: [(start, end, name, corr)], memcpys: [(start, end, kind, bytes, corr)]"""
    kernels = []
    for t in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL"):
        if t not in tset:
            continue
        cols = columns(con, t)
        name_col = (
            "shortName"
            if "shortName" in cols
            else ("demangledName" if "demangledName" in cols else "NULL")
        )
        corr_col = "correlationId" if "correlationId" in cols else "NULL"
        for start, end, name_id, corr in con.execute(
            f"SELECT start, end, {name_col}, {corr_col} FROM {t}"
        ):
            name = strings.get(name_id, name_id if isinstance(name_id, str) else "?")
            kernels.append((start, end, name or "?", corr))

    memcpys = []
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tset:
        cols = columns(con, "CUPTI_ACTIVITY_KIND_MEMCPY")
        kind_col = "copyKind" if "copyKind" in cols else "NULL"
        bytes_col = "bytes" if "bytes" in cols else ("size" if "size" in cols else "0")
        corr_col = "correlationId" if "correlationId" in cols else "NULL"
        for start, end, kind, nbytes, corr in con.execute(
            f"SELECT start, end, {kind_col}, {bytes_col}, {corr_col} "
            "FROM CUPTI_ACTIVITY_KIND_MEMCPY"
        ):
            memcpys.append((start, end, MEMCPY_KIND.get(kind, "other"), nbytes or 0, corr))
    return kernels, memcpys


def load_api_calls(con, tset):
    """correlationId -> (api_start_ns, tid), from runtime and/or driver API rows."""
    out = {}
    for t in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
        if t not in tset:
            continue
        cols = columns(con, t)
        if not {"start", "correlationId", "globalTid"} <= cols:
            continue
        for start, corr, tid in con.execute(
            f"SELECT start, correlationId, globalTid FROM {t}"
        ):
            out[corr] = (start, tid)
    return out


def build_range_lookup(nvtx):
    """Per-tid sweep structure answering: which ranges enclose time t?

    NVTX push/pop ranges nest per thread, so a sweep with a stack
    reconstructs the enclosing chain (outermost..innermost) at any instant.
    Returns a function chain_at(tid, t) -> [names].
    """
    events = defaultdict(list)  # tid -> [(time, kind, name, end)]
    for start, end, name, tid in nvtx:
        events[tid].append((start, 0, name, end))
    snapshots = {}  # tid -> (times[], chains[])
    for tid, evs in events.items():
        evs.sort()
        # sweep over starts; maintain stack of (end, name), popping expired
        times, chains = [], []
        stack = []
        for start, _, name, end in evs:
            while stack and stack[-1][0] <= start:
                t_end = stack[-1][0]
                stack.pop()
                times.append(t_end)
                chains.append([n for _, n in stack])
            stack.append((end, name))
            times.append(start)
            chains.append([n for _, n in stack])
        while stack:
            t_end = stack[-1][0]
            stack.pop()
            times.append(t_end)
            chains.append([n for _, n in stack])
        snapshots[tid] = (times, chains)

    def chain_at(tid, t):
        snap = snapshots.get(tid)
        if not snap:
            return []
        times, chains = snap
        i = bisect.bisect_right(times, t) - 1
        return chains[i] if i >= 0 else []

    return chain_at


def base_name(nvtx_name):
    """Strip the '[shape]' payload: 'lde_tree_base[n=.. m=..]' -> 'lde_tree_base'."""
    return nvtx_name.split("[", 1)[0]


def merge_intervals(intervals):
    """Union length of [(start, end)]."""
    total = 0
    last_s = last_e = None
    for s, e in sorted(intervals):
        if last_e is None or s > last_e:
            if last_e is not None:
                total += last_e - last_s
            last_s, last_e = s, e
        else:
            last_e = max(last_e, e)
    if last_e is not None:
        total += last_e - last_s
    return total


def clipped_union(intervals_sorted, windows):
    """Union length of intervals clipped to the union of windows."""
    clipped = []
    for w0, w1 in windows:
        lo = bisect.bisect_left(intervals_sorted, (w0, w0)) - 1
        for i in range(max(lo, 0), len(intervals_sorted)):
            s, e = intervals_sorted[i]
            if s >= w1:
                break
            if e > w0:
                clipped.append((max(s, w0), min(e, w1)))
    return merge_intervals(clipped)


def fmt_ms(ns):
    return f"{ns / 1e6:.1f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sqlite", help="output of `nsys export --type sqlite`")
    ap.add_argument("--top", type=int, default=15, help="top-N kernels overall")
    args = ap.parse_args()

    con = sqlite3.connect(f"file:{args.sqlite}?mode=ro", uri=True)
    tset = tables(con)
    strings = load_strings(con, tset)
    nvtx = load_nvtx(con, tset, strings)
    kernels, memcpys = load_gpu_rows(con, tset, strings)
    api = load_api_calls(con, tset)

    if not kernels:
        sys.exit("no kernel rows in export — was the run traced with -t cuda?")
    if not nvtx:
        print(
            "warning: no NVTX ranges — build the prover with --features nvtx",
            file=sys.stderr,
        )

    chain_at = build_range_lookup(nvtx)
    by_correlation = bool(api)

    def chain_for(corr, start):
        if by_correlation and corr in api:
            api_start, tid = api[corr]
            c = chain_at(tid, api_start)
            if c:
                return c
        # fallback: wall-clock overlap on any thread (async kernels may
        # misattribute here; correlation is the trustworthy path)
        for _, _, name, tid in nvtx:
            c = chain_at(tid, start)
            if c:
                return c
        return []

    def coarse_of(chain):
        """Innermost *phase* range: instruments spans carry no '[shape]'
        payload, math-cuda entry points do — so the deepest bracket-free
        name is the enclosing prover phase (r1_main_commit, rounds_2to4...)."""
        for name in reversed(chain):
            if "[" not in name:
                return name
        return base_name(chain[0]) if chain else "(no NVTX range)"

    # --- attribute kernels & memcpys -------------------------------------
    coarse = defaultdict(lambda: {"kernel_ns": 0, "kernels": defaultdict(int), "n": 0})
    fine = defaultdict(lambda: {"kernel_ns": 0, "kernels": defaultdict(int), "n": 0})
    copy_by_phase = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # phase->kind->[ns,bytes]
    kernel_totals = defaultdict(lambda: [0, 0])  # name -> [count, ns]

    for start, end, name, corr in kernels:
        dur = end - start
        kernel_totals[name][0] += 1
        kernel_totals[name][1] += dur
        chain = chain_for(corr, start)
        coarse_key = coarse_of(chain) if chain else "(no NVTX range)"
        fine_key = base_name(chain[-1]) if chain else "(no NVTX range)"
        coarse[coarse_key]["kernel_ns"] += dur
        coarse[coarse_key]["kernels"][name] += dur
        coarse[coarse_key]["n"] += 1
        fine[fine_key]["kernel_ns"] += dur
        fine[fine_key]["kernels"][name] += dur
        fine[fine_key]["n"] += 1

    for start, end, kind, nbytes, corr in memcpys:
        chain = chain_for(corr, start)
        key = coarse_of(chain) if chain else "(no NVTX range)"
        copy_by_phase[key][kind][0] += end - start
        copy_by_phase[key][kind][1] += nbytes

    # --- headline numbers --------------------------------------------------
    cap_lo = min(s for s, *_ in kernels)
    cap_hi = max(e for _, e, *_ in kernels)
    kernel_union = merge_intervals([(s, e) for s, e, _, _ in kernels])
    print("# GPU phase report")
    print()
    print(f"- attribution: {'API correlation' if by_correlation else 'WALL-CLOCK OVERLAP (fallback — treat with suspicion)'}")
    print(f"- kernels: {len(kernels)}   memcpys: {len(memcpys)}   NVTX ranges: {len(nvtx)}")
    print(
        f"- GPU busy {fmt_ms(kernel_union)} ms over {fmt_ms(cap_hi - cap_lo)} ms of GPU activity span "
        f"({100 * kernel_union / max(cap_hi - cap_lo, 1):.0f}%)"
    )
    print()

    # --- coarse phase table -------------------------------------------------
    # phase wall = union of that phase's NVTX windows; busy = kernel union in them
    windows_by_phase = defaultdict(list)
    for start, end, name, _tid in nvtx:
        windows_by_phase[base_name(name)].append((start, end))
    kern_sorted = sorted((s, e) for s, e, _, _ in kernels)

    print("## Phases (innermost enclosing instruments span)")
    print()
    print("| phase | wall ms | kernel-sum ms | gpu-busy ms | busy% | h2d ms/MiB | d2h ms/MiB | top kernels |")
    print("|---|---|---|---|---|---|---|---|")
    for key, agg in sorted(coarse.items(), key=lambda kv: -kv[1]["kernel_ns"]):
        wins = windows_by_phase.get(key, [])
        wall = merge_intervals(wins)
        busy = clipped_union(kern_sorted, sorted(wins)) if wins else 0
        busy_pct = f"{100 * busy / wall:.0f}%" if wall else "-"
        cp = copy_by_phase.get(key, {})
        h2d = cp.get("h2d", [0, 0])
        d2h = cp.get("d2h", [0, 0])
        top = sorted(agg["kernels"].items(), key=lambda kv: -kv[1])[:3]
        top_s = "; ".join(f"{n} {fmt_ms(ns)}" for n, ns in top)
        print(
            f"| {key} | {fmt_ms(wall) if wall else '-'} | {fmt_ms(agg['kernel_ns'])} "
            f"| {fmt_ms(busy) if wins else '-'} | {busy_pct} "
            f"| {fmt_ms(h2d[0])}/{h2d[1] / 2**20:.0f} | {fmt_ms(d2h[0])}/{d2h[1] / 2**20:.0f} "
            f"| {top_s} |"
        )
    print()

    # --- fine entry-point table ----------------------------------------------
    print("## math-cuda entry points (innermost NVTX range)")
    print()
    print("| entry point | launches | kernel-sum ms | top kernel |")
    print("|---|---|---|---|")
    for key, agg in sorted(fine.items(), key=lambda kv: -kv[1]["kernel_ns"]):
        top = max(agg["kernels"].items(), key=lambda kv: kv[1])
        print(
            f"| {key} | {agg['n']} | {fmt_ms(agg['kernel_ns'])} "
            f"| {top[0]} {fmt_ms(top[1])} |"
        )
    print()

    # --- per-epoch view (continuations) ----------------------------------------
    # `epoch[i=N]` ranges come from prove_continuation. Two numbers per epoch
    # matter for parallelization: GPU busy% inside the epoch window (host-bound
    # vs GPU-bound epochs) and GPU busy inside the gap to the next epoch
    # (≈0 means the GPU sits idle between epochs — the pipelining headroom).
    epoch_wins = sorted(
        (start, end, name)
        for start, end, name, _tid in nvtx
        if name.startswith("epoch[")
    )
    if epoch_wins:
        print("## Epochs (continuations)")
        print()
        print("| epoch | wall ms | gpu-busy ms | busy% | gap→next ms | gpu-busy in gap ms |")
        print("|---|---|---|---|---|---|")
        total_gap = total_gap_busy = 0
        for i, (w0, w1, name) in enumerate(epoch_wins):
            busy = clipped_union(kern_sorted, [(w0, w1)])
            if i + 1 < len(epoch_wins):
                g0, g1 = w1, epoch_wins[i + 1][0]
                gap = max(g1 - g0, 0)
                gap_busy = clipped_union(kern_sorted, [(g0, g1)]) if gap else 0
                total_gap += gap
                total_gap_busy += gap_busy
                gap_s, gap_busy_s = fmt_ms(gap), fmt_ms(gap_busy)
            else:
                gap_s = gap_busy_s = "-"
            print(
                f"| {name} | {fmt_ms(w1 - w0)} | {fmt_ms(busy)} "
                f"| {100 * busy / (w1 - w0):.0f}% | {gap_s} | {gap_busy_s} |"
            )
        print()
        print(
            f"total inter-epoch gap: {fmt_ms(total_gap)} ms with only "
            f"{fmt_ms(total_gap_busy)} ms of GPU work inside it — GPU-idle gap time "
            f"is the upper bound on what epoch pipelining (overlap execute/trace-build "
            f"of epoch N+1 with GPU proving of epoch N) can recover."
        )
        print()

    # --- top kernels overall ---------------------------------------------------
    total_ns = sum(ns for _, ns in kernel_totals.values())
    print(f"## Top {args.top} kernels overall")
    print()
    print("| kernel | count | total ms | avg µs | % gpu time |")
    print("|---|---|---|---|---|")
    for name, (cnt, ns) in sorted(kernel_totals.items(), key=lambda kv: -kv[1][1])[
        : args.top
    ]:
        print(
            f"| {name} | {cnt} | {fmt_ms(ns)} | {ns / cnt / 1e3:.1f} "
            f"| {100 * ns / total_ns:.1f}% |"
        )


if __name__ == "__main__":
    main()
