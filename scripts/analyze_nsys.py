#!/usr/bin/env python3
"""Summarize an nsys sqlite export of a prove run into a bottleneck report.

Usage: analyze_nsys.py <report.sqlite> [--gap-ms 1.0] [--top 15]

Input: `nsys export --type sqlite` of a run traced with `-t cuda,nvtx` and the
repo's `nvtx` feature enabled (see docs/gpu_profiling.md). Everything is
computed from four tables: CUPTI kernels, memcpys, driver-API calls
(CUPTI_ACTIVITY_KIND_RUNTIME) and NVTX_EVENTS.

Output (markdown, stdout):
  1. capture window + GPU busy/idle accounting (kernels vs memcpy vs idle)
  2. kernel-concurrency histogram (how many streams are actually overlapping)
  3. wall-phase table: per main-thread instruments span, wall vs GPU busy inside
  4. GPU time attributed to pipeline phases (via the launching thread's
     innermost NVTX range at launch time)
  5. host time blocked in driver API calls, total and per phase
  6. largest GPU-idle gaps, attributed to the phase that ran next
  7. top kernels by total GPU time
"""

import argparse
import bisect
import sqlite3
import sys
from collections import defaultdict

MS = 1e6  # ns per ms


def union_len(intervals):
    """Total covered length of [start, end) intervals, in ns."""
    total = 0
    last_end = None
    for s, e in sorted(intervals):
        if last_end is None or s >= last_end:
            total += e - s
            last_end = e
        elif e > last_end:
            total += e - last_end
            last_end = e
    return total


def union_intervals(intervals):
    """Merge intervals; returns sorted disjoint list."""
    merged = []
    for s, e in sorted(intervals):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def clip_len(merged, lo, hi):
    """Length of pre-merged disjoint intervals clipped to [lo, hi]."""
    total = 0
    for s, e in merged:
        s2, e2 = max(s, lo), min(e, hi)
        if s2 < e2:
            total += e2 - s2
    return total


# Map an innermost/thread NVTX label to a pipeline phase bucket.
PHASE_RULES = [
    ("r1_main:", "R1 main commit"),
    ("gpu:commit_row_major", "R1 main commit"),
    ("gpu:lde_batch_base", "R1 main commit"),
    ("r1_aux_build:", "R1 aux build (logup)"),
    ("gpu:logup", "R1 aux build (logup)"),
    ("r1_aux_commit:", "R1 aux commit"),
    ("gpu:commit_ext3", "R1 aux commit"),
    ("round2", "R2 composition"),
    ("gpu:eval_composition", "R2 composition"),
    ("gpu:extend_halves", "R2 composition"),
    ("gpu:parts_lde", "R2 composition"),
    ("gpu:comp_poly_tree", "R2 composition"),
    ("gpu:lde_batch_ext3", "R2 composition"),
    ("round3", "R3 OOD"),
    ("gpu:bary", "R3 OOD"),
    ("gpu:r3_prep", "R3 OOD"),
    ("gpu:inv_denoms", "R3/R4 inv denoms"),
    ("round4", "R4"),
    ("gpu:deep", "R4 DEEP"),
    ("gpu:fri", "R4 FRI"),
    ("fri_layer", "R4 FRI"),
    ("gpu:gather_proofs", "R4 queries"),
    ("gpu:gather_merkle", "R4 queries"),
    ("r1_prepass", "R1 prepass"),
    ("proving", "(proving, unattributed)"),
]

# Main-thread wall spans worth tabulating (instruments spans emitted as NVTX).
WALL_SPANS = [
    "proving",
    "r1_prepass",
    "r1_main_commit",
    "r1_aux_build",
    "r1_aux_commit",
    "rounds_2to4",
]


def classify(stack):
    """Phase bucket for a launching-thread NVTX stack (innermost first)."""
    for label in stack:  # innermost first
        for prefix, phase in PHASE_RULES:
            if label.startswith(prefix):
                # Prefer round2/3/4 context over generic gpu:* only when the
                # gpu:* rule itself matched — the loop order already gives the
                # innermost label priority, which is what we want.
                return phase
    return "(unattributed)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--gap-ms", type=float, default=1.0)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    db = sqlite3.connect(args.sqlite)
    strings = dict(db.execute("SELECT id, value FROM StringIds"))

    kernels = [
        (s, e, strings.get(name, "?"), stream, corr)
        for s, e, name, stream, corr in db.execute(
            'SELECT start, "end", shortName, streamId, correlationId '
            "FROM CUPTI_ACTIVITY_KIND_KERNEL"
        )
    ]
    memcpys = [
        (s, e, kind, b, corr)
        for s, e, kind, b, corr in db.execute(
            'SELECT start, "end", copyKind, bytes, correlationId '
            "FROM CUPTI_ACTIVITY_KIND_MEMCPY"
        )
    ]
    apis = list(
        db.execute(
            'SELECT start, "end", nameId, correlationId, globalTid '
            "FROM CUPTI_ACTIVITY_KIND_RUNTIME"
        )
    )
    nvtx = [
        (s, e, text, tid)
        for s, e, text, tid in db.execute(
            'SELECT start, "end", text, globalTid FROM NVTX_EVENTS '
            "WHERE eventType = 59 AND text IS NOT NULL AND \"end\" IS NOT NULL"
        )
    ]
    if not kernels:
        print("no kernels in capture — was the GPU path active?")
        return 1

    # --- capture window -----------------------------------------------------
    t0 = min(s for s, *_ in kernels + memcpys)
    t1 = max(e for _, e, *_ in kernels + memcpys)
    wall = t1 - t0

    kern_iv = union_intervals([(s, e) for s, e, *_ in kernels])
    copy_iv = union_intervals([(s, e) for s, e, *_ in memcpys])
    busy_iv = union_intervals([(s, e) for s, e, *_ in kernels]
                              + [(s, e) for s, e, *_ in memcpys])
    kern_busy = union_len([(s, e) for s, e, *_ in kernels])
    copy_busy = union_len([(s, e) for s, e, *_ in memcpys])
    busy = union_len([(s, e) for s, e, *_ in kernels]
                     + [(s, e) for s, e, *_ in memcpys])

    out = []
    out.append("# GPU proving-time breakdown (nsys)\n")
    out.append(f"capture window: **{wall / MS / 1000:.3f} s** "
               f"(GPU activity from first kernel/copy to last)\n")
    out.append("## 1. Where the window goes (GPU device view)\n")
    out.append("| bucket | time | % of window |")
    out.append("|---|---|---|")
    for name, v in [
        ("GPU busy (kernels ∪ memcpy)", busy),
        ("· kernels (union)", kern_busy),
        ("· memcpy (union)", copy_busy),
        ("GPU idle", wall - busy),
    ]:
        out.append(f"| {name} | {v / MS:.0f} ms | {100 * v / wall:.1f}% |")

    # --- concurrency histogram ----------------------------------------------
    events = []
    for s, e, *_ in kernels:
        events.append((s, 1))
        events.append((e, -1))
    events.sort()
    depth_time = defaultdict(int)
    depth, prev = 0, events[0][0]
    for t, d in events:
        depth_time[depth] += t - prev
        depth, prev = depth + d, t
    out.append("\n## 2. Kernel concurrency (streams actually overlapping)\n")
    out.append("| concurrent kernels | time | % of window |")
    out.append("|---|---|---|")
    for d in sorted(depth_time):
        v = depth_time[d]
        if d == 0:
            v = wall - sum(x for k, x in depth_time.items() if k > 0)
        if v / wall > 0.001:
            out.append(f"| {d} | {v / MS:.0f} ms | {100 * v / wall:.1f}% |")

    # --- wall spans (main-thread instruments phases) --------------------------
    out.append("\n## 3. Wall phases vs GPU busy inside each\n")
    out.append("| phase (host span) | wall | GPU busy inside | GPU util |")
    out.append("|---|---|---|---|")
    for name in WALL_SPANS:
        spans = [(s, e) for s, e, text, _ in nvtx if text == name]
        if not spans:
            continue
        w = sum(e - s for s, e in spans)
        g = sum(clip_len(busy_iv, max(s, t0), min(e, t1)) for s, e in spans)
        out.append(f"| {name} | {w / MS:.0f} ms | {g / MS:.0f} ms "
                   f"| {100 * g / w if w else 0:.0f}% |")

    # --- attribution: kernel/memcpy GPU time by launching thread's NVTX stack --
    api_by_corr = {corr: (s, tid) for s, e, _, corr, tid in apis}
    nvtx_by_tid = defaultdict(list)
    for s, e, text, tid in nvtx:
        nvtx_by_tid[tid].append((s, e, text))
    for tid in nvtx_by_tid:
        nvtx_by_tid[tid].sort()
    starts_by_tid = {tid: [s for s, _, _ in v] for tid, v in nvtx_by_tid.items()}

    def stack_at(tid, t):
        """Innermost-first NVTX labels open on thread tid at time t."""
        ranges = nvtx_by_tid.get(tid)
        if not ranges:
            return []
        idx = bisect.bisect_right(starts_by_tid[tid], t)
        open_ranges = [
            (s, e, text) for s, e, text in ranges[:idx] if e > t
        ]
        # innermost = latest start
        open_ranges.sort(key=lambda r: r[0], reverse=True)
        return [text for _, _, text in open_ranges]

    phase_kern = defaultdict(int)
    phase_copy = defaultdict(int)
    phase_launches = defaultdict(int)
    kern_attr = {}  # correlationId -> phase (reused for gap attribution)
    for s, e, name, stream, corr in kernels:
        api = api_by_corr.get(corr)
        phase = classify(stack_at(api[1], api[0])) if api else "(unattributed)"
        phase_kern[phase] += e - s
        phase_launches[phase] += 1
        kern_attr[corr] = phase
    for s, e, kind, b, corr in memcpys:
        api = api_by_corr.get(corr)
        phase = classify(stack_at(api[1], api[0])) if api else "(unattributed)"
        phase_copy[phase] += e - s

    out.append("\n## 4. GPU time by pipeline phase "
               "(attributed via launching thread's NVTX stack)\n")
    out.append("| phase | kernel time | memcpy time | launches |")
    out.append("|---|---|---|---|")
    for phase in sorted(phase_kern, key=phase_kern.get, reverse=True):
        out.append(f"| {phase} | {phase_kern[phase] / MS:.0f} ms "
                   f"| {phase_copy.get(phase, 0) / MS:.0f} ms "
                   f"| {phase_launches[phase]} |")

    # --- host blocked in driver API -------------------------------------------
    api_total = defaultdict(int)
    api_calls = defaultdict(int)
    api_phase = defaultdict(int)
    for s, e, name_id, corr, tid in apis:
        name = strings.get(name_id, "?")
        api_total[name] += e - s
        api_calls[name] += 1
        if name.startswith(("cuMemcpy", "cuStreamSynchronize", "cuMemHostAlloc",
                            "cuMemFreeHost", "cuLaunchKernel", "cuMemAllocAsync",
                            "cuMemFreeAsync", "cuMemsetD")):
            phase = classify(stack_at(tid, s))
            api_phase[(phase, name)] += e - s

    out.append("\n## 5. Host time inside driver API calls "
               "(sums across threads — can exceed wall)\n")
    out.append("| API | total | calls |")
    out.append("|---|---|---|")
    for name in sorted(api_total, key=api_total.get, reverse=True)[: args.top]:
        out.append(f"| {name} | {api_total[name] / MS:.0f} ms "
                   f"| {api_calls[name]} |")
    out.append("\nper phase, top blockers:\n")
    out.append("| phase | API | total |")
    out.append("|---|---|---|")
    for (phase, name), v in sorted(api_phase.items(), key=lambda kv: -kv[1])[: args.top]:
        out.append(f"| {phase} | {name} | {v / MS:.0f} ms |")

    # --- idle gaps -------------------------------------------------------------
    gaps = []
    prev_end = t0
    for s, e in busy_iv:
        if s - prev_end > args.gap_ms * MS:
            gaps.append((prev_end, s))
        prev_end = max(prev_end, e)
    if t1 - prev_end > args.gap_ms * MS:
        gaps.append((prev_end, t1))
    kern_sorted = sorted(kernels)
    kern_starts = [s for s, *_ in kern_sorted]

    def next_phase_after(t):
        i = bisect.bisect_left(kern_starts, t)
        if i < len(kern_sorted):
            return kern_attr.get(kern_sorted[i][4], "(end)")
        return "(end of capture)"

    gap_by_phase = defaultdict(int)
    for s, e in gaps:
        gap_by_phase[next_phase_after(e)] += e - s
    out.append(f"\n## 6. GPU idle gaps > {args.gap_ms:.0f} ms "
               "(grouped by the phase whose kernel ran next)\n")
    out.append("| waiting for phase | total idle | gaps | largest |")
    out.append("|---|---|---|---|")
    largest = defaultdict(int)
    count = defaultdict(int)
    for s, e in gaps:
        p = next_phase_after(e)
        largest[p] = max(largest[p], e - s)
        count[p] += 1
    for p, v in sorted(gap_by_phase.items(), key=lambda kv: -kv[1]):
        out.append(f"| {p} | {v / MS:.0f} ms | {count[p]} "
                   f"| {largest[p] / MS:.0f} ms |")

    # --- top kernels -----------------------------------------------------------
    kern_by_name = defaultdict(int)
    kern_n = defaultdict(int)
    for s, e, name, *_ in kernels:
        kern_by_name[name] += e - s
        kern_n[name] += 1
    out.append(f"\n## 7. Top {args.top} kernels by total GPU time\n")
    out.append("| kernel | total | launches | avg |")
    out.append("|---|---|---|---|")
    for name in sorted(kern_by_name, key=kern_by_name.get, reverse=True)[: args.top]:
        v, n = kern_by_name[name], kern_n[name]
        out.append(f"| {name} | {v / MS:.0f} ms | {n} | {v / n / 1e3:.0f} µs |")

    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
