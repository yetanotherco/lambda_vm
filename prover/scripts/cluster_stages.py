#!/usr/bin/env python3
"""Print range-attempt host timelines from coordinator and worker logs.

Usage: python3 prover/scripts/cluster_stages.py coordinator.log worker-0.log
For the serial reference, pass its single log, which contains the logical
owner lanes. Align logs from the same host or from synchronized clocks.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re


STAGE = re.compile(
    r"\bCLUSTER STAGE (C|W\d+) (\S+) (\S+) t=(\d+\.\d{6}) unix_ms=(\d+)"
    r"(?: seconds=(\d+\.\d{6}))?\s*$"
)
WAIT_ENDS = {
    "wait-input": "input-received",
    "wait-index-base": "index-base-received",
    "wait-range-readers": "range-readers-done",
    "wait-assign": "assign-received",
    "wait-receipts": "receipts-received",
    "wait-bind": "bind-received",
    "wait-leaf-relay": "leaf-relay-done",
    "wait-finish": "finish-received",
}


@dataclass(frozen=True)
class Event:
    stage: str
    t: float
    unix_ms: int
    seconds: str | None


def read_logs(paths):
    attempts = defaultdict(lambda: defaultdict(list))
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as log:
            for line in log:
                if match := STAGE.search(line):
                    role, attempt, stage, t, unix_ms, seconds = match.groups()
                    attempts[attempt][role].append(
                        Event(stage, float(t), int(unix_ms), seconds)
                    )
    return attempts


def wait_ends(stage):
    if stage.startswith("wait-worker-connect-"):
        return (stage.replace("wait-worker-connect-", "worker-connected-", 1),)
    if stage.startswith("wait-finish-W"):
        peer = stage.removeprefix("wait-finish-")
        return (f"finish-received-{peer}", f"finish-failed-{peer}")
    return (WAIT_ENDS[stage],) if stage in WAIT_ENDS else ()


def union_seconds(intervals):
    seconds = 0.0
    end = 0.0
    for start, stop in sorted(intervals):
        seconds += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return seconds


def print_split(role, events):
    at = {event.stage: event.t for event in events}
    waits = []
    parallel = []
    elapsed = events[-1].t
    if "reference-range-start" in at and role.startswith("W"):
        # A reference owner is a lane on the same process clock. Its two
        # active intervals exclude the intervening work of the other owners.
        elapsed = at.get("range-build-done", elapsed) - at["reference-range-start"]
        if "reference-leaf-start" in at:
            elapsed += at.get("leaf-prove-end", events[-1].t) - at["reference-leaf-start"]
    incomplete = []
    for event in events:
        if not event.stage.startswith("wait-"):
            continue
        ends = [at[name] for name in wait_ends(event.stage) if name in at]
        if not ends:
            incomplete.append(event.stage)
            continue
        target = parallel if event.stage == "wait-index-base" else waits
        end = min(ends)
        target.append((event.t, end))
        print(f"  {role} {event.stage}: {end - event.t:.6f}s")
    waiting = union_seconds(waits)
    print(
        f"{role}: {elapsed:.6f}s elapsed = {elapsed - waiting:.6f}s compute/other"
        f" + {waiting:.6f}s wait; {union_seconds(parallel):.6f}s parallel index-base wait"
    )
    if incomplete:
        print(f"  Missing wait end: {', '.join(incomplete)}")


def print_attempt(attempt, lanes):
    roles = sorted(lanes, key=lambda role: -1 if role == "C" else int(role[1:]))
    origins_ms = {}
    reference = any(
        event.stage == "reference-range-start" for events in lanes.values() for event in events
    )
    for role in roles:
        lanes[role].sort(key=lambda event: event.t)
        first = lanes[role][0]
        origins_ms[role] = first.unix_ms - 1000 * first.t
    origin = origins_ms[roles[0]]
    rows = [
        (event.t + (0 if reference else (origins_ms[role] - origin) / 1000), role, event)
        for role in roles
        for event in lanes[role]
    ]
    rows.sort(key=lambda row: (row[0], row[1]))
    width = max(len("stage"), *(len(event.stage) for _, _, event in rows))
    print(f"Attempt {attempt} ({'one-process reference' if reference else 'TCP'}, aligned seconds)")
    print(f"{'stage':<{width}}  " + "  ".join(f"{role + ' t':>10}" for role in roles)
          + "  group seconds")
    for t, role, event in rows:
        cells = [" " * 10 if lane != role else f"{t:10.6f}" for lane in roles]
        line = f"{event.stage:<{width}}  " + "  ".join(cells)
        if event.seconds:
            line += f"  {event.seconds}"
        print(line.rstrip())
    print("Role offsets have millisecond granularity. Wait totals use the union of host waits.")
    print("Index-base waits can overlap execution and appear beside the split.")
    for role in roles:
        print_split(role, lanes[role])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", metavar="LOG", type=Path, nargs="+")
    args = parser.parse_args()
    attempts = read_logs(args.logs)
    if not attempts:
        parser.error("no CLUSTER STAGE lines found")
    for attempt, lanes in sorted(
        attempts.items(),
        key=lambda item: min(event.unix_ms for events in item[1].values() for event in events),
    ):
        print_attempt(attempt, lanes)
        print()


if __name__ == "__main__":
    main()
