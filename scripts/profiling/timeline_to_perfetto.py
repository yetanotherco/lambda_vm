#!/usr/bin/env python3
"""Convert an instruments timeline JSON (LAMBDA_VM_TIMELINE_JSON) to Chrome
trace format, viewable in Perfetto (ui.perfetto.dev) or chrome://tracing.

Usage:  timeline_to_perfetto.py timeline.json > trace.json

Spans open/close on the main thread at phase boundaries (see
crypto/stark/src/instruments.rs), so a single track with nested duration
events reproduces the tree. Timestamps are the spans' wall-clock epoch, so a
trace can be compared side-by-side with an nsys report or an nvml_sampler CSV
from the same run.
"""

import json
import sys


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    with open(sys.argv[1]) as f:
        spans = json.load(f)

    events = []
    for s in sorted(spans, key=lambda s: s["order"]):
        events.append(
            {
                "name": s["label"],
                "ph": "X",
                "ts": s["start_ns"] / 1e3,  # chrome trace wants µs
                "dur": s["wall_ns"] / 1e3,
                "pid": 1,
                "tid": 1,
                "args": {"depth": s["depth"], "order": s["order"]},
            }
        )
    json.dump(
        {
            "traceEvents": events,
            "displayTimeUnit": "ms",
        },
        sys.stdout,
    )


if __name__ == "__main__":
    main()
