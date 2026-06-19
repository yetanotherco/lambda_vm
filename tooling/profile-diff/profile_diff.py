#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Diff two Lambda VM profiles and report what moved.

Consumes the folded-stack format emitted by `cli execute --flamegraph` (and the
syscall-aware frames), where each line is `frame;frame;frame <count>`. Produces:

  * a regression table on stdout: the frames whose count changed the most,
    biggest absolute movers first, with before/after/delta/percent columns; and
  * optionally, differential folded stacks (`--folded-out`) where each frame's
    count is its delta, suitable for `inferno-flamegraph --negate` style diff
    rendering.

`before` is the baseline; `after` is the new run. A positive delta means the
frame got *more* expensive in `after`.

Examples:
    # human-readable regression table
    uv run tooling/profile-diff/profile_diff.py base.folded new.folded

    # only show frames that moved by >=1000 and render a diff flamegraph
    uv run tooling/profile-diff/profile_diff.py base.folded new.folded \
        --min-delta 1000 --folded-out diff.folded
    cat diff.folded | inferno-flamegraph > diff.svg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_folded(path: Path) -> dict[str, int]:
    """Parse a folded-stack file into {stack: count}.

    Each non-empty line is `<stack> <count>`. The count is the last
    whitespace-separated token, so frame names may contain spaces. Lines without
    a trailing integer count are skipped (with a warning) rather than aborting
    the diff.
    """
    counts: dict[str, int] = {}
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        idx = line.rfind(" ")
        if idx == -1:
            print(f"{path}:{lineno}: no count, skipping: {line!r}", file=sys.stderr)
            continue
        stack, count_str = line[:idx].strip(), line[idx + 1 :].strip()
        try:
            count = int(count_str)
        except ValueError:
            print(
                f"{path}:{lineno}: count {count_str!r} is not an integer, skipping",
                file=sys.stderr,
            )
            continue
        # Folded files should have unique stacks, but sum defensively.
        counts[stack] = counts.get(stack, 0) + count
    return counts


def diff_counts(
    before: dict[str, int], after: dict[str, int]
) -> list[tuple[str, int, int, int]]:
    """Return [(stack, before_count, after_count, delta)] for every stack that
    appears in either profile, sorted by descending |delta| then by stack."""
    stacks = set(before) | set(after)
    rows = []
    for stack in stacks:
        b = before.get(stack, 0)
        a = after.get(stack, 0)
        rows.append((stack, b, a, a - b))
    rows.sort(key=lambda r: (-abs(r[3]), r[0]))
    return rows


def fmt_pct(before: int, after: int) -> str:
    """Percent change from before to after; handles the zero-baseline case."""
    if before == 0:
        return "new" if after != 0 else "0.0%"
    return f"{(after - before) / before * 100:+.1f}%"


def print_table(
    rows: list[tuple[str, int, int, int]],
    total_before: int,
    total_after: int,
    min_delta: int,
    top: int | None,
) -> None:
    shown = [r for r in rows if abs(r[3]) >= min_delta]
    if top is not None:
        shown = shown[:top]

    name_w = max((len(r[0]) for r in shown), default=5)
    name_w = min(max(name_w, 16), 80)

    print("=== PROFILE DIFF (after - before) ===")
    delta_total = total_after - total_before
    print(
        f"  total: {total_before} -> {total_after} "
        f"(delta {delta_total:+}, {fmt_pct(total_before, total_after)})"
    )
    print()
    print(f"  {'Frame':<{name_w}} {'Before':>14} {'After':>14} {'Delta':>14} {'%':>8}")
    print(f"  {'-' * (name_w + 52)}")
    for stack, b, a, d in shown:
        label = stack if len(stack) <= name_w else "..." + stack[-(name_w - 3) :]
        print(f"  {label:<{name_w}} {b:>14} {a:>14} {d:>+14} {fmt_pct(b, a):>8}")
    hidden = len(rows) - len(shown)
    if hidden > 0:
        print(f"  ({hidden} frames below threshold or beyond --top not shown)")


def write_folded_diff(rows: list[tuple[str, int, int, int]], out: Path) -> None:
    """Write differential folded stacks: each frame's count is |delta|, with a
    `+`/`-` suffix appended to the leaf so the direction survives rendering."""
    lines = []
    for stack, _b, _a, d in rows:
        if d == 0:
            continue
        direction = "+" if d > 0 else "-"
        # Tag the leaf frame so the diff direction is visible in the flamegraph.
        lines.append(f"{stack} [{direction}] {abs(d)}")
    out.write_text("\n".join(lines) + ("\n" if lines else ""))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path, help="baseline folded-stack file")
    parser.add_argument("after", type=Path, help="new folded-stack file")
    parser.add_argument(
        "--min-delta",
        type=int,
        default=1,
        help="hide frames whose |delta| is below this (default: 1)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="show only the N biggest movers (default: all above --min-delta)",
    )
    parser.add_argument(
        "--folded-out",
        type=Path,
        default=None,
        help="also write differential folded stacks here (for a diff flamegraph)",
    )
    args = parser.parse_args()

    for p in (args.before, args.after):
        if not p.is_file():
            print(f"error: not a file: {p}", file=sys.stderr)
            return 2

    before = parse_folded(args.before)
    after = parse_folded(args.after)
    rows = diff_counts(before, after)

    print_table(
        rows,
        total_before=sum(before.values()),
        total_after=sum(after.values()),
        min_delta=args.min_delta,
        top=args.top,
    )

    if args.folded_out is not None:
        write_folded_diff(rows, args.folded_out)
        print(f"\nDifferential folded stacks written to {args.folded_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
