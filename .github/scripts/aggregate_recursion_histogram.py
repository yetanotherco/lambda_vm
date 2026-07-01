#!/usr/bin/env python3
"""Format the recursion-guest per-function profile as a Markdown PR comment.

`test_recursion_profile_1query`/`_multiquery` print a global top-25 functions
table (folded over all verifier steps, % of total run cycles), followed by
one top-25 table per verifier step (% of that step's own cycles, so the
table shows what dominates *within* the step) — e.g. how much of
`step4:openings` is `keccak`. We parse all of those tables and render them
as Markdown.

    Top 25 functions by cycle count (aggregated over their PCs, all steps; % of total cycles):
      rank          cycles        %    cum %    PCs  function
         1         5335072   24.95%   24.95%     72  <...>::visit_seq::<...>

    Top 25 functions by cycle count — step airs_bus_balance (% of this step's 5129138364 cycles):
      rank          cycles        %    cum %    PCs  function
         1         5335072   24.95%   24.95%     72  <...>::visit_seq::<...>

Reads the test's captured output from argv[1]; writes the Markdown body to
argv[2] (or stdout).
"""

import re
import sys
from collections import OrderedDict

# A per-function summary row: rank, cycles, pct%, cum%, pcs, function.
FN_ROW = re.compile(
    r"^\s*\d+\s+(\d+)\s+([\d.]+)%\s+([\d.]+)%\s+(\d+)\s+(.*\S)\s*$"
)
HEADER_ROW = re.compile(r"^\s*rank\s+cycles")
GLOBAL_TABLE_START = re.compile(
    r"Top \d+ functions by cycle count \(aggregated over their PCs, all steps"
)
STEP_TABLE_START = re.compile(
    r"Top \d+ functions by cycle count — step (\S+) \(% of this step's (\d+) cycles\):"
)
TOTAL_CYCLES = re.compile(r"Total cycles\s*:\s*(\d+)")
UNIQUE_PCS = re.compile(r"Unique PCs\s*:\s*(\d+)")
EXEC_TIME = re.compile(r"Exec time\s*:\s*(\S+)")

GLOBAL_KEY = "__global__"


def parse(text):
    total_cycles = unique_pcs = exec_time = None
    # GLOBAL_KEY -> {"denom": int|None, "rows": [...]}, then one entry per
    # step tag in first-seen order.
    tables = OrderedDict()
    current = None
    skip_header = False
    for line in text.splitlines():
        if total_cycles is None and (m := TOTAL_CYCLES.search(line)):
            total_cycles = int(m.group(1))
        if unique_pcs is None and (m := UNIQUE_PCS.search(line)):
            unique_pcs = int(m.group(1))
        if exec_time is None and (m := EXEC_TIME.search(line)):
            exec_time = m.group(1)

        if GLOBAL_TABLE_START.search(line):
            current = GLOBAL_KEY
            tables[current] = {"denom": total_cycles, "rows": []}
            skip_header = True
            continue
        if m := STEP_TABLE_START.search(line):
            current = m.group(1)
            tables[current] = {"denom": int(m.group(2)), "rows": []}
            skip_header = True
            continue

        if current is None:
            continue
        if skip_header:
            # The header row right after a table-start line; anything else
            # (e.g. a stray blank line) just ends the table early, which is
            # fine — an empty table renders as "no rows".
            skip_header = False
            if HEADER_ROW.match(line):
                continue
        if m := FN_ROW.match(line):
            tables[current]["rows"].append(
                {
                    "cycles": int(m.group(1)),
                    "pct": m.group(2),
                    "cum": m.group(3),
                    "pcs": int(m.group(4)),
                    "fn": m.group(5),
                }
            )
        else:
            current = None

    return total_cycles, unique_pcs, exec_time, tables


def short(name, width=90):
    return name if len(name) <= width else name[: width - 1] + "…"


def render_table(rows, denom_label):
    if not rows:
        return "> _no rows_\n"
    body = "| Rank | Cycles | % | Cum % | PCs | Function |\n"
    body += "|-----:|-------:|--:|------:|----:|----------|\n"
    for i, r in enumerate(rows, 1):
        body += (
            f"| {i} | {r['cycles']:,} | {r['pct']}% | {r['cum']}% | "
            f"{r['pcs']} | `{short(r['fn'])}` |\n"
        )
    last_cum = rows[-1]["cum"]
    body += (
        f"\n<sub>Each function's cycles are summed over all its program counters "
        f"in this table's scope; the top {len(rows)} cover {last_cum}% of "
        f"{denom_label}.</sub>\n"
    )
    return body


def render(total_cycles, unique_pcs, exec_time, tables, title="Recursion guest profile"):
    if not tables.get(GLOBAL_KEY, {}).get("rows"):
        return (
            f"### {title}\n\n"
            "> ⚠️ No per-function rows found in the test output — the run may "
            "have failed before printing the table. Check the workflow logs.\n"
        )

    body = f"### {title}\n\n"
    if total_cycles is not None:
        body += f"**Total cycles:** {total_cycles:,}"
        if unique_pcs is not None:
            body += f" · **Unique PCs:** {unique_pcs:,}"
        if exec_time:
            body += f" · **Exec time:** {exec_time}"
        body += "\n\n"

    global_rows = tables[GLOBAL_KEY]["rows"]
    body += f"#### Top {len(global_rows)} functions by cycles (all steps)\n\n"
    body += render_table(global_rows, "total cycles")

    for step, table in tables.items():
        if step == GLOBAL_KEY:
            continue
        rows, denom = table["rows"], table["denom"]
        denom_note = f" of {denom:,} step cycles" if denom is not None else ""
        body += (
            f"\n<details><summary>Step <code>{step}</code>{denom_note} — "
            f"top {len(rows)} functions</summary>\n\n"
        )
        body += render_table(rows, "this step's cycles")
        body += "\n</details>\n"

    return body


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", help="captured test output to parse")
    ap.add_argument("-o", "--out", help="write Markdown here instead of stdout")
    ap.add_argument(
        "-t",
        "--title",
        default="Recursion guest profile",
        help="section heading (e.g. the test/config name)",
    )
    args = ap.parse_args()

    with open(args.log, "r", errors="replace") as f:
        text = f.read()
    body = render(*parse(text), title=args.title)
    if args.out:
        with open(args.out, "w") as f:
            f.write(body)
    else:
        sys.stdout.write(body)


if __name__ == "__main__":
    main()
