#!/usr/bin/env python3
"""Format the recursion-guest per-function profile as a Markdown PR comment.

`test_recursion_pc_histogram` prints a per-function summary table (cycles folded
over each function's PCs, computed across the *full* histogram) followed by a
per-address detail table. We extract the per-function table — the view that
shows where the cycles actually go — and render it as Markdown.

    Top 25 functions by cycle count (aggregated over their PCs):
    rank          cycles        %    cum %    PCs  function (file:line)
       1         5335072   24.95%   24.95%     72  <...>::visit_seq::<...>

Reads the test's captured output from argv[1]; writes the Markdown body to
argv[2] (or stdout).
"""

import re
import sys

# A per-function summary row: rank, cycles, pct%, cum%, pcs, function.
# Distinguished from the per-PC detail rows by the absence of a 0x<pc> column.
FN_ROW = re.compile(
    r"^\s*\d+\s+(\d+)\s+([\d.]+)%\s+([\d.]+)%\s+(\d+)\s+(.*\S)\s*$"
)
FN_TABLE_START = re.compile(r"Top \d+ functions by cycle count")
PC_TABLE_START = re.compile(r"Top \d+ PCs by cycle count")
TOTAL_CYCLES = re.compile(r"Total cycles\s*:\s*(\d+)")
UNIQUE_PCS = re.compile(r"Unique PCs\s*:\s*(\d+)")
EXEC_TIME = re.compile(r"Exec time\s*:\s*(\S+)")


def parse(text):
    total_cycles = unique_pcs = exec_time = None
    rows = []
    in_fn_table = False
    for line in text.splitlines():
        if total_cycles is None and (m := TOTAL_CYCLES.search(line)):
            total_cycles = int(m.group(1))
        if unique_pcs is None and (m := UNIQUE_PCS.search(line)):
            unique_pcs = int(m.group(1))
        if exec_time is None and (m := EXEC_TIME.search(line)):
            exec_time = m.group(1)
        if FN_TABLE_START.search(line):
            in_fn_table = True
            continue
        if PC_TABLE_START.search(line):
            in_fn_table = False
            continue
        if in_fn_table and (m := FN_ROW.match(line)):
            rows.append(
                {
                    "cycles": int(m.group(1)),
                    "pct": m.group(2),
                    "cum": m.group(3),
                    "pcs": int(m.group(4)),
                    "fn": m.group(5),
                }
            )
    return total_cycles, unique_pcs, exec_time, rows


def short(name, width=90):
    return name if len(name) <= width else name[: width - 1] + "…"


def render(total_cycles, unique_pcs, exec_time, rows, title="Recursion guest profile"):
    if not rows:
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

    body += f"#### Top {len(rows)} functions by cycles (folded over their PCs)\n\n"
    body += "| Rank | Cycles | % | Cum % | PCs | Function |\n"
    body += "|-----:|-------:|--:|------:|----:|----------|\n"
    for i, r in enumerate(rows, 1):
        body += (
            f"| {i} | {r['cycles']:,} | {r['pct']}% | {r['cum']}% | "
            f"{r['pcs']} | `{short(r['fn'])}` |\n"
        )

    last_cum = rows[-1]["cum"]
    body += (
        f"\n<sub>Each function's cycles are summed over all its program counters "
        f"across the full histogram; the top {len(rows)} cover {last_cum}% of total "
        f"cycles. Percentages are of total cycles.</sub>\n"
    )
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
