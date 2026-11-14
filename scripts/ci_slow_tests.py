#!/usr/bin/env python3
"""The filter GitHub CI's prover shards run under: every test of .config/ci-slow-tests.txt skipped, nothing else.

    python3 scripts/ci_slow_tests.py expr
        prints the nextest filter expression (`-E`): not (package(=P) & (test(=N1) | test(=N2) | ...)), one group a
        package, exact names only
    python3 scripts/ci_slow_tests.py check <archive.tar.zst>
        lists the archive's tests twice, unfiltered and under the expression, and fails (exit 1) unless the filter
        skips exactly the listed tests the archive holds and every test binary keeps at least one test; prints the
        counts and the listed names the archive no longer holds (renamed or removed tests: they run, nothing is lost)

The list is read relative to the repository root (this file's parent directory's parent).
"""
import json
import os
import subprocess
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIST = os.path.join(ROOT, ".config", "ci-slow-tests.txt")


def listed():
    """[(package, test name)] in file order, comments and blank lines dropped."""
    out = []
    with open(LIST) as f:
        for n, line in enumerate(f, 1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                sys.exit(f"{LIST}:{n}: want `<package> <test name>`, got {line!r}")
            out.append((parts[0], parts[1]))
    return out


def expr():
    by_pkg = defaultdict(list)
    for pkg, name in listed():
        by_pkg[pkg].append(name)
    if not by_pkg:
        return "all()"
    groups = [f"(package(={pkg}) & ({' | '.join(f'test(={n})' for n in names)}))" for pkg, names in by_pkg.items()]
    return f"not ({' | '.join(groups)})"


def run_list(archive, filt):
    cmd = ["cargo", "nextest", "list", "--archive-file", archive, "--message-format", "json"]
    if filt:
        cmd += ["-E", filt]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=ROOT).stdout
    doc = json.loads(out[out.index("{"):])
    tests = set()
    for suite in doc["rust-suites"].values():
        binary = suite["binary-id"]
        for name, case in suite["testcases"].items():
            match = case.get("filter-match", {}).get("status", "matches")
            if match == "matches" and not case.get("ignored", False):
                tests.add((binary, name))
    return tests


def check(archive):
    full = run_list(archive, None)
    kept = run_list(archive, expr())
    names = {name for _, name in listed()}
    skipped = full - kept
    held = {(b, n) for b, n in full if n in names}
    stale = sorted(names - {n for _, n in full})
    bins = defaultdict(lambda: [0, 0])
    for b, _ in full:
        bins[b][0] += 1
    for b, _ in kept:
        bins[b][1] += 1
    emptied = sorted(b for b, (n_full, n_kept) in bins.items() if n_full and not n_kept)
    print(f"ci-slow-tests: {len(full)} tests in the archive, {len(kept)} run on GitHub, {len(skipped)} skipped "
          f"({len(names)} listed, {len(stale)} no longer in the archive)")
    for n in stale:
        print(f"  listed, not in the archive (runs nowhere under this name; drop the line): {n}")
    bad = []
    if skipped != held:
        extra = sorted(skipped - held)[:10]
        bad.append(f"the filter skips {len(skipped)} tests, the list names {len(held)} of them: extra {extra}")
    if emptied:
        bad.append(f"the filter leaves no test in {emptied}")
    for b in bad:
        print(f"ci-slow-tests: FAIL: {b}")
    return 1 if bad else 0


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "expr":
        print(expr())
        return 0
    if len(sys.argv) == 3 and sys.argv[1] == "check":
        return check(sys.argv[2])
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
