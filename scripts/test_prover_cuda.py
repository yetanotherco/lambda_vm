#!/usr/bin/env python3
"""Run the full prover card suite and check it against nextest's test list."""

import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


def check_results(expected, report):
    results = ET.parse(report).getroot().findall("testsuite/testcase")
    passed = {
        (test.attrib["classname"], test.attrib["name"])
        for test in results
        if test.find("failure") is None
        and test.find("error") is None
        and test.find("skipped") is None
    }
    if passed == expected and len(results) == len(expected):
        print(f"CUDA prover suite: all {len(expected)} listed tests passed", flush=True)
        return True
    print(f"CUDA prover suite: expected {len(expected)} tests, {len(passed)} passed", file=sys.stderr)
    for binary, name in sorted(expected - passed):
        print(f"  Missing or failed: {binary} {name}", file=sys.stderr)
    for binary, name in sorted(passed - expected):
        print(f"  Unexpected pass: {binary} {name}", file=sys.stderr)
    return False


def main(cargo_args):
    listed = subprocess.check_output(
        ["cargo", "nextest", "list", "--message-format=json", *cargo_args], text=True
    )
    expected = {
        (suite["binary-id"], name)
        for suite in json.loads(listed)["rust-suites"].values()
        for name, test in suite["testcases"].items()
        if not test["ignored"]
    }
    if not expected:
        sys.exit("CUDA prover suite: no non-ignored tests were listed")

    report = Path("target/nextest/cuda/junit.xml")
    report.unlink(missing_ok=True)
    run = subprocess.run(
        ["cargo", "nextest", "run", "--profile=cuda", "--no-fail-fast", "--retries=0", "--color=never", *cargo_args]
    )
    # Nextest runs unit and integration tests. Keep Cargo's doctests in the gate.
    docs = subprocess.run(["cargo", "test", *cargo_args, "--doc", "--", "--test-threads=1"])
    if not report.is_file():
        sys.exit(f"CUDA prover suite: nextest did not write {report}")
    complete = check_results(expected, report)
    return run.returncode or docs.returncode or int(not complete)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
