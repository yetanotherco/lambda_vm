#!/usr/bin/env python3
"""Resolve a raw (address-keyed) folded-stacks file into names + inlined
call chains + file:line, via `addr2line`.

Usage:
    scripts/enrich_flamegraph.py raw_folded.txt --elf target/release/guest.elf > resolved_folded.txt
    scripts/enrich_flamegraph.py raw_folded.txt --elf guest.elf --addr2line-bin llvm-addr2line
"""

import argparse
import subprocess
import sys


DESCRIPTION = (__doc__ or "").split("\n\n")[0]


def parse_args():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("input", help="raw folded-stacks file (address-keyed, from --flamegraph-raw)")
    parser.add_argument("--elf", required=True, help="ELF file the addresses belong to (needs debug info)")
    parser.add_argument(
        "--addr2line-bin",
        default="addr2line",
        help="addr2line binary to invoke (default: addr2line; e.g. llvm-addr2line on macOS)",
    )
    parser.add_argument(
        "--output",
        default="-",
        help="output path for the resolved folded stacks (default: stdout)",
    )
    return parser.parse_args()


def read_raw_stacks(path):
    """Yields (list_of_addr_strings, count) per line of a raw folded file."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            stack, count = line.rsplit(" ", 1)
            yield stack.split(";"), int(count)


def query_addr2line(elf_path, addresses, addr2line_bin):
    """Returns {address: [(function, location), ...]}, innermost inlined
    frame first (addr2line's own order), ending with the physical function.
    """
    if not addresses:
        return {}

    proc = subprocess.Popen(
        [addr2line_bin, "-e", elf_path, "-f", "-i", "-C", "-a"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        text=True,
    )
    stdin_text = "\n".join(addresses) + "\n"
    stdout_text, _ = proc.communicate(input=stdin_text)
    if proc.returncode != 0:
        raise RuntimeError(f"{addr2line_bin} exited with status {proc.returncode}")

    return parse_addr2line_output(addresses, stdout_text)


def is_address_echo(line):
    """True if `line` is an addr2line `-a` echo: "0x" + hex digits only."""
    if not line.startswith("0x") or len(line) <= 2:
        return False
    return all(c in "0123456789abcdefABCDEF" for c in line[2:])


def parse_addr2line_output(addresses, stdout_text):
    result = {}
    lines = stdout_text.splitlines()
    pos = 0

    for addr in addresses:
        if pos >= len(lines):
            break
        pos += 1

        frames = []
        while pos + 1 < len(lines) and not is_address_echo(lines[pos]):
            frames.append((lines[pos], lines[pos + 1]))
            pos += 2
        result[addr] = frames

    return result


def main():
    args = parse_args()

    raw_stacks = list(read_raw_stacks(args.input))

    unique_addresses = list({addr for stack, _ in raw_stacks for addr in stack})
    resolved = query_addr2line(args.elf, unique_addresses, args.addr2line_bin)

    counts = {}
    for stack, count in raw_stacks:
        frame_names = []
        for addr in stack:
            frames = resolved.get(addr)
            if frames and frames[0][0] != "??":
                for function, location in reversed(frames):
                    frame_names.append(f"{function} ({location})")
            else:
                frame_names.append(addr)
        key = ";".join(frame_names)
        counts[key] = counts.get(key, 0) + count

    out = sys.stdout if args.output == "-" else open(args.output, "w")
    try:
        for stack in sorted(counts):
            out.write(f"{stack} {counts[stack]}\n")
    finally:
        if out is not sys.stdout:
            out.close()


if __name__ == "__main__":
    main()
