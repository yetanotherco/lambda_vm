#!/usr/bin/env python3
"""Refresh ethrex fixture SHA-256 values in executor/tests/README.md."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


FIXTURES = (
    "ethrex_empty_block.bin",
    "ethrex_simple_tx.bin",
    "ethrex_10_transfers.bin",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def refresh_readme(readme: Path, checksums: dict[str, str]) -> str:
    lines = readme.read_text().splitlines()
    replaced: set[str] = set()

    for index, line in enumerate(lines[:-1]):
        fixture = line.strip()
        if fixture not in checksums:
            continue

        checksum_index = index + 1
        if not lines[checksum_index].startswith("  sha256: "):
            raise SystemExit(f"{readme}: expected sha256 line after {fixture}")

        lines[checksum_index] = f"  sha256: {checksums[fixture]}"
        replaced.add(fixture)

    missing = set(checksums) - replaced
    if missing:
        raise SystemExit(
            f"{readme}: missing fixture sections: {', '.join(sorted(missing))}"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if executor/tests/README.md has stale checksums",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    tests_dir = repo_root / "executor" / "tests"
    readme = tests_dir / "README.md"
    checksums = {fixture: sha256_file(tests_dir / fixture) for fixture in FIXTURES}

    updated = refresh_readme(readme, checksums)
    current = readme.read_text()

    if args.check:
        if updated != current:
            raise SystemExit(f"{readme}: fixture checksums are stale")
    else:
        readme.write_text(updated)

    for fixture in FIXTURES:
        print(f"{fixture}: {checksums[fixture]}")


if __name__ == "__main__":
    main()
