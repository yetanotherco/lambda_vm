#!/usr/bin/env python3
"""Emit a guest crate's rustflags with `--remap-path-prefix` appended, in
`CARGO_ENCODED_RUSTFLAGS` form.

★ WHY THIS EXISTS. A guest ELF is a program constant: its Merkle roots, its
genesis page set and its cycle count are all read off it and pinned. Without
remapping, the ELF also embeds every absolute path of the machine that built it
— the workspace directory, the cargo registry checkout, the rustup toolchain,
the sysroot — so those constants are partly a function of the filesystem. Two
checkouts of one commit, on one machine, produce different ELFs; two machines
differ by kilobytes. That makes "same commit, same ELF" untrue and unassertable,
and a pin taken in one worktree is not a claim about another's guest.

⚠ WHY A SCRIPT RATHER THAN A LINE IN EACH `.cargo/config.toml`. The prefixes
contain absolute paths that only the build knows, so they cannot be checked in.
The obvious alternative — `RUSTFLAGS` from the Makefile — is wrong: that
variable REPLACES `[target.<triple>].rustflags` rather than extending it, and
the guests' flag sets are not uniform (some carry `link-arg=-e main`, some
`getrandom_backend`, in four different combinations). A blanket value would
silently drop flags that decide whether a guest links or randomises correctly.

So this reads each guest's OWN flags out of its config and appends to them. The
config stays the single source of truth; nothing is duplicated, and a guest that
gains a flag needs no change here.

⛔ WHAT THIS DOES NOT FIX, measured rather than assumed. After remapping, an ELF
has zero `/Users/...` strings, but two differently NAMED checkouts of one commit
still differ (3,948,944 against 3,950,136 bytes on one machine). The residue is
cargo's `-C metadata` disambiguator: cargo derives it from a package's absolute
source path and hands it to rustc before any remapping applies. The boundary is
exact — the guest crate and `lambda-vm-ethrex-crypto`, both PATH packages, get
different disambiguators, while `ethrex-trie` (a git dependency) and `core`
(build-std) are identical. Closing that needs a canonical build path, not a
flag.

What this DOES fix is the larger and more dangerous half: the ELF no longer
depends on the HOST. The home directory, the cargo registry location, the
sysroot and the rustup toolchain path — which carries the host triple, and so
differed between a Linux box and a macOS laptop in every one of the many strings
that embed it — are gone. That is what made "the same sources" produce 3,948,520
on one machine and 3,952,608 on another.

`CARGO_ENCODED_RUSTFLAGS` rather than `RUSTFLAGS` because the encoded form is
separated by `\\x1f` and needs no quoting: `--cfg getrandom_backend="custom"`
survives verbatim, where the space-separated form depends on the shell.
"""

from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path

SEPARATOR = "\x1f"


def config_rustflags(guest_dir: Path, triple: str) -> list[str]:
    """The guest's own flags, exactly as cargo would have applied them.

    Absent config or absent section is not an error: a guest with no flags of
    its own still wants the remaps.
    """
    config = guest_dir / ".cargo" / "config.toml"
    if not config.is_file():
        return []
    with config.open("rb") as handle:
        parsed = tomllib.load(handle)
    target = parsed.get("target", {}).get(triple, {})
    flags = target.get("rustflags", [])
    if not isinstance(flags, list) or not all(isinstance(f, str) for f in flags):
        sys.exit(f"{config}: [target.{triple}].rustflags must be an array of strings")
    return list(flags)


def remap_flags(prefixes: list[tuple[str, str]]) -> list[str]:
    """One `--remap-path-prefix` per real prefix, longest first.

    ⚠ The order matters. rustc applies the LAST matching remapping, so a
    shorter prefix listed later would win over a longer one that is more
    specific — a sysroot inside the workspace, say, would come out labelled as
    workspace rather than sysroot. Sorting longest-first and letting the last
    (shortest) match lose is the stable rule; it also makes the output
    independent of the order the caller happened to pass them in.
    """
    out: list[str] = []
    for real, virtual in sorted(prefixes, key=lambda p: len(p[0]), reverse=True):
        if not real:
            continue
        # No trailing separator: rustc matches on the raw string, and a path
        # equal to the prefix itself should map too.
        out.append(f"--remap-path-prefix={real.rstrip('/')}={virtual}")
    return out


def main() -> None:
    if len(sys.argv) != 4:
        sys.exit(
            "usage: guest_rustflags.py <guest-dir> <workspace-root> <sysroot>\n"
            "       CARGO_HOME and RUSTUP_HOME are read from the environment."
        )
    guest_dir = Path(sys.argv[1]).resolve()
    workspace = Path(sys.argv[2]).resolve()
    sysroot = Path(sys.argv[3]).resolve()

    home = Path.home()
    cargo_home = Path(os.environ.get("CARGO_HOME") or home / ".cargo").resolve()
    # The build-std sources live under the rustup toolchain, and they are the
    # BULK of the embedded paths — every `library/core/src/...` panic location.
    # Omitting this one leaves the ELF host-specific while looking fixed.
    rustup_home = Path(os.environ.get("RUSTUP_HOME") or home / ".rustup").resolve()

    flags = config_rustflags(guest_dir, "riscv64im-lambda-vm-elf")
    flags += remap_flags(
        [
            (str(workspace), "/lambda-vm"),
            (str(cargo_home), "/cargo"),
            (str(rustup_home), "/rustup"),
            (str(sysroot), "/sysroot"),
        ]
    )
    sys.stdout.write(SEPARATOR.join(flags))


if __name__ == "__main__":
    main()
