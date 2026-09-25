#!/usr/bin/env python3
"""Read `cargo ... --no-run --message-format=json[-render-diagnostics]` on stdin and print
one thing from it:

    cargo_test_bin.py exe --name lambda_vm_prover --kind lib   # the unit-test binary of a lib
    cargo_test_bin.py exe --name rpx_occupancy_sweep --kind test
    cargo_test_bin.py outdir --package math-cuda              # that package's build-script OUT_DIR

Exact artifacts from cargo itself, rather than `ls -t target/release/deps/<name>-*`, which
picks whichever build of any feature set happened to be newest. Exits 1 when absent.
"""
import argparse
import json
import re
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)
    e = sub.add_parser("exe")
    e.add_argument("--name", required=True, help="cargo target name (lib: the crate name with underscores)")
    e.add_argument("--kind", default=None, help="target kind to require: lib, test, bin, ...")
    o = sub.add_parser("outdir")
    o.add_argument("--package", required=True, help="package name, e.g. math-cuda")
    a = ap.parse_args(argv)
    # package ids: "math-cuda 0.1.0 (path+file:///…)" (old cargo) or "path+file:///…/math-cuda#0.1.0" (new)
    pkg = re.compile(r"(^|[/ ])" + re.escape(getattr(a, "package", "") or "\0") + r"([ #@]|$)")
    found = None
    for line in sys.stdin:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if a.what == "exe" and msg.get("reason") == "compiler-artifact":
            tgt = msg.get("target", {})
            if (tgt.get("name") == a.name and msg.get("executable") and msg.get("profile", {}).get("test")
                    and (a.kind is None or a.kind in tgt.get("kind", []))):
                found = msg["executable"]
        elif a.what == "outdir" and msg.get("reason") == "build-script-executed":
            if pkg.search(msg.get("package_id", "")) and msg.get("out_dir"):
                found = msg["out_dir"]
    if not found:
        sys.stderr.write("cargo_test_bin.py: no matching {} in cargo's messages\n".format(a.what))
        return 1
    print(found)
    return 0


if __name__ == "__main__":
    sys.exit(main())
