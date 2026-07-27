"""Recompute the curve constants from the SEC2 definition and diff them
against the byte arrays hardcoded in crypto/ecsm/src/lib.rs.

Parses the Rust source text directly so a repo edit can't silently desync.
"""

import re
import sys

from ec_ref import P, N, B, GX, GY, recover_even_y

REPO_LIB = "/Users/maurofab/workspace/lambda_vm/crypto/ecsm/src/lib.rs"


def parse_rust_byte_array(src, name):
    m = re.search(rf"pub const {name}: \[u8; \d+\] = \[(.*?)\];", src, re.S)
    assert m, f"{name} not found"
    return bytes(int(t, 16) for t in re.findall(r"0x([0-9A-Fa-f]{2})", m.group(1)))


def main():
    src = open(REPO_LIB).read()
    ok = True

    # Independent recomputation from the standard.
    p_indep = 2**256 - 2**32 - 977
    # SEC2 v2.0 published hex for p (transcribed from the standard):
    p_sec2 = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    n_sec2 = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
    checks = [
        ("p formula == SEC2 hex", p_indep == p_sec2 == P),
        ("N == SEC2 hex", N == n_sec2),
        ("G on curve", (GY * GY - GX**3 - B) % P == 0),
        ("G.y even (canonical form exists)", recover_even_y(GX) is not None),
    ]

    p_bytes = parse_rust_byte_array(src, "P_BYTES")
    n_bytes = parse_rust_byte_array(src, "N_BYTES")
    r_bytes = parse_rust_byte_array(src, "R_BYTES")
    b_m = re.search(r"pub const B: u64 = (\d+);", src)

    checks += [
        ("repo P_BYTES == p (LE)", int.from_bytes(p_bytes, "little") == P),
        ("repo N_BYTES == N (LE)", int.from_bytes(n_bytes, "little") == N),
        ("repo R_BYTES == 3p (LE)", int.from_bytes(r_bytes, "little") == 3 * P),
        ("repo B == 7", b_m is not None and int(b_m.group(1)) == B == 7),
    ]

    for name, res in checks:
        print(f"{'PASS' if res else 'FAIL'}  {name}")
        ok &= res
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
