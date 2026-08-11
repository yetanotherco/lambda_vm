"""Constructs a REAL secp256k1 point whose `y` lies in the non-canonical band
`y < 2^256 − p = 2^32 + 977`, i.e. a point for which `y + p` is still a 32-byte value.

Why this file exists: PR #879's soundness section claims the `YrLtP` range check is
load-bearing because "such points are constructible: `3 | p−1` makes cubing 3-to-1, so a
small target `y` has a cube-root preimage about a third of the time". That is an existence
claim about the attack surface, and the honest way to review it is to build the point.
`gate/a2_yr_lt_p.py` then uses the point as a concrete forgery instance rather than an
abstract SAT.

Method. Fix a target `y`; the curve equation demands `x³ = y² − 7`. Cubing on `F_p*` is
3-to-1 exactly when `3 | p − 1`, which holds, with 3-adic valuation `v_3(p−1) = 1`. Write
`p − 1 = 3m`, `gcd(m, 3) = 1`. Then `c` is a cubic residue iff `c^m = 1`, and in that case
`c^d` with `3d ≡ 1 (mod m)` is a cube root:

    (c^d)³ = c^{3d} = c^{1 + λm} = c · (c^m)^λ = c.

So sweep `y = 0, 1, 2, …`, keep the first `y` whose `y² − 7` is a cubic residue. About one
in three qualifies, so the sweep is a handful of iterations — the point is found in the
very bottom of the band, far below `2^32 + 977`.

Run: `python small_y_point.py`
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ecsm_affine_ref import (
    N,
    NONCANONICAL_BAND,
    P,
    affine_mul,
    is_on_curve,
    mul,
)

M = (P - 1) // 3
assert (P - 1) % 3 == 0 and M % 3 != 0, "v_3(p-1) must be exactly 1 for the fast cube root"
D = pow(3, -1, M)  # 3·D ≡ 1 (mod m)


def is_cubic_residue(c):
    return pow(c % P, M, P) == 1


def cube_root(c):
    """A cube root of `c` mod p, or None."""
    c %= P
    if c == 0:
        return 0
    if not is_cubic_residue(c):
        return None
    r = pow(c, D, P)
    assert (r * r % P * r - c) % P == 0
    return r


def find_small_y_point(limit=4096):
    """Smallest `y ≥ 1` in the band for which a curve point `(x, y)` exists."""
    for y in range(1, limit):
        x = cube_root((y * y - 7) % P)
        if x is None:
            continue
        assert is_on_curve((x, y)), "cube root did not land on the curve"
        assert y < NONCANONICAL_BAND, "y escaped the non-canonical band"
        return x, y
    return None


def ecsm_instance(q):
    """An ECSM *call* whose output is `q`, so the forged `yR = y_q + p` is reachable
    through the chip rather than only through the curve.

    `k = 2` and `P = 2^{-1}·q` (inverse mod N, the group order) gives `2·P = q`. `k = 2`
    also keeps the chip on its generic path: one ECDAS doubling row, no `k = 1` echo
    (where the drain is forced equal to the seed and `yR` inherits `YG`'s byte checks
    directly)."""
    inv2 = pow(2, -1, N)
    pt = mul(inv2, q)
    assert mul(2, pt) == q
    return 2, pt


def main():
    found = find_small_y_point()
    if found is None:
        print("[FAIL] no small-y point found in the swept range")
        return 1
    x, y = found

    print("Non-canonical band  2^256 - p = 2^32 + 977 =", NONCANONICAL_BAND)
    print()
    print("Curve point with y inside the band:")
    print(f"  y  = {y}  ({y} < {NONCANONICAL_BAND}: {y < NONCANONICAL_BAND})")
    print(f"  x  = 0x{x:064x}")
    print(f"  on curve: {is_on_curve((x, y))}")
    print(f"  y + p    = 0x{(y + P):064x}  (< 2^256: {y + P < 2**256})")
    print(f"  headroom: y + p is {(2**256 - (y + P))} below 2^256")
    print()

    k, pt = ecsm_instance((x, y))
    got = affine_mul(k, pt[0], pt[1])
    print("Reachable through the affine ecall:")
    print(f"  k   = {k}")
    print(f"  xG  = 0x{pt[0]:064x}")
    print(f"  yG  = 0x{pt[1]:064x}")
    print(f"  ->  xR = 0x{got[0]:064x}")
    print(f"      yR = 0x{got[1]:064x}   (== y: {got[1] == y})")
    print()
    print("So an unconstrained witness may publish yR' = yR + p, a DIFFERENT 32-byte")
    print("value that satisfies every byte range check and the ECDAS yR relation with")
    print("q2 reduced by one. YrLtP is what excludes it.")

    out = {
        "band": NONCANONICAL_BAND,
        "small_y_point": {"x": f"{x:064x}", "y": f"{y:064x}"},
        "ecsm_instance": {"k": k, "x_g": f"{pt[0]:064x}", "y_g": f"{pt[1]:064x}"},
        "expected": {"x_r": f"{got[0]:064x}", "y_r": f"{got[1]:064x}"},
        "forged_y_r": f"{(got[1] + P):064x}",
    }
    dest = Path(__file__).parent / "small_y_point.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {dest.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
