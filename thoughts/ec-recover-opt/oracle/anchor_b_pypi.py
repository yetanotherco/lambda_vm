"""Anchor B: random differential of the oracle against the `ecdsa` PyPI
package (pure-Python, independent lineage from both our oracle and k256).

- 500 random (k, P) pairs: x(k·P) oracle vs ecdsa Point arithmetic
- 60 random k·G checks
- edge scalars: 1, 2, 3, N-1, N-2, 2^i and 2^i - 1 for schedule-shaped edges

Deterministic seed so the run is reproducible.

coincurve (libsecp256k1) was attempted as a third lineage but has no wheel
for this Python and fails to build from source; noted in README.
"""

import random

from ecdsa import SECP256k1
from ecdsa.ellipticcurve import Point

from ec_ref import N, P, recover_even_y, scalar_mul, x_only_mul_ints

rng = random.Random(20260724)

CURVE = SECP256k1.curve
G_LIB = SECP256k1.generator


def random_point():
    while True:
        x = rng.randrange(P)
        y = recover_even_y(x)
        if y is not None:
            return x, y


def lib_mul_x(k, x, y):
    pt = Point(CURVE, x, y) * k
    return pt.x()


def main():
    fails = 0

    # 1. random (k, P)
    for i in range(500):
        x, y = random_point()
        k = rng.randrange(1, N)
        got = x_only_mul_ints(x, k)
        want = lib_mul_x(k, x, y)
        if got != want:
            fails += 1
            print(f"FAIL random pair {i}: k={k:x} x={x:x} got={got:x} want={want:x}")

    # 2. k·G
    for i in range(60):
        k = rng.randrange(1, N)
        got = x_only_mul_ints(SECP256k1.generator.x(), k) if False else scalar_mul(k, (G_LIB.x(), G_LIB.y()))[0]
        want = (G_LIB * k).x()
        if got != want:
            fails += 1
            print(f"FAIL kG {i}: k={k:x}")

    # 3. edge scalars on a fixed random point and on G
    edges = [1, 2, 3, N - 1, N - 2]
    for i in (1, 8, 64, 128, 255):
        edges += [2**i, 2**i - 1]
    edges += [int("10" * 128, 2), int("01" * 128, 2), 2**256 % N]  # alternating bits
    x, y = random_point()
    for k in edges:
        k = k % N or 1
        for (px, py) in ((x, y), (G_LIB.x(), G_LIB.y())):
            got = x_only_mul_ints(px, k)
            want = lib_mul_x(k, px, py)
            if got != want:
                fails += 1
                print(f"FAIL edge k={k:x} x={px:x}: got={got:x} want={want:x}")

    n_edge = len(edges) * 2
    print(f"ANCHOR B: {'PASS' if fails == 0 else 'FAIL'} "
          f"(500 random pairs + 60 kG + {n_edge} edge cases, {fails} failures)")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
