"""Generate vectors.json: the canonical ECSM edge-case suite for the z3 gate
and any future implementation. Values come from the oracle (ec_ref), which is
validated by anchors A (Wycheproof), B (ecdsa PyPI), C (repo k256 differential).

Every entry: x_le / k_le (hex, little-endian byte order as the ABI takes),
x / k (big-endian ints as hex, for humans), expected xr or error kind,
provenance + why-this-case tags.
"""

import json
import random

from ec_ref import EcError, GX, N, P, recover_even_y, x_only_mul_ints

rng = random.Random(424242)


def le_hex(v):
    return v.to_bytes(32, "little").hex()


def random_valid_x():
    while True:
        x = rng.randrange(P)
        if recover_even_y(x) is not None:
            return x


def nonresidue_x():
    while True:
        x = rng.randrange(P)
        if recover_even_y(x) is None:
            return x


def entry(x, k, why, prov="oracle(ec_ref) anchored A/B/C"):
    e = {"x": f"{x:x}", "k": f"{k:x}", "x_le": le_hex(x), "k_le": le_hex(k),
         "why": why, "provenance": prov}
    try:
        e["xr"] = f"{x_only_mul_ints(x, k):x}"
    except EcError as err:
        e["error"] = err.kind
    return e


def main():
    vs = []
    # schedule-shaped scalars on G
    vs.append(entry(GX, 1, "k=1 echo: no steps, xr == xG"))
    vs.append(entry(GX, 2, "k=2: single double row"))
    vs.append(entry(GX, 3, "k=3: double+add, minimal add path"))
    vs.append(entry(GX, N - 1, "k=N-1: x((N-1)P) == x(P) since (N-1)P = -P"))
    vs.append(entry(GX, N - 2, "k=N-2"))
    for i in (1, 8, 64, 128, 255):
        vs.append(entry(GX, 2**i, f"k=2^{i}: all-double schedule, len_k={i}"))
        vs.append(entry(GX, 2**i - 1, f"k=2^{i}-1: double+add every row"))
    vs.append(entry(GX, int("10" * 128, 2) % N, "alternating 10 bits"))
    vs.append(entry(GX, int("01" * 128, 2), "alternating 01 bits"))
    vs.append(entry(GX, (1 << 200) + 1, "long zero run between MSB and LSB"))
    vs.append(entry(GX, ((1 << 56) - 1) << 100, "long one-run mid-scalar"))

    # random points x random/edge scalars
    for i in range(20):
        x = random_valid_x()
        vs.append(entry(x, rng.randrange(1, N), f"random point/scalar #{i}"))
    x = random_valid_x()
    for k in (1, N - 1, 2**255):
        vs.append(entry(x, k, "edge scalar on random point"))

    # error paths (exact contract)
    vx = random_valid_x()
    vs.append(entry(vx, 0, "k=0 -> ScalarIsZero"))
    vs.append(entry(vx, N, "k=N -> ScalarOutOfRange"))
    vs.append(entry(vx, N + 1, "k=N+1 -> ScalarOutOfRange"))
    vs.append(entry(vx, 2**256 - 1, "k=2^256-1 -> ScalarOutOfRange"))
    vs.append(entry(P, 5, "x=p -> CoordinateOutOfRange"))
    vs.append(entry(P + 1, 5, "x=p+1 -> CoordinateOutOfRange"))
    vs.append(entry(2**256 - 1, 5, "x=2^256-1 -> CoordinateOutOfRange"))
    vs.append(entry(nonresidue_x(), 5, "non-residue x -> NotOnCurve"))
    vs.append(entry(nonresidue_x(), rng.randrange(1, N), "non-residue x, random k"))
    # check-order witnesses
    vs.append(entry(P, 0, "k=0 beats x=p: ScalarIsZero (order witness)"))
    vs.append(entry(nonresidue_x(), N, "k=N beats non-residue: ScalarOutOfRange (order witness)"))

    json.dump({"curve": "secp256k1", "function": "xr = x(k*P), P=(x, even y)",
               "abi": "32-byte little-endian x, k, xr; errors trap the VM",
               "generated": "2026-07-24 ec-oracle",
               "vectors": vs}, open("vectors.json", "w"), indent=1)
    ok = sum(1 for v in vs if "xr" in v)
    err = sum(1 for v in vs if "error" in v)
    print(f"vectors.json: {len(vs)} vectors ({ok} valid, {err} error-path)")


if __name__ == "__main__":
    main()
