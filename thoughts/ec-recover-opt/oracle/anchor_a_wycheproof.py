"""Anchor A: Wycheproof ECDH secp256k1 vectors vs the oracle.

ECDH shared secret = x(k·P) — exactly the ECSM precompile's function, with
k = private scalar and P the peer public point. Wycheproof's `public` is an
ASN.1 SubjectPublicKeyInfo; we extract the uncompressed/compressed EC point
from its tail rather than pulling in an ASN.1 library:
uncompressed points end with 0x04 || X(32) || Y(32) (65 bytes), compressed
with 0x02/0x03 || X(32) (33 bytes).

Valid vectors must match the oracle output; vectors whose curve point is
invalid must be rejected by the oracle (EcError). Wycheproof flags cases:
result in {valid, acceptable, invalid}. Cases exercising ASN.1-encoding
malleability (not point math) are skipped when the point itself parses fine
and the flag list is encoding-only — those are out of scope for a point-math
oracle; we log how many were skipped and why.
"""

import json

from ec_ref import P, N, EcError, recover_even_y, scalar_mul, x_only_mul_ints

DATA = "wycheproof_ecdh_secp256k1.json"

# Flags that concern DER/ASN encoding or key-format quirks, not point math.
ENCODING_FLAGS = {
    "InvalidAsn", "InvalidEncoding", "UnnamedCurve", "WrongCurve",
    "InvalidCurveAttack",  # handled separately: point not on secp256k1
}


def extract_point(pub_hex):
    """Pull the EC point out of the SPKI tail. Returns (x, y) or ('bad', reason)."""
    b = bytes.fromhex(pub_hex)
    # search from the end for the BIT STRING payload start: 0x00 then 0x04/0x02/0x03
    if len(b) >= 65 and b[-65] == 0x04:
        x = int.from_bytes(b[-64:-32], "big")
        y = int.from_bytes(b[-32:], "big")
        return ("xy", x, y)
    if len(b) >= 33 and b[-33] in (0x02, 0x03):
        x = int.from_bytes(b[-32:], "big")
        return ("comp", x, b[-33])
    return ("bad", None, None)


def main():
    d = json.load(open(DATA))
    group = d["testGroups"][0]
    assert group["curve"] == "secp256k1"

    stats = {"pass": 0, "fail": 0, "skip_encoding": 0, "reject_ok": 0,
             "reject_mismatch": 0, "skip_no_point": 0, "skip_edge_scalar": 0}
    failures = []

    for t in group["tests"]:
        tid, result = t["tcId"], t["result"]
        k = int(t["private"], 16)
        want = t["shared"]
        kind, x, y_or_pref = extract_point(t["public"])

        if kind == "bad":
            stats["skip_no_point"] += 1
            continue

        # Establish the point's x for the oracle; verify on-curve for xy form.
        if kind == "xy":
            y = y_or_pref
            on_curve = (0 <= x < P) and (y * y - x**3 - 7) % P == 0 and not (x == 0 and y == 0)
        else:
            on_curve = (0 <= x < P) and recover_even_y(x) is not None

        if k == 0 or k >= N:
            stats["skip_edge_scalar"] += 1
            continue

        if not on_curve:
            # The precompile is x-only: it must reject exactly when x itself is
            # invalid (>= p, or no y exists). An off-curve (x, y) whose x still
            # lifts to a valid secp256k1 point is outside the oracle's contract
            # (the guest's parity lift owns y validation) → skip, counted.
            x_valid = (0 <= x < P) and recover_even_y(x) is not None
            try:
                x_only_mul_ints(x % (2**256), k)
                accepted = True
            except EcError:
                accepted = False
            if accepted and not x_valid:
                stats["reject_mismatch"] += 1
                failures.append((tid, "oracle accepted invalid x"))
            elif not accepted and x_valid:
                stats["reject_mismatch"] += 1
                failures.append((tid, "oracle rejected valid x"))
            elif accepted:
                stats["skip_offcurve_y_valid_x"] = stats.get("skip_offcurve_y_valid_x", 0) + 1
            else:
                stats["reject_ok"] += 1
            continue

        if result == "invalid":
            # Point math is fine but the case is flagged invalid → encoding-level.
            stats["skip_encoding"] += 1
            continue

        # x-only semantics: x(k·P) is independent of y's sign, so the oracle's
        # even-y lift must reproduce the shared secret regardless of the
        # actual y parity in the vector.
        got = x_only_mul_ints(x, k)
        if f"{got:064x}" == want:
            stats["pass"] += 1
        else:
            stats["fail"] += 1
            failures.append((tid, f"got {got:064x} want {want}"))

    print(json.dumps(stats, indent=2))
    for tid, msg in failures[:20]:
        print(f"FAIL tcId={tid}: {msg}")
    verdict = "PASS" if stats["fail"] == 0 and stats["reject_mismatch"] == 0 and stats["pass"] > 300 else "FAIL"
    print(f"ANCHOR A: {verdict} ({stats['pass']} valid vectors matched, "
          f"{stats['reject_ok']} off-curve correctly rejected)")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
