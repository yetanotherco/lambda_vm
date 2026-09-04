"""Anchor harness for the ECSM-affine oracle.

Anchors are what make the gate's UNSATs meaningful: they establish that
`ecsm_affine_ref.py` defines the SAME function the executor computes and the chip claims
to prove. Each anchor is independent — a missing fixture or a missing optional package
SKIPs only itself — the BLAKE3 harness's cascade defect, recorded under "Harness defects"
in `thoughts/blake3/README.md` on branch feat/blake3-accelerator, is deliberately not
repeated here.

  A0  repo constants   — p, N, b, 3p parsed out of `crypto/ecsm/src/lib.rs` and compared
                         against the values recomputed here.
  A1  curve/group      — G on curve; N·G = O; published small multiples of G; k·(N−k)
                         symmetry; associativity spot-checks.
  A2  x-only agreement — `affine_mul(k, xG, even_lift(xG)).x == x_only_mul(k, xG)`. The new
                         ecall must not change the pre-existing one's answer.
  A3  root dependence  — `affine_mul(k, x, p−y) == (X, p−Y)`, and `x` is invariant. This is
                         the parity gap the AIR's yG-read closes; the anchor pins that it is
                         real, not hypothetical.
  A4  validation       — the executor's accept/reject set: 0 < k < N, xG < p, yG < p,
                         on-curve, plus the "k = 1 and k = N−1 are ordinary" claim.
  A5  ABI predicates   — `addr_limb_ok`, the overlap guard as exact interval disjointness,
                         and the u64-wrap negative control.
  A6  ecrecover use    — the y-from-x reconstruction the affine ecall REPLACES agrees with
                         the y the chip now returns, over random instances. Anchors the
                         claim that dropping `solve_y` is semantics-preserving.
  A7  optional cross   — differential against the `ecdsa`/`coincurve` PyPI package if
                         installed (SKIP otherwise).

Run: `python test_oracle.py`
"""

import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ecsm_affine_ref import (
    B,
    G,
    GX,
    GY,
    N,
    P,
    EcsmError,
    add,
    addr_limb_ok,
    affine_mul,
    from_le32,
    inv,
    is_on_curve,
    mul,
    neg,
    operands_disjoint,
    operands_disjoint_u64_buggy,
    recover_y_canonical,
    sqrt_mod_p,
    to_le32,
    x_only_mul,
)

random.seed(0xEC5A)


def find_repo_root(start=None):
    """The lambda_vm root, located by marker (workspace `Cargo.toml` next to `prover/`)
    rather than by a hard-coded `parents[N]`, which breaks silently when the campaign
    directory moves. Anchor A0 reads repo constants by path, so a wrong root would compare
    against nothing; returns None so A0 SKIPs with a named reason instead of guessing."""
    here = (start or Path(__file__)).resolve()
    for cand in here.parents:
        if (cand / "Cargo.toml").is_file() and (cand / "prover").is_dir():
            if "[workspace]" in (cand / "Cargo.toml").read_text():
                return cand
    return None


REPO = find_repo_root()
results = []


def report(name, verdict, detail=""):
    results.append((name, verdict, detail))
    print(f"[{verdict:5}] {name}  {detail}")


# ── A0: repo constants ───────────────────────────────────────────────────────

def _parse_le_bytes(src, name):
    m = re.search(rf"pub const {name}: \[u8; \d+\] = \[(.*?)\];", src, re.S)
    if not m:
        return None
    vals = [int(v, 16) for v in re.findall(r"0x([0-9A-Fa-f]{2})", m.group(1))]
    return int.from_bytes(bytes(vals), "little")


def a0_constants():
    if REPO is None:
        report("A0 repo constants", "SKIP",
               "no lambda_vm repo root found above this file")
        return
    src_path = REPO / "crypto" / "ecsm" / "src" / "lib.rs"
    if not src_path.exists():
        report("A0 repo constants", "SKIP", f"{src_path} not found")
        return
    src = src_path.read_text()
    checks = {
        "p (P_BYTES)": (_parse_le_bytes(src, "P_BYTES"), P),
        "N (N_BYTES)": (_parse_le_bytes(src, "N_BYTES"), N),
        "3p (R_BYTES)": (_parse_le_bytes(src, "R_BYTES"), 3 * P),
    }
    m = re.search(r"pub const B: u64 = (\d+);", src)
    checks["b (B)"] = (int(m.group(1)) if m else None, B)
    bad = [k for k, (got, want) in checks.items() if got != want]
    report("A0 repo constants", "PASS" if not bad else "FAIL",
           f"{len(checks)} constants parsed from crypto/ecsm/src/lib.rs match"
           if not bad else f"mismatched: {bad}")


# ── A1: curve / group sanity ─────────────────────────────────────────────────

# Published secp256k1 multiples of G (widely cited; any independent source agrees).
KNOWN_MULTIPLES = {
    1: (GX, GY),
    2: (0xC6047F9441ED7D6D3045406E95C07CD85C778E4B8CEF3CA7ABAC09B95C709EE5,
        0x1AE168FEA63DC339A3C58419466CEAEEF7F632653266D0E1236431A950CFE52A),
    3: (0xF9308A019258C31049344F85F89D5229B531C845836F99B08601F113BCE036F9,
        0x388F7B0F632DE8140FE337E62A37F3566500A99934C2231B6CB9FD7584B8E672),
    4: (0xE493DBF1C10D80F3581E4904930B1404CC6C13900EE0758474FA94ABE8C4CD13,
        0x51ED993EA0D455B75642E2098EA51448D967AE33BFBDFE40CFE97BDC47739922),
    5: (0x2F8BDE4D1A07209355B4A7250A5C5128E88B84BDDC619AB7CBA8D569B240EFE4,
        0xD8AC222636E5E3D6D4DBA9DDA6C9C426F788271BAB0D6840DCA87D3AA6AC62D6),
}


def a1_group():
    ok = is_on_curve(G) and mul(N - 1, G) == neg(G) and mul(N, G) is None
    for k, want in KNOWN_MULTIPLES.items():
        got = mul(k, G)
        ok &= got == want
        ok &= is_on_curve(got)
    # k·G and (N−k)·G are negatives; associativity spot-check.
    for _ in range(40):
        k = random.randrange(1, N)
        ok &= mul(N - k, G) == neg(mul(k, G))
    for _ in range(20):
        a, b = random.randrange(1, N), random.randrange(1, N)
        ok &= add(mul(a, G), mul(b, G)) == mul((a + b) % N, G)
    report("A1 curve/group", "PASS" if ok else "FAIL",
           f"G on curve, N·G = O, {len(KNOWN_MULTIPLES)} published multiples, "
           "40 negation + 20 additivity checks")


# ── A2: x-only agreement ─────────────────────────────────────────────────────

def a2_xonly_agreement():
    ok = True
    n_checked = 0
    ks = [1, 2, 3, 7, N - 1, N - 2, 2**255, 2**255 - 1, (N - 1) // 2]
    ks += [random.randrange(1, N) for _ in range(15)]
    xs = [GX] + [mul(random.randrange(1, N), G)[0] for _ in range(8)]
    for x in xs:
        y = recover_y_canonical(x)
        assert y is not None and y % 2 == 0
        for k in ks:
            ok &= affine_mul(k, x, y)[0] == x_only_mul(k, x)
            n_checked += 1
    report("A2 x-only agreement", "PASS" if ok else "FAIL",
           f"{n_checked} (k, xG) pairs: affine_mul(...).x == x_only_mul(...), "
           "incl. k = 1, 2, N−1, N−2, 2^255")


# ── A3: root dependence (the parity gap the AIR must close) ──────────────────

def a3_root_dependence(sample=200):
    ok = True
    differed = 0
    for _ in range(sample):
        k = random.randrange(2, N - 1)
        x, y = mul(random.randrange(1, N), G)
        xr, yr = affine_mul(k, x, y)
        xr2, yr2 = affine_mul(k, x, (P - y) % P)
        ok &= xr2 == xr                    # x-only path cannot see the parity ...
        ok &= yr2 == (P - yr) % P          # ... but the affine one returns a different y
        differed += yr2 != yr
    ok &= differed == sample               # the two roots NEVER agree (y ≠ 0 on secp256k1)
    report("A3 root dependence", "PASS" if ok else "FAIL",
           f"{sample} instances: x invariant under yG → p−yG, y negated in ALL "
           f"{differed} of them ⇒ publishing yR makes the input parity observable")


# ── A4: validation set ───────────────────────────────────────────────────────

def _rejects(fn, *args):
    try:
        fn(*args)
        return None
    except EcsmError as e:
        return str(e)


def a4_validation():
    x, y = mul(7, G)
    cases = [
        ("k = 0", (0, x, y), "ScalarIsZero"),
        ("k = N", (N, x, y), "ScalarOutOfRange"),
        ("k > N", (N + 1, x, y), "ScalarOutOfRange"),
        ("xG = p", (3, P, y), "CoordinateOutOfRange"),
        ("yG = p", (3, x, P), "CoordinateOutOfRange"),
        ("yG = p + small", (3, x, P + 1), "CoordinateOutOfRange"),
        ("off curve", (3, x, (y + 1) % P), "NotOnCurve"),
        ("yG = 0", (3, x, 0), "NotOnCurve"),
    ]
    ok = all(_rejects(affine_mul, *a) == want for _, a, want in cases)
    # Accepted: the whole open scalar range, including the two the x-only path had to
    # treat as degenerate (PR #879: "with y supplied by the chip those scalars are
    # ordinary — cofactor 1 and prime N, so k·P ≠ O for every k ∈ (0, N)").
    for k in [1, 2, N - 2, N - 1]:
        r = affine_mul(k, x, y)
        ok &= is_on_curve(r) and r == mul(k, (x, y))
    ok &= affine_mul(1, x, y) == (x, y)              # k = 1 echoes the input point
    ok &= affine_mul(N - 1, x, y) == (x, (P - y) % P)  # k = N−1 negates it
    report("A4 validation set", "PASS" if ok else "FAIL",
           f"{len(cases)} rejections match the executor's error kinds; "
           "k ∈ {1, 2, N−2, N−1} all accepted and ordinary")


# ── A5: ABI predicates ──────────────────────────────────────────────────────

def a5_abi():
    ok = True
    # addr_limb_ok is exactly "the span fits below the next 2^32 boundary".
    for span in (31, 63):
        boundary = 2**32 - span
        ok &= addr_limb_ok(boundary - 1, span)
        ok &= not addr_limb_ok(boundary, span)
        ok &= not addr_limb_ok(2**32 - 1, span)
        ok &= addr_limb_ok(0, span)
        # the high limb is irrelevant, by design (the AIR reuses it unchanged)
        ok &= addr_limb_ok((7 << 32) + boundary - 1, span)
        ok &= not addr_limb_ok((7 << 32) + boundary, span)
    # The overlap guard is exact interval disjointness, not a distance bound: a scalar
    # placed immediately below the point (addr_k + 32 == addr_xg) IS disjoint.
    base = 0x8000_0000
    ok &= operands_disjoint(base, base - 32)          # k directly below the point
    ok &= not operands_disjoint(base, base - 31)      # one byte of overlap
    ok &= operands_disjoint(base, base + 64)          # k directly above
    ok &= not operands_disjoint(base, base + 63)
    for d in range(-96, 97):
        want = not (set(range(base, base + 64)) & set(range(base + d, base + d + 32)))
        ok &= operands_disjoint(base, base + d) == want
    # NEGATIVE CONTROL: the pre-fix u64 form skips the guard at the wrap address, and
    # that address passes addr_limb_ok(·, 63) so it is reachable.
    wrap = 2**64 - 64
    reachable = addr_limb_ok(wrap, 63)
    buggy_accepts_overlap = operands_disjoint_u64_buggy(wrap, wrap) and not operands_disjoint(wrap, wrap)
    ok &= reachable and buggy_accepts_overlap
    report("A5 ABI predicates", "PASS" if ok else "FAIL",
           "limb bound exact at both spans (31/63), overlap guard == interval "
           "disjointness over 193 offsets; u64-wrap control: addr 2^64−64 passes "
           f"addr_limb_ok (={reachable}) yet the pre-fix guard misses a total overlap")


# ── A6: the ecrecover reconstruction the affine ecall replaces ───────────────

def _solve_y_from_two_x(xg, yg, x1, x2):
    """The x-only recovery the guest used to do: two accelerator queries give
    `x1 = x(k·P)` and `x2 = x((k+1)·P)`; the chord law through the known base point then
    fixes which root of `x1³ + b` is `y(k·P)`.

    Rather than reproduce the guest's exact algebra (which is what PR #879 deletes), take
    both roots and keep the one whose chord with `(xG, yG)` lands on `x2`. The anchor is
    about the VALUE, not the arithmetic route. The disambiguation is total: flipping `y1`
    changes `λ² = ((y1 − yG)/(x1 − xG))²` unless `4·y1·yG ≡ 0`, and secp256k1 has no
    point with `y = 0`."""
    if x1 == xg:
        return None  # k·P = ±P: the chord degenerates, guest handles it separately
    cand = sqrt_mod_p((x1 * x1 % P * x1 + B) % P)
    if cand is None:
        return None
    for y1 in (cand, (P - cand) % P):
        lam = (y1 - yg) * inv(x1 - xg) % P
        if (lam * lam - x1 - xg) % P == x2:
            return y1
    return None


def a6_ecrecover_equivalence(sample=60):
    ok = True
    n = 0
    for _ in range(sample):
        k = random.randrange(2, N - 2)
        xg, yg = mul(random.randrange(1, N), G)
        # what the chip now returns directly:
        x1, y1 = affine_mul(k, xg, yg)
        # what the guest used to compute: two x-only queries + the chord law
        x2 = affine_mul(k + 1, xg, yg)[0]
        rec = _solve_y_from_two_x(xg, yg, x1, x2)
        if rec is None:
            continue  # degenerate chord (k·P = ±P); not what this anchor measures
        ok &= rec == y1
        n += 1
    report("A6 ecrecover equivalence", "PASS" if ok else "FAIL",
           f"{n} instances: y recovered from x(k·P) + x((k+1)·P) + the chord law "
           "equals the y the affine ecall returns ⇒ dropping `solve_y` is "
           "semantics-preserving")


# ── A7: optional third-party cross-check ────────────────────────────────────

def a7_third_party(sample=25):
    try:
        from ecdsa.ellipticcurve import Point  # type: ignore
        from ecdsa.curves import SECP256k1  # type: ignore
    except Exception:
        report("A7 third-party cross-check", "SKIP",
               "python `ecdsa` package not installed (pip install ecdsa)")
        return
    curve = SECP256k1.curve
    gen = SECP256k1.generator
    ok = True
    for _ in range(sample):
        k = random.randrange(1, N)
        j = random.randrange(1, N)
        base = mul(j, G)
        theirs = Point(curve, base[0], base[1]) * k
        ok &= affine_mul(k, base[0], base[1]) == (theirs.x(), theirs.y())
    report("A7 third-party cross-check", "PASS" if ok else "FAIL",
           f"{sample} random k·P against the `ecdsa` package")


# ── codec sanity (cheap, keeps the ABI wire form honest) ─────────────────────

def a8_codec():
    ok = all(from_le32(to_le32(v)) == v
             for v in [0, 1, P - 1, N - 1, 2**256 - 1, GX, GY])
    ok &= to_le32(1)[0] == 1 and to_le32(1)[31] == 0  # little-endian, as the ABI states
    report("A8 LE32 codec", "PASS" if ok else "FAIL", "round-trip + endianness")


def main():
    a0_constants()
    a1_group()
    a2_xonly_agreement()
    a3_root_dependence()
    a4_validation()
    a5_abi()
    a6_ecrecover_equivalence()
    a7_third_party()
    a8_codec()

    passed = [n for n, v, _ in results if v == "PASS"]
    skipped = [n for n, v, _ in results if v == "SKIP"]
    failed = [n for n, v, _ in results if v == "FAIL"]
    print()
    if failed:
        status = "NOT VALIDATED"
    elif skipped:
        status = "PARTIALLY VALIDATED"
    else:
        status = "VALIDATED"
    print(f"ORACLE STATUS: {status}  ({len(passed)} pass, {len(skipped)} skip, "
          f"{len(failed)} fail)")
    # Say what it is NOT anchored on, rather than printing a banner that outlives the
    # evidence (the defect BLAKE3's harness shipped with; see README there).
    if skipped:
        print("  NOT anchored on: " + ", ".join(skipped))
    if failed:
        print("  FAILURES: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
