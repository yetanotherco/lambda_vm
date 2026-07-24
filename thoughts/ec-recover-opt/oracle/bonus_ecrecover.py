"""Bonus checks.

1. End-to-end ecrecover differential: sign with the `ecdsa` PyPI package
   (RFC6979 deterministic, independent lineage), recover with our
   ec_ref.ecrecover, and also cross-check against `ecdsa`'s own
   VerifyingKey.from_public_key_recovery_with_digest (an independent
   ecrecover implementation). 3-way agreement on 40 signatures.

2. Numeric validation of the lambda-linear y-recovery identity used by
   crypto/ethrex-crypto/src/lib.rs::solve_y (the guest-side x-only
   reconstruction):
       t = xc + xa + xp,  dx = xa - xp
       lambda = (xa^3 - xp^3 - t*dx^2) / (2*yp*dx),   lambda^2 == t
       ya = yp + lambda*dx
   On 200 random (P, k): the reconstructed ya must equal the true y(k*P),
   and the identity must NOT hold for the wrong sign (checked via the
   lambda^2 == t consistency check on x((k-1)P) chord), mirroring the
   guard's rationale.
"""

import hashlib
import random

from ecdsa import SECP256k1, SigningKey
from ecdsa.ellipticcurve import Point
from ecdsa.util import sigdecode_string

from ec_ref import (GX, GY, N, P, ecrecover, finv, recover_even_y,
                    scalar_mul, x_only_mul_ints)

rng = random.Random(777)


def part1():
    fails = 0
    from ecdsa import VerifyingKey
    for i in range(40):
        sk = SigningKey.generate(curve=SECP256k1, entropy=lambda n: rng.randbytes(n))
        vk = sk.get_verifying_key()
        msg = rng.randbytes(32 + i % 7)
        digest = hashlib.sha256(msg).digest()
        sig = sk.sign_digest_deterministic(digest)
        r, s = sigdecode_string(sig, N)

        # recid: which parity of R.y recovers the true key (r < N < P here;
        # the r+N branch has ~2^-128 probability and never occurs randomly)
        pub = vk.pubkey.point
        want = (pub.x(), pub.y())
        got = None
        for v in (0, 1):
            cand = ecrecover(digest, v, r, s)
            if cand == want:
                got = (v, cand)
                break
        if got is None:
            fails += 1
            print(f"FAIL roundtrip {i}: neither parity recovers the signer key")
            continue

        # 3-way: python-ecdsa's own recovery must contain the same key
        cands = VerifyingKey.from_public_key_recovery_with_digest(
            sig, digest, curve=SECP256k1, sigdecode=sigdecode_string)
        lib_pts = {(c.pubkey.point.x(), c.pubkey.point.y()) for c in cands}
        if want not in lib_pts:
            fails += 1
            print(f"FAIL lib-recovery {i}: ecdsa lib does not recover signer")
        # and OUR recovered set must be a subset of the lib's candidate set
        ours = set()
        for v in (0, 1):
            c = ecrecover(digest, v, r, s)
            if c is not None:
                ours.add(c)
        if not ours <= lib_pts:
            fails += 1
            print(f"FAIL candidate-set {i}: ours={ours} lib={lib_pts}")
    print(f"BONUS 1 (ecrecover 3-way, 40 sigs): {'PASS' if fails == 0 else 'FAIL'} ({fails} failures)")
    return fails


def part2():
    fails = 0
    for i in range(200):
        # random on-curve P (both parities exercised) and 2 <= k < N-1
        while True:
            xp = rng.randrange(P)
            yp0 = recover_even_y(xp)
            if yp0 is not None:
                break
        yp = yp0 if rng.random() < 0.5 else (P - yp0) % P
        k = rng.randrange(2, N - 1)

        A = scalar_mul(k, (xp, yp))
        C = scalar_mul(k + 1, (xp, yp))
        xa, ya_true = A
        xc = C[0]
        dx = (xa - xp) % P
        if dx == 0:
            continue  # degenerate, guarded in guest code
        t = (xc + xa + xp) % P
        lam = ((pow(xa, 3, P) - pow(xp, 3, P) - t * dx * dx) * finv((2 * yp * dx) % P)) % P
        if (lam * lam) % P != t:
            fails += 1
            print(f"FAIL identity {i}: lambda^2 != t (k={k:x} xp={xp:x})")
            continue
        ya = (yp + lam * dx) % P
        if ya != ya_true:
            fails += 1
            print(f"FAIL identity {i}: recovered ya wrong (k={k:x})")

        # wrong-sign separation: the chord through (xp, yp) and (xa, -ya)
        # lands on x((k-1)P), which must differ from xc (k not near edges)
        Cm = scalar_mul(k - 1, (xp, yp)) if k >= 2 else None
        if Cm is not None and Cm[0] == xc:
            # only possible if 2k = 0 or -1 = 1 mod N: impossible here
            fails += 1
            print(f"FAIL separation {i}: x((k-1)P) == x((k+1)P) at k={k:x}")
    print(f"BONUS 2 (solve_y identity, 200 cases): {'PASS' if fails == 0 else 'FAIL'} ({fails} failures)")
    return fails


if __name__ == "__main__":
    raise SystemExit(1 if (part1() + part2()) else 0)
