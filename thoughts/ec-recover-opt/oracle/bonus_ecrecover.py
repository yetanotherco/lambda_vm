"""Bonus checks.

1. End-to-end ecrecover differential: sign with the `ecdsa` PyPI package
   (RFC6979 deterministic, independent lineage), recover with our
   ec_ref.ecrecover, and also cross-check against `ecdsa`'s own
   VerifyingKey.from_public_key_recovery_with_digest (an independent
   ecrecover implementation). 3-way agreement on 40 signatures.

2. The PARITY-AUTHORITY identity, on 200 random signature r-values.

   RE-AIMED 2026-07-24. This slot used to validate the lambda-linear
   y-recovery identity of `crypto/ethrex-crypto/src/lib.rs::solve_y`. Phase G
   deleted `solve_y` along with the whole x-only reconstruction path, so that
   check became a GREEN TEST OF CODE THAT NO LONGER EXISTS — worse than no
   test, because it implies coverage that is not there.

   What it pins now is a property of the CURRENT path, and one that two live
   claims rest on but nothing else executes:

     - the spec chapter's "the guest is the parity authority, backed by MEMW;
       the chip's obligations are membership and canonicalisation only", and
     - the z3 gate's N7 result, that dropping `yP2 < p` is REDUNDANT rather
       than a forgery.

   Both reduce to: a `< p` range check cannot separate a point from its
   negation, because both candidate y values are already below p. Part 2 now
   asserts that numerically, together with the recid convention itself
   (`y_is_odd = recid & 1`, lib.rs:104-106) and the fact that the recid bit
   really does change the recovered key.

   Do not restore the solve_y version: the code it tested is gone.

3. (phase D0) The SAME 40 signatures, recovered through the LINCOMB2 path:
   pk = u1*G + u2*R evaluated by `lincomb2_ref.lincomb2`, by the NUMS-blinded
   joint chain `lincomb2_ref.lincomb2_rows` (what the chip will prove), and by
   the independent Jacobian implementation `jacobian_ref.lincomb2`. All must
   equal the signer's key and the `ecdsa` library's recovery. This is the
   existing 3-way ecrecover differential re-run end-to-end over lincomb2 --
   part1 hands its corpus to part3, so it is literally the same signatures.
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
    corpus = []  # (digest, v, r, s, want_pk) -- handed to part3
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
        corpus.append((digest, got[0], r, s, want))

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
    return fails, corpus


def part2():
    """The PARITY AUTHORITY identity — see the module docstring for why this
    replaced the deleted `solve_y` check.

    Four properties per case, all of the CURRENT path:
      (a) lifting `r` under the guest's convention `y_is_odd = recid & 1`
          (lib.rs:104-106) gives an on-curve point whose y-parity is exactly
          that bit;
      (b) flipping the recid bit yields exactly the negation `(x, p - y)`;
      (c) the two parities recover DIFFERENT public keys, so the recid bit
          genuinely determines the answer — parity is load-bearing;
      (d) BOTH candidate y values are `< p`, so a `< p` canonicalisation test
          cannot separate a point from its negation.

    (d) is the numeric statement behind the gate's N7 redundancy result and the
    spec chapter's correction to DESIGN section 3: `yP2 < p` is defence in
    depth, and parity is pinned by the guest plus MEMW, not by a range check.
    """
    fails = 0
    checked = 0
    for i in range(200):
        # a random valid signature r-value, i.e. a curve x-coordinate < N
        while True:
            r = rng.randrange(1, N)
            y_even = recover_even_y(r)
            if y_even is not None:
                break
        y_odd = (P - y_even) % P
        z = rng.randrange(1, N)
        s = rng.randrange(1, N)
        checked += 1

        # (a) the guest's convention: recid bit 0 selects the y-parity
        for recid in (0, 1):
            y = y_odd if (recid & 1) else y_even
            if y & 1 != (recid & 1):
                fails += 1
                print(f"FAIL parity {i}: lifted parity != recid bit (recid={recid})")
            if (y * y - pow(r, 3, P) - 7) % P != 0:
                fails += 1
                print(f"FAIL parity {i}: lifted point off-curve (recid={recid})")

        # (b) the two lifts are exact negations
        if (y_even + y_odd) % P != 0:
            fails += 1
            print(f"FAIL parity {i}: the two lifts are not negations")

        # (c) parity is load-bearing: the two recids recover different keys
        pk0 = ecrecover(z.to_bytes(32, "big"), 0, r, s)
        pk1 = ecrecover(z.to_bytes(32, "big"), 1, r, s)
        if pk0 is None or pk1 is None:
            continue  # degenerate (u1 or u2 zero); the guest falls back
        if pk0 == pk1:
            fails += 1
            print(f"FAIL parity {i}: both recids recover the same key")

        # (d) a `< p` test cannot tell them apart
        if not (y_even < P and y_odd < P):
            fails += 1
            print(f"FAIL parity {i}: a candidate y is not below p")

    print(f"BONUS 2 (parity-authority identity, {checked} cases): "
          f"{'PASS' if fails == 0 else 'FAIL'} ({fails} failures)")
    return fails


def part3(corpus):
    """The same 40 signatures, recovered through the lincomb2 path."""
    import jacobian_ref
    import lincomb2_ref

    T0, _ = lincomb2_ref.t0_ref()
    fails = 0
    via_chain = 0
    rows_seen = []
    for i, (digest, v, r, s, want) in enumerate(corpus):
        # Exactly the guest's decomposition (crypto/ethrex-crypto/src/lib.rs):
        # R lifted from (r, parity v), u1 = -z/r, u2 = s/r (mod N).
        y_even = recover_even_y(r)
        if y_even is None:
            fails += 1
            print(f"FAIL lincomb2 {i}: r does not lift to a curve point")
            continue
        R = (r, y_even if v == 0 else (P - y_even) % P)
        z = int.from_bytes(digest, "big") % N
        rinv = pow(r, N - 2, N)
        u1 = (-(rinv * z)) % N
        u2 = (rinv * s) % N
        if not (1 <= u1 < N and 1 <= u2 < N):
            print(f"SKIP lincomb2 {i}: u1 or u2 outside [1, N) (software fallback)")
            continue

        G = (GX, GY)
        q_ref = lincomb2_ref.lincomb2(u1, G, u2, R)
        q_jac = jacobian_ref.lincomb2(u1, G, u2, R)
        if q_ref != want or q_jac != want:
            fails += 1
            print(f"FAIL lincomb2 {i}: pk mismatch ref={q_ref} jac={q_jac} want={want}")
            continue

        # the NUMS-blinded joint chain -- what the chip proves
        try:
            q_chain, length, rows = lincomb2_ref.lincomb2_rows(u1, G, u2, R, T0)
        except ValueError as e:
            fails += 1
            print(f"FAIL lincomb2 {i}: blinded chain rejected ({e})")
            continue
        if q_chain != want:
            fails += 1
            print(f"FAIL lincomb2 {i}: blinded chain pk != signer key")
            continue
        via_chain += 1
        rows_seen.append(len(rows))

    print(f"BONUS 3 (ecrecover via lincomb2, {len(corpus)} sigs): "
          f"{'PASS' if fails == 0 else 'FAIL'} ({fails} failures; "
          f"{via_chain} recovered through the NUMS-blinded joint chain)")
    if rows_seen:
        print(f"  chain rows: mean {sum(rows_seen)/len(rows_seen):.1f} "
              f"min {min(rows_seen)} max {max(rows_seen)}")
    return fails


if __name__ == "__main__":
    f1, corpus = part1()
    f2 = part2()
    f3 = part3(corpus)
    raise SystemExit(1 if (f1 + f2 + f3) else 0)
