"""Validate the independent reference: constants + full hash.

The reference generates K and H0 from their FIPS 180-4 definitions (fractional
parts of cube/square roots of the first primes). The repo tables below are
pasted here ONLY to cross-check those generated values; correctness is anchored
to the spec definitions plus hashlib, never to the repo.
"""
import hashlib
import random
from sha256_ref import K, H0, compress, schedule, sha256

# executor/src/sha256.rs:3-12 (K) and :13-15 (IV), as of the branch this gate
# was authored against. Cross-check only.
REPO_K = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]
REPO_IV = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
]

print("=== constant cross-checks (spec-generated vs repo) ===")
print("K  match:", K == REPO_K)
print("H0 match:", H0 == REPO_IV)
assert K == REPO_K, (K[:4], REPO_K[:4])
assert H0 == REPO_IV, (H0, REPO_IV)

print("\n=== SHA-256 vs hashlib (external impl) ===")
# NIST vectors plus the padding edge cases: one block exactly, one byte short of
# the length field, and the two-block padding case.
tests = [b"", b"abc", b"a" * 55, b"a" * 56, b"a" * 63, b"a" * 64, b"a" * 65,
         b"The quick brown fox jumps over the lazy dog", bytes(range(256))]
allok = True
for t in tests:
    mine, ref = sha256(t).hex(), hashlib.sha256(t).hexdigest()
    ok = mine == ref
    allok &= ok
    print(f"  len={len(t):3d}  match={ok}  {mine[:32]}...")
assert allok

print("\n=== random inputs vs hashlib (200) ===")
rng = random.Random(0xC0FFEE)
for _ in range(200):
    t = bytes(rng.randrange(256) for _ in range(rng.randrange(0, 300)))
    assert sha256(t) == hashlib.sha256(t).digest(), t.hex()
print("  200/200 match")

print("\n=== compression is the ECALL's function, not the full hash ===")
# The accelerator's ECALL takes (state, 64-byte chunk) and returns the updated
# state, feed-forward included. Hashing one block through it must reproduce
# hashlib for a 55-byte message (single padded block).
msg = b"a" * 55
padded = msg + b"\x80" + b"\x00" * (55 - len(msg)) + (len(msg) * 8).to_bytes(8, "big")
assert len(padded) == 64
out = compress(H0, padded)
assert b"".join(x.to_bytes(4, "big") for x in out) == hashlib.sha256(msg).digest()
print("  single-block compress matches hashlib")

print("\n=== schedule is 64 words and the first 16 are the block ===")
blk = bytes(range(64))
w = schedule(blk)
assert len(w) == 64
assert all(w[i] == int.from_bytes(blk[4 * i:4 * i + 4], "big") for i in range(16))
print("  schedule shape and prefix OK")

print("\nALL REFERENCE VALIDATIONS PASSED")
