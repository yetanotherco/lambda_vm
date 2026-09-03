"""Validate the independent reference: constants + full permutation."""
import hashlib
from keccak_ref import RC, RHO, sha3_256

# Repo constants (KECCAK_RC / KECCAK_RHO in
# executor/src/vm/instruction/execution.rs), pasted
# here ONLY to cross-check my spec-generated values. Correctness is anchored to
# FIPS-202 (my generators) + hashlib, not to these.
REPO_RC = [
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000,
    0x000000000000808B, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
    0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
]
# KECCAK_RHO[x][y] in the repo
REPO_RHO = [
    [0, 36, 3, 41, 18],
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14],
]

print("=== constant cross-checks (spec-generated vs repo) ===")
print("RC  match:", RC == REPO_RC)
print("RHO match:", RHO == REPO_RHO)
assert RC == REPO_RC, (RC, REPO_RC)
assert RHO == REPO_RHO, (RHO, REPO_RHO)

print("\n=== SHA3-256 vs hashlib (external NIST impl) ===")
tests = [b"", b"abc", b"The quick brown fox jumps over the lazy dog", bytes(range(200))]
allok = True
for t in tests:
    mine = sha3_256(t).hex()
    ref = hashlib.sha3_256(t).hexdigest()
    ok = mine == ref
    allok &= ok
    print(f"  len={len(t):3d}  match={ok}  {mine}")
assert allok
print("\nALL REFERENCE VALIDATIONS PASSED")
