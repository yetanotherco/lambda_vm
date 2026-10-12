#!/usr/bin/env python3
"""Run Plonky3's generator's subspace-trail checks (Algorithms 1-3 of eprint
2020/500) on the circulant MDS rows in p1_params.py, then print the W16
permutation vectors under Plonky3's generator's own reference permutation.

Usage: p3_subspace_checks.py <patched Plonky3 generate_constants.py dir> <W16 constants json>
(crosscheck.sh prepares both.)"""
import json
import sys
import time

sys.path.insert(0, sys.argv[1])
import gen16 as g  # noqa: E402

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import p1_params as mine  # noqa: E402

p = g.FIELDS["goldilocks"]["prime"]
for t in (8, 12, 16):
    M = mine.circulant(mine.MDS_ROW[t])
    t0 = time.time()
    r1, r2, r3 = g.algorithm_1(M, t, p), g.algorithm_2(M, t, p), g.algorithm_3(M, t, p)
    print(f"# W{t} circulant: alg1 {r1} alg2 {r2} alg3 {r3} ({time.time() - t0:.1f}s)", flush=True)
j = json.load(open(sys.argv[2]))
rc = [[int(x, 16) for x in row] for row in j["round_constants"]]
M = mine.circulant(mine.MDS_ROW[16])
for x in mine.kat_inputs(16):
    print("in  " + " ".join(map(str, x)))
    print("out " + " ".join(map(str, g.poseidon1_permutation(x, M, rc, 7, p, 16, 8, 22))))
