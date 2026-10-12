#!/usr/bin/env python3
"""The Poseidon paper's optimized partial rounds (eprint 2019/458, Appendix B) for the width-16 Goldilocks
instance, derived and checked against the textbook permutation in p1_params.py.

Two exact rewrites of the 22 partial rounds (S-box on lane 0 only):

1. Constant folding. With c'_k = rc[4+k] + d_k, only c'_k[0] is added before the lane-0 S-box; the rest,
   pushed through the linear layer, is carried: d_{k+1} = M (c'_k with lane 0 zeroed), d_0 = 0. After the
   last partial round the carry d_22 joins the first terminal full round's constants.
2. Sparse matrices. Every operation T_k (add a lane-0 constant, S-box lane 0) commutes with a block-diagonal
   B = diag(1, B^) , so walking the partial rounds from the last one back, N = Sp_k B_k with
   B_k = diag(1, N^), Sp_k = N B_k^-1 = [[n00, n_row N^^-1], [n_col, I]], and N <- B_k M for the round before.
   The leftover B_0 is applied once, right after the last initial full round.

Per partial round the linear layer is then new0 = n00 x0 + w.x[1..], new_i = v_i x0 + x_i (31 products),
instead of the dense MDS.

Usage: p1_sparse.py            # derive, check on the KAT inputs and 1,000 random states, print the op counts
"""
import random
import sys

import p1_params as pp

P = pp.P
T = 16
RF, RP = 8, 22
HALF = RF // 2


def mat_mul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(len(B))) % P for j in range(len(B[0]))] for i in range(len(A))]


def mat_vec(A, x):
    return [sum(a * b for a, b in zip(row, x)) % P for row in A]


def mat_inv(A):
    n = len(A)
    M = [row[:] + [int(i == j) for j in range(n)] for i, row in enumerate(A)]
    for c in range(n):
        piv = next(r for r in range(c, n) if M[r][c] % P)
        M[c], M[piv] = M[piv], M[c]
        inv = pow(M[c][c], P - 2, P)
        M[c] = [x * inv % P for x in M[c]]
        for r in range(n):
            if r != c and M[r][c]:
                f = M[r][c]
                M[r] = [(x - f * y) % P for x, y in zip(M[r], M[c])]
    return [row[n:] for row in M]


def derive(rc, mds):
    # 1. constant folding
    scalars, d = [], [0] * T
    for k in range(RP):
        c = [(a + b) % P for a, b in zip(rc[HALF + k], d)]
        scalars.append(c[0])
        rest = [0] + c[1:]
        d = mat_vec(mds, rest)
    carry = d
    # 2. sparse factors, last partial round first
    N = [row[:] for row in mds]
    sparse = [None] * RP
    B = None
    for k in range(RP - 1, -1, -1):
        hat = [row[1:] for row in N[1:]]
        hat_inv = mat_inv(hat)
        n00 = N[0][0]
        w = mat_vec([list(col) for col in zip(*hat_inv)], N[0][1:])  # n_row . hat^-1
        v = [N[i][0] for i in range(1, T)]
        sparse[k] = (n00, w, v)
        B = [[1] + [0] * (T - 1)] + [[0] + hat[i] for i in range(T - 1)]
        N = mat_mul(B, mds)
    # the last factor computed is B_0, the one left to the right of the first partial round
    return scalars, carry, sparse, B


def permute_sparse(state, rc, mds, scalars, carry, sparse, B0):
    s = [x % P for x in state]
    for r in range(HALF):
        s = [pow((s[i] + rc[r][i]) % P, 7, P) for i in range(T)]
        s = mat_vec(mds, s)
    s = mat_vec(B0, s)
    for k in range(RP):
        n00, w, v = sparse[k]
        x0 = pow((s[0] + scalars[k]) % P, 7, P)
        new0 = (n00 * x0 + sum(a * b for a, b in zip(w, s[1:]))) % P
        s = [new0] + [(v[i] * x0 + s[i + 1]) % P for i in range(T - 1)]
    rc_first_terminal = [(a + b) % P for a, b in zip(rc[HALF + RP], carry)]
    for j, r in enumerate(range(HALF + RP, RF + RP)):
        c = rc_first_terminal if j == 0 else rc[r]
        s = [pow((s[i] + c[i]) % P, 7, P) for i in range(T)]
        s = mat_vec(mds, s)
    return s


def main():
    rc = pp.round_constants(T, RF, RP)
    mds = pp.circulant(pp.MDS_ROW[T])
    scalars, carry, sparse, B0 = derive(rc, mds)
    ok = True
    inputs = pp.kat_inputs(T)
    rnd = random.Random(7)
    inputs += [[rnd.randrange(P) for _ in range(T)] for _ in range(1000)]
    for x in inputs:
        a = pp.permute(x, rc, mds, RF, RP)
        b = permute_sparse(x, rc, mds, scalars, carry, sparse, B0)
        ok &= a == b
    print(f"sparse partial rounds == textbook on {len(inputs)} states: {'YES' if ok else 'NO'}")
    dense_macs = RP * T * T
    sparse_muls = (T - 1) * (T - 1) + RP * (2 * (T - 1) + 1)
    print(f"partial-round linear layers: textbook {dense_macs} small-constant MACs (x2 halves) vs sparse "
          f"{sparse_muls} full products (B_0 {(T - 1) ** 2} once + {2 * (T - 1) + 1} per round)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
