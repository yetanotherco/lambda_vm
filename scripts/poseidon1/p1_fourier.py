#!/usr/bin/env python3
"""The width-16 instance's partial rounds in the Fourier domain of its circulant MDS, checked against the
textbook permutation in p1_params.py.

A circulant M[i][j] = row[(j - i) mod 16] is diagonalised by the 16-point DFT over Goldilocks (which has 16th
roots of unity; omega = 2^12 has order 16 because 2^96 = -1 mod p): with F[j][k] = omega^(j*k),
F M F^-1 = D = diag(d_0..d_15), d = the DFT of M's first COLUMN. In that domain the dense MDS is 16 products.

A partial round is s <- M S0(s + c), S0 = x^7 on lane 0 only. With s^ = F s:
    s0     = (1/16) * sum_j s^_j                   (row 0 of F^-1 is all 1/16)
    delta  = (s0 + c[0])^7 - (s0 + c[0])            (what the S-box adds to lane 0)
    s^_j  <- d_j * (s^_j + c^_j + delta)            (F e0 is the all-ones vector; c^ = F c)
so the 22 partial rounds cost 22 * (16 + 5) products instead of 22 dense MDS products, plus ONE forward DFT
(the last initial full round's MDS becomes s^ = D * F x) and ONE inverse DFT before the terminal full rounds.

Usage: p1_fourier.py        # derive, check on the KAT inputs and 1,000 random states
"""
import random
import sys

import p1_params as pp

P = pp.P
T = 16
RF, RP = 8, 22
HALF = RF // 2
OMEGA = pow(2, 12, P)
INV16 = pow(16, P - 2, P)


def dft(x, root):
    return [sum(x[k] * pow(root, j * k, P) for k in range(T)) % P for j in range(T)]


def idft(x):
    inv_root = pow(OMEGA, P - 2, P)
    return [v * INV16 % P for v in dft(x, inv_root)]


def eigenvalues(mds):
    col = [mds[i][0] for i in range(T)]
    return dft(col, OMEGA)


def permute_fourier(state, rc, mds):
    d = eigenvalues(mds)
    s = [x % P for x in state]
    for r in range(HALF - 1):
        s = [pow((s[i] + rc[r][i]) % P, 7, P) for i in range(T)]
        s = pp_mat_vec(mds, s)
    # last initial full round: S-box in the time domain, then its MDS as D * F
    x = [pow((s[i] + rc[HALF - 1][i]) % P, 7, P) for i in range(T)]
    sh = [dj * v % P for dj, v in zip(d, dft(x, OMEGA))]
    for k in range(RP):
        c = rc[HALF + k]
        ch = dft(c, OMEGA)
        s0 = sum(sh) * INV16 % P
        a0 = (s0 + c[0]) % P
        delta = (pow(a0, 7, P) - a0) % P
        sh = [dj * ((v + cj + delta) % P) % P for dj, v, cj in zip(d, sh, ch)]
    s = idft(sh)
    for r in range(HALF + RP, RF + RP):
        s = [pow((s[i] + rc[r][i]) % P, 7, P) for i in range(T)]
        s = pp_mat_vec(mds, s)
    return s


def pp_mat_vec(A, x):
    return [sum(a * b for a, b in zip(row, x)) % P for row in A]


def main():
    assert pow(OMEGA, 16, P) == 1 and pow(OMEGA, 8, P) != 1, "omega must have order 16"
    rc = pp.round_constants(T, RF, RP)
    mds = pp.circulant(pp.MDS_ROW[T])
    # the diagonalisation itself: F M = D F on the unit vectors
    d = eigenvalues(mds)
    for k in range(T):
        e = [int(i == k) for i in range(T)]
        assert dft(pp_mat_vec(mds, e), OMEGA) == [dj * v % P for dj, v in zip(d, dft(e, OMEGA))]
    inputs = pp.kat_inputs(T)
    rnd = random.Random(11)
    inputs += [[rnd.randrange(P) for _ in range(T)] for _ in range(1000)]
    ok = all(pp.permute(x, rc, mds, RF, RP) == permute_fourier(x, rc, mds) for x in inputs)
    print(f"Fourier-domain partial rounds == textbook on {len(inputs)} states: {'YES' if ok else 'NO'}")
    print("eigenvalues d_j (DFT of the MDS first column):")
    print("  " + ", ".join(f"0x{v:016x}" for v in d))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
