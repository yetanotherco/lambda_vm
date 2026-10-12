#!/usr/bin/env python3
"""The width-16 and width-8 instances' partial rounds in the Fourier domain of their circulant MDS, checked
against the textbook permutation in p1_params.py.

A circulant M[i][j] = row[(j - i) mod T] is diagonalised by the T-point DFT over Goldilocks, which has T-th roots
of unity for T | 2^32 (omega_T = 2^(192/T) has order T because 2^96 = -1 mod p: 2^12 at T = 16, 2^24 at T = 8):
with F[j][k] = omega^(j*k), F M F^-1 = D = diag(d_0..d_{T-1}), d = the DFT of M's first COLUMN. In that domain the
dense MDS is T products.

A partial round is s <- M S0(s + c), S0 = x^7 on lane 0 only. With s^ = F s:
    s0     = (1/T) * sum_j s^_j                    (row 0 of F^-1 is all 1/T)
    delta  = (s0 + c[0])^7 - (s0 + c[0])            (what the S-box adds to lane 0)
    s^_j  <- d_j * (s^_j + c^_j + delta)            (F e0 is the all-ones vector; c^ = F c)
so the 22 partial rounds cost 22 * (T + 5) products instead of 22 dense MDS products, plus ONE forward DFT
(the last initial full round's MDS becomes s^ = D * F x) and ONE inverse DFT before the terminal full rounds.

Usage: p1_fourier.py        # derive, check on the KAT inputs and 1,000 random states, at T = 16 and T = 8
"""
import random
import sys

import p1_params as pp

P = pp.P
RF, RP = 8, 22
HALF = RF // 2


def omega(t):
    """A primitive t-th root of unity: 2^(192/t)."""
    assert 192 % t == 0
    return pow(2, 192 // t, P)


# The width-16 names `emit_cuda` reads.
T = 16
OMEGA = omega(16)
INV16 = pow(16, P - 2, P)


def dft(x, root):
    t = len(x)
    return [sum(x[k] * pow(root, j * k, P) for k in range(t)) % P for j in range(t)]


def idft(x):
    t = len(x)
    inv_root = pow(omega(t), P - 2, P)
    inv_t = pow(t, P - 2, P)
    return [v * inv_t % P for v in dft(x, inv_root)]


def eigenvalues(mds):
    t = len(mds)
    col = [mds[i][0] for i in range(t)]
    return dft(col, omega(t))


def permute_fourier(state, rc, mds):
    t = len(state)
    w = omega(t)
    inv_t = pow(t, P - 2, P)
    d = eigenvalues(mds)
    s = [x % P for x in state]
    for r in range(HALF - 1):
        s = [pow((s[i] + rc[r][i]) % P, 7, P) for i in range(t)]
        s = pp_mat_vec(mds, s)
    # last initial full round: S-box in the time domain, then its MDS as D * F
    x = [pow((s[i] + rc[HALF - 1][i]) % P, 7, P) for i in range(t)]
    sh = [dj * v % P for dj, v in zip(d, dft(x, w))]
    for k in range(RP):
        c = rc[HALF + k]
        ch = dft(c, w)
        s0 = sum(sh) * inv_t % P
        a0 = (s0 + c[0]) % P
        delta = (pow(a0, 7, P) - a0) % P
        sh = [dj * ((v + cj + delta) % P) % P for dj, v, cj in zip(d, sh, ch)]
    s = idft(sh)
    for r in range(HALF + RP, RF + RP):
        s = [pow((s[i] + rc[r][i]) % P, 7, P) for i in range(t)]
        s = pp_mat_vec(mds, s)
    return s


def pp_mat_vec(A, x):
    return [sum(a * b for a, b in zip(row, x)) % P for row in A]


def check(t):
    w = omega(t)
    assert pow(w, t, P) == 1 and pow(w, t // 2, P) != 1, f"omega must have order {t}"
    rc = pp.round_constants(t, RF, RP)
    mds = pp.circulant(pp.MDS_ROW[t])
    # the diagonalisation itself: F M = D F on the unit vectors
    d = eigenvalues(mds)
    for k in range(t):
        e = [int(i == k) for i in range(t)]
        assert dft(pp_mat_vec(mds, e), w) == [dj * v % P for dj, v in zip(d, dft(e, w))]
    inputs = pp.kat_inputs(t)
    rnd = random.Random(11)
    inputs += [[rnd.randrange(P) for _ in range(t)] for _ in range(1000)]
    ok = all(pp.permute(x, rc, mds, RF, RP) == permute_fourier(x, rc, mds) for x in inputs)
    print(f"W{t}: Fourier-domain partial rounds == textbook on {len(inputs)} states: {'YES' if ok else 'NO'}")
    print(f"W{t}: eigenvalues d_j (DFT of the MDS first column):")
    print("  " + ", ".join(f"0x{v:016x}" for v in d))
    return ok


def main():
    ok = check(16) and check(8)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
