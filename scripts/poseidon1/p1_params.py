#!/usr/bin/env python3
"""Poseidon1 over Goldilocks: round numbers, Grain LFSR round constants and a
reference permutation, written from the Poseidon paper (Grassi, Khovratovich,
Rechberger, Roy, Schofnegger, USENIX Security 2021; eprint 2019/458).

  - Round numbers: the paper's Section 5.5 inequalities as encoded by its
    reference `calc_round_numbers.py` (statistical, interpolation, the three
    Groebner bounds), plus the eprint 2023/537 binomial Groebner bound, then the
    paper's security margin (R_F + 2, R_P * 1.075).
  - Round constants: the Grain LFSR of Appendix E (80-bit state, field_type=1,
    sbox=0 for x^alpha, n, t, R_F, R_P, 30 ones; 160 discarded clocks; bits
    taken in pairs, the second kept iff the first is 1; n-bit big-endian
    integers, rejected while >= p), in the order round 0 lane 0 .. round
    R_F+R_P-1 lane t-1.
  - Permutation: R_F/2 full rounds, R_P partial rounds (S-box on lane 0), R_F/2
    full rounds; each round is add-constants, S-box, then state <- M * state.
  - MDS: a circulant M[i][j] = row[(j - i) mod t] (Plonky3's Goldilocks small
    circulants for widths 8/12/16; `goldilocks/src/mds.rs`, MIT/Apache).

Usage:
  p1_params.py rounds  [--width T]          # show the inequalities and (R_F, R_P)
  p1_params.py consts  --width T            # round constants, one hex per line
  p1_params.py kat     --width T            # permutation of [0..T) and three more
  p1_params.py selftest                     # reproduce Plonky3's W8/W12 constants and KATs
"""

import argparse
import re
import sys
from math import ceil, comb, floor, log, log2

P = (1 << 64) - (1 << 32) + 1
N_BITS = 64
ALPHA = 7

# Plonky3 `goldilocks/src/mds.rs` MATRIX_CIRC_MDS_{8,12,16}_SML_ROW (first ROW).
MDS_ROW = {
    8: [7, 1, 3, 8, 8, 3, 4, 9],
    12: [1, 1, 2, 1, 8, 9, 10, 7, 5, 9, 4, 10],
    16: [1, 1, 51, 1, 11, 17, 2, 1, 101, 63, 15, 2, 67, 22, 13, 3],
}


# ----------------------------------------------------------------------------
# Round numbers
# ----------------------------------------------------------------------------


def bounds(p, t, alpha, M, R_F, R_P):
    """Every lower bound on R_F implied by the paper's inequalities at (t, R_P),
    and the 2023/537 Groebner cost; returns (dict of named bounds, gb4 cost)."""
    n = ceil(log2(p))
    la2 = log(2, alpha)  # log_alpha(2)
    b = {}
    # Statistical (differential/linear): 6 full rounds suffice while
    # M <= (floor(log2 p) - (alpha-1)/2) * (t+1), else 10.
    b["statistical"] = 6 if M <= floor(log2(p) - (alpha - 1) / 2.0) * (t + 1) else 10
    # Interpolation: R_F + R_P >= 1 + ceil(log_alpha(2) * min(M, n)) + ceil(log_alpha(t)).
    b["interpolation"] = 1 + ceil(la2 * min(M, n)) + ceil(log(t, alpha)) - R_P
    # Groebner 1: R_F + R_P >= log_alpha(2) * min(M, log2 p).
    b["groebner1"] = la2 * min(M, log2(p)) - R_P
    # Groebner 2: R_F + R_P >= t - 1 + log_alpha(2) * min(M/(t+1), log2(p)/2).
    b["groebner2"] = t - 1 + la2 * min(M / float(t + 1), log2(p) / 2.0) - R_P
    # Groebner 3: (t-1) R_F + R_P >= t - 2 + M / (2 log2 alpha).
    b["groebner3"] = (t - 2 + M / (2.0 * log2(alpha)) - R_P) / float(t - 1)
    # eprint 2023/537: 2 * log2 binom(over, under) >= M (omega = 2, conservative).
    r = floor(t / 3.0)
    over = (R_F - 1) * t + R_P + r + r * (R_F / 2.0) + R_P + alpha
    under = r * (R_F / 2.0) + R_P + alpha
    # R_F is even, so both are integers.
    assert over == int(over) and under == int(under)
    gb4 = ceil(2 * log2(comb(int(over), int(under))))
    return b, gb4


def satisfies(p, t, alpha, M, R_F, R_P):
    b, gb4 = bounds(p, t, alpha, M, R_F, R_P)
    return R_F >= max(ceil(v) for v in b.values()) and gb4 >= M


def round_numbers(p, t, alpha, M=128):
    """Cheapest (R_F, R_P) by t*R_F + R_P after the margin, ties to smaller R_F.
    Returns (R_F, R_P, unmargined R_F, unmargined R_P)."""
    best = None
    for R_P in range(1, 500):
        for R_F in range(4, 100, 2):
            if satisfies(p, t, alpha, M, R_F, R_P):
                rf, rp = R_F + 2, ceil(R_P * 1.075)
                cost = t * rf + rp
                key = (cost, rf)
                if best is None or key < best[0]:
                    best = (key, rf, rp, R_F, R_P)
                break  # larger R_F at the same R_P only costs more
    return best[1:]


# ----------------------------------------------------------------------------
# Grain LFSR (Appendix E)
# ----------------------------------------------------------------------------


class Grain:
    def __init__(self, n, t, R_F, R_P, field_type=1, sbox=0):
        bits = []
        for value, width in ((field_type, 2), (sbox, 4), (n, 12), (t, 12), (R_F, 10), (R_P, 10)):
            bits += [(value >> (width - 1 - i)) & 1 for i in range(width)]
        bits += [1] * 30
        assert len(bits) == 80
        self.s = bits
        for _ in range(160):
            self._clock()

    def _clock(self):
        s = self.s
        b = s[62] ^ s[51] ^ s[38] ^ s[23] ^ s[13] ^ s[0]
        s.pop(0)
        s.append(b)
        return b

    def bit(self):
        while True:
            a = self._clock()
            b = self._clock()
            if a == 1:
                return b

    def field_element(self, n, p):
        while True:
            v = 0
            for _ in range(n):
                v = (v << 1) | self.bit()
            if v < p:
                return v


    def raw_element(self, n, p):
        """n bits, big-endian, reduced mod p (the reference's MDS sampling)."""
        v = 0
        for _ in range(n):
            v = (v << 1) | self.bit()
        return v % p


def round_constants(t, R_F, R_P, p=P, n=N_BITS, grain=None):
    g = grain or Grain(n, t, R_F, R_P)
    return [[g.field_element(n, p) for _ in range(t)] for _ in range(R_F + R_P)]


def cauchy_candidates(t, R_F, R_P, p=P, n=N_BITS):
    """The reference procedure's MDS: continue the same Grain stream after the
    round constants, draw 2t elements (redraw all while not distinct), and set
    M[i][j] = 1/(x_i + y_j) with x = the first t, y = the last t (redraw if any
    x_i + y_j = 0). Yields successive candidates; the caller keeps the first one
    that passes the subspace-trail checks (Algorithms 1-3 of eprint 2020/500)."""
    g = Grain(n, t, R_F, R_P)
    round_constants(t, R_F, R_P, p, n, grain=g)
    while True:
        xs = [g.raw_element(n, p) for _ in range(2 * t)]
        while len(set(xs)) != 2 * t:
            xs = [g.raw_element(n, p) for _ in range(2 * t)]
        x, y = xs[:t], xs[t:]
        if any((x[i] + y[j]) % p == 0 for i in range(t) for j in range(t)):
            continue
        yield [[pow((x[i] + y[j]) % p, p - 2, p) for j in range(t)] for i in range(t)]


# ----------------------------------------------------------------------------
# Permutation
# ----------------------------------------------------------------------------


def circulant(row):
    t = len(row)
    return [[row[(j - i) % t] for j in range(t)] for i in range(t)]


def permute(state, rc, mds, R_F, R_P, alpha=ALPHA, p=P):
    t = len(state)
    s = [x % p for x in state]

    def mix(s):
        return [sum(mds[i][j] * s[j] for j in range(t)) % p for i in range(t)]

    r = 0
    for _ in range(R_F // 2):
        s = [pow((s[i] + rc[r][i]) % p, alpha, p) for i in range(t)]
        s = mix(s)
        r += 1
    for _ in range(R_P):
        s = [(s[i] + rc[r][i]) % p for i in range(t)]
        s[0] = pow(s[0], alpha, p)
        s = mix(s)
        r += 1
    for _ in range(R_F // 2):
        s = [pow((s[i] + rc[r][i]) % p, alpha, p) for i in range(t)]
        s = mix(s)
        r += 1
    return s


# ----------------------------------------------------------------------------
# W16 sponge and 4-ary node (the measurement instance's conventions)
# ----------------------------------------------------------------------------
#
# State lanes: rate 0..12, capacity 12..16, digest 0..4 (RPX's lane rule at
# width 16). Capacity lane 0 carries the padding flag len mod 12, lane 1 the
# domain tag. The 4-ary node permutes the four child digests (16 felts) and
# truncates to lanes 0..4: no capacity, so no domain tag (D-HASH section 5).

RATE16 = 12
DIGEST = 4
DOMAIN_LEAF16 = int.from_bytes(b"P1WL", "little")


def sponge_leaf16(felts, rc, mds, R_F=8, R_P=22):
    s = [0] * 16
    s[RATE16] = len(felts) % RATE16
    s[RATE16 + 1] = DOMAIN_LEAF16
    if not felts:
        return s[:DIGEST]
    for i in range(0, len(felts), RATE16):
        block = felts[i:i + RATE16]
        for lane in range(RATE16):
            s[lane] = block[lane] % P if lane < len(block) else 0
        s = permute(s, rc, mds, R_F, R_P)
    return s[:DIGEST]


def compress4_16(children, rc, mds, R_F=8, R_P=22):
    assert len(children) == 4 and all(len(c) == DIGEST for c in children)
    s = [x for c in children for x in c]
    return permute(s, rc, mds, R_F, R_P)[:DIGEST]


def kat_inputs(t):
    """[0..t), all zeros, all p-1, and a fixed pseudo-random vector."""
    x = 0x9E3779B97F4A7C15
    rnd = []
    for _ in range(t):
        x = (x * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        rnd.append(x % P)
    return [list(range(t)), [0] * t, [P - 1] * t, rnd]


# ----------------------------------------------------------------------------
# Plonky3 cross-check data (goldilocks/src/poseidon1.rs @ 6374a36f)
# ----------------------------------------------------------------------------

PLONKY3_KAT = {
    8: [2431226948502761687, 9427563026145807618, 6827549936272051660, 16907684411084503785,
        10131745626715172913, 17448305483431576765, 9066501914269485014, 12095238468458521303],
    12: [15595088881848875364, 9564850329150784619, 13607005230761744521, 12117102595842533385,
         2814257411756993122, 11640647689983397089, 14363867760831937423, 13323891071259596526,
         11219803511311150468, 9221595262780869902, 5898229059046891887, 18181291031484020550],
}


def plonky3_rc(path, t):
    """Parse GOLDILOCKS_POSEIDON1_RC_{t} out of Plonky3's poseidon1.rs."""
    src = open(path).read()
    start = src.index(f"pub const GOLDILOCKS_POSEIDON1_RC_{t}:")
    body = src[start:src.index("]);", start)]
    vals = [int(h, 16) for h in re.findall(r"0x[0-9a-fA-F]{16}", body)]
    return [vals[i * t:(i + 1) * t] for i in range(len(vals) // t)]


def leaf_kat_inputs():
    """Leaf lengths around the rate: empty, short, one block, one over, the
    64-felt round-0 coset and the 48-felt folded one."""
    out = []
    for n in (0, 1, 11, 12, 13, 48, 64):
        out.append([(i * 0x0123456789ABCDEF + n) % P for i in range(n)])
    return out


def emit_rust(out_dir):
    """Write constants.rs and kat.rs for the W16 host reference."""
    t = 16
    rf, rp, _, _ = round_numbers(P, t, ALPHA, 128)
    assert (rf, rp) == (8, 22)
    rc = round_constants(t, rf, rp)
    mds = circulant(MDS_ROW[t])
    hx = lambda v: f"0x{v:016x}"
    lines = [
        "//! Poseidon1 over Goldilocks at width 16: round constants and the MDS row.",
        "//!",
        "//! GENERATED by `scripts/poseidon1/p1_params.py rust`; do not edit by hand.",
        "//! Round constants: the Poseidon paper's Grain LFSR (Appendix E) with",
        "//! `field_type=1, sbox=0, n=64, t=16, R_F=8, R_P=22`; reproduced by Plonky3's",
        "//! independent generator (`poseidon1/generate_constants.py`, MIT/Apache).",
        "//! MDS: Plonky3's `MATRIX_CIRC_MDS_16_SML_ROW` (`goldilocks/src/mds.rs`,",
        "//! MIT/Apache), checked MDS over Goldilocks by `scripts/poseidon1/mds_check.rs`.",
        "",
        "/// First ROW of the circulant MDS matrix: `M[i][j] = MDS_CIRC_ROW[(j - i) mod 16]`.",
        "pub const MDS_CIRC_ROW: [u64; 16] = [" + ", ".join(map(str, MDS_ROW[t])) + "];",
        "",
        "/// Round constants `[round][lane]`: 4 initial full, 22 partial, 4 terminal full.",
        f"pub const ROUND_CONSTANTS: [[u64; 16]; {rf + rp}] = [",
    ]
    for row in rc:
        lines.append("    [")
        for v in row:
            lines.append(f"        {hx(v)},")
        lines.append("    ],")
    lines.append("];")
    cm = next(cauchy_candidates(t, rf, rp))
    lines += [
        "",
        "/// The paper's MDS alternative: the Grain Cauchy matrix `M[i][j] = 1/(x_i + y_j)`, drawn from",
        "/// the same Grain stream after the round constants (the reference procedure; its first",
        "/// candidate passes the subspace-trail checks, and Plonky3's generator picks the same matrix).",
        "pub const CAUCHY_MDS: [[u64; 16]; 16] = [",
    ]
    for row in cm:
        lines.append("    [" + ", ".join(hx(v) for v in row) + "],")
    lines.append("];")
    open(f"{out_dir}/constants.rs", "w").write("\n".join(lines) + "\n")

    k = [
        "//! Known-answer vectors for the W16 host reference.",
        "//!",
        "//! GENERATED by `scripts/poseidon1/p1_params.py rust` from the Python",
        "//! reference; the permutation vectors are also reproduced by Plonky3's",
        "//! generator's reference permutation and by Plonky3's Rust `Poseidon1`",
        "//! (sparse partial rounds) fed these constants.",
        "",
        "/// `(input, permute(input))`.",
        "pub const PERMUTATION_VECTORS: [([u64; 16], [u64; 16]); 4] = [",
    ]
    for x in kat_inputs(t):
        y = permute(x, rc, mds, rf, rp)
        k.append("    (")
        k.append("        [" + ", ".join(hx(v) for v in x) + "],")
        k.append("        [" + ", ".join(hx(v) for v in y) + "],")
        k.append("    ),")
    k.append("];")
    k.append("")
    k.append("/// `(leaf length, digest)`; felt `i` of a length-`n` leaf is")
    k.append("/// `(i * 0x0123456789abcdef + n) mod p`.")
    leaves = leaf_kat_inputs()
    k.append(f"pub const LEAF_VECTORS: [(usize, [u64; 4]); {len(leaves)}] = [")
    for f in leaves:
        d = sponge_leaf16(f, rc, mds)
        k.append(f"    ({len(f)}, [" + ", ".join(hx(v) for v in d) + "]),")
    k.append("];")
    k.append("")
    k.append("/// `(four children, node)`; child `c` lane `l` is the KAT input `[0..16)`'s")
    k.append("/// permutation output lane `4c + l`, i.e. `PERMUTATION_VECTORS[0].1`.")
    children = permute(list(range(16)), rc, mds, rf, rp)
    node = compress4_16([children[4 * c:4 * c + 4] for c in range(4)], rc, mds)
    k.append("pub const NODE4_VECTOR: [u64; 4] = [" + ", ".join(hx(v) for v in node) + "];")
    # The same vectors under the Cauchy MDS (the instance's alternative), from this reference;
    # Plonky3's generator's permutation reproduces the four permutation vectors.
    cm = next(cauchy_candidates(t, rf, rp))
    k.append("")
    k.append("/// `(input, permute_cauchy(input))` for the inputs of [`PERMUTATION_VECTORS`].")
    k.append("pub const CAUCHY_PERMUTATION_VECTORS: [([u64; 16], [u64; 16]); 4] = [")
    for x in kat_inputs(t):
        y = permute(x, rc, cm, rf, rp)
        k.append("    (")
        k.append("        [" + ", ".join(hx(v) for v in x) + "],")
        k.append("        [" + ", ".join(hx(v) for v in y) + "],")
        k.append("    ),")
    k.append("];")
    k.append("")
    k.append("/// [`LEAF_VECTORS`]' leaves under the Cauchy MDS.")
    k.append(f"pub const CAUCHY_LEAF_VECTORS: [(usize, [u64; 4]); {len(leaves)}] = [")
    for f in leaves:
        d = sponge_leaf16(f, rc, cm)
        k.append(f"    ({len(f)}, [" + ", ".join(hx(v) for v in d) + "]),")
    k.append("];")
    children_c = permute(list(range(16)), rc, cm, rf, rp)
    node_c = compress4_16([children_c[4 * c:4 * c + 4] for c in range(4)], rc, cm)
    k.append("")
    k.append("/// The 4-ary node over `CAUCHY_PERMUTATION_VECTORS[0].1`'s four digests.")
    k.append("pub const CAUCHY_NODE4_VECTOR: [u64; 4] = [" + ", ".join(hx(v) for v in node_c) + "];")
    open(f"{out_dir}/kat.rs", "w").write("\n".join(k) + "\n")


def emit_rust8(out_dir):
    """Write constants.rs for the W8 host permutation (ZisK's grinding hash)."""
    t = 8
    rf, rp, _, _ = round_numbers(P, t, ALPHA, 128)
    assert (rf, rp) == (8, 22)
    rc = round_constants(t, rf, rp)
    lines = [
        "//! Poseidon1 over Goldilocks at width 8: round constants and the MDS row.",
        "//!",
        "//! GENERATED by `scripts/poseidon1/p1_params.py rust8`; do not edit by hand.",
        "//! Round constants: the Poseidon paper's Grain LFSR (Appendix E) with",
        "//! `field_type=1, sbox=0, n=64, t=8, R_F=8, R_P=22`, equal to Plonky3's published",
        "//! width-8 table (`p1_params.py selftest`). MDS: Plonky3's",
        "//! `MATRIX_CIRC_MDS_8_SML_ROW` (`goldilocks/src/mds.rs`, MIT/Apache).",
        "",
        "/// First ROW of the circulant MDS matrix: `M[i][j] = MDS_CIRC_ROW[(j - i) mod 8]`.",
        "pub const MDS_CIRC_ROW: [u64; 8] = [" + ", ".join(map(str, MDS_ROW[t])) + "];",
        "",
        "/// Round constants `[round][lane]`: 4 initial full, 22 partial, 4 terminal full.",
        f"pub const ROUND_CONSTANTS: [[u64; 8]; {rf + rp}] = [",
    ]
    for row in rc:
        lines.append("    [" + ", ".join(f"0x{v:016x}" for v in row) + "],")
    lines.append("];")
    open(f"{out_dir}/constants.rs", "w").write("\n".join(lines) + "\n")


def emit_cuda(kernel_dir, kat_dir):
    """Write kernels/p1w16_constants.cuh and tests/host_kat/p1w16_kat_vectors.h."""
    t = 16
    rf, rp, _, _ = round_numbers(P, t, ALPHA, 128)
    rc = round_constants(t, rf, rp)
    mds = circulant(MDS_ROW[t])
    c = [
        "// Poseidon1 over Goldilocks at width 16: round constants and the MDS row.",
        "// GENERATED by `scripts/poseidon1/p1_params.py cuda`; do not edit by hand.",
        "// The same tables as `crypto/crypto/src/hash/poseidon1_w16/constants.rs`.",
        "#pragma once",
        "#include <cstdint>",
        "namespace p1w16 {",
        f"__device__ __constant__ uint64_t RC[{rf + rp}][16] = {{",
    ]
    for row in rc:
        c.append("    {" + ", ".join(f"0x{v:016x}ull" for v in row) + "},")
    c.append("};")
    c.append("// First ROW of the circulant MDS, `M[i][j] = MDS_ROW[(j - i) mod 16]`. `constexpr`, so a")
    c.append("// fully unrolled product folds every entry into an immediate. Row sum 371 < 2^9.")
    c.append("__device__ constexpr uint32_t MDS_ROW[16] = {" + ", ".join(map(str, MDS_ROW[t])) + "};")
    # The Fourier-domain partial rounds (p1_fourier.py): omega = 2^12 of order 16, the MDS eigenvalues, the
    # last partial round's eigenvalues with the inverse DFT's 1/16 folded in, and each partial round's constants
    # in the Fourier domain.
    import p1_fourier as pf
    d = pf.eigenvalues(mds)
    hx = lambda v: f"0x{v:016x}ull"
    c.append("// Fourier-domain partial rounds (`scripts/poseidon1/p1_fourier.py`): OMEGA_POW[k] = 2^(12k) mod p")
    c.append("// (omega = 2^12 has order 16); FD = the circulant's eigenvalues (DFT of its first column); FD_LAST =")
    c.append("// FD / 16, the inverse DFT's scale folded into the last partial round; FC[k] = DFT(RC[4 + k]).")
    c.append("__device__ constexpr uint64_t OMEGA_POW[16] = {" + ", ".join(hx(pow(pf.OMEGA, k, P)) for k in range(16)) + "};")
    c.append("__device__ constexpr uint64_t FD[16] = {" + ", ".join(hx(v) for v in d) + "};")
    c.append("__device__ constexpr uint64_t FD_LAST[16] = {" + ", ".join(hx(v * pf.INV16 % P) for v in d) + "};")
    c.append(f"__device__ constexpr uint64_t INV16 = {hx(pf.INV16)};")
    c.append(f"__device__ __constant__ uint64_t FC[{rp}][16] = {{")
    for k in range(rp):
        c.append("    {" + ", ".join(hx(v) for v in pf.dft(rc[rf // 2 + k], pf.OMEGA)) + "},")
    c.append("};")
    # The Cauchy alternative: the dense matrix for the full rounds, and the paper's App. B sparse
    # partial rounds derived for it (p1_sparse.py: constant folding + block-diagonal/sparse factors).
    import p1_sparse as psp
    cm = next(cauchy_candidates(t, rf, rp))
    scal, carry, sparse, b0 = psp.derive(rc, cm)
    c.append("// The Grain Cauchy MDS (the instance's alternative, `CAUCHY_MDS` on the host) and its sparse")
    c.append("// partial rounds (`scripts/poseidon1/p1_sparse.py`, the paper's Appendix B): CAUCHY_B0 acts on")
    c.append("// lanes 1..16 once after the last initial full round; partial round k adds CAUCHY_C0[k] to lane 0,")
    c.append("// S-boxes it, then new0 = N00[k]·x0 + W[k]·x[1..], new_i = V[k][i-1]·x0 + x_i; the folded")
    c.append("// carry joins the first terminal round's constants (CAUCHY_RC_T0).")
    c.append("__device__ constexpr uint64_t CAUCHY_M[16][16] = {")
    for row in cm:
        c.append("    {" + ", ".join(hx(v) for v in row) + "},")
    c.append("};")
    c.append("__device__ constexpr uint64_t CAUCHY_B0[15][15] = {")
    for i in range(1, 16):
        c.append("    {" + ", ".join(hx(b0[i][j]) for j in range(1, 16)) + "},")
    c.append("};")
    c.append(f"__device__ __constant__ uint64_t CAUCHY_C0[{rp}] = {{" + ", ".join(hx(v) for v in scal) + "};")
    c.append(f"__device__ __constant__ uint64_t CAUCHY_N00[{rp}] = {{" + ", ".join(hx(sp[0]) for sp in sparse) + "};")
    c.append(f"__device__ __constant__ uint64_t CAUCHY_W[{rp}][15] = {{")
    for sp in sparse:
        c.append("    {" + ", ".join(hx(v) for v in sp[1]) + "},")
    c.append("};")
    c.append(f"__device__ __constant__ uint64_t CAUCHY_V[{rp}][15] = {{")
    for sp in sparse:
        c.append("    {" + ", ".join(hx(v) for v in sp[2]) + "},")
    c.append("};")
    t0 = [(a + b) % P for a, b in zip(rc[rf // 2 + rp], carry)]
    c.append("__device__ constexpr uint64_t CAUCHY_RC_T0[16] = {" + ", ".join(hx(v) for v in t0) + "};")
    c.append("}  // namespace p1w16")
    open(f"{kernel_dir}/p1w16_constants.cuh", "w").write("\n".join(c) + "\n")

    hx = lambda v: f"0x{v:016x}ull"
    k = [
        "// Known answers for `kernels/p1w16.cu`, from the Python reference",
        "// (`scripts/poseidon1/p1_params.py cuda`); the permutation vectors are the ones four",
        "// implementations agree on (`crypto::hash::poseidon1_w16::kat`).",
        "#pragma once",
        "#include <cstdint>",
        "static const uint64_t P1_PERM_IN[4][16] = {",
    ]
    ins = kat_inputs(t)
    for x in ins:
        k.append("    {" + ", ".join(hx(v) for v in x) + "},")
    k.append("};")
    k.append("static const uint64_t P1_PERM_OUT[4][16] = {")
    for x in ins:
        k.append("    {" + ", ".join(hx(v) for v in permute(x, rc, mds, rf, rp)) + "},")
    k.append("};")
    leaves = leaf_kat_inputs()
    k.append("// Leaf of n felts, felt i = (i * 0x0123456789abcdef + n) mod p.")
    k.append(f"static const uint64_t P1_LEAF_N[{len(leaves)}] = {{" + ", ".join(str(len(f)) for f in leaves) + "};")
    k.append(f"static const uint64_t P1_LEAF_DIGEST[{len(leaves)}][4] = {{")
    for f in leaves:
        k.append("    {" + ", ".join(hx(v) for v in sponge_leaf16(f, rc, mds)) + "},")
    k.append("};")
    children = permute(list(range(16)), rc, mds, rf, rp)
    node = compress4_16([children[4 * i:4 * i + 4] for i in range(4)], rc, mds)
    k.append("// The 4-ary node over the four digests P1_PERM_OUT[0][4c..4c+4].")
    k.append("static const uint64_t P1_NODE4[4] = {" + ", ".join(hx(v) for v in node) + "};")
    inner = [1, 2, 3, 4]
    k.append("// Grind head: lane 0 of sponge_leaf([1, 2, 3, 4, nonce]) for nonce 0..8.")
    k.append("static const uint64_t P1_GRIND_INNER[4] = {1, 2, 3, 4};")
    k.append("static const uint64_t P1_GRIND_HEAD[8] = {" +
             ", ".join(hx(sponge_leaf16(inner + [n], rc, mds)[0]) for n in range(8)) + "};")
    cm = next(cauchy_candidates(t, rf, rp))
    k.append("// The same vectors under the Grain Cauchy MDS (`CAUCHY_*` on the host).")
    k.append("static const uint64_t P1C_PERM_OUT[4][16] = {")
    for x in ins:
        k.append("    {" + ", ".join(hx(v) for v in permute(x, rc, cm, rf, rp)) + "},")
    k.append("};")
    k.append(f"static const uint64_t P1C_LEAF_DIGEST[{len(leaves)}][4] = {{")
    for f in leaves:
        k.append("    {" + ", ".join(hx(v) for v in sponge_leaf16(f, rc, cm)) + "},")
    k.append("};")
    ch_c = permute(list(range(16)), rc, cm, rf, rp)
    node_c = compress4_16([ch_c[4 * i:4 * i + 4] for i in range(4)], rc, cm)
    k.append("// The 4-ary node over the four digests P1C_PERM_OUT[0][4c..4c+4].")
    k.append("static const uint64_t P1C_NODE4[4] = {" + ", ".join(hx(v) for v in node_c) + "};")
    k.append("static const uint64_t P1C_GRIND_HEAD[8] = {" +
             ", ".join(hx(sponge_leaf16(inner + [n], rc, cm)[0]) for n in range(8)) + "};")
    open(f"{kat_dir}/p1w16_kat_vectors.h", "w").write("\n".join(k) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["rounds", "consts", "kat", "cauchy", "rust", "rust8", "cuda", "selftest"])
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--security", type=int, default=128)
    ap.add_argument("--out", default=".")
    ap.add_argument("--mds", choices=["circulant", "cauchy"], default="circulant")
    ap.add_argument("--kat-out", default=".")
    ap.add_argument("--plonky3", default="/Users/maurofab/workspace/Plonky3/goldilocks/src/poseidon1.rs")
    a = ap.parse_args()
    t = a.width

    if a.cmd == "rounds":
        rf, rp, rf0, rp0 = round_numbers(P, t, ALPHA, a.security)
        print(f"t={t} alpha={ALPHA} M={a.security} log2(p)={log2(P):.6f}")
        print(f"unmargined minimum: R_F={rf0} R_P={rp0}; with margin: R_F={rf} R_P={rp} "
              f"(S-boxes {t * rf + rp})")
        for label, R_F, R_P in (("unmargined", rf0, rp0), ("unmargined, R_P-1", rf0, rp0 - 1),
                                ("margined", rf, rp)):
            b, gb4 = bounds(P, t, ALPHA, a.security, R_F, R_P)
            ok = satisfies(P, t, ALPHA, a.security, R_F, R_P)
            parts = ", ".join(f"{k} R_F>={v:.3f}" for k, v in b.items())
            print(f"  {label:18s} R_F={R_F} R_P={R_P}: {parts}; gb4 cost {gb4} >= {a.security}: "
                  f"{gb4 >= a.security}; satisfied: {ok}")
        return
    if a.cmd == "rust":
        emit_rust(a.out)
        return
    if a.cmd == "rust8":
        emit_rust8(a.out)
        return
    if a.cmd == "cuda":
        emit_cuda(a.out, a.kat_out)
        return
    if a.cmd == "cauchy":
        # First Cauchy candidate only; the subspace-trail checks run separately.
        rf, rp, _, _ = round_numbers(P, t, ALPHA, a.security)
        m = next(cauchy_candidates(t, rf, rp))
        for row in m:
            print(" ".join(f"0x{v:016x}" for v in row))
        return
    if a.cmd in ("consts", "kat"):
        rf, rp, _, _ = round_numbers(P, t, ALPHA, a.security)
        rc = round_constants(t, rf, rp)
        if a.cmd == "consts":
            for row in rc:
                print(" ".join(f"0x{v:016x}" for v in row))
            return
        # --mds cauchy: the paper's Grain Cauchy matrix (the first candidate; the
        # subspace-trail checks run separately, p3_subspace_checks.py covers the
        # circulant rows only).
        mds = next(cauchy_candidates(t, rf, rp)) if a.mds == "cauchy" else circulant(MDS_ROW[t])
        for x in kat_inputs(t):
            y = permute(x, rc, mds, rf, rp)
            print("in  " + " ".join(str(v) for v in x))
            print("out " + " ".join(str(v) for v in y))
        return
    # selftest
    ok = True
    for t in (8, 12):
        rf, rp, _, _ = round_numbers(P, t, ALPHA, 128)
        rc = round_constants(t, rf, rp)
        got = permute(list(range(t)), rc, circulant(MDS_ROW[t]), rf, rp)
        good = got == PLONKY3_KAT[t]
        rc_good = rc == plonky3_rc(a.plonky3, t)
        ok &= good and rc_good and (rf, rp) == (8, 22)
        print(f"W{t}: R_F={rf} R_P={rp}; {len(rc) * t} round constants vs Plonky3: "
              f"{'MATCH' if rc_good else 'MISMATCH'}; KAT [0..{t}) vs Plonky3: {'MATCH' if good else 'MISMATCH'}")
    print("SELFTEST", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
