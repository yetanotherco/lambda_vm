#!/usr/bin/env python3
"""Derives the "canonicalisation witness" row of `rpx_kat_vectors.h`.

WHY. `rpx::permute` (kernels/rpx.cu) ends in a loop that canonicalises the
state, which is what makes device digests byte-comparable to the host's. A
known-answer check cannot see that loop unless some output lane is a raw twin
(`value + p`, in `[p, 2^64)`) before it — a 2^-32 event per lane on random
inputs. This script builds an input for which it is certain.

HOW. The permutation's last operation is `out_i = add(m_i, ARK1[6][i])`, where
`m_i` is the M-round MDS output. With `m_i` canonical and
`m_i + ARK1[6][i] < 2^64`, the device `add` returns `m_i + ARK1[6][i]` as is; if
that sum lies in `[p, 2^64)` it is the raw twin of `sum − p`. So choose the
canonical MDS output `u` with `u_0 = p − ARK1[6][0] + 1` (raw `out_0 = p + 1`,
field value 1), fill the other eleven lanes at random, invert the MDS to get the
M-round input, and invert rounds 5..0 — `x^{1/7}` in `GF(p³)` for the E rounds,
`x^7` / `MDS⁻¹` / `x^{1/7}` / `MDS⁻¹` for the FB rounds — to get the
permutation input. `m_0` cannot itself be a twin (`u_0 + p > 2^64`), so the raw
lane is deterministic whatever representation the earlier rounds happen to
carry.

TRUST. This is a THIRD transcription of the permutation, so it trusts nothing
about itself: before printing, it reproduces every row of the header's Table 2
forward and inverts each one back to its input. Run from anywhere:

    python3 crypto/math-cuda/tests/host_kat/rpx_canon_witness.py

The printed input goes into `prover/tests/rpx_host_kat_vectors.rs`
(`permutation_inputs`, the row named "canonicalisation witness"); its output
row comes from that generator, never from here.
"""
import pathlib
import random
import re

REPO = pathlib.Path(__file__).resolve().parents[4]
P = (1 << 64) - (1 << 32) + 1
INV_ALPHA = 10540996611094048183  # rpo.rs:96
assert (7 * INV_ALPHA) % (P - 1) == 1
ROW = [7, 23, 8, 26, 13, 10, 9, 7, 6, 22, 21, 8]  # rpo.rs:114

RPO_RS = (REPO / "prover/src/lfm/rpo.rs").read_text()


def constant_table(name):
    m = re.search(
        r"pub const %s: \[\[u64; HASH_STATE_FELTS\]; NUM_ROUNDS\] = \[(.*?)\n\];" % name,
        RPO_RS,
        re.S,
    )
    rows = re.findall(r"\[\s*((?:\d+,\s*)+)\]", m.group(1))
    vals = [[int(x) for x in re.findall(r"\d+", r)] for r in rows]
    assert len(vals) == 7 and all(len(r) == 12 for r in vals), name
    return vals


ARK1, ARK2 = constant_table("ARK1"), constant_table("ARK2")


# --- the field, the MDS and its inverse, the cubic extension -------------------------

def mds(s):
    return [sum(ROW[(j - i) % 12] * s[j] for j in range(12)) % P for i in range(12)]


def matrix_inverse_mod_p(m):
    n = len(m)
    a = [row[:] + [1 if i == j else 0 for j in range(n)] for i, row in enumerate(m)]
    for col in range(n):
        piv = next(r for r in range(col, n) if a[r][col] % P)
        a[col], a[piv] = a[piv], a[col]
        inv = pow(a[col][col], P - 2, P)
        a[col] = [(v * inv) % P for v in a[col]]
        for r in range(n):
            if r != col and a[r][col]:
                f = a[r][col]
                a[r] = [(vr - f * vc) % P for vr, vc in zip(a[r], a[col])]
    return [row[n:] for row in a]


MDS_INV = matrix_inverse_mod_p([[ROW[(j - i) % 12] for j in range(12)] for i in range(12)])


def mds_inv(s):
    return [sum(MDS_INV[i][j] * s[j] for j in range(12)) % P for i in range(12)]


def ext_mul(a, b):  # rpx.rs:118-125, φ³ = φ + 1
    return [
        (a[0] * b[0] + a[1] * b[2] + a[2] * b[1]) % P,
        (a[0] * b[1] + a[1] * b[0] + a[1] * b[2] + a[2] * b[1] + a[2] * b[2]) % P,
        (a[0] * b[2] + a[1] * b[1] + a[2] * b[0] + a[2] * b[2]) % P,
    ]


def ext_pow(a, e):
    r, b = [1, 0, 0], a[:]
    while e:
        if e & 1:
            r = ext_mul(r, b)
        b = ext_mul(b, b)
        e >>= 1
    return r


EXT_INV7 = pow(7, -1, P**3 - 1)  # x ↦ x^7 permutes GF(p³) (rpx.rs tests), so this exists


# --- the permutation, forward (rpx.rs:280-316) and inverse ----------------------------

def add_constants(s, table, r, sign=1):
    return [(v + sign * table[r][i]) % P for i, v in enumerate(s)]


def fb_round(s, r):
    s = add_constants(mds(s), ARK1, r)
    s = mds([pow(v, 7, P) for v in s])
    return [pow(v, INV_ALPHA, P) for v in add_constants(s, ARK2, r)]


def ext_round(s, r):
    s = add_constants(s, ARK1, r)
    return sum((ext_pow(s[3 * e:3 * e + 3], 7) for e in range(4)), [])


def final_round(s):
    return add_constants(mds(s), ARK1, 6)


def permute(s):
    for r in range(6):
        s = fb_round(s, r) if r % 2 == 0 else ext_round(s, r)
    return final_round(s)


def fb_round_inv(s, r):
    s = add_constants([pow(v, 7, P) for v in s], ARK2, r, -1)
    s = [pow(v, INV_ALPHA, P) for v in mds_inv(s)]
    return mds_inv(add_constants(s, ARK1, r, -1))


def ext_round_inv(s, r):
    s = sum((ext_pow(s[3 * e:3 * e + 3], EXT_INV7) for e in range(4)), [])
    return add_constants(s, ARK1, r, -1)


def rounds_0_to_5_inv(t):
    for r in (5, 4, 3, 2, 1, 0):
        t = ext_round_inv(t, r) if r % 2 == 1 else fb_round_inv(t, r)
    return t


# --- self-check against the header's oracle table before trusting any of the above ----

HEADER = (REPO / "crypto/math-cuda/tests/host_kat/rpx_kat_vectors.h").read_text()
body = HEADER.split("RPX_PERMUTATION_VECTORS[NUM_RPX_PERMUTATION_VECTORS] = {")[1].split("};")[0]
rows = re.findall(r'\{"([^"]*)",\s*\{([^}]*)\},\s*\{([^}]*)\}\}', body)
assert len(rows) >= 8, "oracle table has %d rows" % len(rows)
for name, inp, outp in rows:
    x = [int(v) for v in re.findall(r"\d+", inp)]
    y = [int(v) for v in re.findall(r"\d+", outp)]
    assert permute(x) == y, "forward transcription disagrees with the oracle on %r" % name
    t = rounds_0_to_5_inv(mds_inv(add_constants(y, ARK1, 6, -1)))
    assert t == x, "inverse permutation does not round-trip on %r" % name
print("self-check: %d/%d oracle rows reproduced forward and inverted back" % (len(rows), len(rows)))

# --- the witness -------------------------------------------------------------------------

c0 = ARK1[6][0]
rng = random.Random(0x4B57)  # "KW"; one generator, eleven draws
u = [P - c0 + 1] + [rng.randrange(P) for _ in range(11)]
assert P - c0 <= u[0] < P - c0 + (1 << 32) - 1
x = rounds_0_to_5_inv(mds_inv(u))
y = permute(x)
assert y[0] == 1 and y == add_constants(u, ARK1, 6)
print("witness input :", ", ".join(str(v) for v in x))
print("witness output:", ", ".join(str(v) for v in y), "   (lane 0 raw on device: %d = p + 1)" % (u[0] + c0))

# The generator's hard-coded row must be exactly this input, or the header's
# witness and this derivation have drifted apart.
GENERATOR = (REPO / "prover/tests/rpx_host_kat_vectors.rs").read_text()
block = GENERATOR.split('"canonicalisation witness"', 1)[1].split("]", 1)[0]
assert [int(v) for v in re.findall(r"\d+", block)] == x, "the generator's witness row is not this derivation's"
print("generator row check: prover/tests/rpx_host_kat_vectors.rs carries this exact input")
