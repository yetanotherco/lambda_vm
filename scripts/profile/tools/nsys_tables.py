#!/usr/bin/env python3
"""Windows + families over the text tables nsys_extract.py wrote.

    import nsys_tables as R; c = R.load(dir); R.window(c, a_unix, b_unix, rules)
"""
import csv
import os
import re
from collections import defaultdict


def _tsv(p):
    with open(p) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load(d):
    x = os.path.join(d, "x")
    c = {"dir": d}
    c["summary"] = {r["key"]: r["value"] for r in _tsv(os.path.join(x, "summary.tsv"))}
    c["t0"] = int(c["summary"]["t0_unix_ns"]) / 1e9
    c["bin"] = int(c["summary"]["bin_ns"]) / 1e9
    c["names"] = {}
    for r in _tsv(os.path.join(x, "names.tsv")):
        c["names"][(r["kind"], int(r["key"]))] = (r["name"], r["demangled"])
    c["share"] = [(int(r["bin"]), int(r["key"]), int(r["share_ns"]), int(r["raw_ns"]), int(r["launches"])) for r in _tsv(os.path.join(x, "share.tsv"))]
    c["bins"] = {int(r["bin"]): (int(r["busy_ns"]), int(r["kernel_union_ns"]), int(r["kernels_ge2_ns"])) for r in _tsv(os.path.join(x, "bins.tsv"))}
    c["gaps"] = [(int(r["bin"]), r["class"], r["prev"], r["next"], int(r["idle_ns"])) for r in _tsv(os.path.join(x, "gaps.tsv"))]
    c["copies"] = [(int(r["bin"]), int(r["kind"]), int(r["src"]), int(r["dst"]), int(r["count"]), int(r["bytes"]), int(r["raw_ns"])) for r in _tsv(os.path.join(x, "copies.tsv"))]
    c["api"] = [(int(r["bin"]), int(r["name_id"]), int(r["count"]), int(r["raw_ns"])) for r in _tsv(os.path.join(x, "api.tsv"))]
    return c


def keyname(c, k):
    if k >= 0:
        return c["names"].get(("K", k), ("?", ""))
    if k == -1:
        return ("MEMSET", "")
    return c["names"].get(("M", k), ("MEMCPY ?", ""))


def family(c, k, rules):
    n, dem = keyname(c, k)
    if k < 0:
        if n.startswith("MEMCPY"):
            p = n.split()
            return "copy " + {"Host-to-Device": "H2D", "Device-to-Host": "D2H", "Device-to-Device": "D2D"}.get(p[1], p[1])
        return "memset"
    for fam, rx in rules:
        if re.search(rx, n) or (dem and re.search(rx, dem)):
            return fam
    return "other kernels"


def bins_of(c, a, b):
    """Bins fully or partly inside [a, b) unix seconds, as (first, last_exclusive); edges rounded to the 0.1 s bin."""
    return int(round((a - c["t0"]) / c["bin"])), int(round((b - c["t0"]) / c["bin"]))


def window(c, a, b, rules):
    b0, b1 = bins_of(c, a, b)
    wall = (b1 - b0) * c["bin"]
    busy = sum(c["bins"].get(i, (0, 0, 0))[0] for i in range(b0, b1)) / 1e9
    kun = sum(c["bins"].get(i, (0, 0, 0))[1] for i in range(b0, b1)) / 1e9
    k2 = sum(c["bins"].get(i, (0, 0, 0))[2] for i in range(b0, b1)) / 1e9
    fam = defaultdict(float)
    top = defaultdict(lambda: [0.0, 0.0, 0])
    for bi, k, sh, raw, n in c["share"]:
        if b0 <= bi < b1:
            fam[family(c, k, rules)] += sh / 1e9
            t = top[k]; t[0] += sh / 1e9; t[1] += raw / 1e9; t[2] += n
    gaps = defaultdict(float)
    gctx = defaultdict(float)
    for bi, cls, pv, nx, ns in c["gaps"]:
        if b0 <= bi < b1:
            gaps[cls] += ns / 1e9
            if cls in ("<10ms", "<100ms", ">=100ms"):
                gctx[f"{pv}->{nx}"] += ns / 1e9
    cp = defaultdict(lambda: [0, 0.0, 0])
    for bi, kind, s, d, n, by, raw in c["copies"]:
        if b0 <= bi < b1:
            e = cp[(kind, s, d)]; e[0] += by; e[1] += raw / 1e9; e[2] += n
    return dict(wall=wall, busy=busy, kun=kun, k2=k2, idle=wall - busy, fam=dict(fam), top=dict(top), gaps=dict(gaps), gctx=dict(gctx), copies=dict(cp))


def busy_series(c, a, b, step=1.0):
    b0, b1 = bins_of(c, a, b)
    per = int(round(step / c["bin"]))
    out = []
    for s in range(b0, b1, per):
        out.append(sum(c["bins"].get(i, (0, 0, 0))[0] for i in range(s, min(s + per, b1))) / 1e9 / (min(s + per, b1) - s) / c["bin"])
    return out


def stamps(path, patterns):
    """First unix stamp of each regex in a stamp.py log."""
    out = {}
    with open(path, errors="replace") as f:
        for line in f:
            t, _, rest = line.partition(" ")
            for k, rx in patterns.items():
                if k not in out and re.search(rx, rest):
                    out[k] = float(t)
    return out


OURS = [
    ("RPX grind", r"^rpx_grind"),
    ("RPX leaves", r"^rpx_(leaves|comp_poly_leaves|fri_group_leaves|fri_leaves|.*leaves)"),
    ("RPX Merkle internal", r"^rpx_(merkle|tail)"),
    ("RPX other", r"^rpx_"),
    ("NTT / LDE", r"^(ntt_|mobius_|lift_spread|pointwise_mul|scalar_mul|coset|lde_|bit_reverse_lde)"),
    ("sumcheck / GKR / fold", r"^(sumcheck_|sum_partials|whir_|eq_expand|eq_seed|fraction_fold|mle_|fri_fold|fri_update|add_scaled|fill_ext3|gkr_|zc_)"),
    ("constraint / LogUp / quotient", r"^(constraint_|ccomp_|program_|interp|factors_from|logup_|decompose|comp_h|quotient|bytecode|compiled_|lfm_constraint)"),
    ("DEEP / barycentric / inversion", r"^(deep_|barycentric|compute_denoms|batch_inverse|invert|block_inclusive_scan|apply_block_offsets|ood_)"),
    ("transpose / gather / pack", r"^(matrix_transpose|transpose|bit_reverse|gather|merkle_gather|pack|unpack|narrow|widen_)"),
]
