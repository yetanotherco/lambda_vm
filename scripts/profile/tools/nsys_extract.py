#!/usr/bin/env python3
"""Text tables from ONE Nsight Systems sqlite export.

Runs on the machine that captured. The sqlite (and the .nsys-rep) hold the process environment, secrets included, in
META_DATA_CAPTURE / TARGET_INFO_SYSTEM_ENV: this script never reads those tables, and the profile deletes the raw files
after it. It reads only: TARGET_INFO_SESSION_START_TIME, StringIds (names of the rows it reads), PROCESSES (pid and a
truncated basename), ThreadNames, the CUPTI kernel / memcpy / memset / runtime tables, OSRT_API and DIAGNOSTIC_EVENT
(severity counts, plus loss messages without '=').

Method: busy = the union of kernels, copies and memsets; each instant of busy time is shared
equally among the device operations active at it, so per-name shares add up to busy. Time is binned (default 0.1 s,
trace clock = ns since the session start; unix = t0_unix_ns + trace ns) so phase windows from the provers' own logs can
be applied afterwards, on text.

usage: nsys_extract.py <export.sqlite> <outdir> [bin_seconds]
writes: summary.tsv names.tsv bins.tsv share.tsv gaps.tsv copies.tsv api.tsv osrt.tsv threads.tsv procs.tsv diag.tsv
"""
import os
import re
import sqlite3
import sys

import numpy as np

DB, OUT = sys.argv[1], sys.argv[2]
BIN = int(round(float(sys.argv[3] if len(sys.argv) > 3 else 0.1) * 1e9))
os.makedirs(OUT, exist_ok=True)
con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
TABLES = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
FORBIDDEN = ("META_DATA_CAPTURE", "TARGET_INFO_SYSTEM_ENV")


def q(sql, ncol, chunk=2_000_000):
    assert not any(f in sql for f in FORBIDDEN)
    cur = con.execute(sql)
    parts = []
    while True:
        rows = cur.fetchmany(chunk)
        if not rows:
            break
        parts.append(np.array(rows, dtype=np.int64).reshape(-1, ncol))
    return np.concatenate(parts) if parts else np.zeros((0, ncol), np.int64)


def columns(t):
    return {r[1] for r in con.execute(f"pragma table_info({t})")} if t in TABLES else set()


def w(name, header, rows):
    with open(os.path.join(OUT, name), "w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")


T0 = con.execute("select utcEpochNs from TARGET_INFO_SESSION_START_TIME").fetchone()[0]
STR = {}
_wanted = set()


def sid(ids):
    _wanted.update(int(x) for x in np.unique(ids) if x >= 0)


def load_strings():
    ids = sorted(_wanted)
    for i in range(0, len(ids), 50000):
        part = ids[i:i + 50000]
        for k, v in con.execute(f"select id, value from StringIds where id in ({','.join(map(str, part))})"):
            STR[k] = v


def clean(s, n):
    s = (s or "").replace("\t", " ").replace("\n", " ")
    return s[:n]


# ------------------------------------------------------------------ device operations
KT = "CUPTI_ACTIVITY_KIND_KERNEL"
kc = columns(KT)
gcol = "coalesce(graphNodeId,-1)" if "graphNodeId" in kc else "-1"
K = q(f"select start,end,coalesce(shortName,-1),coalesce(demangledName,-1),streamId,coalesce(globalPid,0),{gcol} from {KT}", 7) if kc else np.zeros((0, 7), np.int64)
M = q("select start,end,bytes,copyKind,coalesce(srcKind,7),coalesce(dstKind,7),streamId,coalesce(globalPid,0) from CUPTI_ACTIVITY_KIND_MEMCPY", 8) \
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in TABLES else np.zeros((0, 8), np.int64)
S = q("select start,end,bytes,streamId,coalesce(globalPid,0) from CUPTI_ACTIVITY_KIND_MEMSET", 5) \
    if "CUPTI_ACTIVITY_KIND_MEMSET" in TABLES else np.zeros((0, 5), np.int64)
sid(K[:, 2]); sid(K[:, 3])

# op keys: kernels by shortName id (>= 0); memcpy -(1000 + 100 kind + 10 src + dst); memset -1
mkey = -(1000 + 100 * M[:, 3] + 10 * M[:, 4] + M[:, 5])
starts = np.concatenate([K[:, 0], M[:, 0], S[:, 0]])
ends = np.concatenate([K[:, 1], M[:, 1], S[:, 1]])
ends = np.maximum(ends, starts)
keys = np.concatenate([K[:, 2], mkey, -np.ones(len(S), np.int64)])
iskern = np.concatenate([np.ones(len(K), bool), np.zeros(len(M) + len(S), bool)])
NOPS = len(starts)
tmax = int(ends.max()) if NOPS else 0
nb = tmax // BIN + 1
edges = np.arange(0, (nb + 1) * BIN, BIN, dtype=np.int64)

T = np.unique(np.concatenate([starts, ends, edges]))
si = np.searchsorted(T, starts)
ei = np.searchsorted(T, ends)


def active(mask):
    d = np.bincount(si[mask], minlength=len(T) + 1)[: len(T) + 1] - np.bincount(ei[mask], minlength=len(T) + 1)[: len(T) + 1]
    return np.cumsum(d)[: len(T) - 1]


seg = np.diff(T).astype(np.float64)
cnt = active(np.ones(NOPS, bool))
kcnt = active(iskern)
segbin = np.minimum(T[:-1] // BIN, nb - 1)
wseg = np.where(cnt > 0, seg / np.maximum(cnt, 1), 0.0)
P = np.concatenate([[0.0], np.cumsum(wseg)])
Pt = lambda t: P[np.searchsorted(T, t)]  # noqa: E731 — every t passed here is in T


def binned_split(s, e, val_fn):
    """Per-op values split over the bins each op touches; returns (bin, op index, value) arrays."""
    b0 = s // BIN
    b1 = np.maximum(e - 1, s) // BIN
    one = b0 == b1
    idx = np.flatnonzero(one)
    bins = [b0[idx]]
    ops = [idx]
    vals = [val_fn(s[idx], e[idx])]
    for i in np.flatnonzero(~one):
        for b in range(int(b0[i]), int(b1[i]) + 1):
            lo, hi = max(int(s[i]), b * BIN), min(int(e[i]), (b + 1) * BIN)
            if hi > lo:
                bins.append(np.array([b])); ops.append(np.array([i])); vals.append(val_fn(np.array([lo]), np.array([hi])))
    return np.concatenate(bins), np.concatenate(ops), np.concatenate(vals).astype(np.float64)


def group_sum(gkeys, vals):
    u, inv = np.unique(gkeys, return_inverse=True)
    return u, np.bincount(inv, weights=vals)


# per-bin busy, kernel union, >= 2 kernels at once
busy_b = np.bincount(segbin, weights=seg * (cnt > 0), minlength=nb)
kun_b = np.bincount(segbin, weights=seg * (kcnt > 0), minlength=nb)
k2_b = np.bincount(segbin, weights=seg * (kcnt > 1), minlength=nb)

# per (bin, op key): share (equal split of busy), raw duration, launches (by start bin)
ukeys, kidx = np.unique(keys, return_inverse=True)
NK = len(ukeys)
bb, oo, sh = binned_split(starts, ends, lambda a, b: Pt(b) - Pt(a))
_, _, raw = binned_split(starts, ends, lambda a, b: (b - a))
g, sh_sum = group_sum(bb * NK + kidx[oo], sh)
g2, raw_sum = group_sum(bb * NK + kidx[oo], raw)
assert np.array_equal(g, g2)
g3, n_sum = group_sum((starts // BIN) * NK + kidx, np.ones(NOPS))
nmap = dict(zip(g3.tolist(), n_sum.tolist()))
w("share.tsv", ["bin", "key", "share_ns", "raw_ns", "launches"],
  ((int(x // NK), int(ukeys[x % NK]), int(round(a)), int(round(b)), int(nmap.get(int(x), 0))) for x, a, b in zip(g.tolist(), sh_sum, raw_sum)))

# idle gaps inside [first op, last op]: maximal idle runs, classed by full length, split over bins exactly
first = int(starts.min()) if NOPS else 0
idle = (cnt == 0) & (T[:-1] >= first) & (T[1:] <= tmax)
run_start = idle & np.concatenate([[True], ~idle[:-1]])
gid = np.cumsum(run_start) - 1
gid_idle = gid[idle]
glen = np.bincount(gid_idle, weights=seg[idle]) if idle.any() else np.zeros(0)
CLS = np.array([1e5, 1e6, 1e7, 1e8])  # <100 us, <1 ms, <10 ms, <100 ms, >= 100 ms
gcls = np.searchsorted(CLS, glen, side="right")
# what bounds each gap: the last op ending at its start, the first op starting at its end
cat = np.concatenate([np.zeros(len(K), np.int64), np.where(M[:, 3] == 1, 1, np.where(M[:, 3] == 2, 2, 3)), np.full(len(S), 4)])
CATN = ["kernel", "H2D", "D2H", "copy-other", "memset", "-"]
gstart = T[:-1][run_start]
gend = gstart + np.round(glen).astype(np.int64) if len(glen) else gstart
o_end = np.argsort(ends, kind="stable")
o_st = np.argsort(starts, kind="stable")
pi = np.searchsorted(ends[o_end], gstart, side="right") - 1
ni = np.searchsorted(starts[o_st], gend, side="left")
prev_c = np.where(pi >= 0, cat[o_end[np.maximum(pi, 0)]], 5)
next_c = np.where(ni < NOPS, cat[o_st[np.minimum(ni, NOPS - 1)]], 5)
gb = segbin[idle]
gk = ((gb * 5 + gcls[gid_idle]) * 6 + prev_c[gid_idle]) * 6 + next_c[gid_idle]
u, v = group_sum(gk, seg[idle]) if idle.any() else (np.zeros(0, np.int64), np.zeros(0))
gcount = {}
for k_ in ((gstart // BIN) * 5 + gcls) * 36 + prev_c * 6 + next_c:
    gcount[int(k_)] = gcount.get(int(k_), 0) + 1
w("gaps.tsv", ["bin", "class", "prev", "next", "idle_ns", "gaps_starting"],
  ((int(x // 180), ["<100us", "<1ms", "<10ms", "<100ms", ">=100ms"][int(x // 36 % 5)], CATN[int(x // 6 % 6)], CATN[int(x % 6)], int(round(y)), gcount.get(int(x), 0))
   for x, y in zip(u.tolist(), v)))

w("bins.tsv", ["bin", "t0_s", "busy_ns", "kernel_union_ns", "kernels_ge2_ns"],
  ((b, f"{b * BIN / 1e9:.1f}", int(round(busy_b[b])), int(round(kun_b[b])), int(round(k2_b[b]))) for b in range(nb) if busy_b[b] > 0 or b * BIN >= first))

# copies: (bin, kind, src, dst) -> count, bytes (by start bin), raw ns (split)
if len(M):
    mb, mo, mraw = binned_split(M[:, 0], np.maximum(M[:, 1], M[:, 0]), lambda a, b: (b - a))
    ck = ((M[:, 3] * 10 + M[:, 4]) * 10 + M[:, 5])
    u1, v1 = group_sum(mb * 10000 + ck[mo], mraw)
    u2, c2 = group_sum((M[:, 0] // BIN) * 10000 + ck, np.ones(len(M)))
    u3, by3 = group_sum((M[:, 0] // BIN) * 10000 + ck, M[:, 2].astype(np.float64))
    cnts = dict(zip(u2.tolist(), c2.tolist())); byts = dict(zip(u3.tolist(), by3.tolist()))
    allk = sorted(set(u1.tolist()) | set(u2.tolist()))
    vmap = dict(zip(u1.tolist(), v1.tolist()))
    w("copies.tsv", ["bin", "kind", "src", "dst", "count", "bytes", "raw_ns"],
      ((x // 10000, x % 10000 // 100, x % 100 // 10, x % 10, int(cnts.get(x, 0)), int(byts.get(x, 0)), int(round(vmap.get(x, 0)))) for x in allk))
else:
    w("copies.tsv", ["bin", "kind", "src", "dst", "count", "bytes", "raw_ns"], [])


# host API tables: (bin, name) raw ns split over bins; per-thread totals
def host_table(table, fname):
    if table not in TABLES:
        w(fname, ["bin", "name_id", "count", "raw_ns"], [])
        return np.zeros((0, 4), np.int64)
    A = q(f"select start,end,coalesce(nameId,-1),coalesce(globalTid,0) from {table}", 4)
    sid(A[:, 2])
    if not len(A):
        w(fname, ["bin", "name_id", "count", "raw_ns"], [])
        return A
    e = np.maximum(A[:, 1], A[:, 0])
    ab, ao, araw = binned_split(A[:, 0], e, lambda a, b: (b - a))
    un, ni_ = np.unique(A[:, 2], return_inverse=True)
    u_, v_ = group_sum(ab * len(un) + ni_[ao], araw)
    uc, vc = group_sum((A[:, 0] // BIN) * len(un) + ni_, np.ones(len(A)))
    cm = dict(zip(uc.tolist(), vc.tolist()))
    w(fname, ["bin", "name_id", "count", "raw_ns"],
      ((x // len(un), int(un[x % len(un)]), int(cm.get(x, 0)), int(round(y))) for x, y in zip(u_.tolist(), v_)))
    return A


API = host_table("CUPTI_ACTIVITY_KIND_RUNTIME", "api.tsv")
OS_ = host_table("OSRT_API", "osrt.tsv")

# per-thread totals (top 60 threads by API + OSRT time), with thread names
tn = {}
if "ThreadNames" in TABLES:
    for nid, gt in con.execute("select nameId, globalTid from ThreadNames"):
        tn[gt] = nid
        _wanted.add(nid)
rows = []
for lab, A in (("api", API), ("osrt", OS_)):
    if not len(A):
        continue
    d = (np.maximum(A[:, 1], A[:, 0]) - A[:, 0]).astype(np.float64)
    gk_ = A[:, 3]
    ut, vt = group_sum(gk_, d)
    for t_, v_ in zip(ut.tolist(), vt.tolist()):
        rows.append((lab, t_, v_))
tot = {}
for lab, t_, v_ in rows:
    tot[t_] = tot.get(t_, 0) + v_
top = set(sorted(tot, key=lambda x: -tot[x])[:60])
load_strings()
w("threads.tsv", ["table", "globalTid", "pid", "tid", "thread_name", "raw_ns"],
  ((lab, t_, t_ >> 24 & 0xFFFFFF, t_ & 0xFFFFFF, clean(STR.get(tn.get(t_, -1), ""), 40), int(round(v_))) for lab, t_, v_ in rows if t_ in top))

# names: kernels (short + demangled, truncated), copies, memset; API / OSRT names
ENUMK = dict(con.execute("select id, label from ENUM_CUDA_MEMCPY_OPER")) if "ENUM_CUDA_MEMCPY_OPER" in TABLES else {}
ENUMM = dict(con.execute("select id, label from ENUM_CUDA_MEM_KIND")) if "ENUM_CUDA_MEM_KIND" in TABLES else {}
dem = {}
for a, b in zip(K[:, 2].tolist(), K[:, 3].tolist()):
    dem.setdefault(a, b)
nrows = []
for k_ in ukeys.tolist():
    if k_ >= 0:
        nrows.append(("K", k_, clean(STR.get(k_, "?"), 120), clean(STR.get(dem.get(k_, -1), ""), 300)))
    elif k_ == -1:
        nrows.append(("S", k_, "MEMSET", ""))
    else:
        c_ = -k_ - 1000
        nrows.append(("M", k_, f"MEMCPY {ENUMK.get(c_ // 100, c_ // 100)} {ENUMM.get(c_ // 10 % 10, c_ // 10 % 10)}->{ENUMM.get(c_ % 10, c_ % 10)}", ""))
for lab, A in (("API", API), ("OSRT", OS_)):
    for n_ in (np.unique(A[:, 2]).tolist() if len(A) else []):
        nrows.append((lab, n_, clean(STR.get(n_, "?"), 120), ""))
w("names.tsv", ["kind", "key", "name", "demangled"], nrows)

# processes: pid and a truncated basename; per process kernel / copy counts and raw time
procs = []
if "PROCESSES" in TABLES:
    for gp, pid, name in con.execute("select globalPid, pid, name from PROCESSES"):
        km = K[:, 5] == gp
        mm = M[:, 7] == gp
        procs.append((pid, clean(os.path.basename(name or ""), 40), int(km.sum()), int((K[km, 1] - K[km, 0]).sum()), int(mm.sum()), int(M[mm, 2].sum())))
w("procs.tsv", ["pid", "name", "kernels", "kernel_raw_ns", "memcpys", "memcpy_bytes"], procs)

# diagnostics: severity counts; loss / overflow messages only, never one with '=' in it
drows = []
if "DIAGNOSTIC_EVENT" in TABLES:
    sev = {}
    for s_, t_ in con.execute("select severity, text from DIAGNOSTIC_EVENT"):
        sev[s_] = sev.get(s_, 0) + 1
        if t_ and re.search(r"lost|drop|overflow|discard|incomplete|buffer", t_, re.I) and "=" not in t_:
            drows.append(("msg", s_, clean(t_, 160)))
    drows = [("severity", s_, n_) for s_, n_ in sorted(sev.items())] + drows[:40]
w("diag.tsv", ["what", "severity", "value"], drows)

# summary
busy = float(busy_b.sum()); kun = float(kun_b.sum()); k2 = float(k2_b.sum())
span = (tmax - first) if NOPS else 0
graph_k = int((K[:, 6] >= 0).sum()) if len(K) else 0
summ = [("t0_unix_ns", T0), ("bin_ns", BIN), ("first_op_s", f"{first / 1e9:.3f}"), ("last_op_s", f"{tmax / 1e9:.3f}"),
        ("kernels", len(K)), ("kernels_in_graphs", graph_k), ("memcpys", len(M)), ("memsets", len(S)), ("api_calls", len(API)), ("osrt_calls", len(OS_)),
        ("busy_s", f"{busy / 1e9:.3f}"), ("kernel_union_s", f"{kun / 1e9:.3f}"), ("kernels_ge2_s", f"{k2 / 1e9:.3f}"),
        ("active_span_s", f"{span / 1e9:.3f}"), ("busy_pct_of_span", f"{100 * busy / span:.1f}" if span else "-"),
        ("idle_in_span_s", f"{(span - busy) / 1e9:.3f}")]
for kind, lab in ((1, "h2d"), (2, "d2h"), (8, "d2d")):
    m_ = M[:, 3] == kind
    summ.append((f"{lab}_bytes", int(M[m_, 2].sum())))
    summ.append((f"{lab}_raw_s", f"{(M[m_, 1] - M[m_, 0]).sum() / 1e9:.3f}"))
    summ.append((f"{lab}_pinned_bytes", int(M[m_ & ((M[:, 4] == 1) | (M[:, 5] == 1)), 2].sum())))
w("summary.tsv", ["key", "value"], summ)
print("NSYS EXTRACT: " + " · ".join(f"{k} {v}" for k, v in summ if k not in ("t0_unix_ns", "bin_ns")))
