#!/usr/bin/env python3
"""G6 ledger: the #1010 production tree's base, from one nsys trace (runs ON THE BOX).

Usage:
  g6_trace.py --db <export.sqlite> --log <tree log> --out <dir> [--sampler <g6_sampler tsv>] [--no-mechanism]
  g6_trace.py --log-only --log <tree log> --sampler <tsv> --out <dir>      (a plain arm: the log and the sampler)

Built on G1's g1_trace.py (thoughts/zf/box, md5 f41867ba…): classify, the interval helpers, Trace, Log, Partition and
Mechanism are copied unchanged. The windows follow today's tree (pure WHIR, fan-in 5, head-ahead), and the base is
split into its sub-phases.

Reads ONLY activity tables (CUPTI kernel/memcpy/memset/runtime, OSRT_API), the StringIds entries those tables
reference, ThreadNames, ENUM_CUDA_MEM_KIND, TARGET_INFO_SESSION_START_TIME, TARGET_INFO_GPU and PROFILER_OVERHEAD.
It never reads META_DATA_*, TARGET_INFO_SYSTEM_ENV or PROCESSES (the environment lives there). Writes text and TSV
only. Clock: trace ns = unix ns - TARGET_INFO_SESSION_START_TIME.utcEpochNs; the log's `t=` stamps and the sampler's
are unix seconds on the same host clock.

Windows:
  base   = [`BASE HEAD (WHIR): start t=`, `★ LEVEL 1 (wide) START t=`]   (both ms stamps)
  level1 = [LEVEL 1 START, + `level 1 (wide): … in Xs`]                  (the wall is printed to 0.1 s)
  root   = the rest, to the base start + `WHOLE RUN … Xs total`          (0.1 s)
  A log without those lines (before head-ahead and the wide tree) falls back to G1's MARK-based windows.
Base sub-phases, disjoint by construction, every edge a log stamp:
  head          base start -> the prover thread's first phase (epoch 0's prep or absorb);
  <scope> absorb+prep · commit · argue · open   for scope = the epochs, and the global stage (`global`):
                argue = [prove start + challenge, + argue] and open = [argue end, prove end], from each prove span and
                its `WHIR PROVE SPLIT` line (challenge, argue, open_groups, open_prepared are timed in that order
                inside multi_prove, crypto/stark/src/multilinear_table.rs);
  prover gaps   the rest of the base: the prover thread between its phases (waiting at the hand-off, prints).
Partition of every window and sub-phase (exact, checked to 2 ms), as G1: an instant with a kernel is split equally
among the kernels (stage rows 1-6, 0), else among copies/memsets (row 7, by kind: H2D from pageable or pinned, D2H,
D2D, memset), else idle (row 8). Sumcheck-family kernels are stage 4 inside an argue window and stage 5 inside an
open window.
Host contention (the sampler, g6_sampler.py, every 0.05 s; a counter over an interval is interpolated between
samples): per sub-phase, the process's on-CPU and runqueue-wait seconds, the prover and producer threads' own, the
box's busy CPUs (/proc/stat) against its online CPUs, the cgroup's CPU and its CFS-throttled seconds. The prover
thread = the thread with the most CUDA API time inside the argue windows (trace); without a trace (--log-only), the
test's own thread by the name rule (the lowest tid named after the test), which the traced arm checks against the
CUDA API. The producer = the busiest other thread inside the `execute` spans (its rayon helpers are not in that
column: they are in the process total). Named host work from the log: the producer's stages per epoch (execute,
collect, build, prep@producer, and the hand-off wait = the handoff span minus prep@producer), the head's helper, the
level-1 prologues in the base's tail, and their overlap with every sub-phase."""
import argparse, collections, os, re, sqlite3
import numpy as np

# ------------------------------------------------------------------ kernel -> stage / family
# Ordered; the first match wins. (stage, family, regex on the kernel's short name). G1's rules at 0428c393b, plus the
# two sumcheck kernels the head added since (crypto/math-cuda/kernels/sumcheck.cu: batched_column_ext3,
# gather_factor_heads_ext3); every other kernel in crypto/math-cuda/kernels/*.cu at 9e2728955 is covered below.
RULES = [
    ("6", "6 grind", r"(^|_)grind_search"),
    ("3", "3a leaves", r"^(rpx|blake3|keccak256|keccak)_(leaves_|comp_poly_leaves|fri_leaves|fri_group_leaves)"),
    ("3", "3b merkle internal", r"^(rpx|blake3|keccak)_merkle_(level|tail)"),
    ("2", "2 NTT column-major engine (K1)", r"^ntt_cm_di[tf]_k\d$"),
    ("2", "2 NTT legacy", r"^ntt_(dit|dif)"),
    ("2", "2 Mobius (multilinear->coeff)", r"^mobius_"),
    ("2", "2 spread/scale/pointwise", r"^(lift_spread|pointwise_mul|scalar_mul)"),
    ("2", "2 bit-reverse/transpose", r"^(bit_reverse_permute|bit_reverse_row_major|matrix_transpose)"),
    ("4", "4 quotient (constraint IR)", r"^(constraint_composition_kernel|constraint_interp_kernel|comp_h_to_slabs|decompose_d2)"),
    ("1", "1 LogUp aux trace", r"^logup_"),
    ("5", "5 DEEP", r"^(deep_composition|bit_reverse_ext3_interleaved)"),
    ("5", "5 DEEP/OOD inversion", r"^(compute_denoms|block_inclusive_scan|apply_block_offsets|batch_inverse_combine|invert_total|invert_denoms_rowwise)"),
    ("5", "5 OOD barycentric", r"^(barycentric_|gather_rows_)"),
    ("5", "5 FRI fold", r"^(fri_fold|fri_update_twiddles)"),
    ("5", "5 WHIR fold", r"^(whir_fold|whir_lean_|gather_cosets)"),
    ("5", "5 query gather", r"^(gather_ext3_at|merkle_gather_paths)"),
    ("45", "sumcheck family", r"^(sumcheck_|sum_partials|program_map|factors_from_columns|eq_expand|eq_seed|fraction_fold|mle_fold|mle_lift|add_scaled|fill_ext3|batched_column_ext3|gather_factor_heads_ext3)"),
    ("0", "0 arith helpers", r"^(gl_|ext3_(add|sub|mul)_kernel|vector_add_u64)"),
]
_RULES_RX = [(s, f, re.compile(r)) for s, f, r in RULES]
STAGES = ["1", "2", "3", "4", "5", "6", "0"]
STAGE_NAMES = {
    "1": "1 execution/trace generation (device part)", "2": "2 LDE/encoding (NTT)", "3": "3 commitment hashing",
    "4": "4 constraint evaluation", "5": "5 openings", "6": "6 grinding", "0": "0 unmapped/other kernels",
    "7": "7 copy/memset-only", "8": "8 idle card"}


def classify(name):
    for s, f, rx in _RULES_RX:
        if rx.search(name):
            return s, f
    return "0", "0 unmapped: " + name


# ------------------------------------------------------------------ interval helpers
def union(starts, ends):
    starts = np.asarray(starts, np.int64); ends = np.asarray(ends, np.int64)
    if len(starts) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    o = np.argsort(starts, kind="stable")
    s = starts[o]; e = ends[o]
    emax = np.maximum.accumulate(e)
    new = np.ones(len(s), bool)
    new[1:] = s[1:] > emax[:-1]
    idx = np.flatnonzero(new)
    return s[idx], np.append(emax[idx[1:] - 1], emax[-1])


def clip_len(S, E, a, b):
    s = np.clip(S, a, b); e = np.clip(E, a, b)
    return int(np.sum(np.maximum(e - s, 0)))


def complement(S, E, a, b):
    m = (E > a) & (S < b)
    s = np.clip(S[m], a, b); e = np.clip(E[m], a, b)
    gs = np.concatenate([[a], e]).astype(np.int64); ge = np.concatenate([s, [b]]).astype(np.int64)
    k = ge > gs
    return gs[k], ge[k]


def intersect(S1, E1, S2, E2):
    out_s, out_e = [], []
    i = j = 0
    while i < len(S1) and j < len(S2):
        s = max(S1[i], S2[j]); e = min(E1[i], E2[j])
        if e > s:
            out_s.append(s); out_e.append(e)
        if E1[i] < E2[j]:
            i += 1
        else:
            j += 1
    return np.array(out_s, np.int64), np.array(out_e, np.int64)


def subtract(S1, E1, S2, E2):
    """S1 minus S2 (both disjoint and sorted)."""
    if len(S1) == 0 or len(S2) == 0:
        return S1, E1
    out_s, out_e = [], []
    j = 0; n2 = len(S2)
    for s, e in zip(S1.tolist(), E1.tolist()):
        cur = s
        while j < n2 and E2[j] <= cur:
            j += 1
        k = j
        while k < n2 and S2[k] < e:
            if S2[k] > cur:
                out_s.append(cur); out_e.append(min(S2[k], e))
            cur = max(cur, E2[k])
            if cur >= e:
                break
            k += 1
        if cur < e:
            out_s.append(cur); out_e.append(e)
    return np.array(out_s, np.int64), np.array(out_e, np.int64)


def measure_fn(S_, E_):
    """F(t) = measure of the disjoint sorted set (S_, E_) inside (-inf, t]."""
    if len(S_) == 0:
        return lambda t: np.zeros(np.shape(t))
    cum = np.concatenate([[0], np.cumsum(E_ - S_)])

    def F(t):
        t = np.asarray(t)
        j = np.searchsorted(S_, t, side="right") - 1
        jj = np.maximum(j, 0)
        part = np.where(j >= 0, np.maximum(np.minimum(t, E_[jj]) - S_[jj], 0), 0)
        return np.where(j >= 0, cum[jj] + part, 0)
    return F


def sec(ns):
    return float(ns) / 1e9


# ------------------------------------------------------------------ trace
class Trace:
    def __init__(self, db):
        self.c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        tabs = {r[0] for r in self.c.execute("select name from sqlite_master where type='table'")}
        self.t0 = self.c.execute("select utcEpochNs from TARGET_INFO_SESSION_START_TIME").fetchone()[0]

        def q(sql, ncol):
            rows = self.c.execute(sql).fetchall()
            return np.array(rows, dtype=np.int64) if rows else np.zeros((0, ncol), np.int64)
        self.K = q("select start,end,streamId,correlationId,shortName,gridX,gridY,gridZ,blockX,blockY,blockZ,"
                   "registersPerThread,globalPid from CUPTI_ACTIVITY_KIND_KERNEL order by start", 13)
        self.M = q("select start,end,streamId,correlationId,bytes,copyKind,srcKind,dstKind from "
                   "CUPTI_ACTIVITY_KIND_MEMCPY order by start", 8) if "CUPTI_ACTIVITY_KIND_MEMCPY" in tabs else np.zeros((0, 8), np.int64)
        self.Z = q("select start,end,streamId,correlationId,bytes from CUPTI_ACTIVITY_KIND_MEMSET order by start", 5) \
            if "CUPTI_ACTIVITY_KIND_MEMSET" in tabs else np.zeros((0, 5), np.int64)
        self.R = q("select start,end,globalTid,nameId,correlationId from CUPTI_ACTIVITY_KIND_RUNTIME order by start", 5)
        self.O = q("select start,end,globalTid,nameId from OSRT_API where nestingLevel=0 order by start", 4) \
            if "OSRT_API" in tabs else np.zeros((0, 4), np.int64)
        ids = set(self.K[:, 4].tolist()) | set(self.R[:, 3].tolist()) | set(self.O[:, 3].tolist())
        self.S = {}
        for i in ids:
            r = self.c.execute("select value from StringIds where id=?", (int(i),)).fetchone()
            self.S[i] = r[0] if r else f"<id {i}>"
        self.kname = np.array([self.S[i] for i in self.K[:, 4]], dtype=object)
        self.rname = np.array([self.S[i] for i in self.R[:, 3]], dtype=object)
        self.oname = np.array([self.S[i] for i in self.O[:, 3]], dtype=object)
        self.tnames = {}
        if "ThreadNames" in tabs:
            for nid, gtid in self.c.execute("select nameId, globalTid from ThreadNames"):
                r = self.c.execute("select value from StringIds where id=?", (nid,)).fetchone()
                self.tnames[gtid] = r[0] if r else "?"
        self.memkind = dict(self.c.execute("select id, label from ENUM_CUDA_MEM_KIND")) if "ENUM_CUDA_MEM_KIND" in tabs else {}
        self.gpu = self.c.execute("select name, smCount from TARGET_INFO_GPU").fetchone() if "TARGET_INFO_GPU" in tabs else None
        self.overhead = []
        if "PROFILER_OVERHEAD" in tabs:
            try:
                cols = [r[1] for r in self.c.execute("pragma table_info(PROFILER_OVERHEAD)")]
                if "nameId" in cols:
                    for nid, n, d in self.c.execute("select nameId, count(*), sum(end-start) from PROFILER_OVERHEAD group by nameId"):
                        r = self.c.execute("select value from StringIds where id=?", (nid,)).fetchone()
                        self.overhead.append((r[0] if r else "?", n, d or 0))
            except sqlite3.Error:
                pass
        self.pids = (self.R[:, 2] >> 24) & 0xFFFFFF
        self.PID = collections.Counter(self.pids.tolist()).most_common(1)[0][0] if len(self.R) else -1
        self.op_s = np.concatenate([self.K[:, 0], self.M[:, 0], self.Z[:, 0]])
        self.op_e = np.concatenate([self.K[:, 1], self.M[:, 1], self.Z[:, 1]])
        self.op_c = np.concatenate([self.K[:, 3], self.M[:, 3], self.Z[:, 3]])
        self.BS, self.BE = union(self.op_s, self.op_e)

    def ns(self, unix_s):
        return int(round(unix_s * 1e9)) - self.t0


# ------------------------------------------------------------------ the log
class Log:
    def __init__(self, path):
        self.L = open(path, encoding="utf-8", errors="replace").read().splitlines()

    def first(self, rx, cast=float):
        r = re.compile(rx)
        for l in self.L:
            m = r.search(l)
            if m:
                return cast(m.group(1))
        return None

    def all(self, rx):
        r = re.compile(rx)
        return [m for m in (r.search(l) for l in self.L) if m]


def parse_spans(lg):
    """Every stamped span: (kind, label, a_unix, b_unix, extra). G1's kinds plus the head's (`BASE HEAD (WHIR)`)."""
    out = []
    for m in lg.all(r"^BASE EPOCH (\w+): (\w+) ([\d.]+)s t=\[([\d.]+),([\d.]+)\]"):
        out.append(("epoch", f"{m[1]}:{m[2]}", float(m[4]), float(m[5]), {"epoch": m[1], "phase": m[2]}))
    for m in lg.all(r"^BASE PREP (\w+): (\S+) ([\d.]+)s t=\[([\d.]+),([\d.]+)\]"):
        out.append(("baseprep", f"{m[1]}:{m[2]}", float(m[4]), float(m[5]), {"epoch": m[1]}))
    for m in lg.all(r"^BASE HEAD \(WHIR\): (.+?) ([\d.]+)s t=\[([\d.]+),([\d.]+)\]"):
        out.append(("head", m[1], float(m[3]), float(m[4]), {}))
    for m in lg.all(r"^CARD HOLD #(\d+) (\w+): waited ([\d.]+)s · held ([\d.]+)s · t=\[([\d.]+),([\d.]+)\]"):
        out.append(("hold", m[2], float(m[5]), float(m[6]), {"n": int(m[1]), "waited": float(m[3])}))
    for m in lg.all(r"^STAGE (.*?): ([\d.]+)s t=\[([\d.]+),([\d.]+)\]"):
        out.append(("stage", m[1], float(m[3]), float(m[4]), {}))
    return out


def parse_whir_splits(lg):
    """WHIR PROVE SPLIT #n / GLOBAL -> {epoch key: {challenge, argue, open_groups, open_prepared}} (G1's reader)."""
    out = {}
    rx = re.compile(r"^WHIR PROVE SPLIT (#\d+|GLOBAL)[^:]*: .*?inside\[Σ\] challenge ([\d.]+) · argue ([\d.]+)"
                    r".*?open_groups ([\d.]+).*?open_prepared ([\d.]+)")
    for l in lg.L:
        m = rx.search(l)
        if m:
            key = "global" if m[1] == "GLOBAL" else m[1][1:]
            out[key] = dict(challenge=float(m[2]), argue=float(m[3]), open_groups=float(m[4]), open_prepared=float(m[5]))
    return out


def windows_for(lg):
    """Named windows in unix seconds, and what defined them."""
    W = collections.OrderedDict()
    head = lg.first(r"^BASE HEAD \(WHIR\): start t=([\d.]+)")
    l1 = lg.first(r"LEVEL 1 \(wide\) START t=([\d.]+)")
    l1_s = lg.first(r"level 1 \(wide\): \d+ wide nodes over \d+ epochs in ([\d.]+)s")
    total = lg.first(r"WHOLE RUN: host peak .*?, ([\d.]+)s total")
    if None not in (head, l1, l1_s, total):
        W["base"] = (head, l1)
        W["level1"] = (l1, l1 + l1_s)
        W["root"] = (l1 + l1_s, head + total)
        return W, dict(run_start=head, level1_start=l1, level1=l1_s, whole_run=total, base=l1 - head)
    # ⓘ G1's windows, for a log that predates head-ahead and the wide tree (the laptop's smoke test on wt90nsys).
    mark = lg.first(r"MARK AFTER the WHIR base.*t=([\d.]+)")
    base_s = lg.first(r"base \(WHIR\): \d+ epochs in ([\d.]+)s")
    l0_s = lg.first(r"level 0: \d+ WHIR wraps in ([\d.]+)s")
    int_s = lg.first(r"levels 1\.\.=\d+ POOLED: \d+ nodes in ([\d.]+)s")
    if None in (mark, base_s, l0_s, int_s, total):
        raise SystemExit(f"a window line is missing (head {head} level-1 start {l1} level-1 wall {l1_s} total {total}; "
                         f"legacy: mark {mark} base {base_s} l0 {l0_s} int {int_s})")
    run0 = mark - base_s
    W["base"] = (run0, mark)
    W["level0"] = (mark, mark + l0_s)
    W["interior"] = (mark + l0_s, mark + l0_s + int_s)
    W["root"] = (mark + l0_s + int_s, run0 + total)
    return W, dict(run_start=run0, whole_run=total, mark=mark, base=base_s, level0=l0_s, interior=int_s, legacy=1)


def base_subphases(spans, wsplits, base):
    """The base's disjoint sub-phases as {label: [(a, b) unix]}, and the per-scope argue/open windows.

    Scopes: every epoch key (`0`, `1`, …) and `global`. A scope's prove span [a, b] splits into challenge, argue,
    open (= open_groups + open_prepared + its remainder, to b)."""
    a0, b0 = base
    sub = collections.OrderedDict()
    per_scope = collections.OrderedDict()
    prover_first = None
    for kind, lab, a, b, ex in spans:
        if kind != "epoch" or not (a0 <= a < b0):
            continue
        ph = ex["phase"]
        if ph not in ("prep", "absorb", "commit", "prove"):
            continue
        scope = "global" if ex["epoch"] == "global" else "epochs"
        prover_first = a if prover_first is None else min(prover_first, a)
        if ph in ("prep", "absorb"):
            sub.setdefault(f"{scope} absorb+prep", []).append((a, b))
        elif ph == "commit":
            sub.setdefault(f"{scope} commit", []).append((a, b))
            per_scope.setdefault(ex["epoch"], {})["commit"] = (a, b)
        else:
            d = wsplits.get(ex["epoch"])
            if d is None:
                sub.setdefault(f"{scope} prove (no WHIR PROVE SPLIT line)", []).append((a, b))
                continue
            ga = min(a + d["challenge"], b)
            gb = min(ga + d["argue"], b)
            if ga > a:
                sub.setdefault(f"{scope} challenge", []).append((a, ga))
            sub.setdefault(f"{scope} argue", []).append((ga, gb))
            sub.setdefault(f"{scope} open", []).append((gb, b))
            per_scope.setdefault(ex["epoch"], {})["argue"] = (ga, gb)
            per_scope[ex["epoch"]]["open"] = (gb, b)
    if prover_first is None:
        return sub, per_scope, None
    out = collections.OrderedDict()
    out["head"] = [(a0, prover_first)]
    for k in sorted(sub, key=lambda k: (k.split(" ")[0] != "epochs", k)):
        out[k] = sub[k]
    return out, per_scope, prover_first


# ------------------------------------------------------------------ the partition
class Partition:
    """Elementary intervals between consecutive event times (every op edge and every cut).
    For each interval: ak kernels, ac copies/memsets active. Weights:
      kernel op   : dt / ak                (per-op exclusive time; per stage: dt * ak_stage / ak)
      copy/memset : dt / ac when ak == 0   (copy-only time)
      idle        : dt when ak == ac == 0.
    Cumulative sums over the intervals make any [a, b) with a, b event times two lookups."""

    def __init__(self, T, kstage_idx, nst, cuts):
        nK = len(T.K); nC = len(T.M) + len(T.Z)
        cs = np.concatenate([T.M[:, 0], T.Z[:, 0]]); ce = np.concatenate([T.M[:, 1], T.Z[:, 1]])
        cuts = np.asarray(sorted(set(int(c) for c in cuts)), np.int64)
        t = np.concatenate([T.K[:, 0], T.K[:, 1], cs, ce, cuts])
        uniq, inv = np.unique(t, return_inverse=True)
        U = len(uniq)
        dk = np.zeros(U, np.int64); dc = np.zeros(U, np.int64)
        np.add.at(dk, inv[:nK], 1); np.add.at(dk, inv[nK:2 * nK], -1)
        np.add.at(dc, inv[2 * nK:2 * nK + nC], 1); np.add.at(dc, inv[2 * nK + nC:2 * nK + 2 * nC], -1)
        ak = np.cumsum(dk)[:-1]; ac = np.cumsum(dc)[:-1]
        dt = np.diff(uniq).astype(np.float64)
        self.uniq = uniq
        cum = lambda w: np.concatenate([[0.0], np.cumsum(w)])
        self.CWk = cum(np.where(ak > 0, dt / np.maximum(ak, 1), 0.0))
        self.CWc = cum(np.where((ak == 0) & (ac > 0), dt / np.maximum(ac, 1), 0.0))
        self.CWcopy = cum(np.where((ak == 0) & (ac > 0), dt, 0.0))
        self.CWidle = cum(np.where((ak == 0) & (ac == 0), dt, 0.0))
        self.CWbusy = cum(np.where((ak > 0) | (ac > 0), dt, 0.0))
        self.CWstage = []
        for s in range(nst):
            m = kstage_idx == s
            d = np.zeros(U, np.int64)
            np.add.at(d, inv[:nK][m], 1); np.add.at(d, inv[nK:2 * nK][m], -1)
            a_s = np.cumsum(d)[:-1]
            self.CWstage.append(cum(np.where(ak > 0, dt * a_s / np.maximum(ak, 1), 0.0)))
        idle_m = (ak == 0) & (ac == 0) & (dt > 0)
        self.IS = uniq[:-1][idle_m]; self.IE = uniq[1:][idle_m]

    def between(self, CW, s, e):
        i = np.searchsorted(self.uniq, s); j = np.searchsorted(self.uniq, e)
        return CW[j] - CW[i]

    def over(self, CW, S_, E_):
        """Σ over disjoint intervals whose edges are event times."""
        if len(S_) == 0:
            return 0.0
        return float(np.sum(self.between(CW, np.asarray(S_), np.asarray(E_))))


# ------------------------------------------------------------------ idle by mechanism (W1 method)
def api_cat(n):
    if "Synchronize" in n: return "API sync (stream/event/ctx)"
    if re.search(r"MemcpyDtoH|MemcpyAsync$|Memcpy$", n) and "HtoD" not in n: return "API readback DtoH"
    if re.search(r"MemAlloc|MemFree|HostAlloc|HostRegister|HostUnregister|MemPool|FreeHost|MallocHost", n): return "API alloc/free"
    if "MemcpyHtoD" in n: return "API upload HtoD"
    if "Launch" in n: return "API launch"
    if "Memset" in n: return "API memset"
    return "API other"


def os_cat(n):
    if n in ("futex", "pthread_cond_wait", "pthread_cond_timedwait", "sem_wait", "pthread_mutex_lock", "sem_timedwait",
             "pthread_rwlock_wrlock", "pthread_rwlock_rdlock", "pthread_join"):
        return "OS wait on host threads (futex/cond)"
    if n == "ioctl": return "OS ioctl (outside API)"
    if n in ("mmap", "munmap", "mremap", "madvise", "mprotect", "brk", "mmap64"): return "OS mmap/munmap/madvise"
    if n in ("read", "write", "pread64", "pwrite64", "open", "open64", "close", "fopen", "fclose", "openat", "fread",
             "fwrite", "readv", "writev", "fsync", "fdatasync"): return "OS file IO"
    if n in ("poll", "select", "epoll_wait", "sched_yield", "usleep", "nanosleep", "sleep", "ppoll", "pselect6"): return "OS poll/sleep/yield"
    return "OS other"


UNTRACED = "untraced (host compute or descheduled)"


class Mechanism:
    """Each idle gap inside the GPU span is charged to the thread that submitted the op ending it, by
    that thread's state during the gap: inside a CUDA API call (by category), inside an OS call outside
    the API (by category), or neither (untraced: host code, or descheduled). 'late' = the submitting call
    started after the gap began (the host had not yet asked for the next op)."""

    def __init__(self, T, IS, IE):
        self.T = T
        self.rcat = np.array([api_cat(n) for n in T.rname], dtype=object)
        self.ocat = np.array([os_cat(n) for n in T.oname], dtype=object)
        self.cats = sorted(set(self.rcat.tolist())) + sorted(set(self.ocat.tolist())) + [UNTRACED]
        self.cache = {}
        self.gS, self.gE = IS, IE
        order = np.argsort(T.op_s, kind="stable")
        ops_s = T.op_s[order]; ops_c = T.op_c[order]
        nxt_raw = np.searchsorted(ops_s, self.gE)
        nxt = np.minimum(nxt_raw, len(order) - 1)
        ncorr = ops_c[nxt]
        ro = np.argsort(T.R[:, 4]); rc = T.R[ro, 4]
        j = np.minimum(np.searchsorted(rc, ncorr), len(rc) - 1)
        ok = (rc[j] == ncorr) & (nxt_raw < len(order))  # a gap after the last op has no op ending it
        api_row = np.where(ok, ro[j], -1)
        self.sub_tid = np.where(api_row >= 0, T.R[np.maximum(api_row, 0), 2], -1)
        sub_start = np.where(api_row >= 0, T.R[np.maximum(api_row, 0), 0], 0)
        self.late = np.where(api_row >= 0, np.clip(np.minimum(sub_start, self.gE) - self.gS, 0, None), 0)
        self.charge = {c: np.zeros(len(self.gS)) for c in self.cats}
        for tid in np.unique(self.sub_tid):
            k = self.sub_tid == tid
            if tid < 0:
                self.charge[UNTRACED][k] += (self.gE - self.gS)[k]; continue
            fns = self.thread_fns(tid)
            tot = np.zeros(int(k.sum()))
            for c, Fn in fns.items():
                v = Fn(self.gE[k]) - Fn(self.gS[k]); self.charge[c][k] += v; tot += v
            self.charge[UNTRACED][k] += np.maximum((self.gE - self.gS)[k] - tot, 0)

    def thread_fns(self, tid):
        if tid in self.cache: return self.cache[tid]
        T = self.T
        rm = T.R[:, 2] == tid
        fns = {}
        aS, aE = union(T.R[rm, 0], T.R[rm, 1])
        for c in set(self.rcat[rm].tolist()):
            k = rm & (self.rcat == c)
            fns[c] = measure_fn(*union(T.R[k, 0], T.R[k, 1]))
        if len(T.O):
            om = T.O[:, 2] == tid
            for c in set(self.ocat[om].tolist()):
                k = om & (self.ocat == c)
                S_, E_ = subtract(*union(T.O[k, 0], T.O[k, 1]), aS, aE)
                fns[c] = measure_fn(S_, E_)
        self.cache[tid] = fns
        return fns

    def report(self, a, b):
        w = np.maximum(np.minimum(self.gE, b) - np.maximum(self.gS, a), 0).astype(np.float64)
        frac = w / np.maximum(self.gE - self.gS, 1).astype(np.float64)
        late = (np.minimum(self.late, self.gE - self.gS) * frac).sum()
        rows = {c: (self.charge[c] * frac).sum() for c in self.cats}
        th = collections.Counter()
        for t_ in np.unique(self.sub_tid[w > 0]):
            th[t_] = w[self.sub_tid == t_].sum()
        return w.sum(), late, rows, th


def mech_report_over(mech, S_, E_):
    """Mechanism.report over a SET of disjoint intervals (trace ns): idle, 'not yet requested', and per state."""
    w = np.zeros(len(mech.gS))
    for a, b in zip(S_, E_):
        w += np.maximum(np.minimum(mech.gE, b) - np.maximum(mech.gS, a), 0)
    frac = w / np.maximum(mech.gE - mech.gS, 1).astype(np.float64)
    late = (np.minimum(mech.late, mech.gE - mech.gS) * frac).sum()
    rows = {c: (mech.charge[c] * frac).sum() for c in mech.cats}
    return w.sum(), late, rows


# ------------------------------------------------------------------ the host sampler (g6_sampler.py)
class Sampler:
    """Per-thread cumulative on-CPU and runqueue-wait ns of the prover process (`/proc/<pid>/task/*/schedstat`), and
    the box's and the cgroup's CPU counters, every ~0.05 s (g6_sampler.py). A counter's value over [a, b] is
    interpolated between its samples; a thread's is flat before its first sample and after its last, so a thread that
    was not alive contributes nothing."""

    def __init__(self, path):
        per = collections.defaultdict(lambda: ([], [], []))
        self.pid_lines = collections.Counter()
        self.stamps = []
        self.hdr = {}
        names = {}
        C = []
        for line in open(path, encoding="utf-8", errors="replace"):
            if line.startswith("#"):
                for k, v in re.findall(r"(clk_tck|cpus_online|affinity) (-?\d+)", line):
                    self.hdr[k] = int(v)
                m = re.search(r"cpu\.max (.+?) ·", line)
                if m:
                    self.hdr["cpu_max"] = m[1]
                continue
            p = line.rstrip("\n").split("\t")
            if p[0] == "T" and len(p) >= 6:
                t, pid, tid = float(p[1]), int(p[2]), int(p[3])
                self.pid_lines[pid] += 1
                a = per[(pid, tid)]
                a[0].append(t); a[1].append(int(p[4])); a[2].append(int(p[5]))
            elif p[0] == "N" and len(p) >= 4:
                names[(int(p[1]), int(p[2]))] = p[3]
            elif p[0] == "C" and len(p) >= 5:
                self.stamps.append(float(p[1]))
                C.append([float(x) for x in p[1:5]] + [float(p[5]) if len(p) > 5 else -1.0, float(p[6]) if len(p) > 6 else -1.0])
        self.pid = self.pid_lines.most_common(1)[0][0] if self.pid_lines else None
        self.th = {tid: (np.array(ts), np.array(on, np.float64), np.array(w, np.float64))
                   for (pid, tid), (ts, on, w) in per.items() if pid == self.pid}
        self.interval = float(np.median(np.diff(self.stamps))) if len(self.stamps) > 2 else float("nan")
        self.C = np.array(C, np.float64) if C else np.zeros((0, 6))  # t, cgroup ns, busy, total, nr_throttled, throttled ns
        self.names = {tid: n for (pid, tid), n in names.items() if pid == self.pid}

    def test_thread(self, prefix="lfm::"):
        """The test's own thread by the name rule: the lowest tid whose name starts with the test path (libtest names
        the thread after the test; threads it spawns without a name inherit that name, and come later)."""
        c = sorted(t for t, n in self.names.items() if n.startswith(prefix))
        return c[0] if c else None

    def box(self, S_, E_):
        """Over intervals in unix seconds: {box busy CPU-s, box CPU-s (all online CPUs), cgroup CPU-s, throttled s,
        throttled periods}; NaN where a counter was unreadable (-1)."""
        out = dict(busy=float("nan"), total=float("nan"), cgroup=float("nan"), throttled=float("nan"), periods=float("nan"))
        if len(self.C) < 2 or len(S_) == 0:
            return out
        x = np.concatenate([np.asarray(S_, np.float64), np.asarray(E_, np.float64)])
        n = len(S_)
        tck = float(self.hdr.get("clk_tck", 100))

        def d(col, scale):
            v = self.C[:, col]
            if (v < 0).any():
                return float("nan")
            iv = np.interp(x, self.C[:, 0], v)
            return float((iv[n:] - iv[:n]).sum() / scale)
        out.update(busy=d(2, tck), total=d(3, tck), cgroup=d(1, 1e9), throttled=d(5, 1e9), periods=d(4, 1.0))
        return out

    def of(self, tid, S_, E_):
        """(on-CPU s, runqueue-wait s) of one thread over intervals given in unix seconds."""
        if tid not in self.th or len(S_) == 0:
            return 0.0, 0.0
        ts, on, w = self.th[tid]
        x = np.concatenate([np.asarray(S_, np.float64), np.asarray(E_, np.float64)])
        io = np.interp(x, ts, on); iw = np.interp(x, ts, w)
        n = len(S_)
        return float((io[n:] - io[:n]).sum() / 1e9), float((iw[n:] - iw[:n]).sum() / 1e9)

    def all_of(self, S_, E_):
        on = wt = 0.0
        for tid in self.th:
            a, b = self.of(tid, S_, E_); on += a; wt += b
        return on, wt

    def top(self, S_, E_, k=5, exclude=()):
        rows = [(tid, *self.of(tid, S_, E_)) for tid in self.th if tid not in exclude]
        return sorted(rows, key=lambda r: -r[1])[:k]


def ivs(pairs):
    """Sorted disjoint (S, E) float arrays from (a, b) pairs."""
    if not pairs:
        return np.zeros(0), np.zeros(0)
    a = np.array([p[0] for p in pairs], np.float64); b = np.array([p[1] for p in pairs], np.float64)
    o = np.argsort(a, kind="stable"); a = a[o]; b = b[o]
    emax = np.maximum.accumulate(b)
    new = np.ones(len(a), bool); new[1:] = a[1:] > emax[:-1]
    idx = np.flatnonzero(new)
    return a[idx], np.append(emax[idx[1:] - 1], emax[-1])


def ivs_minus(S1, E1, S2, E2):
    """(S1, E1) minus (S2, E2); both sorted and disjoint (float)."""
    out = []
    j = 0
    for s, e in zip(S1.tolist(), E1.tolist()):
        cur = s
        while j < len(S2) and E2[j] <= cur:
            j += 1
        k = j
        while k < len(S2) and S2[k] < e:
            if S2[k] > cur:
                out.append((cur, S2[k]))
            cur = max(cur, E2[k])
            k += 1
        if cur < e:
            out.append((cur, e))
    return ivs(out)


def ivs_overlap(S1, E1, S2, E2):
    """Σ length of (S1,E1) ∩ (S2,E2) (float seconds)."""
    tot = 0.0; i = j = 0
    while i < len(S1) and j < len(S2):
        s = max(S1[i], S2[j]); e = min(E1[i], E2[j])
        if e > s:
            tot += e - s
        if E1[i] < E2[j]:
            i += 1
        else:
            j += 1
    return tot


def disjoint_subphases(sub, base, prover_first):
    """Make the labelled sub-phases disjoint (earlier labels win) and add `prover gaps` = the rest of the base after
    the head. {label: (S, E)} in unix seconds."""
    out = collections.OrderedDict()
    taken_S, taken_E = np.zeros(0), np.zeros(0)
    for lab, pairs in sub.items():
        S_, E_ = ivs([(max(a, base[0]), min(b, base[1])) for a, b in pairs if min(b, base[1]) > max(a, base[0])])
        S_, E_ = ivs_minus(S_, E_, taken_S, taken_E)
        out[lab] = (S_, E_)
        taken_S, taken_E = ivs(list(zip(taken_S.tolist(), taken_E.tolist())) + list(zip(S_.tolist(), E_.tolist())))
    if prover_first is not None:
        gS, gE = ivs_minus(np.array([prover_first]), np.array([base[1]]), taken_S, taken_E)
        out["prover gaps"] = (gS, gE)
    return out


def host_work_spans(spans, base):
    """Host work the log names inside the base, as {label: (S, E)} unix: the producer's CPU stages (execute, collect,
    build, prep@producer), its hand-off wait, the head's helper, and the level-1 prologues in the base's tail.

    ⓘ A `handoff` span runs from the end of `build` to the hand-off and CONTAINS that epoch's prep@producer (both start
    at build's end, BASE PREP k and BASE EPOCH k: handoff); the wait is the handoff span minus the prep spans."""
    g = collections.defaultdict(list)
    handoff = []
    for kind, lab, a, b, ex in spans:
        if b <= base[0] or a >= base[1]:
            continue
        if kind == "epoch" and ex["phase"] in ("execute", "collect", "build"):
            g[f"producer {ex['phase']}"].append((a, b))
        elif kind == "epoch" and ex["phase"] == "handoff":
            handoff.append((a, b))
        elif kind == "baseprep":
            g["producer prep@producer"].append((a, b))
        elif kind == "head":
            g[f"head helper: {lab}"].append((a, b))
        elif kind == "stage" and "prologue" in lab:
            g["level-1 prologues in the base's tail"].append((a, b))
    out = collections.OrderedDict()
    for k in sorted(g):
        out[k] = ivs(g[k])
    if handoff:
        out["producer hand-off wait (handoff minus prep)"] = ivs_minus(*ivs(handoff), *ivs(g.get("producer prep@producer", [])))
    work = [p for k, v in g.items() for p in v if k.startswith("producer ")]
    out["producer CPU stages (execute+collect+build+prep)"] = ivs(work)
    return out


def producer_rows(spans, base, say, fout):
    """The producer's own stages per epoch, from the log: execute, collect, build, prep@producer, and the hand-off wait
    (the handoff span minus prep@producer). The producer is one thread; its stages are sequential."""
    per = collections.OrderedDict()
    for kind, lab, a, b, ex in spans:
        if b <= base[0] or a >= base[1]:
            continue
        if kind == "epoch" and ex["phase"] in ("execute", "collect", "build", "handoff"):
            per.setdefault(ex["epoch"], collections.defaultdict(list))[ex["phase"]].append((a, b))
        elif kind == "baseprep":
            per.setdefault(ex["epoch"], collections.defaultdict(list))["prep@producer"].append((a, b))
    cols = ("execute", "collect", "build", "prep@producer", "wait")
    print("epoch\t" + "\t".join(cols), file=fout)
    tot = collections.Counter()
    for e, d in per.items():
        v = {k: sum(b - a for a, b in d.get(k, [])) for k in cols[:4]}
        hS, hE = ivs(d.get("handoff", []))
        pS, pE = ivs(d.get("prep@producer", []))
        wS, wE = ivs_minus(hS, hE, pS, pE)
        v["wait"] = float((wE - wS).sum())
        print(f"{e}\t" + "\t".join(f"{v[k]:.4f}" for k in cols), file=fout)
        tot.update(v)
    say(f"  producer, {len(per)} epoch(s)/stage(s): " + " · ".join(f"{k} Σ {tot[k]:.2f} s" for k in cols[:4])
        + f" · work Σ {sum(tot[k] for k in cols[:4]):.2f} s · hand-off wait Σ {tot['wait']:.2f} s")
    return tot


def contention_rows(S, sub, host_spans, prover_tid, say, fout):
    """Host CPU during every sub-phase, from the sampler."""
    prod = host_spans.get("producer execute", (np.zeros(0), np.zeros(0)))
    producer_tid = None
    if S is not None and len(prod[0]):
        tops = S.top(*prod, k=3, exclude=(prover_tid,) if prover_tid is not None else ())
        producer_tid = tops[0][0] if tops else None
    say(f"  sampler: pid {S.pid if S else '-'} · {len(S.th) if S else 0} threads · interval {S.interval if S else float('nan'):.3f} s · "
        f"prover thread {prover_tid if prover_tid is not None else 'n/a (no trace)'} · producer thread {producer_tid}")
    if S:
        say(f"  box: cpus online {S.hdr.get('cpus_online', '?')} · the sampler's affinity {S.hdr.get('affinity', '?')} · "
            f"cgroup cpu.max {S.hdr.get('cpu_max', '?')} · clock ticks {S.hdr.get('clk_tck', '?')}/s")
    print("subphase\twall_s\tprocess_on_s\tprocess_cores\tprocess_wait_s\tprover_on_s\tprover_wait_s\tproducer_on_s\t"
          "producer_wait_s\tothers_on_s\tbox_busy_cores\tbox_cores\tcgroup_cores\tthrottled_s\tthrottled_periods", file=fout)
    for lab, (S_, E_) in sub.items():
        wall = float((E_ - S_).sum())
        if wall <= 0:
            continue
        on, wt = S.all_of(S_, E_) if S else (0.0, 0.0)
        pon, pwt = S.of(prover_tid, S_, E_) if S and prover_tid is not None else (0.0, 0.0)
        qon, qwt = S.of(producer_tid, S_, E_) if S and producer_tid is not None else (0.0, 0.0)
        nan = float("nan")
        bx = S.box(S_, E_) if S else dict(busy=nan, total=nan, cgroup=nan, throttled=nan, periods=nan)
        print(f"{lab}\t{wall:.4f}\t{on:.4f}\t{on / wall:.2f}\t{wt:.4f}\t{pon:.4f}\t{pwt:.4f}\t{qon:.4f}\t{qwt:.4f}\t"
              f"{on - pon - qon:.4f}\t{bx['busy'] / wall:.2f}\t{bx['total'] / wall:.2f}\t{bx['cgroup'] / wall:.2f}\t"
              f"{bx['throttled']:.4f}\t{bx['periods']:.0f}", file=fout)
        say(f"  {lab:34s} wall {wall:7.3f} · process on-CPU {on:7.2f} s ({on / wall:5.2f} cores) · runqueue wait {wt:6.2f} s · "
            f"prover on {pon:6.2f} wait {pwt:5.3f} · producer on {qon:6.2f} wait {qwt:5.3f} · box busy "
            f"{bx['busy'] / wall:5.2f} of {bx['total'] / wall:5.1f} cores · cgroup {bx['cgroup'] / wall:5.2f} cores · "
            f"throttled {bx['throttled']:.3f} s")
        if S and prover_tid is None and on > 0.05:
            # no trace to name the prover thread: the busiest threads, anonymous
            say(f"  {'':34s} busiest threads (tid on/wait s): " + " · ".join(
                f"{tid} {a:.2f}/{b:.3f}" for tid, a, b in S.top(S_, E_, k=4)))
    return producer_tid


def overlap_rows(sub, host_spans, say, fout):
    """How much named host work runs inside every sub-phase (overlap seconds; a stage can overlap several)."""
    print("subphase\thost_work\toverlap_s", file=fout)
    for lab, (S_, E_) in sub.items():
        if float((E_ - S_).sum()) <= 0:
            continue
        parts = []
        for hl, (HS, HE) in host_spans.items():
            o = ivs_overlap(S_, E_, HS, HE)
            if o > 0:
                print(f"{lab}\t{hl}\t{o:.4f}", file=fout)
                if o >= 0.005:
                    parts.append(f"{hl} {o:.2f}")
        if parts:
            say(f"  {lab:34s} " + " · ".join(parts))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db")
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sampler")
    ap.add_argument("--no-mechanism", action="store_true")
    ap.add_argument("--log-only", action="store_true")
    A = ap.parse_args()
    if not A.log_only and not A.db:
        raise SystemExit("--db is required unless --log-only")
    os.makedirs(A.out, exist_ok=True)
    lg = Log(A.log)
    W, wdef = windows_for(lg)
    spans = parse_spans(lg)
    wsplits = parse_whir_splits(lg)
    S = Sampler(A.sampler) if A.sampler and os.path.exists(A.sampler) else None
    fopen = lambda name: open(os.path.join(A.out, name), "w")
    summ = fopen("summary.txt")

    def say(*x):
        print(*x); print(*x, file=summ)

    say(f"G6 analysis · log {os.path.basename(A.log)} · " + ("LOG ONLY (no trace)" if A.log_only else f"db {os.path.basename(A.db)}")
        + (f" · sampler {os.path.basename(A.sampler)}" if S else " · no sampler"))
    say("windows: " + " · ".join(f"{k} {b - a:.3f} s" for k, (a, b) in W.items()))
    say("  defined by: " + ", ".join(f"{k}={v:.3f}" for k, v in wdef.items()))
    base = W["base"]
    sub0, per_scope, prover_first = base_subphases(spans, wsplits, base)
    sub = disjoint_subphases(sub0, base, prover_first)
    hs = host_work_spans(spans, base)
    covered = sum(float((E_ - S_).sum()) for S_, E_ in sub.values())
    say(f"base sub-phases: {len(sub)} · Σ {covered:.3f} s of the base's {base[1] - base[0]:.3f} s "
        + ("· PARTITION OK" if abs(covered - (base[1] - base[0])) < 0.002 else "· ⚠ NOT A PARTITION"))
    say("  " + " · ".join(f"{k} {float((E_ - S_).sum()):.3f}" for k, (S_, E_) in sub.items()))

    if A.log_only:
        say("\n=== HOST CPU PER BASE SUB-PHASE (sampler) ===")
        if S:
            rule = S.test_thread()
            say(f"  prover thread = the test's thread by the name rule (checked against the CUDA API in the traced arm): "
                f"tid {rule} ({S.names.get(rule, '?')})")
            contention_rows(S, sub, hs, rule, say, fopen("contention.tsv"))
        else:
            say("  no sampler file")
        say("\n=== NAMED HOST WORK INSIDE EACH SUB-PHASE (log spans, overlap seconds) ===")
        overlap_rows(sub, hs, say, fopen("overlap.tsv"))
        producer_rows(spans, base, say, fopen("producer.tsv"))
        ep = fopen("epochs_host.tsv")
        print("scope\tphase\twall_s\tprocess_cores\tprocess_wait_s\tproducer_work_overlap_s\tprologue_overlap_s", file=ep)
        prod = hs.get("producer CPU stages (execute+collect+build+prep)", (np.zeros(0), np.zeros(0)))
        prol = hs.get("level-1 prologues in the base's tail", (np.zeros(0), np.zeros(0)))
        for scope, d in per_scope.items():
            for ph in ("commit", "argue", "open"):
                if ph not in d or d[ph][1] <= d[ph][0]:
                    continue
                a, b = d[ph]
                on, wt = S.all_of([a], [b]) if S else (float("nan"), float("nan"))
                print(f"{scope}\t{ph}\t{b - a:.4f}\t{on / (b - a):.2f}\t{wt:.4f}\t"
                      f"{ivs_overlap(np.array([a]), np.array([b]), *prod):.4f}\t{ivs_overlap(np.array([a]), np.array([b]), *prol):.4f}", file=ep)
        summ.close()
        return

    T = Trace(A.db)
    say(f"kernels {len(T.K)} · copies {len(T.M)} · memsets {len(T.Z)} · API calls {len(T.R)} · OSRT calls {len(T.O)} · "
        f"prover pid {T.PID}" + (f" · GPU {T.gpu[0]} ({T.gpu[1]} SMs)" if T.gpu else ""))
    first_op, last_op = int(T.op_s.min()), int(T.op_e.max())
    Wns = collections.OrderedDict((k, (T.ns(a), T.ns(b))) for k, (a, b) in W.items())
    subns = collections.OrderedDict((k, (np.array([T.ns(x) for x in S_], np.int64), np.array([T.ns(x) for x in E_], np.int64)))
                                    for k, (S_, E_) in sub.items())
    say(f"  GPU span: first op {sec(first_op - Wns['base'][0]):+.3f} s from the base's start · last op "
        f"{sec(last_op - list(Wns.values())[-1][1]):+.3f} s from the last window's end")

    # ---------------- stage per kernel; sumcheck-family by argue/open window (epochs and global alike)
    cls = {n: classify(n) for n in set(T.kname.tolist())}
    kstage = np.array([cls[n][0] for n in T.kname], dtype=object)
    kfam = np.array([cls[n][1] for n in T.kname], dtype=object)
    arg = [subns[k] for k in subns if k.endswith(" argue")]
    opn = [subns[k] for k in subns if k.endswith(" open")]
    argS, argE = union(np.concatenate([x[0] for x in arg]) if arg else [], np.concatenate([x[1] for x in arg]) if arg else [])
    opnS, opnE = union(np.concatenate([x[0] for x in opn]) if opn else [], np.concatenate([x[1] for x in opn]) if opn else [])
    mid = (T.K[:, 0] + T.K[:, 1]) // 2

    def inside(S_, E_):
        if len(S_) == 0:
            return np.zeros(len(mid), bool)
        j = np.searchsorted(S_, mid, side="right") - 1
        return (j >= 0) & (mid < E_[np.maximum(j, 0)])
    in_arg = inside(argS, argE); in_opn = inside(opnS, opnE)
    amb = np.isin(kfam, ["sumcheck family", "1 LogUp aux trace", "0 arith helpers"])
    kstage[amb & in_arg] = "4"
    kfam[amb & in_arg & (kfam != "0 arith helpers")] = "4 zerocheck/GKR/LogUp (argue)"
    kfam[amb & in_arg & (kfam == "0 arith helpers")] = "4 arith helpers (argue)"
    m5 = amb & in_opn & ~in_arg
    kstage[m5] = "5"
    kfam[m5 & (kfam != "0 arith helpers")] = "5 WHIR open sumcheck"
    kfam[m5 & (kfam == "0 arith helpers")] = "5 arith helpers (open)"
    left = kfam == "sumcheck family"
    kstage[left] = "0"; kfam[left] = "0 sumcheck family outside argue/open windows"
    kidx = np.array([STAGES.index(s) for s in kstage], np.int64)
    say(f"base: {len(argS)} argue windows (Σ {sec((argE - argS).sum()):.2f} s) · {len(opnS)} open windows (Σ {sec((opnE - opnS).sum()):.2f} s)")

    # ---------------- the partition, with every window and sub-phase edge as a cut
    cuts = [v for ab in Wns.values() for v in ab]
    for S_, E_ in subns.values():
        cuts += S_.tolist() + E_.tolist()
    for d in per_scope.values():
        for a, b in d.values():
            cuts += [T.ns(a), T.ns(b)]
    holds = [(T.ns(sp[2]), T.ns(sp[3])) for sp in spans if sp[0] == "hold"]
    for h in holds:
        cuts += list(h)
    P = Partition(T, kidx, len(STAGES), cuts)
    ks, ke = T.K[:, 0], T.K[:, 1]
    cs = np.concatenate([T.M[:, 0], T.Z[:, 0]]); ce = np.concatenate([T.M[:, 1], T.Z[:, 1]])
    cbytes = np.concatenate([T.M[:, 4], T.Z[:, 4]])
    nM = len(T.M)

    def copy_label(k, s_, d_):
        if k == 1: return f"7 H2D (from {T.memkind.get(s_, s_)})"
        if k == 2: return f"7 D2H (to {T.memkind.get(d_, d_)})"
        if k == 8: return "7 D2D"
        return f"7 copy kind {k}"
    clab = np.empty(len(cs), dtype=object)
    clab[nM:] = "7 memset"
    if nM:
        combos, inv_c = np.unique(T.M[:, 5:8], axis=0, return_inverse=True)
        labs = np.array([copy_label(int(k), int(s_), int(d_)) for k, s_, d_ in combos], dtype=object)
        clab[:nM] = labs[inv_c.ravel()]
    thr = (T.K[:, 5] * T.K[:, 6] * T.K[:, 7] * T.K[:, 8] * T.K[:, 9] * T.K[:, 10]).astype(np.int64)
    fams = sorted(set(kfam.tolist()), key=lambda f: (f.split(" ")[0], f))
    clabs = sorted(set(clab.tolist()))

    def rows_over(S_, E_):
        """Exclusive seconds per family and per copy label over a set of intervals (trace ns), plus Σ-kernel/copy
        seconds, op counts and threads/bytes; and the stage, copy-only and idle totals."""
        rows = collections.OrderedDict()
        ek = np.zeros(len(ks)); ec = np.zeros(len(cs)); sk = np.zeros(len(ks)); sc = np.zeros(len(cs))
        for a, b in zip(S_.tolist(), E_.tolist()):
            ek += P.between(P.CWk, np.clip(ks, a, b), np.clip(ke, a, b))
            ec += P.between(P.CWc, np.clip(cs, a, b), np.clip(ce, a, b))
            sk += np.maximum(np.minimum(ke, b) - np.maximum(ks, a), 0)
            sc += np.maximum(np.minimum(ce, b) - np.maximum(cs, a), 0)
        inw = sk > 0; inc = sc > 0
        for f in fams:
            m = inw & (kfam == f)
            if m.any():
                rows[f] = (ek[m].sum(), sk[m].sum(), int(m.sum()), int(thr[m].sum()))
        for lab in clabs:
            m = inc & (clab == lab)
            if m.any():
                rows[lab] = (ec[m].sum(), sc[m].sum(), int(m.sum()), int(cbytes[m].sum()))
        idle = P.over(P.CWidle, S_, E_)
        rows["8 idle card"] = (idle, idle, 0, 0)
        st = {s: P.over(P.CWstage[i], S_, E_) for i, s in enumerate(STAGES)}
        return rows, st, P.over(P.CWcopy, S_, E_), idle, P.over(P.CWbusy, S_, E_), inw

    # ---------------- per window
    ledger = fopen("ledger.tsv")
    print("window\tstage\tfamily\texcl_s\tsum_s\tops\tthreads_or_bytes", file=ledger)
    kt = fopen("kernels.tsv")
    print("window\tkernel\tstage\tfamily\tlaunches\tsum_s\tthreads", file=kt)
    closure_bad = []
    totals = collections.OrderedDict()
    for wn, (a, b) in Wns.items():
        rows, st, co, idle, busy, inw = rows_over(np.array([a]), np.array([b]))
        wall = b - a
        if abs(sum(v[0] for v in rows.values()) - wall) > 2e6 or abs(sum(st.values()) + co + idle - wall) > 2e6:
            closure_bad.append((wn, round(sec(sum(v[0] for v in rows.values())), 4), round(sec(wall), 4)))
        totals[wn] = (wall, rows, st, co, idle, busy)
        for f, (x, s_, n, t_) in rows.items():
            print(f"{wn}\t{f.split(' ')[0]}\t{f}\t{sec(x):.4f}\t{sec(s_):.4f}\t{n}\t{t_}", file=ledger)
        sk = np.maximum(np.minimum(ke, b) - np.maximum(ks, a), 0)
        keyset = collections.defaultdict(list)
        for i in np.flatnonzero(inw):
            keyset[(T.kname[i], kfam[i])].append(i)
        for (n, f), idx in sorted(keyset.items(), key=lambda kv: -sk[kv[1]].sum()):
            idx = np.array(idx)
            print(f"{wn}\t{n}\t{f.split(' ')[0]}\t{f}\t{len(idx)}\t{sec(sk[idx].sum()):.4f}\t{int(thr[idx].sum())}", file=kt)
    say("\n=== PARTITION OF EACH WINDOW (exclusive s: kernels first, then copy/memset-only, then idle) ===")
    say("window | wall | " + " | ".join(STAGES + ["7", "8"]) + " | busy | Σ | closure")
    for wn, (wall, rows, st, co, idle, busy) in totals.items():
        vals = [st[s] for s in STAGES] + [co, idle]
        say(f"{wn} | {sec(wall):.3f} | " + " | ".join(f"{sec(v):.3f}" for v in vals) + f" | {sec(busy):.3f} | {sec(sum(vals)):.3f} | "
            + ("ok" if abs(sum(vals) - wall) <= 2e6 else "MISS"))
    say("key: " + " · ".join(f"{k}={v}" for k, v in STAGE_NAMES.items()))
    say("\nper family (exclusive s / Σ-kernel-or-copy s / ops):")
    for wn, (wall, rows, st, co, idle, busy) in totals.items():
        say(f"  [{wn}]")
        for f, v in rows.items():
            say(f"     {f:52s} {sec(v[0]):8.3f} / {sec(v[1]):8.3f} / {v[2]}")

    # ---------------- the base's sub-phases
    say("\n=== BASE SUB-PHASES (disjoint; exclusive s) ===")
    sp = fopen("subphases.tsv")
    print("subphase\tn\twall_s\tbusy_s\tidle_s\tcopy_only_s\t" + "\t".join(f"stage{s}" for s in STAGES) + "\tcopy_labels", file=sp)
    subrows = collections.OrderedDict()
    for lab, (S_, E_) in subns.items():
        wall = int((E_ - S_).sum())
        if wall <= 0:
            continue
        rows, st, co, idle, busy, _ = rows_over(S_, E_)
        subrows[lab] = (wall, rows, st, co, idle, busy)
        cl = " · ".join(f"{k[2:]} {sec(v[0]):.3f}s/{v[3] / 1e9:.2f}GB" for k, v in rows.items() if k.startswith("7 "))
        print(f"{lab}\t{len(S_)}\t{sec(wall):.4f}\t{sec(busy):.4f}\t{sec(idle):.4f}\t{sec(co):.4f}\t"
              + "\t".join(f"{sec(st[s]):.4f}" for s in STAGES) + f"\t{cl}", file=sp)
        say(f"  {lab:34s} n={len(S_):3d} wall {sec(wall):7.3f} · busy {sec(busy):7.3f} · idle {sec(idle):6.3f} · copy-only {sec(co):6.3f} · "
            + " ".join(f"s{s}={sec(st[s]):.2f}" for s in STAGES if st[s] > 5e6))
        if cl:
            say(f"  {'':34s} copies: {cl}")
    tw = sum(v[0] for v in subrows.values()); ti = sum(v[4] for v in subrows.values())
    say(f"  Σ sub-phases {sec(tw):.3f} s (base {sec(Wns['base'][1] - Wns['base'][0]):.3f}) · idle Σ {sec(ti):.3f} s "
        f"(base window's idle {sec(totals['base'][4]):.3f})")

    # ---------------- per scope (epoch k, global)
    ep = fopen("epochs.tsv")
    print("scope\tphase\twall_s\tbusy_s\tidle_s\tcopy_only_s\tstage4_s\tprocess_cores\tproducer_work_overlap_s\tprologue_overlap_s", file=ep)
    prod = hs.get("producer CPU stages (execute+collect+build+prep)", (np.zeros(0), np.zeros(0)))
    prol = hs.get("level-1 prologues in the base's tail", (np.zeros(0), np.zeros(0)))
    for scope, d in per_scope.items():
        for ph in ("commit", "argue", "open"):
            if ph not in d:
                continue
            a, b = d[ph]
            Sa, Ea = np.array([T.ns(a)]), np.array([T.ns(b)])
            rows, st, co, idle, busy, _ = rows_over(Sa, Ea)
            cores = (S.all_of([a], [b])[0] / (b - a)) if S and b > a else float("nan")
            print(f"{scope}\t{ph}\t{b - a:.4f}\t{sec(busy):.4f}\t{sec(idle):.4f}\t{sec(co):.4f}\t{sec(st['4']):.4f}\t{cores:.2f}\t"
                  f"{ivs_overlap(np.array([a]), np.array([b]), *prod):.4f}\t{ivs_overlap(np.array([a]), np.array([b]), *prol):.4f}", file=ep)

    # ---------------- idle by mechanism, per window and per sub-phase
    prover_tid = None
    if not A.no_mechanism and len(T.R):
        say("\n=== IDLE BY MECHANISM (W1: the thread submitting the op that ends each gap, and its state) ===")
        mech = Mechanism(T, *union(P.IS, P.IE))
        mt = fopen("idle_mechanism.tsv")
        print("scope\tidle_in_gpu_span_s\tnot_yet_requested_s\tstate\tseconds", file=mt)
        for wn, (a, b) in Wns.items():
            tot, late, rows, th = mech.report(a, b)
            say(f"  [{wn}] idle {sec(tot):.3f} s · next op not yet requested {sec(late):.3f} s · "
                + ", ".join(f"{c} {sec(v):.3f}" for c, v in sorted(rows.items(), key=lambda kv: -kv[1]) if v > 1e7))
            for c, v in rows.items():
                print(f"{wn}\t{sec(tot):.4f}\t{sec(late):.4f}\t{c}\t{sec(v):.4f}", file=mt)
        for lab, (S_, E_) in subns.items():
            if int((E_ - S_).sum()) <= 0:
                continue
            tot, late, rows = mech_report_over(mech, S_, E_)
            if tot <= 0:
                continue
            say(f"  [base: {lab}] idle {sec(tot):.3f} s · not yet requested {sec(late):.3f} s · "
                + ", ".join(f"{c} {sec(v):.3f}" for c, v in sorted(rows.items(), key=lambda kv: -kv[1]) if v > 5e6))
            for c, v in rows.items():
                print(f"base: {lab}\t{sec(tot):.4f}\t{sec(late):.4f}\t{c}\t{sec(v):.4f}", file=mt)
    # the prover thread: the most CUDA API time inside the argue windows
    if len(T.R) and len(argS):
        j = np.searchsorted(argS, T.R[:, 0], side="right") - 1
        ina = (j >= 0) & (T.R[:, 0] < argE[np.maximum(j, 0)])
        if ina.any():
            c = collections.Counter()
            for g_, d_ in zip(T.R[ina, 2].tolist(), (T.R[ina, 1] - T.R[ina, 0]).tolist()):
                c[g_] += d_
            g_, d_ = c.most_common(1)[0]
            prover_tid = g_ & 0xFFFFFF
            say(f"prover thread: tid {prover_tid} ({T.tnames.get(g_, '?')}) · {sec(d_):.3f} s of CUDA API time inside the argue windows "
                f"of {sec(sum(c.values())):.3f} s over {len(c)} thread(s)")

    # ---------------- transfers per sub-phase (all bytes, not only exclusive time)
    say("\n=== H2D / D2H / D2D / memset PER SUB-PHASE (bytes; copy-only exclusive s) ===")
    tr = fopen("transfers.tsv")
    print("subphase\tlabel\tcount\tbytes\texcl_s\tsum_s", file=tr)
    for lab, (wall, rows, st, co, idle, busy) in subrows.items():
        parts = []
        for k, (x, s_, n, by) in rows.items():
            if k.startswith("7 "):
                print(f"{lab}\t{k}\t{n}\t{by}\t{sec(x):.4f}\t{sec(s_):.4f}", file=tr)
                if by > 1e8 or x > 5e6:
                    parts.append(f"{k[2:]} {by / 1e9:.2f} GB ({sec(x):.3f} s copy-only)")
        if parts:
            say(f"  {lab:34s} " + " · ".join(parts))

    # ---------------- host contention (sampler) and named host work (log)
    say("\n=== HOST CPU PER BASE SUB-PHASE (sampler; on-CPU and runqueue wait from /proc/<pid>/task/*/schedstat) ===")
    if S:
        rule = S.test_thread()
        say(f"  name rule: the test's thread is tid {rule} ({S.names.get(rule, '?')}); the CUDA API names tid {prover_tid}: "
            + ("AGREE — the plain arms may use the rule" if rule is not None and rule == prover_tid else "DISAGREE — the plain arms' prover-thread column is not the prover"))
        contention_rows(S, sub, hs, prover_tid, say, fopen("contention.tsv"))
    else:
        say("  no sampler file")
    say("\n=== NAMED HOST WORK INSIDE EACH SUB-PHASE (log spans, overlap seconds) ===")
    overlap_rows(sub, hs, say, fopen("overlap.tsv"))
    producer_rows(spans, base, say, fopen("producer.tsv"))

    # ---------------- names and residual
    nm = fopen("names.tsv")
    print("kernel\trule_stage\trule_family\tfamilies_assigned\tlaunches\tsum_s", file=nm)
    unm = []
    for n in sorted(set(T.kname.tolist())):
        m = T.kname == n
        seen = sorted(set(kfam[m].tolist()))
        print(f"{n}\t{cls[n][0]}\t{cls[n][1]}\t{'; '.join(seen)}\t{int(m.sum())}\t{sec((ke[m] - ks[m]).sum()):.4f}", file=nm)
        for f in seen:
            if f.startswith("0 "):
                mm = m & (kfam == f)
                unm.append((n, f, int(mm.sum()), sec((ke[mm] - ks[mm]).sum())))
    say(f"\nkernel names in the trace: {len(set(T.kname.tolist()))} (names.tsv) · on the residual row 0: {len(unm)} "
        f"(Σ {sum(x[3] for x in unm):.3f} s of kernel time)")
    for n, f, c, s_ in unm:
        say(f"  RESIDUAL {n} [{f}] launches {c} · Σ {s_:.3f} s")

    # ---------------- checks
    say("\n=== CHECKS ===")
    say(f"closure: every window's partition sums to its wall within 2 ms: {'PASS' if not closure_bad else 'FAIL ' + str(closure_bad)}")
    say(f"base sub-phases cover the base: {'PASS' if abs(tw - (Wns['base'][1] - Wns['base'][0])) <= 2e6 else 'FAIL'} "
        f"({sec(tw):.3f} of {sec(Wns['base'][1] - Wns['base'][0]):.3f} s)")
    HS, HE = union([h[0] for h in holds], [h[1] for h in holds])
    if len(HS):
        tot = inside_ = 0
        for wn, (a, b) in Wns.items():
            if wn == "base":
                continue
            m = (ke > a) & (ks < b)
            d = np.minimum(ke[m], b) - np.maximum(ks[m], a)
            j = np.searchsorted(HS, ks[m], side="right") - 1
            ins = (j >= 0) & (ks[m] < HE[np.maximum(j, 0)])
            tot += d.sum(); inside_ += d[ins].sum()
        if tot:
            say(f"clock: tree-window kernel time starting inside a CARD HOLD: {100 * inside_ / tot:.2f} % (expect >= 95 %)")
    lh = [s_ for s_ in spans if s_[0] == "hold" and s_[1] == "multi_prove"]
    if lh:
        last_h = max(lh, key=lambda s_: s_[3])
        say(f"clock: last GPU op ends {sec(last_op - T.ns(last_h[3])):+.3f} s from the last multi_prove hold's end (expect |x| < 0.05)")
    say("nsys PROFILER_OVERHEAD: " + (" · ".join(f"{n} n={c} Σ {sec(d):.3f} s" for n, c, d in T.overhead) if T.overhead else "table absent or empty"))
    summ.close()


if __name__ == "__main__":
    main()
