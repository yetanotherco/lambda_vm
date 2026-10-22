#!/usr/bin/env python3
"""The block profile's read-out (block_profile.sh). Text in, text out.

    profile_read.py run <run dir>    one run: the per-run tables (run_read.run: stage.json, win.tsv, fam.tsv, top.tsv,
                                     gaps.tsv, rate.tsv, run.md), plus holds.tsv (the card-permit holds the prover
                                     printed under LFM_CARD_TRACE=1) and idle.tsv (nvidia-smi idle, power and SM clock
                                     per stage window)
    profile_read.py tag <tag dir>    the rows over the tag's runs, and ONE line `VERDICT: BLOCK PROFILE …`

Run directories are NN-<M|P><w|c|n> (median or large; warm-up, control, capture). The tag directory holds expect.json,
written before the first run: {"mode": "capture"|"nocapture", "why": str, "shape": {"M": [tables, groups, cells_G],
"P": [...]}, "reference": {"M": whole, "P": whole} or null}, and hygiene.txt (ok / FAIL) written after the runs.
Rows V1-V4 gate (exit 11 when one is OUT; V3 only when a reference bench of the same machine is given); E1-E5 are
expectations fixed before the run: each prints MET, MISSED or UNREAD with its figures, and none changes the verdict.
"""
import calendar
import csv
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_read as AW  # noqa: E402

V2_BAND = (0.99, 1.04)
V3_TOL = {"M": 1.5, "P": 3.0}
HOLD = re.compile(r"^CARD HOLD #(\d+) (\S+): waited ([\d.]+)s · held ([\d.]+)s · t=\[([\d.]+),([\d.]+)\](.*)$")
# E2's kernel groups, by the names in nsys's cuda_gpu_kern_sum (whole run, each kernel by its own duration)
LEAF = ("p1s_leaves_base_coset",)
EXT = ("ntt_cm_dit_k7", "ntt_cm_dit_k8", "mobius_tile", "mobius_low_levels")
GKR = ("gkr_round_gruen",)
LEAN = ("whir_lean_round", "whir_lean_materialize")
TILE_PREFIX = "whir_tile_"
# The windows E1 reads (run_read's names, from the run's own stamps)
E1_WIN = [("A1 before the first commit", "head", 90.0, 100.0), ("A2 commits", "commits", 0.0, 12.0),
          ("B proving phase", "proving phase", 7.0, 16.0), ("L0", "level 0", 22.0, 38.0), ("WHOLE", "whole", 12.0, 19.0)]
E3_WIN = [("A windows", "walk", 6.0, 10.0), ("A finish", "finish", 8.0, 14.0), ("L0", "level 0", 12.0, 18.0)]


def rd(p, default=""):
    try:
        with open(p, errors="replace") as f:
            return f.read()
    except OSError:
        return default


def smi_rows(run_dir):
    """[(unix t, util %, SM MHz, power W)] from smi.csv (machine-local timestamps; tz.txt holds the offset)."""
    tz = rd(f"{run_dir}/tz.txt").strip() or "+0000"
    off = (1 if tz[0] == "+" else -1) * (int(tz[1:3]) * 3600 + int(tz[3:5]) * 60)
    out = []
    for line in rd(f"{run_dir}/smi.csv").splitlines():
        q = [x.strip() for x in line.split(",")]
        m = re.match(r"(\d+)/(\d+)/(\d+) (\d+):(\d+):(\d+)\.(\d+)", q[0]) if q else None
        if not m or len(q) < 5:
            continue
        y, mo, d, h, mi, s = (int(x) for x in m.groups()[:6])
        t = calendar.timegm((y, mo, d, h, mi, s)) + float("0." + m.group(7)) - off
        try:
            out.append((t, float(q[2]), float(q[3]), float(q[4])))
        except ValueError:
            continue
    return out


def window_stats(rows, a, b):
    """(wall, idle s, util %, mean power W, mean SM MHz) over [a, b]; each sample reports the period ending at it."""
    busy = span = pw = clk = 0.0
    for i in range(len(rows) - 1):
        lo, hi = max(rows[i][0], a), min(rows[i + 1][0], b)
        if hi <= lo:
            continue
        dt = hi - lo
        busy += dt * rows[i + 1][1] / 100.0
        clk += dt * rows[i + 1][2]
        pw += dt * rows[i + 1][3]
        span += dt
    if not span:
        return None
    return (b - a, span - busy, 100.0 * busy / span, pw / span, clk / span)


def holds(run_dir):
    out = []
    for line in rd(f"{run_dir}/out.raw.log").splitlines():
        m = HOLD.match(line)
        if m:
            out.append((int(m.group(1)), m.group(2), float(m.group(3)), float(m.group(4)), float(m.group(5)),
                        float(m.group(6)), m.group(7).strip(" ·")))
    return out


def union(spans):
    out = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def overlap(a, b, spans):
    return sum(max(0.0, min(b, y) - max(a, x)) for x, y in spans)


def level0_split(rows, hs, a, b):
    """Level 0's nvidia-smi idle cut into the part inside a card-permit hold and the part between holds."""
    u = union([(h[4], h[5]) for h in hs if h[5] > a and h[4] < b])
    idle = inside = 0.0
    for i in range(len(rows) - 1):
        lo, hi = max(rows[i][0], a), min(rows[i + 1][0], b)
        if hi <= lo:
            continue
        frac_idle = 1.0 - rows[i + 1][1] / 100.0
        idle += (hi - lo) * frac_idle
        inside += overlap(lo, hi, u) * frac_idle
    held = sum(y - x for x, y in u)
    n = sum(1 for h in hs if h[5] > a and h[4] < b)
    return {"idle": idle, "inside": inside, "between": idle - inside, "held_union": held, "holds": n}


def stage_of(run_dir):
    try:
        return json.load(open(f"{run_dir}/stage.json"))
    except (OSError, ValueError):
        return None


def windows_abs(st):
    t0 = st["stage"]["prove_start_unix"]
    return {n: (t0 + a, t0 + b) for n, a, b in st.get("windows", [])}


def run(run_dir):
    rec = AW.run(run_dir)
    st = stage_of(run_dir)
    if not st or "stage" not in st:
        return rec
    rows = smi_rows(run_dir)
    hs = holds(run_dir)
    with open(f"{run_dir}/holds.tsv", "w") as f:
        f.write("seq\tphase\twaited_s\theld_s\tt0\tt1\twho\n")
        for h in hs:
            f.write("\t".join(str(x) for x in h) + "\n")
    w = windows_abs(st)
    with open(f"{run_dir}/idle.tsv", "w") as f:
        f.write("window\twall_s\tidle_s\tutil_pct\tpower_w\tsm_mhz\n")
        for name, (a, b) in w.items():
            s = window_stats(rows, a, b)
            if s:
                f.write(f"{name}\t{s[0]:.2f}\t{s[1]:.2f}\t{s[2]:.1f}\t{s[3]:.0f}\t{s[4]:.0f}\n")
    if "L0" in w:
        sp = level0_split(rows, hs, *w["L0"])
        with open(f"{run_dir}/l0split.json", "w") as f:
            json.dump(sp, f, indent=1)
        print(f"L0 {os.path.basename(run_dir)}: {sp['holds']} holds · held (union) {sp['held_union']:.2f} s · "
              f"idle {sp['idle']:.2f} s = inside holds {sp['inside']:.2f} + between {sp['between']:.2f}")
    return rec


# ---------------------------------------------------------------------------------------------- the tag's rows
def tsv(path):
    try:
        with open(path) as f:
            return list(csv.DictReader(f, delimiter="\t"))
    except OSError:
        return []


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def s(x, n=2):
    return "-" if x is None else f"{x:.{n}f}"


def within(x, lo, hi):
    return x is not None and lo <= x <= hi


def kern_sum(run_dir):
    """{kernel name: seconds} from nsys's cuda_gpu_kern_sum, or None."""
    p = f"{run_dir}/stats/s_cuda_gpu_kern_sum.csv"
    if not os.path.exists(p):
        return None
    out = {}
    with open(p) as f:
        for r in csv.DictReader(f):
            try:
                out[r["Name"]] = out.get(r["Name"], 0.0) + float(r["Total Time (ns)"]) / 1e9
            except (KeyError, ValueError):
                continue
    return out or None


class Run:
    def __init__(self, tagdir, name):
        self.name, self.dir = name, f"{tagdir}/{name}"
        self.block, self.kind = name[3], name[4]
        self.st = stage_of(self.dir) or {}
        self.stage = self.st.get("stage") or {}
        self.checks = self.st.get("checks") or AW.checks(AW.W.stamped(f"{self.dir}/out.log"))
        self.rc = rd(f"{self.dir}/rc.txt", "-").strip() or "-"
        self.win = {r["window"]: r for r in tsv(f"{self.dir}/win.tsv")}
        self.idle = {r["window"]: r for r in tsv(f"{self.dir}/idle.tsv")}
        self.l0 = None
        try:
            self.l0 = json.load(open(f"{self.dir}/l0split.json"))
        except (OSError, ValueError):
            pass
        self.kern = kern_sum(self.dir)

    @property
    def whole(self):
        return self.stage.get("whole")


def tag(tagdir):
    exp = json.load(open(f"{tagdir}/expect.json"))
    mode = exp.get("mode", "capture")
    names = sorted(d for d in os.listdir(tagdir) if re.match(r"^\d\d-[MP][wcn]$", d) and os.path.isdir(f"{tagdir}/{d}"))
    runs = [Run(tagdir, n) for n in names]
    read = [r for r in runs if r.kind != "w"]
    for r in runs:
        st = r.stage
        if st:
            print(f"RUN {r.name} rc {r.rc} · whole {s(st.get('whole'))} = A {s(st.get('A'))} + B {s(st.get('B'))} + "
                  f"recursion {s(st.get('recursion'))} · argue {s(st.get('B_argue'))} open {s(st.get('B_open'))} "
                  f"encode {s(st.get('B_encode'))} · shape {st.get('tables')}/{st.get('ngroups')}/{st.get('cells_G')} · "
                  f"leaves beside {st.get('leaves_beside')} · capture {'yes' if r.kern else 'no'}")
        else:
            print(f"RUN {r.name} rc {r.rc} · UNFINISHED")
    out_rows = []

    # V1
    why = []
    for r in read:
        c = r.checks
        if r.rc != "0":
            why.append(f"{r.name} rc {r.rc}")
        if not c.get("result_ok") or c.get("failures_list"):
            why.append(f"{r.name} libtest not ok")
        if not (c.get("harness") and c.get("tree")):
            why.append(f"{r.name} a verifier did not accept")
        dev = c.get("device") or {}
        if len(dev) != len(AW.DEVICE_KEYS) or any(v != 0 for v in dev.values()):
            why.append(f"{r.name} device counters {dev}")
        want = exp["shape"][r.block]
        got = [r.stage.get("tables"), r.stage.get("ngroups"), r.stage.get("cells_G")]
        if got != want:
            why.append(f"{r.name} shape {got} against {want}")
    v1 = not why and len(read) > 0
    out_rows.append(("V1", v1, f"{len(read)} runs read: rc 0, libtest ok, both verifiers ACCEPTED, six device counters 0, "
                     f"shape M {exp['shape']['M']} P {exp['shape']['P']}" if v1 else "; ".join(why) or "no run read"))

    # V2 (captures only)
    if mode == "capture":
        parts, ok = [], True
        for b in "MP":
            if not any(r.block == b for r in read):
                parts.append(f"{b}: not run")
                continue
            ctl = mean([r.whole for r in read if r.block == b and r.kind == "c"])
            caps = [r.whole for r in read if r.block == b and r.kind == "n"]
            if ctl is None or not caps or any(x is None for x in caps):
                ok = False
                parts.append(f"{b}: unreadable")
                continue
            ratios = [x / ctl for x in caps]
            ok &= all(V2_BAND[0] <= x <= V2_BAND[1] for x in ratios)
            parts.append(f"{b}: capture ÷ control {' / '.join(f'{x:.3f}' for x in ratios)} (control {ctl:.2f} s)")
        out_rows.append(("V2", ok, " · ".join(parts) + f" · registered {V2_BAND[0]}–{V2_BAND[1]}"))
    else:
        out_rows.append(("V2", None, f"no capture: {exp.get('why', 'controls only')}"))

    # V3: each control against a reference bench of the same machine, when one is given; otherwise V2 is the check
    parts, ok = [], True
    ref = exp.get("reference")
    if not ref:
        out_rows.append(("V3", None, "no reference bench on this machine: the internal controls (V2) are the check"))
    for b in ("MP" if ref else ""):
        rv = ref.get(b)
        for r in [r for r in read if r.block == b and r.kind == "c"]:
            d = None if (rv is None or r.whole is None) else r.whole - rv
            ok &= d is not None and abs(d) <= V3_TOL[b]
            parts.append(f"{r.name} {s(r.whole)} against the reference {s(rv)} ({s(d)}; ±{V3_TOL[b]})")
    if ref:
        if not parts:
            ok = False
        out_rows.append(("V3", ok, " · ".join(parts) or "no control read"))

    # V4
    hyg = rd(f"{tagdir}/hygiene.txt").strip()
    out_rows.append(("V4", hyg == "ok", f"hygiene {hyg or 'unread'}"))

    exps = []
    mc = [r for r in read if r.block == "M" and r.kind == "c"]
    mn = [r for r in read if r.block == "M" and r.kind == "n"]

    # E1: card idle share by window, median controls (nvidia-smi)
    if mc:
        parts, met = [], True
        for wn, label, lo, hi in E1_WIN:
            vals = []
            for r in mc:
                i = r.idle.get(wn)
                if i and fnum(i["wall_s"]):
                    vals.append(100.0 * fnum(i["idle_s"]) / fnum(i["wall_s"]))
            v = mean(vals)
            met &= within(v, lo, hi)
            parts.append(f"{label} {s(v, 1)} % [{lo:g}–{hi:g}]")
        exps.append(("E1", "MET" if met else "MISSED", "median control, card idle as a share of each window: " + " · ".join(parts)))
    else:
        exps.append(("E1", "UNREAD", "no median control"))

    # E2: kernel shares, median captures (whole run, own durations)
    ks = [r.kern for r in mn if r.kern]
    if ks:
        def share(k, names, prefix=None):
            tot = sum(k.values())
            v = sum(t for n, t in k.items() if n in names or (prefix and n.startswith(prefix)))
            return 100.0 * v / tot if tot else None
        leaf = mean([share(k, LEAF) for k in ks])
        ext = mean([share(k, EXT) for k in ks])
        gkr = mean([share(k, GKR) for k in ks])
        first = mean([share(k, LEAN, TILE_PREFIX) for k in ks])
        met = within(leaf, 20, 30) and within(ext, 20, 30) and within(gkr, 10, 17) and within(first, 0, 4)
        tot = mean([sum(k.values()) for k in ks])
        top = sorted(ks[0].items(), key=lambda x: -x[1])[:12]
        exps.append(("E2", "MET" if met else "MISSED",
                     f"median capture, share of all kernel time ({s(tot)} s): commit leaf hash {s(leaf, 1)} % [20–30] · "
                     f"the extension's four kernels {s(ext, 1)} % [20–30] · GKR layer rounds {s(gkr, 1)} % [10–17] · "
                     f"first-round sumcheck (lean + tile kernels) {s(first, 1)} % [0–4] · top: "
                     + " · ".join(f"{n} {t:.2f}" for n, t in top)))
    else:
        exps.append(("E2", "UNREAD", "no median capture with nsys stats"))

    # E3: prover cores by window, median controls (the 0.5 s sampler)
    if mc:
        parts, met = [], True
        for wn, label, lo, hi in E3_WIN:
            v = mean([fnum(r.win.get(wn, {}).get("prover_cores")) for r in mc])
            met &= within(v, lo, hi)
            parts.append(f"{label} {s(v, 1)} [{lo:g}–{hi:g}]")
        exps.append(("E3", "MET" if met else "MISSED", "median control, prover cores busy: " + " · ".join(parts)
                     + " (bands from a 16-core, 32-thread desktop CPU; another CPU makes a miss a finding about the machine)"))
    else:
        exps.append(("E3", "UNREAD", "no median control"))

    # E4: level 0's idle inside vs between card holds, median controls
    l0 = [r.l0 for r in mc if r.l0 and r.l0.get("holds")]
    if l0:
        idle = mean([x["idle"] for x in l0])
        inside = mean([x["inside"] for x in l0])
        frac = inside / idle if idle else None
        exps.append(("E4", "MET" if within(frac, 0.5, 1.0) else "MISSED",
                     f"median control, level 0: idle {s(idle)} s, inside a hold {s(inside)} s ({s(100 * frac if frac is not None else None, 0)} %) "
                     f"[inside ≥ 50 %] · holds {mean([x['holds'] for x in l0]):.0f} · held (union) {s(mean([x['held_union'] for x in l0]))} s"))
    else:
        exps.append(("E4", "UNREAD", "no CARD HOLD line inside level 0 (LFM_CARD_TRACE absent or no hold)"))

    # E5: the commit window's kernel cover, median captures
    cov = [fnum(r.win.get("A2 commits", {}).get("busy_s")) / fnum(r.win["A2 commits"]["wall_s"])
           for r in mn if fnum(r.win.get("A2 commits", {}).get("busy_s")) is not None and fnum(r.win.get("A2 commits", {}).get("wall_s"))]
    if cov:
        v = mean(cov)
        exps.append(("E5", "MET" if within(v, 0.85, 0.97) else "MISSED", f"median capture, commits window: a kernel runs {s(100 * v, 1)} % of it [85–97]"))
    else:
        exps.append(("E5", "UNREAD", "no median capture"))

    # REPORTED: power and SM clock per window, and level 0's split per run
    rep = []
    for r in mc[:1] + [x for x in read if x.block == "P" and x.kind == "c"][:1]:
        rep.append(f"{r.name}: " + " · ".join(f"{wn} {i['power_w']} W {i['sm_mhz']} MHz" for wn, i in r.idle.items()
                                             if wn in ("A2 commits", "B proving phase", "L0")))
    for r in read:
        if r.l0:
            rep.append(f"{r.name} L0 idle {s(r.l0['idle'])} = inside {s(r.l0['inside'])} + between {s(r.l0['between'])} ({r.l0['holds']} holds)")
    for k, ok, text in out_rows:
        print(f"ROW {k} {'UNREAD' if ok is None else ('IN' if ok else 'OUT')}: {text}")
    for k, state, text in exps:
        print(f"EXPECT {k} {state}: {text}")
    print("REPORTED: " + " | ".join(rep))
    gate_out = [k for k, ok, _ in out_rows if ok is False]
    wholes = " · ".join(f"{r.name} {s(r.whole)}" for r in runs)
    em = sum(1 for _, st_, _ in exps if st_ == "MET")
    ex = " ".join(f"{k}:{st_}" for k, st_, _ in exps)
    if gate_out:
        print(f"VERDICT: BLOCK PROFILE FAIL — rows out: {' '.join(gate_out)} · wholes {wholes} · expectations met {em} of 5 ({ex})")
        return 11
    tagline = "" if mode == "capture" else f" (no capture: {exp.get('why', 'controls only')})"
    print(f"VERDICT: BLOCK PROFILE PASS{tagline} — rows {' '.join(k for k, ok, _ in out_rows if ok)} IN · wholes {wholes} · "
          f"expectations met {em} of 5 ({ex})")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("run", "tag"):
        print(__doc__)
        print("VERDICT: REFUSED rc=1 — usage")
        sys.exit(1)
    if sys.argv[1] == "run":
        run(sys.argv[2].rstrip("/"))
        sys.exit(0)
    sys.exit(tag(sys.argv[2].rstrip("/")))
