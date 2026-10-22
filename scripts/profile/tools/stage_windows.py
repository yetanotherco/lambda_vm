#!/usr/bin/env python3
"""Phase windows from the prover's own stamped log, applied to the text tables nsys_extract.py wrote, the host sampler
and nvidia-smi. Text in, text out.

    stage_windows.py run <run dir> W|S      → <run dir>/win.tsv fam.tsv top.tsv gaps.tsv run.md; prints one RUN line
    stage_windows.py agg <tag dir> W|S      → <tag dir>/agg.md (per block x arm: means over the nsys runs); prints it

W = #1014 (W3 lines: PROVE START unix, BLOCK PHASES A, W3 BASE, W3 LEVEL 0, W3 RECURSION), S = #1013 (stamps of BLOCK
PHASE stream / prove, harvest, BLOCK LEVEL 0, the NO-EPOCH BLOCK line). Card numbers: busy = the union of
device operations; each busy instant shared equally among the operations active at it), from 0.1 s bins, so window edges
are good to ±0.05 s. Host numbers: cores = Δ(utime+stime) / Δt of the prover process (and of the whole cgroup).
"""
import csv
import datetime
import os
import re
import statistics as st
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nsys_tables as R  # noqa: E402

# Families, first match wins. P1 = the Poseidon1 base kernels (p1s_* arity-4 trees, p1w16_* width-16 leaves, p1w8_* grind).
FAM = [
    ("hash P1 leaves", r"^(p1s_(leaves|fri_group_leaves)|p1w16_zleaves)"),
    ("hash P1 Merkle", r"^p1s_merkle"),
    ("hash P1 grind", r"^(p1s_grind|p1w8_grind)"),
    ("transpose / gather / pack", r"^(p1s_gather|matrix_transpose|transpose|bit_reverse|gather|merkle_gather|pack|unpack|narrow|widen_)"),
    ("hash P1 other", r"^p1(s|w8|w16)_"),
    ("hash RPX grind", r"^rpx_grind"),
    ("hash RPX leaves", r"^rpx_(leaves|comp_poly_leaves|fri_group_leaves|fri_leaves|.*leaves)"),
    ("hash RPX Merkle", r"^rpx_(merkle|tail)"),
    ("hash RPX other", r"^rpx_"),
    ("hash other (keccak / blake3)", r"^(keccak|blake3)"),
    ("NTT / LDE", r"^(ntt_|mobius_|lift_spread|pointwise_mul|scalar_mul|coset|lde_)"),
    ("sumcheck / GKR / fold", r"^(sumcheck_|sum_partials|whir_|eq_expand|eq_seed|fraction_fold|mle_|fri_fold|fri_update|add_scaled|fill_ext3|gkr_|zc_)"),
    ("constraint / LogUp / quotient", r"^(constraint_|ccomp_|si_|program_|interp|factors_from|logup_|decompose|comp_h|quotient|bytecode|compiled_|lfm_constraint)"),
    ("DEEP / barycentric / inversion", r"^(deep_|barycentric|compute_denoms|batch_inverse|invert|block_inclusive_scan|apply_block_offsets|ood_)"),
]
HASH = [f for f, _ in FAM if f.startswith("hash ")]
COLS = ["hash P1", "hash RPX", "NTT / LDE", "sumcheck / GKR / fold", "constraint / LogUp / quotient",
        "DEEP / barycentric / inversion", "transpose / gather / pack", "copy H2D", "copy D2H", "copy D2D", "memset", "other kernels"]
SHORT = ["P1 hash", "RPX hash", "NTT/LDE", "sumcheck/GKR/fold", "constraint", "DEEP/bary", "transpose/gather", "H2D", "D2H", "D2D",
         "memset", "other"]


def col_of(fam):
    if fam.startswith("hash P1"):
        return "hash P1"
    if fam.startswith("hash RPX") or fam.startswith("hash other"):
        return "hash RPX"
    return fam if fam in COLS else "other kernels"


def rd(p, d=""):
    try:
        with open(p, errors="replace") as f:
            return f.read()
    except OSError:
        return d


def stamped(path):
    """[(unix, text)] from a stamp.py log ('<unix> <line>')."""
    out = []
    for line in rd(path).splitlines():
        t, _, txt = line.partition(" ")
        try:
            out.append((float(t), txt))
        except ValueError:
            pass
    return out


def first(lt, rx):
    c = re.compile(rx)
    for t, txt in lt:
        m = c.search(txt)
        if m:
            return t, m
    return None, None


def windows(lt, kind):
    """[(name, a, b)] in unix seconds, plus a dict of the parsed phase numbers. Empty when the run did not finish."""
    info = {}
    if kind == "W":
        _, m0 = first(lt, r"^W3 PROVE START: unix ([\d.]+)")
        _, mp = first(lt, r"^BLOCK PHASES: .*?· A ([\d.]+) \(wait ([\d.]+) .*?· B ([\d.]+)")
        _, mb = first(lt, r"^W3 BASE: ([\d.]+)s")
        _, ml = first(lt, r"^W3 LEVEL 0: ([\d.]+)s wall")
        _, mr = first(lt, r"^W3 RECURSION: ([\d.]+)s after the base .*?whole block ([\d.]+)s")
        _, mh = first(lt, r"^W3 BASE HASH: (\w+)")
        _, mg = first(lt, r"^W3 PLAN: (\d+) groups · .*? · (\d+) leaves ")
        info["hash"] = mh.group(1) if mh else "-"
        info["plan"] = f"{mg.group(1)}g/{mg.group(2)}L" if mg else "-"
        if not (m0 and mp and mb and ml and mr):
            return [], info
        t0, A, wait, base, L0 = float(m0.group(1)), float(mp.group(1)), float(mp.group(2)), float(mb.group(1)), float(ml.group(1))
        rec, whole = float(mr.group(1)), float(mr.group(2))
        info.update(A=A, B=base - A, base=base, L0=L0, rec=rec, whole=whole, wait=wait)
        return [("A: pre-roll (wait for group 0)", t0, t0 + wait), ("A: build + commits", t0 + wait, t0 + A),
                ("B: argue + open", t0 + A, t0 + base), ("L0 (leaves)", t0 + base, t0 + base + L0),
                ("interior (L1..top)", t0 + base + L0, t0 + whole), ("phase A", t0, t0 + A), ("recursion", t0 + base, t0 + whole),
                ("WHOLE", t0, t0 + whole)], info
    te, me = first(lt, r"★★★ NO-EPOCH BLOCK: base ([\d.]+)s · harvest ([\d.]+)s · level 0 ([\d.]+)s .*?recursion ([\d.]+)s · whole ([\d.]+)s")
    tA, _ = first(lt, r"BLOCK PHASE stream [\d.]+s")
    tB, _ = first(lt, r"BLOCK PHASE prove [\d.]+s")
    th, _ = first(lt, r"^\s+harvest: [\d.]+s \(")
    tl, _ = first(lt, r"^\s+BLOCK LEVEL 0: ")
    _, mh = first(lt, r"^BASE HASH: (\w+)")
    _, mi = first(lt, r"harvest: [\d.]+s .*?· (\d+) instances")
    _, ml = first(lt, r"^\s+BLOCK LEVEL 0: (\d+) leaves")
    info["hash"] = mh.group(1) if mh else "rpx?"
    info["plan"] = f"{mi.group(1) if mi else '?'}i/{ml.group(1) if ml else '?'}L"
    if not (te and tA and tB and th and tl):
        return [], info
    base, harvest, L0, rec, whole = (float(me.group(i)) for i in (1, 2, 3, 4, 5))
    t0 = te - whole
    info.update(A=tA - t0, B=tB - tA, base=base, L0=L0, rec=rec, whole=whole, harvest=harvest)
    return [("phase A (execute + build + streamed commits)", t0, tA), ("phase B (setup + prove)", tA, tB), ("harvest", tB, th),
            ("L0 (leaves)", th, tl), ("interior (L1..top)", tl, te), ("recursion", tB, te), ("WHOLE", t0, te)], info


def host(rd_, a, b):
    """(prover cores, cgroup cores, prover VmRSS peak GiB) over [a, b] from the sampler; Nones when absent."""
    rows = []
    for i, line in enumerate(rd(f"{rd_}/host.tsv").splitlines()):
        q = line.split("\t")
        if i == 0 or len(q) < 9:
            continue
        try:
            rows.append((float(q[0]), int(q[3]), int(q[5]), int(q[6])))
        except ValueError:
            pass
    inw = [r for r in rows if a <= r[0] <= b]
    if len(inw) < 2:
        return None, None, None
    tk = [r for r in inw if r[3] >= 0]
    pc = (tk[-1][3] - tk[0][3]) / 100.0 / (tk[-1][0] - tk[0][0]) if len(tk) >= 2 and tk[-1][0] > tk[0][0] else None
    cg = (inw[-1][1] - inw[0][1]) / 1e6 / (inw[-1][0] - inw[0][0]) if inw[-1][0] > inw[0][0] and inw[0][1] >= 0 else None
    rss = max((r[2] for r in inw), default=-1) / 2**20
    return pc, cg, (rss if rss > 0 else None)


def smi(rd_, a, b):
    """(mean util.gpu %, median SM MHz while util ≥ 50, max VRAM MiB) over [a, b] from nvidia-smi -lms 200 (machine-local time)."""
    tz = rd(f"{rd_}/tz.txt").strip() or "+0000"
    off = (1 if tz[0] == "+" else -1) * (int(tz[1:3]) * 3600 + int(tz[3:5]) * 60)
    u, sm, vr = [], [], 0
    for line in rd(f"{rd_}/smi.csv").splitlines():
        q = [x.strip() for x in line.split(",")]
        try:
            t = datetime.datetime.strptime(q[0], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=datetime.timezone.utc).timestamp() - off
            if a <= t <= b:
                u.append(int(q[2])); vr = max(vr, int(q[1]))
                if int(q[2]) >= 50:
                    sm.append(int(q[3]))
        except (ValueError, IndexError):
            pass
    return (sum(u) / len(u) if u else None), (int(st.median(sm)) if sm else None), (vr or None)


def f2(x, n=2):
    return "-" if x is None else f"{x:.{n}f}"


def run(rd_, kind):
    lt = stamped(f"{rd_}/out.log")
    wins, info = windows(lt, kind)
    log = "\n".join(t for _, t in lt)
    ok = "test result: ok. 1 passed" in log and (
        re.search(r"^W3 TREE VERIFY: ACCEPTED", log, re.M) is not None if kind == "W"
        else "BLOCK VERIFIER: verify_block_tree" in log)
    has_x = os.path.exists(f"{rd_}/x/summary.tsv")
    c = R.load(rd_) if has_x else None
    wrows, frows, trows, grows = [], [], [], []
    md = [f"# {os.path.basename(rd_)} · {kind} · hash {info.get('hash')} · plan {info.get('plan')} · verified {ok}", ""]
    md.append("| window | wall s | card busy s | busy % | idle s | ≥2 kernels s | " + " | ".join(SHORT) + " | hash % of busy | prover cores | cgroup cores | smi util % | RSS peak GiB |")
    md.append("|---" * (13 + len(SHORT)) + "|")
    for name, a, b in wins:
        pc, cg, rss = host(rd_, a, b)
        su, _, _ = smi(rd_, a, b)
        if c:
            w = R.window(c, a, b, FAM)
            cols = defaultdict(float)
            for fam, s in w["fam"].items():
                cols[col_of(fam)] += s
            hs = sum(w["fam"].get(f, 0) for f in HASH)
            busy, idle, k2 = w["busy"], w["wall"] - w["busy"], w["k2"]
            wall = w["wall"]
            for fam, s in sorted(w["fam"].items(), key=lambda x: -x[1]):
                frows.append((name, fam, f"{s:.3f}"))
            for k, (sh, raw, n) in sorted(w["top"].items(), key=lambda x: -x[1][0])[:25]:
                trows.append((name, R.keyname(c, k)[0][:90], f"{sh:.3f}", f"{raw:.3f}", n))
            for cls, s in sorted(w["gaps"].items()):
                grows.append((name, "class", cls, f"{s:.3f}"))
            for ctx, s in sorted(w["gctx"].items(), key=lambda x: -x[1])[:6]:
                grows.append((name, "bounded by", ctx, f"{s:.3f}"))
        else:
            cols, hs, busy, idle, k2, wall = {}, None, None, None, None, b - a
        wrows.append((name, f"{a:.3f}", f"{b:.3f}", f"{wall:.2f}", f2(busy), f2(idle), f2(k2), f2(hs), f2(pc), f2(cg), f2(su, 1), f2(rss))
                     + tuple(f2(cols.get(k)) if c else "-" for k in COLS))
        md.append(f"| {name} | {wall:.2f} | {f2(busy)} | {f2(100 * busy / wall if busy is not None and wall else None, 1)} | {f2(idle)} | {f2(k2)} | "
                  + " | ".join(f2(cols.get(k)) if c else "-" for k in COLS)
                  + f" | {f2(100 * hs / busy if hs is not None and busy else None, 1)} | {f2(pc, 1)} | {f2(cg, 1)} | {f2(su, 1)} | {f2(rss)} |")
    hdr = ["window", "a", "b", "wall_s", "busy_s", "idle_s", "k2_s", "hash_s", "prover_cores", "cgroup_cores", "smi_util", "rss_gib"] + COLS
    for fn, h, rows in (("win.tsv", hdr, wrows), ("fam.tsv", ["window", "family", "card_s"], frows),
                        ("top.tsv", ["window", "kernel", "share_s", "raw_s", "launches"], trows),
                        ("gaps.tsv", ["window", "what", "key", "idle_s"], grows)):
        with open(f"{rd_}/{fn}", "w") as f:
            f.write("\t".join(h) + "\n")
            for r in rows:
                f.write("\t".join(str(x) for x in r) + "\n")
    _, sm, vr = smi(rd_, wins[-1][1], wins[-1][2]) if wins else (None, None, None)
    if c and wins:
        md += ["", "Top kernels, WHOLE (card-s, equal share):", ""]
        for r in [r for r in trows if r[0] == "WHOLE"][:15]:
            md.append(f"- {r[1]}: {r[2]} (raw {r[3]}, {r[4]} launches)")
    with open(f"{rd_}/run.md", "w") as f:
        f.write("\n".join(md) + "\n")
    whole = info.get("whole")
    wb = next((r for r in wrows if r[0] == "WHOLE"), None)
    print(f"RUN {os.path.basename(rd_)} rc {rd(f'{rd_}/rc.txt', '-').strip() or '-'} · verified {ok} · hash {info.get('hash')} · plan {info.get('plan')} · "
          f"whole {f2(whole)} (A {f2(info.get('A'))} · B {f2(info.get('B'))} · recursion {f2(info.get('rec'))}) · "
          f"card busy {wb[4] if wb else '-'} s · hash {wb[7] if wb else '-'} s · SM {sm or '-'} MHz · VRAM {vr or '-'} MiB")
    return ok, whole, info


def agg(tag, kind):
    runs = sorted(d for d in os.listdir(tag) if re.match(r"^\d\d-[bmp][RP][cn]$", d) and os.path.isdir(f"{tag}/{d}"))
    blocks = {"b": "bench 25368371", "m": "median 25475471", "p": "p90 25481021"}
    arms = {"R": "RPX base", "P": "Poseidon1 base"}
    out = [f"# stage windows {kind} · {tag}", ""]
    for bk in "bmp":
        for arm in "PR":
            rs = [d for d in runs if d[3] == bk and d[4] == arm]
            if not rs:
                continue
            ctl, nsy = [], []
            for d in rs:
                lt = stamped(f"{tag}/{d}/out.log")
                _, info = windows(lt, kind)
                ok = os.path.exists(f"{tag}/{d}/win.tsv") and rd(f"{tag}/{d}/rc.txt").strip() == "0"
                (nsy if d[5] == "n" else ctl).append((d, info, ok))
            out.append(f"## {blocks[bk]} · {arms[arm]} · hash {next((i.get('hash') for _, i, _ in nsy + ctl), '-')} · "
                       f"plan {next((i.get('plan') for _, i, _ in nsy + ctl), '-')}")
            out.append("")
            out.append("| run | rc | whole s | A s | B s | recursion s |")
            out.append("|---|---|---|---|---|---|")
            for d, info, ok in ctl + nsy:
                out.append(f"| {d} ({'control' if d[5] == 'c' else 'nsys'}) | {rd(f'{tag}/{d}/rc.txt', '-').strip()} | {f2(info.get('whole'))} | "
                           f"{f2(info.get('A'))} | {f2(info.get('B'))} | {f2(info.get('rec'))} |")
            good = [d for d, _, ok in nsy if ok and os.path.exists(f"{tag}/{d}/x/summary.tsv")]
            if not good:
                out += ["", "(no extracted capture)", ""]
                continue
            tabs = []
            for d in good:
                with open(f"{tag}/{d}/win.tsv") as f:
                    tabs.append({r["window"]: r for r in csv.DictReader(f, delimiter="\t")})
            out += ["", f"Mean over {len(good)} capture(s) ({', '.join(good)}); card-seconds, equal share:", ""]
            out.append("| window | wall s | busy % | idle s | " + " | ".join(SHORT) + " | hash % of busy | prover cores | cgroup cores |")
            out.append("|---" * (8 + len(SHORT)) + "|")

            def mean(win, key):
                v = []
                for t in tabs:
                    x = t.get(win, {}).get(key, "-")
                    if x not in ("-", "", None):
                        v.append(float(x))
                return sum(v) / len(v) if v else None
            for win in tabs[0]:
                wall, busy, hs = mean(win, "wall_s"), mean(win, "busy_s"), mean(win, "hash_s")
                out.append(f"| {win} | {f2(wall)} | {f2(100 * busy / wall if busy is not None and wall else None, 1)} | {f2(mean(win, 'idle_s'))} | "
                           + " | ".join(f2(mean(win, k)) for k in COLS)
                           + f" | {f2(100 * hs / busy if hs is not None and busy else None, 1)} | {f2(mean(win, 'prover_cores'), 1)} | {f2(mean(win, 'cgroup_cores'), 1)} |")
            tops = defaultdict(list)
            for d in good:
                with open(f"{tag}/{d}/top.tsv") as f:
                    for r in csv.DictReader(f, delimiter="\t"):
                        if r["window"] == "WHOLE":
                            tops[r["kernel"]].append(float(r["share_s"]))
            out += ["", "Top kernels over WHOLE (mean card-s over the captures):", ""]
            for k, v in sorted(tops.items(), key=lambda x: -sum(x[1]) / len(good))[:12]:
                out.append(f"- {k}: {sum(v) / len(good):.2f}")
            out.append("")
    with open(f"{tag}/agg.md", "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    mode, d, kind = sys.argv[1], sys.argv[2], sys.argv[3]
    assert kind in ("W", "S")
    if mode == "run":
        run(d, kind)
    else:
        agg(d, kind)
