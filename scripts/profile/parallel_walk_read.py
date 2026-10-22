#!/usr/bin/env python3
"""parallel-walk.sh's read-out: can a block's windows be walked in parallel? Python 3 standard library only.

usage: parallel_walk_read.py <runs dir> <expected runs, comma-separated>

Each run is a directory: S-I (the small block), S-M1, S-M2 (the median); a warm-up W-I is not read. Its out.log
carries the harness's `PARALLEL WALK` lines, card.txt the card's readings while it ran, rc.txt the test's status.
Rows (registered before any run; the large block's rows are left out, it does not fit this machine's memory):
  LANDING (one-sided, every read run holds it; L3 and L4 at the median)
    RUNS  every expected run printed its DONE line
    CARD  the card stayed idle: memory.used rose at most 256 MiB over its reading before the run, no compute process
    L1    windows whose walk again differs from the sequential walk's, any P, any arm (copies, entry states, beside
          generators), plus entry bytes that end otherwise: = 0
    L2    memory cells and register cells where the state pass differs from the walk at the samples' starts: = 0
    L2b   bytes the samples' memory rows say were read first (within width) that the pass did not record: = 0
    L3    state pass ns/cycle <= executor ns/cycle (resume + copy, alone, same run)
    L4    speedup at P = 8 over P = 1 from copies: alone >= 3.0; beside 6 generators >= 2.0
  MECHANISM (two-sided; the mean of the median's runs)
    M1 walk ns/cycle 45-60 · M2 executor ns/cycle 16-22 · M3 state pass ns/cycle 5-15 · M4 P = 1 from a copy /
    sequential 0.90-1.15 · M5 speedup alone P = 2 / 4 / 8 / 16: 1.8-2.0 / 3.2-3.9 / 5.0-7.5 / 7-12 · M6 beside 6
    generators at P = 8: 3.5-6.5 · M7 entries a cycle 0.22-0.45, largest window <= 6 M · M8 setup as built before:
    small 0.05-0.4 s, median 0.5-1.2 s
The decision rule: windows walked in parallel GO if L1-L4 are IN; L3 OUT re-plans around patching the entry state
up after the walk; L4 OUT stops it (a bandwidth-bound walk). The setup change is a GO if the median's setup before is
>= 0.5 s or its image and seeding part >= 0.3 s. The verdict line starts `VERDICT: PARALLEL WALK`.
"""
import os
import re
import statistics as st
import sys

out = sys.argv[1]
expected = [r for r in sys.argv[2].split(",") if r]
SIZE = {"I": "small", "M": "median", "P": "large"}


def rd(p):
    try:
        return open(p, errors="replace").read()
    except OSError:
        return ""


def f1(pattern, text, cast=float):
    m = re.search(pattern, text)
    return cast(m.group(1)) if m else None


def parse(name):
    d = os.path.join(out, name)
    log = rd(f"{d}/out.log")
    r = {"name": name, "size": name[2], "rc": rd(f"{d}/rc.txt").strip() or "-"}
    r["done"] = "PARALLEL WALK DONE:" in log
    r["before"] = f1(r"PARALLEL WALK SETUP: before ([\d.]+) s", log)
    r["image_before"] = f1(r"image and memory before ([\d.]+) s", log)
    r["saved"] = f1(r"\|\| saved ([\d.]+) s", log)
    r["exec_ns"] = f1(r"PARALLEL WALK EXECUTOR: .*? ([\d.]+) ns/cycle", log)
    r["walk_ns"] = f1(r"PARALLEL WALK SEQUENTIAL: .*?walk [\d.]+ s \(([\d.]+) ns/cycle\)", log)
    r["pass_ns"] = f1(r"PARALLEL WALK SEQUENTIAL: .*?state pass [\d.]+ s \(([\d.]+) ns/cycle\)", log)
    r["samples"] = f1(r"PARALLEL WALK SEQUENTIAL: .*?samples (\d+) at", log, int)
    r["per_cycle"] = f1(r"PARALLEL WALK ENTRIES: total \d+ \(([\d.]+) a cycle\)", log)
    r["largest"] = f1(r"PARALLEL WALK ENTRIES: .*?largest window (\d+)", log, int)
    r["census_missing"] = f1(r"census bytes not among the entries (\d+)", log, int)
    r["pass_cells"] = f1(r"PARALLEL WALK PASS: .*?cells differing (\d+)", log, int)
    r["pass_regs"] = f1(r"PARALLEL WALK PASS: .*?register cells differing (\d+)", log, int)
    r["copy"] = {int(p): float(s) for p, s in re.findall(r"PARALLEL WALK COPIES P=(\d+): .*?speedup ([\d.]+) ", log)}
    r["entry"] = {int(p): float(s) for p, s in re.findall(r"PARALLEL WALK ENTRY P=(\d+): .*?speedup ([\d.]+) ", log)}
    r["beside"] = f1(r"PARALLEL WALK BESIDE P=8 G=6: .*?speedup ([\d.]+) over", log)
    r["from_copy"] = f1(r"PARALLEL WALK FROM A COPY: P=1 window time ([\d.]+) of", log)
    r["mismatches"] = f1(r"PARALLEL WALK DONE: mismatches (\d+)", log, int)
    r["misses"] = f1(r"PARALLEL WALK DONE: .*?exit misses (\d+)", log, int)
    card = rd(f"{d}/card.txt")
    r["card_rise"] = f1(r"rise (-?\d+) MiB", card, int)
    r["card_apps"] = f1(r"compute processes seen (\d+)", card, int)
    r["card_absent"] = "no nvidia-smi" in card
    r["peak_gib"] = f1(r"peak VmHWM ([\d.]+) GiB", rd(f"{d}/host.txt"))
    return r


runs = [parse(n) for n in expected if os.path.isdir(os.path.join(out, n))]
fmt = lambda x, p=2: f"{x:.{p}f}" if isinstance(x, (int, float)) else "-"
print(f"PARALLEL WALK read-out · expected runs {','.join(expected)}")
print("run | rc | done | setup before | image before | saved | exec ns | walk ns | pass ns | samples | entries/cyc | "
      "largest | pass cells/regs | census miss | P8 alone | P8 beside | mismatches | exit misses | card rise MiB / apps | peak GiB")
for r in runs:
    print(" | ".join(str(x) for x in (r["name"], r["rc"], r["done"], fmt(r["before"], 3), fmt(r["image_before"], 3),
          fmt(r["saved"], 3), fmt(r["exec_ns"]), fmt(r["walk_ns"]), fmt(r["pass_ns"]), r["samples"], fmt(r["per_cycle"], 4),
          r["largest"], f"{r['pass_cells']}/{r['pass_regs']}", r["census_missing"], fmt(r["copy"].get(8)),
          fmt(r["beside"]), r["mismatches"], r["misses"],
          "no card" if r["card_absent"] else f"{r['card_rise']} / {r['card_apps']}", fmt(r["peak_gib"]))))
    print(f"  {r['name']} speedup from copies: " + " · ".join(f"P={p} {s:.2f}" for p, s in sorted(r["copy"].items()))
          + " || from entry states: " + " · ".join(f"P={p} {s:.2f}" for p, s in sorted(r["entry"].items())))

land, mech = [], []
read = [r for r in runs if r["done"]]
land.append(("RUNS", len(read) == len(expected), f"{len(read)} of {len(expected)} expected runs printed their DONE line"))
cb = [r["name"] for r in runs if not r["card_absent"] and not (r["card_rise"] is not None and r["card_rise"] <= 256
                                                                  and r["card_apps"] == 0)]
land.append(("CARD", not cb, "the card stayed idle in every run" + (f" — not: {', '.join(cb)}" if cb else "")))
l1 = sum((r["mismatches"] or 0) + (r["misses"] or 0) for r in read)
land.append(("L1", bool(read) and l1 == 0, f"walks again that differ + entry bytes that end otherwise: {l1}"))
l2 = sum((r["pass_cells"] or 0) + (r["pass_regs"] or 0) for r in read)
land.append(("L2", bool(read) and l2 == 0 and all(r["pass_cells"] is not None for r in read),
             f"pass cells + register cells differing at the samples' starts: {l2}"))
l2b = sum(r["census_missing"] or 0 for r in read)
land.append(("L2b", bool(read) and l2b == 0, f"census bytes the pass did not record: {l2b}"))
med = [r for r in read if r["size"] == "M"]
land.append(("L3", bool(med) and all(r["pass_ns"] is not None and r["exec_ns"] is not None and r["pass_ns"] <= r["exec_ns"]
                                     for r in med),
             "pass ns/cycle <= executor's: " + " · ".join(f"{r['name']} {fmt(r['pass_ns'])} vs {fmt(r['exec_ns'])}" for r in med)))
land.append(("L4", bool(med) and all((r["copy"].get(8) or 0) >= 3.0 and (r["beside"] or 0) >= 2.0 for r in med),
             "P=8 speedup alone >= 3.0 / beside >= 2.0: " + " · ".join(
                 f"{r['name']} {fmt(r['copy'].get(8))} / {fmt(r['beside'])}" for r in med)))


def mean_of(size, get):
    v = [x for x in (get(r) for r in read if r["size"] == size) if x is not None]
    return st.mean(v) if v else None


def band(name, size, get, lo, hi, what, p=2):
    m = mean_of(size, get)
    mech.append((f"{name}-{size}", m is not None and lo <= m <= hi, f"{SIZE[size]} {what} {fmt(m, p)} in [{lo}, {hi}]"))
    return m


band("M1", "M", lambda r: r["walk_ns"], 45, 60, "walk ns/cycle")
band("M2", "M", lambda r: r["exec_ns"], 16, 22, "executor ns/cycle")
band("M3", "M", lambda r: r["pass_ns"], 5, 15, "state pass ns/cycle")
band("M4", "M", lambda r: r["from_copy"], 0.90, 1.15, "P=1 from a copy / sequential", 3)
for p, lo, hi in ((2, 1.8, 2.0), (4, 3.2, 3.9), (8, 5.0, 7.5), (16, 7.0, 12.0)):
    band(f"M5P{p}", "M", lambda r, p=p: r["copy"].get(p), lo, hi, f"speedup alone P={p}")
band("M6", "M", lambda r: r["beside"], 3.5, 6.5, "speedup beside 6 generators P=8")
band("M7", "M", lambda r: r["per_cycle"], 0.22, 0.45, "entries a cycle", 4)
lm = mean_of("M", lambda r: r["largest"])
mech.append(("M7L-M", lm is not None and lm <= 6e6, f"median largest window's entries {fmt(lm, 0)} <= 6 M"))
band("M8", "I", lambda r: r["before"], 0.05, 0.4, "setup before (s)", 3)
m8m = band("M8", "M", lambda r: r["before"], 0.5, 1.2, "setup before (s)", 3)
img_m = mean_of("M", lambda r: r["image_before"])
for name, ok, text in land:
    print(f"ROW {name} LANDING {'IN' if ok else 'OUT'}: {text}")
for name, ok, text in mech:
    print(f"ROW {name} MECHANISM {'IN' if ok else 'OUT'}: {text}")
lo_ = [n for n, ok, _ in land if not ok]
mo_ = [n for n, ok, _ in mech if not ok]
walk = ("GO" if not lo_ else "STOP (L4: a bandwidth-bound walk)" if lo_ == ["L4"] else
        "RE-PLAN (L3: patch the entry state up after the walk)" if lo_ == ["L3"] else "NOT DECIDED (" + " ".join(lo_) + ")")
setup = "GO" if (m8m is not None and m8m >= 0.5) or (img_m is not None and img_m >= 0.3) else "DROP"
print(f"VERDICT: PARALLEL WALK LANDING {'PASS' if not lo_ else 'FAIL (' + ' '.join(lo_) + ')'} · MECHANISM "
      f"{'as registered' if not mo_ else 'NOT as registered (' + ' '.join(mo_) + ')'} · walk in parallel {walk} · setup "
      f"change {setup} (median setup before {fmt(m8m, 3)} s, image + seeding {fmt(img_m, 3)} s, saved "
      f"{fmt(mean_of('M', lambda r: r['saved']), 3)} s) · median P=8 alone {fmt(mean_of('M', lambda r: r['copy'].get(8)))} "
      f"beside {fmt(mean_of('M', lambda r: r['beside']))} · pass / executor ns {fmt(mean_of('M', lambda r: r['pass_ns']))} / "
      f"{fmt(mean_of('M', lambda r: r['exec_ns']))}")
