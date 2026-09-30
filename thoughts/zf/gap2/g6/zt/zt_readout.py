#!/usr/bin/env python3
"""ZERO-TAIL readout: the default (A: every column byte sent, one stream of pageable copies) against
LAMBDA_VM_TRACE_UPLOAD=zerotail (B: each column's all-zero tail of >= 64 KiB left behind and zeroed on the card, the
rest the same pageable copies) on #1010's production tree at fix2/1010-zerotail, A B B A A B B A.

usage: python3 zt_readout.py <manifest.tsv>           (on the box; the harness's manifest)
       python3 zt_readout.py NAME:log ...              (copied logs; NAME is A or B)
       python3 zt_readout.py --selftest <A-log> <B-log> (laptop: job 290's logs, B's staged lines rewritten)

Pre-registered in G6-LEDGER.md §8 before the run. Standard library only. One row per check, then a final
`READOUT:` line.
"""
import re
import sys

F = r"([0-9]+(?:\.[0-9]+)?)"
UPLOAD = re.compile(r"^COLUMNS UPLOAD: " + F + r" GB in " + F + r"s, " + F + r" GB zero tails not sent, " + F
                    + r" GB/s sent \((pageable|zerotail, scan on the path " + F + r"s)\) t=" + F)
HEAD_START = re.compile(r"^BASE HEAD \(WHIR\): start t=" + F)
COMMIT0 = re.compile(r"^BASE EPOCH 0: commit " + F + r"s t=\[" + F + "," + F + r"\]")
L1_START = re.compile(r"LEVEL 1 \(wide\) START t=" + F)
L1N2 = re.compile(r"L1N2 \(arity 5\) TIMING: .*harvest-epochs " + F + r"s · emit\+arenas " + F + r"s · artifacts "
                  + F + r"s · prove " + F + r"s .*wall " + F + "s")
IDS = re.compile(r"IDENTITY: program_id")

# Pre-registered (G6-LEDGER.md §8). B − A unless named; seconds, GB (1e9 bytes).
D_WHOLE = (-0.45, -0.10)      # point −0.3
D_BASE = (-0.55, -0.15)       # point −0.4
A_UPLOAD = (1.5, 2.1)         # A's Σ upload in the base; job 290's A arms 1.791 / 1.781 s
B_SKIPPED = (7.70, 7.95)      # GB not sent in the base; job 290's B arms 7.82 / 7.82
D_UPLOAD = (-0.50, -0.25)     # 7.82 GB at A's 18.9 GB/s = −0.41 s
B_RATE_MIN = 17.0             # GB/s of sent bytes; the copies stay pageable-fast (A 18.9–19.0)
B_SCAN_MAX = 0.03             # s scanned on the uploading thread, Σ over the base
D_COMMIT0 = (-0.10, 0.10)     # the head is untouched
L1N2_HARVEST_MAX = 2.4        # an arm above this repeats wt1291's anomaly (2.68 s; others 2.01–2.13)
L1N2_ARTIFACTS_MAX = 0.6      # (1.08 s; others 0.27–0.29)


def fnum(rx, text, group=1):
    m = re.search(rx, text, re.M)
    return float(m.group(group)) if m else None


def read(path):
    text = open(path, errors="replace").read()
    r = {"path": path}
    r["whole"] = fnum(r"WHOLE RUN: host peak " + F + r" GiB at t=" + F + r", " + F + r"s total", text, 3)
    r["base"] = fnum(r"base \(WHIR\): [0-9]+ epochs in " + F + "s", text)
    head = fnum(HEAD_START.pattern, text)
    l1 = fnum(L1_START.pattern, text)
    cm = re.search(COMMIT0.pattern, text, re.M)
    r["to_commit0"] = float(cm.group(2)) - head if cm and head is not None else None
    base_up, rest_up = [], []
    for line in text.splitlines():
        m = UPLOAD.match(line)
        if not m:
            continue
        rec = (float(m.group(1)), float(m.group(2)), float(m.group(3)), m.group(5).split(",")[0],
               float(m.group(6)) if m.group(6) else 0.0)
        t = float(m.group(7))
        (base_up if head is not None and l1 is not None and head <= t <= l1 else rest_up).append(rec)
    r["uploads"], r["rest_uploads"] = len(base_up), len(rest_up)
    r["up_gb"] = sum(u[0] for u in base_up)
    r["up_s"] = sum(u[1] for u in base_up)
    r["skip_gb"] = sum(u[2] for u in base_up)
    r["scan_s"] = sum(u[4] for u in base_up)
    r["rest_s"] = sum(u[1] for u in rest_up)
    r["rest_skip_gb"] = sum(u[2] for u in rest_up)
    r["paths"] = sorted({u[3] for u in base_up + rest_up})
    r["rate"] = (r["up_gb"] - r["skip_gb"]) / r["up_s"] if r["up_s"] > 0 else None
    m = re.search(L1N2.pattern, text, re.M)
    r["l1n2"] = tuple(float(m.group(i)) for i in (1, 3, 4, 5)) if m else None
    r["ids"] = sorted(l.strip() for l in text.splitlines() if IDS.search(l))
    r["proved"] = "THE BLOCK IS COMPRESSED UNDER WHIR" in text or "PROVED AND VERIFIED" in text
    return r


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def inside(x, band):
    return x is not None and band[0] <= x <= band[1]


def fmt(x, nd=3):
    return "NA" if x is None else f"{x:.{nd}f}"


def arms_from(argv):
    if len(argv) == 1 and argv[0].endswith(".tsv"):
        arms = []
        for line in open(argv[0]):
            if line.startswith("#") or not line.strip():
                continue
            tag, name, _knobs, log = line.rstrip("\n").split("\t")[:4]
            arms.append((tag, name, log))
        return arms
    return [(f"arm{i}", a.split(":", 1)[0], a.split(":", 1)[1]) for i, a in enumerate(argv)]


def readout(arms):
    rows, ok_all = [], True
    reads = [(tag, name, read(log)) for tag, name, log in arms]
    print("| tag | arm | whole s | base s | uploads | GB | upload s | GB not sent | GB/s sent | scan on path s "
          "| after-base upload s (not sent GB) | path | head→commit0 | L1N2 harvest / artifacts / prove / wall |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for tag, name, r in reads:
        l = r["l1n2"]
        l1n2 = "NA" if l is None else f"{l[0]:.2f} / {l[1]:.2f} / {l[2]:.2f} / {l[3]:.2f}"
        print(f"| {tag} | {name} | {fmt(r['whole'], 1)} | {fmt(r['base'], 1)} | {r['uploads']} | {fmt(r['up_gb'], 2)} "
              f"| {fmt(r['up_s'])} | {fmt(r['skip_gb'], 2)} | {fmt(r['rate'], 1)} | {fmt(r['scan_s'])} "
              f"| {fmt(r['rest_s'])} ({fmt(r['rest_skip_gb'], 2)}) | {','.join(r['paths']) or '-'} "
              f"| {fmt(r['to_commit0'])} | {l1n2} |")
    A = [r for _, n, r in reads if n == "A"]
    B = [r for _, n, r in reads if n == "B"]

    def check(label, ok, detail):
        nonlocal ok_all
        ok_all &= bool(ok)
        rows.append((label, "PASS" if ok else "FAIL", detail))

    check("arms", len(A) >= 1 and len(B) >= 1, f"A {len(A)} · B {len(B)}")
    if not (A and B):
        for row in rows:
            print(" · ".join(row))
        return "READOUT: INCOMPLETE — an arm setting is missing", False
    check("each arm ran its own path",
          all(r["paths"] == ["pageable"] for r in A) and all(r["paths"] == ["zerotail"] for r in B),
          f"A {[r['paths'] for r in A]} · B {[r['paths'] for r in B]}")
    check("proved", all(r["proved"] for r in A + B), "every arm compressed and verified")
    ids = {tuple(r["ids"]) for r in A + B}
    check("program ids identical across settings", len(ids) == 1 and len(next(iter(ids))) == 5,
          f"{len(ids)} distinct id set(s)")
    check("same bytes to upload", len({round(r['up_gb'], 2) for r in A + B}) == 1,
          f"{sorted({round(r['up_gb'], 2) for r in A + B})} GB")
    a_up, b_up = mean(r["up_s"] for r in A), mean(r["up_s"] for r in B)
    check("A's upload (control)", inside(a_up, A_UPLOAD), f"{fmt(a_up)} s in {A_UPLOAD}")
    check("A sent every byte", all(r["skip_gb"] == 0 for r in A), "0 GB not sent")
    b_skip = mean(r["skip_gb"] for r in B)
    check("B's zero tails not sent", all(inside(r["skip_gb"], B_SKIPPED) for r in B),
          f"{[round(r['skip_gb'], 2) for r in B]} GB in {B_SKIPPED}")
    d_up = None if a_up is None or b_up is None else b_up - a_up
    check("Δ upload", inside(d_up, D_UPLOAD), f"{fmt(d_up)} s in {D_UPLOAD}")
    b_rate = mean(r["rate"] for r in B)
    check("B's sent rate stays pageable-fast", b_rate is not None and b_rate >= B_RATE_MIN,
          f"{fmt(b_rate, 1)} GB/s ≥ {B_RATE_MIN} (A {fmt(mean(r['rate'] for r in A), 1)})")
    b_scan = max(r["scan_s"] for r in B)
    check("B scanned off the path", b_scan <= B_SCAN_MAX, f"max {fmt(b_scan)} s ≤ {B_SCAN_MAX}")
    a_c, b_c = mean(r["to_commit0"] for r in A), mean(r["to_commit0"] for r in B)
    d_c = None if a_c is None or b_c is None else b_c - a_c
    check("Δ head → epoch 0's commit (untouched)", inside(d_c, D_COMMIT0), f"{fmt(d_c)} s in {D_COMMIT0}")
    a_b, b_b = mean(r["base"] for r in A), mean(r["base"] for r in B)
    d_b = None if a_b is None or b_b is None else b_b - a_b
    check("Δ base", inside(d_b, D_BASE), f"{fmt(d_b, 2)} s in {D_BASE}")
    for row in rows:
        print(" · ".join(row))
    # Level 1's last node: not a gate, a count. Does wt1291's anomaly come back, and on which setting?
    odd = [(tag, name) for tag, name, r in reads if r["l1n2"] is not None
           and (r["l1n2"][0] > L1N2_HARVEST_MAX or r["l1n2"][1] > L1N2_ARTIFACTS_MAX)]
    print(f"L1N2 anomaly (harvest-epochs > {L1N2_HARVEST_MAX} s or artifacts > {L1N2_ARTIFACTS_MAX} s): "
          f"{len(odd)} arm(s) {odd or ''}")
    a_w, b_w = mean(r["whole"] for r in A), mean(r["whole"] for r in B)
    d_w = None if a_w is None or b_w is None else b_w - a_w
    spread_a = max(r["whole"] for r in A) - min(r["whole"] for r in A) if len(A) > 1 else None
    spread_b = max(r["whole"] for r in B) - min(r["whole"] for r in B) if len(B) > 1 else None
    # The two ABBAs apart, in arm order: A B B A | A B B A.
    halves = []
    names = [n for _, n, _ in reads]
    if len(reads) == 8 and names == list("ABBAABBA"):
        for h in (reads[:4], reads[4:]):
            ha = mean(r["whole"] for _, n, r in h if n == "A")
            hb = mean(r["whole"] for _, n, r in h if n == "B")
            halves.append(hb - ha)
    print(f"whole: A {fmt(a_w, 2)} (spread {fmt(spread_a, 2)}) · B {fmt(b_w, 2)} (spread {fmt(spread_b, 2)}) · "
          f"Δ {fmt(d_w, 2)} (band {D_WHOLE})" + (f" · per ABBA {[round(h, 2) for h in halves]}" if halves else ""))
    if d_w is None:
        verdict = "INCOMPLETE — no whole-run wall"
    elif d_w <= D_WHOLE[1]:
        verdict = f"EFFECTIVE Δ whole {d_w:+.2f} s" + ("" if inside(d_w, D_WHOLE) else " (beyond the band)")
    elif d_w >= 0.3:
        verdict = f"REGRESSION Δ whole {d_w:+.2f} s"
    else:
        verdict = f"NO EFFECT Δ whole {d_w:+.2f} s"
    failed = [label for label, v, _ in rows if v == "FAIL"]
    tail = "all mechanism checks PASS" if not failed else "FAIL: " + "; ".join(failed)
    return f"READOUT: {verdict} · {tail} · L1N2 anomaly {len(odd)} arm(s)", ok_all


def selftest(a_log, b_log):
    """Job 290's A log reads as A. Its B log, with the staged lines rewritten as zerotail lines (same bytes,
    faster), reads as B. The checks see both, and a B arm that ran pageable is caught."""
    import os
    import tempfile
    a_text = open(a_log, errors="replace").read()
    b_text = open(b_log, errors="replace").read()
    b_text = re.sub(r"\(staged x4\)", "(zerotail, scan on the path 0.001s)", b_text)
    d = tempfile.mkdtemp()
    pa, pb = os.path.join(d, "a.log"), os.path.join(d, "b.log")
    open(pa, "w").write(a_text)
    open(pb, "w").write(b_text)
    ra, rb = read(pa), read(pb)
    assert ra["uploads"] == 16 and ra["paths"] == ["pageable"], ra
    assert rb["uploads"] == 16 and rb["paths"] == ["zerotail"], rb
    assert abs(rb["skip_gb"] - 7.82) < 0.01, rb["skip_gb"]
    assert abs(rb["scan_s"] - 0.016) < 1e-9, rb["scan_s"]
    assert ra["l1n2"] is not None and rb["l1n2"] is not None
    line, _ = readout([("a1", "A", pa), ("b1", "B", pb), ("b2", "B", pb), ("a2", "A", pa),
                       ("a3", "A", pa), ("b3", "B", pb), ("b4", "B", pb), ("a4", "A", pa)])
    print(line)
    assert "each arm ran its own path" not in line and "B's zero tails not sent" not in line, line
    line2, ok2 = readout([("a1", "A", pa), ("b1", "B", pa)])
    print(line2)
    assert not ok2 and "each arm ran its own path" in line2, line2
    print("SELFTEST OK")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--selftest":
        selftest(sys.argv[2], sys.argv[3])
        sys.exit(0)
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    line, ok = readout(arms_from(sys.argv[1:]))
    print(line)
    sys.exit(0 if ok else 2)
