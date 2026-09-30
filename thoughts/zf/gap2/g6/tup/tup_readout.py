#!/usr/bin/env python3
"""TRACE-UPLOAD readout: the default (A: one stream of pageable column copies, DECODE's prepared opening before the
pipeline) against LAMBDA_VM_TRACE_UPLOAD=1 (B: the columns staged from 4 threads with their zero tails left behind,
the opening on a helper beside epoch 0), on #1010's production tree at fix2/1010-trace-upload.

usage: python3 tup_readout.py <manifest.tsv>          (on the box; the harness's manifest)
       python3 tup_readout.py NAME:log ...             (copied logs; NAME is A or B)
       python3 tup_readout.py --selftest <log>         (laptop: a real default log, and a B copy made from it)

Pre-registered in G6-LEDGER.md §7 before the run. Standard library only. One row per check, then a final
`READOUT:` line.
"""
import re
import sys

F = r"([0-9]+(?:\.[0-9]+)?)"
UPLOAD = re.compile(r"^COLUMNS UPLOAD: " + F + r" GB in " + F + r"s, " + F + r" GB zero tails not sent, " + F
                    + r" GB/s sent \((pageable|staged x([0-9]+))\) t=" + F)
HEAD_START = re.compile(r"^BASE HEAD \(WHIR\): start t=" + F)
EXEC0 = re.compile(r"^BASE EPOCH 0: execute " + F + r"s t=\[" + F + "," + F + r"\]")
COMMIT0 = re.compile(r"^BASE EPOCH 0: commit " + F + r"s t=\[" + F + "," + F + r"\]")
L1_START = re.compile(r"LEVEL 1 \(wide\) START t=" + F)
WAITED = re.compile(r"^BASE HEAD \(WHIR\): the first prove waited " + F + r"s for DECODE prepared")
AHEAD = "BASE HEAD (WHIR): DECODE prepared (ahead)"
SERIAL = re.compile(r"^BASE HEAD \(WHIR\): DECODE prepared " + F + "s")
IDS = re.compile(r"IDENTITY: program_id")

# Pre-registered (G6-LEDGER.md §7). B − A unless named; seconds, GB (1e9 bytes).
D_WHOLE = (-1.8, -0.3)        # point −1.0
D_BASE = (-1.8, -0.3)
A_UPLOAD = (1.3, 2.3)         # A's Σ upload s in the base; G6 traced 1.84 s of H2D in the base, 1.79 exclusive
B_UPLOAD_MAX = 1.3
D_UPLOAD = (-1.2, -0.3)
B_SKIPPED = (3.0, 4.6)        # GB of zero tails not sent in the base; census: 4.46 GB of row padding
A_TO_EXEC0 = (0.40, 0.70)     # head start → epoch 0 executes; G6 P2 0.543 s
D_TO_EXEC0 = (-0.45, -0.15)   # model −0.36 (decode_prepared_for off the path)
D_TO_COMMIT0 = (-0.45, -0.05)
B_WAITED_MAX = 0.10


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
    r["head_start"], r["l1_start"] = head, l1
    ex = re.search(EXEC0.pattern, text, re.M)
    cm = re.search(COMMIT0.pattern, text, re.M)
    r["to_exec0"] = float(ex.group(2)) - head if ex and head is not None else None
    r["to_commit0"] = float(cm.group(2)) - head if cm and head is not None else None
    ups = []
    for line in text.splitlines():
        m = UPLOAD.match(line)
        if not m:
            continue
        t = float(m.group(7))
        if head is not None and l1 is not None and head <= t <= l1:
            ups.append((float(m.group(1)), float(m.group(2)), float(m.group(3)), m.group(5)))
    r["uploads"] = len(ups)
    r["up_gb"] = sum(u[0] for u in ups)
    r["up_s"] = sum(u[1] for u in ups)
    r["skip_gb"] = sum(u[2] for u in ups)
    r["paths"] = sorted({u[3] for u in ups})
    r["ahead"] = AHEAD in text
    r["waited"] = fnum(WAITED.pattern, text)
    r["serial_prepared"] = fnum(SERIAL.pattern, text)
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
            tag, name, knobs, log = line.rstrip("\n").split("\t")[:4]
            arms.append((tag, name, log))
        return arms
    return [(f"arm{i}", a.split(":", 1)[0], a.split(":", 1)[1]) for i, a in enumerate(argv)]


def readout(arms):
    rows, ok_all = [], True
    reads = [(tag, name, read(log)) for tag, name, log in arms]
    print("| tag | arm | whole s | base s | uploads | GB | upload s | GB not sent | path | head→exec0 | head→commit0 "
          "| prepared | waited |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for tag, name, r in reads:
        print(f"| {tag} | {name} | {fmt(r['whole'], 1)} | {fmt(r['base'], 1)} | {r['uploads']} | {fmt(r['up_gb'], 2)} "
              f"| {fmt(r['up_s'])} | {fmt(r['skip_gb'], 2)} | {','.join(r['paths']) or '-'} | {fmt(r['to_exec0'])} "
              f"| {fmt(r['to_commit0'])} | {'ahead' if r['ahead'] else fmt(r['serial_prepared'], 2)} "
              f"| {fmt(r['waited'], 2)} |")
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
          all(r["paths"] == ["pageable"] and not r["ahead"] for r in A)
          and all(r["paths"] and all(p.startswith("staged") for p in r["paths"]) and r["ahead"] for r in B),
          f"A {[r['paths'] for r in A]} ahead {[r['ahead'] for r in A]} · B {[r['paths'] for r in B]} "
          f"ahead {[r['ahead'] for r in B]}")
    check("proved", all(r["proved"] for r in A + B), "every arm compressed and verified")
    ids = {tuple(r["ids"]) for r in A + B}
    check("program ids identical across settings", len(ids) == 1 and len(next(iter(ids))) == 5,
          f"{len(ids)} distinct id set(s)")
    check("same bytes uploaded", len({round(r['up_gb'], 2) for r in A + B}) == 1,
          f"{sorted({round(r['up_gb'], 2) for r in A + B})} GB")
    a_up, b_up = mean(r["up_s"] for r in A), mean(r["up_s"] for r in B)
    check("A's upload (control)", inside(a_up, A_UPLOAD), f"{fmt(a_up)} s in {A_UPLOAD}")
    check("B's upload", b_up is not None and b_up <= B_UPLOAD_MAX, f"{fmt(b_up)} s ≤ {B_UPLOAD_MAX}")
    d_up = None if a_up is None or b_up is None else b_up - a_up
    check("Δ upload", inside(d_up, D_UPLOAD), f"{fmt(d_up)} s in {D_UPLOAD}")
    b_skip = mean(r["skip_gb"] for r in B)
    check("A sent every byte", all(r["skip_gb"] == 0 for r in A), "0 GB not sent")
    check("B's zero tails not sent", inside(b_skip, B_SKIPPED), f"{fmt(b_skip, 2)} GB in {B_SKIPPED}")
    a_x, b_x = mean(r["to_exec0"] for r in A), mean(r["to_exec0"] for r in B)
    check("A's head → epoch 0 executes (control)", inside(a_x, A_TO_EXEC0), f"{fmt(a_x)} s in {A_TO_EXEC0}")
    d_x = None if a_x is None or b_x is None else b_x - a_x
    check("Δ head → epoch 0 executes", inside(d_x, D_TO_EXEC0), f"{fmt(d_x)} s in {D_TO_EXEC0}")
    a_c, b_c = mean(r["to_commit0"] for r in A), mean(r["to_commit0"] for r in B)
    d_c = None if a_c is None or b_c is None else b_c - a_c
    check("Δ head → epoch 0's commit", inside(d_c, D_TO_COMMIT0), f"{fmt(d_c)} s in {D_TO_COMMIT0}")
    b_w = mean(r["waited"] for r in B)
    check("B's first prove waited for the opening", b_w is not None and b_w <= B_WAITED_MAX,
          f"{fmt(b_w, 2)} s ≤ {B_WAITED_MAX}")
    a_b, b_b = mean(r["base"] for r in A), mean(r["base"] for r in B)
    a_w, b_w2 = mean(r["whole"] for r in A), mean(r["whole"] for r in B)
    d_b = None if a_b is None or b_b is None else b_b - a_b
    d_w = None if a_w is None or b_w2 is None else b_w2 - a_w
    spread = max(r["whole"] for r in A) - min(r["whole"] for r in A) if len(A) > 1 else None
    check("Δ base", inside(d_b, D_BASE), f"{fmt(d_b, 2)} s in {D_BASE}")
    for row in rows:
        print(" · ".join(row))
    print(f"whole: A {fmt(a_w, 2)} · B {fmt(b_w2, 2)} · Δ {fmt(d_w, 2)} (band {D_WHOLE}) · A spread {fmt(spread, 2)}")
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
    return f"READOUT: {verdict} · {tail}", ok_all


def selftest(log):
    """A real default log reads as A. A B copy — its upload lines rewritten staged and faster, the head's opening
    moved — must read as B, and the checks must see both."""
    import os
    import tempfile
    text = open(log, errors="replace").read()
    head = fnum(HEAD_START.pattern, text)
    l1 = fnum(L1_START.pattern, text)
    assert head is not None and l1 is not None, "the log has no base window"
    # The log predates the upload line: give it 16 uploads of 2.2 GB, pageable, at 20.5 GB/s, inside the base.
    lines = text.splitlines()
    a_lines, b_lines = [], []
    for i, line in enumerate(lines):
        a_lines.append(line)
        b_lines.append(line.replace("BASE HEAD (WHIR): DECODE prepared", "BASE HEAD (WHIR): DECODE prepared (ahead)"))
        m = re.match(r"^BASE EPOCH ([0-9]+|global): commit " + F + r"s t=\[" + F, line)
        if m:
            t = float(m.group(3)) + 0.01
            a_lines.append(f"COLUMNS UPLOAD: 2.200 GB in 0.107s, 0.000 GB zero tails not sent, 20.5 GB/s sent "
                           f"(pageable) t={t:.3f}")
            b_lines.append(f"COLUMNS UPLOAD: 2.200 GB in 0.060s, 0.250 GB zero tails not sent, 32.5 GB/s sent "
                           f"(staged x4) t={t:.3f}")
        mx = EXEC0.match(line)
        if mx:
            b_lines[-1] = re.sub(r"t=\[" + F, lambda _: f"t=[{float(mx.group(2)) - 0.3:.3f}", b_lines[-1])
        if line.startswith("BASE HEAD (WHIR): epoch 0's prep waited"):
            b_lines.append("BASE HEAD (WHIR): the first prove waited 0.00s for DECODE prepared")
    d = tempfile.mkdtemp()
    pa, pb = os.path.join(d, "a.log"), os.path.join(d, "b.log")
    open(pa, "w").write("\n".join(a_lines) + "\n")
    open(pb, "w").write("\n".join(b_lines) + "\n")
    ra, rb = read(pa), read(pb)
    assert ra["uploads"] == 16 and ra["paths"] == ["pageable"] and not ra["ahead"], ra
    assert rb["uploads"] == 16 and rb["paths"] == ["staged x4"] and rb["ahead"], rb
    assert abs(ra["up_s"] - 16 * 0.107) < 1e-6 and abs(rb["skip_gb"] - 4.0) < 1e-6
    assert rb["waited"] == 0.0
    assert abs((rb["to_exec0"] - ra["to_exec0"]) + 0.3) < 1e-3, (ra["to_exec0"], rb["to_exec0"])
    line, _ = readout([("a1", "A", pa), ("b1", "B", pb), ("b2", "B", pb), ("a2", "A", pa)])
    print(line)
    assert "each arm ran its own path" not in line and "Δ upload" not in line, line
    # A B arm that ran pageable must be caught.
    line2, ok2 = readout([("a1", "A", pa), ("b1", "B", pa)])
    print(line2)
    assert not ok2 and "each arm ran its own path" in line2, line2
    print("SELFTEST OK")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--selftest":
        selftest(sys.argv[2])
        sys.exit(0)
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    line, ok = readout(arms_from(sys.argv[1:]))
    print(line)
    sys.exit(0 if ok else 2)
