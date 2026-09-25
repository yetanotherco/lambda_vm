#!/usr/bin/env python3
"""Run B's launch plan: which launches of which kernel ncu profiles, and whether it got them.

ncu selects launches by a kernel's own launch index (-k NAME -s SKIP -c COUNT); it has no
grid-size filter. So each pass is a window [SKIP, SKIP+COUNT) of that kernel's launches,
chosen for its launch SHAPES, and the plan carries the exact shape sequence it expects:

    derive   PLAN_IN.tsv TRACE.sqlite > PLAN.tsv  fill in each window's expected shapes (and
                                                  its t_ref) from a trace: how the default
                                                  plan in block_profile.sh was made
    anchor   PLAN.tsv TRACE.sqlite > PLAN.tsv     re-find each window in THIS box's trace
                                                  (run A's): the nearest index whose shapes
                                                  match the expected sequence exactly
    verify   PLAN.tsv RAW_DIR > VERIFY.tsv        after run B: did each pass profile the
                                                  shapes it was aimed at (ncu raw exports)?
    keep     PLAN.tsv PASS FILE...                a SELECT pass: filter its exports in place to
                                                  the launches its predicate keeps (below)
    mode     PLAN.tsv PASS                        prints `select` or `window`

SELECT PASSES. An ordinal window is only as good as the launch order it was taken from, and in
the tree stages sibling proofs run concurrently, so the Nth launch of a kernel there is not the
same launch from one run to the next (09-25: `merkle_wide` aimed at 12044 got grids 32 and 16).
A pass whose shapes field is a predicate

    select:gridx>=N,min=M

selects by SHAPE instead: run B runs it under `ncu --filter-mode per-launch-config`, where
-s/-c apply to each distinct launch configuration (grid, block, shared memory) separately, so
the pass profiles the first `count` launches (after `skip`) of EVERY configuration the kernel
is launched with — the wide ones included, wherever they fall in the run. `keep` then drops the
launches with gridDim.x < N from the pass's exports, and `verify` passes it only when at least
M launches with gridDim.x >= N were profiled (else MISMATCH, which run B turns into a FAIL).
A select pass has no ordinal, so derive and anchor leave it as it is.

Plan columns (tab-separated): pass, kernel, skip, count, t_ref_s, shapes, note, target.
kernel is matched as the regex ^kernel$. shapes = ';'-joined "gx,gy,gz/bx,by,bz", one per
launch of the window, in launch order. target is the binary the pass runs: `block` (the
default: the WHIR block test) or another name the driver knows (e.g. `grind_counted`); only
`block` windows are re-anchored, since only the block run is in run A's trace.
The launch order is the host launch order (CUPTI runtime start), which is what ncu counts.
Pure standard library. `--selftest` with no trace runs on a synthetic one.
"""
import csv
import os
import re
import sqlite3
import sys

COLS = ["pass", "kernel", "skip", "count", "t_ref_s", "shapes", "note", "target"]


def read_plan(path):
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if parts[0] == "pass":
                continue
            parts += [""] * (len(COLS) - len(parts))
            rows.append({c: ("" if v == "-" else v) for c, v in zip(COLS, parts)})
    return rows


def write_plan(rows, out=None):
    """Never an empty field: a shell `IFS=$'\\t' read` collapses runs of tabs, so '-' stands in."""
    out = out or sys.stdout  # resolved per call: a default of sys.stdout would bind it at definition
    out.write("\t".join(COLS) + "\n")
    for r in rows:
        vals = [str(r.get(c) or ("block" if c == "target" else "-")) for c in COLS]
        out.write("\t".join(v.replace("\t", " ") for v in vals) + "\n")


SELECT_RE = re.compile(r"^select:gridx>=(\d+),min=(\d+)$")


def select_spec(shapes):
    """(min_gridx, min_launches) for a select pass, else None."""
    m = SELECT_RE.match(shapes or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def gridx(shape):
    """'16384,1,1/128,1,1' -> 16384."""
    return int(shape.split(",", 1)[0])


def add_note(r, text):
    r["note"] = "{} | {}".format(r["note"], text) if r.get("note") else text


def launches(db, kernel):
    """[(start_ns, 'gx,gy,gz/bx,by,bz')] of one kernel, in host launch order."""
    name_col = "shortName"
    q = ("SELECT k.start, k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, k.blockZ "
         "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = k.{} "
         "LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = k.correlationId "
         "WHERE s.value = ? ORDER BY coalesce(r.start, k.start)").format(name_col)
    return [(st, "{},{},{}/{},{},{}".format(gx, gy, gz, bx, by, bz))
            for st, gx, gy, gz, bx, by, bz in db.execute(q, (kernel,))]


def connect(path):
    return sqlite3.connect("file:{}?mode=ro".format(os.path.abspath(path)), uri=True)


def derive(plan_path, sqlite_path):
    db = connect(sqlite_path)
    rows = read_plan(plan_path)
    for r in rows:
        if select_spec(r["shapes"]):
            add_note(r, "derive: select pass, no window to derive")
            continue
        seq = launches(db, r["kernel"])
        s, c = int(r["skip"]), int(r["count"])
        win = seq[s:s + c]
        if len(win) < c:
            add_note(r, "derive: only {} of {} launches exist".format(len(win), c))
        r["shapes"] = ";".join(sh for _, sh in win)
        r["t_ref_s"] = "{:.1f}".format(win[-1][0] / 1e9) if win else ""
    write_plan(rows)
    return 0


def anchor(plan_path, sqlite_path):
    db = connect(sqlite_path)
    rows = read_plan(plan_path)
    cache = {}
    for r in rows:
        if (r.get("target") or "block") != "block":
            add_note(r, "anchor: not a block-run window (target {}), kept".format(r["target"]))
            continue
        if select_spec(r["shapes"]):
            add_note(r, "anchor: select pass (per launch configuration), no ordinal to anchor")
            continue
        k = r["kernel"]
        if k not in cache:
            cache[k] = launches(db, k)
        seq = cache[k]
        want = r["shapes"].split(";") if r["shapes"] else []
        s, c = int(r["skip"]), int(r["count"])
        if not want or not seq:
            add_note(r, "anchor: {} (kept skip {})".format("no expected shapes" if not want else "kernel absent from this trace", s))
            continue
        shapes = [sh for _, sh in seq]
        found = None
        for d in range(0, len(shapes) + 1):
            for j in ((s + d, s - d) if d else (s,)):
                if 0 <= j <= len(shapes) - c and shapes[j:j + c] == want:
                    found = j
                    break
            if found is not None:
                break
        if found is None:
            add_note(r, "anchor: expected shapes not found in this trace (kept skip {})".format(s))
        else:
            add_note(r, "anchor: skip {} -> {}".format(s, found) if found != s else "anchor: skip {} confirmed".format(s))
            r["skip"] = str(found)
            r["t_ref_s"] = "{:.1f}".format(seq[found + c - 1][0] / 1e9)
    write_plan(rows)
    return 0


def norm_dim(v):
    """'(16384, 1, 1)' -> '16384,1,1'."""
    return ",".join(re.findall(r"\d+", v or ""))


def raw_shapes(path):
    """Shapes of the launches in one `ncu --page raw --csv` export, in ID order."""
    with open(path, newline="", errors="replace") as f:
        rdr = list(csv.reader(ln for ln in f if not ln.startswith("==")))
    hdr_i = next((i for i, r in enumerate(rdr) if "ID" in r and "Grid Size" in r and "Block Size" in r), None)
    if hdr_i is None:
        return None
    hdr = rdr[hdr_i]
    ci, cg, cb = hdr.index("ID"), hdr.index("Grid Size"), hdr.index("Block Size")
    out = []
    for r in rdr[hdr_i + 1:]:
        if len(r) <= max(ci, cg, cb) or not r[ci].strip().isdigit():
            continue  # the units row, or a truncated line
        out.append((int(r[ci]), "{}/{}".format(norm_dim(r[cg]), norm_dim(r[cb]))))
    return [sh for _, sh in sorted(out)]


def verify(plan_path, raw_dir):
    rows = read_plan(plan_path)
    w = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
    w.writerow(["pass", "kernel", "verdict", "profiled", "expected", "got"])
    bad = 0
    for r in rows:
        want = r["shapes"].split(";") if r["shapes"] else []
        path = os.path.join(raw_dir, r["pass"] + ".raw.csv")
        got = raw_shapes(path) if os.path.exists(path) else None
        spec = select_spec(r["shapes"])
        if got is None:
            verdict = "unverified (no raw export with Grid/Block Size)"
        elif spec:
            wide = sum(1 for sh in got if gridx(sh) >= spec[0])
            verdict = "ok" if wide >= spec[1] else "MISMATCH ({} launch(es) with gridx >= {}, {} needed)".format(wide, spec[0], spec[1])
        elif got == want:
            verdict = "ok"
        elif got and got == want[:len(got)]:
            verdict = "partial ({} of {} launches, shapes as planned)".format(len(got), len(want))
        else:
            verdict = "MISMATCH"
        if verdict != "ok":
            bad += 1
        w.writerow([r["pass"], r["kernel"], verdict, "" if got is None else len(got), ";".join(want),
                    "" if got is None else ";".join(got)])
    return 0 if bad == 0 else 1


def find_row(plan_path, pass_name):
    for r in read_plan(plan_path):
        if r["pass"] == pass_name:
            return r
    return None


def keep_csv(path, min_gx):
    """Filter a --csv export (raw or details) to rows whose Grid Size has x >= min_gx. -> kept IDs."""
    with open(path, newline="", errors="replace") as f:
        lines = f.read().splitlines(True)
    pre = [ln for ln in lines if ln.startswith("==")]
    body = [ln for ln in lines if not ln.startswith("==")]
    rdr = list(csv.reader(body))
    hdr_i = next((i for i, r in enumerate(rdr) if "ID" in r and "Grid Size" in r), None)
    if hdr_i is None:
        return None
    hdr = rdr[hdr_i]
    ci, cg = hdr.index("ID"), hdr.index("Grid Size")
    out, kept = rdr[:hdr_i + 1], set()
    for r in rdr[hdr_i + 1:]:
        if len(r) <= max(ci, cg) or not r[ci].strip().isdigit():
            out.append(r)  # the units row
            continue
        dims = re.findall(r"\d+", r[cg])
        if dims and int(dims[0]) >= min_gx:
            out.append(r)
            kept.add(int(r[ci]))
    with open(path, "w", newline="") as f:
        f.writelines(pre)
        csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator="\n").writerows(out)
    return kept


DETAILS_HDR = re.compile(r"^  \S.* \((\d+), \d+, \d+\)x\(\d+, \d+, \d+\), Context ")


def keep_details(path, min_gx):
    """Filter a --page details text to the launch blocks with grid x >= min_gx. -> blocks kept."""
    with open(path, errors="replace") as f:
        lines = f.readlines()
    out, keep, n, seen = [], True, 0, False
    for ln in lines:
        m = DETAILS_HDR.match(ln)
        if m:
            seen = True
            keep = int(m.group(1)) >= min_gx
            n += keep
        if keep or not seen:
            out.append(ln)
    with open(path, "w") as f:
        f.writelines(out)
    return n


def keep(plan_path, pass_name, files):
    r = find_row(plan_path, pass_name)
    spec = select_spec(r["shapes"]) if r else None
    if not spec:
        sys.stderr.write("keep: {} is not a select pass of {}\n".format(pass_name, plan_path))
        return 2
    for p in files:
        if not os.path.exists(p):
            continue
        n = keep_details(p, spec[0]) if p.endswith(".txt") else keep_csv(p, spec[0])
        print("{}: kept {} launch(es) with gridx >= {}".format(p, n if isinstance(n, int) else len(n or ()), spec[0]))
    return 0


def mode(plan_path, pass_name):
    r = find_row(plan_path, pass_name)
    print("select" if r and select_spec(r["shapes"]) else "window")
    return 0


def synthetic_trace(path):
    """A minimal nsys-sqlite with the three tables `launches` reads, shaped like the block run:
    sumcheck rounds, then Merkle trees whose widest level is 16384 or 8192."""
    db = sqlite3.connect(path)
    db.executescript(
        "CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);"
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INT, correlationId INT, shortName INT,"
        " gridX INT, gridY INT, gridZ INT, blockX INT, blockY INT, blockZ INT);"
        "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INT, correlationId INT);")
    db.execute("INSERT INTO StringIds VALUES (1, 'sumcheck_round_ext3'), (2, 'rpx_merkle_level')")
    t, cid = 0, 0

    def add(name_id, g, b):
        nonlocal t, cid
        t += 1000
        cid += 1
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?,?,?)",
                   (t + 50, cid, name_id, g[0], g[1], g[2], b[0], b[1], b[2]))
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?)", (t, cid))
    for _ in range(340):
        add(1, (512, 3, 1), (32, 1, 1))
    for g in [4096] * 5 + [2048]:
        add(1, (g, 1, 1), (256, 1, 1))
    for top in [512, 8192, 16384, 16384]:
        g = top
        while g >= 1:
            add(2, (g, 1, 1), (128, 1, 1))
            g //= 2
    db.commit()
    db.close()


def selftest(sqlite_path=None):
    """On a real trace: derive, anchor after a deliberate shift, and verify both ways."""
    import tempfile
    d = tempfile.mkdtemp(prefix="ncu_plan_selftest.")
    synthetic = sqlite_path is None
    if synthetic:
        sqlite_path = os.path.join(d, "synthetic.sqlite")
        synthetic_trace(sqlite_path)
    # real trace: the block run's windows; synthetic: the same kinds of window in its launches
    # (merkle trees 512.., 8192.., 16384.., 16384..: the first 16384 is launch 10 + 14 = 24)
    sc_skip, sc_n, ml_skip = (340, 18, 12493) if not synthetic else (340, 6, 24)
    p_in = os.path.join(d, "in.tsv")
    with open(p_in, "w") as f:
        f.write("pass\tkernel\tskip\tcount\tt_ref_s\tshapes\n")
        f.write("sc_a\tsumcheck_round_ext3\t{}\t{}\n".format(sc_skip, sc_n))
        f.write("ml_b\trpx_merkle_level\t{}\t2\n".format(ml_skip))
        f.write("ml_sel\trpx_merkle_level\t0\t2\t110\tselect:gridx>=8192,min=2\n")
    import io
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    derive(p_in, sqlite_path)
    sys.stdout = old
    derived = buf.getvalue()
    plan = os.path.join(d, "plan.tsv")
    with open(plan, "w") as f:
        f.write(derived)
    rows = read_plan(plan)
    checks = [("derive: {} shapes for sc_a".format(sc_n), len(rows[0]["shapes"].split(";")), sc_n),
              ("derive: sc_a starts at the 2^21 round", rows[0]["shapes"].split(";")[0], "4096,1,1/256,1,1"),
              ("derive: ml_b is the widest level", rows[1]["shapes"].split(";")[0], "16384,1,1/128,1,1"),
              ("derive: a select pass is left as it is", rows[2]["shapes"], "select:gridx>=8192,min=2")]
    # shift both skips, as another box's index drift would, and let anchor find them again
    shifted = os.path.join(d, "shifted.tsv")
    with open(shifted, "w") as f:
        for r in rows:
            if not select_spec(r["shapes"]):
                r["skip"] = str(int(r["skip"]) + 7)
        write_plan(rows, f)
    buf = io.StringIO()
    sys.stdout = buf
    anchor(shifted, sqlite_path)
    sys.stdout = old
    anchored = os.path.join(d, "anchored.tsv")
    with open(anchored, "w") as f:
        f.write(buf.getvalue())
    ar = read_plan(anchored)
    checks += [("anchor: sc_a back to {}".format(sc_skip), ar[0]["skip"], str(sc_skip)),
               ("anchor: ml_b back to {}".format(ml_skip), ar[1]["skip"], str(ml_skip)),
               ("anchor: a select pass keeps skip 0", ar[2]["skip"], "0"),
               ("mode: ml_sel is a select pass", _capture(mode, anchored, "ml_sel").strip(), "select"),
               ("mode: ml_b is a window pass", _capture(mode, anchored, "ml_b").strip(), "window")]
    # verify against raw exports: one matching, one not
    raw = os.path.join(d, "raw")
    os.makedirs(raw)

    def fake_raw(name, shapes):
        with open(os.path.join(raw, name + ".raw.csv"), "w", newline="") as f:
            w = csv.writer(f, quoting=csv.QUOTE_ALL)
            w.writerow(["ID", "Kernel Name", "Grid Size", "Block Size", "gpu__time_duration.sum"])
            w.writerow(["", "", "", "", "usecond"])
            for i, sh in enumerate(shapes):
                g, b = sh.split("/")
                w.writerow([str(i), "k", "({})".format(", ".join(g.split(","))), "({})".format(", ".join(b.split(","))), "1"])
    fake_raw("sc_a", ar[0]["shapes"].split(";"))
    fake_raw("ml_b", ["8192,1,1/128,1,1", "16384,1,1/128,1,1"])
    # the select pass as per-launch-config profiles it: two of every configuration, 09-25's
    # failure shape (only narrow ones) and the good one (wide ones present)
    sel_all = ["{},1,1/128,1,1".format(g) for g in (512, 256, 16384, 8192, 32, 16) for _ in range(2)]
    fake_raw("ml_sel", ["32,1,1/128,1,1", "16,1,1/128,1,1"])
    rc_narrow, ver_n = _verify(anchored, raw)
    fake_raw("ml_sel", sel_all)
    rc, ver = _verify(anchored, raw)
    checks += [("verify: sc_a ok", ver["sc_a"]["verdict"], "ok"), ("verify: ml_b MISMATCH", ver["ml_b"]["verdict"], "MISMATCH"),
               ("verify: rc 1 on a mismatch", rc, 1),
               ("verify: select pass with only narrow launches fails", ver_n["ml_sel"]["verdict"].split(" ")[0], "MISMATCH"),
               ("verify: select pass with 4 wide launches ok", ver["ml_sel"]["verdict"], "ok")]
    # keep: filter the select pass's raw csv and a details text to the wide launches
    raw_sel = os.path.join(raw, "ml_sel.raw.csv")
    det = os.path.join(d, "ml_sel.details.txt")
    with open(det, "w") as f:
        f.write("[1] proc@host\n")
        for i, sh in enumerate(sel_all):
            g = sh.split("/")[0].split(",")
            f.write("  rpx_merkle_level ({}, {}, {})x(128, 1, 1), Context 1, Stream 7, Device 0, CC 12.0\n".format(*g))
            f.write("    Section: x\n    Duration ms {}\n".format(i))
    _capture(keep, anchored, "ml_sel", [raw_sel, det])
    kept = raw_shapes(raw_sel)
    with open(det) as f:
        det_txt = f.read()
    rc_after, ver_after = _verify(anchored, raw)
    checks += [("keep: raw csv holds the 4 wide launches only", kept, ["16384,1,1/128,1,1"] * 2 + ["8192,1,1/128,1,1"] * 2),
               ("keep: details text holds 4 blocks", det_txt.count("rpx_merkle_level ("), 4),
               ("keep: details text keeps its preamble", det_txt.startswith("[1] proc@host"), True),
               ("keep: no narrow block left", "(512, 1, 1)x" in det_txt or "(32, 1, 1)x" in det_txt, False),
               ("verify after keep: still ok", ver_after["ml_sel"]["verdict"], "ok")]
    bad = [c for c in checks if c[1] != c[2]]
    for n, got, want in checks:
        print("{} {:<40} got {!r}".format("ok  " if got == want else "FAIL", n, got))
    print("SELFTEST {} ({} checks, {} failed)".format("PASS" if not bad else "FAIL", len(checks), len(bad)))
    return 0 if not bad else 1


def _capture(fn, *args):
    import io
    buf, old = io.StringIO(), sys.stdout
    sys.stdout = buf
    try:
        fn(*args)
    finally:
        sys.stdout = old
    return buf.getvalue()


def _verify(plan, raw):
    import io
    buf, old = io.StringIO(), sys.stdout
    sys.stdout = buf
    try:
        rc = verify(plan, raw)
    finally:
        sys.stdout = old
    return rc, {r["pass"]: r for r in csv.DictReader(io.StringIO(buf.getvalue()), delimiter="\t")}


def main(argv):
    if len(argv) == 3 and argv[0] == "derive":
        return derive(argv[1], argv[2])
    if len(argv) == 3 and argv[0] == "anchor":
        return anchor(argv[1], argv[2])
    if len(argv) == 3 and argv[0] == "verify":
        return verify(argv[1], argv[2])
    if len(argv) >= 4 and argv[0] == "keep":
        return keep(argv[1], argv[2], argv[3:])
    if len(argv) == 3 and argv[0] == "mode":
        return mode(argv[1], argv[2])
    if len(argv) in (1, 2) and argv[0] == "--selftest":
        return selftest(argv[1] if len(argv) == 2 else None)
    sys.stderr.write(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
