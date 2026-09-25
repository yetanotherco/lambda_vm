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

Plan columns (tab-separated): pass, kernel, skip, count, t_ref_s, shapes, note, target.
kernel is matched as the regex ^kernel$. shapes = ';'-joined "gx,gy,gz/bx,by,bz", one per
launch of the window, in launch order. target is the binary the pass runs: `block` (the
default: the WHIR block test) or another name the driver knows (e.g. `grind_counted`); only
`block` windows are re-anchored, since only the block run is in run A's trace.
The launch order is the host launch order (CUPTI runtime start), which is what ncu counts.
Pure standard library.
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
        if got is None:
            verdict = "unverified (no raw export with Grid/Block Size)"
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


def selftest(sqlite_path):
    """On a real trace: derive, anchor after a deliberate shift, and verify both ways."""
    import tempfile
    d = tempfile.mkdtemp(prefix="ncu_plan_selftest.")
    p_in = os.path.join(d, "in.tsv")
    with open(p_in, "w") as f:
        f.write("pass\tkernel\tskip\tcount\n")
        f.write("sc_a\tsumcheck_round_ext3\t340\t18\n")
        f.write("ml_b\trpx_merkle_level\t12493\t2\n")
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
    checks = [("derive: 18 shapes for sc_a", len(rows[0]["shapes"].split(";")), 18),
              ("derive: sc_a starts at the 2^21 round", rows[0]["shapes"].split(";")[0], "4096,1,1/256,1,1"),
              ("derive: ml_b is the widest level", rows[1]["shapes"].split(";")[0], "16384,1,1/128,1,1")]
    # shift both skips, as another box's index drift would, and let anchor find them again
    shifted = os.path.join(d, "shifted.tsv")
    with open(shifted, "w") as f:
        for r in rows:
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
    checks += [("anchor: sc_a back to 340", ar[0]["skip"], "340"), ("anchor: ml_b back to 12493", ar[1]["skip"], "12493")]
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
    buf = io.StringIO()
    sys.stdout = buf
    rc = verify(anchored, raw)
    sys.stdout = old
    ver = {r["pass"]: r for r in csv.DictReader(io.StringIO(buf.getvalue()), delimiter="\t")}
    checks += [("verify: sc_a ok", ver["sc_a"]["verdict"], "ok"), ("verify: ml_b MISMATCH", ver["ml_b"]["verdict"], "MISMATCH"),
               ("verify: rc 1 on a mismatch", rc, 1)]
    bad = [c for c in checks if c[1] != c[2]]
    for n, got, want in checks:
        print("{} {:<40} got {!r}".format("ok  " if got == want else "FAIL", n, got))
    print("SELFTEST {} ({} checks, {} failed)".format("PASS" if not bad else "FAIL", len(checks), len(bad)))
    return 0 if not bad else 1


def main(argv):
    if len(argv) == 3 and argv[0] == "derive":
        return derive(argv[1], argv[2])
    if len(argv) == 3 and argv[0] == "anchor":
        return anchor(argv[1], argv[2])
    if len(argv) == 3 and argv[0] == "verify":
        return verify(argv[1], argv[2])
    if len(argv) == 2 and argv[0] == "--selftest":
        return selftest(argv[1])
    sys.stderr.write(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
