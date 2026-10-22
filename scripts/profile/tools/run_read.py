#!/usr/bin/env python3
"""Per-run read-out of a block profile run. Text in, text out.

    run_read.py run <run dir>     one run: stage.json (the prover's own stamps), the stage windows applied to the
                                  capture's text tables, the host sampler and nvidia-smi (win.tsv fam.tsv top.tsv
                                  gaps.tsv run.md, by stage_windows.run), rate.tsv (seconds a group per window)

Run directories are NN-<M|P><w|c|n>: median or large block; warm-up, control, capture. out.log is stamped
`<unix> <line>` by stamp.py. The rows over a whole set of runs are profile_read.py's.
"""
import csv
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_summary as RS  # noqa: E402
import stage_windows as W  # noqa: E402

OPEN_FAM = "open: whir rounds and folds"
ARGUE_FAM = "sumcheck / GKR / fold"
DEVICE_KEYS = ["commit/encode to the host", "device commit errors", "openings on the host", "argue device fallbacks",
               "GKR tree refusals", "fused declines"]


def stage(rd):
    """(stage record or None, stamped lines, error text). The record is run_summary's, read from the de-stamped log."""
    lt = W.stamped(f"{rd}/out.log")
    raw = f"{rd}/out.raw.log"
    with open(raw, "w") as f:
        f.write("\n".join(t for _, t in lt) + "\n")
    try:
        r = RS.parse(raw)
        r["derived"] = RS.derived(r)
    except Exception as e:  # an unfinished run has no BLOCK PHASES / W3 RECURSION line
        return None, lt, f"{type(e).__name__}: {e}"
    return r, lt, None


def checks(lt):
    """What V1 reads, from the log's own closing lines."""
    log = "\n".join(t for _, t in lt)
    c = {}
    c["result_ok"] = re.search(r"^test result: ok\. 1 passed; 0 failed", log, re.M) is not None
    c["failures_list"] = re.search(r"^failures:", log, re.M) is not None
    c["harness"] = re.search(r"^W3 HARNESS VERIFY: .*ACCEPTED", log, re.M) is not None
    c["tree"] = re.search(r"^W3 TREE VERIFY: ACCEPTED", log, re.M) is not None
    m = re.search(r"^W3 TOP DIGEST: ([0-9a-f]+)", log, re.M)
    c["top"] = m.group(1) if m else None
    m = re.search(r"^W3 STATEMENT DIGEST: ([0-9a-f]+)", log, re.M)
    c["statement"] = m.group(1) if m else None
    m = re.search(r"^W3 DEVICE: (.*)", log, re.M)
    dev = {}
    if m:
        for k in DEVICE_KEYS:
            mk = re.search(re.escape(k) + r" (\d+)", m.group(1))
            dev[k] = int(mk.group(1)) if mk else None
    c["device"] = dev
    return c


def regen_end(r):
    """Seconds since the prove started at which the last rebuilt chunk was ready, or None."""
    bs = r["base"] - r["B"]
    ends = []
    if r.get("regen_ready_to") is not None:
        ends.append(bs - (r.get("regen_head_start") or 0.0) + r["regen_ready_to"])
    if r.get("regen_rest_ready_to") is not None:
        ends.append(bs + r["regen_rest_ready_to"])
    return max(ends) if ends else None


def my_windows(r):
    """[(name, a, b)] in unix seconds from the run's own stamps, and stage_windows' info dict."""
    t0 = r["prove_start_unix"]
    g = r["groups"]
    A, base, whole = r["A"], r["base"], r["whole"]
    bs = base - r["B"]
    w = []

    def add(name, a, b):
        if b - a >= 0.05:
            w.append((name, t0 + a, t0 + b))

    first = g[0]["a_from"] if g else r["A_wait"]
    add("A1 before the first commit", 0.0, first)
    add("A2 commits", first, A)
    if r.get("windows_at") and r.get("run_built_at"):
        add("A windows", 0.0, r["windows_at"])
        add("A finish", r["windows_at"], r["run_built_at"])
    add("phase A", 0.0, A)
    add("B proving phase", bs, base)
    third = r["B"] / 3.0
    add("B first third", bs, bs + third)
    add("B later two thirds", bs + third, base)
    ready = r.get("leaves_emitted_by")
    if ready and bs < ready < base:
        add("B emission", bs, ready)
        add("B after emission", ready, base)
        rend = regen_end(r)
        if rend and ready < rend < base - 1.0:
            add("B rebuild running", ready, rend)
            add("B after the rebuild", rend, base)
    L0 = r["levels"][0]["wall"] if r["levels"] else 0.0
    add("L0", base, base + L0)
    add("L1..top", base + L0, whole)
    add("recursion", base, whole)
    add("WHOLE", 0.0, whole)
    info = {"hash": "Poseidon1", "plan": f"{r.get('ngroups', '-')}g/{r['levels'][0]['n'] if r['levels'] else '-'}L",
            "A": A, "B": r["B"], "base": base, "L0": L0, "rec": r["recursion"], "whole": whole, "wait": r["A_wait"]}
    return w, info


def run(rd):
    r, lt, err = stage(rd)
    c = checks(lt)
    name = os.path.basename(rd.rstrip("/"))
    rec = {"run": name, "error": err, "checks": c, "rc": W.rd(f"{rd}/rc.txt", "-").strip() or "-",
           "environ_bytes": W.rd(f"{rd}/environ.bytes", "").strip() or None}
    if r is None:
        json.dump(rec, open(f"{rd}/stage.json", "w"), indent=1, ensure_ascii=False)
        print(f"RUN {name} rc {rec['rc']} · UNFINISHED ({err})")
        return rec
    wins, info = my_windows(r)
    # the openings' own kernels apart from the argument's, by name; the other families as they are
    if not any(f == OPEN_FAM for f, _ in W.FAM):
        i = next(k for k, (f, _) in enumerate(W.FAM) if f == ARGUE_FAM)
        W.FAM.insert(i, (OPEN_FAM, r"^whir_"))
    W.windows = lambda _lt, _kind: (wins, info)
    W.run(rd, "W")
    # seconds a group per window, from the argue lines' arrival times
    ts = [t for t, txt in lt if txt.startswith("BATCHED ARGUE SPLIT")]
    rows = []
    for wn, a, b in wins:
        if not wn.startswith("B "):
            continue
        k = sum(1 for t in ts if a <= t < b)
        rows.append((wn, f"{b - a:.2f}", k, f"{(b - a) / k:.3f}" if k else "-"))
    with open(f"{rd}/rate.tsv", "w") as f:
        f.write("window\twall_s\tgroups\ts_per_group\n")
        for row in rows:
            f.write("\t".join(str(x) for x in row) + "\n")
    rec["stage"] = {k: v for k, v in r.items() if k not in ("groups", "path")}
    rec["windows"] = [(n, round(a - r["prove_start_unix"], 3), round(b - r["prove_start_unix"], 3)) for n, a, b in wins]
    json.dump(rec, open(f"{rd}/stage.json", "w"), indent=1, ensure_ascii=False)
    return rec


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "run":
        print(__doc__)
        sys.exit(1)
    run(sys.argv[2].rstrip("/"))
