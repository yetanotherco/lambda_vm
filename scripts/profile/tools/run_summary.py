#!/usr/bin/env python3
"""One row of stage terms per run, read from the prover's own lines.

usage: run_summary.py <run dir or out.log> [...]   (prints a TSV block per run, then a side-by-side table)

Every figure is read from one named line of out.log; nothing is modelled here.
The line each figure comes from is in the key's comment below.
"""
import re
import sys
import os
import json

F = r"([0-9]+\.[0-9]+|[0-9]+)"


def grab(pat, text, cast=float, default=None):
    m = re.search(pat, text)
    if not m:
        return default
    if m.lastindex == 1:
        return cast(m.group(1))
    return tuple(cast(g) for g in m.groups())


def parse(path):
    if os.path.isdir(path):
        path = os.path.join(path, "out.log")
    t = open(path, errors="replace").read()
    r = {"path": path}
    # BLOCK PHASES: execute 17.39 · build 30.73 · prep 51.05 · A 31.29 (wait … upload … paid … commit … retire …) · B 42.00 (argue … open … tax … = upload … + encode …)
    m = re.search(
        rf"BLOCK PHASES: execute {F} · build {F} · prep {F} · A {F} \(wait {F} upload {F} paid {F} commit {F} retire {F}\) · B {F} \(argue {F} open {F} tax {F} = upload {F} \+ encode {F}\)",
        t,
    )
    keys = "execute build prep A A_wait A_upload A_paid A_commit A_retire B B_argue B_open B_tax B_upload B_encode".split()
    for k, v in zip(keys, m.groups()):
        r[k] = float(v)
    # W3 BASE: 73.52s · statement at 31.32s · plan + 13 leaves emitted by 39.76s
    r["base"] = grab(rf"W3 BASE: {F}s", t)
    r["statement_at"] = grab(rf"statement at {F}s", t)
    r["leaves_emitted_by"] = grab(rf"leaves emitted by {F}s", t)
    r["leaves_beside"] = grab(r"plan \+ ([0-9]+)(?: of [0-9]+)? leaves emitted", t, int)
    # W3 RECURSION: 12.69s after the base (tree 12.53s) · whole block 86.21s
    r["recursion"], r["tree"], r["whole"] = grab(
        rf"W3 RECURSION: {F}s after the base \(tree {F}s\) · whole block {F}s", t
    )
    # W3 LEVEL n: 9.12s wall · per program build+prove a+b · …
    r["levels"] = []
    for m in re.finditer(rf"W3 LEVEL ([0-9]+): {F}s wall · per program build\+prove (.*)", t):
        progs = [tuple(float(x) for x in p.split("+")) for p in m.group(3).split(" · ")]
        r["levels"].append(
            {
                "level": int(m.group(1)),
                "wall": float(m.group(2)),
                "n": len(progs),
                "sum_build": round(sum(p[0] for p in progs), 2),
                "sum_prove": round(sum(p[1] for p in progs), 2),
            }
        )
    # W3 LEAF ARTIFACTS: … first leaf proved@2.28 · first level-1 program@4.34 · last leaf built@1.37
    m = re.search(rf"first leaf proved@{F} · first level-1 program@{F} · last leaf built@{F}", t)
    if m:
        r["first_leaf_proved"], r["first_l1_program"], r["last_leaf_built"] = (float(x) for x in m.groups())
    # BLOCK STREAM: windows collected at 17.77s · run built at 30.73s · … layout busy 51.05s · phase A ended at 31.29s
    r["windows_at"] = grab(rf"windows collected at {F}s", t)
    r["run_built_at"] = grab(rf"run built at {F}s", t)
    # BLOCK WINDOWS: 284 windows · walk 14.85s … route+append 3.90s · chunk handout 4.26s … finish 12.96s
    m = re.search(rf"BLOCK WINDOWS: ([0-9]+) windows · walk {F}s .*? route\+append {F}s · chunk handout {F}s .*? finish {F}s", t)
    if m:
        r["windows"] = int(m.group(1))
        r["walk"], r["route"], r["handout"], r["finish"] = (float(x) for x in m.groups()[1:])
    # BLOCK PRODUCER: executor send-blocked … · walker recv-wait … send-blocked … · accumulator recv-wait … send-blocked … · layout hand-on blocked … · workers permit-wait … recv-wait …
    m = re.search(
        rf"BLOCK PRODUCER: executor send-blocked {F}s · walker recv-wait {F}s send-blocked {F}s · accumulator recv-wait {F}s send-blocked {F}s · layout hand-on blocked {F}s · workers permit-wait {F}s recv-wait {F}s",
        t,
    )
    if m:
        ks = "exec_send_blocked walker_recv_wait walker_send_blocked acc_recv_wait acc_send_blocked layout_handon_blocked workers_permit_wait workers_recv_wait".split()
        for k, v in zip(ks, m.groups()):
            r[k] = float(v)
    # BLOCK OPEN SPLIT (host s on the prover's thread): grind … · sumcheck … · …
    m = re.search(r"BLOCK OPEN SPLIT \(host s on the prover's thread\): (.*)", t)
    if m:
        r["open_split"] = {kv.rsplit(" ", 1)[0]: float(kv.rsplit(" ", 1)[1]) for kv in m.group(1).split(" · ")}
    # BATCHED ARGUE SPLIT: tables 9 · bins 5 · trees 0.043s · ladders 0.130s · zc setup 0.028s · zc rounds 0.026s · reduce 0.000s
    a = {"n": 0, "trees": 0.0, "ladders": 0.0, "zc_setup": 0.0, "zc_rounds": 0.0, "reduce": 0.0}
    for m in re.finditer(
        rf"BATCHED ARGUE SPLIT: tables [0-9]+ · bins [0-9]+ · trees {F}s · ladders {F}s · zc setup {F}s · zc rounds {F}s · reduce {F}s", t
    ):
        a["n"] += 1
        for k, v in zip(["trees", "ladders", "zc_setup", "zc_rounds", "reduce"], m.groups()):
            a[k] += float(v)
    r["argue_split"] = {k: round(v, 3) for k, v in a.items()}
    # BLOCK GROUP g: tables … · polys … · cells … M || A wait … upload … paid … commit … retire … from@… done@… || B upload … argue … encode … open …
    g = []
    for m in re.finditer(
        rf"BLOCK GROUP ([0-9]+): tables ([0-9]+) · polys ([0-9]+) · cells {F} M \|\| A wait {F} upload {F} paid {F} commit {F} retire {F} from@{F} done@{F} \|\| B upload {F} argue {F} encode {F} open {F}",
        t,
    ):
        v = m.groups()
        g.append(
            dict(
                g=int(v[0]), tables=int(v[1]), polys=int(v[2]), cells=float(v[3]), a_wait=float(v[4]), a_upload=float(v[5]),
                a_paid=float(v[6]), a_commit=float(v[7]), a_retire=float(v[8]), a_from=float(v[9]), a_done=float(v[10]),
                b_upload=float(v[11]), b_argue=float(v[12]), b_encode=float(v[13]), b_open=float(v[14]),
            )
        )
    r["groups"] = g
    # BLOCK SHAPE: tables 969 · groups 90 · chains 269 · cells 33.483 G
    m = re.search(rf"BLOCK SHAPE: tables ([0-9]+) · groups ([0-9]+) · chains ([0-9]+) · cells {F} G", t)
    if m:
        r["tables"], r["ngroups"], r["chains"] = int(m.group(1)), int(m.group(2)), int(m.group(3))
        r["cells_G"] = float(m.group(4))
    # BLOCK REGEN window: … depositors waited 51.88 s · takers waited 0.16 s ;  BLOCK REGEN rest window: … takers waited 0.45 s
    r["regen_takers_streamed"] = grab(rf"BLOCK REGEN window: .*? takers waited {F} s", t)
    r["regen_takers_rest"] = grab(rf"BLOCK REGEN rest window: .*? takers waited {F} s", t)
    m = re.search(rf"BLOCK REGEN live: .*? wall {F} s · CPU exec {F} · walk {F} · 3 generators {F} = {F} s", t)
    if m:
        r["regen_wall"], r["regen_cpu_exec"], r["regen_cpu_walk"], r["regen_cpu_gen"], r["regen_cpu"] = (float(x) for x in m.groups())
    m = re.search(rf"BLOCK REGEN rest: .*? wall {F} s · 3 generators CPU {F} s", t)
    if m:
        r["regen_rest_wall"], r["regen_rest_cpu"] = float(m.group(1)), float(m.group(2))
    r["regen_head_start"] = grab(rf"BLOCK REGEN head start: {F} s before phase B", t)
    # … · ready from 1.0 s to 39.0 s · … : seconds since that regenerator started (the streamed one starts
    # `regen_head_start` before phase B, the rest's with phase B)
    r["regen_ready_to"] = grab(rf"BLOCK REGEN live: .*? ready from {F} s to {F} s", t, default=(None, None))[1]
    r["regen_rest_ready_to"] = grab(rf"BLOCK REGEN rest: .*? ready from {F} s to {F} s", t, default=(None, None))[1]
    # W3 TREE DOWNLOAD (staged): … staged 133 trees 9.74 GB in 1.91 s
    m = re.search(rf"staged ([0-9]+) trees {F} GB in {F} s", t)
    if m:
        r["tree_dl_n"], r["tree_dl_gb"], r["tree_dl_s"] = int(m.group(1)), float(m.group(2)), float(m.group(3))
    # W3 HARNESS VERIFY: … ACCEPTED in 13.55s ; W3 TREE VERIFY: ACCEPTED in 14.79s
    r["harness_verify"] = grab(rf"W3 HARNESS VERIFY: .*? ACCEPTED in {F}s", t)
    r["tree_verify"] = grab(rf"W3 TREE VERIFY: ACCEPTED in {F}s", t)
    # W3 TREE PROGRAM BUDGET end: … admissions waited Σ 0.00s
    r["budget_waits"] = grab(rf"admissions waited Σ {F}s", t)
    # BLOCK RECOMMIT: … on the card 269 calls in 0.290 s
    r["kept_top_paths_s"] = grab(rf"on the card [0-9]+ calls in {F} s", t)
    r["prove_start_unix"] = grab(rf"W3 PROVE START: unix {F}", t)
    return r


def derived(r):
    d = {}
    d["pause"] = round(r["base"] - r["A"] - r["B"], 2)
    d["B_other"] = round(r["B"] - r["B_argue"] - r["B_open"] - r["B_tax"], 2)
    d["A_other"] = round(r["A"] - r["A_wait"] - r["A_paid"] - r["A_commit"] - r["A_retire"], 2)
    d["after_tree"] = round(r["recursion"] - r["tree"], 2)
    d["sum"] = round(r["A"] + d["pause"] + r["B"] + r["recursion"], 2)
    g = r["groups"]
    if g:
        d["A_first_from"] = g[0]["a_from"]
        d["A_wait_g0"] = g[0]["a_wait"]
        # idle between one group's commit end and the next one's start, from the from@/done@ stamps
        gaps = [round(g[i + 1]["a_from"] - g[i]["a_done"], 3) for i in range(len(g) - 1)]
        d["A_gaps_sum"] = round(sum(gaps), 2)
        d["A_gaps_over_50ms"] = round(sum(x for x in gaps if x > 0.05), 2)
        d["A_last_done"] = g[-1]["a_done"]
        d["commit_mean"] = round(sum(x["a_commit"] for x in g) / len(g), 4)
        d["cells_M_sum"] = round(sum(x["cells"] for x in g), 1)
        d["commit_s_per_Gcell"] = round(sum(x["a_commit"] for x in g) / (sum(x["cells"] for x in g) / 1000), 4)
        for k in ("b_argue", "b_encode", "b_open", "b_upload"):
            d[k + "_mean"] = round(sum(x[k] for x in g) / len(g), 4)
        kr = [x for x in g if x["tables"] <= 5]
        if kr:
            d["argue_wide_groups_n"] = len(kr)
            d["argue_wide_groups_mean"] = round(sum(x["b_argue"] for x in kr) / len(kr), 3)
            rest = [x for x in g if x["tables"] > 5]
            d["argue_other_groups_mean"] = round(sum(x["b_argue"] for x in rest) / len(rest), 3)
    a = r["argue_split"]
    d["argue_split_sum"] = round(a["trees"] + a["ladders"] + a["zc_setup"] + a["zc_rounds"] + a["reduce"], 2)
    d["argue_unsplit"] = round(r["B_argue"] - d["argue_split_sum"], 2)
    return d


def main():
    runs = [parse(p) for p in sys.argv[1:]]
    out = []
    for r in runs:
        d = derived(r)
        name = os.path.basename(os.path.dirname(r["path"])) or r["path"]
        flat = {"run": name}
        for k, v in r.items():
            if k in ("groups", "path"):
                continue
            flat[k] = v
        flat.update({"d_" + k: v for k, v in d.items()})
        out.append(flat)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
