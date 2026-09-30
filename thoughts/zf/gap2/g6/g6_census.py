#!/usr/bin/env python3
"""G6 padding census (Q2, lever c.5), from one tree log run with LAMBDA_VM_ROW_CENSUS=1 (standard library only; runs on
the box or the laptop).

  g6_census.py <tree log> --out <dir>

Lines read (branch fix2/1010-g6-census, all knob-gated, printed per epoch `#k` and for the global stage `global`):
  BASE ROWS <who>: <TABLE> <real>/<padded> n<instances> · …     every trace generator's real rows and padded height
  BASE SHAPES <who>: <AIR name> w<main width> m<variables> · …  every table handed to the prover, in prover order
  BASE STACK <who>: tables a..b cells <C> polys <P> vars <V> · …  each commit group: the columns' own cells, and the
                                                                 stacked polynomials (P × 2^V cells) they sit in
  ARGUE TABLE <who> <i> <AIR name>: vars m · cols c · argue s   every table's argue seconds (multi_prove's own timer)
and each scope's `WHIR PROVE SPLIT` argue sum, as a cross-check of the per-table times.

Per table: padded rows = Σ 2^m over its instances (the shapes); real rows = the generators' Σ (BASE ROWS). Tables
with no generator note are fixed-size lookup/constant tables (BITWISE 2^20, KECCAK_RC, HALT): real = padded by
construction. DECODE is generated once at the head, so its one note (in the first epoch's line) serves every scope.
  padded share of committed cells  = Σ cols × (padded − real) / Σ cols × padded       (rows' own padding)
  + the stacking's alignment padding = Σ (P × 2^V) − Σ cells                          (on top, per commit group)
  padded share of argue work [E]   = Σ_t argue_t × (1 − real_t / padded_t) / Σ argue_t, with real/padded per table
                                     name: argue time taken as linear in the padded rows (the sumchecks run over the
                                     full cube), an upper bound where a table's time is per-round overhead."""
import argparse, collections, os, re

FIXED_REAL_IS_PADDED = ("BITWISE", "KECCAK_RC", "HALT")
# The continuation AIRs carry no name (`air.name()` is `unknown`: continuation.rs's l2g_memory_air, l2g_global_air and
# global_memory_air never call `with_name`), and one generator builds both L2G traces.
ALIAS = {"L2G_MEMORY": "L2G", "L2G_GLOBAL": "L2G"}


def relabel(scope, sh, r):
    """Name the unnamed (`unknown`) tables of a scope from the code's order. An epoch has one: its L2G_MEMORY. The global
    stage (prep_global_ahead) pushes one l2g_global_air per epoch, then one global_memory_air per page config: the first
    k unnamed tables are L2G_GLOBAL, k = the global stage's L2G notes, the rest GLOBAL_MEMORY."""
    unk = [i for i, (n, w, m) in enumerate(sh) if base_name(n) == "unknown"]
    if not unk:
        return sh, None
    new = list(sh)
    if scope == "global":
        k = r.get("L2G", (0, 0, 0))[2]
        for j, i in enumerate(unk):
            new[i] = ("L2G_GLOBAL" if j < k else "GLOBAL_MEMORY",) + tuple(sh[i][1:])
        return new, None
    if len(unk) == 1 and "L2G" in r:
        new[unk[0]] = ("L2G_MEMORY",) + tuple(sh[unk[0]][1:])
        return new, None
    return sh, f"{scope}: {len(unk)} unnamed tables left as `unknown`"


def base_name(n):
    return n.split("[", 1)[0]


def parse(path):
    rows, shapes, stacks, argue, split_argue = {}, {}, {}, collections.defaultdict(list), {}
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.rstrip("\n")
        if m := re.match(r"^BASE ROWS (\S+): (.*)$", line):
            d = {}
            for part in m[2].split(" · "):
                if mm := re.match(r"^(\S+) (\d+)/(\d+) n(\d+)$", part.strip()):
                    d[mm[1]] = (int(mm[2]), int(mm[3]), int(mm[4]))
            rows[m[1]] = d
        elif m := re.match(r"^BASE SHAPES (\S+): (.*)$", line):
            lst = []
            for part in m[2].split(" · "):
                if mm := re.match(r"^(\S+) w(\d+) m(\d+)$", part.strip()):
                    lst.append((mm[1], int(mm[2]), int(mm[3])))
            shapes[m[1]] = lst
        elif m := re.match(r"^BASE STACK (\S+): (.*)$", line):
            lst = []
            for part in m[2].split(" · "):
                if mm := re.match(r"^tables (\d+)\.\.(\d+) cells (\d+) polys (\d+) vars (\d+)$", part.strip()):
                    lst.append(tuple(int(x) for x in mm.groups()))
            stacks[m[1]] = lst
        elif m := re.match(r"^ARGUE TABLE (\S+) (\d+) (\S+): vars (\d+) · cols (\d+) · argue ([\d.]+)s$", line):
            argue[m[1]].append((int(m[2]), m[3], int(m[4]), int(m[5]), float(m[6])))
        elif m := re.match(r"^WHIR PROVE SPLIT (#\d+|GLOBAL)[^:]*: .*?argue ([\d.]+)", line):
            split_argue["global" if m[1] == "GLOBAL" else m[1][1:]] = float(m[2])
    return rows, shapes, stacks, argue, split_argue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--out", required=True)
    A = ap.parse_args()
    os.makedirs(A.out, exist_ok=True)
    rows, shapes, stacks, argue, split_argue = parse(A.log)
    summ = open(os.path.join(A.out, "census_summary.txt"), "w")

    def say(*x):
        print(*x); print(*x, file=summ)

    scopes = [w for w in shapes if w in argue]
    say(f"G6 census · {os.path.basename(A.log)} · scopes with shapes and argue lines: {len(scopes)} "
        f"({', '.join(scopes[:3])}{' …' if len(scopes) > 3 else ''}) · BASE ROWS lines {len(rows)} · BASE STACK lines {len(stacks)}")
    if not scopes:
        raise SystemExit("no census lines: was LAMBDA_VM_ROW_CENSUS=1 set (and LAMBDA_VM_BASE_SPLIT=1, for the argue lines)?")
    decode = next(((r["DECODE"]) for r in rows.values() if "DECODE" in r), None)
    checks = []
    per_table = collections.OrderedDict()      # (scope, name) -> dict
    tot = collections.Counter()
    tsv = open(os.path.join(A.out, "rows_by_table.tsv"), "w")
    print("scope\ttable\tinstances\treal_rows\tpadded_rows\tcols\treal_cells\tpadded_cells\targue_s\tsource", file=tsv)
    for scope in scopes:
        sh0 = shapes[scope]
        ar = {i: (n, v, c, s) for i, n, v, c, s in argue[scope]}
        if len(ar) != len(sh0):
            checks.append(f"{scope}: {len(ar)} ARGUE TABLE lines for {len(sh0)} tables")
        r = dict(rows.get(scope, {}))
        if decode and "DECODE" not in r:
            r["DECODE"] = decode
        sh, why = relabel(scope, sh0, r)
        if why:
            checks.append(why)
        # group the scope's tables by base name, in prover order
        groups = collections.OrderedDict()
        for i, (name, w, m) in enumerate(sh):
            n, v, c, s = ar.get(i, (sh0[i][0], m, w, 0.0))
            if n != sh0[i][0] or v != m:
                checks.append(f"{scope}: table {i} is {sh0[i][0]} m{m} in SHAPES but {n} m{v} in ARGUE TABLE")
            g = groups.setdefault(base_name(name), dict(inst=0, padded=0, cols=set(), pcells=0, argue=0.0, vars=[]))
            g["inst"] += 1; g["padded"] += 1 << m; g["cols"].add(c); g["pcells"] += c << m; g["argue"] += s; g["vars"].append(m)
        used = set()
        for b, g in groups.items():
            label = b if b in r else (ALIAS[b] if ALIAS.get(b) in r else None)
            if label is None:  # an AIR named otherwise than its generator: match by instances and padded rows
                cands = [k for k, (re_, pa, n) in r.items() if k not in groups and k not in used and n == g["inst"] and pa == g["padded"]]
                label = cands[0] if len(cands) == 1 else None
            if label is not None:
                real, pad, n = r[label]; used.add(label)
                src = "generator" if label == b else (f"generator {label}" if ALIAS.get(b) == label else f"generator {label} (matched by shape)")
                if pad != g["padded"] or n != g["inst"]:
                    checks.append(f"{scope}: {b} padded {g['padded']} in {g['inst']} tables but the generator noted {pad} in {n}")
            elif b in FIXED_REAL_IS_PADDED:
                real, src = g["padded"], "fixed size: real = padded"
            else:
                real, src = g["padded"], "NO NOTE: counted as real = padded"
                checks.append(f"{scope}: {b} has no generator note")
            cols = max(g["cols"])
            if len(g["cols"]) > 1:
                checks.append(f"{scope}: {b} instances commit different column counts {sorted(g['cols'])}")
            rc = cols * real; pc = g["pcells"]
            per_table[(scope, b)] = dict(inst=g["inst"], real=real, padded=g["padded"], cols=cols, rcells=rc, pcells=pc,
                                         argue=g["argue"], vars=g["vars"])
            print(f"{scope}\t{b}\t{g['inst']}\t{real}\t{g['padded']}\t{cols}\t{rc}\t{pc}\t{g['argue']:.4f}\t{src}", file=tsv)
            k = "global" if scope == "global" else "epochs"
            tot[(k, "real_cells")] += rc; tot[(k, "padded_cells")] += pc
            tot[(k, "argue")] += g["argue"]
            tot[(k, "argue_padded")] += g["argue"] * (1 - real / g["padded"]) if g["padded"] else 0.0
            if max(g["vars"]) >= 16:
                tot[(k, "argue_big")] += g["argue"]
                tot[(k, "argue_big_padded")] += g["argue"] * (1 - real / g["padded"]) if g["padded"] else 0.0
        scope_cells = 0
        for ra, rb, cells, polys, vars_ in stacks.get(scope, []):
            k = "global" if scope == "global" else "epochs"
            tot[(k, "stacked_cells")] += polys << vars_
            tot[(k, "column_cells")] += cells
            scope_cells += cells
        pcells = sum(d["pcells"] for (sc, _), d in per_table.items() if sc == scope)
        if scope in stacks and scope_cells != pcells:
            checks.append(f"{scope}: the stacks hold {scope_cells:,} column cells, the tables {pcells:,}")
        s_arg = sum(x[4] for x in argue[scope])
        if scope in ("global",) or scope.startswith("#"):
            key = scope[1:] if scope.startswith("#") else "global"
            if key in split_argue:
                d = s_arg - split_argue[key]
                if abs(d) > max(0.02, 0.05 * split_argue[key]):
                    checks.append(f"{scope}: Σ ARGUE TABLE {s_arg:.3f} s vs WHIR PROVE SPLIT argue {split_argue[key]:.3f} s")
                tot["argue_split"] += split_argue[key]
        tot["argue_tables"] += s_arg

    say("\n=== PADDED SHARE (Q2) ===")
    for k in ("epochs", "global"):
        pc = tot[(k, "padded_cells")]
        if not pc:
            continue
        rc = tot[(k, "real_cells")]
        st = tot[(k, "stacked_cells")]; cc = tot[(k, "column_cells")]
        say(f"  [{k}] committed cells: padded {pc:,} · real {rc:,} · row padding {pc - rc:,} = {100 * (pc - rc) / pc:.2f} % of the tables' cells")
        if st:
            say(f"  [{k}] stacked polynomials {st:,} cells over the tables' {cc:,}: alignment padding {st - cc:,} = {100 * (st - cc) / st:.2f} % "
                f"of the stack; real cells are {100 * rc / st:.2f} % of what is committed (all padding {100 * (st - rc) / st:.2f} %)")
        a = tot[(k, "argue")]; ap_ = tot[(k, "argue_padded")]
        if a:
            say(f"  [{k}] argue: Σ per-table {a:.3f} s · padded share [E, linear in padded rows] {ap_:.3f} s = {100 * ap_ / a:.2f} %"
                + (f" · tables with ≥ 2^16 rows: {tot[(k, 'argue_big_padded')]:.3f} of {tot[(k, 'argue_big')]:.3f} s" if tot[(k, "argue_big")] else ""))
    ep = [s for s in scopes if s.startswith("#")]
    say(f"  argue per-table Σ {tot['argue_tables']:.3f} s vs the WHIR PROVE SPLIT argue Σ {tot['argue_split']:.3f} s over the same scopes")

    # the table × scope view
    names = []
    for (sc, b) in per_table:
        if b not in names:
            names.append(b)
    say("\n=== REAL / PADDED ROWS PER TABLE, SUMMED OVER THE EPOCHS (instances; columns; argue s) ===")
    agg = collections.OrderedDict()
    for (sc, b), d in per_table.items():
        if not sc.startswith("#"):
            continue
        a = agg.setdefault(b, collections.Counter())
        a["real"] += d["real"]; a["padded"] += d["padded"]; a["inst"] += d["inst"]; a["rc"] += d["rcells"]; a["pc"] += d["pcells"]
        a["argue"] += d["argue"]; a["cols"] = max(a["cols"], d["cols"])
    for b, a in sorted(agg.items(), key=lambda kv: -kv[1]["pc"]):
        say(f"  {b:12s} rows {a['real']:>12,} / {a['padded']:>12,} ({100 * (a['padded'] - a['real']) / a['padded']:5.1f} % padding) · "
            f"{a['inst']:3d} tables · {a['cols']:3d} cols · cells {a['pc'] / 1e6:9.1f} M ({100 * a['pc'] / max(1, tot[('epochs', 'padded_cells')]):4.1f} % of all) · "
            f"argue {a['argue']:6.3f} s")
    mat = open(os.path.join(A.out, "rows_matrix.tsv"), "w")
    print("table\t" + "\t".join(ep), file=mat)
    for b in names:
        cells = []
        for sc in ep:
            d = per_table.get((sc, b))
            cells.append(f"{d['real']}/{d['padded']}" if d else "")
        print(f"{b}\t" + "\t".join(cells), file=mat)

    say("\n=== CHECKS ===")
    say(f"consistency: {len(checks)} finding(s)" + ("" if checks else " — every table's padded rows equal its generator's note"))
    for c in checks[:40]:
        say("  " + c)
    summ.close()


if __name__ == "__main__":
    main()
