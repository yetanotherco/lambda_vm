# G6 — where the #1010 base's idle is now (SOTA §d Q5), and how much of the base is padding (Q2)

Lane G6 of the SOTA round, from the lead's message of 2026-09-30 (no brief file). Measurement only. Inputs:
`SOTA.md` §c.3, §c.5, §d; method from `G1-LEDGER.md` (the last full ledger, at 0428c393b).

## Summary (one screen)

**Status 2026-09-30 14:45Z (11:45 UTC−3): RAN as FAST job 249 (wt1200–1203). Results in §4; §4.0 gives the c.3 and
c.5 sizing and the recommendation. Headline: walls P1 39.8 · N 41.0 · P2 40.4 · C 40.3 s; the base card's idle (traced) is
7.50 of 31.54 s; padding (corrected for census finding F2) is 13.62 % of the tables' rows and 27.43 % of the stack; the
argue padded share is 17.04 %. c.3 caps at 2.57 s and c.5 at 3.8–4.3 s. The bullets below are the pre-registration as
written before the run.**

- **What runs (FAST, one job, ≈ 15–25 min):** four arms of the production tree at #1010's head `9e2728955` (tree =
  `33232d688`, the 40.20 s of record). P1 is plain. N runs under nsys with the host sampler. P2 is plain with the
  sampler. C is the census build (`fix2/1010-g6-census` @ `fd6da62a2`, log lines behind `LAMBDA_VM_ROW_CENSUS=1`)
  with the sampler.
- **Q5 (sizes c.3):** per base sub-phase (head, commit, argue, open, the global stage, gaps): card busy against
  idle, copy-only time split H2D pageable against pinned, idle by mechanism, host contention (process cores, runqueue
  wait, the prover thread's own wait, box busy CPUs, cgroup throttling). Also the producer's
  execute/collect/build/prep per epoch.
- **Q2 (sizes c.5):** real against padded rows per table per epoch. The padded share of committed cells, from the
  rows and with the stack's alignment. The padded share of argue work (an upper bound). **Rule: under 10 % ⇒ c.5 is
  dropped.**
- **Expected [E]:**
  - base card idle 5.5–10 s: argue 3.0–5.5, head ≥ 1.4, open 0.4–1.5, commit 0.3–1.0;
  - H2D pageable copy-only 1.5–3.0 s;
  - row padding 8–25 % of committed cells, 15–40 % with the stack's alignment.
- **BOX REQUEST (§1):** `g6-ledger-box.sh` md5 `889e189623ead0578309e0ace2da7dc2`, plus three tools. **Push
  `fix2/1010-g6-census` first.**

Markers: **[V]** verified (read in code, or measured in a named run or file) · **[I]** inferred (says what would
confirm it) · **[E]** estimate, arithmetic shown.

## 0. The code measured [V]

| what | sha | check |
|---|---|---|
| #1010 head (`whir-recursion-rpx`) | `9e2728955f8d1f0b6e71f4e21d48d592aab06f5c` | `origin/whir-recursion-rpx` at the laptop's last fetch; tree `4799a00364d1e012fe5fb0cc4c06bef6e86388c9` = 33232d688's (the box asserts it) |
| census branch `fix2/1010-g6-census` | `fd6da62a20f5c55d7b62471f5c6fddd2459d8154` | signed (G); parent 9e2728955; `git diff --shortstat` = 27 files, +243 −0 (the box asserts all three) |

- **The number of record:** #1010 = 40.20 s (job 243, wt1060–1063 @ 33232d688: B arms 40.3/40.1, base 30.8/30.6;
  PLAN.md 2026-09-29 14:09Z). Job 245 at deb9726b3 (9e2728955 plus the early claim, since reverted) gave 40.30 s in
  both arms.
- **The census commit is log lines only, off by default.**
  - `whir_split::note_rows` returns at once unless `census_enabled()` (`LAMBDA_VM_ROW_CENSUS` non-empty and not `0`).
    The 25 generators call it once each, after their padding line.
  - The per-table argue seconds are pushed inside `note_table`, which returns first unless the split
    (`LAMBDA_VM_BASE_SPLIT`) is on. The launcher exports the split by default (v10 :390).
  - The prover side builds `ArgueCensus` only when the census is on, and prints `BASE ROWS`, `BASE SHAPES`,
    `BASE STACK` and `ARGUE TABLE` lines. Nothing it records feeds a proof.
  - Tests: `census_disabled_is_inert` (knob off ⇒ a note and a table time record nothing) and
    `rows_line_sums_per_table`. Run, 2 passed.
  - `make fmt` and `make lint` green.
  - Byte gates were not run: no proof code changed. The box checks that C's program ids equal the default arms'
    (`identities … IDENTICAL`).

## 1. BOX REQUEST (lane G6)

- **Files** (laptop `thoughts/zf/gap2/g6/`, gitignored). Copy all four to `/root/zf/g6/` on FAST:

  | file | md5 | role |
  |---|---|---|
  | `g6-ledger-box.sh` | `889e189623ead0578309e0ace2da7dc2` | the driver |
  | `g6_trace.py` | `175f0450adf082bc0dac31b0f8829f97` | trace + log analysis (numpy; G1's partition, the base's sub-phases, contention) |
  | `g6_sampler.py` | `56eff730175a96591c7546c62fd33caf` | host sampler (stdlib) |
  | `g6_census.py` | `883228a366f97c1f540b92c078a2ffc2` | the padding census (stdlib) |

  The driver checks the three tools' md5s at preflight.
- **Push first:** `fix2/1010-g6-census` (fd6da62a2) to origin. The driver fetches it and refuses with rc 4 in three
  cases: the sha is absent, its parent is not 9e2728955, or the diff is not the census's 27 files, +243.
- **EXPECT_HEAD:** `9e2728955f8d1f0b6e71f4e21d48d592aab06f5c` for P1, N and P2; `fd6da62a2…` for C. The harness
  asserts the head before each arm, and the driver asserts it again after each arm.
- **Run** (the dry run is CPU only; start it when FAST is not measuring anything):
  ```
  cd /root/zf/g6
  DRY=1 bash g6-ledger-box.sh     # VERDICT: G6 DRY-RUN OK …  (≈ 2 min)
  bash g6-ledger-box.sh           # VERDICT: G6 LEDGER RUNS DONE …
  ```
  - The dry run does all of this:
    - preflight and git;
    - generates the three nsys launcher copies, asserting the line counts 1/2/5;
    - runs four harness `--dry-run`s;
    - runs the sampler's selftest on real /proc;
    - runs `g6_trace.py` on the 09-25 trace `wt90nsys`, which must print `MATCH` against the laptop's numbers;
    - makes a log-only pass over the newest head-ahead tree log (wt1061/1062/1071/1072/1031), which must print
      `PARTITION OK`;
    - runs the census on a synthetic log, which must give 0 findings.
  - A red step exits 5.
- **Tags** `wt1200`–`wt1203` (`G6_N0=<n>` moves them; the driver refuses a used tag). The nsys report is
  `/root/prof/nsys/g6-wt1201.nsys-rep`.
- **Checks and exits:**

  | condition | exit |
  |---|---|
  | card not idle (< 500 MiB and 0 compute apps, checked at start and before every arm); a prover binary running (argv[0]-anchored, PIDs only); too little disk | 9 |
  | `/root/.box.lock` or `/root/zf/.zf.lock` held | 3 |
  | P1 or P2 failed | 11 |
  | P1, N and P2 did not run one binary | 12 |
  | N left no report | 13 |
  | analysis failed | 14 |
  | the secrets scan hit (also any .sqlite, .nsys-rep or raw sampler file under out/) | 15 |
  | the sampler died at start | 16 |

  If C fails, the job still finishes with rc 0 and the VERDICT says `census: none (C failed)`.
- **Pinned deployed inputs:**
  - `/root/zf/bin/zf-whir-arms.sh` 86951bc2 and `zf_summary.py` a0228cf0;
  - `/root/A-tree-whir.v10.sh` 88dd1587 and `/root/whir_tree17.sh` d243d42e, called unchanged for P1, P2 and C.
- **Runtime [E]: 15–25 min.**
  - Four arms at ≈ 1 min each. Job 241 ran four in 3 min 52 s.
  - P1 rebuilds if the harness worktree sits at another sha: 1–4 min.
  - C rebuilds on the census commit: 1–3 min.
  - nsys finalisation and export: 3–6 min.
  - Analysis: ≈ 2 min. The laptop ran the 107.8 s wt90 trace in 66 s at 1.7 GB RSS.
- **Disk: 3 GB** (the job refuses below 8 GB free). The report and sqlite ≤ 1 GB and stay on FAST, as do the sampler
  files (`/root/zf/g6/sampler/`).
- **What comes off:** `/root/zf/g6/g6-out.tar.gz`. It holds text and TSV only, secrets-scanned: tree logs, summaries,
  the generated launchers, `g6.out` + TSVs per arm, `census.out` + TSVs, and `ab.md` (walls, identities).
- **Left behind:** the harness worktree at fd6da62a2. The next harness job checks out its own sha.

## 2. Method

### 2.1 Windows, sub-phases, partition (G1's, extended) [V: code in `g6_trace.py`]

- **Windows**, from log stamps:
  - base = [`BASE HEAD (WHIR): start`, `★ LEVEL 1 (wide) START`];
  - level 1 = [its start, + the printed wall];
  - root = the rest to `WHOLE RUN`.
- **The base's sub-phases** (disjoint; every edge a stamp):
  - **head** = base start → the prover thread's first phase;
  - per scope (the epochs; the global stage): **absorb+prep**, **commit**, **argue**, **open**. Argue and open are
    cut from each prove span by its `WHIR PROVE SPLIT` line (challenge → argue → open, timed in that order in
    `multi_prove`);
  - **prover gaps** = the rest.
- **Partition:** at each instant the kernels split the time equally (stages 1–6, 0); otherwise copies and memsets do
  (row 7, split by kind: H2D from pageable, H2D from pinned, D2H, D2D, memset); otherwise it is idle (row 8).
  Exact, checked to 2 ms per window and per sub-phase.
- **Kernel → stage map:** G1's, plus the two sumcheck kernels added since (`batched_column_ext3`,
  `gather_factor_heads_ext3`). Sumcheck-family kernels count as stage 4 inside argue windows and stage 5 inside open
  windows. A kernel no rule knows lands on row 0 and is listed by name.
- **Idle by mechanism** (W1): each idle gap is charged to the thread submitting the op that ends it, by that thread's
  state: a CUDA API category, an OS call, or untraced (host code or descheduled). "Not yet requested" = the
  submitting call began after the gap began.

### 2.2 Host contention (the sampler) [V: `g6_sampler.py`, `g6_trace.py`]

- **What it reads, every 0.05 s:**
  - per thread of the prover test binary: on-CPU ns and runqueue-wait ns (`/proc/<pid>/task/<tid>/schedstat`);
  - `/proc/stat` busy and total jiffies;
  - the cgroup's CPU ns, throttled periods and throttled ns;
  - once, the online CPUs, affinity and `cpu.max`.
- **What it writes:** numbers, plus each thread's name if it is made of `[A-Za-z0-9_:.-]`. It never writes a command
  line or an environment. The process is matched on argv[0], or on `/proc/<pid>/exe` when argv[0] is relative.
- **Resolution:** a counter over [a, b] is interpolated between samples, so edge error ≤ one interval per edge.
  At 0.1 s this cost ≈ 4 % on the argue's on-CPU (laptop synthetic), hence 0.05 s.
- **Which thread is the prover:** in N, the thread with the most CUDA API time inside the argue windows. In P2 and C,
  the name rule: the lowest tid named after the test (libtest names the test's thread; threads it spawns inherit the
  name and come later). N prints whether the two agree. A DISAGREE voids the prover-thread column of P2 and C.
- **The producer column:** the busiest other thread inside the `execute` spans. Work the producer hands to rayon
  appears only in the process total.
- **Per sub-phase it reports:**
  - wall;
  - process on-CPU (and cores);
  - process runqueue wait;
  - the prover's and the producer's on-CPU and wait;
  - box busy CPUs against online CPUs;
  - cgroup cores;
  - CFS-throttled seconds.
- **Why three sampled arms:**
  - N: the prover thread is named by the trace, but nsys perturbs the host;
  - P2: the same without nsys; P2 − P1 is the sampler's own cost;
  - C: the same once more, on the census build.

### 2.3 Named host work [V]

- **Per epoch, from the log:** the producer's execute, collect, build and prep@producer, and its hand-off wait. The
  handoff span contains prep@producer (both start at build's end, e.g. job 241 wt1031 epoch 0: `BASE PREP 0:
  prep@producer 0.46s t=[…233.772,…234.235]` = `handoff 0.46s` over the same stamps), so the wait = handoff minus
  prep.
- Also from the log: the head's helper (DECODE prepared and root) and the level-1 prologues in the base's tail.
- Each one's overlap with every sub-phase is reported.

### 2.4 The padding census (Q2) [V: `g6_census.py`, the branch's lines]

- **Per scope** (epoch `#k`, `global`):
  - `BASE ROWS`: every generator's real rows and padded height, summed per table, with the instance count;
  - `BASE SHAPES`: every table's name, main width and variables, in prover order;
  - `BASE STACK`: per commit group, Σ cols × 2^m against the stacked polynomials P × 2^V;
  - `ARGUE TABLE`: per table, the argue seconds `multi_prove` timed.
- **Joins:**
  - a table's padded rows = Σ 2^m over its instances; its real rows = the generators' note;
  - BITWISE, KECCAK_RC and HALT are fixed-size (real = padded);
  - DECODE is generated once at the head and its note serves every scope;
  - the continuation AIRs carry no name (`air.name()` = `unknown`: `l2g_memory_air`, `l2g_global_air` and
    `global_memory_air` never call `with_name`, continuation.rs:173–270). They are named from the code's order: an
    epoch's one unnamed table is its L2G_MEMORY. The global stage pushes one L2G_GLOBAL per epoch, then one
    GLOBAL_MEMORY per page config (`prep_global_ahead`), and both L2Gs match the `L2G` note.
- **Padded share of committed cells:**
  - rows only: Σ cols × (padded − real) ÷ Σ cols × padded;
  - with the stack's alignment: (Σ P × 2^V − Σ real cells) ÷ Σ P × 2^V. This is what a jagged PCS stops committing,
    before its own rounding.
- **Padded share of argue work [E]:** Σ_t argue_t × (1 − real_t ÷ padded_t) ÷ Σ argue_t. It takes argue time as
  linear in padded rows. That is an **upper bound** on what skipping padding saves: a per-table fixed cost (host glue,
  round trips) does not shrink.
- **Checks** (each a finding, listed):
  - SHAPES = the generators' padded rows and instance counts;
  - ARGUE TABLE names and variables = SHAPES;
  - Σ STACK cells = Σ table cells;
  - Σ ARGUE TABLE = the `WHIR PROVE SPLIT` argue per scope (±5 %).

## 3. Pre-registration (written 2026-09-30, before any G6 run)

### 3.1 Walls, overhead, identity, closure

| item | band | expected | source |
|---|---|---|---|
| P1, P2 WHOLE RUN | [38.7, 41.7] s | 40.2 | job 243 B 40.3/40.1; job 245 40.4/40.2 |
| P1, P2 base | [29.3, 32.3] s | 30.7 | job 243 B 30.8/30.6; wt1031 30.58 |
| P2 − P1 (the sampler's cost) | [−0.5, +0.6] s | ±0.2 | same-sha spread 0.2 s (job 243) |
| N − mean(P1, P2) (nsys) | [−0.5, +2.0] s | +0.4 … +1.2 | G1: +1.30 s on 60.4 s (2.15 %), base +1.8 % |
| C − mean(P1, P2) | whole ±0.6 s, base ±0.4 s | ≈ 0 | log lines only; another binary (build lottery: memory `from-fn-closure-codegen-lottery`) |
| program ids | N = P, C = P (IDENTICAL) | | log-only knobs |
| closure | every window and the base's sub-phases partition to 2 ms | | by construction, checked |
| prover thread | the name rule AGREEs with the CUDA API in N | | §2.2 |

A miss on the wall bands means the box or the head moved. The job is then read against its own P arms, not against
40.20 s.

### 3.2 The base's ledger [E]

Sub-phase walls are job 241's wt1031 (d117ffedd with the head-ahead knob, the head's base code; log-only pass, §6).
The idle and term expectations come from G1's partition at 0428c393b and D-ARGUE's split at 9e2728955 (argue 11.76 s
= GKR 4.25, of which device rounds 3.00, + zerocheck 4.28, of which device 4.03, + rest 3.23).

| sub-phase | wall (wt1031) | card idle expected | main terms expected |
|---|---|---|---|
| head | 1.86 | ≥ 1.4 | DECODE root 1.22 s on the helper, beside epoch 0's producer 1.42 s; no card work before epoch 0's commit |
| epochs commit | 9.10 | 0.3–1.0 | 3a leaves + 3b Merkle + 2 NTT; H2D pageable copy-only 1.0–3.0 s (G1: 55.6 GB, 2.92 s in the whole base) |
| epochs argue | 11.70 | 3.0–5.5 | stage 4 6.0–8.0 s |
| epochs open | 6.35 | 0.4–1.5 | 5 fold/sumcheck, 6 grind, tree rebuilds (I-WHIR Q4: 2.41 s) |
| global (4 parts) | 1.30 | 0.2–0.6 | |
| absorb+prep, gaps | 0.28 | ≈ all | |
| **base** | **30.58** | **5.5–10.0** | H2D pageable in the base 45–60 GB; pinned ≈ 0 |

- **Producer** (wt1031): execute Σ 1.08, collect 4.72, build 5.15, prep@producer 8.16 ⇒ work 19.11 s; hand-off wait
  9.64 s. Expected within ±15 %.
- **Contention in the epochs' argue:**
  - process on-CPU ≥ 3 cores average. The producer's prep@producer overlaps 6.17 s and build 1.58 s of the argue's
    11.70 s (wt1031 overlaps).
  - the prover thread's runqueue wait ≤ 0.3 s if the box has idle CPUs. **Above 1.0 s, contention is a first-order
    c.3 term.**
  - CFS throttling 0 s unless the cgroup has a quota; any throttled seconds are reported.

### 3.3 Padding (Q2) [E]

- **CPU:** 0 % padding in the 14 full epochs (2^21 cycles/epoch and `LAMBDA_VM_MAX_ROWS_LOG2=21`, v10 :86, :331).
- **Other tables:** for log-uniform heights the padding averages 1 − 1/(2 ln 2) ≈ 28 % of a table's cells.
- **Epochs' committed cells:**
  - row padding 8–25 %, pulled down by CPU's share of the cells;
  - with the stack's alignment 15–40 %;
  - argue padded share [E] 8–25 %.
- **Rule** (the lead's, made concrete): c.5 is **dropped** if the stack's all-padding share and the argue padded
  share are both under 10 %. Otherwise c.5 keeps a band (§5.2). The jagged PCS's own cost (≈ 5 multiplications per
  trace-area element, ePrint 2025/917 Thm 1.4) is not sized here.

## 4. Results (FAST job 249, tags wt1200–1203, 2026-09-30; filled 2026-09-30 14:45Z)

### 4.0 Verdict for c.3 and c.5 (read this first)

- **c.3 cap 2.57 s (6.4 % of 40.10 s)** = head 0.71 (DECODE root 1.12 s becomes the floor) + serialized trace H2D
  1.79 + contention 0.07; producer 0 (it waits 9.6–10.6 s). **Pre-register −1.3 s [−2.3, −0.4].**
- **c.5 cap 3.8–4.3 s (9.5–10.7 %)** gross, after keep-futile + whole trees, only with stack chunks ≤ 2^24 (at 2^27 a
  jagged pack saves 9.1 %, not 27.4 %). A format change, and the recursion's cost is unsized. **Base −2.5 s [−3.8, −1.0].**
- **Build first: c.3's cheap slice** (pinned + per-table pipelined upload, no upload of all-zero padding,
  `decode_prepared_for` off the head). It is byte-identical, needs no decision from Mauro, and aims at measured terms
  (97 % of the 1.84 s H2D runs with no kernel beside it). **−1.0 s [−1.8, −0.3].**
- **c.5: set aside by Mauro (09-30) for a separate evaluation** (§7.5). Full device generators come last: they pay only
  once the card side falls ≈ 10 s and the producer's 19 s becomes the floor.

Markers in this section: **[L]** read from a log stopwatch (the prover's own stamps; `ab.md`, `*-tree.log`) ·
**[T]** from the nsys trace of N (`g6.out`, `ledger.tsv`, `transfers.tsv`; device timestamps) · **[S]** from the
host sampler (`/proc` schedstat, 0.05 s) · **[C]** from the census lines of C (`census.out`, `rows_by_table.tsv`) ·
**[A]** arithmetic on the above (shown) · [I] / [E] as before. Files: `gap2/g6/box-out/out/` (tarball md5
`770e03daed12fa84de6c78e9445a3fde`).

### 4.1 Walls, overhead, identity, closure (against §3.1)

| arm | tag | sha | WHOLE RUN [L] | base [L] | level 1 [L] | root [L] | device peak (10 ms sampler) | band |
|---|---|---|---|---|---|---|---|---|
| P1 | wt1200 | 9e2728955 | 39.8 s | 30.45 s | 7.70 s | 1.65 s | 27,634 MiB | in |
| N (nsys + sampler) | wt1201 | 9e2728955 | 41.0 s | 31.54 s | 7.80 s | 1.66 s | 27,440 MiB | in |
| P2 (sampler) | wt1202 | 9e2728955 | 40.4 s | 30.84 s | 7.90 s | 1.66 s | 27,570 MiB | in |
| C (census + sampler) | wt1203 | fd6da62a2 | 40.3 s | 30.62 s | 8.00 s | 1.68 s | 27,794 MiB | in |

- Mean of the plain setting (P1, P2) = **40.10 s** [A], spread 0.60 s. This is the denominator below: the box
  read 0.10 s under job 243's 40.20 s.
- P2 − P1 = +0.6 s whole, +0.39 s base: the top of the pre-registered [−0.5, +0.6]. P1 and P2 are each n = 1, so
  the sampler's cost is not resolved from same-sha noise.
- N − mean(P) = **+0.90 s** (band [−0.5, +2.0]). Per sub-phase, nsys lands mostly in the argue: 12.39 against a
  P mean of 11.80 s (+0.59), open +0.17, commit +0.08 [L].
- C − mean(P) = +0.20 s whole, −0.02 s base.
- Program ids IDENTICAL across all four arms; per-level cells, instructions and `LFM_HASH` perms identical; 0/0
  fallbacks; PROVED AND VERIFIED in all four [L, `ab.md`].
- Closure: every window partitions to its wall within 2 ms; the 10 base sub-phases cover 31.536 of 31.536 s; clock
  checks 99.86 % and −0.000 s [T]. Prover thread: the name rule and the CUDA API agree (tid 318386) [T+S].

### 4.2 The base's ledger (N, traced; P arms for the walls)

| sub-phase | wall N [L] | walls P1 / P2 / C [L] | busy [T] | idle [T] | copy-only [T] | main terms [T] |
|---|---|---|---|---|---|---|
| head | 1.868 | 1.866 / 1.833 / 1.868 | 0.015 | **1.853** | 0.002 | no card work before epoch 0's commit |
| epochs absorb+prep | 0.090 | 0.091 / 0.092 / 0.095 | 0.000 | 0.090 | 0.000 | |
| epochs commit | 9.183 | 9.089 / 9.123 / 9.078 | 8.918 | 0.265 | 1.650 | s2 1.08 · s3 6.19 · H2D pageable 33.02 GB, 1.581 s copy-only |
| epochs argue | 12.390 | 11.670 / 12.000 / 11.730 | 8.121 | **4.269** | 0.302 | s4 7.81 |
| epochs open | 6.471 | 6.294 / 6.350 / 6.264 | 5.940 | 0.531 | 0.058 | s3 2.87 (tree rebuilds) · s5 1.81 · s6 1.20 |
| global (4 parts) | 1.328 | 1.283 / 1.307 / 1.281 | 1.039 | 0.289 | 0.063 | argue idle 0.232 |
| prover gaps | 0.206 | 0.159 / 0.132 / 0.306 | 0.005 | 0.201 | 0.001 | |
| **base** | **31.536** | 30.452 / 30.837 / 30.622 | **24.039** | **7.497** | 2.077 | |

- **Base by stage [T], exclusive seconds:**

  | stage | seconds |
  |---|---|
  | 2 NTT + Möbius | 1.130 |
  | 3 hashing (3a leaves 7.636, 3b Merkle 1.880) | 9.516 |
  | 4 argue | 7.933 |
  | 5 open | 1.903 |
  | 6 grind | 1.479 |
  | 0 unmapped | 0.002 |
  | 7 copy/memset-only | 2.077 |
  | 8 idle | 7.497 |

- **Idle by mechanism** (the base's 7.497 s) [T]:
  - 5.912 s of it ended with an op not yet requested when the gap began;
  - by the submitting thread's state: untraced host code 2.566, futex waits 1.858, API sync 0.785, readback D2H
    0.668, alloc/free 0.570, upload H2D 0.503, launch 0.401, other 0.144.
  - The head's 1.853 s is 1.459 futex: the prover waiting on the producer.
  - The argue's 4.269 s is 1.783 untraced host code, 2.185 in API calls and 0.300 in futex waits: the argue's host glue (memory
    `argue-idle-is-host-glue`). nsys inflates it; the plain arms' argue is 0.4–0.7 s shorter, so the untraced
    argue idle is ≈ 3.6–3.9 s [E].
- **Per epoch** (`epochs.tsv`) [T]: epochs 3–6 open in 0.885–0.908 s, epochs 0–2 in 0.294–0.297 s. The difference is
  the tree rebuilds that keep-futile (−1.25 s) and whole trees (−1.00 s) remove (jobs 246, 248).
- **Transfers in the base** [T]:
  - H2D from pageable: 35.60 GB, Σ 1.840 s, exclusive 1.785 s, so 97 % of it runs with no kernel beside it [A].
  - The commit's 33.02 GB moved at 20.5 GB/s (33.02 ÷ 1.6125 s) [A]. That is the padded tables: 4,094.6 M cells ×
    8 B = 32.76 GB [A].
  - H2D from pinned: 0. D2H to pinned: 33 MB. The pre-registration expected 45–60 GB of H2D (G1's head uploaded
    55.6 GB); this head uploads 35.6 GB.

### 4.3 Host: producer, head, contention

- **Producer** [L] (execute / collect / build / prep@producer):

  | arm | execute | collect | build | prep@producer | work Σ | hand-off wait |
  |---|---|---|---|---|---|---|
  | P1 | 1.06 | 4.70 | 5.13 | 8.03 | 18.92 | 9.64 |
  | N | 1.04 | 4.70 | 5.02 | 8.26 | 19.02 | 10.56 |
  | P2 | 1.08 | 4.70 | 5.06 | 8.02 | 18.86 | 9.98 |
  | C | 1.04 | 4.61 | 5.24 | 8.29 | 19.18 | 9.62 |

  - Pre-registered: work 19.11 ± 15 %, all four in band.
  - After the head the prover never waits for the producer: every `BASE EPOCH k: prep` reads 0.00 s in P2 (Σ = 0)
    [L]. The producer waits instead, 9.6–10.6 s per run.
  - **So c.3's steady-state term is 0 today.** It binds once the card side of the base falls by ≈ 10 s [A].
- **The head's critical path** (P2, log lines 64–89; code `multilinear_continuation.rs:2616–2640`) [L+V]:

  | t from start | what | on the critical path? |
  |---|---|---|
  | 0.003 → 1.119 | DECODE root, on the helper (1.12 s) | no: epoch 0's prep waited 0.00 s |
  | → 0.065 | ELF load, `DecodeArtifacts::from_elf` | yes |
  | 0.065 → 0.425 | `decode_prepared_for` (0.36 s), serial before the pipeline | **yes** |
  | 0.425 → 0.543 | before epoch 0 executes (0.12 s) | yes [I: what it is] |
  | → 1.833 | epoch 0: execute 0.09, collect 0.51, build 0.28, prep@producer 0.41 | yes; the prover blocks in futex |

- **Contention** [S], in the epochs' argue, arms N / P2 / C:

  | measure | N | P2 | C |
  |---|---|---|---|
  | process cores | 10.73 | 10.72 | 10.99 |
  | process runqueue wait | 2.90 s | 3.20 s | 2.81 s |
  | the prover thread's runqueue wait | 0.074 s | 0.039 s | 0.026 s |
  | the prover thread's on-CPU (of the argue's wall) | 11.94 s | 11.38 s | 11.31 s |
  | box busy (of 32 cores) | 10.9 | 10.8 | 11.1 |
  | CFS throttled | 0.007 s | 0.043 s | 0.045 s |

  - The prover thread is on a CPU for 96 % of the argue and is almost never descheduled.
  - **Contention by descheduling is not a first-order term** (the pre-registered threshold was 1.0 s). Slowdown
    through memory bandwidth or cache is invisible to schedstat [I]. Only an arm that pauses the producer during
    the argue would show it.
  - The cgroup quota is 30.72 cores (`cpu.max 3071999 100000`), under the 32 online. The head runs at 18.4–19.3
    cores and is throttled for 0.06–0.09 s.

### 4.4 Padding census (C) [C], corrected for finding F2 (§4.5)

| scope | measure | as printed | corrected (F2) |
|---|---|---|---|
| epochs | the tables' padded cells | 4,094,628,724 | same |
| epochs | real cells | 3,556,532,603 | **3,536,971,973** |
| epochs | row padding (share of the tables' cells) | 13.14 % | **13.62 %** |
| epochs | stack P × 2^V | 4,873,781,248 | same |
| epochs | alignment padding (share of the stack) | 15.99 % | same |
| epochs | all padding (share of the stack) | 27.03 % | **27.43 %** |
| epochs | argue padded share [E, linear in rows] | 1.973 of 11.733 s = 16.81 % | **1.999 s = 17.04 %** |
| epochs | the same, tables ≥ 2^16 rows per instance only | 1.314 of 9.118 s | 1.340 of 9.118 s (14.7 %) |
| global | row padding | 24.17 % | same |
| global | all padding (share of 243,269,632 cells) | 57.67 % | same |
| global | argue padded share | 13.34 % | same |

- The per-table argue seconds sum to 12.060 s, equal to the `WHIR PROVE SPLIT` argue over the same scopes.
- CPU: 0 % padding in epochs 0–13. Its 3.0 % overall is all epoch 14's (1,138,690 of 2,097,152 rows).
- The largest padded cells by table [C]:
  - MEMW_A: 32.8 % of its 456 M cells;
  - LT: 22.2 % of 304 M;
  - KECCAK_RND: 16.2 % of 452 M;
  - DECODE (corrected): 37.8 % of 94 M;
  - L2G_MEMORY: 33.1 % of 99 M;
  - STORE: 32.5 % of 117 M;
  - LOAD: 23.3 % of 151 M.
- **What a jagged pack saves depends on its chunk size** [A]. Per epoch: real cells rounded up to whole chunks of
  2^V, against today's stack.

  | chunk | the epochs' committed cells | saving against today's stack |
  |---|---|---|
  | today (stacked at 2^27, aligned) | 4,873,781,248 | — |
  | 2^27 | 4,429,185,024 | **9.1 %** |
  | 2^25 | 3,825,205,248 | 21.5 % |
  | 2^24 | 3,657,433,088 | **25.0 %** |
  | 2^22 | 3,573,547,008 | 26.7 % |
  | the real cells (no rounding) | 3,536,971,973 | 27.4 % |

  Epochs 6 and 13 (four 2^27-scale polynomials for ≈ 250 M real cells) carry most of the saving at 2^27. A single
  dense polynomial at the next power of two is worse than today for epochs 3–5.
- **Rule of §3.3:** the all-padding share (27.4 %) and the argue share (17.0 %) are both over 10 %. **c.5 is kept.**

### 4.5 The 17 consistency findings, triaged

| # | findings | what | effect on the numbers | verdict |
|---|---|---|---|---|
| F1 | 15, one per epoch: "L2G_MEMORY padded 2^k in 1 table but the generator noted 2^k + 1 in 2" | `build_traces` builds a throwaway empty L2G trace each epoch (`trace_builder.rs:4398`, `generate_local_to_global_trace(&[])`; 0 real rows padded to 1 by `.max(1)`, `local_to_global.rs:269`). The continuation installs the real one afterwards (comment at :4390) [V code] | none: padded comes from SHAPES, and the extra note has 0 real rows | benign census artifact. The trace is 1 row × 9 columns; the comment's "avoid building a throwaway … trace" is stale about this empty one |
| F2 | 2: "#0 DECODE padded 1,048,576 in 1 table, the generator noted 4,194,304 in 4" and "#14 … in 3" | `ROWS` is one process-wide buffer (`whir_split.rs:778`) drained per scope. DECODE's generator also runs outside the epochs, and those notes land in whichever scope drains next: the head's `DecodeArtifacts::from_elf` (`trace_builder.rs:3315`) and `decode_prepared_for` → `preprocessed_columns` (`decode.rs:266`) [V code]; the helper's root [I]; 28 more notes in the global scope [I: the level-1 prologues or the global prep] | #0 read real 2,608,084 > padded 1,048,576. Corrected to 652,021 of 1,048,576 per epoch: real −19,560,630 cells, shares as in §4.4 | census join error for DECODE only. Corrected here; the prover is unaffected |
| — | 0 | stack Σ = table cells; ARGUE TABLE names and variables = SHAPES; Σ ARGUE TABLE = WHIR PROVE SPLIT; global L2G_GLOBAL real 7,363,343 = Σ of the epochs' L2G_MEMORY real [C] | — | pass |

- **By-catch from F2:** DECODE's trace generator ran 48 times in the base: 20 notes in the epoch scopes and 28 in
  the global scope [C]. Each run sorts 652,021 entries and fills a 2^20 × 6 trace, and the result is constant for a
  program.
  - Only one run's cost is known: `decode_prepared_for`, 0.32–0.36 s [L], on the head's critical path.
  - The others' cost and critical-path position are not measured [I]. For D-TRACE: cache one DECODE trace per ELF.

### 4.6 Against the pre-registration (§3.2, §3.3)

| item | expected | measured | |
|---|---|---|---|
| base card idle | 5.5–10.0 s | 7.497 s [T] | in |
| head idle | ≥ 1.4 s | 1.853 s | in |
| argue idle | 3.0–5.5 s | 4.269 s (N); ≈ 3.6–3.9 s untraced [E] | in |
| open idle | 0.4–1.5 s | 0.531 s | in |
| commit idle | 0.3–1.0 s | 0.265 s | just under |
| global idle | 0.2–0.6 s | 0.289 s | in |
| stage 4 in the argue | 6.0–8.0 s | 7.81 s | in |
| H2D pageable copy-only | 1.5–3.0 s (1.0–3.0 s in the commit) | 1.785 s (1.581 s in the commit) | in |
| H2D bytes in the base | 45–60 GB | 35.60 GB | under: fewer bytes than at G1's head |
| pinned H2D | ≈ 0 | 0 | in |
| producer work | 19.11 s ± 15 % | 18.86–19.18 s | in |
| prover thread's runqueue wait in the argue | ≤ 0.3 s | 0.026–0.074 s | in; not first-order |
| row padding | 8–25 % | 13.62 % | in |
| all padding (with the stack's alignment) | 15–40 % | 27.43 % | in |
| argue padded share | 8–25 % | 17.04 % | in |

### 4.7 c.3 sized (§5.1), per term

Denominator: #1010's wall at 9e2728955, 40.10 s (the plain arms' mean). A base second is taken as a wall second: the
level-1 wide prologues overlap the base's tail, and level 1 waits for the last epoch [I].

| term | measured | how much device generation can remove | cap |
|---|---|---|---|
| T1 the head | idle 1.853 s [T]. Path 1.83 s = pre-execute 0.54 (of which `decode_prepared_for` 0.36) + epoch 0 1.29 [L]. DECODE root 1.12 s beside it [L] | epoch 0's collect + build (0.79) and part of prep. DECODE root then binds: head ≥ 1.12 s | **0.71 s** (0.36 of it host-only: move `decode_prepared_for` off the path) |
| T2 trace upload | H2D pageable 35.60 GB, exclusive 1.785 s, 97 % serialized; the commit's 33.02 GB at 20.5 GB/s [T] | records ≤ 52 B/cycle × 30.5 M = 1.6 GB [E] instead of 33 GB | **1.79 s** gross |
| T3 contention | the prover thread's runqueue wait in the argue 0.026–0.074 s [S] | all of it | **0.07 s** (bandwidth interference unmeasured [I]) |
| T4 the producer | work 18.9–19.2 s, hand-off wait 9.6–10.6 s; the prover's prep waits Σ 0.00 [L] | nothing today | **0** until the card side falls ≈ 10 s |
| **Σ** | | | **2.57 s = 6.4 %** |

- **What device generation adds:** its kernels. They cost 0.02 s at the bandwidth floor (33 GB written at
  ≈ 1.8 TB/s) up to ≈ 1.8 s (Ceno's 1.13 s for 19.2 M instructions, scaled to 30.5 M cycles) [E]. They sit on the
  card's critical path: the argue's idle is ms-scale host-glue gaps, not windows a generator could fill [I].
- **Pre-register c.3 (full): −1.3 s [−2.3, −0.4]** whole run: T1 0.5–0.7, T2 net 0–1.7, T3 ≤ 0.07.
- **The pair it must beat, per memory `feedback_pair_remedies_uncontested`:** a byte-identical slice with no
  device generators. It reaches T2 and the host part of T1:
  - pinned staging;
  - a per-table pipelined upload (table k+1 uploads while table k's NTT runs);
  - no upload of all-zero padding rows (13.6 % of the bytes; valid only for tables whose padding is all zero [I]);
  - `decode_prepared_for` moved beside the DECODE root, or cached per ELF.

  The box's pinned H2D rate is not measured [I]; PCIe 5 × 16 would roughly double 20.5 GB/s. **Pre-register the
  slice: −1.0 s [−1.8, −0.3].**

### 4.8 c.5 sized (§5.2), per term (after F2's correction)

| term | measured | share removed | cap |
|---|---|---|---|
| C1 commit kernels | s2 1.08 + s3 6.19 = 7.27 s in the epochs' commit [T] | the jagged saving at the chunk size (§4.4): 9.1 % at 2^27, 25.0 % at 2^24 | 0.66 s at 2^27 · **1.82 s at 2^24** |
| C2 commit upload | 1.58 s copy-only [T] | row padding 13.6 % | 0.22 s. The same bytes as c.3's T2: count once |
| C3 open | s3 2.87 + s5 1.81 s [T]. Keep-futile and whole trees take ≈ 2.21 s of the s3 (tree_rebuild −1.39, −0.82; jobs 246, 248) [I at this head] | 25.0 % of what remains, (0.66 + 1.81) s | **0.62 s** (1.17 s before A and B land) |
| C4 argue | s4 7.81 s [T]; padded share 17.0 % (all tables), 14.7 % (tables ≥ 2^16 rows per instance) [C] | linear in rows: 17.0 % × 7.81 = 1.33. Big tables only: 14.7 % × 7.81 = 1.15. Per-table seconds × the argue's busy fraction 0.655: 2.00 × 0.655 = 1.31 → 1.34 × 0.655 = 0.88 [A] | **0.87–1.33 s** |
| C5 global | all padding 57.7 % of 243 M cells; its commit + open kernels ≈ 0.6 s [T] | ≈ half [E] | ≤ 0.3 s |
| **Σ** at 2^24, after A and B | | | **3.8–4.3 s = 9.5–10.7 %** (at 2^27: 2.1–2.6 s) |

- **Not in the caps:**
  - the jagged evaluation sumcheck: ≈ 5 multiplications per trace-area element, over 3.54 G real cells, ≤ 0.1–0.3 s
    on the card [E];
  - the recursion's added cost. At 2^24 an epoch commits 11–20 polynomials instead of 3–4, so level 1 opens more
    polynomials per query. It is unsized and needs D-WHIR's `model.py` [I].
- **What is not a format change** [I, needs D-ARGUE's reading]:
  - C4: the zerocheck and LogUp over padding rows can be skipped with SP1's geq correction, leaving the transcript
    unchanged. That belongs with i-argue's stage 1 (its corner skip may already take part of it).
  - C2's zero-row upload skip.
  - C1 and C3 need the jagged PCS, a new proof format: Mauro's decision.
- **Pre-register c.5: −2.5 s [−3.8, −1.0] at the base**, at 2^24 chunks after A and B land. No whole-run band until
  the recursion's cost is modelled.

## 5. How the results size c.3 and c.5

### 5.1 c.3 (GPU trace generation), against its real terms

The gain is at most the sum of these terms, less what GPU generation adds:

- **the head's card idle** — minus its floor: executing epoch 0 and the first record upload;
- **H2D pageable copy-only in the base** — the traces' upload. Compact records are 27–52 B/cycle against ≈ 1.8 KB of
  trace per cycle today.
- **the argue's contention term** — bounded by the prover thread's runqueue wait in the argue. Slowdown without
  descheduling (memory bandwidth, cache) is invisible to schedstat; only an arm that pauses the producer would show it
  [I].

What GPU generation adds: its own kernels. Ceno spends 1.13 s on 19.2 M instructions, so ≈ 1.8 s for our 30.5 M
cycles [E]. That is free only where it lands in idle windows.

**If the terms sum to under 2 s, c.3's −2 … −4 s band shrinks to what they allow.**

### 5.2 c.5 (commit and argue only the real rows)

- **Commit:** at most the stack's all-padding share × the commit's busy seconds.
- **Argue:** at most the argue padded share × the argue's stage-4 seconds.
- Both are upper bounds, before the jagged PCS's own cost.

## 6. Laptop validation (what was tested before handing over)

- **`g6_trace.py` on the 09-25 trace** (laptop copy of `wt90nsys.sqlite`): reproduces G1's partition to the ms.
  - busy/idle: base 46.162/20.238, level 0 17.513/6.587, interior 10.995/3.405, root 1.602/1.298;
  - closure PASS; clock checks 99.92 % and −0.000 s;
  - prover thread tid 2061881 `lfm::per_table_` (the test's thread).
- **`--log-only` on wt1031** (job 241): base 30.584 s in 10 sub-phases, PARTITION OK. Sub-phases and producer as in
  §3.2.
- **Sampler:**
  - its loop, on a faked /proc: picks up a short-lived `--list` process, then the test; writes 6-field C lines;
  - the contention arithmetic, on a synthetic file with known rates: exact to the digit (argue 11.70 s ⇒ 26.91 s =
    2.30 cores, prover wait 1.170 s, box 16.00 of 32.0 cores);
  - the Linux selftest runs in the box's dry run.
- **Census**, on synthetic logs:
  - 0 findings when consistent;
  - the stack check caught a deliberate 2,560-cell error;
  - relabels the unnamed L2G_MEMORY, L2G_GLOBAL and GLOBAL_MEMORY;
  - the WHIR PROVE SPLIT cross-check matches.
- **Driver, in a laptop rig** (fake `/root`, stub harness, nvidia-smi, nsys, git and flock; the real tools; bash
  5.2): dry and real modes green.

  | negative case | exit |
  |---|---|
  | P1 fails | 11 |
  | no report | 13 |
  | card busy | 9 |
  | a secret marker in a log | 15 |
  | used tags | 5 |
  | C fails | 0, with `census: none (C failed)` |

  - The rig found one bug, now fixed: a `$( [ … ] && echo … )` inside an assignment silently ended the real mode under
    `set -e`.
  - The rig cannot show real /proc, real nsys, the real harness, or what the census lines hold.

## 7. c.3's cheap slice: `LAMBDA_VM_TRACE_UPLOAD` (pre-registered 2026-09-30 15:11Z, before any box run)

The lead's go (2026-09-30): build the slice §4.7 recommends. It must stay byte-identical, sit behind a knob that is
off by default, and live on `fix2/1010-trace-upload`, branched from 73342bc66 (keep-futile, gated).

### 7.1 What was built [V: the code at `d0bf1ad8c`]

| part | where | default | knob on |
|---|---|---|---|
| staged column upload | `math_cuda::columns::DeviceColumns::upload` (the epoch's `upload_columns`, `multilinear_table.rs:595`) | one stream of per-column pageable `memcpy_htod` | 4 threads (`LAMBDA_VM_TRACE_UPLOAD_THREADS`, at most 8), each taking the next column through a staging pair of its own (`htod_staged_raw`, the I6 pairs, now addressable by raw pointer) |
| zero tails | the same | every byte sent | a column's all-zero tail ≥ 64 KiB (`ZERO_TAIL_MIN_BYTES`) is not sent; a `cuMemsetD8Async` zeroes it on the card |
| DECODE's prepared opening | `prove_continuation_scheduled` (`PreparedDecode`) | `decode_prepared_for` serial before the pipeline | on a helper from the ELF bytes; the first prove, or the observer's share just before it, joins it |
| log | `COLUMNS UPLOAD: <GB> in <s>, <GB> zero tails not sent, <GB/s> sent (pageable\|staged xN) t=` under `LAMBDA_VM_BASE_SPLIT`, on both arms; `BASE HEAD (WHIR): DECODE prepared (ahead)` and `… the first prove waited <s> for DECODE prepared` | | |

- **Knob:** `LAMBDA_VM_TRACE_UPLOAD`. `1` turns on both parts. `columns` or `head` turns on one, for attribution.
  Anything else is off.
- **Not built, and why:**
  - **The per-table upload overlapped with the commit's kernels.** An epoch's first stacked polynomial holds columns
    from every table (`BASE STACK #0: tables 0..19 … polys 2 vars 27`), so its Möbius transform cannot start before
    the last of those tables has landed. Overlap would need a different stacking, so the slice parallelizes the
    upload instead.
  - **Uploading epoch k+1 during epoch k's open.** It would hide the whole upload, but it adds ≈ 2.2 GB of VRAM per
    epoch next to the open's trees. i-whir measured the argue's high-water within 7 MiB of the budget in epoch 5.
    That is a separate lever with its own VRAM case.
- **Prior art that sets the risk** (`crypto/math-cuda/tests/h2d_bench.rs`, measured on a box before): one thread
  copying through a pinned pair ran at 13.9 GiB/s, slower than pageable at 18.6 GiB/s. Registering page locks on
  the fly cost as much as it saved. The slice therefore parallelizes the host copy (4 threads) instead of relying
  on one pinned stream. The laptop cannot measure the resulting rate [I]: it is what the A/B reads.

### 7.2 Gates (FAST2, job 185)

- Script: `thoughts/zf/box/extra-1010-trace-upload.txt` in `lambda_vm-tup` (md5 `a27f4da68fbba6edbbf5780ebde8f364`),
  through `zf-gates.sh d0bf1ad8c 1010-trace-upload`.
- Unit tests: the columns module's host tests (5). The setting and the prepared-ahead proving test (in the lib
  suite).
- On the card:
  - `columns_upload`: the store is read back from a pool dirtied with `u64::MAX`. It must equal the host columns
    and the pageable upload. A control shows the dirty pool hands back garbage.
  - `trace_upload_identity`: a `multi_prove` over three EQ tables, with the knob off, on and off again. The bytes
    must be identical. Controls: the two off proves agree, and the staged path ran and skipped tails. Mutation:
    another valid table changes the bytes.
- The production tree at the default and at `=1`: the same root, and the 5 ids = `ids-1010-f5.txt`. Each arm's
  own path lines must appear.
- Laptop, done: `make fmt`, `make lint` green; `cargo test -p math-cuda --lib columns::tests` 5/5; the setting
  test 1/1.

### 7.3 The A/B (FAST job 290): A = default, B = `LAMBDA_VM_TRACE_UPLOAD=1`, A B B A, wt1290–1293

- Script `tup-ab.sh` (md5 `e11bf9937a9917258feae64bfcf99f06`) with `tup_readout.py` (md5
  `1309e6265c87de78ce512e3f528066ea`), through the harness `zf-whir-arms.sh` 86951bc2.
- Bands, B − A unless named. The readout prints one row per mechanism row:

| row | what | band | why |
|---|---|---|---|
| M0 | each arm ran its own path; ids identical; proved; same GB uploaded | exact | the knob, and byte-identity at block level |
| M1 | A's Σ `COLUMNS UPLOAD` seconds in the base (control) | 1.3–2.3 s | G6: H2D pageable 1.84 s of op time in the base, 1.79 exclusive |
| M2 | B's Σ upload seconds | ≤ 1.3 s | 31–33 GB sent at ≥ 24 GB/s. Pinned DMA read 23.7 GiB/s on one stream in isolation and 39 GiB/s in a proof (`h2d_bench.rs`) [I] |
| M3 | Δ upload (the exclusive-H2D row) | [−1.2, −0.3] s | point −0.8 |
| M4 | B's zero tails not sent in the base (upload bytes skipped) | 3.0–4.6 GB | the census: 4.46 GB of row padding in the epochs' tables. Only tails ≥ 64 KiB are cut, and tables whose padding is not zero keep theirs |
| M5 | A: head start → epoch 0 executes (control) | 0.40–0.70 s | G6 P2 0.543 s |
| M6 | Δ head → epoch 0 executes (the head's critical path) | [−0.45, −0.15] s | `decode_prepared_for` 0.32–0.36 s leaves the path; the helper competes with epoch 0 for CPU |
| M7 | Δ head → epoch 0's commit | [−0.45, −0.05] s | |
| M8 | B's first prove waited for the opening | ≤ 0.10 s | the helper has ≈ 1.3 s before the first prove |
| M9 | Δ base | [−1.8, −0.3] s | M3 + M7 |
| — | **Δ whole run** | **−1.0 s [−1.8, −0.3]** | EFFECTIVE if ≤ −0.3; NO EFFECT if within ±0.3; REGRESSION if ≥ +0.3 |

- **What would falsify the mechanism, not just the number:**
  - M2 above 1.3 s means the host copy, not the bus, is the bottleneck. The next step would then be more threads,
    not tracegen.
  - M6 inside ±0.1 s means the head is not bound where §4.3 says.
  - A whole-run gain with M3 and M6 both null would be noise. With the A spread at 0.6 s (job 249), a single ABBA
    resolves only about 0.4 s.
- **Replication:** if the verdict is promoted to a default flip, a second ABBA on new tags comes first (memory:
  replicate an arm promoted after a run).

### 7.4 Addendum to §7.3: the rule that decides the pinned part, and one band (written 2026-09-30 15:14Z, before job 290's readout)

Written while job 290 was building its first arm: only `ARM wt1290 (1/4)` was in `tup.log`, and no `readout.txt`
existed. The readout script is md5-pinned and already deployed, so the rule below is applied by hand to the rows it
prints (M1, M2). The readout itself is unchanged.

- **The pinned part keeps its place only if B's exclusive H2D ≤ A's ÷ 1.3.** Read as B's mean Σ `COLUMNS UPLOAD`
  seconds (M2) against A's (M1), both taken over the base window.
  - This stands in for D-TRACE's skipped microbench and its 1.3× stop rule.
  - D-TRACE §1.1 puts one thread's host copy into pinned memory at ≈ 7.5 GB/s (ds941). Four threads reach 1.3× the
    20.5 GB/s pageable rate only if the host copies scale. That is the open question, and this row answers it.
- **If the 1.3× rule misses, or the whole-run Δ misses its band,** the landing is `head` only. The `head` part gets
  its own short A/B (`LAMBDA_VM_TRACE_UPLOAD=head` against off). Its share cannot be subtracted from this combined
  run.
- **One band for this A/B, replacing §7.3's −1.0 s [−1.8, −0.3]: Δ whole −0.9 s [−1.4, −0.3].**
  - **Head:** −0.36 s. This is D-TRACE §3.1's stage-0 row (head 1.83 → ≈ 1.47 s with `decode_prepared_for` off
    the path), and it agrees with M6's [−0.45, −0.15]. D-TRACE's larger head numbers need its device DECODE root,
    which this slice does not port: −0.97 s with the root on the card, −0.47 s without. They are not in this band.
  - **Upload:** −0.25 … −0.95 s. At the bottom, 4 threads × 7.5 GB/s host copies land near the 1.3× line on
    ≈ 31 GB sent, which is −0.4 s from rate plus −0.2 s from the 4 GB of zero tails not sent. At the top, the DMA
    runs at ≈ 39 GiB/s.
  - **Upper edge tightened** from −1.8 to −1.4 s: nothing in this slice can remove more than the head's 0.36 s plus
    the upload's 1.79 s exclusive, and the upload cannot fall to zero.
  - **D-TRACE's stage 1b** (pinned column slots written by the producer; −0.75 s [−1.1, −0.35], +5.9 GB of pinned
    host memory) is a different mechanism: the producer writes into pinned memory, with no copy on the prover's
    side. It sits inside this band, but this run does not measure it.
- **Verdict mapping, unchanged:** EFFECTIVE if Δ whole ≤ −0.3 s. The landing is the full knob only if the 1.3× rule
  also holds; otherwise it is `head` alone, after its own A/B.

### 7.5 Mauro's rulings (relayed 2026-09-30)

- **c.5:** set aside by Mauro (09-30) for a separate evaluation. It is not a next step, and its argue part goes to
  no one.
- **D-TRACE stage 0** (canonical row order) goes to f-sidle, not this lane.
- **D1** (G6's ledger re-run at the slice's landed head, FAST 290–294) runs only after the slice lands. Its stop
  rule is D-TRACE's: stop the generator track if the residual head + upload < 0.6 s and the producer's hand-off
  wait Σ ≥ 3 s.

### 7.6 Job 290's result and diagnosis (written 2026-09-30 15:31Z, from job 290's logs only; no new run)

**Readout** (FAST job 290, wt1290–1293, done 15:17Z): `REGRESSION Δ whole +0.35 s`. The A arms read 38.7 and 39.0 s,
the B arms 39.6 and 38.8 s. By §7.4 the combined landing is off, and FAST2 job 185 was cancelled before it started.

#### The columns path did engage [V: every `COLUMNS UPLOAD` line of the four logs]

- All 21 uploads in each B arm print `(staged x4)`; all 21 in each A arm print `(pageable)`.
- B left **7.82 GB** of the base's 33.84 GB behind as zero tails, the same in both B arms. "B's zero tails not sent"
  failed from ABOVE its band (3.0–4.6 GB), not below.
- The band was mine and it was wrong. It assumed zero tails are padding rows (4.46 GB), but real rows also end in
  long zero runs in many columns [I: which columns is not logged].
- The cut is correct: both B arms compressed and verified at the root, with ids identical to A's. A cut that dropped
  a nonzero value would break a constraint that level 1 checks.

#### Why the upload did not get faster

| arm | Σ upload in the base [L] | sent | sent rate | host memcpy into pinned, Σ over threads [L: `AFTER the WHIR base: staging:`] |
|---|---|---|---|---|
| A wt1290 / wt1293 | 1.791 / 1.781 s | 33.84 GB | 18.9 / 19.0 GB/s | — |
| B wt1291 / wt1292 | 1.722 / 1.655 s | 26.02 GB | 15.1 / 15.7 GB/s | 26.02 GB in 4.26 s (6.1 GB/s) / 26.03 GB in 3.98 s (6.5 GB/s) |

- **The 1.3× rule (§7.4) misses:** B 1.688 s against A's 1.786 ÷ 1.3 = 1.374 s, a ratio of 1.06×. Per upload, B's
  sent rate is 10.5–23.5 GB/s, mostly 14–18; A's is 16.0–22.9. The 23 % of bytes not sent only offset the slower
  rate: Δ upload −0.10 s.
- **Cause 1, the prover-side copy into pinned memory runs at 6.1–6.5 GB/s per thread** [L]. That is D-TRACE's
  ≈ 7.5 GB/s (ds941) and h2d_bench's chunked-pinned regression again. Four threads would need perfect overlap to
  reach ≈ 25 GB/s, only 1.3× pageable before any DMA limit.
- **Cause 2, the four threads overlapped only ≈ 2.4–2.5×** (4.26 ÷ 1.722, 3.98 ÷ 1.655) [A], and the code shows why
  [V code; the size of the effect is inferred]:
  - each column is one `htod_staged_raw` call that lends a pair and starts at buffer 0 (`device.rs:2051`);
  - the pool is LIFO (`:1769`), so a thread gets its own pair back and waits on buffer 0's event (`:1805-1806`),
    which is its previous column's DMA, before copying the next column;
  - for a column of one chunk (≤ 32 MiB, every column but the largest), a thread therefore alternates its copy and
    its own DMA with no overlap;
  - the four threads' DMAs also queue on one stream.
- The scan for the zero tails (`sent_len`) runs on the same threads and is not timed separately [I].

#### Why the head did not move the first commit [V: `BASE HEAD` / `BASE EPOCH 0` / `BASE PREP 0` lines]

Seconds after `BASE HEAD (WHIR): start`:

| arm | epoch 0 executes | collect | build | DECODE root done (helper) | prep waited for root | prep | hand-off |
|---|---|---|---|---|---|---|---|
| A wt1290 | 0.493 | 0.54 | 0.28 | 1.111 | 0.00 | 0.44 | **1.839** |
| A wt1293 | 0.449 | 0.56 | 0.31 | 1.223 | 0.00 | 0.44 | **1.835** |
| B wt1291 | 0.147 | 0.57 | **0.49** | **1.310** | 0.04 | 0.52 | **1.830** |
| B wt1292 | 0.149 | 0.56 | **0.50** | **1.387** | 0.11 | 0.46 | **1.843** |

- The head part did what it was built to do. Epoch 0 executes **0.32 s earlier**, the helper's opening took 0.28 s,
  and the first prove waited 0.00 s.
- But the path moved to the DECODE root. In A the producer binds: its build ends at 1.40 s, after the root. In B
  the root binds: prep waits for it.
- The root itself slowed from 1.11–1.22 s to 1.31–1.39 s, and epoch 0's build from 0.28–0.31 s to 0.49–0.50 s. In B
  the build runs while the root is still hashing; in A it mostly did not [I: contention, read from the overlap; no
  sampler ran].
- Net: the hand-off lands at the same instant (Δ head → epoch 0's commit −0.000 s).
- This row measures the head part alone. `upload_staged` runs only inside `DeviceColumns::upload`, which starts
  inside epoch 0's commit, so the columns part cannot act before that commit begins [V code].

#### The whole-run Δ

- Base Δ is +0.05 s (29.1 / 29.3 against 29.2 / 29.3).
- wt1291's extra lives in level 1's last node, L1N2: wall 8.71 s against 7.86–8.02 s in the other three arms.
  - Its harvest-epochs took 2.68 s (others 2.01–2.13) and its artifacts 1.08 s (others 0.27–0.29).
  - Its prove, 2.08 s, was the fastest of the four.
  - Neither part of the knob touches harvest or artifacts [I].
- The verdict stands as the readout printed it. The arm is not dropped after the fact.

#### Pre-registration for what could follow (written before any of it is queued; the lead decides)

1. **The head-only A/B as built** (`LAMBDA_VM_TRACE_UPLOAD=head` against off, A B B A, FAST 291, wt1294–1297).
   - Predicted from the rows above: Δ head → epoch 0 executes −0.32 s [−0.45, −0.20]; Δ head → epoch 0's commit
     0.00 s [−0.10, +0.05]; **Δ whole 0.0 s [−0.3, +0.3] ⇒ NO EFFECT.**
   - **Recommendation: do not run it.** The combined run already measured this part on its own window.
2. **Head + the DECODE root on the card** (D-TRACE §2.3: port #1009's `commitment_from_elf_device_or_host`, 0.29 s on
   the card, pinned to the host root). Not built.
   - Predicted: the producer path with an uncontended build is 0.15 + 0.07 + 0.56 + 0.30 + prep 0.44 ≈ 1.52 s.
     That gives Δ head → epoch 0's commit −0.30 s [−0.45, −0.15] and Δ whole −0.3 s [−0.5, 0.0].
   - One ABBA resolves only ≈ 0.4 s, so it would need 6–8 arms.
3. **The zero-tail skip alone, on the pageable path, with the scan hidden on a helper thread.** Not built.
   - At most 7.82 GB ÷ 19 GB/s ≈ −0.41 s of upload. Band −0.3 s [−0.45, −0.1] whole, which needs 6–8 arms.
4. **The pinned part is dropped by the rule.** Only D-TRACE's stage 1b removes the prover-side copy that caps it:
   the producer writes pinned slots, and the prover copies nothing. The logs do not measure the pinned DMA ceiling
   on FAST [I].

## 8. The zero-tail skip alone, and D-TRACE's pinned microbench (pre-registered 2026-09-30 16:00Z, before any box run)

The lead's decisions after job 290:
- the head-only A/B: not run;
- pinned staging: dropped;
- the zero-tail skip on the pageable path: GO, with the scan off the critical path, behind
  `LAMBDA_VM_TRACE_UPLOAD=zerotail` (default off), a byte-identity gate and the mutation;
- in the same FAST job, before the arms: D-TRACE's box request 1, a pure pinned-vs-pageable DMA microbench;
- the device DECODE root port: parked until both read.

### 8.1 What was built [V: `fix2/1010-zerotail` @ `d9aab005a`, two signed commits on 73342bc66]

- **`ceea5d178`, the zero-tail upload.**
  - Under `zerotail`, `DeviceColumns::upload` sends each column up to its last nonzero value; `memset_zeros` zeroes
    the rest on the card. A tail shorter than 64 KiB is still sent, the same `sent_len` rule job 290 measured.
  - The copies stay one stream of pageable `memcpy_htod`, exactly as the default.
  - Two scanner threads find the lengths ahead, in column order. Each reads only its tail, backwards, eight values
    at a time. A column whose length is not ready yet is scanned on the uploading thread instead of waited for, so a
    slow scanner costs time, never a hang. That time prints as `scan on the path`.
  - The log line is `COLUMNS UPLOAD: … (pageable)` or `(zerotail, scan on the path <s>)`.
  - **No pinned code and no head change** remain on this branch.
- **`d9aab005a`, the microbench** `crypto/math-cuda/tests/h2d_pinned_bench.rs` (ignored). It runs pageable
  `memcpy_htod` from a `Vec`, a `cuMemcpyHtoDAsync` from `cuMemHostAlloc` memory, and the same over two streams:
  - at 256 KiB, 1, 4, 16, 32 and 64 MiB per copy, over 2 GiB;
  - over epoch 0's real 393 columns (2.06 GB, from the G6 census);
  - with no host copy anywhere, each rate the median of five runs.
- **Where the scan runs, and why not on the producer.** The scanned slice is the uploaded slice, borrowed immutably
  for the whole upload, so a stale length cannot exist.
  - The producer has ≈ 10 s of slack, but a length computed there would travel through `EpochPrepped` →
    `CommittedTable` → the upload. The prover could not check it without re-reading the tail.
  - The upload-side scan also covers the global stage and level 1's uploads (2.39 GB of tails after the base in job
    290).
- **Laptop gates:** `make fmt`, `make lint` green; `cargo test -p math-cuda --lib columns::tests` 5/5, including the
  wide scan checked against a plain one for every length up to 40 and every position of the last nonzero value.

### 8.2 Part 1, the microbench (FAST job 291, before the arms)

- **Expected:**
  - pageable over the epoch-0 mix: 17–21 GB/s. Job 290's pageable sent rate in the proof was 18.9–19.0 GB/s.
  - pinned: h2d_bench.rs's earlier single-stream page-locked reading was 23.7 GiB/s (25.4 GB/s), 1.27–1.34× that.
    PCIe 5 × 16 would allow ≈ 2×. Ratio band 1.2–2.3× [I].
- **Stop rule (D-TRACE box request 1): the best pinned rate (one or two streams) ≥ 1.3× pageable over the epoch-0
  mix ⇒ stage 1b (producer writes pinned slots) is worth designing further; < 1.3× ⇒ stage 1b is dropped.**
- The per-size rows show whether small columns lose on either path. They inform the design; they do not gate.

### 8.3 Part 2, the A/B: A = default, B = `LAMBDA_VM_TRACE_UPLOAD=zerotail`, A B B A A B B A, wt1294–1301

- Harness `zf-whir-arms.sh` 86951bc2 with `ZF_ALLOW_OTHER_KNOBS=1`.
- Scripts: `zt-ab.sh` (md5 `53c0300e3f419c3aed2ae71f5681dc9a`) and `zt_readout.py` (md5
  `0c4755093707a6da8dad27cde247c52a`). The readout's selftest passes on job 290's real logs.

| row | what | band | from |
|---|---|---|---|
| M0 | each arm's own path; ids identical across settings; proved; the same GB to upload | exact | the knob; byte-identity at block level |
| M1 | A's Σ upload in the base (control) | 1.5–2.1 s | job 290 A: 1.791 / 1.781 s |
| M2 | B's zero tails not sent in the base, **each** B arm | **7.70–7.95 GB** | job 290 B: 7.82 GB in both arms. Same rule and same data: per upload identical except epoch 0's 0.573 / 0.570 GB (the HashMap-ordered tables) |
| M3 | Δ upload | [−0.50, −0.25] s | 7.82 GB ÷ 18.9 GB/s = −0.41 s |
| M4 | B's sent rate | ≥ 17.0 GB/s | the copies must stay pageable-fast with the scanners reading beside them (A: 18.9) |
| M5 | B's scan on the path, max over B arms of the base's Σ | ≤ 0.03 s | two scanners reading ≈ 0.5 GB of tails per epoch stay ahead of ≈ 0.08 s of copies |
| M6 | Δ head → epoch 0's commit | [−0.10, +0.10] s | the head is untouched |
| M7 | Δ base | [−0.55, −0.15] s | M3 |
| L1N2 | arms whose level-1 last node has harvest-epochs > 2.4 s or artifacts > 0.6 s, by setting | a count, not a gate | wt1291 read 2.68 / 1.08 s; the other 7 arms of jobs 249 and 290 read 2.01–2.13 / 0.27–0.29 s |
| — | **Δ whole, 8 arms** | **−0.3 s [−0.45, −0.1]** | the lead's band; M3 plus ≈ −0.1 s from the 2.39 GB of tails after the base, where they bind [I] |

- **Verdict:** EFFECTIVE if Δ whole ≤ −0.1 s; NO EFFECT if within (−0.1, +0.3); REGRESSION if ≥ +0.3. The Δ of each
  ABBA is printed alongside.
- **L1N2 reading:**
  - an anomaly in both settings is box noise;
  - one only in B arms points at the knob [I];
  - none is consistent with job 290's being noise.
- **What falsifies the mechanism:**
  - M4 under 17 GB/s means the scanners' reads slow the copies. The fix would then be the producer-side scan.
  - M5 over 0.03 s means the scanners fall behind.
  - A whole-run gain with M3 null would be noise.
- **Landing:** if EFFECTIVE, with M2–M5 PASS and job 186's gates green, the lead decides whether to flip the default.
  A flip is re-gated at the flipped sha.

### 8.4 Gates (FAST2 job 186)

- `extra-1010-zerotail.txt` (md5 `87ddfc86fb749d340db38ef57bfc8410`) through `zf-gates.sh d9aab005a 1010-zerotail`.
- Checks:
  - the columns module's unit tests;
  - `columns_upload` (dirty pool, value for value, and its control);
  - `trace_upload_identity` (off, on, off: identical bytes; control; mutation);
  - the production tree at the default and at `zerotail`, each with its own path lines and the 5 ids =
    `ids-1010-f5.txt` (md5 5cc1445a, which job 290's A arm also printed).

### 8.5 Results (FAST job 291, 16:01:43–16:08:53Z; written 2026-09-30 16:55Z)

- **Measured at `fix2/1010-zerotail` @ `d9aab005a`, on 73342bc66.**
  - The landing candidate `land/1010-zerotail` @ `aa332efef` sits on 7c8272701. It differs from the measured sha only
    by the whole-trees commits (3332c1bf3..7c8272701), and its change lines are identical to d9aab005a's.
  - The upload term does not depend on tree retention.
  - FAST2 job 186 gated d9aab005a (it started at 16:14:04Z, before the rebase was asked for). No gate has run at
    aa332efef.

#### Part 1, the microbench [V: `microbench.log`] — stage 1b **WORTH IT**

| columns | pageable GB/s | pinned GB/s | pinned × 2 streams | pinned ÷ pageable |
|---|---|---|---|---|
| 256 KiB × 8192 | 21.1 | 47.9 | 47.1 | 2.27× |
| 1 MiB × 2048 | 22.9 | 54.5 | 53.9 | 2.38× |
| 4 MiB × 512 | 23.0 | 57.0 | 56.9 | 2.48× |
| 16 / 32 / 64 MiB | 23.0 | 57.7–57.9 | 57.6–57.8 | 2.50–2.51× |
| **epoch 0's 393 columns (2.06 GB)** | **23.0** | **57.3** | **57.2** | **2.49×** |

- The stop rule (≥ 1.3×) is met 2.49×, so D-TRACE's stage 1b (the producer writes pinned slots) is worth designing.
- A second stream adds nothing: one stream already fills the link.
- The in-proof pageable rate is 18.9–19.0 GB/s against 23.0 alone, so the proof's own host load costs pageable
  ≈ 18 %.
- At 57 GB/s the base's 33.84 GB would take ≈ 0.59 s against today's 1.79 s [E, if the producer's writes into pinned
  memory cost the prover nothing].

#### Part 2, the A/B [V: `readout.txt`; per-phase sums from the tree logs] — **NO EFFECT, Δ whole −0.08 s**

| arm | whole | base (exact) | Σ upload | Σ commit | Σ argue | Σ open | L1N2 harvest / artifacts |
|---|---|---|---|---|---|---|---|
| A wt1294 | 38.6 | 29.021 | 1.780 | 9.08 | 11.58 | 4.88 | **2.63** / 0.29 |
| B wt1295 | 38.4 | 28.942 | 1.531 | 8.89 | 11.69 | 4.93 | 2.06 / 0.27 |
| B wt1296 | 38.4 | 29.106 | 1.521 | 8.88 | 11.84 | 4.93 | 2.10 / 0.28 |
| A wt1297 | 38.6 | 29.238 | 1.758 | 9.11 | 11.70 | 4.99 | 2.18 / 0.29 |
| A wt1298 | 38.5 | 29.114 | 1.790 | 9.13 | 11.63 | 4.91 | **2.43** / 0.28 |
| B wt1299 | 38.9 | 29.128 | 1.531 | 8.87 | 11.78 | 4.97 | 2.06 / **0.85** |
| B wt1300 | 38.4 | 29.097 | 1.538 | 8.89 | 11.72 | 4.98 | 2.18 / 0.27 |
| A wt1301 | 38.7 | 29.099 | 1.797 | 9.15 | 11.68 | 4.86 | **2.61** / **0.62** |

- **Mechanism rows, all PASS:**
  - M2: 7.82 GB not sent in each B arm.
  - M3: Δ upload −0.251 s, on the edge of [−0.50, −0.25].
  - M4: B's sent rate 17.0 GB/s, on the floor (A 19.0).
  - M5: scan on the path 0.000 s.
  - M6: head Δ −0.006 s.
- **M7 FAILs:** Δ base −0.05 s.
- **The commit gained what the upload saved:** Σ commit Δ −0.235 s, with every B arm below every A arm.
- **The argue gave most of it back:** Σ argue Δ +0.11 s (A 11.58–11.70, B 11.69–11.84; three of four B arms above
  A's maximum), and open +0.04 s (grind noise).
  - The argue's +0.11 s is either noise (t ≈ 2.6 on four arms each) or the argue absorbing the earlier start into
    more overlap with the producer's prep and build [I]. G6 saw no descheduling of the prover thread, so
    bandwidth, not CPU, would be the channel. No sampler ran in this job, so it cannot tell which.
- **Why M3 fell short of −0.41 s:** B's sent rate dropped from 19.0 to 17.0 GB/s. The per-column memsets and two
  scanners reading beside the copies are the candidates [I].
- **Whole:** A 38.60 (spread 0.20), B 38.52 (spread 0.50), Δ −0.08 s; per ABBA −0.20 / +0.05.
- **L1N2:** the anomaly appeared in 4 of 8 arms, **3 A and 1 B**. It is box noise, not the knob and not pinned
  memory, and wt1291's reading in job 290 was the same noise.
- **Decision by §8.3's rule:** NO EFFECT, so `zerotail` does not land. The stop rule makes stage 1b the live upload
  lever. It should be sized against the in-proof 19 GB/s with the argue-overlap caveat above, and D-TRACE owns its
  design.
