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
- **Then c.5 on Mauro's go.** Its argue part (0.87–1.33 s) is prover-only and can join i-argue [I]. Full device generators
  come last: they pay only once the card side falls ≈ 10 s and the producer's 19 s becomes the floor.

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
