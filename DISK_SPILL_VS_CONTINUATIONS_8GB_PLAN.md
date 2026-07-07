# Disk spill vs continuations under an 8 GB memory budget — investigation + experiment plan

> Target budget is **8 GB** (§10). The 4 GB material below is the earlier framing kept as
> context — the mechanisms, runbook and overhead analysis are budget-independent.

> **STATUS: PREP — server acquired, access pending (2026-07-07).** Target budget is now
> **8 GB** (see §10). Server: Scaleway `admin@51.15.139.16`, 8 GB RAM, publickey-only.
> Blockers before the run: (a) authorize the SSH key on the server (§10.1 — currently
> `Permission denied (publickey)`); (b) clone the repo there (private, §10.2). Then:
> (1) I relevé the box (cgroup v2 / swap / disk / toolchain); (2) build once with
> `--features disk-spill` (§8); (3) run the three cases under `/usr/bin/time -v` at an
> ~7.5 GB cgroup cap; (4) fill the matrix and apply §6. Anatomy of the overhead and what
> exactly we compare is in §9. Measured cycle counts and exact commands are below.
>
> Earlier standby note (2026-07-02): investigation + local prep were done; nothing had
> been run on a server yet.

> Goal: decide, with data, whether removing disk spill (branch `remove-disk-spill` in
> lambda_vm7) is justified. The framing question handed down: *"if I want to run in
> ~4 GB of RAM, what's better — disk spill or continuations? Build something that
> actually runs so we can see times (and the rest) and decide well."*

## 1. What the two mechanisms are, in THIS codebase

Both exist on `main` today. They solve the same problem (prove a workload whose
prover state does not fit in RAM) in different ways.

### Disk spill (`StorageMode::Disk`, cargo feature `disk-spill`)
- **Same monolithic proof, same verifier.** Nothing about the protocol or proof
  format changes. Only *where* intermediate prover state lives changes.
- When RAM is tight, LDE columns + Merkle trees are backed by `mmap`ed temp files
  instead of the heap (`crypto/stark/src/table.rs`, `crypto/crypto/src/mmap_util.rs`).
- Storage mode is chosen automatically: `prover/src/auto_storage.rs::decide()`
  analytically estimates peak RAM from the trace shape (`peak_bytes`) and compares it
  to `sysinfo::available_memory() * 9/10`. If the estimate exceeds that, it picks
  `Disk`, else `Ram`. `FORCE_DISK_SPILL=1` forces `Disk`.
- Call site: `prover/src/lib.rs:816-820`, gated behind `#[cfg(feature = "disk-spill")]`.
- Tradeoff (from `storage_mode.rs`): **"Disk trades wall time for peak RAM."** Works
  for any size, slower due to I/O, zero protocol change.

### Continuations (`prover/src/continuation.rs::prove_continuation`)
- Splits execution into epochs of `2^epoch_size_log2` cycles. Each epoch is proven
  **independently and in RAM**, plus one cross-epoch global-memory linking proof.
- Peak RAM is bounded by the *epoch* trace, not the whole run → "flat peak memory".
  The lever is `epoch_size_log2`: smaller epoch ⇒ less RAM, more epochs.
- **Continuations do NOT spill**: `multi_prove` is called with an explicit
  `StorageMode::Ram` (`continuation.rs:408-410, 581-582`).
- Costs vs monolithic: (a) a **bundle of N proofs** instead of one → larger proof
  size; (b) a **different verify path** (`verify_continuation`); (c) fixed per-epoch
  overhead; (d) a hard cap of `MAX_EPOCHS = 2^20` epochs; (e) more moving parts
  (global memory / local-to-global binding).
- CLI floor: `--epoch-size-log2` must be ≥ 18 ("tiny epochs are dominated by fixed
  overhead"). Default 20.

### Known numbers (already in the CLI `--help`, ethrex 10-transfer, distinct accounts)
Peak heap by epoch size: **log2 19 ≈ 6.9 GB, 20 ≈ 9.5 GB, 21 ≈ 15.8 GB, 22 ≈ 26.8 GB.**
All are above 4 GB. Extrapolating ~linearly, `log2 18 ≈ 3.5 GB` — i.e. for this
workload continuations only *maybe* fits 4 GB at the smallest CLI-allowed epoch, and
smaller epochs are rejected by the CLI. This is the crux the experiment must pin down.

## 2. What lambda_vm7 (`remove-disk-spill`) removes
5 commits, ~2098 lines net deleted: `auto_storage.rs` (309), disk-spill machinery in
`trace_builder.rs`/`table.rs`/`prover.rs`, `spill_safe.rs`, `mmap_util.rs`, the
`disk-spill` cargo features, the calibration script + `calibration.rs`, and the
disk-spill CI job. Continuations is left intact. So the branch's implicit bet is:
**"continuations covers the low-memory case; disk spill is dead weight."** The
experiment tests exactly that bet at 4 GB.

## 3. Existing tooling we can reuse (don't build from scratch)
- `prover/benches/bench_continuation.rs` — purpose-built, `harness = false`. Modes:
  `count` (cycles), `footprint` (touched-memory breakdown), `main` (monolithic prove),
  `cont <epoch_size_log2>` (continuation prove+verify). Designed to be wrapped in
  `/usr/bin/time -v` to capture peak RSS. NOTE: `main` mode uses `prove_with_inputs`,
  so to exercise the *spill* path it must be built `--features disk-spill` and run with
  `FORCE_DISK_SPILL=1`.
- CLI: `cli prove [--continuations --epoch-size-log2 N]` writes the proof/bundle to a
  file (→ proof-size metric) and prints proving time + epoch count.
- `docs/continuations_design.md` — design doc.
- `scripts/calibrate_threshold.sh` + `prover/tests/calibration.rs` — threshold
  calibration for auto-storage.

## 4. Key subtlety for a "4 GB" run
`auto_storage` reads `sysinfo::available_memory()`. Under a cgroup/container limit on
cgroup v2, sysinfo reads the *limit*; under other setups it may read host memory and
mis-decide. To get a clean, reproducible 4 GB test, either:
- run inside a hard cap (`systemd-run --scope -p MemoryMax=4G …`, or a container with
  `--memory=4g`) — cgroup v2 also fixes the sysinfo reading; and/or
- use `FORCE_DISK_SPILL=1` to force the spill path regardless, and *verify* peak RSS
  stays under 4 GB via `time -v`.
Peak RSS must be measured with Linux `/usr/bin/time -v` → this needs the Linux bench
box (this dev machine is macOS).

## 5. Experiment design

**Metrics per run:** wall time, peak RSS (`Maximum resident set size`), proof/bundle
size (bytes), correctness (verify passes + output matches monolithic).

**Ceiling:** hard 4 GB cap via cgroup/container (so an over-budget run OOMs instead of
swapping and lying about time).

**Measured locally (2026-07-02, `bench_continuation count`, executor only, no proving):**
empty_block 992,414 cyc · simple_tx 1,795,547 cyc · 3_transfers 2,945,337 cyc ·
**10_transfers 6,880,199 cyc** (the exact workload the `--help` heap numbers reference).
Trying to extrapolate a peak-heap "floor" from the four `--help` points (log2 19–22) is
unreliable at the small-epoch end (the curve is clearly sub-linear), so whether
continuations fits 4 GB at the CLI floor (log2 18) genuinely has to be *measured*, not
guessed — that is the whole point of the run.

**Inputs (2–3, increasing size so we see where each mode breaks):**
- small — fits in RAM both ways (sanity + fixed-overhead baseline).
- ethrex transfers — the realistic target (a variant that peaks well above 4 GB
  monolithic). `executor/tests/ethrex_3_transfers.bin` and the 10-transfer input from
  the CLI help are candidates.
- optionally a scalable `fibonacci(n)` to dial the peak precisely around 4 GB.

**Runs, all under the 4 GB cap + `time -v`:**
1. Monolithic, no spill (pure RAM) — expected to OOM on the big input. Documents *why*
   spill/continuations exist at all.
2. Monolithic + disk spill (`--features disk-spill`, `FORCE_DISK_SPILL=1`) — time, peak
   RSS (should stay < 4 GB), proof size, verify.
3. Continuations sweep over `epoch_size_log2` from the smallest that fits upward — for
   each: time, peak RSS, bundle size, epochs, verify. Find the largest epoch that
   stays ≤ 4 GB (biggest epoch = least overhead).

**Cheap-first pre-step (local, no server, no proving):** run `bench_continuation count`
/ the analytical `peak_bytes` estimator per candidate input and epoch size to *predict*
which configs can fit 4 GB, so the server sweep is targeted instead of blind.

## 6. Decision criteria
- Continuations fits 4 GB at an acceptable epoch size **and** time is comparable-or-
  better **and** bundle size is acceptable ⇒ **removing disk spill is justified**
  (less code to maintain, no I/O tax).
- Some target input fits 4 GB **only** via disk spill (continuations can't get small
  enough under the CLI floor, or the bundle explodes) ⇒ **keep disk spill**, or keep
  both and pick per-workload.
- Realistic likely outcome to watch for: continuations wins on RAM ceiling but pays in
  proof size / total time; disk spill wins on simplicity + single proof but pays in
  wall time. The 4 GB number decides which tax is acceptable.

## 7. Open decisions (need Juan)
1. **Where to run** — `vm-benchmarks-1` (needs explicit authorization), or a local
   Linux VM/container?
2. **Which inputs** — is `ethrex_3_transfers.bin` representative, or should we target
   the 10-transfer workload from the CLI help / a bigger block?
3. **Is 4 GB the only ceiling**, or also sweep e.g. 2/8/16 GB to draw the crossover
   curve (more runs, better decision)?

## 8. Verified runbook (ground-truth against the code, 2026-07-07)

Three gotchas confirmed by reading the source — all easy to get wrong:

1. **`sysinfo` is NOT cgroup-aware.** `auto_storage::decide()` (`prover/src/auto_storage.rs:219-228`) compares the analytical `peak_bytes` estimate against `System::available_memory()` (`auto_storage.rs:300-309`), threshold `available * 9/10` (`SAFETY_FRACTION_NUM/DEN`, `auto_storage.rs:44-46`). There is no cgroup/`/proc/meminfo` handling anywhere, so inside a `--memory=4g` cgroup it reads the **host's** available RAM and may pick `Ram` and OOM before spilling.
   - For the **spill** run, force it: `FORCE_DISK_SPILL=1` (`auto_storage.rs:220`; *any* value, even empty, triggers it — it short-circuits before the estimate).
   - For the **no-spill baseline**, leave `FORCE` unset → it stays `Ram` → OOMs under the cap. That is the baseline we want (no separate feature-off build needed).
2. **`TMPDIR` gotcha (load-bearing).** Spill files are `tempfile::tempfile()` (`crypto/crypto/src/mmap_util.rs:13-43`); on systemd distros `/tmp` is tmpfs (RAM-backed) and its pages count against the cgroup cap → the spill run OOMs too *and* saves no RAM. The code itself warns about this (`mmap_util.rs:52-53`). Set `TMPDIR` to a disk-backed path (e.g. `/var/tmp/spill`).
3. **Proof size comes from the CLI, not the bench.** The bench binary proves in-memory and never writes a file. The CLI `prove` requires `-o/--output` and bincode-writes the bundle (`bin/cli/src/main.rs:127-128,468-489,641-660`) → proof size = output file size. The `epoch_size_log2 >= 18` floor is CLI-only (`main.rs:22,769`); the library floor is `>= 2` (`continuation.rs:766`) and the bench default is 16 — so probe epochs **< 18** with the bench binary, not the CLI.

**Tools:** CLI under `/usr/bin/time -v` gives wall + peak RSS + proof size (correctness via `cli verify <proof> <elf> [--continuations]`, `main.rs:167-187`). Bench binary (`prover/benches/bench_continuation.rs`, modes `count|footprint|main|cont <log2>`) does prove+verify in one and can go below the CLI epoch floor.

**Build (one disk-spill build covers all three cases):**
```bash
cargo build --release --features disk-spill -p cli
cargo build --release --features disk-spill -p lambda-vm-prover --bench bench_continuation
```
Package names confirmed: prover crate `lambda-vm-prover` (lib `lambda_vm_prover`), CLI package+binary `cli`. Blowup is 2 on every path (monolithic `prove_with_inputs` `lib.rs:782`, bench `cont` `bench_continuation.rs:120`, CLI `--blowup` default 2) → fair comparison.

**Runs (Linux, cgroup v2; per input — `ethrex_10_transfers.bin` hard, `ethrex_3_transfers.bin` medium):**
```bash
ELF=executor/program_artifacts/rust/ethrex.elf
IN=executor/tests/ethrex_10_transfers.bin
mkdir -p /var/tmp/spill

# (1) monolithic, no spill → OOM expected on the hard input (baseline for WHY)
systemd-run --scope -p MemoryMax=4G -p MemorySwapMax=0 \
  bash -c "/usr/bin/time -v ./target/release/cli prove $ELF -o /var/tmp/mono.bin --private-input $IN"

# (2) monolithic + spill
systemd-run --scope -p MemoryMax=4G -p MemorySwapMax=0 \
  bash -c "TMPDIR=/var/tmp/spill FORCE_DISK_SPILL=1 /usr/bin/time -v ./target/release/cli prove $ELF -o /var/tmp/spill.bin --private-input $IN"
ls -l /var/tmp/spill.bin && ./target/release/cli verify /var/tmp/spill.bin $ELF

# (3) continuations sweep (>=18 via CLI)
for k in 18 19 20; do
  systemd-run --scope -p MemoryMax=4G -p MemorySwapMax=0 \
    bash -c "/usr/bin/time -v ./target/release/cli prove $ELF -o /var/tmp/cont_$k.bin --private-input $IN --continuations --epoch-size-log2 $k"
  ls -l /var/tmp/cont_$k.bin && ./target/release/cli verify /var/tmp/cont_$k.bin $ELF --continuations
done
# probe <18 only if 18 doesn't fit 4 GB (bench binary):
BENCH=$(ls target/release/deps/bench_continuation-* | grep -v '\.d$')
for k in 16 17; do
  systemd-run --scope -p MemoryMax=4G -p MemorySwapMax=0 \
    bash -c "BENCH_PRIVATE_INPUT=$IN /usr/bin/time -v $BENCH cont $ELF $k"
done
```
For an 8 GB budget, rerun with `MemoryMax=8G` (physical 8 GB box → realistic cap ceiling ~7.5 GB).

**Metric extraction:** from `time -v` → `Maximum resident set size` (KB) and `Elapsed (wall clock) time`; proof size = `ls -l` of the `-o` file; correctness = `cli verify` exit 0. `MemorySwapMax=0` is essential so an over-budget run OOMs instead of swapping and lying about wall time. Docker equivalent: `docker run --memory=4g --memory-swap=4g -e TMPDIR=/spill …`.

## 9. Anatomy of the overhead — and what exactly we compare

This is the crux: continuations is **not** "one prove vs another prove". The two are
structurally different, and that difference *is* the overhead.

### What "one prove" means in each case
- **Monolithic** → ONE STARK over the whole trace. One `VmProof`, one verify.
- **Continuations** → a composite bundle (`ContinuationProof`, `continuation.rs:382`):
  ```
  ContinuationProof = [ EpochProof_1, ..., EpochProof_N ]   ← N independent epoch STARKs
                    +   global: MultiProof                   ← 1 proof that links them
                    +   touched_page_bases
  ```
  Each `EpochProof` is a full, independent STARK of that epoch (`prove_epoch` →
  `Prover::multi_prove`, `continuation.rs:505`); `global` is *another* whole STARK that
  anchors cross-epoch memory consistency on the GlobalMemory bus. A different verify
  path (`verify_continuation`).

### Where the overhead comes from (everything continuations does that monolithic doesn't)
1. **Per-epoch fixed STARK cost × N.** Every epoch has its own LDE + Merkle commit +
   FRI, with a fixed floor that does not shrink with epoch size. That is why tiny epochs
   are "dominated by fixed overhead" → the CLI floor `epoch_size_log2 >= 18`.
2. **An extra L2G (local-to-global) table per epoch** (`continuation.rs:501-504`): one
   row per memory cell the epoch touches, with its own commitment. Monolithic has none.
3. **REGISTER preprocessed with FINI** (`NUM_PREPROCESSED_COLS_WITH_FINI`,
   `continuation.rs:420-423`): extra preprocessed columns per epoch to bind
   `init(epoch i+1) == fini(epoch i)`. Monolithic doesn't need it.
4. **The whole global linking proof** (`global: MultiProof`) — pure additional work.
5. **A bundle of N proofs** instead of one → on-disk size grows ~linearly with N.

**Key scaling law:** overhead ∝ **number of epochs** ≈ `total_cycles / 2^epoch_size_log2`.
Smaller epoch ⇒ lower peak RAM but MORE overhead (more epochs). That tension is exactly
what the sweep measures. The design bounds the *number* of epochs to `< 2^20`
(`MAX_EPOCHS`, `local_to_global.rs:83`), not their size.

### What we compare (end-to-end totals for proving the SAME program to the SAME output)
| Metric | Monolithic | Continuations |
|---|---|---|
| Wall time | the single prove | sum of ALL epochs + the global proof |
| Peak RSS | peak of the whole run | peak of a SINGLE epoch (epochs run one at a time, free memory between) |
| On-disk size | 1 `VmProof` | the whole bundle (`N × EpochProof + global`) |
| Correctness | verify OK | `verify_continuation` OK **and identical public output** |

The "identical public output" is what makes the comparison fair: same input, same
result, different machinery.

### How we quantify "how much overhead"
Reference zero-point = **monolithic in RAM, no cap** (the fast, no-mechanism prove):
- **spill overhead** = `time(monolithic+spill) − time(monolithic-RAM)` → the I/O tax;
  proof size is *identical* (protocol unchanged).
- **continuations overhead** = `time(continuation total) − time(monolithic-RAM)` **plus**
  the bundle-size blow-up → the structural tax (items 1–5).

Methodological note: that zero-point is exactly the config that OOMs at the budget, so
its *time* must be measured **without** the cap (or on a bigger box) purely to establish
the reference; under the cap it only serves as the "why" baseline (OOM). Bottom line:
both taxes buy the same thing (fitting the budget); the experiment measures which tax is
smaller, and the continuation tax scales with how many epochs are needed to hit the cap.

## 10. Re-plan for an 8 GB budget + server

Target budget updated from 4 GB to **8 GB** (2026-07-07, user's call). Everything in §8
still applies; only the cap value and the expected matrix change.

### 10.1 Server + access (blocker)
- Server: `admin@51.15.139.16` (Scaleway, 8 GB RAM), **publickey-only** (no password →
  `ssh-copy-id` won't work).
- The workstation key `~/.ssh/id_rsa.pub` (`jbulacios@fi.uba.ar`, saved verbatim in
  `~/Documents/ssh_public_key.md`) is **not yet authorized** — `ssh` returns
  `Permission denied (publickey)`. Authorize it out-of-band (Scaleway console or a
  machine that already has access): append the pubkey to the server's
  `~/.ssh/authorized_keys` (`chmod 700 ~/.ssh`, `chmod 600 authorized_keys`).
- Once authorized, relevé the box before running: OS/kernel, `stat -fc %T /sys/fs/cgroup`
  (want `cgroup2fs`), `swapon --show`, `free -h`, `df -h`, whether `/tmp` and `/var/tmp`
  are tmpfs, `systemd-run` presence, `/usr/bin/time` (GNU) presence, `rustc`/`cargo`.

### 10.2 Repo (clone on the server — user does this)
Private repo `git@github.com:yetanotherco/lambda_vm.git`. Cloning on the server needs
GitHub auth: easiest is SSH agent forwarding (`ssh -A`, after `ssh-add ~/.ssh/id_rsa` on
the workstation — the agent is currently empty), or a deploy key / HTTPS token.

### 10.3 Physical-headroom caveat (important)
The box has **8 GB physical**. `MemoryMax=8G` lets the cgroup consume all 8 GB, starving
the OS → thrash / global OOM (could kill sshd). So the **clean testable budget on this
box is ~7–7.5 GB** (leave ~0.5–1 GB for the OS). A *true* 8 GB budget (process alone gets
8 GB) needs a box with ≥12 GB. **Open decision for Juan:** accept ~7.5 GB as the tested
budget (with this note), or get a bigger box for a genuine 8 GB.

### 10.4 Expected matrix at 8 GB (ethrex 10-transfers, from the CLI `--help` heap numbers)
| Run | Expected peak | Fits ~7.5 GB? |
|---|---|---|
| Monolithic, no spill | ≫ 8 GB | ❌ OOM (baseline for WHY) |
| Monolithic + spill | bounded (mmap) | ✅ |
| Continuations log2=18 | ~3.5 GB | ✅ (comfortable) |
| Continuations log2=19 | ~6.9 GB | ⚠️ **borderline — the case to measure precisely** |
| Continuations log2=20 | ~9.5 GB | ❌ OOM |

### 10.5 What changes vs the 4 GB plan
Bigger budget lets continuations use **larger epochs** (19 instead of 18) → **fewer
epochs → less overhead** (per §9's scaling law). So at 8 GB the continuation tax should
be *lower* than at 4 GB. The decisive measurement is whether log2=19 (~6.9 GB) fits the
clean ~7.5 GB cap or spills over — that determines whether 8 GB lets continuations win
comfortably or pins it to log2=18 like the 4 GB case.

### 10.6 Sweep + cap for 8 GB
Sweep `epoch_size_log2 = 18, 19, 20` (20 to confirm it OOMs). Cap wrapper:
`systemd-run --scope -p MemoryMax=7500M -p MemorySwapMax=0 …` (everything else per §8:
`FORCE_DISK_SPILL=1` + disk-backed `TMPDIR` for the spill run, CLI `-o` for proof size,
`cli verify` for correctness, bench binary to probe epochs below the CLI floor of 18).

### 10.7 Runner script (prepared locally, `bash -n` + shellcheck clean)
`bench_spill_vs_cont.sh` (repo root, local — **not** committed, so `scp` it to the server;
it won't come with the GitHub clone) automates the whole matrix: preflight (cgroup v2 /
swap / disk-backed TMPDIR / capped-scope self-test / inputs present), one disk-spill
build, the three run classes per input under `MemoryMax=$CAP -p MemorySwapMax=0` with
`/usr/bin/time -v`, then parses peak RSS + wall + proof size + verify into `RESULTS.md`.
`set -e` is off on purpose so the expected OOM baseline is recorded, not fatal.

Config vars to confirm on the box after the relevamiento (defaults in the script):
`REPO` (clone path), `SPILL_TMPDIR` (must be disk-backed, not tmpfs), `SYSTEMD_RUN`
(plain / `sudo` / `--user` depending on cgroup delegation — preflight tests it), `CAP`
(`7500M`). Run: `scp bench_spill_vs_cont.sh admin@51.15.139.16:~/ && ssh … 'REPO=~/lambda_vm ./bench_spill_vs_cont.sh'`.
Pull results back promptly: `scp -r admin@51.15.139.16:~/spill_vs_cont_* .`.
