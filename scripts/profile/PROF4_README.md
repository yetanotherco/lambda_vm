# prof4: a GPU-counter profile of the block prover (for Mauro's RTX 5090)

One script, `prof4.sh`, measures where the block prover's time goes on a GPU whose performance counters are readable.
The rented boxes block the counters, so this machine is the only place to collect them. You run it; nobody logs in.

## What it measures

- **Run A** (Nsight Systems, with GPU metrics at 2 kHz). Each workload is proved once. The script splits each run into
  stages using the prover's own log lines: phase A, phase B, level 0, the interior levels, and the whole. For each stage
  it reports card busy %, SMs active, SM issue, warps in flight, DRAM and PCIe throughput, kernel time by family, the
  top kernels, and host CPU cores.
- **Run B** (Nsight Compute at base clocks). For each kernel that carries the bench block's time, it profiles one
  representative launch: SM and memory throughput, occupancy, pipes, and stall reasons. The kernels are picked from
  run A's bench captures.

| workload | what it is | block |
|---|---|---|
| W, Poseidon1 base (the main arm) | #1014 no-epoch WHIR on `whirp1/s6b-box` 4082b0df3, the build of the 10-07 same-hash benchmark | bench 25368371, median 25475471 |
| W, RPX base (reference) | the same binary, base hash RPX | median 25475471 |
| S, Poseidon1 base | #1013 no-epoch STARK on `noepoch/stark` 4841cf5a9 | bench 25368371, median 25475471 |

The p90 block is left out. It needs about 77–83 GiB of RAM at the box posture, and this machine has 60. The rented
boxes cover it, without counters.

## What to install first

- The NVIDIA driver and a CUDA toolkit with `nvcc` (13.0 is fine).
- Nsight Systems (`nsys`) and Nsight Compute (`ncu`) 2025.3 or newer.
- GPU counters readable by your user. They already were on 10-01.
- `rustup` with the toolchain 1.94.0: `rustup toolchain install 1.94.0`.
- `git`, `curl`, and `python3` 3.8 or newer. The script uses only the Python standard library; no pip packages.
- **Disk:** at least 80 GB free where you run it (builds ≈ 15 GB, captures ≈ 10 GB, a possible spill ≈ 30 GB).
- **RAM:** 56 GiB or more, with at least 46 GiB free before each run. Close big programs first.

## The command

Run it from any directory, inside `tmux` or `screen` so a dropped session does not end it. Replace `<SHA>` with
the commit the lead sends (or the tip of `prof/iprof4-5090`):

```bash
curl -fsSLO https://raw.githubusercontent.com/yetanotherco/lambda_vm/<SHA>/scripts/profile/prof4.sh && PROF4_REV=<SHA> bash prof4.sh
```

It works in `./lambda-vm-prof4`: a clone of the repository, two worktrees, two builds, the fixtures, and the results.
It checks its tools first (≈ 1–3 min, including the clone) and stops with a clear message if something is missing.

Other modes:

- `bash prof4.sh --preflight-only` runs the checks alone.
- `bash prof4.sh --quick` runs only W with the Poseidon1 base (bench + median) and 5 ncu passes.
- `--skip-build` reuses the builds of an earlier run in the same directory.
- `--gpu N` picks another GPU. `--dir DIR` picks another work directory.

## How long

| mode | time |
|---|---|
| full (default) | ≈ 75–110 min: builds ≈ 10–20, run A ≈ 25–35, run B ≈ 35–45. A deadline stops it at 170 min. |
| `--quick` | ≈ 40–55 min |

The GPU is busy the whole time. Don't run other GPU work, and leave most of the RAM to it.

## Safety

- The profiled runs get an explicit environment (`env -i`): nothing from your shell reaches them.
- A memory watchdog stops a run if available RAM falls below 1.5 GiB. The script reports the stop and moves on, so
  the desktop never runs out of memory.
- If a run aborts on the device (on driver 580.65, the 10-01 run saw freed device memory not being reused), the script
  retries it once with a smaller VRAM budget and records that in the results.
- The raw captures (`.nsys-rep`, `.ncu-rep`, `.sqlite`) embed the machine's environment. They stay in
  `lambda-vm-prof4/runs/*/big/`. Each one is deleted right after its summary, unless you pass `--keep-raw`.
  **Never send these files.**

## What to send back

When it finishes, it prints one path:

```
=== DONE. Send back this one file (text only): <dir>/prof4-send-<stamp>.tar.gz
```

Send that tarball. It holds plain text only: stage tables, kernel lists, ncu metrics, logs, and run facts (tool
versions, GPU, driver). It contains no environment. Your home directory and the work directory are replaced by
`<HOME>` and `<W>`. Before writing it, the script scans it for credential markers and Nsight files. If the scan finds
anything, it writes no tarball. It then says which lines matched (the details are in `runs/*/big/bundle-scan.txt`;
don't send that file).

## Reading the results

- `send/SUMMARY.md`: every stage table, then run B's table of kernels and their roofs.
- `send/runa/<run>/`: `stages.tsv`, `families.tsv`, `kernels.tsv`, `metrics.tsv` (every GPU metric per stage),
  `copies.tsv`, the stamped prover log, and `nsys_*.csv`.
- `send/ncu/<pass>.launches.tsv` and `.details.csv`; `send/plan-w.tsv` / `plan-s.tsv` (which launch each pass aimed at).
- `send/preflight.txt`, `facts.txt`, `steps.tsv`, `prof4.log`.

ncu runs the SMs near 2.0 GHz at base clocks, while the block runs near 2.5–2.76 GHz. A compute-bound launch therefore
reads about 1.3–1.4× its in-block time.
