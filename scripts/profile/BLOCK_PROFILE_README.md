# block_profile: where PR #1014's block prover spends a median and a large block (for one RTX 5090)

One script, `block_profile.sh`, profiles PR #1014's head (`origin/noepoch/whir` 85eb596e0 plus the three group-cap
commits, `5d1b28030`, pinned by commit and by tree) on a machine whose GPU counters are readable. The machine's owner runs it; nobody logs in.

## What it measures

- **Plain runs and Nsight Systems captures of the same block.** The median block is proved once to warm up, once plain
  (the control) and twice under `nsys` (CUDA and OS-runtime trace, no GPU metrics). The large block is proved once plain
  and once under `nsys`. Every run uses the production settings plus `LFM_CARD_TRACE=1`, which prints one line each time
  a recursion proof takes the card.
- **Per stage, from the prover's own log lines:** card busy and idle, the card's power and SM clock, the kernels by
  family and by name, host CPU cores, and how much of the first recursion level's idle falls inside a proof's turn on
  the card.
- **Nsight Compute on a few kernels** (base clocks, one representative launch each, on the small block): the first-round
  tile pass, the commit's leaf hashing, and the lookup argument's layer rounds; in full mode also the other tile kernels,
  the fraction fold, the constraint rounds and the transform.
- **Checks:** every run proves and verifies with the right block shape; each capture costs within 0.99–1.04× its control.

| block | runs | memory |
|---|---|---|
| median 25475471 | warm-up, control, 2 captures (`--quick`: 1) | the prover's default target, MemTotal − 10 GiB; ≥ 46 GiB available before each run |
| large 25481021 | control, 1 capture (skipped by `--quick` and `--no-large`) | `LAMBDA_VM_BLOCK_SPILL_TARGET_GIB=50`; runs only with ≥ 54 GiB available; it peaked near 51 GiB at a 54 GiB target |
| small 25368371 | 1 capture to plan the ncu passes, then one ncu pass per kernel | ≈ 11 GiB |

## What to install first

- the NVIDIA driver (595.71 is the tested one) and a CUDA toolkit with `nvcc`;
- Nsight Systems (`nsys`) and Nsight Compute (`ncu`) 2025.3 or newer, with GPU counters readable by your user;
- `rustup` with the toolchain 1.94.0 (`rustup toolchain install 1.94.0`);
- `git`, `curl`, `top`, and `python3` 3.8 or newer; `numpy` is optional (with it, the captures' per-bin tables are
  extracted too: `apt install python3-numpy`);
- **disk:** 60 GB free in the work directory; **RAM:** 56 GiB or more (58 for the large block).

## The command

Run it inside `tmux` or `screen`. Replace `<SHA>` with the commit you were sent:

```bash
curl -fsSLO https://raw.githubusercontent.com/yetanotherco/lambda_vm/<SHA>/scripts/profile/block_profile.sh && PROFILE_REV=<SHA> bash block_profile.sh
```

It works in `~/mauro/lambda-vm-profile`: a clone, one worktree, one build, the fixtures, and the results. It checks its
tools first (≈ 2 min, including the clone) and stops with a clear message if something is missing.

Other modes:

- `--preflight-only` runs the checks alone.
- `--quick`: median control + 1 capture, no large block, 3 ncu passes.
- `--no-large`, `--no-ncu`, `--skip-build` (reuse the build in the same directory), `--gpu N`, `--dir DIR`.

## How long

| mode | time |
|---|---|
| full (default) | ≈ 65–95 min: build ≈ 10–20, median runs ≈ 20, large runs ≈ 20, ncu ≈ 12. A deadline stops it at 150 min. |
| `--quick` | ≈ 30–45 min |

Leave the GPU and most of the RAM to it.

## Safety

- The profiled runs get an explicit environment (`env -i`): nothing from your shell reaches them.
- A memory watchdog stops a run if available RAM falls below 1.5 GiB, and the large block waits for 54 GiB free or is
  skipped. The script says so in its log and goes on.
- A run that aborts on the device gets 60 s to exit and is then stopped.
- The raw captures (`.nsys-rep`, `.ncu-rep`, `.sqlite`) embed the machine's environment. They stay in
  `runs/*/raw/` and are deleted right after their summary, unless you pass `--keep-raw`. **Never send these files.**

## What to send back

When it finishes it prints one path:

```
=== DONE. Send back this one file (text only): <dir>/profile-send-<stamp>.tar.gz
```

That tarball is plain text only: per-run stage tables, kernel tables, card-hold lines, ncu metrics, logs, and run facts
(tool versions, GPU, driver). It contains no environment; your home and the work directory are replaced by `<HOME>` and
`<W>`. Before writing it, the script scans it for credential markers and Nsight files, and writes nothing if the scan
finds any (it names the lines in `runs/*/raw/bundle-scan.txt`; don't send that file).

## Reading the results

- `send/SUMMARY.md`: run facts, the ncu table (kernel, SM %, DRAM %, occupancy, top pipe, top stall, bound), then the
  rows (`readout.txt`).
- `send/runs/<run>/`: `stage.json`, `win.tsv` (per stage: busy, idle, host cores), `idle.tsv` (power, SM clock),
  `holds.tsv` and `l0split.json` (card-permit holds), `runa/` (stage tables from the export), `fam.tsv` / `top.tsv` /
  `gaps.tsv` and `x/` (per-bin tables, with numpy), `stats/` (nsys kernel / copy / API sums), the stamped prover log.
- `send/plan.tsv` and `send/ncu/<pass>.launches.tsv` / `.details.csv`.

ncu runs the SMs near 2.0 GHz at base clocks while the block runs near 2.7 GHz on this card, so a compute-bound launch
reads about 1.3–1.4× its in-block time.
