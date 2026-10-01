# noepoch_counters.sh

GPU counters on the two no-epoch provers of block 25368371, for a Linux machine with an RTX 5090
whose performance counters are open. The script answers three questions and sends back text only:

1. **No-epoch WHIR base** (PR #1014, about 18 s of a 22.3 s block). Per phase (phase A: the streamed
   build and commits; phase B: the argue and the openings): kernel time by family, card busy against
   idle, and whether the top kernels sit at a hardware roof.
2. **No-epoch STARK recursion** (PR #1013, about 10.3 s of a 31.8 s block). For each LFM proof's device
   phase: how busy the card is inside it, and whether its kernels sit at a roof.
3. **The host's CPU** during the same runs, per stage and thread role.

## Before you start

- Close big programs. The WHIR run peaks near 45 GiB of host RAM; the script checks for 48 GiB
  available before every run, and a watchdog ends a run if available memory falls under 2 GiB.
- Run it inside tmux or screen, with the GPU otherwise idle.
- Optional: `export LAMBDA_VM_NVTX_LIB=~/nvtx/libnvToolsExt.so.1` if you have that library. Otherwise
  the script downloads one (the nvidia-nvtx-cu12 12.8.90 wheel) and checks it by sha256.

## Run

```bash
cd ~/mauro
curl -fsSLO https://raw.githubusercontent.com/yetanotherco/lambda_vm/<sha>/scripts/profile/noepoch_counters.sh
bash noepoch_counters.sh --preflight-only   # the checks alone, about 2 minutes
bash noepoch_counters.sh --quick            # about 15 minutes after the build (about 20 the first time)
bash noepoch_counters.sh --full             # about 35-55 minutes including the build
```

The script works only inside `./lambda-vm-prof2` (set `NP_WORKDIR=/path` to use another directory).
It clones the public repository at two pinned commits, downloads the block input and the guest ELF
(both checked by sha256), builds both provers, and runs everything with an explicit environment
(`env -i`). The build takes 4-5 minutes the first time and is reused by a re-run.

| mode | runtime | peak host RAM | answers |
|---|---|---|---|
| `--quick` | about 15 min (+5 for a first build) | about 48 GiB | Q1 per-phase busy/idle, kernel families and GPU-metric means; Q2 per-hold busy and families; Q3; the roofs of the top kernels (WHIR phase-A coset leaves, phase-B GKR and lean rounds; LFM leaves and Merkle levels) |
| `--full` | about 35-55 min including the build | about 48 GiB | all of quick, plus the unprofiled walls and the roof of every kernel that carries a phase or a hold |

Nothing new starts after a deadline (30 minutes for quick, 58 for full), so a slow machine skips the
last passes instead of running long.

## What to send back

At the end the script prints the path of one file: `lambda-vm-prof2/send-back.tar.gz` (run from
`~/mauro`, that is `~/mauro/lambda-vm-prof2/send-back.tar.gz`). It is a few MB of text, with paths,
hostname and IP addresses replaced, and it has passed a self-check. Do not send anything from
`runs/<UTC>/keep/`: the Nsight reports there store this machine's environment.

## If something goes wrong

- **The counters are locked (ERR_NVGPUCTRPERM):** the preflight prints the fix. Either run as root,
  or set `NVreg_RestrictProfilingToAdminUsers=0` and reboot.
- **NVTX filtering fails on the probe:** the preflight warns. Run A still runs; the Nsight Compute
  passes, which each work inside one NVTX window, are skipped.
- **The self-check refuses the bundle:** it names file:line only. Look at those lines, then run
  `bash noepoch_counters.sh --bundle-only <run dir>`.
- **To see every command without running anything:** `bash noepoch_counters.sh --print-commands`.
- **Options and knobs:** `bash noepoch_counters.sh --help`.
