# noepoch_counters.sh

GPU counters for one question: why the no-epoch STARK base (block 25368371 as one proof, PR #1013)
takes about 31 s when #1009's epoch base takes about 26 s on the same card. The script profiles
both on one binary and sends back text only. It runs on a Linux machine with an RTX 5090 whose
performance counters are open (the 09-25 and 09-28 runs used the same machine).

## Before you start

- Close big programs. The no-epoch run peaks at 43.9 GiB of host RAM, and the checks refuse to
  start with less than 47 GiB available.
- Make the NVTX library visible, as on 09-28, so the prover's phases show up as named ranges:
  `export LAMBDA_VM_NVTX_LIB=~/nvtx/libnvToolsExt.so.1`
- Run it inside tmux or screen, with the GPU otherwise idle.

## Run

```bash
cd ~/mauro
curl -fsSLO https://raw.githubusercontent.com/yetanotherco/lambda_vm/profile/noepoch-counters/scripts/profile/noepoch_counters.sh
bash noepoch_counters.sh --preflight-only   # the checks alone, about 2 minutes
bash noepoch_counters.sh                    # everything
```

The script works only inside `./lambda-vm-noepoch-prof` (to use another directory, set
`NP_WORKDIR=/path`). It clones the public repository at a pinned commit, downloads the block input
and the guest ELF (both checked by sha256), builds the prover, and runs everything with an explicit
environment (`env -i`).

- **Runtime:** about 60 to 90 minutes. The build takes 5 to 30 minutes, the reference runs 2 to 3,
  run A (Nsight Systems) 8 to 12, and run B (Nsight Compute, 30 passes) 40 to 50.
- **Short version:** `NP_RUN_B=0 bash noepoch_counters.sh` runs the reference and run A only, in
  about 15 minutes plus the build.
- **Disk:** about 25 to 35 GiB, and the checks require 40 GiB free. After you send the bundle you
  can delete `runs/<UTC>/keep/`.

## What to send back

At the end the script prints one path:
`lambda-vm-noepoch-prof/runs/<UTC>/noepoch-counters-<UTC>.tar.gz`.

That file is a few MB of text, with paths, hostname and IP addresses replaced, and it has passed a
self-check. If you would rather send less, `send/SUMMARY.md` inside it answers the question on its
own. Do not send anything from `keep/`: the Nsight reports there store this machine's environment.

## What each part answers

| output | question |
|---|---|
| `reference/runs.tsv` | the wall of each workload on this machine without a profiler |
| `runa/<w>/runa-<w>.md` | (a) per stage (head, prepass, main commit, between, fused, and the recommit inside fused): card busy %, SM active and issue %, warps in flight, DRAM and PCIe %, VRAM, how many fused tasks were open, the top kernels; the fused stage second by second |
| `runa/<w>/runa-<w>-nvtx-kernels.tsv` | the kernels each prover phase launched; `r1_main_recommit_table` is the no-epoch recommit |
| `runa/<w>/api-<w>-by-*.tsv` | (c) CUDA API seconds in alloc, free, pinned host memory, synchronize and copies, per thread and per stage |
| `summary/kernels.md` | (b) per kernel and workload: SM, DRAM and L2 % of peak, occupancy, and whether it is compute-, memory- or latency-bound, at every instance size each workload ran |

## If something goes wrong

- **The counters are locked (ERR_NVGPUCTRPERM):** the preflight prints the fix. Either run as root,
  or set `NVreg_RestrictProfilingToAdminUsers=0` and reboot.
- **A no-epoch pass runs out of memory:** re-run only the failed passes with a smaller card budget,
  for example `NP_NCU_VRAM_MB=12000 NP_RUN_A=0 NP_REFERENCE=0 NP_PASSES="noepoch_merkle" bash noepoch_counters.sh`.
  `--print-plan` lists the pass names.
- **The self-check refuses the bundle:** it names file:line only. Look at those lines, then run
  `bash noepoch_counters.sh --bundle-only <run dir>`.
- **To see every command without running anything:** `bash noepoch_counters.sh --print-commands`.
- **Options and knobs:** `bash noepoch_counters.sh --help`.
