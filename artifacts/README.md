# CUDA checkpoint artifacts

Six checkpoints on top of `origin/main` (base commit `0b0e6d38`).
Each ships as a git bundle + per-commit mbox patches.

```
checkpoint-barycentric/                        # 11 commits, last: 763d3776
  cuda-barycentric.bundle                      # — barycentric OOD infra only
  …

checkpoint-r2-commit-tree/                     # 17 commits, last: 04988c41
  cuda-r2-commit-tree.bundle                   # — through R2 commit fuse,
  …                                            # FRI-tree kernel (unwired).
                                               # fib_1M ~13.0s (1.40× CPU).

checkpoint-experimental-lde-resident/          # 19 commits, last: 3ac687e0
  cuda-experimental-lde-resident.bundle        # — adds GPU-resident LDE
  …                                            # handles + GPU R4 deep.
                                               # fib_1M ~12.66s (1.44× CPU),
                                               # fib_4M ~29.75s. Experimental.

checkpoint-exp-2-zisk-tricks/                  # 27 commits, last: 7082c0f2
  cuda-exp-2-zisk-tricks.bundle                # — GPU R3 OOD on handles,
  …                                            # skip CPU slab extraction,
                                               # GPU FRI commit fully on
                                               # device. fib_1M ~11.64 s
                                               # (1.57× CPU), fib_4M ~28.3s.

checkpoint-exp-3-tier2/                        # 28 commits, last: 2ba3af77
  cuda-exp-3-tier2.bundle                      # — adds comp-parts on device
  …                                            # (handle threaded through R2
                                               # → R4). Architecturally clean
                                               # but neutral within noise on
                                               # fib_1M because `num_parts==2`
                                               # branch dominates. Still
                                               # ~1.57× CPU.

checkpoint-exp-4-tier3/                        # 29 commits, last: ad78a93a
  cuda-exp-4-tier3.bundle                      # — tier-3 investigation
  …                                            # doc. No code changes land:
                                               # per-item analysis showed
                                               # each candidate (stream
                                               # overlap, warp bary reduce,
                                               # GPU batch inverse) either
                                               # falls below run-to-run
                                               # variance or requires
                                               # parallel-scan scope. Perf
                                               # unchanged from tier 2.

cuda-checkpoints-all.tar.gz                    # all six in ~2.0 MB archive
```

## Applying — bundle (preferred)

```bash
# In a clone of yetanotherco/lambda_vm:
git fetch cuda-exp-3-tier2.bundle \
    cuda/exp-3-tier2:cuda/exp-3-tier2
git checkout cuda/exp-3-tier2   # most recent code changes

# Build & verify
make test-cuda                 # math-cuda parity tests
cargo test -p stark -F cuda    # 121 stark tests
cargo test -p lambda-vm-prover --release --features cuda,instruments \
    --test bench_gpu bench_prove_fib_1m_long -- --ignored --nocapture
```

## Verify integrity

    54381ef4c4f6acbfe1dc37aa0b6138cac5e1befc4530e445eac1e876fed1b628  cuda-barycentric.bundle
    cb04120f861747825d99bc624be78ff4d8d43a2f48ba069d77b0e27280e32af9  cuda-r2-commit-tree.bundle
    cd087c4ad203be92201392acf877643df25379dddccce27a970e10e921669012  cuda-experimental-lde-resident.bundle
    07f5b4b684dbdfb38a97c4e0c7d4536d0605da265eb855b15b174b100df829ee  cuda-exp-2-zisk-tricks.bundle
    e765794d1a3310e2716b5e88d307a464c5c9c36f103aef83e29137175911d2b0  cuda-exp-3-tier2.bundle
    16a2e70fd3440edc97c3dccca4f3dcc498762ae8c307d6672a3e2a186220e75e  cuda-exp-4-tier3.bundle
    306f31ad77eeff9ae0a6c9559cac553e517cc1286794b5ed8f5ce3b15ee48a22  cuda-checkpoints-all.tar.gz

Each bundle base is `0b0e6d38` (commit `Bench vs other vms (#365)`
on main).

## Branch lineage

    (main)
      │
      └─ cuda/batched-ntt … 04988c41 (r2-commit-tree checkpoint)
                    │
                    └─ cuda/experimental-lde-resident … 3ac687e0
                                │
                                └─ cuda/exp-2-zisk-tricks … 7082c0f2
                                            │
                                            └─ cuda/exp-3-tier2 … 2ba3af77
                                                        │
                                                        └─ cuda/exp-4-tier3 … ad78a93a

`cuda/exp-3-tier2` has the most recent perf-positive code. `exp-4-tier3`
adds only an analysis doc. Both are experimental (not yet merged back
into the shipping line).
