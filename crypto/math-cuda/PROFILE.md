# nsys profile of fib_iterative_1M (2 proves: 1 warmup + 1 measured)

## TL;DR

The GPU is **not** the bottleneck. Out of ~12 s wall-clock per proof,
only ~2.6 s is *any* CUDA activity (kernels + memcpy combined). The
remaining ~9.4 s is CPU work that we can't meaningfully shrink
without porting program logic (trace build, aux trace build,
constraint eval, query-phase openings).

Tile-based NTT layout — the optimisation that was on the tier-2/3
shortlist — would land at most ~100 ms wall because the NTT is only
243 ms of GPU time and much of that already overlaps with CPU /
other-table compute.

## CUDA activity breakdown (2 proves worth)

| Operation                              | Time (ms) | % CUDA | Invocations | Total MB |
|----------------------------------------|-----------|--------|-------------|----------|
| `[CUDA memcpy Device-to-Host]`         |   1275.1  | 49.9 % |         690 |   16336  |
| `[CUDA memcpy Host-to-Device]`         |    638.7  | 25.0 % |        1674 |   10311  |
| `ntt_dit_level_batched`                |    243.1  |  9.5 % |        1176 |       — |
| `barycentric_ext3_batched_strided`     |     74.4  |  2.9 % |          28 |       — |
| `keccak_merkle_level`                  |     65.5  |  2.6 % |        3312 |       — |
| `bit_reverse_permute_batched`          |     56.1  |  2.2 % |          98 |       — |
| `keccak256_leaves_ext3_batched`        |     53.0  |  2.1 % |          14 |       — |
| `keccak256_leaves_base_batched`        |     35.1  |  1.4 % |          12 |       — |
| `barycentric_base_batched_strided`     |     33.8  |  1.3 % |          24 |       — |
| `ntt_dit_8_levels_batched`             |     25.0  |  1.0 % |          98 |       — |
| `keccak_comp_poly_leaves_ext3`         |     20.7  |  0.8 % |          14 |       — |
| `deep_composition_ext3_row`            |     12.3  |  0.5 % |          12 |       — |
| `keccak_fri_leaves_ext3`               |      8.0  |  0.3 % |         258 |       — |
| `[CUDA memset]`                        |      6.9  |  0.3 % |         134 |       — |
| `pointwise_mul_batched`                |      6.7  |  0.3 % |          56 |       — |
| `fri_fold_ext3`                        |      1.0  |    —   |         272 |       — |
| `fri_update_twiddles`                  |      0.3  |    —   |         258 |       — |
| **TOTAL CUDA**                         | **2555.6**|        |             |          |
|   — of which kernel compute            |    634.9  | 24.8 % |             |          |
|   — of which memcpy / memset           |   1920.7  | 75.2 % |             |          |

## What this tells us

1. **Kernel compute total is 635 ms across 2 proves** (so ~320 ms per
   proof). The GPU is not under-utilised — this is what it takes to
   do the actual field arithmetic + hashing.

2. **Memcpy totals ~1.9 s across 2 proves** (~950 ms per proof). Most
   of this is overlapped with compute on parallel streams. The
   memcpy wall-time contribution is only partially additive.

3. **16.3 GB of D2H** per 2 proves = ~8 GB per proof. Largest single
   D2H is 856 MB (pinned-staging flush for the biggest table).

4. **1176 invocations of `ntt_dit_level_batched`** — the per-level
   non-fused kernel used for levels outside the shared-memory fusion
   window. 207 μs average. The 8-level fused kernel fires 98 times.

5. **Memcpy is 3× the kernel time.** Most of it is D2H of the LDE
   back to host (for query-phase openings that happen on CPU).

## Where the 12 s wall time actually goes

The instrument dump earlier in the session gave us:

- Trace build (CPU, program-specific):    **~2.4 s wall**
- Aux trace build (CPU, per-AIR):         **~2.4 s wall**
- Round 1 LDE + Merkle (GPU-bound):       ~1.5 s wall
- Rounds 2–4 (mostly GPU, some CPU):      ~4.8 s wall
- Misc CPU prelude / setup / finalize:    ~0.9 s wall

The ~2.6 s of CUDA activity from this profile sits *inside* Rounds
1 + 2–4 — mostly overlapped with CPU work.

## Implications for the remaining optimisation list

### Tile-based NTT layout (previously the candidate for tier 3)

**Reject.** Even a perfect 2× speedup on every NTT kernel would save
(243 + 25 + 56) / 2 = 162 ms of GPU kernel time. Most of that is
hidden behind memcpy / CPU work, so the wall-time saving is well
under 100 ms. A 1700 LoC NTT rewrite for <1 % wall is the wrong
call.

### GPU Montgomery batch inverse (Blelloch scan)

**Still viable** at ~50–100 ms wall savings, but confirmed marginal.
Only worth doing if done opportunistically (e.g. as part of a larger
Round 3/4 CPU-prelude port).

### Reducing D2H traffic

**Real lever.** 16.3 GB D2H per 2 proves includes data that the CPU
path needs for query-phase openings. But some D2H is redundant:
- LDE D2H for tables/rounds where the device handle was already used
- Full tree D2H when queries only touch log(N) path nodes

Quantifying this needs per-call tracing; skipped for this session.

### Constraint eval interpreter (item 5a)

**Biggest lever remaining.** CPU constraint eval is ~0.5–0.8 s wall.
Moving to GPU needs a per-AIR AST → bytecode serializer + a device
interpreter (pil2-proofman's pattern, ~800+ LoC). Touches constraint
code, which is the reason we flagged the memory rule.

### Aux trace build / trace build on GPU

**Biggest two levers overall** (~4.8 s wall combined) but these are
per-AIR / per-VM-executor logic. Multi-day porting work, plus the
risk of diverging from the CPU reference (which remains the
verifier-authoritative path).

## Conclusion

The profile confirms what the aggregate instruments measurements
already suggested but more precisely:

> **GPU-side kernel compute is ~320 ms per proof. Any further
> optimisation confined to the GPU side has a hard ceiling there.**

The remaining ~9+ seconds of wall time is on the CPU (trace build,
aux trace build, constraint eval, query phase openings). Pushing
past 1.6× on fib_1M requires porting one of those, not further GPU
tuning.
