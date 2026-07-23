# ==========================================================================
# 🟢🟢 RESUME HERE — SESSION 7 (fresh session, saved 2026-07-23). READ THIS BLOCK FIRST.
# ==========================================================================
#
# WHAT THIS IS: moving the zkVM TRACE-GENERATION onto GPU (prover-only). Branch `tracegen-gpu-full`,
# base `fb204215`, working-tree overlay, NOTHING COMMITTED. Local: /Users/joaquin/work/lambda/lambda_vm_3.
# RULES: never commit/push unless told; never cache build steps; ALWAYS test on ethrex_5tx (not fibonacci);
# measure speed with the WARM bench / cli --time TIMELINE, never a single cold cross-process run.
#
# ── STATE: TREE IS GREEN, e2e prove+verify PASSES under LAMBDA_VM_GPU_FULL=1. ──────────────────────
# The whole data-parallel trace-gen runs on GPU and verifies. This session (6) added, all e2e-verified:
#   • Step A: device `from_log` cpu_ops SoA (build_shared_devops from logs).  • Step B: skip dead host
#     op-vecs under gpu_full (−145ms).  • Step C: ECSM/ECDAS carry witness → GPU (`ecdas_carries` kernel,
#     −192ms in p2a).  • c1/x254: device register final-state, WIRED (device_register_final_state).
#   • fix(a): "walk-once" — register walk's IS_HALF histogram (device_is_half) reused so the histogram
#     skips a 2nd register walk. Correct + e2e-verified but MEASURED WALL-NEUTRAL (see analysis). APPLIED.
#   • CLEANUP done: deleted dead c2b (reg_value_propagate kernel/launcher/test); removed ALL temp probes
#     ([p3/p4/pc/p1-probe], c2-dbg); deleted 6 diagnostic `diag_*` tests from gpu_reg_emit_parity.rs +
#     fixed imports. No probes/dead code remain. trace_build ~1.6s (was 1.895s baseline this box).
#
# ── DEEP PERF ANALYSIS (this session) — the answer to "where's the slow part / what's left": ───────
# READ the "🔬 DEEP PERF ANALYSIS" block below for the full measured detail. TL;DR:
#   • trace_build ~1.6s = p3to5 ~1.15s (75%). p3to5 = region A (walks+reconstruction) ~590ms DOMINANT +
#     B (histogram) ~327ms + C (p5 chip-gens) ~267ms PARALLEL-HIDDEN (don't touch C).
#   • ⭐ The GPU kernels are FAST (register walk = ~31ms on GPU). The cost is HOST SoA MARSHALLING —
#     extracting packed/rv1/rv2/rvd/next_pc Vecs from the 4M cpu_ops (~56ms), done ~5× across the walks +
#     shared_devops + register_final_state. trace-gen is HOST-BOUND, NOT GPU-bound.
#   • THREE "obvious" fixes were wall-neutral (walk-once, scatter-rewrite, sort). Optimizing GPU kernels
#     does NOT move the wall. p1 (138ms) is irreducible (bandwidth-bound; rayon was slower).
#   • ⇒ trace-gen is NEAR ITS SPEED FLOOR. The ONE remaining lever = the FULLY-RESIDENT refactor:
#     build the device SoA ONCE (shared_devops, already built from logs) and REUSE its resident buffers
#     for the register + memory walks (+ register_final_state) instead of re-extracting from cpu_ops on
#     host each time. Multi-step (reorder shared_devops before the walks + device-buffer walk variants),
#     soundness-critical, and UNCERTAIN payoff (async GPU/host overlap confounds — MUST confirm with the
#     warm gpu_resident_bench, not the TIMELINE). Est. ceiling ~150-250ms if all host extractions removed.
#
# ── OPEN DECISIONS (ask the user): ─────────────────────────────────────────────────────────────────
#   1. fix(a) walk-once is APPLIED but wall-neutral — KEEP (correct, removes redundant work) or REVERT
#      (cleaner)? User leaned unsure.
#   2. Attempt the fully-resident refactor (the only speed lever, uncertain payoff)? Or declare trace-gen
#      done? Recommendation: trace-gen is essentially optimized; only do the refactor if the user insists,
#      and gate it with the WARM bench (3 prior fixes were wall-neutral — don't guess).
#   NOTE: whole-prove is NTT/FFT-bound (NTT=33% of GPU, trace-gen fills=0.5%); user cares ONLY about
#   trace-gen, so NTT is OUT OF SCOPE — do not propose it as work.
#
# ── BOX + BUILD/TEST ───────────────────────────────────────────────────────────────────────────
#   ⚠️ THE OLD BOX (`-p 63154 root@174.31.67.221`) DIED 2026-07-23 (connection refused). Don't waste
#   time on it. PROVISION A NEW RTX 5090 (CUDA ≥13) and re-sync from LOCAL (the source of truth — local
#   is complete + was e2e-verified green before the box died) per the "BOX + re-sync" section below,
#   then write /tmp/runenv.sh. Old box invocation (template for the new one's opts):
#   ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o IdentityAgent=none -o IdentitiesOnly=yes
#     -o ConnectTimeout=30 -o ServerAliveInterval=15 -o ServerAliveCountMax=8 -i ~/.ssh/id_ed25519
#     -p <PORT> root@<IP>   ·   repo /workspace/lambda_vm   ·   `source /tmp/runenv.sh` (sets
#     CUDA_HOME, PATH, LAMBDA_VM_BENCH_ELF=.../rust/ethrex.elf, LAMBDA_VM_BENCH_INPUT=.../ethrex_5tx.bin).
#   e2e (correctness gate): LAMBDA_VM_GPU_FULL=1 cargo test -p lambda-vm-prover --release --features
#     cuda,instruments --lib gpu_resident_e2e -- --ignored --nocapture   → "OK: full prove+verify passed".
#   TIMELINE (speed): cargo build --release -p cli --features "instruments,prover/cuda"; then
#     LAMBDA_VM_GPU_FULL=1 ./target/release/cli prove $LAMBDA_VM_BENCH_ELF --output /tmp/p.bin
#     --private-input $LAMBDA_VM_BENCH_INPUT --time  (grep trace_build / p3to5_build_traces).
#   SYNC: after editing, rsync each file (`rsync -aR -e "ssh <opts> -p 63154" <files> root@IP:/workspace/
#     lambda_vm/`) + md5-verify (`md5 -q` local vs `md5sum` remote). zsh: put ssh -o opts INLINE (var
#     word-splitting fails). Builds are FAST (~6-40s, deps cached). Background long runs; poll with Monitor.
#   CAVEAT: eprintln timers around GPU work are UNRELIABLE (async overlap) — for real kernel timing add
#     stream.synchronize() between phases (temp) or use the warm bench. The TIMELINE trace_build wall is
#     noisy ±80ms run-to-run; take 3 runs.
#
# ── CHANGED CODE FILES (this whole effort; for re-sync if the box died — all must be md5-synced): ──
#   crypto/ecsm/src/{witness.rs, lib.rs}; crypto/math-cuda/{Cargo.toml, build.rs, kernels/{bitwise_hist.cu,
#   keccak.cu, trace_cpu.cu, trace_ops.cu, trace_walk.cu}, src/{bitwise_hist.rs, device.rs, lde.rs, lib.rs,
#   trace_cpu.rs, trace_ops.rs, trace_walk.rs, precompile.rs}}; crypto/stark/src/{gpu_lde.rs, prover.rs};
#   executor/src/vm/{execution.rs, instruction/execution.rs, memory.rs}; prover/src/{lib.rs, paged_mem.rs,
#   tables/{bitwise.rs, gpu_trace.rs, page.rs, trace_builder.rs, types.rs}, tests/*} + fixtures
#   executor/tests/ethrex_5tx.bin + executor/program_artifacts/rust/ethrex.elf (gitignored → rsync explicit).
#   Session-6 files specifically: crypto/ecsm/src/{witness.rs,lib.rs}, crypto/math-cuda/{kernels/{trace_cpu.cu,
#   trace_walk.cu}, src/{precompile.rs,device.rs,trace_walk.rs,bitwise_hist.rs}}, prover/src/tables/
#   {trace_builder.rs,gpu_trace.rs}, prover/src/tests/{gpu_ecdas_carries_parity.rs,gpu_reg_final_parity.rs,
#   gpu_reg_emit_parity.rs,mod.rs}.
# ==========================================================================


# GPU trace-gen — full session handoff (hand this to a fresh session for complete context)

Branch: `tracegen-gpu`. Goal (user mandate): move the **entire zkVM trace-generation pipeline onto
the GPU**, prover-only, validated on the ethrex 5-tx block. Completeness first ("everything on GPU"),
speed second. NEVER commit/push unless told. NEVER cache build steps. ALWAYS test on ethrex_5tx
(never fibonacci). This doc is self-contained: read it + the two linked docs and you have full state.

Companion docs: `reports/tracegen/RESIDENT-PIPELINE-PLAN.md` (the P1–P6 plan) and auto-memory
`gpu-tracegen-v2-progress.md` (running log). This HANDOFF supersedes/summarizes both.

---

# ========================================================================
# 🔬 DEEP PERF ANALYSIS (2026-07-22, SESSION 6) — where trace-gen time REALLY goes, measured
# ========================================================================
# User asked for a rigorous "is it optimal / how to make it faster" analysis (trace-gen ONLY). Ran 3
# read-only analysis agents (p3to5 efficiency, kernel efficiency, whole-prove ceiling) + dynamic
# profiling on the box. DECISIVE, MEASURED findings (ethrex_5tx, RTX 5090, GPU_FULL):
#
# 1) trace_build ~1.6s breakdown: p0 45 + p1 138 + p2a 100 + p2b 34 + **p3to5 ~1.15s (the 75%)**.
#    p3to5 sub-split (measured): **A=region "walks+reconstruction" ~590ms (DOMINANT, serial) ; B=histogram
#    +base ~327ms ; C=p5 chip-table gens ~267ms (PARALLEL-HIDDEN — moving it saves ~0 wall).**
# 2) Region A sub-split (measured): **register walk (build_device_memw_register) ~outer 213-283ms ;
#    memory walk (build_device_memw_ls) ~247ms** (img_sort+walk 147, unpack 30, recon 30-48, page 29,
#    memw_lt 11). NOTE the OLD handoff "478ms build_device_memw_ls" does NOT hold at current scale.
# 3) ⭐ THE KEY FINDING (register-walk internal profile, synced): **the GPU walk kernels are TINY** —
#    uploads 16ms + emit 2.5 + route_core(sort+route+scan+is_half) 9 + fill 3.5 = **~31ms GPU**. The
#    HOST SoA extraction (build packed/rv1/rv2/rvd/next_pc Vecs from the 4M cpu_ops) = **~56ms**. The
#    big outer number is inflated by async GPU/host overlap (eprintln can't cleanly time async GPU —
#    use the warm bench). ⇒ **trace-gen's cost is HOST-SIDE MARSHALLING (SoA extraction from cpu_ops,
#    done ~5× — register walk, memory walk, shared_devops, register_final_state, hist), NOT the GPU
#    kernels.** Same nature as p1 (bandwidth-bound struct access).
# 4) ⚠️ THREE "obvious" fixes were / would-have-been WALL-NEUTRAL (all measured or reasoned): (a) walk-once
#    dedup of the register walk [IMPLEMENTED this session, e2e-verified, MEASURED wall-neutral — the
#    duplicated walk wasn't critical-path]; (b) rewriting the counting-sort scatter [would save ~9ms of
#    31 — a trap avoided by profiling]; (c) the sort in general is not the bottleneck. LESSON: the GPU is
#    NOT the bottleneck in trace-gen; optimizing kernels doesn't move wall. ALWAYS measure (warm bench).
# 5) THE ONLY REAL LEVER LEFT (for trace-gen speed): the FULLY-RESIDENT refactor — build the device SoA
#    ONCE (shared_devops, already built from logs in Step A) and REUSE its resident packed/rv1/rv2/rvd
#    buffers for the register + memory walks (+ register_final_state) instead of re-extracting from
#    cpu_ops on host each time. Removes the redundant ~56ms×N host extraction. Multi-step (reorder
#    shared_devops before the walks + device-buffer variants of the walk launchers), soundness-critical,
#    and payoff is UNCERTAIN (async overlap confounds; must confirm with the warm gpu_resident_bench).
# 6) STRATEGIC (honoring "trace-gen only" but for the record): trace_build is ~14% of the ~11s prove;
#    the prover is NTT/FFT-bound (NTT 33% of GPU time, composition 23%, barycentric 15% ≈ 70% of GPU;
#    trace-gen fills = 0.5% of GPU). The "Total FFT 199.7%" line is a SUM of parallel per-worker timers,
#    NOT wall — ignore it. Biggest whole-prove lever = the NTT launch path (1480 per-level launches; a
#    fused `ntt_dit_8_levels_batched` exists but is ~0.8% used). OUT OF SCOPE per the user.
# CONCLUSION: trace-gen is near its speed floor. The GPU kernels are fast; the residual is host marshalling
# + the irreducible p1. The one lever (fully-resident SoA reuse) is multi-step with uncertain payoff.
# STATE: fix(a) walk-once applied + e2e-verified (wall-neutral); all profiling timers removed; tree GREEN.

# ========================================================================
# 🟢 RESUME HERE (handoff saved 2026-07-21 END OF SESSION 3) — read THIS block first, then the dated sections
# ========================================================================

**Working tree** (source of truth, NOTHING COMMITTED): `/Users/joaquin/work/lambda/lambda_vm_3`, branch
`tracegen-gpu-full`, base commit `fb204215`. **Mandate:** the WHOLE trace-gen on GPU (completeness first; the
user decides scope — do NOT editorialize on worth/speed). **NEVER commit unless told. NEVER cache build steps.
ALWAYS test on ethrex_5tx.**

## 🔵 SESSION 6 cont. — CAMINO 2 (drop host p1/from_log): a 4-piece device-resident arc, IN PROGRESS
User chose to grind the full fully-resident endgame (drop the host `cpu_ops`/`from_log`, 138ms). Traced:
host `cpu_ops` is consumed pervasively (build_shared_devops decode, device register walk, register
advance, ecall assembly, mul/dvrm/cpu32 derivations, gen_cpus, count guards) → dropping it = making ALL
of trace-gen device-resident, incl. the soundness-critical register-advance + ecall on device. NO partial
win until the final big-bang (C2-bd). Staged (each parity/e2e-gated), tasks #4–#7:
- **✅ C2-c1 DONE + parity-verified**: device REGISTER final-state snapshot. Kernels `reg_final_seed/maxts/
  gather` (trace_walk.cu) + `gpu_register_final_snapshot` (trace_walk.rs): seed init → atomicMax(ts) →
  gather value, over the device register access stream. Parity `gpu_reg_final_parity` bit-exact over 66
  word-states (regs 0-31 + PC) vs host `to_final_state_map` (now pub(crate)). NOT yet wired into
  build_traces (wiring is part of c2/bd). **x254 (commit index) EXCLUDED** — tracked separately
  (`index_register` via write_index), derived on device in c2.
- **✅ C2-c2 DE-RISK DONE (all device-derivation kernels parity-verified, NOT yet wired)**:
  - (a) **x254 commit index** — `reg_x254_scan` (atomicAdd commit_count + atomicMax ts over ecall-commit
    ops) folded into `gpu_register_final_snapshot`; `gpu_reg_final_parity` bit-exact over 67 word-states
    (regs+pc+x254). ⇒ **register FINAL STATE fully device-derivable.**
  - (b) **ecsm-read value propagation** — [DERIVED + parity-verified this session, then PRUNED as dead code
    on 2026-07-22 cleanup — it was never wired (it was infra for the abandoned backbone rewrite), so keeping
    it was dead unwired code]. The approach if the backbone rewrite is ever done: a per-bucket sequential
    scan over the stable-sorted register access stream (unknown ecsm read takes `last`, known access updates
    `last` — device analog of `register_state.read`); it was parity-verified to resolve 60 ECSM reads
    bit-exact over 13.3M accesses. Re-derive from this note (~20-line kernel + launcher) when needed.
    commit/keccak write KNOWN op values → not unknown; only ecsm reads need propagation.
  - ⇒ **The ENTIRE register/ecall device-derivation (final state + x254 + ecall read values) is built and
    parity-verified as standalone kernels.** What remains is C2-bd WIRING.
- **🔶 C2-bd IN PROGRESS (the big integration — lands the 138ms)**:
  - ✅ **register_final_state WIRED to device + e2e-VERIFIED**: `device_register_final_state` (gpu_trace.rs)
    assembles the REGISTER final-state map from the device snapshot (c1 regs+pc + x254 scan), used under
    gpu_full in build_traces instead of the host `to_final_state_map`. Two finalization subtleties the
    standalone parity missed (caught by a temp device-vs-host diff, now removed): (1) HALT writes x1-x31=0
    at ts=u64::MAX (`is_final`), (2) the PC token is finalized to `(1, halt_ts+4*num_padding+1)`. With both
    mirrored, **device-vs-host map diff=0 over 67 entries, e2e prove+verify PASSES.** ⇒ the device register
    final-state is now PROVEN bus-correct in a real proof (not just standalone). NOTE: this is a correctness/
    integration milestone — the host register advance STILL RUNS (feeds the ecall handlers), so NO 138ms yet.
  - ⏳ REMAINING to actually drop `from_log`/advance (COUPLING MAP, from reading push_ecall_memw_ops + the 3
    handlers — this is why it's a fresh-session refactor, not a turn-end rush; it's precompile-witness
    soundness-critical):
    * **c2b must integrate INTO the device walk, not run as a host-array pass.** Feeding handlers off
      register_state needs the ecsm read VALUES (addresses). Building the 13M-access stream on host + uploading
      = reintroduces the serial marshalling that was the original +40% slowdown. The walk (build_device_memw_register)
      ALREADY emits the stream on device + computes old_value per access; the ecsm reads' old_value IS the
      address. So add the value-propagation (reg_value_propagate, built + parity-verified) INTO the resident
      walk and read back the ~60 ecsm addresses — avoids marshalling.
    * **Hidden host-kept artifacts** (push_ecall_memw_ops): the x254 width-1 commit-index MEMW row is NOT
      discarded (only width-2 register + non-register memory rows are) → it stays host-side and needs
      old_index/old_ts (derivable: track current_commit_index + prev-commit ts in the loop). The captured
      width-2 ecall reg accesses need values (commit/keccak: op fields; ecsm: c2b addresses). ecsm memory rows
      are device-owned (device_memory_drop) but the captured mem accesses need base=addr (ecsm addresses).
    * Then: skip register_state reads/writes/asserts (x10==1 debug_assert) + the 4M advance under gpu_full;
      build a `cpu_ops` RESIDUAL (only word/mul/divrem/ecall ops — for the p2b derivations + handlers, ~300-600k
      ops ≈ ~15-30ms from_log) instead of the full 4M; point gen_cpus/build_device_memw_register/count-guards/
      num_padding_rows/halt_op-lookup at device/residual. ⇒ drops the 138ms.
    * GATE: e2e on ethrex_5tx is necessary but NOT sufficient (exercises commit+keccak+ecsm but one config);
      add parity for each device-derived ecall artifact (like c1/x254/c2b) BEFORE trusting it in the proof.
  - ⚠️ **MEASURED (2026-07-22) — the p1 "138ms" is largely IRREDUCIBLE, which caps the payoff of dropping it:**
    a [p1-probe] split of `collect_cpu_ops` = **~62% host decode-lookup** (hashmap get + `from_instruction`)
    + **~38% `from_log` arithmetic**, and the whole thing is **memory-bandwidth-bound** (building the ~480MB
    `Vec<CpuOperation>`). Rayon-parallelizing it measured **155ms (SLOWER than 138ms serial)** — coordination
    over a bandwidth-saturated build — so it was reverted. ⇒ A GPU `from_log` (the backbone rewrite above)
    can only remove the ~38% arithmetic (~50ms) and must still materialize/download for the host consumers,
    so realistic net ≈ small and likely erased by marshalling. **Recommendation: the p1 drop is LOW-ROI;
    the cluster is already −19% (1.54s). Only pursue the backbone rewrite for the MANDATE ("all on GPU"),
    not for speed.** Both probes removed; tree green at 1.54s.
- **⏳ C2-a**: MUL/DVRM MSB16 chunk-dedup tail on device (self-contained, still pending).
- **⏳ C2-a**: MUL/DVRM MSB16 chunk-dedup tail on device (per-chunk dedup bit-exact — fiddly).
- **⏳ C2-bd (big-bang)**: device decode SoA (packed/imm/pc/next_pc from logs+instructions, not host
  cpu_ops) + rewire register-advance/derivations/gen_cpus/count-guards to device + skip `collect_cpu_ops`.
  Only here do the 138ms land. e2e-gated.
Tree GREEN at A/B/C (1.54s); c1 is additive (kernel + parity test), NOT in the hot path yet, e2e unaffected.
Nothing committed; all local↔box md5-synced.

## ✅ SESSION 6 (2026-07-22) — the DECODE+COLLECT cluster → GPU (Steps A/B/C, all e2e-VERIFIED)
Box `-p 63154 root@174.31.67.221` (RTX 5090, CUDA 13.1). Baseline this box (GPU_FULL TIMELINE): trace_build
**1.895s**; cluster p0_decode 45 + p1_cpu_ops 138 + p2a_collect 308 + p2b_collect 102 = **596ms serial host**.
- **A (device cpu_ops seam) — done, SPEED-NEUTRAL (mandate win).** `build_shared_devops` (gpu_trace.rs) now
  takes `logs` and, under gpu_full, builds the resident SoA via `gpu_build_cpu_ops_resident` (recompute
  `from_log` rv1/rv2/arg2/res/rvd/flags ON DEVICE from log+decode SoA) instead of `gpu_upload_cpu_ops_resident`.
  Threaded `logs` into `build_traces` + both callers. build_shared_devops ~101ms unchanged (p1 still builds host
  cpu_ops for the collectors) — as predicted marginal; keystone for the mandate (`from_log` now on GPU).
- **B (skip dead host op-vecs) — done, −145ms.** New `skip_opvec` flag (= gpu_full && !disabled, like
  `skip_bitwise`) threaded through `collect_ops_from_cpu_inner` + `collect_all_ops` + the Phase-3 memw→lt block.
  Skips building op-vecs that the resident chip TABLES (built from `devops`) + the device histogram no longer
  consume: **load, lt, shift** (p2a), **branch, eq, bytewise, store** + dvrm→lt (p2b), and the ~1M-push memw→lt
  `lt_ops` wrap (Phase 3, keeps the `memw_lt` pairs). KEEPS `mul_ops`/`dvrm_ops` (host MSB16 dedup tail still
  reads them) + `cpu32_ops` (drives cpu32→mul/dvrm derivations). Verified dead by grep + e2e (an empty op-vec
  that were consumed would unbalance its chip bus → verify fails). p2b_collect **102→35ms**, trace_build **→~1.75s**.
- **C (ECSM carry witness → GPU) — done, −192ms.** DIAGNOSIS OVERTURNED THE HANDOFF: p2a's cost is NOT the
  register-state advance (that's array-based, only **~24ms**) — it is the **20 ECSM ecalls' EC-scalar-mult
  witness (267ms)**. Split (`::ecsm::compute_witness`): replay_double_and_add (k256) **58ms** + quotients
  (big-int) **6ms** + **carry convolutions (`conv`) 190ms**. Moved ONLY the carries to GPU (they're pure 8-bit
  limb-convolution integer math — NO secp256k1 field ops; replay+quotients stay CPU, k256 is fast+audited):
  - New CUDA kernel `ecdas_carries` (trace_cpu.cu) ports `conv`/`limb_carries`/`carries_{lambda,xr,yr}` bit-exact
    (`__int128`, p/3p embedded, block=32 for register pressure). Launcher `gpu_build_ecdas_carries[_dev]`
    (precompile.rs), registered device.rs. **Parity `gpu_ecdas_carries_parity` bit-exact over 1249 steps.**
  - `ecsm::compute_witness_carryless` skips the 190ms per-step carries (keeps replay + quotients + the per-ecall
    x2/yG carries). `collect_ecsm_ops` uses it under gpu_full. `build_traces` calls
    `gpu_trace::fill_ecdas_carries_device(&mut ecdas_ops)` (packs point+quotient bytes → kernel → writes c0/c1/c2
    back) BEFORE the bitwise collector (`collect_bitwise_from_ecdas`, which reads carries) + the ECDAS fill —
    mandatory `assert!` (a carryless step with no device fill = zero carries). p2a_collect **308→100ms**.
- **NET: cluster 596→~318ms; trace_build 1.895→~1.54s (−19%, stable over 3 runs 1.527/1.538/1.547).
  e2e prove+verify PASSES; CPU-path regression
  (LAMBDA_VM_CPU_TRACE=1) PASSES (all new code gated on gpu_full → default byte-identical).** Files:
  crypto/ecsm/src/{witness.rs, lib.rs}, crypto/math-cuda/{kernels/trace_cpu.cu, src/precompile.rs, src/device.rs},
  prover/src/tables/{trace_builder.rs, gpu_trace.rs}, prover/src/tests/{gpu_ecdas_carries_parity.rs, mod.rs}.
  Local↔box md5-synced; nothing committed. My temp `[pc-probe]` eprintlns REMOVED; pre-existing
  `[p4-probe]`/`[p3-probe]` still present (not this session's).
- **Remaining host in the cluster** (open, diminishing): p1_cpu_ops 138ms (`from_log` for the host collectors —
  needed until the register advance + mul/dvrm/cpu32 derivations move off host), register advance ~24ms, ECSM
  replay 58ms (by design on CPU: k256 audited, sequential, GPU-hostile). Next candidates if pushing further:
  device MUL/DVRM dedup tail (frees mul_ops/dvrm_ops) → then drop host cpu_ops (p1); or stop (mandate largely met,
  the last big serial monster ECSM-carries is now on GPU).

## ✅ SESSION 5 (2026-07-21) — LT-resident-table STEP 1: device `memw→lt` derivation (the hard 75%)
User: "push ahead" on LT. LT breaks into instruction+dvrm→lt (~25%, routine) + memw→lt (~75%, the
entangled bulk). Did the memw→lt DERIVATION on device (the risky/hard part):
- New kernels `memw_lt_widths` / `memw_lt_emit_aligned` / `memw_lt_emit_general` (trace_walk.cu) +
  `math_cuda::trace_walk::gpu_memw_lt_pairs(pa,na,pg,ng)` — from the packed MEMW_A/MEMW rows, emit the
  timestamp-ordering LT operands `(lhs=old_ts[i], rhs=timestamp)` on device (aligned→1, general→width),
  compacted via excl_scan of widths. Device analog of `collect_lt_from_memw`/`_aligned`.
- PARITY `gpu_memw_lt_parity.rs::gpu_memw_lt_pairs_matches_cpu`: 200000 pairs MULTISET-identical to CPU.
- WIRED: `build_device_memw_ls` generates the pairs from `pa`/`pg` → `DeviceMemwLs.memw_lt_pairs`;
  `build_traces` p3 uses them for the REGULAR rows (`LtOperation::new(lhs,rhs,false)` into lt_ops) and
  runs the host `collect_lt_from_memw` ONLY over the retained+ecall slices (`[reg_*_lo,+reg_*_n)` skipped)
  → no double-count. **GPU_FULL e2e prove+verify PASSES** (LT bus balances → the device memw→lt is
  bit-correct in the real pipeline over 917k rows). Files: crypto/math-cuda/{kernels/trace_walk.cu,
  src/device.rs, src/trace_walk.rs}, prover/src/tables/trace_builder.rs, prover/src/tests/
  gpu_memw_lt_parity.rs (+ mod.rs). Local↔box md5-synced; nothing committed.
- ✅ STEP 2A DONE + e2e-VERIFIED (session 5, new box): the LT HISTOGRAM source is now FULLY on device —
  instruction+dvrm→lt scattered via device key gathers (`lt_key_gather`/`dvrm_lt_key_gather` +
  `bitwise_hist_lt`) inside `gpu_bitwise_hist_full`; memw→lt via the device-derived pairs fed as
  `OpVecSources.lt`. `gpu_bitwise_hist_full_sources` gained `memw_lt_lhs/rhs` params; build_traces p3
  captures the full memw→lt pairs (device regular ⊕ host ecall/retained) into `memw_lt_lhs/rhs`.
  **⇒ ALL 7 histogram op-vec sources (branch/load/cpu32/shift/mul/dvrm/lt) now DERIVED on device**
  (memw_aligned's IS_HALF is the only remaining host-built op-vec array, from device-walk rows). GPU_FULL
  e2e PASSES. Files: crypto/math-cuda/src/bitwise_hist.rs (device LT scatter block),
  prover/src/tables/trace_builder.rs. lt_ops STILL built for the TABLE (below).
- ✅ STEP 2B DONE + e2e-VERIFIED (session 5, new box): the LT TABLE is now built RESIDENT on device — the
  LAST ALU chip table to go resident. `gpu_build_lt_full_resident_from_devops` (trace_ops.rs) merges the 3
  sources on device (instruction `lt_key_gather` + dvrm→lt `dvrm_lt_key_gather` + memw→lt via the new
  `lt_memw_key_write` kernel, k0=0), does ONE GLOBAL dedup (`dedup3_core`), then SPLITS the unique rows into
  ≤max_rows.LT chunks (memcpy_dtod each chunk's packed rows → `lt_fill` → device-input `TraceTable`). Global-
  dedup-then-split is BUS-EQUIVALENT to the host per-chunk dedup (LT is a LogUp multiset), and multiple
  device-input LT chunk tables commit fine — GPU_FULL prove+verify PASSES. `build_lt_resident_tables_from_devops`
  (gpu_trace.rs) wraps the chunks; `gen_lts` uses it under gpu_full (falls back to host `gpu_build_lt_tables`
  otherwise). Files: crypto/math-cuda/{kernels/trace_ops.cu (lt_memw_key_write), src/device.rs, src/trace_ops.rs},
  prover/src/tables/{gpu_trace.rs, trace_builder.rs}.
  **⇒ LT is now FULLY on device (table + histogram), and LT was the last ALU chip table still host-fed.**
- ⚠️ REMAINING (small, OPTIONAL — times only): host `lt_ops` is STILL assembled (instruction LT push +
  dvrm→lt push + the ~1M-push memw wrap) but is now UNUSED under gpu_full (gen_lts uses the resident table;
  the histogram uses the memw_lt pairs). Dropping the dead lt_ops build (skip flags in collect_ops_from_cpu_inner
  ~929 + collect_all_ops + the p3 memw wrap) is a pure speed cleanup (the TRACE is 100% GPU-built already).
  The REAL remaining mandate work is the DECODE+COLLECT CLUSTER (p1/p2a/p2b + register walk + ecall assembly).
  - STEP 2 CONFIRMED DETAILS (session 5): LT dedup key = `k0 = signed|invert<<1, k1=lhs, k2=rhs`
    (`lt_key_gather` trace_ops.cu:470); dvrm→lt and memw→lt use `k0=0` (unsigned). So the resident full
    LT table = extend `gpu_build_lt_instr_dvrm_resident` (which merges instruction f0 + dvrm f5 into
    k0/k1/k2 then dedup3_core+lt_fill) with a 3rd source: write the memw→lt pairs into k1/k2 at
    `base=rows_lt+rows_dv` (k0 stays 0 from alloc_zeros) via a tiny `lt_memw_key_write` kernel, bump
    total, dedup all. STEP 2 = all-or-nothing (lt_ops feeds BOTH the table `gpu_build_lt_tables` AND the
    histogram `OpVecSources.lt`): must (a) resident LT table from the 3 sources, (b) histogram LT scatter
    from the 3 sources (empty OpVecSources.lt), (c) skip host lt_ops assembly (instruction LT push
    cpu.rs~929, dvrm→lt push collect_all_ops~4221, memw→lt wrap) under gpu_full, (d) thread the memw→lt
    pairs (device regular ⊕ host ecall/retained) to BOTH p4 (hist) + p5 (table). Full LtOperation flags:
    instruction from alu_flags, others 0. ecall/retained memw→lt = tiny; fold into the memw pair arrays.

## 🎯 HEADLINE STATE (updated session 4)
The WHOLE trace-gen runs on GPU behind ONE flag **`LAMBDA_VM_GPU_FULL=1`** and **full-prove-VERIFIES on
ethrex_5tx**, still **−17% vs the 8-core CPU warm** (trace_build ~**3417ms**, was ~3398 baseline; warm-bench
run-to-run noise ~±40ms). Session 4 (user goal clarified = WHOLE trace-gen on GPU, times don't matter) moved
**6 of the 7 remaining histogram op-vec sources on-device** (branch, load, cpu32, shift, mul, dvrm) — all
parity bit-exact + e2e-verified; net trace_build ~flat within noise. Only **LT** remains host (memw-walk
coupled). Default path (no flag) = CPU, unchanged + regression-clean. `LAMBDA_VM_CPU_TRACE=1` = all-CPU
kill-switch. Everything local↔box **md5-synced**; TREE IS GREEN.

## ✅ SESSION 4 (2026-07-21) — S3: moved 6/7 histogram op-vec sources ON-DEVICE (branch/load/cpu32/shift/mul/dvrm)
User picked "histogram op-vecs on-device"; after cpu32 clarified the goal is COMPLETENESS (whole trace-gen on
GPU, times don't matter), so all sources are kept ON even when they regress warm. Each source: reuse the
device op-stream enumeration the resident chip TABLES already do, add a `*_packed` scatter (a shared
`__device__` emit helper serves both the SoA kernel and the packed kernel — no bump-logic duplication),
launch it in `gpu_bitwise_hist_full`, empty that source in `OpVecSources`, drop its host SoA. Each is
parity-gated (synthetic, in `gpu_bitwise_opvec_parity.rs`, run `--test-threads=1` — the GPU parity tests
RACE when parallel on the shared backend; a known harness hazard, NOT a bug) + GPU_FULL e2e prove+verify.
- **branch+load** — `bitwise_hist_branch_load_packed` (self-routes packed/flags; BRANCH next_pc from
  pc/imm/rv1/jalr; LOAD Msb8 from rvd+width). Needed new `pc_dev`/`imm_dev`/`flags_dev` params on
  `gpu_bitwise_hist_full`. Parity 575000 bumps == CPU.
- **cpu32** — reuse `build_cpu32_ops` (res validated) → `bitwise_hist_cpu32_packed` (reads pack_cpu32_op rows).
  Parity 1,599,824 == CPU.
- **shift** — reuse `build_shift_ops`+`cpu32_shift_ops` (3-u64 rows [value,shift_amount,flags]; shift =
  shift_amount&0xff) → `bitwise_hist_shift_packed` (shared `bh_shift_emit`). Parity 3,004,682 == CPU.
- **mul (per-op)** — reuse the 4-source merged key gather (`mul_key_gather`+`mul_dvrm_key_gather`+
  `cpu32_mul_ops`+`cpu32_dvrm_mul_key_gather`; k0=flags,k1=lhs,k2=rhs) → `bitwise_hist_mul_perop_packed`
  (shared `bh_mul_perop_emit`). Parity 2,000,000 == CPU.
- **dvrm (per-op)** — reuse the 2-source key gather (`dvrm_key_gather`+`cpu32_dvrm_ops`; k0=flags,k1=n,k2=d)
  → `bitwise_hist_dvrm_perop_packed` (shared `bh_dvrm_perop_emit`). Parity 2,200,000 == CPU.
  ⚠️ MUL/DVRM: only the PER-OP part moved. The chunk-deduped MSB16/NEG-ZERO tail
  (`collect_bitwise_from_{mul,dvrm}_dedup`) STAYS host (reads the host `mul_ops`/`dvrm_ops` Vecs, which are
  still built) — this is correct (under `gpu_opvec` the host runs only the dedup tail; device covers per-op).
- **Cost note:** each source adds route+scan+gather passes over ALL ~4M cycles (and `chipop_alu_route` is
  re-run in the shift/mul/dvrm blocks — redundant). Net warm ~flat (host_soa_build 48→36ms offsets it), still
  −17%. FUTURE speed cleanup: compute the routes ONCE + share the op-streams across p4(hist)+p5(tables).
- **LT = NOT DONE (the hard one) — INVESTIGATED in detail (session 4).** `lt_ops` = instruction ⊕ dvrm→lt ⊕
  **memw→lt** (`collect_lt_from_memw`/`_aligned`, trace_builder.rs:4437-4438). Under gpu_full the device
  histogram still gets ALL of LT via the host-built `lt_lhs`/`lt_rhs` SoA. Breakdown (ethrex_5tx: lt_ops
  ~1.37M): **instruction+dvrm→lt ~25%** (routine — `lt_key_gather` + `dvrm_lt_key_gather` exist; scatter
  `bitwise_hist_lt`; but needs the collectors to skip instr+dvrm→lt under gpu_full, and it's only 25%);
  **memw→lt ~75%** (the bulk, entangled). memw→lt = per general memw row `width` LTs `(old_ts[i], timestamp)`,
  per aligned row 1 LT `(old_ts[0], timestamp)`, all unsigned.
  - The REGULAR-row memw→lt IS device-computable: inside `gpu_build_memw_ls_resident_ecall` (trace_walk.rs
    ~1846), right before `memw_pack`, the per-op **`g_ts`(old_ts[8]), `opts_d`(op timestamp), `width_d`,
    `fa`/`fg`(aligned/general flags)** are all resident. Emit `LT(g_ts[op*8+i], opts_d[op])` per op (aligned=1,
    general=width), scatter `bitwise_hist_lt` → return a hist delta EXACTLY like FR4a's `page_hist`
    (DeviceMemwLs.page_hist → device_page_hist → merged p4). Add `memw_lt_hist` the same way.
  - CRUX (why it's not clean): the memw buckets MIX device-regular rows (`d.general`/`d.aligned`) with
    HOST/device **ecall** rows (`d.ecall_*`) + retained (register||ecall) rows (trace_builder.rs:4411-4421).
    The device delta covers only the regular rows; the ecall/retained memw→lt must STAY host — so
    `collect_lt_from_memw` must be restructured to run over ONLY the ecall/retained rows (not d.general/
    d.aligned) to avoid double-count. That row-provenance split is the A2-tangled memw-walk integration.
  - CONFIRMED opts_d = op timestamp (`op_ts[r] = i*4+4` in memacc_emit); g_ts = old_ts[8] per op; single
    caller of the walk (build_device_memw_ls). So the walk-delta kernel is: per op, aligned(fa)→1 LT, general
    (fg)→width LTs, `LT(g_ts[op*8+i], opts_d[op])`, scatter `bitwise_hist_lt` (extract a `bh_lt_emit` helper).
  - ⚠️ EXTRA COUPLING (why LT can't be partial-ed): `lt_ops` feeds BOTH the histogram source AND the **LT
    trace TABLE** (`gen_lts` → `gpu_build_lt_tables(&lt_ops)`) — and **LT is the one chip table still built
    host-from-`lt_ops`** (all other chip tables are resident-from-devops). Emptying `lt_ops` for the histogram
    would break the LT table. So moving LT to device means moving BOTH the histogram source AND the table, and
    both need all 3 sources INCLUDING memw→lt. `gpu_build_lt_instr_dvrm_resident` exists (instr ⊕ dvrm→lt) but
    is MISSING the memw→lt source. Doing instruction+dvrm→lt alone is NOT a clean increment (lt_ops still
    needed for the table + memw→lt).
  - ⇒ LT is the "LT-resident-table" project: a single coherent multi-file, soundness-sensitive memw-walk
    integration (add memw→lt to the resident LT table `gpu_build_lt_instr_dvrm_resident` + the histogram
    delta; restructure the memw→lt row split; strong e2e bus-balance gate). NOT a clean 1-increment scatter
    like the other 6. Best done fresh; overlaps the real mandate blocker (the collect cluster).
- Files: crypto/math-cuda/{kernels/bitwise_hist.cu (5 packed kernels + 3 shared emit helpers; NOTE a `#if 0`
  legacy `bitwise_hist_mul_perop_OLD` block — DELETE before commit), src/device.rs (5 regs),
  src/bitwise_hist.rs (5 launches in gpu_bitwise_hist_full + 5 parity wrappers)},
  prover/src/tables/trace_builder.rs (`gpu_bitwise_hist_full_sources`: branch/load/cpu32/shift/mul/dvrm SoA
  dropped, LT + memw_aligned kept), prover/src/tests/gpu_bitwise_opvec_parity.rs (5 `*_packed` parity tests).
  Local↔box md5-synced; nothing committed. TEMP `[p4-probe]`/`[p3-probe]` eprintlns + the `#if 0` block still
  present — remove before commit.
- **⚠️ BIG PICTURE (told the user):** finishing all 7 histogram sources does NOT drop a host trace-gen PHASE.
  The host STILL runs the DECODE+COLLECT CLUSTER: p1 `collect_cpu_ops` (4M `Vec<CpuOperation>`; device SoA is
  UPLOADED from it, S1 not wired), p2a/p2b op-vec collectors, the 4M-op `register_state` advance (final state
  + ecall indices, A3), ecall/precompile assembly. THAT cluster (GPU-DECODE-COLLECT-REARCH S2/S3/S4) is the
  real "whole trace-gen on GPU" blocker. NEXT: LT (memw-coupled), then the collect cluster.

## 🔥 SESSION 3 ARC (what flipped it from a +40% LOSS to a −17% WIN) — details in the dated sections below
1. **FIXED the warm bench** (it couldn't run GPU_FULL — panicked `ECSM ScalarIsZero`): now uses
   `Traces::from_elf_and_logs_with_precompiles(&elf,&logs, Some(&result.precompile_inputs), …)`. First TRUE
   warm number: GPU_FULL was **+40% SLOWER** — and the cause was HOST-SIDE MARSHALLING, not GPU compute
   (the atomic-scatter is 22ms). **FR5 (SMEM scatter) = MOOT/DELETED.**
2. **FR4a** device-resident PAGE ARE_BYTES histogram (`bitwise_hist_page_snapshot`, from the device snapshot)
   → killed the ~1s serial `build_page_bitwise_arrays`. −1072ms (5705→4633).
3. **Device MEMW_R tables** built DIRECTLY from the register walk (`gpu_build_memw_register_tables_from_walk`
   → `gpu_walk_route_memw_register_ecall_chunked`/`fill_chunk_on`, via `device_memw_tables`; `want_tables`) →
   killed the 447MB row download + 9.3M host RegRow build. −944ms (4633→3610). **⇒ now beats CPU.**
4. **FR6** skip building the ~28M host `bitwise_ops` under gpu_full (`skip_bitwise` param) — device covers
   in-walk. −166ms p2a (3610→3466).
5. **FR4b (partial)** STORE/EQ/BYTEWISE op-vec histogram computed ON-DEVICE from resident packed/rv1/rv2/arg2
   (`bitwise_hist_opvec_packed`). ~flat (mandate-purity; 3466→3398).

## ⚠️ CRITICAL BEFORE ANY COMMIT: remove TEMP probes
`[p4-probe]` eprintlns in `prover/src/tables/trace_builder.rs` (build_shared_devops / build_page_arrays /
base_assembly+merges + host_soa/device_call in `gpu_bitwise_hist_full_sources`) and `[p4-probe]` in
`crypto/math-cuda/src/bitwise_hist.rs` (`gpu_bitwise_hist_full` scatter/reduce split); `[p3-probe]` eprintlns
in `build_device_memw_ls` (trace_builder.rs) + the register block. Plus a `[gpu-memory]`/`[bitwise-src]`
eprintln. Grep `p4-probe|p3-probe|gpu-memory|bitwise-src` and remove before committing.

## 🧭 COMMITTED NEXT DIRECTION (user-chosen s3): FULL decode+collect cluster on GPU (multi-session)
Move p1 `collect_cpu_ops` + p2a/p2b collect (~1.15s SERIAL, the last big CPU chunk) onto the GPU as a unit.
**Plan: `reports/tracegen/GPU-DECODE-COLLECT-REARCH.md`** (staged S1–S5, bit-exact parity-gated).
- **S1 (device `cpu_ops` builder) = ALREADY DONE + VALIDATED** (original resident-pipeline work):
  `math_cuda::trace_ops::gpu_build_cpu_ops_resident` builds the full `DeviceCpuOpsResident` on device from
  logs+decode; `gpu_cpu_ops_parity.rs::gpu_build_cpu_ops_matches_from_log` PASSES bit-exact over 4.03M cycles
  (re-confirmed green this session). Device ALU chip-op derivation also exists (trace_ops.cu:137+).
- **CRUX = S3/S4** (op-vectors on device + DROP the host `collect_cpu_ops`/`collect_ops_from_cpu`). S2 alone
  is marginal (cpu_ops still needed on host for the collectors — lt_ops for the LT table, mul/dvrm for dedup,
  ecall/precompile threading). **START S3/S4 fresh next session.**
- NO-GOs found this session (do NOT re-attempt): FR5 SMEM scatter (scatter=22ms); A1 device PAGE *table*
  (VERIFIES but net-loss — gen_pages is PARALLEL-HIDDEN in p5; kept OFF, infra retained incl. a working
  2-tree device-commit path); A2/A3 (entangled). See "TIER A" + "DEEP PROJECT" sections.

## MEASUREMENT RULE (hard-won): warm bench ONLY
Measure speed ONLY with the WARM in-process bench `gpu_resident_bench` (≥3 iters, compare best/median), and
it MUST pass `result.precompile_inputs` via `from_elf_and_logs_with_precompiles` (else GPU_FULL panics
`ECSM ScalarIsZero`). Single cross-process `cargo test` runs pay cold CUDA init → inflate trace_build (burned
this effort as a phantom win). See auto-memory `measure-gpu-tracegen-warm.md`.

## BOX + BUILD/TEST (vast.ai RTX 5090, EPHEMERAL — may be gone; if so get a new RTX 5090 + CUDA≥13 & re-sync)
- **CURRENT box (2026-07-22): `ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
  -o IdentityAgent=none -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 -p 63154 root@174.31.67.221`** · repo
  `/workspace/lambda_vm` · RTX 5090, CUDA 13.1, cargo 1.94, `/tmp/runenv.sh` written. Re-synced from base
  fb204215 + the 58-file working-tree overlay + fixtures (all md5-matched); green baseline VERIFIED here
  (build ~48s since workspace deps were cached → FAST iteration). Old box `-p 48889 root@47.164.117.172` DEAD.
- Re-sync recipe used on a fresh box: `git checkout -f fb204215`; `mkdir -p executor/program_artifacts/rust`;
  `rsync -a --files-from=<list of `git status` code files + ethrex.elf> -e "ssh <opts> -p PORT>" ./ root@IP:
  /workspace/lambda_vm/` (rsync ONE connection — per-file scp of ~58 files times out); md5-verify all; write
  /tmp/runenv.sh. (`ethrex.elf` is gitignored → must be rsynced explicitly; `ethrex_5tx.bin` is in git status.)
- (historical) `-p 48889 root@47.164.117.172` · repo `/workspace/lambda_vm` · CUDA 13.1 `/usr/local/cuda`.
- Env (already at `/tmp/runenv.sh` on the box): `source $HOME/.cargo/env; export CUDA_HOME=/usr/local/cuda
  PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH LAMBDA_VM_BENCH_ELF=$PWD/executor/program_artifacts/rust/
  ethrex.elf LAMBDA_VM_BENCH_INPUT=$PWD/executor/tests/ethrex_5tx.bin`.
- **e2e verify:** `LAMBDA_VM_GPU_FULL=1 cargo test -p lambda-vm-prover --release --features cuda,instruments
  --lib gpu_resident_e2e -- --ignored --nocapture` (expect `[gpu-memory] … drop=true` + "OK: full prove+verify").
- **warm bench:** same but `gpu_resident_bench` (prints trace_build + phase tree). Default regression: drop the
  flag. **S1 parity:** `cargo test … --lib gpu_build_cpu_ops -- --ignored --nocapture` (expect "parity OK").
- **Re-sync if a new box:** `git checkout fb204215` on the box, then scp the LOCAL working-tree diff of
  `crypto/ executor/ prover/` (individual changed files → `/tmp` → `cp` into place; full-tree tar is blocked),
  + fixtures `executor/tests/ethrex_5tx.bin` + `executor/program_artifacts/rust/ethrex.elf`. md5-verify each.

## FILES TOUCHED THIS SESSION (working tree; all local↔box md5-synced)
- `crypto/math-cuda/kernels/{bitwise_hist.cu (page_snapshot + opvec_packed), trace_cpu.cu (page_fill_snapshot,
  A1-inert)}`, `crypto/math-cuda/src/{bitwise_hist.rs, device.rs, trace_cpu.rs, lde.rs (no_merkle_dev)}`.
- `crypto/stark/src/{gpu_lde.rs (try_lde_row_major_no_merkle_dev), prover.rs (Path C device preprocessed LDE)}`
  — the 2-tree device-commit infra (verifies past commit; A1 kept OFF).
- `prover/src/tables/{trace_builder.rs, gpu_trace.rs, types.rs (fe_vec_from_u64), tests/gpu_resident_bench.rs
  (precompile_inputs fix)}`.
- NEW report: `reports/tracegen/GPU-DECODE-COLLECT-REARCH.md` (the committed re-arch plan).

## ⚠️⚠️ 2026-07-21 (session 3) — WARM MEASUREMENT CORRECTION (invalidates FR5 premise, reframes the plan)
FIXED the warm bench so it can run GPU_FULL (it PANICKED before: `from_elf_and_logs` passed `None` precompile
recordings → ECSM `ScalarIsZero` under memory-drop; the bench now uses `from_elf_and_logs_with_precompiles`
with `result.precompile_inputs`). First TRUE warm GPU_FULL number on ethrex_5tx:
- **trace_build: GPU_FULL 5705ms vs all-CPU (`LAMBDA_VM_CPU_TRACE=1`) 4085ms → GPU is +40% SLOWER warm.**
- Split (GPU_FULL): p2a 732ms (drop win −490) · p3to5 3908ms (+2157 regression) [p4_bitwise 1863ms · p5 315ms ·
  p3to5-OTHER ~1690ms = device memw walk + `from_image_and_snapshot` reconstruction + row unpack, host loops].
- **p4 internal probe (the smoking gun):** `build_page_bitwise_arrays` = **~1020ms (serial host loop over
  ~4.7M cells)** · base_assembly+merges ~430ms · build_shared_devops ~160ms · **GPU histogram total only
  ~135ms (device scatter = 22ms, reduce+dl 58ms, host SoA build 50ms).**
- ⇒ **The GPU atomic-scatter is 22ms — it was NEVER the bottleneck. FR5 (SMEM scatter) is MOOT/DELETED.**
  The +40% loss is HOST-SIDE serial marshalling around fast device calls: the ~1s `build_page_bitwise_arrays`
  (p4) + the ~1.69s device-walk/reconstruction (p3to5-other). Device COMPUTE is fast; feeding/harvesting it
  (SoA build, snapshot download, MemoryState reconstruction, per-byte host loops) is the cost.
- Probes are TEMP `eprintln!("[p4-probe] …")` in trace_builder.rs (build_shared_devops/build_page_arrays/
  base_assembly+merges) + bitwise_hist.rs `gpu_bitwise_hist_full` (scatter/reduce split) + host_soa/device in
  `gpu_bitwise_hist_full_sources`. **REMOVE all `[p4-probe]` eprintlns before commit.**

## ✅ FR4a DONE + e2e-VERIFIED + warm-measured (2026-07-21 s3) — recovered ~1072ms
Device-resident PAGE ARE_BYTES: new kernel `bitwise_hist_page_snapshot` (crypto/math-cuda/kernels/bitwise_hist.cu)
+ `math_cuda::bitwise_hist::gpu_bitwise_hist_page_snapshot` + registered in device.rs. Computed INSIDE
`build_device_memw_ls` (image `img_addr/img_val` + snapshot `snap_addr/snap_val` both live there; page_bases
from the reconstructed state, ~18 pages), returned as `DeviceMemwLs.page_hist`, captured into
`device_page_hist`, merged in p4 under `use_device_page = gpu_full && device_page_hist.is_some()`. Gated:
`build_page_bitwise_arrays` SKIPPED (gpu_page_arrays=None) + `cpu_page &&= !use_device_page` (no double count)
+ gpu_full_hist gets empty page arrays. Graceful fallback: device fail → host page path. **e2e GPU_FULL full
prove+verify PASSES** (ARE_BYTES bus balances → bit-correct). **WARM (ethrex_5tx):** build_page_arrays
1020ms→0 · p4_bitwise 1863→730ms · trace_build **5705→4633ms**; GPU_FULL warm gap vs all-CPU (4085ms)
**+40% → +13%**. Files: crypto/math-cuda/{kernels/bitwise_hist.cu, src/device.rs, src/bitwise_hist.rs},
prover/src/tables/trace_builder.rs. (Local↔box md5-synced; box `-p 48889 root@47.164.117.172`.)

## ✅✅ MILESTONE (2026-07-21 s3): GPU_FULL trace-gen now BEATS the 8-core CPU warm (−12%)
Two device-resident fixes this session removed ~2000ms of HOST marshalling and FLIPPED the result:
- FR4a device-resident PAGE (−1072ms).
- **Device-resident MEMW_R tables** (−944ms): `build_device_memw_register` wired to
  `gpu_walk_route_memw_register_ecall_chunked` (via new `gpu_trace::gpu_build_memw_register_tables_from_walk`)
  — the register walk fills the MEMW_R chunk tables DIRECTLY on device (`fill_chunk_on`), returned as
  resident `TraceTable`s through the `device_memw_tables` path; NO 447MB row download, NO 9.3M host RegRow
  build. `build_device_memw_register` gained `want_tables` (=gpu_full); under gpu_full `memw_register_rows`
  is EMPTY (tables feed gen_memw_registers directly; IS_HALF from the resident walk hist). The MEMW_R bus is
  partition-invariant across chunks → the deferred "multi-chunk not verified" concern was just UNTESTED, and
  **e2e GPU_FULL full prove+verify PASSES.** `device_memw_register` probe 1020ms→**76ms**.
- **WARM trace_build (ethrex_5tx): 5705 (start) → 4633 (FR4a) → 3610ms (MEMW_R tables). all-CPU = 4085ms →
  GPU_FULL is now ~475ms FASTER (−12%).** This OVERTURNS the effort's long-standing "GPU trace-gen ties/loses
  to the 8-core CPU" conclusion — with host marshalling removed, the WHOLE trace-gen on GPU WINS.
  (Trace-gen is ~9% of a ~24s prove, so net prove speedup is modest ~2-3%, but the mandate 'everything on GPU'
  is met with a real trace_build win, not a regression.)
- Files: prover/src/tables/{gpu_trace.rs (new `gpu_build_memw_register_tables_from_walk`), trace_builder.rs
  (`DeviceMemwRegister.tables`, `build_device_memw_register` want_tables, register block wiring)}.
- ⚠️ Default + GPU_REGISTERS-only paths UNCHANGED (want_tables=false → legacy rows path). Regression:
  see box /tmp/memwreg_regression.log.

## ✅ FR6 Stage 1 (2026-07-21 s3): skip host `bitwise_ops` under gpu_full — −166ms p2a
Under gpu_full the WHOLE in-walk BITWISE histogram is on device, so the CPU no longer builds the ~28M
`bitwise_ops` (in-walk per-op 3×ARE_BYTES+4×IS_HALF + LOAD MSB8 + CPU32 bitwise) — they were dropped
uncounted anyway (`cpu_bitwise_sources=false`). New `skip_bitwise` param threaded through
`collect_ops_from_cpu_inner` + `collect_all_ops` (gates lines ~791/796/935/cpu32-extend + the 28M Vec alloc),
computed `= gpu_full_enabled() && !disabled` at the p2a call site. Same GPU-mandatory contract as
device_memory_drop (gpu_full_hist is `.expect`, so no silent gap). **e2e GPU_FULL + default regression PASS.**
WARM: **p2a 732→566ms, trace_build 3610→3466ms (now −15% vs all-CPU 4085).**

## FR4b ROI is now SMALL (measured): the big p2a waste (bitwise_ops) is gone. The op-vec HOST SoA build for
the histogram is only ~50ms (probe), and the op-vectors that FR4b would let us drop (store/eq/bytewise/branch/
load) are cheap to build and partly still needed (lt table needs lt_ops; mul/dvrm dedup needs mul/dvrm_ops;
resident chips already read `devops` not the op-vectors). So FR4b (10 device kernels reading `packed`) is a
~50-100ms + mandate-purity move, not a big speed lever. p2a's remaining 566ms is largely inherent: the 4M-op
iteration + register_state advance (for final state/HALT) + load/lt/shift/cpu32 op building + precompile ecall
extraction. (A separate lever: get the FINAL register state from the device walk to drop the 4M-op host
register advance — new marshalling change, not FR4b.)

## ✅ FR4b (partial) 2026-07-21 s3: STORE + EQ + BYTEWISE op-vec computed ON-DEVICE from resident packed
New kernel `bitwise_hist_opvec_packed` (crypto/math-cuda/kernels/bitwise_hist.cu) + registered in device.rs;
`gpu_bitwise_hist_full` gained `arg2_dev` and launches it (self-filters by packed decode: store = memory∧
mem_flags0 → 8 ARE_BYTES on rv2; eq = !word∧alu∧alu_op==3 → 4 IS_HALF+ZERO on rv1-arg2; bytewise =
!word∧alu∧alu_op≤2 → 8 BYTE_ALU on rv1/arg2). `gpu_bitwise_hist_full_sources` leaves those 3 `OpVecSources`
empty (no host SoA) + passes `&devops.arg2`. **e2e GPU_FULL + default regression PASS** (bus balances → bit-
correct). WARM trace_build 3466→**3398ms (−17% vs all-CPU)** — mostly flat (mandate-purity, not speed).
- **ENTANGLEMENT WALL for the rest:** `host_soa_build` stayed ~48ms because the remaining op-vec sources
  (lt, shift, mul, dvrm) have DERIVED contributions — dvrm→lt (abs_r/abs_d), dvrm→mul (d*q), cpu32→shift/mul/
  dvrm, memw→lt — that are NOT 1:1 with a single cpu_op's packed, so they can't be pure packed-resident
  without also computing those derived ops on device. load/branch/cpu32 ARE 1:1-tractable (load: rvd+width
  Msb8; cpu32: hil/alu_flags/rs1/rs2/rd from packed + rv1/rv2/res; branch: needs next_pc computation) but
  each is ~0-few ms. So "no host op-vec SoA" (full mandate) is blocked on the entangled 4; FR4b delivered the
  clean 3 + the pattern. Files: crypto/math-cuda/{kernels/bitwise_hist.cu, src/device.rs, src/bitwise_hist.rs},
  prover/src/tables/trace_builder.rs.

## 🧭 COMMITTED DIRECTION (2026-07-21 s3, user-chosen): FULL decode+collect cluster on GPU
After exhausting the clean wins (−17%), the user committed to the big re-architecture: move p1
`collect_cpu_ops` + p2a/p2b collect (~1.15s SERIAL) onto the GPU as a unit. **Plan doc:
`reports/tracegen/GPU-DECODE-COLLECT-REARCH.md`** (staged S1–S5, bit-exact parity gated). **⭐ S1 (FOUNDATION)
turns out ALREADY DONE + VALIDATED** (original resident-pipeline work): `gpu_build_cpu_ops_resident`
(trace_ops.rs:229) builds the FULL `DeviceCpuOpsResident` on device from logs+decode; parity test
`gpu_build_cpu_ops_matches_from_log` PASSES bit-exact over 4.03M cycles (re-confirmed green this session).
Device ALU chip-op derivation also exists (trace_ops.cu:137+). ⇒ the CRUX is **S3/S4** (op-vectors on device
+ drop the host `collect_cpu_ops`/`collect_ops_from_cpu`) — S2 alone is marginal (cpu_ops still needed on
host for the collectors). Next real step = S3/S4 (the entangled multi-session core), start fresh.

## ⚠️ TIER A (2026-07-21 s3) — attempted A1, hit ENTANGLEMENT WALLS (2 of 3 blocked on deeper machinery)
The synthetic "Tier A = clean device-residency wins" OVER-ESTIMATED cleanliness. On investigation:
- **A1 (device PAGE table) — BLOCKED.** Built the kernel (`page_fill_snapshot`) + wrappers +
  `gpu_build_page_tables_from_snapshot`, wired via a device_page_tables path. FAILS at prove with
  `PrecomputedCommitmentMismatch`: PAGE (like BITWISE/DECODE/REGISTER) uses the **2-Merkle-tree
  PREPROCESSED commit** (`precomputed_tree` + `mult_tree`, `main_trees=2`; auto_storage.rs ~180). The
  resident device-buffer path (`set_main_input_dev`) only supports single-tree WITNESS tables (that's why
  the chips/MEMW_R worked). A device PAGE buffer can't satisfy the ELF-bound preprocessed root. → needs the
  COMMIT MACHINERY extended to split a device buffer into the 2 trees on-device. REVERTED to green (the
  kernel + `gpu_build_page_tables_from_snapshot` kept INERT as ready infra; `let tables = None` in
  build_device_memw_ls with a "A1 BLOCKED" comment). e2e re-verifies at 3398ms.
- **A2 (device MEMW_A/MEMW tables) — BLOCKED (clean version).** The tables MIX device-regular walk rows
  (pa/pg) with HOST-assembled ecall + register-fallback + halt rows. Clean device fill needs "ecall MEMW
  rows on device" (mixed-width emitting) — the known-deferred piece. Not the clean MEMW_R pattern (~85ms).
- **A3 (final register state from walk) — ENTANGLED.** The 4M-op host `register_state` advance also feeds
  the ecall collectors mid-execution (commit index x254) + HALT — not only the final state. Can't drop
  cleanly without device per-ecall register values + final state.
- **Takeaway:** the CLEAN host-marshalling wins are DONE (→ −17%). What's left touches the COMMIT structure
  (2-tree preprocessed) + HOST STATE THREADING (register_state → collectors) — bigger machinery projects,
  not table fills. The single highest-leverage deeper project = **make the resident/device commit path
  support 2-tree preprocessed tables** (unlocks A1 PAGE + potentially device DECODE/REGISTER/BITWISE).

## ⚙️ DEEP PROJECT ATTEMPTED (2026-07-21 s3): 2-tree device commit for preprocessed tables → A1 = NO-GO
Built the machinery to commit a device-resident preprocessed (2-tree) table:
- `math_cuda::lde::coset_lde_row_major_no_merkle_dev` (device-input, no-Merkle LDE via memcpy_dtod).
- `stark::gpu_lde::try_lde_row_major_no_merkle_dev` (wrapper).
- `stark::prover::commit_main_trace` Path C: for preprocessed + `main_input_dev`, LDE the device buffer
  directly → the existing 2-tree subset split. **This WORKS — a device preprocessed table now commits
  past the `PrecomputedCommitmentMismatch` check** (real, kept as inert ready-infra).
- REMAINING blocker for a FULLY-resident device preprocessed table: the **R2 aux/multiplicity build reads
  the host `main_table`** (a zeroed placeholder for a device table) → LogUp aux wrong → proof doesn't verify.
- LANDED A1 correctness via device-fill → DOWNLOAD to a real host matrix (`types::fe_vec_from_u64` +
  `gpu_build_page_tables_from_snapshot`); standard preprocessed path then handles commit+aux. VERIFIES.
  Bug found+fixed: PAGE `ts==0` must emit `(init, 0)` (not the stored value) — FR4a only checked values.
- **BUT A1 is a NET LOSS and is kept OFF:** `gen_pages` (~285ms) is PARALLEL-HIDDEN in the p5 rayon scope
  (p5 wall ~300ms bounded by other tables), so removing it saves ~0 wall; the device fill + 189MB download
  adds ~103ms serially → trace_build **3398→3463ms (+65ms)**. `page_fill_snapshot` + the whole 2-tree
  device-commit path are retained as ready infra (OFF via `let tables = None`).
- **MEASUREMENT LESSON (again):** a big-looking p5 `gen_*` time is often parallel-hidden — moving it to a
  SERIAL device fill+download loses. Only worth it if (a) it's the p5 critical path, or (b) fully resident
  (no download) AND the aux build reads the device buffer. Neither holds for PAGE today.
- **NET after the deep project: unchanged at GPU_FULL 3398ms (−17% vs CPU); A1 OFF.** The reusable win is
  the 2-tree device-commit capability (verifies past commit) for a FUTURE fully-resident path once the aux
  build is device-aware.

## REMAINING (GPU_FULL 3398 beats CPU 4085 by 17%). Next chunks if pushing further:
- devstate+memw_extend+memw->lt ~646ms (build_device_memw_ls 471 [img_sort+walk 216, unpack 87, recon 108,
  page 67] + memw retain/extend + MEMW→LT ~175). base_assembly+merges ~400ms (p4). build_shared_devops ~150ms.
- These are diminishing returns (already winning). FR4b (op-vec resident) is mandate-only (~50ms).

<details><summary>Pre-MEMW_R-tables gap analysis (historical)</summary>

## REMAINING warm gap = +548ms (GPU_FULL 4633 vs CPU 4085) — FULLY LOCALIZED (2026-07-21 s3):
- **`build_device_memw_register` ~1020ms ← #1 NEXT LEVER (same pattern as FR4a).** The device REGISTER walk
  (`gpu_walk_route_memw_register_ecall_rows_host`, trace_builder.rs ~1606) downloads ~9.3M rows (~447MB) and
  builds 9.3M `RegRow` structs on host via a SINGLE-THREADED `rows.iter().map(RegRow::new).collect()` (~1614),
  feeding a 155ms table gen (gen_memw_registers). Two fixes: (a) QUICK/off-mandate — `rows.par_iter()` +
  parallel SoA build (~saves 300-400ms, CPU); (b) MANDATE — build MEMW_R tables DIRECTLY on device from the
  walk (skip the 447MB download + host RegRow build): this is the deferred TODO at trace_builder.rs ~1587
  ("fill_chunk_on does not yet verify multi-chunk"), the analog of the resident chips. Under GPU_FULL the
  memw_reg IS_HALF already comes from the resident walk histogram, so these rows feed ONLY the MEMW_R tables.
- **build_device_memw_ls ~471ms**: img_sort+walk 215 (host image sort + GPU walk) · unpack+ecall_asm 85 ·
  `from_image_and_snapshot` reconstruction 108 (host, ~3M `set()`; still needed for gen_pages + touched_cells)
  · FR4a page_kernel 63.
- **base_assembly+merges ~400ms** (p4): base hist alloc (84MB zeroed) + 2× add_raw_counts (10.5M each:
  gpu_full_hist + device_page_hist) + EC/precompile + mul/dvrm-dedup CPU collector tail.
- devstate+memw_extend+memw->lt total ~640ms (= build_device_memw_ls 471 + retain/extend + MEMW→LT ~170).
- build_shared_devops ~150ms; GPU hist ~128ms (host_soa 50 + device 78, scatter 20).
- ⇒ the two ~1s host-marshalling levers were build_page_arrays (FR4a ✅ DONE) and device_memw_register (NEXT).
  Both = "device computes fast, host marshals the results slowly." Best case after fixing both ≈ GPU_FULL TIES
  the 8-core CPU (~4085ms) → "everything on GPU at no regression" (mandate met, no net prove speedup since
  trace-gen is ~9% of a ~24s prove). [UPDATE: actually BEAT it — 3610ms, see milestone above.]
</details>

## NEXT (in order):
- **FR4b:** OP-VEC resident from `packed` — mandate completeness only; op-vec host SoA build is ~50ms so
  speed value is small.
- **p3to5-other:** localize + attack (device-resident PAGE tables / touched_cells to kill the reconstruction).
- **FR6:** thread recordings through `from_image_and_logs`/continuation (the bench fix already threads them
  for `from_elf_and_logs_with_precompiles`).
- ~~FR5 SMEM scatter~~ DELETED (scatter=20ms).

<details><summary>FR4a original plan (done)</summary>
- **FR4a (THE #1 LEVER ~1s):** replace the serial `build_page_bitwise_arrays` (~1020ms) with a
  DEVICE kernel computing ARE_BYTES[init,fini] per byte of every page in page_bases, from the device snapshot
  (`snap_addr`/`snap_val`, available in `build_device_memw_ls` before/at download) + the sorted image
  (`img_addr`/`img_val`, already built there for the walk seed). page_bases = pages(image) ∪ pages(snapshot).
  Kernel: 1 thread/byte → binary-search image for init, binary-search snapshot for fini(=init if untouched),
  scatter ARE_BYTES. Compute it INSIDE `build_device_memw_ls` (data lives there) → return the page hist
  contribution → merge in p4 under `gpu_full`, skip `build_page_bitwise_arrays`. Gate: `gpu_full`. Validate
  bit-identical (e2e verify + a bin diag vs `collect_bitwise_from_page`), then warm-measure p4.
- **FR4b:** OP-VEC resident — compute lt/store/…/dvrm fields on-device from resident `packed` (was framed as
  killing uploads; note op-vec host SoA build is only ~50ms, so FR4b's speed value is small — do for mandate
  completeness, not speed).
- **(p3to5-other ~1.69s):** the OTHER big lever — device memw walk + `from_image_and_snapshot` reconstruction
  + row unpack are host loops; localize + make resident (keep snapshot/rows on device, don't reconstruct on
  host) to cut the biggest chunk. Not yet localized (would take one more probe).
- **FR6:** thread executor precompile recordings through `from_image_and_logs`/continuation so GPU_FULL
  memory-drop works off the monolithic/continuation path too (the bench fix already threads them for
  `from_elf_and_logs_with_precompiles`).
- ~~FR5 SMEM scatter~~ **DELETED** — scatter is 22ms, not a bottleneck.
</details>

## Box + re-sync (vast.ai boxes are EPHEMERAL — may be gone "later"):
- Last box: `ssh -p 48889 root@47.164.117.172` (RTX 5090, CUDA 13.1 at `/usr/local/cuda`, cargo 1.94).
- If down, get a NEW RTX 5090 + CUDA≥13 from the user. Setup: on the box `git checkout fb204215` (base is
  usually present; else fetch), then transfer the LOCAL working-tree diff. Fast path used this session:
  `{ git diff fb204215 --name-only -- crypto executor prover; git ls-files --others --exclude-standard --
  crypto executor prover; } | grep -vE '\.bin$|Cargo\.lock$|/target/' > /tmp/fl.txt; tar czf /tmp/wt.tgz -T
  /tmp/fl.txt` → scp → `git checkout fb204215 && git clean -fdq -- crypto executor prover && tar xzf`.
  Then scp fixtures `executor/tests/ethrex_5tx.bin` + `executor/program_artifacts/rust/ethrex.elf`.
- SSH opts (agent refuses the key): `-o IdentityAgent=none -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519`.
- Build/test env: `source $HOME/.cargo/env && export CUDA_HOME=/usr/local/cuda
  PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH LAMBDA_VM_BENCH_ELF=$PWD/executor/program_artifacts/rust/
  ethrex.elf LAMBDA_VM_BENCH_INPUT=$PWD/executor/tests/ethrex_5tx.bin`. Verify:
  `LAMBDA_VM_GPU_FULL=1 cargo test -p lambda-vm-prover --release --features cuda,instruments --lib
  gpu_resident_e2e -- --ignored --nocapture` (expect `[gpu-memory] … drop=true` + "OK: full prove+verify").

## Key files touched this effort (working tree; full list via `git status`):
- executor/src/vm/{execution.rs, memory.rs, instruction/execution.rs} (Option A recordings).
- prover/src/{lib.rs, tables/trace_builder.rs, tables/gpu_trace.rs, tables/page.rs, tables/bitwise.rs}.
- crypto/math-cuda/{src/{bitwise_hist.rs, trace_walk.rs, device.rs, trace_ops.rs, ...}, kernels/{trace_walk.cu,
  bitwise_hist.cu, ...}}.
- prover/src/tests/gpu_reg_emit_parity.rs (diags: diag_page_state, diag_ecall_oldstate, etc.).
- FR3/FR4 anchors: `gpu_bitwise_hist_full` + `scatter_opvec` (bitwise_hist.rs ~942);
  `gpu_bitwise_hist_full_sources` (trace_builder.rs, before the phase-0 walk-cost fns); wiring in
  `build_traces` p4 gated on `gpu_full` (`gpu_full_enabled()` + shared_devops).
- One opt-in `eprintln!("[gpu-memory] …")` in build_traces confirms the device path; remove before commit.

# ========================================================================

## ⭐ SESSION 2026-07-21 (cont.) — FULL-RESIDENT PIPELINE build (user directive: WHOLE trace-gen on GPU)

User directive: build the WHOLE trace generation on GPU; do NOT editorialize on worth/speed (see memory
[[no-worth-editorializing]]). Progress this sub-session (all verified on ethrex_5tx, box `-p 48889
root@47.164.117.172`, nothing committed):
- **FR1 ✅** all resident paths ON together (chips+registers+memory_drop+bitwise+memw_reg) → full
  prove+verify passes.
- **FR2 ✅** master flag **`LAMBDA_VM_GPU_FULL=1`** (gpu_trace.rs `gpu_full_enabled()`, OR'd into every
  gate: gpu_resident_chips_enabled, device_registers_eligible, device_memory_drop_eligible, use_gpu_bitwise,
  gpu_opvec_hist). One switch turns on the whole resident pipeline; verified alone.
- **FR4 (memw_reg) ✅** under GPU_FULL the memw_reg IS_HALF comes from the RESIDENT walk histogram
  (`gpu_memw_reg_hist` gate now allows it under gpu_full; `mrr=&[]` guard prevents double-count;
  `device_is_half` is always None) → kills the ~150MB `memw_register_rows` upload. Verified.
- Reverted the earlier resident-chips default flip (kept opt-in; warm ~flat).
- Files touched: prover/src/tables/{gpu_trace.rs, trace_builder.rs}. Local↔box md5-synced.

- **FR3 ✅ DONE + e2e-verified:** `math_cuda::bitwise_hist::gpu_bitwise_hist_full(...)` (+ factored
  `scatter_opvec`) scatters in_walk (resident packed/res) + op-vec + page + memw_reg (resident register
  walk, reusing resident `packed`) into ONE replicated histogram with ONE reduce/download. WIRED into
  `build_traces` via `gpu_bitwise_hist_full_sources` helper: under `gpu_full` (= GPU_FULL + shared_devops
  present) it gates the 3 separate device histograms (`gpu_opvec_hist`/`gpu_bitwise_hist`/`gpu_memw_reg_hist`)
  OFF, sets the CPU-skip flags (`gpu_opvec ||= gpu_full`, `cpu_bitwise_sources &&= !gpu_full`), and merges
  the ONE `gpu_full_hist` (required — `.expect` under the opt-in flag, no silent gap). `GPU_FULL=1` full
  prove+verify PASSES. Files: crypto/math-cuda/src/bitwise_hist.rs, prover/src/tables/trace_builder.rs.

- **FR4 (memw_reg) ✅ DEEPER:** the memw_reg walk inside `gpu_bitwise_hist_full` now reuses the resident
  `DeviceCpuOpsResident` buffers (packed/rv1/rv2/rvd) — only `next_pc` + tiny ecall arrays are uploaded
  (~96MB more upload eliminated). Verified under GPU_FULL.

**REMAINING for fully-resident (queued, FR4-page/op-vec / FR5 / FR6):** FR3 combine the 2-3 device histograms
into ONE shared hist buffer + one reduce/download (kill host add_raw_counts merges); FR4 page-resident
(scatter from device snapshot, ~9MB upload) + op-vec-resident (compute fields on-device from resident
`packed` instead of uploading derived arrays); FR5 block-privatized SMEM histogram scatter (the actual
histogram bottleneck: 46M atomics into 80MB); FR6 empty the CPU `p2a` collect for resident sources +
thread executor precompile recordings through `from_image_and_logs`/continuation (so GPU_FULL memory-drop
works off the monolithic path too — today the bench/continuation use `None` recordings). Measure ALL
speed with the WARM `gpu_resident_bench` (3 iters), never single cross-process runs.

---

## ⭐ SESSION 2026-07-21 — P4b DONE (no-go) + ⚠️ MEASUREMENT CORRECTION

New box: `ssh -p 48889 root@47.164.117.172` (RTX 5090, CUDA 13.1 at `/usr/local/cuda`, cargo 1.94).
Re-synced the local working tree onto it (base `fb204215` + diff, fixtures scp'd). All work still
working-tree-only (nothing committed).

**⚠️ CRITICAL MEASUREMENT CORRECTION (invalidates several earlier speed claims):** single `cargo test`
runs are SEPARATE PROCESSES → each pays a one-time cold cost (CUDA context init + PTX load + GPU clock
ramp) that lands in the first prove's `trace_build`. This session's early "OFF trace_build = 7.76s" was
that COLD artifact, NOT the real baseline. The reliable measure is the WARM in-process bench
(`gpu_resident_bench`, 3 iters). Warm truth on ethrex_5tx:
- resident-chips ON: trace_build ~4.1s · all-CPU (kill-switch): ~3.96s → **resident chips ~flat / ~0.15s
  SLOWER** (p5_generate 330ms vs 278ms — the device chip build loses to the multicore CPU).
- ⇒ **NO net speed win from GPU trace-gen warm.** This RE-CONFIRMS the handoff's long-standing truth:
  "partial GPU trace-gen ties/loses to the 8-core CPU; the win only appears when the WHOLE pipeline is
  resident (no uploads, empty CPU)." Earlier per-piece speed numbers this effort (incl. Option A's
  "p2a 643→495ms") were single-run/cross-process → cold-confounded → treat as NOISE, not wins. The
  CORRECTNESS results (everything verifies) are solid; the SPEED results are ~flat.

**P4b (bitwise histogram) — DONE, verdict NO-GO on speed (kept CPU):** the GPU bitwise assembly already
existed + VERIFIES (correctness/completeness ✅). But it's a net LOSS (~+0.6-0.8s at p4): the loss is
the GPU atomic-SCATTER (46M bumps into an 80MB×32-copy histogram, contention-bound), NOT uploads —
making memw_reg resident (`MEMW_REG_RESIDENT=1`, kills the ~150MB upload) barely moved it (2.56→2.45s).
Residency can't win; only a different histogram algorithm (block-privatized SMEM) could, for a ≤1.8s
ceiling that loses today. Decision: keep bitwise on CPU. `LAMBDA_VM_GPU_BITWISE` stays opt-in.

**"Make default" — attempted + REVERTED.** Flipped `gpu_resident_chips_enabled` default-on, warm-measured
it as ~flat/slight-loss (above), reverted. All GPU trace-gen stays OPT-IN (default = CPU path, which is
as fast or faster warm). No default change is warranted for speed.

**NET STATE:** the "everything on GPU" mandate is met on the COMPLETENESS axis — registers
(`LAMBDA_VM_GPU_REGISTERS=1`), memory tables (`LAMBDA_VM_GPU_MEMORY=1`), memory_state-replay DROP
(`LAMBDA_VM_GPU_MEMORY_DROP=1`), resident chips (`LAMBDA_VM_GPU_RESIDENT_CHIPS=1`), and the bitwise
histogram (`LAMBDA_VM_GPU_BITWISE=1`) ALL build on GPU + full-prove-VERIFY on ethrex_5tx. On the SPEED
axis they're ~flat vs the multicore CPU (warm), so they remain opt-in. The only path to an actual
trace_build win is the FULL resident pipeline (every source resident, CPU empty, one histogram) — a
large effort with a modest ceiling (trace_build ≈ p3to5 ~1.7s of a ~24s prove ≈ 7%). Open options: pursue
that full-resident pipeline, broaden e2e across guests, or treat the completeness milestone as the
deliverable. NOTE: re-measure ANY future speed claim with the WARM bench, never single cross-process runs.

**(Prior "box down" note from 2026-07-20 resolved — new box above.)** All changed files: `git status`
(under crypto/executor/prover) + §7 + the Option-A/B file lists below.

**To resume you NEED a GPU box** (RTX 5090 + CUDA ≥13). Ask the user for the new box's SSH connection
string (host/port). Then set it up per §2 (adapt the IP/port):
1. On the new box: get the repo at base `fb204215` (git fetch/clone the committed base).
2. Transfer the LOCAL working-tree diff. All changed/new files are under `crypto/ executor/ prover/`
   (see the full list in `git status` + §7 + the A1/A2 file lists below). Easiest: on the local
   machine `git diff fb204215 -- crypto executor prover > /tmp/wt.patch` for tracked changes, scp +
   `git apply` it, then scp the UNTRACKED new source files (`crypto/math-cuda/src/precompile.rs`,
   `prover/src/tests/gpu_*.rs` — the `??` entries in git status). Or scp each changed file (as done
   all session). md5-verify each after copy.
3. scp the gitignored fixtures: `executor/tests/ethrex_5tx.bin` + `executor/program_artifacts/rust/
   ethrex.elf` (no riscv toolchain on a fresh box).
4. Build/test env: §2 (adapt CUDA path, e.g. `/usr/local/cuda`). Sanity:
   `cargo test -p lambda-vm-prover --release --features cuda --lib gpu_resident_e2e -- --ignored --nocapture`.

**WHERE TO RESUME: P4b, step P4b.1 (NO P4b code written yet).** P4b is APPROVED (plan at
`~/.claude/plans/drifting-tumbling-falcon.md`, same machine; summary below). KEY finding from this
session's exploration: **the GPU-bitwise-histogram assembly ALREADY EXISTS** —
`math_cuda::bitwise_hist::gpu_bitwise_hist_opvec(&OpVecSources)` (all op-vec sources → ONE histogram →
ONE reduce) + `gpu_bitwise_hist_sources` (in_walk resident via `DeviceCpuOpsResident` + memw_reg + page),
both wired in `build_traces` p4 under `LAMBDA_VM_GPU_BITWISE=1`, merged via `add_raw_counts`, CPU keeps
only EC/precompile + mul/dvrm dedup tails. So P4b = **validate + make-resident + measure**, not build:
- **P4b.1 (start here):** build with instruments; run `gpu_resident_e2e` (a) default OFF, (b)
  `LAMBDA_VM_GPU_BITWISE=1 LAMBDA_VM_GPU_RESIDENT_CHIPS=1`. Does (b) VERIFY? capture `p4_bitwise` +
  `trace_build` + `prove_total` for both. If (b) doesn't verify → add a bin-for-bin diag (GPU-assembled
  histogram vs CPU baseline, all 10 lanes) to localize the gap (likely mul/dvrm signed dedup MSB16/ZERO
  tails: GPU does `*_perop`, CPU does `collect_bitwise_from_{mul,dvrm}_dedup` ~trace_builder.rs 2762/2796
  — confirm no gap/double-count) and fix.
- **P4b.2 (the win):** the current path LOSES to uploads (handoff §5: +141ms). Make the dominant
  uploads resident: **memw_reg** (~9.3M rows ≈150MB) via `gpu_memw_reg_hist_resident_ecall`
  (bitwise_hist.rs ~1109) / `LAMBDA_VM_GPU_MEMW_REG_RESIDENT` (coordinate with STEP-1 register walk, no
  double walk); **page** (~4.7M) fed from the A2/B1 device snapshot instead of uploaded host arrays.
  in_walk already resident. op-vec uploads: assess after.
- **P4b.3:** measure p4/trace_build off vs on (composed w/ `LAMBDA_VM_GPU_REGISTERS=1` +
  `LAMBDA_VM_GPU_MEMORY_DROP=1`), regression off, confirm net win or document residual.
- Kernels/wrappers all exist + registered (bitwise_hist.cu 17 scatter + `bitwise_hist_reduce`;
  device.rs 571-592). `HIST_COPIES=32`, NUM_ROWS=2^20, NUM_LOOKUP_TYPES=10 (Msb8=0 Msb16=1 Zero=2
  AreBytes=3 IsHalf=4 IsB20=5 Hwsl=6 ByteAlu{And,Or,Xor}=7/8/9). Full inventory: the 3 P4b Explore
  reports are in this session's transcript.

**STATUS RECAP (all validated on ethrex_5tx last time the box was up; opt-in flags, default unchanged,
nothing committed):** STEP 1 registers (`LAMBDA_VM_GPU_REGISTERS=1`) ✅ · Option B memory tables
(`LAMBDA_VM_GPU_MEMORY=1`) ✅ · Option A memory_state-replay DROP (`LAMBDA_VM_GPU_MEMORY_DROP=1`) ✅ ·
P4b bitwise-histogram = APPROVED, NOT STARTED.

---

## ⭐⭐⭐ SESSION 2026-07-20 (cont. 2) — OPTION A COMPLETE (A1+A2.1+A2.2+A2.3+A2.4) + e2e-VERIFIED

Pursuing Option A (drop the CPU `memory_state` replay entirely), staged A1→A2. Full drop confirmed with
the user; standalone speed win is modest (memory replay is cheap; `p2a_collect`'s bulk = regular op
collection = P4b/P5), so this is a completeness ("memory fully off CPU") milestone. Nothing committed.

**✅ A1 DONE + e2e-verified (always-on, byte-identical).** The executor records each precompile's INPUT
bytes; the prover's KECCAK/ECSM/COMMIT chip collectors build from the recordings instead of replaying
`memory_state` for inputs. `memory_state` STILL threaded (timeline/rows/final-state) — A1 only removes the
input-value coupling; it's the A2 prerequisite.
- Executor: `PrecompileInputs {keccak: Vec<[u64;25]>, ecsm: Vec<([u8;32],[u8;32])>, commit: Vec<Vec<u8>>}`
  on `Memory` + surfaced on `ExecutionResult` (executor/src/vm/{memory.rs,execution.rs}). Recorded at the
  ecall handlers (instruction/execution.rs): keccak input state, ecsm (xg,k), commit via `commit_public_output`.
- Prover: `collect_ops_from_cpu_inner(cpu_ops, Option<&PrecompileInputs>, ...)` (wrapper keeps the old
  4-arg sig → zero caller churn); `from_elf_and_logs_with_precompiles` + `from_image_and_logs_inner`
  wrappers pass it through. Only the monolithic prove path (lib.rs `prove_with_options_and_inputs`) passes
  `Some`; continuation/tests keep `None` (legacy memory_state path). Collectors: keccak input (loop),
  `collect_ecsm_ops` (recorded xg/k), `expand_commit_operations_for_ecall` (recorded bytes) use the
  recording when `Some`, else memory_state. Matched by in-order per-type cursor.
- Verified: `gpu_resident_e2e` full prove+verify byte-identical (default AND +GPU_MEMORY+GPU_REGISTERS);
  `cargo check --workspace` clean. Files: executor/src/vm/{execution.rs,memory.rs,instruction/execution.rs},
  prover/src/{lib.rs, tables/trace_builder.rs}. Local↔box md5-synced.

**🔬 A2 (in progress) — remove `memory_state` entirely (behind flag; needs GPU).**

**✅ A2.1 DONE (additive).** `EcallAccesses.mem_ops: Vec<EcallMemOp{base,width,is_read,flat_start}>` — per
ecall MEMORY op metadata (per-byte `mem_*` capture alone loses width grouping: commit=1, keccak/ecsm=8).
Populated in `capture_ecall_reg_accesses`. Compiles; nothing consumes it yet (A2.3 does).

**✅ A2.2 DONE + VALIDATED (device returns ecall old-state).** `memacc_emit_ecall` now records each ecall
byte's combined-stream position into a new `ecall_pos` array; new kernel `ecall_oldstate_gather`
(crypto/math-cuda/kernels/trace_walk.cu) reads back `old_ts`/`old_value` at those positions.
`gpu_build_memw_ls_resident_ecall` now returns a 9-tuple (…, `ecall_old_ts`, `ecall_old_val`) parallel to the
flat ecall_* inputs. Registered in device.rs (`ecall_oldstate_gather`). Callers updated (destructures use
`..`; `gpu_memw_ls_ecall_interleaved` uses `.0-.3` so unaffected). VALIDATED by new diag
`diag_ecall_oldstate_device_vs_cpu` (gpu_reg_emit_parity.rs): device ecall old_ts/old_val ==
CPU memory_state old-state BIT-EXACT (3775 ecall mem ops / 29080 bytes, 0 mismatches). Existing diags
(diag_memw_ls, diag_page_state, ecall_interleaved) still green. The device now exposes EVERYTHING the CPU
needs to drop memory_state.

**⭐ KEY DE-RISKING FINDING (verified in code):** the op-vec BITWISE sources — `store_ops`, `eq_ops`,
`bytewise_ops` (and branch/mul/dvrm) — are derived in `collect_all_ops` FROM `cpu_ops` (op operands
rv1/rv2/arg2/res), NOT from the memw rows or `memory_state`. So dropping `memory_state` does NOT break the
bitwise histogram. Only the MEMW rows + `collect_lt_from_memw*` + `collect_bitwise_from_memw_aligned` consume
the rows, and those rows come from the device (Option B regular + A2.2 ecall). **⇒ A2.3 is CONTAINED, not
P4b-entangled** (my initial fear was wrong).

**✅ A2.3 + A2.4 DONE + e2e-VERIFIED — `memory_state` REPLAY DROPPED (`LAMBDA_VM_GPU_MEMORY_DROP=1`).**
Option A is COMPLETE: with the flag, the CPU `memory_state` sequential replay is gone — all memory data
comes from the executor recordings (precompile inputs, A1) + the device walk (regular MEMW rows + snapshot,
Option B + ecall MEMW rows assembled post-walk from the device old-state, A2.2/A2.3). Full prove+verify
PASSES (drop-only AND composed with `LAMBDA_VM_GPU_REGISTERS=1`); default-off regression clean.
- `device_memory_drop_eligible` (`LAMBDA_VM_GPU_MEMORY_DROP=1`) implies `device_memory` (device walk runs).
- collect drop path (trace_builder.rs): regular loads → `build_load_op_and_bitwise` (LOAD op + bitwise
  only, no memw row, no memory_state); regular stores skipped; `push_ecall_memw_ops` skips ecall MEMORY
  rows (keeps ecall REGISTER rows). `memory_state` left init-only (regular replay gone; ecall collectors
  still run against it, cheap, their mem rows discarded).
- `build_device_memw_ls` assembles ecall MEMW rows from `ecall_accesses.mem_ops` (A2.1) + `mem_val` +
  device `ecall_old_ts`/`ecall_old_val` (A2.2), routes aligned/general via `classify_memw`; `build_traces`
  splices them (`d.ecall_aligned/ecall_general`) under drop. Final state = B1 device snapshot.
- **KEY BUG FOUND + FIXED:** `collect_commit_memw_ops` read the commit-buffer VALUE from `memory_state`
  (only commit does — keccak/ecsm take values from recordings/crypto), so under the init-only drop state
  the captured `mem_val` was wrong → first e2e failed. Fix: route the commit MEMW-row value from the
  executor recording too (`recorded: Option<&[u8]>`, like A1's `expand_commit`). Byte-identical for
  non-drop; correct for drop.
- Diags (gpu_reg_emit_parity.rs): `diag_ecall_oldstate_device_vs_cpu` extended to compare the FULL
  assembled ecall row (value/old/old_ts/width/is_read + count) vs CPU — 3775 ops/29080 bytes, 0 mismatches.
- **MEASURED** (instruments, ethrex_5tx): `p2a_collect_cpu` 643ms (off) → 495ms (drop), ~148ms saved by
  removing the CPU memory replay; but trace_build (~2.5s) + prove_total (~23s) are flat (device-walk GPU
  work offsets it). CONFIRMS: completeness win ("memory fully off CPU"), negligible net speed — as
  predicted. The big `p2a_collect` bulk (in_walk/register/chip op-vec building) is still CPU (that's P4b/P5).
- Files touched (A2.3/A2.4, working tree): `prover/src/tables/trace_builder.rs`,
  `prover/src/tests/gpu_reg_emit_parity.rs` (+ A2.2's crypto/math-cuda changes). Local↔box md5-synced.
- **NEXT (open): P4b/P5** — move the op-vec + bitwise-histogram building (the real `p2a_collect` bulk)
  resident on GPU to actually cut trace_build; or flip the memory/drop path on by default; or broaden e2e.

<details><summary>Original A2.3 design (kept for reference)</summary>

gate behind a flag (extend
`LAMBDA_VM_GPU_MEMORY` or new `..._FULL`). When on:
- Thread `Option<&mut MemoryState>` into `collect_ops_from_cpu_inner` + the 3 ecall collectors
  (keccak/commit/ecsm). `None` ⇒ don't read/write memory_state: ecall mem rows built with 0 old-state (only
  their value/base/width/is_read/ts are captured — capture already ignores old-state), NOT pushed via
  `push_ecall_memw_ops` (device provides them post-walk); regular loads/stores build only load_op + bitwise
  (both from the op, no memory_state), no memw row.
- POST-walk (in build_traces, extend the Option-B block): assemble ecall MEMW rows from `ecall_accesses`
  (`mem_ops` + `mem_val`) + device `ecall_old_ts`/`ecall_old_val` (value[b]=mem_val[flat_start+b],
  old[b]=ecall_old_val[…], old_timestamp[b]=ecall_old_ts[…]); splice into memw_aligned_ops/memw_ops (aligned
  vs general by `is_aligned` on the assembled row). Keep ecall REGISTER rows (register_state) in-loop.
- Drop `memory_state` from collect entirely; final state via B1 `from_image_and_snapshot` (already wired).
- A2.4: new diag (assembled ecall rows == CPU ecall rows bit-exact — reuses the A2.2 (base,ts) matching),
  full prove+verify with the flag (composed with registers), regression off, measure p2a_collect.

The device timeline is ALREADY validated (B1 snapshot + A2.2 old-state) → A2.3 surfaces/assembles existing
validated data, not a rebuild.
</details>

---

## ⭐⭐ SESSION 2026-07-20 (cont.) — OPTION B DONE + e2e-VERIFIED

**Option B (memory-side tables from the device walk, CPU replay KEPT) is COMPLETE + full-prove-verified**
on ethrex_5tx, behind the new opt-in flag `LAMBDA_VM_GPU_MEMORY=1` (default path untouched; composes with
`LAMBDA_VM_GPU_REGISTERS=1`; regression clean). This resolves the STEP 2 "completeness only" branch — it
does NOT drop the CPU walk (that is Option A, the executor re-architecture, still open). Nothing committed.

- **B1 — PAGE-FINI + ARE_BYTES from the device final-memory snapshot.** New `MemoryState::from_image_and_snapshot`
  (trace_builder.rs ~111) reconstructs the final state from `image + snapshot`; `build_traces` shadows
  `memory_state` with it for the 3 post-collect consumers (`generate_page_tables`, `build_page_bitwise_arrays`,
  `touched_cells_from_memory_state`). Validated: new diag `diag_page_state_device_vs_cpu` (gpu_reg_emit_parity.rs)
  — reconstructed state == CPU `memory_state` CELL-FOR-CELL both directions (cpu_cells=dev_cells=3079095, 0/0).
- **B2 — MEMW_A/MEMW regular rows from the device walk.** `build_device_memw_ls` (trace_builder.rs, analog of
  `build_device_memw_register`) runs `gpu_build_memw_ls_resident_ecall`, unpacks `pa`/`pg` via new
  `gpu_trace::{unpack_memw_aligned, unpack_memw}` (inverses of the pack fns; aligned drops old_ts[1..8] =
  don't-care, general fully lossless). `build_traces` then FILTERS OUT the CPU regular rows
  (`retain(is_register || ecall_ts)` — the exact `diag_memw_ls_device_vs_cpu` split) and splices the device rows
  in. Chose filter-and-replace over the planned collect null-sink: strictly safer (GPU-failure at runtime →
  keep CPU rows; no `from_logs`/no-image breakage) at the cost of the CPU building discardable regular rows
  (fine — B2 makes NO speed claim). Validated: e2e prove+verify (bus balance = the gate); device rows
  aligned=901428 general=16054 (== the validated diag counts).
- Gate: `device_memory_eligible` (trace_builder.rs, mirrors `device_registers_eligible`) reads
  `LAMBDA_VM_GPU_MEMORY=1`. Both `from_elf_and_logs` + `from_logs` thread `device_memory` to `build_traces`.
  A one-line opt-in `eprintln!("[gpu-memory] ...")` confirms the device path was taken (present with the flag,
  absent without) — remove before any commit if undesired.
- Files touched THIS sub-session (working tree only): `prover/src/tables/trace_builder.rs`,
  `prover/src/tables/gpu_trace.rs`, `prover/src/tests/gpu_reg_emit_parity.rs`. Local↔box md5-verified in sync.
- **NEXT**: Option B is a shippable completeness milestone. The remaining speed win is Option A (executor records
  the bytes each precompile reads → drop the CPU memory walk + p2a_collect). Also could: flip B on by default
  (drop the opt-in gate, keep `LAMBDA_VM_CPU_TRACE` kill-switch) + broaden e2e across guests.

---

## ⭐ LATEST SESSION (2026-07-20) — READ THIS FIRST

Branch now: `tracegen-gpu-full`. Box: `ssh -p 44902 root@79.116.18.158` (vast.ai RTX 5090; driver =
CUDA 13.0 max → build/test with `CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH`;
repo at `/workspace/lambda_vm`). Sync individual files via scp to `/tmp` then `cp` into place (full-tree
tar is blocked). Nothing committed. All work compiles (cuda + non-cuda); local↔box md5-verified in sync.

**Test cmd (any diagnostic/e2e):**
```
source $HOME/.cargo/env && export CUDA_HOME=/usr/local/cuda-13.0 PATH=$HOME/.cargo/bin:/usr/local/cuda-13.0/bin:$PATH \
  LAMBDA_VM_BENCH_ELF=$PWD/executor/program_artifacts/rust/ethrex.elf LAMBDA_VM_BENCH_INPUT=$PWD/executor/tests/ethrex_5tx.bin
cargo test -p lambda-vm-prover --release --features cuda --lib <name> -- --ignored --nocapture
```

### ✅ STEP 1 — register walk feeds MEMW_R, CPU register walk DROPPED — DONE + e2e-VERIFIED
`LAMBDA_VM_GPU_REGISTERS=1` on ethrex_5tx: full prove+verify PASSES (standalone, all-resident
chips+bitwise+registers, and default regression). Opt-in flag; default path intact.
- What: `build_device_memw_register` (trace_builder.rs ~1345) runs the device register walk (regular +
  interleaved ecall accesses), returns the routed MEMW_R RegRows + aligned/general fallbacks → feeds
  `memw_register_rows` → the SAME validated downstream (gen_memw_registers tables + `mrr` IS_HALF histogram)
  as the sequential walk. `device_registers_eligible` (~1302) gates on `LAMBDA_VM_GPU_REGISTERS=1` (now
  allows precompile runs). collect uses NullMemwSink for regular reg ops when device_registers.
- 5 nested bugs fixed (see progress-log memory); the LAST + subtlest: **x254 (commit index) is a WIDTH-1
  register access** (`old_timestamps=[ts,0]`), but the device walk treats every reg access as width-2 →
  `build_reg_fallback` made a width-2 MEMW_A row → bus imbalance. FIX: `capture_ecall_reg_accesses` +
  `push_ecall_memw_ops` keep width-1 register accesses on CPU (only width-2 go to the device walk).
- Debug method that cracked it: a full-field checksum of memw_register_rows + aligned + general BEFORE
  Phase 3, compared device_registers true vs false → isolated the diff to `aligned` (register + general
  identical) → per-field aligned diff (width/old_timestamp[1]) the 6-tuple diagnostics missed.
- Diagnostics (permanent regression tests, prover/src/tests/gpu_reg_emit_parity.rs):
  `diag_memw_reg_resident_vs_cpu_regular`, `diag_memw_reg_resident_vs_cpu_ecall`,
  `diag_memw_reg_device_rows_vs_cpu`.

### 🔬 STEP 2 — memory walk + snapshot: DEVICE BUILDING BLOCKS BUILT + VALIDATED; full drop BLOCKED
- **Device memory walk** (`gpu_build_memw_ls_resident_ecall`, trace_walk.rs ~1846) produces MEMW_A/MEMW
  rows bit-exact vs the CPU (validated by `diag_memw_ls_device_vs_cpu`: aligned=901428, general=16054,
  meaningful fields match — the ecall memory rows are non-emitting on device by design/stay CPU-side, and
  old bytes beyond access width are don't-care per store.toml, masked at byte granularity).
- **Final-memory snapshot on device** (NEW kernels `mem_final_flag` + `mem_final_gather`, trace_walk.cu
  after mem_link) — last access per address after the radix sort = final (value,ts). Function now returns
  a 7-tuple `(pa,na,pg,ng,snap_addr,snap_val,snap_ts)`. Validated: 294899 entries, 0 mismatches, 0
  touched-missing vs CPU MemoryState (`MemoryState::iter_cells`, #[cfg(test)]). This is what PAGE-FINI +
  ARE_BYTES need.
- ⚠️ **BLOCKER for dropping the CPU memory walk (step 3): precompile-input coupling.** The precompile
  collectors read INPUTS from `memory_state` MID-EXECUTION — `collect_keccak_memw_ops` reads the 200-byte
  keccak state (trace_builder.rs ~736), commit reads its buffer, ecsm reads point bytes, all at the ecall's
  ts. That needs the sequential replay; the device FINAL snapshot can't serve arbitrary mid-execution reads.
  Registers had no such coupling (inputs from the op-log). So the `memory_state` threading MUST stay for
  precompile inputs → moving the memory walk to GPU is a DOUBLE walk (no speed win) until precompile inputs
  are re-architected.

### NEXT STEPS (in priority order)
1. **Decision gate**: memory-side has modest speed upside (memory "walk" is cheap CPU threading, not a big
   sort like registers; trace-gen is ~9% of prove). Confirm whether to pursue memory completeness at all.
2. **If completeness wanted (no CPU-walk drop):** wire MEMW_A/MEMW + PAGE-FINI/ARE_BYTES from the device
   snapshot as a validated milestone, keeping the CPU `memory_state` replay (double walk). Reconstruct a
   MemoryState from `init_image + snapshot` (byte-identical to CPU by the validated parity) and feed the
   existing `generate_page_tables` + ARE_BYTES. Low risk (guaranteed-equal input).
3. **If the real win (drop p2a_collect) wanted:** re-architect precompile inputs — have the EXECUTOR record
   the memory bytes each precompile (keccak/commit/ecsm) reads, and feed those to the collectors instead of
   re-reading `memory_state`. Then the CPU memory replay is no longer needed and the device walk + snapshot
   fully replace it. This is a cross-crate (executor) change — the biggest remaining piece.
4. Ecall memory ROWS on device (optional, for a fully-resident MEMW): the per-byte ecall capture can't 1:1
   emit mixed-width ops (commit=1, keccak/ecsm lanes=8) — needs OP-metadata capture (base/width/is_read/
   value_word) + an emitting path, analog of the reg_access_*_ecall kernels.

---

## 1. TL;DR — where we are

**DONE + validated (byte/bin-identical to CPU, on RTX 5090):**
- **Every trace TABLE builds on the GPU** — 9 instruction chips (resident, single-upload seam +
  route-once), memory tables, and ALL precompiles (commit, keccak-main, keccak_rnd, ECDSA/ecdas,
  ECSM). Full prove+verify passes with everything on (`gpu_resident_e2e`, flag-on AND flag-off).
- **Resident-pipeline machinery** (the hard, uncertain part): device access emission (register +
  memory), BOTH walks resident (register + memory, no per-access upload), image-on-device lookup,
  and histogram-source kernels covering **~94% of bumps** (in_walk 62% + memw_reg 20% + page 10% +
  memw_aligned 2%).

**DONE this session (2026-07-17): ALL op-vec histogram kernels (P4a) built + GPU-validated**
bin-for-bin vs the REAL CPU collectors — lt/store/bytewise/eq/load/cpu32/branch/shift (full) +
mul/dvrm (per-op part; signed dedup tail deferred to P4b). See §4 P4a. → every BITWISE histogram
SOURCE now has a validated device kernel.

**NOT done (the remaining speed-win integration — multi-day):**
- Signed mul/dvrm MSB16/NEG-ZERO dedup tails (small; ride on the already-deduped MUL/DVRM table rows).
- Page-resident fini (needs final memory state on device; init via image_lookup exists).
- **P1-ecall accesses** — COMMIT/KECCAK/ECSM register+memory accesses through the walks. The
  CORRECTNESS gate for a full resident ethrex proof (memw_reg, memw_aligned, page-fini, lt-from-memw
  all need the complete ecall-inclusive streams). Flagged "small" but it is the real gate.
- **P4b**: assemble all sources into ONE resident histogram (extend `gpu_bitwise_hist_resident_upload`),
  wire into `build_traces` p4, empty the CPU, MEASURE (first speed win).
- **P5**: drop the host `p2a_collect` (the big trace-build drop).
- **P6**: make resident the default.

**Key measured truth:** partial GPU trace-gen LOSES to the 8-core CPU (proven ~6×) because of
uploads + CPU-pool imbalance. The win only appears when the WHOLE pipeline is resident (no uploads,
CPU histogram/collect empty). Trace-gen is only ~9% of the full prove, so even a full win moves total
prove time modestly — but it completes "everything on GPU."

---

## 2. Box + build/test setup (vast.ai RTX 5090)

```
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o IdentityAgent=none \
    -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 -p 42011 root@159.48.242.15
```
- Repo: **/workspace/lambda_vm**. CUDA 13.x, `CUDA_HOME=/usr/local/cuda`.
- Env for every build/test:
  ```
  source $HOME/.cargo/env
  export CUDA_HOME=/usr/local/cuda PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH \
    LAMBDA_VM_BENCH_ELF=$PWD/executor/program_artifacts/rust/ethrex.elf \
    LAMBDA_VM_BENCH_INPUT=$PWD/executor/tests/ethrex_5tx.bin
  ```
- **Sync** (local→box): `scp -P 42011 <files> root@159.48.242.15:/tmp/` then `cp /tmp/<f> <repo path>`
  on the box. (Full-tree tar is BLOCKED by an exfil classifier; scp individual changed files.)
- **Fixtures** on the box already: `ethrex.elf` + `ethrex_5tx.bin` (gitignored; box has no riscv
  toolchain). If a fresh box: `git fetch` the committed base + `git apply` local diff + scp fixtures.
- **Run a GPU test:** `cargo test -p lambda-vm-prover --release --features cuda --lib <filter> -- --ignored --nocapture`
  (many are `#[ignore]` + need the bench env). Timing tests add `,instruments` to features.
- ALWAYS `--nocapture`: without it, "N passed" can be N *skips* ("no CUDA backend").

**Flags** (all opt-in; default build unchanged, no regression):
- `LAMBDA_VM_GPU_RESIDENT_CHIPS=1` — 9 resident chips (single-build seam) + all precompile tables.
- `LAMBDA_VM_GPU_BITWISE=1` — GPU bitwise histogram (in_walk+memw_reg+page via upload; correct but
  measured slower — see §6).
- `LAMBDA_VM_CPU_TRACE=1` — kill-switch (all-CPU trace-gen).

---

## 3. Everything validated this effort (file : symbol : test)

### Trace TABLES on GPU
- **9 resident chips** (CPU32/LOAD/STORE/SHIFT per-row; EQ/BYTEWISE/MUL/DVRM/BRANCH deduped):
  - `crypto/math-cuda/src/trace_ops.rs`: `DeviceCpuOpsResident` (holds packed/imm/pc/rv1/rv2/arg2/
    res/rvd/flags + `routes: DeviceChipRoutes`), `gpu_upload_cpu_ops_resident`, `compute_chip_routes`
    (route-once: all flag arrays in one pass), and `gpu_build_<chip>_resident_from_devops` for all 9.
  - `prover/src/tables/gpu_trace.rs`: `build_shared_devops`, `build_<chip>_resident_tables_from_devops`,
    `devops_table` helper.
  - `prover/src/tables/trace_builder.rs` p5: builds `shared_devops` once before the rayon scope;
    each chip closure reads it. (DeviceCpuOpsResident is Send+Sync → shared by ref across the scope.)
  - Tests: `gpu_resident_seam_parity` (all 9 from one upload), `gpu_cpu32_pipeline_parity`,
    `gpu_dedup2_resident_parity`, `gpu_resident_e2e` (full prove+verify, flag-on + off).
- **Precompile tables** (formatting fills — NO new modular/EC CUDA needed; the crypto runs in the
  executor, trace-gen just formats the witness):
  - `crypto/math-cuda/kernels/trace_cpu.cu`: `commit_fill`(19 cols), `keccak_table_fill`(511),
    `ecdas_fill`(521), `ecsm_fill`(667), `keccak_rnd_fill`(1480, 24 rows/op, 1 thread/op, block=32).
  - `crypto/math-cuda/kernels/keccak.cu`: `keccak_f1600_batch` (reuses the Merkle-tree `keccak_f1600`).
  - `crypto/math-cuda/src/precompile.rs` (NEW module): `gpu_build_{commit,keccak,ecdas,ecsm,keccak_rnd}_trace[_dev]`,
    `gpu_keccak_f1600_batch`. Strides: commit 7, keccak 52, ecdas bytes326/carries192-i64, ecsm
    bytes354/carries128/addrs4, keccak_rnd 26.
  - `prover/src/tables/gpu_trace.rs`: `build_{commit,keccak,ecdas,ecsm,keccak_rnd}_resident_table`;
    wired into p5 `gen_{commit,keccak,ecdas,ecsm,keccak_rnd}`.
  - Tests: `gpu_fill_tests::{gpu_commit_fill,gpu_keccak_table_fill,gpu_ecdas_fill,gpu_ecsm_fill,
    gpu_keccak_rnd_fill}_matches_cpu`, `gpu_keccak_f1600_parity`.

### Resident pipeline (P1–P3 + image)
- **P1 register emission**: `trace_walk.cu::{reg_access_counts,reg_access_scatter}`;
  `trace_walk.rs::{gpu_emit_register_accesses, emit_register_accesses_dev(device core)}`. Emits M1
  rs1@ts / M3 rs2@ts+1 / M5 rd@ts+2 (flag&&reg!=0) + implicit PC write@ts+1 (reg 510, row_index=-1).
  Test `gpu_reg_emit_matches_cpu` (13.4M accesses, 9.3M emitting).
- **P1 memory emission**: `trace_walk.cu::{memacc_counts,memacc_emit}`;
  `trace_walk.rs::{gpu_emit_memory_accesses→DeviceMemAccesses}`. Per load/store op: `width`
  byte-accesses + op metadata. Test `gpu_mem_emit_matches_cpu` (6.0M byte-accesses).
- **Image-on-device**: `trace_walk.cu::image_lookup` (binary search sorted image);
  `trace_walk.rs::{image_lookup_dev(core), gpu_image_lookup}`. Test `gpu_image_lookup_matches_cpu`.
- **P2 register walk resident**: `trace_cpu.rs::gpu_walk_fill_memw_register_resident[_host]`
  (emit_register_accesses_dev → walk_core → memw_register_fill, no per-access upload). Test
  `gpu_reg_walk_fill_resident_matches_host` (9.3M MEMW_R rows).
- **P2 memory walk resident**: `trace_walk.rs::gpu_build_memw_ls_resident` (emit accesses →
  image_lookup_dev init → radix walk → gather → classify → pack MEMW_A/MEMW). Test
  `gpu_memw_ls_resident_matches_host` (901K MEMW_A + 16K MEMW).
- **P3 memw_reg histogram (from resident walk, no upload)**: `bitwise_hist.cu::bitwise_hist_memw_reg_masked`
  (IS_HALF per emitting row, skips row_index<0); `bitwise_hist.rs::gpu_bitwise_hist_memw_reg_masked`.
  Test `gpu_bitwise_memw_reg_masked_matches_cpu`.

### Histogram source kernels (94% of bumps)
- in_walk: `bitwise_hist.cu::bitwise_hist_cpu_ops_packed` (unpacks decode from `packed`+reads `res`);
  `bitwise_hist.rs::gpu_bitwise_hist_in_walk_devbuf`. Test `gpu_bitwise_in_walk_resident_matches_host_soa`.
- page: `bitwise_hist.cu::bitwise_hist_page` (ARE_BYTES[init,fini]/byte); `gpu_bitwise_hist_page_only`.
  Test `gpu_bitwise_page_scatter_matches_cpu`.
- memw_aligned: `bitwise_hist.cu::bitwise_hist_memw_aligned` (IS_HALF[base_low+mask]/aligned op);
  `gpu_bitwise_hist_memw_aligned`. Test `gpu_bitwise_memw_aligned_matches_cpu`.
- In-build wiring: `trace_builder.rs::gpu_bitwise_hist_sources` → `bitwise_hist.rs::
  gpu_bitwise_hist_resident_upload` (in_walk from packed+res + memw_reg + page); p4 routes page to
  device + skips it on CPU. (Correct, but a measured LOSS via upload — see §6.)

---

## 4. Exact remaining work (P4 tail + P5 + P6)

### P4a — op-vec histogram kernels ✅ DONE + GPU-VALIDATED (2026-07-17, box 42011)
ALL op-vec histogram source decompositions now exist as device kernels, each BIN-FOR-BIN identical
to the REAL CPU collector (tests build synthetic ops → call the actual `collect_*` → scatter the
returned `BitwiseOperation`s via `lookup_type_index`/`row_index` → compare the full counter array).
Kernels in `bitwise_hist.cu`, wrappers `math_cuda::bitwise_hist::gpu_bitwise_hist_<src>` (share the
`run_source` scaffold), loaders in `device.rs`, tests in `prover/src/tests/gpu_bitwise_opvec_parity.rs`
(filter `gpu_bitwise_opvec`, `--ignored --nocapture`, 10 tests all green):
- **lt** `bitwise_hist_lt(lhs,rhs)` — 2 Msb16 + 6 IS_HALF/op. (200k→1.6M bumps)
- **store** `bitwise_hist_store(value)` — 8 ARE_BYTES/op. **bytewise** `bitwise_hist_bytewise(a,b,op)`
  — 8 BYTE_ALU[7+op]/op. **eq** `bitwise_hist_eq(a,b)` — 4 IS_HALF + 1 ZERO[Σ]/op.
- **load** `bitwise_hist_load(res[8/op],width)` — 1 Msb8/op (skip width 8). **cpu32**
  `bitwise_hist_cpu32(hil,alu_flags,rs1,rs2,rd,rv1,rv2,res)` — 5 ARE_BYTES + 8 IS_HALF + 1 ByteAluAnd
  + (signed?2)+1 Msb16 (signed = alu_flags bit5). **branch** `bitwise_hist_branch(next_pc,next_pc_unmasked)`
  — ARE_BYTES + ByteAluAnd + 3 IS_HALF/op.
- **shift** `bitwise_hist_shift(value,shift,shift_amount,flags)` — recomputes compute_aux
  (is_negative/bit_shift/zbs, bit-for-bit from `shift_fill`) then the full decomposition incl 5×HWSL.
  (200k→3.0M bumps.)
- **mul** `bitwise_hist_mul_perop(lhs,rhs,flags)` — PER-OP part: 16 IS_HALF + 4 IS_B20 (reuses the
  `mul_fill` 128-bit product/raw math + `compute_carries`). **dvrm** `bitwise_hist_dvrm_perop(n,d,flags)`
  — PER-OP part: 20 IS_HALF + 2 ZERO (reuses `cpu32_dvrm` div/rem). Validated with UNSIGNED ops (where
  full collector == per-op). **DEFERRED to P4b**: the signed per-chunk MSB16 (mul+dvrm) + NEG-template
  ZERO (dvrm) — chunk-coupled dedup; ride on the deduped MUL/DVRM table rows (already device-deduped
  via gpu_dedup2) with the chunk-boundary "sent twice if spanning two instances" handling.
Bump→bin recap: type lanes Msb8=0 Msb16=1 Zero=2 AreBytes=3 IsHalf=4 IsB20=5 Hwsl=6 ByteAluAnd/Or/Xor=7/8/9;
`hist[copy_base + type*num_rows + row_index]`, row_index(x,y,z)=x+y*256+z*65536, halfword h→h,
byte b→b, byte_op(a,b)→a+b*256, HWSL z=bit_shift, ZERO value→value(<2^20).

**NOTE on lt/memw coupling (unchanged gate):** the KERNEL is validated standalone on any lt op-stream.
Wiring the FULL lt histogram source in P4b still needs the complete lt op-vector = instruction ⊕ dvrm→lt
⊕ memw→lt(+aligned), and memw→lt includes ecall-generated memory ops → gated on P1-ecall (same gate as
the LT chip table + the correct full proof).

### P4b — assemble + wire into build_traces + MEASURE (the payoff)
- Assemble ALL sources into ONE replicated histogram (scatter each into the shared `hist` copies,
  reduce once) — accumulation is trivially correct; validate the total bin-for-bin.
- Wire into `build_traces` (prover/src/tables/trace_builder.rs, p4 region ~line 3560+): run the
  RESIDENT register walk (for memw_reg, no upload — currently the build runs the CPU walk) + build
  `shared_devops` before p4 (for in_walk) + page-resident, and set `cpu_bitwise_sources`/collectors
  so the **CPU histogram is EMPTY** (only ~8k EC bumps stay). Then MEASURE p4 (bench
  `gpu_resident_bench`, `--features cuda,instruments`, flag-on vs off). Baseline p4 ≈ 460ms.
- Page-resident: needs init (image_lookup — have it) + fini (final memory state on device — NOT built;
  either upload the final-state map sorted + binary search, or derive from the walk's last-write).

### P5 — drop host p2a_collect
Once chip tables + walks + histogram all read resident data, `collect_ops_from_cpu` is redundant for
them → remove it (keep ecall-access prep). Reclaims ~592ms. Measure trace_build (bench above).

### P6 — default + broaden
Flip resident on by default (drop the opt-in gate; keep `LAMBDA_VM_CPU_TRACE` kill-switch); broaden
e2e across guests; final measurement.

---

## 5. Key numbers (ethrex_5tx, 4.04M cycles, RTX 5090, best-of-3)
- trace_build ≈ 1750ms. Breakdown: p2a_collect 592ms, p4_bitwise 460ms, p5_generate 215ms,
  p1_cpu_ops 166ms, p0_decode 29ms, p2b 100ms. Trace-gen ≈ 9% of full prove.
- Histogram bumps ≈ 46M: in_walk 28.5M(62%), memw_reg 9.3M(20%), page 4.7M(10%), op-vec ~3.4M(8%),
  EC/precompile ~8k.
- Resident chips ON (route-once): 1818ms vs 1768 OFF (+50ms, mid-transition loss — host collect
  still runs). GPU bitwise ON (in_walk+memw_reg+page via upload): p4 630 vs 489 (+141ms — uploads
  dominate). ⇒ the win needs NO uploads + empty CPU (the resident pipeline).

---

## 6. Gotchas / conventions (so you don't re-derive them)
- `ts = i*4+4` (cpu_op timestamp = cycle index). `reg_addr = 2*reg`; PC register = 510 (PC_WORD_ADDR).
- packed_decode bits: READ_REG1=0, READ_REG2=1, WRITE_REG=2, WORD_INSTR=3, ALU=4, ADD=5, SUB=6,
  MEMORY=7, BRANCH=8, ECALL=9, RS1=10, RS2=18, RD=26, HIL=34, ALU_FLAGS=42, MEM_FLAGS=50 (all
  byte-wide except the 1-bit flags). mem_flags: bit0=store, bit1=signed, bits2/3/4=2B/4B/8B (else 1B);
  value = rvd(load)/rv2(store). alu_op = alu_flags & 0x1F (AND0 OR1 XOR2 EQ3 LT4 SHIFT5 SHIFTW6 MUL7
  DIVREM8).
- `excl_scan` (trace_walk.rs) SUMS arbitrary u32 (not just 0/1) → usable for variable per-op counts.
- Register-heavy kernels overflow the default 1024-block → CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES; use a
  manual small `LaunchConfig` (keccak_f1600 block=64; keccak_rnd block=32).
- `fe_from_i64(c)`: c>=0 → c; c<0 → GOLD_P - |c| (GOLD_P=0xFFFFFFFF00000001). Kernels use `ec_fe`.
- FE→u64 in parity tests: `unsafe { *(e.value() as *const u64) }` over `table.main_data_row_major()`.
- Deduped chips: compare as MULTISET (device output is radix-sorted vs the CPU HashMap order; both are
  valid — LogUp buses are permutation-invariant). Per-row/fill tables: byte-identical.
- Old `build_X_resident_tables` (non-devops) are kept (used by `gpu_cpu32_pipeline_parity`) → they
  show dead-code warnings in the release lib; harmless (no deny(warnings)).
- Precompile EC/keccak math runs in the EXECUTOR (CPU) during `run()`; trace-gen only FORMATS the
  witness → NO 256-bit/secp256k1 CUDA needed (a key finding — don't rebuild the crypto).

---

## 7. Files changed this effort (user commits; assistant never does)
Modified: `crypto/math-cuda/{Cargo.toml, build.rs, kernels/bitwise_hist.cu, kernels/keccak.cu,
kernels/trace_cpu.cu, kernels/trace_ops.cu, kernels/trace_walk.cu, src/bitwise_hist.rs, src/device.rs,
src/lib.rs, src/trace_cpu.rs, src/trace_ops.rs, src/trace_walk.rs}`, `prover/src/paged_mem.rs`,
`prover/src/tables/{bitwise.rs, gpu_trace.rs, page.rs, trace_builder.rs}`, `prover/src/tests/mod.rs`.
New: `crypto/math-cuda/src/precompile.rs`; prover tests `gpu_{cpu_ops,alu_chipops,cpu32,load,
mem_walk,memw_routing,memw_fill,dedup,dedup2_resident,lt_resident,cpu32_pipeline,resident_chips,
resident_seam,resident_e2e,resident_bench,bitwise_resident,bitwise_opvec,keccak_f1600,reg_emit}_parity.rs` +
`gpu_fill_tests.rs` (extended). Fixtures (gitignored): `executor/tests/ethrex_5tx.bin`.
NOTE (2026-07-17): `collect_bitwise_from_{lt,branch}` + `collect_cpu32_bitwise` made `pub(crate)` for
the opvec parity tests (no behavior change).

---

## 8. How to resume next session
1. Give the assistant this file + `reports/tracegen/RESIDENT-PIPELINE-PLAN.md`.
2. Confirm box 42011 is up (`nvidia-smi`); if down, get a new vast.ai RTX 5090 + CUDA≥13, sync per §2.
3. P4a (all op-vec kernels) is DONE + validated (§4). Next is **P4b + P1-ecall** — the large
   integration with the correctness gate. Recommended order:
   a. **P1-ecall** first (the correctness gate) — FULLY DESIGNED + MAPPED, see
      `reports/tracegen/P1-ECALL-PLAN.md` (Option Z: inject ecall accesses into the resident walks as
      non-emitting timeline events, keep the tiny ecall rows on CPU; architecture map + 4 GPU-validated
      increments + gotchas incl. the dual register-row-path subtlety). Emit COMMIT/KECCAK/ECSM
      register+memory accesses into the resident walk streams so the walks are COMPLETE on ethrex.
      Without this, memw_reg / memw_aligned / page-fini / lt-from-memw undercount → invalid proof.
   b. **P4b assembly**: extend `gpu_bitwise_hist_resident_upload` (or a new "assemble-all") to scatter
      every source (in_walk devbuf + memw_reg-masked-from-walk + page init/fini + memw_aligned + the 10
      op-vec kernels + the mul/dvrm signed dedup tails from the deduped tables) into ONE replicated
      histogram, reduce once, validate the TOTAL bin-for-bin vs the CPU histogram on ethrex_5tx.
   c. Wire into `build_traces` p4: run the resident register walk + `shared_devops` before p4, set
      `cpu_bitwise_sources` so the CPU histogram is EMPTY (only ~8k EC bumps). Full prove+verify + MEASURE.
   d. Then **P5** (drop host p2a_collect) and **P6** (default). Do each with e2e + bench validation.
   The op-vec kernels validated this session are the building blocks for (b). All 10 pass via filter
   `gpu_bitwise_opvec` (`--ignored --nocapture`, needs the bench env).
