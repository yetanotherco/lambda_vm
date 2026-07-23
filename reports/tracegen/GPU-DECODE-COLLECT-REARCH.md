# ==========================================================================
# 🟢 RESUME HERE — HANDOFF for the decode+collect→GPU work (fresh session, 2026-07-22)
# ==========================================================================
# This is the LAST big host chunk of trace-generation. Everything else (all chip tables incl. LT,
# register+memory walks, the whole bitwise histogram) is already on GPU and full-prove-VERIFIES.
# READ THIS BLOCK FIRST, then the staged plan below.

## Mandate / scope (from the user)
Move the **decode+collect cluster** of TRACE-GENERATION onto the GPU. The user cares ONLY about
trace-generation (NOT the rest of the prover). Prover-only, validated on ethrex_5tx. Completeness first,
speed second. NEVER commit/push unless told. NEVER cache build steps. ALWAYS test on ethrex_5tx.

## Box (vast.ai, EPHEMERAL — may be gone; if so get a new RTX 5090 + CUDA≥13 & re-sync per HANDOFF.md "BOX")
`ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o IdentityAgent=none -o IdentitiesOnly=yes
-i ~/.ssh/id_ed25519 -p 63154 root@174.31.67.221` · repo `/workspace/lambda_vm` · `/tmp/runenv.sh` set ·
base fb204215 + working-tree overlay (all md5-synced). Build FAST (~10-50s, deps cached). Re-sync recipe in
`HANDOFF.md` "BOX" section (rsync ONE connection — per-file scp of ~58 files times out; ethrex.elf gitignored
→ rsync explicitly). e2e: `LAMBDA_VM_GPU_FULL=1 cargo test -p lambda-vm-prover --release --features
cuda,instruments --lib gpu_resident_e2e -- --ignored --nocapture`. Full-prove TIMELINE:
`LAMBDA_VM_GPU_FULL=1 ./target/release/cli prove $ELF --output /tmp/p.bin --private-input $INPUT --time`
(build with `--features "instruments,prover/cuda"`).

## MEASURED baseline (ethrex_5tx, THIS box, branch GPU_FULL, full-prove TIMELINE)
trace_build = **1.97s** (17.6% of an ~11.2s prove). Inside it, the cluster to move:
`p0_decode 47ms + p1_cpu_ops 139ms + p2a_collect_cpu 563ms + p2b_collect_all 101ms ≈ 850ms` (HOST, SERIAL —
the GPU is idle during this). `p3to5_build_traces 976ms` is already GPU (tables + histogram). (main = CPU
trace-gen here = ~1.97s trace_build too; this box has a STRONG CPU, so the GPU trace-gen edge is modest — but
the cluster is ~40% of the trace stage and runs serial with the GPU idle, so removing it is a real stage win.)

## WILL GPU BE FASTER THAN CPU FOR THIS? (my assessment — likely YES, with one hard slice)
The ~850ms splits three ways:
1. **decode + per-cycle `from_log` (p0/p1 + part of p2a)** — embarrassingly parallel (4M independent cycles).
   The device builder **ALREADY EXISTS + is VALIDATED bit-exact**: `math_cuda::trace_ops::
   gpu_build_cpu_ops_resident` (trace_ops.rs) + parity `gpu_cpu_ops_parity.rs` (4.03M cycles). It is NOT
   wired into the hot path — today `build_shared_devops` UPLOADS the host-built cpu_ops SoA
   (`gpu_upload_cpu_ops_resident`) instead of recomputing on device. Wiring it → this drops to ~tens of ms.
2. **op-vector building (p2b + part of p2a)** — SURPRISE: much of this is now **DEAD under GPU_FULL**. The
   resident chip tables read the device SoA (`devops`), and the histogram op-vec sources are device-computed
   (this session). So the host op-vecs (lt/shift/mul/dvrm/store/eq/bytewise/branch/load/cpu32) are largely
   UNUSED under gpu_full — like `lt_ops` already is. They can be SKIPPED (flag-gate, like `skip_bitwise`),
   not ported. EXCEPTION: `mul_ops`/`dvrm_ops` still feed the chunk-deduped MSB16 histogram TAIL
   (`collect_bitwise_from_{mul,dvrm}_dedup`) on host — keep those or move the tail too.
3. **register_state advance + ecall/precompile assembly (part of p2a, ~the hard core)** — the A3
   entanglement: the 4M-op `register_state` advance feeds (a) HALT/final state and (b) per-ecall register
   indices (commit x254). The ecall ops (COMMIT/KECCAK/ECSM) are assembled from the executor RECORDINGS. This
   slice is sequential-ish + recording-tied → hardest to move, may stay partly host. It's the risk.
⇒ Expect a REAL stage win (the parallel bulk + skipping dead work), with the register/ecall residual as the
hard limiter. Not a slam-dunk 10× — but should be faster than the serial CPU cluster.

## CONCRETE NEXT STEPS (staged, each e2e-gated + bit-exact parity where it computes)
- **A. Wire S1 (device cpu_ops seam).** In `build_shared_devops` (gpu_trace.rs), under gpu_full, build the
  resident cpu_ops via `gpu_build_cpu_ops_resident` (from the log SoA: cpc/npc/s1/s2/dv + per-log pc/imm/
  packed) INSTEAD of `gpu_upload_cpu_ops_resident` (host cpu_ops). Kills p1's host build + the SoA upload.
  Gate: e2e verify (the device SoA is already parity-validated). Measure trace_build.
- **B. Skip the dead host op-vec building under gpu_full.** Thread a `skip_opvec` flag (like `skip_bitwise`)
  through `collect_ops_from_cpu_inner` + `collect_all_ops` to NOT build the op-vecs the resident tables /
  device histogram no longer consume (lt already done; add store/eq/bytewise/branch/load/shift/cpu32; keep
  mul/dvrm for the dedup tail unless that's moved too). Verify each source is truly dead first (grep its
  consumers under gpu_full). Kills most of p2b + part of p2a. Gate: e2e.
- **C. The residual (register state + ecall).** Hardest. The register WALK is already on GPU (MEMW_R); the
  register_state STRUCT (for final state + ecall indices) is the host piece. Options: keep it (small), or
  derive final state + per-ecall indices from the device walk (A3). Ecall/precompile op assembly from
  recordings is tiny — likely stays host. Measure what's left after A+B before deciding how hard to push C.
- **D. Measure + broaden** (warm bench + full-prove TIMELINE; e2e on ethrex_5tx). Target: cluster ~850ms → a
  few hundred ms.

## Current state to build on (session 5, all e2e-VERIFIED, nothing committed)
All chip tables resident on GPU (incl. LT — STEP 2A/2B done this session), all 7 histogram op-vec sources
device-computed, memw→lt derived on device. So the collectors' op-vec OUTPUTS are already reproduced on
device — step B is mostly "stop building the dead host copies", not "port". Cleanups pending before any
commit: temp `[p4-probe]`/`[p3-probe]` eprintlns; the `#if 0` legacy `bitwise_hist_mul_perop_OLD` block in
bitwise_hist.cu; the dead host `lt_ops` assembly. See `reports/tracegen/HANDOFF.md` "SESSION 5" for the full
LT/histogram detail + file list.

# ==========================================================================
# (original plan below — S1–S5 staging + the coupling analysis)
# ==========================================================================

# GPU decode+collect re-architecture — plan (multi-session)

Goal (user-committed 2026-07-21 s3): move the entire **serial decode+collect cluster** — `p1 collect_cpu_ops`
(~421ms) + `p2a collect_ops_from_cpu_inner` (~566ms) + `p2b collect_all_ops` (~161ms) ≈ **1.15s serial** —
onto the GPU as a unit. This is the last large chunk of trace-gen still on CPU (after this session got
`GPU_FULL` to −17% vs the 8-core CPU). It's soundness-sensitive (`cpu_ops` feeds the whole proof) and modest
payoff (trace-gen ≈ 9% of a ~24s prove), so stage it with a bit-exact parity gate at every step.

## Why it's coupled (the thing that makes it a "unit", not a table fill)
`cpu_ops: Vec<CpuOperation>` is produced by p1 and consumed by BOTH:
- the host collectors (p2a/p2b) → op-vectors (`lt_ops`, `store_ops`, `eq_ops`, `bytewise_ops`, `branch_ops`,
  `mul_ops`, `dvrm_ops`, `load_ops`, `shift_ops`, `cpu32_ops`) + ecall accesses + precompile ops, AND
- `build_shared_devops(&cpu_ops)` → the device SoA the resident chips/histogram/walks already read.
So moving p1 alone just adds a ~300MB `cpu_ops` download (host collect still needs it). The win only lands
when p1+p2a+p2b move TOGETHER: build `cpu_ops` on device, derive the op-vectors on device, and leave only a
tiny host residual (the things that genuinely can't be device-side).

## The pure function to port: `CpuOperation::from_log_and_instruction` (prover/src/tables/cpu.rs:380)
`= DecodeEntry::from_instruction(pc, instruction, 4)` then `from_log(log, ts, decode)` (cpu.rs:~356). All
PURE arithmetic over `(log, decode.fields, decode.imm, decode.pc)`, GPU-portable:
- `Log` (executor/src/vm/logs.rs:15) = 5×u64: `current_pc, next_pc, src1_val, src2_val, dst_val` (~160MB/4M).
- decode fields (packed): rs1/rs2/rd/hil/alu_flags/mem_flags + the bit flags (read_register1/2, write_register,
  word_instr, alu, add, sub, memory, branch, ecall). Layout = `packed_decode_shrunk` (types.rs:512+), the SAME
  `packed` the bitwise/walk kernels already unpack.
- `from_log` computes: `rv1` (x255→pc, else src1_val if read_register1), `rv2` (src2_val if read_register2),
  `arg2` (MEMORY→imm / BRANCH→rv2 / else rv2+imm), `res` (add→rv1+arg2 / sub→rv1-arg2 / alu-branch→cond /
  alu→dst_val / …), `branch_cond` (jalr→true / `branch_taken(f,rv1,rv2)`), `next_pc`, and the ecall flags
  (`ecall_commit` = ecall∧src1==Commit; `commit_buf_addr/count` from src2/dst; keccak/ecsm syscall numbers).
- `CpuOperation` (cpu.rs:158) fields: decode(packed), timestamp=i*4+4, next_pc, rvd(=dst_val), rv1, rv2,
  arg2, res, branch_cond, ecall_commit, commit_buf_addr, commit_count, ecall_keccak, keccak_state_addr,
  ecall_ecsm, … Most map straight to the `shared_devops` SoA (packed/imm/pc/rv1/rv2/arg2/res/rvd/flags).

## ⭐ STATUS UPDATE (2026-07-21 s3): S1 is ALREADY DONE + VALIDATED (built by the original resident-pipeline
effort). `math_cuda::trace_ops::build_cpu_ops` (kernel, trace_ops.cu:60) is a bit-for-bit `from_log` port;
`gpu_build_cpu_ops_resident` (trace_ops.rs:229) produces the FULL `DeviceCpuOpsResident` (packed/imm/pc/rv1/
rv2/arg2/res/rvd/flags + routes) on device from logs + per-log (pc,imm,packed); parity test
`gpu_cpu_ops_parity.rs::gpu_build_cpu_ops_matches_from_log` **PASSES bit-exact over 4,036,972 cycles** (re-ran
green this session). The device ALU chip-op derivation (LT/SHIFT/EQ/BYTEWISE/MUL/DVRM from the SoA) also
exists (trace_ops.cu:137+). ⇒ the FOUNDATION is built. The remaining work is S2→S4 wiring, and the CRUX is
**S3/S4** (op-vectors on device + drop the host `collect_*`) — S2 alone is marginal (cpu_ops still built on
host for the collectors, so wiring `shared_devops` from logs just trades the 9-array pack+upload ~150ms for a
redundant device `from_log`; the real 421+566+161ms drop needs S3/S4). Next real step = S3/S4, best fresh.

## Staging (each step independently GREEN + bit-exact parity gated)
- **S1 (FOUNDATION) — ✅ DONE + validated (see status above).** device `cpu_ops` SoA from logs+decode via
  (per log: binary-search a sorted `(pc→packed_decode, imm)` table for the decode, run the `from_log` math,
  write the `shared_devops` SoA `packed/rv1/rv2/arg2/res/rvd/next_pc` + the ecall metadata arrays). Upload
  logs (160MB) + the decode table (small, per-unique-pc). Validate BIT-EXACT vs host `collect_cpu_ops` →
  `build_shared_devops` (a parity test comparing every SoA field). NO wiring yet — proves the device build
  matches. This is the risk-retirement step; do it first and thoroughly (the branch logic in `from_log` is
  the correctness surface).
- **S2 — wire `shared_devops` from the device build.** `build_shared_devops` uses the device-built SoA
  (from S1) instead of uploading host `cpu_ops`. Kills the shared_devops H2D (~150ms). e2e verify. `cpu_ops`
  still built on host for the collectors (download or keep host build in parallel) — measure whether to keep
  the host build or download the device SoA.
- **S3 — op-vectors on device.** Derive lt/store/eq/bytewise/branch/load/cpu32 + mul/dvrm PER-OP from the
  device SoA (extends FR4b's `bitwise_hist_opvec_packed`; the histogram already consumes them). The DERIVED
  contributions (dvrm→lt/mul, cpu32→shift/mul/dvrm, memw→lt) must also be computed device-side or kept as a
  small host tail. Whatever's still needed on host (LT table `lt_ops`, mul/dvrm dedup) is downloaded.
- **S4 — drop the host `collect_cpu_ops` + `collect_ops_from_cpu_inner`/`collect_all_ops`** for the resident
  sources; keep only the irreducible host residual (ecall/precompile op assembly from recordings, which is
  tiny). Measure p1+p2a+p2b → target ~0 host.
- **S5 — measure + broaden** (warm bench; e2e on ethrex_5tx AND a second guest/larger block).

## Risks / gotchas
- `cpu_ops` feeds EVERYTHING → any field wrong = wrong proof. Parity-gate S1 field-by-field before wiring.
- The decode-by-pc resolution: upload a sorted `(pc, packed_decode, imm)` table (unique instructions) and
  binary-search per log (like `image_lookup`). Host builds it once (cheap).
- `from_log` branch logic (word_instr delegate rows, arg2 multiplex, branch_taken, jalr, x255=PC) must be
  ported bit-exactly — this is the main correctness surface.
- Keep the host path as a fallback (flag-gated) until S4; never remove the CPU path without the parity gate.
- Measure warm (`gpu_resident_bench`, `from_elf_and_logs_with_precompiles`). Watch for the "parallel-hidden"
  trap: p1/p2a/p2b are SERIAL (good — real wall wins, unlike the parallel-hidden p5 table gens).

## Payoff estimate
Best case removes ~1.15s serial host work (+ the ~150ms shared_devops upload) → trace_build ~3398 →
potentially ~2.2–2.5s (needs the logs upload + any residual download netted out). Net prove ~+3–5%. The
bigger value is mandate: the entire post-execution trace-gen on device, with only the executor VM on CPU.

## First action next session = S1 (device `cpu_ops` packer + parity test). Everything downstream is already
device-resident and reads the SoA, so S1 is the keystone.
