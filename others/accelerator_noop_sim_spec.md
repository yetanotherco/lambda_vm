# No-op-syscall simulation experiments — spec (2026-07-20)

Purpose: per-candidate OPTIMISTIC ceilings for the accelerator matrix (see
`others/accelerator_archaeology.md` + memory `accelerator-prioritization-plan`). Each experiment adds
a TRUSTED executor ecall that computes the correct value host-side and returns it in 1 cycle,
unproven, replacing in-guest computation. Ceiling = exact cycle delta vs baseline on the real target.

## Ground rules
- **Execute-only.** Measurement = `cli execute <elf> --private-input <blob> --cycles`
  (bin/cli/src/main.rs `cmd_execute`; cycles = logs.len(), every ecall costs exactly 1 cycle).
  **NEVER prove a stub build** — an ecall with no chip table unbalances the Ecall LogUp bus (same as
  the known Print-ecall caveat, syscalls/src/syscalls.rs:37-40).
- Stub returns the **CORRECT value** (trusted passthrough), so the guest still ACCEPTS the proof and
  attests. Acceptance check + mandatory tamper test (flip 1 blob byte → guest must panic/exit≠0) are
  both execute-only.
- Fixed-blob exact protocol: one blob dump per baseline; every candidate ELF measured against the
  same file → integer deltas, no noise.
- Machine: **ON HOLD — user decides local Mac vs a remote server before any build/run.**

## Measurement base branch (compose once, share across experiments)
`origin/main` @ 3ea4f916 (has #844+#845) + `bench/recursion-full-queries` @ 51faaa52 (dump-test env
knobs, blowup2/blowup4 presets, cont guests, bench scripts). `git merge-tree` says 4 content
conflicts: executor/src/vm/memory.rs, prover/src/continuation.rs, prover/src/recursion.rs,
prover/src/tests/recursion_smoke_test.rs — bench's prover-side continuation code is SUPERSEDED by
main's #844/#845; resolve toward main, keep the harness knobs/presets (the param-sweep agent made
the same compose server-side incl. #853 and confirmed this resolution works; its branch
`sweep/throwaway` on 195.154.212.27 is a reference).
- Baseline dump: `RECURSION_DUMP_PRESET=blowup2 RECURSION_DUMP_INNER_ELF=<ethrex.elf>
  RECURSION_DUMP_INNER_INPUT=<ethrex_bench_4.bin> RECURSION_DUMP_EPOCH_LOG2=21 cargo test
  test_dump_recursion_input ...` → /tmp/recursion_input.bin (~35s–2min; roots always embedded →
  guest takes the #844/#845 supplied-roots path). Measure `recursion-cont-blowup2.elf` → expected
  ≈2.68B-class baseline. Keep a copy of the blob outside /tmp.
- ELF builds: `make compile-recursion-elfs SYSROOT_DIR=$HOME/.lambda-vm-sysroot` (Makefile:238-250
  bench; cont targets are bench-only).
- #768-composed re-measure is a FOLLOW-UP baseline (deferred; #2's ceiling predicted invariant —
  treat that as a testable claim).

## Adding a measurement-only ecall (verified checklist, anchors on origin/main)
1. syscalls/src/syscalls.rs — number const next to :29-34 (`usize::MAX - N`, N unused; keccak=MAX-1,
   ecsm=MAX-10), guest asm wrapper pair modeled on :150-166 (a7=id, a0/a1/a2=args).
2. executor/src/vm/instruction/execution.rs — matching `u64::MAX - N` const near :24, SyscallNumbers
   variant :10-19, TryFrom arm :41-49, accelerator() arm :64-73 (exhaustive match — must add), dispatch
   arm in :362-467 modeled on keccak :398-423 (read regs 10/11/12 → validate → load_doubleword loop →
   host compute → store_doubleword → src2_val), ExecutionError variants near :606-632.
3. bin/cli/src/main.rs — counter :481, increment arm :512-516, print :536-538 (report the new ecall's
   call count alongside keccak).
4. Guest-side swap of the software path: cfg/feature-gated in the verifier crate (crypto/stark /
   crypto/crypto), feature enabled only by the stub guest build (recursion guest Cargo features —
   pattern: presets are already features, Makefile build_guest_elf `--features`). Executor/syscalls
   crates have NO feature machinery today (verified) — keep the stub cfg in the GUEST-side crates; the
   executor handler can exist unconditionally (unused unless the guest calls it).

## Experiment 2 — REDUCED_OPENING stub (candidate #2, DEEP α-ladder)
**Discovery (scout, struct-verified): our verifier does NOT run a per-query Horner ladder.** The γ
(DEEP challenge) powers are precomputed ONCE per proof (verifier.rs:1448-1456 successors ladder →
trace_term_coeffs [col][row] grid), so the per-query hot loop is the WEIGHTED-SUM form
`acc += coeff·value` — verifier.rs:1000-1037 (trace terms; base cols = Goldilocks×ext asymmetric
mul, aux = ext×ext; g·z pruning: rows ≥ step_size only next_row_cols) + :1054-1064 (composition
parts `h_sum += h_i·γ_j`), then `(sum − ood_row_sum)·denom` per row (:1035-1036, ood_row_sum hoisted
query-invariant at :778-795). Nesting: queries × ood_rows × columns, all ext accumulators, called
from step_3_verify_fri → reconstruct_..._for_all_queries (:859) → ..._evaluation_pair (:925).
**Do NOT stub** the FRI commit-phase butterfly `v=(v+v_sym)+x⁻¹ζ(v−v_sym)` (verifier.rs:729-730) —
that stays (its chip died in both reference systems; plain guest algebra).
- **Level A (primary, chip-ABI-faithful):** ecall per (query × ood_row): host computes
  base_row_sum + base_row_sum_sym for that row (the :1005-1034 column loop). Args: pointers to the
  eval slices + coeff column grid + row idx/dims; returns 2×Fp3 (6 u64). Mirrors "one call per opened
  row" — the ceiling for the openvm-shaped chip. (Chip-design note: real chip can regenerate coeffs
  from γ by Horner instead of reading the grid — decision deferred, and the MEASUREMENT COVERS BOTH:
  the stub swallows the loop and the returned value is identical under either form, so this ceiling
  applies to a Horner chip as-is. Horner-vs-grid only moves the CHIP-COST column (values-only vs
  values+coeffs memory rows). Fine print: a Horner chip additionally lets the guest skip
  materializing the coeff grid — a once-per-proof term this stub keeps, so the Horner ceiling is
  slightly understated; measure separately only if it registers in the flamegraph. The
  strided-vs-consecutive powers question is chip-layout only (check ood.rs build_trace_term_coeffs
  ordering at chip-spec time), invisible to this experiment.)
- **Level B (secondary, fusion upper-bound):** one ecall per query returning the whole
  (deep_eval, deep_eval_sym) pair (entire :925 function body host-side). Protocol-shaped — NOT a
  build candidate, measured only to bound what deeper fusion could ever buy.
- Report: cycles, keccak_calls (should be unchanged), reduced_opening_calls, exact deltas.

## Experiment 1 — field-native hash/transcript stub (candidate #1)
**Design principle (matches the chip decision): STATELESS ecalls, sponge state stays in guest
memory.** The guest transcript state is just a streaming Keccak256 (syscalls sponge: [u64;25]+offset)
— no field-element state exists (default_transcript.rs:15-18). Each ecall takes a state_ptr into
guest memory, loads the sponge struct, applies the exact same update/finalize semantics host-side
(reuse the same crypto code → byte-identical), stores back.

Ecall family (host computes; all trusted/unproven):
- `ABSORB_FELTS(state_ptr, elems_ptr, count, kind)` kind∈{base=1 limb, ext=Fp3=3 limbs}: host reads
  raw limbs, applies the canonical stream_bytes serialization, absorbs. THE field-native ABI — kills
  the per-element marshaling.

**FIELD-TYPE CORRECTION (verified 2026-07-20, supersedes any Fp2 mention):** the production
verifier's FieldExtension = **Degree3GoldilocksExtensionField (Fp3, 3 limbs/24B)** — types.rs:25-28,
test_utils.rs:109-110. Fp2/Degree2 has NO AsBytes/HasDefaultTranscript impl and cannot reach the
transcript or Merkle paths. Transcript append_field_element sees ONLY Fp3; Merkle leaves are
monomorphic per tree (main/precomputed = base Goldilocks 8B; aux/composition/FRI-layer = Fp3 24B;
never mixed in one leaf); parent hash always fixed 64B keccak256_pair. Serialization: base =
canonical_u64 8B **BE**; Fp3 = limbs 0,1,2 each 8B BE (extensions_goldilocks.rs:566-571; do NOT use
the reversed [FpE;3] ByteConversion impl at :42-49).
- `ABSORB_BYTES(state_ptr, bytes_ptr, len)` for the few raw-byte absorbs.
- `TRANSCRIPT_SAMPLE(state_ptr, out32_ptr)`: the whole sample() step (finalize_reset + reverse +
  re-absorb, default_transcript.rs:43-48) as one call. ChaCha expansion stays guest-side (measured
  small post-#841 analysis).
- `HASH_PAIR(l_ptr, r_ptr, out_ptr)`: fixed 64B Merkle parent (replaces keccak256_pair sponge).
- `HASH_FELTS(elems_ptr, count, kind, out_ptr)`: one-shot leaf-row hash (replaces hash_streamed).

Guest swap sites (all behind #[cfg(target_arch="riscv64")] + a sim feature, per the platform_keccak
:7/:61 template):
- Transcript: crypto/crypto/src/fiat_shamir/default_transcript.rs — append_field_element :70 →
  ABSORB_FELTS; append_bytes :66 → ABSORB_BYTES; sample :43-48 → TRANSCRIPT_SAMPLE
  (sample_field_element :78 / sample_u64 :83 keep their guest-side expansion on top).
- Merkle parent: crypto/crypto/src/merkle_tree/backends/field_element_vector.rs
  hash_new_parent_bytes :74 (guest fast path :78-86) → HASH_PAIR.
- Merkle leaf: field_element_vector.rs hash_streamed :43 (guest fast path :46-55; also
  FieldElementPairBackend::hash_data :122, hash_data_from_slices :173) → HASH_FELTS.
- Path walk (Proof::verify in merkle_tree/proof.rs) is untouched — it just calls hash_new_parent.
CAUTION: the guest fast paths bypass the PlatformKeccak256 adapter via TypeId dispatch
(field_element_vector.rs:46-55/:78-86) — the stub must preserve byte-identity on BOTH paths (see
crypto/crypto/src/hash/platform_keccak.rs:14-21 load-bearing note).

NOTE: this stub REPLACES in-guest keccak_permute ecalls → keccak_calls DROPS toward 0 on these
paths; count each new ecall separately (CLI accelerator_of + tally) so the adjusted score can be
recomputed under different chip-cost assumptions. Report: cycles, keccak_calls, per-ecall counts.

## Deliverable per experiment
{baseline cycles, candidate cycles, Δ and %, keccak/new-ecall counts, accept+attest OK, tamper
rejected} → feeds the matrix's "optimistic impact" column; chip-cost column comes from openvm's
height model + our per-table widths (#853 reports / sweep data).
