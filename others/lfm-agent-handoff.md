# R1d handoff — keccak-probe → successor

Written 2026-07-29 ~20:30Z. R1d is PARTIAL. Foundations are built and verified;
the `TranscriptReplay` emitter is not started. Handing off on context, per the
team lead's "quality over completion" instruction.

**State: 69/69 `cargo test -p lambda-vm-prover --lib lfm`, `make lint` 0,
nothing committed.** Worktree at `e0add1d5` (post-merge, #841 present). Tracked
diff is 19 lines (`prover/src/lib.rs` +1, `prover/src/tables/types.rs` +18) —
both pre-existing, not mine. Everything else of ours is untracked:
`prover/src/lfm/`, `prover/src/bin/compute_lfm_registry.rs`, `others/`.

---

## 1. What is DONE (R1d)

### 1a. Reversed-digest primitive — `sample()` replayed and PROVED
- `layout::keccak`: `REV_ADDR0/1`, `REV_MULT0/1` (prep width 52 → 56).
- `chips::keccak`: two extra `LfmMem` sends whose lanes are reversed-coefficient
  `Linear`s over the **existing** OUT byte columns —
  `reversed half h = Σ_k OUT[31 − 4h − k]·256^k`. Zero new value columns.
- `instr::KeccakOperands.rev: Option<KeccakReversedDigest>`,
  `LfmBuilder::keccak_absorb_rev`, `edsl::keccak256_rev`,
  `programs::keccak_sample_program(len)`.
- Tests: `machine_reversed_digest_matches_default_transcript_sample`
  (execute-only, vs the REAL `DefaultTranscript`, lengths 0/1/135/202) and
  `machine_proves_the_sample_replay` (prove+verify, lengths 0/135/202).

### 1b. Host model — `keccak_host::TranscriptModel`
Mirrors post-#841 `DefaultTranscript` (`segment` / `buf` / `pos`). Verified by
`transcript_model_matches_default_transcript` across: draining a squeeze then
forcing a refill on the 5th candidate, an absorb mid-buffer, a raw `sample()`
(which ALSO invalidates), and absorbs of length 1/135/136/200.

### 1c. ★ The reversal-cancellation identity — `candidate_from_state`
**Candidate `i` of a squeeze is the plain digest's `u64` lane `3 − i`.**
`Σ_{k<8} reversed[8i+k]·2^(8(7−k))` with `reversed[j] = digest[31−j]`, sub
`m = 7−k` ⇒ `Σ_{m<8} digest[24−8i+m]·2^(8m)` = the LE u64 at digest byte offset
`24−8i` = state lane `3−i`. The BE read and the byte reversal cancel.
Verified by `be_candidates_are_plain_state_lanes`. **This is now the spec** (team
lead accepted it): sampling needs no reversal, no extra `Linear`, no BitDec.

---

## 2. What REMAINS (against the R1d spec)

1. **`edsl::TranscriptReplay`** (new file `transcript_replay.rs` suggested).
   Host-side emit-time state mirroring `TranscriptModel`: the segment as a
   `Vec<Felt>` of `u32` halves **plus a byte length** (padding is length-driven),
   the current squeeze's 8 half-cells, and `out_pos`.
   - `append_halves(...)` / `append_word(...)`: extend the segment, set
     `out_pos = SQUEEZE_LEN`.
   - `sample(&mut self, b) -> [Cell; 2]`: emit `keccak_absorb_rev` over the
     segment; the row yields BOTH the plain digest (state words 0,1 — the
     candidate source) and the reversed digest (the re-absorb prefix). Set the
     next segment to the reversed digest's 8 halves; set `out_pos = SQUEEZE_LEN`.
2. **Candidate extraction.** `unpack` the two PLAIN digest words → 8 half-felts
   `h[0..8]`. Candidate `i` = `(lo = h[6−2i], hi = h[7−2i])`. Two `unpack`s per
   squeeze. Refill when `out_pos + 8 > 32`.
3. **`sample_field_element`** with the canonicity guard. `candidate ≥ p` iff
   `hi = 2^32−1 ∧ lo ≠ 0` (derivation: `p−1 = (2^32−1)·2^32`; if `hi < 2^32−1`
   the max is `(2^32−1)·2^32 − 1 < p`). Emit `g = (2^32−1) − hi`, `z = is_zero(g)`
   via hinted-and-verified inverse (`z·g = 0` and `z + g·ginv = 1` pin `z`
   uniquely), then `assert z·lo = 0`. Felt = `hi·2^32 + lo`.
   - **Arena rule note to put in a comment:** hinting `z`/`ginv` is sound because
     they are VERIFIED in-circuit; the rule bans unverified TRANSCRIPT inputs,
     not verified auxiliary witnesses.
4. **Zero-rejection variant + completeness doc.** The emitted program cannot
   prove an inner proof whose transcript ever rejected a candidate
   (p ≈ 2⁻³² per draw). **Fold in the ext3 correction: an extension draw is 3
   independently rejection-sampled candidates, so ~3× the per-draw figure.**
   State the resulting **per-proof bound at real draw counts**, not a ratio.
   Structure so a k-rejection variant is an emitter PARAMETER later, not a
   redesign.
5. **`sample_u64_pow2(nbits)`**: low `nbits` of the candidate. Confirmed from
   source: `threshold = upper_bound.wrapping_neg() % upper_bound` is 0 at powers
   of two, so it never rejects and returns `candidate % 2^n`. For `nbits ≤ 32`
   only `lo` matters → `b.bit_dec(lo, nbits)` then recombine `Σ 2^i·b_i`.
   **Assert `nbits ≤ 32`** rather than silently mishandling more (team lead: FRI
   query bounds are ≤ 2^25, so the bound is real).
6. **Acceptance**: a scripted interleaving (absorbs of several lengths /
   3× `sample_field_element` incl. one ext3 / absorb / `sample_u64(1<<20)` /
   `sample_field_element`) producing IDENTICAL values to a host
   `DefaultTranscript`, proved+verified e2e with sampled values `public`ed;
   plus tamper one absorbed half → reject. Register `TranscriptReplayV0`,
   regenerate the registry, add a drift test.

---

## 3. Non-obvious decisions and WHY (not visible in the diff)

- **`sample()` returns the SAME 32 bytes it re-absorbs.** One value serves as
  both the challenge and the next segment's prefix — do not emit two.
- **Reversed digest is scoped to RE-ABSORB ONLY** after the cancellation finding.
  Do not use it for candidates.
- **`sample()` invalidates the buffer too**, not just `append_bytes` /
  `append_field_element`. All three set `out_pos = SQUEEZE_LEN`.
- **Prep-column growth moves ALL registry digests.** Any layout change ⇒
  stub `LFM_REGISTRY` to `&[]`, `cargo run --release --bin compute_lfm_registry`,
  paste, rebuild. The bin cannot build while the table is stale — that is why
  the stub step exists.
- **`KeccakF` is boxed** (`Instr::KeccakF(Box<KeccakOperands>)`): inline, its
  312-byte payload quadrupled the whole `Instr` enum and failed clippy's
  `large_enum_variant`. Keep it boxed when adding fields.
- **Slot 11 (`KECCAK_RND`) has no preprocessed columns** — all-zero sentinel root,
  height 0. `LfmAirs::new`'s roots array is a PARTIAL FUNCTION. `build_air_no_prep`
  exists solely for it.
- **The digest binds the static `KECCAK_RC` / `BITWISE` roots**, so a change to
  those production tables moves every LFM program digest. Deliberate.
- **`verify_against(roots, program_id, …)`** exists so per-shape programs can be
  proved AND verified without a registry entry. It is NOT a registry off-switch;
  `lfm_verify` still hard-errors on a miss. Keep that distinction in comments.
- **Length is program shape.** Each message length is a distinct program and
  identity; register one representative, verify the rest via `verify_against`.
- **Rate-region pass-through constraint** `MODE_PERM·(PERM_IN − STATE) = 0` is
  load-bearing and NO bus catches it (see §4).

---

## 4. Test-oracle gotchas — read before writing tests

**(a) Execute-only tests are VACUOUS with respect to the chip.**
For any value the executor computes host-side AND the chip recomputes on the
bus, `execute()` never evaluates the bus interaction — so chip-side corruption
cannot move the result. I hit this exactly: my reversed-digest bit-exactness
test was execute-only, and neutralising the chip's reversed-coefficient `Linear`
left it GREEN. Fix was `machine_proves_the_sample_replay` (prove+verify), which
fails under the same neutralisation. **Both tests are kept on purpose** — the
execute-only one validates the executor mirror against the real
`DefaultTranscript`, the proving one validates chip-vs-executor agreement.
Anything R1d adds to the adapter needs a PROVING test.

**(b) Scrutinise the oracle as hard as the thing under test.**
`transcript_model_matches_default_transcript` failed on first run and the MODEL
was right — my comparison was wrong. `sample_u64(2^n)` returns `candidate % 2^n`
(threshold 0 at powers of two), not the raw candidate; the delta was exactly
2^63. Mask the model's raw candidate, and compare raw 32-byte squeezes via
`sample()` separately.

**(c) Coherent forgeries, not trace tampering, find constraint holes.**
The permute-mode hole (R1c) is invisible to trace tampering — tampering desyncs
the round chip and the bus catches it first. It took building a forgery where
KECCAK_RND, BITWISE multiplicities, the reply token, the output words AND the
claimed public words were all internally consistent, so every bus balanced and
only the constraint stood in the way. Then neutralise the constraint and confirm
the forgery is ACCEPTED. That pattern is the standard here; see
`permute_row_cannot_substitute_the_permuted_state`.

**(d) Falsify every new mechanism.** Every load-bearing piece in R1a–R1d was
confirmed by breaking it and watching the right test fail: R1a token lane order,
R1b half-recomposition byte order, R1c block byte mapping + the rate-equality
constraint, R1d the chip's reversal. If a falsification does NOT fail, the test
is vacuous — see (a).

---

## 5. Status log

`others/lfm-agent-status.log`, one line per slice boundary. Last line is marked
`R1d-PARTIAL-HANDOFF`. Append at each slice; the team lead polls it if the
mailbox goes quiet.

## 6. Process notes

- Mailbox messages crossed repeatedly. The team lead now uses
  `others/lfm-team-lead-*.md` for anything authorization-shaped; **check this
  directory when a blocker answer seems overdue.**
- `git stash list` has a pre-existing unrelated `bench-keccak-vs-leanvm WIP`
  entry. Leave it alone.
- Nothing is committed and nothing should be without the user's say-so.
- `make lint` from the repo root is the gate (`cargo fmt --check` is not a
  substitute); it runs four clippy configurations.

---

## 7. File map — everything added or changed

All under `prover/src/lfm/` unless noted. Nothing is committed; every file below
except the two tracked ones is UNTRACKED.

**New files**
- `keccak_adapter.rs` (517L) — the raw keccak-family contract: the two
  `BusId::Keccak` tokens, `KECCAK_RND`/`RC`/`BITWISE` trace drivers, the
  per-round BITWISE feed (forked from `trace_builder::collect_bitwise_from_keccak`),
  the `u32`-half state↔words conversion, the absorb XOR feed, and the host mirror
  of the reversed digest.
- `keccak_probe.rs` (290L) — standalone probe of the UNCHANGED production family.
  Deliberately untouched since R1a; it documents the raw contract including the
  live tag-swap hazard.
- `keccak_host.rs` (198L) — byte-stream packing convention, `pad10*1`,
  `PlatformKeccak256` reference wrapper, `TranscriptModel`, `candidate_from_state`.
- `others/lfm-agent-status.log`, `others/lfm-agent-handoff.md` (this file).

**Changed files**
- `layout.rs` — `mod keccak`: prep layout (tags, 13+13 addrs, mults, 9 block
  addrs, 2 mode selectors, 2 rev addrs + 2 rev mults = 56 wide) and `tag_for_row`.
- `instr.rs` — `Instr::KeccakF(Box<KeccakOperands>)`, `KeccakMode`,
  `KeccakReversedDigest`; `writes()`/`reads()` arms.
- `builder.rs` — `keccak_f`, `keccak_absorb`, `keccak_absorb_rev`, shared
  `emit_keccak`.
- `compiler.rs` — keccak group emission (tag = row ordinal, mode one-hot, block
  and rev addrs) + multiplicity backfill.
- `executor.rs` — the `KeccakF` arm (permute and absorb), `KeccakRow` record,
  `NotU32Half` / `KeccakSpareLaneNonZero`.
- `validator.rs` — keccak partition count, mode one-hot, padding; **check 8**
  (`check_keccak_tags`) with `DuplicateKeccakTag` / `MalformedKeccakTag`.
- `chips.rs` — the `LFM_KECCAK` chip: 788 columns, 173 interactions,
  `KeccakAdapterConstraints` (201 constraints, degree 2).
- `trace.rs` — the keccak chip trace plus the three production family traces.
- `airs.rs` — 10 → 14 chips, `build_air_no_prep`, `KECCAK_RND_SLOT`,
  `keccak_rnd_rows`, extended `lfm_cell_counts`.
- `registry.rs` — `build_artifacts` over 14 slots with the slot-class doc,
  `KeccakChainV0` / `KeccakSpongeV0` kinds, regenerated table (4 entries).
- `programs.rs` — `keccak_chain_program`, `keccak_sponge_program(len)`,
  `keccak_sample_program(len)`, `KECCAK_SPONGE_LEN = 202`.
- `edsl.rs` — `keccak256`, `keccak256_rev`, shared `keccak256_absorb_all`.
- `proof.rs` — split out `prove_traces` and `verify_against`.
- `machine_tests.rs` (1008L) — all R1a–R1d tests.
- `mod.rs` — module registrations.
- `prover/src/bin/compute_lfm_registry.rs` — the three new programs.
- TRACKED (pre-existing, not mine): `prover/src/lib.rs` +1 (`pub mod lfm;`),
  `prover/src/tables/types.rs` +18 (BusId 32/33/34).

## 8. Guard-test map — what each test pins

| Test | Hole it pins |
|---|---|
| `keccak_probe::duplicate_tag_output_swap_accepts_demonstrating_hazard` | Documents the LIVE forgery on the raw family — **asserts it SUCCEEDS**. If it ever starts failing, something began binding request→reply; re-derive before relaxing. |
| `preprocessed_tags_close_the_output_swap_hazard` | Closes the above, 3 legs: distinct tags / swap now rejects / prover can't collide tags (`PrecomputedCommitmentMismatch`). |
| `duplicate_keccak_tags_fail_admission` | The registrar's independent gate on tag uniqueness. |
| `permute_row_cannot_substitute_the_permuted_state` | ★ The permute-mode hole. NO bus catches it; only `MODE_PERM·(PERM_IN−STATE)=0` does. Built as a coherent forgery. |
| `tampered_absorb_xor_rejects` | The absorb XOR is pinned by the 136 BITWISE lookups. |
| `keccak_rejects_non_u32_half`, `keccak_rejects_nonzero_spare_lane` | Executor guards on the `u32`-half word convention and the 2 spare slots. |
| `tampered_keccak_input_half_rejects` / `_output_half_rejects` | State byte columns bound to memory and to the family. |
| `keccak256_matches_platform_hasher` | Bit-exactness vs the production hasher, 8 boundary lengths. |
| `machine_proves_the_sample_replay` | Chip-vs-executor agreement on the reversed digest (the execute-only sibling canNOT see this — see §4a). |
| `transcript_model_matches_default_transcript` | The emit-time oracle is correct. |
| `be_candidates_are_plain_state_lanes` | The reversal-cancellation identity the emitter rests on. |
| `registry_drift_*` (4) | Program identity; investigate, never re-bless. |

## 9. Packing conventions (get these wrong and nothing balances)

- **State**: 25 `u64` lanes → 50 `u32` halves → 13 words. Half `h` = low (`h`
  even) / high (`h` odd) 32 bits of lane `h/2`. Word `j` carries halves `4j..4j+3`.
  Last word's top **two** slots are unused, pinned zero as tuple constants.
- **Byte columns** are lane-major: `STATE + lane*8 + b`, little-endian in-lane.
  Note `(h/2)*8 + 4*(h%2) == 4h`, so half `h` starts at byte column `4h`.
- **The column-major trap is TOKEN-order only.** Keccak bus element
  `3 + 8(5x+y) + b` is byte `b` of lane `x+5y` (lanes visited 0,5,10,15,20,1,…).
  The COLUMN layout is plain lane-major — block byte `k` pairs with state byte `k`.
- **Rate block**: 136 bytes = 17 lanes = 34 halves = 9 words, 2 spare slots.
- **Byte streams** (`keccak_host::pack_stream`): `u32` halves, 4 bytes each, LE,
  final partial half ZERO-PADDED. The emitter's `stream_half + pad_const` equals
  a bitwise merge only because of that zero-padding —
  `assert_high_bytes_zero` states the obligation executably.
- **Digest**: first 32 state bytes = halves 0..7 = words 0,1. Digest byte `j` =
  byte `j%4` of half `j/4`. Matches `PlatformKeccak256` output order exactly.

## 10. Half-formed emitter intentions (what I would have done next)

- **New file `transcript_replay.rs`**, not more `edsl.rs`. `edsl.rs` is already
  the FRI/sponge library; the transcript is its own concern.
- **Shape**: `TranscriptReplay { segment: Vec<Felt>, segment_bytes: usize,
  buf: Option<[Felt; 8]>, out_pos: usize, hints: ArenaId, hint_cursor: u32 }`.
- **Carry a `TranscriptModel` alongside and `debug_assert` they agree at every
  step.** The consumption schedule is static, so a divergence is a BUILD-time
  bug; catching it at emit time beats discovering it as a failed proof.
- **Host hint generation next to `TranscriptModel`** in `keccak_host.rs`, so the
  `z`/`ginv` vector is produced by the same code that models the schedule —
  one source of truth for ordering.
- `sample_field_element` → `felt = hi·2^32 + lo` as a single `mul_add` against an
  interned `2^32` constant. ext3 = three consecutive draws, assembled with
  `pack_ext`.
- **★ THE ONE REAL DESIGN DECISION I DID NOT RESOLVE — partial-half appends.**
  The production transcript absorbs ARBITRARY byte lengths, but our segment is a
  vector of 4-byte halves. An append whose length is not a multiple of 4 puts a
  partial half in the MIDDLE of a segment, where the next append's bytes must
  continue inside that same half. That is the mixed-half problem from R1c padding,
  except it can recur mid-stream instead of only at the end, and the
  `stream_half + pad_const` trick does not generalise (the later bytes are not
  known-zero-padded, they are real data). Options I weighed:
  (a) restrict `append` to whole halves and ASSERT it — fine for the FRI verifier,
      whose absorbs are digests and field elements (all multiples of 4/8 bytes),
      and I would start here;
  (b) carry a partial-half accumulator in the emitter and merge with an in-machine
      `mul_add` when the next append arrives — correct in general, more instructions;
  (c) re-pack the whole segment per sample — simplest, most wasteful.
  **Recommendation: (a) with a loud assert, then (b) only if a real caller needs
  it.** Do not silently truncate or pad — that would diverge from the host
  transcript in a way no test in this suite would catch unless it specifically
  exercises a non-multiple-of-4 append. If you take (a), ADD a test that the
  assert fires, so the limitation is pinned rather than latent.
