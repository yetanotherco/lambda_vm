# start_index binding — production reading (ledger entry 2)

This file replaces a raw 518 KB JSONL session transcript that was committed

under this name at e105dea2 (recoverable from git history if the full session

is ever needed). What follows is the research agent's final report, extracted

verbatim from that transcript. Findings are also summarised in

`lfm-assembly-obligations.md` entry 2.

---

## Answer: production carries `start_index` as a register value (x254), and binds it by *reconstructing the AIR from the previous epoch's FINI* — never by an arithmetic cross-epoch check

### 1. Where it lives (✓ VERIFIED, all read)

- **Executor: no notion of it.** `Memory::commit_public_output` only appends bytes to `self.public_output` (`executor/src/vm/memory.rs:208-215`); its own doc says "the COMMIT AIR concatenates calls via the running `x254` index" (`memory.rs:41-43`, `:204-207`). No offset/counter is exported.
- **Prover trace side:** the counter is a *synthetic single-word register x254 at word-address 508*, held in `RegisterState.index_register` (`prover/src/tables/trace_builder.rs:159-160`), **seeded from `init[X254_INDEX]`** (`trace_builder.rs:194`), advanced by `count` per COMMIT ecall (`trace_builder.rs:1254-1274`, `:605-612` with a sync debug-assert), and emitted as the REGISTER final state at address 508 (`trace_builder.rs:266-273`). `register::X254_INDEX = 64` (`prover/src/tables/register.rs:134`).
- **Verifier side:** `compute_commit_bus_offset(public_output, start_index, z, alpha)` builds the COMMIT-bus receiver terms at `global_index = start_index + i` (`prover/src/lib.rs:909-946`, esp. `:929`), reached via `compute_expected_commit_bus_balance_view` (`lib.rs:969-978`).

### 2. Is it a public/absorbed value? **No — not directly.** (✓ VERIFIED)

`absorb_statement` absorbs only: domain tag, ELF digest, length-prefixed `public_output`, the 14 `table_counts`, `num_private_input_pages` (hard-coded 0 for epochs, `continuation.rs:103-104`), `fri_final_poly_log_degree`, `runtime_page_ranges`, and `epoch_label` (`prover/src/statement.rs:80-152`). **No `start_index`, no register vectors.** The `EpochProof` bundle fields are `proof, public_output, table_counts, runtime_page_ranges, reg_fini, l2g_root` (`continuation.rs:393-411`) — `start_index` is not a field; it is *derived* from `reg_fini` of the predecessor.

### 3. What the verifier actually checks between epochs (✓ VERIFIED — read the body)

`verify_continuation_view` loops epochs and does exactly this (`continuation.rs:1310-1338`):

```rust
let mut register_init = register::register_init_from_entry_point(elf.entry_point); // :1310
...
public_output.extend_from_slice(epoch_public_output);   // :1334
register_init = epoch.reg_fini()?;                      // :1337
```

and `verify_epoch` derives (`continuation.rs:840-851`):

```rust
let commit_start_index = register_init.get(register::X254_INDEX).copied().unwrap_or(0) as u64;
```

**There is NO explicit `epoch[i].start_index == epoch[i-1].start_index + epoch[i-1].output_len` check.** I confirmed structurally, not by term search: the only `len()` uses in `continuation.rs` are proof/table/epoch counts and `reg_fini_len()` (grep of all `.len()` sites, lines 335–2337; the only length validations are `expected_proof_count != proof.len()` at `:806-809` and `reg_fini_len() != NUM_REGISTER_ADDRESSES` at `:1302-1307`). Output bytes are simply concatenated in order (`:1334`) and returned (`:1399`).

The binding is **structural**, in three composed locks:

1. **Preprocessed REGISTER (OFFSET, INIT, FINI).** Each epoch's AIR is rebuilt by the verifier with `compute_precomputed_commitment_with_fini(opts, register_init, reg_fini)` and `NUM_PREPROCESSED_COLS_WITH_FINI = 3` (`continuation.rs:656-659`; `register.rs:67`, `:302-322`). The STARK verifier **rejects unless the proof's preprocessed root equals the AIR-recomputed one**, then absorbs it (`crypto/stark/src/verifier.rs:1184-1209`). So trace INIT/FINI are locked to the verifier's u32 vectors.
2. **REG-C2 on the epoch-local Memory bus** sends `(1, address, timestamp, FINI)`, matching MEMW's last receive (`register.rs:406-434`), so FINI = real last write to x254.
3. **The verifier reuses the *same* vector** as epoch i's FINI and epoch i+1's INIT (`continuation.rs:1337` feeding `:820`), so `init(i+1) == fini(i)` holds by construction — documented at `register.rs:59-67` and `docs/continuations_design.md:445-470` ("two locks").

Epoch 0 is anchored: `init_value_for_address(508, _) => 0` (`register.rs:150-158`), so `start_index = 0` at genesis, likewise for monolithic proofs.

**The `start = prev_start + prev_len` relation is a derived consequence, not a check:** the COMMIT chip's MEMW token asserts `old x254 == INDEX` and writes `INDEX + count` (`commit.rs:592-652`, esp. the `value` linear form `INDEX + count_0 + 2^32·count_1` at `:616-629`); the CommitNextByte chain forces `index+1, address+1, count-1` per row (`commit.rs:262-332`) with ZERO-bus end detection (`:399-430`); and the COMMIT-bus sender `(INDEX, VALUE)` with mult `mu-end` (`commit.rs:708-722`) must balance against the verifier's receiver terms over the claimed `public_output` at `start_index + i`. So the epoch's byte count *equals* its x254 delta, and a lying `public_output` length imbalances the bus.

### 4. Inside the AIR? **No boundary constraint exists at all.** (✓ VERIFIED)

`CommitConstraints` emits only 8 transition constraints — three IS_BIT, `(first+end)(1−mu)`, and two ADD pairs (`commit.rs:730-771`). There is no row-0/last-row anchor on `INDEX`. Structurally confirmed: every VM AIR is `AirWithBuses<F, E, NullBoundaryConstraintBuilder, (), CS>` (`prover/src/test_utils.rs:122`, COMMIT at `:844-853`), and `NullBoundaryConstraintBuilder`'s impl body is **empty** (`crypto/stark/src/lookup.rs:1647-1653`); a repo-wide grep for `boundary_constraints|BoundaryConstraint` under `prover/src/tables/` returns nothing. So `start_index` is bound *only* by the MEMW-token → REGISTER-preprocessed-commitment path, plus the verifier-side offset.

### 5. Register file: yes, x254 is reg slot 64 (✓ VERIFIED)

`NUM_REGISTER_ADDRESSES = 67` (x0–x31 → 0..63, x254 → 508, x255 → 510/511) (`register.rs:43-52`, `:116-128`); `fini_from_trace` reads FINI on those 67 rows (`register.rs:265-269`). Enforcement of `reg_init[254](N) == reg_fini[254](N−1)` = the three locks in §3 — the verifier literally passes the same `Vec<u32>` to both sides, so equality cannot be violated; the *untrusted* part is `reg_fini` itself, pinned by lock 1 + lock 2.

### Notes / caveats

- Because FINI is committed as `FE::from(u32)` (`register.rs:318`) and the root is verifier-recomputed, `start_index < 2^32` is forced on the production continuation path (relevant to LFM ledger OPEN entry 1, `others/lfm-assembly-obligations.md:15-23`).
- The recursion attestation carries only `program_id || public_output` (`prover/src/recursion.rs:286-294`, `:311-314`) — **no start_index, no per-epoch offsets**, so nothing at the attestation layer binds position either.
- ✗ UNVERIFIED (did not trace): that no ordinary RV instruction can address word 508 (5-bit rd/rs fields ⇒ ≤ addr 62). If LFM ever re-derives x254 writes, that bound deserves its own check.
- Regression test that exercises the carry: `test_commit_across_epochs_verifies` (`continuation.rs:1457-1498`), which forces a commit into an epoch where x254 is already 2.

**For LFM:** production's binding is *"rebuild epoch N's REGISTER preprocessed commitment from epoch N−1's FINI vector"* — an AIR-reconstruction binding, not an in-machine arithmetic one. The in-machine analogue is to make the arena's `start_index` word the same word the REGISTER preprocessed-commitment recompute consumes (i.e. bind it to `reg_fini[64]` of the previous epoch, which the reg leg already handles), rather than inventing a `start + len` equation the production verifier does not have.