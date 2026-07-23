# Precomputed-commitment device path — design scope

Scopes moving the 6 **preprocessed** tables onto GPU / a caching path. These are
the tables the per-op device-resident seam (PR-1/PR-2) cannot touch, because
`commit_main_trace`'s device branch is gated on `precomputed.is_none()`
(`crypto/stark/src/prover.rs:784`, `:821`).

The 6: **BITWISE, DECODE, PAGE, GLOBAL_MEMORY (→PAGE), REGISTER, KECCAK_RC**.

---

## 1. How preprocessed commit works today (grounded)

A preprocessed table splits its columns into **static** (`[0, num_precomputed)`,
program-constant) and **dynamic** (`[num_precomputed, total)`, the per-execution
multiplicity `MU`). Passed to the prover as
`precomputed: Option<(Commitment, usize)>` = `(expected_static_root, num_static)`
(`prover.rs:768`, set at `:2199`).

At commit (`prover.rs:893-940`), for `Some((expected_root, num_static))`:

1. LDE the **full** trace row-major (static + dynamic together) — `:877`.
2. `commit_rows_bit_reversed_subset(main_data, total, 0, num_static)` → **static
   tree** + root (`:904`); assert `root == expected_root` else
   `PrecomputedCommitmentMismatch` (`:920`).
3. `commit_rows_bit_reversed_subset(main_data, total, num_static, total)` →
   **dynamic (mult) tree** + root (`:913`).
4. `TableCommit::preprocessed(mult_tree, mult_root, static_tree, static_root, num_static)`
   (`:932`). Both roots go to the transcript (`:2218`); both trees are needed for
   Round-4 openings.

`TableCommit` (`prover.rs:111-173`) already carries the two-tree shape:
`tree/root` (dynamic) + `precomputed_tree/precomputed_root/num_precomputed_cols`.

### The static/dynamic split per table

| Table | static cols (num) | dynamic cols | static content | per-proof update |
|---|---|---|---|---|
| **BITWISE** | 11 (X,Y,Z,AND,OR,XOR,MSB8,MSB16,ZERO,SLL,SLLC), **2^20 rows** | 10 × `MU_*` | **fully constant** (byte-op lookup) | `histogram.fill_multiplicities` (`trace_builder.rs:3607`) |
| **DECODE** | 5 (PC0,PC1,PACKED_DECODE,IMM0,IMM1) | 1 × `MU` | **per-ELF** (program instructions) | `update_multiplicities(pc_to_row, lookups)` (`decode.rs:178`) |
| **PAGE** | 2 (OFFSET,INIT) | 3 (FINI,TS_LO,TS_HI) | per-page (ELF data / zero) | filled in `generate_page_trace` |
| **REGISTER** | 2–3 (OFFSET,INIT[,FINI]) | 2 (TS_LO,TS_HI) | per-init (entry point) | filled in `generate_register_trace` |
| **KECCAK_RC** | 9 (ROUND,RC0..7), 32 rows | 1 × `MU` | **fully constant** (round consts) | `update_multiplicities(num_ops)` (`keccak_rc.rs:215`) |

**Caching that exists today:** only the static **root** is shortcut — hardcoded
`static_commitment(blowup)` for BITWISE/KECCAK_RC/zero-init PAGE (blowup∈{2,4,8},
coset=3), and per-ELF `compute_precomputed_commitment` for DECODE/REGISTER, all
injected into the AIR via `.with_preprocessed(...)` (`prover/src/lib.rs:468..584`).

**Crucially: the static tree itself is rebuilt every proof** (step 2 above — LDE +
Merkle over the static columns), just to (a) obtain the tree for openings and
(b) re-verify the root. The hardcoded root only saves the AIR-setup recompute, not
the prove-time one.

---

## 2. Key insight

The static tree is **independent of the execution** and independent of the dynamic
columns — the two subset commits (`prover.rs:904`, `:913`) touch disjoint column
ranges of the same LDE buffer. So it can be **built once and reused across every
proof** (globally for BITWISE/KECCAK_RC; per-program for DECODE/REGISTER/PAGE).

The only genuinely per-proof work for these tables is the **dynamic (MU) columns'
LDE + Merkle** — a handful of columns (1–10), tiny next to the static side
(especially BITWISE: 10 dynamic vs the 11 static × 2^20 rows recomputed today).

This makes the real lever **caching the static tree**, which is largely orthogonal
to GPU. GPU then accelerates the small remaining dynamic part and can hold the
cached static tree resident for fast openings.

---

## 3. Two opportunities

### Opportunity A — device preprocessed commit (seam parity)
Teach the device seam to emit a preprocessed `TableCommit`: device-LDE the full
resident trace, then build **two subset Merkle trees on-device** and assert the
static root. Needs a new device primitive: **subset-column Merkle over the LDE
output with bit-reversed, `ROWS_PER_LEAF` leaves** (device analog of
`commit_rows_bit_reversed_subset`, `prover.rs:578`). The full-column device tree
already exists (`coset_lde_row_major_with_merkle_tree_keep_dev`).
- **Win:** removes the per-table H2D for these tables.
- **Verdict:** marginal on its own — it still recomputes the static tree every
  proof. Only worthwhile *combined with* B.

### Opportunity B — static-tree cache (the real lever) ✅ recommended
Restructure the preprocessed commit so the static tree is computed **once** and
cached; per-proof, only the dynamic columns are LDE'd + Merkled, then combined
with the cached static tree/root.

Per-proof preprocessed commit becomes:
1. LDE **only the dynamic columns** (`[num_static, total)`) — column-independent,
   so a straight subset of the existing LDE.
2. Merkle → `mult_tree`/`mult_root`.
3. `TableCommit::preprocessed(mult_tree, mult_root, cached_static_tree,
   cached_static_root, num_static)` — reuse the cached static tree/root
   (root already equals `expected`, so the assert is trivially satisfied).

Skips the static LDE + static Merkle **entirely** on every proof after the first.
- **BITWISE** is the standout: 11 × 2^20 static cols LDE'd + Merkled every proof
  today, fully constant → cache once, save on every proof.
- **DECODE**: constant per program → save on every proof of that program.
- Works **CPU-only** already (a host cache is a real win); GPU adds device-side
  dynamic LDE and a resident static tree for Round-4 openings.

---

## 4. Concrete changes (Opportunity B, GPU-inclusive)

1. **Static-tree cache** (new). Key: `(table_id, program_id_or_const, blowup,
   coset)`. Value: `Arc<BatchedMerkleTree<F>>` + root (+ optional device handle).
   BITWISE/KECCAK_RC key on `(blowup, coset)` only; DECODE/REGISTER/PAGE include a
   program/init hash. Populated lazily on first proof, or eagerly by the existing
   `compute_static_commitments` binary (already computes these — extend it to
   persist the tree/LDE, not just the root).
2. **`commit_main_trace` preprocessed branch** (`prover.rs:893`): when a cached
   static tree is available, LDE **only** `[num_static, total)`, Merkle → mult
   tree, and assemble `TableCommit::preprocessed` with the cached static side.
   Fall back to today's full path on cache miss.
3. **Device dynamic path** (cuda): device-LDE the dynamic column subset from the
   resident trace + device Merkle (reuse `*_keep_dev` with a column-range),
   returning the mult tree handle. Only MU columns cross H2D (or none, if the GPU
   trace-gen fills MU on-device).
4. **Resident static tree for openings** (cuda, optional/last): keep the cached
   static tree's nodes device-resident so Round-4 decommits don't re-touch host.
   VRAM cost dominated by BITWISE (~350 MiB at blowup 4) + DECODE (~160 MiB) — so
   this step is opt-in / capacity-gated.

---

## 5. Value & effort

**Value ranking:** BITWISE ≫ DECODE ≫ PAGE/REGISTER ≫ KECCAK_RC (constant but 32
rows). The win is **recurring** (every proof), unlike the one-op seam.

**Not one table-port.** Suggested breakdown:
- **PR-A (CPU, proves the win):** static-tree cache + dynamic-only LDE in the
  preprocessed branch. No GPU. Measurable on BITWISE immediately. Lowest risk,
  highest info.
- **PR-B (GPU dynamic):** device LDE+Merkle for the dynamic subset; MU fills
  on-device where a histogram/collect exists.
- **PR-C (GPU resident static):** device-resident cached static trees for
  openings, capacity-gated.

**Risk:** medium — touches the commitment/opening path (consensus-critical). The
static root assert (`prover.rs:920`) is a built-in correctness net; openings must
index the cached tree identically to a freshly-built one (same bit-reversal,
`ROWS_PER_LEAF`, leaf layout). Validate with the existing e2e prove+verify tests.

## 6. Open questions
- Cache lifetime/eviction across programs (a prover service proving many programs
  vs one hot program). Bound VRAM for device-resident static trees.
- DECODE/REGISTER/PAGE program-keying: reuse the existing per-ELF commitment
  identity as the cache key.
- Interaction with continuation epochs (REGISTER's FINI variant,
  `register.rs:302`) — static set changes per epoch boundary.
