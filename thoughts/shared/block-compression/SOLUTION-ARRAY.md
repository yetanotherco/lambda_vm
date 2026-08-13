# The solution array — memory × throughput exploration plan

**Mandate (Mauro, 2026-08-13):** "I don't mind each solution tbh — batched FRI first is
fine, improving disk spill is fine, cleverly sending tables to the 5090 and keeping
others in memory is fine too. Make a plan to explore the solution array and get some
conclusions."

**The question this campaign answers:** what is the production wrap-prover configuration
at the endgame budget (5090 mandatory; **64 GiB RAM preferred, 128 acceptable**),
minimizing GPU-hours per block — and in what order should the remaining levers be built?

## 1. The array

| # | Lever | What it does | Status | Build effort |
|---|---|---|---|---|
| A | S3 host-recompute (Phase A+B) | drop LDE after root; recompute on CPU; free dead aux | **LANDED** (4 commits, oracles green) | — |
| B | S3 **device**-recompute (Phase A2) | same seam; re-expand into VRAM; one NTT on card | designed | **M (small)** — delta over A |
| C | Disk spill (traces+trees) | mmap page-out; measured ladder exists | **LANDED** (c5ffadf3) | — |
| C+ | **LDE spill** (new) | mmap-backed `LDETraceTable` — page LDEs out instead of recomputing them | not built | M |
| D | VRAM residency scheduling | admission gate + `device_only` + threshold lever ("send some tables to the 5090, keep others in RAM") | exists as knobs (gate, threshold, per-table heuristics) | S per-heuristic |
| E | Batched FRI | one FRI for all tables: 2.0-2.8× fewer leg perms | scoped (port #768 primitives) | M |
| F | Batched MMCS | shared commitment trees: +1.3× | scoped (same port) | M (with E) |
| G | P-a blake3-6r inner | ÷~4 on everything | in flight (separate track) | — |
| H | TABLE_PARALLELISM / k | measured: k≥4 saturates time; k=1 minimizes memory | exists | — |

Existing evidence folded in (NOT re-measured): the spill ladder (CENSUS Part 3), the S3
CPU cost (+7.8% single-pair; the mission's box ladder refines it), the GPU threshold
lever (−57% on the fixture wrap), the MMCS/FRI projections (unit-exact model), the
Gate A/D1 censuses.

## 2. The benchmark protocol (common to every cell)

- **Two fixed points**: MID = the largest epoch the current 60 GiB box completes
  (mission Phase 3 determines it); LARGE = the largest epoch the Japan box (258 GiB /
  5090 / 2.8 TB NVMe) completes. Same block (25368371), same inner params
  (blowup4/110q), same commit.
- **Metrics per cell**: peak host RSS (`time -v`) + peak anon, peak VRAM (1 Hz sampler),
  wall, verify green + falsifications, spill/disk volume, and the derived
  **$/wrap at vast prices**.
- **Repeat policy**: single run to place a cell; ABBA pairs only where two cells land
  within 15% of each other AND the difference would change a conclusion.
- **Fit verdicts judged against 64 and 128 GiB**, not the box's actual RAM.

## 3. The rounds

### Round 1 — measure what exists (no new code)
The matrix on both points: {Retain+GPU+threshold-lever, Retain+GPU+gate-default,
A (cpu recompute), A+C (recompute+spill), C alone+TP1, D variants (threshold sweep ×
device_only on/off)} × {k=1, k=4}. ~12-16 cells, most are minutes each. The mission's
Phase-3 config table seeds this; Round 1 completes it on the Japan box.
**Interim conclusion 1:** the best NO-NEW-CODE config at 64 and at 128 GiB, and the gap
to close (if any).

### Round 2 — the head-to-head the array actually turns on: B vs C+
Both attack the same buffer (the LDE) by opposite means: **recompute it on the GPU** vs
**page it to NVMe**. Build both (each M), measure at both points, same matrix slots.
Decision rule, stated now: **if B holds VRAM under budget via the admission gate and
lands within 15% wall of the best Round-1 config, B is the production mode and C+ is
discarded for the hot path** (kept only if B fails on VRAM pressure or the #927 cliff
class resurfaces). If both fail at 64 GiB, the trace side (S6 lazy traces) joins Round 2.

### Round 3 — the throughput lever: batched FRI (E), then MMCS (F) if E confirms
Port #768's primitives per MMCS-PLAN (M-12 terminal-poly fix + M-13 width absorption
included; streaming-per-matrix acceptance test mandatory). Measure the SAME matrix
winner ± batching. Decision rule: **batching ships if it improves $/wrap ≥2× at the
LARGE point** (the projection says 2-2.8× for E alone; a measured <1.5× means the model
missed something — stop and reconcile before F).

### Round 4 — conclusions document
- The Pareto table (memory × wall × $/wrap) across all measured cells.
- **The production recommendation**: one named config for 64 GiB and one for 128 GiB,
  each with its measured numbers and its failure modes.
- The discard list — levers measured and retired, with the number that retired them.
- The build order for whatever remains (e.g., "E after G lands; F with E; C+ retired").

## 4. Sequencing against in-flight work

- Round 1 starts when the Japan box lands (mission Phase 3 seeds it from the current box
  meanwhile). No code, no worktree contention.
- Round 2's builds queue on the branch AFTER the mission's commits (same worktree);
  B before C+ (B is the smaller delta and the posture favorite).
- Round 3 serializes with P-a Stage 2 (both rewrite fri/ — MMCS-PLAN M-3's rule).
- P-a (G) proceeds independently; every Round re-runs its winner under G when G lands
  (the multipliers compose, the ORDER of winners shouldn't change — if it does, that is
  itself a finding).

## 5. What would change the plan

- The mission's Phase-3 numbers landing far from the census model (>2×) → re-anchor
  before Round 1.
- The M-11 reconciliation (−57% vs −76.7%) resolving AGAINST the model → shrink Round-3
  expectations before building.
- A 64-GiB fit from Round 1 alone → Rounds 2-3 become pure economics, run at lower
  priority behind the tower.
