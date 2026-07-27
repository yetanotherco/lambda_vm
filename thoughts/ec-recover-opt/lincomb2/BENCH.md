# Phase H — bench harness

**Preparation only. Nothing here has been run.** Benches in this project run on
the remote bench server; a local run on a box that has already OOMed on
ethrex-20tx produces a misleading number.

The harness is built. Below: what each measurement answers, the exact commands
to paste on the bench server, what a good result looks like, and what would
falsify the claim.

## 0. The headline, and the number NOT to quote

> **The claim to confirm is −61.9% EC cells at the mean, against the live
> post-pairing baseline.**
>
> | | rows | cells/ecrecover | vs baseline |
> |---|---|---|---|
> | baseline (4× `ecsm_mul`, post-pairing `42ba68ff`) | 4 × 382 | **1.467M** | — |
> | lincomb2, mean | **449.1** | **0.559M** | **−61.9%** |
> | lincomb2, worst case | **514** | **0.640M** | **−56.4%** |

**Do not quote DESIGN's −74.3%, and do not quote IMPL-PLAN §9's −70.2% either.**
Both are stale, for two different reasons:

- **−74.3%** is denominated against the *pre-pairing* 1.69M baseline. The
  AreBytes pairing already shipped in `42ba68ff`, so quoting it re-banks a win
  that is already in the bank.
- **−70.2%** (IMPL-PLAN §9, and its 0.437M figure) is *post-pairing but
  pre-`D_INV`*. The non-degeneracy relation that closes the degenerate-add
  forgery landed afterwards and costs ~129 logic columns plus ~96 interactions
  per ECDAS2 row. **§9 should be updated**; I have not edited it (not mine).

Recomputing from the chips as they stand:

```
ECDAS2 : 658 logic + ⌈388 interactions⌉ ×1.5  = 658 + 582 = 1,240 cells/row
mean   : 449.1 × 1,240 + ECSM2 (1 row, ~2.9k) = 0.559M
worst  : 514   × 1,240 + ECSM2                = 0.640M
baseline: 4×382×956 + 4×1,439                 = 1.466M
```

−61.9% is still better than 3× the 2× bar, so the engineering verdict does not
change — but the honest headline moved, and it moved because we bought
unconditional soundness with it. That trade is worth stating explicitly when the
number is reported.

## 1. What exists

| | path | what it is |
|---|---|---|
| guest | `executor/programs/bench/ecrecover/` | drives the **real** shipping path: `LambdaVmEcsmCrypto::secp256k1_ecrecover` → guest `(r,v)` decompression → `ecsm_lincomb2` → ECSM2/ECDAS2 → `keccak256(pk)`. Not a synthetic ladder. |
| driver | `thoughts/ec-recover-opt/gen_ecrecover_bench.sh` | `build` / `input` / `cells` / `slope` / `share` |
| (existing) | `thoughts/ec-recover-opt/gen_ec_bench.sh` | the *synthetic* ECSM ladder from the pairing work — left alone |

**One ELF serves every configuration.** The workload is selected by private
input, so an A/B never compares two different binaries:

```
byte 0     case: 0 = mean corpus (8 real RFC 6979 signatures, cycled)
                 1 = worst case (u1,u2) = (2^255, 2^255−1), 514 chain rows
bytes 1..3 signature count, little-endian u16
```

The guest commits the XOR of every recovered address — a deterministic function
of the workload, so **the two A/B arms must commit identical bytes.** That is
the cross-check that the switch did not change results.

The mean corpus is 8 real signatures whose chain lengths average **451.8 rows**
(population mean 449.1; an 8-sample mean scatters). Cycling 8 rather than
repeating 1 keeps the measurement near the mean instead of pinning it to one
signature's bit pattern.

### Build status

**The guest compiles and links for the real target** — 228,344-byte ELF at
`executor/program_artifacts/bench/ecrecover.elf`. Compiling is not benching, so
this was done locally; no measurement was taken.

Two things the build surfaced, both now fixed in-tree and worth knowing if the
guest is ever moved:

- A guest crate needs its own **`.cargo/config.toml`**. Without
  `--cfg getrandom_backend="custom"` the build dies in `getrandom` with
  *"target is not supported"* — `lambda-vm-syscalls` pulls it transitively. This
  guest uses the `bench/ecsm` form (no `[env]` block); the `rust/ethrex` form
  additionally hardcodes `--sysroot=/opt/lambda-vm-sysroot`, which is only
  needed for guests with C dependencies and would be a stale path elsewhere.
- If `/opt` is not writable, `make` wants `sudo` for the sysroot. The Makefile's
  own tip works: prefix with `SYSROOT_DIR=$HOME/.lambda-vm-sysroot`.

## 2. The A/B — and an honest problem with it

**Arm A (old path) cannot be built from the working tree.** As of 18:34 today,
`crypto/ethrex-crypto/src/lib.rs` contains zero occurrences of
`lincomb2_with_oracle`, `solve_y` or `ecsm_oracle` — phase G has switched the
guest. `HEAD` still has them. That file is phase G's; I did not touch it, and
I am not reconstructing the deleted code.

So the A/B is **across two checkouts, not two branches of one file** — which is
the honest form anyway, since it also captures the executor and chip changes:

```sh
# find the last commit that still had the old path
OLD=$(git log -1 --format=%H -S 'fn lincomb2_with_oracle' -- crypto/ethrex-crypto/src/lib.rs)
git worktree add /tmp/lincomb2-before "$OLD"

# the bench guest is new, so copy it into the old checkout (it calls only the
# stable `Crypto::secp256k1_ecrecover` entry point, which exists in both)
cp -r executor/programs/bench/ecrecover /tmp/lincomb2-before/executor/programs/bench/

( cd /tmp/lincomb2-before && make compile-bench )   # arm A
( cd .                    && make compile-bench )   # arm B
```

Then run §3 against each. If `$OLD` turns out to be the tip (phase G is still
uncommitted at bench time), arm A is simply `HEAD` — check before assuming.

**Shortcut**: phase G already saved both guest ELFs — `ethrex_old.elf` and
`ethrex_new.elf` in its scratchpad, verified md5-identical to the installed
build. If they are still around, they spare you the worktree for the *ethrex*
guest. They do not cover the `ecrecover` bench guest (which is new), so the
recipe above is still the reproducible form and the one to use if the artifacts
have been cleaned up.

**Cycle delta — use the measurement, not DESIGN's estimate.** Phase G measured
**−78,823 guest cycles per ecrecover** on the 5-transfer block and **−78,493**
on the 20-transfer, agreeing to 0.4%. DESIGN predicted 100–150k; the measurement
is **~21% below the low end**. Quote ~78.5k. The secondary win is real but
smaller than advertised, and that belongs in the writeup rather than being
rounded up to the prediction.

**CI caveat for any before/after that uses a cached ELF.** Five `pr_main.yaml`
cache keys hashed `executor/programs/rust/**` and `syscalls/**` but not
`crypto/ethrex-crypto/**`, which the ethrex guest depends on by path — so CI
would restore a stale `ethrex.elf` and keep proving the old x-only path. Fixed
(cache-key only), but any CI-side comparison taken before that fix landed was
measuring the old path on both arms. Re-run rather than trust such a number.

**If you would rather not carry a second checkout**, the fallback is a
documented before-number: the baseline row/cell counts are already measured and
recorded (`chips-map.md` census, gate-confirmed), so arm B alone plus the 1.467M
constant answers the question. That is weaker — it infers rather than measures
the baseline — and should be labelled as such if used.

## 3. Commands to paste on the bench server

```sh
cd <repo>
SYSROOT_DIR=$HOME/.lambda-vm-sysroot \
  ./thoughts/ec-recover-opt/gen_ecrecover_bench.sh build   # omit SYSROOT_DIR if /opt is writable

# (a) the headline: cells per ecrecover, from a two-point slope.
#     The slope cancels the fixed overhead (CPU floor, EC_T0, preprocessed
#     tables), which a single-point measurement would wrongly attribute to EC.
./thoughts/ec-recover-opt/gen_ecrecover_bench.sh slope mean  64 512

# (b) the adversarial shape — 514 rows, not the mean
./thoughts/ec-recover-opt/gen_ecrecover_bench.sh slope worst 64 512

# (c) the always-on cost of EC_T0 and the rest of the fixed floor:
#     n = 1 is (almost) all overhead; compare against the slope's intercept
./thoughts/ec-recover-opt/gen_ecrecover_bench.sh cells mean 1

# (d) the EC share of a real block — the multiplier that turns −61.9% EC cells
#     into a whole-prover number. MUST be a real block: see section 4.1.
(cd thoughts/ec-recover-opt/bench-harness && cargo build --release)
./thoughts/ec-recover-opt/bench-harness/target/release/ec-bench-harness \
    executor/program_artifacts/rust/ethrex.elf <block-fixture.bin>
```

(d) prints every table's committed base cells sorted by cost, marks the EC
tables, and reports the share directly — no `n_ecrecover` bookkeeping. The
harness is standalone, so its first build compiles the prover into its own
target dir (one-time, minutes).

`slope`, `cells` and `share` all use `cli count-elements`, which builds the trace
and counts committed field elements **without running the STARK proof** — so
they are minutes, not hours, and are not memory-bound the way proving is. Only
run a full `prove` if you want wall-clock as well; the cell count is what the
−61.9% claim is about.

Total committed base cells = `main + 3 × aux_ef_cols` (one extension element is
3 base elements; LogUp packs two interactions per aux column, which is the same
1.5-base-cells-per-interaction rule the cost model uses).

## 4.1 The EC share is ONLY meaningful on a real ethrex block

**Read this before quoting any EC-share number.** Two of the VM's tables are
**fixed size regardless of workload** — BITWISE is a flat 2^16 rows, and PAGE is
fixed per page. On a small guest they swamp everything, and the EC share you
measure is an artefact of the harness rather than a property of the design:

```
test_ecsm_lincomb2_full   19.3M cells   EC share  4.59%   (BITWISE 81.6%, PAGE 13.6%, ECDAS2 4.5%)
fib_iterative_372k        61.1M cells   EC share  0.03%
```

Neither number is interpretable. On a real block, CPU and MEMW scale with the
work and those fixed costs amortise away.

**Reading 4.59% off a unit test and reporting it as the whole-prover multiplier
would understate the win by an order of magnitude**, and it would be very hard
to unpick later — the number looks like an answer. This is the same family as
the fallback trap in §5: a harness artefact wearing the shape of a result.

So: run §3(d) against `ethrex.elf` with a real block fixture, and if you only
have a unit-test guest to hand, **report no share at all** rather than a small
one.

A smoke run of the §3(d) harness on `ecsm.elf` (a trivial unit-test guest, run
only to check the tool prints — **not** a measurement) reproduces the artefact
exactly: BITWISE 78.34% + PAGE 21.15% = 99.5% of the trace, every EC table under
0.03%. That output is worthless as a share and useful only as a demonstration of
why.

**One thing to confirm on the real block rather than assume**: the per-table view
also reports `EC_T0` directly, which answers §3(c)'s always-on question — but on
the smoke run it shows ~1k cells, far below the 256 rows × ~66 columns the table
holds. That suggests preprocessed columns are accounted differently from witness
columns (committed once, not per proof). Check how `count_elements_by_table`
treats preprocessed tables before concluding EC_T0 is free.

## 4. What a good result looks like

| measurement | expected | tolerance |
|---|---|---|
| (a) mean slope | **≈ 0.559M cells/ecrecover**, **−61.9%** vs 1.467M | ±3% on the slope |
| (b) worst slope | **≈ 0.640M**, **−56.4%** | ±3% |
| (b)/(a) ratio | **≈ 1.145** (514/449.1) | ±2% |
| (c) n=1 overhead | small vs 512× the slope; EC_T0 contributes 256 rows × ~66 cols | — |
| A/B committed output | **byte-identical between arms** | exact |

The ratio check in row 3 is the cheapest sanity test: it depends only on row
counts, not on the cell model, so if it holds while the absolute numbers miss,
the row schedule is right and the cell model is wrong.

## 5. What would falsify the claim

- **Mean slope materially above ~0.6M.** That is the cell model being wrong, not
  noise. Report it plainly rather than reframing the target — the same rule
  IMPL-PLAN §9 states.
- **Slope ≈ 1.4M, i.e. no improvement.** Almost certainly the *fallback trap*
  below rather than a real result.
- **Ratio (b)/(a) ≉ 1.145.** The worst case is not costing what the row model
  says, so the row model is wrong somewhere.
- **A/B outputs differ.** A correctness regression in the switch — stop and
  escalate; that outranks any performance number.
- **n=1 overhead comparable to the per-signature slope.** The always-on EC_T0 +
  preprocessed floor would then be a real cost on every proof, EC or not, and
  IMPL-PLAN §9's `include_halt` precedent for a conditionally-included AIR
  becomes live.

### The fallback trap — read this before trusting any number

`LambdaVmEcsmCrypto` falls back to pure-Rust `ProjectivePoint::lincomb` whenever
the accelerator returns a non-zero status. **The fallback returns the same
answer**, so the committed output cannot detect it — a silently-degraded run
looks correct and benches the wrong path.

Two ways to catch it, both cheap:

1. **The slope.** The software path spends its cost in CPU rows, not EC rows.
   A slope anywhere near the baseline, or a mean/worst ratio near 1.0 (the
   software path does not care about joint digit density), means the precompile
   is not being exercised.
2. **Force the issue.** Temporarily make the fallback panic in a scratch build
   and confirm the bench still runs. Do not commit that.

## 6. What I could not do from here

- ~~A per-table cell breakdown would make §3(d) exact.~~ **CLOSED.**
  `prover::count_elements_by_table(elf, private_inputs) -> Vec<(&'static str,
  u64, u64)>` now exists (`prover/src/lib.rs:1075`), so §3(d) is direct rather
  than inferred. One wrinkle: it is a **library function with no CLI
  subcommand**, so `cli count-elements` cannot reach it. Rather than edit shared
  code I added `thoughts/ec-recover-opt/bench-harness/` — a standalone crate
  (same pattern as `oracle/repo-harness/`) that calls it and prints the sorted
  breakdown plus the EC share. **If you would rather have it in the CLI, a
  `CountElementsByTable` subcommand is ~15 lines and would build against the
  shared target dir instead of a private one** — worth doing if this gets used
  more than once.
- **IMPL-PLAN §9's targets are stale** (§0 above). Not my file to edit.
- **The guest joins `make compile-bench`**, which `make compile-programs` and
  hence `make test` depend on. It pulls the same pinned ethrex git rev the
  `ethrex` guest already uses, so the marginal build cost should be small — but
  it is a shared-build side effect, and worth a look before this lands.

## 7. Provenance of the bench vectors

Generated with the phase-D0 oracle (`thoughts/ec-recover-opt/oracle/`), not by
hand: 8 RFC 6979 signatures from the `ecdsa` package, recid chosen by checking
which parity recovers the signer, and the worst case constructed by fixing
`(u1, u2) = (2^255, 2^255 − 1)`, picking `R = 3·G`, and back-solving
`z = −u1·r`, `s = u2·r`. The round trip
(`u1 = −z/r`, `u2 = s/r`, `R = lift_x(r)`) is asserted in the generator, so the
guest's own decomposition reproduces the intended scalars.
