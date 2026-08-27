//! ★ THE WRAP — the assembled epoch verifier PROVED, not just executed.
//!
//! [`super::epoch_verify_tests`] runs the whole epoch verifier under
//! [`super::executor::execute`]: every check is an assert inside the program, so
//! reaching the end of the execution is the verification passing. What that says
//! nothing about is the CHIPS (standing-decisions method rule 2 — where the
//! executor mirrors a computation the chip also does, only a prove+verify test
//! sees the chip). This module is the wrap: the same program, the same real
//! epoch, through [`lfm_prove`] and [`verify_against`].
//!
//! ## What the numbers here are, and are not
//!
//! Every cost figure names the epoch's trace-length profile (assembly ledger
//! entry 10). Two different epoch shapes are involved and they must never be
//! conflated:
//!
//! - the INNER epoch's shape — the proof being verified: its per-table trace
//!   lengths, its blowup and its query count. This is what makes the emitted
//!   verifier program big or small.
//! - the WRAP's own [`ProofOptions`] — blowup 2, the same options every other
//!   LFM prove test uses. It fixes what proving the verifier costs, not what the
//!   verifier does.
//!
//! ## What this module cannot see
//!
//! The hash. Every permutation here is `TestPermutation` inside the LFM chips
//! plus the production keccak family hosted for `keccak256`; the point of
//! measuring cells at all is to have the first column of a matrix whose other
//! columns (blake, Poseidon) do not exist yet. It also cannot see prove time or
//! peak memory as a property of the machine — those are measured around the
//! process, by the harness that runs it, and are reported as observations of one
//! box rather than as machine invariants.

use std::time::Instant;

use stark::proof::options::{GoldilocksCubicProofOptions, ProofOptions};

use super::airs::{HeightRule, LfmChipCells, lfm_cell_counts, lfm_chip_census};
use super::compiler::LfmProgram;
use super::edsl::WrapHash;
use super::epoch_tests::EpochInputs;
use super::executor::execute;
use super::hash::TestPermutation;
use super::instr::Instr;
use super::proof::{LfmProveError, lfm_prove, lfm_prove_with_residency, verify_against};
use super::registry::build_artifacts;

use crate::tables::types::FE;

/// The WRAP proof's own options: blowup 2, the framework's 128-bit query count.
///
/// Deliberately the same `prove_options()` every leg suite proved under
/// (`join_tests`, `fri_tests`, `constraint_tests`), so a wrap cost is comparable
/// with a leg cost. The inner epoch's options are a different thing entirely and
/// are named per measurement.
fn wrap_options() -> ProofOptions {
    GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 is valid")
}

/// Instructions of each kind, for the shape line every measurement prints.
fn instruction_mix(program: &LfmProgram) -> String {
    let count = |f: fn(&Instr) -> bool| program.instrs.iter().filter(|i| f(i)).count();
    format!(
        "const {} / base-alu {} / ext-alu {} / select {} / bitdec {} / hash {} / \
         keccak {} / hint {} / pack {} / unpack {} / public {}",
        count(|i| matches!(i, Instr::Const { .. })),
        count(|i| matches!(i, Instr::BaseAlu { .. })),
        count(|i| matches!(i, Instr::ExtAlu { .. })),
        count(|i| matches!(i, Instr::Select { .. })),
        count(|i| matches!(i, Instr::BitDec { .. })),
        count(|i| matches!(i, Instr::Hash { .. })),
        count(|i| matches!(i, Instr::KeccakF(_))),
        count(|i| matches!(i, Instr::Hint { .. })),
        count(|i| matches!(i, Instr::Pack { .. })),
        count(|i| matches!(i, Instr::Unpack { .. })),
        count(|i| matches!(i, Instr::Public { .. })),
    )
}

/// Keccak permutations a program requests.
pub(super) fn permutations(program: &LfmProgram) -> usize {
    program
        .instrs
        .iter()
        .filter(|i| matches!(i, Instr::KeccakF(_)))
        .count()
}

/// The production wrap hash's own operations in a program — keccak
/// permutations or BLAKE3 compressions, whichever `hash` names. The census's
/// closed-form check compares like with like: `query_permutations_for` counts
/// compressions under the same hash's block rule.
pub(super) fn hash_ops(program: &LfmProgram, hash: WrapHash) -> usize {
    program
        .instrs
        .iter()
        .filter(|i| match hash {
            WrapHash::Keccak => matches!(i, Instr::KeccakF(_)),
            WrapHash::Blake3 => matches!(i, Instr::Blake3(_)),
            WrapHash::Algebraic => matches!(i, Instr::Hash { .. }),
        })
        .count()
}

/// Arena words a program declares.
pub(super) fn arena_words(program: &LfmProgram) -> usize {
    program.arena_schema.lens.iter().map(|l| *l as usize).sum()
}

/// A chip's headroom as the census table prints it: a percentage for the chips
/// that can actually grow into it, and the reason otherwise.
fn headroom_cell(c: &LfmChipCells) -> String {
    match c.height_rule {
        HeightRule::Workload => format!("{:.1}%", 100.0 * c.headroom()),
        HeightRule::Fixed => "fixed".to_string(),
        HeightRule::Chunked => "chunked".to_string(),
    }
}

/// ★ THE ROW-CLIFF PANEL — what the census could not say before.
///
/// The wrap's cell count is a STEP function of the instruction mix: every
/// workload-sized chip commits `real_rows.next_power_of_two()` rows, so a chip
/// sitting just under a step doubles its entire contribution on the next 1% of
/// growth, while one just over a step absorbs a near-doubling for free.
///
/// The census has always computed both heights and printed only the padded one,
/// which is why #903's cost was invisible until a rented box measured it: four
/// near-empty BLAKE3 rows grew the mix ~21%, and the campaign happened to be
/// standing 1.2% under a step on `LFM_LANES` and within 18% on four more chips,
/// so that growth tripped FIVE simultaneous doublings and turned +21% of
/// permutations into +31% of cells
/// (`thoughts/shared/block-compression/WRAP-GROWTH-BISECT.md`).
///
/// This panel makes that legible at emission time. It reports the tightest
/// chips, what crossing each would cost, and the total exposure — the cells the
/// wrap would gain if every chip within the warning band crossed at once, which
/// is exactly the quantity Stage 4 spent without anyone pricing it.
fn report_row_cliffs(census: &[LfmChipCells]) {
    /// Headroom under which a chip is called out. A change of this order is
    /// routine — Stage 4's was ~21% — so anything inside the band should be
    /// read as "will cross on the next ordinary change", not as a safe margin.
    const WARN: f64 = 0.20;

    let total: u64 = census.iter().map(|c| c.cliff_cost()).sum();
    let mut at_risk: Vec<&LfmChipCells> = census.iter().filter(|c| c.at_risk()).collect();
    at_risk.sort_by(|a, b| a.headroom().total_cmp(&b.headroom()));

    println!("\n   ★ ROW-CLIFF PANEL — cells are a step function of the mix");
    if at_risk.is_empty() {
        println!("      no workload-sized chips in this census");
        return;
    }

    let exposed: Vec<&&LfmChipCells> = at_risk.iter().filter(|c| c.headroom() < WARN).collect();
    let pct = |n: u64| 100.0 * n as f64 / total as f64;

    for c in at_risk.iter().take(5) {
        let flag = if c.headroom() < WARN { "⚠" } else { " " };
        println!(
            "      {flag} {:>12} {:>7.1}% headroom ({} of {} rows) — crossing costs \
             {} base-field equivalents ({:.1}% of the total)",
            c.name,
            100.0 * c.headroom(),
            c.real_rows,
            c.rows,
            c.cliff_cost(),
            pct(c.cliff_cost()),
        );
    }

    let exposure: u64 = exposed.iter().map(|c| c.cliff_cost()).sum();
    println!(
        "      {} of {} workload-sized chips are within {:.0}% of a step; if all of them \
         crossed the wrap would gain {} base-field equivalents (+{:.1}%)",
        exposed.len(),
        at_risk.len(),
        100.0 * WARN,
        exposure,
        pct(exposure),
    );
}

/// ★ The registry-entry shape record: what the machine proves for one program.
///
/// Prints the chip census — one line per SUB-PROOF, since `KECCAK_RND`'s chunks
/// are separate AIRs at separate heights — the per-chip row-cliff headroom, and
/// the totals the hash matrix wants.
/// Returns `(main_cells, aux_cells)` so a caller can assert on them.
pub(super) fn report_census(label: &str, program: &LfmProgram) -> (u64, u64) {
    let census = lfm_chip_census(program);
    let (main, aux) = lfm_cell_counts(program);
    // The census is `lfm_cell_counts`' own decomposition, so summing it is not an
    // independent check of the total — it is the same arithmetic. What IS
    // independent is that the sub-proof COUNT the census implies must equal the
    // AIR count the verifier builds from the program's chunk policy.
    // Mask-aware, like the census itself: an absent family contributes no
    // AIRs, so the expected count comes from the program's own chip set and
    // its family-gated chunk count — `num_lfm_airs` is the FULL-mask count and
    // no real program is FULL (a program uses at most one hash family).
    let chip_set = super::airs::ChipSet::for_program(program);
    assert_eq!(
        census.len(),
        chip_set.num_airs(
            chip_set.keccak_rnd_chunks(
                program
                    .chunking
                    .chunk_count(program.groups.keccak.real_rows)
            ),
            program.blake3_chunk_count(),
        ),
        "the census must have one entry per sub-proof the AIR set builds"
    );
    println!("\n★ CHIP CENSUS — {label}");
    println!(
        "   {:>12} {:>10} {:>12} {:>9} {:>6} {:>6} {:>16} {:>14}",
        "chip", "rows", "used", "headroom", "main", "aux", "main cells", "aux cells"
    );
    for c in &census {
        println!(
            "   {:>12} {:>10} {:>12} {:>9} {:>6} {:>6} {:>16} {:>14}",
            c.name,
            c.rows,
            c.real_rows,
            headroom_cell(c),
            c.main_cols,
            c.aux_cols,
            c.main_cells(),
            c.aux_cells()
        );
    }
    println!(
        "   {:>12} {:>10} {:>12} {:>9} {:>6} {:>6} {:>16} {:>14}",
        "TOTAL",
        census.iter().map(|c| c.rows).sum::<u64>(),
        census.iter().map(|c| c.real_rows).sum::<u64>(),
        "",
        "",
        "",
        main,
        aux
    );
    report_row_cliffs(&census);
    println!(
        "   cells per verify = {main} main + {aux} aux ext = {} base-field equivalents \
         (an ext element is 3 base felts)",
        main + 3 * aux
    );

    // ★ THE HASH SHARE — what the matrix is about.
    //
    // The whole reason to count cells is to price the hash, so the census says
    // outright how much of the machine IS the hash. `LFM_KECCAK` (the adapter row
    // that requests a permutation) and `KECCAK_RND` (its 24 rounds) are the
    // permutation itself; `KECCAK_RC` and `BITWISE` are the lookup tables it reads,
    // and they are reported separately because they are FIXED-height — a different
    // hash would delete the first pair and shrink but not necessarily remove the
    // second.
    let share = |names: &[&str]| -> (u64, u64) {
        census
            .iter()
            .filter(|c| names.contains(&c.name))
            .fold((0u64, 0u64), |(m, a), c| {
                (m + c.main_cells(), a + c.aux_cells())
            })
    };
    let (perm_main, perm_aux) = share(&["LFM_KECCAK", "KECCAK_RND"]);
    let (tab_main, tab_aux) = share(&["KECCAK_RC", "BITWISE"]);
    let total = (main + 3 * aux) as f64;
    println!(
        "   keccak permutation chips (LFM_KECCAK + KECCAK_RND): {perm_main} main + \
         {perm_aux} aux = {:.1}% of cells\n   \
         its lookup tables (KECCAK_RC + BITWISE, fixed height): {tab_main} main + \
         {tab_aux} aux = {:.1}%\n   \
         everything else (the verifier's own arithmetic): {:.1}%",
        100.0 * (perm_main + 3 * perm_aux) as f64 / total,
        100.0 * (tab_main + 3 * tab_aux) as f64 / total,
        100.0 * (main + 3 * aux - perm_main - 3 * perm_aux - tab_main - 3 * tab_aux) as f64 / total,
    );
    println!("   instruction mix: {}", instruction_mix(program));
    (main, aux)
}

/// The three headline shape numbers, printed with the epoch profile that fixes
/// them (ledger entry 10).
pub(super) fn report_program(label: &str, profile: &str, program: &LfmProgram) {
    println!(
        "\n★ {label}\n   epoch trace lengths (log2): {profile}\n   \
         {} instructions / {} keccak permutations / {} arena words / {} chunks",
        program.instrs.len(),
        permutations(program),
        arena_words(program),
        program
            .chunking
            .chunk_count(program.groups.keccak.real_rows),
    );
}

/// The INNER epoch's own committed trace cells — `(main, aux ext)` — summed over
/// its sub-proofs.
///
/// The denominator of the recursion ratio, and the only honest one available from
/// shapes alone: `rows x main_width` and `rows x aux_width` per sub-proof, which is
/// the same accounting [`lfm_chip_census`] applies to the machine (value columns
/// plus aux, one ext element per aux column per row).
///
/// What it CANNOT see, on both sides equally: preprocessed columns, the
/// composition polynomial's own commitment, the LDE, and the Merkle trees. So the
/// ratio it feeds is "trace cells to verify one epoch's trace cells", not "total
/// prover work", and it is quoted that way.
fn inner_epoch_cells(e: &super::epoch_tests::RealEpoch) -> (u64, u64) {
    e.legs
        .iter()
        .map(|l| {
            let rows = 1u64 << l.verify.sub.deep.log2_trace_length;
            let aux_width = l.verify.sub.deep.num_total_cols - l.verify.main_width;
            (rows * l.verify.main_width as u64, rows * aux_width as u64)
        })
        .fold((0, 0), |(m, a), (dm, da)| (m + dm, a + da))
}

/// Prints the recursion ratio: machine cells per verify against the verified
/// epoch's own cells. The kill-risk-3 question, asked of a real epoch at last.
fn report_ratio(e: &super::epoch_tests::RealEpoch, main: u64, aux: u64) {
    let (inner_main, inner_aux) = inner_epoch_cells(e);
    let inner = inner_main + 3 * inner_aux;
    let outer = main + 3 * aux;
    println!(
        "   the epoch VERIFIED carries {inner_main} main + {inner_aux} aux ext = {inner} \
         base-field-equivalent trace cells\n   \
         so verifying it costs {:.1}x its own trace cells (trace-to-trace; neither \
         side counts preprocessed columns, LDEs or trees)",
        outer as f64 / inner as f64,
    );
}

/// The epoch's trace-length profile as ledger entry 10 wants it printed.
pub(super) fn epoch_profile(e: &super::epoch_tests::RealEpoch) -> String {
    let mut lengths: Vec<u32> = e
        .legs
        .iter()
        .map(|l| l.verify.sub.deep.log2_trace_length)
        .collect();
    lengths.sort_unstable();
    let mut runs: Vec<String> = Vec::new();
    for len in &lengths {
        let n = lengths.iter().filter(|l| *l == len).count();
        let entry = if n > 1 {
            format!("{len} x{n}")
        } else {
            format!("{len}")
        };
        if !runs.contains(&entry) {
            runs.push(entry);
        }
    }
    format!("[{}]", runs.join(", "))
}

/// ★ SLICE 0 — the wrap on the min-preset fixture epoch: prove, verify, tamper.
///
/// `#[ignore]`d, and the reason is the cost: the assembled verifier is ~2.25M
/// instructions, so the LFM traces are an order of magnitude past anything else
/// in this suite and the run is minutes of CPU and tens of gigabytes. It is the
/// wrap run's own harness, not a test the suite can afford on every PR.
///
/// Run with:
/// `cargo test --release -p lambda-vm-prover --lib lfm::wrap_tests::the_wrap_proves_and_verifies -- --ignored --nocapture`
#[test]
#[ignore]
fn the_wrap_proves_and_verifies() {
    wrap_run(super::proof_fixture::fixture_options());
}

/// ★ SLICE 0's GPU-dispatch census (`thoughts/shared/gpu-recursion/EXPLORATION.md`,
/// Stage 0). The min-preset wrap proved once, with the process-global GPU call
/// counters reset right before `lfm_prove` — after the inner epoch is built,
/// because building it proves an RV64 continuation whose own GPU traffic (the
/// VM's preprocessed tables clear the size gate even for a 16-cycle epoch) would
/// otherwise pollute the machine's numbers. Prints every counter rather than
/// asserting floors: this is the falsification harness for the GPU map, and the
/// predictions are the document's to state, not the test's to freeze. Needs
/// `--test-threads=1` (the counters are process-global) and, like the rest of
/// the cuda suite, `--ignored` so the no-GPU CI path keeps skipping it.
#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn the_wrap_reports_gpu_counters() {
    use stark::gpu_lde as g;

    let e = super::epoch_tests::real_epoch_with(super::proof_fixture::fixture_options());
    let program = super::epoch_tests::epoch_program(&e, true);
    let arenas = super::epoch_tests::epoch_arena_words(&e, true);
    let opts = wrap_options();
    let artifacts = build_artifacts(&program, &opts);
    println!("   chip log-heights: {:?}", artifacts.log_heights);

    g::reset_all_gpu_call_counters();
    let t = Instant::now();
    let proved = lfm_prove(&program, &artifacts, &arenas, &opts).expect("the wrap must prove");
    let prove_secs = t.elapsed().as_secs_f64();
    println!(
        "\n★ GPU DISPATCH COUNTERS (min-preset wrap, lfm_prove only, {prove_secs:.1}s):\n   \
         lde {} / leaf_hash {} / merkle_tree {} / extend_halves {} / logup {}\n   \
         composition {} / comp_poly_tree {} / parts_lde {} / bary {} / deep {}\n   \
         batch_invert {} / fri {} / opening_gather {} / device_only {}",
        g::gpu_lde_calls(),
        g::gpu_leaf_hash_calls(),
        g::gpu_merkle_tree_calls(),
        g::gpu_extend_halves_calls(),
        g::gpu_logup_calls(),
        g::gpu_composition_calls(),
        g::gpu_comp_poly_tree_calls(),
        g::gpu_parts_lde_calls(),
        g::gpu_bary_calls(),
        g::gpu_deep_calls(),
        g::gpu_batch_invert_calls(),
        g::gpu_fri_calls(),
        g::gpu_opening_gather_calls(),
        g::gpu_device_only_calls(),
    );
    assert!(
        verify_against(
            &artifacts.roots,
            &artifacts.program_id,
            artifacts.keccak_rnd_chunks,
            &proved.proof,
            &proved.public_words,
            &opts,
            artifacts.hasher,
            artifacts.chip_set,
        ),
        "the wrap proof must verify"
    );
}

/// ★ SLICE 1 (local rung) — the wrap at the inner proof's BLOWUP-8 GEOMETRY.
///
/// The standing decision is that the inner proof is at blowup 8, and blowup is
/// not a rescaling of blowup 2: the LDE is four times deeper, so every Merkle walk
/// climbs two more levels, the FRI chain commits more layers, and the terminal
/// polynomial is reached from further away. None of that is exercised by slice 0.
///
/// The QUERY count is the one thing reduced, and reduced for a stated reason: at
/// the real 73 queries the wrap's own trace does not fit in any box we have (see
/// [`the_wrap_census_at_blowup_8`], which measures the program and prints what
/// proving it would need). ONE query is what a 36 GiB local box holds, and it is
/// enough to make every blowup-8 structure real — the deeper walk, the longer fold
/// chain, the terminal polynomial reached from further away — since what falls out
/// at one query is only the REPETITION of that structure. An honest partial: the
/// GEOMETRY is proved, the query COUNT is not, and the two are separable because
/// per-query cost is a closed form over the shapes that
/// [`the_wrap_census_at_blowup_8`] asserts the emitted program against.
/// `LFM_WRAP_QUERIES` raises the inner query count above the 1 this asserts at,
/// which is how the residency ladder walks the wrap up until a box refuses it.
/// Unset — every CI and local run — it is exactly the one-query test described
/// above.
#[test]
#[ignore]
fn the_wrap_proves_at_blowup_8_geometry() {
    let queries = match std::env::var("LFM_WRAP_QUERIES") {
        Ok(v) => v.parse().expect("LFM_WRAP_QUERIES must be an integer"),
        Err(_) => 1,
    };
    wrap_run(inner_blowup_8_with_queries(queries));
}

/// The inner proof's blowup-8 options with the query count overridden.
///
/// NOT a security parameter set at anything below 73 queries, and never used as
/// one: the query count is what this reduces and every measurement taken under it
/// says so in its label.
fn inner_blowup_8_with_queries(queries: usize) -> ProofOptions {
    let mut o = crate::recursion::Preset::Blowup8.options();
    o.fri_number_of_queries = queries;
    o
}

/// The wrap, end to end, under supplied INNER proof options, over whatever epoch
/// [`EpochInputs::from_env`] names — the fibonacci fixture unless a measurement
/// run overrode it.
fn wrap_run(inner: ProofOptions) {
    wrap_run_from(inner, EpochInputs::from_env());
}

/// [`wrap_run`] over an explicitly supplied epoch: build it, emit the verifier,
/// prove it, verify it, and run the three falsifications.
fn wrap_run_from(inner: ProofOptions, inputs: EpochInputs) {
    let t_epoch = Instant::now();
    let e = super::epoch_tests::real_epoch_from(inner.clone(), inputs);
    let profile = epoch_profile(&e);
    println!(
        "inner epoch: {} sub-proofs, blowup {}, {} quer{} per table, grinding {} — built in {:.1}s",
        e.legs.len(),
        1 << e.tables[0].shape.log2_blowup,
        e.legs[0].verify.num_queries,
        if e.legs[0].verify.num_queries == 1 {
            "y"
        } else {
            "ies"
        },
        e.tables[0].shape.grinding_factor,
        t_epoch.elapsed().as_secs_f64()
    );

    // The GEOMETRY the blowup fixes, stated per run: what the walks climb and what
    // the FRI chain folds. This is what separates a blowup-8 run from a blowup-2 one
    // at the same query count, so it is printed rather than left to the label.
    let big = e
        .legs
        .iter()
        .max_by_key(|l| l.verify.sub.deep.log2_trace_length)
        .expect("the epoch has sub-proofs");
    println!(
        "   geometry: widest sub-proof 2^{} trace -> 2^{} LDE, {} Merkle levels per group, \
         {} committed FRI layers ({} across the epoch); widest leaf {} bytes",
        big.verify.sub.deep.log2_trace_length,
        big.verify.sub.log2_lde_length,
        big.verify.sub.merkle_depth,
        big.verify.fri.num_committed(),
        e.legs
            .iter()
            .map(|l| l.verify.fri.num_committed())
            .sum::<usize>(),
        e.legs
            .iter()
            .flat_map(|l| l.verify.sub.groups())
            .map(|g| g.leaf_bytes())
            .max()
            .expect("the epoch has groups"),
    );

    let program = super::epoch_tests::epoch_program(&e, true);
    let arenas = super::epoch_tests::epoch_arena_words(&e, true);
    report_program("THE WRAPPED PROGRAM", &profile, &program);
    let (main, aux) = report_census(
        &format!("assembled epoch verifier, epoch {profile}"),
        &program,
    );

    // ---- the spine/legs split, and the legs' permutations against a CLOSED FORM.
    //
    // Both halves matter and for different reasons. The split is what makes two
    // runs at different query counts comparable at all: the SPINE also grows with
    // the query count (it samples an index per query, and every sample is
    // transcript work), so "permutations per query" taken from the total is wrong
    // and taken from the difference is right. The closed form is the absolute
    // check rule 7's refinement demands — `query_permutations` is arithmetic over
    // byte widths and tree depths, not a second pass of this emitter, so a leg
    // that quietly stopped hashing a group fails here rather than printing a
    // smaller number.
    let spine = super::epoch_tests::epoch_program(&e, false);
    // The legs hash with the PRODUCTION wrap hash — keccak pre-flip, BLAKE3
    // after it — so both sides of the closed-form check follow it: the emitted
    // count is that hash's own instruction, and the prediction uses its block
    // rule. Counting keccak against a BLAKE3 leg reads 0 and fails spuriously.
    let wrap_hash = WrapHash::production();
    let leg_hash_ops = hash_ops(&program, wrap_hash) - hash_ops(&spine, wrap_hash);
    let predicted: usize = e
        .legs
        .iter()
        .map(|l| super::epoch_verify::query_permutations_for(&l.verify, wrap_hash))
        .sum();
    assert_eq!(
        leg_hash_ops, predicted,
        "the emitted leg {wrap_hash:?} operations must equal the closed form over the shapes"
    );
    let queries = e.legs[0].verify.num_queries;
    println!(
        "   spine {} instr / {} {:?} ops / {} words   legs {} / {} / {}   \
         per query: {:.1} ops ({} queries, closed form checked)",
        spine.instrs.len(),
        hash_ops(&spine, wrap_hash),
        wrap_hash,
        arena_words(&spine),
        program.instrs.len() - spine.instrs.len(),
        leg_hash_ops,
        arena_words(&program) - arena_words(&spine),
        leg_hash_ops as f64 / queries as f64,
        queries,
    );

    report_ratio(&e, main, aux);

    let opts = wrap_options();
    let artifacts = build_artifacts(&program, &opts);
    println!(
        "   wrap options: blowup {}, {} queries, grinding {}\n   chip log-heights: {:?}",
        opts.blowup_factor, opts.fri_number_of_queries, opts.grinding_factor, artifacts.log_heights
    );

    // ---- PROVE.
    let t = Instant::now();
    let proved = lfm_prove(&program, &artifacts, &arenas, &opts).expect("the wrap must prove");
    let prove_secs = t.elapsed().as_secs_f64();

    let size = rkyv::to_bytes::<rkyv::rancor::Error>(&proved.proof)
        .expect("the wrap proof must serialize")
        .len();

    // ---- VERIFY.
    let t = Instant::now();
    assert!(
        verify_against(
            &artifacts.roots,
            &artifacts.program_id,
            artifacts.keccak_rnd_chunks,
            &proved.proof,
            &proved.public_words,
            &opts,
            artifacts.hasher,
            artifacts.chip_set,
        ),
        "the wrap proof must verify"
    );
    let verify_secs = t.elapsed().as_secs_f64();
    println!(
        "\n★ WRAP PROVED AND VERIFIED (inner epoch {profile}, blowup {}, {} quer{})\n   \
         prove {prove_secs:.1}s / verify {verify_secs:.2}s / proof {size} bytes / \
         {} published words / {} sub-proofs\n   \
         cells {main} main + {aux} aux ext; the projection for this run was \
         {:.1} GiB of peak RSS — compare against what the harness measured around \
         the process",
        inner.blowup_factor,
        inner.fri_number_of_queries,
        if inner.fri_number_of_queries == 1 {
            "y"
        } else {
            "ies"
        },
        proved.public_words.len(),
        proved.proof.proofs.len(),
        projected_peak_bytes(main, aux) / (1u64 << 30) as f64,
    );

    // ---- the published words are the ones the execution produced, so the
    // spine's differential still holds of the PROVED run and not only of an
    // execution. Checked by value against the epoch's own oracles.
    let pub_ext =
        |i: usize| super::word::word_as_ext(&proved.public_words[i].1).expect("an ext challenge");
    assert_eq!(pub_ext(0), e.z_alpha.0, "the proved run publishes z");
    assert_eq!(pub_ext(1), e.z_alpha.1, "the proved run publishes alpha");
    assert_eq!(
        super::word::word_as_ext(&proved.public_words[proved.public_words.len() - 1].1)
            .expect("the bus total is ext"),
        e.expected_bus_balance,
        "the proved run reaches production's own COMMIT-bus target"
    );

    // ---- FALSIFICATION 1: a tampered inner proof makes the wrap UNBUILDABLE.
    //
    // Not "unverifiable": every check is an assert inside a straight-line
    // program, so a false statement has no execution at all — there is no branch
    // to take and no error path to return, and `lfm_prove` fails in `execute`
    // before a trace exists. That is the designed behaviour of the machine, and
    // it is why the positive result above ("it proved") is the verification.
    let ix = super::epoch_verify_tests::arena_index(&e, 0);
    let mut tampered = arenas.clone();
    tampered[ix.openings][0][0] += FE::one();
    match lfm_prove(&program, &artifacts, &tampered, &opts) {
        Err(LfmProveError::Exec(err)) => {
            println!("   TAMPERED opened value 0 of table 0: the wrap is UNBUILDABLE ({err:?})")
        }
        Err(LfmProveError::Prover(err)) => {
            panic!("a tampered inner proof must fail in execution, not in the prover: {err:?}")
        }
        Ok(_) => panic!("a tampered opened value must not produce a wrap proof"),
    }

    // ---- FALSIFICATION 2: the honest proof against a MOVED claimed statement.
    //
    // The other half of the pair: the wrap proof is bound to the public words it
    // published (`absorb_lfm_statement`), so a verifier handed the real proof and
    // a different claim must reject. This is the path that rejects rather than
    // failing to build, and both must exist — a machine where only the first
    // existed would prove nothing about what the proof says.
    let mut moved = proved.public_words.clone();
    moved[0].1[0] += FE::one();
    assert!(
        !verify_against(
            &artifacts.roots,
            &artifacts.program_id,
            artifacts.keccak_rnd_chunks,
            &proved.proof,
            &moved,
            &opts,
            artifacts.hasher,
            artifacts.chip_set,
        ),
        "a moved claimed public word must make the wrap proof UNVERIFIABLE"
    );
    println!("   MOVED claimed public word 0: the wrap proof is UNVERIFIABLE");

    // ---- FALSIFICATION 3: the same proof against another program's identity.
    //
    // The registry premise. `verify_against` takes the roots and the digest, and
    // a proof of THIS program must not verify as a proof of a different one.
    let mut other = artifacts.program_id;
    other[0] ^= 1;
    assert!(
        !verify_against(
            &artifacts.roots,
            &other,
            artifacts.keccak_rnd_chunks,
            &proved.proof,
            &proved.public_words,
            &opts,
            artifacts.hasher,
            artifacts.chip_set,
        ),
        "a moved program digest must make the wrap proof UNVERIFIABLE"
    );
    println!("   MOVED program digest: the wrap proof is UNVERIFIABLE");
}

/// ★ THE RESIDENCY ORACLE — `ResidencyMode::RecomputeLde` produces the same wrap
/// commitments as `Retain`.
///
/// [`crate::tests::residency_mode_tests`] in the stark crate pins the mechanism
/// on a three-table toy, byte for byte. This pins it on the workload the mode
/// exists for: the wrap has preprocessed tables (whose main LDE carries the
/// precomputed columns the split trees were built from), `KECCAK_RND` chunks
/// (the family whose retention the mode drops), and the real transcript.
///
/// ## Why this compares commitments and not proof bytes
///
/// Proving is **not** reproducible run to run, and it is worth being exact
/// about why, because the fixture-blob note in
/// [`super::proof_fixture`] attributes it to sub-proofs committing to different
/// roots — which is not what happens here. Measured on this test: two runs
/// build byte-identical LFM traces, produce identical roots at every stage, and
/// still serialize to different proof bytes. The cause is the grinding search,
/// `grinding::generate_nonce`, which under the `parallel` feature is
/// `into_par_iter().find_any(..)` — *any* valid nonce, so which one comes back
/// depends on thread scheduling. The nonce is absorbed before the query indices
/// are sampled, so a different nonce opens different leaves. Both proofs are
/// valid; grinding is a proof of work and any witness satisfies it.
///
/// So everything the nonce cannot reach is compared, which is everything the
/// recomputed LDE feeds: the main, aux, precomputed, composition and FRI-layer
/// roots, the out-of-domain evaluations, the final polynomial, and the bus
/// public inputs. A recomputed LDE that disagreed with the tree Round 1
/// committed would move the composition root and the OOD evaluations — the two
/// values in that set derived from the LDE rather than from the trace.
///
/// The run is its own control: `Retain` is proved twice, and the first
/// comparison is `Retain` against `Retain`. If that one ever fails, the
/// nondeterminism has reached the commitments and this oracle — not the
/// residency mode — is what needs fixing.
///
/// `#[ignore]`d for the same reason as [`the_wrap_proves_and_verifies`], three
/// times over: it proves the wrap three times.
///
/// Run with:
/// `cargo test --release -p lambda-vm-prover --lib lfm::wrap_tests::the_wrap_commitments_match_across_residency_modes -- --ignored --nocapture`
#[test]
#[ignore]
fn the_wrap_commitments_match_across_residency_modes() {
    use stark::residency_mode::ResidencyMode;

    let e = super::epoch_tests::real_epoch_with(super::proof_fixture::fixture_options());
    let program = super::epoch_tests::epoch_program(&e, true);
    let arenas = super::epoch_tests::epoch_arena_words(&e, true);
    let opts = wrap_options();
    let artifacts = build_artifacts(&program, &opts);

    let prove_under = |residency: ResidencyMode| {
        let t = Instant::now();
        let proved = lfm_prove_with_residency(
            &program,
            &artifacts,
            &arenas,
            &opts,
            artifacts.hasher,
            residency,
        )
        .expect("the wrap must prove");
        let secs = t.elapsed().as_secs_f64();
        assert!(
            verify_against(
                &artifacts.roots,
                &artifacts.program_id,
                artifacts.keccak_rnd_chunks,
                &proved.proof,
                &proved.public_words,
                &opts,
                artifacts.hasher,
                artifacts.chip_set,
            ),
            "the wrap proof must verify under {residency:?}"
        );
        println!("   {residency:?}: proved in {secs:.1}s, verified");
        proved
    };

    // Everything the grinding nonce cannot reach.
    let commitments = |p: &super::proof::LfmProof| {
        p.proof
            .proofs
            .iter()
            .map(|q| {
                (
                    q.trace_length,
                    q.lde_trace_main_merkle_root,
                    q.lde_trace_aux_merkle_root,
                    q.lde_trace_precomputed_merkle_root,
                    q.composition_poly_root,
                    q.composition_poly_parts_ood_evaluation.clone(),
                    q.trace_ood_evaluations.row_major_data().to_vec(),
                    q.trace_ood_next_evaluations.row_major_data().to_vec(),
                    q.fri_layers_merkle_roots.clone(),
                    q.fri_final_poly_coeffs.clone(),
                    q.bus_public_inputs.as_ref().map(|b| b.table_contribution),
                )
            })
            .collect::<Vec<_>>()
    };

    let retained = prove_under(ResidencyMode::Retain);
    let control = prove_under(ResidencyMode::Retain);
    assert!(
        commitments(&retained) == commitments(&control),
        "CONTROL FAILED: two Retain runs disagree on their commitments, so this \
         oracle cannot say anything about the residency mode"
    );
    println!("   control: two Retain runs agree on every commitment");

    let recomputed = prove_under(ResidencyMode::RecomputeLde);
    assert!(
        commitments(&retained) == commitments(&recomputed),
        "the wrap commitments moved between residency modes"
    );
    assert_eq!(
        retained.public_words, recomputed.public_words,
        "the published words moved between residency modes"
    );

    // Not asserted — recorded. The nonces are expected to differ; printing them
    // keeps the reason this test compares commitments visible in its own output
    // rather than only in its doc comment.
    let nonces = |p: &super::proof::LfmProof| {
        p.proof
            .proofs
            .iter()
            .filter_map(|q| q.nonce)
            .collect::<Vec<_>>()
    };
    println!(
        "   grinding nonces equal across the two Retain runs: {} (expected false under `parallel`)",
        nonces(&retained) == nonces(&control)
    );
    println!("\n★ RESIDENCY ORACLE: the wrap commits identically under Retain and RecomputeLde");
}

/// ★ GATE B — a REAL Ethereum-block epoch, wrapped.
///
/// Everything else in this module wraps the 16-cycle fibonacci fixture, which
/// exercises every structure but at a size no production workload has. This
/// wraps one epoch of a real mainnet block at a SECURE inner preset
/// (blowup 4 / 110 queries, grinding as the preset sets it): one real block
/// epoch proof, compressed into one LFM proof.
///
/// The guest and the block input are multi-megabyte binaries that cannot be
/// checked in, so the test requires them by path and says so rather than
/// quietly proving the fixture and reporting it as a block:
///
/// ```text
/// LFM_CENSUS_ELF=/path/to/ethrex.elf \
/// LFM_CENSUS_INPUT=/path/to/ethrex_mainnet_25368371.bin \
/// LFM_CENSUS_EPOCH_LOG2=16 \
/// cargo test --release -p lambda-vm-prover --lib \
///   lfm::wrap_tests::the_real_block_epoch_wraps -- --ignored --nocapture
/// ```
///
/// ## The two knobs, and which one actually binds
///
/// `LFM_CENSUS_EPOCH_LOG2` sets the epoch size and `LFM_WRAP_QUERIES` overrides
/// the inner query count (default: the preset's 110, which is the secure one —
/// anything lower is NOT a security parameter set and every number taken under
/// it carries the count, exactly as
/// [`the_wrap_proves_at_blowup_8_geometry`] does).
///
/// Measured on a 60 GiB box: the epoch size is the *weak* knob and the query
/// count is the strong one. What decides whether a run fits is the number of
/// `KECCAK_RND` chunks the wrap's own trace needs, and that is
/// `(spine + per_query x queries) / 21,845` permutations. Per-query cost is
/// dominated by leaf absorption, which is set by table WIDTH and so barely
/// moves with epoch size — shrinking the epoch does not meaningfully shrink the
/// chunk count, and shrinking the query count does, linearly.
#[test]
#[ignore]
fn the_real_block_epoch_wraps() {
    for var in ["LFM_CENSUS_ELF", "LFM_CENSUS_INPUT"] {
        assert!(
            std::env::var(var).is_ok(),
            "{var} must name a file: this test wraps a REAL block epoch, and \
             without it the harness would build the fibonacci fixture and report \
             it under this test's name"
        );
    }
    let inputs = EpochInputs::from_env();
    let mut inner = crate::recursion::Preset::Blowup4.options();
    if let Ok(v) = std::env::var("LFM_WRAP_QUERIES") {
        inner.fri_number_of_queries = v.parse().expect("LFM_WRAP_QUERIES must be an integer");
    }
    println!(
        "★ REAL-BLOCK WRAP: guest {}, {} bytes of private input, 2^{} cycles/epoch, \
         inner blowup {} / {} queries{}",
        inputs.label,
        inputs.private_input.len(),
        inputs.epoch_log2,
        inner.blowup_factor,
        inner.fri_number_of_queries,
        if inner.fri_number_of_queries < 110 {
            "  (REDUCED — not a security parameter set)"
        } else {
            "  (the secure preset)"
        },
    );
    wrap_run_from(inner, inputs);
}

/// The census and shape of the assembled verifier WITHOUT proving it — the cheap
/// half of the wrap run, so the numbers exist even where the prove does not fit.
///
/// Also the spine/legs split, since the census is what says which chips the legs
/// actually cost: at the min preset the verifier is ~50/50 Fiat-Shamir and
/// verification by instruction count, and this is where that becomes a per-chip
/// statement.
#[test]
#[ignore]
fn the_wrap_census() {
    let e = super::epoch_tests::real_epoch();
    let profile = epoch_profile(&e);
    let program = super::epoch_tests::epoch_program(&e, true);
    let spine = super::epoch_tests::epoch_program(&e, false);

    report_program("ASSEMBLED (spine + legs)", &profile, &program);
    report_program("SPINE ALONE (no legs)", &profile, &spine);
    let (main, aux) = report_census(&format!("assembled, epoch {profile}"), &program);
    let (spine_main, spine_aux) = report_census(&format!("spine alone, epoch {profile}"), &spine);
    println!(
        "\n   legs' marginal cells: {} main + {} aux (assembled {} / {} against spine {} / {})",
        main - spine_main,
        aux - spine_aux,
        main,
        aux,
        spine_main,
        spine_aux
    );

    // The fixed-machine floor: what a program of NO instructions still pays for
    // the 14 chips. The number every cells-per-verify figure sits on top of.
    let empty = super::compiler::compile(super::builder::LfmBuilder::new().finish());
    let (floor_main, floor_aux) = lfm_cell_counts(&empty);
    println!(
        "   fixed-machine floor (an empty program): {floor_main} main + {floor_aux} aux — \
         {:.1}% of the assembled verifier's main cells",
        100.0 * floor_main as f64 / main as f64
    );
    assert!(
        main > floor_main,
        "the verifier must cost more than the floor"
    );
}

/// Peak prover memory the census implies, in bytes, from a MEASURED coefficient.
///
/// The measured point is slice 0: 481,327,124 base-field-equivalent cells peaked
/// at 16,228,499,456 bytes of RSS (15.1 GiB), i.e. 33.7 bytes per cell — a trace word, its
/// blowup-2 LDE, and the Merkle/quotient working set on top. Stated as a
/// coefficient rather than derived from first principles because the derivation
/// would be a guess about the prover's allocation pattern and this is an
/// observation of it. What it CANNOT see: whether the coefficient holds at ten
/// times the size (allocator behaviour, and the fact that a bigger program is
/// bigger in different chips), so it is a projection and is labelled as one
/// wherever it is printed.
const MEASURED_BYTES_PER_CELL: f64 = 16_228_499_456.0 / 481_327_124.0;

fn projected_peak_bytes(main: u64, aux: u64) -> f64 {
    (main + 3 * aux) as f64 * MEASURED_BYTES_PER_CELL
}

/// ★ SLICE 1 — the PRODUCTION-SHAPED census: the inner epoch at blowup 8 with its
/// real 73-query count, which is the standing decision for the inner proof.
///
/// This is the cells-per-verify number the hash matrix wants, and it is a
/// MEASUREMENT of the emitted program rather than a projection from a per-leg
/// cost: the same emitter, the same real epoch, the same 25 sub-proofs, with only
/// the inner proof's options moved. Whether the resulting program can be PROVED is
/// a separate question and the test answers it with the projection above rather
/// than by pretending to have run it.
#[test]
#[ignore]
fn the_wrap_census_at_blowup_8() {
    // This instrument compares against wave-6 PINNED keccak-era predictions
    // (openings 100,959; FRI 14,454/sub-proof) and decomposes with the
    // keccak-rate helpers. Under a BLAKE3 wrap those comparisons are not wrong
    // by a constant — they are about a different program. Re-derive the pins
    // under the production hash before running it there; until then, fail with
    // the cause named instead of an inscrutable count mismatch.
    assert_eq!(
        WrapHash::production(),
        WrapHash::Keccak,
        "the blowup-8 census's pinned predictions are keccak-era; re-derive them \
         under the production wrap hash before running this instrument"
    );
    let inner = crate::recursion::Preset::Blowup8.options();
    let t = Instant::now();
    let e = super::epoch_tests::real_epoch_with(inner.clone());
    let profile = epoch_profile(&e);
    println!(
        "inner epoch: {} sub-proofs, blowup {}, {} queries per table, grinding {}, \
         fri final poly log degree {} — proved and accepted in {:.1}s",
        e.legs.len(),
        inner.blowup_factor,
        inner.fri_number_of_queries,
        inner.grinding_factor,
        inner.fri_final_poly_log_degree,
        t.elapsed().as_secs_f64()
    );

    let t = Instant::now();
    let program = super::epoch_tests::epoch_program(&e, true);
    println!(
        "   emitted the assembled verifier in {:.1}s",
        t.elapsed().as_secs_f64()
    );
    report_program("ASSEMBLED VERIFIER @ inner blowup 8", &profile, &program);
    let (main, aux) = report_census(
        &format!("assembled, epoch {profile}, inner blowup 8"),
        &program,
    );
    report_ratio(&e, main, aux);

    // ---- MEASURED against the phase's pinned predictions, number by number.
    let openings: usize = e
        .legs
        .iter()
        .map(|l| {
            l.verify.num_queries
                * (super::epoch_verify::leaf_permutations(&l.verify.sub)
                    + l.verify.sub.groups().len() * l.verify.sub.merkle_depth)
        })
        .sum();
    let fri: usize = e
        .legs
        .iter()
        .map(|l| l.verify.num_queries * l.verify.fri.permutations_per_query())
        .sum();
    let spine = super::epoch_tests::epoch_program(&e, false);
    println!(
        "\n  MEASURED vs PREDICTED (epoch {profile}, inner blowup 8, {} queries):\n\
         \x20 openings   {openings:>9}   [predicted 100,959 — wave 6's projection of THIS epoch]\n\
         \x20 FRI        {fri:>9}   [pinned 14,454 per 2^20 sub-proof at blowup 8]\n\
         \x20 legs total {:>9}   = emitted assembled - spine\n\
         \x20 epoch bill {:>9}   [design target ~460,000 for a PRODUCTION-sized epoch]",
        e.legs[0].verify.num_queries,
        permutations(&program) - permutations(&spine),
        permutations(&program),
    );
    assert_eq!(
        permutations(&program) - permutations(&spine),
        openings + fri,
        "the emitted leg permutations must be the closed form over the shapes"
    );

    // ---- can it be proved? The projection, with its coefficient named.
    let bytes = projected_peak_bytes(main, aux);
    println!(
        "\n  PROVING THIS: {} main + {} aux ext = {} base-field-equivalent cells\n\
         \x20 projected peak RSS {:.1} GiB at the measured {:.1} bytes/cell \
         (slice 0's 15.1 GiB / 481.3M cells)\n\
         \x20 the measurement box has 124 GiB, so this is {:.1}x what fits",
        main,
        aux,
        main + 3 * aux,
        bytes / (1 << 30) as f64,
        MEASURED_BYTES_PER_CELL,
        bytes / (124.0 * (1u64 << 30) as f64),
    );
}

/// Falsification of the census instrument itself: it must agree with what the
/// PROVER actually builds and with what the VERIFIER's AIR set declares.
///
/// A census computed from the program alone would report the same numbers under a
/// broken trace builder, which is the "measures nothing" failure the method rules
/// name. Two independent oracles, both of which the census is not derived from:
///
/// - the real [`super::trace::LfmTraces`] — the tables `multi_prove` receives —
///   for the heights;
/// - the AIR set [`super::airs::LfmAirs`] builds, for the NAMES and the widths.
///   The names matter more than they look: the census maps `per_chip` array slots
///   onto `LFM_CHIP_NAMES` across the `KECCAK_RND` slot, and nothing about a
///   height or a width can see that mapping being off by one. `air_refs` is the
///   frozen order's own definition, so comparing against it is what catches it.
#[test]
fn the_census_agrees_with_the_traces_the_prover_builds() {
    // The two-permutation keccak chain: small, and it exercises every chip class
    // the census names except `LFM_PUBLIC`'s value columns.
    let program = super::programs::keccak_chain_program();
    let state: [u64; 25] =
        core::array::from_fn(|i| 0x9E37_79B9_7F4A_7C15u64.wrapping_mul(i as u64 + 1));
    let arenas = vec![super::keccak_adapter::state_to_words(&state).to_vec()];
    let exec = execute(&program, &arenas, &TestPermutation).expect("the chain program runs");
    let traces = super::trace::build_traces(&program, &exec.records);
    let census = lfm_chip_census(&program);

    // The frozen AIR order, as the census emits it and `air_trace_pairs` proves
    // it. Built from the trace set so a chip whose height the census got from the
    // wrong group shows up here.
    let dims = |t: &stark::trace::TraceTable<
        crate::tables::types::GoldilocksField,
        crate::tables::types::GoldilocksExtension,
    >| (t.num_rows(), t.num_main_columns);
    // Gated exactly as `air_trace_pairs` gates: this list is what the PROVER
    // proves, and the chain program hashes with keccak, so it carries the
    // keccak family and omits LFM_BLAKE3.
    let chip_set = super::airs::ChipSet::for_program(&program);
    assert_eq!(
        (chip_set.keccak, chip_set.blake3),
        (true, false),
        "the keccak chain must be a keccak-family program with no BLAKE3 work"
    );
    let mut built: Vec<(usize, usize)> = vec![
        dims(&traces.const_),
        dims(&traces.balu),
        dims(&traces.xalu),
        dims(&traces.select),
        dims(&traces.bitdec),
        dims(&traces.hash),
    ];
    if chip_set.keccak {
        built.push(dims(&traces.keccak));
    }
    built.push(dims(&traces.lanes));
    built.push(dims(&traces.hint));
    built.push(dims(&traces.public));
    built.push(dims(&traces.range));
    if chip_set.blake3 {
        built.extend(traces.blake3.iter().map(dims));
    }
    if chip_set.keccak {
        built.extend(traces.keccak_rnd.iter().map(dims));
        built.push(dims(&traces.keccak_rc));
    }
    built.push(dims(&traces.bitwise));

    assert_eq!(
        census.len(),
        built.len(),
        "the census must have one entry per trace the prover proves"
    );
    for (c, (rows, width)) in census.iter().zip(&built) {
        assert_eq!(
            c.rows, *rows as u64,
            "{}: the census height must be the trace's own",
            c.name
        );
        // The census counts VALUE columns, so the trace's full width less the
        // preprocessed prefix must be what it reports.
        assert!(
            c.main_cols <= *width,
            "{}: the census cannot count more value columns than the trace has",
            c.name
        );
    }

    // ---- the AIR set: the names and the widths, in the frozen order.
    let opts = wrap_options();
    let artifacts = build_artifacts(&program, &opts);
    let airs = super::airs::LfmAirs::new(
        &artifacts.roots,
        &opts,
        artifacts.keccak_rnd_chunks,
        artifacts.chip_set,
    );
    let refs = airs.air_refs();
    assert_eq!(
        census.len(),
        refs.len(),
        "the census must have one entry per AIR the verifier builds"
    );
    for (c, air) in census.iter().zip(&refs) {
        assert_eq!(
            c.name,
            air.name(),
            "the census and the AIR set disagree about the frozen chip order"
        );
        let (main_width, aux_width) = air.trace_layout();
        let prep = if air.is_preprocessed() {
            air.num_precomputed_columns()
        } else {
            0
        };
        assert_eq!(
            c.main_cols,
            main_width - prep,
            "{}: the census must count the AIR's value columns",
            c.name
        );
        assert_eq!(
            c.aux_cols, aux_width,
            "{}: the census must count the AIR's aux columns",
            c.name
        );
    }

    let cells_of = |c: &[LfmChipCells], name: &str| -> u64 {
        c.iter()
            .filter(|e| e.name == name)
            .map(|e| e.main_cells())
            .sum()
    };
    assert!(
        cells_of(&census, "KECCAK_RND") > 0,
        "the chain program hashes, so KECCAK_RND must carry rows"
    );
}

/// ★ R3 — the row-cliff instrument reports the two heights honestly.
///
/// The panel's whole value is that `real_rows` is the height the workload
/// actually occupies and `rows` is the one the prover pays for. If those two
/// were ever the same number the headroom column would read 0% everywhere and
/// silently stop warning, which is exactly the failure mode the panel exists to
/// end — so this pins the relationship rather than the values.
#[test]
fn the_census_reports_real_and_committed_heights_separately() {
    use super::airs::HeightRule;
    use super::layout::MIN_GROUP_ROWS;

    let program = super::programs::keccak_chain_program();
    let census = lfm_chip_census(&program);
    let chunk_perms = super::airs::keccak_rnd_chunk_permutations(&program);
    let mut chunk = chunk_perms.iter();

    for c in &census {
        assert!(
            c.real_rows <= c.rows,
            "{}: a chip cannot occupy more rows than it commits ({} > {})",
            c.name,
            c.real_rows,
            c.rows
        );
        // The headroom column is this subtraction and nothing else, so an entry
        // whose two heights are inconsistent with the padding rule would print a
        // number that means nothing.
        let expected = (c.rows - c.real_rows) as f64 / c.rows as f64;
        assert!(
            (c.headroom() - expected).abs() < 1e-12,
            "{}: headroom must be the padding fraction",
            c.name
        );

        match c.height_rule {
            HeightRule::Workload => {
                assert_eq!(
                    c.rows,
                    (c.real_rows as usize)
                        .next_power_of_two()
                        .max(MIN_GROUP_ROWS) as u64,
                    "{}: a workload-sized chip commits its real height padded to the \
                     next power of two — if this ever stops holding, the headroom is \
                     not a distance to a cliff",
                    c.name
                );
                assert!(
                    c.at_risk(),
                    "{}: workload-sized chips are watchable",
                    c.name
                );
            }
            HeightRule::Fixed => {
                assert_eq!(
                    c.real_rows, c.rows,
                    "{}: a lookup table is full, so its two heights coincide",
                    c.name
                );
                assert!(
                    !c.at_risk(),
                    "{}: a fixed table reads 0% headroom because it is full, not \
                     because it is about to double — the panel must not warn on it",
                    c.name
                );
            }
            HeightRule::Chunked => {
                let perms = chunk.next().expect("one census chunk per policy chunk");
                assert_eq!(
                    c.real_rows,
                    (perms * super::chunking::KECCAK_RND_ROWS_PER_PERMUTATION) as u64,
                    "a chunk occupies 24 rows per permutation it carries"
                );
                assert!(
                    !c.at_risk(),
                    "a full chunk sits permanently just under a power of two and can \
                     never cross it — the policy emits another chunk instead"
                );
            }
        }
    }
    assert!(
        chunk.next().is_none(),
        "every policy chunk must appear in the census"
    );
    assert!(
        census.iter().any(|c| c.height_rule == HeightRule::Workload),
        "the census must contain workload-sized chips for the panel to watch"
    );
}

/// ★ R3 — the headroom the panel would have printed for the artifact.
///
/// The bisect measured `LFM_LANES` at 4,141,992 of 4,194,304 rows on the real
/// block at 110 queries and called it 1.2% — the margin that let #903's ~21%
/// growth trip five chip doublings at once. That number is the instrument's
/// reason to exist, so it is pinned here against the recorded measurement
/// (`thoughts/shared/block-compression/WRAP-GROWTH-BISECT.md`, "Headroom at the
/// artifact"). A 110q census needs ~30 GiB to emit, so the arithmetic is pinned
/// on the recorded row counts rather than by re-running it.
#[test]
fn the_row_cliff_panel_reproduces_the_artifacts_measured_headroom() {
    use super::airs::HeightRule;

    // (chip, rows the mix occupied, rows committed, headroom the bisect reports)
    let artifact = [
        ("LFM_LANES", 4_141_992u64, 4_194_304u64, 1.2),
        ("LFM_SELECT", 463_650, 524_288, 11.6),
        ("LFM_BALU", 110_147_086, 134_217_728, 17.9),
        ("LFM_BITDEC", 1_717_958, 2_097_152, 18.1),
        ("LFM_CONST", 1_656, 2_048, 19.1),
        ("LFM_HINT", 1_523_011, 2_097_152, 27.4),
        ("LFM_XALU", 1_485_219, 2_097_152, 29.2),
    ];

    for (name, real_rows, rows, expected_pct) in artifact {
        let c = LfmChipCells {
            name,
            rows,
            real_rows,
            height_rule: HeightRule::Workload,
            main_cols: 1,
            aux_cols: 0,
        };
        assert_eq!(
            rows,
            (real_rows as usize).next_power_of_two() as u64,
            "{name}: the recorded committed height must be the recorded real height padded",
        );
        let got = 100.0 * c.headroom();
        assert!(
            (got - expected_pct).abs() < 0.05,
            "{name}: panel would print {got:.1}%, the bisect measured {expected_pct}%",
        );
        // Crossing doubles the height, so the chip adds exactly what it already
        // contributes — the quantity the panel prices.
        assert_eq!(c.cliff_cost(), c.main_cells() + 3 * c.aux_cells());
    }
}

// ===================== the BATCHED wrap (M-8 / T3) =====================

/// The batched sibling of [`wrap_run_from`]: the same census-env epoch proved
/// through `multi_prove_batched`, its ASSEMBLED BATCHED verifier emitted, and
/// that program PROVED on the per-table LFM prover — batching the wrap itself
/// is out of scope; the wrap-side economy under measurement is the verifier
/// program's, not the wrap prover's.
fn batched_wrap_run_from(inner: ProofOptions, inputs: EpochInputs) {
    let t_epoch = Instant::now();
    let e = super::epoch_tests::real_batched_epoch_from(inner.clone(), inputs);
    let n = e.proof.tables.len();
    let h_min = e.shape.heights.iter().copied().min().expect("tables");
    let h_max = e.shape.heights.iter().copied().max().expect("tables");
    let profile = format!(
        "{n} tables, LDE 2^{h_min}..2^{h_max}, batched {}/standalone {}",
        e.challenges.fri.plan.batched.len(),
        e.challenges.fri.plan.standalone.len(),
    );
    println!(
        "batched inner epoch: {profile}, blowup {}, {} queries, grinding {} — built and \
         HOST-VERIFIED in {:.1}s",
        inner.blowup_factor,
        e.fri_params.num_queries,
        e.fri_params.grinding_factor,
        t_epoch.elapsed().as_secs_f64()
    );

    let t = Instant::now();
    let program = super::epoch_tests::batched_epoch_program_with(&e, true, false);
    let mut arenas = super::epoch_tests::batched_epoch_arenas(&e);
    arenas.push(super::epoch_verify_tests::batched_opening_arena(&e));
    arenas.push(super::epoch_verify_tests::batched_fri_arena(&e));
    println!(
        "   emitted the assembled BATCHED verifier in {:.1}s",
        t.elapsed().as_secs_f64()
    );
    report_program("THE BATCHED WRAPPED PROGRAM", &profile, &program);
    let (main, aux) = report_census(&format!("assembled batched verifier, {profile}"), &program);

    // ---- the spine/legs split, against the batched CLOSED FORM — the number
    // the campaign predicts: leg hashing collapses to ~one mixed path per
    // round per query plus the small prep trees.
    let spine = super::epoch_tests::batched_epoch_program(&e);
    let wrap_hash = WrapHash::production();
    let leg_hash_ops = hash_ops(&program, wrap_hash) - hash_ops(&spine, wrap_hash);
    let per_query = super::batched_epoch_verify::batched_query_permutations_for(
        &e.shape,
        &e.fri_params,
        wrap_hash,
    );
    assert_eq!(
        leg_hash_ops,
        e.proof.queries.len() * per_query,
        "the emitted leg {wrap_hash:?} operations must equal the batched closed form"
    );
    println!(
        "   spine {} instr / {} {:?} ops / {} words   legs {} / {} / {}   \
         per query: {per_query} ops ({} queries, closed form checked)",
        spine.instrs.len(),
        hash_ops(&spine, wrap_hash),
        wrap_hash,
        arena_words(&spine),
        program.instrs.len() - spine.instrs.len(),
        leg_hash_ops,
        arena_words(&program) - arena_words(&spine),
        e.proof.queries.len(),
    );
    println!(
        "   projected peak RSS for this run: {:.1} GiB",
        projected_peak_bytes(main, aux) / (1u64 << 30) as f64
    );

    let opts = wrap_options();
    let artifacts = build_artifacts(&program, &opts);
    println!(
        "   wrap options: blowup {}, {} queries, grinding {}\n   chip log-heights: {:?}",
        opts.blowup_factor, opts.fri_number_of_queries, opts.grinding_factor, artifacts.log_heights
    );

    // ---- PROVE (the per-table LFM prover, deliberately).
    let t = Instant::now();
    let proved =
        lfm_prove(&program, &artifacts, &arenas, &opts).expect("the batched wrap must prove");
    let prove_secs = t.elapsed().as_secs_f64();
    let size = rkyv::to_bytes::<rkyv::rancor::Error>(&proved.proof)
        .expect("the wrap proof must serialize")
        .len();

    // ---- VERIFY.
    let t = Instant::now();
    assert!(
        verify_against(
            &artifacts.roots,
            &artifacts.program_id,
            artifacts.keccak_rnd_chunks,
            &proved.proof,
            &proved.public_words,
            &opts,
            artifacts.hasher,
            artifacts.chip_set,
        ),
        "the batched wrap proof must verify"
    );
    let verify_secs = t.elapsed().as_secs_f64();
    println!(
        "\n★ BATCHED WRAP PROVED AND VERIFIED ({profile}, inner blowup {}, {} queries)\n   \
         prove {prove_secs:.1}s / verify {verify_secs:.2}s / proof {size} bytes / \
         {} published words / {} sub-proofs\n   cells {main} main + {aux} aux ext",
        inner.blowup_factor,
        e.fri_params.num_queries,
        proved.public_words.len(),
        proved.proof.proofs.len(),
    );

    // ---- the published words are the execution's own, so the spine's
    // differential holds of the PROVED run: the shared pair, the attestation,
    // and the closure, by value against the harness's oracles.
    let pub_ext =
        |i: usize| super::word::word_as_ext(&proved.public_words[i].1).expect("an ext challenge");
    let [z, alpha] = e.challenges.lookup.as_slice() else {
        panic!("the shared pair is (z, alpha)");
    };
    assert_eq!(pub_ext(0), *z, "the proved run publishes z");
    assert_eq!(pub_ext(1), *alpha, "the proved run publishes alpha");
    assert_eq!(
        super::word::word_as_ext(&proved.public_words[proved.public_words.len() - 1].1)
            .expect("the bus total is ext"),
        e.expected_bus_balance,
        "the proved run reaches production's own COMMIT-bus target"
    );

    // ---- FALSIFICATION 1: a tampered inner opening makes the wrap
    // UNBUILDABLE (the checks are asserts in a straight-line program; a false
    // statement has no execution at all).
    let open_idx = arenas.len() - 2;
    let mut tampered = arenas.clone();
    tampered[open_idx][0][0] += FE::one();
    match lfm_prove(&program, &artifacts, &tampered, &opts) {
        Err(LfmProveError::Exec(err)) => {
            println!("   TAMPERED opening word 0: the batched wrap is UNBUILDABLE ({err:?})")
        }
        Err(LfmProveError::Prover(err)) => {
            panic!("a tampered inner proof must fail in execution, not in the prover: {err:?}")
        }
        Ok(_) => panic!("a tampered opened value must not produce a wrap proof"),
    }

    // ---- FALSIFICATION 2: the honest proof against a MOVED claimed statement
    // must reject at verification.
    let mut moved = proved.public_words.clone();
    moved[0].1[0] += FE::one();
    assert!(
        !verify_against(
            &artifacts.roots,
            &artifacts.program_id,
            artifacts.keccak_rnd_chunks,
            &proved.proof,
            &moved,
            &opts,
            artifacts.hasher,
            artifacts.chip_set,
        ),
        "a moved claimed word must be rejected"
    );
    println!("   MOVED claimed word 0: rejected");
}

/// ★ GATE B's batched sibling — a REAL Ethereum-block epoch, proved through
/// the BATCHED base layer and wrapped. Same env contract as
/// [`the_real_block_epoch_wraps`]; run both on the same box for the T3
/// comparison the campaign exists to make — memory first, at 2^16 and at the
/// 2^24 posture.
#[test]
#[ignore]
fn the_real_block_epoch_wraps_batched() {
    for var in ["LFM_CENSUS_ELF", "LFM_CENSUS_INPUT"] {
        assert!(
            std::env::var(var).is_ok(),
            "{var} must name a file: this test wraps a REAL block epoch"
        );
    }
    let inputs = EpochInputs::from_env();
    let mut inner = crate::recursion::Preset::Blowup4.options();
    if let Ok(v) = std::env::var("LFM_WRAP_QUERIES") {
        inner.fri_number_of_queries = v.parse().expect("LFM_WRAP_QUERIES must be an integer");
    }
    println!(
        "★ REAL-BLOCK BATCHED WRAP: guest {}, {} bytes of private input, 2^{} cycles/epoch, \
         inner blowup {} / {} queries{}",
        inputs.label,
        inputs.private_input.len(),
        inputs.epoch_log2,
        inner.blowup_factor,
        inner.fri_number_of_queries,
        if inner.fri_number_of_queries < 110 {
            "  (REDUCED — not a security parameter set)"
        } else {
            "  (the secure preset)"
        },
    );
    batched_wrap_run_from(inner, inputs);
}

/// ★ The P2 DRIVER'S FLOW at the fixture, not ignored: a batched-carved
/// continuation bundle's FINAL epoch reconstructs from proofs alone, its
/// CARVED program wraps end to end, and the wrap PUBLISHES the carved L2G
/// root — byte-compared against the bundle's claimed root, exactly the check
/// P3's aggregator makes. Gated on every suite run, so the block driver's box
/// run cannot be the first execution of any of it.
#[test]
fn the_fixture_continuation_epoch_wraps_batched_from_proofs() {
    let elf_bytes = super::proof_fixture::read_inner_elf();
    let inner = super::proof_fixture::fixture_options();
    let bundle = crate::continuation::prove_continuation_batched(
        &elf_bytes,
        &[],
        super::proof_fixture::FIXTURE_EPOCH_LOG2,
        &inner,
    )
    .expect("the fixture continuation must prove batched");
    let n = bundle.num_epochs();
    assert!(n >= 2, "the fixture continuation must have a final epoch");

    let e = super::epoch_tests::real_batched_epoch_from_continuation(
        &inner,
        &elf_bytes,
        &bundle,
        n - 1,
        None,
    )
    .expect("the final epoch must reconstruct from proofs alone");
    let program = super::epoch_tests::batched_epoch_program_with(&e, true, false);
    let mut arenas = super::epoch_tests::batched_epoch_arenas(&e);
    arenas.push(super::epoch_verify_tests::batched_opening_arena(&e));
    arenas.push(super::epoch_verify_tests::batched_fri_arena(&e));
    let opts = wrap_options();
    let artifacts = build_artifacts(&program, &opts);

    let proved =
        lfm_prove(&program, &artifacts, &arenas, &opts).expect("the carved wrap must prove");
    assert!(
        verify_against(
            &artifacts.roots,
            &artifacts.program_id,
            artifacts.keccak_rnd_chunks,
            &proved.proof,
            &proved.public_words,
            &opts,
            artifacts.hasher,
            artifacts.chip_set,
        ),
        "the carved wrap of the final epoch must verify"
    );

    // The published-word schema's aggregator-facing check: the last 8 words
    // are the carved L2G root, byte-equal to the bundle's claimed root.
    let root = bundle.epoch_view(n - 1).l2g_root();
    let published_root: Vec<FE> = proved.public_words[proved.public_words.len() - 8..]
        .iter()
        .map(|w| super::word::word_as_base(&w.1).expect("a root half is a base word"))
        .collect();
    let expected_root: Vec<FE> = root
        .chunks(4)
        .map(|c: &[u8]| {
            FE::from(u32::from_le_bytes(c.try_into().expect("a root is 32 bytes")) as u64)
        })
        .collect();
    assert_eq!(
        published_root, expected_root,
        "the wrap must publish the carved L2G root it verified under"
    );
    println!(
        "★ P2 fixture driver flow: FINAL carved epoch wrapped, verified, and its          published L2G root matches the bundle's claim ({} published words)",
        proved.public_words.len()
    );
}

/// The batched wrap at the FIXTURE, not ignored — the whole T3 instrument's
/// flow (batched inner, emitted verifier, per-table LFM prove, verify, both
/// falsification arms) gated on every suite run, so the box run cannot be the
/// first execution of any of it.
#[test]
fn the_fixture_epoch_wraps_batched() {
    batched_wrap_run_from(
        super::proof_fixture::fixture_options(),
        EpochInputs::fixture(),
    );
}

/// ★ GATE B (P1) — a from-proof epoch wraps end to end, and it is the FINAL
/// epoch of its continuation (HALT on board): the shape the real block's last
/// epoch has, which the session harness cannot build. The epoch reaches the
/// wrap through [`super::epoch_tests::real_epoch_from_continuation`] — proofs
/// alone, no live proving session — and the proved run must publish the
/// epoch's own oracles.
///
/// Run with:
/// `cargo test --release -p lambda-vm-prover --lib lfm::wrap_tests::the_from_proof_final_epoch_wraps -- --ignored --exact --nocapture`
#[test]
#[ignore]
fn the_from_proof_final_epoch_wraps() {
    let elf_bytes = super::proof_fixture::read_inner_elf();
    let inner = super::epoch_tests::from_proof_gate_options();
    let bundle = crate::continuation::prove_continuation(
        &elf_bytes,
        &[],
        super::proof_fixture::FIXTURE_EPOCH_LOG2,
        &inner,
    )
    .expect("the fixture continuation must prove");
    let n = bundle.num_epochs();
    assert!(n >= 2, "the fixture continuation must have a final epoch");

    let e =
        super::epoch_tests::real_epoch_from_continuation(&inner, &elf_bytes, &bundle, n - 1, None)
            .expect("the final epoch must reconstruct from proofs alone");
    let program = super::epoch_tests::epoch_program(&e, true);
    let arenas = super::epoch_tests::epoch_arena_words(&e, true);
    let opts = wrap_options();
    let artifacts = build_artifacts(&program, &opts);

    let t = Instant::now();
    let proved = lfm_prove(&program, &artifacts, &arenas, &opts).expect("the wrap must prove");
    let prove_secs = t.elapsed().as_secs_f64();
    assert!(
        verify_against(
            &artifacts.roots,
            &artifacts.program_id,
            artifacts.keccak_rnd_chunks,
            &proved.proof,
            &proved.public_words,
            &opts,
            artifacts.hasher,
            artifacts.chip_set,
        ),
        "the wrap of the final epoch must verify"
    );

    // The proved run publishes the epoch's own oracles.
    let pub_ext =
        |i: usize| super::word::word_as_ext(&proved.public_words[i].1).expect("an ext challenge");
    assert_eq!(pub_ext(0), e.z_alpha.0, "the proved run publishes z");
    assert_eq!(pub_ext(1), e.z_alpha.1, "the proved run publishes alpha");
    assert_eq!(
        super::word::word_as_ext(&proved.public_words[proved.public_words.len() - 1].1)
            .expect("the bus total is ext"),
        e.expected_bus_balance,
        "the proved run reaches production's own COMMIT-bus target"
    );
    println!(
        "\n★ P1 GATE B: the FINAL epoch (epoch {} of {n}), wrapped from proofs alone — \
         proved in {prove_secs:.1}s, verified, {} sub-proofs, {} published words",
        n - 1,
        proved.proof.proofs.len(),
        proved.public_words.len(),
    );
}

/// Peak RSS high-water mark of this process, GiB — `VmHWM` on Linux (the box);
/// `None` elsewhere.
pub(super) fn peak_rss_gib() -> Option<f64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("VmHWM:"))?;
    let kb: f64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb / (1024.0 * 1024.0))
}

/// ★ P1 — THE BLOCK, end to end, in one process: the production continuation
/// prove (every epoch + the global proof), full bundle host verification
/// (every epoch, the global proof, and the L2G root-equality binding), then
/// EVERY epoch wrapped from the proofs alone. The numbers this prints are
/// BLOCK numbers; per-epoch lines are supporting detail.
///
/// Residency: the epoch proves are `Retain` (hardcoded in
/// `prove_continuation`); the wrap proves are `Retain` (the `lfm_prove`
/// default). Proof BYTES are not run-reproducible (grinding nonces); roots
/// are.
///
/// env (required): `LFM_CENSUS_ELF`, `LFM_CENSUS_INPUT`. Optional:
/// `LFM_CENSUS_EPOCH_LOG2` (epoch size), `LFM_WRAP_QUERIES` (inner query
/// count override; the default is the secure preset's).
///
/// Run at the 2^24 posture:
/// ```text
/// LFM_CENSUS_ELF=/path/to/ethrex.elf \
/// LFM_CENSUS_INPUT=/path/to/ethrex_mainnet_25368371.bin \
/// LFM_CENSUS_EPOCH_LOG2=24 LAMBDA_VM_MAX_ROWS_LOG2=24 \
/// cargo test --release -p lambda-vm-prover --lib \
///   lfm::wrap_tests::the_real_block_proves_and_wraps_end_to_end -- --ignored --exact --nocapture
/// ```
#[test]
#[ignore]
fn the_real_block_proves_and_wraps_end_to_end() {
    for var in ["LFM_CENSUS_ELF", "LFM_CENSUS_INPUT"] {
        assert!(
            std::env::var(var).is_ok(),
            "{var} must name a file: this test proves a REAL block, and without \
             it the harness would build the fibonacci fixture and report it \
             under this test's name"
        );
    }
    let inputs = EpochInputs::from_env();
    let mut inner = crate::recursion::Preset::Blowup4.options();
    if let Ok(v) = std::env::var("LFM_WRAP_QUERIES") {
        inner.fri_number_of_queries = v.parse().expect("LFM_WRAP_QUERIES must be an integer");
    }
    println!(
        "★ P1 BLOCK RUN: guest {}, {} bytes of private input, 2^{} cycles/epoch, \
         inner blowup {} / {} queries{}  — epoch residency Retain, wrap residency Retain",
        inputs.label,
        inputs.private_input.len(),
        inputs.epoch_log2,
        inner.blowup_factor,
        inner.fri_number_of_queries,
        if inner.fri_number_of_queries < 110 {
            "  (REDUCED — not a security parameter set)"
        } else {
            "  (the secure preset)"
        },
    );

    let t_total = Instant::now();

    // ---- the base layer: every epoch + the global proof, production's path.
    let t = Instant::now();
    let bundle = crate::continuation::prove_continuation(
        &inputs.elf_bytes,
        &inputs.private_input,
        inputs.epoch_log2,
        &inner,
    )
    .expect("the block must prove");
    let base_secs = t.elapsed().as_secs_f64();
    let n = bundle.num_epochs();
    let bundle_bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&bundle)
        .expect("the bundle must serialize")
        .len();
    println!(
        "   base: {n} epochs + global proof in {base_secs:.1}s ({bundle_bytes} bundle bytes), \
         peak RSS so far {:?} GiB",
        peak_rss_gib(),
    );

    // ---- full host verification: every epoch, the global proof, the binding.
    let t = Instant::now();
    let out = crate::continuation::verify_continuation(&inputs.elf_bytes, &bundle, &inner)
        .expect("the bundle must be well-formed");
    assert!(
        out.is_some(),
        "the block bundle must host-verify (epochs + global + L2G root binding)"
    );
    let host_verify_secs = t.elapsed().as_secs_f64();
    println!("   host verify (epochs + global + binding): {host_verify_secs:.1}s");

    // ---- every epoch, wrapped from the proofs alone.
    let elf = executor::elf::Elf::load(&inputs.elf_bytes).expect("the inner ELF must load");
    let decode = crate::tables::decode::commitment_from_elf(&elf, &inner)
        .expect("the DECODE commitment must compute");
    let wrap_opts = wrap_options();
    let (mut construct_secs, mut wrap_prove_secs, mut wrap_verify_secs) = (0f64, 0f64, 0f64);
    let mut wrap_sizes = Vec::new();
    for i in 0..n {
        let t = Instant::now();
        let e = super::epoch_tests::real_epoch_from_continuation(
            &inner,
            &inputs.elf_bytes,
            &bundle,
            i,
            Some(decode),
        )
        .unwrap_or_else(|err| panic!("epoch {i} must reconstruct from the bundle: {err}"));
        let program = super::epoch_tests::epoch_program(&e, true);
        let arenas = super::epoch_tests::epoch_arena_words(&e, true);
        let artifacts = build_artifacts(&program, &wrap_opts);
        let c = t.elapsed().as_secs_f64();
        construct_secs += c;

        let t = Instant::now();
        let proved = lfm_prove(&program, &artifacts, &arenas, &wrap_opts)
            .unwrap_or_else(|err| panic!("epoch {i}'s wrap must prove: {err:?}"));
        let p = t.elapsed().as_secs_f64();
        wrap_prove_secs += p;

        let t = Instant::now();
        assert!(
            verify_against(
                &artifacts.roots,
                &artifacts.program_id,
                artifacts.keccak_rnd_chunks,
                &proved.proof,
                &proved.public_words,
                &wrap_opts,
                artifacts.hasher,
                artifacts.chip_set,
            ),
            "epoch {i}'s wrap must verify"
        );
        let v = t.elapsed().as_secs_f64();
        wrap_verify_secs += v;

        let size = rkyv::to_bytes::<rkyv::rancor::Error>(&proved.proof)
            .expect("the wrap proof must serialize")
            .len();
        wrap_sizes.push(size);
        println!(
            "   epoch {i}: reconstruct+emit {c:.1}s, wrap prove {p:.1}s, verify {v:.2}s, \
             {size} bytes, {} sub-proofs",
            proved.proof.proofs.len(),
        );
    }

    let total = t_total.elapsed().as_secs_f64();
    println!(
        "\n★★★ P1 BLOCK RECORD (per-table): {n} epochs @2^{} cycles, inner blowup {} / {}q, \
         wrap blowup {} / {}q, residency Retain both layers\n    \
         base prove {base_secs:.1}s + host verify {host_verify_secs:.1}s + wrap constructs \
         {construct_secs:.1}s + wrap proves {wrap_prove_secs:.1}s + wrap verifies \
         {wrap_verify_secs:.1}s\n    TOTAL WALL {total:.1}s ({:.1} min)\n    \
         proofs: bundle {bundle_bytes} B, wraps {wrap_sizes:?} B\n    \
         peak RSS (VmHWM): {:?} GiB",
        inputs.epoch_log2,
        inner.blowup_factor,
        inner.fri_number_of_queries,
        wrap_opts.blowup_factor,
        wrap_opts.fri_number_of_queries,
        total / 60.0,
        peak_rss_gib(),
    );
}

/// ★★★ THE P2 BLOCK DRIVER — [`the_real_block_proves_and_wraps_end_to_end`]
/// on the BATCHED format: every epoch proven as one mixed-MMCS proof with the
/// L2G main matrix carved standalone (`prove_continuation_batched`), the
/// bundle completely host-verified (epochs, global proof, the root-equality
/// binding reading the carved roots), then every epoch wrapped from the
/// proofs alone through the batched from-proof constructor and the CARVED
/// emitted verifier. One process; the epoch proves are `Retain`; the wrap
/// proves are `Retain`. Same env contract as the per-table driver; run both
/// on the same box for the P2 comparison the campaign exists to make.
///
/// The real block's cross-epoch PAGE CENSUS — execution and collection only,
/// nothing proven. Prints the numbers the aggregator's closed-form census
/// consumes: the global memory proof carries one GLOBAL_MEMORY table per
/// touched page, so the aggregation program's global-verify legs scale with
/// exactly what this prints. Runs locally in minutes (same env contract as
/// the block drivers).
#[test]
#[ignore]
fn the_real_blocks_page_census() {
    for var in ["LFM_CENSUS_ELF", "LFM_CENSUS_INPUT"] {
        assert!(
            std::env::var(var).is_ok(),
            "{var} must name a file: this census is only meaningful on a real block"
        );
    }
    let inputs = EpochInputs::from_env();
    let census = crate::continuation::block_page_census(
        &inputs.elf_bytes,
        &inputs.private_input,
        inputs.epoch_log2,
    )
    .expect("the census run must execute");
    let mut hist: std::collections::BTreeMap<u32, usize> = std::collections::BTreeMap::new();
    for &cells in &census.page_cells {
        *hist
            .entry(cells.next_power_of_two().max(4).trailing_zeros())
            .or_default() += 1;
    }
    println!(
        "★ PAGE CENSUS {}: {} epochs, {} touched pages ({} private-input), \
         crossing cells per epoch {:?}",
        inputs.label,
        census.num_epochs,
        census.touched_page_bases.len(),
        census.num_private_input_pages,
        census.l2g_cells,
    );
    println!("   page-table height histogram (log2 padded rows -> pages): {hist:?}");
}

/// Run at the 2^24 posture:
/// ```text
/// LFM_CENSUS_ELF=/path/to/ethrex.elf \
/// LFM_CENSUS_INPUT=/path/to/ethrex_mainnet_25368371.bin \
/// LFM_CENSUS_EPOCH_LOG2=24 LAMBDA_VM_MAX_ROWS_LOG2=24 \
/// cargo test --release -p lambda-vm-prover --lib \
///   lfm::wrap_tests::the_real_block_proves_and_wraps_end_to_end_batched -- --ignored --exact --nocapture
/// ```
#[test]
#[ignore]
fn the_real_block_proves_and_wraps_end_to_end_batched() {
    for var in ["LFM_CENSUS_ELF", "LFM_CENSUS_INPUT"] {
        assert!(
            std::env::var(var).is_ok(),
            "{var} must name a file: this test proves a REAL block, and without \
             it the harness would build the fibonacci fixture and report it \
             under this test's name"
        );
    }
    let inputs = EpochInputs::from_env();
    let mut inner = crate::recursion::Preset::Blowup4.options();
    if let Ok(v) = std::env::var("LFM_WRAP_QUERIES") {
        inner.fri_number_of_queries = v.parse().expect("LFM_WRAP_QUERIES must be an integer");
    }
    println!(
        "★ P2 BLOCK RUN (batched): guest {}, {} bytes of private input, 2^{} cycles/epoch, \
         inner blowup {} / {} queries{}  — epoch residency Retain, wrap residency Retain",
        inputs.label,
        inputs.private_input.len(),
        inputs.epoch_log2,
        inner.blowup_factor,
        inner.fri_number_of_queries,
        if inner.fri_number_of_queries < 110 {
            "  (REDUCED — not a security parameter set)"
        } else {
            "  (the secure preset)"
        },
    );

    let t_total = Instant::now();

    // ---- the base layer: every epoch BATCHED-CARVED + the (per-table)
    // global proof, production's path.
    let t = Instant::now();
    let bundle = crate::continuation::prove_continuation_batched(
        &inputs.elf_bytes,
        &inputs.private_input,
        inputs.epoch_log2,
        &inner,
    )
    .expect("the block must prove batched");
    let base_secs = t.elapsed().as_secs_f64();
    let n = bundle.num_epochs();
    let bundle_bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&bundle)
        .expect("the bundle must serialize")
        .len();
    println!(
        "   base (batched): {n} epochs + global proof in {base_secs:.1}s \
         ({bundle_bytes} bundle bytes), peak RSS so far {:?} GiB",
        peak_rss_gib(),
    );

    // ---- full host verification: every epoch's carved batched verify, the
    // global proof, and the binding view reading the carved roots.
    let t = Instant::now();
    let out = crate::continuation::verify_continuation(&inputs.elf_bytes, &bundle, &inner)
        .expect("the bundle must be well-formed");
    assert!(
        out.is_some(),
        "the batched block bundle must host-verify (epochs + global + L2G root binding)"
    );
    let host_verify_secs = t.elapsed().as_secs_f64();
    println!("   host verify (epochs + global + binding): {host_verify_secs:.1}s");

    // ---- every epoch, wrapped from the proofs alone: the CARVED program.
    let elf = executor::elf::Elf::load(&inputs.elf_bytes).expect("the inner ELF must load");
    let decode = crate::tables::decode::commitment_from_elf(&elf, &inner)
        .expect("the DECODE commitment must compute");
    let wrap_opts = wrap_options();
    let (mut construct_secs, mut wrap_prove_secs, mut wrap_verify_secs) = (0f64, 0f64, 0f64);
    let mut wrap_sizes = Vec::new();
    for i in 0..n {
        let t = Instant::now();
        let e = super::epoch_tests::real_batched_epoch_from_continuation(
            &inner,
            &inputs.elf_bytes,
            &bundle,
            i,
            Some(decode),
        )
        .unwrap_or_else(|err| panic!("epoch {i} must reconstruct from the bundle: {err}"));
        let program = super::epoch_tests::batched_epoch_program_with(&e, true, false);
        let mut arenas = super::epoch_tests::batched_epoch_arenas(&e);
        arenas.push(super::epoch_verify_tests::batched_opening_arena(&e));
        arenas.push(super::epoch_verify_tests::batched_fri_arena(&e));
        let artifacts = build_artifacts(&program, &wrap_opts);
        let c = t.elapsed().as_secs_f64();
        construct_secs += c;

        let t = Instant::now();
        let proved = lfm_prove(&program, &artifacts, &arenas, &wrap_opts)
            .unwrap_or_else(|err| panic!("epoch {i}'s wrap must prove: {err:?}"));
        let p = t.elapsed().as_secs_f64();
        wrap_prove_secs += p;

        let t = Instant::now();
        assert!(
            verify_against(
                &artifacts.roots,
                &artifacts.program_id,
                artifacts.keccak_rnd_chunks,
                &proved.proof,
                &proved.public_words,
                &wrap_opts,
                artifacts.hasher,
                artifacts.chip_set,
            ),
            "epoch {i}'s wrap must verify"
        );
        let v = t.elapsed().as_secs_f64();
        wrap_verify_secs += v;

        // The published-word schema: the wrap's last 8 published words are
        // the carved L2G root's halves — byte-compare them against the
        // bundle's claimed root, exactly the check P3's aggregator makes.
        let root = bundle.epoch_view(i).l2g_root();
        let published_root: Vec<FE> = proved.public_words[proved.public_words.len() - 8..]
            .iter()
            .map(|w| super::word::word_as_base(&w.1).expect("a root half is a base word"))
            .collect();
        let expected_root: Vec<FE> = root
            .chunks(4)
            .map(|c: &[u8]| {
                FE::from(u32::from_le_bytes(c.try_into().expect("a root is 32 bytes")) as u64)
            })
            .collect();
        assert_eq!(
            published_root, expected_root,
            "epoch {i}: the wrap must publish its carved L2G root"
        );

        let size = rkyv::to_bytes::<rkyv::rancor::Error>(&proved.proof)
            .expect("the wrap proof must serialize")
            .len();
        wrap_sizes.push(size);
        println!(
            "   epoch {i}: reconstruct+emit {c:.1}s, wrap prove {p:.1}s, verify {v:.2}s, \
             {size} bytes, {} sub-proofs, L2G root published",
            proved.proof.proofs.len(),
        );
    }

    let total = t_total.elapsed().as_secs_f64();
    println!(
        "\n★★★ P2 BLOCK RECORD (batched): {n} epochs @2^{} cycles, inner blowup {} / {}q, \
         wrap blowup {} / {}q, residency Retain both layers\n    \
         base prove {base_secs:.1}s + host verify {host_verify_secs:.1}s + wrap constructs \
         {construct_secs:.1}s + wrap proves {wrap_prove_secs:.1}s + wrap verifies \
         {wrap_verify_secs:.1}s\n    TOTAL WALL {total:.1}s ({:.1} min)\n    \
         proofs: bundle {bundle_bytes} B, wraps {wrap_sizes:?} B\n    \
         peak RSS (VmHWM): {:?} GiB",
        inputs.epoch_log2,
        inner.blowup_factor,
        inner.fri_number_of_queries,
        wrap_opts.blowup_factor,
        wrap_opts.fri_number_of_queries,
        total / 60.0,
        peak_rss_gib(),
    );
}
