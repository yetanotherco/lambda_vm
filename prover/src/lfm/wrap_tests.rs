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

use super::airs::{LfmChipCells, lfm_cell_counts, lfm_chip_census};
use super::compiler::LfmProgram;
use super::executor::execute;
use super::hash::TestPermutation;
use super::instr::Instr;
use super::proof::{LfmProveError, lfm_prove, verify_against};
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

/// Arena words a program declares.
pub(super) fn arena_words(program: &LfmProgram) -> usize {
    program.arena_schema.lens.iter().map(|l| *l as usize).sum()
}

/// ★ The registry-entry shape record: what the machine proves for one program.
///
/// Prints the chip census — one line per SUB-PROOF, since `KECCAK_RND`'s chunks
/// are separate AIRs at separate heights — and the totals the hash matrix wants.
/// Returns `(main_cells, aux_cells)` so a caller can assert on them.
pub(super) fn report_census(label: &str, program: &LfmProgram) -> (u64, u64) {
    let census = lfm_chip_census(program);
    let (main, aux) = lfm_cell_counts(program);
    // The census is `lfm_cell_counts`' own decomposition, so summing it is not an
    // independent check of the total — it is the same arithmetic. What IS
    // independent is that the sub-proof COUNT the census implies must equal the
    // AIR count the verifier builds from the program's chunk policy.
    assert_eq!(
        census.len(),
        super::airs::num_lfm_airs(
            program
                .chunking
                .chunk_count(program.groups.keccak.real_rows)
        ),
        "the census must have one entry per sub-proof the AIR set builds"
    );
    println!("\n★ CHIP CENSUS — {label}");
    println!(
        "   {:>12} {:>10} {:>6} {:>6} {:>16} {:>14}",
        "chip", "rows", "main", "aux", "main cells", "aux cells"
    );
    for c in &census {
        println!(
            "   {:>12} {:>10} {:>6} {:>6} {:>16} {:>14}",
            c.name,
            c.rows,
            c.main_cols,
            c.aux_cols,
            c.main_cells(),
            c.aux_cells()
        );
    }
    println!(
        "   {:>12} {:>10} {:>6} {:>6} {:>16} {:>14}",
        "TOTAL",
        census.iter().map(|c| c.rows).sum::<u64>(),
        "",
        "",
        main,
        aux
    );
    println!(
        "   cells per verify = {main} main + {aux} aux ext = {} base-field equivalents \
         (an ext element is 3 base felts)",
        main + 3 * aux
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
    let t_epoch = Instant::now();
    let e = super::epoch_tests::real_epoch();
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

    let program = super::epoch_tests::epoch_program(&e, true);
    let arenas = super::epoch_tests::epoch_arena_words(&e, true);
    report_program("THE WRAPPED PROGRAM", &profile, &program);
    report_census(
        &format!("assembled epoch verifier, epoch {profile}"),
        &program,
    );

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
        ),
        "the wrap proof must verify"
    );
    let verify_secs = t.elapsed().as_secs_f64();
    println!(
        "\n★ WRAP PROVED AND VERIFIED (inner epoch {profile})\n   \
         prove {prove_secs:.1}s / verify {verify_secs:.2}s / proof {size} bytes / \
         {} published words / {} sub-proofs",
        proved.public_words.len(),
        proved.proof.proofs.len(),
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
        ),
        "a moved program digest must make the wrap proof UNVERIFIABLE"
    );
    println!("   MOVED program digest: the wrap proof is UNVERIFIABLE");
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
    let mut built: Vec<(usize, usize)> = vec![
        dims(&traces.const_),
        dims(&traces.balu),
        dims(&traces.xalu),
        dims(&traces.select),
        dims(&traces.bitdec),
        dims(&traces.hash),
        dims(&traces.keccak),
        dims(&traces.lanes),
        dims(&traces.hint),
        dims(&traces.public),
        dims(&traces.range),
    ];
    built.extend(traces.keccak_rnd.iter().map(dims));
    built.push(dims(&traces.keccak_rc));
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
    let airs = super::airs::LfmAirs::new(&artifacts.roots, &opts, artifacts.keccak_rnd_chunks);
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
