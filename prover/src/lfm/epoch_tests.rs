//! The assembled verifier's challenge replay, differentialled against
//! production's own.
//!
//! ## The oracle
//!
//! `Verifier::replay_rounds_after_round_1` — the function `multi_verify_views`
//! itself calls. Nothing here models Fiat-Shamir; the expected `β`, `z`, `γ`,
//! `ζ_k` and `ι_s` are the values the production verifier computed for a real
//! proof of a real AIR, and the machine is asked to reproduce them from the
//! proof's own bytes.
//!
//! ## What this suite can see that no leg-side suite could
//!
//! The FRI commit phase's INTERLEAVING (ledger entry 4). `fri_tests` supplies
//! `ζ_k` from the same replay it checks against, so absorbing the layer roots
//! in the wrong order — or not at all — moves nothing there. Here every
//! challenge is derived from the absorbed bytes, so a misordered absorb changes
//! `ζ`, and a `ζ` change moves the fold. The four fixtures span
//! `num_committed = 0, 1, 2, 3`, so the loop is exercised at zero, one and
//! several layers.
//!
//! ## What it cannot see
//!
//! It stops at the challenges. That the legs then CONSUME these cells is
//! [`the_legs_consume_the_replayed_challenges`]'s job, and the whole-epoch
//! composition (24 sub-proofs behind one statement) is not built here.

use stark::config::Commitment;
use stark::proof::stark::MultiProof;
use stark::proof::view::StarkProofView;
use stark::traits::AIR;

use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};

use super::builder::LfmBuilder;
use super::compiler::{LfmProgram, compile};
use super::edsl;
use super::epoch::{
    RootCells, TableAbsorbs, TableChallengeShape, emit_table_challenges, fork_table,
};
use super::executor::execute;
use super::fri::FriShape;
use super::hash::TestPermutation;
use super::instr::ArenaId;
use super::transcript_replay::TranscriptReplay;
use super::validator::validate;
use super::word::{LfmWord, base_word, ext_word, word_as_ext};

type Gl = GoldilocksField;
type Ext3 = GoldilocksExtension;

/// Everything one real sub-proof supplies to the replay, plus the challenges
/// production derived from it.
#[derive(Clone)]
pub(super) struct HostTable {
    pub(super) shape: TableChallengeShape,
    /// The verifier's HARDCODED precomputed commitment, when the AIR is
    /// preprocessed. A program constant, not arena data: the verifier does not
    /// take this from the proof (`verifier.rs:1187`).
    precomputed_root: Option<Commitment>,
    main_root: Commitment,
    aux_root: Option<Commitment>,
    pub(super) contribution: Option<FEE>,
    composition_root: Commitment,
    /// Row-major, as `row_major_data` carries it.
    pub(super) ood_current: Vec<FEE>,
    pub(super) ood_next: Vec<FEE>,
    pub(super) parts: Vec<FEE>,
    fri_roots: Vec<Commitment>,
    pub(super) fri_coeffs: Vec<FEE>,
    nonce: Option<u64>,
    needs_lookup_challenges: bool,

    // ---- the oracle ----
    pub(super) beta: FEE,
    pub(super) z: FEE,
    pub(super) gamma: FEE,
    pub(super) zetas: Vec<FEE>,
    pub(super) iotas: Vec<usize>,
}

/// Read a real single-table proof into [`HostTable`], taking the challenges
/// from the production verifier rather than recomputing them.
fn host_table(
    air: &dyn AIR<Field = Gl, FieldExtension = Ext3, PublicInputs = ()>,
    proof: &MultiProof<Gl, Ext3, ()>,
) -> HostTable {
    let sp = super::constraint_tests::open_sub_proof(air, proof);
    let view = StarkProofView::Owned(&proof.proofs[0]);
    let opts = air.options();

    let trace_length = view.trace_length();
    let log2_trace_length = trace_length.trailing_zeros();
    let log2_blowup = (opts.blowup_factor as usize).trailing_zeros();
    let fri = FriShape::from_options(opts, log2_trace_length + log2_blowup);

    let ood_c = view.trace_ood_evaluations();
    let ood_n = view.trace_ood_next_evaluations();

    // `γ` comes back from the DEEP shape derivation, which recovers it from the
    // verifier's own coefficient run — the same route `join_tests` uses.
    let (_deep, gamma) = super::constraint_tests::deep_shape(&sp, air);

    let shape = TableChallengeShape {
        index: 0,
        num_tables: 1,
        has_aux_root: view.lde_trace_aux_merkle_root().is_some(),
        has_contribution: view.bus_table_contribution().is_some(),
        log2_trace_length,
        log2_blowup,
        coset_offset: FE::from(opts.coset_offset),
        ood_current_dims: (ood_c.width(), ood_c.height()),
        ood_next_dims: (ood_n.width(), ood_n.height()),
        num_parts: view.composition_poly_parts_ood_evaluation().len(),
        fri,
        grinding_factor: opts.grinding_factor,
        num_queries: opts.fri_number_of_queries,
    };

    HostTable {
        shape,
        precomputed_root: air.is_preprocessed().then(|| air.precomputed_commitment()),
        main_root: *view.lde_trace_main_merkle_root(),
        aux_root: view.lde_trace_aux_merkle_root().copied(),
        contribution: view.bus_table_contribution(),
        composition_root: *view.composition_poly_root(),
        ood_current: ood_c.row_major_data().to_vec(),
        ood_next: ood_n.row_major_data().to_vec(),
        parts: view.composition_poly_parts_ood_evaluation().to_vec(),
        fri_roots: view.fri_layers_merkle_roots().to_vec(),
        fri_coeffs: view.fri_final_poly_coeffs().to_vec(),
        nonce: view.nonce(),
        needs_lookup_challenges: air.has_aux_trace(),
        beta: sp.beta,
        z: sp.challenges.z,
        gamma,
        zetas: sp.challenges.zetas.clone(),
        iotas: sp.challenges.iotas.clone(),
    }
}

/// Arena identifiers of the challenge program, in declaration order.
struct Arenas {
    main_root: ArenaId,
    aux_root: Option<ArenaId>,
    contribution: Option<ArenaId>,
    composition_root: ArenaId,
    ood_current: ArenaId,
    ood_next: ArenaId,
    parts: ArenaId,
    fri_roots: ArenaId,
    fri_coeffs: ArenaId,
    nonce: Option<ArenaId>,
    /// The verification legs' two arenas, present only in the ASSEMBLED
    /// verifier — the trace openings and the FRI layer openings. `None` in the
    /// spine-only program, which verifies nothing and so opens nothing.
    legs: Option<super::epoch_verify::TableQueryArenas>,
}

/// A program that replays ONE table's challenges and publishes them.
///
/// The transcript prefix is `multi_verify_views`' single-table Phase A: the
/// hardcoded precomputed commitment when the AIR is preprocessed, the main
/// root, then the shared LogUp challenges. The fork follows, then rounds 2-4.
fn challenge_program(h: &HostTable) -> LfmProgram {
    let mut b = LfmBuilder::new();
    let shape = &h.shape;

    let a = Arenas {
        main_root: b.declare_arena(2),
        aux_root: shape.has_aux_root.then(|| b.declare_arena(2)),
        contribution: shape.has_contribution.then(|| b.declare_arena(1)),
        composition_root: b.declare_arena(2),
        ood_current: b.declare_arena((shape.ood_current_dims.0 * shape.ood_current_dims.1) as u32),
        ood_next: b.declare_arena((shape.ood_next_dims.0 * shape.ood_next_dims.1) as u32),
        parts: b.declare_arena(shape.num_parts as u32),
        fri_roots: b.declare_arena(2 * shape.fri.num_committed() as u32),
        fri_coeffs: b.declare_arena(shape.fri.num_terminal_coeffs() as u32),
        nonce: (shape.grinding_factor > 0).then(|| b.declare_arena(1)),
        legs: None,
    };

    let mut t = TranscriptReplay::new(&[]);
    if let Some(prep) = h.precomputed_root {
        t.append_const_bytes(&prep);
    }
    let main = RootCells::hint(&mut b, a.main_root, 0);
    t.append_halves(&main.halves());
    if h.needs_lookup_challenges {
        for _ in 0..stark::lookup::LOGUP_NUM_CHALLENGES {
            t.sample_ext(&mut b);
        }
    }

    let aux = a.aux_root.map(|id| RootCells::hint(&mut b, id, 0));
    let contribution = a.contribution.map(|id| b.hint_word(id, 0).as_ext());
    let composition = RootCells::hint(&mut b, a.composition_root, 0);
    let ood_current: Vec<_> = (0..(shape.ood_current_dims.0 * shape.ood_current_dims.1) as u32)
        .map(|i| b.hint_word(a.ood_current, i).as_ext())
        .collect();
    let ood_next: Vec<_> = (0..(shape.ood_next_dims.0 * shape.ood_next_dims.1) as u32)
        .map(|i| b.hint_word(a.ood_next, i).as_ext())
        .collect();
    let parts: Vec<_> = (0..shape.num_parts as u32)
        .map(|i| b.hint_word(a.parts, i).as_ext())
        .collect();
    let fri_roots: Vec<_> = (0..shape.fri.num_committed())
        .map(|i| RootCells::hint(&mut b, a.fri_roots, 2 * i as u32))
        .collect();
    let fri_coeffs: Vec<_> = (0..shape.fri.num_terminal_coeffs() as u32)
        .map(|i| b.hint_word(a.fri_coeffs, i).as_ext())
        .collect();
    let nonce = a.nonce.map(|id| b.hint_felt(id, 0));

    let mut fork = fork_table(&t, shape.index, shape.num_tables);
    let ch = emit_table_challenges(
        &mut b,
        &mut fork,
        shape,
        &TableAbsorbs {
            aux_root: aux.as_ref(),
            contribution,
            composition_root: &composition,
            ood_current: &ood_current,
            ood_next: &ood_next,
            parts: &parts,
            fri_roots: &fri_roots,
            fri_coeffs: &fri_coeffs,
            nonce,
        },
    );

    b.public(ch.beta.as_cell());
    b.public(ch.z.as_cell());
    b.public(ch.gamma.as_cell());
    for zeta in &ch.zetas {
        b.public(zeta.as_cell());
    }
    for bits in &ch.iota_bits {
        let felt = edsl::bits_to_felt(&mut b, bits);
        b.public(felt.as_cell());
    }

    let program = compile(b.finish());
    validate(&program).expect("the challenge program must be admissible");
    program
}

/// The arenas [`challenge_program`] declares, in the same order.
fn challenge_arenas(h: &HostTable) -> Vec<Vec<LfmWord>> {
    let mut out = vec![super::proof_arena::commitments_to_arena(&[h.main_root])];
    if let Some(r) = h.aux_root {
        out.push(super::proof_arena::commitments_to_arena(&[r]));
    }
    if let Some(c) = h.contribution {
        out.push(vec![ext_word(&c)]);
    }
    out.push(super::proof_arena::commitments_to_arena(&[
        h.composition_root
    ]));
    out.push(h.ood_current.iter().map(ext_word).collect());
    out.push(h.ood_next.iter().map(ext_word).collect());
    out.push(h.parts.iter().map(ext_word).collect());
    out.push(super::proof_arena::commitments_to_arena(&h.fri_roots));
    out.push(h.fri_coeffs.iter().map(ext_word).collect());
    if let Some(n) = h.nonce {
        out.push(vec![base_word(FE::from(n))]);
    }
    out
}

/// Run the program and read the published challenges back.
fn run(h: &HostTable) -> (FEE, FEE, FEE, Vec<FEE>, Vec<u64>) {
    let program = challenge_program(h);
    let arenas = challenge_arenas(h);
    let exec = execute(&program, &arenas, &TestPermutation).expect("the replay must execute");

    let pub_ext = |i: usize| word_as_ext(&exec.public_words[i].1).expect("an ext challenge");
    let beta = pub_ext(0);
    let z = pub_ext(1);
    let gamma = pub_ext(2);
    let zetas: Vec<FEE> = (0..h.zetas.len()).map(|k| pub_ext(3 + k)).collect();
    let base = 3 + h.zetas.len();
    let iotas: Vec<u64> = (0..h.shape.num_queries)
        .map(|q| {
            let w = exec.public_words[base + q].1;
            let felt = super::word::word_as_base(&w).expect("an index is a base felt");
            felt.to_raw()
        })
        .collect();
    (beta, z, gamma, zetas, iotas)
}

/// ★ The whole point of the leg: every challenge the legs consume is the one
/// production derived, and it came out of the transcript rather than an arena.
///
/// Swept over four real proofs whose committed FRI layer counts are 0, 1, 2 and
/// 3, because the Round-4 interleaving only has anything to get wrong once a
/// layer exists.
#[test]
fn the_challenge_replay_matches_production() {
    for (boundaries, committed) in [(4usize, 0usize), (512, 1), (1024, 2), (2048, 3)] {
        let (air, proof) = super::fri_tests::folding_fixture(boundaries, 2);
        let h = host_table(&*air, &proof);
        assert_eq!(
            h.shape.fri.num_committed(),
            committed,
            "fixture of {boundaries} boundaries must commit {committed} layers"
        );
        assert_eq!(
            h.zetas.len(),
            if h.shape.fri.total_folds() > 0 {
                committed + 1
            } else {
                0
            },
            "folds exceed committed layers by one, and vanish when nothing folds"
        );

        let (beta, z, gamma, zetas, iotas) = run(&h);
        assert_eq!(beta, h.beta, "beta at {boundaries} boundaries");
        assert_eq!(z, h.z, "z at {boundaries} boundaries");
        assert_eq!(gamma, h.gamma, "gamma at {boundaries} boundaries");
        assert_eq!(zetas, h.zetas, "the FRI zetas at {boundaries} boundaries");
        let want: Vec<u64> = h.iotas.iter().map(|i| *i as u64).collect();
        assert_eq!(iotas, want, "the query indices at {boundaries} boundaries");
    }
}

/// ★ Two defects the differential above CANNOT see, pinned so they are not
/// mistaken for coverage.
///
/// Both are degenerate parameters of the single-table L2G fixture, and the
/// falsification runs found them rather than reasoning predicting them:
/// injecting a ROW-major OOD absorb and deleting the fork's domain separator
/// both left `the_challenge_replay_matches_production` green.
#[test]
fn the_single_table_fixture_is_blind_to_two_defects() {
    let (air, proof) = super::fri_tests::folding_fixture(2048, 2);
    let h = host_table(&*air, &proof);
    println!(
        "ood_current {:?}  ood_next {:?}  num_tables {}",
        h.shape.ood_current_dims, h.shape.ood_next_dims, h.shape.num_tables
    );
    assert_eq!(
        h.shape.ood_current_dims.1, 1,
        "a one-ROW OOD block reads the same column-major as row-major, so this \
         fixture cannot witness the absorb order"
    );
    assert_eq!(
        h.shape.ood_next_dims.1, 1,
        "likewise for the next-row block"
    );
    assert_eq!(
        h.shape.num_tables, 1,
        "production emits no domain separator at one table (verifier.rs:1264), \
         so this fixture cannot witness the fork's separator"
    );
}

/// ★ The `z` guard, driven with the points production rejects.
///
/// `sample_z_ood_with_domain_params` loops until `z` is outside both the trace
/// domain and the LDE coset. The machine cannot loop, so it constrains the
/// first draw — and that constraint is unreachable from a real transcript,
/// which is why the guard is emitted against a HINTED `z` here. Both rejection
/// branches are exercised separately, with a generic `z` as the positive
/// control: without it, a guard that rejected everything would look identical.
#[test]
fn the_z_guard_rejects_a_point_in_either_domain() {
    use math::field::traits::IsFFTField;

    let shape = TableChallengeShape {
        index: 0,
        num_tables: 1,
        has_aux_root: false,
        has_contribution: false,
        log2_trace_length: 4,
        log2_blowup: 1,
        coset_offset: FE::from(3u64),
        ood_current_dims: (1, 1),
        ood_next_dims: (0, 0),
        num_parts: 1,
        fri: FriShape::from_options(&super::proof_fixture::fixture_options(), 5),
        grinding_factor: 0,
        num_queries: 1,
    };

    let program = {
        let mut b = LfmBuilder::new();
        let a = b.declare_arena(1);
        let z = b.hint_word(a, 0).as_ext();
        super::epoch::assert_z_outside_domains(&mut b, z, &shape);
        let program = compile(b.finish());
        validate(&program).expect("the guard program must be admissible");
        program
    };
    let runs = |z: FEE| execute(&program, &[vec![ext_word(&z)]], &TestPermutation).is_ok();

    // Positive control: a generic point passes, so a guard that rejected
    // everything would not be mistaken for a working one.
    assert!(
        runs(FEE::new([FE::from(7u64), FE::from(11u64), FE::from(13u64)])),
        "a generic z must pass both non-memberships"
    );

    // In the trace domain: a 16th root of unity, so z^16 = 1.
    let g = <Gl as IsFFTField>::get_primitive_root_of_unity(4).expect("root of unity");
    for k in [0u64, 1, 5] {
        let z = g.pow(k).to_extension::<Ext3>();
        assert!(
            !runs(z),
            "z = g^{k} is in the trace domain and production would have redrawn"
        );
    }

    // On the LDE coset: z = offset · ω^k with ω the 32nd root of unity, so
    // z^32 = offset^32.
    let w = <Gl as IsFFTField>::get_primitive_root_of_unity(5).expect("root of unity");
    for k in [0u64, 3, 17] {
        let z = (FE::from(3u64) * w.pow(k)).to_extension::<Ext3>();
        assert!(
            !runs(z),
            "z = 3·ω^{k} is on the LDE coset and production would have redrawn"
        );
    }
}

/// ★ The grinding check, which the challenge differential is structurally
/// blind to.
///
/// A wrong nonce changes the query indices, so the differential above would
/// simply compare different-but-consistent values; deleting
/// `emit_grinding_check` entirely left it green (falsification run
/// `grinding_check`). What makes a wrong nonce REJECT is the proof-of-work
/// predicate, and this drives it: the proof's own nonce runs, and eight
/// neighbouring nonces — which absorb just as happily — do not.
///
/// This matters beyond tidiness. The nonce is absorbed before the query
/// indices are drawn, so an unchecked nonce is a free re-roll of every query
/// index: a prover with a bad codeword re-grinds until the indices miss it,
/// at the cost production charges 2^20 hashes for.
#[test]
fn a_nonce_that_did_not_grind_is_rejected() {
    let (air, proof) = super::fri_tests::folding_fixture(4, 2);
    let h = host_table(&*air, &proof);
    assert!(
        h.shape.grinding_factor > 0,
        "the fixture must actually grind, or this test proves nothing"
    );
    let real = h.nonce.expect("a grinding proof carries a nonce");

    let program = challenge_program(&h);
    let runs = |nonce: u64| {
        let mut h2 = h.clone();
        h2.nonce = Some(nonce);
        execute(&program, &challenge_arenas(&h2), &TestPermutation).is_ok()
    };

    assert!(runs(real), "the proof's own nonce must satisfy the check");
    let mut rejected = 0;
    for delta in 1..=8u64 {
        if !runs(real.wrapping_add(delta)) {
            rejected += 1;
        }
    }
    assert_eq!(
        rejected, 8,
        "at a grinding factor of {} a neighbouring nonce passes with \
         probability 2^-{}, so all eight must be rejected",
        h.shape.grinding_factor, h.shape.grinding_factor
    );
}

// =============================================================================
// The whole epoch: one statement, Phase A over every sub-proof, then a fork per
// table.
// =============================================================================

/// A real continuation epoch, proved over the production epoch AIR set, with
/// production's own per-table challenges extracted for every sub-proof.
///
/// Built the way `prove_continuation` builds epoch 0 — the same construction
/// `logup_tests::a_zero_row_fixed_table_carries_some_zero_not_none` proves and
/// production ACCEPTS. Nothing here is synthetic: the statement is the real
/// one, the forks carry the real domain separators, and the challenges come
/// from `replay_rounds_after_round_1` on each fork.
pub(super) struct RealEpoch {
    pub(super) statement: super::statement_replay::EpochStatementShape,
    elf_digest: [u8; 32],
    pub(super) public_output: Vec<u8>,
    epoch_label: u64,
    /// Per table, in sub-proof order: the hardcoded precomputed commitment
    /// (when the AIR is preprocessed) and the proof's main trace root.
    phase_a: Vec<(Option<Commitment>, Commitment)>,
    /// Per table, everything the fork absorbs plus the oracle challenges.
    pub(super) tables: Vec<HostTable>,
    /// Per table, everything the VERIFICATION LEGS read — the shapes, the
    /// constraint analysis and the per-query openings. Built in the same pass as
    /// `tables` because it needs the AIRs and the proof view, which do not
    /// outlive this function.
    pub(super) legs: Vec<super::epoch_verify_tests::TableLegs>,
    /// The shared LogUp challenges Phase A ends on.
    pub(super) z_alpha: (FEE, FEE),
    /// The carried commit index — `reg_init[X254_INDEX]` of this epoch, which
    /// is the PREVIOUS epoch's `reg_fini[64]`.
    pub(super) start_index: u64,
    /// The COMMIT-bus target production computed, and therefore the value the
    /// closure must reach.
    pub(super) expected_bus_balance: FEE,
}

pub(super) fn real_epoch() -> RealEpoch {
    use crate::tables::trace_builder::{Traces, build_initial_image_paged};
    use crate::tables::{MaxRowsConfig, bitwise, local_to_global, register};
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use crypto::fiat_shamir::is_transcript::IsTranscript;
    use executor::elf::Elf;
    use executor::vm::execution::Executor;
    use stark::proof::view::MultiProofView;
    use stark::verifier::IsStarkVerifier;

    let opts = super::proof_fixture::fixture_options();
    let elf_bytes = super::proof_fixture::read_inner_elf();
    let elf = Elf::load(&elf_bytes).expect("the fixture ELF must load");
    let epoch_size = 1usize << super::proof_fixture::FIXTURE_EPOCH_LOG2;

    let mut executor = Executor::new(&elf, vec![]).expect("executor");
    let image = build_initial_image_paged(&elf, &[]);
    let register_init = register::register_init_from_entry_point(elf.entry_point);
    let logs = executor
        .resume_with_limit(epoch_size)
        .expect("resume")
        .expect("the guest runs at least one epoch")
        .to_vec();
    let is_final = executor.pc() == 0;
    assert!(!is_final, "wanted an INTERMEDIATE epoch");

    let mut traces = Traces::from_image_and_logs(
        &elf,
        &image,
        &register_init,
        &logs,
        &MaxRowsConfig::default(),
        &[],
        is_final,
        true,
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .expect("the epoch trace must build");

    let label = local_to_global::epoch_label(0);
    let mut provenance =
        local_to_global::genesis_provenance(image.iter().map(|(a, v)| (a, v as u64)));
    let boundary =
        local_to_global::epoch_boundary(&mut provenance, label, &traces.touched_memory_cells);
    bitwise::update_multiplicities(
        &mut traces.bitwise,
        &local_to_global::collect_bitwise_from_l2g(&boundary),
    );

    let reg_fini = register::fini_from_trace(&traces.register);
    let table_counts = traces.table_counts();
    let public_output = traces.public_output_bytes.clone();
    let runtime_page_ranges = traces.runtime_page_ranges();

    let airs = crate::VmAirs::new(
        &elf,
        &opts,
        false,
        &[],
        &table_counts,
        None,
        is_final,
        None,
        None,
        Some((
            register::compute_precomputed_commitment_with_fini(&opts, &register_init, &reg_fini),
            register::NUM_PREPROCESSED_COLS_WITH_FINI,
        )),
    );
    let l2g_air = crate::continuation::l2g_memory_air(&opts, label);
    let mut l2g_trace = local_to_global::generate_local_to_global_trace(&boundary);

    let seed = || {
        let mut t = DefaultTranscript::<Ext3>::new(&[]);
        crate::statement::absorb_statement(
            &mut t,
            crate::statement::StatementKind::ContinuationEpoch { epoch_label: label },
            &elf_bytes,
            &public_output,
            &table_counts,
            0,
            &runtime_page_ranges,
            opts.fri_final_poly_log_degree,
        );
        t
    };

    let proof = {
        let mut pairs = airs.air_trace_pairs(&mut traces);
        pairs.push((&l2g_air, &mut l2g_trace, &()));
        crate::test_utils::multi_prove_ram(pairs, &mut seed()).expect("the epoch must prove")
    };
    let refs = {
        let mut r = airs.air_refs();
        r.push(&l2g_air);
        r
    };
    let view = MultiProofView::Owned(&proof);
    assert_eq!(refs.len(), view.len(), "one AIR per sub-proof");

    // ---- production must ACCEPT it, or nothing below describes a real epoch.
    let start_index = register_init[register::X254_INDEX] as u64;
    let expected = crate::compute_expected_commit_bus_balance_view(
        &refs,
        view,
        &public_output,
        start_index,
        &mut seed(),
    )
    .expect("the COMMIT bus target must compute");
    assert!(
        stark::verifier::Verifier::multi_verify_views(&refs, view, &mut seed(), &expected),
        "production must accept the epoch this suite differentials against"
    );

    // ---- Phase A, transcribed from `multi_verify_views:1160-1227`.
    let mut transcript = seed();
    let mut phase_a = Vec::new();
    for (idx, air) in refs.iter().enumerate() {
        let v = view.get(idx);
        if air.is_preprocessed() {
            let prep = air.precomputed_commitment();
            transcript.append_bytes(&prep);
            transcript.append_bytes(v.lde_trace_main_merkle_root());
            phase_a.push((Some(prep), *v.lde_trace_main_merkle_root()));
        } else {
            transcript.append_bytes(v.lde_trace_main_merkle_root());
            phase_a.push((None, *v.lde_trace_main_merkle_root()));
        }
    }
    let needs_lookup_challenges = refs.iter().any(|a| a.has_aux_trace());
    assert!(needs_lookup_challenges, "an epoch uses LogUp");
    let lookup_challenges: Vec<FEE> = (0..stark::lookup::LOGUP_NUM_CHALLENGES)
        .map(|_| transcript.sample_field_element())
        .collect();
    let z_alpha = (lookup_challenges[0], lookup_challenges[1]);

    // ---- one fork per table, and the rounds replayed on it.
    let num_tables = refs.len();
    let tables = refs
        .iter()
        .enumerate()
        .map(|(idx, air)| {
            let v = view.get(idx);
            let mut fork = transcript.clone();
            if num_tables > 1 {
                fork.append_bytes(&(idx as u64).to_le_bytes());
            }
            if let Some(root) = v.lde_trace_aux_merkle_root() {
                fork.append_bytes(root);
            }
            if let Some(c) = v.bus_table_contribution() {
                fork.append_field_element(&c);
            }
            host_table_forked(*air, v, idx, num_tables, &mut fork, &lookup_challenges)
        })
        .collect();

    // The legs' own reading of the same sub-proofs. Separate pass rather than
    // part of `host_table_forked` because the challenge replay must run against
    // a fork positioned exactly as production leaves it, and this reads nothing
    // from the transcript at all.
    let legs = refs
        .iter()
        .enumerate()
        .map(|(idx, air)| {
            super::epoch_verify_tests::build_table_legs(*air, view.get(idx), &lookup_challenges)
        })
        .collect();

    RealEpoch {
        statement: super::statement_replay::EpochStatementShape {
            public_output_len: public_output.len(),
            table_counts: [
                table_counts.cpu as u64,
                table_counts.lt as u64,
                table_counts.memw as u64,
                table_counts.memw_aligned as u64,
                table_counts.load as u64,
                table_counts.mul as u64,
                table_counts.dvrm as u64,
                table_counts.shift as u64,
                table_counts.branch as u64,
                table_counts.memw_register as u64,
                table_counts.eq as u64,
                table_counts.bytewise as u64,
                table_counts.store as u64,
                table_counts.cpu32 as u64,
            ],
            num_private_input_pages: 0,
            fri_final_poly_log_degree: opts.fri_final_poly_log_degree,
            page_ranges: runtime_page_ranges
                .iter()
                .map(|r| (r.base, r.count))
                .collect(),
        },
        elf_digest: crate::statement::elf_digest(&elf_bytes),
        public_output,
        epoch_label: label,
        phase_a,
        tables,
        legs,
        z_alpha,
        start_index,
        expected_bus_balance: expected,
    }
}

/// [`host_table`] for a sub-proof inside a multi-table epoch: the fork is
/// already positioned (separator, aux root and `L` absorbed), so the oracle
/// comes from `replay_rounds_after_round_1` on THAT transcript.
fn host_table_forked(
    air: &dyn AIR<Field = Gl, FieldExtension = Ext3, PublicInputs = ()>,
    view: StarkProofView<'_, Gl, Ext3, ()>,
    index: usize,
    num_tables: usize,
    fork: &mut crypto::fiat_shamir::default_transcript::DefaultTranscript<Ext3>,
    lookup_challenges: &[FEE],
) -> HostTable {
    use stark::domain::new_verifier_domain;
    use stark::verifier::IsStarkVerifier;
    use stark::verifier::Verifier;

    let opts = air.options();
    let trace_length = view.trace_length();
    let log2_trace_length = trace_length.trailing_zeros();
    let log2_blowup = (opts.blowup_factor as usize).trailing_zeros();
    let domain = new_verifier_domain(air, trace_length);
    let layout = Verifier::<Gl, Ext3, ()>::ood_layout(air);
    let challenges = Verifier::<Gl, Ext3, ()>::replay_rounds_after_round_1(
        air,
        view,
        &(),
        &domain,
        fork,
        lookup_challenges.to_vec(),
        &layout,
    );

    let nt = challenges.transition_coeffs.len();
    let beta = if nt > 1 {
        challenges.transition_coeffs[1]
    } else {
        challenges.boundary_coeffs[0]
    };
    // `γ` is the second term of the DEEP coefficient run, which starts at one —
    // the same recovery `constraint_tests::deep_shape` makes.
    let gamma = challenges.trace_term_coeffs[1][0];

    let ood_c = view.trace_ood_evaluations();
    let ood_n = view.trace_ood_next_evaluations();
    let shape = TableChallengeShape {
        index,
        num_tables,
        has_aux_root: view.lde_trace_aux_merkle_root().is_some(),
        has_contribution: view.bus_table_contribution().is_some(),
        log2_trace_length,
        log2_blowup,
        coset_offset: FE::from(opts.coset_offset),
        ood_current_dims: (ood_c.width(), ood_c.height()),
        ood_next_dims: (ood_n.width(), ood_n.height()),
        num_parts: view.composition_poly_parts_ood_evaluation().len(),
        fri: FriShape::from_options(opts, log2_trace_length + log2_blowup),
        grinding_factor: opts.grinding_factor,
        num_queries: opts.fri_number_of_queries,
    };

    HostTable {
        shape,
        precomputed_root: air.is_preprocessed().then(|| air.precomputed_commitment()),
        main_root: *view.lde_trace_main_merkle_root(),
        aux_root: view.lde_trace_aux_merkle_root().copied(),
        contribution: view.bus_table_contribution(),
        composition_root: *view.composition_poly_root(),
        ood_current: ood_c.row_major_data().to_vec(),
        ood_next: ood_n.row_major_data().to_vec(),
        parts: view.composition_poly_parts_ood_evaluation().to_vec(),
        fri_roots: view.fri_layers_merkle_roots().to_vec(),
        fri_coeffs: view.fri_final_poly_coeffs().to_vec(),
        nonce: view.nonce(),
        needs_lookup_challenges: true,
        beta,
        z: challenges.z,
        gamma,
        zetas: challenges.zetas.clone(),
        iotas: challenges.iotas.clone(),
    }
}

/// The whole epoch's Fiat-Shamir spine, as one program.
///
/// Statement, then Phase A over every sub-proof, then a fork per table and its
/// rounds 2-4. This is the assembled verifier's skeleton: what hangs off each
/// fork (constraint evaluation, the query legs, the closure) consumes the cells
/// this returns.
///
/// ## ⚠ The preprocessed commitments are HINTED here, and they must not stay so
///
/// Production takes each preprocessed root from the AIR
/// (`verifier.rs:1187`), never from the proof, and rejects the sub-proof unless
/// the proof's copy matches. Only one of those roots has an in-machine
/// derivation today — REGISTER's, from the previous epoch's `reg_fini`
/// (`programs::register_derivation_program`). The others (BITWISE, DECODE,
/// KECCAK_RC, PAGE) are hinted, and PAGE's in particular CANNOT become a
/// program constant: it is a function of the inner ELF, which is per-proof arena
/// data. Baking it would make program identity proof-dependent. So each is a
/// derivation the assembly still owes; see the ledger entry this leg added.
fn epoch_challenge_program(e: &RealEpoch) -> LfmProgram {
    epoch_program(e, false)
}

/// The epoch program, with or without the verification LEGS hung off the spine.
///
/// One emitter for both, deliberately. A second copy of the spine would be a
/// place for the assembled verifier's Fiat-Shamir to drift from the one
/// `the_epoch_challenge_spine_matches_production` checks against production —
/// and drift is exactly what the leg wiring must not introduce, since every leg
/// consumes the cells this spine bound. `with_legs = false` declares no leg
/// arenas and emits no verification, so the spine test's own arena-word count is
/// untouched.
pub(super) fn epoch_program(e: &RealEpoch, with_legs: bool) -> LfmProgram {
    use super::statement_replay::{EpochStatementVars, PhaseATable, absorb_epoch_statement};

    let mut b = LfmBuilder::new();
    let n = e.tables.len();
    assert_eq!(e.legs.len(), n, "one leg reading per sub-proof");

    // ---- arenas, in declaration order ----
    let stmt_halves = 8 + e.statement.public_output_len.div_ceil(4) + 2;
    let a_stmt = b.declare_arena(stmt_halves as u32);
    let num_prep = e.phase_a.iter().filter(|(p, _)| p.is_some()).count();
    let a_prep_roots = b.declare_arena(2 * num_prep as u32);
    let a_main_roots = b.declare_arena(2 * n as u32);
    // The register boundary vector, declared at production's width. Only the
    // carried commit index is READ today; the rest is the arena the REGISTER
    // preprocessed derivation will consume, and declaring it here is what makes
    // `start_index` the same cell that derivation binds rather than a word of
    // its own.
    let a_reg_init = b.declare_arena(crate::tables::register::NUM_REGISTER_ADDRESSES as u32);
    let per_table: Vec<Arenas> = e
        .tables
        .iter()
        .zip(&e.legs)
        .map(|(h, leg)| Arenas {
            main_root: a_main_roots,
            aux_root: h.shape.has_aux_root.then(|| b.declare_arena(2)),
            contribution: h.shape.has_contribution.then(|| b.declare_arena(1)),
            composition_root: b.declare_arena(2),
            ood_current: b
                .declare_arena((h.shape.ood_current_dims.0 * h.shape.ood_current_dims.1) as u32),
            ood_next: b.declare_arena((h.shape.ood_next_dims.0 * h.shape.ood_next_dims.1) as u32),
            parts: b.declare_arena(h.shape.num_parts as u32),
            fri_roots: b.declare_arena(2 * h.shape.fri.num_committed() as u32),
            fri_coeffs: b.declare_arena(h.shape.fri.num_terminal_coeffs() as u32),
            nonce: (h.shape.grinding_factor > 0).then(|| b.declare_arena(1)),
            legs: with_legs
                .then(|| super::epoch_verify::declare_table_arenas(&mut b, &leg.verify)),
        })
        .collect();

    // ---- the statement ----
    let stmt: Vec<_> = (0..stmt_halves as u32)
        .map(|i| b.hint_felt(a_stmt, i))
        .collect();
    let out_halves = e.statement.public_output_len.div_ceil(4);
    let (elf_digest, rest) = stmt.split_at(8);
    let (public_output, epoch_label) = rest.split_at(out_halves);

    let mut t = TranscriptReplay::new(&[]);
    absorb_epoch_statement(
        &mut t,
        &e.statement,
        &EpochStatementVars {
            elf_digest,
            public_output,
            epoch_label,
        },
    );

    // ---- Phase A ----
    let prep_cells: Vec<RootCells> = (0..num_prep)
        .map(|i| RootCells::hint(&mut b, a_prep_roots, 2 * i as u32))
        .collect();
    let main_cells: Vec<RootCells> = (0..n)
        .map(|i| RootCells::hint(&mut b, a_main_roots, 2 * i as u32))
        .collect();
    let prep_halves: Vec<Vec<_>> = prep_cells.iter().map(RootCells::halves).collect();
    let main_halves: Vec<Vec<_>> = main_cells.iter().map(RootCells::halves).collect();
    let mut next_prep = 0usize;
    let tables: Vec<PhaseATable> = e
        .phase_a
        .iter()
        .enumerate()
        .map(|(i, (prep, _))| {
            let preprocessed_root = prep.map(|_| {
                let h = &prep_halves[next_prep][..];
                next_prep += 1;
                h
            });
            PhaseATable {
                preprocessed_root,
                main_root: &main_halves[i][..],
            }
        })
        .collect();
    let (z, alpha) = super::statement_replay::replay_phase_a(&mut t, &mut b, &tables);
    b.public(z.as_cell());
    b.public(alpha.as_cell());

    // ---- one fork per table ----
    let mut contributions: Vec<super::builder::Ext> = Vec::new();
    for (i, h) in e.tables.iter().enumerate() {
        let a = &per_table[i];
        let aux = a.aux_root.map(|id| RootCells::hint(&mut b, id, 0));
        let contribution = a.contribution.map(|id| b.hint_word(id, 0).as_ext());
        let composition = RootCells::hint(&mut b, a.composition_root, 0);
        let ood_current: Vec<_> = (0..(h.shape.ood_current_dims.0 * h.shape.ood_current_dims.1)
            as u32)
            .map(|k| b.hint_word(a.ood_current, k).as_ext())
            .collect();
        let ood_next: Vec<_> = (0..(h.shape.ood_next_dims.0 * h.shape.ood_next_dims.1) as u32)
            .map(|k| b.hint_word(a.ood_next, k).as_ext())
            .collect();
        let parts: Vec<_> = (0..h.shape.num_parts as u32)
            .map(|k| b.hint_word(a.parts, k).as_ext())
            .collect();
        let fri_roots: Vec<_> = (0..h.shape.fri.num_committed())
            .map(|k| RootCells::hint(&mut b, a.fri_roots, 2 * k as u32))
            .collect();
        let fri_coeffs: Vec<_> = (0..h.shape.fri.num_terminal_coeffs() as u32)
            .map(|k| b.hint_word(a.fri_coeffs, k).as_ext())
            .collect();
        let nonce = a.nonce.map(|id| b.hint_felt(id, 0));

        if let Some(c) = contribution {
            contributions.push(c);
        }
        let mut fork = fork_table(&t, h.shape.index, h.shape.num_tables);
        let absorbs = TableAbsorbs {
            aux_root: aux.as_ref(),
            contribution,
            composition_root: &composition,
            ood_current: &ood_current,
            ood_next: &ood_next,
            parts: &parts,
            fri_roots: &fri_roots,
            fri_coeffs: &fri_coeffs,
            nonce,
        };
        let ch = emit_table_challenges(&mut b, &mut fork, &h.shape, &absorbs);

        // ---- ★ THE SEAM: the verification legs, on the cells just absorbed and
        // the challenges just derived. `absorbs` is passed on by REFERENCE rather
        // than rebuilt, so there is no second reading of the proof for a leg to
        // disagree with the transcript about.
        if let Some(leg_arenas) = &a.legs {
            let leg = &e.legs[i];
            let out = super::epoch_verify::emit_table_verification(
                &mut b,
                &leg.verify,
                &leg.analysis,
                &ch,
                &absorbs,
                &super::epoch_verify::TableInputs {
                    // The precomputed root Phase A absorbed — the SAME cells,
                    // which is what makes production's explicit
                    // proof-copy-equals-AIR-copy check the absence of a second
                    // value here rather than a comparison.
                    precomputed_root: e.phase_a[i]
                        .0
                        .is_some()
                        .then(|| &prep_cells[e.phase_a[..i].iter().filter(|(p, _)| p.is_some()).count()]),
                    main_root: &main_cells[i],
                    rap_challenges: &[z, alpha],
                },
                leg_arenas,
            );
            b.public(out.composition.as_cell());
            for v in &out.fri_terminal {
                b.public(v.as_cell());
            }
        }
        b.public(ch.beta.as_cell());
        b.public(ch.z.as_cell());
        b.public(ch.gamma.as_cell());
        for zeta in &ch.zetas {
            b.public(zeta.as_cell());
        }
        for bits in &ch.iota_bits {
            let felt = edsl::bits_to_felt(&mut b, bits);
            b.public(felt.as_cell());
        }
    }

    // ---- the LogUp closure, on the cells the forks already absorbed ----
    //
    // Every `L` here is the cell its own fork bound into the transcript, and
    // the output bytes are derived from the halves the statement absorbed — so
    // the closure cannot be summing a different `L`, or folding a different
    // output, from the one the challenges were drawn against.
    let shape = super::logup::LogUpShape {
        num_contributing_tables: contributions.len(),
        num_output_bytes: e.statement.public_output_len,
    };
    let start = b.hint_felt(a_reg_init, crate::tables::register::X254_INDEX as u32);
    let bytes = super::epoch::emit_output_bytes(&mut b, public_output, shape.num_output_bytes);
    let target = super::logup::emit_commit_bus_target(&mut b, &shape, z, alpha, start, &bytes);
    let total = super::logup::emit_bus_closure(&mut b, &shape, &contributions, target);
    b.public(total.as_cell());

    let program = compile(b.finish());
    validate(&program).expect("the epoch challenge program must be admissible");
    program
}

/// The arenas [`epoch_challenge_program`] declares, in the same order.
fn epoch_arenas(e: &RealEpoch) -> Vec<Vec<LfmWord>> {
    epoch_arena_words(e, false)
}

/// The arenas [`epoch_program`] declares, in the same order.
pub(super) fn epoch_arena_words(e: &RealEpoch, with_legs: bool) -> Vec<Vec<LfmWord>> {
    let mut stmt: Vec<FE> = Vec::new();
    let halves = |bytes: &[u8]| -> Vec<FE> {
        bytes
            .chunks(4)
            .map(|c| {
                let mut w = [0u8; 4];
                w[..c.len()].copy_from_slice(c);
                FE::from(u32::from_le_bytes(w) as u64)
            })
            .collect()
    };
    stmt.extend(halves(&e.elf_digest));
    stmt.extend(halves(&e.public_output));
    stmt.extend(halves(&e.epoch_label.to_le_bytes()));

    let prep: Vec<Commitment> = e.phase_a.iter().filter_map(|(p, _)| *p).collect();
    let main: Vec<Commitment> = e.phase_a.iter().map(|(_, m)| *m).collect();

    let mut reg_init = vec![base_word(FE::zero()); crate::tables::register::NUM_REGISTER_ADDRESSES];
    reg_init[crate::tables::register::X254_INDEX] = base_word(FE::from(e.start_index));
    let mut out = vec![
        stmt.iter().map(|h| base_word(*h)).collect(),
        super::proof_arena::commitments_to_arena(&prep),
        super::proof_arena::commitments_to_arena(&main),
        reg_init,
    ];
    for (h, leg) in e.tables.iter().zip(&e.legs) {
        if let Some(r) = h.aux_root {
            out.push(super::proof_arena::commitments_to_arena(&[r]));
        }
        if let Some(c) = h.contribution {
            out.push(vec![ext_word(&c)]);
        }
        out.push(super::proof_arena::commitments_to_arena(&[
            h.composition_root
        ]));
        out.push(h.ood_current.iter().map(ext_word).collect());
        out.push(h.ood_next.iter().map(ext_word).collect());
        out.push(h.parts.iter().map(ext_word).collect());
        out.push(super::proof_arena::commitments_to_arena(&h.fri_roots));
        out.push(h.fri_coeffs.iter().map(ext_word).collect());
        if let Some(nc) = h.nonce {
            out.push(vec![base_word(FE::from(nc))]);
        }
        if with_legs {
            out.push(leg.opening_arena());
            out.push(leg.fri_arena());
        }
    }
    out
}

/// ★ THE RUN: the assembled verifier's Fiat-Shamir spine, executed against a
/// real continuation epoch proof that production accepts.
///
/// This is what the single-table differential could not reach. Both defects
/// `the_single_table_fixture_is_blind_to_two_defects` pins are live here — the
/// epoch has many tables, so the fork's domain separator matters, and the
/// production AIRs have multi-row OOD blocks, so the column-major absorb
/// matters.
#[test]
fn the_epoch_challenge_spine_matches_production() {
    let e = real_epoch();
    let program = epoch_challenge_program(&e);
    let arenas = epoch_arenas(&e);
    let exec = execute(&program, &arenas, &TestPermutation).expect("the epoch spine must execute");

    let pub_ext = |i: usize| word_as_ext(&exec.public_words[i].1).expect("an ext challenge");
    assert_eq!(pub_ext(0), e.z_alpha.0, "the shared LogUp challenge z");
    assert_eq!(pub_ext(1), e.z_alpha.1, "the shared LogUp challenge alpha");

    let mut cursor = 2usize;
    let mut multi_row_ood = 0;
    for (i, h) in e.tables.iter().enumerate() {
        assert_eq!(pub_ext(cursor), h.beta, "beta of table {i}");
        assert_eq!(pub_ext(cursor + 1), h.z, "z of table {i}");
        assert_eq!(pub_ext(cursor + 2), h.gamma, "gamma of table {i}");
        cursor += 3;
        for (k, want) in h.zetas.iter().enumerate() {
            assert_eq!(pub_ext(cursor + k), *want, "zeta {k} of table {i}");
        }
        cursor += h.zetas.len();
        for q in 0..h.shape.num_queries {
            let w = exec.public_words[cursor + q].1;
            let got = super::word::word_as_base(&w).expect("an index is a base felt");
            assert_eq!(got, FE::from(h.iotas[q] as u64), "iota {q} of table {i}");
        }
        cursor += h.shape.num_queries;
        if h.shape.ood_current_dims.1 > 1 || h.shape.ood_next_dims.1 > 1 {
            multi_row_ood += 1;
        }
        println!(
            "  table {i:2}: ood_current {:?}  ood_next {:?}  parts {}  fri_layers {}               log2_trace {}",
            h.shape.ood_current_dims,
            h.shape.ood_next_dims,
            h.shape.num_parts,
            h.shape.fri.num_committed(),
            h.shape.log2_trace_length
        );
    }
    // The closure's total, published last. Reaching it at all means the
    // in-machine `assert_eq_ext` against the COMMIT-bus target already held.
    assert_eq!(
        word_as_ext(&exec.public_words[cursor].1).expect("the bus total is ext"),
        e.expected_bus_balance,
        "the LogUp closure must reach production's own COMMIT-bus target"
    );
    cursor += 1;
    assert_eq!(
        cursor,
        exec.public_words.len(),
        "every published word must be checked"
    );

    // The blindness this fixture removes, asserted rather than hoped for.
    assert!(
        e.tables.len() > 1,
        "the fork's domain separator needs more than one table to matter"
    );
    // ★ MEASURED, not assumed: every one of the epoch's OOD blocks is ONE row
    // tall, so column-major and row-major absorbs coincide on all of them. The
    // current block's height IS `step_size` (`ood.rs:110-114`), and the phase
    // already knows `step_size = 1` collapses production; the next block's is
    // `num_eval_points − step_size`, which is 1 whenever an AIR has two
    // transition offsets. So the absorb ORDER has no production witness at all,
    // and closing it needs a synthetic AIR — see the ledger entry this leg added.
    assert_eq!(
        multi_row_ood, 0,
        "an OOD block taller than one row appeared: the absorb-order blindness \
         recorded here is over, and the differential now covers it"
    );
    // ---- THE MEASUREMENT ----
    //
    // What this is and is NOT: the spine is the Fiat-Shamir half of the
    // verifier — statement, Phase A, 24 forks, rounds 2-4 and the LogUp
    // closure. The opening/DEEP/FRI-walk and constraint legs are NOT in this
    // program, so these numbers say nothing about the composed per-epoch
    // predictions (213,744 opening permutations at blowup 8, ~460k total).
    // Those remain unconfirmed. This is the first per-epoch figure that is a
    // RUN rather than a composition, and it is the cost of the part that had
    // no per-epoch number at all.
    let perms = program
        .instrs
        .iter()
        .filter(|i| matches!(i, super::instr::Instr::KeccakF(_)))
        .count();
    let hints = program
        .instrs
        .iter()
        .filter(|i| matches!(i, super::instr::Instr::Hint { .. }))
        .count();
    let arena_words: usize = program.arena_schema.lens.iter().map(|l| *l as usize).sum();
    let bit_decs = program
        .instrs
        .iter()
        .filter(|i| matches!(i, super::instr::Instr::BitDec { .. }))
        .count();
    // Attribution, not a guess: every EXTENSION value the transcript absorbs is
    // three base felts, and each base felt is streamed BIG-endian, which costs
    // one `felt_be_halves` — a `BitDec` plus its recomposition. So the absorbed
    // ext count times three should account for nearly every `BitDec` here.
    let ext_absorbs: usize = e
        .tables
        .iter()
        .map(|h| {
            h.ood_current.len()
                + h.ood_next.len()
                + h.parts.len()
                + h.fri_coeffs.len()
                + usize::from(h.contribution.is_some())
        })
        .sum();
    println!(
        "\nepoch spine (min preset: blowup 2, {} quer{}/table, grinding {}):\n\
         \x20 sub-proofs        {}\n\
         \x20 instructions      {}\n\
         \x20 keccak perms      {}\n\
         \x20 arena words       {} ({} hinted)\n\
         \x20 published words   {}\n\
         \x20 multi-row OOD     {}\n\
         \x20 BitDec rows       {}\n\
         \x20 ext values absorbed {} (x3 felts = {} big-endian streams, \
         {:.1}% of the BitDecs)",
        e.tables[0].shape.num_queries,
        if e.tables[0].shape.num_queries == 1 {
            "y"
        } else {
            "ies"
        },
        e.tables[0].shape.grinding_factor,
        e.tables.len(),
        program.instrs.len(),
        perms,
        arena_words,
        hints,
        exec.public_words.len(),
        multi_row_ood,
        bit_decs,
        ext_absorbs,
        3 * ext_absorbs,
        100.0 * (3 * ext_absorbs) as f64 / bit_decs as f64
    );
}

/// ★ An ABSOLUTE structural guard (standing-decisions rule 7): no proof value
/// in the assembled spine is hinted twice.
///
/// The two-consumer class hides exactly where a differential cannot look — a
/// value hinted once per consumer, with the host packing the same number into
/// both, passes every comparison against production and still lets a real
/// prover supply two different numbers. So this is a count over the emitted
/// program, not a comparison of two runs: every arena word is read by at most
/// one `Hint`, and the arenas whose values have two consumers (the roots, the
/// contributions, the statement's public output) are read exactly once.
///
/// The register-boundary arena is the deliberate exception: only the carried
/// commit index is read today, and the rest is the space the REGISTER
/// derivation will consume.
#[test]
fn the_spine_hints_each_proof_value_once() {
    use std::collections::HashMap;

    let e = real_epoch();
    let program = epoch_challenge_program(&e);

    let mut hints: HashMap<(super::instr::ArenaId, u32), usize> = HashMap::new();
    for instr in &program.instrs {
        if let super::instr::Instr::Hint { arena, index, .. } = instr {
            *hints.entry((*arena, *index)).or_default() += 1;
        }
    }
    let doubled: Vec<_> = hints.iter().filter(|(_, n)| **n > 1).collect();
    assert!(
        doubled.is_empty(),
        "these arena words are hinted more than once, which is the two-consumer \
         hazard the assembly exists to remove: {doubled:?}"
    );

    // Positive control: the count is nonzero and covers the whole proof, so a
    // guard that simply found no hints would not pass for the wrong reason.
    let declared: usize = program.arena_schema.lens.iter().map(|l| *l as usize).sum();
    let reg_init = crate::tables::register::NUM_REGISTER_ADDRESSES;
    assert_eq!(
        hints.len(),
        declared - reg_init + 1,
        "every declared arena word must be read exactly once, bar the register \
         boundary vector of which only the commit index is read yet"
    );
}

/// ★ The closure's two joins, falsified by tampering.
///
/// The COMMIT-bus target is a function of the carried commit index and of the
/// public output, and both reach it through cells another consumer already
/// used — `start_index` from the register-boundary arena the REGISTER
/// derivation will bind, the output bytes from the halves the STATEMENT
/// absorbed. Moving either must break the run.
#[test]
fn the_closure_rejects_a_moved_index_or_output() {
    let e = real_epoch();
    let program = epoch_challenge_program(&e);
    let good = epoch_arenas(&e);
    assert!(
        execute(&program, &good, &TestPermutation).is_ok(),
        "the untampered epoch must run"
    );

    // The carried commit index. Production derives it from the previous epoch's
    // FINI vector; a machine that let the prover pick it would let them
    // renumber the whole output stream.
    for delta in [1u64, 2, 7] {
        let mut arenas = good.clone();
        arenas[3][crate::tables::register::X254_INDEX] = base_word(FE::from(e.start_index + delta));
        assert!(
            execute(&program, &arenas, &TestPermutation).is_err(),
            "start_index + {delta} must not close the bus"
        );
    }

    // The public output. Moving a half moves both the statement the challenges
    // were drawn against and the bytes the target folds, so this rejects
    // whichever check notices first — but reject it must.
    assert!(
        !e.public_output.is_empty(),
        "the fixture epoch must actually commit output, or this proves nothing"
    );
    for half in 0..e.statement.public_output_len.div_ceil(4) {
        let mut arenas = good.clone();
        let idx = 8 + half;
        let bumped = arenas[0][idx][0] + FE::one();
        arenas[0][idx] = base_word(bumped);
        assert!(
            execute(&program, &arenas, &TestPermutation).is_err(),
            "moving output half {half} must not verify"
        );
    }
}
