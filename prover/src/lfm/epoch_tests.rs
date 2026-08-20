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
//! composition (25 sub-proofs behind one statement) is not built here.

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
    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
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
        let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
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
/// ★ Where a preprocessed commitment COMES FROM — assembly ledger entry 7,
/// as a type.
///
/// Production absorbs every preprocessed root from the AIR and never from the
/// proof, so the machine owes each one a provenance of the same strength. The
/// three variants are the three that exist, and which one a commitment gets is
/// decided by what the commitment is a function of:
///
/// - options only ⇒ [`Self::Constant`], interned as program text. Safe because
///   the proof options are already program SHAPE.
/// - the previous epoch's register boundary ⇒ [`Self::Register`], DERIVED
///   in-machine. Interning it would pin one LFM program per register file;
///   hinting it would leave the boundary — the carried commit index among it — a
///   free arena word (ledger entry 2).
/// - the inner ELF ⇒ [`Self::ElfDependent`], an arena cell bound one level up by
///   the attestation's `program_id` fold. Interning it would make LFM program
///   identity a function of the guest ELF, which is an always-stop item.
///
/// [`prep_source`] decides the variant by MATCHING the AIR's own commitment
/// against production's candidate functions, so an epoch that grew a preprocessed
/// table with no known provenance panics instead of quietly hinting an unbound
/// root.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PrepSource {
    /// A function of `ProofOptions` alone: BITWISE, KECCAK_RC, or PAGE's
    /// shared zero-init root.
    Constant(Commitment),
    /// REGISTER: `compute_precomputed_commitment_with_fini(options, INIT, FINI)`.
    Register(Commitment),
    /// A function of the inner ELF: DECODE, or an ELF-data page's root.
    ElfDependent(Commitment),
}

impl PrepSource {
    /// Whether this root occupies arena words — true for exactly the
    /// ELF-dependent family.
    fn is_arena(self) -> bool {
        matches!(self, PrepSource::ElfDependent(_))
    }
}

/// How many of an epoch's preprocessed roots come from each source —
/// `(options-only, derived, ELF-dependent)`.
pub(super) fn prep_source_census(e: &RealEpoch) -> (usize, usize, usize) {
    let mut census = (0, 0, 0);
    for source in e.phase_a.iter().filter_map(|(p, _)| *p) {
        match source {
            PrepSource::Constant(_) => census.0 += 1,
            PrepSource::Register(_) => census.1 += 1,
            PrepSource::ElfDependent(_) => census.2 += 1,
        }
    }
    census
}

/// Classify one preprocessed commitment by recomputing every candidate
/// production has and seeing which one it IS.
///
/// A match is not a heuristic: these are keccak Merkle roots over different
/// tables, so two candidates agreeing would be a collision. What the function
/// really buys is the failure mode — a preprocessed AIR whose root matches
/// nothing known is a root the machine has no binding for, and this panics
/// rather than hinting it.
fn prep_source(
    root: Commitment,
    opts: &crate::ProofOptions,
    elf: &executor::elf::Elf,
    register_init: &[u32],
    reg_fini: &[u32],
) -> PrepSource {
    use crate::tables::{bitwise, decode, keccak_rc, page, register};

    if root == bitwise::preprocessed_commitment(opts)
        || root == keccak_rc::preprocessed_commitment(opts)
        || root == page::zero_init_preprocessed_commitment(opts)
    {
        return PrepSource::Constant(root);
    }
    if root == register::compute_precomputed_commitment_with_fini(opts, register_init, reg_fini) {
        return PrepSource::Register(root);
    }
    if root == decode::commitment_from_elf(elf, opts).expect("the DECODE commitment must compute") {
        return PrepSource::ElfDependent(root);
    }
    panic!(
        "a preprocessed sub-proof carries a root matching none of production's \
         candidate sources (BITWISE, KECCAK_RC, PAGE zero-init, REGISTER-with-FINI, \
         DECODE-from-ELF). The machine has no binding for it, so it must not be \
         hinted: extend the taxonomy (assembly ledger entry 7) rather than this list"
    );
}

pub(super) struct RealEpoch {
    pub(super) statement: super::statement_replay::EpochStatementShape,
    elf_digest: [u8; 32],
    pub(super) public_output: Vec<u8>,
    epoch_label: u64,
    /// Per table, in sub-proof order: the preprocessed commitment and WHERE IT
    /// COMES FROM (when the AIR is preprocessed), and the proof's main trace root.
    phase_a: Vec<(Option<PrepSource>, Commitment)>,
    /// The epoch's INIT register file — production's `register_init`. The whole
    /// vector, not just the carried commit index: the REGISTER preprocessed
    /// commitment is derived from it.
    register_init: Vec<u32>,
    /// The epoch's FINAL register file, the other half of that derivation.
    reg_fini: Vec<u32>,
    /// The inner ELF's entry point — `program_id`'s `pc_start`.
    pc_start: u64,
    /// The ELF-data page genesis roots the attestation folds. EMPTY for a
    /// continuation epoch's own verification: continuation epochs carry no PAGE
    /// sub-proof at all (`continuation.rs:695-702`), so these belong to the
    /// GLOBAL proof and reach the fold from outside.
    page_commitments: Vec<(u64, Commitment)>,
    /// The inner proof's LDE domain, for the REGISTER derivation. Both fields are
    /// proof OPTIONS, hence program shape.
    reg_shape: super::programs::RegisterDerivationShape,
    /// `recursion::program_id_from_digest` over this epoch's own inputs — the
    /// oracle for the attestation fold.
    pub(super) expected_program_id: [u8; 32],
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
    real_epoch_with(super::proof_fixture::fixture_options())
}

/// What an epoch is built FROM: the inner guest, its private input, and the
/// epoch size. Separated from the proof options because the two axes are
/// independent — options change what verifying the epoch costs, these change
/// what the epoch IS.
///
/// [`EpochInputs::fixture`] is the 16-cycle fibonacci fixture every existing
/// test builds; [`EpochInputs::from_env`] is that with the three overrides a
/// measurement run needs, and it is what [`real_epoch_with`] uses, so with
/// nothing set every caller keeps the exact path it had.
pub(super) struct EpochInputs {
    pub(super) elf_bytes: Vec<u8>,
    pub(super) private_input: Vec<u8>,
    pub(super) epoch_log2: u32,
    /// Names the guest in printed measurements, since a real-block run and the
    /// fixture otherwise report identically shaped numbers.
    pub(super) label: String,
}

impl EpochInputs {
    /// The fibonacci fixture: the ELF the recursion suite builds, no private
    /// input, [`FIXTURE_EPOCH_LOG2`](super::proof_fixture::FIXTURE_EPOCH_LOG2).
    pub(super) fn fixture() -> Self {
        Self {
            elf_bytes: super::proof_fixture::read_inner_elf(),
            private_input: Vec::new(),
            epoch_log2: super::proof_fixture::FIXTURE_EPOCH_LOG2,
            label: "fibonacci fixture".to_string(),
        }
    }

    /// [`EpochInputs::fixture`] with the measurement overrides applied:
    ///
    /// - `LFM_CENSUS_ELF` — path to the inner guest ELF.
    /// - `LFM_CENSUS_INPUT` — path to a file holding its private input.
    /// - `LFM_CENSUS_EPOCH_LOG2` — epoch size, log2.
    ///
    /// The input override exists because a guest's epoch count is a property of
    /// its INPUT, not just its ELF: the fibonacci guest reads its iteration
    /// count from private input, so a run that needs a multi-epoch execution has
    /// to be able to ask for one without a recompile. The ELF and epoch-size
    /// overrides are what let the same harness build a real Ethereum-block
    /// epoch, which is far too large to be a checked-in fixture.
    ///
    /// With none set this is byte-for-byte [`EpochInputs::fixture`].
    pub(super) fn from_env() -> Self {
        let mut inputs = Self::fixture();
        if let Ok(p) = std::env::var("LFM_CENSUS_ELF") {
            inputs.elf_bytes =
                std::fs::read(&p).unwrap_or_else(|e| panic!("LFM_CENSUS_ELF {p}: {e}"));
            inputs.label = p;
        }
        if let Ok(p) = std::env::var("LFM_CENSUS_INPUT") {
            inputs.private_input =
                std::fs::read(&p).unwrap_or_else(|e| panic!("LFM_CENSUS_INPUT {p}: {e}"));
        }
        if let Ok(v) = std::env::var("LFM_CENSUS_EPOCH_LOG2") {
            inputs.epoch_log2 = v
                .parse()
                .unwrap_or_else(|e| panic!("LFM_CENSUS_EPOCH_LOG2 {v}: {e}"));
        }
        inputs
    }
}

/// [`real_epoch`] under supplied proof options — the wrap run's blowup axis.
///
/// The options are the INNER proof's, so they change what the verifier has to do:
/// the query count, the LDE depth every Merkle walk climbs, and how many FRI
/// layers commit. What the epoch IS comes from [`EpochInputs::from_env`], which
/// is the fibonacci fixture unless a measurement run overrode it — so two runs
/// at different options stay comparable, and assembly ledger entry 10 still
/// holds: the trace-length profile travels with every number.
/// ★ THE BASE-LAYER A/B — the real block's epoch 0 proved per-table vs
/// BATCHED-MMCS, one arm per process.
///
/// `AB_MODE` selects the arm (`per_table` | `batched`); `LAMBDA_VM_RESIDENCY`
/// moves BOTH arms through the same lever, so a residency difference between
/// them cannot be an artifact of two code paths reading two knobs. Peak anon
/// is a process-lifetime high-water mark, measured by the harness around the
/// process — two arms sharing a process would each report the larger of the
/// two and the comparison would be vacuous.
///
/// The epoch construction is byte-for-byte the census harness's
/// ([`real_epoch_from`]): same executor slice, same traces, same L2G bookend,
/// same statement-seeded transcript. Only the prove call differs.
///
/// ⚠ NEITHER arm verifies here, deliberately. The per-table construction is
/// production-accepted every time `the_real_block_epoch_wraps` runs (its own
/// gate), so it needs no second acceptance; the batched arm CANNOT verify —
/// no pinned preprocessed round exists for the VM AIR set (the per-table
/// verifier pins each table's root individually; the batched verifier needs
/// ONE root over the whole round, which nothing blesses yet — the VM-side
/// analogue of the M-8 round-coverage gap), and `multi_verify_batched` fails
/// closed on that by design. This instrument measures the PROVE.
#[test]
#[ignore]
fn the_real_block_base_epoch_ab() {
    use crate::tables::trace_builder::{Traces, build_initial_image_paged};
    use crate::tables::{MaxRowsConfig, bitwise, local_to_global, register};
    use executor::elf::Elf;
    use executor::vm::execution::Executor;
    use stark::prover::IsStarkProver;

    for var in ["LFM_CENSUS_ELF", "LFM_CENSUS_INPUT"] {
        assert!(
            std::env::var(var).is_ok(),
            "{var} must name a file: this A/B measures a REAL block epoch"
        );
    }
    let mode = std::env::var("AB_MODE").expect("AB_MODE must be per_table or batched");
    let residency = match std::env::var("LAMBDA_VM_RESIDENCY").as_deref() {
        Ok("recompute") => stark::residency_mode::ResidencyMode::RecomputeLde,
        _ => stark::residency_mode::ResidencyMode::Retain,
    };

    let EpochInputs {
        elf_bytes,
        private_input,
        epoch_log2,
        label: guest_label,
    } = EpochInputs::from_env();
    let opts = crate::recursion::Preset::Blowup4.options();
    let mut inner = opts;
    if let Ok(v) = std::env::var("LFM_WRAP_QUERIES") {
        inner.fri_number_of_queries = v.parse().expect("LFM_WRAP_QUERIES must be an integer");
    }
    let opts = inner;
    println!(
        "★ BASE A/B ARM: mode={mode} residency={residency:?}  guest {guest_label}, \
         2^{epoch_log2} cycles/epoch, blowup {} / {} queries",
        opts.blowup_factor, opts.fri_number_of_queries,
    );

    let elf = Elf::load(&elf_bytes).expect("the inner ELF must load");
    let epoch_size = 1usize << epoch_log2;
    let mut executor = Executor::new(&elf, private_input.clone()).expect("executor");
    let image = build_initial_image_paged(&elf, &private_input);
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
        &private_input,
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
        let mut t = stark::config::DefaultStarkTranscript::<Ext3>::new(&[]);
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

    let mut pairs = airs.air_trace_pairs(&mut traces);
    pairs.push((&l2g_air, &mut l2g_trace, &()));

    // ---- THE MEASURED PROVE. Everything above is identical shared setup.
    let t = std::time::Instant::now();
    match mode.as_str() {
        "per_table" => {
            let proof = stark::prover::Prover::<Gl, Ext3, ()>::multi_prove(
                pairs,
                &mut seed(),
                #[cfg(feature = "disk-spill")]
                stark::storage_mode::StorageMode::Ram,
                residency,
            )
            .expect("the epoch must prove");
            let prove_secs = t.elapsed().as_secs_f64();
            let size = rkyv::to_bytes::<rkyv::rancor::Error>(&proof)
                .expect("the epoch proof must serialize")
                .len();
            println!(
                "★ BASE A/B RESULT mode=per_table PROVE_SECS={prove_secs:.2} \
                 SUB_PROOFS={} PROOF_BYTES={size}",
                proof.proofs.len(),
            );
        }
        "batched" => {
            let (proof, stats) = stark::batched::prover::multi_prove_batched::<
                Gl,
                Ext3,
                (),
                stark::config::DefaultStarkHash,
                stark::prover::Prover<Gl, Ext3, ()>,
            >(
                pairs,
                &mut seed(),
                #[cfg(feature = "disk-spill")]
                stark::storage_mode::StorageMode::Ram,
                residency,
            )
            .expect("the batched epoch must prove");
            let prove_secs = t.elapsed().as_secs_f64();
            println!(
                "★ BASE A/B RESULT mode=batched PROVE_SECS={prove_secs:.2} \
                 TABLES={} QUERIES={} FRI_LAYERS={} PREP_TABLES={}",
                proof.tables.len(),
                proof.queries.len(),
                proof.fri_layer_roots.len(),
                proof.queries.first().map_or(0, |q| q.prep.len()),
            );
            println!("   BATCHED_STATS {stats:?}");
        }
        other => panic!("AB_MODE must be per_table or batched, not {other}"),
    }
}

pub(super) fn real_epoch_with(opts: crate::ProofOptions) -> RealEpoch {
    real_epoch_from(opts, EpochInputs::from_env())
}

/// [`real_epoch_with`] with the guest, its input and the epoch size supplied
/// rather than read from the environment.
///
/// Only epoch 0 is built: the boundary starts from genesis provenance and the
/// label is `epoch_label(0)`, so a later epoch would need the previous one's
/// provenance carried in. That is a real limit of this harness and not an
/// oversight — the first epoch is what the compression work needs.
pub(super) fn real_epoch_from(opts: crate::ProofOptions, inputs: EpochInputs) -> RealEpoch {
    use crate::tables::trace_builder::{Traces, build_initial_image_paged};
    use crate::tables::{MaxRowsConfig, bitwise, local_to_global, register};
    use crypto::fiat_shamir::is_transcript::IsTranscript;
    use executor::elf::Elf;
    use executor::vm::execution::Executor;
    use stark::proof::view::MultiProofView;
    use stark::verifier::IsStarkVerifier;

    let EpochInputs {
        elf_bytes,
        private_input,
        epoch_log2,
        label: guest_label,
    } = inputs;
    let elf = Elf::load(&elf_bytes).expect("the inner ELF must load");
    let epoch_size = 1usize << epoch_log2;

    let mut executor = Executor::new(&elf, private_input.clone()).expect("executor");
    let image = build_initial_image_paged(&elf, &private_input);
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
        &private_input,
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
    // The attestation fold's DECODE input, from PRODUCTION's own function — the
    // same value `VmAirs::new` puts on the DECODE AIR, and the same one
    // `recursion::check_attestation` recomputes from a trusted ELF.
    let decode_root = crate::tables::decode::commitment_from_elf(&elf, &opts)
        .expect("the DECODE commitment must compute");
    let l2g_air = crate::continuation::l2g_memory_air(&opts, label);
    let mut l2g_trace = local_to_global::generate_local_to_global_trace(&boundary);

    let seed = || {
        let mut t = stark::config::DefaultStarkTranscript::<Ext3>::new(&[]);
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
        let t = std::time::Instant::now();
        let proof =
            crate::test_utils::multi_prove_ram(pairs, &mut seed()).expect("the epoch must prove");
        // The inner prove is the expensive half of a real-block run and is
        // otherwise invisible inside the wrap's own timings, so it reports
        // itself — with the guest and epoch size, since a number without them
        // does not identify a workload.
        eprintln!(
            "inner epoch: {guest_label}, 2^{epoch_log2} cycles, {} cycles executed, \
             {} sub-proofs, proved in {:.1}s",
            logs.len(),
            proof.proofs.len(),
            t.elapsed().as_secs_f64()
        );
        proof
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
            phase_a.push((
                Some(prep_source(prep, &opts, &elf, &register_init, &reg_fini)),
                *v.lde_trace_main_merkle_root(),
            ));
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
                table_counts.blake3 as u64,
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
        register_init,
        reg_fini,
        pc_start: elf.entry_point,
        // ★ EMPTY, and it is a claim about PRODUCTION rather than about this
        // fixture: `prove_epoch` REJECTS an epoch with any PAGE config
        // ("continuation epoch must have no PAGE configs (L2G bookend replaces
        // PAGE)", `continuation.rs:695-702`) and both `build_epoch_airs` call
        // sites pass `&[]`. The ELF-data page genesis roots the attestation folds
        // are the GLOBAL proof's GlobalMemory AIRs' preprocessed commitments
        // (`continuation.rs:997-1010`), never an epoch's.
        page_commitments: Vec::new(),
        reg_shape: super::programs::RegisterDerivationShape {
            blowup: opts.blowup_factor as usize,
            coset_offset: opts.coset_offset,
        },
        expected_program_id: crate::recursion::program_id_from_digest(
            &crate::statement::elf_digest(&elf_bytes),
            elf.entry_point,
            &decode_root,
            &[],
        ),
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
    fork: &mut stark::config::DefaultStarkTranscript<Ext3>,
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
    epoch_program_with(e, with_legs, false)
}

/// The epoch program, optionally with the DECODE cell SPLIT — a deliberately
/// broken control, and the falsification the entry-7 ruling asked for.
///
/// `split_decode = true` gives the attestation fold its own arena copy of the
/// DECODE root instead of the cell Phase A absorbed. Nothing about the program
/// then looks wrong: every assert still passes, the challenges are still
/// production's, and an honest host that fills both copies with the same 32 bytes
/// gets the same published `program_id`. That is exactly why the join has to be
/// denied STRUCTURALLY rather than by a differential —
/// [`a_split_decode_cell_forges_the_attestation`] runs the coherent forgery this
/// admits, and
/// [`the_assembled_verifier_declares_exactly_the_shape_words`] is what refuses it.
///
/// The extra arena is declared LAST so no existing arena index moves.
fn epoch_program_with(e: &RealEpoch, with_legs: bool, split_decode: bool) -> LfmProgram {
    use super::statement_replay::{EpochStatementVars, PhaseATable, absorb_epoch_statement};

    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let n = e.tables.len();
    assert_eq!(e.legs.len(), n, "one leg reading per sub-proof");

    // ---- arenas, in declaration order ----
    let stmt_halves = 8 + e.statement.public_output_len.div_ceil(4) + 2;
    let a_stmt = b.declare_arena(stmt_halves as u32);
    // ★ Only the ELF-DEPENDENT preprocessed roots are arena data (ledger entry
    // 7). The options-only ones are interned as program text and the REGISTER one
    // is derived in-machine, so neither takes a word here.
    let num_arena_prep = e
        .phase_a
        .iter()
        .filter(|(p, _)| p.is_some_and(PrepSource::is_arena))
        .count();
    let a_prep_roots = b.declare_arena(2 * num_arena_prep as u32);
    let a_main_roots = b.declare_arena(2 * n as u32);
    // The register boundary vectors, at production's width. `start_index` is slot
    // 64 of INIT, and the REGISTER preprocessed root is COMPUTED from both — which
    // is what ties the index to the chain (ledger entry 2): production has no
    // arithmetic `start + len` check anywhere, it rebuilds the commitment from
    // these vectors and rejects unless the absorbed root matches.
    let num_reg = crate::tables::register::NUM_REGISTER_ADDRESSES as u32;
    let a_reg_init = b.declare_arena(num_reg);
    let a_reg_fini = b.declare_arena(num_reg);
    // The attestation fold's own inputs. `elf_digest` is NOT here — it is the
    // statement's, which is the join. `pc_start` has one consumer in an epoch
    // verifier, and the page roots have none at all (a continuation epoch carries
    // no PAGE sub-proof), so both are plain proof data the fold hashes.
    let a_pc_start = b.declare_arena(2);
    let a_page_roots = (!e.page_commitments.is_empty())
        .then(|| b.declare_arena(10 * e.page_commitments.len() as u32));
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
            legs: with_legs.then(|| super::epoch_verify::declare_table_arenas(&mut b, &leg.verify)),
        })
        .collect();
    // Last in declaration order, so turning the control on shifts no other arena.
    let a_split_decode = split_decode.then(|| b.declare_arena(2));

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

    // ---- ★ the preprocessed roots, each from the source its provenance admits
    //
    // Ledger entry 7, and entry 2 closes with it. `PrepSource` was decided
    // host-side by recomputing production's candidate functions, so the split here
    // is not a hardcoded sub-proof index: a preprocessed table with an unknown
    // provenance would already have panicked.
    let reg_init: Vec<_> = (0..num_reg).map(|r| b.hint_felt(a_reg_init, r)).collect();
    let reg_fini: Vec<_> = (0..num_reg).map(|r| b.hint_felt(a_reg_fini, r)).collect();
    // ★ LEDGER ENTRY 1. Production's boundary vectors are `Vec<u32>` and the TYPE
    // is the whole enforcement; an arena is untyped felts, so without this the
    // machine would derive a commitment over a value no production epoch can hold.
    // The entry's stated default was "emit the check if the no->u32 argument is
    // still unverified when assembly arrives" — it is, and assembly has arrived.
    for cell in reg_init.iter().chain(&reg_fini) {
        super::epoch::assert_u32(&mut b, *cell);
    }
    let reg_shape = e.reg_shape;

    let mut next_arena_prep = 0usize;
    let mut decode_cells: Option<RootCells> = None;
    let prep_cells: Vec<Option<RootCells>> = e
        .phase_a
        .iter()
        .map(|(prep, _)| match prep {
            None => None,
            Some(PrepSource::Constant(c)) => Some(RootCells::constant(&mut b, c)),
            Some(PrepSource::Register(_)) => {
                let digest = super::programs::emit_register_commitment(
                    &mut b, reg_shape, &reg_init, &reg_fini,
                );
                Some(RootCells::from_digest(&mut b, digest))
            }
            Some(PrepSource::ElfDependent(_)) => {
                let cells = RootCells::hint(&mut b, a_prep_roots, 2 * next_arena_prep as u32);
                next_arena_prep += 1;
                // Every ELF-dependent root of a continuation EPOCH is DECODE (the
                // page family lives in the global proof), and the attestation
                // folds exactly one DECODE root — so a second one here would mean
                // the fold's input is ambiguous, not that the fold needs a loop.
                assert!(
                    decode_cells.is_none(),
                    "a continuation epoch has one ELF-dependent preprocessed root \
                     (DECODE); a second one has no place in the program_id fold"
                );
                decode_cells = Some(cells.clone());
                Some(cells)
            }
        })
        .collect();
    assert_eq!(
        next_arena_prep, num_arena_prep,
        "every declared preprocessed arena word must be read"
    );

    // ---- Phase A ----
    let main_cells: Vec<RootCells> = (0..n)
        .map(|i| RootCells::hint(&mut b, a_main_roots, 2 * i as u32))
        .collect();
    let prep_halves: Vec<Option<Vec<_>>> = prep_cells
        .iter()
        .map(|c| c.as_ref().map(RootCells::halves))
        .collect();
    let main_halves: Vec<Vec<_>> = main_cells.iter().map(RootCells::halves).collect();
    // The interned bytes, hoisted so Phase A can borrow them for the whole replay.
    let prep_constants: Vec<Option<Commitment>> = e
        .phase_a
        .iter()
        .map(|(p, _)| match p {
            Some(PrepSource::Constant(c)) => Some(*c),
            _ => None,
        })
        .collect();
    let tables: Vec<PhaseATable> = (0..n)
        .map(|i| PhaseATable {
            // A program-text root absorbs as literal BYTES — no splice arithmetic
            // at all, which is the whole economy of interning it. A derived or
            // supplied one absorbs as the cells its consumers share.
            preprocessed_root: match (&prep_constants[i], &prep_halves[i]) {
                (Some(bytes), _) => {
                    Some(super::statement_replay::PhaseAPreprocessed::Constant(bytes))
                }
                (None, Some(halves)) => Some(super::statement_replay::PhaseAPreprocessed::Cells(
                    &halves[..],
                )),
                (None, None) => None,
            },
            main_root: &main_halves[i][..],
        })
        .collect();
    let (z, alpha) = super::statement_replay::replay_phase_a(&mut t, &mut b, &tables);
    b.public(z.as_cell());
    b.public(alpha.as_cell());

    // ---- ★ the attestation join: the DECODE cell Phase A absorbed, folded
    //
    // One cell, two consumers. Without this the DECODE root would be a free arena
    // word — the machine would absorb whatever the prover offered and publish
    // nothing that depended on it.
    {
        let pc_start: Vec<_> = (0..2).map(|i| b.hint_felt(a_pc_start, i)).collect();
        let page_cells: Vec<(Vec<_>, RootCells)> = e
            .page_commitments
            .iter()
            .enumerate()
            .map(|(k, _)| {
                let base = 10 * k as u32;
                let arena = a_page_roots.expect("a page arena exists when pages do");
                let base_halves: Vec<_> = (0..2).map(|j| b.hint_felt(arena, base + j)).collect();
                let root_halves: Vec<_> =
                    (0..8).map(|j| b.hint_felt(arena, base + 2 + j)).collect();
                (
                    base_halves,
                    RootCells {
                        lanes: [
                            [
                                root_halves[0],
                                root_halves[1],
                                root_halves[2],
                                root_halves[3],
                            ],
                            [
                                root_halves[4],
                                root_halves[5],
                                root_halves[6],
                                root_halves[7],
                            ],
                        ],
                    },
                )
            })
            .collect();
        let page_halves: Vec<(Vec<_>, Vec<_>)> = page_cells
            .iter()
            .map(|(base, root)| (base.clone(), root.halves()))
            .collect();
        let page_refs: Vec<(&[_], &[_])> = page_halves
            .iter()
            .map(|(base, root)| (&base[..], &root[..]))
            .collect();
        let decode = match a_split_decode {
            // ★ THE BROKEN CONTROL: a second, independent reading of the DECODE
            // root. The fold now attests to a value Phase A never absorbed.
            Some(arena) => RootCells::hint(&mut b, arena, 0).halves(),
            None => decode_cells
                .as_ref()
                .expect("a continuation epoch has a DECODE sub-proof")
                .halves(),
        };
        let id = super::programs::emit_program_id(
            &mut b,
            super::programs::ProgramIdShape {
                num_pages: e.page_commitments.len(),
            },
            elf_digest,
            &pc_start,
            &decode,
            &page_refs,
        );
        b.public(id[0]);
        b.public(id[1]);
    }

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
                    precomputed_root: prep_cells[i].as_ref(),
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
    // ★ LEDGER ENTRY 2 CLOSES HERE. The carried commit index is not a word of its
    // own and not even a second READ of one: it is the very cell the REGISTER
    // preprocessed derivation consumed as INIT slot 64, so the COMMIT-bus target
    // and the root Phase A absorbed are functions of one value. Production binds
    // `start_index` exactly this way — it has no arithmetic `start + len` check
    // anywhere, it rebuilds the commitment from the boundary vectors and rejects
    // unless the absorbed root matches.
    let start = reg_init[crate::tables::register::X254_INDEX];
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

/// How many EPOCH-WIDE arenas [`epoch_program`] declares before the first
/// table's — statement, ELF-dependent preprocessed roots, main roots, the two
/// register boundary vectors, `pc_start`, and the page roots when there are any.
///
/// Exposed rather than hardcoded because a test that walks to a per-table arena
/// by index silently tampers the WRONG arena when this changes, and reports a
/// pass: wiring ledger entry 7 moved it from 4 to 6.
pub(super) fn num_epoch_wide_arenas(e: &RealEpoch) -> usize {
    6 + usize::from(!e.page_commitments.is_empty())
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

    // Only the ELF-DEPENDENT roots take arena words; the rest are program text or
    // derived in-machine.
    let prep: Vec<Commitment> = e
        .phase_a
        .iter()
        .filter_map(|(p, _)| match p {
            Some(PrepSource::ElfDependent(c)) => Some(*c),
            _ => None,
        })
        .collect();
    let main: Vec<Commitment> = e.phase_a.iter().map(|(_, m)| *m).collect();

    // The register boundary, at production's width. The carried commit index sits
    // in slot 64 of INIT, and the REGISTER preprocessed root is derived from both
    // vectors — so this arena is not padding around one word any more.
    let reg = |v: &[u32]| -> Vec<LfmWord> {
        assert_eq!(
            v.len(),
            crate::tables::register::NUM_REGISTER_ADDRESSES,
            "a register boundary vector is one word per register word address"
        );
        v.iter()
            .map(|w| base_word(FE::from(u64::from(*w))))
            .collect()
    };
    assert_eq!(
        e.register_init[crate::tables::register::X254_INDEX] as u64,
        e.start_index,
        "the carried commit index must BE slot 64 of the INIT vector, or the \
         COMMIT-bus target and the REGISTER derivation are reading two values"
    );
    let mut out = vec![
        stmt.iter().map(|h| base_word(*h)).collect(),
        super::proof_arena::commitments_to_arena(&prep),
        super::proof_arena::commitments_to_arena(&main),
        reg(&e.register_init),
        reg(&e.reg_fini),
        super::keccak_host::pack_stream(&e.pc_start.to_le_bytes())
            .into_iter()
            .map(base_word)
            .collect(),
    ];
    if !e.page_commitments.is_empty() {
        let mut pages: Vec<LfmWord> = Vec::new();
        for (base, c) in &e.page_commitments {
            pages.extend(
                super::keccak_host::pack_stream(&base.to_le_bytes())
                    .into_iter()
                    .map(base_word),
            );
            pages.extend(
                super::keccak_host::pack_stream(c)
                    .into_iter()
                    .map(base_word),
            );
        }
        out.push(pages);
    }
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

/// A keccak digest published as two words starting at `at` — eight `u32` halves,
/// four per word, each four bytes little-endian.
fn published_digest(public: &[(u32, LfmWord)], at: usize) -> [u8; 32] {
    use math::field::traits::IsPrimeField;
    let mut out = [0u8; 32];
    for h in 0..8 {
        let lane = public[at + h / 4].1[h % 4];
        let half = GoldilocksField::canonical(lane.value()) as u32;
        out[4 * h..4 * h + 4].copy_from_slice(&half.to_le_bytes());
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

    // ★ The attestation fold, published right after Phase A. Its DECODE input is
    // the very cell Phase A absorbed, so this differential is simultaneously a
    // check of the fold and of the join: had the fold read a second copy, this
    // would still pass — which is why the split is denied STRUCTURALLY by
    // `the_assembled_verifier_declares_exactly_the_shape_words` and demonstrated
    // by `a_split_decode_cell_forges_the_attestation`.
    assert_eq!(
        published_digest(&exec.public_words, 2),
        e.expected_program_id,
        "the attestation program_id must equal production's \
         `program_id_from_digest` over the same inputs"
    );

    let mut cursor = 4usize;
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
/// The exception this test used to carry is GONE: the register-boundary arena
/// had only its commit index read while the REGISTER derivation was unbuilt, and
/// wiring the derivation (ledger entries 7 and 2) makes every declared word live.
/// So the positive control is now exact — `declared` words, `declared` reads —
/// which is a strictly stronger statement than the one it replaces.
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
    assert_eq!(
        hints.len(),
        declared,
        "every declared arena word must be read exactly once"
    );
}

/// Arena words the epoch program MUST declare, as arithmetic over the epoch's
/// shapes.
///
/// Deliberately not derived from the emitter (standing-decisions rule 7's
/// refinement: a count taken from our own emitter is still a relative test). Every
/// term here comes from the production proof view `real_epoch` read, so the
/// comparison against the compiled program is absolute.
fn expected_arena_words(e: &RealEpoch, with_legs: bool) -> usize {
    let num_reg = crate::tables::register::NUM_REGISTER_ADDRESSES;
    let mut total = 8 + e.statement.public_output_len.div_ceil(4) + 2;
    // ★ Two words per ELF-DEPENDENT preprocessed root and NOT ONE MORE. The
    // options-only roots are program text and the REGISTER root is derived, so a
    // program that hinted any of them — or that kept a second copy of DECODE for
    // the attestation fold — declares more words than this.
    total += 2 * e
        .phase_a
        .iter()
        .filter(|(p, _)| p.is_some_and(PrepSource::is_arena))
        .count();
    total += 2 * e.tables.len();
    total += 2 * num_reg;
    total += 2;
    total += 10 * e.page_commitments.len();
    for (h, leg) in e.tables.iter().zip(&e.legs) {
        let s = &h.shape;
        total += 2 * usize::from(s.has_aux_root);
        total += usize::from(s.has_contribution);
        total += 2;
        total += s.ood_current_dims.0 * s.ood_current_dims.1;
        total += s.ood_next_dims.0 * s.ood_next_dims.1;
        total += s.num_parts;
        total += 2 * s.fri.num_committed();
        total += s.fri.num_terminal_coeffs();
        total += usize::from(s.grinding_factor > 0);
        if with_legs {
            total += leg.verify.opening_words() + leg.verify.fri_words();
        }
    }
    total
}

/// ★ An ABSOLUTE guard on the arena SCHEMA — the structural half of the
/// attestation join (entry-7 ruling, condition (a)).
///
/// The hinted-once guard denies a value being read twice from ONE word. It cannot
/// deny a value being supplied twice in TWO words, which is the whole two-consumer
/// hazard: an honest host fills both copies alike, every differential passes, and a
/// real prover supplies two different roots. What denies that is the schema itself
/// — the program declares exactly the words the epoch's shapes prescribe, so there
/// is nowhere for a second copy to live.
///
/// Together the two guards are complete for this class: a second copy must either
/// re-read an existing word (hinted-once fails) or add one (this fails). A fold
/// that instead read some OTHER existing value would publish a `program_id` that is
/// not production's, which the spine differential catches.
#[test]
fn the_assembled_verifier_declares_exactly_the_shape_words() {
    let e = real_epoch();
    for with_legs in [false, true] {
        let program = epoch_program(&e, with_legs);
        let declared: usize = program.arena_schema.lens.iter().map(|l| *l as usize).sum();
        assert_eq!(
            declared,
            expected_arena_words(&e, with_legs),
            "with_legs = {with_legs}: the arena schema must be exactly the epoch's \
             shapes and nothing more — a surplus word is where a second copy of a \
             joined value hides"
        );
    }

    // Positive control on the guard itself: the split-cell control program DOES
    // declare a surplus word, and this is the comparison that sees it.
    let split = epoch_program_with(&e, false, true);
    let split_declared: usize = split.arena_schema.lens.iter().map(|l| *l as usize).sum();
    assert_eq!(
        split_declared,
        expected_arena_words(&e, false) + 2,
        "the split-cell control must declare exactly two surplus words, or it is \
         not the forgery this guard claims to deny"
    );
}

/// ★ FALSIFICATION of the attestation join, as a COHERENT FORGERY rather than a
/// count (standing-decisions method rule 4).
///
/// The attack: verify a real epoch proof of ELF X while attesting to the
/// `program_id` of a different ELF Y. A consumer who trusts Y's id accepts the
/// proof, and X is whatever the prover likes.
///
/// On the SPLIT program this succeeds completely — every assert passes, the run
/// finishes, and the published id is the one computed from the substituted root,
/// not from the root the proof was made against. On the JOINED program the attack
/// is not merely rejected, it cannot be EXPRESSED: there is one cell, so changing
/// the fold's input changes what Phase A absorbed, which moves every challenge and
/// the run dies. Both halves are asserted, because "the joined program rejects it"
/// alone would be satisfied by a program that rejects everything.
#[test]
fn a_split_decode_cell_forges_the_attestation() {
    let e = real_epoch();
    let honest = epoch_arena_words(&e, false);

    // A DECODE root for some other program. Any 32 bytes the honest arena does not
    // carry will do; what matters is the id it produces.
    let real_decode = e
        .phase_a
        .iter()
        .find_map(|(p, _)| match p {
            Some(PrepSource::ElfDependent(c)) => Some(*c),
            _ => None,
        })
        .expect("the epoch has a DECODE sub-proof");
    let mut substituted = real_decode;
    substituted[0] ^= 0xa5;
    substituted[31] ^= 0x5a;
    let forged_id =
        crate::recursion::program_id_from_digest(&e.elf_digest, e.pc_start, &substituted, &[]);
    assert_ne!(
        forged_id, e.expected_program_id,
        "the substituted root must produce a different id, or this proves nothing"
    );

    // ---- (a) the SPLIT program: the forgery runs and publishes the forged id.
    let split = epoch_program_with(&e, false, true);
    let mut split_arenas = honest.clone();
    split_arenas.push(super::proof_arena::commitments_to_arena(&[substituted]));
    let exec = execute(&split, &split_arenas, &TestPermutation).expect(
        "the split-cell program must RUN on the forgery — that is the hazard, and \
         a rejection here would mean this control does not demonstrate it",
    );
    assert_eq!(
        published_digest(&exec.public_words, 2),
        forged_id,
        "the split program must attest to the SUBSTITUTED root while verifying a \
         proof made against the real one"
    );
    // And it is genuinely a proof of the real epoch: the same program, given the
    // honest root in the surplus arena, publishes the honest id.
    let mut split_honest = honest.clone();
    split_honest.push(super::proof_arena::commitments_to_arena(&[real_decode]));
    let exec_honest = execute(&split, &split_honest, &TestPermutation)
        .expect("the split program must also run honestly");
    assert_eq!(
        published_digest(&exec_honest.public_words, 2),
        e.expected_program_id,
        "the split program's two runs differ only in the surplus arena, so the \
         forgery is a free choice and not a broken proof"
    );

    // ---- (b) the JOINED program: the same substitution is inexpressible.
    //
    // There is no surplus arena to put it in, so the only way to move the fold's
    // input is to move the cell Phase A absorbed — which moves every challenge
    // derived after it.
    let joined = epoch_program(&e, false);
    let mut joined_arenas = honest.clone();
    joined_arenas[1] = super::proof_arena::commitments_to_arena(&[substituted]);
    assert!(
        execute(&joined, &joined_arenas, &TestPermutation).is_err(),
        "with one cell, substituting the DECODE root must break the run: the \
         transcript absorbed it, so the challenges cannot survive it"
    );
}

/// ★ LEDGER ENTRY 2, closed and falsified: the whole register boundary is bound,
/// not just the commit index.
///
/// Production ties epoch N's carried commit index to the chain by REBUILDING the
/// REGISTER preprocessed commitment from epoch N−1's FINI vector and rejecting
/// unless the absorbed root matches — there is no arithmetic `start + len` check
/// anywhere (`lfm-team-lead-start-index-research.md`). So the machine's binding is
/// the derivation, and what must be true is that moving ANY word of either vector
/// makes the epoch unverifiable.
///
/// Before the derivation was wired, 66 of the 67 INIT words were declared and never
/// read: moving them changed nothing at all. The positive control for that is
/// structural rather than historical — `the_spine_hints_each_proof_value_once` now
/// requires every declared word to be read, and it did not before.
///
/// Slot 64 is the commit index and is tested separately by
/// [`the_closure_rejects_a_moved_index_or_output`]; the slots here are deliberately
/// elsewhere, including the first and last of each vector, because a derivation that
/// only really consumed a prefix would pass a test that only moved slot 64.
#[test]
fn the_derivation_binds_every_register_boundary_word() {
    let e = real_epoch();
    let program = epoch_challenge_program(&e);
    let good = epoch_arenas(&e);
    assert!(
        execute(&program, &good, &TestPermutation).is_ok(),
        "the untampered epoch must run"
    );

    let last = crate::tables::register::NUM_REGISTER_ADDRESSES - 1;
    let x254 = crate::tables::register::X254_INDEX;
    let mut moved = 0;
    for (arena, what) in [(3usize, "INIT"), (4, "FINI")] {
        for slot in [0usize, 1, 33, x254 + 1, last] {
            let mut arenas = good.clone();
            let bumped = arenas[arena][slot][0] + FE::one();
            arenas[arena][slot] = base_word(bumped);
            assert!(
                execute(&program, &arenas, &TestPermutation).is_err(),
                "{what} slot {slot} moved by one must not verify: the REGISTER \
                 preprocessed root is derived from it, and the transcript absorbed \
                 that root"
            );
            moved += 1;
        }
    }
    assert_eq!(moved, 10, "every planned vector must have been run");
}

/// ★ LEDGER ENTRY 1, in two halves — and the obvious formulation of this test is
/// VACUOUS, which is worth stating because I wrote it first.
///
/// The tempting test is "set a boundary word to `2^32` and watch the assembled
/// epoch fail". It does fail — and it fails with the check REMOVED too, because a
/// wide value moves the derived REGISTER root, which moves every challenge drawn
/// after Phase A absorbs it. So that test says nothing about the width check at
/// all; it is the same rejection
/// `the_derivation_binds_every_register_boundary_word` already gets from moving a
/// word by one.
///
/// What is not vacuous is the pair below, and together they are complete:
///
/// 1. **What [`super::epoch::assert_u32`] does**, in isolation: the whole `u32`
///    range runs and everything at or above `2^32` is unprovable. Absolute — it is
///    a property of the check's own output, with no epoch involved.
/// 2. **That it is applied to every one of the 134 boundary cells**, structurally:
///    each register-arena `Hint` output is the INPUT of a 32-bit `BitDec`. A check
///    emitted over a prefix — the failure mode a value-tamper test cannot see,
///    since any single moved word rejects anyway — fails this.
#[test]
fn the_register_boundary_is_width_checked() {
    // ---- (1) the check itself.
    let drive = |v: u64| {
        let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
        let arena = b.declare_arena(1);
        let cell = b.hint_felt(arena, 0);
        super::epoch::assert_u32(&mut b, cell);
        let program = compile(b.finish());
        validate(&program).expect("the width check must be admissible");
        execute(&program, &[vec![base_word(FE::from(v))]], &TestPermutation).is_ok()
    };
    // ⚠ The bad values are CANONICAL felts, and that is not pedantry — it is the
    // exact size of the gap. An arena word is a field element, so `FE::from(v)`
    // reduces: `u64::MAX − 1` is the felt `2^32 − 3`, a perfectly good `u32`, and a
    // test that used it would report the check broken when it is not (it did). The
    // widening entry 1 names is therefore the interval `[2^32, p)` and nothing
    // beyond — there is no felt at or above `p` to worry about.
    const P_MINUS_1: u64 = 0xFFFF_FFFF_0000_0000; // Goldilocks p − 1 = 2^64 − 2^32
    for ok in [0u64, 1, 255, 1 << 31, (1u64 << 32) - 1] {
        assert!(drive(ok), "{ok} is a u32 and must be admitted");
    }
    for bad in [1u64 << 32, (1u64 << 32) + 1, 1 << 40, P_MINUS_1] {
        assert!(
            !drive(bad),
            "{bad} is not a u32 and must be unprovable: production's boundary \
             vectors are Vec<u32> and the TYPE is their only enforcement"
        );
    }

    // ---- (2) every boundary cell reaches it, in the assembled program.
    use std::collections::HashSet;
    let e = real_epoch();
    let program = epoch_challenge_program(&e);
    let num_reg = crate::tables::register::NUM_REGISTER_ADDRESSES;

    // The two register arenas are the ones whose declared length is
    // NUM_REGISTER_ADDRESSES; identified by length rather than by index so that
    // adding an epoch-wide arena cannot silently point this test at the wrong one.
    let reg_arenas: Vec<super::instr::ArenaId> = program
        .arena_schema
        .lens
        .iter()
        .enumerate()
        .filter(|(_, l)| **l as usize == num_reg)
        .map(|(i, _)| i as super::instr::ArenaId)
        .collect();
    assert_eq!(
        reg_arenas.len(),
        2,
        "expected exactly the INIT and FINI arenas to have the register width"
    );

    let mut boundary_cells: HashSet<super::instr::Addr> = HashSet::new();
    for instr in &program.instrs {
        if let super::instr::Instr::Hint { arena, out, .. } = instr
            && reg_arenas.contains(arena)
        {
            boundary_cells.insert(*out);
        }
    }
    assert_eq!(
        boundary_cells.len(),
        2 * num_reg,
        "every declared boundary word must be read exactly once"
    );

    let decomposed: HashSet<super::instr::Addr> = program
        .instrs
        .iter()
        .filter_map(|i| match i {
            super::instr::Instr::BitDec { input, bits } if bits.len() == 32 => Some(*input),
            _ => None,
        })
        .collect();
    let unchecked: Vec<_> = boundary_cells.difference(&decomposed).collect();
    assert!(
        unchecked.is_empty(),
        "these register-boundary cells are never bit-decomposed, so their width \
         is unconstrained: {unchecked:?}"
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
