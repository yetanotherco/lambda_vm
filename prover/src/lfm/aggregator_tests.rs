//! The aggregation layer's building block: a batched-LFM VERIFY LEG — the
//! emitted verifier of one batched-format wrap proof.
//!
//! The aggregation program is N of these legs (one per wrap) plus the binding
//! legs and the final attestation. A leg is the first emitted verifier whose
//! TARGET is an LFM-machine proof rather than a VM epoch: the spine replays
//! [`super::statement::absorb_lfm_statement`] byte for byte (the wrap's
//! program id is an EMIT-TIME CONSTANT — the aggregator is compiled for five
//! named wrap identities, which fold into its own program identity), the
//! preprocessed roots absorb from the AIR set as constants, and the LogUp
//! closure's target is the LFM_PUBLIC balance recomputed from the wrap's
//! claimed public words — the machine twin of
//! `verify_against_batched`'s `expected_public_balance`.
//!
//! Everything soundness-critical is the SAME emission machinery the wrap
//! program already gates: `emit_batched_epoch_challenges` for the spine,
//! `emit_mixed_verify_batch` / `emit_group_authentication` for the walks,
//! `emit_analyzed` / `emit_quotient` / `emit_deep_*` for the legs,
//! `emit_query_mix` / `emit_batched_query_fri` / the standalone terminal
//! checks for FRI. This module contributes no new cryptographic arithmetic —
//! only the LFM-shaped statement, the public-word hinting (canonicity-guarded
//! halves), and the balance target.

use stark::batched::proof::BatchedMultiProof;
use stark::batched::shape::{EpochFriParams, EpochShape};
use stark::batched::verifier::{EpochChallenges, replay_epoch_transcript};
use stark::config::Commitment;

use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};

use super::airs::LfmAirs;
use super::builder::{Cell, Ext, Felt, LfmBuilder};
use super::compiler::{LfmProgram, compile};
use super::edsl;
use super::epoch::RootCells;
use super::executor::execute;
use super::hash::TestPermutation;
use super::instr::ArenaId;
use super::proof::{BatchedLfmProof, aggregation_wrap_options, verify_against_batched};
use super::registry::{LfmArtifacts, build_artifacts};
use super::statement::{LFM_MACHINE_VERSION, LFM_STATEMENT_TAG, absorb_lfm_statement};
use super::transcript_replay::{Candidate, TranscriptReplay, assert_canonical, candidate_to_felt};
use super::word::{LfmWord, base_word, ext_word};

type Gl = GoldilocksField;
type Ext3 = GoldilocksExtension;

/// One wrap proof, production-accepted, with everything its emitted verify
/// leg needs — the LFM sibling of `RealBatchedEpoch`, minus the VM statement
/// machinery it has no use for.
pub(super) struct RealBatchedLfm {
    pub(super) opts: crate::ProofOptions,
    pub(super) artifacts: LfmArtifacts,
    pub(super) proof: BatchedMultiProof<Gl, Ext3, ()>,
    pub(super) public_words: Vec<(u32, LfmWord)>,
    pub(super) shape: EpochShape,
    pub(super) fri_params: EpochFriParams,
    /// Production's own challenge replay — the differential oracle.
    pub(super) challenges: EpochChallenges<Ext3>,
}

impl RealBatchedLfm {
    /// The chip AIR set in slot order — rebuilt on demand exactly as
    /// `verify_against_batched` rebuilds it (the AIRs borrow the airs value,
    /// so the set is materialized per use rather than stored).
    pub(super) fn airs(&self) -> LfmAirs {
        LfmAirs::new_with_hasher(
            &self.artifacts.roots,
            &self.opts,
            self.artifacts.keccak_rnd_chunks,
            self.artifacts.hasher,
            self.artifacts.chip_set,
        )
    }
}

/// Build the harness from a production-accepted wrap. Panics loudly on a wrap
/// production would reject — nothing downstream may read one.
pub(super) fn real_batched_lfm(
    artifacts: LfmArtifacts,
    opts: crate::ProofOptions,
    wrap: &BatchedLfmProof,
) -> RealBatchedLfm {
    assert!(
        verify_against_batched(&artifacts, &wrap.proof, &wrap.public_words, &opts),
        "the harness only reads wraps production accepts"
    );
    let airs = LfmAirs::new_with_hasher(
        &artifacts.roots,
        &opts,
        artifacts.keccak_rnd_chunks,
        artifacts.hasher,
        artifacts.chip_set,
    );
    let refs = airs.air_refs();
    let mut t = stark::config::DefaultStarkTranscript::<Ext3>::new(&[]);
    absorb_lfm_statement(
        &mut t,
        &artifacts.program_id,
        &wrap.public_words,
        opts.fri_final_poly_log_degree,
    );
    let (shape, fri_params, challenges) =
        replay_epoch_transcript(&refs, &wrap.proof, &mut t).expect("an accepted wrap replays");
    RealBatchedLfm {
        opts,
        artifacts,
        proof: wrap.proof.clone(),
        public_words: wrap.public_words.clone(),
        shape,
        fri_params,
        challenges,
    }
}

// ======================= arena serializers (T1) ==========================

/// The wrap-leg's opening arena — `batched_opening_arena`'s body over an LFM
/// proof (no carve; the closed-form word count is the shared
/// `batched_opening_words_per_query`).
pub(super) fn lfm_opening_arena(e: &RealBatchedLfm) -> Vec<LfmWord> {
    use stark::fri::mmcs::MixedOpening;
    fn push_mixed_base(out: &mut Vec<LfmWord>, o: &MixedOpening<Gl>) {
        for m in &o.per_matrix {
            out.extend(m.evaluations.iter().map(|v| base_word(*v)));
            out.extend(m.evaluations_sym.iter().map(|v| base_word(*v)));
        }
        out.extend(super::proof_arena::commitments_to_arena(
            &o.proof.merkle_path,
        ));
    }
    fn push_mixed_ext(out: &mut Vec<LfmWord>, o: &MixedOpening<Ext3>) {
        for m in &o.per_matrix {
            out.extend(m.evaluations.iter().map(ext_word));
            out.extend(m.evaluations_sym.iter().map(ext_word));
        }
        out.extend(super::proof_arena::commitments_to_arena(
            &o.proof.merkle_path,
        ));
    }

    let mut out = Vec::new();
    for q in &e.proof.queries {
        for p in &q.prep {
            out.extend(p.evaluations.iter().map(|v| base_word(*v)));
            out.extend(p.evaluations_sym.iter().map(|v| base_word(*v)));
            out.extend(super::proof_arena::commitments_to_arena(
                &p.proof.merkle_path,
            ));
        }
        assert!(q.carved_main.is_none(), "an LFM wrap has no carved table");
        push_mixed_base(&mut out, &q.main);
        if let Some(aux) = &q.aux {
            push_mixed_ext(&mut out, aux);
        }
        push_mixed_ext(&mut out, &q.parts);
    }
    assert_eq!(
        out.len(),
        e.proof.queries.len()
            * super::epoch_verify_tests::batched_opening_words_per_query(&e.shape),
        "the leg's opening arena must fill exactly what the shape declares"
    );
    out
}

/// The wrap-leg's FRI arena — `batched_fri_arena`'s body over an LFM proof.
pub(super) fn lfm_fri_arena(e: &RealBatchedLfm) -> Vec<LfmWord> {
    let mut out = Vec::new();
    for q in &e.proof.queries {
        assert_eq!(
            q.fri.layers_evaluations_sym.len(),
            q.fri.layers_auth_paths.len(),
            "every committed layer opens a symmetric evaluation AND a path"
        );
        for (sym, path) in q
            .fri
            .layers_evaluations_sym
            .iter()
            .zip(&q.fri.layers_auth_paths)
        {
            out.push(ext_word(sym));
            out.extend(super::proof_arena::commitments_to_arena(&path.merkle_path));
        }
    }
    assert_eq!(
        out.len(),
        e.proof.queries.len()
            * super::epoch_verify_tests::batched_fri_words_per_query(&e.shape, &e.fri_params),
        "the leg's FRI arena must fill exactly what the shape declares"
    );
    out
}

/// The wrap's public words as the leg's arena expects them: per word, the
/// four lanes each as `[low32, high32]` halves — eight halves per word, in
/// the wrap's own publish order.
pub(super) fn lfm_publics_arena(words: &[(u32, LfmWord)]) -> Vec<LfmWord> {
    let mut out = Vec::new();
    for (_, word) in words {
        for lane in word {
            let v: u64 = lane.canonical();
            out.push(base_word(FE::from(v & 0xFFFF_FFFF)));
            out.push(base_word(FE::from(v >> 32)));
        }
    }
    out
}

// ==================== the emitted statement + publics ====================

/// One hinted public word: the emit-time-constant index, the eight hinted
/// halves (absorbed by the statement), and the four lanes reassembled as
/// CANONICITY-GUARDED felts (consumed by the balance and the binding legs).
pub(super) struct HintedPublicWord {
    pub(super) index: u32,
    pub(super) halves: Vec<Felt>,
    pub(super) lanes: Vec<Felt>,
}

/// Hint the wrap's public words from `arena` (eight halves per word, the
/// serializer's layout) and reassemble each lane under the canonicity guard —
/// the same `(lo, hi)` predicate the transcript replay's own digest-to-felt
/// path enforces, so a hinted half pair CANNOT alias a second representation
/// of the same felt while absorbing different bytes.
pub(super) fn hint_public_words(
    b: &mut LfmBuilder,
    arena: ArenaId,
    words: &[(u32, LfmWord)],
) -> Vec<HintedPublicWord> {
    let mut cursor = 0u32;
    words
        .iter()
        .map(|(index, _)| {
            let mut halves = Vec::with_capacity(8);
            let mut lanes = Vec::with_capacity(4);
            for _ in 0..4 {
                let lo = b.hint_felt(arena, cursor);
                let hi = b.hint_felt(arena, cursor + 1);
                cursor += 2;
                let c = Candidate { lo, hi };
                assert_canonical(b, c);
                lanes.push(candidate_to_felt(b, c));
                halves.push(lo);
                halves.push(hi);
            }
            HintedPublicWord {
                index: *index,
                halves,
                lanes,
            }
        })
        .collect()
}

/// Emits [`absorb_lfm_statement`] byte for byte: the tag, the wrap's program
/// id (a PROGRAM CONSTANT — verdict condition 3's pinning), the machine
/// version, the word count, each word's emit-time-constant index and hinted
/// lane halves, and the FRI terminal byte.
pub(super) fn emit_lfm_statement(
    t: &mut TranscriptReplay,
    program_id: &Commitment,
    words: &[HintedPublicWord],
    fri_final_poly_log_degree: u8,
) {
    t.append_const_bytes(LFM_STATEMENT_TAG);
    t.append_const_bytes(program_id);
    t.append_const_bytes(&LFM_MACHINE_VERSION.to_le_bytes());
    t.append_const_bytes(&(words.len() as u64).to_le_bytes());
    for word in words {
        t.append_const_bytes(&word.index.to_le_bytes());
        t.append_halves_misaligned(&word.halves);
    }
    t.append_const_bytes(&[fri_final_poly_log_degree]);
}

/// The LFM_PUBLIC balance the leg's LogUp closure must reach —
/// `expected_public_balance`'s machine twin:
/// `Σ_i 1/(z − (LfmPublic + index_i·α + Σ_l lane_l·α^{2+l}))`, with each
/// division by an interned one so a fingerprint collision with `z` is
/// unprovable rather than `0/0 = 1`.
pub(super) fn emit_public_balance(
    b: &mut LfmBuilder,
    words: &[HintedPublicWord],
    z: Ext,
    alpha: Ext,
) -> Ext {
    let bus = b.ext_const(&FEE::from(crate::tables::types::BusId::LfmPublic as u64));
    let one = b.ext_const(&FEE::one());
    // α¹..α⁵ — index takes α, lane l takes α^{2+l}.
    let mut powers = Vec::with_capacity(5);
    powers.push(alpha);
    for i in 1..5 {
        let next = b.emul(powers[i - 1], alpha);
        powers.push(next);
    }
    let mut acc: Option<Ext> = None;
    for word in words {
        let idx = b.felt_const(FE::from(word.index as u64));
        let idx_term = b.emul_base(powers[0], idx);
        let mut linear = b.eadd(bus, idx_term);
        for (l, lane) in word.lanes.iter().enumerate() {
            let term = b.emul_base(powers[1 + l], *lane);
            linear = b.eadd(linear, term);
        }
        let fingerprint = b.esub(z, linear);
        let term = b.ediv(one, fingerprint);
        acc = Some(match acc {
            None => term,
            Some(a) => b.eadd(a, term),
        });
    }
    acc.unwrap_or_else(|| b.ext_const(&FEE::zero()))
}

// =========================== the verify leg ==============================

/// The leg's arenas, declared in ABSORB ORDER — the caller declares one set
/// per wrap, in wrap order, before emitting any leg, so the aggregation
/// program's declaration order is its absorb order end to end.
pub(super) struct LfmLegArenas {
    publics: ArenaId,
    main_root: ArenaId,
    aux_root: Option<ArenaId>,
    contrib: Vec<Option<ArenaId>>,
    ood: Vec<(ArenaId, ArenaId, ArenaId)>,
    parts_root: ArenaId,
    standalone: Vec<Option<ArenaId>>,
    fri_roots: ArenaId,
    fri_coeffs: ArenaId,
    nonce: Option<ArenaId>,
    openings: Option<ArenaId>,
    fri_legs: Option<ArenaId>,
}

pub(super) fn declare_lfm_leg_arenas(
    b: &mut LfmBuilder,
    e: &RealBatchedLfm,
    with_openings: bool,
) -> LfmLegArenas {
    let has_aux = !e.shape.aux.dims.is_empty();
    LfmLegArenas {
        publics: b.declare_arena(8 * e.public_words.len() as u32),
        main_root: b.declare_arena(super::edsl::digest_words(b)),
        aux_root: has_aux.then(|| b.declare_arena(super::edsl::digest_words(b))),
        contrib: e
            .airs()
            .air_refs()
            .iter()
            .map(|air| air.has_aux_trace().then(|| b.declare_arena(1)))
            .collect(),
        ood: e
            .proof
            .tables
            .iter()
            .map(|t| {
                (
                    b.declare_arena(
                        (t.trace_ood_evaluations.width * t.trace_ood_evaluations.height) as u32,
                    ),
                    b.declare_arena(
                        (t.trace_ood_next_evaluations.width * t.trace_ood_next_evaluations.height)
                            as u32,
                    ),
                    b.declare_arena(t.composition_poly_parts_ood_evaluation.len() as u32),
                )
            })
            .collect(),
        parts_root: b.declare_arena(super::edsl::digest_words(b)),
        standalone: {
            let fri = super::batched_epoch::BatchedFriShape::new(
                &e.shape.heights,
                e.fri_params.blowup_log,
                e.fri_params.final_poly_log_degree,
            );
            (0..e.proof.tables.len())
                .map(|t| {
                    fri.plan.standalone.contains(&t).then(|| {
                        b.declare_arena(
                            1u32 << (e.shape.heights[t] as u32 - e.fri_params.blowup_log),
                        )
                    })
                })
                .collect()
        },
        fri_roots: b
            .declare_arena(super::edsl::digest_words(b) * e.proof.fri_layer_roots.len() as u32),
        fri_coeffs: b.declare_arena(e.proof.fri_final_poly_coeffs.len() as u32),
        nonce: (e.fri_params.grinding_factor > 0).then(|| b.declare_arena(1)),
        openings: with_openings.then(|| {
            b.declare_arena(
                (e.proof.queries.len()
                    * super::epoch_verify_tests::batched_opening_words_per_query(&e.shape))
                    as u32,
            )
        }),
        fri_legs: with_openings.then(|| {
            b.declare_arena(
                (e.proof.queries.len()
                    * super::epoch_verify_tests::batched_fri_words_per_query(
                        &e.shape,
                        &e.fri_params,
                    )) as u32,
            )
        }),
    }
}

/// What a leg hands the aggregator's binding layer: the wrap's hinted public
/// words (index + canonicity-guarded lanes — byte-compare material) and the
/// challenge cells (diagnostic publishes for the gates).
pub(super) struct LfmLegCells {
    pub(super) publics: Vec<HintedPublicWord>,
    pub(super) lookup: (Ext, Ext),
    pub(super) betas: Vec<Ext>,
    pub(super) zs: Vec<Ext>,
    pub(super) gammas: Vec<Ext>,
    pub(super) alpha: Ext,
    pub(super) zetas: Vec<Ext>,
    pub(super) iota_bits: Vec<Vec<super::builder::Bit>>,
}

/// Emit ONE wrap's complete verification: statement, spine, LogUp closure
/// against the public balance, and every opening walk — the batched wrap
/// program's own structure with the LFM statement and prep-as-constants in
/// place of the VM epoch's statement and provenance machinery.
pub(super) fn emit_lfm_leg(
    b: &mut LfmBuilder,
    e: &RealBatchedLfm,
    a: &LfmLegArenas,
) -> LfmLegCells {
    use super::batched_epoch::{
        BatchedEpochAbsorbs, BatchedEpochShape, BatchedFriShape, BatchedPrepRoot, BatchedTableOod,
        BatchedTableShape, emit_batched_epoch_challenges,
    };
    use super::batched_epoch_verify::{
        MixedMatrixOpening, emit_mixed_verify_batch, reduce_iota_bits,
    };
    use super::deep::DeepOpening;
    use super::sub_proof::{GroupCommitment, GroupOpening, GroupShape};

    let airs = e.airs();
    let refs = airs.air_refs();
    let n = e.proof.tables.len();

    // ---- the emitted shape (the leg's compile-time truth) ----
    let tables: Vec<BatchedTableShape> = e
        .proof
        .tables
        .iter()
        .zip(&refs)
        .map(|(t, air)| BatchedTableShape {
            log2_trace_length: t.trace_length.trailing_zeros(),
            has_contribution: air.has_aux_trace(),
            ood_current_dims: (
                t.trace_ood_evaluations.width,
                t.trace_ood_evaluations.height,
            ),
            ood_next_dims: (
                t.trace_ood_next_evaluations.width,
                t.trace_ood_next_evaluations.height,
            ),
            num_parts: t.composition_poly_parts_ood_evaluation.len(),
        })
        .collect();
    let shape = BatchedEpochShape {
        tables,
        heights: e.shape.heights.clone(),
        total_widths: e.shape.total_widths(),
        log2_blowup: e.fri_params.blowup_log,
        coset_offset: FE::from(e.fri_params.coset_offset),
        has_aux: !e.shape.aux.dims.is_empty(),
        carved_main: None,
        fri: BatchedFriShape::new(
            &e.shape.heights,
            e.fri_params.blowup_log,
            e.fri_params.final_poly_log_degree,
        ),
        grinding_factor: e.fri_params.grinding_factor,
        num_queries: e.fri_params.num_queries,
    };

    // ---- the statement ----
    let publics = hint_public_words(b, a.publics, &e.public_words);
    let mut t = TranscriptReplay::new(&[]);
    emit_lfm_statement(
        &mut t,
        &e.artifacts.program_id,
        &publics,
        e.opts.fri_final_poly_log_degree,
    );

    // ---- preprocessed roots: EMIT-TIME CONSTANTS from the AIR set ----
    let prep_consts: Vec<Option<Commitment>> = refs
        .iter()
        .map(|air| air.is_preprocessed().then(|| air.precomputed_commitment()))
        .collect();
    let prep_cells: Vec<Option<RootCells>> = prep_consts
        .iter()
        .map(|c| c.as_ref().map(|c| RootCells::constant(b, c)))
        .collect();
    let prep_slots: Vec<Option<BatchedPrepRoot<'_>>> = prep_consts
        .iter()
        .map(|c| c.as_ref().map(BatchedPrepRoot::Constant))
        .collect();

    // ---- the proof-carried cells ----
    let main_cells = RootCells::hint(b, a.main_root, 0);
    let aux_cells = a.aux_root.map(|id| RootCells::hint(b, id, 0));
    let contribs: Vec<Option<Ext>> = a
        .contrib
        .iter()
        .map(|id| id.map(|id| b.hint_word(id, 0).as_ext()))
        .collect();
    let ood_cells: Vec<(Vec<Ext>, Vec<Ext>, Vec<Ext>)> = shape
        .tables
        .iter()
        .zip(&a.ood)
        .map(|(t, (ac, an, ap))| {
            (
                (0..(t.ood_current_dims.0 * t.ood_current_dims.1) as u32)
                    .map(|k| b.hint_word(*ac, k).as_ext())
                    .collect(),
                (0..(t.ood_next_dims.0 * t.ood_next_dims.1) as u32)
                    .map(|k| b.hint_word(*an, k).as_ext())
                    .collect(),
                (0..t.num_parts as u32)
                    .map(|k| b.hint_word(*ap, k).as_ext())
                    .collect(),
            )
        })
        .collect();
    let parts_cells = RootCells::hint(b, a.parts_root, 0);
    let standalone_cells: Vec<Option<Vec<Ext>>> = a
        .standalone
        .iter()
        .enumerate()
        .map(|(t, id)| {
            id.map(|id| {
                (0..1u32 << (e.shape.heights[t] as u32 - e.fri_params.blowup_log))
                    .map(|k| b.hint_word(id, k).as_ext())
                    .collect()
            })
        })
        .collect();
    let fri_root_cells: Vec<RootCells> = (0..e.proof.fri_layer_roots.len())
        .map(|k| {
            RootCells::hint(
                b,
                a.fri_roots,
                super::proof_arena::words_per_root() as u32 * k as u32,
            )
        })
        .collect();
    let coeff_cells: Vec<Ext> = (0..e.proof.fri_final_poly_coeffs.len() as u32)
        .map(|k| b.hint_word(a.fri_coeffs, k).as_ext())
        .collect();
    let nonce = a.nonce.map(|id| b.hint_felt(id, 0));

    // ---- the ONE-transcript spine ----
    let oods: Vec<BatchedTableOod<'_>> = ood_cells
        .iter()
        .map(|(c, x, p)| BatchedTableOod {
            current: c,
            next: x,
            parts: p,
        })
        .collect();
    let ch = emit_batched_epoch_challenges(
        b,
        &mut t,
        &shape,
        &BatchedEpochAbsorbs {
            prep_roots: &prep_slots,
            carved_root: None,
            main_root: &main_cells,
            aux_root: aux_cells.as_ref(),
            contributions: &contribs,
            parts_root: &parts_cells,
            ood: &oods,
            standalone_coeffs: &standalone_cells,
            fri_roots: &fri_root_cells,
            fri_coeffs: &coeff_cells,
            nonce,
        },
    );

    // ---- the LogUp closure against the PUBLIC balance ----
    let contributions: Vec<Ext> = contribs.iter().copied().flatten().collect();
    let target = emit_public_balance(b, &publics, ch.lookup.0, ch.lookup.1);
    let lshape = super::logup::LogUpShape {
        num_contributing_tables: contributions.len(),
        num_output_bytes: 0,
    };
    super::logup::emit_bus_closure(b, &lshape, &contributions, target);

    // ---- the opening walks (the wrap program's own skeleton, no carve) ----
    let h_max_fri = e.shape.heights.iter().copied().max().expect("chips");
    let prep_pos: Vec<Option<usize>> = (0..n)
        .map(|t| e.shape.prep.tables.iter().position(|&x| x == t))
        .collect();
    let main_pos: Vec<Option<usize>> = (0..n)
        .map(|t| e.shape.main.tables.iter().position(|&x| x == t))
        .collect();
    let aux_pos: Vec<Option<usize>> = (0..n)
        .map(|t| e.shape.aux.tables.iter().position(|&x| x == t))
        .collect();
    let parts_pos: Vec<usize> = (0..n)
        .map(|t| {
            e.shape
                .parts
                .tables
                .iter()
                .position(|&x| x == t)
                .expect("every chip has a parts matrix")
        })
        .collect();

    struct Leg {
        deep: super::deep::DeepShape,
        analysis: super::constraints::Analysis,
        quotient: super::constraints::QuotientShape,
        main_width: usize,
        num_alpha_powers: usize,
    }
    let legs: Vec<Leg> = refs
        .iter()
        .zip(&e.proof.tables)
        .map(|(air, data)| {
            use stark::verifier::{IsStarkVerifier, Verifier};
            let layout = Verifier::<Gl, Ext3, ()>::ood_layout(*air);
            let artifact = stark::constraint_ir::ConstraintArtifact::capture(*air);
            let (main_width, aux_width) = air.trace_layout();
            let num_total_cols = main_width + aux_width;
            let has_aux = air.has_aux_trace();
            Leg {
                deep: super::deep::DeepShape {
                    step_size: layout.step_size(),
                    num_eval_points: artifact.shape.transition_offsets.len() * layout.step_size(),
                    num_total_cols,
                    next_row_cols: layout.next_row_cols().to_vec(),
                    num_composition_parts: data.composition_poly_parts_ood_evaluation.len(),
                    log2_trace_length: data.trace_length.trailing_zeros(),
                },
                analysis: super::constraints::analyze(&artifact),
                quotient: super::constraints::QuotientShape {
                    log2_trace_length: data.trace_length.trailing_zeros(),
                    num_composition_parts: data.composition_poly_parts_ood_evaluation.len(),
                    boundary: super::epoch_verify::boundary_terms(has_aux, num_total_cols),
                },
                main_width,
                num_alpha_powers: if has_aux {
                    artifact.shape.max_bus_elements as usize
                } else {
                    0
                },
            }
        })
        .collect();

    let dinvs: Vec<super::deep::DeepInvariants> = (0..n)
        .map(|t_i| {
            let leg = &legs[t_i];
            let grid = super::epoch::emit_reconstruct_ood(
                b,
                &leg.deep,
                &ood_cells[t_i].0,
                &ood_cells[t_i].1,
            );
            let alpha_powers = if leg.num_alpha_powers > 0 {
                super::constraints::emit_alpha_powers(b, ch.lookup.1, leg.num_alpha_powers)
            } else {
                Vec::new()
            };
            let table_offset = match contribs[t_i] {
                Some(l) => {
                    super::constraints::emit_table_offset(b, l, leg.quotient.log2_trace_length)
                }
                None => b.felt_const(FE::zero()).as_ext(),
            };
            let steps = super::epoch_verify::frame_step_view(&grid, leg.deep.step_size);
            let ood_ops = super::constraints::OodOperands {
                steps,
                main_width: leg.main_width,
                rap_challenges: vec![ch.lookup.0, ch.lookup.1],
                alpha_powers,
                table_offset,
            };
            let evals = super::constraints::emit_analyzed(b, &leg.analysis, &ood_ops);
            let q = super::constraints::emit_quotient(
                b,
                &leg.quotient,
                &ood_ops,
                ch.zs[t_i],
                ch.betas[t_i],
                &evals,
                &ood_cells[t_i].2,
            );
            b.assert_eq_ext(q.claimed, q.composition);
            super::deep::emit_deep_invariants(
                b,
                &leg.deep,
                ch.gammas[t_i],
                ch.zs[t_i],
                &grid,
                &ood_cells[t_i].2,
            )
        })
        .collect();
    let fri_layer_commitments: Vec<super::fri::LayerCommitment> = fri_root_cells
        .iter()
        .map(|c| super::fri::LayerCommitment {
            root_lanes: c.lanes.clone(),
        })
        .collect();

    let mut cursor: u32 = 0;
    let mut fri_cursor: u32 = 0;
    let (a_open, a_fri) = match (a.openings, a.fri_legs) {
        (Some(o), Some(f)) => (o, f),
        _ => {
            return LfmLegCells {
                publics,
                lookup: ch.lookup,
                betas: ch.betas,
                zs: ch.zs,
                gammas: ch.gammas,
                alpha: ch.alpha,
                zetas: ch.zetas,
                iota_bits: ch.iota_bits,
            };
        }
    };
    for bits in &ch.iota_bits {
        // ---- preprocessed walks (roots are program constants) ----
        let mut prep_values: Vec<Vec<Cell>> = Vec::new();
        for (slot, &(h, w)) in e.shape.prep.tables.iter().zip(e.shape.prep.dims.iter()) {
            let cells = prep_cells[*slot]
                .as_ref()
                .expect("a preprocessed chip has root cells");
            let values = hint_run(b, a_open, &mut cursor, 2 * w);
            let siblings = hint_digests(b, a_open, &mut cursor, h - 1);
            let tbits = reduce_iota_bits(bits, h_max_fri, h);
            super::sub_proof::emit_group_authentication(
                b,
                &GroupCommitment::from_lanes(
                    cells.lanes.clone(),
                    GroupShape {
                        num_columns: w,
                        is_ext: false,
                    },
                ),
                &GroupOpening {
                    values: values.clone(),
                    siblings,
                },
                tbits,
            );
            prep_values.push(values);
        }

        // ---- the three mixed rounds ----
        let mut round_values: Vec<Vec<Vec<Cell>>> = Vec::new();
        let mut rounds: Vec<(&stark::batched::shape::RoundShape, &RootCells, bool)> =
            vec![(&e.shape.main, &main_cells, false)];
        if let Some(aux) = aux_cells.as_ref() {
            rounds.push((&e.shape.aux, aux, true));
        }
        rounds.push((&e.shape.parts, &parts_cells, true));
        for (round, root, is_ext) in rounds {
            let h_round = round.h_max().expect("a committed round is non-empty");
            let per_values: Vec<Vec<Cell>> = round
                .dims
                .iter()
                .map(|&(_, w)| hint_run(b, a_open, &mut cursor, 2 * w))
                .collect();
            let siblings = hint_digests(b, a_open, &mut cursor, h_round - 1);
            let matrices: Vec<MixedMatrixOpening<'_>> = round
                .dims
                .iter()
                .zip(&per_values)
                .map(|(&(h, w), values)| MixedMatrixOpening {
                    shape: GroupShape {
                        num_columns: w,
                        is_ext,
                    },
                    log_height: h,
                    values,
                })
                .collect();
            let rbits = reduce_iota_bits(bits, h_max_fri, h_round);
            emit_mixed_verify_batch(b, root, &matrices, &siblings, rbits);
            round_values.push(per_values);
        }
        let main_values = &round_values[0];
        let aux_values = aux_cells.as_ref().map(|_| &round_values[1]);
        let parts_values = round_values.last().expect("the parts round");

        // ---- the crossing ----
        let mut points: Vec<(Felt, Felt)> = Vec::with_capacity(n);
        let mut deep_pairs: Vec<(Ext, Ext)> = Vec::with_capacity(n);
        for t_i in 0..n {
            let h_t = e.shape.heights[t_i];
            let rbits = reduce_iota_bits(bits, h_max_fri, h_t);
            let (point, point_sym) =
                super::sub_proof::emit_points_from_bits(b, h_t as u32, shape.coset_offset, rbits);

            let mut trace = Vec::with_capacity(legs[t_i].deep.num_total_cols);
            let mut trace_sym = Vec::with_capacity(legs[t_i].deep.num_total_cols);
            if let Some(m) = prep_pos[t_i] {
                let w = e.shape.prep.dims[m].1;
                let vals = &prep_values[m];
                trace.extend((0..w).map(|c| vals[c].as_ext()));
                trace_sym.extend((0..w).map(|c| vals[w + c].as_ext()));
            }
            let m = main_pos[t_i].expect("every LFM chip has a main matrix");
            {
                let w = e.shape.main.dims[m].1;
                let vals = &main_values[m];
                trace.extend((0..w).map(|c| vals[c].as_ext()));
                trace_sym.extend((0..w).map(|c| vals[w + c].as_ext()));
            }
            if let Some(m) = aux_pos[t_i] {
                let w = e.shape.aux.dims[m].1;
                let vals = &aux_values.expect("an aux position implies an aux round")[m];
                trace.extend((0..w).map(|c| vals[c].as_ext()));
                trace_sym.extend((0..w).map(|c| vals[w + c].as_ext()));
            }
            assert_eq!(
                trace.len(),
                legs[t_i].deep.num_total_cols,
                "the crossing must cover exactly the DEEP column set"
            );
            let m = parts_pos[t_i];
            let w = e.shape.parts.dims[m].1;
            assert_eq!(
                w, legs[t_i].deep.num_composition_parts,
                "the parts matrix is one column per composition part"
            );
            let vals = &parts_values[m];
            let parts: Vec<Ext> = (0..w).map(|c| vals[c].as_ext()).collect();
            let parts_sym: Vec<Ext> = (0..w).map(|c| vals[w + c].as_ext()).collect();

            let regular = DeepOpening {
                point,
                trace,
                parts,
            };
            let symmetric = DeepOpening {
                point: point_sym,
                trace: trace_sym,
                parts: parts_sym,
            };
            deep_pairs.push((
                super::deep::emit_deep_point(
                    b,
                    &legs[t_i].deep,
                    ch.gammas[t_i],
                    &dinvs[t_i],
                    &regular,
                ),
                super::deep::emit_deep_point(
                    b,
                    &legs[t_i].deep,
                    ch.gammas[t_i],
                    &dinvs[t_i],
                    &symmetric,
                ),
            ));
            points.push((point, point_sym));
        }

        // ---- the mix, the batched instance, the standalone class ----
        let (p0, p0_sym, buckets) = super::batched_epoch_verify::emit_query_mix(
            b,
            &shape.fri.plan.batched,
            &e.shape.heights,
            h_max_fri,
            ch.alpha,
            &deep_pairs,
            bits,
        );
        let tallest = e
            .shape
            .heights
            .iter()
            .position(|&h| h == h_max_fri)
            .expect("a tallest chip exists");
        let fri_openings_q: Vec<super::fri::LayerOpening> = (0..shape.fri.num_committed())
            .map(|i| {
                let sym = {
                    let c = b.hint_word(a_fri, fri_cursor);
                    fri_cursor += 1;
                    c.as_ext()
                };
                let siblings = hint_digests(b, a_fri, &mut fri_cursor, h_max_fri - i - 2);
                super::fri::LayerOpening { sym, siblings }
            })
            .collect();
        super::batched_epoch_verify::emit_batched_query_fri(
            b,
            &shape.fri.layout,
            h_max_fri,
            &fri_layer_commitments,
            &ch.zetas,
            &coeff_cells,
            bits,
            points[tallest].0,
            points[tallest].1,
            p0,
            p0_sym,
            &buckets,
            &fri_openings_q,
        );
        for &t_i in &shape.fri.plan.standalone {
            let coeffs = standalone_cells[t_i]
                .as_ref()
                .expect("a standalone chip has terminal cells");
            super::batched_epoch_verify::emit_standalone_terminal_check(
                b,
                coeffs,
                points[t_i].0,
                points[t_i].1,
                deep_pairs[t_i].0,
                deep_pairs[t_i].1,
            );
        }
    }

    LfmLegCells {
        publics,
        lookup: ch.lookup,
        betas: ch.betas,
        zs: ch.zs,
        gammas: ch.gammas,
        alpha: ch.alpha,
        zetas: ch.zetas,
        iota_bits: ch.iota_bits,
    }
}

fn hint_run(b: &mut LfmBuilder, arena: ArenaId, cursor: &mut u32, count: usize) -> Vec<Cell> {
    (0..count)
        .map(|_| {
            let c = b.hint_word(arena, *cursor);
            *cursor += 1;
            c
        })
        .collect()
}

fn hint_digests(
    b: &mut LfmBuilder,
    arena: ArenaId,
    cursor: &mut u32,
    count: usize,
) -> Vec<edsl::WrapDigest> {
    (0..count)
        .map(|_| {
            // The stride is the DIGEST's width, not a literal two.
            let d = super::edsl::hint_digest(b, arena, *cursor);
            *cursor += super::edsl::digest_words(b);
            d
        })
        .collect()
}

// ========================= the aggregation program =======================

/// Where each schema field sits in a carved wrap's published words — pure
/// arithmetic over the wrap's shape, every term an emit-time constant. The
/// publish order is the wrap program's own: the LogUp pair, the attestation
/// id, β/z/γ per table, the DEEP α, the fold ζs, the ι felts, the bus total,
/// then the carved schema — register init and fini vectors, the epoch label
/// halves, the output bytes, the carved L2G root halves.
pub(super) struct WrapPublicLayout {
    pub(super) n_tables: usize,
    pub(super) n_zetas: usize,
    pub(super) n_iotas: usize,
    pub(super) num_reg: usize,
    pub(super) out_bytes: usize,
}

impl WrapPublicLayout {
    /// The layout comes from the INNER epoch the wrap program verifies — the
    /// published words are the wrap PROGRAM's outputs, so every count here is
    /// the inner epoch's (its table count, its committed FRI layers, its
    /// query count), never the wrap proof's own. The caller builds it where
    /// the wrap program was emitted; `assert_covers` then pins it against
    /// the wrap's actual published length, so a level confusion is a loud
    /// failure at assembly time rather than a silent mis-binding.
    pub(super) fn of_inner(e: &super::epoch_tests::RealBatchedEpoch) -> Self {
        Self {
            n_tables: e.proof.tables.len(),
            n_zetas: e.challenges.fri.betas.len(),
            n_iotas: e.fri_params.num_queries,
            num_reg: crate::tables::register::NUM_REGISTER_ADDRESSES,
            out_bytes: e.statement.public_output_len,
        }
    }
    fn total(&self) -> usize {
        self.schema_start() + 2 * self.num_reg + 2 + self.out_bytes + 8
    }
    fn assert_covers(&self, wrap: &RealBatchedLfm) {
        assert_eq!(
            self.total(),
            wrap.public_words.len(),
            "the layout must cover the wrap's published words exactly \
             (n={}, zetas={}, iotas={}, num_reg={}, out={})",
            self.n_tables,
            self.n_zetas,
            self.n_iotas,
            self.num_reg,
            self.out_bytes,
        );
    }
    fn id(&self, half: usize) -> usize {
        2 + half
    }
    fn schema_start(&self) -> usize {
        2 + 2 + 3 * self.n_tables + 1 + self.n_zetas + self.n_iotas + 1
    }
    fn reg_init(&self, r: usize) -> usize {
        self.schema_start() + r
    }
    fn reg_fini(&self, r: usize) -> usize {
        self.schema_start() + self.num_reg + r
    }
    fn label(&self, half: usize) -> usize {
        self.schema_start() + 2 * self.num_reg + half
    }
    fn out_byte(&self, i: usize) -> usize {
        self.schema_start() + 2 * self.num_reg + 2 + i
    }
    fn l2g_half(&self, h: usize) -> usize {
        self.schema_start() + 2 * self.num_reg + 2 + self.out_bytes + h
    }
}

/// Assert two hinted public words carry the same value, lane by lane.
fn assert_words_equal(b: &mut LfmBuilder, x: &HintedPublicWord, y: &HintedPublicWord) {
    for (xl, yl) in x.lanes.iter().zip(&y.lanes) {
        let xe = xl.as_ext();
        let ye = yl.as_ext();
        b.assert_eq_ext(xe, ye);
    }
}

/// Assert a hinted public word's base value equals a program constant (lanes
/// 1..4 must be zero — a base publish).
fn assert_word_is_const(b: &mut LfmBuilder, x: &HintedPublicWord, v: u64) {
    let c = b.ext_const(&FEE::from(v));
    let x0 = x.lanes[0].as_ext();
    b.assert_eq_ext(x0, c);
    let zero = b.ext_const(&FEE::zero());
    for lane in &x.lanes[1..] {
        let le = lane.as_ext();
        b.assert_eq_ext(le, zero);
    }
}

/// The cross-wrap binding legs (verdict conditions: the chain is a CHECK on
/// published words, never a trust): one shared attestation id across every
/// wrap, each wrap's register fini vector equal to the next wrap's init
/// vector, and each wrap's epoch label pinned to its chain position as an
/// emit-time constant.
fn emit_wrap_chain_bindings(
    b: &mut LfmBuilder,
    legs: &[LfmLegCells],
    layouts: &[WrapPublicLayout],
    labels: &[u64],
) {
    assert_eq!(legs.len(), layouts.len());
    assert_eq!(legs.len(), labels.len());
    for k in 1..legs.len() {
        for half in 0..2 {
            assert_words_equal(
                b,
                &legs[0].publics[layouts[0].id(half)],
                &legs[k].publics[layouts[k].id(half)],
            );
        }
    }
    for k in 0..legs.len() - 1 {
        for r in 0..layouts[k].num_reg {
            assert_words_equal(
                b,
                &legs[k].publics[layouts[k].reg_fini(r)],
                &legs[k + 1].publics[layouts[k + 1].reg_init(r)],
            );
        }
    }
    for (k, &label) in labels.iter().enumerate() {
        assert_word_is_const(
            b,
            &legs[k].publics[layouts[k].label(0)],
            label & 0xFFFF_FFFF,
        );
        assert_word_is_const(b, &legs[k].publics[layouts[k].label(1)], label >> 32);
    }
}

/// The assembled aggregation program — the block proof's statement:
///
/// SIX uniform batched-LFM verify legs (the five epoch wraps + the wrap of
/// the global-verifier program), the chain bindings (one shared attestation
/// id, register fini→init across every seam, labels pinned to chain
/// positions), the ★ L2G byte-compare (each epoch wrap's published carved
/// root equals the global wrap's published re-commit root for that epoch —
/// the root-equality binding, in-VM), and the ★ final attestation: the
/// num_pages > 0 program-id fold over the hinted (elf, pc, decode) — joined
/// to every wrap's published id through the num_pages = 0 fold of the SAME
/// cells — plus the block's genesis page commitments.
///
/// Published words, in order (the block artifact's own schema):
/// the final attestation id (2 words), wrap 0's register init vector, the
/// final wrap's register fini vector, the final wrap's output bytes, each
/// epoch's L2G root halves (8 per epoch), each folded page's base halves
/// (2 per page), the private-input page count, and the touched-page-base
/// list (count then bases, as constants of this block's program).
pub(super) struct BlockContext<'a> {
    pub(super) num_l2g: usize,
    pub(super) pages: usize,
    pub(super) touched_pages: &'a [u64],
    pub(super) num_private_input_pages: usize,
}

pub(super) fn aggregator_program(
    wraps: &[RealBatchedLfm],
    layouts: &[WrapPublicLayout],
    labels: &[u64],
    global_wrap: &RealBatchedLfm,
    ctx: &BlockContext<'_>,
) -> LfmProgram {
    let BlockContext {
        num_l2g,
        pages,
        touched_pages,
        num_private_input_pages,
    } = *ctx;
    assert!(!wraps.is_empty());
    assert_eq!(wraps.len(), num_l2g, "one epoch wrap per L2G re-commit");
    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let arenas: Vec<LfmLegArenas> = wraps
        .iter()
        .map(|e| declare_lfm_leg_arenas(&mut b, e, true))
        .collect();
    let g_arena = declare_lfm_leg_arenas(&mut b, global_wrap, true);
    // The attestation fold's inputs, LAST in declaration order: the ELF
    // digest, the entry point, the DECODE root, then per folded page a u64
    // base and a 32-byte commitment (the epoch program's own page layout).
    let a_att = b.declare_arena(8 + 2 + 8 + 10 * pages as u32);

    let legs: Vec<LfmLegCells> = wraps
        .iter()
        .zip(&arenas)
        .map(|(e, a)| emit_lfm_leg(&mut b, e, a))
        .collect();
    let g_leg = emit_lfm_leg(&mut b, global_wrap, &g_arena);
    for (layout, wrap) in layouts.iter().zip(wraps) {
        layout.assert_covers(wrap);
    }
    assert_eq!(
        global_wrap.public_words.len(),
        2 + 8 * num_l2g,
        "the global wrap publishes its pair and one root per epoch"
    );
    emit_wrap_chain_bindings(&mut b, &legs, layouts, labels);

    // ---- ★ the L2G root-equality binding, in-VM: epoch wrap k's published
    // carved root == the global wrap's published re-commit root k ----
    for (k, (leg, layout)) in legs.iter().zip(layouts).enumerate() {
        for h in 0..8 {
            assert_words_equal(
                &mut b,
                &leg.publics[layout.l2g_half(h)],
                &g_leg.publics[2 + 8 * k + h],
            );
        }
    }

    // ---- ★ the attestation join and the final fold ----
    let elf_digest: Vec<Felt> = (0..8).map(|i| b.hint_felt(a_att, i)).collect();
    let pc_start: Vec<Felt> = (0..2).map(|i| b.hint_felt(a_att, 8 + i)).collect();
    let decode: Vec<Felt> = (0..8).map(|i| b.hint_felt(a_att, 10 + i)).collect();
    let page_halves: Vec<(Vec<Felt>, Vec<Felt>)> = (0..pages)
        .map(|k| {
            let base = 18 + 10 * k as u32;
            (
                (0..2).map(|j| b.hint_felt(a_att, base + j)).collect(),
                (0..8).map(|j| b.hint_felt(a_att, base + 2 + j)).collect(),
            )
        })
        .collect();
    let id0 = super::programs::emit_program_id(
        &mut b,
        super::programs::ProgramIdShape { num_pages: 0 },
        &elf_digest,
        &pc_start,
        &decode,
        &[],
    );
    let id0_cells = RootCells::from_digest(&mut b, id0);
    // One (elf, pc, decode) triple answers for EVERY wrap: the fold of the
    // hinted cells must equal each wrap's published attestation id.
    for (leg, layout) in legs.iter().zip(layouts) {
        for (w, lanes) in id0_cells.lanes.iter().enumerate() {
            let hinted = &leg.publics[layout.id(w)];
            for (l, lane) in lanes.iter().enumerate() {
                let computed = lane.as_ext();
                let claimed = hinted.lanes[l].as_ext();
                b.assert_eq_ext(computed, claimed);
            }
        }
    }
    let page_refs: Vec<(&[Felt], &[Felt])> = page_halves
        .iter()
        .map(|(base, root)| (&base[..], &root[..]))
        .collect();
    let id_final = super::programs::emit_program_id(
        &mut b,
        super::programs::ProgramIdShape { num_pages: pages },
        &elf_digest,
        &pc_start,
        &decode,
        &page_refs,
    );

    // ---- the block artifact's published words ----
    b.public(id_final[0]);
    b.public(id_final[1]);
    let first = &legs[0];
    let last = legs.last().expect("nonempty");
    let l_first = &layouts[0];
    let l_last = layouts.last().expect("nonempty");
    for r in 0..l_first.num_reg {
        b.public(first.publics[l_first.reg_init(r)].lanes[0].as_cell());
    }
    for r in 0..l_last.num_reg {
        b.public(last.publics[l_last.reg_fini(r)].lanes[0].as_cell());
    }
    for i in 0..l_last.out_bytes {
        b.public(last.publics[l_last.out_byte(i)].lanes[0].as_cell());
    }
    for (leg, layout) in legs.iter().zip(layouts.iter()) {
        for h in 0..8 {
            b.public(leg.publics[layout.l2g_half(h)].lanes[0].as_cell());
        }
    }
    for (base, _) in &page_halves {
        for half in base {
            b.public(half.as_cell());
        }
    }
    let npriv = b.felt_const(FE::from(num_private_input_pages as u64));
    b.public(npriv.as_cell());
    let count = b.felt_const(FE::from(touched_pages.len() as u64));
    b.public(count.as_cell());
    for base in touched_pages {
        let lo = b.felt_const(FE::from(*base & 0xFFFF_FFFF));
        b.public(lo.as_cell());
        let hi = b.felt_const(FE::from(*base >> 32));
        b.public(hi.as_cell());
    }
    compile(b.finish())
}

// ==================== the global-verifier leg (option 3) ==================

/// The cross-epoch global memory proof, production-accepted, harvested for
/// emission: per-table shapes and challenges (the per-table machinery's own
/// harvest), the Phase-A prep constants (page genesis commitments — AIR-set
/// constants at emit time), and the statement bytes (every field an
/// emit-time constant of the block).
pub(super) struct RealGlobal {
    pub(super) statement_bytes: Vec<u8>,
    pub(super) tables: Vec<super::epoch_tests::HostTable>,
    pub(super) legs: Vec<super::epoch_verify_tests::TableLegs>,
    pub(super) num_l2g: usize,
    pub(super) z_alpha: (FEE, FEE),
}

/// Harvest the bundle's global proof. Panics loudly on a proof production
/// rejects. Mirrors `verify_global`'s AIR reconstruction exactly (the
/// no-supplied-roots arm: data-page genesis recomputed from the ELF).
pub(super) fn real_global(
    elf_bytes: &[u8],
    bundle: &crate::continuation::ContinuationProof,
    opts: &crate::ProofOptions,
) -> RealGlobal {
    use crypto::fiat_shamir::is_transcript::IsTranscript;
    use executor::elf::Elf;
    use stark::verifier::IsStarkVerifier;

    let elf = Elf::load(elf_bytes).expect("the ELF must load");
    let num_epochs = bundle.num_epochs();
    let npriv = bundle.num_private_pages();
    let page_bases: Vec<u64> = {
        let mut b: Vec<u64> = bundle.touched_pages().to_vec();
        b.sort_unstable();
        b.dedup();
        b
    };
    let l2g_airs: Vec<_> = (0..num_epochs)
        .map(|i| {
            crate::continuation::l2g_global_air(
                opts,
                crate::tables::local_to_global::epoch_label(i as u64),
            )
        })
        .collect();
    let gm_configs = crate::continuation::global_memory_configs(&page_bases, &elf, npriv);
    let gm_airs: Vec<_> = gm_configs
        .iter()
        .map(|config| crate::continuation::global_memory_air(opts, config, None))
        .collect();
    let mut refs: Vec<
        &dyn stark::traits::AIR<Field = Gl, FieldExtension = Ext3, PublicInputs = ()>,
    > = l2g_airs
        .iter()
        .map(|a| a as &dyn stark::traits::AIR<Field = Gl, FieldExtension = Ext3, PublicInputs = ()>)
        .collect();
    for air in &gm_airs {
        refs.push(air);
    }

    // The statement, byte for byte — `absorb_continuation_global_statement`'s
    // encoding over emit-time constants, pinned by the harness differential
    // (the seed below absorbs through the production function; the leg's
    // emitted challenges must then match the harvested ones, which fails if
    // this local encoding ever drifts).
    let mut statement_bytes = Vec::new();
    statement_bytes.extend_from_slice(crate::statement::CONTINUATION_GLOBAL_TAG);
    statement_bytes.extend_from_slice(&crate::statement::elf_digest(elf_bytes));
    statement_bytes.extend_from_slice(&(num_epochs as u64).to_le_bytes());
    statement_bytes.extend_from_slice(&(npriv as u64).to_le_bytes());
    statement_bytes.push(opts.fri_final_poly_log_degree);
    statement_bytes.extend_from_slice(&(page_bases.len() as u64).to_le_bytes());
    for base in &page_bases {
        statement_bytes.extend_from_slice(&u64::to_le_bytes(*base));
    }

    let seed = || {
        let mut t = stark::config::DefaultStarkTranscript::<Ext3>::new(&[]);
        crate::statement::absorb_continuation_global_statement(
            &mut t,
            elf_bytes,
            num_epochs,
            npriv,
            opts.fri_final_poly_log_degree,
            &page_bases,
        );
        t
    };
    let view = bundle.global_proof_view();
    assert_eq!(refs.len(), view.len(), "one AIR per global sub-proof");
    assert!(
        stark::verifier::Verifier::multi_verify_views(&refs, view, &mut seed(), &FEE::zero()),
        "production's verifier must accept the global proof"
    );

    // Phase A + the shared LogUp pair, transcribed as the epoch harvest does.
    let mut transcript = seed();
    for (idx, air) in refs.iter().enumerate() {
        let v = view.get(idx);
        if air.is_preprocessed() {
            transcript.append_bytes(&air.precomputed_commitment());
        }
        transcript.append_bytes(v.lde_trace_main_merkle_root());
    }
    let lookup: Vec<FEE> = (0..stark::lookup::LOGUP_NUM_CHALLENGES)
        .map(|_| transcript.sample_field_element())
        .collect();
    let z_alpha = (lookup[0], lookup[1]);

    let num_tables = refs.len();
    let tables: Vec<super::epoch_tests::HostTable> = refs
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
            super::epoch_tests::host_table_forked(*air, v, idx, num_tables, &mut fork, &lookup)
        })
        .collect();
    let legs = refs
        .iter()
        .enumerate()
        .map(|(idx, air)| super::epoch_verify_tests::build_table_legs(*air, view.get(idx), &lookup))
        .collect();

    RealGlobal {
        statement_bytes,
        tables,
        legs,
        num_l2g: num_epochs,
        z_alpha,
    }
}

/// Per-table arena set of the global-verifier program, in declaration order.
struct GlobalTableArenas {
    aux_root: Option<ArenaId>,
    contribution: Option<ArenaId>,
    composition_root: ArenaId,
    ood_current: ArenaId,
    ood_next: ArenaId,
    parts: ArenaId,
    fri_roots: ArenaId,
    fri_coeffs: ArenaId,
    nonce: Option<ArenaId>,
    legs: super::epoch_verify::TableQueryArenas,
}

/// The emitted verifier of the global proof — the per-table program's own
/// structure (statement, Phase A, one fork per table, full verification
/// legs, the LogUp closure) with the global statement as one constant run,
/// every preprocessed root an AIR-set constant, and the bus target ZERO
/// (`verify_global`'s own expected balance). PUBLISHES: the shared pair,
/// then each epoch's L2G re-commit main root (eight halves each, epoch
/// order) — the byte-compare material the aggregator binds against the five
/// wraps' published carved roots.
pub(super) fn global_verifier_program(g: &RealGlobal) -> LfmProgram {
    use super::epoch::{TableAbsorbs, fork_table};
    use super::statement_replay::{PhaseAPreprocessed, PhaseATable, replay_phase_a};

    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let n = g.tables.len();

    // ---- arenas, declaration order = absorb order ----
    let a_main_roots = b.declare_arena(super::edsl::digest_words(&b) * n as u32);
    let per_table: Vec<GlobalTableArenas> = g
        .tables
        .iter()
        .zip(&g.legs)
        .map(|(h, leg)| GlobalTableArenas {
            aux_root: h
                .shape
                .has_aux_root
                .then(|| b.declare_arena(super::edsl::digest_words(&b))),
            contribution: h.shape.has_contribution.then(|| b.declare_arena(1)),
            composition_root: b.declare_arena(super::edsl::digest_words(&b)),
            ood_current: b
                .declare_arena((h.shape.ood_current_dims.0 * h.shape.ood_current_dims.1) as u32),
            ood_next: b.declare_arena((h.shape.ood_next_dims.0 * h.shape.ood_next_dims.1) as u32),
            parts: b.declare_arena(h.shape.num_parts as u32),
            fri_roots: b
                .declare_arena(super::edsl::digest_words(&b) * h.shape.fri.num_committed() as u32),
            fri_coeffs: b.declare_arena(h.shape.fri.num_terminal_coeffs() as u32),
            nonce: (h.shape.grinding_factor > 0).then(|| b.declare_arena(1)),
            legs: super::epoch_verify::declare_table_arenas(&mut b, &leg.verify),
        })
        .collect();

    // ---- the statement: one constant run ----
    let mut t = TranscriptReplay::new(&[]);
    t.append_const_bytes(&g.statement_bytes);

    // ---- Phase A: prep constants, hinted main roots ----
    let main_cells: Vec<RootCells> = (0..n)
        .map(|i| {
            RootCells::hint(
                &mut b,
                a_main_roots,
                super::proof_arena::words_per_root() as u32 * i as u32,
            )
        })
        .collect();
    let main_halves: Vec<Vec<Felt>> = main_cells.iter().map(RootCells::lanes_flat).collect();
    let prep_cells: Vec<Option<RootCells>> = g
        .tables
        .iter()
        .map(|h| {
            h.precomputed_root
                .as_ref()
                .map(|c| RootCells::constant(&mut b, c))
        })
        .collect();
    let phase_a: Vec<PhaseATable> = g
        .tables
        .iter()
        .enumerate()
        .map(|(i, h)| PhaseATable {
            preprocessed_root: h
                .precomputed_root
                .as_ref()
                .map(PhaseAPreprocessed::Constant),
            main_root: &main_halves[i][..],
        })
        .collect();
    let (z, alpha) = replay_phase_a(&mut t, &mut b, &phase_a);
    b.public(z.as_cell());
    b.public(alpha.as_cell());
    // The aggregator's byte-compare material: each epoch's L2G re-commit
    // root, the very cells Phase A absorbed.
    for cells in main_cells.iter().take(g.num_l2g) {
        for half in cells.lanes_flat() {
            b.public(half.as_cell());
        }
    }

    // ---- one fork per table, with the full verification legs ----
    let mut contributions: Vec<Ext> = Vec::new();
    for (i, h) in g.tables.iter().enumerate() {
        let a = &per_table[i];
        let aux = a.aux_root.map(|id| RootCells::hint(&mut b, id, 0));
        let contribution = a.contribution.map(|id| b.hint_word(id, 0).as_ext());
        let composition = RootCells::hint(&mut b, a.composition_root, 0);
        let ood_current: Vec<Ext> = (0..(h.shape.ood_current_dims.0 * h.shape.ood_current_dims.1)
            as u32)
            .map(|k| b.hint_word(a.ood_current, k).as_ext())
            .collect();
        let ood_next: Vec<Ext> = (0..(h.shape.ood_next_dims.0 * h.shape.ood_next_dims.1) as u32)
            .map(|k| b.hint_word(a.ood_next, k).as_ext())
            .collect();
        let parts: Vec<Ext> = (0..h.shape.num_parts as u32)
            .map(|k| b.hint_word(a.parts, k).as_ext())
            .collect();
        let fri_roots: Vec<RootCells> = (0..h.shape.fri.num_committed())
            .map(|k| {
                RootCells::hint(
                    &mut b,
                    a.fri_roots,
                    super::proof_arena::words_per_root() as u32 * k as u32,
                )
            })
            .collect();
        let fri_coeffs: Vec<Ext> = (0..h.shape.fri.num_terminal_coeffs() as u32)
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
        let ch = super::epoch::emit_table_challenges(&mut b, &mut fork, &h.shape, &absorbs);
        let leg = &g.legs[i];
        super::epoch_verify::emit_table_verification(
            &mut b,
            &leg.verify,
            &leg.analysis,
            &ch,
            &absorbs,
            &super::epoch_verify::TableInputs {
                precomputed_root: prep_cells[i].as_ref(),
                main_root: &main_cells[i],
                rap_challenges: &[z, alpha],
            },
            &a.legs,
        );
    }

    // ---- the closure: the global bus balances to ZERO ----
    let shape = super::logup::LogUpShape {
        num_contributing_tables: contributions.len(),
        num_output_bytes: 0,
    };
    let target = b.ext_const(&FEE::zero());
    super::logup::emit_bus_closure(&mut b, &shape, &contributions, target);

    compile(b.finish())
}

/// The global program's arenas, in its declaration order.
pub(super) fn global_arena_words(g: &RealGlobal) -> Vec<Vec<LfmWord>> {
    let mut arenas: Vec<Vec<LfmWord>> = Vec::new();
    arenas.push(super::proof_arena::commitments_to_arena(
        &g.tables.iter().map(|h| h.main_root).collect::<Vec<_>>(),
    ));
    for (h, leg) in g.tables.iter().zip(&g.legs) {
        if let Some(root) = &h.aux_root {
            arenas.push(super::proof_arena::commitments_to_arena(&[*root]));
        }
        if let Some(c) = &h.contribution {
            arenas.push(vec![ext_word(c)]);
        }
        arenas.push(super::proof_arena::commitments_to_arena(&[
            h.composition_root
        ]));
        arenas.push(h.ood_current.iter().map(ext_word).collect());
        arenas.push(h.ood_next.iter().map(ext_word).collect());
        arenas.push(h.parts.iter().map(ext_word).collect());
        arenas.push(super::proof_arena::commitments_to_arena(&h.fri_roots));
        arenas.push(h.fri_coeffs.iter().map(ext_word).collect());
        if let Some(nonce) = h.nonce {
            arenas.push(vec![base_word(FE::from(nonce))]);
        }
        arenas.push(leg.opening_arena());
        arenas.push(leg.fri_arena());
    }
    arenas
}

// ============================ the gates ==================================

/// The fixture wrap at the aggregation preset, and its leg program that
/// publishes every challenge (the differential surface).
fn fixture_leg() -> (RealBatchedLfm, LfmProgram) {
    use super::programs::trivial_program;
    use super::proof::lfm_prove_batched;

    let opts = aggregation_wrap_options();
    let program = trivial_program();
    let artifacts = build_artifacts(&program, &opts);
    let arenas: Vec<Vec<LfmWord>> = vec![
        (0..4u64)
            .map(|i| core::array::from_fn(|j| FE::from(1_000 * (i + 1) + j as u64)))
            .collect(),
    ];
    let proved = lfm_prove_batched(&program, &artifacts, &arenas, &opts)
        .expect("the fixture wrap must prove at the aggregation preset");
    let e = real_batched_lfm(artifacts, opts, &proved);

    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let a = declare_lfm_leg_arenas(&mut b, &e, true);
    let cells = emit_lfm_leg(&mut b, &e, &a);
    b.public(cells.lookup.0.as_cell());
    b.public(cells.lookup.1.as_cell());
    for v in cells.betas.iter().chain(&cells.zs).chain(&cells.gammas) {
        b.public(v.as_cell());
    }
    b.public(cells.alpha.as_cell());
    for zeta in &cells.zetas {
        b.public(zeta.as_cell());
    }
    for bits in &cells.iota_bits {
        let felt = edsl::bits_to_felt(&mut b, bits);
        b.public(felt.as_cell());
    }
    (e, compile(b.finish()))
}

/// The leg's arenas for one wrap, in the declaration order above.
fn leg_arena_words(e: &RealBatchedLfm) -> Vec<Vec<LfmWord>> {
    let mut arenas: Vec<Vec<LfmWord>> = Vec::new();
    arenas.push(lfm_publics_arena(&e.public_words));
    arenas.push(super::proof_arena::commitments_to_arena(&[e
        .proof
        .main_root]));
    if !e.shape.aux.dims.is_empty() {
        arenas.push(super::proof_arena::commitments_to_arena(&[e
            .proof
            .aux_root
            .expect("an aux shape has an aux root")]));
    }
    for t in &e.proof.tables {
        if let Some(bus) = &t.bus_public_inputs {
            arenas.push(vec![ext_word(&bus.table_contribution)]);
        }
    }
    let block_words = |block: &stark::table::Table<Ext3>| -> Vec<LfmWord> {
        (0..block.height)
            .flat_map(|r| block.get_row(r).iter().map(ext_word).collect::<Vec<_>>())
            .collect()
    };
    for t in &e.proof.tables {
        arenas.push(block_words(&t.trace_ood_evaluations));
        arenas.push(block_words(&t.trace_ood_next_evaluations));
        arenas.push(
            t.composition_poly_parts_ood_evaluation
                .iter()
                .map(ext_word)
                .collect(),
        );
    }
    arenas.push(super::proof_arena::commitments_to_arena(&[e
        .proof
        .parts_root]));
    for t in &e.proof.tables {
        if let Some(coeffs) = &t.standalone_final_poly_coeffs {
            arenas.push(coeffs.iter().map(ext_word).collect());
        }
    }
    arenas.push(super::proof_arena::commitments_to_arena(
        &e.proof.fri_layer_roots,
    ));
    arenas.push(e.proof.fri_final_poly_coeffs.iter().map(ext_word).collect());
    if let Some(nonce) = e.proof.nonce {
        arenas.push(vec![base_word(FE::from(nonce))]);
    }
    arenas.push(lfm_opening_arena(e));
    arenas.push(lfm_fri_arena(e));
    arenas
}

/// ★ THE LEG RUNS — and its challenges are production's own. Executing the
/// leg proves every emitted assert held: the statement bytes matched the
/// spine's absorbs, the LogUp closure reached the PUBLIC balance recomputed
/// from the hinted words, every walk authenticated against the absorbed
/// roots, every quotient identity held, and FRI folded to the terminal. The
/// published challenges are then differentialled against
/// `replay_epoch_transcript`'s on the same wrap.
#[test]
fn the_lfm_wrap_leg_runs_and_matches_the_host_replay() {
    let (e, program) = fixture_leg();
    let arenas = leg_arena_words(&e);
    let exec = execute(&program, &arenas, &TestPermutation).expect("the leg must execute");

    let pub_ext = |i: usize| super::word::word_as_ext(&exec.public_words[i].1).expect("an ext");
    assert_eq!(pub_ext(0), e.challenges.lookup[0], "z");
    assert_eq!(pub_ext(1), e.challenges.lookup[1], "alpha");
    let n = e.proof.tables.len();
    for (i, beta) in e.challenges.betas.iter().enumerate() {
        assert_eq!(pub_ext(2 + i), *beta, "beta[{i}]");
    }
    for (i, z) in e.challenges.zs.iter().enumerate() {
        assert_eq!(pub_ext(2 + n + i), *z, "z[{i}]");
    }
    for (i, g) in e.challenges.deep_gammas.iter().enumerate() {
        assert_eq!(pub_ext(2 + 2 * n + i), *g, "gamma[{i}]");
    }
    assert_eq!(pub_ext(2 + 3 * n), e.challenges.fri.alpha, "DEEP alpha");
    for (i, zeta) in e.challenges.fri.betas.iter().enumerate() {
        assert_eq!(pub_ext(2 + 3 * n + 1 + i), *zeta, "fold beta[{i}]");
    }
    let iota_base = 2 + 3 * n + 1 + e.challenges.fri.betas.len();
    for (i, iota) in e.challenges.fri.iotas.iter().enumerate() {
        let got =
            super::word::word_as_base(&exec.public_words[iota_base + i].1).expect("an iota felt");
        assert_eq!(got, FE::from(*iota as u64), "iota[{i}]");
    }
}

/// A tampered wrap is UNPROVABLE through the leg: flip one opened main-round
/// value and the walk's authentication cannot reach the absorbed root.
#[test]
fn the_lfm_wrap_leg_rejects_a_tampered_proof() {
    let (e, program) = fixture_leg();
    let mut tampered_proof = e.proof.clone();
    tampered_proof.queries[0].main.per_matrix[0].evaluations[0] += FE::one();
    let tampered = RealBatchedLfm {
        proof: tampered_proof,
        ..e
    };
    let arenas = leg_arena_words(&tampered);
    assert!(
        execute(&program, &arenas, &TestPermutation).is_err(),
        "a tampered opening must make the leg unprovable"
    );
}

/// And a moved PUBLIC WORD is unprovable too — the balance target moves, the
/// closure's assert fails. This is the aggregator's claimed-public binding.
#[test]
fn the_lfm_wrap_leg_rejects_a_moved_public_word() {
    let (e, program) = fixture_leg();
    let mut words = e.public_words.clone();
    let w = words.first_mut().expect("the fixture publishes words");
    w.1[0] += FE::one();
    let moved = RealBatchedLfm {
        public_words: words,
        ..e
    };
    let arenas = leg_arena_words(&moved);
    assert!(
        execute(&program, &arenas, &TestPermutation).is_err(),
        "a moved public word must make the leg unprovable"
    );
}

/// The whole fixture pipeline below the aggregator: a batched-carved
/// continuation bundle, EVERY epoch wrapped from proofs alone in the BATCHED
/// format at the AGGREGATION preset, plus the chain-position labels.
#[allow(clippy::type_complexity)]
fn fixture_wraps() -> (
    Vec<RealBatchedLfm>,
    Vec<WrapPublicLayout>,
    Vec<u64>,
    crate::continuation::ContinuationProof,
    Vec<u8>,
) {
    use super::proof::lfm_prove_batched;

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
    assert!(n >= 2, "the aggregate needs a chain");

    let opts = aggregation_wrap_options();
    let mut wraps = Vec::with_capacity(n);
    let mut layouts = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);
    for k in 0..n {
        let e = super::epoch_tests::real_batched_epoch_from_continuation(
            &inner, &elf_bytes, &bundle, k, None,
        )
        .expect("every epoch must reconstruct from proofs alone");
        labels.push(e.epoch_label);
        layouts.push(WrapPublicLayout::of_inner(&e));
        let program = super::epoch_tests::batched_epoch_program_with(&e, true, false);
        let mut arenas = super::epoch_tests::batched_epoch_arenas(&e);
        arenas.push(super::epoch_verify_tests::batched_opening_arena(&e));
        arenas.push(super::epoch_verify_tests::batched_fri_arena(&e));
        let artifacts = build_artifacts(&program, &opts);
        let proved = lfm_prove_batched(&program, &artifacts, &arenas, &opts)
            .expect("the epoch's wrap must prove batched at the aggregation preset");
        wraps.push(real_batched_lfm(artifacts, opts.clone(), &proved));
    }
    (wraps, layouts, labels, bundle, elf_bytes)
}

/// Everything the six-leg fixture aggregate needs beyond the epoch wraps:
/// the global wrap and the attestation inputs, from the SAME bundle.
struct FixtureAggregate {
    wraps: Vec<RealBatchedLfm>,
    layouts: Vec<WrapPublicLayout>,
    labels: Vec<u64>,
    global_wrap: RealBatchedLfm,
    elf_digest: [u8; 32],
    pc_start: u64,
    decode_root: stark::config::Commitment,
    pages: Vec<(u64, stark::config::Commitment)>,
    touched: Vec<u64>,
    npriv: usize,
}

fn fixture_aggregate() -> FixtureAggregate {
    use super::proof::lfm_prove_batched;
    use executor::elf::Elf;

    let (wraps, layouts, labels, bundle, elf_bytes) = fixture_wraps();
    let inner = super::proof_fixture::fixture_options();
    let opts = aggregation_wrap_options();

    let g = real_global(&elf_bytes, &bundle, &inner);
    let program = global_verifier_program(&g);
    let arenas = global_arena_words(&g);
    let artifacts = build_artifacts(&program, &opts);
    let proved = lfm_prove_batched(&program, &artifacts, &arenas, &opts)
        .expect("the global wrap must prove batched at the aggregation preset");
    let global_wrap = real_batched_lfm(artifacts, opts, &proved);

    let elf = Elf::load(&elf_bytes).expect("the ELF must load");
    let (decode_root, mut pages) =
        crate::continuation::continuation_precomputed_commitments(&elf_bytes, &bundle, &inner)
            .expect("the consumer recompute must run");
    pages.sort_by_key(|(base, _)| *base);
    FixtureAggregate {
        wraps,
        layouts,
        labels,
        global_wrap,
        elf_digest: crate::statement::elf_digest(&elf_bytes),
        pc_start: elf.entry_point,
        decode_root,
        pages,
        touched: bundle.touched_pages().to_vec(),
        npriv: bundle.num_private_pages(),
    }
}

/// The attestation arena's words: elf digest, entry point, DECODE root, then
/// per folded page the base and commitment — all as u32 halves.
fn attestation_arena_words(f: &FixtureAggregate) -> Vec<LfmWord> {
    fn root_halves(out: &mut Vec<LfmWord>, root: &[u8; 32]) {
        for c in root.chunks(4) {
            out.push(base_word(FE::from(
                u32::from_le_bytes(c.try_into().expect("4 bytes")) as u64,
            )));
        }
    }
    let mut out = Vec::new();
    root_halves(&mut out, &f.elf_digest);
    out.push(base_word(FE::from(f.pc_start & 0xFFFF_FFFF)));
    out.push(base_word(FE::from(f.pc_start >> 32)));
    root_halves(&mut out, &f.decode_root);
    for (base, root) in &f.pages {
        out.push(base_word(FE::from(*base & 0xFFFF_FFFF)));
        out.push(base_word(FE::from(*base >> 32)));
        root_halves(&mut out, root);
    }
    out
}

/// ★ THE AGGREGATE RUNS — THE BLOCK STATEMENT AT FIXTURE SCALE: every epoch
/// of a batched-carved chain wrapped at the aggregation preset, the global
/// proof wrapped the same way, and ONE emitted program verifies all of them
/// plus the bindings — the chain (id, registers, labels), the in-VM L2G
/// root-equality against the global wrap, and the attestation join with the
/// final num_pages > 0 fold. The published id is then checked against the
/// CONSUMER'S OWN recompute (`program_id_from_digest` over
/// `continuation_precomputed_commitments`) — the contract's compare, run
/// here as the gate's oracle.
#[test]
fn the_assembled_aggregator_runs_on_the_fixture_chain() {
    let f = fixture_aggregate();
    let program = aggregator_program(
        &f.wraps,
        &f.layouts,
        &f.labels,
        &f.global_wrap,
        &BlockContext {
            num_l2g: f.wraps.len(),
            pages: f.pages.len(),
            touched_pages: &f.touched,
            num_private_input_pages: f.npriv,
        },
    );
    let mut arenas: Vec<Vec<LfmWord>> = f.wraps.iter().flat_map(leg_arena_words).collect();
    arenas.extend(leg_arena_words(&f.global_wrap));
    arenas.push(attestation_arena_words(&f));
    let exec = execute(&program, &arenas, &TestPermutation).expect("the aggregate must execute");

    // The consumer's own recompute is the oracle for the published id.
    let expected = crate::recursion::program_id_from_digest(
        &f.elf_digest,
        f.pc_start,
        &f.decode_root,
        &f.pages,
    );
    for w in 0..2 {
        let got = exec.public_words[w].1;
        let want: Vec<FE> = expected[16 * w..16 * (w + 1)]
            .chunks(4)
            .map(|c| FE::from(u32::from_le_bytes(c.try_into().expect("4 bytes")) as u64))
            .collect();
        // A digest word carries four u32 lanes.
        assert_eq!(got.to_vec(), want, "published id word {w}");
    }
    println!(
        "★ six-leg aggregate over {} epoch wraps + the global wrap: {} instructions,          {} published words; the published id MATCHES the consumer recompute",
        f.wraps.len(),
        program.instrs.len(),
        exec.public_words.len()
    );
}

/// The chain bindings DISCRIMINATE: a fini→init mismatch at a seam makes the
/// aggregate unprovable.
#[test]
fn the_aggregator_rejects_a_broken_register_chain() {
    let f = fixture_aggregate();
    let program = aggregator_program(
        &f.wraps,
        &f.layouts,
        &f.labels,
        &f.global_wrap,
        &BlockContext {
            num_l2g: f.wraps.len(),
            pages: f.pages.len(),
            touched_pages: &f.touched,
            num_private_input_pages: f.npriv,
        },
    );
    let mut arenas: Vec<Vec<LfmWord>> = f.wraps.iter().flat_map(leg_arena_words).collect();
    arenas.extend(leg_arena_words(&f.global_wrap));
    arenas.push(attestation_arena_words(&f));
    let word_index = f.layouts[0].reg_fini(0);
    arenas[0][8 * word_index] = base_word(
        super::word::word_as_base(&arenas[0][8 * word_index]).expect("a half") + FE::one(),
    );
    assert!(
        execute(&program, &arenas, &TestPermutation).is_err(),
        "a broken register chain must make the aggregate unprovable"
    );
}

/// The attestation join DISCRIMINATES: a flipped DECODE half in the fold's
/// arena makes the num_pages = 0 fold disagree with every wrap's published
/// id — unprovable, and nothing else about the proofs changed.
#[test]
fn the_aggregator_rejects_a_forged_attestation_input() {
    let f = fixture_aggregate();
    let program = aggregator_program(
        &f.wraps,
        &f.layouts,
        &f.labels,
        &f.global_wrap,
        &BlockContext {
            num_l2g: f.wraps.len(),
            pages: f.pages.len(),
            touched_pages: &f.touched,
            num_private_input_pages: f.npriv,
        },
    );
    let mut arenas: Vec<Vec<LfmWord>> = f.wraps.iter().flat_map(leg_arena_words).collect();
    arenas.extend(leg_arena_words(&f.global_wrap));
    let mut att = attestation_arena_words(&f);
    att[10][0] += FE::one(); // the DECODE root's first half
    arenas.push(att);
    assert!(
        execute(&program, &arenas, &TestPermutation).is_err(),
        "a forged attestation input must make the aggregate unprovable"
    );
}

/// The L2G binding DISCRIMINATES through the global side: a flipped root
/// half in the GLOBAL wrap's publics arena breaks its own leg's statement —
/// and would break the root-equality compare even if it did not.
#[test]
fn the_aggregator_rejects_a_moved_global_root() {
    let f = fixture_aggregate();
    let program = aggregator_program(
        &f.wraps,
        &f.layouts,
        &f.labels,
        &f.global_wrap,
        &BlockContext {
            num_l2g: f.wraps.len(),
            pages: f.pages.len(),
            touched_pages: &f.touched,
            num_private_input_pages: f.npriv,
        },
    );
    let mut arenas: Vec<Vec<LfmWord>> = f.wraps.iter().flat_map(leg_arena_words).collect();
    let g_base = arenas.len();
    arenas.extend(leg_arena_words(&f.global_wrap));
    arenas.push(attestation_arena_words(&f));
    // The global wrap's publics arena is its leg's first: word 2 is root 0
    // half 0 (after the pair), eight halves per word.
    arenas[g_base][8 * 2] =
        base_word(super::word::word_as_base(&arenas[g_base][8 * 2]).expect("a half") + FE::one());
    assert!(
        execute(&program, &arenas, &TestPermutation).is_err(),
        "a moved global L2G root must make the aggregate unprovable"
    );
}

/// ★ THE GLOBAL LEG RUNS: the emitted verifier of a REAL fixture bundle's
/// cross-epoch global proof — per-table verification of the L2G re-commits
/// and one GLOBAL_MEMORY table per touched page behind one constant-run
/// statement, closing the GlobalMemory bus at ZERO — and publishes each
/// epoch's L2G re-commit root. Differentialled against the harvest's own
/// production challenges via the published pair; tampered via a flipped
/// L2G main root (Phase A absorbs it, so the walk cannot reach it).
#[test]
fn the_global_verifier_leg_runs_and_rejects_tampers() {
    let elf_bytes = super::proof_fixture::read_inner_elf();
    let inner = super::proof_fixture::fixture_options();
    let bundle = crate::continuation::prove_continuation_batched(
        &elf_bytes,
        &[],
        super::proof_fixture::FIXTURE_EPOCH_LOG2,
        &inner,
    )
    .expect("the fixture continuation must prove batched");
    let g = real_global(&elf_bytes, &bundle, &inner);
    let program = global_verifier_program(&g);
    let arenas = global_arena_words(&g);
    let exec = execute(&program, &arenas, &TestPermutation).expect("the global leg must execute");

    let pub_ext = |i: usize| super::word::word_as_ext(&exec.public_words[i].1).expect("an ext");
    assert_eq!(pub_ext(0), g.z_alpha.0, "the global z");
    assert_eq!(pub_ext(1), g.z_alpha.1, "the global alpha");
    // The published L2G re-commit roots equal the harvested main roots.
    for k in 0..g.num_l2g {
        for h in 0..8 {
            let got = super::word::word_as_base(&exec.public_words[2 + 8 * k + h].1)
                .expect("a root half");
            let want = FE::from(u32::from_le_bytes(
                g.tables[k].main_root[4 * h..4 * h + 4]
                    .try_into()
                    .expect("a root is 32 bytes"),
            ) as u64);
            assert_eq!(got, want, "L2G root {k} half {h}");
        }
    }
    println!(
        "★ global leg: {} tables ({} L2G + {} pages), {} instructions, {} published words",
        g.tables.len(),
        g.num_l2g,
        g.tables.len() - g.num_l2g,
        program.instrs.len(),
        exec.public_words.len()
    );

    // Tamper: flip one byte of one L2G main root in the arena — Phase A then
    // absorbs a root the walks cannot authenticate against.
    let mut tampered = global_arena_words(&g);
    tampered[0][0][0] += FE::one();
    assert!(
        execute(&program, &tampered, &TestPermutation).is_err(),
        "a flipped L2G re-commit root must make the global leg unprovable"
    );
}

/// ★ The aggregate's QUERY CENSUS, per leg: the walks' wrap-hash
/// permutations are exactly the in-code closed form
/// (`batched_query_permutations_for` over the WRAP PROOF's shape), measured
/// as the delta between the with-walks and spine-only single-leg programs —
/// absolute, and hash-aware (the other hash's delta must be zero). The same
/// discipline the VM epoch census gate pins, generalized to the LFM legs the
/// aggregator is made of; the plan-level census rides this formula.
#[test]
fn the_aggregate_leg_census_matches_the_closed_form() {
    use super::programs::trivial_program;
    use super::proof::lfm_prove_batched;

    let opts = aggregation_wrap_options();
    let program = trivial_program();
    let artifacts = build_artifacts(&program, &opts);
    let arenas: Vec<Vec<LfmWord>> = vec![
        (0..4u64)
            .map(|i| core::array::from_fn(|j| FE::from(1_000 * (i + 1) + j as u64)))
            .collect(),
    ];
    let proved = lfm_prove_batched(&program, &artifacts, &arenas, &opts)
        .expect("the fixture wrap must prove");
    let e = real_batched_lfm(artifacts, opts, &proved);

    let build = |with: bool| -> LfmProgram {
        let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
        let a = declare_lfm_leg_arenas(&mut b, &e, with);
        let _ = emit_lfm_leg(&mut b, &e, &a);
        compile(b.finish())
    };
    let count = |p: &LfmProgram, keccak: bool| -> usize {
        p.instrs
            .iter()
            .filter(|i| match i {
                super::instr::Instr::KeccakF(_) => keccak,
                super::instr::Instr::Blake3(_) => !keccak,
                _ => false,
            })
            .count()
    };
    let spine = build(false);
    let full = build(true);
    let hash = super::edsl::WrapHash::production();
    let per_query =
        super::batched_epoch_verify::batched_query_permutations_for(&e.shape, &e.fri_params, hash);
    let is_keccak = matches!(hash, super::edsl::WrapHash::Keccak);
    let wrap_delta = count(&full, is_keccak) - count(&spine, is_keccak);
    let other_delta = count(&full, !is_keccak) - count(&spine, !is_keccak);
    assert_eq!(
        wrap_delta,
        e.proof.queries.len() * per_query,
        "an aggregator leg's walks must hash exactly the census closed form"
    );
    assert_eq!(other_delta, 0, "the walks hash under the wrap hash alone");

    // ★ The closed form is CHUNK-INVARIANT, and that is a property, not an
    // accident. `LFM_BLAKE3` chunking redistributes the chip's rows over AIR
    // instances AFTER compilation, so the instruction stream this census counts
    // is the same program either way. Asserted rather than argued: a chunking
    // that reached back into emission would move the census silently, and the
    // aggregation program is exactly where chunking gets switched on.
    let per = full.groups.blake3.real_rows.div_ceil(3).max(1);
    let chunked =
        full.with_blake3_chunking(super::chunking::Blake3Chunking::from_compressions(per));
    assert!(
        chunked.blake3_chunk_count() > 1,
        "the control needs a real split, got {} chunks of {per}",
        chunked.blake3_chunk_count()
    );
    assert_eq!(
        count(&chunked, is_keccak) - count(&spine, is_keccak),
        wrap_delta,
        "chunking must not move the leg's wrap-hash census"
    );

    eprintln!(
        "aggregate leg census: {per_query} wrap permutations/query over {} chips at the          aggregation preset",
        e.proof.tables.len()
    );
}

/// ★★★ THE BLOCK DRIVER — ONE PROOF FOR THE BLOCK, end to end in one
/// process. Box-tier; the same env contract as the P1/P2 drivers.
///
/// ```text
/// LFM_CENSUS_ELF=/path/to/ethrex.elf \
/// LFM_CENSUS_INPUT=/path/to/ethrex_mainnet_25368371.bin \
/// LFM_CENSUS_EPOCH_LOG2=24 LAMBDA_VM_MAX_ROWS_LOG2=24 \
/// cargo test --release -p lambda-vm-prover --lib \
///   lfm::aggregator_tests::the_real_block_aggregates_end_to_end -- --ignored --exact --nocapture
/// ```
///
/// Phases, each timed and printed: the batched-carved base (5 epochs +
/// global proof) → full bundle verification → 5 epoch wraps + the global
/// wrap, all batched at the aggregation preset, from proofs alone → the
/// aggregation program (six legs + bindings + attestation) → ★ THE
/// AGGREGATION PROVE → its complete verification → ★ THE CONSUMER RITUAL
/// (the contract's steps: the pinned verify just ran; recompute the
/// expected id from the trusted ELF + the artifact's published page data;
/// byte-compare against the published id; read the outputs) — timed, its
/// cost named in the record.
#[test]
#[ignore]
fn the_real_block_aggregates_end_to_end() {
    use super::proof::lfm_prove_batched;
    use executor::elf::Elf;
    use std::time::Instant;

    for var in ["LFM_CENSUS_ELF", "LFM_CENSUS_INPUT"] {
        assert!(
            std::env::var(var).is_ok(),
            "{var} must name a file: this driver proves the REAL block"
        );
    }
    let inputs = super::epoch_tests::EpochInputs::from_env();
    let inner = crate::recursion::Preset::Blowup4.options();
    let agg_opts = aggregation_wrap_options();
    println!(
        "★ P3 BLOCK RUN: guest {}, {} input bytes, 2^{} cycles/epoch, inner blowup {} / {} q, \
         wrap+aggregation blowup {} / {} q / fp{}",
        inputs.label,
        inputs.private_input.len(),
        inputs.epoch_log2,
        inner.blowup_factor,
        inner.fri_number_of_queries,
        agg_opts.blowup_factor,
        agg_opts.fri_number_of_queries,
        agg_opts.fri_final_poly_log_degree,
    );
    let t_total = Instant::now();

    // The artifact cache: with P3_ARTIFACT_DIR set, the bundle and all six
    // wrap proofs persist to disk after production (the rkyv wire), and a
    // relaunch LOADS them — an aggregation attempt never re-pays the base
    // and wrap proves. Programs and artifacts are re-emitted either way
    // (minutes, deterministic); only the PROVES are cached.
    let art_dir = std::env::var("P3_ARTIFACT_DIR").ok();
    let cache_path = |name: &str| art_dir.as_ref().map(|d| std::path::Path::new(d).join(name));
    let bundle_cached = cache_path("bundle.rkyv").is_some_and(|p| p.exists());

    // ---- base ----
    let t = Instant::now();
    let bundle = if bundle_cached {
        let bytes = std::fs::read(cache_path("bundle.rkyv").expect("cache path"))
            .expect("the cached bundle must read");
        let mut aligned = rkyv::util::AlignedVec::<16>::with_capacity(bytes.len());
        aligned.extend_from_slice(&bytes);
        rkyv::from_bytes::<crate::continuation::ContinuationProof, rkyv::rancor::Error>(&aligned)
            .expect("the cached bundle must deserialize")
    } else {
        crate::continuation::prove_continuation_batched(
            &inputs.elf_bytes,
            &inputs.private_input,
            inputs.epoch_log2,
            &inner,
        )
        .expect("the block must prove batched")
    };
    let n = bundle.num_epochs();
    if let (false, Some(dir)) = (bundle_cached, &art_dir) {
        std::fs::create_dir_all(dir).expect("the artifact dir must create");
        let bytes =
            rkyv::to_bytes::<rkyv::rancor::Error>(&bundle).expect("the bundle must serialize");
        std::fs::write(cache_path("bundle.rkyv").expect("cache path"), &bytes)
            .expect("the bundle must persist");
    }
    println!(
        "   base: {n} epochs + global proof in {:.1}s ({}), peak RSS {:?} GiB",
        t.elapsed().as_secs_f64(),
        if bundle_cached {
            "LOADED from cache"
        } else {
            "proved"
        },
        super::wrap_tests::peak_rss_gib(),
    );
    let t = Instant::now();
    let out = crate::continuation::verify_continuation(&inputs.elf_bytes, &bundle, &inner)
        .expect("the bundle must verify");
    assert!(out.is_some(), "the bundle must verify completely");
    println!("   host verify: {:.1}s", t.elapsed().as_secs_f64());

    // ---- the six wraps ----
    let t = Instant::now();
    let mut wraps = Vec::with_capacity(n);
    let mut layouts = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);
    for k in 0..n {
        let tk = Instant::now();
        let e = super::epoch_tests::real_batched_epoch_from_continuation(
            &inner,
            &inputs.elf_bytes,
            &bundle,
            k,
            None,
        )
        .expect("every epoch must reconstruct from proofs alone");
        labels.push(e.epoch_label);
        layouts.push(WrapPublicLayout::of_inner(&e));
        let program = super::epoch_tests::batched_epoch_program_with(&e, true, false);
        let mut arenas = super::epoch_tests::batched_epoch_arenas(&e);
        arenas.push(super::epoch_verify_tests::batched_opening_arena(&e));
        arenas.push(super::epoch_verify_tests::batched_fri_arena(&e));
        let artifacts = build_artifacts(&program, &agg_opts);
        let wrap_file = format!("wrap_{k}.rkyv");
        let cached = cache_path(&wrap_file).is_some_and(|p| p.exists());
        let tp = Instant::now();
        let proved = if cached {
            let bytes = std::fs::read(cache_path(&wrap_file).expect("cache path"))
                .expect("the cached wrap must read");
            let mut aligned = rkyv::util::AlignedVec::<16>::with_capacity(bytes.len());
            aligned.extend_from_slice(&bytes);
            rkyv::from_bytes::<BatchedLfmProof, rkyv::rancor::Error>(&aligned)
                .expect("the cached wrap must deserialize")
        } else {
            let proved = lfm_prove_batched(&program, &artifacts, &arenas, &agg_opts)
                .expect("the epoch wrap must prove");
            if let Some(p) = cache_path(&wrap_file) {
                let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&proved)
                    .expect("the wrap must serialize");
                std::fs::write(p, &bytes).expect("the wrap must persist");
            }
            proved
        };
        println!(
            "   epoch {k}: construct {:.1}s, wrap prove {:.1}s ({}), {} program instrs",
            tk.elapsed().as_secs_f64() - tp.elapsed().as_secs_f64(),
            tp.elapsed().as_secs_f64(),
            if cached { "LOADED" } else { "proved" },
            program.instrs.len(),
        );
        wraps.push(real_batched_lfm(artifacts, agg_opts.clone(), &proved));
    }
    let tg = Instant::now();
    let g = real_global(&inputs.elf_bytes, &bundle, &inner);
    let g_program = global_verifier_program(&g);
    let g_arenas = global_arena_words(&g);
    let g_artifacts = build_artifacts(&g_program, &agg_opts);
    let g_cached = cache_path("global_wrap.rkyv").is_some_and(|p| p.exists());
    let tp = Instant::now();
    let g_proved = if g_cached {
        let bytes = std::fs::read(cache_path("global_wrap.rkyv").expect("cache path"))
            .expect("the cached global wrap must read");
        let mut aligned = rkyv::util::AlignedVec::<16>::with_capacity(bytes.len());
        aligned.extend_from_slice(&bytes);
        rkyv::from_bytes::<BatchedLfmProof, rkyv::rancor::Error>(&aligned)
            .expect("the cached global wrap must deserialize")
    } else {
        let proved = lfm_prove_batched(&g_program, &g_artifacts, &g_arenas, &agg_opts)
            .expect("the global wrap must prove");
        if let Some(p) = cache_path("global_wrap.rkyv") {
            let bytes =
                rkyv::to_bytes::<rkyv::rancor::Error>(&proved).expect("the wrap must serialize");
            std::fs::write(p, &bytes).expect("the global wrap must persist");
        }
        proved
    };
    println!(
        "   global: construct {:.1}s, wrap prove {:.1}s, {} tables, {} program instrs",
        tg.elapsed().as_secs_f64() - tp.elapsed().as_secs_f64(),
        tp.elapsed().as_secs_f64(),
        g.tables.len(),
        g_program.instrs.len(),
    );
    let global_wrap = real_batched_lfm(g_artifacts, agg_opts.clone(), &g_proved);
    println!("   wraps total: {:.1}s", t.elapsed().as_secs_f64());

    // ---- the aggregation ----
    let elf = Elf::load(&inputs.elf_bytes).expect("the ELF must load");
    // Timed on its own line: this native FFT+Merkle pass is the consumer
    // ritual's expensive half (design-review condition 1 asked for its
    // price; run 4 left it inside a ~401 s unaccounted gap).
    let t = Instant::now();
    let (decode_root, mut pages) = crate::continuation::continuation_precomputed_commitments(
        &inputs.elf_bytes,
        &bundle,
        &inner,
    )
    .expect("the consumer recompute must run");
    println!(
        "   consumer precompute (decode_root + pages, native FFT+Merkle): {:.1}s",
        t.elapsed().as_secs_f64()
    );
    pages.sort_by_key(|(base, _)| *base);
    let elf_digest = crate::statement::elf_digest(&inputs.elf_bytes);
    let t = Instant::now();
    // ★ LFM_BLAKE3 chunking, chosen at EMISSION time. The aggregation program is
    // where the chip's ~1.39M compressions land in ONE table, whose blowup-2 LDE
    // is a single ~102 GB allocation; `LFM_BLAKE3_MAX_CHUNK_ROWS_LOG2=k` splits
    // it into 2^k-row tables. Applied HERE and nowhere else: the wraps are cached
    // artifacts at the census point, and re-chunking them would invalidate them.
    // The chunk shape is bound into `program_id`, so the aggregation identity
    // moves with the knob — which is fine, and is what the consumer contract
    // pins.
    let blake3_chunking = super::chunking::Blake3Chunking::from_env();
    let mut program = aggregator_program(
        &wraps,
        &layouts,
        &labels,
        &global_wrap,
        &BlockContext {
            num_l2g: n,
            pages: pages.len(),
            touched_pages: bundle.touched_pages(),
            num_private_input_pages: bundle.num_private_pages(),
        },
    );
    if let Some(chunking) = blake3_chunking {
        program = program.with_blake3_chunking(chunking);
        println!(
            "   aggregation LFM_BLAKE3 chunking: {} compressions/chunk -> {} chunks of {:?} rows \
             ({} compressions) ({})",
            chunking.compressions_per_chunk(),
            program.blake3_chunk_count(),
            super::airs::blake3_chunk_rows(&program),
            program.groups.blake3.real_rows,
            super::chunking::BLAKE3_MAX_CHUNK_ROWS_LOG2_ENV,
        );
    }
    let program = program;
    let mut arenas: Vec<Vec<LfmWord>> = wraps.iter().flat_map(leg_arena_words).collect();
    arenas.extend(leg_arena_words(&global_wrap));
    let f = FixtureAggregate {
        wraps,
        layouts,
        labels,
        global_wrap,
        elf_digest,
        pc_start: elf.entry_point,
        decode_root,
        pages: pages.clone(),
        touched: bundle.touched_pages().to_vec(),
        npriv: bundle.num_private_pages(),
    };
    arenas.push(attestation_arena_words(&f));
    println!(
        "   aggregation program: {} instructions, emitted in {:.1}s",
        program.instrs.len(),
        t.elapsed().as_secs_f64()
    );

    // The TERMINAL layer's own options, decoupled from the wrap layer's: the
    // wraps and the aggregation PROGRAM keep Design A's blowup4/110q (the
    // census point — cached wrap proofs stay valid), while the aggregation
    // prove itself may take a smaller blowup. P3_AGG_TERMINAL_BLOWUP=2 halves
    // every LDE term — the 483 GiB box OOM-killed three straight attempts at
    // blowup 4 (P3-OOM-REPORT.md). The query count re-derives from the same
    // 128-bit Johnson target by construction (`with_blowup`), so 2 -> 219 q;
    // the FRI terminal stays at the aggregation preset's fp8.
    let terminal_opts = match std::env::var("P3_AGG_TERMINAL_BLOWUP") {
        Ok(b) => {
            let blowup: u8 = b
                .parse()
                .expect("P3_AGG_TERMINAL_BLOWUP must be a power-of-two u8");
            let mut o = stark::proof::options::GoldilocksCubicProofOptions::with_blowup(blowup)
                .expect("P3_AGG_TERMINAL_BLOWUP must be a valid blowup");
            o.fri_final_poly_log_degree = agg_opts.fri_final_poly_log_degree;
            println!(
                "   aggregation TERMINAL options: blowup {} / {} q / fp{} (P3_AGG_TERMINAL_BLOWUP)",
                o.blowup_factor, o.fri_number_of_queries, o.fri_final_poly_log_degree
            );
            o
        }
        Err(_) => agg_opts.clone(),
    };
    let t = Instant::now();
    let agg_artifacts = build_artifacts(&program, &terminal_opts);
    println!(
        "   aggregation artifacts built in {:.1}s",
        t.elapsed().as_secs_f64()
    );
    // The aggregation prove's own residency posture, decoupled from the wrap
    // proves': P3_AGG_RESIDENCY=recompute trades ~2× prove time for the LDE
    // peak (the first real-scale Retain attempt OOM-killed a 483 GiB box).
    // env::set_var is process-global and this driver is single-threaded by
    // contract (--test-threads=1); unsafe per the 2024 edition's signature.
    if let Ok(residency) = std::env::var("P3_AGG_RESIDENCY") {
        println!("   aggregation residency: {residency} (P3_AGG_RESIDENCY)");
        unsafe { std::env::set_var("LAMBDA_VM_RESIDENCY", residency) };
    }
    let agg_artifacts_ = &agg_artifacts;
    let t = Instant::now();
    let final_proof = lfm_prove_batched(&program, agg_artifacts_, &arenas, &terminal_opts)
        .expect("★ THE AGGREGATION MUST PROVE");
    let agg_prove_s = t.elapsed().as_secs_f64();
    let final_proof_bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&final_proof)
        .expect("the block proof must serialize");
    let final_bytes = final_proof_bytes.len();
    // THE deliverable persists: run 4 proved the block and saved nothing
    // but a byte count. Same cache dir as the inputs; the record run's
    // proof is the artifact of record.
    if let Some(path) = cache_path("block_proof.rkyv") {
        std::fs::write(&path, &final_proof_bytes).expect("the block proof must persist");
        println!("   block proof persisted: {}", path.display());
    }
    println!(
        "   ★ AGGREGATION PROVE: {agg_prove_s:.1}s, THE BLOCK PROOF = {final_bytes} bytes, \
         peak RSS {:?} GiB",
        super::wrap_tests::peak_rss_gib(),
    );

    // ---- verification + THE CONSUMER RITUAL ----
    let t = Instant::now();
    assert!(
        verify_against_batched(
            &agg_artifacts,
            &final_proof.proof,
            &final_proof.public_words,
            &terminal_opts
        ),
        "the block proof must verify against the pinned aggregator identity"
    );
    let verify_s = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let expected = crate::recursion::program_id_from_digest(
        &elf_digest,
        elf.entry_point,
        &decode_root,
        &pages,
    );
    for w in 0..2 {
        let got = final_proof.public_words[w].1;
        let want: Vec<FE> = expected[16 * w..16 * (w + 1)]
            .chunks(4)
            .map(|c| FE::from(u32::from_le_bytes(c.try_into().expect("4 bytes")) as u64))
            .collect();
        assert_eq!(
            got.to_vec(),
            want,
            "★ THE CONSUMER RITUAL: the published id must equal the recompute"
        );
    }
    let ritual_s = t.elapsed().as_secs_f64();
    println!(
        "   verify {verify_s:.2}s; consumer ritual (expected-id recompute + compare) {ritual_s:.2}s"
    );
    println!(
        "★★★ ONE PROOF FOR THE BLOCK: {final_bytes} bytes, total wall {:.1}s ({:.1} min), \
         peak RSS {:?} GiB — {} published words",
        t_total.elapsed().as_secs_f64(),
        t_total.elapsed().as_secs_f64() / 60.0,
        super::wrap_tests::peak_rss_gib(),
        final_proof.public_words.len(),
    );
}
