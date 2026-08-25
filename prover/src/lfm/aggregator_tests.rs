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
use stark::traits::AIR;

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
    openings: ArenaId,
    fri_legs: ArenaId,
}

pub(super) fn declare_lfm_leg_arenas(b: &mut LfmBuilder, e: &RealBatchedLfm) -> LfmLegArenas {
    let has_aux = !e.shape.aux.dims.is_empty();
    LfmLegArenas {
        publics: b.declare_arena(8 * e.public_words.len() as u32),
        main_root: b.declare_arena(2),
        aux_root: has_aux.then(|| b.declare_arena(2)),
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
        parts_root: b.declare_arena(2),
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
                            1u32 << (e.shape.heights[t] as u32 - e.fri_params.blowup_log as u32),
                        )
                    })
                })
                .collect()
        },
        fri_roots: b.declare_arena(2 * e.proof.fri_layer_roots.len() as u32),
        fri_coeffs: b.declare_arena(e.proof.fri_final_poly_coeffs.len() as u32),
        nonce: (e.fri_params.grinding_factor > 0).then(|| b.declare_arena(1)),
        openings: b.declare_arena(
            (e.proof.queries.len()
                * super::epoch_verify_tests::batched_opening_words_per_query(&e.shape))
                as u32,
        ),
        fri_legs: b.declare_arena(
            (e.proof.queries.len()
                * super::epoch_verify_tests::batched_fri_words_per_query(&e.shape, &e.fri_params))
                as u32,
        ),
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
                (0..1u32 << (e.shape.heights[t] as u32 - e.fri_params.blowup_log as u32))
                    .map(|k| b.hint_word(id, k).as_ext())
                    .collect()
            })
        })
        .collect();
    let fri_root_cells: Vec<RootCells> = (0..e.proof.fri_layer_roots.len())
        .map(|k| RootCells::hint(b, a.fri_roots, 2 * k as u32))
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
            root_lanes: c.lanes,
        })
        .collect();

    let mut cursor: u32 = 0;
    let mut fri_cursor: u32 = 0;
    for bits in &ch.iota_bits {
        // ---- preprocessed walks (roots are program constants) ----
        let mut prep_values: Vec<Vec<Cell>> = Vec::new();
        for (slot, &(h, w)) in e.shape.prep.tables.iter().zip(e.shape.prep.dims.iter()) {
            let cells = prep_cells[*slot]
                .as_ref()
                .expect("a preprocessed chip has root cells");
            let values = hint_run(b, a.openings, &mut cursor, 2 * w);
            let siblings = hint_digests(b, a.openings, &mut cursor, h - 1);
            let tbits = reduce_iota_bits(bits, h_max_fri, h);
            super::sub_proof::emit_group_authentication(
                b,
                &GroupCommitment::from_lanes(
                    cells.lanes,
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
                .map(|&(_, w)| hint_run(b, a.openings, &mut cursor, 2 * w))
                .collect();
            let siblings = hint_digests(b, a.openings, &mut cursor, h_round - 1);
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
                    let c = b.hint_word(a.fri_legs, fri_cursor);
                    fri_cursor += 1;
                    c.as_ext()
                };
                let siblings = hint_digests(b, a.fri_legs, &mut fri_cursor, h_max_fri - i - 2);
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
            let d = [b.hint_word(arena, *cursor), b.hint_word(arena, *cursor + 1)];
            *cursor += 2;
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
            num_reg: crate::tables::register::NUM_REGISTER_ADDRESSES as usize,
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

/// The assembled aggregation program over N wraps: N verify legs (arena
/// declaration order = absorb order, leg by leg), the chain bindings, and
/// the aggregate's own publishes — the shared attestation id, the block's
/// register boundary vectors (wrap 0's init, the final wrap's fini), the
/// final wrap's output bytes, and every wrap's carved L2G root halves (the
/// global-side byte-compare material, published so the binding against the
/// global proof can live in-VM or at the consumer without re-plumbing).
pub(super) fn aggregator_program(
    wraps: &[RealBatchedLfm],
    layouts: &[WrapPublicLayout],
    labels: &[u64],
) -> LfmProgram {
    assert!(!wraps.is_empty());
    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let arenas: Vec<LfmLegArenas> = wraps
        .iter()
        .map(|e| declare_lfm_leg_arenas(&mut b, e))
        .collect();
    let legs: Vec<LfmLegCells> = wraps
        .iter()
        .zip(&arenas)
        .map(|(e, a)| emit_lfm_leg(&mut b, e, a))
        .collect();
    for (layout, wrap) in layouts.iter().zip(wraps) {
        layout.assert_covers(wrap);
    }
    emit_wrap_chain_bindings(&mut b, &legs, layouts, labels);

    let first = &legs[0];
    let last = legs.last().expect("nonempty");
    let l_first = &layouts[0];
    let l_last = layouts.last().expect("nonempty");
    for half in 0..2 {
        b.public(first.publics[l_first.id(half)].lanes[0].as_cell());
    }
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
    compile(b.finish())
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
    let a = declare_lfm_leg_arenas(&mut b, &e);
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
fn fixture_wraps() -> (Vec<RealBatchedLfm>, Vec<WrapPublicLayout>, Vec<u64>) {
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
    (wraps, layouts, labels)
}

/// ★ THE AGGREGATE RUNS: every epoch of a batched-carved fixture chain wraps
/// at the aggregation preset, the assembled aggregation program verifies ALL
/// of them in one execution — statements, spines, walks, DEEP, FRI, LogUp
/// closures against the public balances — and the chain bindings hold: one
/// attestation id, register fini→init across every seam, each label at its
/// chain position. Reaching the end IS the check; the published words are
/// then spot-checked against the wraps' own.
#[test]
fn the_assembled_aggregator_runs_on_the_fixture_chain() {
    let (wraps, layouts, labels) = fixture_wraps();
    let program = aggregator_program(&wraps, &layouts, &labels);
    let arenas: Vec<Vec<LfmWord>> = wraps.iter().flat_map(leg_arena_words).collect();
    let exec = execute(&program, &arenas, &TestPermutation).expect("the aggregate must execute");

    // The aggregate's own publishes: id halves, block register boundaries,
    // final output bytes, then every wrap's L2G root halves — compare the
    // roots against the wraps' published words.
    let num_reg = crate::tables::register::NUM_REGISTER_ADDRESSES as usize;
    let l_last = layouts.last().expect("nonempty");
    let root_base = 2 + 2 * num_reg + l_last.out_bytes;
    for (k, wrap) in wraps.iter().enumerate() {
        let layout = &layouts[k];
        for h in 0..8 {
            let got = super::word::word_as_base(&exec.public_words[root_base + 8 * k + h].1)
                .expect("a root half");
            let want = super::word::word_as_base(&wrap.public_words[layout.l2g_half(h)].1)
                .expect("a root half");
            assert_eq!(got, want, "wrap {k} root half {h}");
        }
    }
    println!(
        "★ aggregate over {} wraps: {} instructions, {} published words",
        wraps.len(),
        program.instrs.len(),
        exec.public_words.len()
    );
}

/// The chain bindings DISCRIMINATE: a fini→init mismatch at a seam makes the
/// aggregate unprovable (flip one register half in one wrap's publics arena —
/// the leg's own balance check then pins every downstream use, and the
/// binding compares the flipped cell against the neighbor).
#[test]
fn the_aggregator_rejects_a_broken_register_chain() {
    let (wraps, layouts, labels) = fixture_wraps();
    let program = aggregator_program(&wraps, &layouts, &labels);
    let mut arenas: Vec<Vec<LfmWord>> = wraps.iter().flat_map(leg_arena_words).collect();

    // Arena 0 of wrap 0 is its publics arena (eight halves per word); flip
    // the LOW half of reg_fini[0]'s lane 0 — the value the binding compares
    // against wrap 1's reg_init[0].
    let word_index = layouts[0].reg_fini(0);
    arenas[0][8 * word_index] = base_word(
        super::word::word_as_base(&arenas[0][8 * word_index]).expect("a half") + FE::one(),
    );
    assert!(
        execute(&program, &arenas, &TestPermutation).is_err(),
        "a broken register chain must make the aggregate unprovable"
    );
}

/// And a wrong chain-position label is unprovable — replay protection at the
/// aggregate: the same wraps presented in a swapped order cannot execute.
#[test]
fn the_aggregator_rejects_swapped_wrap_order() {
    let (mut wraps, mut layouts, labels) = fixture_wraps();
    wraps.swap(0, 1);
    layouts.swap(0, 1);
    let program = aggregator_program(&wraps, &layouts, &labels);
    let arenas: Vec<Vec<LfmWord>> = wraps.iter().flat_map(leg_arena_words).collect();
    assert!(
        execute(&program, &arenas, &TestPermutation).is_err(),
        "swapped wraps must fail the label pins (and the register chain)"
    );
}
