//! Reading a proof production accepts into the arenas the recursion's programs
//! verify it from: per sub-proof what the transcript absorbs ([`HostTable`])
//! and what the legs open ([`TableLegs`]), and a child LFM proof as a whole
//! ([`HarvestedChild`]). The block tree's driver (`block_tree`) fills its
//! leaves from the base proof and its nodes from their children with these.
//!
//! Every shape is the AIR's at the proof's trace length; the proof is checked
//! against it, never read into it, and a proof that disagrees is an error. The
//! oracle fields the suites compare against (the challenges production derived,
//! the child's LogUp pair) are read the same way in every build, and read only
//! by the suites.
//!
//! The suites that grew these readers (`epoch_tests`, `epoch_verify_tests`,
//! `per_table_aggregator_tests`) keep their names and their panicking forms
//! through re-exports and shims.

use stark::config::Commitment;
use stark::proof::view::StarkProofView;
use stark::traits::AIR;

use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};

use super::constraints::{Analysis, BoundaryTerm};
use super::epoch::TableChallengeShape;
use super::epoch_verify::TableVerifyShape;
use super::fri::FriShape;
use super::word::{LfmWord, base_word, ext_word};

type Gl = GoldilocksField;
type Ext3 = GoldilocksExtension;

/// Everything one real sub-proof supplies to the replay, plus the challenges
/// production derived from it.
#[derive(Clone)]
pub(crate) struct HostTable {
    pub(crate) shape: TableChallengeShape,
    /// The verifier's HARDCODED precomputed commitment, when the AIR is
    /// preprocessed. A program constant, not arena data: the verifier does not
    /// take this from the proof (`verifier.rs:1187`).
    pub(crate) precomputed_root: Option<Commitment>,
    pub(crate) main_root: Commitment,
    pub(crate) aux_root: Option<Commitment>,
    pub(crate) contribution: Option<FEE>,
    pub(crate) composition_root: Commitment,
    /// Row-major, as `row_major_data` carries it.
    pub(crate) ood_current: Vec<FEE>,
    pub(crate) ood_next: Vec<FEE>,
    pub(crate) parts: Vec<FEE>,
    pub(crate) fri_roots: Vec<Commitment>,
    pub(crate) fri_coeffs: Vec<FEE>,
    pub(crate) nonce: Option<u64>,
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) needs_lookup_challenges: bool,

    // ---- the oracle ----
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) beta: FEE,
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) z: FEE,
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) gamma: FEE,
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) zetas: Vec<FEE>,
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) iotas: Vec<usize>,
}

/// A sub-proof inside a multi-table proof, read into [`HostTable`]: the fork
/// is already positioned (separator, aux root and `L` absorbed), so the oracle
/// comes from `replay_rounds_after_round_1` on THAT transcript.
pub(crate) fn host_table_forked(
    air: &dyn AIR<Field = Gl, FieldExtension = Ext3, PublicInputs = ()>,
    view: StarkProofView<'_, Gl, Ext3, ()>,
    index: usize,
    num_tables: usize,
    fork: &mut crate::hash_pin::BlockTranscript,
    lookup_challenges: &[FEE],
) -> Result<HostTable, String> {
    use crate::hash_pin::BlockVerifier as Verifier;
    use stark::domain::new_verifier_domain;
    use stark::verifier::IsStarkVerifier;

    let trace_length = view.trace_length();
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
    // The shape is the AIR's at this trace length; the proof is checked against
    // it, never read into it.
    let shape = TableChallengeShape::derive(air, index, num_tables, trace_length);
    let proved = (
        view.lde_trace_aux_merkle_root().is_some(),
        view.bus_table_contribution().is_some(),
        (ood_c.width(), ood_c.height()),
        (ood_n.width(), ood_n.height()),
        view.composition_poly_parts_ood_evaluation().len(),
    );
    let derived = (
        shape.has_aux_root,
        shape.has_contribution,
        shape.ood_current_dims,
        shape.ood_next_dims,
        shape.num_parts,
    );
    if proved != derived {
        return Err(format!(
            "table {index}: the proof's aux root, L, OOD blocks and part count are the AIR's: \
             {proved:?} != {derived:?}"
        ));
    }

    Ok(HostTable {
        shape,
        precomputed_root: if air.is_preprocessed() {
            Some(layout_precomputed_commitment(air, view.trace_length())?)
        } else {
            None
        },
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
    })
}

/// Everything the verification legs read about one real sub-proof.
///
/// The split against [`HostTable`] is by CONSUMER, not by convenience: that
/// struct holds what the transcript absorbs, this one holds what the legs open.
/// Nothing appears in both — which is the arena-join obligation showing up in
/// the reader as well as in the emitted program.
pub(crate) struct TableLegs {
    pub(crate) verify: TableVerifyShape,
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) analysis: Analysis,
    /// `[query][group]` — the row pair in leaf order, then the path.
    pub(crate) openings: Vec<Vec<(Vec<LfmWord>, Vec<Commitment>)>>,
    /// `[query][layer]` — `(opened values, path)`: the sibling `pᵢ(−υ^(2ⁱ))`
    /// under `pair`, the whole `2^{d_j}` group under a fold schedule.
    pub(crate) fri_openings: Vec<Vec<(Vec<FEE>, Vec<Commitment>)>>,
    /// Every capped tree's cap, split off query 0's (owner) path, in the caps
    /// arena's order: the committed matrices in group order, then the capped
    /// FRI layers. Empty at the default format.
    pub(crate) caps: Vec<Commitment>,
    /// Production's OWN boundary-constraint list for this AIR, kept so
    /// `the_boundary_terms_are_program_shape` can compare the program-shape
    /// rule against the call rather than against a belief about it.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) production_boundary: Vec<BoundaryTerm>,
    /// `AIR::has_aux_trace`, the rule's input.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) has_aux_trace: bool,
    /// Preprocessed-column count, zero when the AIR is not preprocessed. Which
    /// sub-proofs are preprocessed is what assembly ledger entry 7 is about.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) num_precomputed_cols: usize,
    /// The commitment production absorbs for this table, when preprocessed —
    /// `air.precomputed_commitment()`, taken from the AIR and never from the
    /// proof.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) precomputed_commitment: Option<Commitment>,
}

/// `Err` with `what` unless `got == want`.
fn check_eq<T: PartialEq + std::fmt::Debug>(got: T, want: T, what: &str) -> Result<(), String> {
    if got == want {
        Ok(())
    } else {
        Err(format!("{what}: {got:?} != {want:?}"))
    }
}

/// Read one real sub-proof into the shapes and openings the legs consume.
///
/// Every shape here is derived from the AIR and the proof OPTIONS. The one
/// parameter that is neither is `log2_trace_length` — a table's chunk length is
/// chosen by the prover's row counts — and it is program shape in the assembled
/// verifier for the reason the arena schema makes it one: the program is emitted
/// for a specific epoch shape, and a proof whose trace length disagreed would
/// not match the arenas it declares.
pub(crate) fn build_table_legs(
    air: &dyn AIR<Field = Gl, FieldExtension = Ext3, PublicInputs = ()>,
    view: StarkProofView<'_, Gl, Ext3, ()>,
    rap_challenges: &[FEE],
) -> Result<TableLegs, String> {
    // The shapes, from the AIR and the trace length alone; the proof's blocks
    // are then checked against them, never read into them.
    let trace_length = view.trace_length();
    let (verify, analysis) = TableVerifyShape::derive(air, trace_length)
        .map_err(|e| format!("the legs' shape derives: {e}"))?;
    let (main_width, aux_width) = air.trace_layout();
    let num_precomputed = if air.is_preprocessed() {
        air.num_precomputed_columns()
    } else {
        0
    };
    let log2_trace_length = trace_length.trailing_zeros();
    check_eq(
        view.composition_poly_parts_ood_evaluation().len(),
        verify.quotient.num_composition_parts,
        "the proof's part count is the AIR's degree bound",
    )?;
    // The grid the machine rebuilds and the blocks the proof carries must
    // describe one table. Checked rather than assumed because the machine's
    // reconstruction is indexed by the SHAPE and filled from the BLOCKS: a width
    // disagreement would silently scatter the next-row values into wrong columns.
    let deep = &verify.sub.deep;
    let ood_c = view.trace_ood_evaluations();
    let ood_n = view.trace_ood_next_evaluations();
    check_eq(
        ood_c.width(),
        deep.num_total_cols,
        "the current-row OOD block is the full trace width",
    )?;
    check_eq(
        ood_c.height(),
        deep.step_size,
        "the current-row block's height IS step_size (ood.rs:110-114)",
    )?;
    check_eq(
        ood_n.width(),
        deep.next_row_cols.len(),
        "the next-row block is as wide as the transition window",
    )?;
    check_eq(
        ood_n.height(),
        deep.num_eval_points - deep.step_size,
        "the next-row block covers every evaluation point past the first step",
    )?;

    // ---- the owner split: query 0 of a capped tree carries the cap at the end
    // of its path; the arenas take the `D − c` siblings, the caps arena the cap.
    let mut trace_caps: Vec<Vec<Commitment>> = Vec::new();
    let mut split = |q: usize,
                     path: &[Commitment],
                     depth: usize,
                     c: usize|
     -> Result<Vec<Commitment>, String> {
        if c == 0 || q != 0 {
            check_eq(
                path.len(),
                depth - c,
                &format!("query {q}: a path to the cap"),
            )?;
            return Ok(path.to_vec());
        }
        let (siblings, cap) = crypto::merkle_tree::cap::split_owner_path(path, depth, c)
            .ok_or("the owner path is D − c + 2^c long")?;
        trace_caps.push(cap.to_vec());
        Ok(siblings.to_vec())
    };
    let (depth, c_trace) = (verify.sub.merkle_depth, verify.sub.trace_cap);

    // ---- the openings, per query, in the emitter's group order.
    let mut openings = Vec::with_capacity(view.deep_poly_openings_len());
    for q in 0..view.deep_poly_openings_len() {
        let o = view.deep_poly_opening(q);
        let mut groups: Vec<(Vec<LfmWord>, Vec<Commitment>)> = Vec::new();
        if num_precomputed > 0 {
            let p = o
                .precomputed_trace_polys()
                .ok_or("a preprocessed air opens its precomputed columns")?;
            groups.push((
                p.evaluations()
                    .iter()
                    .chain(p.evaluations_sym())
                    .map(|v| base_word(*v))
                    .collect(),
                split(q, p.merkle_path(), depth, c_trace)?,
            ));
        }
        let m = o.main_trace_polys();
        groups.push((
            m.evaluations()
                .iter()
                .chain(m.evaluations_sym())
                .map(|v| base_word(*v))
                .collect(),
            split(q, m.merkle_path(), depth, c_trace)?,
        ));
        if aux_width > 0 {
            let a = o.aux_trace_polys().ok_or("an aux opening")?;
            groups.push((
                a.evaluations()
                    .iter()
                    .chain(a.evaluations_sym())
                    .map(ext_word)
                    .collect(),
                split(q, a.merkle_path(), depth, c_trace)?,
            ));
        }
        let c = o.composition_poly();
        groups.push((
            c.evaluations()
                .iter()
                .chain(c.evaluations_sym())
                .map(ext_word)
                .collect(),
            split(q, c.merkle_path(), depth, c_trace)?,
        ));
        openings.push(groups);
    }

    let (fri_openings, fri_caps) = fri_layer_openings(view, verify.fri)?;

    // Production's own boundary list, for the premise check only. It takes the
    // bus public inputs, which are PROOF data — which is exactly why the emitted
    // program must not be built from this call.
    let bus_public_inputs = view
        .bus_table_contribution()
        .map(stark::lookup::BusPublicInputs::from_contribution);
    let generator = <Gl as math::field::traits::IsFFTField>::get_primitive_root_of_unity(
        log2_trace_length as u64,
    )
    .map_err(|_| "a power-of-two trace length has a root of unity")?;
    let production_boundary = air
        .boundary_constraints(
            &(),
            rap_challenges,
            bus_public_inputs.as_ref(),
            trace_length,
        )
        .constraints
        .iter()
        .map(|c| BoundaryTerm {
            col: if c.is_aux { main_width + c.col } else { c.col },
            point: generator.pow(c.step as u64),
            value: c.value,
        })
        .collect();

    let caps: Vec<Commitment> = trace_caps.into_iter().flatten().chain(fri_caps).collect();
    check_eq(
        caps.len() * super::proof_arena::words_per_root(),
        verify.cap_words(super::proof_arena::words_per_root()),
        "every capped tree's cap, and nothing else",
    )?;

    Ok(TableLegs {
        verify,
        analysis,
        openings,
        fri_openings,
        caps,
        production_boundary,
        has_aux_trace: air.has_aux_trace(),
        num_precomputed_cols: num_precomputed,
        precomputed_commitment: if air.is_preprocessed() {
            Some(layout_precomputed_commitment(air, trace_length)?)
        } else {
            None
        },
    })
}

/// [`super::epoch_verify::layout_precomputed_commitment`], where a layout
/// with no root is an error. At row pairs it IS
/// `air.precomputed_commitment()`.
pub(crate) fn layout_precomputed_commitment<PI>(
    air: &dyn AIR<Field = Gl, FieldExtension = Ext3, PublicInputs = PI>,
    trace_length: usize,
) -> Result<Commitment, String> {
    super::epoch_verify::layout_precomputed_commitment(air, trace_length)
        .ok_or_else(|| format!("no precomputed commitment at {trace_length} rows"))
}

/// One query's FRI layer openings, per layer `(opened values, path)`.
type FriOpenings = Vec<Vec<(Vec<FEE>, Vec<Commitment>)>>;

/// Every query's FRI layer openings, per layer `(opened values, path)`, and
/// the capped layers' caps (layer order) split off query 0's owner paths.
///
/// The proof's flat `layers_evaluations_sym` is one sibling per layer under
/// `pair` and every layer's full group (`2^{d_j}` values, position order)
/// under a fold schedule; `FriShape::layer_values` says which.
/// Each path is cut at its layer's cap: query 0 of a capped layer carries
/// `D − c + 2^c` nodes, every other query `D − c`.
pub(crate) fn fri_layer_openings<PI>(
    view: StarkProofView<'_, Gl, Ext3, PI>,
    fri: FriShape,
) -> Result<(FriOpenings, Vec<Commitment>), String>
where
    PI: rkyv::Archive,
    <PI as rkyv::Archive>::Archived: rkyv::Deserialize<PI, stark::proof::view::PiDeserializer>,
{
    let mut caps: Vec<Commitment> = Vec::new();
    let mut openings = Vec::with_capacity(view.query_list_len());
    for q in 0..view.query_list_len() {
        let d = view.query(q);
        let flat = d.layers_evaluations_sym();
        let per_query: usize = (0..fri.num_committed()).map(|j| fri.layer_values(j)).sum();
        check_eq(
            flat.len(),
            per_query,
            &format!("query {q}: the opened values per query"),
        )?;
        let mut offset = 0usize;
        let mut layers = Vec::with_capacity(fri.num_committed());
        for i in 0..fri.num_committed() {
            let values = flat[offset..offset + fri.layer_values(i)].to_vec();
            offset += fri.layer_values(i);
            let path = d.layer_auth_path(i);
            let (depth, c) = (fri.layer_depth(i), fri.layer_cap(i));
            if c == 0 || q != 0 {
                check_eq(path.len(), depth - c, &format!("query {q} FRI layer {i}"))?;
                layers.push((values, path.to_vec()));
                continue;
            }
            let (siblings, cap) = crypto::merkle_tree::cap::split_owner_path(path, depth, c)
                .ok_or("the owner path is D − c + 2^c long")?;
            caps.extend_from_slice(cap);
            layers.push((values, siblings.to_vec()));
        }
        openings.push(layers);
    }
    Ok((openings, caps))
}

impl TableLegs {
    /// Per query, per group: the row-pair values then the sibling digests.
    ///
    /// NO index word, which is the whole difference from
    /// `join_tests::HostSubProof::query_arena`: the assembled verifier's index is
    /// the transcript's own bits, so an arena that carried one would be offering
    /// the prover a second index.
    pub(crate) fn try_opening_arena(&self) -> Result<Vec<LfmWord>, String> {
        let mut out = Vec::new();
        for query in &self.openings {
            for (values, siblings) in query {
                out.extend(values.iter().copied());
                out.extend(super::proof_arena::commitments_to_arena(siblings));
            }
        }
        check_eq(
            out.len(),
            self.verify
                .opening_words(super::proof_arena::words_per_root()),
            "the opening arena must fill exactly what the shape declares",
        )?;
        Ok(out)
    }

    /// The sub-proof's Merkle caps, once — `None` at the default format, where
    /// the emitter declares no caps arena
    /// (`epoch_verify::declare_table_arenas`).
    pub(crate) fn try_caps_arena(&self) -> Result<Option<Vec<LfmWord>>, String> {
        let words = self.verify.cap_words(super::proof_arena::words_per_root());
        if words == 0 {
            check_eq(self.caps.len(), 0, "no caps at a format without caps")?;
            return Ok(None);
        }
        let out = super::proof_arena::commitments_to_arena(&self.caps);
        check_eq(
            out.len(),
            words,
            "the caps arena is what the shape declares",
        )?;
        Ok(Some(out))
    }

    /// Per query, per committed layer: the symmetric evaluation then its path.
    pub(crate) fn try_fri_arena(&self) -> Result<Vec<LfmWord>, String> {
        let mut out = Vec::new();
        for query in &self.fri_openings {
            for (values, path) in query {
                out.extend(values.iter().map(ext_word));
                out.extend(super::proof_arena::commitments_to_arena(path));
            }
        }
        check_eq(
            out.len(),
            self.verify.fri_words(super::proof_arena::words_per_root()),
            "the FRI arena must fill exactly what the shape declares",
        )?;
        Ok(out)
    }

    /// [`Self::try_opening_arena`], panicking: the suites' form.
    #[cfg(test)]
    pub(crate) fn opening_arena(&self) -> Vec<LfmWord> {
        self.try_opening_arena().unwrap_or_else(|e| panic!("{e}"))
    }

    /// [`Self::try_caps_arena`], panicking: the suites' form.
    #[cfg(test)]
    pub(crate) fn caps_arena(&self) -> Option<Vec<LfmWord>> {
        self.try_caps_arena().unwrap_or_else(|e| panic!("{e}"))
    }

    /// [`Self::try_fri_arena`], panicking: the suites' form.
    #[cfg(test)]
    pub(crate) fn fri_arena(&self) -> Vec<LfmWord> {
        self.try_fri_arena().unwrap_or_else(|e| panic!("{e}"))
    }
}

/// One child LFM proof, production-accepted, harvested for emission.
///
/// The per-table sibling of `epoch_tests::RealEpoch`, over an LFM machine's
/// proof rather than the VM's. ★ Nothing in the harvest is LFM-specific:
/// [`host_table_forked`] and [`build_table_legs`] take `(&dyn AIR,
/// StarkProofView)`, so the same two functions read a wrap proof, a node proof
/// and a VM epoch proof alike. That is what makes one node emitter serve every
/// level.
pub(crate) struct HarvestedChild {
    pub(crate) artifacts: super::registry::LfmArtifacts,
    pub(crate) opts: crate::ProofOptions,
    pub(crate) public_words: Vec<(u32, LfmWord)>,
    pub(crate) tables: Vec<HostTable>,
    pub(crate) legs: Vec<TableLegs>,
    /// The child's OWN shared LogUp pair, recovered host-side by
    /// `verify_against_chunked`'s own Phase A replay — the oracle the node's leg
    /// must reproduce in-machine. Consumed by
    /// `the_leaf_node_verifies_and_binds_two_wraps` through
    /// `NodePublishSet::Diagnostic`.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) z_alpha: (FEE, FEE),
}

/// [`harvest_child`], after verifying the proof against its artifacts, and the
/// seconds that verify cost: a child read from a proof production would reject
/// describes nothing, so a refusal is an error.
pub(crate) fn harvest_child_verified(
    artifacts: super::registry::LfmArtifacts,
    opts: crate::ProofOptions,
    proved: &super::proof::LfmProof,
) -> Result<(HarvestedChild, f64), String> {
    let t_verify = std::time::Instant::now();
    if !super::proof::verify_against_artifacts(
        &artifacts,
        &proved.proof,
        &proved.public_words,
        &opts,
    ) {
        return Err("the child proof does not verify against its artifacts".to_string());
    }
    let verify_secs = t_verify.elapsed().as_secs_f64();
    Ok((harvest_child(artifacts, opts, proved)?, verify_secs))
}

/// A child proof read without its verify, for a caller that verifies the same
/// proof elsewhere and fails the run on a refusal before it reports anything
/// (the block tree's children, verified beside the timed path).
pub(crate) fn harvest_child(
    artifacts: super::registry::LfmArtifacts,
    opts: crate::ProofOptions,
    proved: &super::proof::LfmProof,
) -> Result<HarvestedChild, String> {
    use crypto::fiat_shamir::is_transcript::IsTranscript;
    use stark::proof::view::MultiProofView;

    // The AIR set the artifacts describe — `KECCAK_RND`/`LFM_BLAKE3` chunks, the
    // `LFM_HASH` chunks and (S2) the one-row preprocessed roots, exactly
    // as `verify_against_artifacts` builds it: a one-row chip's Phase A root and
    // leg compare use them.
    let airs = super::airs::LfmAirs::for_artifacts(&artifacts, &opts);
    let refs = airs.air_refs();
    let view = MultiProofView::Owned(&proved.proof);
    check_eq(refs.len(), view.len(), "one AIR per sub-proof")?;

    // The seed IS `verify_against_chunked`'s: the LFM statement over the claimed
    // words, and nothing before it.
    let seed = || {
        let mut t = crate::hash_pin::block_transcript(&[]);
        super::statement::absorb_lfm_statement(
            &mut t,
            &artifacts.program_id,
            &proved.public_words,
            opts.fri_final_poly_log_degree,
        );
        t
    };

    let mut transcript = seed();
    for (idx, air) in refs.iter().enumerate() {
        let v = view.get(idx);
        if air.is_preprocessed() {
            transcript.append_bytes(&layout_precomputed_commitment(*air, v.trace_length())?);
        }
        transcript.append_bytes(v.lde_trace_main_merkle_root());
    }
    let lookup: Vec<FEE> = (0..stark::lookup::LOGUP_NUM_CHALLENGES)
        .map(|_| transcript.sample_field_element())
        .collect();

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
            host_table_forked(*air, v, idx, num_tables, &mut fork, &lookup)
        })
        .collect::<Result<Vec<_>, String>>()?;
    let legs = refs
        .iter()
        .enumerate()
        .map(|(idx, air)| build_table_legs(*air, view.get(idx), &lookup))
        .collect::<Result<Vec<_>, String>>()?;

    Ok(HarvestedChild {
        artifacts,
        opts,
        public_words: proved.public_words.clone(),
        tables,
        legs,
        z_alpha: (lookup[0], lookup[1]),
    })
}

/// The child's arenas, in `declare_leg_arenas`' declaration order.
pub(crate) fn try_child_arena_words(c: &HarvestedChild) -> Result<Vec<Vec<LfmWord>>, String> {
    let mut arenas: Vec<Vec<LfmWord>> = Vec::new();
    // The published words, eight halves each — the statement's own layout.
    let mut publics = Vec::with_capacity(8 * c.public_words.len());
    for (_, word) in &c.public_words {
        for lane in word {
            let v: u64 = lane.canonical();
            publics.push(base_word(FE::from(v & 0xFFFF_FFFF)));
            publics.push(base_word(FE::from(v >> 32)));
        }
    }
    arenas.push(publics);
    arenas.push(super::proof_arena::commitments_to_arena(
        &c.tables.iter().map(|h| h.main_root).collect::<Vec<_>>(),
    ));
    for (h, leg) in c.tables.iter().zip(&c.legs) {
        if let Some(root) = &h.aux_root {
            arenas.push(super::proof_arena::commitments_to_arena(&[*root]));
        }
        if let Some(l) = &h.contribution {
            arenas.push(vec![ext_word(l)]);
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
        arenas.push(leg.try_opening_arena()?);
        arenas.push(leg.try_fri_arena()?);
        arenas.extend(leg.try_caps_arena()?);
    }
    Ok(arenas)
}
