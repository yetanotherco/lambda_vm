//! ★ The assembled epoch verifier — spine plus legs — run on a real
//! continuation epoch proof.
//!
//! [`super::epoch_tests`] built the Fiat-Shamir spine and checked all 79 of a
//! real 16-sub-proof epoch's challenges against production's own replay. Every
//! verification leg, meanwhile, was driven by its own isolation program with
//! HINTED challenges. This module hangs the legs off the spine: per sub-proof the
//! OOD grid is rebuilt from the two pruned blocks the transcript absorbed, the
//! constraint evaluation and quotient check run at the spine's `z` and `β`, and
//! each query's index bits go straight from `TableChallenges::iota_bits` into the
//! Merkle walk, the DEEP fold and the FRI chain.
//!
//! ## The oracle, and what is left of it
//!
//! There is deliberately LESS oracle here than in any leg suite, and that is the
//! point. A leg suite checks a computed value against production's own answer for
//! the same inputs. Here the checks are INSIDE the program: the quotient check is
//! `assert_eq_ext(claimed, composition)`, every Merkle walk ends in
//! `assert_word_eq_lanes` against a root the transcript absorbed, and the FRI
//! chain ends in `assert_eq_ext` against the terminal polynomial. A program that
//! executes at all has passed them. So the differential that remains is the
//! spine's — the 79 challenges, still checked — plus the fact of execution, and
//! the falsification tests below are what turn "it executed" into evidence, by
//! showing what does NOT execute.
//!
//! ## What this suite cannot see
//!
//! The preset. The fixture epoch is proved at the MIN preset (blowup 2, one
//! query per table, grinding factor 1), because that is what
//! `proof_fixture::fixture_options` gives and what keeps a 16-sub-proof epoch
//! provable in a unit test. Every per-query cost here is therefore ONE query's,
//! and the blowup-8 predictions the phase pinned (73 queries, 14,454 FRI
//! permutations per sub-proof) are reached by scaling, not by measurement — the
//! scaling factors are stated in [`the_assembled_epoch_verifier_runs`]'s output
//! rather than hidden in a comment. It also cannot see PAGE's preprocessed
//! commitment problem (ledger entry 7), which is about where a root COMES from
//! and not about what is done with it.

use stark::config::Commitment;
use stark::constraint_ir::ConstraintArtifact;
use stark::proof::view::StarkProofView;
use stark::traits::AIR;
use stark::verifier::{IsStarkVerifier, Verifier};

use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};

use super::constraints::{Analysis, BoundaryTerm, QuotientShape, analyze};
use super::deep::DeepShape;
use super::epoch_verify::{TableVerifyShape, boundary_terms};
use super::executor::execute;
use super::fri::FriShape;
use super::sub_proof::{GroupShape, SubProofShape};
use super::word::{LfmWord, base_word, ext_word, word_as_ext};

type Gl = GoldilocksField;
type Ext3 = GoldilocksExtension;
type V = Verifier<Gl, Ext3, ()>;

/// Everything the verification legs read about one real sub-proof.
///
/// The split against `epoch_tests::HostTable` is by CONSUMER, not by
/// convenience: that struct holds what the transcript absorbs, this one holds
/// what the legs open. Nothing appears in both — which is the arena-join
/// obligation showing up in the test fixture as well as in the emitted program.
pub(super) struct TableLegs {
    pub(super) verify: TableVerifyShape,
    pub(super) analysis: Analysis,
    /// `[query][group]` — the row pair in leaf order, then the path.
    openings: Vec<Vec<(Vec<LfmWord>, Vec<Commitment>)>>,
    /// `[query][layer]` — `(opened values, path)`: the sibling `pᵢ(−υ^(2ⁱ))`
    /// under `pair`, the whole `2^{d_j}` group under a fold schedule.
    fri_openings: Vec<Vec<(Vec<FEE>, Vec<Commitment>)>>,
    /// Every capped tree's cap, split off query 0's (owner) path, in the caps
    /// arena's order: the committed matrices in group order, then the capped
    /// FRI layers. Empty at the default format.
    caps: Vec<Commitment>,
    /// Production's OWN boundary-constraint list for this AIR, kept so
    /// [`the_boundary_terms_are_program_shape`] can compare the program-shape
    /// rule against the call rather than against a belief about it.
    production_boundary: Vec<BoundaryTerm>,
    /// `AIR::has_aux_trace`, the rule's input.
    has_aux_trace: bool,
    /// Preprocessed-column count, zero when the AIR is not preprocessed. Which
    /// sub-proofs are preprocessed is what assembly ledger entry 7 is about.
    pub(super) num_precomputed_cols: usize,
    /// The commitment production absorbs for this table, when preprocessed —
    /// `air.precomputed_commitment()`, taken from the AIR and never from the
    /// proof.
    pub(super) precomputed_commitment: Option<Commitment>,
}

/// Read one real sub-proof into the shapes and openings the legs consume.
///
/// Every shape here is derived from the AIR and the proof OPTIONS. The one
/// parameter that is neither is `log2_trace_length` — a table's chunk length is
/// chosen by the prover's row counts — and it is program shape in the assembled
/// verifier for the reason the arena schema makes it one: the program is emitted
/// for a specific epoch shape, and a proof whose trace length disagreed would
/// not match the arenas it declares.
pub(super) fn build_table_legs(
    air: &dyn AIR<Field = Gl, FieldExtension = Ext3, PublicInputs = ()>,
    view: StarkProofView<'_, Gl, Ext3, ()>,
    rap_challenges: &[FEE],
) -> TableLegs {
    let opts = air.options();
    let layout = V::ood_layout(air);
    let artifact = ConstraintArtifact::capture(air);

    let (main_width, aux_width) = air.trace_layout();
    let num_total_cols = main_width + aux_width;
    let num_precomputed = if air.is_preprocessed() {
        air.num_precomputed_columns()
    } else {
        0
    };

    let trace_length = view.trace_length();
    let log2_trace_length = trace_length.trailing_zeros();
    let log2_blowup = (opts.blowup_factor as usize).trailing_zeros();
    let log2_lde_length = log2_trace_length + log2_blowup;
    let claimed_parts = view.composition_poly_parts_ood_evaluation();

    // The trace matrices in DEEP column order — precomputed, main, aux — as the
    // proof carries them and `build_host_sub_proof` reads them.
    let mut trace_groups = Vec::new();
    if num_precomputed > 0 {
        trace_groups.push(GroupShape {
            num_columns: num_precomputed,
            is_ext: false,
        });
    }
    trace_groups.push(GroupShape {
        num_columns: main_width - num_precomputed,
        is_ext: false,
    });
    if aux_width > 0 {
        trace_groups.push(GroupShape {
            num_columns: aux_width,
            is_ext: true,
        });
    }

    let deep = DeepShape {
        step_size: layout.step_size(),
        num_eval_points: artifact.shape.transition_offsets.len() * layout.step_size(),
        num_total_cols,
        next_row_cols: layout.next_row_cols().to_vec(),
        num_composition_parts: claimed_parts.len(),
        log2_trace_length,
    };
    // The grid the machine rebuilds and the blocks the proof carries must
    // describe one table. Asserted rather than assumed because the machine's
    // reconstruction is indexed by the SHAPE and filled from the BLOCKS: a width
    // disagreement would silently scatter the next-row values into wrong columns.
    let ood_c = view.trace_ood_evaluations();
    let ood_n = view.trace_ood_next_evaluations();
    assert_eq!(
        ood_c.width(),
        num_total_cols,
        "the current-row OOD block is the full trace width"
    );
    assert_eq!(
        ood_c.height(),
        deep.step_size,
        "the current-row block's height IS step_size (ood.rs:110-114)"
    );
    assert_eq!(
        ood_n.width(),
        deep.next_row_cols.len(),
        "the next-row block is as wide as the transition window"
    );
    assert_eq!(
        ood_n.height(),
        deep.num_eval_points - deep.step_size,
        "the next-row block covers every evaluation point past the first step"
    );

    // The table's leaf layout (S2): the host prover's and verifier's own
    // per-table resolution, so `auto` mixes layouts across a proof's tables.
    let leaf_layout = stark::leaf_layout::table_leaf_layout(air, trace_length);
    let merkle_depth = leaf_layout.tree_depth(log2_lde_length as usize);
    let sub = SubProofShape {
        deep,
        trace_groups,
        merkle_depth,
        log2_lde_length,
        coset_offset: FE::from(opts.coset_offset),
        trace_cap: opts
            .format
            .merkle_cap
            .height(opts.fri_number_of_queries, merkle_depth),
        layout: leaf_layout,
    };
    let has_aux_trace = air.has_aux_trace();
    let verify = TableVerifyShape {
        quotient: QuotientShape {
            log2_trace_length,
            num_composition_parts: claimed_parts.len(),
            boundary: boundary_terms(has_aux_trace, num_total_cols),
        },
        fri: FriShape::for_layout(opts, log2_lde_length, leaf_layout),
        main_width,
        num_alpha_powers: if has_aux_trace {
            artifact.shape.max_bus_elements as usize
        } else {
            0
        },
        num_queries: opts.fri_number_of_queries,
        sub,
    };

    // ---- the cap heights: the in-guest shapes' against the host's own
    // `StarkCaps` (the prover's and the verifier's), so the two sides derive
    // every tree's height and depth from one function.
    let host_caps = stark::merkle_caps::StarkCaps::for_options(
        opts,
        log2_lde_length as usize,
        leaf_layout.is_one_row(),
    )
    .expect("a format the host lays out");
    assert_eq!(host_caps.trace_depth, verify.sub.merkle_depth);
    assert_eq!(
        host_caps.trace, verify.sub.trace_cap,
        "the trace trees' cap"
    );
    assert_eq!(host_caps.fri.len(), verify.fri.num_committed());
    for (i, (&d, &c)) in host_caps.fri_depths.iter().zip(&host_caps.fri).enumerate() {
        assert_eq!(d, verify.fri.layer_depth(i), "FRI layer {i}'s tree depth");
        assert_eq!(c, verify.fri.layer_cap(i), "FRI layer {i}'s cap");
    }

    // ---- the owner split: query 0 of a capped tree carries the cap at the end
    // of its path; the arenas take the `D − c` siblings, the caps arena the cap.
    let mut trace_caps: Vec<Vec<Commitment>> = Vec::new();
    let mut split = |q: usize, path: &[Commitment], depth: usize, c: usize| -> Vec<Commitment> {
        if c == 0 || q != 0 {
            assert_eq!(path.len(), depth - c, "query {q}: a path to the cap");
            return path.to_vec();
        }
        let (siblings, cap) = crypto::merkle_tree::cap::split_owner_path(path, depth, c)
            .expect("the owner path is D − c + 2^c long");
        trace_caps.push(cap.to_vec());
        siblings.to_vec()
    };
    let (depth, c_trace) = (verify.sub.merkle_depth, verify.sub.trace_cap);

    // ---- the openings, per query, in the emitter's group order.
    let openings = (0..view.deep_poly_openings_len())
        .map(|q| {
            let o = view.deep_poly_opening(q);
            let mut groups: Vec<(Vec<LfmWord>, Vec<Commitment>)> = Vec::new();
            if num_precomputed > 0 {
                let p = o
                    .precomputed_trace_polys()
                    .expect("a preprocessed air opens its precomputed columns");
                groups.push((
                    p.evaluations()
                        .iter()
                        .chain(p.evaluations_sym())
                        .map(|v| base_word(*v))
                        .collect(),
                    split(q, p.merkle_path(), depth, c_trace),
                ));
            }
            let m = o.main_trace_polys();
            groups.push((
                m.evaluations()
                    .iter()
                    .chain(m.evaluations_sym())
                    .map(|v| base_word(*v))
                    .collect(),
                split(q, m.merkle_path(), depth, c_trace),
            ));
            if aux_width > 0 {
                let a = o.aux_trace_polys().expect("an aux opening");
                groups.push((
                    a.evaluations()
                        .iter()
                        .chain(a.evaluations_sym())
                        .map(ext_word)
                        .collect(),
                    split(q, a.merkle_path(), depth, c_trace),
                ));
            }
            let c = o.composition_poly();
            groups.push((
                c.evaluations()
                    .iter()
                    .chain(c.evaluations_sym())
                    .map(ext_word)
                    .collect(),
                split(q, c.merkle_path(), depth, c_trace),
            ));
            groups
        })
        .collect();

    let (fri_openings, fri_caps) = fri_layer_openings(view, verify.fri);

    // Production's own boundary list, for the premise check only. It takes the
    // bus public inputs, which are PROOF data — which is exactly why the emitted
    // program must not be built from this call.
    let bus_public_inputs = view
        .bus_table_contribution()
        .map(stark::lookup::BusPublicInputs::from_contribution);
    let generator = <Gl as math::field::traits::IsFFTField>::get_primitive_root_of_unity(
        log2_trace_length as u64,
    )
    .expect("a power-of-two trace length has a root of unity");
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
    assert_eq!(
        caps.len() * super::proof_arena::words_per_root(),
        verify.cap_words(super::proof_arena::words_per_root()),
        "every capped tree's cap, and nothing else"
    );

    TableLegs {
        verify,
        analysis: analyze(&artifact),
        openings,
        fri_openings,
        caps,
        production_boundary,
        has_aux_trace,
        num_precomputed_cols: num_precomputed,
        precomputed_commitment: air
            .is_preprocessed()
            .then(|| layout_precomputed_commitment(air, trace_length)),
    }
}

/// The precomputed-columns commitment the host verifier takes for `air` over
/// a trace of `trace_length` rows: `precomputed_commitment_for` the table's
/// resolved leaf layout (S2 — a layout with no root is a hard
/// error, never the other layout's root). At row pairs it IS
/// `air.precomputed_commitment()`.
pub(super) fn layout_precomputed_commitment<PI>(
    air: &dyn AIR<Field = Gl, FieldExtension = Ext3, PublicInputs = PI>,
    trace_length: usize,
) -> Commitment {
    let layout = stark::leaf_layout::table_leaf_layout(air, trace_length);
    air.precomputed_commitment_for(layout)
        .unwrap_or_else(|| panic!("no precomputed commitment at {layout:?}"))
}

/// Every query's FRI layer openings, per layer `(opened values, path)`, and
/// the capped layers' caps (layer order) split off query 0's owner paths.
///
/// The proof's flat `layers_evaluations_sym` is one sibling per layer under
/// `pair` and every layer's full group (`2^{d_j}` values, position order)
/// under a fold schedule; `FriShape::layer_values` says which.
/// Each path is cut at its layer's cap: query 0 of a capped layer carries
/// `D − c + 2^c` nodes, every other query `D − c`.
#[allow(clippy::type_complexity)]
pub(super) fn fri_layer_openings<PI>(
    view: StarkProofView<'_, Gl, Ext3, PI>,
    fri: FriShape,
) -> (Vec<Vec<(Vec<FEE>, Vec<Commitment>)>>, Vec<Commitment>)
where
    PI: rkyv::Archive,
    <PI as rkyv::Archive>::Archived: rkyv::Deserialize<PI, stark::proof::view::PiDeserializer>,
{
    let mut caps: Vec<Commitment> = Vec::new();
    let openings = (0..view.query_list_len())
        .map(|q| {
            let d = view.query(q);
            let flat = d.layers_evaluations_sym();
            let per_query: usize = (0..fri.num_committed()).map(|j| fri.layer_values(j)).sum();
            assert_eq!(
                flat.len(),
                per_query,
                "query {q}: the opened values per query"
            );
            let mut offset = 0usize;
            (0..fri.num_committed())
                .map(|i| {
                    let values = flat[offset..offset + fri.layer_values(i)].to_vec();
                    offset += fri.layer_values(i);
                    let path = d.layer_auth_path(i);
                    let (depth, c) = (fri.layer_depth(i), fri.layer_cap(i));
                    if c == 0 || q != 0 {
                        assert_eq!(path.len(), depth - c, "query {q} FRI layer {i}");
                        return (values, path.to_vec());
                    }
                    let (siblings, cap) =
                        crypto::merkle_tree::cap::split_owner_path(path, depth, c)
                            .expect("the owner path is D − c + 2^c long");
                    caps.extend_from_slice(cap);
                    (values, siblings.to_vec())
                })
                .collect()
        })
        .collect();
    (openings, caps)
}

impl TableLegs {
    /// Per query, per group: the row-pair values then the sibling digests.
    ///
    /// NO index word, which is the whole difference from
    /// `join_tests::HostSubProof::query_arena`: the assembled verifier's index is
    /// the transcript's own bits, so an arena that carried one would be offering
    /// the prover a second index.
    pub(super) fn opening_arena(&self) -> Vec<LfmWord> {
        let mut out = Vec::new();
        for query in &self.openings {
            for (values, siblings) in query {
                out.extend(values.iter().copied());
                out.extend(super::proof_arena::commitments_to_arena(siblings));
            }
        }
        assert_eq!(
            out.len(),
            self.verify
                .opening_words(super::proof_arena::words_per_root()),
            "the opening arena must fill exactly what the shape declares"
        );
        out
    }

    /// The sub-proof's Merkle caps, once — `None` at the default format, where
    /// the emitter declares no caps arena
    /// (`epoch_verify::declare_table_arenas`).
    pub(super) fn caps_arena(&self) -> Option<Vec<LfmWord>> {
        let words = self.verify.cap_words(super::proof_arena::words_per_root());
        if words == 0 {
            assert!(self.caps.is_empty());
            return None;
        }
        let out = super::proof_arena::commitments_to_arena(&self.caps);
        assert_eq!(
            out.len(),
            words,
            "the caps arena is what the shape declares"
        );
        Some(out)
    }

    /// Per query, per committed layer: the symmetric evaluation then its path.
    pub(super) fn fri_arena(&self) -> Vec<LfmWord> {
        let mut out = Vec::new();
        for query in &self.fri_openings {
            for (values, path) in query {
                out.extend(values.iter().map(ext_word));
                out.extend(super::proof_arena::commitments_to_arena(path));
            }
        }
        assert_eq!(
            out.len(),
            self.verify.fri_words(super::proof_arena::words_per_root()),
            "the FRI arena must fill exactly what the shape declares"
        );
        out
    }
}

/// ★ THE RUN: the whole epoch verifier — spine AND legs — on a real
/// continuation epoch proof that production accepts.
///
/// What executing proves, stated precisely. Every check is an assert inside the
/// program, so reaching the end means: all 16 quotient identities held at the
/// spine's own `z` and `β`; every one of the 16 sub-proofs' opened row pairs
/// hashed to a leaf that walked to the root the transcript absorbed, at the index
/// the transcript sampled; every DEEP reconstruction fed a FRI chain that folded
/// to the terminal polynomial the transcript absorbed; and the LogUp closure
/// reached production's COMMIT-bus target. The 79 published challenges are
/// checked against production's replay on top, so the Fiat-Shamir the whole thing
/// hangs from is still differentialled.
#[test]
fn the_assembled_epoch_verifier_runs() {
    let e = super::epoch_tests::real_epoch();
    let program = super::epoch_tests::epoch_program(&e, true);
    let arenas = super::epoch_tests::epoch_arena_words(&e, true);
    // ★ The PINNED socket permutation, not a literal. `epoch_program` builds at
    // `WrapHash::production()`, and the classification rule is that such a
    // program emits `Instr::Hash` and must run under `BLOCK_HASHER`; only a
    // program pinning a byte hash on its own builder may take the default.
    //
    // ⚠ Under a BYTE pin this is inert — `ByteWrapHash` lowers to the KECCAK /
    // `LFM_BLAKE3` chips and emits no `Instr::Hash`, so the socket is never
    // consulted and a toy permutation was free and correct. Under an ALGEBRAIC
    // pin the walks ARE `Instr::Hash`: a toy would rebuild roots the host never
    // committed, and this test would fail on its HONEST path, naming nothing.
    let exec = execute(&program, &arenas, &crate::hash_pin::BLOCK_HASHER)
        .expect("the assembled verifier must execute");

    // ---- the spine's differential, unchanged: production's own challenges.
    let pub_ext = |i: usize| word_as_ext(&exec.public_words[i].1).expect("an ext challenge");
    assert_eq!(pub_ext(0), e.z_alpha.0, "the shared LogUp challenge z");
    assert_eq!(pub_ext(1), e.z_alpha.1, "the shared LogUp challenge alpha");

    // The attestation fold is published right after Phase A (two digest words),
    // and its DECODE input is the cell Phase A absorbed — the join ledger entry 7
    // rests on. Its value is differentialled in the spine test; here it only has to
    // be skipped, and skipped by NAME rather than by a literal.
    let program_id_words = 2usize;
    // ★ Then the BLOCK-BINDING SCHEMA — the register boundary vectors, the epoch
    // label, the public-output halves and the L2G re-commit root. Skipped by NAME
    // (`epoch_tests::schema_words`) rather than by a literal, for the same reason
    // the id is: a literal here would start checking `beta of table 0` against a
    // register slot the moment the schema moves, and would report a pass while
    // doing it. Its VALUES are the aggregator's subject and are differentialled
    // there; this gate only has to walk past them and still account for every word.
    let schema_words = super::epoch_tests::schema_words(&e);
    let mut cursor = 2 + program_id_words + schema_words;
    let mut checked = 2usize;
    for (i, (h, leg)) in e.tables.iter().zip(&e.legs).enumerate() {
        // The legs publish first: the recomputed composition, then a terminal
        // value per query.
        cursor += 1 + leg.verify.num_queries;
        assert_eq!(pub_ext(cursor), h.beta, "beta of table {i}");
        assert_eq!(pub_ext(cursor + 1), h.z, "z of table {i}");
        assert_eq!(pub_ext(cursor + 2), h.gamma, "gamma of table {i}");
        cursor += 3;
        checked += 3;
        for (k, want) in h.zetas.iter().enumerate() {
            assert_eq!(pub_ext(cursor + k), *want, "zeta {k} of table {i}");
        }
        cursor += h.zetas.len();
        checked += h.zetas.len();
        for q in 0..h.shape.num_queries {
            let w = exec.public_words[cursor + q].1;
            let got = super::word::word_as_base(&w).expect("an index is a base felt");
            assert_eq!(got, FE::from(h.iotas[q] as u64), "iota {q} of table {i}");
        }
        cursor += h.shape.num_queries;
        checked += h.shape.num_queries;
    }
    // ★ A literal on purpose — deriving the expectation from `e.tables` would
    // restate the loop above and check nothing. What moved it from 111 is
    // named rather than absorbed: an intermediate epoch was 24 sub-proofs when
    // 111 was written, and each always-on RV64 table (the campaign's HINT,
    // then #903's BLAKE3) adds one sub-proof and, at the MIN preset, its four
    // challenges — a (β, z, γ) triple and one query index, with no DEEP zeta
    // because a near-empty fixed table opens nothing. The accounting is
    // asserted, so the next always-on table fails here saying which input
    // moved instead of leaving a bare literal to re-bless.
    //
    // It has now moved the other way, by the same accounting: BLAKE3 stopped
    // being always-on (it is `TableCounts::blake3`, and this epoch does not use
    // it), so 26 → 25 and 119 → 115. This epoch is exactly the workload the
    // bisect priced — a near-empty always-on table is four challenges here and
    // +31.2% of the wrap's cells at the secure preset.
    //
    // ★ And then FURTHER the same way, for the same reason at six times the
    // size.
    //
    // MOVER: `892c7d1bc` (main's `c2ac5d546`, #977) — arm (i), the TABLE SET.
    // That merge's two arms are (i) the table set shrank, empty tables now
    // being elided rather than padded, and (ii) the absorbed epoch statement
    // grew 49 bytes. This pin is on (i); `blake3_chip_tests`'
    // STATEMENT_REPLAY_BLAKE3_ROWS and `machine_tests`' statement byte length
    // are on (ii).
    //
    // #977 did to the six accelerator
    // chips what #903's revert did to BLAKE3: `FIXED_TABLE_COUNT` 11 → 5, with
    // `commit`, `keccak`, `keccak_rnd`, `ecsm`, `ecdas` and `hint` becoming
    // `TableCounts` fields that a run which never reaches them reports as zero.
    // This fixture epoch reaches none of the six, so 25 → 16 and 115 → 79.
    // Six of the nine are those accelerators: the pre-#977 25 was 14 split
    // families + 10 intermediate fixed (11 less HALT) + 1 L2G_MEMORY, and the
    // fixed term is now 4, which lands at 19. The remaining three are split
    // families this fixture leaves empty, and they are NOT named here because
    // the measurement does not name them — only `e.tables.len()` is read.
    //
    // The accounting below is what says the whole move is empty tables at four
    // challenges each: every one of them opens nothing, so it contributes a
    // (β, z, γ) triple and one query index and no DEEP zeta, exactly as an
    // always-on table did. If that model is wrong for any of the nine, the
    // `checked` assertion two below fails and says so.
    //
    // The identity is spelled with both sides positive because `SUB_PROOFS` is
    // now BELOW the 24 it used to be measured against, and `SUB_PROOFS - 24` on
    // a `usize` is an underflow rather than a failed assertion. Moved this way
    // it still fails — in either direction — if the four-challenge model stops
    // describing the move.
    //
    // `LFM_BLAKE3` (P-a Stage 5) does NOT appear in either number: it is a chip
    // of the LFM machine, counted by `NUM_LFM_CHIPS`, and this is the RV64
    // epoch the LFM machine verifies.
    const SUB_PROOFS: usize = 16;
    const CHALLENGES_AT_MIN_PRESET: usize = 79;
    const CHALLENGES_PER_ALWAYS_ON_TABLE: usize = 4;
    assert_eq!(
        e.tables.len(),
        SUB_PROOFS,
        "the epoch's sub-proof count is what the challenge count below is a \
         function of"
    );
    assert_eq!(
        CHALLENGES_AT_MIN_PRESET + CHALLENGES_PER_ALWAYS_ON_TABLE * 24,
        111 + CHALLENGES_PER_ALWAYS_ON_TABLE * SUB_PROOFS,
        "the tables that left the proof account for the whole move from the \
         original 111"
    );
    assert_eq!(
        checked, CHALLENGES_AT_MIN_PRESET,
        "the same challenges the spine test checks must still be checked"
    );
    assert_eq!(
        word_as_ext(&exec.public_words[cursor].1).expect("the bus total is ext"),
        e.expected_bus_balance,
        "the LogUp closure must reach production's own COMMIT-bus target"
    );
    assert_eq!(
        cursor + 1,
        exec.public_words.len(),
        "every published word must be accounted for"
    );

    // ---- THE MEASUREMENT ----
    let spine = super::epoch_tests::epoch_program(&e, false);
    // The CONFIGURED hash's compressions — the closed form counts Merkle
    // levels and query paths, which is hash-independent, so counting keccak
    // specifically read zero under BLAKE3.
    let perms = |p: &_| super::machine_tests::wrap_hash_instrs(p);
    let words = |p: &super::compiler::LfmProgram| -> usize {
        p.arena_schema.lens.iter().map(|l| *l as usize).sum()
    };

    let queries = e.legs[0].verify.num_queries;
    let opening_perms = perms(&program) - perms(&spine);
    let legs_published: usize = e.legs.iter().map(|l| 1 + l.verify.num_queries).sum();
    println!(
        "\n★ ASSEMBLED EPOCH VERIFIER (min preset: blowup 2, {queries} quer\
         {}/table, grinding {}):\n\
         \x20                    spine        +legs         legs alone\n\
         \x20 instructions   {:>10}   {:>10}   {:>10}\n\
         \x20 keccak perms   {:>10}   {:>10}   {:>10}\n\
         \x20 arena words    {:>10}   {:>10}   {:>10}\n\
         \x20 published      {:>10}   {:>10}   {:>10}",
        if queries == 1 { "y" } else { "ies" },
        e.tables[0].shape.grinding_factor,
        spine.instrs.len(),
        program.instrs.len(),
        program.instrs.len() - spine.instrs.len(),
        perms(&spine),
        perms(&program),
        opening_perms,
        words(&spine),
        words(&program),
        words(&program) - words(&spine),
        // The spine's own published count. Was `len - (x - x)` — a leftover that
        // printed the assembled figure in the spine column.
        exec.public_words.len() - legs_published,
        exec.public_words.len(),
        legs_published,
    );

    // ---- the constraint leg's share, from the analyses themselves.
    //
    // `Analysis::report` is the count of what the lowering pass DID, and
    // `emit_analyzed` runs over the very analysis reported here — the module's own
    // doc comment makes that a construction, not a coincidence — so summing the
    // reports attributes the constraint evaluation inside the assembled program
    // without a second emitter pass. `alu_rows` excludes constants because the
    // builder interns them program-wide, so the sum is a lower bound on the
    // constraint leg's instructions and not the whole of it.
    let constraint_alu: usize = e.legs.iter().map(|l| l.analysis.report().alu_rows()).sum();
    let constraint_unfused: usize = e
        .legs
        .iter()
        .map(|l| l.analysis.report().unfused_alu_rows())
        .sum();
    // The recombination half, measured in ISOLATION against its own plumbing
    // baseline and compared against a number that did not come from this emitter
    // (`others/lfm-constraint-lowering-design.md:604` splits the pinned 57,252
    // into 54,358 lowering + 2,894 recombination). That is what makes a
    // two-pass difference admissible here — the comparison target is external.
    let recombination: usize = e
        .legs
        .iter()
        .map(|l| {
            let plumb = |b: &mut super::builder::LfmBuilder| {
                let n = 2
                    + l.verify.sub.deep.num_composition_parts
                    + l.verify.num_frame_steps() * l.verify.sub.deep.num_total_cols
                    + l.analysis.report().nodes;
                let a = b.declare_arena(n as u32);
                let mut i = 0u32;
                let mut take = |b: &mut super::builder::LfmBuilder| {
                    let c = b.hint_word(a, i).as_ext();
                    i += 1;
                    c
                };
                let z = take(b);
                let beta = take(b);
                let parts: Vec<_> = (0..l.verify.sub.deep.num_composition_parts)
                    .map(|_| take(b))
                    .collect();
                let steps: Vec<Vec<_>> = (0..l.verify.num_frame_steps())
                    .map(|_| {
                        (0..l.verify.sub.deep.num_total_cols)
                            .map(|_| take(b))
                            .collect()
                    })
                    .collect();
                // One evaluation cell per constraint root, which is what
                // `emit_analyzed` returns and `emit_quotient` folds.
                let evals: Vec<_> = (0..l.analysis.program().roots.len())
                    .map(|_| take(b))
                    .collect();
                (z, beta, parts, steps, evals)
            };
            let mut bare = super::builder::LfmBuilder::new()
                .with_wrap_hash(super::edsl::WrapHash::production());
            let _ = plumb(&mut bare);
            let baseline = bare.finish().instrs.len();

            let mut full = super::builder::LfmBuilder::new()
                .with_wrap_hash(super::edsl::WrapHash::production());
            let (z, beta, parts, steps, evals) = plumb(&mut full);
            let ood = super::constraints::OodOperands {
                steps,
                main_width: l.verify.main_width,
                rap_challenges: Vec::new(),
                alpha_powers: Vec::new(),
                table_offset: z,
            };
            super::constraints::emit_quotient(
                &mut full,
                &l.verify.quotient,
                &ood,
                z,
                beta,
                &evals,
                &parts,
            );
            full.finish().instrs.len() - baseline
        })
        .sum();
    println!(
        "\x20 constraint leg inside the assembled verifier: {constraint_alu} ALU \
         rows lowering ({constraint_unfused} unfused) + {recombination} \
         recombination = {} over 16 sub-proofs  [pinned: see the run output]\
         \n\x20 that is {:.1}% of the legs' {} instructions",
        constraint_alu + recombination,
        100.0 * (constraint_alu + recombination) as f64
            / (program.instrs.len() - spine.instrs.len()) as f64,
        program.instrs.len() - spine.instrs.len(),
    );

    // ---- the permutation bill, against a CLOSED FORM over the shapes.
    //
    // Not a difference of two emitter passes (which rule 7's refinement rules
    // out) but arithmetic over byte widths: every group's leaf costs the
    // configured hash's block count, every Merkle level is ONE compression
    // under EVERY hash — a byte parent is 64 bytes, inside keccak's rate and
    // exactly one BLAKE3 block, and an algebraic parent's two digest cells fill
    // the rate-8 sponge exactly — and FRI splits the same way.
    // Asserted, not printed, so a leg that silently stopped hashing a group
    // would fail here.
    //
    // The leaf and FRI-leaf halves are the ones that move with the hash —
    // absorption is block-sensitive, compression is not — so they go through
    // `blocks_for` while the walk terms stay plain counts.
    use super::epoch_verify::blocks_for;
    let hash = super::edsl::WrapHash::production();
    let mut fri_perms = 0usize;
    let mut leaf_perms = 0usize;
    let mut walk_perms = 0usize;
    for leg in &e.legs {
        let groups = leg.verify.sub.groups().len();
        let fri_leaves =
            leg.verify.fri.num_committed() * blocks_for(super::epoch_verify::FRI_LEAF_FELTS, hash);
        fri_perms += leg.verify.num_queries * (fri_leaves + leg.verify.fri.path_steps_per_query());
        leaf_perms += leg.verify.num_queries
            * leg
                .verify
                .sub
                .groups()
                .iter()
                .map(|g| blocks_for(super::epoch_verify::group_leaf_felts(g), hash))
                .sum::<usize>();
        walk_perms += leg.verify.num_queries * groups * leg.verify.sub.merkle_depth;
    }
    let predicted: usize = e
        .legs
        .iter()
        .map(|l| {
            super::epoch_verify::query_permutations_for(
                &l.verify,
                super::edsl::WrapHash::production(),
            )
        })
        .sum();
    assert_eq!(
        predicted,
        leaf_perms + walk_perms + fri_perms,
        "the closed form must decompose into exactly its three parts"
    );
    assert_eq!(
        opening_perms, predicted,
        "the emitted permutation count must equal the closed form over the shapes"
    );
    println!(
        "\x20 leg permutations = {leaf_perms} leaves + {walk_perms} Merkle levels \
         + {fri_perms} FRI = {predicted} (closed form) = {opening_perms} (emitted)"
    );
    println!(
        "\x20 FRI layers committed across the epoch: {}  |  widest leaf: {} bytes",
        e.legs
            .iter()
            .map(|l| l.verify.fri.num_committed())
            .sum::<usize>(),
        e.legs
            .iter()
            .flat_map(|l| l.verify.sub.groups())
            .map(|g| g.leaf_bytes())
            .max()
            .expect("the epoch has groups")
    );

    // ---- RECONCILIATION with the phase's pinned blowup-8 predictions.
    //
    // The pinned 213,744 came from `join_tests::join_leg_cost`, whose stated
    // assumptions are: all 28 PRODUCTION AIRs, every trace at a UNIFORM
    // 2^20, blowup 8, 73 queries, and NO FRI (the joined leg has none). The
    // measurement above is: this epoch's 16 sub-proofs, at their REAL trace
    // lengths, blowup 2, one query, FRI included. Three parameters differ, so
    // the two numbers cannot be compared directly — they are projected onto each
    // other one parameter at a time instead, which is also what says which
    // assumption carries the difference.
    let at_blowup_8 = |leg: &TableLegs, uniform_log2_trace: Option<u32>| -> TableVerifyShape {
        let log2_trace = uniform_log2_trace.unwrap_or(leg.verify.sub.deep.log2_trace_length);
        let log2_lde = log2_trace + 3;
        let mut out = leg.verify.clone();
        out.sub.log2_lde_length = log2_lde;
        out.sub.merkle_depth = log2_lde as usize - 1;
        out.sub.deep.log2_trace_length = log2_trace;
        out.quotient.log2_trace_length = log2_trace;
        out.fri = FriShape {
            log2_lde_length: log2_lde,
            blowup_log: 3,
            num_queries: 73,
            ..leg.verify.fri
        };
        out.num_queries = 73;
        out
    };
    let openings_only = |s: &TableVerifyShape| -> usize {
        s.num_queries
            * (super::epoch_verify::leaf_permutations(&s.sub)
                + s.sub.groups().len() * s.sub.merkle_depth)
    };

    let real_lengths: Vec<TableVerifyShape> = e.legs.iter().map(|l| at_blowup_8(l, None)).collect();
    let uniform: Vec<TableVerifyShape> = e.legs.iter().map(|l| at_blowup_8(l, Some(20))).collect();
    let sum = |v: &[TableVerifyShape], f: &dyn Fn(&TableVerifyShape) -> usize| -> usize {
        v.iter().map(f).sum()
    };

    // ---- THE HASH MATRIX'S PERMUTATION AXIS, at the production shape.
    //
    // A candidate hash moves two independent things: cells per permutation (its
    // AIR's shape, which needs the AIR) and permutations per verify (the sponge's
    // rate, which needs only arithmetic over these shapes). This block pins the
    // second WITHOUT any candidate permutation existing, so the remaining unknown
    // in a candidate's predicted column is one factor and not two.
    //
    // The differential: `query_permutations_at_rate` is written through felts and a
    // rate, `query_permutations` through bytes and `keccak_host::num_blocks`.
    // Neither delegates to the other, so their agreement at rate 17 is a real check
    // on the felt-side reformulation — and the existing assert above already ties
    // `query_permutations` to the EMITTED count, so the chain reaches the emitter.
    use super::epoch_verify::{
        FRI_LEAF_FELTS, KECCAK_RATE_FELTS, LFM_HASH_RATE_FELTS, blocks_at_rate,
        fri_leaf_permutations_at_rate, group_leaf_felts, query_permutations_at_rate,
    };
    for s in &real_lengths {
        assert_eq!(
            query_permutations_at_rate(s, KECCAK_RATE_FELTS),
            super::epoch_verify::query_permutations(s),
            "the felt-side closed form must reproduce the byte-side one at keccak's rate"
        );
    }
    // ⚠ A FRI layer leaf is six felts. It fits ONE keccak block and it does NOT
    // fit one block at the candidate's rate — which was 8 under the deleted
    // three-cell duplex and is 4 under the B1 compress chain. So the FRI leaf
    // term is rate-SENSITIVE and is no longer part of the invariant remainder.
    // The old premise assertion here (`6 <= LFM_HASH_RATE_FELTS`) is gone: it
    // was true at 8, is false at 4, and re-asserting it would have pinned the
    // model to a construction that no longer exists.
    assert_eq!(blocks_at_rate(FRI_LEAF_FELTS, KECCAK_RATE_FELTS), 1);
    assert_eq!(blocks_at_rate(FRI_LEAF_FELTS, LFM_HASH_RATE_FELTS), 2);

    let keccak_p = sum(&real_lengths, &|s| {
        query_permutations_at_rate(s, KECCAK_RATE_FELTS)
    });
    let cand_p = sum(&real_lengths, &|s| {
        query_permutations_at_rate(s, LFM_HASH_RATE_FELTS)
    });
    // Decompose so the penalty is attributed rather than asserted in aggregate.
    // The split is ABSORPTION vs COMPRESSION, not leaf vs rest: leaves of both
    // kinds absorb and move with the rate, Merkle parents of both kinds
    // compress and do not.
    let absorb_at = |rate: usize| {
        sum(&real_lengths, &|s| {
            s.num_queries
                * (super::epoch_verify::leaf_permutations_at_rate(&s.sub, rate)
                    + fri_leaf_permutations_at_rate(&s.fri, rate))
        })
    };
    let leaf_k = absorb_at(KECCAK_RATE_FELTS);
    let leaf_c = absorb_at(LFM_HASH_RATE_FELTS);
    let paths = keccak_p - leaf_k;
    assert_eq!(
        cand_p,
        leaf_c + paths,
        "only ABSORPTION may move with the rate; compression (Merkle parents, \
         trace trees and FRI path steps alike) must not"
    );
    assert!(
        cand_p > keccak_p,
        "the candidate's smaller rate must COST permutations — if this ever fails, \
         the rate penalty reasoning in others/lfm-hash-matrix-scope.md is wrong"
    );
    let widest = real_lengths
        .iter()
        .map(|s| {
            s.sub
                .groups()
                .iter()
                .map(group_leaf_felts)
                .max()
                .unwrap_or(0)
        })
        .max()
        .expect("the epoch has groups");
    println!(
        "\n  ★ HASH MATRIX — the PERMUTATION axis at blowup 8 / 73 queries, real \
         trace lengths (no candidate permutation exists yet; this is shape \
         arithmetic only):\n\
         \x20 keccak   rate {KECCAK_RATE_FELTS:>2} felts/perm: {keccak_p:>9} permutations \
         ({leaf_k} absorbed + {paths} compressed)\n\
         \x20 LFM_HASH rate {LFM_HASH_RATE_FELTS:>2} felts/perm: {cand_p:>9} permutations \
         ({leaf_c} absorbed + {paths} compressed)\n\
         \x20 candidate/keccak = {:.4}x   (absorption term alone {:.4}x; the \
         ceiling is 17/{LFM_HASH_RATE_FELTS} = {:.3}x and only absorption pays it)\n\
         \x20 absorption-bound share of the keccak bill: {:.1}%   widest leaf: \
         {widest} felts\n\
         \x20 ⚠ the candidate rate is 4, not the 8 this model carried before \
         option B1 — the compress chain absorbs ONE cell per step, so both the \
         trace-group leaves and the 6-felt FRI-layer leaves take two blocks",
        cand_p as f64 / keccak_p as f64,
        leaf_c as f64 / leaf_k as f64,
        KECCAK_RATE_FELTS as f64 / LFM_HASH_RATE_FELTS as f64,
        100.0 * leaf_k as f64 / keccak_p as f64,
    );

    println!(
        "\n  RECONCILIATION against the pinned blowup-8 predictions (projections \
         from shapes — this run is at the min preset and measures none of them):\n\
         \x20 openings only, 73 queries, UNIFORM 2^20 (deep-join's own \
         assumption, over this epoch's 16 sub-proofs): {}   [pinned: see the run output \
         over all 28 production AIRs]\n\
         \x20 openings only, 73 queries, this epoch's REAL trace lengths: {}\n\
         \x20 openings + FRI, 73 queries, real lengths: {}\n\
         \x20 FRI alone, 73 queries, real lengths: {}   [pinned: 14,454 per \
         sub-proof at blowup 8, i.e. for a 2^20 table]",
        sum(&uniform, &openings_only),
        sum(&real_lengths, &openings_only),
        sum(&real_lengths, &|s| super::epoch_verify::query_permutations(
            s
        )),
        sum(&real_lengths, &|s: &TableVerifyShape| s.num_queries
            * s.fri.permutations_per_query()),
    );
    // The one sub-proof that IS a 2^20 table, so the per-sub-proof FRI figure the
    // FRI leg pinned has something to be checked against.
    let biggest = e
        .legs
        .iter()
        .max_by_key(|l| l.verify.sub.deep.log2_trace_length)
        .expect("the epoch has sub-proofs");
    let big8 = at_blowup_8(biggest, None);
    println!(
        "\x20 the epoch's 2^{} sub-proof at blowup 8: FRI {} permutations \
         ({} committed layers), openings {}",
        big8.sub.deep.log2_trace_length,
        big8.num_queries * big8.fri.permutations_per_query(),
        big8.fri.num_committed(),
        openings_only(&big8),
    );
    println!("\x20 trace lengths in this epoch (log2): {:?}", {
        let mut v: Vec<u32> = e
            .legs
            .iter()
            .map(|l| l.verify.sub.deep.log2_trace_length)
            .collect();
        v.sort_unstable();
        v
    });
}

/// Where each per-table arena sits in the declaration order
/// `epoch_tests::epoch_arena_words` produces.
///
/// Computed from the presence flags rather than hardcoded, because a table
/// without an aux root or without grinding shifts every arena behind it — which
/// is precisely the failure mode the per-field arena packing exists to prevent
/// and a hardcoded index would reintroduce in the TEST.
pub(super) struct ArenaIndex {
    pub(super) openings: usize,
    fri: usize,
    parts: usize,
    ood_current: usize,
}

pub(super) fn arena_index(e: &super::epoch_tests::RealEpoch, table: usize) -> ArenaIndex {
    // The epoch-wide arenas come first, and their COUNT comes from the emitter's
    // own side rather than from a literal here: wiring ledger entry 7 added the
    // second register vector, `pc_start` and (when non-empty) the page roots, and a
    // literal `4` would have left every vector below tampering the wrong arena.
    let mut at = super::epoch_tests::num_epoch_wide_arenas(e);
    for (i, h) in e.tables.iter().enumerate() {
        let aux = usize::from(h.shape.has_aux_root);
        let contribution = usize::from(h.shape.has_contribution);
        let nonce = usize::from(h.shape.grinding_factor > 0);
        let composition = at + aux + contribution;
        if i == table {
            return ArenaIndex {
                ood_current: composition + 1,
                parts: composition + 3,
                openings: composition + 6 + nonce,
                fri: composition + 7 + nonce,
            };
        }
        // The table's last arena is `fri` at `composition + 7 + nonce`, so the
        // next table starts one past it. Getting this stride wrong is how the
        // first version of this test came to tamper an EMPTY arena two tables
        // later — which is why the loop below checks every computed index
        // against the arena lengths the shapes fix.
        at = composition + 8 + nonce;
    }
    unreachable!("table index out of range");
}

/// ★ FALSIFICATION: run the attacks the wiring denies, and watch each fail.
///
/// Every check the legs add is an `assert` inside the program, so "it executed"
/// is the whole positive result — which makes this test the entire negative half.
/// Each vector is a single arena word moved by one, and each must make the
/// program unexecutable. What each one proves is different, so they are labelled
/// rather than swept anonymously:
///
/// - an OPENED VALUE: the leaf hash changes, so the walk reaches a root the
///   transcript never absorbed. This is also the two-consumer join — the same
///   cell is what DEEP folds, so there is no way to move one without the other.
/// - a MERKLE SIBLING, both words: a path that authenticates nothing. Both words
///   are hit deliberately; a past tamper suite in this phase touched only byte 0
///   of every digest, so a digest's second word was never checked.
/// - a FRI SYMMETRIC EVALUATION and a FRI SIBLING: the layer walk, on the one
///   sub-proof of this epoch that actually folds (12 committed layers).
/// - a CLAIMED COMPOSITION PART: this one is absorbed, so it moves the
///   challenges as well — it must reject, and the interesting part is that it
///   cannot reject "only" the quotient check.
/// - an OOD CELL: likewise absorbed, and read by both the constraint fold and
///   DEEP.
#[test]
fn the_assembled_verifier_rejects_tampered_leg_data() {
    let e = super::epoch_tests::real_epoch();
    let program = super::epoch_tests::epoch_program(&e, true);
    let good = super::epoch_tests::epoch_arena_words(&e, true);
    // The pin, for the same reason as `the_assembled_epoch_verifier_runs`: this
    // is the same `WrapHash::production()` program. It matters most on THIS
    // arm — the honest control is what a wrong socket permutation breaks first,
    // and a tamper suite whose control is broken rejects everything and passes.
    assert!(
        execute(&program, &good, &crate::hash_pin::BLOCK_HASHER).is_ok(),
        "the untampered assembled verifier must run"
    );

    // The sub-proof that folds, so the FRI vectors reach the layer walk.
    let folding = e
        .legs
        .iter()
        .position(|l| l.verify.fri.num_committed() > 0)
        .expect("this epoch has a sub-proof with committed FRI layers");

    // ★ The index arithmetic above is a claim about the declaration order, and a
    // WRONG index would make this whole test lie — it would tamper some other
    // arena, still get a rejection, and report a pass. So the claim is checked
    // against the arena LENGTHS, which the shapes fix independently.
    for (t, leg) in e.legs.iter().enumerate() {
        let ix = arena_index(&e, t);
        assert_eq!(
            good[ix.openings].len(),
            leg.verify
                .opening_words(super::proof_arena::words_per_root()),
            "table {t}: the arena at the computed openings index is not the \
             openings arena"
        );
        assert_eq!(
            good[ix.fri].len(),
            leg.verify.fri_words(super::proof_arena::words_per_root()),
            "table {t}: the arena at the computed FRI index is not the FRI arena"
        );
        assert_eq!(
            good[ix.parts].len(),
            e.tables[t].parts.len(),
            "table {t}: the arena at the computed parts index is not the parts arena"
        );
        assert_eq!(
            good[ix.ood_current].len(),
            e.tables[t].ood_current.len(),
            "table {t}: the arena at the computed OOD index is not the OOD arena"
        );
    }

    let mut vectors: Vec<(String, usize, usize)> = Vec::new();
    // Trace openings: the first value and both words of the first sibling
    // digest, on three tables including the folding one.
    for &t in &[0usize, 1, folding] {
        let ix = arena_index(&e, t);
        let leg = &e.legs[t];
        let values = leg.verify.sub.groups()[0].num_values();
        vectors.push((format!("table {t}: opened value 0"), ix.openings, 0));
        vectors.push((
            format!("table {t}: last opened value of group 0"),
            ix.openings,
            values - 1,
        ));
        vectors.push((format!("table {t}: sibling lo"), ix.openings, values));
        vectors.push((format!("table {t}: sibling hi"), ix.openings, values + 1));
        vectors.push((format!("table {t}: claimed part 0"), ix.parts, 0));
        vectors.push((format!("table {t}: OOD cell 0"), ix.ood_current, 0));
    }
    // FRI: the first layer's symmetric evaluation, then both words of its first
    // sibling.
    let fri_ix = arena_index(&e, folding).fri;
    vectors.push(("FRI layer 0 sym".to_string(), fri_ix, 0));
    vectors.push(("FRI layer 0 sibling lo".to_string(), fri_ix, 1));
    vectors.push(("FRI layer 0 sibling hi".to_string(), fri_ix, 2));

    for (label, arena, word) in &vectors {
        let mut arenas = good.clone();
        let before = arenas[*arena][*word];
        arenas[*arena][*word][0] = before[0] + FE::one();
        assert!(
            execute(&program, &arenas, &crate::hash_pin::BLOCK_HASHER).is_err(),
            "tampering {label} must make the assembled verifier unexecutable, \
             and did not"
        );
    }
    println!("  {} tamper vectors, all rejected", vectors.len());
}

/// ★ The boundary list the emitted program carries is a PROGRAM CONSTANT, and
/// this is the premise that makes it one.
///
/// `AIR::boundary_constraints` takes the public inputs and the bus public inputs
/// — both proof data — so building the emitted list from that call would make the
/// program depend on the proof it verifies. `epoch_verify::boundary_terms` builds
/// it from a rule instead. The rule is only safe while it agrees with the call on
/// every AIR of a real epoch, so this compares them as SETS: a term the rule
/// missed would be a constraint the machine silently never checks.
#[test]
fn the_boundary_terms_are_program_shape() {
    let e = super::epoch_tests::real_epoch();
    let mut with_boundary = 0usize;
    for (i, leg) in e.legs.iter().enumerate() {
        let rule = boundary_terms(leg.has_aux_trace, leg.verify.sub.deep.num_total_cols);
        let want = &leg.production_boundary;
        assert_eq!(
            rule.len(),
            want.len(),
            "table {i}: the rule and production disagree about how many boundary \
             constraints the AIR has"
        );
        for (r, w) in rule.iter().zip(want) {
            assert_eq!(r.col, w.col, "table {i}: boundary column");
            assert_eq!(r.point, w.point, "table {i}: boundary point");
            assert_eq!(r.value, w.value, "table {i}: boundary value");
        }
        if !want.is_empty() {
            with_boundary += 1;
        }
    }
    // Positive control: a suite where every AIR had an empty list would pass
    // vacuously, and the rule's interesting branch would be untested.
    assert!(
        with_boundary > 0,
        "no sub-proof carries a boundary constraint, so this proves nothing about \
         the rule's non-empty branch"
    );
    println!(
        "  boundary premise: {with_boundary} of {} sub-proofs carry the \
         framework's acc[0] = 0 and nothing else",
        e.legs.len()
    );
}

/// ★ An ABSOLUTE structural guard over the ASSEMBLED verifier: no proof value is
/// hinted twice, legs included.
///
/// This is the count that closes assembly obligation 3. The spine's own version
/// (`epoch_tests::the_spine_hints_each_proof_value_once`) could only say the
/// spine hinted nothing twice — the legs were not in the program, so their second
/// consumers had nothing to disagree with. Now they are, and the same absolute
/// property must hold over the whole thing: the OOD grid, the claimed parts,
/// every root and every challenge reach the legs as cells, never as a second
/// read.
#[test]
fn the_assembled_verifier_hints_each_proof_value_once() {
    use std::collections::HashMap;

    let e = super::epoch_tests::real_epoch();
    let program = super::epoch_tests::epoch_program(&e, true);

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

    let declared: usize = program.arena_schema.lens.iter().map(|l| *l as usize).sum();
    assert_eq!(
        hints.len(),
        declared,
        "every declared arena word must be read exactly once"
    );
    // The legs are actually IN this program — without this the guard would pass
    // just as happily over the spine alone.
    let spine = super::epoch_tests::epoch_program(&e, false);
    assert!(
        declared
            > spine
                .arena_schema
                .lens
                .iter()
                .map(|l| *l as usize)
                .sum::<usize>(),
        "the assembled program must declare more arena words than the spine, or \
         the legs are not wired and this guard is vacuous"
    );
}

/// ★ The preprocessed-commitment inventory of a real epoch — the EVIDENCE
/// assembly ledger entry 7 was opened without.
///
/// Entry 7 says five AIRs are preprocessed (BITWISE, DECODE, KECCAK_RC, REGISTER,
/// PAGE), that three are compile-time constants, that REGISTER has a derivation
/// and that PAGE cannot become a program constant because it is a function of the
/// inner ELF. That is a claim about the AIR SET. This test asks the real epoch
/// which of its sub-proofs are actually preprocessed, and how many columns each
/// commits, so the proposal that closes the entry is built on a census rather
/// than on a recollection.
///
/// `VmAirs::air_refs` fixes the order (`lib.rs:610-625`): BITWISE, DECODE, COMMIT,
/// KECCAK, KECCAK_RND, KECCAK_RC, ECSM, ECDAS, REGISTER, then optional HALT, then
/// the chunked tables, then the PAGE tables, and this suite appends L2G_MEMORY.
/// So a preprocessed sub-proof at index 8 is REGISTER and one past the chunked
/// tables is a PAGE — which is what makes "which sub-proof is which AIR" program
/// shape rather than proof data.
#[test]
fn the_preprocessed_commitments_of_a_real_epoch() {
    let e = super::epoch_tests::real_epoch();
    let preprocessed: Vec<(usize, usize)> = e
        .legs
        .iter()
        .enumerate()
        .filter(|(_, l)| l.num_precomputed_cols > 0)
        .map(|(i, l)| (i, l.num_precomputed_cols))
        .collect();
    println!(
        "  {} of {} sub-proofs are preprocessed: {:?} (index, precomputed columns)",
        preprocessed.len(),
        e.legs.len(),
        preprocessed
    );
    // The REGISTER slot, checked by its column count rather than assumed from its
    // index: the derivation commits OFFSET ‖ INIT ‖ FINI.
    let register = e.legs.iter().position(|l| {
        l.num_precomputed_cols == crate::tables::register::NUM_PREPROCESSED_COLS_WITH_FINI
    });
    println!(
        "  the sub-proof whose preprocessed width is NUM_PREPROCESSED_COLS_WITH_FINI \
         ({}): index {:?}",
        crate::tables::register::NUM_PREPROCESSED_COLS_WITH_FINI,
        register
    );
    // Every preprocessed sub-proof's commitment must actually be present, or the
    // spine would be absorbing something it did not get from the AIR.
    for (i, _) in &preprocessed {
        assert!(
            e.legs[*i].precomputed_commitment.is_some(),
            "sub-proof {i} declares preprocessed columns but has no AIR commitment"
        );
    }
    assert!(
        !preprocessed.is_empty(),
        "an epoch with no preprocessed sub-proof cannot witness entry 7 at all"
    );

    // ---- ★ the PROVENANCE census, which is what entry 7 actually turns on.
    //
    // `epoch_tests::prep_source` decided each root's source by recomputing every
    // candidate production has; reaching this line means every preprocessed root of
    // a real epoch matched one, so nothing is hinted without a binding. What is
    // asserted here is the SHAPE of the taxonomy — that the epoch is not all
    // constants (which would make the derivation and the fold untested) and not all
    // ELF-dependent (which would mean interning bought nothing).
    let sources = super::epoch_tests::prep_source_census(&e);
    println!(
        "  provenance: {} options-only (interned as program text), {} derived \
         in-machine (REGISTER), {} ELF-dependent (arena cell + attestation join)",
        sources.0, sources.1, sources.2
    );
    assert_eq!(
        sources.0 + sources.1 + sources.2,
        preprocessed.len(),
        "every preprocessed sub-proof must have exactly one classified source"
    );
    assert!(
        sources.0 > 0,
        "no options-only root: the interning path is unexercised"
    );
    assert_eq!(
        sources.1, 1,
        "exactly one derived root — the REGISTER commitment, from the epoch's own \
         register boundary"
    );
    assert_eq!(
        sources.2, 1,
        "exactly one ELF-dependent root in a continuation epoch — DECODE. A second \
         would mean the attestation fold's input is ambiguous"
    );

    // ★ AND THE PAGE HALF OF ENTRY 7 IS NOT A FIXTURE ARTEFACT. `prove_epoch`
    // rejects any epoch carrying a PAGE config ("continuation epoch must have no
    // PAGE configs (L2G bookend replaces PAGE)", `continuation.rs:695-702`) and both
    // `build_epoch_airs` call sites pass `&[]`. So no continuation epoch of any
    // guest has a PAGE sub-proof, and the ELF-data page genesis roots the
    // attestation folds are the GLOBAL proof's GlobalMemory AIRs' preprocessed
    // commitments (`continuation.rs:997-1010`) — a different proof, out of an epoch
    // verifier's scope.
    //
    // Asserted rather than remembered. The width test is unambiguous only because
    // a continuation epoch's REGISTER always uses the WITH_FINI layout
    // (`build_epoch_airs` always supplies `register_preprocessed`), and PAGE's
    // width coincides with the non-FINI REGISTER one — so that premise is checked
    // first. An ELF-data page root would in any case make `prep_source` panic,
    // since its provenance is not in the classifier's candidate list.
    assert!(
        register.is_some(),
        "a continuation epoch's REGISTER is preprocessed WITH FINI; without that \
         the width check below cannot tell a PAGE table from a REGISTER one"
    );
    assert!(
        e.legs
            .iter()
            .all(|l| l.num_precomputed_cols != crate::tables::page::NUM_PREPROCESSED_COLS),
        "a sub-proof with PAGE's preprocessed width appeared: continuation epochs \
         are supposed to carry none, and the entry-7 taxonomy changes if they do"
    );
}

/// ★ The composition and FRI-terminal CHECKS are in the program, counted.
///
/// This guard exists because falsification found the hole it closes. Deleting
/// `assert_eq_ext(q.claimed, q.composition)` from the emitter fails NOTHING in
/// this suite: with honest data the two values ARE equal, so no differential and
/// no arena tamper can see the assert's absence. And no arena tamper ever will —
/// every input to the quotient identity (the OOD grid, the claimed parts, `z`,
/// `β`) is absorbed by the transcript, so moving any of them moves the challenges
/// and the run fails at the Merkle walk instead, for the wrong reason.
///
/// What DOES witness the check is a mutation that makes the identity false while
/// leaving the transcript alone — emptying the boundary-term list does exactly
/// that, and three tests catch it. But "a mutation elsewhere catches it" is not
/// the same as "the check is present", so this counts the checks directly.
///
/// `assert_eq_ext(a, b)` lowers to `esub` then `ediv(diff, ZERO)`
/// (`builder.rs:243-247`): division by the interned zero has a witness only when
/// the numerator vanishes, since `OUT · 0 = A` forces `A = 0`. So an extension
/// division whose DIVISOR is the pooled zero constant is an equality assertion,
/// and nothing else in the machine produces one — every other `ediv` here
/// inverts against the interned ONE.
///
/// The expected count is arithmetic over the shapes, not a second emitter pass:
/// one composition check per sub-proof, plus per query one FRI terminal check
/// when the codeword folds and TWO when it does not (the zero-fold shape checks
/// `P` at both `υ` and `−υ`).
#[test]
fn the_assembled_verifier_contains_every_composition_and_terminal_check() {
    use super::instr::{ExtOp, Instr};

    let e = super::epoch_tests::real_epoch();
    let program = super::epoch_tests::epoch_program(&e, true);
    let spine = super::epoch_tests::epoch_program(&e, false);

    let asserts = |p: &super::compiler::LfmProgram| -> usize {
        // The interned all-zero word. `felt_const(0)` and `ext_const(0)` are the
        // same word, and the builder interns program-wide, so there is one.
        let zeros: Vec<_> = p
            .instrs
            .iter()
            .filter_map(|i| match i {
                Instr::Const { out, value, .. } if value.iter().all(|v| *v == FE::zero()) => {
                    Some(*out)
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            zeros.len(),
            1,
            "the zero word must be interned exactly once, or this count is \
             ambiguous"
        );
        let zero = zeros[0];
        p.instrs
            .iter()
            .filter(|i| {
                matches!(
                    i,
                    Instr::ExtAlu {
                        op: ExtOp::Div,
                        b,
                        ..
                    } if *b == zero
                )
            })
            .count()
    };

    let expected: usize = e
        .legs
        .iter()
        .map(|l| {
            let terminal = if l.verify.fri.total_folds() > 0 { 1 } else { 2 };
            1 + l.verify.num_queries * terminal
        })
        .sum();
    assert_eq!(
        asserts(&program) - asserts(&spine),
        expected,
        "the legs must add exactly one composition check per sub-proof plus the \
         FRI terminal checks the shapes call for"
    );
    // Positive control: the count must be nonzero and the shapes must actually
    // include both FRI branches, or the formula's second case is untested.
    assert!(expected > 0);
    assert!(
        e.legs.iter().any(|l| l.verify.fri.total_folds() > 0)
            && e.legs.iter().any(|l| l.verify.fri.total_folds() == 0),
        "this epoch must exercise BOTH the folding and the zero-fold terminal \
         shapes, or the expected count is only half checked"
    );
    println!(
        "  {} equality assertions added by the legs (25 composition + FRI \
         terminals)",
        expected
    );
}

/// The rate model's corrected pieces, WITHOUT a real epoch.
///
/// The hash-matrix permutation-axis block that consumes these lives inside
/// `the_assembled_epoch_verifier_runs`, which needs the fixture guest ELF. That is
/// exactly how `LFM_HASH_RATE_FELTS = 8` outlived the three-cell duplex it was
/// derived from: nothing that ran in a bare checkout touched it. This test does,
/// on shapes built by hand.
///
/// What it pins is the correction itself — the constant's derivation, and that
/// the FRI-leaf term is rate-sensitive and still reduces to `num_committed()` at
/// keccak's rate, which is the identity that keeps the felt-side and byte-side
/// closed forms agreeing.
#[test]
fn the_candidate_rate_model_is_derived_not_remembered() {
    use super::epoch_verify::{
        FRI_LEAF_FELTS, KECCAK_RATE_FELTS, LFM_HASH_RATE_FELTS, blocks_at_rate,
        fri_leaf_permutations_at_rate,
    };
    use super::fri::FriShape;
    use super::hash::HASH_DIGEST_FELTS;

    // The chain absorbs ONE cell per step, so the rate IS the digest width.
    // Written as the derivation, not as a literal, because the literal is what
    // went stale.
    assert_eq!(LFM_HASH_RATE_FELTS, HASH_DIGEST_FELTS);
    assert_eq!(LFM_HASH_RATE_FELTS, 4, "was 8 under the deleted duplex");

    // ⚠ The premise the old model folded the FRI leaf into: "a layer leaf fits
    // one block at the candidate's rate". True at 8, FALSE at 4.
    assert_eq!(blocks_at_rate(FRI_LEAF_FELTS, KECCAK_RATE_FELTS), 1);
    assert_eq!(blocks_at_rate(FRI_LEAF_FELTS, 8), 1, "the old rate did fit");
    assert_eq!(blocks_at_rate(FRI_LEAF_FELTS, LFM_HASH_RATE_FELTS), 2);

    let fri = FriShape {
        log2_lde_length: 20,
        blowup_log: 3,
        final_poly_log_degree: 3,
        coset_offset: 3,
        num_queries: 73,
        format: stark::proof::options::ProofFormat::DEFAULT,
    };
    assert!(fri.num_committed() > 0, "the shape must exercise the term");

    // At keccak's rate the new term reduces to the old `num_committed()`, which
    // is why splitting it out did not move the rate-17 column.
    assert_eq!(
        fri_leaf_permutations_at_rate(&fri, KECCAK_RATE_FELTS),
        fri.num_committed()
    );
    // At the candidate's rate it doubles — the cost the old model hid.
    assert_eq!(
        fri_leaf_permutations_at_rate(&fri, LFM_HASH_RATE_FELTS),
        2 * fri.num_committed()
    );

    // A shape with nothing committed contributes nothing at any rate, so the
    // correction cannot invent cost where there is no FRI leg.
    let terminal = FriShape {
        log2_lde_length: 6,
        ..fri
    };
    assert_eq!(terminal.num_committed(), 0);
    for rate in [KECCAK_RATE_FELTS, LFM_HASH_RATE_FELTS] {
        assert_eq!(fri_leaf_permutations_at_rate(&terminal, rate), 0);
    }
}

/// Queries the knob-on twin proves at: enough openings that `auto` caps every
/// tall tree at height 3 (from 20 openings on).
const PROCESS_FORMAT_QUERIES: usize = 24;

/// ★ The KNOB-ON TWIN of [`the_assembled_epoch_verifier_runs`] (box only): a
/// real continuation epoch proved at the PROCESS format — `ZfFormat::global()`,
/// i.e. `LAMBDA_VM_ZF_CAP` / `LAMBDA_VM_ZF_FRI` — at the MIN preset with
/// [`PROCESS_FORMAT_QUERIES`] queries, verified by the assembled machine.
///
/// Asserts, per format: the program executes (every cap authenticated once per
/// tree, every opening checked against it, every FRI group folded to the
/// terminal); the legs' emitted permutations equal the closed form
/// `Σ table_permutations_for` (per-query paths cut at each tree's cap plus
/// `2^c − 1` once per capped tree; group leaves and group paths under
/// `fri = dp`); a moved cap word does not execute. Prints the census to
/// compare across arms (instructions, permutations, `Select`s, cells per
/// chip). It proves the row-pair formats (`LAMBDA_VM_ZF_ONE_ROW=0`): at the MIN
/// preset's blowup 2 no one-row static root ships, so a format that puts
/// KECCAK_RC on one row — `one_row=1`, and `one_row=auto`, the default — is a
/// proving error there (a hard miss, never a recompute). The one-row twin is
/// [`the_assembled_epoch_verifier_runs_at_blowup_4_at_the_process_format`].
#[test]
#[ignore = "a real epoch proof at 24 queries and its assembled verifier: box only"]
fn the_assembled_epoch_verifier_runs_at_the_process_format() {
    let mut opts = super::proof_fixture::fixture_options();
    opts.fri_number_of_queries = PROCESS_FORMAT_QUERIES;
    assembled_twin_at_the_process_format(opts);
}

/// ★ [`the_assembled_epoch_verifier_runs_at_the_process_format`] at BLOWUP 4
/// — the S2 (one-row) twin, box only. One-row static roots exist at blowup 4
/// only (`STATIC_BLOWUP_FACTORS_ONE_ROW`; a missing twin is a
/// proving error), so the MIN preset's blowup 2 cannot prove a one-row
/// BITWISE; this arm keeps every other MIN-preset option and lifts the blowup
/// to 4 for every format, so its knob-off and knob-on runs are one A/B. Under
/// `one_row = auto` the epoch's tables resolve their layouts one by one
/// (printed per leg), so the assembled machine verifies a MIXED-layout proof;
/// the REGISTER root is derived in-machine at that table's own layout.
#[test]
#[ignore = "a real epoch proof at blowup 4, 24 queries, and its assembled verifier: box only"]
fn the_assembled_epoch_verifier_runs_at_blowup_4_at_the_process_format() {
    let mut opts = super::proof_fixture::fixture_options();
    opts.fri_number_of_queries = PROCESS_FORMAT_QUERIES;
    opts.blowup_factor = 4;
    assembled_twin_at_the_process_format(opts);
}

/// The body of the assembled-verifier twins: `base` with the process format
/// stamped on, proved, harvested and verified by the assembled machine.
fn assembled_twin_at_the_process_format(base: crate::ProofOptions) {
    let format = crate::zf_format::ZfFormat::global();
    let opts = format.options(base);
    let e = super::epoch_tests::real_epoch_with(opts.clone());
    let program = super::epoch_tests::epoch_program(&e, true);
    let arenas = super::epoch_tests::epoch_arena_words(&e, true);
    execute(&program, &arenas, &crate::hash_pin::BLOCK_HASHER)
        .expect("the assembled verifier must execute at the process format");

    let spine = super::epoch_tests::epoch_program(&e, false);
    let perms = |p: &_| super::machine_tests::wrap_hash_instrs(p);
    let selects = |p: &super::compiler::LfmProgram| {
        p.instrs
            .iter()
            .filter(|i| matches!(i, super::instr::Instr::Select { .. }))
            .count()
    };
    let hash = super::edsl::WrapHash::production();
    let emitted = perms(&program) - perms(&spine);
    let predicted: usize = e
        .legs
        .iter()
        .map(|l| super::epoch_verify::table_permutations_for(&l.verify, hash))
        .sum();
    let cap_perms: usize = e
        .legs
        .iter()
        .map(|l| super::epoch_verify::cap_permutations(&l.verify))
        .sum();
    println!(
        "\n★ ASSEMBLED EPOCH VERIFIER AT THE PROCESS FORMAT\n  {}\n  opts: blowup {}, \
         {} queries, grinding {}, k {}\n  sub-proofs {}  |  legs: {} instructions, \
         {} permutations ({} of them cap roots), {} selects  |  whole: {} instructions, \
         {} permutations",
        format.banner(),
        opts.blowup_factor,
        opts.fri_number_of_queries,
        opts.grinding_factor,
        opts.fri_final_poly_log_degree,
        e.legs.len(),
        program.instrs.len() - spine.instrs.len(),
        emitted,
        cap_perms,
        selects(&program) - selects(&spine),
        program.instrs.len(),
        perms(&program),
    );
    for (i, l) in e.legs.iter().enumerate() {
        let f = l.verify.fri;
        println!(
            "  leg {i:>2}: log2(lde) {:>2}  layout {:?}  trace cap {}  FRI schedule {:?} \
             depths {:?} caps {:?}  {} permutations",
            l.verify.sub.log2_lde_length,
            l.verify.sub.layout,
            l.verify.sub.trace_cap,
            f.schedule(),
            (0..f.num_committed())
                .map(|j| f.layer_depth(j))
                .collect::<Vec<_>>(),
            (0..f.num_committed())
                .map(|j| f.layer_cap(j))
                .collect::<Vec<_>>(),
            super::epoch_verify::table_permutations_for(&l.verify, hash),
        );
    }
    for c in super::airs::lfm_chip_census(&program) {
        println!(
            "  CENSUS {:<14} real {:>10} padded {:>10} cells {:>12}",
            c.name,
            c.real_rows,
            c.rows,
            c.main_cells()
        );
    }
    assert_eq!(
        emitted, predicted,
        "the legs' emitted permutations must equal the closed form at the process format"
    );
    println!("  emitted permutations == closed form: {emitted}");
    // One parseable line for the box wrapper's cross-arm comparison.
    println!(
        "ZFTWIN legs_permutations={emitted} cap_root_permutations={cap_perms} \
         legs_instructions={} legs_selects={} whole_instructions={}",
        program.instrs.len() - spine.instrs.len(),
        selects(&program) - selects(&spine),
        program.instrs.len(),
    );
    // S2: how many legs verify one-row tables (0 at `one_row = 0`, every leg
    // at `1`, the AIR widths' choice at `auto`), and the blowup of the arm.
    let one_row_legs = e
        .legs
        .iter()
        .filter(|l| l.verify.sub.layout.is_one_row())
        .count();
    println!(
        "ZFS2TWIN blowup={} legs={} one_row_legs={one_row_legs} row_pair_legs={}",
        opts.blowup_factor,
        e.legs.len(),
        e.legs.len() - one_row_legs,
    );
    match opts.format.one_row {
        stark::proof::options::OneRowMode::Off => assert_eq!(one_row_legs, 0),
        stark::proof::options::OneRowMode::On => assert_eq!(one_row_legs, e.legs.len()),
        stark::proof::options::OneRowMode::Auto => {}
    }
    // A one-row leg's input-tree group value (query 0, layer 0, value 0) moved
    // must not execute — the input group is authenticated and slot-checked.
    // The FRI arena is found by content rather than by a hand-counted offset.
    if let Some((k, words)) = e
        .legs
        .iter()
        .enumerate()
        .find(|(_, l)| l.verify.sub.layout.is_one_row() && l.verify.fri.num_committed() > 0)
        .map(|(k, l)| (k, l.fri_arena()))
    {
        let at = arenas
            .iter()
            .position(|a| *a == words)
            .expect("the one-row leg's FRI arena is among the program's arenas");
        let mut bad = arenas.clone();
        bad[at][0][0] += FE::one();
        execute(&program, &bad, &crate::hash_pin::BLOCK_HASHER)
            .expect_err("a moved input-tree value must not execute");
        println!("  leg {k}: a moved one-row input-tree value is refused");
    }

    // A moved cap word must not execute (only when the format caps a tree).
    // The caps arena is found by content rather than by a hand-counted offset.
    if let Some((k, words)) = e
        .legs
        .iter()
        .enumerate()
        .find_map(|(k, l)| l.caps_arena().map(|w| (k, w)))
    {
        let at = arenas
            .iter()
            .position(|a| *a == words)
            .expect("the caps arena is among the program's arenas");
        let mut bad = arenas.clone();
        bad[at][0][0] += FE::one();
        execute(&program, &bad, &crate::hash_pin::BLOCK_HASHER)
            .expect_err("a moved cap word must not execute");
        println!("  leg {k}: a moved cap word is refused");
    }
}
