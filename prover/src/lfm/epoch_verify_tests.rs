//! ★ The assembled epoch verifier — spine plus legs — run on a real
//! continuation epoch proof.
//!
//! [`super::epoch_tests`] built the Fiat-Shamir spine and checked all 111 of a
//! real 24-sub-proof epoch's challenges against production's own replay. Every
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
//! spine's — the 111 challenges, still checked — plus the fact of execution, and
//! the falsification tests below are what turn "it executed" into evidence, by
//! showing what does NOT execute.
//!
//! ## What this suite cannot see
//!
//! The preset. The fixture epoch is proved at the MIN preset (blowup 2, one
//! query per table, grinding factor 1), because that is what
//! `proof_fixture::fixture_options` gives and what keeps a 24-sub-proof epoch
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
use super::hash::TestPermutation;
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
    /// `[query][layer]` — `(pᵢ(−υ^(2ⁱ)), path)`.
    fri_openings: Vec<Vec<(FEE, Vec<Commitment>)>>,
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

    let sub = SubProofShape {
        deep,
        trace_groups,
        merkle_depth: log2_lde_length as usize - 1,
        log2_lde_length,
        coset_offset: FE::from(opts.coset_offset),
    };
    let has_aux_trace = air.has_aux_trace();
    let verify = TableVerifyShape {
        quotient: QuotientShape {
            log2_trace_length,
            num_composition_parts: claimed_parts.len(),
            boundary: boundary_terms(has_aux_trace, num_total_cols),
        },
        fri: FriShape::from_options(opts, log2_lde_length),
        main_width,
        num_alpha_powers: if has_aux_trace {
            artifact.shape.max_bus_elements as usize
        } else {
            0
        },
        num_queries: opts.fri_number_of_queries,
        sub,
    };

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
                    p.merkle_path().to_vec(),
                ));
            }
            let m = o.main_trace_polys();
            groups.push((
                m.evaluations()
                    .iter()
                    .chain(m.evaluations_sym())
                    .map(|v| base_word(*v))
                    .collect(),
                m.merkle_path().to_vec(),
            ));
            if aux_width > 0 {
                let a = o.aux_trace_polys().expect("an aux opening");
                groups.push((
                    a.evaluations()
                        .iter()
                        .chain(a.evaluations_sym())
                        .map(ext_word)
                        .collect(),
                    a.merkle_path().to_vec(),
                ));
            }
            let c = o.composition_poly();
            groups.push((
                c.evaluations()
                    .iter()
                    .chain(c.evaluations_sym())
                    .map(ext_word)
                    .collect(),
                c.merkle_path().to_vec(),
            ));
            groups
        })
        .collect();

    let fri_openings = (0..view.query_list_len())
        .map(|q| {
            let d = view.query(q);
            d.layers_evaluations_sym()
                .iter()
                .enumerate()
                .map(|(i, sym)| (*sym, d.layer_auth_path(i).to_vec()))
                .collect()
        })
        .collect();

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

    TableLegs {
        verify,
        analysis: analyze(&artifact),
        openings,
        fri_openings,
        production_boundary,
        has_aux_trace,
        num_precomputed_cols: num_precomputed,
        precomputed_commitment: air.is_preprocessed().then(|| air.precomputed_commitment()),
    }
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
            self.verify.opening_words(),
            "the opening arena must fill exactly what the shape declares"
        );
        out
    }

    /// Per query, per committed layer: the symmetric evaluation then its path.
    pub(super) fn fri_arena(&self) -> Vec<LfmWord> {
        let mut out = Vec::new();
        for query in &self.fri_openings {
            for (sym, path) in query {
                out.push(ext_word(sym));
                out.extend(super::proof_arena::commitments_to_arena(path));
            }
        }
        assert_eq!(
            out.len(),
            self.verify.fri_words(),
            "the FRI arena must fill exactly what the shape declares"
        );
        out
    }
}

/// ★ THE RUN: the whole epoch verifier — spine AND legs — on a real
/// continuation epoch proof that production accepts.
///
/// What executing proves, stated precisely. Every check is an assert inside the
/// program, so reaching the end means: all 24 quotient identities held at the
/// spine's own `z` and `β`; every one of the 24 sub-proofs' opened row pairs
/// hashed to a leaf that walked to the root the transcript absorbed, at the index
/// the transcript sampled; every DEEP reconstruction fed a FRI chain that folded
/// to the terminal polynomial the transcript absorbed; and the LogUp closure
/// reached production's COMMIT-bus target. The 111 published challenges are
/// checked against production's replay on top, so the Fiat-Shamir the whole thing
/// hangs from is still differentialled.
#[test]
fn the_assembled_epoch_verifier_runs() {
    let e = super::epoch_tests::real_epoch();
    let program = super::epoch_tests::epoch_program(&e, true);
    let arenas = super::epoch_tests::epoch_arena_words(&e, true);
    let exec =
        execute(&program, &arenas, &TestPermutation).expect("the assembled verifier must execute");

    // ---- the spine's differential, unchanged: production's own challenges.
    let pub_ext = |i: usize| word_as_ext(&exec.public_words[i].1).expect("an ext challenge");
    assert_eq!(pub_ext(0), e.z_alpha.0, "the shared LogUp challenge z");
    assert_eq!(pub_ext(1), e.z_alpha.1, "the shared LogUp challenge alpha");

    // The attestation fold is published right after Phase A (two digest words),
    // and its DECODE input is the cell Phase A absorbed — the join ledger entry 7
    // rests on. Its value is differentialled in the spine test; here it only has to
    // be skipped, and skipped by NAME rather than by a literal.
    let program_id_words = 2usize;
    let mut cursor = 2 + program_id_words;
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
    assert_eq!(
        checked, 111,
        "the same 111 challenges the spine test checks must still be checked"
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
    let count = |p: &super::compiler::LfmProgram, f: fn(&super::instr::Instr) -> bool| {
        p.instrs.iter().filter(|i| f(i)).count()
    };
    let perms = |p: &_| count(p, |i| matches!(i, super::instr::Instr::KeccakF(_)));
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
            let mut bare = super::builder::LfmBuilder::new();
            let _ = plumb(&mut bare);
            let baseline = bare.finish().instrs.len();

            let mut full = super::builder::LfmBuilder::new();
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
         recombination = {} over 24 sub-proofs  [pinned: 54,358 + 2,894 = 57,252]\
         \n\x20 that is {:.1}% of the legs' {} instructions",
        constraint_alu + recombination,
        100.0 * (constraint_alu + recombination) as f64
            / (program.instrs.len() - spine.instrs.len()) as f64,
        program.instrs.len() - spine.instrs.len(),
    );

    // ---- the permutation bill, against a CLOSED FORM over the shapes.
    //
    // Not a difference of two emitter passes (which rule 7's refinement rules
    // out) but arithmetic over byte widths: every group's leaf is
    // `⌊bytes/136⌋ + 1` rate blocks, every Merkle level is one, and FRI's own
    // per-query figure is the one the FRI leg pinned. Asserted, not printed, so
    // a leg that silently stopped hashing a group would fail here.
    let mut fri_perms = 0usize;
    let mut leaf_perms = 0usize;
    let mut walk_perms = 0usize;
    for leg in &e.legs {
        let groups = leg.verify.sub.groups().len();
        fri_perms += leg.verify.num_queries * leg.verify.fri.permutations_per_query();
        leaf_perms +=
            leg.verify.num_queries * super::epoch_verify::leaf_permutations(&leg.verify.sub);
        walk_perms += leg.verify.num_queries * groups * leg.verify.sub.merkle_depth;
    }
    let predicted: usize = e
        .legs
        .iter()
        .map(|l| super::epoch_verify::query_permutations(&l.verify))
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
    // measurement above is: this epoch's 24 sub-proofs, at their REAL trace
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
    println!(
        "\n  RECONCILIATION against the pinned blowup-8 predictions (projections \
         from shapes — this run is at the min preset and measures none of them):\n\
         \x20 openings only, 73 queries, UNIFORM 2^20 (deep-join's own \
         assumption, over this epoch's 24 sub-proofs): {}   [pinned: 213,744 \
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
struct ArenaIndex {
    openings: usize,
    fri: usize,
    parts: usize,
    ood_current: usize,
}

fn arena_index(e: &super::epoch_tests::RealEpoch, table: usize) -> ArenaIndex {
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
    assert!(
        execute(&program, &good, &TestPermutation).is_ok(),
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
            leg.verify.opening_words(),
            "table {t}: the arena at the computed openings index is not the \
             openings arena"
        );
        assert_eq!(
            good[ix.fri].len(),
            leg.verify.fri_words(),
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
            execute(&program, &arenas, &TestPermutation).is_err(),
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
        "  {} equality assertions added by the legs (24 composition + FRI \
         terminals)",
        expected
    );
}
