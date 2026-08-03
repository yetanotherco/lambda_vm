//! The DEEP/Merkle join: DEEP across a full sub-proof, folding the SAME arena
//! cells the Merkle authentication authenticates.
//!
//! ## The oracle
//!
//! Two production functions, neither of them re-derived here:
//! `reconstruct_deep_composition_poly_evaluation_pair` for the fold, and the
//! proof's own committed roots for the authentication. The fixture is a real
//! proof of a real production AIR, produced by the production prover, and its
//! query indices come from a replay of the production verifier's transcript
//! rather than from a search.
//!
//! ## What this suite cannot see
//!
//! The FRI leg that consumes `DEEP(υ)` — nothing here checks that the
//! reconstructed value is the one the folding chain expects, only that it is
//! the value the production verifier would have computed. It also cannot see
//! whether the epoch's OTHER sub-proofs compose, since a sub-proof is verified
//! in isolation here.
//!
//! DEPTH it sees only as far as the fixtures go: six levels on the
//! preprocessed fixture (64 rows at blowup 2), two on the production one.
//! `join_leg_cost` emits at depth 22 but never runs it, and R1f's
//! `keccak_merkle_walk_authenticates_a_real_opening` remains the only executed
//! walk at production depth (20, main trace only). Six levels is enough to
//! distinguish a per-level walk from a two-level one; it is not enough to catch
//! something that only appears past a word boundary in the index.

use math::field::traits::IsFFTField;
use stark::config::Commitment;
use stark::domain::new_verifier_domain;
use stark::proof::view::StarkProofView;
use stark::verifier::{IsStarkVerifier, Verifier};

use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};

use super::builder::LfmBuilder;
use super::compiler::compile;
use super::constraint_tests::{deep_shape, open_sub_proof, real_fixture};
use super::executor::execute;
use super::hash::TestPermutation;
use super::sub_proof::{
    GroupShape, ROWS_PER_LEAF, SubProofShape, emit_sub_proof, emit_sub_proof_with_bits,
};
use super::validator::validate;
use super::word::{LfmWord, base_word, ext_word, word_as_ext};

type Gl = GoldilocksField;
type Ext3 = GoldilocksExtension;
type V = Verifier<Gl, Ext3, ()>;

/// One committed matrix's data for one query, host side: the row pair in leaf
/// order and the path that authenticates it.
pub(super) struct HostGroupOpening {
    /// `evaluations ‖ evaluations_sym`, as arena words.
    values: Vec<LfmWord>,
    siblings: Vec<Commitment>,
}

/// Everything the machine reads about one sub-proof, read off a real proof.
///
/// Assembled once and shared, because `open_sub_proof` replays the whole
/// verifier transcript and the fixture proof is regenerated on every call.
pub(super) struct HostSubProof {
    pub(super) shape: SubProofShape,
    pub(super) gamma: FEE,
    pub(super) zeta: FEE,
    /// The OOD grid, row-major.
    ood: Vec<FEE>,
    claimed_parts: Vec<FEE>,
    /// One root per group, in `SubProofShape::groups` order.
    roots: Vec<Commitment>,
    /// `[query][group]`.
    openings: Vec<Vec<HostGroupOpening>>,
    pub(super) iotas: Vec<usize>,
    /// The FRI folding challenges, from the production verifier's own
    /// `replay_rounds_after_round_1` (`verifier.rs:1461-1483`) — one per
    /// committed layer plus the final-fold one. The FRI leg reads them; the
    /// trace leg does not.
    pub(super) zetas: Vec<FEE>,
    /// The production reconstruction's answer per query, `(regular, sym)`.
    pub(super) expected: Vec<(FEE, FEE)>,
    /// The same, asked of production with the PRECOMPUTED and MAIN slices
    /// swapped — the alternative column order a fixture without a precomputed
    /// group cannot distinguish. Empty when there is no precomputed group, or
    /// when the two base groups are different widths (the swap would not be a
    /// well-formed reading).
    expected_base_swapped: Vec<(FEE, FEE)>,
    /// Production's query points, kept so the machine's derivation can be
    /// checked against them rather than against a local formula.
    points: Vec<(FE, FE)>,
}

fn host_sub_proof() -> &'static HostSubProof {
    use std::sync::OnceLock;
    static CELL: OnceLock<HostSubProof> = OnceLock::new();
    CELL.get_or_init(|| {
        let (air, proof) = real_fixture();
        build_host_sub_proof(&*air, &proof)
    })
}

pub(super) fn build_host_sub_proof(
    air: &dyn stark::traits::AIR<Field = Gl, FieldExtension = Ext3, PublicInputs = ()>,
    proof: &stark::proof::stark::MultiProof<Gl, Ext3, ()>,
) -> HostSubProof {
    let sp = open_sub_proof(air, proof);
    let (deep, gamma) = deep_shape(&sp, air);
    let view = StarkProofView::Owned(&proof.proofs[0]);

    let (main_width, aux_width) = air.trace_layout();
    let num_precomputed = if air.is_preprocessed() {
        air.num_precomputed_columns()
    } else {
        0
    };
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

    let blowup = air.options().blowup_factor as usize;
    let lde_length = view.trace_length() * blowup;
    let shape = SubProofShape {
        deep: deep.clone(),
        trace_groups,
        merkle_depth: lde_length.trailing_zeros() as usize - 1,
        log2_lde_length: lde_length.trailing_zeros(),
        coset_offset: FE::from(air.options().coset_offset),
    };

    let mut roots = vec![];
    if num_precomputed > 0 {
        roots.push(
            *view
                .lde_trace_precomputed_merkle_root()
                .expect("a preprocessed air commits its precomputed columns"),
        );
    }
    roots.push(*view.lde_trace_main_merkle_root());
    if aux_width > 0 {
        roots.push(*view.lde_trace_aux_merkle_root().expect("an aux root"));
    }
    roots.push(*view.composition_poly_root());

    let domain = new_verifier_domain(air, view.trace_length());
    let layout = V::ood_layout(air);
    let invariants = V::compute_query_invariant_deep_terms(
        &sp.challenges,
        view,
        &sp.ood_full,
        layout.next_row_cols(),
        layout.step_size(),
    )
    .expect("a real proof's invariant terms");
    let generator = <Gl as IsFFTField>::get_primitive_root_of_unity(deep.log2_trace_length as u64)
        .expect("root of unity");

    let swap_is_well_formed =
        num_precomputed > 0 && main_width - num_precomputed == num_precomputed;
    let mut openings = Vec::new();
    let mut expected = Vec::new();
    let mut expected_base_swapped = Vec::new();
    let mut points = Vec::new();
    for (q, iota) in sp.challenges.iotas.iter().enumerate() {
        let o = view.deep_poly_opening(q);
        let mut groups: Vec<HostGroupOpening> = Vec::new();
        if num_precomputed > 0 {
            let p = o.precomputed_trace_polys().expect("precomputed opening");
            groups.push(HostGroupOpening {
                values: p
                    .evaluations()
                    .iter()
                    .chain(p.evaluations_sym())
                    .map(|v| base_word(*v))
                    .collect(),
                siblings: p.merkle_path().to_vec(),
            });
        }
        let m = o.main_trace_polys();
        groups.push(HostGroupOpening {
            values: m
                .evaluations()
                .iter()
                .chain(m.evaluations_sym())
                .map(|v| base_word(*v))
                .collect(),
            siblings: m.merkle_path().to_vec(),
        });
        if aux_width > 0 {
            let a = o.aux_trace_polys().expect("aux opening");
            groups.push(HostGroupOpening {
                values: a
                    .evaluations()
                    .iter()
                    .chain(a.evaluations_sym())
                    .map(ext_word)
                    .collect(),
                siblings: a.merkle_path().to_vec(),
            });
        }
        let c = o.composition_poly();
        groups.push(HostGroupOpening {
            values: c
                .evaluations()
                .iter()
                .chain(c.evaluations_sym())
                .map(ext_word)
                .collect(),
            siblings: c.merkle_path().to_vec(),
        });
        openings.push(groups);

        let point = V::query_challenge_to_evaluation_point(*iota, false, &domain);
        let point_sym = V::query_challenge_to_evaluation_point(*iota, true, &domain);
        let empty_base: &[FE] = &[];
        let (want, want_sym) = V::reconstruct_deep_composition_poly_evaluation_pair(
            &point,
            &point_sym,
            &generator,
            &sp.challenges,
            &invariants,
            layout.next_row_cols(),
            layout.step_size(),
            o.precomputed_trace_polys()
                .map(|p| p.evaluations())
                .unwrap_or(empty_base),
            m.evaluations(),
            o.aux_trace_polys().map(|a| a.evaluations()).unwrap_or(&[]),
            c.evaluations(),
            o.precomputed_trace_polys()
                .map(|p| p.evaluations_sym())
                .unwrap_or(empty_base),
            m.evaluations_sym(),
            o.aux_trace_polys()
                .map(|a| a.evaluations_sym())
                .unwrap_or(&[]),
            c.evaluations_sym(),
        )
        .expect("a real proof reconstructs");
        expected.push((want, want_sym));
        if swap_is_well_formed {
            let p = o.precomputed_trace_polys().expect("precomputed opening");
            let swapped = V::reconstruct_deep_composition_poly_evaluation_pair(
                &point,
                &point_sym,
                &generator,
                &sp.challenges,
                &invariants,
                layout.next_row_cols(),
                layout.step_size(),
                m.evaluations(),
                p.evaluations(),
                o.aux_trace_polys().map(|a| a.evaluations()).unwrap_or(&[]),
                c.evaluations(),
                m.evaluations_sym(),
                p.evaluations_sym(),
                o.aux_trace_polys()
                    .map(|a| a.evaluations_sym())
                    .unwrap_or(&[]),
                c.evaluations_sym(),
            )
            .expect("the swapped reading is well formed, so it reconstructs");
            expected_base_swapped.push(swapped);
        }
        points.push((point, point_sym));
    }

    let ood: Vec<FEE> = (0..deep.num_eval_points)
        .flat_map(|r| sp.ood_full.get_row(r)[..deep.num_total_cols].to_vec())
        .collect();

    HostSubProof {
        shape,
        gamma,
        zeta: sp.zeta,
        ood,
        claimed_parts: sp.claimed_parts.clone(),
        roots,
        openings,
        iotas: sp.challenges.iotas.clone(),
        zetas: sp.challenges.zetas.clone(),
        expected,
        expected_base_swapped,
        points,
    }
}

impl HostSubProof {
    /// The arenas [`emit_sub_proof`] declares, in its declaration order.
    pub(super) fn arenas(&self, queries: &[usize]) -> Vec<Vec<LfmWord>> {
        vec![
            vec![ext_word(&self.gamma), ext_word(&self.zeta)],
            self.ood.iter().map(ext_word).collect(),
            self.claimed_parts.iter().map(ext_word).collect(),
            super::proof_arena::commitments_to_arena(&self.roots),
            self.query_arena(queries),
        ]
    }

    /// Per query: the index, then per group the row-pair values and the
    /// sibling digests — the order the emitter's cursor walks.
    pub(super) fn query_arena(&self, queries: &[usize]) -> Vec<LfmWord> {
        let mut out = Vec::new();
        for &q in queries {
            out.push(base_word(FE::from(self.iotas[q] as u64)));
            for group in &self.openings[q] {
                out.extend(group.values.iter().copied());
                out.extend(super::proof_arena::commitments_to_arena(&group.siblings));
            }
        }
        out
    }
}

/// ★ Scrutinise the oracle before anything is built on it.
///
/// Four separate premises the join rests on, each checked against the real
/// proof rather than assumed: that every group commits at the SAME depth (one
/// index addresses all four trees), that the depth is one below the LDE domain
/// (a leaf is a row pair), that the machine's point derivation reproduces
/// production's `query_challenge_to_evaluation_point` at every one of the
/// proof's indices, and that the symmetric point really is the negation.
#[test]
fn the_join_premises_hold_on_a_real_proof() {
    let h = host_sub_proof();
    let s = &h.shape;
    let groups = s.groups();

    println!(
        "sub-proof: {} groups {:?}, depth {}, log2(lde) {}, {} queries",
        groups.len(),
        groups
            .iter()
            .map(|g| (g.num_columns, g.is_ext))
            .collect::<Vec<_>>(),
        s.merkle_depth,
        s.log2_lde_length,
        h.iotas.len()
    );

    assert_eq!(
        s.merkle_depth + 1,
        s.log2_lde_length as usize,
        "a leaf is a row pair, so the tree has one level fewer than the domain"
    );
    assert_eq!(
        ROWS_PER_LEAF,
        stark::commitment::ROWS_PER_LEAF,
        "the machine's leaf shape is a copy of the commitment layer's constant; \
         if that moves, every leaf hash and every DEEP index in this module goes \
         with it, and no differential would say so because both sides would move \
         together"
    );
    for (q, per_group) in h.openings.iter().enumerate() {
        for (g, opening) in per_group.iter().enumerate() {
            assert_eq!(
                opening.siblings.len(),
                s.merkle_depth,
                "query {q} group {g}: every tree must have the same depth, or \
                 one index cannot address them all"
            );
            assert_eq!(
                opening.values.len(),
                groups[g].num_values(),
                "query {q} group {g}: width"
            );
        }
    }

    // The point derivation, run IN THE MACHINE at every one of the proof's
    // indices and compared against production's own function. Recomputing the
    // bit weights here instead would only check a host formula against
    // production and leave the emitter unexamined — the same oracle mistake
    // the method rules warn about, one level up.
    let mut b = LfmBuilder::new();
    let arena = b.declare_arena(1);
    let index = b.hint_felt(arena, 0);
    let bits = b.bit_dec(index, s.merkle_depth);
    let (point, point_sym) = super::sub_proof::emit_query_points(&mut b, s, &bits);
    b.public(point.as_cell());
    b.public(point_sym.as_cell());
    let program = compile(b.finish());
    validate(&program).expect("the point-derivation program is admissible");

    for (q, iota) in h.iotas.iter().enumerate() {
        let arenas = vec![vec![base_word(FE::from(*iota as u64))]];
        let exec = execute(&program, &arenas, &TestPermutation).expect("the derivation executes");
        assert_eq!(
            exec.public_words[0].1[0], h.points[q].0,
            "query {q}: the machine's point must be \
             query_challenge_to_evaluation_point(iota, false)"
        );
        assert_eq!(
            exec.public_words[1].1[0], h.points[q].1,
            "query {q}: the machine's symmetric point must be \
             query_challenge_to_evaluation_point(iota, true)"
        );
    }
    println!(
        "in-machine point derivation checked against production at all {} indices",
        h.iotas.len()
    );
}

/// ★ The headline differential: the machine's DEEP equals the production
/// verifier's, at every query of a full sub-proof, with every opened value
/// authenticated to the proof's own committed roots in the same run.
///
/// The authentication is not a separate assertion here — it is `assert_word_eq`
/// inside the program, so a run in which any leaf failed to reach its root
/// would not execute at all. That the run produces DEEP values is already the
/// statement that the values it folded are the committed ones.
#[test]
fn the_join_matches_the_production_verifier_on_every_query() {
    let h = host_sub_proof();
    let all: Vec<usize> = (0..h.iotas.len()).collect();

    let mut b = LfmBuilder::new();
    let (_, outs) = emit_sub_proof(&mut b, &h.shape, all.len());
    for (p, s) in &outs {
        b.public(p.as_cell());
        b.public(s.as_cell());
    }
    let program = compile(b.finish());
    validate(&program).expect("the joined sub-proof program is admissible");

    let exec = execute(&program, &h.arenas(&all), &TestPermutation)
        .expect("an honest sub-proof must authenticate and fold");

    let mut nonzero = 0usize;
    for q in &all {
        let (want, want_sym) = h.expected[*q];
        assert_eq!(
            word_as_ext(&exec.public_words[2 * q].1).expect("ext"),
            want,
            "query {q}: DEEP at the regular point"
        );
        assert_eq!(
            word_as_ext(&exec.public_words[2 * q + 1].1).expect("ext"),
            want_sym,
            "query {q}: DEEP at the symmetric point"
        );
        if want != FEE::zero() {
            nonzero += 1;
        }
    }
    assert_eq!(
        nonzero,
        all.len(),
        "a vacuously zero reconstruction would make the differential empty"
    );
    println!(
        "joined sub-proof: {} queries, {} instructions, {} distinct indices",
        all.len(),
        program.instrs.len(),
        {
            let mut d = h.iotas.clone();
            d.sort_unstable();
            d.dedup();
            d.len()
        }
    );
}

// =============================================================================
// Cost
// =============================================================================

/// Precomputed columns a table carries IN PRODUCTION.
///
/// `test_utils::production_airs` builds the AIR objects without their
/// preprocessed commitments — the commitments need an ELF, a register file or a
/// page config, none of which a shape census has. `is_preprocessed()` is
/// therefore FALSE on five tables that are preprocessed in the real epoch
/// (`lib.rs`'s `VmAirs::new` wires BITWISE, DECODE, KECCAK_RC, REGISTER and
/// PAGE; `continuation.rs` wires GLOBAL_MEMORY), and reading the flag off these
/// objects would drop one opening group — one leaf hash and one path walk —
/// from each of them.
///
/// The split is what matters here, not the commitment value: a preprocessed
/// table's columns `0..n` are committed in their own tree and the rest in the
/// main tree, so the same columns are hashed as TWO leaves instead of one.
fn production_num_precomputed(
    label: &str,
    air: &dyn stark::traits::AIR<Field = Gl, FieldExtension = Ext3, PublicInputs = ()>,
) -> usize {
    use crate::tables::{bitwise, decode, keccak_rc, page, register};

    let wired = match label {
        "BITWISE" => bitwise::NUM_PRECOMPUTED_COLS,
        "DECODE" => decode::NUM_PRECOMPUTED_COLS,
        "KECCAK_RC" => keccak_rc::NUM_PRECOMPUTED_COLS,
        "REGISTER" => register::NUM_PREPROCESSED_COLS,
        "PAGE" => page::NUM_PREPROCESSED_COLS,
        _ => 0,
    };
    if air.is_preprocessed() {
        // Already wired by the constructor (GLOBAL_MEMORY): trust the object.
        assert_eq!(
            wired, 0,
            "{label} is wired preprocessed AND listed above; one of the two is stale"
        );
        return air.num_precomputed_columns();
    }
    wired
}

/// The DEEP shape and the opening groups of a production AIR, as a sub-proof of
/// `log2_trace_length` rows at `blowup` would carry them.
fn shape_for(
    air: &dyn stark::traits::AIR<Field = Gl, FieldExtension = Ext3, PublicInputs = ()>,
    num_precomputed: usize,
    log2_trace_length: u32,
    log2_blowup: u32,
) -> SubProofShape {
    use stark::constraint_ir::ConstraintArtifact;

    let artifact = ConstraintArtifact::capture(air);
    let layout = V::ood_layout(air);
    let (main_width, aux_width) = air.trace_layout();

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

    SubProofShape {
        deep: super::deep::DeepShape {
            step_size: layout.step_size(),
            num_eval_points: artifact.shape.transition_offsets.len() * layout.step_size(),
            num_total_cols: main_width + aux_width,
            next_row_cols: layout.next_row_cols().to_vec(),
            num_composition_parts: artifact.shape.composition_degree_multiplier as usize,
            log2_trace_length,
        },
        trace_groups,
        merkle_depth: (log2_trace_length + log2_blowup) as usize - 1,
        log2_lde_length: log2_trace_length + log2_blowup,
        coset_offset: FE::from(3u64),
    }
}

fn count<F: Fn(&super::instr::Instr) -> bool>(
    program: &super::compiler::LfmProgram,
    f: F,
) -> usize {
    program.instrs.iter().filter(|i| f(i)).count()
}

fn permutations(program: &super::compiler::LfmProgram) -> usize {
    count(program, |i| matches!(i, super::instr::Instr::KeccakF(_)))
}

/// Byte swaps — one `LFM_BITDEC` row each. Every field element that enters a
/// leaf hash needs one; nothing else in this leg decomposes, except the one
/// index decomposition per query.
fn bit_decs(program: &super::compiler::LfmProgram) -> usize {
    count(program, |i| matches!(i, super::instr::Instr::BitDec { .. }))
}

/// Marginal per-query cost of one shape, by emitting one query and two and
/// differencing — so no per-sub-proof plumbing (the invariants, the OOD grid,
/// the hoisted root unpacks) leaks into the figure.
struct PerQuery {
    instrs: usize,
    perms: usize,
    swaps: usize,
}

fn marginal(shape: &SubProofShape) -> PerQuery {
    let mut one = LfmBuilder::new();
    emit_sub_proof(&mut one, shape, 1);
    let one = compile(one.finish());
    let mut two = LfmBuilder::new();
    emit_sub_proof(&mut two, shape, 2);
    let two = compile(two.finish());
    PerQuery {
        instrs: two.instrs.len() - one.instrs.len(),
        perms: permutations(&two) - permutations(&one),
        swaps: bit_decs(&two) - bit_decs(&one),
    }
}

/// The DEEP fold alone, both points, with no authentication — the same
/// measurement `constraint_tests::deep_leg_cost` reports, repeated here so the
/// two halves of the joined leg can be compared on one line.
fn deep_only_rows(shape: &SubProofShape) -> usize {
    use super::deep::{DeepOpening, emit_deep_invariants, emit_deep_point};

    let d = &shape.deep;
    let plumb = |b: &mut LfmBuilder| {
        let n = 2
            + d.num_eval_points * d.num_total_cols
            + d.num_composition_parts
            + 2 * (d.num_total_cols + d.num_composition_parts)
            + 2;
        let arena = b.declare_arena(n as u32);
        let mut i = 0u32;
        let mut take = |b: &mut LfmBuilder| {
            let c = b.hint_word(arena, i).as_ext();
            i += 1;
            c
        };
        let g = take(b);
        let z = take(b);
        let steps: Vec<Vec<_>> = (0..d.num_eval_points)
            .map(|_| (0..d.num_total_cols).map(|_| take(b)).collect())
            .collect();
        let parts: Vec<_> = (0..d.num_composition_parts).map(|_| take(b)).collect();
        let openings: Vec<(Vec<_>, Vec<_>)> = (0..2)
            .map(|_| {
                (
                    (0..d.num_total_cols).map(|_| take(b)).collect(),
                    (0..d.num_composition_parts).map(|_| take(b)).collect(),
                )
            })
            .collect();
        let points: Vec<_> = (0..2)
            .map(|_| super::builder::Felt(take(b).addr()))
            .collect();
        (g, z, steps, parts, openings, points)
    };

    let mut bare = LfmBuilder::new();
    let _ = plumb(&mut bare);
    let baseline = bare.finish().instrs.len();

    let mut inv_only = LfmBuilder::new();
    let (g, z, steps, parts, _, _) = plumb(&mut inv_only);
    let _ = emit_deep_invariants(&mut inv_only, d, g, z, &steps, &parts);
    let invariant_rows = inv_only.finish().instrs.len() - baseline;

    let mut full = LfmBuilder::new();
    let (g, z, steps, parts, openings, points) = plumb(&mut full);
    let inv = emit_deep_invariants(&mut full, d, g, z, &steps, &parts);
    for (k, (trace, qparts)) in openings.into_iter().enumerate() {
        emit_deep_point(
            &mut full,
            d,
            g,
            &inv,
            &DeepOpening {
                point: points[k],
                trace,
                parts: qparts,
            },
        );
    }
    full.finish().instrs.len() - baseline - invariant_rows
}

/// ★ What the joined leg costs, per query and per epoch, and how the bill
/// splits between folding the values and authenticating them.
///
/// Measured by emitting one query and two and differencing, so the marginal
/// figure carries no per-sub-proof plumbing. Three currencies, because the
/// sizing rule in `others/lfm-target-shape.md` says rows of different chips are
/// not comparable: instructions, keccak permutations, and main-trace CELLS —
/// the last being the only one in which a byteswap and a permutation can be
/// added together.
///
/// ### What this instrument cannot see
///
/// The trace LENGTH of each table in a real epoch. It is workload-dependent and
/// enters only through the Merkle depth (`log2(N·blowup) − 1`), which the walk
/// is linear in, so the line below is parameterised on one uniform length
/// rather than measured. It also cannot see FRI, whose own layer openings are a
/// separate authentication bill this leg does not carry, nor the query COUNT,
/// which is a proof-options property.
#[test]
fn join_leg_cost() {
    /// Queries at blowup 8 — a proof-options property, stated not measured.
    const QUERIES: usize = 73;
    const LOG2_BLOWUP: u32 = 3;
    const LOG2_TRACE: u32 = 20;

    let swap_cells = super::machine_tests::byteswap_cells();
    let perm_cells = super::machine_tests::permutation_cells();

    let opts = stark::proof::options::GoldilocksCubicProofOptions::with_blowup(1 << LOG2_BLOWUP)
        .expect("a power-of-two blowup is valid");
    let airs = crate::test_utils::production_airs(&opts);
    assert_eq!(airs.len(), crate::test_utils::NUM_PRODUCTION_AIRS);

    println!(
        "\nJoined DEEP+authentication, per query, at log2(N) = {LOG2_TRACE}, \
         blowup 2^{LOG2_BLOWUP} (Merkle depth {})",
        LOG2_TRACE + LOG2_BLOWUP - 1
    );
    println!(
        "{:<14} {:>5} {:>4} {:>9} {:>8} {:>7} {:>8} {:>12}",
        "table", "cols", "grp", "instr/qry", "of it DEEP", "perm", "swaps", "cells/qry"
    );

    let mut total_instr = 0usize;
    let mut total_deep = 0usize;
    let mut total_perm = 0usize;
    let mut total_swaps = 0usize;
    for (label, air) in &airs {
        let num_precomputed = production_num_precomputed(label, &**air);
        let shape = shape_for(&**air, num_precomputed, LOG2_TRACE, LOG2_BLOWUP);
        let per = marginal(&shape);
        let deep = deep_only_rows(&shape);

        total_instr += per.instrs;
        total_deep += deep;
        total_perm += per.perms;
        total_swaps += per.swaps;
        println!(
            "{:<14} {:>5} {:>4} {:>9} {:>8} {:>7} {:>8} {:>12}",
            label,
            shape.deep.num_total_cols,
            shape.groups().len(),
            per.instrs,
            deep,
            per.perms,
            per.swaps,
            per.perms as u64 * perm_cells + per.swaps as u64 * swap_cells,
        );
    }

    let total_cells = total_perm as u64 * perm_cells + total_swaps as u64 * swap_cells;
    println!(
        "\nOne query, all {} AIRs: {total_instr} instructions ({total_deep} of \
         them the DEEP fold, {:.1}%), {total_perm} permutations, \
         {total_swaps} byteswaps.",
        airs.len(),
        100.0 * total_deep as f64 / total_instr as f64,
    );
    println!(
        "In main-trace CELLS: {} hashing, {} byteswapping — hashing is {:.1}x \
         the swap bill.",
        total_perm as u64 * perm_cells,
        total_swaps as u64 * swap_cells,
        (total_perm as u64 * perm_cells) as f64 / (total_swaps as u64 * swap_cells) as f64,
    );
    println!(
        "At {QUERIES} queries: {} instructions, {} permutations, {} cells.",
        total_instr * QUERIES,
        total_perm * QUERIES,
        total_cells * QUERIES as u64,
    );

    // ---- what a SHARED commitment would cost, exactly, under one assumption -
    //
    // `others/lfm-team-lead-shared-commitment-ruling.md` parks the lever and
    // pins a prediction of 55-70k permutations, noting that leaf WIDENING under
    // a shared tree is unmeasured and could offset the walk saving. It can be
    // settled without building anything: a permutation count is a function of
    // the shape alone -- `ceil(leaf_bytes / 136)` absorbs plus one per level --
    // so the only thing being assumed is the SHAPE (one tree per sub-proof
    // whose leaf is the four matrices' row pairs concatenated in matrix order).
    // Nothing about the arithmetic is estimated.
    //
    // Widening cannot offset much, and the reason is structural: absorbs scale
    // with total bytes, which do not change when the matrices share a leaf,
    // while walks scale with the number of TREES, which is what collapses. The
    // only bytes lost are the per-leaf padding of the groups that disappear.
    const RATE_BYTES: usize = 136;
    let mut shared_perm = 0usize;
    for (label, air) in &airs {
        let num_precomputed = production_num_precomputed(label, &**air);
        let shape = shape_for(&**air, num_precomputed, LOG2_TRACE, LOG2_BLOWUP);
        let leaf_bytes: usize = shape.groups().iter().map(GroupShape::leaf_bytes).sum();
        shared_perm += leaf_bytes.div_ceil(RATE_BYTES) + shape.merkle_depth;
    }
    println!(
        "\nOne shared tree per sub-proof instead of four: {shared_perm} \
         permutations per query against {total_perm} ({:.0}% collapse), \
         {} per epoch at {QUERIES} queries.",
        100.0 * (1.0 - shared_perm as f64 / total_perm as f64),
        shared_perm * QUERIES,
    );
    println!(
        "Absorbs are {} of the shared figure and walks {}; widening costs \
         nothing here because total leaf BYTES do not change when matrices \
         share a leaf -- only the padding of the vanished leaves.",
        shared_perm - airs.len() * (LOG2_TRACE + LOG2_BLOWUP - 1) as usize,
        airs.len() * (LOG2_TRACE + LOG2_BLOWUP - 1) as usize,
    );
}

// =============================================================================
// Falsification: the join, and the two attacks it denies
// =============================================================================

use super::builder::{Bit, Cell, Ext, Felt};
use super::deep::{DeepOpening, emit_deep_invariants, emit_deep_point};
use super::proof::{lfm_prove, verify_against};
use super::registry::build_artifacts;
use super::sub_proof::{
    GroupCommitment, GroupOpening, emit_group_authentication, emit_query_points,
};

pub(super) fn prove_options() -> stark::proof::options::ProofOptions {
    stark::proof::options::GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 is valid")
}

/// How a control program differs from the joined one. Each variant is an
/// attack surface the join closes, built so the attack can be RUN rather than
/// argued about.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Control {
    /// DEEP folds a second arena instead of the authenticated cells — the
    /// "two parallel copies" shape.
    SplitValues,
    /// The query point is hinted instead of derived from the walk's own index
    /// bits, so the leaf may be authenticated at one index and folded at
    /// another point.
    HintedPoint,
}

/// A one-query program in one of the control shapes.
///
/// Deliberately NOT a variant of [`emit_sub_proof`]: the production emitter has
/// no switch that could produce these, and giving it one would be a runtime
/// off-switch on a soundness obligation. This is a test artifact that exists to
/// be attacked.
///
/// Arenas: the joined program's five, plus one extra carrying whatever the
/// control decouples.
fn control_program_source(
    shape: &SubProofShape,
    control: Control,
) -> super::builder::LfmProgramSource {
    let mut b = LfmBuilder::new();
    let groups = shape.groups();

    let uniforms = b.declare_arena(2);
    let ood = b.declare_arena((shape.deep.num_eval_points * shape.deep.num_total_cols) as u32);
    let parts_arena = b.declare_arena(shape.deep.num_composition_parts as u32);
    let roots = b.declare_arena(2 * groups.len() as u32);
    let queries = b.declare_arena(shape.query_words() as u32);
    let extra = b.declare_arena(match control {
        // A second copy of every folded value, both points.
        Control::SplitValues => {
            2 * (shape.deep.num_total_cols + shape.deep.num_composition_parts) as u32
        }
        // The two points.
        Control::HintedPoint => 2,
    });

    let gamma = b.hint_word(uniforms, 0).as_ext();
    let zeta = b.hint_word(uniforms, 1).as_ext();
    let mut next = 0u32;
    let ood_steps: Vec<Vec<Ext>> = (0..shape.deep.num_eval_points)
        .map(|_| {
            (0..shape.deep.num_total_cols)
                .map(|_| {
                    let c = b.hint_word(ood, next).as_ext();
                    next += 1;
                    c
                })
                .collect()
        })
        .collect();
    let claimed_parts: Vec<Ext> = (0..shape.deep.num_composition_parts as u32)
        .map(|j| b.hint_word(parts_arena, j).as_ext())
        .collect();
    let commitments: Vec<GroupCommitment> = groups
        .iter()
        .enumerate()
        .map(|(i, g)| GroupCommitment::hint(&mut b, roots, 2 * i as u32, *g))
        .collect();
    let inv = emit_deep_invariants(&mut b, &shape.deep, gamma, zeta, &ood_steps, &claimed_parts);

    let mut cursor = 0u32;
    let index = b.hint_felt(queries, cursor);
    cursor += 1;
    let openings: Vec<GroupOpening> = groups
        .iter()
        .map(|g| {
            let values: Vec<Cell> = (0..g.num_values())
                .map(|_| {
                    let c = b.hint_word(queries, cursor);
                    cursor += 1;
                    c
                })
                .collect();
            let siblings: Vec<[Cell; 2]> = (0..shape.merkle_depth)
                .map(|_| {
                    let lo = b.hint_word(queries, cursor);
                    let hi = b.hint_word(queries, cursor + 1);
                    cursor += 2;
                    [lo, hi]
                })
                .collect();
            GroupOpening { values, siblings }
        })
        .collect();

    let bits: Vec<Bit> = b.bit_dec(index, shape.merkle_depth);
    for (commitment, opening) in commitments.iter().zip(&openings) {
        emit_group_authentication(&mut b, commitment, opening, &bits);
    }

    let (point, point_sym) = match control {
        Control::HintedPoint => (
            Felt(b.hint_word(extra, 0).addr()),
            Felt(b.hint_word(extra, 1).addr()),
        ),
        Control::SplitValues => emit_query_points(&mut b, shape, &bits),
    };

    let read = |b: &mut LfmBuilder, k: usize, point: Felt| -> DeepOpening {
        let width = shape.deep.num_total_cols + shape.deep.num_composition_parts;
        let base = (k * width) as u32;
        let (trace, parts): (Vec<Ext>, Vec<Ext>) = match control {
            Control::SplitValues => (
                (0..shape.deep.num_total_cols)
                    .map(|c| b.hint_word(extra, base + c as u32).as_ext())
                    .collect(),
                (0..shape.deep.num_composition_parts)
                    .map(|j| {
                        b.hint_word(extra, base + (shape.deep.num_total_cols + j) as u32)
                            .as_ext()
                    })
                    .collect(),
            ),
            Control::HintedPoint => {
                let mut trace = Vec::new();
                for (opening, g) in openings.iter().zip(&groups).take(shape.trace_groups.len()) {
                    for c in 0..g.num_columns {
                        trace.push(opening.values[k * g.num_columns + c].as_ext());
                    }
                }
                let parts_opening = openings.last().expect("parts");
                let np = shape.deep.num_composition_parts;
                (
                    trace,
                    (0..np)
                        .map(|j| parts_opening.values[k * np + j].as_ext())
                        .collect(),
                )
            }
        };
        DeepOpening {
            point,
            trace,
            parts,
        }
    };
    let regular = read(&mut b, 0, point);
    let symmetric = read(&mut b, 1, point_sym);
    let got = emit_deep_point(&mut b, &shape.deep, gamma, &inv, &regular);
    let got_sym = emit_deep_point(&mut b, &shape.deep, gamma, &inv, &symmetric);
    b.public(got.as_cell());
    b.public(got_sym.as_cell());
    b.finish()
}

impl HostSubProof {
    /// The values one query folds, in the order a [`Control::SplitValues`]
    /// program reads them: the regular point's trace then parts, then the
    /// symmetric point's.
    fn split_values(&self, q: usize) -> Vec<LfmWord> {
        let groups = self.shape.groups();
        let mut out = Vec::new();
        for k in 0..ROWS_PER_LEAF {
            for (opening, g) in self.openings[q]
                .iter()
                .zip(&groups)
                .take(self.shape.trace_groups.len())
            {
                out.extend(&opening.values[k * g.num_columns..(k + 1) * g.num_columns]);
            }
            let parts = self.openings[q].last().expect("parts");
            let np = self.shape.deep.num_composition_parts;
            out.extend(&parts.values[k * np..(k + 1) * np]);
        }
        out
    }
}

/// ★ The joined program authenticates and folds under a real PROOF, not just
/// an execution.
///
/// Method rule 2: the executor mirrors the ALU it is checking, so nothing run
/// so far says the CHIPS agree. One query, because the whole point of this test
/// is the chips and 219 of them would only repeat the same rows.
#[test]
fn the_join_proves_and_verifies() {
    let h = host_sub_proof();
    let opts = prove_options();
    let queries = [0usize];

    let mut b = LfmBuilder::new();
    let (_, outs) = emit_sub_proof(&mut b, &h.shape, queries.len());
    for (p, s) in &outs {
        b.public(p.as_cell());
        b.public(s.as_cell());
    }
    let program = compile(b.finish());
    let artifacts = build_artifacts(&program, &opts);
    let proved = lfm_prove(&program, &artifacts, &h.arenas(&queries), &opts)
        .expect("the joined sub-proof must prove");

    let (want, want_sym) = h.expected[queries[0]];
    assert_eq!(
        word_as_ext(&proved.public_words[0].1).expect("ext"),
        want,
        "the proved run must publish the production reconstruction"
    );
    assert_eq!(
        word_as_ext(&proved.public_words[1].1).expect("ext"),
        want_sym
    );
    assert!(
        verify_against(
            &artifacts.roots,
            &artifacts.program_id,
            artifacts.keccak_rnd_chunks,
            &proved.proof,
            &proved.public_words,
            &opts,
        ),
        "the joined run must verify"
    );
}

/// ★ The join, stated as the property it exists for: there is no arena the
/// prover can move that changes the folded value without breaking the
/// authentication.
///
/// Every opened value of one query is tampered, one at a time, and each vector
/// is run BOTH ways. Incoherent (claim the real root) must not execute.
/// Coherent (also claim the root the tampered leaf really folds to, so nothing
/// in the run is inconsistent) must execute — and then publish a DEEP value
/// that is not the production one, against a root that is not the committed
/// one. A prover who wants the wrong fold must pay with the wrong root.
///
/// Run over BOTH fixtures. The three-matrix one is the real production table;
/// the four-matrix one is the only fixture in which the precomputed group's own
/// leaf and path are ever tampered, and its 64-row trace is the only executed
/// walk in this file deeper than two levels.
#[test]
fn no_tampered_value_can_move_the_fold_without_moving_the_root() {
    sweep_tampers(host_sub_proof(), "L2G_MEMORY (3 matrices, depth 2)");
    sweep_tampers(
        preprocessed_sub_proof(),
        "PREPROCESSED_FIXTURE (4 matrices, depth 6)",
    );
}

fn sweep_tampers(h: &HostSubProof, label: &str) {
    use super::proof_arena::{commitments_to_arena, walk_to_root};

    let q = 0usize;
    let groups = h.shape.groups();

    let mut b = LfmBuilder::new();
    let (_, outs) = emit_sub_proof(&mut b, &h.shape, 1);
    for (p, s) in &outs {
        b.public(p.as_cell());
        b.public(s.as_cell());
    }
    let program = compile(b.finish());
    let honest = execute(&program, &h.arenas(&[q]), &TestPermutation).expect("honest");

    // Sweep every value slot of every group, so no vector class (first group,
    // first column, regular point) is silently the only one tested.
    let mut vectors = 0usize;
    for (g, group) in groups.iter().enumerate() {
        for slot in 0..group.num_values() {
            let mut arenas = h.arenas(&[q]);
            let word_of_slot = {
                // Offset of this group's value `slot` inside the query arena.
                let mut off = 1usize;
                for prior in groups.iter().take(g) {
                    off += prior.num_values() + 2 * h.shape.merkle_depth;
                }
                off + slot
            };
            arenas[4][word_of_slot][0] += FE::one();

            // Incoherent: the real roots, a moved leaf.
            let err = execute(&program, &arenas, &TestPermutation)
                .err()
                .unwrap_or_else(|| {
                    panic!("{label}: group {g} slot {slot}: a moved value must not authenticate")
                });

            // Coherent: recompute the leaf the tampered values really give and
            // the root that leaf really reaches, using PRODUCTION's hashers.
            let leaf = tampered_leaf(h, q, g, slot);
            let forged = walk_to_root(leaf, h.iotas[q], &h.openings[q][g].siblings);
            assert_ne!(
                forged, h.roots[g],
                "{label}: group {g} slot {slot}: the tamper must move the root, or the \
                 vector is vacuous"
            );
            let mut coherent_roots = h.roots.clone();
            coherent_roots[g] = forged;
            arenas[3] = commitments_to_arena(&coherent_roots);
            let forged_run = execute(&program, &arenas, &TestPermutation).unwrap_or_else(|e| {
                panic!("{label}: group {g} slot {slot}: the coherent forgery must execute: {e:?}")
            });
            // Which of the two points moves is not incidental: a leaf holds
            // the row PAIR, its first half is the regular point and its second
            // the symmetric, and folding the halves into the wrong point is a
            // mistake no root check would catch. Asserting exactly one moved,
            // and which, is what pins that split.
            let moved = [
                forged_run.public_words[0].1 != honest.public_words[0].1,
                forged_run.public_words[1].1 != honest.public_words[1].1,
            ];
            let regular_half = slot < group.num_columns;
            assert_eq!(
                moved,
                [regular_half, !regular_half],
                "{label}: group {g} slot {slot}: a value in the leaf's {} half must move \
                 DEEP at {} and nothing else",
                if regular_half { "first" } else { "second" },
                if regular_half {
                    "the regular point"
                } else {
                    "-v"
                },
            );
            if vectors == 0 {
                println!("first incoherent rejection: {err:?}");
            }
            vectors += 1;
        }
    }
    println!("{label}: {vectors} tamper vectors, every value slot of every group, both ways round");

    // ---- the index, which this leg binds to the POINT as well as the leaf --
    //
    // R1f authenticated a leaf at an index; here the same bits also derive the
    // evaluation point, so moving the index has to move the reconstruction as
    // well as the walk. A padding-heavy table can have several indices that
    // authenticate (identical rows give identical leaves), which would make the
    // walk half of this vector vacuous — so that is asserted, not assumed.
    for level in 0..h.shape.merkle_depth {
        let bad = h.iotas[q] ^ (1 << level);
        let mut arenas = h.arenas(&[q]);
        arenas[4][0] = base_word(FE::from(bad as u64));

        let mut moved_a_root = false;
        let mut coherent_roots = h.roots.clone();
        for (g, group) in groups.iter().enumerate() {
            let words = &h.openings[q][g].values;
            let leaf = if group.is_ext {
                type ExtBackend = stark::config::BatchedMerkleTreeBackend<Ext3>;
                let v: Vec<FEE> = words.iter().map(|w| FEE::new([w[0], w[1], w[2]])).collect();
                ExtBackend::hash_data_from_slices(&v, &[])
            } else {
                type BaseBackend = stark::config::BatchedMerkleTreeBackend<Gl>;
                let v: Vec<FE> = words.iter().map(|w| w[0]).collect();
                BaseBackend::hash_data_from_slices(&v, &[])
            };
            coherent_roots[g] = walk_to_root(leaf, bad, &h.openings[q][g].siblings);
            moved_a_root |= coherent_roots[g] != h.roots[g];
        }
        assert!(
            moved_a_root,
            "{label}: flipping index bit {level} left every root unchanged — the fixture's \
             trees are degenerate at this index and the walk half of this vector \
             tests nothing"
        );
        execute(&program, &arenas, &TestPermutation)
            .err()
            .unwrap_or_else(|| {
                panic!("{label}: index bit {level}: a moved index must not authenticate")
            });

        arenas[3] = commitments_to_arena(&coherent_roots);
        let forged = execute(&program, &arenas, &TestPermutation).unwrap_or_else(|e| {
            panic!("{label}: index bit {level}: coherent forgery must execute: {e:?}")
        });
        assert_ne!(
            forged.public_words[0].1, honest.public_words[0].1,
            "{label}: index bit {level}: the index derives the evaluation point, so a \
             forged walk at another index must also fold at another point"
        );
    }
    println!(
        "{label}: {} index vectors, one per level",
        h.shape.merkle_depth
    );

    // ---- a sibling, at every level -------------------------------------
    for level in 0..h.shape.merkle_depth {
        let mut siblings = h.openings[q][0].siblings.clone();
        siblings[level][0] ^= 1;
        let mut arenas = h.arenas(&[q]);
        let base = 1 + groups[0].num_values();
        arenas[4][base..base + 2 * h.shape.merkle_depth]
            .copy_from_slice(&commitments_to_arena(&siblings));
        execute(&program, &arenas, &TestPermutation)
            .err()
            .unwrap_or_else(|| {
                panic!("{label}: sibling level {level}: a moved path must not authenticate")
            });
    }
    println!(
        "{label}: {} sibling vectors, one per level",
        h.shape.merkle_depth
    );

    /// The leaf hash a tampered opening really produces, under production's own
    /// backend rather than a local model.
    fn tampered_leaf(h: &HostSubProof, q: usize, g: usize, slot: usize) -> Commitment {
        type BaseBackend = stark::config::BatchedMerkleTreeBackend<Gl>;
        type ExtBackend = stark::config::BatchedMerkleTreeBackend<Ext3>;
        let group = h.shape.groups()[g];
        let words = &h.openings[q][g].values;
        if group.is_ext {
            let mut v: Vec<FEE> = words.iter().map(|w| FEE::new([w[0], w[1], w[2]])).collect();
            v[slot] = &v[slot] + FEE::new([FE::one(), FE::zero(), FE::zero()]);
            ExtBackend::hash_data_from_slices(&v, &[])
        } else {
            let mut v: Vec<FE> = words.iter().map(|w| w[0]).collect();
            v[slot] += FE::one();
            BaseBackend::hash_data_from_slices(&v, &[])
        }
    }
}

/// ★ The two attacks the join denies, RUN against control programs that permit
/// them.
///
/// A join is a negative claim — "these cannot disagree" — and a negative claim
/// is only worth what its counterexample is worth. So each control is the
/// joined program with exactly one link cut, fed inputs that are honest
/// everywhere else, and each one accepts a reconstruction the production
/// verifier would not have produced. That is the thing the joined program has
/// to refuse, and the test above shows it does.
#[test]
fn the_controls_show_what_the_join_denies() {
    let h = host_sub_proof();
    let q = 0usize;

    // ---- Control 1: DEEP folds a parallel copy. -------------------------
    let program = compile(control_program_source(&h.shape, Control::SplitValues));
    validate(&program).expect("admissible");
    let mut arenas = h.arenas(&[q]);
    arenas.push(h.split_values(q));
    let clean = execute(&program, &arenas, &TestPermutation)
        .expect("the control must accept honest inputs");
    assert_eq!(
        word_as_ext(&clean.public_words[0].1).expect("ext"),
        h.expected[q].0,
        "the control must agree with production before it is attacked, or the \
         attack below proves nothing"
    );

    let mut attacked = arenas.clone();
    attacked[5][0][0] += FE::one();
    let forged = execute(&program, &attacked, &TestPermutation).expect(
        "SplitValues: authenticating one set of values and folding another is \
         exactly what this control permits",
    );
    assert_ne!(
        word_as_ext(&forged.public_words[0].1).expect("ext"),
        h.expected[q].0,
        "the attack must actually move the reconstruction"
    );
    println!("SplitValues control: forged fold accepted against honest roots");

    // ---- Control 2: the query point is hinted. --------------------------
    // Two queries with DIFFERENT indices: authenticate one, fold at the
    // other's point.
    let other = (0..h.iotas.len())
        .find(|&i| h.iotas[i] != h.iotas[q])
        .expect("the fixture must carry two distinct query indices");
    let program = compile(control_program_source(&h.shape, Control::HintedPoint));
    validate(&program).expect("admissible");
    let mut arenas = h.arenas(&[q]);
    arenas.push(vec![base_word(h.points[q].0), base_word(h.points[q].1)]);
    let clean = execute(&program, &arenas, &TestPermutation).expect("honest");
    assert_eq!(
        word_as_ext(&clean.public_words[0].1).expect("ext"),
        h.expected[q].0
    );

    let mut attacked = arenas.clone();
    attacked[5] = vec![base_word(h.points[other].0), base_word(h.points[other].1)];
    let forged = execute(&program, &attacked, &TestPermutation).expect(
        "HintedPoint: a hinted point is not tied to the authenticated index, \
         which is what this control permits",
    );
    assert_ne!(
        word_as_ext(&forged.public_words[0].1).expect("ext"),
        h.expected[q].0,
        "folding query {q}'s values at query {other}'s point must give a \
         different answer"
    );
    assert_ne!(
        word_as_ext(&forged.public_words[0].1).expect("ext"),
        h.expected[other].0,
        "and it must not accidentally be the other query's answer either"
    );
    println!(
        "HintedPoint control: query {q}'s leaf authenticated, folded at query \
         {other}'s point, accepted"
    );
}

// =============================================================================
// The degenerate parameter this leg introduced: the precomputed group
// =============================================================================

/// A PREPROCESSED sub-proof, so the four-group shape is exercised.
///
/// L2G_MEMORY — the fixture everything above runs on — is not preprocessed, and
/// neither is any AIR a single-table proof can cheaply be built from: the real
/// preprocessed tables are BITWISE (2^20 rows), DECODE, KECCAK_RC, REGISTER and
/// PAGE. So on that fixture the precomputed group is ABSENT, DEEP's column
/// order `precomputed ‖ main ‖ aux` degenerates to `main ‖ aux`, and an emitter
/// that put main first would pass every test in this file. That is the same
/// hazard as `step_size = 1` and it needs the same answer: a case production
/// does not produce.
///
/// Built the way `tests::bitwise_tests` builds its preprocessed receiver — a
/// small `AirWithBuses` whose commitment comes from the prover's own
/// `compute_precomputed_commitment_for_testing`, so the precomputed root in the
/// proof and the one the AIR declares are computed by the same code the real
/// tables use. Widths are 2 and 2 so the two base groups can be SWAPPED, which
/// the falsification half needs.
fn preprocessed_fixture() -> (
    super::constraint_tests::BoxedAir,
    stark::proof::stark::MultiProof<Gl, Ext3, ()>,
) {
    use crate::tables::types::{BusId, alu_op};
    use crate::test_utils::multi_prove_ram;
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use stark::lookup::{
        AirWithBuses, AuxiliaryTraceBuildData, BusInteraction, BusValue, Multiplicity,
        NullBoundaryConstraintBuilder, Packing,
    };
    use stark::prover::IsStarkProver;
    use stark::trace::TraceTable;
    use stark::traits::AIR;

    /// Columns 0..3 are precomputed (x, y, x&y); 3..6 are the multiplicity
    /// block (a copy of x&y, a spare, and the bus multiplicity). The copy is
    /// there so the table carries a real TRANSITION constraint: `EmptyConstraints`
    /// leaves a single coefficient in the run and `open_sub_proof` recovers
    /// `beta` from its second element.
    const NUM_COLS: usize = 6;
    const NUM_PRECOMPUTED: usize = 3;
    /// 64 rows, not the 4 the other fixture uses. Trace length only enters this
    /// leg through the Merkle depth, and at 4 rows every executed walk in the
    /// suite is two levels deep — enough to hide a level-count error. 64 rows at
    /// blowup 2 gives depth 6, which is the only executed multi-level walk over
    /// all four committed matrices.
    const NUM_ROWS: usize = 64;

    let opts = stark::proof::options::GoldilocksCubicProofOptions::with_blowup(2)
        .expect("blowup=2 is valid");

    let build = |commitment: Option<stark::config::Commitment>| {
        let air = AirWithBuses::<Gl, Ext3, NullBoundaryConstraintBuilder, (), CopiedColumn>::new(
            NUM_COLS,
            AuxiliaryTraceBuildData {
                interactions: vec![BusInteraction::receiver(
                    BusId::ByteAlu,
                    // The multiplicity is the LAST column, past the precomputed
                    // block — the production split (`0..n` precomputed, the
                    // rest multiplicities).
                    Multiplicity::Column(5),
                    vec![
                        BusValue::constant(alu_op::AND as u64),
                        BusValue::Packed {
                            start_column: 0,
                            packing: Packing::Direct,
                        },
                        BusValue::Packed {
                            start_column: 1,
                            packing: Packing::Direct,
                        },
                        BusValue::Packed {
                            start_column: 2,
                            packing: Packing::Direct,
                        },
                    ],
                )],
            },
            &opts,
            1,
            CopiedColumn,
        )
        .with_name("PREPROCESSED_FIXTURE");
        match commitment {
            Some(c) => air.with_preprocessed(c, NUM_PRECOMPUTED),
            None => air,
        }
    };

    // Distinct rows, so the committed leaves are distinct and the tree is not
    // the degenerate one R1f warns about.
    let make_trace = || {
        let mut data = vec![FE::zero(); NUM_ROWS * NUM_COLS];
        for r in 0..NUM_ROWS {
            let x = 5u64 + r as u64;
            let y = 3u64 + 2 * r as u64;
            data[r * NUM_COLS] = FE::from(x);
            data[r * NUM_COLS + 1] = FE::from(y);
            data[r * NUM_COLS + 2] = FE::from(x & y);
            data[r * NUM_COLS + 3] = FE::from(x & y);
            data[r * NUM_COLS + 5] = FE::one();
        }
        TraceTable::<Gl, Ext3>::new_main(data, NUM_COLS, 1)
    };

    let trace = make_trace();
    let commitment = <stark::prover::Prover<Gl, Ext3, ()> as IsStarkProver<Gl, Ext3, ()>>::
        compute_precomputed_commitment_for_testing(&trace, &build(None), NUM_PRECOMPUTED)
        .expect("the precomputed columns commit");

    let air = build(Some(commitment));
    let mut trace = make_trace();
    let pairs: Vec<(
        &dyn AIR<Field = Gl, FieldExtension = Ext3, PublicInputs = ()>,
        _,
        _,
    )> = vec![(&air, &mut trace, &())];
    let proof = multi_prove_ram(pairs, &mut DefaultTranscript::<Ext3>::new(&[]))
        .expect("the preprocessed fixture must prove");
    (Box::new(air), proof)
}

/// `main[3] == main[2]` — one transition constraint, satisfied by the fixture
/// trace, spanning the precomputed/multiplicity boundary.
struct CopiedColumn;

impl<F: math::field::traits::IsField, E: math::field::traits::IsField>
    stark::constraints::builder::ConstraintSet<F, E> for CopiedColumn
{
    fn eval<B: stark::constraints::builder::ConstraintBuilder<F, E>>(&self, b: &mut B) {
        let precomputed_and = b.main(0, 2);
        let copied_and = b.main(0, 3);
        b.emit_base(0, copied_and - precomputed_and);
    }
}

fn preprocessed_sub_proof() -> &'static HostSubProof {
    use std::sync::OnceLock;
    static CELL: OnceLock<HostSubProof> = OnceLock::new();
    CELL.get_or_init(|| {
        let (air, proof) = preprocessed_fixture();
        build_host_sub_proof(&*air, &proof)
    })
}

/// ★ The four-group shape, and the witness that the group ORDER is
/// load-bearing.
///
/// Both halves, as the degenerate-parameter rule requires. The machine
/// reproduces the production reconstruction on a proof that HAS a precomputed
/// group — and production's own reconstruction, handed the same two base
/// groups in the opposite order, gives a DIFFERENT answer. Without the second
/// half this test would pass against a main-first emitter, which is the exact
/// failure mode it exists to prevent.
#[test]
fn the_precomputed_group_comes_first_and_that_is_checkable() {
    let h = preprocessed_sub_proof();
    let groups = h.shape.groups();

    assert_eq!(
        groups.len(),
        4,
        "the point of this fixture is a sub-proof with all four committed \
         matrices; got {groups:?}"
    );
    assert_eq!(h.shape.trace_groups[0].num_columns, 3, "precomputed width");
    assert!(!h.shape.trace_groups[0].is_ext);
    assert_eq!(h.shape.trace_groups[1].num_columns, 3, "main width");
    assert_eq!(
        h.shape.trace_groups[0].num_columns, h.shape.trace_groups[1].num_columns,
        "the two base groups must be the same width, or the swap below is not \
         a well-formed alternative reading"
    );
    println!(
        "preprocessed fixture: groups {:?}, depth {}, {} queries",
        groups
            .iter()
            .map(|g| (g.num_columns, g.is_ext))
            .collect::<Vec<_>>(),
        h.shape.merkle_depth,
        h.iotas.len()
    );

    // ---- half one: the machine agrees with production. -------------------
    let queries: Vec<usize> = (0..h.iotas.len().min(16)).collect();
    let mut b = LfmBuilder::new();
    let (_, outs) = emit_sub_proof(&mut b, &h.shape, queries.len());
    for (p, s) in &outs {
        b.public(p.as_cell());
        b.public(s.as_cell());
    }
    let program = compile(b.finish());
    validate(&program).expect("admissible");
    let exec = execute(&program, &h.arenas(&queries), &TestPermutation)
        .expect("the four-group sub-proof must authenticate and fold");
    for (k, q) in queries.iter().enumerate() {
        assert_eq!(
            word_as_ext(&exec.public_words[2 * k].1).expect("ext"),
            h.expected[*q].0,
            "query {q}: DEEP at the regular point"
        );
        assert_eq!(
            word_as_ext(&exec.public_words[2 * k + 1].1).expect("ext"),
            h.expected[*q].1,
            "query {q}: DEEP at the symmetric point"
        );
    }

    // ---- half two: the swapped reading DISAGREES. ------------------------
    //
    // Asked of production's own reconstruction, not of a model of it: hand it
    // the main slice where the precomputed one belongs and vice versa. If that
    // came out equal, the order would be unobservable and this fixture would be
    // no witness at all.
    let swapped = &h.expected_base_swapped;
    assert_eq!(
        swapped.len(),
        h.expected.len(),
        "the swapped reading must have been computed for this fixture"
    );
    let mut differs = 0usize;
    for q in &queries {
        if swapped[*q].0 != h.expected[*q].0 || swapped[*q].1 != h.expected[*q].1 {
            differs += 1;
        }
    }
    assert_eq!(
        differs,
        queries.len(),
        "swapping the precomputed and main slices must change the \
         reconstruction at every query, or the column order is not observable \
         on this fixture and it witnesses nothing"
    );
    println!(
        "column order is load-bearing: the swapped reading differs at all {} \
         checked queries",
        queries.len()
    );
}

/// ⚠ The FRI leg's instrument problem, pinned: the proof fixture carries ZERO
/// committed FRI layers, so a differential over it cannot see the fold loop,
/// the per-layer walks, or the terminal check.
///
/// `FriFoldLayout::new(lde_log, blowup_log, k)` sets
/// `terminal_log = min(blowup_log + k, lde_log)` and
/// `num_committed = (lde_log - terminal_log) - 1`. The fixture is the `min`
/// preset — blowup 2 (`blowup_log = 1`), `fri_final_poly_log_degree = 7` — over
/// an epoch of 2^4 steps, so its sub-proof has `log2(lde) = 3` and
/// `terminal_log = min(8, 3) = 3`: no folds at all, and `query_phase` returns
/// the empty-decommitment branch.
///
/// This is the degenerate-parameter rule in its most extreme form. Not "one
/// value hides a difference between two implementations" but "the production
/// instance exercises none of the mechanism". The assertions below are still
/// exactly true of THIS fixture, and this test still earns its place: the day
/// the fixture grows and starts folding, the change is announced rather than
/// silently altering what the FRI tests cover.
///
/// ## ⚠ CORRECTION — the conclusion drawn from this was wrong
///
/// This test's original text went on to say that no amount of care with real
/// data could repair the gap, and that the FRI leg's primary instrument had to
/// be SYNTHETIC codewords. That is false, and the counterexample is one line of
/// the fixture: the trace is `boundaries.len().next_power_of_two()`
/// (`local_to_global.rs:269`), and `num_committed = trace_bits − 8`. So the same
/// construction with 512, 1024 or 2048 boundaries yields real production proofs
/// with one, two or three committed layers — real roots, real paths, real
/// terminal coefficients, real folding challenges — in under a second each.
/// `fri_tests::the_real_prover_folds_and_the_layer_count_follows_the_row_count`
/// is that sweep, and the FRI leg is differentialled entirely against real
/// proofs. Nothing in it is synthetic.
///
/// The lesson is narrower than the one first drawn here. "The production
/// instances all share a degenerate parameter" was a claim about the fixtures on
/// hand, not about the prover, and the two are not the same claim. Worth
/// checking which one is being made before concluding that real data cannot
/// reach a mechanism.
///
/// The zero-layer case is not merely an artifact to route around, either — it
/// is a real production path (small tables fold no further than their terminal)
/// and the emitted verifier handles it as a first-class shape.
#[test]
fn the_fixture_carries_no_fri_layers_so_it_cannot_witness_the_fold() {
    let (_air, proof) = real_fixture();
    assert_eq!(
        proof.proofs.len(),
        1,
        "the join fixture is a single sub-proof"
    );
    let p = &proof.proofs[0];
    println!(
        "fixture sub-proof: fri_layers_merkle_roots = {}, fri_final_poly_coeffs = {}, \
         query decommitments = {}",
        p.fri_layers_merkle_roots.len(),
        p.fri_final_poly_coeffs.len(),
        p.deep_poly_openings.len(),
    );
    assert_eq!(
        p.fri_layers_merkle_roots.len(),
        0,
        "the fixture is expected to carry no committed FRI layers; if it now \
         folds, the FRI leg's coverage story changed and its synthetic sweep \
         should be re-justified against what the real proof now exercises"
    );
    // The coefficient count is `2^effective_k` with
    // `effective_k = terminal_log - blowup_log = 3 - 1`. Checking it is what
    // says the layout arithmetic above is read correctly rather than merely
    // asserted: a wrong reading of `FriFoldLayout` would land on a different
    // power of two here.
    assert_eq!(
        p.fri_final_poly_coeffs.len(),
        4,
        "terminal codeword encodes a degree-<2^2 polynomial at this shape"
    );
}

/// ★ The bits handed to a later leg are the cells the WALK ITSELF consumed.
///
/// The FRI leg reuses a query's index per layer (leaf position `index >> 1`,
/// partner `index ^ 1`, halving each layer). Were it to decompose its own copy
/// it would authenticate at one index and fold at another — the gap this module
/// closes, reopened one level up.
///
/// ## Why this is not the obvious test
///
/// The obvious test compares the program `emit_sub_proof` emits against the one
/// `emit_sub_proof_with_bits` emits and asserts they are identical. That test is
/// VACUOUS and I wrote it before catching it: `emit_sub_proof` is implemented by
/// delegating to `emit_sub_proof_with_bits`, so the two sides are the same
/// program by construction and any defect lands on both and cancels. Injecting a
/// second `bit_dec` — the precise failure this is meant to deny — left it green.
///
/// What discriminates is an ABSOLUTE property rather than a relative one: every
/// returned bit must be consumed by a `Select`. The walk selects sibling order
/// on each bit and `pow_bits` selects the point factors on the same bits, so a
/// bit the emitter actually used is necessarily read by one. A freshly
/// decomposed second copy would be read by nothing.
#[test]
fn the_exposed_bits_are_the_cells_the_walk_consumed() {
    let h = host_sub_proof();
    const QUERIES: usize = 3;

    let mut b = LfmBuilder::new();
    let (_, out) = emit_sub_proof_with_bits(&mut b, &h.shape, QUERIES);
    let src = b.finish();
    assert_eq!(out.len(), QUERIES);

    // Every address any Select reads as its selector.
    let selector_bits: std::collections::HashSet<u64> = src
        .instrs
        .iter()
        .filter_map(|i| match i {
            super::instr::Instr::Select { bit, .. } => Some(bit.0),
            _ => None,
        })
        .collect();
    assert!(
        !selector_bits.is_empty(),
        "the walk and the point derivation both select on bits; an empty set \
         means this test is looking at the wrong instruction"
    );

    for (q, output) in out.iter().enumerate() {
        assert_eq!(
            output.bits.len(),
            h.shape.merkle_depth,
            "query {q}: one bit per Merkle level"
        );
        for (level, bit) in output.bits.iter().enumerate() {
            assert!(
                selector_bits.contains(&bit.0.0),
                "query {q} level {level}: the returned bit is read by no Select, \
                 so it is not a cell the walk or the point derivation used — a \
                 second decomposition of the index has been handed out"
            );
        }
    }
}

// ==================== FRI slice 1: the fold layout ====================

use super::fri::FriShape;

/// ★ The shape mirror against production's observable BEHAVIOUR on the real
/// proof — the vector lengths the verifier structurally enforces.
///
/// `FriFoldLayout` is `pub(crate)` inside `crypto/stark`, so the mirror cannot
/// be compared against the struct. It is compared against what a real proof
/// actually carries instead, which is the better oracle: `verifier.rs:426-448`
/// rejects on exactly these two lengths before its query loop runs, and the
/// spec notes they are the ONLY thing pinning vectors Fiat-Shamir does not bind.
#[test]
fn the_fri_shape_predicts_the_real_proofs_vector_lengths() {
    let (_air, proof) = real_fixture();
    let h = host_sub_proof();
    let opts = prove_options();
    let shape = FriShape::from_options(&opts, h.shape.log2_lde_length);
    shape.check();

    let p = &proof.proofs[0];
    println!(
        "FRI shape: lde 2^{}, blowup 2^{}, k {}, terminal_log {}, total_folds {}, \
         committed {}, coeffs {}",
        shape.log2_lde_length,
        shape.blowup_log,
        shape.final_poly_log_degree,
        shape.terminal_log(),
        shape.total_folds(),
        shape.num_committed(),
        shape.num_terminal_coeffs(),
    );
    assert_eq!(
        p.fri_layers_merkle_roots.len(),
        shape.num_committed(),
        "committed layer count must match what the proof carries"
    );
    assert_eq!(
        p.fri_final_poly_coeffs.len(),
        shape.num_terminal_coeffs(),
        "terminal coefficient count must match 2^effective_k"
    );
    assert_eq!(
        shape.coset_offset, opts.coset_offset,
        "the shape must take its coset offset from the options, not a literal"
    );
}

/// ★ The synthetic sweep §7 requires, and the reason it is needed.
///
/// Production pins `k = 7` and `coset_offset = 3` in every configuration, so no
/// real proof distinguishes an implementation that reads `k` from one that
/// hardcodes 7, and none reaches the clamp (`trace_bits <= 7`) at all. In LFM
/// these are not dead emitted branches — shape is compile-time, so they are
/// host-side arithmetic — which is exactly why they are testable here for free,
/// with no proving.
///
/// Each row is `(trace_bits, blowup_log, k)` with its expected
/// `(total_folds, num_committed, effective_k, terminal_len)`, derived by hand
/// from `terminal.rs:45-54` rather than from this module.
///
/// ## ★ Falsified, and the result is the leg's blindness finding made concrete
///
/// Deleting the `saturating_sub(1)` from `FriShape::num_committed` — the
/// off-by-one that makes a verifier authenticate one layer FEWER than the proof
/// commits — fails this test and
/// [`the_fri_sizing_prediction`], and **passes**
/// [`the_fri_shape_predicts_the_real_proofs_vector_lengths`]. The fixture has
/// `total_folds = 0`, so `0` and `0.saturating_sub(1)` are the same number and
/// the real proof cannot tell the two implementations apart.
///
/// So the most soundness-relevant constant in this leg is invisible to the only
/// real data available. That is not an argument for a better fixture; it is the
/// reason these synthetic rows are the primary instrument rather than a
/// supplement.
#[test]
fn the_fold_layout_is_right_off_productions_constants() {
    /// `(total_folds, num_committed, effective_k, terminal_len)`.
    type Layout = (u32, usize, u32, usize);
    // (trace_bits, blowup_log, k) -> Layout
    let cases: [(u32, u32, u32, Layout); 10] = [
        // The production point, at three blowups. k = 7 throughout. Note
        // `total_folds = trace_bits - k` is INDEPENDENT of the blowup: the
        // blowup enters `n` and `terminal_log` identically and cancels. That
        // cancellation is what makes the scout's `num_committed = trace_bits - 8`
        // invariant hold across every preset, and mis-expanding it (subtracting
        // the blowup twice) is how the first version of this table was wrong.
        (20, 1, 7, (13, 12, 7, 256)),
        (20, 2, 7, (13, 12, 7, 512)),
        (20, 3, 7, (13, 12, 7, 1024)),
        // k = 0: fold all the way down to one coefficient per coset.
        (10, 1, 0, (10, 9, 0, 2)),
        // k = 6, one below production.
        (10, 1, 6, (4, 3, 6, 128)),
        // The CLAMP regime, trace_bits <= k: terminal_log pins to lde_log, so
        // nothing folds and effective_k drops below the requested k.
        (7, 1, 7, (0, 0, 7, 256)),
        (4, 1, 7, (0, 0, 4, 32)),
        (2, 3, 7, (0, 0, 2, 32)),
        // k = 63 — far past any real trace, so the clamp always wins.
        (5, 1, 63, (0, 0, 5, 64)),
        // A single fold — `trace_bits = k + 1` — commits NOTHING, because the
        // last fold is never committed. The row that catches the off-by-one.
        (8, 1, 7, (1, 0, 7, 256)),
    ];
    for (trace_bits, blowup_log, k, expected) in cases {
        let shape = FriShape {
            log2_lde_length: trace_bits + blowup_log,
            blowup_log,
            final_poly_log_degree: k,
            coset_offset: 3,
            num_queries: 1,
        };
        shape.check();
        let got = (
            shape.total_folds(),
            shape.num_committed(),
            shape.effective_k(),
            shape.terminal_len(),
        );
        assert_eq!(
            got, expected,
            "trace_bits {trace_bits} blowup 2^{blowup_log} k {k}: \
             (total_folds, committed, effective_k, terminal_len)"
        );
        // Folds exceed committed layers by exactly one whenever anything folds.
        if shape.total_folds() > 0 {
            assert_eq!(
                shape.num_folds(),
                shape.num_committed() + 1,
                "the final fold is never committed"
            );
        }
    }
}

/// ★ The sizing prediction, PINNED BEFORE MEASURING — the leg's
/// measurements-vs-prediction target.
///
/// Derived from `others/lfm-fri-verify-spec.md` §8: `pathlen(i) = n − i − 2`,
/// so a query walks `Σ pathlen(i)` steps and pays one permutation per step plus
/// one leaf hash per committed layer. The blowup-2 row reproduces the spec's
/// own worked example (162 steps, 174 permutations, 38,106 total), which is
/// what says the formula is being read as written rather than re-derived.
///
/// Recorded as a test rather than a comment so that the emitter, when it
/// arrives, is measured against a number that was fixed beforehand.
#[test]
fn the_fri_sizing_prediction() {
    println!("blowup  n   C   Q   steps/q  perms/q     total");
    // (blowup_log, queries, expected steps/q, perms/q, total) at trace_bits = 20.
    for (blowup_log, queries, steps, perms, total) in [
        (1u32, 219usize, 162usize, 174usize, 38_106usize),
        (2, 110, 174, 186, 20_460),
        (3, 73, 186, 198, 14_454),
    ] {
        let shape = FriShape {
            log2_lde_length: 20 + blowup_log,
            blowup_log,
            final_poly_log_degree: 7,
            coset_offset: 3,
            num_queries: queries,
        };
        shape.check();
        println!(
            "  2^{blowup_log} {:>3} {:>3} {:>4} {:>8} {:>8} {:>9}",
            shape.log2_lde_length,
            shape.num_committed(),
            shape.num_queries,
            shape.path_steps_per_query(),
            shape.permutations_per_query(),
            shape.permutations(),
        );
        assert_eq!(
            shape.path_steps_per_query(),
            steps,
            "blowup 2^{blowup_log} steps"
        );
        assert_eq!(
            shape.permutations_per_query(),
            perms,
            "blowup 2^{blowup_log} perms/query"
        );
        assert_eq!(shape.permutations(), total, "blowup 2^{blowup_log} total");
    }
}
