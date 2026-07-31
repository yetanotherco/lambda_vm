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
use super::sub_proof::{GroupShape, ROWS_PER_LEAF, SubProofShape, emit_sub_proof};
use super::validator::validate;
use super::word::{LfmWord, base_word, ext_word, word_as_ext};

type Gl = GoldilocksField;
type Ext3 = GoldilocksExtension;
type V = Verifier<Gl, Ext3, ()>;

/// One committed matrix's data for one query, host side: the row pair in leaf
/// order and the path that authenticates it.
struct HostGroupOpening {
    /// `evaluations ‖ evaluations_sym`, as arena words.
    values: Vec<LfmWord>,
    siblings: Vec<Commitment>,
}

/// Everything the machine reads about one sub-proof, read off a real proof.
///
/// Assembled once and shared, because `open_sub_proof` replays the whole
/// verifier transcript and the fixture proof is regenerated on every call.
struct HostSubProof {
    shape: SubProofShape,
    gamma: FEE,
    zeta: FEE,
    /// The OOD grid, row-major.
    ood: Vec<FEE>,
    claimed_parts: Vec<FEE>,
    /// One root per group, in `SubProofShape::groups` order.
    roots: Vec<Commitment>,
    /// `[query][group]`.
    openings: Vec<Vec<HostGroupOpening>>,
    iotas: Vec<usize>,
    /// The production reconstruction's answer per query, `(regular, sym)`.
    expected: Vec<(FEE, FEE)>,
    /// Production's query points, kept so the machine's derivation can be
    /// checked against them rather than against a local formula.
    points: Vec<(FE, FE)>,
}

fn host_sub_proof() -> &'static HostSubProof {
    use std::sync::OnceLock;
    static CELL: OnceLock<HostSubProof> = OnceLock::new();
    CELL.get_or_init(build_host_sub_proof)
}

fn build_host_sub_proof() -> HostSubProof {
    let (air, proof) = real_fixture();
    let sp = open_sub_proof(&*air, &proof);
    let (deep, gamma) = deep_shape(&sp, &*air);
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

    let domain = new_verifier_domain(&*air, view.trace_length());
    let layout = V::ood_layout(&*air);
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

    let mut openings = Vec::new();
    let mut expected = Vec::new();
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
        expected,
        points,
    }
}

impl HostSubProof {
    /// The arenas [`emit_sub_proof`] declares, in its declaration order.
    fn arenas(&self, queries: &[usize]) -> Vec<Vec<LfmWord>> {
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
    fn query_arena(&self, queries: &[usize]) -> Vec<LfmWord> {
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
        ROWS_PER_LEAF, 2,
        "the row-pair leaf is what makes that true"
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

fn prove_options() -> stark::proof::options::ProofOptions {
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
#[test]
fn no_tampered_value_can_move_the_fold_without_moving_the_root() {
    use super::proof_arena::{commitments_to_arena, walk_to_root};

    let h = host_sub_proof();
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
                    panic!("group {g} slot {slot}: a moved value must not authenticate")
                });

            // Coherent: recompute the leaf the tampered values really give and
            // the root that leaf really reaches, using PRODUCTION's hashers.
            let leaf = tampered_leaf(h, q, g, slot);
            let forged = walk_to_root(leaf, h.iotas[q], &h.openings[q][g].siblings);
            assert_ne!(
                forged, h.roots[g],
                "group {g} slot {slot}: the tamper must move the root, or the \
                 vector is vacuous"
            );
            let mut coherent_roots = h.roots.clone();
            coherent_roots[g] = forged;
            arenas[3] = commitments_to_arena(&coherent_roots);
            let forged_run = execute(&program, &arenas, &TestPermutation).unwrap_or_else(|e| {
                panic!("group {g} slot {slot}: the coherent forgery must execute: {e:?}")
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
                "group {g} slot {slot}: a value in the leaf's {} half must move \
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
    println!("{vectors} tamper vectors, every value slot of every group, both ways round");

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
            "flipping index bit {level} left every root unchanged — the fixture's \
             trees are degenerate at this index and the walk half of this vector \
             tests nothing"
        );
        execute(&program, &arenas, &TestPermutation)
            .err()
            .unwrap_or_else(|| panic!("index bit {level}: a moved index must not authenticate"));

        arenas[3] = commitments_to_arena(&coherent_roots);
        let forged = execute(&program, &arenas, &TestPermutation)
            .unwrap_or_else(|e| panic!("index bit {level}: coherent forgery must execute: {e:?}"));
        assert_ne!(
            forged.public_words[0].1, honest.public_words[0].1,
            "index bit {level}: the index derives the evaluation point, so a \
             forged walk at another index must also fold at another point"
        );
    }
    println!("{} index vectors, one per level", h.shape.merkle_depth);

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
            .unwrap_or_else(|| panic!("sibling level {level}: a moved path must not authenticate"));
    }
    println!("{} sibling vectors, one per level", h.shape.merkle_depth);

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
