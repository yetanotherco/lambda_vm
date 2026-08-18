//! The FRI leg: the per-layer walk, the fold chain, and the terminal check.
//!
//! ## The instrument problem this suite had to solve first
//!
//! `join_tests::the_fixture_carries_no_fri_layers_so_it_cannot_witness_the_fold`
//! pins the difficulty: the join fixture's sub-proof has `total_folds = 0`, so a
//! differential over it sees no fold, no per-layer walk and no terminal lookup.
//! The retiring FRI agent concluded the only witness was synthetic codewords
//! driven through production's commit and query phases.
//!
//! It is better than that, and the reason is one line of the fixture: the trace
//! is `boundaries.len().next_power_of_two()` rows
//! (`local_to_global.rs:269`). Ask for 512 boundaries instead of 4 and the same
//! production prover, the same AIR and the same verifier replay produce a proof
//! that FOLDS — real committed layer roots, real authentication paths, real
//! terminal coefficients, and the folding challenges out of production's own
//! `replay_rounds_after_round_1`. Nothing in this suite is synthetic. The layer
//! count is swept by asking for more rows.
//!
//! ## What this suite cannot see
//!
//! `k = 7` and `coset_offset = 3` in every configuration the prover can be
//! asked for, so — exactly as spec §7 says — nothing here distinguishes an
//! implementation that reads them from one that hardcodes them. That half is
//! discharged host-side by `join_tests::the_fold_layout_is_right_off_productions_constants`,
//! which sweeps `k ∈ {0, 6, 7, 63}` and the clamp regime over the shape
//! arithmetic. What IS now witnessed on real data is everything the shape feeds:
//! the `num_committed = total_folds − 1` off-by-one, the fold chain, the parity
//! branch, the walk depths and the terminal check.
//!
//! It also sees one sub-proof at a time. Nothing here says an epoch's sub-proofs
//! compose, and nothing here binds the terminal coefficients or the folding
//! challenges to a transcript — they arrive as arena values, and tying them to a
//! replay is assembly's obligation.

use math::field::traits::IsPrimeField;
use math::polynomial::Polynomial;
use stark::config::Commitment;
use stark::proof::stark::MultiProof;
use stark::traits::AIR;

use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};

use super::builder::LfmBuilder;
use super::compiler::{LfmProgram, compile};
use super::constraint_tests::BoxedAir;
use super::executor::execute;
use super::fri::{
    FRI_LEAF_GROUP, FriQuery, FriShape, declare_fri, emit_query_fri, hint_layer_openings,
};
use super::hash::TestPermutation;
use super::join_tests::{HostSubProof, build_host_sub_proof};
use super::validator::validate;
use super::word::{LfmWord, base_word, ext_word, word_as_ext};

type Gl = GoldilocksField;
type Ext3 = GoldilocksExtension;

/// A base element as its image in the cubic extension — component 0, the
/// embedding `IsSubFieldOf` uses implicitly wherever production multiplies a
/// base by an ext.
fn embed(x: &FE) -> FEE {
    FEE::new([*x, FE::zero(), FE::zero()])
}

// =============================================================================
// A real proof that folds
// =============================================================================

/// Proves L2G_MEMORY over `num_boundaries` boundary claims at `blowup`.
///
/// The same AIR, prover and options path as `constraint_tests::real_fixture` —
/// only the row count differs, and the row count is what decides whether FRI
/// folds. `num_boundaries` must be a power of two so the trace length is exactly
/// it (the generator pads to the next power of two, which would silently change
/// the shape this suite is measuring).
pub(super) fn folding_fixture(
    num_boundaries: usize,
    blowup: usize,
) -> (BoxedAir, MultiProof<Gl, Ext3, ()>) {
    use crate::tables::local_to_global::{
        CellBoundary, FiniClaim, InitClaim, generate_local_to_global_trace,
    };
    use crate::test_utils::{EPOCH_TEST_LABEL, multi_prove_ram};
    use stark::config::DefaultStarkTranscript;

    assert!(
        num_boundaries.is_power_of_two(),
        "the trace is padded to a power of two, so a non-power-of-two row count \
         would not be the shape asked for"
    );
    let opts = stark::proof::options::GoldilocksCubicProofOptions::with_blowup(blowup as u8)
        .expect("a power-of-two blowup is valid");
    let air = crate::continuation::l2g_memory_air(&opts, EPOCH_TEST_LABEL);

    let boundaries: Vec<CellBoundary> = (0..num_boundaries as u64)
        .map(|i| CellBoundary {
            address: 0x1000 + 8 * i,
            init: InitClaim {
                value: i + 1,
                timestamp: 0,
                originating_epoch: 0,
            },
            fini: FiniClaim {
                value: 2 * i + 3,
                epoch: EPOCH_TEST_LABEL,
                timestamp: 17 + i,
            },
        })
        .collect();
    let mut trace = generate_local_to_global_trace(&boundaries);

    let pairs: Vec<(
        &dyn AIR<Field = Gl, FieldExtension = Ext3, PublicInputs = ()>,
        _,
        _,
    )> = vec![(&air, &mut trace, &())];
    let proof = multi_prove_ram(pairs, &mut DefaultStarkTranscript::<Ext3>::new(&[]))
        .expect("the L2G_MEMORY fixture must prove at any power-of-two row count");

    (Box::new(air), proof)
}

/// Everything the FRI leg reads about one real sub-proof.
struct HostFri {
    shape: FriShape,
    /// The trace-side host fixture over the SAME proof: the openings, the roots,
    /// and production's own DEEP answers, which are this leg's `p₀`.
    trace: HostSubProof,
    /// One root per committed layer, in fold order.
    layer_roots: Vec<Commitment>,
    /// `ζ₀ .. ζ_C` from the verifier's replay.
    zetas: Vec<FEE>,
    /// The terminal polynomial's coefficients, low-to-high.
    coeffs: Vec<FEE>,
    /// `[query][layer]` — `(pᵢ(−υ^(2ⁱ)), path)`.
    openings: Vec<Vec<(FEE, Vec<Commitment>)>>,
}

/// Build the FRI host fixture for a real proof of `num_boundaries` rows.
fn host_fri(num_boundaries: usize, blowup: usize) -> HostFri {
    let (air, proof) = folding_fixture(num_boundaries, blowup);
    host_fri_from(&*air, &proof)
}

/// [`host_fri`] for a proof the caller already holds — needed where the test
/// also wants the AIR's verifier domain.
fn host_fri_from(
    air: &dyn AIR<Field = Gl, FieldExtension = Ext3, PublicInputs = ()>,
    proof: &MultiProof<Gl, Ext3, ()>,
) -> HostFri {
    use stark::proof::view::StarkProofView;

    let trace = build_host_sub_proof(air, proof);
    let view = StarkProofView::Owned(&proof.proofs[0]);
    let opts = air.options();
    let shape = FriShape::from_options(opts, trace.shape.log2_lde_length);
    shape.check();

    let openings = (0..view.query_list_len())
        .map(|q| {
            let d = view.query(q);
            d.layers_evaluations_sym()
                .iter()
                .enumerate()
                .map(|(i, sym)| (*sym, d.layer_auth_path(i).to_vec()))
                .collect()
        })
        .collect();

    HostFri {
        shape,
        layer_roots: view.fri_layers_merkle_roots().to_vec(),
        zetas: trace.zetas.clone(),
        coeffs: view.fri_final_poly_coeffs().to_vec(),
        openings,
        trace,
    }
}

impl HostFri {
    /// The arenas the FRI-only program declares, for the given queries.
    fn fri_arenas(&self, queries: &[usize]) -> Vec<Vec<LfmWord>> {
        vec![
            super::proof_arena::commitments_to_arena(&self.layer_roots),
            self.zetas.iter().map(ext_word).collect(),
            self.coeffs.iter().map(ext_word).collect(),
            self.query_arena(queries),
        ]
    }

    /// Per query, per layer: the symmetric evaluation then its path.
    fn query_arena(&self, queries: &[usize]) -> Vec<LfmWord> {
        let mut out = Vec::new();
        for &q in queries {
            for (sym, path) in &self.openings[q] {
                out.push(ext_word(sym));
                out.extend(super::proof_arena::commitments_to_arena(path));
            }
        }
        out
    }

    /// The terminal codeword, rebuilt exactly as `terminal_codeword_from_coeffs`
    /// does (`fri/terminal.rs:134-155`) out of production's own FFT and
    /// bit-reverse permutation.
    ///
    /// That module is `pub(crate)` inside `crypto/stark`, so this is a mirror of
    /// its three lines rather than a call to it. The mirror is what
    /// [`the_terminal_point_is_the_query_point_folded`] tests the emitter's
    /// evaluation against — and the mirror itself is checked, because the same
    /// codeword must reproduce the values the PROVER folded to, which no reading
    /// of these three lines could fake.
    fn terminal_codeword(&self) -> Vec<FEE> {
        use math::fft::bit_reversing::in_place_bit_reverse_permute;

        let coset_offset = FE::from(self.shape.coset_offset);
        let terminal_offset = coset_offset.pow(1u64 << self.shape.total_folds());
        let poly = Polynomial::new(&self.coeffs);
        let blowup = self.shape.terminal_len() / self.coeffs.len();
        let mut natural = Polynomial::evaluate_offset_fft::<Gl>(
            &poly,
            blowup,
            Some(self.coeffs.len()),
            &terminal_offset,
        )
        .expect("the terminal coset is a power of two inside the two-adicity");
        in_place_bit_reverse_permute(&mut natural);
        natural
    }
}

// =============================================================================
// The owed check: is the leaf gadget reusable?
// =============================================================================

/// ★ The check the retiring agent OWED and never ran: the machine's leaf hash at
/// `GroupShape { num_columns: 1, is_ext: true }` is byte-identical to the FRI
/// layer leaf, run against BOTH production backends.
///
/// The two sides genuinely use different types — the prover commits under
/// `PairKeccak256Backend` (`fri/mod.rs:100`) and the verifier authenticates
/// under `BatchedMerkleTreeBackend` (`verifier.rs:643`) — and the spec's claim is
/// that they are byte-identical. This asserts both, so a divergence shows up as
/// a named failure rather than as a mysterious walk that will not reach its root.
///
/// ## The vectors, and the lesson they encode
///
/// A tamper suite whose every vector differed in byte 0 is one of the holes this
/// phase found by falsifying its own guards. A leaf here is 48 bytes: two
/// extension elements, three components each, eight big-endian bytes each. So the
/// vectors are built to make **every one of the 48 byte positions carry a
/// distinct value**, with no component equal to another and none symmetric under
/// byte reversal. A wrong component order, a wrong element order, a
/// little-endian limb or a dropped high byte each move a different subset of the
/// 48, and all of them move at least one.
#[test]
fn the_fri_leaf_is_byte_identical_to_productions_own_backends() {
    use crypto::merkle_tree::traits::IsMerkleTreeBackend;
    use stark::config::{BatchedMerkleTreeBackend, FriLayerMerkleTreeBackend};

    // Six distinct components, each with six distinct nonzero bytes in
    // descending positions, so no two of the 48 bytes agree and no component is
    // a byte-reversal of itself or of another.
    let component = |i: u64| FE::from(0x0102_0304_0506_0708u64 * (i + 1) + 0x11 * (i + 1));
    let ext = |base: u64| FEE::new([component(base), component(base + 1), component(base + 2)]);
    let vectors: [(FEE, FEE); 4] = [
        (ext(0), ext(3)),
        // Order-sensitivity: the same two elements swapped must hash differently
        // (checked below), which is what says the leaf is ordered at all.
        (ext(3), ext(0)),
        // A zero element beside a maximal one: catches a gadget that skips or
        // truncates a zero limb, and a canonicity slip at p−1.
        (
            FEE::zero(),
            FEE::new([
                FE::from(Gl::modulus_minus_one()),
                FE::one(),
                FE::from(0xFFFF_FFFF_0000_0000u64),
            ]),
        ),
        // One bit apart in the LAST byte of the LAST component — the position a
        // suite that only ever varied byte 0 would never reach.
        (ext(9), FEE::new([component(12), component(13), FE::one()])),
    ];

    // The wrap hash production commits under: this leg's whole claim is that
    // the machine's leaf IS the verifier's leaf, and the verifier's backend
    // follows the aliases.
    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let arena = b.declare_arena(2);
    let v0 = b.hint_word(arena, 0);
    let v1 = b.hint_word(arena, 1);
    let leaf = super::sub_proof::emit_leaf_hash(&mut b, FRI_LEAF_GROUP, &[v0, v1]);
    b.public(leaf[0]);
    b.public(leaf[1]);
    let program = compile(b.finish());
    validate(&program).expect("the leaf program is admissible");

    let mut digests = Vec::new();
    for (i, (a, c)) in vectors.iter().enumerate() {
        let arenas = vec![vec![ext_word(a), ext_word(c)]];
        let exec = execute(&program, &arenas, &TestPermutation).expect("the leaf hash executes");
        let got = [exec.public_words[0].1, exec.public_words[1].1];

        let batched =
            <BatchedMerkleTreeBackend<Ext3> as IsMerkleTreeBackend>::hash_data(&vec![*a, *c]);
        let paired = <FriLayerMerkleTreeBackend<Ext3> as IsMerkleTreeBackend>::hash_data(&[*a, *c]);
        assert_eq!(
            batched, paired,
            "vector {i}: the spec's claim is that the prover's pair backend and \
             the verifier's batched backend are byte-identical; they are not"
        );
        assert_eq!(
            got,
            super::proof_arena::commitment_words(&batched),
            "vector {i}: the machine's leaf must be the verifier's leaf — this \
             is the byte-level check the FRI leg was handed as unverified"
        );
        digests.push(got);
    }
    assert_ne!(
        digests[0], digests[1],
        "swapping the two elements must change the leaf, or the emitted order is \
         not carried into the hash and the parity Select is decoration"
    );
    println!(
        "machine leaf == BatchedMerkleTreeBackend == PairKeccak256Backend on {} \
         vectors covering all 48 bytes",
        vectors.len()
    );
}

// =============================================================================
// The emitter, against production
// =============================================================================

/// The FRI leg alone, driven by a hinted index and a hinted DEEP pair.
///
/// Used where the point of the test is FRI rather than the join: the trace legs
/// cost ~5,000 instructions per query per group and would dominate a run whose
/// subject is the fold. The index still goes through one `bit_dec` and the point
/// still comes from [`super::sub_proof::emit_points_from_bits`], so the
/// machine's own derivation is under test rather than a supplied point.
///
/// Arena order: the per-query `(index, p₀, p₀ˢ)` block, then the four
/// [`FriArenas`].
fn fri_only_program(shape: FriShape, num_queries: usize) -> LfmProgram {
    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let q = b.declare_arena(3 * num_queries as u32);
    let (arenas, fri) = declare_fri(&mut b, shape, num_queries);
    for i in 0..num_queries {
        let index = b.hint_felt(q, 3 * i as u32);
        let p0 = b.hint_word(q, 3 * i as u32 + 1).as_ext();
        let p0_sym = b.hint_word(q, 3 * i as u32 + 2).as_ext();
        let bits = b.bit_dec(index, shape.index_bits());
        let (point, point_sym) = super::sub_proof::emit_points_from_bits(
            &mut b,
            shape.log2_lde_length,
            FE::from(shape.coset_offset),
            &bits,
        );
        let openings = hint_layer_openings(&mut b, shape, &arenas, i);
        let v = emit_query_fri(
            &mut b,
            shape,
            &fri,
            &FriQuery {
                p0,
                p0_sym,
                point,
                point_sym,
                bits: &bits,
            },
            &openings,
        );
        b.public(v.as_cell());
    }
    let program = compile(b.finish());
    validate(&program).expect("the FRI program must be admissible");
    program
}

impl HostFri {
    /// The `(index, p₀, p₀ˢ)` arena [`fri_only_program`] reads.
    fn deep_arena(&self, queries: &[usize]) -> Vec<LfmWord> {
        let mut out = Vec::new();
        for &q in queries {
            out.push(base_word(FE::from(self.trace.iotas[q] as u64)));
            out.push(ext_word(&self.trace.expected[q].0));
            out.push(ext_word(&self.trace.expected[q].1));
        }
        out
    }

    /// Every arena [`fri_only_program`] declares, in order.
    fn all_arenas(&self, queries: &[usize]) -> Vec<Vec<LfmWord>> {
        let mut all = vec![self.deep_arena(queries)];
        all.extend(self.fri_arenas(queries));
        all
    }
}

/// ★ The premise of this suite, checked before anything is built on it: the
/// production prover FOLDS when the trace is big enough, and the layer count is
/// steerable by the row count.
///
/// This is the finding that retires the leg's instrument problem. The FRI leg
/// was handed the conclusion that "the production instance exercises none of the
/// mechanism" and that synthetic codewords were the only witness. That was true
/// of the fixture as written and false of the fixture as available: the row count
/// is `boundaries.len().next_power_of_two()`, and `num_committed = trace_bits − 8`,
/// so 512 boundaries commit one layer and 2048 commit three. Everything below
/// therefore differentials against production data rather than against a
/// synthesized input, and the `saturating_sub(1)` off-by-one that
/// `join_tests::the_fold_layout_is_right_off_productions_constants` could only
/// catch host-side is now caught by an executed walk that fails to reach a real
/// root.
///
/// The four shapes are the sweep the successor brief asked for — `num_committed`
/// over 0, 1, 2, 3 — and the zero row is the original fixture, unchanged.
#[test]
fn the_real_prover_folds_and_the_layer_count_follows_the_row_count() {
    println!("rows      n  folds  committed  coeffs  zetas  queries  terminal_len");
    for (rows, committed) in [(4usize, 0usize), (512, 1), (1024, 2), (2048, 3)] {
        let h = host_fri(rows, 2);
        println!(
            "{rows:>5} {:>6} {:>6} {:>10} {:>7} {:>6} {:>8} {:>13}",
            h.shape.log2_lde_length,
            h.shape.total_folds(),
            h.shape.num_committed(),
            h.coeffs.len(),
            h.zetas.len(),
            h.trace.iotas.len(),
            h.shape.terminal_len(),
        );
        assert_eq!(
            h.shape.num_committed(),
            committed,
            "{rows} rows must commit {committed} FRI layers"
        );
        // The three structural lengths the verifier rejects on before its query
        // loop (`verifier.rs:426-448`), asked of the real proof.
        assert_eq!(
            h.layer_roots.len(),
            h.shape.num_committed(),
            "committed roots"
        );
        assert_eq!(
            h.coeffs.len(),
            h.shape.num_terminal_coeffs(),
            "terminal coefficients"
        );
        assert_eq!(
            h.zetas.len(),
            if h.shape.total_folds() > 0 {
                h.shape.num_committed() + 1
            } else {
                0
            },
            "folds exceed committed layers by one, and nothing folds at all when \
             the codeword is already terminal"
        );
        for (q, per_layer) in h.openings.iter().enumerate() {
            assert_eq!(per_layer.len(), h.shape.num_committed(), "query {q} layers");
            for (i, (_, path)) in per_layer.iter().enumerate() {
                assert_eq!(
                    path.len(),
                    h.shape.layer_path_len(i),
                    "query {q} layer {i}: the layer tree is one level shallower \
                     per fold, so its path length is n − i − 2"
                );
            }
        }
    }
}

/// ★ The deviation from spec §5, justified numerically against production's own
/// FFT: the terminal codeword's value at position `iota >> C` is the terminal
/// polynomial evaluated at `υ^(2^total_folds)`.
///
/// The emitter checks `P(υ^(2^total_folds)) = v` where production checks
/// `terminal_codeword[index] = v`. If those are not the same number the emitter
/// is wrong, and the machine's own assertions would not say so — they would
/// simply both be wrong together. So the identity is checked here, host-side,
/// over the real proofs at every one of their 219 indices: the codeword side
/// comes from production's `evaluate_offset_fft` + `in_place_bit_reverse_permute`
/// (`terminal.rs:150-155`) and the point side from production's own
/// `query_challenge_to_evaluation_point` raised by repeated squaring.
///
/// A missing coset offset, a wrong exponent, a dropped bit reversal or the wrong
/// shift on `iota` each break it. The zero-fold shape is included, where the
/// claim is that the two positions `2·iota` and `2·iota+1` are `υ` and `−υ` —
/// the identity that lets one emitted shape serve production's two branches.
#[test]
fn the_terminal_point_is_the_query_point_folded() {
    use stark::domain::new_verifier_domain;
    use stark::verifier::{IsStarkVerifier, Verifier};
    type V = Verifier<Gl, Ext3, ()>;

    for rows in [4usize, 512, 1024, 2048] {
        let (air, proof) = folding_fixture(rows, 2);
        let h = host_fri_from(&*air, &proof);
        let codeword = h.terminal_codeword();
        assert_eq!(codeword.len(), h.shape.terminal_len());
        let domain = new_verifier_domain(&*air, proof.proofs[0].trace_length);
        let c = h.shape.num_committed();

        let mut distinct = std::collections::HashSet::new();
        for &iota in &h.trace.iotas {
            let point = V::query_challenge_to_evaluation_point(iota, false, &domain);
            let mut x = point;
            for _ in 0..h.shape.total_folds() {
                x = x.square();
            }
            let at = Polynomial::new(&h.coeffs).evaluate(&embed(&x));

            // The position production compares at, per branch. The two branches
            // index DIFFERENTLY and conflating them is the mistake this test
            // made on its first run: the folding branch walks `index` down from
            // `iota` (`verifier.rs:735`), so it lands on `iota >> C`, while the
            // zero-fold branch never has an `index` at all and reads the pair
            // positions `2·iota` and `2·iota+1` directly (`verifier.rs:684-690`)
            // — its terminal codeword IS the deep codeword, in which `iota`
            // numbers pairs rather than elements.
            let position = if h.shape.total_folds() == 0 {
                iota * 2
            } else {
                iota >> c
            };
            assert_eq!(
                at,
                codeword[position],
                "rows {rows} iota {iota}: P(υ^(2^{})) must be the terminal \
                 codeword at position {position}",
                h.shape.total_folds()
            );
            if h.shape.total_folds() == 0 {
                assert_eq!(x, point, "nothing folds, so the point is unchanged");
                assert_eq!(
                    Polynomial::new(&h.coeffs).evaluate(&embed(
                        &V::query_challenge_to_evaluation_point(iota, true, &domain)
                    )),
                    codeword[iota * 2 + 1],
                    "iota {iota}: the symmetric position must be −υ"
                );
            }
            distinct.insert(position);
        }
        println!(
            "rows {rows:>5}: identity holds at all {} indices ({} distinct \
             terminal positions of {})",
            h.trace.iotas.len(),
            distinct.len(),
            codeword.len()
        );
        assert!(
            distinct.len() > 1 || codeword.len() == 1,
            "if every query landed on the same terminal position the check would \
             be one equation, not a sweep"
        );
    }
}

/// ★ The parity branch is REACHED, at every layer.
///
/// The leaf order is `[sym, v]` at odd index and `[v, sym]` at even
/// (`verifier.rs:637-641`), selected on bit `i` of `iota` at layer `i`. An
/// implementation with the two arms swapped, or with no `Select` at all, is
/// invisible to a fixture whose indices all share a parity — the same
/// degenerate-parameter trap as the fold itself, one level down. This asserts the
/// real proof's 219 indices carry both parities at every committed layer, which
/// is what makes `no_tampered_fri_value_can_pass` able to catch the swap.
#[test]
fn the_real_indices_reach_both_leaf_parities_at_every_layer() {
    for rows in [512usize, 1024, 2048] {
        let h = host_fri(rows, 2);
        for layer in 0..h.shape.num_committed() {
            let (even, odd): (Vec<_>, Vec<_>) = h
                .trace
                .iotas
                .iter()
                .map(|iota| (iota >> layer) & 1)
                .partition(|b| *b == 0);
            assert!(
                !even.is_empty() && !odd.is_empty(),
                "rows {rows} layer {layer}: {} even and {} odd indices — a layer \
                 reached by only one parity leaves the leaf-order Select \
                 unexercised",
                even.len(),
                odd.len()
            );
        }
        println!(
            "rows {rows:>5}: both parities present at all {} layers",
            h.shape.num_committed()
        );
    }
}

/// ★ THE HEADLINE: the emitted FRI leg verifies every query of a real proof that
/// really folds, at three layer counts, and its terminal value is the one
/// production would have looked up.
///
/// ## Why this is a strong check and not just an endpoint check
///
/// The published value is only the LAST link. Every intermediate `v` is pinned
/// too, and not by an assertion this test writes — by the proof itself. At layer
/// `i` the machine hashes `{v, sym}` into a leaf and walks it to
/// `fri_layers_merkle_roots[i]`, a root the production prover committed to its
/// own folded codeword. So a `v` that were wrong at any layer could not reach
/// that root, and the run would not execute at all. The fold chain, the point
/// chain, the parity ordering, the walk depths and the layer-to-`ζ` alignment are
/// all inside that.
///
/// What the published comparison adds is the terminal link, which no Merkle root
/// covers: the final fold is never committed (`fri/mod.rs:114-118`), so `v` at
/// the terminal layer is checked only against the coefficients. That is compared
/// here against production's own codeword at production's own position.
#[test]
fn the_fri_emitter_verifies_every_query_of_a_real_folding_proof() {
    for rows in [4usize, 512, 1024, 2048] {
        let h = host_fri(rows, 2);
        let all: Vec<usize> = (0..h.trace.iotas.len()).collect();
        let program = fri_only_program(h.shape, all.len());
        let exec = execute(&program, &h.all_arenas(&all), &TestPermutation).expect(
            "an honest FRI decommitment must authenticate every layer and reach \
             the terminal polynomial",
        );

        let codeword = h.terminal_codeword();
        let c = h.shape.num_committed();
        let mut nonzero = 0usize;
        for (k, &q) in all.iter().enumerate() {
            let v = word_as_ext(&exec.public_words[k].1).expect("the fold output is ext");
            let iota = h.trace.iotas[q];
            let position = if h.shape.total_folds() == 0 {
                iota * 2
            } else {
                iota >> c
            };
            assert_eq!(
                v, codeword[position],
                "rows {rows} query {q} (iota {iota}): the machine's terminal value \
                 must be the terminal codeword at the position production compares \
                 at"
            );
            if v != FEE::zero() {
                nonzero += 1;
            }
        }
        assert_eq!(
            nonzero,
            all.len(),
            "a vacuously zero fold would make the differential empty"
        );
        println!(
            "rows {rows:>5}: {} queries, {} committed layers, {} instructions, \
             {} permutations — every terminal value matches production's codeword",
            all.len(),
            c,
            program.instrs.len(),
            permutations(&program),
        );
    }
}

/// ★ Both legs as one program, on a real folding proof: the openings
/// authenticated, DEEP folded from the authenticated cells, and FRI folded from
/// DEEP's own output at DEEP's own point.
///
/// This is the seam the leg exists to close. The FRI leg could be correct in
/// isolation and still verify a different query than the trace leg did — folding
/// `p₀` values it was handed while the walks authenticated some other index. Here
/// there is nothing to hand: `emit_sub_proof_with_fri` takes the `QueryOutput`
/// cells, so `p₀`, `υ` and the index bits are the same addresses in both legs by
/// construction.
///
/// Run over a subset of queries because the trace side is ~50× the FRI side per
/// query at this shape; the coverage of the FRI mechanism itself is
/// [`the_fri_emitter_verifies_every_query_of_a_real_folding_proof`]'s, over all
/// 219.
#[test]
fn the_two_legs_verify_one_real_folding_proof_as_one_program() {
    let h = host_fri(1024, 2);
    let queries: Vec<usize> = (0..6).collect();
    assert_eq!(
        h.shape.num_committed(),
        2,
        "the 1024-row shape commits two layers"
    );
    // The query count is program shape, so a subset run is a different shape and
    // has to say so — `emit_sub_proof_with_fri` refuses to emit a query count
    // that disagrees with the one in the shape it was handed.
    let shape = FriShape {
        num_queries: queries.len(),
        ..h.shape
    };

    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let (_, _, terminal) =
        super::fri::emit_sub_proof_with_fri(&mut b, &h.trace.shape, shape, queries.len());
    for v in &terminal {
        b.public(v.as_cell());
    }
    let program = compile(b.finish());
    validate(&program).expect("the joined program is admissible");

    let mut arenas = h.trace.arenas(&queries);
    arenas.extend(h.fri_arenas(&queries));
    let exec = execute(&program, &arenas, &TestPermutation)
        .expect("the honest proof must authenticate, fold and reach the terminal");

    let codeword = h.terminal_codeword();
    for (k, &q) in queries.iter().enumerate() {
        let v = word_as_ext(&exec.public_words[k].1).expect("ext");
        assert_eq!(
            v,
            codeword[h.trace.iotas[q] >> h.shape.num_committed()],
            "query {q}: the joined program's terminal value"
        );
    }
    println!(
        "joined trace+DEEP+FRI over {} queries of a folding proof: {} \
         instructions, {} permutations",
        queries.len(),
        program.instrs.len(),
        permutations(&program),
    );
}

fn permutations(program: &LfmProgram) -> usize {
    program
        .instrs
        .iter()
        .filter(|i| matches!(i, super::instr::Instr::KeccakF(_)))
        .count()
}

fn count_matching<F: Fn(&super::instr::Instr) -> bool>(program: &LfmProgram, f: F) -> usize {
    program.instrs.iter().filter(|i| f(i)).count()
}

/// The marginal per-query cost of a shape, by emitting one query and two and
/// differencing — so no per-sub-proof plumbing (the hoisted root unpacks, the
/// coefficient hints) lands in the figure.
struct PerQuery {
    perms: usize,
    swaps: usize,
    instrs: usize,
}

fn marginal_fri(shape: FriShape) -> PerQuery {
    let one = fri_only_program(
        FriShape {
            num_queries: 1,
            ..shape
        },
        1,
    );
    let two = fri_only_program(
        FriShape {
            num_queries: 2,
            ..shape
        },
        2,
    );
    let dec =
        |p: &LfmProgram| count_matching(p, |i| matches!(i, super::instr::Instr::BitDec { .. }));
    PerQuery {
        perms: permutations(&two) - permutations(&one),
        swaps: dec(&two) - dec(&one),
        instrs: two.instrs.len() - one.instrs.len(),
    }
}

/// ★ MEASURED against the prediction pinned before the emitter existed.
///
/// `join_tests::the_fri_sizing_prediction` recorded 174/186/198 permutations per
/// query and 38,106/20,460/14,454 per sub-proof at blowup 2/4/8, `trace_bits =
/// 20`, derived from spec §8. This counts the `LFM_KECCAK` rows the emitter
/// actually emits at those shapes and asserts the same numbers.
///
/// The 2^20-row shape is emitted, not proved — a real proof at that size is a
/// prover run, not a test — but the quantity predicted IS the emitted
/// permutation count, so this is a measurement of the thing predicted rather
/// than a model of it. That the same formula holds on EXECUTED programs is
/// [`the_fri_emitter_verifies_every_query_of_a_real_folding_proof`]'s doing at
/// n = 10/11/12, where 219 queries produced exactly 1,971 / 4,161 / 6,570
/// permutations against `219 × (C + Σ pathlen)` = 219 × 9 / 19 / 30.
///
/// ## Two currencies that point opposite ways
///
/// The other columns are reported because permutations alone hide where the rows
/// go, and because the two honest answers disagree. Rendering the two extension
/// values of a layer leaf into 48 big-endian bytes costs `6C` byteswaps per
/// query, each one `LFM_BITDEC` row plus 64 `LFM_BALU` rows — which makes
/// byteswapping the majority of the leg's INSTRUCTIONS. In main-trace CELLS the
/// same comparison inverts by two orders of magnitude, because a permutation
/// expands into 24 `KECCAK_RND` rounds of 1,480 columns while a byteswap carries
/// 322 cells. `others/lfm-target-shape.md`'s rule that rows of different chips
/// are not comparable is exactly this, so both are printed and neither is called
/// "the" cost.
#[test]
fn the_emitted_permutation_count_meets_the_pinned_prediction() {
    const TRACE_BITS: u32 = 20;
    let swap_cells = super::machine_tests::byteswap_cells();
    let perm_cells = super::machine_tests::permutation_cells();
    println!(
        "blowup   C   Q  perms/q  predicted    total  predicted  swaps/q  instr/q           hash cells/q  swap cells/q"
    );
    for (blowup_log, queries, predicted_per_query, predicted_total) in [
        (1u32, 219usize, 174usize, 38_106usize),
        (2, 110, 186, 20_460),
        (3, 73, 198, 14_454),
    ] {
        let shape = FriShape {
            log2_lde_length: TRACE_BITS + blowup_log,
            blowup_log,
            final_poly_log_degree: 7,
            coset_offset: 3,
            num_queries: queries,
        };
        shape.check();
        let per = marginal_fri(shape);
        println!(
            "   2^{blowup_log} {:>3} {:>3} {:>8} {:>10} {:>8} {:>10} {:>8} {:>8}              {:>13} {:>13}",
            shape.num_committed(),
            queries,
            per.perms,
            predicted_per_query,
            per.perms * queries,
            predicted_total,
            per.swaps,
            per.instrs,
            per.perms as u64 * perm_cells,
            per.swaps as u64 * swap_cells,
        );
        assert_eq!(
            per.perms, predicted_per_query,
            "blowup 2^{blowup_log}: emitted permutations per query against the \
             pinned prediction"
        );
        assert_eq!(
            per.perms * queries,
            predicted_total,
            "blowup 2^{blowup_log}: emitted permutations per sub-proof"
        );
        // The same number, from the shape arithmetic rather than from the
        // emitted program. Equal counts here mean the emitter walks the depths
        // the shape says it should — the one place a wrong `layer_path_len`
        // would show up as agreement between two wrongs is if BOTH came from the
        // shape, and only one of these does.
        assert_eq!(
            per.perms,
            shape.permutations_per_query(),
            "the emitted program and the shape arithmetic must agree"
        );
        // One byteswap per extension component per leaf value: two values, three
        // components, per committed layer.
        assert_eq!(
            per.swaps,
            1 + 6 * shape.num_committed(),
            "the index decomposition plus six component byteswaps per layer"
        );
        // The inversion, asserted rather than left to the reader: byteswapping
        // is the majority of the instructions and a rounding error in cells.
        let swap_instrs = per.swaps * 65;
        assert!(
            swap_instrs * 2 > per.instrs,
            "byteswapping should be the majority of the leg's instructions              ({swap_instrs} of {})",
            per.instrs
        );
        assert!(
            per.perms as u64 * perm_cells > 100 * per.swaps as u64 * swap_cells,
            "and a rounding error in main-trace cells"
        );
    }
}

/// ★ ABSOLUTE (rule 7): the joined program contains ONE point derivation per
/// query and ONE decomposition of the index, and every term of the count comes
/// from a SHAPE rather than from a second emission.
///
/// ## This test was wrong first, and how it was caught matters more than the fix
///
/// Its first form measured the FRI leg's marginal `Select` count as
/// `selects(joined) − selects(trace_only)` and asserted the difference was
/// `C + 2 · path_steps`, reasoning that a second point derivation would add
/// `index_bits`. That is vacuous, and injecting the exact defect it denies — a
/// `QueryOutput` handing out a freshly derived point — left it GREEN. The reason
/// is rule 7's failure mode wearing a different hat: the defect lives in
/// `emit_sub_proof_with_bits`, which is what BOTH sides of the subtraction call,
/// so both gained `index_bits` selects and the difference never moved.
///
/// **A difference of two counts taken from our own emitter is still a relative
/// test, however much it looks like a count.** The marginal-cost idiom this phase
/// uses everywhere is safe only when the RESULT is compared against a number that
/// did not come from the emitter — a pinned prediction, or a closed form over the
/// shapes:
///
/// ```text
///   selects/query = index_bits                     (pow_bits, once per query)
///                 + 2 · merkle_depth · num_groups  (trace walks)
///                 + num_committed                  (FRI leaf ordering)
///                 + 2 · path_steps_per_query       (FRI walks)
/// ```
///
/// `pow_bits` emits one `Select` per bit (`edsl.rs:257-262`) and each walk level
/// two, since a digest is two words and both must swap on the same bit
/// (`edsl.rs:164-169`). A second derivation makes the measured count exceed the
/// closed form by exactly `index_bits`, and nothing cancels it. Re-falsified in
/// that form: the injected defect now fails with "a surplus of 11 index bits".
#[test]
fn the_fri_join_adds_no_second_point_derivation() {
    let h = host_fri(2048, 2);
    let sub = &h.trace.shape;
    let groups = sub.groups();
    let selects =
        |p: &LfmProgram| count_matching(p, |i| matches!(i, super::instr::Instr::Select { .. }));
    let decs =
        |p: &LfmProgram| count_matching(p, |i| matches!(i, super::instr::Instr::BitDec { .. }));

    let emit = |n: usize| {
        let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
        super::fri::emit_sub_proof_with_fri(
            &mut b,
            sub,
            FriShape {
                num_queries: n,
                ..h.shape
            },
            n,
        );
        compile(b.finish())
    };
    // Marginal, so the per-sub-proof plumbing is out of the figure — but the
    // figure is then compared against the shapes, never against another emission.
    let one = emit(1);
    let two = emit(2);
    let per_query_selects = selects(&two) - selects(&one);
    let per_query_decs = decs(&two) - decs(&one);

    let expected_selects = h.shape.index_bits()
        + 2 * sub.merkle_depth * groups.len()
        + h.shape.num_committed()
        + 2 * h.shape.path_steps_per_query();
    assert_eq!(
        per_query_selects,
        expected_selects,
        "selects per query: {} index bits for the ONE point derivation, {} for \
         {} trace walks over {} levels, {} FRI leaf orderings, {} for {} FRI path \
         steps. A surplus of {} index bits is a second point derivation or a \
         second index decomposition",
        h.shape.index_bits(),
        2 * sub.merkle_depth * groups.len(),
        groups.len(),
        sub.merkle_depth,
        h.shape.num_committed(),
        2 * h.shape.path_steps_per_query(),
        h.shape.path_steps_per_query(),
        h.shape.index_bits(),
    );

    // One decomposition of the index, plus one byteswap per field element that
    // enters a leaf: a base element is one, an extension element three.
    let leaf_swaps: usize = groups
        .iter()
        .map(|g| g.num_values() * if g.is_ext { 3 } else { 1 })
        .sum();
    let expected_decs = 1 + leaf_swaps + 6 * h.shape.num_committed();
    assert_eq!(
        per_query_decs,
        expected_decs,
        "decompositions per query: ONE for the index, {leaf_swaps} for the trace \
         leaves, {} for the FRI layer leaves. A surplus of one is a second index \
         decomposition",
        6 * h.shape.num_committed(),
    );
    println!(
        "per query: {per_query_selects} selects and {per_query_decs} \
         decompositions, both equal to the closed form over the shapes — one \
         point derivation, one index decomposition"
    );
}

// =============================================================================
// Falsification: what the leg denies
// =============================================================================

/// ★ No arena value the FRI leg reads can be moved without the run failing.
///
/// The emitted checks are `assert_eq` inside the program, which lowers to
/// `diff / 0` — provable and executable only when `diff` is zero. So "the tamper
/// is caught" and "the run does not execute" are the same statement, and a
/// tamper that still executed would be a hole.
///
/// The vectors sweep every KIND of value the leg reads, and the last one is the
/// only interesting attack: a COHERENT forgery in the sense of method rule 4 —
/// every value in it is a genuine value the production prover committed to, just
/// belonging to a different query. Nothing in it is malformed, no hash is
/// invented, and the leaf it builds is a leaf that really exists in the real
/// layer tree. What rejects it is only that the walk climbs at the index bits of
/// THIS query, so a real leaf at the wrong position cannot reach the root.
///
/// ## What is deliberately absent
///
/// There is no vector that tampers `p₀`. It is not an arena value here — it is a
/// cell the DEEP leg computed — and moving it is
/// `join_tests::no_tampered_value_can_move_the_fold_without_moving_the_root`'s
/// subject one leg back. A FRI-side tamper of `p₀` would only be possible in the
/// standalone driver, where it is hinted for isolation, and would prove nothing
/// about the joined program.
#[test]
fn no_tampered_fri_value_can_pass() {
    const ROWS: usize = 2048;
    let h = host_fri(ROWS, 2);
    let c = h.shape.num_committed();
    assert_eq!(c, 3, "this suite wants several layers to tamper inside");

    // Two queries whose layer-0 parities DIFFER, so the splice below moves a
    // leaf between positions of opposite parity as well as of different index.
    let a = (0..h.trace.iotas.len())
        .find(|&q| h.trace.iotas[q].is_multiple_of(2))
        .expect("an even index");
    let b = (0..h.trace.iotas.len())
        .find(|&q| !h.trace.iotas[q].is_multiple_of(2))
        .expect("an odd index");
    let queries = vec![a, b];
    let shape = FriShape {
        num_queries: queries.len(),
        ..h.shape
    };
    let program = fri_only_program(shape, queries.len());
    let honest = h.all_arenas(&queries);
    execute(&program, &honest, &TestPermutation).expect("the honest run must execute");

    let stride = h.shape.query_words();
    // (label, arena, word) — arena order is the driver's: deep, roots, zetas,
    // coeffs, queries.
    let bump: Vec<(&str, usize, usize)> = vec![
        ("query index", 0, 0),
        ("layer 0 root", 1, 0),
        ("layer 0 root, second word", 1, 1),
        ("layer 2 root", 1, 2 * (c - 1)),
        ("zeta_0 (the DEEP fold's challenge)", 2, 0),
        ("zeta_C (the uncommitted final fold)", 2, c),
        ("terminal coefficient 0", 3, 0),
        ("terminal coefficient 127", 3, h.coeffs.len() - 1),
        ("layer 0 symmetric evaluation", 4, 0),
        ("layer 0 sibling, leaf level", 4, 1),
        (
            "layer 0 sibling, top level",
            4,
            2 * h.shape.layer_path_len(0) - 1,
        ),
        ("second query's layer 0 evaluation", 4, stride),
    ];
    for (label, arena, word) in bump {
        let mut tampered = honest.clone();
        tampered[arena][word][0] += FE::one();
        let err = execute(&program, &tampered, &TestPermutation).expect_err(&format!(
            "moving the {label} must make the program unexecutable"
        ));
        println!("  {label:<40} rejected: {err:?}");
    }

    // The coherent forgery: query `a` presented with query `b`'s layer-0
    // decommitment. Every word is a real prover value.
    let mut spliced = honest.clone();
    let (from, to) = (stride, 0usize);
    let len = 1 + 2 * h.shape.layer_path_len(0);
    let borrowed: Vec<LfmWord> = spliced[4][from..from + len].to_vec();
    assert_ne!(
        borrowed,
        spliced[4][to..to + len],
        "the two queries must actually have different layer-0 openings, or the \
         splice is a no-op and this vector proves nothing"
    );
    spliced[4][to..to + len].copy_from_slice(&borrowed);
    let err = execute(&program, &spliced, &TestPermutation).expect_err(
        "a REAL leaf and a REAL path, at the wrong index, must still be rejected \
         — the walk climbs at this query's own bits",
    );
    println!(
        "  {:<40} rejected: {err:?}",
        "another query's real layer-0 opening"
    );
}

/// ★ The three structural length checks production performs at RUNTIME are, in
/// this machine, impossible to fail — and that is worth demonstrating rather
/// than asserting.
///
/// `verifier.rs:426-448` rejects on three lengths before its query loop, and the
/// comment there is emphatic about why: the per-query auth-path and
/// evaluation-sym vectors are **not** bound into the Fiat-Shamir transcript, so a
/// prover could send them EMPTY — making the fold loop run zero iterations and
/// accept the query vacuously — and that length check is the only thing pinning
/// them.
///
/// In LFM there is no vector to send. `declare_fri` fixes each arena's length
/// from the shape, and the executor refuses an arena of any other length
/// (`ArenaLenMismatch`) before a single instruction runs. So the attack the
/// production comment describes is not defended against here, it is
/// unrepresentable: there is no encoding of "a proof with no FRI layers" that the
/// program for a 3-layer shape will accept. This test spells out each of the
/// three, including the vacuous-fold one.
#[test]
fn the_shape_pins_the_lengths_production_must_check_at_runtime() {
    use super::executor::LfmExecError;

    let h = host_fri(2048, 2);
    let queries = vec![0usize];
    let shape = FriShape {
        num_queries: 1,
        ..h.shape
    };
    let program = fri_only_program(shape, 1);
    let honest = h.all_arenas(&queries);
    execute(&program, &honest, &TestPermutation).expect("the honest run must execute");

    // (label, arena, what the truncation would buy a prover)
    let attacks: [(&str, usize, &str); 3] = [
        (
            "no committed layer roots",
            1,
            "production's `fri_layers_merkle_roots().len() != num_committed` check",
        ),
        (
            "fewer terminal coefficients",
            3,
            "production's `fri_final_poly_coeffs().len() != 1 << effective_k` check",
        ),
        (
            "an EMPTY per-query decommitment — the vacuous fold",
            4,
            "production's per-query `layers_auth_paths_len()` check, the one its \
             comment calls the only thing pinning these vecs",
        ),
    ];
    for (label, arena, mirrors) in attacks {
        let mut truncated = honest.clone();
        truncated[arena].clear();
        let err = execute(&program, &truncated, &TestPermutation)
            .expect_err(&format!("{label} must be refused"));
        assert!(
            matches!(err, LfmExecError::ArenaLenMismatch { .. }),
            "{label} must be refused for its LENGTH, before any instruction \
             runs — got {err:?}"
        );
        println!("  {label:<52} refused as {err:?}\n    mirrors {mirrors}");
    }
}

/// ★ The FRI leg PROVES and VERIFIES — method rule 2, discharged rather than
/// argued.
///
/// Every other test here calls `execute`, which runs the executor and the
/// arena/assert semantics but never builds a trace or a proof. Rule 2 is explicit
/// that an execute-only test says nothing about the chips: where the executor
/// mirrors a computation the chip also does, only a prove+verify run sees the
/// chip.
///
/// It is tempting to argue the coverage away — the FRI leg emits no instruction
/// the trace legs do not already emit, and `join_tests::the_join_proves_and_verifies`
/// proves those. That argument is probably true and is exactly the kind of thing
/// rule 5 says to check instead of assert, so this proves the JOINED program: the
/// openings authenticated, DEEP folded, and FRI folded to the terminal check, all
/// in one proved and verified run over a real folding sub-proof.
///
/// One query, because the point is the chips rather than the sweep — the fold
/// mechanism's coverage is
/// [`the_fri_emitter_verifies_every_query_of_a_real_folding_proof`]'s, over all
/// 219 of three shapes.
#[test]
fn the_fri_leg_proves_and_verifies() {
    use super::proof::{lfm_prove, verify_against};
    use super::registry::build_artifacts;

    let h = host_fri(512, 2);
    assert_eq!(
        h.shape.num_committed(),
        1,
        "one committed layer is enough to put a leaf hash, a walk, a root compare \
         and both folds through the prover"
    );
    let queries = [0usize];
    let opts = super::join_tests::prove_options();
    let shape = FriShape {
        num_queries: queries.len(),
        ..h.shape
    };

    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let (_, _, terminal) =
        super::fri::emit_sub_proof_with_fri(&mut b, &h.trace.shape, shape, queries.len());
    for v in &terminal {
        b.public(v.as_cell());
    }
    let program = compile(b.finish());
    validate(&program).expect("the joined program is admissible");

    let mut arenas = h.trace.arenas(&queries);
    arenas.extend(h.fri_arenas(&queries));
    let artifacts = build_artifacts(&program, &opts);
    let proved = lfm_prove(&program, &artifacts, &arenas, &opts)
        .expect("the joined trace+DEEP+FRI program must prove");

    // The proved run's published terminal value, against production's own
    // codeword — so the proof is not merely valid but computes the right thing.
    let codeword = h.terminal_codeword();
    assert_eq!(
        word_as_ext(&proved.public_words[0].1).expect("ext"),
        codeword[h.trace.iotas[queries[0]] >> h.shape.num_committed()],
        "the PROVED run must publish the terminal codeword value production \
         would have looked up"
    );
    assert!(
        verify_against(
            &artifacts.roots,
            &artifacts.program_id,
            artifacts.keccak_rnd_chunks,
            &proved.proof,
            &proved.public_words,
            &opts,
            artifacts.hasher,
        ),
        "the joined FRI run must verify"
    );
    println!(
        "proved and verified: {} instructions, {} permutations, {} committed layer",
        program.instrs.len(),
        permutations(&program),
        h.shape.num_committed(),
    );
}
