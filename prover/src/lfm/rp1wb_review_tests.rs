//! Reviewer r-p1w-b's tests on the Poseidon1 WHIR leaf (REV-P1W-B): the
//! child-order walk is bound to the transcript's query index (no sidecar), and
//! the sparse width-8 permutation is the dense one. (The grind's accept set is
//! pinned by `rp1wa_review_tests`.)

use crypto::fiat_shamir::is_transcript::IsTranscript;
use crypto::fiat_shamir::p1_transcript::P1Transcript;
use crypto::hash::poseidon1_w8;
use multilinear::mle::Mle;
use multilinear::whir::Domain;
use multilinear::whir_chain::{
    CapPolicy, ChainConfig, ChainFormat, ChainProof, GrindBits, RoundOpenings, commit, prove,
    verify,
};
use multilinear::whir_hash::P1Whir;

use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};

use super::algebraic_commit::commitment_to_digest;
use super::builder::{Ext, LfmBuilder};
use super::compiler::{LfmProgram, compile};
use super::edsl::WrapHash;
use super::executor::execute;
use super::hash::HasherKind;
use super::p1w16_emit::{Lane, w8_permute, w8_permute_lanes};
use super::validator::validate;
use super::whir_chain::{ChainShape, RoundStorage, emit_verify_weighted, push_round_words, round_words};
use super::whir_poly::emit_eq_eval;
use super::whir_transcript::WhirTranscript;
use super::word::{LfmWord, base_word, ext_word};

type F = GoldilocksField;
type E = GoldilocksExtension;

const SOCKET: HasherKind = HasherKind::Poseidon1W16;

fn p1_builder() -> LfmBuilder {
    LfmBuilder::new().with_wrap_hash(WrapHash::Poseidon1)
}

fn felt(seed: u64, i: usize) -> FE {
    let mut z = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(i as u64 + 1);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    FE::from(z ^ (z >> 31))
}

fn fee(seed: u64, i: usize) -> FEE {
    FEE::new([
        felt(seed, 3 * i),
        felt(seed, 3 * i + 1),
        felt(seed, 3 * i + 2),
    ])
}

fn run(b: LfmBuilder, arenas: &[Vec<LfmWord>]) -> Result<Vec<FE>, String> {
    let program = compile(b.finish());
    validate(&program).map_err(|e| format!("not admissible: {e:?}"))?;
    let exec = execute(&program, arenas, &SOCKET).map_err(|e| format!("{e:?}"))?;
    Ok(exec.public_words.iter().flat_map(|(_, w)| *w).collect())
}

// ======================= the walk is bound to the index =====================

fn p1_config(num_queries: usize, grind: u8, cap4: CapPolicy) -> ChainConfig {
    ChainConfig {
        log_blowup: 2,
        log_folding: 4,
        num_queries,
        grind: GrindBits::uniform(grind),
        format: ChainFormat {
            arity4_cap: cap4,
            ..ChainFormat::DEFAULT
        },
    }
}

struct Fixture {
    cfg: ChainConfig,
    proof: ChainProof<F, E>,
    z: Vec<FEE>,
    y: FEE,
    root_bytes: [u8; 32],
    domain: Domain<F>,
    shape: ChainShape,
}

fn fixture(num_vars: usize, cfg: ChainConfig) -> Fixture {
    let vals = (0..1usize << num_vars)
        .map(|i| FE::from((i as u64).wrapping_mul(0x2545_F491_4F6C_DD1D) >> 13))
        .collect();
    let f = Mle::new(vals).expect("a cube");
    let z: Vec<FEE> = (0..num_vars).map(|i| fee(77, i)).collect();
    let y = f.evaluate_in::<E>(&z).expect("f takes its own point");
    let (commitment, domain) = commit::<F, P1Whir>(&f, &cfg, true).expect("commits");
    let mut proving = P1Transcript::new();
    let proof =
        prove::<F, E, _, P1Whir>(&f, &z, &commitment, &domain, &cfg, &mut proving).expect("proves");
    let root_bytes = commitment.root();
    let shape = ChainShape::new_at(&cfg, num_vars, 4);
    let fx = Fixture {
        cfg,
        proof,
        z,
        y,
        root_bytes,
        domain,
        shape,
    };
    assert!(host_accepts(&fx, &fx.proof), "the host accepts its own proof");
    fx
}

fn host_accepts(f: &Fixture, proof: &ChainProof<F, E>) -> bool {
    let mut t = P1Transcript::new();
    verify::<F, E, _, P1Whir>(proof, &f.root_bytes, &f.z, f.y, &f.domain, &f.cfg, &mut t).is_ok()
}

fn chain_program(shape: &ChainShape) -> LfmProgram {
    let mut b = p1_builder();
    let mut total = (shape.num_vars + 3) as u32;
    let first_round = total;
    for r in 0..shape.rounds() {
        total += round_words(shape, r);
    }
    let arena = b.declare_arena(total);
    let mut transcript = WhirTranscript::for_builder(&mut b);
    let z: Vec<Ext> = (0..shape.num_vars)
        .map(|i| b.hint_word(arena, i as u32).as_ext())
        .collect();
    let y = b.hint_word(arena, shape.num_vars as u32).as_ext();
    let root = b.hint_word(arena, shape.num_vars as u32 + 1);
    let root_lanes = b.unpack(root);
    let final_value = b.hint_word(arena, shape.num_vars as u32 + 2).as_ext();
    let storage = RoundStorage::hint(&mut b, arena, first_round, shape);
    let (current, next) = storage.openings();
    let wires = storage.wires(&current, &next);
    let domain = Domain::<F>::new(shape.domain_log[0]).expect("the first domain");
    emit_verify_weighted(
        &mut b,
        &mut transcript,
        &wires,
        &root_lanes,
        final_value,
        y,
        shape,
        &domain,
        |b, alphas| emit_eq_eval(b, &z, alphas),
    );
    let program = compile(b.finish());
    validate(&program).expect("admissible");
    program
}

fn chain_arena(f: &Fixture, proof: &ChainProof<F, E>) -> Vec<LfmWord> {
    let mut words: Vec<LfmWord> = f.z.iter().map(ext_word).collect();
    words.push(ext_word(&f.y));
    words.push(commitment_to_digest(&f.root_bytes));
    words.push(ext_word(&proof.final_value));
    push_round_words(&mut words, &f.shape, proof);
    words
}

/// Swaps two non-owner openings of one side of round `r` (current or
/// successor) when they open different blocks: each is a valid opening of
/// the committed tree, at the OTHER query's index. `false` when the side has
/// fewer than three openings or the two open the same block.
fn swap_openings(proof: &mut ChainProof<F, E>, r: usize, current: bool) -> bool {
    fn side<V: math::field::traits::IsField>(
        v: &mut [multilinear::whir_commit::CosetOpening<V>],
    ) -> bool {
        if v.len() < 3 || v[1].values == v[2].values {
            return false;
        }
        v.swap(1, 2);
        true
    }
    match (&mut proof.rounds[r].openings, current) {
        (RoundOpenings::Base(p), true) => side(&mut p.current),
        (RoundOpenings::Base(p), false) => side(&mut p.next),
        (RoundOpenings::Extension(p), true) => side(&mut p.current),
        (RoundOpenings::Extension(p), false) => side(&mut p.next),
    }
}

/// ★ The leaf's arena is filled from the proof alone (no index sidecar), and
/// the walk takes the proof's siblings in child order: the opening a query
/// checks must still be the one at the TRANSCRIPT's index. Two openings of the
/// committed tree, each valid at its own index, swapped between two queries,
/// are refused by the host and by the machine — on round 0's base tree, on
/// a later round's extension tree, and on a successor tree, uncapped and at
/// the production 4-ary cap (C2).
#[test]
fn rp1wb_an_opening_moved_to_another_query_index_is_refused() {
    for cap4 in [CapPolicy::Off, CapPolicy::Fixed(2)] {
        let f = fixture(10, p1_config(4, 0, cap4));
        let program = chain_program(&f.shape);
        assert!(
            execute(&program, &[chain_arena(&f, &f.proof)], &SOCKET).is_ok(),
            "the untouched proof executes (caps {:?})",
            f.shape.caps
        );
        let mut tried = 0;
        for (r, current) in [(0usize, true), (0, false), (1, true)] {
            let mut forged = f.proof.clone();
            if !swap_openings(&mut forged, r, current) {
                continue;
            }
            tried += 1;
            assert!(
                !host_accepts(&f, &forged),
                "round {r} current={current} caps {:?}: the host must refuse",
                f.shape.caps
            );
            assert!(
                execute(&program, &[chain_arena(&f, &forged)], &SOCKET).is_err(),
                "round {r} current={current} caps {:?}: the machine must refuse an opening \
                 at another query's index",
                f.shape.caps
            );
        }
        assert!(tried >= 2, "at least two sides were swapped (caps {cap4:?})");
    }
}

/// The sparse width-8 permutation equals the dense one (and the host's) on
/// random inputs, every lane variable and under random constant/variable
/// patterns, at every `keep`.
#[test]
fn rp1wb_the_sparse_w8_is_the_dense_w8() {
    for trial in 0..6u64 {
        let input: [FE; 8] = core::array::from_fn(|i| felt(100 + trial, i));
        let want = poseidon1_w8::permute(input);
        let mask = (trial * 0x5b) as u8; // which lanes are constants
        for keep in [1usize, 4, 8] {
            let mut b = p1_builder();
            let a = b.declare_arena(8);
            let vars: [_; 8] = core::array::from_fn(|i| b.hint_felt(a, i as u32));
            let lanes: [Lane; 8] = core::array::from_fn(|i| {
                if mask >> i & 1 == 1 {
                    Lane::Const(input[i])
                } else {
                    Lane::Var(vars[i])
                }
            });
            let sparse = w8_permute_lanes(&mut b, lanes, keep);
            let dense = w8_permute(&mut b, vars);
            for (s, d) in sparse.iter().zip(&dense) {
                b.public(s.as_cell());
                b.public(d.as_cell());
            }
            let arena: Vec<LfmWord> = input.iter().map(|v| base_word(*v)).collect();
            let got = run(b, &[arena]).expect("executes");
            for k in 0..keep {
                assert_eq!(got[8 * k], want[k], "trial {trial} keep {keep} sparse lane {k}");
                assert_eq!(got[8 * k + 4], want[k], "trial {trial} keep {keep} dense lane {k}");
            }
        }
    }
}
