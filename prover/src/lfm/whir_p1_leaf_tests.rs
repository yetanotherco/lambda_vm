//! Gates for the WHIR leaf's Poseidon1 arm (D-WHIR-P1 S5): the transcript
//! replay, the child-order 4-ary walk and the assembled chain, each against
//! the host function it mirrors.
//!
//! Every program here is built by a [`WrapHash::Poseidon1`] builder and run on
//! the width-16 socket (`HasherKind::Poseidon1W16`, the executor's host
//! permutation). As in `whir_chain_tests`, executing an honest proof IS the
//! stream comparison: every drawn value feeds a refusal.

use crypto::fiat_shamir::is_transcript::IsTranscript;
use crypto::fiat_shamir::p1_transcript::P1Transcript;
use crypto::hash::poseidon1_stark::Merkle4;
use multilinear::mle::Mle;
use multilinear::whir::Domain;
use multilinear::whir_chain::{
    CapPolicy, ChainConfig, ChainFormat, ChainProof, GrindBits, RoundOpenings, commit, prove,
    verify,
};
use multilinear::whir_hash::P1Whir;

use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};

use super::algebraic_commit::commitment_to_digest;
use super::builder::{Ext, Felt, LfmBuilder};
use super::compiler::{LfmProgram, compile};
use super::edsl::{self, WrapDigest, WrapHash};
use super::executor::execute;
use super::hash::HasherKind;
use super::instr::Instr;
use super::p1w16_emit::{hint_order, walk4, walk4_child};
use super::validator::validate;
use super::whir_chain::{
    ChainShape, RoundStorage, chain_opening_perms, chain_perms, emit_verify_weighted,
    push_round_words, round_words,
};
use super::whir_poly::emit_eq_eval;
use super::whir_transcript::{SpongeEntry, WhirTranscript};
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
    FEE::new([felt(seed, 3 * i), felt(seed, 3 * i + 1), felt(seed, 3 * i + 2)])
}

/// Compile, validate and run `b`'s program; the published words, flattened.
fn run(b: LfmBuilder, arenas: &[Vec<LfmWord>]) -> Result<Vec<FE>, String> {
    let program = compile(b.finish());
    validate(&program).map_err(|e| format!("not admissible: {e:?}"))?;
    let exec = execute(&program, arenas, &SOCKET).map_err(|e| format!("{e:?}"))?;
    Ok(exec.public_words.iter().flat_map(|(_, w)| *w).collect())
}

// ============================== the transcript =============================

/// One call of a transcript stream, both sides.
#[derive(Clone)]
enum Call {
    Bytes(Vec<u8>),
    Ext(FEE),
    Root([u8; 32]),
    Nonce(u64),
    DrawExt,
    DrawBits(usize),
    State,
}

/// ★ The in-guest Poseidon1 transcript replays `P1Transcript` call for call:
/// constant `append_bytes` of every length class (empty, short, a felt, not a
/// felt multiple, more than a rate), field elements, roots, nonces, extension
/// and bounded draws, and `state()` reads at every buffer fill — each output
/// published and compared with the host's.
#[test]
fn the_p1_whir_transcript_replays_the_hosts_stream() {
    let root = |seed: u64| -> [u8; 32] {
        crypto::hash::rpx::digest_to_commitment(&core::array::from_fn::<_, 4, _>(|k| {
            felt(seed, k)
        }))
    };
    let mut calls = vec![
        Call::Bytes(b"LAMBDAVM_MULTILINEAR_BLOCK_STATEMENT_V1/P1W16/C2".to_vec()),
        Call::Bytes(Vec::new()),
        Call::Bytes(vec![7; 8]),
        Call::Bytes(vec![1, 2, 3]),
        Call::State,
        Call::Root(root(1)),
        Call::Root(root(2)),
        Call::DrawExt,
        Call::DrawExt,
        Call::Bytes((0..100u8).collect()),
        Call::Ext(fee(3, 0)),
        Call::State,
        Call::Ext(fee(3, 1)),
        Call::Nonce(0x1234_5678_9abc),
        Call::DrawBits(10),
        Call::DrawBits(17),
    ];
    // Twenty draws in a row cross a permutation's sixteen output lanes.
    calls.extend((0..20).map(|i| Call::DrawBits(1 + i % 20)));
    for i in 0..13 {
        calls.push(Call::Ext(fee(4, i)));
        if i % 4 == 3 {
            calls.push(Call::State);
            calls.push(Call::DrawExt);
        }
    }
    calls.push(Call::State);

    // Host.
    let mut host = P1Transcript::new();
    let mut want: Vec<FE> = Vec::new();
    let mut arena: Vec<LfmWord> = Vec::new();
    for call in &calls {
        match call {
            Call::Bytes(bytes) => host.append_bytes(bytes),
            Call::Ext(v) => {
                host.append_field_element(v);
                arena.push(ext_word(v));
            }
            Call::Root(r) => {
                host.append_bytes(r);
                arena.push(commitment_to_digest(r));
            }
            Call::Nonce(n) => {
                host.append_bytes(&n.to_be_bytes());
                arena.push(base_word(FE::from(*n)));
            }
            Call::DrawExt => want.extend(ext_word(&host.sample_field_element())),
            Call::DrawBits(n) => {
                let v = host.sample_u64(1u64 << n);
                want.extend((0..*n).map(|j| FE::from((v >> j) & 1)));
            }
            Call::State => {
                let s = host.state();
                want.extend(commitment_to_digest(&s));
            }
        }
    }

    // Guest.
    let mut b = p1_builder();
    let a = b.declare_arena(arena.len() as u32);
    let mut t = WhirTranscript::for_builder(&mut b);
    let mut at = 0u32;
    for call in &calls {
        match call {
            Call::Bytes(bytes) => t.absorb_const_bytes(bytes),
            Call::Ext(_) => {
                let v = b.hint_word(a, at).as_ext();
                at += 1;
                t.absorb_ext(&mut b, v);
            }
            Call::Root(_) => {
                let r = b.hint_word(a, at);
                at += 1;
                t.absorb_digest(&mut b, r);
            }
            Call::Nonce(_) => {
                let n = b.hint_felt(a, at);
                at += 1;
                t.absorb_nonce(&mut b, n);
            }
            Call::DrawExt => {
                let e = t.sample_ext(&mut b);
                b.public(e.as_cell());
            }
            Call::DrawBits(n) => {
                for bit in t.sample_u64_pow2(&mut b, *n) {
                    b.public(Felt(bit.addr()).as_cell());
                }
            }
            Call::State => {
                let s = t.state(&mut b);
                b.public(s);
            }
        }
    }
    // A bit publishes as a word whose low lane is the bit.
    let got = run(b, &[arena]).expect("the replay executes");
    let mut expect_words: Vec<FE> = Vec::new();
    let mut k = 0usize;
    for call in &calls {
        let n = match call {
            Call::DrawExt | Call::State => 4,
            Call::DrawBits(n) => *n,
            _ => 0,
        };
        for _ in 0..n {
            let v = want[k];
            k += 1;
            if matches!(call, Call::DrawBits(_)) {
                expect_words.extend([v, FE::zero(), FE::zero(), FE::zero()]);
            } else {
                expect_words.push(v);
            }
        }
    }
    assert_eq!(got.len(), expect_words.len(), "one published word per output");
    assert_eq!(got, expect_words, "the replay's draws and states are the host's");
}

/// The byte arm is untouched: a [`WhirTranscript::for_builder`] of an RPX
/// builder emits exactly what [`WhirTranscript::new`] does.
#[test]
fn an_rpx_builder_keeps_the_byte_transcript() {
    let emit = |for_builder: bool| -> LfmProgram {
        let mut b = LfmBuilder::new().with_wrap_hash(WrapHash::production());
        let a = b.declare_arena(3);
        let mut t = if for_builder {
            WhirTranscript::for_builder(&mut b)
        } else {
            WhirTranscript::new()
        };
        t.absorb_const_bytes(b"statement");
        t.absorb_const_bytes(&[0u8; 7]);
        let r = b.hint_word(a, 0);
        t.absorb_digest(&mut b, r);
        let lanes = b.unpack(r);
        t.absorb_root_lanes(&mut b, &lanes);
        let n = b.hint_felt(a, 1);
        t.absorb_nonce(&mut b, n);
        let e = t.sample_ext(&mut b);
        b.public(e.as_cell());
        let s = t.state(&mut b);
        b.public(s);
        compile(b.finish())
    };
    assert_eq!(
        format!("{:?}", emit(true).instrs),
        format!("{:?}", emit(false).instrs)
    );
}

// ================================= the walk ================================

/// The child-order walk folds a leaf onto `Merkle4`'s root at every index of
/// trees of 1 to 6 index bits (odd tops included), equals the hint-order walk
/// over the same path, and refuses a moved sibling.
#[test]
fn the_child_order_walk_reaches_the_hosts_root() {
    for bits in 1..=6usize {
        let n = 1usize << bits;
        let leaves: Vec<[FE; 4]> = (0..n)
            .map(|i| core::array::from_fn(|k| felt(bits as u64, 4 * i + k)))
            .collect();
        let tree = Merkle4::new(&leaves).expect("a tree");
        let root = tree.root();
        let indices: Vec<usize> = if n <= 16 {
            (0..n).collect()
        } else {
            vec![0, 1, 2, 3, n / 2 + 1, n - 2, n - 1]
        };
        for index in indices {
            let path = tree.path(index).expect("a leaf");
            let triples: Vec<[[FE; 4]; 3]> = path
                .iter()
                .map(|s| core::array::from_fn(|j| core::array::from_fn(|k| s[4 * j + k])))
                .collect();
            // The proof's own order, the odd top cut to its partner.
            let mut child: Vec<[FE; 4]> = triples.iter().flatten().copied().collect();
            if bits % 2 == 1 {
                child.truncate(child.len() - 2);
            }
            let hints = hint_order(index, bits, &triples);
            for moved in [None, Some(0usize), Some(child.len() - 1)] {
                let mut sib = child.clone();
                if let Some(at) = moved {
                    sib[at][1] += FE::one();
                }
                let mut b = p1_builder();
                let a = b.declare_arena((1 + 1 + sib.len() + hints.len()) as u32);
                let leaf = edsl::hint_digest(&mut b, a, 0);
                let idx = b.hint_felt(a, 1);
                let index_bits = b.bit_dec(idx, bits);
                let s: Vec<WrapDigest> = (0..sib.len())
                    .map(|i| edsl::hint_digest(&mut b, a, 2 + i as u32))
                    .collect();
                let h: Vec<WrapDigest> = (0..hints.len())
                    .map(|i| edsl::hint_digest(&mut b, a, (2 + sib.len() + i) as u32))
                    .collect();
                let walked = walk4_child(&mut b, leaf, &index_bits, &s);
                let reference = walk4(&mut b, leaf, &index_bits, &h);
                b.public(walked.cells()[0]);
                b.public(reference.cells()[0]);
                let mut arena = vec![leaves[index], base_word(FE::from(index as u64))];
                arena.extend(sib.iter().copied());
                arena.extend(hints.iter().copied());
                let got = run(b, &[arena]).expect("the walks execute");
                let (child_root, hint_root) = (&got[..4], &got[4..8]);
                assert_eq!(hint_root, &root[..], "bits {bits} index {index}: hint order");
                if moved.is_none() {
                    assert_eq!(child_root, &root[..], "bits {bits} index {index}: child order");
                } else {
                    assert_ne!(
                        child_root,
                        &root[..],
                        "bits {bits} index {index}: a moved sibling moves the root"
                    );
                }
            }
        }
    }
}

// ================================= the chain ===============================

fn pseudo_mle(num_vars: usize, seed: u64) -> Mle<F> {
    let vals = (0..1usize << num_vars)
        .map(|i| {
            let mixed = (i as u64)
                .wrapping_mul(0x2545_F491_4F6C_DD1D)
                .wrapping_add(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            FE::from(mixed >> 13)
        })
        .collect();
    Mle::new(vals).expect("a cube")
}

fn point(num_vars: usize, seed: u64) -> Vec<FEE> {
    (0..num_vars)
        .map(|i| {
            FEE::new([
                FE::from(101 + seed + i as u64),
                FE::from(7 * i as u64 + 3),
                FE::from(i as u64 + 1),
            ])
        })
        .collect()
}

/// A small chain config at arity 4: blowup 4, folds of 4, `cap4` the 4-ary
/// cap policy (`ChainFormat::arity4_cap`).
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

/// Proves one chain over `P1Whir` and checks the host verifier accepts it.
fn fixture(num_vars: usize, cfg: ChainConfig) -> Fixture {
    let f = pseudo_mle(num_vars, 11);
    let z = point(num_vars, 0);
    let y = f.evaluate_in::<E>(&z).expect("f takes its own point");
    let (commitment, domain) =
        commit::<F, P1Whir>(&f, &cfg, true).expect("the polynomial commits");
    let mut proving = P1Transcript::new();
    let proof = prove::<F, E, _, P1Whir>(&f, &z, &commitment, &domain, &cfg, &mut proving)
        .expect("the chain proves");
    let root_bytes = commitment.root();
    let mut checking = P1Transcript::new();
    verify::<F, E, _, P1Whir>(&proof, &root_bytes, &z, y, &domain, &cfg, &mut checking)
        .expect("the host accepts its own proof");
    let shape = ChainShape::new_at(&cfg, num_vars, 4);
    Fixture {
        cfg,
        proof,
        z,
        y,
        root_bytes,
        domain,
        shape,
    }
}

fn host_accepts(f: &Fixture, proof: &ChainProof<F, E>) -> bool {
    let mut t = P1Transcript::new();
    verify::<F, E, _, P1Whir>(proof, &f.root_bytes, &f.z, f.y, &f.domain, &f.cfg, &mut t).is_ok()
}

/// The chain program for `shape` under Poseidon1: `z`, `y`, the root and the
/// final value hinted, then the rounds, as `whir_chain_tests::chain_program`.
fn p1_chain_program(shape: &ChainShape) -> LfmProgram {
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
    validate(&program).expect("the P1 chain leg is admissible (no twelve-felt hash row)");
    program
}

fn p1_chain_arena(f: &Fixture, proof: &ChainProof<F, E>) -> Vec<LfmWord> {
    let mut words: Vec<LfmWord> = f.z.iter().map(ext_word).collect();
    words.push(ext_word(&f.y));
    words.push(commitment_to_digest(&f.root_bytes));
    words.push(ext_word(&proof.final_value));
    push_round_words(&mut words, &f.shape, proof);
    words
}

fn hash16_rows(program: &LfmProgram) -> usize {
    program
        .instrs
        .iter()
        .filter(|i| matches!(i, Instr::Hash16(_)))
        .count()
}

/// The shapes the chain gates run: odd and even tree depths, one to three
/// rounds, uncapped and at the C2 cap, grind off and on.
fn chain_cases() -> Vec<(usize, ChainConfig)> {
    let mut cases = Vec::new();
    for cap4 in [CapPolicy::Off, CapPolicy::Fixed(2)] {
        for (num_vars, num_queries, grind) in
            [(5usize, 3usize, 0u8), (6, 4, 0), (9, 3, 0), (6, 3, 6), (9, 4, 6)]
        {
            cases.push((num_vars, p1_config(num_queries, grind, cap4)));
        }
    }
    cases
}

/// ★ The assembled Poseidon1 chain executes on every proof the host accepts.
#[test]
fn a_p1_chain_executes_on_a_proof_the_host_accepts() {
    for (num_vars, cfg) in chain_cases() {
        let f = fixture(num_vars, cfg);
        let program = p1_chain_program(&f.shape);
        let arena = p1_chain_arena(&f, &f.proof);
        execute(&program, &[arena], &SOCKET).unwrap_or_else(|e| {
            panic!(
                "S={num_vars} caps {:?} grind {:?}: the machine refused an accepted proof: {e:?}",
                f.shape.caps, f.shape.grind
            )
        });
    }
}

/// ★ The socket rows the chain emits are its closed form: openings, cap
/// checks and the sponge's own permutations (the width-8 grind is ALU work).
#[test]
fn a_p1_chain_emits_its_permutation_closed_form() {
    for (num_vars, cfg) in chain_cases() {
        let f = fixture(num_vars, cfg);
        let program = p1_chain_program(&f.shape);
        let entry = SpongeEntry::fresh_for(WrapHash::Poseidon1);
        assert_eq!(
            hash16_rows(&program),
            chain_perms(&f.shape, entry),
            "S={num_vars} caps {:?} grind {:?}: Hash16 rows against the closed form \
             (openings {})",
            f.shape.caps,
            f.shape.grind,
            chain_opening_perms(&f.shape)
        );
    }
}

/// ★ Every forgery the host rejects, the machine refuses: a final value, a
/// sumcheck evaluation, an out-of-domain value, a grind nonce, a current and a
/// successor Merkle sibling, and (capped) a cap node.
#[test]
fn a_tampered_p1_chain_cannot_execute() {
    for cap4 in [CapPolicy::Off, CapPolicy::Fixed(2)] {
        let f = fixture(9, p1_config(3, 6, cap4));
        let program = p1_chain_program(&f.shape);
        assert!(
            execute(&program, &[p1_chain_arena(&f, &f.proof)], &SOCKET).is_ok(),
            "the untouched proof must execute, or the arm below proves nothing"
        );
        let mut sites: Vec<(&str, ChainProof<F, E>)> = Vec::new();
        let mut forged = f.proof.clone();
        forged.final_value += FEE::one();
        sites.push(("the final value", forged));
        let mut forged = f.proof.clone();
        forged.rounds[0].sumcheck[0].evaluations[0] += FEE::one();
        sites.push(("a sumcheck evaluation", forged));
        let mut forged = f.proof.clone();
        if let Some(v) = forged.rounds[0].ood_value.as_mut() {
            *v += FEE::one();
        }
        sites.push(("an out-of-domain value", forged));
        let mut forged = f.proof.clone();
        forged.rounds[0].nonces.query ^= 1;
        sites.push(("a grind nonce", forged));
        let mut forged = f.proof.clone();
        match &mut forged.rounds[0].openings {
            RoundOpenings::Base(p) => p.current[1].proof.merkle_path[0][0] ^= 1,
            RoundOpenings::Extension(p) => p.current[1].proof.merkle_path[0][0] ^= 1,
        }
        sites.push(("a current Merkle sibling", forged));
        // A successor sibling: the first round whose successor paths walk at
        // all (a tree shorter than its cap has none).
        let mut forged = f.proof.clone();
        let moved = forged.rounds.iter_mut().any(|round| {
            let next = match &mut round.openings {
                RoundOpenings::Base(p) => &mut p.next,
                RoundOpenings::Extension(p) => &mut p.next,
            };
            match next.get_mut(1) {
                Some(o) if !o.proof.merkle_path.is_empty() => {
                    o.proof.merkle_path[0][0] ^= 1;
                    true
                }
                _ => false,
            }
        });
        if moved {
            sites.push(("a successor Merkle sibling", forged));
        }
        if cap4 != CapPolicy::Off {
            // The owner path ends with the cap: its last node is a cap node.
            let mut forged = f.proof.clone();
            match &mut forged.rounds[0].openings {
                RoundOpenings::Base(p) => {
                    let path = &mut p.current[0].proof.merkle_path;
                    let last = path.len() - 1;
                    path[last][0] ^= 1;
                }
                RoundOpenings::Extension(p) => {
                    let path = &mut p.current[0].proof.merkle_path;
                    let last = path.len() - 1;
                    path[last][0] ^= 1;
                }
            }
            sites.push(("a cap node", forged));
        }
        for (name, forged) in &sites {
            assert!(
                !host_accepts(&f, forged),
                "{name} (caps {:?}): the host must reject the forgery",
                f.shape.caps
            );
            assert!(
                execute(&program, &[p1_chain_arena(&f, forged)], &SOCKET).is_err(),
                "{name} (caps {:?}): the machine must refuse the forgery",
                f.shape.caps
            );
        }
    }
}
