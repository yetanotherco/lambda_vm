//! A chain's first successor encoded from its bound message
//! ([`math_cuda::whir::encode_successor`], D-SPEED lever 5a) against the fold
//! it replaces, on the device.
//!
//! Runs on a GPU box. The message is bound by the opening session's own rounds
//! over the product program lowered by `multilinear::gpu::lower`, at the
//! challenges the fold then takes, so the pair compared is the production one:
//! [`OpeningSession::encode_successor`] and [`fold_resident_fused`] over the
//! committed codeword. As VALUES (canonical limbs): the encode computes the
//! fold's field elements by another sequence of operations, so raw limbs agree
//! wherever both are canonical; the ones that do not are counted and printed.
//! The host fold (`multilinear::whir::fold_codeword_k_on_host`) is the third
//! witness.
//!
//! [`fold_resident_fused`]: math_cuda::whir::fold_resident_fused

use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;
use math::field::goldilocks::GoldilocksField as Gl;
use math_cuda::whir_open::OpeningSession;
use multilinear::whir::Domain;

type FE = FieldElement<Gl>;
type FE3 = FieldElement<Ext3>;

const P: u64 = 0xFFFF_FFFF_0000_0001;

fn canonical(limbs: &[u64]) -> Vec<u64> {
    limbs
        .iter()
        .map(|&v| if v >= P { v - P } else { v })
        .collect()
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 % P
    }
}

/// The opening's rule, `w·f`, lowered as the device session runs it.
fn product() -> multilinear::gpu::Lowered {
    let mut builder = multilinear::program::Builder::<Ext3>::new();
    let weight = builder.var(0);
    let message = builder.var(1);
    let root = builder.mul(weight, message);
    let program = builder.finish(root).expect("program");
    multilinear::gpu::lower(&program).expect("lowers")
}

/// Two-inverse and the inverse generator of each of `levels` levels from a
/// domain with generator `g`: what `multilinear::gpu` passes a fold.
fn fold_scalars(g: FE, levels: usize) -> (u64, Vec<u64>) {
    let two_inv = *FE::from(2u64).inv().expect("2 is invertible").value();
    let mut g_inv = g.inv().expect("a generator is invertible");
    let mut g_invs = Vec::with_capacity(levels);
    for _ in 0..levels {
        g_invs.push(*g_inv.value());
        g_inv = g_inv.square();
    }
    (two_inv, g_invs)
}

fn ext(raw: &[u64]) -> Vec<FE3> {
    raw.chunks_exact(3)
        .map(|l| FE3::new([FE::from(l[0]), FE::from(l[1]), FE::from(l[2])]))
        .collect()
}

/// Commits `2^num_vars` base values at blowup 4 and a first fold of `levels`,
/// binds the message with the session's rounds, and compares the encoded
/// successor with the fused fold and the host fold at the same challenges.
fn the_encoded_successor_is_the_fold(num_vars: usize, levels: usize, seed: u64) {
    let log_blowup = 2;
    let mut rng = Rng(seed);
    let len = 1usize << num_vars;
    let evals: Vec<u64> = (0..len).map(|_| rng.next()).collect();
    let weight: Vec<u64> = (0..len * 3).map(|_| rng.next()).collect();
    let (committed, _root) = math_cuda::whir::commit_codeword(
        &evals,
        log_blowup,
        levels,
        true,
        math_cuda::DeviceHash::Keccak256,
    )
    .unwrap_or_else(|e| panic!("device commit (needs a GPU): {e:?}"));

    let mut session = OpeningSession::new(&weight, &evals).expect("session");
    let lowered = product();
    let mut rounds = session
        .sumcheck(
            &lowered.nodes,
            &lowered.consts,
            lowered.num_slots,
            lowered.root_slot,
        )
        .expect("session rounds");
    let mut alphas = Vec::with_capacity(levels * 3);
    for _ in 0..levels {
        rounds.round(&[1, 0, 0, 2, 0, 0]).expect("round");
        let alpha = [rng.next(), rng.next(), rng.next()];
        rounds.fold(&alpha).expect("fold");
        alphas.extend_from_slice(&alpha);
    }
    drop(rounds);
    session.bound(levels);
    let (_, message_before) = session.tables_to_host().expect("read back");

    let encoded = session
        .encode_successor(&committed, levels)
        .expect("encode the successor");
    let domain = Domain::<Gl>::new(num_vars + log_blowup).expect("domain");
    let (two_inv, g_invs) = fold_scalars(*domain.generator(), levels);
    let folded = math_cuda::whir::fold_resident_fused(&committed, two_inv, &g_invs, &alphas)
        .expect("fused fold");
    assert_eq!(encoded.elements(), folded.elements());
    assert_eq!(encoded.elements(), len << log_blowup >> levels);
    assert!(!encoded.is_base(), "the successor is extension-valued");

    let encoded_raw = encoded.to_host().expect("read back");
    let folded_raw = folded.to_host().expect("read back");
    let raw_differ = encoded_raw
        .iter()
        .zip(&folded_raw)
        .filter(|(a, b)| a != b)
        .count();
    println!(
        "2^{num_vars}, fold {levels}: successor of {} values; raw limbs differing {raw_differ} of {}",
        encoded.elements(),
        encoded_raw.len()
    );
    assert_eq!(
        canonical(&encoded_raw),
        canonical(&folded_raw),
        "2^{num_vars}, fold {levels}: the encoded successor is not the fold"
    );

    // The host fold of the host's own codeword, at the same challenges.
    let host_codeword = multilinear::whir::encode::<Gl, Gl>(
        &multilinear::whir::lift_coefficients(
            &multilinear::mle::Mle::new(evals.iter().map(|v| FE::from(*v)).collect())
                .expect("power of two"),
        ),
        &domain,
    )
    .expect("host encode");
    let (host, _) = multilinear::whir::fold_codeword_k_on_host::<Gl, Gl, Ext3>(
        &host_codeword,
        &domain,
        &ext(&alphas),
    )
    .expect("host fold");
    assert!(
        host == ext(&encoded_raw),
        "2^{num_vars}: the encoded successor is not the host fold"
    );

    // The message is the next group's factor: the encode read a copy.
    let (_, message_after) = session.tables_to_host().expect("read back");
    assert_eq!(
        message_after, message_before,
        "the encode moved the message"
    );

    // A near miss must differ: the fold at another first challenge (one
    // limb moved, which a fold of any number of levels has).
    let mut moved = alphas.clone();
    moved[0] = if moved[0] == 0 { 1 } else { moved[0] - 1 };
    let other = math_cuda::whir::fold_resident_fused(&committed, two_inv, &g_invs, &moved)
        .expect("fused fold");
    assert_ne!(
        canonical(&other.to_host().expect("read back")),
        canonical(&encoded_raw),
        "the comparison must see the challenge"
    );
}

/// The production first fold, at two stacks: the session's rounds and the
/// column engine's transform at more than one shape.
#[test]
fn the_encoded_successor_is_the_fold_at_the_production_first_fold() {
    the_encoded_successor_is_the_fold(20, 6, 0x5A5A);
    the_encoded_successor_is_the_fold(12, 6, 0x0A0A);
}

/// Other first folds: the encode is not tied to six.
#[test]
fn the_encoded_successor_is_the_fold_at_other_first_folds() {
    for levels in [1, 4] {
        the_encoded_successor_is_the_fold(16, levels, 0x1234 + levels as u64);
    }
}
