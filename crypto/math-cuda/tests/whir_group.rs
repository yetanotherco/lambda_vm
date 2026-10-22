//! A grouped WHIR chain's two card steps against host references
//! (`GroupFold::Indexed`, D-ONECHAIN §2.5): the run's bound tables laid slot by
//! slot for the index rounds ([`OpeningSession::gather_slots`]), and its
//! polynomials' folds combined into the round-0 successor
//! ([`math_cuda::whir::combine_codewords`]).
//!
//! Runs on a GPU box (`cargo test -p math-cuda --release --test whir_group`).
//! As VALUES (canonical limbs), as every card step here is compared. The whole
//! grouped chain against the host's bytes is
//! `crypto/multilinear/tests/whir_group_device.rs`.

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

fn ext(raw: &[u64]) -> Vec<FE3> {
    raw.chunks_exact(3)
        .map(|l| FE3::new([FE::from(l[0]), FE::from(l[1]), FE::from(l[2])]))
        .collect()
}

/// `parts` sessions over `2^num_vars`, each then taken down to its first
/// `2^(num_vars − bound)` values as a sumcheck's rounds would leave it, laid
/// into `2^index_vars` slots: the gathered tables are each part's live values
/// in slot order, then zeros — in the first part's own allocation when it has
/// the room (`bound ≥ index_vars`), in a fresh one otherwise.
fn the_slots_are_the_parts_in_order(
    parts: usize,
    index_vars: usize,
    num_vars: usize,
    bound: usize,
    seed: u64,
) {
    let mut rng = Rng(seed);
    let len = 1usize << num_vars;
    let live = len >> bound;
    let mut sessions = Vec::with_capacity(parts);
    let mut want_weight = Vec::new();
    let mut want_message = Vec::new();
    for _ in 0..parts {
        let weight: Vec<u64> = (0..len * 3).map(|_| rng.next()).collect();
        let message: Vec<u64> = (0..len).map(|_| rng.next()).collect();
        let mut session = OpeningSession::new(&weight, &message)
            .unwrap_or_else(|e| panic!("session (needs a GPU): {e:?}"));
        session.bound(bound);
        want_weight.extend_from_slice(&weight[..live * 3]);
        // The message is lifted on the card: `[v, 0, 0]`.
        want_message.extend(message[..live].iter().flat_map(|&v| [v, 0, 0]));
        sessions.push(session);
    }
    want_weight.resize((live * 3) << index_vars, 0);
    want_message.resize((live * 3) << index_vars, 0);
    let gathered = OpeningSession::gather_slots(sessions, index_vars).expect("gather");
    assert_eq!(gathered.len(), live << index_vars, "the slots' cube");
    let (weight, message) = gathered.tables_to_host().expect("read back");
    let tag = format!("{parts} parts in 2^{index_vars} slots, 2^{num_vars} bound {bound}");
    assert_eq!(
        canonical(&weight),
        canonical(&want_weight),
        "{tag}: the weight"
    );
    assert_eq!(
        canonical(&message),
        canonical(&want_message),
        "{tag}: the message"
    );
}

#[test]
fn gathered_slots_are_the_parts_in_slot_order_with_the_rest_zero() {
    for (parts, index_vars) in [(2, 1), (3, 2), (4, 2)] {
        // In place: the first part's allocation holds every slot.
        the_slots_are_the_parts_in_order(parts, index_vars, 12, index_vars, 7 + parts as u64);
        the_slots_are_the_parts_in_order(parts, index_vars, 12, 3, 11 + parts as u64);
        // A fresh pair: it does not.
        the_slots_are_the_parts_in_order(parts, index_vars, 12, index_vars - 1, 13 + parts as u64);
    }
}

#[test]
fn gathering_more_parts_than_slots_or_unequal_cubes_is_refused() {
    let mut rng = Rng(5);
    let session = |rng: &mut Rng, num_vars: usize| {
        let len = 1usize << num_vars;
        let weight: Vec<u64> = (0..len * 3).map(|_| rng.next()).collect();
        let message: Vec<u64> = (0..len).map(|_| rng.next()).collect();
        OpeningSession::new(&weight, &message)
            .unwrap_or_else(|e| panic!("session (needs a GPU): {e:?}"))
    };
    let three: Vec<OpeningSession> = (0..3).map(|_| session(&mut rng, 10)).collect();
    assert!(
        OpeningSession::gather_slots(three, 1).is_err(),
        "three parts in two slots"
    );
    let unequal = vec![session(&mut rng, 10), session(&mut rng, 9)];
    assert!(
        OpeningSession::gather_slots(unequal, 1).is_err(),
        "cubes of two sizes"
    );
}

/// Two-inverse and each level's inverse generator from a domain with
/// generator `g`: what `multilinear::gpu` passes a fold.
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

/// `parts` base codewords committed and folded `levels` times at one set of
/// challenges, then combined with one scale each: the sum is the host's
/// `Σ_p scale_p·fold_p` of the same folds, value for value. A near miss — two
/// scales exchanged — differs.
fn the_combined_successor_is_the_scaled_sum(parts: usize, num_vars: usize, seed: u64) {
    the_combined_successor_is_the_scaled_sum_at(parts, num_vars, seed, 2);
}

/// [`the_combined_successor_is_the_scaled_sum`] at blowup `2^log_blowup`.
fn the_combined_successor_is_the_scaled_sum_at(
    parts: usize,
    num_vars: usize,
    seed: u64,
    log_blowup: usize,
) {
    let levels = 6;
    let mut rng = Rng(seed);
    let len = 1usize << num_vars;
    let domain = Domain::<Gl>::new(num_vars + log_blowup).expect("domain");
    let (two_inv, g_invs) = fold_scalars(*domain.generator(), levels);
    let alphas: Vec<u64> = (0..levels * 3).map(|_| rng.next()).collect();
    let folds: Vec<math_cuda::whir::DeviceCodeword> = (0..parts)
        .map(|_| {
            let evals: Vec<u64> = (0..len).map(|_| rng.next()).collect();
            let (committed, _root) = math_cuda::whir::commit_codeword(
                &evals,
                log_blowup,
                levels,
                true,
                math_cuda::DeviceHash::Keccak256,
            )
            .unwrap_or_else(|e| panic!("device commit (needs a GPU): {e:?}"));
            math_cuda::whir::fold_resident_fused(&committed, two_inv, &g_invs, &alphas)
                .expect("fused fold")
        })
        .collect();
    let scales: Vec<[u64; 3]> = (0..parts)
        .map(|_| [rng.next(), rng.next(), rng.next()])
        .collect();
    let refs: Vec<&math_cuda::whir::DeviceCodeword> = folds.iter().collect();
    let combined = math_cuda::whir::combine_codewords(&refs, &scales).expect("combine");
    assert!(!combined.is_base(), "the successor is extension-valued");
    assert_eq!(combined.elements(), folds[0].elements());

    let host_folds: Vec<Vec<FE3>> = folds
        .iter()
        .map(|f| ext(&f.to_host().expect("read back")))
        .collect();
    let scaled_sum = |scales: &[[u64; 3]]| -> Vec<FE3> {
        let mut sum = vec![FE3::zero(); host_folds[0].len()];
        for (fold, scale) in host_folds.iter().zip(scales) {
            let scale = ext(scale)[0];
            for (acc, v) in sum.iter_mut().zip(fold) {
                *acc += &scale * v;
            }
        }
        sum
    };
    let got = ext(&combined.to_host().expect("read back"));
    assert!(
        got == scaled_sum(&scales),
        "{parts} parts at 2^{num_vars}: the combined successor is not the scaled sum"
    );
    let mut exchanged = scales.clone();
    exchanged.swap(0, 1);
    assert!(
        got != scaled_sum(&exchanged),
        "{parts} parts: the comparison must see which scale weighs which fold"
    );
}

#[test]
fn a_combined_successor_is_the_eq_weighted_sum_of_the_folds() {
    for parts in 2..=4 {
        the_combined_successor_is_the_scaled_sum(parts, 14, 31 + parts as u64);
    }
}

/// Blowup 2 (`LAMBDA_VM_ZF_WHIR_BLOWUP=2`): the same sum over codewords half
/// as long.
#[test]
fn a_combined_successor_at_blowup_two_is_the_eq_weighted_sum_of_the_folds() {
    for parts in 2..=4 {
        the_combined_successor_is_the_scaled_sum_at(parts, 15, 41 + parts as u64, 1);
    }
}

#[test]
fn combining_codewords_of_two_lengths_or_a_base_one_is_refused() {
    let mut rng = Rng(9);
    let commit = |rng: &mut Rng, num_vars: usize| {
        let evals: Vec<u64> = (0..1usize << num_vars).map(|_| rng.next()).collect();
        math_cuda::whir::commit_codeword(&evals, 2, 6, true, math_cuda::DeviceHash::Keccak256)
            .unwrap_or_else(|e| panic!("device commit (needs a GPU): {e:?}"))
            .0
    };
    let base = commit(&mut rng, 14);
    let other = commit(&mut rng, 14);
    let scale = [[1, 0, 0], [2, 0, 0]];
    assert!(
        math_cuda::whir::combine_codewords(&[&base, &other], &scale).is_err(),
        "base codewords"
    );
    let domain = |n: usize| Domain::<Gl>::new(n + 2).expect("domain");
    let fold = |c: &math_cuda::whir::DeviceCodeword, n: usize| {
        let (two_inv, g_invs) = fold_scalars(*domain(n).generator(), 6);
        math_cuda::whir::fold_resident_fused(c, two_inv, &g_invs, &[3u64; 18]).expect("fold")
    };
    let short = commit(&mut rng, 13);
    let (a, b) = (fold(&base, 14), fold(&short, 13));
    assert!(
        math_cuda::whir::combine_codewords(&[&a, &b], &scale).is_err(),
        "two lengths"
    );
    assert!(
        math_cuda::whir::combine_codewords(&[&a], &scale).is_err(),
        "a scale without a part"
    );
}
