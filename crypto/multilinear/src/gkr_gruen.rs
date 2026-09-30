//! A GKR layer's device rounds with Gruen's split (D-ARGUE S1-3,
//! `LAMBDA_VM_ARGUE_GKR_GRUEN`): the same round messages as today's, from half
//! the work.
//!
//! # What a round sends, and what the card sums
//!
//! A layer's relation is `eq(u, x)·h(x)` with `h = p_lo·q_hi + p_hi·q_lo +
//! λ·q_lo·q_hi` over `m` variables. Round `j`, after challenges `s_{<j}`, sends
//! `s(1), s(2), s(3)` of
//!
//! ```text
//! s(t) = E·ℓ(t)·H(t),   E = Π_{i<j} eq₁(u_i, s_i),   ℓ(t) = eq₁(u_j, t) = (1 − u_j) + t·(2u_j − 1),
//! H(t) = Σ_{x'} eq(u_{>j}, x')·h(s_{<j}, t, x')
//! ```
//!
//! by the product form of `eq`. `H` is a quadratic. The card sums `H(1)` and
//! `H(2)` — no `eq` factor in the walk, every node by additions — and the host
//! finishes: `s(1) = E·u_j·H(1)`, `s(0) = claim − s(1)`, `E·H(0) = s(0)/(1 − u_j)`,
//! `H(3) = H(0) + 3·(H(2) − H(1))` (a quadratic's third difference is zero),
//! then `s(2)`, `s(3)`. Where `1 − u_j = 0` the card sums `H(0)` too.
//!
//! # Why the messages are today's
//!
//! Each is the field element today's round computes: the product form and
//! distributivity move no value, the extrapolation is exact for a quadratic,
//! and `s(0) + s(1) = claim` holds for today's own messages — the verifier
//! checks it — whenever the layer is the fold of the one below, which it is
//! for a tree the card folded. So the transcript and the canonical bytes do not
//! move. Raw limbs may (`thoughts/zf/gap2/fix2/D-ARGUE.md` §2.7 d); no host-only
//! proof reaches this path.
//!
//! The rounds stop at a cube of [`gkr_gruen_tail`] and hand the host today's
//! five factors there, `eq` included, for the tail `gkr::prove` already runs.

// The host's half of the rounds, the counters and the fault are the device
// driver's (`gpu::DeviceTree::prove_layer_gruen`): a build without a device
// reaches only the knobs and the tests.
#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

use std::sync::atomic::{AtomicU8, AtomicU64, AtomicUsize, Ordering};

use math::field::{element::FieldElement, traits::IsField};

use crate::gkr::Cubic;

/// Whether a device GKR layer runs Gruen's rounds (`LAMBDA_VM_ARGUE_GKR_GRUEN`,
/// any non-empty value other than `0`). Off by default until its A/B; off is
/// today's rounds exactly. Read once, with a banner.
pub fn argue_gkr_gruen() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    forced(&GRUEN_FORCED, || {
        *ON.get_or_init(|| {
            let on = on_from(std::env::var("LAMBDA_VM_ARGUE_GKR_GRUEN").ok().as_deref());
            eprintln!(
                "★ ARGUE GKR GRUEN: {}",
                if on {
                    format!(
                        "on (LAMBDA_VM_ARGUE_GKR_GRUEN=1; the host tail from a cube of {})",
                        gkr_gruen_tail()
                    )
                } else {
                    "off (today's layer rounds)".to_string()
                }
            );
            on
        })
    })
}

/// Whether every Gruen round is checked against today's
/// (`LAMBDA_VM_ARGUE_GKR_GRUEN_XCHECK`): today's program walks the same folded
/// halves beside it, with its own `eq` table folded on the same challenges,
/// and a difference fails the prove. The bound factors are compared too.
pub fn argue_gkr_gruen_xcheck() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    forced(&XCHECK_FORCED, || {
        *ON.get_or_init(|| {
            on_from(
                std::env::var("LAMBDA_VM_ARGUE_GKR_GRUEN_XCHECK")
                    .ok()
                    .as_deref(),
            )
        })
    })
}

/// The cube a Gruen layer hands the host (`LAMBDA_VM_ARGUE_GKR_GRUEN_TAIL`, a
/// power of two, at least 2; default [`GRUEN_TAIL_DEFAULT`]).
///
/// Today's layers stop at `HOST_CUBE_DIRECT`, 512: a round there cost a
/// program walk, a fold launch and a round trip. A Gruen round is one launch
/// and one read-back, which undercuts a host round over more than a few dozen
/// pairs.
pub fn gkr_gruen_tail() -> usize {
    match TAIL_FORCED.load(Ordering::Relaxed) {
        0 => {
            static TAIL: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
            *TAIL.get_or_init(|| {
                tail_from(
                    std::env::var("LAMBDA_VM_ARGUE_GKR_GRUEN_TAIL")
                        .ok()
                        .as_deref(),
                )
            })
        }
        cube => cube,
    }
}

/// Where a Gruen layer's rounds stop unless the knob says otherwise.
pub const GRUEN_TAIL_DEFAULT: usize = 64;

fn tail_from(value: Option<&str>) -> usize {
    match value.map(str::parse::<usize>) {
        None => GRUEN_TAIL_DEFAULT,
        Some(Ok(cube)) if cube >= 2 && cube.is_power_of_two() => cube,
        Some(_) => {
            eprintln!(
                "[gkr] LAMBDA_VM_ARGUE_GKR_GRUEN_TAIL must be a power of two ≥ 2; using {GRUEN_TAIL_DEFAULT}"
            );
            GRUEN_TAIL_DEFAULT
        }
    }
}

fn on_from(value: Option<&str>) -> bool {
    value.is_some_and(|v| !v.is_empty() && v != "0")
}

static GRUEN_FORCED: AtomicU8 = AtomicU8::new(0);
static XCHECK_FORCED: AtomicU8 = AtomicU8::new(0);
static TAIL_FORCED: AtomicUsize = AtomicUsize::new(0);

fn forced(cell: &AtomicU8, env: impl FnOnce() -> bool) -> bool {
    match cell.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => env(),
    }
}

fn store_forced(cell: &AtomicU8, on: Option<bool>) {
    cell.store(
        match on {
            Some(true) => 1,
            Some(false) => 2,
            None => 0,
        },
        Ordering::Relaxed,
    );
}

/// Overrides `LAMBDA_VM_ARGUE_GKR_GRUEN` for the whole process — for a test
/// that proves both ways in one binary. `None` restores the environment's.
#[doc(hidden)]
pub fn force_argue_gkr_gruen(on: Option<bool>) {
    store_forced(&GRUEN_FORCED, on);
}

/// The same for `LAMBDA_VM_ARGUE_GKR_GRUEN_XCHECK`.
#[doc(hidden)]
pub fn force_argue_gkr_gruen_xcheck(on: Option<bool>) {
    store_forced(&XCHECK_FORCED, on);
}

/// The same for `LAMBDA_VM_ARGUE_GKR_GRUEN_TAIL`; `None` restores it.
#[doc(hidden)]
pub fn force_gkr_gruen_tail(cube: Option<usize>) {
    let cube = cube.unwrap_or(0);
    assert!(
        cube == 0 || (cube >= 2 && cube.is_power_of_two()),
        "a tail cube is a power of two ≥ 2"
    );
    TAIL_FORCED.store(cube, Ordering::Relaxed);
}

/// On this thread, every round sums `H(0)` on the card as well, where it would
/// come from the claim — so a test can show that path sends the same messages.
/// Not for production callers.
#[doc(hidden)]
pub fn force_gkr_gruen_direct_h0(on: bool) {
    DIRECT_H0.with(|cell| cell.set(on));
}

pub(crate) fn direct_h0_forced() -> bool {
    DIRECT_H0.with(std::cell::Cell::get)
}

thread_local! {
    static DIRECT_H0: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };

    /// ⛔ A FAULT: while armed on a thread, each Gruen layer it proves adds one
    /// to its first round's `s(1)` — what a round that summed one term wrong
    /// would send — so the checks after it can be shown to see one. Per
    /// thread: a test's fault must not reach a proof running beside it.
    static GRUEN_FAULT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[doc(hidden)]
pub fn force_gkr_gruen_fault(on: bool) {
    GRUEN_FAULT.with(|cell| cell.set(on));
}

pub(crate) fn gkr_gruen_fault() -> bool {
    GRUEN_FAULT.with(std::cell::Cell::get)
}

static GRUEN_LAYERS: AtomicU64 = AtomicU64::new(0);
static GRUEN_ROUNDS: AtomicU64 = AtomicU64::new(0);
static GRUEN_XCHECKED: AtomicU64 = AtomicU64::new(0);
static GRUEN_XCHECK_SKIPPED: AtomicU64 = AtomicU64::new(0);
static GRUEN_DIRECT_H0: AtomicU64 = AtomicU64::new(0);

/// Device GKR layers that ran Gruen's rounds.
pub fn gruen_layers() -> u64 {
    GRUEN_LAYERS.load(Ordering::Relaxed)
}

/// Their card rounds.
pub fn gruen_rounds() -> u64 {
    GRUEN_ROUNDS.load(Ordering::Relaxed)
}

/// Of [`gruen_layers`], the ones the cross-check compared with today's rounds,
/// round by round and factor by factor, and found equal.
pub fn gruen_xchecked() -> u64 {
    GRUEN_XCHECKED.load(Ordering::Relaxed)
}

/// Layers the cross-check could not shadow: the card had no room for today's
/// `eq` table beside them.
pub fn gruen_xcheck_skipped() -> u64 {
    GRUEN_XCHECK_SKIPPED.load(Ordering::Relaxed)
}

/// Rounds that summed `H(0)` on the card (`1 − u_j = 0`, or forced).
pub fn gruen_direct_h0() -> u64 {
    GRUEN_DIRECT_H0.load(Ordering::Relaxed)
}

pub(crate) fn note_layer(rounds: u64, xchecked: bool) {
    GRUEN_LAYERS.fetch_add(1, Ordering::Relaxed);
    GRUEN_ROUNDS.fetch_add(rounds, Ordering::Relaxed);
    if xchecked {
        GRUEN_XCHECKED.fetch_add(1, Ordering::Relaxed);
    }
}

pub(crate) fn note_xcheck_skipped() {
    GRUEN_XCHECK_SKIPPED.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn note_direct_h0() {
    GRUEN_DIRECT_H0.fetch_add(1, Ordering::Relaxed);
}

/// The host's half of a Gruen layer: the claim and `E` carried from round to
/// round, and each round's message made from the card's sums.
pub(crate) struct Chain<F: IsField> {
    claim: FieldElement<F>,
    /// `E = Π_{i<j} eq₁(u_i, s_i)`.
    kappa: FieldElement<F>,
    cubic: Cubic<F>,
}

/// What a round sends, and the `s(0)` the claim implies — the fourth value the
/// next claim is interpolated from.
pub(crate) struct Message<F: IsField> {
    pub sent: Vec<FieldElement<F>>,
    pub s0: FieldElement<F>,
}

impl<F: IsField> Chain<F> {
    /// From the claim the layer's sumcheck starts at, `p + λ·q` of the level
    /// above. `None` if the cubic's denominators have no inverse (a field of
    /// characteristic 2 or 3).
    pub(crate) fn new(claim: FieldElement<F>) -> Option<Self> {
        Some(Self {
            claim,
            kappa: FieldElement::one(),
            cubic: Cubic::new()?,
        })
    }

    /// `E`, the scalar the rounds so far leave on the weight.
    pub(crate) fn kappa(&self) -> &FieldElement<F> {
        &self.kappa
    }

    /// Round `j`'s message from the card's `H(1)`, `H(2)` and, when there is
    /// no `1/(1 − u_j)`, `H(0)`. `None` when neither is given.
    pub(crate) fn message(
        &self,
        rho: &FieldElement<F>,
        inverse_left: Option<&FieldElement<F>>,
        h1: &FieldElement<F>,
        h2: &FieldElement<F>,
        h0: Option<&FieldElement<F>>,
    ) -> Option<Message<F>> {
        let one = FieldElement::<F>::one();
        let big1 = &self.kappa * h1;
        let big2 = &self.kappa * h2;
        let s1 = rho * &big1;
        let s0 = &self.claim - &s1;
        let big0 = match h0 {
            Some(h0) => &self.kappa * h0,
            None => &s0 * inverse_left?,
        };
        let big3 = &big0 + (&big2 - &big1) * FieldElement::<F>::from(3u64);
        let step = rho + rho - &one;
        let l2 = rho + &step;
        let l3 = &l2 + &step;
        Some(Message {
            sent: vec![s1, &l2 * &big2, &l3 * &big3],
            s0,
        })
    }

    /// Past round `j`, whose challenge was `z`.
    pub(crate) fn advance(
        &mut self,
        rho: &FieldElement<F>,
        message: &Message<F>,
        z: &FieldElement<F>,
    ) {
        let one = FieldElement::<F>::one();
        let [s1, s2, s3] = [&message.sent[0], &message.sent[1], &message.sent[2]];
        self.claim = self.cubic.at([&message.s0, s1, s2, s3], z);
        let step = rho + rho - &one;
        self.kappa = &self.kappa * ((&one - rho) + z * &step);
    }
}

/// `1/(1 − u_j)` for each round, `None` where there is none — those rounds sum
/// `H(0)` on the card.
pub(crate) fn inverse_lefts<F: IsField>(rhos: &[FieldElement<F>]) -> Vec<Option<FieldElement<F>>> {
    let one = FieldElement::<F>::one();
    let lefts: Vec<FieldElement<F>> = rhos.iter().map(|r| &one - r).collect();
    let zero = FieldElement::<F>::zero();
    let mut live: Vec<FieldElement<F>> = lefts
        .iter()
        .map(|l| if *l == zero { one.clone() } else { l.clone() })
        .collect();
    if FieldElement::inplace_batch_inverse(&mut live).is_err() {
        return vec![None; rhos.len()];
    }
    lefts
        .iter()
        .zip(live)
        .map(|(l, inv)| (*l != zero && !direct_h0_forced()).then_some(inv))
        .collect()
}

/// The device algorithm on the host, value for value — the fused fold, the
/// split weight, the claim's `H(0)` — for checking its arithmetic against the
/// generic rounds without a card. Returns the rounds, their challenges and the
/// five factors (`eq` first) over `2^low` cells, as `gpu::DeviceTree::prove_layer`
/// hands them back.
#[cfg(test)]
pub(crate) fn host_rounds<F, T>(
    halves: [Vec<FieldElement<F>>; 4],
    point: &[FieldElement<F>],
    lambda: &FieldElement<F>,
    claim: FieldElement<F>,
    low: usize,
    transcript: &mut T,
) -> crate::gpu::LayerRounds<F>
where
    F: IsField + 'static,
    T: crypto::fiat_shamir::is_transcript::IsTranscript<F>,
{
    let m = point.len();
    let low = low.min(m);
    let rounds = m - low;
    // `e_lo = eq(u_{J..})`, and `e_hi` level j = `eq(u_{j+1..J})`, as the kernel lays them out.
    let e_lo = crate::eq::eq_evals(&point[rounds..]);
    let e_hi: Vec<Vec<FieldElement<F>>> = (0..rounds)
        .map(|j| crate::eq::eq_evals(&point[j + 1..rounds]))
        .collect();
    let lefts = inverse_lefts(&point[..rounds]);
    let mut f = halves;
    let mut chain = Chain::new(claim).expect("a field with 2 and 3 invertible");
    let mut proofs = Vec::new();
    let mut challenges: Vec<FieldElement<F>> = Vec::new();
    let h = |v: [&FieldElement<F>; 4]| v[0] * v[3] + v[1] * v[2] + lambda * &(v[2] * v[3]);
    for j in 0..rounds {
        let quarter = 1usize << (m - j - 1);
        if let Some(s) = challenges.last() {
            for g in f.iter_mut() {
                for x in 0..2 * quarter {
                    let folded = &g[x] + s * &(&g[x + 2 * quarter] - &g[x]);
                    g[x] = folded;
                }
            }
        }
        let want_h0 = lefts[j].is_none();
        let (mut h1, mut h2, mut h0) = (
            FieldElement::zero(),
            FieldElement::zero(),
            FieldElement::zero(),
        );
        for x in 0..quarter {
            let w = &e_hi[j][x >> low] * &e_lo[x & ((1 << low) - 1)];
            let lo: [&FieldElement<F>; 4] = std::array::from_fn(|k| &f[k][x]);
            let hi: [&FieldElement<F>; 4] = std::array::from_fn(|k| &f[k][x + quarter]);
            let two: [FieldElement<F>; 4] = std::array::from_fn(|k| hi[k] + hi[k] - lo[k]);
            h1 += &w * h(hi);
            h2 += &w * h([&two[0], &two[1], &two[2], &two[3]]);
            if want_h0 {
                h0 += &w * h(lo);
            }
        }
        let message = chain
            .message(
                &point[j],
                lefts[j].as_ref(),
                &h1,
                &h2,
                want_h0.then_some(&h0),
            )
            .expect("H(0) or its inverse");
        for value in &message.sent {
            transcript.append_field_element(value);
        }
        let z = transcript.sample_field_element();
        chain.advance(&point[j], &message, &z);
        proofs.push(crate::sumcheck::RoundProof {
            evaluations: message.sent,
        });
        challenges.push(z);
    }
    let cells = 1usize << low;
    if let Some(s) = challenges.last() {
        for g in f.iter_mut() {
            for y in 0..cells {
                let folded = &g[y] + s * &(&g[y + cells] - &g[y]);
                g[y] = folded;
            }
        }
    }
    let eq: Vec<FieldElement<F>> = e_lo.iter().map(|v| chain.kappa() * v).collect();
    let mut factors = vec![crate::mle::Mle::new(eq).expect("a cube")];
    for g in f {
        factors.push(crate::mle::Mle::new(g[..cells].to_vec()).expect("a cube"));
    }
    (proofs, challenges, factors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gkr::LayerRelation;
    use crypto::fiat_shamir::{default_transcript::DefaultTranscript, is_transcript::IsTranscript};
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;

    type FE = FieldElement<Ext3>;

    fn value(seed: &mut u64) -> FE {
        let mut next = || {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            *seed
        };
        FE::new([next().into(), next().into(), next().into()])
    }

    /// A layer of `2^(m+1)` fractions, its point, `λ` and the claim the level
    /// above makes of it — what `gkr::prove` hands a device layer.
    fn layer(m: usize, seed: u64) -> ([Vec<FE>; 4], Vec<FE>, FE, FE) {
        let mut seed = seed | 1;
        let halves: [Vec<FE>; 4] =
            std::array::from_fn(|_| (0..1 << m).map(|_| value(&mut seed)).collect());
        let point: Vec<FE> = (0..m).map(|_| value(&mut seed)).collect();
        let lambda = value(&mut seed);
        let eq = crate::eq::eq_evals(&point);
        let claim = (0..1usize << m).fold(FE::zero(), |acc, x| {
            let [a, b, c, d] = [&halves[0][x], &halves[1][x], &halves[2][x], &halves[3][x]];
            acc + &eq[x] * (a * d + b * c + &lambda * &(c * d))
        });
        (halves, point, lambda, claim)
    }

    /// Today's rounds over the same layer, as a device session runs them: the
    /// generic relation, stopped at `2^low`.
    fn today(
        halves: &[Vec<FE>; 4],
        point: &[FE],
        lambda: &FE,
        low: usize,
        transcript: &mut DefaultTranscript<Ext3>,
    ) -> crate::gpu::LayerRounds<Ext3> {
        let mut polys = vec![crate::eq::eq_mle(point).unwrap()];
        polys.extend(
            halves
                .iter()
                .map(|h| crate::mle::Mle::new(h.clone()).unwrap()),
        );
        let mut relation = LayerRelation::from_factors(polys, *lambda).unwrap();
        let rounds = point.len() - low.min(point.len());
        let (proofs, challenges) =
            crate::sumcheck::prove_rounds(&mut relation, rounds, transcript).unwrap();
        (proofs, challenges, relation.polys_for_tests().to_vec())
    }

    /// The rounds' values and the factors' tables, which compare.
    fn values(layer: &crate::gpu::LayerRounds<Ext3>) -> (Vec<Vec<FE>>, Vec<Vec<FE>>) {
        (
            layer.0.iter().map(|r| r.evaluations.clone()).collect(),
            layer.2.iter().map(|f| f.evals().to_vec()).collect(),
        )
    }

    fn assert_same(m: usize, low: usize, seed: u64) {
        let (halves, point, lambda, claim) = layer(m, seed);
        let mut a = DefaultTranscript::<Ext3>::new(b"gruen");
        let mut b = DefaultTranscript::<Ext3>::new(b"gruen");
        let want = today(&halves, &point, &lambda, low, &mut a);
        let got = host_rounds(halves, &point, &lambda, claim, low, &mut b);
        let (got_rounds, got_factors) = values(&got);
        let (want_rounds, want_factors) = values(&want);
        assert_eq!(
            got_rounds, want_rounds,
            "m {m}, low {low}: the rounds differ"
        );
        assert_eq!(got.1, want.1, "m {m}, low {low}: the challenges differ");
        assert_eq!(
            got_factors, want_factors,
            "m {m}, low {low}: the factors handed to the tail differ"
        );
        assert_eq!(
            a.state(),
            b.state(),
            "m {m}, low {low}: the transcripts parted"
        );
    }

    /// ★ The device algorithm, run on the host, sends today's rounds and hands
    /// the tail today's factors, at every split of card rounds and host tail.
    #[test]
    fn gruen_rounds_are_todays() {
        for m in 1..=9 {
            for low in [0, 1, 2, 5, 6, 9] {
                assert_same(m, low, 0x9E37_79B9 + (m * 31 + low) as u64);
            }
        }
    }

    /// The same with `H(0)` summed on the card every round.
    #[test]
    fn gruen_rounds_with_a_direct_h0_are_todays() {
        force_gkr_gruen_direct_h0(true);
        let outcome = std::panic::catch_unwind(|| {
            for m in [3, 7] {
                assert_same(m, 2, 0xABCD + m as u64);
            }
        });
        force_gkr_gruen_direct_h0(false);
        outcome.unwrap();
    }

    /// A point coordinate of one — `1 − u_j = 0`, no inverse — sums `H(0)`
    /// directly for that round, and still sends today's rounds.
    #[test]
    fn a_coordinate_of_one_takes_the_direct_h0() {
        let (halves, mut point, lambda, _) = layer(6, 77);
        point[1] = FE::one();
        point[3] = FE::one();
        let eq = crate::eq::eq_evals(&point);
        let claim = (0..1usize << 6).fold(FE::zero(), |acc, x| {
            let [a, b, c, d] = [&halves[0][x], &halves[1][x], &halves[2][x], &halves[3][x]];
            acc + &eq[x] * (a * d + b * c + &lambda * &(c * d))
        });
        assert_eq!(
            inverse_lefts(&point).iter().filter(|l| l.is_none()).count(),
            2,
            "two rounds have no 1/(1 − u_j)"
        );
        let mut a = DefaultTranscript::<Ext3>::new(b"gruen");
        let mut b = DefaultTranscript::<Ext3>::new(b"gruen");
        let want = today(&halves, &point, &lambda, 1, &mut a);
        let got = host_rounds(halves, &point, &lambda, claim, 1, &mut b);
        assert_eq!(values(&got), values(&want));
    }

    /// ⛔ The comparison above can fail: a claim off by one changes the rounds
    /// that take `H(0)` from it.
    #[test]
    fn a_wrong_claim_changes_the_rounds() {
        let (halves, point, lambda, claim) = layer(5, 5);
        let mut a = DefaultTranscript::<Ext3>::new(b"gruen");
        let mut b = DefaultTranscript::<Ext3>::new(b"gruen");
        let want = today(&halves, &point, &lambda, 1, &mut a);
        let got = host_rounds(halves, &point, &lambda, claim + FE::one(), 1, &mut b);
        assert_ne!(
            values(&got).0,
            values(&want).0,
            "the claim must reach the messages"
        );
    }

    #[test]
    fn the_knobs_read_their_variables() {
        assert!(!on_from(None));
        assert!(!on_from(Some("0")));
        assert!(!on_from(Some("")));
        assert!(on_from(Some("1")));
        assert_eq!(tail_from(None), GRUEN_TAIL_DEFAULT);
        assert_eq!(tail_from(Some("32")), 32);
        assert_eq!(tail_from(Some("512")), 512);
        assert_eq!(tail_from(Some("48")), GRUEN_TAIL_DEFAULT);
        assert_eq!(tail_from(Some("1")), GRUEN_TAIL_DEFAULT);
        assert_eq!(tail_from(Some("x")), GRUEN_TAIL_DEFAULT);
    }
}
