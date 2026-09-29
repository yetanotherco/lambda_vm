//! LogUp as a tree of fractions: `p₁/q₁ + p₂/q₂ = (p₁q₂ + p₂q₁)/(q₁q₂)`, added
//! pairwise up a binary tree with GKR proving each layer against the one below.
//! Only the input layer is ever committed.
//!
//! Proves the sum is whatever the output claims. Checking that the output
//! numerator is zero — the bus balance — is the caller's.

use crypto::fiat_shamir::is_transcript::IsTranscript;
use math::field::{element::FieldElement, traits::IsField};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::{
    Error,
    eq::{eq_eval, eq_mle},
    mle::Mle,
    poly::SumcheckPolynomial,
    program::{Builder, Program},
    sumcheck::{self, SumcheckProof},
    whir_split,
};

/// One level of the tree: numerators and denominators over the same cube.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FractionLayer<F: IsField> {
    pub p: Mle<F>,
    pub q: Mle<F>,
}

impl<F: IsField + 'static> FractionLayer<F> {
    pub fn new(p: Mle<F>, q: Mle<F>) -> Result<Self, Error> {
        if p.num_vars() != q.num_vars() {
            return Err(Error::VariableCountMismatch {
                expected: p.num_vars(),
                got: q.num_vars(),
            });
        }
        Ok(Self { p, q })
    }

    pub fn num_vars(&self) -> usize {
        self.p.num_vars()
    }

    /// Adds the two halves pointwise, giving the layer one level up.
    pub fn fold(&self) -> Result<Self, Error>
    where
        FieldElement<F>: Send + Sync,
    {
        if self.num_vars() == 0 {
            return Err(Error::NoVariablesLeft);
        }
        let half = self.p.len() / 2;
        let (p_lo, p_hi) = self.p.evals().split_at(half);
        let (q_lo, q_hi) = self.q.evals().split_at(half);

        // Every index folds on its own, and building the tree is a pass over
        // the input layer at every level, so this is worth the pool.
        let both = |i: usize| {
            (
                &p_lo[i] * &q_hi[i] + &p_hi[i] * &q_lo[i],
                &q_lo[i] * &q_hi[i],
            )
        };
        #[cfg(feature = "parallel")]
        let (next_p, next_q): (Vec<_>, Vec<_>) = if half >= crate::SERIAL_BELOW {
            (0..half).into_par_iter().map(both).unzip()
        } else {
            (0..half).map(both).unzip()
        };
        #[cfg(not(feature = "parallel"))]
        let (next_p, next_q): (Vec<_>, Vec<_>) = (0..half).map(both).unzip();

        Self::new(Mle::new(next_p)?, Mle::new(next_q)?)
    }
}

/// The whole tree, from the input layer down to the single output fraction.
///
/// `layers[0]` is the output (zero variables); the last entry is the input.
#[derive(Debug)]
pub struct FractionTree<F: IsField> {
    /// Empty when the tree lives on a device, which holds every layer.
    layers: Vec<FractionLayer<F>>,
    device: Option<crate::gpu::DeviceTree>,
    num_layers: usize,
    output: (FieldElement<F>, FieldElement<F>),
}

impl<F: IsField + 'static> FractionTree<F> {
    /// Builds every layer by repeated folding.
    ///
    /// On a device the layers stay there: the tree is the biggest thing a
    /// table's argument holds, and GKR reads every level of it.
    pub fn build(input: FractionLayer<F>) -> Result<Self, Error>
    where
        FieldElement<F>: Send + Sync,
    {
        if let Some(device) = crate::gpu::build_tree(&input.p, &input.q) {
            return Self::from_device(device);
        }

        let mut layers = vec![input];
        while layers.last().expect("non-empty").num_vars() > 0 {
            let next = layers.last().expect("non-empty").fold()?;
            layers.push(next);
        }
        layers.reverse();
        let top = &layers[0];
        let output = (top.p.evals()[0].clone(), top.q.evals()[0].clone());
        let num_layers = layers.len();
        Ok(Self {
            layers,
            device: None,
            num_layers,
            output,
        })
    }

    /// A tree a device already holds, layers and all.
    pub fn from_device(device: crate::gpu::DeviceTree) -> Result<Self, Error> {
        let num_layers = device.num_layers();
        let output = device.output()?;
        Ok(Self {
            layers: Vec::new(),
            device: Some(device),
            num_layers,
            output,
        })
    }

    /// The output fraction `(p, q)`. The bus balances when `p` is zero.
    pub fn output(&self) -> (FieldElement<F>, FieldElement<F>) {
        self.output.clone()
    }

    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// The layer, for a tree that kept them here.
    pub fn layer(&self, i: usize) -> &FractionLayer<F> {
        &self.layers[i]
    }

    /// The device holding every layer, when one does.
    pub fn device(&self) -> Option<&crate::gpu::DeviceTree> {
        self.device.as_ref()
    }

    pub fn input_layer(&self) -> &FractionLayer<F> {
        self.layers.last().expect("non-empty")
    }

    /// The levels near the output, back here — `prefix[i]` is layer `i`.
    ///
    /// Empty for a tree that already lives here. One download brings the whole
    /// prefix: the levels above the deepest of them are its folds, and the
    /// deepest is a few kilobytes.
    fn host_prefix(&self) -> Vec<FractionLayer<F>>
    where
        FieldElement<F>: Send + Sync,
    {
        let Some(device) = self.device.as_ref() else {
            return Vec::new();
        };
        // Never the input layer: a tree that dropped it writes it again for its
        // own sumcheck, and it is the one level too big to walk here.
        let deepest = HOST_LAYER_VARS.min(self.num_layers.saturating_sub(2));
        let Some((p, q)) = device.layer_to_host::<F>(deepest) else {
            return Vec::new();
        };
        let Ok(layer) = FractionLayer::new(p, q) else {
            return Vec::new();
        };
        if layer.num_vars() != deepest {
            return Vec::new();
        }
        let mut prefix = vec![layer];
        while prefix.last().expect("non-empty").num_vars() > 0 {
            let Ok(next) = prefix.last().expect("non-empty").fold() else {
                return Vec::new();
            };
            prefix.push(next);
        }
        prefix.reverse();
        prefix
    }
}

/// The deepest level that comes here whole: the one whose halves are already
/// a cube of [`crate::HOST_CUBE_DIRECT`], so its sumcheck never goes to a device.
const HOST_LAYER_VARS: usize = crate::HOST_CUBE_DIRECT.trailing_zeros() as usize + 1;

/// The layer relation: `Σ_x eq(r,x)·[p_lo·q_hi + p_hi·q_lo + λ·q_lo·q_hi]`,
/// which equals `p_out(r) + λ·q_out(r)` when the layer really is the fold.
struct LayerRelation<F: IsField> {
    /// `[eq, p_lo, p_hi, q_lo, q_hi]`.
    polys: Vec<Mle<F>>,
    lambda: FieldElement<F>,
    /// The same rule as straight-line code, for the device path. `None` for a
    /// relation that is here on purpose — a level the tree handed over, or the
    /// tail of one — so the dispatch does not send it back.
    program: Option<Program<F>>,
}

impl<F: IsField + 'static> LayerRelation<F> {
    const EQ: usize = 0;
    const P_LO: usize = 1;
    const P_HI: usize = 2;
    const Q_LO: usize = 3;
    const Q_HI: usize = 4;

    fn new(
        next: &FractionLayer<F>,
        r: &[FieldElement<F>],
        lambda: FieldElement<F>,
        dispatchable: bool,
    ) -> Result<Self, Error> {
        let half = next.p.len() / 2;
        let split = |m: &Mle<F>| -> Result<(Mle<F>, Mle<F>), Error> {
            Ok((
                Mle::new(m.evals()[..half].to_vec())?,
                Mle::new(m.evals()[half..].to_vec())?,
            ))
        };
        let (p_lo, p_hi) = split(&next.p)?;
        let (q_lo, q_hi) = split(&next.q)?;
        Ok(Self {
            polys: vec![eq_mle(r)?, p_lo, p_hi, q_lo, q_hi],
            program: dispatchable
                .then(|| Self::program_for(&lambda))
                .transpose()?,
            lambda,
        })
    }

    /// The same relation over factors a device already folded: the weight and
    /// the four halves as its last round left them.
    fn from_factors(polys: Vec<Mle<F>>, lambda: FieldElement<F>) -> Result<Self, Error> {
        if polys.len() != 5 {
            return Err(Error::VariableCountMismatch {
                expected: 5,
                got: polys.len(),
            });
        }
        Ok(Self {
            polys,
            program: None,
            lambda,
        })
    }

    /// `eq·(p_lo·q_hi + p_hi·q_lo + lambda·q_lo·q_hi)`, the same expression
    /// [`combine`](SumcheckPolynomial::combine) evaluates.
    fn program_for(lambda: &FieldElement<F>) -> Result<Program<F>, Error> {
        let mut b = Builder::<F>::new();
        let eq = b.var(Self::EQ);
        let p_lo = b.var(Self::P_LO);
        let p_hi = b.var(Self::P_HI);
        let q_lo = b.var(Self::Q_LO);
        let q_hi = b.var(Self::Q_HI);
        let first = b.mul(p_lo, q_hi);
        let second = b.mul(p_hi, q_lo);
        let numerator = b.add(first, second);
        let denominator = b.mul(q_lo, q_hi);
        let weighted = b.fixed(lambda.clone());
        let scaled = b.mul(weighted, denominator);
        let sum = b.add(numerator, scaled);
        let root = b.mul(eq, sum);
        b.finish(root)
    }
}

/// `eq` times a product of two layer values.
const LAYER_DEGREE: usize = 3;

impl<F: IsField + 'static> SumcheckPolynomial<F> for LayerRelation<F> {
    fn num_vars(&self) -> usize {
        self.polys[Self::EQ].num_vars()
    }

    fn degree(&self) -> usize {
        LAYER_DEGREE
    }

    fn polys(&self) -> &[Mle<F>] {
        &self.polys
    }

    fn combine(&self, v: &[FieldElement<F>]) -> FieldElement<F> {
        let numerator = &v[Self::P_LO] * &v[Self::Q_HI] + &v[Self::P_HI] * &v[Self::Q_LO];
        let denominator = &v[Self::Q_LO] * &v[Self::Q_HI];
        &v[Self::EQ] * (numerator + &self.lambda * denominator)
    }

    fn fix_first_variable(&mut self, r: &FieldElement<F>) -> Result<(), Error> {
        for p in &mut self.polys {
            p.fix_first_variable_in_place(r)?;
        }
        Ok(())
    }

    fn program(&self) -> Option<&Program<F>> {
        self.program.as_ref()
    }

    /// The layer relation is four multiplications written out, not a program
    /// the host walks — so its rounds are worth taking back much earlier.
    fn host_cube(&self) -> usize {
        crate::HOST_CUBE_DIRECT
    }

    fn accept_folded(&mut self, polys: Vec<Mle<F>>) -> Result<(), Error> {
        if polys.len() != self.polys.len() {
            return Err(Error::VariableCountMismatch {
                expected: self.polys.len(),
                got: polys.len(),
            });
        }
        self.polys = polys;
        Ok(())
    }
}

/// One layer's transcript: the sumcheck plus the four values it reduces to.
#[derive(
    Clone,
    Debug,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
#[serde(bound = "")]
pub struct LayerProof<F: IsField> {
    pub sumcheck: SumcheckProof<F>,
    pub p_lo: FieldElement<F>,
    pub p_hi: FieldElement<F>,
    pub q_lo: FieldElement<F>,
    pub q_hi: FieldElement<F>,
}

/// A proof for the whole tree, output layer first.
#[derive(
    Clone,
    Debug,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
#[serde(bound = "")]
pub struct GkrProof<F: IsField> {
    pub layers: Vec<LayerProof<F>>,
}

/// What the verifier is left holding about the **input** layer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GkrClaim<F: IsField> {
    pub point: Vec<FieldElement<F>>,
    pub p: FieldElement<F>,
    pub q: FieldElement<F>,
}

/// What proving leaves the caller holding.
///
/// The claim is the same one [`verify`] arrives at: the prover needs it to
/// discharge the input layer, which is where the trace is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GkrOutput<F: IsField> {
    pub proof: GkrProof<F>,
    pub claim: GkrClaim<F>,
}

/// `(1 − c)·lo + c·hi` — the multilinear interpolation that turns the two
/// restricted claims back into one claim on the fuller layer.
fn combine_halves<F: IsField>(
    lo: &FieldElement<F>,
    hi: &FieldElement<F>,
    c: &FieldElement<F>,
) -> FieldElement<F> {
    lo + c * &(hi - lo)
}

/// Where a device layer's sumcheck stood when the device handed it back: the
/// layer's point, the claim the sumcheck started from, and the rounds the device
/// ran with the challenges they drew. What the lean tail needs beyond the
/// relation.
struct DeviceStart<'a, F: IsField> {
    point: &'a [FieldElement<F>],
    claim: FieldElement<F>,
    rounds: &'a [sumcheck::RoundProof<F>],
    challenges: &'a [FieldElement<F>],
}

/// Values at `0, 1, 2, 3` interpolated at a point: [`sumcheck::interpolate`] for
/// a cubic, with the four Lagrange denominators inverted once rather than on
/// every call.
struct Cubic<F: IsField> {
    /// `1 / Π_{j≠i} (i − j)` for `i = 0..4`: `−1/6, 1/2, −1/2, 1/6`.
    inverse: [FieldElement<F>; 4],
}

impl<F: IsField> Cubic<F> {
    fn new() -> Option<Self> {
        let six = FieldElement::<F>::from(6u64);
        let two = FieldElement::<F>::from(2u64);
        let mut inverse = [-six.clone(), two.clone(), -two, six];
        FieldElement::inplace_batch_inverse(&mut inverse).ok()?;
        Some(Self { inverse })
    }

    fn at(&self, values: [&FieldElement<F>; 4], x: &FieldElement<F>) -> FieldElement<F> {
        let d: [FieldElement<F>; 4] =
            std::array::from_fn(|j| x - FieldElement::<F>::from(j as u64));
        let low = &d[0] * &d[1];
        let high = &d[2] * &d[3];
        let others = [&d[1] * &high, &d[0] * &high, &low * &d[3], &low * &d[2]];
        values
            .iter()
            .zip(others.iter().zip(&self.inverse))
            .fold(FieldElement::zero(), |acc, (v, (o, inv))| {
                acc + *v * o * inv
            })
    }
}

/// ⛔ A FAULT: while armed, the lean tail adds one to its first round's `s(1)` —
/// what a lean round that summed one term wrong would send — so the checks after
/// it can be shown to see one. Never armed outside a test.
static LEAN_TAIL_FAULT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[doc(hidden)]
pub fn force_lean_tail_fault(on: bool) {
    LEAN_TAIL_FAULT.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// A layer's rounds on the host: lean when the device handed the layer back and
/// `LAMBDA_VM_ARGUE_LEAN_TAIL` is on, generic otherwise — and generic too when
/// the lean tail declines, which it does before absorbing anything.
fn finish_layer<F, T>(
    relation: &mut LayerRelation<F>,
    start: Option<&DeviceStart<'_, F>>,
    transcript: &mut T,
) -> Result<sumcheck::RoundGroup<F>, Error>
where
    F: IsField + 'static,
    T: IsTranscript<F>,
{
    if let Some(start) = start
        && let Some(outcome) = lean_tail(relation, start, transcript)
    {
        return outcome;
    }
    let left = relation.num_vars();
    sumcheck::prove_rounds(relation, left, transcript)
}

/// A device layer's host tail with its `eq` weight pulled out of the round
/// polynomial (`LAMBDA_VM_ARGUE_LEAN_TAIL`): the rounds
/// [`sumcheck::prove_rounds`] sends, value for value, from two sums over the
/// cube a round where it makes three, and no multiplication to extend a factor.
///
/// # Why the rounds are the same
///
/// The weight a device hands back is `eq(point, ·)` folded on the device's
/// challenges, so over the variables left it is a constant times `eq(ρ, ·)`,
/// `ρ` being the rest of the point. At round `k` it factors through the variable
/// being bound: `E(t, x) = ℓ(t)·w(x)`, with `ℓ(t) = eq(ρ_k, t) = (1 − ρ_k) +
/// t·(2ρ_k − 1)` and `w = E(0, ·) + E(1, ·)`. So
///
/// ```text
/// s(t) = Σ_x E(t, x)·g(t, x) = ℓ(t)·h(t),    h(t) = Σ_x w(x)·g(t, x),
/// ```
///
/// with `g = p_lo·q_hi + p_hi·q_lo + λ·q_lo·q_hi` of degree two in `t`, so `h` is
/// too. The tail sums `h(1)` and `h(2)`. `s(0) + s(1)` is the claim the round
/// carries, which gives `h(0) = (claim − s(1)) / ℓ(0)`; and a quadratic's third
/// difference is zero, which gives `h(3) = h(0) + 3·(h(2) − h(1))`. Each value is
/// the field element the generic round computes, so the transcript and the
/// proof's canonical bytes do not move.
///
/// The factors reach `t = 2` as `2·hi − lo`, by addition. The next round's `w` is
/// this one's two halves added, with the `ℓ(z)` the bound variable leaves kept as
/// one scalar. Only the four halves are folded, by the same fold the generic
/// rounds use.
///
/// # ⚠ Only for an `eq` weight
///
/// The factoring is what makes two sums enough, so only a layer a device handed
/// back takes this path; a level proved here from the start keeps the generic
/// rounds. `None`, before anything is absorbed, when the point, the device's
/// rounds and the relation do not line up, or when some `1 − ρ_k` has no inverse.
///
/// Raw limbs can differ from the generic round's: Goldilocks keeps a value in
/// `[0, 2^64)`, and a different sum reaches a different representative of the
/// same element (`thoughts/zf/gap2/fix2/I-GFS.md` §9.5). No host-only proof — none
/// of the pinned byte gates — reaches this path.
fn lean_tail<F, T>(
    relation: &mut LayerRelation<F>,
    start: &DeviceStart<'_, F>,
    transcript: &mut T,
) -> Option<Result<sumcheck::RoundGroup<F>, Error>>
where
    F: IsField + 'static,
    T: IsTranscript<F>,
{
    lean_tail_with(
        relation,
        start,
        transcript,
        crate::gpu::argue_xcheck(),
        LEAN_TAIL_FAULT.load(std::sync::atomic::Ordering::Relaxed),
    )
}

/// [`lean_tail`], with `LAMBDA_VM_ARGUE_XCHECK` and the fault as arguments, so a
/// test can set them for itself without touching the process.
///
/// Under `xcheck` every round is summed the generic way too, on a copy of the
/// relation, and the tail fails the proof where the two disagree.
fn lean_tail_with<F, T>(
    relation: &mut LayerRelation<F>,
    start: &DeviceStart<'_, F>,
    transcript: &mut T,
    xcheck: bool,
    fault: bool,
) -> Option<Result<sumcheck::RoundGroup<F>, Error>>
where
    F: IsField + 'static,
    T: IsTranscript<F>,
{
    let m = relation.num_vars();
    let done = start.challenges.len();
    if m == 0
        || relation.program.is_some()
        || start.rounds.len() != done
        || start.point.len() != done + m
        || start
            .rounds
            .iter()
            .any(|round| round.evaluations.len() != LAYER_DEGREE)
    {
        return None;
    }
    let rest = &start.point[done..];
    let one = FieldElement::<F>::one();
    // `1 / ℓ_k(0) = 1 / (1 − ρ_k)` for every round, in one inversion.
    let mut inverse_left: Vec<FieldElement<F>> = rest.iter().map(|r| &one - r).collect();
    FieldElement::inplace_batch_inverse(&mut inverse_left).ok()?;
    let cubic = Cubic::new()?;
    // The claim the tail carries in: what the verifier arrives at after the
    // device's rounds.
    let mut claim =
        start
            .rounds
            .iter()
            .zip(start.challenges)
            .fold(start.claim.clone(), |claim, (round, z)| {
                let [s1, s2, s3] = [
                    &round.evaluations[0],
                    &round.evaluations[1],
                    &round.evaluations[2],
                ];
                cubic.at([&(&claim - s1), s1, s2, s3], z)
            });
    let mut shadow = if xcheck {
        Some(LayerRelation::from_factors(relation.polys.clone(), relation.lambda.clone()).ok()?)
    } else {
        None
    };

    // Nothing can decline past here.
    let lambda = relation.lambda.clone();
    let mut polys = std::mem::take(&mut relation.polys);
    let weight = polys.remove(LayerRelation::<F>::EQ).into_evals();
    let half = weight.len() / 2;
    let mut w: Vec<FieldElement<F>> = (0..half).map(|j| &weight[j] + &weight[j + half]).collect();
    let mut kappa = one.clone();
    let three = FieldElement::<F>::from(3u64);
    let mut proofs = Vec::with_capacity(m);
    let mut challenges = Vec::with_capacity(m);

    for (k, rho) in rest.iter().enumerate() {
        let half = polys[0].len() / 2;
        let [p_lo, p_hi, q_lo, q_hi] = [
            polys[0].evals(),
            polys[1].evals(),
            polys[2].evals(),
            polys[3].evals(),
        ];
        let (mut h1, mut h2) = (FieldElement::<F>::zero(), FieldElement::<F>::zero());
        for (j, weight) in w.iter().enumerate() {
            let (a0, a1) = (&p_lo[j], &p_lo[j + half]);
            let (b0, b1) = (&p_hi[j], &p_hi[j + half]);
            let (c0, c1) = (&q_lo[j], &q_lo[j + half]);
            let (d0, d1) = (&q_hi[j], &q_hi[j + half]);
            let g1 = a1 * d1 + b1 * c1 + &lambda * (c1 * d1);
            let (a2, b2, c2, d2) = (a1 + a1 - a0, b1 + b1 - b0, c1 + c1 - c0, d1 + d1 - d0);
            let g2 = &a2 * &d2 + &b2 * &c2 + &lambda * (&c2 * &d2);
            h1 += weight * g1;
            h2 += weight * g2;
        }
        let h1 = &kappa * h1;
        let h2 = &kappa * h2;
        let l0 = &one - rho;
        let step = rho + rho - &one;
        let l2 = rho + &step;
        let l3 = &l2 + &step;
        let mut s1 = rho * &h1;
        if fault && k == 0 {
            s1 += one.clone();
        }
        let s0 = &claim - &s1;
        let h0 = &s0 * &inverse_left[k];
        let h3 = &h0 + (&h2 - &h1) * &three;
        let s2 = &l2 * &h2;
        let s3 = &l3 * &h3;
        let sent = vec![s1, s2, s3];

        if let Some(shadow) = shadow.as_mut() {
            let want = sumcheck::round_evaluations(&*shadow, LAYER_DEGREE, false);
            if want != sent {
                return Some(Err(Error::DeviceFailed { stage: "lean tail" }));
            }
        }
        for value in &sent {
            transcript.append_field_element(value);
        }
        let z = transcript.sample_field_element();

        claim = cubic.at([&s0, &sent[0], &sent[1], &sent[2]], &z);
        for poly in &mut polys {
            if let Err(error) = poly.fix_first_variable_in_place(&z) {
                return Some(Err(error));
            }
        }
        if let Some(shadow) = shadow.as_mut()
            && let Err(error) = shadow.fix_first_variable(&z)
        {
            return Some(Err(error));
        }
        kappa *= &l0 + &z * &step;
        if w.len() > 1 {
            let half = w.len() / 2;
            w = (0..half).map(|j| &w[j] + &w[j + half]).collect();
        }
        proofs.push(sumcheck::RoundProof { evaluations: sent });
        challenges.push(z);
    }

    // The weight's own value at the point, for a relation that reads it after.
    let weight = kappa * &w[0];
    polys.insert(LayerRelation::<F>::EQ, Mle::constant(weight));
    if let Some(shadow) = shadow.as_ref()
        && (1..polys.len()).any(|slot| shadow.polys()[slot].evals() != polys[slot].evals())
    {
        return Some(Err(Error::DeviceFailed { stage: "lean tail" }));
    }
    relation.polys = polys;
    crate::gpu::note_lean_tail(xcheck);
    Some(Ok((proofs, challenges)))
}

/// Proves the tree, from the output fraction down to the input layer.
pub fn prove<F, T>(tree: &FractionTree<F>, transcript: &mut T) -> Result<GkrOutput<F>, Error>
where
    F: IsField + 'static,
    T: IsTranscript<F>,
{
    let mut layers = Vec::with_capacity(tree.num_layers().saturating_sub(1));
    // The output layer has no variables, so the first claim sits at the empty
    // point and needs no challenge.
    let mut point: Vec<FieldElement<F>> = Vec::new();
    let (mut p_claim, mut q_claim) = tree.output();
    // The levels near the output, fetched once. Their rounds are over cubes a
    // core walks in microseconds, and a device pays a launch for each.
    let prefix = tree.host_prefix();

    for i in 0..tree.num_layers() - 1 {
        // A layer the device runs, whose host work the instrument splits
        // (`whir_split`, under `LAMBDA_VM_BASE_SPLIT` only).
        let on_card = prefix.get(i + 1).is_none() && tree.device().is_some();
        let clock = || if on_card { whir_split::tick() } else { None };

        let t = clock();
        let lambda: FieldElement<F> = transcript.sample_field_element();
        whir_split::add_tick(&whir_split::GKR_LAMBDA, t);

        // What the device ran of this layer, and the relation it left behind.
        // A tree the device holds proves its layer where it lies — the halves
        // are the sumcheck's factors in place — until the cube reaches the
        // crossover, and hands the factors over from there.
        let (mut rounds, mut z, mut relation) = match prefix.get(i + 1) {
            Some(here) => (
                Vec::new(),
                Vec::new(),
                LayerRelation::new(here, &point, lambda, false)?,
            ),
            None => match tree.device() {
                Some(device) => {
                    let t = clock();
                    let program = LayerRelation::program_for(&lambda)?;
                    whir_split::add_tick(&whir_split::GKR_PROGRAM, t);
                    let attempt = device.prove_layer(
                        i + 1,
                        &point,
                        &program,
                        LAYER_DEGREE,
                        crate::HOST_CUBE_DIRECT,
                        |sent| {
                            for value in sent {
                                transcript.append_field_element(value);
                            }
                            transcript.sample_field_element()
                        },
                    );
                    let Some(outcome) = attempt else {
                        return Err(Error::DeviceFailed {
                            stage: "layer sumcheck",
                        });
                    };
                    let (rounds, z, factors) = outcome?;
                    (rounds, z, LayerRelation::from_factors(factors, lambda)?)
                }
                None => (
                    Vec::new(),
                    Vec::new(),
                    LayerRelation::new(tree.layer(i + 1), &point, lambda, true)?,
                ),
            },
        };

        // The tail, however much of it is left: all of it for a level that came
        // here whole, none for one the device ran out. A device layer's can run
        // lean (`LAMBDA_VM_ARGUE_LEAN_TAIL`): its weight is `eq(point, ·)`.
        let left = relation.num_vars();
        let start = (on_card && crate::gpu::argue_lean_tail()).then(|| DeviceStart {
            point: &point,
            claim: &p_claim + &relation.lambda * &q_claim,
            rounds: &rounds,
            challenges: &z,
        });
        let t = clock();
        let (tail, tail_z) = if t.is_some() {
            // The same calls on the same transcript, with its share timed.
            let mut timed =
                whir_split::Timed::new(&mut *transcript, &whir_split::GKR_TAIL_TRANSCRIPT);
            finish_layer(&mut relation, start.as_ref(), &mut timed)?
        } else {
            finish_layer(&mut relation, start.as_ref(), transcript)?
        };
        whir_split::add_tick(&whir_split::GKR_TAIL, t);
        if on_card {
            whir_split::bump_by(&whir_split::GKR_TAIL_ROUNDS, left as u64);
        }
        rounds.extend(tail);
        z.extend(tail_z);
        let sumcheck = SumcheckProof { rounds };

        // The four restricted values the verifier needs to close the round are
        // what the sumcheck's own factors have become: binding every variable
        // to `z` is the evaluation at `z`. Evaluating the halves again would be
        // a second pass over the layer, and the layers are the biggest thing
        // the fraction tree holds.
        let t = clock();
        let bound = |slot: usize| -> Result<FieldElement<F>, Error> {
            relation.polys()[slot]
                .as_constant()
                .cloned()
                .ok_or(Error::NoVariablesLeft)
        };
        let p_lo = bound(LayerRelation::<F>::P_LO)?;
        let p_hi = bound(LayerRelation::<F>::P_HI)?;
        let q_lo = bound(LayerRelation::<F>::Q_LO)?;
        let q_hi = bound(LayerRelation::<F>::Q_HI)?;

        for v in [&p_lo, &p_hi, &q_lo, &q_hi] {
            transcript.append_field_element(v);
        }
        let c = transcript.sample_field_element();

        // Next layer's claim lives at (c, z).
        p_claim = combine_halves(&p_lo, &p_hi, &c);
        q_claim = combine_halves(&q_lo, &q_hi, &c);
        point = std::iter::once(c).chain(z).collect();

        layers.push(LayerProof {
            sumcheck,
            p_lo,
            p_hi,
            q_lo,
            q_hi,
        });
        whir_split::add_tick(&whir_split::GKR_CLOSE, t);
    }

    Ok(GkrOutput {
        proof: GkrProof { layers },
        claim: GkrClaim {
            point,
            p: p_claim,
            q: q_claim,
        },
    })
}

/// Verifies the tree against a claimed output fraction.
pub fn verify<F, T>(
    proof: &GkrProof<F>,
    output: (FieldElement<F>, FieldElement<F>),
    transcript: &mut T,
) -> Result<GkrClaim<F>, Error>
where
    F: IsField + 'static,
    T: IsTranscript<F>,
{
    let (mut p_claim, mut q_claim) = output;
    let mut point: Vec<FieldElement<F>> = Vec::new();

    for (i, layer) in proof.layers.iter().enumerate() {
        let lambda = transcript.sample_field_element();
        let claimed_sum = &p_claim + &lambda * &q_claim;

        let claim = sumcheck::verify(&layer.sumcheck, claimed_sum, point.len(), 3, transcript)?;

        // The sumcheck's residual must be the layer relation at that point.
        let eq_at = eq_eval(&point, &claim.point)?;
        let numerator = &layer.p_lo * &layer.q_hi + &layer.p_hi * &layer.q_lo;
        let denominator = &layer.q_lo * &layer.q_hi;
        let expected = eq_at * (numerator + &lambda * denominator);
        if expected != claim.expected_evaluation {
            return Err(Error::LayerRelationMismatch { layer: i });
        }

        for v in [&layer.p_lo, &layer.p_hi, &layer.q_lo, &layer.q_hi] {
            transcript.append_field_element(v);
        }
        let c = transcript.sample_field_element();

        p_claim = combine_halves(&layer.p_lo, &layer.p_hi, &c);
        q_claim = combine_halves(&layer.q_lo, &layer.q_hi, &c);
        point = std::iter::once(c).chain(claim.point).collect();
    }

    Ok(GkrClaim {
        point,
        p: p_claim,
        q: q_claim,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use math::field::goldilocks::GoldilocksField as F;

    type FE = FieldElement<F>;

    fn transcript() -> DefaultTranscript<F> {
        DefaultTranscript::<F>::new(b"gkr-test")
    }

    fn mle(vals: &[u64]) -> Mle<F> {
        Mle::new(vals.iter().map(|v| FE::from(*v)).collect()).unwrap()
    }

    fn layer(p: &[u64], q: &[u64]) -> FractionLayer<F> {
        FractionLayer::new(mle(p), mle(q)).unwrap()
    }

    /// Σ pᵢ/qᵢ computed directly, for comparison against the tree.
    fn direct_sum(p: &[u64], q: &[u64]) -> FE {
        p.iter().zip(q).fold(FE::zero(), |acc, (pi, qi)| {
            acc + FE::from(*pi) * FE::from(*qi).inv().unwrap()
        })
    }

    /// A LogUp-shaped input layer: `mult / (alpha - fingerprint)`, with the
    /// sends and receives arranged to cancel.
    fn balanced_logup_layer(num_vars: usize) -> FractionLayer<F> {
        let size = 1usize << num_vars;
        let alpha = FE::from(0x9E37_79B9u64);
        let mut p = Vec::with_capacity(size);
        let mut q = Vec::with_capacity(size);
        for i in 0..size {
            // Each fingerprint appears once as a send (+1) and once as a
            // receive (−1), so the whole bus balances.
            let fingerprint = FE::from((i as u64 / 2) * 7 + 5);
            let sign = if i % 2 == 0 { FE::one() } else { -FE::one() };
            p.push(sign);
            q.push(alpha - fingerprint);
        }
        FractionLayer::new(Mle::new(p).unwrap(), Mle::new(q).unwrap()).unwrap()
    }

    #[test]
    fn folding_adds_the_two_halves() {
        let l = layer(&[1, 2, 3, 4], &[5, 6, 7, 8]);
        let folded = l.fold().unwrap();
        assert_eq!(folded.num_vars(), 1);

        // Variable 0 is the most significant bit, so index i pairs with i + 2.
        let p = [1u64, 2, 3, 4];
        let q = [5u64, 6, 7, 8];
        for (i, (a, b)) in [(0usize, 2usize), (1, 3)].iter().enumerate() {
            let expected = FE::from(p[*a]) * FE::from(q[*a]).inv().unwrap()
                + FE::from(p[*b]) * FE::from(q[*b]).inv().unwrap();
            let got = folded.p.evals()[i] * folded.q.evals()[i].inv().unwrap();
            assert_eq!(expected, got, "pair ({a}, {b})");
        }
    }

    #[test]
    fn the_tree_output_is_the_sum_of_every_input_fraction() {
        let p = [3u64, 1, 4, 1, 5, 9, 2, 6];
        let q = [2u64, 7, 1, 8, 2, 8, 1, 8];
        let tree = FractionTree::build(layer(&p, &q)).unwrap();

        let (out_p, out_q) = tree.output();
        assert_eq!(out_p * out_q.inv().unwrap(), direct_sum(&p, &q));
    }

    #[test]
    fn layer_count_is_one_per_variable_plus_the_output() {
        let tree = FractionTree::build(balanced_logup_layer(4)).unwrap();
        assert_eq!(tree.num_layers(), 5);
        assert_eq!(tree.layer(0).num_vars(), 0);
        assert_eq!(tree.input_layer().num_vars(), 4);
    }

    #[test]
    fn a_balanced_bus_has_a_zero_numerator() {
        let tree = FractionTree::build(balanced_logup_layer(4)).unwrap();
        let (p, q) = tree.output();
        assert_eq!(p, FE::zero());
        assert_ne!(q, FE::zero(), "denominators must not vanish");
    }

    #[test]
    fn an_unbalanced_bus_does_not() {
        let mut input = balanced_logup_layer(4);
        // Drop one receive: the bus no longer cancels.
        let mut p = input.p.evals().to_vec();
        p[3] = FE::zero();
        input = FractionLayer::new(Mle::new(p).unwrap(), input.q).unwrap();

        let tree = FractionTree::build(input).unwrap();
        assert_ne!(tree.output().0, FE::zero());
    }

    #[test]
    fn prove_and_verify_round_trip() {
        let tree = FractionTree::build(balanced_logup_layer(5)).unwrap();
        let output = tree.output();

        let proof = prove(&tree, &mut transcript()).unwrap().proof;
        assert_eq!(proof.layers.len(), tree.num_layers() - 1);

        let claim = verify(&proof, output, &mut transcript()).unwrap();

        // The residual claim must be the input layer at the final point.
        let input = tree.input_layer();
        assert_eq!(claim.point.len(), input.num_vars());
        assert_eq!(input.p.evaluate(&claim.point).unwrap(), claim.p);
        assert_eq!(input.q.evaluate(&claim.point).unwrap(), claim.q);
    }

    #[test]
    fn round_trips_on_an_unbalanced_bus_too() {
        // GKR proves the sum is whatever it is; deciding that it is zero is the
        // caller's check on the output numerator, not part of this protocol.
        let mut p = balanced_logup_layer(4).p.evals().to_vec();
        p[3] = FE::from(9);
        let input = FractionLayer::new(Mle::new(p).unwrap(), balanced_logup_layer(4).q).unwrap();
        let tree = FractionTree::build(input).unwrap();

        let proof = prove(&tree, &mut transcript()).unwrap().proof;
        let claim = verify(&proof, tree.output(), &mut transcript()).unwrap();
        assert_eq!(
            tree.input_layer().p.evaluate(&claim.point).unwrap(),
            claim.p
        );
    }

    #[test]
    fn the_prover_arrives_at_the_claim_the_verifier_does() {
        // The prover has to discharge the input-layer claim against the trace,
        // so it needs the same claim the verifier ends up holding.
        let tree = FractionTree::build(balanced_logup_layer(3)).unwrap();
        let out = prove(&tree, &mut transcript()).unwrap();
        let claim = verify(&out.proof, tree.output(), &mut transcript()).unwrap();
        assert_eq!(out.claim, claim);

        // And it is the input layer's value at that point.
        let input = tree.input_layer();
        assert_eq!(claim.p, input.p.evaluate(&claim.point).unwrap());
        assert_eq!(claim.q, input.q.evaluate(&claim.point).unwrap());
    }

    #[test]
    fn a_wrong_output_claim_is_rejected() {
        let tree = FractionTree::build(balanced_logup_layer(4)).unwrap();
        let (p, q) = tree.output();
        let proof = prove(&tree, &mut transcript()).unwrap().proof;

        assert!(verify(&proof, (p + FE::one(), q), &mut transcript()).is_err());
    }

    #[test]
    fn a_tampered_half_value_is_rejected() {
        let tree = FractionTree::build(balanced_logup_layer(4)).unwrap();
        let output = tree.output();
        let mut proof = prove(&tree, &mut transcript()).unwrap().proof;

        proof.layers[1].q_lo += FE::one();
        let err = verify(&proof, output, &mut transcript()).unwrap_err();
        assert!(matches!(err, Error::LayerRelationMismatch { layer: 1 }));
    }

    #[test]
    fn a_tampered_sumcheck_round_is_rejected() {
        let tree = FractionTree::build(balanced_logup_layer(4)).unwrap();
        let output = tree.output();
        let mut proof = prove(&tree, &mut transcript()).unwrap().proof;

        proof.layers[2].sumcheck.rounds[0].evaluations[0] += FE::one();
        assert!(verify(&proof, output, &mut transcript()).is_err());
    }

    #[test]
    fn a_proof_replayed_under_another_transcript_is_rejected() {
        let tree = FractionTree::build(balanced_logup_layer(4)).unwrap();
        let output = tree.output();
        let proof = prove(&tree, &mut transcript()).unwrap().proof;

        verify(&proof, output, &mut transcript()).unwrap();
        let mut other = DefaultTranscript::<F>::new(b"another-statement");
        assert!(verify(&proof, output, &mut other).is_err());
    }

    #[test]
    fn a_layer_carries_on_from_its_own_folded_factors() {
        // What the crossover relies on: a relation rebuilt from the factors a
        // few rounds left behind is the same relation. On a device those
        // factors are read back off the layer; here the check is that picking
        // them up mid-sumcheck changes nothing the verifier sees.
        let tree = FractionTree::build(balanced_logup_layer(6)).unwrap();
        let lambda = FE::from(7);
        let layer = tree.layer(4);

        let whole = {
            let mut relation = LayerRelation::new(layer, &[FE::from(3)], lambda, true).unwrap();
            let vars = relation.num_vars();
            sumcheck::prove_rounds(&mut relation, vars, &mut transcript()).unwrap()
        };

        let mut relation = LayerRelation::new(layer, &[FE::from(3)], lambda, true).unwrap();
        let mut t = transcript();
        let (mut rounds, mut z) = sumcheck::prove_rounds(&mut relation, 1, &mut t).unwrap();
        let mut carried = LayerRelation::from_factors(relation.polys().to_vec(), lambda).unwrap();
        let left = carried.num_vars();
        let (tail, tail_z) = sumcheck::prove_rounds(&mut carried, left, &mut t).unwrap();
        rounds.extend(tail);
        z.extend(tail_z);

        assert_eq!(whole, (rounds, z));
    }

    /// splitmix64: the lean-tail cases' values, reproducible from a seed.
    fn next(seed: &mut u64) -> u64 {
        *seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A layer as a device hands it back, `done` rounds in: the relation the
    /// generic rounds and the lean tail both carry on from, the transcript
    /// there, and where the sumcheck stood.
    #[allow(clippy::type_complexity)]
    fn handed_back<E, T>(
        num_vars: usize,
        done: usize,
        seed: u64,
        element: impl Fn(&mut u64) -> FieldElement<E>,
        transcript: impl Fn() -> T,
    ) -> (
        LayerRelation<E>,
        T,
        Vec<FieldElement<E>>,
        FieldElement<E>,
        sumcheck::RoundGroup<E>,
    )
    where
        E: IsField + 'static,
        T: IsTranscript<E>,
    {
        let mut seed = seed;
        let size = 2usize << num_vars;
        let p: Vec<FieldElement<E>> = (0..size).map(|_| element(&mut seed)).collect();
        let q: Vec<FieldElement<E>> = (0..size).map(|_| element(&mut seed)).collect();
        let next_layer = FractionLayer::new(Mle::new(p).unwrap(), Mle::new(q).unwrap()).unwrap();
        let point: Vec<FieldElement<E>> = (0..num_vars).map(|_| element(&mut seed)).collect();
        let lambda = element(&mut seed);
        let mut relation = LayerRelation::new(&next_layer, &point, lambda.clone(), false).unwrap();
        // The claim the layer's sumcheck starts from, summed outright.
        let claim = (0..1usize << num_vars).fold(FieldElement::<E>::zero(), |acc, x| {
            let values: Vec<FieldElement<E>> = relation
                .polys()
                .iter()
                .map(|poly| poly.evals()[x].clone())
                .collect();
            acc + relation.combine(&values)
        });
        let mut transcript = transcript();
        let device = sumcheck::prove_rounds(&mut relation, done, &mut transcript).unwrap();
        let carried = LayerRelation::from_factors(relation.polys().to_vec(), lambda).unwrap();
        (carried, transcript, point, claim, device)
    }

    /// A group's round values and challenges, comparable field element by field
    /// element.
    #[allow(clippy::type_complexity)]
    fn sent<E: IsField>(
        group: &sumcheck::RoundGroup<E>,
    ) -> (Vec<Vec<FieldElement<E>>>, Vec<FieldElement<E>>) {
        (
            group
                .0
                .iter()
                .map(|round| round.evaluations.clone())
                .collect(),
            group.1.clone(),
        )
    }

    fn ext3(seed: &mut u64) -> FieldElement<Ext3> {
        FieldElement::<Ext3>::new([
            FE::from(next(seed)),
            FE::from(next(seed)),
            FE::from(next(seed)),
        ])
    }

    fn base(seed: &mut u64) -> FE {
        FE::from(next(seed))
    }

    fn ext3_transcript() -> DefaultTranscript<Ext3> {
        DefaultTranscript::<Ext3>::new(b"lean-tail")
    }

    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;

    /// ★ The lean tail sends what the generic rounds send, round for round, at
    /// every size up to the device's crossover cube and past it, from every
    /// point a device could hand a layer back at: the same values, the same
    /// challenges, the same transcript after them, and the same factors left
    /// bound for the layer's close.
    fn the_lean_tail_is_the_generic_tail<E, T>(
        element: impl Fn(&mut u64) -> FieldElement<E> + Copy,
        transcript: impl Fn() -> T + Copy,
    ) where
        E: IsField + 'static,
        T: IsTranscript<E>,
    {
        for num_vars in 1..=11 {
            for done in 0..num_vars.min(4) {
                let seed = (num_vars * 16 + done) as u64;
                let (mut generic, mut generic_t, point, claim, device) =
                    handed_back(num_vars, done, seed, element, transcript);
                let (mut lean, mut lean_t, ..) =
                    handed_back(num_vars, done, seed, element, transcript);
                let left = generic.num_vars();
                let want = sumcheck::prove_rounds(&mut generic, left, &mut generic_t).unwrap();
                let start = DeviceStart {
                    point: &point,
                    claim,
                    rounds: &device.0,
                    challenges: &device.1,
                };
                let got = lean_tail_with(&mut lean, &start, &mut lean_t, false, false)
                    .unwrap_or_else(|| panic!("2^{num_vars}, {done} in: the lean tail declined"))
                    .unwrap();
                assert_eq!(
                    sent(&got),
                    sent(&want),
                    "2^{num_vars}, {done} in: the rounds differ"
                );
                assert_eq!(
                    lean_t.state(),
                    generic_t.state(),
                    "2^{num_vars}, {done} in: the transcripts parted"
                );
                for slot in 0..5 {
                    assert_eq!(
                        lean.polys()[slot].evals(),
                        generic.polys()[slot].evals(),
                        "2^{num_vars}, {done} in: factor {slot} is not bound to the same value"
                    );
                }
            }
        }
    }

    #[test]
    fn the_lean_tail_sends_what_the_generic_rounds_send() {
        the_lean_tail_is_the_generic_tail(base, transcript);
        the_lean_tail_is_the_generic_tail(ext3, ext3_transcript);
    }

    /// Under the cross-check, the lean tail is summed the generic way too and
    /// agrees; with the fault armed, the cross-check refuses to go on — and
    /// without it the faulted tail's rounds are not the generic ones, so the
    /// fault is what a verifier would see.
    #[test]
    fn a_wrong_lean_round_is_seen_by_the_cross_check() {
        let (mut generic, mut generic_t, point, claim, device) =
            handed_back(9, 3, 7, ext3, ext3_transcript);
        let left = generic.num_vars();
        let want = sumcheck::prove_rounds(&mut generic, left, &mut generic_t).unwrap();
        let start = DeviceStart {
            point: &point,
            claim,
            rounds: &device.0,
            challenges: &device.1,
        };
        let run = |xcheck: bool, fault: bool| {
            let (mut lean, mut t, ..) = handed_back(9, 3, 7, ext3, ext3_transcript);
            lean_tail_with(&mut lean, &start, &mut t, xcheck, fault)
                .expect("the lean tail runs")
                .map(|group| sent(&group))
        };
        assert_eq!(run(true, false), Ok(sent(&want)), "the cross-check agrees");
        assert_eq!(
            run(true, true),
            Err(Error::DeviceFailed { stage: "lean tail" }),
            "the cross-check refuses a wrong lean round"
        );
        assert_ne!(
            run(false, true),
            Ok(sent(&want)),
            "the fault changes what is sent"
        );
    }

    /// Where `1 − ρ_k` has no inverse the lean tail declines, before anything
    /// is absorbed or folded, and the generic rounds carry on from the same
    /// state.
    #[test]
    fn the_lean_tail_declines_before_the_transcript_moves() {
        let (mut lean, mut t, mut point, claim, device) =
            handed_back(6, 2, 11, ext3, ext3_transcript);
        point[4] = FieldElement::one();
        let before = (t.state(), lean.polys().to_vec());
        let start = DeviceStart {
            point: &point,
            claim,
            rounds: &device.0,
            challenges: &device.1,
        };
        assert!(lean_tail_with(&mut lean, &start, &mut t, false, false).is_none());
        assert_eq!(t.state(), before.0, "nothing was absorbed");
        for (slot, poly) in lean.polys().iter().enumerate() {
            assert_eq!(poly.evals(), before.1[slot].evals(), "factor {slot} moved");
        }
        // A point of the wrong length is declined the same way.
        let short = DeviceStart {
            point: &point[1..],
            ..start
        };
        assert!(lean_tail_with(&mut lean, &short, &mut t, false, false).is_none());
    }

    /// The cubic the claim is carried with is the verifier's interpolation.
    #[test]
    fn the_cubic_is_the_verifiers_interpolation() {
        let cubic = Cubic::<Ext3>::new().unwrap();
        let mut seed = 3;
        for _ in 0..32 {
            let values: [FieldElement<Ext3>; 4] = std::array::from_fn(|_| ext3(&mut seed));
            for x in [
                ext3(&mut seed),
                FieldElement::from(2u64),
                FieldElement::zero(),
            ] {
                assert_eq!(
                    cubic.at([&values[0], &values[1], &values[2], &values[3]], &x),
                    sumcheck::interpolate(&values, &x)
                );
            }
        }
    }

    #[test]
    fn a_single_fraction_needs_no_layers() {
        let tree = FractionTree::build(layer(&[7], &[3])).unwrap();
        assert_eq!(tree.num_layers(), 1);

        let proof = prove(&tree, &mut transcript()).unwrap().proof;
        assert!(proof.layers.is_empty());

        let claim = verify(&proof, tree.output(), &mut transcript()).unwrap();
        assert!(claim.point.is_empty());
        assert_eq!(claim.p, FE::from(7));
        assert_eq!(claim.q, FE::from(3));
    }
}
