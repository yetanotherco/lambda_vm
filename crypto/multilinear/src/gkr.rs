//! LogUp as a tree of fractions: `p₁/q₁ + p₂/q₂ = (p₁q₂ + p₂q₁)/(q₁q₂)`, added
//! pairwise up a binary tree with GKR proving each layer against the one below.
//! Only the input layer is ever committed.
//!
//! Proves the sum is whatever the output claims. Checking that the output
//! numerator is zero — the bus balance — is the caller's.

use crypto::fiat_shamir::is_transcript::IsTranscript;
use math::field::{element::FieldElement, traits::IsField};

use crate::{
    Error,
    eq::{eq_eval, eq_mle},
    mle::Mle,
    poly::SumcheckPolynomial,
    sumcheck::{self, SumcheckProof},
};

/// One level of the tree: numerators and denominators over the same cube.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FractionLayer<F: IsField> {
    pub p: Mle<F>,
    pub q: Mle<F>,
}

impl<F: IsField> FractionLayer<F> {
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
    pub fn fold(&self) -> Result<Self, Error> {
        if self.num_vars() == 0 {
            return Err(Error::NoVariablesLeft);
        }
        let half = self.p.len() / 2;
        let (p_lo, p_hi) = self.p.evals().split_at(half);
        let (q_lo, q_hi) = self.q.evals().split_at(half);

        let mut next_p = Vec::with_capacity(half);
        let mut next_q = Vec::with_capacity(half);
        for i in 0..half {
            next_p.push(&p_lo[i] * &q_hi[i] + &p_hi[i] * &q_lo[i]);
            next_q.push(&q_lo[i] * &q_hi[i]);
        }
        Self::new(Mle::new(next_p)?, Mle::new(next_q)?)
    }
}

/// The whole tree, from the input layer down to the single output fraction.
///
/// `layers[0]` is the output (zero variables); the last entry is the input.
#[derive(Clone, Debug)]
pub struct FractionTree<F: IsField> {
    layers: Vec<FractionLayer<F>>,
}

impl<F: IsField> FractionTree<F> {
    /// Builds every layer by repeated folding.
    pub fn build(input: FractionLayer<F>) -> Result<Self, Error> {
        let mut layers = vec![input];
        while layers.last().expect("non-empty").num_vars() > 0 {
            let next = layers.last().expect("non-empty").fold()?;
            layers.push(next);
        }
        layers.reverse();
        Ok(Self { layers })
    }

    /// The output fraction `(p, q)`. The bus balances when `p` is zero.
    pub fn output(&self) -> (FieldElement<F>, FieldElement<F>) {
        let top = &self.layers[0];
        (top.p.evals()[0].clone(), top.q.evals()[0].clone())
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn layer(&self, i: usize) -> &FractionLayer<F> {
        &self.layers[i]
    }

    pub fn input_layer(&self) -> &FractionLayer<F> {
        self.layers.last().expect("non-empty")
    }
}

/// The layer relation: `Σ_x eq(r,x)·[p_lo·q_hi + p_hi·q_lo + λ·q_lo·q_hi]`,
/// which equals `p_out(r) + λ·q_out(r)` when the layer really is the fold.
struct LayerRelation<F: IsField> {
    /// `[eq, p_lo, p_hi, q_lo, q_hi]`.
    polys: Vec<Mle<F>>,
    lambda: FieldElement<F>,
}

impl<F: IsField> LayerRelation<F> {
    const EQ: usize = 0;
    const P_LO: usize = 1;
    const P_HI: usize = 2;
    const Q_LO: usize = 3;
    const Q_HI: usize = 4;

    fn new(
        next: &FractionLayer<F>,
        r: &[FieldElement<F>],
        lambda: FieldElement<F>,
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
            lambda,
        })
    }
}

impl<F: IsField> SumcheckPolynomial<F> for LayerRelation<F> {
    fn num_vars(&self) -> usize {
        self.polys[Self::EQ].num_vars()
    }

    fn degree(&self) -> usize {
        // eq times a product of two layer values.
        3
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
}

/// One layer's transcript: the sumcheck plus the four values it reduces to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerProof<F: IsField> {
    pub sumcheck: SumcheckProof<F>,
    pub p_lo: FieldElement<F>,
    pub p_hi: FieldElement<F>,
    pub q_lo: FieldElement<F>,
    pub q_hi: FieldElement<F>,
}

/// A proof for the whole tree, output layer first.
#[derive(Clone, Debug, PartialEq, Eq)]
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

/// `(1 − c)·lo + c·hi` — the multilinear interpolation that turns the two
/// restricted claims back into one claim on the fuller layer.
fn combine_halves<F: IsField>(
    lo: &FieldElement<F>,
    hi: &FieldElement<F>,
    c: &FieldElement<F>,
) -> FieldElement<F> {
    lo + c * &(hi - lo)
}

/// Proves the tree, from the output fraction down to the input layer.
pub fn prove<F, T>(tree: &FractionTree<F>, transcript: &mut T) -> Result<GkrProof<F>, Error>
where
    F: IsField,
    T: IsTranscript<F>,
{
    let mut layers = Vec::with_capacity(tree.num_layers().saturating_sub(1));
    // The output layer has no variables, so the first claim sits at the empty
    // point and needs no challenge.
    let mut point: Vec<FieldElement<F>> = Vec::new();

    for i in 0..tree.num_layers() - 1 {
        let next = tree.layer(i + 1);
        let lambda = transcript.sample_field_element();

        let relation = LayerRelation::new(next, &point, lambda)?;
        let half_vars = relation.num_vars();
        let (sumcheck, z) = sumcheck::prove(relation, transcript)?;

        // The four restricted values the verifier needs to close the round.
        let half = next.p.len() / 2;
        let p_lo = Mle::new(next.p.evals()[..half].to_vec())?.evaluate(&z)?;
        let p_hi = Mle::new(next.p.evals()[half..].to_vec())?.evaluate(&z)?;
        let q_lo = Mle::new(next.q.evals()[..half].to_vec())?.evaluate(&z)?;
        let q_hi = Mle::new(next.q.evals()[half..].to_vec())?.evaluate(&z)?;
        debug_assert_eq!(z.len(), half_vars);

        for v in [&p_lo, &p_hi, &q_lo, &q_hi] {
            transcript.append_field_element(v);
        }
        let c = transcript.sample_field_element();

        layers.push(LayerProof {
            sumcheck,
            p_lo,
            p_hi,
            q_lo,
            q_hi,
        });

        // Next layer's claim lives at (c, z).
        point = std::iter::once(c).chain(z).collect();
    }

    Ok(GkrProof { layers })
}

/// Verifies the tree against a claimed output fraction.
pub fn verify<F, T>(
    proof: &GkrProof<F>,
    output: (FieldElement<F>, FieldElement<F>),
    transcript: &mut T,
) -> Result<GkrClaim<F>, Error>
where
    F: IsField,
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

        let proof = prove(&tree, &mut transcript()).unwrap();
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

        let proof = prove(&tree, &mut transcript()).unwrap();
        let claim = verify(&proof, tree.output(), &mut transcript()).unwrap();
        assert_eq!(
            tree.input_layer().p.evaluate(&claim.point).unwrap(),
            claim.p
        );
    }

    #[test]
    fn a_wrong_output_claim_is_rejected() {
        let tree = FractionTree::build(balanced_logup_layer(4)).unwrap();
        let (p, q) = tree.output();
        let proof = prove(&tree, &mut transcript()).unwrap();

        assert!(verify(&proof, (p + FE::one(), q), &mut transcript()).is_err());
    }

    #[test]
    fn a_tampered_half_value_is_rejected() {
        let tree = FractionTree::build(balanced_logup_layer(4)).unwrap();
        let output = tree.output();
        let mut proof = prove(&tree, &mut transcript()).unwrap();

        proof.layers[1].q_lo += FE::one();
        let err = verify(&proof, output, &mut transcript()).unwrap_err();
        assert!(matches!(err, Error::LayerRelationMismatch { layer: 1 }));
    }

    #[test]
    fn a_tampered_sumcheck_round_is_rejected() {
        let tree = FractionTree::build(balanced_logup_layer(4)).unwrap();
        let output = tree.output();
        let mut proof = prove(&tree, &mut transcript()).unwrap();

        proof.layers[2].sumcheck.rounds[0].evaluations[0] += FE::one();
        assert!(verify(&proof, output, &mut transcript()).is_err());
    }

    #[test]
    fn a_proof_replayed_under_another_transcript_is_rejected() {
        let tree = FractionTree::build(balanced_logup_layer(4)).unwrap();
        let output = tree.output();
        let proof = prove(&tree, &mut transcript()).unwrap();

        verify(&proof, output, &mut transcript()).unwrap();
        let mut other = DefaultTranscript::<F>::new(b"another-statement");
        assert!(verify(&proof, output, &mut other).is_err());
    }

    #[test]
    fn a_single_fraction_needs_no_layers() {
        let tree = FractionTree::build(layer(&[7], &[3])).unwrap();
        assert_eq!(tree.num_layers(), 1);

        let proof = prove(&tree, &mut transcript()).unwrap();
        assert!(proof.layers.is_empty());

        let claim = verify(&proof, tree.output(), &mut transcript()).unwrap();
        assert!(claim.point.is_empty());
        assert_eq!(claim.p, FE::from(7));
        assert_eq!(claim.q, FE::from(3));
    }
}
