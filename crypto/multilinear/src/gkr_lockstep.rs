//! LogUp-GKR over several fraction trees at once, in **lockstep** (D-BATCH §2.2,
//! phase G): one sumcheck per step for every tree still descending.
//!
//! The trees stay separate and keep their own heights. What is shared is the
//! ladder: they are aligned at the OUTPUT, so step `i` proves layer `i + 1` of
//! every tree with more than `i` input variables, all against the same point.
//! Every challenge is shared, so the point every active tree sits at is the
//! same one, one `eq` table serves them all, and one sumcheck batches them with
//! a fresh `μ`:
//!
//! ```text
//! claim_i = Σ_a μ^{2a}·p̄_a + μ^{2a+1}·q̄_a
//!         = Σ_x eq(P, x) · Σ_a [ μ^{2a}·(p_lo q_hi + p_hi q_lo) + μ^{2a+1}·q_lo q_hi ]_a(x)
//! ```
//!
//! With one tree this is [`gkr::prove`](crate::gkr::prove) step for step, `μ`
//! in the place of its `λ`. A tree leaves the ladder at the step that reaches
//! its input layer, with its input claim at the point the ladder is at then —
//! which is what the caller settles against its trace.
//!
//! Soundness is SWIRL Thm 4.1.1 per step (every polynomial has `i` variables)
//! plus the GKR line through `lo`/`hi` at `c`: a false claim on any active tree
//! dooms the step, and the batching costs `(2|A_i| − 1)/|F|`.
//!
//! The bus outputs are NOT absorbed here: the caller absorbs them where the
//! format puts them (per bin, before its ladder).

use crypto::fiat_shamir::is_transcript::IsTranscript;
use math::field::{element::FieldElement, traits::IsField};

use crate::{
    Error, challenge_powers,
    eq::{eq_eval, eq_mle},
    gkr::{FractionLayer, GkrClaim},
    mle::Mle,
    poly::SumcheckPolynomial,
    sumcheck::{self, SumcheckProof},
};

/// One tree's four restricted values at the end of a step.
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
pub struct TreeHalves<F: IsField> {
    pub p_lo: FieldElement<F>,
    pub p_hi: FieldElement<F>,
    pub q_lo: FieldElement<F>,
    pub q_hi: FieldElement<F>,
}

/// One step of the ladder: the shared sumcheck, and the halves of every tree
/// active in it, in the caller's tree order.
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
pub struct LockstepLayer<F: IsField> {
    pub sumcheck: SumcheckProof<F>,
    pub halves: Vec<TreeHalves<F>>,
}

/// The whole ladder, output end first: `max_T k_T` steps.
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
pub struct LockstepProof<F: IsField> {
    pub layers: Vec<LockstepLayer<F>>,
}

/// Which trees step `i` proves: those with more than `i` input variables, in
/// the caller's order. Both sides call this.
pub fn active(input_vars: &[usize], step: usize) -> Vec<usize> {
    (0..input_vars.len())
        .filter(|&t| input_vars[t] > step)
        .collect()
}

/// `Σ_a w_{2a}·(p_lo q_hi + p_hi q_lo) + w_{2a+1}·q_lo q_hi` over the trees'
/// values, laid out four a tree.
fn relation<F: IsField>(
    weights: &[FieldElement<F>],
    halves: &[FieldElement<F>],
) -> FieldElement<F> {
    halves
        .chunks(4)
        .zip(weights.chunks(2))
        .fold(FieldElement::zero(), |acc, (v, w)| {
            let numerator = &v[0] * &v[3] + &v[1] * &v[2];
            let denominator = &v[2] * &v[3];
            acc + &w[0] * numerator + &w[1] * denominator
        })
}

/// The step's polynomial: `eq(P, ·)` and four halves per active tree.
struct LockstepRelation<F: IsField> {
    /// `[eq, (p_lo, p_hi, q_lo, q_hi) per tree]`.
    polys: Vec<Mle<F>>,
    /// `μ` powers, two per tree.
    weights: Vec<FieldElement<F>>,
}

impl<F: IsField + 'static> SumcheckPolynomial<F> for LockstepRelation<F> {
    fn num_vars(&self) -> usize {
        self.polys[0].num_vars()
    }

    fn degree(&self) -> usize {
        3
    }

    fn polys(&self) -> &[Mle<F>] {
        &self.polys
    }

    fn combine(&self, v: &[FieldElement<F>]) -> FieldElement<F> {
        &v[0] * relation(&self.weights, &v[1..])
    }

    fn fix_first_variable(&mut self, r: &FieldElement<F>) -> Result<(), Error> {
        for p in &mut self.polys {
            p.fix_first_variable_in_place(r)?;
        }
        Ok(())
    }
}

/// `(1 − c)·lo + c·hi`.
fn line<F: IsField>(
    lo: &FieldElement<F>,
    hi: &FieldElement<F>,
    c: &FieldElement<F>,
) -> FieldElement<F> {
    lo + c * &(hi - lo)
}

/// Proves every tree's layers in lockstep, on the host.
///
/// `trees[t]` is tree `t`'s layers, output first (`trees[t][0]` has no
/// variables, the last is the input layer). Returns the ladder and each tree's
/// input-layer claim, in tree order. A tree with no variables at all is its
/// own input claim at the empty point and takes no step.
pub fn prove<F, T>(
    trees: &[&[FractionLayer<F>]],
    transcript: &mut T,
) -> Result<(LockstepProof<F>, Vec<GkrClaim<F>>), Error>
where
    F: IsField + 'static,
    T: IsTranscript<F>,
    FieldElement<F>: Send + Sync,
{
    let mut input_vars = Vec::with_capacity(trees.len());
    for tree in trees {
        let top = tree.first().ok_or(Error::EmptyPolynomial)?;
        if top.num_vars() != 0 {
            return Err(Error::VariableCountMismatch {
                expected: 0,
                got: top.num_vars(),
            });
        }
        for (i, layer) in tree.iter().enumerate() {
            if layer.num_vars() != i {
                return Err(Error::VariableCountMismatch {
                    expected: i,
                    got: layer.num_vars(),
                });
            }
        }
        input_vars.push(tree.len() - 1);
    }
    let steps = input_vars.iter().copied().max().unwrap_or(0);

    let mut claims: Vec<(FieldElement<F>, FieldElement<F>)> = trees
        .iter()
        .map(|tree| (tree[0].p.evals()[0].clone(), tree[0].q.evals()[0].clone()))
        .collect();
    let mut inputs: Vec<Option<GkrClaim<F>>> = (0..trees.len())
        .map(|t| {
            (input_vars[t] == 0).then(|| GkrClaim {
                point: Vec::new(),
                p: claims[t].0.clone(),
                q: claims[t].1.clone(),
            })
        })
        .collect();
    let mut point: Vec<FieldElement<F>> = Vec::new();
    let mut layers = Vec::with_capacity(steps);

    for i in 0..steps {
        let active = active(&input_vars, i);
        let mu: FieldElement<F> = transcript.sample_field_element();
        let weights = challenge_powers(&mu, 2 * active.len());

        let mut polys = Vec::with_capacity(1 + 4 * active.len());
        polys.push(eq_mle(&point)?);
        for &t in &active {
            let next = &trees[t][i + 1];
            let half = next.p.len() / 2;
            for m in [&next.p, &next.q] {
                polys.push(Mle::new(m.evals()[..half].to_vec())?);
                polys.push(Mle::new(m.evals()[half..].to_vec())?);
            }
        }
        // Reordered to `p_lo, p_hi, q_lo, q_hi`, which is what `relation` reads:
        // the loop above pushed exactly that.
        let mut step = LockstepRelation { polys, weights };
        let (rounds, z) = sumcheck::prove_rounds(&mut step, i, transcript)?;

        let bound = |slot: usize| -> Result<FieldElement<F>, Error> {
            step.polys[slot]
                .as_constant()
                .cloned()
                .ok_or(Error::NoVariablesLeft)
        };
        let mut halves = Vec::with_capacity(active.len());
        for a in 0..active.len() {
            let h = TreeHalves {
                p_lo: bound(1 + 4 * a)?,
                p_hi: bound(2 + 4 * a)?,
                q_lo: bound(3 + 4 * a)?,
                q_hi: bound(4 + 4 * a)?,
            };
            for v in [&h.p_lo, &h.p_hi, &h.q_lo, &h.q_hi] {
                transcript.append_field_element(v);
            }
            halves.push(h);
        }
        let c: FieldElement<F> = transcript.sample_field_element();
        point = std::iter::once(c.clone()).chain(z).collect();
        for (&t, h) in active.iter().zip(&halves) {
            claims[t] = (line(&h.p_lo, &h.p_hi, &c), line(&h.q_lo, &h.q_hi, &c));
            if input_vars[t] == i + 1 {
                inputs[t] = Some(GkrClaim {
                    point: point.clone(),
                    p: claims[t].0.clone(),
                    q: claims[t].1.clone(),
                });
            }
        }
        layers.push(LockstepLayer {
            sumcheck: SumcheckProof { rounds },
            halves,
        });
    }

    let inputs = inputs
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or(Error::NoVariablesLeft)?;
    Ok((LockstepProof { layers }, inputs))
}

/// Verifies a ladder against the trees' claimed outputs and their input
/// heights (`input_vars[t]` = `k_t`, from the statement, never the proof).
/// Returns each tree's input-layer claim, in tree order.
pub fn verify<F, T>(
    proof: &LockstepProof<F>,
    outputs: &[(FieldElement<F>, FieldElement<F>)],
    input_vars: &[usize],
    transcript: &mut T,
) -> Result<Vec<GkrClaim<F>>, Error>
where
    F: IsField + 'static,
    T: IsTranscript<F>,
{
    verify_with(proof, outputs, input_vars, transcript, true)
}

/// [`verify`], with the step residual's check switchable — `false` only in
/// the mutation test that shows the check is load-bearing.
#[doc(hidden)]
pub fn verify_with<F, T>(
    proof: &LockstepProof<F>,
    outputs: &[(FieldElement<F>, FieldElement<F>)],
    input_vars: &[usize],
    transcript: &mut T,
    check_residual: bool,
) -> Result<Vec<GkrClaim<F>>, Error>
where
    F: IsField + 'static,
    T: IsTranscript<F>,
{
    if outputs.len() != input_vars.len() {
        return Err(Error::ArgueShapeMismatch {
            part: "tree outputs",
        });
    }
    let steps = input_vars.iter().copied().max().unwrap_or(0);
    if proof.layers.len() != steps {
        return Err(Error::ArgueShapeMismatch { part: "ladder" });
    }

    let mut claims: Vec<(FieldElement<F>, FieldElement<F>)> = outputs.to_vec();
    let mut inputs: Vec<Option<GkrClaim<F>>> = (0..outputs.len())
        .map(|t| {
            (input_vars[t] == 0).then(|| GkrClaim {
                point: Vec::new(),
                p: claims[t].0.clone(),
                q: claims[t].1.clone(),
            })
        })
        .collect();
    let mut point: Vec<FieldElement<F>> = Vec::new();

    for (i, layer) in proof.layers.iter().enumerate() {
        let active = active(input_vars, i);
        if layer.halves.len() != active.len() {
            return Err(Error::ArgueShapeMismatch {
                part: "ladder step",
            });
        }
        let mu: FieldElement<F> = transcript.sample_field_element();
        let weights = challenge_powers(&mu, 2 * active.len());
        let claimed = active
            .iter()
            .zip(weights.chunks(2))
            .fold(FieldElement::<F>::zero(), |acc, (&t, w)| {
                acc + &w[0] * &claims[t].0 + &w[1] * &claims[t].1
            });

        let claim = sumcheck::verify(&layer.sumcheck, claimed, point.len(), 3, transcript)?;

        let values: Vec<FieldElement<F>> = layer
            .halves
            .iter()
            .flat_map(|h| {
                [
                    h.p_lo.clone(),
                    h.p_hi.clone(),
                    h.q_lo.clone(),
                    h.q_hi.clone(),
                ]
            })
            .collect();
        let expected = eq_eval(&point, &claim.point)? * relation(&weights, &values);
        if check_residual && expected != claim.expected_evaluation {
            return Err(Error::LayerRelationMismatch { layer: i });
        }

        for v in &values {
            transcript.append_field_element(v);
        }
        let c: FieldElement<F> = transcript.sample_field_element();
        point = std::iter::once(c.clone()).chain(claim.point).collect();
        for (&t, h) in active.iter().zip(&layer.halves) {
            claims[t] = (line(&h.p_lo, &h.p_hi, &c), line(&h.q_lo, &h.q_hi, &c));
            if input_vars[t] == i + 1 {
                inputs[t] = Some(GkrClaim {
                    point: point.clone(),
                    p: claims[t].0.clone(),
                    q: claims[t].1.clone(),
                });
            }
        }
    }

    inputs
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or(Error::ArgueShapeMismatch { part: "ladder" })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gkr::{self, FractionTree};
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;
    use math::field::goldilocks::GoldilocksField as F;

    type FE = FieldElement<F>;

    fn transcript() -> DefaultTranscript<F> {
        DefaultTranscript::<F>::new(b"gkr-lockstep-test")
    }

    /// A random fraction layer of `2^num_vars` cells, reproducible from `seed`.
    fn random_layer(num_vars: usize, seed: u64) -> FractionLayer<F> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            FE::from(s)
        };
        let size = 1usize << num_vars;
        let p: Vec<FE> = (0..size).map(|_| next()).collect();
        let q: Vec<FE> = (0..size).map(|_| next()).collect();
        FractionLayer::new(Mle::new(p).unwrap(), Mle::new(q).unwrap()).unwrap()
    }

    /// Every layer, output first, by host folding.
    fn layers(input: FractionLayer<F>) -> Vec<FractionLayer<F>> {
        let mut layers = vec![input];
        while layers.last().unwrap().num_vars() > 0 {
            let next = layers.last().unwrap().fold().unwrap();
            layers.push(next);
        }
        layers.reverse();
        layers
    }

    fn round_trip(heights: &[usize]) -> (Vec<Vec<FractionLayer<F>>>, LockstepProof<F>) {
        let trees: Vec<Vec<FractionLayer<F>>> = heights
            .iter()
            .enumerate()
            .map(|(t, &k)| layers(random_layer(k, 17 + t as u64)))
            .collect();
        let refs: Vec<&[FractionLayer<F>]> = trees.iter().map(|t| t.as_slice()).collect();
        let mut prover = transcript();
        let (proof, claims) = prove(&refs, &mut prover).unwrap();

        let outputs: Vec<(FE, FE)> = trees
            .iter()
            .map(|t| (t[0].p.evals()[0], t[0].q.evals()[0]))
            .collect();
        let mut verifier = transcript();
        let got = verify(&proof, &outputs, heights, &mut verifier).unwrap();
        assert_eq!(
            got, claims,
            "the prover and the verifier reach the same claims"
        );
        assert_eq!(prover.state(), verifier.state(), "and the same transcript");
        // Each claim is its tree's input layer at the claim point.
        for (tree, claim) in trees.iter().zip(&claims) {
            let input = tree.last().unwrap();
            assert_eq!(claim.point.len(), input.num_vars());
            assert_eq!(input.p.evaluate(&claim.point).unwrap(), claim.p);
            assert_eq!(input.q.evaluate(&claim.point).unwrap(), claim.q);
        }
        (trees, proof)
    }

    /// Trees of different heights, one of them without variables, share one
    /// ladder; each leaves it with its own input claim.
    #[test]
    fn trees_of_different_heights_round_trip_in_lockstep() {
        round_trip(&[5, 2, 0, 5, 3]);
        round_trip(&[1]);
        round_trip(&[0]);
        round_trip(&[4, 4]);
    }

    /// With one tree the ladder is today's GKR, message for message: `μ` is
    /// drawn where `λ` was and plays its role.
    #[test]
    fn one_tree_is_todays_gkr() {
        for k in 0..6 {
            let input = random_layer(k, 99 + k as u64);
            let tree = FractionTree::build(input.clone()).unwrap();
            let today = gkr::prove(&tree, &mut transcript()).unwrap();
            let layers = layers(input);
            let (proof, claims) = prove(&[layers.as_slice()], &mut transcript()).unwrap();
            assert_eq!(claims, vec![today.claim.clone()], "k = {k}");
            assert_eq!(proof.layers.len(), today.proof.layers.len());
            for (ours, theirs) in proof.layers.iter().zip(&today.proof.layers) {
                assert_eq!(ours.sumcheck, theirs.sumcheck, "k = {k}");
                let h = &ours.halves[0];
                assert_eq!(
                    [&h.p_lo, &h.p_hi, &h.q_lo, &h.q_hi],
                    [&theirs.p_lo, &theirs.p_hi, &theirs.q_lo, &theirs.q_hi]
                );
            }
        }
    }

    /// One tree's halves wrong at a mid step of a multi-tree ladder: the step's
    /// residual refuses it, and names the step.
    #[test]
    fn one_trees_halves_wrong_at_a_mid_step_are_refused() {
        let heights = [4, 3, 4];
        let (trees, proof) = round_trip(&heights);
        let outputs: Vec<(FE, FE)> = trees
            .iter()
            .map(|t| (t[0].p.evals()[0], t[0].q.evals()[0]))
            .collect();
        for tree in 0..3 {
            let mut bad = proof.clone();
            bad.layers[2].halves[tree].q_lo += FE::one();
            assert_eq!(
                verify(&bad, &outputs, &heights, &mut transcript()).unwrap_err(),
                Error::LayerRelationMismatch { layer: 2 },
                "tree {tree}"
            );
        }
        // A round value of the shared sumcheck, likewise.
        let mut bad = proof.clone();
        bad.layers[3].sumcheck.rounds[1].evaluations[2] += FE::one();
        assert_eq!(
            verify(&bad, &outputs, &heights, &mut transcript()).unwrap_err(),
            Error::LayerRelationMismatch { layer: 3 }
        );
        // A wrong output of one tree.
        let mut wrong = outputs.clone();
        wrong[1].0 += FE::one();
        assert!(verify(&proof, &wrong, &heights, &mut transcript()).is_err());
    }

    /// The residual is load-bearing: made inert, the mid-step tamper above is
    /// no longer refused at its step.
    #[test]
    fn the_residual_check_made_inert_lets_a_bad_step_through() {
        let heights = [4, 3, 4];
        let (trees, proof) = round_trip(&heights);
        let outputs: Vec<(FE, FE)> = trees
            .iter()
            .map(|t| (t[0].p.evals()[0], t[0].q.evals()[0]))
            .collect();
        let mut bad = proof.clone();
        bad.layers[2].halves[1].q_lo += FE::one();
        assert!(verify_with(&bad, &outputs, &heights, &mut transcript(), false).is_ok());
    }

    /// The shape is the statement's: a ladder a step short, a step with a tree
    /// missing, or heights the proof was not made for are refused.
    #[test]
    fn a_ladder_of_the_wrong_shape_is_refused() {
        let heights = [4, 2];
        let (trees, proof) = round_trip(&heights);
        let outputs: Vec<(FE, FE)> = trees
            .iter()
            .map(|t| (t[0].p.evals()[0], t[0].q.evals()[0]))
            .collect();
        let mut short = proof.clone();
        short.layers.pop();
        assert!(verify(&short, &outputs, &heights, &mut transcript()).is_err());
        let mut missing = proof.clone();
        missing.layers[1].halves.pop();
        assert!(verify(&missing, &outputs, &heights, &mut transcript()).is_err());
        assert!(verify(&proof, &outputs, &[4, 3], &mut transcript()).is_err());
        assert!(verify(&proof, &outputs[..1], &heights, &mut transcript()).is_err());
    }

    /// Over the extension too, which is where the argue runs.
    #[test]
    fn the_ladder_runs_over_the_extension() {
        let lift = |layer: FractionLayer<F>| {
            let up = |m: &Mle<F>| {
                Mle::new(m.evals().iter().map(|v| v.to_extension::<Ext3>()).collect()).unwrap()
            };
            FractionLayer::new(up(&layer.p), up(&layer.q)).unwrap()
        };
        let trees: Vec<Vec<FractionLayer<Ext3>>> = [3usize, 1, 3]
            .iter()
            .enumerate()
            .map(|(t, &k)| {
                let mut layers = vec![lift(random_layer(k, t as u64))];
                while layers.last().unwrap().num_vars() > 0 {
                    let next = layers.last().unwrap().fold().unwrap();
                    layers.push(next);
                }
                layers.reverse();
                layers
            })
            .collect();
        let refs: Vec<&[FractionLayer<Ext3>]> = trees.iter().map(|t| t.as_slice()).collect();
        let mut prover = DefaultTranscript::<Ext3>::new(b"ext");
        let (proof, claims) = prove(&refs, &mut prover).unwrap();
        let outputs: Vec<_> = trees
            .iter()
            .map(|t| (t[0].p.evals()[0], t[0].q.evals()[0]))
            .collect();
        let got = verify(
            &proof,
            &outputs,
            &[3, 1, 3],
            &mut DefaultTranscript::<Ext3>::new(b"ext"),
        )
        .unwrap();
        for (a, b) in got.iter().zip(&claims) {
            assert!(a.point == b.point && a.p == b.p && a.q == b.q);
        }
        assert_eq!(got.len(), claims.len());
    }
}
