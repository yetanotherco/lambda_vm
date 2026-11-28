use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext;
use math::field::goldilocks::GoldilocksField as Base;
use multilinear::Error;
use multilinear::gkr::FractionLayer;
use multilinear::gkr_lockstep::{self as ladder, GkrLadder, HostLayers, LadderTree, TreeStep};
use multilinear::mle::Mle;

type E = FieldElement<Ext>;
type B = FieldElement<Base>;

struct Rng(u64);

impl Rng {
    fn limb(&mut self) -> B {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        B::from(self.0)
    }

    fn ext(&mut self) -> E {
        E::new(std::array::from_fn(|_| self.limb()))
    }
}

fn transcript() -> DefaultTranscript<Ext> {
    DefaultTranscript::new(b"gkr base numerators")
}

fn random_layer(vars: usize, real: usize, rng: &mut Rng) -> FractionLayer<Ext> {
    let p = (0..real)
        .map(|_| rng.limb().to_extension())
        .chain(std::iter::repeat(E::zero()))
        .take(1 << vars)
        .collect();
    let q = (0..real)
        .map(|_| rng.ext())
        .chain(std::iter::repeat(E::one()))
        .take(1 << vars)
        .collect();
    FractionLayer {
        p: Mle::new(p).unwrap(),
        q: Mle::new(q).unwrap(),
    }
}

// Bind the highest variable within each of the two halves or four quarters.
// p's difference is in the base field until it is multiplied by the challenge.
fn bind_input(
    layer: &FractionLayer<Ext>,
    partitions: usize,
    r: E,
    wrong: bool,
) -> FractionLayer<Ext> {
    let half = layer.p.len() / (2 * partitions);
    let pairs = (0..partitions).flat_map(|j| (0..half).map(move |x| 2 * j * half + x));
    let (p, q) = pairs
        .map(|i| {
            let (lo, hi) = (
                layer.p.evals()[i].value()[0],
                layer.p.evals()[i + half].value()[0],
            );
            let difference = if wrong && i == 0 { hi } else { hi - lo };
            let (q_lo, q_hi) = (layer.q.evals()[i], layer.q.evals()[i + half]);
            (lo + difference * r, q_lo + r * (q_hi - q_lo))
        })
        .unzip();
    FractionLayer {
        p: Mle::new(p).unwrap(),
        q: Mle::new(q).unwrap(),
    }
}

fn bound_weights(u: E, r: E, weights: [E; 2]) -> [E; 2] {
    let eq = u * r + (E::one() - u) * (E::one() - r);
    weights.map(|w| w * eq)
}

fn base_first_round(
    input: &FractionLayer<Ext>,
    parts: usize,
    point: &[E],
    weights: [E; 2],
) -> Vec<E> {
    let half = input.p.len() / (2 * parts);
    let degree = if parts == 2 { 3 } else { 5 };
    (1..=degree)
        .map(|node| {
            let t = B::from(node);
            let [wp, wq] = bound_weights(point[0], t.to_extension(), weights);
            (0..half).fold(E::zero(), |sum, x| {
                let cell = |part: usize| {
                    let i = part * 2 * half + x;
                    let p = input.p.evals();
                    let q = input.q.evals();
                    let p_lo = p[i].value()[0];
                    (
                        p_lo + t * (p[i + half].value()[0] - p_lo),
                        q[i] + t * (q[i + half] - q[i]),
                    )
                };
                let pair = |lo, hi| {
                    let (p, q) = cell(lo);
                    let (p_hi, q_hi) = cell(hi);
                    (p * q_hi + p_hi * q, q * q_hi)
                };
                let (p, q) = if parts == 2 {
                    pair(0, 1)
                } else {
                    let (pa, qa) = pair(0, 2);
                    let (pb, qb) = pair(1, 3);
                    (pa * qb + pb * qa, qa * qb)
                };
                let eq = point[1..]
                    .iter()
                    .rev()
                    .enumerate()
                    .fold(E::one(), |acc, (bit, u)| {
                        acc * if x & (1 << bit) == 0 {
                            E::one() - u
                        } else {
                            *u
                        }
                    });
                sum + eq * (wp * p + wq * q)
            })
        })
        .collect()
}

fn half_values(halves: Vec<ladder::TreeHalves<Ext>>) -> Vec<[E; 4]> {
    halves
        .into_iter()
        .map(|h| [h.p_lo, h.p_hi, h.q_lo, h.q_hi])
        .collect()
}

#[test]
fn the_base_bind_matches_both_host_steps() {
    type Open = fn(&FractionLayer<Ext>, &[E], [E; 2]) -> Result<Box<dyn TreeStep<Ext>>, Error>;
    let steps: [(usize, Open); 2] = [
        (2, ladder::host_step::<Ext>),
        (4, ladder::host_step_double::<Ext>),
    ];
    let mut rng = Rng(0x174ea250);
    for vars in 4..=8 {
        for real in [1 << vars, (1 << (vars - 1)) + 1, 7 << (vars - 3)] {
            let input = random_layer(vars, real, &mut rng);
            for (partitions, open) in steps {
                let point: Vec<_> = (0..vars - partitions.trailing_zeros() as usize)
                    .map(|_| rng.ext())
                    .collect();
                let weights = [rng.ext(), rng.ext()];
                let r = rng.ext();
                let mut today: Box<dyn TreeStep<Ext>> = open(&input, &point, weights).unwrap();
                assert_eq!(
                    today.round().unwrap(),
                    base_first_round(&input, partitions, &point, weights)
                );
                today.bind(&r).unwrap();
                let reduced = bind_input(&input, partitions, r, false);
                let mut base: Box<dyn TreeStep<Ext>> =
                    open(&reduced, &point[1..], bound_weights(point[0], r, weights)).unwrap();
                for _ in &point[1..] {
                    assert_eq!(today.round().unwrap(), base.round().unwrap());
                    let bind = rng.ext();
                    today.bind(&bind).unwrap();
                    base.bind(&bind).unwrap();
                }
                assert_eq!(
                    half_values(today.halves().unwrap()),
                    half_values(base.halves().unwrap())
                );
            }
        }
    }
}

struct WrongBind<'a> {
    step: Box<dyn TreeStep<Ext>>,
    input: Option<&'a FractionLayer<Ext>>,
    point: Vec<E>,
    weights: [E; 2],
}

impl TreeStep<Ext> for WrongBind<'_> {
    fn round(&mut self) -> Result<Vec<E>, Error> {
        self.step.round()
    }

    fn bind(&mut self, r: &E) -> Result<(), Error> {
        if let Some(input) = self.input.take() {
            let folded = bind_input(input, 4, *r, true);
            self.step = ladder::host_step_double(
                &folded,
                &self.point[1..],
                bound_weights(self.point[0], *r, self.weights),
            )?;
            Ok(())
        } else {
            self.step.bind(r)
        }
    }

    fn halves(&self) -> Result<Vec<ladder::TreeHalves<Ext>>, Error> {
        self.step.halves()
    }
}

struct Mutant<'a>(HostLayers<'a, Ext>);

impl LadderTree<Ext> for Mutant<'_> {
    fn input_vars(&self) -> usize {
        self.0.input_vars()
    }

    fn output(&self) -> (E, E) {
        self.0.output()
    }

    fn step(
        &self,
        layer: usize,
        point: &[E],
        weights: [E; 2],
        claim: &(E, E),
    ) -> Result<Box<dyn TreeStep<Ext> + '_>, Error> {
        self.0.step(layer, point, weights, claim)
    }

    fn step_double(
        &self,
        layer: usize,
        point: &[E],
        weights: [E; 2],
        claim: &(E, E),
    ) -> Result<Box<dyn TreeStep<Ext> + '_>, Error> {
        if layer != self.input_vars() {
            return self.0.step_double(layer, point, weights, claim);
        }
        let input = &self.0.0[layer];
        Ok(Box::new(WrongBind {
            step: ladder::host_step_double(input, point, weights)?,
            input: Some(input),
            point: point.to_vec(),
            weights,
        }))
    }
}

#[test]
fn the_verifier_refuses_a_wrong_base_numerator_bind() {
    let mut layers = vec![random_layer(6, 57, &mut Rng(0x4a60fea))];
    while layers.last().unwrap().num_vars() > 0 {
        layers.push(layers.last().unwrap().fold().unwrap());
    }
    layers.reverse();
    let host = HostLayers(&layers);
    let mutant = Mutant(HostLayers(&layers));
    let check = |tree: &dyn LadderTree<Ext>| {
        let (proof, _) =
            ladder::prove_trees(&[tree], GkrLadder::Double, &mut transcript()).unwrap();
        ladder::verify(
            &proof,
            &[host.output()],
            &[host.input_vars()],
            GkrLadder::Double,
            &mut transcript(),
        )
    };
    assert!(check(&host).is_ok());
    assert_eq!(
        check(&mutant).unwrap_err(),
        Error::LayerRelationMismatch { layer: 2 }
    );
}

#[cfg(feature = "cuda")]
#[test]
fn a_device_ladder_proves_the_same_bytes_with_base_numerators() {
    use crypto::fiat_shamir::is_transcript::IsTranscript;
    use multilinear::claim_reduce::FactorSource;
    use multilinear::gkr::{FractionTree, GkrClaim};
    use multilinear::gkr_input_base as base;
    use multilinear::gpu;
    use multilinear::logup::{self, Affine, Interaction};

    if math_cuda::device::backend().is_err() {
        eprintln!("no device; skipping");
        return;
    }

    let proof_values = |proof: &ladder::LockstepProof<Ext>| {
        proof
            .layers
            .iter()
            .map(|layer| {
                let rounds: Vec<_> = layer
                    .sumcheck
                    .rounds
                    .iter()
                    .map(|round| round.evaluations.clone())
                    .collect();
                let halves: Vec<_> = layer
                    .halves
                    .iter()
                    .map(|h| [h.p_lo, h.p_hi, h.q_lo, h.q_hi])
                    .collect();
                (rounds, halves)
            })
            .collect::<Vec<_>>()
    };
    let claim_values = |claims: &[GkrClaim<Ext>]| {
        claims
            .iter()
            .map(|claim| (claim.point.clone(), claim.p, claim.q))
            .collect::<Vec<_>>()
    };

    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            base::force(None);
        }
    }
    let _restore = Restore;
    let mut rng = Rng(0xb2b2c3d);
    let rows = 1 << 11;
    let columns: Vec<Mle<Base>> = (0..3)
        .map(|_| Mle::new((0..rows).map(|_| rng.limb()).collect()).unwrap())
        .collect();
    let factors: Vec<_> = (0..columns.len())
        .map(|i| Some(FactorSource::direct(i)))
        .collect();
    let plans: Vec<_> = [5, 7, 8]
        .into_iter()
        .map(|count| {
            let interactions: Vec<_> = (0..count)
                .map(|i| {
                    Interaction::new(
                        Affine::new(
                            vec![(0, rng.limb().to_extension()), (1, -E::one())],
                            rng.limb().to_extension(),
                        ),
                        Affine::new(vec![(1, rng.ext()), (2, rng.ext())], E::from(i as u64)),
                    )
                })
                .collect();
            let plan = logup::input_plan(&interactions, &factors, rows).unwrap();
            assert!(plan.base_numerators());
            plan
        })
        .collect();
    let mut extension_plan = plans[0].clone();
    extension_plan.coeffs[1] = 0xffff_ffff_0000_0001;
    assert!(!extension_plan.base_numerators());
    extension_plan.constants[2] = 1;
    extension_plan.coeffs[1] = 0;
    assert!(!extension_plan.base_numerators());

    let refs: Vec<_> = columns.iter().collect();
    let resident = std::sync::Arc::new(gpu::upload_columns(&refs).expect("card required"));
    for form in [GkrLadder::Double, GkrLadder::Single] {
        let prove = |on| {
            base::force(Some(on));
            let start = base::steps();
            let trees: Vec<_> = gpu::with_gkr_tree_layout(form, || {
                plans
                    .iter()
                    .chain(std::iter::once(&extension_plan))
                    .map(|plan| {
                        let tree = gpu::input_layer_tree_from_columns_deferred(
                            resident.clone(),
                            0,
                            columns.len(),
                            rows,
                            plan,
                        )
                        .expect("tree builds on card");
                        FractionTree::from_device(tree).unwrap()
                    })
                    .collect()
            });
            let dyn_trees: Vec<&dyn LadderTree<Ext>> =
                trees.iter().map(|t| t as &dyn LadderTree<Ext>).collect();
            let mut tr = transcript();
            let (proof, claims) = ladder::prove_trees(&dyn_trees, form, &mut tr).unwrap();
            let outputs: Vec<_> = trees.iter().map(FractionTree::output).collect();
            let heights: Vec<_> = dyn_trees.iter().map(|tree| tree.input_vars()).collect();
            let verified =
                ladder::verify(&proof, &outputs, &heights, form, &mut transcript()).unwrap();
            assert_eq!(claim_values(&verified), claim_values(&claims));
            let expected = if on && form == GkrLadder::Double {
                plans.len() as u64
            } else {
                0
            };
            assert_eq!(
                base::steps() - start,
                expected,
                "base steps with {form:?}, knob {on}"
            );
            (proof, claim_values(&claims), tr.state())
        };
        let off = prove(false);
        let on = prove(true);
        let off_bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&off.0).unwrap();
        let on_bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&on.0).unwrap();
        assert_eq!(
            proof_values(&off.0),
            proof_values(&on.0),
            "canonical {form:?} proof"
        );
        assert_eq!(off.1, on.1, "{form:?} claims");
        assert_eq!(off.2, on.2, "{form:?} transcript");
        if off_bytes.as_slice() != on_bytes.as_slice() {
            let bytes = off_bytes
                .iter()
                .zip(on_bytes.iter())
                .filter(|(a, b)| a != b)
                .count();
            eprintln!("{form:?}: {bytes} raw proof bytes differ, canonical proof equal");
        } else {
            eprintln!("{form:?}: {} raw proof bytes identical", off_bytes.len());
        }
    }
}
