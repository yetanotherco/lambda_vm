//! Several sumcheck statements over one cube, proved in a single pass.
//!
//! A zerocheck is `Σ_x eq(r,x)·C(x) = 0`. A LogUp-GKR input-layer claim is
//! `Σ_x eq(z,x)·P(x) = p(z)`. Both have the shape `Σ_x weight(x)·poly(x) =
//! claimed` over the same trace, so batching them with powers of a challenge
//! costs one pass instead of one per argument.
//!
//! The factors are **one list, shared by every statement**: a column read by a
//! constraint and by a bus fingerprint is folded once, not twice. Each
//! statement is a rule that indexes that list, so a statement's weight table is
//! just another factor it multiplies in.

use crypto::fiat_shamir::is_transcript::IsTranscript;
use math::field::{element::FieldElement, traits::IsField};

use crate::{
    Error, challenge_powers,
    mle::Mle,
    poly::SumcheckPolynomial,
    program::{self, Program},
    sumcheck::{self, SumcheckProof},
};

/// A weight table the batch multiplies its statements by, as the caller has it:
/// built, or as the point of an `eq` table.
///
/// The second is what lets a device build the table where it is folded —
/// `2^n` cells from `n` coordinates — instead of the host building it and
/// sending it across (`LAMBDA_VM_ARGUE_DEVICE_TABLES`). Whoever runs the rounds
/// on the host builds it here, the same table either way.
#[derive(Clone, Debug)]
pub enum Weight<F: IsField> {
    Table(Mle<F>),
    /// `eq(point, ·)`.
    Eq(Vec<FieldElement<F>>),
}

impl<F: IsField + 'static> Weight<F> {
    /// The table, built here if it came as a point.
    pub fn materialize(self) -> Result<Mle<F>, Error> {
        match self {
            Self::Table(table) => Ok(table),
            Self::Eq(point) => crate::eq::eq_mle(&point),
        }
    }

    fn view(&self) -> crate::gpu::WeightView<'_, F> {
        match self {
            Self::Table(table) => crate::gpu::WeightView::Table(table),
            Self::Eq(point) => crate::gpu::WeightView::Eq(point),
        }
    }
}

/// One statement's rule: what it makes of the batch's factors.
///
/// Reads the **whole** factor list, so a sub-argument that already combines a
/// prefix of it can be used unchanged.
pub struct Rule<'a, F: IsField> {
    body: Body<'a, F>,
    degree: usize,
}

/// A rule is either a closure or the program it is. A statement that can say
/// which program it is gets the sumcheck's device path; the closure form is for
/// the ones that cannot.
enum Body<'a, F: IsField> {
    Closure(RuleFn<'a, F>),
    Compiled(Program<F>),
}

/// A statement's value from the batch's factor values.
type RuleFn<'a, F> = Box<dyn Fn(&[FieldElement<F>]) -> FieldElement<F> + Sync + 'a>;

impl<'a, F: IsField> Rule<'a, F> {
    /// `degree` must upper-bound the rule's total degree in the factors, its
    /// weight table included.
    pub fn new(
        degree: usize,
        eval: impl Fn(&[FieldElement<F>]) -> FieldElement<F> + Sync + 'a,
    ) -> Self {
        Self {
            body: Body::Closure(Box::new(eval)),
            degree,
        }
    }

    /// The same rule as straight-line code over the factor values.
    pub fn compiled(degree: usize, program: Program<F>) -> Self {
        Self {
            body: Body::Compiled(program),
            degree,
        }
    }

    pub fn degree(&self) -> usize {
        self.degree
    }

    /// The program this rule is, when it has one.
    pub fn program(&self) -> Option<&Program<F>> {
        match &self.body {
            Body::Compiled(program) => Some(program),
            Body::Closure(_) => None,
        }
    }

    pub fn apply(&self, values: &[FieldElement<F>]) -> FieldElement<F> {
        self.apply_in(values, &mut Vec::new())
    }

    /// The same, reusing the caller's scratch for a compiled rule's steps.
    pub fn apply_in(
        &self,
        values: &[FieldElement<F>],
        scratch: &mut Vec<FieldElement<F>>,
    ) -> FieldElement<F> {
        match &self.body {
            Body::Closure(eval) => eval(values),
            Body::Compiled(program) => program.eval(values, scratch),
        }
    }
}

/// The batch as one polynomial: `Σ_i lambda^i · rule_i`.
pub struct Batched<'a, F: IsField> {
    polys: Vec<Mle<F>>,
    rules: Vec<Rule<'a, F>>,
    lambdas: Vec<FieldElement<F>>,
    num_vars: usize,
    degree: usize,
    /// The whole batch as one program, when every rule is compiled. The round
    /// loop runs this instead of the rules, and it is what a device gets.
    program: Option<Program<F>>,
    /// Whether the rounds may still go to a device. False once one has handed
    /// its factors back: what it stopped for is that the cube got too small to
    /// be worth sending anywhere.
    dispatch: bool,
}

impl<'a, F: IsField + 'static> Batched<'a, F> {
    /// `lambdas` weights the statements; there must be one per rule.
    pub fn new(
        polys: Vec<Mle<F>>,
        rules: Vec<Rule<'a, F>>,
        lambdas: Vec<FieldElement<F>>,
    ) -> Result<Self, Error> {
        if rules.is_empty() {
            return Err(Error::EmptyPolynomial);
        }
        if lambdas.len() != rules.len() {
            return Err(Error::VariableCountMismatch {
                expected: rules.len(),
                got: lambdas.len(),
            });
        }
        let num_vars = polys.first().map(|p| p.num_vars()).unwrap_or(0);
        for p in &polys {
            if p.num_vars() != num_vars {
                return Err(Error::VariableCountMismatch {
                    expected: num_vars,
                    got: p.num_vars(),
                });
            }
        }
        let degree = rules.iter().map(Rule::degree).max().unwrap_or(0);
        let programs: Option<Vec<&Program<F>>> = rules.iter().map(Rule::program).collect();
        let program = match programs {
            Some(programs) => Some(program::combine(&programs, &lambdas)?),
            None => None,
        };
        Ok(Self {
            polys,
            rules,
            lambdas,
            num_vars,
            degree,
            program,
            dispatch: true,
        })
    }

    /// Puts `polys` in front of the ones already here.
    ///
    /// A batch whose first factors a device holds is built over the rest
    /// alone; this is what brings them back when the device turns the rounds
    /// down and the host has to run them.
    pub fn prepend(&mut self, mut polys: Vec<Mle<F>>) -> Result<(), Error> {
        if polys.is_empty() {
            return Ok(());
        }
        let num_vars = polys[0].num_vars();
        for p in polys.iter().chain(&self.polys) {
            if p.num_vars() != num_vars {
                return Err(Error::VariableCountMismatch {
                    expected: num_vars,
                    got: p.num_vars(),
                });
            }
        }
        polys.append(&mut self.polys);
        self.polys = polys;
        self.num_vars = num_vars;
        Ok(())
    }

    /// The batch's factors, replaced whole by the ones a device folded.
    ///
    /// A batch whose leading factors a device holds is built over the rest
    /// alone; when the device stops part-way it hands back **every** factor,
    /// resident or not, as its rounds left them — so this replaces the list
    /// rather than adding to it. The rounds that follow stay here: the cube
    /// the device stopped at is the one it was no longer worth sending.
    pub fn adopt(&mut self, polys: Vec<Mle<F>>) -> Result<(), Error> {
        let num_vars = polys.first().map(Mle::num_vars).unwrap_or(0);
        for p in &polys {
            if p.num_vars() != num_vars {
                return Err(Error::VariableCountMismatch {
                    expected: num_vars,
                    got: p.num_vars(),
                });
            }
        }
        self.polys = polys;
        self.num_vars = num_vars;
        self.dispatch = false;
        Ok(())
    }

    /// The batch as one program, when it has one.
    pub fn program(&self) -> Option<&Program<F>> {
        self.program.as_ref()
    }
}

impl<F: IsField + 'static> SumcheckPolynomial<F> for Batched<'_, F> {
    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn degree(&self) -> usize {
        self.degree
    }

    fn polys(&self) -> &[Mle<F>] {
        &self.polys
    }

    fn combine(&self, values: &[FieldElement<F>]) -> FieldElement<F> {
        self.combine_in(values, &mut Vec::new())
    }

    fn combine_in(
        &self,
        values: &[FieldElement<F>],
        scratch: &mut Vec<FieldElement<F>>,
    ) -> FieldElement<F> {
        if let Some(program) = &self.program {
            return program.eval(values, scratch);
        }
        self.rules
            .iter()
            .zip(&self.lambdas)
            .fold(FieldElement::zero(), |acc, (rule, lambda)| {
                acc + lambda * rule.apply_in(values, scratch)
            })
    }

    fn fix_first_variable(&mut self, r: &FieldElement<F>) -> Result<(), Error> {
        for p in &mut self.polys {
            p.fix_first_variable_in_place(r)?;
        }
        self.num_vars -= 1;
        Ok(())
    }

    fn program(&self) -> Option<&Program<F>> {
        self.program.as_ref().filter(|_| self.dispatch)
    }

    fn accept_folded(&mut self, polys: Vec<Mle<F>>) -> Result<(), Error> {
        if polys.len() != self.polys.len() {
            return Err(Error::VariableCountMismatch {
                expected: self.polys.len(),
                got: polys.len(),
            });
        }
        self.num_vars = polys.first().map(Mle::num_vars).unwrap_or(0);
        self.polys = polys;
        Ok(())
    }
}

/// The degree the batched sumcheck runs at: the worst statement's.
pub fn degree_of<F: IsField + 'static>(rules: &[Rule<'_, F>]) -> usize {
    rules.iter().map(Rule::degree).max().unwrap_or(0)
}

/// Proves every statement in one sumcheck, returning the proof and its point.
///
/// `claims[i]` is what `Σ_x rule_i(x)` must come to. They are absorbed before
/// the batching challenge, so the prover cannot pick a statement after seeing
/// it.
pub fn prove<F, T>(
    polys: Vec<Mle<F>>,
    rules: Vec<Rule<'_, F>>,
    claims: &[FieldElement<F>],
    transcript: &mut T,
) -> Result<(SumcheckProof<F>, Vec<FieldElement<F>>), Error>
where
    F: IsField + 'static,
    T: IsTranscript<F>,
{
    if claims.len() != rules.len() {
        return Err(Error::VariableCountMismatch {
            expected: rules.len(),
            got: claims.len(),
        });
    }
    for claim in claims {
        transcript.append_field_element(claim);
    }
    let lambdas = challenge_powers(&transcript.sample_field_element(), rules.len());
    sumcheck::prove(Batched::new(polys, rules, lambdas)?, transcript)
}

/// A batched sumcheck's proof, the point its rounds drew, and what every
/// factor slot was bound to there — empty when the factors were the host's own
/// and folded away as they went.
type ResidentProof<F> = (SumcheckProof<F>, Vec<FieldElement<F>>, Vec<FieldElement<F>>);

/// The same, over factors a device already holds — the batch's first ones, in
/// the order the rules read them.
///
/// Returns the proof, the point the rounds drew, and **what every slot was
/// bound to there**, in slot order. That last one is free on the device path —
/// the rounds fold the factors where they lie, so the values are already up
/// there — and empty when the host ran the rounds, which drops each factor as
/// it folds.
///
/// `extra` is what the device does not have (the weight tables), and `absent`
/// makes what it does. That second one is a builder and not a value because on
/// the device path it is never called: building a table's factors here is most
/// of what the argument used to spend on the host, and the whole point of them
/// being up there is not to.
///
/// It *is* called when the device turns the rounds down. Nothing has been said
/// about the factors to the transcript by then — only the claims and the
/// batching challenge, which are the same either way — so the host path
/// continues from where the device left off and produces the same proof.
///
/// A weight given as a point ([`Weight::Eq`]) stays one until then: the device
/// builds it with the rounds, or the host does when they are turned down.
pub fn prove_resident<F, T, B>(
    extra: Vec<Weight<F>>,
    device: Option<std::sync::Arc<crate::gpu::DeviceFactors>>,
    absent: B,
    rules: Vec<Rule<'_, F>>,
    claims: &[FieldElement<F>],
    transcript: &mut T,
) -> Result<ResidentProof<F>, Error>
where
    F: IsField + 'static,
    FieldElement<F>: Send + Sync,
    T: IsTranscript<F>,
    B: Fn() -> Result<Vec<Mle<F>>, Error>,
{
    prove_resident_with::<F, F, T, B>(extra, device, absent, rules, claims, None, transcript)
}

/// [`prove_resident`], with the zerocheck's rounds the stage-1 way when
/// `fused` describes the batch and `LAMBDA_VM_ARGUE_FUSED` is on
/// ([`crate::gpu_fused`]): the same messages, the same proof. A decline runs
/// today's device rounds, then the host's, as before.
#[allow(clippy::too_many_arguments)]
pub fn prove_resident_with<Base, F, T, B>(
    extra: Vec<Weight<F>>,
    device: Option<std::sync::Arc<crate::gpu::DeviceFactors>>,
    absent: B,
    rules: Vec<Rule<'_, F>>,
    claims: &[FieldElement<F>],
    fused: Option<crate::gpu_fused::FusedInput<'_, Base, F>>,
    transcript: &mut T,
) -> Result<ResidentProof<F>, Error>
where
    Base: IsField + math::field::traits::IsSubFieldOf<F> + 'static,
    F: IsField + 'static,
    FieldElement<F>: Send + Sync,
    T: IsTranscript<F>,
    B: Fn() -> Result<Vec<Mle<F>>, Error>,
{
    let _ = &fused;
    if claims.len() != rules.len() {
        return Err(Error::VariableCountMismatch {
            expected: rules.len(),
            got: claims.len(),
        });
    }
    let Some(device) = device else {
        let mut polys = absent()?;
        for weight in extra {
            polys.push(weight.materialize()?);
        }
        let (proof, point) = prove(polys, rules, claims, transcript)?;
        return Ok((proof, point, Vec::new()));
    };

    for claim in claims {
        transcript.append_field_element(claim);
    }
    let lambdas = challenge_powers(&transcript.sample_field_element(), rules.len());
    // Built weights go in the batch now, as they always did. When any came as a
    // point, the batch starts empty and the weights wait: for the device to
    // build them with its rounds, or for the host to, if it turns them down.
    let pending = extra.iter().any(|weight| matches!(weight, Weight::Eq(_)));
    let (built, waiting) = if pending {
        (Vec::new(), extra)
    } else {
        let tables = extra
            .into_iter()
            .map(Weight::materialize)
            .collect::<Result<Vec<_>, _>>()?;
        (tables, Vec::new())
    };
    #[cfg(feature = "cuda")]
    let weights = lambdas.clone();
    let mut batched = Batched::new(built, rules, lambdas)?;
    let degree = batched.degree().max(1);
    if let Some(program) = batched.program() {
        let views: Vec<crate::gpu::WeightView<'_, F>> = if pending {
            waiting.iter().map(Weight::view).collect()
        } else {
            batched
                .polys()
                .iter()
                .map(crate::gpu::WeightView::Table)
                .collect()
        };
        #[cfg(feature = "cuda")]
        if let Some(input) = fused.as_ref().filter(|_| crate::gpu_fused::argue_fused()) {
            let claim = weights
                .iter()
                .zip(claims)
                .fold(FieldElement::<F>::zero(), |acc, (l, c)| acc + l * c);
            let check = crate::gpu_fused::argue_fused_xcheck();
            let start = crate::whir_split::tick();
            let attempt = crate::gpu_fused::prove_fused(
                &device,
                input,
                &weights,
                &claim,
                degree,
                check,
                |evaluations| {
                    for e in evaluations {
                        transcript.append_field_element(e);
                    }
                    transcript.sample_field_element()
                },
            );
            crate::whir_split::add_tick(&crate::whir_split::ZC_FUSED_TIME, start);
            match attempt {
                None => {
                    crate::gpu_fused::note_decline();
                    crate::whir_split::bump(&crate::whir_split::ZC_FUSED_DECLINED);
                    if check || crate::gpu_fused::argue_fused_log() {
                        eprintln!(
                            "ARGUE FUSED: declined · factors {} n={} (today's rounds run)",
                            device.width(),
                            device.len().trailing_zeros(),
                        );
                    }
                }
                Some(Err(error)) => return Err(error),
                Some(Ok((mut rounds, mut point, factors, times))) => {
                    crate::whir_split::bump(&crate::whir_split::ZC_FUSED);
                    if check {
                        cross_check(
                            &device, &views, program, degree, &rounds, &point, &factors, times,
                        )?;
                    } else if crate::gpu_fused::argue_fused_log() {
                        eprintln!(
                            "ARGUE FUSED: factors {} roots {} terms {} slots {} n={} · fused {:.3} ms (grid \
                             {:.3} fold2 {:.3} rounds {:.3})",
                            device.width(),
                            times.roots,
                            times.terms,
                            times.slots,
                            device.len().trailing_zeros(),
                            times.total * 1e3,
                            times.grid * 1e3,
                            times.fold2 * 1e3,
                            times.rounds * 1e3,
                        );
                    }
                    let host_tail = crate::whir_split::tick();
                    batched.adopt(factors)?;
                    let left = batched.num_vars();
                    let (tail, tail_point) =
                        sumcheck::prove_rounds(&mut batched, left, transcript)?;
                    rounds.extend(tail);
                    point.extend(tail_point);
                    let bound: Option<Vec<FieldElement<F>>> = batched
                        .polys()
                        .iter()
                        .map(|factor| factor.as_constant().cloned())
                        .collect();
                    let bound = bound.ok_or(Error::NoVariablesLeft)?;
                    crate::whir_split::add_tick(&crate::whir_split::ZC_TAIL, host_tail);
                    return Ok((SumcheckProof { rounds }, point, bound));
                }
            }
        }
        let attempt =
            crate::gpu::prove_sumcheck_resident(&device, &views, program, degree, |evaluations| {
                for e in evaluations {
                    transcript.append_field_element(e);
                }
                transcript.sample_field_element()
            });
        if let Some(outcome) = attempt {
            let (mut rounds, mut point, factors) = outcome?;
            let host_tail = crate::whir_split::tick();
            // The device stopped where the cube stopped being worth sending;
            // the rest of the rounds run over the factors it folded.
            batched.adopt(factors)?;
            let left = batched.num_vars();
            let (tail, tail_point) = sumcheck::prove_rounds(&mut batched, left, transcript)?;
            rounds.extend(tail);
            point.extend(tail_point);
            // Binding every variable is evaluating at the point, so the factor
            // values the caller needs are the factors themselves by now.
            let bound: Option<Vec<FieldElement<F>>> = batched
                .polys()
                .iter()
                .map(|factor| factor.as_constant().cloned())
                .collect();
            let bound = bound.ok_or(Error::NoVariablesLeft)?;
            crate::whir_split::add_tick(&crate::whir_split::ZC_TAIL, host_tail);
            return Ok((SumcheckProof { rounds }, point, bound));
        }
    }
    // Declined before the first round: the host runs them, and for that the
    // factors have to be here after all — and any weight that waited as a point.
    if pending {
        let weights = waiting
            .into_iter()
            .map(Weight::materialize)
            .collect::<Result<Vec<_>, _>>()?;
        batched.prepend(weights)?;
    }
    batched.prepend(absent()?)?;
    let (proof, point) = sumcheck::prove(batched, transcript)?;
    Ok((proof, point, Vec::new()))
}

/// `LAMBDA_VM_ARGUE_FUSED_XCHECK`: today's device rounds over the same factors
/// with the fused rounds' challenges, every message and the factors at the
/// crossover compared, both timed. The fused rounds only read the lifted
/// factors, so today's can still fold them. A difference refuses the prove.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn cross_check<F>(
    device: &crate::gpu::DeviceFactors,
    views: &[crate::gpu::WeightView<'_, F>],
    program: &Program<F>,
    degree: usize,
    rounds: &[crate::sumcheck::RoundProof<F>],
    point: &[FieldElement<F>],
    factors: &[Mle<F>],
    times: crate::gpu_fused::FusedTimes,
) -> Result<(), Error>
where
    F: IsField + 'static,
{
    let inert = crate::gpu_fused::faults().xcheck_inert;
    let mut at = 0usize;
    let mut parted: Option<usize> = None;
    let started = std::time::Instant::now();
    let replay =
        crate::gpu::prove_sumcheck_resident(device, views, program, degree, |evaluations| {
            let k = at.min(point.len() - 1);
            if parted.is_none() && rounds.get(at).is_none_or(|r| r.evaluations != evaluations) {
                parted = Some(at);
            }
            at += 1;
            point[k].clone()
        });
    let today = started.elapsed().as_secs_f64();
    let verdict = match replay {
        None => "unchecked (today's rounds declined)".to_string(),
        Some(Err(error)) => format!("unchecked (today's rounds failed: {error})"),
        Some(Ok((today_rounds, _, today_factors))) => {
            if parted.is_none() && today_rounds.len() != rounds.len() {
                parted = Some(today_rounds.len().min(rounds.len()));
            }
            let same_factors = today_factors.len() == factors.len()
                && today_factors
                    .iter()
                    .zip(factors)
                    .all(|(a, b)| a.evals() == b.evals());
            match (parted, same_factors) {
                (None, true) => "ok".to_string(),
                (Some(round), _) => format!("PARTED at round {round}"),
                (None, false) => "PARTED at the crossover's factors".to_string(),
            }
        }
    };
    eprintln!(
        "ARGUE FUSED: factors {} roots {} terms {} slots {} n={} · fused {:.3} ms (grid {:.3} fold2 {:.3} rounds \
         {:.3}) · today {:.3} ms · R {:.3} · xcheck {verdict}",
        device.width(),
        times.roots,
        times.terms,
        times.slots,
        device.len().trailing_zeros(),
        times.total * 1e3,
        times.grid * 1e3,
        times.fold2 * 1e3,
        times.rounds * 1e3,
        today * 1e3,
        times.total / today.max(1e-9),
    );
    if verdict.starts_with("PARTED") && !inert {
        return Err(Error::DeviceFailed {
            stage: "fused cross-check",
        });
    }
    if verdict == "ok" {
        crate::gpu_fused::note_xcheck();
    }
    Ok(())
}

/// Checks the batched sumcheck against the factor values it reduces to.
///
/// `values_at` is handed the sumcheck point and returns every factor's value
/// there. A weight table is not committed, so the verifier computes it in
/// closed form; the trace factors come from the proof, and binding *those* to
/// the committed columns is the caller's next step.
///
/// Returns the point.
pub fn verify<F, T, V>(
    proof: &SumcheckProof<F>,
    rules: &[Rule<'_, F>],
    claims: &[FieldElement<F>],
    values_at: V,
    num_vars: usize,
    transcript: &mut T,
) -> Result<Vec<FieldElement<F>>, Error>
where
    F: IsField + 'static,
    T: IsTranscript<F>,
    V: FnOnce(&[FieldElement<F>]) -> Result<Vec<FieldElement<F>>, Error>,
{
    if rules.is_empty() {
        return Err(Error::EmptyPolynomial);
    }
    if claims.len() != rules.len() {
        return Err(Error::VariableCountMismatch {
            expected: rules.len(),
            got: claims.len(),
        });
    }
    for claim in claims {
        transcript.append_field_element(claim);
    }
    let lambdas = challenge_powers(&transcript.sample_field_element(), rules.len());

    let claimed = lambdas
        .iter()
        .zip(claims)
        .fold(FieldElement::<F>::zero(), |acc, (l, c)| acc + l * c);
    let claim = sumcheck::verify(proof, claimed, num_vars, degree_of(rules), transcript)?;

    let values = values_at(&claim.point)?;
    let rebuilt = rules
        .iter()
        .zip(&lambdas)
        .fold(FieldElement::<F>::zero(), |acc, (rule, lambda)| {
            acc + lambda * rule.apply(&values)
        });
    if rebuilt != claim.expected_evaluation {
        return Err(Error::BatchMismatch);
    }

    Ok(claim.point)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use math::field::goldilocks::GoldilocksField as F;

    use crate::{
        eq::{eq_eval, eq_mle},
        gkr::{self, FractionLayer, FractionTree},
    };

    type FE = FieldElement<F>;

    fn transcript() -> DefaultTranscript<F> {
        DefaultTranscript::<F>::new(b"batch-test")
    }

    /// Factor layout shared by every statement below.
    const MULT: usize = 0;
    const VALUE: usize = 1;
    const SQUARE: usize = 2;
    const EQ_R: usize = 3;
    const EQ_Z: usize = 4;

    /// A bus and a constraint over the same two columns: `mult` is the signed
    /// multiplicity, `value` the fingerprint, `square` its square. Each
    /// fingerprint is sent once and received once, so the bus balances.
    fn columns(num_vars: usize) -> [Mle<F>; 3] {
        let size = 1usize << num_vars;
        let half = size / 2;
        let value: Vec<FE> = (0..size).map(|i| FE::from((i % half) as u64 + 1)).collect();
        let mult: Vec<FE> = (0..size)
            .map(|i| if i < half { FE::one() } else { -FE::one() })
            .collect();
        let square: Vec<FE> = value.iter().map(|x| x * x).collect();
        [
            Mle::new(mult).unwrap(),
            Mle::new(value).unwrap(),
            Mle::new(square).unwrap(),
        ]
    }

    /// The three statements, all indexing the one factor list:
    ///
    /// - the zerocheck, `Σ_x eq(r,x)·(value² − square) = 0`;
    /// - the bus's numerator claim, `Σ_x eq(z,x)·mult = p(z)`;
    /// - its denominator claim, `Σ_x eq(z,x)·(alpha − value) = q(z)`.
    ///
    /// The denominator is never a column: it is `alpha − value`, read off the
    /// committed `value`.
    fn statements(alpha: FE) -> Vec<Rule<'static, F>> {
        vec![
            Rule::new(3, |v: &[FE]| v[EQ_R] * (v[VALUE] * v[VALUE] - v[SQUARE])),
            Rule::new(2, |v: &[FE]| v[EQ_Z] * v[MULT]),
            Rule::new(2, move |v: &[FE]| v[EQ_Z] * (alpha - v[VALUE])),
        ]
    }

    /// Runs GKR over the bus, then settles its input-layer claim in the same
    /// sumcheck as the constraint's zerocheck.
    ///
    /// `batch_mult` overrides the multiplicity column the *batch* reads, while
    /// the tree keeps the original — a prover arguing about a different table
    /// than the one it ran GKR over.
    fn fuse(
        cols: [Mle<F>; 3],
        num_vars: usize,
        batch_mult: Option<Mle<F>>,
    ) -> Result<usize, Error> {
        let alpha = FE::from(97);
        let [mult, value, square] = cols;
        let batch_mult = batch_mult.unwrap_or_else(|| mult.clone());
        let denominator = Mle::new(value.evals().iter().map(|x| alpha - x).collect())?;
        let tree = FractionTree::build(FractionLayer::new(mult.clone(), denominator)?)?;
        let output = tree.output();

        let mut prover = transcript();
        let gkr_out = gkr::prove(&tree, &mut prover)?;
        // The zerocheck challenge, drawn from the same transcript.
        let r: Vec<FE> = (0..num_vars)
            .map(|_| prover.sample_field_element())
            .collect();
        let z = gkr_out.claim.point.clone();

        let factors = || -> Result<Vec<Mle<F>>, Error> {
            Ok(vec![
                batch_mult.clone(),
                value.clone(),
                square.clone(),
                eq_mle(&r)?,
                eq_mle(&z)?,
            ])
        };
        let claims = [FE::zero(), gkr_out.claim.p, gkr_out.claim.q];

        let (proof, point) = prove(factors()?, statements(alpha), &claims, &mut prover)?;
        let trace_values: Vec<FE> = factors()?[..=SQUARE]
            .iter()
            .map(|f| f.evaluate(&point))
            .collect::<Result<_, _>>()?;

        let mut verifier = transcript();
        let gkr_claim = gkr::verify(&gkr_out.proof, output, &mut verifier)?;
        let vr: Vec<FE> = (0..num_vars)
            .map(|_| verifier.sample_field_element())
            .collect();

        let checked = verify(
            &proof,
            &statements(alpha),
            &claims,
            // The weights are not committed: the verifier computes them.
            |at: &[FE]| {
                let mut values = trace_values.clone();
                values.push(eq_eval(&vr, at)?);
                values.push(eq_eval(&gkr_claim.point, at)?);
                Ok(values)
            },
            num_vars,
            &mut verifier,
        )?;
        assert_eq!(checked, point);
        Ok(proof.rounds.len())
    }

    /// The design decision this module exists for: the constraint's zerocheck
    /// and the bus's input-layer claim share one sumcheck, so every column is
    /// folded once.
    #[test]
    fn a_bus_claim_settles_in_the_constraint_s_sumcheck() {
        for num_vars in 1..=4usize {
            let rounds = fuse(columns(num_vars), num_vars, None)
                .unwrap_or_else(|e| panic!("num_vars={num_vars}: {e:?}"));
            // One pass over the cube for all three statements, not one each.
            assert_eq!(rounds, num_vars);
        }
    }

    #[test]
    fn a_constraint_broken_in_one_row_is_rejected() {
        let mut cols = columns(3);
        let mut square = cols[2].evals().to_vec();
        square[5] += FE::one();
        cols[2] = Mle::new(square).unwrap();

        assert!(fuse(cols, 3, None).is_err());
    }

    /// The GKR's input-layer claim is only worth anything if it lands on the
    /// same table the rest of the argument reads.
    #[test]
    fn a_bus_table_the_tree_was_not_built_on_is_rejected() {
        let cols = columns(3);
        let mut mult = cols[0].evals().to_vec();
        mult[2] += FE::one();

        assert!(fuse(cols, 3, Some(Mle::new(mult).unwrap())).is_err());
    }

    // ---------------------------------------------------------------
    // The batching itself.
    // ---------------------------------------------------------------

    fn mle(vals: &[u64]) -> Mle<F> {
        Mle::new(vals.iter().map(|v| FE::from(*v)).collect()).unwrap()
    }

    /// Two statements over one factor list: `Σ eq(r,x)·a(x) = ã(r)` and
    /// `Σ eq(r,x)·b(x) = b̃(r)`, sharing the weight table.
    fn two_evaluation_claims(r: &[FE]) -> (Vec<Mle<F>>, Vec<Rule<'static, F>>, [FE; 2]) {
        let a = mle(&[3, 5, 8, 13]);
        let b = mle(&[21, 34, 55, 89]);
        let claims = [a.evaluate(r).unwrap(), b.evaluate(r).unwrap()];
        let polys = vec![a, b, eq_mle(r).unwrap()];
        let rules = vec![
            Rule::new(2, |v: &[FE]| v[2] * v[0]),
            Rule::new(2, |v: &[FE]| v[2] * v[1]),
        ];
        (polys, rules, claims)
    }

    fn run_two(r: &[FE], claims: [FE; 2]) -> Result<(), Error> {
        let (polys, rules, _) = two_evaluation_claims(r);
        let values = polys.clone();
        let (proof, _) = prove(polys, rules, &claims, &mut transcript())?;

        let (_, rules, _) = two_evaluation_claims(r);
        verify(
            &proof,
            &rules,
            &claims,
            |at: &[FE]| values.iter().map(|p| p.evaluate(at)).collect(),
            r.len(),
            &mut transcript(),
        )
        .map(|_| ())
    }

    /// A weight given as its point proves exactly what its table proves. With
    /// no device the host builds the point into the same table before the
    /// rounds, so the proof, the point, the transcript after them — everything
    /// — is the table's.
    #[test]
    fn a_weight_given_as_its_point_proves_what_its_table_proves() {
        let r = [FE::from(11), FE::from(13)];
        let (polys, _, claims) = two_evaluation_claims(&r);
        let run = |weight: Weight<F>| {
            let (_, rules, _) = two_evaluation_claims(&r);
            let mut t = transcript();
            let (proof, point, bound) = prove_resident(
                vec![weight],
                None,
                || Ok(polys[..2].to_vec()),
                rules,
                &claims,
                &mut t,
            )
            .unwrap();
            (proof, point, bound, t.state())
        };
        let from_table = run(Weight::Table(eq_mle(&r).unwrap()));
        let from_point = run(Weight::Eq(r.to_vec()));
        assert_eq!(from_table, from_point);

        let (_, rules, _) = two_evaluation_claims(&r);
        verify(
            &from_point.0,
            &rules,
            &claims,
            |at: &[FE]| polys.iter().map(|p| p.evaluate(at)).collect(),
            r.len(),
            &mut transcript(),
        )
        .expect("the proof from a point verifies");
    }

    #[test]
    fn statements_sharing_a_weight_table_hold_it_once() {
        let r = [FE::from(11), FE::from(13)];
        let (polys, _, claims) = two_evaluation_claims(&r);
        // Three factors for two statements: the weight is shared.
        assert_eq!(polys.len(), 3);
        run_two(&r, claims).unwrap();
    }

    #[test]
    fn a_false_claim_in_one_statement_is_rejected() {
        let r = [FE::from(11), FE::from(13)];
        let (_, _, claims) = two_evaluation_claims(&r);
        let lying = [claims[0], claims[1] + FE::one()];
        assert!(run_two(&r, lying).is_err());
    }

    #[test]
    fn the_batch_degree_is_the_worst_statement_s() {
        let rules: Vec<Rule<'static, F>> = vec![
            Rule::new(2, |v: &[FE]| v[0]),
            Rule::new(5, |v: &[FE]| v[0]),
            Rule::new(3, |v: &[FE]| v[0]),
        ];
        assert_eq!(degree_of(&rules), 5);
    }

    #[test]
    fn a_claim_per_statement_is_required() {
        let r = [FE::from(11), FE::from(13)];
        let (polys, rules, _) = two_evaluation_claims(&r);
        assert_eq!(
            prove(polys, rules, &[FE::zero()], &mut transcript()).unwrap_err(),
            Error::VariableCountMismatch {
                expected: 2,
                got: 1
            }
        );
    }

    #[test]
    fn factors_of_differing_heights_are_rejected() {
        let rules: Vec<Rule<'static, F>> = vec![Rule::new(1, |v: &[FE]| v[0])];
        let result = Batched::new(
            vec![mle(&[1, 2]), mle(&[1, 2, 3, 4])],
            rules,
            vec![FE::one()],
        );
        assert!(matches!(
            result.err(),
            Some(Error::VariableCountMismatch { .. })
        ));
    }

    #[test]
    fn a_proof_replayed_under_another_transcript_is_rejected() {
        let r = [FE::from(11), FE::from(13)];
        let (polys, rules, claims) = two_evaluation_claims(&r);
        let values = polys.clone();
        let (proof, _) = prove(polys, rules, &claims, &mut transcript()).unwrap();

        let (_, rules, _) = two_evaluation_claims(&r);
        let mut other = DefaultTranscript::<F>::new(b"a-different-statement");
        assert!(
            verify(
                &proof,
                &rules,
                &claims,
                |at: &[FE]| values.iter().map(|p| p.evaluate(at)).collect(),
                r.len(),
                &mut other,
            )
            .is_err()
        );
    }
}
