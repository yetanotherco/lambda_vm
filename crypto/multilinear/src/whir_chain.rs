//! WHIR with its rounds chained: fold `k` variables at a time, committing the
//! successor codeword each round, so a query opens a block of `2^k` instead of
//! the whole message.
//!
//! The **committed** codeword is base-field, because a trace is. The extension
//! only enters at the first fold, so the largest structure in the proof — the
//! codeword and the Merkle leaves over it, `blowup` times the trace — costs a
//! third of what it would over the extension. That is why the first round's
//! openings have a different value type from every later one, and the type says
//! so rather than leaving it to a reader.
//!
//! [`whir_eval`](crate::whir_eval) is the one-round case, and there a block
//! *is* the message — which is what dominates its proof size and its verifier's
//! work. Here the sumcheck runs in groups of `k`: after each group the codeword
//! folds by `2^k`, the successor is committed **before** the queries are drawn,
//! and each query checks that folding a current block really lands on the
//! successor's value there. The last group has no successor — the message is a
//! constant by then, and that constant is sent.
//!
//! Each round also draws an **out-of-domain** point `z0` and takes the
//! successor's value there. Nothing about the domain constrains that value, so
//! answering it is what forces the committed word to be near a *single*
//! codeword rather than merely near the code — the in-domain queries alone
//! leave a prover room to be near several. The claim is folded into the weight
//! the next group carries, batched with a challenge, which is why the whole
//! thing runs on the weighted form:
//!
//! ```text
//! w_{r+1} = w_r(alpha_r, ·) + gamma_r·eq(ood_r, ·)
//! C_{r+1} = C_r'            + gamma_r·y0_r
//! ```
//!
//! Three challenges per round are worth redrawing for a cheating prover — the
//! folding randomness, the out-of-domain batching challenge and the query
//! positions — so each is preceded by a [proof of
//! work](crypto::grinding): retrying one costs `2^bits` hashes. That is what
//! lets a query count buy more soundness than its own bits.
//!
//! The claim is the weighted one, `Σ_x w(x)·f(x) = y`, so a stacked multi-point
//! claim chains just as an evaluation does.

use crypto::fiat_shamir::is_transcript::IsTranscript;
use math::{
    field::{
        element::FieldElement,
        traits::{IsFFTField, IsField, IsPrimeField, IsSubFieldOf},
    },
    traits::AsBytes,
};

use crate::{
    Error,
    eq::{eq_eval, eq_mle},
    mle::Mle,
    poly::Composed,
    sumcheck::{self, RoundProof as SumcheckRoundProof},
    whir::{Domain, encode, fold_codeword_k, lift_coefficients},
    whir_commit::{Codeword, CodewordCommitment, Commitment, fold_coset, verify_opening},
    whir_hash::{GrindingDigest, WhirHash},
    whir_round::{self, RoundCommitments, RoundConfig, RoundProof},
};

/// `w(x)·f(x)`, the shape every group's sumcheck runs over.
fn product<E: IsField>(values: &[FieldElement<E>]) -> FieldElement<E> {
    &values[0] * &values[1]
}

/// The multilinear point where the univariate lift is `z0`.
///
/// [`lift_coefficients`] puts variable `j` on the monomial `X^(2^j)`, so
/// `F(z0) = f(z0, z0^2, z0^4, ..)`.
pub fn ood_point<E: IsField>(z0: &FieldElement<E>, num_vars: usize) -> Vec<FieldElement<E>> {
    let mut power = z0.clone();
    (0..num_vars)
        .map(|_| {
            let current = power.clone();
            power = power.square();
            current
        })
        .collect()
}

/// Rejects a point that landed inside `domain`, where it would constrain
/// nothing beyond the in-domain queries.
///
/// Sampling from the transcript makes this negligible; it is checked rather
/// than assumed.
fn require_out_of_domain<F, E>(z0: &FieldElement<E>, domain: &Domain<F>) -> Result<(), Error>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E>,
    E: IsField + Send + Sync + 'static,
{
    let mut power = z0.clone();
    for _ in 0..domain.log_size() {
        power = power.square();
    }
    if power == FieldElement::<E>::one() {
        return Err(Error::OodPointInDomain);
    }
    Ok(())
}

/// Proof of work over the transcript state, absorbed before the next challenge.
///
/// Retrying that challenge then costs `2^bits` hashes. Zero bits is a no-op, so
/// a caller that has not chosen its parameters yet pays nothing.
fn grind<E, T, H>(transcript: &mut T, bits: u8) -> Result<u64, Error>
where
    E: IsField + Send + Sync + 'static,
    T: IsTranscript<E>,
    H: WhirHash,
{
    if bits == 0 {
        return Ok(0);
    }
    let nonce =
        crypto::grinding::generate_nonce_maybe_gpu::<GrindingDigest<H>>(&transcript.state(), bits)
            .ok_or(Error::GrindingFailed { bits })?;
    transcript.append_bytes(&nonce.to_be_bytes());
    Ok(nonce)
}

/// The verifier's half: the nonce must pass against the same state, and it is
/// absorbed the same way.
fn check_grind<E, T, H>(transcript: &mut T, bits: u8, nonce: u64) -> Result<(), Error>
where
    E: IsField + Send + Sync + 'static,
    T: IsTranscript<E>,
    H: WhirHash,
{
    if bits == 0 {
        return Ok(());
    }
    if !crypto::grinding::is_valid_nonce::<GrindingDigest<H>>(&transcript.state(), nonce, bits) {
        return Err(Error::GrindingRejected { bits });
    }
    transcript.append_bytes(&nonce.to_be_bytes());
    Ok(())
}

/// `w + gamma·eq`, the weight the next group carries.
fn batch_weight<E: IsField + 'static>(
    w: &Mle<E>,
    eq: &Mle<E>,
    gamma: &FieldElement<E>,
) -> Result<Mle<E>, Error> {
    Mle::new(
        w.evals()
            .iter()
            .zip(eq.evals())
            .map(|(a, b)| a + gamma * b)
            .collect(),
    )
}

/// Proof-of-work bits, in the three places a prover would want to redraw.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GrindBits {
    /// Before a group's folding challenges.
    pub folding: u8,
    /// Before the out-of-domain batching challenge.
    pub ood: u8,
    /// Before the query positions.
    pub query: u8,
}

impl GrindBits {
    /// The same count in all three places.
    pub const fn uniform(bits: u8) -> Self {
        Self {
            folding: bits,
            ood: bits,
            query: bits,
        }
    }
}

/// Blowup, fold factor, query count and proof of work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChainConfig {
    /// `log2` of the code's inverse rate.
    pub log_blowup: usize,
    /// Variables folded per round. The last round takes whatever is left.
    pub log_folding: usize,
    /// Positions checked per round.
    pub num_queries: usize,
    /// Proof of work before each redrawable challenge.
    pub grind: GrindBits,
}

impl ChainConfig {
    /// Parameters for a security target, in the **same regime the univariate
    /// prover uses**: the Johnson bound, `proximity = 1 − √rate − 1/300`, so
    /// each query buys `−log2(1 − proximity)` bits. Grinding on the query phase
    /// is subtracted from what the queries have to buy, as it is there.
    ///
    /// Every round is spot-checked independently, so their errors add: the
    /// target carries `log2(rounds)` of margin on top (a union bound).
    ///
    /// **What this is not.** WHIR's own analysis is per-round and gets to use
    /// the out-of-domain point, which this ignores entirely. So it is the
    /// conservative mirror of the parameters the repo already ships — read it as
    /// "no weaker than the FRI ones next door" — and **not** a soundness proof
    /// about this protocol. Doing that analysis is what would let the query
    /// count come down.
    pub fn with_security(
        log_blowup: usize,
        log_folding: usize,
        num_vars: usize,
        security_bits: u8,
        grind: GrindBits,
    ) -> Self {
        let rounds = num_vars.div_ceil(log_folding.max(1)).max(1);
        // ★ Integers, not `f64`. The arithmetic and its provenance are in
        // [`crate::query_count`]; what matters here is that the count a
        // verifier has to reproduce no longer needs floating point to
        // reproduce it, and that the answers did not move — the shipped
        // posture's 110 / 112 / 113 are pinned in both places.
        let num_queries =
            crate::query_count::num_queries(log_blowup, rounds, security_bits, grind.query);

        Self {
            log_blowup,
            log_folding,
            num_queries,
            grind,
        }
    }

    /// Variables folded in each round: `log_folding` until the remainder.
    pub fn schedule(&self, num_vars: usize) -> Vec<usize> {
        let step = self.log_folding.max(1);
        let mut left = num_vars;
        let mut out = Vec::new();
        while left > 0 {
            let take = step.min(left);
            out.push(take);
            left -= take;
        }
        out
    }
}

/// A round's proof-of-work nonces.
///
/// Zero where the config asks for no bits, and the out-of-domain one is unused
/// on the last round.
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
#[serde(bound = "")]
pub struct RoundNonces {
    pub folding: u64,
    pub ood: u64,
    pub query: u64,
}

/// The blocks one round opens.
///
/// Only the first round's current codeword is base-field; every later one has
/// already been folded with an extension challenge. The successor is always in
/// the extension.
#[derive(
    Clone,
    Debug,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
#[serde(bound = "")]
pub enum RoundOpenings<F: IsField, E: IsField> {
    Base(RoundProof<F, E>),
    Extension(RoundProof<E, E>),
}

/// One round of the chain.
#[derive(
    Clone,
    Debug,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
#[serde(bound = "")]
pub struct ChainRound<F: IsField, E: IsField> {
    /// The sumcheck rounds this group consumed.
    pub sumcheck: Vec<SumcheckRoundProof<E>>,
    /// The successor's root. `None` on the last round, where the folded message
    /// is a constant and is sent instead.
    pub next_root: Option<Commitment>,
    /// The successor's value at the out-of-domain point. `None` on the last
    /// round: there is no successor left to be near a codeword.
    pub ood_value: Option<FieldElement<E>>,
    pub nonces: RoundNonces,
    pub openings: RoundOpenings<F, E>,
}

/// A chained proof that `Σ_x w(x)·f(x) = y`.
#[derive(
    Clone,
    Debug,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
#[serde(bound = "")]
pub struct ChainProof<F: IsField, E: IsField> {
    pub rounds: Vec<ChainRound<F, E>>,
    /// The constant the codeword folds to — the prover's claim for `f(α)`.
    pub final_value: FieldElement<E>,
}

impl<F: IsField, E: IsField> ChainProof<F, E> {
    /// Codeword elements the openings carry. What chaining is for.
    pub fn opened_elements(&self) -> usize {
        self.rounds
            .iter()
            .map(|r| match &r.openings {
                RoundOpenings::Base(p) => block_size(&p.current) + block_size::<E>(&p.next),
                RoundOpenings::Extension(p) => {
                    block_size::<E>(&p.current) + block_size::<E>(&p.next)
                }
            })
            .sum()
    }
}

fn block_size<V: IsField>(openings: &[crate::whir_commit::CosetOpening<V>]) -> usize {
    openings.iter().map(|c| c.values.len()).sum()
}

/// Commits `f`, blocked for the first round's fold.
///
/// The codeword stays in the base field: nothing extension-valued has touched
/// the trace yet, and this is the biggest allocation in the proof.
/// Commits one polynomial's codeword.
///
/// `transient` says whether this commitment has to promise the device room its
/// own commit and opening take — true when it stands alone, false when a
/// caller committing several has promised that turn once for all of them.
/// A polynomial to commit and open, which may not exist yet.
///
/// A stacked polynomial is its columns written at their offsets and zeros in
/// between. Assembling it here is a pass over every byte the device is about to
/// read anyway, so the parts are handed over as they are and whoever needs a
/// whole one builds it — which on the device path is nobody.
pub struct Stacked<'a, F: IsField> {
    /// `(column, offset in elements)`.
    pub parts: Vec<(&'a Mle<F>, usize)>,
    /// The same parts as `(index into the epoch's columns, offset)`, when the
    /// card already holds them — then the scatter is a copy at device
    /// bandwidth instead of the trace crossing the bus again.
    pub resident: Option<(&'a crate::gpu::ResidentColumns, Vec<(usize, usize)>)>,
    pub num_vars: usize,
}

impl<F: IsField + 'static> Stacked<'_, F> {
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// The polynomial itself, assembled here. Only the host paths ask — a
    /// device writes the parts where they go and never sees a whole one.
    pub fn assemble(&self) -> Result<Mle<F>, Error> {
        let mut buffer = vec![FieldElement::<F>::zero(); 1usize << self.num_vars];
        for (column, offset) in &self.parts {
            buffer[*offset..*offset + column.len()].clone_from_slice(column.evals());
        }
        Mle::new(buffer)
    }
}

/// A whole polynomial is a stacked one of a single part at offset zero.
pub fn commit<F, H>(
    f: &Mle<F>,
    config: &ChainConfig,
    transient: bool,
) -> Result<(CodewordCommitment<F, H>, Domain<F>), Error>
where
    F: IsFFTField + IsPrimeField + Send + Sync + 'static,
    H: WhirHash,
    FieldElement<F>: AsBytes + Sync + Send,
{
    commit_stacked(
        &Stacked {
            parts: vec![(f, 0)],
            resident: None,
            num_vars: f.num_vars(),
        },
        config,
        transient,
    )
}

pub fn commit_stacked<F, H>(
    f: &Stacked<'_, F>,
    config: &ChainConfig,
    transient: bool,
) -> Result<(CodewordCommitment<F, H>, Domain<F>), Error>
where
    F: IsFFTField + IsPrimeField + Send + Sync + 'static,
    H: WhirHash,
    FieldElement<F>: AsBytes + Sync + Send,
{
    let num_vars = f.num_vars();
    let schedule = config.schedule(num_vars);
    let first = schedule.first().copied().unwrap_or(0);
    let domain = Domain::<F>::new(num_vars + config.log_blowup)?;
    // On a device the codeword stays there: the chain folds it and opens a
    // handful of its values, and it is the biggest array the proof holds.
    let attempt = match &f.resident {
        Some((store, parts)) => crate::gpu::commit_resident(
            store,
            parts,
            num_vars,
            config.log_blowup,
            first,
            transient,
            H::DEVICE,
        ),
        None => crate::gpu::commit_parts(
            &f.parts,
            num_vars,
            config.log_blowup,
            first,
            transient,
            H::DEVICE,
        ),
    };
    let commitment = match attempt {
        Some((codeword, nodes)) => CodewordCommitment::from_device(codeword, nodes, first)?,
        // ⚠ COUNTED, because this arm is otherwise silent. The device declining
        // is not a slower path to the same place: the codeword is assembled,
        // lifted and encoded here, and the commitment then holds that codeword
        // AND its node array on the host until the proof ends. An epoch that
        // fills the card partway through lands here for every commit after,
        // and the only visible symptoms are host memory and a utilisation
        // figure — neither of which names the cause.
        None => {
            crate::gpu::note_host_fallback();
            CodewordCommitment::from_codeword(
                encode::<F, F>(&lift_coefficients(&f.assemble()?), &domain)?,
                first,
            )?
        }
    };
    Ok((commitment, domain))
}

/// The two tables a chain's rounds fold: the weight it carries and the message
/// it is opening.
///
/// A group of rounds folds them, the group after that reads what is left, and
/// they are each the width of a stacked polynomial — so on a device they stay
/// there and only the round's few field elements cross the bus. The flow
/// around them is the same either way: this is the only thing that differs.
enum Factors<F: IsField + IsSubFieldOf<E>, E: IsField> {
    Host {
        weight: Mle<E>,
        message: Mle<E>,
        field: core::marker::PhantomData<F>,
    },
    Device(crate::gpu::OpeningFactors),
}

impl<F, E> Factors<F, E>
where
    F: IsField + IsSubFieldOf<E> + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<E>: Send + Sync,
{
    /// The product of the two, which is what a round evaluates.
    fn program() -> Result<crate::program::Program<E>, Error> {
        let mut builder = crate::program::Builder::<E>::new();
        let weight = builder.var(0);
        let message = builder.var(1);
        let root = builder.mul(weight, message);
        builder.finish(root)
    }

    fn new(f: &Mle<F>, weight: Mle<E>) -> Result<Self, Error> {
        if let Some(device) = crate::gpu::open_on_device(f, &weight, &Self::program()?) {
            return Ok(Self::Device(device));
        }
        // The sumcheck's factors have to share a field, so the message is
        // lifted here. The codeword is not, which is where the size is.
        let message = Mle::new(
            f.evals()
                .iter()
                .map(|v| v.clone().to_extension::<E>())
                .collect(),
        )?;
        Ok(Self::Host {
            weight,
            message,
            field: core::marker::PhantomData,
        })
    }

    /// The same from a weight given as shares: on a device they are written
    /// straight into its buffer, and the host builds the table only if none
    /// takes them.
    fn from_shares(
        f: &Stacked<'_, F>,
        shares: &[crate::stacked_eval::WeightShare<'_, E>],
        n_stack: usize,
    ) -> Result<Self, Error> {
        if let Some(device) = crate::gpu::open_shared(f, shares, n_stack, &Self::program()?) {
            return Ok(Self::Device(device));
        }
        // Only here does a stacked polynomial have to exist on the host.
        Self::new(
            &f.assemble()?,
            crate::stacked_eval::weight_table(shares, n_stack)?,
        )
    }

    fn num_vars(&self) -> usize {
        match self {
            Self::Host { message, .. } => message.num_vars(),
            Self::Device(device) => device.num_vars(),
        }
    }

    /// One group of `rounds` rounds, leaving the factors folded on them.
    fn rounds<T: IsTranscript<E>>(
        &mut self,
        rounds: usize,
        transcript: &mut T,
    ) -> Result<sumcheck::RoundGroup<E>, Error> {
        match self {
            Self::Host {
                weight, message, ..
            } => {
                let polys = vec![
                    core::mem::replace(weight, Mle::new(vec![FieldElement::<E>::zero()])?),
                    core::mem::replace(message, Mle::new(vec![FieldElement::<E>::zero()])?),
                ];
                let mut poly = Composed::new(polys, product::<E>, 2)?;
                let out = sumcheck::prove_rounds(&mut poly, rounds, transcript)?;
                let mut parts = poly.into_polys();
                *message = parts.pop().expect("two factors");
                *weight = parts.pop().expect("two factors");
                Ok(out)
            }
            Self::Device(device) => {
                let (proofs, challenges, _) = device.rounds(rounds, 2, |evaluations| {
                    for value in evaluations {
                        transcript.append_field_element(value);
                    }
                    transcript.sample_field_element()
                })?;
                Ok((proofs, challenges))
            }
        }
    }

    fn evaluate_message(&self, point: &[FieldElement<E>]) -> Result<FieldElement<E>, Error> {
        match self {
            Self::Host { message, .. } => message.evaluate(point),
            Self::Device(device) => device.evaluate_message(point),
        }
    }

    /// `weight += gamma·eq(point, ·)`, the weight the next group carries.
    fn add_scaled_eq(
        &mut self,
        point: &[FieldElement<E>],
        gamma: &FieldElement<E>,
    ) -> Result<(), Error> {
        match self {
            Self::Host { weight, .. } => {
                *weight = batch_weight(weight, &eq_mle(point)?, gamma)?;
                Ok(())
            }
            Self::Device(device) => device.add_scaled_eq(point, gamma),
        }
    }
}

/// What a round folds: the codeword and the commitment that answers for it.
///
/// Only the first round's is base-field. Folding it with an extension challenge
/// is what lifts it, so every later round is `Extension`.
enum Current<'a, F: IsField + 'static, E: IsField + 'static, H: WhirHash>
where
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    Base(&'a CodewordCommitment<F, H>),
    Extension(CodewordCommitment<E, H>),
}

/// Proves `f(z) = y`.
pub fn prove<F, E, T, H>(
    f: &Mle<F>,
    z: &[FieldElement<E>],
    commitment: &CodewordCommitment<F, H>,
    domain: &Domain<F>,
    config: &ChainConfig,
    transcript: &mut T,
) -> Result<ChainProof<F, E>, Error>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    T: IsTranscript<E>,
    H: WhirHash,
{
    prove_weighted::<F, E, T, H>(f, eq_mle(z)?, commitment, domain, config, transcript)
}

/// The same for a weight given as the shares of a stacked polynomial's
/// columns, which a device writes into its own buffer and the host
/// materializes only if none does.
#[allow(clippy::too_many_arguments)]
pub fn prove_shared<F, E, T, H>(
    f: &Stacked<'_, F>,
    shares: &[crate::stacked_eval::WeightShare<'_, E>],
    n_stack: usize,
    commitment: &CodewordCommitment<F, H>,
    domain: &Domain<F>,
    config: &ChainConfig,
    transcript: &mut T,
) -> Result<ChainProof<F, E>, Error>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    T: IsTranscript<E>,
    H: WhirHash,
{
    let factors = Factors::<F, E>::from_shares(f, shares, n_stack)?;
    prove_with_factors::<F, E, T, H>(
        f.num_vars(),
        factors,
        commitment,
        domain,
        config,
        transcript,
    )
}

/// Proves `Σ_x w(x)·f(x) = y` for a weight the verifier can evaluate itself.
pub fn prove_weighted<F, E, T, H>(
    f: &Mle<F>,
    weight: Mle<E>,
    commitment: &CodewordCommitment<F, H>,
    domain: &Domain<F>,
    config: &ChainConfig,
    transcript: &mut T,
) -> Result<ChainProof<F, E>, Error>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    T: IsTranscript<E>,
    H: WhirHash,
{
    let factors = Factors::<F, E>::new(f, weight)?;
    prove_with_factors::<F, E, T, H>(
        f.num_vars(),
        factors,
        commitment,
        domain,
        config,
        transcript,
    )
}

/// The chain itself, over factors that are wherever they are.
fn prove_with_factors<F, E, T, H>(
    num_vars: usize,
    mut factors: Factors<F, E>,
    commitment: &CodewordCommitment<F, H>,
    domain: &Domain<F>,
    config: &ChainConfig,
    transcript: &mut T,
) -> Result<ChainProof<F, E>, Error>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    T: IsTranscript<E>,
    H: WhirHash,
{
    let schedule = config.schedule(num_vars);
    // The codeword comes out of the commitment rather than being encoded
    // again: it is the same array, and the NTT is not cheap.
    let mut current = Current::<F, E, H>::Base(commitment);
    let mut current_domain = domain.clone();

    let mut rounds = Vec::with_capacity(schedule.len());
    let mut final_value = FieldElement::<E>::zero();

    for (r, &k) in schedule.iter().enumerate() {
        let mut nonces = RoundNonces {
            folding: grind::<E, T, H>(transcript, config.grind.folding)?,
            ..RoundNonces::default()
        };

        let (sumcheck_rounds, alphas) = factors.rounds(k, transcript)?;

        // The fold lands in the extension whichever field it started in, so
        // the field is the only thing the two cases differ in — and a codeword
        // the device holds is folded where it is.
        let (folded, folded_domain) = match &current {
            Current::Base(held) => fold_held::<F, F, E>(held.codeword(), &current_domain, &alphas)?,
            Current::Extension(held) => {
                fold_held::<F, E, E>(held.codeword(), &current_domain, &alphas)?
            }
        };

        // The successor is committed before the queries are drawn, so it cannot
        // be chosen to match them.
        let next = match schedule.get(r + 1) {
            Some(&next_k) => {
                let next = commit_folded::<E, H>(folded, next_k)?;
                transcript.append_bytes(&next.root());
                Some(next)
            }
            None => {
                final_value = first_value::<E>(&folded)?;
                transcript.append_field_element(&final_value);
                None
            }
        };
        let next_root = next.as_ref().map(|c| c.root());

        // Out of domain: a value the queries cannot vouch for, so answering it
        // pins the successor to one codeword.
        let ood_value = if next.is_some() {
            let z0: FieldElement<E> = transcript.sample_field_element();
            require_out_of_domain::<F, E>(&z0, &folded_domain)?;
            let point = ood_point(&z0, factors.num_vars());
            let y0 = factors.evaluate_message(&point)?;
            transcript.append_field_element(&y0);

            nonces.ood = grind::<E, T, H>(transcript, config.grind.ood)?;
            let gamma: FieldElement<E> = transcript.sample_field_element();
            factors.add_scaled_eq(&point, &gamma)?;
            Some(y0)
        } else {
            None
        };

        nonces.query = grind::<E, T, H>(transcript, config.grind.query)?;
        let round_config = RoundConfig {
            num_queries: config.num_queries,
            log_folding: k,
        };
        let openings = match (&current, &next) {
            (Current::Base(held), Some(next)) => {
                RoundOpenings::Base(whir_round::prove(*held, next, &round_config, transcript)?)
            }
            (Current::Base(held), None) => RoundOpenings::Base(final_openings::<F, E, T, H>(
                held,
                &round_config,
                transcript,
            )?),
            (Current::Extension(held), Some(next)) => {
                RoundOpenings::Extension(whir_round::prove(held, next, &round_config, transcript)?)
            }
            (Current::Extension(held), None) => {
                RoundOpenings::Extension(final_openings::<E, E, T, H>(
                    held,
                    &round_config,
                    transcript,
                )?)
            }
        };

        rounds.push(ChainRound {
            sumcheck: sumcheck_rounds,
            next_root,
            ood_value,
            nonces,
            openings,
        });
        if let Some(next) = next {
            current = Current::Extension(next);
        }
        current_domain = folded_domain;
    }

    Ok(ChainProof {
        rounds,
        final_value,
    })
}

/// Folds a codeword wherever it is, leaving the result where it was.
fn fold_held<F, C, N>(
    codeword: &Codeword<C>,
    domain: &Domain<F>,
    alphas: &[FieldElement<N>],
) -> Result<(Codeword<N>, Domain<F>), Error>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<C> + IsSubFieldOf<N> + 'static,
    C: IsField + IsSubFieldOf<N> + 'static,
    N: IsField + 'static,
    FieldElement<C>: AsBytes + Sync + Send,
{
    if let Some(folded) = codeword
        .device()
        .and_then(|device| device.fold(domain.generator(), alphas))
    {
        let mut folded_domain = domain.clone();
        for _ in alphas {
            folded_domain = folded_domain.squared()?;
        }
        return Ok((Codeword::Device(folded), folded_domain));
    }
    // A codeword the device holds has no copy here: if the device turned the
    // fold down, this round cannot happen anywhere.
    let values = codeword.host().ok_or(Error::DeviceFailed {
        stage: "codeword fold",
    })?;
    let (folded, folded_domain) = fold_codeword_k::<F, C, N>(values, domain, alphas)?;
    Ok((Codeword::Host(folded), folded_domain))
}

/// Commits a folded codeword where it is.
fn commit_folded<N, H>(
    codeword: Codeword<N>,
    log_folding: usize,
) -> Result<CodewordCommitment<N, H>, Error>
where
    N: IsField + 'static,
    H: WhirHash,
    FieldElement<N>: AsBytes + Sync + Send,
{
    match codeword {
        Codeword::Host(values) => CodewordCommitment::from_codeword(values, log_folding),
        Codeword::Device(device) => {
            let nodes = device
                .commit(log_folding, H::DEVICE)
                .ok_or(Error::DeviceFailed { stage: "fold tree" })?;
            CodewordCommitment::from_device(device, nodes, log_folding)
        }
    }
}

/// The value the last fold leaves behind.
fn first_value<N>(codeword: &Codeword<N>) -> Result<FieldElement<N>, Error>
where
    N: IsField + 'static,
{
    match codeword {
        Codeword::Host(values) => values.first().cloned().ok_or(Error::EmptyPolynomial),
        Codeword::Device(device) => device.first().ok_or(Error::DeviceFailed {
            stage: "final value",
        }),
    }
}

/// The last round's openings: only the current codeword's blocks, since what
/// they fold to is the constant the prover sends.
///
/// Mirrors [`whir_round`]'s query draw, so both sides sample the same
/// positions.
fn final_openings<C, N, T, H>(
    current: &CodewordCommitment<C, H>,
    config: &RoundConfig,
    transcript: &mut T,
) -> Result<RoundProof<C, N>, Error>
where
    C: IsField + 'static,
    N: IsField,
    FieldElement<C>: AsBytes + Sync + Send,
    T: IsTranscript<N>,
    H: WhirHash,
{
    let queries: Vec<usize> = (0..config.num_queries)
        .map(|_| transcript.sample_u64(current.num_leaves() as u64) as usize)
        .collect();
    Ok(RoundProof {
        current: current.open_many(&queries)?,
        next: Vec::new(),
    })
}

/// Verifies `f(z) = y`.
pub fn verify<F, E, T, H>(
    proof: &ChainProof<F, E>,
    root: &Commitment,
    z: &[FieldElement<E>],
    y: FieldElement<E>,
    domain: &Domain<F>,
    config: &ChainConfig,
    transcript: &mut T,
) -> Result<(), Error>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    T: IsTranscript<E>,
    H: WhirHash,
{
    verify_weighted::<F, E, T, _, H>(
        proof,
        root,
        |alphas: &[FieldElement<E>]| eq_eval(z, alphas),
        y,
        z.len(),
        domain,
        config,
        transcript,
    )
}

/// Verifies `Σ_x w(x)·f(x) = y`.
///
/// `weight_at` is the weight's closed form, evaluated at the concatenation of
/// every round's challenges.
#[allow(clippy::too_many_arguments)]
pub fn verify_weighted<F, E, T, W, H>(
    proof: &ChainProof<F, E>,
    root: &Commitment,
    weight_at: W,
    y: FieldElement<E>,
    num_vars: usize,
    domain: &Domain<F>,
    config: &ChainConfig,
    transcript: &mut T,
) -> Result<(), Error>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    T: IsTranscript<E>,
    W: FnOnce(&[FieldElement<E>]) -> Result<FieldElement<E>, Error>,
    H: WhirHash,
{
    let schedule = config.schedule(num_vars);
    if proof.rounds.len() != schedule.len() {
        return Err(Error::RoundCountMismatch {
            expected: schedule.len(),
            got: proof.rounds.len(),
        });
    }

    let mut claim = y;
    let mut alphas: Vec<FieldElement<E>> = Vec::with_capacity(num_vars);
    let mut current_root = *root;
    let mut current_domain = domain.clone();
    // Each round's out-of-domain claim, and how many variables were bound when
    // it entered the weight — the challenges after that are where its `eq`
    // lands.
    let mut ood: Vec<(FieldElement<E>, Vec<FieldElement<E>>, usize)> = Vec::new();

    for (r, (round, &k)) in proof.rounds.iter().zip(&schedule).enumerate() {
        if round.sumcheck.len() != k {
            return Err(Error::RoundCountMismatch {
                expected: k,
                got: round.sumcheck.len(),
            });
        }
        // Only the first round's current codeword is base-field. Said here
        // rather than left to a Merkle check failing on the byte width.
        if (r == 0) != matches!(round.openings, RoundOpenings::Base(_)) {
            return Err(Error::RoundCountMismatch {
                expected: 0,
                got: r,
            });
        }
        check_grind::<E, T, H>(transcript, config.grind.folding, round.nonces.folding)?;
        // The weight raises the degree of the plain `f` term to two.
        let group = sumcheck::verify_rounds(&round.sumcheck, claim, 2, transcript)?;
        claim = group.expected_evaluation;

        let mut next_domain = current_domain.clone();
        for _ in 0..k {
            next_domain = next_domain.squared()?;
        }
        let bound = alphas.len() + k;

        let round_config = RoundConfig {
            num_queries: config.num_queries,
            log_folding: k,
        };
        match (
            &round.next_root,
            round.ood_value.as_ref(),
            schedule.get(r + 1),
        ) {
            (Some(next_root), Some(y0), Some(&next_k)) => {
                transcript.append_bytes(next_root);

                let z0: FieldElement<E> = transcript.sample_field_element();
                require_out_of_domain::<F, E>(&z0, &next_domain)?;
                let point = ood_point(&z0, num_vars - bound);
                transcript.append_field_element(y0);

                check_grind::<E, T, H>(transcript, config.grind.ood, round.nonces.ood)?;
                let gamma: FieldElement<E> = transcript.sample_field_element();
                claim += &gamma * y0;
                ood.push((gamma, point, bound));

                check_grind::<E, T, H>(transcript, config.grind.query, round.nonces.query)?;
                let commitments = RoundCommitments {
                    current_root: &current_root,
                    next_root,
                    next_num_leaves: next_domain.size() >> next_k,
                };
                match &round.openings {
                    RoundOpenings::Base(openings) => whir_round::verify::<F, F, E, T, H>(
                        openings,
                        commitments,
                        &current_domain,
                        &group.point,
                        &round_config,
                        transcript,
                    )?,
                    RoundOpenings::Extension(openings) => whir_round::verify::<F, E, E, T, H>(
                        openings,
                        commitments,
                        &current_domain,
                        &group.point,
                        &round_config,
                        transcript,
                    )?,
                }
                current_root = *next_root;
            }
            (None, None, None) => {
                transcript.append_field_element(&proof.final_value);
                check_grind::<E, T, H>(transcript, config.grind.query, round.nonces.query)?;
                match &round.openings {
                    RoundOpenings::Base(openings) => verify_final::<F, F, E, T, H>(
                        openings,
                        &current_root,
                        &current_domain,
                        &group.point,
                        &round_config,
                        &proof.final_value,
                        transcript,
                    )?,
                    RoundOpenings::Extension(openings) => verify_final::<F, E, E, T, H>(
                        openings,
                        &current_root,
                        &current_domain,
                        &group.point,
                        &round_config,
                        &proof.final_value,
                        transcript,
                    )?,
                }
            }
            // A successor or an out-of-domain value where the schedule ends, or
            // one missing where it does not.
            _ => {
                return Err(Error::RoundCountMismatch {
                    expected: schedule.len(),
                    got: r,
                });
            }
        }

        alphas.extend(group.point);
        current_domain = next_domain;
    }

    // The accumulated weight at the full challenge point: the caller's own,
    // plus each round's batched `eq` over the challenges that came after it.
    let mut weight = weight_at(&alphas)?;
    for (gamma, point, bound) in &ood {
        weight += gamma * eq_eval(point, &alphas[*bound..])?;
    }

    // The wire: the sumcheck's residual is w(α)·f(α), and the fully folded
    // codeword is f(α). Both must name the same value.
    let required = weight
        .inv()
        .ok()
        .map(|inv| claim * inv)
        .ok_or(Error::DegenerateEvaluationPoint)?;
    if proof.final_value != required {
        return Err(Error::EvaluationMismatch);
    }

    Ok(())
}

/// The last round: every queried block must fold to the constant that was sent.
fn verify_final<F, C, N, T, H>(
    openings: &RoundProof<C, N>,
    current_root: &Commitment,
    current_domain: &Domain<F>,
    alphas: &[FieldElement<N>],
    config: &RoundConfig,
    final_value: &FieldElement<N>,
    transcript: &mut T,
) -> Result<(), Error>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<C> + IsSubFieldOf<N>,
    C: IsField + IsSubFieldOf<N> + 'static,
    N: IsField + 'static,
    FieldElement<C>: AsBytes + Sync + Send,
    T: IsTranscript<N>,
    H: WhirHash,
{
    if openings.current.len() != config.num_queries || !openings.next.is_empty() {
        return Err(Error::QueryCountMismatch {
            expected: config.num_queries,
            got: openings.current.len(),
        });
    }
    let num_leaves = current_domain.size() >> config.log_folding;

    for (i, opening) in openings.current.iter().enumerate() {
        let q = transcript.sample_u64(num_leaves as u64) as usize;
        if !verify_opening::<C, H>(current_root, q, opening) {
            return Err(Error::OpeningRejected { query: i });
        }
        if fold_coset::<F, C, N>(&opening.values, current_domain, q, alphas)? != *final_value {
            return Err(Error::FoldInconsistent { query: i });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use math::field::goldilocks::GoldilocksField as F;

    use crate::{eq::eq_evals, whir_eval, whir_hash::KeccakWhir};

    type FE = FieldElement<F>;

    fn transcript() -> DefaultTranscript<F> {
        DefaultTranscript::<F>::new(b"whir-chain-test")
    }

    /// The current codeword's blocks, whichever variant holds them. In these
    /// tests `F` and `E` coincide, so both arms read the same.
    fn current_blocks(openings: &RoundOpenings<F, F>) -> &[crate::whir_commit::CosetOpening<F>] {
        match openings {
            RoundOpenings::Base(p) => &p.current,
            RoundOpenings::Extension(p) => &p.current,
        }
    }

    fn current_blocks_mut(
        openings: &mut RoundOpenings<F, F>,
    ) -> &mut [crate::whir_commit::CosetOpening<F>] {
        match openings {
            RoundOpenings::Base(p) => &mut p.current,
            RoundOpenings::Extension(p) => &mut p.current,
        }
    }

    fn config(log_folding: usize) -> ChainConfig {
        ChainConfig {
            log_blowup: 2,
            log_folding,
            num_queries: 3,
            grind: GrindBits::default(),
        }
    }

    /// Deterministic pseudo-random values; the seed is mixed before the shift so
    /// nearby seeds cannot collapse to the same polynomial.
    fn pseudo_mle(num_vars: usize, seed: u64) -> Mle<F> {
        let vals: Vec<FE> = (0..(1u64 << num_vars))
            .map(|i| {
                let mixed = i
                    .wrapping_add(seed)
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
                FE::from(mixed >> 13)
            })
            .collect();
        Mle::new(vals).unwrap()
    }

    fn point(num_vars: usize) -> Vec<FE> {
        (0..num_vars).map(|i| FE::from(101 + i as u64)).collect()
    }

    fn run(num_vars: usize, log_folding: usize) -> Result<ChainProof<F, F>, Error> {
        let cfg = config(log_folding);
        let f = pseudo_mle(num_vars, 11);
        let z = point(num_vars);
        let y = f.evaluate(&z).unwrap();

        let (commitment, domain) = commit::<F, KeccakWhir>(&f, &cfg, true)?;
        let proof =
            prove::<F, F, _, KeccakWhir>(&f, &z, &commitment, &domain, &cfg, &mut transcript())?;
        verify::<F, F, _, KeccakWhir>(
            &proof,
            &commitment.root(),
            &z,
            y,
            &domain,
            &cfg,
            &mut transcript(),
        )?;
        Ok(proof)
    }

    #[test]
    fn an_honest_evaluation_verifies() {
        for num_vars in 1..=6usize {
            for log_folding in 1..=3usize {
                run(num_vars, log_folding)
                    .unwrap_or_else(|e| panic!("num_vars={num_vars}, k={log_folding}: {e:?}"));
            }
        }
    }

    #[test]
    fn the_query_count_follows_the_johnson_bound() {
        // Blowup 4, 128 bits, 20 bits of query grinding, one round: the same
        // 110 the univariate prover's own accounting gives.
        let cfg = ChainConfig::with_security(
            2,
            4,
            4,
            128,
            GrindBits {
                query: 20,
                ..GrindBits::default()
            },
        );
        assert_eq!(cfg.num_queries, 110);

        // A wider blowup buys more per query.
        let wider = ChainConfig::with_security(
            3,
            4,
            4,
            128,
            GrindBits {
                query: 20,
                ..GrindBits::default()
            },
        );
        assert!(wider.num_queries < cfg.num_queries);

        // More rounds cost a union bound, and grinding takes queries off.
        let deeper = ChainConfig::with_security(2, 4, 64, 128, GrindBits::default());
        let ground = ChainConfig::with_security(2, 4, 64, 128, GrindBits::uniform(20));
        assert!(deeper.num_queries > cfg.num_queries);
        assert!(ground.num_queries < deeper.num_queries);
    }

    /// And the parameters it picks actually run.
    #[test]
    fn a_proof_at_the_chosen_parameters_verifies() {
        let num_vars = 5;
        // Small bits so the grind is instant; the query count is the real one.
        let cfg = ChainConfig::with_security(2, 2, num_vars, 32, GrindBits::uniform(8));
        assert!(cfg.num_queries > 10);

        let (proof, root, domain, y) = prove_ground(num_vars, &cfg);
        verify_ground(&proof, &root, &domain, y, &cfg, num_vars).unwrap();
    }

    #[test]
    fn the_schedule_takes_the_remainder_last() {
        assert_eq!(config(3).schedule(10), vec![3, 3, 3, 1]);
        assert_eq!(config(3).schedule(9), vec![3, 3, 3]);
        assert_eq!(config(4).schedule(3), vec![3]);
        assert_eq!(config(1).schedule(3), vec![1, 1, 1]);
    }

    /// The point of chaining: a query opens a block of `2^k`, not the message.
    #[test]
    fn a_block_is_the_fold_size_not_the_message() {
        let num_vars = 6;
        let log_folding = 2;
        let proof = run(num_vars, log_folding).unwrap();

        assert_eq!(proof.rounds.len(), 3);
        for round in &proof.rounds {
            for opening in current_blocks(&round.openings) {
                assert_eq!(opening.values.len(), 1 << log_folding);
            }
        }
    }

    /// And what that buys, against the one-round argument on the same
    /// polynomial.
    #[test]
    fn chaining_opens_far_less_than_one_round_does() {
        let num_vars = 8;
        let cfg = config(2);
        let f = pseudo_mle(num_vars, 13);
        let z = point(num_vars);

        let (commitment, domain) = commit::<F, KeccakWhir>(&f, &cfg, true).unwrap();
        let chained =
            prove::<F, F, _, KeccakWhir>(&f, &z, &commitment, &domain, &cfg, &mut transcript())
                .unwrap();

        let one_round_cfg = whir_eval::EvalConfig {
            log_blowup: cfg.log_blowup,
            num_queries: cfg.num_queries,
        };
        let (one_commitment, one_domain) =
            whir_eval::commit::<F, F, KeccakWhir>(&f, &one_round_cfg).unwrap();
        let one_round = whir_eval::prove::<F, F, _, KeccakWhir>(
            &f,
            &z,
            &one_commitment,
            &one_domain,
            &one_round_cfg,
            &mut transcript(),
        )
        .unwrap();

        let one_round_elements: usize = one_round.openings.iter().map(|o| o.values.len()).sum();
        // One round sends `num_queries` blocks of the whole message; chaining
        // sends small blocks, two per query per round.
        assert_eq!(one_round_elements, 3 * (1 << num_vars));
        assert!(
            chained.opened_elements() * 4 < one_round_elements,
            "chained {} vs one round {one_round_elements}",
            chained.opened_elements()
        );
    }

    #[test]
    fn a_false_evaluation_is_rejected() {
        let cfg = config(2);
        let num_vars = 5;
        let f = pseudo_mle(num_vars, 17);
        let z = point(num_vars);
        let y = f.evaluate(&z).unwrap();

        let (commitment, domain) = commit::<F, KeccakWhir>(&f, &cfg, true).unwrap();
        let proof =
            prove::<F, F, _, KeccakWhir>(&f, &z, &commitment, &domain, &cfg, &mut transcript())
                .unwrap();

        assert!(
            verify::<F, F, _, KeccakWhir>(
                &proof,
                &commitment.root(),
                &z,
                y + FE::one(),
                &domain,
                &cfg,
                &mut transcript(),
            )
            .is_err()
        );
    }

    #[test]
    fn a_forged_final_value_is_rejected() {
        let cfg = config(2);
        let num_vars = 5;
        let f = pseudo_mle(num_vars, 19);
        let z = point(num_vars);
        let y = f.evaluate(&z).unwrap();

        let (commitment, domain) = commit::<F, KeccakWhir>(&f, &cfg, true).unwrap();
        let mut proof =
            prove::<F, F, _, KeccakWhir>(&f, &z, &commitment, &domain, &cfg, &mut transcript())
                .unwrap();
        proof.final_value += FE::one();

        assert!(
            verify::<F, F, _, KeccakWhir>(
                &proof,
                &commitment.root(),
                &z,
                y,
                &domain,
                &cfg,
                &mut transcript()
            )
            .is_err()
        );
    }

    #[test]
    fn a_tampered_opening_in_a_middle_round_is_rejected() {
        let cfg = config(2);
        let num_vars = 6;
        let f = pseudo_mle(num_vars, 23);
        let z = point(num_vars);
        let y = f.evaluate(&z).unwrap();

        let (commitment, domain) = commit::<F, KeccakWhir>(&f, &cfg, true).unwrap();
        let mut proof =
            prove::<F, F, _, KeccakWhir>(&f, &z, &commitment, &domain, &cfg, &mut transcript())
                .unwrap();
        assert!(proof.rounds.len() >= 3);
        current_blocks_mut(&mut proof.rounds[1].openings)[0].values[0] += FE::one();

        let err = verify::<F, F, _, KeccakWhir>(
            &proof,
            &commitment.root(),
            &z,
            y,
            &domain,
            &cfg,
            &mut transcript(),
        )
        .unwrap_err();
        assert!(matches!(err, Error::OpeningRejected { .. }));
    }

    /// The first round's blocks are hashed as base-field bytes, so tampering
    /// with one has to be caught by that hash and not by a later fold.
    #[test]
    fn a_tampered_opening_in_the_base_round_is_rejected() {
        let cfg = config(2);
        let num_vars = 6;
        let f = pseudo_mle(num_vars, 79);
        let z = point(num_vars);
        let y = f.evaluate(&z).unwrap();

        let (commitment, domain) = commit::<F, KeccakWhir>(&f, &cfg, true).unwrap();
        let mut proof =
            prove::<F, F, _, KeccakWhir>(&f, &z, &commitment, &domain, &cfg, &mut transcript())
                .unwrap();
        assert!(matches!(proof.rounds[0].openings, RoundOpenings::Base(_)));
        current_blocks_mut(&mut proof.rounds[0].openings)[0].values[0] += FE::one();

        let err = verify::<F, F, _, KeccakWhir>(
            &proof,
            &commitment.root(),
            &z,
            y,
            &domain,
            &cfg,
            &mut transcript(),
        )
        .unwrap_err();
        assert!(matches!(err, Error::OpeningRejected { .. }));
    }

    /// A prover who commits one polynomial and argues about another: the fold
    /// check between consecutive codewords is what catches it.
    #[test]
    fn a_proof_about_a_different_polynomial_is_rejected() {
        let cfg = config(2);
        let num_vars = 5;
        let f = pseudo_mle(num_vars, 29);
        let g = pseudo_mle(num_vars, 31);
        let z = point(num_vars);

        let (f_commitment, domain) = commit::<F, KeccakWhir>(&f, &cfg, true).unwrap();
        let proof =
            prove::<F, F, _, KeccakWhir>(&g, &z, &f_commitment, &domain, &cfg, &mut transcript())
                .unwrap();

        let err = verify::<F, F, _, KeccakWhir>(
            &proof,
            &f_commitment.root(),
            &z,
            g.evaluate(&z).unwrap(),
            &domain,
            &cfg,
            &mut transcript(),
        )
        .unwrap_err();
        // The codeword now comes out of the commitment, so a prover cannot be
        // inconsistent between the two: the lie has nowhere to go but the wire
        // between the sumcheck and the folded value.
        assert_eq!(err, Error::EvaluationMismatch);
    }

    #[test]
    fn a_round_missing_its_successor_is_rejected() {
        let cfg = config(2);
        let num_vars = 6;
        let f = pseudo_mle(num_vars, 37);
        let z = point(num_vars);
        let y = f.evaluate(&z).unwrap();

        let (commitment, domain) = commit::<F, KeccakWhir>(&f, &cfg, true).unwrap();
        let mut proof =
            prove::<F, F, _, KeccakWhir>(&f, &z, &commitment, &domain, &cfg, &mut transcript())
                .unwrap();
        proof.rounds[0].next_root = None;

        assert!(
            verify::<F, F, _, KeccakWhir>(
                &proof,
                &commitment.root(),
                &z,
                y,
                &domain,
                &cfg,
                &mut transcript()
            )
            .is_err()
        );
    }

    #[test]
    fn a_proof_replayed_under_another_transcript_is_rejected() {
        let cfg = config(2);
        let num_vars = 5;
        let f = pseudo_mle(num_vars, 41);
        let z = point(num_vars);
        let y = f.evaluate(&z).unwrap();

        let (commitment, domain) = commit::<F, KeccakWhir>(&f, &cfg, true).unwrap();
        let proof =
            prove::<F, F, _, KeccakWhir>(&f, &z, &commitment, &domain, &cfg, &mut transcript())
                .unwrap();

        let mut other = DefaultTranscript::<F>::new(b"a-different-statement");
        assert!(
            verify::<F, F, _, KeccakWhir>(
                &proof,
                &commitment.root(),
                &z,
                y,
                &domain,
                &cfg,
                &mut other
            )
            .is_err()
        );
    }

    /// A weight that is not an `eq`: the shape a stacked multi-point claim
    /// takes, chained the same way.
    #[test]
    fn a_two_point_weighted_claim_chains() {
        let cfg = config(2);
        let num_vars = 5;
        let f = pseudo_mle(num_vars, 43);
        let a = point(num_vars);
        let b: Vec<FE> = a.iter().map(|x| x + FE::from(7)).collect();
        let gamma = FE::from(5);

        // w = eq(a, ·) + gamma·eq(b, ·)
        let table: Vec<FE> = eq_evals(&a)
            .into_iter()
            .zip(eq_evals(&b))
            .map(|(p, q)| p + gamma * q)
            .collect();
        let weight = Mle::new(table).unwrap();
        let y = f.evaluate(&a).unwrap() + gamma * f.evaluate(&b).unwrap();

        let (commitment, domain) = commit::<F, KeccakWhir>(&f, &cfg, true).unwrap();
        let proof = prove_weighted::<F, F, _, KeccakWhir>(
            &f,
            weight,
            &commitment,
            &domain,
            &cfg,
            &mut transcript(),
        )
        .unwrap();

        verify_weighted::<F, F, _, _, KeccakWhir>(
            &proof,
            &commitment.root(),
            |at: &[FE]| Ok(eq_eval(&a, at)? + gamma * eq_eval(&b, at)?),
            y,
            num_vars,
            &domain,
            &cfg,
            &mut transcript(),
        )
        .unwrap();
    }

    /// The committed codeword is base-field, so only the first round's blocks
    /// are — that is where the memory goes, and the type says which round is
    /// which rather than leaving it implicit.
    #[test]
    fn only_the_first_round_opens_base_field_blocks() {
        let proof = run(6, 2).unwrap();
        assert!(proof.rounds.len() >= 3);
        assert!(matches!(proof.rounds[0].openings, RoundOpenings::Base(_)));
        for round in &proof.rounds[1..] {
            assert!(matches!(round.openings, RoundOpenings::Extension(_)));
        }
    }

    /// And the verifier says so: a first round presenting extension blocks is
    /// refused outright, rather than left to fail on a byte-width mismatch
    /// inside a hash.
    #[test]
    fn a_first_round_claiming_extension_blocks_is_rejected() {
        let cfg = config(2);
        let num_vars = 6;
        let f = pseudo_mle(num_vars, 73);
        let z = point(num_vars);
        let y = f.evaluate(&z).unwrap();

        let (commitment, domain) = commit::<F, KeccakWhir>(&f, &cfg, true).unwrap();
        let mut proof =
            prove::<F, F, _, KeccakWhir>(&f, &z, &commitment, &domain, &cfg, &mut transcript())
                .unwrap();

        // `F` and `E` coincide here, so the same blocks fit the other variant.
        if let RoundOpenings::Base(openings) = proof.rounds[0].openings.clone() {
            proof.rounds[0].openings = RoundOpenings::Extension(openings);
        }

        assert!(
            verify::<F, F, _, KeccakWhir>(
                &proof,
                &commitment.root(),
                &z,
                y,
                &domain,
                &cfg,
                &mut transcript()
            )
            .is_err()
        );
    }

    // ---------------------------------------------------------------
    // Out of domain.
    // ---------------------------------------------------------------

    /// The convention the out-of-domain claim rests on: the multilinear point
    /// `(z, z^2, z^4, ..)` is where the univariate lift is `z`. Pinned against
    /// the codeword itself, at every domain point.
    #[test]
    fn the_out_of_domain_point_is_where_the_lift_is_sampled() {
        use crate::whir::{encode, lift_coefficients};

        let num_vars = 4;
        let f = pseudo_mle(num_vars, 53);
        let domain = Domain::<F>::new(num_vars + 2).unwrap();
        let codeword = encode::<F, F>(&lift_coefficients(&f), &domain).unwrap();

        let mut x = FE::one();
        for value in &codeword {
            assert_eq!(*value, f.evaluate(&ood_point(&x, num_vars)).unwrap());
            x *= domain.generator();
        }
    }

    /// And the property the whole round rests on: the folded codeword encodes
    /// the message the sumcheck folded, in the same variable order.
    #[test]
    fn the_folded_codeword_encodes_the_folded_message() {
        use crate::whir::{encode, lift_coefficients};

        let num_vars = 4;
        let f = pseudo_mle(num_vars, 59);
        let domain = Domain::<F>::new(num_vars + 2).unwrap();
        let codeword = encode::<F, F>(&lift_coefficients(&f), &domain).unwrap();

        let alphas = [FE::from(3), FE::from(5)];
        let (folded, folded_domain) =
            crate::whir::fold_codeword_k::<F, F, F>(&codeword, &domain, &alphas).unwrap();

        let mut message = f;
        for alpha in &alphas {
            message.fix_first_variable_in_place(alpha).unwrap();
        }
        let re_encoded = encode::<F, F>(&lift_coefficients(&message), &folded_domain).unwrap();
        assert_eq!(folded, re_encoded);
    }

    #[test]
    fn every_round_but_the_last_answers_out_of_domain() {
        let proof = run(6, 2).unwrap();
        assert_eq!(proof.rounds.len(), 3);
        for round in &proof.rounds[..2] {
            assert!(round.ood_value.is_some());
        }
        assert!(proof.rounds[2].ood_value.is_none());
    }

    /// The out-of-domain value is batched into the next group's claim, so
    /// getting it wrong breaks the chain.
    #[test]
    fn a_forged_out_of_domain_value_is_rejected() {
        let cfg = config(2);
        let num_vars = 6;
        let f = pseudo_mle(num_vars, 61);
        let z = point(num_vars);
        let y = f.evaluate(&z).unwrap();

        let (commitment, domain) = commit::<F, KeccakWhir>(&f, &cfg, true).unwrap();
        let mut proof =
            prove::<F, F, _, KeccakWhir>(&f, &z, &commitment, &domain, &cfg, &mut transcript())
                .unwrap();
        proof.rounds[0].ood_value = Some(proof.rounds[0].ood_value.unwrap() + FE::one());

        assert!(
            verify::<F, F, _, KeccakWhir>(
                &proof,
                &commitment.root(),
                &z,
                y,
                &domain,
                &cfg,
                &mut transcript()
            )
            .is_err()
        );
    }

    #[test]
    fn a_round_missing_its_out_of_domain_value_is_rejected() {
        let cfg = config(2);
        let num_vars = 6;
        let f = pseudo_mle(num_vars, 67);
        let z = point(num_vars);
        let y = f.evaluate(&z).unwrap();

        let (commitment, domain) = commit::<F, KeccakWhir>(&f, &cfg, true).unwrap();
        let mut proof =
            prove::<F, F, _, KeccakWhir>(&f, &z, &commitment, &domain, &cfg, &mut transcript())
                .unwrap();
        proof.rounds[0].ood_value = None;

        assert!(
            verify::<F, F, _, KeccakWhir>(
                &proof,
                &commitment.root(),
                &z,
                y,
                &domain,
                &cfg,
                &mut transcript()
            )
            .is_err()
        );
    }

    #[test]
    fn a_point_inside_the_domain_is_refused() {
        // The generator of the folded domain is in it, so it must be rejected;
        // a point outside must not be.
        let domain = Domain::<F>::new(5).unwrap();
        assert_eq!(
            require_out_of_domain::<F, F>(domain.generator(), &domain).err(),
            Some(Error::OodPointInDomain)
        );
        assert!(require_out_of_domain::<F, F>(&FE::from(7), &domain).is_ok());
    }

    // ---------------------------------------------------------------
    // Proof of work.
    // ---------------------------------------------------------------

    /// Enough bits to be a real search, few enough to be instant.
    fn ground(log_folding: usize) -> ChainConfig {
        ChainConfig {
            grind: GrindBits::uniform(8),
            ..config(log_folding)
        }
    }

    fn prove_ground(
        num_vars: usize,
        cfg: &ChainConfig,
    ) -> (ChainProof<F, F>, Commitment, Domain<F>, FE) {
        let f = pseudo_mle(num_vars, 71);
        let z = point(num_vars);
        let y = f.evaluate(&z).unwrap();
        let (commitment, domain) = commit::<F, KeccakWhir>(&f, cfg, true).unwrap();
        let proof =
            prove::<F, F, _, KeccakWhir>(&f, &z, &commitment, &domain, cfg, &mut transcript())
                .unwrap();
        (proof, commitment.root(), domain, y)
    }

    fn verify_ground(
        proof: &ChainProof<F, F>,
        root: &Commitment,
        domain: &Domain<F>,
        y: FE,
        cfg: &ChainConfig,
        num_vars: usize,
    ) -> Result<(), Error> {
        verify::<F, F, _, KeccakWhir>(
            proof,
            root,
            &point(num_vars),
            y,
            domain,
            cfg,
            &mut transcript(),
        )
    }

    #[test]
    fn a_ground_proof_verifies() {
        let num_vars = 6;
        let cfg = ground(2);
        let (proof, root, domain, y) = prove_ground(num_vars, &cfg);

        // Every round grinds before its folding challenges and its queries, and
        // before the out-of-domain challenge where there is one.
        for round in &proof.rounds {
            assert_ne!(round.nonces.folding, 0);
            assert_ne!(round.nonces.query, 0);
        }
        assert_ne!(proof.rounds[0].nonces.ood, 0);
        assert_eq!(proof.rounds.last().unwrap().nonces.ood, 0);

        verify_ground(&proof, &root, &domain, y, &cfg, num_vars).unwrap();
    }

    #[test]
    fn zero_bits_grinds_nothing() {
        let proof = run(6, 2).unwrap();
        for round in &proof.rounds {
            assert_eq!(round.nonces, RoundNonces::default());
        }
    }

    #[test]
    fn a_forged_folding_nonce_is_rejected() {
        let num_vars = 6;
        let cfg = ground(2);
        let (mut proof, root, domain, y) = prove_ground(num_vars, &cfg);
        proof.rounds[1].nonces.folding += 1;

        assert_eq!(
            verify_ground(&proof, &root, &domain, y, &cfg, num_vars).unwrap_err(),
            Error::GrindingRejected { bits: 8 }
        );
    }

    #[test]
    fn a_forged_out_of_domain_nonce_is_rejected() {
        let num_vars = 6;
        let cfg = ground(2);
        let (mut proof, root, domain, y) = prove_ground(num_vars, &cfg);
        proof.rounds[0].nonces.ood += 1;

        assert_eq!(
            verify_ground(&proof, &root, &domain, y, &cfg, num_vars).unwrap_err(),
            Error::GrindingRejected { bits: 8 }
        );
    }

    #[test]
    fn a_forged_query_nonce_is_rejected() {
        let num_vars = 6;
        let cfg = ground(2);
        let (mut proof, root, domain, y) = prove_ground(num_vars, &cfg);
        proof.rounds[0].nonces.query += 1;

        assert_eq!(
            verify_ground(&proof, &root, &domain, y, &cfg, num_vars).unwrap_err(),
            Error::GrindingRejected { bits: 8 }
        );
    }

    /// A verifier asking for bits the prover did not spend.
    #[test]
    fn a_proof_ground_for_fewer_bits_is_rejected() {
        let num_vars = 6;
        let (proof, root, domain, y) = prove_ground(num_vars, &ground(2));
        let demanding = ChainConfig {
            grind: GrindBits::uniform(30),
            ..config(2)
        };

        assert!(verify_ground(&proof, &root, &domain, y, &demanding, num_vars).is_err());
    }

    #[test]
    fn the_bits_can_differ_per_place() {
        let num_vars = 5;
        let cfg = ChainConfig {
            grind: GrindBits {
                folding: 6,
                ood: 0,
                query: 8,
            },
            ..config(2)
        };
        let (proof, root, domain, y) = prove_ground(num_vars, &cfg);
        assert_eq!(proof.rounds[0].nonces.ood, 0);
        verify_ground(&proof, &root, &domain, y, &cfg, num_vars).unwrap();
    }

    /// The field tower: a base-field domain with extension-valued codewords.
    #[test]
    fn the_chain_runs_over_a_field_tower() {
        use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext;
        type ExtE = FieldElement<Ext>;

        let cfg = config(2);
        let num_vars = 5;
        // The committed polynomial is base-field; only the point is not.
        let f = Mle::new(
            (0..(1u64 << num_vars))
                .map(|i| FE::from(i * 7 + 3))
                .collect(),
        )
        .unwrap();
        let z: Vec<ExtE> = (0..num_vars).map(|i| ExtE::from(101 + i as u64)).collect();
        let y = f.evaluate_in(&z).unwrap();

        let (commitment, domain) = commit::<F, KeccakWhir>(&f, &cfg, true).unwrap();
        let mut prover = DefaultTranscript::<Ext>::new(b"tower");
        let proof = prove::<F, Ext, _, KeccakWhir>(&f, &z, &commitment, &domain, &cfg, &mut prover)
            .unwrap();

        let mut verifier = DefaultTranscript::<Ext>::new(b"tower");
        verify::<F, Ext, _, KeccakWhir>(
            &proof,
            &commitment.root(),
            &z,
            y,
            &domain,
            &cfg,
            &mut verifier,
        )
        .unwrap();
    }
}
