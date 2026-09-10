//! Encoding a multilinear as a Reed-Solomon codeword, and folding it.
//!
//! The domain is a two-adic subgroup, so it lives in the base field `F`, while
//! codeword values live in `E`. Mixed products keep the base element on the
//! left, which is the only direction the field tower implements.
//!
//! The identity everything rests on: folding the codeword with `α` encodes the
//! polynomial with one variable fixed to `α`. [`lift_coefficients`] reverses the
//! coefficient index so that variable is the *first*, matching sumcheck.

use math::field::{
    element::FieldElement,
    traits::{IsFFTField, IsField, IsPrimeField, IsSubFieldOf},
};

#[cfg(not(feature = "parallel"))]
use math::fft::bowers_fft::bowers_fft_opt_fused;
#[cfg(feature = "parallel")]
use math::fft::bowers_fft::bowers_fft_opt_fused_parallel;
use math::fft::{bit_reversing::in_place_bit_reverse_permute, bowers_fft::LayerTwiddles};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::{Error, mle::Mle};

/// The evaluation domain: a coset-free multiplicative subgroup of order `2^k`.
#[derive(Clone, Debug)]
pub struct Domain<F: IsField> {
    generator: FieldElement<F>,
    log_size: usize,
}

impl<F: IsFFTField + IsPrimeField> Domain<F> {
    /// The subgroup of order `2^log_size`.
    pub fn new(log_size: usize) -> Result<Self, Error> {
        let generator = F::get_primitive_root_of_unity(log_size as u64).map_err(|_| {
            Error::SkipDomainUnavailable {
                l_skip: log_size,
                two_adicity: F::TWO_ADICITY as usize,
            }
        })?;
        Ok(Self {
            generator,
            log_size,
        })
    }

    /// The domain reached by squaring every element: order `2^(log_size - 1)`.
    pub fn squared(&self) -> Result<Self, Error> {
        if self.log_size == 0 {
            return Err(Error::NoVariablesLeft);
        }
        Ok(Self {
            generator: self.generator.square(),
            log_size: self.log_size - 1,
        })
    }

    pub fn log_size(&self) -> usize {
        self.log_size
    }

    pub fn size(&self) -> usize {
        1usize << self.log_size
    }

    pub fn generator(&self) -> &FieldElement<F> {
        &self.generator
    }

    /// Every element, as `1, g, g², …`.
    pub fn elements(&self) -> Vec<FieldElement<F>> {
        let mut out = Vec::with_capacity(self.size());
        let mut current = FieldElement::<F>::one();
        for _ in 0..self.size() {
            out.push(current.clone());
            current *= &self.generator;
        }
        out
    }
}

/// The multilinear's monomial coefficients, in hypercube index order.
///
/// The inverse of the evaluation map: reading a multilinear's `2^m` hypercube
/// values as `Σ_S ĉ_S ∏_{i∈S} x_i`. Computed by the Möbius transform, in
/// `O(m·2^m)`.
pub fn monomial_coefficients<F: IsField>(mle: &Mle<F>) -> Vec<FieldElement<F>> {
    let mut coeffs = mle.evals().to_vec();
    let n = coeffs.len();
    let mut stride = 1;
    while stride < n {
        let mut start = 0;
        while start < n {
            for i in start..start + stride {
                let lo = coeffs[i].clone();
                coeffs[i + stride] = &coeffs[i + stride] - &lo;
            }
            start += stride * 2;
        }
        stride *= 2;
    }
    coeffs
}

/// The univariate lift used by the folding argument.
///
/// Reverses the coefficient index so variable 0 is the low bit, making one fold
/// bind the variable one sumcheck round binds.
pub fn lift_coefficients<F: IsField>(mle: &Mle<F>) -> Vec<FieldElement<F>> {
    let coeffs = monomial_coefficients(mle);
    let num_vars = mle.num_vars();
    (0..coeffs.len())
        .map(|i| coeffs[reverse_bits(i, num_vars)].clone())
        .collect()
}

/// Reverses the low `width` bits of `index`.
fn reverse_bits(index: usize, width: usize) -> usize {
    (0..width).fold(0, |acc, i| acc | (((index >> i) & 1) << (width - 1 - i)))
}

/// Evaluates the univariate lift on every point of `domain`, in domain order:
/// `out[j] = F(g^j)`.
///
/// One NTT, threaded under the `parallel` feature. The Bowers transform leaves
/// its output bit-reversed, and the ordering is load-bearing —
/// [`fold_codeword`] pairs `j` with `j + N/2` because `g^(j + N/2) = −g^j` — so
/// it is permuted back.
///
/// The `Send + Sync` bounds are unconditional so the signature does not change
/// with the feature; the serial path does not need them, and every field this
/// crate is used with satisfies them.
pub fn encode<F, E>(
    coeffs: &[FieldElement<E>],
    domain: &Domain<F>,
) -> Result<Vec<FieldElement<E>>, Error>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E>,
    E: IsField + Send + Sync,
    FieldElement<F>: Send + Sync,
    FieldElement<E>: Send + Sync,
{
    if coeffs.len() > domain.size() {
        return Err(Error::CodewordTooShort {
            coefficients: coeffs.len(),
            domain: domain.size(),
        });
    }
    let mut values = coeffs.to_vec();
    values.resize(domain.size(), FieldElement::<E>::zero());
    if domain.log_size == 0 {
        return Ok(values);
    }

    let twiddles =
        LayerTwiddles::<F>::new(domain.log_size as u64).ok_or(Error::SkipDomainUnavailable {
            l_skip: domain.log_size,
            two_adicity: F::TWO_ADICITY as usize,
        })?;
    #[cfg(feature = "parallel")]
    bowers_fft_opt_fused_parallel::<F, E>(&mut values, &twiddles)
        .map_err(|_| Error::NotPowerOfTwo(domain.size()))?;
    #[cfg(not(feature = "parallel"))]
    bowers_fft_opt_fused::<F, E>(&mut values, &twiddles)
        .map_err(|_| Error::NotPowerOfTwo(domain.size()))?;
    in_place_bit_reverse_permute(&mut values);
    Ok(values)
}

/// Folds a codeword once: `F_α = F₀ + α·F₁`, recovering the halves from `F(x)`
/// and `F(−x)`. `−x` is half a period away, so `j` pairs with `j + N/2`.
///
/// The values go in over `C` and come out over `N`: a committed trace codeword
/// is base-field, and the first fold is what lifts it, since `α` is an
/// extension challenge. Later folds are `N → N`.
pub fn fold_codeword<F, C, N>(
    codeword: &[FieldElement<C>],
    domain: &Domain<F>,
    alpha: &FieldElement<N>,
) -> Result<Vec<FieldElement<N>>, Error>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<C> + IsSubFieldOf<N>,
    C: IsField + IsSubFieldOf<N>,
    N: IsField,
{
    if codeword.len() != domain.size() {
        return Err(Error::CodewordTooShort {
            coefficients: codeword.len(),
            domain: domain.size(),
        });
    }
    if domain.log_size == 0 {
        return Err(Error::NoVariablesLeft);
    }
    let half = codeword.len() / 2;
    let two_inv = (FieldElement::<F>::one() + FieldElement::<F>::one())
        .inv()
        .expect("2 is invertible");
    // The odd half needs `1/g^j`, and those are the powers of `1/g` — so one
    // inversion, not one per position. Folded together with `1/2` while we are
    // at it, since they only ever appear as a product.
    let step = domain
        .generator()
        .inv()
        .expect("a domain generator is nonzero");
    // `odd_scale` walks the powers of `1/g`, which is a chain — so a chunk
    // starts from `two_inv · step^start` and walks from there. That is what
    // lets the fold go out to the pool at all: one `pow` per chunk instead of
    // one inversion per position.
    let fold_chunk = |start: usize, out: &mut Vec<FieldElement<N>>, len: usize| {
        let mut odd_scale = &two_inv * step.pow(start as u64);
        for j in start..start + len {
            let (a, b) = (&codeword[j], &codeword[j + half]);
            let even = &two_inv * (a + b);
            let odd = &odd_scale * (a - b);
            // The base element on the left: the only direction the tower gives.
            out.push(even + odd * alpha);
            odd_scale *= &step;
        }
    };

    #[cfg(feature = "parallel")]
    {
        const SERIAL_BELOW: usize = 1 << 12;
        if half >= SERIAL_BELOW {
            let chunk = half.div_ceil(rayon::current_num_threads().max(1));
            let parts: Vec<Vec<FieldElement<N>>> = (0..half)
                .into_par_iter()
                .step_by(chunk)
                .map(|start| {
                    let len = chunk.min(half - start);
                    let mut out = Vec::with_capacity(len);
                    fold_chunk(start, &mut out, len);
                    out
                })
                .collect();
            return Ok(parts.concat());
        }
    }

    let mut out = Vec::with_capacity(half);
    fold_chunk(0, &mut out, half);
    Ok(out)
}

/// Folds `k` times, squaring the domain at each step.
///
/// The first fold lifts `C` into `N`; the rest stay there.
pub fn fold_codeword_k<F, C, N>(
    codeword: &[FieldElement<C>],
    domain: &Domain<F>,
    alphas: &[FieldElement<N>],
) -> Result<(Vec<FieldElement<N>>, Domain<F>), Error>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<C> + IsSubFieldOf<N>,
    C: IsField + IsSubFieldOf<N>,
    N: IsField,
{
    let Some((first, rest)) = alphas.split_first() else {
        let lifted = codeword
            .iter()
            .map(|v| v.clone().to_extension::<N>())
            .collect();
        return Ok((lifted, domain.clone()));
    };

    let mut current = fold_codeword::<F, C, N>(codeword, domain, first)?;
    let mut current_domain = domain.squared()?;
    for alpha in rest {
        current = fold_codeword::<F, N, N>(&current, &current_domain, alpha)?;
        current_domain = current_domain.squared()?;
    }
    Ok((current, current_domain))
}

#[cfg(test)]
mod tests {
    use super::*;
    use math::field::goldilocks::GoldilocksField as F;

    type FE = FieldElement<F>;

    fn mle(vals: &[u64]) -> Mle<F> {
        Mle::new(vals.iter().map(|v| FE::from(*v)).collect()).unwrap()
    }

    fn pseudo_mle(num_vars: usize, seed: u64) -> Mle<F> {
        let vals: Vec<u64> = (0..(1u64 << num_vars))
            .map(|i| (i.wrapping_mul(6364136223846793005).wrapping_add(seed)) >> 13)
            .collect();
        mle(&vals)
    }

    /// Evaluates the univariate lift directly, for comparison.
    fn eval_univariate(coeffs: &[FE], x: &FE) -> FE {
        coeffs.iter().rfold(FE::zero(), |acc, c| acc * x + c)
    }

    #[test]
    #[ignore = "timing, not a property"]
    fn measure_encode_against_horner() {
        let num_vars = 12;
        let f = pseudo_mle(num_vars, 5);
        let coeffs = lift_coefficients(&f);
        let domain = Domain::<F>::new(num_vars + 2).unwrap();

        let t = std::time::Instant::now();
        let ntt = encode::<F, F>(&coeffs, &domain).unwrap();
        let ntt_time = t.elapsed();

        let t = std::time::Instant::now();
        let horner: Vec<FE> = domain
            .elements()
            .iter()
            .map(|x| coeffs.iter().rfold(FE::zero(), |acc, c| x * acc + c))
            .collect();
        let horner_time = t.elapsed();

        assert_eq!(ntt, horner);
        println!(
            "MEASURE encode 2^{num_vars} on 2^{}: ntt {ntt_time:?} vs horner {horner_time:?}",
            domain.log_size()
        );
    }

    /// The property a base-field commitment rests on: folding a base codeword
    /// with an extension challenge gives what folding the lifted one would.
    /// Without it, committing the trace over `F` would not be the same
    /// statement as committing it over `E`.
    #[test]
    fn folding_a_base_codeword_matches_folding_the_lifted_one() {
        use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext;
        type ExtE = FieldElement<Ext>;

        for num_vars in 1..=4usize {
            let f = pseudo_mle(num_vars, 3 * num_vars as u64);
            let domain = Domain::<F>::new(num_vars + 2).unwrap();
            let alphas: Vec<ExtE> = (0..num_vars).map(|i| ExtE::from(31 + i as u64)).collect();

            // Left: a base codeword, folded with extension challenges.
            let base = encode::<F, F>(&lift_coefficients(&f), &domain).unwrap();
            let (from_base, _) = fold_codeword_k::<F, F, Ext>(&base, &domain, &alphas).unwrap();

            // Right: lift first, then fold.
            let lifted =
                Mle::new(f.evals().iter().map(|v| v.to_extension::<Ext>()).collect()).unwrap();
            let extension = encode::<F, Ext>(&lift_coefficients(&lifted), &domain).unwrap();
            let (from_extension, _) =
                fold_codeword_k::<F, Ext, Ext>(&extension, &domain, &alphas).unwrap();

            assert_eq!(from_base, from_extension, "num_vars={num_vars}");
        }
    }

    #[test]
    fn the_domain_has_the_stated_order() {
        for log_size in 1..=6usize {
            let d = Domain::<F>::new(log_size).unwrap();
            assert_eq!(d.size(), 1 << log_size);
            assert_eq!(d.elements().len(), 1 << log_size);
            assert_eq!(d.generator().pow(d.size() as u64), FE::one());
        }
    }

    #[test]
    fn squaring_the_domain_halves_it() {
        let d = Domain::<F>::new(4).unwrap();
        let sq = d.squared().unwrap();
        assert_eq!(sq.log_size(), 3);
        // Squaring every element lands in the smaller subgroup, and covers it:
        // each of its elements is hit exactly twice.
        let smaller = sq.elements();
        let mut hits = vec![0usize; smaller.len()];
        for x in d.elements() {
            let y = x.square();
            let at = smaller
                .iter()
                .position(|e| *e == y)
                .expect("square left the subgroup");
            hits[at] += 1;
        }
        assert!(hits.iter().all(|h| *h == 2), "hits = {hits:?}");
    }

    #[test]
    fn opposite_points_are_half_a_period_apart() {
        // The pairing `fold_codeword` relies on: g^(j + N/2) = −g^j.
        let d = Domain::<F>::new(4).unwrap();
        let els = d.elements();
        let half = d.size() / 2;
        for j in 0..half {
            assert_eq!(els[j + half], -&els[j], "j={j}");
        }
    }

    #[test]
    fn monomial_coefficients_reproduce_the_evaluations() {
        // Σ_S ĉ_S ∏_{i∈S} x_i must give back f on every corner.
        let f = mle(&[7, 11, 13, 17, 19, 23, 29, 31]);
        let coeffs = monomial_coefficients(&f);
        let n = 3;

        for idx in 0..8usize {
            // Variable 0 is the most significant bit of the index.
            let x: Vec<FE> = (0..n)
                .map(|i| FE::from(((idx >> (n - 1 - i)) & 1) as u64))
                .collect();
            let mut total = FE::zero();
            for (mask, c) in coeffs.iter().enumerate() {
                let mut term = *c;
                for (i, x_i) in x.iter().enumerate() {
                    if (mask >> (n - 1 - i)) & 1 == 1 {
                        term *= *x_i;
                    }
                }
                total += term;
            }
            assert_eq!(total, f.evals()[idx], "corner {idx}");
        }
    }

    #[test]
    fn encoding_evaluates_the_lift_on_the_domain() {
        let f = pseudo_mle(3, 1);
        let coeffs = monomial_coefficients(&f);
        let domain = Domain::<F>::new(5).unwrap();
        let codeword = encode(&coeffs, &domain).unwrap();

        for (i, x) in domain.elements().iter().enumerate() {
            assert_eq!(codeword[i], eval_univariate(&coeffs, x), "point {i}");
        }
    }

    #[test]
    fn a_codeword_longer_than_the_domain_is_rejected() {
        let f = pseudo_mle(4, 1);
        let coeffs = monomial_coefficients(&f);
        let domain = Domain::<F>::new(3).unwrap();
        assert!(matches!(
            encode(&coeffs, &domain).unwrap_err(),
            Error::CodewordTooShort { .. }
        ));
    }

    /// The identity the whole construction rests on.
    #[test]
    fn folding_commutes_with_fixing_a_variable() {
        for num_vars in 1..=4usize {
            for log_blowup in 1..=2usize {
                let f = pseudo_mle(num_vars, 7 * num_vars as u64 + log_blowup as u64);
                let domain = Domain::<F>::new(num_vars + log_blowup).unwrap();
                let codeword = encode(&monomial_coefficients(&f), &domain).unwrap();

                let alpha = FE::from(31 + num_vars as u64);

                // Left: fold the codeword.
                let folded = fold_codeword(&codeword, &domain, &alpha).unwrap();

                // Right: fix the last variable, then encode on the squared domain.
                let mut fixed = f;
                fixed.fix_last_variable_in_place(&alpha).unwrap();
                let smaller = domain.squared().unwrap();
                let re_encoded = encode(&monomial_coefficients(&fixed), &smaller).unwrap();

                assert_eq!(
                    folded, re_encoded,
                    "num_vars={num_vars}, log_blowup={log_blowup}"
                );
            }
        }
    }

    #[test]
    fn folding_k_times_matches_fixing_k_variables() {
        let num_vars = 4;
        let log_blowup = 2;
        let f = pseudo_mle(num_vars, 99);
        let domain = Domain::<F>::new(num_vars + log_blowup).unwrap();
        let codeword = encode(&monomial_coefficients(&f), &domain).unwrap();

        let alphas = [FE::from(3), FE::from(5), FE::from(7)];
        let (folded, folded_domain) = fold_codeword_k(&codeword, &domain, &alphas).unwrap();

        let mut fixed = f;
        for alpha in &alphas {
            fixed.fix_last_variable_in_place(alpha).unwrap();
        }
        let re_encoded = encode(&monomial_coefficients(&fixed), &folded_domain).unwrap();

        assert_eq!(
            folded_domain.log_size(),
            num_vars + log_blowup - alphas.len()
        );
        assert_eq!(folded, re_encoded);
        assert_eq!(fixed.num_vars(), num_vars - alphas.len());
    }

    #[test]
    fn folding_preserves_the_rate() {
        // The point of folding: the codeword and the message shrink together,
        // so the code's rate is unchanged and the proximity test still applies.
        let (num_vars, log_blowup) = (4usize, 2usize);
        let f = pseudo_mle(num_vars, 5);
        let domain = Domain::<F>::new(num_vars + log_blowup).unwrap();
        let codeword = encode(&monomial_coefficients(&f), &domain).unwrap();

        let alphas = [FE::from(11), FE::from(13)];
        let (folded, folded_domain) = fold_codeword_k(&codeword, &domain, &alphas).unwrap();

        let message_len = 1usize << (num_vars - alphas.len());
        assert_eq!(folded.len(), folded_domain.size());
        assert_eq!(folded.len(), message_len << log_blowup);
    }

    #[test]
    fn folding_with_zero_keeps_only_the_even_part() {
        // α = 0 selects F₀, which is f with its last variable set to 0 — the
        // even-indexed hypercube values.
        let f = mle(&[3, 5, 8, 13]);
        let domain = Domain::<F>::new(3).unwrap();
        let codeword = encode(&monomial_coefficients(&f), &domain).unwrap();
        let folded = fold_codeword(&codeword, &domain, &FE::zero()).unwrap();

        let expected_mle = mle(&[3, 8]); // f(·, 0)
        let expected = encode(
            &monomial_coefficients(&expected_mle),
            &domain.squared().unwrap(),
        )
        .unwrap();
        assert_eq!(folded, expected);
    }

    #[test]
    fn folding_an_exhausted_domain_is_an_error() {
        let domain = Domain::<F>::new(0).unwrap();
        let codeword = vec![FE::from(4)];
        assert!(fold_codeword(&codeword, &domain, &FE::one()).is_err());
        assert!(domain.squared().is_err());
    }

    #[test]
    fn bit_reversal_is_an_involution() {
        for width in 0..=5usize {
            for i in 0..(1usize << width) {
                assert_eq!(reverse_bits(reverse_bits(i, width), width), i);
            }
        }
    }

    /// The alignment the evaluation argument depends on: with the lift,
    /// one fold binds the variable one sumcheck round binds.
    #[test]
    fn lifted_folding_binds_the_first_variable() {
        for num_vars in 1..=4usize {
            for log_blowup in 1..=2usize {
                let f = pseudo_mle(num_vars, 3 * num_vars as u64 + log_blowup as u64);
                let domain = Domain::<F>::new(num_vars + log_blowup).unwrap();
                let codeword = encode(&lift_coefficients(&f), &domain).unwrap();

                let alpha = FE::from(23 + num_vars as u64);
                let folded = fold_codeword(&codeword, &domain, &alpha).unwrap();

                let mut fixed = f;
                fixed.fix_first_variable_in_place(&alpha).unwrap();
                let re_encoded =
                    encode(&lift_coefficients(&fixed), &domain.squared().unwrap()).unwrap();

                assert_eq!(folded, re_encoded, "n={num_vars}, blowup={log_blowup}");
            }
        }
    }

    #[test]
    fn folding_every_variable_leaves_the_evaluation() {
        // Fold all the way and the codeword is the constant f(α), which is the
        // value the evaluation argument compares against.
        let (num_vars, log_blowup) = (4usize, 2usize);
        let f = pseudo_mle(num_vars, 77);
        let domain = Domain::<F>::new(num_vars + log_blowup).unwrap();
        let codeword = encode(&lift_coefficients(&f), &domain).unwrap();

        let alphas: Vec<FE> = (0..num_vars).map(|i| FE::from(5 + i as u64)).collect();
        let (folded, folded_domain) = fold_codeword_k(&codeword, &domain, &alphas).unwrap();

        assert_eq!(folded_domain.log_size(), log_blowup);
        let expected = f.evaluate(&alphas).unwrap();
        assert!(
            folded.iter().all(|v| *v == expected),
            "the fully folded codeword must be constant f(α)"
        );
    }
}
