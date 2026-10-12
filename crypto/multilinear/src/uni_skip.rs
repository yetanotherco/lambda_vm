//! The univariate skip: the first `l_skip` Boolean variables become one variable
//! ranging over `D`, the multiplicative subgroup of order `2^l_skip`, so the
//! domain is a prism `D × {0,1}^n`. `eq_D` is the Lagrange kernel on `D`.

use math::field::{
    element::FieldElement,
    traits::{IsField, IsPrimeField},
};

use crate::{Error, eq::eq_and_rot_eval, eq::eq_eval};

/// Successive squarings `x, x², x⁴, …`.
fn squarings<F: IsField>(x: &FieldElement<F>) -> impl Iterator<Item = FieldElement<F>> + use<F> {
    let mut current = x.clone();
    std::iter::from_fn(move || {
        let out = current.clone();
        current = current.square();
        Some(out)
    })
}

/// `2^{-l_skip}`, the normalizing factor every `eq_D` carries.
fn inv_two_pow<F: IsField>(l_skip: usize) -> FieldElement<F> {
    let half = (FieldElement::<F>::one() + FieldElement::<F>::one())
        .inv()
        .expect("2 is invertible in an odd-characteristic field");
    (0..l_skip).fold(FieldElement::<F>::one(), |acc, _| acc * &half)
}

/// The Lagrange kernel on `D`, the subgroup of order `2^l_skip`. On `D × D` it
/// is the equality indicator.
pub fn eq_uni<F: IsField>(
    l_skip: usize,
    x: &FieldElement<F>,
    y: &FieldElement<F>,
) -> FieldElement<F> {
    let one = FieldElement::<F>::one();
    let mut res = one.clone();
    for (x_pow, y_pow) in squarings(x).zip(squarings(y)).take(l_skip) {
        res = (&x_pow + &y_pow) * res + (&x_pow - &one) * (&y_pow - &one);
    }
    res * inv_two_pow::<F>(l_skip)
}

/// `eq_D(x, 1)`, which collapses to a product of `x^{2^i} + 1`.
pub fn eq_uni_at_one<F: IsField>(l_skip: usize, x: &FieldElement<F>) -> FieldElement<F> {
    let one = FieldElement::<F>::one();
    let mut res = one.clone();
    for x_pow in squarings(x).take(l_skip) {
        res *= x_pow + &one;
    }
    res * inv_two_pow::<F>(l_skip)
}

/// A univariate polynomial in coefficient form, lowest degree first.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnivariatePoly<F: IsField>(Vec<FieldElement<F>>);

impl<F: IsField> UnivariatePoly<F> {
    pub fn new(coeffs: Vec<FieldElement<F>>) -> Self {
        Self(coeffs)
    }

    pub fn coeffs(&self) -> &[FieldElement<F>] {
        &self.0
    }

    pub fn into_coeffs(self) -> Vec<FieldElement<F>> {
        self.0
    }

    /// Horner evaluation.
    pub fn evaluate(&self, x: &FieldElement<F>) -> FieldElement<F> {
        self.0
            .iter()
            .rfold(FieldElement::<F>::zero(), |acc, c| acc * x + c)
    }
}

/// `eq_D(x, ·)` as a polynomial in its second argument.
///
/// `eq_D` is the Lagrange basis at `x`, which on `D` is the character sum
/// `eq_D(x, z) = 2^{-l} Σ_k x^k z^{N−k}`; reading that as coefficients in `z`
/// gives the reversed powers of `x`.
pub fn eq_uni_poly<F: IsField>(l_skip: usize, x: &FieldElement<F>) -> UnivariatePoly<F> {
    let n = 1usize << l_skip;
    let n_inv = inv_two_pow::<F>(l_skip);

    let mut coeffs = Vec::with_capacity(n);
    let mut x_pow = x.clone();
    for _ in 0..n {
        coeffs.push(&x_pow * &n_inv);
        x_pow *= x;
    }
    coeffs.reverse();
    coeffs[0] = n_inv;
    UnivariatePoly::new(coeffs)
}

/// The prism analogue of `eq`: subgroup on the first coordinate, hypercube on
/// the rest.
pub fn eq_prism<F: IsField>(
    l_skip: usize,
    x: &[FieldElement<F>],
    y: &[FieldElement<F>],
) -> Result<FieldElement<F>, Error> {
    if x.is_empty() || y.is_empty() {
        return Err(Error::VariableCountMismatch {
            expected: 1,
            got: 0,
        });
    }
    Ok(eq_uni(l_skip, &x[0], &y[0]) * eq_eval(&x[1..], &y[1..])?)
}

/// The Möbius variant of `eq`, `∏_i ((1 − 2u_i)(1 − x_i) + u_i·x_i)`.
pub fn mobius_eq_eval<F: IsField>(
    u: &[FieldElement<F>],
    x: &[FieldElement<F>],
) -> Result<FieldElement<F>, Error> {
    if u.len() != x.len() {
        return Err(Error::VariableCountMismatch {
            expected: u.len(),
            got: x.len(),
        });
    }
    let one = FieldElement::<F>::one();
    Ok(u.iter().zip(x).fold(one.clone(), |acc, (u_i, x_i)| {
        let w0 = &one - u_i - u_i;
        acc * (w0 * (&one - x_i) + u_i * x_i)
    }))
}

/// The rotation kernel on the prism. A step inside `D` multiplies by `ω`; only
/// at the end of `D` does it carry into the cube. `omega` must generate `D`.
pub fn rot_kernel_prism<F: IsField>(
    l_skip: usize,
    omega: &FieldElement<F>,
    x: &[FieldElement<F>],
    y: &[FieldElement<F>],
) -> Result<FieldElement<F>, Error> {
    if x.len() != y.len() || x.is_empty() {
        return Err(Error::VariableCountMismatch {
            expected: x.len().max(1),
            got: y.len(),
        });
    }
    let (eq_cube, rot_cube) = eq_and_rot_eval(&x[1..], &y[1..])?;
    let y0_omega = &y[0] * omega;

    Ok(eq_uni(l_skip, &x[0], &y0_omega) * &eq_cube
        + eq_uni_at_one(l_skip, &x[0]) * eq_uni_at_one(l_skip, &y0_omega) * (rot_cube - &eq_cube))
}

/// Generator of `D`, the subgroup of order `2^l_skip`.
pub fn skip_domain_generator<F>(l_skip: usize) -> Result<FieldElement<F>, Error>
where
    F: math::field::traits::IsFFTField + IsPrimeField,
{
    F::get_primitive_root_of_unity(l_skip as u64).map_err(|_| Error::SkipDomainUnavailable {
        l_skip,
        two_adicity: F::TWO_ADICITY as usize,
    })
}

/// Every element of `D`, in the order `1, ω, ω², …`.
pub fn skip_domain<F>(l_skip: usize) -> Result<Vec<FieldElement<F>>, Error>
where
    F: math::field::traits::IsFFTField + IsPrimeField,
{
    let omega = skip_domain_generator::<F>(l_skip)?;
    let mut out = Vec::with_capacity(1 << l_skip);
    let mut current = FieldElement::<F>::one();
    for _ in 0..(1usize << l_skip) {
        out.push(current.clone());
        current *= &omega;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use math::field::goldilocks::GoldilocksField as F;

    type FE = FieldElement<F>;

    #[test]
    fn eq_uni_is_the_indicator_on_the_subgroup() {
        for l_skip in 1..=4usize {
            let d = skip_domain::<F>(l_skip).unwrap();
            assert_eq!(d.len(), 1 << l_skip);
            for (i, z1) in d.iter().enumerate() {
                for (j, z2) in d.iter().enumerate() {
                    let expected = if i == j { FE::one() } else { FE::zero() };
                    assert_eq!(eq_uni(l_skip, z1, z2), expected, "l={l_skip}, {i} vs {j}");
                }
            }
        }
    }

    #[test]
    fn the_domain_is_a_subgroup_of_the_right_order() {
        for l_skip in 1..=5usize {
            let omega = skip_domain_generator::<F>(l_skip).unwrap();
            let order = 1u64 << l_skip;
            assert_eq!(omega.pow(order), FE::one(), "l={l_skip}");
            if l_skip > 0 {
                assert_ne!(omega.pow(order / 2), FE::one(), "l={l_skip} not primitive");
            }
        }
    }

    #[test]
    fn eq_uni_at_one_agrees_with_the_general_form() {
        let one = FE::one();
        for l_skip in 0..=4usize {
            for x in [FE::from(3), FE::from(17), FE::from(1)] {
                assert_eq!(
                    eq_uni_at_one(l_skip, &x),
                    eq_uni(l_skip, &x, &one),
                    "l={l_skip}"
                );
            }
        }
    }

    #[test]
    fn eq_uni_is_symmetric() {
        let (x, y) = (FE::from(11), FE::from(29));
        for l_skip in 0..=4usize {
            assert_eq!(eq_uni(l_skip, &x, &y), eq_uni(l_skip, &y, &x));
        }
    }

    #[test]
    fn eq_uni_sums_to_one_over_the_domain() {
        // Lagrange bases partition unity: Σ_{z ∈ D} eq_D(x, z) = 1.
        for l_skip in 1..=4usize {
            let d = skip_domain::<F>(l_skip).unwrap();
            let x = FE::from(1234);
            let total = d
                .iter()
                .fold(FE::zero(), |acc, z| acc + eq_uni(l_skip, &x, z));
            assert_eq!(total, FE::one(), "l={l_skip}");
        }
    }

    #[test]
    fn eq_uni_poly_is_eq_uni_in_its_second_argument() {
        for l_skip in 1..=4usize {
            let x = FE::from(97);
            let poly = eq_uni_poly(l_skip, &x);
            assert_eq!(poly.coeffs().len(), 1 << l_skip);
            for z in [FE::from(2), FE::from(5), FE::from(1000)] {
                assert_eq!(poly.evaluate(&z), eq_uni(l_skip, &x, &z), "l={l_skip}");
            }
        }
    }

    #[test]
    fn horner_matches_the_written_out_polynomial() {
        // 4 + 3x + 2x² + x³
        let poly = UnivariatePoly::new(vec![FE::from(4), FE::from(3), FE::from(2), FE::from(1)]);
        let x = FE::from(5);
        let expected = FE::from(4) + FE::from(3) * x + FE::from(2) * x * x + x * x * x;
        assert_eq!(poly.evaluate(&x), expected);
    }

    #[test]
    fn a_zero_skip_leaves_the_plain_hypercube_eq() {
        // With l_skip = 0 the subgroup is trivial and eq_D is the constant one,
        // so the prism degenerates to the cube.
        let x = vec![FE::from(3), FE::from(5), FE::from(7)];
        let y = vec![FE::from(11), FE::from(13), FE::from(17)];
        assert_eq!(eq_uni(0, &x[0], &y[0]), FE::one());
        assert_eq!(
            eq_prism(0, &x, &y).unwrap(),
            eq_eval(&x[1..], &y[1..]).unwrap()
        );
    }

    #[test]
    fn eq_prism_is_the_indicator_on_the_prism() {
        let l_skip = 2;
        let d = skip_domain::<F>(l_skip).unwrap();
        let cube = [FE::zero(), FE::one()];

        for (i, u) in d.iter().enumerate() {
            for a in cube {
                for (j, v) in d.iter().enumerate() {
                    for b in cube {
                        let x = [*u, a];
                        let y = [*v, b];
                        let expected = if i == j && a == b {
                            FE::one()
                        } else {
                            FE::zero()
                        };
                        assert_eq!(eq_prism(l_skip, &x, &y).unwrap(), expected);
                    }
                }
            }
        }
    }

    #[test]
    fn mobius_eq_matches_its_written_form() {
        let u = vec![FE::from(3), FE::from(5)];
        let x = vec![FE::from(7), FE::from(11)];
        let one = FE::one();
        let expected = ((one - FE::from(3) - FE::from(3)) * (one - FE::from(7))
            + FE::from(3) * FE::from(7))
            * ((one - FE::from(5) - FE::from(5)) * (one - FE::from(11))
                + FE::from(5) * FE::from(11));
        assert_eq!(mobius_eq_eval(&u, &x).unwrap(), expected);
    }

    #[test]
    fn rot_on_the_prism_steps_within_the_domain() {
        // Inside D a step is multiplication by omega, with the cube part fixed.
        let l_skip = 2;
        let omega = skip_domain_generator::<F>(l_skip).unwrap();
        let d = skip_domain::<F>(l_skip).unwrap();
        let cube_pt = [FE::zero(), FE::one()];

        for (i, u) in d.iter().enumerate() {
            let x: Vec<FE> = std::iter::once(*u).chain(cube_pt.iter().cloned()).collect();
            for (j, v) in d.iter().enumerate() {
                let y: Vec<FE> = std::iter::once(*v).chain(cube_pt.iter().cloned()).collect();
                let got = rot_kernel_prism(l_skip, &omega, &x, &y).unwrap();
                // A step inside D keeps the cube point. The one exception is
                // stepping off the last element of D, which must carry into the
                // cube instead — so with the cube point held fixed it is zero.
                let steps_within_d = i == j + 1;
                let expected = if steps_within_d {
                    FE::one()
                } else {
                    FE::zero()
                };
                assert_eq!(got, expected, "u={i}, v={j}");
            }
        }
    }

    #[test]
    fn rot_on_the_prism_carries_into_the_cube_at_the_domain_boundary() {
        // Stepping off the last element of D advances the cube coordinate.
        let l_skip = 2;
        let omega = skip_domain_generator::<F>(l_skip).unwrap();
        let d = skip_domain::<F>(l_skip).unwrap();
        let last = *d.last().unwrap();
        let first = d[0];

        // cube index 0 -> 1 on a single cube variable.
        let y = vec![last, FE::zero()];
        let x = vec![first, FE::one()];
        assert_eq!(rot_kernel_prism(l_skip, &omega, &x, &y).unwrap(), FE::one());
    }

    #[test]
    fn a_domain_beyond_two_adicity_is_rejected() {
        let err = skip_domain_generator::<F>(64).unwrap_err();
        assert!(matches!(
            err,
            Error::SkipDomainUnavailable { l_skip: 64, .. }
        ));
    }
}
