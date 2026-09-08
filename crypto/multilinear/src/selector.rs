//! Row selectors: which steps a constraint applies to.
//!
//! A transition constraint that reads the next step cannot hold on the last
//! step — there is nothing after it. The univariate prover handles this by
//! shrinking the zerofier; on the hypercube the constraint is instead
//! multiplied by a **selector** `s(x)`, one on the steps where it applies and
//! zero elsewhere, so `s·C` really does vanish everywhere.
//!
//! Our constraint metadata expresses this as `end_exemptions = k`: the
//! constraint applies to steps `0 .. N − k`. The selector is therefore the
//! indicator of `index(x) < N − k`.
//!
//! The prover materializes the table; the verifier needs the same value at a
//! random point without touching `2^n` entries, so [`Selector::evaluate`] is a
//! closed form costing `O(n)` field operations. Both are multilinear, so a
//! selector adds exactly one to the degree of whatever it multiplies.

use math::field::{element::FieldElement, traits::IsField};

use crate::{Error, mle::Mle};

/// The indicator of `index(x) < 2^n − end_exemptions`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Selector {
    pub end_exemptions: usize,
}

impl Selector {
    /// Applies on every step.
    pub const ALL: Selector = Selector { end_exemptions: 0 };

    pub const fn except_last(k: usize) -> Selector {
        Selector { end_exemptions: k }
    }

    /// True when this selector is the constant one and can be skipped.
    pub fn is_trivial(&self) -> bool {
        self.end_exemptions == 0
    }

    /// Number of steps the constraint applies to.
    pub fn active_steps(&self, num_vars: usize) -> usize {
        (1usize << num_vars).saturating_sub(self.end_exemptions)
    }

    /// The selector's hypercube table.
    pub fn table<F: IsField>(&self, num_vars: usize) -> Result<Mle<F>, Error> {
        let size = 1usize << num_vars;
        if self.end_exemptions > size {
            return Err(Error::TooManyExemptions {
                exemptions: self.end_exemptions,
                size,
            });
        }
        let cutoff = size - self.end_exemptions;
        let evals = (0..size)
            .map(|i| {
                if i < cutoff {
                    FieldElement::<F>::one()
                } else {
                    FieldElement::<F>::zero()
                }
            })
            .collect();
        Mle::new(evals)
    }

    /// The selector at an arbitrary point, in `O(num_vars)`.
    ///
    /// `s(x) = 1 − geq(x, cutoff)`, where `geq` is the multilinear indicator of
    /// `index(x) >= cutoff`: either `x` matches `cutoff` bit for bit, or they
    /// first differ at a position where `cutoff` has a zero and `x` a one.
    pub fn evaluate<F: IsField>(
        &self,
        point: &[FieldElement<F>],
    ) -> Result<FieldElement<F>, Error> {
        let num_vars = point.len();
        let size = 1usize << num_vars;
        if self.end_exemptions > size {
            return Err(Error::TooManyExemptions {
                exemptions: self.end_exemptions,
                size,
            });
        }
        if self.is_trivial() {
            return Ok(FieldElement::one());
        }
        let cutoff = size - self.end_exemptions;
        if cutoff == 0 {
            // Every step is exempt: the constraint applies nowhere.
            return Ok(FieldElement::zero());
        }

        let one = FieldElement::<F>::one();
        // Running product of "x agrees with cutoff on every earlier bit".
        let mut prefix = one.clone();
        let mut geq = FieldElement::<F>::zero();

        for (i, x_i) in point.iter().enumerate() {
            // Variable 0 is the most significant bit.
            let bit = (cutoff >> (num_vars - 1 - i)) & 1;
            if bit == 0 {
                // x exceeds cutoff here: everything above matched, x_i = 1.
                geq += &prefix * x_i;
                prefix *= &one - x_i;
            } else {
                prefix *= x_i;
            }
        }
        // The remaining prefix is the "x == cutoff" case.
        Ok(one - (geq + prefix))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use math::field::goldilocks::GoldilocksField as F;

    type FE = FieldElement<F>;

    fn corner(index: usize, num_vars: usize) -> Vec<FE> {
        (0..num_vars)
            .map(|i| FE::from(((index >> (num_vars - 1 - i)) & 1) as u64))
            .collect()
    }

    #[test]
    fn all_is_the_constant_one() {
        let s = Selector::ALL;
        assert!(s.is_trivial());
        let table = s.table::<F>(3).unwrap();
        assert!(table.evals().iter().all(|v| *v == FE::one()));
        assert_eq!(s.evaluate(&corner(5, 3)).unwrap(), FE::one());
    }

    #[test]
    fn except_last_one_masks_only_the_final_step() {
        let s = Selector::except_last(1);
        let table = s.table::<F>(3).unwrap();
        for i in 0..8 {
            let expected = if i < 7 { FE::one() } else { FE::zero() };
            assert_eq!(table.evals()[i], expected, "step {i}");
        }
    }

    #[test]
    fn except_last_two_masks_the_final_pair() {
        let s = Selector::except_last(2);
        let table = s.table::<F>(4).unwrap();
        for i in 0..16 {
            let expected = if i < 14 { FE::one() } else { FE::zero() };
            assert_eq!(table.evals()[i], expected, "step {i}");
        }
    }

    #[test]
    fn closed_form_matches_the_table_on_every_corner() {
        for num_vars in 1..=5usize {
            let size = 1usize << num_vars;
            for k in 0..=size {
                let s = Selector::except_last(k);
                let table = s.table::<F>(num_vars).unwrap();
                for i in 0..size {
                    assert_eq!(
                        s.evaluate(&corner(i, num_vars)).unwrap(),
                        table.evals()[i],
                        "num_vars={num_vars}, k={k}, step={i}"
                    );
                }
            }
        }
    }

    #[test]
    fn closed_form_is_the_multilinear_extension_off_the_cube() {
        // The verifier evaluates the closed form at a random point; it must be
        // the same polynomial the prover's table extends to.
        for k in [1usize, 2, 3, 5] {
            let s = Selector::except_last(k);
            let table = s.table::<F>(4).unwrap();
            let point = vec![FE::from(9), FE::from(17), FE::from(3), FE::from(41)];
            assert_eq!(
                s.evaluate(&point).unwrap(),
                table.evaluate(&point).unwrap(),
                "k={k}"
            );
        }
    }

    #[test]
    fn exempting_everything_selects_nothing() {
        let s = Selector::except_last(8);
        let table = s.table::<F>(3).unwrap();
        assert!(table.evals().iter().all(|v| *v == FE::zero()));
        assert_eq!(s.evaluate(&corner(0, 3)).unwrap(), FE::zero());
        assert_eq!(s.evaluate(&[FE::from(7); 3]).unwrap(), FE::zero());
    }

    #[test]
    fn active_step_count_matches_the_table() {
        for k in 0..=8usize {
            let s = Selector::except_last(k);
            let table = s.table::<F>(3).unwrap();
            let ones = table.evals().iter().filter(|v| **v == FE::one()).count();
            assert_eq!(ones, s.active_steps(3), "k={k}");
        }
    }

    #[test]
    fn more_exemptions_than_steps_is_an_error() {
        let s = Selector::except_last(9);
        assert_eq!(
            s.table::<F>(3).unwrap_err(),
            Error::TooManyExemptions {
                exemptions: 9,
                size: 8
            }
        );
        assert!(s.evaluate(&corner(0, 3)).is_err());
    }
}
