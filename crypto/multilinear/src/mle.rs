//! Multilinear extensions, held as their `2^n` hypercube evaluations.
//!
//! Index `i` is read with **variable 0 as the most significant bit**. Every
//! fold in this crate assumes that.

use std::sync::Arc;

use math::field::{
    element::FieldElement,
    traits::{IsField, IsSubFieldOf},
};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::Error;

/// Host memory holding several columns' evaluations back to back, written
/// once and only read after that: an epoch's trace columns, say, laid out
/// where a copy to the card reads them directly. [`Mle::shared`] makes a
/// column that is a view into it.
///
/// The slice must not change for as long as the backing lives: every view
/// hands it out as its evaluations, and clones of a view share it.
pub trait HostColumns<F: IsField>:
    Send + Sync + std::panic::RefUnwindSafe + std::panic::UnwindSafe
{
    /// The whole backing.
    fn slice(&self) -> &[FieldElement<F>];

    /// Whether the backing is page-locked, so a copy to the card reads it
    /// without staging it first.
    fn is_pinned(&self) -> bool;
}

/// Where an [`Mle`]'s evaluations are.
enum Evals<F: IsField> {
    /// Its own `Vec`.
    Owned(Vec<FieldElement<F>>),
    /// `len` values from `start` of a backing other columns share. From
    /// `nonzero` on, the values are zero in their raw representation, as the
    /// backing's writer found them.
    Shared {
        backing: Arc<dyn HostColumns<F>>,
        start: usize,
        len: usize,
        nonzero: usize,
    },
}

impl<F: IsField> Evals<F> {
    fn as_slice(&self) -> &[FieldElement<F>] {
        match self {
            Self::Owned(values) => values,
            Self::Shared {
                backing,
                start,
                len,
                ..
            } => &backing.slice()[*start..*start + *len],
        }
    }
}

impl<F: IsField> Clone for Evals<F> {
    /// A view's clone is another view of the same backing: its values cannot
    /// change, so the two read the same either way.
    fn clone(&self) -> Self {
        match self {
            Self::Owned(values) => Self::Owned(values.clone()),
            Self::Shared {
                backing,
                start,
                len,
                nonzero,
            } => Self::Shared {
                backing: backing.clone(),
                start: *start,
                len: *len,
                nonzero: *nonzero,
            },
        }
    }
}

/// A multilinear polynomial held by its hypercube evaluations.
///
/// The evaluations are either its own or a view into a [`HostColumns`] backing
/// ([`Mle::shared`]). Every method reads the two alike; the folds in place and
/// [`Mle::into_evals`] copy a view out first, since the backing is read-only.
pub struct Mle<F: IsField> {
    evals: Evals<F>,
    num_vars: usize,
}

impl<F: IsField> Clone for Mle<F> {
    fn clone(&self) -> Self {
        Self {
            evals: self.evals.clone(),
            num_vars: self.num_vars,
        }
    }
}

/// Equal when the evaluations are, wherever each side holds them.
impl<F: IsField> PartialEq for Mle<F> {
    fn eq(&self, other: &Self) -> bool {
        self.evals.as_slice() == other.evals.as_slice() && self.num_vars == other.num_vars
    }
}

impl<F: IsField> Eq for Mle<F> {}

/// The evaluations and the variable count, wherever the evaluations are held.
impl<F: IsField> std::fmt::Debug for Mle<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mle")
            .field("evals", &self.evals.as_slice())
            .field("num_vars", &self.num_vars)
            .finish()
    }
}

impl<F: IsField + 'static> Mle<F> {
    /// Builds an MLE from `2^n` evaluations in hypercube order.
    pub fn new(evals: Vec<FieldElement<F>>) -> Result<Self, Error> {
        let len = evals.len();
        if !len.is_power_of_two() {
            return Err(Error::NotPowerOfTwo(len));
        }
        Ok(Self {
            num_vars: len.trailing_zeros() as usize,
            evals: Evals::Owned(evals),
        })
    }

    /// An MLE over `len = 2^n` evaluations from `start` of `backing`, which
    /// stay there: the MLE holds the backing, not a copy of its values.
    ///
    /// `nonzero` is where the view's all-zero tail starts, zero in the raw
    /// representation rather than as field elements, as whoever wrote the
    /// backing found it. It is carried for [`Mle::nonzero_len`], never checked
    /// here.
    pub fn shared(
        backing: Arc<dyn HostColumns<F>>,
        start: usize,
        len: usize,
        nonzero: usize,
    ) -> Result<Self, Error> {
        if !len.is_power_of_two() {
            return Err(Error::NotPowerOfTwo(len));
        }
        let size = backing.slice().len();
        if start.checked_add(len).is_none_or(|end| end > size) || nonzero > len {
            return Err(Error::ViewOutOfBounds {
                start,
                len,
                nonzero,
                backing: size,
            });
        }
        Ok(Self {
            num_vars: len.trailing_zeros() as usize,
            evals: Evals::Shared {
                backing,
                start,
                len,
                nonzero,
            },
        })
    }

    /// The constant polynomial on zero variables.
    pub fn constant(value: FieldElement<F>) -> Self {
        Self {
            evals: Evals::Owned(vec![value]),
            num_vars: 0,
        }
    }

    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    pub fn len(&self) -> usize {
        match &self.evals {
            Evals::Owned(values) => values.len(),
            Evals::Shared { len, .. } => *len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn evals(&self) -> &[FieldElement<F>] {
        self.evals.as_slice()
    }

    /// Where a view's all-zero tail starts, as [`Mle::shared`] was told; `None`
    /// for evaluations of its own, whose tail nobody has looked at.
    pub fn nonzero_len(&self) -> Option<usize> {
        match &self.evals {
            Evals::Owned(_) => None,
            Evals::Shared { nonzero, .. } => Some(*nonzero),
        }
    }

    /// Whether the evaluations are a view into page-locked host memory.
    pub fn is_pinned(&self) -> bool {
        match &self.evals {
            Evals::Owned(_) => false,
            Evals::Shared { backing, .. } => backing.is_pinned(),
        }
    }

    /// Detaches a view from its backing, copying its values into a `Vec` of its
    /// own; evaluations already its own stay where they are.
    pub fn detach(&mut self) {
        if let Evals::Shared { .. } = self.evals {
            let values = self.take_owned();
            self.evals = Evals::Owned(values);
        }
    }

    pub fn into_evals(mut self) -> Vec<FieldElement<F>> {
        self.take_owned()
    }

    /// The evaluations as a `Vec` to fold or hand out, leaving an empty one
    /// behind: its own moved out, or a view's values copied.
    fn take_owned(&mut self) -> Vec<FieldElement<F>> {
        match std::mem::replace(&mut self.evals, Evals::Owned(Vec::new())) {
            Evals::Owned(values) => values,
            shared => shared.as_slice().to_vec(),
        }
    }

    /// Fixes variable 0 to `r`, returning a polynomial on `n - 1` variables.
    ///
    /// `new[j] = (1 - r)·old[j] + r·old[j + 2^(n-1)]`, which is the multilinear
    /// interpolation between the two halves of the table.
    pub fn fix_first_variable(&self, r: &FieldElement<F>) -> Result<Self, Error> {
        if self.num_vars == 0 {
            return Err(Error::NoVariablesLeft);
        }
        let values = self.evals();
        let half = values.len() / 2;
        let evals = (0..half)
            .map(|j| {
                let lo = &values[j];
                let hi = &values[j + half];
                // lo + r·(hi - lo) — one multiplication instead of two.
                lo + r * &(hi - lo)
            })
            .collect();
        Ok(Self {
            evals: Evals::Owned(evals),
            num_vars: self.num_vars - 1,
        })
    }

    /// Fixes variable 0 in place. Same arithmetic as [`Self::fix_first_variable`],
    /// but reuses the allocation — sumcheck folds once per round.
    pub fn fix_first_variable_in_place(&mut self, r: &FieldElement<F>) -> Result<(), Error> {
        if self.num_vars == 0 {
            return Err(Error::NoVariablesLeft);
        }
        let mut evals = self.take_owned();
        let half = evals.len() / 2;
        let (lo, hi) = evals.split_at_mut(half);
        // Every index is independent, and the halves are disjoint slices, so the
        // split is what lets the rows go out to the pool at all.
        let fold = |(a, b): (&mut FieldElement<F>, &FieldElement<F>)| *a = &*a + r * &(b - &*a);
        #[cfg(feature = "parallel")]
        if half >= crate::SERIAL_BELOW {
            lo.par_iter_mut().zip(hi.par_iter()).for_each(fold);
        } else {
            lo.iter_mut().zip(hi.iter()).for_each(fold);
        }
        #[cfg(not(feature = "parallel"))]
        lo.iter_mut().zip(hi.iter()).for_each(fold);
        evals.truncate(half);
        self.evals = Evals::Owned(evals);
        self.num_vars -= 1;
        Ok(())
    }

    /// Fixes the **last** variable to `r`, returning a polynomial on `n - 1`
    /// variables.
    ///
    /// The last variable is the low bit of the index, so this pairs `2j` with
    /// `2j + 1`. Codeword folding binds variables from this end, which is why
    /// it exists alongside [`Self::fix_first_variable_in_place`].
    pub fn fix_last_variable_in_place(&mut self, r: &FieldElement<F>) -> Result<(), Error> {
        if self.num_vars == 0 {
            return Err(Error::NoVariablesLeft);
        }
        let mut evals = self.take_owned();
        let half = evals.len() / 2;
        for j in 0..half {
            let lo = evals[2 * j].clone();
            let delta = &evals[2 * j + 1] - &lo;
            evals[j] = lo + r * &delta;
        }
        evals.truncate(half);
        self.evals = Evals::Owned(evals);
        self.num_vars -= 1;
        Ok(())
    }

    /// Evaluates the extension at an arbitrary point in `F^n`.
    pub fn evaluate(&self, point: &[FieldElement<F>]) -> Result<FieldElement<F>, Error> {
        if point.len() != self.num_vars {
            return Err(Error::VariableCountMismatch {
                expected: self.num_vars,
                got: point.len(),
            });
        }
        Self::evaluate_at(self.evals(), point)
    }

    /// The extension of `evals` at `point`, without owning an [`Mle`].
    ///
    /// The first fold reads the slice and writes the half-size buffer the rest
    /// fold in place, so the table is never copied at full width. A caller
    /// holding a slice of a bigger table — a GKR layer's half, say — evaluates
    /// it without materializing it at all.
    pub fn evaluate_at(
        evals: &[FieldElement<F>],
        point: &[FieldElement<F>],
    ) -> Result<FieldElement<F>, Error> {
        if evals.len() != 1usize << point.len() {
            return Err(Error::VariableCountMismatch {
                expected: point.len(),
                got: evals.len().trailing_zeros() as usize,
            });
        }
        if let Some(value) = crate::gpu::evaluate_mle(evals, point) {
            return Ok(value);
        }
        let Some((first, rest)) = point.split_first() else {
            return Ok(evals[0].clone());
        };

        let half = evals.len() / 2;
        let (lo, hi) = evals.split_at(half);
        let combine = |(l, h): (&FieldElement<F>, &FieldElement<F>)| l + first * &(h - l);
        #[cfg(feature = "parallel")]
        let mut current: Vec<FieldElement<F>> = if half >= crate::SERIAL_BELOW {
            lo.par_iter().zip(hi.par_iter()).map(combine).collect()
        } else {
            lo.iter().zip(hi.iter()).map(combine).collect()
        };
        #[cfg(not(feature = "parallel"))]
        let mut current: Vec<FieldElement<F>> = lo.iter().zip(hi.iter()).map(combine).collect();

        for r in rest {
            let half = current.len() / 2;
            let (lo, hi) = current.split_at_mut(half);
            let fold = |(a, b): (&mut FieldElement<F>, &FieldElement<F>)| *a = &*a + r * &(b - &*a);
            #[cfg(feature = "parallel")]
            if half >= crate::SERIAL_BELOW {
                lo.par_iter_mut().zip(hi.par_iter()).for_each(fold);
            } else {
                lo.iter_mut().zip(hi.iter()).for_each(fold);
            }
            #[cfg(not(feature = "parallel"))]
            lo.iter_mut().zip(hi.iter()).for_each(fold);
            current.truncate(half);
        }
        Ok(current.swap_remove(0))
    }

    /// The extension at a point in a **larger** field.
    ///
    /// A trace column lives in the base field while the challenges do not, so
    /// this is how a committed column answers a claim: the first fold lifts,
    /// the rest stay up. Lifting the whole table first would instead cost its
    /// size times the extension degree.
    pub fn evaluate_in<E>(&self, point: &[FieldElement<E>]) -> Result<FieldElement<E>, Error>
    where
        F: IsSubFieldOf<E>,
        E: IsField + 'static,
    {
        if point.len() != self.num_vars {
            return Err(Error::VariableCountMismatch {
                expected: self.num_vars,
                got: point.len(),
            });
        }
        if let Some(value) = crate::gpu::evaluate_mle(self.evals(), point) {
            return Ok(value);
        }
        self.evaluate_in_on_host(point)
    }

    /// [`evaluate_in`](Self::evaluate_in) with the device never asked: the
    /// oracle a device value is checked against, whatever the table's size.
    pub(crate) fn evaluate_in_on_host<E>(
        &self,
        point: &[FieldElement<E>],
    ) -> Result<FieldElement<E>, Error>
    where
        F: IsSubFieldOf<E>,
        E: IsField + 'static,
    {
        if point.len() != self.num_vars {
            return Err(Error::VariableCountMismatch {
                expected: self.num_vars,
                got: point.len(),
            });
        }
        let values = self.evals();
        let Some((first, rest)) = point.split_first() else {
            return Ok(values[0].clone().to_extension::<E>());
        };

        let half = values.len() / 2;
        let mut current: Vec<FieldElement<E>> = (0..half)
            .map(|j| {
                let lo = &values[j];
                let hi = &values[j + half];
                // The base element on the left: the only direction the tower
                // gives.
                lo.clone().to_extension::<E>() + (hi - lo) * first
            })
            .collect();

        for r in rest {
            let half = current.len() / 2;
            for j in 0..half {
                let delta = &current[j + half] - &current[j];
                current[j] = &current[j] + r * &delta;
            }
            current.truncate(half);
        }
        Ok(current.into_iter().next().expect("one value remains"))
    }

    /// The single remaining evaluation, once every variable has been fixed.
    pub fn as_constant(&self) -> Option<&FieldElement<F>> {
        (self.num_vars == 0).then(|| &self.evals()[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use math::field::goldilocks::GoldilocksField as F;

    type FE = FieldElement<F>;

    fn mle(vals: &[u64]) -> Mle<F> {
        Mle::new(vals.iter().map(|v| FE::from(*v)).collect()).unwrap()
    }

    #[test]
    fn rejects_non_power_of_two() {
        let evals: Vec<FE> = (0..3).map(FE::from).collect();
        assert_eq!(Mle::new(evals).unwrap_err(), Error::NotPowerOfTwo(3));
    }

    #[test]
    fn num_vars_is_log2_of_the_table() {
        assert_eq!(mle(&[1]).num_vars(), 0);
        assert_eq!(mle(&[1, 2]).num_vars(), 1);
        assert_eq!(mle(&[1, 2, 3, 4]).num_vars(), 2);
        assert_eq!(mle(&[0; 256]).num_vars(), 8);
    }

    #[test]
    fn agrees_with_the_table_on_hypercube_corners() {
        // f(x0, x1) with x0 the high bit: [f(00), f(01), f(10), f(11)]
        let f = mle(&[7, 11, 13, 17]);
        for (i, expected) in [7u64, 11, 13, 17].iter().enumerate() {
            let x0 = FE::from(((i >> 1) & 1) as u64);
            let x1 = FE::from((i & 1) as u64);
            assert_eq!(f.evaluate(&[x0, x1]).unwrap(), FE::from(*expected));
        }
    }

    #[test]
    fn is_multilinear_in_each_variable() {
        // A multilinear polynomial is affine along every axis, so the midpoint
        // evaluation is the average of the two endpoints.
        let f = mle(&[7, 11, 13, 17]);
        let two_inv = FE::from(2).inv().unwrap();
        let x1 = FE::from(5);

        let at_0 = f.evaluate(&[FE::zero(), x1]).unwrap();
        let at_1 = f.evaluate(&[FE::one(), x1]).unwrap();
        let at_mid = f.evaluate(&[two_inv, x1]).unwrap();

        assert_eq!(at_mid, (at_0 + at_1) * two_inv);
    }

    #[test]
    fn fixing_a_variable_matches_evaluating_it() {
        let f = mle(&[3, 5, 8, 13, 21, 34, 55, 89]);
        let r = FE::from(42);
        let folded = f.fix_first_variable(&r).unwrap();

        assert_eq!(folded.num_vars(), 2);
        for a in 0..2u64 {
            for b in 0..2u64 {
                let rest = [FE::from(a), FE::from(b)];
                let via_fold = folded.evaluate(&rest).unwrap();
                let direct = f.evaluate(&[r, FE::from(a), FE::from(b)]).unwrap();
                assert_eq!(via_fold, direct);
            }
        }
    }

    #[test]
    fn in_place_fold_matches_the_allocating_one() {
        let f = mle(&[3, 5, 8, 13, 21, 34, 55, 89]);
        let r = FE::from(9);
        let expected = f.fix_first_variable(&r).unwrap();

        let mut g = f;
        g.fix_first_variable_in_place(&r).unwrap();
        assert_eq!(g, expected);
    }

    #[test]
    fn folding_every_variable_leaves_the_evaluation() {
        let f = mle(&[3, 5, 8, 13]);
        let point = [FE::from(6), FE::from(7)];
        let expected = f.evaluate(&point).unwrap();

        let mut g = f;
        for r in &point {
            g.fix_first_variable_in_place(r).unwrap();
        }
        assert_eq!(g.num_vars(), 0);
        assert_eq!(g.as_constant().unwrap(), &expected);
    }

    #[test]
    fn folding_a_constant_is_an_error() {
        let mut f = Mle::<F>::constant(FE::from(4));
        assert_eq!(
            f.fix_first_variable_in_place(&FE::from(1)).unwrap_err(),
            Error::NoVariablesLeft
        );
    }

    #[test]
    fn evaluate_rejects_a_point_of_the_wrong_arity() {
        let f = mle(&[1, 2, 3, 4]);
        assert_eq!(
            f.evaluate(&[FE::from(1)]).unwrap_err(),
            Error::VariableCountMismatch {
                expected: 2,
                got: 1
            }
        );
    }

    #[test]
    fn fixing_the_last_variable_matches_evaluating_it() {
        let f = mle(&[3, 5, 8, 13, 21, 34, 55, 89]);
        let r = FE::from(23);
        let mut folded = f.clone();
        folded.fix_last_variable_in_place(&r).unwrap();

        assert_eq!(folded.num_vars(), 2);
        for a in 0..2u64 {
            for b in 0..2u64 {
                let via_fold = folded.evaluate(&[FE::from(a), FE::from(b)]).unwrap();
                let direct = f.evaluate(&[FE::from(a), FE::from(b), r]).unwrap();
                assert_eq!(via_fold, direct, "a={a}, b={b}");
            }
        }
    }

    #[test]
    fn evaluating_in_a_larger_field_matches_lifting_first() {
        use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext;
        type ExtE = FieldElement<Ext>;

        for num_vars in 0..=4usize {
            let f = mle(&(0..(1u64 << num_vars))
                .map(|i| i.wrapping_mul(6364136223846793005) >> 13)
                .collect::<Vec<_>>());
            let point: Vec<ExtE> = (0..num_vars).map(|i| ExtE::from(101 + i as u64)).collect();

            let lifted =
                Mle::new(f.evals().iter().map(|v| v.to_extension::<Ext>()).collect()).unwrap();

            assert_eq!(
                f.evaluate_in(&point).unwrap(),
                lifted.evaluate(&point).unwrap(),
                "num_vars={num_vars}"
            );
        }
    }

    #[test]
    fn evaluating_in_the_same_field_is_evaluating() {
        let f = mle(&[3, 5, 8, 13]);
        let point = [FE::from(6), FE::from(7)];
        assert_eq!(f.evaluate_in(&point).unwrap(), f.evaluate(&point).unwrap());
    }

    #[test]
    fn first_and_last_folds_bind_different_ends() {
        let f = mle(&[1, 2, 3, 4]);
        let r = FE::from(5);
        let mut by_first = f.clone();
        by_first.fix_first_variable_in_place(&r).unwrap();
        let mut by_last = f;
        by_last.fix_last_variable_in_place(&r).unwrap();
        assert_ne!(by_first, by_last);
    }

    // ── Views into shared host columns ───────────────────────────────────────

    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext;
    use std::sync::atomic::{AtomicBool, Ordering};

    type ExtE = FieldElement<Ext>;

    /// Columns laid end to end in a `Vec`: the shape an epoch's pinned slot
    /// has, without the pinning. Raises `dropped` when it goes.
    struct Backing {
        values: Vec<FE>,
        pinned: bool,
        dropped: Arc<AtomicBool>,
    }

    impl HostColumns<F> for Backing {
        fn slice(&self) -> &[FE] {
            &self.values
        }

        fn is_pinned(&self) -> bool {
            self.pinned
        }
    }

    impl Drop for Backing {
        fn drop(&mut self) {
            self.dropped.store(true, Ordering::SeqCst);
        }
    }

    fn backing(values: Vec<FE>, pinned: bool) -> (Arc<dyn HostColumns<F>>, Arc<AtomicBool>) {
        let dropped = Arc::new(AtomicBool::new(false));
        let backing = Backing {
            values,
            pinned,
            dropped: dropped.clone(),
        };
        (Arc::new(backing), dropped)
    }

    /// splitmix64: a fixed stream, so a failure reproduces.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }

        fn fe(&mut self) -> FE {
            FE::from(self.next())
        }

        fn ext(&mut self) -> ExtE {
            ExtE::new([self.fe(), self.fe(), self.fe()])
        }

        fn table(&mut self, len: usize) -> Vec<FE> {
            (0..len).map(|_| self.fe()).collect()
        }
    }

    /// Every method of a view answers what the same table owned answers, for
    /// tables of 0 to 7 variables sitting at every place in a backing: first,
    /// between two others, and last.
    #[test]
    fn a_view_answers_every_method_as_the_owned_table_does() {
        let mut rng = Rng(0x5eed);
        for num_vars in 0..=7usize {
            let size = 1usize << num_vars;
            let tables = [rng.table(size), rng.table(size), rng.table(size)];
            // Three values before the first table, so no view starts at 0.
            let mut laid = rng.table(3);
            let mut starts = Vec::new();
            for table in &tables {
                starts.push(laid.len());
                laid.extend_from_slice(table);
            }
            let (host, _) = backing(laid, true);

            for (table, &start) in tables.iter().zip(&starts) {
                let owned = Mle::new(table.clone()).unwrap();
                let view = Mle::shared(host.clone(), start, size, size).unwrap();
                let ctx = format!("num_vars {num_vars}, start {start}");

                assert_eq!(view.num_vars(), owned.num_vars(), "{ctx}");
                assert_eq!(view.len(), owned.len(), "{ctx}");
                assert_eq!(view.is_empty(), owned.is_empty(), "{ctx}");
                assert_eq!(view.evals(), owned.evals(), "{ctx}");
                assert_eq!(view.as_constant(), owned.as_constant(), "{ctx}");
                assert_eq!(format!("{view:?}"), format!("{owned:?}"), "{ctx}");

                let point: Vec<FE> = (0..num_vars).map(|_| rng.fe()).collect();
                assert_eq!(
                    view.evaluate(&point).unwrap(),
                    owned.evaluate(&point).unwrap(),
                    "{ctx}"
                );
                assert_eq!(
                    Mle::evaluate_at(view.evals(), &point).unwrap(),
                    Mle::evaluate_at(owned.evals(), &point).unwrap(),
                    "{ctx}"
                );
                let lifted: Vec<ExtE> = (0..num_vars).map(|_| rng.ext()).collect();
                assert_eq!(
                    view.evaluate_in(&lifted).unwrap(),
                    owned.evaluate_in(&lifted).unwrap(),
                    "{ctx}"
                );
                assert_eq!(
                    view.evaluate_in_on_host(&lifted).unwrap(),
                    owned.evaluate_in_on_host(&lifted).unwrap(),
                    "{ctx}"
                );

                if num_vars > 0 {
                    let r = rng.fe();
                    assert_eq!(
                        view.fix_first_variable(&r).unwrap(),
                        owned.fix_first_variable(&r).unwrap(),
                        "{ctx}"
                    );
                    let (mut v, mut o) = (view.clone(), owned.clone());
                    v.fix_first_variable_in_place(&r).unwrap();
                    o.fix_first_variable_in_place(&r).unwrap();
                    assert_eq!(v, o, "{ctx}");
                    let (mut v, mut o) = (view.clone(), owned.clone());
                    v.fix_last_variable_in_place(&r).unwrap();
                    o.fix_last_variable_in_place(&r).unwrap();
                    assert_eq!(v, o, "{ctx}");
                } else {
                    let r = rng.fe();
                    let (mut v, mut o) = (view.clone(), owned.clone());
                    assert_eq!(
                        v.fix_first_variable_in_place(&r),
                        o.fix_first_variable_in_place(&r)
                    );
                    assert_eq!(
                        v.fix_last_variable_in_place(&r),
                        o.fix_last_variable_in_place(&r)
                    );
                    assert_eq!(view.fix_first_variable(&r), owned.fix_first_variable(&r));
                }

                // Equality runs both ways and sees a changed value.
                assert_eq!(view, owned, "{ctx}");
                assert_eq!(owned, view, "{ctx}");
                let mut changed = table.clone();
                changed[size - 1] += FE::one();
                assert_ne!(view, Mle::new(changed).unwrap(), "{ctx}");

                let cloned = view.clone();
                assert_eq!(cloned, owned, "{ctx}");
                assert_eq!(cloned.nonzero_len(), Some(size), "{ctx}");
                assert!(cloned.is_pinned(), "a view's clone is still a view");

                assert_eq!(view.into_evals(), owned.into_evals(), "{ctx}");
            }
        }
    }

    /// Folding a view in place copies it out first: the backing, and every
    /// other view of it, still read the original values.
    #[test]
    fn a_fold_in_place_leaves_the_backing_alone() {
        let mut rng = Rng(0xc0de);
        let table = rng.table(16);
        let (host, _) = backing(table.clone(), true);
        let mut folded = Mle::shared(host.clone(), 0, 16, 16).unwrap();
        let mut folded_last = folded.clone();
        let mut detached = folded.clone();
        let other = folded.clone();

        let r = rng.fe();
        folded.fix_first_variable_in_place(&r).unwrap();
        folded_last.fix_last_variable_in_place(&r).unwrap();
        detached.detach();

        assert_eq!(host.slice(), table.as_slice(), "the backing changed");
        assert_eq!(other.evals(), table.as_slice(), "another view changed");
        assert_eq!(
            folded,
            Mle::new(table.clone())
                .unwrap()
                .fix_first_variable(&r)
                .unwrap()
        );
        for (copy, what) in [(&folded, "first"), (&folded_last, "last")] {
            assert_eq!(copy.nonzero_len(), None, "the {what} fold left a view");
            assert!(!copy.is_pinned(), "the {what} fold left a view");
        }
        assert_eq!(detached, other);
        assert_eq!(detached.nonzero_len(), None);
        assert!(!detached.is_pinned());
        assert!(other.is_pinned());
    }

    /// A view holds its backing: dropping the handle it was made from keeps
    /// the values readable, and the backing goes exactly when the last view
    /// does — which is when a pinned slot may be written again.
    #[test]
    fn the_backing_goes_with_its_last_view() {
        let (host, dropped) = backing((0..8).map(FE::from).collect(), true);
        let a = Mle::shared(host.clone(), 0, 4, 4).unwrap();
        let b = Mle::shared(host.clone(), 4, 4, 2).unwrap();
        drop(host);
        let a2 = a.clone();
        drop(a);
        assert!(!dropped.load(Ordering::SeqCst));
        assert_eq!(a2.evals(), &(0..4).map(FE::from).collect::<Vec<_>>()[..]);
        assert_eq!(b.nonzero_len(), Some(2));
        let owned_copy = b.clone().into_evals();
        drop(a2);
        assert!(!dropped.load(Ordering::SeqCst));
        drop(b);
        assert!(dropped.load(Ordering::SeqCst), "the last view let go");
        assert_eq!(owned_copy, (4..8).map(FE::from).collect::<Vec<_>>());
    }

    /// Views the backing cannot hold are refused, as is a zero tail starting
    /// past the view's end; the edges themselves are fine.
    #[test]
    fn a_view_that_does_not_fit_is_refused() {
        let (host, _) = backing(vec![FE::one(); 12], false);
        assert_eq!(
            Mle::shared(host.clone(), 0, 3, 3).unwrap_err(),
            Error::NotPowerOfTwo(3)
        );
        assert_eq!(
            Mle::shared(host.clone(), 0, 0, 0).unwrap_err(),
            Error::NotPowerOfTwo(0)
        );
        let refused = |start: usize, len: usize, nonzero: usize| Error::ViewOutOfBounds {
            start,
            len,
            nonzero,
            backing: 12,
        };
        assert_eq!(
            Mle::shared(host.clone(), 5, 8, 8).unwrap_err(),
            refused(5, 8, 8)
        );
        assert_eq!(
            Mle::shared(host.clone(), 0, 16, 16).unwrap_err(),
            refused(0, 16, 16)
        );
        assert_eq!(
            Mle::shared(host.clone(), usize::MAX, 4, 4).unwrap_err(),
            refused(usize::MAX, 4, 4)
        );
        assert_eq!(
            Mle::shared(host.clone(), 0, 4, 5).unwrap_err(),
            refused(0, 4, 5)
        );

        let last = Mle::shared(host.clone(), 4, 8, 0).unwrap();
        assert_eq!(last.num_vars(), 3);
        assert_eq!(last.nonzero_len(), Some(0));
        assert!(!last.is_pinned(), "the backing says it is not pinned");
        assert!(Mle::shared(host, 11, 1, 1).is_ok());
    }

    /// A table of its own reports no zero tail and no pinning, before and after
    /// a fold.
    #[test]
    fn an_owned_table_is_neither_a_view_nor_pinned() {
        let mut f = mle(&[1, 2, 3, 0]);
        assert_eq!(f.nonzero_len(), None);
        assert!(!f.is_pinned());
        f.detach();
        assert_eq!(f, mle(&[1, 2, 3, 0]));
        f.fix_first_variable_in_place(&FE::from(3)).unwrap();
        assert_eq!(f.nonzero_len(), None);
        assert!(!f.is_pinned());
    }
}
