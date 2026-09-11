//! Packs columns of differing heights into shared `2^n_stack` polynomials.
//!
//! Each column of height `2^m` sits at an offset that is a multiple of `2^m`,
//! making it a subcube: `column(z) = stacked(prefix_bits ‖ z)`, with no protocol
//! needed. The price is alignment padding; registering widest-first minimizes it.

use math::field::{element::FieldElement, traits::IsField};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::{Error, mle::Mle};

/// Where one column ended up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Placement {
    /// Index of the stacked polynomial holding this column.
    pub poly: usize,
    /// Start offset inside that polynomial. Always a multiple of `2^num_vars`.
    pub offset: usize,
    /// The column's own variable count; its height is `2^num_vars`.
    pub num_vars: usize,
    /// Variables of the stacked polynomial, `>= num_vars`.
    pub n_stack: usize,
}

impl Placement {
    /// The high bits that select this column's subcube, most significant first.
    pub fn prefix_bits(&self) -> Vec<bool> {
        let prefix_len = self.n_stack - self.num_vars;
        let index = self.offset >> self.num_vars;
        (0..prefix_len)
            .map(|i| (index >> (prefix_len - 1 - i)) & 1 == 1)
            .collect()
    }

    /// Lifts a point on this column's cube to a point on the stacked cube.
    pub fn point_in_stacked<F: IsField>(
        &self,
        point: &[FieldElement<F>],
    ) -> Result<Vec<FieldElement<F>>, Error> {
        if point.len() != self.num_vars {
            return Err(Error::VariableCountMismatch {
                expected: self.num_vars,
                got: point.len(),
            });
        }
        let mut out: Vec<FieldElement<F>> = self
            .prefix_bits()
            .into_iter()
            .map(|b| {
                if b {
                    FieldElement::<F>::one()
                } else {
                    FieldElement::<F>::zero()
                }
            })
            .collect();
        out.extend_from_slice(point);
        Ok(out)
    }
}

/// A fixed, public assignment of columns to stacked polynomials.
///
/// Both sides derive the same layout from the same column heights, so it never
/// travels in a proof.
#[derive(Clone, Debug)]
pub struct StackedLayout {
    n_stack: usize,
    placements: Vec<Placement>,
    num_polys: usize,
}

impl StackedLayout {
    /// Places columns of the given heights, in the order given.
    ///
    /// A column that does not fit in the current polynomial starts a new one; a
    /// column taller than the stacked cube is an error.
    pub fn build(heights_log2: &[usize], n_stack: usize) -> Result<Self, Error> {
        let capacity = 1usize << n_stack;
        let mut placements = Vec::with_capacity(heights_log2.len());
        let mut poly = 0usize;
        let mut cursor = 0usize;

        for &m in heights_log2 {
            if m > n_stack {
                return Err(Error::ColumnTallerThanStack {
                    column_vars: m,
                    n_stack,
                });
            }
            let size = 1usize << m;
            // Round the cursor up to this column's alignment.
            let aligned = cursor.div_ceil(size) * size;
            let (poly, offset) = if aligned + size > capacity {
                poly += 1;
                cursor = size;
                (poly, 0)
            } else {
                cursor = aligned + size;
                (poly, aligned)
            };
            placements.push(Placement {
                poly,
                offset,
                num_vars: m,
                n_stack,
            });
        }

        let num_polys = placements_poly_count(&placements);
        Ok(Self {
            n_stack,
            placements,
            num_polys,
        })
    }

    pub fn n_stack(&self) -> usize {
        self.n_stack
    }

    pub fn num_polys(&self) -> usize {
        self.num_polys
    }

    pub fn placements(&self) -> &[Placement] {
        &self.placements
    }

    pub fn placement(&self, column: usize) -> Option<&Placement> {
        self.placements.get(column)
    }

    /// Cells that carry data, versus the `num_polys · 2^n_stack` committed.
    pub fn occupancy(&self) -> (usize, usize) {
        let used: usize = self.placements.iter().map(|p| 1usize << p.num_vars).sum();
        (used, self.num_polys * (1usize << self.n_stack))
    }

    /// Builds the stacked polynomials, zero-filling the padding.
    ///
    /// One polynomial per worker: they share no cells, and between the zeroing
    /// and the copies this is gigabytes of memory traffic on a trace of any
    /// size.
    pub fn stack<F: IsField + 'static>(&self, columns: &[&Mle<F>]) -> Result<Vec<Mle<F>>, Error> {
        if columns.len() != self.placements.len() {
            return Err(Error::VariableCountMismatch {
                expected: self.placements.len(),
                got: columns.len(),
            });
        }
        for (column, place) in columns.iter().zip(&self.placements) {
            if column.len() != 1usize << place.num_vars {
                return Err(Error::NotPowerOfTwo(column.len()));
            }
        }
        let size = 1usize << self.n_stack;
        // Which columns land in which polynomial, so a worker owning one
        // polynomial has its whole list without scanning the placements.
        let mut by_poly: Vec<Vec<usize>> = vec![Vec::new(); self.num_polys];
        for (index, place) in self.placements.iter().enumerate() {
            by_poly[place.poly].push(index);
        }
        let fill = |members: &Vec<usize>| {
            let mut buffer = vec![FieldElement::<F>::zero(); size];
            for index in members {
                let place = &self.placements[*index];
                let span = 1usize << place.num_vars;
                buffer[place.offset..place.offset + span].clone_from_slice(columns[*index].evals());
            }
            buffer
        };
        #[cfg(feature = "parallel")]
        let polys: Vec<Vec<FieldElement<F>>> = by_poly.par_iter().map(fill).collect();
        #[cfg(not(feature = "parallel"))]
        let polys: Vec<Vec<FieldElement<F>>> = by_poly.iter().map(fill).collect();

        polys.into_iter().map(Mle::new).collect()
    }
}

/// Every column, by reference — what [`stack`](StackedLayout::stack) reads.
pub fn borrow<F: IsField>(columns: &[Mle<F>]) -> Vec<&Mle<F>> {
    columns.iter().collect()
}

fn placements_poly_count(placements: &[Placement]) -> usize {
    placements.iter().map(|p| p.poly + 1).max().unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use math::field::goldilocks::GoldilocksField as F;

    type FE = FieldElement<F>;

    fn column(len: usize, seed: u64) -> Mle<F> {
        Mle::new(
            (0..len as u64)
                .map(|i| FE::from(i.wrapping_mul(2654435761).wrapping_add(seed)))
                .collect(),
        )
        .unwrap()
    }

    /// Heights in the shape our tables actually have, scaled down.
    const REAL_SHAPE: [usize; 6] = [5, 5, 6, 6, 4, 0];

    #[test]
    fn every_offset_is_aligned_to_its_own_height() {
        let layout = StackedLayout::build(&REAL_SHAPE, 8).unwrap();
        for place in layout.placements() {
            assert_eq!(
                place.offset % (1usize << place.num_vars),
                0,
                "offset {} is not aligned to 2^{}",
                place.offset,
                place.num_vars
            );
        }
    }

    #[test]
    fn placements_never_overlap() {
        let layout = StackedLayout::build(&REAL_SHAPE, 8).unwrap();
        let mut occupied: Vec<(usize, usize, usize)> = layout
            .placements()
            .iter()
            .map(|p| (p.poly, p.offset, p.offset + (1usize << p.num_vars)))
            .collect();
        occupied.sort();
        for pair in occupied.windows(2) {
            let (poly_a, _, end_a) = pair[0];
            let (poly_b, start_b, _) = pair[1];
            if poly_a == poly_b {
                assert!(end_a <= start_b, "ranges overlap: {:?}", pair);
            }
        }
    }

    #[test]
    fn everything_fits_inside_the_stacked_cube() {
        let n_stack = 8;
        let layout = StackedLayout::build(&REAL_SHAPE, n_stack).unwrap();
        for place in layout.placements() {
            assert!(place.offset + (1usize << place.num_vars) <= 1usize << n_stack);
        }
    }

    #[test]
    fn a_column_taller_than_the_cube_is_rejected() {
        let err = StackedLayout::build(&[3, 9], 8).unwrap_err();
        assert_eq!(
            err,
            Error::ColumnTallerThanStack {
                column_vars: 9,
                n_stack: 8
            }
        );
    }

    #[test]
    fn overflowing_columns_open_another_polynomial() {
        // Four columns of 2^7 need two cubes of 2^8.
        let layout = StackedLayout::build(&[7, 7, 7, 7], 8).unwrap();
        assert_eq!(layout.num_polys(), 2);
        assert_eq!(layout.placement(0).unwrap().poly, 0);
        assert_eq!(layout.placement(2).unwrap().poly, 1);
        assert_eq!(layout.placement(2).unwrap().offset, 0);
    }

    #[test]
    fn stacked_cells_land_where_the_layout_says() {
        let layout = StackedLayout::build(&REAL_SHAPE, 8).unwrap();
        let columns: Vec<Mle<F>> = REAL_SHAPE
            .iter()
            .enumerate()
            .map(|(i, &m)| column(1 << m, i as u64 + 1))
            .collect();
        let stacked = layout.stack(&borrow(&columns)).unwrap();

        for (col_idx, place) in layout.placements().iter().enumerate() {
            for (j, cell) in columns[col_idx].evals().iter().enumerate() {
                assert_eq!(
                    stacked[place.poly].evals()[place.offset + j],
                    *cell,
                    "column {col_idx}, cell {j}"
                );
            }
        }
    }

    #[test]
    fn padding_is_zero() {
        let layout = StackedLayout::build(&[3, 1], 5).unwrap();
        let columns = vec![column(8, 1), column(2, 2)];
        let stacked = layout.stack(&borrow(&columns)).unwrap();

        let (used, committed) = layout.occupancy();
        assert_eq!(used, 10);
        assert_eq!(committed, 32);

        let zeros = stacked[0]
            .evals()
            .iter()
            .filter(|v| **v == FE::zero())
            .count();
        // The 22 padding cells are zero (a data cell could coincidentally be
        // zero too, so this is a lower bound the layout must clear).
        assert!(zeros >= 22, "expected at least 22 zeros, got {zeros}");
    }

    /// The property the whole construction exists for.
    #[test]
    fn a_column_evaluation_is_the_stacked_evaluation_at_the_lifted_point() {
        let layout = StackedLayout::build(&REAL_SHAPE, 8).unwrap();
        let columns: Vec<Mle<F>> = REAL_SHAPE
            .iter()
            .enumerate()
            .map(|(i, &m)| column(1 << m, i as u64 + 1))
            .collect();
        let stacked = layout.stack(&borrow(&columns)).unwrap();

        for (col_idx, place) in layout.placements().iter().enumerate() {
            let mle = &columns[col_idx];
            // A point off the cube, where a wrong lift would show up.
            let z: Vec<FE> = (0..place.num_vars)
                .map(|k| FE::from(1000 + k as u64 + col_idx as u64))
                .collect();

            let direct = mle.evaluate(&z).unwrap();
            let lifted = place.point_in_stacked(&z).unwrap();
            let via_stack = stacked[place.poly].evaluate(&lifted).unwrap();

            assert_eq!(direct, via_stack, "column {col_idx}");
        }
    }

    #[test]
    fn the_lifted_point_has_the_stacked_arity() {
        let layout = StackedLayout::build(&REAL_SHAPE, 8).unwrap();
        let place = layout.placement(4).unwrap();
        let z = vec![FE::from(3); place.num_vars];
        assert_eq!(place.point_in_stacked(&z).unwrap().len(), 8);
    }

    #[test]
    fn prefix_bits_are_the_offset_index_most_significant_first() {
        // A 2^2 column at offset 20 in a 2^5 cube: 20/4 = 5 = 0b101.
        let place = Placement {
            poly: 0,
            offset: 20,
            num_vars: 2,
            n_stack: 5,
        };
        assert_eq!(place.prefix_bits(), vec![true, false, true]);
    }

    #[test]
    fn a_single_row_column_still_lifts() {
        // HALT is one row: no variables of its own, all prefix.
        let layout = StackedLayout::build(&[0], 4).unwrap();
        let columns = vec![Mle::new(vec![FE::from(42)]).unwrap()];
        let stacked = layout.stack(&borrow(&columns)).unwrap();
        let place = layout.placement(0).unwrap();

        let lifted = place.point_in_stacked::<F>(&[]).unwrap();
        assert_eq!(lifted.len(), 4);
        assert_eq!(stacked[0].evaluate(&lifted).unwrap(), FE::from(42));
    }

    #[test]
    fn lifting_a_point_of_the_wrong_arity_is_an_error() {
        let layout = StackedLayout::build(&[3], 5).unwrap();
        let place = layout.placement(0).unwrap();
        assert_eq!(
            place.point_in_stacked(&[FE::one()]).unwrap_err(),
            Error::VariableCountMismatch {
                expected: 3,
                got: 1
            }
        );
    }

    #[test]
    fn widest_first_wastes_less_than_interleaved() {
        // Same columns, two orders: alignment padding depends on the order.
        let interleaved = StackedLayout::build(&[0, 4, 0, 4], 6).unwrap();
        let widest_first = StackedLayout::build(&[4, 4, 0, 0], 6).unwrap();

        let end_of = |l: &StackedLayout| -> usize {
            l.placements()
                .iter()
                .map(|p| p.offset + (1usize << p.num_vars))
                .max()
                .unwrap()
        };
        assert!(
            end_of(&widest_first) < end_of(&interleaved),
            "widest-first should pack tighter"
        );
    }
}
