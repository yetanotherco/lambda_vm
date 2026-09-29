//! The register multiplexer's constant coefficients.
//!
//! `f_j` is the Lagrange basis polynomial on `0..=N+1` with `f_j(j) = 1`. It is
//! degree-split as `f_j = Σ_k x^{e_k} f_{j,k}` with `e_0 = 0`,
//! `e_k = (D − 1) + (k − 1)(D − 2)`, `deg f_{j,0} ≤ D − 2` and
//! `deg f_{j,k} ≤ D − 3` for `k ≥ 1`, so that `imm0 · reg · f_j(x)` stays of
//! degree `D` once `x^{e_k}` is a committed column. `g(x) = Π (x − j)` range-checks
//! register indices using one extra coefficient in the last split polynomial.

use super::isa::NUM_REGS;
use crate::tables::types::FE;

/// Maximum constraint degree.
pub const D: usize = 5;
/// Number of split polynomials after the first.
pub const T: usize = 1;

const _: () = assert!(D >= 3 && T >= 1);
const _: () = assert!(
    NUM_REGS <= (D - 1) + T * (D - 2),
    "too many registers for (D, T)"
);

/// The exponent of `x` that multiplies split polynomial `k`.
pub const fn split_exponent(k: usize) -> usize {
    if k == 0 {
        0
    } else {
        (D - 1) + (k - 1) * (D - 2)
    }
}

/// Coefficients as `[k][l]`: the coefficient of `x^{split_exponent(k) + l}`.
pub type Split = [[FE; D]; T + 1];

fn split(coeffs: &[FE]) -> Split {
    let mut out = [[FE::zero(); D]; T + 1];
    for (e, c) in coeffs.iter().enumerate() {
        let (k, l) = if e <= D - 2 {
            (0, e)
        } else if e == split_exponent(T) + (D - 2) {
            (T, D - 2)
        } else {
            let r = e - (D - 1);
            (1 + r / (D - 2), r % (D - 2))
        };
        assert!(
            k <= T,
            "polynomial of degree {} does not fit the split",
            coeffs.len() - 1
        );
        out[k][l] = *c;
    }
    out
}

fn poly_mul_linear(p: &[FE], root: FE) -> Vec<FE> {
    let mut out = vec![FE::zero(); p.len() + 1];
    for (i, c) in p.iter().enumerate() {
        out[i + 1] += *c;
        out[i] = out[i] - *c * root;
    }
    out
}

fn lagrange(j: usize) -> Vec<FE> {
    let mut num = vec![FE::one()];
    let mut den = FE::one();
    for m in (0..NUM_REGS).filter(|&m| m != j) {
        num = poly_mul_linear(&num, FE::from(m as u64));
        den *= FE::from(j as u64) - FE::from(m as u64);
    }
    let inv = den.inv().expect("distinct points");
    num.into_iter().map(|c| c * inv).collect()
}

#[derive(Clone, Debug)]
pub struct MuxTables {
    /// `mux[j]`: the split of `f_j`.
    pub mux: [Split; NUM_REGS],
    /// The split of `g`.
    pub range: Split,
}

impl MuxTables {
    pub fn new() -> Self {
        let mux = core::array::from_fn(|j| split(&lagrange(j)));
        let mut g = vec![FE::one()];
        for m in 0..NUM_REGS {
            g = poly_mul_linear(&g, FE::from(m as u64));
        }
        Self {
            mux,
            range: split(&g),
        }
    }
}

impl Default for MuxTables {
    fn default() -> Self {
        Self::new()
    }
}

/// The powers `x^{split_exponent(k)}` for `k = 0..=T`.
pub fn split_powers(x: FE) -> [FE; T + 1] {
    core::array::from_fn(|k| x.pow(split_exponent(k) as u64))
}

/// Evaluates a split polynomial at `x` the way the constraints do.
pub fn eval_split(s: &Split, x: FE) -> FE {
    let pows = split_powers(x);
    let mut acc = FE::zero();
    for (p, split) in pows.iter().zip(s) {
        for (l, c) in split.iter().enumerate() {
            acc += *p * c * x.pow(l as u64);
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mux_selects_one_register() {
        let t = MuxTables::new();
        for j in 0..NUM_REGS {
            for x in 0..NUM_REGS {
                let expected = if j == x { FE::one() } else { FE::zero() };
                assert_eq!(
                    eval_split(&t.mux[j], FE::from(x as u64)),
                    expected,
                    "f_{j}({x})"
                );
            }
        }
    }

    #[test]
    fn mux_split_respects_degree_budget() {
        let t = MuxTables::new();
        for s in &t.mux {
            assert!(s[0][D - 1] == FE::zero());
            for split in &s[1..] {
                assert!(split[D - 2] == FE::zero() && split[D - 1] == FE::zero());
            }
        }
        assert!(t.range[0][D - 1] == FE::zero());
        for split in &t.range[1..T] {
            assert!(split[D - 2] == FE::zero() && split[D - 1] == FE::zero());
        }
        assert!(t.range[T][D - 1] == FE::zero());
    }

    #[test]
    fn range_vanishes_exactly_on_register_indices() {
        let t = MuxTables::new();
        for x in 0..NUM_REGS as u64 {
            assert_eq!(eval_split(&t.range, FE::from(x)), FE::zero());
        }
        for x in [NUM_REGS as u64, NUM_REGS as u64 + 1, 1000, u64::MAX - 100] {
            assert_ne!(eval_split(&t.range, FE::from(x)), FE::zero(), "g({x})");
        }
    }
}
