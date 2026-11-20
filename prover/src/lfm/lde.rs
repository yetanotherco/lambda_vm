//! Low-degree extension, emitted as machine instructions.
//!
//! A preprocessed column reaches its Merkle commitment as evaluations on the
//! LDE domain, and production gets there in two steps —
//! `Polynomial::interpolate_fft` then `evaluate_polynomial_on_lde_domain`
//! (`tables/register.rs::commit_register_columns` is one caller of exactly this
//! pair). Any leg that must DERIVE a preprocessed commitment rather than read
//! it has to emit that transform, because the values are what the derivation
//! binds; hinting the extended column would hand the prover a degree of freedom
//! the protocol does not give them.
//!
//! Everything here is shape-static. The domain size, the blowup, the coset
//! offset and therefore every twiddle are compile-time constants of the emitted
//! program: host-side `for` loops unroll and nothing loop-shaped reaches the
//! machine, exactly as in [`super::edsl`].
//!
//! ## Why cosets rather than one big transform
//!
//! Production zero-pads the `n` coefficients to `n·blowup` and runs a single
//! transform of that size. Emitting that shape would cost
//! `(n·blowup)/2 · log₂(n·blowup)` butterflies. Splitting the output domain
//! into its `blowup` cosets of the size-`n` subgroup instead costs
//! `blowup · (n/2 · log₂ n)` butterflies plus `blowup · n` scaling
//! multiplications. At two `LFM_BALU` rows per butterfly and one per scaling
//! (see [`butterfly`]) that is `n·log₂n + blowup·(n + n·log₂n)` rows per column
//! against `n·log₂n + n + n·blowup·log₂(n·blowup)` — **9,088 against 11,264**
//! at the register shape (`n = 128`, blowup 8), and the gap widens with the
//! blowup.
//!
//! Both are arithmetic, not measurement, and both are PER COLUMN;
//! `machine_tests::register_derivation_cost` asserts the emitted total (two
//! columns, 18,176 rows at that shape) against the first formula. The two
//! schemes evaluate the same polynomial on the same points, and the
//! differential tests against production are what says so.
//!
//! ## What this cannot see
//!
//! The emitter is validated by differential tests against production's own
//! `interpolate_fft`/`evaluate_polynomial_on_lde_domain` pair over the field
//! elements those functions produce. It says nothing about domains whose size
//! exceeds Goldilocks' two-adicity (`root_of_unity` panics there rather than
//! emitting a wrong program), and nothing about extension-field columns —
//! preprocessed columns are base-field throughout.

use math::fft::bit_reversing::reverse_index;
use math::field::traits::IsFFTField;

use crate::tables::types::{FE, GoldilocksField};

use super::builder::{Felt, LfmBuilder};

/// The `2^log_n`-th root of unity production's FFT uses.
///
/// `get_primitive_root_of_unity` is the same entry point `LayerTwiddles` builds
/// its twiddles from, and it is defined by repeated squaring of the field's
/// two-adic generator, so `root_of_unity(k + 1)² == root_of_unity(k)`. That
/// nesting is what lets the coset decomposition below index the big domain with
/// the small domain's root.
pub fn root_of_unity(log_n: u32) -> FE {
    GoldilocksField::get_primitive_root_of_unity(log_n as u64)
        .expect("the LDE domain must fit Goldilocks' two-adicity")
}

/// One radix-2 butterfly: `(u + w·x, u − w·x)`.
///
/// `w` is a program constant, so the subtracting half is `mul_add` against the
/// interned constant `−w` rather than a separate negation — two `LFM_BALU` rows
/// per butterfly, not three. At `w = 1` (every level-1 butterfly, and the first
/// of every later block) there is no multiplication at all.
fn butterfly(b: &mut LfmBuilder, u: Felt, x: Felt, w: FE) -> (Felt, Felt) {
    if w == FE::one() {
        (b.add(u, x), b.sub(u, x))
    } else {
        let pos = b.felt_const(w);
        let neg = b.felt_const(-w);
        (b.mul_add(pos, x, u), b.mul_add(neg, x, u))
    }
}

/// Decimation-in-time radix-2 transform: `out[m] = Σᵢ input[i]·root^(i·m)`.
///
/// `input` is in natural order (the bit-reverse the algorithm needs is a
/// host-side index permutation and costs nothing). Pass `root = ω_n` for the
/// forward direction and `root = ω_n⁻¹` for the inverse — the inverse's `1/n`
/// is NOT applied here, so callers that follow it with a scaling pass fold the
/// factor into their own constants.
fn dit(b: &mut LfmBuilder, input: &[Felt], root: FE) -> Vec<Felt> {
    let n = input.len();
    let log_n = n.trailing_zeros();
    let mut a: Vec<Felt> = (0..n).map(|i| input[reverse_index(i, n as u64)]).collect();
    for s in 1..=log_n {
        let m = 1usize << s;
        let step = root.pow(n / m);
        for k in (0..n).step_by(m) {
            let mut w = FE::one();
            for j in 0..m / 2 {
                let (hi, lo) = butterfly(b, a[k + j], a[k + j + m / 2], w);
                a[k + j] = hi;
                a[k + j + m / 2] = lo;
                w *= step;
            }
        }
    }
    a
}

/// Emit the low-degree extension of a column given by its `n` evaluations on
/// the size-`n` subgroup, onto `coset_offset · ⟨ω_{n·blowup}⟩`.
///
/// The result is in NATURAL domain order — output `j` is the value at
/// `coset_offset · ω_{n·blowup}^j` — which is the layout
/// `stark::commitment::commit_bit_reversed` consumes (it applies the
/// bit-reversal itself).
///
/// ## The decomposition
///
/// With `c = iFFT(values)` the polynomial's coefficients and
/// `s_k = coset_offset · ω_{n·blowup}^k`, domain index `j = k + blowup·m`
/// carries the point `s_k · ω_n^m`, so the `k`-th coset is one size-`n` forward
/// transform of `c` scaled by `s_k^i`. The interpolation's `1/n` rides along in
/// those scaling constants, which is why the inverse pass emits no scaling of
/// its own.
pub fn coset_lde(
    b: &mut LfmBuilder,
    values: &[Felt],
    blowup: usize,
    coset_offset: FE,
) -> Vec<Felt> {
    let n = values.len();
    assert!(
        n.is_power_of_two(),
        "the interpolation domain is a subgroup"
    );
    assert!(
        blowup.is_power_of_two() && blowup > 0,
        "blowup is a power of two"
    );

    let log_n = n.trailing_zeros();
    let omega_n = root_of_unity(log_n);
    let omega_big = root_of_unity(log_n + blowup.trailing_zeros());

    // `n · coefficients`: the 1/n is folded into the per-coset scaling below.
    let scaled_coeffs = dit(
        b,
        values,
        omega_n.inv().expect("a root of unity is invertible"),
    );

    let n_inv = FE::from(n as u64)
        .inv()
        .expect("the domain size is nonzero in Goldilocks");

    let mut out: Vec<Option<Felt>> = vec![None; n * blowup];
    for k in 0..blowup {
        let s = coset_offset * omega_big.pow(k);
        let mut weight = n_inv;
        let coset_coeffs: Vec<Felt> = (0..n)
            .map(|i| {
                let c = b.felt_const(weight);
                let scaled = b.mul(scaled_coeffs[i], c);
                weight *= s;
                scaled
            })
            .collect();
        for (m, value) in dit(b, &coset_coeffs, omega_n).into_iter().enumerate() {
            out[k + blowup * m] = Some(value);
        }
    }
    out.into_iter()
        .map(|v| v.expect("every domain index is written exactly once"))
        .collect()
}
