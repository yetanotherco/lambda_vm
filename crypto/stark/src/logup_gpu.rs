//! GPU LogUp aux build: compile a table's bus interactions into a flat
//! descriptor the device fingerprint kernel can walk, plus a CPU evaluator that
//! mirrors the kernel exactly (the parity test pins them together).
//!
//! Fingerprint per interaction k at row i:
//!   lc = bus_id + Σ_e α^{alpha_idx(e)} · e(i)
//!   fp = z - lc
//! where each bus element e is `const + Σ_t coef_t · col_t[i]` in the base field
//! (Goldilocks), matching `BusValue::accumulate_fingerprint` /
//! `Packing::accumulate_fingerprint_with`.

use std::any::TypeId;

use crate::lookup::{
    compute_alpha_powers, BusInteraction, BusValue, LinearTerm, Multiplicity, Packing,
    LOGUP_CHALLENGE_ALPHA,
};
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::{IsField, IsSubFieldOf};

/// Minimum trace rows for the GPU aux-build path. Below this the CPU build wins
/// (dispatch + upload overhead). Correctness is unaffected: the fallback is
/// byte identical.
const GPU_LOGUP_MIN_ROWS: usize = 1 << 10;

/// Goldilocks modulus 2^64 - 2^32 + 1. Coefficients are stored canonical so the
/// device path does plain Goldilocks arithmetic.
const GOLDILOCKS_P: u64 = 0xFFFF_FFFF_0000_0001;

// Packing shift constants (powers of two), canonical Goldilocks.
const SHIFT_8: u64 = 1 << 8;
const SHIFT_16: u64 = 1 << 16;
const SHIFT_24: u64 = 1 << 24;

/// Reduce a signed coefficient into canonical Goldilocks, matching
/// `FieldElement::<Goldilocks>::from(i64)`. |c| < 2^63 < p so no overflow.
fn i64_to_canonical(c: i64) -> u64 {
    if c >= 0 {
        c as u64 % GOLDILOCKS_P
    } else {
        GOLDILOCKS_P - (c.unsigned_abs() % GOLDILOCKS_P)
    }
}

/// Canonical Goldilocks negation.
fn neg_canonical(x: u64) -> u64 {
    if x == 0 {
        0
    } else {
        GOLDILOCKS_P - x
    }
}

/// Committed-pair / absorbed split, mirroring `lookup::split_interactions`.
fn split_interactions(num: usize) -> (usize, usize) {
    if num <= 2 {
        (0, num)
    } else if num % 2 == 1 {
        ((num - 1) / 2, 1)
    } else {
        ((num - 2) / 2, 2)
    }
}

/// Encode a multiplicity as a signed linear form `const + Σ coef·col` (sign
/// baked in: negated for receivers so `term = m'·recip` needs no extra sign).
/// Mirrors `Multiplicity::evaluate_with` for every variant.
fn encode_signed_multiplicity(m: &Multiplicity, is_sender: bool) -> (u64, Vec<(u64, u32)>) {
    let mut cst: u128 = 0;
    let mut terms: Vec<(u64, u32)> = Vec::new();
    match m {
        Multiplicity::One => cst = 1,
        Multiplicity::Column(c) => terms.push((1, *c as u32)),
        Multiplicity::Sum(a, b) => {
            terms.push((1, *a as u32));
            terms.push((1, *b as u32));
        }
        Multiplicity::Negated(c) => {
            cst = 1;
            terms.push((neg_canonical(1), *c as u32));
        }
        Multiplicity::Diff(a, b) => {
            terms.push((1, *a as u32));
            terms.push((neg_canonical(1), *b as u32));
        }
        Multiplicity::Sum3(a, b, c) => {
            terms.push((1, *a as u32));
            terms.push((1, *b as u32));
            terms.push((1, *c as u32));
        }
        Multiplicity::Linear(ts) => {
            for t in ts {
                match *t {
                    LinearTerm::Column {
                        coefficient,
                        column,
                    } => terms.push((i64_to_canonical(coefficient), column as u32)),
                    LinearTerm::ColumnUnsigned {
                        coefficient,
                        column,
                    } => terms.push((coefficient % GOLDILOCKS_P, column as u32)),
                    LinearTerm::Constant(v) => cst += i64_to_canonical(v) as u128,
                }
            }
        }
    }
    let mut cst = (cst % GOLDILOCKS_P as u128) as u64;
    if !is_sender {
        cst = neg_canonical(cst);
        for t in terms.iter_mut() {
            t.0 = neg_canonical(t.0);
        }
    }
    (cst, terms)
}

/// Flat descriptor for one table's fingerprints. CSR layout: interactions index
/// into elements, elements index into terms. All coefficients canonical
/// Goldilocks. Ready to upload to the device fingerprint kernel.
#[derive(Clone, Debug, Default)]
pub struct FingerprintDescriptor {
    pub num_interactions: usize,
    /// `alpha_powers` must hold this many powers `[1, α, ... α^{len-1}]`.
    pub alpha_powers_len: usize,
    /// Per interaction: the α^0 (bus id) constant.
    pub bus_ids: Vec<u64>,
    /// Per interaction CSR offsets into the element arrays (len + 1).
    pub elem_offsets: Vec<u32>,
    /// Per element: the α power index (>= 1).
    pub elem_alpha_idx: Vec<u32>,
    /// Per element: additive constant (canonical; 0 for packings).
    pub elem_const: Vec<u64>,
    /// Per element CSR offsets into the term arrays (len + 1).
    pub term_offsets: Vec<u32>,
    /// Per term: coefficient (canonical Goldilocks).
    pub term_coef: Vec<u64>,
    /// Per term: main column index.
    pub term_col: Vec<u32>,

    // --- term-combine (K3) data ---
    /// Number of output term columns = committed pairs + 1 virtual.
    pub num_out_cols: usize,
    /// Per interaction: signed multiplicity constant (negated for receivers).
    pub mult_const: Vec<u64>,
    /// Per interaction CSR offsets into the multiplicity term arrays (len + 1).
    pub mult_term_offsets: Vec<u32>,
    /// Per multiplicity term: coefficient (signed, canonical Goldilocks).
    pub mult_term_coef: Vec<u64>,
    /// Per multiplicity term: main column index.
    pub mult_term_col: Vec<u32>,
    /// Per output column CSR offsets into `out_col_interactions` (len + 1).
    pub out_col_offsets: Vec<u32>,
    /// Interaction indices grouped per output column.
    pub out_col_interactions: Vec<u32>,
}

impl FingerprintDescriptor {
    fn push_element(&mut self, alpha_idx: u32, const_val: u64, terms: &[(u64, u32)]) {
        self.elem_alpha_idx.push(alpha_idx);
        self.elem_const.push(const_val);
        for &(coef, col) in terms {
            self.term_coef.push(coef);
            self.term_col.push(col);
        }
        self.term_offsets.push(self.term_coef.len() as u32);
    }

    /// Expand one `BusValue` into elements starting at `alpha_off`; return the
    /// number of bus elements (alpha powers) consumed. Mirrors
    /// `BusValue::accumulate_fingerprint` exactly.
    fn push_bus_value(&mut self, bv: &BusValue, alpha_off: u32) -> u32 {
        match bv {
            BusValue::Packed {
                start_column,
                packing,
            } => {
                let c = *start_column as u32;
                match packing {
                    Packing::Direct => self.push_element(alpha_off, 0, &[(1, c)]),
                    Packing::Word2L => {
                        self.push_element(alpha_off, 0, &[(1, c), (SHIFT_16, c + 1)])
                    }
                    Packing::Word4L => self.push_element(
                        alpha_off,
                        0,
                        &[(1, c), (SHIFT_8, c + 1), (SHIFT_16, c + 2), (SHIFT_24, c + 3)],
                    ),
                    Packing::DWordWL => {
                        self.push_element(alpha_off, 0, &[(1, c)]);
                        self.push_element(alpha_off + 1, 0, &[(1, c + 1)]);
                    }
                    Packing::DWordHHW => {
                        self.push_element(alpha_off, 0, &[(1, c)]);
                        self.push_element(alpha_off + 1, 0, &[(1, c + 1), (SHIFT_16, c + 2)]);
                    }
                    Packing::DWordWHH => {
                        self.push_element(alpha_off, 0, &[(1, c), (SHIFT_16, c + 1)]);
                        self.push_element(alpha_off + 1, 0, &[(1, c + 2)]);
                    }
                    Packing::DWordHL => {
                        self.push_element(alpha_off, 0, &[(1, c), (SHIFT_16, c + 1)]);
                        self.push_element(alpha_off + 1, 0, &[(1, c + 2), (SHIFT_16, c + 3)]);
                    }
                    Packing::DWordBL => {
                        self.push_element(
                            alpha_off,
                            0,
                            &[(1, c), (SHIFT_8, c + 1), (SHIFT_16, c + 2), (SHIFT_24, c + 3)],
                        );
                        self.push_element(
                            alpha_off + 1,
                            0,
                            &[
                                (1, c + 4),
                                (SHIFT_8, c + 5),
                                (SHIFT_16, c + 6),
                                (SHIFT_24, c + 7),
                            ],
                        );
                    }
                    Packing::QuadHL => {
                        for i in 0..4u32 {
                            let cc = c + i * 2;
                            self.push_element(alpha_off + i, 0, &[(1, cc), (SHIFT_16, cc + 1)]);
                        }
                    }
                    Packing::QuadWL => {
                        for i in 0..4u32 {
                            self.push_element(alpha_off + i, 0, &[(1, c + i)]);
                        }
                    }
                }
                packing.num_bus_elements() as u32
            }
            BusValue::Linear(terms) => {
                let mut const_val: u128 = 0;
                let mut t: Vec<(u64, u32)> = Vec::new();
                for term in terms {
                    match *term {
                        LinearTerm::Column {
                            coefficient,
                            column,
                        } => t.push((i64_to_canonical(coefficient), column as u32)),
                        LinearTerm::ColumnUnsigned {
                            coefficient,
                            column,
                        } => t.push((coefficient % GOLDILOCKS_P, column as u32)),
                        LinearTerm::Constant(value) => {
                            const_val += i64_to_canonical(value) as u128;
                        }
                    }
                }
                self.push_element(alpha_off, (const_val % GOLDILOCKS_P as u128) as u64, &t);
                1
            }
        }
    }
}

/// Compile a table's interactions into a [`FingerprintDescriptor`].
pub fn build_fingerprint_descriptor(interactions: &[BusInteraction]) -> FingerprintDescriptor {
    let mut d = FingerprintDescriptor {
        num_interactions: interactions.len(),
        ..Default::default()
    };
    d.elem_offsets.push(0);
    d.term_offsets.push(0);
    d.mult_term_offsets.push(0);
    let mut max_bus_elements = 0usize;
    for it in interactions {
        d.bus_ids.push(it.bus_id % GOLDILOCKS_P);
        max_bus_elements = max_bus_elements.max(it.num_bus_elements());
        let mut alpha_off = 1u32;
        for bv in &it.values {
            alpha_off += d.push_bus_value(bv, alpha_off);
        }
        d.elem_offsets.push(d.elem_alpha_idx.len() as u32);

        // Signed multiplicity for the term combine.
        let (cst, terms) = encode_signed_multiplicity(&it.multiplicity, it.is_sender);
        d.mult_const.push(cst);
        for (coef, col) in terms {
            d.mult_term_coef.push(coef);
            d.mult_term_col.push(col);
        }
        d.mult_term_offsets.push(d.mult_term_coef.len() as u32);
    }
    d.alpha_powers_len = max_bus_elements;

    // Output term columns: committed pair p = {2p, 2p+1}; the trailing 1-2
    // absorbed interactions form one virtual column.
    let (committed_pairs, absorbed) = split_interactions(interactions.len());
    d.out_col_offsets.push(0);
    for p in 0..committed_pairs {
        d.out_col_interactions.push(2 * p as u32);
        d.out_col_interactions.push(2 * p as u32 + 1);
        d.out_col_offsets.push(d.out_col_interactions.len() as u32);
    }
    for k in (interactions.len() - absorbed)..interactions.len() {
        d.out_col_interactions.push(k as u32);
    }
    d.out_col_offsets.push(d.out_col_interactions.len() as u32);
    d.num_out_cols = committed_pairs + 1;
    d
}

impl FingerprintDescriptor {
    /// Borrow the static arrays as the math-cuda flat descriptor (challenges
    /// `alpha_powers`/`z` are passed separately at call time).
    pub fn as_cuda(&self) -> math_cuda::logup::LogupDescriptor<'_> {
        math_cuda::logup::LogupDescriptor {
            num_interactions: self.num_interactions,
            bus_ids: &self.bus_ids,
            elem_offsets: &self.elem_offsets,
            elem_alpha_idx: &self.elem_alpha_idx,
            elem_const: &self.elem_const,
            term_offsets: &self.term_offsets,
            term_coef: &self.term_coef,
            term_col: &self.term_col,
            num_out_cols: self.num_out_cols,
            out_col_offsets: &self.out_col_offsets,
            out_col_interactions: &self.out_col_interactions,
            mult_const: &self.mult_const,
            mult_term_offsets: &self.mult_term_offsets,
            mult_term_coef: &self.mult_term_coef,
            mult_term_col: &self.mult_term_col,
        }
    }
}

/// GPU aux-build term columns. Returns `(committed_columns, virtual_column)`
/// byte identical to the CPU path, or `None` to fall back (non Goldilocks,
/// below threshold, no GPU, or a GPU error). The committed columns are written
/// to the aux trace; the virtual column feeds the accumulated column.
pub fn try_build_term_columns_gpu<F, E>(
    interactions: &[BusInteraction],
    main_cols: &[Vec<FieldElement<F>>],
    trace_len: usize,
    challenges: &[FieldElement<E>],
) -> Option<(Vec<Vec<FieldElement<E>>>, Vec<FieldElement<E>>)>
where
    F: IsField + 'static,
    E: IsField + 'static,
{
    if TypeId::of::<F>() != TypeId::of::<GoldilocksField>()
        || TypeId::of::<E>() != TypeId::of::<Degree3GoldilocksExtensionField>()
    {
        return None;
    }
    if trace_len < GPU_LOGUP_MIN_ROWS || main_cols.is_empty() || interactions.is_empty() {
        return None;
    }
    // Escape hatch for A/B measurement: force the CPU aux build.
    if std::env::var_os("LAMBDA_VM_NO_GPU_LOGUP").is_some() {
        return None;
    }

    let desc = build_fingerprint_descriptor(interactions);
    if desc.num_out_cols == 0 {
        return None;
    }

    // main trace -> column-major u64. SAFETY: F == Goldilocks (repr(u64)).
    let num_cols = main_cols.len();
    let mut main_flat = vec![0u64; num_cols * trace_len];
    for (c, col) in main_cols.iter().enumerate() {
        for (r, e) in col.iter().enumerate() {
            main_flat[c * trace_len + r] = unsafe { *(e.value() as *const _ as *const u64) };
        }
    }

    // z + alpha powers. SAFETY: E == ext3 (repr [u64; 3]).
    let z_arr = unsafe { *(challenges[0].value() as *const _ as *const [u64; 3]) };
    let alpha = &challenges[LOGUP_CHALLENGE_ALPHA];
    let alpha_powers = compute_alpha_powers(alpha, desc.alpha_powers_len);
    let mut alpha_flat = vec![0u64; alpha_powers.len() * 3];
    for (i, p) in alpha_powers.iter().enumerate() {
        let l = unsafe { *(p.value() as *const _ as *const [u64; 3]) };
        alpha_flat[i * 3..i * 3 + 3].copy_from_slice(&l);
    }

    let md = desc.as_cuda();
    let term_flat =
        math_cuda::logup::logup_term_columns(&main_flat, trace_len, &md, &alpha_flat, z_arr).ok()?;
    crate::gpu_lde::GPU_LOGUP_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    // term_flat layout [(col*trace_len + row)*3 + limb]; last column is virtual.
    let mut cols: Vec<Vec<FieldElement<E>>> = Vec::with_capacity(desc.num_out_cols);
    for col in 0..desc.num_out_cols {
        let lo = col * trace_len * 3;
        cols.push(crate::gpu_lde::u64_to_ext3_vec::<E>(&term_flat[lo..lo + trace_len * 3]));
    }
    let virtual_column = cols.pop().unwrap();
    Some((cols, virtual_column))
}

/// GPU-resident aux build: produces the row-major aux columns on device (fed
/// straight to the aux LDE, no host round-trip) + the table contribution `L`.
/// Returns `None` to fall back (non Goldilocks, below threshold, no GPU, GPU
/// error). This is the residency path that avoids the term-column download.
pub fn try_build_aux_resident_gpu<F, E>(
    interactions: &[BusInteraction],
    main_cols: &[Vec<FieldElement<F>>],
    trace_len: usize,
    challenges: &[FieldElement<E>],
) -> Option<math_cuda::logup::ResidentAux>
where
    F: IsField + 'static,
    E: IsField + 'static,
{
    if TypeId::of::<F>() != TypeId::of::<GoldilocksField>()
        || TypeId::of::<E>() != TypeId::of::<Degree3GoldilocksExtensionField>()
    {
        return None;
    }
    if trace_len < GPU_LOGUP_MIN_ROWS || main_cols.is_empty() || interactions.is_empty() {
        return None;
    }
    if std::env::var_os("LAMBDA_VM_NO_GPU_LOGUP").is_some() {
        return None;
    }
    let desc = build_fingerprint_descriptor(interactions);
    if desc.num_out_cols == 0 {
        return None;
    }

    let num_cols = main_cols.len();
    let mut main_flat = vec![0u64; num_cols * trace_len];
    for (c, col) in main_cols.iter().enumerate() {
        for (r, e) in col.iter().enumerate() {
            main_flat[c * trace_len + r] = unsafe { *(e.value() as *const _ as *const u64) };
        }
    }
    let z_arr = unsafe { *(challenges[0].value() as *const _ as *const [u64; 3]) };
    let alpha = &challenges[LOGUP_CHALLENGE_ALPHA];
    let alpha_powers = compute_alpha_powers(alpha, desc.alpha_powers_len);
    let mut alpha_flat = vec![0u64; alpha_powers.len() * 3];
    for (i, p) in alpha_powers.iter().enumerate() {
        let l = unsafe { *(p.value() as *const _ as *const [u64; 3]) };
        alpha_flat[i * 3..i * 3 + 3].copy_from_slice(&l);
    }
    // 1/N embedded in ext3 (matches the CPU offset = L * FieldElement::<E>::from(N).inv()).
    let inv_n_e = FieldElement::<E>::from(trace_len as u64).inv().ok()?;
    let inv_n = unsafe { *(inv_n_e.value() as *const _ as *const [u64; 3]) };

    let be = math_cuda::device::backend().ok()?;
    let stream = be.next_stream();
    let md = desc.as_cuda();
    let ra = math_cuda::logup::logup_aux_resident(
        &main_flat, trace_len, &md, &alpha_flat, z_arr, inv_n, &stream,
    )
    .ok()?;
    crate::gpu_lde::GPU_LOGUP_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Some(ra)
}

/// CPU reference: evaluate the fingerprint of interaction `k` at one row from
/// the descriptor. The device kernel performs the identical computation. Used by
/// the parity test and as the spec the kernel mirrors.
pub fn eval_fingerprint<'a, F, E>(
    d: &FingerprintDescriptor,
    k: usize,
    get_col: impl Fn(usize) -> &'a FieldElement<F>,
    alpha_powers: &[FieldElement<E>],
    z: &FieldElement<E>,
) -> FieldElement<E>
where
    F: IsField + IsSubFieldOf<E> + 'a,
    E: IsField,
{
    let mut lc = FieldElement::<E>::from(d.bus_ids[k]);
    let e_lo = d.elem_offsets[k] as usize;
    let e_hi = d.elem_offsets[k + 1] as usize;
    for e in e_lo..e_hi {
        let mut base = FieldElement::<F>::from(d.elem_const[e]);
        let t_lo = d.term_offsets[e] as usize;
        let t_hi = d.term_offsets[e + 1] as usize;
        for t in t_lo..t_hi {
            let coef = FieldElement::<F>::from(d.term_coef[t]);
            base = base + &coef * get_col(d.term_col[t] as usize);
        }
        lc = lc + &base * &alpha_powers[d.elem_alpha_idx[e] as usize];
    }
    z - &lc
}

/// CPU reference: term column `out_col` at `row` = Σ over the column's
/// interactions of `signed_multiplicity · reciprocal`. `reciprocals` is laid out
/// `[k * num_rows + row]` (batch inverse of the fingerprints). Mirrors the K3
/// kernel and the production term/accumulate combine.
pub fn eval_term<'a, F, E>(
    d: &FingerprintDescriptor,
    out_col: usize,
    row: usize,
    num_rows: usize,
    get_col: impl Fn(usize) -> &'a FieldElement<F>,
    reciprocals: &[FieldElement<E>],
) -> FieldElement<E>
where
    F: IsField + IsSubFieldOf<E> + 'a,
    E: IsField,
{
    let mut term = FieldElement::<E>::zero();
    let lo = d.out_col_offsets[out_col] as usize;
    let hi = d.out_col_offsets[out_col + 1] as usize;
    for ki in lo..hi {
        let k = d.out_col_interactions[ki] as usize;
        let mut m = FieldElement::<F>::from(d.mult_const[k]);
        let t_lo = d.mult_term_offsets[k] as usize;
        let t_hi = d.mult_term_offsets[k + 1] as usize;
        for t in t_lo..t_hi {
            m = m + &FieldElement::<F>::from(d.mult_term_coef[t]) * get_col(d.mult_term_col[t] as usize);
        }
        term = term + &m * &reciprocals[k * num_rows + row];
    }
    term
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lookup::{compute_alpha_powers, PackingShifts};
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
    use math::field::goldilocks::GoldilocksField;

    type F = GoldilocksField;
    type E = Degree3GoldilocksExtensionField;

    // Reference fingerprint via the production accumulate path (source of truth).
    fn reference_fp(
        it: &BusInteraction,
        main: &[Vec<FieldElement<F>>],
        row: usize,
        alpha_powers: &[FieldElement<E>],
        z: &FieldElement<E>,
        shifts: &PackingShifts<F>,
    ) -> FieldElement<E> {
        let mut lc = FieldElement::<E>::from(it.bus_id);
        let mut off = 1usize;
        for bv in &it.values {
            off += bv.accumulate_fingerprint(main, row, alpha_powers, off, &mut lc, shifts);
        }
        z - &lc
    }

    fn lcg(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *state
    }

    #[test]
    fn descriptor_fingerprint_matches_accumulate_path() {
        use crate::lookup::Multiplicity::One;
        let interactions = vec![
            BusInteraction::sender(0u64, One, Packing::Direct.columns(&[0])),
            BusInteraction::sender(1u64, One, Packing::Word2L.columns(&[1])),
            BusInteraction::sender(2u64, One, Packing::Word4L.columns(&[1])),
            BusInteraction::sender(3u64, One, Packing::DWordWL.columns(&[2])),
            BusInteraction::sender(4u64, One, Packing::DWordHHW.columns(&[0])),
            BusInteraction::sender(5u64, One, Packing::DWordWHH.columns(&[0])),
            BusInteraction::sender(6u64, One, Packing::DWordHL.columns(&[0])),
            BusInteraction::sender(7u64, One, Packing::QuadWL.columns(&[0])),
            BusInteraction::sender(8u64, One, Packing::QuadHL.columns(&[0])),
            BusInteraction::sender(
                9u64,
                One,
                vec![
                    BusValue::linear(vec![
                        LinearTerm::Column { coefficient: 3, column: 1 },
                        LinearTerm::Column { coefficient: -2, column: 4 },
                        LinearTerm::Constant(42),
                    ]),
                    BusValue::column(5),
                ],
            ),
        ];

        let num_cols = 8;
        let num_rows = 16;
        let mut st = 0x1234_5678_9abc_def0u64;
        let main: Vec<Vec<FieldElement<F>>> = (0..num_cols)
            .map(|_| (0..num_rows).map(|_| FieldElement::<F>::from(lcg(&mut st))).collect())
            .collect();

        // Base-embedded random alpha/z: distinct powers exercise every coef/index.
        let alpha = FieldElement::<E>::from(lcg(&mut st));
        let z = FieldElement::<E>::from(lcg(&mut st));
        let shifts = PackingShifts::<F>::new();

        let desc = build_fingerprint_descriptor(&interactions);
        let max_be = interactions.iter().map(|i| i.num_bus_elements()).max().unwrap();
        assert_eq!(desc.alpha_powers_len, max_be);
        let alpha_powers = compute_alpha_powers(&alpha, max_be);

        for (k, it) in interactions.iter().enumerate() {
            for row in 0..num_rows {
                let got = eval_fingerprint::<F, E>(&desc, k, |c| &main[c][row], &alpha_powers, &z);
                let want = reference_fp(it, &main, row, &alpha_powers, &z, &shifts);
                assert_eq!(got, want, "fingerprint mismatch interaction {k} row {row}");
            }
        }
    }

    fn mk_ext3(st: &mut u64) -> FieldElement<E> {
        FieldElement::<E>::new([
            FieldElement::<F>::from(lcg(st)),
            FieldElement::<F>::from(lcg(st)),
            FieldElement::<F>::from(lcg(st)),
        ])
    }

    fn limbs(e: &FieldElement<E>) -> [u64; 3] {
        let v = e.value();
        [*v[0].value(), *v[1].value(), *v[2].value()]
    }

    // GPU fingerprint kernel vs the CPU evaluator, byte for byte (full ext3
    // alpha/z so mul_base is exercised). Runs on the GPU box.
    #[test]
    #[ignore = "requires GPU; run with --ignored"]
    fn gpu_fingerprints_match_cpu() {
        use crate::lookup::Multiplicity::One;
        let interactions = vec![
            BusInteraction::sender(0u64, One, Packing::Direct.columns(&[0])),
            BusInteraction::sender(1u64, One, Packing::Word4L.columns(&[0])),
            BusInteraction::sender(2u64, One, Packing::DWordHL.columns(&[0])),
            BusInteraction::sender(3u64, One, Packing::QuadHL.columns(&[0])),
            BusInteraction::sender(
                4u64,
                One,
                vec![
                    BusValue::linear(vec![
                        LinearTerm::Column { coefficient: 3, column: 1 },
                        LinearTerm::Column { coefficient: -2, column: 2 },
                        LinearTerm::Constant(7),
                    ]),
                    BusValue::column(3),
                ],
            ),
        ];

        let num_cols = 8;
        let num_rows = 64;
        let mut st = 0xabcd_ef01_2345_6789u64;
        let main: Vec<Vec<FieldElement<F>>> = (0..num_cols)
            .map(|_| (0..num_rows).map(|_| FieldElement::<F>::from(lcg(&mut st))).collect())
            .collect();
        let alpha = mk_ext3(&mut st);
        let z = mk_ext3(&mut st);

        let desc = build_fingerprint_descriptor(&interactions);
        let alpha_powers = compute_alpha_powers(&alpha, desc.alpha_powers_len);

        // CPU reference, layout [(k*num_rows + row)*3 + limb].
        let mut cpu = vec![0u64; interactions.len() * num_rows * 3];
        for k in 0..interactions.len() {
            for row in 0..num_rows {
                let fp = eval_fingerprint::<F, E>(&desc, k, |c| &main[c][row], &alpha_powers, &z);
                let o = (k * num_rows + row) * 3;
                cpu[o..o + 3].copy_from_slice(&limbs(&fp));
            }
        }

        // Flatten GPU inputs.
        let mut main_flat = vec![0u64; num_cols * num_rows];
        for c in 0..num_cols {
            for r in 0..num_rows {
                main_flat[c * num_rows + r] = *main[c][r].value();
            }
        }
        let mut alpha_flat = vec![0u64; alpha_powers.len() * 3];
        for (i, p) in alpha_powers.iter().enumerate() {
            alpha_flat[i * 3..i * 3 + 3].copy_from_slice(&limbs(p));
        }

        let be = math_cuda::device::backend().unwrap();
        let stream = be.next_stream();
        let md = desc.as_cuda();
        let out_dev =
            math_cuda::logup::logup_fingerprints_dev(&main_flat, num_rows, &md, &alpha_flat, limbs(&z), &stream)
                .unwrap();
        let gpu: Vec<u64> = stream.clone_dtoh(&out_dev).unwrap();
        stream.synchronize().unwrap();

        assert_eq!(gpu, cpu, "GPU fingerprints mismatch CPU evaluator");
    }

    // Faithful reference for one term column: fingerprint every interaction,
    // batch invert, then Σ ±(multiplicity·recip). Mirrors compute_logup_term_column.
    fn reference_term_column(
        ints: &[&BusInteraction],
        main: &[Vec<FieldElement<F>>],
        num_rows: usize,
        alpha_powers: &[FieldElement<E>],
        z: &FieldElement<E>,
        shifts: &PackingShifts<F>,
    ) -> Vec<FieldElement<E>> {
        let mut fps: Vec<FieldElement<E>> = Vec::with_capacity(ints.len() * num_rows);
        for it in ints {
            for row in 0..num_rows {
                fps.push(reference_fp(it, main, row, alpha_powers, z, shifts));
            }
        }
        FieldElement::inplace_batch_inverse(&mut fps).unwrap();
        let mut out = vec![FieldElement::<E>::zero(); num_rows];
        for (row, slot) in out.iter_mut().enumerate() {
            let mut acc = FieldElement::<E>::zero();
            for (k, it) in ints.iter().enumerate() {
                let m = it.multiplicity.evaluate_at_row(main, row);
                let t = &m * &fps[k * num_rows + row];
                acc = acc + if it.is_sender { t } else { -t };
            }
            *slot = acc;
        }
        out
    }

    // Interaction set exercising committed pairs + virtual and several
    // multiplicity forms (5 interactions -> 2 pairs + 1 virtual, absorbed=1).
    fn term_test_interactions() -> Vec<BusInteraction> {
        use crate::lookup::Multiplicity;
        vec![
            BusInteraction::sender(0u64, Multiplicity::Column(4), Packing::Direct.columns(&[0])),
            BusInteraction::receiver(1u64, Multiplicity::One, Packing::Word4L.columns(&[0])),
            BusInteraction::sender(2u64, Multiplicity::Sum(4, 5), Packing::DWordHL.columns(&[0])),
            BusInteraction::receiver(3u64, Multiplicity::Negated(6), Packing::QuadHL.columns(&[0])),
            BusInteraction::sender(
                4u64,
                Multiplicity::Linear(vec![
                    LinearTerm::Column { coefficient: 1, column: 4 },
                    LinearTerm::Column { coefficient: -1, column: 5 },
                ]),
                vec![BusValue::column(1), BusValue::column(2)],
            ),
        ]
    }

    // CPU-only: descriptor term combine (eval_term over host-inverted
    // eval_fingerprint) matches the reference. De-risks the multiplicity
    // descriptor + output grouping without a GPU.
    #[test]
    fn descriptor_term_matches_reference_cpu() {
        let interactions = term_test_interactions();
        let num_cols = 8;
        let num_rows = 32;
        let mut st = 0x9e37_79b9_7f4a_7c15u64;
        let main: Vec<Vec<FieldElement<F>>> = (0..num_cols)
            .map(|_| (0..num_rows).map(|_| FieldElement::<F>::from(lcg(&mut st) % 251)).collect())
            .collect();
        let alpha = FieldElement::<E>::from(lcg(&mut st));
        let z = FieldElement::<E>::from(lcg(&mut st));
        let shifts = PackingShifts::<F>::new();

        let desc = build_fingerprint_descriptor(&interactions);
        let alpha_powers = compute_alpha_powers(&alpha, desc.alpha_powers_len);

        // Reciprocals of every interaction's fingerprint, laid out [k*num_rows+row].
        let mut recips: Vec<FieldElement<E>> = Vec::with_capacity(interactions.len() * num_rows);
        for k in 0..interactions.len() {
            for row in 0..num_rows {
                recips.push(eval_fingerprint::<F, E>(&desc, k, |c| &main[c][row], &alpha_powers, &z));
            }
        }
        FieldElement::inplace_batch_inverse(&mut recips).unwrap();

        let groups: [Vec<&BusInteraction>; 3] = [
            vec![&interactions[0], &interactions[1]],
            vec![&interactions[2], &interactions[3]],
            vec![&interactions[4]],
        ];
        assert_eq!(desc.num_out_cols, 3);
        for (col, g) in groups.iter().enumerate() {
            let want = reference_term_column(g, &main, num_rows, &alpha_powers, &z, &shifts);
            for row in 0..num_rows {
                let got = eval_term::<F, E>(&desc, col, row, num_rows, |c| &main[c][row], &recips);
                assert_eq!(got, want[row], "term mismatch col {col} row {row}");
            }
        }
    }

    // Full GPU term pipeline (fingerprint -> batch invert -> term) vs the CPU
    // reference, byte for byte. Covers committed pairs + the virtual column.
    #[test]
    #[ignore = "requires GPU; run with --ignored"]
    fn gpu_term_columns_match_cpu() {
        // 5 interactions -> 2 committed pairs + 1 virtual (odd, absorbed=1).
        let interactions = term_test_interactions();

        let num_cols = 8;
        let num_rows = 64;
        let mut st = 0x5151_2323_9797_0e0eu64;
        // Small column values so multiplicities like Negated (0/1) stay meaningful.
        let main: Vec<Vec<FieldElement<F>>> = (0..num_cols)
            .map(|_| (0..num_rows).map(|_| FieldElement::<F>::from(lcg(&mut st) % 251)).collect())
            .collect();
        let alpha = mk_ext3(&mut st);
        let z = mk_ext3(&mut st);
        let shifts = PackingShifts::<F>::new();

        let desc = build_fingerprint_descriptor(&interactions);
        let alpha_powers = compute_alpha_powers(&alpha, desc.alpha_powers_len);

        // CPU reference term columns: 2 committed pairs + virtual (last 1).
        let mut cpu = vec![0u64; desc.num_out_cols * num_rows * 3];
        let mut ref_cols: Vec<Vec<FieldElement<E>>> = Vec::new();
        ref_cols.push(reference_term_column(
            &[&interactions[0], &interactions[1]],
            &main,
            num_rows,
            &alpha_powers,
            &z,
            &shifts,
        ));
        ref_cols.push(reference_term_column(
            &[&interactions[2], &interactions[3]],
            &main,
            num_rows,
            &alpha_powers,
            &z,
            &shifts,
        ));
        ref_cols.push(reference_term_column(
            &[&interactions[4]],
            &main,
            num_rows,
            &alpha_powers,
            &z,
            &shifts,
        ));
        for (col, rc) in ref_cols.iter().enumerate() {
            for (row, v) in rc.iter().enumerate() {
                let o = (col * num_rows + row) * 3;
                cpu[o..o + 3].copy_from_slice(&limbs(v));
            }
        }

        // GPU pipeline.
        let mut main_flat = vec![0u64; num_cols * num_rows];
        for c in 0..num_cols {
            for r in 0..num_rows {
                main_flat[c * num_rows + r] = *main[c][r].value();
            }
        }
        let mut alpha_flat = vec![0u64; alpha_powers.len() * 3];
        for (i, p) in alpha_powers.iter().enumerate() {
            alpha_flat[i * 3..i * 3 + 3].copy_from_slice(&limbs(p));
        }
        let md = desc.as_cuda();
        let gpu =
            math_cuda::logup::logup_term_columns(&main_flat, num_rows, &md, &alpha_flat, limbs(&z))
                .unwrap();

        assert_eq!(desc.num_out_cols, 3);
        assert_eq!(gpu, cpu, "GPU term columns mismatch CPU reference");
    }

    // Reference accumulated column, mirroring build_accumulated_column_from_terms.
    fn reference_accumulate(
        cols: &[Vec<FieldElement<E>>],
        num_rows: usize,
    ) -> (Vec<FieldElement<E>>, FieldElement<E>) {
        let mut total = FieldElement::<E>::zero();
        for row in 0..num_rows {
            for c in cols {
                total = &total + &c[row];
            }
        }
        let n = FieldElement::<E>::from(num_rows as u64);
        let offset = &total * n.inv().unwrap();
        let mut acc = FieldElement::<E>::zero();
        let mut out = Vec::with_capacity(num_rows);
        for row in 0..num_rows {
            let mut rs = FieldElement::<E>::zero();
            for c in cols {
                rs = &rs + &c[row];
            }
            acc = &acc + &rs - &offset;
            out.push(acc.clone());
        }
        (out, total)
    }

    // Full resident aux pipeline (fingerprint → invert → term → scan → assemble)
    // vs the CPU reference, byte for byte: the row-major aux buffer (committed +
    // accumulated) and the table_contribution L. Runs on the GPU box.
    #[test]
    #[ignore = "requires GPU; run with --ignored"]
    fn gpu_aux_resident_matches_cpu() {
        let interactions = term_test_interactions(); // 2 committed pairs + 1 virtual
        let num_cols = 8;
        // > BLOCK_SIZE (256) so the grid wide scan recurses across multiple blocks.
        let num_rows = 1024;
        let mut st = 0x243f_6a88_85a3_08d3u64;
        let main: Vec<Vec<FieldElement<F>>> = (0..num_cols)
            .map(|_| (0..num_rows).map(|_| FieldElement::<F>::from(lcg(&mut st) % 251)).collect())
            .collect();
        let alpha = mk_ext3(&mut st);
        let z = mk_ext3(&mut st);
        let shifts = PackingShifts::<F>::new();
        let desc = build_fingerprint_descriptor(&interactions);
        let alpha_powers = compute_alpha_powers(&alpha, desc.alpha_powers_len);

        let committed = vec![
            reference_term_column(&[&interactions[0], &interactions[1]], &main, num_rows, &alpha_powers, &z, &shifts),
            reference_term_column(&[&interactions[2], &interactions[3]], &main, num_rows, &alpha_powers, &z, &shifts),
        ];
        let virtual_col = reference_term_column(&[&interactions[4]], &main, num_rows, &alpha_powers, &z, &shifts);
        let mut all = committed.clone();
        all.push(virtual_col);
        let (acc, total) = reference_accumulate(&all, num_rows);

        let num_aux = committed.len() + 1;
        let mut expected = vec![0u64; num_aux * num_rows * 3];
        for row in 0..num_rows {
            for (col, c) in committed.iter().enumerate() {
                let o = (row * num_aux + col) * 3;
                expected[o..o + 3].copy_from_slice(&limbs(&c[row]));
            }
            let o = (row * num_aux + committed.len()) * 3;
            expected[o..o + 3].copy_from_slice(&limbs(&acc[row]));
        }

        let mut main_flat = vec![0u64; num_cols * num_rows];
        for c in 0..num_cols {
            for r in 0..num_rows {
                main_flat[c * num_rows + r] = *main[c][r].value();
            }
        }
        let mut alpha_flat = vec![0u64; alpha_powers.len() * 3];
        for (i, p) in alpha_powers.iter().enumerate() {
            alpha_flat[i * 3..i * 3 + 3].copy_from_slice(&limbs(p));
        }
        let inv_n = limbs(&FieldElement::<E>::from(num_rows as u64).inv().unwrap());

        let be = math_cuda::device::backend().unwrap();
        let stream = be.next_stream();
        let md = desc.as_cuda();
        let ra = math_cuda::logup::logup_aux_resident(
            &main_flat, num_rows, &md, &alpha_flat, limbs(&z), inv_n, &stream,
        )
        .unwrap();
        assert_eq!(ra.num_aux_cols, num_aux);
        let gpu: Vec<u64> = stream.clone_dtoh(&*ra.buf).unwrap();
        stream.synchronize().unwrap();

        assert_eq!(ra.table_contribution, limbs(&total), "table_contribution L mismatch");
        assert_eq!(gpu, expected, "resident aux buffer mismatch CPU reference");
    }
}
