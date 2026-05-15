use math::{
    fft::cpu::roots_of_unity::get_powers_of_primitive_root_coset,
    field::{
        element::FieldElement,
        traits::{IsFFTField, IsField, IsSubFieldOf},
    },
};

/// Precomputed constants for barycentric interpolation on the trace-size coset.
///
/// Derived from a [`Domain`]: the N evaluation points at stride `blowup_factor`
/// within the LDE coset, plus the field scalars that appear in every barycentric
/// evaluation. Computed once in round 3 and shared across composition-poly and
/// trace OOD evaluations.
pub struct DomainConstants<F: IsField> {
    /// The N trace-size coset points: `lde_coset[i * blowup_factor]` for `i in 0..N`.
    pub points: Vec<FieldElement<F>>,
    /// `coset_offset ^ N`.
    pub offset_pow_n: FieldElement<F>,
    /// `1 / N` in the base field.
    pub size_inv: FieldElement<F>,
    /// `(coset_offset ^ N) ^ -1`.
    pub offset_pow_n_inv: FieldElement<F>,
}

impl<F: IsFFTField> DomainConstants<F> {
    pub fn from_domain(domain: &Domain<F>) -> Self {
        let n = domain.interpolation_domain_size;
        let bf = domain.blowup_factor;
        let points = (0..n)
            .map(|i| domain.lde_roots_of_unity_coset[i * bf].clone())
            .collect();
        let offset_pow_n = domain.coset_offset.pow(n);
        let size_inv = FieldElement::<F>::from(n as u64)
            .inv()
            .expect("domain size is non-zero; field characteristic must not divide n");
        let offset_pow_n_inv = offset_pow_n.inv().expect("coset_offset_pow_n is non-zero");
        Self {
            points,
            offset_pow_n,
            size_inv,
            offset_pow_n_inv,
        }
    }
}

use super::traits::AIR;

/// Full domain with pre-computed roots of unity. Used by the prover which needs
/// all elements for FFT operations.
pub struct Domain<F: IsFFTField> {
    pub(crate) root_order: u32,
    pub(crate) lde_roots_of_unity_coset: Vec<FieldElement<F>>,
    pub(crate) trace_primitive_root: FieldElement<F>,
    pub(crate) trace_roots_of_unity: Vec<FieldElement<F>>,
    pub(crate) coset_offset: FieldElement<F>,
    pub(crate) blowup_factor: usize,
    pub(crate) interpolation_domain_size: usize,
}

impl<F: IsFFTField> Domain<F> {
    pub fn new<A>(air: &A, trace_length: usize) -> Self
    where
        A: AIR<Field = F>,
    {
        // Initial definitions
        let blowup_factor = air.options().blowup_factor as usize;
        let coset_offset = FieldElement::from(air.options().coset_offset);
        let interpolation_domain_size = trace_length;
        let root_order = trace_length.trailing_zeros();
        // * Generate Coset
        let trace_primitive_root = F::get_primitive_root_of_unity(root_order as u64).unwrap();
        let trace_roots_of_unity = get_powers_of_primitive_root_coset(
            root_order as u64,
            interpolation_domain_size,
            &FieldElement::one(),
        )
        .unwrap();

        let lde_root_order = (trace_length * blowup_factor).trailing_zeros();
        let lde_roots_of_unity_coset = get_powers_of_primitive_root_coset(
            lde_root_order as u64,
            trace_length * blowup_factor,
            &coset_offset,
        )
        .unwrap();

        Self {
            root_order,
            lde_roots_of_unity_coset,
            trace_primitive_root,
            trace_roots_of_unity,
            blowup_factor,
            coset_offset,
            interpolation_domain_size,
        }
    }
}

/// Quotient evaluation domain for the chunks-based prover (Phase 1 of the
/// migration to a Plonky3-style commitment).
///
/// Sized as `num_chunks * trace_length` where `num_chunks = next_pow2(d_max)`.
/// For `d_max=1` AIRs (like fib_pair) the quotient domain coincides with the
/// trace coset (size N). For `d_max=3` (e.g., Keccak) it has size 4N.
///
/// Each evaluation point at index `i` is `coset_offset * omega^i`, where
/// `omega` is the primitive root of unity of order `num_chunks * trace_length`.
///
/// This struct is added in parallel to `Domain` while the chunks code path
/// matures. The existing single-H prover does NOT use this struct.
pub struct QuotientDomain<F: IsFFTField> {
    /// Number of chunks the quotient is split into = `next_pow2(d_max)`.
    pub num_chunks: usize,
    /// Trace domain size N. Each chunk will be size N.
    pub trace_length: usize,
    /// Total quotient evaluation domain size = `num_chunks * trace_length`.
    pub size: usize,
    /// Coset offset (matches `Domain.coset_offset`, typically the field generator).
    pub coset_offset: FieldElement<F>,
    /// Pre-computed coset points: `coset_offset * omega^i` for `i in 0..size`,
    /// where `omega` has order `size`.
    pub roots_of_unity_coset: Vec<FieldElement<F>>,
    /// `log2(size)`, useful for FFT routines.
    pub log_size: u32,
}

impl<F: IsFFTField> QuotientDomain<F> {
    /// Construct a quotient domain for an AIR with the given `d_max` (maximum
    /// constraint degree). The size is `next_pow2(d_max) * trace_length`.
    ///
    /// Assumes `domain.interpolation_domain_size` is a power of two (Lambda's
    /// existing invariant).
    pub fn new(domain: &Domain<F>, d_max: usize) -> Self {
        Self::from_parts(
            domain.interpolation_domain_size,
            domain.coset_offset.clone(),
            d_max,
        )
    }

    /// Verifier-friendly constructor that only needs `trace_length` and
    /// `coset_offset` (which the verifier has via `VerifierDomain` or the AIR
    /// options) rather than a full prover-side [`Domain`].
    pub fn from_parts(trace_length: usize, coset_offset: FieldElement<F>, d_max: usize) -> Self {
        let num_chunks = d_max.next_power_of_two().max(1);
        let size = num_chunks * trace_length;
        let log_size = size.trailing_zeros();
        let roots_of_unity_coset =
            get_powers_of_primitive_root_coset(log_size as u64, size, &coset_offset).unwrap();
        Self {
            num_chunks,
            trace_length,
            size,
            coset_offset,
            roots_of_unity_coset,
            log_size,
        }
    }

    /// Get the i-th point of the quotient domain (`coset_offset * omega^i`).
    #[inline]
    pub fn point_at(&self, index: usize) -> &FieldElement<F> {
        &self.roots_of_unity_coset[index]
    }

    /// Split an eval vector of size `self.size` into `self.num_chunks` chunks
    /// of size `self.trace_length` each, using P3-style **interleaved** split:
    ///
    /// - chunk_0 gets indices `0, num_chunks, 2*num_chunks, ...`
    /// - chunk_1 gets indices `1, num_chunks+1, 2*num_chunks+1, ...`
    /// - chunk_i gets indices `i, i+num_chunks, i+2*num_chunks, ...`
    ///
    /// This matches `commit_quotient` in
    /// `plonky3/commit/src/domain.rs::split_evals` (interleaved decomposition
    /// of a coset `gH` into disjoint sub-cosets `g·h^i·K`).
    ///
    /// Each resulting chunk represents the evaluations of an implicit
    /// polynomial `P_i(x)` of degree `< trace_length` on the sub-coset
    /// `{coset_offset · omega^i · (omega^num_chunks)^j : j=0..trace_length-1}`.
    pub fn split_evals_interleaved<E: IsField>(
        &self,
        evals: &[FieldElement<E>],
    ) -> Vec<Vec<FieldElement<E>>> {
        assert_eq!(
            evals.len(),
            self.size,
            "split_evals_interleaved: evals.len() = {} but quotient_domain.size = {}",
            evals.len(),
            self.size,
        );
        let num_chunks = self.num_chunks;
        let chunk_size = self.trace_length;
        let mut chunks: Vec<Vec<FieldElement<E>>> = (0..num_chunks)
            .map(|_| Vec::with_capacity(chunk_size))
            .collect();
        for (idx, val) in evals.iter().enumerate() {
            chunks[idx % num_chunks].push(val.clone());
        }
        chunks
    }

    /// Return the sub-coset offset and generator for chunk `i`.
    /// - Sub-coset offset: `coset_offset · omega^i`
    /// - Sub-coset generator: `omega^num_chunks` (which has order `trace_length`)
    pub fn chunk_subdomain(&self, chunk_idx: usize) -> (FieldElement<F>, FieldElement<F>) {
        assert!(chunk_idx < self.num_chunks);
        let sub_offset = self.roots_of_unity_coset[chunk_idx].clone();
        let sub_generator = if self.size == self.num_chunks {
            // chunk_size = 1 — degenerate, sub_generator is irrelevant
            FieldElement::one()
        } else {
            // omega^num_chunks
            self.roots_of_unity_coset[self.num_chunks].clone() * &self.coset_offset.inv().unwrap()
        };
        (sub_offset, sub_generator)
    }

    /// Reconstruct `H(z)` from chunk openings `Q_i(z)` using the P3-style
    /// Lagrange identity over the disjoint sub-cosets:
    ///
    /// ```text
    /// H(z) = sum_{i=0..K-1} zps[i] * Q_i(z)
    /// zps[i] = product_{j != i} V_j(z) / V_j(first_point_i)
    /// ```
    ///
    /// where `V_j(x) = x^N - first_point_j^N` is the vanishing polynomial of the
    /// j-th sub-coset, `first_point_j = coset_offset · omega^j = point_at(j)`,
    /// `K = num_chunks` and `N = trace_length`.
    ///
    /// The identity holds because (a) each `zps[i]` is a polynomial of degree
    /// `<= (K-1)·N` that evaluates to `1` on the i-th sub-coset and to `0` on
    /// every other sub-coset, and (b) `Q_i` agrees with `H` on the i-th
    /// sub-coset by construction of the interleaved split. The sum is therefore
    /// a degree-`< K·N` polynomial that agrees with `H` on `K·N` distinct
    /// points, i.e., the same polynomial.
    ///
    /// Matches `recompose_quotient_from_chunks` in p3-uni-stark 0.5.2
    /// (`src/verifier.rs`, lines 25–69) — the entry point the verifier uses to
    /// fold chunk openings back into `H(z)` for the constraint check at `z`.
    pub fn recompose_at<E>(
        &self,
        chunk_evals_at_z: &[FieldElement<E>],
        z: &FieldElement<E>,
    ) -> FieldElement<E>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        assert_eq!(
            chunk_evals_at_z.len(),
            self.num_chunks,
            "recompose_at: chunk_evals_at_z.len() = {} but num_chunks = {}",
            chunk_evals_at_z.len(),
            self.num_chunks,
        );
        let n = self.trace_length;

        // first_pow_n[j] = (coset_offset · omega^j)^N, in the base field F.
        let first_pow_n: Vec<FieldElement<F>> = (0..self.num_chunks)
            .map(|j| self.point_at(j).pow(n))
            .collect();
        let z_pow_n = z.pow(n);

        // Both V_j(z) and V_j(first_point_i) are written with the F-term on the
        // left of the subtraction so the mixed F→E coercion goes through
        // lambdaworks' `IsSubFieldOf::sub` (which only supports F − E). The
        // overall sign of the ratio is unchanged because both numerator and
        // denominator are negated.
        let mut result = FieldElement::<E>::zero();
        for i in 0..self.num_chunks {
            let mut zp = FieldElement::<E>::one();
            for j in 0..self.num_chunks {
                if j == i {
                    continue;
                }
                let num_e = &first_pow_n[j] - &z_pow_n;
                let denom_inv = (&first_pow_n[j] - &first_pow_n[i])
                    .inv()
                    .expect("disjoint sub-cosets ⇒ shift_i^N != shift_j^N");
                zp *= &num_e;
                zp *= &denom_inv;
            }
            result = result + &zp * &chunk_evals_at_z[i];
        }
        result
    }
}

/// Lightweight domain without pre-computed roots of unity. Used by the verifier
/// which only needs to compute specific elements on-demand.
/// This avoids allocating O(n) memory for domains that would be O(millions) of elements.
pub struct VerifierDomain<F: IsFFTField> {
    pub(crate) root_order: u32,
    pub(crate) trace_length: usize,
    pub(crate) lde_length: usize,
    pub(crate) trace_primitive_root: FieldElement<F>,
    pub(crate) lde_primitive_root: FieldElement<F>,
    pub(crate) coset_offset: FieldElement<F>,
}

impl<F: IsFFTField> VerifierDomain<F> {
    /// Compute an LDE coset element at a specific index on-demand.
    /// Element at index i is: coset_offset * lde_primitive_root^i
    #[inline]
    pub fn lde_coset_element(&self, index: usize) -> FieldElement<F> {
        &self.coset_offset * self.lde_primitive_root.pow(index)
    }
}

pub fn new_domain<Field, FieldExtension, PI>(
    air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
    trace_length: usize,
) -> Domain<Field>
where
    Field: IsSubFieldOf<FieldExtension> + IsFFTField + Send + Sync,
    FieldExtension: Send + Sync + IsField,
{
    // Initial definitions
    let blowup_factor = air.options().blowup_factor as usize;
    let coset_offset = FieldElement::from(air.options().coset_offset);
    let interpolation_domain_size = trace_length;
    let root_order = trace_length.trailing_zeros();
    // * Generate Coset
    let trace_primitive_root = Field::get_primitive_root_of_unity(root_order as u64).unwrap();
    let trace_roots_of_unity = get_powers_of_primitive_root_coset(
        root_order as u64,
        interpolation_domain_size,
        &FieldElement::one(),
    )
    .unwrap();

    let lde_root_order = (trace_length * blowup_factor).trailing_zeros();
    let lde_roots_of_unity_coset = get_powers_of_primitive_root_coset(
        lde_root_order as u64,
        trace_length * blowup_factor,
        &coset_offset,
    )
    .unwrap();

    Domain {
        root_order,
        lde_roots_of_unity_coset,
        trace_primitive_root,
        trace_roots_of_unity,
        blowup_factor,
        coset_offset,
        interpolation_domain_size,
    }
}

/// Creates a lightweight verifier domain without pre-computing roots of unity.
/// This is O(1) instead of O(trace_length * blowup_factor) for domain creation.
pub fn new_verifier_domain<Field, FieldExtension, PI>(
    air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
    trace_length: usize,
) -> VerifierDomain<Field>
where
    Field: IsSubFieldOf<FieldExtension> + IsFFTField + Send + Sync,
    FieldExtension: Send + Sync + IsField,
{
    let blowup_factor = air.options().blowup_factor as usize;
    let coset_offset = FieldElement::from(air.options().coset_offset);
    let lde_length = trace_length * blowup_factor;
    let root_order = trace_length.trailing_zeros();

    let trace_primitive_root = Field::get_primitive_root_of_unity(root_order as u64).unwrap();

    let lde_root_order = lde_length.trailing_zeros();
    let lde_primitive_root = Field::get_primitive_root_of_unity(lde_root_order as u64).unwrap();

    VerifierDomain {
        root_order,
        trace_length,
        lde_length,
        trace_primitive_root,
        lde_primitive_root,
        coset_offset,
    }
}
