use std::sync::Arc;

use math::{
    fft::roots_of_unity::get_powers_of_primitive_root_coset,
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
    pub(crate) lde_roots_of_unity_coset: Vec<FieldElement<F>>,
    pub(crate) trace_primitive_root: FieldElement<F>,
    pub(crate) trace_roots_of_unity: Vec<FieldElement<F>>,
    pub(crate) coset_offset: FieldElement<F>,
    pub(crate) blowup_factor: usize,
    pub(crate) interpolation_domain_size: usize,
    /// Domain-derived values that rounds 2-4 otherwise rebuild per table per
    /// epoch (each involves an LDE-size-order batch inversion or clone).
    ood_constants: std::sync::OnceLock<DomainConstants<F>>,
    fri_inv_twiddles: std::sync::OnceLock<Vec<FieldElement<F>>>,
    boundary_z_inv: std::sync::Mutex<std::collections::HashMap<usize, Arc<Vec<FieldElement<F>>>>>,
}

impl<F: IsFFTField> Domain<F> {
    /// Builds the interpolation and LDE domains used by the prover.
    ///
    /// - Interpolation domain: the `trace_length` roots of unity (must be a power of 2).
    /// - LDE domain: a coset of size `trace_length * blowup_factor`, shifted by
    ///   `air.options().coset_offset`.
    pub fn new<A>(air: &A, trace_length: usize) -> Self
    where
        A: AIR<Field = F> + ?Sized,
    {
        let blowup_factor = air.options().blowup_factor as usize;
        let coset_offset = FieldElement::from(air.options().coset_offset);
        let root_order = trace_length.trailing_zeros();
        let trace_primitive_root = F::get_primitive_root_of_unity(root_order as u64).unwrap();
        let trace_roots_of_unity = get_powers_of_primitive_root_coset(
            root_order as u64,
            trace_length,
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
            lde_roots_of_unity_coset,
            trace_primitive_root,
            trace_roots_of_unity,
            blowup_factor,
            coset_offset,
            interpolation_domain_size: trace_length,
            ood_constants: std::sync::OnceLock::new(),
            fri_inv_twiddles: std::sync::OnceLock::new(),
            boundary_z_inv: std::sync::Mutex::new(std::collections::HashMap::new()),
        }
    }

    /// Boundary-zerofier inverse evaluations `1/(x − g^step)` over the LDE
    /// coset, cached per step: boundary constraints, tables, and epochs that
    /// share this domain otherwise each pay an LDE-size batch inversion.
    pub(crate) fn boundary_zerofier_inv(&self, step: usize) -> Arc<Vec<FieldElement<F>>> {
        if let Some(v) = self.boundary_z_inv.lock().unwrap().get(&step) {
            return v.clone();
        }
        let point = self.trace_primitive_root.pow(step as u64);
        let mut evals: Vec<FieldElement<F>> = self
            .lde_roots_of_unity_coset
            .iter()
            .map(|v| v - &point)
            .collect();
        FieldElement::inplace_batch_inverse(&mut evals)
            .expect("LDE coset points never coincide with a trace root");
        let v = Arc::new(evals);
        self.boundary_z_inv
            .lock()
            .unwrap()
            .insert(step, v.clone());
        v
    }

    /// Barycentric OOD constants (round 3), computed once per domain.
    pub fn ood_constants(&self) -> &DomainConstants<F> {
        self.ood_constants
            .get_or_init(|| DomainConstants::from_domain(self))
    }

    /// FRI folding inverse twiddles for the LDE coset (round 4), computed once
    /// per domain. Callers copy them into their per-layer working buffer.
    pub(crate) fn fri_inv_twiddles(&self) -> &[FieldElement<F>] {
        self.fri_inv_twiddles.get_or_init(|| {
            crate::fri::fri_functions::compute_coset_twiddles_inv(
                &self.coset_offset,
                self.interpolation_domain_size * self.blowup_factor,
            )
        })
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
