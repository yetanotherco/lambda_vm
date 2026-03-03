use math::{
    fft::cpu::roots_of_unity::get_powers_of_primitive_root_coset,
    field::{
        element::FieldElement,
        traits::{IsFFTField, IsField, IsSubFieldOf},
    },
};

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
