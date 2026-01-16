use math::field::{
    element::FieldElement,
    traits::{IsField, IsSubFieldOf},
};

/// The functionality of a transcript to be used in the STARK Prove and Verify protocols.
pub trait IsTranscript<F: IsField> {
    /// Appends a field element to the transcript.
    fn append_field_element(&mut self, element: &FieldElement<F>);
    /// Appends a bytes to the transcript.
    fn append_bytes(&mut self, new_bytes: &[u8]);
    /// Returns the inner state of the transcript that fully determines its outputs.
    fn state(&self) -> [u8; 32];
    /// Returns a random field element.
    fn sample_field_element(&mut self) -> FieldElement<F>;
    /// Returns a random index between 0 and `upper_bound`.
    fn sample_u64(&mut self, upper_bound: u64) -> u64;
}

pub trait IsStarkTranscript<F: IsField, S: IsField + IsSubFieldOf<F>>: IsTranscript<F> {
    /// Returns a field element not contained in the trace domain or LDE coset.
    /// Uses mathematical membership check: z is in a subgroup of order n iff z^n = 1.
    /// For a coset g*H where H has order n: z is in g*H iff z^n = g^n.
    ///
    /// This is O(log n) instead of O(n) linear search.
    fn sample_z_ood_with_domain_params(
        &mut self,
        trace_length: usize,
        lde_length: usize,
        coset_offset: &FieldElement<S>,
    ) -> FieldElement<F> {
        // Pre-compute coset_offset^lde_length once (for coset membership check)
        let coset_offset_pow_lde: FieldElement<F> =
            coset_offset.clone().to_extension().pow(lde_length);

        loop {
            let z: FieldElement<F> = self.sample_field_element();

            // Check z NOT in trace domain: z^trace_length != 1
            // (trace domain is the group of trace_length-th roots of unity)
            let in_trace_domain = z.pow(trace_length) == FieldElement::one();

            // Check z NOT in LDE coset: z^lde_length != coset_offset^lde_length
            // (LDE coset is coset_offset * <lde_primitive_root>)
            let in_lde_coset = z.pow(lde_length) == coset_offset_pow_lde;

            if !in_trace_domain && !in_lde_coset {
                return z;
            }
        }
    }

    /// Returns a field element not contained in `lde_roots_of_unity_coset` or `trace_roots_of_unity`.
    /// This is a convenience method that extracts parameters and calls `sample_z_ood_with_domain_params`.
    fn sample_z_ood(
        &mut self,
        lde_roots_of_unity_coset: &[FieldElement<S>],
        trace_roots_of_unity: &[FieldElement<S>],
    ) -> FieldElement<F> {
        let trace_length = trace_roots_of_unity.len();
        let lde_length = lde_roots_of_unity_coset.len();
        // First element of coset is: coset_offset * primitive_root^0 = coset_offset
        let coset_offset = &lde_roots_of_unity_coset[0];
        self.sample_z_ood_with_domain_params(trace_length, lde_length, coset_offset)
    }
}
