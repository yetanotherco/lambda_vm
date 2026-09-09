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
    /// Returns a digest of everything absorbed so far (the sponge state).
    ///
    /// This binds the absorbed input stream, but it does NOT capture any
    /// buffered squeeze output an implementation may hold (see
    /// `DefaultTranscript`'s duplex output buffer): two transcripts with equal
    /// `state()` produce identical future samples only if they also share the
    /// same absorb/sample history. Prover and verifier stay synchronized
    /// because they perform the same sequence of calls, not because `state()`
    /// alone determines outputs.
    fn state(&self) -> [u8; 32];
    /// Returns a random field element.
    fn sample_field_element(&mut self) -> FieldElement<F>;
    /// Returns a random index in [0, `upper_bound`).
    fn sample_u64(&mut self, upper_bound: u64) -> u64;
}

pub trait IsStarkTranscript<F: IsField, S: IsField + IsSubFieldOf<F>>: IsTranscript<F> {
    /// Returns a field element not contained in the trace domain or LDE coset.
    /// Uses mathematical membership check: z is in a subgroup of order n iff z^n = 1.
    /// For a coset g*H where H has order n: z is in g*H iff z^n = g^n.
    ///
    /// This is O(log n) instead of O(n) linear search.
    ///
    /// # Precondition
    ///
    /// `trace_length > 0` and `lde_length` is an exact multiple of
    /// `trace_length`. The LDE-coset check is computed as
    /// `(z^trace_length)^(lde_length / trace_length)`, which equals
    /// `z^lde_length` only under this invariant. A violation silently
    /// evaluates the wrong power and can fail to reject points inside the
    /// LDE coset (soundness regression). Debug builds assert this; release
    /// builds trust the caller.
    fn sample_z_ood_with_domain_params(
        &mut self,
        trace_length: usize,
        lde_length: usize,
        coset_offset: &FieldElement<S>,
    ) -> FieldElement<F> {
        debug_assert!(
            trace_length > 0 && lde_length.is_multiple_of(trace_length),
            "sample_z_ood_with_domain_params: lde_length ({lde_length}) must be a positive multiple of trace_length ({trace_length})",
        );
        // Coset membership reference value, precomputed once. The power map runs
        // in the base field S (cheap) and the scalar result is lifted to F — we
        // never lift `coset_offset` into F before exponentiating.
        let coset_offset_pow_lde: FieldElement<F> = coset_offset.pow(lde_length).to_extension();
        // lde_length = trace_length * blowup_factor, so z^lde = (z^trace)^blowup.
        let blowup_factor = lde_length / trace_length;

        loop {
            let z: FieldElement<F> = self.sample_field_element();

            // z is in the trace domain (trace_length-th roots of unity) iff z^trace_length == 1.
            let z_pow_trace = z.pow(trace_length);
            let in_trace_domain = z_pow_trace == FieldElement::one();

            // z is in the LDE coset (coset_offset * <lde root>) iff z^lde == coset_offset^lde.
            let in_lde_coset = z_pow_trace.pow(blowup_factor) == coset_offset_pow_lde;

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
