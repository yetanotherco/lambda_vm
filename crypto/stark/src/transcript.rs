use crypto::fiat_shamir::is_transcript::IsTranscript;
use math::{
    field::{element::FieldElement, traits::IsFFTField},
    traits::AsBytes,
};

/// Returns a batch of size `size` of field elements sampled from the transcript `transcript`.
pub fn batch_sample_challenges<F: IsFFTField>(
    size: usize,
    transcript: &mut impl IsTranscript<F>,
) -> Vec<FieldElement<F>>
where
    FieldElement<F>: AsBytes,
{
    (0..size)
        .map(|_| transcript.sample_field_element())
        .collect()
}
