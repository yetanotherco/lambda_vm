//! Shared test helpers for the stark crate.

use crate::proof::stark::MultiProof;
use crate::prover::{IsStarkProver, Prover, ProvingError};
use crate::trace::TraceTable;
use crate::traits::AIR;
use crypto::fiat_shamir::is_transcript::IsStarkTranscript;
// MatrixTag is re-exported via `synth_main_tags`; no direct use here.
use math::field::element::FieldElement;
use math::field::traits::{IsFFTField, IsField, IsSubFieldOf};
use math::spill_safe::SpillSafe;
use math::traits::{AsBytes, ByteConversion};

type AirTracePair<'a, Field, FieldExtension, PI> = (
    &'a dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
    &'a mut TraceTable<Field, FieldExtension>,
    &'a PI,
);

pub use crate::mmcs_leaf::synth_main_tags;

pub fn multi_verify_ram<Field, FieldExtension, PI>(
    airs: &[&dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>],
    multi_proof: &MultiProof<Field, FieldExtension, PI>,
    transcript: &mut (impl IsStarkTranscript<FieldExtension, Field> + Clone),
    expected_bus_balance: &FieldElement<FieldExtension>,
) -> bool
where
    Field: IsSubFieldOf<FieldExtension> + IsFFTField + Send + Sync + Copy + 'static,
    FieldExtension: IsField + Send + Sync + Copy + 'static,
    FieldElement<Field>: AsBytes + ByteConversion + Sync + Send,
    FieldElement<FieldExtension>: AsBytes + ByteConversion + Sync + Send,
    <Field as IsField>::BaseType: SpillSafe,
    <FieldExtension as IsField>::BaseType: SpillSafe,
{
    use crate::verifier::{IsStarkVerifier, Verifier};
    let main_tags = synth_main_tags(airs.len());
    Verifier::<Field, FieldExtension, PI>::multi_verify(
        airs,
        multi_proof,
        &main_tags,
        transcript,
        expected_bus_balance,
    )
}

pub fn multi_prove_ram<Field, FieldExtension, PI>(
    air_trace_pairs: Vec<AirTracePair<'_, Field, FieldExtension, PI>>,
    transcript: &mut (impl IsStarkTranscript<FieldExtension, Field> + Clone + Send),
) -> Result<MultiProof<Field, FieldExtension, PI>, ProvingError>
where
    Field: IsSubFieldOf<FieldExtension> + IsFFTField + Send + Sync + Copy + 'static,
    FieldExtension: IsField + Send + Sync + Copy + 'static,
    PI: Send + Sync + Clone,
    FieldElement<Field>: AsBytes + ByteConversion,
    FieldElement<FieldExtension>: AsBytes + ByteConversion,
    <Field as IsField>::BaseType: SpillSafe,
    <FieldExtension as IsField>::BaseType: SpillSafe,
{
    let main_tags = synth_main_tags(air_trace_pairs.len());
    Prover::<Field, FieldExtension, PI>::multi_prove(
        air_trace_pairs,
        &main_tags,
        transcript,
        #[cfg(feature = "disk-spill")]
        crate::storage_mode::StorageMode::Ram,
    )
}
