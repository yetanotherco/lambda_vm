//! Shared test helpers for the stark crate.

use crate::proof::stark::{GkrMultiProof, MultiProof};
use crate::prover::{IsStarkProver, Prover, ProvingError};
use crate::trace::TraceTable;
use crate::traits::AIR;
use crypto::fiat_shamir::is_transcript::IsStarkTranscript;
use math::field::element::FieldElement;
use math::field::traits::{IsFFTField, IsField, IsSubFieldOf};
use math::spill_safe::SpillSafe;
use math::traits::{AsBytes, ByteConversion};

type AirTracePair<'a, Field, FieldExtension, PI> = (
    &'a dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
    &'a mut TraceTable<Field, FieldExtension>,
    &'a PI,
);

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
    Prover::<Field, FieldExtension, PI>::multi_prove(
        air_trace_pairs,
        transcript,
        #[cfg(feature = "disk-spill")]
        crate::storage_mode::StorageMode::Ram,
    )
}

/// [`multi_prove_ram`] for [`crate::lookup::LogUpMode::Gkr`] tables.
pub fn multi_prove_gkr_ram<Field, FieldExtension, PI>(
    air_trace_pairs: Vec<AirTracePair<'_, Field, FieldExtension, PI>>,
    transcript: &mut (impl IsStarkTranscript<FieldExtension, Field> + Clone + Send),
) -> Result<GkrMultiProof<Field, FieldExtension, PI>, ProvingError>
where
    Field: IsSubFieldOf<FieldExtension> + IsFFTField + Send + Sync + Copy + 'static,
    FieldExtension: IsField + Send + Sync + Copy + 'static,
    PI: Send + Sync + Clone,
    FieldElement<Field>: AsBytes + ByteConversion,
    FieldElement<FieldExtension>: AsBytes + ByteConversion,
    <Field as IsField>::BaseType: SpillSafe,
    <FieldExtension as IsField>::BaseType: SpillSafe,
{
    Prover::<Field, FieldExtension, PI>::multi_prove_gkr(
        air_trace_pairs,
        transcript,
        #[cfg(feature = "disk-spill")]
        crate::storage_mode::StorageMode::Ram,
    )
}
