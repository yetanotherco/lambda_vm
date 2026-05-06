//! Shared test helpers for the stark crate.

use crate::proof::stark::MultiProof;
use crate::prover::{IsStarkProver, Prover, ProvingError};
use crate::trace::TraceTable;
use crate::traits::AIR;
use crypto::fiat_shamir::is_transcript::IsStarkTranscript;
use math::field::element::FieldElement;
use math::field::traits::{IsFFTField, IsField, IsSubFieldOf};
use math::spill_safe::SpillSafe;
use math::traits::{AsBytes, ByteConversion};

/// Multi-AIR prove with `StorageMode::Ram`. Test convenience.
pub fn multi_prove_ram<Field, FieldExtension, PI>(
    air_trace_pairs: Vec<(
        &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        &mut TraceTable<Field, FieldExtension>,
        &PI,
    )>,
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
