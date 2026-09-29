//! Proving and verifying a Field VM execution: `FIELD_VM`, `FIELD_VM_DECODE`
//! and `FIELD_VM_MEM` in one `multi_prove`, with a closed bus system.

use crypto::fiat_shamir::is_transcript::IsTranscript;
use stark::config::Commitment;
use stark::constraints::builder::EmptyConstraints;
use stark::lookup::{AirWithBuses, AuxiliaryTraceBuildData, NullBoundaryConstraintBuilder};
use stark::proof::options::{GoldilocksCubicProofOptions, ProofOptions};
use stark::proof::stark::MultiProof;
use stark::prover::{IsStarkProver, ProvingError};
#[cfg(feature = "disk-spill")]
use stark::storage_mode::StorageMode;
use stark::trace::TraceTable;
use stark::traits::AIR;
use stark::verifier::IsStarkVerifier;

use super::air::{self, FieldVmBoundary, FieldVmConstraints, FvmPublicInputs};
use super::decode;
use super::executor::Execution;
use super::isa::Program;
use super::mem::{self, MemBoundary, MemConstraints};
use crate::tables::types::{FEE, GoldilocksExtension, GoldilocksField};

type F = GoldilocksField;
type E = GoldilocksExtension;

pub type FvmProof = MultiProof<F, E, FvmPublicInputs>;

/// Minimum height of every table.
pub const MIN_ROWS: usize = 8;

const DOMAIN_TAG: &[u8] = b"lambda-vm/field-vm/v0";

/// Blowup 4, as the spec's degree-5 constraints require.
pub fn default_options() -> ProofOptions {
    GoldilocksCubicProofOptions::with_blowup(4).expect("blowup 4 is valid")
}

struct Airs {
    fvm: AirWithBuses<F, E, FieldVmBoundary, FvmPublicInputs, FieldVmConstraints>,
    decode: AirWithBuses<F, E, NullBoundaryConstraintBuilder, FvmPublicInputs, EmptyConstraints>,
    mem: AirWithBuses<F, E, MemBoundary, FvmPublicInputs, MemConstraints>,
}

impl Airs {
    fn new(program_id: Commitment, options: &ProofOptions) -> Self {
        let aux = |interactions| AuxiliaryTraceBuildData { interactions };
        Self {
            fvm: AirWithBuses::new(
                air::cols::NUM_COLUMNS,
                aux(air::bus_interactions()),
                options,
                1,
                FieldVmConstraints::new(),
            )
            .with_name("FIELD_VM"),
            decode: AirWithBuses::new(
                decode::NUM_COLUMNS,
                aux(decode::bus_interactions()),
                options,
                1,
                EmptyConstraints,
            )
            .with_name("FIELD_VM_DECODE")
            .with_preprocessed(program_id, decode::NUM_PRECOMPUTED_COLS),
            mem: AirWithBuses::new(
                mem::cols::NUM_COLUMNS,
                aux(mem::bus_interactions()),
                options,
                1,
                MemConstraints,
            )
            .with_name("FIELD_VM_MEM"),
        }
    }

    fn refs(&self) -> [&dyn AIR<Field = F, FieldExtension = E, PublicInputs = FvmPublicInputs>; 3] {
        [&self.fvm, &self.decode, &self.mem]
    }
}

fn transcript(program_id: &Commitment) -> crate::hash_pin::BlockTranscript {
    let mut t = crate::hash_pin::block_transcript(DOMAIN_TAG);
    t.append_bytes(program_id);
    t
}

pub fn program_id(program: &Program, options: &ProofOptions) -> Commitment {
    decode::program_commitment(program, options, MIN_ROWS)
}

/// The three traces of one execution: `FIELD_VM`, `FIELD_VM_DECODE`, `FIELD_VM_MEM`.
pub struct FvmTraces {
    pub fvm: TraceTable<F, E>,
    pub decode: TraceTable<F, E>,
    pub mem: TraceTable<F, E>,
}

pub fn generate_traces(program: &Program, exec: &Execution) -> FvmTraces {
    let (fvm, decode_mult) = air::generate_trace(program, exec, MIN_ROWS);
    FvmTraces {
        fvm,
        decode: decode::generate_trace(program, &decode_mult, MIN_ROWS),
        mem: mem::generate_trace(&exec.mem, &exec.mem_mult, MIN_ROWS),
    }
}

pub fn prove_traces(
    program_id: &Commitment,
    traces: &mut FvmTraces,
    options: &ProofOptions,
) -> Result<FvmProof, ProvingError> {
    let pi = FvmPublicInputs {
        last_row: traces.fvm.num_rows() - 1,
    };
    let none = FvmPublicInputs::default();
    let airs = Airs::new(*program_id, options);
    let [fvm, dec, mem] = airs.refs();
    crate::hash_pin::BlockProver::<F, E, FvmPublicInputs>::multi_prove(
        vec![
            (fvm, &mut traces.fvm, &pi),
            (dec, &mut traces.decode, &none),
            (mem, &mut traces.mem, &none),
        ],
        &mut transcript(program_id),
        #[cfg(feature = "disk-spill")]
        StorageMode::Ram,
        stark::residency_mode::ResidencyMode::default(),
    )
}

pub fn prove(
    program: &Program,
    exec: &Execution,
    options: &ProofOptions,
) -> Result<FvmProof, ProvingError> {
    prove_traces(
        &program_id(program, options),
        &mut generate_traces(program, exec),
        options,
    )
}

pub fn verify(program_id: &Commitment, proof: &FvmProof, options: &ProofOptions) -> bool {
    let [fvm, dec, mem] = match proof.proofs.as_slice() {
        [a, b, c] => [a, b, c],
        _ => return false,
    };
    if fvm.public_inputs.last_row + 1 != fvm.trace_length
        || dec.public_inputs != FvmPublicInputs::default()
        || mem.public_inputs != FvmPublicInputs::default()
    {
        return false;
    }
    let airs = Airs::new(*program_id, options);
    crate::hash_pin::BlockVerifier::<F, E, FvmPublicInputs>::multi_verify(
        &airs.refs(),
        proof,
        &mut transcript(program_id),
        &FEE::zero(),
    )
}
