//! Proving and verifying a Field VM execution: `FIELD_VM`, `FIELD_VM_DECODE`
//! and `FIELD_VM_MEM` in one `multi_prove`.
//!
//! The statement is the program id and the public memory cells. Every bus
//! closes internally except `FIELD_VM_PUBLIC`, whose balance the verifier
//! recomputes from the claimed cells.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::fiat_shamir::is_transcript::IsTranscript;
use stark::config::Commitment;
use stark::constraints::builder::EmptyConstraints;
use stark::lookup::{AirWithBuses, AuxiliaryTraceBuildData, NullBoundaryConstraintBuilder};
use stark::proof::options::{GoldilocksCubicProofOptions, ProofOptions};
use stark::proof::stark::MultiProof;
use stark::prover::{IsStarkProver, Prover, ProvingError};
#[cfg(feature = "disk-spill")]
use stark::storage_mode::StorageMode;
use stark::trace::TraceTable;
use stark::traits::AIR;
use stark::verifier::{IsStarkVerifier, Verifier};

use math::field::element::FieldElement;

use super::air::{self, FieldVmBoundary, FieldVmConstraints, FvmPublicInputs};
use super::decode;
use super::executor::Execution;
use super::isa::Program;
use super::mem::{self, MemBoundary, MemConstraints};
use crate::tables::types::{BusId, FEE, GoldilocksExtension, GoldilocksField};

type F = GoldilocksField;
type E = GoldilocksExtension;

pub type FvmProof = MultiProof<F, E, FvmPublicInputs>;

/// Minimum height of every table.
pub const MIN_ROWS: usize = 8;

const DOMAIN_TAG: &[u8] = b"lambda-vm/field-vm/v1";

/// A public memory cell: an input the program reads or an output it leaves.
pub type PublicCell = (u64, FEE);

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

fn transcript(program_id: &Commitment, public: &[PublicCell]) -> DefaultTranscript<E> {
    let mut t = DefaultTranscript::<E>::new(DOMAIN_TAG);
    t.append_bytes(program_id);
    t.append_bytes(&(public.len() as u64).to_le_bytes());
    for (addr, value) in public {
        t.append_bytes(&addr.to_le_bytes());
        for c in value.value() {
            t.append_bytes(&c.canonical().to_le_bytes());
        }
    }
    t
}

/// The cells at `addrs` after `exec`, for [`generate_traces`] and [`verify`].
pub fn public_cells(exec: &Execution, addrs: &[u64]) -> Vec<PublicCell> {
    addrs.iter().map(|&a| (a, exec.mem[a as usize])).collect()
}

/// `Σ 1/(z − (FIELD_VM_PUBLIC + addr·α + Σ_k v_k·α^{2+k}))`, the fingerprint of
/// the MEM table's public sender.
fn expected_public_balance(public: &[PublicCell], z: &FEE, alpha: &FEE) -> Option<FEE> {
    let bus = FEE::from(BusId::FieldVmPublic as u64);
    let powers: Vec<FEE> = (1..=4).map(|k| alpha.pow(k as u64)).collect();
    let mut fingerprints: Vec<FEE> = public
        .iter()
        .map(|(addr, value)| {
            let mut acc = bus + FEE::from(*addr) * powers[0];
            for (k, c) in value.value().iter().enumerate() {
                acc += c.to_extension::<E>() * powers[1 + k];
            }
            *z - acc
        })
        .collect();
    FieldElement::inplace_batch_inverse(&mut fingerprints).ok()?;
    Some(fingerprints.iter().fold(FEE::zero(), |acc, t| acc + t))
}

/// Replays the prover's Phase A (main commitments) to recover the LogUp
/// challenges `(z, α)`.
fn logup_challenges(
    airs: &Airs,
    proof: &FvmProof,
    transcript: &mut DefaultTranscript<E>,
) -> (FEE, FEE) {
    for (air, p) in airs.refs().iter().zip(&proof.proofs) {
        if air.is_preprocessed() {
            transcript.append_bytes(&air.precomputed_commitment());
        }
        transcript.append_bytes(&p.lde_trace_main_merkle_root);
    }
    (
        transcript.sample_field_element(),
        transcript.sample_field_element(),
    )
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

/// `public` are the addresses of the public cells.
pub fn generate_traces(program: &Program, exec: &Execution, public: &[u64]) -> FvmTraces {
    let (fvm, decode_mult) = air::generate_trace(program, exec, MIN_ROWS);
    FvmTraces {
        fvm,
        decode: decode::generate_trace(program, &decode_mult, MIN_ROWS),
        mem: mem::generate_trace(&exec.mem, &exec.mem_mult, public, MIN_ROWS),
    }
}

pub fn prove_traces(
    program_id: &Commitment,
    public: &[PublicCell],
    traces: &mut FvmTraces,
    options: &ProofOptions,
) -> Result<FvmProof, ProvingError> {
    let pi = FvmPublicInputs {
        last_row: traces.fvm.num_rows() - 1,
    };
    let none = FvmPublicInputs::default();
    let airs = Airs::new(*program_id, options);
    let [fvm, dec, mem] = airs.refs();
    Prover::multi_prove(
        vec![
            (fvm, &mut traces.fvm, &pi),
            (dec, &mut traces.decode, &none),
            (mem, &mut traces.mem, &none),
        ],
        &mut transcript(program_id, public),
        #[cfg(feature = "disk-spill")]
        StorageMode::Ram,
    )
}

/// Proves `exec` with the cells at `public` as the public inputs and outputs.
pub fn prove(
    program: &Program,
    exec: &Execution,
    public: &[u64],
    options: &ProofOptions,
) -> Result<FvmProof, ProvingError> {
    prove_traces(
        &program_id(program, options),
        &public_cells(exec, public),
        &mut generate_traces(program, exec, public),
        options,
    )
}

/// Verifies that `program_id` halts with memory holding `public`, whose
/// addresses must be strictly increasing.
pub fn verify(
    program_id: &Commitment,
    public: &[PublicCell],
    proof: &FvmProof,
    options: &ProofOptions,
) -> bool {
    if !public.windows(2).all(|w| w[0].0 < w[1].0) {
        return false;
    }
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
    let mut t = transcript(program_id, public);
    let (z, alpha) = logup_challenges(&airs, proof, &mut t.clone());
    let Some(expected) = expected_public_balance(public, &z, &alpha) else {
        return false;
    };
    Verifier::multi_verify(&airs.refs(), proof, &mut t, &expected)
}
