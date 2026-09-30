//! Proving and verifying a Field VM execution: `FIELD_VM`, `FIELD_VM_DECODE`
//! and `FIELD_VM_MEM` in one `multi_prove`.
//!
//! The statement is the program id and the public memory cells. Every bus
//! closes internally except `FIELD_VM_PUBLIC`, whose balance the verifier
//! recomputes from the claimed cells.

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

/// The smallest blowup the constraint degree allows: 4 at the spec's degree 5.
pub fn default_options() -> ProofOptions {
    let blowup = (super::mux::D - 1).next_power_of_two().max(2) as u8;
    production_format(GoldilocksCubicProofOptions::with_blowup(blowup).expect("valid blowup"))
}

/// With `FVM_ZF_FORMAT` set, the recursion pipeline's proof format: its
/// Merkle caps and FRI fold schedule, and the terminal at degree 2^8, as the
/// LFM wraps are proved (`lfm::proof::aggregation_wrap_options`).
pub fn production_format(mut options: ProofOptions) -> ProofOptions {
    if std::env::var_os("FVM_ZF_FORMAT").is_some() {
        options.fri_final_poly_log_degree = 8;
        options = crate::zf_format::ZfFormat::global().options(options);
    }
    options
}

/// A table proved alongside the Field VM's own, sharing its buses.
pub type ExtraAir = Box<dyn AIR<Field = F, FieldExtension = E, PublicInputs = FvmPublicInputs>>;

struct Airs {
    /// One per FIELD_VM segment.
    fvm: Vec<AirWithBuses<F, E, FieldVmBoundary, FvmPublicInputs, FieldVmConstraints>>,
    /// One per DECODE chunk, each preprocessed under its program-id root.
    decode:
        Vec<AirWithBuses<F, E, NullBoundaryConstraintBuilder, FvmPublicInputs, EmptyConstraints>>,
    mem: AirWithBuses<F, E, MemBoundary, FvmPublicInputs, MemConstraints>,
    extra: Vec<ExtraAir>,
}

impl Airs {
    fn new(
        program_id: &[Commitment],
        options: &ProofOptions,
        segments: usize,
        extra: Vec<ExtraAir>,
        mem_lfm: bool,
    ) -> Self {
        let aux = |interactions| AuxiliaryTraceBuildData { interactions };
        let fvm = AirWithBuses::new(
            air::cols::NUM_COLUMNS,
            aux(air::bus_interactions()),
            options,
            1,
            FieldVmConstraints::new(),
        );
        Self {
            fvm: (0..segments)
                .map(|k| fvm.clone().with_name(&format!("FIELD_VM[{k}]")))
                .collect(),
            decode: program_id
                .iter()
                .enumerate()
                .map(|(k, root)| {
                    AirWithBuses::new(
                        decode::NUM_COLUMNS,
                        aux(decode::bus_interactions()),
                        options,
                        1,
                        EmptyConstraints,
                    )
                    .with_name(&format!("FIELD_VM_DECODE[{k}]"))
                    .with_preprocessed(*root, decode::NUM_PRECOMPUTED_COLS)
                })
                .collect(),
            mem: AirWithBuses::new(
                if mem_lfm {
                    mem::cols::NUM_COLUMNS_LFM
                } else {
                    mem::cols::NUM_COLUMNS
                },
                aux(if mem_lfm {
                    mem::bus_interactions_lfm()
                } else {
                    mem::bus_interactions()
                }),
                options,
                1,
                MemConstraints { lfm: mem_lfm },
            )
            .with_name("FIELD_VM_MEM"),
            extra,
        }
    }

    /// The segments, then the DECODE chunks, then MEM, then the extra tables.
    fn refs(&self) -> Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = FvmPublicInputs>> {
        let mut refs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = FvmPublicInputs>> =
            self.fvm.iter().map(|a| a as _).collect();
        refs.extend(
            self.decode.iter().map(|a| {
                a as &dyn AIR<Field = F, FieldExtension = E, PublicInputs = FvmPublicInputs>
            }),
        );
        refs.push(&self.mem);
        refs.extend(self.extra.iter().map(|a| a.as_ref()));
        refs
    }
}

/// The statement: program id, public cells and every segment boundary state.
fn transcript(
    program_id: &[Commitment],
    public: &[PublicCell],
    segments: &[FvmPublicInputs],
) -> crate::hash_pin::BlockTranscript {
    let mut t = crate::hash_pin::block_transcript(DOMAIN_TAG);
    t.append_bytes(&(program_id.len() as u64).to_le_bytes());
    for root in program_id {
        t.append_bytes(root);
    }
    t.append_bytes(&(public.len() as u64).to_le_bytes());
    for (addr, value) in public {
        t.append_bytes(&addr.to_le_bytes());
        for c in value.value() {
            t.append_bytes(&c.canonical().to_le_bytes());
        }
    }
    t.append_bytes(&(segments.len() as u64).to_le_bytes());
    for pi in segments {
        t.append_bytes(&(pi.last_row as u64).to_le_bytes());
        for w in pi.first.iter().chain(&pi.last) {
            t.append_bytes(&w.to_le_bytes());
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
    transcript: &mut crate::hash_pin::BlockTranscript,
) -> (FEE, FEE) {
    for (air, p) in airs.refs().into_iter().zip(&proof.proofs) {
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

/// One Merkle root per DECODE chunk.
pub type ProgramId = Vec<Commitment>;

pub fn program_id(program: &Program, options: &ProofOptions) -> ProgramId {
    decode::program_commitment(program, options, MIN_ROWS)
}

/// The traces of one execution: the FIELD_VM segments, DECODE and MEM.
pub struct FvmTraces {
    pub fvm: Vec<TraceTable<F, E>>,
    pub segments: Vec<FvmPublicInputs>,
    pub decode: Vec<TraceTable<F, E>>,
    pub mem: TraceTable<F, E>,
}

/// `public` are the addresses of the public cells.
pub fn generate_traces(program: &Program, exec: &Execution, public: &[u64]) -> FvmTraces {
    generate_traces_with(program, exec, public, None)
}

/// [`generate_traces`] with every FIELD_VM segment capped at `max_rows`.
pub fn generate_traces_with(
    program: &Program,
    exec: &Execution,
    public: &[u64],
    max_rows: Option<usize>,
) -> FvmTraces {
    generate_traces_reading(program, exec, public, max_rows, &[])
}

/// [`generate_traces_with`] where MEM also serves one read of each address in
/// `extra_reads` to a table outside the Field VM.
pub fn generate_traces_reading(
    program: &Program,
    exec: &Execution,
    public: &[u64],
    max_rows: Option<usize>,
    extra_reads: &[u64],
) -> FvmTraces {
    generate_traces_lfm(program, exec, public, max_rows, extra_reads, None)
}

/// [`generate_traces_reading`] with MEM on `LfmMem` for `lfm` cells.
pub fn generate_traces_lfm(
    program: &Program,
    exec: &Execution,
    public: &[u64],
    max_rows: Option<usize>,
    extra_reads: &[u64],
    lfm: Option<&[mem::LfmCell]>,
) -> FvmTraces {
    let mut seg = air::generate_segments(program, exec, MIN_ROWS, max_rows);
    for &a in extra_reads {
        let a = a as usize;
        if seg.mem_mult.len() <= a {
            seg.mem_mult.resize(a + 1, 0);
        }
        seg.mem_mult[a] += 1;
    }
    FvmTraces {
        fvm: seg.traces,
        segments: seg.public,
        decode: decode::generate_traces(program, &seg.decode_mult, MIN_ROWS),
        mem: match lfm {
            Some(cells) => {
                mem::generate_trace_lfm(&exec.mem, &seg.mem_mult, public, MIN_ROWS, cells)
            }
            None => mem::generate_trace(&exec.mem, &seg.mem_mult, public, MIN_ROWS),
        },
    }
}

pub fn prove_traces(
    program_id: &[Commitment],
    public: &[PublicCell],
    traces: &mut FvmTraces,
    options: &ProofOptions,
) -> Result<FvmProof, ProvingError> {
    prove_traces_with(program_id, public, traces, Vec::new(), &mut [], false, options)
}

/// [`prove_traces`] with `extra` tables, one trace each, after MEM.
pub fn prove_traces_with(
    program_id: &[Commitment],
    public: &[PublicCell],
    traces: &mut FvmTraces,
    extra: Vec<ExtraAir>,
    extra_traces: &mut [TraceTable<F, E>],
    mem_lfm: bool,
    options: &ProofOptions,
) -> Result<FvmProof, ProvingError> {
    assert_eq!(extra.len(), extra_traces.len(), "one trace per extra table");
    let none = FvmPublicInputs::default();
    let n = traces.fvm.len();
    let airs = Airs::new(program_id, options, n, extra, mem_lfm);
    let refs = airs.refs();
    let mut pairs: Vec<_> = refs
        .iter()
        .zip(traces.fvm.iter_mut())
        .zip(&traces.segments)
        .map(|((air, trace), pi)| (*air, trace, pi))
        .collect();
    let chunks = traces.decode.len();
    for (air, trace) in refs[n..n + chunks].iter().zip(traces.decode.iter_mut()) {
        pairs.push((*air, trace, &none));
    }
    pairs.push((refs[n + chunks], &mut traces.mem, &none));
    for (air, trace) in refs[n + chunks + 1..].iter().zip(extra_traces.iter_mut()) {
        pairs.push((*air, trace, &none));
    }
    crate::hash_pin::BlockProver::<F, E, FvmPublicInputs>::multi_prove(
        pairs,
        &mut transcript(program_id, public, &traces.segments),
        #[cfg(feature = "disk-spill")]
        StorageMode::Ram,
        stark::residency_mode::ResidencyMode::default(),
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
    program_id: &[Commitment],
    public: &[PublicCell],
    proof: &FvmProof,
    options: &ProofOptions,
) -> bool {
    verify_with(program_id, public, proof, Vec::new(), false, options)
}

/// [`verify`] for a proof carrying `extra` tables after MEM.
pub fn verify_with(
    program_id: &[Commitment],
    public: &[PublicCell],
    proof: &FvmProof,
    extra: Vec<ExtraAir>,
    mem_lfm: bool,
    options: &ProofOptions,
) -> bool {
    if !public.windows(2).all(|w| w[0].0 < w[1].0) {
        return false;
    }
    let Some(n) = proof
        .proofs
        .len()
        .checked_sub(program_id.len() + 1 + extra.len())
        .filter(|&n| n >= 1)
    else {
        return false;
    };
    let (fvm, rest) = proof.proofs.split_at(n);
    if rest
        .iter()
        .any(|p| p.public_inputs != FvmPublicInputs::default())
    {
        return false;
    }
    let segments: Vec<FvmPublicInputs> = fvm.iter().map(|p| p.public_inputs.clone()).collect();
    // The chain: starts at the start state, each segment ends where the next
    // begins, and the last ends in the halt state.
    let well_formed = fvm.iter().zip(&segments).all(|(p, pi)| {
        pi.last_row + 1 == p.trace_length
            && pi.first.len() == air::STATE_WORDS
            && pi.last.len() == air::STATE_WORDS
    });
    let chained = segments.windows(2).all(|w| w[0].last == w[1].first);
    if !well_formed
        || !chained
        || segments[0].first != air::start_words()
        || segments[n - 1].last != air::halt_words()
    {
        return false;
    }
    let airs = Airs::new(program_id, options, n, extra, mem_lfm);
    let mut t = transcript(program_id, public, &segments);
    let (z, alpha) = logup_challenges(&airs, proof, &mut t.clone());
    let Some(expected) = expected_public_balance(public, &z, &alpha) else {
        return false;
    };
    crate::hash_pin::BlockVerifier::<F, E, FvmPublicInputs>::multi_verify(
        &airs.refs(),
        proof,
        &mut t,
        &expected,
    )
}
