//! Approach 1's proving pass: walk the execution again and prove each table
//! against the challenge the Commit phase produced.
//!
//! The spec's third step is a re-execution. It has to be: the LogUp columns are
//! a function of the challenge, and the challenge is not known until every main
//! root has been absorbed — by which time the tables that produced them are
//! gone. The ordinary prover avoids the second walk by keeping every trace
//! resident across the Round 1 barrier, which is exactly the residency this
//! approach refuses to pay.
//!
//! So the tables are rebuilt. `build_main` is deterministic, so the trace a
//! chunk gets here is byte-identical to the one the Commit phase committed, and
//! the aux columns are therefore the ones that root answers for.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::fiat_shamir::is_transcript::IsTranscript;
use stark::proof::options::ProofOptions;
use stark::proof::stark::StarkProof;
use stark::prover::IsStarkProver;

use crate::Error;
use crate::challenge_phase::Challenge;
use crate::pass::{self, ChunkAirs, Resident, Visitor};
use crate::streaming::NUM_FIXED_AIRS;
use crate::tables::MaxRowsConfig;
use crate::tables::trace_builder::TableKind;
use crate::tables::types::*;
use executor::elf::Elf;
use math::field::element::FieldElement;
use stark::trace::TraceTable;

/// What the pass produced.
pub struct LogUp {
    /// One proof per table, in `VmAirs::air_trace_pairs` order.
    pub tables: Vec<StarkProof<GoldilocksField, GoldilocksExtension, ()>>,
    /// The tables the pass could not retire, rebuilt by this walk.
    pub resident: Resident,
}

type Item = (
    TableKind,
    usize,
    TraceTable<GoldilocksField, GoldilocksExtension>,
);

/// One table's finished proof, tagged with where it sits in the AIR order.
type Proved = (usize, StarkProof<GoldilocksField, GoldilocksExtension, ()>);

/// Where a batch of proofs lands. Shared because the batch runs in parallel.
type Proofs = std::sync::Mutex<Vec<Proved>>;

struct BuildAux<'a> {
    batch: pass::Batched<'a, Item>,
}

impl<'a> BuildAux<'a> {
    fn new(airs: &'a ChunkAirs, challenge: &'a Challenge, done: &'a Proofs) -> Self {
        Self {
            batch: pass::Batched::new(move |items| rounds_batch(airs, challenge, done, items)),
        }
    }
}

/// One batch, in parallel. Each table runs against its own transcript fork, so
/// nothing crosses between them — the shared state is only where results land.
fn rounds_batch(
    airs: &ChunkAirs,
    challenge: &Challenge,
    done: &Proofs,
    items: Vec<Item>,
) -> Result<(), Error> {
    use rayon::prelude::*;
    let order = &challenge.order;
    let n = order.len();
    let built: Result<Vec<_>, Error> = items
        .into_par_iter()
        .map(|(kind, chunk, mut trace)| {
            let idx = order.index_of(kind, chunk).ok_or_else(|| {
                Error::Prover(format!(
                    "logup phase: {kind:?} chunk {chunk} is not in the layout the Commit phase produced"
                ))
            })?;
            let mut transcript = fork(&challenge.transcript, idx, n);
            let rounds = prove_table(
                airs.get(kind).as_ref(),
                &mut trace,
                &challenge.challenges,
                &mut transcript,
            )
            .map_err(|e| Error::Prover(format!("logup phase: {kind:?} chunk {chunk}: {e}")))?;
            Ok((idx, rounds))
        })
        .collect();
    done.lock().expect("logup results").extend(built?);
    Ok(())
}

impl Visitor for BuildAux<'_> {
    fn table(
        &mut self,
        kind: TableKind,
        chunk: usize,
        trace: TraceTable<GoldilocksField, GoldilocksExtension>,
    ) -> Result<(), Error> {
        self.batch.push((kind, chunk, trace))
    }

    fn flush(&mut self) -> Result<(), Error> {
        self.batch.drain()
    }
}

/// A table's own transcript: the shared state after the challenge, separated by
/// AIR index. Reproduces the fused prover's forking exactly — a single-table
/// proof takes no index, and getting that wrong shifts every challenge.
fn fork(
    shared: &DefaultTranscript<GoldilocksExtension>,
    idx: usize,
    num_airs: usize,
) -> DefaultTranscript<GoldilocksExtension> {
    let mut t = shared.clone();
    if num_airs > 1 {
        t.append_bytes(&(idx as u64).to_le_bytes());
    }
    t
}

fn prove_table(
    air: &dyn stark::traits::AIR<
        Field = GoldilocksField,
        FieldExtension = GoldilocksExtension,
        PublicInputs = (),
    >,
    trace: &mut TraceTable<GoldilocksField, GoldilocksExtension>,
    challenges: &[FieldElement<GoldilocksExtension>],
    transcript: &mut DefaultTranscript<GoldilocksExtension>,
) -> Result<StarkProof<GoldilocksField, GoldilocksExtension, ()>, String> {
    type P = stark::prover::Prover<GoldilocksField, GoldilocksExtension, ()>;
    <P as IsStarkProver<_, _, _>>::prove_table_from_trace(air, &(), trace, challenges, transcript)
        .map_err(|e| format!("{e:?}"))
}

/// Run the LogUp pass over `elf`, against the challenge `challenge` sampled.
///
/// `challenge` has to come from the Commit phase's roots over this same
/// execution: an aux trace built against a different challenge commits to a bus
/// the main traces never balanced.
pub fn run(
    elf: &Elf,
    private_input: &[u8],
    max_rows: &MaxRowsConfig,
    proof_options: &ProofOptions,
    challenge: &Challenge,
) -> Result<LogUp, Error> {
    let airs = ChunkAirs::new(proof_options);
    let done = std::sync::Mutex::new(Vec::new());
    let mut visitor = BuildAux::new(&airs, challenge, &done);
    let mut resident = pass::run(elf, private_input, max_rows, &mut visitor)?;
    drop(visitor);

    let chunks = done.into_inner().expect("logup results");
    let tables = assemble(chunks, &mut resident, elf, proof_options, challenge)?;
    Ok(LogUp { tables, resident })
}

/// Every table's rounds 2-3 in `VmAirs::air_trace_pairs` order.
///
/// The chunked tables come back keyed by the index the walk resolved; the
/// tables that stay have theirs built here, as the Challenge phase built their
/// mains.
fn assemble(
    chunks: Vec<Proved>,
    resident: &mut Resident,
    elf: &Elf,
    proof_options: &ProofOptions,
    challenge: &Challenge,
) -> Result<Vec<StarkProof<GoldilocksField, GoldilocksExtension, ()>>, Error> {
    let order = &challenge.order;
    let airs = crate::VmAirs::new(
        elf,
        proof_options,
        false,
        &resident.page_configs,
        order.counts(),
        None,
        true,
        None,
        None,
        None,
    );

    let mut slots: Vec<Option<StarkProof<GoldilocksField, GoldilocksExtension, ()>>> =
        (0..order.len()).map(|_| None).collect();
    for (idx, rounds) in chunks {
        let slot = slots
            .get_mut(idx)
            .ok_or_else(|| Error::Prover(format!("logup phase: table {idx} is past the layout")))?;
        if slot.is_some() {
            return Err(Error::Prover(format!(
                "logup phase: table {idx} was built twice"
            )));
        }
        *slot = Some(rounds);
    }

    let ch = &challenge.challenges;
    let n = order.len();
    let build = |idx: usize,
                 air: &crate::VmAir,
                 trace: &mut TraceTable<GoldilocksField, GoldilocksExtension>|
     -> Result<StarkProof<GoldilocksField, GoldilocksExtension, ()>, Error> {
        let mut transcript = fork(&challenge.transcript, idx, n);
        prove_table(air.as_ref(), trace, ch, &mut transcript)
            .map_err(|e| Error::Prover(format!("logup phase: table {idx}: {e}")))
    };

    let fixed: [(
        &crate::VmAir,
        &mut TraceTable<GoldilocksField, GoldilocksExtension>,
    ); NUM_FIXED_AIRS] = [
        (&airs.bitwise, &mut resident.bitwise),
        (&airs.decode, &mut resident.decode),
        (&airs.commit, &mut resident.accumulated.commit),
        (&airs.keccak, &mut resident.accumulated.keccak),
        (&airs.keccak_rnd, &mut resident.accumulated.keccak_rnd),
        (&airs.keccak_rc, &mut resident.accumulated.keccak_rc),
        (&airs.ecsm, &mut resident.accumulated.ecsm),
        (&airs.ecdas, &mut resident.accumulated.ecdas),
        (&airs.hint, &mut resident.accumulated.hint),
        (&airs.register, &mut resident.register),
    ];
    for (idx, (air, trace)) in fixed.into_iter().enumerate() {
        slots[idx] = Some(build(idx, air, trace)?);
    }
    if airs.include_halt {
        slots[NUM_FIXED_AIRS] = Some(build(NUM_FIXED_AIRS, &airs.halt, &mut resident.halt)?);
    }
    for (i, (air, trace)) in airs.pages.iter().zip(resident.pages.iter_mut()).enumerate() {
        let idx = order
            .page_index(i)
            .ok_or_else(|| Error::Prover(format!("logup phase: page {i} is not in the layout")))?;
        slots[idx] = Some(build(idx, air, trace)?);
    }

    slots
        .into_iter()
        .enumerate()
        .map(|(idx, slot)| {
            slot.ok_or_else(|| Error::Prover(format!("logup phase: table {idx} was never built")))
        })
        .collect()
}
