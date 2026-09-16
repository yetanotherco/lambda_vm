//! Approach 1's Commit phase: walk the execution, committing and retiring each
//! table as it fills.
//!
//! The spec has the prover go through execution "and once the memory pressure
//! becomes too large, batch commit to all full tables in memory; then these
//! tables are dropped". This is that pass, expressed as a [`pass::Visitor`]:
//! what it does with a finished table is commit its main trace and let the
//! table die. What it produces is a root per chunk and, at the end, the tables
//! that cannot be retired — which the Challenge phase commits and samples from.

use stark::proof::options::ProofOptions;
use stark::prover::{IsStarkProver, MainRoots};

use crate::Error;
use crate::pass::{self, ChunkAirs, Resident, Visitor};
use crate::tables::MaxRowsConfig;
use crate::tables::trace_builder::{TableKind, Traces};
use crate::tables::types::*;
use executor::elf::Elf;
use stark::trace::TraceTable;

/// One chunked table's commitment, by kind and position.
pub type ChunkCommitment = (TableKind, usize, MainRoots);

/// Every chunk committed, and what the Commit phase leaves resident.
pub struct Committed {
    /// A root per chunk of every chunked table, walk-closed and tail alike.
    pub chunks: Vec<ChunkCommitment>,
    /// The tables the walk could not commit, still as traces.
    pub remaining: Resident,
}

/// What the walk alone produced.
pub struct CommitPhase {
    /// One entry per chunk closed during the walk. Unordered: the batch runs
    /// its tables in parallel, so what reads this indexes by `(kind, chunk)`.
    pub closed: Vec<ChunkCommitment>,
    /// Everything the walk still held when the execution ended.
    pub walked: pass::Walked,
}

/// One table on its way to a pass: what it is, where it sits, and its trace.
type Item = (
    TableKind,
    usize,
    TraceTable<GoldilocksField, GoldilocksExtension>,
);

/// Commits each table's main trace and drops it, `k` tables at a time.
struct CommitMain<'a> {
    batch: pass::Batched<'a, Item>,
}

impl<'a> CommitMain<'a> {
    fn new(airs: &'a ChunkAirs, roots: &'a std::sync::Mutex<Vec<ChunkCommitment>>) -> Self {
        Self {
            batch: pass::Batched::new(move |items| commit_batch(airs, roots, items)),
        }
    }
}

/// One batch, in parallel. The tables in a batch are independent — each commits
/// its own trace against its own AIR — so the only shared thing is where the
/// roots land.
fn commit_batch(
    airs: &ChunkAirs,
    roots: &std::sync::Mutex<Vec<ChunkCommitment>>,
    items: Vec<Item>,
) -> Result<(), Error> {
    use rayon::prelude::*;
    type P = stark::prover::Prover<GoldilocksField, GoldilocksExtension, ()>;
    let done: Result<Vec<ChunkCommitment>, Error> = items
        .into_par_iter()
        .map(|(kind, chunk, trace)| {
            <P as IsStarkProver<_, _, _>>::commit_table_root(airs.get(kind).as_ref(), &trace)
                .map(|root| (kind, chunk, root))
                .ok_or_else(|| {
                    Error::Prover(format!("commit phase: no commitment for a {kind:?} chunk"))
                })
        })
        .collect();
    roots.lock().expect("roots").extend(done?);
    Ok(())
}

impl Visitor for CommitMain<'_> {
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

/// The Commit phase's walk, stopping before the end-of-run tables.
///
/// Each chunk is committed the moment it fills and its trace is dropped, so
/// nothing that has been committed is still resident.
pub fn run(
    elf: &Elf,
    private_input: &[u8],
    max_rows: &MaxRowsConfig,
    proof_options: &ProofOptions,
) -> Result<CommitPhase, Error> {
    let airs = ChunkAirs::new(proof_options);
    let roots = std::sync::Mutex::new(Vec::new());
    let mut visitor = CommitMain::new(&airs, &roots);
    let walked = pass::walk(elf, private_input, max_rows, &mut visitor)?;
    visitor.flush()?;
    drop(visitor);
    Ok(CommitPhase {
        closed: roots.into_inner().expect("roots"),
        walked,
    })
}

/// The Commit phase end to end.
///
/// The walk, then the padding of everything it could not close. What comes back
/// is a root per chunk of every chunked table and, still resident, only the
/// tables that are not built from an op list: the preprocessed ones and the
/// accumulators. That is the state the Challenge phase starts from.
pub fn run_to_end(
    elf: &Elf,
    private_input: &[u8],
    max_rows: &MaxRowsConfig,
    proof_options: &ProofOptions,
) -> Result<Committed, Error> {
    let airs = ChunkAirs::new(proof_options);
    let roots = std::sync::Mutex::new(Vec::new());
    let mut visitor = CommitMain::new(&airs, &roots);
    let remaining = pass::run(elf, private_input, max_rows, &mut visitor)?;
    drop(visitor);
    Ok(Committed {
        chunks: roots.into_inner().expect("roots"),
        remaining,
    })
}

/// Pad and commit what a walk could not close.
///
/// The Commit phase's half of the end-of-run step, kept as its own entry point
/// for the caller that ran [`run`] and wants the tails separately.
pub fn commit_remaining(
    walked: pass::Walked,
    private_input: &[u8],
    max_rows: &MaxRowsConfig,
    proof_options: &ProofOptions,
) -> Result<(Vec<ChunkCommitment>, Resident), Error> {
    let airs = ChunkAirs::new(proof_options);
    let roots = std::sync::Mutex::new(Vec::new());
    let mut visitor = CommitMain::new(&airs, &roots);
    let resident = pass::finish(walked, private_input, max_rows, &mut visitor)?;
    drop(visitor);
    Ok((roots.into_inner().expect("roots"), resident))
}

/// The ordinary build, for comparison against [`run_to_end`].
///
/// The same execution and the same tables, built the way `prove` builds them:
/// every trace resident at once and nothing committed. It lives beside the
/// Commit phase so the two arms of the comparison are driven identically, and
/// so the storage-mode argument stays on this side of the feature gate — the
/// lint enables `lambda-vm-prover/disk-spill` without enabling the CLI's, and a
/// caller there would not agree with this signature.
pub fn build_resident(
    elf: &Elf,
    private_input: &[u8],
    max_rows: &MaxRowsConfig,
) -> Result<Traces, Error> {
    let executed = executor::vm::execution::Executor::new(elf, private_input.to_vec())
        .and_then(|e| e.run())
        .map_err(|e| Error::Prover(format!("execution failed: {e}")))?;
    Traces::from_elf_and_logs(
        elf,
        &executed.logs,
        max_rows,
        private_input,
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
}
