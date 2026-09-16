//! Approach 1's Commit phase: walk the execution, committing and retiring each
//! table as it fills.
//!
//! The spec has the prover go through execution "and once the memory pressure
//! becomes too large, batch commit to all full tables in memory; then these
//! tables are dropped". This is that pass. What it produces is a commitment per
//! closed chunk and, at the end, whatever the walk could not close — which the
//! Challenge phase pads and commits, as the spec's next step.

use stark::proof::options::ProofOptions;
use stark::prover::{IsStarkProver, MainRoots};

use crate::Error;
use crate::tables::MaxRowsConfig;
use crate::tables::trace_builder::{TableKind, Traces, WalkLeftover, build_initial_image};
use crate::tables::{register, types::*};
use executor::elf::Elf;
use stark::trace::TraceTable;

/// What the Commit phase produced.
pub struct CommitPhase {
    /// One entry per chunk closed during the walk, in the order they closed.
    pub closed: Vec<ChunkCommitment>,
    /// Everything the walk still held when the execution ended.
    pub leftover: WalkLeftover,
}

/// One chunked table's commitment, by kind and position.
pub type ChunkCommitment = (TableKind, usize, MainRoots);

/// Run the Commit phase over `elf`.
///
/// Each chunk is committed the moment it fills and its trace is dropped, so
/// nothing that has been committed is still resident. The per-chunk AIRs differ
/// only by the name used in reports — the commitment depends on the trace and
/// the domain — so one AIR per kind serves every chunk of that kind, which is
/// what lets a table be committed before the number of chunks is known.
pub fn run(
    elf: &Elf,
    private_input: &[u8],
    max_rows: &MaxRowsConfig,
    proof_options: &ProofOptions,
) -> Result<CommitPhase, Error> {
    let image = build_initial_image(elf, private_input);
    let register_init = register::register_init_from_entry_point(elf.entry_point);
    let artifacts = crate::tables::trace_builder::DecodeArtifacts::from_elf(elf)?;
    walk_and_commit(
        &artifacts,
        elf,
        private_input,
        &image,
        &register_init,
        max_rows,
        proof_options,
    )
}

/// The walk itself, over an already-built image and decode table.
#[allow(clippy::too_many_arguments)]
fn walk_and_commit<I: crate::paged_mem::ImageSource + Sync>(
    artifacts: &crate::tables::trace_builder::DecodeArtifacts,
    elf: &Elf,
    private_input: &[u8],
    image: &I,
    register_init: &[u32],
    max_rows: &MaxRowsConfig,
    proof_options: &ProofOptions,
) -> Result<CommitPhase, Error> {
    let cpu = crate::test_utils::create_cpu_air(proof_options);
    let memw = crate::test_utils::create_memw_air(proof_options);
    let memw_aligned = crate::test_utils::create_memw_aligned_air(proof_options);
    let memw_register = crate::test_utils::create_memw_register_air(proof_options);
    let load = crate::test_utils::create_load_air(proof_options);
    let cpu32 = crate::test_utils::create_cpu32_air(proof_options);
    let branch = crate::test_utils::create_branch_air(proof_options);
    let eq = crate::test_utils::create_eq_air(proof_options);
    let bytewise = crate::test_utils::create_bytewise_air(proof_options);
    let store = crate::test_utils::create_store_air(proof_options);

    let mut closed = Vec::new();
    let mut failed: Option<TableKind> = None;
    let leftover = Traces::walk_and_emit_chunks(
        artifacts,
        elf,
        private_input.to_vec(),
        image,
        register_init,
        max_rows,
        |kind, chunk, table| {
            let air: &dyn stark::traits::AIR<
                Field = GoldilocksField,
                FieldExtension = GoldilocksExtension,
                PublicInputs = (),
            > = match kind {
                TableKind::Cpu => &cpu,
                TableKind::Memw => &memw,
                TableKind::MemwAligned => &memw_aligned,
                TableKind::MemwRegister => &memw_register,
                TableKind::Load => &load,
                TableKind::Cpu32 => &cpu32,
                TableKind::Branch => &branch,
                TableKind::Eq => &eq,
                TableKind::Bytewise => &bytewise,
                TableKind::Store => &store,
                other => {
                    // Unreachable through `CHUNKED_KINDS`; recorded rather than
                    // panicked so a kind added there without an AIR here fails
                    // the run instead of committing under the wrong one.
                    failed = Some(other);
                    return;
                }
            };
            type P = stark::prover::Prover<GoldilocksField, GoldilocksExtension, ()>;
            match <P as IsStarkProver<_, _, _>>::commit_table_root(air, &table) {
                Some(root) => closed.push((kind, chunk, root)),
                None => failed = Some(kind),
            }
        },
    )?;

    if let Some(kind) = failed {
        return Err(Error::Prover(format!(
            "commit phase: no commitment for a {kind:?} chunk"
        )));
    }

    Ok(CommitPhase { closed, leftover })
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
    let image = build_initial_image(elf, private_input);
    let register_init = register::register_init_from_entry_point(elf.entry_point);
    let artifacts = crate::tables::trace_builder::DecodeArtifacts::from_elf(elf)?;

    let phase = walk_and_commit(
        &artifacts,
        elf,
        private_input,
        &image,
        &register_init,
        max_rows,
        proof_options,
    )?;
    let mut chunks = phase.closed;
    let mut remaining = commit_remaining(
        phase.leftover,
        &artifacts,
        &image,
        &register_init,
        private_input,
        max_rows,
        proof_options,
    )?;
    chunks.append(&mut remaining.chunks);

    Ok(Committed { chunks, remaining })
}

/// Every chunk committed, and what the Commit phase leaves resident.
pub struct Committed {
    /// A root per chunk of every chunked table, walk-closed and tail alike.
    pub chunks: Vec<ChunkCommitment>,
    /// The tables the walk could not commit, still as traces.
    pub remaining: Remaining,
}

/// The Challenge phase's first half: pad and commit what the walk could not
/// close.
///
/// The spec's step after Commit is "at the end of the execution, the remaining
/// tables are padded and commited to". That is this: the partial tail of every
/// table the walk closed, plus every chunk of the tables it could not close
/// because CPU32 and DVRM keep feeding them.
///
/// Finalization comes first, as it does in the ordinary build: HALT appends 33
/// register MEMW ops, and the MEMW-derived LT ops are collected after them so
/// those accesses get their timestamp checks.
///
/// The preprocessed and accumulator tables — BITWISE, DECODE, REGISTER, PAGE and
/// the rest — are not here yet. They are built from the ELF and from counts
/// accumulated across the whole run rather than from an op list, so they need
/// their own step.
#[allow(clippy::too_many_arguments)]
pub fn commit_remaining<I: crate::paged_mem::ImageSource + Sync>(
    mut leftover: WalkLeftover,
    artifacts: &crate::tables::trace_builder::DecodeArtifacts,
    initial_image: &I,
    register_init: &[u32],
    private_input: &[u8],
    max_rows: &MaxRowsConfig,
    proof_options: &ProofOptions,
) -> Result<Remaining, Error> {
    leftover.finalize(max_rows);

    let cpu = crate::test_utils::create_cpu_air(proof_options);
    let memw = crate::test_utils::create_memw_air(proof_options);
    let memw_aligned = crate::test_utils::create_memw_aligned_air(proof_options);
    let memw_register = crate::test_utils::create_memw_register_air(proof_options);
    let load = crate::test_utils::create_load_air(proof_options);
    let cpu32 = crate::test_utils::create_cpu32_air(proof_options);
    let branch = crate::test_utils::create_branch_air(proof_options);
    let eq = crate::test_utils::create_eq_air(proof_options);
    let bytewise = crate::test_utils::create_bytewise_air(proof_options);
    let store = crate::test_utils::create_store_air(proof_options);
    let lt = crate::test_utils::create_lt_air(proof_options);
    let mul = crate::test_utils::create_mul_air(proof_options);
    let dvrm = crate::test_utils::create_dvrm_air(proof_options);
    let shift = crate::test_utils::create_shift_air(proof_options);

    // HALT and REGISTER first: REGISTER's final PC token is derived from the CPU
    // padding, and the padding of the tail cannot be counted once the tail has
    // been drained into a chunk below.
    let (halt, register) = leftover.build_halt_and_register(register_init)?;

    type P = stark::prover::Prover<GoldilocksField, GoldilocksExtension, ()>;
    let mut out = Vec::new();
    for kind in ALL_CHUNKED {
        let air: &dyn stark::traits::AIR<
            Field = GoldilocksField,
            FieldExtension = GoldilocksExtension,
            PublicInputs = (),
        > = match kind {
            TableKind::Cpu => &cpu,
            TableKind::Memw => &memw,
            TableKind::MemwAligned => &memw_aligned,
            TableKind::MemwRegister => &memw_register,
            TableKind::Load => &load,
            TableKind::Cpu32 => &cpu32,
            TableKind::Branch => &branch,
            TableKind::Eq => &eq,
            TableKind::Bytewise => &bytewise,
            TableKind::Store => &store,
            TableKind::Lt => &lt,
            TableKind::Mul => &mul,
            TableKind::Dvrm => &dvrm,
            TableKind::Shift => &shift,
        };
        // Chunks the walk already closed keep their numbering; what is left
        // continues from there.
        let first = leftover.emitted(kind);
        for (offset, table) in leftover
            .take_remaining(kind, max_rows)
            .into_iter()
            .enumerate()
        {
            let root =
                <P as IsStarkProver<_, _, _>>::commit_table_root(air, &table).ok_or_else(|| {
                    Error::Prover(format!("commit phase: no commitment for a {kind:?} tail"))
                })?;
            out.push((kind, first + offset, root));
        }
    }
    // BITWISE comes back as a table rather than a commitment: it is
    // preprocessed, so its commitment splits into two trees, and that path does
    // not exist here yet. The lookups are what this phase is responsible for
    // having kept — the chunks that owed them are long gone.
    let public_output = leftover.public_output_bytes();
    let accumulated = leftover.build_accumulated();
    let decode = leftover.build_decode(
        artifacts.decode_trace.clone(),
        &artifacts.decode_pc_to_row,
        max_rows,
    );
    // PAGE last: it owes BITWISE lookups of its own, so BITWISE is written only
    // once those are in.
    let mut hist = leftover.bitwise_histogram();
    let (pages, page_configs) = leftover.build_pages(initial_image, private_input, &mut hist);
    let bitwise = WalkLeftover::build_bitwise_from(&hist);

    Ok(Remaining {
        chunks: out,
        public_output,
        bitwise,
        decode,
        halt,
        register,
        pages,
        page_configs,
        accumulated,
    })
}

/// What the end-of-run phase produced.
pub struct Remaining {
    /// The tails, and every chunk of the tables the walk could not close.
    ///
    /// [`run_to_end`] drains this into [`Committed::chunks`]; read it there.
    pub chunks: Vec<ChunkCommitment>,
    /// The bytes the run committed, which the statement binds into the
    /// transcript before any root is absorbed.
    pub public_output: Vec<u8>,
    /// The BITWISE table, carrying the lookups of every retired chunk.
    pub bitwise: TraceTable<GoldilocksField, GoldilocksExtension>,
    /// The DECODE table, with one lookup counted per executed cycle and per
    /// padding row.
    pub decode: TraceTable<GoldilocksField, GoldilocksExtension>,
    /// HALT, from the run's terminating ECALL.
    pub halt: TraceTable<GoldilocksField, GoldilocksExtension>,
    /// REGISTER, whose final PC token has to match the last padding write.
    pub register: TraceTable<GoldilocksField, GoldilocksExtension>,
    /// The PAGE tables and their configs.
    pub pages: Vec<TraceTable<GoldilocksField, GoldilocksExtension>>,
    pub page_configs: Vec<crate::tables::page::PageConfig>,
    /// The tables written once from an accumulated op list.
    pub accumulated: crate::tables::trace_builder::AccumulatedTables,
}

/// Every chunked table, closable mid-walk or not.
const ALL_CHUNKED: [TableKind; 14] = [
    TableKind::Cpu,
    TableKind::Memw,
    TableKind::MemwAligned,
    TableKind::MemwRegister,
    TableKind::Load,
    TableKind::Cpu32,
    TableKind::Branch,
    TableKind::Eq,
    TableKind::Bytewise,
    TableKind::Store,
    TableKind::Lt,
    TableKind::Mul,
    TableKind::Dvrm,
    TableKind::Shift,
];
