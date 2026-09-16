//! The walk Approach 1 makes once per pass.
//!
//! The approach goes through the execution more than once: to commit the main
//! traces, to build the auxiliary columns against the challenge that commit
//! produced, and to open the Merkle trees. The three differ only in what they
//! do with a finished table — the walk itself, the end-of-run finalization and
//! the tables that are not built from an op list are the same every time, and
//! live here once.
//!
//! Trading re-execution for memory is the whole bargain: a pass never keeps a
//! chunk it has dealt with, so what it holds is one table plus the tables that
//! cannot be retired, no matter how long the run is.

use stark::proof::options::ProofOptions;

use crate::Error;
use crate::tables::MaxRowsConfig;
use crate::tables::trace_builder::{
    DecodeArtifacts, TableKind, Traces, WalkLeftover, build_initial_image,
};
use crate::tables::{register, types::*};
use executor::elf::Elf;
use stark::trace::TraceTable;

/// What a pass does with each table the walk produces.
///
/// `chunk` numbers the table within its kind and is continuous across the
/// walk's chunks and the tail the end-of-run step pads, so a visitor can index
/// by `(kind, chunk)` without knowing which of the two produced it.
pub trait Visitor {
    fn table(
        &mut self,
        kind: TableKind,
        chunk: usize,
        trace: TraceTable<GoldilocksField, GoldilocksExtension>,
    ) -> Result<(), Error>;

    /// Called once after the last table. A visitor that holds tables back to
    /// work on several at a time deals with the remainder here.
    fn flush(&mut self) -> Result<(), Error> {
        Ok(())
    }
}

/// How many tables a pass works on at a time.
///
/// The walk hands tables over one by one, and doing the work right there means
/// one table's worth of parallelism on a machine with far more of it. Holding
/// `k` back costs `k` times one table's working set and no more, which is the
/// bargain the whole approach is built on — bounded residency, not minimal.
///
/// Measured on the ethrex mainnet block, 96 cores, through the LogUp pass:
///
/// | k | time | peak |
/// |---|---|---|
/// | 1 | 233.4s | 21550 MB |
/// | 4 | 143.5s | 21554 MB |
/// | 8 | 129.9s | 21550 MB |
/// | 16 | 123.5s | 21550 MB |
/// | 32 | 122.3s | 26640 MB |
///
/// Up to 16 the peak does not move at all, because it is set at the end of the
/// run by the tables that cannot be retired — BITWISE, DECODE, the pages — and
/// a batch of chunks is small beside them. At 32 the batch itself becomes the
/// peak (it moves to halfway through the run) and buys 1% of time for 5 GB.
///
/// So the knee is 16 on that machine, and the bound is memory rather than
/// cores: a smaller run has a smaller resident set for a batch to hide behind.
/// `A1_TABLE_PARALLELISM` overrides it.
pub fn table_parallelism() -> usize {
    if let Some(k) = std::env::var("A1_TABLE_PARALLELISM")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|k| *k > 0)
    {
        return k;
    }
    std::thread::available_parallelism()
        .map(|n| (n.get() / 6).clamp(1, 16))
        .unwrap_or(1)
}

/// Collects tables until there are `k` of them, then hands the batch over.
///
/// Sits between the walk and a pass so the pass only says what to do with one
/// batch; the batching, and the bound on how many tables are alive, live here
/// once. The action is boxed so a pass names `Batched<'_, Item>` and not a
/// closure type.
pub struct Batched<'a, T> {
    batch: Vec<T>,
    k: usize,
    #[allow(clippy::type_complexity)]
    run: Box<dyn FnMut(Vec<T>) -> Result<(), Error> + 'a>,
}

impl<'a, T> Batched<'a, T> {
    pub fn new(run: impl FnMut(Vec<T>) -> Result<(), Error> + 'a) -> Self {
        Self {
            batch: Vec::new(),
            k: table_parallelism(),
            run: Box::new(run),
        }
    }

    pub fn push(&mut self, item: T) -> Result<(), Error> {
        self.batch.push(item);
        if self.batch.len() >= self.k {
            return self.drain();
        }
        Ok(())
    }

    pub fn drain(&mut self) -> Result<(), Error> {
        if self.batch.is_empty() {
            return Ok(());
        }
        (self.run)(std::mem::take(&mut self.batch))
    }
}

/// The tables a pass cannot retire, still as traces.
///
/// They are the ones not built from an op list — the preprocessed tables, the
/// accumulators and PAGE — so there is no compact intermediate to rebuild them
/// from and nothing to be gained by dropping them. Every pass gets them back
/// and decides what to do with them.
pub struct Resident {
    pub bitwise: TraceTable<GoldilocksField, GoldilocksExtension>,
    pub decode: TraceTable<GoldilocksField, GoldilocksExtension>,
    pub halt: TraceTable<GoldilocksField, GoldilocksExtension>,
    pub register: TraceTable<GoldilocksField, GoldilocksExtension>,
    pub pages: Vec<TraceTable<GoldilocksField, GoldilocksExtension>>,
    pub page_configs: Vec<crate::tables::page::PageConfig>,
    pub accumulated: crate::tables::trace_builder::AccumulatedTables,
    /// The bytes the run committed, which the statement binds into the
    /// transcript before any root is absorbed.
    pub public_output: Vec<u8>,
}

/// Walk the execution once, handing every chunked table to `visitor`.
///
/// The walk closes a table the moment it fills; what it could not close — the
/// partial tail of each kind, and every chunk of the kinds CPU32 and DVRM keep
/// feeding — is padded by [`finish`] and handed over the same way, numbered
/// where the walk left off.
pub fn run<V: Visitor>(
    elf: &Elf,
    private_input: &[u8],
    max_rows: &MaxRowsConfig,
    visitor: &mut V,
) -> Result<Resident, Error> {
    let walked = walk(elf, private_input, max_rows, visitor)?;
    finish(walked, private_input, max_rows, visitor)
}

/// What the walk leaves behind, and what [`finish`] needs to close it out.
///
/// The image and the decode artifacts are built once and kept because the
/// end-of-run step reads them: rebuilding them there would walk the ELF a
/// second time for no reason.
pub struct Walked {
    /// Everything the walk still held when the execution ended.
    pub leftover: WalkLeftover,
    artifacts: DecodeArtifacts,
    image: std::collections::HashMap<u64, u8>,
    register_init: Vec<u32>,
}

/// The walk alone.
///
/// Separate from [`finish`] for the caller that wants the leftover itself —
/// which is what shows the walk is retiring as it goes rather than deferring
/// every chunk to the end.
pub fn walk<V: Visitor>(
    elf: &Elf,
    private_input: &[u8],
    max_rows: &MaxRowsConfig,
    visitor: &mut V,
) -> Result<Walked, Error> {
    let image = build_initial_image(elf, private_input);
    let register_init = register::register_init_from_entry_point(elf.entry_point);
    let artifacts = DecodeArtifacts::from_elf(elf)?;

    let mut failed: Option<Error> = None;
    let leftover = Traces::walk_and_emit_chunks(
        &artifacts,
        elf,
        private_input.to_vec(),
        &image,
        &register_init,
        max_rows,
        |kind, chunk, table| {
            if failed.is_none()
                && let Err(e) = visitor.table(kind, chunk, table)
            {
                failed = Some(e);
            }
        },
    )?;
    match failed {
        Some(e) => Err(e),
        None => Ok(Walked {
            leftover,
            artifacts,
            image,
            register_init,
        }),
    }
}

/// Pad and hand over what the walk could not close, then build the tables that
/// stay.
///
/// The spec's step after the walk is "at the end of the execution, the
/// remaining tables are padded and commited to". Finalization comes first, as
/// it does in the ordinary build: HALT appends 33 register MEMW ops, and the
/// MEMW-derived LT ops are collected after them so those accesses get their
/// timestamp checks.
pub fn finish<V: Visitor>(
    walked: Walked,
    private_input: &[u8],
    max_rows: &MaxRowsConfig,
    visitor: &mut V,
) -> Result<Resident, Error> {
    let Walked {
        mut leftover,
        artifacts,
        image,
        register_init,
    } = walked;
    leftover.finalize(max_rows);

    // HALT and REGISTER first: REGISTER's final PC token is derived from the CPU
    // padding, and the padding of the tail cannot be counted once the tail has
    // been drained into a chunk below.
    let (halt, register) = leftover.build_halt_and_register(&register_init)?;

    for kind in ALL_CHUNKED {
        // Chunks the walk already closed keep their numbering; what is left
        // continues from there.
        let first = leftover.emitted(kind);
        for (offset, table) in leftover
            .take_remaining(kind, max_rows)
            .into_iter()
            .enumerate()
        {
            visitor.table(kind, first + offset, table)?;
        }
    }
    visitor.flush()?;

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
    let (pages, page_configs) = leftover.build_pages(&image, private_input, &mut hist);
    let bitwise = WalkLeftover::build_bitwise_from(&hist);

    Ok(Resident {
        bitwise,
        decode,
        halt,
        register,
        pages,
        page_configs,
        accumulated,
        public_output,
    })
}

/// Every chunked table, closable mid-walk or not.
pub const ALL_CHUNKED: [TableKind; 14] = [
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

/// The AIRs a pass dispatches a chunk to, one per kind.
///
/// The per-chunk AIRs differ only by the name used in reports — what a table
/// commits to depends on the trace and the domain — so one AIR per kind serves
/// every chunk of that kind, which is what lets a table be dealt with before
/// the number of chunks is known.
pub struct ChunkAirs {
    cpu: crate::VmAir,
    memw: crate::VmAir,
    memw_aligned: crate::VmAir,
    memw_register: crate::VmAir,
    load: crate::VmAir,
    cpu32: crate::VmAir,
    branch: crate::VmAir,
    eq: crate::VmAir,
    bytewise: crate::VmAir,
    store: crate::VmAir,
    lt: crate::VmAir,
    mul: crate::VmAir,
    dvrm: crate::VmAir,
    shift: crate::VmAir,
}

impl ChunkAirs {
    pub fn new(proof_options: &ProofOptions) -> Self {
        use crate::test_utils::*;
        Self {
            cpu: Box::new(create_cpu_air(proof_options)),
            memw: Box::new(create_memw_air(proof_options)),
            memw_aligned: Box::new(create_memw_aligned_air(proof_options)),
            memw_register: Box::new(create_memw_register_air(proof_options)),
            load: Box::new(create_load_air(proof_options)),
            cpu32: Box::new(create_cpu32_air(proof_options)),
            branch: Box::new(create_branch_air(proof_options)),
            eq: Box::new(create_eq_air(proof_options)),
            bytewise: Box::new(create_bytewise_air(proof_options)),
            store: Box::new(create_store_air(proof_options)),
            lt: Box::new(create_lt_air(proof_options)),
            mul: Box::new(create_mul_air(proof_options)),
            dvrm: Box::new(create_dvrm_air(proof_options)),
            shift: Box::new(create_shift_air(proof_options)),
        }
    }

    pub fn get(&self, kind: TableKind) -> &crate::VmAir {
        match kind {
            TableKind::Cpu => &self.cpu,
            TableKind::Memw => &self.memw,
            TableKind::MemwAligned => &self.memw_aligned,
            TableKind::MemwRegister => &self.memw_register,
            TableKind::Load => &self.load,
            TableKind::Cpu32 => &self.cpu32,
            TableKind::Branch => &self.branch,
            TableKind::Eq => &self.eq,
            TableKind::Bytewise => &self.bytewise,
            TableKind::Store => &self.store,
            TableKind::Lt => &self.lt,
            TableKind::Mul => &self.mul,
            TableKind::Dvrm => &self.dvrm,
            TableKind::Shift => &self.shift,
        }
    }
}
