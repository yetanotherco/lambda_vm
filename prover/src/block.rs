//! One proof per block, no epochs: the monolithic prover over a whole block at
//! uniform full-height table instances, on the device.
//!
//! The proof is the monolithic [`VmProof`], checked by the unchanged
//! [`crate::verify_with_options`]: the statement (`StatementKind::Monolithic`),
//! then every instance's main root in AIR order, one set of LogUp challenges,
//! and a transcript fork per instance for its aux, quotient and FRI. Memory
//! across the block is the monolithic PAGE argument; there is no
//! local-to-global table and no global proof.
//!
//! What makes a whole block fit the card is the residency mode, not the
//! protocol: under [`ResidencyMode::RecomputeLdeDevice`] Round 1 keeps only
//! each instance's root, and each instance's fused task commits its trace on
//! the device again (root checked equal to the absorbed one). The host keeps
//! the traces between the two phases.
//!
//! This is a measurement driver beside the epoch pipeline
//! ([`crate::continuation`]), which stays the production path.

use std::time::Instant;

use executor::elf::Elf;
use executor::vm::execution::Executor;
use stark::prover::IsStarkProver;
use stark::residency_mode::ResidencyMode;

use crate::statement::{StatementKind, absorb_statement};
use crate::tables::MaxRowsConfig;
use crate::tables::trace_builder::Traces;
use crate::{Commitment, Error, ProofOptions, VmAirs, VmProof};

/// Rows per full-height instance: every splittable table is cut into
/// instances of `2^BLOCK_ROWS_LOG2` rows (the tail padded to its power of two).
pub const BLOCK_ROWS_LOG2: u32 = 21;

/// The block's table caps: [`BLOCK_ROWS_LOG2`] for every splittable table.
pub fn block_max_rows() -> MaxRowsConfig {
    MaxRowsConfig::uniform(1 << BLOCK_ROWS_LOG2)
}

/// Wall times of one block prove, in seconds, for the readout.
#[derive(Debug, Clone, Default)]
pub struct BlockTimes {
    /// The whole block's execution (DECODE's root is derived beside it).
    pub execute: f64,
    /// Every trace, built from the execution logs.
    pub build: f64,
    /// AIR construction, the statement and the transcript.
    pub setup: f64,
    /// `multi_prove`: both phases, every instance.
    pub prove: f64,
}

impl BlockTimes {
    /// Execution to proof.
    pub fn total(&self) -> f64 {
        self.execute + self.build + self.setup + self.prove
    }
}

/// [`prove_block_with`] at the block's shape: [`block_max_rows`] and
/// [`ResidencyMode::RecomputeLdeDevice`].
pub fn prove_block(
    elf_bytes: &[u8],
    private_input: &[u8],
    opts: &ProofOptions,
) -> Result<(VmProof, BlockTimes), Error> {
    prove_block_with(
        elf_bytes,
        private_input,
        opts,
        &block_max_rows(),
        ResidencyMode::RecomputeLdeDevice,
    )
}

/// Execute the whole program, build every trace at `max_rows`, and prove it as
/// one monolithic proof under `residency`. DECODE's commitment, a function of
/// the ELF and `opts` alone, is derived on a helper thread beside the
/// execution and handed to the AIRs, as the epoch pipeline does.
pub fn prove_block_with(
    elf_bytes: &[u8],
    private_input: &[u8],
    opts: &ProofOptions,
    max_rows: &MaxRowsConfig,
    residency: ResidencyMode,
) -> Result<(VmProof, BlockTimes), Error> {
    let mut times = BlockTimes::default();
    let program = Elf::load(elf_bytes).map_err(|e| Error::ElfLoad(format!("{e}")))?;

    let t = Instant::now();
    let (decode_commitment, run) = std::thread::scope(|s| {
        let decode = s.spawn(|| {
            crate::tables::decode::commitment_from_elf_device_or_host(&program, opts)
                .map_err(|e| Error::Recursion(format!("DECODE commitment from ELF: {e}")))
        });
        let run = Executor::new(&program, private_input.to_vec())
            .map_err(|e| Error::Execution(format!("{e}")))
            .and_then(|executor| executor.run().map_err(|e| Error::Execution(format!("{e}"))));
        let decode = decode
            .join()
            .unwrap_or_else(|payload| std::panic::resume_unwind(payload));
        (decode, run)
    });
    let (decode_commitment, run) = (decode_commitment?, run?);
    times.execute = t.elapsed().as_secs_f64();
    eprintln!("BLOCK PHASE execute {:.2}s", times.execute);

    let t = Instant::now();
    let mut traces = Traces::from_elf_and_logs(
        &program,
        &run.logs,
        max_rows,
        private_input,
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )?;
    drop(run);
    times.build = t.elapsed().as_secs_f64();
    eprintln!("BLOCK PHASE build {:.2}s", times.build);

    let proof = prove_block_traces(
        elf_bytes,
        &program,
        &mut traces,
        opts,
        Some(decode_commitment),
        residency,
        &mut times,
    )?;
    eprintln!(
        "BLOCK PHASE total {:.2}s (execute {:.2} · build {:.2} · setup {:.2} · prove {:.2})",
        times.total(),
        times.execute,
        times.build,
        times.setup,
        times.prove
    );
    Ok((proof, times))
}

/// The prove step over built traces: the AIRs [`crate::prove_with_options_and_inputs`]
/// builds (with DECODE's commitment supplied when given), the monolithic
/// statement in the transcript, and one `multi_prove` under `residency`.
/// Prints the instance census before proving.
pub fn prove_block_traces(
    elf_bytes: &[u8],
    program: &Elf,
    traces: &mut Traces,
    opts: &ProofOptions,
    decode_commitment: Option<Commitment>,
    residency: ResidencyMode,
    times: &mut BlockTimes,
) -> Result<VmProof, Error> {
    let t = Instant::now();
    let table_counts = traces.table_counts();
    let airs = VmAirs::new(
        program,
        opts,
        false,
        &traces.page_configs,
        &table_counts,
        decode_commitment,
        true,
        None,
        None,
        None,
    );
    let runtime_page_ranges = traces.runtime_page_ranges();
    let num_private_input_pages = traces
        .page_configs
        .iter()
        .filter(|c| c.is_private_input)
        .count();
    let mut transcript = crate::hash_pin::block_transcript(&[]);
    absorb_statement(
        &mut transcript,
        StatementKind::Monolithic,
        elf_bytes,
        &traces.public_output_bytes,
        &table_counts,
        num_private_input_pages,
        &runtime_page_ranges,
        opts.fri_final_poly_log_degree,
    );
    let public_output = traces.public_output_bytes.clone();
    let pairs = airs.air_trace_pairs(traces);
    print_census(
        pairs
            .iter()
            .map(|(air, trace, _)| (air.name(), trace.num_rows(), trace.num_main_columns)),
    );
    times.setup = t.elapsed().as_secs_f64();
    eprintln!(
        "BLOCK PHASE setup {:.2}s · residency {residency:?}",
        times.setup
    );

    let t = Instant::now();
    let proof = crate::hash_pin::BlockProver::multi_prove(
        pairs,
        &mut transcript,
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
        residency,
    )
    .map_err(|e| Error::Prover(format!("{e:?}")))?;
    times.prove = t.elapsed().as_secs_f64();
    eprintln!("BLOCK PHASE prove {:.2}s", times.prove);

    Ok(VmProof {
        proof,
        runtime_page_ranges,
        table_counts,
        public_output,
        num_private_input_pages,
    })
}

/// One `BLOCK CENSUS` line per table type (instances, rows, main cells) and a
/// total, from `(AIR name, rows, main columns)` in AIR order. A type is the
/// name before its `[i]` instance suffix.
fn print_census<'a>(instances: impl Iterator<Item = (&'a str, usize, usize)>) {
    let mut types: Vec<(String, usize, u64, u64)> = Vec::new();
    for (name, rows, cols) in instances {
        let ty = name.split('[').next().unwrap_or(name).to_string();
        let cells = (rows * cols) as u64;
        match types.iter_mut().find(|(t, ..)| *t == ty) {
            Some(entry) => {
                entry.1 += 1;
                entry.2 += rows as u64;
                entry.3 += cells;
            }
            None => types.push((ty, 1, rows as u64, cells)),
        }
    }
    let (mut instances_total, mut cells_total) = (0usize, 0u64);
    for (ty, count, rows, cells) in &types {
        eprintln!("BLOCK CENSUS {ty}: {count} instance(s), {rows} rows, {cells} main cells");
        instances_total += count;
        cells_total += cells;
    }
    eprintln!(
        "BLOCK CENSUS total: {instances_total} sub-proofs, {} table types, {cells_total} main cells",
        types.len()
    );
}
