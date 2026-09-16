//! Approach 1's Commit phase: walk the execution, committing and retiring each
//! table as it fills.
//!
//! The spec has the prover go through execution "and once the memory pressure
//! becomes too large, batch commit to all full tables in memory; then these
//! tables are dropped". This is that pass. What it produces is a commitment per
//! closed chunk and, at the end, whatever the walk could not close — which the
//! Challenge phase pads and commits, as the spec's next step.

use stark::config::Commitment;
use stark::proof::options::ProofOptions;
use stark::prover::IsStarkProver;

use crate::Error;
use crate::tables::MaxRowsConfig;
use crate::tables::trace_builder::{TableKind, Traces, WalkLeftover, build_initial_image};
use crate::tables::{register, types::*};
use executor::elf::Elf;

/// What the Commit phase produced.
pub struct CommitPhase {
    /// One entry per chunk closed during the walk, in the order they closed.
    pub closed: Vec<(TableKind, usize, Commitment)>,
    /// Everything the walk still held when the execution ended.
    pub leftover: WalkLeftover,
}

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
        &artifacts,
        elf,
        private_input.to_vec(),
        &image,
        &register_init,
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
