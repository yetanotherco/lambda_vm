//! Approach 1's Challenge phase: absorb every root and sample the one challenge
//! the whole execution shares.
//!
//! The spec's step after Commit is to pad and commit the remaining tables and
//! then sample the LogUp challenges. [`crate::commit_phase::run_to_end`] does
//! the padding; this does the sampling, and it is where Approach 1 differs from
//! the continuations in `main`. There, each epoch samples its own challenges, so
//! the tables of different epochs live on different buses and need the
//! local-to-global apparatus to be tied back together. Here every chunk of the
//! run is absorbed into one transcript and answers to one `(z, alpha)`, so there
//! is nothing to tie.
//!
//! The order the roots are absorbed in *is* the protocol: it has to be the AIR
//! order the ordinary prover uses, with a preprocessed table's precomputed root
//! ahead of its own. `challenge_matches_the_ordinary_prover` pins that against a
//! real proof rather than against this file's idea of the order.

use std::collections::HashMap;

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::fiat_shamir::is_transcript::IsTranscript;
use stark::proof::options::ProofOptions;
use stark::prover::MainRoots;

use crate::Error;
use crate::commit_phase::Committed;
use crate::statement::{StatementKind, absorb_statement};
use crate::streaming::{GROUP_ORDER, NUM_FIXED_AIRS};
use crate::tables::trace_builder::{TableKind, runtime_page_ranges};
use crate::tables::types::{GoldilocksExtension, GoldilocksField};
use crate::{TableCounts, VmAirs};
use executor::elf::Elf;
use math::field::element::FieldElement;
use stark::trace::TraceTable;

/// The one challenge the whole execution shares, and the roots it was drawn
/// from.
pub struct Challenge {
    /// `z` and `alpha`, in sampling order.
    pub challenges: Vec<FieldElement<GoldilocksExtension>>,
    /// Every root absorbed, in AIR order.
    pub roots: Vec<MainRoots>,
    /// The AIRs of this proof, built once here: their preprocessed commitments
    /// (DECODE from the ELF, one per ELF data page, ...) are the bulk of this pass.
    pub(crate) airs: crate::VmAirs,
    /// The transcript right after the sampling, which every later pass forks
    /// per table. Kept rather than rebuilt: re-absorbing 227 roots to get back
    /// to this state is both slower and a second place for the order to be
    /// wrong.
    pub transcript: DefaultTranscript<GoldilocksExtension>,
    /// The layout the roots were assembled in, so a later pass can ask where a
    /// chunk sits without recounting.
    pub(crate) order: crate::streaming::AirOrder,
}

/// Sample the shared LogUp challenges from a finished Commit phase.
///
/// `elf_bytes` is the raw program: the statement binds its digest, so the
/// challenge depends on the program proved and not only on the tables it
/// produced.
pub fn run(
    committed: &Committed,
    elf: &Elf,
    elf_bytes: &[u8],
    proof_options: &ProofOptions,
) -> Result<Challenge, Error> {
    let remaining = &committed.remaining;
    let table_counts = count_chunks_by_kind(
        committed
            .chunks
            .iter()
            .map(|(kind, chunk, _)| (*kind, *chunk)),
    );
    let airs = VmAirs::new(
        elf,
        proof_options,
        false,
        &remaining.page_configs,
        &table_counts,
        None,
        true,
        None,
        None,
        None,
    );

    let roots = assemble_roots(committed, &airs, &table_counts)?;

    let mut transcript = DefaultTranscript::<GoldilocksExtension>::new(&[]);
    absorb_statement(
        &mut transcript,
        StatementKind::Monolithic,
        elf_bytes,
        &remaining.public_output,
        &table_counts,
        remaining
            .page_configs
            .iter()
            .filter(|c| c.is_private_input)
            .count(),
        &runtime_page_ranges(&remaining.page_configs),
        proof_options.fri_final_poly_log_degree,
    );
    for root in &roots {
        if let Some(ref precomputed) = root.precomputed {
            transcript.append_bytes(precomputed);
        }
        transcript.append_bytes(&root.main);
    }

    let challenges = (0..stark::lookup::LOGUP_NUM_CHALLENGES)
        .map(|_| transcript.sample_field_element())
        .collect();

    let order = crate::streaming::AirOrder::new(
        table_counts,
        airs.include_halt,
        remaining.page_configs.len(),
    );
    Ok(Challenge {
        challenges,
        roots,
        airs,
        transcript,
        order,
    })
}

/// Every root the transcript absorbs, in `VmAirs::air_trace_pairs` order.
///
/// The tables the walk committed come back keyed by `(kind, chunk)` in the order
/// they closed, which is not the AIR order; the ones it could not commit are
/// still traces and are committed here.
fn assemble_roots(
    committed: &Committed,
    airs: &VmAirs,
    table_counts: &TableCounts,
) -> Result<Vec<MainRoots>, Error> {
    let remaining = &committed.remaining;
    let accumulated = &remaining.accumulated;

    let mut roots = Vec::new();
    let fixed: [(
        &crate::VmAir,
        &TraceTable<GoldilocksField, GoldilocksExtension>,
        &str,
    ); NUM_FIXED_AIRS] = [
        (&airs.bitwise, &remaining.bitwise, "BITWISE"),
        (&airs.decode, &remaining.decode, "DECODE"),
        (&airs.commit, &accumulated.commit, "COMMIT"),
        (&airs.keccak, &accumulated.keccak, "KECCAK"),
        (&airs.keccak_rnd, &accumulated.keccak_rnd, "KECCAK_RND"),
        (&airs.keccak_rc, &accumulated.keccak_rc, "KECCAK_RC"),
        (&airs.ecsm, &accumulated.ecsm, "ECSM"),
        (&airs.ecdas, &accumulated.ecdas, "ECDAS"),
        (&airs.hint, &accumulated.hint, "HINT"),
        (&airs.register, &remaining.register, "REGISTER"),
    ];
    for (air, trace, name) in fixed {
        roots.push(commit_resident(air, trace, name)?);
    }
    if airs.include_halt {
        roots.push(commit_resident(&airs.halt, &remaining.halt, "HALT")?);
    }

    let mut by_slot: HashMap<(TableKind, usize), &MainRoots> = HashMap::new();
    for (kind, chunk, root) in &committed.chunks {
        if by_slot.insert((*kind, *chunk), root).is_some() {
            return Err(Error::Prover(format!(
                "challenge phase: {kind:?} chunk {chunk} committed twice"
            )));
        }
    }

    let mut page_airs = airs.pages.iter().zip(remaining.pages.iter());
    for group in GROUP_ORDER {
        let Some(kind) = group else {
            // PAGE is built from the ELF image rather than from an op list, so
            // it is never retired and is committed here with the rest.
            for (air, trace) in page_airs.by_ref() {
                roots.push(commit_resident(air, trace, "PAGE")?);
            }
            continue;
        };
        for chunk in 0..count_for(table_counts, kind) {
            let root = by_slot.remove(&(kind, chunk)).ok_or_else(|| {
                Error::Prover(format!(
                    "challenge phase: no root for {kind:?} chunk {chunk}"
                ))
            })?;
            roots.push(root.clone());
        }
    }
    if let Some(((kind, chunk), _)) = by_slot.into_iter().next() {
        return Err(Error::Prover(format!(
            "challenge phase: {kind:?} chunk {chunk} has a root but no AIR"
        )));
    }

    Ok(roots)
}

fn commit_resident(
    air: &crate::VmAir,
    trace: &TraceTable<GoldilocksField, GoldilocksExtension>,
    name: &str,
) -> Result<MainRoots, Error> {
    type P = stark::prover::Prover<GoldilocksField, GoldilocksExtension, ()>;
    <P as stark::prover::IsStarkProver<_, _, _>>::commit_table_root(air.as_ref(), trace)
        .ok_or_else(|| Error::Prover(format!("challenge phase: no commitment for {name}")))
}

/// How many chunks a pass produced per kind.
///
/// Taken as `(kind, chunk)` pairs rather than as a phase's own output, because
/// every pass over the execution produces the same layout and each has its own
/// per-chunk payload.
pub(crate) fn count_chunks_by_kind(
    chunks: impl Iterator<Item = (TableKind, usize)>,
) -> TableCounts {
    let mut counts = TableCounts {
        cpu: 0,
        lt: 0,
        memw: 0,
        memw_aligned: 0,
        load: 0,
        mul: 0,
        dvrm: 0,
        shift: 0,
        branch: 0,
        memw_register: 0,
        eq: 0,
        bytewise: 0,
        store: 0,
        cpu32: 0,
    };
    for (kind, chunk) in chunks {
        let slot = slot_for(&mut counts, kind);
        *slot = (*slot).max(chunk + 1);
    }
    counts
}

pub(crate) fn count_for(counts: &TableCounts, kind: TableKind) -> usize {
    match kind {
        TableKind::Cpu => counts.cpu,
        TableKind::Lt => counts.lt,
        TableKind::Memw => counts.memw,
        TableKind::MemwAligned => counts.memw_aligned,
        TableKind::Load => counts.load,
        TableKind::Mul => counts.mul,
        TableKind::Dvrm => counts.dvrm,
        TableKind::Shift => counts.shift,
        TableKind::Branch => counts.branch,
        TableKind::MemwRegister => counts.memw_register,
        TableKind::Eq => counts.eq,
        TableKind::Bytewise => counts.bytewise,
        TableKind::Store => counts.store,
        TableKind::Cpu32 => counts.cpu32,
    }
}

fn slot_for(counts: &mut TableCounts, kind: TableKind) -> &mut usize {
    match kind {
        TableKind::Cpu => &mut counts.cpu,
        TableKind::Lt => &mut counts.lt,
        TableKind::Memw => &mut counts.memw,
        TableKind::MemwAligned => &mut counts.memw_aligned,
        TableKind::Load => &mut counts.load,
        TableKind::Mul => &mut counts.mul,
        TableKind::Dvrm => &mut counts.dvrm,
        TableKind::Shift => &mut counts.shift,
        TableKind::Branch => &mut counts.branch,
        TableKind::MemwRegister => &mut counts.memw_register,
        TableKind::Eq => &mut counts.eq,
        TableKind::Bytewise => &mut counts.bytewise,
        TableKind::Store => &mut counts.store,
        TableKind::Cpu32 => &mut counts.cpu32,
    }
}
