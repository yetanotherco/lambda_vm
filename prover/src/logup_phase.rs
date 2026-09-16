//! Approach 1's LogUp pass: walk the execution again and build each table's
//! auxiliary columns against the challenge the Commit phase produced.
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

use stark::config::Commitment;
use stark::lookup::BusPublicInputs;
use stark::proof::options::ProofOptions;
use stark::prover::IsStarkProver;

use crate::Error;
use crate::challenge_phase::Challenge;
use crate::pass::{self, ChunkAirs, Resident, Visitor};
use crate::streaming::{GROUP_ORDER, NUM_FIXED_AIRS};
use crate::tables::MaxRowsConfig;
use crate::tables::trace_builder::TableKind;
use crate::tables::types::*;
use executor::elf::Elf;
use math::field::element::FieldElement;
use stark::trace::TraceTable;

/// A table's auxiliary commitment.
///
/// `bus` is what the aux build reported for the LogUp bus; the proof carries it
/// per table, so it travels with the root rather than being recomputed.
pub struct AuxRoot {
    pub root: Commitment,
    pub bus: Option<BusPublicInputs<GoldilocksExtension>>,
}

/// What the LogUp pass produced.
pub struct LogUp {
    /// One entry per table, in `VmAirs::air_trace_pairs` order. `None` for a
    /// table with no auxiliary trace.
    pub aux: Vec<Option<AuxRoot>>,
    /// The tables the pass could not retire, rebuilt by this walk.
    pub resident: Resident,
}

struct CommitAux<'a> {
    airs: &'a ChunkAirs,
    challenges: &'a [FieldElement<GoldilocksExtension>],
    roots: Vec<(TableKind, usize, Option<AuxRoot>)>,
}

impl Visitor for CommitAux<'_> {
    fn table(
        &mut self,
        kind: TableKind,
        chunk: usize,
        trace: &mut TraceTable<GoldilocksField, GoldilocksExtension>,
    ) -> Result<(), Error> {
        let aux = commit_aux(self.airs.get(kind).as_ref(), trace, self.challenges)
            .map_err(|e| Error::Prover(format!("logup phase: {kind:?} chunk {chunk}: {e}")))?;
        self.roots.push((kind, chunk, aux));
        Ok(())
    }
}

fn commit_aux(
    air: &dyn stark::traits::AIR<
        Field = GoldilocksField,
        FieldExtension = GoldilocksExtension,
        PublicInputs = (),
    >,
    trace: &mut TraceTable<GoldilocksField, GoldilocksExtension>,
    challenges: &[FieldElement<GoldilocksExtension>],
) -> Result<Option<AuxRoot>, String> {
    if !air.has_aux_trace() {
        return Ok(None);
    }
    type P = stark::prover::Prover<GoldilocksField, GoldilocksExtension, ()>;
    let (root, bus) = <P as IsStarkProver<_, _, _>>::commit_aux_root(air, trace, challenges)
        .ok_or_else(|| "no auxiliary commitment".to_string())?;
    Ok(Some(AuxRoot { root, bus }))
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
    let mut visitor = CommitAux {
        airs: &airs,
        challenges: &challenge.challenges,
        roots: Vec::new(),
    };
    let mut resident = pass::run(elf, private_input, max_rows, &mut visitor)?;

    let aux = assemble(visitor.roots, &mut resident, elf, proof_options, challenge)?;
    Ok(LogUp { aux, resident })
}

/// Every auxiliary root in `VmAirs::air_trace_pairs` order.
///
/// The chunked tables come back in the order the walk produced them; the tables
/// that stay have their aux built here, as the Challenge phase built their
/// mains.
fn assemble(
    chunks: Vec<(TableKind, usize, Option<AuxRoot>)>,
    resident: &mut Resident,
    elf: &Elf,
    proof_options: &ProofOptions,
    challenge: &Challenge,
) -> Result<Vec<Option<AuxRoot>>, Error> {
    use std::collections::HashMap;

    let counts = crate::challenge_phase::count_chunks_by_kind(
        chunks.iter().map(|(kind, chunk, _)| (*kind, *chunk)),
    );
    let airs = crate::VmAirs::new(
        elf,
        proof_options,
        false,
        &resident.page_configs,
        &counts,
        None,
        true,
        None,
        None,
        None,
    );

    let mut by_slot: HashMap<(TableKind, usize), Option<AuxRoot>> = HashMap::new();
    for (kind, chunk, aux) in chunks {
        if by_slot.insert((kind, chunk), aux).is_some() {
            return Err(Error::Prover(format!(
                "logup phase: {kind:?} chunk {chunk} built twice"
            )));
        }
    }

    let ch = &challenge.challenges;
    let mut out = Vec::new();
    let fixed: [(
        &crate::VmAir,
        &mut TraceTable<GoldilocksField, GoldilocksExtension>,
    ); 10] = [
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
    debug_assert_eq!(fixed.len(), NUM_FIXED_AIRS);
    for (air, trace) in fixed {
        out.push(commit_aux(air.as_ref(), trace, ch).map_err(Error::Prover)?);
    }
    if airs.include_halt {
        out.push(commit_aux(airs.halt.as_ref(), &mut resident.halt, ch).map_err(Error::Prover)?);
    }

    let mut page_airs = airs.pages.iter().zip(resident.pages.iter_mut());
    for group in GROUP_ORDER {
        let Some(kind) = group else {
            for (air, trace) in page_airs.by_ref() {
                out.push(commit_aux(air.as_ref(), trace, ch).map_err(Error::Prover)?);
            }
            continue;
        };
        for chunk in 0..crate::challenge_phase::count_for(&counts, kind) {
            out.push(by_slot.remove(&(kind, chunk)).ok_or_else(|| {
                Error::Prover(format!("logup phase: no aux for {kind:?} chunk {chunk}"))
            })?);
        }
    }
    if let Some(((kind, chunk), _)) = by_slot.into_iter().next() {
        return Err(Error::Prover(format!(
            "logup phase: {kind:?} chunk {chunk} has an aux root but no AIR"
        )));
    }

    Ok(out)
}
