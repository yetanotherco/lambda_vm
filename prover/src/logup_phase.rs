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
#[cfg(test)]
pub(crate) fn fork_for(
    challenge: &Challenge,
    idx: usize,
    num_airs: usize,
) -> DefaultTranscript<GoldilocksExtension> {
    fork(&challenge.transcript, idx, num_airs)
}

/// Rebuild only the tables a pass cannot retire, for a caller that wants one of
/// them without proving the run.
#[cfg(test)]
pub(crate) fn resident_tables(
    elf: &Elf,
    private_input: &[u8],
    max_rows: &MaxRowsConfig,
) -> Result<pass::Resident, Error> {
    struct Skip;
    impl Visitor for Skip {
        fn table(
            &mut self,
            _kind: TableKind,
            _chunk: usize,
            _trace: TraceTable<GoldilocksField, GoldilocksExtension>,
        ) -> Result<(), Error> {
            Ok(())
        }
    }
    pass::run(elf, private_input, max_rows, &mut Skip)
}

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

/// A table's half of a batched proof: everything it contributes that is not a
/// FRI, which is now its group's business.
///
/// This is what the per-table `StarkProof` keeps once the layers, the final
/// polynomial, the queries and the nonce move to the group — the 57.9% of the
/// proof that stops being paid once per table.
pub struct TablePublic {
    pub trace_rows: usize,
    pub main_root: stark::config::Commitment,
    pub precomputed_root: Option<stark::config::Commitment>,
    pub aux_root: Option<stark::config::Commitment>,
    pub composition_poly_root: stark::config::Commitment,
    pub trace_ood: stark::table::Table<GoldilocksExtension>,
    pub trace_ood_next: stark::table::Table<GoldilocksExtension>,
    pub parts_ood: Vec<FieldElement<GoldilocksExtension>>,
    pub bus_public_inputs: Option<stark::lookup::BusPublicInputs<GoldilocksExtension>>,
}

/// One FRI per height group, instead of one per table.
pub struct Batched {
    /// Per table, in AIR order: what it contributes besides its codeword.
    pub tables: Vec<TablePublic>,
    /// Per group, in ascending domain size: the domain and its FRI instance —
    /// layers, final polynomial, the shared query indices and their
    /// decommitments.
    pub groups: Vec<(usize, stark::prover::GroupFri<GoldilocksExtension>)>,
    /// How many tables each group folded, so the collapse is visible.
    pub members: Vec<usize>,
    /// Which group each table belongs to, by AIR index — the group whose query
    /// indices its openings answer.
    pub group_of: Vec<usize>,
    /// The AIR indices in the order they were folded. The coefficient of a
    /// table depends on every table folded before it, so the verifier has to
    /// replay this order and the proof carries it.
    pub fold_order: Vec<usize>,
    pub resident: Resident,
}

type Deep = stark::prover::TableDeep<GoldilocksExtension>;
/// What the fold walk carries between batches.
///
/// The seed and the accumulators are advanced sequentially — the coefficient of
/// a table depends on every table folded before it — while the codewords that
/// feed them are computed in parallel. So the expensive half stays concurrent
/// and only the folding is serialised.
struct FoldState {
    seed: DefaultTranscript<GoldilocksExtension>,
    /// One accumulator per distinct domain, and the rows behind it.
    acc: std::collections::BTreeMap<usize, (Vec<FieldElement<GoldilocksExtension>>, usize, usize)>,
    /// The AIR indices in the order they were folded, which the proof carries
    /// so a verifier can replay the same sequence of coefficients.
    order: Vec<usize>,
    tables: Vec<(usize, TablePublic)>,
}

type Deeps = std::sync::Mutex<FoldState>;

struct BuildDeep<'a> {
    batch: pass::Batched<'a, Item>,
}

impl<'a> BuildDeep<'a> {
    fn new(airs: &'a ChunkAirs, challenge: &'a Challenge, done: &'a Deeps) -> Self {
        Self {
            batch: pass::Batched::new(move |items| deep_batch(airs, challenge, done, items)),
        }
    }
}

fn deep_batch(
    airs: &ChunkAirs,
    challenge: &Challenge,
    done: &Deeps,
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
                    "batched phase: {kind:?} chunk {chunk} is not in the layout"
                ))
            })?;
            let mut transcript = fork(&challenge.transcript, idx, n);
            let deep = deep_of(
                airs.get(kind).as_ref(),
                &mut trace,
                &challenge.challenges,
                &mut transcript,
                challenge.roots.get(idx).cloned(),
            )
            .map_err(|e| Error::Prover(format!("batched phase: {kind:?} chunk {chunk}: {e}")))?;
            Ok((idx, deep))
        })
        .collect();
    // Folded in a fixed order within the batch, so the sequence is a function
    // of the walk and not of which thread finished first.
    let mut built = built?;
    built.sort_by_key(|(idx, _)| *idx);
    let mut state = done.lock().expect("fold state");
    for (idx, deep) in built {
        fold_one(&mut state, idx, deep);
    }
    Ok(())
}

/// Absorb a table, draw its coefficient, add it to its group, drop it.
fn fold_one(state: &mut FoldState, idx: usize, deep: Deep) {
    type P = stark::prover::Prover<GoldilocksField, GoldilocksExtension, ()>;
    let coefficient = <P as IsStarkProver<_, _, _>>::fold_coefficient(&mut state.seed, &deep);
    let entry = state
        .acc
        .entry(deep.lde_size)
        .or_insert_with(|| (Vec::new(), deep.trace_rows, 0));
    <P as IsStarkProver<_, _, _>>::accumulate(&mut entry.0, &coefficient, &deep.deep);
    entry.2 += 1;
    state.order.push(idx);
    state.tables.push((
        idx,
        TablePublic {
            trace_rows: deep.trace_rows,
            main_root: deep.main_roots.main,
            precomputed_root: deep.main_roots.precomputed,
            aux_root: deep.aux_root,
            composition_poly_root: deep.composition_poly_root,
            trace_ood: deep.trace_ood,
            trace_ood_next: deep.trace_ood_next,
            parts_ood: deep.parts_ood,
            bus_public_inputs: deep.bus_public_inputs,
        },
    ));
    // `deep` dies here, which is the whole point.
}

fn deep_of(
    air: &dyn stark::traits::AIR<
        Field = GoldilocksField,
        FieldExtension = GoldilocksExtension,
        PublicInputs = (),
    >,
    trace: &mut TraceTable<GoldilocksField, GoldilocksExtension>,
    challenges: &[FieldElement<GoldilocksExtension>],
    transcript: &mut DefaultTranscript<GoldilocksExtension>,
    known_main: Option<stark::prover::MainRoots>,
) -> Result<Deep, String> {
    type P = stark::prover::Prover<GoldilocksField, GoldilocksExtension, ()>;
    <P as IsStarkProver<_, _, _>>::deep_for_table(
        air,
        &(),
        trace,
        challenges,
        transcript,
        known_main,
    )
    .map_err(|e| format!("{e:?}"))
}

impl Visitor for BuildDeep<'_> {
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

/// Walk the execution and fold every table into one FRI per height group.
///
/// The codewords are held, not the LDEs they came from — one extension element
/// per row instead of every column — which is what lets the fold wait until
/// every table is done without walking the execution a third time.
///
/// Grouping is by exact domain and can only be: the fold squares the coset
/// offset each layer, so a short codeword never lines up with a tall fold.
pub fn run_batched(
    elf: &Elf,
    private_input: &[u8],
    max_rows: &MaxRowsConfig,
    proof_options: &ProofOptions,
    challenge: &Challenge,
) -> Result<Batched, Error> {
    type P = stark::prover::Prover<GoldilocksField, GoldilocksExtension, ()>;

    let chunk_airs = ChunkAirs::new(proof_options);
    let done = std::sync::Mutex::new(FoldState {
        seed: challenge.transcript.clone(),
        acc: std::collections::BTreeMap::new(),
        order: Vec::new(),
        tables: Vec::new(),
    });
    let mut visitor = BuildDeep::new(&chunk_airs, challenge, &done);
    let mut resident = pass::run(elf, private_input, max_rows, &mut visitor)?;
    drop(visitor);

    // The tables the walk could not retire, folded after it in AIR order.
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
    let n = order.len();
    {
        let mut state = done.lock().expect("fold state");
        let build = |state: &mut FoldState,
                     idx: usize,
                     air: &crate::VmAir,
                     trace: &mut TraceTable<GoldilocksField, GoldilocksExtension>|
         -> Result<(), Error> {
            let mut transcript = fork(&challenge.transcript, idx, n);
            let deep = deep_of(
                air.as_ref(),
                trace,
                &challenge.challenges,
                &mut transcript,
                challenge.roots.get(idx).cloned(),
            )
            .map_err(|e| Error::Prover(format!("batched phase: table {idx}: {e}")))?;
            fold_one(state, idx, deep);
            Ok(())
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
            build(&mut state, idx, air, trace)?;
        }
        if airs.include_halt {
            build(&mut state, NUM_FIXED_AIRS, &airs.halt, &mut resident.halt)?;
        }
        for (i, (air, trace)) in airs.pages.iter().zip(resident.pages.iter_mut()).enumerate() {
            let idx = order.page_index(i).ok_or_else(|| {
                Error::Prover(format!("batched phase: page {i} is not in the layout"))
            })?;
            build(&mut state, idx, air, trace)?;
        }
    }

    let FoldState {
        mut seed,
        acc,
        order: fold_order,
        mut tables,
    } = done.into_inner().expect("fold state");
    if tables.len() != n {
        return Err(Error::Prover(format!(
            "batched phase: {} tables for a layout of {n}",
            tables.len()
        )));
    }

    // One FRI per accumulator, and the mapping from table to group.
    let mut group_of = vec![usize::MAX; n];
    let sizes: Vec<usize> = acc.keys().copied().collect();
    for (&idx, _) in tables.iter().map(|(i, t)| (i, t)) {
        let rows = tables
            .iter()
            .find(|(i, _)| *i == idx)
            .map(|(_, t)| t.trace_rows)
            .expect("table just listed");
        let lde = rows * proof_options.blowup_factor as usize;
        group_of[idx] = sizes.iter().position(|s| *s == lde).ok_or_else(|| {
            Error::Prover(format!(
                "batched phase: table {idx} has no group of size {lde}"
            ))
        })?;
    }

    let any = chunk_airs.get(TableKind::Cpu).as_ref();
    let (mut groups, mut members) = (Vec::new(), Vec::new());
    for (lde_size, (codeword, trace_rows, count)) in acc {
        let fri = <P as IsStarkProver<_, _, _>>::batch_fri(any, codeword, trace_rows, &mut seed)
            .ok_or_else(|| Error::Prover(format!("batched phase: no FRI for size {lde_size}")))?;
        groups.push((lde_size, fri));
        members.push(count);
    }

    tables.sort_by_key(|(idx, _)| *idx);
    Ok(Batched {
        tables: tables.into_iter().map(|(_, t)| t).collect(),
        fold_order,
        groups,
        members,
        group_of,
        resident,
    })
}

/// What the Open pass produced: every table's rows at its group's indices.
pub struct Opened {
    /// One entry per table, in AIR order.
    pub openings:
        Vec<stark::proof::stark::DeepPolynomialOpenings<GoldilocksField, GoldilocksExtension>>,
    pub resident: Resident,
}

type Open = stark::proof::stark::DeepPolynomialOpenings<GoldilocksField, GoldilocksExtension>;
type Opens = std::sync::Mutex<Vec<(usize, Open)>>;

struct OpenTables<'a> {
    batch: pass::Batched<'a, Item>,
}

impl<'a> OpenTables<'a> {
    fn new(
        airs: &'a ChunkAirs,
        challenge: &'a Challenge,
        batched: &'a Batched,
        done: &'a Opens,
    ) -> Self {
        Self {
            batch: pass::Batched::new(move |items| {
                open_batch(airs, challenge, batched, done, items)
            }),
        }
    }
}

fn iotas_of(batched: &Batched, idx: usize) -> Result<&[usize], Error> {
    let g = *batched
        .group_of
        .get(idx)
        .ok_or_else(|| Error::Prover(format!("open pass: table {idx} has no group")))?;
    let (_, fri) = batched
        .groups
        .get(g)
        .ok_or_else(|| Error::Prover(format!("open pass: table {idx} points at group {g}")))?;
    Ok(&fri.iotas)
}

fn open_batch(
    airs: &ChunkAirs,
    challenge: &Challenge,
    batched: &Batched,
    done: &Opens,
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
                    "open pass: {kind:?} chunk {chunk} is not in the layout"
                ))
            })?;
            let mut transcript = fork(&challenge.transcript, idx, n);
            let opening = open_of(
                airs.get(kind).as_ref(),
                &mut trace,
                &challenge.challenges,
                &mut transcript,
                iotas_of(batched, idx)?,
            )
            .map_err(|e| Error::Prover(format!("open pass: {kind:?} chunk {chunk}: {e}")))?;
            Ok((idx, opening))
        })
        .collect();
    done.lock().expect("openings").extend(built?);
    Ok(())
}

fn open_of(
    air: &dyn stark::traits::AIR<
        Field = GoldilocksField,
        FieldExtension = GoldilocksExtension,
        PublicInputs = (),
    >,
    trace: &mut TraceTable<GoldilocksField, GoldilocksExtension>,
    challenges: &[FieldElement<GoldilocksExtension>],
    transcript: &mut DefaultTranscript<GoldilocksExtension>,
    iotas: &[usize],
) -> Result<Open, String> {
    type P = stark::prover::Prover<GoldilocksField, GoldilocksExtension, ()>;
    <P as IsStarkProver<_, _, _>>::open_for_table(air, &(), trace, challenges, transcript, iotas)
        .map_err(|e| format!("{e:?}"))
}

impl Visitor for OpenTables<'_> {
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

/// Approach 1's fifth pass: walk the execution once more and open every table
/// at the indices its group settled on.
///
/// The last walk, and the one the spec puts last for a reason — the indices do
/// not exist until the batched FRI is over, so nothing here could have been
/// folded into an earlier pass.
pub fn run_open(
    elf: &Elf,
    private_input: &[u8],
    max_rows: &MaxRowsConfig,
    proof_options: &ProofOptions,
    challenge: &Challenge,
    batched: &Batched,
) -> Result<Opened, Error> {
    let chunk_airs = ChunkAirs::new(proof_options);
    let done = std::sync::Mutex::new(Vec::new());
    let mut visitor = OpenTables::new(&chunk_airs, challenge, batched, &done);
    let mut resident = pass::run(elf, private_input, max_rows, &mut visitor)?;
    drop(visitor);
    let mut opens = done.into_inner().expect("openings");

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
    let n = order.len();
    let build = |idx: usize,
                 air: &crate::VmAir,
                 trace: &mut TraceTable<GoldilocksField, GoldilocksExtension>|
     -> Result<(usize, Open), Error> {
        let mut transcript = fork(&challenge.transcript, idx, n);
        let opening = open_of(
            air.as_ref(),
            trace,
            &challenge.challenges,
            &mut transcript,
            iotas_of(batched, idx)?,
        )
        .map_err(|e| Error::Prover(format!("open pass: table {idx}: {e}")))?;
        Ok((idx, opening))
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
        opens.push(build(idx, air, trace)?);
    }
    if airs.include_halt {
        opens.push(build(NUM_FIXED_AIRS, &airs.halt, &mut resident.halt)?);
    }
    for (i, (air, trace)) in airs.pages.iter().zip(resident.pages.iter_mut()).enumerate() {
        let idx = order
            .page_index(i)
            .ok_or_else(|| Error::Prover(format!("open pass: page {i} is not in the layout")))?;
        opens.push(build(idx, air, trace)?);
    }

    opens.sort_by_key(|(idx, _)| *idx);
    if opens.len() != n {
        return Err(Error::Prover(format!(
            "open pass: {} tables for a layout of {n}",
            opens.len()
        )));
    }
    Ok(Opened {
        openings: opens.into_iter().map(|(_, o)| o).collect(),
        resident,
    })
}

/// A batched proof: what the five passes produce, assembled.
///
/// Additive, not a replacement. `StarkProof` and `multi_verify` are untouched
/// and still produce byte-identical proofs; this is a second format alongside
/// them, for the path that folds one FRI per domain instead of one per table.
///
/// The split is the whole point. A table keeps what only it can answer for —
/// its roots, its out-of-domain values, its openings — and a group carries the
/// FRI those tables share. That is the 57.9% of a per-table proof that stops
/// being paid 227 times.
pub struct BatchedProof {
    /// Per table, in AIR order.
    pub tables: Vec<TablePublic>,
    /// Per table, in AIR order: its rows at its group's indices.
    pub openings: Vec<Open>,
    /// Which group each table belongs to.
    pub group_of: Vec<usize>,
    /// Per group, in ascending domain: the FRI they share.
    pub groups: Vec<(usize, stark::prover::GroupFri<GoldilocksExtension>)>,
    /// The statement, which the verifier binds before absorbing any root.
    pub public_output: Vec<u8>,
    pub page_configs: Vec<crate::tables::page::PageConfig>,
}

/// Assemble what the batched and Open passes produced.
///
/// Takes them rather than running them, so the two walks stay independently
/// testable and a caller can measure either on its own.
pub fn assemble_batched_proof(batched: Batched, opened: Opened) -> Result<BatchedProof, Error> {
    if batched.tables.len() != opened.openings.len() {
        return Err(Error::Prover(format!(
            "assemble: {} tables against {} openings",
            batched.tables.len(),
            opened.openings.len()
        )));
    }
    Ok(BatchedProof {
        tables: batched.tables,
        openings: opened.openings,
        group_of: batched.group_of,
        groups: batched.groups,
        public_output: opened.resident.public_output,
        page_configs: opened.resident.page_configs,
    })
}
