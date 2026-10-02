//! The finish's hand-off to phase A's committers (I-SCHED lever 1): each plain
//! table the finish generates goes, by value, to a [`FinishSink`] as soon as it
//! exists, so its Round-1 commit is made on the card while the rest of the
//! finish runs, not in phase B's main commit.
//!
//! The contract, for a finish that hands tables off:
//! - [`hand_or_keep`] at each table it generates: the sink takes the table and
//!   the slot gets the streamed chunks' placeholder (no rows, no columns), or
//!   the sink declines (its committers have stopped) and the table stays in
//!   its slot, to be committed in phase B as before.
//! - Only the kinds in [`FinishedKind`] go: the plain tables held in a `Vec`.
//!   The preprocessed tables (BITWISE, DECODE, KECCAK_RC, REGISTER, PAGE) and
//!   the singletons (HALT, BLAKE3) stay with phase B.
//!
//! The committer side: [`precommit_finished`] makes the table's commit under
//! the AIR [`crate::VmAirs::new`] gives it ([`air_for`]), and
//! [`insert_finished`] puts the tables back into their placeholders once the
//! finish has returned. The prove places each precommit on its AIR by name and
//! absorbs every root in AIR order, as for the streamed chunks: the transcript
//! and the proof bytes do not depend on when, or on which thread, a table was
//! committed.

use stark::prover::IsStarkProver;
use stark::residency_mode::ResidencyMode;
use stark::trace::TraceTable;

use crate::Error;
use crate::ProofOptions;
use crate::tables::trace_builder::{StreamTable, StreamedChunk, Traces};
use crate::tables::types::{GoldilocksExtension, GoldilocksField};
use crate::test_utils::{
    VmAir, create_branch_air, create_bytewise_air, create_commit_air, create_cpu_air,
    create_cpu32_air, create_dvrm_air, create_ecdas_air, create_ecsm_air, create_eq_air,
    create_hint_air, create_keccak_air, create_keccak_rnd_air, create_load_air, create_lt_air,
    create_memw_air, create_memw_aligned_air, create_memw_register_air, create_mul_air,
    create_shift_air, create_store_air,
};

/// A main trace as the builders make it.
pub type Trace = TraceTable<GoldilocksField, GoldilocksExtension>;

/// A table's Round-1 commit made ahead of the prove
/// ([`IsStarkProver::precommit_main`]).
pub type Precommit =
    stark::prover::PrecommittedMain<GoldilocksField, crate::hash_pin::BlockStarkHash>;

/// The plain tables a finish may hand off: every table [`Traces`] holds in a
/// `Vec` whose AIR has no preprocessed columns.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FinishedKind {
    Cpu,
    MemwRegister,
    MemwAligned,
    Memw,
    Load,
    Lt,
    Shift,
    Store,
    Mul,
    Dvrm,
    Branch,
    Eq,
    Bytewise,
    Cpu32,
    Commit,
    Keccak,
    KeccakRnd,
    Ecsm,
    Ecdas,
    Hint,
}

impl FinishedKind {
    pub const ALL: [FinishedKind; 20] = [
        FinishedKind::Cpu,
        FinishedKind::MemwRegister,
        FinishedKind::MemwAligned,
        FinishedKind::Memw,
        FinishedKind::Load,
        FinishedKind::Lt,
        FinishedKind::Shift,
        FinishedKind::Store,
        FinishedKind::Mul,
        FinishedKind::Dvrm,
        FinishedKind::Branch,
        FinishedKind::Eq,
        FinishedKind::Bytewise,
        FinishedKind::Cpu32,
        FinishedKind::Commit,
        FinishedKind::Keccak,
        FinishedKind::KeccakRnd,
        FinishedKind::Ecsm,
        FinishedKind::Ecdas,
        FinishedKind::Hint,
    ];

    /// The AIR's name for instance `index`, as [`crate::VmAirs::new`] names it.
    pub fn name(self, index: usize) -> String {
        let ty = match self {
            FinishedKind::Cpu => "CPU",
            FinishedKind::MemwRegister => "MEMW_R",
            FinishedKind::MemwAligned => "MEMW_A",
            FinishedKind::Memw => "MEMW",
            FinishedKind::Load => "LOAD",
            FinishedKind::Lt => "LT",
            FinishedKind::Shift => "SHIFT",
            FinishedKind::Store => "STORE",
            FinishedKind::Mul => "MUL",
            FinishedKind::Dvrm => "DVRM",
            FinishedKind::Branch => "BRANCH",
            FinishedKind::Eq => "EQ",
            FinishedKind::Bytewise => "BYTEWISE",
            FinishedKind::Cpu32 => "CPU32",
            FinishedKind::Commit => "COMMIT",
            FinishedKind::Keccak => "KECCAK",
            FinishedKind::KeccakRnd => "KECCAK_RND",
            FinishedKind::Ecsm => "ECSM",
            FinishedKind::Ecdas => "ECDAS",
            FinishedKind::Hint => "HINT",
        };
        format!("{ty}[{index}]")
    }

    /// This kind's slots in `traces`.
    fn slots(self, traces: &mut Traces) -> &mut Vec<Trace> {
        match self {
            FinishedKind::Cpu => &mut traces.cpus,
            FinishedKind::MemwRegister => &mut traces.memw_registers,
            FinishedKind::MemwAligned => &mut traces.memw_aligneds,
            FinishedKind::Memw => &mut traces.memws,
            FinishedKind::Load => &mut traces.loads,
            FinishedKind::Lt => &mut traces.lts,
            FinishedKind::Shift => &mut traces.shifts,
            FinishedKind::Store => &mut traces.stores,
            FinishedKind::Mul => &mut traces.muls,
            FinishedKind::Dvrm => &mut traces.dvrms,
            FinishedKind::Branch => &mut traces.branches,
            FinishedKind::Eq => &mut traces.eqs,
            FinishedKind::Bytewise => &mut traces.bytewises,
            FinishedKind::Cpu32 => &mut traces.cpu32s,
            FinishedKind::Commit => &mut traces.commits,
            FinishedKind::Keccak => &mut traces.keccaks,
            FinishedKind::KeccakRnd => &mut traces.keccak_rnds,
            FinishedKind::Ecsm => &mut traces.ecsms,
            FinishedKind::Ecdas => &mut traces.ecdases,
            FinishedKind::Hint => &mut traces.hints,
        }
    }
}

impl From<StreamTable> for FinishedKind {
    fn from(table: StreamTable) -> Self {
        match table {
            StreamTable::Cpu => FinishedKind::Cpu,
            StreamTable::MemwRegister => FinishedKind::MemwRegister,
            StreamTable::MemwAligned => FinishedKind::MemwAligned,
            StreamTable::Memw => FinishedKind::Memw,
            StreamTable::Load => FinishedKind::Load,
            StreamTable::Lt => FinishedKind::Lt,
            StreamTable::Shift => FinishedKind::Shift,
            StreamTable::Store => FinishedKind::Store,
        }
    }
}

/// The AIR [`crate::VmAirs::new`] builds for instance `index` of `kind`: the
/// same constructor and the same name, so a commit made under it is the one
/// the prove would make.
pub fn air_for(kind: FinishedKind, index: usize, opts: &ProofOptions) -> VmAir {
    let name = kind.name(index);
    match kind {
        FinishedKind::Cpu => Box::new(create_cpu_air(opts).with_name(&name)),
        FinishedKind::MemwRegister => Box::new(create_memw_register_air(opts).with_name(&name)),
        FinishedKind::MemwAligned => Box::new(create_memw_aligned_air(opts).with_name(&name)),
        FinishedKind::Memw => Box::new(create_memw_air(opts).with_name(&name)),
        FinishedKind::Load => Box::new(create_load_air(opts).with_name(&name)),
        FinishedKind::Lt => Box::new(create_lt_air(opts).with_name(&name)),
        FinishedKind::Shift => Box::new(create_shift_air(opts).with_name(&name)),
        FinishedKind::Store => Box::new(create_store_air(opts).with_name(&name)),
        FinishedKind::Mul => Box::new(create_mul_air(opts).with_name(&name)),
        FinishedKind::Dvrm => Box::new(create_dvrm_air(opts).with_name(&name)),
        FinishedKind::Branch => Box::new(create_branch_air(opts).with_name(&name)),
        FinishedKind::Eq => Box::new(create_eq_air(opts).with_name(&name)),
        FinishedKind::Bytewise => Box::new(create_bytewise_air(opts).with_name(&name)),
        FinishedKind::Cpu32 => Box::new(create_cpu32_air(opts).with_name(&name)),
        FinishedKind::Commit => Box::new(create_commit_air(opts).with_name(&name)),
        FinishedKind::Keccak => Box::new(create_keccak_air(opts).with_name(&name)),
        FinishedKind::KeccakRnd => Box::new(create_keccak_rnd_air(opts).with_name(&name)),
        FinishedKind::Ecsm => Box::new(create_ecsm_air(opts).with_name(&name)),
        FinishedKind::Ecdas => Box::new(create_ecdas_air(opts).with_name(&name)),
        FinishedKind::Hint => Box::new(create_hint_air(opts).with_name(&name)),
    }
}

/// One table the finish generated: instance `index` of `kind` (its slot in
/// [`Traces`]) and its trace.
pub struct FinishedTable {
    pub kind: FinishedKind,
    pub index: usize,
    pub trace: Trace,
}

/// A streamed chunk is a table of its kind: the committers take both alike.
impl From<StreamedChunk> for FinishedTable {
    fn from(chunk: StreamedChunk) -> Self {
        Self {
            kind: chunk.table.into(),
            index: chunk.index,
            trace: chunk.trace,
        }
    }
}

/// Where a finish hands its tables. `hand` must not block: the finish calls it
/// from its generators (rayon workers), as each table exists.
pub trait FinishSink: Sync {
    /// Takes the table (`None`), or gives it back when it cannot commit it: the
    /// finish then keeps the table in its slot.
    fn hand(&self, table: FinishedTable) -> Option<FinishedTable>;
}

/// The finish's one call per generated table: the table to leave in its slot.
/// With a sink that takes it, the streamed chunks' placeholder (no rows, no
/// columns); without a sink, or when the sink declines, the table itself.
pub fn hand_or_keep(
    sink: Option<&dyn FinishSink>,
    kind: FinishedKind,
    index: usize,
    trace: Trace,
) -> Trace {
    let Some(sink) = sink else {
        return trace;
    };
    match sink.hand(FinishedTable { kind, index, trace }) {
        None => crate::tables::trace_builder::streamed_placeholder(),
        Some(declined) => declined.trace,
    }
}

/// Every plain table `traces` holds (not a placeholder) handed to `sink` at
/// once, in [`FinishedKind::ALL`] order: what a finish that hands nothing off
/// leaves for phase B. The tables handed.
pub fn hand_built(traces: &mut Traces, sink: &dyn FinishSink) -> usize {
    let mut handed = 0;
    for kind in FinishedKind::ALL {
        let slots = kind.slots(traces);
        for (index, slot) in slots.iter_mut().enumerate() {
            if slot.main_table.width == 0 {
                continue;
            }
            let trace =
                std::mem::replace(slot, crate::tables::trace_builder::streamed_placeholder());
            *slot = hand_or_keep(Some(sink), kind, index, trace);
            handed += usize::from(slot.main_table.width == 0);
        }
    }
    handed
}

/// A sink that sends each table into a committer queue of `T`s (a channel the
/// committers drain with the streamed chunks). It declines once the receiver
/// is gone, so a finish whose committers stopped keeps its tables.
pub struct ChannelSink<T> {
    tx: std::sync::mpsc::Sender<T>,
    unwrap: fn(T) -> FinishedTable,
}

impl<T: From<FinishedTable> + Send> ChannelSink<T> {
    /// `unwrap` recovers the table from a `T` the channel refused.
    pub fn new(tx: std::sync::mpsc::Sender<T>, unwrap: fn(T) -> FinishedTable) -> Self {
        Self { tx, unwrap }
    }
}

impl<T: From<FinishedTable> + Send> FinishSink for ChannelSink<T> {
    fn hand(&self, table: FinishedTable) -> Option<FinishedTable> {
        self.tx
            .send(T::from(table))
            .err()
            .map(|refused| (self.unwrap)(refused.0))
    }
}

/// The committer's half: the table's Round-1 commit under [`air_for`], exactly
/// the commit [`IsStarkProver::multi_prove_precommitted`] would make of it in
/// phase B, and the AIR name the prove places it by.
pub fn precommit_finished(
    table: &FinishedTable,
    opts: &ProofOptions,
    residency: ResidencyMode,
) -> Result<(String, Precommit), Error> {
    let air = air_for(table.kind, table.index, opts);
    let pre = crate::hash_pin::BlockProver::precommit_main(
        air.as_ref(),
        &table.trace,
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
        residency,
    )
    .map_err(|e| Error::Prover(format!("{}: {e:?}", air.name())))?;
    Ok((air.name().to_string(), pre))
}

/// Puts committed tables back into the placeholders their finish left. Refuses
/// a table with no slot or for a slot that is not a placeholder, as
/// [`Traces::insert_streamed`] does for the streamed chunks.
pub fn insert_finished(traces: &mut Traces, tables: Vec<FinishedTable>) -> Result<(), Error> {
    for table in tables {
        let slots = table.kind.slots(traces);
        let len = slots.len();
        let slot = slots.get_mut(table.index).ok_or_else(|| {
            Error::Prover(format!(
                "finished {} has no slot ({len} tables)",
                table.kind.name(table.index)
            ))
        })?;
        if slot.main_table.width != 0 {
            return Err(Error::Prover(format!(
                "finished {} lands on a slot that is not a placeholder",
                table.kind.name(table.index)
            )));
        }
        *slot = table.trace;
    }
    Ok(())
}

/// A sink for tests: keeps every table it is handed, or declines them all.
#[cfg(test)]
pub(crate) struct CollectingSink {
    pub(crate) tables: std::sync::Mutex<Vec<FinishedTable>>,
    pub(crate) decline: bool,
}

#[cfg(test)]
impl CollectingSink {
    pub(crate) fn new(decline: bool) -> Self {
        Self {
            tables: std::sync::Mutex::new(Vec::new()),
            decline,
        }
    }
}

#[cfg(test)]
impl FinishSink for CollectingSink {
    fn hand(&self, table: FinishedTable) -> Option<FinishedTable> {
        if self.decline {
            return Some(table);
        }
        self.tables.lock().unwrap().push(table);
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use executor::elf::Elf;
    use executor::vm::execution::Executor;

    use crate::tables::MaxRowsConfig;
    use crate::test_utils::asm_elf_bytes;

    /// The block's format over blowup 4 with no grinding (the block tests'
    /// `bytes_options`), so two proves of one build compare byte for byte.
    fn bytes_options() -> ProofOptions {
        let base = stark::proof::options::GoldilocksCubicProofOptions::with_params(4, 128, 0)
            .expect("options");
        crate::zf_format::ZfFormat::DEFAULT.options(base)
    }

    fn build(name: &str, max_rows: &MaxRowsConfig) -> (Vec<u8>, Elf, Traces) {
        let elf_bytes = asm_elf_bytes(name);
        let program = Elf::load(&elf_bytes).expect("load the ELF");
        let run = Executor::new(&program, vec![])
            .expect("executor")
            .run()
            .expect("run");
        let traces = Traces::from_elf_and_logs(
            &program,
            &run.logs,
            max_rows,
            &[],
            #[cfg(feature = "disk-spill")]
            stark::storage_mode::StorageMode::Ram,
        )
        .expect("build the traces");
        (elf_bytes, program, traces)
    }

    /// The same table, row for row.
    fn same(a: &Trace, b: &Trace) -> bool {
        let rows = |t: &Trace| -> Vec<Vec<_>> {
            (0..t.main_table.height)
                .map(|r| t.main_table.get_row(r).to_vec())
                .collect()
        };
        (a.main_table.width, a.main_table.height) == (b.main_table.width, b.main_table.height)
            && rows(a) == rows(b)
    }

    /// Every instance of every kind present in `traces`, in `ALL` order.
    fn instances(traces: &mut Traces) -> Vec<(FinishedKind, usize)> {
        FinishedKind::ALL
            .iter()
            .flat_map(|&kind| (0..kind.slots(traces).len()).map(move |i| (kind, i)))
            .collect()
    }

    /// A finish double: hands every plain table to `sink` from rayon workers,
    /// as a finish's generators would, and keeps what the sink declines.
    fn hand_all(traces: &mut Traces, sink: &dyn FinishSink) {
        use rayon::prelude::*;
        for kind in FinishedKind::ALL {
            let slots = kind.slots(traces);
            let taken: Vec<Trace> = std::mem::take(slots);
            *kind.slots(traces) = taken
                .into_par_iter()
                .enumerate()
                .map(|(i, trace)| hand_or_keep(Some(sink), kind, i, trace))
                .collect();
        }
    }

    /// `air_for` is the AIR `VmAirs::new` builds at the same name: the same
    /// layout, no preprocessed columns, the same constraints, for every kind,
    /// at more than one index; and every kind's name exists among those AIRs.
    #[test]
    fn air_for_is_the_air_the_prove_builds() {
        let opts = bytes_options();
        let counts = crate::TableCounts {
            cpu: 2,
            lt: 2,
            memw: 2,
            memw_aligned: 2,
            load: 2,
            mul: 2,
            dvrm: 2,
            shift: 2,
            branch: 2,
            memw_register: 2,
            eq: 2,
            bytewise: 2,
            store: 2,
            cpu32: 2,
            keccak: 2,
            keccak_rnd: 2,
            ecsm: 2,
            ecdas: 2,
            hint: 2,
            commit: 2,
            blake3: 0,
        };
        let program = Elf::load(&asm_elf_bytes("add")).expect("load the ELF");
        let airs = crate::VmAirs::new(
            &program,
            &opts,
            false,
            &[],
            &counts,
            None,
            true,
            None,
            None,
            None,
        );
        let refs = airs.air_refs();
        for kind in FinishedKind::ALL {
            for index in 0..2 {
                let ours = air_for(kind, index, &opts);
                let theirs = refs
                    .iter()
                    .find(|a| a.name() == ours.name())
                    .unwrap_or_else(|| panic!("VmAirs::new has no {}", ours.name()));
                assert_eq!(
                    ours.trace_layout(),
                    theirs.trace_layout(),
                    "{}",
                    ours.name()
                );
                assert!(
                    !ours.is_preprocessed() && !theirs.is_preprocessed(),
                    "{}",
                    ours.name()
                );
                assert_eq!(
                    ours.num_transition_constraints(),
                    theirs.num_transition_constraints()
                );
                assert_eq!(ours.step_size(), theirs.step_size(), "{}", ours.name());
                assert_eq!(
                    ours.context().transition_offsets,
                    theirs.context().transition_offsets,
                    "{}",
                    ours.name()
                );
            }
        }
    }

    /// The streamed tables map onto their kinds by name.
    #[test]
    fn a_streamed_table_is_its_kind() {
        for table in StreamTable::ALL {
            let kind = FinishedKind::from(table);
            assert_eq!(
                kind.name(3),
                format!("{}[3]", kind.name(3).split('[').next().unwrap())
            );
            assert!(FinishedKind::ALL.contains(&kind));
        }
        assert_eq!(
            FinishedKind::from(StreamTable::MemwRegister).name(0),
            "MEMW_R[0]"
        );
    }

    /// The finish double hands every plain table off from rayon workers and
    /// leaves placeholders; putting the tables back gives the build it started
    /// from. A declining sink leaves the build untouched.
    #[test]
    fn handed_tables_go_back_into_their_placeholders() {
        let (_, _, mut traces) = build("test_keccak_multi", &MaxRowsConfig::small());
        let before = traces.clone();
        let present = instances(&mut traces);
        assert!(present.len() > 10, "a build with many plain tables");

        let declining = CollectingSink::new(true);
        hand_all(&mut traces, &declining);
        assert!(declining.tables.lock().unwrap().is_empty());
        for &(kind, i) in &present {
            assert!(
                same(
                    &kind.slots(&mut traces)[i],
                    &kind.slots(&mut before.clone())[i]
                ),
                "a declined {} stays in its slot",
                kind.name(i)
            );
        }

        let sink = CollectingSink::new(false);
        hand_all(&mut traces, &sink);
        let handed = std::mem::take(&mut *sink.tables.lock().unwrap());
        assert_eq!(handed.len(), present.len(), "every plain table was handed");
        for &(kind, i) in &present {
            assert_eq!(
                kind.slots(&mut traces)[i].main_table.width,
                0,
                "{} placeholder",
                kind.name(i)
            );
        }
        insert_finished(&mut traces, handed).expect("back into the placeholders");
        for &(kind, i) in &present {
            assert!(
                same(
                    &kind.slots(&mut traces)[i],
                    &kind.slots(&mut before.clone())[i]
                ),
                "{} is the build's again",
                kind.name(i)
            );
        }

        // `hand_built` hands every table still in a slot, once.
        let sink = CollectingSink::new(false);
        assert_eq!(hand_built(&mut traces, &sink), present.len());
        assert_eq!(
            hand_built(&mut traces, &sink),
            0,
            "only placeholders are left"
        );
        let handed = std::mem::take(&mut *sink.tables.lock().unwrap());
        insert_finished(&mut traces, handed).expect("back into the placeholders");
        for &(kind, i) in &present {
            assert!(same(
                &kind.slots(&mut traces)[i],
                &kind.slots(&mut before.clone())[i]
            ));
        }
    }

    /// The same table, column for column, packed or not.
    fn same_columns(what: &str, a: &Trace, b: &Trace) {
        assert_eq!(
            (a.main_table.width, a.main_table.height),
            (b.main_table.width, b.main_table.height),
            "{what}: shape"
        );
        assert!(
            a.columns_main() == b.columns_main(),
            "{what}: columns differ"
        );
    }

    /// ★ Phase A's committers also commit the tables the finish built
    /// (`LAMBDA_VM_BLOCK_FINISH_COMMIT=after`): the stream builds the same
    /// traces, precommits every streamed instance it did before and every
    /// plain table the finish built, leaves the ledger empty, and a card gate
    /// that admits one commit at a time changes none of it — with the
    /// committers generating, or generators ahead of them.
    #[test]
    fn the_stream_commits_the_finish_tables_in_phase_a() {
        let opts = crate::lfm::proof::block_base_options();
        let max_rows = MaxRowsConfig {
            keccak_rnd: 48,
            ..MaxRowsConfig::small()
        };
        for name in ["all_instructions_64", "test_keccak_multi"] {
            let program = Elf::load(&asm_elf_bytes(name)).expect("load the ELF");
            let (mut today, mut names) =
                crate::block::stream_for_test(&program, &opts, &max_rows, 3, 0, None)
                    .expect("the stream as it is");
            let mut every: Vec<String> = instances(&mut today)
                .into_iter()
                .map(|(kind, i)| kind.name(i))
                .collect();
            every.sort();
            names.sort();
            assert!(names.len() < every.len(), "{name}: the finish built tables");
            for (committers, generators) in [(3, 0), (2, 3)] {
                let (mut after, mut after_names) = crate::block::stream_finish_after_for_test(
                    &program,
                    &opts,
                    &max_rows,
                    committers,
                    generators,
                    Some(1),
                )
                .expect("the finish's tables committed in phase A");
                after_names.sort();
                assert_eq!(after_names, every, "{name}: every plain table precommitted");
                for (kind, i) in instances(&mut today) {
                    same_columns(
                        &format!("{name} {}", kind.name(i)),
                        &kind.slots(&mut today)[i],
                        &kind.slots(&mut after)[i],
                    );
                }
                same_columns("BITWISE", &today.bitwise, &after.bitwise);
                same_columns("DECODE", &today.decode, &after.decode);
                same_columns("REGISTER", &today.register, &after.register);
                same_columns("HALT", &today.halt, &after.halt);
                assert_eq!(today.pages.len(), after.pages.len());
                for (i, (a, b)) in today.pages.iter().zip(&after.pages).enumerate() {
                    same_columns(&format!("PAGE {i}"), a, b);
                }
            }
        }
    }

    /// A table for a slot that holds a table, or for a slot that does not exist,
    /// is refused.
    #[test]
    fn a_finished_table_only_lands_on_a_placeholder() {
        let (_, _, mut traces) = build("add", &MaxRowsConfig::default());
        let cpu = traces.cpus[0].clone();
        let err = insert_finished(
            &mut traces,
            vec![FinishedTable {
                kind: FinishedKind::Cpu,
                index: 0,
                trace: cpu.clone(),
            }],
        )
        .expect_err("CPU[0] is not a placeholder");
        assert!(format!("{err:?}").contains("not a placeholder"), "{err:?}");
        let n = traces.cpus.len();
        let err = insert_finished(
            &mut traces,
            vec![FinishedTable {
                kind: FinishedKind::Cpu,
                index: n,
                trace: cpu,
            }],
        )
        .expect_err("no such slot");
        assert!(format!("{err:?}").contains("has no slot"), "{err:?}");
    }

    /// A channel sink sends while its committers listen and gives the table
    /// back once they are gone.
    #[test]
    fn a_channel_sink_declines_once_its_receiver_is_gone() {
        enum Job {
            Finished(FinishedTable),
        }
        impl From<FinishedTable> for Job {
            fn from(t: FinishedTable) -> Self {
                Job::Finished(t)
            }
        }
        let (tx, rx) = std::sync::mpsc::channel::<Job>();
        let sink = ChannelSink::new(tx, |job| match job {
            Job::Finished(t) => t,
        });
        let (_, _, traces) = build("add", &MaxRowsConfig::default());
        let trace = traces.cpus[0].clone();
        let left = hand_or_keep(Some(&sink), FinishedKind::Cpu, 0, trace.clone());
        assert_eq!(left.main_table.width, 0, "taken: a placeholder is left");
        let Job::Finished(got) = rx.recv().expect("sent");
        assert!(got.kind == FinishedKind::Cpu && got.index == 0 && same(&got.trace, &trace));
        drop(rx);
        let kept = hand_or_keep(Some(&sink), FinishedKind::Cpu, 0, trace.clone());
        assert!(same(&kept, &trace), "declined: the finish keeps its table");
        let none = hand_or_keep(None, FinishedKind::Cpu, 0, trace.clone());
        assert!(same(&none, &trace), "no sink: nothing moves");
    }

    /// The proof is the same bytes whether every plain table was committed in
    /// phase B or handed off and precommitted on a committer thread (the
    /// lever's whole claim), and it verifies; a precommit made under the wrong
    /// table is refused before any proof exists.
    #[test]
    #[ignore = "proves a VM program twice at blowup 4; GPU box gate (cuda)"]
    fn finished_precommits_prove_the_same_bytes() {
        use crate::block::{BlockTimes, prove_block_traces};
        let (elf_bytes, program, traces) = build("test_keccak_multi", &MaxRowsConfig::small());
        let opts = bytes_options();
        let residency = ResidencyMode::RecomputeLdeDevice;
        let prove = |traces: &mut Traces, precommits: Vec<(String, Precommit)>| {
            let decode = crate::tables::decode::commitment_from_elf_device_or_host(&program, &opts)
                .expect("DECODE commitment");
            prove_block_traces(
                &elf_bytes,
                &program,
                traces,
                &opts,
                Some(decode),
                residency,
                precommits,
                &mut BlockTimes::default(),
                &mut |_| {},
            )
        };
        let plain = prove(&mut traces.clone(), Vec::new()).expect("prove, phase-B commits");

        let mut handed_off = traces.clone();
        let sink = CollectingSink::new(false);
        hand_all(&mut handed_off, &sink);
        let handed = std::mem::take(&mut *sink.tables.lock().unwrap());
        let precommits: Vec<(String, Precommit)> = std::thread::scope(|s| {
            let jobs: Vec<_> = handed
                .iter()
                .map(|t| s.spawn(|| precommit_finished(t, &opts, residency).expect("precommit")))
                .collect();
            jobs.into_iter().map(|j| j.join().unwrap()).collect()
        });
        let n = precommits.len();
        insert_finished(&mut handed_off, handed).expect("back into the placeholders");
        let precommitted = prove(&mut handed_off, precommits).expect("prove, phase-A commits");
        let bytes = |p: &crate::VmProof| {
            rkyv::to_bytes::<rkyv::rancor::Error>(p)
                .expect("serialize")
                .to_vec()
        };
        assert!(
            bytes(&plain) == bytes(&precommitted),
            "{n} finished precommits moved the proof bytes"
        );
        assert!(matches!(
            crate::block::verify_block(&precommitted, &elf_bytes, &opts),
            Ok(true)
        ));

        // CPU[1]'s commit placed on CPU[0]: the prove must not turn it into a
        // proof the verifier accepts. Refused = an error, a prover panic (a
        // device root check), or a proof the block verifier rejects.
        let mut swapped = traces.clone();
        assert!(swapped.cpus.len() >= 2, "two CPU chunks at the small caps");
        let wrong = precommit_finished(
            &FinishedTable {
                kind: FinishedKind::Cpu,
                index: 0,
                trace: swapped.cpus[1].clone(),
            },
            &opts,
            residency,
        )
        .expect("precommit");
        let accepted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            prove(&mut swapped, vec![wrong]).is_ok_and(|p| {
                matches!(crate::block::verify_block(&p, &elf_bytes, &opts), Ok(true))
            })
        }))
        .unwrap_or(false);
        assert!(!accepted, "a precommit of another trace was accepted");
    }
}
