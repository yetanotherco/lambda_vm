//! Canonical witness handout and host placement for the in-process cluster.

use super::*;
use crypto::hash::blake3::chain::Blake3Std;
use digest::Digest;
use std::cmp::Reverse;
use std::sync::OnceLock;

pub(super) struct HostAirs {
    pub stream: StreamAirs,
    pub run: OnceLock<VmAirs>,
    pub early: OnceLock<Vec<crate::VmAir>>,
}

impl HostAirs {
    pub fn new(opts: &ProofOptions, format: &BlockFormat, options: &BlockOptions) -> Self {
        let form = if options.gpack && options.narrow != multilinear_block::Narrowing::Wide {
            TraceForm::Narrow
        } else {
            TraceForm::Wide
        };
        Self {
            stream: StreamAirs::new(
                opts,
                form,
                options.max_rows.cpu,
                format.keccak_rnd,
                format.memw_aligned,
                format.registers,
            ),
            run: OnceLock::new(),
            early: OnceLock::new(),
        }
    }
}

#[derive(Clone, Copy)]
pub(super) enum Pass<'a, 'p> {
    Local,
    Coordinator {
        workers: multilinear_block::InprocWorkers,
        airs: &'a [HostAirs],
    },
    Worker {
        id: usize,
        plan: &'p Plan,
    },
}

pub(super) type Committed<'a> = BlockCommitted<'a, F, E>;
pub(super) type Owned<'a> = Vec<(usize, Committed<'a>)>;

#[allow(clippy::large_enum_variant)]
pub(super) enum Run<'a> {
    Proof(BlockWhirProof<LocalArgues>),
    Worker(Owned<'a>),
}

#[derive(Debug, PartialEq, Eq)]
struct TableDescriptor {
    key: Key,
    name: String,
    shape: (usize, usize),
    records: [u8; 32],
}

pub(super) struct WitnessTable<'a> {
    key: Key,
    air: AirRef<'a>,
    shape: (usize, usize),
    data: WitnessData,
}

enum WitnessData {
    Trace(TraceTable<F, E>, bool),
    Bitwise(crate::tables::trace_builder::BitwiseWitness),
    Chunk(crate::tables::trace_builder::DeferredChunk, TraceForm),
    Local(LocalCounts, LocalBytesFault),
}

impl<'a> WitnessTable<'a> {
    pub fn trace(
        key: Key,
        air: AirRef<'a>,
        shape: (usize, usize),
        trace: TraceTable<F, E>,
        check: bool,
    ) -> HandoutTable<'a> {
        HandoutTable::Witness(Box::new(Self {
            key,
            air,
            shape,
            data: WitnessData::Trace(trace, check),
        }))
    }

    pub fn bitwise(
        air: AirRef<'a>,
        shape: (usize, usize),
        slot: &crate::tables::trace_builder::BitwiseSlot,
    ) -> Result<HandoutTable<'a>, Error> {
        Ok(HandoutTable::Witness(Box::new(Self {
            key: Key::Air(0),
            air,
            shape,
            data: WitnessData::Bitwise(slot.take()?),
        })))
    }

    pub fn chunk(
        key: Key,
        air: AirRef<'a>,
        shape: (usize, usize),
        chunk: crate::tables::trace_builder::DeferredChunk,
        form: TraceForm,
    ) -> HandoutTable<'a> {
        HandoutTable::Witness(Box::new(Self {
            key,
            air,
            shape,
            data: WitnessData::Chunk(chunk, form),
        }))
    }

    pub fn local(
        index: usize,
        air: AirRef<'a>,
        shape: (usize, usize),
        counts: LocalCounts,
        fault: LocalBytesFault,
    ) -> HandoutTable<'a> {
        HandoutTable::Witness(Box::new(Self {
            key: Key::Local(index),
            air,
            shape,
            data: WitnessData::Local(counts, fault),
        }))
    }

    fn spans_parts(&self) -> bool {
        matches!(&self.data, WitnessData::Chunk(chunk, _) if chunk.spans_parts)
    }

    pub fn bytes(&self) -> usize {
        match &self.data {
            WitnessData::Trace(trace, _) => memlog::rows_bytes(trace),
            WitnessData::Bitwise(bitwise) => bitwise.bytes(),
            WitnessData::Chunk(chunk, _) => chunk.heap_bytes(),
            WitnessData::Local(counts, _) => [&counts.are_bytes, &counts.and, &counts.xor]
                .iter()
                .map(|column| column.capacity() * size_of::<u32>())
                .sum(),
        }
    }

    fn describe(&self) -> TableDescriptor {
        let mut digest = Blake3Std::new();
        match &self.data {
            WitnessData::Local(counts, fault) => {
                digest.update(b"LOCAL_BYTES");
                digest.update([*fault as u8]);
                for column in [&counts.are_bytes, &counts.and, &counts.xor] {
                    put_words(&mut digest, column.iter().copied().map(u64::from));
                }
            }
            WitnessData::Chunk(chunk, _) => {
                digest.update(b"chunk");
                digest.update(chunk.digest());
            }
            WitnessData::Bitwise(bitwise) => {
                digest.update(b"BITWISE");
                put_words(&mut digest, bitwise.counts().iter().copied());
            }
            WitnessData::Trace(trace, _) => {
                if let Some(records) = trace.records_main() {
                    digest.update(b"records");
                    records
                        .fill()
                        .words(|word| digest.update(word.to_le_bytes()));
                } else if let Some(packed) = trace.narrow_main() {
                    digest.update(b"packed");
                    digest.update(packed.widths());
                    digest.update(packed.data());
                } else {
                    digest.update(b"rows");
                    put_words(
                        &mut digest,
                        trace.main_data_row_major().0.iter().map(|v| v.to_raw()),
                    );
                }
            }
        }
        TableDescriptor {
            key: self.key,
            name: self.air.name().to_owned(),
            shape: self.shape,
            records: digest.finalize().into(),
        }
    }

    pub fn materialize(self) -> Result<CommittedTable<'a, F, E>, Error> {
        let (mut trace, check) = match self.data {
            WitnessData::Trace(trace, check) => (trace, check),
            WitnessData::Bitwise(bitwise) => (bitwise.materialize(), true),
            WitnessData::Chunk(chunk, form) => (chunk.materialize(form), true),
            WitnessData::Local(counts, fault) => local_trace(counts, fault)?,
        };
        if (trace.main_table.width, trace.main_table.height) != (self.shape.0, 1 << self.shape.1) {
            return Err(Error::Prover(format!(
                "cluster group table {} materialized with a different shape",
                self.air.name()
            )));
        }
        if trace.records_main().is_some() {
            table_of_records(self.air, &mut trace, self.shape)
        } else if trace.narrow_main().is_some() {
            table_of_narrow(self.air, &mut trace, self.shape)
        } else {
            table_of_checked(self.air, &mut trace, self.shape, true, check)
        }
    }
}

fn put_words(digest: &mut Blake3Std, words: impl IntoIterator<Item = u64>) {
    let mut bytes = [0; 4096];
    let mut size = 0;
    for word in words {
        bytes[size..size + 8].copy_from_slice(&word.to_le_bytes());
        size += 8;
        if size == bytes.len() {
            digest.update(bytes);
            size = 0;
        }
    }
    digest.update(&bytes[..size]);
}

pub(super) fn local_counts(
    form: crate::tables::keccak_rnd::KeccakRndForm,
    air: AirRef<'_>,
    trace: &TraceTable<F, E>,
) -> Result<Option<LocalCounts>, Error> {
    if !form.byte_lookups_local() || !local_bytes::sends_local(air.bus_interactions()) {
        return Ok(None);
    }
    local_bytes::local_counts_from_trace(trace, air.bus_interactions(), form).map_err(Error::Prover)
}

pub(super) struct Plan {
    groups: Vec<Vec<TableDescriptor>>,
    pub owners: Vec<usize>,
    pub statement: Vec<Vec<u8>>,
    pub spanning_groups: usize,
    pub terminal_groups: usize,
}

impl Plan {
    pub fn collect<'a>(
        groups: impl IntoIterator<Item = CommitAssignment<'a>>,
        workers: usize,
        pin_prepared: bool,
    ) -> Result<Self, Error> {
        let mut plan = Self {
            groups: Vec::new(),
            owners: Vec::new(),
            statement: Vec::new(),
            spanning_groups: 0,
            terminal_groups: 0,
        };
        let mut loads = vec![0u128; workers];
        let mut ready = Vec::with_capacity(workers);
        for assignment in groups {
            if assignment.index != plan.groups.len() {
                return Err(Error::Prover(
                    "cluster coordinator handed out groups out of order".into(),
                ));
            }
            let price = stark::phase_b_split::split_price(
                assignment
                    .witness
                    .iter()
                    .map(|table| (table.shape.0 as u64) << table.shape.1)
                    .sum(),
                assignment.witness.len(),
            );
            let pinned = pin_prepared
                && assignment.witness.iter().any(|table| {
                    table.air.name() == "DECODE"
                        || (table.air.is_preprocessed() && table.air.name().starts_with("PAGE"))
                });
            plan.spanning_groups +=
                usize::from(assignment.witness.iter().any(WitnessTable::spans_parts));
            plan.terminal_groups += usize::from(
                assignment
                    .witness
                    .iter()
                    .any(|table| matches!(table.air.name(), "HALT" | "REGISTER")),
            );
            plan.groups.push(describe(&assignment));
            plan.owners.push(0);
            ready.push((assignment.index, price, pinned));
            if ready.len() == workers {
                place_ready(&mut ready, &mut loads, &mut plan.owners);
            }
        }
        place_ready(&mut ready, &mut loads, &mut plan.owners);
        Ok(plan)
    }

    pub fn read<'a>(
        &self,
        worker: usize,
        groups: impl IntoIterator<Item = CommitAssignment<'a>>,
        mut commit: impl FnMut(CommitAssignment<'a>) -> Result<Committed<'a>, Error>,
        #[cfg(test)] fault: Option<Fault>,
    ) -> Result<Owned<'a>, Error> {
        let mut owned = Vec::new();
        let mut seen = 0;
        for group in groups {
            let index = group.index;
            let got = describe(&group);
            #[cfg(test)]
            let got = if worker == 0 {
                Fault::inject(fault, index, got)
            } else {
                got
            };
            if index != seen || self.groups.get(index) != Some(&got) {
                let difference = self.groups.get(index).and_then(|expected| {
                    expected.iter().zip(&got).find_map(|(a, b)| {
                        let field = if (a.key, &a.name) != (b.key, &b.name) {
                            "key"
                        } else if a.shape != b.shape {
                            "shape"
                        } else if a.records != b.records {
                            if a.name == "BITWISE" || a.name == "LOCAL_BYTES" {
                                "lookup counts"
                            } else {
                                "ordered records"
                            }
                        } else {
                            return None;
                        };
                        Some(format!("{} {field}", a.name))
                    })
                });
                let detail = difference
                    .as_deref()
                    .unwrap_or("group order or table count");
                return Err(Error::Prover(format!(
                    "cluster replica {worker} disagrees with coordinator on group {index} ({detail})"
                )));
            }
            if self.owners[index] == worker {
                owned.push((index, commit(group)?));
            }
            seen += 1;
        }
        if seen != self.groups.len() {
            return Err(Error::Prover(format!(
                "cluster replica {worker} handed out {seen} of {} groups",
                self.groups.len()
            )));
        }
        Ok(owned)
    }
}

fn describe(group: &CommitAssignment<'_>) -> Vec<TableDescriptor> {
    debug_assert!(group.tables.is_empty());
    group.witness.iter().map(WitnessTable::describe).collect()
}

// A bounded batch of closed groups is ready at each placement. Batch limits
// follow canonical indices so they do not depend on host or channel timing.
fn place_ready(ready: &mut Vec<(usize, u64, bool)>, loads: &mut [u128], owners: &mut [usize]) {
    ready.sort_unstable_by_key(|&(index, price, pinned)| (!pinned, Reverse(price), index));
    for (index, price, pinned) in ready.drain(..) {
        let owner = if pinned {
            0
        } else {
            (0..loads.len()).min_by_key(|&id| (loads[id], id)).unwrap()
        };
        owners[index] = owner;
        loads[owner] += u128::from(price);
    }
}

#[cfg(test)]
#[derive(Clone, Copy)]
pub(crate) enum Fault {
    Key,
    Height,
    Record,
    LookupCount,
}

#[cfg(test)]
impl Fault {
    fn inject(
        fault: Option<Self>,
        group: usize,
        mut tables: Vec<TableDescriptor>,
    ) -> Vec<TableDescriptor> {
        match fault {
            Some(Self::Key) if group == 0 => tables[0].key = Key::Air(usize::MAX),
            Some(Self::Height) if group == 0 => tables[0].shape.1 += 1,
            Some(Self::Record) if group == 0 => tables[0].records[0] ^= 1,
            Some(Self::LookupCount) => {
                if let Some(bitwise) = tables.iter_mut().find(|table| table.name == "BITWISE") {
                    bitwise.records[0] ^= 1;
                }
            }
            _ => {}
        }
        tables
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use stark::proof::options::BaseFormat;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn replica_divergence_is_refused_before_observers() {
        let elf = crate::test_utils::asm_elf_bytes("test_commit_4");
        let program = Elf::load(&elf).unwrap();
        let opts = ProofOptions::default_test_options();
        let format = crate::tests::block_whir_p1_tests::many_groups(BaseFormat::B3_WHIR);
        let mut options = BlockOptions::production();
        options.max_rows = crate::tables::MaxRowsConfig::small();
        options.max_rows.cpu = 8;
        options.window_log2 = Some(2);
        options.regen = Some(regen::RegenMode::Off);
        options.layout_workers = 0;
        options.memlog = false;
        let workers = multilinear_block::InprocWorkers::parse("inproc:2")
            .unwrap()
            .unwrap();
        let airs = HostAirs::new(&opts, &format, &options);
        let replicas: Vec<_> = (0..workers.count())
            .map(|_| HostAirs::new(&opts, &format, &options))
            .collect();
        let observed = AtomicBool::new(false);
        let deviations = Deviations {
            replica_fault: Some(Fault::Record),
            walkers: Some(2),
            ..Default::default()
        };
        let result = crate::base_only::with_blake3_levels(|| {
            prove_streamed_on_host(
                &program,
                &elf,
                &[],
                4,
                &opts,
                &format,
                &options,
                &deviations,
                Pass::Coordinator {
                    workers,
                    airs: &replicas,
                },
                &airs,
                &|_, _| observed.store(true, Ordering::Relaxed),
                &|_| observed.store(true, Ordering::Relaxed),
                None,
                &mut BlockStamps::default(),
            )
        });
        let error = result.err().expect("a divergent replica must fail");
        assert!(
            error
                .to_string()
                .contains("cluster replica 0 disagrees with coordinator on group 0"),
            "{error}"
        );
        assert!(!observed.load(Ordering::Relaxed));
    }

    #[test]
    fn every_replica_checks_keys_heights_record_order_and_lookup_counts() {
        use crate::tables::bitwise;
        let opts = ProofOptions::default_test_options();
        let air = crate::test_utils::create_lt_air(&opts);
        let make = |records: [u64; 2]| {
            let values: Vec<_> = records.into_iter().map(FieldElement::<F>::from).collect();
            let trace = TraceTable::new_main(values, 1, 1);
            let HandoutTable::Witness(mut table) =
                WitnessTable::trace(Key::Air(0), &air, (1, 1), trace, true)
            else {
                unreachable!()
            };
            let mut tail = table.describe();
            tail.name = "BITWISE".into();
            // The real BITWISE shape and a lookup payload distinct from the
            // ordinary ordered trace above.
            tail.shape = (
                bitwise::cols::NUM_COLUMNS,
                bitwise::NUM_ROWS.ilog2() as usize,
            );
            table.key = Key::Air(1);
            (table, tail)
        };
        let (table, lookup) = make([1, 2]);
        let mut expected = vec![table.describe(), lookup];
        let (reversed, _) = make([2, 1]);
        assert_ne!(expected[0].records, reversed.describe().records);
        for fault in [Fault::Key, Fault::Height, Fault::Record, Fault::LookupCount] {
            let got = Fault::inject(Some(fault), 0, {
                let (_, lookup) = make([1, 2]);
                vec![table.describe(), lookup]
            });
            assert_ne!(expected, got);
        }
        expected.pop();
        let plan = Plan {
            groups: vec![expected],
            owners: vec![1],
            statement: Vec::new(),
            spanning_groups: 0,
            terminal_groups: 0,
        };
        let assignment = |words| CommitAssignment {
            index: 0,
            tables: Vec::new(),
            witness: vec![*make(words).0],
        };
        assert!(
            plan.read(
                0,
                [assignment([1, 2])],
                |_| panic!("a non-owner cannot commit"),
                None
            )
            .unwrap()
            .is_empty()
        );
        assert!(
            plan.read(
                0,
                [assignment([2, 1])],
                |_| panic!("a non-owner cannot commit"),
                None
            )
            .is_err()
        );
    }

    #[test]
    fn ready_groups_use_prices_pins_and_stable_ties() {
        let mut loads = [0; 3];
        let mut owners = vec![usize::MAX; 5];
        place_ready(
            &mut vec![(0, 5, false), (1, 7, false), (2, 3, true)],
            &mut loads,
            &mut owners,
        );
        place_ready(
            &mut vec![(3, 5, false), (4, 5, false)],
            &mut loads,
            &mut owners,
        );
        assert_eq!(owners, [2, 1, 0, 0, 2]);
        assert_eq!(loads, [8, 7, 10]);
    }
}
