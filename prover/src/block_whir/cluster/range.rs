//! Cycle-range ownership for the block's phase A.

use super::stages::Clock;
use super::*;
use crate::tables::types::VmTable;
use multilinear::whir_hash::WhirHash;
use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, mpsc};

pub(in crate::block_whir) struct Job<'a> {
    pub id: usize,
    pub first: usize,
    pub end: usize,
    pub worker: Option<&'a tcp::Worker<'a>>,
    pub clock: Clock,
    coordinator_base: Option<(Mutex<mpsc::Receiver<usize>>, Instant)>,
    coordinator_failure: Option<&'a std::sync::OnceLock<String>>,
    start_cycle: AtomicUsize,
    end_cycle: AtomicUsize,
    output: Mutex<Vec<u8>>,
}

impl<'a> Job<'a> {
    pub fn new(cuts: &[usize], id: usize, worker: Option<&'a tcp::Worker<'a>>) -> Self {
        Self {
            id,
            first: id.checked_sub(1).map_or(0, |previous| cuts[previous]),
            end: cuts.get(id).copied().unwrap_or(usize::MAX),
            worker,
            clock: Clock::default(),
            coordinator_base: None,
            coordinator_failure: None,
            start_cycle: AtomicUsize::new(usize::MAX),
            end_cycle: AtomicUsize::new(0),
            output: Mutex::new(Vec::new()),
        }
    }

    pub(super) fn with_clock(mut self, clock: Clock) -> Self {
        self.clock = clock;
        self
    }

    pub(super) fn with_stream_base(
        mut self,
        base: mpsc::Receiver<usize>,
        deadline: Instant,
        failure: &'a std::sync::OnceLock<String>,
    ) -> Self {
        self.coordinator_base = Some((Mutex::new(base), deadline));
        self.coordinator_failure = Some(failure);
        self
    }

    pub fn check_cancelled(&self) -> Result<(), Error> {
        if self
            .coordinator_failure
            .is_some_and(|failure| failure.get().is_some())
        {
            return Err(invalid("coordinator range build cancelled"));
        }
        Ok(())
    }

    pub fn commits_streamed(&self) -> bool {
        self.worker.is_some() || self.coordinator_base.is_some()
    }

    pub fn stream_base(&self) -> Result<usize, Error> {
        self.clock.mark("wait-index-base");
        let index = if let Some(worker) = self.worker {
            worker.stream_base()?
        } else {
            let (base, deadline) = self
                .coordinator_base
                .as_ref()
                .expect("streamed commitment owner");
            base.lock()
                .unwrap()
                .recv_timeout(deadline.saturating_duration_since(Instant::now()))
                .map_err(|e| invalid(&format!("coordinator streamed index base: {e}")))?
        };
        self.clock.mark("index-base-received");
        Ok(index)
    }

    pub fn terminal(&self) -> bool {
        self.end == usize::MAX
    }

    pub fn note_window(&self, first: usize, end: usize) {
        if self.end_cycle.load(Ordering::Relaxed) == 0 {
            self.start_cycle.store(first, Ordering::Relaxed);
        }
        self.end_cycle.store(end, Ordering::Relaxed);
    }

    pub fn note_output(&self, bytes: &[u8]) {
        self.output.lock().unwrap().extend_from_slice(bytes);
    }

    pub fn output(&self) -> Vec<u8> {
        self.output.lock().unwrap().clone()
    }

    pub fn actual_start(&self) -> usize {
        let cycle = self.start_cycle.load(Ordering::Relaxed);
        if cycle == usize::MAX { 0 } else { cycle }
    }

    pub fn actual_end(&self) -> usize {
        self.end_cycle.load(Ordering::Relaxed)
    }
}

pub(super) fn parse_cuts(value: &str) -> Result<Vec<usize>, Error> {
    if value.is_empty() {
        return Err(invalid("empty window list"));
    }
    value
        .split(',')
        .map(|cut| cut.parse().map_err(|_| invalid("invalid window index")))
        .collect()
}

// A cut list names at most `MAX_WORKERS` ranges, each of which builds one
// ECREC table or none: the block statement takes that many.
const _: () = assert!(wire::MAX_WORKERS <= BLOCK_MAX_ECREC);

pub(super) fn validate_cuts(cuts: &[usize], window: usize, cpu_rows: usize) -> Result<(), Error> {
    if cuts.len() >= wire::MAX_WORKERS
        || window < 4
        || !(4..=window.saturating_mul(2)).contains(&cpu_rows)
        || !cpu_rows.is_power_of_two()
        || cuts.first() == Some(&0)
        || cuts.windows(2).any(|pair| pair[0] >= pair[1])
        || cuts
            .last()
            .is_some_and(|&end| end > crate::tables::cpu::MAX_CPU_ROWS as usize / window)
    {
        return Err(invalid("invalid window ranges or CPU chunk size"));
    }
    Ok(())
}

#[cfg(test)]
pub(in crate::block_whir) fn tamper_boundary(
    logs: &mut Vec<executor::vm::logs::Log>,
    history: &mut Option<executor::vm::history::WindowHistory>,
) {
    // Cycles 4..8 of test_commit_4 start with sb t0, 1(sp) and end with
    // addi t0, zero, 0xDD. This worker owns only that window.
    let fault = std::env::var("LAMBDA_VM_CLUSTER_TEST_FAULT");
    let fault = match fault.as_deref() {
        Ok(fault @ ("boundary-register" | "boundary-memory" | "boundary-drop-cycle")) => fault,
        _ => return,
    };
    let history = history.as_mut().expect("boundary history");
    assert_eq!(history.first, 4);
    match fault {
        "boundary-register" => {
            history.registers.as_mut().unwrap().regs[5].0 ^= 1;
            if !history.lean {
                logs[0].src2_val ^= 1;
            }
        }
        "boundary-memory" => history.word_accesses[0].word ^= 1,
        _ if history.lean => history.cycles -= 1,
        _ => {
            logs.pop().unwrap();
            history.extras.pop().unwrap();
        }
    }
}

/// Test only: what the one-process range reference does wrong to the ECREC
/// tables of a run cut into two ranges that each hold a call, for the
/// verifier to refuse.
#[cfg(all(test, feature = "test-security"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EcrecTamper {
    /// The second range's first call row made a padding row (all zero): a
    /// call the CPU made with no ECREC row.
    DropRow,
    /// The first range's first call row copied into the second range's
    /// first padding row: one call with a row in both tables.
    RowInBoth,
    /// The second range's first call row keyed one CPU step later.
    WrongKey,
}

#[cfg(all(test, feature = "test-security"))]
thread_local! {
    /// Test only: on this thread, the range reference tampers its ECREC
    /// tables so ([`EcrecTamper`]) between the build and the commit.
    pub(crate) static ECREC_TAMPER: std::cell::Cell<Option<EcrecTamper>> =
        const { std::cell::Cell::new(None) };
}

/// Under the word memory format the tables are ECREC_W's, whose columns are
/// ECREC's and then `BAIL`, so each tamper edits the same columns.
#[cfg(all(test, feature = "test-security"))]
fn tamper_ecrec(
    built: &mut [Built<'_>],
    tamper: EcrecTamper,
    memory: crate::tables::fused_memory::MemoryFormat,
) {
    use crate::tables::ecrec::cols;
    let form = if memory == crate::tables::fused_memory::MemoryFormat::Words {
        crate::tables::ecrec_w::NAME
    } else {
        "ECREC"
    };
    let mut tables: Vec<_> = built
        .iter_mut()
        .filter_map(|range| {
            range
                .groups
                .iter_mut()
                .flat_map(|group| group.witness.iter_mut())
                .find(|table| family(table.name()).starts_with("ECREC"))
        })
        .collect();
    assert_eq!(tables.len(), 2, "an ECREC table in each of two ranges");
    assert!(
        tables.iter().all(|table| family(table.name()) == form),
        "the memory format's ECREC tables: {form}"
    );
    let (first, width) = tables[0].rows_for_test();
    let (mut second, _) = tables[1].rows_for_test();
    let calls = |rows: &[u64]| {
        rows.chunks(width)
            .take_while(|row| row[cols::MU] != 0)
            .count()
    };
    let n = calls(&second);
    assert!(calls(&first) >= 1, "a call in the first range");
    assert!(
        n >= 1 && (n + 1) * width <= second.len(),
        "a call and a padding row in the second range"
    );
    match tamper {
        EcrecTamper::DropRow => second[..width].fill(0),
        EcrecTamper::RowInBoth => {
            second[n * width..(n + 1) * width].copy_from_slice(&first[..width]);
        }
        EcrecTamper::WrongKey => {
            second[cols::TIMESTAMP_0] += crate::tables::cpu::TIMESTAMP_STRIDE;
        }
    }
    tables[1].put_rows_for_test(second);
}

fn invalid(reason: &str) -> Error {
    Error::Prover(format!("cluster ranges: {reason}"))
}

pub(super) fn global(name: &str) -> bool {
    matches!(
        name,
        "BITWISE" | "DECODE" | "KECCAK_RC" | "REGISTER" | "HALT"
    ) || name.starts_with("PAGE")
}

fn family(name: &str) -> &str {
    name.split_once('[').map_or(name, |(family, _)| family)
}

pub(in crate::block_whir) struct Built<'a> {
    pub laid: Laid,
    pub groups: Vec<CommitAssignment<'a>>,
    pub committed: Owned<'a>,
}

#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[cfg_attr(test, derive(Clone))]
pub(super) struct Metadata {
    pub start: usize,
    pub end: usize,
    pub tables: Vec<(String, (usize, usize))>,
    pub groups: Vec<Vec<u32>>,
    pub streamed: usize,
    pub finish: usize,
}

#[derive(Default, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[cfg_attr(test, derive(Clone))]
pub(super) struct Counts {
    pub bitwise: Vec<u64>,
    pub decode: Vec<(usize, u64)>,
    pub keccak: usize,
}

const BITWISE_LEN: usize = crate::tables::bitwise::BitwiseHistogram::LEN;
const BITWISE_BITMAP: usize = BITWISE_LEN.div_ceil(8);
const BITWISE_BYTES: usize = BITWISE_LEN * size_of::<u64>();

/// A fixed bitmap in dense BITWISE order, then one unsigned LEB128 count for
/// each set bit in ascending order. Empty ranges use the original frame.
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub(super) struct CompactCounts {
    bitwise: Vec<u8>,
    decode: Vec<(usize, u64)>,
    keccak: usize,
}

impl Counts {
    pub(super) fn compact(&self) -> Option<CompactCounts> {
        if self.bitwise.len() != BITWISE_LEN {
            return None;
        }
        let mut bitwise = vec![0; BITWISE_BITMAP];
        for (byte, chunk) in self.bitwise.chunks(8).enumerate() {
            let mut mask = 0;
            for (bit, &count) in chunk.iter().enumerate() {
                if count == 0 {
                    continue;
                }
                mask |= 1 << bit;
                let mut value = count;
                while value >= 128 {
                    bitwise.push(value as u8 | 128);
                    value >>= 7;
                }
                bitwise.push(value as u8);
                if bitwise.len() >= BITWISE_BYTES {
                    return None;
                }
            }
            bitwise[byte] = mask;
        }
        Some(CompactCounts {
            bitwise,
            decode: self.decode.clone(),
            keccak: self.keccak,
        })
    }
}

impl CompactCounts {
    pub(super) fn expand(self) -> Result<Counts, Error> {
        let bad = || invalid("invalid compact range BITWISE counts");
        if !(BITWISE_BITMAP..BITWISE_BYTES).contains(&self.bitwise.len()) {
            return Err(bad());
        }
        let (bitmap, values) = self.bitwise.split_at(BITWISE_BITMAP);
        let mut values = values.iter().copied();
        let mut bitwise = vec![0; BITWISE_LEN];
        for (byte, &mask) in bitmap.iter().enumerate() {
            let mut mask = mask;
            while mask != 0 {
                let index = byte * 8 + mask.trailing_zeros() as usize;
                mask &= mask - 1;
                let count = bitwise.get_mut(index).ok_or_else(bad)?;
                for shift in (0..=63).step_by(7) {
                    let value = values.next().ok_or_else(bad)?;
                    if shift == 63 && value > 1 {
                        return Err(bad());
                    }
                    *count |= u64::from(value & 127) << shift;
                    if value < 128 {
                        if value == 0 {
                            return Err(bad());
                        }
                        break;
                    }
                }
            }
        }
        if values.next().is_some() {
            return Err(bad());
        }
        Ok(Counts {
            bitwise,
            decode: self.decode,
            keccak: self.keccak,
        })
    }
}

fn multiplicity(trace: &TraceTable<F, E>, row: usize, col: usize) -> u64 {
    trace.narrow_main().map_or_else(
        || trace.get_main(row, col).to_raw(),
        |cols| cols.get(row, col),
    )
}

impl Built<'_> {
    fn first_global_group(&self) -> usize {
        self.groups
            .iter()
            .position(|group| group.witness.iter().any(WitnessTable::is_global))
            .unwrap_or(self.groups.len())
    }

    pub(super) fn describe(&mut self, job: &Job<'_>) -> Result<(Metadata, Counts), Error> {
        let start = job.actual_start();
        let end = job.actual_end();
        if end == 0 {
            self.groups.clear();
            return Ok((
                Metadata {
                    start,
                    end,
                    tables: Vec::new(),
                    groups: Vec::new(),
                    streamed: 0,
                    finish: 0,
                },
                Counts::default(),
            ));
        }
        let mut counts = Counts::default();
        let mut stream_tables = vec![false; self.laid.shapes.len()];
        for &(_, _, table) in &self.laid.streamed_air {
            stream_tables[table] = true;
        }
        let streamed = self
            .laid
            .groups
            .iter()
            .take_while(|group| group.iter().any(|&table| stream_tables[table as usize]))
            .count();
        let finish = self.first_global_group();
        if !job.terminal() {
            for group in self.groups.drain(finish..) {
                for table in group.witness {
                    match (table.air.name(), table.data) {
                        ("BITWISE", WitnessData::Bitwise(witness)) => {
                            counts.bitwise = witness.counts().to_vec()
                        }
                        ("DECODE", WitnessData::Trace(trace, _)) => {
                            counts.decode = (0..trace.num_rows())
                                .filter_map(|row| {
                                    let count =
                                        multiplicity(&trace, row, crate::tables::decode::cols::MU);
                                    (count != 0).then_some((row, count))
                                })
                                .collect();
                        }
                        ("KECCAK_RC", WitnessData::Trace(trace, _)) => {
                            counts.keccak =
                                multiplicity(&trace, 0, crate::tables::keccak_rc::cols::MU)
                                    as usize;
                        }
                        ("HALT" | "REGISTER", _) => {}
                        _ => return Err(invalid("unexpected global table from a prefix")),
                    }
                }
            }
        }
        let mut indices = vec![None; self.laid.shapes.len()];
        let tables = self
            .laid
            .table_names
            .iter()
            .zip(&self.laid.shapes)
            .enumerate()
            .filter_map(|(i, (name, shape))| {
                if job.terminal() || !global(name) {
                    Some((i, (name.clone(), *shape)))
                } else {
                    None
                }
            })
            .enumerate()
            .map(|(i, (old, table))| {
                indices[old] = Some(i as u32);
                table
            })
            .collect();
        let groups = self.laid.groups[..self.groups.len()]
            .iter()
            .map(|group| {
                group
                    .iter()
                    .map(|&table| {
                        indices[table as usize]
                            .ok_or_else(|| invalid("mixed range and global group"))
                    })
                    .collect()
            })
            .collect::<Result<_, _>>()?;
        Ok((
            Metadata {
                start,
                end,
                tables,
                groups,
                streamed,
                finish,
            },
            counts,
        ))
    }

    pub(super) fn add_counts(&mut self, counts: &Counts) -> Result<(), Error> {
        use crate::tables::{decode, keccak_rc};
        let first = self.first_global_group();
        let mut found = [false; 3];
        for table in self.groups[first..]
            .iter_mut()
            .flat_map(|group| &mut group.witness)
        {
            match (table.air.name(), &mut table.data) {
                ("BITWISE", WitnessData::Bitwise(witness)) => {
                    witness.add_range_counts(&counts.bitwise)?;
                    found[0] = true;
                }
                ("DECODE", WitnessData::Trace(trace, _)) => {
                    let rows = trace.num_rows();
                    let own: Vec<_> = (0..rows)
                        .map(|row| multiplicity(trace, row, decode::cols::MU))
                        .collect();
                    let mut sums = own;
                    let mut previous = None;
                    for &(row, count) in &counts.decode {
                        if previous.is_some_and(|previous| row <= previous) {
                            return Err(invalid("unordered DECODE count rows"));
                        }
                        previous = Some(row);
                        let slot = sums
                            .get_mut(row)
                            .ok_or_else(|| invalid("invalid DECODE row"))?;
                        *slot = slot
                            .checked_add(count)
                            .filter(|&n| n <= crate::tables::cpu::MAX_CPU_ROWS)
                            .ok_or_else(|| invalid("invalid DECODE count"))?;
                    }
                    // The fixed columns are the same ones the existing finish built.
                    let mut out = TraceTable::new_main(
                        crate::tables::types::zeroed_fe_vec(rows * decode::cols::NUM_COLUMNS),
                        decode::cols::NUM_COLUMNS,
                        1,
                    );
                    for (row, count) in sums.into_iter().enumerate() {
                        for col in 0..decode::cols::NUM_COLUMNS {
                            let value = if col == decode::cols::MU {
                                count
                            } else {
                                multiplicity(trace, row, col)
                            };
                            out.main_table.set_u64(row, col, value);
                        }
                    }
                    if trace.narrow_main().is_some() {
                        out.pack_main_narrow();
                    }
                    *trace = out;
                    found[1] = true;
                }
                ("KECCAK_RC", WitnessData::Trace(trace, _)) => {
                    let count = (multiplicity(trace, 0, keccak_rc::cols::MU) as usize)
                        .checked_add(counts.keccak)
                        .filter(|&n| n <= crate::tables::cpu::MAX_CPU_ROWS as usize)
                        .ok_or_else(|| invalid("invalid KECCAK_RC count"))?;
                    let mut out = keccak_rc::generate_keccak_rc_trace();
                    keccak_rc::update_multiplicities(&mut out, count);
                    if trace.narrow_main().is_some() {
                        out.pack_main_narrow();
                    }
                    *trace = out;
                    found[2] = true;
                }
                ("HALT" | "REGISTER", _) => {}
                (name, _) if name.starts_with("PAGE") => {}
                (name, _) => return Err(invalid(&format!("unexpected range count table {name}"))),
            }
        }
        if found != [true; 3] {
            return Err(invalid("missing BITWISE, DECODE or KECCAK_RC count table"));
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build<'a>(
    job: &Job<'_>,
    airs: &'a HostAirs,
    program: &Elf,
    elf: &[u8],
    input: &[u8],
    window: usize,
    opts: &ProofOptions,
    format: &BlockFormat,
    options: &BlockOptions,
) -> Result<Built<'a>, Error> {
    let Run::Range(built) = prove_streamed_on_host(
        program,
        elf,
        input,
        window,
        opts,
        format,
        options,
        &Deviations::default(),
        Pass::Range(job),
        airs,
        &|_, _| {},
        None,
        &|_| {},
        None,
        &mut BlockStamps::default(),
    )?
    else {
        unreachable!("range builds return only their witness groups");
    };
    job.clock.mark("range-build-done");
    Ok(built)
}

pub(super) fn airs(
    job: &Job<'_>,
    window: usize,
    seed: [u8; 16],
    opts: &ProofOptions,
    format: &BlockFormat,
    options: &BlockOptions,
) -> HostAirs {
    let mut airs = HostAirs::new(opts, format, options);
    airs.stream.cpu_start = job.first * window;
    airs.stream.trace_seed = Some(seed);
    airs.stream.hash_witness = false;
    airs
}

pub(super) fn seed() -> Result<[u8; 16], Error> {
    if std::env::var("LAMBDA_VM_FIXED_TRACE_HASH").is_ok_and(|value| value.trim() == "1") {
        Ok([0; 16])
    } else {
        wire::random().map_err(|e| Error::Prover(format!("cluster trace seed: {e}")))
    }
}

pub(super) fn order(metas: &[Metadata]) -> Vec<(usize, usize)> {
    let windows = metas
        .iter()
        .enumerate()
        .flat_map(|(id, m)| (0..m.streamed).map(move |k| (id, k)));
    let finish = metas
        .iter()
        .enumerate()
        .flat_map(|(id, m)| (m.streamed..m.groups.len()).map(move |k| (id, k)));
    windows.chain(finish).collect()
}

/// The ranges' LOCAL_BYTES tables `locals` (range, table, shape), each
/// range's in its own group order, put in the block's group order
/// ([`order`]). Every group with a local sender ends with its own LOCAL_BYTES
/// table, and the verifier takes the `j`-th such group's to be the `j`-th
/// LOCAL_BYTES table (`check_local_groups`). The block takes every range's
/// streamed groups before any range's finish groups, so a range's finish
/// tables follow the next range's streamed ones.
fn local_bytes_in_block_order(
    metas: &[Metadata],
    order: &[(usize, usize)],
    locals: &mut VecDeque<(usize, usize, (usize, usize))>,
) -> Result<(), Error> {
    let mut place = BTreeMap::new();
    for (position, &(id, k)) in order.iter().enumerate() {
        for &table in &metas[id].groups[k] {
            place.insert((id, table as usize), position);
        }
    }
    let mut placed = locals
        .drain(..)
        .map(|local| {
            place
                .get(&(local.0, local.1))
                .map(|&position| (position, local))
                .ok_or_else(|| invalid("missing range table"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    placed.sort_by_key(|&(position, _)| position);
    locals.extend(placed.into_iter().map(|(_, local)| local));
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn assemble(
    metas: &[Metadata],
    cuts: &[usize],
    window: usize,
    terminal: &Laid,
    elf: &[u8],
    program: &Elf,
    opts: &ProofOptions,
    format: &BlockFormat,
) -> Result<(Laid, Plan), Error> {
    if metas.len() != cuts.len() + 1 {
        return Err(invalid("missing range receipt"));
    }
    check_table_count(metas.iter().map(|list| list.tables.len()).sum())?;
    let last = metas.len() - 1;
    let actual_end = metas[last].end;
    if actual_end == 0 {
        return Err(invalid("missing terminal window"));
    }
    let terminal_window = (actual_end - 1) / window * window;
    let mut tables = BTreeMap::<&str, VecDeque<(usize, usize, (usize, usize))>>::new();
    let mut index = Vec::with_capacity(metas.len());
    for (id, list) in metas.iter().enumerate() {
        let cut = |window_id: usize| (window_id * window).min(terminal_window);
        let start = id.checked_sub(1).map_or(0, |previous| cut(cuts[previous]));
        let end = cuts.get(id).map_or(actual_end, |&bound| cut(bound));
        let empty = start == end;
        if (empty
            && (list.start != 0
                || list.end != 0
                || !list.tables.is_empty()
                || !list.groups.is_empty()))
            || (!empty && (list.start != start || list.end != end))
            || list.streamed > list.finish
            || list.finish > list.groups.len()
            || list.groups.len() > format.max_groups
            || (id != last && list.finish != list.groups.len())
        {
            return Err(invalid("range receipt does not cover its assigned cycles"));
        }
        let mut seen = vec![false; list.tables.len()];
        for group in &list.groups {
            if group.is_empty() {
                return Err(invalid("empty range group"));
            }
            for &table in group {
                let flag = seen
                    .get_mut(table as usize)
                    .ok_or_else(|| invalid("range table index out of bounds"))?;
                if std::mem::replace(flag, true) {
                    return Err(invalid("duplicate range table"));
                }
            }
        }
        if seen.iter().any(|flag| !flag) {
            return Err(invalid("missing range table"));
        }
        index.push(vec![0; list.tables.len()]);
        let mut cpu_rows = 0;
        for (i, (name, shape)) in list.tables.iter().enumerate() {
            if (id != last && global(name)) || shape.1 > BLOCK_MAX_TABLE_VARS || shape.0 == 0 {
                return Err(invalid("invalid range table descriptor"));
            }
            if family(name) == "CPU" {
                cpu_rows += 1usize << shape.1;
            }
            tables
                .entry(family(name))
                .or_default()
                .push_back((id, i, *shape));
        }
        if id != last && cpu_rows != end - start || id == last && cpu_rows < end - start {
            return Err(invalid("CPU table heights disagree with range boundaries"));
        }
    }
    let n = |family: &str| tables.get(family).map_or(0, VecDeque::len);
    let table_counts = TableCounts {
        cpu: n("CPU"),
        lt: n("LT"),
        memw: n("MEMW"),
        memw_aligned: n("MEMW_A"),
        load: n("LOAD"),
        mul: n("MUL"),
        dvrm: n("DVRM"),
        shift: n("SHIFT"),
        branch: n("BRANCH"),
        memw_register: n("MEMW_R"),
        eq: n("EQ"),
        bytewise: n("BYTEWISE"),
        store: n("STORE"),
        cpu32: n("CPU32"),
        keccak: n("KECCAK") + n("KECCAK_W"),
        keccak_rnd: n("KECCAK_RND"),
        ecsm: n("ECSM"),
        ecdas: n("ECDAS"),
        hint: n("HINT"),
        copy: n("COPY") + n("COPY_W"),
        blake3: n("BLAKE3"),
        is_wh: n("IS_WH"),
        ecrec: n("ECREC") + n("ECREC_W"),
    };
    validate_block_counts(&table_counts)?;
    let fused = ["MEM_LOAD", "MEM_STORE"].map(n);
    let words = crate::tables::word_cpu::WORD_TABLES.map(n);
    let page_configs = Traces::page_configs_from_elf_and_runtime(
        program,
        &terminal.runtime_page_ranges,
        terminal.num_private_input_pages,
        max_block_pages(metas[last].tables.len(), format),
    )?;
    let heights = tables["CPU"]
        .iter()
        .map(|&(_, _, (_, height))| 1 << height)
        .collect();
    let inputs = BlockAirInputs::new(opts, program, heights, &terminal.public_output)?;
    let mut airs = block_airs(
        program,
        opts,
        page_air_pages(format.memory, &page_configs),
        &table_counts,
        &inputs,
        format.keccak_rnd,
        format.memw_aligned,
        format.registers,
        format.ecdas,
    );
    airs.with_fused_memory(opts, fused);
    if format.memory == crate::tables::fused_memory::MemoryFormat::Words {
        airs.with_word_memory_at(opts, words, &page_configs, format.page_runs())?;
    }
    let local_count = n("LOCAL_BYTES");
    airs.with_local_bytes(opts, local_count);
    let order = order(metas);
    if let Some(locals) = tables.get_mut("LOCAL_BYTES") {
        local_bytes_in_block_order(metas, &order, locals)?;
    }
    let refs = airs.air_refs();
    let mut shapes = Vec::with_capacity(refs.len());
    for (canonical, air) in refs.iter().enumerate() {
        let name = family(air.name());
        let Some((id, i, shape)) = tables.get_mut(name).and_then(VecDeque::pop_front) else {
            return Err(invalid("missing canonical range table"));
        };
        if air.trace_layout().0 != shape.0 {
            return Err(invalid("range table width disagrees with AIR"));
        }
        index[id][i] = canonical as u32;
        shapes.push(shape);
    }
    if tables.values().any(|family| !family.is_empty()) {
        return Err(invalid("unassigned range table"));
    }
    check_group_count(order.len(), format.max_groups)?;
    let mut plan = Plan {
        groups: Vec::with_capacity(order.len()),
        owners: Vec::with_capacity(order.len()),
        statement: Vec::new(),
        spanning_groups: 0,
        terminal_groups: 0,
    };
    let groups = order
        .into_iter()
        .map(|(id, k)| {
            plan.owners.push(id);
            let group = &metas[id].groups[k];
            plan.groups.push(
                group
                    .iter()
                    .map(|&i| {
                        let (name, shape) = &metas[id].tables[i as usize];
                        TableDescriptor {
                            key: Key::Air(i as usize),
                            name: name.clone(),
                            shape: *shape,
                            records: [0; 32],
                        }
                    })
                    .collect(),
            );
            group.iter().map(|&i| index[id][i as usize]).collect()
        })
        .collect();
    let laid = Laid {
        table_names: Vec::new(),
        table_counts,
        fused_memory: fused.map(|n| n as u64),
        word_memory: words.map(|n| n as u64),
        runtime_page_ranges: terminal.runtime_page_ranges.clone(),
        num_private_input_pages: terminal.num_private_input_pages,
        public_output: terminal.public_output.clone(),
        shapes,
        groups,
        prepared: Vec::new(),
        busy: 0.0,
        layout: LayoutStamps::default(),
        streamed_air: Vec::new(),
    };
    let heights = laid
        .shapes
        .iter()
        .map(|&(_, n)| n as u8)
        .collect::<Vec<_>>();
    plan.statement = block_statement_calls(
        laid.statement(&heights),
        &block_format_tag(format),
        elf,
        &format.chain_config(&laid.shapes),
        format.memory,
    )
    .map_err(Error::Prover)?;
    Ok((laid, plan))
}

pub(in crate::block_whir) fn commit<'a, H: multilinear::whir_hash::WhirHash>(
    group: CommitAssignment<'a>,
    global_index: usize,
    config: &multilinear::whir_chain::ChainConfig,
    options: &BlockOptions,
    format: &BlockFormat,
) -> Result<Committed<'a>, Error> {
    let tables = group
        .witness
        .into_iter()
        .map(WitnessTable::materialize)
        .collect::<Result<_, _>>()?;
    let local = format
        .local_lookups()
        .then(|| multilinear_block::LocalPhaseA {
            buses: local_bytes::LOCAL_BUSES.to_vec(),
            lane: false,
            binding: stark::multilinear_table::batched::LocalBinding::ALL,
        });
    BlockCommitted::commit_replica_group::<H, H::Sponge>(
        global_index,
        tables,
        config,
        options.drop_levels,
        options.narrow,
        local,
    )
    .map_err(|e| Error::Prover(format!("cluster range group {global_index}: {e:?}")))
}

/// An explicit cut list runs the same range builder in one process. With
/// in-process workers, the same list is built concurrently.
#[cfg(test)]
pub(in crate::block_whir) fn configured() -> Result<Option<Vec<usize>>, Error> {
    match std::env::var("LAMBDA_VM_CLUSTER_WINDOWS") {
        Ok(word) if word == "-" => Ok(Some(Vec::new())),
        Ok(word) => parse_cuts(&word).map(Some),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(e) => Err(invalid(&format!("LAMBDA_VM_CLUSTER_WINDOWS: {e}"))),
    }
}

#[allow(clippy::too_many_arguments)]
pub(in crate::block_whir) fn prove_local(
    cuts: &[usize],
    workers: Option<multilinear_block::InprocWorkers>,
    program: &Elf,
    elf: &[u8],
    input: &[u8],
    window: usize,
    opts: &ProofOptions,
    format: &BlockFormat,
    options: &BlockOptions,
    on_statement: StatementObserver<'_>,
    on_group: GroupObserver<'_>,
    stamps: &mut BlockStamps,
    cluster_map: Option<&mut crate::lfm::cluster_levels::LeafOwners>,
    clock: Clock,
) -> Result<BlockWhirProof<LocalArgues>, Error> {
    validate_cuts(cuts, window, options.max_rows.cpu)?;
    let start = Instant::now();
    set_up_card();
    let seed = seed()?;
    let jobs: Vec<_> = (0..=cuts.len())
        .map(|id| {
            let lane = if id == cuts.len() {
                clock
            } else {
                clock.for_worker(id)
            };
            Job::new(cuts, id, None).with_clock(lane)
        })
        .collect();
    let airs: Vec<_> = jobs
        .iter()
        .map(|job| airs(job, window, seed, opts, format, options))
        .collect();
    let blake3 = format.zf.base == BaseFormat::B3_WHIR;
    if blake3 {
        checked_base_only(&format.zf.base)?;
    }
    let build_one = |job: &Job<'_>, airs| {
        let run = || {
            job.clock.mark("reference-range-start");
            build(
                job, airs, program, elf, input, window, opts, format, options,
            )
        };
        if blake3 {
            crate::base_only::with_blake3_levels(run)
        } else {
            run()
        }
    };
    let mut built: Vec<_> = if workers.is_some() {
        std::thread::scope(|scope| {
            jobs.iter()
                .zip(&airs)
                .map(|(job, airs)| {
                    let build_one = &build_one;
                    scope.spawn(move || build_one(job, airs))
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| {
                    handle
                        .join()
                        .map_err(|_| invalid("range builder panicked"))?
                })
                .collect::<Result<_, _>>()
        })?
    } else {
        jobs.iter()
            .zip(&airs)
            .map(|(job, airs)| build_one(job, airs))
            .collect::<Result<_, _>>()?
    };
    #[cfg(all(test, feature = "test-security"))]
    if let Some(tamper) = ECREC_TAMPER.with(std::cell::Cell::get) {
        tamper_ecrec(&mut built, tamper, format.memory);
    }
    let mut metas = Vec::with_capacity(built.len());
    let mut counts = Vec::new();
    for (range, job) in built.iter_mut().zip(&jobs) {
        let (meta, owned_counts) = range.describe(job)?;
        metas.push(meta);
        if !owned_counts.bitwise.is_empty() {
            counts.push(owned_counts);
        }
    }
    let terminal = built
        .last_mut()
        .ok_or_else(|| invalid("missing terminal range"))?;
    for counts in &counts {
        terminal.add_counts(counts)?;
    }
    let (laid, plan) = assemble(
        &metas,
        cuts,
        window,
        &terminal.laid,
        elf,
        program,
        opts,
        format,
    )?;
    let verify_base = cluster_map.is_none();
    if let Some(map) = cluster_map {
        *map = crate::lfm::cluster_levels::LeafMap::from_owners(&plan.owners, cuts.len() + 1)
            .map_err(|e| invalid(&e))?;
    }
    let mut pending: Vec<_> = built
        .into_iter()
        .map(|b| b.groups.into_iter().map(Some).collect::<Vec<_>>())
        .collect();
    let heights = laid
        .shapes
        .iter()
        .map(|&(_, n)| n as u8)
        .collect::<Vec<_>>();
    let statement = laid.statement(&heights);
    let frame = block_frame(statement, elf, opts, format)?;
    crate::with_base_hash!(format.zf.base, |H| {
        let mut parts = Vec::with_capacity(laid.groups.len());
        for (index, (id, k)) in order(&metas).into_iter().enumerate() {
            let group = pending[id][k]
                .take()
                .ok_or_else(|| invalid("duplicate group index"))?;
            parts.push(Some(commit::<H>(
                group,
                index,
                &frame.config,
                options,
                format,
            )?));
        }
        clock.mark("final-commits-done");
        let mut block = BlockCommitted::assemble_replica_groups(parts)
            .map_err(|e| invalid(&format!("{e:?}")))?;
        let prepared = commit_groups_on_host::<H>(
            group_stacks(
                prepared_tables(&frame.airs, &frame.page_configs, format)?,
                &laid.groups,
            )?,
            &frame.config,
        )?;
        let roots = prepared_roots(&prepared);
        clock.mark("prepared-roots-done");
        let borrowed: Vec<Vec<&Mle<F>>> = prepared
            .iter()
            .map(|p| multilinear::stacking::borrow(&p.columns))
            .collect();
        let openings: Vec<_> = prepared
            .iter()
            .zip(&borrowed)
            .map(|(p, columns)| multilinear_block::BlockPrepared {
                card: 0,
                group: p.group,
                tables: p.tables.clone(),
                commitment: (&p.commitment).into(),
                columns,
                packed: None,
            })
            .collect();
        stamps.phase_a = start.elapsed().as_secs_f64();
        stamps.tables = laid.shapes.len();
        let started = Instant::now();
        let mut transcript = H::sponge();
        if let Some(workers) = workers {
            let refs = frame.airs.air_refs();
            let fixed = cluster_preprocessed(&laid.groups, |i| refs.get(i).copied())?;
            block.set_inproc_owners(workers, plan.statement, fixed, Some(plan.owners));
        } else {
            for bytes in plan.statement {
                transcript.append_bytes(&bytes);
            }
        }
        on_statement(statement, &roots);
        clock.mark("base-openings-start");
        let opening_marks = (!verify_base).then(|| {
            laid.groups
                .iter()
                .map(|_| OnceLock::new())
                .collect::<Vec<_>>()
        });
        let order: Vec<_> = (0..laid.groups.len()).collect();
        let (proof, argues, prepared, groups, commit_ood, local) =
            multilinear_block::block_prove_in_order::<_, _, _, H>(
                block,
                &frame.config,
                &mut transcript,
                &openings,
                &[],
                &|g| g,
                &Default::default(),
                &|group| {
                    if let Some(marks) = &opening_marks
                        && let Some(stamp) = clock.sample()
                    {
                        let _ = marks[group.group].set(stamp);
                    }
                    on_group(group);
                },
                &order,
                None,
            )
            .map_err(|e| invalid(&format!("{e:?}")))?;
        // The reference can pipeline base groups. Use the prover's own
        // per-group spans and the wall stamp from the opening callback.
        if let Some(marks) = opening_marks {
            for (group, (stamp, timing)) in marks.into_iter().zip(&groups).enumerate() {
                if let Some(stamp) = stamp.into_inner() {
                    clock.opened_at(group, stamp, timing.end_b - timing.start_b);
                }
            }
        }
        clock.mark("base-openings-end");
        stamps.phase_b = started.elapsed().as_secs_f64();
        stamps.groups = groups;
        let proof = BlockWhirProof {
            proof,
            table_num_vars: heights,
            runtime_page_ranges: laid.runtime_page_ranges,
            table_counts: laid.table_counts,
            public_output: laid.public_output,
            num_private_input_pages: laid.num_private_input_pages,
            groups: laid.groups,
            prepared,
            argues,
            commit_ood,
            fused_memory: laid.fused_memory,
            word_memory: laid.word_memory,
            local,
        };
        if !verify_base {
            return Ok(proof);
        }
        let mut proof = proof;
        let local = format
            .local_lookups()
            .then(|| std::mem::take(&mut proof.local));
        let mut proof = proof.with_local(local);
        verify_cluster_base(&proof, elf, opts, format)?;
        let local = proof.local.take().unwrap_or_default();
        Ok(proof.with_local(local))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use stark::proof::options::BaseFormat;

    // test_commit_4 runs 18 cycles: windows of four cycles, the fifth holds
    // the halt.
    const WINDOW: usize = 4;

    /// A refusal's expected reason and the alteration that should cause it.
    type Case<'a, T> = (&'static str, &'a dyn Fn(&mut T));

    struct Fixture {
        elf: Vec<u8>,
        program: Elf,
        opts: ProofOptions,
        format: BlockFormat,
        options: BlockOptions,
    }

    fn fixture() -> Fixture {
        fixture_of("test_commit_4")
    }

    fn fixture_of(program: &str) -> Fixture {
        let elf = crate::test_utils::asm_elf_bytes(program);
        let program = Elf::load(&elf).unwrap();
        let mut options = crate::tests::block_whir_p1_tests::options(3);
        options.keccak_rnd_rows_log2 = 3;
        options.window_log2 = Some(WINDOW.ilog2() as usize);
        options.max_rows.cpu = 2 * WINDOW;
        options.drop_streamed_ops = true;
        Fixture {
            elf,
            program,
            opts: ProofOptions::default_test_options(),
            format: crate::tests::block_whir_p1_tests::many_groups(BaseFormat::B3_WHIR),
            options,
        }
    }

    impl Fixture {
        fn airs(&self, jobs: &[Job<'_>]) -> Vec<HostAirs> {
            jobs.iter()
                .map(|job| {
                    airs(
                        job,
                        1 << self.options.window_log2.unwrap(),
                        [0; 16],
                        &self.opts,
                        &self.format,
                        &self.options,
                    )
                })
                .collect()
        }

        fn build<'a>(&self, job: &Job<'_>, airs: &'a HostAirs) -> Built<'a> {
            crate::base_only::with_blake3_levels(|| {
                build(
                    job,
                    airs,
                    &self.program,
                    &self.elf,
                    &[],
                    1 << self.options.window_log2.unwrap(),
                    &self.opts,
                    &self.format,
                    &self.options,
                )
            })
            .expect("an honest range builds")
        }

        /// Every range's receipt and counts, and the terminal range's layout.
        fn receipts(&self, cuts: &[usize]) -> (Vec<Metadata>, Vec<Counts>, Laid) {
            let jobs: Vec<_> = (0..=cuts.len())
                .map(|id| Job::new(cuts, id, None))
                .collect();
            let airs = self.airs(&jobs);
            let mut metas = Vec::new();
            let mut counts = Vec::new();
            let mut terminal = None;
            for (job, airs) in jobs.iter().zip(&airs) {
                let mut built = self.build(job, airs);
                let (meta, own) = built.describe(job).unwrap();
                metas.push(meta);
                if job.terminal() {
                    terminal = Some(built.laid);
                } else if !own.bitwise.is_empty() {
                    counts.push(own);
                }
            }
            (metas, counts, terminal.unwrap())
        }

        fn assemble(
            &self,
            metas: &[Metadata],
            cuts: &[usize],
            terminal: &Laid,
            format: &BlockFormat,
        ) -> Result<(), String> {
            assemble(
                metas,
                cuts,
                1 << self.options.window_log2.unwrap(),
                terminal,
                &self.elf,
                &self.program,
                &self.opts,
                format,
            )
            .map(drop)
            .map_err(|e| e.to_string())
        }

        /// A fresh terminal range of `cuts`, refusing or accepting `counts`.
        fn add_counts(
            &self,
            cuts: &[usize],
            counts: &[Counts],
            alter: impl FnOnce(&mut Built<'_>),
        ) -> Result<(), String> {
            let job = Job::new(cuts, cuts.len(), None);
            let airs = self.airs(std::slice::from_ref(&job));
            let mut built = self.build(&job, &airs[0]);
            alter(&mut built);
            counts
                .iter()
                .try_for_each(|counts| built.add_counts(counts))
                .map_err(|e| e.to_string())
        }
    }

    fn refused(result: Result<(), String>, reason: &str) {
        let error = result.expect_err(reason);
        assert!(error.contains(reason), "expected {reason:?}, got {error:?}");
    }

    /// Two ranges whose LOCAL_BYTES tables close a streamed and a finish
    /// group each: the block takes both streamed groups first, so range 0's
    /// finish table comes after range 1's streamed one, and the canonical
    /// LOCAL_BYTES order follows the groups, not the ranges. A table in no
    /// group is refused.
    ///
    /// Mutation: the sort dropped and the ranges' order is kept.
    #[test]
    fn local_bytes_tables_take_the_blocks_group_order() {
        let meta = |groups: Vec<Vec<u32>>| Metadata {
            start: 0,
            end: 0,
            tables: Vec::new(),
            groups,
            streamed: 1,
            finish: 2,
        };
        // Range 0: CPU 0 + LOCAL 1 streamed, ECDAS 2 + LOCAL 3 at the
        // finish; range 1 the same.
        let metas = [
            meta(vec![vec![0, 1], vec![2, 3]]),
            meta(vec![vec![0, 1], vec![2, 3]]),
        ];
        let groups = order(&metas);
        assert_eq!(groups, [(0, 0), (1, 0), (0, 1), (1, 1)]);
        let shape = (3, 16);
        let mut locals: VecDeque<_> = [(0, 1, shape), (0, 3, shape), (1, 1, shape), (1, 3, shape)]
            .into_iter()
            .collect();
        local_bytes_in_block_order(&metas, &groups, &mut locals).unwrap();
        assert_eq!(
            locals.iter().map(|&(id, i, _)| (id, i)).collect::<Vec<_>>(),
            [(0, 1), (1, 1), (0, 3), (1, 3)]
        );
        // One range: its own order, unchanged.
        let one = [meta(vec![vec![0, 1], vec![2, 3]])];
        let mut locals: VecDeque<_> = [(0, 1, shape), (0, 3, shape)].into_iter().collect();
        local_bytes_in_block_order(&one, &order(&one), &mut locals).unwrap();
        assert_eq!(
            locals.iter().map(|&(id, i, _)| (id, i)).collect::<Vec<_>>(),
            [(0, 1), (0, 3)]
        );
        let mut stray: VecDeque<_> = [(0, 5, shape)].into_iter().collect();
        refused(
            local_bytes_in_block_order(&one, &order(&one), &mut stray).map_err(|e| e.to_string()),
            "missing range table",
        );
    }

    /// ★ keccak256_calls_in_two_ranges (three keccak256 calls, one
    /// permutation each, at cycles 5, 20 and 25) cut at cycle 16 with windows
    /// of 8 cycles, under the cluster's BLAKE3 format, whose KECCAK_RND tables
    /// send their byte lookups to their group's own LOCAL_BYTES table, with
    /// the word memory format, and with KECCAK_RND's 32-row tables handed out
    /// during the windows: range 0's one call (24 rows) fills no table, so its
    /// KECCAK_RND table and LOCAL_BYTES close a finish group; range 1's two
    /// calls fill one, which closes a streamed group. The block takes range
    /// 1's streamed groups before range 0's finish groups, so the assembled
    /// statement numbers the LOCAL_BYTES tables in that order, and the
    /// verifier's own frame (its group rule among its checks) accepts it.
    /// (Under the split memory format each range's KECCAK calls make an IS_WH
    /// table of their own, and a block takes one; the word format's keccak256
    /// makes none.)
    ///
    /// Mutation: the LOCAL_BYTES tables left in range order in `assemble`
    /// (the sort's call removed) and the frame refuses the statement ("group
    /// N ends with LOCAL_BYTES table M, not M − 1").
    #[test]
    fn local_bytes_tables_of_two_ranges_meet_the_verifiers_group_rule() {
        let mut f = fixture_of("keccak256_calls_in_two_ranges");
        f.format = crate::lfm::levels::pin_family(
            BlockFormat::pinned_with(multilinear::whir_chain::GroupFold::Indexed),
            crate::lfm::wlfm::WlfmFamily::B3,
            crate::block_whir::B3Rounds::Fri,
        )
        .unwrap()
        .with_memory(crate::tables::fused_memory::MemoryFormat::Words);
        assert!(
            f.format.local_lookups(),
            "the cluster format's local lookups"
        );
        f.opts = crate::lfm::proof::block_base_options_under(&f.format.zf);
        let window = 8;
        f.options.window_log2 = Some(3);
        f.options.max_rows.cpu = 2 * window;
        f.options.layout_workers = 2;
        f.options.keccak_rnd_rows_log2 = 5;
        f.options.finish_keccak_rnd_chunks = true;
        let cuts = [16 / window];
        // KECCAK_RND's tables handed out during the windows, every other
        // build setting unset.
        let mut settings = vec![None; super::super::tcp::BUILD_SETTINGS.len()];
        let stream = super::super::tcp::BUILD_SETTINGS
            .iter()
            .position(|&name| name == "LAMBDA_VM_STREAM_KECCAK_RND")
            .expect("a build setting");
        settings[stream] = Some("1".to_string());
        let (metas, _, terminal) =
            super::super::tcp::with_settings(Some(settings.into()), || f.receipts(&cuts));
        // Where each range's LOCAL_BYTES tables sit: (range, group, streamed).
        let locals: Vec<(usize, usize, bool)> = metas
            .iter()
            .enumerate()
            .flat_map(|(id, meta)| {
                meta.groups
                    .iter()
                    .enumerate()
                    .filter_map(move |(k, group)| {
                        group
                            .iter()
                            .any(|&i| family(&meta.tables[i as usize].0) == "LOCAL_BYTES")
                            .then_some((id, k, k < meta.streamed))
                    })
            })
            .collect();
        assert!(
            locals.iter().any(|&(id, _, streamed)| id == 0 && !streamed),
            "range 0 closes a finish group with LOCAL_BYTES: {locals:?}"
        );
        assert!(
            locals.iter().any(|&(id, _, streamed)| id == 1 && streamed),
            "range 1 closes a streamed group with LOCAL_BYTES: {locals:?}"
        );
        let (laid, _) = assemble(
            &metas, &cuts, window, &terminal, &f.elf, &f.program, &f.opts, &f.format,
        )
        .expect("honest receipts assemble");
        let heights: Vec<u8> = laid.shapes.iter().map(|&(_, n)| n as u8).collect();
        crate::block_whir::block_frame(laid.statement(&heights), &f.elf, &f.opts, &f.format)
            .expect("the verifier's frame accepts the assembled statement");
    }

    #[test]
    fn range_lists_refuse_each_bad_word_and_cut() {
        assert_eq!(parse_cuts("1,2").unwrap(), [1, 2]);
        for (word, reason) in [
            ("", "empty window list"),
            ("1,no", "invalid window index"),
            (",1", "invalid window index"),
            ("-1", "invalid window index"),
        ] {
            refused(
                parse_cuts(word).map(drop).map_err(|e| e.to_string()),
                reason,
            );
        }
        let last = crate::tables::cpu::MAX_CPU_ROWS as usize / WINDOW;
        for (cuts, window, rows) in [
            (vec![], WINDOW, 8),
            (vec![1, 2], WINDOW, 8),
            (vec![1, last], WINDOW, 4),
            ((1..wire::MAX_WORKERS).collect(), WINDOW, 8),
        ] {
            validate_cuts(&cuts, window, rows).expect("a valid list");
        }
        for (cuts, window, rows) in [
            ((1..=wire::MAX_WORKERS).collect(), WINDOW, 8),
            (vec![1], 2, 4),
            (vec![1], WINDOW, 2),
            (vec![1], WINDOW, 16),
            (vec![1], 8, 12),
            (vec![0, 1], WINDOW, 8),
            (vec![2, 2], WINDOW, 8),
            (vec![3, 2], WINDOW, 8),
            (vec![1, last + 1], WINDOW, 8),
            (vec![1], usize::MAX, 8),
        ] {
            refused(
                validate_cuts(&cuts, window, rows).map_err(|e| e.to_string()),
                "invalid window ranges or CPU chunk size",
            );
        }
    }

    #[test]
    fn assemble_refuses_each_bad_range_receipt() {
        let f = fixture();
        let cuts = [1, 2];
        let (metas, _, terminal) = f.receipts(&cuts);
        let last = metas.len() - 1;
        f.assemble(&metas, &cuts, &terminal, &f.format)
            .expect("honest receipts assemble");
        let family_index = |id: usize, family: &str| {
            metas[id]
                .tables
                .iter()
                .position(|(name, _)| super::family(name) == family)
                .unwrap_or_else(|| panic!("range {id} states a {family} table"))
        };
        let other = metas[0]
            .tables
            .iter()
            .position(|(name, _)| super::family(name) != "CPU")
            .expect("range 0 states a table besides CPU");
        let cpu = family_index(0, "CPU");
        let decode = family_index(last, "DECODE");
        let altered = |alter: &dyn Fn(&mut Vec<Metadata>)| {
            let mut metas = metas.clone();
            alter(&mut metas);
            f.assemble(&metas, &cuts, &terminal, &f.format)
        };
        let cover = "range receipt does not cover its assigned cycles";
        let descriptor = "invalid range table descriptor";
        let cases: [Case<'_, Vec<Metadata>>; 21] = [
            ("missing range receipt", &|m| {
                m.pop();
            }),
            ("missing terminal window", &|m| m[last].end = 0),
            (cover, &|m| m[1].start += 1),
            (cover, &|m| m[0].end -= 1),
            (cover, &|m| m[last].streamed = m[last].finish + 1),
            (cover, &|m| m[last].finish = m[last].groups.len() + 1),
            (cover, &|m| {
                m[last].groups = vec![m[last].groups[0].clone(); BLOCK_MAX_GROUPS + 1];
                m[last].finish = m[last].groups.len();
            }),
            (cover, &|m| m[0].finish = m[0].groups.len() - 1),
            ("empty range group", &|m| m[last].groups.push(Vec::new())),
            ("range table index out of bounds", &|m| {
                let index = m[0].tables.len() as u32;
                m[0].groups[0].push(index);
            }),
            ("duplicate range table", &|m| {
                let index = m[0].groups[0][0];
                m[0].groups[0].push(index);
            }),
            ("missing range table", &|m| {
                let extra = m[0].tables[other].clone();
                m[0].tables.push(extra);
            }),
            (descriptor, &|m| m[0].tables[other].0 = "BITWISE".into()),
            (descriptor, &|m| {
                m[0].tables[other].1.1 = BLOCK_MAX_TABLE_VARS + 1
            }),
            (descriptor, &|m| m[0].tables[other].1.0 = 0),
            ("CPU table heights disagree with range boundaries", &|m| {
                m[0].tables[cpu].1.1 += 1
            }),
            ("CPU table heights disagree with range boundaries", &|m| {
                for (name, shape) in &mut m[last].tables {
                    if super::family(name) == "CPU" {
                        shape.1 = 0;
                    }
                }
            }),
            ("missing canonical range table", &|m| {
                m[last].tables[decode].0 = "UNKNOWN".into()
            }),
            ("unassigned range table", &|m| {
                m[0].tables.push(("UNKNOWN".into(), (1, 2)));
                let index = m[0].tables.len() as u32 - 1;
                m[0].groups[0].push(index);
            }),
            ("range table width disagrees with AIR", &|m| {
                m[0].tables[other].1.0 += 1
            }),
            // Each range stays under the block's table cap; their sum does not.
            ("a block states at most", &|m| {
                for meta in &mut m[..2] {
                    let extra = BLOCK_MAX_TABLES / 2 - meta.tables.len() + 1;
                    let table = meta.tables[0].clone();
                    meta.tables.extend(std::iter::repeat_n(table, extra));
                    assert!(meta.tables.len() <= BLOCK_MAX_TABLES);
                }
            }),
        ];
        for (reason, alter) in cases {
            refused(altered(alter), reason);
        }
        // Each range stays under the group cap; the block's sum does not.
        let total = metas.iter().map(|m| m.groups.len()).sum::<usize>();
        assert!(metas.iter().all(|m| m.groups.len() < total - 1));
        let mut capped = f.format;
        capped.max_groups = total - 1;
        refused(
            f.assemble(&metas, &cuts, &terminal, &capped),
            "a block takes at most",
        );
    }

    #[test]
    fn assemble_takes_empty_ranges_and_refuses_an_empty_range_with_cycles() {
        let f = fixture();
        let cuts = [1, 10, 20];
        let (metas, _, terminal) = f.receipts(&cuts);
        assert_eq!((metas[2].start, metas[2].end), (0, 0));
        f.assemble(&metas, &cuts, &terminal, &f.format)
            .expect("an overestimated list assembles");
        for alter in [
            (|m: &mut Metadata| m.end = 1) as fn(&mut Metadata),
            |m| m.start = 1,
            |m| m.groups.push(vec![0]),
            |m| m.tables.push(("LT[0]".into(), (1, 2))),
        ] {
            let mut metas = metas.clone();
            alter(&mut metas[2]);
            refused(
                f.assemble(&metas, &cuts, &terminal, &f.format),
                "range receipt does not cover its assigned cycles",
            );
        }
    }

    #[test]
    fn in_row_cpu_finishes_empty_ranges() {
        let mut f = fixture();
        f.format.registers = crate::tables::register::RegisterFormat::InRow;
        f.options.layout_workers = 2;
        f.options.keccak_rnd_rows_log2 = 5;
        let production_window = 1 << BLOCK_WINDOW_LOG2;
        let cases = [
            (
                production_window,
                [1].as_slice(),
                [(0, 0), (0, 18)].as_slice(),
            ),
            (production_window, &[2], &[(0, 0), (0, 18)]),
            (WINDOW, &[4], &[(0, 16), (16, 18)]),
            (WINDOW, &[4, 5], &[(0, 16), (0, 0), (16, 18)]),
        ];
        for finish_stream in [true, false] {
            f.options.finish_stream = finish_stream;
            for (window, cuts, expected) in cases {
                f.options.window_log2 = Some(window.ilog2() as usize);
                f.options.max_rows.cpu = 2 * window;
                let (metas, _, terminal) = f.receipts(cuts);
                let cycles: Vec<_> = metas.iter().map(|m| (m.start, m.end)).collect();
                assert_eq!(cycles, expected, "window {window}, cuts {cuts:?}");
                f.assemble(&metas, cuts, &terminal, &f.format)
                    .expect("honest in-row receipts assemble");
            }
        }
    }

    #[test]
    fn add_counts_refuses_each_bad_count_and_table_kind() {
        let f = fixture();
        let cuts = [1, 2];
        let (_, counts, _) = f.receipts(&cuts);
        assert_eq!(counts.len(), 2);
        assert!(counts[0].decode.len() >= 2, "two DECODE rows to reorder");
        f.add_counts(&cuts, &counts, |_| {})
            .expect("honest counts add");
        const OVER: u64 = crate::tables::cpu::MAX_CPU_ROWS + 1;
        let cases: [Case<'_, Counts>; 9] = [
            ("invalid range BITWISE counts", &|c| {
                c.bitwise.pop();
            }),
            ("invalid range BITWISE counts", &|c| c.bitwise.push(0)),
            ("invalid range BITWISE counts", &|c| {
                c.bitwise[0] = math::field::goldilocks::GOLDILOCKS_PRIME
            }),
            ("unordered DECODE count rows", &|c| c.decode.swap(0, 1)),
            ("unordered DECODE count rows", &|c| {
                c.decode[1].0 = c.decode[0].0
            }),
            ("invalid DECODE row", &|c| c.decode.push((usize::MAX, 1))),
            ("invalid DECODE count", &|c| c.decode[0].1 = OVER),
            ("invalid DECODE count", &|c| c.decode[0].1 = u64::MAX),
            ("invalid KECCAK_RC count", &|c| c.keccak = OVER as usize),
        ];
        for (reason, alter) in cases {
            let mut counts = counts.clone();
            alter(&mut counts[0]);
            refused(f.add_counts(&cuts, &counts, |_| {}), reason);
        }
        // A table other than the global ones among the global groups.
        refused(
            f.add_counts(&cuts, &counts, |built| {
                let group = built
                    .groups
                    .iter()
                    .position(|group| group.witness.iter().any(|table| !table.is_global()))
                    .expect("the terminal range builds its own tables");
                let at = built.groups[group]
                    .witness
                    .iter()
                    .position(|table| !table.is_global())
                    .unwrap();
                let table = built.groups[group].witness.remove(at);
                built.groups.last_mut().unwrap().witness.push(table);
            }),
            "unexpected range count table",
        );
        for name in ["BITWISE", "DECODE", "KECCAK_RC"] {
            refused(
                f.add_counts(&cuts, &counts, |built| {
                    for group in &mut built.groups {
                        group.witness.retain(|table| table.air.name() != name);
                    }
                }),
                "missing BITWISE, DECODE or KECCAK_RC count table",
            );
        }
    }

    #[test]
    fn cluster_compact_counts_preserve_dense_merge() {
        use crate::tables::bitwise::BitwiseHistogram;

        let mut dense_merge = BitwiseHistogram::new();
        let mut compact_merge = BitwiseHistogram::new();
        for n in [127, 16_384] {
            let mut dense = Counts {
                bitwise: vec![0; BITWISE_LEN],
                decode: vec![(2, 11), (9, 20)],
                keccak: 3,
            };
            for (index, count) in [(0, 1), (7, n), (8, n), (BITWISE_LEN - 1, 1 << 40)] {
                dense.bitwise[index] = count;
            }
            dense_merge.add_range_counts(&dense.bitwise).unwrap();
            let compact = dense.compact().unwrap();
            let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&compact).unwrap();
            let expanded = super::super::wire::decode::<CompactCounts>(&bytes)
                .unwrap()
                .expand()
                .unwrap();
            assert_eq!(expanded.bitwise, dense.bitwise);
            assert_eq!(expanded.decode, dense.decode);
            assert_eq!(expanded.keccak, dense.keccak);
            compact_merge.add_range_counts(&expanded.bitwise).unwrap();
        }
        assert!(compact_merge == dense_merge);
    }

    #[test]
    fn cluster_compact_counts_refuse_noncanonical_values() {
        for values in [&[][..], &[0], &[0x80, 0], &[1, 1], &[0xff; 10]] {
            let mut bitwise = vec![0; BITWISE_BITMAP];
            bitwise[0] = 1;
            bitwise.extend_from_slice(values);
            let compact = CompactCounts {
                bitwise,
                decode: Vec::new(),
                keccak: 0,
            };
            assert!(compact.expand().is_err(), "{values:?}");
        }
    }
}
