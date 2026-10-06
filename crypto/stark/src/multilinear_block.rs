//! One multilinear proof for a whole block, proved in two phases.
//!
//! The block's tables — every chunk of every table type, run at full height —
//! are split into **groups**: contiguous runs in table order, each as many
//! tables as fit a few stacked polynomials. That is the unit a card holds. The
//! proof then goes:
//!
//! 1. **Phase A (commit).** Group by group: the columns go up, the stack is
//!    committed, the roots are taken, and the codewords go — only the top of
//!    each tree stays, on the host ([`RetiredStack`]). No card holds a block's
//!    codewords, and every root must exist before the first challenge.
//! 2. **The roots block.** Every root, in group order, into the transcript;
//!    then `(z, α, β)` once, shared by every table of the block
//!    ([`absorb_roots_and_challenge`], the same function every other proof
//!    uses). The transcript's state after the draw is the block's `S_post`.
//! 3. **Phase B (prove), per group `g`, on the fork `S_post ‖ g`.** The group's
//!    columns go up again; each of its tables is argued (today's
//!    [`prove`]) — or, under [`ArgueFormat::Batched`], the group's tables are
//!    argued together ([`batched::prove_argue`]: a lockstep GKR per bin and one
//!    constraint sumcheck for the group); the group's codewords are recomputed
//!    — no hash, the tree's top is kept — and its stack is opened at the
//!    tables' points ([`stacked_eval::prove`]).
//!
//! Between the phases a group's columns can be held **narrow** ([`Narrowing`]):
//! packed on the card right after the group's commit, each column at the bytes
//! its words need, and widened on the card again when phase B uploads the group.
//! The words come back bit for bit, so the proof is the same byte for byte; a
//! widening that went wrong is refused when the opening reads a path from the
//! kept tree top (the recomputed codeword does not hash to it).
//!
//! The tables whose leading preprocessed columns are settled out of band are
//! stacked per group ([`BlockPrepared`]): each such stack's derived roots are
//! absorbed after the groups' roots, and its opening proved on its group's fork
//! after the group's own opening.
//!
//! The verifier mirrors it and checks the bus balance ONCE, over every table of
//! the block: the LogUp challenges are shared, so the block's fractions sum to
//! what the statement owes exactly as one monolithic proof's do.
//!
//! # Why a fork per group is sound
//!
//! Each fork starts from `S_post`, which binds the statement and every root of
//! the block, and then the group's index; everything the group's argument and
//! opening draw is a function of that and of the group's own messages. This is
//! the STARK prover's per-table fork (`crypto/stark/src/prover.rs`, `append(idx)`)
//! at the granularity of a group. A prover cannot move a group's messages to
//! another group (the index is absorbed), cannot choose a root after a
//! challenge (all are absorbed before `z`), and cannot mix columns of two
//! groups (each opening is against its own group's roots, sliced by the
//! verifier's own layouts).
//!
//! ★ THE GROUPS ARE THE VERIFIER'S, never the proof's: [`block_groups`] is a
//! function of the tables' shapes and two format constants, and both sides call
//! it.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::time::Instant;

use math::{
    field::{
        element::FieldElement,
        traits::{IsFFTField, IsField, IsPrimeField, IsSubFieldOf},
    },
    traits::AsBytes,
};
use multilinear::{
    Error as MlError,
    mle::Mle,
    narrow::ColumnOf,
    stacked_eval::{self, Claimed, ColumnsAt, RetiredStack, StackedCommitment, StackedProof},
    stacking::StackedLayout,
    whir::Domain,
    whir_chain::{ArgueFormat, ChainConfig, StackVars},
    whir_commit::Commitment,
    whir_hash::WhirHash,
};

use crate::narrow::NarrowMain;
use crate::spill::{Prefetch, ReadPhase, SpillStore, SpilledMain};

use crate::multilinear_table::{
    CommittedTable, MultiProof, TableProof, TableStatement, absorb_roots_and_challenge,
    batched::{self, BatchedArgue, ProverFaults, VerifierChecks, Where},
    check_preprocessed, contribution, global_layout, prove, verify,
};

/// How the block's tables split into groups: contiguous runs in table order,
/// each as many tables as its stack holds in at most `max_polys` polynomials
/// under the stack cap. A table that alone needs more is a group of its own.
///
/// `shapes` is `(committed columns, height in variables)` per table. Returns
/// the size of each group, in order; they sum to `shapes.len()`.
pub fn block_groups(
    shapes: &[(usize, usize)],
    cap: StackVars,
    max_polys: usize,
) -> Result<Vec<usize>, MlError> {
    let mut sizes = Vec::new();
    let mut start = 0usize;
    for end in 1..=shapes.len() {
        let polys = global_layout(&shapes[start..end], cap)?.num_polys();
        if polys > max_polys && end - start > 1 {
            sizes.push(end - 1 - start);
            start = end - 1;
        }
    }
    if start < shapes.len() {
        sizes.push(shapes.len() - start);
    }
    Ok(sizes)
}

/// What a group cost, for the readout. Seconds of wall time on the prover's
/// thread.
#[derive(Clone, Debug, Default)]
pub struct GroupStamps {
    pub tables: usize,
    pub polys: usize,
    pub cells: usize,
    /// Phase A: the wait for the group's tables (a streamed producer), the
    /// columns up, the commit, the roots, the tops home.
    pub wait_a: f64,
    pub upload_a: f64,
    /// Of `upload_a`, the seconds the committer waited for: all of it unless
    /// the upload ran beside the previous group's commit, when only what
    /// outlasted that commit is paid (`upload_a − upload_paid` is hidden).
    pub upload_paid: f64,
    pub commit: f64,
    pub retire: f64,
    /// Phase B: the columns up again, the tables' arguments, the codewords
    /// recomputed, the opening.
    pub upload_b: f64,
    pub argue: f64,
    pub encode: f64,
    pub open: f64,
    /// Host bytes kept of the group's trees between the phases.
    pub tree_bytes: usize,
    /// The device ledger's peak promise through the group's argue — the
    /// argue's own and the next group's columns uploaded beside it. Bytes; 0
    /// without a device.
    pub argue_reserved: u64,
    /// The ledger's promise just before the group's argue (its own columns
    /// on the card), and the ledger's peak through its revive and openings
    /// (the next group's columns, uploaded beside the argue, included), with
    /// what the revive itself added (the openings' room). Bytes; 0 without a
    /// device. Phase B's overlap reads them: group g's openings beside group
    /// g + 1's argue would hold about `open_reserved[g] + argue_reserved[g+1]
    /// − argue_base[g+1]`.
    pub argue_base: u64,
    pub open_reserved: u64,
    pub open_room: u64,
    /// Phase A's ledger: the promise just before the group's commit (its own
    /// columns on the card, and the previous group's while it packs) and the
    /// peak through the commit (the next group's columns, uploaded beside it,
    /// included). Bytes; 0 without a device.
    pub commit_base: u64,
    pub commit_reserved: u64,
    /// The group's columns were to go up beside the previous group's commit
    /// and the ledger refused them; they went up after it.
    pub ahead_refused: bool,
    /// When the group's commit ended, seconds since phase A started.
    pub committed_at: f64,
    /// Phase A: the group's tables packed narrow after the commit
    /// ([`Narrowing`]) — the seconds the committer waited for the pack, the
    /// packer's own seconds (beside the next group's upload), how many tables,
    /// their cells and the packed bytes.
    pub pack: f64,
    pub pack_busy: f64,
    pub packed_tables: usize,
    pub packed_cells: usize,
    pub packed_bytes: usize,
}

/// How a block holds its committed columns between phase A and phase B.
///
/// The proof is the same byte for byte under every choice: a narrow table
/// keeps its raw words, each column at the bytes its largest needs (1, 2, 4 or
/// 8), and the card widens them back into the store phase B reads.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Narrowing {
    /// As field elements, eight bytes a cell.
    #[default]
    Wide,
    /// Packed on the card from the group's columns there, after the group's
    /// commit; only the packed bytes come back. A table of fewer than
    /// `min_cells` cells, or one the card did not hold, stays wide.
    Card { min_cells: usize },
    /// Packed on the host after the group's commit, every table: a test's, or
    /// a host without a card. The same packed bytes as [`Self::Card`].
    Host,
}

impl Narrowing {
    /// Production: on the card, every table of [`NARROW_MIN_CELLS`] or more.
    pub const CARD: Self = Self::Card {
        min_cells: NARROW_MIN_CELLS,
    };
}

/// Production's smallest table packed narrow: under it a table's eight bytes
/// a cell are little, and packing it costs the card two round trips.
pub const NARROW_MIN_CELLS: usize = 1 << 16;

/// Where a block's committed columns are on the host as they move, in bytes:
/// a memory log's terms (the block prover's `LAMBDA_VM_BLOCK_MEMLOG`), read
/// by its sampler while the block proves. It counts and changes nothing.
pub struct BlockMem {
    /// The group phase A is uploading and committing, eight bytes a cell.
    pub committing: AtomicUsize,
    /// The next group, taken and uploaded beside that commit (uploading
    /// ahead), eight bytes a cell.
    pub ahead: AtomicUsize,
    /// Committed groups still held wide while their packers run, until their
    /// packed tables are installed.
    pub packing_wide: AtomicUsize,
    /// What finished packers hold and nothing installed yet.
    pub packed_ready: AtomicUsize,
    /// Committed tables as held: packed, and eight bytes a cell.
    pub held_narrow: AtomicUsize,
    pub held_wide: AtomicUsize,
    /// The groups' kept tree tops.
    pub tree_tops: AtomicUsize,
    /// Held tables' packed bytes handed to a spill store ([`BlockSpill`]):
    /// in its writer queue until written, on disk after, and back in
    /// `held_narrow` once phase B reads them back.
    pub spilled: AtomicUsize,
    mark: Box<dyn Fn(&str) + Send + Sync>,
}

impl BlockMem {
    /// Every term at zero; `mark` is called at each event: a group committed,
    /// a group's packed tables installed, a phase-B group's end.
    pub fn new(mark: impl Fn(&str) + Send + Sync + 'static) -> Self {
        Self {
            committing: AtomicUsize::new(0),
            ahead: AtomicUsize::new(0),
            packing_wide: AtomicUsize::new(0),
            packed_ready: AtomicUsize::new(0),
            held_narrow: AtomicUsize::new(0),
            held_wide: AtomicUsize::new(0),
            tree_tops: AtomicUsize::new(0),
            spilled: AtomicUsize::new(0),
            mark: Box::new(mark),
        }
    }

    fn mark(&self, label: &str) {
        (self.mark)(label);
    }

    /// Counts `tables` as held from here on, packed or wide.
    fn hold<'t, 'a: 't, F, E>(&self, tables: impl IntoIterator<Item = &'t CommittedTable<'a, F, E>>)
    where
        F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
        E: IsField + Send + Sync + 'static,
        FieldElement<F>: AsBytes + Sync + Send,
        FieldElement<E>: AsBytes + Sync + Send,
    {
        for table in tables.into_iter().filter(|t| !t.is_spilled()) {
            match table.narrow() {
                Some(packed) => self.held_narrow.fetch_add(packed.data().len(), Relaxed),
                None => self.held_wide.fetch_add(wide_bytes(table), Relaxed),
            };
        }
    }

    /// Stops counting `tables` as held ([`Self::hold`]'s inverse).
    fn release<'t, 'a: 't, F, E>(
        &self,
        tables: impl IntoIterator<Item = &'t CommittedTable<'a, F, E>>,
    ) where
        F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
        E: IsField + Send + Sync + 'static,
        FieldElement<F>: AsBytes + Sync + Send,
        FieldElement<E>: AsBytes + Sync + Send,
    {
        // A spilled table's bytes left the count when they were spilled, or
        // when phase B let them go after its group.
        for table in tables.into_iter().filter(|t| !t.is_spilled()) {
            match table.narrow() {
                Some(packed) => self.held_narrow.fetch_sub(packed.data().len(), Relaxed),
                None => self.held_wide.fetch_sub(wide_bytes(table), Relaxed),
            };
        }
    }
}

/// Groups phase B reads before a read-back could land: never spilled.
const SPILL_RESIDENT_GROUPS: usize = 2;

/// Whether a committed table of `bytes` packed bytes is to be spilled, given
/// the packed bytes kept so far and the main cells committed so far (this
/// table's included): `(kept, cells, bytes)` ([`BlockSpill::new`]).
pub type SpillWanted = dyn Fn(u64, u64, u64) -> bool + Send + Sync;

/// A block's spill ([`crate::spill`]). Phase A considers each committed
/// group's packed tables as the group is installed: past the first
/// [`SPILL_RESIDENT_GROUPS`] groups, a table the policy wants out and the
/// store's writer queue has room for goes to `store`, so the committer never
/// waits on the writers; any other is parked, and one hand-off
/// ([`Self::hand_off`]) may move parked tables to `store` later in phase A.
/// Phase B reads them back in group order, ahead of their upload, and lets
/// each group's go after its opening.
#[derive(Clone)]
pub struct BlockSpill {
    pub store: Arc<SpillStore>,
    /// The writer queue the store was opened with
    /// ([`crate::spill::SpillOptions::queue_bytes`]).
    pub queue_bytes: u64,
    /// The policy's choice for one table.
    pub wanted: Arc<SpillWanted>,
    /// Packed bytes of the committed tables kept, and main cells committed.
    kept: Arc<std::sync::atomic::AtomicU64>,
    cells: Arc<std::sync::atomic::AtomicU64>,
    /// The read-back's report ([`Prefetch::report`]), once phase B has run.
    pub prefetch: Arc<std::sync::Mutex<Option<String>>>,
    /// Every table parked past the first [`SPILL_RESIDENT_GROUPS`] groups, in
    /// the order it was parked: group order. `parked_more` wakes a hand-off
    /// waiting for the next one, or for phase A's end (`closed`).
    parked: Arc<std::sync::Mutex<Vec<Arc<Parked>>>>,
    parked_more: Arc<std::sync::Condvar>,
    closed: Arc<std::sync::atomic::AtomicBool>,
    /// The hand-off ([`Self::hand_off`]): its thread while it runs, then its
    /// report.
    hand_off: Arc<std::sync::Mutex<HandOff>>,
    /// Set while a hand-off runs: the committer parks what it would spill, so
    /// the store keeps one producer and the committer never waits on it.
    handing: Arc<std::sync::atomic::AtomicBool>,
}

/// A committed table's packed columns parked by the spill: held here until a
/// hand-off ([`BlockSpill::hand_off`]) moves them to the store, and read back
/// from whichever holds them before the group's upload (`restore_group`).
struct Parked {
    len: u64,
    state: std::sync::Mutex<ParkedState>,
}

enum ParkedState {
    Resident(multilinear::narrow::NarrowColumns),
    Spilled(SpilledMain),
    /// Taken and not put back (a store that refused its bytes' shape): phase B
    /// refuses the table.
    Lost,
}

impl Parked {
    /// The store's handle, once handed off.
    fn handle(&self) -> Option<SpilledMain> {
        match &*self.state.lock().unwrap_or_else(|e| e.into_inner()) {
            ParkedState::Spilled(handle) => Some(handle.clone()),
            _ => None,
        }
    }
}

/// Where a committed table's packed columns are while phase A holds them out
/// of the table: in the store, or parked.
#[derive(Clone)]
enum Out {
    Spilled(SpilledMain),
    Parked(Arc<Parked>),
}

impl Out {
    /// The store's handle, when the columns are there.
    fn handle(&self) -> Option<SpilledMain> {
        match self {
            Self::Spilled(handle) => Some(handle.clone()),
            Self::Parked(parked) => parked.handle(),
        }
    }
}

#[derive(Default)]
struct HandOff {
    thread: Option<std::thread::JoinHandle<HandOffReport>>,
    report: Option<HandOffReport>,
    /// Phase A is over: no hand-off starts after it.
    closed: bool,
}

/// What a hand-off did: the bytes it was asked for and handed, the tables, its
/// seconds.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HandOffReport {
    pub asked: u64,
    pub handed: u64,
    pub tables: usize,
    pub secs: f64,
}

impl BlockSpill {
    pub fn new(store: SpillStore, queue_bytes: u64, wanted: Arc<SpillWanted>) -> Self {
        Self {
            store: Arc::new(store),
            queue_bytes,
            wanted,
            kept: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            cells: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            prefetch: Arc::new(std::sync::Mutex::new(None)),
            parked: Arc::new(std::sync::Mutex::new(Vec::new())),
            parked_more: Arc::new(std::sync::Condvar::new()),
            closed: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            hand_off: Arc::new(std::sync::Mutex::new(HandOff::default())),
            handing: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Hands the parked tables to the store, oldest first, until `need` bytes
    /// have gone or phase A ends, through the store's own hand-off
    /// ([`SpillStore::spill`]): the tables parked by then, and those parked
    /// after, as they come (while it runs the committer parks what it would
    /// have spilled). It runs on a thread of its own, so the caller does not
    /// wait on the writers, and phase A's end joins it. Once only, and never
    /// after phase A's end: `false` then (or when the thread cannot start).
    pub fn hand_off(&self, need: u64, mem: Option<Arc<BlockMem>>) -> bool {
        let mut hand_off = self.hand_off.lock().unwrap_or_else(|e| e.into_inner());
        if hand_off.closed || hand_off.thread.is_some() || hand_off.report.is_some() {
            return false;
        }
        let (parked, parked_more, closed, store, handing) = (
            Arc::clone(&self.parked),
            Arc::clone(&self.parked_more),
            Arc::clone(&self.closed),
            Arc::clone(&self.store),
            Arc::clone(&self.handing),
        );
        handing.store(true, std::sync::atomic::Ordering::SeqCst);
        let thread = std::thread::Builder::new()
            .name("block-hand-off".to_string())
            .spawn(move || {
                let started = Instant::now();
                let mut report = HandOffReport {
                    asked: need,
                    ..HandOffReport::default()
                };
                let mut next = 0usize;
                while report.handed < need {
                    // The next parked table, waiting for one while phase A runs.
                    let slot = {
                        let mut list = parked.lock().unwrap_or_else(|e| e.into_inner());
                        loop {
                            if let Some(slot) = list.get(next) {
                                break Some(Arc::clone(slot));
                            }
                            if closed.load(std::sync::atomic::Ordering::SeqCst) {
                                break None;
                            }
                            list = parked_more
                                .wait_timeout(list, std::time::Duration::from_millis(100))
                                .unwrap_or_else(|e| e.into_inner())
                                .0;
                        }
                    };
                    let Some(slot) = slot else {
                        break;
                    };
                    next += 1;
                    let taken = std::mem::replace(
                        &mut *slot.state.lock().unwrap_or_else(|e| e.into_inner()),
                        ParkedState::Lost,
                    );
                    let packed = match taken {
                        ParkedState::Resident(packed) => packed,
                        other => {
                            *slot.state.lock().unwrap_or_else(|e| e.into_inner()) = other;
                            continue;
                        }
                    };
                    // A shape the store refuses stays `Lost`: phase B refuses it.
                    let Some(main) = to_store(packed) else {
                        break;
                    };
                    let mut state = slot.state.lock().unwrap_or_else(|e| e.into_inner());
                    match store.spill(main) {
                        Ok(handle) => {
                            *state = ParkedState::Spilled(handle);
                            report.handed += slot.len;
                            report.tables += 1;
                            if let Some(mem) = &mem {
                                mem.held_narrow.fetch_sub(slot.len as usize, Relaxed);
                                mem.spilled.fetch_add(slot.len as usize, Relaxed);
                            }
                        }
                        // The store failed: what is left stays parked.
                        Err(main) => {
                            if let Some(packed) = from_store(main) {
                                *state = ParkedState::Resident(packed);
                            }
                            break;
                        }
                    }
                }
                handing.store(false, std::sync::atomic::Ordering::SeqCst);
                report.secs = started.elapsed().as_secs_f64();
                report
            });
        match thread {
            Ok(thread) => {
                hand_off.thread = Some(thread);
                true
            }
            Err(_) => {
                self.handing
                    .store(false, std::sync::atomic::Ordering::SeqCst);
                false
            }
        }
    }

    /// Phase A's end: joins a running hand-off, and none starts after this.
    fn close_hand_off(&self) {
        let thread = {
            let mut hand_off = self.hand_off.lock().unwrap_or_else(|e| e.into_inner());
            hand_off.closed = true;
            hand_off.thread.take()
        };
        {
            let _list = self.parked.lock().unwrap_or_else(|e| e.into_inner());
            self.closed.store(true, std::sync::atomic::Ordering::SeqCst);
            self.parked_more.notify_all();
        }
        if let Some(thread) = thread {
            let report = thread
                .join()
                .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
            self.hand_off
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .report = Some(report);
        }
    }

    /// What the hand-off did, once phase A's end joined it; `None` when none
    /// ran.
    pub fn hand_off_report(&self) -> Option<HandOffReport> {
        self.hand_off
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .report
    }

    /// Whether `len` more bytes fit the writers' queue now. The committer is
    /// the store's one producer, so the room it sees stays: the spill that
    /// follows does not wait.
    fn has_room(&self, len: u64) -> bool {
        let stats = self.store.stats();
        let pending = stats.bytes.saturating_sub(stats.bytes_written);
        pending == 0 || pending + len <= self.queue_bytes
    }
}

/// Packed columns as the store's payload ([`NarrowMain`]): the same parts,
/// the bytes moved, never copied. Both types hold the same shape checks, so
/// columns that exist convert.
fn to_store(packed: multilinear::narrow::NarrowColumns) -> Option<NarrowMain> {
    let (rows, widths, data) = packed.into_parts();
    NarrowMain::from_parts(rows, widths, data)
}

/// [`to_store`]'s inverse.
fn from_store(main: NarrowMain) -> Option<multilinear::narrow::NarrowColumns> {
    let (rows, widths, data) = main.into_parts();
    multilinear::narrow::NarrowColumns::from_parts(rows, widths, data)
}

/// Considers group `g`'s narrow tables for the spill, `first` being the
/// first's index; where each table taken out of its group went goes in `out`.
/// The first [`SPILL_RESIDENT_GROUPS`] groups keep theirs in place. Past them,
/// a table the policy wants out, while no hand-off runs, and that the store's
/// writer queue has room for goes to `store`; any other is parked
/// ([`Parked`]): on the host until a hand-off moves it. Every table counts
/// toward the cells committed, and every one not spilled toward the bytes
/// kept.
fn spill_group<F, E>(
    spill: &BlockSpill,
    g: usize,
    first: usize,
    tables: &mut [CommittedTable<'_, F, E>],
    out: &mut [Option<Out>],
    mem: Option<&BlockMem>,
) -> Result<(), MlError>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    for (k, (table, slot)) in tables.iter_mut().zip(out.iter_mut()).enumerate() {
        let table_cells = (table.num_committed_columns() as u64) << table.num_vars();
        let cells = spill.cells.fetch_add(table_cells, Relaxed) + table_cells;
        let Some(len) = table.narrow().map(|packed| packed.data().len()) else {
            continue;
        };
        if g < SPILL_RESIDENT_GROUPS {
            spill.kept.fetch_add(len as u64, Relaxed);
            continue;
        }
        let failed = || MlError::SpillFailed {
            table: first + k,
            reason: "its packed parts did not move to the store and back",
        };
        let packed = table.take_narrow_for_spill().ok_or_else(failed)?;
        let spill_now = (spill.wanted)(spill.kept.load(Relaxed), cells, len as u64)
            && !spill.handing.load(std::sync::atomic::Ordering::SeqCst)
            && spill.has_room(len as u64);
        let packed = if spill_now {
            match spill.store.spill(to_store(packed).ok_or_else(failed)?) {
                Ok(handle) => {
                    if let Some(mem) = mem {
                        mem.held_narrow.fetch_sub(len, Relaxed);
                        mem.spilled.fetch_add(len, Relaxed);
                    }
                    *slot = Some(Out::Spilled(handle));
                    continue;
                }
                Err(main) => from_store(main).ok_or_else(failed)?,
            }
        } else {
            packed
        };
        let parked = Arc::new(Parked {
            len: len as u64,
            state: std::sync::Mutex::new(ParkedState::Resident(packed)),
        });
        spill
            .parked
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(Arc::clone(&parked));
        spill.parked_more.notify_all();
        *slot = Some(Out::Parked(parked));
        spill.kept.fetch_add(len as u64, Relaxed);
    }
    Ok(())
}

/// Brings `tables`' columns back before their upload, `first` being the
/// first's index: a parked table's from the host, or from the store when a
/// hand-off moved them; a spilled table's from the read-back ahead when it
/// has them, or read here. Refused when a read fails or does not match what
/// was written.
fn restore_group<F, E>(
    first: usize,
    tables: &mut [CommittedTable<'_, F, E>],
    out: &mut [Option<Out>],
    prefetch: Option<&Prefetch>,
    mem: Option<&BlockMem>,
) -> Result<(), MlError>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    for (k, (table, slot)) in tables.iter_mut().zip(out.iter_mut()).enumerate() {
        let Some(taken) = slot.take() else {
            continue;
        };
        let index = first + k;
        let handle = match taken {
            Out::Spilled(handle) => handle,
            Out::Parked(parked) => {
                let state = std::mem::replace(
                    &mut *parked.state.lock().unwrap_or_else(|e| e.into_inner()),
                    ParkedState::Lost,
                );
                match state {
                    ParkedState::Resident(packed) => {
                        if !table.restore_narrow(packed) {
                            return Err(MlError::SpillFailed {
                                table: index,
                                reason: "the parked columns do not have the table's shape",
                            });
                        }
                        continue;
                    }
                    ParkedState::Spilled(handle) => handle,
                    // The table stays out, and the group's check refuses it.
                    ParkedState::Lost => continue,
                }
            }
        };
        let read = match prefetch.and_then(|p| p.take(ReadPhase::Fused, index)) {
            Some(read) => read,
            None => handle.into_narrow(),
        };
        let main = read.map_err(|e| MlError::SpillFailed {
            table: index,
            reason: match e {
                crate::spill::SpillError::Mismatch => {
                    "the bytes read back are not the ones written"
                }
                crate::spill::SpillError::Io(_) => "the read failed",
            },
        })?;
        let len = main.data().len();
        if !from_store(main).is_some_and(|packed| table.restore_narrow(packed)) {
            return Err(MlError::SpillFailed {
                table: index,
                reason: "the bytes read back do not have the table's shape",
            });
        }
        if let Some(mem) = mem {
            mem.spilled.fetch_sub(len, Relaxed);
            mem.held_narrow.fetch_add(len, Relaxed);
        }
    }
    Ok(())
}

/// A table's committed columns as held on the host: packed, or eight bytes a
/// cell.
fn held_bytes<F, E>(table: &CommittedTable<'_, F, E>) -> usize
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    table
        .narrow()
        .map_or_else(|| wide_bytes(table), |packed| packed.data().len())
}

/// A group's columns put on the card for its commit: a group of wide tables
/// as their field elements, one with a table held narrow (built packed) as
/// [`upload_group`] puts it — packed, and widened on the card.
fn upload_for_commit<F, E>(group: &[CommittedTable<'_, F, E>]) -> Store
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    if group.iter().any(|t| t.narrow().is_some()) {
        return upload_group(group);
    }
    let columns: Vec<&Mle<F>> = group.iter().flat_map(|t| t.columns()).collect();
    multilinear::gpu::upload_columns(&columns).map(Arc::new)
}

/// A table's committed columns at eight bytes a cell.
fn wide_bytes<F, E>(table: &CommittedTable<'_, F, E>) -> usize
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    (table.num_committed_columns() << table.num_vars()) * core::mem::size_of::<FieldElement<F>>()
}

/// A group's PREPARED opening in a block: one commitment, which both sides
/// derive from the program, over the leading preprocessed columns of the
/// group's prepared tables, opened on the group's fork after the group's own
/// opening — each table's columns at that table's point.
///
/// It is the epoch's prepared opening ([`crate::multilinear_table::Prepared`])
/// per group: its roots are DERIVED by the verifier and absorbed after the group
/// roots, before `z`; the columns stay in the group's stack too, and the opening
/// proves the derived commitment takes the values each table's own argument
/// settled on, at that table's point. One stack a group pays one chain for all
/// of its prepared tables.
pub struct BlockPrepared<'a, F, H>
where
    F: IsFFTField + IsPrimeField + 'static,
    H: WhirHash,
    FieldElement<F>: AsBytes + Sync + Send,
{
    /// The group whose fork opens it.
    pub group: usize,
    /// The tables it settles, in stack order: each table's position in the
    /// proof's table order and the length of its leading prefix.
    pub tables: Vec<(usize, usize)>,
    pub commitment: &'a StackedCommitment<F, H>,
    /// The tables' prefixes, concatenated in `tables`' order.
    pub columns: &'a [&'a Mle<F>],
}

/// What the verifier settles a [`BlockPrepared`] opening against.
pub struct BlockPreparedCheck<'a, F>
where
    F: IsFFTField + IsPrimeField + 'static,
{
    pub group: usize,
    /// As [`BlockPrepared::tables`]: the prefixes `check_preprocessed` skips.
    pub tables: Vec<(usize, usize)>,
    /// Derived from the program by the verifier, never read from the proof.
    pub roots: &'a [Commitment],
    pub layout: &'a StackedLayout,
    pub domain: &'a Domain<F>,
}

/// The prepared openings' shape, checked the same way by both sides: at most one
/// per group, in group order; each settles distinct tables of its own group in
/// increasing order, each prefix at least one column and at most `max_prefix` of
/// the table's; and the stack holds exactly those columns (`columns`, when the
/// caller knows them).
/// One prepared opening's shape: its group, its `(table, prefix)` list, and the
/// stack's column count when the caller knows it.
type PreparedEntry<'a> = (usize, &'a [(usize, usize)], Option<usize>);

fn prepared_shape(
    entries: &[PreparedEntry<'_>],
    starts: &[usize],
    num_tables: usize,
    max_prefix: &dyn Fn(usize) -> usize,
) -> Result<(), MlError> {
    let mut last_group: Option<usize> = None;
    for &(group, tables, columns) in entries {
        let range = starts
            .get(group)
            .copied()
            .zip(starts.get(group + 1).copied().or(Some(num_tables)))
            .ok_or(MlError::UnknownPolynomial {
                index: group,
                len: starts.len(),
            })?;
        if last_group.is_some_and(|l| group <= l) || tables.is_empty() {
            return Err(MlError::UnknownPolynomial {
                index: group,
                len: starts.len(),
            });
        }
        last_group = Some(group);
        let mut last: Option<usize> = None;
        for &(table, n) in tables {
            if table < range.0 || table >= range.1 || last.is_some_and(|l| table <= l) {
                return Err(MlError::UnknownPolynomial {
                    index: table,
                    len: num_tables,
                });
            }
            if n == 0 || n > max_prefix(table) {
                return Err(MlError::QueryCountMismatch {
                    expected: max_prefix(table),
                    got: n,
                });
            }
            last = Some(table);
        }
        let total: usize = tables.iter().map(|&(_, n)| n).sum();
        if columns.is_some_and(|c| c != total) {
            return Err(MlError::QueryCountMismatch {
                expected: total,
                got: columns.unwrap_or(0),
            });
        }
    }
    Ok(())
}

/// A group's prepared claims: per table of `tables`, its prefix's columns at the
/// table's point (one point per column) and the values its argument settled on.
/// `firsts` are the group's tables' first columns, `start` its first table.
#[allow(clippy::type_complexity)]
fn prepared_claims<E: IsField>(
    tables: &[(usize, usize)],
    start: usize,
    firsts: &[usize],
    points: &[Vec<FieldElement<E>>],
    values: &[FieldElement<E>],
) -> (Vec<Vec<FieldElement<E>>>, Vec<FieldElement<E>>) {
    let mut at_points = Vec::new();
    let mut at_values = Vec::new();
    for &(table, n) in tables {
        let first = firsts[table - start];
        at_points.extend_from_slice(&points[first..first + n]);
        at_values.extend_from_slice(&values[first..first + n]);
    }
    (at_points, at_values)
}

/// How a test makes the prover open a group's prepared stack wrongly, for the
/// verifier to refuse.
#[doc(hidden)]
#[derive(Clone, Debug, Default)]
pub struct PreparedDeviation {
    /// Which opening, by its index in the `prepared` list.
    pub open: usize,
    /// `(table, other)`: open `table`'s block at `other`'s point (both positions
    /// in the proof's table order, in the opening's group).
    pub at_point_of: Vec<(usize, usize)>,
    /// Claim every value as the committed column evaluated at its claimed point,
    /// so the opening is internally consistent and only the verifier's binding
    /// of it to the tables (and to its derived roots) refuses it.
    pub consistent: bool,
}

/// How a test makes the prover argue a group wrongly under the batched argue,
/// or argue every group on the host, for the card-vs-host byte gate. The
/// production prover takes [`ArgueDeviation::default`]: no fault, the card.
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct ArgueDeviation {
    /// The group whose batched argue takes `faults`.
    pub group: Option<usize>,
    pub faults: ProverFaults,
    /// Where every group's batched argue runs.
    pub at: Where,
}

impl Default for ArgueDeviation {
    fn default() -> Self {
        Self {
            group: None,
            faults: ProverFaults::default(),
            at: Where::Device,
        }
    }
}

/// What phase B proves: the [`MultiProof`] (its `tables` empty under the
/// batched argue), the groups' batched argues (one per group under the batched
/// argue, none under the per-table one), the prepared openings in the
/// `prepared` list's order, and every group's stamps.
pub type BlockProved<F, E> = (
    MultiProof<F, E>,
    Vec<BatchedArgue<E>>,
    Vec<StackedProof<F, E>>,
    Vec<GroupStamps>,
);

/// The block after phase A: every table, the groups' roots and the tops of
/// their trees. The codewords are gone.
pub struct BlockCommitted<'a, F, E>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    tables: Vec<CommittedTable<'a, F, E>>,
    sizes: Vec<usize>,
    groups: Vec<RetiredStack<F>>,
    roots: Vec<Commitment>,
    stamps: Vec<GroupStamps>,
    /// A memory log's terms, counted through phase B as well.
    mem: Option<Arc<BlockMem>>,
    /// The spill, when one is on, and each table's slot in it.
    spill: Option<BlockSpill>,
    /// Each table taken out of its group in phase A: spilled, or parked.
    spilled: Vec<Option<Out>>,
}

impl<'a, F, E> BlockCommitted<'a, F, E>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    /// Phase A: commits each group in turn and lets its codewords go, keeping
    /// each tree but its bottom `drop_levels` levels on the host.
    ///
    /// `sizes` must be [`block_groups`]'s over the same shapes and the
    /// config's stack cap, or the verifier rebuilds other stacks.
    pub fn commit<H: WhirHash>(
        tables: Vec<CommittedTable<'a, F, E>>,
        sizes: &[usize],
        config: &ChainConfig,
        drop_levels: multilinear::whir_commit::TreeDrop,
        narrow: Narrowing,
        upload_ahead: bool,
    ) -> Result<Self, MlError> {
        if sizes.iter().sum::<usize>() != tables.len() {
            return Err(MlError::QueryCountMismatch {
                expected: tables.len(),
                got: sizes.iter().sum(),
            });
        }
        let mut tables = tables.into_iter();
        let groups = sizes
            .iter()
            .map(|&size| tables.by_ref().take(size).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        Self::commit_streamed::<H>(groups, sizes, config, drop_levels, narrow, upload_ahead)
    }

    /// [`Self::commit`] over groups handed over one at a time, in group order —
    /// by a producer still preparing the later ones while the card commits
    /// the earlier. Each group's wait for its tables is stamped (`wait_a`). A
    /// producer that stops early leaves fewer groups than `sizes` names, which
    /// is refused.
    pub fn commit_streamed<H: WhirHash>(
        groups: impl IntoIterator<Item = Vec<CommittedTable<'a, F, E>>>,
        sizes: &[usize],
        config: &ChainConfig,
        drop_levels: multilinear::whir_commit::TreeDrop,
        narrow: Narrowing,
        upload_ahead: bool,
    ) -> Result<Self, MlError> {
        let block = Self::commit_groups::<H>(
            groups.into_iter().take(sizes.len()),
            config,
            drop_levels,
            narrow,
            upload_ahead,
        )?;
        if block.sizes != sizes {
            return Err(MlError::QueryCountMismatch {
                expected: sizes.len(),
                got: block.sizes.len(),
            });
        }
        Ok(block)
    }

    /// Phase A over whatever groups arrive, in arrival order, until the
    /// producer stops: the groups (and so their sizes) are the prover's, and the
    /// statement carries them. Each group's wait for its tables is stamped, and
    /// each group is held as `narrow` says once committed.
    ///
    /// `upload_ahead`: the next group is taken, and its columns put on the card,
    /// while this group commits (on a thread of its own) — once the commit has
    /// asked the card for its room, so the upload only takes what is left; a
    /// store the ledger refuses is uploaded after the commit, as without it.
    /// The commits, their order and their bytes are the same either way.
    pub fn commit_groups<H: WhirHash>(
        groups: impl IntoIterator<Item = Vec<CommittedTable<'a, F, E>>>,
        config: &ChainConfig,
        drop_levels: multilinear::whir_commit::TreeDrop,
        narrow: Narrowing,
        upload_ahead: bool,
    ) -> Result<Self, MlError> {
        Self::commit_groups_logged::<H>(
            groups,
            config,
            drop_levels,
            narrow,
            upload_ahead,
            None,
            None,
        )
    }

    /// [`Self::commit_groups`], counting where the columns are in `mem` (a
    /// memory log, through phase B too) as they move. The commits are the same.
    pub fn commit_groups_logged<H: WhirHash>(
        groups: impl IntoIterator<Item = Vec<CommittedTable<'a, F, E>>>,
        config: &ChainConfig,
        drop_levels: multilinear::whir_commit::TreeDrop,
        narrow: Narrowing,
        upload_ahead: bool,
        mem: Option<Arc<BlockMem>>,
        spill: Option<BlockSpill>,
    ) -> Result<Self, MlError> {
        let mut tables = Vec::new();
        let mut spilled: Vec<Option<Out>> = Vec::new();
        let mut sizes = Vec::new();
        let mut retired_groups = Vec::new();
        let mut roots = Vec::new();
        let mut stamps = Vec::new();
        let mut incoming = groups.into_iter();
        let started = Instant::now();
        // The previous group's pack on the card, running beside this group's
        // upload (or, uploading ahead, beside this group's commit).
        let mut packing: Option<Packing> = None;
        // The next group, taken and uploaded beside this group's commit.
        let mut ahead: Option<Ahead<'a, F, E>> = None;
        let mut exhausted = false;
        loop {
            let (mut group, mut stamp, store, wide) = match ahead.take() {
                Some(next) => {
                    // A memory log: the group taken ahead is now the one in
                    // its commit.
                    if let Some(mem) = &mem {
                        mem.ahead.fetch_sub(next.wide, Relaxed);
                        mem.committing.fetch_add(next.wide, Relaxed);
                    }
                    let mut stamp = GroupStamps {
                        tables: next.group.len(),
                        wait_a: next.wait_paid,
                        upload_a: next.upload,
                        upload_paid: next.upload_paid,
                        ..Default::default()
                    };
                    let store = match next.store {
                        Some(store) => Some(store),
                        None => {
                            // Refused beside the commit (or no card): up now,
                            // with the commit's room given back.
                            stamp.ahead_refused = multilinear::gpu::reserve_budget() > 0;
                            let t = Instant::now();
                            let store = upload_for_commit(&next.group);
                            let secs = t.elapsed().as_secs_f64();
                            stamp.upload_a += secs;
                            stamp.upload_paid += secs;
                            store
                        }
                    };
                    (next.group, stamp, store, next.wide)
                }
                None => {
                    if exhausted {
                        break;
                    }
                    let waited = Instant::now();
                    let Some(group) = incoming.next() else {
                        break;
                    };
                    let mut stamp = GroupStamps {
                        tables: group.len(),
                        wait_a: waited.elapsed().as_secs_f64(),
                        ..Default::default()
                    };
                    let wide = mem.as_ref().map_or(0, |mem| {
                        let held = group.iter().map(held_bytes).sum();
                        mem.committing.fetch_add(held, Relaxed);
                        held
                    });
                    let t = Instant::now();
                    // Held in an `Arc` like phase B's: once the tree tops are
                    // home it goes to the group's packer, or is dropped.
                    let store = upload_for_commit(&group);
                    stamp.upload_a = t.elapsed().as_secs_f64();
                    stamp.upload_paid = stamp.upload_a;
                    (group, stamp, store, wide)
                }
            };
            let size = group.len();
            if size == 0 {
                return Err(MlError::QueryCountMismatch {
                    expected: 1,
                    got: 0,
                });
            }
            sizes.push(size);
            if !upload_ahead && let Some(packed) = packing.take() {
                install_and_spill(
                    packed,
                    &mut tables,
                    &mut spilled,
                    &mut stamps,
                    &sizes,
                    mem.as_deref(),
                    spill.as_ref(),
                )?;
            }
            let shapes: Vec<(usize, usize)> = group
                .iter()
                .map(|t| (t.num_committed_columns(), t.num_vars()))
                .collect();
            stamp.cells = shapes.iter().map(|&(w, n)| w << n).sum();
            let layout = global_layout(&shapes, config.format.stack)?;
            stamp.polys = layout.num_polys();
            stamp.commit_base = multilinear::gpu::ledger_reserved();
            multilinear::gpu::reset_reserved_window();
            // Handles, not field elements: a table held narrow is read on the
            // host only by a host fallback.
            let handles = group_columns(&group);
            let columns: Vec<&ColumnOf<'_, _>> = handles.iter().collect();
            let resident = store.as_ref().map(|store| (&**store, 0));
            let committed = if upload_ahead {
                let (committed, next) = std::thread::scope(|scope| {
                    let (room_tx, room_rx) = std::sync::mpsc::channel::<()>();
                    let columns = &columns;
                    let committer = scope.spawn(move || {
                        let signal = move || {
                            let _ = room_tx.send(());
                        };
                        let committed = commit_and_retire::<F, H, _>(
                            layout,
                            columns,
                            resident,
                            config,
                            drop_levels,
                            &signal,
                        );
                        (committed, Instant::now())
                    });
                    // Beside the commit: the next group, then its columns once
                    // the commit has asked for its room (or ended).
                    let next = incoming.next();
                    let arrived = Instant::now();
                    let next = next.map(|group| {
                        let wide = mem.as_ref().map_or(0, |mem| {
                            let held = group.iter().map(held_bytes).sum();
                            mem.ahead.fetch_add(held, Relaxed);
                            held
                        });
                        let _ = room_rx.recv();
                        let t = Instant::now();
                        let store = upload_for_commit(&group);
                        (
                            group,
                            store,
                            wide,
                            t.elapsed().as_secs_f64(),
                            Instant::now(),
                        )
                    });
                    let (committed, commit_end) = committer
                        .join()
                        .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
                    let next = next.map(|(group, store, wide, upload, uploaded)| {
                        // What the committer waited for once its commit ended:
                        // the group's arrival, then the rest of its upload —
                        // never more than the upload (a commit that ended before
                        // the upload began paid all of it, and no more).
                        let ready = arrived.max(commit_end);
                        Ahead {
                            group,
                            store,
                            wide,
                            upload,
                            wait_paid: arrived.saturating_duration_since(commit_end).as_secs_f64(),
                            upload_paid: uploaded
                                .saturating_duration_since(ready)
                                .as_secs_f64()
                                .min(upload),
                        }
                    });
                    (committed, next)
                });
                exhausted = next.is_none();
                ahead = next;
                committed
            } else {
                commit_and_retire::<F, H, _>(
                    layout,
                    &columns,
                    resident,
                    config,
                    drop_levels,
                    &|| {},
                )
            };
            let (group_roots, retired, commit, retire) = committed?;
            drop(columns);
            drop(handles);
            roots.extend(group_roots);
            stamp.commit = commit;
            stamp.retire = retire;
            stamp.commit_reserved = multilinear::gpu::reserved_window_peak();
            stamp.tree_bytes = retired.tree_bytes();
            stamp.committed_at = started.elapsed().as_secs_f64();
            if upload_ahead && let Some(packed) = packing.take() {
                install_and_spill(
                    packed,
                    &mut tables,
                    &mut spilled,
                    &mut stamps,
                    &sizes,
                    mem.as_deref(),
                    spill.as_ref(),
                )?;
            }
            // Which tables were committed wide (a packer's, from here on).
            let was_wide: Vec<bool> = group.iter().map(|t| t.narrow().is_none()).collect();
            packing =
                narrow_group(&mut group, store, narrow, &mut stamp, mem.clone()).map(|handle| {
                    Packing {
                        first_table: tables.len(),
                        group: stamps.len(),
                        wide: group
                            .iter()
                            .zip(&was_wide)
                            .filter(|&(_, &w)| w)
                            .map(|(t, _)| wide_bytes(t))
                            .sum(),
                        was_wide: was_wide.clone(),
                        handle,
                    }
                });
            if let Some(mem) = &mem {
                mem.committing.fetch_sub(wide, Relaxed);
                mem.tree_tops.fetch_add(stamp.tree_bytes, Relaxed);
                match &packing {
                    Some(packed) => {
                        mem.packing_wide.fetch_add(packed.wide, Relaxed);
                        // The tables committed narrow are held as they are.
                        mem.hold(
                            group
                                .iter()
                                .zip(&was_wide)
                                .filter(|&(_, &w)| !w)
                                .map(|(t, _)| t),
                        );
                    }
                    None => mem.hold(group.iter()),
                }
                mem.mark(&format!("group {} committed", stamps.len()));
            }
            retired_groups.push(retired);
            stamps.push(stamp);
            let first = tables.len();
            tables.extend(group);
            spilled.resize(tables.len(), None);
            // A group no packer took is held as it is from here on.
            if packing.is_none()
                && let Some(spill) = &spill
            {
                spill_group(
                    spill,
                    stamps.len() - 1,
                    first,
                    &mut tables[first..],
                    &mut spilled[first..],
                    mem.as_deref(),
                )?;
            }
        }
        if let Some(packed) = packing.take() {
            install_and_spill(
                packed,
                &mut tables,
                &mut spilled,
                &mut stamps,
                &sizes,
                mem.as_deref(),
                spill.as_ref(),
            )?;
        }
        // Phase A's end: a hand-off still writing is joined, and none starts
        // after this, so phase B's read-back plans every handle.
        if let Some(spill) = &spill {
            spill.close_hand_off();
        }
        Ok(Self {
            tables,
            sizes,
            groups: retired_groups,
            roots,
            stamps,
            mem,
            spill,
            spilled,
        })
    }

    /// Every group's roots, in group order — what the transcript absorbs.
    pub fn roots(&self) -> &[Commitment] {
        &self.roots
    }

    pub fn sizes(&self) -> &[usize] {
        &self.sizes
    }

    pub fn tables(&self) -> &[CommittedTable<'a, F, E>] {
        &self.tables
    }

    /// Phase A's stamps, one per group.
    pub fn stamps(&self) -> &[GroupStamps] {
        &self.stamps
    }

    /// A test's fault: the first narrow table's width map made wrong
    /// ([`multilinear::narrow::NarrowColumns::fault_width_map`]), so phase B
    /// widens other words than were committed. Returns whether a table was
    /// there to break.
    #[doc(hidden)]
    pub fn fault_narrow_width_map(&mut self) -> bool {
        self.tables
            .iter_mut()
            .filter_map(|table| table.narrow_mut())
            .any(|packed| packed.fault_width_map())
    }

    /// A test's fault: the slot of the first table whose columns are in the
    /// store (spilled, or parked and handed off) lost, so its columns never
    /// come back. Returns whether such a table was there to lose.
    #[doc(hidden)]
    pub fn fault_lose_spilled_slot(&mut self) -> bool {
        self.spilled
            .iter_mut()
            .find(|slot| slot.as_ref().and_then(Out::handle).is_some())
            .map(Option::take)
            .is_some()
    }

    /// A test's fault: one byte of the first spilled table flipped on disk,
    /// once the writers are done, so phase B reads back other bytes than were
    /// written. Returns whether a written table was there to break; `false`
    /// outside test builds.
    #[doc(hidden)]
    pub fn fault_spilled_byte(&mut self) -> bool {
        #[cfg(any(test, feature = "test-utils"))]
        if let Some(spill) = &self.spill {
            spill.store.flush();
            if let Some(handle) = self.spilled.iter().flatten().find_map(Out::handle) {
                return spill.store.corrupt_on_disk(&handle, 0).unwrap_or(false);
            }
        }
        false
    }
}

/// The next group, taken and uploaded beside the current group's commit
/// ([`BlockCommitted::commit_groups`]'s `upload_ahead`): its tables, its store
/// (`None` when the ledger refused it, or there is no card), the upload's
/// seconds, and what of the wait and of the upload outlasted the commit.
struct Ahead<'a, F, E>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    group: Vec<CommittedTable<'a, F, E>>,
    store: Store,
    /// Its columns at eight bytes a cell (counted only for a memory log).
    wide: usize,
    upload: f64,
    wait_paid: f64,
    upload_paid: f64,
}

/// A group committed and retired: its roots, its retired stack, and the
/// commit's and the retire's seconds. `on_room` as
/// [`StackedCommitment::commit_signalled`].
fn commit_and_retire<F, H, C>(
    layout: StackedLayout,
    columns: &[&C],
    resident: Option<(&multilinear::gpu::ResidentColumns, usize)>,
    config: &ChainConfig,
    drop_levels: multilinear::whir_commit::TreeDrop,
    on_room: &dyn Fn(),
) -> Result<(Vec<Commitment>, RetiredStack<F>, f64, f64), MlError>
where
    F: IsFFTField + IsPrimeField + Send + Sync + 'static,
    H: WhirHash,
    C: multilinear::narrow::HostColumn<F>,
    FieldElement<F>: AsBytes + Sync + Send,
{
    let t = Instant::now();
    let stacked = multilinear::stacked_eval::with_pairs_off_pool(commit_pairs_off_pool(), || {
        StackedCommitment::<F, H>::commit_signalled(
            layout,
            columns,
            resident.map(|(store, first)| (store, ColumnsAt::From(first))),
            config,
            on_room,
        )
    })?;
    let roots = stacked.roots();
    let commit = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let retired = stacked.retire(drop_levels, config)?;
    Ok((roots, retired, commit, t.elapsed().as_secs_f64()))
}

/// `LAMBDA_VM_BLOCK_COMMIT_OFF_POOL`: phase A commits each group's pairs of
/// polynomials on the committer's own threads (unset or anything but `0`, the
/// default), or on the global rayon pool (`0`, the opt-out). The roots are the
/// same either way. The committer is not a pool thread, so on the pool its pair
/// waits in the injector behind every job the finish's generators queue
/// (rayon-core's `find_work`: own deque, then stealing, then injected jobs),
/// and the card waits with it: at the median block one commit took 2.7–2.9 s
/// against 0.68 s off the pool (ULTRA 096). Read once.
pub fn commit_pairs_off_pool() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        commit_pairs_off_pool_from(
            std::env::var("LAMBDA_VM_BLOCK_COMMIT_OFF_POOL")
                .ok()
                .as_deref(),
        )
    })
}

/// [`commit_pairs_off_pool`]'s reading of its variable.
fn commit_pairs_off_pool_from(value: Option<&str>) -> bool {
    value.is_none_or(|v| v.trim() != "0")
}

/// What a packer hands back: one packed table or none per table of its group,
/// and its own seconds.
type Packed = (Vec<Option<multilinear::narrow::NarrowColumns>>, f64);

/// A group's tables being packed on the card, on a thread of their own:
/// where the group's tables start in the block, its stamp, and the packer.
struct Packing {
    first_table: usize,
    group: usize,
    /// The group's columns committed wide, at eight bytes a cell, and which
    /// tables those are (counted only for a memory log).
    wide: usize,
    was_wide: Vec<bool>,
    handle: std::thread::JoinHandle<Packed>,
}

impl Packing {
    /// Waits for the packer and holds each table it packed narrow from here
    /// on. A packer that failed leaves its tables wide.
    fn install<F, E>(
        self,
        tables: &mut [CommittedTable<'_, F, E>],
        stamps: &mut [GroupStamps],
        mem: Option<&BlockMem>,
    ) where
        F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
        E: IsField + Send + Sync + 'static,
        FieldElement<F>: AsBytes + Sync + Send,
        FieldElement<E>: AsBytes + Sync + Send,
    {
        let t = Instant::now();
        let joined = self.handle.join();
        let stamp = &mut stamps[self.group];
        stamp.pack += t.elapsed().as_secs_f64();
        let installed = match joined {
            Ok((packed, busy)) => {
                stamp.pack_busy = busy;
                let ready: usize = packed.iter().flatten().map(|p| p.data().len()).sum();
                for (table, packed) in tables[self.first_table..].iter_mut().zip(packed) {
                    let Some(packed) = packed else {
                        continue;
                    };
                    let bytes = packed.data().len();
                    if table.install_narrow(packed) {
                        stamp.packed_tables += 1;
                        stamp.packed_cells += table.num_committed_columns() << table.num_vars();
                        stamp.packed_bytes += bytes;
                    }
                }
                ready
            }
            Err(_) => 0,
        };
        if let Some(mem) = mem {
            mem.packed_ready.fetch_sub(installed, Relaxed);
            mem.packing_wide.fetch_sub(self.wide, Relaxed);
            mem.hold(
                tables[self.first_table..]
                    .iter()
                    .zip(&self.was_wide)
                    .filter(|&(_, &w)| w)
                    .map(|(t, _)| t),
            );
            mem.mark(&format!("group {} installed", self.group));
        }
    }
}

/// Installs a packer's tables ([`Packing::install`]), then hands its group to
/// the spill when one is on.
fn install_and_spill<F, E>(
    packed: Packing,
    tables: &mut [CommittedTable<'_, F, E>],
    spilled: &mut [Option<Out>],
    stamps: &mut [GroupStamps],
    sizes: &[usize],
    mem: Option<&BlockMem>,
    spill: Option<&BlockSpill>,
) -> Result<(), MlError>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    let (g, first) = (packed.group, packed.first_table);
    packed.install(tables, stamps, mem);
    let Some(spill) = spill else {
        return Ok(());
    };
    let end = first + sizes[g];
    spill_group(
        spill,
        g,
        first,
        &mut tables[first..end],
        &mut spilled[first..end],
        mem,
    )
}

/// Holds a just-committed group's tables as `narrow` says. On the card, each
/// table of at least `min_cells` cells is packed from its run in `store` by a
/// thread of its own, returned for the caller to [`Packing::install`] once the
/// next group's columns are up: the committer is phase A's critical path, and
/// the pack's download overlaps that upload. On the host, every table is packed
/// here. A table that cannot be packed stays as it is.
fn narrow_group<F, E>(
    group: &mut [CommittedTable<'_, F, E>],
    store: Store,
    narrow: Narrowing,
    stamp: &mut GroupStamps,
    mem: Option<Arc<BlockMem>>,
) -> Option<std::thread::JoinHandle<Packed>>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    match (narrow, store) {
        (Narrowing::Wide, _) | (Narrowing::Card { .. }, None) => None,
        (Narrowing::Card { min_cells }, Some(store)) => {
            // Each table's run in the store, if it is to be packed.
            let mut first = 0usize;
            let runs: Vec<Option<(usize, usize)>> = group
                .iter()
                .map(|table| {
                    let width = table.num_committed_columns();
                    // A table built packed is held narrow already.
                    let run = (table.narrow().is_none()
                        && (width << table.num_vars()) >= min_cells)
                        .then_some((first, width));
                    first += width;
                    run
                })
                .collect();
            Some(std::thread::spawn(move || {
                let t = Instant::now();
                let packed: Vec<Option<multilinear::narrow::NarrowColumns>> = runs
                    .into_iter()
                    .map(|run| {
                        run.and_then(|(first, width)| {
                            multilinear::gpu::pack_resident(&store, first, width)
                        })
                    })
                    .collect();
                if let Some(mem) = mem {
                    let ready = packed.iter().flatten().map(|p| p.data().len()).sum();
                    mem.packed_ready.fetch_add(ready, Relaxed);
                }
                (packed, t.elapsed().as_secs_f64())
            }))
        }
        (Narrowing::Host, _) => {
            let t = Instant::now();
            for table in group.iter_mut() {
                if table.pack_on_host() {
                    stamp.packed_tables += 1;
                    stamp.packed_cells += table.num_committed_columns() << table.num_vars();
                    stamp.packed_bytes += table.narrow().map_or(0, |packed| packed.data().len());
                }
            }
            stamp.pack = t.elapsed().as_secs_f64();
            stamp.pack_busy = stamp.pack;
            None
        }
    }
}

/// A group's columns on the card, shared by its tables.
type Store = Option<Arc<multilinear::gpu::ResidentColumns>>;

/// Uploads a group's columns, in table order — a narrow table's packed, and
/// widened on the card; `None` when the card declines (every reader then takes
/// the host copy, widening a narrow table's there).
fn upload_group<F, E>(group: &[CommittedTable<'_, F, E>]) -> Store
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    let tables: Vec<multilinear::gpu::TableColumns<'_, F>> =
        group.iter().map(|t| t.upload_view()).collect();
    multilinear::gpu::upload_tables(&tables).map(Arc::new)
}

/// A group's columns in table order, each read on the host only by a path
/// that asks ([`multilinear::narrow::HostColumn`]): the revive and the opening take them so, and
/// the card holds them.
fn group_columns<'t, F, E>(
    group: &'t [CommittedTable<'_, F, E>],
) -> Vec<ColumnOf<'t, multilinear::constraint_argument::TraceData<F, E>>>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    group
        .iter()
        .flat_map(|table| {
            (0..table.num_committed_columns()).map(move |index| ColumnOf {
                columns: table.trace(),
                index,
            })
        })
        .collect()
}

/// The fork a group proves on: `S_post` then the group's index.
fn group_fork<E, T>(transcript: &T, group: usize) -> T
where
    E: IsField + 'static,
    T: crypto::fiat_shamir::is_transcript::IsTranscript<E> + Clone,
{
    let mut fork = transcript.clone();
    fork.append_bytes(&(group as u64).to_le_bytes());
    fork
}

/// Phase B and the proof: the roots block on `transcript`, then each group on
/// its own fork. Consumes the block — each group's host columns and kept tree
/// top are released as its opening ends. Returns the proof and every group's
/// stamps (phase A's, completed by phase B's).
///
/// The proof is a [`MultiProof`] — the roots, a table proof per table in
/// order, an opening per group — read under the forked schedule by
/// [`block_verify`] and by nothing else.
pub fn block_prove<F, E, T, H>(
    committed: BlockCommitted<'_, F, E>,
    config: &ChainConfig,
    transcript: &mut T,
) -> Result<(MultiProof<F, E>, Vec<GroupStamps>), MlError>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    H: WhirHash,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    T: crypto::fiat_shamir::is_transcript::IsTranscript<E>
        + crypto::fiat_shamir::transcript_hash::HasTranscriptHash<Hash = <H as WhirHash>::Transcript>
        + Clone,
{
    // The batched argue's messages are not a `MultiProof`'s: its callers take
    // them from `block_prove_on_forks`.
    if config.format.argue != ArgueFormat::PerTable {
        return Err(MlError::ArgueFormatMismatch);
    }
    let (proof, _, _, stamps) = block_prove_on_forks::<F, E, T, H>(
        committed,
        config,
        transcript,
        &[],
        &[],
        &|g| g,
        &ArgueDeviation::default(),
    )?;
    Ok((proof, stamps))
}

/// [`block_prove`] with the tables' prepared openings, and group `g` proved on
/// the fork of index `fork_of(g)`, under the config's argue format
/// ([`BlockProved`]).
///
/// Only a test proves on a fork map other than the identity, or with an
/// [`ArgueDeviation`] other than the default: it is how a proof whose groups
/// sit on the wrong forks, or argue wrongly, is built, for the verifier to
/// refuse.
#[doc(hidden)]
pub fn block_prove_on_forks<F, E, T, H>(
    committed: BlockCommitted<'_, F, E>,
    config: &ChainConfig,
    transcript: &mut T,
    prepared: &[BlockPrepared<'_, F, H>],
    deviations: &[PreparedDeviation],
    fork_of: &dyn Fn(usize) -> usize,
    argue: &ArgueDeviation,
) -> Result<BlockProved<F, E>, MlError>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    H: WhirHash,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    T: crypto::fiat_shamir::is_transcript::IsTranscript<E>
        + crypto::fiat_shamir::transcript_hash::HasTranscriptHash<Hash = <H as WhirHash>::Transcript>
        + Clone,
{
    block_prove_on_forks_observed::<F, E, T, H>(
        committed,
        config,
        transcript,
        prepared,
        deviations,
        fork_of,
        argue,
        &|_| {},
    )
}

/// One group's share of the proof, the moment its opening is done: what a
/// consumer of the finished groups (the block tree's first leaf) reads before
/// the rest of phase B ends. Every field is a part of the proof as
/// [`block_prove_on_forks`] returns it.
pub struct GroupOpened<'a, F: IsField, E: IsField> {
    pub group: usize,
    /// Every group's root, as the proof carries them.
    pub roots: &'a [Commitment],
    /// The group's batched argue (the batched format), or `None`.
    pub argue: Option<&'a BatchedArgue<E>>,
    /// The group's tables' proofs, in table order (the per-table format;
    /// empty under the batched one).
    pub tables: &'a [TableProof<E>],
    /// The group's opening.
    pub opening: &'a StackedProof<F, E>,
    /// The group's prepared opening, when the group carries one.
    pub prepared: Option<&'a StackedProof<F, E>>,
}

/// [`block_prove_on_forks`], calling `on_group` after each group's opening
/// with that group's share of the proof ([`GroupOpened`]). The observer reads;
/// the proof is the same with any observer.
#[allow(clippy::too_many_arguments)]
#[doc(hidden)]
pub fn block_prove_on_forks_observed<F, E, T, H>(
    committed: BlockCommitted<'_, F, E>,
    config: &ChainConfig,
    transcript: &mut T,
    prepared: &[BlockPrepared<'_, F, H>],
    deviations: &[PreparedDeviation],
    fork_of: &dyn Fn(usize) -> usize,
    argue: &ArgueDeviation,
    on_group: &dyn Fn(GroupOpened<'_, F, E>),
) -> Result<BlockProved<F, E>, MlError>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    H: WhirHash,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    T: crypto::fiat_shamir::is_transcript::IsTranscript<E>
        + crypto::fiat_shamir::transcript_hash::HasTranscriptHash<Hash = <H as WhirHash>::Transcript>
        + Clone,
{
    let BlockCommitted {
        mut tables,
        sizes,
        groups,
        roots,
        mut stamps,
        mem,
        spill,
        mut spilled,
    } = committed;
    let starts: Vec<usize> = sizes
        .iter()
        .scan(0usize, |at, &size| {
            let start = *at;
            *at += size;
            Some(start)
        })
        .collect();
    let entries: Vec<PreparedEntry<'_>> = prepared
        .iter()
        .map(|p| (p.group, p.tables.as_slice(), Some(p.columns.len())))
        .collect();
    prepared_shape(&entries, &starts, tables.len(), &|t| {
        tables[t].num_committed_columns()
    })?;
    let derived: Vec<Commitment> = prepared.iter().flat_map(|p| p.commitment.roots()).collect();
    let (z, alpha, beta) = absorb_roots_and_challenge::<E, T>(transcript, &roots, &derived);

    let mut table_proofs = Vec::with_capacity(tables.len());
    let mut argues = Vec::new();
    let mut openings = Vec::with_capacity(sizes.len());
    let mut prepared_openings = Vec::with_capacity(prepared.len());
    let mut at = 0usize;
    // The next group's columns, uploaded during this group's argument — the
    // card idles through the argument's host glue, and a group's store is a
    // few GiB beside the argument's working set, not beside its codewords.
    let mut pre_uploaded: Option<Store> = None;
    // The spilled tables read back in group order, two groups' bytes ahead of
    // their uploads.
    let prefetch = spill.as_ref().and_then(|_| {
        let reads: Vec<(ReadPhase, usize, SpilledMain)> = spilled
            .iter()
            .enumerate()
            .filter_map(|(t, slot)| {
                slot.as_ref()
                    .and_then(Out::handle)
                    .map(|handle| (ReadPhase::Fused, t, handle))
            })
            .collect();
        let widest = starts
            .iter()
            .zip(&sizes)
            .map(|(&start, &size)| {
                spilled[start..start + size]
                    .iter()
                    .flatten()
                    .filter_map(Out::handle)
                    .map(|handle| handle.len() as u64)
                    .sum::<u64>()
            })
            .max()
            .unwrap_or(0);
        (!reads.is_empty()).then(|| Prefetch::start(reads, 2 * widest.max(1)))
    });
    for (g, (retired, &size)) in groups.into_iter().zip(&sizes).enumerate() {
        let mut fork = group_fork::<E, T>(transcript, fork_of(g));
        let tables_before = table_proofs.len();
        let argues_before = argues.len();
        let prepared_before = prepared_openings.len();
        if spill.is_some() {
            // This group's columns and the next's, back before their uploads
            // (the next one's goes up beside this group's argue).
            let end = (at + size + sizes.get(g + 1).copied().unwrap_or(0)).min(tables.len());
            restore_group(
                at,
                &mut tables[at..end],
                &mut spilled[at..end],
                prefetch.as_ref(),
                mem.as_deref(),
            )?;
            // No reader of these groups meets a table whose columns are still
            // out: one whose slot did not come back is refused here, before
            // its upload.
            if let Some(k) = tables[at..end].iter().position(|t| t.is_spilled()) {
                return Err(MlError::SpillFailed {
                    table: at + k,
                    reason: "its columns are still spilled when its group is read",
                });
            }
        }
        let (head, tail) = tables.split_at_mut(at + size);
        let group = &mut head[at..];
        let next = sizes.get(g + 1).map(|&n| &tail[..n]);

        let t = Instant::now();
        let store = match pre_uploaded.take() {
            Some(store) => store,
            None => upload_group(group),
        };
        if let Some(store) = &store {
            let mut first = 0usize;
            for table in group.iter_mut() {
                table.set_resident(store.clone(), first);
                first += table.num_committed_columns();
            }
        }
        stamps[g].upload_b += t.elapsed().as_secs_f64();

        // The group's tables, one after another, each leaving its columns
        // claimed at its own point — or, batched, all of them in one argue —
        // with the next group's upload beside them.
        let t = Instant::now();
        stamps[g].argue_base = multilinear::gpu::ledger_reserved();
        multilinear::gpu::reset_reserved_window();
        let mut points: Vec<Vec<FieldElement<E>>> = Vec::new();
        let mut values: Vec<FieldElement<E>> = Vec::new();
        let group_ref: &[CommittedTable<'_, F, E>] = group;
        let (argued, next_store, joined) = std::thread::scope(|scope| {
            let uploader = next.map(|next| scope.spawn(move || upload_group(next)));
            let argued = (|| -> Result<(), MlError> {
                match config.format.argue {
                    ArgueFormat::PerTable => {
                        for table in group_ref.iter() {
                            let (proof, point) = prove(table, &z, &alpha, &beta, &mut fork, None)?;
                            for _ in 0..table.num_committed_columns() {
                                points.push(point.clone());
                            }
                            values.extend(proof.constraint.reduce.column_values.iter().cloned());
                            table_proofs.push(proof);
                        }
                    }
                    ArgueFormat::Batched { bin_log_cells } => {
                        let faults = match argue.group {
                            Some(faulty) if faulty == g => argue.faults,
                            _ => ProverFaults::default(),
                        };
                        let (proof, reduced) = batched::prove_argue(
                            group_ref,
                            &z,
                            &alpha,
                            &beta,
                            bin_log_cells,
                            &mut fork,
                            faults,
                            argue.at,
                        )?;
                        for (table, claim) in group_ref.iter().zip(reduced) {
                            for _ in 0..table.num_committed_columns() {
                                points.push(claim.point.clone());
                            }
                            values.extend(claim.column_values);
                        }
                        argues.push(proof);
                    }
                }
                Ok(())
            })();
            let argue_end = Instant::now();
            let next_store = uploader.map(|handle| handle.join());
            (argued, next_store, argue_end.elapsed().as_secs_f64())
        });
        argued?;
        stamps[g].argue = t.elapsed().as_secs_f64() - joined;
        stamps[g].argue_reserved = multilinear::gpu::reserved_window_peak();
        if let Some(next_store) = next_store {
            // The wait for the upload after the argument ended is the next
            // group's upload cost; the rest of it hid behind this argument.
            stamps[g + 1].upload_b += joined;
            pre_uploaded = Some(next_store.map_err(|_| MlError::DeviceFailed {
                stage: "uploading the next group's columns",
            })?);
        }

        let handles = group_columns(group);
        let columns: Vec<&ColumnOf<'_, _>> = handles.iter().collect();
        let tree_tops = retired.tree_bytes();
        let t = Instant::now();
        multilinear::gpu::reset_reserved_window();
        let before_revive = multilinear::gpu::ledger_reserved();
        let stacked = retired.revive::<H, _>(
            &columns,
            store.as_ref().map(|store| (&**store, ColumnsAt::From(0))),
            config,
        )?;
        stamps[g].encode = t.elapsed().as_secs_f64();
        stamps[g].open_room = multilinear::gpu::ledger_reserved().saturating_sub(before_revive);
        let t = Instant::now();
        openings.push(stacked_eval::prove::<F, E, T, H, _>(
            &stacked,
            &columns,
            store.as_ref().map(|store| (&**store, 0)),
            &Claimed::PerColumn(&points),
            &values,
            config,
            &mut fork,
        )?);
        // The group's prepared tables, in table order: each opened at its own
        // point for its own prefix, on this fork.
        let firsts: Vec<usize> = group
            .iter()
            .scan(0usize, |first, table| {
                let at = *first;
                *first += table.num_committed_columns();
                Some(at)
            })
            .collect();
        if let Some(k) = prepared.iter().position(|p| p.group == g) {
            let p = &prepared[k];
            let (mut at_points, mut at_values) =
                prepared_claims(&p.tables, at, &firsts, &points, &values);
            if let Some(d) = deviations.iter().find(|d| d.open == k) {
                // Each block's columns sit at its table's place in the stack.
                let mut column = 0usize;
                for &(table, n) in &p.tables {
                    if let Some(&(_, other)) = d.at_point_of.iter().find(|(t, _)| *t == table) {
                        let local = other.checked_sub(at).filter(|&l| l < firsts.len()).ok_or(
                            MlError::UnknownPolynomial {
                                index: other,
                                len: firsts.len(),
                            },
                        )?;
                        for point in &mut at_points[column..column + n] {
                            *point = points[firsts[local]].clone();
                        }
                    }
                    column += n;
                }
                if d.consistent {
                    at_values = p
                        .columns
                        .iter()
                        .zip(&at_points)
                        .map(|(c, point)| c.evaluate_in(point))
                        .collect::<Result<_, _>>()?;
                }
            }
            prepared_openings.push(stacked_eval::prove::<F, E, T, H, _>(
                p.commitment,
                p.columns,
                None,
                &Claimed::PerColumn(&at_points),
                &at_values,
                config,
                &mut fork,
            )?);
        }
        stamps[g].open = t.elapsed().as_secs_f64();
        stamps[g].open_reserved = multilinear::gpu::reserved_window_peak();
        if let Some(opening) = openings.last() {
            on_group(GroupOpened {
                group: g,
                roots: &roots,
                argue: argues[argues_before..].first(),
                tables: &table_proofs[tables_before..],
                opening,
                prepared: prepared_openings[prepared_before..].first(),
            });
        }
        drop(stacked);
        drop(columns);
        drop(handles);
        for table in group.iter_mut() {
            table.clear_resident();
            table.drop_widened();
            // While a spill is on, a finished group's packed columns go:
            // nothing reads them after its opening.
            if spill.is_some()
                && let Some(packed) = table.take_narrow_for_spill()
                && let Some(mem) = &mem
            {
                mem.held_narrow.fetch_sub(packed.data().len(), Relaxed);
            }
        }
        drop(store);
        if let Some(mem) = &mem {
            mem.tree_tops.fetch_sub(tree_tops, Relaxed);
            mem.mark(&format!("phase B group {g} end"));
        }
        at += size;
    }
    // The tables go with this function: a memory log stops holding them.
    if let Some(mem) = &mem {
        mem.release(tables.iter());
    }
    if let (Some(spill), Some(prefetch)) = (&spill, &prefetch) {
        *spill
            .prefetch
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(prefetch.report());
    }
    Ok((
        MultiProof {
            roots,
            tables: table_proofs,
            columns: openings,
            preprocessed: None,
        },
        argues,
        prepared_openings,
        stamps,
    ))
}

/// Verifies a [`block_prove`] proof: the roots block, each group on its fork,
/// and the bus balance over every table of the block against `expected`.
///
/// `layouts`/`domains` are one per group, rebuilt by the caller from the
/// shapes and `sizes` ([`block_groups`]) — never from the proof. `prepared` are
/// the tables' prepared checks the caller derived from the program, and
/// `prepared_openings` the proof's openings of them, one each, in that order.
#[allow(clippy::too_many_arguments)]
pub fn block_verify<F, E, T, H>(
    proof: &MultiProof<F, E>,
    prepared_openings: &[StackedProof<F, E>],
    prepared: &[BlockPreparedCheck<'_, F>],
    statements: &[TableStatement<'_, F, E>],
    layouts: &[StackedLayout],
    domains: &[Domain<F>],
    sizes: &[usize],
    expected: &FieldElement<E>,
    config: &ChainConfig,
    transcript: &mut T,
) -> Result<(), MlError>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    H: WhirHash,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    T: crypto::fiat_shamir::is_transcript::IsTranscript<E>
        + crypto::fiat_shamir::transcript_hash::HasTranscriptHash<Hash = <H as WhirHash>::Transcript>
        + Clone,
{
    block_verify_with::<F, E, T, H>(
        proof,
        &[],
        prepared_openings,
        prepared,
        statements,
        layouts,
        domains,
        sizes,
        expected,
        config,
        transcript,
        false,
        VerifierChecks::ALL,
    )
}

/// [`block_verify`] under either argue format, the config's: per table, the
/// proof's `tables`; batched, `argues`, one per group, each read on its
/// group's fork ([`batched::verify_argue`]). The other format's messages must
/// be absent.
///
/// The prepared openings are left unchecked when `skip_prepared` — their
/// prefixes still skipped by `check_preprocessed` — and the batched argue's
/// checks and the bus balance are switched by `checks`. Both are mutations: a
/// test shows each check is what refuses its forgery. Production passes
/// `false` and [`VerifierChecks::ALL`].
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn block_verify_with<F, E, T, H>(
    proof: &MultiProof<F, E>,
    argues: &[BatchedArgue<E>],
    prepared_openings: &[StackedProof<F, E>],
    prepared: &[BlockPreparedCheck<'_, F>],
    statements: &[TableStatement<'_, F, E>],
    layouts: &[StackedLayout],
    domains: &[Domain<F>],
    sizes: &[usize],
    expected: &FieldElement<E>,
    config: &ChainConfig,
    transcript: &mut T,
    skip_prepared: bool,
    checks: VerifierChecks,
) -> Result<(), MlError>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    H: WhirHash,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    T: crypto::fiat_shamir::is_transcript::IsTranscript<E>
        + crypto::fiat_shamir::transcript_hash::HasTranscriptHash<Hash = <H as WhirHash>::Transcript>
        + Clone,
{
    // The format is the verifier's config's, never the proof's: that format's
    // messages present in full, the other's absent.
    let (table_proofs, group_argues) = match config.format.argue {
        ArgueFormat::PerTable => (statements.len(), 0),
        ArgueFormat::Batched { .. } => (0, sizes.len()),
    };
    if proof.tables.len() != table_proofs || argues.len() != group_argues {
        return Err(MlError::ArgueFormatMismatch);
    }
    if layouts.len() != sizes.len()
        || domains.len() != sizes.len()
        || proof.columns.len() != sizes.len()
        || sizes.iter().sum::<usize>() != statements.len()
    {
        return Err(MlError::QueryCountMismatch {
            expected: sizes.len(),
            got: proof.columns.len(),
        });
    }
    // Every root the proof carries is one some group opens: a root no layout
    // claims would be absorbed and bound to nothing.
    let polys: usize = layouts.iter().map(StackedLayout::num_polys).sum();
    if polys != proof.roots.len() {
        return Err(MlError::QueryCountMismatch {
            expected: polys,
            got: proof.roots.len(),
        });
    }
    if proof.preprocessed.is_some() {
        return Err(MlError::QueryCountMismatch {
            expected: 0,
            got: 1,
        });
    }
    // One opening per group with a prepared stack, each table settled at most
    // once and by its own group, each prefix within the table's preprocessed
    // columns — never more than it has, which would skip a check nothing
    // replaced — and the stack holding exactly those columns.
    if prepared_openings.len() != prepared.len() {
        return Err(MlError::QueryCountMismatch {
            expected: prepared.len(),
            got: prepared_openings.len(),
        });
    }
    let starts: Vec<usize> = sizes
        .iter()
        .scan(0usize, |at, &size| {
            let start = *at;
            *at += size;
            Some(start)
        })
        .collect();
    let entries: Vec<PreparedEntry<'_>> = prepared
        .iter()
        .map(|p| {
            (
                p.group,
                p.tables.as_slice(),
                Some(p.layout.placements().len()),
            )
        })
        .collect();
    prepared_shape(&entries, &starts, statements.len(), &|t| {
        statements[t].num_preprocessed
    })?;
    let mut settled = vec![0usize; statements.len()];
    for p in prepared {
        for &(table, n) in &p.tables {
            settled[table] = n;
        }
    }
    let derived: Vec<Commitment> = prepared
        .iter()
        .flat_map(|p| p.roots.iter().copied())
        .collect();
    let (z, alpha, beta) = absorb_roots_and_challenge::<E, T>(transcript, &proof.roots, &derived);

    let mut balance = FieldElement::<E>::zero();
    let mut statement_at = 0usize;
    let mut root_at = 0usize;
    for (g, (((opening, layout), domain), &size)) in proof
        .columns
        .iter()
        .zip(layouts)
        .zip(domains)
        .zip(sizes)
        .enumerate()
    {
        let mut fork = group_fork::<E, T>(transcript, g);
        let mut points: Vec<Vec<FieldElement<E>>> = Vec::new();
        let mut values: Vec<FieldElement<E>> = Vec::new();
        let group_statements = &statements[statement_at..statement_at + size];
        let group_settled = &settled[statement_at..statement_at + size];
        match config.format.argue {
            ArgueFormat::PerTable => {
                for ((table, statement), &settled) in proof.tables
                    [statement_at..statement_at + size]
                    .iter()
                    .zip(group_statements)
                    .zip(group_settled)
                {
                    let (output, reduced) =
                        verify(table, *statement, &z, &alpha, &beta, &mut fork, settled)?;
                    balance += contribution(&output).ok_or(MlError::BusImbalance)?;
                    for _ in 0..statement.slot_of.len() {
                        points.push(reduced.point.clone());
                    }
                    values.extend(reduced.column_values);
                }
            }
            ArgueFormat::Batched { bin_log_cells } => {
                // The group's bins come from its statements and the
                // verifier's cap, never from the proof.
                let argue = &argues[g];
                let reduced = batched::verify_argue(
                    argue,
                    group_statements,
                    bin_log_cells,
                    &z,
                    &alpha,
                    &beta,
                    &mut fork,
                    checks,
                )?;
                for output in &argue.bus_outputs {
                    balance += contribution(output).ok_or(MlError::BusImbalance)?;
                }
                for ((statement, reduced), &settled) in
                    group_statements.iter().zip(reduced).zip(group_settled)
                {
                    if checks.preprocessed {
                        check_preprocessed(*statement, &reduced, settled)?;
                    }
                    for _ in 0..statement.slot_of.len() {
                        points.push(reduced.point.clone());
                    }
                    values.extend(reduced.column_values);
                }
            }
        }
        let roots = &proof.roots[root_at..root_at + layout.num_polys()];
        stacked_eval::verify::<F, E, T, H>(
            opening,
            layout,
            roots,
            &Claimed::PerColumn(&points),
            &values,
            domain,
            config,
            &mut fork,
        )?;
        // The group's prepared stack, on this fork: the derived commitment must
        // take the values each table settled on, at that table's point.
        if let Some(k) = prepared
            .iter()
            .position(|p| p.group == g)
            .filter(|_| !skip_prepared)
        {
            let firsts: Vec<usize> = statements[statement_at..statement_at + size]
                .iter()
                .scan(0usize, |first, s| {
                    let at = *first;
                    *first += s.slot_of.len();
                    Some(at)
                })
                .collect();
            let p = &prepared[k];
            let (at_points, at_values) =
                prepared_claims(&p.tables, statement_at, &firsts, &points, &values);
            stacked_eval::verify::<F, E, T, H>(
                &prepared_openings[k],
                p.layout,
                p.roots,
                &Claimed::PerColumn(&at_points),
                &at_values,
                p.domain,
                config,
                &mut fork,
            )?;
        }
        statement_at += size;
        root_at += layout.num_polys();
    }
    if checks.balance && balance != *expected {
        return Err(MlError::BusImbalance);
    }
    Ok(())
}

#[cfg(test)]
mod spill_tests {
    use super::{commit_pairs_off_pool_from, from_store, to_store};
    use multilinear::narrow::NarrowColumns;

    /// The packed columns move to the store's payload and back without a
    /// copy: the same bytes at the same address.
    /// The default commits the pairs off the pool; `0` is the opt-out to the pool.
    #[test]
    fn the_commit_pairs_go_off_the_pool_by_default_and_zero_opts_out() {
        assert!(commit_pairs_off_pool_from(None));
        assert!(commit_pairs_off_pool_from(Some("1")));
        assert!(commit_pairs_off_pool_from(Some("yes")));
        assert!(!commit_pairs_off_pool_from(Some("0")));
        assert!(!commit_pairs_off_pool_from(Some(" 0 ")));
    }

    #[test]
    fn the_columns_move_to_the_store_and_back_without_a_copy() {
        let words: Vec<u64> = (0..3 * 4096u64).map(|i| i * 0x9e37_79b9 % 70_000).collect();
        let packed = NarrowColumns::pack_row_major(&words, 3).expect("packs");
        let copy = packed.clone();
        let at = packed.data().as_ptr();
        let main = to_store(packed).expect("converts");
        assert_eq!(main.data().as_ptr(), at, "to the store");
        let back = from_store(main).expect("converts back");
        assert_eq!(back.data().as_ptr(), at, "back");
        assert_eq!(back, copy);
    }
}
