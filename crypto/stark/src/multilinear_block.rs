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
use crate::regen::{RegenError, RegenProducer, RegenSlot, RegenWindow};
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
    /// Phase B: when the group's turn began and ended, seconds since phase B
    /// began (its columns back, up, argued, opened and let go). Under
    /// [`PhaseBOverlap::On`] a group's turn ends once the next group's
    /// argument does, so the turns overlap.
    pub start_b: f64,
    pub end_b: f64,
    /// Host bytes kept of the group's trees between the phases.
    pub tree_bytes: usize,
    /// Phase B: the bytes of records its tables held as their records, let go
    /// with the group after its openings (under a spill).
    pub records_released: usize,
    /// The device ledger's peak promise through the group's argue — the
    /// argue's own and the next group's columns uploaded beside it, and under
    /// [`PhaseBOverlap::On`] the previous group's openings beside it too
    /// (`transition_reserved`). Bytes; 0 without a device.
    pub argue_reserved: u64,
    /// The ledger's promise just before the group's argue (its own columns
    /// on the card), and the ledger's peak through its revive and openings
    /// (the next group's columns, uploaded beside the argue, included), with
    /// what the revive itself added (the openings' room). Bytes; 0 without a
    /// device. Phase B's overlap reads them: group g's openings beside group
    /// g + 1's argue would hold about `open_reserved[g] + argue_reserved[g+1]
    /// − argue_base[g+1]`. When the openings ran beside the next group's
    /// argue, the two share one window: `open_reserved` is that window's peak
    /// (the next group's `transition_reserved`), and `open_room` is 0.
    pub argue_base: u64,
    pub open_reserved: u64,
    pub open_room: u64,
    /// Of the group's openings (`encode` + `open`), the seconds phase B's
    /// driver waited for: all of them when it ran them itself; beside the next
    /// group's argue ([`PhaseBOverlap::On`]), what outlasted that argue and
    /// the upload beside it.
    pub open_paid: f64,
    /// The group's openings ran beside the next group's argue.
    pub overlapped: bool,
    /// Under [`PhaseBOverlap::On`], the group's openings ran alone because an
    /// earlier transition saw the card refuse a promise.
    pub serial_after_refusal: bool,
    /// The ledger's peak through the group's argue with the previous group's
    /// openings beside it (and the next group's upload); 0 when no openings
    /// ran beside it. Bytes.
    pub transition_reserved: u64,
    /// Stream draws while the group's openings ran, `(foreground pool,
    /// background pool)`. Beside an argue the foreground count holds the
    /// argue's draws too; alone, only the openings'.
    pub open_draws: (u64, u64),
    /// Phase B's driver thread over the group's turn on it (from the turn's
    /// start to the next turn's), `(CPU ns, run-queue delay ns)` from its
    /// schedstat, when [`BlockCommitted::with_sched_stamps`] asks for them
    /// and the file reads; `None` otherwise.
    pub sched_b: Option<(u64, u64)>,
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
            match table.compact_bytes() {
                Some(bytes) => self.held_narrow.fetch_add(bytes, Relaxed),
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
            match table.compact_bytes() {
                Some(bytes) => self.held_narrow.fetch_sub(bytes, Relaxed),
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
///
/// With no store the policy still decides, and what it would move off the
/// host stays parked. With live regeneration ([`Self::with_regen`]) a table
/// phase B can rebuild leaves the host as a drop instead ([`BlockRegen`]).
#[derive(Clone)]
pub struct BlockSpill {
    pub store: Option<Arc<SpillStore>>,
    /// The writer queue the store was opened with
    /// ([`crate::spill::SpillOptions::queue_bytes`]).
    pub queue_bytes: u64,
    /// The policy's choice for one table.
    pub wanted: Arc<SpillWanted>,
    /// Live regeneration, when it is on.
    pub regen: Option<BlockRegen>,
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
/// hand-off ([`BlockSpill::hand_off`]) moves them to the store or live
/// regeneration drops them, and read back from whichever holds them before the
/// group's upload (`restore_group`).
struct Parked {
    len: u64,
    /// The main cells committed once it was, its own included
    /// ([`BlockRegen::keep_cells`]).
    cells: u64,
    /// Its class and rank when phase B can rebuild it ([`BlockRegen`]).
    rank: Option<(usize, u64)>,
    state: std::sync::Mutex<ParkedState>,
}

enum ParkedState {
    Resident(multilinear::narrow::NarrowColumns),
    /// A table held as the records its rows come from
    /// ([`multilinear::records`]): live regeneration drops it as them, into a
    /// records slot ([`RegenWindow::slot_for_records`]).
    ResidentRecords(multilinear::records::Records),
    Spilled(SpilledMain),
    /// Dropped after its digest: phase B's regenerator deposits it again.
    Dropped(RegenSlot),
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

    /// The slot it was dropped into, once dropped.
    fn dropped(&self) -> Option<RegenSlot> {
        match &*self.state.lock().unwrap_or_else(|e| e.into_inner()) {
            ParkedState::Dropped(slot) => Some(slot.clone()),
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

    /// The class the columns were dropped in, when they were.
    fn dropped_class(&self) -> Option<usize> {
        match self {
            Self::Spilled(_) => None,
            Self::Parked(parked) => parked
                .dropped()
                .and_then(|_| parked.rank.map(|(class, _)| class)),
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
    /// A spill to `store`, or deciding with none (what the policy would move
    /// off the host stays parked).
    pub fn new(store: Option<SpillStore>, queue_bytes: u64, wanted: Arc<SpillWanted>) -> Self {
        Self {
            store: store.map(Arc::new),
            queue_bytes,
            wanted,
            regen: None,
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

    /// Live regeneration on: a table `regen` can rebuild leaves the host as a
    /// drop ([`BlockRegen`]).
    pub fn with_regen(mut self, regen: BlockRegen) -> Self {
        self.regen = Some(regen);
        self
    }

    /// Hands the parked tables to the store, oldest first, until `need` bytes
    /// have gone or phase A ends, through the store's own hand-off
    /// ([`SpillStore::spill`]): the tables parked by then, and those parked
    /// after, as they come (while it runs the committer parks what it would
    /// have spilled). It runs on a thread of its own, so the caller does not
    /// wait on the writers, and phase A's end joins it. Once only, and never
    /// after phase A's end: `false` then (or when the thread cannot start).
    ///
    /// With live regeneration it arms it first ([`BlockRegen::arm`]): every
    /// parked table phase B can rebuild is dropped, and only the others go to
    /// the store; with no store, nothing else leaves. Partial drops
    /// ([`BlockRegen::partial`]) drop back parked tables, in rank order, only
    /// until `need` bytes are on their way.
    pub fn hand_off(&self, need: u64, mem: Option<Arc<BlockMem>>) -> bool {
        if let Some(regen) = &self.regen {
            regen.arm_at(self.cells.load(Relaxed));
            // Partial drops hand back only what the hand-off asks for.
            if regen.is_partial() {
                regen.drop_back(need);
            }
        }
        let Some(store) = self.store.clone() else {
            return self.regen.is_some();
        };
        let mut hand_off = self.hand_off.lock().unwrap_or_else(|e| e.into_inner());
        if hand_off.closed || hand_off.thread.is_some() || hand_off.report.is_some() {
            return false;
        }
        let (parked, parked_more, closed, handing) = (
            Arc::clone(&self.parked),
            Arc::clone(&self.parked_more),
            Arc::clone(&self.closed),
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
                    // Regeneration's: dropped, not written.
                    if slot.rank.is_some() {
                        continue;
                    }
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

    /// Phase A's end: joins a running hand-off and live regeneration's drops,
    /// and none starts after this.
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
        if let Some(regen) = &self.regen {
            regen.close_drops();
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
        let Some(store) = &self.store else {
            return false;
        };
        let stats = store.stats();
        let pending = stats.bytes.saturating_sub(stats.bytes_written);
        pending == 0 || pending + len <= self.queue_bytes
    }
}

/// Live regeneration on a block (D-WHIR-NODISK §2.1). `droppable(t)` is the
/// rank of the table at block index `t` when phase B can rebuild its columns
/// (a streamed chunk: its place in the hand-out), `None` otherwise. Phase A
/// drops such a table instead of keeping or spilling it: its packed columns
/// leave the host after their digest, on a thread of their own, and a
/// [`RegenSlot`] in `window` stands for them; phase B waits on the slot before
/// the table's group needs it (`restore_group`), and a deposit that is not
/// the dropped columns refuses the table there, before any device work, with
/// the kept tree top's check behind the digest.
///
/// Under `always` every droppable table is dropped (the byte-identity test
/// mode). Otherwise nothing drops until the policy first moves a table off
/// the host (or the walk's hand-off): that arms it, every parked droppable
/// table is dropped back in rank order, and every later one is dropped. A
/// block that never arms drops nothing.
#[derive(Clone)]
pub struct BlockRegen {
    droppable: Arc<dyn Fn(usize) -> Option<(usize, u64)> + Send + Sync>,
    /// One window a class: each class's regenerator deposits, and phase B
    /// takes, in that class's rank order.
    windows: Vec<Arc<RegenWindow>>,
    always: bool,
    /// Each table is dropped only when the policy wants it out, armed or not
    /// ([`Self::partial`]).
    partial: bool,
    /// Partial drops' keep budget at arming, in main cells committed
    /// ([`Self::keep_cells`]).
    keep_cells: Option<u64>,
    /// Partial drops' keep budget relative to the arming point: this many main
    /// cells below those committed at arming ([`Self::keep_below`]).
    keep_below: Option<u64>,
    started: Instant,
    inner: Arc<std::sync::Mutex<RegenState>>,
    /// Wakes [`Self::await_rank`] at each decision and when the decisions
    /// close.
    decided: Arc<std::sync::Condvar>,
}

/// A droppable table's fate, as a regenerator started before phase A's end
/// needs it ([`BlockRegen::await_rank`]).
#[derive(Clone)]
pub enum RankDecision {
    /// Dropped: its slot, which the regenerator fills.
    Dropped(RegenSlot),
    /// Not dropped, and never will be: held by its group's class
    /// ([`drop_one_class`]), gone to the store, or phase A ended without
    /// dropping it.
    Kept,
    /// Its class's window closed (the prove ended or failed) before the table
    /// was decided.
    Stopped,
}

#[derive(Default)]
struct RegenState {
    /// When it armed, seconds after it was made.
    armed: Option<f64>,
    /// At arming: the main cells committed, when known; and the candidates
    /// parked by then (tables, bytes), before the keep budget's drop-back.
    armed_cells: Option<u64>,
    armed_kept: (usize, u64),
    /// What the keep budget dropped back at arming (tables, bytes).
    keep_back: (usize, u64),
    /// The budget arming used, in main cells committed, when it had one.
    budget: Option<u64>,
    /// Each window's first producer, kept for its regenerator.
    producers: Vec<Option<RegenProducer>>,
    /// Parked droppable tables, for drop-back at arming.
    candidates: Vec<Arc<Parked>>,
    tx: Option<std::sync::mpsc::Sender<DropJob>>,
    thread: Option<std::thread::JoinHandle<()>>,
    /// Every drop, in the order made: its class and rank, its slot, its bytes
    /// and whether drop-back made it.
    dropped: Vec<((usize, u64), RegenSlot, u64, bool)>,
    /// Drops that found their table gone from the host (handed to the store).
    refused_late: usize,
    /// Partial drops: the candidates were settled ([`BlockRegen::settle`]),
    /// and a table not dropped from now on is kept at once.
    settled: bool,
    /// Tables phase B could rebuild that stay, their group dropping another
    /// class ([`drop_one_class`]).
    held: usize,
    /// A memory log, which counts each drop out of the held bytes.
    mem: Option<Arc<BlockMem>>,
    /// Each decided table by class and rank: its slot when dropped; the
    /// ranks decided to stay; and whether the decisions are over (phase A's
    /// end: an undecided table then stays).
    slots: std::collections::HashMap<(usize, u64), RegenSlot>,
    kept: std::collections::HashSet<(usize, u64)>,
    decisions_closed: bool,
    /// Test only: a pause before each drop of this class, so a regenerator
    /// started early meets undecided tables.
    drop_delay: Option<(usize, std::time::Duration)>,
}

/// A table for the drop thread, and whether drop-back sent it.
struct DropJob {
    parked: Arc<Parked>,
    back: bool,
}

/// What live regeneration dropped ([`BlockRegen::report`]).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DropReport {
    pub armed: Option<f64>,
    pub tables: usize,
    pub bytes: u64,
    pub back_tables: usize,
    pub back_bytes: u64,
    pub refused_late: usize,
    /// By class: the tables and bytes dropped.
    pub by_class: Vec<(usize, u64)>,
    /// Tables phase B could rebuild kept on the host because their group
    /// dropped another class ([`drop_one_class`]).
    pub held: usize,
    /// At arming: the main cells committed, when known, and the candidates
    /// parked by then (tables, bytes) before the keep budget's drop-back.
    pub armed_cells: Option<u64>,
    pub armed_kept: (usize, u64),
    /// The keep budget ([`BlockRegen::keep_cells`]) and what it dropped back
    /// at arming (tables, bytes).
    pub keep_cells: Option<u64>,
    pub keep_back: (usize, u64),
    /// The budget relative to the arming point ([`BlockRegen::keep_below`]),
    /// and the budget arming used, in main cells committed.
    pub keep_below: Option<u64>,
    pub budget: Option<u64>,
    /// Of `tables` and `bytes`, those dropped as the records their rows come
    /// from (their records' bytes).
    pub records: (usize, u64),
}

/// One class's regeneration for phase B ([`BlockRegen::into_plan`]): its
/// window, its dropped tables' ranks and slots in the window's order, and the
/// window's first producer for its regenerator.
pub struct ClassPlan {
    pub window: Arc<RegenWindow>,
    pub dropped: Vec<(u64, RegenSlot)>,
    pub producer: Option<RegenProducer>,
}

impl BlockRegen {
    /// Live regeneration over `droppable` (a table's class and rank, when
    /// phase B can rebuild it), a class's slots paced by a window of its
    /// `aheads` bytes ([`RegenWindow::new`]). Its drop thread starts here;
    /// phase A's end joins it.
    pub fn new(
        droppable: Arc<dyn Fn(usize) -> Option<(usize, u64)> + Send + Sync>,
        aheads: &[u64],
        always: bool,
    ) -> Self {
        let (windows, producers): (Vec<_>, Vec<_>) = aheads
            .iter()
            .map(|&ahead| {
                let (window, producer) = RegenWindow::new(ahead);
                (window, Some(producer))
            })
            .unzip();
        let inner = Arc::new(std::sync::Mutex::new(RegenState {
            producers,
            ..RegenState::default()
        }));
        let (tx, rx) = std::sync::mpsc::channel::<DropJob>();
        let decided = Arc::new(std::sync::Condvar::new());
        let (windows_in, inner_in, decided_in) =
            (windows.clone(), Arc::clone(&inner), Arc::clone(&decided));
        let thread = std::thread::Builder::new()
            .name("block-regen-drop".to_string())
            .spawn(move || {
                for job in rx {
                    let rank = job.parked.rank;
                    let (mem, delay) = {
                        let state = inner_in.lock().unwrap_or_else(|e| e.into_inner());
                        (state.mem.clone(), state.drop_delay)
                    };
                    if let (Some((class, pause)), Some((c, _))) = (delay, rank)
                        && class == c
                    {
                        std::thread::sleep(pause);
                    }
                    let dropped = rank.and_then(|(class, rank)| {
                        let window = windows_in.get(class)?;
                        drop_parked(window, &job.parked, rank, mem.as_deref())
                    });
                    let mut state = inner_in.lock().unwrap_or_else(|e| e.into_inner());
                    match (dropped, rank) {
                        (Some(slot), Some(rank)) => {
                            state.slots.insert(rank, slot.clone());
                            state.dropped.push((rank, slot, job.parked.len, job.back))
                        }
                        (None, Some(rank)) => {
                            state.kept.insert(rank);
                            state.refused_late += 1
                        }
                        _ => state.refused_late += 1,
                    }
                    decided_in.notify_all();
                }
            });
        {
            let mut state = inner.lock().unwrap_or_else(|e| e.into_inner());
            match thread {
                Ok(thread) => {
                    state.tx = Some(tx);
                    state.thread = Some(thread);
                }
                // No drop thread: nothing drops, and every table stays as the
                // policy leaves it.
                Err(_) => drop(tx),
            }
        }
        Self {
            droppable,
            windows,
            always,
            partial: false,
            keep_cells: None,
            keep_below: None,
            started: Instant::now(),
            inner,
            decided,
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, RegenState> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The class and rank of the table at block index `index`, when it may
    /// be dropped (a class with no window may not).
    pub fn rank_of(&self, index: usize) -> Option<(usize, u64)> {
        (self.droppable)(index).filter(|&(class, _)| class < self.windows.len())
    }

    /// Each class's window, the dropped tables' slots in it.
    pub fn windows(&self) -> &[Arc<RegenWindow>] {
        &self.windows
    }

    /// Whether it armed.
    pub fn is_armed(&self) -> bool {
        self.lock().armed.is_some()
    }

    /// Whether every droppable table is dropped.
    pub fn always(&self) -> bool {
        self.always
    }

    /// Partial drops: arming drops back nothing and does not make later tables
    /// drop. Each committed table is dropped only when the policy wants it out
    /// as it is installed ([`Self::drop_one`]), and the parked ones only as far
    /// as a hand-off asks ([`Self::drop_back`]). The rest stay on the host.
    pub fn partial(mut self) -> Self {
        self.partial = true;
        self
    }

    /// Whether drops are partial ([`Self::partial`]).
    pub fn is_partial(&self) -> bool {
        self.partial
    }

    /// Partial drops' keep budget: at arming, the candidates committed past
    /// `cells` main cells (the block's cells committed once each was, its own
    /// included) are dropped back, in rank order, and only those; the others
    /// stay candidates as before. What arming keeps then no longer follows how
    /// far phase A got when the policy first wanted a table out. `None` (the
    /// default): arming drops back nothing. Nothing for drops that are not
    /// partial.
    pub fn keep_cells(mut self, cells: Option<u64>) -> Self {
        self.keep_cells = cells;
        self
    }

    /// Partial drops' keep budget relative to the arming point: `cells` main
    /// cells below those committed when it arms ([`Self::arm_at`]), so the
    /// candidates committed in the last `cells` before arming are dropped
    /// back, in rank order. Where the arming point moves (the host's reading at
    /// the forecast's crossing), the budget moves with it. With
    /// [`Self::keep_cells`] too, the lower of the two holds. Arming without the
    /// cells committed ([`Self::arm`]) uses only the absolute budget. `None`
    /// (the default): no relative budget. Nothing for drops that are not
    /// partial.
    pub fn keep_below(mut self, cells: Option<u64>) -> Self {
        self.keep_below = cells;
        self
    }

    /// Arm: from now on every droppable table is dropped, and every parked one
    /// is dropped back, in rank order; partial drops ([`Self::partial`]) note
    /// when, and drop back only what their keep budget asks
    /// ([`Self::keep_cells`]). Returns whether this call armed.
    pub fn arm(&self) -> bool {
        self.arm_with(None)
    }

    /// [`Self::arm`] with the main cells committed by then, which the drop
    /// report keeps.
    pub fn arm_at(&self, cells: u64) -> bool {
        self.arm_with(Some(cells))
    }

    fn arm_with(&self, cells: Option<u64>) -> bool {
        let mut state = self.lock();
        if state.armed.is_some() {
            return false;
        }
        state.armed = Some(self.started.elapsed().as_secs_f64());
        state.armed_cells = cells;
        if self.partial {
            state.armed_kept = (
                state.candidates.len(),
                state.candidates.iter().map(|p| p.len).sum(),
            );
            let below = self
                .keep_below
                .zip(cells)
                .map(|(below, at)| at.saturating_sub(below));
            state.budget = match (self.keep_cells, below) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (a, b) => a.or(b),
            };
            if let Some(budget) = state.budget {
                let (mut past, within): (Vec<_>, Vec<_>) = std::mem::take(&mut state.candidates)
                    .into_iter()
                    .partition(|p| p.cells > budget);
                past.sort_by_key(|p| p.rank);
                state.keep_back = (past.len(), past.iter().map(|p| p.len).sum());
                state.candidates = within;
                for parked in past {
                    Self::queue(&state, parked, true);
                }
            }
            return true;
        }
        let mut candidates = std::mem::take(&mut state.candidates);
        candidates.sort_by_key(|p| p.rank);
        for parked in candidates {
            Self::queue(&state, parked, true);
        }
        true
    }

    /// A memory log for the drops to count in.
    fn set_mem(&self, mem: Option<Arc<BlockMem>>) {
        self.lock().mem = mem;
    }

    /// A parked droppable table: dropped now when armed (not partial) or
    /// `always`, else a drop-back candidate.
    fn parked(&self, parked: Arc<Parked>) {
        let mut state = self.lock();
        if self.always || (state.armed.is_some() && !self.partial) {
            Self::queue(&state, parked, false);
        } else if self.partial && state.settled {
            if let Some(rank) = parked.rank {
                state.kept.insert(rank);
            }
            drop(state);
            self.decided.notify_all();
        } else {
            state.candidates.push(parked);
        }
    }

    /// Partial drops: every parked candidate kept from now on, and every table
    /// not dropped later kept as it is parked, so a regenerator started before
    /// phase A's end learns each table's fate as it comes
    /// ([`Self::drops_all_from_now`]). Called once the last drop-back is past
    /// (the walk's end, after its hand-off). Nothing for drops that are not
    /// partial.
    pub fn settle(&self) {
        if !self.partial {
            return;
        }
        let mut state = self.lock();
        let candidates = std::mem::take(&mut state.candidates);
        for parked in candidates {
            if let Some(rank) = parked.rank {
                state.kept.insert(rank);
            }
        }
        state.settled = true;
        drop(state);
        self.decided.notify_all();
    }

    /// A parked droppable table the policy wants out: dropped now.
    fn drop_one(&self, parked: Arc<Parked>) {
        Self::queue(&self.lock(), parked, false);
    }

    /// Partial drops: the parked candidates dropped back, in rank order, until
    /// `need` bytes are on their way (all of them for `u64::MAX`). Returns the
    /// bytes it sent.
    pub fn drop_back(&self, need: u64) -> u64 {
        let mut state = self.lock();
        let mut candidates = std::mem::take(&mut state.candidates);
        candidates.sort_by_key(|p| p.rank);
        let mut sent = 0u64;
        let mut rest = Vec::new();
        for parked in candidates {
            if sent < need {
                sent = sent.saturating_add(parked.len);
                Self::queue(&state, parked, true);
            } else {
                rest.push(parked);
            }
        }
        state.candidates = rest;
        sent
    }

    fn queue(state: &RegenState, parked: Arc<Parked>, back: bool) {
        if let Some(tx) = &state.tx {
            let _ = tx.send(DropJob { parked, back });
        }
    }

    /// No more drops: the drop thread finishes what is queued and ends, and
    /// every table not dropped by then stays ([`RankDecision::Kept`]).
    fn close_drops(&self) {
        let thread = {
            let mut state = self.lock();
            state.tx = None;
            state.thread.take()
        };
        if let Some(thread) = thread {
            let _ = thread.join();
        }
        self.lock().decisions_closed = true;
        self.decided.notify_all();
    }

    /// Tables kept because their group dropped another class.
    fn held(&self, ranks: &[(usize, u64)]) {
        let mut state = self.lock();
        state.held += ranks.len();
        state.kept.extend(ranks.iter().copied());
        drop(state);
        self.decided.notify_all();
    }

    /// Whether the fate of every table phase B can rebuild that phase A still
    /// commits is known as it is installed: dropped (armed, or `always`), or
    /// under partial drops once settled ([`Self::settle`]), dropped or kept. A
    /// regenerator may then start before phase A's end and take each table's
    /// fate from [`Self::await_rank`].
    pub fn drops_all_from_now(&self) -> bool {
        let state = self.lock();
        self.always || (state.armed.is_some() && !self.partial) || (self.partial && state.settled)
    }

    /// Class `class`'s first producer, for a regenerator that starts before
    /// phase A's end ([`Self::into_plan`] then hands out none for the class).
    pub fn take_producer(&self, class: usize) -> Option<RegenProducer> {
        self.lock().producers.get_mut(class).and_then(Option::take)
    }

    /// The fate of class `class`'s table of rank `rank`, waiting for it:
    /// dropped (its slot), kept, or `Stopped` once the class's window
    /// closes. Returns too how long it waited. A table phase A never dropped
    /// is kept once phase A's end closes the decisions; nothing is taken as
    /// kept before that, so a regenerator never skips a table that drops
    /// later.
    pub fn await_rank(&self, class: usize, rank: u64) -> (RankDecision, std::time::Duration) {
        let started = Instant::now();
        let window = self.windows.get(class).cloned();
        let mut state = self.lock();
        loop {
            if let Some(slot) = state.slots.get(&(class, rank)) {
                return (RankDecision::Dropped(slot.clone()), started.elapsed());
            }
            if state.kept.contains(&(class, rank)) || state.decisions_closed {
                return (RankDecision::Kept, started.elapsed());
            }
            if window.as_ref().is_none_or(|w| w.is_closed()) {
                return (RankDecision::Stopped, started.elapsed());
            }
            // A closed window does not wake this condvar: poll for it.
            state = self
                .decided
                .wait_timeout(state, std::time::Duration::from_millis(50))
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
    }

    /// Whether phase A's end has closed the decisions.
    pub fn decisions_closed(&self) -> bool {
        self.lock().decisions_closed
    }

    /// Test only: pause before each drop of class `class`, so a regenerator
    /// started early meets undecided tables.
    #[doc(hidden)]
    pub fn delay_drops(&self, class: usize, pause: std::time::Duration) {
        self.lock().drop_delay = Some((class, pause));
    }

    /// What it dropped.
    pub fn report(&self) -> DropReport {
        let state = self.lock();
        let back = state.dropped.iter().filter(|d| d.3);
        let mut by_class = vec![(0usize, 0u64); self.windows.len()];
        for ((class, _), _, bytes, _) in &state.dropped {
            if let Some(c) = by_class.get_mut(*class) {
                c.0 += 1;
                c.1 += bytes;
            }
        }
        let records = state.dropped.iter().filter(|d| d.1.is_records());
        DropReport {
            armed: state.armed,
            tables: state.dropped.len(),
            bytes: state.dropped.iter().map(|d| d.2).sum(),
            back_tables: back.clone().count(),
            back_bytes: back.map(|d| d.2).sum(),
            refused_late: state.refused_late,
            by_class,
            held: state.held,
            armed_cells: state.armed_cells,
            armed_kept: state.armed_kept,
            keep_cells: self.keep_cells.filter(|_| self.partial),
            keep_back: state.keep_back,
            keep_below: self.keep_below.filter(|_| self.partial),
            budget: state.budget,
            records: (records.clone().count(), records.map(|d| d.2).sum()),
        }
    }

    /// Phase B's regeneration, a class each ([`ClassPlan`]): a class that
    /// dropped nothing has no dropped tables and no producer (it went with
    /// the plan, so its window's slots, none, need no regenerator).
    pub fn into_plan(self) -> Vec<ClassPlan> {
        let mut state = self.lock();
        let producers = std::mem::take(&mut state.producers);
        let mut plans: Vec<ClassPlan> = self
            .windows
            .iter()
            .zip(producers)
            .map(|(window, producer)| ClassPlan {
                window: Arc::clone(window),
                dropped: Vec::new(),
                producer,
            })
            .collect();
        for ((class, rank), slot, _, _) in &state.dropped {
            if let Some(plan) = plans.get_mut(*class) {
                plan.dropped.push((*rank, slot.clone()));
            }
        }
        for plan in &mut plans {
            plan.dropped.sort_by_key(|(_, slot)| slot.order_key());
            if plan.dropped.is_empty() {
                plan.producer = None;
            }
        }
        plans
    }
}

/// Drops a parked table's packed columns: they leave `parked` under its lock,
/// are digested into a slot of `window` outside it, are freed, and the slot
/// takes their place. `None`, and nothing changed, when the table is no
/// longer held on the host.
fn drop_parked(
    window: &Arc<RegenWindow>,
    parked: &Parked,
    rank: u64,
    mem: Option<&BlockMem>,
) -> Option<RegenSlot> {
    let held = {
        let mut state = parked.state.lock().unwrap_or_else(|e| e.into_inner());
        match std::mem::replace(&mut *state, ParkedState::Lost) {
            held @ (ParkedState::Resident(_) | ParkedState::ResidentRecords(_)) => held,
            other => {
                *state = other;
                return None;
            }
        }
    };
    // Records are digested as they lie; their slot takes the packed columns
    // the regenerator makes from records it rebuilt.
    let (slot, len) = match &held {
        ParkedState::Resident(packed) => (window.slot(packed, rank), packed.data().len()),
        ParkedState::ResidentRecords(records) => {
            (window.slot_for_records(records, rank), records.bytes())
        }
        _ => return None,
    };
    drop(held);
    *parked.state.lock().unwrap_or_else(|e| e.into_inner()) = ParkedState::Dropped(slot.clone());
    if let Some(mem) = mem {
        mem.held_narrow.fetch_sub(len, Relaxed);
    }
    Some(slot)
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
///
/// With live regeneration a table phase B can rebuild is parked in any group
/// and is dropped instead ([`BlockRegen`]): at once under `always`, once the
/// policy first wants a table out (which arms it), and back at arming while
/// parked. It counts toward the bytes kept only while it is not dropped.
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
    // A group's tables drop in one class only ([`drop_one_class`]). A table
    // held as its records drops as them, and is kept where it has no rank.
    let mut ranks: Vec<Option<(usize, u64)>> = (0..tables.len())
        .map(|k| {
            tables[k].compact_bytes()?;
            spill.regen.as_ref()?.rank_of(first + k)
        })
        .collect();
    let before = ranks.clone();
    drop_one_class(&mut ranks);
    if let Some(regen) = &spill.regen {
        let held: Vec<(usize, u64)> = before
            .iter()
            .zip(&ranks)
            .filter_map(|(b, a)| b.filter(|_| a.is_none()))
            .collect();
        if !held.is_empty() {
            regen.held(&held);
        }
    }
    for (k, (table, slot)) in tables.iter_mut().zip(out.iter_mut()).enumerate() {
        let table_cells = (table.num_committed_columns() as u64) << table.num_vars();
        let cells = spill.cells.fetch_add(table_cells, Relaxed) + table_cells;
        let Some(len) = table.compact_bytes() else {
            continue;
        };
        let rank = ranks[k];
        // Records never go to the store: without a rank they stay.
        let records = table.records().is_some();
        if (g < SPILL_RESIDENT_GROUPS || records) && rank.is_none() {
            spill.kept.fetch_add(len as u64, Relaxed);
            continue;
        }
        let failed = || MlError::SpillFailed {
            table: first + k,
            reason: "its packed parts did not move to the store and back",
        };
        if let (Some(regen), Some(_)) = (&spill.regen, rank) {
            let held = if records {
                ParkedState::ResidentRecords(table.take_records_for_spill().ok_or_else(failed)?)
            } else {
                ParkedState::Resident(table.take_narrow_for_spill().ok_or_else(failed)?)
            };
            let parked = Arc::new(Parked {
                len: len as u64,
                cells,
                rank,
                state: std::sync::Mutex::new(held),
            });
            *slot = Some(Out::Parked(Arc::clone(&parked)));
            let wanted =
                !regen.always() && (spill.wanted)(spill.kept.load(Relaxed), cells, len as u64);
            if wanted {
                regen.arm_at(cells);
            }
            if regen.is_partial() && !regen.always() {
                // Out only while the policy wants it out; else kept, and a
                // candidate for a hand-off's drop-back.
                if wanted {
                    regen.drop_one(parked);
                } else {
                    spill.kept.fetch_add(len as u64, Relaxed);
                    regen.parked(parked);
                }
                continue;
            }
            if !regen.always() && !regen.is_armed() {
                spill.kept.fetch_add(len as u64, Relaxed);
            }
            regen.parked(parked);
            continue;
        }
        let packed = table.take_narrow_for_spill().ok_or_else(failed)?;
        let spill_now = spill.store.is_some()
            && (spill.wanted)(spill.kept.load(Relaxed), cells, len as u64)
            && !spill.handing.load(std::sync::atomic::Ordering::SeqCst)
            && spill.has_room(len as u64);
        let packed = match (&spill.store, spill_now) {
            (Some(store), true) => match store.spill(to_store(packed).ok_or_else(failed)?) {
                Ok(handle) => {
                    if let Some(mem) = mem {
                        mem.held_narrow.fetch_sub(len, Relaxed);
                        mem.spilled.fetch_add(len, Relaxed);
                    }
                    *slot = Some(Out::Spilled(handle));
                    continue;
                }
                Err(main) => from_store(main).ok_or_else(failed)?,
            },
            _ => packed,
        };
        let parked = Arc::new(Parked {
            len: len as u64,
            cells,
            rank: None,
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
/// has them, or read here; a dropped table's from its regeneration slot,
/// waiting for its deposit. Refused when a read fails or does not match what
/// was written, or a deposit is not what was dropped.
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
                    ParkedState::ResidentRecords(records) => {
                        if !table.restore_records(records) {
                            return Err(MlError::SpillFailed {
                                table: index,
                                reason: "the parked records do not have the table's shape",
                            });
                        }
                        continue;
                    }
                    ParkedState::Dropped(regen) => {
                        regen.wait();
                        let packed = regen.take().map_err(|e| MlError::SpillFailed {
                            table: index,
                            reason: match e {
                                RegenError::Mismatch => {
                                    "the regenerated columns are not the ones dropped"
                                }
                                RegenError::Failed(_) => {
                                    "the regenerator did not bring its columns back"
                                }
                                RegenError::Closed(_) => {
                                    "the regeneration window closed before its columns came back"
                                }
                            },
                        })?;
                        let len = packed.data().len();
                        if !table.restore_narrow(packed) {
                            return Err(MlError::SpillFailed {
                                table: index,
                                reason: "the regenerated columns do not have the table's shape",
                            });
                        }
                        if let Some(mem) = mem {
                            mem.held_narrow.fetch_add(len, Relaxed);
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

/// A table's committed columns as held on the host: packed, as records, or
/// eight bytes a cell.
fn held_bytes<F, E>(table: &CommittedTable<'_, F, E>) -> usize
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    table.compact_bytes().unwrap_or_else(|| wide_bytes(table))
}

/// A group's columns put on the card for its commit: a group of wide tables
/// as their field elements, one with a table held narrow (built packed) or as
/// records as [`upload_group`] puts it — packed and widened on the card, or
/// filled there from the records.
fn upload_for_commit<F, E>(group: &[CommittedTable<'_, F, E>]) -> Store
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    if group.iter().any(|t| t.compact_bytes().is_some()) {
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

/// Whether phase B opens each group beside the next group's argue
/// ([`BlockCommitted::with_phase_b_overlap`]). The proof is the same byte for
/// byte either way: each group proves on its own fork `S_post ‖ g`, so only
/// when its openings run moves.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PhaseBOverlap {
    /// One group at a time: its argue, then its openings.
    #[default]
    Off,
    /// A group's openings on a thread of their own, drawing the card's
    /// background streams, while the driver argues the next group; at most one
    /// group's openings in flight, the last group's opened alone. After a
    /// transition in which the card refused a promise, the rest of phase B goes
    /// one group at a time.
    On,
    /// As [`Self::On`], and the openings' thread reads back from the card
    /// through a pinned buffer with a blocking-sync event wait
    /// ([`multilinear::gpu::pinned_readbacks`]): its read-backs leave the
    /// driver's pageable path, where the argument's waited on them. The
    /// argument's own read-backs are unchanged.
    Pinned,
}

/// How a test disturbs phase B's overlap ([`PhaseBOverlap::On`]): each failure
/// must come back as the prove's error — never a panic, never another proof —
/// and a refusal must turn the rest of phase B serial. Places are positions in
/// phase B's order. The production prover takes the default: none.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default)]
pub struct OverlapFaults {
    /// The openings of the group at this place fail, on whichever thread.
    pub open_fails: Option<usize>,
    /// The argue of the group at this place fails.
    pub argue_fails: Option<usize>,
    /// The opening thread of the group at this place panics (test builds).
    pub open_panics: Option<usize>,
    /// The opening thread waits this long before each group's openings.
    pub open_stall: Option<std::time::Duration>,
    /// The transition that argues the group at this place reads as one in
    /// which the card refused a promise.
    pub refusal_at: Option<usize>,
}

/// Every count of a promise the card refused, or of a wait on every stream at
/// once, summed: a transition of phase B's overlap in which it rose turns the
/// rest of phase B serial. The revive's room and an opening's turn refused,
/// the GKR trees and other argue work the card refused, and drain-and-trims.
/// Openings whose factors were built on the host are not among them: a small
/// stack declines the card by policy, with no refusal behind it. Each is 0
/// without a device.
fn phase_b_refusals() -> u64 {
    use multilinear::gpu;
    [
        gpu::room_revive_refusals(),
        gpu::room_turn_refusals(),
        gpu::gkr_tree_refusals(),
        gpu::argue_device_fallbacks(),
        gpu::device_trims(),
    ]
    .into_iter()
    .fold(0u64, u64::saturating_add)
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
    /// Phase B's overlap ([`Self::with_phase_b_overlap`]), and a test's faults
    /// in it.
    overlap: PhaseBOverlap,
    overlap_faults: OverlapFaults,
    /// Phase B stamps its driver thread's schedstat a group turn
    /// ([`Self::with_sched_stamps`]).
    sched: bool,
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
        if let Some(regen) = spill.as_ref().and_then(|spill| spill.regen.as_ref()) {
            regen.set_mem(mem.clone());
        }
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
            let was_wide: Vec<bool> = group.iter().map(|t| t.compact_bytes().is_none()).collect();
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
            overlap: PhaseBOverlap::Off,
            overlap_faults: OverlapFaults::default(),
            sched: false,
        })
    }

    /// Phase B reads its driver thread's CPU time and run-queue delay from
    /// `/proc/thread-self/schedstat` at each group turn ([`GroupStamps::sched_b`]);
    /// off unless set here. A report: the proof is the same.
    pub fn with_sched_stamps(mut self, on: bool) -> Self {
        self.sched = on;
        self
    }

    /// Phase B opens each group beside the next group's argue, or not
    /// ([`PhaseBOverlap`]); `Off` unless set here. The proof is the same.
    pub fn with_phase_b_overlap(mut self, overlap: PhaseBOverlap) -> Self {
        self.overlap = overlap;
        self
    }

    /// A test's faults in phase B's overlap ([`OverlapFaults`]).
    #[doc(hidden)]
    pub fn with_overlap_faults(mut self, faults: OverlapFaults) -> Self {
        self.overlap_faults = faults;
        self
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

    /// By group, the class of the tables it dropped for phase B to rebuild
    /// ([`BlockRegen`]), or `None`: what [`interleaved_order`] orders by.
    /// Refused with the first group that dropped tables of two classes
    /// ([`one_class`]): phase B could not take it in both classes' orders.
    pub fn rebuilt_classes(&self) -> Result<Vec<Option<usize>>, usize> {
        let mut at = 0usize;
        self.sizes
            .iter()
            .enumerate()
            .map(|(g, &size)| {
                let classes = self.spilled[at..at + size]
                    .iter()
                    .flatten()
                    .map(Out::dropped_class);
                at += size;
                one_class(classes).ok_or(g)
            })
            .collect()
    }

    /// Live regeneration, when phase A ran with it.
    pub fn regen(&self) -> Option<&BlockRegen> {
        self.spill.as_ref().and_then(|spill| spill.regen.as_ref())
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
        if let Some(store) = self.spill.as_ref().and_then(|spill| spill.store.as_ref()) {
            store.flush();
            if let Some(handle) = self.spilled.iter().flatten().find_map(Out::handle) {
                return store.corrupt_on_disk(&handle, 0).unwrap_or(false);
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

/// The calling thread's CPU time and run-queue delay so far, in ns, from
/// `/proc/thread-self/schedstat` (Linux); `None` where the file does not read.
/// A kernel without scheduler statistics prints zeros.
fn thread_schedstat() -> Option<(u64, u64)> {
    parse_schedstat(&std::fs::read_to_string("/proc/thread-self/schedstat").ok()?)
}

/// A schedstat line's first two fields: CPU ns and run-queue delay ns.
fn parse_schedstat(text: &str) -> Option<(u64, u64)> {
    let mut fields = text.split_whitespace().map(|w| w.parse::<u64>().ok());
    Some((fields.next()??, fields.next()??))
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
                    // A table built packed is held narrow already, and one
                    // held as records keeps them.
                    let run = (table.compact_bytes().is_none()
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
        + Clone
        + Send,
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
        + Clone
        + Send,
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
    on_group: &(dyn Fn(GroupOpened<'_, F, E>) + Sync),
) -> Result<BlockProved<F, E>, MlError>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    H: WhirHash,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    T: crypto::fiat_shamir::is_transcript::IsTranscript<E>
        + crypto::fiat_shamir::transcript_hash::HasTranscriptHash<Hash = <H as WhirHash>::Transcript>
        + Clone
        + Send,
{
    let order: Vec<usize> = (0..committed.sizes.len()).collect();
    block_prove_in_order::<F, E, T, H>(
        committed, config, transcript, prepared, deviations, fork_of, argue, on_group, &order,
    )
}

/// The groups that hold a table whose columns phase B rebuilds rather than
/// reads back (`rebuilt`, by group) last, each set in group order: an order
/// for [`block_prove_in_order`] that gives the rebuilding the other groups'
/// phase B as a head start. With nothing rebuilt it is the group order.
pub fn rebuilt_last(rebuilt: &[bool]) -> Vec<usize> {
    (0..rebuilt.len())
        .filter(|&g| !rebuilt[g])
        .chain((0..rebuilt.len()).filter(|&g| rebuilt[g]))
        .collect()
}

/// The ranks of a group's tables that drop, `ranks` holding each table's rank
/// in its class (`None` for one no regenerator rebuilds): those of the class of
/// its first droppable table only. Phase B takes each class's groups in that
/// class's rank order, and a group of two classes would have to come last in
/// one and first in the other; the other class's tables in it stay.
fn drop_one_class(ranks: &mut [Option<(usize, u64)>]) {
    let class = ranks.iter().flatten().next().map(|&(class, _)| class);
    for rank in ranks {
        if rank.is_some_and(|(c, _)| Some(c) != class) {
            *rank = None;
        }
    }
}

/// The one class among a group's tables' dropped classes (`None` for a table
/// not dropped), `Some(None)` when none dropped, `None` when two classes did.
fn one_class(classes: impl IntoIterator<Item = Option<usize>>) -> Option<Option<usize>> {
    let mut one = None;
    for class in classes.into_iter().flatten() {
        match one {
            Some(c) if c != class => return None,
            _ => one = Some(class),
        }
    }
    Some(one)
}

/// Phase B's order with two classes rebuilt (D-WHIR-NODISK N3): the groups
/// with nothing dropped first, in group order (they need no regenerator, and
/// give every regenerator a head start); then the rebuilt groups, each class
/// in group order (its rank order), the classes interleaved so each class's
/// share of the `weights` taken so far stays at its share of the whole. Each
/// regenerator then has to keep up with only its share of phase B's pace. With
/// one class it is [`rebuilt_last`].
pub fn interleaved_order(classes: &[Option<usize>], weights: &[u64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..classes.len())
        .filter(|&g| classes[g].is_none())
        .collect();
    let n = classes.iter().flatten().max().map_or(0, |&c| c + 1);
    let lists: Vec<Vec<usize>> = (0..n)
        .map(|c| {
            (0..classes.len())
                .filter(|&g| classes[g] == Some(c))
                .collect()
        })
        .collect();
    let weight = |g: usize| weights.get(g).copied().unwrap_or(0).max(1) as u128;
    let totals: Vec<u128> = lists
        .iter()
        .map(|l| l.iter().map(|&g| weight(g)).sum())
        .collect();
    let mut next = vec![0usize; n];
    let mut taken = vec![0u128; n];
    while (0..n).any(|c| next[c] < lists[c].len()) {
        // The class furthest behind its share: taken_c / total_c the least
        // (compared without division), ties to the lower class.
        let Some(c) = (0..n)
            .filter(|&c| next[c] < lists[c].len())
            .min_by(|&a, &b| (taken[a] * totals[b]).cmp(&(taken[b] * totals[a])))
        else {
            break;
        };
        let g = lists[c][next[c]];
        next[c] += 1;
        taken[c] += weight(g);
        order.push(g);
    }
    order
}

/// A group argued and not yet opened: what its openings read, all of it its
/// own — its fork, its kept tree tops, its columns on the card and the claims
/// its argue settled on — and its share of the proof so far.
struct Argued<F, E, T>
where
    F: IsFFTField + IsPrimeField + 'static,
    E: IsField,
{
    group: usize,
    /// Its place in phase B's order.
    place: usize,
    /// Its first table, and how many it has.
    at: usize,
    size: usize,
    fork: T,
    /// Taken by its openings.
    retired: Option<RetiredStack<F>>,
    store: Store,
    tree_tops: usize,
    points: Vec<Vec<FieldElement<E>>>,
    values: Vec<FieldElement<E>>,
    proofs: Vec<TableProof<E>>,
    argue: Option<BatchedArgue<E>>,
}

/// A group's openings done: its opening, its prepared one when it has one,
/// and what they cost.
struct Opened<F: IsField, E: IsField> {
    opening: StackedProof<F, E>,
    prepared: Option<StackedProof<F, E>>,
    encode: f64,
    open: f64,
    /// What the revive added to the ledger, and the ledger's peak through the
    /// openings: read only when the openings had the ledger's window to
    /// themselves (0 and `None` otherwise).
    open_room: u64,
    open_reserved: Option<u64>,
    /// Stream draws while they ran, `(foreground pool, background pool)`.
    draws: (u64, u64),
}

/// A group's openings on its own fork: its stack revived over its columns (on
/// the card when its store holds them), opened at the points and values its
/// argue settled on, then its prepared stack when it has one. Returns them,
/// and the revived stack for the caller to let go.
///
/// It reads the group's own tables, fork, claims and kept tree tops, nothing
/// another group proves, so phase B may run it on a thread of its own beside
/// the next group's argue ([`PhaseBOverlap::On`]): nothing it calls may read a
/// setting the driver's thread holds per thread. `own_window`: the ledger's
/// window is the openings' alone (one group at a time), so it is opened here
/// and its peak read. `fails`: a test's failed opening.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn open_group<F, E, T, H>(
    argued: &mut Argued<F, E, T>,
    group: &[CommittedTable<'_, F, E>],
    prepared: &[BlockPrepared<'_, F, H>],
    deviations: &[PreparedDeviation],
    config: &ChainConfig,
    own_window: bool,
    fails: bool,
) -> Result<(Opened<F, E>, StackedCommitment<F, H>), MlError>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    H: WhirHash,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    T: crypto::fiat_shamir::is_transcript::IsTranscript<E>
        + crypto::fiat_shamir::transcript_hash::HasTranscriptHash<Hash = <H as WhirHash>::Transcript>,
{
    if fails {
        return Err(MlError::DeviceFailed {
            stage: "a test's failed opening",
        });
    }
    let retired = argued.retired.take().ok_or(MlError::QueryCountMismatch {
        expected: 1,
        got: 0,
    })?;
    let draws = multilinear::gpu::stream_draws();
    let handles = group_columns(group);
    let columns: Vec<&ColumnOf<'_, _>> = handles.iter().collect();
    let store = argued.store.as_ref();
    let t = Instant::now();
    if own_window {
        multilinear::gpu::reset_reserved_window();
    }
    let before_revive = own_window.then(multilinear::gpu::ledger_reserved);
    let stacked = retired.revive::<H, _>(
        &columns,
        store.map(|store| (&**store, ColumnsAt::From(0))),
        config,
    )?;
    let encode = t.elapsed().as_secs_f64();
    let open_room = before_revive.map_or(0, |before| {
        multilinear::gpu::ledger_reserved().saturating_sub(before)
    });
    let t = Instant::now();
    let opening = stacked_eval::prove::<F, E, T, H, _>(
        &stacked,
        &columns,
        store.map(|store| (&**store, 0)),
        &Claimed::PerColumn(&argued.points),
        &argued.values,
        config,
        &mut argued.fork,
    )?;
    // The group's prepared tables, in table order: each opened at its own
    // point for its own prefix, on this fork.
    let (at, points, values) = (argued.at, &argued.points, &argued.values);
    let firsts: Vec<usize> = group
        .iter()
        .scan(0usize, |first, table| {
            let at = *first;
            *first += table.num_committed_columns();
            Some(at)
        })
        .collect();
    let prepared_opening = match prepared.iter().position(|p| p.group == argued.group) {
        Some(k) => {
            let p = &prepared[k];
            let (mut at_points, mut at_values) =
                prepared_claims(&p.tables, at, &firsts, points, values);
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
            Some(stacked_eval::prove::<F, E, T, H, _>(
                p.commitment,
                p.columns,
                None,
                &Claimed::PerColumn(&at_points),
                &at_values,
                config,
                &mut argued.fork,
            )?)
        }
        None => None,
    };
    let open = t.elapsed().as_secs_f64();
    let open_reserved = own_window.then(multilinear::gpu::reserved_window_peak);
    let after = multilinear::gpu::stream_draws();
    Ok((
        Opened {
            opening,
            prepared: prepared_opening,
            encode,
            open,
            open_room,
            open_reserved,
            draws: (
                after.0.saturating_sub(draws.0),
                after.1.saturating_sub(draws.1),
            ),
        },
        stacked,
    ))
}

/// Lets a group go once its openings are done: its tables' columns off the
/// card and out of their widened form — under a spill its packed columns too,
/// and the records of a table held as them, which nothing reads after its
/// openings — its store, and its kept tree tops out of the memory log. The
/// bytes of records let go.
fn release_group<F, E>(
    group: &mut [CommittedTable<'_, F, E>],
    g: usize,
    store: Store,
    tree_tops: usize,
    spilling: bool,
    mem: Option<&BlockMem>,
) -> usize
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    let mut records_released = 0;
    for table in group.iter_mut() {
        table.clear_resident();
        table.drop_widened();
        // While a spill is on, a finished group's packed columns go, and so do
        // the records of a table held as them: nothing reads either after its
        // opening.
        if !spilling {
            continue;
        }
        let released = match table.take_narrow_for_spill() {
            Some(packed) => Some(packed.data().len()),
            None => table.take_records_for_spill().map(|records| {
                records_released += records.bytes();
                records.bytes()
            }),
        };
        if let (Some(bytes), Some(mem)) = (released, mem) {
            mem.held_narrow.fetch_sub(bytes, Relaxed);
        }
    }
    drop(store);
    if let Some(mem) = mem {
        mem.tree_tops.fetch_sub(tree_tops, Relaxed);
        mem.mark(&format!("phase B group {g} end"));
    }
    records_released
}

/// [`block_prove_on_forks_observed`] with phase B taking the groups in
/// `order`, a permutation of their indices. Each group proves on its own fork
/// `S_post ‖ g` and nothing one group proves enters another's transcript, so
/// the proof — each group's share, assembled in group order — is the same in
/// any order; only when each group's columns are needed moves. The next
/// group's columns go up beside this group's argue in `order` too. An `order`
/// that is not a permutation of the groups is refused.
///
/// Under [`PhaseBOverlap::On`] ([`BlockCommitted::with_phase_b_overlap`]) each
/// group's openings run on a thread of their own beside the next group's
/// argue, which keeps each fork's own sequence — argue, openings, prepared
/// opening — so the proof is the same again. `on_group` is then called on that
/// thread, as each group's openings end, in `order` all the same.
#[allow(clippy::too_many_arguments)]
#[doc(hidden)]
pub fn block_prove_in_order<F, E, T, H>(
    committed: BlockCommitted<'_, F, E>,
    config: &ChainConfig,
    transcript: &mut T,
    prepared: &[BlockPrepared<'_, F, H>],
    deviations: &[PreparedDeviation],
    fork_of: &dyn Fn(usize) -> usize,
    argue: &ArgueDeviation,
    on_group: &(dyn Fn(GroupOpened<'_, F, E>) + Sync),
    order: &[usize],
) -> Result<BlockProved<F, E>, MlError>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    H: WhirHash,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    T: crypto::fiat_shamir::is_transcript::IsTranscript<E>
        + crypto::fiat_shamir::transcript_hash::HasTranscriptHash<Hash = <H as WhirHash>::Transcript>
        + Clone
        + Send,
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
        overlap,
        overlap_faults,
        sched,
    } = committed;
    let n = sizes.len();
    let mut seen = vec![false; n];
    if order.len() != n
        || !order
            .iter()
            .all(|&g| g < n && !std::mem::replace(&mut seen[g], true))
    {
        return Err(MlError::QueryCountMismatch {
            expected: n,
            got: order.len(),
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
        .map(|p| (p.group, p.tables.as_slice(), Some(p.columns.len())))
        .collect();
    prepared_shape(&entries, &starts, tables.len(), &|t| {
        tables[t].num_committed_columns()
    })?;
    let entered = Instant::now();
    let derived: Vec<Commitment> = prepared.iter().flat_map(|p| p.commitment.roots()).collect();
    let (z, alpha, beta) = absorb_roots_and_challenge::<E, T>(transcript, &roots, &derived);

    // Each group's share, by group, assembled in group order at the end.
    let mut retired: Vec<Option<RetiredStack<F>>> = groups.into_iter().map(Some).collect();
    let mut group_tables: Vec<Vec<TableProof<E>>> = (0..n).map(|_| Vec::new()).collect();
    let mut group_argue: Vec<Option<BatchedArgue<E>>> = (0..n).map(|_| None).collect();
    let mut group_opening: Vec<Option<StackedProof<F, E>>> = (0..n).map(|_| None).collect();
    let mut group_prepared: Vec<Option<StackedProof<F, E>>> = (0..n).map(|_| None).collect();
    // The next group's columns, uploaded during this group's argument — the
    // card idles through the argument's host glue, and a group's store is a
    // few GiB beside the argument's working set, not beside its codewords.
    let mut pre_uploaded: Option<Store> = None;
    // Under the overlap: the group argued last, whose openings run beside the
    // next group's argue; and whether a transition saw the card refuse a
    // promise, after which every group opens alone.
    let overlapping = overlap != PhaseBOverlap::Off;
    // Under `Pinned`, the openings' thread (a background-class one) reads back
    // through pinned staging; this thread, the argument's, does not.
    let _pinned = (overlap == PhaseBOverlap::Pinned).then(multilinear::gpu::pinned_readbacks);
    let mut prev: Option<Argued<F, E, T>> = None;
    let mut latched = false;
    // The spilled tables read back in phase B's order, two groups' bytes ahead
    // of their uploads.
    let prefetch = spill.as_ref().and_then(|_| {
        let reads: Vec<(ReadPhase, usize, SpilledMain)> = order
            .iter()
            .flat_map(|&g| starts[g]..starts[g] + sizes[g])
            .filter_map(|t| {
                spilled[t]
                    .as_ref()
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
    // The driver thread's schedstat at the start of the turn under way, and
    // its group: each turn's share is closed at the next turn's start.
    let mut turn_sched: Option<((u64, u64), usize)> = None;
    for (p, &g) in order.iter().enumerate() {
        stamps[g].start_b = entered.elapsed().as_secs_f64();
        if sched {
            let now = thread_schedstat();
            if let (Some((then, h)), Some(now)) = (turn_sched, now) {
                stamps[h].sched_b =
                    Some((now.0.saturating_sub(then.0), now.1.saturating_sub(then.1)));
            }
            turn_sched = now.map(|now| (now, g));
        }
        let (at, size) = (starts[g], sizes[g]);
        let next_g = order.get(p + 1).copied();
        let mut fork = group_fork::<E, T>(transcript, fork_of(g));
        if spill.is_some() {
            // This group's columns and the next's, back before their uploads
            // (the next one's goes up beside this group's argue).
            for (first, len) in
                std::iter::once((at, size)).chain(next_g.map(|next| (starts[next], sizes[next])))
            {
                restore_group(
                    first,
                    &mut tables[first..first + len],
                    &mut spilled[first..first + len],
                    prefetch.as_ref(),
                    mem.as_deref(),
                )?;
                // No reader of these groups meets a table whose columns are
                // still out: one whose slot did not come back is refused here,
                // before its upload.
                if let Some(k) = tables[first..first + len]
                    .iter()
                    .position(|t| t.is_spilled())
                {
                    return Err(MlError::SpillFailed {
                        table: first + k,
                        reason: "its columns are still spilled when its group is read",
                    });
                }
            }
        }
        let retired = retired[g].take().ok_or(MlError::QueryCountMismatch {
            expected: n,
            got: order.len(),
        })?;

        let t = Instant::now();
        let store = match pre_uploaded.take() {
            Some(store) => store,
            None => upload_group(&tables[at..at + size]),
        };
        if let Some(store) = &store {
            let mut first = 0usize;
            for table in tables[at..at + size].iter_mut() {
                table.set_resident(store.clone(), first);
                first += table.num_committed_columns();
            }
        }
        stamps[g].upload_b += t.elapsed().as_secs_f64();

        // The group's tables, one after another, each leaving its columns
        // claimed at its own point — or, batched, all of them in one argue —
        // with the next group's upload beside them, and under the overlap the
        // previous group's openings too.
        let t = Instant::now();
        stamps[g].argue_base = multilinear::gpu::ledger_reserved();
        multilinear::gpu::reset_reserved_window();
        let refusals_before = prev.as_ref().map(|_| phase_b_refusals());
        let mut points: Vec<Vec<FieldElement<E>>> = Vec::new();
        let mut values: Vec<FieldElement<E>> = Vec::new();
        let mut proofs: Vec<TableProof<E>> = Vec::new();
        let mut batched_argue: Option<BatchedArgue<E>> = None;
        let all: &[CommittedTable<'_, F, E>] = &tables;
        let group_ref = &all[at..at + size];
        let next = next_g.map(|next| &all[starts[next]..starts[next] + sizes[next]]);
        let roots_ref = &roots;
        let (argue_result, argue_end, next_store, joined, opened, open_paid) =
            std::thread::scope(|scope| {
                let uploader = next.map(|next| scope.spawn(move || upload_group(next)));
                let opener = prev.as_mut().map(|earlier| {
                    let its_tables = &all[earlier.at..earlier.at + earlier.size];
                    scope.spawn(move || -> Result<Opened<F, E>, MlError> {
                        // Every stream this thread draws is a background one,
                        // so the argue's kernels are dispatched first.
                        let _background = multilinear::gpu::background_streams();
                        if let Some(stall) = overlap_faults.open_stall {
                            std::thread::sleep(stall);
                        }
                        #[cfg(any(test, feature = "test-utils"))]
                        if overlap_faults.open_panics == Some(earlier.place) {
                            panic!("a test's opening thread panics");
                        }
                        let fails = overlap_faults.open_fails == Some(earlier.place);
                        let (opened, stacked) = open_group::<F, E, T, H>(
                            earlier, its_tables, prepared, deviations, config, false, fails,
                        )?;
                        on_group(GroupOpened {
                            group: earlier.group,
                            roots: roots_ref,
                            argue: earlier.argue.as_ref(),
                            tables: &earlier.proofs,
                            opening: &opened.opening,
                            prepared: opened.prepared.as_ref(),
                        });
                        drop(stacked);
                        Ok(opened)
                    })
                });
                let argue_result = (|| -> Result<(), MlError> {
                    if overlap_faults.argue_fails == Some(p) {
                        return Err(MlError::DeviceFailed {
                            stage: "a test's failed argue",
                        });
                    }
                    match config.format.argue {
                        ArgueFormat::PerTable => {
                            for table in group_ref.iter() {
                                let (proof, point) =
                                    prove(table, &z, &alpha, &beta, &mut fork, None)?;
                                for _ in 0..table.num_committed_columns() {
                                    points.push(point.clone());
                                }
                                values
                                    .extend(proof.constraint.reduce.column_values.iter().cloned());
                                proofs.push(proof);
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
                            batched_argue = Some(proof);
                        }
                    }
                    Ok(())
                })();
                let argue_end = Instant::now();
                let next_store = uploader.map(|handle| handle.join());
                let joined = argue_end.elapsed().as_secs_f64();
                // What of the previous group's openings outlasted the argue
                // and the upload: the overlap's cost on this thread.
                let waited = Instant::now();
                let opened = opener.map(|handle| handle.join());
                (
                    argue_result,
                    argue_end,
                    next_store,
                    joined,
                    opened,
                    waited.elapsed().as_secs_f64(),
                )
            });
        // The previous group's openings answer first: their error is the one
        // one group at a time would have met first. A panic on their thread
        // is refused, never resumed.
        let opened = match opened {
            None => None,
            Some(Ok(Ok(opened))) => Some(opened),
            Some(Ok(Err(e))) => return Err(e),
            Some(Err(_)) => {
                return Err(MlError::DeviceFailed {
                    stage: "the previous group's openings",
                });
            }
        };
        argue_result?;
        stamps[g].argue = argue_end.saturating_duration_since(t).as_secs_f64();
        stamps[g].argue_reserved = multilinear::gpu::reserved_window_peak();
        if let (Some(next_store), Some(next)) = (next_store, next_g) {
            // The wait for the upload after the argument ended is the next
            // group's upload cost; the rest of it hid behind this argument.
            stamps[next].upload_b += joined;
            pre_uploaded = Some(next_store.map_err(|_| MlError::DeviceFailed {
                stage: "uploading the next group's columns",
            })?);
        }
        if let (Some(opened), Some(earlier)) = (opened, prev.take()) {
            // The previous group's openings ran beside this argue: one window
            // on the ledger for both.
            let h = earlier.group;
            let window = stamps[g].argue_reserved;
            stamps[g].transition_reserved = window;
            let s = &mut stamps[h];
            s.encode = opened.encode;
            s.open = opened.open;
            s.open_room = opened.open_room;
            s.open_reserved = window;
            s.open_paid = open_paid;
            s.overlapped = true;
            s.open_draws = opened.draws;
            group_tables[h] = earlier.proofs;
            group_argue[h] = earlier.argue;
            group_opening[h] = Some(opened.opening);
            group_prepared[h] = opened.prepared;
            stamps[h].records_released = release_group(
                &mut tables[earlier.at..earlier.at + earlier.size],
                h,
                earlier.store,
                earlier.tree_tops,
                spill.is_some(),
                mem.as_deref(),
            );
            stamps[h].end_b = entered.elapsed().as_secs_f64();
            // A transition in which the card refused a promise turns the rest
            // of phase B serial: the work went where it goes without the
            // overlap, and no later transition risks it again.
            if refusals_before.is_some_and(|before| phase_b_refusals() > before)
                || overlap_faults.refusal_at == Some(p)
            {
                latched = true;
            }
        }

        let tree_tops = retired.tree_bytes();
        let mut argued = Argued {
            group: g,
            place: p,
            at,
            size,
            fork,
            retired: Some(retired),
            store,
            tree_tops,
            points,
            values,
            proofs,
            argue: batched_argue,
        };
        if overlapping && !latched && next_g.is_some() {
            // Opened beside the next group's argue.
            prev = Some(argued);
            continue;
        }
        stamps[g].serial_after_refusal = overlapping && latched && next_g.is_some();
        let paid = Instant::now();
        let (opened, stacked) = open_group::<F, E, T, H>(
            &mut argued,
            &tables[at..at + size],
            prepared,
            deviations,
            config,
            true,
            overlap_faults.open_fails == Some(p),
        )?;
        stamps[g].encode = opened.encode;
        stamps[g].open_room = opened.open_room;
        stamps[g].open = opened.open;
        stamps[g].open_reserved = opened.open_reserved.unwrap_or(0);
        stamps[g].open_paid = paid.elapsed().as_secs_f64();
        stamps[g].open_draws = opened.draws;
        on_group(GroupOpened {
            group: g,
            roots: &roots,
            argue: argued.argue.as_ref(),
            tables: &argued.proofs,
            opening: &opened.opening,
            prepared: opened.prepared.as_ref(),
        });
        group_tables[g] = argued.proofs;
        group_argue[g] = argued.argue;
        group_opening[g] = Some(opened.opening);
        group_prepared[g] = opened.prepared;
        drop(stacked);
        stamps[g].records_released = release_group(
            &mut tables[at..at + size],
            g,
            argued.store,
            argued.tree_tops,
            spill.is_some(),
            mem.as_deref(),
        );
        stamps[g].end_b = entered.elapsed().as_secs_f64();
    }
    if sched && let (Some((then, h)), Some(now)) = (turn_sched, thread_schedstat()) {
        stamps[h].sched_b = Some((now.0.saturating_sub(then.0), now.1.saturating_sub(then.1)));
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
    let openings = group_opening
        .into_iter()
        .map(|opening| {
            opening.ok_or(MlError::QueryCountMismatch {
                expected: n,
                got: order.len(),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((
        MultiProof {
            roots,
            tables: group_tables.into_iter().flatten().collect(),
            columns: openings,
            preprocessed: None,
        },
        group_argue.into_iter().flatten().collect(),
        group_prepared.into_iter().flatten().collect(),
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

    /// ★ Under a spill, a finished group lets go of its tables' records as it
    /// does of their packed columns: the table is out (spilled), the records'
    /// last handle is gone, and the bytes come back as the group's records
    /// released; packed columns go too but are not records. With no spill,
    /// nothing is taken and nothing counts.
    #[test]
    fn a_finished_group_lets_go_of_its_records_under_a_spill() {
        use super::release_group;
        use crate::examples::multi_table_lookup::new_cpu_air_with_lookup;
        use crate::multilinear_air::Uniforms;
        use crate::multilinear_table::{CommittedTable, TableLayout};
        use crate::proof::options::ProofOptions;
        use crate::traits::AIR;
        use math::field::{
            element::FieldElement, extensions_goldilocks::Degree3GoldilocksExtensionField as Ext,
            goldilocks::GoldilocksField as Fp,
        };
        use multilinear::records::{KeccakRndOp, Records, RecordsFill, TableRecords};
        use std::sync::Arc;

        #[derive(Debug)]
        struct Ops(Vec<KeccakRndOp>);
        impl TableRecords for Ops {
            fn rows(&self) -> usize {
                8
            }
            fn cols(&self) -> usize {
                5
            }
            fn fill(&self) -> RecordsFill<'_> {
                RecordsFill::KeccakRnd {
                    ops: &self.0,
                    skip: 0,
                }
            }
            fn host_fill(&self) -> Option<NarrowColumns> {
                None
            }
        }
        let air = new_cpu_air_with_lookup(&ProofOptions::default_test_options());
        let layout = || {
            TableLayout::<Fp, Ext>::new(
                air.constraint_program(),
                air.constraints_meta(),
                air.bus_interactions(),
                5,
                3,
                Uniforms::default(),
            )
            .expect("a layout")
        };
        let records = Records(Arc::new(Ops(vec![
            KeccakRndOp {
                timestamp: 3,
                input: [7; 25],
            };
            2
        ])));
        let weak = Arc::downgrade(&records.0);
        let bytes = records.bytes();
        assert!(bytes > 0);
        let held = || CommittedTable::from_records(layout(), records.clone()).expect("a table");
        let packed = || {
            let mut table = CommittedTable::<Fp, Ext>::new(
                air.constraint_program(),
                air.constraints_meta(),
                air.bus_interactions(),
                5,
                3,
                Uniforms::default(),
                |col| {
                    (0..8u64)
                        .map(|r| FieldElement::from(r * 5 + u64::from(col)))
                        .collect()
                },
            )
            .expect("a table");
            assert!(table.pack_on_host(), "packs");
            table
        };

        // No spill: the records stay where they are.
        let mut group = vec![held()];
        assert_eq!(release_group(&mut group, 0, None, 0, false, None), 0);
        assert!(group[0].records().is_some() && !group[0].is_spilled());
        drop(group);

        // A spill: the records go, with their last handle, and are counted;
        // the packed table goes uncounted.
        let mut group = vec![packed(), held()];
        drop(records);
        assert_eq!(release_group(&mut group, 0, None, 0, true, None), bytes);
        assert!(group.iter().all(|t| t.is_spilled()), "both tables let go");
        assert!(weak.upgrade().is_none(), "the records' memory is freed");
    }

    /// A schedstat line reads as its CPU and run-queue delay nanoseconds; a
    /// line that is not one reads as nothing.
    #[test]
    fn a_schedstat_line_reads_as_cpu_and_run_queue_delay() {
        use super::parse_schedstat;
        assert_eq!(parse_schedstat("123456 7890 42\n"), Some((123456, 7890)));
        assert_eq!(parse_schedstat("0 0 0\n"), Some((0, 0)));
        assert_eq!(parse_schedstat("123456\n"), None);
        assert_eq!(parse_schedstat("cpu wait\n"), None);
        assert_eq!(parse_schedstat(""), None);
    }

    /// Two classes rebuilt (N3): the plain groups first, then each class in
    /// group order, interleaved by its share of the weights; one class is
    /// `rebuilt_last`.
    #[test]
    fn interleaved_classes_keep_their_share_and_their_order() {
        use super::interleaved_order;
        let classes = [None, Some(0), Some(0), Some(1), None, Some(1), Some(0)];
        assert_eq!(
            interleaved_order(&classes, &[1; 7]),
            vec![0, 4, 1, 3, 2, 5, 6]
        );
        // A heavy rest group waits until the streamed class has caught up.
        assert_eq!(
            interleaved_order(&[Some(0), Some(0), Some(1), Some(0)], &[1, 1, 3, 1]),
            vec![0, 2, 1, 3]
        );
        // One class: the groups with nothing dropped, then it (rebuilt_last).
        let one = [true, true, false, true, false];
        let as_classes: Vec<Option<usize>> = one.iter().map(|&r| r.then_some(0)).collect();
        assert_eq!(
            interleaved_order(&as_classes, &[5; 5]),
            super::rebuilt_last(&one)
        );
        assert_eq!(interleaved_order(&[None; 3], &[]), vec![0, 1, 2]);
        // Any mix: a permutation, the plain groups first, each class ascending.
        let mut seed = 0x9e37_79b9_u64;
        for _ in 0..200 {
            let n = 1 + (seed % 40) as usize;
            let classes: Vec<Option<usize>> = (0..n)
                .map(|i| {
                    seed = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    match (seed >> 33) % 3 {
                        0 => None,
                        c => Some(c as usize - 1),
                    }
                    .filter(|_| i < n)
                })
                .collect();
            let weights: Vec<u64> = (0..n).map(|i| 1 + (i as u64 * 7919) % 13).collect();
            let order = interleaved_order(&classes, &weights);
            let mut sorted = order.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, (0..n).collect::<Vec<_>>());
            let plain = classes.iter().filter(|c| c.is_none()).count();
            assert!(order[..plain].iter().all(|&g| classes[g].is_none()));
            for c in 0..2 {
                let seq: Vec<usize> = order
                    .iter()
                    .copied()
                    .filter(|&g| classes[g] == Some(c))
                    .collect();
                assert!(seq.windows(2).all(|w| w[0] < w[1]), "class {c} ascending");
            }
        }
    }

    /// One class a group (N3): a group's tables drop in the class of its first
    /// droppable one only, and phase B refuses a group that dropped two.
    #[test]
    fn a_group_drops_in_the_class_of_its_first_droppable_table() {
        use super::{drop_one_class, one_class};
        let mut ranks = [None, Some((0, 7)), Some((1, 0)), Some((0, 8)), Some((1, 1))];
        drop_one_class(&mut ranks);
        assert_eq!(ranks, [None, Some((0, 7)), None, Some((0, 8)), None]);
        let mut ranks = [Some((1, 3)), None, Some((0, 9)), Some((1, 4))];
        drop_one_class(&mut ranks);
        assert_eq!(ranks, [Some((1, 3)), None, None, Some((1, 4))]);
        let mut ranks = [None, None];
        drop_one_class(&mut ranks);
        assert_eq!(ranks, [None, None]);

        assert_eq!(one_class([None, Some(1), None, Some(1)]), Some(Some(1)));
        assert_eq!(one_class([None, None]), Some(None));
        assert_eq!(one_class([]), Some(None));
        assert_eq!(one_class([Some(0), None, Some(1)]), None);
    }

    /// The decision feed a regenerator started before phase A's end reads
    /// (N4c): a table dropped after the wait began is handed over with its
    /// slot; a held table is kept; a table never dropped is kept only once
    /// phase A's end closes the decisions; and a waiter on an undecided table
    /// returns `Stopped` as soon as its class's window closes (a failed or
    /// ended prove), never hanging.
    #[test]
    fn a_regenerator_started_early_waits_for_each_tables_fate() {
        use super::{BlockRegen, Parked, ParkedState, RankDecision};
        use std::sync::Arc;
        use std::time::Duration;
        let words: Vec<u64> = (0..2 * 1024u64).map(|i| i * 7919 % 50_000).collect();
        let packed = || NarrowColumns::pack_row_major(&words, 2).expect("packs");
        let parked = |class: usize, rank: u64| {
            Arc::new(Parked {
                len: packed().data().len() as u64,
                cells: 0,
                rank: Some((class, rank)),
                state: std::sync::Mutex::new(ParkedState::Resident(packed())),
            })
        };
        let regen = BlockRegen::new(Arc::new(|_| None), &[1 << 30, 1 << 30], true);
        let _producers = (regen.take_producer(0), regen.take_producer(1));
        assert!(regen.drops_all_from_now());
        // Dropped after the wait began: handed over with its slot.
        regen.delay_drops(1, Duration::from_millis(300));
        let waiter = {
            let regen = regen.clone();
            std::thread::spawn(move || regen.await_rank(1, 7))
        };
        std::thread::sleep(Duration::from_millis(50));
        regen.parked(parked(1, 7));
        let (decision, waited) = waiter.join().expect("joins");
        assert!(matches!(decision, RankDecision::Dropped(ref slot) if slot.rank() == 7));
        assert!(waited >= Duration::from_millis(200), "waited {waited:?}");
        // Held by its group's class: kept at once.
        regen.held(&[(1, 9)]);
        assert!(matches!(regen.await_rank(1, 9).0, RankDecision::Kept));
        // Never dropped: undecided while phase A runs, kept once it ends.
        let waiter = {
            let regen = regen.clone();
            std::thread::spawn(move || regen.await_rank(1, 11))
        };
        std::thread::sleep(Duration::from_millis(150));
        assert!(!waiter.is_finished(), "kept before phase A's end");
        regen.close_drops();
        assert!(matches!(
            waiter.join().expect("joins").0,
            RankDecision::Kept
        ));
        assert!(regen.decisions_closed());
        // Liveness: a waiter on an undecided table of a class whose window
        // closes (the prove failed) returns at once, Stopped.
        let regen = BlockRegen::new(Arc::new(|_| None), &[1 << 30, 1 << 30], true);
        let waiter = {
            let regen = regen.clone();
            std::thread::spawn(move || regen.await_rank(1, 3))
        };
        std::thread::sleep(Duration::from_millis(100));
        assert!(!waiter.is_finished());
        regen.windows()[1].close("the prove failed");
        let (decision, waited) = waiter.join().expect("joins");
        assert!(matches!(decision, RankDecision::Stopped));
        assert!(waited < Duration::from_secs(2), "waited {waited:?}");
    }

    /// Partial drops: arming drops back nothing and later tables are not
    /// dropped on their own; a table the policy wants out drops, and a
    /// hand-off's drop-back takes the parked candidates in rank order only
    /// until its bytes are on their way. Once settled, what stays is kept, so
    /// a regenerator started early learns every fate as it comes.
    #[test]
    fn partial_drops_take_only_what_is_asked() {
        use super::{BlockRegen, Parked, ParkedState, RankDecision};
        use std::sync::Arc;
        let words: Vec<u64> = (0..2 * 1024u64).map(|i| i * 7919 % 50_000).collect();
        let packed = || NarrowColumns::pack_row_major(&words, 2).expect("packs");
        let len = packed().data().len() as u64;
        let parked = |rank: u64| {
            Arc::new(Parked {
                len,
                cells: 0,
                rank: Some((0, rank)),
                state: std::sync::Mutex::new(ParkedState::Resident(packed())),
            })
        };
        let regen = BlockRegen::new(Arc::new(|_| None), &[1 << 30], false).partial();
        assert!(regen.is_partial());
        let _producer = regen.take_producer(0);
        // Parked before and after arming: candidates either way.
        regen.parked(parked(0));
        regen.parked(parked(3));
        assert!(regen.arm());
        assert!(!regen.arm(), "arms once");
        assert!(!regen.drops_all_from_now());
        regen.parked(parked(1));
        regen.parked(parked(2));
        // The policy wants this one out.
        regen.drop_one(parked(4));
        // A hand-off asking for one table and a byte: the two lowest ranks.
        assert_eq!(regen.drop_back(len + 1), 2 * len);
        regen.close_drops();
        let report = regen.report();
        assert_eq!(report.tables, 3, "{report:?}");
        assert_eq!(report.back_tables, 2, "{report:?}");
        for rank in [0, 1, 4] {
            assert!(
                matches!(regen.await_rank(0, rank).0, RankDecision::Dropped(_)),
                "rank {rank} dropped"
            );
        }
        for rank in [2, 3] {
            assert!(
                matches!(regen.await_rank(0, rank).0, RankDecision::Kept),
                "rank {rank} kept"
            );
        }
        // Asked for exactly one table's bytes: that table and not the next;
        // asked for none: nothing.
        let regen = BlockRegen::new(Arc::new(|_| None), &[1 << 30], false).partial();
        let _producer = regen.take_producer(0);
        regen.parked(parked(0));
        regen.parked(parked(1));
        assert!(regen.arm());
        assert_eq!(regen.drop_back(0), 0, "nothing asked");
        assert_eq!(regen.drop_back(len), len, "one table's bytes");
        regen.close_drops();
        assert!(matches!(regen.await_rank(0, 0).0, RankDecision::Dropped(_)));
        assert!(matches!(regen.await_rank(0, 1).0, RankDecision::Kept));
        // Settled: the candidates left are kept, and so is a table parked
        // after, at once (a waiter returns before phase A's end); their fates
        // are known from now on.
        let fate = |regen: &BlockRegen, rank: u64| {
            let regen = regen.clone();
            let waiter = std::thread::spawn(move || regen.await_rank(0, rank).0);
            let t = std::time::Instant::now();
            while !waiter.is_finished() {
                assert!(
                    t.elapsed() < std::time::Duration::from_secs(5),
                    "rank {rank}'s fate is not known"
                );
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            waiter.join().expect("joins")
        };
        let regen = BlockRegen::new(Arc::new(|_| None), &[1 << 30], false).partial();
        let _producer = regen.take_producer(0);
        regen.parked(parked(0));
        assert!(regen.arm());
        assert!(!regen.drops_all_from_now(), "not before it settles");
        regen.settle();
        assert!(regen.drops_all_from_now());
        assert!(matches!(fate(&regen, 0), RankDecision::Kept));
        regen.parked(parked(1));
        assert!(matches!(fate(&regen, 1), RankDecision::Kept));
        regen.drop_one(parked(2));
        assert!(matches!(fate(&regen, 2), RankDecision::Dropped(_)));
        assert_eq!(regen.drop_back(u64::MAX), 0, "nothing left to drop back");
        regen.close_drops();
        // Not partial: arming drops back every candidate and every later one.
        let regen = BlockRegen::new(Arc::new(|_| None), &[1 << 30], false);
        let _producer = regen.take_producer(0);
        regen.parked(parked(0));
        assert!(regen.arm());
        assert!(regen.drops_all_from_now());
        regen.parked(parked(1));
        regen.close_drops();
        assert_eq!(regen.report().tables, 2);
    }

    /// The keep budget's two halves at arming ([`super::BlockRegen::keep_cells`]),
    /// for the same candidates parked out of rank order (cells as committed,
    /// ranks not): a budget at or above every candidate's cells drops nothing
    /// more than no budget (same drops, same fates, every kept table's bytes
    /// still on the host), and a budget below some drops back exactly the
    /// candidates committed past it, in rank order, at arming only.
    #[test]
    fn the_keep_budget_drops_back_only_what_was_committed_past_it() {
        use super::{BlockRegen, Parked, ParkedState, RankDecision};
        use std::sync::Arc;
        let words: Vec<u64> = (0..2 * 1024u64).map(|i| i * 7919 % 50_000).collect();
        let packed = || NarrowColumns::pack_row_major(&words, 2).expect("packs");
        let len = packed().data().len() as u64;
        let parked = |rank: u64, cells: u64| {
            Arc::new(Parked {
                len,
                cells,
                rank: Some((0, rank)),
                state: std::sync::Mutex::new(ParkedState::Resident(packed())),
            })
        };
        // (rank, cells committed once it was): committed in cells order,
        // ranks shuffled.
        let committed = [(5, 10), (1, 20), (7, 30), (3, 40), (0, 50), (6, 60)];
        // Parks them, arms at 60 cells, parks one more past every budget
        // below, lets the policy drop one and a hand-off drop back one table's
        // bytes; returns the regen, the parked tables and the drop thread's
        // order.
        let run = |budget: Option<u64>| {
            let regen = BlockRegen::new(Arc::new(|_| None), &[1 << 30], false)
                .partial()
                .keep_cells(budget);
            let _producer = regen.take_producer(0);
            let tables: Vec<_> = committed.iter().map(|&(r, c)| parked(r, c)).collect();
            for table in &tables {
                regen.parked(Arc::clone(table));
            }
            assert!(regen.arm_at(60));
            let late = parked(2, 70);
            regen.parked(Arc::clone(&late));
            regen.drop_one(parked(9, 80));
            let back = regen.drop_back(1);
            regen.close_drops();
            let order: Vec<(u64, bool)> = regen
                .lock()
                .dropped
                .iter()
                .map(|((_, rank), _, _, back)| (*rank, *back))
                .collect();
            (regen, tables, late, back, order)
        };
        let fates = |regen: &BlockRegen| -> Vec<bool> {
            (0..10)
                .map(|rank| matches!(regen.await_rank(0, rank).0, RankDecision::Dropped(_)))
                .collect()
        };

        // No budget, and budgets at and above the last candidate's cells: the
        // same drops (the policy's rank 9, the hand-off's lowest rank 0), the
        // same fates, nothing dropped back at arming, and every kept table's
        // columns still the ones parked.
        let (none, _, _, none_back, none_order) = run(None);
        assert_eq!(none_order, vec![(9, false), (0, true)]);
        for budget in [60, 61, u64::MAX] {
            let (regen, tables, late, back, order) = run(Some(budget));
            assert_eq!(order, none_order, "budget {budget}");
            assert_eq!(back, none_back, "budget {budget}");
            assert_eq!(fates(&regen), fates(&none), "budget {budget}");
            let (report, plain) = (regen.report(), none.report());
            assert_eq!(report.keep_back, (0, 0), "budget {budget}");
            assert_eq!(report.keep_cells, Some(budget));
            assert_eq!(
                (
                    report.tables,
                    report.bytes,
                    report.back_tables,
                    report.back_bytes
                ),
                (
                    plain.tables,
                    plain.bytes,
                    plain.back_tables,
                    plain.back_bytes
                ),
                "budget {budget}"
            );
            assert_eq!(
                (report.armed_cells, report.armed_kept),
                (Some(60), (6, 6 * len))
            );
            for table in tables.iter().chain([&late]) {
                if table.rank == Some((0, 0)) {
                    continue;
                }
                let state = table.state.lock().unwrap_or_else(|e| e.into_inner());
                assert!(
                    matches!(&*state, ParkedState::Resident(p) if p.data() == packed().data()),
                    "budget {budget}: rank {:?} left the host",
                    table.rank
                );
            }
        }
        assert_eq!(none.report().keep_back, (0, 0));
        assert_eq!(none.report().keep_cells, None);

        // A budget of 30 cells: the candidates committed past it (cells 40, 50,
        // 60: ranks 3, 0, 6) are dropped back at arming in rank order, before
        // anything else; the one at the budget (rank 7, cells 30) stays, and so
        // does the late one parked after arming, past the budget (rank 2). The
        // hand-off then takes the lowest rank left (1).
        let (regen, tables, late, back, order) = run(Some(30));
        assert_eq!(
            order,
            vec![(0, true), (3, true), (6, true), (9, false), (1, true)]
        );
        assert_eq!(back, len);
        let report = regen.report();
        assert_eq!(report.keep_back, (3, 3 * len), "{report:?}");
        assert_eq!(report.armed_kept, (6, 6 * len));
        let dropped: Vec<u64> = (0..10)
            .filter(|&rank| matches!(regen.await_rank(0, rank).0, RankDecision::Dropped(_)))
            .collect();
        assert_eq!(dropped, vec![0, 1, 3, 6, 9]);
        for table in tables.iter().chain([&late]) {
            let kept = matches!(
                &*table.state.lock().unwrap_or_else(|e| e.into_inner()),
                ParkedState::Resident(_)
            );
            let rank = table.rank.map(|(_, r)| r).unwrap_or_default();
            assert_eq!(kept, !dropped.contains(&rank), "rank {rank}");
        }

        // Not partial: the budget is not used, and arming drops back every
        // candidate as before.
        let regen = BlockRegen::new(Arc::new(|_| None), &[1 << 30], false).keep_cells(Some(30));
        let _producer = regen.take_producer(0);
        for &(r, c) in &committed {
            regen.parked(parked(r, c));
        }
        assert!(regen.arm_at(60));
        regen.close_drops();
        let report = regen.report();
        assert_eq!(
            (report.tables, report.keep_back, report.keep_cells),
            (6, (0, 0), None)
        );
    }

    /// The keep budget relative to the arming point
    /// ([`super::BlockRegen::keep_below`]): arming at 60 cells with 25 below
    /// drops back exactly the candidates committed past 35, in rank order, and
    /// only at arming; with an absolute budget too the lower one holds; a
    /// margin past the arming point drops back every candidate; arming without
    /// the cells committed uses only the absolute budget; and none is today's.
    #[test]
    fn the_relative_keep_budget_drops_back_what_was_committed_in_its_last_cells() {
        use super::{BlockRegen, Parked, ParkedState, RankDecision};
        use std::sync::Arc;
        let words: Vec<u64> = (0..2 * 1024u64).map(|i| i * 7919 % 50_000).collect();
        let packed = || NarrowColumns::pack_row_major(&words, 2).expect("packs");
        let len = packed().data().len() as u64;
        let parked = |rank: u64, cells: u64| {
            Arc::new(Parked {
                len,
                cells,
                rank: Some((0, rank)),
                state: std::sync::Mutex::new(ParkedState::Resident(packed())),
            })
        };
        let committed = [(5, 10), (1, 20), (7, 30), (3, 40), (0, 50), (6, 60)];
        // Parks them, arms (at 60 cells when `at` is set, else without the
        // cells), parks one more past every budget; returns the drop thread's
        // order and the report.
        let run = |absolute: Option<u64>, below: Option<u64>, at: bool| {
            let regen = BlockRegen::new(Arc::new(|_| None), &[1 << 30], false)
                .partial()
                .keep_cells(absolute)
                .keep_below(below);
            let _producer = regen.take_producer(0);
            for &(r, c) in &committed {
                regen.parked(parked(r, c));
            }
            assert!(if at { regen.arm_at(60) } else { regen.arm() });
            regen.parked(parked(2, 70));
            regen.close_drops();
            let order: Vec<u64> = regen
                .lock()
                .dropped
                .iter()
                .map(|((_, rank), _, _, back)| {
                    assert!(*back, "a drop-back");
                    *rank
                })
                .collect();
            let kept: Vec<u64> = (0..10)
                .filter(|&r| matches!(regen.await_rank(0, r).0, RankDecision::Kept))
                .collect();
            (order, kept, regen.report())
        };
        // 25 below 60: past 35 are cells 40, 50, 60 (ranks 3, 0, 6), in rank
        // order; cells 30 stays, and so does the table parked after arming.
        let (order, kept, report) = run(None, Some(25), true);
        assert_eq!(order, vec![0, 3, 6]);
        assert_eq!(kept, vec![1, 2, 4, 5, 7, 8, 9]);
        assert_eq!(
            (report.budget, report.keep_below, report.keep_back),
            (Some(35), Some(25), (3, 3 * len))
        );
        // With an absolute 45 too, the lower (35) holds; with an absolute 25,
        // that one does (past 25: cells 30 as well).
        assert_eq!(run(Some(45), Some(25), true).0, vec![0, 3, 6]);
        assert_eq!(run(Some(25), Some(25), true).0, vec![0, 3, 6, 7]);
        // A margin at or past the arming point drops back every candidate.
        assert_eq!(run(None, Some(60), true).0, vec![0, 1, 3, 5, 6, 7]);
        assert_eq!(run(None, Some(1_000), true).0, vec![0, 1, 3, 5, 6, 7]);
        // Without the cells committed, only the absolute budget.
        assert_eq!(run(None, Some(25), false).0, Vec::<u64>::new());
        assert_eq!(run(Some(45), Some(25), false).0, vec![0, 6]);
        // None: arming drops back nothing.
        let (order, _, report) = run(None, None, true);
        assert!(order.is_empty());
        assert_eq!((report.budget, report.keep_below), (None, None));
    }

    /// ★ A parked table held as its records drops as them: its slot keeps the
    /// records' digest and shape and takes only a checked deposit, the drop
    /// report counts it among the records, and a records table not dropped
    /// leaves its parking as it came.
    #[test]
    fn a_parked_records_table_drops_as_its_records() {
        use super::{BlockRegen, Parked, ParkedState, RankDecision, drop_parked};
        use multilinear::records::{KeccakRndOp, Records, RecordsFill, TableRecords};
        use std::sync::Arc;

        #[derive(Debug)]
        struct Ops(Vec<KeccakRndOp>);
        impl TableRecords for Ops {
            fn rows(&self) -> usize {
                1024
            }
            fn cols(&self) -> usize {
                2
            }
            fn fill(&self) -> RecordsFill<'_> {
                RecordsFill::KeccakRnd {
                    ops: &self.0,
                    skip: 0,
                }
            }
            fn host_fill(&self) -> Option<NarrowColumns> {
                None
            }
        }
        let records = |seed: u64| {
            Records(Arc::new(Ops(vec![
                KeccakRndOp {
                    timestamp: seed,
                    input: [seed; 25],
                };
                3
            ])))
        };
        let held = records(5);
        let parked = Arc::new(Parked {
            len: held.bytes() as u64,
            cells: 0,
            rank: Some((0, 4)),
            state: std::sync::Mutex::new(ParkedState::ResidentRecords(held.clone())),
        });
        let regen = BlockRegen::new(Arc::new(|_| None), &[1 << 30], true);
        let _producer = regen.take_producer(0);
        regen.parked(Arc::clone(&parked));
        let RankDecision::Dropped(slot) = regen.await_rank(0, 4).0 else {
            panic!("dropped");
        };
        assert!(slot.is_records());
        assert_eq!(
            (slot.digest(), slot.rows(), slot.cols()),
            (held.digest(), 1024, 2)
        );
        assert!(parked.dropped().is_some(), "its slot in its place");
        let words: Vec<u64> = (0..2 * 1024u64).map(|i| i * 7919 % 50_000).collect();
        let packed = NarrowColumns::pack_row_major(&words, 2).expect("packs");
        slot.deposit_checked(packed, records(5).digest())
            .expect("columns from the same records");
        regen.close_drops();
        let report = regen.report();
        assert_eq!(
            (report.tables, report.records),
            (1, (1, held.bytes() as u64))
        );
        // Not dropped: a second drop of the same parking finds nothing held.
        let window = &regen.windows()[0];
        assert!(
            drop_parked(window, &parked, 4, None).is_none(),
            "dropped once"
        );
    }

    /// The rebuilt groups go last, each set in group order; nothing rebuilt is
    /// the group order.
    #[test]
    fn rebuilt_groups_go_last() {
        assert_eq!(
            super::rebuilt_last(&[true, true, false, true, false]),
            vec![2, 4, 0, 1, 3]
        );
        assert_eq!(super::rebuilt_last(&[false; 4]), vec![0, 1, 2, 3]);
        assert_eq!(super::rebuilt_last(&[true; 3]), vec![0, 1, 2]);
    }
}
