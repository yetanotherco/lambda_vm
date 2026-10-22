//! The opening's two factors, resident across the groups of rounds.
//!
//! A chained WHIR does not run one sumcheck: it runs a group of rounds, folds
//! its codeword, commits the successor, answers an out-of-domain point and
//! folds the next group's weight into the one it carries. The factors — the
//! message and the weight, each the width of a stacked polynomial — are the
//! same two tables throughout, so they stay here and the host drives them.
//!
//! The message arrives in the base field, which is where it is committed, and
//! is lifted on the way in: the sumcheck runs in one field.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg};

use crate::Result;
use crate::device::backend;
use crate::sumcheck::SumcheckSession;

/// The weight and the message, on device, with the cube they currently span.
pub struct OpeningSession {
    stream: Arc<CudaStream>,
    weight: Arc<CudaSlice<u64>>,
    message: Arc<CudaSlice<u64>>,
    len: usize,
}

impl OpeningSession {
    /// `weight` is interleaved ext3; `message` is the base-field polynomial
    /// being opened, lifted here.
    pub fn new(weight: &[u64], message: &[u64]) -> Result<Self> {
        assert!(weight.len().is_multiple_of(3), "three u64 per ext3 element");
        let len = weight.len() / 3;
        let be = backend()?;
        let stream = be.next_stream();
        let weight_dev = crate::device::htod_or_trim(&stream, weight)?;
        Self::with_weight(stream, weight_dev, len, message)
    }

    /// The same with the weight built here: each share is a column's subcube
    /// offset, the point it is claimed at, and the scale it carries, and its
    /// cells are `scale·eq(point, ·)`. The gaps stay zero.
    ///
    /// A stacked polynomial's weight is as wide as the polynomial — writing it
    /// on the host and sending it costs more than the rounds that read it.
    pub fn from_shares(
        shares: &[(usize, Vec<u64>, [u64; 3])],
        len: usize,
        message: &[u64],
    ) -> Result<Self> {
        let be = backend()?;
        let stream = be.next_stream();
        // Zeroed because the gaps between the shares' subcubes are part of the
        // weight and nothing writes them.
        let mut weight = crate::device::alloc_zeros_or_trim::<u64>(&stream, len * 3)?;
        crate::sumcheck::eq_expand_shares_ext3(&stream, &mut weight, shares)?;
        Self::with_weight(stream, weight, len, message)
    }

    /// The same again with the message given as the columns it is stacked from,
    /// so it is never assembled on the host — `(column, offset in elements)`.
    pub fn from_shares_and_parts(
        shares: &[(usize, Vec<u64>, [u64; 3])],
        len: usize,
        parts: &[(&[u64], usize)],
    ) -> Result<Self> {
        Self::from_shares_and_source(shares, len, |stream, base| {
            for (column, offset) in parts {
                let mut at = base.slice_mut(*offset..*offset + column.len());
                stream.memcpy_htod(*column, &mut at)?;
            }
            Ok(())
        })
    }

    /// The same for parts the card already holds: `(column index, offset)` into
    /// the epoch's columns, so the message is assembled there instead of
    /// crossing the bus a second time.
    pub fn from_shares_and_resident(
        shares: &[(usize, Vec<u64>, [u64; 3])],
        len: usize,
        store: &crate::columns::DeviceColumns,
        parts: &[(usize, usize)],
    ) -> Result<Self> {
        Self::from_shares_and_source(shares, len, |stream, base| {
            for (column, offset) in parts {
                store.copy_into(*column, base, *offset, stream)?;
            }
            Ok(())
        })
    }

    fn from_shares_and_source(
        shares: &[(usize, Vec<u64>, [u64; 3])],
        len: usize,
        write: impl FnOnce(&Arc<CudaStream>, &mut CudaSlice<u64>) -> Result<()>,
    ) -> Result<Self> {
        let be = backend()?;
        let stream = be.next_stream();
        let mut weight = crate::device::alloc_zeros_or_trim::<u64>(&stream, len * 3)?;
        crate::sumcheck::eq_expand_shares_ext3(&stream, &mut weight, shares)?;
        // Zeroed because what the parts do not cover is the stacking's padding.
        let mut base = crate::device::alloc_zeros_or_trim::<u64>(&stream, len)?;
        write(&stream, &mut base)?;
        Self::lift_into(stream, weight, len, base)
    }

    fn with_weight(
        stream: Arc<CudaStream>,
        weight_dev: CudaSlice<u64>,
        len: usize,
        message: &[u64],
    ) -> Result<Self> {
        assert_eq!(message.len(), len, "the message spans the weight's cube");
        let base = crate::device::htod_or_trim(&stream, message)?;
        Self::lift_into(stream, weight_dev, len, base)
    }

    fn lift_into(
        stream: Arc<CudaStream>,
        weight_dev: CudaSlice<u64>,
        len: usize,
        base: CudaSlice<u64>,
    ) -> Result<Self> {
        assert!(len.is_power_of_two(), "the cube is a power of two");

        let be = backend()?;
        // SAFETY: the kernel writes every element it is sized for.
        let mut lifted = unsafe { stream.alloc::<u64>(len * 3) }?;
        let count = len as u64;
        unsafe {
            stream
                .launch_builder(&be.mle_lift_base_ext3)
                .arg(&base)
                .arg(&count)
                .arg(&mut lifted)
                .launch(LaunchConfig::for_num_elems(len as u32))?;
        }

        Ok(Self {
            stream,
            weight: Arc::new(weight_dev),
            message: Arc::new(lifted),
            len,
        })
    }

    /// A session over factors already on the card, both extension-valued and
    /// `len` wide — what [`LeanRound0::materialize`] hands over once its
    /// rounds have bound the first variables.
    pub fn from_tables(
        stream: Arc<CudaStream>,
        weight: CudaSlice<u64>,
        message: CudaSlice<u64>,
        len: usize,
    ) -> Self {
        assert!(len.is_power_of_two(), "the cube is a power of two");
        assert!(
            weight.len() >= len * 3 && message.len() >= len * 3,
            "the tables span the cube"
        );
        Self {
            stream,
            weight: Arc::new(weight),
            message: Arc::new(message),
            len,
        }
    }

    /// What a session over `2^num_vars` built from shares holds — the weight
    /// and the lifted message, three limbs each — and the base-field staging it
    /// builds the message through, which is freed once the lift has run.
    pub const fn planned_bytes(num_vars: usize) -> (u64, u64) {
        (48u64 << num_vars, 8u64 << num_vars)
    }

    /// Device bytes the two tables occupy — their allocations, which the rounds
    /// fold in place and never shrink.
    pub fn device_bytes(&self) -> u64 {
        ((self.weight.len() + self.message.len()) * 8) as u64
    }

    /// The weight's and the message's live cube, `len` ext3 values each, read
    /// back — for the parity tests against the lean opening.
    pub fn tables_to_host(&self) -> Result<(Vec<u64>, Vec<u64>)> {
        let weight = self
            .stream
            .clone_dtoh(&self.weight.slice(0..self.len * 3))?;
        let message = self
            .stream
            .clone_dtoh(&self.message.slice(0..self.len * 3))?;
        self.stream.synchronize()?;
        Ok((weight, message))
    }

    /// Cube indices the factors still span.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn num_vars(&self) -> usize {
        self.len.trailing_zeros() as usize
    }

    /// A sumcheck over `[weight, message]` as they stand.
    ///
    /// The rounds fold them in place, so the caller tells this session how far
    /// they got with [`bound`](Self::bound).
    pub fn sumcheck(
        &self,
        nodes: &[u64],
        consts: &[u64],
        num_slots: usize,
        root_slot: u32,
    ) -> Result<SumcheckSession> {
        let addresses = {
            let (weight, _weight_guard) = self.weight.device_ptr(&self.stream);
            let (message, _message_guard) = self.message.device_ptr(&self.stream);
            [weight, message]
        };
        SumcheckSession::from_device(
            self.stream.clone(),
            &addresses,
            self.len,
            vec![self.weight.clone(), self.message.clone()],
            nodes,
            consts,
            num_slots,
            root_slot,
        )
    }

    /// Records that `rounds` variables have been bound by a sumcheck over
    /// these factors.
    pub fn bound(&mut self, rounds: usize) {
        self.len >>= rounds;
    }

    /// The message's value at `point`, which the out-of-domain answer needs.
    ///
    /// Folds a copy: the message is the next group's factor and the rounds
    /// have not bound this point.
    pub fn evaluate_message(&self, point: &[u64]) -> Result<[u64; 3]> {
        assert_eq!(point.len(), self.num_vars() * 3, "the point spans the cube");
        let mut copy = self.stream.alloc_zeros::<u64>(self.len * 3)?;
        {
            let source = self.message.slice(0..self.len * 3);
            self.stream.memcpy_dtod(&source, &mut copy)?;
        }
        crate::sumcheck::evaluate_resident_ext3(&self.stream, &mut copy, self.len, point)
    }

    /// The successor a `levels`-level fold of `codeword` would make, encoded
    /// from the message as these factors have bound it
    /// ([`crate::whir::encode_successor`]). Reads a copy: the message stays the
    /// next group's factor.
    pub fn encode_successor(
        &self,
        codeword: &crate::whir::DeviceCodeword,
        levels: usize,
    ) -> Result<crate::whir::DeviceCodeword> {
        crate::whir::encode_successor(codeword, levels, &self.stream, &self.message, self.len)
    }

    /// [`Self::encode_successor`] of a codeword of `elements` values that is
    /// not held, inside `room` (`crate::whir::encode_successor_sized`).
    pub fn encode_successor_sized(
        &self,
        elements: usize,
        room: &std::sync::Arc<crate::device::DeviceReservation>,
        levels: usize,
    ) -> Result<crate::whir::DeviceCodeword> {
        crate::whir::encode_successor_sized(
            elements,
            room,
            levels,
            &self.stream,
            &self.message,
            self.len,
        )
    }

    /// `weight += scale · eq(point, ·)`, the weight the next group carries.
    pub fn add_scaled_eq(&self, point: &[u64], scale: &[u64]) -> Result<()> {
        assert_eq!(point.len(), self.num_vars() * 3, "the point spans the cube");
        assert_eq!(scale.len(), 3, "an ext3 scale");
        let be = backend()?;
        let eq = crate::sumcheck::eq_table_ext3(&self.stream, point, self.len)?;
        let scale_dev = crate::device::htod_or_trim(&self.stream, scale)?;
        let count = self.len as u64;
        // The kernel writes through a shared reference, the way the tree's
        // layers are folded: what orders these is the stream, and the weight
        // has one owner.
        unsafe {
            self.stream
                .launch_builder(&be.add_scaled_ext3)
                .arg(self.weight.as_ref())
                .arg(&eq)
                .arg(&count)
                .arg(&scale_dev)
                .launch(LaunchConfig::for_num_elems(self.len as u32))?;
        }
        Ok(())
    }
}

/// Nodes a lean round evaluates at (`WHIR_LEAN_MAX_T` in `whir_fold.cu`): the
/// opening's product is degree two, so two.
const LEAN_MAX_T: usize = 4;
/// The partial-sum grid a lean round is capped at, as the sumcheck's rounds are.
const LEAN_MAX_GRID: u32 = 4096;
const LEAN_BLOCK: u32 = 256;
/// Share ids are `u16` on the card, and `0xFFFF` marks a position no column
/// covers.
const LEAN_MAX_SHARES: usize = 0xFFFF;

/// Rounds a stacked opening runs over its shares before it materialises its
/// factors at `2^(n − rounds)`, when `LAMBDA_VM_WHIR_LEAN_ROUNDS` does not say
/// otherwise. Every setting proves the same values.
///
/// 3, not the 6 WBATCH chose at the 2^25 stack as a memory lever: at today's
/// 2^27 stack each lean round is a pass over the whole stack, and three of them
/// plus the materialisation cost less than six. ULTRA 129 (D-SPEED §12):
/// median phase-B openings −2.74 s (t −141), whole −2.54 s, the top proof's
/// bytes unchanged at 6, 3 and 2, no refusals; the open peak +672 MiB (20.5 of
/// 25.1 GiB usable). `LAMBDA_VM_WHIR_LEAN_ROUNDS=6` is the way back.
pub const LEAN_ROUNDS_DEFAULT: usize = 3;

/// `LAMBDA_VM_WHIR_LEAN_ROUNDS`'s value read: a count, clamped to the widest
/// fold ([`crate::whir::FUSED_MAX_FOLD`]); unset or not a number is
/// [`LEAN_ROUNDS_DEFAULT`].
pub fn lean_rounds_setting(value: Option<&str>) -> usize {
    value
        .and_then(|v| v.trim().parse::<usize>().ok())
        .map(|v| v.min(crate::whir::FUSED_MAX_FOLD))
        .unwrap_or(LEAN_ROUNDS_DEFAULT)
}
/// The round scratch a lean opening keeps: the nodes, one partial per node
/// and block, and the sums.
const LEAN_SCRATCH_BYTES: u64 =
    ((LEAN_MAX_T * 3 + LEAN_MAX_T * LEAN_MAX_GRID as usize * 3 + LEAN_MAX_T * 3) * 8) as u64;

/// Slot tables a tile-form opening may hold (`WHIR_TILE_MAX_SLOTS` in
/// `whir_fold.cu`).
pub const TILE_MAX_SLOTS: usize = 8;

/// ★ The lean rounds from ONE pass over the message: the tile form
/// (`whir_tile_*` in `whir_fold.cu`, where the identity is written out).
///
/// The share form walks every stack position in every round. Here the stack is
/// cut into tiles of `2^tile_bits` positions — every share at least that long,
/// its point's `eq` table split there — so the weight inside a tile is one
/// number times a small table its class shares. One pass over the message
/// reduces each tile to a value per slot (`Φ`), one multiply a tile gives the
/// weight's (`Ω`), and every round then reads those tables, `2^tile_bits`
/// times smaller than the stack. The bound tables are written from `Ω` and
/// the classes' tables, the message through a `u16` a tile: the share form's
/// map of a `u16` a position is never built. The same field elements as the
/// share form's rounds and tables; the bound message raw-identical.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileSpec {
    /// A tile is `2^tile_bits` stack positions: every share's `lo_bits`.
    pub tile_bits: usize,
    /// The rounds the opening runs before it writes its tables. A column of
    /// tiles is the `2^lean_rounds` tiles those rounds bind together.
    pub lean_rounds: usize,
    /// Slot tables held: at least the classes (distinct `lo_at`) any column
    /// of tiles meets.
    pub slots: usize,
    /// The pass over the message as one block a tile (a warp reads consecutive
    /// words) rather than one thread a tile. The same values.
    pub wide: bool,
    /// Every round also run in share form and compared
    /// ([`tile_check_counts`]), and the bound tables written by the share
    /// form and compared with the tile form's write
    /// ([`tile_table_check_counts`]): a difference is counted, and the share
    /// form's values are the ones sent and kept, so the proof is today's
    /// either way. The share form's map is built for it, so a checking
    /// opening holds the map and the tile form at once.
    pub check: bool,
}

impl TileSpec {
    /// Device bytes the tile form holds through its rounds: a `u16` per tile,
    /// each column's slots, `Ω` and `Φ` (three limbs a tile and slot each),
    /// and a challenge.
    pub const fn device_bytes(&self, num_vars: usize) -> u64 {
        self.write_bytes(num_vars) + ((24 * self.slots as u64) << (num_vars - self.tile_bits))
    }

    /// What it keeps beside the bound tables while it writes them: all of
    /// the above but `Φ`, which is freed first.
    pub const fn write_bytes(&self, num_vars: usize) -> u64 {
        let tiles = 1u64 << (num_vars - self.tile_bits);
        let columns = tiles >> self.lean_rounds;
        let slots = self.slots as u64;
        2 * tiles + 8 * slots * columns + 24 * slots * tiles + 24
    }

    /// Whether a stack of `2^num_vars` takes this form: the rounds bind tile
    /// variables only, the slots are ones the kernels have room for, and the
    /// opening never holds more than the share form does when it writes its
    /// tables — the position map (a `u16` a position, which the tile form
    /// does not allocate outside [`check`](Self::check)) beside the bound
    /// tables (`48 · 2^(num_vars − lean_rounds)` bytes). Through the rounds
    /// ALL the form holds is no larger than those tables, which are allocated
    /// once `Φ` is freed; through the write what it keeps is no larger than
    /// the map.
    pub const fn fits(&self, num_vars: usize) -> bool {
        self.tile_bits <= num_vars
            && self.lean_rounds >= 1
            && self.lean_rounds <= num_vars - self.tile_bits
            && self.slots >= 1
            && self.slots <= TILE_MAX_SLOTS
            && self.device_bytes(num_vars) <= 48u64 << (num_vars - self.lean_rounds)
            && self.write_bytes(num_vars) <= 2u64 << num_vars
    }
}

/// What `LAMBDA_VM_WHIR_LEAN_TILES` asks of a stacked opening's lean rounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeanTiles {
    /// The share form: `whir_lean_round` over every position, every round
    /// (`0`).
    Off,
    /// The tile form ([`TileSpec`]), one thread a tile (`1`).
    On,
    /// The tile form with the pass over the message one block a tile
    /// (`wide`, and [`LEAN_TILES_DEFAULT`]).
    Wide,
    /// The tile form, every round also run in share form and compared; the
    /// share form's values are the ones sent.
    Check,
}

/// A stacked opening's lean rounds when `LAMBDA_VM_WHIR_LEAN_TILES` is unset:
/// the tile form, its pass over the message one block a tile. The same values
/// as the share form. Against the share form on one card (a 5090 held at
/// 400 W, a 9950X), two runs of each: the whole median block −5.6 s and the
/// whole large block −13.0 s, the openings' sumcheck a quarter of its
/// seconds, an opening in tile form 0.23 GiB lighter on the card. At 2^27 a
/// block a tile was the faster pass where a column of tiles meets one class
/// (most chains), a thread a tile where it meets two or more. `=0` is the way
/// back.
pub const LEAN_TILES_DEFAULT: LeanTiles = LeanTiles::Wide;

/// `LAMBDA_VM_WHIR_LEAN_TILES`'s value read: unset, [`LEAN_TILES_DEFAULT`];
/// `1` the tile form a thread a tile, `wide` a block a tile, `check` the tile
/// form compared with the share form on every round; `0` or anything else is
/// the share form.
pub fn lean_tiles_setting(value: Option<&str>) -> LeanTiles {
    match value.map(str::trim) {
        None => LEAN_TILES_DEFAULT,
        Some("1") => LeanTiles::On,
        Some("wide") => LeanTiles::Wide,
        Some("check") => LeanTiles::Check,
        _ => LeanTiles::Off,
    }
}

static TILE_CHECKED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static TILE_MISMATCHED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Rounds a checking tile-form opening ([`TileSpec::check`]) compared with
/// the share form since the process started, and how many differed. A round
/// that differed still sent the share form's values.
pub fn tile_check_counts() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (TILE_CHECKED.load(Relaxed), TILE_MISMATCHED.load(Relaxed))
}

static LEAN_PEAKS: [std::sync::atomic::AtomicU64; 3] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// ★ The most one lean opening held on the card at once since the process
/// started, by form — `[share, tile, checking]` — counted from its own
/// allocations: once it is built (its rounds allocate nothing that lasts)
/// and while its bound tables are written. What a block gate reads to show
/// the tile form holds no more than the share form.
pub fn lean_peak_bytes() -> [u64; 3] {
    use std::sync::atomic::Ordering::Relaxed;
    [
        LEAN_PEAKS[0].load(Relaxed),
        LEAN_PEAKS[1].load(Relaxed),
        LEAN_PEAKS[2].load(Relaxed),
    ]
}

static TILE_TABLES_CHECKED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static TILE_TABLES_DIFFERING: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Openings whose bound tables a checking opening ([`TileSpec::check`])
/// compared with the tile form's write since the process started, and the
/// positions where either factor differed. The tables kept are the share
/// form's either way.
pub fn tile_table_check_counts() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (
        TILE_TABLES_CHECKED.load(Relaxed),
        TILE_TABLES_DIFFERING.load(Relaxed),
    )
}

static TILE_CHECK_FAULT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Moves every tile-form round value a checking opening compares by one, and
/// one entry of the bound weight it compares the tile form's write with,
/// before the comparison and nowhere else — for the test that shows the
/// comparisons count a difference. What the opening sends and keeps is
/// unchanged. Not for production callers.
#[doc(hidden)]
pub fn force_tile_check_fault(on: bool) {
    TILE_CHECK_FAULT.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Whether `shares` are a stack the tile form reads under `spec`: the spec
/// fits, every share is split at the tile, is at least a tile long and sits at
/// a multiple of its own size, and the slots cover the classes a column of
/// tiles can meet.
fn tiles_read(shares: &[LeanShare], num_vars: usize, spec: &TileSpec) -> bool {
    if !spec.fits(num_vars) {
        return false;
    }
    let mut classes = std::collections::BTreeSet::new();
    for share in shares {
        if share.lo_bits != spec.tile_bits
            || share.num_vars < spec.tile_bits
            || share.stack_offset & ((1usize << share.num_vars) - 1) != 0
        {
            return false;
        }
        classes.insert(share.lo_at);
    }
    spec.slots >= classes.len().min(1usize << spec.lean_rounds)
}

/// The tile form's device state.
struct Tiles {
    spec: TileSpec,
    tilemap: CudaSlice<u16>,
    slot_lo: CudaSlice<u64>,
    omega: CudaSlice<u64>,
    /// `None` once the rounds are spent: the write does not read it.
    phi: Option<CudaSlice<u64>>,
    r_dev: CudaSlice<u64>,
    /// Challenges the slot tables are bound at. The rest of the opening's
    /// are folded in before the next round reads them.
    folded: usize,
}

impl Tiles {
    /// The tile map, each column's slots and `Ω`, and the one pass over the
    /// message for `Φ`, queued on `stream`.
    #[allow(clippy::too_many_arguments)]
    fn build(
        stream: &Arc<CudaStream>,
        spec: TileSpec,
        num_vars: usize,
        shares: &CudaSlice<u64>,
        eq: &CudaSlice<u64>,
        data: &CudaSlice<u64>,
        starts: &[u64],
        ends: &[u64],
    ) -> Result<Self> {
        let be = backend()?;
        let nu = num_vars - spec.tile_bits;
        let tiles = 1usize << nu;
        let columns = tiles >> spec.lean_rounds;
        let tile_starts: Vec<u64> = starts.iter().map(|s| s >> spec.tile_bits).collect();
        let tile_ends: Vec<u64> = ends.iter().map(|e| e >> spec.tile_bits).collect();
        let starts_dev = crate::device::htod_or_trim(stream, &tile_starts)?;
        let ends_dev = crate::device::htod_or_trim(stream, &tile_ends)?;
        // SAFETY: each kernel below writes every element of what it is handed.
        let mut tilemap = unsafe { crate::device::alloc_or_trim::<u16>(stream, tiles) }?;
        let mut slot_lo =
            unsafe { crate::device::alloc_or_trim::<u64>(stream, columns * spec.slots) }?;
        let mut omega =
            unsafe { crate::device::alloc_or_trim::<u64>(stream, spec.slots * tiles * 3) }?;
        let mut phi =
            unsafe { crate::device::alloc_or_trim::<u64>(stream, spec.slots * tiles * 3) }?;
        let r_dev = crate::device::alloc_zeros_or_trim::<u64>(stream, 3)?;
        let grid_for = |work: usize| {
            (work as u64)
                .div_ceil(LEAN_BLOCK as u64)
                .min(LEAN_MAX_GRID as u64) as u32
        };
        let count = starts.len() as u32;
        let (nu_arg, lean_arg, u_arg, slots_arg) = (
            nu as u32,
            spec.lean_rounds as u32,
            spec.tile_bits as u32,
            spec.slots as u32,
        );
        unsafe {
            stream
                .launch_builder(&be.whir_lean_colmap)
                .arg(&mut tilemap)
                .arg(&(tiles as u64))
                .arg(&starts_dev)
                .arg(&ends_dev)
                .arg(&count)
                .launch(LaunchConfig {
                    grid_dim: (grid_for(tiles).max(1), 1, 1),
                    block_dim: (LEAN_BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                })?;
            stream
                .launch_builder(&be.whir_tile_plan)
                .arg(&tilemap)
                .arg(shares)
                .arg(eq)
                .arg(&nu_arg)
                .arg(&lean_arg)
                .arg(&u_arg)
                .arg(&slots_arg)
                .arg(&mut slot_lo)
                .arg(&mut omega)
                .launch(LaunchConfig {
                    grid_dim: (grid_for(columns).max(1), 1, 1),
                    block_dim: (LEAN_BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                })?;
            if spec.wide {
                // One block a tile at a time, its threads the tile's cells in
                // turn: never more threads than cells.
                let block = LEAN_BLOCK.min(1u32 << spec.tile_bits.min(31));
                stream
                    .launch_builder(&be.whir_tile_contract_wide)
                    .arg(&tilemap)
                    .arg(shares)
                    .arg(eq)
                    .arg(data)
                    .arg(&slot_lo)
                    .arg(&nu_arg)
                    .arg(&lean_arg)
                    .arg(&u_arg)
                    .arg(&slots_arg)
                    .arg(&mut phi)
                    .launch(LaunchConfig {
                        grid_dim: ((tiles as u64).min(LEAN_MAX_GRID as u64) as u32, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: block * 3 * 8,
                    })?;
            } else {
                stream
                    .launch_builder(&be.whir_tile_contract)
                    .arg(&tilemap)
                    .arg(shares)
                    .arg(eq)
                    .arg(data)
                    .arg(&slot_lo)
                    .arg(&nu_arg)
                    .arg(&lean_arg)
                    .arg(&u_arg)
                    .arg(&slots_arg)
                    .arg(&mut phi)
                    .launch(LaunchConfig {
                        grid_dim: (grid_for(tiles).max(1), 1, 1),
                        block_dim: (LEAN_BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
            }
        }
        Ok(Self {
            spec,
            tilemap,
            slot_lo,
            omega,
            phi: Some(phi),
            r_dev,
            folded: 0,
        })
    }

    fn device_bytes(&self) -> u64 {
        let phi = self.phi.as_ref().map_or(0, CudaSlice::len);
        (self.tilemap.len() * 2
            + (self.slot_lo.len() + self.omega.len() + phi + self.r_dev.len()) * 8) as u64
    }
}

/// One column of a stacked polynomial as the lean rounds read it: where it sits
/// in the stack, how many variables it spans, its weight's scale, and where its
/// point's `eq` table sits in the shared buffer, in two halves.
///
/// `eq(z, row) = hi[row >> lo_bits] · lo[row & (2^lo_bits − 1)]`: the high half
/// is the table of the point's first `num_vars − lo_bits` coordinates, the low
/// half of the rest, each indexed with its most significant bit on its first
/// coordinate, as every `eq` table here is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LeanShare {
    /// The first stack index the column occupies.
    pub stack_offset: usize,
    /// Its variables: it spans `2^num_vars` stack indices.
    pub num_vars: usize,
    /// Variables in the low half of its point's `eq` table.
    pub lo_bits: usize,
    /// Where the high and the low half start in the `eq` buffer, in ext3
    /// elements.
    pub hi_at: usize,
    pub lo_at: usize,
    /// The weight's scale, as ext3 limbs.
    pub scale: [u64; 3],
}

/// Where a lean opening reads the message.
pub enum LeanMessage<'a> {
    /// In the epoch's resident columns: `(store column, stack offset)` per
    /// part. Nothing is copied.
    Resident {
        store: &'a crate::columns::DeviceColumns,
        parts: &'a [(usize, usize)],
    },
    /// Host columns at their stack offsets, staged once into a base-field
    /// buffer the width of the stack — what the materialised path uploads too,
    /// before lifting it.
    Parts(&'a [(&'a [u64], usize)]),
}

/// A WHIR chain's first rounds over its factors in SHARE form
/// (`whir_lean_*` in `whir_fold.cu`).
///
/// The materialised opening ([`OpeningSession`]) holds the weight and the
/// lifted message at the stack's full width — `2 · 2^n` extension values, 1.5×
/// the committed base codeword, plus the base staging while it is built — for
/// rounds that bind the top variables first. Here the weight stays a list of
/// shares and small `eq` tables, the message is read where the columns lie, a
/// `u16` per stack index says which column holds it, and each round's
/// evaluations are computed from those with the challenges bound so far. The
/// factors exist only once [`materialize`](Self::materialize) writes them at
/// `2^(n − bound)`.
///
/// ★ The evaluations and the bound tables EQUAL the materialised path's
/// (`tests/whir_lean.rs`): the same field values by a different sequence of
/// operations, so raw limbs agree whenever both are canonical — the parity the
/// device path already has with the host.
pub struct LeanRound0 {
    stream: Arc<CudaStream>,
    /// The stack's variables.
    num_vars: usize,
    /// The share holding each position: `None` for an opening in tile form
    /// that does not check itself, which never reads it.
    colmap: Option<CudaSlice<u16>>,
    shares: CudaSlice<u64>,
    eq: CudaSlice<u64>,
    data: Arc<CudaSlice<u64>>,
    /// Whether `data` is a staging buffer this opening allocated.
    staged: bool,
    /// The challenges bound so far, three limbs each.
    bound: Vec<u64>,
    t_dev: CudaSlice<u64>,
    t_host: Vec<u64>,
    partials: CudaSlice<u64>,
    sums: CudaSlice<u64>,
    /// The rounds' tables in tile form ([`TileSpec`]), when the opening was
    /// built with one; `None` is the share form.
    tiles: Option<Tiles>,
}

impl LeanRound0 {
    /// `Ok(None)` when the shares do not describe a stack this can read — out
    /// of order, overlapping, past the end, more than a `u16` can name, or a
    /// part missing for one of them — and the caller materialises instead.
    pub fn new(
        shares: &[LeanShare],
        eq: &[u64],
        num_vars: usize,
        message: LeanMessage<'_>,
    ) -> Result<Option<Self>> {
        Self::with_tiles(shares, eq, num_vars, message, None)
    }

    /// [`new`](Self::new), with the rounds in tile form when `tiles` is given
    /// ([`TileSpec`]). `Ok(None)` too when the shares are not a stack the
    /// tile form reads under that spec — a share shorter than a tile, split
    /// elsewhere or off its own alignment, more classes than slots, a form
    /// larger than the tables written after it — and the caller opens in
    /// share form instead.
    pub fn with_tiles(
        shares: &[LeanShare],
        eq: &[u64],
        num_vars: usize,
        message: LeanMessage<'_>,
        tiles: Option<TileSpec>,
    ) -> Result<Option<Self>> {
        let len = 1usize << num_vars;
        if shares.is_empty() || shares.len() >= LEAN_MAX_SHARES || eq.is_empty() {
            return Ok(None);
        }
        if let Some(spec) = &tiles
            && !tiles_read(shares, num_vars, spec)
        {
            return Ok(None);
        }
        let mut end = 0usize;
        for share in shares {
            let span = 1usize << share.num_vars;
            if share.stack_offset < end
                || share.stack_offset + span > len
                || share.lo_bits > share.num_vars
                || (share.hi_at + (span >> share.lo_bits)) * 3 > eq.len()
                || (share.lo_at + (1usize << share.lo_bits)) * 3 > eq.len()
            {
                return Ok(None);
            }
            end = share.stack_offset + span;
        }

        let be = backend()?;
        let stream = be.next_stream();
        // Each share's data offset: where its column starts in `data`.
        let (data, staged, data_offsets): (Arc<CudaSlice<u64>>, bool, Vec<usize>) = match message {
            LeanMessage::Resident { store, parts } => {
                let mut offsets = Vec::with_capacity(shares.len());
                for share in shares {
                    let Some(&(column, _)) = parts
                        .iter()
                        .find(|(_, offset)| *offset == share.stack_offset)
                    else {
                        return Ok(None);
                    };
                    let (at, rows) = store.span(column);
                    if rows != 1usize << share.num_vars {
                        return Ok(None);
                    }
                    offsets.push(at);
                }
                (store.buffer(), false, offsets)
            }
            LeanMessage::Parts(parts) => {
                // Zeroed: what the parts do not cover is the stacking's padding.
                let mut base = crate::device::alloc_zeros_or_trim::<u64>(&stream, len)?;
                for (column, offset) in parts {
                    if offset + column.len() > len {
                        return Ok(None);
                    }
                    let mut at = base.slice_mut(*offset..*offset + column.len());
                    stream.memcpy_htod(*column, &mut at)?;
                }
                let offsets = shares.iter().map(|s| s.stack_offset).collect();
                (Arc::new(base), true, offsets)
            }
        };

        let mut rows = Vec::with_capacity(shares.len() * 9);
        let mut starts = Vec::with_capacity(shares.len());
        let mut ends = Vec::with_capacity(shares.len());
        for (share, data_at) in shares.iter().zip(&data_offsets) {
            rows.extend_from_slice(&[
                share.stack_offset as u64,
                *data_at as u64,
                share.lo_bits as u64,
                share.hi_at as u64,
                share.lo_at as u64,
                share.scale[0],
                share.scale[1],
                share.scale[2],
                0,
            ]);
            starts.push(share.stack_offset as u64);
            ends.push((share.stack_offset + (1usize << share.num_vars)) as u64);
        }
        let shares_dev = crate::device::htod_or_trim(&stream, &rows)?;
        let eq_dev = crate::device::htod_or_trim(&stream, eq)?;
        // The share form's map, a `u16` a position: what its rounds and its
        // write read, and a checking tile form's. The tile form reads a `u16`
        // a tile instead.
        let colmap = if tiles.is_none_or(|spec| spec.check) {
            let starts_dev = crate::device::htod_or_trim(&stream, &starts)?;
            let ends_dev = crate::device::htod_or_trim(&stream, &ends)?;
            // SAFETY: the kernel writes every position.
            let mut colmap = unsafe { crate::device::alloc_or_trim::<u16>(&stream, len) }?;
            let count = shares.len() as u32;
            let len_arg = len as u64;
            let grid = (len as u64)
                .div_ceil(LEAN_BLOCK as u64)
                .min(LEAN_MAX_GRID as u64) as u32;
            unsafe {
                stream
                    .launch_builder(&be.whir_lean_colmap)
                    .arg(&mut colmap)
                    .arg(&len_arg)
                    .arg(&starts_dev)
                    .arg(&ends_dev)
                    .arg(&count)
                    .launch(LaunchConfig {
                        grid_dim: (grid.max(1), 1, 1),
                        block_dim: (LEAN_BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
            }
            Some(colmap)
        } else {
            None
        };
        let t_dev = crate::device::alloc_zeros_or_trim::<u64>(&stream, LEAN_MAX_T * 3)?;
        let partials = crate::device::alloc_zeros_or_trim::<u64>(
            &stream,
            LEAN_MAX_T * LEAN_MAX_GRID as usize * 3,
        )?;
        let sums = crate::device::alloc_zeros_or_trim::<u64>(&stream, LEAN_MAX_T * 3)?;
        let tiles = match tiles {
            Some(spec) => Some(Tiles::build(
                &stream,
                spec,
                num_vars,
                &shares_dev,
                &eq_dev,
                data.as_ref(),
                &starts,
                &ends,
            )?),
            None => None,
        };
        let lean = Self {
            stream,
            num_vars,
            colmap,
            shares: shares_dev,
            eq: eq_dev,
            data,
            staged,
            bound: Vec::new(),
            t_dev,
            t_host: Vec::new(),
            partials,
            sums,
            tiles,
        };
        lean.note_peak(lean.device_bytes());
        Ok(Some(lean))
    }

    /// Adds what this opening holds right now to [`lean_peak_bytes`], under
    /// its form.
    fn note_peak(&self, bytes: u64) {
        let form = match &self.tiles {
            None => 0,
            Some(tiles) if !tiles.spec.check => 1,
            Some(_) => 2,
        };
        LEAN_PEAKS[form].fetch_max(bytes, std::sync::atomic::Ordering::Relaxed);
    }

    /// Whether the rounds run in tile form.
    pub fn is_tiled(&self) -> bool {
        self.tiles.is_some()
    }

    /// Waits for everything queued on this opening's stream — for a test or a
    /// measurement that times its steps one by one.
    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize()
    }

    /// Variables still free.
    pub fn num_vars(&self) -> usize {
        self.num_vars - self.bound.len() / 3
    }

    /// Variables bound so far.
    pub fn bound(&self) -> usize {
        self.bound.len() / 3
    }

    /// What a lean opening of `2^num_vars` holds with `shares` columns and
    /// `eq_values` extension values of `eq` halves: a `u16` per position, nine
    /// u64 per share, the halves, the round scratch — and the message staged
    /// in the base field when it is not resident.
    pub const fn planned_bytes(
        num_vars: usize,
        shares: usize,
        eq_values: usize,
        staged: bool,
    ) -> u64 {
        let staging = if staged { 8u64 << num_vars } else { 0 };
        (2u64 << num_vars)
            + 72 * shares as u64
            + 24 * eq_values as u64
            + LEAN_SCRATCH_BYTES
            + staging
    }

    /// Device bytes this opening holds: the share map when it has one, the
    /// shares, the `eq` halves, the round scratch, the staged message when it
    /// staged one, and the tile form's tables.
    pub fn device_bytes(&self) -> u64 {
        let staged = if self.staged { self.data.len() * 8 } else { 0 };
        (self.colmap.as_ref().map_or(0, CudaSlice::len) * 2
            + (self.shares.len() + self.eq.len() + self.partials.len() + self.sums.len()) * 8
            + self.t_dev.len() * 8
            + staged) as u64
            + self.tiles.as_ref().map_or(0, Tiles::device_bytes)
    }

    /// `eq(bound challenges, ·)` over the `2^bound` prefixes, on the card —
    /// `[1]` before any is bound.
    fn prefix_weights(&self) -> Result<CudaSlice<u64>> {
        if self.bound.is_empty() {
            return crate::device::htod_or_trim(&self.stream, &[1u64, 0, 0]);
        }
        crate::sumcheck::eq_table_ext3(&self.stream, &self.bound, 1usize << self.bound())
    }

    /// The next round's evaluations at the nodes `t` (ext3, three u64 each),
    /// as a [`SumcheckSession::round`] over the materialised factors would
    /// return them.
    pub fn round(&mut self, t: &[u64]) -> Result<Vec<u64>> {
        assert!(t.len().is_multiple_of(3), "three u64 per ext3 node");
        let num_t = t.len() / 3;
        assert!(num_t > 0 && num_t <= LEAN_MAX_T, "nodes per round");
        assert!(self.num_vars() >= 1, "a round needs a variable to bind");
        match self.tiles.as_ref().map(|tiles| tiles.spec.check) {
            None => self.share_round(t),
            Some(false) => self.tile_round(t),
            Some(true) => {
                use std::sync::atomic::Ordering::Relaxed;
                let mut tiled = self.tile_round(t)?;
                let shared = self.share_round(t)?;
                TILE_CHECKED.fetch_add(1, Relaxed);
                if TILE_CHECK_FAULT.load(Relaxed) {
                    tiled[0] = tiled[0].wrapping_add(1) % 0xFFFF_FFFF_0000_0001;
                }
                let canonical = |limbs: &[u64]| -> Vec<u64> {
                    const P: u64 = 0xFFFF_FFFF_0000_0001;
                    limbs
                        .iter()
                        .map(|&v| if v >= P { v - P } else { v })
                        .collect()
                };
                if canonical(&tiled) != canonical(&shared) {
                    // Every one counted, the first few printed.
                    if TILE_MISMATCHED.fetch_add(1, Relaxed) < 4 {
                        eprintln!(
                            "[gpu] WHIR lean tiles: round {} of a 2^{} stack differs from the \
                             share form: {tiled:?} against {shared:?}",
                            self.bound() + 1,
                            self.num_vars
                        );
                    }
                }
                Ok(shared)
            }
        }
    }

    /// The nodes on the card, sent when they change: they are the same every
    /// round of an opening.
    fn send_nodes(&mut self, t: &[u64]) -> Result<()> {
        if self.t_host != t {
            let mut head = self.t_dev.slice_mut(0..t.len());
            self.stream.memcpy_htod(t, &mut head)?;
            self.t_host.clear();
            self.t_host.extend_from_slice(t);
        }
        Ok(())
    }

    /// Binds the slot tables at the challenges they have not been bound at
    /// yet, each one fold over what the one before left.
    fn fold_tiles(&mut self) -> Result<()> {
        let Some(tiles) = self.tiles.as_mut() else {
            return Ok(());
        };
        let be = backend()?;
        let stride = 1u64 << (self.num_vars - tiles.spec.tile_bits);
        let slots = tiles.spec.slots as u32;
        let Tiles {
            omega,
            phi,
            r_dev,
            folded,
            ..
        } = tiles;
        let phi = phi
            .as_mut()
            .expect("Φ is held until the tables are written");
        while *folded < self.bound.len() / 3 {
            let at = *folded * 3;
            self.stream.memcpy_htod(&self.bound[at..at + 3], r_dev)?;
            let half = (stride >> *folded) / 2;
            let tasks = 2 * slots as u64 * half;
            let grid = tasks.div_ceil(LEAN_BLOCK as u64).min(LEAN_MAX_GRID as u64) as u32;
            unsafe {
                self.stream
                    .launch_builder(&be.whir_tile_fold)
                    .arg(&mut *omega)
                    .arg(&mut *phi)
                    .arg(&stride)
                    .arg(&slots)
                    .arg(&half)
                    .arg(&*r_dev)
                    .launch(LaunchConfig {
                        grid_dim: (grid.max(1), 1, 1),
                        block_dim: (LEAN_BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
            }
            *folded += 1;
        }
        Ok(())
    }

    /// [`round`](Self::round) over the slot tables: no stack position is read.
    fn tile_round(&mut self, t: &[u64]) -> Result<Vec<u64>> {
        let num_t = t.len() / 3;
        let be = backend()?;
        self.fold_tiles()?;
        self.send_nodes(t)?;
        let tiles = self.tiles.as_ref().expect("a tile round runs in tile form");
        // Past its lean rounds a pair would join two columns of tiles, whose
        // slots name different classes.
        assert!(
            tiles.folded < tiles.spec.lean_rounds,
            "a tile round binds one of the lean rounds' variables"
        );
        let stride = 1u64 << (self.num_vars - tiles.spec.tile_bits);
        let half = (stride >> tiles.folded) / 2;
        let grid = half.div_ceil(LEAN_BLOCK as u64).min(LEAN_MAX_GRID as u64) as u32;
        let slots = tiles.spec.slots as u32;
        let num_t_arg = num_t as u32;
        unsafe {
            self.stream
                .launch_builder(&be.whir_tile_round)
                .arg(&tiles.omega)
                .arg(
                    tiles
                        .phi
                        .as_ref()
                        .expect("Φ is held until the tables are written"),
                )
                .arg(&stride)
                .arg(&slots)
                .arg(&half)
                .arg(&self.t_dev)
                .arg(&num_t_arg)
                .arg(&mut self.partials)
                .launch(LaunchConfig {
                    grid_dim: (grid.max(1), 1, 1),
                    block_dim: (LEAN_BLOCK, 1, 1),
                    shared_mem_bytes: LEAN_BLOCK * 3 * 8,
                })?;
            self.stream
                .launch_builder(&be.sum_partials_ext3)
                .arg(&self.partials)
                .arg(&(grid.max(1) as u64))
                .arg(&mut self.sums)
                .launch(LaunchConfig {
                    grid_dim: (num_t as u32, 1, 1),
                    block_dim: (LEAN_BLOCK, 1, 1),
                    shared_mem_bytes: LEAN_BLOCK * 3 * 8,
                })?;
        }
        let sums = self.stream.clone_dtoh(&self.sums.slice(0..num_t * 3))?;
        self.stream.synchronize()?;
        Ok(sums)
    }

    /// [`round`](Self::round) in share form: every stack position, read where
    /// the columns lie.
    fn share_round(&mut self, t: &[u64]) -> Result<Vec<u64>> {
        let num_t = t.len() / 3;
        let be = backend()?;
        let round = self.bound() + 1;
        let eqc = self.prefix_weights()?;
        self.send_nodes(t)?;
        let rests = 1u64 << (self.num_vars - round);
        let grid = rests.div_ceil(LEAN_BLOCK as u64).min(LEAN_MAX_GRID as u64) as u32;
        let n_arg = self.num_vars as u32;
        let s_arg = round as u32;
        let num_t_arg = num_t as u32;
        let colmap = self
            .colmap
            .as_ref()
            .expect("the share form runs where its map is built");
        unsafe {
            self.stream
                .launch_builder(&be.whir_lean_round)
                .arg(colmap)
                .arg(&self.shares)
                .arg(&self.eq)
                .arg(self.data.as_ref())
                .arg(&n_arg)
                .arg(&s_arg)
                .arg(&eqc)
                .arg(&self.t_dev)
                .arg(&num_t_arg)
                .arg(&mut self.partials)
                .launch(LaunchConfig {
                    grid_dim: (grid.max(1), 1, 1),
                    block_dim: (LEAN_BLOCK, 1, 1),
                    shared_mem_bytes: LEAN_BLOCK * 3 * 8,
                })?;
            self.stream
                .launch_builder(&be.sum_partials_ext3)
                .arg(&self.partials)
                .arg(&(grid.max(1) as u64))
                .arg(&mut self.sums)
                .launch(LaunchConfig {
                    grid_dim: (num_t as u32, 1, 1),
                    block_dim: (LEAN_BLOCK, 1, 1),
                    shared_mem_bytes: LEAN_BLOCK * 3 * 8,
                })?;
        }
        let sums = self.stream.clone_dtoh(&self.sums.slice(0..num_t * 3))?;
        self.stream.synchronize()?;
        Ok(sums)
    }

    /// Binds the round's variable to `r` (ext3, three u64). Nothing moves on
    /// the card: the next round reads the shares with one more prefix bound.
    pub fn bind(&mut self, r: &[u64]) {
        assert_eq!(r.len(), 3, "an ext3 challenge");
        assert!(self.num_vars() >= 1, "a bind needs a variable");
        self.bound.extend_from_slice(r);
    }

    /// The factors at the challenges bound so far, written out at
    /// `2^(n − bound)`: the session the rest of the chain runs on.
    ///
    /// In tile form they are written from the slot tables
    /// (`whir_tile_materialize`): `Ω` bound at every challenge so far, and
    /// `Φ` freed before the tables are allocated ([`TileSpec::fits`]). A
    /// checking opening writes the share form's tables, keeps them, and
    /// compares the tile form's write with them
    /// ([`tile_table_check_counts`]).
    pub fn materialize(mut self) -> Result<OpeningSession> {
        let be = backend()?;
        self.fold_tiles()?;
        if let Some(tiles) = self.tiles.as_mut() {
            tiles.phi = None;
        }
        let bound = self.bound();
        let len = 1usize << (self.num_vars - bound);
        let eqfull = self.prefix_weights()?;
        // SAFETY: the kernel writes every element of both.
        let mut weight = unsafe { crate::device::alloc_or_trim::<u64>(&self.stream, len * 3) }?;
        let mut message = unsafe { crate::device::alloc_or_trim::<u64>(&self.stream, len * 3) }?;
        self.note_peak(
            self.device_bytes() + ((weight.len() + message.len() + eqfull.len()) * 8) as u64,
        );
        let n_arg = self.num_vars as u32;
        let bound_arg = bound as u32;
        let grid = (len as u64)
            .div_ceil(LEAN_BLOCK as u64)
            .min(LEAN_MAX_GRID as u64) as u32;
        let config = LaunchConfig {
            grid_dim: (grid.max(1), 1, 1),
            block_dim: (LEAN_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        match &self.tiles {
            Some(tiles) if !tiles.spec.check => {
                let (lean_arg, u_arg, slots_arg) = (
                    tiles.spec.lean_rounds as u32,
                    tiles.spec.tile_bits as u32,
                    tiles.spec.slots as u32,
                );
                unsafe {
                    self.stream
                        .launch_builder(&be.whir_tile_materialize)
                        .arg(&tiles.tilemap)
                        .arg(&self.shares)
                        .arg(&self.eq)
                        .arg(self.data.as_ref())
                        .arg(&tiles.slot_lo)
                        .arg(&tiles.omega)
                        .arg(&n_arg)
                        .arg(&lean_arg)
                        .arg(&u_arg)
                        .arg(&slots_arg)
                        .arg(&bound_arg)
                        .arg(&eqfull)
                        .arg(&mut weight)
                        .arg(&mut message)
                        .launch(config)?;
                }
            }
            tiles => {
                let colmap = self
                    .colmap
                    .as_ref()
                    .expect("the share form writes where its map is built");
                unsafe {
                    self.stream
                        .launch_builder(&be.whir_lean_materialize)
                        .arg(colmap)
                        .arg(&self.shares)
                        .arg(&self.eq)
                        .arg(self.data.as_ref())
                        .arg(&n_arg)
                        .arg(&bound_arg)
                        .arg(&eqfull)
                        .arg(&mut weight)
                        .arg(&mut message)
                        .launch(config)?;
                }
                if let Some(tiles) = tiles {
                    self.check_write(tiles, &eqfull, &mut weight, &message, config)?;
                }
            }
        }
        Ok(OpeningSession::from_tables(
            self.stream.clone(),
            weight,
            message,
            len,
        ))
    }

    /// A checking opening's comparison of the tile form's write with the
    /// share form's tables just written: the positions where either factor
    /// differs, counted on the card and added to
    /// [`tile_table_check_counts`]. The tables are left as written.
    fn check_write(
        &self,
        tiles: &Tiles,
        eqfull: &CudaSlice<u64>,
        weight: &mut CudaSlice<u64>,
        message: &CudaSlice<u64>,
        config: LaunchConfig,
    ) -> Result<()> {
        use std::sync::atomic::Ordering::Relaxed;
        let be = backend()?;
        // The fault moves the weight's first entry for the comparison only.
        let kept = if TILE_CHECK_FAULT.load(Relaxed) {
            let kept = self.stream.clone_dtoh(&weight.slice(0..3))?;
            self.stream.synchronize()?;
            let mut moved = kept.clone();
            moved[0] = moved[0].wrapping_add(1) % 0xFFFF_FFFF_0000_0001;
            self.stream
                .memcpy_htod(&moved, &mut weight.slice_mut(0..3))?;
            self.stream.synchronize()?;
            Some(kept)
        } else {
            None
        };
        let mut differing = crate::device::alloc_zeros_or_trim::<u64>(&self.stream, 1)?;
        let (n_arg, lean_arg, u_arg, slots_arg, bound_arg) = (
            self.num_vars as u32,
            tiles.spec.lean_rounds as u32,
            tiles.spec.tile_bits as u32,
            tiles.spec.slots as u32,
            self.bound() as u32,
        );
        unsafe {
            self.stream
                .launch_builder(&be.whir_tile_materialize_check)
                .arg(&tiles.tilemap)
                .arg(&self.shares)
                .arg(&self.eq)
                .arg(self.data.as_ref())
                .arg(&tiles.slot_lo)
                .arg(&tiles.omega)
                .arg(&n_arg)
                .arg(&lean_arg)
                .arg(&u_arg)
                .arg(&slots_arg)
                .arg(&bound_arg)
                .arg(eqfull)
                .arg(&*weight)
                .arg(message)
                .arg(&mut differing)
                .launch(config)?;
        }
        let count = self.stream.clone_dtoh(&differing)?;
        self.stream.synchronize()?;
        if let Some(kept) = kept {
            self.stream
                .memcpy_htod(&kept, &mut weight.slice_mut(0..3))?;
            self.stream.synchronize()?;
        }
        TILE_TABLES_CHECKED.fetch_add(1, Relaxed);
        if count[0] > 0 && TILE_TABLES_DIFFERING.fetch_add(count[0], Relaxed) == 0 {
            eprintln!(
                "[gpu] WHIR lean tiles: the tables written at {} bound of a 2^{} stack differ from \
                 the share form's at {} positions",
                self.bound(),
                self.num_vars,
                count[0]
            );
        }
        Ok(())
    }
}

/// What one chain of a stacked opening allocates on the card depends on: the
/// stack, the chain's first two folds, which kernels run it, and what the lean
/// rounds read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpeningShape {
    /// The stacked polynomial's variables.
    pub num_vars: usize,
    pub log_blowup: usize,
    /// The chain's first and second fold widths; `next_fold` is 0 for a chain
    /// of one round.
    pub first_fold: usize,
    pub next_fold: usize,
    /// Rounds run over the shares before the factors are written; 0 writes
    /// them at full width up front.
    pub lean_rounds: usize,
    /// Whether a fold runs every level in one launch.
    pub fused: bool,
    /// Whether the first successor may be encoded from the bound message
    /// instead of folded ([`crate::whir::encode_successor`]).
    pub encoded: bool,
    /// Whether the message is sent rather than read where the columns lie.
    pub staged: bool,
    /// Columns in the stacked polynomial, and the extension values of their
    /// points' `eq` halves.
    pub shares: usize,
    pub eq_values: usize,
    /// Slots the opening's program lowers to.
    pub slots: usize,
}

/// ★ The most one chain of a stacked opening holds on the card beside the
/// codeword it opens — what a group's openings take their turn for, one chain
/// at a time.
///
/// Three phases, each of which frees what it no longer needs before the next
/// allocates, so the turn is the largest of them rather than their sum:
///
/// - **Share form** (the lean rounds): the share map, the shares, the `eq`
///   halves, the round scratch and the staged message when there is one —
///   [`LeanRound0::planned_bytes`] — until the factors are written at the
///   width the rounds leave, beside it. Without the lean rounds the factors
///   are written at full width from a base copy of the message, whether that
///   copy is staged or gathered from the resident columns.
/// - **Rounds** over the written factors: the factors and a session's slot
///   file and partials ([`SumcheckSession::scratch_bytes`]).
/// - **The first fold and what follows it**: the factors, then the fold's
///   transient ([`fold_transient_bytes`]) — or, when the successor may be
///   encoded from the bound message instead, the larger of that and the
///   encode's ([`crate::whir::encode_successor_transient_bytes`]), since a
///   declined encode folds; after it, beside the fold's output,
///   one at a time: the successor's tree, the tree over the committed codeword
///   its queries' paths are read from, and the out-of-domain answer's copy of
///   the message with the next weight's `eq` table.
///
/// Every later round works on a codeword at most `2^first_fold` times
/// smaller, so it is inside these. A leaf layer kept for retention is promised
/// by the codeword it belongs to, not by the turn.
pub fn opening_transient_bytes(shape: &OpeningShape) -> u64 {
    use crate::whir::{
        FUSED_MAX_FOLD, encode_successor_transient_bytes, fold_transient_bytes, tree_bytes,
    };

    let n = shape.num_vars;
    let lean = shape.lean_rounds > 0 && shape.shares > 0 && shape.shares < LEAN_MAX_SHARES;
    let bound = if lean {
        shape.lean_rounds.min(shape.first_fold).min(n)
    } else {
        0
    };
    let factors = 48u64 << (n - bound);
    let written = if lean {
        LeanRound0::planned_bytes(n, shape.shares, shape.eq_values, shape.staged) + factors
    } else {
        factors + (8u64 << n)
    };
    let rounds = factors + SumcheckSession::scratch_bytes(1usize << (n - bound), shape.slots);
    let log_codeword = n + shape.log_blowup;
    let folded = log_codeword - shape.first_fold;
    let transient = fold_transient_bytes(
        log_codeword,
        shape.first_fold,
        shape.fused && shape.first_fold <= FUSED_MAX_FOLD,
    );
    let transient = if shape.encoded {
        transient.max(encode_successor_transient_bytes(
            folded,
            n - shape.first_fold.min(n),
        ))
    } else {
        transient
    };
    let successor = if shape.next_fold > 0 {
        tree_bytes(folded - shape.next_fold)
    } else {
        0
    };
    let out_of_domain = 2 * (24u64 << (n - shape.first_fold.min(n)));
    let beside = tree_bytes(folded).max(successor).max(out_of_domain);
    let fold = factors + transient.max((24u64 << folded) + beside);
    written.max(rounds).max(fold)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ★ The working set the lean opening exists to shrink, in numbers, at the
    /// production stack (25) and at two fatter ones (27, 28). The materialised
    /// session is `1.5 C` held plus `0.25 C` staging (`C` the committed base
    /// codeword, `2^(n+2)·8` bytes); the lean one over resident columns is
    /// `C/16` for the share map plus kilobytes — 400 columns and 30 points'
    /// `eq` halves at `2^11` values each is generous.
    #[test]
    fn the_lean_opening_holds_a_sixteenth_of_a_codeword() {
        for n in [25usize, 27, 28] {
            let codeword = 8u64 << (n + 2);
            let (held, staging) = OpeningSession::planned_bytes(n);
            let lean = LeanRound0::planned_bytes(n, 400, 30 * 2 * 2048, false);
            let lean_staged = LeanRound0::planned_bytes(n, 400, 30 * 2 * 2048, true);
            println!(
                "2^{n}: materialised {:.2} C + {:.2} C staging; lean {:.4} C resident, {:.4} C staged",
                held as f64 / codeword as f64,
                staging as f64 / codeword as f64,
                lean as f64 / codeword as f64,
                lean_staged as f64 / codeword as f64,
            );
            assert_eq!(held * 2, codeword * 3);
            assert!(lean * 15 < codeword, "2^{n}: {lean} B");
            assert!(
                lean * 20 < held + staging,
                "2^{n}: {lean} B against {} B",
                held + staging
            );
        }
    }

    /// Three lean rounds unless the knob says otherwise; 7 or more is the
    /// widest fold; unreadable is the default.
    #[test]
    fn three_lean_rounds_by_default_and_the_knob_clamps_to_the_widest_fold() {
        use crate::whir::FUSED_MAX_FOLD;
        assert_eq!(LEAN_ROUNDS_DEFAULT, 3);
        assert_eq!(lean_rounds_setting(None), 3);
        assert_eq!(lean_rounds_setting(Some("6")), 6, "the way back");
        assert_eq!(lean_rounds_setting(Some(" 2 ")), 2);
        assert_eq!(lean_rounds_setting(Some("0")), 0, "materialise up front");
        assert_eq!(FUSED_MAX_FOLD, 6);
        assert_eq!(lean_rounds_setting(Some("7")), FUSED_MAX_FOLD);
        assert_eq!(lean_rounds_setting(Some("99")), FUSED_MAX_FOLD);
        assert_eq!(lean_rounds_setting(Some("three")), LEAN_ROUNDS_DEFAULT);
        assert_eq!(lean_rounds_setting(Some("-1")), LEAN_ROUNDS_DEFAULT);
    }

    /// A stack of `LEAN_MAX_SHARES` shares or more cannot be named by the
    /// card's `u16` share ids: the lean rounds decline before the device is
    /// touched, and the opening materialises at full width as before.
    #[test]
    fn a_stack_of_too_many_shares_declines_the_lean_rounds() {
        let shares: Vec<LeanShare> = (0..LEAN_MAX_SHARES)
            .map(|i| LeanShare {
                stack_offset: i,
                num_vars: 0,
                lo_bits: 0,
                hi_at: 0,
                lo_at: 0,
                scale: [1, 0, 0],
            })
            .collect();
        let eq = vec![1u64, 0, 0];
        let lean = LeanRound0::new(&shares, &eq, 16, LeanMessage::Parts(&[]))
            .expect("declining touches no device");
        assert!(lean.is_none(), "{} shares must materialise", shares.len());
    }

    /// `LAMBDA_VM_WHIR_LEAN_TILES`: unset, the tile form a block a tile; the
    /// share form at `0` or any word but `1`, `wide` and `check`.
    #[test]
    fn the_lean_rounds_run_in_tile_form_unless_the_knob_says_otherwise() {
        assert_eq!(LEAN_TILES_DEFAULT, LeanTiles::Wide);
        assert_eq!(lean_tiles_setting(None), LEAN_TILES_DEFAULT);
        assert_eq!(lean_tiles_setting(Some("0")), LeanTiles::Off);
        assert_eq!(lean_tiles_setting(Some("")), LeanTiles::Off);
        assert_eq!(lean_tiles_setting(Some("yes")), LeanTiles::Off);
        assert_eq!(lean_tiles_setting(Some("1")), LeanTiles::On);
        assert_eq!(lean_tiles_setting(Some(" 1 ")), LeanTiles::On);
        assert_eq!(lean_tiles_setting(Some("wide")), LeanTiles::Wide);
        assert_eq!(lean_tiles_setting(Some("check")), LeanTiles::Check);
    }

    fn tile_share(stack_offset: usize, num_vars: usize, lo_bits: usize, lo_at: usize) -> LeanShare {
        LeanShare {
            stack_offset,
            num_vars,
            lo_bits,
            hi_at: 0,
            lo_at,
            scale: [1, 0, 0],
        }
    }

    fn spec(tile_bits: usize, lean_rounds: usize, slots: usize) -> TileSpec {
        TileSpec {
            tile_bits,
            lean_rounds,
            slots,
            wide: false,
            check: false,
        }
    }

    /// ★ The tile form adds nothing to an opening's working set. It never
    /// allocates the share form's position map (outside `check`); wherever it
    /// fits, ALL it holds through the rounds is no larger than the tables
    /// written once they are spent — `materialize` frees `Φ` before it
    /// allocates them — and what it keeps beside those tables while it writes
    /// them is no larger than the map. So at no step does an opening hold
    /// more than the share form at its write, and the turn a group's openings
    /// take ([`opening_transient_bytes`]) is what it was. At the production
    /// stack and three rounds a slot is 6.4 MiB against the tables' 768 MiB,
    /// and 3.3 MiB against the map's 256 MiB.
    #[test]
    fn the_tile_form_is_no_larger_than_the_tables_written_after_it() {
        for n in [16usize, 22, 25, 27, 28] {
            for lean_rounds in 1..=6usize {
                for tile_bits in 0..=10usize {
                    for slots in 1..=TILE_MAX_SLOTS {
                        let form = spec(tile_bits, lean_rounds, slots);
                        if form.fits(n) {
                            let tables = 48u64 << (n - lean_rounds);
                            let map = 2u64 << n;
                            assert!(form.device_bytes(n) <= tables);
                            assert!(form.write_bytes(n) <= map);
                            assert_eq!(
                                form.device_bytes(n) - form.write_bytes(n),
                                (24 * slots as u64) << (n - tile_bits),
                                "the write frees Φ alone"
                            );
                        }
                    }
                }
            }
        }
        assert_eq!(spec(10, 3, 1).write_bytes(27), (3 << 20) + (3 << 17) + 24);
        assert!(
            spec(3, 1, 1).device_bytes(27) <= 48 << 26 && !spec(3, 1, 1).fits(27),
            "at one round, tiles of eight positions: the write would keep 480 MiB beside the \
             tables, more than the 256 MiB map"
        );
        assert!(spec(4, 1, 1).fits(27));
        assert_eq!(spec(10, 3, 1).device_bytes(27), (6 << 20) + (3 << 17) + 24);
        assert_eq!(spec(10, 3, 8).device_bytes(27), (49 << 20) + (1 << 18) + 24);
        assert!(spec(10, 3, 8).fits(27) && spec(8, 3, 8).fits(27) && spec(7, 3, 8).fits(27));
        assert!(
            !spec(6, 3, 8).fits(27),
            "eight slots of 2^21 tiles are 788 MiB"
        );
        assert!(spec(4, 3, 1).fits(27) && !spec(3, 3, 1).fits(27));
        // Not a form at all: no round, a round past the tiles, no slot, too many.
        assert!(!spec(10, 0, 1).fits(27) && !spec(10, 18, 1).fits(27));
        assert!(!spec(10, 3, 0).fits(27) && !spec(10, 3, TILE_MAX_SLOTS + 1).fits(27));
        assert!(!spec(28, 1, 1).fits(27));
    }

    /// What the tile form reads: every share split at the tile, at least a
    /// tile long, at a multiple of its own size, and no more classes in a
    /// column of tiles than slots.
    #[test]
    fn the_tile_form_reads_aligned_shares_split_at_the_tile() {
        let table = [
            tile_share(0, 12, 6, 0),
            tile_share(1 << 12, 12, 6, 0),
            tile_share(1 << 13, 8, 6, 64),
        ];
        assert!(tiles_read(&table, 16, &spec(6, 3, 2)));
        assert!(
            !tiles_read(&table, 16, &spec(6, 3, 1)),
            "two classes, one slot"
        );
        // Two rounds bind four tiles: a column of tiles meets two classes at most.
        let many: Vec<LeanShare> = (0..5).map(|c| tile_share(c << 12, 12, 6, c * 64)).collect();
        assert!(!tiles_read(&many, 16, &spec(6, 3, 4)));
        assert!(tiles_read(&many, 16, &spec(6, 2, 4)));
        assert!(
            !tiles_read(&[tile_share(0, 12, 5, 0)], 16, &spec(6, 3, 1)),
            "split elsewhere"
        );
        assert!(
            !tiles_read(&[tile_share(0, 5, 6, 0)], 16, &spec(6, 3, 1)),
            "shorter than a tile"
        );
        assert!(
            !tiles_read(&[tile_share(1 << 11, 12, 6, 0)], 16, &spec(6, 3, 1)),
            "off its alignment"
        );
        assert!(
            !tiles_read(&table, 16, &spec(6, 11, 2)),
            "the spec does not fit"
        );
    }

    /// A stack the tile form does not read declines before the device is
    /// touched, and the caller opens in share form.
    #[test]
    fn a_stack_the_tile_form_cannot_read_declines_before_the_device() {
        let shares = [tile_share(0, 12, 5, 0)];
        let eq = vec![1u64, 0, 0];
        let lean = LeanRound0::with_tiles(
            &shares,
            &eq,
            16,
            LeanMessage::Parts(&[]),
            Some(spec(6, 3, 1)),
        )
        .expect("declining touches no device");
        assert!(lean.is_none());
    }

    /// The production chain's shape at `2^n`: first6 then four, 400 columns,
    /// the default lean rounds.
    fn production(n: usize) -> OpeningShape {
        OpeningShape {
            num_vars: n,
            log_blowup: 2,
            first_fold: 6,
            next_fold: 4,
            lean_rounds: LEAN_ROUNDS_DEFAULT,
            fused: true,
            encoded: false,
            staged: false,
            shares: 400,
            eq_values: 30 * 2 * 2048,
            slots: 2,
        }
    }

    /// ★ The turn a group's openings take, in numbers.
    ///
    /// With both kernels the largest phase is the first fold's: the factors
    /// bound three variables down at the default (`3C/16`), the fold's output
    /// (`3C/64`) and the tree its queries' paths come from (`C/8`) — `23/64` of
    /// the committed base codeword `C`, over a third, at every stack. Six lean
    /// rounds (the way back) bind the factors to `3C/128` and the turn to about
    /// a fifth. Without the kernels it is the factors at full width (`1.5 C`)
    /// beside a level-by-level fold (`2.25 C`): three and three quarters,
    /// EXACTLY — the working set that ran nearly three codewords past the
    /// one-codeword room every opening had promised.
    #[test]
    fn an_opening_turn_is_a_third_of_a_codeword_with_the_kernels_a_fifth_at_six() {
        for n in [16usize, 22, 25, 27, 28] {
            let codeword = 8u64 << (n + 2);
            let lean = opening_transient_bytes(&production(n));
            let six = opening_transient_bytes(&OpeningShape {
                lean_rounds: 6,
                ..production(n)
            });
            let today = opening_transient_bytes(&OpeningShape {
                lean_rounds: 0,
                fused: false,
                ..production(n)
            });
            let staged = opening_transient_bytes(&OpeningShape {
                staged: true,
                ..production(n)
            });
            println!(
                "2^{n}: opening turn {:.4} C resident, {:.4} C staged, {:.4} C at six rounds; \
                 {:.4} C without the kernels",
                lean as f64 / codeword as f64,
                staged as f64 / codeword as f64,
                six as f64 / codeword as f64,
                today as f64 / codeword as f64,
            );
            if n >= 22 {
                assert!(
                    lean * 3 > codeword && lean * 64 <= codeword * 23,
                    "2^{n}: {lean} B is not 23/64 of {codeword} B"
                );
                assert!(
                    six * 5 < codeword && six * 6 > codeword,
                    "2^{n}: {six} B at six rounds is not about a fifth of {codeword} B"
                );
            }
            assert_eq!(today * 4, codeword * 15, "2^{n}: {today} B");
            assert!(staged >= lean && staged - lean <= 8u64 << n);
        }
        // Each kernel alone: the lean rounds without the fused fold still meet
        // the level-by-level transient, and the fused fold without the lean
        // rounds still meets the factors at full width.
        let n = 25;
        let codeword = 8u64 << (n + 2);
        let lean_only = opening_transient_bytes(&OpeningShape {
            fused: false,
            ..production(n)
        });
        let fused_only = opening_transient_bytes(&OpeningShape {
            lean_rounds: 0,
            ..production(n)
        });
        assert!(lean_only * 4 > codeword * 9, "{lean_only} B");
        assert!(fused_only * 2 > codeword * 3, "{fused_only} B");
    }

    /// The shares and the stack decide which path the turn is sized for: past
    /// what a `u16` can name, the lean opening declines and the turn is the
    /// materialised one's, as the opening itself will be.
    #[test]
    fn a_stack_the_lean_rounds_cannot_read_is_sized_as_materialised() {
        let n = 25;
        let too_many = opening_transient_bytes(&OpeningShape {
            shares: LEAN_MAX_SHARES,
            ..production(n)
        });
        let materialised = opening_transient_bytes(&OpeningShape {
            lean_rounds: 0,
            shares: LEAN_MAX_SHARES,
            ..production(n)
        });
        assert_eq!(too_many, materialised);
        assert!(too_many > opening_transient_bytes(&production(n)));
    }

    /// A successor encoded from the bound message holds more beside the
    /// factors than the fused fold — the interleaved successor, one limb's
    /// plane and its encoding (`17/8 · 2^n` bytes at a first fold of six)
    /// against the output alone (`3/2 · 2^n`) — but never more than the fold's
    /// output and the tree the first round's paths are read from
    /// (`32·2^f + 8·2^(n−k) ≤ 40·2^f` against `88·2^f` less a node, `f` the
    /// folded codeword's log size), so the turn is the one the fold takes, at
    /// every first fold.
    #[test]
    fn an_encoded_successor_fits_the_turn_the_fold_takes() {
        use crate::whir::{encode_successor_transient_bytes, fold_transient_bytes, tree_bytes};
        for n in [16usize, 22, 25, 27, 28] {
            let folded = n + 2 - 6;
            let encode = encode_successor_transient_bytes(folded, n - 6);
            assert_eq!(encode * 8, 17u64 << n);
            assert_eq!(fold_transient_bytes(n + 2, 6, true) * 2, 3u64 << n);
            for first_fold in 1..=6 {
                for lean_rounds in [0, 3, 6] {
                    let shape = OpeningShape {
                        first_fold,
                        lean_rounds,
                        ..production(n)
                    };
                    let folded = n + 2 - first_fold;
                    assert!(
                        encode_successor_transient_bytes(folded, n - first_fold)
                            < (24u64 << folded) + tree_bytes(folded)
                    );
                    assert_eq!(
                        opening_transient_bytes(&OpeningShape {
                            encoded: true,
                            ..shape
                        }),
                        opening_transient_bytes(&shape),
                        "2^{n}, first fold {first_fold}, {lean_rounds} lean rounds"
                    );
                }
            }
        }
    }
}
