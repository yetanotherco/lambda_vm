//! Device dispatch for the multilinear path.
//!
//! Every entry point here returns `None` when the device declines — a field
//! the kernels do not cover, a size below the launch threshold, a kill switch,
//! or any CUDA error — and the caller runs the host path. A dispatch that
//! succeeded is counted, so a bench can tell a GPU number from a CPU one
//! wearing its label.

use core::sync::atomic::{AtomicU64, Ordering};

/// Successful device commits of a stacked polynomial.
static COMMIT_CALLS: AtomicU64 = AtomicU64::new(0);
/// ★ Commits that asked the device and got nothing, and encoded on the host.
///
/// The counter H4's arm needed and did not have. A device commit that declines
/// is INVISIBLE in every other number here: `COMMIT_CALLS` simply does not
/// rise, and a count that is merely lower than expected says nothing when the
/// expected count is itself derived. It matters because falling back is not a
/// slower way to do the same thing — `from_codeword` then retains a host
/// codeword and a host node array for the rest of the proof, so a card that
/// fills near the end of an epoch turns into gigabytes of host memory and a
/// utilisation figure that looks like a scheduling problem.
static HOST_FALLBACKS: AtomicU64 = AtomicU64::new(0);
/// ★ Commits the device was ASKED for and answered with an error — the part of
/// [`host_fallbacks`] that is not a policy decline (the size threshold, the
/// kill switch): a refused reservation (`CUDA_ERROR_OUT_OF_MEMORY`), a failed
/// allocation, launch or copy. Each one is also logged with its error. The
/// launch that can never succeed is why this exists: a WHIR base commit at
/// stack 27 launched a tile past CUDA's grid limit, every such commit fell back
/// to the host, and the error — the one thing that named the cause — was
/// dropped (`thoughts/zf/gap/fix/STRUCT.md` §9.2).
static COMMIT_ERRORS: AtomicU64 = AtomicU64::new(0);
/// ★ Merkle trees the device was ASKED to build over a HOST codeword (an
/// extension-field fold of a chain whose codeword is on the host, through
/// [`commit_tree_ext3`]) and answered with an error. The host builds that tree
/// instead. Kept apart from [`COMMIT_ERRORS`], which stays the part of
/// [`host_fallbacks`] that failed: such a tree is not a commit fallback. Each
/// one is also logged with its error.
static TREE_COMMIT_ERRORS: AtomicU64 = AtomicU64::new(0);
/// Sumchecks whose rounds ran on device.
static SUMCHECK_CALLS: AtomicU64 = AtomicU64::new(0);
/// Rounds within them, so a declined tail shows up.
static SUMCHECK_ROUNDS: AtomicU64 = AtomicU64::new(0);
/// Multilinear evaluations bound on device.
static EVALUATE_CALLS: AtomicU64 = AtomicU64::new(0);
/// Columns the claim reduce evaluated on the HOST, one at a time: the tables
/// the batched device evaluation did not take, at a height where a single
/// column stays here too ([`evaluates_on_device`]). The count
/// `LAMBDA_VM_ARGUE_DEVICE_COLUMNS` exists to move: with it off (`=0`) every
/// table under 2^16 rows lands here, however wide.
static HOST_EVALUATE_CALLS: AtomicU64 = AtomicU64::new(0);
/// Columns whose device value `LAMBDA_VM_ARGUE_XCHECK` recomputed on the host
/// and found equal — so a gate run can show the check ran, not only that
/// nothing failed.
static ARGUE_XCHECKS: AtomicU64 = AtomicU64::new(0);
/// Challenge tables the card built for the argument
/// (`LAMBDA_VM_ARGUE_DEVICE_TABLES`): the zerocheck's `eq` weights and the claim
/// reduce's shift tables and batched columns, each of which the host would
/// otherwise have built and uploaded.
static ARGUE_TABLES_ON_CARD: AtomicU64 = AtomicU64::new(0);
/// Of those, the ones `LAMBDA_VM_ARGUE_XCHECK` compared with the host's.
static ARGUE_TABLE_XCHECKS: AtomicU64 = AtomicU64::new(0);
/// Device sumchecks whose end was read back in one gathered copy
/// (`LAMBDA_VM_ARGUE_LEAN_READS`).
static READS_GATHERED: AtomicU64 = AtomicU64::new(0);
/// Factors read back one synchronous copy at a time — what the gathered read
/// replaces.
static READS_PER_FACTOR: AtomicU64 = AtomicU64::new(0);
/// Device GKR layers whose host tail ran lean (`LAMBDA_VM_ARGUE_LEAN_TAIL`).
static LEAN_TAILS: AtomicU64 = AtomicU64::new(0);
/// Of those, the ones `LAMBDA_VM_ARGUE_XCHECK` checked round by round against
/// the generic rounds.
static LEAN_TAIL_XCHECKS: AtomicU64 = AtomicU64::new(0);
/// Zerocheck sessions whose device rounds walked their program on demand
/// (`LAMBDA_VM_ARGUE_LEAN_PROGRAM`).
static LEAN_PROGRAMS: AtomicU64 = AtomicU64::new(0);
/// Of those, the ones `LAMBDA_VM_ARGUE_XCHECK` checked round by round against
/// today's program walking the same factors.
static LEAN_PROGRAM_XCHECKS: AtomicU64 = AtomicU64::new(0);
/// Fraction trees built and kept on device.
static TREE_CALLS: AtomicU64 = AtomicU64::new(0);
/// Tables whose factors were uploaded once and reused.
static FACTOR_CALLS: AtomicU64 = AtomicU64::new(0);
/// Openings whose two factors stayed on device across their groups.
static OPEN_CALLS: AtomicU64 = AtomicU64::new(0);
/// Of those, the ones that began in share form.
static LEAN_OPEN_CALLS: AtomicU64 = AtomicU64::new(0);
/// Device folds that ran every level in one launch.
static FUSED_FOLD_CALLS: AtomicU64 = AtomicU64::new(0);
/// Folds of a codeword the device holds, in one launch or one per level: what
/// [`FUSED_FOLD_CALLS`] is a share of.
static RESIDENT_FOLD_CALLS: AtomicU64 = AtomicU64::new(0);
/// ★ Stacked openings whose codeword the device holds but whose factors the
/// device did not build from the shares.
///
/// The opening's twin of [`HOST_FALLBACKS`], and until this counter it had no
/// name either: `open_shared` declining sends the chain to `Factors::new`,
/// which assembles the stacked polynomial and its weight table on the host — a
/// base copy of the stack and an extension table three times its size — before
/// trying the device once more with the tables built. Nothing else rises or
/// falls when that happens: [`OPEN_CALLS`] only fails to rise, and an expected
/// count is itself derived. It matters most where the card is fullest, which
/// is where an opening runs without the turn its group asked for (see
/// `stacked_eval::StackedCommitment`).
static OPEN_HOST_FALLBACKS: AtomicU64 = AtomicU64::new(0);
/// Groups that gave their room back when their commits ended.
static ROOM_PARKS: AtomicU64 = AtomicU64::new(0);
/// Group openings that took their room back.
static ROOM_TURNS: AtomicU64 = AtomicU64::new(0);
/// ★ Group openings whose room the card would not give back — see
/// [`take_turn`] for what the opening does then.
static ROOM_TURN_REFUSALS: AtomicU64 = AtomicU64::new(0);
/// Rooms a group sized, for its commits or its openings' turn.
static ROOM_SIZINGS: AtomicU64 = AtomicU64::new(0);
/// Of those, the ones sized to the turn they cover rather than to a whole
/// base codeword (`LAMBDA_VM_NO_WHIR_ROOM_RESIZE`).
static ROOMS_TURN_SIZED: AtomicU64 = AtomicU64::new(0);
/// ★ GKR fraction trees whose whole-tree promise the budget refused on the
/// consume path, after refusing their carry: the host built that table's
/// factors and tree. See [`note_gkr_tree_refusal`]; each one is also a
/// `math_cuda::device::device_fallbacks`.
static GKR_TREE_REFUSALS: AtomicU64 = AtomicU64::new(0);
/// Fraction trees whose input layer was written from the base columns
/// (`LAMBDA_VM_ARGUE_GKR_INPUT`).
static INPUT_COLUMNS_TREES: AtomicU64 = AtomicU64::new(0);

/// Fraction trees whose input layer was written from the base columns.
pub fn input_columns_trees() -> u64 {
    INPUT_COLUMNS_TREES.load(Ordering::Relaxed)
}

/// Tables whose factors were set up as base columns, with no lift
/// (`LAMBDA_VM_ARGUE_NO_LIFT`).
static COLUMN_FACTOR_TABLES: AtomicU64 = AtomicU64::new(0);

pub fn column_factor_tables() -> u64 {
    COLUMN_FACTOR_TABLES.load(Ordering::Relaxed)
}

/// Tables the no-lift path lifted after all: the fused rounds declined and
/// today's rounds need the lifted factors.
static LATE_LIFTS: AtomicU64 = AtomicU64::new(0);

pub fn late_lifts() -> u64 {
    LATE_LIFTS.load(Ordering::Relaxed)
}

pub(crate) fn note_late_lift() {
    LATE_LIFTS.fetch_add(1, Ordering::Relaxed);
    crate::whir_split::bump(&crate::whir_split::LATE_LIFT);
}

/// Under `LAMBDA_VM_ARGUE_TREE_SYNC` (a diagnostic), wait for the stream's
/// kernels so the tree's split charges each part its own card time.
#[cfg(feature = "cuda")]
macro_rules! tree_sync {
    ($stream:expr) => {
        if crate::whir_split::tree_sync() {
            let _ = $stream.synchronize();
        }
    };
}

pub fn commit_calls() -> u64 {
    COMMIT_CALLS.load(Ordering::Relaxed)
}

pub fn host_fallbacks() -> u64 {
    HOST_FALLBACKS.load(Ordering::Relaxed)
}

pub fn commit_errors() -> u64 {
    COMMIT_ERRORS.load(Ordering::Relaxed)
}

pub fn tree_commit_errors() -> u64 {
    TREE_COMMIT_ERRORS.load(Ordering::Relaxed)
}

/// Device errors printed per counter per process; past it only the counter
/// moves.
const DEVICE_ERRORS_LOGGED: u64 = 16;

/// Count one device error in `counter` and print the first
/// [`DEVICE_ERRORS_LOGGED`] of them. `describe` says what failed and where the
/// work goes instead; `kind` names the errors in the note on the last line
/// printed.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn note_device_error(counter: &AtomicU64, kind: &str, describe: impl FnOnce() -> String) {
    let seen = counter.fetch_add(1, Ordering::Relaxed);
    if seen < DEVICE_ERRORS_LOGGED {
        let last = if seen + 1 == DEVICE_ERRORS_LOGGED {
            format!(" (further {kind} are counted, not printed)")
        } else {
            String::new()
        };
        eprintln!("[whir] {}{last}", describe());
    }
}

/// Count and log a device commit that returned an error, before its caller
/// declines and the chain encodes on the host (where [`note_host_fallback`]
/// counts it too).
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn note_commit_error(what: &str, log_evals: usize, log_blowup: usize, err: &dyn core::fmt::Debug) {
    note_device_error(&COMMIT_ERRORS, "commit errors", || {
        format!(
            "device commit ({what}) of 2^{log_evals} evaluations at blowup 2^{log_blowup} \
             failed: {err:?}; encoding on the host"
        )
    });
}

/// Count and log a device tree over a host codeword that returned an error,
/// before its caller hashes the tree on the host.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn note_tree_commit_error(log_len: u32, log_folding: usize, err: &dyn core::fmt::Debug) {
    note_device_error(&TREE_COMMIT_ERRORS, "tree commit errors", || {
        format!(
            "device commit (ext3 tree) of 2^{log_len} values in blocks of 2^{log_folding} \
             failed: {err:?}; hashing on the host"
        )
    });
}

/// Called where a commit gives up on the device. Counts in non-cuda builds
/// too, where every commit takes that path and the number is the commit count.
pub(crate) fn note_host_fallback() {
    HOST_FALLBACKS.fetch_add(1, Ordering::Relaxed);
}

pub fn sumcheck_calls() -> u64 {
    SUMCHECK_CALLS.load(Ordering::Relaxed)
}

/// A device sumcheck another path ran (`gpu_fused`): counted as one call and
/// its rounds, as this file's own sessions are.
#[cfg(feature = "cuda")]
pub(crate) fn note_device_sumcheck(rounds: u64) {
    SUMCHECK_CALLS.fetch_add(1, Ordering::Relaxed);
    SUMCHECK_ROUNDS.fetch_add(rounds, Ordering::Relaxed);
}

pub fn sumcheck_rounds() -> u64 {
    SUMCHECK_ROUNDS.load(Ordering::Relaxed)
}

pub fn evaluate_calls() -> u64 {
    EVALUATE_CALLS.load(Ordering::Relaxed)
}

pub fn host_evaluate_calls() -> u64 {
    HOST_EVALUATE_CALLS.load(Ordering::Relaxed)
}

pub fn argue_xchecks() -> u64 {
    ARGUE_XCHECKS.load(Ordering::Relaxed)
}

pub fn argue_tables_on_card() -> u64 {
    ARGUE_TABLES_ON_CARD.load(Ordering::Relaxed)
}

pub fn argue_table_xchecks() -> u64 {
    ARGUE_TABLE_XCHECKS.load(Ordering::Relaxed)
}

pub fn reads_gathered() -> u64 {
    READS_GATHERED.load(Ordering::Relaxed)
}

pub fn reads_per_factor() -> u64 {
    READS_PER_FACTOR.load(Ordering::Relaxed)
}

pub fn lean_tails() -> u64 {
    LEAN_TAILS.load(Ordering::Relaxed)
}

pub fn lean_tail_xchecks() -> u64 {
    LEAN_TAIL_XCHECKS.load(Ordering::Relaxed)
}

pub fn lean_programs() -> u64 {
    LEAN_PROGRAMS.load(Ordering::Relaxed)
}

pub fn lean_program_xchecks() -> u64 {
    LEAN_PROGRAM_XCHECKS.load(Ordering::Relaxed)
}

/// Counts a layer the lean tail finished, and whether the cross-check ran on it.
pub(crate) fn note_lean_tail(xchecked: bool) {
    LEAN_TAILS.fetch_add(1, Ordering::Relaxed);
    crate::whir_split::bump(&crate::whir_split::TAILS_LEAN);
    if xchecked {
        LEAN_TAIL_XCHECKS.fetch_add(1, Ordering::Relaxed);
        crate::whir_split::bump(&crate::whir_split::TAILS_XCHECKED);
    }
}

pub fn tree_calls() -> u64 {
    TREE_CALLS.load(Ordering::Relaxed)
}

pub fn factor_calls() -> u64 {
    FACTOR_CALLS.load(Ordering::Relaxed)
}

/// Stacked openings that ran their first rounds over the shares
/// ([`math_cuda::whir_open::LeanRound0`]) rather than materialising up front.
pub fn lean_open_calls() -> u64 {
    LEAN_OPEN_CALLS.load(Ordering::Relaxed)
}

/// Device folds that ran every level in one launch.
pub fn fused_fold_calls() -> u64 {
    FUSED_FOLD_CALLS.load(Ordering::Relaxed)
}

/// Folds of a codeword the device holds, fused or level by level.
pub fn resident_fold_calls() -> u64 {
    RESIDENT_FOLD_CALLS.load(Ordering::Relaxed)
}

/// The rounds a stacked opening runs over its shares, as this process reads
/// `LAMBDA_VM_WHIR_LEAN_ROUNDS` (or a test's override). `0` in a build without
/// a device, where no opening runs there.
pub fn lean_rounds_in_effect() -> usize {
    #[cfg(feature = "cuda")]
    {
        lean_rounds()
    }
    #[cfg(not(feature = "cuda"))]
    {
        0
    }
}

/// Stacked openings over a device codeword whose factors were built on the
/// host — see [`OPEN_HOST_FALLBACKS`]. Zero in a build without a device, where
/// no codeword is on one.
pub fn open_host_fallbacks() -> u64 {
    OPEN_HOST_FALLBACKS.load(Ordering::Relaxed)
}

/// Called where a stacked opening over a device codeword builds its factors on
/// the host: counted, and said, because nothing else would show it.
pub(crate) fn note_open_host_fallback(n_stack: usize) {
    let count = OPEN_HOST_FALLBACKS.fetch_add(1, Ordering::Relaxed) + 1;
    eprintln!(
        "[whir] opening factors built on the HOST over a device codeword (stack 2^{n_stack}); \
         open host fallbacks {count}"
    );
}

pub fn open_calls() -> u64 {
    OPEN_CALLS.load(Ordering::Relaxed)
}

/// Groups that gave their room back when their commits ended.
pub fn room_parks() -> u64 {
    ROOM_PARKS.load(Ordering::Relaxed)
}

/// Group openings that took their room back.
pub fn room_turns() -> u64 {
    ROOM_TURNS.load(Ordering::Relaxed)
}

/// Group openings whose room the card would not give back.
pub fn room_turn_refusals() -> u64 {
    ROOM_TURN_REFUSALS.load(Ordering::Relaxed)
}

/// GKR fraction trees the host built because the card refused their promise.
pub fn gkr_tree_refusals() -> u64 {
    GKR_TREE_REFUSALS.load(Ordering::Relaxed)
}

/// Rooms a group sized, for its commits or its openings' turn.
pub fn room_sizings() -> u64 {
    ROOM_SIZINGS.load(Ordering::Relaxed)
}

/// Rooms sized to the turn they cover rather than to a whole base codeword.
pub fn rooms_turn_sized() -> u64 {
    ROOMS_TURN_SIZED.load(Ordering::Relaxed)
}

/// Called where a group sizes a room: `turn_sized` unless
/// `LAMBDA_VM_NO_WHIR_ROOM_RESIZE` has it promise a whole codeword.
pub(crate) fn note_room_sized(turn_sized: bool) {
    ROOM_SIZINGS.fetch_add(1, Ordering::Relaxed);
    if turn_sized {
        ROOMS_TURN_SIZED.fetch_add(1, Ordering::Relaxed);
    }
}

pub fn reset_call_counters() {
    COMMIT_CALLS.store(0, Ordering::Relaxed);
    HOST_FALLBACKS.store(0, Ordering::Relaxed);
    COMMIT_ERRORS.store(0, Ordering::Relaxed);
    TREE_COMMIT_ERRORS.store(0, Ordering::Relaxed);
    SUMCHECK_CALLS.store(0, Ordering::Relaxed);
    SUMCHECK_ROUNDS.store(0, Ordering::Relaxed);
    EVALUATE_CALLS.store(0, Ordering::Relaxed);
    HOST_EVALUATE_CALLS.store(0, Ordering::Relaxed);
    ARGUE_XCHECKS.store(0, Ordering::Relaxed);
    ARGUE_TABLES_ON_CARD.store(0, Ordering::Relaxed);
    ARGUE_TABLE_XCHECKS.store(0, Ordering::Relaxed);
    READS_GATHERED.store(0, Ordering::Relaxed);
    READS_PER_FACTOR.store(0, Ordering::Relaxed);
    LEAN_TAILS.store(0, Ordering::Relaxed);
    LEAN_TAIL_XCHECKS.store(0, Ordering::Relaxed);
    LEAN_PROGRAMS.store(0, Ordering::Relaxed);
    LEAN_PROGRAM_XCHECKS.store(0, Ordering::Relaxed);
    TREE_CALLS.store(0, Ordering::Relaxed);
    FACTOR_CALLS.store(0, Ordering::Relaxed);
    OPEN_CALLS.store(0, Ordering::Relaxed);
    LEAN_OPEN_CALLS.store(0, Ordering::Relaxed);
    FUSED_FOLD_CALLS.store(0, Ordering::Relaxed);
    RESIDENT_FOLD_CALLS.store(0, Ordering::Relaxed);
    OPEN_HOST_FALLBACKS.store(0, Ordering::Relaxed);
    ROOM_PARKS.store(0, Ordering::Relaxed);
    ROOM_TURNS.store(0, Ordering::Relaxed);
    ROOM_TURN_REFUSALS.store(0, Ordering::Relaxed);
    ROOM_SIZINGS.store(0, Ordering::Relaxed);
    ROOMS_TURN_SIZED.store(0, Ordering::Relaxed);
}

/// A sumcheck's round proofs, the challenges they drew, and what every slot
/// was bound to — the factors are folded where they lie, so their values at
/// the sumcheck's point are already there when the rounds end.
/// What a resident sumcheck's rounds on device leave: the rounds, the point
/// they drew, and **every** factor — resident and not — as the last fold left
/// it. One value each when the device ran the cube out, a cube when it stopped
/// at the crossover for the caller to finish.
type ResidentRounds<E> = (
    Vec<crate::sumcheck::RoundProof<E>>,
    Vec<math::field::element::FieldElement<E>>,
    Vec<crate::mle::Mle<E>>,
);

type SumcheckRounds<E> = (
    Vec<crate::sumcheck::RoundProof<E>>,
    Vec<math::field::element::FieldElement<E>>,
    Vec<crate::mle::Mle<E>>,
);

/// Codeword size below which the host wins: the kernels are a dozen launches
/// and a round trip, and a small NTT finishes in host cache before that.
#[cfg(feature = "cuda")]
const COMMIT_THRESHOLD: usize = 1 << 16;

/// Per query an authentication path, and the tree's Merkle cap.
pub(crate) type PathsAndCap = (Vec<Vec<[u8; 32]>>, Vec<[u8; 32]>);

/// A byte buffer of Merkle nodes, relabelled as nodes without copying.
///
/// A tree over a stacked polynomial is hundreds of megabytes; chunking it into
/// arrays would copy all of it to change nothing but the type.
#[cfg(feature = "cuda")]
fn nodes_in_place(bytes: Vec<u8>) -> Option<Vec<[u8; 32]>> {
    if !bytes.len().is_multiple_of(32) || !bytes.capacity().is_multiple_of(32) {
        return None;
    }
    // SAFETY: `[u8; 32]` has the alignment of `u8` and 32 times its size, so
    // the allocation describes the same bytes either way.
    let mut bytes = core::mem::ManuallyDrop::new(bytes);
    Some(unsafe {
        Vec::from_raw_parts(
            bytes.as_mut_ptr() as *mut [u8; 32],
            bytes.len() / 32,
            bytes.capacity() / 32,
        )
    })
}

/// Op tags the sumcheck kernel reads. MUST stay in sync with
/// `crypto/math-cuda/kernels/sumcheck.cu`.
pub mod op {
    pub const FIXED: u32 = 0;
    pub const VAR: u32 = 1;
    pub const ADD: u32 = 2;
    pub const SUB: u32 = 3;
    pub const MUL: u32 = 4;
    pub const NEG: u32 = 5;
}

/// A program lowered for the device: the nodes (two u64 each, `op | a << 32`
/// then `b | res << 32`), the ext3 constants they read (three u64 each), the
/// slot file's width and the slot the root lands in.
#[derive(Clone, Debug)]
pub struct Lowered {
    pub nodes: Vec<u64>,
    pub consts: Vec<u64>,
    pub num_slots: usize,
    pub root_slot: u32,
}

/// Slots the round kernel will hold per thread before the dispatch declines.
///
/// The slot file is `slots * 24 * threads` bytes and the scratch budget is
/// fixed, so a wider program buys fewer threads. This is where that stops
/// being a trade: at 8192 live values a single block of 256 already wants
/// 48 MiB, and a launch of one block is not a launch.
///
/// It is a cliff, not a dial. The real AIRs peak near a thousand — the widest
/// precompile lowers to 1036 — and a cap below that sends the tables with the
/// *most* work per row to the host, which is where they cost the most.
pub const MAX_SLOTS: usize = 8192;

/// Cube size below which the host wins for a table of one or two factors: the
/// rounds are a launch and a round trip each, and a small cube fits in cache.
#[cfg(feature = "cuda")]
const SUMCHECK_THRESHOLD: usize = 1 << 12;

/// Device bytes promised for as long as the returned handle lives.
///
/// The caller is a structure whose parts are allocated and freed one at a time
/// — a group of commitments and the working set its commits and openings take
/// turns with — so what it promises is the turn, not the sum.
#[cfg(feature = "cuda")]
pub fn reserve_room(bytes: u64) -> Option<DeviceRoom> {
    math_cuda::device::reserve(bytes).map(DeviceRoom)
}

#[cfg(not(feature = "cuda"))]
pub fn reserve_room(_bytes: u64) -> Option<DeviceRoom> {
    None
}

/// Whether a group gives its room back between its last commit and its first
/// opening, and takes it again for the openings ([`take_turn`]). `true` unless
/// `LAMBDA_VM_NO_WHIR_ROOM_PARK` is set, which holds it from the commit to the
/// last opening as before — the A/B's control, off the same binary.
///
/// Between the two runs the whole per-table argument, which reserves against
/// the same budget and never touches the room: the room is a promise for the
/// turns the commits and the openings take, and the argument takes none. Held
/// across it, it is a codeword's worth of budget the argument is refused.
pub(crate) fn room_park() -> bool {
    match ROOM_PARK_FORCED.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => {
            static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            !*OFF.get_or_init(|| std::env::var_os("LAMBDA_VM_NO_WHIR_ROOM_PARK").is_some())
        }
    }
}

static ROOM_PARK_FORCED: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Overrides `LAMBDA_VM_NO_WHIR_ROOM_PARK` for the whole process — for a test
/// that commits and opens both ways in one binary. `None` restores the
/// environment's setting. Not for production callers.
#[doc(hidden)]
pub fn force_room_park(on: Option<bool>) {
    ROOM_PARK_FORCED.store(
        match on {
            Some(true) => 1,
            Some(false) => 2,
            None => 0,
        },
        Ordering::Relaxed,
    );
}

/// Called where a group gives its room back after its commits.
pub(crate) fn note_room_parked() {
    ROOM_PARKS.fetch_add(1, Ordering::Relaxed);
}

/// Takes a parked group's room back for its openings: `bytes` promised until
/// the returned handle drops, or `None`, counted and said.
///
/// ⛔ WHY IT CAN BE REFUSED. Anything that reserves between the group's last
/// commit and this call can have taken the bytes. In the production drivers
/// that is the argument of the same proof, whose reservations are all released
/// by the time its last table is argued — and the leaf layers the commits
/// retained, which `reserve` evicts on a miss, so a retained layer never
/// outranks a turn. A second proof on the same card in the same process is the
/// case that can hold the bytes past this point, and a test binary running in
/// parallel is one.
///
/// ⛔ WHAT A REFUSAL DOES: the opening runs on the device anyway, UNPROMISED.
/// Its codewords are on the card with no host copy (`fold_held` has nowhere to
/// fold a device codeword the device declines) and the transcript has absorbed
/// their roots, so there is no host path to fall back to — declining here would
/// be failing the proof. What runs unpromised is one opening's working set:
/// with the lean opening and the fused fold, about a quarter of a codeword —
/// where before them every opening ran nearly three codewords past the room it
/// had. If the card then really is full, the factors fall to the host and
/// [`open_host_fallbacks`] counts them, or a fold fails with
/// [`crate::Error::DeviceFailed`] — both loud, neither silent.
pub(crate) fn take_turn(bytes: u64, polys: usize, n_stack: usize) -> Option<DeviceRoom> {
    if let Some(room) = reserve_room(bytes) {
        ROOM_TURNS.fetch_add(1, Ordering::Relaxed);
        return Some(room);
    }
    let count = ROOM_TURN_REFUSALS.fetch_add(1, Ordering::Relaxed) + 1;
    eprintln!(
        "[whir] opening turn REFUSED: {} MiB for a group of {polys} at 2^{n_stack}, {} — \
         opening on the device UNPROMISED; room turn refusals {count}",
        bytes >> 20,
        budget_headroom(),
    );
    None
}

/// The budget left to promise, for a refusal's line.
#[cfg(feature = "cuda")]
fn budget_headroom() -> String {
    match math_cuda::device::backend() {
        Ok(be) => format!(
            "budget headroom {} MiB",
            be.vram_budget_bytes().saturating_sub(be.reserved_bytes()) >> 20
        ),
        Err(_) => "no device".to_string(),
    }
}

#[cfg(not(feature = "cuda"))]
fn budget_headroom() -> String {
    "no device".to_string()
}

/// Opens a window over the device ledger's peak
/// ([`math_cuda::device::reset_window_high_water`]); nothing without a device.
#[cfg(feature = "cuda")]
pub fn reset_reserved_window() {
    math_cuda::device::reset_window_high_water();
}

#[cfg(not(feature = "cuda"))]
pub fn reset_reserved_window() {}

/// The ledger's peak since [`reset_reserved_window`]; 0 without a device.
#[cfg(feature = "cuda")]
pub fn reserved_window_peak() -> u64 {
    math_cuda::device::window_high_water()
}

#[cfg(not(feature = "cuda"))]
pub fn reserved_window_peak() -> u64 {
    0
}

/// What the ledger may promise in all; 0 without a device.
#[cfg(feature = "cuda")]
pub fn reserve_budget() -> u64 {
    math_cuda::device::backend()
        .map(|be| be.vram_budget_bytes())
        .unwrap_or(0)
}

#[cfg(not(feature = "cuda"))]
pub fn reserve_budget() -> u64 {
    0
}

/// Whether a group's room is sized to the turns it covers — its commits', two
/// in flight, and one chain's opening, as the kernels that run them allocate —
/// rather than one base codeword. `true` unless `LAMBDA_VM_NO_WHIR_ROOM_RESIZE`
/// is set, which promises the codeword as before.
///
/// One codeword was the right number once: two commits in flight at a fold of
/// four hold (half a codeword, less a node) each. At the production first fold
/// of six a commit holds a quarter, and an opening under the lean rounds and
/// the fused fold about a fifth — where before them an opening held nearly
/// four codewords against the one it had promised.
pub(crate) fn room_resize() -> bool {
    match ROOM_RESIZE_FORCED.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => {
            static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            !*OFF.get_or_init(|| std::env::var_os("LAMBDA_VM_NO_WHIR_ROOM_RESIZE").is_some())
        }
    }
}

static ROOM_RESIZE_FORCED: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Overrides `LAMBDA_VM_NO_WHIR_ROOM_RESIZE` for the whole process. `None`
/// restores the environment's setting. Not for production callers.
#[doc(hidden)]
pub fn force_room_resize(on: Option<bool>) {
    ROOM_RESIZE_FORCED.store(
        match on {
            Some(true) => 1,
            Some(false) => 2,
            None => 0,
        },
        Ordering::Relaxed,
    );
}

/// The two turns a stacked group's room covers, in bytes beside the codewords.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RoomTurns {
    /// The commits', two in flight ([`math_cuda::whir::commit_transient_bytes`]).
    pub commit: u64,
    /// One chain's opening at a time, from the kernels that will run it
    /// ([`math_cuda::whir_open::opening_transient_bytes`]).
    pub open: u64,
}

/// [`RoomTurns`] for a group laid out as `layout`, opened from resident
/// columns or not. Every stacked polynomial of a group has the same variable
/// count, so the widest share list and the most `eq` values over them bound
/// every chain's opening.
#[cfg(feature = "cuda")]
pub fn room_turns_for(
    layout: &crate::stacking::StackedLayout,
    config: &crate::whir_chain::ChainConfig,
    staged: bool,
) -> RoomTurns {
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;

    let n = layout.n_stack();
    let schedule = config.schedule(n);
    let first_fold = schedule.first().copied().unwrap_or(0);
    let next_fold = schedule.get(1).copied().unwrap_or(0);
    let in_flight = layout.num_polys().min(2) as u64;
    let commit =
        in_flight * math_cuda::whir::commit_transient_bytes(n, config.log_blowup, first_fold);
    // A share's `eq` halves are its point's high and low coordinates' tables,
    // the low half `num_vars / 2` wide — counted per share, with no credit for
    // the neighbours that share a point.
    let mut shares = vec![0usize; layout.num_polys()];
    let mut eq_values = vec![0usize; layout.num_polys()];
    for place in layout.placements() {
        shares[place.poly] += 1;
        let lo_bits = place.num_vars / 2;
        eq_values[place.poly] += (1usize << (place.num_vars - lo_bits)) + (1usize << lo_bits);
    }
    let slots = crate::whir_chain::opening_program::<Ext3>()
        .ok()
        .and_then(|program| lower(&program))
        .map_or(0, |lowered| lowered.num_slots);
    let open = math_cuda::whir_open::opening_transient_bytes(&math_cuda::whir_open::OpeningShape {
        num_vars: n,
        log_blowup: config.log_blowup,
        first_fold,
        next_fold,
        lean_rounds: lean_rounds(),
        fused: fused_fold(),
        staged,
        shares: shares.iter().copied().max().unwrap_or(0),
        eq_values: eq_values.iter().copied().max().unwrap_or(0),
        slots,
    });
    RoomTurns { commit, open }
}

/// No device: nothing is ever promised, so nothing is sized.
#[cfg(not(feature = "cuda"))]
pub fn room_turns_for(
    _layout: &crate::stacking::StackedLayout,
    _config: &crate::whir_chain::ChainConfig,
    _staged: bool,
) -> RoomTurns {
    RoomTurns { commit: 0, open: 0 }
}

/// What a device sumcheck that runs to the end hands back: the round proofs,
/// and the challenges the rounds were bound at.
///
/// A named type rather than the tuple written out, because the tuple appears in
/// a return position wrapped in a `Result` and reads as punctuation there. The
/// sibling [`ResidentRounds`] carries a third member — the tables left folded
/// at the crossover — for the path that stops early.
#[cfg(feature = "cuda")]
type ClosedRounds<E> = (
    Vec<crate::sumcheck::RoundProof<E>>,
    Vec<math::field::element::FieldElement<E>>,
);

/// A promise held on someone else's behalf. Dropping it gives the room back.
///
/// The field is never read, and that is the design: it is an RAII guard whose
/// `Drop` returns the reservation, so holding it IS the behaviour. Deleting it
/// to satisfy the lint would delete the promise.
#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct DeviceRoom(#[allow(dead_code)] math_cuda::device::DeviceReservation);

/// A promise no device made. Never constructed.
#[cfg(not(feature = "cuda"))]
#[derive(Debug)]
pub struct DeviceRoom(std::convert::Infallible);

impl DeviceRoom {
    /// Gives the promise back now, rather than when its holder drops.
    pub(crate) fn give_back(self) {}
}

/// How much room handing the input layer back has to save before it is worth
/// doing.
///
/// It cannot fail while it is being carried, and it can fail when it is asked
/// for again — with the transcript already moved and no host path left. A
/// table that saves a few megabytes buys a second chance to fail for nothing;
/// the widest precompiles save gigabytes, and for them it is the difference
/// between a device and the host.
#[cfg(feature = "cuda")]
const WORTH_HANDING_BACK: u64 = 1 << 30;

/// [`WORTH_HANDING_BACK`], unless a test has set its own.
#[cfg(feature = "cuda")]
fn worth_handing_back() -> u64 {
    match HAND_BACK_FOR_TESTS.load(Ordering::Relaxed) {
        u64::MAX => WORTH_HANDING_BACK,
        bytes => bytes,
    }
}

/// A test's hand-back threshold; `u64::MAX` is none.
static HAND_BACK_FOR_TESTS: AtomicU64 = AtomicU64::new(u64::MAX);

/// Tests only: the room a tree's carry must save before it is handed back,
/// in place of [`WORTH_HANDING_BACK`]; `None` restores it. At `Some(0)` every
/// refused carry is handed back, which is the only way a table smaller than the
/// widest precompiles reaches the tree's whole-tree promise and its refusal.
/// Process-wide: a test that sets it runs alone.
#[doc(hidden)]
pub fn set_hand_back_threshold_for_tests(bytes: Option<u64>) {
    HAND_BACK_FOR_TESTS.store(bytes.unwrap_or(u64::MAX), Ordering::Relaxed);
}

/// Tests only: every tree built from here on hands its input layer back and
/// writes it again for its last sumcheck, as the widest precompiles do when the
/// card cannot carry it — so a test can reach the rewrite on any table.
/// Process-wide: a test that sets it runs alone.
static HAND_BACK_FORCED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[doc(hidden)]
pub fn force_hand_back_for_tests(on: bool) {
    HAND_BACK_FORCED.store(on, Ordering::Relaxed);
}

/// Cells — factors times rows — below which the host wins.
///
/// The precompiles are short and very wide: a thousand factors over four
/// thousand rows is four million cells, and a gate that reads the row count
/// alone calls that small. It is not, and it is the shape where the host costs
/// the most, so what the gate has to read is the work.
#[cfg(feature = "cuda")]
const DEVICE_CELLS: usize = 1 << 12;

/// A cube too short to fill a few warps is a round trip for nothing, however
/// wide the table is.
#[cfg(feature = "cuda")]
const MIN_CUBE: usize = 1 << 6;

/// Whether a table of `width` factors over a cube of `len` is worth a device.
#[cfg(feature = "cuda")]
fn worth_the_device(width: usize, len: usize) -> bool {
    len >= MIN_CUBE && width.saturating_mul(len) >= DEVICE_CELLS
}

/// Assigns every step a slot, reusing the slot of a value whose last read has
/// passed.
///
/// This is what makes the kernel possible at all: the precompile tables compile
/// to tens of thousands of steps, and a slot per step would be a megabyte of
/// scratch per thread.
pub fn lower<E>(program: &crate::program::Program<E>) -> Option<Lowered>
where
    E: math::field::traits::IsField + 'static,
{
    use crate::program::Op;
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;

    // The blob is ext3: the kernel reads three limbs per value, whether or not
    // this particular program happens to hold a constant that would say so.
    if std::any::TypeId::of::<E>() != std::any::TypeId::of::<Ext3>() {
        return None;
    }
    let steps = program.steps();
    // Last step that reads each value; the root is read by the caller, so it is
    // never freed.
    let mut last_use: Vec<usize> = vec![0; steps.len()];
    for (i, step) in steps.iter().enumerate() {
        let mut mark = |operand: u32| last_use[operand as usize] = i;
        match *step {
            Op::Fixed(_) | Op::Var(_) => {}
            Op::Neg(a) => mark(a),
            Op::Add(a, b) | Op::Sub(a, b) | Op::Mul(a, b) => {
                mark(a);
                mark(b);
            }
        }
    }
    last_use[program.root() as usize] = usize::MAX;

    let mut slot_of: Vec<u32> = vec![u32::MAX; steps.len()];
    let mut free: Vec<u32> = Vec::new();
    let mut num_slots = 0usize;
    let mut nodes: Vec<u64> = Vec::with_capacity(steps.len() * 2);
    let mut consts: Vec<u64> = Vec::new();

    for (i, step) in steps.iter().enumerate() {
        let (op, a, b, operands) = match *step {
            Op::Fixed(ref value) => {
                let at = consts.len() / 3;
                consts.extend_from_slice(&ext3_raw(value)?);
                (op::FIXED, at as u32, 0, [None, None])
            }
            Op::Var(slot) => (op::VAR, slot, 0, [None, None]),
            Op::Neg(a) => (op::NEG, slot_of[a as usize], 0, [Some(a), None]),
            Op::Add(a, b) | Op::Sub(a, b) | Op::Mul(a, b) => {
                let tag = match *step {
                    Op::Add(..) => op::ADD,
                    Op::Sub(..) => op::SUB,
                    _ => op::MUL,
                };
                (
                    tag,
                    slot_of[a as usize],
                    slot_of[b as usize],
                    [Some(a), Some(b)],
                )
            }
        };
        // Freed before the result is allocated: the kernel loads both operands
        // before it stores, so the result may take a slot this step frees. A
        // value read twice — `x·x`, which is what a squaring compiles to —
        // frees its slot once, or two later values would be handed the same
        // one.
        if let Some(x) = operands[0].filter(|x| last_use[*x as usize] == i) {
            free.push(slot_of[x as usize]);
        }
        if let Some(y) =
            operands[1].filter(|y| last_use[*y as usize] == i && operands[0] != Some(*y))
        {
            free.push(slot_of[y as usize]);
        }
        let res = free.pop().unwrap_or_else(|| {
            let slot = num_slots as u32;
            num_slots += 1;
            slot
        });
        if num_slots > MAX_SLOTS {
            return None;
        }
        slot_of[i] = res;
        nodes.push(u64::from(op) | (u64::from(a) << 32));
        nodes.push(u64::from(b) | (u64::from(res) << 32));
    }

    Some(Lowered {
        nodes,
        consts,
        num_slots,
        root_slot: slot_of[program.root() as usize],
    })
}

/// An ext3 element's three limbs, or `None` when `E` is not that field.
pub fn ext3_raw<E>(value: &math::field::element::FieldElement<E>) -> Option<[u64; 3]>
where
    E: math::field::traits::IsField + 'static,
{
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;
    if std::any::TypeId::of::<E>() != std::any::TypeId::of::<Ext3>() {
        return None;
    }
    // SAFETY: `E == Ext3`, whose `FieldElement` is a transparent wrapper over
    // three Goldilocks limbs, each transparent over its `u64`.
    let limbs = unsafe { *(value as *const _ as *const [u64; 3]) };
    Some(limbs)
}

/// Rebuilds an ext3 element from its three limbs.
pub fn ext3_from_raw<E>(limbs: &[u64]) -> math::field::element::FieldElement<E>
where
    E: math::field::traits::IsField + 'static,
{
    use math::field::element::FieldElement;
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;
    use math::field::goldilocks::GoldilocksField as Gl;

    let value = FieldElement::<Ext3>::new([
        FieldElement::<Gl>::from_raw(limbs[0]),
        FieldElement::<Gl>::from_raw(limbs[1]),
        FieldElement::<Gl>::from_raw(limbs[2]),
    ]);
    // SAFETY: only called under a TypeId check that `E == Ext3`.
    unsafe { core::mem::transmute_copy::<FieldElement<Ext3>, FieldElement<E>>(&value) }
}

/// Runs a batched sumcheck's rounds on device.
///
/// `challenge` absorbs a round's evaluations and returns the challenge drawn
/// from them, which is the whole of the host's part: the transcript is
/// sequential by definition and stays where it is.
///
/// The factors are consumed — they are left folded on device and dropped — so
/// this is for a caller that wants the rounds and the point, not the tables.
/// The prover's `g(0) + g(1)` self-check does not run on this path: it costs
/// the extra interpolation node the protocol exists to skip.
///
/// `None` means the device declined **before the transcript moved**, and the
/// host path runs. `Some(Err)` means it failed after: the challenges are drawn,
/// the transcript cannot be rewound, and the proof fails rather than being
/// finished from a state the verifier will not reproduce.
#[cfg(feature = "cuda")]
pub(crate) fn prove_sumcheck<E>(
    polys: &[crate::mle::Mle<E>],
    program: &crate::program::Program<E>,
    degree: usize,
    host_cube: usize,
    challenge: impl FnMut(
        &[math::field::element::FieldElement<E>],
    ) -> math::field::element::FieldElement<E>,
) -> Option<Result<SumcheckRounds<E>, crate::Error>>
where
    E: math::field::traits::IsField + 'static,
{
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;

    if std::any::TypeId::of::<E>() != std::any::TypeId::of::<Ext3>() {
        return None;
    }
    let first = polys.first()?;
    let num_vars = first.num_vars();
    if !worth_the_device(polys.len(), first.len()) || num_vars == 0 {
        return None;
    }
    if degree == 0 || degree > math_cuda::sumcheck::MAX_NODES {
        return None;
    }
    if polys.iter().any(|p| p.len() != first.len()) {
        return None;
    }
    // A slot past the factor list would be an out-of-bounds device read, which
    // no kernel can check for itself.
    if program
        .max_slot()
        .is_some_and(|slot| slot as usize >= polys.len())
    {
        return None;
    }
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *DISABLED.get_or_init(|| std::env::var_os("LAMBDA_VM_NO_GPU_SUMCHECK").is_some()) {
        return None;
    }
    let lowered = lower(program)?;

    // SAFETY: `E == Ext3` is established above, and its `FieldElement` is a
    // transparent wrapper over three `u64` limbs — the layout the kernel reads.
    let raw: Vec<&[u64]> = polys
        .iter()
        .map(|p| unsafe {
            core::slice::from_raw_parts(p.evals().as_ptr() as *const u64, p.len() * 3)
        })
        .collect();

    let mut session = math_cuda::sumcheck::SumcheckSession::new(
        &raw,
        &lowered.nodes,
        &lowered.consts,
        lowered.num_slots,
        lowered.root_slot,
    )
    .ok()?;

    // Diagnostic hook: recompute each round on the host from the factors the
    // device holds and stop at the first disagreement, naming the round. A
    // device round that differs otherwise surfaces as a proof that does not
    // verify, minutes and 55 tables later.
    static XCHECK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let xcheck = *XCHECK.get_or_init(|| std::env::var_os("LAMBDA_VM_GPU_XCHECK").is_some());
    let reference = |session: &math_cuda::sumcheck::SumcheckSession| {
        // A session over factors that were already there has no layout to read
        // back, so there is nothing to rebuild the host round from.
        if !xcheck || !session.can_download() {
            return None;
        }
        let tables = session.download().expect("the device holds its factors");
        let factors: Vec<crate::mle::Mle<E>> = tables
            .iter()
            .map(|table| {
                crate::mle::Mle::new(table.chunks_exact(3).map(ext3_from_raw::<E>).collect())
            })
            .collect::<Result<_, _>>()
            .expect("the device holds power-of-two tables");
        Some(
            crate::sumcheck::round_evaluations_for_program(&factors, program, degree)
                .expect("the host round"),
        )
    };

    // The rounds stop where the cube reaches the crossover: past it a round is
    // one thread walking the whole program, and a core here walks it far
    // faster. The caller carries on from the factors this leaves folded.
    let there = num_vars.saturating_sub(host_cube.max(1).trailing_zeros() as usize);
    let outcome = run_rounds(&mut session, degree, there, challenge, reference);
    let (rounds, challenges) = match outcome {
        Ok(rounds) => rounds,
        Err(error) => return Some(Err(error)),
    };
    // Each factor as the last fold left it, read where it lies — which is the
    // only thing a session over factors it does not own can say.
    let Ok(values) = session_values(&session) else {
        return Some(Err(crate::Error::DeviceFailed { stage: "download" }));
    };
    let folded: Result<Vec<crate::mle::Mle<E>>, crate::Error> = values
        .iter()
        .map(|factor| {
            crate::mle::Mle::new(factor.chunks_exact(3).map(ext3_from_raw::<E>).collect())
        })
        .collect();
    let Ok(folded) = folded else {
        return Some(Err(crate::Error::DeviceFailed {
            stage: "folded tables",
        }));
    };
    SUMCHECK_CALLS.fetch_add(1, Ordering::Relaxed);
    Some(Ok((rounds, challenges, folded)))
}

/// The round loop: the device sums over the cube, the host draws the challenge
/// from what it sent, the device binds it.
///
/// Past the first round the transcript has moved, so every failure here is an
/// error — there is no going back to the host path.
#[cfg(feature = "cuda")]
fn run_rounds<E>(
    session: &mut math_cuda::sumcheck::SumcheckSession,
    degree: usize,
    num_vars: usize,
    challenge: impl FnMut(
        &[math::field::element::FieldElement<E>],
    ) -> math::field::element::FieldElement<E>,
    reference: impl FnMut(
        &math_cuda::sumcheck::SumcheckSession,
    ) -> Option<Vec<math::field::element::FieldElement<E>>>,
) -> Result<ClosedRounds<E>, crate::Error>
where
    E: math::field::traits::IsField + 'static,
{
    run_rounds_timed(session, degree, num_vars, challenge, reference, None)
}

/// Where a zerocheck's device rounds go on the base split's `ARGUE ZEROCHECK`
/// line: a big batch's split at [`whir_split::LATE_HALF`](crate::whir_split::LATE_HALF)
/// into early and late rounds, every other batch's together.
#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
enum ZerocheckRounds {
    Big,
    Other,
}

#[cfg(feature = "cuda")]
impl ZerocheckRounds {
    fn note(self, half: usize, start: Option<std::time::Instant>) {
        use crate::whir_split::{self, add_tick, bump};
        match self {
            Self::Big if half >= whir_split::LATE_HALF => {
                add_tick(&whir_split::ZC_BIG_EARLY, start);
                bump(&whir_split::ZC_BIG_EARLY_ROUNDS);
            }
            Self::Big => {
                add_tick(&whir_split::ZC_BIG_LATE, start);
                bump(&whir_split::ZC_BIG_LATE_ROUNDS);
            }
            Self::Other => add_tick(&whir_split::ZC_OTHER, start),
        }
    }
}

/// Today's program walking a zerocheck's factors beside the program on demand
/// (`LAMBDA_VM_ARGUE_XCHECK`): each round, the shadow's sums are taken before
/// the leader's round and compared with its answer before the challenge is
/// drawn. Inert without a shadow.
#[cfg(feature = "cuda")]
struct ShadowRounds {
    shadow: std::cell::RefCell<Option<math_cuda::sumcheck::SumcheckSession>>,
    /// The interpolation nodes, as the round loop sends them.
    t: Vec<u64>,
    /// The shadow's sums for the round under way; empty if its round failed.
    expected: std::cell::RefCell<Option<Vec<u64>>>,
    round: std::cell::Cell<usize>,
    /// The first round the two disagreed at.
    parted: std::cell::Cell<Option<usize>>,
}

#[cfg(feature = "cuda")]
impl ShadowRounds {
    fn new(shadow: Option<math_cuda::sumcheck::SumcheckSession>, degree: usize) -> Option<Self> {
        use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;
        let mut t = Vec::with_capacity(degree * 3);
        for node in 1..=degree as u64 {
            t.extend_from_slice(&ext3_raw(
                &math::field::element::FieldElement::<Ext3>::from(node),
            )?);
        }
        Some(Self {
            shadow: std::cell::RefCell::new(shadow),
            t,
            expected: std::cell::RefCell::new(None),
            round: std::cell::Cell::new(0),
            parted: std::cell::Cell::new(None),
        })
    }

    /// Before the leader's round: the shadow's, over the cube the leader has
    /// left. Never a reference for the round loop to assert on.
    fn walk<E>(
        &self,
        leader: &math_cuda::sumcheck::SumcheckSession,
    ) -> Option<Vec<math::field::element::FieldElement<E>>>
    where
        E: math::field::traits::IsField + 'static,
    {
        if let Some(shadow) = self.shadow.borrow_mut().as_mut() {
            shadow.follow(leader);
            *self.expected.borrow_mut() = Some(shadow.round(&self.t).unwrap_or_default());
        }
        None
    }

    /// After it: the leader's answer against the shadow's sums.
    fn compare<E>(&self, evaluations: &[math::field::element::FieldElement<E>])
    where
        E: math::field::traits::IsField + 'static,
    {
        let round = self.round.replace(self.round.get() + 1);
        let Some(sums) = self.expected.borrow_mut().take() else {
            return;
        };
        let same = sums.len() == evaluations.len() * 3
            && sums
                .chunks_exact(3)
                .zip(evaluations)
                .all(|(sum, value)| ext3_from_raw::<E>(sum) == *value);
        if !same && self.parted.get().is_none() {
            self.parted.set(Some(round));
        }
    }

    fn parted(&self) -> Option<usize> {
        self.parted.get()
    }
}

/// [`run_rounds`], each round — its kernels, its read-back, the challenge drawn
/// and its fold — timed onto the zerocheck's split when `zerocheck` says which.
#[cfg(feature = "cuda")]
fn run_rounds_timed<E>(
    session: &mut math_cuda::sumcheck::SumcheckSession,
    degree: usize,
    num_vars: usize,
    mut challenge: impl FnMut(
        &[math::field::element::FieldElement<E>],
    ) -> math::field::element::FieldElement<E>,
    mut reference: impl FnMut(
        &math_cuda::sumcheck::SumcheckSession,
    ) -> Option<Vec<math::field::element::FieldElement<E>>>,
    zerocheck: Option<ZerocheckRounds>,
) -> Result<ClosedRounds<E>, crate::Error>
where
    E: math::field::traits::IsField + 'static,
{
    use math::field::element::FieldElement;

    // The interpolation nodes are `1..=degree`: `g(0)` is not sent, the claim
    // carried into the round fixes it.
    let mut t = Vec::with_capacity(degree * 3);
    for node in 1..=degree {
        t.extend_from_slice(&ext3_raw(&FieldElement::<E>::from(node as u64)).ok_or(
            crate::Error::DeviceFailed {
                stage: "interpolation node",
            },
        )?);
    }

    let failed = |stage| crate::Error::DeviceFailed { stage };
    let mut rounds = Vec::with_capacity(num_vars);
    let mut challenges = Vec::with_capacity(num_vars);
    for round in 0..num_vars {
        let expected = reference(session);
        let half = session.len() / 2;
        let start = zerocheck.and_then(|_| crate::whir_split::tick());
        let sums = session.round(&t).map_err(|_| failed("round"))?;
        let evaluations: Vec<FieldElement<E>> =
            sums.chunks_exact(3).map(ext3_from_raw::<E>).collect();
        if let Some(expected) = expected {
            assert_eq!(
                evaluations, expected,
                "the device round {round} differs from the host"
            );
        }
        let r = challenge(&evaluations);
        let raw = ext3_raw(&r).ok_or_else(|| failed("challenge"))?;
        session.fold(&raw).map_err(|_| failed("fold"))?;
        if let Some(zerocheck) = zerocheck {
            zerocheck.note(half, start);
        }
        rounds.push(crate::sumcheck::RoundProof { evaluations });
        challenges.push(r);
        SUMCHECK_ROUNDS.fetch_add(1, Ordering::Relaxed);
    }
    Ok((rounds, challenges))
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn prove_sumcheck<E>(
    _polys: &[crate::mle::Mle<E>],
    _program: &crate::program::Program<E>,
    _degree: usize,
    _host_cube: usize,
    _challenge: impl FnMut(
        &[math::field::element::FieldElement<E>],
    ) -> math::field::element::FieldElement<E>,
) -> Option<Result<SumcheckRounds<E>, crate::Error>>
where
    E: math::field::traits::IsField + 'static,
{
    None
}

/// The same sumcheck, with the trace's factors already on the device and only
/// the weight tables here.
///
/// The rounds fold the resident factors where they lie, which spends them: a
/// table's argument runs one of these.
///
/// `None` is a decline, and it is always before the first round — so the
/// caller can still build those factors here and run the host path from the
/// same transcript. Once the rounds start, a failure is a failure.
#[cfg(feature = "cuda")]
pub(crate) fn prove_sumcheck_resident<E>(
    resident: &DeviceFactors,
    extra: &[WeightView<'_, E>],
    program: &crate::program::Program<E>,
    degree: usize,
    challenge: impl FnMut(
        &[math::field::element::FieldElement<E>],
    ) -> math::field::element::FieldElement<E>,
) -> Option<Result<ResidentRounds<E>, crate::Error>>
where
    E: math::field::traits::IsField + 'static,
{
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;

    if std::any::TypeId::of::<E>() != std::any::TypeId::of::<Ext3>() {
        return None;
    }
    let len = resident.0.len();
    let num_vars = len.trailing_zeros() as usize;
    if !worth_the_device(resident.0.width() + extra.len(), len) || num_vars == 0 {
        return None;
    }
    if degree == 0 || degree > math_cuda::sumcheck::MAX_NODES {
        return None;
    }
    if extra.iter().any(|weight| weight.cells() != len) {
        return None;
    }
    // A slot past the factor list would be an out-of-bounds device read, which
    // no kernel can check for itself.
    let width = resident.0.width() + extra.len();
    if program
        .max_slot()
        .is_some_and(|slot| slot as usize >= width)
    {
        return None;
    }
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *DISABLED.get_or_init(|| std::env::var_os("LAMBDA_VM_NO_GPU_SUMCHECK").is_some()) {
        return None;
    }
    // A big batch's program on demand holds few enough values that its slot
    // file is sized for every node from the first round.
    let ZerocheckLowering {
        run: lowered,
        held,
        big,
    } = lower_for_zerocheck(program)?;
    let lean = held.is_some();
    let spread = if lean { degree } else { 1 };

    // A table goes up as it is; an `eq` weight goes up as its point, and the
    // card builds it (`LAMBDA_VM_ARGUE_DEVICE_TABLES`).
    let mut points = Vec::with_capacity(extra.len());
    for weight in extra {
        points.push(match weight {
            WeightView::Eq(point) => raw_point(point)?,
            WeightView::Table(_) => Vec::new(),
        });
    }
    let raw: Vec<math_cuda::sumcheck::Extra<'_>> = extra
        .iter()
        .zip(&points)
        .map(|(weight, point)| match weight {
            // SAFETY: `E == Ext3` is established above, and its `FieldElement`
            // is a transparent wrapper over three `u64` limbs — the layout the
            // kernel reads.
            WeightView::Table(table) => math_cuda::sumcheck::Extra::Table(unsafe {
                core::slice::from_raw_parts(table.evals().as_ptr() as *const u64, table.len() * 3)
            }),
            WeightView::Eq(_) => math_cuda::sumcheck::Extra::Eq(point),
        })
        .collect();
    let mut session = resident
        .0
        .session_spread(
            &raw,
            &lowered.nodes,
            &lowered.consts,
            lowered.num_slots,
            lowered.root_slot,
            spread,
        )
        .ok()?;
    // Under the cross-check, today's program walks the same factors beside it,
    // a round at a time: the block's identity gate for the schedule, as proof
    // bytes cannot be one. A shadow the card has no room for declines here,
    // before the first round, and is counted as unchecked.
    let shadow = match &held {
        Some(today) if argue_xcheck() => Some(
            session
                .shadow(
                    &today.nodes,
                    &today.consts,
                    today.num_slots,
                    today.root_slot,
                )
                .ok()?,
        ),
        _ => None,
    };
    let checked = shadow.is_some();
    let beside = ShadowRounds::new(shadow, degree)?;
    if big {
        crate::whir_split::bump(&crate::whir_split::ZC_BIG);
    }
    if lean {
        LEAN_PROGRAMS.fetch_add(1, Ordering::Relaxed);
        crate::whir_split::bump(&crate::whir_split::ZC_LEAN);
    }

    // The weights the card built, by their slot in the session.
    let built: Vec<(usize, &[math::field::element::FieldElement<E>])> = extra
        .iter()
        .enumerate()
        .filter_map(|(k, weight)| match weight {
            WeightView::Eq(point) => Some((resident.0.width() + k, *point)),
            WeightView::Table(_) => None,
        })
        .collect();
    if let Some(&(slot, _)) = built.first() {
        note_tables_on_card(built.len());
        if table_fault() == Some(TableFault::Eq) {
            session.set_cell(slot, 0, &FAULT_CELL).ok()?;
        }
    }
    // Nothing is absorbed until the first round, so a check that cannot read
    // the card still declines; one that reads a wrong table fails the proof.
    if argue_xcheck() {
        for &(slot, point) in &built {
            let card = session.factor(slot).ok()?;
            if let Some(cell) = first_difference(&card, &crate::eq::eq_evals(point)) {
                eprintln!(
                    "[argue] XCHECK: the card's eq weight in slot {slot} (2^{num_vars} cells) is not \
                     the host's at cell {cell}"
                );
                return Some(Err(crate::Error::DeviceFailed { stage: "eq weight" }));
            }
            note_tables_xchecked(1);
        }
    }

    // The rounds stop where the cube reaches the crossover: past it a round is
    // one thread walking the whole program, and a core here walks it far
    // faster. The caller finishes from the factors this hands back.
    let there = num_vars.saturating_sub(crate::HOST_CUBE_COMPILED.trailing_zeros() as usize);
    let mut challenge = challenge;
    let outcome = run_rounds_timed(
        &mut session,
        degree,
        there,
        |evaluations| {
            beside.compare(evaluations);
            challenge(evaluations)
        },
        |leader| beside.walk(leader),
        Some(if big {
            ZerocheckRounds::Big
        } else {
            ZerocheckRounds::Other
        }),
    );
    let (rounds, challenges) = match outcome {
        Ok(rounds) => rounds,
        Err(error) => return Some(Err(error)),
    };
    if let Some(round) = beside.parted() {
        eprintln!(
            "[argue] XCHECK: a {num_vars}-variable zerocheck's rounds on demand ({} values a \
             thread) parted from today's program ({} values) at round {round}",
            lowered.num_slots,
            held.as_ref().map_or(0, |today| today.num_slots),
        );
        return Some(Err(crate::Error::DeviceFailed {
            stage: "program on demand",
        }));
    }
    if checked {
        LEAN_PROGRAM_XCHECKS.fetch_add(1, Ordering::Relaxed);
        crate::whir_split::bump(&crate::whir_split::ZC_LEAN_XCHECKED);
    }
    // The rounds folded every factor where it lies, so reading them back is
    // what spares the caller a pass over the trace to compute what the device
    // already has — the values at the point, when the cube ran out here.
    let Ok(values) = session_values(&session) else {
        return Some(Err(crate::Error::DeviceFailed {
            stage: "factor values",
        }));
    };
    let factors: Result<Vec<crate::mle::Mle<E>>, crate::Error> = values
        .iter()
        .map(|factor| {
            crate::mle::Mle::new(factor.chunks_exact(3).map(ext3_from_raw::<E>).collect())
        })
        .collect();
    let Ok(factors) = factors else {
        return Some(Err(crate::Error::DeviceFailed {
            stage: "factor values",
        }));
    };
    SUMCHECK_CALLS.fetch_add(1, Ordering::Relaxed);
    Some(Ok((rounds, challenges, factors)))
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn prove_sumcheck_resident<E>(
    _resident: &DeviceFactors,
    _extra: &[WeightView<'_, E>],
    _program: &crate::program::Program<E>,
    _degree: usize,
    _challenge: impl FnMut(
        &[math::field::element::FieldElement<E>],
    ) -> math::field::element::FieldElement<E>,
) -> Option<Result<ResidentRounds<E>, crate::Error>>
where
    E: math::field::traits::IsField + 'static,
{
    None
}

/// The claim reduce's sumcheck with its tables built on the card
/// (`LAMBDA_VM_ARGUE_DEVICE_TABLES`): `eq(alpha)` once, each offset's shift
/// table a rotated copy of it, each offset's batched column out of the
/// epoch's resident columns (`math_cuda::sumcheck::reduce_session`).
///
/// The gates are [`prove_sumcheck`]'s over the same `2 · offsets` tables, so the
/// card takes exactly the reductions it took with the tables uploaded, when
/// the table's columns are a resident run. The rounds are [`prove_sumcheck`]'s
/// too, down to the same crossover, and the tables come back folded to it for
/// the caller to finish.
///
/// `None` is a decline before the transcript moved: the caller builds the
/// tables on the host, as before.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn prove_reduce_resident<F, E>(
    columns: &[crate::mle::Mle<F>],
    sources: &[crate::claim_reduce::FactorSource],
    weights: &[math::field::element::FieldElement<E>],
    offsets: &[usize],
    alpha: &[math::field::element::FieldElement<E>],
    resident: Option<(&ResidentColumns, usize)>,
    program: &crate::program::Program<E>,
    challenge: impl FnMut(
        &[math::field::element::FieldElement<E>],
    ) -> math::field::element::FieldElement<E>,
) -> Option<Result<SumcheckRounds<E>, crate::Error>>
where
    F: math::field::traits::IsField + math::field::traits::IsSubFieldOf<E> + 'static,
    E: math::field::traits::IsField + 'static,
{
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;
    use math::field::goldilocks::GoldilocksField as Gl;
    use std::any::TypeId;

    if TypeId::of::<F>() != TypeId::of::<Gl>() || TypeId::of::<E>() != TypeId::of::<Ext3>() {
        return None;
    }
    let (store, first) = resident?;
    if !store.0.is_run(first, columns.len()) {
        return None;
    }
    let num_vars = alpha.len();
    if num_vars == 0 || num_vars >= usize::BITS as usize {
        return None;
    }
    let len = 1usize << num_vars;
    if columns.iter().any(|column| column.len() != len) {
        return None;
    }
    // The reduce's rule is a sum of kernel-times-column pairs: degree two.
    let degree = 2;
    let width = 2 * offsets.len();
    if offsets.is_empty() || !worth_the_device(width, len) {
        return None;
    }
    if program
        .max_slot()
        .is_some_and(|slot| slot as usize >= width)
    {
        return None;
    }
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *DISABLED.get_or_init(|| std::env::var_os("LAMBDA_VM_NO_GPU_SUMCHECK").is_some()) {
        return None;
    }
    let lowered = lower(program)?;
    let raw_alpha = raw_point(alpha)?;
    // Per offset, every source reading at it: its column in the table's run
    // and its batching weight — `claim_reduce::batched_column`'s terms.
    let mut groups = Vec::with_capacity(offsets.len());
    for &offset in offsets {
        let mut members = Vec::new();
        let mut member_weights = Vec::new();
        for (source, weight) in sources.iter().zip(weights) {
            if source.offset == offset {
                members.push(source.column as u64);
                member_weights.extend_from_slice(&ext3_raw(weight)?);
            }
        }
        groups.push((offset, members, member_weights));
    }
    let mut session = math_cuda::sumcheck::reduce_session(
        &store.0,
        first,
        columns.len(),
        &raw_alpha,
        &groups,
        &lowered.nodes,
        &lowered.consts,
        lowered.num_slots,
        lowered.root_slot,
    )
    .ok()?;
    note_tables_on_card(width);
    match table_fault() {
        Some(TableFault::Eq) => session.set_cell(0, 0, &FAULT_CELL).ok()?,
        Some(TableFault::Batched) => session.set_cell(1, 0, &FAULT_CELL).ok()?,
        None => {}
    }
    // Nothing is absorbed until the first round, so a check that cannot read
    // the card still declines; one that reads a wrong table fails the proof.
    if argue_xcheck() {
        for (i, &offset) in offsets.iter().enumerate() {
            let shift = crate::eq::shift_evals(alpha, offset);
            let batched = match crate::claim_reduce::batched_column(
                columns, sources, weights, offset, num_vars,
            ) {
                Ok(batched) => batched,
                Err(error) => return Some(Err(error)),
            };
            for (slot, host, what) in [
                (2 * i, &shift[..], "shift table"),
                (2 * i + 1, batched.evals(), "batched column"),
            ] {
                let card = session.factor(slot).ok()?;
                if let Some(cell) = first_difference(&card, host) {
                    eprintln!(
                        "[argue] XCHECK: the card's {what} for offset {offset} (2^{num_vars} cells) \
                         is not the host's at cell {cell}"
                    );
                    return Some(Err(crate::Error::DeviceFailed {
                        stage: "reduce tables",
                    }));
                }
                note_tables_xchecked(1);
            }
        }
    }

    // The same crossover `prove_sumcheck` stops at for the reduce's
    // polynomial, whose rule the host walks through the program interpreter.
    let there = num_vars.saturating_sub(crate::HOST_CUBE_COMPILED.trailing_zeros() as usize);
    let outcome = run_rounds(&mut session, degree, there, challenge, |_| None);
    let (rounds, challenges) = match outcome {
        Ok(rounds) => rounds,
        Err(error) => return Some(Err(error)),
    };
    let Ok(values) = session_values(&session) else {
        return Some(Err(crate::Error::DeviceFailed { stage: "download" }));
    };
    let folded: Result<Vec<crate::mle::Mle<E>>, crate::Error> = values
        .iter()
        .map(|table| crate::mle::Mle::new(table.chunks_exact(3).map(ext3_from_raw::<E>).collect()))
        .collect();
    let Ok(folded) = folded else {
        return Some(Err(crate::Error::DeviceFailed {
            stage: "folded tables",
        }));
    };
    SUMCHECK_CALLS.fetch_add(1, Ordering::Relaxed);
    Some(Ok((rounds, challenges, folded)))
}

#[cfg(not(feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn prove_reduce_resident<F, E>(
    _columns: &[crate::mle::Mle<F>],
    _sources: &[crate::claim_reduce::FactorSource],
    _weights: &[math::field::element::FieldElement<E>],
    _offsets: &[usize],
    _alpha: &[math::field::element::FieldElement<E>],
    _resident: Option<(&ResidentColumns, usize)>,
    _program: &crate::program::Program<E>,
    _challenge: impl FnMut(
        &[math::field::element::FieldElement<E>],
    ) -> math::field::element::FieldElement<E>,
) -> Option<Result<SumcheckRounds<E>, crate::Error>>
where
    F: math::field::traits::IsField + math::field::traits::IsSubFieldOf<E> + 'static,
    E: math::field::traits::IsField + 'static,
{
    None
}

/// Codeword size below which the host fold wins: one launch per level plus the
/// round trip, against a pass a few cores finish in microseconds.
#[cfg(feature = "cuda")]
const FOLD_THRESHOLD: usize = 1 << 16;

/// Folds a codeword `alphas.len()` times on device.
///
/// `generator` is the fold domain's, and the domain squares with every level —
/// the kernel takes each level's inverse generator, which is this one's inverse
/// squared level by level.
#[cfg(feature = "cuda")]
pub(crate) fn fold_codeword_k<F, C, N>(
    codeword: &[math::field::element::FieldElement<C>],
    generator: &math::field::element::FieldElement<F>,
    alphas: &[math::field::element::FieldElement<N>],
) -> Option<Vec<math::field::element::FieldElement<N>>>
where
    F: math::field::traits::IsField + 'static,
    C: math::field::traits::IsField + 'static,
    N: math::field::traits::IsField + 'static,
{
    use math::field::element::FieldElement;
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;
    use math::field::goldilocks::GoldilocksField as Gl;
    use std::any::TypeId;

    if TypeId::of::<F>() != TypeId::of::<Gl>() || TypeId::of::<N>() != TypeId::of::<Ext3>() {
        return None;
    }
    let base = TypeId::of::<C>() == TypeId::of::<Gl>();
    if !base && TypeId::of::<C>() != TypeId::of::<Ext3>() {
        return None;
    }
    if alphas.is_empty() || codeword.len() < FOLD_THRESHOLD {
        return None;
    }
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *DISABLED.get_or_init(|| std::env::var_os("LAMBDA_VM_NO_GPU_WHIR_FOLD").is_some()) {
        return None;
    }

    // SAFETY: `F == GoldilocksField`, a transparent wrapper over `u64`.
    let generator = unsafe { *(generator as *const _ as *const u64) };
    let generator = FieldElement::<Gl>::from_raw(generator);
    let two = FieldElement::<Gl>::from(2u64);
    let two_inv = *two.inv().ok()?.value();
    let mut g_inv = generator.inv().ok()?;
    let mut g_invs = Vec::with_capacity(alphas.len());
    for _ in 0..alphas.len() {
        g_invs.push(*g_inv.value());
        g_inv = g_inv.square();
    }

    let mut raw_alphas = Vec::with_capacity(alphas.len() * 3);
    for alpha in alphas {
        raw_alphas.extend_from_slice(&ext3_raw(alpha)?);
    }

    // SAFETY: `C` is one of the two fields checked above, and both wrap their
    // limbs transparently — one `u64` per base element, three per ext3.
    let limbs = if base { 1 } else { 3 };
    let raw = unsafe {
        core::slice::from_raw_parts(codeword.as_ptr() as *const u64, codeword.len() * limbs)
    };
    let folded = if base {
        math_cuda::whir::fold_codeword_base(raw, two_inv, &g_invs, &raw_alphas).ok()?
    } else {
        math_cuda::whir::fold_codeword_ext3(raw, two_inv, &g_invs, &raw_alphas).ok()?
    };

    // SAFETY: `N == Ext3`, three limbs per element and no drop glue, so the
    // allocation changes type in place instead of being copied.
    let mut folded = core::mem::ManuallyDrop::new(folded);
    Some(unsafe {
        Vec::from_raw_parts(
            folded.as_mut_ptr() as *mut FieldElement<N>,
            folded.len() / 3,
            folded.capacity() / 3,
        )
    })
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn fold_codeword_k<F, C, N>(
    _codeword: &[math::field::element::FieldElement<C>],
    _generator: &math::field::element::FieldElement<F>,
    _alphas: &[math::field::element::FieldElement<N>],
) -> Option<Vec<math::field::element::FieldElement<N>>>
where
    F: math::field::traits::IsField + 'static,
    C: math::field::traits::IsField + 'static,
    N: math::field::traits::IsField + 'static,
{
    None
}

/// Merkle-commits an ext3 codeword's fold blocks on device, returning the tree
/// in the host node layout. A device error is logged and counted in
/// [`tree_commit_errors`] before the `None` that sends the caller to the host.
#[cfg(feature = "cuda")]
pub(crate) fn commit_tree_ext3<F>(
    codeword: &[math::field::element::FieldElement<F>],
    log_folding: usize,
    hash: crate::whir_hash::DeviceHashKey,
) -> Option<Vec<[u8; 32]>>
where
    F: math::field::traits::IsField + 'static,
{
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;

    if std::any::TypeId::of::<F>() != std::any::TypeId::of::<Ext3>() {
        return None;
    }
    if codeword.len() < COMMIT_THRESHOLD || codeword.len() >> log_folding < 2 {
        return None;
    }
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *DISABLED.get_or_init(|| std::env::var_os("LAMBDA_VM_NO_GPU_WHIR_COMMIT").is_some()) {
        return None;
    }
    // No device is a decline, not an error (see `commit_parts`).
    math_cuda::device::backend().ok()?;
    // SAFETY: `F == Ext3`, three transparent `u64` limbs per element.
    let raw =
        unsafe { core::slice::from_raw_parts(codeword.as_ptr() as *const u64, codeword.len() * 3) };
    let nodes = match math_cuda::whir::commit_codeword_ext3(raw, log_folding, hash.into_math_cuda())
    {
        Ok(nodes) => nodes,
        Err(err) => {
            note_tree_commit_error(codeword.len().trailing_zeros(), log_folding, &err);
            return None;
        }
    };
    let nodes = nodes_in_place(nodes)?;
    COMMIT_CALLS.fetch_add(1, Ordering::Relaxed);
    Some(nodes)
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn commit_tree_ext3<F>(
    _codeword: &[math::field::element::FieldElement<F>],
    _log_folding: usize,
    _hash: crate::whir_hash::DeviceHashKey,
) -> Option<Vec<[u8; 32]>>
where
    F: math::field::traits::IsField + 'static,
{
    None
}

/// Table size below which the host evaluation wins: one launch per variable
/// against a couple of passes a few cores finish in microseconds.
#[cfg(feature = "cuda")]
const EVALUATE_THRESHOLD: usize = 1 << 16;

/// Whether the claim reduce evaluates a resident table's columns on the card
/// once the TABLE holds [`EVALUATE_THRESHOLD`] cells, rather than only once
/// each column is that tall. On by default; `LAMBDA_VM_ARGUE_DEVICE_COLUMNS=0`
/// is the opt-out, and it is the old path exactly.
///
/// The default was turned on by its A/B on the block (FAST, 2026-09-28): the
/// whole run 1.05 s faster, the columns the host walked 24,195 → 3,478, every
/// arm proved and verified on the record's identities.
///
/// # Why the height is the wrong measure for a resident table
///
/// The threshold was set for columns that had to be uploaded first, where a
/// short column's upload costs more than the host loop it replaces. Columns
/// the epoch already put on the card cost no upload, and the launches are one
/// per variable for the whole table, so what the host would spend is the
/// table's CELLS: the widest precompile — 1,480 columns of 2^15 rows — is a
/// 48-million-cell walk the host did one column after another, 263–303 ms of a
/// single thread per table, while the card reads it in well under a
/// millisecond (`thoughts/zf/gap2/fix2/D-GFS-NOTE.md` §1.2, G2).
///
/// The values are the same either way — each is the column's multilinear
/// extension at the point, in exact arithmetic — so the proof is too.
pub fn argue_device_columns() -> bool {
    match ARGUE_DEVICE_COLUMNS_FORCED.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => {
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ON.get_or_init(|| env_not_off("LAMBDA_VM_ARGUE_DEVICE_COLUMNS"))
        }
    }
}

static ARGUE_DEVICE_COLUMNS_FORCED: std::sync::atomic::AtomicU8 =
    std::sync::atomic::AtomicU8::new(0);

/// Overrides `LAMBDA_VM_ARGUE_DEVICE_COLUMNS` for the whole process — for a
/// test that proves both ways in one binary. `None` restores the environment's
/// setting. Not for production callers.
#[doc(hidden)]
pub fn force_argue_device_columns(on: Option<bool>) {
    ARGUE_DEVICE_COLUMNS_FORCED.store(
        match on {
            Some(true) => 1,
            Some(false) => 2,
            None => 0,
        },
        Ordering::Relaxed,
    );
}

/// Whether every column value the card computes for the claim reduce is
/// recomputed on the host and compared — `LAMBDA_VM_ARGUE_XCHECK=1`. A
/// diagnostic for an UNTIMED gate run: it puts the host walk back.
///
/// ⛔ It is the block's identity gate, because proof bytes are not one. Two
/// proves of the same block never share their bytes: six table builders lay
/// their rows out in `HashMap` order (`prover/src/tables/eq.rs` and five
/// others), so every root and every challenge after it differs between two
/// processes whatever this file does. What can be compared is each value the
/// card produced against the host's, in the same process, at the moment it is
/// produced — which is this.
pub fn argue_xcheck() -> bool {
    match ARGUE_XCHECK_FORCED.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => {
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ON.get_or_init(|| env_on("LAMBDA_VM_ARGUE_XCHECK"))
        }
    }
}

static ARGUE_XCHECK_FORCED: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Overrides `LAMBDA_VM_ARGUE_XCHECK` for the whole process. `None` restores
/// the environment's setting. Not for production callers.
#[doc(hidden)]
pub fn force_argue_xcheck(on: Option<bool>) {
    ARGUE_XCHECK_FORCED.store(
        match on {
            Some(true) => 1,
            Some(false) => 2,
            None => 0,
        },
        Ordering::Relaxed,
    );
}

/// `name` set to anything but empty or `0`.
fn env_on(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| !v.is_empty() && v != "0")
}

/// `name` unset or set to anything but `0`: a knob that is on by default.
fn env_not_off(name: &str) -> bool {
    not_off(std::env::var(name).ok().as_deref())
}

/// A default-on knob's reading of its variable: only `0` turns it off.
fn not_off(value: Option<&str>) -> bool {
    value != Some("0")
}

/// Whether the argument's challenge tables are built on the card: the
/// zerocheck's weights `eq(r)` and `eq(row)` from their points, and the claim
/// reduce's shift tables and batched columns from `alpha` and the resident
/// columns. On by default; `LAMBDA_VM_ARGUE_DEVICE_TABLES=0` is the opt-out, and
/// it is the old path exactly.
///
/// The default was turned on by its A/B on the block (FAST, 2026-09-29): the
/// whole run 5.20 s faster, the argue 19.7 → 14.7 s, 1,364 tables built on the
/// card a run, every arm proved and verified on the record's identities, and
/// the cross-check arm — every card table compared with the host's — clean.
///
/// # Why
///
/// Off, both are host tables. They are built on the pool, and the ARGUE thread
/// waits in the join — 3.56 s of it on the head's trace, where the producer's
/// prep holds the pool at the same time — and then they cross as pageable
/// uploads, 20 GB a block (`thoughts/zf/gap2/fix2/I-GFS.md` §6). On the card
/// they are the same values, made from a few kilobytes: the points, the columns
/// each offset reads, and their weights.
pub fn argue_device_tables() -> bool {
    match ARGUE_DEVICE_TABLES_FORCED.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => {
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ON.get_or_init(|| env_not_off("LAMBDA_VM_ARGUE_DEVICE_TABLES"))
        }
    }
}

static ARGUE_DEVICE_TABLES_FORCED: std::sync::atomic::AtomicU8 =
    std::sync::atomic::AtomicU8::new(0);

/// Overrides `LAMBDA_VM_ARGUE_DEVICE_TABLES` for the whole process — for a test
/// that proves both ways in one binary. `None` restores the environment's
/// setting. Not for production callers.
#[doc(hidden)]
pub fn force_argue_device_tables(on: Option<bool>) {
    ARGUE_DEVICE_TABLES_FORCED.store(
        match on {
            Some(true) => 1,
            Some(false) => 2,
            None => 0,
        },
        Ordering::Relaxed,
    );
}

/// Whether a device sumcheck's end is read back in one gathered copy —
/// `LAMBDA_VM_ARGUE_LEAN_READS=1` (any non-empty value other than `0`) —
/// rather than one synchronous copy per factor. The same bytes either way; off
/// by default until its A/B, and off is today's path.
///
/// Every session the argument runs ends by reading its factors back: the
/// zerocheck's hundreds or thousands (the widest precompile has 1,482), each
/// GKR layer's five, each reduce's pairs. One copy each is a round trip each,
/// ~6 µs: about 0.3 s a block on the head's trace, 49 thousand copies of a few
/// hundred bytes (`thoughts/zf/gap2/fix2/I-GFS.md` §8).
pub fn argue_lean_reads() -> bool {
    match ARGUE_LEAN_READS_FORCED.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => {
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ON.get_or_init(|| env_on("LAMBDA_VM_ARGUE_LEAN_READS"))
        }
    }
}

static ARGUE_LEAN_READS_FORCED: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Overrides `LAMBDA_VM_ARGUE_LEAN_READS` for the whole process — for a test
/// that proves both ways in one binary. `None` restores the environment's
/// setting. Not for production callers.
#[doc(hidden)]
pub fn force_argue_lean_reads(on: Option<bool>) {
    ARGUE_LEAN_READS_FORCED.store(
        match on {
            Some(true) => 1,
            Some(false) => 2,
            None => 0,
        },
        Ordering::Relaxed,
    );
}

/// ⛔ A FAULT: while armed, a gathered read hands back its first factor's first
/// word plus one — what a gather that read one wrong word would do — so the
/// checks on a session's end can be shown to see one. Never armed outside a
/// test. Not for production callers.
static READ_FAULT: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

#[doc(hidden)]
pub fn force_read_fault(on: bool) {
    READ_FAULT.store(on, Ordering::Relaxed);
}

/// Whether a device GKR layer's host tail runs lean — `LAMBDA_VM_ARGUE_LEAN_TAIL=1`
/// (any non-empty value other than `0`): the layer's `eq` weight pulled out of
/// the round polynomial and its factors extended by addition
/// ([`gkr`](crate::gkr)'s `lean_tail`). The same round values either way; off by
/// default until its A/B, and off is today's path.
///
/// Every device layer hands its last nine rounds to the host, over a cube of
/// 512: on the head's split that tail's arithmetic is 52 % of the 264 µs the
/// host spends between two device layers (`thoughts/zf/gap2/fix2/I-GFS.md` §9).
pub fn argue_lean_tail() -> bool {
    match ARGUE_LEAN_TAIL_FORCED.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => {
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ON.get_or_init(|| env_on("LAMBDA_VM_ARGUE_LEAN_TAIL"))
        }
    }
}

static ARGUE_LEAN_TAIL_FORCED: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Whether a table's GKR input layer is written straight from its resident
/// base columns (D-BATCH M1-2): one launch for every interaction, base × ext3,
/// where the old path runs two programs an interaction over the lifted factors.
/// The same cells, so the same tree and proof. On by default;
/// `LAMBDA_VM_ARGUE_GKR_INPUT=0` is the old path exactly. Read once, with a
/// banner.
///
/// Turned on by its A/B on the block (FAST job 333, 4 + 4 arms): base −0.75 s,
/// the argue −0.97 s, the whole run −0.73 s, the tree region 1.98 → 1.11 s,
/// every arm on the record's program ids.
pub fn argue_gkr_input() -> bool {
    match ARGUE_GKR_INPUT_FORCED.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => {
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ON.get_or_init(|| {
                let on = env_not_off("LAMBDA_VM_ARGUE_GKR_INPUT");
                eprintln!(
                    "★ ARGUE GKR INPUT: {}",
                    if on {
                        "from the base columns (the default; LAMBDA_VM_ARGUE_GKR_INPUT=0 is the lifted \
                         factors)"
                    } else {
                        "from the lifted factors (LAMBDA_VM_ARGUE_GKR_INPUT=0)"
                    }
                );
                on
            })
        }
    }
}

static ARGUE_GKR_INPUT_FORCED: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Whether a table's factors stay base columns on the card, with no lift
/// (D-BATCH M1-2, its second half). The input layer is written from the columns
/// ([`argue_gkr_input`], which this needs) and the fused zerocheck's first pass
/// reads them where they lie, so the `W·rows·24 B` lift is made only when the
/// old rounds need it. The same rounds and proof. On by default;
/// `LAMBDA_VM_ARGUE_NO_LIFT=0` lifts every table's factors as before.
///
/// Turned on by its A/B on the block (FAST job 335, 4 + 4 arms over the input
/// from the columns): base −0.90 s, the whole run −1.55 s, the argue's
/// reserved peak −1.5 GiB, kept WHIR trees evicted 13 → 2 a run and the
/// openings' tree rebuilds 0.74 → 0.04 s, every arm on the record's program
/// ids.
pub fn argue_no_lift() -> bool {
    match ARGUE_NO_LIFT_FORCED.load(Ordering::Relaxed) {
        1 => argue_gkr_input(),
        2 => false,
        _ => {
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ON.get_or_init(|| {
                let wanted = env_not_off("LAMBDA_VM_ARGUE_NO_LIFT");
                let input = argue_gkr_input();
                eprintln!(
                    "★ ARGUE NO LIFT: {}",
                    match (wanted, input) {
                        (true, true) => {
                            "on (the default, with the input from the columns; \
                             LAMBDA_VM_ARGUE_NO_LIFT=0 lifts the factors)"
                        }
                        (false, _) => "off (LAMBDA_VM_ARGUE_NO_LIFT=0)",
                        (true, false) => "off (it needs the input from the columns)",
                    }
                );
                wanted && input
            })
        }
    }
}

static ARGUE_NO_LIFT_FORCED: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Overrides `LAMBDA_VM_ARGUE_NO_LIFT` for the whole process; `None` restores it.
#[doc(hidden)]
pub fn force_argue_no_lift(on: Option<bool>) {
    ARGUE_NO_LIFT_FORCED.store(
        match on {
            Some(true) => 1,
            Some(false) => 2,
            None => 0,
        },
        Ordering::Relaxed,
    );
}

/// Overrides `LAMBDA_VM_ARGUE_GKR_INPUT` for the whole process — for a test
/// that proves both ways in one binary. `None` restores the environment's.
#[doc(hidden)]
pub fn force_argue_gkr_input(on: Option<bool>) {
    ARGUE_GKR_INPUT_FORCED.store(
        match on {
            Some(true) => 1,
            Some(false) => 2,
            None => 0,
        },
        Ordering::Relaxed,
    );
}

/// Overrides `LAMBDA_VM_ARGUE_LEAN_TAIL` for the whole process — for a test that
/// proves both ways in one binary. `None` restores the environment's setting.
/// Not for production callers.
#[doc(hidden)]
pub fn force_argue_lean_tail(on: Option<bool>) {
    ARGUE_LEAN_TAIL_FORCED.store(
        match on {
            Some(true) => 1,
            Some(false) => 2,
            None => 0,
        },
        Ordering::Relaxed,
    );
}

/// Whether a big batch's device rounds walk its program on demand
/// ([`Program::on_demand`](crate::program::Program::on_demand)), with the slot
/// file sized for every interpolation node from the first round. The same
/// round values either way. On by default; `LAMBDA_VM_ARGUE_LEAN_PROGRAM=0` is
/// the opt-out, and it is the old path exactly.
///
/// The default was turned on by its A/B on the block (FAST, 2026-09-29): the
/// whole run 1.35 s faster, the big batches' early device rounds 1,527 → 442 ms
/// and the argue 1.43 s faster, every arm proved and verified on the record's
/// identities, and the cross-check arm — every big session's rounds compared,
/// round by round, with today's program over the same factors — clean.
///
/// A batch is big when the values its lowered program holds a thread, more
/// than [`LEAN_ABOVE_SLOTS`], leave a round fewer than 64 k threads: the head's
/// widest ran its first rounds on 8,096 threads, at 94 ms a launch. What fills
/// that slot file is the order the batch was written in — every read held from
/// its first use to its last, every root and every interaction's side held
/// until the sum at the end. On demand the widest holds 207 values instead of
/// 2,763, and the four big batches 14 to 207 (`thoughts/zf/gap2/fix2/I-GFS.md`
/// §16).
pub fn argue_lean_program() -> bool {
    match ARGUE_LEAN_PROGRAM_FORCED.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => {
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ON.get_or_init(|| env_not_off("LAMBDA_VM_ARGUE_LEAN_PROGRAM"))
        }
    }
}

static ARGUE_LEAN_PROGRAM_FORCED: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Overrides `LAMBDA_VM_ARGUE_LEAN_PROGRAM` for the whole process — for a test
/// that proves both ways in one binary. `None` restores the environment's
/// setting. Not for production callers.
#[doc(hidden)]
pub fn force_argue_lean_program(on: Option<bool>) {
    ARGUE_LEAN_PROGRAM_FORCED.store(
        match on {
            Some(true) => 1,
            Some(false) => 2,
            None => 0,
        },
        Ordering::Relaxed,
    );
}

/// A batch whose lowered program holds more values a thread than this is
/// big: at 342 or more, the round kernel's slot budget leaves fewer than 64 k
/// threads (`math_cuda::sumcheck::thread_ceiling`; pinned by a test). The VM's
/// four big batches hold 859 to 2,763; the next widest, 117.
pub const LEAN_ABOVE_SLOTS: usize = 341;

/// A test's own gate in place of [`LEAN_ABOVE_SLOTS`] (`u64::MAX` = none),
/// so a fixture whose batches are small can still take the lean path.
static LEAN_GATE_FORCED: AtomicU64 = AtomicU64::new(u64::MAX);

#[doc(hidden)]
pub fn force_lean_program_gate(above_slots: Option<usize>) {
    LEAN_GATE_FORCED.store(
        above_slots.map_or(u64::MAX, |slots| slots as u64),
        Ordering::Relaxed,
    );
}

/// Whether a program holding `num_slots` values a thread is a big batch.
pub fn is_big_batch(num_slots: usize) -> bool {
    match LEAN_GATE_FORCED.load(Ordering::Relaxed) {
        u64::MAX => num_slots > LEAN_ABOVE_SLOTS,
        forced => num_slots as u64 > forced,
    }
}

/// ⛔ A FAULT: while armed, every constant of a program run on demand is off by
/// one in its first limb — what a schedule that read wrong values would do — so
/// the checks after it can be shown to see it. Never armed outside a test.
static LEAN_PROGRAM_FAULT: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

#[doc(hidden)]
pub fn force_lean_program_fault(on: bool) {
    LEAN_PROGRAM_FAULT.store(on, Ordering::Relaxed);
}

/// What a zerocheck's device rounds walk.
#[cfg(feature = "cuda")]
struct ZerocheckLowering {
    /// The program the rounds run, lowered.
    run: Lowered,
    /// Today's lowering, when `run` is the same program on demand in its place.
    held: Option<Lowered>,
    /// Whether the batch is big ([`is_big_batch`]).
    big: bool,
}

/// `program` lowered — or, under `LAMBDA_VM_ARGUE_LEAN_PROGRAM` for a big batch,
/// the same program on demand, when that holds fewer values.
#[cfg(feature = "cuda")]
fn lower_for_zerocheck<E>(program: &crate::program::Program<E>) -> Option<ZerocheckLowering>
where
    E: math::field::traits::IsField + 'static,
{
    let today = lower(program)?;
    let big = is_big_batch(today.num_slots);
    let as_today = |today| ZerocheckLowering {
        run: today,
        held: None,
        big,
    };
    if !big || !argue_lean_program() {
        return Some(as_today(today));
    }
    let Some(mut demand) = lower(&program.on_demand()) else {
        return Some(as_today(today));
    };
    if demand.num_slots >= today.num_slots {
        return Some(as_today(today));
    }
    if LEAN_PROGRAM_FAULT.load(Ordering::Relaxed) {
        for constant in demand.consts.chunks_exact_mut(3) {
            constant[0] = constant[0].wrapping_add(1);
        }
    }
    Some(ZerocheckLowering {
        run: demand,
        held: Some(today),
        big,
    })
}

/// A device sumcheck's factors as its last fold left them: one gathered copy
/// under `LAMBDA_VM_ARGUE_LEAN_READS`, a copy per factor otherwise — the same
/// bytes, counted both ways so an arm's log says which it took.
#[cfg(feature = "cuda")]
fn session_values(
    session: &math_cuda::sumcheck::SumcheckSession,
) -> math_cuda::Result<Vec<Vec<u64>>> {
    if !argue_lean_reads() {
        let values = session.values()?;
        READS_PER_FACTOR.fetch_add(values.len() as u64, Ordering::Relaxed);
        crate::whir_split::bump_by(&crate::whir_split::READS_PER_FACTOR, values.len() as u64);
        return Ok(values);
    }
    let mut values = session.values_gathered()?;
    READS_GATHERED.fetch_add(1, Ordering::Relaxed);
    crate::whir_split::bump(&crate::whir_split::READS_GATHERED);
    if READ_FAULT.load(Ordering::Relaxed)
        && let Some(word) = values.first_mut().and_then(|factor| factor.first_mut())
    {
        *word = word.wrapping_add(1);
    }
    Ok(values)
}

/// Which table the card built a fault corrupts ([`force_table_fault`]).
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TableFault {
    /// The first table built from `eq`: the zerocheck's `eq(r)`, or the claim
    /// reduce's first shift table.
    Eq,
    /// The claim reduce's first batched column.
    Batched,
}

/// ⛔ A FAULT: while armed, the first cell of the chosen table the card builds
/// is overwritten before the rounds read it — what a kernel that wrote one
/// wrong cell would leave. So the checks on the tables can be shown to fail.
/// Never armed outside a test.
static TABLE_FAULT: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// What [`TABLE_FAULT`] writes: a value no table here holds in its first cell
/// but by a coincidence of one in 2^64.
#[cfg(feature = "cuda")]
const FAULT_CELL: [u64; 3] = [12345, 0, 0];

#[doc(hidden)]
pub fn force_table_fault(fault: Option<TableFault>) {
    TABLE_FAULT.store(
        match fault {
            None => 0,
            Some(TableFault::Eq) => 1,
            Some(TableFault::Batched) => 2,
        },
        Ordering::Relaxed,
    );
}

#[cfg(feature = "cuda")]
fn table_fault() -> Option<TableFault> {
    match TABLE_FAULT.load(Ordering::Relaxed) {
        1 => Some(TableFault::Eq),
        2 => Some(TableFault::Batched),
        _ => None,
    }
}

/// Called where the card built `tables` of the argument's challenge tables.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn note_tables_on_card(tables: usize) {
    ARGUE_TABLES_ON_CARD.fetch_add(tables as u64, Ordering::Relaxed);
    crate::whir_split::bump_by(&crate::whir_split::TABLES_ON_CARD, tables as u64);
}

/// Called where `LAMBDA_VM_ARGUE_XCHECK` found `tables` the card built equal to
/// the host's, cell for cell.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn note_tables_xchecked(tables: usize) {
    ARGUE_TABLE_XCHECKS.fetch_add(tables as u64, Ordering::Relaxed);
    crate::whir_split::bump_by(&crate::whir_split::TABLES_XCHECKED, tables as u64);
}

/// A batch weight as the device is handed it: a table the host built, or the
/// point of an `eq` table the card builds (`LAMBDA_VM_ARGUE_DEVICE_TABLES`).
#[derive(Clone, Copy, Debug)]
pub(crate) enum WeightView<'a, E: math::field::traits::IsField> {
    Table(&'a crate::mle::Mle<E>),
    Eq(&'a [math::field::element::FieldElement<E>]),
}

impl<E: math::field::traits::IsField + 'static> WeightView<'_, E> {
    /// Cells the weight spans.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    fn cells(&self) -> usize {
        match self {
            Self::Table(table) => table.len(),
            Self::Eq(point) => 1usize << point.len(),
        }
    }
}

/// The first cell at which a table the card built is not the host's.
#[cfg(feature = "cuda")]
fn first_difference<E>(
    card: &[u64],
    host: &[math::field::element::FieldElement<E>],
) -> Option<usize>
where
    E: math::field::traits::IsField + 'static,
{
    if card.len() != host.len() * 3 {
        return Some(card.len().min(host.len() * 3) / 3);
    }
    card.chunks_exact(3)
        .zip(host)
        .position(|(cell, value)| ext3_from_raw::<E>(cell) != *value)
}

/// ⛔ A FAULT: while armed, the batched device evaluation hands back its first
/// column's value plus one — what a kernel that read one wrong cell would do.
///
/// It exists so the checks that compare the card against the host can be shown
/// to fail: an identity test that passes whatever the card returns certifies
/// nothing. Never armed outside a test. Not for production callers.
static COLUMN_VALUE_FAULT: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

#[doc(hidden)]
pub fn force_column_value_fault(on: bool) {
    COLUMN_VALUE_FAULT.store(on, Ordering::Relaxed);
}

/// Whether [`evaluate_mle`] sends a table of `len` evaluations to the device —
/// so a caller evaluating many one at a time can tell a host loop, which it may
/// spread over the pool, from device calls, which it must not.
#[cfg(feature = "cuda")]
pub(crate) fn evaluates_on_device(len: usize) -> bool {
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    len >= EVALUATE_THRESHOLD
        && !*DISABLED.get_or_init(|| std::env::var_os("LAMBDA_VM_NO_GPU_MLE_EVAL").is_some())
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn evaluates_on_device(_len: usize) -> bool {
    false
}

/// Called where the claim reduce walks `columns` on the host, one at a time.
pub(crate) fn note_host_evaluations(columns: usize) {
    HOST_EVALUATE_CALLS.fetch_add(columns as u64, Ordering::Relaxed);
    crate::whir_split::bump_by(&crate::whir_split::COLUMNS_ON_HOST, columns as u64);
}

/// Called where `LAMBDA_VM_ARGUE_XCHECK` found `columns` device values equal to
/// the host's.
pub(crate) fn note_xchecked(columns: usize) {
    ARGUE_XCHECKS.fetch_add(columns as u64, Ordering::Relaxed);
    crate::whir_split::bump_by(&crate::whir_split::COLUMNS_XCHECKED, columns as u64);
}

/// Every base-field column's value at one point, folded together.
///
/// The columns of a table are evaluated at the same point and one at a time
/// that is an upload and a launch per level each. Together it is one upload
/// and one launch per level for all of them.
#[cfg(feature = "cuda")]
pub(crate) fn evaluate_many_base<F, E>(
    columns: &[crate::mle::Mle<F>],
    point: &[math::field::element::FieldElement<E>],
    resident: Option<(&ResidentColumns, usize)>,
) -> Option<Vec<math::field::element::FieldElement<E>>>
where
    F: math::field::traits::IsField + 'static,
    E: math::field::traits::IsField + 'static,
{
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;
    use math::field::goldilocks::GoldilocksField as Gl;
    use std::any::TypeId;

    if TypeId::of::<F>() != TypeId::of::<Gl>() || TypeId::of::<E>() != TypeId::of::<Ext3>() {
        return None;
    }
    let rows = columns.first()?.len();
    if point.is_empty() || rows != 1 << point.len() {
        return None;
    }
    // Columns already on the card cost no upload, so under the knob a resident
    // table is worth the launches once it has the cells; any other table still
    // has to pay its upload, and is worth it only once each column is tall.
    let resident_run = argue_device_columns()
        && resident.is_some_and(|(store, first)| store.0.is_run(first, columns.len()));
    let size = if resident_run {
        rows.saturating_mul(columns.len())
    } else {
        rows
    };
    if size < EVALUATE_THRESHOLD {
        return None;
    }
    if columns.iter().any(|column| column.len() != rows) {
        return None;
    }
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *DISABLED.get_or_init(|| std::env::var_os("LAMBDA_VM_NO_GPU_MLE_EVAL").is_some()) {
        return None;
    }

    let mut raw_point = Vec::with_capacity(point.len() * 3);
    for coordinate in point {
        raw_point.extend_from_slice(&ext3_raw(coordinate)?);
    }
    // SAFETY: `F == Gl`, a transparent wrapper over one `u64` per element.
    let raw: Vec<&[u64]> = columns
        .iter()
        .map(|column| unsafe {
            core::slice::from_raw_parts(column.evals().as_ptr() as *const u64, column.len())
        })
        .collect();
    let values =
        math_cuda::sumcheck::evaluate_many_base(columns_at(resident, &raw), &raw_point).ok()?;
    EVALUATE_CALLS.fetch_add(values.len() as u64, Ordering::Relaxed);
    crate::whir_split::bump_by(&crate::whir_split::COLUMNS_ON_CARD, values.len() as u64);
    let mut values: Vec<math::field::element::FieldElement<E>> =
        values.iter().map(|v| ext3_from_raw::<E>(v)).collect();
    if COLUMN_VALUE_FAULT.load(Ordering::Relaxed)
        && let Some(first) = values.first_mut()
    {
        *first += math::field::element::FieldElement::<E>::one();
    }
    Some(values)
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn evaluate_many_base<F, E>(
    _columns: &[crate::mle::Mle<F>],
    _point: &[math::field::element::FieldElement<E>],
    _resident: Option<(&ResidentColumns, usize)>,
) -> Option<Vec<math::field::element::FieldElement<E>>>
where
    F: math::field::traits::IsField + 'static,
    E: math::field::traits::IsField + 'static,
{
    None
}

/// A multilinear's value at `point`, bound variable by variable on device.
///
/// `evals` may be base-field or ext3; the point is always ext3, which is what
/// a challenge is.
#[cfg(feature = "cuda")]
pub(crate) fn evaluate_mle<C, E>(
    evals: &[math::field::element::FieldElement<C>],
    point: &[math::field::element::FieldElement<E>],
) -> Option<math::field::element::FieldElement<E>>
where
    C: math::field::traits::IsField + 'static,
    E: math::field::traits::IsField + 'static,
{
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;
    use math::field::goldilocks::GoldilocksField as Gl;
    use std::any::TypeId;

    if TypeId::of::<E>() != TypeId::of::<Ext3>() {
        return None;
    }
    let base = TypeId::of::<C>() == TypeId::of::<Gl>();
    if !base && TypeId::of::<C>() != TypeId::of::<Ext3>() {
        return None;
    }
    if point.is_empty() || evals.len() < EVALUATE_THRESHOLD || evals.len() != 1 << point.len() {
        return None;
    }
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *DISABLED.get_or_init(|| std::env::var_os("LAMBDA_VM_NO_GPU_MLE_EVAL").is_some()) {
        return None;
    }

    let mut raw_point = Vec::with_capacity(point.len() * 3);
    for coordinate in point {
        raw_point.extend_from_slice(&ext3_raw(coordinate)?);
    }
    // SAFETY: `C` is one of the two fields checked above, and both wrap their
    // limbs transparently — one `u64` per base element, three per ext3.
    let limbs = if base { 1 } else { 3 };
    let raw =
        unsafe { core::slice::from_raw_parts(evals.as_ptr() as *const u64, evals.len() * limbs) };
    let value = if base {
        math_cuda::sumcheck::evaluate_mle_base(raw, &raw_point).ok()?
    } else {
        math_cuda::sumcheck::evaluate_mle_ext3(raw, &raw_point).ok()?
    };
    EVALUATE_CALLS.fetch_add(1, Ordering::Relaxed);
    Some(ext3_from_raw::<E>(&value))
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn evaluate_mle<C, E>(
    _evals: &[math::field::element::FieldElement<C>],
    _point: &[math::field::element::FieldElement<E>],
) -> Option<math::field::element::FieldElement<E>>
where
    C: math::field::traits::IsField + 'static,
    E: math::field::traits::IsField + 'static,
{
    None
}

/// Input-layer size below which the host tree wins: the levels are a launch
/// each and the fold is a pass a few cores finish in microseconds.
#[cfg(feature = "cuda")]
const TREE_THRESHOLD: usize = 1 << 14;

/// A LogUp fraction tree the device holds.
///
/// The levels are here; the input layer is here only if it was cheap to carry.
/// A tree built from resident factors gives it back after the first fold and
/// writes it again for its own sumcheck — see
/// [`input_layer_tree`](crate::gpu::input_layer_tree).
#[cfg(feature = "cuda")]
pub struct DeviceTree {
    /// Taken when the input layer's sumcheck comes: every level above it is
    /// spent by then, and letting them go is what makes room for it.
    tree: std::sync::Mutex<Option<math_cuda::gkr::DeviceFractionTree>>,
    /// Writes the input layer again, for a tree that did not keep it.
    #[allow(clippy::type_complexity)]
    rebuild: Option<Box<dyn Fn() -> Option<math_cuda::gkr::InputLayer> + Send + Sync>>,
    num_layers: usize,
    input_num_vars: usize,
    /// The output fraction, read off the top level.
    ///
    /// Populated at build on the eager path (whoever asks may be asking after
    /// the levels are gone). Left empty by the prefetch build so the fold
    /// kernels stay in flight and are not waited on here; [`Self::output`] then
    /// reads it from the retained tree on first access, at the consume site,
    /// which is before any level is spent.
    output: std::sync::OnceLock<([u64; 3], [u64; 3])>,
    /// The room the whole thing promised itself, the handed-back layer
    /// included.
    _room: Option<math_cuda::device::DeviceReservation>,
}

#[cfg(feature = "cuda")]
impl std::fmt::Debug for DeviceTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceTree")
            .field("layers", &self.num_layers)
            .finish()
    }
}

/// A device tree the build declined to make. Never constructed.
#[cfg(not(feature = "cuda"))]
#[derive(Debug)]
pub struct DeviceTree(std::convert::Infallible);

#[cfg(not(feature = "cuda"))]
impl DeviceTree {
    pub(crate) fn num_layers(&self) -> usize {
        match self.0 {}
    }

    pub(crate) fn output<E>(
        &self,
    ) -> Result<
        (
            math::field::element::FieldElement<E>,
            math::field::element::FieldElement<E>,
        ),
        crate::Error,
    >
    where
        E: math::field::traits::IsField + 'static,
    {
        match self.0 {}
    }

    pub(crate) fn prove_layer<E>(
        &self,
        _layer: usize,
        _point: &[math::field::element::FieldElement<E>],
        _program: &crate::program::Program<E>,
        _degree: usize,
        _tail: usize,
        _challenge: impl FnMut(
            &[math::field::element::FieldElement<E>],
        ) -> math::field::element::FieldElement<E>,
    ) -> Option<Result<LayerRounds<E>, crate::Error>>
    where
        E: math::field::traits::IsField + 'static,
    {
        match self.0 {}
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prove_layer_gruen<E>(
        &self,
        _layer: usize,
        _point: &[math::field::element::FieldElement<E>],
        _lambda: &math::field::element::FieldElement<E>,
        _claim: math::field::element::FieldElement<E>,
        _tail: usize,
        _today: Option<&crate::program::Program<E>>,
        _challenge: impl FnMut(
            &[math::field::element::FieldElement<E>],
        ) -> math::field::element::FieldElement<E>,
    ) -> Option<Result<LayerRounds<E>, crate::Error>>
    where
        E: math::field::traits::IsField + 'static,
    {
        match self.0 {}
    }

    pub(crate) fn layer_to_host<E>(
        &self,
        _layer: usize,
    ) -> Option<(crate::mle::Mle<E>, crate::mle::Mle<E>)>
    where
        E: math::field::traits::IsField + 'static,
    {
        match self.0 {}
    }
}

/// What a layer's rounds on device leave: the rounds themselves, the point
/// they drew, and the five factors as the last fold left them.
///
/// The factors are one value each when the device ran the layer out, and a
/// cube when it stopped at the crossover for the host to finish — the caller
/// carries on from them either way.
pub type LayerRounds<E> = (
    Vec<crate::sumcheck::RoundProof<E>>,
    Vec<math::field::element::FieldElement<E>>,
    Vec<crate::mle::Mle<E>>,
);

/// Builds the tree on device from an input layer, folding every level there.
#[cfg(feature = "cuda")]
pub(crate) fn build_tree<E>(p: &crate::mle::Mle<E>, q: &crate::mle::Mle<E>) -> Option<DeviceTree>
where
    E: math::field::traits::IsField + 'static,
{
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;

    if std::any::TypeId::of::<E>() != std::any::TypeId::of::<Ext3>() {
        return None;
    }
    if p.len() < TREE_THRESHOLD || p.len() != q.len() {
        return None;
    }
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *DISABLED.get_or_init(|| std::env::var_os("LAMBDA_VM_NO_GPU_GKR").is_some()) {
        return None;
    }
    // SAFETY: `E == Ext3`, three transparent `u64` limbs per element.
    let raw = |table: &crate::mle::Mle<E>| unsafe {
        core::slice::from_raw_parts(table.evals().as_ptr() as *const u64, table.len() * 3)
    };
    let tree = math_cuda::gkr::DeviceFractionTree::build(raw(p), raw(q)).ok()?;
    let output = tree.output().ok()?;
    let num_layers = tree.num_layers();
    let input_num_vars = tree.layer_num_vars(num_layers - 1);
    TREE_CALLS.fetch_add(1, Ordering::Relaxed);
    // This one keeps its input layer: it was uploaded whole, so there is
    // nothing cheaper to hand back. Eager: the output is read at build.
    let output_cell = std::sync::OnceLock::new();
    let _ = output_cell.set(output);
    Some(DeviceTree {
        tree: std::sync::Mutex::new(Some(tree)),
        rebuild: None,
        num_layers,
        input_num_vars,
        output: output_cell,
        _room: None,
    })
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn build_tree<E>(_p: &crate::mle::Mle<E>, _q: &crate::mle::Mle<E>) -> Option<DeviceTree>
where
    E: math::field::traits::IsField + 'static,
{
    None
}

#[cfg(feature = "cuda")]
impl DeviceTree {
    pub(crate) fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// The output fraction, which is what says whether the bus balances.
    ///
    /// On the eager path the value was read at build and is returned straight.
    /// On the prefetch path it was left for here: the top fraction is read off
    /// the retained tree now — the one device sync the prefetch deferred — and
    /// cached. This runs at the consume site, before GKR spends any level, so
    /// the top is still there to read.
    pub(crate) fn output<E>(
        &self,
    ) -> Result<
        (
            math::field::element::FieldElement<E>,
            math::field::element::FieldElement<E>,
        ),
        crate::Error,
    >
    where
        E: math::field::traits::IsField + 'static,
    {
        let (p, q) = self.materialized_output()?;
        Ok((ext3_from_raw::<E>(&p), ext3_from_raw::<E>(&q)))
    }

    /// The output limbs, reading them off the device on first access for a tree
    /// the prefetch build left unread, then caching. `DeviceFailed` if the
    /// retained tree is gone or the device read fails — the caller falls back
    /// to a host build, and the transcript has not moved yet at this point.
    fn materialized_output(&self) -> Result<([u64; 3], [u64; 3]), crate::Error> {
        if let Some(v) = self.output.get() {
            return Ok(*v);
        }
        let read = {
            let held = self.tree.lock().map_err(|_| crate::Error::DeviceFailed {
                stage: "gkr tree output",
            })?;
            let tree = held.as_ref().ok_or(crate::Error::DeviceFailed {
                stage: "gkr tree output",
            })?;
            tree.output().map_err(|_| crate::Error::DeviceFailed {
                stage: "gkr tree output",
            })?
        };
        let _ = self.output.set(read);
        Ok(read)
    }

    /// A level's halves back here, for the levels near the output that GKR
    /// proves on the host.
    pub(crate) fn layer_to_host<E>(
        &self,
        layer: usize,
    ) -> Option<(crate::mle::Mle<E>, crate::mle::Mle<E>)>
    where
        E: math::field::traits::IsField + 'static,
    {
        let held = self.tree.lock().ok()?;
        let (p, q) = held.as_ref()?.layer_to_host(layer).ok()?;
        let table = |raw: Vec<u64>| {
            crate::mle::Mle::new(raw.chunks_exact(3).map(ext3_from_raw::<E>).collect()).ok()
        };
        Some((table(p)?, table(q)?))
    }

    /// One layer's sumcheck, folded in place: the rounds, the point they drew,
    /// and the five factors as the last fold left them.
    ///
    /// The rounds stop once the cube is down to `tail`, which the caller
    /// finishes here — below a few hundred indices a round is a launch and a
    /// wait around a kernel with almost nothing to sum. `tail` of one runs the
    /// layer out there, and the factors come back as one value each.
    ///
    /// The layer is spent afterwards, which is what makes the halves usable as
    /// factors without copying them: GKR reads each layer once.
    pub(crate) fn prove_layer<E>(
        &self,
        layer: usize,
        point: &[math::field::element::FieldElement<E>],
        program: &crate::program::Program<E>,
        degree: usize,
        tail: usize,
        challenge: impl FnMut(
            &[math::field::element::FieldElement<E>],
        ) -> math::field::element::FieldElement<E>,
    ) -> Option<Result<LayerRounds<E>, crate::Error>>
    where
        E: math::field::traits::IsField + 'static,
    {
        use crate::whir_split::{self, add_tick, tick};

        let t = tick();
        let lowered = lower(program)?;
        let mut raw_point = Vec::with_capacity(point.len() * 3);
        for coordinate in point {
            raw_point.extend_from_slice(&ext3_raw(coordinate)?);
        }
        add_tick(&whir_split::GKR_LOWER, t);
        let input_layer = layer + 1 == self.num_layers;
        let rebuilds = input_layer && self.rebuild.is_some();
        let t = tick();
        let session = if rebuilds {
            // The one the tree gave back. Everything above it has been proved,
            // so the levels go first and the layer is written where they were.
            let num_vars = self.input_num_vars.checked_sub(1)?;
            if num_vars * 3 != raw_point.len() {
                return None;
            }
            drop(self.tree.lock().ok()?.take());
            let rebuilt = (self.rebuild.as_ref()?)()?;
            rebuilt.sumcheck(
                &raw_point,
                &lowered.nodes,
                &lowered.consts,
                lowered.num_slots,
                lowered.root_slot,
            )
        } else {
            let held = self.tree.lock().ok()?;
            let tree = held.as_ref()?;
            if tree.layer_num_vars(layer).checked_sub(1)? * 3 != raw_point.len() {
                return None;
            }
            tree.layer_sumcheck(
                layer,
                &raw_point,
                &lowered.nodes,
                &lowered.consts,
                lowered.num_slots,
                lowered.root_slot,
            )
        };
        if rebuilds {
            add_tick(&whir_split::GKR_REBUILD, t);
            whir_split::bump(&whir_split::GKR_REBUILDS);
        } else {
            add_tick(&whir_split::GKR_SESSION, t);
        }
        let num_vars = if input_layer {
            self.input_num_vars.checked_sub(1)?
        } else {
            self.tree
                .lock()
                .ok()?
                .as_ref()?
                .layer_num_vars(layer)
                .checked_sub(1)?
        };
        let Ok(mut session) = session else {
            return None;
        };
        // What is left for the host: the rounds stop where the cube reaches it.
        let here = tail.max(1).trailing_zeros() as usize;
        let there = num_vars.saturating_sub(here);

        // Past here the transcript moves: the host path is no longer an option.
        let failed = |stage| crate::Error::DeviceFailed { stage };
        let t = tick();
        let outcome = run_rounds(&mut session, degree, there, challenge, |_| None);
        add_tick(&whir_split::GKR_ROUNDS, t);
        let (rounds, challenges) = match outcome {
            Ok(rounds) => rounds,
            Err(error) => return Some(Err(error)),
        };
        let t = tick();
        let Ok(values) = session_values(&session) else {
            return Some(Err(failed("layer values")));
        };
        add_tick(&whir_split::GKR_VALUES, t);
        // Factor 0 is the weight; the four the layer reduces to follow.
        if values.len() != 5 {
            return Some(Err(failed("layer factors")));
        }
        let t = tick();
        let factors: Result<Vec<crate::mle::Mle<E>>, crate::Error> = values
            .iter()
            .map(|factor| {
                crate::mle::Mle::new(factor.chunks_exact(3).map(ext3_from_raw::<E>).collect())
            })
            .collect();
        let Ok(factors) = factors else {
            return Some(Err(failed("layer factors")));
        };
        add_tick(&whir_split::GKR_FACTORS, t);
        whir_split::bump(&whir_split::GKR_LAYERS);
        SUMCHECK_CALLS.fetch_add(1, Ordering::Relaxed);
        Some(Ok((rounds, challenges, factors)))
    }
}

#[cfg(feature = "cuda")]
impl DeviceTree {
    /// [`prove_layer`](Self::prove_layer) with Gruen's rounds
    /// ([`crate::gkr_gruen`]): the same rounds, challenges and factors, from a
    /// pass that sums `H(1)` and `H(2)` and folds the previous challenge on the
    /// way in. `claim` is `p + λ·q` of the level above, the value round 0's
    /// `s(0) + s(1)` has to meet; `tail` the cube the host finishes from.
    ///
    /// With `today` (`LAMBDA_VM_ARGUE_GKR_GRUEN_XCHECK`), today's program walks
    /// the same folded halves every round beside these, and the prove fails
    /// where the two disagree — rounds or factors.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prove_layer_gruen<E>(
        &self,
        layer: usize,
        point: &[math::field::element::FieldElement<E>],
        lambda: &math::field::element::FieldElement<E>,
        claim: math::field::element::FieldElement<E>,
        tail: usize,
        today: Option<&crate::program::Program<E>>,
        mut challenge: impl FnMut(
            &[math::field::element::FieldElement<E>],
        ) -> math::field::element::FieldElement<E>,
    ) -> Option<Result<LayerRounds<E>, crate::Error>>
    where
        E: math::field::traits::IsField + 'static,
    {
        use crate::gkr_gruen::{self as gruen, Chain};
        use crate::whir_split::{self, add_tick, tick};
        use math::field::element::FieldElement;

        let t = tick();
        let mut raw_point = Vec::with_capacity(point.len() * 3);
        for coordinate in point {
            raw_point.extend_from_slice(&ext3_raw(coordinate)?);
        }
        let raw_lambda: [u64; 3] = ext3_raw(lambda)?;
        let lowered = match today {
            Some(program) => Some(lower(program)?),
            None => None,
        };
        let low = tail.max(1).trailing_zeros() as usize;
        add_tick(&whir_split::GKR_LOWER, t);
        let input_layer = layer + 1 == self.num_layers;
        let rebuilds = input_layer && self.rebuild.is_some();
        let t = tick();
        let session = if rebuilds {
            let num_vars = self.input_num_vars.checked_sub(1)?;
            if num_vars * 3 != raw_point.len() {
                return None;
            }
            drop(self.tree.lock().ok()?.take());
            let rebuilt = (self.rebuild.as_ref()?)()?;
            rebuilt.gruen(&raw_point, low)
        } else {
            let held = self.tree.lock().ok()?;
            let tree = held.as_ref()?;
            if tree.layer_num_vars(layer).checked_sub(1)? * 3 != raw_point.len() {
                return None;
            }
            tree.layer_gruen(layer, &raw_point, low)
        };
        let Ok(mut session) = session else {
            return None;
        };
        // Today's session beside it, when the card has room for its `eq` table.
        let shadow = match lowered.as_ref() {
            Some(lowered) => {
                let room = math_cuda::device::reserve(session.shadow_bytes(lowered.num_slots));
                let made = room.as_ref().and_then(|_| {
                    session
                        .shadow(
                            &lowered.nodes,
                            &lowered.consts,
                            lowered.num_slots,
                            lowered.root_slot,
                        )
                        .ok()
                });
                if made.is_none() {
                    gruen::note_xcheck_skipped();
                }
                made.map(|shadow| (shadow, room))
            }
            None => None,
        };
        if rebuilds {
            add_tick(&whir_split::GKR_REBUILD, t);
            whir_split::bump(&whir_split::GKR_REBUILDS);
        } else {
            add_tick(&whir_split::GKR_SESSION, t);
        }
        let rounds_here = session.rounds();
        let lefts = gruen::inverse_lefts(&point[..rounds_here]);
        let chain = Chain::new(claim);
        let mut nodes = Vec::with_capacity(9);
        for node in 1..=3u64 {
            nodes.extend_from_slice(&ext3_raw(&FieldElement::<E>::from(node))?);
        }

        // Past here the transcript moves: the host path is no longer an option.
        let failed = |stage| crate::Error::DeviceFailed { stage };
        let mut shadow = shadow;
        let t = tick();
        let outcome = (|| -> Result<_, crate::Error> {
            let mut chain = chain.ok_or(failed("gruen chain"))?;
            let mut rounds = Vec::with_capacity(rounds_here);
            let mut challenges: Vec<FieldElement<E>> = Vec::with_capacity(rounds_here);
            let mut previous: Option<[u64; 3]> = None;
            for (j, left) in lefts.iter().enumerate() {
                let want_h0 = left.is_none();
                let sums = session
                    .round(previous.as_ref(), &raw_lambda, want_h0)
                    .map_err(|_| failed("gruen round"))?;
                let sum = |k: usize| ext3_from_raw::<E>(&sums[k * 3..k * 3 + 3]);
                let h0 = want_h0.then(|| sum(2));
                if want_h0 {
                    gruen::note_direct_h0();
                }
                let mut message = chain
                    .message(&point[j], left.as_ref(), &sum(0), &sum(1), h0.as_ref())
                    .ok_or(failed("gruen message"))?;
                if j == 0 && gruen::gkr_gruen_fault() {
                    // `s(0) = claim − s(1)`: what the claim then implies.
                    message.sent[0] += FieldElement::<E>::one();
                    message.s0 = &message.s0 - FieldElement::<E>::one();
                }
                if let Some((shadow, _)) = shadow.as_mut() {
                    if let Some(r) = previous.as_ref() {
                        shadow
                            .fold_first(1, r)
                            .map_err(|_| failed("gruen shadow"))?;
                    }
                    let want: Vec<FieldElement<E>> = shadow
                        .round(&nodes)
                        .map_err(|_| failed("gruen shadow"))?
                        .chunks_exact(3)
                        .map(ext3_from_raw::<E>)
                        .collect();
                    let parted = want != message.sent; // XCHECK-COMPARE
                    if parted {
                        return Err(failed("gkr gruen xcheck"));
                    }
                }
                let z = challenge(&message.sent);
                chain.advance(&point[j], &message, &z);
                previous = Some(ext3_raw(&z).ok_or(failed("challenge"))?);
                rounds.push(crate::sumcheck::RoundProof {
                    evaluations: message.sent,
                });
                challenges.push(z);
                SUMCHECK_ROUNDS.fetch_add(1, Ordering::Relaxed);
            }
            Ok((rounds, challenges, chain, previous))
        })();
        add_tick(&whir_split::GKR_ROUNDS, t);
        let (rounds, challenges, chain, previous) = match outcome {
            Ok(outcome) => outcome,
            Err(error) => return Some(Err(error)),
        };
        let t = tick();
        let Ok(values) = session.finish(previous.as_ref()) else {
            return Some(Err(failed("layer values")));
        };
        add_tick(&whir_split::GKR_VALUES, t);
        let t = tick();
        let cells = 1usize << session.low();
        if values.len() != 4 * cells * 3 {
            return Some(Err(failed("layer factors")));
        }
        let e_lo = crate::eq::eq_evals(&point[rounds_here..]);
        let mut factors = Vec::with_capacity(5);
        let eq: Vec<FieldElement<E>> = e_lo.iter().map(|v| chain.kappa() * v).collect();
        let Ok(eq) = crate::mle::Mle::new(eq) else {
            return Some(Err(failed("layer factors")));
        };
        factors.push(eq);
        for half in values.chunks_exact(cells * 3) {
            let Ok(table) =
                crate::mle::Mle::new(half.chunks_exact(3).map(ext3_from_raw::<E>).collect())
            else {
                return Some(Err(failed("layer factors")));
            };
            factors.push(table);
        }
        add_tick(&whir_split::GKR_FACTORS, t);
        let checked = match shadow.as_mut() {
            Some((shadow, _)) => {
                if let Some(r) = previous.as_ref()
                    && shadow.fold_first(1, r).is_err()
                {
                    return Some(Err(failed("gruen shadow")));
                }
                let Ok(want) = shadow.values_gathered() else {
                    return Some(Err(failed("gruen shadow")));
                };
                let same = want.len() == 5
                    && want.iter().zip(&factors).all(|(raw, table)| {
                        raw.len() == table.len() * 3
                            && raw
                                .chunks_exact(3)
                                .zip(table.evals())
                                .all(|(cell, value)| ext3_from_raw::<E>(cell) == *value)
                    });
                if !same {
                    return Some(Err(failed("gkr gruen xcheck")));
                }
                true
            }
            None => false,
        };
        gruen::note_layer(rounds_here as u64, checked);
        whir_split::bump(&whir_split::GKR_LAYERS);
        whir_split::bump(&whir_split::GKR_GRUEN);
        if checked {
            whir_split::bump(&whir_split::GKR_GRUEN_XCHECKED);
        }
        SUMCHECK_CALLS.fetch_add(1, Ordering::Relaxed);
        Some(Ok((rounds, challenges, factors)))
    }
}

/// One layer's sumcheck on the card, stepped a round at a time by its caller:
/// the batched argue's lockstep ladder (D-BATCH B-3), where every active tree's
/// round is summed on the host before the one shared challenge is drawn.
///
/// The same session [`DeviceTree::prove_layer`] opens — over the layer where it
/// lies, the input layer written again for a tree that gave it back — with the
/// rounds left to the caller.
#[cfg(feature = "cuda")]
pub struct LayerSession {
    session: math_cuda::sumcheck::SumcheckSession,
    /// The interpolation nodes `1..=degree`, three u64 each.
    nodes: Vec<u64>,
    /// A round queued by [`Self::enqueue`] and not yet read back.
    pending: Option<usize>,
}

/// A layer session the build could not open. Never constructed.
#[cfg(not(feature = "cuda"))]
pub struct LayerSession(std::convert::Infallible);

#[cfg(feature = "cuda")]
impl DeviceTree {
    /// Opens layer `layer`'s sumcheck under `program` at `point`, or `None`
    /// before anything moves when the card declines.
    pub(crate) fn layer_session<E>(
        &self,
        layer: usize,
        point: &[math::field::element::FieldElement<E>],
        program: &crate::program::Program<E>,
        degree: usize,
    ) -> Option<LayerSession>
    where
        E: math::field::traits::IsField + 'static,
    {
        let lowered = lower(program)?;
        let mut raw_point = Vec::with_capacity(point.len() * 3);
        for coordinate in point {
            raw_point.extend_from_slice(&ext3_raw(coordinate)?);
        }
        let mut nodes = Vec::with_capacity(degree * 3);
        for node in 1..=degree as u64 {
            nodes.extend_from_slice(&ext3_raw(&math::field::element::FieldElement::<E>::from(
                node,
            ))?);
        }
        let input_layer = layer + 1 == self.num_layers;
        let session = if input_layer && self.rebuild.is_some() {
            let num_vars = self.input_num_vars.checked_sub(1)?;
            if num_vars * 3 != raw_point.len() {
                return None;
            }
            drop(self.tree.lock().ok()?.take());
            let rebuilt = (self.rebuild.as_ref()?)()?;
            whir_split_bump_rebuild();
            rebuilt.sumcheck(
                &raw_point,
                &lowered.nodes,
                &lowered.consts,
                lowered.num_slots,
                lowered.root_slot,
            )
        } else {
            let held = self.tree.lock().ok()?;
            let tree = held.as_ref()?;
            if tree.layer_num_vars(layer).checked_sub(1)? * 3 != raw_point.len() {
                return None;
            }
            tree.layer_sumcheck(
                layer,
                &raw_point,
                &lowered.nodes,
                &lowered.consts,
                lowered.num_slots,
                lowered.root_slot,
            )
        };
        let session = session.ok()?;
        Some(LayerSession {
            session,
            nodes,
            pending: None,
        })
    }
}

#[cfg(feature = "cuda")]
fn whir_split_bump_rebuild() {
    crate::whir_split::bump(&crate::whir_split::GKR_REBUILDS);
}

#[cfg(not(feature = "cuda"))]
impl DeviceTree {
    pub(crate) fn layer_session<E>(
        &self,
        _layer: usize,
        _point: &[math::field::element::FieldElement<E>],
        _program: &crate::program::Program<E>,
        _degree: usize,
    ) -> Option<LayerSession>
    where
        E: math::field::traits::IsField + 'static,
    {
        match self.0 {}
    }
}

#[cfg(feature = "cuda")]
impl LayerSession {
    /// Variables left to bind.
    pub(crate) fn num_vars(&self) -> usize {
        self.session.len().trailing_zeros() as usize
    }

    /// Queues this round's launches without waiting for them, so the ladder
    /// can queue every active tree's before it reads any back.
    pub(crate) fn enqueue(&mut self) -> Result<(), crate::Error> {
        if self.pending.is_none() {
            let num_t = self
                .session
                .round_enqueue(&self.nodes)
                .map_err(|_| crate::Error::DeviceFailed { stage: "round" })?;
            self.pending = Some(num_t);
        }
        Ok(())
    }

    /// This round's polynomial at `1..=degree`: the queued round read back,
    /// or queued and read back now.
    pub(crate) fn round<E>(
        &mut self,
    ) -> Result<Vec<math::field::element::FieldElement<E>>, crate::Error>
    where
        E: math::field::traits::IsField + 'static,
    {
        self.enqueue()?;
        let num_t = self.pending.take().unwrap_or(0);
        let sums = self
            .session
            .round_collect(num_t)
            .map_err(|_| crate::Error::DeviceFailed { stage: "round" })?;
        SUMCHECK_ROUNDS.fetch_add(1, Ordering::Relaxed);
        Ok(sums.chunks_exact(3).map(ext3_from_raw::<E>).collect())
    }

    /// Binds the round's variable to `r` in every factor.
    pub(crate) fn fold<E>(
        &mut self,
        r: &math::field::element::FieldElement<E>,
    ) -> Result<(), crate::Error>
    where
        E: math::field::traits::IsField + 'static,
    {
        let raw = ext3_raw(r).ok_or(crate::Error::DeviceFailed { stage: "challenge" })?;
        self.session
            .fold(&raw)
            .map_err(|_| crate::Error::DeviceFailed { stage: "fold" })
    }

    /// Every factor as the folds left it — the weight, then the four halves.
    pub(crate) fn factors<E>(&self) -> Result<Vec<crate::mle::Mle<E>>, crate::Error>
    where
        E: math::field::traits::IsField + 'static,
    {
        let values = session_values(&self.session).map_err(|_| crate::Error::DeviceFailed {
            stage: "layer values",
        })?;
        values
            .iter()
            .map(|factor| {
                crate::mle::Mle::new(factor.chunks_exact(3).map(ext3_from_raw::<E>).collect())
            })
            .collect()
    }
}

#[cfg(not(feature = "cuda"))]
impl LayerSession {
    pub(crate) fn num_vars(&self) -> usize {
        match self.0 {}
    }

    pub(crate) fn enqueue(&mut self) -> Result<(), crate::Error> {
        match self.0 {}
    }

    pub(crate) fn round<E>(
        &mut self,
    ) -> Result<Vec<math::field::element::FieldElement<E>>, crate::Error>
    where
        E: math::field::traits::IsField + 'static,
    {
        match self.0 {}
    }

    pub(crate) fn fold<E>(
        &mut self,
        _r: &math::field::element::FieldElement<E>,
    ) -> Result<(), crate::Error>
    where
        E: math::field::traits::IsField + 'static,
    {
        match self.0 {}
    }

    pub(crate) fn factors<E>(&self) -> Result<Vec<crate::mle::Mle<E>>, crate::Error>
    where
        E: math::field::traits::IsField + 'static,
    {
        match self.0 {}
    }
}

/// The epoch's columns on the card, read by everything that would otherwise
/// upload its own copy of them.
#[cfg(feature = "cuda")]
pub struct ResidentColumns(math_cuda::columns::DeviceColumns);

/// One that could not be made. Never constructed.
#[cfg(not(feature = "cuda"))]
pub struct ResidentColumns(std::convert::Infallible);

impl std::fmt::Debug for ResidentColumns {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ResidentColumns")
    }
}

/// Puts every column of an epoch on the card, in the order given.
///
/// `None` when there is no device or it will not promise the room, and then
/// each caller uploads what it needs as before.
#[cfg(feature = "cuda")]
pub fn upload_columns<F>(columns: &[&crate::mle::Mle<F>]) -> Option<ResidentColumns>
where
    F: math::field::traits::IsField + 'static,
{
    use math::field::goldilocks::GoldilocksField;

    if std::any::TypeId::of::<F>() != std::any::TypeId::of::<GoldilocksField>() {
        return None;
    }
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *DISABLED.get_or_init(|| std::env::var_os("LAMBDA_VM_NO_GPU_COLUMNS").is_some()) {
        return None;
    }
    // SAFETY: `F == GoldilocksField`, a transparent wrapper over `u64`.
    let raw: Vec<&[u64]> = columns
        .iter()
        .map(|column| unsafe {
            core::slice::from_raw_parts(column.evals().as_ptr() as *const u64, column.len())
        })
        .collect();
    math_cuda::columns::DeviceColumns::upload(&raw).map(ResidentColumns)
}

#[cfg(not(feature = "cuda"))]
pub fn upload_columns<F>(_columns: &[&crate::mle::Mle<F>]) -> Option<ResidentColumns>
where
    F: math::field::traits::IsField + 'static,
{
    None
}

/// Where a run of columns is, for the entry points that take either.
#[cfg(feature = "cuda")]
fn columns_at<'a>(
    resident: Option<(&'a ResidentColumns, usize)>,
    host: &'a [&'a [u64]],
) -> math_cuda::columns::Columns<'a> {
    match resident {
        Some((store, first)) if store.0.is_run(first, host.len()) => {
            math_cuda::columns::Columns::Device {
                store: &store.0,
                first,
                width: host.len(),
            }
        }
        _ => math_cuda::columns::Columns::Host(host),
    }
}

/// A table's factors, uploaded once for everything that walks them.
#[cfg(feature = "cuda")]
pub struct DeviceFactors(math_cuda::sumcheck::DeviceFactors);

#[cfg(feature = "cuda")]
impl std::fmt::Debug for DeviceFactors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceFactors")
            .field("factors", &self.0.width())
            .field("cells", &self.0.len())
            .finish()
    }
}

#[cfg(feature = "cuda")]
impl DeviceFactors {
    /// Under `LAMBDA_VM_ARGUE_TREE_SYNC` (a diagnostic), wait for the kernels
    /// that built the factors, so the tree's split charges the lift its own
    /// card time.
    pub fn tree_sync(&self) {
        tree_sync!(self.0.stream());
    }

    /// The card's handle, for the fused rounds (`gpu_fused`).
    pub(crate) fn inner(&self) -> &math_cuda::sumcheck::DeviceFactors {
        &self.0
    }

    /// The factors held.
    pub fn width(&self) -> usize {
        self.0.width()
    }

    /// The cube they span.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// Factors a build declined to upload. Never constructed.
#[cfg(not(feature = "cuda"))]
#[derive(Debug)]
pub struct DeviceFactors(std::convert::Infallible);

#[cfg(not(feature = "cuda"))]
impl DeviceFactors {
    pub fn tree_sync(&self) {
        match self.0 {}
    }
}

/// A table's factors as its base columns on the card, with no lift (D-BATCH
/// M1-2, `LAMBDA_VM_ARGUE_NO_LIFT`): the fused zerocheck's rounds 0 and 1 read
/// a committed factor where the epoch's columns reside, and a public one from a
/// base copy. Keeps the columns alive for as long as it lives.
#[cfg(feature = "cuda")]
pub struct ColumnFactors {
    inner: math_cuda::sumcheck::ColumnFactors,
    _columns: std::sync::Arc<ResidentColumns>,
}

#[cfg(feature = "cuda")]
impl ColumnFactors {
    pub(crate) fn view(&self) -> math_cuda::sumcheck::FactorView<'_> {
        self.inner.view()
    }
}

#[cfg(feature = "cuda")]
impl std::fmt::Debug for ColumnFactors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ColumnFactors").finish()
    }
}

/// Column factors a build cannot make. Never constructed.
#[cfg(not(feature = "cuda"))]
#[derive(Debug)]
pub struct ColumnFactors(std::convert::Infallible);

/// The table's factors as [`ColumnFactors`], or `None` — then the caller lifts
/// them as today. `None` unless every committed factor reads its column
/// unshifted from a resident run (a cyclic shift is not an address), every
/// public table is base-valued, and the device would take the lifted factors.
#[cfg(feature = "cuda")]
pub fn column_factors<F, E>(
    columns: &[crate::mle::Mle<F>],
    kinds: &[crate::constraint_argument::FactorKind],
    public: &[crate::mle::Mle<E>],
    resident: Option<(std::sync::Arc<ResidentColumns>, usize)>,
) -> Option<ColumnFactors>
where
    F: math::field::traits::IsField + 'static,
    E: math::field::traits::IsField + 'static,
{
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;
    use math::field::goldilocks::GoldilocksField as Gl;
    use std::any::TypeId;

    if TypeId::of::<F>() != TypeId::of::<Gl>() || TypeId::of::<E>() != TypeId::of::<Ext3>() {
        return None;
    }
    let (store, first) = resident?;
    let rows = columns.first()?.len();
    if kinds.is_empty()
        || !worth_the_device(kinds.len(), rows)
        || !store.0.is_run(first, columns.len())
    {
        return None;
    }
    // A public table's base values, when nothing sits above them.
    let mut public_base: Vec<Vec<u64>> = Vec::new();
    for table in public {
        let mut base = Vec::with_capacity(rows);
        for value in table.evals() {
            let [lo, hi1, hi2] = ext3_raw(value)?;
            let canonical = ext3_raw(&math::field::element::FieldElement::<E>::from(lo))?;
            if [lo, hi1, hi2] != canonical {
                return None;
            }
            base.push(lo);
        }
        public_base.push(base);
    }
    let mut slots = Vec::with_capacity(kinds.len());
    let mut next_public = 0usize;
    for kind in kinds {
        match kind.source() {
            Some(source) if source.offset % rows == 0 && source.column < columns.len() => {
                slots.push(math_cuda::sumcheck::ColumnSlot::Column(source.column));
            }
            Some(_) => return None,
            None => {
                slots.push(math_cuda::sumcheck::ColumnSlot::Public(
                    public_base.get(next_public)?,
                ));
                next_public += 1;
            }
        }
    }
    let inner = {
        let run = store.0.view(first, columns.len());
        math_cuda::sumcheck::ColumnFactors::new(&run, rows, &slots).ok()?
    };
    COLUMN_FACTOR_TABLES.fetch_add(1, Ordering::Relaxed);
    crate::whir_split::bump(&crate::whir_split::NO_LIFT);
    Some(ColumnFactors {
        inner,
        _columns: store,
    })
}

#[cfg(not(feature = "cuda"))]
pub fn column_factors<F, E>(
    _columns: &[crate::mle::Mle<F>],
    _kinds: &[crate::constraint_argument::FactorKind],
    _public: &[crate::mle::Mle<E>],
    _resident: Option<(std::sync::Arc<ResidentColumns>, usize)>,
) -> Option<ColumnFactors>
where
    F: math::field::traits::IsField + 'static,
    E: math::field::traits::IsField + 'static,
{
    None
}

/// A table's factors, built on the device out of its base columns.
///
/// A committed factor is a column read at a frame-step offset and lifted, so
/// the columns are a third of what the factors are and the lift is a gather.
/// Building them there rather than here sends the trace instead of its lift —
/// on a real proof that is most of what crosses the bus — and the host never
/// holds the extension copy at all.
#[cfg(feature = "cuda")]
pub fn upload_factors_from_columns<F, E>(
    columns: &[crate::mle::Mle<F>],
    kinds: &[crate::constraint_argument::FactorKind],
    public: &[crate::mle::Mle<E>],
    resident: Option<(&ResidentColumns, usize)>,
) -> Option<DeviceFactors>
where
    F: math::field::traits::IsField + 'static,
    E: math::field::traits::IsField + 'static,
{
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;
    use math::field::goldilocks::GoldilocksField as Gl;
    use std::any::TypeId;

    if TypeId::of::<F>() != TypeId::of::<Gl>() || TypeId::of::<E>() != TypeId::of::<Ext3>() {
        return None;
    }
    let rows = columns.first()?.len();
    if kinds.is_empty() || !worth_the_device(kinds.len(), rows) {
        return None;
    }
    if columns.iter().any(|column| column.len() != rows)
        || public.iter().any(|table| table.len() != rows)
    {
        return None;
    }
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *DISABLED.get_or_init(|| std::env::var_os("LAMBDA_VM_NO_GPU_FACTORS").is_some()) {
        return None;
    }
    // Three u64 per committed factor — where its column starts in the
    // concatenated columns, its shift, and the slot it fills — and the public
    // tables paired with theirs.
    let mut plan = Vec::with_capacity(kinds.len() * 3);
    let mut public_slots = Vec::new();
    let mut next_public = 0usize;
    for (slot, kind) in kinds.iter().enumerate() {
        match kind.source() {
            Some(source) => {
                if source.column >= columns.len() {
                    return None;
                }
                plan.push((source.column * rows) as u64);
                plan.push((source.offset % rows) as u64);
                plan.push(slot as u64);
            }
            None => {
                public_slots.push((slot, public.get(next_public)?));
                next_public += 1;
            }
        }
    }
    if next_public != public.len() {
        return None;
    }

    // SAFETY: `F == Gl` and `E == Ext3`, each wrapping its limbs transparently
    // — one `u64` per base element, three per ext3.
    let raw_columns: Vec<&[u64]> = columns
        .iter()
        .map(|column| unsafe {
            core::slice::from_raw_parts(column.evals().as_ptr() as *const u64, column.len())
        })
        .collect();
    let raw_public: Vec<(usize, &[u64])> = public_slots
        .iter()
        .map(|(slot, table)| {
            (*slot, unsafe {
                core::slice::from_raw_parts(table.evals().as_ptr() as *const u64, table.len() * 3)
            })
        })
        .collect();

    let uploaded = math_cuda::sumcheck::DeviceFactors::from_columns(
        columns_at(resident, &raw_columns),
        &plan,
        &raw_public,
        rows,
        kinds.len(),
    )
    .ok()?;
    FACTOR_CALLS.fetch_add(1, Ordering::Relaxed);
    Some(DeviceFactors(uploaded))
}

#[cfg(not(feature = "cuda"))]
pub fn upload_factors_from_columns<F, E>(
    _columns: &[crate::mle::Mle<F>],
    _kinds: &[crate::constraint_argument::FactorKind],
    _public: &[crate::mle::Mle<E>],
    _resident: Option<(&ResidentColumns, usize)>,
) -> Option<DeviceFactors>
where
    F: math::field::traits::IsField + 'static,
    E: math::field::traits::IsField + 'static,
{
    None
}

/// The fraction tree's input layer, written by the device from the factors it
/// already holds.
///
/// Each interaction contributes a `p` and a `q` slab of `rows`, in the order
/// [`logup::input_layer`](crate::logup::input_layer) lays them out.
///
/// `padding` says whether the slots the interaction count is rounded up to are
/// written as the `0/1` they are. They are, for the layer that answers its own
/// sumcheck; they are not for the one that is only folded once, where they are
/// half the memory of the widest precompiles and the fold reads them as
/// constants instead.
#[cfg(feature = "cuda")]
fn write_input_layer<E>(
    factors: &DeviceFactors,
    numerators: &[crate::program::Program<E>],
    denominators: &[crate::program::Program<E>],
    padding: bool,
) -> Option<math_cuda::gkr::Halves>
where
    E: math::field::traits::IsField + 'static,
{
    use math::field::element::FieldElement;

    let rows = factors.0.len();
    let slots = numerators.len().next_power_of_two();
    let cells = if padding {
        slots * rows
    } else {
        numerators.len() * rows
    };
    let stream = factors.0.stream().clone();
    let mut p = math_cuda::device::alloc_zeros_or_trim::<u64>(&stream, cells * 3).ok()?;
    let mut q = math_cuda::device::alloc_zeros_or_trim::<u64>(&stream, cells * 3).ok()?;

    for (i, (numerator, denominator)) in numerators.iter().zip(denominators).enumerate() {
        for (program, out) in [(numerator, &mut p), (denominator, &mut q)] {
            let t = crate::whir_split::tick();
            let lowered = lower(program)?;
            crate::whir_split::add_tick(&crate::whir_split::TREE_LOWER, t);
            factors
                .0
                .map_program(
                    &lowered.nodes,
                    &lowered.consts,
                    lowered.num_slots,
                    lowered.root_slot,
                    out,
                    i * rows,
                )
                .ok()?;
        }
    }
    if padding {
        // The padding interactions: numerator zero (already), denominator one.
        let one = ext3_raw(&FieldElement::<E>::one())?;
        let tail = (slots - numerators.len()) * rows;
        math_cuda::sumcheck::fill_ext3(&stream, &mut q, numerators.len() * rows, tail, &one)
            .ok()?;
    }
    Some((p, q))
}

/// The tree over that input layer, built without carrying the padding.
///
/// The layer is folded once and let go; when its own sumcheck comes — last,
/// with every level above it spent — it is written again, padding and all.
/// Writing it twice costs one pass of the interactions' programs; carrying it
/// costs the memory that decides whether the widest precompiles get a device
/// at all.
#[cfg(feature = "cuda")]
pub fn input_layer_tree<E>(
    factors: std::sync::Arc<DeviceFactors>,
    numerators: Vec<crate::program::Program<E>>,
    denominators: Vec<crate::program::Program<E>>,
) -> Option<DeviceTree>
where
    E: math::field::traits::IsField + 'static,
{
    let rows = factors.0.len();
    let interactions = numerators.len();
    let stream = factors.0.stream().clone();
    input_layer_tree_impl(
        stream,
        rows,
        interactions,
        lifted_writer(factors, numerators, denominators),
        false,
    )
}

/// Writes a tree's input layer, padded to a power of two or not — the call a
/// tree makes at its build and, when it gave the layer back, for its last
/// sumcheck.
#[cfg(feature = "cuda")]
type InputWriter = std::sync::Arc<dyn Fn(bool) -> Option<math_cuda::gkr::Halves> + Send + Sync>;

/// The writer that runs each interaction's two programs over the lifted
/// factors (`write_input_layer`).
#[cfg(feature = "cuda")]
fn lifted_writer<E>(
    factors: std::sync::Arc<DeviceFactors>,
    numerators: Vec<crate::program::Program<E>>,
    denominators: Vec<crate::program::Program<E>>,
) -> InputWriter
where
    E: math::field::traits::IsField + 'static,
{
    std::sync::Arc::new(move |padding| {
        write_input_layer(&factors, &numerators, &denominators, padding)
    })
}

/// The tree over a table's input layer written straight from its base
/// columns (D-BATCH M1-2, `LAMBDA_VM_ARGUE_GKR_INPUT`): one launch writes
/// every interaction's two sides from the epoch's resident columns, where
/// [`input_layer_tree`] runs two programs an interaction over the lifted
/// factors. The same cells, so the same tree, output and GKR proof.
///
/// `plan` is [`crate::logup::input_plan`]'s, over the table's run of `width`
/// columns from `first`. `None` when that run is not resident or the card
/// declines, before anything is written.
#[cfg(feature = "cuda")]
pub fn input_layer_tree_from_columns(
    resident: std::sync::Arc<ResidentColumns>,
    first: usize,
    width: usize,
    rows: usize,
    plan: &crate::logup::InputPlan,
) -> Option<DeviceTree> {
    if !resident.0.is_run(first, width) || plan.interactions() == 0 {
        return None;
    }
    let be = math_cuda::device::backend().ok()?;
    let stream = be.next_stream();
    let uploaded = std::sync::Arc::new(
        math_cuda::gkr::InputPlan::upload(
            &stream,
            &plan.side_start,
            &plan.terms,
            &plan.coeffs,
            &plan.constants,
        )
        .ok()?,
    );
    let interactions = plan.interactions();
    let writer_stream = stream.clone();
    let write: InputWriter = std::sync::Arc::new(move |padding| {
        let view = resident.0.view(first, width);
        math_cuda::gkr::input_from_columns(&writer_stream, &view, rows, &uploaded, padding).ok()
    });
    let tree = input_layer_tree_impl(stream, rows, interactions, write, false)?;
    INPUT_COLUMNS_TREES.fetch_add(1, Ordering::Relaxed);
    crate::whir_split::bump(&crate::whir_split::TREE_FROM_COLUMNS);
    Some(tree)
}

/// The prefetch sibling of [`input_layer_tree`]: builds the tree WITHOUT
/// reading its output fraction, so its fold kernels stay in flight and overlap
/// the caller's argue instead of being waited on here. The output is read at
/// the consume site (`DeviceTree::output`), by which time the kernels have run.
///
/// `None` unless a second tree fits WITHOUT evicting the round-2 retained
/// layers: prefetch is subordinate to argue and to retention, never displacing
/// either. The headroom is the budget's free bytes (`vram_budget_bytes` less
/// what is already reserved, which INCLUDES retention), checked against the
/// carry peak so the build's own reservations then never need to evict.
#[cfg(feature = "cuda")]
pub fn input_layer_tree_deferred<E>(
    factors: std::sync::Arc<DeviceFactors>,
    numerators: Vec<crate::program::Program<E>>,
    denominators: Vec<crate::program::Program<E>>,
) -> Option<DeviceTree>
where
    E: math::field::traits::IsField + 'static,
{
    let rows = factors.0.len();
    let slots = numerators.len().next_power_of_two();
    let full = slots * rows;
    // The carry peak — also what `from_device` reserves internally — so a pass
    // here means none of the build's reservations displaces anything.
    let need = (4 * full) as u64 * 24;
    let be = math_cuda::device::backend().ok()?;
    if be.vram_budget_bytes().saturating_sub(be.reserved_bytes()) < need {
        return None;
    }
    let interactions = numerators.len();
    let stream = factors.0.stream().clone();
    input_layer_tree_impl(
        stream,
        rows,
        interactions,
        lifted_writer(factors, numerators, denominators),
        true,
    )
}

/// Count a GKR tree whose whole-tree promise the budget refused on the consume
/// path — in [`GKR_TREE_REFUSALS`] and in the argue surface's
/// `math_cuda::device::device_fallbacks` — and say so as it happens.
#[cfg(feature = "cuda")]
fn note_gkr_tree_refusal(bytes: u64, cells: usize) {
    let count = GKR_TREE_REFUSALS.fetch_add(1, Ordering::Relaxed) + 1;
    math_cuda::device::note_device_fallback();
    eprintln!(
        "[gkr] tree promise REFUSED: {} MiB for {cells} cells, {} — the host builds this \
         table's factors and tree; gkr tree refusals {count}",
        bytes >> 20,
        budget_headroom(),
    );
}

/// `defer` skips the one output read (the sync), leaving it for the consume
/// site; everything else is identical, so the eager caller is byte-for-byte the
/// path it always was.
#[cfg(feature = "cuda")]
fn input_layer_tree_impl(
    stream: std::sync::Arc<math_cuda::CudaStream>,
    rows: usize,
    interactions: usize,
    write: InputWriter,
    defer: bool,
) -> Option<DeviceTree> {
    let slots = interactions.next_power_of_two();
    let full = slots * rows;
    let real = interactions * rows;

    // Carrying the layer cannot fail late; handing it back can, and by then
    // the transcript has moved and there is no host path left. So it is
    // carried whenever the card has room for it, and handed back only when
    // that is what the card cannot do **and** the difference is worth the
    // second chance to fail — which is the widest precompiles, where it is
    // gigabytes, and nothing else.
    let eager = (4 * full) as u64 * 24;
    let lazy = math_cuda::gkr::padded_peak_bytes(real, full);
    let carried =
        math_cuda::device::reserve(eager).filter(|_| !HAND_BACK_FORCED.load(Ordering::Relaxed));
    let hand_back = HAND_BACK_FORCED.load(Ordering::Relaxed);
    if !hand_back && (carried.is_some() || eager - lazy < worth_handing_back()) {
        drop(carried);
        let t = crate::whir_split::tick();
        let (p, q) = write(true)?;
        tree_sync!(stream);
        crate::whir_split::add_tick(&crate::whir_split::TREE_WRITE, t);
        let t = crate::whir_split::tick();
        let tree = math_cuda::gkr::DeviceFractionTree::from_device(stream.clone(), p, q).ok()?;
        tree_sync!(stream);
        crate::whir_split::add_tick(&crate::whir_split::TREE_FOLD, t);
        let output = std::sync::OnceLock::new();
        if !defer {
            let t = crate::whir_split::tick();
            let _ = output.set(tree.output().ok()?);
            crate::whir_split::add_tick(&crate::whir_split::TREE_OUTPUT, t);
        }
        let num_layers = tree.num_layers();
        let input_num_vars = tree.layer_num_vars(num_layers - 1);
        TREE_CALLS.fetch_add(1, Ordering::Relaxed);
        return Some(DeviceTree {
            tree: std::sync::Mutex::new(Some(tree)),
            rebuild: None,
            num_layers,
            input_num_vars,
            output,
            _room: None,
        });
    }

    // One promise for the whole tree, held past it: the layer handed back for
    // the last sumcheck is part of the same structure.
    //
    // ⛔ ITS REFUSAL IS A HOST FALLBACK on the consume path: `logup::resident_tree`
    // gets `None` and the table's factors and whole tree are built on the host.
    // So it is counted, here where it is refused, and only when `!defer`: a
    // prefetch that is refused only means no prefetch, and the consume site asks
    // again.
    let promise = math_cuda::gkr::padded_peak_bytes(real, full);
    let Some(room) = math_cuda::device::reserve(promise) else {
        if !defer {
            note_gkr_tree_refusal(promise, full);
        }
        return None;
    };

    let t = crate::whir_split::tick();
    let (p, q) = write(false)?;
    tree_sync!(stream);
    crate::whir_split::add_tick(&crate::whir_split::TREE_WRITE, t);
    let num_vars = full.trailing_zeros() as usize;
    let t = crate::whir_split::tick();
    let tree =
        math_cuda::gkr::DeviceFractionTree::from_padded_input(stream.clone(), p, q, real, num_vars)
            .ok()?;
    tree_sync!(stream);
    crate::whir_split::add_tick(&crate::whir_split::TREE_FOLD, t);
    let output = std::sync::OnceLock::new();
    if !defer {
        let t = crate::whir_split::tick();
        let _ = output.set(tree.output().ok()?);
        crate::whir_split::add_tick(&crate::whir_split::TREE_OUTPUT, t);
    }
    let num_layers = tree.num_layers();

    let rebuild = move || {
        let (p, q) = write(true)?;
        Some(math_cuda::gkr::InputLayer::new(
            stream.clone(),
            p,
            q,
            num_vars,
        ))
    };

    TREE_CALLS.fetch_add(1, Ordering::Relaxed);
    Some(DeviceTree {
        tree: std::sync::Mutex::new(Some(tree)),
        rebuild: Some(Box::new(rebuild)),
        num_layers,
        input_num_vars: num_vars,
        output,
        _room: Some(room),
    })
}

#[cfg(not(feature = "cuda"))]
pub fn input_layer_tree<E>(
    _factors: std::sync::Arc<DeviceFactors>,
    _numerators: Vec<crate::program::Program<E>>,
    _denominators: Vec<crate::program::Program<E>>,
) -> Option<DeviceTree>
where
    E: math::field::traits::IsField + 'static,
{
    None
}

#[cfg(not(feature = "cuda"))]
pub fn input_layer_tree_from_columns(
    _resident: std::sync::Arc<ResidentColumns>,
    _first: usize,
    _width: usize,
    _rows: usize,
    _plan: &crate::logup::InputPlan,
) -> Option<DeviceTree> {
    None
}

#[cfg(not(feature = "cuda"))]
pub fn input_layer_tree_deferred<E>(
    _factors: std::sync::Arc<DeviceFactors>,
    _numerators: Vec<crate::program::Program<E>>,
    _denominators: Vec<crate::program::Program<E>>,
) -> Option<DeviceTree>
where
    E: math::field::traits::IsField + 'static,
{
    None
}

/// The opening's two factors on a device: the weight the chain carries and the
/// message it is opening, resident across groups of rounds.
#[cfg(feature = "cuda")]
pub struct OpeningFactors {
    state: OpeningState,
    lowered: Lowered,
}

/// Where the factors are: still in share form for the first rounds, or
/// materialised.
#[cfg(feature = "cuda")]
enum OpeningState {
    /// The first `rounds` rounds run over the shares
    /// ([`math_cuda::whir_open::LeanRound0`]); the tables are written only
    /// once they are bound.
    Lean {
        lean: Box<math_cuda::whir_open::LeanRound0>,
        rounds: usize,
    },
    Session(math_cuda::whir_open::OpeningSession),
    /// A materialisation failed, after the transcript had moved.
    Gone,
}

/// Rounds a stacked opening runs over its shares before materialising its
/// factors (`LAMBDA_VM_WHIR_LEAN_ROUNDS`, `0..=6`; default 6, a first6 chain's
/// whole first group). `0` materialises at full width up front, which is the
/// path before this existed; every setting proves the same values.
///
/// The factors at full width are the weight and the lifted message, `2·2^n`
/// extension values — 1.5× the committed base codeword, plus the base staging
/// while they are built — and they grow with the stack. Bound first, they are
/// `2^(n − rounds)` wide.
#[cfg(feature = "cuda")]
pub const LEAN_ROUNDS_DEFAULT: usize = 6;

#[cfg(feature = "cuda")]
static LEAN_ROUNDS_FORCED: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(usize::MAX);

#[cfg(feature = "cuda")]
fn lean_rounds() -> usize {
    let forced = LEAN_ROUNDS_FORCED.load(Ordering::Relaxed);
    if forced != usize::MAX {
        return forced;
    }
    static SET: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *SET.get_or_init(|| {
        std::env::var("LAMBDA_VM_WHIR_LEAN_ROUNDS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .map(|v| v.min(math_cuda::whir::FUSED_MAX_FOLD))
            .unwrap_or(LEAN_ROUNDS_DEFAULT)
    })
}

/// Overrides [`LEAN_ROUNDS_DEFAULT`] / `LAMBDA_VM_WHIR_LEAN_ROUNDS` for the
/// whole process — for a parity test that proves both ways in one binary.
/// `None` restores the environment's setting. Not for production callers.
#[doc(hidden)]
#[cfg(feature = "cuda")]
pub fn force_lean_rounds(rounds: Option<usize>) {
    LEAN_ROUNDS_FORCED.store(rounds.unwrap_or(usize::MAX), Ordering::Relaxed);
}

/// Whether a device fold runs every level in one launch
/// ([`math_cuda::whir::fold_resident_fused`]) or one launch per level. `true`
/// unless `LAMBDA_VM_NO_WHIR_FUSED_FOLD` is set; the two are raw-identical.
#[cfg(feature = "cuda")]
static FUSED_FOLD_FORCED: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

#[cfg(feature = "cuda")]
fn fused_fold() -> bool {
    match FUSED_FOLD_FORCED.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => {
            static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            !*OFF.get_or_init(|| std::env::var_os("LAMBDA_VM_NO_WHIR_FUSED_FOLD").is_some())
        }
    }
}

/// The same override for the fused fold. Not for production callers.
#[doc(hidden)]
#[cfg(feature = "cuda")]
pub fn force_fused_fold(on: Option<bool>) {
    FUSED_FOLD_FORCED.store(
        match on {
            Some(true) => 1,
            Some(false) => 2,
            None => 0,
        },
        Ordering::Relaxed,
    );
}

/// Whether [`open_shared`] declines as a full card would make it, so a test can
/// reach the path [`note_open_host_fallback`] counts without filling one.
#[cfg(feature = "cuda")]
static SHARED_OPEN_DECLINED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Makes every stacked opening's device factors decline while `on` — for the
/// test that the host path they fall to is counted. Not for production callers.
#[doc(hidden)]
#[cfg(feature = "cuda")]
pub fn force_shared_open_declined(on: bool) {
    SHARED_OPEN_DECLINED.store(on, Ordering::Relaxed);
}

/// Factors a build declined to upload. Never constructed.
#[cfg(not(feature = "cuda"))]
pub struct OpeningFactors(std::convert::Infallible);

/// Puts the chain's two factors on a device, lifting the message on the way.
///
/// `program` is the rule the rounds evaluate — the product of the two — which
/// the host path runs as a closure.
#[cfg(feature = "cuda")]
pub(crate) fn open_on_device<F, E>(
    message: &crate::mle::Mle<F>,
    weight: &crate::mle::Mle<E>,
    program: &crate::program::Program<E>,
) -> Option<OpeningFactors>
where
    F: math::field::traits::IsField + 'static,
    E: math::field::traits::IsField + 'static,
{
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;
    use math::field::goldilocks::GoldilocksField as Gl;
    use std::any::TypeId;

    if TypeId::of::<F>() != TypeId::of::<Gl>() || TypeId::of::<E>() != TypeId::of::<Ext3>() {
        return None;
    }
    if message.len() != weight.len() || message.len() < SUMCHECK_THRESHOLD {
        return None;
    }
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *DISABLED.get_or_init(|| std::env::var_os("LAMBDA_VM_NO_GPU_OPEN").is_some()) {
        return None;
    }
    let lowered = lower(program)?;

    // SAFETY: the fields are the two checked above, each wrapping its limbs
    // transparently — one `u64` per base element, three per ext3.
    let raw_weight = unsafe {
        core::slice::from_raw_parts(weight.evals().as_ptr() as *const u64, weight.len() * 3)
    };
    let raw_message = unsafe {
        core::slice::from_raw_parts(message.evals().as_ptr() as *const u64, message.len())
    };
    let session = math_cuda::whir_open::OpeningSession::new(raw_weight, raw_message).ok()?;
    OPEN_CALLS.fetch_add(1, Ordering::Relaxed);
    Some(OpeningFactors {
        state: OpeningState::Session(session),
        lowered,
    })
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn open_on_device<F, E>(
    _message: &crate::mle::Mle<F>,
    _weight: &crate::mle::Mle<E>,
    _program: &crate::program::Program<E>,
) -> Option<OpeningFactors>
where
    F: math::field::traits::IsField + 'static,
    E: math::field::traits::IsField + 'static,
{
    None
}

#[cfg(feature = "cuda")]
impl OpeningFactors {
    pub(crate) fn num_vars(&self) -> usize {
        match &self.state {
            OpeningState::Lean { lean, .. } => lean.num_vars(),
            OpeningState::Session(session) => session.num_vars(),
            OpeningState::Gone => 0,
        }
    }

    /// Writes the lean factors out, if they are still in share form.
    fn materialize(&mut self) -> Result<&mut math_cuda::whir_open::OpeningSession, crate::Error> {
        materialize_state(&mut self.state)
    }

    /// One group of rounds, with the host drawing each challenge.
    ///
    /// The factors are folded in place and stay for the next group, which is
    /// the whole point: they are the width of a stacked polynomial. A lean
    /// opening runs its first rounds over the shares and materialises when
    /// they are spent or the group ends — whichever comes first, since what
    /// follows a group (the out-of-domain answer, the weight's update) reads
    /// the tables.
    pub(crate) fn rounds<E>(
        &mut self,
        group: usize,
        degree: usize,
        mut challenge: impl FnMut(
            &[math::field::element::FieldElement<E>],
        ) -> math::field::element::FieldElement<E>,
    ) -> Result<SumcheckRounds<E>, crate::Error>
    where
        E: math::field::traits::IsField + 'static,
    {
        use math::field::element::FieldElement;

        let failed = |stage| crate::Error::DeviceFailed { stage };
        let mut proofs = Vec::with_capacity(group);
        let mut challenges = Vec::with_capacity(group);
        if let OpeningState::Lean { lean, rounds } = &mut self.state {
            // The opening's rule is `w·f`, so its rounds are degree two: the
            // lean kernel evaluates that product and nothing else.
            if degree != 2 {
                return Err(failed("lean opening degree"));
            }
            let mut t = Vec::with_capacity(degree * 3);
            for node in 1..=degree {
                t.extend_from_slice(
                    &ext3_raw(&FieldElement::<E>::from(node as u64))
                        .ok_or_else(|| failed("interpolation node"))?,
                );
            }
            while proofs.len() < group && lean.bound() < *rounds && lean.num_vars() > 0 {
                let sums = lean.round(&t).map_err(|_| failed("lean round"))?;
                let evaluations: Vec<FieldElement<E>> =
                    sums.chunks_exact(3).map(ext3_from_raw::<E>).collect();
                let r = challenge(&evaluations);
                let raw = ext3_raw(&r).ok_or_else(|| failed("challenge"))?;
                lean.bind(&raw);
                proofs.push(crate::sumcheck::RoundProof { evaluations });
                challenges.push(r);
                SUMCHECK_ROUNDS.fetch_add(1, Ordering::Relaxed);
            }
        }
        let left = group - proofs.len();
        let session = materialize_state(&mut self.state)?;
        let lowered = &self.lowered;
        if left > 0 {
            let mut rounds = session
                .sumcheck(
                    &lowered.nodes,
                    &lowered.consts,
                    lowered.num_slots,
                    lowered.root_slot,
                )
                .map_err(|_| failed("opening session"))?;
            let (more, points) = run_rounds(&mut rounds, degree, left, challenge, |_| None)?;
            session.bound(left);
            proofs.extend(more);
            challenges.extend(points);
        }
        SUMCHECK_CALLS.fetch_add(1, Ordering::Relaxed);
        // The factors stay on device; the caller reads them through this
        // handle, not through the tables it no longer has.
        Ok((proofs, challenges, Vec::new()))
    }

    /// The message's value at `point`, which the out-of-domain answer needs.
    pub(crate) fn evaluate_message<E>(
        &mut self,
        point: &[math::field::element::FieldElement<E>],
    ) -> Result<math::field::element::FieldElement<E>, crate::Error>
    where
        E: math::field::traits::IsField + 'static,
    {
        let raw = raw_point(point).ok_or(crate::Error::DeviceFailed { stage: "ood point" })?;
        let value = self
            .materialize()?
            .evaluate_message(&raw)
            .map_err(|_| crate::Error::DeviceFailed { stage: "ood value" })?;
        Ok(ext3_from_raw::<E>(&value))
    }

    /// `weight += gamma · eq(point, ·)`: the weight the next group carries.
    pub(crate) fn add_scaled_eq<E>(
        &mut self,
        point: &[math::field::element::FieldElement<E>],
        gamma: &math::field::element::FieldElement<E>,
    ) -> Result<(), crate::Error>
    where
        E: math::field::traits::IsField + 'static,
    {
        let failed = |stage| crate::Error::DeviceFailed { stage };
        let raw = raw_point(point).ok_or_else(|| failed("weight point"))?;
        let scale = ext3_raw(gamma).ok_or_else(|| failed("weight scale"))?;
        self.materialize()?
            .add_scaled_eq(&raw, &scale)
            .map_err(|_| failed("weight"))
    }

    /// Device bytes the factors hold right now — the share-form working set
    /// before the lean rounds are spent, the tables after.
    pub fn device_bytes(&self) -> u64 {
        match &self.state {
            OpeningState::Lean { lean, .. } => lean.device_bytes(),
            OpeningState::Session(session) => session.device_bytes(),
            OpeningState::Gone => 0,
        }
    }

    /// Whether the factors are still in share form.
    pub fn is_lean(&self) -> bool {
        matches!(self.state, OpeningState::Lean { .. })
    }
}

/// A point as the limbs the kernels read.
#[cfg(feature = "cuda")]
fn raw_point<E>(point: &[math::field::element::FieldElement<E>]) -> Option<Vec<u64>>
where
    E: math::field::traits::IsField + 'static,
{
    let mut raw = Vec::with_capacity(point.len() * 3);
    for coordinate in point {
        raw.extend_from_slice(&ext3_raw(coordinate)?);
    }
    Some(raw)
}

#[cfg(not(feature = "cuda"))]
impl OpeningFactors {
    pub(crate) fn num_vars(&self) -> usize {
        match self.0 {}
    }

    pub(crate) fn rounds<E>(
        &mut self,
        _group: usize,
        _degree: usize,
        _challenge: impl FnMut(
            &[math::field::element::FieldElement<E>],
        ) -> math::field::element::FieldElement<E>,
    ) -> Result<SumcheckRounds<E>, crate::Error>
    where
        E: math::field::traits::IsField + 'static,
    {
        match self.0 {}
    }

    pub(crate) fn evaluate_message<E>(
        &mut self,
        _point: &[math::field::element::FieldElement<E>],
    ) -> Result<math::field::element::FieldElement<E>, crate::Error>
    where
        E: math::field::traits::IsField + 'static,
    {
        match self.0 {}
    }

    pub(crate) fn add_scaled_eq<E>(
        &mut self,
        _point: &[math::field::element::FieldElement<E>],
        _gamma: &math::field::element::FieldElement<E>,
    ) -> Result<(), crate::Error>
    where
        E: math::field::traits::IsField + 'static,
    {
        match self.0 {}
    }
}

/// The same with the weight written on device from its shares: a stacked
/// polynomial's weight is as wide as the polynomial, and sending it would cost
/// more than the rounds that read it.
#[cfg(feature = "cuda")]
pub(crate) fn open_shared<F, E>(
    message: &crate::whir_chain::Stacked<'_, F>,
    shares: &[crate::stacked_eval::WeightShare<'_, E>],
    n_stack: usize,
    program: &crate::program::Program<E>,
) -> Option<OpeningFactors>
where
    F: math::field::traits::IsField + 'static,
    E: math::field::traits::IsField + 'static,
{
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;
    use math::field::goldilocks::GoldilocksField as Gl;
    use std::any::TypeId;

    if TypeId::of::<F>() != TypeId::of::<Gl>() || TypeId::of::<E>() != TypeId::of::<Ext3>() {
        return None;
    }
    let len = 1usize << message.num_vars();
    if message.num_vars() != n_stack || len < SUMCHECK_THRESHOLD {
        return None;
    }
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *DISABLED.get_or_init(|| std::env::var_os("LAMBDA_VM_NO_GPU_OPEN").is_some())
        || SHARED_OPEN_DECLINED.load(Ordering::Relaxed)
    {
        return None;
    }
    let lowered = lower(program)?;
    let shares: Vec<(usize, Vec<u64>, [u64; 3])> = shares
        .iter()
        .map(|share| {
            Some((
                share.offset,
                raw_point(share.point)?,
                ext3_raw(&share.scale)?,
            ))
        })
        .collect::<Option<_>>()?;

    // SAFETY: `F == GoldilocksField`, a transparent wrapper over `u64`.
    let raw = |table: &crate::mle::Mle<F>| unsafe {
        core::slice::from_raw_parts(table.evals().as_ptr() as *const u64, table.len())
    };
    // The columns go straight into the device buffer: what the host would
    // assemble first is a copy of the bytes about to be sent.
    if message
        .parts
        .iter()
        .any(|(column, offset)| offset + column.len() > len)
    {
        return None;
    }
    let host_parts: Vec<(&[u64], usize)> = message
        .parts
        .iter()
        .map(|(column, offset)| (raw(column), *offset))
        .collect();
    // ★ The first rounds over the shares, when the device takes them: the
    // factors are then written at `2^(n − rounds)` instead of `2^n`. The same
    // values either way; a stack the lean path cannot read materialises as
    // before, before any challenge is drawn.
    let rounds = lean_rounds().min(n_stack);
    if rounds > 0
        && let Some(lean) = lean_opening(&shares, n_stack, message, &host_parts)
    {
        OPEN_CALLS.fetch_add(1, Ordering::Relaxed);
        LEAN_OPEN_CALLS.fetch_add(1, Ordering::Relaxed);
        return Some(OpeningFactors {
            state: OpeningState::Lean {
                lean: Box::new(lean),
                rounds,
            },
            lowered,
        });
    }
    let session = match &message.resident {
        Some((store, parts)) => math_cuda::whir_open::OpeningSession::from_shares_and_resident(
            &shares, len, &store.0, parts,
        ),
        None => {
            math_cuda::whir_open::OpeningSession::from_shares_and_parts(&shares, len, &host_parts)
        }
    }
    .ok()?;
    OPEN_CALLS.fetch_add(1, Ordering::Relaxed);
    Some(OpeningFactors {
        state: OpeningState::Session(session),
        lowered,
    })
}

/// The factors as tables: written out of share form if they still are. Past
/// the first round the transcript has moved, so a failure here is final.
#[cfg(feature = "cuda")]
fn materialize_state(
    state: &mut OpeningState,
) -> Result<&mut math_cuda::whir_open::OpeningSession, crate::Error> {
    if let OpeningState::Lean { .. } = state {
        let OpeningState::Lean { lean, .. } = core::mem::replace(state, OpeningState::Gone) else {
            unreachable!("matched above")
        };
        let session = (*lean)
            .materialize()
            .map_err(|_| crate::Error::DeviceFailed {
                stage: "lean materialize",
            })?;
        *state = OpeningState::Session(session);
    }
    match state {
        OpeningState::Session(session) => Ok(session),
        _ => Err(crate::Error::DeviceFailed {
            stage: "opening factors",
        }),
    }
}

/// The lean opening of a stacked polynomial: its shares with their points'
/// `eq` tables in halves (computed here, a few kilobytes each), and the
/// message where it already is. `None` when the stack is not one it can read.
#[cfg(feature = "cuda")]
fn lean_opening<F>(
    shares: &[(usize, Vec<u64>, [u64; 3])],
    n_stack: usize,
    message: &crate::whir_chain::Stacked<'_, F>,
    host_parts: &[(&[u64], usize)],
) -> Option<math_cuda::whir_open::LeanRound0>
where
    F: math::field::traits::IsField + 'static,
{
    use math::field::element::FieldElement;
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;

    // Sorted by offset: the map from a stack position to its column is a
    // binary search over them.
    let mut order: Vec<usize> = (0..shares.len()).collect();
    order.sort_by_key(|i| shares[*i].0);
    let mut lean_shares = Vec::with_capacity(shares.len());
    let mut eq: Vec<u64> = Vec::new();
    // A table's columns share its point and sit next to each other: one pair
    // of half tables per run of equal points.
    let mut last: Option<(&[u64], usize, usize, usize)> = None;
    for &i in &order {
        let (offset, point, scale) = &shares[i];
        let num_vars = point.len() / 3;
        let lo_bits = num_vars / 2;
        let (hi_at, lo_at) = match last {
            Some((p, lo, hi_at, lo_at)) if p == point.as_slice() && lo == lo_bits => (hi_at, lo_at),
            _ => {
                let coordinates: Vec<FieldElement<Ext3>> =
                    point.chunks_exact(3).map(ext3_from_raw::<Ext3>).collect();
                let (hi, lo) = coordinates.split_at(num_vars - lo_bits);
                let hi_at = eq.len() / 3;
                for value in crate::eq::eq_evals(hi) {
                    eq.extend_from_slice(&ext3_raw(&value)?);
                }
                let lo_at = eq.len() / 3;
                for value in crate::eq::eq_evals(lo) {
                    eq.extend_from_slice(&ext3_raw(&value)?);
                }
                (hi_at, lo_at)
            }
        };
        last = Some((point.as_slice(), lo_bits, hi_at, lo_at));
        lean_shares.push(math_cuda::whir_open::LeanShare {
            stack_offset: *offset,
            num_vars,
            lo_bits,
            hi_at,
            lo_at,
            scale: *scale,
        });
    }
    let source = match &message.resident {
        Some((store, parts)) => math_cuda::whir_open::LeanMessage::Resident {
            store: &store.0,
            parts,
        },
        None => math_cuda::whir_open::LeanMessage::Parts(host_parts),
    };
    math_cuda::whir_open::LeanRound0::new(&lean_shares, &eq, n_stack, source)
        .ok()
        .flatten()
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn open_shared<F, E>(
    _message: &crate::whir_chain::Stacked<'_, F>,
    _shares: &[crate::stacked_eval::WeightShare<'_, E>],
    _n_stack: usize,
    _program: &crate::program::Program<E>,
) -> Option<OpeningFactors>
where
    F: math::field::traits::IsField + 'static,
    E: math::field::traits::IsField + 'static,
{
    None
}

/// A codeword the device holds: the commit leaves one there and the chain
/// folds it there, so the array itself never crosses the bus.
#[cfg(feature = "cuda")]
pub struct DeviceCodeword(math_cuda::whir::DeviceCodeword);

/// A codeword a commit declined to keep. Never constructed.
#[cfg(not(feature = "cuda"))]
pub struct DeviceCodeword(std::convert::Infallible);

#[cfg(feature = "cuda")]
impl std::fmt::Debug for DeviceCodeword {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceCodeword")
            .field("elements", &self.0.elements())
            .field("base", &self.0.is_base())
            .finish()
    }
}

#[cfg(not(feature = "cuda"))]
impl std::fmt::Debug for DeviceCodeword {
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {}
    }
}

/// The same, over a stacked polynomial given as the columns it is made of.
///
/// The parts go straight into the device buffer, so what the host would have
/// assembled first — a copy of every byte about to be uploaded — is never
/// built. `parts` is `(column, offset in elements)`.
#[cfg(feature = "cuda")]
pub(crate) fn commit_parts<F>(
    parts: &[(&crate::mle::Mle<F>, usize)],
    log_evals: usize,
    log_blowup: usize,
    log_folding: usize,
    transient: bool,
    hash: crate::whir_hash::DeviceHashKey,
) -> Option<(DeviceCodeword, [u8; 32])>
where
    F: math::field::traits::IsField + 'static,
{
    use math::field::goldilocks::GoldilocksField;

    if std::any::TypeId::of::<F>() != std::any::TypeId::of::<GoldilocksField>() {
        return None;
    }
    if (1usize << log_evals) << log_blowup < COMMIT_THRESHOLD {
        return None;
    }
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *DISABLED.get_or_init(|| std::env::var_os("LAMBDA_VM_NO_GPU_WHIR_COMMIT").is_some()) {
        return None;
    }
    // A part past the end would be an out-of-bounds device write.
    if parts
        .iter()
        .any(|(column, offset)| offset + column.len() > (1usize << log_evals))
    {
        return None;
    }
    // SAFETY: `F == GoldilocksField`, a transparent wrapper over `u64`.
    let raw: Vec<(&[u64], usize)> = parts
        .iter()
        .map(|(column, offset)| unsafe {
            (
                core::slice::from_raw_parts(column.evals().as_ptr() as *const u64, column.len()),
                *offset,
            )
        })
        .collect();
    // No device is a decline, not an error: the device is asked only once one
    // exists (a GPU-less `cuda` build fails here on every commit).
    math_cuda::device::backend().ok()?;
    let (codeword, root) = match math_cuda::whir::commit_codeword_parts(
        &raw,
        log_evals,
        log_blowup,
        log_folding,
        transient,
        hash.into_math_cuda(),
    ) {
        Ok(committed) => committed,
        Err(err) => {
            note_commit_error("parts", log_evals, log_blowup, &err);
            return None;
        }
    };
    COMMIT_CALLS.fetch_add(1, Ordering::Relaxed);
    Some((DeviceCodeword(codeword), root))
}

/// The same for parts the card already holds.
#[cfg(feature = "cuda")]
pub(crate) fn commit_resident(
    store: &ResidentColumns,
    parts: &[(usize, usize)],
    log_evals: usize,
    log_blowup: usize,
    log_folding: usize,
    transient: bool,
    hash: crate::whir_hash::DeviceHashKey,
) -> Option<(DeviceCodeword, [u8; 32])> {
    if (1usize << log_evals) << log_blowup < COMMIT_THRESHOLD {
        return None;
    }
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *DISABLED.get_or_init(|| std::env::var_os("LAMBDA_VM_NO_GPU_WHIR_COMMIT").is_some()) {
        return None;
    }
    // No device is a decline, not an error (see `commit_parts`).
    math_cuda::device::backend().ok()?;
    let (codeword, root) = match math_cuda::whir::commit_codeword_resident(
        &store.0,
        parts,
        log_evals,
        log_blowup,
        log_folding,
        transient,
        hash.into_math_cuda(),
    ) {
        Ok(committed) => committed,
        Err(err) => {
            note_commit_error("resident", log_evals, log_blowup, &err);
            return None;
        }
    };
    COMMIT_CALLS.fetch_add(1, Ordering::Relaxed);
    Some((DeviceCodeword(codeword), root))
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn commit_resident(
    _store: &ResidentColumns,
    _parts: &[(usize, usize)],
    _log_evals: usize,
    _log_blowup: usize,
    _log_folding: usize,
    _transient: bool,
    _hash: crate::whir_hash::DeviceHashKey,
) -> Option<(DeviceCodeword, [u8; 32])> {
    None
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn commit_parts<F>(
    _parts: &[(&crate::mle::Mle<F>, usize)],
    _log_evals: usize,
    _log_blowup: usize,
    _log_folding: usize,
    _transient: bool,
    _hash: crate::whir_hash::DeviceHashKey,
) -> Option<(DeviceCodeword, [u8; 32])>
where
    F: math::field::traits::IsField + 'static,
{
    None
}

#[cfg(feature = "cuda")]
impl DeviceCodeword {
    pub(crate) fn elements(&self) -> usize {
        self.0.elements()
    }

    /// Folds it `alphas.len()` times, leaving the result on device too.
    pub(crate) fn fold<F, N>(
        &self,
        generator: &math::field::element::FieldElement<F>,
        alphas: &[math::field::element::FieldElement<N>],
    ) -> Option<Self>
    where
        F: math::field::traits::IsField + 'static,
        N: math::field::traits::IsField + 'static,
    {
        let (two_inv, g_invs, raw_alphas) = fold_scalars(generator, alphas)?;
        // ★ Every level in one launch when it fits: the level-by-level fold
        // holds a 2.25× codeword transient beside the committed codeword, the
        // fused one only its output. Raw-identical either way.
        let folded = if fused_fold() && g_invs.len() <= math_cuda::whir::FUSED_MAX_FOLD {
            let folded =
                math_cuda::whir::fold_resident_fused(&self.0, two_inv, &g_invs, &raw_alphas)
                    .ok()?;
            FUSED_FOLD_CALLS.fetch_add(1, Ordering::Relaxed);
            folded
        } else {
            math_cuda::whir::fold_resident(&self.0, two_inv, &g_invs, &raw_alphas).ok()?
        };
        RESIDENT_FOLD_CALLS.fetch_add(1, Ordering::Relaxed);
        Some(Self(folded))
    }

    /// The root of the tree over its fold blocks.
    ///
    /// The tree is kept past this call, inside the codeword's promise and
    /// evictable — whole by default, or its leaf layer under
    /// `LFM_WHIR_WHOLE_TREES=0` — because the only other thing a proof wants
    /// from it is a path per query, which [`paths`](Self::paths) reads when the
    /// queries are known (see `math_cuda::whir::DeviceCodeword::commit`).
    pub(crate) fn commit(
        &self,
        log_folding: usize,
        hash: crate::whir_hash::DeviceHashKey,
    ) -> Option<[u8; 32]> {
        let root = self.0.commit(log_folding, hash.into_math_cuda()).ok()?;
        COMMIT_CALLS.fetch_add(1, Ordering::Relaxed);
        Some(root)
    }

    /// One authentication path per index, against the same tree.
    pub(crate) fn paths(
        &self,
        log_folding: usize,
        indices: &[usize],
        hash: crate::whir_hash::DeviceHashKey,
    ) -> Option<Vec<Vec<[u8; 32]>>> {
        let leaves = self.0.elements() >> log_folding;
        if indices.iter().any(|index| *index >= leaves) {
            return None;
        }
        let positions: Vec<u32> = indices.iter().map(|index| *index as u32).collect();
        let bytes = self
            .0
            .paths(log_folding, &positions, hash.into_math_cuda())
            .ok()?;
        let depth = leaves.trailing_zeros() as usize;
        let nodes = nodes_in_place(bytes)?;
        Some(nodes.chunks_exact(depth).map(<[_]>::to_vec).collect())
    }

    /// [`paths`](Self::paths) and the tree's Merkle cap at `cap_height` (its
    /// `2^cap_height` nodes that height below the root, left to right), both
    /// from ONE rebuild of the tree — so the cap and the paths are of the same
    /// tree, whether its leaf layer was hashed or served from retention.
    pub(crate) fn paths_and_cap(
        &self,
        log_folding: usize,
        indices: &[usize],
        cap_height: usize,
        hash: crate::whir_hash::DeviceHashKey,
    ) -> Option<PathsAndCap> {
        let leaves = self.0.elements() >> log_folding;
        let depth = leaves.trailing_zeros() as usize;
        if indices.iter().any(|index| *index >= leaves) || cap_height > depth {
            return None;
        }
        let positions: Vec<u32> = indices.iter().map(|index| *index as u32).collect();
        let (bytes, cap_bytes) = self
            .0
            .paths_and_cap(log_folding, &positions, cap_height, hash.into_math_cuda())
            .ok()?;
        let nodes = nodes_in_place(bytes)?;
        // `2^c` nodes: copied rather than reinterpreted in place, because a
        // cap is a few hundred bytes and its allocation's capacity is not ours
        // to vouch for.
        if cap_bytes.len() != 32usize << cap_height {
            return None;
        }
        let cap: Vec<[u8; 32]> = cap_bytes
            .chunks_exact(32)
            .map(|node| <[u8; 32]>::try_from(node).ok())
            .collect::<Option<_>>()?;
        if cap.len() != 1usize << cap_height {
            return None;
        }
        Some((nodes.chunks_exact(depth).map(<[_]>::to_vec).collect(), cap))
    }

    /// The blocks `indices` open, gathered where they lie — one launch and one
    /// copy back for the whole round.
    pub(crate) fn cosets<F>(
        &self,
        indices: &[usize],
        num_leaves: usize,
        block: usize,
    ) -> Option<Vec<Vec<math::field::element::FieldElement<F>>>>
    where
        F: math::field::traits::IsField + 'static,
    {
        let indices: Vec<u64> = indices.iter().map(|index| *index as u64).collect();
        let raw = self.0.cosets(&indices, num_leaves, block).ok()?;
        let values = values_from_raw::<F>(&raw, self.0.is_base())?;
        Some(values.chunks_exact(block).map(<[_]>::to_vec).collect())
    }

    /// Its first value, which is what the last fold leaves behind.
    pub(crate) fn first<F>(&self) -> Option<math::field::element::FieldElement<F>>
    where
        F: math::field::traits::IsField + 'static,
    {
        let raw = self.0.first().ok()?;
        values_from_raw(&raw, false)?.into_iter().next()
    }
}

/// `two_inv`, each level's inverse generator, and the challenges as limbs.
#[cfg(feature = "cuda")]
fn fold_scalars<F, N>(
    generator: &math::field::element::FieldElement<F>,
    alphas: &[math::field::element::FieldElement<N>],
) -> Option<(u64, Vec<u64>, Vec<u64>)>
where
    F: math::field::traits::IsField + 'static,
    N: math::field::traits::IsField + 'static,
{
    use math::field::element::FieldElement;
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;
    use math::field::goldilocks::GoldilocksField as Gl;
    use std::any::TypeId;

    if TypeId::of::<F>() != TypeId::of::<Gl>() || TypeId::of::<N>() != TypeId::of::<Ext3>() {
        return None;
    }
    if alphas.is_empty() {
        return None;
    }
    // SAFETY: `F == GoldilocksField`, a transparent wrapper over `u64`.
    let generator = unsafe { *(generator as *const _ as *const u64) };
    let generator = FieldElement::<Gl>::from_raw(generator);
    let two_inv = *FieldElement::<Gl>::from(2u64).inv().ok()?.value();
    let mut g_inv = generator.inv().ok()?;
    let mut g_invs = Vec::with_capacity(alphas.len());
    for _ in 0..alphas.len() {
        g_invs.push(*g_inv.value());
        g_inv = g_inv.square();
    }
    let mut raw = Vec::with_capacity(alphas.len() * 3);
    for alpha in alphas {
        raw.extend_from_slice(&ext3_raw(alpha)?);
    }
    Some((two_inv, g_invs, raw))
}

/// Limbs as field elements, one `u64` each for a base codeword and three for
/// an extension one.
#[cfg(feature = "cuda")]
fn values_from_raw<F>(raw: &[u64], base: bool) -> Option<Vec<math::field::element::FieldElement<F>>>
where
    F: math::field::traits::IsField + 'static,
{
    use math::field::element::FieldElement;
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;
    use math::field::goldilocks::GoldilocksField as Gl;
    use std::any::TypeId;

    if base {
        if TypeId::of::<F>() != TypeId::of::<Gl>() {
            return None;
        }
        return Some(
            raw.iter()
                .map(|limb| {
                    let value = FieldElement::<Gl>::from_raw(*limb);
                    // SAFETY: `F == Gl`, checked above; same representation.
                    unsafe {
                        core::mem::transmute_copy::<FieldElement<Gl>, FieldElement<F>>(&value)
                    }
                })
                .collect(),
        );
    }
    if TypeId::of::<F>() != TypeId::of::<Ext3>() {
        return None;
    }
    Some(raw.chunks_exact(3).map(ext3_from_raw::<F>).collect())
}

#[cfg(not(feature = "cuda"))]
impl DeviceCodeword {
    pub(crate) fn elements(&self) -> usize {
        match self.0 {}
    }

    pub(crate) fn fold<F, N>(
        &self,
        _generator: &math::field::element::FieldElement<F>,
        _alphas: &[math::field::element::FieldElement<N>],
    ) -> Option<Self>
    where
        F: math::field::traits::IsField + 'static,
        N: math::field::traits::IsField + 'static,
    {
        match self.0 {}
    }

    pub(crate) fn commit(
        &self,
        _log_folding: usize,
        _hash: crate::whir_hash::DeviceHashKey,
    ) -> Option<[u8; 32]> {
        match self.0 {}
    }

    pub(crate) fn paths(
        &self,
        _log_folding: usize,
        _indices: &[usize],
        _hash: crate::whir_hash::DeviceHashKey,
    ) -> Option<Vec<Vec<[u8; 32]>>> {
        match self.0 {}
    }

    pub(crate) fn paths_and_cap(
        &self,
        _log_folding: usize,
        _indices: &[usize],
        _cap_height: usize,
        _hash: crate::whir_hash::DeviceHashKey,
    ) -> Option<PathsAndCap> {
        match self.0 {}
    }

    pub(crate) fn cosets<F>(
        &self,
        _indices: &[usize],
        _num_leaves: usize,
        _block: usize,
    ) -> Option<Vec<Vec<math::field::element::FieldElement<F>>>>
    where
        F: math::field::traits::IsField + 'static,
    {
        match self.0 {}
    }

    pub(crate) fn first<F>(&self) -> Option<math::field::element::FieldElement<F>>
    where
        F: math::field::traits::IsField + 'static,
    {
        match self.0 {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::program::Builder;
    use math::field::element::FieldElement;
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;
    use math::field::goldilocks::GoldilocksField as Gl;

    type FE = FieldElement<Ext3>;

    /// The kernel's walk, in Rust: the same slot file, the same node encoding.
    ///
    /// This is what pins the lowering without a device — a slot freed too early
    /// or an operand read from the wrong class shows up here as a wrong value,
    /// not as a proof that does not verify an hour later.
    fn run_lowered(lowered: &Lowered, values: &[FE]) -> FE {
        let mut slots = vec![FE::zero(); lowered.num_slots];
        for node in lowered.nodes.chunks_exact(2) {
            let op = (node[0] & 0xFFFF_FFFF) as u32;
            let a = (node[0] >> 32) as u32 as usize;
            let b = (node[1] & 0xFFFF_FFFF) as u32 as usize;
            let res = (node[1] >> 32) as u32 as usize;
            slots[res] = match op {
                op::FIXED => ext3_from_raw::<Ext3>(&lowered.consts[a * 3..a * 3 + 3]),
                op::VAR => values[a],
                op::ADD => slots[a] + slots[b],
                op::SUB => slots[a] - slots[b],
                op::MUL => slots[a] * slots[b],
                op::NEG => -slots[a],
                _ => panic!("unknown op {op}"),
            };
        }
        slots[lowered.root_slot as usize]
    }

    fn values(n: usize) -> Vec<FE> {
        (0..n as u64)
            .map(|i| {
                FE::new([
                    FieldElement::<Gl>::from(i * 31 + 7),
                    FieldElement::<Gl>::from(i * 17 + 2),
                    FieldElement::<Gl>::from(i + 5),
                ])
            })
            .collect()
    }

    /// Every op, a constant, and a chain long enough that slots have to be
    /// recycled.
    fn sample_program() -> crate::program::Program<Ext3> {
        let mut b = Builder::<Ext3>::new();
        let mut acc = b.var(0);
        for slot in 1..6 {
            let v = b.var(slot);
            let doubled = b.add(v, v);
            let scaled = b.mul(doubled, acc);
            let shifted = b.sub(scaled, v);
            acc = b.neg(shifted);
        }
        let seven = b.fixed(FE::from(7u64));
        let root = b.add(acc, seven);
        b.finish(root).unwrap()
    }

    /// `LAMBDA_VM_ARGUE_DEVICE_COLUMNS`, `LAMBDA_VM_ARGUE_DEVICE_TABLES` and
    /// `LAMBDA_VM_ARGUE_LEAN_PROGRAM` are on by default: unset, empty or any value
    /// reads on, and only `0` — the opt-out — reads off.
    #[test]
    fn only_zero_turns_a_default_on_knob_off() {
        assert!(not_off(None), "unset is the default, on");
        assert!(not_off(Some("")), "empty is not the opt-out");
        assert!(not_off(Some("1")));
        assert!(!not_off(Some("0")), "`0` is the opt-out");
    }

    /// The GKR input knob reads its variable the default-on way, whatever the
    /// environment sets. ⚠ No test in this binary forces the knob, which is
    /// what makes the reading the environment's.
    #[test]
    fn the_gkr_input_knob_is_on_unless_its_variable_is_zero() {
        let variable = std::env::var("LAMBDA_VM_ARGUE_GKR_INPUT").ok();
        assert_eq!(argue_gkr_input(), not_off(variable.as_deref()));
    }

    /// No lift reads its variable the default-on way, and needs the input
    /// from the columns.
    #[test]
    fn the_no_lift_knob_is_on_unless_either_variable_is_zero() {
        let input = std::env::var("LAMBDA_VM_ARGUE_GKR_INPUT").ok();
        let variable = std::env::var("LAMBDA_VM_ARGUE_NO_LIFT").ok();
        assert_eq!(
            argue_no_lift(),
            not_off(variable.as_deref()) && not_off(input.as_deref())
        );
    }

    /// The tables knob reads its variable the default-on way, whatever the
    /// environment sets. ⚠ No test in this binary forces the knob, which is
    /// what makes the reading the environment's.
    #[test]
    fn the_tables_knob_is_on_unless_its_variable_is_zero() {
        let variable = std::env::var("LAMBDA_VM_ARGUE_DEVICE_TABLES").ok();
        assert_eq!(argue_device_tables(), not_off(variable.as_deref()));
    }

    /// The lean program's knob reads its variable the default-on way, whatever
    /// the environment sets. ⚠ No test in this binary forces the knob, which is
    /// what makes the reading the environment's.
    #[test]
    fn the_lean_program_knob_is_on_unless_its_variable_is_zero() {
        let variable = std::env::var("LAMBDA_VM_ARGUE_LEAN_PROGRAM").ok();
        assert_eq!(argue_lean_program(), not_off(variable.as_deref()));
    }

    #[test]
    fn the_lowered_program_computes_what_the_program_does() {
        let program = sample_program();
        let lowered = lower(&program).expect("lowers");
        let v = values(6);
        let mut scratch = Vec::new();
        assert_eq!(run_lowered(&lowered, &v), program.eval(&v, &mut scratch));
    }

    /// A value read twice in the step that kills it — `x·x` — must not free its
    /// slot twice, or two later values are handed the same one and the second
    /// clobbers the first. Squarings are everywhere in a real constraint
    /// program, so this is the shape that matters.
    #[test]
    fn a_value_read_twice_frees_its_slot_once() {
        let mut b = Builder::<Ext3>::new();
        let mut acc = b.var(0);
        // Each square kills its operand, and the sums below keep enough values
        // live that a doubly-freed slot gets reused while it is still needed.
        let mut squares = Vec::new();
        for slot in 1..8 {
            let v = b.var(slot);
            let squared = b.mul(v, v);
            let with_acc = b.add(squared, acc);
            squares.push(with_acc);
            acc = b.mul(with_acc, with_acc);
        }
        squares.push(acc);
        let root = b.sum(&squares);
        let program = b.finish(root).unwrap();

        let lowered = lower(&program).expect("lowers");
        let v = values(8);
        let mut scratch = Vec::new();
        assert_eq!(run_lowered(&lowered, &v), program.eval(&v, &mut scratch));
    }

    /// The point of the slot file: a long chain of dead intermediates does not
    /// widen it.
    #[test]
    fn slots_are_reused_once_a_value_is_dead() {
        let lowered = lower(&sample_program()).expect("lowers");
        assert!(
            lowered.num_slots < lowered.nodes.len() / 2,
            "{} slots for {} steps is no reuse at all",
            lowered.num_slots,
            lowered.nodes.len() / 2
        );
    }

    /// A program wider than the slot file declines rather than asking a device
    /// for scratch it cannot have.
    #[test]
    fn a_program_past_the_slot_ceiling_declines() {
        let mut b = Builder::<Ext3>::new();
        // Every value stays live to the end, so the slots cannot be recycled.
        let terms: Vec<u32> = (0..=MAX_SLOTS).map(|slot| b.var(slot)).collect();
        let root = b.sum(&terms);
        let program = b.finish(root).unwrap();
        assert!(lower(&program).is_none());
    }

    /// The lowering of a program on demand — a read a step, the same values
    /// in fewer slots — walks to the program's value, as the kernel walks it.
    #[test]
    fn a_lowered_program_on_demand_computes_what_the_program_does() {
        let mut b = Builder::<Ext3>::new();
        let mut acc = b.var(0);
        let mut squares = Vec::new();
        for slot in 1..8 {
            let v = b.var(slot);
            let squared = b.mul(v, v);
            let with_acc = b.add(squared, acc);
            squares.push(with_acc);
            acc = b.mul(with_acc, with_acc);
        }
        squares.push(acc);
        let root = b.sum(&squares);
        let v = values(8);
        let mut scratch = Vec::new();
        for program in [sample_program(), b.finish(root).unwrap()] {
            let demand = lower(&program.on_demand()).expect("lowers");
            assert_eq!(run_lowered(&demand, &v), program.eval(&v, &mut scratch));
        }
    }

    /// A batch is big past [`LEAN_ABOVE_SLOTS`] values a thread, unless a test
    /// sets its own gate.
    #[test]
    fn the_big_batch_gate_reads_its_override() {
        assert!(!is_big_batch(LEAN_ABOVE_SLOTS));
        assert!(is_big_batch(LEAN_ABOVE_SLOTS + 1));
        force_lean_program_gate(Some(0));
        let forced = (is_big_batch(0), is_big_batch(1));
        force_lean_program_gate(None);
        assert_eq!(forced, (false, true), "a gate of 0 makes every program big");
        assert!(!is_big_batch(LEAN_ABOVE_SLOTS), "and `None` restores it");
    }

    /// ★ The gate is where the slot budget leaves a round fewer than 64 k
    /// threads: 341 values a thread still run 65,600, 342 run 65,408.
    #[cfg(feature = "cuda")]
    #[test]
    fn the_big_batch_gate_is_where_a_round_drops_below_64k_threads() {
        use math_cuda::sumcheck::thread_ceiling;
        assert!(thread_ceiling(LEAN_ABOVE_SLOTS) >= 1 << 16);
        assert!(thread_ceiling(LEAN_ABOVE_SLOTS + 1) < 1 << 16);
    }

    #[test]
    fn a_field_the_kernel_does_not_cover_declines() {
        let mut b = Builder::<Gl>::new();
        let root = b.var(0);
        let program = b.finish(root).unwrap();
        assert!(lower(&program).is_none());
    }

    /// ⛔ THE GKR TREE'S REFUSAL IS COUNTED AT ITS ONE SITE, and only there —
    /// math-cuda's census of its own five, extended to this file. The argue
    /// surface's device-fallback counter is bumped here once, inside
    /// `note_gkr_tree_refusal`, which the whole-tree promise's refusal calls once.
    /// The carry's refusal (speculative) and the prefetch's (no prefetch) must
    /// not reach it: removing the call, or adding one, reddens this test by name.
    /// The box test (`stark::multilinear_table`'s
    /// `a_refused_gkr_tree_is_counted_and_the_host_builds_the_same_tree`)
    /// drives the site.
    #[test]
    fn the_gkr_tree_refusal_is_counted_at_exactly_its_one_site() {
        let source = include_str!("gpu.rs");
        let bumps = source
            .matches(concat!("math_cuda::device::", "note_device_fallback()"))
            .count();
        let notes = source
            .matches(concat!("note_gkr_tree_refusal(", "promise, full)"))
            .count();
        assert_eq!(
            (bumps, notes),
            (1, 1),
            "the device-fallback counter must be bumped once in this file, in \
             `note_gkr_tree_refusal`, and that called once, at the whole-tree \
             promise's refusal (found {bumps} bumps, {notes} calls)"
        );
    }
}
