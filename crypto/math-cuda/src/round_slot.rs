//! A round's sums without the second launch and the pageable read-back
//! (`LAMBDA_VM_ARGUE_ROUND_SLOT`): the round kernel's last block reduces them
//! (`slot_sums` in `kernels/sumcheck.cu`) into a slot of pinned host memory the
//! card writes through a mapping, and the host spins on the slot's sequence
//! word instead of copying and waiting.
//!
//! The sums are the limbs today's `sum_partials_ext3` writes, and the caller
//! reduces them by `ext3_from_raw` as it does today, so no value the host sees
//! moves. `Check` runs today's second launch and read-back as well, compares
//! the two limb for limb, counts a difference and uses today's.
//!
//! A slot is 128 u64: the sums at `[0, 96)` (32 rows of three), then the
//! sequence word at [`SEQ`]; the words after it are unused.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use cudarc::driver::{CudaSlice, CudaStream};

use crate::Result;

/// Words a slot spans.
pub const SLOT_WORDS: usize = 128;
/// Where the last block writes the launch's sequence number, after the sums.
pub const SEQ: usize = 96;
/// Rows a slot holds.
pub const MAX_ROWS: usize = 32;

/// How a session's round sums come back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoundSlot {
    /// Today's second launch and pageable read-back.
    Off,
    /// The last block's reduction, through a slot.
    On,
    /// Both, compared every round; today's are used.
    Check,
}

/// Shared memory a `_slot` kernel's block asks for: its own reduction's, or
/// the last block's 256 lanes when that is more, and the last block's flag.
pub fn shared_bytes(block: u32) -> u32 {
    block.max(256) * 24 + 16
}

static SLOT_ROUNDS: AtomicU64 = AtomicU64::new(0);
static FALL_BACKS: AtomicU64 = AtomicU64::new(0);
static DECLINES: AtomicU64 = AtomicU64::new(0);
static SECOND_LAUNCHES: AtomicU64 = AtomicU64::new(0);
static CHECKED: AtomicU64 = AtomicU64::new(0);
static MISMATCHES: AtomicU64 = AtomicU64::new(0);

/// What the slots did this process.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    /// Rounds whose sums were read from a slot.
    pub slot_rounds: u64,
    /// Of those, the ones whose spin ran out and that waited on the stream.
    pub fall_backs: u64,
    /// Sessions that asked for a slot and got none (today's path instead).
    pub declines: u64,
    /// Today's second launches (`sum_partials_ext3`) by the GKR layers' and
    /// the fused zerocheck's rounds.
    pub second_launches: u64,
    /// Rounds `Check` compared.
    pub checked: u64,
    /// Rounds whose sums `Check` found differing.
    pub mismatches: u64,
}

pub fn counts() -> Counts {
    Counts {
        slot_rounds: SLOT_ROUNDS.load(Ordering::Relaxed),
        fall_backs: FALL_BACKS.load(Ordering::Relaxed),
        declines: DECLINES.load(Ordering::Relaxed),
        second_launches: SECOND_LAUNCHES.load(Ordering::Relaxed),
        checked: CHECKED.load(Ordering::Relaxed),
        mismatches: MISMATCHES.load(Ordering::Relaxed),
    }
}

pub fn reset_counts() {
    for c in [
        &SLOT_ROUNDS,
        &FALL_BACKS,
        &DECLINES,
        &SECOND_LAUNCHES,
        &CHECKED,
        &MISMATCHES,
    ] {
        c.store(0, Ordering::Relaxed);
    }
}

/// Called where a round launches today's second launch.
pub(crate) fn note_second_launch() {
    SECOND_LAUNCHES.fetch_add(1, Ordering::Relaxed);
}

/// The spin before yielding, and the patience before waiting on the stream.
const SPIN: Duration = Duration::from_micros(50);
const PATIENCE: Duration = Duration::from_millis(5);

/// Waits until `words[SEQ]` is `seq`: a spin for `spin`, then yields for
/// `patience`, then `fall_back` (the stream's synchronize) and one more look.
/// Returns the first `3·rows` words, read after the sequence word with an
/// acquire fence, and whether the wait fell back. A word that is still not
/// `seq` after the fall-back is an error: the launch never wrote its slot.
///
/// # Safety
/// `words` points to [`SLOT_WORDS`] u64 that stay valid for the call; another
/// agent (the card, or a test's thread) may write them meanwhile.
pub(crate) unsafe fn wait_on(
    words: *const u64,
    seq: u64,
    rows: usize,
    spin: Duration,
    patience: Duration,
    fall_back: impl FnOnce() -> Result<()>,
) -> Result<(Vec<u64>, bool)> {
    if rows > MAX_ROWS {
        return Err(invalid());
    }
    let arrived = || unsafe { std::ptr::read_volatile(words.add(SEQ)) } == seq;
    let started = Instant::now();
    let mut fell_back = false;
    let mut seen = arrived();
    while !seen && started.elapsed() < spin {
        for _ in 0..64 {
            std::hint::spin_loop();
        }
        seen = arrived();
    }
    while !seen && started.elapsed() < spin + patience {
        std::thread::yield_now();
        seen = arrived();
    }
    if !seen {
        fall_back()?;
        fell_back = true;
        if !arrived() {
            return Err(invalid());
        }
    }
    std::sync::atomic::fence(Ordering::Acquire);
    let sums = (0..3 * rows)
        .map(|i| unsafe { std::ptr::read_volatile(words.add(i)) })
        .collect();
    Ok((sums, fell_back))
}

fn invalid() -> cudarc::driver::DriverError {
    cudarc::driver::DriverError(cudarc::driver::sys::CUresult::CUDA_ERROR_INVALID_VALUE)
}

/// Sequence numbers, one a launch, never repeated in a process: a slot taken
/// again cannot mistake an earlier owner's word for its own launch's.
static NEXT_SEQ: AtomicU64 = AtomicU64::new(1);

fn next_seq() -> u64 {
    NEXT_SEQ.fetch_add(1, Ordering::Relaxed)
}

/// A slot's host and device addresses.
#[derive(Clone, Copy)]
struct Place {
    host: usize,
    dev: u64,
}

/// Slots no session holds, from 64 KiB pages of pinned, card-mapped memory
/// that are never given back (at most [`MAX_PAGES`]).
static FREE: Mutex<Vec<Place>> = Mutex::new(Vec::new());
static PAGES: AtomicU64 = AtomicU64::new(0);
const PAGE_BYTES: usize = 64 * 1024;
const MAX_PAGES: u64 = 16;

fn new_page() -> Option<Vec<Place>> {
    use cudarc::driver::sys;
    if PAGES.fetch_add(1, Ordering::Relaxed) >= MAX_PAGES {
        PAGES.fetch_sub(1, Ordering::Relaxed);
        return None;
    }
    let be = crate::device::backend().ok()?;
    be.ctx.bind_to_thread().ok()?;
    // SAFETY: a fresh allocation of `PAGE_BYTES`, mapped into the card's
    // address space; never freed, so the addresses below stay valid.
    unsafe {
        let host = cudarc::driver::result::malloc_host(
            PAGE_BYTES,
            sys::CU_MEMHOSTALLOC_DEVICEMAP | sys::CU_MEMHOSTALLOC_PORTABLE,
        )
        .ok()? as *mut u64;
        let mut dev: sys::CUdeviceptr = 0;
        if sys::cuMemHostGetDevicePointer_v2(&mut dev, host as *mut _, 0)
            .result()
            .is_err()
        {
            let _ = sys::cuMemFreeHost(host as *mut _);
            PAGES.fetch_sub(1, Ordering::Relaxed);
            return None;
        }
        std::ptr::write_bytes(host, 0, PAGE_BYTES / 8);
        let stride = SLOT_WORDS * 8;
        Some(
            (0..PAGE_BYTES / stride)
                .map(|i| Place {
                    host: host as usize + i * stride,
                    dev: dev + (i * stride) as u64,
                })
                .collect(),
        )
    }
}

/// One session's slot, given back to the pool when the session drops — after
/// its stream is idle, so no launch of this owner can still write it.
pub struct MappedSlot {
    place: Place,
    stream: Arc<CudaStream>,
}

// SAFETY: the slot's memory is only touched through this handle (and by the
// card, ordered by the stream it was launched on), and it is not shared.
unsafe impl Send for MappedSlot {}
unsafe impl Sync for MappedSlot {}

impl MappedSlot {
    /// A slot for launches on `stream`, or `None` when the pool is spent or
    /// the card will not map host memory.
    pub fn take(stream: &Arc<CudaStream>) -> Option<Self> {
        let place = {
            let mut free = FREE.lock().ok()?;
            match free.pop() {
                Some(place) => place,
                None => {
                    let mut page = new_page()?;
                    let place = page.pop()?;
                    free.extend(page);
                    place
                }
            }
        };
        // SAFETY: the slot is this handle's; no launch of an earlier owner is
        // in flight (that owner's stream was synchronized before it gave the
        // slot back).
        unsafe { std::ptr::write_bytes(place.host as *mut u64, 0, SLOT_WORDS) };
        Some(Self {
            place,
            stream: stream.clone(),
        })
    }

    /// The slot's address on the card.
    pub fn dev(&self) -> u64 {
        self.place.dev
    }

    fn words(&self) -> *mut u64 {
        self.place.host as *mut u64
    }

    /// Waits for the launch numbered `seq` and returns its `rows` sums.
    pub fn wait(&self, seq: u64, rows: usize) -> Result<Vec<u64>> {
        let stream = self.stream.clone();
        // SAFETY: the slot's words are valid while `self` is.
        let (sums, fell_back) = unsafe {
            wait_on(self.words(), seq, rows, SPIN, PATIENCE, || {
                stream.synchronize()
            })
        }?;
        SLOT_ROUNDS.fetch_add(1, Ordering::Relaxed);
        if fell_back {
            FALL_BACKS.fetch_add(1, Ordering::Relaxed);
        }
        Ok(sums)
    }

    /// The slot's host words, for a test that writes a flag beside the sums
    /// (the capability probe). Not for production callers.
    #[doc(hidden)]
    pub fn host_words_for_tests(&self) -> *mut u64 {
        self.words()
    }
}

impl Drop for MappedSlot {
    fn drop(&mut self) {
        // Every launch that writes this slot is on this stream.
        let _ = self.stream.synchronize();
        if let Ok(mut free) = FREE.lock() {
            free.push(self.place);
        }
    }
}

/// A session's rounds through a slot: the slot, the ticket its launches'
/// blocks take, and whether today's second launch runs beside them.
pub struct SlotRounds {
    slot: MappedSlot,
    ticket: CudaSlice<u32>,
    check: bool,
    seq: u64,
}

impl SlotRounds {
    /// For `mode` on `stream`; `None` for `Off`, and — counted — when no slot
    /// or ticket can be had, so the session runs today's rounds.
    pub fn open(stream: &Arc<CudaStream>, mode: RoundSlot) -> Option<Self> {
        if mode == RoundSlot::Off {
            return None;
        }
        let made = MappedSlot::take(stream).and_then(|slot| {
            let ticket = crate::device::alloc_zeros_or_trim::<u32>(stream, 1).ok()?;
            Some(Self {
                slot,
                ticket,
                check: mode == RoundSlot::Check,
                seq: 0,
            })
        });
        if made.is_none() {
            DECLINES.fetch_add(1, Ordering::Relaxed);
        }
        made
    }

    /// Whether today's second launch runs beside the slot's.
    pub fn checking(&self) -> bool {
        self.check
    }

    /// The next launch's arguments: the ticket, the slot's card address and
    /// its sequence number.
    pub fn launch(&mut self) -> (&mut CudaSlice<u32>, u64, u64) {
        self.seq = next_seq();
        (&mut self.ticket, self.slot.dev(), self.seq)
    }

    /// The last launch's `rows` sums. Under `Check`, `today` reads today's
    /// second launch's; the two are compared limb for limb and today's are
    /// returned.
    pub fn collect(
        &self,
        rows: usize,
        today: impl FnOnce() -> Result<Vec<u64>>,
    ) -> Result<Vec<u64>> {
        let sums = self.slot.wait(self.seq, rows)?;
        if !self.check {
            return Ok(sums);
        }
        let want = today()?;
        CHECKED.fetch_add(1, Ordering::Relaxed);
        if want != sums || fault_armed() {
            MISMATCHES.fetch_add(1, Ordering::Relaxed);
        }
        Ok(want)
    }
}

thread_local! {
    /// ⛔ A FAULT: while armed on a thread, every round `Check` compares counts
    /// as differing — so a test can show the count reaches the harness line.
    static FAULT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[doc(hidden)]
pub fn force_check_fault(on: bool) {
    FAULT.with(|cell| cell.set(on));
}

fn fault_armed() -> bool {
    FAULT.with(std::cell::Cell::get)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slot() -> Box<[u64; SLOT_WORDS]> {
        Box::new([0u64; SLOT_WORDS])
    }

    /// A word that arrives late is waited for, and the sums written before it
    /// are the ones read.
    #[test]
    fn a_late_sequence_word_is_waited_for() {
        let mut words = slot();
        let read = words.as_mut_ptr();
        let at = read as usize;
        let writer = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(1));
            let w = at as *mut u64;
            unsafe {
                for i in 0..6 {
                    std::ptr::write_volatile(w.add(i), 100 + i as u64);
                }
                std::sync::atomic::fence(Ordering::Release);
                std::ptr::write_volatile(w.add(SEQ), 7);
            }
        });
        let (sums, fell_back) = unsafe {
            wait_on(
                read as *const u64,
                7,
                2,
                Duration::from_micros(50),
                Duration::from_secs(5),
                || Ok(()),
            )
        }
        .unwrap();
        writer.join().unwrap();
        assert_eq!(sums, vec![100, 101, 102, 103, 104, 105]);
        assert!(!fell_back, "the word came inside the patience");
    }

    /// ⛔ Another launch's sequence word is not this one's: the wait falls
    /// back to the stream and, the word still wrong, fails.
    #[test]
    fn a_wrong_sequence_word_is_not_taken() {
        let mut words = slot();
        words[SEQ] = 6;
        let mut fell = false;
        let got = unsafe {
            wait_on(
                words.as_ptr(),
                7,
                1,
                Duration::from_micros(10),
                Duration::from_micros(10),
                || {
                    fell = true;
                    Ok(())
                },
            )
        };
        assert!(fell, "the wait fell back to the stream");
        assert!(got.is_err(), "a word that never arrives is an error");
    }

    /// The fall-back is what makes a slow launch's word arrive: the stream's
    /// synchronize returns after the card wrote it.
    #[test]
    fn the_fall_back_waits_for_the_launch() {
        let mut words = slot();
        let at = words.as_mut_ptr();
        let (sums, fell_back) = unsafe {
            wait_on(
                at as *const u64,
                9,
                1,
                Duration::from_micros(1),
                Duration::from_micros(1),
                || {
                    std::ptr::write_volatile(at.add(0), 1);
                    std::ptr::write_volatile(at.add(1), 2);
                    std::ptr::write_volatile(at.add(2), 3);
                    std::ptr::write_volatile(at.add(SEQ), 9);
                    Ok(())
                },
            )
        }
        .unwrap();
        assert!(fell_back);
        assert_eq!(sums, vec![1, 2, 3]);
    }

    /// ⛔ The fall-back's own error is the wait's.
    #[test]
    fn a_failed_stream_wait_is_the_waits_error() {
        let words = slot();
        let got = unsafe {
            wait_on(words.as_ptr(), 1, 1, Duration::ZERO, Duration::ZERO, || {
                Err(invalid())
            })
        };
        assert!(got.is_err());
    }

    /// ⛔ More rows than a slot holds is refused before anything is read.
    #[test]
    fn more_rows_than_a_slot_holds_is_refused() {
        let words = slot();
        let got = unsafe {
            wait_on(
                words.as_ptr(),
                0,
                MAX_ROWS + 1,
                Duration::ZERO,
                Duration::ZERO,
                || Ok(()),
            )
        };
        assert!(got.is_err());
    }

    /// The layout the kernel writes: the sums of 32 rows before the sequence
    /// word, inside the slot.
    #[test]
    fn the_slot_layout_holds_what_the_kernel_writes() {
        assert_eq!(MAX_ROWS * 3, SEQ, "the sums end where seq is");
        const { assert!(SEQ < SLOT_WORDS) };
        assert_eq!(shared_bytes(32), 256 * 24 + 16);
        assert_eq!(shared_bytes(256), 256 * 24 + 16);
        assert_eq!(shared_bytes(512), 512 * 24 + 16);
    }

    /// Sequence numbers are never repeated.
    #[test]
    fn sequence_numbers_grow() {
        let a = next_seq();
        let b = next_seq();
        assert!(b > a);
    }
}
