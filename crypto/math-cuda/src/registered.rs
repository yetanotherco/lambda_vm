//! Host buffers page-locked where they lie, for the uploads that read them.
//!
//! [`crate::device::htod_staged`] copies a pageable source into a pinned
//! staging buffer and DMAs that, 32 MiB at a time. When the source itself is
//! page-locked, the staging copy is pure cost: it doubles the host memory
//! traffic of the upload and paces it at one thread's memcpy. [`register`]
//! page-locks an existing buffer in place (`cuMemHostRegister`), a piece of
//! [`PIECE_BYTES`] at a time, and records the pieces; an upload whose source
//! lies in recorded pieces is then sent as direct DMAs ([`direct_plan`]).
//!
//! The bytes that reach the card do not change, only the path they take. With
//! nothing registered (the default), [`any`] is false and every upload takes
//! the staged path exactly as before.
//!
//! The caller owns the buffer and must keep it alive and unmoved until
//! [`unregister`] has run on what [`register`] returned; it must not register
//! a buffer twice. Registering takes the driver's lock: every other thread's
//! driver calls wait while a piece is locked, so a caller does it off the
//! critical path, one piece at a time.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use cudarc::driver::sys;

/// The size of one registered piece. Each piece is its own registration, so
/// no single driver call holds the lock for long.
pub const PIECE_BYTES: usize = 64 << 20;

/// The host page size assumed for the pieces' bounds. A registration is
/// rounded to whole pages; the head and tail of a buffer short of a page
/// stay pageable.
pub const PAGE_BYTES: usize = 4096;

/// The pieces recorded: `(start, len)`, page-aligned, disjoint.
static PIECES: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());
/// How many pieces are recorded: the uploads' fast check, no lock taken.
static COUNT: AtomicUsize = AtomicUsize::new(0);

static REGISTERED_BYTES: AtomicU64 = AtomicU64::new(0);
static REGISTER_NANOS: AtomicU64 = AtomicU64::new(0);
static REFUSED_PIECES: AtomicU64 = AtomicU64::new(0);
static UNREGISTER_NANOS: AtomicU64 = AtomicU64::new(0);
static DIRECT_BYTES: AtomicU64 = AtomicU64::new(0);
static DIRECT_CALLS: AtomicU64 = AtomicU64::new(0);

/// What one [`register`] call locked; hand it to [`unregister`].
#[derive(Debug, Default)]
#[must_use = "a registration must be undone with `unregister` before its buffer is freed"]
pub struct Registered {
    pieces: Vec<(usize, usize)>,
    /// Bytes locked, and pieces the driver refused (left pageable).
    pub bytes: u64,
    pub refused: usize,
    /// Wall seconds of the driver calls.
    pub seconds: f64,
}

/// The counters since the process started (or since [`reset_stats`]).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Stats {
    pub registered_bytes: u64,
    pub register_seconds: f64,
    pub refused_pieces: u64,
    pub unregister_seconds: f64,
    /// Bytes uploaded as direct DMAs from registered pieces, and the uploads.
    pub direct_bytes: u64,
    pub direct_calls: u64,
    /// Pieces recorded now.
    pub pieces_now: usize,
}

pub fn stats() -> Stats {
    let s = |a: &AtomicU64| a.load(Ordering::Relaxed);
    Stats {
        registered_bytes: s(&REGISTERED_BYTES),
        register_seconds: s(&REGISTER_NANOS) as f64 / 1e9,
        refused_pieces: s(&REFUSED_PIECES),
        unregister_seconds: s(&UNREGISTER_NANOS) as f64 / 1e9,
        direct_bytes: s(&DIRECT_BYTES),
        direct_calls: s(&DIRECT_CALLS),
        pieces_now: COUNT.load(Ordering::Acquire),
    }
}

/// Zeroes the cumulative counters (not the recorded pieces).
pub fn reset_stats() {
    for a in [
        &REGISTERED_BYTES,
        &REGISTER_NANOS,
        &REFUSED_PIECES,
        &UNREGISTER_NANOS,
        &DIRECT_BYTES,
        &DIRECT_CALLS,
    ] {
        a.store(0, Ordering::Relaxed);
    }
}

/// Whether any piece is recorded.
pub fn any() -> bool {
    COUNT.load(Ordering::Acquire) > 0
}

/// The page-aligned pieces inside `[ptr, ptr + len)`: each at most
/// [`PIECE_BYTES`], together the buffer less its head and tail short of a page.
pub fn pieces_of(ptr: usize, len: usize) -> Vec<(usize, usize)> {
    let start = ptr.next_multiple_of(PAGE_BYTES);
    let end = (ptr + len) / PAGE_BYTES * PAGE_BYTES;
    let mut out = Vec::new();
    let mut at = start;
    while at < end {
        let n = PIECE_BYTES.min(end - at);
        out.push((at, n));
        at += n;
    }
    out
}

fn driver_lock(at: usize, len: usize) -> bool {
    let Ok(be) = crate::device::backend() else {
        return false;
    };
    if be.ctx.bind_to_thread().is_err() {
        return false;
    }
    // SAFETY: `at..at + len` lies inside a live buffer the caller keeps alive
    // and unmoved until it is unregistered (the module's contract).
    unsafe {
        sys::cuMemHostRegister_v2(at as *mut _, len, sys::CU_MEMHOSTREGISTER_PORTABLE)
            == sys::CUresult::CUDA_SUCCESS
    }
}

fn driver_unlock(at: usize) {
    if let Ok(be) = crate::device::backend() {
        let _ = be.ctx.bind_to_thread();
    }
    // SAFETY: `at` is the start of a piece this module registered.
    let _ = unsafe { sys::cuMemHostUnregister(at as *mut _) };
}

/// Page-locks `[ptr, ptr + len)` in place, a piece at a time, and records the
/// pieces the driver locked. A piece it refuses stays pageable (uploads from
/// it take the staged path). Without a card nothing is locked.
pub fn register(ptr: *const u8, len: usize) -> Registered {
    register_with(ptr as usize, len, driver_lock)
}

/// `lock` page-locks one piece (`false`: refused); the driver's in production.
fn register_with(ptr: usize, len: usize, lock: impl Fn(usize, usize) -> bool) -> Registered {
    let t = Instant::now();
    let mut out = Registered::default();
    for (at, n) in pieces_of(ptr, len) {
        if lock(at, n) {
            out.pieces.push((at, n));
            out.bytes += n as u64;
        } else {
            out.refused += 1;
        }
    }
    if !out.pieces.is_empty() {
        let mut all = PIECES.lock().unwrap_or_else(|e| e.into_inner());
        all.extend_from_slice(&out.pieces);
        all.sort_unstable();
        COUNT.store(all.len(), Ordering::Release);
    }
    out.seconds = t.elapsed().as_secs_f64();
    REGISTERED_BYTES.fetch_add(out.bytes, Ordering::Relaxed);
    REGISTER_NANOS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    REFUSED_PIECES.fetch_add(out.refused as u64, Ordering::Relaxed);
    out
}

/// Undoes a [`register`]: the pieces leave the record first (no upload plans
/// a direct DMA from them after this starts), then the driver releases them.
/// Call it only once no upload from the buffer is in flight.
pub fn unregister(r: Registered) {
    unregister_with(r, driver_unlock)
}

/// `unlock` releases one piece; the driver's in production.
fn unregister_with(r: Registered, unlock: impl Fn(usize)) {
    if r.pieces.is_empty() {
        return;
    }
    let t = Instant::now();
    {
        let mut all = PIECES.lock().unwrap_or_else(|e| e.into_inner());
        all.retain(|p| r.pieces.binary_search(p).is_err());
        COUNT.store(all.len(), Ordering::Release);
    }
    for &(at, _) in &r.pieces {
        unlock(at);
    }
    UNREGISTER_NANOS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
}

/// One segment of a direct upload: `len` bytes at offset `off` of the source,
/// from a registered piece (`locked`) or a pageable head or tail.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Segment {
    pub off: usize,
    pub len: usize,
    pub locked: bool,
}

/// How to send `[ptr, ptr + len)` as direct copies: every byte in a
/// registered piece, except a head and a tail shorter than a page. `None`
/// when any other byte is pageable (the staged path then takes it whole).
pub fn direct_plan(ptr: *const u8, len: usize) -> Option<Vec<Segment>> {
    if len == 0 || !any() {
        return None;
    }
    let all = PIECES.lock().unwrap_or_else(|e| e.into_inner());
    plan_over(&all, ptr as usize, len)
}

fn plan_over(pieces: &[(usize, usize)], ptr: usize, len: usize) -> Option<Vec<Segment>> {
    let end = ptr + len;
    // The first piece that ends after `ptr`.
    let mut i = pieces.partition_point(|&(s, n)| s + n <= ptr);
    let mut out = Vec::new();
    let mut at = ptr;
    while at < end {
        match pieces.get(i) {
            Some(&(s, n)) if s <= at => {
                let to = (s + n).min(end);
                out.push(Segment {
                    off: at - ptr,
                    len: to - at,
                    locked: true,
                });
                at = to;
                i += 1;
            }
            next => {
                // Pageable bytes: allowed only as the head or the tail, each
                // short of a page.
                let to = next.map_or(end, |&(s, _)| s.min(end));
                let head = at == ptr && to < end;
                let tail = to == end && at > ptr;
                if (to - at) >= PAGE_BYTES || !(head || tail) {
                    return None;
                }
                out.push(Segment {
                    off: at - ptr,
                    len: to - at,
                    locked: false,
                });
                at = to;
            }
        }
    }
    out.iter().any(|s| s.locked).then_some(out)
}

/// Counts one direct upload of `bytes`.
pub(crate) fn note_direct(bytes: usize) {
    DIRECT_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
    DIRECT_CALLS.fetch_add(1, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: usize = PAGE_BYTES;

    #[test]
    fn the_pieces_cover_the_pages_inside_the_buffer_at_most_a_piece_each() {
        let ptr = 10 * P + 17;
        let len = PIECE_BYTES + 5 * P;
        let pieces = pieces_of(ptr, len);
        assert_eq!(pieces[0].0, 11 * P, "the head short of a page stays out");
        let end = (ptr + len) / P * P;
        assert_eq!(pieces.last().map(|&(s, n)| s + n), Some(end));
        assert!(
            pieces
                .iter()
                .all(|&(s, n)| s % P == 0 && n <= PIECE_BYTES && n > 0)
        );
        assert!(
            pieces.windows(2).all(|w| w[0].0 + w[0].1 == w[1].0),
            "contiguous"
        );
        assert!(pieces_of(ptr, P / 2).is_empty(), "no whole page, no piece");
    }

    #[test]
    fn a_plan_covers_every_byte_once_in_order() {
        let ptr = 3 * P + 100;
        let len = 2 * PIECE_BYTES + 3 * P;
        let pieces = pieces_of(ptr, len);
        let plan = plan_over(&pieces, ptr, len).expect("registered but for head and tail");
        let mut at = 0;
        for s in &plan {
            assert_eq!(s.off, at);
            at += s.len;
        }
        assert_eq!(at, len);
        assert!(!plan[0].locked && plan[0].len == P - 100, "the head");
        assert!(plan.last().is_some_and(|s| !s.locked), "the tail");
        assert!(plan[1..plan.len() - 1].iter().all(|s| s.locked));
    }

    #[test]
    fn a_pageable_gap_inside_the_source_refuses_the_plan() {
        let ptr = 8 * P;
        let len = 3 * PIECE_BYTES;
        let mut pieces = pieces_of(ptr, len);
        assert!(plan_over(&pieces, ptr, len).is_some());
        pieces.remove(1);
        assert_eq!(plan_over(&pieces, ptr, len), None, "a hole in the middle");
        let pieces = pieces_of(ptr, len);
        assert_eq!(
            plan_over(&pieces[1..], ptr, len),
            None,
            "a whole piece at the head"
        );
        assert_eq!(plan_over(&[], ptr, len), None, "nothing registered");
    }

    #[test]
    fn a_source_inside_one_piece_is_one_locked_segment() {
        let pieces = pieces_of(16 * P, PIECE_BYTES);
        let plan = plan_over(&pieces, 16 * P + 40, 1000).expect("inside");
        assert_eq!(
            plan,
            vec![Segment {
                off: 0,
                len: 1000,
                locked: true
            }]
        );
    }

    /// Registering records only what the driver locked and refuses nothing it
    /// did not see; unregistering takes the pieces out of the record before it
    /// releases them, and only its own.
    #[test]
    fn register_records_the_locked_pieces_and_unregister_removes_only_its_own() {
        // A region of addresses no real buffer of this test process uses:
        // nothing is dereferenced, the stand-in driver only counts.
        let a = 1usize << 46;
        let b = a + (1 << 40);
        let refuse_second = |at: usize, _n: usize| at != a + PIECE_BYTES;
        let ra = register_with(a, 3 * PIECE_BYTES, refuse_second);
        assert_eq!((ra.pieces.len(), ra.refused), (2, 1));
        assert_eq!(ra.bytes, 2 * PIECE_BYTES as u64);
        let rb = register_with(b, PIECE_BYTES, |_, _| true);
        {
            let all = PIECES.lock().unwrap();
            assert!(plan_over(&all, a, PIECE_BYTES).is_some());
            assert_eq!(
                plan_over(&all, a, 2 * PIECE_BYTES),
                None,
                "the refused piece is pageable"
            );
            assert!(plan_over(&all, b, PIECE_BYTES).is_some());
        }
        static UNLOCKED: AtomicUsize = AtomicUsize::new(0);
        unregister_with(ra, |_| {
            UNLOCKED.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(UNLOCKED.load(Ordering::Relaxed), 2);
        {
            let all = PIECES.lock().unwrap();
            assert_eq!(plan_over(&all, a, PIECE_BYTES), None, "a's pieces are gone");
            assert!(plan_over(&all, b, PIECE_BYTES).is_some(), "b's stay");
        }
        unregister_with(rb, |_| {});
        assert!(
            !PIECES
                .lock()
                .unwrap()
                .iter()
                .any(|&(s, _)| s >= a && s < b + PIECE_BYTES)
        );
    }
}
