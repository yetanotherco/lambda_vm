//! Device-to-host read-backs that can stay off the driver's pageable path.
//!
//! A copy from the card into pageable host memory returns only once it has
//! completed, and the driver stages it through a buffer of its own. Two
//! threads reading back that way at once wait on each other inside the
//! driver: under phase B's overlap, the argument's round trips sat behind the
//! openings' read-backs (in a trace of a median block, 91 % of the argument
//! thread's lock waits overlapped another thread's pageable read-back).
//!
//! While [`pinned`] is on, a thread in the background class
//! ([`crate::stream_class`]) reads back through a pinned buffer: an
//! asynchronous copy, then a blocking-sync event the thread sleeps on, then a
//! host copy into the caller's memory. The bytes are the same either way.
//! Every other thread, and every thread while it is off (the default), reads
//! back the stream's own way. The buffers come from a pool: a thread takes
//! one at its first pinned read-back and gives it back when it ends, so the
//! short-lived threads phase B spawns do not pin memory of their own each.

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use cudarc::driver::{CudaStream, DevicePtr, DeviceRepr, DriverError};

use crate::stream_class::{self, StreamClass};

static PINNED: AtomicBool = AtomicBool::new(false);
static PINNED_COPIES: AtomicU64 = AtomicU64::new(0);
static PAGEABLE_COPIES: AtomicU64 = AtomicU64::new(0);
static PINNED_BYTES: AtomicU64 = AtomicU64::new(0);

/// Buffers threads gave back, for the next thread to take.
static POOL: Mutex<Vec<Staging>> = Mutex::new(Vec::new());

/// Buffers the pool keeps; past it a buffer given back is freed.
const POOL_KEEPS: usize = 16;

thread_local! {
    static STAGING: RefCell<Lent> = const { RefCell::new(Lent(None)) };
}

/// While it lives, background-class threads read back through pinned
/// staging. Taken by the thread that runs phase B.
pub struct PinnedReadbacks {
    previous: bool,
}

/// Turns pinned read-backs on until the guard drops.
pub fn pinned() -> PinnedReadbacks {
    PinnedReadbacks {
        previous: PINNED.swap(true, Ordering::AcqRel),
    }
}

impl Drop for PinnedReadbacks {
    fn drop(&mut self) {
        PINNED.store(self.previous, Ordering::Release);
    }
}

/// Background-class read-backs so far, `(through pinned staging, the
/// stream's own way)`, process wide.
pub fn background_copies() -> (u64, u64) {
    (
        PINNED_COPIES.load(Ordering::Relaxed),
        PAGEABLE_COPIES.load(Ordering::Relaxed),
    )
}

/// Host bytes the pinned read-back buffers hold now, lent or pooled
/// (page-locked: resident, and in no allocator's statistics).
pub fn pinned_bytes() -> u64 {
    PINNED_BYTES.load(Ordering::Relaxed)
}

/// Whether this thread's read-backs go through pinned staging; counts a
/// background-class read-back either way.
fn through_staging() -> bool {
    if stream_class::current() != StreamClass::Background {
        return false;
    }
    let pinned = PINNED.load(Ordering::Acquire);
    let counter = if pinned {
        &PINNED_COPIES
    } else {
        &PAGEABLE_COPIES
    };
    counter.fetch_add(1, Ordering::Relaxed);
    pinned
}

/// A pinned buffer and the blocking-sync event its copies record.
struct Staging {
    ptr: *mut u8,
    bytes: usize,
    event: cudarc::driver::sys::CUevent,
}

// SAFETY: the buffer and the event are used by one thread at a time: the
// thread that holds the `Staging`, or the pool's mutex.
unsafe impl Send for Staging {}

impl Staging {
    /// A buffer of at least `bytes`, from the pool or new, in the context bound
    /// to this thread.
    fn take(bytes: usize) -> Result<Staging, DriverError> {
        use cudarc::driver::sys;
        let pooled = POOL.lock().map(|mut pool| pool.pop()).unwrap_or(None);
        let mut staging = match pooled {
            Some(staging) => staging,
            None => {
                let mut event = std::ptr::null_mut();
                let flags = sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC as u32
                    | sys::CUevent_flags::CU_EVENT_DISABLE_TIMING as u32;
                // SAFETY: `event` is a valid out-pointer.
                unsafe { sys::cuEventCreate(&mut event, flags) }.result()?;
                Staging {
                    ptr: std::ptr::null_mut(),
                    bytes: 0,
                    event,
                }
            }
        };
        staging.grow(bytes)?;
        Ok(staging)
    }

    /// Room for `bytes`: the buffer replaced by a larger one if it is short.
    fn grow(&mut self, bytes: usize) -> Result<(), DriverError> {
        if self.bytes >= bytes {
            return Ok(());
        }
        self.free();
        let size = bytes.next_power_of_two().max(1 << 16);
        // SAFETY: a fresh host allocation the driver pins; freed by `free`.
        self.ptr = unsafe { cudarc::driver::result::malloc_host(size, 0) }? as *mut u8;
        self.bytes = size;
        PINNED_BYTES.fetch_add(size as u64, Ordering::Relaxed);
        Ok(())
    }

    /// Frees the buffer; every copy into it was waited for before its
    /// read-back returned.
    fn free(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: the buffer is idle and ours.
            unsafe {
                let _ = cudarc::driver::sys::cuMemFreeHost(self.ptr as *mut _);
            }
            PINNED_BYTES.fetch_sub(self.bytes as u64, Ordering::Relaxed);
            self.ptr = std::ptr::null_mut();
            self.bytes = 0;
        }
    }
}

impl Drop for Staging {
    fn drop(&mut self) {
        self.free();
        // SAFETY: the event is idle and ours.
        unsafe {
            let _ = cudarc::driver::sys::cuEventDestroy_v2(self.event);
        }
    }
}

/// The buffer a thread holds; given back to the pool when the thread ends.
struct Lent(Option<Staging>);

impl Drop for Lent {
    fn drop(&mut self) {
        if let Some(staging) = self.0.take()
            && let Ok(mut pool) = POOL.lock()
            && pool.len() < POOL_KEEPS
        {
            pool.push(staging);
        }
    }
}

/// Copies `src` into `dst` (at least `src.len()` elements) through this
/// thread's pinned buffer and waits for it on a blocking-sync event.
fn through_pinned<T: DeviceRepr, Src: DevicePtr<T>>(
    stream: &Arc<CudaStream>,
    src: &Src,
    dst: &mut [T],
) -> Result<(), DriverError> {
    let bytes = std::mem::size_of_val(&dst[..src.len()]);
    if bytes == 0 {
        return Ok(());
    }
    stream.context().bind_to_thread()?;
    STAGING.with(|cell| {
        let mut lent = cell.borrow_mut();
        let staging = match lent.0.take() {
            Some(mut staging) => {
                staging.grow(bytes)?;
                lent.0.insert(staging)
            }
            None => lent.0.insert(Staging::take(bytes)?),
        };
        {
            let (src_ptr, _record) = src.device_ptr(stream);
            // SAFETY: `staging` is pinned and holds `bytes`; the copy is
            // ordered after the stream's earlier work on `src`, and waited for
            // below before the buffer is read or reused.
            unsafe {
                cudarc::driver::sys::cuMemcpyDtoHAsync_v2(
                    staging.ptr as *mut _,
                    src_ptr,
                    bytes,
                    stream.cu_stream(),
                )
                .result()?;
                let recorded =
                    cudarc::driver::sys::cuEventRecord(staging.event, stream.cu_stream()).result();
                if let Err(e) = recorded {
                    let _ = stream.synchronize();
                    return Err(e);
                }
            }
        }
        // SAFETY: the event was recorded after the copy on its stream.
        unsafe { cudarc::driver::sys::cuEventSynchronize(staging.event) }.result()?;
        // SAFETY: the copy has completed; `dst` holds at least `bytes` and
        // does not overlap the pinned buffer.
        unsafe {
            std::ptr::copy_nonoverlapping(staging.ptr, dst.as_mut_ptr() as *mut u8, bytes);
        }
        Ok(())
    })
}

/// `stream.clone_dtoh(src)`, or through a pinned buffer on a background-class
/// thread while [`pinned`] is on.
pub fn clone_dtoh<T: DeviceRepr, Src: DevicePtr<T>>(
    stream: &Arc<CudaStream>,
    src: &Src,
) -> Result<Vec<T>, DriverError> {
    if !through_staging() {
        return stream.clone_dtoh(src);
    }
    let mut out = Vec::with_capacity(src.len());
    // SAFETY: every element is written by the copy before `out` is read; a
    // failed copy drops `out` without reading it (`T: DeviceRepr` is plain
    // data with no drop glue).
    #[allow(clippy::uninit_vec)]
    unsafe {
        out.set_len(src.len())
    };
    through_pinned(stream, src, &mut out)?;
    Ok(out)
}

/// `stream.memcpy_dtoh(src, dst)` into host memory, or through a pinned
/// buffer on a background-class thread while [`pinned`] is on.
pub fn memcpy_dtoh<T: DeviceRepr, Src: DevicePtr<T>>(
    stream: &Arc<CudaStream>,
    src: &Src,
    dst: &mut [T],
) -> Result<(), DriverError> {
    assert!(dst.len() >= src.len());
    if !through_staging() {
        return stream.memcpy_dtoh(src, dst);
    }
    through_pinned(stream, src, dst)
}

#[cfg(test)]
mod tests {
    use super::{background_copies, pinned, through_staging};
    use crate::stream_class::{StreamClass, enter};

    /// Only a background-class thread takes the pinned path, and only while
    /// the guard lives; each background read-back is counted on the path it
    /// took, a foreground one on neither.
    #[test]
    fn only_background_threads_read_back_pinned_while_it_is_on() {
        let background = || {
            std::thread::scope(|s| {
                s.spawn(|| {
                    let _bg = enter(StreamClass::Background);
                    through_staging()
                })
                .join()
                .expect("joined")
            })
        };
        let before = background_copies();
        assert!(!through_staging() && !background());
        {
            let _on = pinned();
            assert!(!through_staging(), "a foreground thread");
            assert!(background(), "a background thread");
        }
        assert!(!background());
        let after = background_copies();
        assert_eq!((after.0 - before.0, after.1 - before.1), (1, 2));
    }
}
