//! What a cheaper page-lock of the pool would cost on this box: the pages
//! faulted in by many threads (with and without transparent huge pages), then
//! registered with the driver, against the driver populating them as it
//! registers (today's pool), idle and beside a load like a block's phase A;
//! and how long a thread launching on the card stalls while each registers.
//!
//! Each case takes a fresh region of `REGION` bytes, its own mapping, aligned to
//! 2 MiB (huge pages asked for with `MADV_HUGEPAGE` where the case says so):
//! - populate+lock: `cuMemHostRegister` of the fresh region, in pieces of 1 GiB
//!   (today's pool) or 64 MiB, the driver faulting the pages in;
//! - touch: one write a 4 KiB page, split over 1, 8 or 16 threads, wall time;
//!   with huge pages, the share of the region the kernel backed with them;
//! - lock: `cuMemHostRegister` of the touched region, in 1 GiB or 64 MiB pieces.
//!
//! The load (as `lock_rate_probe`'s): six threads allocating, faulting in and
//! freeing 64 MiB over and over, and one thread copying 64 MiB from pageable
//! memory to the card over and over. The launcher queues a 4 KiB device memset
//! and waits for it, as fast as it can; its rate and longest op inside each
//! registration's window.
//!
//! Box only; needs a card:
//! `cargo test --release -p math-cuda --test lock_cheap_probe -- --ignored --exact --nocapture the_cheaper_page_lock`

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use cudarc::driver::sys;
use math_cuda::device::backend;

const REGION: usize = 4 << 30;
const HUGE: usize = 2 << 20;
const PAGE: usize = 4096;
const GIB: f64 = (1u64 << 30) as f64;
const PIECE_1G: usize = 1 << 30;
const PIECE_64M: usize = 64 << 20;
/// Linux's `MADV_HUGEPAGE` (`<asm-generic/mman-common.h>`); the libc crate
/// names it on Linux only, and this probe runs on the box.
const MADV_HUGEPAGE: libc::c_int = 14;

/// A fresh anonymous mapping of `REGION` bytes aligned to 2 MiB, unmapped on
/// drop (unregistered first if registered).
struct Region {
    map: *mut u8,
    map_len: usize,
    base: *mut u8,
    pieces: Vec<*mut u8>,
}

// SAFETY: the region is used by one thread at a time but for the touch, which
// writes disjoint slices; the driver only reads its addresses.
unsafe impl Send for Region {}
unsafe impl Sync for Region {}

impl Region {
    fn fresh(huge: bool) -> Self {
        let map_len = REGION + HUGE;
        // SAFETY: a private anonymous mapping; checked below.
        let map = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                map_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert_ne!(map, libc::MAP_FAILED, "a fresh mapping");
        let map = map.cast::<u8>();
        let offset = (HUGE - (map as usize % HUGE)) % HUGE;
        // SAFETY: inside the mapping.
        let base = unsafe { map.add(offset) };
        if huge {
            // SAFETY: a range of the mapping.
            let rc = unsafe { libc::madvise(base.cast(), REGION, MADV_HUGEPAGE) };
            assert_eq!(rc, 0, "MADV_HUGEPAGE");
        }
        Self {
            map,
            map_len,
            base,
            pieces: Vec::new(),
        }
    }

    /// One write a 4 KiB page, the region split over `threads` threads.
    fn touch(&self, threads: usize) {
        let slice = REGION / threads;
        std::thread::scope(|s| {
            for t in 0..threads {
                let at = self.base as usize + t * slice;
                s.spawn(move || {
                    for off in (0..slice).step_by(PAGE) {
                        // SAFETY: inside this thread's slice of the region.
                        unsafe { ((at + off) as *mut u8).write_volatile(1) };
                    }
                });
            }
        });
    }

    /// `cuMemHostRegister` of the region in pieces of `piece` bytes.
    fn lock(&mut self, piece: usize) {
        for at in (0..REGION).step_by(piece) {
            // SAFETY: inside the region.
            let p = unsafe { self.base.add(at) };
            // SAFETY: the region is mapped and lives until it is unregistered.
            let rc = unsafe {
                sys::cuMemHostRegister_v2(p.cast(), piece, sys::CU_MEMHOSTREGISTER_PORTABLE)
            };
            assert_eq!(rc, sys::CUresult::CUDA_SUCCESS, "cuMemHostRegister");
            self.pieces.push(p);
        }
    }

    /// The bytes of the region the kernel backs with huge pages
    /// (`/proc/self/smaps`' AnonHugePages of the mapping holding it).
    fn huge_bytes(&self) -> u64 {
        let smaps = std::fs::read_to_string("/proc/self/smaps").unwrap_or_default();
        let (lo, hi) = (self.map as usize, self.map as usize + self.map_len);
        let mut inside = false;
        let mut total = 0u64;
        for line in smaps.lines() {
            if let Some((range, _)) = line.split_once(' ')
                && let Some((a, b)) = range.split_once('-')
                && let (Ok(a), Ok(b)) = (usize::from_str_radix(a, 16), usize::from_str_radix(b, 16))
            {
                inside = a < hi && b > lo;
                continue;
            }
            if inside && let Some(kb) = line.strip_prefix("AnonHugePages:") {
                total += kb
                    .trim()
                    .trim_end_matches("kB")
                    .trim()
                    .parse::<u64>()
                    .unwrap_or(0)
                    * 1024;
            }
        }
        total
    }
}

impl Drop for Region {
    fn drop(&mut self) {
        for p in self.pieces.drain(..) {
            // SAFETY: registered by `lock`.
            unsafe { sys::cuMemHostUnregister(p.cast()) };
        }
        // SAFETY: the mapping made in `fresh`.
        unsafe { libc::munmap(self.map.cast(), self.map_len) };
    }
}

/// Seconds a GiB of the region for `f`, and its window.
fn timed(f: impl FnOnce()) -> (f64, Instant, Instant) {
    let t = Instant::now();
    f();
    let end = Instant::now();
    ((end - t).as_secs_f64() / (REGION as f64 / GIB), t, end)
}

/// The launcher's ops: each one's start (ns since `epoch`) and length (ns).
struct Launcher {
    stop: Arc<AtomicBool>,
    handle: std::thread::JoinHandle<Vec<(u64, u32)>>,
}

impl Launcher {
    fn start(epoch: Instant) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let s = Arc::clone(&stop);
        let handle = std::thread::spawn(move || {
            let be = backend().expect("a card");
            be.ctx.bind_to_thread().expect("the context");
            let stream = be.next_stream();
            let mut small = stream.alloc_zeros::<u8>(4096).expect("4 KiB on the card");
            let mut ops = Vec::with_capacity(1 << 23);
            while !s.load(Ordering::Relaxed) {
                let t = Instant::now();
                stream.memset_zeros(&mut small).expect("a memset");
                stream.synchronize().expect("a sync");
                let d = t.elapsed();
                if ops.len() < ops.capacity() {
                    ops.push((
                        (t - epoch).as_nanos() as u64,
                        d.as_nanos().min(u32::MAX as u128) as u32,
                    ));
                }
            }
            ops
        });
        Self { stop, handle }
    }

    fn finish(self) -> Vec<(u64, u32)> {
        self.stop.store(true, Ordering::Relaxed);
        self.handle.join().expect("the launcher")
    }
}

/// The launcher's ops a second and its longest op (ms) inside `[from, to)`.
fn window(ops: &[(u64, u32)], epoch: Instant, from: Instant, to: Instant) -> (f64, f64) {
    let (a, b) = (
        (from - epoch).as_nanos() as u64,
        (to - epoch).as_nanos() as u64,
    );
    let inside: Vec<&(u64, u32)> = ops.iter().filter(|(s, _)| *s >= a && *s < b).collect();
    let secs = (b - a) as f64 * 1e-9;
    let longest = inside.iter().map(|(_, d)| *d).max().unwrap_or(0) as f64 * 1e-6;
    (inside.len() as f64 / secs.max(1e-9), longest)
}

/// The cases of one pass (`tag` idle or load): each a name, its seconds a GiB
/// and its window (for the launcher's read).
fn pass(tag: &str, cases: &mut Vec<(String, f64, Instant, Instant)>) {
    let mut add = |name: String, (secs, from, to): (f64, Instant, Instant)| {
        println!("PROBE {tag}_{name}_s_per_gib {secs:.4}");
        cases.push((format!("{tag}_{name}"), secs, from, to));
    };
    // Today's pool: the driver populates as it registers.
    for (piece, pname) in [(PIECE_1G, "1g"), (PIECE_64M, "64m")] {
        let mut r = Region::fresh(false);
        add(format!("populate_lock_{pname}"), timed(|| r.lock(piece)));
    }
    // Faulted in by threads, then registered.
    for huge in [false, true] {
        let pages = if huge { "thp" } else { "4k" };
        for threads in [1usize, 8, 16] {
            let r = Region::fresh(huge);
            add(
                format!("touch_{pages}_t{threads}"),
                timed(|| r.touch(threads)),
            );
            if huge {
                println!(
                    "PROBE {tag}_touch_thp_t{threads}_huge_share {:.3}",
                    r.huge_bytes() as f64 / REGION as f64
                );
            }
        }
        for (piece, pname) in [(PIECE_1G, "1g"), (PIECE_64M, "64m")] {
            let mut r = Region::fresh(huge);
            r.touch(16);
            add(format!("lock_{pages}_{pname}"), timed(|| r.lock(piece)));
        }
    }
}

#[test]
#[ignore = "box only: page-locking on a card"]
fn the_cheaper_page_lock() {
    let be = backend().expect("a card");
    be.ctx.bind_to_thread().expect("the context");
    let epoch = Instant::now();
    let mut cases = Vec::new();
    pass("idle", &mut cases);

    // The launcher alone, then the load, then the same cases under it.
    let launcher = Launcher::start(epoch);
    std::thread::sleep(Duration::from_secs(2));
    let quiet = (Instant::now(), {
        std::thread::sleep(Duration::from_secs(2));
        Instant::now()
    });
    let stop = Arc::new(AtomicBool::new(false));
    let faulted = Arc::new(AtomicU64::new(0));
    let copied = Arc::new(AtomicU64::new(0));
    let mut load = Vec::new();
    for _ in 0..6 {
        let (stop, faulted) = (Arc::clone(&stop), Arc::clone(&faulted));
        load.push(std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                let v = vec![1u8; 64 << 20];
                std::hint::black_box(&v);
                faulted.fetch_add(64 << 20, Ordering::Relaxed);
            }
        }));
    }
    {
        let (stop, copied) = (Arc::clone(&stop), Arc::clone(&copied));
        load.push(std::thread::spawn(move || {
            let be = backend().expect("a card");
            be.ctx.bind_to_thread().expect("the context");
            let stream = be.next_stream();
            let host = vec![7u8; 64 << 20];
            // SAFETY: every byte is written by the copies below before any read.
            let mut dev = unsafe { stream.alloc::<u8>(64 << 20) }.expect("64 MiB on the card");
            while !stop.load(Ordering::Relaxed) {
                stream
                    .memcpy_htod(&host, &mut dev)
                    .expect("a pageable copy");
                stream.synchronize().expect("a sync");
                copied.fetch_add(64 << 20, Ordering::Relaxed);
            }
        }));
    }
    std::thread::sleep(Duration::from_secs(2));
    let based = Instant::now();
    std::thread::sleep(Duration::from_secs(3));
    let base = (based, Instant::now());
    pass("load", &mut cases);
    let loaded = based.elapsed().as_secs_f64();
    stop.store(true, Ordering::Relaxed);
    load.into_iter()
        .for_each(|t| t.join().expect("a load thread"));
    let ops = launcher.finish();
    println!(
        "PROBE load_faulted_gib_per_s {:.3}",
        faulted.load(Ordering::Relaxed) as f64 / GIB / loaded
    );
    println!(
        "PROBE load_copied_gib_per_s {:.3}",
        copied.load(Ordering::Relaxed) as f64 / GIB / loaded
    );
    for (name, (from, to)) in [("quiet", quiet), ("base", base)] {
        let (rate, longest) = window(&ops, epoch, from, to);
        println!("PROBE launcher_{name}_ops_per_s {rate:.0}");
        println!("PROBE launcher_{name}_longest_ms {longest:.3}");
    }
    for (name, _, from, to) in cases.iter().filter(|c| c.0.starts_with("load_")) {
        let (rate, longest) = window(&ops, epoch, *from, *to);
        println!("PROBE launcher_{name}_ops_per_s {rate:.0}");
        println!("PROBE launcher_{name}_longest_ms {longest:.3}");
    }
    // The cheapest lock found: a touch and a registration of the same pages,
    // against today's.
    let get = |name: &str| {
        cases
            .iter()
            .find(|c| c.0 == name)
            .map(|c| c.1)
            .unwrap_or(f64::NAN)
    };
    for tag in ["idle", "load"] {
        let today = get(&format!("{tag}_populate_lock_1g"));
        let mut best = (f64::INFINITY, String::new());
        for pages in ["4k", "thp"] {
            for pname in ["1g", "64m"] {
                let total = get(&format!("{tag}_touch_{pages}_t16"))
                    + get(&format!("{tag}_lock_{pages}_{pname}"));
                if total < best.0 {
                    best = (total, format!("touch_{pages}_t16+lock_{pages}_{pname}"));
                }
            }
        }
        println!(
            "PROBE {tag}_best {} {:.4} s/GiB against today's {today:.4} · 20 / 38 GiB {:.2} / {:.2} s against {:.2} / {:.2} s",
            best.1,
            best.0,
            20.0 * best.0,
            38.0 * best.0,
            20.0 * today,
            38.0 * today
        );
    }
    println!("PROBE done");
}
