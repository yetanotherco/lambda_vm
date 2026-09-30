//! Pinned slots on a card (D-TRACE stage 1b, S2): the upload from a slot, the
//! pool on real page-locked memory, and what allocating the slots costs.
//!
//! * The upload: columns written into a slot, each sent up to its last raw
//!   nonzero value, must leave the card holding exactly what the pageable
//!   upload of the same columns leaves, read back from a pool that just held
//!   `u64::MAX` everywhere (a tail nobody zeroed reads as garbage). Controls:
//!   the dirty pool does hand back garbage; one column's prefix cut one value
//!   short changes exactly that value.
//! * The pool: a lease misses at once when nothing is free; a dropped lease or
//!   frozen slot gives its block back.
//! * The allocation (D-TRACE-1B R3): two blocks of the production size, the
//!   seconds each takes, and the latency of another thread's driver calls
//!   meanwhile. Printed, with the verdict against S2's gate.
//! * The rate (R4's preview): epoch 0's real column mix from a slot through
//!   `upload_parts`, against the pageable upload.
//!
//! Needs a card: `cargo test --release -p math-cuda --test pinned_slots --
//! --ignored --test-threads=1 --nocapture`.

use std::time::{Duration, Instant};

use math_cuda::columns::{DeviceColumns, upload_totals};
use math_cuda::device::backend;
use math_cuda::pinned_slots::{Miss, SlotPool, slot_bytes_setting};

/// The Goldilocks prime: zero as a field element, not as bytes.
const RAW_P: u64 = 0xFFFF_FFFF_0000_0001;

/// splitmix64, so a failure reproduces.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

/// One past the last raw nonzero value: what the producer records.
fn nonzero_len(column: &[u64]) -> usize {
    column.iter().rposition(|&v| v != 0).map_or(0, |i| i + 1)
}

/// Columns with every tail the upload distinguishes: none, a zero inside and
/// a nonzero last value, all zero, raw `p` as the last nonzero value, raw `p`
/// last of all, one of 48 MiB with a tail of two thirds, and many small ones.
fn columns() -> Vec<Vec<u64>> {
    let mut rng = Rng(0x0b1b);
    let mut filled = |len: usize, live: usize| -> Vec<u64> {
        let mut c = vec![0u64; len];
        for v in &mut c[..live] {
            *v = rng.next() | 1;
        }
        c
    };
    let mut out = vec![filled(1 << 16, 1 << 16)];
    let mut inner_zero = filled(1 << 16, 1 << 16);
    inner_zero[1000] = 0;
    out.push(inner_zero);
    out.push(filled(1 << 16, 0));
    let mut p_then_zeros = filled(1 << 16, 3_000);
    p_then_zeros[3_000] = RAW_P;
    out.push(p_then_zeros);
    let mut p_last = filled(1 << 16, 10);
    p_last[(1 << 16) - 1] = RAW_P;
    out.push(p_last);
    out.push(filled(6 << 20, 2 << 20));
    for k in 0..40 {
        out.push(filled(1 << 12, (k * 97) % (1 << 12)));
    }
    out.push(filled(1, 1));
    out.push(filled(2, 0));
    out
}

/// Fill the pool with `u64::MAX` and give it back, so the next allocation of
/// that size is dirty.
fn dirty_the_pool(elems: usize) {
    let be = backend().expect("a card");
    let stream = be.next_stream();
    let mut junk = stream.alloc_zeros::<u64>(elems).expect("alloc");
    let ones = vec![u64::MAX; elems];
    stream.memcpy_htod(&ones, &mut junk).expect("fill");
    stream.synchronize().expect("sync");
    drop(junk);
    stream.synchronize().expect("sync");
}

fn read_back(store: &DeviceColumns) -> Vec<u64> {
    let stream = store.stream();
    let back = stream.clone_dtoh(&*store.buffer()).expect("dtoh");
    stream.synchronize().expect("sync");
    back
}

/// A pool of one block that holds `elems`, filled.
fn one_block(elems: usize) -> SlotPool {
    let pool = SlotPool::new(1, elems * 8);
    pool.fill().expect("a pinned block");
    pool
}

#[test]
#[ignore = "needs a card"]
fn a_pinned_upload_with_zero_tails_holds_the_columns_exactly() {
    let host = columns();
    let expected: Vec<u64> = host.iter().flatten().copied().collect();
    let refs: Vec<&[u64]> = host.iter().map(Vec::as_slice).collect();

    // The pageable upload, today's path.
    dirty_the_pool(expected.len());
    let pageable = DeviceColumns::upload(&refs).expect("pageable upload");
    let plain = read_back(&pageable);
    assert_eq!(pageable.upload_record().zero_tails, 0);
    assert!(!pageable.upload_record().pinned);
    drop(pageable);

    // The same columns written into a slot, back to back, each with its tail.
    let pool = one_block(expected.len());
    let mut lease = pool.try_lease(expected.len()).expect("the block");
    lease.as_mut_slice().copy_from_slice(&expected);
    let frozen = lease.freeze();
    let slot = frozen.as_slice();
    let mut parts = Vec::with_capacity(host.len());
    let mut at = 0usize;
    for column in &host {
        let view = &slot[at..at + column.len()];
        parts.push((view, nonzero_len(view)));
        at += column.len();
    }
    let expected_tails: u64 = parts.iter().map(|(c, s)| ((c.len() - s) * 8) as u64).sum();
    assert!(
        expected_tails > 0,
        "the fixture has no tail to leave behind"
    );
    let raw_p_kept = parts[3].1;
    assert_eq!(raw_p_kept, 3_001, "raw p counts as nonzero");
    assert_eq!(
        parts[4].1,
        1 << 16,
        "a column ending in raw p is sent whole"
    );
    assert_eq!(parts[2].1, 0, "an all-zero column sends nothing");

    dirty_the_pool(expected.len());
    let before = upload_totals();
    let pinned = DeviceColumns::upload_parts(&parts, true).expect("pinned upload");
    let after = upload_totals();
    let record = pinned.upload_record();
    assert!(record.pinned);
    assert_eq!(after.1, before.1 + 1, "the pinned upload was not counted");
    assert_eq!(record.zero_tails, expected_tails);
    assert_eq!(record.bytes, expected.len() as u64 * 8);
    let got = read_back(&pinned);
    println!("pinned upload: {}", record.line());

    let first_bad = |a: &[u64]| a.iter().zip(&expected).position(|(x, y)| x != y);
    assert_eq!(first_bad(&plain), None, "CONTROL: the pageable upload");
    assert_eq!(first_bad(&got), None, "the pinned upload with zero tails");
    assert_eq!(got, plain, "the two stores differ");
    for (k, column) in host.iter().enumerate() {
        let (at, len) = pinned.span(k);
        assert_eq!(&got[at..at + len], column.as_slice(), "column {k}");
    }
    drop(pinned);

    // The mutation: column 1's prefix cut one short leaves its last value, a
    // nonzero one, zero on the card, and nothing else changes.
    let mut short = parts.clone();
    short[1].1 -= 1;
    dirty_the_pool(expected.len());
    let cut = DeviceColumns::upload_parts(&short, true).expect("cut upload");
    let (at, len) = cut.span(1);
    let bad = read_back(&cut);
    drop(cut);
    let diffs: Vec<usize> = bad
        .iter()
        .zip(&expected)
        .enumerate()
        .filter(|(_, (x, y))| x != y)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(
        diffs,
        [at + len - 1],
        "MUTATION: a prefix one short must change exactly its last value"
    );
    assert_eq!(bad[at + len - 1], 0);
    drop(frozen);
    assert_eq!(pool.stats().free, 1, "the frozen slot did not come back");
}

/// The dirty pool is what makes the tail check bite: a dirtied allocation of
/// the same size, left unwritten, does not read as zeros.
#[test]
#[ignore = "needs a card"]
fn a_dirty_pool_hands_back_garbage() {
    let elems = 1 << 20;
    dirty_the_pool(elems);
    let be = backend().expect("a card");
    let stream = be.next_stream();
    // SAFETY: read back only to show what an unwritten buffer holds.
    let buf = unsafe { math_cuda::device::alloc_or_trim::<u64>(&stream, elems) }.expect("alloc");
    let back = stream.clone_dtoh(&buf).expect("dtoh");
    stream.synchronize().expect("sync");
    assert!(
        back.iter().any(|&v| v != 0),
        "CONTROL: the pool handed back zeros, so the tail check could not fail"
    );
}

/// The pool on real page-locked memory: nothing before the fill, then every
/// lease or miss at once, and blocks back when their holders go.
#[test]
#[ignore = "needs a card"]
fn the_pool_lends_without_waiting() {
    let elems = (64 << 20) / 8;
    let pool = SlotPool::new(2, elems * 8);
    assert_eq!(pool.try_lease(1).err(), Some(Miss::NotReady));
    let took = pool.fill().expect("two pinned blocks");
    assert_eq!(took.len(), 2);
    assert!(pool.fill().expect("a second fill").is_empty(), "overshot");

    assert_eq!(pool.try_lease(elems + 1).err(), Some(Miss::TooSmall));
    let mut a = pool.try_lease(elems).expect("block");
    let b = pool.try_lease(elems / 2).expect("the other block");
    let started = Instant::now();
    assert_eq!(pool.try_lease(1).err(), Some(Miss::NoneFree));
    let waited = started.elapsed();
    assert!(waited < Duration::from_millis(1), "a miss took {waited:?}");
    drop(b);
    let b = pool.try_lease(1).expect("the dropped lease's block");
    drop(b);

    a.as_mut_slice().fill(0xa5a5);
    let frozen = a.freeze();
    assert!(frozen.as_slice().iter().all(|&v| v == 0xa5a5));
    let s = pool.stats();
    assert_eq!((s.created, s.free), (2, 1));
    drop(frozen);
    let s = pool.stats();
    assert_eq!((s.created, s.free), (2, 2));
    assert_eq!(
        (s.leased, s.none_free, s.too_small, s.not_ready),
        (3, 1, 1, 1)
    );
    println!("two 64 MiB blocks in {took:?} s");
}

/// A driver call and its synchronize: a small memset, the probe for how long
/// another thread's calls wait while the blocks are allocated.
fn probe_latencies(until: impl Fn() -> bool, at_least: Duration) -> Vec<f64> {
    let be = backend().expect("a card");
    let stream = be.next_stream();
    let mut scratch = stream.alloc_zeros::<u64>(1 << 17).expect("alloc");
    let started = Instant::now();
    let mut samples = Vec::new();
    loop {
        let t = Instant::now();
        stream.memset_zeros(&mut scratch).expect("memset");
        stream.synchronize().expect("sync");
        samples.push(t.elapsed().as_secs_f64() * 1e3);
        if until() && started.elapsed() >= at_least {
            return samples;
        }
    }
}

fn percentiles(mut ms: Vec<f64>) -> (f64, f64, f64, usize) {
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let at = |q: f64| ms[((ms.len() - 1) as f64 * q).round() as usize];
    (at(0.5), at(0.99), ms[ms.len() - 1], ms.len())
}

/// D-TRACE-1B R3: two blocks of the production size (`LAMBDA_VM_WHIR_PINNED_SLOT_MB`,
/// 3072 MiB by default), timed, while another thread's driver calls are timed.
/// Keep if each block takes ≤ 1.0 s and the calls' p99 stays ≤ 5 ms; stop if a
/// block takes > 2 s or a call stalls > 20 ms.
#[test]
#[ignore = "needs a card"]
fn allocating_the_slots_leaves_the_card_responsive() {
    let bytes = slot_bytes_setting(
        std::env::var("LAMBDA_VM_WHIR_PINNED_SLOT_MB")
            .ok()
            .as_deref(),
    );
    let idle = percentiles(probe_latencies(|| true, Duration::from_millis(300)));

    let pool = SlotPool::new(2, bytes);
    let done = std::sync::atomic::AtomicBool::new(false);
    let (took, during) = std::thread::scope(|s| {
        let filler = s.spawn(|| {
            let took = pool.fill().expect("two production blocks");
            done.store(true, std::sync::atomic::Ordering::SeqCst);
            took
        });
        let during = probe_latencies(
            || done.load(std::sync::atomic::Ordering::SeqCst),
            Duration::ZERO,
        );
        (filler.join().expect("the filler"), during)
    });
    let during = percentiles(during);
    let gb = bytes as f64 / 1e9;
    let worst = took.iter().cloned().fold(0.0, f64::max);
    let verdict = if worst > 2.0 || during.2 > 20.0 {
        "STOP (allocate after epoch 0's commit instead)"
    } else if worst <= 1.0 && during.1 <= 5.0 {
        "KEEP"
    } else {
        "BETWEEN (keep, and watch M7)"
    };
    println!(
        "R3: {} blocks of {gb:.2} GB in {:?} s · driver call idle p50 {:.3} p99 {:.3} max {:.3} ms ({} calls) · \
         during the allocation p50 {:.3} p99 {:.3} max {:.3} ms ({} calls) · VERDICT {verdict}",
        took.len(),
        took.iter().map(|s| format!("{s:.3}")).collect::<Vec<_>>(),
        idle.0,
        idle.1,
        idle.2,
        idle.3,
        during.0,
        during.1,
        during.2,
        during.3,
    );
    assert_eq!(took.len(), 2);
    assert!(during.3 > 0, "no call was timed during the allocation");
}

/// Epoch 0's committed tables in the G6 census: `(rows per column, columns)`.
const EPOCH0: &[(usize, usize)] = &[
    (1 << 20, 21),
    (1 << 20, 6),
    (32, 10),
    (128, 5),
    (1 << 21, 38),
    (1 << 20, 17),
    (1 << 18, 29),
    (1 << 17, 49),
    (1 << 20, 29),
    (1 << 19, 18),
    (1 << 11, 26),
    (1 << 17, 14),
    (1 << 21, 10),
    (1 << 21, 10),
    (1 << 18, 10),
    (1 << 13, 12),
    (1 << 17, 26),
    (1 << 19, 16),
    (1 << 10, 38),
    (1 << 21, 9),
];

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

/// R4's preview: the rate `upload_parts` reaches from a slot over epoch 0's
/// real column mix, with no tail and with a quarter of each column's tail
/// zero, against the pageable upload of the same columns. Median of five,
/// GB = 1e9 bytes over the bytes sent.
#[test]
#[ignore = "needs a card"]
fn a_pinned_upload_runs_at_dma_speed() {
    let lens: Vec<usize> = EPOCH0
        .iter()
        .flat_map(|&(rows, cols)| std::iter::repeat_n(rows, cols))
        .collect();
    let total: usize = lens.iter().sum();
    let mut rng = Rng(0xd4a);
    let host: Vec<Vec<u64>> = lens
        .iter()
        .map(|&n| (0..n).map(|_| rng.next() | 1).collect())
        .collect();
    let refs: Vec<&[u64]> = host.iter().map(Vec::as_slice).collect();

    let pool = one_block(total);
    let mut lease = pool.try_lease(total).expect("the block");
    let mut at = 0usize;
    for column in &host {
        lease.as_mut_slice()[at..at + column.len()].copy_from_slice(column);
        at += column.len();
    }
    let frozen = lease.freeze();
    let slot = frozen.as_slice();
    let mut whole = Vec::new();
    let mut tailed = Vec::new();
    let mut at = 0usize;
    for &n in &lens {
        let view = &slot[at..at + n];
        whole.push((view, n));
        tailed.push((view, n - n / 4));
        at += n;
    }

    let rate = |run: &dyn Fn() -> DeviceColumns| {
        median(
            (0..5)
                .map(|_| {
                    let store = run();
                    let r = store.upload_record();
                    (r.bytes - r.zero_tails) as f64 / 1e9 / r.secs
                })
                .collect(),
        )
    };
    let pageable = rate(&|| DeviceColumns::upload(&refs).expect("pageable"));
    let pinned = rate(&|| DeviceColumns::upload_parts(&whole, true).expect("pinned"));
    let pinned_tails = rate(&|| DeviceColumns::upload_parts(&tailed, true).expect("tails"));
    println!(
        "RATE: epoch 0's {} columns, {:.2} GB · pageable {pageable:.1} GB/s · pinned {pinned:.1} GB/s · \
         pinned with quarter tails {pinned_tails:.1} GB/s (sent) · pinned ÷ pageable {:.2}×",
        lens.len(),
        total as f64 * 8.0 / 1e9,
        pinned / pageable
    );
    assert!(pageable > 0.0 && pinned > 0.0 && pinned_tails > 0.0);
}
