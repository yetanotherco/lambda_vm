//! A GKR layer's rounds with their sums through a slot against today's second
//! launch and read-back (`crate::round_slot`, `LAMBDA_VM_ARGUE_ROUND_SLOT`).
//!
//! ```text
//! cargo test --release -p math-cuda --test round_slot -- --test-threads=1
//! ```
//!
//! Needs a GPU. Every round's sums and the halves after the last round are
//! compared limb for limb: the slot's last-block reduction promises today's
//! limbs, not only today's values. The ignored test is the capability probe
//! that decides whether the next round is ever queued behind a card-side wait:
//! the stream-memory attributes, and three round trips timed.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg};
use math_cuda::device::backend;
use math_cuda::gkr::{DeviceFractionTree, GruenLayer};
use math_cuda::round_slot::{self, MappedSlot, RoundSlot};

static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct Restore {
    _serial: std::sync::MutexGuard<'static, ()>,
}

impl Drop for Restore {
    fn drop(&mut self) {
        round_slot::force_check_fault(false);
    }
}

fn serial() -> Restore {
    Restore {
        _serial: SERIAL.lock().unwrap_or_else(|e| e.into_inner()),
    }
}

const P: u64 = 0xFFFF_FFFF_0000_0001;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn limb(&mut self) -> u64 {
        let v = self.next() % P;
        if self.next() & 7 == 0 && v < 0xFFFF_FFFF {
            v + P
        } else {
            v
        }
    }

    fn ext(&mut self) -> [u64; 3] {
        [self.limb(), self.limb(), self.limb()]
    }
}

fn upload(stream: &Arc<CudaStream>, values: &[u64]) -> CudaSlice<u64> {
    math_cuda::device::htod_or_trim(stream, values).expect("upload")
}

/// A tree over random fractions, the padding `0/1` past `real` (a kept tree
/// when `kept`, today's otherwise).
fn tree(interactions: usize, rows: usize, kept: bool, seed: u64) -> DeviceFractionTree {
    let be = backend().expect("a device");
    let full = interactions.next_power_of_two() * rows;
    let real = interactions * rows;
    let mut rng = Rng(seed | 1);
    let mut p = Vec::with_capacity(full * 3);
    let mut q = Vec::with_capacity(full * 3);
    for i in 0..full {
        if i < real {
            p.extend_from_slice(&rng.ext());
            q.extend_from_slice(&rng.ext());
        } else {
            p.extend_from_slice(&[0, 0, 0]);
            q.extend_from_slice(&[1, 0, 0]);
        }
    }
    let stream = be.next_stream();
    if kept {
        DeviceFractionTree::from_truncated_input(
            stream.clone(),
            upload(&stream, &p[..real * 3]),
            upload(&stream, &q[..real * 3]),
            real,
            full.trailing_zeros() as usize,
        )
    } else {
        DeviceFractionTree::from_device(stream.clone(), upload(&stream, &p), upload(&stream, &q))
    }
    .expect("the tree")
}

/// Every round of a layer and its finish, raw.
fn run(
    layer: &mut GruenLayer,
    lambda: &[u64; 3],
    challenges: &[[u64; 3]],
) -> (Vec<Vec<u64>>, Vec<u64>) {
    let mut sums = Vec::new();
    for j in 0..layer.rounds() {
        let fold = (j > 0).then(|| &challenges[j - 1]);
        sums.push(layer.round(fold, lambda, j % 3 == 1).expect("a round"));
    }
    let last = (layer.rounds() > 0).then(|| &challenges[layer.rounds() - 1]);
    (sums, layer.finish(last).expect("the finish"))
}

/// One layer's rounds today and through a slot under `mode`, over twin trees.
fn compare(interactions: usize, rows: usize, kept: bool, low: usize, mode: RoundSlot, seed: u64) {
    let what = format!("{interactions} x {rows} rows, kept {kept}, low {low}, {mode:?}");
    let today_tree = tree(interactions, rows, kept, seed);
    let slot_tree = tree(interactions, rows, kept, seed);
    let input = today_tree.num_layers() - 1;
    let m = today_tree.layer_num_vars(input) - 1;
    let mut rng = Rng(seed ^ 0x51);
    let point: Vec<u64> = (0..m).flat_map(|_| rng.ext()).collect();
    let lambda = rng.ext();
    let challenges: Vec<[u64; 3]> = (0..m).map(|_| rng.ext()).collect();
    let mut a = today_tree.layer_gruen(input, &point, low).unwrap();
    let mut b = slot_tree.layer_gruen(input, &point, low).unwrap();
    assert!(b.use_slot(mode), "{what}: a slot was had");
    let before = round_slot::counts();
    let (sums_a, halves_a) = run(&mut a, &lambda, &challenges);
    let mid = round_slot::counts();
    let (sums_b, halves_b) = run(&mut b, &lambda, &challenges);
    let after = round_slot::counts();
    for (j, (x, y)) in sums_a.iter().zip(&sums_b).enumerate() {
        assert_eq!(x, y, "{what}: round {j}'s sums, raw");
    }
    assert_eq!(halves_a, halves_b, "{what}: the halves, raw");
    let rounds = sums_b.len() as u64;
    assert_eq!(
        mid.second_launches - before.second_launches,
        rounds,
        "{what}: today's launches"
    );
    assert_eq!(
        after.slot_rounds - mid.slot_rounds,
        rounds,
        "{what}: every round through the slot"
    );
    let second = after.second_launches - mid.second_launches;
    match mode {
        RoundSlot::On => assert_eq!(second, 0, "{what}: no second launch"),
        _ => {
            assert_eq!(second, rounds, "{what}: check runs today's beside it");
            assert_eq!(
                after.checked - mid.checked,
                rounds,
                "{what}: every round compared"
            );
            assert_eq!(after.mismatches, mid.mismatches, "{what}: none differed");
        }
    }
}

/// ★ A layer's rounds through a slot hand back today's sums every round and
/// today's halves at the end, limb for limb, with no second launch: whole and
/// kept layers, every host tail, rounds summing `H(0)` and not.
#[test]
fn the_slot_hands_back_todays_sums_limb_for_limb() {
    let _restore = serial();
    if backend().is_err() {
        eprintln!("no device; skipping");
        return;
    }
    let mut n = 0;
    for (interactions, rows) in [
        (5usize, 8usize),
        (20, 1 << 10),
        (1031, 1 << 3),
        (8, 1 << 9),
        (3, 1 << 12),
    ] {
        for (kept, low) in [(false, 6usize), (true, 0), (true, 6), (false, 0)] {
            compare(interactions, rows, kept, low, RoundSlot::On, 0x5107 + n);
            n += 1;
        }
    }
    eprintln!("round slot: {n} layers, every round and finish raw-identical, no second launch");
}

/// `Check` runs today's second launch beside the slot every round, compares
/// the two and finds none differing.
#[test]
fn the_check_compares_every_round() {
    let _restore = serial();
    if backend().is_err() {
        eprintln!("no device; skipping");
        return;
    }
    compare(20, 1 << 12, true, 6, RoundSlot::Check, 0xC4EC);
    compare(1031, 1 << 4, false, 0, RoundSlot::Check, 0xC4ED);
}

/// ⛔ The check can fail: with the fault armed every compared round counts as
/// differing, and the values handed back are still today's.
#[test]
fn a_differing_round_is_counted_and_todays_is_used() {
    let _restore = serial();
    if backend().is_err() {
        eprintln!("no device; skipping");
        return;
    }
    let today_tree = tree(20, 1 << 10, false, 7);
    let slot_tree = tree(20, 1 << 10, false, 7);
    let input = today_tree.num_layers() - 1;
    let m = today_tree.layer_num_vars(input) - 1;
    let mut rng = Rng(8);
    let point: Vec<u64> = (0..m).flat_map(|_| rng.ext()).collect();
    let lambda = rng.ext();
    let challenges: Vec<[u64; 3]> = (0..m).map(|_| rng.ext()).collect();
    let mut a = today_tree.layer_gruen(input, &point, 6).unwrap();
    let mut b = slot_tree.layer_gruen(input, &point, 6).unwrap();
    assert!(b.use_slot(RoundSlot::Check));
    let want = run(&mut a, &lambda, &challenges);
    round_slot::force_check_fault(true);
    let before = round_slot::counts();
    let got = run(&mut b, &lambda, &challenges);
    let after = round_slot::counts();
    round_slot::force_check_fault(false);
    assert_eq!(want, got, "today's values are used");
    assert_eq!(
        after.mismatches - before.mismatches,
        got.0.len() as u64,
        "every compared round counted"
    );
}

/// The pool hands out at most its pages' slots, declines past them (the
/// session keeps today's rounds, counted), and takes slots back on drop.
#[test]
fn the_pool_declines_past_its_pages_and_takes_slots_back() {
    let _restore = serial();
    let Ok(be) = backend() else {
        eprintln!("no device; skipping");
        return;
    };
    let stream = be.next_stream();
    let mut held = Vec::new();
    while let Some(slot) = MappedSlot::take(&stream) {
        held.push(slot);
        assert!(held.len() <= 16 * 64, "no more slots than 16 pages of 64");
    }
    assert_eq!(held.len(), 16 * 64, "every page's slots were handed out");
    let before = round_slot::counts();
    let declined = round_slot::SlotRounds::open(&stream, RoundSlot::On);
    assert!(declined.is_none(), "a spent pool declines");
    assert_eq!(
        round_slot::counts().declines - before.declines,
        1,
        "the decline counted"
    );
    drop(held);
    assert!(
        MappedSlot::take(&stream).is_some(),
        "a slot given back is taken again"
    );
}

// ── the capability probe ────────────────────────────────────────────────────

/// The word the queued round waits on: the slot's 104th u64, past the
/// sequence word, which the kernel does not write.
const FLAG: usize = round_slot::SEQ + 7;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// ⏱ The probe (run with `--ignored --nocapture`): whether the card takes stream
/// memory operations, and one GKR round's round trip timed three ways over a
/// small layer (a quarter of 256, so the trip is all fixed cost):
/// today's (the round, the second launch, a pageable read-back and a stream
/// wait); a slot (the slot round and a host spin); and a slot with the round
/// already queued behind `cuStreamWaitValue32` on a word the host writes —
/// timed from that write to the spin's end. `PROBE ATTR`, `PROBE TRIP` and
/// `PROBE DECISION` lines; the queued round is worth building only if the
/// attribute is set and the queued trip is at most 0.6 of the spin's.
#[test]
#[ignore]
fn the_round_trip_three_ways() {
    use cudarc::driver::sys;
    let _restore = serial();
    let be = backend().expect("a device");
    const WARM: usize = 200;
    const TIMED: usize = 10_000;
    let attr = |a: sys::CUdevice_attribute| -> i32 {
        let mut v = 0i32;
        unsafe { sys::cuDeviceGetAttribute(&mut v, a, be.ctx.cu_device()) };
        v
    };
    let mem_ops = attr(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1);
    let wait_nor = attr(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR);
    println!("PROBE ATTR stream_mem_ops={mem_ops} wait_value_nor={wait_nor}");

    let stream = be.next_stream();
    let half = 512u64;
    let quarter = 256u64;
    let mut rng = Rng(0x9E0);
    let p: Vec<u64> = (0..2 * half * 3).map(|_| rng.limb()).collect();
    let q: Vec<u64> = (0..2 * half * 3).map(|_| rng.limb()).collect();
    let p_dev = upload(&stream, &p);
    let q_dev = upload(&stream, &q);
    let e_hi = upload(&stream, &vec![1u64; 64 * 3]);
    let e_lo = upload(&stream, &[1u64; 4 * 3]);
    let mut partials = upload(&stream, &vec![0u64; 3 * 2048 * 3]);
    let mut sums = upload(&stream, &[0u64; 9]);
    let mut ticket = math_cuda::device::alloc_zeros_or_trim::<u32>(&stream, 1).unwrap();
    let slot = MappedSlot::take(&stream).expect("a slot");
    let (p_at, q_at) = {
        let (a, _g) = p_dev.device_ptr(&stream);
        let (b, _h) = q_dev.device_ptr(&stream);
        (a, b)
    };
    let (block, grid) = (256u32, 1u32);
    let (zero, fold, low, want) = (0u64, 0u32, 2u32, 0u32);
    let lambda = [3u64, 0, 0];
    let launch_round =
        |slotted: bool, seq: u64, partials: &mut CudaSlice<u64>, ticket: &mut CudaSlice<u32>| {
            let kernel = if slotted {
                &be.gkr_round_gruen_slot
            } else {
                &be.gkr_round_gruen
            };
            let dev = slot.dev();
            unsafe {
                let mut l = stream.launch_builder(kernel);
                l.arg(&p_at)
                    .arg(&q_at)
                    .arg(&half)
                    .arg(&quarter)
                    .arg(&fold)
                    .arg(&zero)
                    .arg(&zero)
                    .arg(&zero)
                    .arg(&lambda[0])
                    .arg(&lambda[1])
                    .arg(&lambda[2])
                    .arg(&e_hi)
                    .arg(&e_lo)
                    .arg(&low)
                    .arg(&want)
                    .arg(partials);
                if slotted {
                    l.arg(ticket).arg(&dev).arg(&seq);
                }
                l.launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (block, 1, 1),
                    shared_mem_bytes: if slotted {
                        round_slot::shared_bytes(block)
                    } else {
                        block * 24
                    },
                })
                .unwrap();
            }
        };
    // Today's trip.
    let mut today = Vec::new();
    for i in 0..WARM + TIMED {
        let from = std::time::Instant::now();
        launch_round(false, 0, &mut partials, &mut ticket);
        unsafe {
            stream
                .launch_builder(&be.sum_partials_ext3)
                .arg(&partials)
                .arg(&(grid as u64))
                .arg(&mut sums)
                .launch(LaunchConfig {
                    grid_dim: (2, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 256 * 24,
                })
                .unwrap();
        }
        let _ = stream.clone_dtoh(&sums.slice(0..6)).unwrap();
        stream.synchronize().unwrap();
        if i >= WARM {
            today.push(from.elapsed().as_secs_f64() * 1e6);
        }
    }
    // The slot's trip; the sequence numbers here are the probe's own.
    let mut spin = Vec::new();
    for i in 0..WARM + TIMED {
        let seq = 1_000_000 + i as u64;
        let from = std::time::Instant::now();
        launch_round(true, seq, &mut partials, &mut ticket);
        let _ = slot.wait(seq, 2).unwrap();
        if i >= WARM {
            spin.push(from.elapsed().as_secs_f64() * 1e6);
        }
    }
    // The queued trip: the round waits on a word the host writes, then runs.
    let mut queued = Vec::new();
    if mem_ops != 0 {
        let flag = slot.dev() + FLAG as u64 * 8;
        let flag_host = |v: u32| unsafe {
            // The host's view of the same word (the slot's 104th u64).
            let words = slot_host(&slot);
            std::ptr::write_volatile(words.add(FLAG) as *mut u32, v);
        };
        flag_host(0);
        for i in 0..WARM + TIMED {
            let seq = 2_000_000 + i as u64;
            let value = (i + 1) as u32;
            unsafe {
                sys::cuStreamWaitValue32_v2(
                    stream.cu_stream(),
                    flag,
                    value,
                    sys::CUstreamWaitValue_flags::CU_STREAM_WAIT_VALUE_GEQ as u32,
                )
                .result()
                .unwrap();
            }
            launch_round(true, seq, &mut partials, &mut ticket);
            std::thread::sleep(std::time::Duration::from_micros(50));
            let from = std::time::Instant::now();
            flag_host(value);
            let _ = slot.wait(seq, 2).unwrap();
            if i >= WARM {
                queued.push(from.elapsed().as_secs_f64() * 1e6);
            }
        }
    }
    let (t, s) = (median(today), median(spin));
    let qd = (!queued.is_empty()).then(|| median(queued));
    println!(
        "PROBE TRIP reps={TIMED} today_us={t:.2} slot_us={s:.2} queued_us={} slot_over_today={:.3} queued_over_slot={}",
        qd.map_or("-".to_string(), |v| format!("{v:.2}")),
        s / t,
        qd.map_or("-".to_string(), |v| format!("{:.3}", v / s))
    );
    let build = wait_nor != 0 && mem_ops != 0 && qd.is_some_and(|v| v <= 0.6 * s);
    println!(
        "PROBE DECISION build the queued round: {} (attribute {} and queued <= 0.6 x slot)",
        if build { "yes" } else { "no" },
        if wait_nor != 0 && mem_ops != 0 {
            "set"
        } else {
            "unset"
        }
    );
}

/// The slot's host words, for the probe's flag (the slot hands out only its
/// card address and its own reads).
fn slot_host(slot: &MappedSlot) -> *mut u64 {
    slot.host_words_for_tests()
}
