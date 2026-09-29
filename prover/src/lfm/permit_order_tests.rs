//! The card permit's order moves no proof byte (`LFM_CARD_PERMIT_ORDER`).
//!
//! `device_permit`'s own tests show the gate serves in the order it claims,
//! bounds how often a prove is overtaken, and never lets two holders in. What
//! they cannot show is that a PROOF does not depend on which proof held the
//! card before it. Device state that one holder leaves to the next — a resident
//! prepared commit, a cached table, the free memory a `query` budget reads — is
//! exactly the coupling a new order could expose. So these tests prove real
//! programs in one process: serially, then `workers` at a time under `fifo` and
//! under `commits-first`, and once with the queue FORCED so that a late commit
//! is served ahead of a waiting prove. Every proof must equal the serial one,
//! byte for byte.
//!
//! The programs are the continuation fixture's W-LFM wraps, one per epoch, and
//! one wide node over all of them (`whir_epoch_program_tests::driver_bundle`).
//! Each test proves them with one prover: the W-LFM one (`WhirTreeChild`) or the
//! per-table STARK one (`RealChild`), the two trees the permit serves.
//!
//! ⚠ RUN WITH `LAMBDA_VM_DETERMINISTIC_GRIND=1`. A grinding search returns *a*
//! valid nonce, and the nonce is absorbed into the transcript, so two proves of
//! one program can differ for a reason that has nothing to do with the permit.
//! The serial control below proves every program twice and says so if they
//! differ.
//!
//! Fixture traces and the card: box tier, never the laptop.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use super::compiler::LfmProgram;
use super::device_permit::{self, Order, PermitStats};
use super::per_table_aggregator::NodePublishSet;
use super::per_table_aggregator_tests::{RealChild, TreeChild, WhirTreeChild};
use super::whir_epoch::{whir_epoch_arena, whir_epoch_program};
use super::whir_wide::{WideEpoch, wide_node_arena, wide_node_program};
use super::word::LfmWord;

/// A program to prove: its label, the program, and its arenas.
type Job = (String, LfmProgram, Vec<Vec<LfmWord>>);

/// The fixture's wraps, one per epoch, and a wide node over every epoch.
fn fixture_jobs() -> Vec<Job> {
    let (elf_bytes, inner, bundle) = super::whir_epoch_program_tests::driver_bundle();
    let elf = executor::elf::Elf::load(&elf_bytes).expect("the inner ELF loads");
    let held: Vec<_> = (0..bundle.epochs.len())
        .map(|k| {
            let epoch = crate::lfm::whir_real_epoch::real_epoch_from_whir_continuation_under::<
                multilinear::whir_hash::RpxWhir,
            >(&inner, &elf_bytes, &bundle, k, None, None)
            .unwrap_or_else(|why| panic!("epoch {k} must harvest: {why}"));
            let airs = crate::multilinear_continuation::epoch_airs_for(
                &elf,
                &inner,
                &bundle.epochs[k],
                &epoch.position.register_init,
                epoch.position.is_final,
                epoch.position.label,
                Some(epoch.decode_commitment),
            );
            (epoch, airs)
        })
        .collect();
    let refs: Vec<Vec<_>> = held.iter().map(|(_, airs)| airs.refs()).collect();
    let mut jobs: Vec<Job> = held
        .iter()
        .zip(&refs)
        .enumerate()
        .map(|(k, ((epoch, _), refs))| {
            (
                format!("wrap {k}"),
                whir_epoch_program(epoch, &refs[..]),
                whir_epoch_arena(epoch, &refs[..]),
            )
        })
        .collect();
    let wide: Vec<WideEpoch<'_>> = held
        .iter()
        .zip(&refs)
        .map(|((epoch, _), refs)| WideEpoch {
            epoch,
            airs: &refs[..],
        })
        .collect();
    let positions: Vec<u64> = (0..held.len())
        .map(|k| crate::tables::local_to_global::epoch_label(k as u64))
        .collect();
    jobs.push((
        format!("wide 0..{}", held.len()),
        wide_node_program(&wide, &positions, NodePublishSet::Aggregation),
        wide_node_arena(&wide),
    ));
    jobs
}

/// Build and prove one job, and serialize its proof.
fn prove_one<C: TreeChild>(
    job: &Job,
    opts: &crate::ProofOptions,
    bytes: fn(&C::Proved) -> Vec<u8>,
) -> Vec<u8> {
    let (label, program, arenas) = job;
    let built = C::build(program, opts);
    let proved = C::prove(label, program, &built, arenas, opts).unwrap_or_else(|e| panic!("{e}"));
    bytes(&proved)
}

/// Arms the permit for `workers` under `order` while it lives, and hands back
/// the environment's order and a disarmed permit when dropped.
struct Armed;

impl Armed {
    fn new(order: Order, workers: usize) -> Self {
        device_permit::set_order_for_tests(Some(order));
        device_permit::arm(workers);
        let _ = device_permit::take_stats();
        assert_eq!(device_permit::order(), order);
        Armed
    }
}

impl Drop for Armed {
    fn drop(&mut self) {
        device_permit::set_order_for_tests(None);
        device_permit::arm(1);
    }
}

/// Every job, `workers` at a time under `order`, each twice: a worker takes the
/// next job, builds it and proves it. The workers start 40 ms apart, so commits
/// arrive while earlier proves hold or wait.
fn prove_pool<C: TreeChild>(
    jobs: &[Job],
    opts: &crate::ProofOptions,
    workers: usize,
    order: Order,
    bytes: fn(&C::Proved) -> Vec<u8>,
) -> (Vec<Vec<u8>>, PermitStats) {
    let _armed = Armed::new(order, workers);
    let next = AtomicUsize::new(0);
    let out: Vec<Mutex<Option<Vec<u8>>>> = (0..2 * jobs.len()).map(|_| Mutex::new(None)).collect();
    std::thread::scope(|s| {
        for w in 0..workers {
            let (next, out) = (&next, &out);
            s.spawn(move || {
                std::thread::sleep(Duration::from_millis(40 * w as u64));
                loop {
                    let i = next.fetch_add(1, Ordering::SeqCst);
                    if i >= out.len() {
                        break;
                    }
                    let proof = prove_one::<C>(&jobs[i % jobs.len()], opts, bytes);
                    *out[i].lock().expect("the slot") = Some(proof);
                }
            });
        }
    });
    let stats = device_permit::take_stats();
    let out = out
        .into_iter()
        .map(|slot| {
            slot.into_inner()
                .expect("the slot")
                .expect("every job proved")
        })
        .collect();
    (out, stats)
}

/// Job `a` proves while job `b` commits, with the queue FORCED: `a` is built,
/// then waits to prove behind a holder, then `b` asks to commit, then the
/// holder lets go. Under `commits-first` `b`'s commit is served first — a
/// reorder by construction, not by luck. Returns both proofs and the stats.
fn prove_forced<C: TreeChild>(
    a: &Job,
    b: &Job,
    opts: &crate::ProofOptions,
    order: Order,
    bytes: fn(&C::Proved) -> Vec<u8>,
) -> (Vec<u8>, Vec<u8>, PermitStats) {
    let _armed = Armed::new(order, 3);
    let (a_built_tx, a_built_rx) = std::sync::mpsc::channel::<()>();
    let (go_tx, go_rx) = std::sync::mpsc::channel::<()>();
    let (pa, pb) = std::thread::scope(|s| {
        let ta = s.spawn(move || {
            let (label, program, arenas) = a;
            let built = C::build(program, opts);
            a_built_tx.send(()).expect("the main thread listens");
            go_rx.recv().expect("the main thread says go");
            let proved =
                C::prove(label, program, &built, arenas, opts).unwrap_or_else(|e| panic!("{e}"));
            bytes(&proved)
        });
        a_built_rx.recv().expect("a builds");
        let holder = device_permit::hold_labeled("multi_prove");
        go_tx.send(()).expect("a listens");
        settle(order, 0, 1);
        let tb = s.spawn(move || prove_one::<C>(b, opts, bytes));
        settle(order, 1, 1);
        drop(holder);
        (ta.join().expect("a proves"), tb.join().expect("b proves"))
    });
    (pa, pb, device_permit::take_stats())
}

/// Until the process gate holds `commits` waiting commits and `proves`
/// waiting proves; under `fifo`, whose `Mutex` shows no queue, a fixed wait.
fn settle(order: Order, commits: usize, proves: usize) {
    if order == Order::Fifo {
        std::thread::sleep(Duration::from_millis(300));
        return;
    }
    let t = std::time::Instant::now();
    while device_permit::gate_queue_for_tests() != (commits, proves) {
        assert!(
            t.elapsed() < Duration::from_secs(60),
            "the gate never queued {commits} commit(s) and {proves} prove(s): {:?}",
            device_permit::gate_queue_for_tests()
        );
        std::thread::sleep(Duration::from_millis(2));
    }
}

/// The whole gate for one prover.
fn the_order_moves_no_byte<C: TreeChild>(bytes: fn(&C::Proved) -> Vec<u8>) {
    let _g = device_permit::TEST_ARM
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    assert!(
        crypto::grinding::deterministic(),
        "run with LAMBDA_VM_DETERMINISTIC_GRIND=1: without it two proves of one program may \
         take different grinding nonces, and every byte after the first grind differs"
    );
    let jobs = fixture_jobs();
    let opts = super::proof::aggregation_wrap_options();
    assert!(jobs.len() >= 3, "the fixture has {} jobs", jobs.len());

    // The control: serially, twice. Two proves of one program must agree, or
    // no comparison below means anything.
    device_permit::arm(1);
    let serial: Vec<Vec<u8>> = jobs
        .iter()
        .map(|j| prove_one::<C>(j, &opts, bytes))
        .collect();
    let again: Vec<Vec<u8>> = jobs
        .iter()
        .map(|j| prove_one::<C>(j, &opts, bytes))
        .collect();
    for (k, (x, y)) in serial.iter().zip(&again).enumerate() {
        assert!(
            x == y,
            "{}: two serial proves differ ({} vs {} bytes) — run with \
             LAMBDA_VM_DETERMINISTIC_GRIND=1, or the proof is not a function of its input",
            jobs[k].0,
            x.len(),
            y.len()
        );
    }
    // And the comparison can fail: distinct programs give distinct proofs.
    assert_ne!(
        serial[0],
        serial[jobs.len() - 1],
        "the comparison is not vacuous"
    );

    for order in [Order::Fifo, Order::CommitsFirst] {
        let (pooled, stats) = prove_pool::<C>(&jobs, &opts, 3, order, bytes);
        for (i, proof) in pooled.iter().enumerate() {
            let k = i % jobs.len();
            assert!(
                proof == &serial[k],
                "{order:?}: {} proved {} bytes against the serial {} — the order moved a byte",
                jobs[k].0,
                proof.len(),
                serial[k].len()
            );
        }
        assert_eq!(stats.peak_holders, 1, "{order:?}: one holder at a time");
        println!(
            "PERMIT ORDER {order:?} pool: {} proofs equal the serial ones · {} acquisitions · {} \
             reorder(s) · a prove overtaken at most {} time(s)",
            pooled.len(),
            stats.acquisitions,
            stats.reorders,
            stats.max_overtaken
        );

        let (a, b) = (&jobs[jobs.len() - 1], &jobs[0]);
        let (pa, pb, stats) = prove_forced::<C>(a, b, &opts, order, bytes);
        assert!(
            pa == serial[jobs.len() - 1],
            "{order:?}: the forced {} moved a byte",
            a.0
        );
        assert!(
            pb == serial[0],
            "{order:?}: the forced {} moved a byte",
            b.0
        );
        assert_eq!(stats.peak_holders, 1, "{order:?}: one holder at a time");
        if order == Order::CommitsFirst {
            assert!(
                stats.reorders >= 1,
                "the forced queue must reorder: {} committed ahead of {}'s prove ({stats:?})",
                b.0,
                a.0
            );
        }
        println!(
            "PERMIT ORDER {order:?} forced: {} and {} equal the serial proofs · {} reorder(s)",
            a.0, b.0, stats.reorders
        );
    }
}

#[test]
#[ignore = "fixture traces and the card, under LAMBDA_VM_DETERMINISTIC_GRIND=1: box only"]
fn the_permit_order_moves_no_w_lfm_proof_byte() {
    the_order_moves_no_byte::<WhirTreeChild>(|p| {
        rkyv::to_bytes::<rkyv::rancor::Error>(&p.proof)
            .expect("a W-LFM proof serializes")
            .to_vec()
    });
}

#[test]
#[ignore = "fixture traces and the card, under LAMBDA_VM_DETERMINISTIC_GRIND=1: box only"]
fn the_permit_order_moves_no_stark_lfm_proof_byte() {
    the_order_moves_no_byte::<RealChild>(|p| {
        rkyv::to_bytes::<rkyv::rancor::Error>(&p.proof)
            .expect("a STARK LFM proof serializes")
            .to_vec()
    });
}
