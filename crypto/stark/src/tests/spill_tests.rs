//! Spilled main traces ([`crate::spill`]) change nothing a verifier can see:
//! the packed words written after (or before) a trace's Round-1 commit come
//! back bit for bit, checked against their digest, and the proof is the
//! resident one's byte for byte. A slot that does not come back refuses the
//! proof; a store that cannot write leaves its traces resident.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;
use math::field::{
    extensions_goldilocks::Degree3GoldilocksExtensionField, goldilocks::GoldilocksField,
};

use super::residency_mode_tests::traces;
use crate::examples::multi_table_lookup::{
    new_add_air_with_lookup, new_cpu_air_with_lookup, new_mul_air_with_lookup,
};
use crate::narrow::NarrowMain;
use crate::proof::options::ProofOptions;
use crate::proof::stark::MultiProof;
use crate::prover::{IsStarkProver, Prover, ProvingError};
use crate::residency_mode::ResidencyMode;
use crate::spill::{DirectIo, Prefetch, ReadPhase, SpillError, SpillOptions, SpillStore};
use crate::trace::TraceTable;
use crate::traits::AIR;
use crate::verifier::{IsStarkVerifier, Verifier};

type F = GoldilocksField;
type E = Degree3GoldilocksExtensionField;

const RESIDENCIES: [ResidencyMode; 3] = [
    ResidencyMode::Retain,
    ResidencyMode::RecomputeLde,
    ResidencyMode::RecomputeLdeDevice,
];

fn store(direct: DirectIo) -> SpillStore {
    SpillStore::open(SpillOptions {
        direct,
        ..SpillOptions::default()
    })
    .expect("a spill store in the default directory")
}

/// Row-major words whose column `c` tops out at `caps[c]` in the last row,
/// so the columns pack at the widths the caps need.
fn words(rows: usize, caps: &[u64]) -> Vec<u64> {
    let cols = caps.len();
    (0..rows * cols)
        .map(|i| {
            let (r, c) = (i / cols, i % cols);
            if r == rows - 1 {
                caps[c]
            } else {
                (r as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) % (caps[c] / 2 + 1)
            }
        })
        .collect()
}

fn packed_trace(rows: usize, caps: &[u64]) -> (TraceTable<F, E>, Vec<u64>) {
    let words = words(rows, caps);
    let elements = words
        .iter()
        .map(|&w| FieldElement::<F>::from_raw(w))
        .collect();
    let mut trace = TraceTable::<F, E>::new_main(elements, caps.len(), 1);
    assert!(trace.pack_main_narrow());
    (trace, words)
}

fn main_words(trace: &TraceTable<F, E>) -> Vec<u64> {
    let (data, _) = trace.main_data_row_major();
    data.iter().map(|e| *e.value()).collect()
}

/// ★ Every width mix, odd row counts and a trace longer than one bounce
/// copy round trip bit for bit, with `O_DIRECT` where the probe allows it
/// and with the buffered fallback forced.
#[test]
fn a_spilled_trace_round_trips_bit_for_bit() {
    let edges = [
        0u64,
        0xff,
        0x100,
        0xffff,
        0x1_0000,
        0xffff_ffff,
        0x1_0000_0000,
        0xffff_ffff_0000_0000,
        u64::MAX,
    ];
    let shapes: [(usize, &[u64]); 6] = [
        (1, &[0xff]),
        (3, &[0xffff, 0xff]),
        (4097, &[u64::MAX, 0, 0xffff_ffff]),
        (4099, &edges),
        (65_537, &[1, 0x100, 0x1_0000]),
        // 38 bytes a row: past the 4 MiB bounce buffer.
        (150_001, &edges),
    ];
    for direct in [DirectIo::Auto, DirectIo::Off] {
        let store = store(direct);
        if direct == DirectIo::Off {
            assert!(!store.is_direct(), "the buffered fallback is forced");
        }
        let mut spilled = Vec::new();
        for (rows, caps) in shapes {
            let (mut trace, words) = packed_trace(rows, caps);
            let narrow = trace.narrow_main().unwrap().clone();
            assert!(trace.spill_main(&store), "{rows} rows: spilled");
            assert!(trace.is_main_spilled() && !trace.is_main_narrow());
            assert_eq!(trace.num_rows(), rows);
            assert_eq!(trace.main_table.width, caps.len());
            spilled.push((trace, narrow, words));
        }
        store.flush();
        let stats = store.stats();
        assert_eq!(stats.written, shapes.len() as u64, "{direct:?}: {stats}");
        for (mut trace, narrow, words) in spilled {
            // The writer took the digest of the bytes it wrote.
            assert_eq!(
                trace.spilled_main().unwrap().digest(),
                Some(narrow.digest())
            );
            // A copy first (the slot stays), then the last read.
            assert_eq!(trace.spilled_main().unwrap().load().unwrap(), narrow);
            trace.unspill_main().unwrap();
            assert!(!trace.is_main_spilled());
            assert_eq!(trace.narrow_main().unwrap(), &narrow, "{direct:?}");
            trace.widen_main_on_host();
            assert!(main_words(&trace) == words, "{direct:?}: words moved");
        }
        let stats = store.stats();
        assert_eq!(stats.mismatches, 0);
        assert_eq!(stats.bytes_read, 2 * stats.bytes_written, "{stats}");
        eprintln!("{direct:?}: {stats}");
    }
}

/// An empty packed trace spills to an empty slot and comes back.
#[test]
fn an_empty_trace_spills() {
    let store = store(DirectIo::Auto);
    let narrow = NarrowMain::pack(&[], 3);
    let spilled = store.spill(narrow.clone()).expect("the store is up");
    store.flush();
    assert!(spilled.is_empty());
    assert_eq!(spilled.load().unwrap(), narrow);
}

/// Only a packed trace whose packed copy nothing else holds is spilled; the
/// others are left as they were.
#[test]
fn only_an_unshared_packed_trace_spills() {
    let store = store(DirectIo::Auto);
    let (mut wide, _) = packed_trace(16, &[0xff]);
    wide.widen_main_on_host();
    assert!(!wide.spill_main(&store), "not packed");
    let (mut shared, _) = packed_trace(16, &[0xff]);
    let other = shared.clone();
    assert!(!shared.spill_main(&store), "the packed copy is shared");
    assert!(shared.is_main_narrow() && other.is_main_narrow());
    assert_eq!(store.stats().slots, 0);
}

/// ★ One byte flipped in the file is refused by the digest.
#[test]
fn a_byte_flipped_on_disk_is_refused() {
    for direct in [DirectIo::Auto, DirectIo::Off] {
        let store = store(direct);
        let (mut trace, _) = packed_trace(4099, &[0xffff, 0xff, u64::MAX]);
        assert!(trace.spill_main(&store));
        let spilled = trace.spilled_main().unwrap();
        assert!(store.corrupt_on_disk(spilled, spilled.len() / 2).unwrap());
        assert!(matches!(spilled.load(), Err(SpillError::Mismatch)));
        assert!(matches!(trace.unspill_main(), Err(SpillError::Mismatch)));
        assert_eq!(store.stats().mismatches, 2, "{direct:?}");
    }
}

/// ★ A store on a tmpfs or ramfs directory is refused: it would be RAM.
#[test]
fn a_ram_backed_directory_is_refused() {
    let refused = SpillStore::open(SpillOptions {
        treat_dir_as_ram: true,
        ..SpillOptions::default()
    });
    match refused {
        Err(e) => assert_eq!(e.kind(), std::io::ErrorKind::Unsupported, "{e}"),
        Ok(_) => panic!("a RAM-backed directory was taken"),
    }
    // A real tmpfs where the host has one.
    #[cfg(target_os = "linux")]
    if std::path::Path::new("/dev/shm").is_dir() {
        let refused = SpillStore::open(SpillOptions {
            dir: Some("/dev/shm".into()),
            ..SpillOptions::default()
        });
        assert!(refused.is_err(), "/dev/shm (tmpfs) was taken");
    }
}

/// ★ ENOSPC: the write that fails fails the store, its trace's bytes stay in
/// memory and come back from there, and later spills are declined with the
/// trace untouched.
#[test]
fn a_failed_write_keeps_the_trace_resident() {
    let store = store(DirectIo::Auto);
    store.fail_writes_after(1);
    let (mut a, a_words) = packed_trace(1000, &[0xffff, 7]);
    let (mut b, b_words) = packed_trace(1001, &[u64::MAX, 0]);
    let (mut c, _) = packed_trace(1002, &[0xff]);
    assert!(a.spill_main(&store));
    store.flush();
    assert!(store.failure().is_none(), "the first write succeeds");
    assert!(b.spill_main(&store));
    store.flush();
    let failure = store.failure().expect("the second write fails");
    assert!(failure.contains("os error"), "{failure}");
    assert!(!c.spill_main(&store), "a failed store declines");
    assert!(c.is_main_narrow(), "and leaves the trace packed in memory");
    for (mut trace, words) in [(a, a_words), (b, b_words)] {
        trace.widen_main_on_host();
        assert!(main_words(&trace) == words);
    }
    let stats = store.stats();
    assert_eq!((stats.written, stats.memory_reads), (1, 1), "{stats}");
}

/// The writers' seconds split into their steps: the digest, the aligned copy
/// (`O_DIRECT` only) and the write calls add up to no more than the whole,
/// and every written slot's bytes leave memory.
#[test]
fn the_writers_seconds_split_into_their_steps() {
    let shapes: [(usize, &[u64]); 2] = [
        (4099, &[u64::MAX, 0xff, 0xffff]),
        // 38 bytes a row: past the 4 MiB bounce buffer.
        (150_001, &[u64::MAX, 0, 0xffff_ffff, 0x1_0000, 0xff]),
    ];
    for direct in [DirectIo::Auto, DirectIo::Off] {
        let store = store(direct);
        let mut spilled = Vec::new();
        for (rows, caps) in shapes {
            let (mut trace, _) = packed_trace(rows, caps);
            assert!(trace.spill_main(&store), "{rows} rows: spilled");
            spilled.push(trace);
        }
        store.flush();
        let stats = store.stats();
        assert_eq!(stats.written, shapes.len() as u64, "{stats}");
        let steps = stats.digest_secs + stats.copy_secs + stats.pwrite_secs;
        assert!(
            stats.pwrite_secs > 0.0 && steps <= stats.write_secs,
            "{stats}"
        );
        assert_eq!(stats.copy_secs > 0.0, store.is_direct(), "{stats}");
        for trace in &spilled {
            assert!(!trace.spilled_main().unwrap().is_resident(), "{stats}");
        }
        eprintln!("{direct:?}: {stats}");
    }
}

/// The read-back holds at most its window ahead of the drivers (one read
/// larger than the window goes alone), hands each read over once, and skips
/// a read a driver took before it got there.
#[test]
fn the_read_back_keeps_to_its_window() {
    let store = store(DirectIo::Auto);
    let mut traces = Vec::new();
    for rows in [100, 200, 300, 400] {
        let (mut trace, _) = packed_trace(rows, &[0xffff]);
        let narrow = trace.narrow_main().unwrap().clone();
        assert!(trace.spill_main(&store));
        traces.push((trace, narrow));
    }
    store.flush();
    let reads = traces
        .iter()
        .enumerate()
        .map(|(i, (t, _))| (ReadPhase::Fused, i, t.spilled_main().unwrap().clone()))
        .collect();
    // One byte: every read is larger, so each goes alone.
    let prefetch = Prefetch::start(reads, 1);
    prefetch.wait(ReadPhase::Fused, 0);
    std::thread::sleep(std::time::Duration::from_millis(50));
    assert!(
        !prefetch.is_ready(ReadPhase::Fused, 1),
        "read 1 waits for read 0 to be taken"
    );
    let first = prefetch.take(ReadPhase::Fused, 0).unwrap().unwrap();
    assert_eq!(first, traces[0].1);
    assert!(prefetch.take(ReadPhase::Fused, 0).is_none(), "taken once");
    // A read taken before the reader reached it is the driver's to make.
    assert!(prefetch.take(ReadPhase::Fused, 2).is_none());
    prefetch.wait(ReadPhase::Fused, 1);
    assert_eq!(
        prefetch.take(ReadPhase::Fused, 1).unwrap().unwrap(),
        traces[1].1
    );
    prefetch.wait(ReadPhase::Fused, 3);
    assert_eq!(
        prefetch.take(ReadPhase::Fused, 3).unwrap().unwrap(),
        traces[3].1
    );
    // Not planned: ready at once, nothing to take.
    assert!(prefetch.is_ready(ReadPhase::Round1, 0));
    assert!(prefetch.take(ReadPhase::Round1, 0).is_none());
    drop(prefetch);
    // The skipped read is still on disk for its driver.
    let (mut skipped, narrow) = traces.swap_remove(2);
    skipped.unspill_main().unwrap();
    assert_eq!(skipped.narrow_main().unwrap(), &narrow);
}

/// Dropping the read-back while it waits for room stops it.
#[test]
fn a_blocked_read_back_stops_when_dropped() {
    let store = store(DirectIo::Auto);
    let mut reads = Vec::new();
    let mut keep = Vec::new();
    for i in 0..3 {
        let (mut trace, _) = packed_trace(64, &[0xff]);
        assert!(trace.spill_main(&store));
        reads.push((ReadPhase::Fused, i, trace.spilled_main().unwrap().clone()));
        keep.push(trace);
    }
    store.flush();
    let prefetch = Prefetch::start(reads, 1);
    prefetch.wait(ReadPhase::Fused, 0);
    drop(prefetch);
}

/// Where a spilled trace is in the prove.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Spill {
    /// Not spilled (packed or not, as `pack` says).
    No,
    /// Packed and spilled after its precommit (the block's streamed tables).
    AfterPrecommit,
    /// Packed and spilled before Round 1, which commits it (the block's
    /// finish-built tables).
    BeforeRoundOne,
}

fn test_options() -> ProofOptions {
    ProofOptions {
        grinding_factor: 0,
        ..ProofOptions::default_test_options()
    }
}

/// The CPU/ADD/MUL instance proved the block's way, its traces spilled to
/// `store` as `spill` says; `tamper` sees the traces before the prove.
fn prove_spilled(
    residency: ResidencyMode,
    spill: Spill,
    pack: bool,
    store: Option<&SpillStore>,
    tamper: impl FnOnce(&[&TraceTable<F, E>; 3]),
) -> Result<MultiProof<F, E, ()>, ProvingError> {
    let (mut cpu_trace, mut add_trace, mut mul_trace) = traces();
    let proof_options = test_options();
    let cpu_air = new_cpu_air_with_lookup(&proof_options);
    let add_air = new_add_air_with_lookup(&proof_options);
    let mul_air = new_mul_air_with_lookup(&proof_options);
    let mut precommitted = Vec::new();
    for (air, trace) in [
        (
            &cpu_air as &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
            &mut cpu_trace,
        ),
        (&add_air, &mut add_trace),
        (&mul_air, &mut mul_trace),
    ] {
        // Packed first, as the block's generators pack each streamed chunk
        // before its committer precommits and then spills it.
        if pack || spill != Spill::No {
            assert!(trace.pack_main_narrow());
        }
        if spill != Spill::BeforeRoundOne {
            precommitted.push(Some(Prover::precommit_main(
                air,
                trace,
                #[cfg(feature = "disk-spill")]
                crate::storage_mode::StorageMode::Ram,
                residency,
            )?));
        }
        if spill != Spill::No {
            // A failed store declines, and the trace stays packed.
            let store = store.expect("a store to spill to");
            let spilled = trace.spill_main(store);
            assert!(spilled || store.failure().is_some());
        }
    }
    if let Some(store) = store {
        store.flush();
    }
    tamper(&[&cpu_trace, &add_trace, &mul_trace]);
    let pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&cpu_air, &mut cpu_trace, &()),
        (&add_air, &mut add_trace, &()),
        (&mul_air, &mut mul_trace, &()),
    ];
    Prover::multi_prove_precommitted(
        pairs,
        &mut DefaultTranscript::<E>::new(&[]),
        #[cfg(feature = "disk-spill")]
        crate::storage_mode::StorageMode::Ram,
        residency,
        precommitted,
    )
}

fn resident(residency: ResidencyMode) -> Vec<u8> {
    bytes(&prove_spilled(residency, Spill::No, false, None, |_| {}).unwrap())
}

fn bytes(proof: &MultiProof<F, E, ()>) -> Vec<u8> {
    bincode::serialize(proof).unwrap()
}

fn verifies(proof: &MultiProof<F, E, ()>) -> bool {
    let proof_options = test_options();
    let cpu_air = new_cpu_air_with_lookup(&proof_options);
    let add_air = new_add_air_with_lookup(&proof_options);
    let mul_air = new_mul_air_with_lookup(&proof_options);
    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];
    Verifier::multi_verify(
        &airs,
        proof,
        &mut DefaultTranscript::<E>::new(&[]),
        &FieldElement::zero(),
    )
}

fn air_names() -> [String; 3] {
    let o = test_options();
    [
        new_cpu_air_with_lookup(&o).name().to_string(),
        new_add_air_with_lookup(&o).name().to_string(),
        new_mul_air_with_lookup(&o).name().to_string(),
    ]
}

/// ★ Every trace spilled — after its precommit, or before Round 1 — proves
/// the resident bytes under every residency, read back from the file
/// through the read-back (and its digest checks), with `O_DIRECT` where the
/// probe allows it and buffered.
#[test]
fn every_trace_spilled_proves_the_resident_bytes() {
    for residency in RESIDENCIES {
        let want = resident(residency);
        for spill in [Spill::AfterPrecommit, Spill::BeforeRoundOne] {
            for direct in [DirectIo::Auto, DirectIo::Off] {
                let store = store(direct);
                let got =
                    bytes(&prove_spilled(residency, spill, true, Some(&store), |_| {}).unwrap());
                let stats = store.stats();
                assert!(
                    want == got,
                    "{residency:?} {spill:?} {direct:?}: proof bytes moved ({stats})"
                );
                assert_eq!(stats.written, 3, "{stats}");
                // Before Round 1 each trace is read twice: its commit's copy,
                // then its fused task's.
                let reads = if spill == Spill::BeforeRoundOne { 6 } else { 3 };
                assert_eq!(stats.reads, reads, "{residency:?} {spill:?}: {stats}");
                assert_eq!(stats.mismatches, 0);
            }
        }
    }
}

/// ★ A slow disk: every read sleeps first, and the drivers wait for theirs
/// before admission; the proof is the resident one's.
#[test]
fn a_slow_read_back_proves_the_resident_bytes() {
    let want = resident(ResidencyMode::RecomputeLdeDevice);
    let store = store(DirectIo::Auto);
    store.set_read_delay_ms(150);
    let got = prove_spilled(
        ResidencyMode::RecomputeLdeDevice,
        Spill::BeforeRoundOne,
        true,
        Some(&store),
        |_| {},
    )
    .unwrap();
    assert!(want == bytes(&got));
}

/// ★ ENOSPC on the second write: the store fails, the unwritten traces stay
/// in memory (or are never spilled), and the prove completes with the
/// resident bytes.
#[test]
fn a_failed_store_proves_the_resident_bytes() {
    for spill in [Spill::AfterPrecommit, Spill::BeforeRoundOne] {
        let want = resident(ResidencyMode::RecomputeLdeDevice);
        let store = store(DirectIo::Auto);
        store.fail_writes_after(1);
        let got = prove_spilled(
            ResidencyMode::RecomputeLdeDevice,
            spill,
            true,
            Some(&store),
            |_| {},
        )
        .unwrap();
        assert!(store.failure().is_some(), "{spill:?}: the store failed");
        assert!(want == bytes(&got), "{spill:?}: proof bytes moved");
        assert_eq!(store.stats().written, 1);
    }
}

/// ★ One byte flipped on disk in any table's slot refuses the proof with
/// `SpilledTraceMismatch` naming that table, under every residency and at
/// both spill points.
#[test]
fn a_corrupted_slot_refuses_the_proof() {
    let names = air_names();
    for residency in RESIDENCIES {
        for spill in [Spill::AfterPrecommit, Spill::BeforeRoundOne] {
            for table in 0..3 {
                let store = store(DirectIo::Auto);
                let out = prove_spilled(residency, spill, true, Some(&store), |traces| {
                    let spilled = traces[table].spilled_main().unwrap();
                    assert!(store.corrupt_on_disk(spilled, spilled.len() / 2).unwrap());
                });
                match out {
                    Err(ProvingError::SpilledTraceMismatch(name)) => {
                        assert_eq!(name, names[table], "{residency:?} {spill:?}")
                    }
                    Err(e) => panic!("{residency:?} {spill:?} table {table}: wrong refusal {e:?}"),
                    Ok(_) => panic!("{residency:?} {spill:?} table {table}: a bad slot was proved"),
                }
            }
        }
    }
}

/// ★ Negative, the digest off: a slot changed on disk after its trace's
/// Round-1 commit is never a proof the verifier accepts unless it is the
/// resident proof itself (a word no stage of phase B reads). Where phase B
/// recomputes the main LDE from the spilled words (`RecomputeLde`, and
/// `RecomputeLdeDevice` on a host build), every perturbation is caught: the
/// openings no longer match the absorbed tree.
#[test]
fn a_perturbed_slot_without_the_digest_is_never_accepted() {
    for residency in RESIDENCIES {
        let want = resident(residency);
        let (mut refused, mut rejected) = (0, 0);
        for table in 0..3 {
            let store = store(DirectIo::Auto);
            store.set_verify(false);
            let out = prove_spilled(
                residency,
                Spill::AfterPrecommit,
                true,
                Some(&store),
                |traces| {
                    let spilled = traces[table].spilled_main().unwrap();
                    assert!(store.corrupt_on_disk(spilled, spilled.len() / 2).unwrap());
                },
            );
            match out {
                Err(_) => refused += 1,
                Ok(proof) if !verifies(&proof) => rejected += 1,
                Ok(proof) => assert!(
                    bytes(&proof) == want,
                    "{residency:?} table {table}: a perturbed trace proved and verified"
                ),
            }
        }
        eprintln!("{residency:?}: refused {refused}, rejected {rejected} of 3");
        if residency == ResidencyMode::RecomputeLde {
            assert_eq!(refused + rejected, 3, "{residency:?}");
        }
    }
}

/// ★ A trace is its packed copy's only owner right after its precommit, so
/// the block's committer can spill it there: packed before the precommit
/// (the generators' order) or after it, under every residency.
#[test]
fn a_just_precommitted_trace_spills() {
    let o = test_options();
    let air = new_cpu_air_with_lookup(&o);
    for residency in RESIDENCIES {
        for pack_first in [true, false] {
            let store = store(DirectIo::Auto);
            let (mut trace, _, _) = traces();
            if pack_first {
                assert!(trace.pack_main_narrow());
            }
            let pre = Prover::precommit_main(
                &air,
                &trace,
                #[cfg(feature = "disk-spill")]
                crate::storage_mode::StorageMode::Ram,
                residency,
            )
            .unwrap();
            if !pack_first {
                assert!(trace.pack_main_narrow());
            }
            assert!(
                trace.spill_main(&store),
                "{residency:?} pack first {pack_first}: the precommit kept a hold on the packed trace"
            );
            drop(pre);
        }
    }
}

/// The prove of [`prove_spilled`] on a thread of its own under `overrides`,
/// within `secs` seconds (a hang is a failure, not a stuck test run).
fn prove_spilled_within(
    secs: u64,
    overrides: crate::spill::test_hooks::ProveOverrides,
    residency: ResidencyMode,
    spill: Spill,
    read_delay_ms: u64,
) -> (
    Result<MultiProof<F, E, ()>, ProvingError>,
    crate::spill::test_hooks::Admissions,
    crate::spill::SpillStats,
) {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let store = store(DirectIo::Auto);
        store.set_read_delay_ms(read_delay_ms);
        let (out, admissions) = crate::spill::test_hooks::with_prove_overrides(overrides, || {
            prove_spilled(residency, spill, true, Some(&store), |_| {})
        });
        let _ = tx.send((out, admissions, store.stats()));
    });
    rx.recv_timeout(std::time::Duration::from_secs(secs))
        .expect("the prove hung")
}

/// ★ A slow disk with the tightest admission: a VRAM budget that admits one
/// table at a time, a read-back window of one read, three drivers and 100 ms
/// per read. Every driver waits for its trace before its permit (checked
/// where it waits: it holds none), the read-back hands the reads over in walk
/// order, and so both phases start their tables in walk order. Were a wait
/// inside a permit, the driver holding it would wait on a read stuck behind
/// one only a permit can take: a hang.
#[test]
fn a_slow_read_back_admits_in_walk_order_holding_no_permit() {
    let overrides = crate::spill::test_hooks::ProveOverrides {
        vram_budget: Some(1),
        window: Some(1),
        drivers: Some(3),
        ..Default::default()
    };
    for residency in RESIDENCIES {
        let want = resident(residency);
        let (out, admissions, stats) =
            prove_spilled_within(120, overrides, residency, Spill::BeforeRoundOne, 100);
        let got = bytes(&out.unwrap());
        assert!(want == got, "{residency:?}: proof bytes moved ({stats})");
        assert_eq!(stats.reads, 6, "{residency:?}: {stats}");
        assert_eq!(
            admissions.r1_started, admissions.r1_walk,
            "{residency:?}: Round 1 out of walk order"
        );
        assert_eq!(
            admissions.fused_started, admissions.fused_walk,
            "{residency:?}: the fused phase out of walk order"
        );
    }
}

/// ★ On the card with kept top levels: every trace spilled (after its
/// precommit, or before Round 1) is read back, widened on the device in its
/// fused task, and proves `Retain`'s bytes.
#[cfg(feature = "cuda")]
#[test_log::test]
#[ignore = "requires a GPU, LAMBDA_VM_GPU_LDE_THRESHOLD=2 and LAMBDA_VM_RECOMMIT_TOP_LEVELS=2; run alone"]
fn every_trace_spilled_widens_on_the_card() {
    use std::sync::atomic::Ordering;
    let want = resident(ResidencyMode::Retain);
    for spill in [Spill::AfterPrecommit, Spill::BeforeRoundOne] {
        for direct in [DirectIo::Auto, DirectIo::Off] {
            let store = store(direct);
            let before = crate::prover::NARROW_DEVICE_WIDENS.load(Ordering::SeqCst);
            let got = prove_spilled(
                ResidencyMode::RecomputeLdeDevice,
                spill,
                true,
                Some(&store),
                |_| {},
            )
            .unwrap();
            let widens = crate::prover::NARROW_DEVICE_WIDENS.load(Ordering::SeqCst) - before;
            let stats = store.stats();
            eprintln!("{spill:?} {direct:?}: {stats}");
            assert_eq!(widens, 3, "{spill:?}: every table widens on the device");
            assert!(
                want == bytes(&got),
                "{spill:?} {direct:?}: proof bytes moved"
            );
            assert_eq!(stats.written, 3);
        }
    }
}

/// ★ On the card: every table's slot corrupted, and the prove is refused with
/// `SpilledTraceMismatch` before any table's phase-B device work (no
/// recompute, widen or recommit counted).
#[cfg(feature = "cuda")]
#[test_log::test]
#[ignore = "requires a GPU, LAMBDA_VM_GPU_LDE_THRESHOLD=2 and LAMBDA_VM_RECOMMIT_TOP_LEVELS=2; run alone"]
fn a_corrupted_slot_is_refused_before_device_work_on_the_card() {
    use std::sync::atomic::Ordering;
    let counters = || {
        (
            crate::prover::NARROW_DEVICE_WIDENS.load(Ordering::SeqCst),
            crate::prover::TOP_TREE_RECOMPUTES.load(Ordering::SeqCst),
            crate::residency_mode::DEVICE_RECOMMITS.load(Ordering::SeqCst),
        )
    };
    let store = store(DirectIo::Auto);
    let mut before = None;
    let out = prove_spilled(
        ResidencyMode::RecomputeLdeDevice,
        Spill::AfterPrecommit,
        true,
        Some(&store),
        |traces| {
            for trace in traces {
                let spilled = trace.spilled_main().unwrap();
                assert!(store.corrupt_on_disk(spilled, spilled.len() / 2).unwrap());
            }
            before = Some(counters());
        },
    );
    assert!(
        matches!(out, Err(ProvingError::SpilledTraceMismatch(_))),
        "{out:?}"
    );
    assert_eq!(Some(counters()), before, "no phase-B device work");
}

/// ★ On the card, the digest off: a slot changed after its precommit is
/// refused by the kept-top check (or rejected by the verifier), never
/// accepted.
#[cfg(feature = "cuda")]
#[test_log::test]
#[ignore = "requires a GPU, LAMBDA_VM_GPU_LDE_THRESHOLD=2 and LAMBDA_VM_RECOMMIT_TOP_LEVELS=2; run alone"]
fn a_perturbed_slot_without_the_digest_is_refused_on_the_card() {
    for table in 0..3 {
        let store = store(DirectIo::Auto);
        store.set_verify(false);
        let out = prove_spilled(
            ResidencyMode::RecomputeLdeDevice,
            Spill::AfterPrecommit,
            true,
            Some(&store),
            |traces| {
                let spilled = traces[table].spilled_main().unwrap();
                assert!(store.corrupt_on_disk(spilled, spilled.len() / 2).unwrap());
            },
        );
        match out {
            Err(ProvingError::RecomputedCommitmentMismatch(_)) => {}
            Ok(proof) => assert!(!verifies(&proof), "table {table}: accepted"),
            Err(e) => panic!("table {table}: unexpected refusal {e:?}"),
        }
    }
}

/// ★ On the card: a slow disk with the tightest admission (one table at a
/// time, one read ahead, three drivers, 100 ms per read): both phases start
/// in walk order, no driver waits holding a permit, every table widens on the
/// device, and the proof is `Retain`'s.
#[cfg(feature = "cuda")]
#[test_log::test]
#[ignore = "requires a GPU, LAMBDA_VM_GPU_LDE_THRESHOLD=2 and LAMBDA_VM_RECOMMIT_TOP_LEVELS=2; run alone"]
fn a_slow_read_back_admits_in_walk_order_on_the_card() {
    use std::sync::atomic::Ordering;
    let want = resident(ResidencyMode::Retain);
    let overrides = crate::spill::test_hooks::ProveOverrides {
        vram_budget: Some(1),
        window: Some(1),
        drivers: Some(3),
        ..Default::default()
    };
    for spill in [Spill::AfterPrecommit, Spill::BeforeRoundOne] {
        let before = crate::prover::NARROW_DEVICE_WIDENS.load(Ordering::SeqCst);
        let (out, admissions, stats) = prove_spilled_within(
            300,
            overrides,
            ResidencyMode::RecomputeLdeDevice,
            spill,
            100,
        );
        let widens = crate::prover::NARROW_DEVICE_WIDENS.load(Ordering::SeqCst) - before;
        eprintln!("{spill:?} slow: {stats} · {admissions:?}");
        assert!(want == bytes(&out.unwrap()), "{spill:?}: proof bytes moved");
        assert_eq!(widens, 3, "{spill:?}: every table widens on the device");
        assert_eq!(admissions.fused_started, admissions.fused_walk, "{spill:?}");
        if spill == Spill::BeforeRoundOne {
            assert_eq!(admissions.r1_started, admissions.r1_walk);
        }
    }
}

/// ★ On the card: a chunk packed by the device after its precommit
/// (`take_narrow`, as the committers install it) is its packed copy's only
/// owner, so it spills.
#[cfg(feature = "cuda")]
#[test_log::test]
#[ignore = "requires a GPU, LAMBDA_VM_GPU_LDE_THRESHOLD=2 and LAMBDA_VM_RECOMMIT_TOP_LEVELS=2; run alone"]
fn a_trace_packed_by_the_device_after_its_precommit_spills() {
    crate::prover::set_default_pack_after_commit(true);
    let o = test_options();
    let air = new_cpu_air_with_lookup(&o);
    let store = store(DirectIo::Auto);
    let (mut trace, _, _) = traces();
    let mut pre = Prover::precommit_main(
        &air,
        &trace,
        #[cfg(feature = "disk-spill")]
        crate::storage_mode::StorageMode::Ram,
        ResidencyMode::RecomputeLdeDevice,
    )
    .unwrap();
    crate::prover::set_default_pack_after_commit(false);
    let narrow = pre.take_narrow().expect("the device packed the trace");
    assert!(trace.install_main_narrow(narrow));
    assert!(trace.spill_main(&store), "the precommit kept a hold");
}
