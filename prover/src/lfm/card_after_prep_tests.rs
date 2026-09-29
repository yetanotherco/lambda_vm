//! `LFM_CARD_AFTER_PREP`: the W-LFM prove takes the card permit after its host
//! prep instead of before it (`whir_proof::card_after_prep`).
//!
//! Two claims carry the change, and each has its test here:
//! - **Only the lock moves.** The same programs, proved with the setting off and
//!   on, in one process, give the same bytes. Proved serially and then three at a
//!   time with the permit armed, so the prep really does run while another proof
//!   holds the card.
//! - **The prep never reaches the device.** Every entry into math-cuda's device
//!   layer (`device::backend`, which every device entry point calls) is counted
//!   across `prep_whir_tables`, and none may happen. The paired control counts
//!   the same across the prove that follows and requires some, so the counter can
//!   see this prove's device work.
//!
//! The programs are the continuation fixture's W-LFM wraps, one per epoch, and a
//! wide node over every epoch (`whir_epoch_program_tests::driver_bundle`).
//!
//! ⚠ The byte test runs under `LAMBDA_VM_DETERMINISTIC_GRIND=1`: the grinding
//! nonce is absorbed into the transcript, and a parallel search returns *a*
//! valid nonce. Fixture traces and the card: box tier, never the laptop. Only the
//! knob's parse runs anywhere.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::compiler::LfmProgram;
use super::whir_proof::{
    WhirLfmBuild, build_whir_artifacts, card_after_prep, lfm_prove_whir, parse_card_after_prep,
    test_card_after_prep,
};
use super::word::LfmWord;

/// A program to prove: its label, the program, and its arenas.
type Job = (String, LfmProgram, Vec<Vec<LfmWord>>);

#[test]
fn the_knob_parses_and_a_misspelt_arm_is_refused() {
    assert!(!parse_card_after_prep(None));
    assert!(!parse_card_after_prep(Some("")));
    assert!(!parse_card_after_prep(Some("0")));
    assert!(parse_card_after_prep(Some("1")));
    for bad in ["on", "true", "2", "yes"] {
        assert!(
            std::panic::catch_unwind(|| parse_card_after_prep(Some(bad))).is_err(),
            "`{bad}` must be refused, not read as the control"
        );
    }
}

/// The test override is what [`card_after_prep`] reports while it is set, and
/// the environment's value once it is cleared.
#[test]
fn the_override_sets_and_clears() {
    let _g = OVERRIDE.lock().unwrap_or_else(|e| e.into_inner());
    test_card_after_prep::set(Some(true));
    assert!(card_after_prep());
    test_card_after_prep::set(Some(false));
    assert!(!card_after_prep());
    test_card_after_prep::set(None);
    assert_eq!(
        card_after_prep(),
        parse_card_after_prep(
            std::env::var(super::whir_proof::CARD_AFTER_PREP_ENV)
                .ok()
                .as_deref()
        )
    );
}

/// The override and the permit's arming are process-global.
static OVERRIDE: Mutex<()> = Mutex::new(());

/// Sets the override while it lives; dropping it gives the environment back.
struct Setting;

impl Setting {
    fn new(on: bool) -> Self {
        test_card_after_prep::set(Some(on));
        Setting
    }
}

impl Drop for Setting {
    fn drop(&mut self) {
        test_card_after_prep::set(None);
    }
}

/// The fixture's wraps, one per epoch, and a wide node over every epoch.
fn fixture_jobs() -> Vec<Job> {
    use super::per_table_aggregator::NodePublishSet;
    use super::whir_epoch::{whir_epoch_arena, whir_epoch_program};
    use super::whir_wide::{WideEpoch, wide_node_arena, wide_node_program};

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

fn build(job: &Job, opts: &crate::ProofOptions) -> WhirLfmBuild {
    build_whir_artifacts(&job.1, opts, crate::hash_pin::BLOCK_HASHER)
        .unwrap_or_else(|e| panic!("{}: the artifacts must build: {e:?}", job.0))
}

fn prove_bytes(job: &Job, built: &WhirLfmBuild, opts: &crate::ProofOptions) -> Vec<u8> {
    let proved = lfm_prove_whir(&job.1, built, &job.2, opts)
        .unwrap_or_else(|e| panic!("{}: the program must prove: {e:?}", job.0));
    rkyv::to_bytes::<rkyv::rancor::Error>(&proved.proof)
        .expect("a W-LFM proof serializes")
        .to_vec()
}

/// Every job, `workers` at a time with the permit armed, each twice: a worker
/// takes the next job, builds it and proves it. Staggered starts, so one
/// proof's prep runs while another holds the card.
fn prove_pool(jobs: &[Job], builds: &[WhirLfmBuild], opts: &crate::ProofOptions) -> Vec<Vec<u8>> {
    const WORKERS: usize = 3;
    super::device_permit::arm(WORKERS);
    let next = AtomicUsize::new(0);
    let out: Vec<Mutex<Option<Vec<u8>>>> = (0..2 * jobs.len()).map(|_| Mutex::new(None)).collect();
    std::thread::scope(|s| {
        for w in 0..WORKERS {
            let (next, out) = (&next, &out);
            s.spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(40 * w as u64));
                loop {
                    let i = next.fetch_add(1, Ordering::SeqCst);
                    if i >= out.len() {
                        break;
                    }
                    let k = i % jobs.len();
                    *out[i].lock().expect("the slot") =
                        Some(prove_bytes(&jobs[k], &builds[k], opts));
                }
            });
        }
    });
    super::device_permit::arm(1);
    out.into_iter()
        .map(|slot| {
            slot.into_inner()
                .expect("the slot")
                .expect("every job proved")
        })
        .collect()
}

/// ★ ONLY THE LOCK MOVES: the same programs, proved with the setting off and on,
/// serially and three at a time with the permit armed, give the same bytes.
#[test]
#[ignore = "fixture traces and the card, under LAMBDA_VM_DETERMINISTIC_GRIND=1: box only"]
fn the_card_after_prep_moves_no_proof_byte() {
    let _g = OVERRIDE.lock().unwrap_or_else(|e| e.into_inner());
    assert!(
        crypto::grinding::deterministic(),
        "run with LAMBDA_VM_DETERMINISTIC_GRIND=1: without it two proves of one program may \
         take different grinding nonces, and every byte after the first grind differs"
    );
    let jobs = fixture_jobs();
    let opts = super::proof::aggregation_wrap_options();
    let builds: Vec<WhirLfmBuild> = jobs.iter().map(|j| build(j, &opts)).collect();

    // The control: the default setting, serially, twice. Two proves of one
    // program must agree, and distinct programs must differ.
    let off = Setting::new(false);
    let serial: Vec<Vec<u8>> = jobs
        .iter()
        .zip(&builds)
        .map(|(j, b)| prove_bytes(j, b, &opts))
        .collect();
    let again: Vec<Vec<u8>> = jobs
        .iter()
        .zip(&builds)
        .map(|(j, b)| prove_bytes(j, b, &opts))
        .collect();
    assert_eq!(serial, again, "two serial proves of one program must agree");
    assert_ne!(
        serial[0],
        serial[jobs.len() - 1],
        "the comparison is not vacuous"
    );
    drop(off);

    let _on = Setting::new(true);
    for (k, (j, b)) in jobs.iter().zip(&builds).enumerate() {
        assert!(
            prove_bytes(j, b, &opts) == serial[k],
            "{}: the permit after the prep moved a byte (serial)",
            j.0
        );
    }
    let pooled = prove_pool(&jobs, &builds, &opts);
    for (i, proof) in pooled.iter().enumerate() {
        let k = i % jobs.len();
        assert!(
            proof == &serial[k],
            "{}: the permit after the prep moved a byte (three at a time)",
            jobs[k].0
        );
    }
    println!(
        "CARD AFTER PREP: {} jobs, serial and pooled ({} proofs), equal to the default's bytes",
        jobs.len(),
        pooled.len()
    );
}

/// ★ THE PREP NEVER REACHES THE DEVICE: no entry into the device layer across
/// `prep_whir_tables`, for every fixture job — and, the control, some across the
/// prove that follows it, so the count is one that sees this prove's device
/// work.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "fixture traces and the card: box only, run alone (the count is process-wide)"]
fn the_w_lfm_prep_never_reaches_the_card() {
    use super::executor::execute;
    use super::trace::build_traces_with_hasher;
    use super::whir_proof::{airs_for, prep_whir_tables, prove_traces_whir};
    use math_cuda::device::backend_entries;

    let _g = OVERRIDE.lock().unwrap_or_else(|e| e.into_inner());
    let jobs = fixture_jobs();
    let opts = super::proof::aggregation_wrap_options();
    for job in &jobs {
        let built = build(job, &opts);
        let hasher = built.artifacts.hasher;
        let exec = execute(&job.1, &job.2, &hasher).expect("the program executes");

        let mut traces = build_traces_with_hasher(&job.1, &exec.records, hasher);
        let before = backend_entries();
        let airs = airs_for(&built.artifacts, &opts);
        let prepped = prep_whir_tables(&built, &airs, &mut traces, true)
            .unwrap_or_else(|e| panic!("{}: the prep must succeed: {e:?}", job.0));
        let during = backend_entries() - before;
        assert_eq!(
            during, 0,
            "{}: the host prep entered the device layer {during} time(s); that step needs the \
             card permit, so it belongs after it",
            job.0
        );
        assert!(
            !prepped.tables.is_empty(),
            "{}: the prep built no table",
            job.0
        );
        drop(prepped);

        // The control: the full prove, on a fresh trace, does reach the card.
        let mut fresh = build_traces_with_hasher(&job.1, &exec.records, hasher);
        let before = backend_entries();
        prove_traces_whir(&built, &mut fresh, &exec.public_words, &opts, true)
            .unwrap_or_else(|e| panic!("{}: the program must prove: {e:?}", job.0));
        let proving = backend_entries() - before;
        assert!(
            proving > 0,
            "{}: the prove entered the device layer zero times — the counter cannot see this \
             prove's device work, so the prep's zero means nothing",
            job.0
        );
        println!(
            "CARD AFTER PREP: {}: 0 device entries in the prep, {proving} in the prove",
            job.0
        );
    }
}
