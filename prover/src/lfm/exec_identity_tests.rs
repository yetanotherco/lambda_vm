//! The gate on `execute`'s schedule: the witness is a pure function of the
//! program and its arenas, so **which order** the executor evaluates the program
//! in may not change one word of it.
//!
//! This is the sibling of [`super::trace_identity_tests`] one stage earlier in
//! the pipeline. That file gates the row walk that turns records into traces;
//! this one gates the interpreter that produces the records in the first place.
//! Together they pin the whole host side of a proof against a schedule change:
//! program → records (here) → traces (there).
//!
//! ## The two arms
//!
//! [`reference`] is the serial `for instr in &program.instrs` loop the file has
//! always had. [`candidate`] is [`Schedule::LevelParallel`]: each depth level's
//! mutually independent hash instructions on rayon's global pool, everything
//! else serial and in program order.
//!
//! ⚠ The knob that selects the schedule in production cannot be the thing this
//! test flips. The precedent (`LFM_ARTIFACT_PARALLEL`, `commit.rs:35`) caches
//! its environment read in a `OnceLock`, so a process reads it once and a test
//! that set it per-case would gate whichever arm happened to run first. The two
//! arms are therefore reached by an explicit argument, the way
//! `build_traces_walked` takes a [`super::trace::Walk`], and the environment
//! chooses only the production default.
//!
//! ## What is compared
//!
//! Everything an [`LfmExecution`] carries: every memory address through the
//! public accessor (including *unwritten*, which must stay unwritten on both
//! sides — the write-once bit is the only thing separating that from a written
//! zero), the const counter, all ten record vectors element for element, and
//! `public_words`.
//!
//! Elements are compared through their derived `Debug`, which prints a
//! `FieldElement`'s **raw** `u64` rather than its canonical residue. That is
//! deliberate and is strictly stronger than `==` on the field, whose `eq`
//! canonicalises both sides (`goldilocks.rs:157`): two executions of the same
//! instruction stream perform the same arithmetic in the same order, so they owe
//! each other bit equality, not just congruence. It also means no field of any
//! record type is transcribed here — a row that grows a column is compared on
//! its new column the day it grows one, with no edit to this file.

use super::executor::{LfmExecError, LfmExecution, Schedule, execute_scheduled};
use super::instr::Addr;
use super::trace_identity_tests::{Case, cases};

/// The cut-off the candidate arm runs at, and it is not the production one.
///
/// ⛔ The laptop cases are 15 wide at their widest level (printed by the gate
/// below). Anything at or above 16 — which `PRODUCTION_COALESCE_BELOW` is —
/// would send every level of every case down the serial branch, and this file
/// would then pass whatever the parallel path did, on every run, forever. At 2,
/// every level of width ≥ 2 really forks.
const GATE_COALESCE_BELOW: usize = 2;

/// The serial reference: the interpreter as it stands.
fn reference(case: &Case) -> LfmExecution {
    execute_scheduled(&case.program, &case.arenas, &case.hasher, Schedule::Serial)
        .unwrap_or_else(|e| panic!("{}: the reference must execute: {e:?}", case.name))
}

/// The schedule under test: levels, on rayon's global pool.
fn candidate(case: &Case) -> LfmExecution {
    execute_scheduled(
        &case.program,
        &case.arenas,
        &case.hasher,
        Schedule::LevelParallel {
            coalesce_below: GATE_COALESCE_BELOW,
        },
    )
    .unwrap_or_else(|e| panic!("{}: the candidate must execute: {e:?}", case.name))
}

/// Two record vectors, element for element, with the first divergence named.
///
/// Generic over the row type and asking only for `Debug`, so it covers a chip
/// whose row struct this file has never heard of.
fn assert_rows_identical<T: std::fmt::Debug>(case: &str, chip: &str, a: &[T], b: &[T]) {
    assert_eq!(
        a.len(),
        b.len(),
        "{case}/{chip}: the two schedules produced different row counts"
    );
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        let (dx, dy) = (format!("{x:?}"), format!("{y:?}"));
        assert_eq!(
            dx,
            dy,
            "{case}/{chip}: the two schedules disagree at record {i} of {}",
            a.len()
        );
    }
}

/// Asserts two executions are the same object, and says where they first differ
/// if they are not.
fn assert_executions_identical(case: &str, num_addrs: u64, a: &LfmExecution, b: &LfmExecution) {
    // Memory first: it is the object every record is derived from, so a
    // divergence here localises the failing instruction by its destination
    // address, which is its position in emission order.
    for i in 0..num_addrs {
        let (x, y) = (a.memory.get(Addr(i)), b.memory.get(Addr(i)));
        let (dx, dy) = (format!("{x:?}"), format!("{y:?}"));
        assert_eq!(
            dx, dy,
            "{case}: the two schedules left different values at address {i}"
        );
    }

    let (ra, rb) = (&a.records, &b.records);
    assert_eq!(
        ra.num_consts, rb.num_consts,
        "{case}: the const count moved with the schedule"
    );
    assert_rows_identical(case, "LFM_BALU", &ra.balu, &rb.balu);
    assert_rows_identical(case, "LFM_XALU", &ra.xalu, &rb.xalu);
    assert_rows_identical(case, "LFM_SELECT", &ra.select, &rb.select);
    assert_rows_identical(case, "LFM_BITDEC", &ra.bitdec, &rb.bitdec);
    assert_rows_identical(case, "LFM_HASH", &ra.hash, &rb.hash);
    assert_rows_identical(case, "LFM_KECCAK", &ra.keccak, &rb.keccak);
    assert_rows_identical(case, "LFM_BLAKE3", &ra.blake3, &rb.blake3);
    assert_rows_identical(case, "LFM_LANES", &ra.lanes, &rb.lanes);
    assert_rows_identical(case, "LFM_HINT", &ra.hint, &rb.hint);
    assert_rows_identical(case, "LFM_PUBLIC", &ra.public, &rb.public);
    assert_rows_identical(case, "public_words", &a.public_words, &b.public_words);
}

/// ★ The gate. Both schedules, every case, word for word.
#[test]
fn the_parallel_executor_is_byte_identical_to_the_serial_reference() {
    // Records a schedule change could reorder. A case list that gave some chip
    // no rows would satisfy the comparison above on that chip vacuously.
    let mut covered: Vec<&str> = Vec::new();
    let (mut widest, mut deepest) = (0u32, 0usize);

    for case in cases() {
        let a = reference(&case);
        let b = candidate(&case);
        assert_executions_identical(case.name, case.program.num_addrs, &a, &b);

        // ★ The candidate must have USED the path under test. Without this the
        // whole file is satisfied by a candidate that coalesced every level onto
        // the calling thread and ran the serial code twice.
        assert_eq!(
            a.split.parallel_levels, 0,
            "{}: the reference arm must not schedule levels at all",
            case.name
        );
        if b.split.levels > 1 {
            assert!(
                b.split.parallel_levels > 0 && b.split.parallel_hashes > 0,
                "{}: the candidate coalesced all {} levels onto the calling \
                 thread, so nothing this test compares went through rayon",
                case.name,
                b.split.levels
            );
        }

        let r = &a.records;
        for (chip, rows) in [
            ("LFM_BALU", r.balu.len()),
            ("LFM_XALU", r.xalu.len()),
            ("LFM_SELECT", r.select.len()),
            ("LFM_BITDEC", r.bitdec.len()),
            ("LFM_HASH", r.hash.len()),
            ("LFM_KECCAK", r.keccak.len()),
            ("LFM_BLAKE3", r.blake3.len()),
            ("LFM_LANES", r.lanes.len()),
            ("LFM_HINT", r.hint.len()),
            ("LFM_PUBLIC", r.public.len()),
        ] {
            if rows > 0 && !covered.contains(&chip) {
                covered.push(chip);
            }
        }

        // ★ The ONE thing that decides whether this gate says anything about a
        // level-parallel schedule: the shape of the DAG it runs over. A case
        // whose hash levels are all one wide takes the serial path in both arms
        // however the candidate is built, and would pass a broken parallel
        // executor without ever forking.
        let p = super::reach_profile::profile(&case.program);
        widest = widest.max(p.widest_level());
        deepest = deepest.max(p.hash_depth);
        println!(
            "{:<18} addrs {:>7}  instrs {:>6}  hash rows {:>6}  \
             records {:>6}  public {:>4}  hash depth {:>4}  widest level {:>4}",
            case.name,
            case.program.num_addrs,
            case.program.instrs.len(),
            r.hash.len(),
            r.balu.len() + r.xalu.len() + r.select.len() + r.bitdec.len() + r.lanes.len(),
            a.public_words.len(),
            p.hash_depth,
            p.widest_level(),
        );
    }

    for chip in RECORDED_CHIPS {
        assert!(
            covered.contains(chip),
            "no case gave {chip} a record, so the two schedules were never \
             compared on it and the gate is vacuous there. Add a case that \
             exercises it rather than dropping it from this list."
        );
    }

    // ⚠ The SHAPE guard, and it is the one that decides what this test is worth.
    // A schedule can only be wrong where there is a choice to get wrong: over a
    // program of depth 1, or of width 1 at every level, every schedule is the
    // same schedule. These bounds are what the case list happens to provide, so
    // a case list that shrank below them would be caught here rather than by a
    // green run that gated nothing.
    assert!(
        widest >= MIN_WIDEST_LEVEL,
        "the widest hash level across every case is {widest}, under {MIN_WIDEST_LEVEL}: \
         there is not enough independent work here for two schedules to differ, \
         so a green run would mean nothing"
    );
    assert!(
        deepest >= MIN_HASH_DEPTH,
        "the deepest hash chain across every case is {deepest}, under {MIN_HASH_DEPTH}: \
         a program this shallow has almost no level boundaries to get wrong"
    );
    println!(
        "shape: widest hash level {widest} · deepest hash chain {deepest} \
         (floors {MIN_WIDEST_LEVEL} / {MIN_HASH_DEPTH})"
    );
}

/// ⛔ The laptop's case list is a TOY next to a wrap (17,627 wide and 2,237 deep
/// on the 2026-09-12 reading; these cases measure **15 wide and 14 deep**, and
/// only the two `FriToyV0` cases have any `Instr::Hash` at all — the two sponge
/// programs drive the keccak and BLAKE3 chips, which are not hash rows). The
/// floors below are what the list does provide, held against shrinkage; they are
/// not a claim that this gate covers production shapes. The gate that does is
/// the tree-scale `IDENTITY:` diff.
///
/// ⚠ **A consequence for the schedule, and it is load-bearing.** A level-parallel
/// executor wants to run narrow levels serially — a width-1 level on rayon is a
/// fork/join for one permutation. Any such cut-off at or above 15 would send
/// **every** level of every case here down the serial path, and this test would
/// then pass whatever the parallel path did. The candidate arm must therefore be
/// reached with the cut-off lowered to 2, so each of these levels really forks;
/// the production default is a separate number.
const MIN_WIDEST_LEVEL: u32 = 8;
const MIN_HASH_DEPTH: usize = 8;

/// The chips the executor pushes a record for. `LFM_CONST` is absent because it
/// keeps a counter rather than a vector, and it is asserted separately.
const RECORDED_CHIPS: &[&str] = &[
    "LFM_BALU",
    "LFM_XALU",
    "LFM_SELECT",
    "LFM_BITDEC",
    "LFM_HASH",
    "LFM_KECCAK",
    "LFM_BLAKE3",
    "LFM_LANES",
    "LFM_HINT",
    "LFM_PUBLIC",
];

/// The record-slot plan's unit cost, pinned.
///
/// A level-parallel executor pre-sizes `records.hash` and has each hash write
/// its own slot, so the vector's footprint is the plan's memory cost and is
/// quoted in the design note. It is twelve `IN` felts and twelve `OUT` felts of
/// eight bytes with nothing else in the struct; pinning it here is what stops
/// that quote from becoming a stale number if a column is ever added.
#[test]
fn a_hash_record_slot_is_192_bytes() {
    assert_eq!(
        std::mem::size_of::<super::executor::HashRow>(),
        192,
        "a hash record is 24 Goldilocks felts and nothing else"
    );
    assert_eq!(
        std::mem::size_of::<crate::tables::types::FE>(),
        8,
        "a Goldilocks felt is one u64, which is what the 192 above is built on"
    );
}

/// Every record row's width, pinned.
///
/// ⛔ These are the multipliers in `LfmRecords::bytes`, and `bytes` is what the
/// split line reports as the size of what `setup` first-touches — the figure the
/// campaign will size the phase against. A row that silently grew a column would
/// move that figure with nothing to say so, and the arithmetic in the handoff
/// would go on quoting the old one.
#[test]
fn the_record_rows_are_the_widths_the_sizing_assumes() {
    use super::blake3_chip::Blake3Values;
    use super::executor::{BaluRow, BitDecRow, HashRow, KeccakRow, SelectRow, XaluRow};
    use super::word::LfmWord;
    use std::mem::size_of;

    for (name, got, want) in [
        ("BaluRow", size_of::<BaluRow>(), 32),
        ("XaluRow", size_of::<XaluRow>(), 96),
        ("SelectRow", size_of::<SelectRow>(), 136),
        ("BitDecRow", size_of::<BitDecRow>(), 528),
        ("HashRow", size_of::<HashRow>(), 192),
        ("KeccakRow", size_of::<KeccakRow>(), 744),
        ("Blake3Values", size_of::<Blake3Values>(), 112),
        ("LfmWord", size_of::<LfmWord>(), 32),
    ] {
        assert_eq!(
            got, want,
            "{name} is {got} bytes, not the {want} the sizing assumes"
        );
    }
}

/// ★★ The property the `unsafe` in `commit_slots` rests on: a walk that fails
/// partway NEVER yields records.
///
/// The level walk writes rows into spare capacity and leaves every vector at
/// length zero until `commit_slots`, which runs at one place after the last
/// instruction. So a program that errors mid-walk returns `Err` and the
/// half-written buffers are freed without a single slot being read — the
/// uninitialised state is not merely unlikely to be observed, it is not
/// reachable through the API at all.
///
/// The merged schedule is a walk that fails partway by construction, which is
/// what makes this checkable rather than assertable: it errors with rows already
/// written, and there is still no value for a caller to inspect.
#[test]
fn a_failed_walk_yields_no_records_at_all() {
    let mut checked = 0;
    for case in cases() {
        if super::reach_profile::profile(&case.program).hash_rows == 0 {
            continue;
        }
        let out = super::executor::execute_with_merged_levels(
            &case.program,
            &case.arenas,
            &case.hasher,
            GATE_COALESCE_BELOW,
            2,
        );
        assert!(
            out.is_err(),
            "{}: this walk must fail partway, or it is not the case this test needs",
            case.name
        );
        checked += 1;
    }
    assert!(checked > 0, "no case exercised a partial walk");
}

/// ★★ THE MUTATION. Merge adjacent depth levels and the executor must FAIL — at
/// the same address, on every run.
///
/// This is what makes the gate above a gate rather than a green light. The level
/// boundary is the only thing separating a hash from the input another hash
/// produces; `merge = 2` removes it, and if the executor still produced a
/// correct witness that would mean the boundary was never doing anything and the
/// identity assertion was passing for some other reason.
///
/// ⓘ It fails deterministically because the workers hold `&WriteOnceMemory` and
/// nothing holds `&mut` during a level: the premature read finds an unwritten
/// cell, which is [`LfmExecError::ReadBeforeWrite`], never a race. Three runs,
/// asserted to name one address, is how that claim is checked rather than
/// asserted.
#[test]
fn merging_adjacent_levels_fails_at_one_address_every_run() {
    let mut checked = 0;
    for case in cases() {
        // A case with no hashes has no level structure to break.
        if super::reach_profile::profile(&case.program).hash_rows == 0 {
            continue;
        }
        let mut seen: Vec<String> = Vec::new();
        for run in 0..3 {
            let err = super::executor::execute_with_merged_levels(
                &case.program,
                &case.arenas,
                &case.hasher,
                GATE_COALESCE_BELOW,
                2,
            )
            .expect_err("a merged schedule launches a hash before its input exists");
            let LfmExecError::ReadBeforeWrite(addr) = err else {
                panic!(
                    "{}: run {run} failed with {err:?}, not ReadBeforeWrite — \
                     the merged schedule has to fail for the schedule's reason",
                    case.name
                );
            };
            // ★ And it has to fail for the RIGHT reason: the cell it read too
            // early must be one a HASH writes. A merge that collapsed the
            // constants into the first hash level would also raise
            // ReadBeforeWrite, and would prove nothing about whether a hash can
            // be launched before the hash it depends on.
            let writer = case
                .program
                .instrs
                .iter()
                .find(|i| i.writes().iter().any(|a| a.0 == addr));
            assert!(
                matches!(writer, Some(super::instr::Instr::Hash { .. })),
                "{}: the premature read was of address {addr}, which is written \
                 by {writer:?} rather than by a hash — the mutation broke some \
                 other ordering",
                case.name
            );
            seen.push(format!("{err:?}"));
        }
        assert!(
            seen.windows(2).all(|w| w[0] == w[1]),
            "{}: three runs of the merged schedule named different addresses \
             ({seen:?}) — the failure is timing-dependent, which is exactly what \
             this design is supposed to make impossible",
            case.name
        );
        println!("{:<18} merged levels -> {}", case.name, seen[0]);
        checked += 1;
    }
    assert!(
        checked > 0,
        "no case had a hash to mis-schedule, so the mutation proved nothing"
    );
}
