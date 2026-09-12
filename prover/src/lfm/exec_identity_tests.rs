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
//! always had. [`candidate`] is the schedule under test. **Today they are the
//! same call**, so this gate passes by construction and proves only that the
//! comparison itself is total and non-vacuous — which is the point: the
//! comparator, the case list and the coverage guard are what a schedule change
//! has to land against, and they are cheaper to get right before there is a
//! second schedule than after. When the level-parallel executor arrives,
//! [`candidate`] is the one line that moves.
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

use super::executor::{LfmExecution, execute};
use super::instr::Addr;
use super::trace_identity_tests::{Case, cases};

/// The serial reference: the interpreter as it stands.
fn reference(case: &Case) -> LfmExecution {
    execute(&case.program, &case.arenas, &case.hasher)
        .unwrap_or_else(|e| panic!("{}: the reference must execute: {e:?}", case.name))
}

/// The schedule under test. ⓘ Identical to [`reference`] until the
/// level-parallel executor lands; this is the single call site that changes.
fn candidate(case: &Case) -> LfmExecution {
    execute(&case.program, &case.arenas, &case.hasher)
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
