//! ★ THE BYTE GATE — the WHIR identity line over a canonically sorted EQ trace.
//!
//! A printing measurement, not an assertion. It exists to be run on two
//! revisions and have its output compared, which is a thing a test harness
//! cannot do for you, so it prints and is `#[ignore]`d.
//!
//! ```text
//! cargo test --release -p lambda-vm-prover --lib \
//!     the_whir_identity_line_over_a_canonically_sorted_eq_trace \
//!     -- --ignored --nocapture
//! ```
//!
//! # (a) The sort is MEASUREMENT-ONLY
//!
//! The rows are canonically ordered HERE, after `generate_eq_trace` returns,
//! and none of the six trace builders is touched. Row order is free to the
//! argument — the bus is a multiset — so a sorted trace is still a valid EQ
//! trace that proves and verifies; it is simply a reproducible one.
//!
//! # (b) Why it exists at all, which is a mistake worth not repeating
//!
//! The obvious version of this measurement — hash the proof of
//! `generate_eq_trace`'s output — **is not reproducible across processes**, and
//! quoting it across two revisions produces a number that looks like evidence
//! and is not. `generate_eq_trace` deduplicates through a
//! `std::collections::HashMap` and lays its rows out in iteration order
//! (`prover/src/tables/eq.rs:128`), which `RandomState` randomises per map;
//! five sibling generators do the same (`bytewise.rs:107`, `branch.rs:166`,
//! `dvrm.rs:298`, `lt.rs:168`, `mul.rs:306`).
//!
//! Four consecutive runs of the unsorted version, on ONE unchanged tree and one
//! unchanged binary, observed 2026-09-15:
//!
//! ```text
//! 9147a1b9ad34e92248608c997506b4b6c06228654fb8717eca04c09f17236bc5
//! 7b8afea2618350600e99bb67200bb4447d962f753b6e858ee0982336436e6dd3
//! f3c9671a29d0f8fe02aaaf9d1ea85e86e13577989489f914c5c89802cb70970e
//! d778a2de322c96e1733757cfa7668ea7180907914af82d79cf6c6a72e81ceca2
//! ```
//!
//! Two of those agreeing by chance across two revisions is roughly a one-in-ten
//! event, and it happened. **Run any instrument twice on one tree before
//! quoting it across two.**
//!
//! # (c) The expected value
//!
//! With the sort, three consecutive runs agree, and the line is identical at
//! PR #988's head and at every commit of this branch:
//!
//! ```text
//! 307d7c00  7b8afea2618350600e99bb67200bb4447d962f753b6e858ee0982336436e6dd3   6880 bytes
//! bcdd3dd2  7b8afea2618350600e99bb67200bb4447d962f753b6e858ee0982336436e6dd3   6880 bytes
//! 29fbb45d  7b8afea2618350600e99bb67200bb4447d962f753b6e858ee0982336436e6dd3   6880 bytes
//! ```
//!
//! ✓ Machine-independent: three runs on a 9950X + RTX 5090 box gave the same
//! line as this laptop.
//!
//! Under `LAMBDA_VM_WHIR_HASH=rpx` the line is
//! `5226e4cfffac7eb2ba629470a0c5ebf879421078b389e3a8065ae63768031adb` — a
//! DIFFERENT digest at the SAME 6880 bytes, which is the whole claim of the
//! seam in one line: the hash moved, the format did not.
//!
//! That is the gate: **the keccak arm's bytes do not move.** A commit that
//! changes this line has changed the proof PR #988 produces, and owes an
//! explanation.
//!
//! # ⛔ AND THE TRAP THIS TEST ITSELF FELL INTO
//!
//! The first version pinned `KeccakWhir` at its prove site, so it never reached
//! a dispatch. Run under `LAMBDA_VM_WHIR_HASH=rpx` on the box it printed **the
//! keccak line and no banner** — a measurement wearing the wrong arm's label,
//! the third instance of that class on this branch and the first inside the
//! instrument meant to catch it. Two lessons, both now enforced below rather
//! than described: a bench that does not print `★ WHIR HASH:` **did not reach a
//! dispatch**, and an arm that cannot produce a different answer is not an arm.
//!
//! # What it does NOT cover
//!
//! Grinding is off, so no nonce reaches the transcript and nothing here
//! exercises the proof-of-work path — see `whir_identity_tests` for why a
//! ground proof cannot be gated this way at all. One table, one group, one
//! stacked polynomial: this is a canary for the seam, not a block-level
//! measurement.

use digest::Digest;
use math::field::element::FieldElement;
use math::field::{
    extensions_goldilocks::Degree3GoldilocksExtensionField as Ext,
    goldilocks::GoldilocksField as Fp,
};

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use multilinear::whir_chain::{ChainConfig, GrindBits};
use stark::multilinear_air::Uniforms;
use stark::multilinear_table::{self, CommittedTable, CommittedTables, TableLayout};
use stark::proof::options::ProofOptions;
use stark::traits::AIR;

use crate::tables::eq::{EqOperation, generate_eq_trace};
use crate::test_utils::{ConcreteVmAir, create_eq_air};

/// The EQ trace's columns with the ROWS sorted into a canonical order.
///
/// Sorted by the whole row read as canonical `u64`s, so the result does not
/// depend on the incoming order — which is the entire point.
fn canonically_sorted_columns() -> Vec<Vec<FieldElement<Fp>>> {
    let ops = vec![
        EqOperation::new(7, 7, false),
        EqOperation::new(7, 9, false),
        EqOperation::new(3, 3, true),
        EqOperation::new(3, 5, true),
    ];
    let columns: Vec<Vec<FieldElement<Fp>>> = generate_eq_trace(&ops).columns_main();
    let rows = columns[0].len();

    let key = |r: usize| -> Vec<u64> { columns.iter().map(|c| *c[r].value()).collect() };
    let mut order: Vec<usize> = (0..rows).collect();
    order.sort_by_key(|r| key(*r));

    columns
        .iter()
        .map(|c| order.iter().map(|r| c[*r]).collect())
        .collect()
}

/// The keccak arm's line.
const KECCAK_LINE: &str = "7b8afea2618350600e99bb67200bb4447d962f753b6e858ee0982336436e6dd3";

/// The RPX arm's line.
///
/// ⚠ Pinned, not merely required to DIFFER from keccak's. `assert_ne!` passes
/// for every wrong answer except one, so it cannot tell "the RPX hash ran" from
/// "something else ran": a third hash, or a transcript whose sampling schedule
/// moved, would both clear it. Measured on the merged branch, twice per arm,
/// and equal at `0cbc9623` — which is what says the merge left the WHIR path's
/// bytes alone.
///
/// ⛔ **WHAT THIS LINE IS A PROOF OF, exactly: RPX TREES UNDER A KECCAK
/// SPONGE.** The RPX arm is a half-flip at this revision, and that is a
/// property of the branch rather than of this test. ✓ VERIFIED: every WHIR
/// prove site builds `DefaultTranscript::<E>::new(..)` — the DEFAULT type
/// parameter, which is `KeccakTranscriptHash` — at `multilinear_prove.rs:226`
/// and `:415` and at four sites in `multilinear_continuation.rs`, and so does
/// the fixture below. `WhirHash::Transcript` is reached only through
/// `GrindingDigest<H>` (`whir_hash.rs:54`) and a compile-time `PhantomData`
/// assertion. So under `LAMBDA_VM_WHIR_HASH=rpx` the Merkle backend, the device
/// kernels and the proof-of-work digest are RPX while Fiat–Shamir is keccak.
///
/// This is the configuration W1's design note called unspellable, and the
/// reason it is spellable anyway is that the transcript was never wired — not
/// that the seam failed. Two consequences worth carrying:
///
/// 1. **This constant WILL move when the wiring lands**, and that move is the
///    expected result, not drift. Re-pin it there; do not reconcile it here.
/// 2. **Any cost attributed to "RPX" on this branch excludes the sponge.** The
///    measured +38.8 s on a 39.7 s keccak prove is trees, kernels and grind
///    only, so it is a LOWER bound on the full swap rather than a measurement
///    of it.
const RPX_LINE: &str = "5226e4cfffac7eb2ba629470a0c5ebf879421078b389e3a8065ae63768031adb";

/// The serialized length, which neither arm may move: 32-byte digests either
/// way and no proof struct gains a field.
const SERIALIZED_LEN: usize = 6880;

/// Prints the identity line and the serialized length, and ASSERTS what each
/// arm owes. See the module header.
#[test]
#[ignore = "a printing measurement: run it on two revisions and compare the output"]
fn the_whir_identity_line_over_a_canonically_sorted_eq_trace() {
    let config = ChainConfig {
        log_blowup: 2,
        log_folding: 2,
        num_queries: 3,
        grind: GrindBits::default(),
    };

    let options = ProofOptions::default_test_options();
    let air: ConcreteVmAir<_> = create_eq_air(&options);
    let columns = canonically_sorted_columns();
    let num_vars = columns[0].len().trailing_zeros() as usize;

    let layout = TableLayout::<Fp, Ext>::new(
        air.constraint_program(),
        air.constraints_meta(),
        air.bus_interactions(),
        columns.len(),
        num_vars,
        Uniforms::default(),
    )
    .expect("layout");
    let table = CommittedTable::from_layout(layout, |col| columns[col as usize].clone())
        .expect("committed table");

    // ★ Through the knob, like every production site. `MultiProof` does not
    // mention the hash in its type — that is the whole point of a 32-byte
    // digest either way — so the proof can leave the dispatch arm and both
    // arms unify. Reaching a dispatch is also what makes the `★ WHIR HASH:`
    // banner print, and its absence from a log is how this test's own trap was
    // found.
    let proof = crate::with_whir_hash!(|H| {
        let committed = CommittedTables::<_, _, H>::commit(vec![table], &config).expect("commit");
        let mut transcript = DefaultTranscript::<Ext>::new(b"whir-identity");
        multilinear_table::multi_prove(&committed, &config, &mut transcript).expect("prove")
    });

    let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&proof).expect("serialize");
    let line: String = crypto::hash::platform_keccak::PlatformKeccak256::digest(bytes.as_ref())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();

    println!("IDENTITY-LINE {line}");
    println!("IDENTITY-LEN  {}", bytes.len());

    // ★★ The assertions that make this an arm rather than a print.
    //
    // The length is the strict one: it may not move under either hash, because
    // a hash swap is not a proof-format change. The line is the opposite — it
    // MUST move, or the rpx label is on a keccak proof, which is exactly what
    // this test printed before it went through a dispatch.
    assert_eq!(
        bytes.len(),
        SERIALIZED_LEN,
        "a hash swap must not change the serialized length"
    );
    match crate::whir_hash_knob::selected() {
        crate::whir_hash_knob::Setting::Keccak => assert_eq!(
            line, KECCAK_LINE,
            "the keccak arm's bytes moved: this commit changed the proof PR #988 produces"
        ),
        crate::whir_hash_knob::Setting::Rpx => {
            assert_ne!(
                line, KECCAK_LINE,
                "the rpx arm produced KECCAK's line — the proof never reached the RPX hash"
            );
            assert_eq!(
                line, RPX_LINE,
                "the rpx arm's bytes moved: this commit changed the proof the RPX seam produces"
            );
        }
    }
}
