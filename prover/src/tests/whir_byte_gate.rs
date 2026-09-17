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
//! `dcc0e8d52a80a6c9ee4ed9911d54b41e7df132d209fbf91e43018b0ac01c4985` — a
//! DIFFERENT digest at the SAME 6880 bytes, which is the whole claim of the
//! seam in one line: the hash moved, the format did not.
//!
//! ⚠ The RPX line was `5226e4cf…031adb` until W1-A2b wired the Fiat-Shamir
//! transcript to the configuration. Until then the RPX arm ran an RPX Merkle
//! backend, an RPX grind and a KECCAK sponge, and this line was the digest of
//! that mixture. Both lines are now PINNED; see [`RPX_LINE`].
//!
//! ⚠ BOTH lines and the length moved again at W1-B step 1, and NOT because of
//! a hash: `MultiProof` gained `preprocessed: Option<StackedProof>` and rkyv
//! writes a discriminant even for `None`. Previous values, for anyone bisecting
//! a byte change to its cause:
//!
//! ```text
//! before step 1   keccak 7b8afea2…6dd3   rpx dcc0e8d5…4985   6880 bytes
//! after  step 1   keccak 0bc7b999…e25e   rpx fcfbf8fe…4682   6904 bytes
//! ```
//!
//! ⛔ This gate does NOT see W1-B's derived-root absorb, and must not be cited
//! as evidence that it happened. The fixture is a single EQ air (`:167`, `:190`)
//! — one table, not DECODE — so it commits no preprocessed group and no derived
//! root. The lines above moved because the STRUCT changed, which is a different
//! fact wearing the same clothes. The absorb's position is pinned by
//! `the_derived_root_is_absorbed_after_the_carried_ones` in
//! `crypto/crypto/tests/transcript_counters.rs`.
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
//!
//! ⛔ **AND NO STATEMENT.** The fixture builds its own transcript from the seed
//! `b"whir-identity"` and calls `multi_prove`, which begins at the roots. So
//! nothing here absorbs an epoch, cross-epoch or monolithic statement, and a
//! change to one of those — the computed statement padding, for instance —
//! CANNOT move these lines. When such a change lands, "the gate is unmoved" is
//! the assertion it owes, not evidence that it did nothing; the instruments that
//! see it are the transcript pair pin's absorb counts and
//! `tests::statement_alignment_tests`. It also means this fixture's own first
//! window (13 seed bytes, then roots) is NOT field element aligned and must
//! never be quoted as a witness that a real proof's is.

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

/// The keccak arm's line. It has never moved and must not.
const KECCAK_LINE: &str = "0bc7b9998634986c89d6913f28361af720102a0329a054c192144d91a60ee25e";

/// ★★ The RPX arm's line, PINNED rather than merely required to differ.
///
/// It was `5226e4cf…031adb` from the seam landing until the transcript was
/// wired to the configuration (W1-A2b). That earlier value is worth keeping in
/// view, because of how the missing wiring was found: W1-A removed the squeeze
/// reversal for RPX, which alters the challenge stream from the first squeeze
/// onward and therefore HAD to move this line — and it did not. The line's
/// refusal to move is what carried the information.
///
/// An `assert_ne!` against [`KECCAK_LINE`], which is what this arm had before,
/// would have passed on that run and said nothing. A pinned constant is the
/// difference between an instrument that can report a surprise and one that can
/// only report a category.
const RPX_LINE: &str = "fcfbf8fe7d2e6b008d41462b478141abfb10dacb2888f2ea668c27d16d044682";

/// The serialized length, which **neither arm may move from the other**.
///
/// ⚠ The invariant this carries is CROSS-ARM equality — "a hash swap is not a
/// proof format change" — not equality with any particular past value. W1-B
/// step 1 moved it 6880 -> 6904 for a reason that is not a hash swap:
/// `MultiProof` gained `preprocessed: Option<StackedProof>`, and rkyv writes a
/// discriminant even when it is `None`.
///
/// So the number was re-pinned ONCE, from a measurement, and BOTH arms were run
/// against the new value — keccak and RPX each printed 6904. Chasing whichever
/// arm happened to be run first would have left the other arm's agreement
/// unasserted and turned this back into a print.
const SERIALIZED_LEN: usize = 6904;

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
        let mut transcript = DefaultTranscript::<
            Ext,
            <H as multilinear::whir_hash::WhirHash>::Transcript,
        >::new(b"whir-identity");
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
            // Both, and in this order: the pin is the real assertion, and the
            // inequality below is what makes a failure legible when the two
            // arms collapse into one.
            assert_ne!(
                line, KECCAK_LINE,
                "the rpx arm produced KECCAK's line — the proof never reached the RPX hash"
            );
            assert_eq!(
                line, RPX_LINE,
                "the rpx arm's bytes moved. If that was intended, say which \
                 change moved them and re-pin; if not, the configuration \
                 reaching the prove is not the one this constant was taken from"
            );
        }
    }
}
