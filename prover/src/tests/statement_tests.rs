//! Tests for statement absorption into the Fiat-Shamir transcript.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::fiat_shamir::is_transcript::IsTranscript;

use crate::statement::{StatementKind, absorb_continuation_global_statement, absorb_statement};
use crate::test_utils::E;
use crate::{RuntimePageRange, TableCounts};

fn sample_counts() -> TableCounts {
    TableCounts {
        cpu: 3,
        lt: 1,
        memw: 2,
        memw_aligned: 1,
        load: 1,
        mul: 1,
        dvrm: 1,
        shift: 1,
        branch: 2,
        memw_register: 1,
        eq: 1,
        bytewise: 1,
        store: 1,
        cpu32: 1,
        keccak_rnd: 2,
        blake3: 1,
    }
}

fn sample_ranges() -> Vec<RuntimePageRange> {
    vec![
        RuntimePageRange {
            base: 0x1000,
            count: 4,
        },
        RuntimePageRange {
            base: 0x8000,
            count: 2,
        },
    ]
}

fn state_after_absorb(
    elf: &[u8],
    out: &[u8],
    counts: &TableCounts,
    priv_pages: usize,
    ranges: &[RuntimePageRange],
    fri_final_poly_log_degree: u8,
) -> [u8; 32] {
    let mut t = DefaultTranscript::<E>::new(&[]);
    absorb_statement(
        &mut t,
        StatementKind::Monolithic,
        elf,
        out,
        counts,
        priv_pages,
        ranges,
        fri_final_poly_log_degree,
    );
    t.state()
}

#[test]
fn state_is_deterministic() {
    let a = state_after_absorb(b"elf", b"out", &sample_counts(), 3, &sample_ranges(), 7);
    let b = state_after_absorb(b"elf", b"out", &sample_counts(), 3, &sample_ranges(), 7);
    assert_eq!(a, b);
}

#[test]
fn state_depends_on_every_field() {
    let baseline = state_after_absorb(b"elf", b"out", &sample_counts(), 1, &sample_ranges(), 7);

    assert_ne!(
        baseline,
        state_after_absorb(
            b"different-elf",
            b"out",
            &sample_counts(),
            1,
            &sample_ranges(),
            7,
        ),
        "state must depend on elf",
    );
    assert_ne!(
        baseline,
        state_after_absorb(
            b"elf",
            b"different-output",
            &sample_counts(),
            1,
            &sample_ranges(),
            7,
        ),
        "state must depend on public_output",
    );

    let mut counts2 = sample_counts();
    counts2.branch += 1;
    assert_ne!(
        baseline,
        state_after_absorb(b"elf", b"out", &counts2, 1, &sample_ranges(), 7),
        "state must depend on table_counts",
    );

    assert_ne!(
        baseline,
        state_after_absorb(b"elf", b"out", &sample_counts(), 2, &sample_ranges(), 7),
        "state must depend on num_private_input_pages",
    );

    assert_ne!(
        baseline,
        state_after_absorb(b"elf", b"out", &sample_counts(), 1, &[], 7),
        "state must depend on runtime_page_ranges",
    );

    assert_ne!(
        baseline,
        state_after_absorb(b"elf", b"out", &sample_counts(), 1, &sample_ranges(), 8),
        "state must depend on fri_final_poly_log_degree",
    );
}

#[test]
fn public_output_length_prefix_prevents_collision() {
    // Without the length prefix on public_output, "empty output + cpu count
    // 0x41" and "output [0x41] + cpu count 0" would absorb identical bytes.
    // The prefix keeps the two statements distinct.
    let mut counts_a = sample_counts();
    counts_a.cpu = 0x41;
    let mut counts_b = sample_counts();
    counts_b.cpu = 0;
    assert_ne!(
        state_after_absorb(b"elf", b"", &counts_a, 0, &[], 7),
        state_after_absorb(b"elf", b"\x41", &counts_b, 0, &[], 7),
    );
}

fn epoch_state(elf: &[u8], label: u64) -> [u8; 32] {
    let mut t = DefaultTranscript::<E>::new(&[]);
    absorb_statement(
        &mut t,
        StatementKind::ContinuationEpoch { epoch_label: label },
        elf,
        b"out",
        &sample_counts(),
        1,
        &sample_ranges(),
        7,
    );
    t.state()
}

#[test]
fn continuation_epoch_state_binds_label_and_program() {
    let baseline = epoch_state(b"elf", 1);
    // Deterministic.
    assert_eq!(baseline, epoch_state(b"elf", 1));
    // Pinned to the epoch's position: a different label diverges (replay across
    // positions is rejected).
    assert_ne!(baseline, epoch_state(b"elf", 2), "must bind epoch_label");
    // Pinned to the program.
    assert_ne!(baseline, epoch_state(b"other-elf", 1), "must bind the ELF");
}

#[test]
fn continuation_epoch_differs_from_monolithic_statement() {
    // A monolithic proof and a continuation epoch proof must never share a
    // transcript seed, even with the same base statement.
    let monolithic = state_after_absorb(b"elf", b"out", &sample_counts(), 1, &sample_ranges(), 7);
    assert_ne!(monolithic, epoch_state(b"elf", 1));
}

fn global_state(
    elf: &[u8],
    num_epochs: usize,
    num_private_input_pages: usize,
    fri_final_poly_log_degree: u8,
    touched_page_bases: &[u64],
) -> [u8; 32] {
    let mut t = DefaultTranscript::<E>::new(&[]);
    absorb_continuation_global_statement(
        &mut t,
        elf,
        num_epochs,
        num_private_input_pages,
        fri_final_poly_log_degree,
        touched_page_bases,
    );
    t.state()
}

#[test]
fn continuation_global_state_binds_program_epoch_count_pages_and_touched_set() {
    let baseline = global_state(b"elf", 3, 1, 7, &[0x1000, 0x2000]);
    assert_eq!(baseline, global_state(b"elf", 3, 1, 7, &[0x1000, 0x2000])); // deterministic
    assert_ne!(
        baseline,
        global_state(b"elf", 4, 1, 7, &[0x1000, 0x2000]),
        "must bind epoch count"
    );
    assert_ne!(
        baseline,
        global_state(b"other-elf", 3, 1, 7, &[0x1000, 0x2000]),
        "must bind the ELF"
    );
    assert_ne!(
        baseline,
        global_state(b"elf", 3, 2, 7, &[0x1000, 0x2000]),
        "must bind the private-input page count"
    );
    assert_ne!(
        baseline,
        global_state(b"elf", 3, 1, 8, &[0x1000, 0x2000]),
        "must bind fri_final_poly_log_degree"
    );
    assert_ne!(
        baseline,
        global_state(b"elf", 3, 1, 7, &[0x1000, 0x3000]),
        "must bind the touched page-base set"
    );
    assert_ne!(
        baseline,
        global_state(b"elf", 3, 1, 7, &[0x1000]),
        "must bind the touched page-base count"
    );
}

/// ★ `TableCounts::total` must move with `keccak_rnd`, at EVERY chunk count.
///
/// The regression this pins cancelled at exactly one chunk: `total` omitted
/// `keccak_rnd` while `FIXED_TABLE_COUNT` still counted KECCAK_RND as
/// always-one, so `reconstruct_epoch_airs`'s `total() + fixed + 1` was right
/// for a one-chunk epoch and short by `n - 1` for an n-chunk one. Every test
/// and a twenty-one-node tree passed on one-chunk epochs; the honest verifier
/// then rejected the first two-chunk epoch it ever saw as "structurally
/// invalid".
///
/// ⚠ So a single-count test cannot catch it. This one spans 1, 2 and 3 —
/// `total` must rise by exactly one per added chunk, and `absorbed` must carry
/// the same value at the same position.
#[test]
fn total_and_absorbed_track_the_keccak_rnd_chunk_count() {
    let base = sample_counts();
    let baseline = TableCounts {
        keccak_rnd: 1,
        ..base.clone()
    };
    for chunks in [1usize, 2, 3] {
        let counts = TableCounts {
            keccak_rnd: chunks,
            ..base.clone()
        };
        assert_eq!(
            counts.total(),
            baseline.total() + (chunks - 1),
            "total() must rise by one per KECCAK_RND chunk (chunks = {chunks})"
        );
        // `total` is a fold of `absorbed`, so the encoding must carry the same
        // number at the same position — index 14, between cpu32 and blake3.
        let absorbed = counts.absorbed();
        assert_eq!(
            absorbed[14], chunks as u64,
            "absorbed()[14] must be the KECCAK_RND count (chunks = {chunks})"
        );
        assert_eq!(
            absorbed.iter().sum::<u64>() as usize,
            counts.total(),
            "total() must equal the sum of absorbed()"
        );
        assert!(
            counts.validate().is_ok(),
            "a positive chunk count must validate (chunks = {chunks})"
        );
    }
    // And zero is refused like every other chunk count.
    assert!(
        TableCounts {
            keccak_rnd: 0,
            ..base
        }
        .validate()
        .is_err(),
        "a zero KECCAK_RND count must be rejected"
    );
}
