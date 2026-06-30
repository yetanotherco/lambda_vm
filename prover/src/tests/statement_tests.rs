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

fn global_state(elf: &[u8], num_epochs: usize) -> [u8; 32] {
    let mut t = DefaultTranscript::<E>::new(&[]);
    absorb_continuation_global_statement(&mut t, elf, num_epochs);
    t.state()
}

#[test]
fn continuation_global_state_binds_program_and_epoch_count() {
    let baseline = global_state(b"elf", 3);
    assert_eq!(baseline, global_state(b"elf", 3)); // deterministic
    assert_ne!(baseline, global_state(b"elf", 4), "must bind epoch count");
    assert_ne!(baseline, global_state(b"other-elf", 3), "must bind the ELF");
}
