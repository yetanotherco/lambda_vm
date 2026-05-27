//! Tests for statement absorption into the Fiat-Shamir transcript.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::fiat_shamir::is_transcript::IsTranscript;

use crate::statement::absorb_statement;
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
) -> [u8; 32] {
    let mut t = DefaultTranscript::<E>::new(&[]);
    absorb_statement(&mut t, elf, out, counts, priv_pages, ranges);
    t.state()
}

#[test]
fn state_is_deterministic() {
    let a = state_after_absorb(b"elf", b"out", &sample_counts(), 3, &sample_ranges());
    let b = state_after_absorb(b"elf", b"out", &sample_counts(), 3, &sample_ranges());
    assert_eq!(a, b);
}

#[test]
fn state_depends_on_every_field() {
    let baseline = state_after_absorb(b"elf", b"out", &sample_counts(), 1, &sample_ranges());

    assert_ne!(
        baseline,
        state_after_absorb(
            b"different-elf",
            b"out",
            &sample_counts(),
            1,
            &sample_ranges()
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
            &sample_ranges()
        ),
        "state must depend on public_output",
    );

    let mut counts2 = sample_counts();
    counts2.branch += 1;
    assert_ne!(
        baseline,
        state_after_absorb(b"elf", b"out", &counts2, 1, &sample_ranges()),
        "state must depend on table_counts",
    );

    assert_ne!(
        baseline,
        state_after_absorb(b"elf", b"out", &sample_counts(), 2, &sample_ranges()),
        "state must depend on num_private_input_pages",
    );

    assert_ne!(
        baseline,
        state_after_absorb(b"elf", b"out", &sample_counts(), 1, &[]),
        "state must depend on runtime_page_ranges",
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
        state_after_absorb(b"elf", b"", &counts_a, 0, &[]),
        state_after_absorb(b"elf", b"\x41", &counts_b, 0, &[]),
    );
}
