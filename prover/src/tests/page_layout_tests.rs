//! Regression tests for the verifier's PAGE-layout reconstruction.
//!
//! `runtime_page_ranges` is a prover-chosen field of `VmProof` carrying a free
//! `u64` base and a free `u64` count, and the verifier turns it into PAGE tables
//! with `Traces::page_configs_from_elf_and_runtime`. These tests pin the two
//! properties that reconstruction must enforce on untrusted input.
//!
//! **One page per address.** Two PAGE tables covering the same base each provide
//! a genesis token for every address in that page. The memory argument's
//! soundness rests on the init set holding exactly one entry per address: with
//! two, a witness can have the real page's row consume the duplicate's token and
//! the duplicate's row consume the real one, injecting a value the program never
//! wrote while the bus still balances. A prover reaches this with no private
//! input at all, by declaring a runtime range aliasing a real ELF data page —
//! and *both* pages then carry correct, verifier-recomputed preprocessed
//! commitments, so nothing is forged at the commitment layer. This is the
//! companion to `page_offset_forgery_poc`: pinning `OFFSET` restores one row per
//! address *within* a page, and this restores one page per address.
//!
//! **Bounded before allocation.** The `expected_proof_count` cross-check would
//! reject a wrong page count, but it runs after the configs are materialised, so
//! a `count: u64::MAX` range exhausts memory first — a verifier DoS on untrusted
//! input.
//!
//! These exercise the verifier's own reconstruction path (the same function
//! `verify_proof_parts` calls). They do not build a forged proof end to end; the
//! full attack demonstration for the duplication route lives with the PoC work.

use crate::tables::page::DEFAULT_PAGE_SIZE;
use crate::tables::trace_builder::Traces;
use crate::test_utils::asm_elf_bytes;
use crate::{Error, RuntimePageRange};

use executor::elf::Elf;

fn test_elf() -> Elf {
    Elf::load(&asm_elf_bytes("poc_rodata_commit")).expect("ELF load")
}

/// Base of some page the ELF itself already defines — the address a duplicate
/// range would alias.
fn an_elf_page_base(elf: &Elf) -> u64 {
    Traces::page_configs_from_elf(elf)
        .first()
        .expect("the ELF must define at least one page")
        .page_base
}

fn layout(
    elf: &Elf,
    ranges: &[RuntimePageRange],
    max_pages: usize,
) -> Result<Vec<crate::tables::page::PageConfig>, Error> {
    Traces::page_configs_from_elf_and_runtime(elf, ranges, 0, max_pages)
}

/// Non-vacuity: the honest shape this all has to keep accepting.
#[test]
fn honest_page_layout_is_accepted() {
    let elf = test_elf();
    let elf_pages = Traces::page_configs_from_elf(&elf).len();

    // A runtime range that does not alias any ELF page: well past the ELF image.
    let base = 0x8000_0000u64;
    let configs = layout(&elf, &[RuntimePageRange { base, count: 3 }], usize::MAX)
        .expect("an honest, non-overlapping layout must be accepted");
    assert_eq!(configs.len(), elf_pages + 3);

    // And the result stays sorted with no repeats — what the checks below defend.
    assert!(configs.windows(2).all(|w| w[0].page_base < w[1].page_base));
}

/// A runtime range aliasing a real ELF page must be rejected: that is the exact
/// shape of the duplication attack, and the one a prover can mount with no
/// private input.
#[test]
fn runtime_range_aliasing_an_elf_page_is_rejected() {
    let elf = test_elf();
    let base = an_elf_page_base(&elf);

    let err = layout(&elf, &[RuntimePageRange { base, count: 1 }], usize::MAX)
        .expect_err("a runtime page aliasing an ELF page must be rejected");
    assert!(
        matches!(&err, Error::MalformedPageLayout(m) if m.contains("exactly")),
        "expected a duplicate-page rejection, got: {err}"
    );
}

/// Two identical runtime ranges are the same violation without involving the ELF.
#[test]
fn duplicate_runtime_ranges_are_rejected() {
    let elf = test_elf();
    let base = 0x8000_0000u64;

    let err = layout(
        &elf,
        &[
            RuntimePageRange { base, count: 1 },
            RuntimePageRange { base, count: 1 },
        ],
        usize::MAX,
    )
    .expect_err("two runtime ranges covering the same base must be rejected");
    assert!(
        matches!(&err, Error::MalformedPageLayout(m) if m.contains("exactly")),
        "expected a duplicate-page rejection, got: {err}"
    );
}

/// Overlapping (not merely identical) ranges are caught by the same check,
/// because alignment makes same-size pages either equal or disjoint.
#[test]
fn overlapping_runtime_ranges_are_rejected() {
    let elf = test_elf();
    let base = 0x8000_0000u64;
    let page = DEFAULT_PAGE_SIZE as u64;

    let err = layout(
        &elf,
        &[
            RuntimePageRange { base, count: 4 },
            RuntimePageRange {
                base: base + 2 * page,
                count: 4,
            },
        ],
        usize::MAX,
    )
    .expect_err("overlapping runtime ranges must be rejected");
    assert!(
        matches!(&err, Error::MalformedPageLayout(m) if m.contains("exactly")),
        "expected a duplicate-page rejection, got: {err}"
    );
}

/// Unaligned bases are rejected. Beyond being malformed, this is what keeps "same
/// base" equivalent to "overlapping": page-aligned pages of one size either share a
/// base or are disjoint, with no partial-overlap case.
#[test]
fn unaligned_runtime_page_base_is_rejected() {
    let elf = test_elf();

    let err = layout(
        &elf,
        &[RuntimePageRange {
            base: 0x8000_0000 + 1,
            count: 1,
        }],
        usize::MAX,
    )
    .expect_err("an unaligned runtime page base must be rejected");
    assert!(
        matches!(&err, Error::MalformedPageLayout(m) if m.contains("aligned")),
        "expected an alignment rejection, got: {err}"
    );
}

/// DoS: a `u64::MAX` count must be refused up front, not after allocating.
///
/// The assertion that matters is not just the `Err` but that this test *returns*
/// — before the bound, `for i in 0..count` would allocate `PageConfig`s until the
/// process died, so a regression here shows up as the suite being OOM-killed.
#[test]
fn unbounded_runtime_page_count_is_rejected_without_allocating() {
    let elf = test_elf();

    for count in [u64::MAX, u64::MAX / 2, 1 << 40] {
        let err = layout(&elf, &[RuntimePageRange { base: 0, count }], 4096)
            .expect_err("an absurd page count must be rejected");
        assert!(
            matches!(&err, Error::MalformedPageLayout(m) if m.contains("more than")),
            "expected a page-count rejection for count={count}, got: {err}"
        );
    }
}

/// The cap is the sub-proof count, so a layout one page over it is refused.
/// Nothing an honest prover produces can trip this: every page needs a sub-proof.
#[test]
fn page_count_above_the_cap_is_rejected() {
    let elf = test_elf();
    let elf_pages = Traces::page_configs_from_elf(&elf).len();

    let ranges = [RuntimePageRange {
        base: 0x8000_0000,
        count: 2,
    }];
    // Exactly enough room: accepted.
    layout(&elf, &ranges, elf_pages + 2).expect("a layout that fits the cap is fine");
    // One short: refused.
    let err = layout(&elf, &ranges, elf_pages + 1)
        .expect_err("a layout needing more pages than the proof has sub-proofs must be rejected");
    assert!(
        matches!(&err, Error::MalformedPageLayout(m) if m.contains("more than")),
        "expected a page-count rejection, got: {err}"
    );
}

/// A zero-count range is meaningless — the honest run-length encoding never emits
/// one — so it is refused rather than silently skipped.
#[test]
fn zero_count_runtime_range_is_rejected() {
    let elf = test_elf();

    let err = layout(
        &elf,
        &[RuntimePageRange {
            base: 0x8000_0000,
            count: 0,
        }],
        usize::MAX,
    )
    .expect_err("a zero-count runtime range must be rejected");
    assert!(
        matches!(&err, Error::MalformedPageLayout(m) if m.contains("count 0")),
        "expected a zero-count rejection, got: {err}"
    );
}

/// The stack's top page must stay accepted.
///
/// It sits at the very top of the address space, so its *exclusive* end is exactly
/// `2^64` and only its last byte is representable. An overflow guard written
/// against the exclusive end rejects it — and therefore rejects every honest proof,
/// since every program has a stack. This is a real regression that shipped in a
/// draft of the guard above and was caught by the PoC harness's honest control.
#[test]
fn the_top_page_of_the_address_space_is_accepted() {
    let elf = test_elf();
    let page = DEFAULT_PAGE_SIZE as u64;
    let top_page_base = u64::MAX - page + 1;
    assert_eq!(top_page_base % page, 0, "the top page must be aligned");

    layout(
        &elf,
        &[RuntimePageRange {
            base: top_page_base,
            count: 1,
        }],
        usize::MAX,
    )
    .expect("the top page of the address space is where the stack lives");
}

/// A range whose span wraps the address space is refused before the arithmetic
/// that would wrap. Uses the highest page-aligned base so the alignment check
/// (which runs first) passes and the overflow guard is the one under test.
#[test]
fn overflowing_runtime_range_is_rejected() {
    let elf = test_elf();
    let page = DEFAULT_PAGE_SIZE as u64;
    let top_aligned_base = (u64::MAX / page) * page;
    assert_eq!(top_aligned_base % page, 0, "the test base must be aligned");

    // count * page_size overflows u64 outright, so the guard fires on the
    // multiply rather than on the base + span add.
    let err = layout(
        &elf,
        &[RuntimePageRange {
            base: top_aligned_base,
            count: 1 << 60,
        }],
        usize::MAX,
    )
    .expect_err("an overflowing runtime range must be rejected");
    assert!(
        matches!(&err, Error::MalformedPageLayout(m) if m.contains("overflows")),
        "expected an overflow rejection, got: {err}"
    );

    // And the base + span add: a count that fits in u64 on its own but pushes
    // the range past the top of the address space.
    let err = layout(
        &elf,
        &[RuntimePageRange {
            base: top_aligned_base,
            count: 2,
        }],
        usize::MAX,
    )
    .expect_err("a range running off the end of the address space must be rejected");
    assert!(
        matches!(&err, Error::MalformedPageLayout(m) if m.contains("overflows")),
        "expected an overflow rejection, got: {err}"
    );
}
