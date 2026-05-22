//! Statement absorbed into the Fiat-Shamir transcript before Phase A.
//!
//! Streams a canonical, domain-separated, length-prefixed encoding directly
//! into the transcript. The transcript is itself a Keccak256 absorber
//! (`DefaultTranscript`), so a single hash suffices — no external digest
//! needed beyond the ELF.
//!
//! All three call sites (prove, verify, bus-balance replay) must absorb
//! identical bytes; any divergence makes every derived challenge differ and
//! verification reject.

use crypto::fiat_shamir::is_transcript::IsTranscript;
use sha3::{Digest, Keccak256};

use crate::test_utils::E;
use crate::{RuntimePageRange, TableCounts};

/// Domain-separation tag. Bump the suffix (`_V2`, ...) on any encoding change.
const DOMAIN_TAG: &[u8] = b"LAMBDAVM_STARK_STATEMENT_V1";

fn elf_digest(elf: &[u8]) -> [u8; 32] {
    let mut h = Keccak256::new();
    h.update(elf);
    h.finalize().into()
}

pub(crate) fn absorb_statement(
    t: &mut impl IsTranscript<E>,
    elf_bytes: &[u8],
    public_output: &[u8],
    table_counts: &TableCounts,
    num_private_input_pages: usize,
    runtime_page_ranges: &[RuntimePageRange],
) {
    t.append_bytes(DOMAIN_TAG);

    // ELF: fixed 32-byte digest — no length prefix needed.
    t.append_bytes(&elf_digest(elf_bytes));

    // public_output: variable length → length-prefix to prevent boundary collisions.
    t.append_bytes(&(public_output.len() as u64).to_le_bytes());
    t.append_bytes(public_output);

    // table_counts: fixed-width u64s in declared order.
    // Reordering or adding a field requires bumping DOMAIN_TAG above.
    for count in [
        table_counts.cpu,
        table_counts.lt,
        table_counts.memw,
        table_counts.memw_aligned,
        table_counts.load,
        table_counts.mul,
        table_counts.dvrm,
        table_counts.shift,
        table_counts.branch,
        table_counts.memw_register,
    ] {
        t.append_bytes(&(count as u64).to_le_bytes());
    }

    t.append_bytes(&(num_private_input_pages as u64).to_le_bytes());

    // runtime_page_ranges: count-prefixed; each entry fixed width.
    t.append_bytes(&(runtime_page_ranges.len() as u64).to_le_bytes());
    for r in runtime_page_ranges {
        t.append_bytes(&r.base.to_le_bytes());
        t.append_bytes(&r.count.to_le_bytes());
    }
}
