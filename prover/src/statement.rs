//! Canonical encoding of the statement a VM proof attests to.
//!
//! The 32-byte digest produced by [`statement_seed`] seeds the Fiat-Shamir
//! transcript on both the prove and verify paths, so every challenge is bound
//! to the program, its public output, and the table layout. Prover and
//! verifier must compute it from identical inputs — any divergence makes every
//! derived challenge differ and verification reject.

use tiny_keccak::{Hasher, Keccak};

use crate::{RuntimePageRange, TableCounts};

/// Bumped whenever the statement encoding changes, so a re-encoding under a new
/// layout cannot collide with one produced by an older layout.
const FORMAT_VERSION: u32 = 1;

/// Fixed domain-separation tag prefixing every statement encoding.
const DOMAIN_TAG: &[u8] = b"LAMBDAVM_STARK_STATEMENT_V1";

fn keccak256(data: &[u8]) -> [u8; 32] {
    let mut hasher = Keccak::v256();
    hasher.update(data);
    let mut out = [0u8; 32];
    hasher.finalize(&mut out);
    out
}

/// Canonical, length-prefixed encoding of the statement.
///
/// Every variable-length field is length-prefixed, so two distinct statements
/// can never produce the same byte string by shifting content across a field
/// boundary.
pub(crate) fn encode_statement(
    elf_bytes: &[u8],
    public_output: &[u8],
    table_counts: &TableCounts,
    num_private_input_pages: usize,
    runtime_page_ranges: &[RuntimePageRange],
) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(DOMAIN_TAG);
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());

    // Program identity: a length-prefixed digest of the ELF (hashed, not
    // inlined, to keep the encoding small).
    out.extend_from_slice(&(elf_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&keccak256(elf_bytes));

    // Public output.
    out.extend_from_slice(&(public_output.len() as u64).to_le_bytes());
    out.extend_from_slice(public_output);

    // Table layout: every field, declared order, fixed width.
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
        out.extend_from_slice(&(count as u64).to_le_bytes());
    }

    out.extend_from_slice(&(num_private_input_pages as u64).to_le_bytes());

    // Runtime page ranges (count-prefixed; each entry fixed width).
    out.extend_from_slice(&(runtime_page_ranges.len() as u64).to_le_bytes());
    for range in runtime_page_ranges {
        out.extend_from_slice(&range.base.to_le_bytes());
        out.extend_from_slice(&range.count.to_le_bytes());
    }

    out
}

/// The 32-byte Fiat-Shamir transcript seed binding the full statement.
pub(crate) fn statement_seed(
    elf_bytes: &[u8],
    public_output: &[u8],
    table_counts: &TableCounts,
    num_private_input_pages: usize,
    runtime_page_ranges: &[RuntimePageRange],
) -> [u8; 32] {
    keccak256(&encode_statement(
        elf_bytes,
        public_output,
        table_counts,
        num_private_input_pages,
        runtime_page_ranges,
    ))
}
