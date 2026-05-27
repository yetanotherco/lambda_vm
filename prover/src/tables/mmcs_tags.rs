//! Per-chip [`MatrixTag`] assignments for the unified MMCS over the main
//! trace (PR2 of the streaming-MMCS plan).
//!
//! ## Why this lives here
//!
//! The MMCS leaf-hash binds matrix identity via a per-matrix `MatrixTag`.
//! Prover and verifier MUST derive the same tag for the same chip-chunk;
//! otherwise the Fiat-Shamir transcript diverges and verification fails
//! silently from the user's POV (just an opaque rejection). Centralising
//! the tag derivation in one place — used by both sides — turns "same tag"
//! from a hope into a compile-time guarantee.
//!
//! ## Encoding
//!
//! ```text
//! MatrixTag = [chip_type_id : u32 (le)] [chunk_index : u32 (le)]
//! ```
//!
//! `chip_type_id` values are **stable** — they go on the wire (indirectly,
//! via the Fiat-Shamir transcript) and must never be reassigned. Adding a
//! new chip type appends a new ID; removing one leaves the gap (do not
//! reuse).
//!
//! `chunk_index` is the 0-based index within a single chip type (e.g. CPU
//! chunk 0, CPU chunk 1, ...). For non-split chips (BITWISE, DECODE, ...)
//! it's always 0.

use crypto::merkle_tree::mmcs::MatrixTag;

// =========================================================================
// Chip type IDs — STABLE. Never reassign. Append-only.
// =========================================================================
// Split tables (multiple chunks possible)
pub const CHIP_CPU: u32 = 0;
pub const CHIP_LT: u32 = 1;
pub const CHIP_MEMW: u32 = 2;
pub const CHIP_MEMW_ALIGNED: u32 = 3;
pub const CHIP_LOAD: u32 = 4;
pub const CHIP_MUL: u32 = 5;
pub const CHIP_DVRM: u32 = 6;
pub const CHIP_SHIFT: u32 = 7;
pub const CHIP_BRANCH: u32 = 8;
pub const CHIP_MEMW_REGISTER: u32 = 9;

// Single-instance tables (chunk_index is always 0)
pub const CHIP_BITWISE: u32 = 100;
pub const CHIP_DECODE: u32 = 101;
pub const CHIP_HALT: u32 = 102;
pub const CHIP_COMMIT: u32 = 103;
pub const CHIP_KECCAK: u32 = 104;
pub const CHIP_KECCAK_RC: u32 = 105;
pub const CHIP_KECCAK_RND: u32 = 106;
pub const CHIP_REGISTER: u32 = 107;

// Per-page tables — chunk_index encodes the page index within the page
// configuration the prover and verifier reconstruct from the proof's
// runtime_page_ranges + num_private_input_pages. ELF-segment pages and
// runtime zero-init pages live here; private-input pages also share this
// space because the AIR is the same kind.
pub const CHIP_PAGE: u32 = 200;

/// Build a [`MatrixTag`] from a chip type ID and a chunk index. The
/// encoding is `chip_type_id` (4 bytes LE) followed by `chunk_index`
/// (4 bytes LE) — total 8 bytes.
#[inline]
pub const fn chip_tag(chip_type_id: u32, chunk_index: u32) -> MatrixTag {
    let ct = chip_type_id.to_le_bytes();
    let ci = chunk_index.to_le_bytes();
    MatrixTag::new([ct[0], ct[1], ct[2], ct[3], ci[0], ci[1], ci[2], ci[3]])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// Every (chip_type, chunk) pair we might realistically use must
    /// produce a distinct tag. This catches accidental ID collisions.
    #[test]
    fn tags_are_unique_across_realistic_assignments() {
        let split_chips = [
            CHIP_CPU,
            CHIP_LT,
            CHIP_MEMW,
            CHIP_MEMW_ALIGNED,
            CHIP_LOAD,
            CHIP_MUL,
            CHIP_DVRM,
            CHIP_SHIFT,
            CHIP_BRANCH,
            CHIP_MEMW_REGISTER,
        ];
        let single_chips = [
            CHIP_BITWISE,
            CHIP_DECODE,
            CHIP_HALT,
            CHIP_COMMIT,
            CHIP_KECCAK,
            CHIP_KECCAK_RC,
            CHIP_KECCAK_RND,
            CHIP_REGISTER,
        ];

        let mut seen: HashSet<[u8; 8]> = HashSet::new();
        for chip in split_chips {
            for chunk in 0..64u32 {
                let tag = chip_tag(chip, chunk);
                assert!(
                    seen.insert(tag.0),
                    "duplicate tag for chip {chip:#x} chunk {chunk}"
                );
            }
        }
        for chip in single_chips {
            let tag = chip_tag(chip, 0);
            assert!(seen.insert(tag.0), "duplicate single-chip tag {chip:#x}");
        }
        for page_idx in 0..256u32 {
            let tag = chip_tag(CHIP_PAGE, page_idx);
            assert!(seen.insert(tag.0), "duplicate PAGE tag at index {page_idx}");
        }
    }

    /// Stability test: specific bytes must match a frozen layout so a
    /// future refactor that reshuffles the encoding fails loudly. If you
    /// need to change the encoding, BUMP a new constant family (V2) and
    /// migrate the verifier alongside.
    #[test]
    fn tag_encoding_is_stable() {
        assert_eq!(chip_tag(CHIP_CPU, 0).0, [0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(chip_tag(CHIP_CPU, 1).0, [0, 0, 0, 0, 1, 0, 0, 0]);
        assert_eq!(chip_tag(CHIP_BITWISE, 0).0, [100, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(
            chip_tag(CHIP_PAGE, 0xABCD).0,
            [200, 0, 0, 0, 0xCD, 0xAB, 0, 0]
        );
    }

    /// chip_type and chunk_index encode into independent halves; flipping
    /// either changes the tag.
    #[test]
    fn changing_chip_type_or_chunk_changes_tag() {
        let base = chip_tag(CHIP_CPU, 0);
        assert_ne!(base, chip_tag(CHIP_LT, 0));
        assert_ne!(base, chip_tag(CHIP_CPU, 1));
        assert_ne!(base, chip_tag(CHIP_CPU, u32::MAX));
    }
}
