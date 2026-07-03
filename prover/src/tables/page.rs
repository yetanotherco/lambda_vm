//! PAGE table for memory initialization and finalization.
//!
//! Each PAGE table instance covers one memory page. Multiple PAGE tables
//! are created for all pages used during execution (ELF + stack + heap).
//!
//! ## Token Model (per spec)
//!
//! - **PAGE-C3**: Receives initial token `(address, ts=0, init)` - balances MEMW's send on first access
//! - **PAGE-C4**: Sends final token `(address, timestamp, fini)` - balances MEMW's receive on last access
//!
//! For non-accessed addresses: PAGE-C3 receives and PAGE-C4 sends the same tuple
//! (ts=0, init=fini), which cancel out.
//!
//! ## Columns (per spec)
//!
//! | Column | Type | Description |
//! |--------|------|-------------|
//! | offset | RowIndex | 0, 1, ..., page_size-1 (preprocessed) |
//! | init | Byte | Initial value (from ELF or 0) |
//! | fini | Byte | Final value after execution |
//! | timestamp | DWordWL | Final timestamp (0 if never accessed) |
//!
//! Virtual: `address = page + offset` where `page` is constant per table instance.
//!
//! ## Bus Interactions
//!
//! | Tag | Bus | Signature | Multiplicity |
//! |-----|-----|-----------|--------------|
//! | PAGE-C1+C2 | ARE_BYTES | `[init, fini]` | 1 (sender) |
//! | PAGE-C3    | Memory  | `[0, address, 0, init]` | -1 (receiver) |
//! | PAGE-C4    | Memory  | `[0, address, timestamp, fini]` | 1 (sender) |

use std::collections::HashMap;

use math::polynomial::Polynomial;
use stark::commitment::{ROWS_PER_LEAF, commit_bit_reversed};
use stark::config::Commitment;
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::proof::options::ProofOptions;
use stark::prover::evaluate_polynomial_on_lde_domain;
use stark::trace::TraceTable;

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField, VmTable};

// =========================================================================
// Constants
// =========================================================================

/// Default page size in bytes (256KB).
pub const DEFAULT_PAGE_SIZE: usize = 1 << 18;

/// Stack top address (where SP starts). Re-exported from executor.
pub use executor::vm::registers::STACK_TOP;

// =========================================================================
// Column indices for PAGE table
// =========================================================================

/// Column definitions for the PAGE table.
///
/// Note: `address` is virtual, computed as `page_base + offset` where `page_base`
/// is a constant per table instance. It is NOT stored as a column.
pub mod cols {
    /// offset: Row index (0, 1, ..., page_size-1) - preprocessed
    pub const OFFSET: usize = 0;

    /// init: Initial byte value (from ELF or 0)
    pub const INIT: usize = 1;

    /// fini: Final byte value after execution
    pub const FINI: usize = 2;

    /// timestamp[0]: Final timestamp low word (0 if never accessed)
    pub const TIMESTAMP_LO: usize = 3;

    /// timestamp[1]: Final timestamp high word
    pub const TIMESTAMP_HI: usize = 4;

    /// Total number of columns
    pub const NUM_COLUMNS: usize = 5;
}

/// Number of preprocessed columns (OFFSET, INIT for ELF pages).
/// For zero-init pages, INIT is also preprocessed (constant 0).
pub const NUM_PREPROCESSED_COLS: usize = 2;

// =========================================================================
// Types
// =========================================================================

/// Final state for a single byte address.
#[derive(Debug, Clone, Copy, Default)]
pub struct FinalByteState {
    /// Final timestamp (0 if never accessed)
    pub timestamp: u64,
    /// Final byte value
    pub value: u8,
}

/// Map from byte address to final state.
pub type FinalStateMap = HashMap<u64, FinalByteState>;

/// Configuration for a single PAGE table instance.
#[derive(Debug, Clone)]
pub struct PageConfig {
    /// Base address of this page (must be page-aligned).
    pub page_base: u64,
    /// Initial byte values; `None` means an all-zero page.
    /// `Some(v)` is not padded, so `v.len()` may be smaller than the page
    /// (`DEFAULT_PAGE_SIZE`); any offset at or past `v.len()` is read as zero.
    pub init_values: Option<Vec<u8>>,
    /// Whether this page holds private input data.
    /// Private-input pages are NOT preprocessed — the verifier does not see
    /// the init values. Instead, all columns (including OFFSET and INIT)
    /// are committed as main trace and constrained via the memory bus.
    pub is_private_input: bool,
}

impl PageConfig {
    /// Create a zero-initialized page.
    pub fn zero_init(page_base: u64) -> Self {
        Self {
            page_base,
            init_values: None,
            is_private_input: false,
        }
    }

    /// Create a page with initial values from ELF data. `data` may be shorter
    /// than the page; the trace/commitment math treats trailing bytes as zero.
    pub fn with_data(page_base: u64, data: Vec<u8>) -> Self {
        assert!(data.len() <= DEFAULT_PAGE_SIZE, "Data exceeds page size");
        Self {
            page_base,
            init_values: Some(data),
            is_private_input: false,
        }
    }

    /// Create a page with initial values from private input data.
    ///
    /// These pages are built NON-preprocessed, so INIT is a committed main-trace column
    /// enforced by the GlobalMemory bus rather than recomputed from the ELF. Privacy comes
    /// from that (the raw input is neither bundled nor recomputed by the verifier), NOT from
    /// this constructor: the verifier rebuilds the config from the ELF alone and never consults
    /// the `data` argument for a private page (it passes an empty vec). Not a ZK/hiding claim —
    /// the committed column is still opened at STARK query positions.
    pub fn with_private_input(page_base: u64, data: Vec<u8>) -> Self {
        assert!(data.len() <= DEFAULT_PAGE_SIZE, "Data exceeds page size");
        Self {
            page_base,
            init_values: Some(data),
            is_private_input: true,
        }
    }
}

// =========================================================================
// Private-input page math (shared by the monolithic and continuation paths)
// =========================================================================

/// Number of pages the private input occupies, starting at
/// `PRIVATE_INPUT_START_INDEX`. The wire format is the 4-byte length prefix plus
/// the data ([`Memory::store_private_inputs`]), and `PRIVATE_INPUT_START_INDEX` is
/// page-aligned, so the span is `ceil((prefix + len) / page_size)` consecutive
/// pages (0 when there is no input).
///
/// SINGLE source of truth: the monolithic trace builder, the continuation prover,
/// and both verifiers' classification all derive from this count — a divergence
/// would make one path build a private page preprocessed (ELF-recomputed) while
/// the other commits it, which is a soundness bug, so do not reimplement it.
///
/// [`Memory::store_private_inputs`]: executor::vm::memory::Memory::store_private_inputs
pub fn private_input_page_count(private_inputs: &[u8]) -> usize {
    use executor::vm::memory::PRIVATE_INPUT_LENGTH_PREFIX_BYTES;
    if private_inputs.is_empty() {
        return 0;
    }
    (PRIVATE_INPUT_LENGTH_PREFIX_BYTES + private_inputs.len()).div_ceil(DEFAULT_PAGE_SIZE)
}

/// Whether `page_base` is one of the first `num_private_input_pages` pages starting
/// at `PRIVATE_INPUT_START_INDEX` — the page-aligned span private input actually
/// occupies (see [`private_input_page_count`]). Classifying by the count (not the
/// raw `[START, START+MAX_PRIVATE_INPUT_SIZE)` byte range) keeps prover and
/// verifier in lockstep regardless of whether the region end is page-aligned.
///
/// NOTE: a page classified private is built non-preprocessed, so its genesis is NOT
/// recomputed from the ELF. This is safe because the private-input area is reserved
/// and the reservation is enforced: `Elf::load` rejects any loadable segment
/// reaching at/above `PRIVATE_INPUT_START_INDEX`
/// (`ElfError::SegmentInPrivateInputRegion`) — covering every page this function
/// can classify private — so no ELF-declared data can live there and have its
/// genesis go unbound.
pub fn is_private_input_page(page_base: u64, num_private_input_pages: usize) -> bool {
    use executor::vm::memory::PRIVATE_INPUT_START_INDEX;
    let page_size = DEFAULT_PAGE_SIZE as u64;
    let end = PRIVATE_INPUT_START_INDEX + num_private_input_pages as u64 * page_size;
    (PRIVATE_INPUT_START_INDEX..end).contains(&page_base)
}

/// The page bases of the first `num_private_input_pages` private-input pages, in
/// ascending order — the enumeration counterpart of [`is_private_input_page`]
/// (`is_private_input_page(b, n)` holds exactly for the aligned bases this yields).
pub fn private_input_page_bases(num_private_input_pages: usize) -> impl Iterator<Item = u64> {
    use executor::vm::memory::PRIVATE_INPUT_START_INDEX;
    let page_size = DEFAULT_PAGE_SIZE as u64;
    (0..num_private_input_pages as u64).map(move |i| PRIVATE_INPUT_START_INDEX + i * page_size)
}

/// Upper bound on `num_private_input_pages` any honest proof can claim: the span of
/// a MAX-size input including its length prefix — no slack (an honest max-size
/// input occupies exactly this many pages). Both the monolithic and continuation
/// verifiers bound the deserialized, untrusted count with this before sizing AIRs.
pub fn max_private_input_pages() -> usize {
    use executor::vm::memory::{MAX_PRIVATE_INPUT_SIZE, PRIVATE_INPUT_LENGTH_PREFIX_BYTES};
    (MAX_PRIVATE_INPUT_SIZE as usize + PRIVATE_INPUT_LENGTH_PREFIX_BYTES)
        .div_ceil(DEFAULT_PAGE_SIZE)
}

// =========================================================================
// Trace generation
// =========================================================================

/// Generates a PAGE trace table for a single page.
///
/// ## Arguments
///
/// * `config` - Page configuration (base address, size, initial values)
/// * `final_state` - Map from byte address to final (timestamp, value) for accessed bytes
///
/// ## Returns
///
/// The trace table for this page.
pub fn generate_page_trace(
    config: &PageConfig,
    final_state: &FinalStateMap,
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let page_size = DEFAULT_PAGE_SIZE;
    let page_base = config.page_base;

    // Page base must be page-aligned
    assert!(
        page_base.is_multiple_of(page_size as u64),
        "Page base must be page-aligned"
    );

    let num_rows = page_size; // One row per byte in the page
    let mut trace = TraceTable::new_main(
        vec![FE::zero(); num_rows * cols::NUM_COLUMNS],
        cols::NUM_COLUMNS,
        1,
    );
    let table = &mut trace.main_table;

    for offset in 0..page_size {
        let byte_addr = page_base + (offset as u64);

        // Offset (preprocessed) - address is virtual: page_base + offset
        table.set_u64(offset, cols::OFFSET, offset as u64);

        // Initial value (init_values may be shorter than the page → trailing zeros)
        let init_value = config
            .init_values
            .as_ref()
            .and_then(|v| v.get(offset).copied())
            .unwrap_or(0);
        table.set_byte(offset, cols::INIT, init_value);

        // Final state: if accessed use final, otherwise use initial
        let (timestamp, fini_value) = if let Some(state) = final_state.get(&byte_addr) {
            (state.timestamp, state.value)
        } else {
            // Never accessed: timestamp=0, fini=init
            (0, init_value)
        };

        table.set_byte(offset, cols::FINI, fini_value);
        table.set_dword_wl(offset, cols::TIMESTAMP_LO, timestamp);
    }

    trace
}

// =========================================================================
// Preprocessed commitment
// =========================================================================

/// Returns the static zero-init PAGE preprocessed commitment for
/// `blowup_factor`, or `None` if no value is shipped for it. Values were
/// generated by the `compute_static_commitments` binary at the project's
/// standard `coset_offset = 3` (the value every in-tree `ProofOptions`
/// constructor pins) and pinned by
/// `zero_page_static_matches_recompute_for_all_blowups` so any drift in the
/// AIR or FFT pipeline is caught at test time. The verifier reads these
/// from its compiled binary — no input data is trusted.
///
/// Because OFFSET is page-relative (`0..DEFAULT_PAGE_SIZE-1`) and INIT is
/// uniformly zero for zero-init pages, the commitment depends only on the
/// blowup factor — not on `page_base` or the program being verified. A
/// single entry covers every zero-init page in the system.
///
/// # Regenerating
///
/// Only regenerate these match arms after a *deliberate, reviewed* change
/// to the PAGE table layout, the AIR's preprocessed column count, or the
/// FFT / LDE / Merkle pipeline. Run:
///
/// ```text
/// cargo run --bin compute_static_commitments --release
/// ```
///
/// and paste the printed match arms over the ones below.
///
/// **If a drift test failed, do not regenerate first.** The drift tests
/// exist to force a human to ask "why did this change?" before the new
/// bytes get blessed. Re-pasting on a drift failure silently launders an
/// unintended table change into the verifier's compiled-in trust anchor.
pub(crate) fn static_zero_page_commitment(blowup_factor: u8) -> Option<Commitment> {
    match blowup_factor {
        2 => Some([
            0x7d, 0x74, 0x85, 0xf0, 0x2b, 0x74, 0xe0, 0x3f, 0x14, 0x99, 0xb3, 0xa0, 0x5f, 0x1d,
            0x6e, 0xf2, 0x21, 0xff, 0xaf, 0x24, 0x7e, 0x30, 0xb0, 0xda, 0x48, 0x79, 0xe1, 0x43,
            0xee, 0xea, 0x6a, 0x0f,
        ]),
        4 => Some([
            0x5c, 0xcc, 0x5b, 0xb1, 0xe8, 0x11, 0x91, 0x81, 0xbd, 0xdd, 0x39, 0x40, 0x77, 0x87,
            0xdc, 0x98, 0x06, 0x06, 0x8c, 0x63, 0xcd, 0xfd, 0xf1, 0xda, 0x4a, 0x55, 0x31, 0x4d,
            0x6a, 0x16, 0x18, 0xd0,
        ]),
        8 => Some([
            0xf0, 0xc0, 0x69, 0xed, 0xf8, 0x59, 0xd6, 0x56, 0x15, 0x3c, 0x2f, 0x93, 0x65, 0xd6,
            0xe9, 0xe9, 0x8e, 0xd1, 0x83, 0x94, 0xf9, 0x75, 0x59, 0xd1, 0xec, 0x16, 0xe1, 0x37,
            0xd5, 0x32, 0xd6, 0xd9,
        ]),
        _ => None,
    }
}

/// Computes the Merkle root commitment over the LDE of PAGE precomputed columns.
///
/// The commitment covers OFFSET (0..page_size-1) and INIT (from config).
/// Each page may have different INIT data, producing a different commitment.
///
/// For zero-init pages, prefer [`zero_init_preprocessed_commitment`], which
/// returns a compile-time constant for the standard proof options instead
/// of rebuilding the FFT + Merkle tree.
pub fn compute_precomputed_commitment(config: &PageConfig, options: &ProofOptions) -> Commitment {
    let page_size = DEFAULT_PAGE_SIZE;
    let num_rows = page_size;

    // Precomputed columns: OFFSET and INIT.
    //
    // OFFSET (col 0): deterministic row index 0..page_size-1, the same for every
    //   page of a given size regardless of the program being proven.
    //
    // INIT (col 1): the initial byte value at each offset. For zero-init pages
    //   (stack, heap, BSS) this is all zeros. For ELF data pages it holds the
    //   bytes loaded from the binary. Either way the column is fully determined
    //   before execution, so the verifier can check it against a preprocessed
    //   commitment instead of including it in the main trace.
    let mut offset_col = vec![FE::zero(); num_rows];
    let mut init_col = vec![FE::zero(); num_rows];

    for i in 0..page_size {
        offset_col[i] = FE::from(i as u64);
        let init_byte = config
            .init_values
            .as_ref()
            .and_then(|v| v.get(i).copied())
            .unwrap_or(0);
        init_col[i] = FE::from(init_byte as u64);
    }

    let columns = [offset_col, init_col];

    let polys: Vec<Polynomial<FE>> = columns
        .iter()
        .map(|col| {
            Polynomial::interpolate_fft::<GoldilocksField>(col)
                .expect("FFT interpolation failed for page column")
        })
        .collect();

    let blowup_factor = options.blowup_factor as usize;
    let coset_offset = FE::from(options.coset_offset);
    let lde_columns: Vec<Vec<FE>> = polys
        .iter()
        .map(|poly| {
            evaluate_polynomial_on_lde_domain(poly, blowup_factor, num_rows, &coset_offset)
                .expect("LDE evaluation failed for page polynomial")
        })
        .collect();

    let (_, root) = commit_bit_reversed(&lde_columns, ROWS_PER_LEAF)
        .expect("Failed to build Merkle tree for page LDE");
    root
}

/// Returns the zero-init PAGE preprocessed commitment.
///
/// Looks up `blowup_factor` in [`static_zero_page_commitment`] when
/// `coset_offset == 3` (the value the static bytes were generated for); on
/// miss — either a non-3 coset or a `blowup_factor` outside the shipped
/// match arms — logs a warning and recomputes from scratch. ELF data pages
/// have program-dependent INIT columns and no static entry; compute their
/// commitments with [`compute_precomputed_commitment`] directly.
pub fn zero_init_preprocessed_commitment(options: &ProofOptions) -> Commitment {
    if options.coset_offset == 3
        && let Some(commitment) = static_zero_page_commitment(options.blowup_factor)
    {
        return commitment;
    }
    log::warn!(
        "zero-init page preprocessed commitment not static for \
         (blowup={}, coset={}); falling back to recompute. Add a match \
         arm to `static_zero_page_commitment` by running \
         `cargo run --bin compute_static_commitments --release`.",
        options.blowup_factor,
        options.coset_offset,
    );
    compute_precomputed_commitment(&PageConfig::zero_init(0), options)
}

// =========================================================================
// Bus interactions
// =========================================================================

/// Creates all bus interactions for a PAGE table.
///
/// The `page_base` is the constant base address for this page instance.
/// The virtual address is computed as `page_base + offset` using linear combination.
///
/// ## Bus Interactions
///
/// - PAGE-C1+C2: ARE_BYTES[init, fini] - sender, multiplicity 1 (batched range check)
/// - PAGE-C3: memory[0, address, 0, init] - receiver, multiplicity -1
/// - PAGE-C4: memory[0, address, timestamp, fini] - sender, multiplicity 1
///
/// ## Arguments
///
/// * `page_base` - The base address for this page (constant per table instance)
pub fn bus_interactions(page_base: u64) -> Vec<BusInteraction> {
    // Split page_base into lo/hi 32-bit parts
    let page_base_lo = page_base & 0xFFFF_FFFF;
    let page_base_hi = page_base >> 32;

    // Address computation: address_lo = page_base_lo + offset (linear combination)
    // address_hi = page_base_hi (constant, since offset < page_size < 2^32)
    let address_lo = BusValue::linear(vec![
        LinearTerm::Constant(page_base_lo as i64),
        LinearTerm::Column {
            coefficient: 1,
            column: cols::OFFSET,
        },
    ]);
    let address_hi = BusValue::constant(page_base_hi);

    vec![
        // PAGE-C1+C2: ARE_BYTES[init, fini] - range check both byte values in one interaction
        BusInteraction::sender(
            BusId::AreBytes,
            Multiplicity::One,
            vec![
                BusValue::Packed {
                    start_column: cols::INIT,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::FINI,
                    packing: Packing::Direct,
                },
            ],
        ),
        // PAGE-C3: memory[0, address, 0, init] - receive initial memory token
        BusInteraction::receiver(
            BusId::Memory,
            Multiplicity::One,
            vec![
                // is_register = 0
                BusValue::constant(0),
                // address_lo = page_base_lo + offset
                address_lo.clone(),
                // address_hi = page_base_hi
                address_hi.clone(),
                // timestamp_lo = 0 (initial)
                BusValue::constant(0),
                // timestamp_hi = 0
                BusValue::constant(0),
                // value = init
                BusValue::Packed {
                    start_column: cols::INIT,
                    packing: Packing::Direct,
                },
            ],
        ),
        // PAGE-C4: memory[0, address, timestamp, fini] - send final token
        BusInteraction::sender(
            BusId::Memory,
            Multiplicity::One,
            vec![
                // is_register = 0
                BusValue::constant(0),
                // address_lo = page_base_lo + offset
                address_lo,
                // address_hi = page_base_hi
                address_hi,
                // timestamp_lo (final)
                BusValue::Packed {
                    start_column: cols::TIMESTAMP_LO,
                    packing: Packing::Direct,
                },
                // timestamp_hi (final)
                BusValue::Packed {
                    start_column: cols::TIMESTAMP_HI,
                    packing: Packing::Direct,
                },
                // value = fini
                BusValue::Packed {
                    start_column: cols::FINI,
                    packing: Packing::Direct,
                },
            ],
        ),
    ]
}

// =========================================================================
// Helper functions for page management
// =========================================================================

/// Compute the page base address for a given byte address.
pub fn page_base_for_address(addr: u64) -> u64 {
    addr & !(DEFAULT_PAGE_SIZE as u64 - 1)
}

/// Compute the offset within a page for a given byte address.
pub fn offset_in_page(addr: u64) -> usize {
    (addr & (DEFAULT_PAGE_SIZE as u64 - 1)) as usize
}
