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
use std::sync::OnceLock;

use math::fft::bit_reversing::in_place_bit_reverse_permute;
use math::polynomial::Polynomial;
use stark::config::{BatchedMerkleTree, Commitment};
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::proof::options::ProofOptions;
use stark::prover::evaluate_polynomial_on_lde_domain;
use stark::trace::{TraceTable, columns2rows};

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField};

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
    /// Initial values for the page. If None, all bytes are zero-initialized.
    /// May be shorter than `DEFAULT_PAGE_SIZE`; any missing trailing bytes are
    /// treated as zero. (All pages are `DEFAULT_PAGE_SIZE`; see that constant.)
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
    /// These pages are NOT preprocessed — the verifier never sees the init values.
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
    let mut data = vec![FE::zero(); num_rows * cols::NUM_COLUMNS];

    for offset in 0..page_size {
        let byte_addr = page_base + (offset as u64);
        let base = offset * cols::NUM_COLUMNS;

        // Offset (preprocessed) - address is virtual: page_base + offset
        data[base + cols::OFFSET] = FE::from(offset as u64);

        // Initial value (init_values may be shorter than the page → trailing zeros)
        let init_value = config
            .init_values
            .as_ref()
            .and_then(|v| v.get(offset).copied())
            .unwrap_or(0);
        data[base + cols::INIT] = FE::from(init_value as u64);

        // Final state: if accessed use final, otherwise use initial
        let (timestamp, fini_value) = if let Some(state) = final_state.get(&byte_addr) {
            (state.timestamp, state.value)
        } else {
            // Never accessed: timestamp=0, fini=init
            (0, init_value)
        };

        data[base + cols::FINI] = FE::from(fini_value as u64);
        data[base + cols::TIMESTAMP_LO] = FE::from(timestamp & 0xFFFF_FFFF);
        data[base + cols::TIMESTAMP_HI] = FE::from(timestamp >> 32);
    }

    TraceTable::new_main(data, cols::NUM_COLUMNS, 1)
}

// =========================================================================
// Preprocessed commitment
// =========================================================================

/// Cached commitment for zero-initialized 4KB pages.
/// All zero-init pages of the same size have identical OFFSET and INIT columns.
///
/// INVARIANT: All callers within a process must use identical `ProofOptions`.
/// The cache is keyed only by page content, not by options.
static ZERO_PAGE_4K_COMMITMENT: OnceLock<Commitment> = OnceLock::new();

/// Computes the Merkle root commitment over the LDE of PAGE precomputed columns.
///
/// The commitment covers OFFSET (0..page_size-1) and INIT (from config).
/// Each page may have different INIT data, producing a different commitment.
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
    let mut lde_columns: Vec<Vec<FE>> = polys
        .iter()
        .map(|poly| {
            evaluate_polynomial_on_lde_domain(poly, blowup_factor, num_rows, &coset_offset)
                .expect("LDE evaluation failed for page polynomial")
        })
        .collect();

    for col in lde_columns.iter_mut() {
        in_place_bit_reverse_permute(col);
    }

    let lde_rows = columns2rows(lde_columns);
    let tree = BatchedMerkleTree::<GoldilocksField>::build(&lde_rows)
        .expect("Failed to build Merkle tree for page LDE");
    tree.root
}

/// Returns the preprocessed commitment for a PAGE table, with caching for zero-init pages.
///
/// Zero-init pages of DEFAULT_PAGE_SIZE share a cached commitment.
/// ELF data pages compute their commitment fresh.
pub fn precomputed_commitment_cached(config: &PageConfig, options: &ProofOptions) -> Commitment {
    if config.init_values.is_none() {
        *ZERO_PAGE_4K_COMMITMENT.get_or_init(|| compute_precomputed_commitment(config, options))
    } else {
        compute_precomputed_commitment(config, options)
    }
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
