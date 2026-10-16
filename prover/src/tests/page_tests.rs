//! Tests for the PAGE table.

use executor::elf::Elf;
use stark::proof::options::GoldilocksCubicProofOptions;

use crate::tables::page::*;
use crate::tables::trace_builder::Traces;
use crate::tables::types::*;
use crate::test_utils::asm_elf_bytes;
use crate::{prove, verify_with_options};

#[test]
fn test_page_base_for_address() {
    // DEFAULT_PAGE_SIZE = 1 << 18 = 0x40000
    assert_eq!(page_base_for_address(0x00000), 0x00000);
    assert_eq!(page_base_for_address(0x1000), 0x00000); // 0x1000 < 0x40000
    assert_eq!(page_base_for_address(0x3FFFF), 0x00000); // last byte of first page
    assert_eq!(page_base_for_address(0x40000), 0x40000); // start of second page
    assert_eq!(page_base_for_address(0x40001), 0x40000); // one byte into second page
}

#[test]
fn test_offset_in_page() {
    // DEFAULT_PAGE_SIZE = 0x40000
    assert_eq!(offset_in_page(0x00000), 0);
    assert_eq!(offset_in_page(0x1000), 0x1000); // 4096
    assert_eq!(offset_in_page(0x3FFFF), 0x3FFFF); // last offset in first page
    assert_eq!(offset_in_page(0x40000), 0); // start of second page → offset 0
    assert_eq!(offset_in_page(0x40001), 1); // one byte into second page
}

#[test]
fn test_generate_page_trace_zero_init() {
    // Use 0 as page_base (aligned to DEFAULT_PAGE_SIZE = 256KB)
    let config = PageConfig::zero_init(0);
    let final_state = FinalStateMap::new();

    let trace = generate_page_trace(&config, &final_state);

    // Trace must have DEFAULT_PAGE_SIZE rows
    assert_eq!(trace.num_rows(), DEFAULT_PAGE_SIZE);

    // Sample: first row
    assert_eq!(*trace.main_table.get(0, cols::OFFSET), FE::zero());
    assert_eq!(*trace.main_table.get(0, cols::INIT), FE::zero());
    assert_eq!(*trace.main_table.get(0, cols::FINI), FE::zero());
    assert_eq!(*trace.main_table.get(0, cols::TIMESTAMP_LO), FE::zero());

    // Sample: some middle row (offset 42)
    assert_eq!(*trace.main_table.get(42, cols::OFFSET), FE::from(42u64));
    assert_eq!(*trace.main_table.get(42, cols::INIT), FE::zero());
    assert_eq!(*trace.main_table.get(42, cols::FINI), FE::zero());

    // Sample: last row
    let last = DEFAULT_PAGE_SIZE - 1;
    assert_eq!(
        *trace.main_table.get(last, cols::OFFSET),
        FE::from(last as u64)
    );
    assert_eq!(*trace.main_table.get(last, cols::INIT), FE::zero());
    assert_eq!(*trace.main_table.get(last, cols::FINI), FE::zero());
}

#[test]
fn test_generate_page_trace_with_data() {
    // Use 0 as page_base (aligned to DEFAULT_PAGE_SIZE = 256KB)
    let data = vec![0x01, 0x02, 0x03, 0x04];
    let config = PageConfig::with_data(0, data);
    let final_state = FinalStateMap::new();

    let trace = generate_page_trace(&config, &final_state);

    // Trace must have DEFAULT_PAGE_SIZE rows
    assert_eq!(trace.num_rows(), DEFAULT_PAGE_SIZE);

    // Check initial values from data (first 4 bytes)
    assert_eq!(*trace.main_table.get(0, cols::INIT), FE::from(0x01u64));
    assert_eq!(*trace.main_table.get(1, cols::INIT), FE::from(0x02u64));
    assert_eq!(*trace.main_table.get(2, cols::INIT), FE::from(0x03u64));
    assert_eq!(*trace.main_table.get(3, cols::INIT), FE::from(0x04u64));
    // Bytes past the supplied data should be zero (trailing zeros)
    assert_eq!(*trace.main_table.get(4, cols::INIT), FE::zero());
    assert_eq!(*trace.main_table.get(100, cols::INIT), FE::zero());

    // Without accesses, fini should equal init
    assert_eq!(*trace.main_table.get(0, cols::FINI), FE::from(0x01u64));
    assert_eq!(*trace.main_table.get(3, cols::FINI), FE::from(0x04u64));
    assert_eq!(*trace.main_table.get(4, cols::FINI), FE::zero());

    // OFFSET column must equal row index
    assert_eq!(*trace.main_table.get(0, cols::OFFSET), FE::from(0u64));
    assert_eq!(*trace.main_table.get(3, cols::OFFSET), FE::from(3u64));
}

#[test]
fn test_generate_page_trace_with_accesses() {
    // Use 0 as page_base (aligned to DEFAULT_PAGE_SIZE = 256KB)
    let data = vec![0xAA, 0xBB];
    let config = PageConfig::with_data(0, data);

    let mut final_state = FinalStateMap::new();
    // Address 0 (= page_base + offset 0) was written with value 0xFF at timestamp 100
    final_state.insert(
        0,
        FinalByteState {
            timestamp: 100,
            value: 0xFF,
        },
    );

    let trace = generate_page_trace(&config, &final_state);

    // Row 0: address 0 (offset 0) - was accessed
    assert_eq!(*trace.main_table.get(0, cols::INIT), FE::from(0xAAu64));
    assert_eq!(*trace.main_table.get(0, cols::FINI), FE::from(0xFFu64));
    assert_eq!(
        *trace.main_table.get(0, cols::TIMESTAMP_LO),
        FE::from(100u64)
    );

    // Row 1: address 1 (offset 1) - not accessed, fini = init = 0xBB
    assert_eq!(*trace.main_table.get(1, cols::INIT), FE::from(0xBBu64));
    assert_eq!(*trace.main_table.get(1, cols::FINI), FE::from(0xBBu64));
    assert_eq!(*trace.main_table.get(1, cols::TIMESTAMP_LO), FE::zero());

    // Row 2: not in data, not accessed — init=0, fini=0
    assert_eq!(*trace.main_table.get(2, cols::INIT), FE::zero());
    assert_eq!(*trace.main_table.get(2, cols::FINI), FE::zero());
    assert_eq!(*trace.main_table.get(2, cols::TIMESTAMP_LO), FE::zero());

    // OFFSET column must equal row index
    assert_eq!(*trace.main_table.get(0, cols::OFFSET), FE::from(0u64));
    assert_eq!(*trace.main_table.get(1, cols::OFFSET), FE::from(1u64));
}

#[test]
fn test_bus_interactions() {
    let interactions = bus_interactions(0); // page_base = 0
    assert_eq!(interactions.len(), 3); // C1+C2 (batched ARE_BYTES), C3, C4
}

#[test]
fn test_bus_interactions_high_address() {
    // Test with high address like stack region
    let stack_page = STACK_TOP & !(DEFAULT_PAGE_SIZE as u64 - 1);
    let interactions = bus_interactions(stack_page);
    assert_eq!(interactions.len(), 3);
}

// =========================================================================
// verify_with_options: optional page_commitments parameter
// =========================================================================

/// Compute the correct ELF-data-page commitments for the given ELF + options.
/// Returns `(page_base, commitment)` pairs for every non-private, non-zero-init
/// page in the verifier's reconstructed page set.
fn elf_data_page_commitments(
    elf_bytes: &[u8],
    vm_proof: &crate::VmProof,
    options: &stark::proof::options::ProofOptions,
) -> Vec<(u64, stark::config::Commitment)> {
    let elf = Elf::load(elf_bytes).expect("ELF load");
    let page_configs = Traces::page_configs_from_elf_and_runtime(
        &elf,
        &vm_proof.runtime_page_ranges,
        vm_proof.num_private_input_pages,
        usize::MAX,
    )
    .expect("honest page layout");
    page_configs
        .iter()
        .filter(|c| !c.is_private_input && c.init_values.is_some())
        .map(|c| (c.page_base, compute_precomputed_commitment(c, options)))
        .collect()
}

#[test]
fn page_commitments_some_matches_default_path() {
    let elf_bytes = asm_elf_bytes("sub");
    let vm_proof = prove(&elf_bytes).expect("prove failed");
    let options = GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 valid");

    let list = elf_data_page_commitments(&elf_bytes, &vm_proof, &options);
    assert!(
        !list.is_empty(),
        "test ELF must have at least one ELF data page for this test to be meaningful",
    );

    let default_ok = verify_with_options(&vm_proof, &elf_bytes, &options, None, None)
        .expect("verify with None should not error");
    let explicit_ok = verify_with_options(&vm_proof, &elf_bytes, &options, None, Some(&list))
        .expect("verify with Some(correct) should not error");

    assert!(default_ok, "default path must accept the proof");
    assert!(
        explicit_ok,
        "Some(correct_page_commitments) must accept the proof"
    );
}

#[test]
fn page_commitments_wrong_value_rejects() {
    let elf_bytes = asm_elf_bytes("sub");
    let vm_proof = prove(&elf_bytes).expect("prove failed");
    let options = GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 valid");

    let mut list = elf_data_page_commitments(&elf_bytes, &vm_proof, &options);
    assert!(!list.is_empty(), "test ELF must have ≥ 1 ELF data page");
    // Flip a byte in the first page's commitment so the Fiat-Shamir transcripts diverge.
    list[0].1[0] ^= 0xFF;

    let result = verify_with_options(&vm_proof, &elf_bytes, &options, None, Some(&list))
        .expect("verify must not return Err — Fiat-Shamir mismatch is Ok(false)");
    assert!(
        !result,
        "tampered page commitment must cause Fiat-Shamir rejection",
    );
}

#[test]
fn page_commitments_zero_bytes_rejects() {
    let elf_bytes = asm_elf_bytes("sub");
    let vm_proof = prove(&elf_bytes).expect("prove failed");
    let options = GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 valid");

    let mut list = elf_data_page_commitments(&elf_bytes, &vm_proof, &options);
    assert!(!list.is_empty(), "test ELF must have ≥ 1 ELF data page");
    // [0u8; 32] is the most plausible accidental default — passing it must
    // not pass verification.
    list[0].1 = [0u8; 32];

    let result = verify_with_options(&vm_proof, &elf_bytes, &options, None, Some(&list))
        .expect("verify must not return Err — Fiat-Shamir mismatch is Ok(false)");
    assert!(
        !result,
        "all-zero page commitment must cause Fiat-Shamir rejection",
    );
}

#[test]
fn page_commitments_empty_list_matches_none() {
    let elf_bytes = asm_elf_bytes("sub");
    let vm_proof = prove(&elf_bytes).expect("prove failed");
    let options = GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 valid");

    let empty: [(u64, stark::config::Commitment); 0] = [];
    let result = verify_with_options(&vm_proof, &elf_bytes, &options, None, Some(&empty))
        .expect("verify with empty list should not error");
    assert!(
        result,
        "empty page_commitments slice must behave like None — every page falls through to recompute",
    );
}

/// Differential test: the dense PAGE generator must produce a byte-identical
/// trace to the sparse `FinalStateMap` generator it replaces, for both the
/// monolithic (`exclude_touched = false`) and continuation-epoch
/// (`exclude_touched = true`) cases. Locks the PR's "exact same PAGE trace"
/// claim directly, instead of relying only on full prove+verify integration.
///
/// Compares `main_table` (PAGE is a main-only table). `Table<F>: PartialEq` holds
/// in both configs (derived without `disk-spill`, hand-impl'd with it), so no
/// feature gate is needed. The whole `TraceTable` still can't be compared because
/// its `aux_table: Table<E>` param needs `E: PartialEq`, which
/// `Degree3GoldilocksExtensionField` lacks.
#[test]
fn generate_page_trace_dense_matches_sparse() {
    use crate::paged_mem::PagedMem;

    let page_base = 0u64;
    let init_bytes = vec![0x01u8, 0x02, 0x03, 0x04];
    let config = PageConfig::with_data(page_base, init_bytes.clone());

    // Mirror production (`MemoryState::from_image` + replay): seed the initial
    // image as (byte, ts=0), then apply runtime writes at ts > 0 (production
    // uses ts = i*4 + 4, always >= 4).
    let mut cells: PagedMem<(u8, u64)> = PagedMem::new((0, 0));
    for (off, &b) in init_bytes.iter().enumerate() {
        cells.set(page_base + off as u64, (b, 0)); // image seed (ts 0)
    }
    cells.set(page_base, (0xFF, 100)); // overwrite an init byte at runtime
    cells.set(page_base + 10, (0x77, 48)); // write a previously-untouched cell

    for exclude_touched in [false, true] {
        // Old path: sparse FinalStateMap, filtered exactly as trace_builder did.
        let final_state: FinalStateMap = cells
            .iter()
            .filter(|(_, cell)| !exclude_touched || cell.1 == 0)
            .map(|(addr, (value, timestamp))| (addr, FinalByteState { timestamp, value }))
            .collect();
        let sparse = generate_page_trace(&config, &final_state);

        // New path: dense per-page slice.
        let dense =
            generate_page_trace_from_dense(&config, cells.page_data(page_base), exclude_touched);

        assert_eq!(
            sparse.main_table, dense.main_table,
            "mismatch with exclude_touched = {exclude_touched}"
        );
    }
}
