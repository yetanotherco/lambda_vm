//! Byte-parity: the dense-read PAGE fill ([`page::generate_page_trace_from_dense`],
//! which reads `(value, timestamp)` straight from the dense per-page store) must
//! produce the exact same trace as the original sparse-`FinalStateMap` fill
//! ([`page::generate_page_trace`]) — for image bytes (ts 0), runtime writes (ts > 0),
//! untouched offsets, and the `exclude_touched` continuation case. PAGE feeds the
//! proof, so this must hold bit-for-bit.

use crate::paged_mem::PagedMem;
use crate::tables::page::{self, DEFAULT_PAGE_SIZE, FinalByteState, FinalStateMap, PageConfig};

fn build() -> (PageConfig, PagedMem<(u8, u64)>) {
    let base = 2 * DEFAULT_PAGE_SIZE as u64; // page-aligned, nonzero
    let mut mem: PagedMem<(u8, u64)> = PagedMem::new((0u8, 0u64));
    // Image bytes (ts 0) + runtime writes (ts > 0), spread across the page incl. ends.
    mem.set(base + 5, (0xAB, 0)); // image byte, never rewritten
    mem.set(base + 6, (0x11, 0));
    mem.set(base + 10, (0xCD, 100)); // runtime write
    mem.set(base + 4096, (0x42, 7));
    mem.set(base + (DEFAULT_PAGE_SIZE as u64 - 1), (0xEF, 50));

    // Init image matches the ts-0 (image) cells; other offsets 0.
    let mut init = vec![0u8; DEFAULT_PAGE_SIZE];
    init[5] = 0xAB;
    init[6] = 0x11;
    let config = PageConfig::with_data(base, init);
    (config, mem)
}

fn hashmap_trace(
    config: &PageConfig,
    mem: &PagedMem<(u8, u64)>,
    exclude_touched: bool,
) -> Vec<u64> {
    let final_state: FinalStateMap = mem
        .iter()
        .filter(|(_, (_, ts))| !exclude_touched || *ts == 0)
        .map(|(addr, (value, timestamp))| (addr, FinalByteState { timestamp, value }))
        .collect();
    let t = page::generate_page_trace(config, &final_state);
    let (fe, _w) = t.main_data_row_major();
    fe.iter()
        .map(|e| unsafe { *(e.value() as *const u64) })
        .collect()
}

fn dense_trace(config: &PageConfig, mem: &PagedMem<(u8, u64)>, exclude_touched: bool) -> Vec<u64> {
    let t = page::generate_page_trace_from_dense(
        config,
        mem.page_data(config.page_base),
        exclude_touched,
    );
    let (fe, _w) = t.main_data_row_major();
    fe.iter()
        .map(|e| unsafe { *(e.value() as *const u64) })
        .collect()
}

#[test]
fn dense_page_fill_matches_hashmap() {
    let (config, mem) = build();
    assert_eq!(
        dense_trace(&config, &mem, false),
        hashmap_trace(&config, &mem, false),
        "dense PAGE fill must match the FinalStateMap fill (monolithic)"
    );
    assert_eq!(
        dense_trace(&config, &mem, true),
        hashmap_trace(&config, &mem, true),
        "dense PAGE fill must match the FinalStateMap fill (exclude_touched)"
    );
}

#[test]
fn dense_page_fill_matches_hashmap_untouched_page() {
    // A page with no runtime cells at all → every offset falls back to (init, 0).
    let base = 3 * DEFAULT_PAGE_SIZE as u64;
    let mem: PagedMem<(u8, u64)> = PagedMem::new((0u8, 0u64));
    let mut init = vec![0u8; DEFAULT_PAGE_SIZE];
    init[0] = 0x7;
    init[99] = 0x9;
    let config = PageConfig::with_data(base, init);
    assert_eq!(
        dense_trace(&config, &mem, false),
        hashmap_trace(&config, &mem, false),
        "dense PAGE fill must match on an untouched page"
    );
}
