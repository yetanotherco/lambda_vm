//! The builder's head: the initial image held as pages and the walk's memory
//! state seeded from them page by page are the image as a map of bytes and the
//! state seeded from it byte by byte.

use executor::elf::Elf;

use super::super::{MemoryState, build_initial_image, build_initial_image_paged};
use super::WindowedTraceBuilder;
use crate::tables::MaxRowsConfig;
use crate::test_utils::asm_elf_bytes;

const PROGRAMS: [&str; 3] = ["all_instructions_64", "test_keccak", "lw_sw_offset_odd"];

/// Private inputs of no bytes, a few, and enough to cross a page (the image's
/// pages and the PAGE tables' are 256 KiB).
fn inputs() -> Vec<Vec<u8>> {
    let long = (0..300_000u32)
        .map(|i| (i.wrapping_mul(2_654_435_761) >> 24) as u8)
        .collect();
    vec![Vec::new(), vec![1, 0, 255, 7, 0], long]
}

/// The two states' pages, cells and set cells are equal.
fn same_state(what: &str, a: &MemoryState, b: &MemoryState) {
    let bases: Vec<u64> = a.cells.page_bases().collect();
    assert_eq!(
        bases,
        b.cells.page_bases().collect::<Vec<_>>(),
        "{what}: pages"
    );
    for base in bases {
        assert!(
            a.cells.page_data(base) == b.cells.page_data(base),
            "{what}: the cells of page {base:#x}"
        );
    }
    assert!(a.cells.iter().eq(b.cells.iter()), "{what}: the set cells");
}

/// ★ The image as pages holds the bytes the image as a map holds, and the
/// memory state seeded from its pages is the state seeded byte by byte from
/// the map: the same pages, the same cells, the same set cells.
#[test]
fn the_state_seeded_from_the_pages_is_the_state_seeded_from_the_map() {
    for name in PROGRAMS {
        let program = Elf::load(&asm_elf_bytes(name)).expect("the ELF loads");
        for input in inputs() {
            let what = format!("{name}, {} input bytes", input.len());
            let map = build_initial_image(&program, &input);
            let pages = build_initial_image_paged(&program, &input);
            let mut bytes: Vec<(u64, u8)> = map.iter().map(|(&a, &b)| (a, b)).collect();
            bytes.sort_unstable();
            assert!(
                bytes.iter().copied().eq(pages.iter()),
                "{what}: the image's bytes"
            );
            same_state(
                &what,
                &MemoryState::from_image(&map),
                &MemoryState::from_pages(&pages),
            );
        }
    }
}

/// The builder starts its walk from that state.
#[test]
fn the_builder_walks_from_the_state_seeded_from_the_map() {
    let max_rows = MaxRowsConfig::small();
    for name in PROGRAMS {
        let program = Elf::load(&asm_elf_bytes(name)).expect("the ELF loads");
        for input in inputs() {
            let builder =
                WindowedTraceBuilder::new(&program, &input, &max_rows).expect("the builder");
            same_state(
                &format!("{name}, {} input bytes", input.len()),
                &builder.memory_state,
                &MemoryState::from_image(&build_initial_image(&program, &input)),
            );
        }
    }
}
