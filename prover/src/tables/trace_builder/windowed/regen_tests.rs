//! Regeneration's pieces of the windowed builder (D-REGEN): the first-touch
//! census of a walked window.

use executor::elf::Elf;
use executor::vm::execution::Executor;

use super::WindowedTraceBuilder;
use crate::tables::MaxRowsConfig;
use crate::test_utils::asm_elf_bytes;

/// [`super::WalkedWindow::first_touch_bytes`] counts each byte a window touches
/// once: on the keccak programs it is the number of distinct memory bytes the
/// window's accesses cover, window by window, at windows of 7 and 33 cycles.
#[test]
fn first_touch_bytes_are_the_distinct_bytes_a_window_touches() {
    for name in ["test_keccak", "test_keccak_multi", "lw_sw_offset_odd"] {
        let program = Elf::load(&asm_elf_bytes(name)).expect("the ELF loads");
        let logs = Executor::new(&program, Vec::new())
            .expect("the executor starts")
            .run()
            .expect("the program runs")
            .logs;
        for window in [7, 33] {
            let mut builder = WindowedTraceBuilder::new(&program, &[], &MaxRowsConfig::small())
                .expect("the builder");
            let (mut walker, _) = builder.split();
            let mut touched_any = false;
            for (i, w) in logs[..logs.len() - 1].chunks(window).enumerate() {
                let walked = walker.walk(w).expect("a window");
                assert_eq!(walked.cycles(), w.len());
                let memw = &walked.walk.memw;
                let distinct: std::collections::BTreeSet<u64> = memw
                    .aligned
                    .iter()
                    .filter(|row| !row.is_register())
                    .map(|row| (row.base_address(), row.width()))
                    .chain(
                        memw.general
                            .iter()
                            .filter(|op| !op.is_register)
                            .map(|op| (op.base_address, op.width)),
                    )
                    .flat_map(|(base, width)| {
                        (0..u64::from(width)).map(move |k| base.wrapping_add(k))
                    })
                    .collect();
                assert_eq!(
                    walked.first_touch_bytes(),
                    distinct.len(),
                    "{name}, windows of {window}, window {i}"
                );
                touched_any |= !distinct.is_empty();
            }
            assert!(touched_any, "{name}: no window touched memory");
        }
    }
}
