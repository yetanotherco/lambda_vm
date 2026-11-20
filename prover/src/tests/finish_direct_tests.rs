//! The finish's tables written straight into their packed form (PAGE by
//! default, `LAMBDA_VM_FINISH_PAGE_DIRECT=0` its rollback;
//! `LAMBDA_VM_FINISH_BITWISE_DIRECT=1`) against
//! today's: built at 8 bytes a cell, then packed. The packed bytes and widths
//! must be the same, so the proof is.

use stark::trace::TraceTable;

use crate::tables::bitwise::{
    self, BitwiseHistogram, BitwiseOperation, BitwiseOperationType, NUM_ROWS,
};
use crate::tables::gpack::{TraceForm, thread_counts};
use crate::tables::page::{
    self, DEFAULT_PAGE_SIZE, PageConfig, generate_page_trace_from_dense,
    generate_page_trace_from_dense_as,
};
use crate::tables::types::{GoldilocksExtension, GoldilocksField};

type Trace = TraceTable<GoldilocksField, GoldilocksExtension>;

fn next(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

/// Traces this thread wrote packed (as they were, narrowed, or again) since
/// `before`: the direct path, not built wide then packed.
fn written_since(before: [usize; 4]) -> usize {
    let now = thread_counts();
    (0..3).map(|i| now[i] - before[i]).sum()
}

/// `trace`'s packed widths and bytes.
fn packed_parts(trace: &Trace, what: &str) -> (Vec<u8>, Vec<u8>) {
    let narrow = trace
        .narrow_main()
        .unwrap_or_else(|| panic!("{what}: not held packed"));
    (narrow.widths().to_vec(), narrow.data().to_vec())
}

/// Today's: the 64-bit table, then packed.
fn packed(mut trace: Trace, what: &str) -> (Vec<u8>, Vec<u8>) {
    assert!(trace.pack_main_narrow(), "{what}: today's table packs");
    packed_parts(&trace, what)
}

/// A page's final cells: about a third written (a value, a timestamp of at
/// least 4), the rest untouched at their init byte.
fn final_page(config: &PageConfig, seed: u64) -> Vec<(u8, u64)> {
    let mut seed = seed | 1;
    (0..DEFAULT_PAGE_SIZE)
        .map(|offset| {
            let init = config
                .init_values
                .as_ref()
                .and_then(|v| v.get(offset).copied())
                .unwrap_or(0);
            if next(&mut seed).is_multiple_of(3) {
                (next(&mut seed) as u8, 4 + next(&mut seed) % (1 << 40))
            } else {
                (init, 0)
            }
        })
        .collect()
}

/// ★ A PAGE table written straight into its packed form is today's packed
/// table byte for byte: zero-init, image-filled and private-input pages, with
/// final cells or none, touched cells kept or excluded; the first trace of the
/// kind (no width hint: built wide, then packed) and the next ones (written
/// packed at the hint's widths).
#[test]
fn a_page_written_packed_is_todays_packed_page() {
    if cfg!(feature = "debug-checks") {
        return; // the debug checks hold every trace at 8 bytes a cell
    }
    let base = 7 * DEFAULT_PAGE_SIZE as u64;
    let data: Vec<u8> = {
        let mut seed = 0xDA7A_u64;
        (0..5000).map(|_| next(&mut seed) as u8).collect()
    };
    let configs = [
        ("zero-init", PageConfig::zero_init(base)),
        ("image", PageConfig::with_data(base, data.clone())),
        ("private input", PageConfig::with_private_input(base, data)),
    ];
    for (k, (name, config)) in configs.iter().enumerate() {
        let cells = final_page(config, 0x9A6E + k as u64);
        for final_cells in [None, Some(&cells[..])] {
            for exclude in [false, true] {
                let what = format!(
                    "{name}, final cells {}, exclude {exclude}",
                    final_cells.is_some()
                );
                let today = packed(
                    generate_page_trace_from_dense(config, final_cells, exclude),
                    &what,
                );
                for pass in 0..2 {
                    let before = thread_counts();
                    let direct = generate_page_trace_from_dense_as(
                        config,
                        final_cells,
                        exclude,
                        TraceForm::Narrow,
                    );
                    assert_eq!(packed_parts(&direct, &what), today, "{what}, pass {pass}");
                    if pass == 1 {
                        assert_eq!(written_since(before), 1, "{what}: written packed");
                    }
                }
            }
        }
    }
    // Wide is today's table, unpacked.
    let config = PageConfig::zero_init(base);
    let wide = generate_page_trace_from_dense_as(&config, None, false, TraceForm::Wide);
    assert!(wide.narrow_main().is_none());
    assert_eq!(
        packed(wide, "wide"),
        packed(
            generate_page_trace_from_dense(&config, None, false),
            "today"
        )
    );
    let _ = page::cols::NUM_COLUMNS;
}

/// ★ BITWISE in one pass, its multiplicities written with its precomputed
/// columns straight into the packed form, is today's two passes then packed,
/// byte for byte: an empty histogram, and one with every lookup type counted,
/// counts past two bytes among them; the first trace and the next.
#[test]
fn bitwise_in_one_packed_pass_is_todays_packed_bitwise() {
    if cfg!(feature = "debug-checks") {
        return;
    }
    let mut counted = BitwiseHistogram::new();
    let mut seed = 0xB17_u64;
    for (t, &kind) in BitwiseOperationType::ALL.iter().enumerate() {
        for _ in 0..20_000 {
            let op = BitwiseOperation {
                lookup_type: kind,
                x: next(&mut seed) as u8,
                y: next(&mut seed) as u8,
                z: (next(&mut seed) % 16) as u8,
            };
            counted.bump_n(op, 1 + next(&mut seed) % 3);
        }
        // A count past two bytes, at a row of its own.
        let op = BitwiseOperation {
            lookup_type: kind,
            x: t as u8,
            y: 0,
            z: 0,
        };
        counted.bump_n(op, 70_000 + t as u64);
    }
    for (name, histogram) in [("empty", BitwiseHistogram::new()), ("counted", counted)] {
        let mut wide = bitwise::generate_bitwise_trace();
        histogram.fill_multiplicities(&mut wide);
        let today = packed(wide, name);
        for pass in 0..2 {
            let before = thread_counts();
            let direct = histogram.trace_as(TraceForm::Narrow);
            assert_eq!(packed_parts(&direct, name), today, "{name}, pass {pass}");
            // Seeded widths: even the first trace is written packed.
            assert_eq!(
                written_since(before),
                1,
                "{name}, pass {pass}: written packed"
            );
        }
    }
    assert_eq!(NUM_ROWS, 1 << 20);
}
