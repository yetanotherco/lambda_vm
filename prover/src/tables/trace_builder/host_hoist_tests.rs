//! `LAMBDA_VM_HOST_HOIST` builds the tables it builds without it: the page
//! layout computed once from the image's whole pages is the layout computed
//! a byte at a time, page for page, on real runs with and without a private
//! input; the image written a run at a time is the image written a byte at a
//! time, and its memory state mapped on the pool the state mapped in turn;
//! and a streamed finish with each part on hands on the same header and the
//! same tables, word for word, under the word and the byte memory formats.
//! With the knob off nothing takes a new path.

use executor::elf::Elf;
use executor::vm::execution::Executor;
use executor::vm::logs::Log;

use super::finish_tests::shape_and_words;
use super::gate::ByteGate;
use super::*;
use crate::tables::MaxRowsConfig;
use crate::test_utils::asm_elf_bytes;

/// The programs: loads and stores of every width over the image, and a
/// keccak run (its state pointers are page cells).
const PROGRAMS: [&str; 2] = ["fused_mem_cases", "test_keccak256"];

/// Private inputs: none, a few bytes, and one that spans a page end and ends
/// inside the next page.
fn inputs() -> Vec<Vec<u8>> {
    let long: Vec<u8> = (0..page::DEFAULT_PAGE_SIZE + 777)
        .map(|i| (i * 31 + 7) as u8)
        .collect();
    vec![Vec::new(), (1..=100u8).collect(), long]
}

/// `name` run on `input`.
fn run_on(name: &str, input: &[u8]) -> (Elf, Vec<Log>) {
    let program = Elf::load(&asm_elf_bytes(name)).expect("the ELF loads");
    let logs = Executor::new(&program, input.to_vec())
        .expect("the executor starts")
        .run()
        .expect("the program runs")
        .logs;
    (program, logs)
}

/// A config's fields, to compare.
fn fields(config: &PageConfig) -> (u64, Option<Vec<u8>>, bool) {
    (
        config.page_base,
        config.init_values.clone(),
        config.is_private_input,
    )
}

/// The whole-pages layout is the byte-at-a-time layout, config for config,
/// on each program's final memory state, and both are the layout the image
/// as a map of bytes gives; a private input makes private-input pages.
#[test]
fn the_hoisted_page_layout_is_the_page_layout() {
    for name in PROGRAMS {
        for input in inputs() {
            let (program, logs) = run_on(name, &input);
            let artifacts = DecodeArtifacts::from_elf(&program).expect("the artifacts");
            let paged = build_initial_image_paged(&program, &input);
            let bytes = build_initial_image(&program, &input);
            let register_init = register::register_init_from_entry_point(program.entry_point);
            let CollectedEpoch { memory_state, .. } =
                Traces::collect_epoch(&artifacts, &paged, &register_init, &logs, true)
                    .expect("collected");
            let today: Vec<_> = page_configs_of(&paged, &memory_state, &input)
                .iter()
                .map(fields)
                .collect();
            let hoisted: Vec<_> = page_configs_hoisted(&paged, &memory_state, &input)
                .iter()
                .map(fields)
                .collect();
            let of_bytes: Vec<_> = page_configs_of(&bytes, &memory_state, &input)
                .iter()
                .map(fields)
                .collect();
            let what = format!("{name}, {} input bytes", input.len());
            assert_eq!(hoisted, today, "{what}");
            assert_eq!(of_bytes, today, "{what}: the image as bytes");
            let private = today.iter().filter(|(_, _, private)| *private).count();
            assert_eq!(
                private > 0,
                !input.is_empty(),
                "{what}: private-input pages"
            );
            assert!(
                today
                    .iter()
                    .any(|(_, init, private)| init.is_some() && !private),
                "{what}: an ELF data page"
            );
            assert!(
                today.iter().any(|(_, init, _)| init.is_none()),
                "{what}: a zero-init page"
            );
            if input.len() > page::DEFAULT_PAGE_SIZE {
                assert!(private >= 2, "{what}: the input spans two pages");
            }
        }
    }
}

type Configure = Box<dyn for<'b> Fn(WindowedTraceBuilder<'b>) -> WindowedTraceBuilder<'b>>;

/// The finish's configurations: the block's under the word memory format
/// (BITWISE counted beside the builds), the word format with BITWISE counted
/// first, and the byte memory format beside the builds.
fn configs() -> Vec<(&'static str, Configure)> {
    fn words(b: WindowedTraceBuilder<'_>) -> WindowedTraceBuilder<'_> {
        use crate::tables::fused_memory::MemoryFormat;
        use crate::tables::memw_aligned::MemwAlignedOrder;
        use crate::tables::register::RegisterFormat;
        b.register_format(RegisterFormat::InRow)
            .expect("before any window")
            .memw_aligned_order(MemwAlignedOrder::Limbs)
            .expect("before any window")
            .memory_format(MemoryFormat::Words)
            .expect("before any window")
    }
    fn cut(b: WindowedTraceBuilder<'_>) -> WindowedTraceBuilder<'_> {
        b.keccak_rnd_chunks_at_finish(32)
            .expect("rows")
            .cuts_at_finish(1, 1, 16, 1)
            .expect("rows")
            .pack_finished_tables()
            .generate_packed()
            .drop_streamed_ops()
            .expect("before any window")
            .lt_merge()
            .p4_rows_on_pool()
            .p4_lookups_direct()
    }
    vec![
        (
            "words, beside",
            Box::new(|b| cut(words(b)).rest_bitwise_last().slide_rest()),
        ),
        ("words, first", Box::new(|b| cut(words(b)).slide_rest())),
        (
            "bytes, beside",
            Box::new(|b| cut(b).rest_bitwise_last().slide_rest()),
        ),
    ]
}

/// The run's streamed finish under `configure` with `hoist`: the header's
/// page configs, counts and public output, and every table handed on as
/// (position, shape and words); and how often the layout was computed each
/// way and the image written a run at a time.
#[allow(clippy::type_complexity)]
fn streamed(
    program: &Elf,
    logs: &[Log],
    input: &[u8],
    configure: &Configure,
    hoist: HostHoist,
) -> (
    (Vec<(u64, Option<Vec<u8>>, bool)>, String, Vec<u8>),
    Vec<(usize, Option<(usize, usize, Vec<u64>)>)>,
    (usize, usize, usize),
) {
    let max_rows = MaxRowsConfig::small();
    let _ = super::host_hoist::calls::take_runs();
    let builder = configure(
        WindowedTraceBuilder::new_with(program, input, &max_rows, hoist).expect("the builder"),
    );
    let runs = super::host_hoist::calls::take_runs();
    assert_eq!(builder.streamed().host_hoist, hoist);
    let window = 64;
    let body = logs.len() - 1;
    let cut = body - body % window;
    let mut builder = builder;
    for w in logs[..cut].chunks(window) {
        builder.push(w).expect("a window");
    }
    let _ = super::host_hoist::calls::take();
    let gate = ByteGate::new(usize::MAX);
    let mut header = None;
    let mut out = Vec::new();
    let mut placed = 0usize;
    builder
        .finish_streamed(&logs[cut..], &gate, |h| {
            header = Some(h);
            Ok(Box::new(|emitted: Emitted| {
                if emitted.table.is_some() {
                    gate.release(placed);
                    placed += 1;
                }
                out.push((
                    emitted.position,
                    emitted.table.as_ref().map(shape_and_words),
                ));
                Ok(())
            }))
        })
        .expect("the streamed finish");
    let (by_byte, hoisted) = super::host_hoist::calls::take();
    let calls = (by_byte, hoisted, runs);
    let header = header.expect("the header came first");
    (
        (
            header.page_configs.iter().map(fields).collect(),
            format!("{:?}", header.table_counts),
            header.public_output_bytes.clone(),
        ),
        out,
        calls,
    )
}

/// ★ A streamed finish with each hoisted part on, alone and together, hands
/// on the header and the tables it hands on with none, word for word, under
/// every configuration and input. Under `pages` it computes the layout once
/// from whole pages, and without it a byte at a time as before, never from
/// pages; under `image` the builder writes its image a run at a time, and
/// without it never.
#[test]
fn the_hoisted_parts_stream_the_same_tables() {
    let parts = [
        ("pages", HostHoist::parse(Some("pages")).expect("a part")),
        ("image", HostHoist::parse(Some("image")).expect("a part")),
        ("all", HostHoist::all()),
    ];
    for name in PROGRAMS {
        for input in inputs() {
            let (program, logs) = run_on(name, &input);
            for (config, configure) in configs() {
                let what = format!("{name}, {} input bytes, {config}", input.len());
                let (header_off, tables_off, calls_off) =
                    streamed(&program, &logs, &input, &configure, HostHoist::default());
                assert_eq!(calls_off.1, 0, "{what}: off never hoists");
                assert!(calls_off.0 >= 1, "{what}: off computes it by byte");
                assert_eq!(calls_off.2, 0, "{what}: off writes no run");
                assert!(
                    tables_off.iter().filter(|(_, t)| t.is_some()).count() > 10,
                    "{what}: tables built"
                );
                for (part, hoist) in parts {
                    let what = format!("{what}, {part}");
                    let (header_on, tables_on, calls_on) =
                        streamed(&program, &logs, &input, &configure, hoist);
                    assert_eq!(header_on, header_off, "{what}: the header");
                    assert_eq!(tables_on.len(), tables_off.len(), "{what}: the tables");
                    for (k, (a, b)) in tables_on.iter().zip(&tables_off).enumerate() {
                        assert_eq!(a, b, "{what}: table {k}");
                    }
                    if hoist.pages {
                        assert_eq!(
                            (calls_on.0, calls_on.1),
                            (0, 1),
                            "{what}: hoisted once, never by byte"
                        );
                    } else {
                        assert_eq!(calls_on.1, 0, "{what}: never hoisted");
                    }
                    assert_eq!(calls_on.2, usize::from(hoist.image), "{what}: runs");
                }
            }
        }
    }
}

/// Equal stores: the same pages, each page's offsets, and the same `set`
/// cells.
fn same_store<T: Copy + PartialEq + std::fmt::Debug>(
    what: &str,
    a: &crate::paged_mem::PagedMem<T>,
    b: &crate::paged_mem::PagedMem<T>,
) {
    let bases: Vec<u64> = a.page_bases().collect();
    assert_eq!(bases, b.page_bases().collect::<Vec<_>>(), "{what}: pages");
    for base in bases {
        assert!(
            a.page_data(base) == b.page_data(base),
            "{what}: page {base:#x}"
        );
    }
    assert!(a.iter().eq(b.iter()), "{what}: the set cells");
}

/// The image written a run at a time is the image written a byte at a time,
/// and the memory state mapped on the pool is the state mapped page by page:
/// on real programs with inputs that end inside a page, at a page's end and
/// past it, and on hand-made segments that overlap (the later wins), cross
/// a page end and wrap the address space.
#[test]
fn the_image_by_runs_is_the_image() {
    let page = page::DEFAULT_PAGE_SIZE as u64;
    let start_offset = page::offset_in_page(executor::vm::memory::PRIVATE_INPUT_START_INDEX);
    let mut programs: Vec<Elf> = Vec::new();
    let mut cases: Vec<(String, usize, Vec<u8>)> = Vec::new();
    for name in [
        "fused_mem_cases",
        "test_keccak256",
        "page_runs",
        "all_instructions_64",
    ] {
        programs.push(Elf::load(&asm_elf_bytes(name)).expect("the ELF loads"));
        let mut all = inputs();
        // The length word and the bytes end exactly at a page's end.
        all.push(vec![9u8; page::DEFAULT_PAGE_SIZE - start_offset - 4]);
        for input in all {
            let what = format!("{name}, {} bytes", input.len());
            cases.push((what, programs.len() - 1, input));
        }
    }
    let segment = |base_addr: u64, words: std::ops::Range<u32>| executor::elf::Segment {
        base_addr,
        values: words
            .map(|w| w.wrapping_mul(0x0101_0101) ^ 0xa5a5_0000)
            .collect(),
        is_executable: false,
    };
    programs.push(Elf {
        entry_point: 0x1000,
        data: vec![
            segment(0x1000, 0..10),
            segment(0x1008, 100..103),
            segment(page - 6, 7..12),
            segment(5 * page, 0..(2 * page as u32 / 4 + 3)),
            segment(u64::MAX - 5, 20..23),
        ],
    });
    cases.push(("hand-made".into(), programs.len() - 1, Vec::new()));
    cases.push((
        "hand-made, input".into(),
        programs.len() - 1,
        (0..50u8).collect(),
    ));
    for (what, k, input) in cases {
        let program = &programs[k];
        let bytes = build_initial_image_paged(program, &input);
        let runs = build_initial_image_by_runs(program, &input);
        same_store(&what, &runs, &bytes);
        assert!(!bytes.is_empty(), "{what}: an image");
        same_store(
            &format!("{what}: the state"),
            &MemoryState::from_pages_on_pool(&runs).cells,
            &MemoryState::from_pages(&bytes).cells,
        );
    }
}

/// With the knob unset no part is on: a builder's default and the stream's
/// default hoist nothing, and `new` writes its image a byte at a time.
#[test]
fn the_knob_off_hoists_nothing() {
    assert_eq!(StreamSkip::default().host_hoist, HostHoist::default());
    assert_eq!(HostHoist::parse(None).expect("unset"), HostHoist::default());
    let program = Elf::load(&asm_elf_bytes("fused_mem_cases")).expect("the ELF loads");
    let _ = super::host_hoist::calls::take_runs();
    let builder =
        WindowedTraceBuilder::new(&program, &[], &MaxRowsConfig::small()).expect("the builder");
    assert_eq!(builder.streamed().host_hoist, HostHoist::default());
    assert_eq!(super::host_hoist::calls::take_runs(), 0);
}
