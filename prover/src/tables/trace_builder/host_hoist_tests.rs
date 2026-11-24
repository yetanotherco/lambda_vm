//! `LAMBDA_VM_HOST_HOIST` builds the tables it builds without it: the page
//! layout computed once from the image's whole pages is the layout computed
//! a byte at a time, page for page, on real runs with and without a private
//! input, and a streamed finish with it on hands on the same header and the
//! same tables, word for word, under the word and the byte memory formats.
//! With the knob off nothing takes the new path.

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
/// way.
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
    (usize, usize),
) {
    let max_rows = MaxRowsConfig::small();
    let builder = configure(
        WindowedTraceBuilder::new(program, input, &max_rows)
            .expect("the builder")
            .host_hoist(hoist),
    );
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
    let calls = super::host_hoist::calls::take();
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

/// ★ A streamed finish with the page layout hoisted hands on the header and
/// the tables it hands on without, word for word, under every configuration
/// and input; it computes the layout once from whole pages, and with the
/// knob off it computes it a byte at a time as before, never from pages.
#[test]
fn the_hoisted_pages_stream_the_same_tables() {
    let on = HostHoist { pages: true };
    for name in PROGRAMS {
        for input in inputs() {
            let (program, logs) = run_on(name, &input);
            for (config, configure) in configs() {
                let what = format!("{name}, {} input bytes, {config}", input.len());
                let (header_off, tables_off, calls_off) =
                    streamed(&program, &logs, &input, &configure, HostHoist::default());
                let (header_on, tables_on, calls_on) =
                    streamed(&program, &logs, &input, &configure, on);
                assert_eq!(header_on, header_off, "{what}: the header");
                assert_eq!(tables_on.len(), tables_off.len(), "{what}: the tables");
                for (k, (a, b)) in tables_on.iter().zip(&tables_off).enumerate() {
                    assert_eq!(a, b, "{what}: table {k}");
                }
                assert!(
                    tables_on.iter().filter(|(_, t)| t.is_some()).count() > 10,
                    "{what}: tables built"
                );
                assert_eq!(calls_on, (0, 1), "{what}: hoisted once, never by byte");
                assert_eq!(calls_off.1, 0, "{what}: off never hoists");
                assert!(calls_off.0 >= 1, "{what}: off computes it by byte");
            }
        }
    }
}

/// With the knob unset no part is on: a builder's default and the stream's
/// default hoist nothing.
#[test]
fn the_knob_off_hoists_nothing() {
    assert_eq!(StreamSkip::default().host_hoist, HostHoist::default());
    assert_eq!(HostHoist::parse(None).expect("unset"), HostHoist::default());
    let program = Elf::load(&asm_elf_bytes("fused_mem_cases")).expect("the ELF loads");
    let builder =
        WindowedTraceBuilder::new(&program, &[], &MaxRowsConfig::small()).expect("the builder");
    assert_eq!(builder.streamed().host_hoist, HostHoist::default());
}
