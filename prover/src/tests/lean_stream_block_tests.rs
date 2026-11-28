//! ★ `LAMBDA_VM_EXEC_STREAM=lean` on real block proves: a block whose
//! executor wrote only its access records, ECALLs and window starting cells,
//! each window re-run by its walker, is the block proved from the executor's
//! logs, under the word memory format: on a CPU run, on runs that call COPY,
//! the commit, Keccak-256 and ECREC, and on one with a private input
//! across two pages; with live regeneration rebuilding every streamed chunk
//! (`RegenMode::Always`, whose digests cover the rebuilt logs); the same
//! partition, heights, counts, output and group roots, both verify, and
//! under the deterministic grind (`LAMBDA_VM_DETERMINISTIC_GRIND`, read once
//! a process: run alone) the same bytes. The proving tests run where a block
//! can be proved (the box's lib run and CI).

use super::word_memory_block_tests::two_pages_of_input;
use crate::block_whir::regen::RegenMode;
use crate::block_whir::{
    BlockFormat, BlockOptions, BlockWhirProof, Deviations, prove_block_whir_with, verify_block_whir,
};
use crate::tables::MaxRowsConfig;
use crate::tables::fused_memory::MemoryFormat;
use crate::tables::trace_builder::host_hoist::calls;
use crate::tables::trace_builder::{FinishEarly, HostHoist, WalkCuts};
use crate::test_utils::asm_elf_bytes;
use stark::proof::options::ProofOptions;

fn bytes(proof: &BlockWhirProof) -> Vec<u8> {
    rkyv::to_bytes::<rkyv::rancor::Error>(proof)
        .expect("serialize")
        .to_vec()
}

/// Whether the process runs record mode (the lean stream's ground): unless
/// the walkers or the history are turned off.
fn record_mode() -> bool {
    std::env::var("LAMBDA_VM_BLOCK_WALKERS").map_or(true, |v| v.trim() != "0")
        && std::env::var("LAMBDA_VM_EXEC_HISTORY").map_or(true, |v| v.trim() != "0")
}

fn words() -> BlockFormat {
    BlockFormat::production()
        .with_memory(MemoryFormat::Words)
        .with_page_run(0)
}

fn proved(
    elf: &[u8],
    input: &[u8],
    options: &BlockOptions,
    deviations: &Deviations,
    what: &str,
) -> BlockWhirProof {
    let opts = ProofOptions::default_test_options();
    let format = words();
    let (proof, stamps) = prove_block_whir_with(elf, input, &opts, &format, options, deviations)
        .unwrap_or_else(|e| panic!("{what}: {e:?}"));
    assert_eq!(
        Some(stamps.executor.lean),
        deviations.exec_stream_lean,
        "{what}: executor stream"
    );
    assert!(
        stamps.executor.windows >= 2,
        "{what}: the executor hands out multiple windows"
    );
    if stamps.executor.lean {
        assert_eq!(
            stamps.executor.log_bytes, 0,
            "{what}: the executor writes no logs"
        );
    }
    if options.regen == Some(RegenMode::Always) {
        let regen = stamps.regen.expect("live regeneration's readout");
        assert!(
            regen.dropped > regen.rest_dropped && regen.regenerated >= regen.dropped,
            "{what}: no streamed chunk dropped and rebuilt: {:?}",
            regen.lines
        );
    }
    assert!(
        matches!(verify_block_whir(&proof, elf, &opts, &format), Ok(true)),
        "{what}: verifies"
    );
    proof
}

fn assert_same_block(on: &BlockWhirProof, off: &BlockWhirProof, what: &str) {
    assert_eq!(on.groups, off.groups, "{what}: the partition");
    assert_eq!(on.table_num_vars, off.table_num_vars, "{what}: the heights");
    assert_eq!(
        format!("{:?}", on.table_counts),
        format!("{:?}", off.table_counts),
        "{what}: the counts"
    );
    assert_eq!(on.word_memory, off.word_memory, "{what}: the word tables");
    assert_eq!(on.public_output, off.public_output, "{what}: the output");
    assert_eq!(on.proof.roots, off.proof.roots, "{what}: the group roots");
    if crypto::grinding::deterministic() {
        assert_eq!(bytes(on), bytes(off), "{what}: the proof's bytes");
    }
}

fn assert_guest_tables(name: &str, proof: &BlockWhirProof) {
    match name {
        "test_keccak256_stream" => assert!(proof.table_counts.keccak_rnd > 1),
        "test_ecrec_stream" => {
            assert!(proof.table_counts.ecrec > 0 && proof.table_counts.ecdas > 0)
        }
        "page_runs" => assert_eq!(proof.num_private_input_pages, 2),
        _ => {}
    }
}

fn stream_options() -> BlockOptions {
    // The small syscall guests still exercise several windows and hand out
    // droppable CPU chunks before the finish.
    let max_rows = MaxRowsConfig {
        cpu: 8,
        ..MaxRowsConfig::small()
    };
    super::block_whir_tests::streamed(max_rows, 3, 3)
}

fn assert_lean_stream(name: &str, input: &[u8], regen: Option<RegenMode>) {
    let what = format!("{name}, regeneration {regen:?}");
    eprintln!("LEAN STREAM WANTS: {what}");
    if !record_mode() {
        eprintln!("the lean stream needs record mode: skipped (walkers or history off)");
        return;
    }
    // Keep the ECREC forgery switch off for the ECREC case.
    let _ecrec = (name == "test_ecrec_stream").then(crate::tables::trace_builder::ecrec_forge_free);
    let elf = asm_elf_bytes(name);
    let mut options = stream_options();
    options.drop_streamed_ops = regen.is_some();
    options.regen = regen;
    let prove = |lean| {
        proved(
            &elf,
            input,
            &options,
            &Deviations {
                exec_stream_lean: Some(lean),
                ..Deviations::default()
            },
            &format!("{what}, lean {lean}"),
        )
    };
    let off = prove(false);
    let on = prove(true);
    assert_guest_tables(name, &on);
    assert_same_block(&on, &off, &what);
    if crypto::grinding::deterministic() {
        eprintln!("LEAN STREAM SAME BYTES: {what}");
    }
}

mod the_lean_stream_proves_the_same_block {
    use super::*;

    macro_rules! guest {
        ($name:ident, $input:expr $(, $without_regeneration:ident)?) => {
            mod $name {
                use super::*;

                $(
                    #[test]
                    fn $without_regeneration() {
                        assert_lean_stream(stringify!($name), &$input, None);
                    }
                )?

                #[test]
                fn regenerate_every_chunk() {
                    assert_lean_stream(stringify!($name), &$input, Some(RegenMode::Always));
                }
            }
        };
    }

    // The stacked cases also compare full and lean with regeneration off.
    guest!(test_copy_ecalls, [], without_regeneration);
    guest!(test_commit_split, [], without_regeneration);
    guest!(fused_mem_cases, []);
    guest!(test_keccak256_stream, []);
    guest!(test_ecrec_stream, []);
    guest!(page_runs, two_pages_of_input());
}

/// Compare the three combinations of lean execution and the stacked host
/// levers with one control for the same walker schedule.
fn assert_lean_stack(name: &str, input: &[u8], walkers: usize) {
    let what = format!("{name}, {walkers} walkers");
    eprintln!("LEAN STACK WANTS: {what}");
    if !record_mode() {
        eprintln!("the lean stream needs record mode: skipped (walkers or history off)");
        return;
    }
    let _ecrec = (name == "test_ecrec_stream").then(crate::tables::trace_builder::ecrec_forge_free);
    let _turn = calls::turn();
    let _ = calls::take_states();
    let elf = asm_elf_bytes(name);
    let mut options = stream_options();
    options.regen = Some(RegenMode::Off);
    let prove = |levers, lean| {
        let (hoist, cuts, early) = if levers {
            (HostHoist::all(), WalkCuts::all(), FinishEarly::all())
        } else {
            (HostHoist::none(), WalkCuts::none(), FinishEarly::none())
        };
        let arm = format!("{what}, levers {levers}, lean {lean}");
        let proof = proved(
            &elf,
            input,
            &options,
            &Deviations {
                walkers: Some(walkers),
                host_hoist: Some(hoist.counted()),
                walk_cuts: Some(cuts),
                finish_early: Some(early),
                exec_stream_lean: Some(lean),
                ..Deviations::default()
            },
            &arm,
        );
        assert_eq!(
            calls::take_states(),
            usize::from(levers),
            "{arm}: the hoisted state"
        );
        proof
    };
    let off = prove(false, false);
    assert_guest_tables(name, &off);
    let lean_only = prove(false, true);
    assert_same_block(&lean_only, &off, &format!("{what}, lean only"));
    drop(lean_only);
    let full = prove(true, false);
    assert_same_block(&full, &off, &format!("{what}, host levers only"));
    let stacked = prove(true, true);
    assert_same_block(&stacked, &off, &format!("{what}, lean and host levers"));
    // The non-regenerating full/lean comparison shares the two proofs above.
    assert_same_block(
        &stacked,
        &full,
        &format!("{what}, lean with the host levers on"),
    );
    if crypto::grinding::deterministic() {
        eprintln!("LEAN STACK SAME BYTES: {what}");
    }
}

mod the_lean_stream_with_the_other_host_levers_proves_the_same_block {
    use super::*;

    macro_rules! guest {
        ($name:ident, $input:expr) => {
            mod $name {
                use super::*;

                #[test]
                fn one_walker() {
                    assert_lean_stack(stringify!($name), &$input, 1);
                }

                #[test]
                fn three_walkers() {
                    assert_lean_stack(stringify!($name), &$input, 3);
                }
            }
        };
    }

    guest!(fused_mem_cases, []);
    guest!(test_keccak256_stream, []);
    guest!(test_ecrec_stream, []);
    guest!(page_runs, two_pages_of_input());
}

/// The lean stream is refused, never ignored, where it does not run: the
/// whole-run build (no windows), and the split memory format.
#[test]
fn the_lean_stream_is_refused_where_it_does_not_run() {
    let elf = asm_elf_bytes("fused_mem_cases");
    let opts = ProofOptions::default_test_options();
    let lean = Deviations {
        exec_stream_lean: Some(true),
        ..Deviations::default()
    };
    let words = BlockFormat::production()
        .with_memory(MemoryFormat::Words)
        .with_page_run(0);
    let whole = super::block_whir_tests::options(MaxRowsConfig::small(), 3);
    let err = prove_block_whir_with(&elf, &[], &opts, &words, &whole, &lean).map(|_| ());
    assert!(
        matches!(&err, Err(crate::Error::Prover(m)) if m.contains("windowed build")),
        "{err:?}"
    );
    let split = BlockFormat::production()
        .with_memory(MemoryFormat::Split)
        .with_page_run(0);
    let streamed = super::block_whir_tests::streamed(MaxRowsConfig::small(), 3, 9);
    let err = prove_block_whir_with(&elf, &[], &opts, &split, &streamed, &lean).map(|_| ());
    assert!(
        matches!(&err, Err(crate::Error::Prover(m)) if m.contains("LAMBDA_VM_EXEC_STREAM=lean needs")),
        "{err:?}"
    );
}
