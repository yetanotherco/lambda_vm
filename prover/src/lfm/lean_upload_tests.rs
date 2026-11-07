//! The socket's lean upload: the `LFM_HASH` trace crosses to the card as each
//! row's preprocessed prefix and sixteen input lanes, and the card rebuilds the
//! round witness (`stark::trace::MainDevProducer::poseidon1_w16_rows`).
//!
//! - On the laptop (the `cuda` feature with no card): the producer is attached
//!   to a `Poseidon1W16` program's socket traces exactly when the knob is on.
//! - On the card (ignored; run on the box with `--ignored --test-threads=1`):
//!   the card's trace equals the host fill word for word, raw `u64`s, padding
//!   rows included; and a program whose socket table crosses the device
//!   threshold proves to the same bytes with the lean upload on and off, the
//!   "on" prove having taken its trace from the producer.

use stark::proof::options::ProofOptions;
use stark::trace::TraceTable;

use crate::tables::types::{FE, GoldilocksExtension, GoldilocksField};

use super::builder::{Cell, LfmBuilder};
use super::compiler::{LfmProgram, compile};
use super::executor::execute;
use super::hash::HasherKind;
use super::p1w16_socket::set_lean_upload;
use super::trace::build_traces_with_hasher;

type F = GoldilocksField;
type E = GoldilocksExtension;

const KIND: HasherKind = HasherKind::Poseidon1W16;

/// Restores the knob to the environment's choice when a test ends, pass or fail.
struct Restore;
impl Drop for Restore {
    fn drop(&mut self) {
        set_lean_upload(None);
    }
}

fn word(seed: u64) -> [FE; 4] {
    core::array::from_fn(|i| FE::from(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (i as u64 + 1)))
}

/// `perms` chained width-16 permutations (each reads the previous output and
/// three constants), the last output published.
fn chained(perms: usize) -> LfmProgram {
    let mut b = LfmBuilder::new();
    let ins: [Cell; 4] = core::array::from_fn(|k| b.digest_const(word(k as u64 + 1)).as_cell());
    let mut out = b.hash16(ins);
    for _ in 1..perms {
        out = b.hash16([out[0], ins[1], out[2], ins[3]]);
    }
    for c in out {
        b.public(c);
    }
    compile(b.finish())
}

fn hash_trace(program: &LfmProgram) -> TraceTable<F, E> {
    let exec = execute(program, &[], &KIND).expect("execute");
    build_traces_with_hasher(program, &exec.records, KIND).hash
}

#[test]
fn the_producer_is_attached_to_the_socket_trace_exactly_when_the_knob_is_on() {
    let _restore = Restore;
    let program = chained(3);
    assert!(program.hash16(), "a Poseidon1W16 program builds the socket");
    set_lean_upload(Some(true));
    assert!(hash_trace(&program).has_main_dev_producer());
    set_lean_upload(Some(false));
    assert!(!hash_trace(&program).has_main_dev_producer());
}

/// The card's trace for a socket table of `2^12` rows, `3000` of them real, is
/// the host fill's, word for word.
#[test]
#[ignore = "card: rebuilds the socket witness on the GPU; run on the box with --ignored"]
fn the_card_rebuilds_the_host_socket_trace_word_for_word() {
    use super::p1w16_socket::{SOCKET_FORM, lean_upload_producer};

    let program = chained(3000);
    let trace = hash_trace(&program);
    let (rows, cols) = (trace.num_rows(), trace.num_cols());
    let real = program.groups.hash.real_rows;
    assert!(
        real < rows,
        "padding rows are part of the check ({real} of {rows})"
    );
    let (data, width) = trace.main_data_row_major();
    assert_eq!(width, cols);
    // SAFETY: `FE` is `#[repr(transparent)]` over `u64`.
    let words: &[u64] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u64, data.len()) };

    let producer = lean_upload_producer(SOCKET_FORM, real);
    let card = (producer.0)(words, rows, cols).expect("the card builds the trace");
    let stream = math_cuda::device::backend().expect("card").next_stream();
    let back = stream.clone_dtoh(&card).expect("download");
    stream.synchronize().expect("sync");
    assert_eq!(back.len(), words.len());
    let first = back.iter().zip(words).position(|(a, b)| a != b);
    assert_eq!(
        first.map(|i| (i / cols, i % cols, back[i], words[i])),
        None,
        "(row, column, card word, host word) of the first difference"
    );
}

/// A program whose socket table crosses the device threshold, proved with the
/// lean upload off and on (grinding off, so the bytes are reproducible): the
/// same proof, verified, and the "on" prove took its socket trace from the
/// producer.
#[test]
#[ignore = "card: proves on the GPU; run on the box with --ignored --test-threads=1"]
fn a_lean_upload_proves_the_same_bytes() {
    use super::proof::{prove_traces_with_hasher, verify_against};
    use super::registry::build_artifacts_with_hasher;
    use stark::proof::options::GoldilocksCubicProofOptions;

    let _restore = Restore;
    let mut opts: ProofOptions = GoldilocksCubicProofOptions::with_blowup(4).expect("options");
    opts.grinding_factor = 0;
    // 2^13 rows at blowup 4 is a 2^15-row LDE, above the device commit floor.
    let program = chained(6000);
    let artifacts = build_artifacts_with_hasher(&program, &opts, KIND);
    let exec = execute(&program, &[], &KIND).expect("execute");

    let prove = |on: bool| {
        set_lean_upload(Some(on));
        let mut traces = build_traces_with_hasher(&program, &exec.records, KIND);
        assert_eq!(traces.hash.has_main_dev_producer(), on);
        let before = stark::trace::main_dev_producer_calls();
        let proof = prove_traces_with_hasher(
            &artifacts,
            &mut traces,
            &exec.public_words,
            &opts,
            KIND,
            stark::residency_mode::ResidencyMode::Retain,
        )
        .expect("prove");
        (proof, stark::trace::main_dev_producer_calls() - before)
    };
    let (off, off_calls) = prove(false);
    let (on, on_calls) = prove(true);
    eprintln!("LEAN UPLOAD producer calls: off {off_calls} · on {on_calls}");
    assert_eq!(off_calls, 0, "the knob off attaches no producer");
    assert!(
        on_calls > 0,
        "the lean prove never took its trace from the producer"
    );
    for proof in [&off, &on] {
        assert!(verify_against(
            &artifacts.roots,
            &artifacts.program_id,
            artifacts.keccak_rnd_chunks,
            proof,
            &exec.public_words,
            &opts,
            KIND,
            artifacts.chip_set,
        ));
    }
    let bytes = |p: &super::proof::LfmStarkProof| {
        rkyv::to_bytes::<rkyv::rancor::Error>(p)
            .expect("rkyv")
            .to_vec()
    };
    assert!(
        bytes(&on) == bytes(&off),
        "the lean upload moved a proof byte"
    );
}

/// The lean upload (round 1: the socket trace built on the card) and the
/// sub-coset composition evaluation (round 2: the constraints walked on `2n`
/// rows) in the same prove: both off against both on, the same proof bytes,
/// and the "on" prove took both paths.
#[test]
#[ignore = "card: proves on the GPU; run on the box with --ignored --test-threads=1"]
fn the_lean_upload_and_the_subcoset_composition_together_prove_the_same_bytes() {
    use super::proof::{prove_traces_with_hasher, verify_against};
    use super::registry::build_artifacts_with_hasher;
    use stark::proof::options::GoldilocksCubicProofOptions;
    use stark::prover::{composition_subcoset_tables, set_composition_subcoset};

    struct RestoreBoth;
    impl Drop for RestoreBoth {
        fn drop(&mut self) {
            set_lean_upload(None);
            set_composition_subcoset(None);
        }
    }
    let _restore = RestoreBoth;
    let mut opts: ProofOptions = GoldilocksCubicProofOptions::with_blowup(4).expect("options");
    opts.grinding_factor = 0;
    // 2^13 rows at blowup 4 is a 2^15-row LDE, above the device commit floor.
    let program = chained(6000);
    let artifacts = build_artifacts_with_hasher(&program, &opts, KIND);
    let exec = execute(&program, &[], &KIND).expect("execute");

    let prove = |on: bool| {
        set_lean_upload(Some(on));
        set_composition_subcoset(Some(on));
        let mut traces = build_traces_with_hasher(&program, &exec.records, KIND);
        assert_eq!(traces.hash.has_main_dev_producer(), on);
        let before = (
            stark::trace::main_dev_producer_calls(),
            composition_subcoset_tables(),
        );
        let proof = prove_traces_with_hasher(
            &artifacts,
            &mut traces,
            &exec.public_words,
            &opts,
            KIND,
            stark::residency_mode::ResidencyMode::Retain,
        )
        .expect("prove");
        let calls = (
            stark::trace::main_dev_producer_calls() - before.0,
            composition_subcoset_tables() - before.1,
        );
        (proof, calls)
    };
    let (off, off_calls) = prove(false);
    let (on, on_calls) = prove(true);
    eprintln!(
        "LEAN UPLOAD + SUBCOSET (producer calls, sub-coset tables): off {off_calls:?} · on {on_calls:?}"
    );
    assert_eq!(off_calls, (0, 0), "both off takes neither path");
    assert!(
        on_calls.0 > 0,
        "the prove never took its trace from the producer"
    );
    assert!(
        on_calls.1 > 0,
        "the prove never evaluated a composition on the sub-coset"
    );
    for proof in [&off, &on] {
        assert!(verify_against(
            &artifacts.roots,
            &artifacts.program_id,
            artifacts.keccak_rnd_chunks,
            proof,
            &exec.public_words,
            &opts,
            KIND,
            artifacts.chip_set,
        ));
    }
    let bytes = |p: &super::proof::LfmStarkProof| {
        rkyv::to_bytes::<rkyv::rancor::Error>(p)
            .expect("rkyv")
            .to_vec()
    };
    assert!(
        bytes(&on) == bytes(&off),
        "the lean upload with the sub-coset composition moved a proof byte"
    );
}
