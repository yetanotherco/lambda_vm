//! The Field VM against the LFM on a real epoch verifier: the min-preset
//! fixture epoch's assembled verifier (`epoch_program(&e, true)`), translated
//! with `field_vm::lfm_translate`.
//!
//! `cargo test --release -p lambda-vm-prover --lib lfm::fvm_epoch_compare -- --ignored --nocapture --test-threads=1`

use std::time::Instant;

use stark::proof::options::{GoldilocksCubicProofOptions, ProofOptions};
use stark::proof::stark::MultiProof;

use crate::field_vm::executor::{Memory, NoHints, execute};
use crate::field_vm::lfm_translate::translate;
use crate::field_vm::prove::{generate_traces, program_id, prove_traces, public_cells, verify};
use crate::tables::types::{GoldilocksExtension, GoldilocksField};

const RUNS: usize = 3;

fn blowup(b: u8) -> ProofOptions {
    GoldilocksCubicProofOptions::with_blowup(b).expect("valid blowup")
}

fn census<PI>(proof: &MultiProof<GoldilocksField, GoldilocksExtension, PI>) -> (usize, usize) {
    proof.proofs.iter().fold((0, 0), |(r, c), p| {
        (
            r + p.trace_length,
            c + p.trace_length * p.trace_ood_evaluations.width,
        )
    })
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

#[test]
#[ignore]
fn epoch_field_vm_vs_lfm() {
    let t = Instant::now();
    let e = super::epoch_tests::real_epoch_from(
        super::proof_fixture::fixture_options(),
        super::epoch_tests::EpochInputs::fixture(),
    );
    let program = super::epoch_tests::epoch_program(&e, true);
    let arenas = super::epoch_tests::epoch_arena_words(&e, true);
    eprintln!(
        "epoch verifier: {} LFM instructions, built in {:.1}s",
        program.instrs.len(),
        t.elapsed().as_secs_f64()
    );
    // Per-chip census: value columns (no preprocessed prefix) plus LogUp aux
    // columns. The field part is what the Field VM translation covers.
    const FIELD: [&str; 6] = [
        "LFM_CONST",
        "LFM_BALU",
        "LFM_XALU",
        "LFM_SELECT",
        "LFM_HINT",
        "LFM_PUBLIC",
    ];
    let (mut field, mut rest) = (0u64, 0u64);
    for c in super::airs::lfm_chip_census(&program) {
        let cells = c.main_cells() + c.aux_cells();
        eprintln!(
            "  chip {:<12} rows={:>8} real={:>8} main={:>4} aux={:>3} cells={:>10}",
            c.name, c.rows, c.real_rows, c.main_cols, c.aux_cols, cells
        );
        if FIELD.contains(&c.name) {
            field += cells;
        } else {
            rest += cells;
        }
    }
    eprintln!("  lfm census: field part {field} cells, hash/bit part {rest} cells");
    let hasher = crate::hash_pin::BLOCK_HASHER;
    let exec =
        super::executor::execute(&program, &arenas, &hasher).expect("the epoch verifier executes");

    for b in [2u8, 4] {
        let opts = blowup(b);
        let s = Instant::now();
        let artifacts = super::registry::build_artifacts_with_hasher(&program, &opts, hasher);
        let build_ms = s.elapsed().as_secs_f64() * 1e3;
        let (mut prove_ms, mut shape) = (Vec::new(), (0, 0));
        for _ in 0..RUNS {
            let s = Instant::now();
            let proof =
                super::proof::lfm_prove(&program, &artifacts, &arenas, &opts).expect("LFM proves");
            prove_ms.push(s.elapsed().as_secs_f64() * 1e3);
            shape = census(&proof.proof);
        }
        eprintln!(
            "  lfm blowup={b} rows={} cells={} build_ms={build_ms:.1} prove_ms={:.1}",
            shape.0,
            shape.1,
            median(prove_ms)
        );
    }

    let opts = blowup(4);
    for bit_dec in [false, true] {
        let tr = translate(&program, &exec, bit_dec);
        eprintln!(
            "  bit_dec={bit_dec} lfm mix={:?} skipped={:?} public={}",
            tr.lfm_counts,
            tr.skipped,
            tr.public.len()
        );
        let run = execute(
            &tr.program,
            Memory::with_init(tr.mem.len(), &tr.mem),
            &mut NoHints,
            1 << 26,
        )
        .expect("the translation executes");
        let s = Instant::now();
        let id = program_id(&tr.program, &opts);
        let build_ms = s.elapsed().as_secs_f64() * 1e3;
        let cells = public_cells(&run, &tr.public);
        let (mut prove_ms, mut verify_ms, mut shape) = (Vec::new(), Vec::new(), (0, 0));
        for _ in 0..RUNS {
            let mut traces = generate_traces(&tr.program, &run, &tr.public);
            let s = Instant::now();
            let proof = prove_traces(&id, &cells, &mut traces, &opts).expect("FVM proves");
            prove_ms.push(s.elapsed().as_secs_f64() * 1e3);
            let s = Instant::now();
            assert!(verify(&id, &cells, &proof, &opts));
            verify_ms.push(s.elapsed().as_secs_f64() * 1e3);
            shape = census(&proof);
        }
        eprintln!(
            "  fvm blowup=4 bit_dec={bit_dec} program={} mem={} rows={} cells={} build_ms={build_ms:.1} prove_ms={:.1} verify_ms={:.1}",
            tr.program.len(),
            tr.mem.len(),
            shape.0,
            shape.1,
            median(prove_ms),
            median(verify_ms)
        );
    }
}
