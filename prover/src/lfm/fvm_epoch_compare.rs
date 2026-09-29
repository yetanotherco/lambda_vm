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

/// Repetitions per configuration: `FVM_COMPARE_RUNS`, default 3.
fn runs() -> usize {
    std::env::var("FVM_COMPARE_RUNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3)
}

fn blowup(b: u8) -> ProofOptions {
    GoldilocksCubicProofOptions::with_blowup(b).expect("valid blowup")
}

/// The inner epoch's options: `FVM_EPOCH_PRESET` = `blowup2` | `blowup4` |
/// `blowup8` picks a production preset; unset is the min-preset fixture.
fn inner_options() -> ProofOptions {
    use crate::recursion::Preset;
    match std::env::var("FVM_EPOCH_PRESET").as_deref() {
        Ok("blowup2") => Preset::Blowup2.options(),
        Ok("blowup4") => Preset::Blowup4.options(),
        Ok("blowup8") => Preset::Blowup8.options(),
        _ => super::proof_fixture::fixture_options(),
    }
}

fn census<PI>(proof: &MultiProof<GoldilocksField, GoldilocksExtension, PI>) -> (usize, usize) {
    proof.proofs.iter().fold((0, 0), |(r, c), p| {
        (
            r + p.trace_length,
            c + p.trace_length * p.trace_ood_evaluations.width,
        )
    })
}

/// `median (cv=…%, n=…)` of timings in milliseconds.
fn stats(mut v: Vec<f64>) -> String {
    v.sort_by(f64::total_cmp);
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let sd = (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n).sqrt();
    format!(
        "{:.1} (cv={:.1}%, n={})",
        v[v.len() / 2],
        100.0 * sd / mean,
        v.len()
    )
}

#[test]
#[ignore]
fn epoch_field_vm_vs_lfm() {
    let t = Instant::now();
    let e = super::epoch_tests::real_epoch_from(
        inner_options(),
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

    // `FVM_COMPARE_SKIP_LFM` proves only the Field VM side.
    let lfm_blowups: &[u8] = if std::env::var_os("FVM_COMPARE_SKIP_LFM").is_some() {
        &[]
    } else {
        &[2, 4]
    };
    for &b in lfm_blowups {
        let opts = blowup(b);
        let s = Instant::now();
        let artifacts = super::registry::build_artifacts_with_hasher(&program, &opts, hasher);
        let build_ms = s.elapsed().as_secs_f64() * 1e3;
        let (mut prove_ms, mut shape) = (Vec::new(), (0, 0));
        for _ in 0..runs() {
            let s = Instant::now();
            let proof =
                super::proof::lfm_prove(&program, &artifacts, &arenas, &opts).expect("LFM proves");
            prove_ms.push(s.elapsed().as_secs_f64() * 1e3);
            shape = census(&proof.proof);
        }
        eprintln!(
            "  lfm blowup={b} rows={} cells={} build_ms={build_ms:.1} prove_ms={}",
            shape.0,
            shape.1,
            stats(prove_ms)
        );
    }

    let opts = crate::field_vm::prove::default_options();
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
        let (mut prove_ms, mut verify_ms, mut shape, mut bytes) =
            (Vec::new(), Vec::new(), (0, 0), 0);
        for _ in 0..runs() {
            let mut traces = generate_traces(&tr.program, &run, &tr.public);
            let s = Instant::now();
            let proof = prove_traces(&id, &cells, &mut traces, &opts).expect("FVM proves");
            prove_ms.push(s.elapsed().as_secs_f64() * 1e3);
            let s = Instant::now();
            assert!(verify(&id, &cells, &proof, &opts));
            verify_ms.push(s.elapsed().as_secs_f64() * 1e3);
            shape = census(&proof);
            bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&proof)
                .map(|b| b.len())
                .unwrap_or(0);
        }
        eprintln!(
            "  fvm blowup={} bit_dec={bit_dec} program={} mem={} rows={} cells={} build_ms={build_ms:.1} prove_ms={} verify_ms={} proof_bytes={bytes}",
            opts.blowup_factor,
            tr.program.len(),
            tr.mem.len(),
            shape.0,
            shape.1,
            stats(prove_ms),
            stats(verify_ms)
        );
    }
}

#[test]
#[ignore]
fn epoch_op_profile() {
    use super::instr::{BaseOp, ExtOp, Instr};
    use std::collections::{BTreeMap, HashMap};
    let e = super::epoch_tests::real_epoch_from(
        inner_options(),
        super::epoch_tests::EpochInputs::fixture(),
    );
    let program = super::epoch_tests::epoch_program(&e, true);
    let op = |i: &Instr| -> String {
        match i {
            Instr::BaseAlu { op, .. } => format!("b{op:?}"),
            Instr::ExtAlu { op, .. } => format!("e{op:?}"),
            other => {
                format!("{:?}", std::mem::discriminant(other))
                    .chars()
                    .take(0)
                    .collect::<String>()
                    + match other {
                        Instr::Const { .. } => "Const",
                        Instr::Select { .. } => "Select",
                        Instr::BitDec { .. } => "BitDec",
                        Instr::Hash { .. } => "Hash",
                        Instr::Hint { .. } => "Hint",
                        Instr::Pack { .. } => "Pack",
                        Instr::Unpack { .. } => "Unpack",
                        Instr::Public { .. } => "Public",
                        _ => "Other",
                    }
            }
        }
    };
    let mut kinds: BTreeMap<String, usize> = BTreeMap::new();
    let mut producer: HashMap<u64, (usize, u64)> = HashMap::new();
    for (idx, i) in program.instrs.iter().enumerate() {
        *kinds.entry(op(i)).or_insert(0) += 1;
        match i {
            Instr::BaseAlu { out, mult, .. }
            | Instr::ExtAlu { out, mult, .. }
            | Instr::Const { out, mult, .. } => {
                producer.insert(out.0, (idx, *mult));
            }
            _ => {}
        }
    }
    let mut pairs: BTreeMap<String, usize> = BTreeMap::new();
    for i in &program.instrs {
        let (kind, ins): (String, Vec<u64>) = match i {
            Instr::BaseAlu { a, b, c, op, .. } => (
                format!("b{op:?}"),
                if matches!(op, BaseOp::MulAdd) {
                    vec![a.0, b.0, c.0]
                } else {
                    vec![a.0, b.0]
                },
            ),
            Instr::ExtAlu { a, b, c, op, .. } => (
                format!("e{op:?}"),
                if matches!(op, ExtOp::MulAdd) {
                    vec![a.0, b.0, c.0]
                } else {
                    vec![a.0, b.0]
                },
            ),
            _ => continue,
        };
        for (pos, a) in ins.iter().enumerate() {
            if let Some(&(p, 1)) = producer.get(a) {
                *pairs
                    .entry(format!("{} -> {kind}[{pos}]", op(&program.instrs[p])))
                    .or_insert(0) += 1;
            }
        }
    }
    eprintln!("kinds: {kinds:?}");
    let mut v: Vec<_> = pairs.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1));
    for (k, n) in v.iter().take(25) {
        eprintln!("  single-use {k}: {n}");
    }
}

/// The translation with `FVM_TRANSLATE_OPT` / `FVM_MEM_COMPACT` executes on the
/// real epoch: every fused or chained row is an assertion the executor checks.
#[test]
#[ignore]
fn epoch_translation_shapes() {
    let e = super::epoch_tests::real_epoch_from(
        inner_options(),
        super::epoch_tests::EpochInputs::fixture(),
    );
    let program = super::epoch_tests::epoch_program(&e, true);
    let arenas = super::epoch_tests::epoch_arena_words(&e, true);
    let exec = super::executor::execute(&program, &arenas, &crate::hash_pin::BLOCK_HASHER).unwrap();
    let tr = translate(&program, &exec, false);
    let run = execute(
        &tr.program,
        Memory::with_init(tr.mem.len(), &tr.mem),
        &mut NoHints,
        1 << 26,
    )
    .expect("the translation executes");
    eprintln!(
        "program={} steps={} mem={} public={} skipped={:?}",
        tr.program.len(),
        run.steps.len(),
        tr.mem.len(),
        tr.public.len(),
        tr.skipped
    );
    // Row classes for a narrow-table split: no registers and no hints (P),
    // and whether every argument value is in the base field.
    use crate::field_vm::isa::REG_ZERO;
    let (mut p_base, mut p_ext, mut other) = (0usize, 0usize, 0usize);
    for step in &run.steps {
        let instr = &tr.program.instrs[step.state.pc as usize];
        let plain = !instr.hint_out
            && instr.hint_in.iter().all(|h| !h)
            && instr
                .args()
                .iter()
                .all(|a| a.reg == REG_ZERO && a.scale == crate::tables::types::FE::zero());
        let base = step.args.iter().all(|v| {
            let c = v.value();
            c[1] == crate::tables::types::FE::zero() && c[2] == crate::tables::types::FE::zero()
        });
        match (plain, base) {
            (true, true) => p_base += 1,
            (true, false) => p_ext += 1,
            _ => other += 1,
        }
    }
    eprintln!("row classes: plain+base={p_base} plain+ext={p_ext} registers/hints={other}");
    // How many arguments of each row read memory, and which positions.
    let mut by_count = [0usize; 5];
    let mut by_pos = [0usize; 4];
    for step in &run.steps {
        let instr = &tr.program.instrs[step.state.pc as usize];
        let args = instr.args();
        by_count[args.iter().filter(|a| a.mem).count()] += 1;
        for (i, a) in args.iter().enumerate() {
            by_pos[i] += a.mem as usize;
        }
    }
    eprintln!("memory args per row: {by_count:?}; per position d,a,b,c: {by_pos:?}");
}
