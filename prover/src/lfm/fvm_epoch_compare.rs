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
    crate::field_vm::prove::production_format(
        GoldilocksCubicProofOptions::with_blowup(b).expect("valid blowup"),
    )
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
        let s = Instant::now();
        let tr = translate(&program, &exec, bit_dec);
        let translate_ms = s.elapsed().as_secs_f64() * 1e3;
        eprintln!(
            "  bit_dec={bit_dec} lfm mix={:?} skipped={:?} public={}",
            tr.lfm_counts,
            tr.skipped,
            tr.public.len()
        );
        let s = Instant::now();
        let run = execute(
            &tr.program,
            Memory::with_init(tr.mem.len(), &tr.mem),
            &mut NoHints,
            1 << 26,
        )
        .expect("the translation executes");
        let execute_ms = s.elapsed().as_secs_f64() * 1e3;
        let s = Instant::now();
        let id = program_id(&tr.program, &opts);
        let build_ms = s.elapsed().as_secs_f64() * 1e3;
        let cells = public_cells(&run, &tr.public);
        let (mut prove_ms, mut verify_ms, mut trace_ms, mut shape, mut bytes) =
            (Vec::new(), Vec::new(), Vec::new(), (0, 0), 0);
        for _ in 0..runs() {
            let s = Instant::now();
            let mut traces = generate_traces(&tr.program, &run, &tr.public);
            trace_ms.push(s.elapsed().as_secs_f64() * 1e3);
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
        eprintln!(
            "  fvm stages bit_dec={bit_dec}: translate_ms={translate_ms:.1} execute_ms={execute_ms:.1} traces_ms={}",
            stats(trace_ms)
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
    reroll_report(&tr.program);
}

/// How much of a straight-line program repeats as a loop body: blocks of the
/// same instruction shapes whose memory offsets move by one stride per
/// repetition (or stay put).
fn reroll_report(program: &crate::field_vm::isa::Program) {
    use crate::field_vm::isa::Instr;
    use std::collections::HashMap;
    let instrs = &program.instrs;
    // The shape: everything but memory offsets.
    let shape = |i: &Instr| {
        let mut h = format!("{}{:?}", i.hint_out as u8, i.hint_in);
        for a in i.args() {
            h.push_str(&format!(
                "|{}:{}:{}",
                a.reg,
                a.scale.canonical(),
                a.mem as u8
            ));
            if !a.mem {
                h.push_str(&format!(":{}", a.offset.canonical()));
            }
        }
        h
    };
    let ids: Vec<u32> = {
        let mut map: HashMap<String, u32> = HashMap::new();
        instrs
            .iter()
            .map(|i| {
                let n = map.len() as u32;
                *map.entry(shape(i)).or_insert(n)
            })
            .collect()
    };
    let offs = |i: &Instr| -> Vec<Option<u64>> {
        i.args()
            .iter()
            .map(|a| a.mem.then(|| a.offset.canonical()))
            .collect()
    };
    // Candidate periods from the distances between repeats of a 32-shape window.
    const K: usize = 32;
    let mut last: HashMap<&[u32], usize> = HashMap::new();
    let mut dist: HashMap<usize, usize> = HashMap::new();
    for i in 0..ids.len().saturating_sub(K) {
        if let Some(&j) = last.get(&ids[i..i + K]) {
            *dist.entry(i - j).or_insert(0) += 1;
        }
        last.insert(&ids[i..i + K], i);
    }
    let mut cands: Vec<(usize, usize)> = dist.into_iter().collect();
    cands.sort_by(|a, b| b.1.cmp(&a.1));
    // For each candidate period: greedy loops of >= 3 repetitions where every
    // memory offset moves by 0 or by one common stride per repetition.
    for &(p, _) in cands.iter().take(6) {
        let (mut covered, mut loops, mut i) = (0usize, 0usize, 0usize);
        while i + 2 * p <= instrs.len() {
            let delta = |k: usize| -> Option<i128> {
                let mut stride: Option<i128> = None;
                for t in 0..p {
                    let (a, b) = (&instrs[i + t], &instrs[i + t + k * p]);
                    if ids[i + t] != ids[i + t + k * p] {
                        return None;
                    }
                    for (x, y) in offs(a).into_iter().zip(offs(b)) {
                        if let (Some(x), Some(y)) = (x, y) {
                            let d = y as i128 - x as i128;
                            if d != 0 {
                                match stride {
                                    None => stride = Some(d),
                                    Some(s) if s == d => {}
                                    _ => return None,
                                }
                            }
                        }
                    }
                }
                Some(stride.unwrap_or(0))
            };
            let Some(s1) = delta(1) else {
                i += 1;
                continue;
            };
            let mut reps = 2;
            while i + (reps + 1) * p <= instrs.len() && delta(reps) == Some(s1 * reps as i128) {
                reps += 1;
            }
            if reps >= 3 {
                covered += reps * p;
                loops += 1;
                i += reps * p;
            } else {
                i += 1;
            }
        }
        eprintln!(
            "reroll period={p}: {loops} loops cover {covered} of {} instructions ({:.1}%)",
            instrs.len(),
            100.0 * covered as f64 / instrs.len() as f64
        );
    }
}

/// The binary half of the epoch verifier, as the LFM runs it: hash rows by
/// mode, the keccak / BLAKE3 accelerator calls and the byte and bit glue.
/// What a 3MI RISC-V half would do instead of the Field VM.
#[test]
#[ignore]
fn epoch_binary_work() {
    use super::instr::{HashMode, Instr};
    use std::collections::BTreeMap;
    let e = super::epoch_tests::real_epoch_from(
        inner_options(),
        super::epoch_tests::EpochInputs::fixture(),
    );
    let program = super::epoch_tests::epoch_program(&e, true);
    let mut counts: BTreeMap<String, u64> = BTreeMap::new();
    let mut bump = |k: &str, n: u64| *counts.entry(k.to_string()).or_insert(0) += n;
    for i in &program.instrs {
        match i {
            Instr::Hash { mode, .. } => bump(
                match mode {
                    HashMode::Compress => "hash.compress",
                    HashMode::Transcript => "hash.transcript",
                    HashMode::Leaf => "hash.leaf",
                    HashMode::Permute => "hash.permute",
                },
                1,
            ),
            Instr::KeccakF(_) => bump("keccak_f", 1),
            Instr::Blake3(_) => bump("blake3", 1),
            Instr::Pack { .. } => bump("pack", 1),
            Instr::Unpack { .. } => bump("unpack", 1),
            Instr::BitDec { bits, halves, .. } => {
                bump("bit_dec", 1);
                bump("bit_dec.bits_used", bits.len() as u64);
                bump("bit_dec.with_halves", halves.is_some() as u64);
            }
            Instr::Hint { .. } => bump("hint", 1),
            Instr::Public { .. } => bump("public", 1),
            _ => bump("field", 1),
        }
    }
    eprintln!("epoch verifier: {} LFM instructions", program.instrs.len());
    for (k, n) in &counts {
        eprintln!("  {k}: {n}");
    }
    eprintln!("  LFM hasher: {:?}", crate::hash_pin::BLOCK_HASHER);
}

/// The RISC-V cost of a Merkle compression, h = keccak256(h ‖ h), under the
/// production base options (blowup 4, the pipeline's format, RPX commitments):
/// prove time and committed cells at several N, whose slope is the per-hash
/// cost a 3MI RISC-V half pays. Needs `executor/program_artifacts/rust/keccak_chain.elf`.
#[test]
#[ignore]
fn riscv_keccak_compress_cost() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let elf = std::fs::read(root.join("../executor/program_artifacts/rust/keccak_chain.elf"))
        .expect("build keccak_chain.elf first");
    let opts = super::proof::block_base_options();
    for n in [1024u32, 4096, 16384, 65536] {
        let mut ms = Vec::new();
        let mut cells = 0;
        for _ in 0..runs() {
            let t = Instant::now();
            let proof = crate::prove_with_options_and_inputs(
                &elf,
                &n.to_le_bytes(),
                &opts,
                &crate::tables::MaxRowsConfig::default(),
            )
            .expect("keccak_chain proves");
            ms.push(t.elapsed().as_secs_f64() * 1e3);
            cells = census(&proof.proof).1;
        }
        eprintln!(
            "riscv keccak_chain n={n} cells={cells} prove_ms={}",
            stats(ms)
        );
    }
}

/// The Field VM translation of `program`'s field half and the hash half it
/// hands to the LFM chips.
struct Hybrid {
    side: crate::field_vm::hash_side::HashSide,
    tr: crate::field_vm::lfm_translate::Translation,
    rows: Vec<crate::field_vm::bridge::BridgeRow>,
    run: crate::field_vm::executor::Execution,
    pid: crate::field_vm::prove::ProgramId,
    id: crate::field_vm::hash_side::HashSideId,
    cells: Vec<crate::field_vm::prove::PublicCell>,
    reads: Vec<u64>,
}

impl Hybrid {
    fn new(
        program: &super::compiler::LfmProgram,
        exec: &super::executor::LfmExecution,
        arenas: &[Vec<super::word::LfmWord>],
        opts: &ProofOptions,
    ) -> Self {
        use crate::field_vm::hash_side;
        let s = Instant::now();
        let mut side = hash_side::split(program, exec, arenas);
        let crossing = side.crossing();
        let tr = crate::field_vm::lfm_translate::translate_keeping(
            program,
            exec,
            false,
            &crossing,
            &hash_side::is_hash_side,
        );
        let faddr: std::collections::HashMap<u64, u64> =
            crossing.iter().copied().zip(tr.kept.iter().copied()).collect();
        let rows = if std::env::var_os("FVM_HYBRID_MERGE").is_some() {
            hash_side::renumber(&mut side, &|a| faddr[&a], tr.mem.len() as u64);
            hash_side::bridge_rows(&side, |a| a)
        } else {
            hash_side::bridge_rows(&side, |a| faddr[&a])
        };
        eprintln!(
            "  split: hash half {} instrs, out {} (words {}), back {}, fvm program {} mem {} split_ms={:.1}",
            side.sub.instrs.len() - side.out.len(),
            side.out.len(),
            side.out.iter().filter(|c| c.is_word).count(),
            side.back.len(),
            tr.program.len(),
            tr.mem.len(),
            s.elapsed().as_secs_f64() * 1e3
        );
        let run = execute(
            &tr.program,
            Memory::with_init(tr.mem.len(), &tr.mem),
            &mut NoHints,
            1 << 26,
        )
        .expect("the translation executes");
        let s = Instant::now();
        let pid = program_id(&tr.program, opts);
        let artifacts =
            super::registry::build_artifacts_with_hasher(&side.sub, opts, crate::hash_pin::BLOCK_HASHER);
        let id = hash_side::side_id(&side, &rows, &artifacts, opts);
        eprintln!("  hybrid build_ms={:.1}", s.elapsed().as_secs_f64() * 1e3);
        let cells = public_cells(&run, &tr.public);
        let reads = if side.merged {
            Vec::new()
        } else {
            rows.iter().map(|r| r.faddr).collect()
        };
        Hybrid {
            side,
            tr,
            rows,
            run,
            pid,
            id,
            cells,
            reads,
        }
    }

    fn traces(&self) -> (crate::field_vm::prove::FvmTraces, Vec<stark::trace::TraceTable<GoldilocksField, GoldilocksExtension>>) {
        let (extra, cells) =
            crate::field_vm::hash_side::traces(&self.side, &self.rows, &self.id, &self.run.mem);
        (
            crate::field_vm::prove::generate_traces_lfm(
                &self.tr.program,
                &self.run,
                &self.tr.public,
                None,
                &self.reads,
                self.side.merged.then_some(&cells[..]),
            ),
            extra,
        )
    }

    /// Proves the traces; `None` when proving or verification fails.
    fn prove(
        &self,
        traces: &mut crate::field_vm::prove::FvmTraces,
        extra: &mut [stark::trace::TraceTable<GoldilocksField, GoldilocksExtension>],
        opts: &ProofOptions,
    ) -> Option<(crate::field_vm::prove::FvmProof, f64, f64)> {
        use crate::field_vm::hash_side::airs;
        use crate::field_vm::prove::{prove_traces_with, verify_with};
        let s = Instant::now();
        let proof =
            prove_traces_with(
                &self.pid,
                &self.cells,
                traces,
                airs(&self.id, opts),
                extra,
                self.side.merged,
                opts,
            )
            .ok()?;
        let prove_ms = s.elapsed().as_secs_f64() * 1e3;
        let s = Instant::now();
        let ok = verify_with(
            &self.pid,
            &self.cells,
            &proof,
            airs(&self.id, opts),
            self.side.merged,
            opts,
        );
        let verify_ms = s.elapsed().as_secs_f64() * 1e3;
        ok.then_some((proof, prove_ms, verify_ms))
    }
}

/// Option 3: the Field VM with the LFM's hash-side chips in the same proof,
/// against the LFM at the same blowup.
#[test]
#[ignore]
fn epoch_field_vm_hash_chip() {
    let _ = env_logger::builder().is_test(true).try_init();

    let e = super::epoch_tests::real_epoch_from(
        inner_options(),
        super::epoch_tests::EpochInputs::fixture(),
    );
    let program = super::epoch_tests::epoch_program(&e, true);
    let arenas = super::epoch_tests::epoch_arena_words(&e, true);
    let hasher = crate::hash_pin::BLOCK_HASHER;
    let exec =
        super::executor::execute(&program, &arenas, &hasher).expect("the epoch verifier executes");
    let opts = crate::field_vm::prove::default_options();

    let hy = Hybrid::new(&program, &exec, &arenas, &opts);

    let (mut prove_ms, mut verify_ms, mut trace_ms, mut shape, mut bytes) =
        (Vec::new(), Vec::new(), Vec::new(), (0, 0), 0);
    let mut tables = String::new();
    for _ in 0..runs() {
        let s = Instant::now();
        let (mut traces, mut extra) = hy.traces();
        trace_ms.push(s.elapsed().as_secs_f64() * 1e3);
        let (proof, p, v) = hy
            .prove(&mut traces, &mut extra, &opts)
            .expect("FVM + hash chips prove and verify");
        prove_ms.push(p);
        verify_ms.push(v);
        shape = census(&proof);
        bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&proof)
            .map(|b| b.len())
            .unwrap_or(0);
        tables = proof
            .proofs
            .iter()
            .map(|p| format!("{}x{}", p.trace_length, p.trace_ood_evaluations.width))
            .collect::<Vec<_>>()
            .join(" ");
    }
    eprintln!("  hybrid tables: {tables}");
    eprintln!(
        "  hybrid blowup={} rows={} cells={} prove_ms={} verify_ms={} proof_bytes={bytes} traces_ms={}",
        opts.blowup_factor,
        shape.0,
        shape.1,
        stats(prove_ms),
        stats(verify_ms),
        stats(trace_ms)
    );

    if std::env::var_os("FVM_COMPARE_SKIP_LFM").is_none() {
        let artifacts = super::registry::build_artifacts_with_hasher(&program, &opts, hasher);
        let (mut prove_ms, mut verify_ms, mut shape, mut bytes) =
            (Vec::new(), Vec::new(), (0, 0), 0);
        for _ in 0..runs() {
            let s = Instant::now();
            let proof =
                super::proof::lfm_prove(&program, &artifacts, &arenas, &opts).expect("LFM proves");
            prove_ms.push(s.elapsed().as_secs_f64() * 1e3);
            let s = Instant::now();
            assert!(super::proof::verify_against_artifacts(
                &artifacts,
                &proof.proof,
                &proof.public_words,
                &opts
            ));
            verify_ms.push(s.elapsed().as_secs_f64() * 1e3);
            shape = census(&proof.proof);
            bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&proof.proof)
                .map(|b| b.len())
                .unwrap_or(0);
            tables = proof
                .proof
                .proofs
                .iter()
                .map(|p| format!("{}x{}", p.trace_length, p.trace_ood_evaluations.width))
                .collect::<Vec<_>>()
                .join(" ");
        }
        eprintln!("  lfm tables: {tables}");
        eprintln!(
            "  lfm blowup={} rows={} cells={} prove_ms={} verify_ms={} proof_bytes={bytes}",
            opts.blowup_factor,
            shape.0,
            shape.1,
            stats(prove_ms),
            stats(verify_ms)
        );
    }
}

/// A cheating prover moving a cell across the bridge: a hash output the field
/// half reads, given another value in MEM and on the bridge, and a field cell
/// handed to the hash half with a fourth lane its writer never set.
#[test]
#[ignore]
fn epoch_hash_chip_rejects_tampering() {
    let e = super::epoch_tests::real_epoch_from(
        super::proof_fixture::fixture_options(),
        super::epoch_tests::EpochInputs::fixture(),
    );
    let program = super::epoch_tests::epoch_program(&e, true);
    let arenas = super::epoch_tests::epoch_arena_words(&e, true);
    let exec = super::executor::execute(&program, &arenas, &crate::hash_pin::BLOCK_HASHER)
        .expect("the epoch verifier executes");
    let opts = crate::field_vm::prove::default_options();
    let hy = Hybrid::new(&program, &exec, &arenas, &opts);
    use crate::field_vm::{bridge, mem};
    use crate::tables::types::FE;

    let (mut traces, mut extra) = hy.traces();
    assert!(hy.prove(&mut traces, &mut extra, &opts).is_some(), "honest");

    let back = hy.rows.iter().position(|r| r.is_in).expect("an incoming cell");
    let faddr = hy.rows[back].faddr as usize;
    let (mut traces, mut extra) = hy.traces();
    let bump = |t: &mut stark::trace::TraceTable<GoldilocksField, GoldilocksExtension>, r, c| {
        let v = *t.main_table.get(r, c) + FE::one();
        t.main_table.set(r, c, v);
    };
    bump(&mut traces.mem, faddr, mem::cols::VALUE);
    if !hy.side.merged {
        bump(&mut extra[0], back, bridge::cols::V0);
    }
    assert!(hy.prove(&mut traces, &mut extra, &opts).is_none(), "hash output moved");

    if hy.side.merged {
        // MEM relaying the cell with a zero fourth lane.
        let (mut traces, mut extra) = hy.traces();
        bump(&mut traces.mem, faddr, mem::cols::M_IN);
        bump(&mut traces.mem, faddr, mem::cols::M_OUT);
        assert!(hy.prove(&mut traces, &mut extra, &opts).is_none(), "relayed");
        return;
    }
    let out = hy
        .rows
        .iter()
        .position(|r| !r.is_in && !r.is_word)
        .expect("an outgoing field cell");
    let (mut traces, mut extra) = hy.traces();
    bump(&mut extra[0], out, bridge::cols::V0 + 3);
    assert!(hy.prove(&mut traces, &mut extra, &opts).is_none(), "fourth lane set");
}
