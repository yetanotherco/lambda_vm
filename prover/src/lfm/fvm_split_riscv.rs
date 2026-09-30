//! The 3MI split verification of the min-preset fixture epoch, end to end:
//! the field half proved on the Field VM, the hash half proved on the RISC-V
//! VM by the `split_verify_b` guest (as a continuation), and the coupling of
//! the two proofs over the shared input and the communication record
//! (`field_vm::split`). Two proofs and a host-side coupling stand in for the
//! spec's single joint proof. `FVM_SPLIT_RISCV_BITS` leaves the query bits
//! to the RISC-V half, in the record, instead of the Field VM decomposing
//! the query indices itself.
//!
//! Needs `executor/program_artifacts/rust/split_verify_b.elf`.
//! `cargo test --release -p lambda-vm-prover --lib lfm::fvm_split_riscv -- --ignored --nocapture --test-threads=1`

use std::time::Instant;

use super::fvm_epoch_compare::{census, runs, stats};
use crate::field_vm::executor::{Execution, Memory, NoHints, execute};
use crate::field_vm::prove::{
    FvmProof, ProgramId, PublicCell, default_options, generate_traces, program_id, prove_traces,
    public_cells, verify,
};
use crate::field_vm::split::{self, Cut};

fn guest_elf() -> Vec<u8> {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    std::fs::read(root.join("../executor/program_artifacts/rust/split_verify_b.elf"))
        .expect("build split_verify_b.elf first")
}

/// Both halves of an epoch's verifier.
struct Split {
    cut: Cut,
    /// The Field VM address of each crossing cell, in [`Cut::crossing`] order.
    faddr: Vec<u64>,
    fvm_program: crate::field_vm::isa::Program,
    run: Execution,
    public: Vec<u64>,
    pid: ProgramId,
    cells: Vec<PublicCell>,
}

impl Split {
    fn new(inner: stark::proof::options::ProofOptions) -> Self {
        let e = super::epoch_tests::real_epoch_from(
            inner,
            super::epoch_tests::EpochInputs::fixture(),
        );
        let program = super::epoch_tests::epoch_program(&e, true);
        let arenas = super::epoch_tests::epoch_arena_words(&e, true);
        let exec = super::executor::execute(&program, &arenas, &crate::hash_pin::BLOCK_HASHER)
            .expect("the epoch verifier executes");
        let s = Instant::now();
        let fvm_bits = std::env::var_os("FVM_SPLIT_RISCV_BITS").is_none();
        let cut = split::cut(&program, &exec, &arenas, fvm_bits);
        let cut_ms = s.elapsed().as_secs_f64() * 1e3;
        let crossing = cut.crossing();
        let tr = crate::field_vm::lfm_translate::translate_keeping(
            &program,
            &exec,
            fvm_bits,
            &crossing,
            &|k, _| cut.riscv[k],
        );
        let faddr = tr.kept.clone();
        let mut public: Vec<u64> = tr.public.iter().chain(&faddr).copied().collect();
        public.sort_unstable();
        public.dedup();
        let run = execute(
            &tr.program,
            Memory::with_init(tr.mem.len(), &tr.mem),
            &mut NoHints,
            1 << 26,
        )
        .expect("the field half executes");
        let opts = default_options();
        let pid = program_id(&tr.program, &opts);
        let cells = public_cells(&run, &public);
        let back_l3 = cut
            .back
            .iter()
            .filter(|a| cut.values[a][3] != crate::tables::types::FE::zero())
            .count();
        eprintln!(
            "  cut: {} LFM instructions, riscv half {:?} ({} field instructions moved), guest cells {}, program {} words, cut_ms={cut_ms:.1}",
            program.instrs.len(),
            cut.counts,
            cut.moved,
            cut.cells,
            cut.words.len(),
        );
        eprintln!(
            "  crossing: record = out {} ({:?}) + back {} (from {:?}; fourth lane set on {back_l3}) = {} cells {} bytes; shared input {} hints {} bytes; guest hint words {}; query bits on the {}",
            cut.out.len(),
            cut.out_origin,
            cut.back.len(),
            cut.back_origin,
            cut.out.len() + cut.back.len(),
            24 * (cut.out.len() + cut.back.len()),
            cut.input.len(),
            24 * cut.input.len(),
            cut.hints.len(),
            if fvm_bits { "Field VM" } else { "RISC-V half" },
        );
        eprintln!(
            "  fvm half: program {} mem {} public {} (epoch {})",
            tr.program.len(),
            tr.mem.len(),
            public.len(),
            tr.public.len(),
        );
        Split {
            cut,
            faddr,
            fvm_program: tr.program,
            run,
            public,
            pid,
            cells,
        }
    }

    fn fvm_prove(&self) -> (FvmProof, f64) {
        let s = Instant::now();
        let mut traces = generate_traces(&self.fvm_program, &self.run, &self.public);
        let proof = prove_traces(&self.pid, &self.cells, &mut traces, &default_options())
            .expect("the field half proves");
        (proof, s.elapsed().as_secs_f64() * 1e3)
    }

    fn coupled(&self, cells: &[PublicCell], riscv_output: &[u8]) -> bool {
        split::coupled(
            &self.cut,
            &self.cut.program_digest(),
            cells,
            &self.faddr,
            riscv_output,
        )
    }

    fn perms(&self) -> usize {
        self.cut.counts.get("hash").copied().unwrap_or(0)
    }
}

/// Runs the guest without proving or keeping its logs: `(cycles, public
/// output)`, `None` when the guest aborts.
fn try_guest_execute(elf: &[u8], input: &[u8]) -> Option<(usize, Vec<u8>)> {
    let program = executor::elf::Elf::load(elf).expect("the guest ELF loads");
    let mut vm = executor::vm::execution::Executor::new(&program, input.to_vec())
        .expect("the guest starts");
    let mut cycles = 0;
    while let Some(chunk) = vm.resume().ok()? {
        cycles += chunk.len();
    }
    Some((cycles, vm.finish().ok()?.memory_values))
}

fn guest_execute(elf: &[u8], input: &[u8]) -> (usize, Vec<u8>) {
    try_guest_execute(elf, input).expect("the guest runs")
}

/// Continuation epochs of `2^FVM_SPLIT_EPOCH_LOG2` cycles (default 21): the
/// guest runs to hundreds of millions of cycles, past a monolithic proof's
/// memory.
fn epoch_log2() -> u32 {
    std::env::var("FVM_SPLIT_EPOCH_LOG2")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(21)
}

fn guest_prove(elf: &[u8], input: &[u8]) -> (crate::continuation::ContinuationProof, f64) {
    let s = Instant::now();
    let proof = crate::continuation::prove_continuation(
        elf,
        input,
        epoch_log2(),
        &super::proof::block_base_options(),
    )
    .expect("the hash half proves");
    (proof, s.elapsed().as_secs_f64() * 1e3)
}

/// The public output when the proof verifies.
fn guest_verify(
    elf: &[u8],
    proof: &crate::continuation::ContinuationProof,
) -> (Option<Vec<u8>>, f64) {
    let s = Instant::now();
    let out =
        crate::continuation::verify_continuation(elf, proof, &super::proof::block_base_options())
            .ok()
            .flatten();
    (out, s.elapsed().as_secs_f64() * 1e3)
}

#[test]
#[ignore]
fn epoch_split_fvm_riscv() {
    let t = Instant::now();
    let sp = Split::new(super::proof_fixture::fixture_options());
    eprintln!("  built in {:.1}s", t.elapsed().as_secs_f64());
    let elf = guest_elf();
    let input = sp.cut.guest_input(&sp.cut.out_values(), 1);

    let s = Instant::now();
    let (cycles, output) = guest_execute(&elf, &input);
    let exec_ms = s.elapsed().as_secs_f64() * 1e3;
    assert_eq!(
        output,
        sp.cut.expected_output(),
        "the guest hashes the program and computes the LFM's input and record"
    );
    assert!(sp.coupled(&sp.cells, &output), "the executions couple");
    eprintln!(
        "  riscv guest: input {} bytes, cycles={cycles} ({:.0} per RPX permutation, {} of them) execute_ms={exec_ms:.1}",
        input.len(),
        cycles as f64 / sp.perms().max(1) as f64,
        sp.perms()
    );
    if std::env::var_os("FVM_SPLIT_NO_PROVE").is_some() {
        return;
    }

    let opts = default_options();
    let (mut f_prove, mut f_verify, mut r_prove, mut r_verify, mut total) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut f_shape, mut f_bytes, mut r_shape, mut r_bytes, mut r_epochs) =
        ((0, 0), 0, (0, 0), 0, 0);
    for _ in 0..runs() {
        let (fproof, fp) = sp.fvm_prove();
        let s = Instant::now();
        assert!(
            verify(&sp.pid, &sp.cells, &fproof, &opts),
            "the field half verifies"
        );
        let fv = s.elapsed().as_secs_f64() * 1e3;
        f_shape = census(&fproof);
        f_bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&fproof)
            .map(|b| b.len())
            .unwrap_or(0);
        drop(fproof);

        let (rproof, rp) = guest_prove(&elf, &input);
        let (routput, rv) = guest_verify(&elf, &rproof);
        let routput = routput.expect("the hash half verifies");
        assert_eq!(routput, output, "the proof commits what the guest ran");
        r_shape = rproof.census();
        r_epochs = rproof.num_epochs();
        r_bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&rproof)
            .map(|b| b.len())
            .unwrap_or(0);
        assert!(
            sp.coupled(&sp.cells, &routput),
            "both halves describe one input and one record"
        );
        f_prove.push(fp);
        f_verify.push(fv);
        r_prove.push(rp);
        r_verify.push(rv);
        total.push(fp + rp);
    }
    eprintln!(
        "  fvm blowup={} rows={} cells={} prove_ms={} verify_ms={} proof_bytes={f_bytes}",
        opts.blowup_factor,
        f_shape.0,
        f_shape.1,
        stats(f_prove),
        stats(f_verify)
    );
    eprintln!(
        "  riscv blowup={} cycles={cycles} epochs={r_epochs}x2^{} rows={} cells={} prove_ms={} verify_ms={} proof_bytes={r_bytes}",
        super::proof::block_base_options().blowup_factor,
        epoch_log2(),
        r_shape.0,
        r_shape.1,
        stats(r_prove),
        stats(r_verify)
    );
    eprintln!("  split total prove_ms={}", stats(total));
}

/// The guest's cost as its program grows: `FVM_SPLIT_REPS` (default `1,2,4`)
/// re-runs of the fixture's RISC-V half, for the per-permutation slope;
/// `FVM_SPLIT_NO_PROVE` only counts cycles.
#[test]
#[ignore]
fn riscv_split_scaling() {
    let sp = Split::new(super::proof_fixture::fixture_options());
    let elf = guest_elf();
    let reps: Vec<u32> = std::env::var("FVM_SPLIT_REPS")
        .unwrap_or_else(|_| "1,2,4".into())
        .split(',')
        .map(|r| r.trim().parse().expect("a repetition count"))
        .collect();
    for k in reps {
        let input = sp.cut.guest_input(&sp.cut.out_values(), k);
        let (cycles, output) = guest_execute(&elf, &input);
        assert_eq!(output, sp.cut.expected_output());
        let perms = sp.perms() * k as usize;
        if std::env::var_os("FVM_SPLIT_NO_PROVE").is_some() {
            eprintln!(
                "  riscv reps={k} perms={perms} cycles={cycles} ({:.0} per permutation)",
                cycles as f64 / perms as f64
            );
            continue;
        }
        let mut ms = Vec::new();
        let mut cells = 0;
        for _ in 0..runs() {
            let (proof, p) = guest_prove(&elf, &input);
            ms.push(p);
            cells = proof.census().1;
        }
        eprintln!(
            "  riscv reps={k} perms={perms} cycles={cycles} epochs=2^{} cells={cells} prove_ms={}",
            epoch_log2(),
            stats(ms)
        );
    }
}

/// One side changes a crossing value and the coupling rejects: a guest run
/// over another shared input (a hint the transcript absorbs, so the run still
/// completes and, with `FVM_SPLIT_PROVE_TAMPERED`, proves and verifies on its
/// own), a guest assuming another `out` value, and a Field VM proof claimed
/// with another record value, which its own verifier rejects too.
#[test]
#[ignore]
fn epoch_split_rejects_mismatch() {
    let sp = Split::new(super::proof_fixture::fixture_options());
    let elf = guest_elf();
    let opts = default_options();
    let (fproof, _) = sp.fvm_prove();
    assert!(verify(&sp.pid, &sp.cells, &fproof, &opts));
    let (_, output) = guest_execute(&elf, &sp.cut.guest_input(&sp.cut.out_values(), 1));
    assert!(sp.coupled(&sp.cells, &output), "honest");

    // RISC-V side: the guest reads a shared hint with one lane moved.
    let mut tampered = None;
    for (&a, &(h, lane)) in sp.cut.input.iter().zip(&sp.cut.input_src).rev().take(64) {
        let mut hints = sp.cut.hints.clone();
        hints[h][lane] += crate::tables::types::FE::one();
        let input = sp.cut.guest_input_with(&sp.cut.out_values(), &hints, 1);
        match try_guest_execute(&elf, &input) {
            Some((_, out)) => {
                tampered = Some((a, input, out));
                break;
            }
            None => eprintln!("  shared hint {a} moved: the guest aborts"),
        }
    }
    let (a, input, tout) = tampered.expect("a shared hint the guest runs through");
    assert_ne!(tout[32..64], output[32..64], "another input digest");
    assert!(
        !sp.coupled(&sp.cells, &tout),
        "the coupling rejects a guest over another input"
    );
    eprintln!("  shared hint {a} moved: the guest runs, the coupling rejects");
    if std::env::var_os("FVM_SPLIT_PROVE_TAMPERED").is_some() {
        let (rproof, _) = guest_prove(&elf, &input);
        let (routput, _) = guest_verify(&elf, &rproof);
        let routput = routput.expect("the tampered guest run is itself a valid proof");
        assert_eq!(routput, tout);
        assert!(!sp.coupled(&sp.cells, &routput));
        eprintln!("  ... and its proof verifies on its own");
    }

    // RISC-V side: another `out` value, when the cut has any.
    if !sp.cut.out.is_empty() {
        let mut out = sp.cut.out_values();
        out[0][0] += crate::tables::types::FE::one();
        if let Some((_, tout)) = try_guest_execute(&elf, &sp.cut.guest_input(&out, 1)) {
            assert!(!sp.coupled(&sp.cells, &tout), "another out value");
        }
    }

    // Field VM side: a record cell (the first challenge) claimed with another
    // value.
    let records = sp.cut.out.len() + sp.cut.back.len();
    for &b in sp.faddr[..records].iter().take(1) {
        let mut cells = sp.cells.clone();
        let at = cells
            .iter()
            .position(|(x, _)| *x == b)
            .expect("a public record cell");
        cells[at].1 += crate::tables::types::FEE::one();
        assert!(
            !verify(&sp.pid, &cells, &fproof, &opts),
            "the field half binds its public cells"
        );
        assert!(
            !sp.coupled(&cells, &output),
            "the coupling rejects a Field VM record that moved"
        );
    }

    // Another program under the same input and record.
    let mut other = output.clone();
    other[0] ^= 1;
    assert!(!sp.coupled(&sp.cells, &other));
}

/// The production epoch (`FVM_EPOCH_PRESET`, e.g. `blowup4`): the field half
/// proved on the Field VM and the RISC-V half only executed, its cycles the
/// input to the RISC-V extrapolation.
#[test]
#[ignore]
fn epoch_split_production_estimate() {
    let t = Instant::now();
    let sp = Split::new(super::fvm_epoch_compare::inner_options());
    eprintln!("  built in {:.1}s", t.elapsed().as_secs_f64());
    let opts = default_options();
    let (mut prove, mut ver, mut shape) = (Vec::new(), Vec::new(), (0, 0));
    for _ in 0..runs() {
        let (proof, p) = sp.fvm_prove();
        let s = Instant::now();
        assert!(
            verify(&sp.pid, &sp.cells, &proof, &opts),
            "the field half verifies"
        );
        ver.push(s.elapsed().as_secs_f64() * 1e3);
        prove.push(p);
        shape = census(&proof);
    }
    eprintln!(
        "  fvm blowup={} rows={} cells={} prove_ms={} verify_ms={}",
        opts.blowup_factor,
        shape.0,
        shape.1,
        stats(prove),
        stats(ver)
    );
    if std::env::var_os("FVM_SPLIT_NO_EXEC").is_some() {
        return;
    }
    let input = sp.cut.guest_input(&sp.cut.out_values(), 1);
    let s = Instant::now();
    let (cycles, output) = guest_execute(&guest_elf(), &input);
    assert_eq!(output, sp.cut.expected_output());
    assert!(sp.coupled(&sp.cells, &output));
    eprintln!(
        "  riscv guest: input {} bytes, cycles={cycles} ({:.0} per permutation, {} of them) execute_ms={:.1}",
        input.len(),
        cycles as f64 / sp.perms().max(1) as f64,
        sp.perms(),
        s.elapsed().as_secs_f64() * 1e3
    );
}
