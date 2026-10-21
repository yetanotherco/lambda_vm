//! Option A on a WHIR block leaf: the leaf's field half translated to the
//! Field VM, its hash half (the width-16 Poseidon1 socket, lanes, selects, bit
//! decompositions) on the LFM's chips, one proof under the block's proof
//! family, against the LFM proving the same leaf.
//!
//! The leaf comes from a small block (`test_commit_4`, the tree tests' fixture)
//! or, with `BLOCK_WHIR_ELF` and `BLOCK_WHIR_INPUT`, from a real block at the
//! production format. `FVM_LEAF_PROOF=path` keeps the real block's base proof
//! (written on the first run, read after), `FVM_LEAF=k` picks the leaf.
//!
//! `cargo test --release -p lambda-vm-prover --lib lfm::fvm_whir_leaf -- --ignored --nocapture --test-threads=1`

use std::time::Instant;

use stark::proof::options::ProofOptions;
use stark::proof::stark::MultiProof;

use crate::block_whir::{BlockFormat, BlockOptions, BlockWhirProof};
use crate::field_vm::executor::{Memory, NoHints, execute};
use crate::field_vm::prove::{program_id, public_cells};
use crate::hash_pin::{BlockFamily, ProofFamily};
use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};

use super::compiler::LfmProgram;
use super::whir_block::{WhirBlockPlan, block_leaf_arena};
use super::word::LfmWord;

type BlockStark = <BlockFamily as ProofFamily>::Stark;

fn runs() -> usize {
    std::env::var("FVM_COMPARE_RUNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3)
}

fn census<PI>(proof: &MultiProof<GoldilocksField, GoldilocksExtension, PI>) -> (usize, usize) {
    proof.proofs.iter().fold((0, 0), |(r, c), p| {
        (
            r + p.trace_length,
            c + p.trace_length * p.trace_ood_evaluations.width,
        )
    })
}

fn tables_of<PI>(p: &MultiProof<GoldilocksField, GoldilocksExtension, PI>) -> String {
    p.proofs
        .iter()
        .map(|p| format!("{}x{}", p.trace_length, p.trace_ood_evaluations.width))
        .collect::<Vec<_>>()
        .join(" ")
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

/// The block's plan and base proof: a real block when `BLOCK_WHIR_ELF` is set,
/// otherwise the tree tests' small block.
pub(super) fn leaf_source() -> (WhirBlockPlan, BlockWhirProof) {
    let Ok(elf_path) = std::env::var("BLOCK_WHIR_ELF") else {
        let format = super::whir_block_tests::small_format();
        let (elf, proof) = super::whir_block_tests::small_block("test_commit_4", &format);
        let plan = super::whir_block_tests::plan_of(&elf, &proof, &format, Some(2));
        return (plan, proof);
    };
    let elf = std::fs::read(elf_path).expect("read the ELF");
    let opts = super::proof::block_base_options();
    let format = BlockFormat::production();
    let saved = std::env::var("FVM_LEAF_PROOF").ok();
    let proof = match saved.as_deref().map(std::fs::read) {
        Some(Ok(bytes)) => {
            eprintln!("  base proof read ({} bytes)", bytes.len());
            rkyv::from_bytes::<BlockWhirProof, rkyv::rancor::Error>(&bytes)
                .expect("a saved block proof")
        }
        _ => {
            let input = std::fs::read(std::env::var("BLOCK_WHIR_INPUT").expect("BLOCK_WHIR_INPUT"))
                .expect("read the input");
            let s = Instant::now();
            let (proof, _) = crate::block_whir::prove_block_whir(
                &elf,
                &input,
                &opts,
                &format,
                &BlockOptions::production(),
            )
            .expect("the block proves");
            eprintln!("  base proved in {:.1}s", s.elapsed().as_secs_f64());
            if let Some(path) = &saved {
                let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&proof).expect("serialize");
                std::fs::write(path, &bytes).expect("write the block proof");
                eprintln!("  base proof written to {path} ({} bytes)", bytes.len());
            }
            proof
        }
    };
    let plan = WhirBlockPlan::derive(&elf, &opts, &format, proof.statement(), None)
        .expect("the plan derives");
    (plan, proof)
}

/// The Field VM translation of `program`'s field half and the hash half it
/// hands to the LFM chips, `MEM` merged onto the hash half's memory.
pub(super) struct Hybrid {
    pub(super) side: crate::field_vm::hash_side::HashSide,
    pub(super) tr: crate::field_vm::lfm_translate::Translation,
    pub(super) rows: Vec<crate::field_vm::bridge::BridgeRow>,
    pub(super) run: crate::field_vm::executor::Execution,
    pub(super) pid: crate::field_vm::prove::ProgramId,
    pub(super) id: crate::field_vm::hash_side::HashSideId,
    pub(super) cells: Vec<crate::field_vm::prove::PublicCell>,
}

impl Hybrid {
    pub(super) fn new(
        program: &LfmProgram,
        exec: &super::executor::LfmExecution,
        arenas: &[Vec<LfmWord>],
        opts: &ProofOptions,
        scopes: &[super::fn_scope::Scope],
    ) -> Self {
        use crate::field_vm::hash_side;
        let s = Instant::now();
        let mut side = hash_side::split(program, exec, arenas);
        let crossing = side.crossing();
        let tr = crate::field_vm::lfm_translate::translate_outlined_canonical(
            program,
            exec,
            false,
            &crossing,
            &|_, i| hash_side::is_hash_side(i),
            scopes,
        );
        let faddr: std::collections::HashMap<u64, u64> = crossing
            .iter()
            .copied()
            .zip(tr.kept.iter().copied())
            .collect();
        hash_side::renumber(&mut side, &|a| faddr[&a], tr.mem.len() as u64);
        let rows = hash_side::bridge_rows(&side, |a| a);
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
        let hasher = side.sub.hasher(BlockFamily::SOCKET);
        let artifacts =
            super::registry::build_artifacts_under::<BlockStark>(&side.sub, opts, hasher);
        let id = hash_side::side_id(&side, &rows, &artifacts, opts);
        eprintln!(
            "  hybrid hasher={hasher:?} build_ms={:.1}",
            s.elapsed().as_secs_f64() * 1e3
        );
        let (_, mut lfm_cells) = hash_side::traces(&side, &rows, &id, &run.mem);
        merge_const_lane3(&mut lfm_cells, &tr.const_lane3);
        let lane3: std::collections::HashMap<u64, FE> = lfm_cells
            .iter()
            .map(|&(addr, _, _, v3)| (addr, v3))
            .collect();
        let primary = public_cells(&run, &tr.public);
        let mut cells = primary.clone();
        cells.extend(primary.into_iter().map(|(addr, _)| {
            (
                addr + (1u64 << 39),
                FEE::new([
                    lane3.get(&addr).copied().unwrap_or(FE::zero()),
                    FE::zero(),
                    FE::zero(),
                ]),
            )
        }));
        Hybrid {
            side,
            tr,
            rows,
            run,
            pid,
            id,
            cells,
        }
    }

    /// What the node above a [`Hybrid`] proof of `program` emits over, with no
    /// run of `program`: the translation, the hash half and every table
    /// height are its shape's, so they come from an execution that wrote
    /// nothing and a Field VM run that does not stop at broken FMAs. The
    /// boundary states are the start and the halt, which needs the run in one
    /// segment (`FVM_ONE_SEGMENT`): a cut between segments falls on a state
    /// the witness decides.
    pub(super) fn shape(
        program: &LfmProgram,
        opts: &ProofOptions,
        scopes: &[super::fn_scope::Scope],
    ) -> Result<
        (
            super::fvm_node::HybridArtifacts,
            usize,
            Vec<crate::field_vm::air::FvmPublicInputs>,
        ),
        String,
    > {
        use crate::field_vm::hash_side;
        let zero: LfmWord = [FE::zero(); 4];
        let arenas: Vec<Vec<LfmWord>> = program
            .arena_schema
            .lens
            .iter()
            .map(|&n| vec![zero; n as usize])
            .collect();
        let mut side = hash_side::split_with(program, &arenas, &|_| zero);
        let crossing = side.crossing();
        let none = super::executor::LfmExecution::unwritten(program.num_addrs as usize);
        let tr = crate::field_vm::lfm_translate::translate_outlined_canonical(
            program,
            &none,
            false,
            &crossing,
            &|_, i| hash_side::is_hash_side(i),
            scopes,
        );
        let faddr: std::collections::HashMap<u64, u64> = crossing
            .iter()
            .copied()
            .zip(tr.kept.iter().copied())
            .collect();
        hash_side::renumber(&mut side, &|a| faddr[&a], tr.mem.len() as u64);
        let rows = hash_side::bridge_rows(&side, |a| a);
        let (run, _) = crate::field_vm::executor::execute_violating(
            &tr.program,
            Memory::with_init(tr.mem.len(), &tr.mem),
            &mut NoHints,
            1 << 26,
        )
        .map_err(|e| format!("the translation's shape run: {e:?}"))?;
        let pid = program_id(&tr.program, opts);
        let artifacts = super::registry::build_artifacts_under::<BlockStark>(
            &side.sub,
            opts,
            side.sub.hasher(BlockFamily::SOCKET),
        );
        let id = hash_side::side_id(&side, &rows, &artifacts, opts);
        let traces = crate::field_vm::prove::generate_traces_lfm(
            &tr.program,
            &run,
            &tr.public,
            None,
            &[],
            None,
        );
        if traces.segments.len() != 1 {
            return Err(format!(
                "{} segments: the states between them are the witness's",
                traces.segments.len()
            ));
        }
        let log2 = |t: &stark::trace::TraceTable<GoldilocksField, GoldilocksExtension>| {
            t.num_rows().trailing_zeros() as u8
        };
        let mut heights: Vec<u8> = traces
            .fvm
            .iter()
            .chain(&traces.decode)
            .chain(std::iter::once(&traces.mem))
            .map(log2)
            .collect();
        heights.extend(hash_side::heights(&side, &rows, &id, &artifacts)?);
        Ok((
            super::fvm_node::HybridArtifacts {
                pid,
                side: id,
                segments: traces.segments.len(),
                mem_lfm: true,
                heights,
            },
            2 * tr.public.len(),
            traces.segments,
        ))
    }

    pub(super) fn traces(
        &self,
    ) -> (
        crate::field_vm::prove::FvmTraces,
        Vec<stark::trace::TraceTable<GoldilocksField, GoldilocksExtension>>,
    ) {
        let (extra, mut cells) =
            crate::field_vm::hash_side::traces(&self.side, &self.rows, &self.id, &self.run.mem);
        merge_const_lane3(&mut cells, &self.tr.const_lane3);
        (
            crate::field_vm::prove::generate_traces_lfm(
                &self.tr.program,
                &self.run,
                &self.tr.public,
                None,
                &[],
                Some(&cells[..]),
            ),
            extra,
        )
    }

    /// Proves the traces; `None` when proving or verification fails.
    pub(super) fn prove(
        &self,
        traces: &mut crate::field_vm::prove::FvmTraces,
        extra: &mut [stark::trace::TraceTable<GoldilocksField, GoldilocksExtension>],
        opts: &ProofOptions,
    ) -> Option<(crate::field_vm::prove::FvmProof, f64, f64)> {
        use crate::field_vm::hash_side::airs;
        use crate::field_vm::prove::{prove_traces_with, verify_with};
        let s = Instant::now();
        let proof = prove_traces_with(
            &self.pid,
            &self.cells,
            traces,
            airs(&self.id, opts),
            extra,
            true,
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
            true,
            opts,
        );
        let verify_ms = s.elapsed().as_secs_f64() * 1e3;
        if !ok {
            debug_public_bus(&traces.mem, &self.cells);
        }
        ok.then_some((proof, prove_ms, verify_ms))
    }
}

/// Print a compact multiset diff between the tuples sent by MEM and the
/// verifier's public statement. Kept on the host verification failure path.
fn debug_public_bus(
    mem: &stark::trace::TraceTable<GoldilocksField, GoldilocksExtension>,
    public: &[crate::field_vm::prove::PublicCell],
) {
    use crate::field_vm::mem::cols;
    use std::collections::BTreeMap;

    let mut counts = BTreeMap::<String, isize>::new();
    let mut record = |sign: isize, tuple: (u64, [FE; 3])| {
        *counts.entry(format!("{tuple:?}")).or_default() += sign;
    };
    for row in 0..mem.num_rows() {
        if *mem.main_table.get(row, cols::PUB) == FE::one() {
            let addr = mem.main_table.get(row, cols::ADDR).canonical();
            record(
                1,
                (
                    addr,
                    [
                        *mem.main_table.get(row, cols::VALUE),
                        *mem.main_table.get(row, cols::VALUE + 1),
                        *mem.main_table.get(row, cols::VALUE + 2),
                    ],
                ),
            );
            if mem.num_cols() > cols::V3 {
                record(
                    1,
                    (
                        addr + (1u64 << 39),
                        [*mem.main_table.get(row, cols::V3), FE::zero(), FE::zero()],
                    ),
                );
            }
        }
    }
    for &(addr, value) in public {
        let v = value.value();
        record(-1, (addr, [v[0], v[1], v[2]]));
    }
    let mismatches: Vec<_> = counts
        .into_iter()
        .filter(|(_, count)| *count != 0)
        .take(8)
        .collect();
    let well_formed = crate::field_vm::prove::public_well_formed(public);
    eprintln!(
        "  FIELD_VM_PUBLIC statement well_formed={well_formed}, MEM/statement multiset mismatches: {mismatches:?}"
    );
}

fn merge_const_lane3(cells: &mut Vec<crate::field_vm::mem::LfmCell>, constants: &[(u64, FE)]) {
    for &(addr, lane3) in constants {
        let mut found = false;
        for cell in cells.iter_mut().filter(|cell| cell.0 == addr) {
            // The hash-side trace may contain several memory events for one
            // address. A constant's lane 3 is program text, so every event at
            // that address must carry the LFM_CONST value; updating only the
            // first event lets a later duplicate overwrite it when the
            // address-to-lane3 map is built for companion public cells.
            cell.3 = lane3;
            found = true;
        }
        if !found {
            cells.push((addr, 0, 0, lane3));
        }
    }
}

/// Option A on one WHIR block leaf, against the LFM proving the same leaf,
/// both at the tree's wrap options, interleaved run by run.
#[test]
#[ignore]
fn whir_leaf_field_vm_hash_chip() {
    let (plan, proof) = leaf_source();
    let k: usize = std::env::var("FVM_LEAF")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let (program, scopes) = super::fn_scope::record(|| plan.leaf_program(k));
    let program = program.expect("the leaf emits");
    let arenas = block_leaf_arena(&plan, &proof, k).expect("the leaf's arena");
    let hasher = program.hasher(BlockFamily::SOCKET);
    let exec = super::executor::execute(&program, &arenas, &hasher).expect("the leaf executes");
    let mut mix = std::collections::BTreeMap::new();
    for i in &program.instrs {
        let kind = format!("{i:?}");
        let kind = kind.split([' ', '(', '{']).next().unwrap_or("").to_string();
        *mix.entry(kind).or_insert(0usize) += 1;
    }
    eprintln!(
        "leaf {k} of {}: {} LFM instructions, hasher {hasher:?}, mix {mix:?}",
        plan.partition().num_leaves(),
        program.instrs.len()
    );
    for c in super::airs::lfm_chip_census(&program) {
        eprintln!(
            "  chip {:<12} rows={:>8} real={:>8} main={:>4} aux={:>3} cells={:>10}",
            c.name,
            c.rows,
            c.real_rows,
            c.main_cols,
            c.aux_cols,
            c.main_cells() + c.aux_cells()
        );
    }
    let opts = super::proof::aggregation_wrap_options();
    let hy = Hybrid::new(&program, &exec, &arenas, &opts, &scopes);
    let with_lfm = std::env::var_os("FVM_COMPARE_SKIP_LFM").is_none();
    let artifacts = with_lfm.then(|| super::whir_block::artifacts_of(&program, &opts));

    let (mut h_prove, mut h_verify, mut l_prove, mut l_verify, mut ratio) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut h_shape, mut h_bytes, mut h_tables) = ((0, 0), 0, String::new());
    let (mut l_shape, mut l_bytes, mut l_tables) = ((0, 0), 0, String::new());
    for _ in 0..runs() {
        let (mut traces, mut extra) = hy.traces();
        let (proof, p, v) = hy
            .prove(&mut traces, &mut extra, &opts)
            .expect("FVM + hash chips prove and verify");
        h_prove.push(p);
        h_verify.push(v);
        h_shape = census(&proof);
        h_bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&proof)
            .map(|b| b.len())
            .unwrap_or(0);
        h_tables = tables_of(&proof);
        drop(proof);
        if let Some(artifacts) = &artifacts {
            let s = Instant::now();
            let proof =
                super::proof::lfm_prove_under::<BlockFamily>(&program, artifacts, &arenas, &opts)
                    .expect("LFM proves");
            let lp = s.elapsed().as_secs_f64() * 1e3;
            l_prove.push(lp);
            ratio.push(p / lp);
            let s = Instant::now();
            assert!(super::proof::verify_against_artifacts_under::<BlockFamily>(
                artifacts,
                &proof.proof,
                &proof.public_words,
                &opts
            ));
            l_verify.push(s.elapsed().as_secs_f64() * 1e3);
            l_shape = census(&proof.proof);
            l_bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&proof.proof)
                .map(|b| b.len())
                .unwrap_or(0);
            l_tables = tables_of(&proof.proof);
        }
    }
    eprintln!("  hybrid tables: {h_tables}");
    eprintln!(
        "  hybrid blowup={} rows={} cells={} prove_ms={} verify_ms={} proof_bytes={h_bytes}",
        opts.blowup_factor,
        h_shape.0,
        h_shape.1,
        stats(h_prove),
        stats(h_verify)
    );
    if with_lfm {
        eprintln!("  lfm tables: {l_tables}");
        eprintln!(
            "  lfm blowup={} rows={} cells={} prove_ms={} verify_ms={} proof_bytes={l_bytes}",
            opts.blowup_factor,
            l_shape.0,
            l_shape.1,
            stats(l_prove),
            stats(l_verify)
        );
        eprintln!(
            "  paired hybrid/lfm prove ratio={} ({} of {} runs faster)",
            stats(ratio.iter().map(|r| r * 1e3).collect()),
            ratio.iter().filter(|&&r| r < 1.0).count(),
            ratio.len()
        );
    }
}

/// What crosses between the halves of a WHIR leaf: the cells the hash half
/// hands to the Field VM (`back`), by the instruction that writes them and by
/// the field instruction that reads them. Cuts only; proves nothing.
#[test]
#[ignore]
fn whir_leaf_crossing_census() {
    use super::instr::Instr;
    use crate::field_vm::hash_side;
    let (plan, proof) = leaf_source();
    let k: usize = std::env::var("FVM_LEAF")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let program = plan.leaf_program(k).expect("the leaf emits");
    let arenas = block_leaf_arena(&plan, &proof, k).expect("the leaf's arena");
    let hasher = program.hasher(BlockFamily::SOCKET);
    let exec = super::executor::execute(&program, &arenas, &hasher).expect("the leaf executes");
    let side = hash_side::split(&program, &exec, &arenas);
    let kind = |i: &Instr| {
        let s = format!("{i:?}");
        s.split([' ', '(', '{']).next().unwrap_or("").to_string()
    };
    let n = program.num_addrs as usize;
    let mut writer: Vec<Option<String>> = vec![None; n];
    let mut buf = Vec::new();
    for i in &program.instrs {
        buf.clear();
        i.writes_into(&mut buf);
        for a in &buf {
            writer[a.0 as usize] = Some(kind(i));
        }
    }
    let mut is_back = vec![false; n];
    for &a in &side.back {
        is_back[a as usize] = true;
    }
    let mut by_writer = std::collections::BTreeMap::new();
    for &a in &side.back {
        *by_writer
            .entry(writer[a as usize].clone().unwrap_or_default())
            .or_insert(0usize) += 1;
    }
    let mut by_reader = std::collections::BTreeMap::new();
    for i in program
        .instrs
        .iter()
        .filter(|i| !hash_side::is_hash_side(i))
    {
        buf.clear();
        i.reads_into(&mut buf);
        for a in &buf {
            if is_back[a.0 as usize] {
                *by_reader.entry(kind(i)).or_insert(0usize) += 1;
            }
        }
    }
    let mut out_by_writer = std::collections::BTreeMap::new();
    for c in &side.out {
        *out_by_writer
            .entry(writer[c.addr as usize].clone().unwrap_or_default())
            .or_insert(0usize) += 1;
    }
    eprintln!(
        "leaf {k}: {} instrs · back {} by writer {by_writer:?} · field reads of back cells by reader {by_reader:?} · out {} by writer {out_by_writer:?}",
        program.instrs.len(),
        side.back.len(),
        side.out.len()
    );
}

/// How much of a WHIR leaf's translated v_fld is straight-line code: program
/// length and executed rows with and without `FVM_REROLL`, and the program
/// addresses executed once against those inside loops. Translates only.
#[test]
#[ignore]
fn whir_leaf_program_census() {
    use crate::field_vm::hash_side;
    let (plan, proof) = leaf_source();
    let k: usize = std::env::var("FVM_LEAF")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    use super::whir_chain::onechain_steps::{Step, record};
    let ((program, scopes), marks) = record(|| super::fn_scope::record(|| plan.leaf_program(k)));
    let program = program.expect("the leaf emits");
    let mut names = std::collections::BTreeMap::new();
    for sc in &scopes {
        *names.entry(sc.name.clone()).or_insert(0usize) += 1;
    }
    eprintln!("  scopes: {names:?}");
    let label = |s: &Step| -> String {
        match s {
            Step::Part(p) => (*p).to_string(),
            Step::Chain(_) => "chain".into(),
            Step::Sumcheck { round } => format!("sumcheck r{round}"),
            Step::Ood { round } => format!("ood r{round}"),
            Step::Opening { round, .. } => format!("opening r{round}"),
            Step::Successor { round, .. } => format!("successor r{round}"),
            Step::Fold { round, .. } => format!("fold r{round}"),
            Step::Final => "final".into(),
        }
    };
    // The part each LFM instruction was emitted under.
    let mut part_of: Vec<String> = vec!["?".into(); program.instrs.len()];
    let mut m = 0;
    let mut current = "?".to_string();
    for (i, p) in part_of.iter_mut().enumerate() {
        while m < marks.len() && marks[m].1 <= i {
            current = label(&marks[m].0);
            m += 1;
        }
        *p = current.clone();
    }
    let arenas = block_leaf_arena(&plan, &proof, k).expect("the leaf's arena");
    let hasher = program.hasher(BlockFamily::SOCKET);
    let exec = super::executor::execute(&program, &arenas, &hasher).expect("the leaf executes");
    let side = hash_side::split(&program, &exec, &arenas);
    let crossing = side.crossing();
    // LFM instructions of the field half, by part and kind.
    let mut kinds: std::collections::BTreeMap<(String, String), usize> = Default::default();
    for (i, instr) in program.instrs.iter().enumerate() {
        if hash_side::is_hash_side(instr) {
            continue;
        }
        let kind = format!("{instr:?}");
        let kind = kind.split([' ', '(', '{']).next().unwrap_or("").to_string();
        let part = part_of[i].split(' ').next().unwrap_or("").to_string();
        *kinds.entry((part, kind)).or_default() += 1;
    }
    if let Some(start) = part_of.iter().position(|p| p == "ood r1") {
        let sample: Vec<String> = (start..program.instrs.len())
            .filter(|&i| !hash_side::is_hash_side(&program.instrs[i]))
            .take(40)
            .map(|i| format!("{i}: {:?}", program.instrs[i]))
            .collect();
        eprintln!("  ood r1 field-half sample:\n    {}", sample.join("\n    "));
    }
    for ((part, kind), n) in &kinds {
        eprintln!("  field half {part:<10} {kind:<8} {n:>8}");
    }
    let mut prev: Option<(
        Vec<crate::tables::types::FEE>,
        crate::field_vm::isa::Program,
    )> = None;
    for (reroll, outline) in [(false, false), (true, false), (false, true), (true, true)] {
        // SAFETY: single-threaded test; the translator reads the flags once.
        unsafe {
            for (flag, on) in [("FVM_REROLL", reroll), ("FVM_OUTLINE", outline)] {
                if on {
                    std::env::set_var(flag, "1");
                } else {
                    std::env::remove_var(flag);
                }
            }
        }
        let tr = crate::field_vm::lfm_translate::translate_outlined(
            &program,
            &exec,
            false,
            &crossing,
            &|_, i| hash_side::is_hash_side(i),
            &scopes,
        );
        if outline {
            if let Some((pmem, pprog)) = &prev {
                let pmem: &Vec<crate::tables::types::FEE> = pmem;
                let pprog: &crate::field_vm::isa::Program = pprog;
                let diff: Vec<usize> = (0..pmem.len().min(tr.mem.len()))
                    .filter(|&a| pmem[a] != tr.mem[a])
                    .collect();
                let same_prefix = (0..pprog.instrs.len().min(tr.program.instrs.len()))
                    .take_while(|&q| pprog.instrs[q] == tr.program.instrs[q])
                    .count();
                eprintln!(
                    "  vs previous outlined: mem len {} vs {}, {} cells differ (first {:?}), programs agree on the first {same_prefix} instrs",
                    pmem.len(),
                    tr.mem.len(),
                    diff.len(),
                    &diff[..diff.len().min(10)]
                );
            }
            prev = Some((tr.mem.clone(), tr.program.clone()));
        }
        let run = match execute(
            &tr.program,
            Memory::with_init(tr.mem.len(), &tr.mem),
            &mut NoHints,
            1 << 26,
        ) {
            Ok(run) => run,
            Err(e) => {
                if let crate::field_vm::executor::ExecError::FmaViolation { pc } = e {
                    let pc = pc as usize;
                    for q in pc.saturating_sub(6)..(pc + 3).min(tr.program.instrs.len()) {
                        let o = tr.origin[q];
                        eprintln!(
                            "  {}{q}: origin {} part {} lfm {:?}\n      {:?}",
                            if q == pc { ">" } else { " " },
                            o,
                            part_of.get(o).map_or("-", |p| p.as_str()),
                            program.instrs.get(o),
                            tr.program.instrs[q]
                        );
                    }
                }
                let ev = crate::field_vm::executor::execute_violating(
                    &tr.program,
                    Memory::with_init(tr.mem.len(), &tr.mem),
                    &mut NoHints,
                    1 << 26,
                );
                if let Err(e2) = &ev {
                    eprintln!("  execute_violating: {e2:?}");
                }
                if let Ok((_, v)) = ev {
                    let main = tr
                        .program
                        .instrs
                        .iter()
                        .position(|i| {
                            i.d.reg == 1
                                && !i.d.mem
                                && i.c.offset != crate::tables::types::FE::zero()
                        })
                        .unwrap_or(0);
                    eprintln!(
                        "  {} violations; first pcs {:?}",
                        v.len(),
                        &v[..v.len().min(20)]
                    );
                    let mut parts = std::collections::BTreeMap::new();
                    for &pc in &v {
                        let o = tr.origin[pc as usize];
                        *parts
                            .entry(part_of.get(o).cloned().unwrap_or_else(|| "control".into()))
                            .or_insert(0) += 1;
                    }
                    eprintln!("  violations by part: {parts:?} (entry jump at {main})");
                }
                panic!("the translation executes: {e:?}");
            }
        };
        let once = run.decode_mult.iter().filter(|&&m| m == 1).count();
        let looped = run.decode_mult.iter().filter(|&&m| m > 1).count();
        let looped_rows: u64 = run.decode_mult.iter().filter(|&&m| m > 1).sum();
        eprintln!(
            "leaf {k} reroll={reroll} outline={outline}: mem {} · program {} · rows {} · addresses run once {once} · in loops {looped} ({looped_rows} rows)",
            tr.mem.len(),
            tr.program.len(),
            run.steps.len(),
        );
        // part -> (program addresses, rows run once, addresses in loops, rows in loops)
        let mut by: std::collections::BTreeMap<String, [u64; 4]> = Default::default();
        for (pc, &mult) in run.decode_mult.iter().enumerate() {
            let o = tr.origin[pc];
            let part = if o == usize::MAX {
                "control".to_string()
            } else {
                part_of[o].clone()
            };
            let e = by.entry(part).or_default();
            e[0] += 1;
            if mult == 1 {
                e[1] += 1;
            } else if mult > 1 {
                e[2] += 1;
                e[3] += mult;
            }
        }
        for (part, [n, once, la, lr]) in &by {
            eprintln!(
                "    {part:<16} program {n:>8} · once {once:>8} · loop addrs {la:>7} rows {lr:>8}"
            );
        }
    }
}

/// The verifier part each LFM instruction was emitted under, from the step marks.
fn parts_of(
    program: &LfmProgram,
    marks: &[(super::whir_chain::onechain_steps::Step, usize)],
) -> Vec<String> {
    use super::whir_chain::onechain_steps::Step;
    let label = |s: &Step| -> String {
        match s {
            Step::Part(p) => (*p).to_string(),
            Step::Chain(_) => "chain".into(),
            Step::Sumcheck { round } => format!("sumcheck r{round}"),
            Step::Ood { round } => format!("ood r{round}"),
            Step::Opening { round, .. } => format!("opening r{round}"),
            Step::Successor { round, .. } => format!("successor r{round}"),
            Step::Fold { round, .. } => format!("fold r{round}"),
            Step::Final => "final".into(),
        }
    };
    let mut part_of: Vec<String> = vec!["?".into(); program.instrs.len()];
    let (mut m, mut current) = (0, "?".to_string());
    for (i, p) in part_of.iter_mut().enumerate() {
        while m < marks.len() && marks[m].1 <= i {
            current = label(&marks[m].0);
            m += 1;
        }
        *p = current.clone();
    }
    part_of
}

/// The generic v_bin (`field_vm::vbin_generic`) over the cut: its schedule,
/// the host reference run, then the guest; BACK and the published words
/// against the LFM's.
fn generic_vbin_parity(
    cut: &crate::field_vm::split::Cut,
    binding: Option<([u64; 4], [u64; 3], [u64; 3])>,
) {
    use crate::field_vm::vbin_generic as vg;
    let (c_bin, alpha, sum) =
        binding.expect("the generic guest binds: FVM_BIND=1 NATIVE_VBIN_BIND=1");
    let t = Instant::now();
    let sched = vg::compile(&cut.words).expect("the cut compiles");
    eprintln!(
        "  generic schedule: {} words ({} KiB), {} ops in {} top items, {} loops, {} TMP slots; compiled in {:.1} ms",
        sched.words.len(),
        4 * sched.words.len() / 1024,
        sched.ops,
        sched.items,
        sched.loops,
        sched.tmp_slots,
        t.elapsed().as_secs_f64() * 1e3
    );
    let canon =
        |w: &crate::lfm::word::LfmWord| -> [u64; 4] { core::array::from_fn(|l| w[l].canonical()) };
    let hints: Vec<[u64; 4]> = cut.hints.iter().map(canon).collect();
    let input = vg::input(&sched, &hints, &cut.in_cells(), alpha);
    let want_back: Vec<[u64; 3]> = cut
        .back
        .iter()
        .map(|a| core::array::from_fn(|l| cut.values[a][l].canonical()))
        .collect();
    let want_pub: Vec<[u64; 4]> = cut.publics.iter().map(canon).collect();
    let t = Instant::now();
    let r = vg::reference::run(&sched, input.clone());
    eprintln!(
        "  generic reference: {} ops run in {:.1} ms, err {}",
        r.ops,
        t.elapsed().as_secs_f64() * 1e3,
        r.err
    );
    let bad: Vec<usize> = (0..want_back.len())
        .filter(|&i| r.back[i][..3] != want_back[i])
        .collect();
    if let Some(&i) = bad.first() {
        eprintln!(
            "  BACK mismatches: {} of {}; first {i}: got {:?} want {:?} (cell {})",
            bad.len(),
            want_back.len(),
            r.back[i],
            want_back[i],
            cut.back[i]
        );
    }
    assert!(bad.is_empty(), "the reference's BACK");
    assert_eq!(r.publics, want_pub, "the reference's published words");
    assert!(!r.err, "the reference's checks pass");
    let expected = vg::expected_output(&sched, &want_pub, c_bin, alpha, sum);
    let elf = vg::guest();
    if elf.is_empty() {
        eprintln!("  generic guest: not built yet");
        return;
    }
    let t = Instant::now();
    let program = executor::elf::Elf::load(elf).expect("the guest ELF loads");
    let mut vm = executor::vm::execution::Executor::new(&program, input).expect("the guest starts");
    let mut cycles = 0;
    while let Some(chunk) = vm.resume().expect("the guest runs") {
        cycles += chunk.len();
    }
    let got_back = vg::back_cells(&sched, |a| {
        vm.memory().load_doubleword(a).expect("BACK reads")
    });
    let out = vm.finish().expect("the guest finishes").memory_values;
    let elf_digest: [u8; 32] = {
        use tiny_keccak::{Hasher, Keccak};
        let mut k = Keccak::v256();
        k.update(elf);
        let mut d = [0u8; 32];
        k.finalize(&mut d);
        d
    };
    eprintln!(
        "  generic guest: cycles={cycles} ({:.2} x 2^20), elf {} bytes (keccak {}), schedule digest {:x?}, execute_ms={:.1}",
        cycles as f64 / (1u64 << 20) as f64,
        elf.len(),
        elf_digest
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>(),
        vg::schedule_digest(&sched.words),
        t.elapsed().as_secs_f64() * 1e3
    );
    assert_eq!(got_back, want_back, "the guest's BACK");
    assert_eq!(out, expected, "the guest's output");
    let input = vg::input(&sched, &hints, &cut.in_cells(), alpha);
    if std::env::var_os("VBG_PROFILE").is_some() {
        eprintln!(
            "  generic cycles by part: {}",
            profile_cycles(elf, &input, vg::guest_profile())
        );
    }
    if std::env::var_os("FVM_VBIN_PROVE").is_some() {
        prove_vbin(elf, &input, &expected);
    }
}

/// Runs a region guest: `(cycles, public output, BACK's bytes)`, `None` when
/// it aborts.
fn run_regions(
    elf: &[u8],
    input: &[u8],
    back_base: u64,
    n_back: usize,
) -> Option<(usize, Vec<u8>, Vec<u8>)> {
    let program = executor::elf::Elf::load(elf).expect("the guest ELF loads");
    let mut vm =
        executor::vm::execution::Executor::new(&program, input.to_vec()).expect("the guest starts");
    let mut cycles = 0;
    while let Some(chunk) = vm.resume().ok()? {
        cycles += chunk.len();
    }
    let back = (0..3 * n_back as u64)
        .flat_map(|k| {
            vm.memory()
                .load_doubleword(back_base + 8 * k)
                .expect("BACK reads")
                .to_le_bytes()
        })
        .collect();
    Some((cycles, vm.finish().ok()?.memory_values, back))
}

/// Cycles of a region guest by the code part each PC falls in.
fn profile_cycles(elf: &[u8], input: &[u8], parts: &[(u64, &'static str)]) -> String {
    let program = executor::elf::Elf::load(elf).expect("the guest ELF loads");
    let mut vm =
        executor::vm::execution::Executor::new(&program, input.to_vec()).expect("the guest starts");
    let mut by: std::collections::BTreeMap<&'static str, u64> = Default::default();
    let mut total = 0u64;
    while let Some(chunk) = vm.resume().expect("the guest runs") {
        for log in chunk {
            let i = parts.partition_point(|p| p.0 <= log.current_pc);
            *by.entry(if i == 0 { "?" } else { parts[i - 1].1 })
                .or_insert(0) += 1;
            total += 1;
        }
    }
    let mut v: Vec<_> = by.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1));
    v.iter()
        .map(|(k, c)| format!("{k} {c} ({:.1}%)", 100.0 * *c as f64 / total as f64))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Rows of a region of `cells` 24-byte cells: a power of two, at least one
/// page's worth.
fn region_rows(cells: usize) -> usize {
    const CELL_BYTES: usize = 24;
    let page = crate::tables::page::DEFAULT_PAGE_SIZE;
    let gcd = |mut a: usize, mut b: usize| {
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    };
    cells.next_power_of_two().max(page / gcd(CELL_BYTES, page))
}

/// Option B's hash half of a WHIR leaf: the cut of the leaf's verifier, its
/// RISC-V half compiled to a native guest (`field_vm::native_vbin`, the
/// width-16 Poseidon1 permutation through its ecall), run on the Lambda VM's
/// executor. The guest's public output and the BACK cells it writes must be
/// the LFM's; reports its cycles by part. Executes only.
#[test]
#[ignore]
fn whir_leaf_native_vbin() {
    use crate::field_vm::{native_vbin, split};
    let (plan, proof) = leaf_source();
    let k: usize = std::env::var("FVM_LEAF")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let ((program, scopes), marks) = super::whir_chain::onechain_steps::record(|| {
        super::fn_scope::record(|| plan.leaf_program(k))
    });
    let program = program.expect("the leaf emits");
    let part_of = parts_of(&program, &marks);
    let arenas = block_leaf_arena(&plan, &proof, k).expect("the leaf's arena");
    let hasher = program.hasher(BlockFamily::SOCKET);
    let exec = super::executor::execute(&program, &arenas, &hasher).expect("the leaf executes");
    let s = Instant::now();
    let fvm_bits = std::env::var_os("FVM_SPLIT_RISCV_BITS").is_none();
    let cut = split::cut(&program, &exec, &arenas, fvm_bits);
    eprintln!(
        "leaf {k}: {} LFM instructions; cut in {:.1} ms: riscv {} instrs (moved {}), out {}, back {}, input {}, hints {}, words {}, counts {:?}",
        program.instrs.len(),
        s.elapsed().as_secs_f64() * 1e3,
        cut.riscv.iter().filter(|&&r| r).count(),
        cut.moved,
        cut.out.len(),
        cut.back.len(),
        cut.input.len(),
        cut.hints.len(),
        cut.words.len(),
        cut.counts
    );
    eprintln!("  back by origin: {:?}", cut.back_origin);
    let s = Instant::now();
    let g = native_vbin::build(&cut.words).expect("the cut's program decodes");
    eprintln!(
        "  native guest: elf {} bytes = code {} words + data {} bytes; {} ops in {} top items, {} loops, depth {}; built in {:.1} ms",
        g.elf.len(),
        g.code_words,
        g.data_bytes,
        g.ops,
        g.top_items,
        g.loops,
        g.max_depth,
        s.elapsed().as_secs_f64() * 1e3,
    );
    eprintln!("  native guest code: {}", g.report);
    let n_in = cut.out.len() + cut.input.len();
    let n_back = cut.back.len();
    let coords = |x: crate::tables::types::FEE| {
        let v = x.value();
        [v[0].canonical(), v[1].canonical(), v[2].canonical()]
    };
    // A binding guest needs α, drawn from the field half's MEM root: prove
    // that half first.
    let binding = if native_vbin::bind_enabled() {
        Some(
            prove_vfld(&program, &exec, &cut, fvm_bits, &scopes, &part_of)
                .expect("FVM_BIND with NATIVE_VBIN_BIND"),
        )
    } else {
        None
    };
    let (input, layout) = match binding {
        Some((_, alpha, _)) => cut.native_input_regions_bound(
            &cut.in_cells(),
            &cut.hints,
            region_rows(n_in),
            region_rows(n_back),
            coords(alpha),
        ),
        None => cut.native_input_regions(
            &cut.in_cells(),
            &cut.hints,
            region_rows(n_in),
            region_rows(n_back),
        ),
    };
    let expected = match binding {
        Some((c_bin, alpha, s)) => {
            cut.expected_output_bound(&layout, c_bin, coords(alpha), coords(s))
        }
        None => cut.expected_output_regions(&layout),
    };
    let back: Vec<u8> = cut
        .back
        .iter()
        .flat_map(|a| {
            cut.values[a][..3]
                .iter()
                .flat_map(|x| x.canonical().to_le_bytes())
                .collect::<Vec<u8>>()
        })
        .collect();
    if std::env::var_os("VBG_GENERIC").is_some() {
        generic_vbin_parity(&cut, binding.map(|(c, a, s)| (c, coords(a), coords(s))));
        return;
    }
    // `FVM_VBIN_DUMP=dir`: keep the guest and its input, for the block tree.
    if let Some(dir) = std::env::var_os("FVM_VBIN_DUMP") {
        let dir = std::path::PathBuf::from(dir);
        std::fs::create_dir_all(&dir).expect("the dump directory");
        std::fs::write(dir.join("vbin.elf"), &g.elf).expect("write the guest");
        std::fs::write(dir.join("vbin_input.bin"), &input).expect("write the input");
        eprintln!("  dumped v_bin guest and input to {dir:?}");
    }
    let s = Instant::now();
    let (cycles, out, got_back) =
        run_regions(&g.elf, &input, layout.back_base, n_back).expect("the native guest runs");
    let ms = s.elapsed().as_secs_f64() * 1e3;
    assert_eq!(
        out, expected,
        "the guest's output (with the binding, the same c_bin and s)"
    );
    assert_eq!(got_back, back, "the guest's BACK");
    eprintln!(
        "  native v_bin: cycles={cycles} ({:.2} x 2^20), input {} bytes, execute_ms={ms:.1}",
        cycles as f64 / (1u64 << 20) as f64,
        input.len()
    );
    eprintln!(
        "  native cycles by part: {}",
        profile_cycles(&g.elf, &input, &g.profile)
    );
    if binding.is_none() && std::env::var_os("FVM_VFLD_PROVE").is_some() {
        prove_vfld(&program, &exec, &cut, fvm_bits, &scopes, &part_of);
    }
    if std::env::var_os("FVM_VBIN_PROVE").is_some() {
        prove_vbin(&g.elf, &input, &out);
    }
}

/// Option B's field half: the cut's field instructions translated to the
/// Field VM (crossing cells kept and published, as the old coupling did),
/// executed and proved at the tree's wrap options. Reports cells and time.
fn prove_vfld(
    program: &LfmProgram,
    exec: &super::executor::LfmExecution,
    cut: &crate::field_vm::split::Cut,
    fvm_bits: bool,
    scopes: &[super::fn_scope::Scope],
    part_of: &[String],
) -> Option<(
    [u64; 4],
    crate::tables::types::FEE,
    crate::tables::types::FEE,
)> {
    use crate::field_vm::prove::{generate_traces, prove_traces, verify};
    let tr = crate::field_vm::lfm_translate::translate_outlined(
        program,
        exec,
        fvm_bits,
        &cut.crossing(),
        &|k, _| cut.riscv[k] && !cut.both[k],
        scopes,
    );
    let mut public: Vec<u64> = tr.public.iter().chain(&tr.kept).copied().collect();
    public.sort_unstable();
    public.dedup();
    let run = execute(
        &tr.program,
        Memory::with_init(tr.mem.len(), &tr.mem),
        &mut NoHints,
        1 << 27,
    )
    .expect("the field half executes");
    eprintln!(
        "  v_fld (B): program {} rows {} mem {} public {} (crossing {})",
        tr.program.len(),
        run.steps.len(),
        tr.mem.len(),
        public.len(),
        tr.kept.len()
    );
    let mut by: std::collections::BTreeMap<String, [u64; 4]> = Default::default();
    for (pc, &mult) in run.decode_mult.iter().enumerate() {
        let o = tr.origin[pc];
        let part = if o == usize::MAX {
            "control".to_string()
        } else {
            let p = &part_of[o];
            let kind = format!("{:?}", program.instrs[o]);
            format!(
                "{} {}",
                p.split(' ').next().unwrap_or(""),
                kind.split([' ', '(', '{']).next().unwrap_or("")
            )
        };
        let e = by.entry(part).or_default();
        e[0] += 1;
        if mult == 1 {
            e[1] += 1;
        } else if mult > 1 {
            e[2] += 1;
            e[3] += mult;
        }
    }
    for (part, [n, once, la, lr]) in &by {
        if *n > 2000 {
            eprintln!(
                "    {part:<24} program {n:>8} · once {once:>8} · loop addrs {la:>7} rows {lr:>8}"
            );
        }
    }
    if std::env::var_os("FVM_BIND").is_some() {
        return Some(prove_vfld_bound(&tr, &run, cut));
    }
    let opts = super::proof::aggregation_wrap_options();
    let pid = program_id(&tr.program, &opts);
    let cells = public_cells(&run, &public);
    let s = Instant::now();
    let mut traces = generate_traces(&tr.program, &run, &public);
    let proof = prove_traces(&pid, &cells, &mut traces, &opts).expect("the field half proves");
    let prove_s = s.elapsed().as_secs_f64();
    let s = Instant::now();
    assert!(
        verify(&pid, &cells, &proof, &opts),
        "the field half verifies"
    );
    let verify_s = s.elapsed().as_secs_f64();
    let (rows, total) = census(&proof);
    let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&proof)
        .map(|b| b.len())
        .unwrap_or(0);
    eprintln!(
        "  v_fld (B) proof: prove {prove_s:.1} s, rows {rows}, cells {total}, verify {verify_s:.2} s, proof {bytes} bytes, tables {}",
        tables_of(&proof)
    );
    None
}

/// Option B's field half bound to the hash half by a checksum: the crossing
/// cells are not published; `FIELD_VM_CKSUM` folds them under `α`, drawn from
/// the guest's digest of its copies and MEM's main root (computed before the
/// table, and checked against the proof's), and publishes `s`. Returns
/// `(c_bin, α, s)` for the guest.
fn prove_vfld_bound(
    tr: &crate::field_vm::lfm_translate::Translation,
    run: &crate::field_vm::executor::Execution,
    cut: &crate::field_vm::split::Cut,
) -> (
    [u64; 4],
    crate::tables::types::FEE,
    crate::tables::types::FEE,
) {
    use crate::field_vm::cksum;
    use crate::field_vm::prove::{
        Airs, ExtraAir, commit_prep_with, generate_traces_reading, prove_traces_with, verify_with,
    };
    use crate::tables::types::{FE, FEE};
    use stark::lookup::{AirWithBuses, AuxiliaryTraceBuildData, NullBoundaryConstraintBuilder};
    let (n_out, n_back) = (cut.out.len(), cut.back.len());
    let kept = &tr.kept;
    // The guest's order: its IN region (`out`, then `input`), then BACK.
    let faddr: Vec<u64> = kept[..n_out]
        .iter()
        .chain(&kept[n_out + n_back..])
        .chain(&kept[n_out..n_out + n_back])
        .copied()
        .collect();
    let in_cells = cut.in_cells();
    let back_cells: Vec<[u64; 3]> = cut
        .back
        .iter()
        .map(|a| {
            let w = cut.values[a];
            [w[0].canonical(), w[1].canonical(), w[2].canonical()]
        })
        .collect();
    let cells: Vec<[u64; 3]> = in_cells.iter().chain(&back_cells).copied().collect();
    let values: Vec<FEE> = faddr.iter().map(|&a| run.mem[a as usize]).collect();
    for (i, (v, c)) in values.iter().zip(&cells).enumerate() {
        let v = v.value();
        assert_eq!(
            [v[0].canonical(), v[1].canonical(), v[2].canonical()],
            *c,
            "crossing cell {i}: the halves hold the same value"
        );
    }
    let opts = super::proof::aggregation_wrap_options();
    let pid = program_id(&tr.program, &opts);
    let mut public = tr.public.clone();
    public.sort_unstable();
    public.dedup();
    let s = Instant::now();
    let mut traces = generate_traces_reading(&tr.program, run, &public, None, &faddr);
    let c_bin = cksum::crossing_digest(&in_cells, &back_cells);
    if std::env::var("FVM_VFLD_WHIR").as_deref() == Ok("1") {
        let build_s = s.elapsed().as_secs_f64();
        let cells: usize = traces
            .fvm
            .iter()
            .chain(&traces.decode)
            .chain(std::iter::once(&traces.mem))
            .map(|t| t.num_rows() * t.num_cols())
            .sum::<usize>()
            + faddr.len().next_power_of_two().max(8) * crate::field_vm::cksum::cols::NUM_COLUMNS;
        let pubs = public_cells(run, &public);
        let s = Instant::now();
        let (proof, alpha) = crate::field_vm::whir::prove_whir_bound(
            &pid,
            &tr.program,
            &pubs,
            &mut traces,
            &c_bin,
            &faddr,
            &values,
            &opts,
        )
        .expect("the bound WHIR field half proves");
        let prove_s = s.elapsed().as_secs_f64();
        let s_fld = crate::field_vm::cksum::horner(&values, alpha);
        let verify_s = {
            let t = Instant::now();
            assert!(
                crate::field_vm::whir::verify_whir_bound(
                    &c_bin,
                    &s_fld,
                    &pid,
                    &tr.program,
                    &pubs,
                    &faddr,
                    &proof,
                    &opts,
                ),
                "the bound WHIR field half verifies"
            );
            t.elapsed().as_secs_f64()
        };
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&proof)
            .map(|b| b.len())
            .unwrap_or(0);
        eprintln!(
            "  v_fld (WHIR, bound): build {build_s:.1} s, prove {prove_s:.1} s, cells {cells}, verify {verify_s:.2} s, proof {bytes} bytes, tables {}",
            proof.table_num_vars.len()
        );
        eprintln!("  binding: c_bin {c_bin:?}, alpha {alpha:?}, s {s_fld:?}");
        return (c_bin, alpha, s_fld);
    }
    let (segments, chunks) = (traces.fvm.len(), traces.decode.len());
    let mem_at = segments + chunks;
    let c_fld = {
        let airs = Airs::new(&pid, &opts, segments, Vec::new(), false);
        let layout =
            stark::leaf_layout::table_leaf_layout(airs.refs()[mem_at], traces.mem.num_rows());
        let t = &traces.mem.main_table;
        let columns: Vec<Vec<FE>> = (0..traces.mem.num_cols())
            .map(|c| (0..traces.mem.num_rows()).map(|r| *t.get(r, c)).collect())
            .collect();
        commit_prep_with(&columns, &opts, layout)
    };
    let alpha = cksum::binding_alpha(&c_bin, &c_fld);
    let (trace, s_fld) = cksum::generate_trace(&faddr, &values, alpha, 8);
    let root = cksum::commitment(&faddr, &opts, 8);
    let air = || -> ExtraAir {
        Box::new(
            AirWithBuses::<_, _, NullBoundaryConstraintBuilder, _, _>::new(
                cksum::cols::NUM_COLUMNS,
                AuxiliaryTraceBuildData {
                    interactions: cksum::bus_interactions(),
                },
                &opts,
                1,
                cksum::CksumConstraints,
            )
            .with_name("FIELD_VM_CKSUM")
            .with_preprocessed(root, cksum::cols::NUM_PRECOMPUTED_COLS),
        )
    };
    let mut pubs = public_cells(run, &public);
    pubs.push((cksum::CKSUM_ADDR, s_fld));
    pubs.push((cksum::CKSUM_ALPHA_ADDR, alpha));
    pubs.sort_by_key(|c| c.0);
    let build_s = s.elapsed().as_secs_f64();
    let s = Instant::now();
    let proof = prove_traces_with(
        &pid,
        &pubs,
        &mut traces,
        vec![air()],
        &mut [trace],
        false,
        &opts,
    )
    .expect("the bound field half proves");
    let prove_s = s.elapsed().as_secs_f64();
    assert_eq!(
        proof.proofs[mem_at].lde_trace_main_merkle_root, c_fld,
        "MEM's root drawn before the table is the proof's"
    );
    let s = Instant::now();
    assert!(
        verify_with(&pid, &pubs, &proof, vec![air()], false, &opts),
        "the bound field half verifies"
    );
    let verify_s = s.elapsed().as_secs_f64();
    // Another s is refused.
    let mut forged = pubs.clone();
    for c in forged.iter_mut().filter(|c| c.0 == cksum::CKSUM_ADDR) {
        c.1 += FEE::one();
    }
    assert!(
        !verify_with(&pid, &forged, &proof, vec![air()], false, &opts),
        "another checksum is refused"
    );
    let (rows, total) = census(&proof);
    let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&proof)
        .map(|b| b.len())
        .unwrap_or(0);
    eprintln!(
        "  v_fld (B, bound): {} crossing cells, build {build_s:.1} s, prove {prove_s:.1} s, rows {rows}, cells {total}, verify {verify_s:.2} s, proof {bytes} bytes, tables {}",
        faddr.len(),
        tables_of(&proof)
    );
    eprintln!("  binding: c_bin {c_bin:?}, alpha {alpha:?}, s {s_fld:?}");
    (c_bin, alpha, s_fld)
}

/// Option B's hash half proved on the Lambda VM: the native guest over its
/// input through the block's WHIR prover at the production format, then
/// verified. Reports time, cells and the proof's size.
fn prove_vbin(elf: &[u8], input: &[u8], output: &[u8]) {
    use crate::block_whir::{prove_block_whir, verify_block_whir};
    let opts = super::proof::block_base_options();
    let format = BlockFormat::production();
    let s = Instant::now();
    let (proof, stamps) = prove_block_whir(elf, input, &opts, &format, &BlockOptions::production())
        .expect("v_bin proves");
    let prove_s = s.elapsed().as_secs_f64();
    assert_eq!(
        proof.public_output, output,
        "the proof states the guest's output"
    );
    let s = Instant::now();
    let ok = verify_block_whir(&proof, elf, &opts, &format).expect("v_bin verifies");
    let verify_s = s.elapsed().as_secs_f64();
    assert!(ok, "v_bin's proof verifies");
    let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&proof)
        .map(|b| b.len())
        .unwrap_or(0);
    eprintln!(
        "  v_bin proof: prove {prove_s:.1} s (execute {:.1}, build {:.1}, phase A {:.1}, phase B {:.1}), tables {}, cells {} ({:.2} G), verify {verify_s:.1} s, proof {bytes} bytes, counts {:?}",
        stamps.execute,
        stamps.build,
        stamps.phase_a,
        stamps.phase_b,
        stamps.tables,
        stamps.cells,
        stamps.cells as f64 / 1e9,
        proof.table_counts
    );
    eprintln!("  v_bin table heights (vars): {:?}", proof.table_num_vars);
}
