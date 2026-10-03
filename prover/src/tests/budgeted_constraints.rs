//! The register-budgeted lowering (`stark::constraint_ir::budgeted`) on every
//! production constraint program: it validates, it stays within its word
//! budget, and its host model computes the device walk's transition sum limb
//! for limb on random frames (full-range limbs, so non-canonical ones too).
//!
//! The census test prints, per program and budget, what the lowering costs
//! (steps against the program's interior nodes, column reads, words a row).

use std::collections::BTreeMap;

use math::field::element::FieldElement;
use stark::constraint_ir::budgeted::{
    BudgetedProgram, Policy, eval_budgeted_row, lower_budgeted, lower_with_split, specialize,
    validate,
};
use stark::constraint_ir::{
    ConstraintProgram, DeviceProgram, Dim, Op, codegen, eval_device_program,
};
use stark::proof::options::GoldilocksCubicProofOptions;

use crate::tables::types::{GoldilocksExtension, GoldilocksField};
use crate::tests::compiled_constraints::{compiled_option_sets, production_programs};

type Program = ConstraintProgram<GoldilocksField, GoldilocksExtension>;
type Fp = FieldElement<GoldilocksField>;
type Fp3 = FieldElement<GoldilocksExtension>;

struct SplitMix64(u64);
impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn e3(&mut self) -> [u64; 3] {
        [self.next_u64(), self.next_u64(), self.next_u64()]
    }
}

/// Every distinct production program (blowups 2 and 4, and the VM tables
/// under `LAMBDA_VM_ZF_LOGUP=k4`: the option sets the compiled kernels cover),
/// keyed by the structural key of its device lowering, with every label that
/// has it.
fn distinct_programs() -> BTreeMap<u64, (Vec<String>, Program)> {
    let mut out: BTreeMap<u64, (Vec<String>, Program)> = BTreeMap::new();
    for (opts, extras, suffix) in compiled_option_sets() {
        for (label, program) in production_programs(&opts, extras, suffix) {
            let key = codegen::structural_key(&DeviceProgram::lower(&program));
            let entry = out.entry(key).or_insert_with(|| (Vec::new(), program));
            if !entry.0.contains(&label) {
                entry.0.push(label);
            }
        }
    }
    out
}

/// A short label list: runs of `L2G[k]` collapse.
fn short(ls: &[String]) -> String {
    let l2g = ls.iter().filter(|l| l.starts_with("L2G[")).count();
    let mut out: Vec<String> = ls
        .iter()
        .filter(|l| !l.starts_with("L2G["))
        .cloned()
        .collect();
    if l2g > 0 {
        out.push(format!("L2G({l2g})"));
    }
    out.join(",")
}

/// Main columns, aux columns, RAP challenges and alpha powers a program reads.
fn footprint(p: &Program) -> (usize, usize, usize, usize) {
    let (mut main, mut aux, mut rap, mut alpha) = (1, 1, 1, 1);
    for op in &p.nodes {
        match *op {
            Op::Var {
                main: true, col, ..
            } => main = main.max(col as usize + 1),
            Op::Var {
                main: false, col, ..
            } => aux = aux.max(col as usize + 1),
            Op::RapChallenge { idx } => rap = rap.max(idx as usize + 1),
            Op::AlphaPow { idx } => alpha = alpha.max(idx as usize + 1),
            _ => {}
        }
    }
    (main, aux, rap, alpha)
}

fn encode_ext(x: &Fp3) -> [u64; 3] {
    let l = x.value();
    [*l[0].value(), *l[1].value(), *l[2].value()]
}

fn decode_ext(l: [u64; 3]) -> Fp3 {
    Fp3::from_raw([Fp::from_raw(l[0]), Fp::from_raw(l[1]), Fp::from_raw(l[2])])
}

/// Random frames through both walks: the budgeted host model against the
/// device walk's evals accumulated in root order (the composition kernel's
/// transition sum).
fn assert_parity(label: &str, p: &Program, bp: &BudgetedProgram, seed: u64, rows: usize) {
    let dev = DeviceProgram::lower(p);
    let (mc, ac, nr, na) = footprint(p);
    let n = p.roots.len();
    let mut rng = SplitMix64(seed);
    for row in 0..rows {
        let main: Vec<Vec<u64>> = (0..2)
            .map(|_| (0..mc).map(|_| rng.next_u64()).collect())
            .collect();
        let aux: Vec<Vec<[u64; 3]>> = (0..2)
            .map(|_| (0..ac).map(|_| rng.e3()).collect())
            .collect();
        let rap: Vec<[u64; 3]> = (0..nr).map(|_| rng.e3()).collect();
        let alpha: Vec<[u64; 3]> = (0..na).map(|_| rng.e3()).collect();
        let offset = rng.e3();
        let beta: Vec<[u64; 3]> = (0..n).map(|_| rng.e3()).collect();

        let mut be = vec![0u64; n];
        let mut ee = vec![[0u64; 3]; n];
        eval_device_program(&dev, &main, &aux, &rap, &alpha, offset, &mut be, &mut ee);
        let mut sum = Fp3::zero();
        for (c, &r) in p.roots.iter().enumerate() {
            let v = match p.dims[r as usize] {
                Dim::Base => Fp::from_raw(be[c]).to_extension::<GoldilocksExtension>(),
                Dim::Ext => decode_ext(ee[c]),
            };
            sum += decode_ext(beta[c]) * v;
        }
        let uni = bp
            .ext_uniform_table(&rap, &alpha, offset)
            .expect("the frame carries every uniform the program reads");
        let got = eval_budgeted_row(bp, &main, &aux, &uni, &beta);
        assert_eq!(
            got,
            encode_ext(&sum),
            "{label}: row {row} (seed {seed:#x}) differs from the device walk"
        );
    }
}

/// ★ Every production program lowers within 48 and 128 words a row, the
/// lowering validates, and its host model is the device walk's transition
/// sum, limb for limb, on two seeds.
#[test]
fn every_production_program_lowers_and_matches_the_device_walk() {
    let programs = distinct_programs();
    assert!(programs.len() >= 30, "{} distinct programs", programs.len());
    for (key, (labels, p)) in &programs {
        let label = short(labels);
        for budget in [48u32, 128] {
            let bp = lower_budgeted(p, budget)
                .unwrap_or_else(|e| panic!("{label}: no lowering within {budget} words: {e:?}"));
            assert!(bp.num_words <= budget, "{label}: {} words", bp.num_words);
            validate(p, &bp).unwrap_or_else(|e| panic!("{label} at {budget}: {e}"));
            for seed in [1u64, 0xDEAD_BEEF] {
                assert_parity(&label, p, &bp, seed ^ key ^ budget as u64, 3);
            }
        }
    }
}

/// The lowering's cost on the two large interpreted programs. D-INTERP §3.4's
/// census chunked the roots greedily in demand order with nothing shared
/// across chunks: KECCAK_RND at 48 words took 13,762 interior steps (1.06×
/// its 13,001 interior nodes), ECDAS at 128 words 29,864 (1.25× its 23,980).
/// This lowering keeps shared values across roots while the budget allows:
/// ECDAS lands well under the census, KECCAK_RND within 1 % of it.
#[test]
fn the_large_programs_cost_about_the_census_or_less() {
    let opts = GoldilocksCubicProofOptions::with_blowup(4).expect("blowup 4");
    let programs = production_programs(&opts, true, "");
    let find = |name: &str| {
        programs
            .iter()
            .find(|(l, _)| l == name)
            .map(|(_, p)| p.clone())
            .unwrap_or_else(|| panic!("no {name}"))
    };
    for (name, budget, bound) in [("KECCAK_RND", 48u32, 13_900usize), ("ECDAS", 128, 29_864)] {
        let p = find(name);
        let bp = lower_budgeted(&p, budget).expect("lowers");
        println!(
            "BUDGETED {name} at {budget} words: {} interior steps ({} distinct nodes), bound {bound}",
            bp.stats.compute_steps, bp.stats.distinct_nodes
        );
        assert!(
            bp.stats.compute_steps <= bound,
            "{name}: {} steps at {budget} words, bound {bound}",
            bp.stats.compute_steps
        );
    }
}

/// Lowering every program: per budget, the steps over the interior nodes,
/// the column reads over the program's distinct column leaves, and the words
/// a row the program ends up using.
#[test]
#[ignore = "a census, not a test: prints the lowering's cost per program"]
fn budgeted_census() {
    for (key, (labels, p)) in distinct_programs() {
        let dev = DeviceProgram::lower(&p);
        let leaves = p
            .nodes
            .iter()
            .filter(|o| matches!(o, Op::Var { .. }))
            .count();
        let unbounded =
            lower_with_split(&p, 4096, 4096, Policy::Furthest).expect("lowers unbounded");
        let mut line = format!(
            "CENSUS {:<34} key {key:016x} nodes {:6} roots {:4} today-words {:5} interior {:6} demand-words {:4}",
            short(&labels),
            dev.nodes.len(),
            p.roots.len(),
            dev.num_base_slots + 3 * dev.num_ext_slots,
            unbounded.stats.distinct_nodes,
            unbounded.num_words,
        );
        for budget in [16u32, 24, 32, 48, 64, 96, 128] {
            match lower_budgeted(&p, budget) {
                Ok(bp) => {
                    line.push_str(&format!(
                        " | B{budget}: w{} x{:.2} r{:.2}",
                        bp.num_words,
                        bp.stats.compute_steps as f64
                            / unbounded.stats.distinct_nodes.max(1) as f64,
                        bp.stats.col_reads as f64 / leaves.max(1) as f64,
                    ));
                }
                Err(_) => line.push_str(&format!(" | B{budget}: -")),
            }
        }
        println!("{line}");
    }
}

/// Writes every distinct production program, lowered at 24, 48 and 128 words,
/// for the host parity check of the device source
/// (`crypto/math-cuda/kernels/tools/si_host_check.cpp`, which builds
/// `constraint_si.cu` and `constraint_interp.cu` as host C++ and compares
/// them), to the directory `LAMBDA_VM_SI_DUMP` names: one `P` line per
/// lowering, its device program's `N` nodes, `R` roots, `C` base and `X` ext
/// constants, then its budgeted `S` steps.
#[test]
#[ignore = "a dump for the host parity check, not a test"]
fn dump_budgeted_programs() {
    use std::fmt::Write as _;
    let dir = std::env::var("LAMBDA_VM_SI_DUMP").expect("LAMBDA_VM_SI_DUMP=<dir>");
    let mut out = String::new();
    for (labels, p) in distinct_programs().into_values() {
        let dev = DeviceProgram::lower(&p);
        let (mc, ac, _, _) = footprint(&p);
        for (budget, fast) in [
            (24u32, false),
            (48, false),
            (128, false),
            (24, true),
            (48, true),
            (128, true),
        ] {
            let Ok(bp) = lower_budgeted(&p, budget) else {
                continue;
            };
            let bp = if fast { specialize(&bp) } else { bp };
            let _ = writeln!(
                out,
                "P {}@{budget}{} {} {} {} {} {} {} {} {} {mc} {ac} {budget}",
                short(&labels).replace(' ', "_"),
                if fast { "/fast" } else { "" },
                dev.nodes.len(),
                dev.roots.len(),
                dev.num_base_slots,
                dev.num_ext_slots,
                bp.steps.len(),
                bp.num_words,
                bp.num_rap,
                bp.num_alpha,
            );
            for n in &dev.nodes {
                let _ = writeln!(out, "N {} {} {} {}", n.op, n.a, n.b, n.res);
            }
            let roots: Vec<String> = dev.roots.iter().map(|r| r.to_string()).collect();
            let _ = writeln!(out, "R {}", roots.join(" "));
            let consts: Vec<String> = dev.base_consts.iter().map(|c| c.to_string()).collect();
            let _ = writeln!(out, "C {}", consts.join(" "));
            let ext: Vec<String> = dev
                .ext_consts
                .iter()
                .flat_map(|e| e.iter().map(|c| c.to_string()))
                .collect();
            let _ = writeln!(out, "X {}", ext.join(" "));
            for s in &bp.steps {
                let _ = writeln!(out, "S {} {} {} {}", s.op, s.a, s.b, s.dst);
            }
        }
    }
    std::fs::write(std::path::Path::new(&dir).join("si_programs.txt"), out)
        .expect("write the dump");
}

/// On the device: the bounded-slot interpreter against the slot-file
/// interpreter, a proof proved both ways, and the kernel-time benchmark.
#[cfg(feature = "cuda")]
mod device {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::Ordering::SeqCst;

    use math_cuda::constraint_interp::{
        SiConfig, SiStore, set_composition_timing, si_blocks_per_sm, si_device_report,
        take_composition_device_ms,
    };
    use math_cuda::device::backend;
    use math_cuda::lde::{GpuLdeBase, GpuLdeExt3};
    use stark::constraint_ir::gpu_interp::{
        CompositionInputs, GPU_COMPOSITION_COMPILED_CALLS, GPU_COMPOSITION_SI_CALLS,
        GpuComposition, SI_MUTATE_ACC, SiMode, SiTuning, override_compiled_constraints,
        override_interp_si, try_eval_composition_gpu,
    };

    use crate::tests::compiled_constraints::{
        BYTES_PROGRAM, FixedTraces, bytes_options, proof_bytes,
    };

    /// Tests that flip the process-wide overrides take this first.
    static OVERRIDE: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn shared(rows: u32, block: u32, budget: u32) -> SiTuning {
        SiTuning {
            cfg: SiConfig {
                store: SiStore::Shared,
                rows_per_thread: rows,
                block,
                staged: false,
                prefetch: false,
            },
            budget,
            fast: true,
            auto: false,
        }
    }

    fn local(block: u32, budget: u32) -> SiTuning {
        SiTuning {
            cfg: SiConfig {
                store: SiStore::Local,
                rows_per_thread: 1,
                block,
                staged: false,
                prefetch: false,
            },
            budget,
            fast: true,
            auto: false,
        }
    }

    fn generic(t: SiTuning) -> SiTuning {
        SiTuning { fast: false, ..t }
    }

    fn staged(t: SiTuning) -> SiTuning {
        SiTuning {
            cfg: SiConfig {
                staged: true,
                ..t.cfg
            },
            ..t
        }
    }

    fn prefetched(t: SiTuning) -> SiTuning {
        SiTuning {
            cfg: SiConfig {
                staged: true,
                prefetch: true,
                ..t.cfg
            },
            ..t
        }
    }

    fn name(t: &SiTuning) -> String {
        if t.auto {
            return "auto".into();
        }
        let s = match t.cfg.store {
            SiStore::Shared => "smem",
            SiStore::Local => "local",
        };
        let f = match (t.fast, t.cfg.staged, t.cfg.prefetch) {
            (true, _, true) => "/pp",
            (false, _, true) => "/gen/pp",
            (true, true, false) => "/ps",
            (false, true, false) => "/gen/ps",
            (true, false, false) => "",
            (false, false, false) => "/gen",
        };
        format!(
            "{s}/r{}/t{}/b{}{f}",
            t.cfg.rows_per_thread, t.cfg.block, t.budget
        )
    }

    /// Random device-resident inputs for one program over `rows` LDE rows.
    struct Inputs {
        main: GpuLdeBase,
        aux: GpuLdeExt3,
        rap: Vec<Fp3>,
        alpha: Vec<Fp3>,
        offset: Fp3,
        beta: Vec<Fp3>,
        z_inv: Vec<Fp>,
        b_col: Vec<usize>,
        b_is_aux: Vec<bool>,
        b_value: Vec<Fp3>,
        b_beta: Vec<Fp3>,
        b_z_inv: Vec<Arc<Vec<Fp>>>,
    }

    fn inputs(p: &Program, rows: usize, seed: u64) -> Inputs {
        let (main_cols, aux_cols, rap_len, alpha_len) = footprint(p);
        let mut rng = SplitMix64(seed);
        let fp3 = |rng: &mut SplitMix64| {
            Fp3::from_raw([
                Fp::from_raw(rng.next_u64()),
                Fp::from_raw(rng.next_u64()),
                Fp::from_raw(rng.next_u64()),
            ])
        };
        let base: Vec<u64> = (0..main_cols * rows).map(|_| rng.next_u64()).collect();
        let ext: Vec<u64> = (0..aux_cols * 3 * rows).map(|_| rng.next_u64()).collect();
        let be = backend().expect("cuda backend");
        let stream = be.next_stream();
        let main_buf = stream.clone_htod(&base).expect("upload main");
        let aux_buf = stream.clone_htod(&ext).expect("upload aux");
        stream.synchronize().expect("sync");
        Inputs {
            main: GpuLdeBase {
                ready: None,
                buf: Arc::new(main_buf),
                m: main_cols,
                lde_size: rows,
                tree: None,
                trace_dev: None,
                trace_rows: 0,
            },
            aux: GpuLdeExt3 {
                ready: None,
                buf: Arc::new(aux_buf),
                m: aux_cols,
                lde_size: rows,
                tree: None,
            },
            rap: (0..rap_len).map(|_| fp3(&mut rng)).collect(),
            alpha: (0..alpha_len).map(|_| fp3(&mut rng)).collect(),
            offset: fp3(&mut rng),
            beta: (0..p.roots.len()).map(|_| fp3(&mut rng)).collect(),
            z_inv: (0..4).map(|_| Fp::from_raw(rng.next_u64())).collect(),
            b_col: vec![0, aux_cols - 1],
            b_is_aux: vec![false, true],
            b_value: (0..2).map(|_| fp3(&mut rng)).collect(),
            b_beta: (0..2).map(|_| fp3(&mut rng)).collect(),
            b_z_inv: (0..2)
                .map(|_| Arc::new((0..rows).map(|_| Fp::from_raw(rng.next_u64())).collect()))
                .collect(),
        }
    }

    /// One composition under the current overrides: `H` host-drained, or
    /// (`keep`) kept on the device and waited for.
    fn composition(p: &Program, x: &Inputs, rows: usize, keep: bool) -> Option<Vec<u64>> {
        let accum = CompositionInputs {
            beta_trans: &x.beta,
            z_inv: &x.z_inv,
            b_col: &x.b_col,
            b_is_aux: &x.b_is_aux,
            b_value: &x.b_value,
            b_beta: &x.b_beta,
            b_z_inv: &x.b_z_inv,
        };
        match try_eval_composition_gpu(
            p, &x.main, &x.aux, &x.rap, &x.alpha, &x.offset, 4, rows, &accum, keep,
        ) {
            Some(GpuComposition::Host(h)) => Some(h),
            Some(GpuComposition::Dev(h)) => {
                h.synchronize().expect("synchronize");
                None
            }
            None => panic!("the GPU composition path must engage"),
        }
    }

    /// The shapes every program is checked under: both stores, both row
    /// counts, a small and a large budget.
    fn parity_shapes() -> Vec<SiTuning> {
        vec![
            shared(1, 128, 48),
            generic(shared(1, 128, 48)),
            shared(2, 64, 32),
            generic(shared(2, 64, 32)),
            shared(1, 64, 128),
            local(128, 48),
            local(64, 24),
            staged(local(128, 48)),
            staged(shared(1, 64, 48)),
            prefetched(local(128, 48)),
            prefetched(shared(1, 64, 24)),
            SiTuning::default(),
        ]
    }

    /// ★ The bounded-slot interpreter computes the slot-file interpreter's
    /// `H`, limb for limb, for every production program, under every parity
    /// shape, on two seeds of random full-range LDE columns and uniforms.
    #[test]
    fn every_si_shape_computes_the_interpreters_h() {
        let _guard = OVERRIDE.lock().unwrap_or_else(|e| e.into_inner());
        const ROWS: usize = 4096;
        let mut compared = 0;
        for (key, (labels, p)) in distinct_programs() {
            let label = short(&labels);
            for seed in [1u64, 0xDEAD_BEEF] {
                let x = inputs(&p, ROWS, seed ^ key);
                override_compiled_constraints(Some(false));
                override_interp_si(Some((SiMode::Off, SiTuning::default())));
                let reference = composition(&p, &x, ROWS, false).expect("host H");
                for t in parity_shapes() {
                    if lower_budgeted(&p, t.budget).is_err() {
                        continue;
                    }
                    override_interp_si(Some((SiMode::All, t)));
                    let before = GPU_COMPOSITION_SI_CALLS.load(SeqCst);
                    let got = composition(&p, &x, ROWS, false).expect("host H");
                    let ran = GPU_COMPOSITION_SI_CALLS.load(SeqCst) - before;
                    assert_eq!(ran, 1, "{label} {}: the bounded-slot kernel ran", name(&t));
                    assert!(
                        got == reference,
                        "{label} (key {key:016x}, seed {seed:#x}, {}): H differs from the interpreter's",
                        name(&t)
                    );
                    compared += 1;
                }
            }
        }
        override_interp_si(None);
        override_compiled_constraints(None);
        println!("SI PARITY: {compared} program-shape-seed runs equal");
        assert!(compared >= 300, "only {compared} runs compared");
    }

    /// ★ A proof made with the bounded-slot interpreter on every program is
    /// byte for byte the proof made with it off (`LAMBDA_VM_GPU_INTERP_SI=0`:
    /// the compiled kernels and the slot-file interpreter), from one set of
    /// traces at grinding 0, after the control (two such proofs equal). Then the
    /// mutation
    /// control: with every accumulation of every budgeted program taking the
    /// next root's coefficient, the proof must change and fail to verify, or
    /// the prover must refuse it. (Moving only the first root's coefficient
    /// was not load-bearing: FAST 780 saw the proof unchanged, most likely
    /// because those first roots are identically zero on this program's LDE.)
    #[test]
    #[ignore = "requires a GPU: proves a program four times"]
    fn the_si_interpreter_proves_the_same_bytes() {
        let _guard = OVERRIDE.lock().unwrap_or_else(|e| e.into_inner());
        let fixed = FixedTraces::build(BYTES_PROGRAM);
        let opts = bytes_options();
        let prove = |mode: SiMode| {
            override_interp_si(Some((mode, SiTuning::default())));
            let proof = fixed.prove(&opts);
            override_interp_si(None);
            proof
        };
        let control = proof_bytes(&prove(SiMode::Off).expect("prove"));
        let default = proof_bytes(&prove(SiMode::Off).expect("prove"));
        assert!(
            control == default,
            "the control: two proofs with the bounded-slot interpreter off differ"
        );
        println!(
            "SI BYTES: the control, two proofs of one set of traces with it off: {} bytes each, equal",
            default.len()
        );
        let before = GPU_COMPOSITION_SI_CALLS.load(SeqCst);
        let compiled_before = GPU_COMPOSITION_COMPILED_CALLS.load(SeqCst);
        let si = prove(SiMode::All).expect("prove");
        let ran = GPU_COMPOSITION_SI_CALLS.load(SeqCst) - before;
        assert!(
            ran > 0,
            "no composition ran on the bounded-slot interpreter"
        );
        assert_eq!(
            GPU_COMPOSITION_COMPILED_CALLS.load(SeqCst),
            compiled_before,
            "a compiled kernel ran under `all`"
        );
        assert!(
            proof_bytes(&si) == default,
            "the bounded-slot interpreter changed the proof"
        );
        assert!(fixed.verifies(&si, &opts), "the proof does not verify");
        println!("SI BYTES: the bounded-slot proof ({ran} compositions) equals it and verifies");

        SI_MUTATE_ACC.store(true, SeqCst);
        let before = GPU_COMPOSITION_SI_CALLS.load(SeqCst);
        let mutated = prove(SiMode::All);
        SI_MUTATE_ACC.store(false, SeqCst);
        let ran = GPU_COMPOSITION_SI_CALLS.load(SeqCst) - before;
        assert!(ran > 0, "the mutant never ran");
        match mutated {
            Err(e) => {
                println!("SI BYTES: the mutant ({ran} compositions): the prover refused: {e:?}")
            }
            Ok(proof) => {
                assert!(
                    proof_bytes(&proof) != default,
                    "the mutant left the proof unchanged"
                );
                assert!(
                    !fixed.verifies(&proof, &opts),
                    "a proof made with the mutant verifies"
                );
                println!(
                    "SI BYTES: the mutant ({ran} compositions): the proof changed and does not verify"
                );
            }
        }
    }

    /// Kernel milliseconds (CUDA events, median of 5 after one warm-up) of one
    /// program under the current overrides.
    fn kernel_ms(p: &Program, x: &Inputs, rows: usize) -> Option<f64> {
        let mut ms = Vec::new();
        for i in 0..6 {
            let _ = take_composition_device_ms();
            composition(p, x, rows, true);
            let t = take_composition_device_ms()?;
            if i > 0 {
                ms.push(t);
            }
        }
        ms.sort_by(|a, b| a.total_cmp(b));
        Some(ms[ms.len() / 2])
    }

    /// The full sweep of shapes (S1).
    fn sweep_shapes() -> Vec<SiTuning> {
        let mut v = Vec::new();
        // Shared-memory slots: occupancy is the budget's (FAST 781), so small
        // budgets, plain, staged and staged with the prefetch.
        for block in [64, 128] {
            for budget in [16, 24, 32, 48] {
                v.push(shared(1, block, budget));
                v.push(staged(shared(1, block, budget)));
                v.push(prefetched(shared(1, block, budget)));
            }
        }
        for budget in [64, 96, 128] {
            v.push(shared(1, 64, budget));
        }
        v.push(shared(2, 64, 24));
        // Local-array slots.
        for block in [64, 128, 256] {
            for budget in [32, 48, 64, 128] {
                v.push(local(block, budget));
            }
        }
        for budget in [32, 48, 128] {
            v.push(staged(local(128, budget)));
            v.push(prefetched(local(128, budget)));
        }
        // The generic opcodes at three shapes: what the specialized ones gain.
        v.push(generic(shared(1, 64, 24)));
        v.push(generic(prefetched(shared(1, 64, 24))));
        v.push(generic(local(128, 48)));
        v
    }

    /// The reduced sweep every program gets (S2b's sizing).
    fn short_shapes() -> Vec<SiTuning> {
        vec![
            shared(1, 128, 32),
            shared(1, 128, 48),
            shared(1, 128, 64),
            shared(1, 64, 48),
            shared(1, 64, 96),
            shared(2, 64, 32),
            shared(2, 64, 48),
            local(128, 32),
            local(128, 48),
            local(128, 64),
            local(256, 48),
            generic(shared(1, 128, 48)),
            staged(local(128, 48)),
            prefetched(local(128, 48)),
            shared(1, 64, 16),
            prefetched(shared(1, 64, 16)),
            prefetched(shared(1, 64, 24)),
            prefetched(shared(1, 128, 32)),
        ]
    }

    /// The LDE rows a program is timed at: the interpreted programs at the
    /// sizes D-INTERP's gate names, every other program at 2^20; halved until
    /// its random LDE columns fit in 6 GiB (the wide LFM chips; FAST 781 ran
    /// out of device memory on LFM_BLAKE3 at 2^20).
    fn bench_rows(labels: &[String], p: &Program) -> usize {
        let has = |n: &str| labels.iter().any(|l| l == n);
        let mut rows = if has("KECCAK_RND") || has("KECCAK") || has("ECSM") {
            1 << 18
        } else if has("ECDAS") {
            1 << 19
        } else {
            1 << 20
        };
        let (mc, ac, _, _) = footprint(p);
        let per_row = mc * 8 + ac * 24 + 16;
        while rows > 4096 && rows * per_row > 6 << 30 {
            rows /= 2;
        }
        rows
    }

    /// S1: kernel time per program, the slot-file interpreter and the compiled
    /// kernel (when the program has one) against the bounded-slot interpreter
    /// under every shape (the full sweep for the programs that carry the
    /// composition time, a reduced one for the rest). One `SI BENCH` line per
    /// program and shape, one `SI BEST` line per program.
    #[test]
    #[ignore = "a benchmark: run on the box"]
    fn bench_si_against_the_interpreter() {
        let _guard = OVERRIDE.lock().unwrap_or_else(|e| e.into_inner());
        println!("SI DEVICE: {}", si_device_report().expect("device report"));
        set_composition_timing(true);
        let full: &[&str] = &[
            "KECCAK_RND",
            "ECDAS",
            "KECCAK",
            "ECSM",
            "CPU",
            "MEMW_R",
            "MEMW_A",
            "LT",
            "LFM LFM_HASH",
        ];
        for (key, (labels, p)) in distinct_programs() {
            let label = short(&labels);
            if labels.iter().any(|l| l.starts_with("L2G[")) && !labels.iter().any(|l| l == "L2G[1]")
            {
                continue; // one L2G program is enough
            }
            let rows = bench_rows(&labels, &p);
            let x = inputs(&p, rows, key);
            override_interp_si(Some((SiMode::Off, SiTuning::default())));
            override_compiled_constraints(Some(false));
            let interp = kernel_ms(&p, &x, rows).expect("interpreter timed");
            override_compiled_constraints(Some(true));
            let compiled = if math_cuda::constraint_interp::compiled_composition_kernel(
                codegen::structural_key(&DeviceProgram::lower(&p)),
            )
            .is_some()
            {
                kernel_ms(&p, &x, rows)
            } else {
                None
            };
            override_compiled_constraints(Some(false));
            let shapes = if labels.iter().any(|l| full.contains(&l.as_str())) {
                sweep_shapes()
            } else {
                short_shapes()
            };
            let nodes = DeviceProgram::lower(&p).nodes.len();
            let mut best: Option<(f64, String)> = None;
            for t in shapes {
                let Ok(bp) = lower_budgeted(&p, t.budget) else {
                    continue;
                };
                if t.cfg.kernel(bp.num_words).is_none() {
                    continue;
                }
                let occ = si_blocks_per_sm(t.cfg, bp.num_words).unwrap_or(0);
                if occ == 0 {
                    println!(
                        "SI BENCH {label:<34} {:<20} words {:3} does not fit",
                        name(&t),
                        bp.num_words
                    );
                    continue;
                }
                override_interp_si(Some((SiMode::All, t)));
                let before = GPU_COMPOSITION_SI_CALLS.load(SeqCst);
                let Some(ms) = kernel_ms(&p, &x, rows) else {
                    println!("SI BENCH {label:<34} {:<20} failed", name(&t));
                    continue;
                };
                assert!(
                    GPU_COMPOSITION_SI_CALLS.load(SeqCst) > before,
                    "{label}: SI ran"
                );
                println!(
                    "SI BENCH {label:<34} {:<20} rows {rows:8} nodes {nodes:6} steps {:6} words {:3} \
                     blocks/SM {occ:2} ms {ms:9.3} vs interp {interp:9.3} compiled {} ps/node-row {:.2}",
                    name(&t),
                    bp.steps.len(),
                    bp.num_words,
                    compiled
                        .map(|c| format!("{c:9.3}"))
                        .unwrap_or_else(|| "        -".into()),
                    ms * 1e9 / (nodes as f64 * rows as f64),
                );
                if best.as_ref().is_none_or(|(b, _)| ms < *b) {
                    best = Some((ms, name(&t)));
                }
            }
            if let Some((ms, shape)) = best {
                println!(
                    "SI BEST {label:<34} rows {rows:8} nodes {nodes:6} best {shape:<20} {ms:9.3} ms · \
                     interp {interp:9.3} ({:.2}x) · compiled {}",
                    interp / ms,
                    compiled
                        .map(|c| format!("{c:9.3} (si/compiled {:.2})", ms / c))
                        .unwrap_or_else(|| "-".into()),
                );
            }
        }
        override_interp_si(None);
        override_compiled_constraints(None);
        set_composition_timing(false);
    }
}

/// The opcode × operand-kind mix of the lowered programs (which handler
/// shapes the kernel runs most).
#[test]
#[ignore = "a census, not a test"]
fn budgeted_opcode_mix() {
    use stark::constraint_ir::budgeted::{
        SI_ACC_B, SI_ACC_E, SI_BNEG, SI_EMBED, SI_ENEG, SIK_SHIFT,
    };
    let opts = GoldilocksCubicProofOptions::with_blowup(4).expect("blowup 4");
    let kinds = ["bslot", "eslot", "main", "aux", "bconst", "euni", "?", "?"];
    for (label, p) in production_programs(&opts, true, "") {
        if !["CPU", "KECCAK_RND", "ECDAS", "MEMW_R", "LT", "MEMW_A"].contains(&label.as_str()) {
            continue;
        }
        let bp = lower_budgeted(&p, 48).expect("lowers");
        let mut mix: BTreeMap<String, usize> = BTreeMap::new();
        for s in &bp.steps {
            let ka = kinds[(s.a >> SIK_SHIFT) as usize];
            let kb = if matches!(s.op, SI_BNEG | SI_ENEG | SI_EMBED | SI_ACC_B | SI_ACC_E) {
                "-"
            } else {
                kinds[(s.b >> SIK_SHIFT) as usize]
            };
            *mix.entry(format!("op{:02} {ka}/{kb}", s.op)).or_default() += 1;
        }
        let mut v: Vec<(String, usize)> = mix.into_iter().collect();
        v.sort_by_key(|x| std::cmp::Reverse(x.1));
        let total = bp.steps.len();
        let line: Vec<String> = v
            .iter()
            .take(14)
            .map(|(k, n)| format!("{k} {:.1}%", 100.0 * *n as f64 / total as f64))
            .collect();
        println!("MIX {label} ({total} steps): {}", line.join(" · "));
    }
}
