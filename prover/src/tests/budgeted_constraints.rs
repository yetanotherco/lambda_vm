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
    BudgetedProgram, Policy, eval_budgeted_row, lower_budgeted, lower_with_split, validate,
};
use stark::constraint_ir::{
    ConstraintProgram, DeviceProgram, Dim, Op, codegen, eval_device_program,
};
use stark::proof::options::GoldilocksCubicProofOptions;

use crate::tables::types::{GoldilocksExtension, GoldilocksField};
use crate::tests::compiled_constraints::production_programs;

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

/// Every distinct production program (blowups 2 and 4), keyed by the
/// structural key of its device lowering, with every label that has it.
fn distinct_programs() -> BTreeMap<u64, (Vec<String>, Program)> {
    let mut out: BTreeMap<u64, (Vec<String>, Program)> = BTreeMap::new();
    for blowup in [2, 4] {
        let opts = GoldilocksCubicProofOptions::with_blowup(blowup).expect("a valid blowup");
        for (label, program) in production_programs(&opts) {
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
    let programs = production_programs(&opts);
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
        for budget in [24u32, 32, 48, 64, 96, 128] {
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
        for budget in [24u32, 48, 128] {
            let Ok(bp) = lower_budgeted(&p, budget) else {
                continue;
            };
            let _ = writeln!(
                out,
                "P {}@{budget} {} {} {} {} {} {} {} {} {mc} {ac} {budget}",
                short(&labels).replace(' ', "_"),
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
