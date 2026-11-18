//! Diagnostic dump (ignored by default): per-table constraint-IR and
//! device-lowering stats — node counts by dim, op mix, and the lowered slot
//! footprint that sizes the GPU kernel's per-thread scratch. Use it to gauge
//! how a lowering change moves the scratch working set (and therefore the
//! constraint kernel's cache behavior / thread-count headroom). Run with:
//! `cargo test -p lambda-vm-prover ir_stats_dump -- --ignored --nocapture`

use stark::constraint_ir::DeviceProgram;
use stark::constraint_ir::{ConstraintProgram, Dim, Op};
use stark::proof::options::GoldilocksCubicProofOptions;
use stark::traits::AIR;

use crate::tables::types::{GoldilocksExtension, GoldilocksField};
use crate::test_utils::*;

type Gl = GoldilocksField;
type Ext3 = GoldilocksExtension;

fn dump(label: &str, prog: &ConstraintProgram<Gl, Ext3>) {
    let n = prog.nodes.len();
    let mut base_nodes = 0usize;
    let mut ext_nodes = 0usize;
    let mut uniform = 0usize; // row-invariant: consts/challenges/alpha/offset
    let mut base_mul = 0usize;
    let mut ext_mul = 0usize;
    let mut base_addsub = 0usize;
    let mut ext_addsub = 0usize;
    let mut vars_main = 0usize;
    let mut vars_aux = 0usize;
    let mut embeds = 0usize;
    // mixed = ext-dim binop with at least one base-dim operand (implicit embed)
    let mut mixed_binops = 0usize;

    for (op, dim) in prog.nodes.iter().zip(prog.dims.iter()) {
        match dim {
            Dim::Base => base_nodes += 1,
            Dim::Ext => ext_nodes += 1,
        }
        match *op {
            Op::ConstBase(_)
            | Op::ConstExt(_)
            | Op::RapChallenge { .. }
            | Op::AlphaPow { .. }
            | Op::TableOffset => uniform += 1,
            Op::Var { main, .. } => {
                if main {
                    vars_main += 1
                } else {
                    vars_aux += 1
                }
            }
            Op::Mul(a, b) => {
                if *dim == Dim::Base {
                    base_mul += 1
                } else {
                    ext_mul += 1;
                    if prog.dims[a as usize] == Dim::Base || prog.dims[b as usize] == Dim::Base {
                        mixed_binops += 1;
                    }
                }
            }
            Op::Add(a, b) | Op::Sub(a, b) => {
                if *dim == Dim::Base {
                    base_addsub += 1
                } else {
                    ext_addsub += 1;
                    if prog.dims[a as usize] == Dim::Base || prog.dims[b as usize] == Dim::Base {
                        mixed_binops += 1;
                    }
                }
            }
            Op::Neg(_) => {}
            Op::Embed(_) => embeds += 1,
        }
    }

    // Each node's degree in the trace's columns (a variable 1, a uniform 0,
    // a product the sum, a sum the larger): the roots' largest is the program's
    // degree before its row selector.
    let mut deg = vec![0usize; n];
    for (i, op) in prog.nodes.iter().enumerate() {
        deg[i] = match *op {
            Op::Var { .. } => 1,
            Op::Mul(a, b) => deg[a as usize] + deg[b as usize],
            Op::Add(a, b) | Op::Sub(a, b) => deg[a as usize].max(deg[b as usize]),
            Op::Neg(a) | Op::Embed(a) => deg[a as usize],
            _ => 0,
        };
    }
    let degree = prog
        .roots
        .iter()
        .map(|&r| deg[r as usize])
        .max()
        .unwrap_or(0);
    let mut by_degree = [0usize; 8];
    for &r in &prog.roots {
        by_degree[deg[r as usize].min(7)] += 1;
    }
    println!("{label:12} degree={degree} roots by degree (0..7+)={by_degree:?}");

    // Lowered slot footprint: old scratch was nodes×24B per thread; new is
    // base_slots×8 + ext_slots×24.
    let dev = DeviceProgram::lower(prog);
    let old_bytes = n * 24;
    let new_bytes = dev.num_base_slots as usize * 8 + dev.num_ext_slots as usize * 24;

    println!(
        "{label:12} nodes={n:6} base={base_nodes:6} ({:4.1}%) ext={ext_nodes:5} uniform={uniform:4} \
         mul(b/e)={base_mul:5}/{ext_mul:4} addsub(b/e)={base_addsub:5}/{ext_addsub:4} \
         mixed={mixed_binops:4} var(m/a)={vars_main:4}/{vars_aux:3} embed={embeds:3} \
         roots={:4} num_base={:4} | slots(b/e)={}/{} scratch/thr {}B -> {}B ({:.1}x)",
        100.0 * base_nodes as f64 / n as f64,
        prog.roots.len(),
        prog.num_base,
        dev.num_base_slots,
        dev.num_ext_slots,
        old_bytes,
        new_bytes,
        old_bytes as f64 / new_bytes as f64,
    );
}

fn stats(air: &dyn AIR<Field = Gl, FieldExtension = Ext3, PublicInputs = ()>, label: &str) {
    dump(label, air.constraint_program());
}

#[test]
#[ignore = "analysis dump, not a test"]
fn dump_ir_stats() {
    let opts = GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 valid");

    stats(&create_cpu_air(&opts, 0), "CPU");
    stats(&create_bitwise_air(&opts), "BITWISE");
    stats(&create_lt_air(&opts), "LT");
    stats(&create_shift_air(&opts), "SHIFT");
    stats(&create_eq_air(&opts), "EQ");
    stats(&create_bytewise_air(&opts), "BYTEWISE");
    stats(&create_store_air(&opts), "STORE");
    stats(&create_cpu32_air(&opts), "CPU32");
    stats(&create_memw_air(&opts), "MEMW");
    stats(&create_memw_aligned_air(&opts), "MEMW_A");
    stats(&create_memw_register_air(&opts), "MEMW_R");
    stats(&create_load_air(&opts), "LOAD");
    stats(&create_decode_air(&opts), "DECODE");
    stats(&create_mul_air(&opts), "MUL");
    stats(&create_dvrm_air(&opts), "DVRM");
    stats(&create_branch_air(&opts), "BRANCH");
    stats(&create_halt_air(&opts), "HALT");
    stats(&create_page_air(&opts, 0x1000), "PAGE");
    stats(&create_register_air(&opts, 0), "REGISTER");
    stats(&create_keccak_air(&opts), "KECCAK");
    stats(&create_keccak_rnd_air(&opts), "KECCAK_RND");
    stats(&create_keccak_rc_air(&opts), "KECCAK_RC");
    stats(&create_ecsm_air(&opts), "ECSM");
    stats(&create_ecdas_air(&opts), "ECDAS");
    stats(&create_ecrec_air(&opts), "ECREC");
    stats(&create_ecrec_w_air(&opts), "ECREC_W");
    stats(&create_keccak_w_air(&opts), "KECCAK_W");
    stats(&create_copy_w_air(&opts), "COPY_W");
    stats(&create_is_wh_air(&opts), "IS_WH");
}

/// The fused zerocheck's slot file a thread, by schedule (a probe, not a test):
/// today's lowering (`lower_fused`, the step order as compiled), the same order
/// with each variable and constant read where it is used (no slot held for
/// them), and a greedy order that steps next whichever ready step frees the
/// most slots. The grid pass keeps every slot of every thread in global
/// memory, so a slot file past the card's L2 makes each step a trip to DRAM.
#[test]
#[ignore = "analysis dump, not a test"]
#[allow(clippy::needless_range_loop)]
fn dump_fused_slots_by_schedule() {
    use multilinear::fused::Constraints;
    use multilinear::program::Op as P;
    let opts = GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 valid");
    let report = |air: &dyn AIR<Field = Gl, FieldExtension = Ext3, PublicInputs = ()>,
                  label: &str| {
        let width = air.trace_layout().0;
        let layout = crate::multilinear_prove::layout_of(air, width, 12).expect("a layout");
        let shape = layout.shape();
        let c = Constraints::<Gl>::from_extension(
            &shape.steps_as_ops(),
            shape.root_steps(),
            shape.selector_of_root(),
            shape.degree(),
        )
        .expect("base constraints");
        let today = multilinear::gpu_fused::lower_fused(&c).map_or(0, |l| l.num_slots);
        let steps = c.steps();
        let n = steps.len();
        let leaf = |s: usize| matches!(steps[s], P::Fixed(_) | P::Var(_));
        let operands = |s: usize| -> Vec<usize> {
            match steps[s] {
                P::Neg(a) => vec![a as usize],
                P::Add(a, b) | P::Sub(a, b) | P::Mul(a, b) => vec![a as usize, b as usize],
                _ => Vec::new(),
            }
        };
        // Readers of each step, roots read once more by their ACC.
        let mut readers = vec![0usize; n];
        for s in 0..n {
            for o in operands(s) {
                readers[o] += 1;
            }
        }
        for &r in c.roots() {
            readers[r as usize] += 1;
        }
        // Max live over an order of the non-leaf steps (leaves read at use).
        let live_over = |order: &[usize]| -> usize {
            let mut left = readers.clone();
            let mut live = 0usize;
            let mut most = 0usize;
            let mut is_root = vec![0usize; n];
            for &r in c.roots() {
                is_root[r as usize] += 1;
            }
            for &s in order {
                live += 1;
                most = most.max(live);
                for o in operands(s) {
                    if !leaf(o) {
                        left[o] -= 1;
                        if left[o] == 0 {
                            live -= 1;
                        }
                    }
                }
                // Its ACC, right after it.
                if is_root[s] > 0 {
                    left[s] -= is_root[s];
                }
                if left[s] == 0 {
                    live -= 1;
                }
            }
            most
        };
        let compiled: Vec<usize> = (0..n).filter(|&s| !leaf(s)).collect();
        // Greedy: among ready steps, the one whose operands it ends most.
        let mut left = readers.clone();
        let mut users: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut waiting = vec![0usize; n];
        for s in 0..n {
            if leaf(s) {
                continue;
            }
            for o in operands(s) {
                if !leaf(o) {
                    users[o].push(s);
                    waiting[s] += 1;
                }
            }
        }
        let mut ready: std::collections::BTreeSet<usize> =
            (0..n).filter(|&s| !leaf(s) && waiting[s] == 0).collect();
        let mut greedy = Vec::with_capacity(compiled.len());
        while !ready.is_empty() {
            // Score: operands this step ends (a non-leaf with one reader left).
            let pick = *ready
                .iter()
                .take(256)
                .max_by_key(|&&s| {
                    let ends = operands(s)
                        .iter()
                        .filter(|&&o| !leaf(o) && left[o] == 1)
                        .count();
                    (ends, std::cmp::Reverse(s))
                })
                .expect("a ready step");
            ready.remove(&pick);
            greedy.push(pick);
            for o in operands(pick) {
                if !leaf(o) {
                    left[o] -= 1;
                }
            }
            for &u in &users[pick] {
                waiting[u] -= 1;
                if waiting[u] == 0 {
                    ready.insert(u);
                }
            }
        }
        assert_eq!(
            greedy.len(),
            compiled.len(),
            "{label}: every step scheduled"
        );
        // Depth first from each root in turn: a step's operands before it, the
        // one needing more first, each step once.
        let mut need = vec![0usize; n];
        for s in 0..n {
            if leaf(s) {
                continue;
            }
            let mut ns: Vec<usize> = operands(s)
                .iter()
                .filter(|&&o| !leaf(o))
                .map(|&o| need[o])
                .collect();
            ns.sort_unstable_by(|a, b| b.cmp(a));
            need[s] = ns
                .iter()
                .enumerate()
                .map(|(i, v)| v + i)
                .max()
                .unwrap_or(0)
                .max(1);
        }
        let mut emitted = vec![false; n];
        let mut dfs = Vec::with_capacity(compiled.len());
        for &root in c.roots() {
            let mut stack = vec![(root as usize, false)];
            while let Some((s, expanded)) = stack.pop() {
                if leaf(s) || emitted[s] {
                    continue;
                }
                if expanded {
                    emitted[s] = true;
                    dfs.push(s);
                    continue;
                }
                stack.push((s, true));
                let mut ops: Vec<usize> = operands(s)
                    .into_iter()
                    .filter(|&o| !leaf(o) && !emitted[o])
                    .collect();
                // The larger need is stepped first: pushed last.
                ops.sort_unstable_by_key(|&o| need[o]);
                for o in ops {
                    stack.push((o, false));
                }
            }
        }
        // Steps no root reaches (none expected) go last, in order.
        for s in 0..n {
            if !leaf(s) && !emitted[s] {
                dfs.push(s);
            }
        }
        println!(
            "FUSED SLOTS {label:<10} steps {n:6} roots {:5} · today {today:5} · leaves at use {:5} · greedy {:5} · depth first {:5}",
            c.roots().len(),
            live_over(&compiled),
            live_over(&greedy),
            live_over(&dfs),
        );
    };
    report(&create_cpu_air(&opts, 0), "CPU");
    report(&create_keccak_rnd_air(&opts), "KECCAK_RND");
    report(&create_keccak_w_air(&opts), "KECCAK_W");
    report(&create_ecdas_air(&opts), "ECDAS");
    report(&create_ecrec_w_air(&opts), "ECREC_W");
    report(&create_copy_w_air(&opts), "COPY_W");
}
