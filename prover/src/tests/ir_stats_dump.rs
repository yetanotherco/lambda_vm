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

    stats(&create_cpu_air(&opts), "CPU");
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
    stats(&create_commit_air(&opts), "COMMIT");
    stats(&create_page_air(&opts, 0x1000), "PAGE");
    stats(&create_register_air(&opts), "REGISTER");
    stats(&create_keccak_air(&opts), "KECCAK");
    stats(&create_keccak_rnd_air(&opts), "KECCAK_RND");
    stats(&create_keccak_rc_air(&opts), "KECCAK_RC");
    stats(&create_ecsm_air(&opts), "ECSM");
    stats(&create_ecdas_air(&opts), "ECDAS");
}
