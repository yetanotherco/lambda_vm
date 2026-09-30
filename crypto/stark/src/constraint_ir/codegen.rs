//! Straight-line CUDA for a lowered [`DeviceProgram`]: the compiled twin of
//! `constraint_composition_kernel` (`crypto/math-cuda/kernels/constraint_interp.cu`).
//!
//! The interpreter walks the flat node list for every LDE row, reading each
//! operand from and writing each result to a per-thread slot file in global
//! memory. The kernel generated here does the same walk unrolled at build time:
//! every node becomes one statement over local variables (registers, spilled by
//! the compiler where the live set is larger), so no node is fetched or
//! decoded and no slot file exists.
//!
//! ⛔ BIT-IDENTICAL BY CONSTRUCTION, and only by construction. Goldilocks values
//! on the device are NON-canonical `u64`s (`goldilocks.cuh`), so a value's bits
//! depend on the exact operations that produced it. Every statement emitted
//! here is the call `eval_program_row` makes for that node's op, operand kinds
//! and result class — including the mixed base/ext shortcuts and the literal
//! `sub(0, ·)` — and it reads the uniform operands from the same tables. The
//! transition sum adds root `c` only after roots `0..c`, with the same
//! `mul`/`mul_base` choice, and the boundary loop is the interpreter's own. A
//! change to the interpreter's op semantics must be mirrored here; the device
//! parity test compares the two kernels limb for limb.
//!
//! A kernel is keyed by [`structural_key`]: everything the generated code
//! depends on (nodes, roots, slot counts, table sizes) and nothing it reads at
//! run time (constant values, challenges). A program whose key has no kernel
//! runs the interpreter.

use std::fmt::Write as _;

use super::device::{
    DeviceProgram, OP_ADD, OP_ALPHA_POW, OP_CONST_BASE, OP_CONST_EXT, OP_EMBED, OP_MUL, OP_NEG,
    OP_RAP_CHALLENGE, OP_SUB, OP_TABLE_OFFSET, OP_VAR, OPK_ALPHA, OPK_BASE_CONST, OPK_BASE_SLOT,
    OPK_EXT_CONST, OPK_EXT_SLOT, OPK_OFFSET, OPK_PAYLOAD_MASK, OPK_RAP, OPK_SHIFT, RES_EXT_BIT,
    unpack_var,
};

/// Bumped whenever the generated code's meaning changes, so a kernel emitted by
/// an older generator can never match a program by key.
const CODEGEN_VERSION: u32 = 1;

/// Threads per block of every generated kernel (`CCOMP_BLOCK` in the source).
pub const COMPILED_BLOCK_DIM: u32 = 128;

/// The largest program the generator emits, in nodes. Above it the program
/// stays on the interpreter: the straight-line kernel's compile time and
/// register spills grow with the node count.
pub const MAX_COMPILED_NODES: usize = 4200;

/// The largest program the generator emits, in estimated device instructions
/// per row ([`estimated_row_cost`]).
pub const MAX_COMPILED_ROW_COST: u64 = 120_000;

/// A stable 64-bit key over everything the generated kernel depends on: the
/// node list, the root slots, the slot counts, `num_base` and the sizes of the
/// two constant tables. Constant VALUES and the per-proof uniforms are read at
/// run time, so they are not part of it. FNV-1a, so the key is the same on
/// every host and in every process.
pub fn structural_key(dev: &DeviceProgram) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut word = |w: u32| {
        for byte in w.to_le_bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    word(CODEGEN_VERSION);
    word(dev.nodes.len() as u32);
    for n in &dev.nodes {
        word(n.op);
        word(n.a);
        word(n.b);
        word(n.res);
    }
    word(dev.roots.len() as u32);
    for &r in &dev.roots {
        word(r);
    }
    word(dev.num_base);
    word(dev.num_base_slots);
    word(dev.num_ext_slots);
    word(dev.base_consts.len() as u32);
    word(dev.ext_consts.len() as u32);
    h
}

/// The generated kernel's name for a program key.
pub fn kernel_name(key: u64) -> String {
    format!("ccomp_{key:016x}")
}

/// The name of the program's [`mutant_composition_kernel`].
pub fn mutant_kernel_name(key: u64) -> String {
    format!("{}_mutant", kernel_name(key))
}

fn kind(enc: u32) -> u32 {
    enc >> OPK_SHIFT
}

fn payload(enc: u32) -> u32 {
    enc & OPK_PAYLOAD_MASK
}

fn is_base_kind(enc: u32) -> bool {
    matches!(kind(enc), OPK_BASE_SLOT | OPK_BASE_CONST)
}

/// A rough per-row instruction count, to keep very large programs on the
/// interpreter: ext3 × ext3 multiplies dominate.
pub fn estimated_row_cost(dev: &DeviceProgram) -> u64 {
    let mut cost = 0u64;
    for n in &dev.nodes {
        let ext = n.res & RES_EXT_BIT != 0;
        cost += match n.op {
            OP_MUL if !ext => 20,
            OP_MUL if is_base_kind(n.a) || is_base_kind(n.b) => 60,
            OP_MUL => 175,
            OP_ADD | OP_SUB | OP_NEG if !ext => 8,
            OP_ADD | OP_SUB | OP_NEG => 24,
            _ => 4,
        };
    }
    cost + 200 * dev.roots.len() as u64
}

/// Whether the generator emits a kernel for this program.
pub fn worth_compiling(dev: &DeviceProgram) -> bool {
    dev.nodes.len() <= MAX_COMPILED_NODES && estimated_row_cost(dev) <= MAX_COMPILED_ROW_COST
}

/// Why a program cannot be emitted as a straight-line kernel. The interpreter
/// runs it instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CodegenError {
    /// An operand encoding the interpreter would read as something else.
    BadOperand { node: usize, enc: u32 },
    /// A node whose result class disagrees with its op.
    ClassMismatch { node: usize },
    /// An op tag the interpreter does not know (it would skip the node).
    UnknownOp { node: usize, op: u32 },
    /// A root no node writes.
    RootNotWritten { root: usize },
}

/// A base-field operand expression (slot or constant), exactly what
/// `load_base_operand` reads.
fn base_operand(node: usize, enc: u32) -> Result<String, CodegenError> {
    let p = payload(enc);
    match kind(enc) {
        OPK_BASE_SLOT => Ok(format!("b{p}")),
        OPK_BASE_CONST => Ok(format!("d_base_consts[{p}]")),
        _ => Err(CodegenError::BadOperand { node, enc }),
    }
}

/// Any operand as an ext3 expression, exactly what `load_ext_operand` reads:
/// base values embed as `{x, 0, 0}`.
fn ext_operand(node: usize, enc: u32) -> Result<String, CodegenError> {
    let p = payload(enc);
    match kind(enc) {
        OPK_BASE_SLOT => Ok(format!("ext3::make(b{p}, 0, 0)")),
        OPK_EXT_SLOT => Ok(format!("e{p}")),
        OPK_BASE_CONST => Ok(format!("ext3::make(d_base_consts[{p}], 0, 0)")),
        OPK_EXT_CONST => Ok(format!("d_ext_consts[{p}]")),
        OPK_RAP => Ok(format!("d_rap_challenges[{p}]")),
        OPK_ALPHA => Ok(format!("d_alpha_powers[{p}]")),
        OPK_OFFSET => Ok("u_offset".to_string()),
        _ => Err(CodegenError::BadOperand { node, enc }),
    }
}

/// The statement for one node: the interpreter's `eval_program_row` case for
/// this op, operand kinds and result class, over local variables.
fn node_statement(i: usize, op: u32, a: u32, b: u32, res: u32) -> Result<String, CodegenError> {
    let slot = res & !RES_EXT_BIT;
    let res_ext = res & RES_EXT_BIT != 0;
    let need = |ext: bool| {
        if ext == res_ext {
            Ok(())
        } else {
            Err(CodegenError::ClassMismatch { node: i })
        }
    };
    let arith = |f: &str| -> Result<String, CodegenError> {
        if !res_ext {
            return Ok(format!(
                "b{slot} = goldilocks::{f}({}, {});",
                base_operand(i, a)?,
                base_operand(i, b)?
            ));
        }
        if is_base_kind(a) {
            let (x, y) = (base_operand(i, a)?, ext_operand(i, b)?);
            return Ok(match f {
                "add" => format!(
                    "{{ uint64_t x = {x}; Fe3 y = {y}; e{slot} = ext3::make(goldilocks::add(x, y.a), y.b, y.c); }}"
                ),
                "sub" => format!(
                    "{{ uint64_t x = {x}; Fe3 y = {y}; e{slot} = ext3::make(goldilocks::sub(x, y.a), goldilocks::sub(0, y.b), goldilocks::sub(0, y.c)); }}"
                ),
                _ => {
                    format!("{{ uint64_t x = {x}; Fe3 y = {y}; e{slot} = ext3::mul_base(y, x); }}")
                }
            });
        }
        if is_base_kind(b) {
            let (x, y) = (ext_operand(i, a)?, base_operand(i, b)?);
            return Ok(match f {
                "add" => format!(
                    "{{ Fe3 x = {x}; uint64_t y = {y}; e{slot} = ext3::make(goldilocks::add(x.a, y), x.b, x.c); }}"
                ),
                "sub" => format!(
                    "{{ Fe3 x = {x}; uint64_t y = {y}; e{slot} = ext3::make(goldilocks::sub(x.a, y), x.b, x.c); }}"
                ),
                _ => {
                    format!("{{ Fe3 x = {x}; uint64_t y = {y}; e{slot} = ext3::mul_base(x, y); }}")
                }
            });
        }
        Ok(format!(
            "e{slot} = ext3::{f}({}, {});",
            ext_operand(i, a)?,
            ext_operand(i, b)?
        ))
    };
    match op {
        OP_VAR => {
            let (main, offset, _row, col) = unpack_var(a, b);
            need(!main)?;
            let r = format!("r{offset}");
            if main {
                Ok(format!(
                    "b{slot} = d_main[(uint64_t){col} * main_stride + {r}];"
                ))
            } else {
                let c = 3 * col as u64;
                Ok(format!(
                    "e{slot} = ext3::make(d_aux[(uint64_t){c} * aux_stride + {r}], d_aux[(uint64_t){} * aux_stride + {r}], d_aux[(uint64_t){} * aux_stride + {r}]);",
                    c + 1,
                    c + 2
                ))
            }
        }
        OP_ADD => arith("add"),
        OP_SUB => arith("sub"),
        OP_MUL => arith("mul"),
        OP_NEG => {
            if res_ext {
                Ok(format!("e{slot} = ext3::neg({});", ext_operand(i, a)?))
            } else {
                Ok(format!(
                    "b{slot} = goldilocks::neg({});",
                    base_operand(i, a)?
                ))
            }
        }
        OP_EMBED => {
            need(true)?;
            Ok(format!("e{slot} = {};", ext_operand(i, a)?))
        }
        OP_CONST_BASE => {
            need(false)?;
            Ok(format!("b{slot} = d_base_consts[{a}];"))
        }
        OP_CONST_EXT => {
            need(true)?;
            Ok(format!("e{slot} = d_ext_consts[{a}];"))
        }
        OP_RAP_CHALLENGE => {
            need(true)?;
            Ok(format!("e{slot} = d_rap_challenges[{a}];"))
        }
        OP_ALPHA_POW => {
            need(true)?;
            Ok(format!("e{slot} = d_alpha_powers[{a}];"))
        }
        OP_TABLE_OFFSET => {
            need(true)?;
            Ok(format!("e{slot} = u_offset;"))
        }
        other => Err(CodegenError::UnknownOp { node: i, op: other }),
    }
}

/// The transition-sum statement for root `c`, as the interpreter's
/// accumulation loop computes it.
fn accumulate(c: usize, root: u32) -> String {
    let slot = root & !RES_EXT_BIT;
    if root & RES_EXT_BIT != 0 {
        format!("sum = ext3::add(sum, ext3::mul(d_beta_trans[{c}], e{slot}));")
    } else {
        format!("sum = ext3::add(sum, ext3::mul_base(d_beta_trans[{c}], b{slot}));")
    }
}

/// The mutant's wrong statement: one added to a node's result.
fn off_by_one(res: u32) -> String {
    let slot = res & !RES_EXT_BIT;
    if res & RES_EXT_BIT != 0 {
        format!("e{slot} = ext3::add(e{slot}, ext3::make(1, 0, 0)); // MUTANT")
    } else {
        format!("b{slot} = goldilocks::add(b{slot}, 1); // MUTANT")
    }
}

/// The source shared by every generated kernel: the includes, the block size
/// and the two helpers. Emitted once at the top of the generated file.
pub fn prelude() -> String {
    format!(
        r#"#include "goldilocks.cuh"
#include "ext3.cuh"

using ext3::Fe3;

#define CCOMP_BLOCK {COMPILED_BLOCK_DIM}

// `var_row` of the interpreter, with the frame offset as an argument.
__device__ __forceinline__ uint64_t ccomp_frame_row(uint64_t row, uint64_t offset,
                                                    uint64_t next_step, uint64_t num_rows) {{
    uint64_t r = row + offset * next_step;
    if (r >= num_rows) {{
        r -= num_rows;
    }}
    return r;
}}

// The interpreter's composition tail, verbatim: z_inv times the transition
// sum, then every boundary term in order.
__device__ __forceinline__ Fe3 ccomp_finish(
    Fe3 sum, uint64_t row, const uint64_t *__restrict__ d_main, uint64_t main_stride,
    const uint64_t *__restrict__ d_aux, uint64_t aux_stride, uint64_t num_rows,
    const uint64_t *__restrict__ d_z_inv, uint64_t z_len, uint64_t num_boundary,
    const uint64_t *__restrict__ d_b_col, const uint64_t *__restrict__ d_b_is_aux,
    const Fe3 *__restrict__ d_b_value, const Fe3 *__restrict__ d_b_beta,
    const uint64_t *__restrict__ d_b_z_inv) {{
    Fe3 h = ext3::mul_base(sum, d_z_inv[row % z_len]);
    for (uint64_t b = 0; b < num_boundary; b++) {{
        uint64_t col = d_b_col[b];
        Fe3 tcell;
        if (d_b_is_aux[b] != 0) {{
            uint64_t base = col * 3;
            tcell = ext3::make(d_aux[(base + 0) * aux_stride + row],
                               d_aux[(base + 1) * aux_stride + row],
                               d_aux[(base + 2) * aux_stride + row]);
        }} else {{
            tcell = ext3::make(d_main[col * main_stride + row], 0, 0);
        }}
        Fe3 bp = ext3::sub(tcell, d_b_value[b]);
        Fe3 zb = ext3::mul_base(d_b_beta[b], d_b_z_inv[b * num_rows + row]);
        h = ext3::add(h, ext3::mul(zb, bp));
    }}
    return h;
}}
"#
    )
}

/// One program's kernel, named `name`, with `constraint_composition_kernel`'s
/// parameter list (the program blob and the scratch are accepted and unused, so
/// one launch site serves both).
pub fn composition_kernel(
    dev: &DeviceProgram,
    name: &str,
    label: &str,
) -> Result<String, CodegenError> {
    emit_kernel(dev, name, label, false)
}

/// The program's kernel with ONE node wrong, for a test's mutation control:
/// the node that last writes the last root adds one to its result, so every
/// row's `H` moves by that root's coefficient. No key maps to it; only a
/// test's substitution launches it.
pub fn mutant_composition_kernel(
    dev: &DeviceProgram,
    name: &str,
    label: &str,
) -> Result<String, CodegenError> {
    emit_kernel(dev, name, label, true)
}

fn emit_kernel(
    dev: &DeviceProgram,
    name: &str,
    label: &str,
    mutant: bool,
) -> Result<String, CodegenError> {
    // Where each root's value is final: the LAST write to its slot. A root's
    // slot is pinned from the root node on, but before it a freed temporary may
    // have held the same slot.
    let mut written_at = vec![None; dev.roots.len()];
    for (c, &root) in dev.roots.iter().enumerate() {
        written_at[c] = dev.nodes.iter().rposition(|n| n.res == root);
        if written_at[c].is_none() {
            return Err(CodegenError::RootNotWritten { root: c });
        }
    }
    let mut offsets: Vec<u8> = dev
        .nodes
        .iter()
        .filter(|n| n.op == OP_VAR)
        .map(|n| unpack_var(n.a, n.b).1)
        .collect();
    offsets.sort_unstable();
    offsets.dedup();

    let mut s = String::new();
    let _ = writeln!(
        s,
        "// {label}: {} nodes, {} roots, {} base / {} ext slots",
        dev.nodes.len(),
        dev.roots.len(),
        dev.num_base_slots,
        dev.num_ext_slots
    );
    let _ = writeln!(
        s,
        r#"extern "C" __global__ void __launch_bounds__(CCOMP_BLOCK) {name}(
    Fe3 *__restrict__ d_h,
    const uint64_t *__restrict__ d_nodes,
    uint64_t num_nodes,
    const uint64_t *__restrict__ d_base_consts,
    const Fe3 *__restrict__ d_ext_consts,
    const uint64_t *__restrict__ d_roots,
    uint64_t num_roots,
    const Fe3 *__restrict__ d_rap_challenges,
    const Fe3 *__restrict__ d_alpha_powers,
    const Fe3 *__restrict__ d_table_offset,
    const uint64_t *__restrict__ d_main,
    uint64_t main_stride,
    const uint64_t *__restrict__ d_aux,
    uint64_t aux_stride,
    uint64_t next_step,
    uint64_t num_rows,
    const Fe3 *__restrict__ d_beta_trans,
    const uint64_t *__restrict__ d_z_inv,
    uint64_t z_len,
    uint64_t num_boundary,
    const uint64_t *__restrict__ d_b_col,
    const uint64_t *__restrict__ d_b_is_aux,
    const Fe3 *__restrict__ d_b_value,
    const Fe3 *__restrict__ d_b_beta,
    const uint64_t *__restrict__ d_b_z_inv,
    uint64_t *__restrict__ d_vals_base,
    uint64_t *__restrict__ d_vals_ext) {{
    (void)d_nodes; (void)num_nodes; (void)d_roots; (void)num_roots; (void)d_vals_base; (void)d_vals_ext;
    const Fe3 u_offset = *d_table_offset;
    const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t row = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; row < num_rows; row += stride) {{"#
    );
    for o in &offsets {
        let _ = writeln!(
            s,
            "        const uint64_t r{o} = ccomp_frame_row(row, {o}, next_step, num_rows);"
        );
    }
    if dev.num_base_slots > 0 {
        let names: Vec<String> = (0..dev.num_base_slots).map(|k| format!("b{k}")).collect();
        let _ = writeln!(s, "        uint64_t {};", names.join(", "));
    }
    if dev.num_ext_slots > 0 {
        let names: Vec<String> = (0..dev.num_ext_slots).map(|k| format!("e{k}")).collect();
        let _ = writeln!(s, "        Fe3 {};", names.join(", "));
    }
    let _ = writeln!(s, "        Fe3 sum = ext3::zero();");
    let mut next_root = 0usize;
    let mutated = if mutant {
        written_at.last().copied().flatten()
    } else {
        None
    };
    for (i, n) in dev.nodes.iter().enumerate() {
        let _ = writeln!(s, "        {}", node_statement(i, n.op, n.a, n.b, n.res)?);
        if mutated == Some(i) {
            let _ = writeln!(s, "        {}", off_by_one(n.res));
        }
        // Add every root whose value exists and whose predecessors are in.
        while next_root < dev.roots.len() && written_at[next_root].is_some_and(|w| w <= i) {
            let _ = writeln!(s, "        {}", accumulate(next_root, dev.roots[next_root]));
            next_root += 1;
        }
    }
    debug_assert_eq!(next_root, dev.roots.len(), "every root accumulated");
    let _ = writeln!(
        s,
        r#"        d_h[row] = ccomp_finish(sum, row, d_main, main_stride, d_aux, aux_stride, num_rows,
                                d_z_inv, z_len, num_boundary, d_b_col, d_b_is_aux, d_b_value,
                                d_b_beta, d_b_z_inv);
    }}
}}"#
    );
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint_ir::device::{DeviceNode, pack_var};

    fn node(op: u32, a: u32, b: u32, res: u32) -> DeviceNode {
        DeviceNode { op, a, b, res }
    }

    fn bslot(s: u32) -> u32 {
        (OPK_BASE_SLOT << OPK_SHIFT) | s
    }

    fn eslot(s: u32) -> u32 {
        (OPK_EXT_SLOT << OPK_SHIFT) | s
    }

    /// A small program touching every op and operand kind the interpreter
    /// knows, with a shared slot and two roots out of node order.
    fn program() -> DeviceProgram {
        let (va, vb) = pack_var(true, 0, 0, 3);
        let (xa, xb) = pack_var(false, 1, 0, 2);
        DeviceProgram {
            nodes: vec![
                node(OP_VAR, va, vb, 0),
                node(OP_VAR, xa, xb, RES_EXT_BIT),
                node(OP_MUL, bslot(0), (OPK_BASE_CONST << OPK_SHIFT) | 1, 1),
                node(OP_SUB, bslot(1), eslot(0), 1 | RES_EXT_BIT),
                node(
                    OP_ADD,
                    eslot(1),
                    (OPK_RAP << OPK_SHIFT) | 2,
                    2 | RES_EXT_BIT,
                ),
                node(OP_NEG, bslot(1), 0, 2),
                node(OP_EMBED, bslot(2), 0, 3 | RES_EXT_BIT),
                node(OP_MUL, eslot(3), OPK_OFFSET << OPK_SHIFT, 3 | RES_EXT_BIT),
            ],
            base_consts: vec![5, 7],
            ext_consts: vec![],
            roots: vec![3 | RES_EXT_BIT, 2],
            num_base: 0,
            num_base_slots: 3,
            num_ext_slots: 4,
        }
    }

    /// Every node becomes the interpreter's call for its op and operand kinds.
    #[test]
    fn each_node_is_the_interpreters_call() {
        let src = composition_kernel(&program(), "ccomp_test", "TEST").expect("emits");
        for line in [
            "b0 = d_main[(uint64_t)3 * main_stride + r0];",
            "e0 = ext3::make(d_aux[(uint64_t)6 * aux_stride + r1], d_aux[(uint64_t)7 * aux_stride + r1], d_aux[(uint64_t)8 * aux_stride + r1]);",
            "b1 = goldilocks::mul(b0, d_base_consts[1]);",
            "{ uint64_t x = b1; Fe3 y = e0; e1 = ext3::make(goldilocks::sub(x, y.a), goldilocks::sub(0, y.b), goldilocks::sub(0, y.c)); }",
            "e2 = ext3::add(e1, d_rap_challenges[2]);",
            "b2 = goldilocks::neg(b1);",
            "e3 = ext3::make(b2, 0, 0);",
            "e3 = ext3::mul(e3, u_offset);",
            "const uint64_t r1 = ccomp_frame_row(row, 1, next_step, num_rows);",
        ] {
            assert!(src.contains(line), "missing `{line}` in:\n{src}");
        }
    }

    /// The transition sum adds root 0 before root 1, after both exist, whatever
    /// order their nodes run in.
    #[test]
    fn roots_are_added_in_root_order() {
        let src = composition_kernel(&program(), "ccomp_test", "TEST").expect("emits");
        let at = |needle: &str| src.find(needle).unwrap_or_else(|| panic!("no `{needle}`"));
        let root0 = at("sum = ext3::add(sum, ext3::mul(d_beta_trans[0], e3));");
        let root1 = at("sum = ext3::add(sum, ext3::mul_base(d_beta_trans[1], b2));");
        assert!(root0 < root1, "root 0 first");
        assert!(
            at("e3 = ext3::mul(e3, u_offset);") < root0,
            "root 0 after its node"
        );
    }

    /// The key moves with the structure and ignores the constant values.
    #[test]
    fn the_key_is_the_structure() {
        let p = program();
        let k = structural_key(&p);
        let mut values = p.clone();
        values.base_consts = vec![11, 13];
        assert_eq!(
            structural_key(&values),
            k,
            "constant values are run time data"
        );
        let mut shape = p.clone();
        shape.roots.swap(0, 1);
        assert_ne!(structural_key(&shape), k, "root order is structure");
        let mut sizes = p;
        sizes.base_consts.push(1);
        assert_ne!(structural_key(&sizes), k, "table sizes are structure");
    }

    /// A root's slot that an earlier temporary also held is added after the
    /// root's own write, not the temporary's.
    #[test]
    fn a_root_is_added_after_its_last_write() {
        let mut p = program();
        // Node 5 writes b2 (a temporary), node 6 embeds it, then a new node 8
        // writes the root's slot b2 for real.
        p.nodes.push(node(OP_ADD, bslot(0), bslot(1), 2));
        let src = composition_kernel(&p, "ccomp_test", "TEST").expect("emits");
        let at = |needle: &str| src.find(needle).unwrap_or_else(|| panic!("no `{needle}`"));
        assert!(
            at("b2 = goldilocks::add(b0, b1);")
                < at("sum = ext3::add(sum, ext3::mul_base(d_beta_trans[1], b2));"),
            "root 1 is added after b2's last write"
        );
    }

    /// A root no node writes, or a node whose class disagrees with its op, is
    /// refused rather than emitted.
    #[test]
    fn malformed_programs_are_refused() {
        let mut p = program();
        p.roots.push(9 | RES_EXT_BIT);
        assert_eq!(
            composition_kernel(&p, "k", "T"),
            Err(CodegenError::RootNotWritten { root: 2 })
        );
        let mut p = program();
        p.nodes[6].res = 3;
        assert_eq!(
            composition_kernel(&p, "k", "T"),
            Err(CodegenError::ClassMismatch { node: 6 })
        );
        let mut p = program();
        p.nodes[2].op = 99;
        assert_eq!(
            composition_kernel(&p, "k", "T"),
            Err(CodegenError::UnknownOp { node: 2, op: 99 })
        );
    }

    /// The mutant is the kernel plus one statement: one added to the last
    /// root's value, right after the node that last writes it.
    #[test]
    fn the_mutant_is_one_node_off() {
        let p = program();
        let good = composition_kernel(&p, "ccomp_test", "TEST").expect("emits");
        let bad = mutant_composition_kernel(&p, "ccomp_test", "TEST").expect("emits");
        let wrong = "        b2 = goldilocks::add(b2, 1); // MUTANT\n";
        assert_eq!(bad.replacen(wrong, "", 1), good, "one statement added");
        assert!(
            bad.contains(&format!("        b2 = goldilocks::neg(b1);\n{wrong}")),
            "right after the last root's write:\n{bad}"
        );
    }
}
