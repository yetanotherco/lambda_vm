//! Shape-independent Field VM driver for WHIR's lean coset fold.

use super::asm::{Asm, Label};
use super::isa::{Arg, Program, REG_PC, gpr};
use crate::tables::types::{FE, FEE, GoldilocksField};

const ALPHAS: usize = 8;
const FACTORS: i64 = 14;
const ETAS: i64 = 46;
const ROUND_STRIDE: usize = ETAS as usize + 6 * 32;
const QUERY_BITS: usize = 32;
const OUTPUT_CELLS: usize = 6 * 64;
const WORK_CELLS: usize = OUTPUT_CELLS;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldRound {
    pub domain_log: usize,
    pub block: usize,
    /// The lean fold's `half_alpha` values, one per level.
    pub alphas: Vec<FEE>,
    pub queries: Vec<(u64, Vec<FEE>, FEE)>,
}

/// Data memory for [`kernel`]. The returned prefix is initialized; the trailing
/// workspaces must be left uninitialized in `Memory`, since the VM defines them.
pub fn layout(rounds: &[FoldRound]) -> Vec<FEE> {
    validate_rounds(rounds);
    let mut mem = vec![FEE::zero(); 2 + rounds.len() * ROUND_STRIDE];
    for r in rounds {
        for (index, values, _) in &r.queries {
            assert_eq!(values.len(), r.block);
            assert!(
                (*index >> r.domain_log) == 0,
                "query index exceeds domain bits"
            );
        }
    }
    let headers_end = mem.len();
    let mut query_bases = Vec::with_capacity(rounds.len());
    let mut all_queries = Vec::new();
    for r in rounds {
        query_bases.push(mem.len());
        let stride = QUERY_BITS + r.block + 1;
        for (index, values, claimed) in &r.queries {
            for i in 0..QUERY_BITS {
                mem.push(if i < r.domain_log && ((index >> i) & 1) != 0 {
                    FEE::one()
                } else {
                    FEE::zero()
                });
            }
            mem.extend(values.iter().copied());
            mem.push(*claimed);
        }
        all_queries.push((stride, r.queries.len()));
    }
    let mut workspace_base = mem.len();
    for (i, r) in rounds.iter().enumerate() {
        let h = 2 + i * ROUND_STRIDE;
        let (record_stride, qcount) = all_queries[i];
        mem[h] = FEE::from(query_bases[i] as u64);
        mem[h + 1] = FEE::from(qcount as u64);
        mem[h + 2] = FEE::from(record_stride as u64);
        mem[h + 3] = FEE::from(workspace_base as u64);
        mem[h + 4] = FEE::from(WORK_CELLS as u64);
        mem[h + 5] = FEE::from(r.block as u64);
        mem[h + 6] = FEE::from(r.domain_log as u64);
        mem[h + 7] = FE::from(2u64)
            .inv()
            .expect("2 is invertible")
            .to_extension();
        for (level, alpha) in r.alphas.iter().enumerate() {
            mem[h + ALPHAS + level] = *alpha;
        }

        let generator = domain_generator(r.domain_log);
        let generator_inv = generator.inv().expect("domain generator is nonzero");
        for i in 0..32 {
            let f = if i < r.domain_log {
                generator_inv.pow(1u64 << i)
            } else {
                FE::one()
            };
            mem[h + FACTORS as usize + i] = (f - FE::one()).to_extension();
        }
        let mut current_len = r.block;
        for level in 0..r.alphas.len() {
            let eta_inv = generator_inv.pow((1usize << r.domain_log) as u64 / current_len as u64);
            for t in 0..32 {
                let eta_t = if t < current_len / 2 {
                    eta_inv.pow(t as u64).to_extension()
                } else {
                    FEE::zero()
                };
                mem[h + ETAS as usize + level * 32 + t] = eta_t;
            }
            current_len /= 2;
        }
        workspace_base += r.queries.len() * WORK_CELLS;
    }
    mem[0] = FEE::from(headers_end as u64);
    mem
}

/// A single program supports all blocks 2..=64 (powers of two), domain logs
/// through 32, any number of rounds, and any query count.
pub fn kernel() -> Program {
    build_kernel()
}

fn build_kernel() -> Program {
    // Generate fixed code for every supported block. The only runtime-varying
    // values (alphas, roots, bits, queries) are read from the round header/data.
    let mut a = Asm::new();
    // Keep only the three long-lived pointers and two working values in
    // registers. The fifth GPR is the sole scratch register. Unused workspace
    // cells hold the query end and per-level inverse-point values.
    let (rp, qp, wp, x, tmp) = (gpr(0), gpr(1), gpr(2), gpr(3), gpr(4));
    let round = a.label();
    let query = a.label();
    let query_done = a.label();
    let round_done = a.label();
    let halt = a.label();
    a.mov(Arg::reg(rp), Arg::imm(2));
    a.bind(round);
    a.sub(Arg::reg(x), Arg::reg(rp), Arg::mem_abs(0));
    a.jump_zero(halt);
    a.mov(Arg::reg(qp), Arg::mem(rp, 0));
    // Workspace cell 383 is beyond every fold output. Copy the round's
    // one-past-the-end query pointer into each query's workspace as we advance.
    // Skip this for empty rounds, which have no workspace.
    a.sub(Arg::reg(x), Arg::mem(rp, 1), Arg::imm(0));
    a.jump_zero(round_done);
    a.mov(Arg::reg(wp), Arg::mem(rp, 3));
    a.fma(
        Arg::mem(wp, 383),
        Arg::mem(rp, 1),
        Arg::mem(rp, 2),
        Arg::mem(rp, 0),
    );
    a.bind(query);

    let dispatch = a.here();
    a.mov(Arg::reg(x), Arg::mem(rp, 5));
    let table_pc = dispatch + 2;
    a.fma_out(
        Arg::reg(REG_PC),
        Arg::imm(0),
        Arg::imm(0),
        Arg::lin(x, 1, table_pc as i64),
    );
    let table: Vec<Label> = (0..=64).map(|_| a.label()).collect();
    let shapes: Vec<Label> = [2usize, 4, 8, 16, 32, 64]
        .iter()
        .map(|_| a.label())
        .collect();
    for (b, &label) in table.iter().enumerate() {
        a.bind(label);
        if let Some(i) = [2usize, 4, 8, 16, 32, 64].iter().position(|v| *v == b) {
            a.jump(shapes[i]);
        } else {
            a.jump(halt);
        }
    }

    for (shape_i, block) in [2usize, 4, 8, 16, 32, 64].into_iter().enumerate() {
        a.bind(shapes[shape_i]);
        // The bit product is a maximal unrolled sequence. Enter at the pair
        // for bit D-1, then run down through bit zero; multiplication order is
        // immaterial, and this makes the dynamic work exactly 2*D rows.
        let levels = block.trailing_zeros() as usize;
        let claim = QUERY_BITS as i64 + block as i64;
        a.mov(Arg::reg(x), Arg::imm(1));
        let bit_pc = a.here();
        a.mov(Arg::reg(tmp), Arg::mem(rp, 6));
        // For D in 1..=32, the entry is 64-2D instructions after bit_pc.
        a.fma_out(
            Arg::reg(REG_PC),
            Arg::imm(0),
            Arg::imm(0),
            Arg::lin(tmp, -2, bit_pc as i64 + 66),
        );
        for i in (0..32).rev() {
            a.fma_out(
                Arg::reg(tmp),
                Arg::mem(qp, i as i64),
                Arg::mem(rp, FACTORS + i as i64),
                Arg::imm(1),
            );
            a.mul(Arg::reg(x), Arg::reg(x), Arg::reg(tmp));
        }
        // Preserve x_inv in an unused cell of this query's workspace. Memory
        // cells are immutable, so each squaring level writes a fresh cell.
        a.fma(Arg::mem(wp, 377), Arg::imm(0), Arg::imm(0), Arg::reg(x));
        let mut current_len = block;
        for level in 0..levels {
            let pairs = current_len / 2;
            if level > 0 {
                a.fma(
                    Arg::mem(wp, 377 + level as i64),
                    Arg::mem(wp, 376 + level as i64),
                    Arg::mem(wp, 376 + level as i64),
                    Arg::imm(0),
                );
            }
            // Save alpha*x_inv once. x is reused for each pair's weight, so
            // keeping this base in a register would corrupt later pairs.
            a.fma(
                Arg::mem(wp, 370 + level as i64),
                Arg::mem(rp, ALPHAS as i64 + level as i64),
                Arg::mem(wp, 377 + level as i64),
                Arg::imm(0),
            );
            for t in 0..pairs {
                let (src_reg, src_a, src_c) = if level == 0 {
                    (
                        qp,
                        QUERY_BITS as i64 + t as i64,
                        QUERY_BITS as i64 + (t + pairs) as i64,
                    )
                } else {
                    (
                        wp,
                        (level as i64 - 1) * 64 + t as i64,
                        (level as i64 - 1) * 64 + (t + pairs) as i64,
                    )
                };
                // Three rows per output: weight, difference, and output.
                a.fma_out(
                    Arg::reg(x),
                    Arg::mem(wp, 370 + level as i64),
                    Arg::mem(rp, ETAS + (level * 32 + t) as i64),
                    Arg::mem(rp, 7),
                );
                a.fma_out(
                    Arg::reg(tmp),
                    Arg::imm(-1),
                    Arg::mem(src_reg, src_c),
                    Arg::mem(src_reg, src_a),
                );
                a.fma(
                    Arg::mem(wp, level as i64 * 64 + t as i64),
                    Arg::reg(tmp),
                    Arg::reg(x),
                    Arg::mem(src_reg, src_c),
                );
            }
            current_len = pairs;
        }
        a.fma(
            Arg::imm(0),
            Arg::imm(-1),
            Arg::mem(qp, claim),
            Arg::mem(wp, ((levels - 1) * 64) as i64),
        );
        a.jump(query_done);
    }

    a.bind(query_done);
    a.add(Arg::reg(qp), Arg::reg(qp), Arg::mem(rp, 2));
    a.sub(Arg::reg(x), Arg::mem(wp, 383), Arg::reg(qp));
    a.jump_zero(round_done);
    a.add(Arg::reg(wp), Arg::reg(wp), Arg::mem(rp, 4));
    // The next query has a fresh, write-once workspace; carry its end pointer
    // there before entering the loop again.
    a.fma(
        Arg::mem(wp, 383),
        Arg::imm(0),
        Arg::imm(0),
        Arg::mem(wp, -1),
    );
    a.jump(query);
    a.bind(round_done);
    a.add(Arg::reg(rp), Arg::reg(rp), Arg::imm(ROUND_STRIDE as i64));
    a.jump(round);
    a.bind(halt);
    a.halt();
    a.finish().expect("generic fold program assembles")
}

pub fn fold_host(round: &FoldRound, index: u64, values: &[FEE]) -> FEE {
    validate_round(round);
    assert_eq!(values.len(), round.block);
    assert!((index >> round.domain_log) == 0);
    let generator = domain_generator(round.domain_log);
    let generator_inv = generator.inv().expect("domain generator is nonzero");
    let mut x_inv = generator_inv.pow(index);
    let half = FE::from(2u64)
        .inv()
        .expect("2 is invertible")
        .to_extension();
    let mut current = values.to_vec();
    for (level, alpha) in round.alphas.iter().enumerate() {
        if level > 0 {
            x_inv *= x_inv;
        }
        let pairs = current.len() / 2;
        let eta_inv = generator_inv.pow((1usize << round.domain_log) as u64 / current.len() as u64);
        let mut next = Vec::with_capacity(pairs);
        for t in 0..pairs {
            let (a, c) = (current[t], current[t + pairs]);
            let w = *alpha * (x_inv * eta_inv.pow(t as u64)).to_extension() + half;
            next.push((a - c) * w + c);
        }
        current = next;
    }
    current[0]
}

/// The generator WHIR's domain of `2^log` points uses.
fn domain_generator(log: usize) -> FE {
    assert!((1..=32).contains(&log));
    *multilinear::whir::Domain::<GoldilocksField>::new(log)
        .expect("Goldilocks supports this domain size")
        .generator()
}

fn validate_rounds(rounds: &[FoldRound]) {
    for r in rounds {
        validate_round(r);
    }
}

fn validate_round(r: &FoldRound) {
    assert!((1..=32).contains(&r.domain_log));
    assert!((2..=64).contains(&r.block) && r.block.is_power_of_two());
    assert!(r.alphas.len() == r.block.trailing_zeros() as usize);
    assert!(r.alphas.len() <= r.domain_log);
}

// The kernel needs five general registers: not under `fvm-narrow`.
#[cfg(all(test, not(any(feature = "fvm-narrow", feature = "fvm-n1"))))]
mod tests {
    use super::*;
    use crate::field_vm::executor::{Execution, Memory, NoHints, execute};

    fn run(rounds: &[FoldRound]) -> Result<Execution, super::super::executor::ExecError> {
        let init = layout(rounds);
        let workspace = rounds
            .iter()
            .map(|r| r.queries.len() * WORK_CELLS)
            .sum::<usize>();
        let out = execute(
            &kernel(),
            Memory::with_init(init.len() + workspace, &init),
            &mut NoHints,
            5_000_000,
        );
        if let Err(super::super::executor::ExecError::FmaViolation { pc }) = &out
            && let Ok((exec, v)) = super::super::executor::execute_violating(
                &kernel(),
                Memory::with_init(init.len() + workspace, &init),
                &mut NoHints,
                5_000_000,
            )
        {
            let program = kernel();
            let at = exec
                .steps
                .iter()
                .position(|st| st.state.pc == *pc)
                .unwrap_or(0);
            for st in &exec.steps[at.saturating_sub(4)..=at] {
                eprintln!(
                    "pc {} instr {:?}\n   regs {:?}\n   premem {:?}\n   args [d,a,b,c] {:?}",
                    st.state.pc,
                    program.instrs[st.state.pc as usize],
                    st.state.regs,
                    st.args_premem,
                    st.args
                );
            }
            eprintln!(
                "{} violations, first pcs {:?}",
                v.len(),
                &v[..v.len().min(12)]
            );
        }
        out
    }

    fn sample(block: usize, domain_log: usize, q: usize, seed: u64) -> FoldRound {
        let mut state = seed;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            FEE::from(state)
        };
        let alphas: Vec<_> = (0..block.trailing_zeros()).map(|_| next()).collect();
        let mut r = FoldRound {
            domain_log,
            block,
            alphas,
            queries: Vec::new(),
        };
        for _ in 0..q {
            // A block's position in the folded domain: `2^log / block` of them.
            let leaves = domain_log - block.trailing_zeros() as usize;
            let index = next().value()[0].canonical() & ((1u64 << leaves) - 1);
            let values: Vec<_> = (0..block).map(|_| next()).collect();
            let claimed = fold_host(&r, index, &values);
            r.queries.push((index, values, claimed));
        }
        r
    }

    #[test]
    fn generic_fold_accepts_and_rejects_claims() {
        let shapes = [(16, 10, 3), (16, 17, 20), (64, 24, 7), (64, 13, 12)];
        for (i, (block, d, q)) in shapes.into_iter().enumerate() {
            let round = sample(block, d, q, 19 + i as u64);
            let execution = run(std::slice::from_ref(&round)).unwrap();
            println!(
                "generic fold block={block} D={d} Q={q}: {} rows/query",
                execution.steps.len() / q
            );
            let straight_line = 2 * d + 3 * (block - 1) + 2 * block.trailing_zeros() as usize - 1;
            assert!(
                10 * execution.steps.len() <= q * (13 * straight_line + 120),
                "{} rows/query exceeds 1.3x straight-line + 12 for block={block} D={d}",
                execution.steps.len() / q,
            );
            let mut bad = round.clone();
            bad.queries[0].2 += FEE::one();
            assert!(matches!(
                run(&[bad]),
                Err(super::super::executor::ExecError::FmaViolation { .. })
            ));
        }
    }

    /// The host reference is WHIR's own fold with `half_alpha = α/2`.
    #[test]
    fn fold_host_is_whirs_fold() {
        for (block, log) in [(16usize, 10usize), (64, 14), (2, 5)] {
            let round = sample(block, log, 3, 7);
            let two = FE::from(2u64).to_extension();
            let alphas: Vec<FEE> = round.alphas.iter().map(|a| *a * two).collect();
            let domain = multilinear::whir::Domain::<GoldilocksField>::new(log).expect("a domain");
            for (index, values, _) in &round.queries {
                let want = multilinear::whir_commit::fold_coset::<GoldilocksField, _, _>(
                    values,
                    &domain,
                    *index as usize,
                    &alphas,
                )
                .expect("WHIR folds");
                assert_eq!(
                    fold_host(&round, *index, values),
                    want,
                    "block {block} log {log}"
                );
            }
        }
    }

    #[test]
    fn kernel_does_not_depend_on_shape() {
        assert_eq!(kernel(), kernel());
        let a = sample(16, 10, 3, 1);
        let b = sample(64, 24, 20, 2);
        assert_eq!(kernel(), kernel());
        run(&[a, b]).unwrap();
    }
}
