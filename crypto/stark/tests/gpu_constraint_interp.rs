//! GPU↔CPU parity for the transition-constraint interpreter kernel
//! (`crypto/math-cuda/kernels/constraint_interp.cu`).
//!
//! The kernel — driven through `gpu_interp::try_eval_program_gpu` — must produce
//! the per-constraint eval matrix bit-for-bit identical to the CPU reference
//! oracle [`eval_device_program`] (the flat-blob forward walk in
//! `constraint_ir::device`). That oracle is itself pinned bit-for-bit to the
//! production folder across all 26 tables by
//! `lambda_vm_prover::tests::constraint_program_device_tests`, so GPU == oracle
//! closes the chain GPU == compiled prover folder without needing a GPU there.
//!
//! Layouts (must match the kernel + the host wrapper):
//!   * base LDE column-major     `buf[col * lde_size + row]`   (`GpuLdeBase`)
//!   * ext3 LDE de-interleaved   `buf[(col*3 + k) * lde_size + row]` (`GpuLdeExt3`)
//!   * `lde_size` is the row stride; here `lde_size == num_rows`, `next_step = 1`.
//!
//! CRITICAL — same-cell reads: the kernel resolves an `Op::Var{offset}` leaf at
//! LDE row `(row + offset * next_step) mod num_rows`. So for kernel row `r` the
//! oracle is fed `main[o][c] = base_lde[c][(r + o) mod num_rows]` and
//! `aux[o][c] = aux_lde[c][(r + o) mod num_rows]` — reproducing the real
//! next-row wrap exactly.
//!
//! Requires the `cuda` feature and a visible GPU.

#![cfg(feature = "cuda")]

use std::sync::Arc;

use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext;
use math::field::goldilocks::GoldilocksField as Gl;

use math_cuda::device::backend;
use math_cuda::lde::{GpuLdeBase, GpuLdeExt3};

use stark::constraint_ir::device::{
    DeviceProgram, OP_ALPHA_POW, OP_RAP_CHALLENGE, OP_VAR, eval_device_program, unpack_var,
};
use stark::constraint_ir::{ConstraintProgram, IrBuilder};

type Fp = FieldElement<Gl>;
type Fp3 = FieldElement<Ext>;

/// Deterministic SplitMix64 (no `rand` needed; matches the device.rs oracle test).
struct SplitMix64(u64);
impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn fp3(&mut self) -> Fp3 {
        // Raw (possibly non-canonical) limbs — the stronger test, and exactly
        // what a real LDE column carries.
        Fp3::from_raw([
            Fp::from_raw(self.next_u64()),
            Fp::from_raw(self.next_u64()),
            Fp::from_raw(self.next_u64()),
        ])
    }
}

fn fp(v: u64) -> Fp {
    Fp::from(v)
}
fn ext3(a: u64, b: u64, c: u64) -> Fp3 {
    Fp3::from_raw([fp(a), fp(b), fp(c)])
}

/// Extension element → raw `[u64; 3]` limbs (the device representation).
fn enc(x: &Fp3) -> [u64; 3] {
    let l = x.value();
    [*l[0].value(), *l[1].value(), *l[2].value()]
}

/// The all-ops synthetic program (mirrors `device.rs`'s own oracle test): every
/// `Op` variant, both dims, a base-rooted constraint plus two ext (LogUp-shaped)
/// roots, next-row reads, and mixed base×ext arithmetic.
fn all_ops_program() -> ConstraintProgram<Gl, Ext> {
    let mut b = IrBuilder::<Gl, Ext>::new();

    // Root 0 (base): (m0 + m1) * 2 - m0_next, all base, incl. next-row.
    let m0 = b.main(0, 0);
    let m1 = b.main(0, 1);
    let m0n = b.main(1, 0);
    let two = b.const_base(2);
    let sum = b.add(m0, m1);
    let scaled = b.mul(sum, two);
    let base_root = b.sub(scaled, m0n);
    b.emit(0, base_root);

    // Root 1 (ext): m0 * challenge(0) + alpha_pow(1) * aux(0,0) - table_offset
    let ch = b.challenge(0);
    let ap = b.alpha_power(1);
    let au = b.aux(0, 0);
    let off = b.table_offset();
    let t1 = b.mul(m0, ch); // base × ext → ext (auto-embed)
    let t2 = b.mul(ap, au); // ext × ext
    let s = b.add(t1, t2);
    let ext_root = b.sub(s, off);
    b.emit(1, ext_root);

    // Root 2 (ext): embed(m1) + (-aux(0,1)) + const_ext
    let em = b.embed(m1);
    let au1 = b.aux(0, 1);
    let nau1 = b.neg(au1); // ext negation
    let ce = b.const_ext(ext3(9, 8, 7));
    let s2 = b.add(em, nau1);
    let ext_root2 = b.add(s2, ce);
    b.emit(2, ext_root2);

    b.finish(1) // 1 base root, 2 ext roots
}

/// Derive the trace/uniform footprint the program actually touches, so the
/// harness works for any program (synthetic or real): #main cols, #aux cols,
/// #rap challenges, #alpha powers, and the max frame offset.
fn program_footprint(dev: &DeviceProgram) -> (usize, usize, usize, usize, usize) {
    let (mut main_cols, mut aux_cols, mut rap_len, mut alpha_len, mut max_off) = (0, 0, 0, 0, 0);
    for n in &dev.nodes {
        match n.op {
            OP_VAR => {
                let (is_main, offset, _row, col) = unpack_var(n.a, n.b);
                let col = col as usize + 1;
                if is_main {
                    main_cols = main_cols.max(col);
                } else {
                    aux_cols = aux_cols.max(col);
                }
                max_off = max_off.max(offset as usize);
            }
            OP_RAP_CHALLENGE => rap_len = rap_len.max(n.a as usize + 1),
            OP_ALPHA_POW => alpha_len = alpha_len.max(n.a as usize + 1),
            _ => {}
        }
    }
    (main_cols, aux_cols, rap_len, alpha_len, max_off)
}

/// Full GPU↔CPU differential for one program over an 8-row random LDE.
fn check_program(prog: &ConstraintProgram<Gl, Ext>, label: &str, seed: u64) {
    const NUM_ROWS: usize = 8;
    const NEXT_STEP: usize = 1;
    let lde_size = NUM_ROWS;

    let dev = DeviceProgram::lower(prog);
    let (main_cols, aux_cols, rap_len, alpha_len, max_off) = program_footprint(&dev);
    let n_off = max_off + 1;
    assert!(
        n_off <= NUM_ROWS,
        "[{label}] program frame span {n_off} exceeds NUM_ROWS {NUM_ROWS}"
    );
    let n = dev.roots.len();
    let num_base = dev.num_base as usize;

    let mut rng = SplitMix64(seed);

    // Host-side random LDE, kept column-major so we can both upload it and feed
    // the oracle the exact same cells.
    let base_host: Vec<Vec<u64>> = (0..main_cols)
        .map(|_| (0..NUM_ROWS).map(|_| rng.next_u64()).collect())
        .collect();
    let aux_host: Vec<Vec<[u64; 3]>> = (0..aux_cols)
        .map(|_| (0..NUM_ROWS).map(|_| enc(&rng.fp3())).collect())
        .collect();

    // Per-proof uniforms.
    let rap: Vec<Fp3> = (0..rap_len.max(1)).map(|_| rng.fp3()).collect();
    let alpha: Vec<Fp3> = (0..alpha_len.max(1)).map(|_| rng.fp3()).collect();
    let offset = rng.fp3();

    // Pack the LDE into the device buffer layouts and upload.
    let mut base_flat = vec![0u64; main_cols * lde_size];
    for (c, col) in base_host.iter().enumerate() {
        for (r, v) in col.iter().enumerate() {
            base_flat[c * lde_size + r] = *v;
        }
    }
    let mut aux_flat = vec![0u64; aux_cols * 3 * lde_size];
    for (c, col) in aux_host.iter().enumerate() {
        for (r, v) in col.iter().enumerate() {
            aux_flat[(c * 3) * lde_size + r] = v[0];
            aux_flat[(c * 3 + 1) * lde_size + r] = v[1];
            aux_flat[(c * 3 + 2) * lde_size + r] = v[2];
        }
    }

    let be = backend().expect("cuda backend");
    let stream = be.next_stream();
    let base_dev = stream.clone_htod(&base_flat).expect("upload base LDE");
    let aux_dev = stream.clone_htod(&aux_flat).expect("upload aux LDE");
    stream.synchronize().expect("sync uploads");

    let main = GpuLdeBase {
        ready: None,
        buf: Arc::new(base_dev),
        m: main_cols,
        lde_size,
        tree: None,
        trace_dev: None,
        trace_rows: 0,
    };
    let aux = GpuLdeExt3 {
        ready: None,
        buf: Arc::new(aux_dev),
        m: aux_cols,
        lde_size,
        tree: None,
    };

    // GPU: launch the interpreter over every row.
    let gpu = stark::constraint_ir::gpu_interp::try_eval_program_gpu(
        prog, &main, &aux, &rap, &alpha, &offset, NEXT_STEP, NUM_ROWS,
    )
    .unwrap_or_else(|| panic!("[{label}] GPU path (Goldilocks ext3) must engage"));
    assert_eq!(gpu.len(), n * NUM_ROWS * 3, "[{label}] eval matrix shape");

    // CPU oracle, row by row, reading the SAME wrapped LDE cells.
    let rap_raw: Vec<[u64; 3]> = rap.iter().map(enc).collect();
    let alpha_raw: Vec<[u64; 3]> = alpha.iter().map(enc).collect();
    let off_raw = enc(&offset);

    for r in 0..NUM_ROWS {
        let main_raw: Vec<Vec<u64>> = (0..n_off)
            .map(|o| {
                (0..main_cols)
                    .map(|c| base_host[c][(r + o) % NUM_ROWS])
                    .collect()
            })
            .collect();
        let aux_raw: Vec<Vec<[u64; 3]>> = (0..n_off)
            .map(|o| {
                (0..aux_cols)
                    .map(|c| aux_host[c][(r + o) % NUM_ROWS])
                    .collect()
            })
            .collect();

        let mut base_o = vec![0u64; num_base];
        let mut ext_o = vec![[0u64; 3]; n];
        eval_device_program(
            &dev,
            &main_raw,
            &aux_raw,
            &rap_raw,
            &alpha_raw,
            off_raw,
            &mut base_o,
            &mut ext_o,
        );

        for c in 0..n {
            let g = |k: usize| gpu[(c * NUM_ROWS + r) * 3 + k];
            if c < num_base {
                assert_eq!(
                    g(0),
                    base_o[c],
                    "[{label}] base constraint {c}, row {r}: GPU {} vs CPU {}",
                    g(0),
                    base_o[c]
                );
                // A base-rooted constraint carries its value in component 0;
                // the embedding pads components 1 and 2 with zero.
                assert_eq!(
                    g(1),
                    0,
                    "[{label}] base constraint {c}, row {r}: comp1 != 0"
                );
                assert_eq!(
                    g(2),
                    0,
                    "[{label}] base constraint {c}, row {r}: comp2 != 0"
                );
            } else {
                let got = [g(0), g(1), g(2)];
                assert_eq!(
                    got, ext_o[c],
                    "[{label}] ext constraint {c}, row {r}: GPU {got:?} vs CPU {:?}",
                    ext_o[c]
                );
            }
        }
    }
}

#[test]
fn gpu_matches_cpu_oracle_all_ops() {
    // A few seeds to exercise the reduce/overflow paths with different limbs.
    for seed in [0x0123_4567_89AB_CDEF, 0xDEAD_BEEF_CAFE_F00D, 1, 42] {
        check_program(&all_ops_program(), "ALL_OPS", seed);
    }
}

// ------------------------------------------------------------------------
// Fused composition-poly kernel (`constraint_composition_kernel`): the GPU
// H(row) must match the CPU accumulation of `constraints::evaluator` applied
// to the same per-constraint evals — z_inv·Σβᵢ·Cᵢ + Σ_b z_b_inv·β_b·(trace−val).
// ------------------------------------------------------------------------

use stark::constraint_ir::gpu_interp::{CompositionInputs, try_eval_composition_gpu};

/// CPU reference for one row: mirror `evaluator.rs` (uniform case) exactly,
/// consuming `eval_device_program`'s per-constraint evals.
#[allow(clippy::too_many_arguments)]
fn composition_oracle_row(
    base_evals: &[u64],
    ext_evals: &[[u64; 3]],
    num_base: usize,
    beta_trans: &[Fp3],
    z_inv_row: Fp,
    b_terms: &[(bool, usize, Fp3, Fp3, Fp)], // (is_aux, col, value, beta, z_inv_row)
    base_row: &[u64],
    aux_row: &[[u64; 3]],
) -> Fp3 {
    let mut sum = Fp3::zero();
    for (c, beta) in beta_trans.iter().enumerate() {
        // eval * beta, base×ext for base constraints (matches evaluator.rs:89/92).
        if c < num_base {
            sum += Fp::from_raw(base_evals[c]) * *beta;
        } else {
            let e = ext_evals[c];
            sum +=
                Fp3::from_raw([Fp::from_raw(e[0]), Fp::from_raw(e[1]), Fp::from_raw(e[2])]) * *beta;
        }
    }
    let mut h = z_inv_row * sum; // z * sum (base×ext)
    for &(is_aux, col, value, beta, zinv) in b_terms {
        let tcell = if is_aux {
            let a = aux_row[col];
            Fp3::from_raw([Fp::from_raw(a[0]), Fp::from_raw(a[1]), Fp::from_raw(a[2])])
        } else {
            Fp::from_raw(base_row[col]).to_extension::<Ext>()
        };
        let bp = tcell - value;
        h += zinv * beta * bp; // (base×ext)×ext, matches evaluator.rs:234
    }
    h
}

fn check_composition(prog: &ConstraintProgram<Gl, Ext>, label: &str, seed: u64) {
    const NUM_ROWS: usize = 8;
    const NEXT_STEP: usize = 1;
    const Z_LEN: usize = 2; // blowup-length cyclic transition zerofier inverse
    let lde_size = NUM_ROWS;

    let dev = DeviceProgram::lower(prog);
    let (main_cols, aux_cols, rap_len, alpha_len, max_off) = program_footprint(&dev);
    assert!(max_off < NUM_ROWS);
    let n = dev.roots.len();
    let num_base = dev.num_base as usize;

    let mut rng = SplitMix64(seed);

    let base_host: Vec<Vec<u64>> = (0..main_cols)
        .map(|_| (0..NUM_ROWS).map(|_| rng.next_u64()).collect())
        .collect();
    let aux_host: Vec<Vec<[u64; 3]>> = (0..aux_cols)
        .map(|_| (0..NUM_ROWS).map(|_| enc(&rng.fp3())).collect())
        .collect();
    let rap: Vec<Fp3> = (0..rap_len.max(1)).map(|_| rng.fp3()).collect();
    let alpha: Vec<Fp3> = (0..alpha_len.max(1)).map(|_| rng.fp3()).collect();
    let offset = rng.fp3();

    // Accumulation inputs (synthetic but shaped exactly like the real ones).
    let beta_trans: Vec<Fp3> = (0..n).map(|_| rng.fp3()).collect();
    let z_inv: Vec<Fp> = (0..Z_LEN).map(|_| fp(rng.next_u64())).collect();

    // Two boundary constraints: one main (col 0), one aux (last aux col).
    let b_defs: Vec<(bool, usize)> = {
        let mut v = Vec::new();
        if main_cols > 0 {
            v.push((false, 0));
        }
        if aux_cols > 0 {
            v.push((true, aux_cols - 1));
        }
        v
    };
    let num_boundary = b_defs.len();
    let b_col: Vec<usize> = b_defs.iter().map(|&(_, c)| c).collect();
    let b_is_aux: Vec<bool> = b_defs.iter().map(|&(a, _)| a).collect();
    let b_value: Vec<Fp3> = (0..num_boundary).map(|_| rng.fp3()).collect();
    let b_beta: Vec<Fp3> = (0..num_boundary).map(|_| rng.fp3()).collect();
    // b_z_inv: one num_rows-length vector per boundary constraint (the
    // per-constraint shape the evaluator hands over; device layout is still
    // b*num_rows + row).
    let b_z_inv: Vec<Vec<Fp>> = (0..num_boundary)
        .map(|_| (0..NUM_ROWS).map(|_| fp(rng.next_u64())).collect())
        .collect();

    // Upload the LDE and build handles.
    let mut base_flat = vec![0u64; main_cols.max(1) * lde_size];
    for (c, col) in base_host.iter().enumerate() {
        for (r, v) in col.iter().enumerate() {
            base_flat[c * lde_size + r] = *v;
        }
    }
    let mut aux_flat = vec![0u64; aux_cols.max(1) * 3 * lde_size];
    for (c, col) in aux_host.iter().enumerate() {
        for (r, v) in col.iter().enumerate() {
            aux_flat[(c * 3) * lde_size + r] = v[0];
            aux_flat[(c * 3 + 1) * lde_size + r] = v[1];
            aux_flat[(c * 3 + 2) * lde_size + r] = v[2];
        }
    }
    let be = backend().expect("cuda backend");
    let stream = be.next_stream();
    let base_dev = stream.clone_htod(&base_flat).expect("upload base");
    let aux_dev = stream.clone_htod(&aux_flat).expect("upload aux");
    stream.synchronize().expect("sync");
    let main = GpuLdeBase {
        ready: None,
        buf: Arc::new(base_dev),
        m: main_cols,
        lde_size,
        tree: None,
        trace_dev: None,
        trace_rows: 0,
    };
    let aux = GpuLdeExt3 {
        ready: None,
        buf: Arc::new(aux_dev),
        m: aux_cols,
        lde_size,
        tree: None,
    };

    let inputs = CompositionInputs {
        beta_trans: &beta_trans,
        z_inv: &z_inv,
        b_col: &b_col,
        b_is_aux: &b_is_aux,
        b_value: &b_value,
        b_beta: &b_beta,
        b_z_inv: &b_z_inv,
    };
    let gpu = try_eval_composition_gpu(
        prog, &main, &aux, &rap, &alpha, &offset, NEXT_STEP, NUM_ROWS, &inputs,
    )
    .unwrap_or_else(|| panic!("[{label}] GPU composition path must engage"));
    assert_eq!(gpu.len(), NUM_ROWS * 3, "[{label}] H shape");

    // CPU oracle, row by row.
    let rap_raw: Vec<[u64; 3]> = rap.iter().map(enc).collect();
    let alpha_raw: Vec<[u64; 3]> = alpha.iter().map(enc).collect();
    let off_raw = enc(&offset);
    let n_off = max_off + 1;

    for r in 0..NUM_ROWS {
        let main_raw: Vec<Vec<u64>> = (0..n_off)
            .map(|o| {
                (0..main_cols)
                    .map(|c| base_host[c][(r + o) % NUM_ROWS])
                    .collect()
            })
            .collect();
        let aux_raw: Vec<Vec<[u64; 3]>> = (0..n_off)
            .map(|o| {
                (0..aux_cols)
                    .map(|c| aux_host[c][(r + o) % NUM_ROWS])
                    .collect()
            })
            .collect();
        let mut base_o = vec![0u64; num_base];
        let mut ext_o = vec![[0u64; 3]; n];
        eval_device_program(
            &dev,
            &main_raw,
            &aux_raw,
            &rap_raw,
            &alpha_raw,
            off_raw,
            &mut base_o,
            &mut ext_o,
        );

        // Boundary terms read the current row (offset 0).
        let b_terms: Vec<(bool, usize, Fp3, Fp3, Fp)> = (0..num_boundary)
            .map(|b| (b_is_aux[b], b_col[b], b_value[b], b_beta[b], b_z_inv[b][r]))
            .collect();

        let h_cpu = composition_oracle_row(
            &base_o,
            &ext_o,
            num_base,
            &beta_trans,
            z_inv[r % Z_LEN],
            &b_terms,
            &main_raw[0],
            &aux_raw[0],
        );

        let h_gpu = [gpu[r * 3], gpu[r * 3 + 1], gpu[r * 3 + 2]];
        assert_eq!(
            h_gpu,
            enc(&h_cpu),
            "[{label}] H mismatch row {r} seed {seed:#x}: GPU {h_gpu:?} vs CPU {:?}",
            enc(&h_cpu)
        );
    }
}

#[test]
fn gpu_composition_matches_cpu_oracle_all_ops() {
    for seed in [0x0123_4567_89AB_CDEF, 0xDEAD_BEEF_CAFE_F00D, 7] {
        check_composition(&all_ops_program(), "ALL_OPS_COMP", seed);
    }
}
