//! GPU↔CPU parity for the transition-constraint interpreter kernel on the *real*
//! production programs.
//!
//! For every production table, lower its captured `ConstraintProgram` to the
//! flat device blob, run `constraint_interp.cu` (via
//! `gpu_interp::try_eval_program_gpu`) over an 8-row random LDE, and assert the
//! per-constraint eval matrix is bit-for-bit identical to the CPU reference
//! oracle `eval_device_program`. The oracle is pinned bit-for-bit to the
//! compiled prover folder by `tests::constraint_program_device_tests`, so
//! GPU == oracle closes the chain GPU == folder across all 26 tables.
//!
//! Complements the synthetic all-ops coverage in
//! `stark`'s `tests/gpu_constraint_interp.rs`: this exercises the kernel on the
//! real node graphs (hundreds of nodes, many roots, every bus/LogUp shape).
//!
//! Requires the `cuda` feature and a visible GPU. Run with:
//! ```text
//! cargo test -p lambda-vm-prover --features cuda --test gpu_constraint_interp_real
//! ```

#![cfg(feature = "cuda")]

use std::sync::Arc;

use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext;
use math::field::goldilocks::GoldilocksField as Gl;

use math_cuda::device::backend;
use math_cuda::lde::{GpuLdeBase, GpuLdeExt3};

use stark::constraint_ir::device::{
    DeviceProgram, OP_ADD, OP_ALPHA_POW, OP_EMBED, OP_MUL, OP_NEG, OP_RAP_CHALLENGE, OP_SUB,
    OP_VAR, OPK_ALPHA, OPK_PAYLOAD_MASK, OPK_RAP, OPK_SHIFT, eval_device_program, unpack_var,
};
use stark::constraint_ir::gpu_interp::try_eval_program_gpu;
use stark::proof::options::GoldilocksCubicProofOptions;
use stark::traits::AIR;

use lambda_vm_prover::test_utils::*;

type Fp = FieldElement<Gl>;
type Fp3 = FieldElement<Ext>;

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
        Fp3::from_raw([
            Fp::from_raw(self.next_u64()),
            Fp::from_raw(self.next_u64()),
            Fp::from_raw(self.next_u64()),
        ])
    }
}

fn enc(x: &Fp3) -> [u64; 3] {
    let l = x.value();
    [*l[0].value(), *l[1].value(), *l[2].value()]
}

/// #main cols, #aux cols, #rap challenges, #alpha powers, max frame offset the
/// program actually references.
fn program_footprint(dev: &DeviceProgram) -> (usize, usize, usize, usize, usize) {
    let (mut main_cols, mut aux_cols, mut rap_len, mut alpha_len, mut max_off) = (0, 0, 0, 0, 0);
    // Uniform leaves are propagated into operand encodings, so the RAP/alpha
    // footprint must be read from the operands of arithmetic nodes (the
    // root-pinned leaf-node forms are kept for completeness).
    let scan_operand = |enc: u32, rap_len: &mut usize, alpha_len: &mut usize| {
        let payload = (enc & OPK_PAYLOAD_MASK) as usize;
        match enc >> OPK_SHIFT {
            OPK_RAP => *rap_len = (*rap_len).max(payload + 1),
            OPK_ALPHA => *alpha_len = (*alpha_len).max(payload + 1),
            _ => {}
        }
    };
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
            OP_ADD | OP_SUB | OP_MUL => {
                scan_operand(n.a, &mut rap_len, &mut alpha_len);
                scan_operand(n.b, &mut rap_len, &mut alpha_len);
            }
            OP_NEG | OP_EMBED => scan_operand(n.a, &mut rap_len, &mut alpha_len),
            _ => {}
        }
    }
    (main_cols, aux_cols, rap_len, alpha_len, max_off)
}

fn check_air(air: &dyn AIR<Field = Gl, FieldExtension = Ext, PublicInputs = ()>, label: &str) {
    const NUM_ROWS: usize = 8;
    const NEXT_STEP: usize = 1;
    let lde_size = NUM_ROWS;

    let prog = air.constraint_program();
    let dev = DeviceProgram::lower(prog);
    let (main_cols, aux_cols, rap_len, alpha_len, max_off) = program_footprint(&dev);
    let n_off = max_off + 1;
    assert!(
        n_off <= NUM_ROWS,
        "[{label}] frame span {n_off} exceeds NUM_ROWS {NUM_ROWS}"
    );
    let n = dev.roots.len();
    let num_base = dev.num_base as usize;
    assert_eq!(n, air.num_transition_constraints(), "[{label}] root count");
    assert_eq!(
        num_base,
        air.num_base_transition_constraints(),
        "[{label}] num_base"
    );

    for seed in [0xABCD_0000u64 ^ label.len() as u64, 0x5EED_1234] {
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

        let gpu = try_eval_program_gpu(
            prog, &main, &aux, &rap, &alpha, &offset, NEXT_STEP, NUM_ROWS,
        )
        .unwrap_or_else(|| panic!("[{label}] GPU path must engage"));
        assert_eq!(gpu.len(), n * NUM_ROWS * 3, "[{label}] matrix shape");

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
                        "[{label}] base c{c} r{r} seed{seed:#x}: GPU {} vs CPU {}",
                        g(0),
                        base_o[c]
                    );
                    assert_eq!(g(1), 0, "[{label}] base c{c} r{r}: comp1 != 0");
                    assert_eq!(g(2), 0, "[{label}] base c{c} r{r}: comp2 != 0");
                } else {
                    let got = [g(0), g(1), g(2)];
                    assert_eq!(
                        got, ext_o[c],
                        "[{label}] ext c{c} r{r} seed{seed:#x}: GPU {got:?} vs CPU {:?}",
                        ext_o[c]
                    );
                }
            }
        }
    }
}

#[test]
fn all_table_programs_gpu_match_cpu_oracle() {
    let opts = GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 valid");

    check_air(&create_cpu_air(&opts), "CPU");
    check_air(&create_bitwise_air(&opts), "BITWISE");
    check_air(&create_lt_air(&opts), "LT");
    check_air(&create_shift_air(&opts), "SHIFT");
    check_air(&create_eq_air(&opts), "EQ");
    check_air(&create_bytewise_air(&opts), "BYTEWISE");
    check_air(&create_store_air(&opts), "STORE");
    check_air(&create_cpu32_air(&opts), "CPU32");
    check_air(&create_memw_air(&opts), "MEMW");
    check_air(&create_memw_aligned_air(&opts), "MEMW_A");
    check_air(&create_memw_register_air(&opts), "MEMW_R");
    check_air(&create_load_air(&opts), "LOAD");
    check_air(&create_decode_air(&opts), "DECODE");
    check_air(&create_mul_air(&opts), "MUL");
    check_air(&create_dvrm_air(&opts), "DVRM");
    check_air(&create_branch_air(&opts), "BRANCH");
    check_air(&create_halt_air(&opts), "HALT");
    check_air(&create_commit_air(&opts), "COMMIT");
    check_air(&create_page_air(&opts, 0x1000), "PAGE");
    check_air(&create_register_air(&opts), "REGISTER");
    check_air(&create_keccak_air(&opts), "KECCAK");
    check_air(&create_keccak_rnd_air(&opts), "KECCAK_RND");
    check_air(&create_keccak_rc_air(&opts), "KECCAK_RC");
    check_air(&create_ecsm_air(&opts), "ECSM");
    check_air(&create_ecdas_air(&opts), "ECDAS");
    check_air(&create_dma_air(&opts), "DMA");
    check_air(&create_dma_set_air(&opts), "DMA_SET");
    check_air(&create_hint_air(&opts), "HINT");
}
