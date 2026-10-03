//! The compiled constraint-composition kernels
//! (`crypto/math-cuda/kernels/constraint_compiled.cu`): which programs get one,
//! and a check that the committed source is what the current constraint IR
//! generates.
//!
//! Every production table's captured program (the VM tables, the per-epoch
//! local-to-global table, the LFM chips) is lowered to its [`DeviceProgram`]
//! and, when [`codegen::worth_compiling`], emitted as a straight-line kernel
//! keyed by [`codegen::structural_key`]. A program that changes shape changes
//! key: its old kernel is never used for it, and the freshness test below goes
//! red until the source is regenerated with
//! `LAMBDA_VM_REGEN_COMPILED_CONSTRAINTS=1 cargo test -p lambda-vm-prover --lib
//! the_compiled_constraint_kernels_are_fresh`.
//!
//! The generated source also holds one mutant, BITWISE's kernel with one node
//! wrong ([`codegen::mutant_composition_kernel`]). No key maps to it; the
//! proof-bytes test swaps it in as its mutation control.

use std::collections::BTreeMap;

use executor::elf::Elf;
use executor::vm::execution::Executor;
use stark::constraint_ir::{ConstraintProgram, DeviceProgram, codegen};
use stark::proof::options::{GoldilocksCubicProofOptions, ProofOptions};
use stark::prover::ProvingError;
use stark::traits::AIR;

use crate::VmProof;
use crate::tables::trace_builder::Traces;
use crate::tables::types::{GoldilocksExtension, GoldilocksField};
use crate::test_utils::*;

type DynAir<'a> =
    &'a dyn AIR<Field = GoldilocksField, FieldExtension = GoldilocksExtension, PublicInputs = ()>;

/// The generated CUDA source, relative to this crate.
const CU_PATH: &str = "../crypto/math-cuda/kernels/constraint_compiled.cu";
/// The generated key table, relative to this crate.
const KEYS_PATH: &str = "../crypto/math-cuda/src/constraint_compiled_keys.rs";
/// Regenerate both files instead of comparing them.
const REGEN_ENV: &str = "LAMBDA_VM_REGEN_COMPILED_CONSTRAINTS";

/// The epoch labels a block's local-to-global table is built for (1-based: its
/// range check carries the label).
const L2G_LABELS: u64 = 64;

/// The table whose kernel gets a mutant. BITWISE is the 2^20-row
/// preprocessed table, so every VM proof composes it on the device.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
const MUTANT_TABLE: &str = "BITWISE";

/// The program the proof-bytes tests prove.
pub(crate) const BYTES_PROGRAM: &str = "add";

type Program = ConstraintProgram<GoldilocksField, GoldilocksExtension>;

/// Every production program, labelled, under the options it is built with.
pub(crate) fn production_programs(opts: &ProofOptions) -> Vec<(String, Program)> {
    let mut out = Vec::new();
    let mut push = |label: String, air: DynAir<'_>| {
        out.push((label, air.constraint_program().clone()));
    };
    push("CPU".into(), &create_cpu_air(opts));
    push("BITWISE".into(), &create_bitwise_air(opts));
    push("LT".into(), &create_lt_air(opts));
    push("SHIFT".into(), &create_shift_air(opts));
    push("EQ".into(), &create_eq_air(opts));
    push("BYTEWISE".into(), &create_bytewise_air(opts));
    push("STORE".into(), &create_store_air(opts));
    push("CPU32".into(), &create_cpu32_air(opts));
    push("MEMW".into(), &create_memw_air(opts));
    push("MEMW_A".into(), &create_memw_aligned_air(opts));
    push("MEMW_R".into(), &create_memw_register_air(opts));
    push("LOAD".into(), &create_load_air(opts));
    push("DECODE".into(), &create_decode_air(opts));
    push("MUL".into(), &create_mul_air(opts));
    push("DVRM".into(), &create_dvrm_air(opts));
    push("BRANCH".into(), &create_branch_air(opts));
    push("HALT".into(), &create_halt_air(opts));
    push("COMMIT".into(), &create_commit_air(opts));
    push("PAGE".into(), &create_page_air(opts, 0x1000));
    push("REGISTER".into(), &create_register_air(opts));
    push("KECCAK".into(), &create_keccak_air(opts));
    push("KECCAK_RND".into(), &create_keccak_rnd_air(opts));
    push("KECCAK_RC".into(), &create_keccak_rc_air(opts));
    push("ECSM".into(), &create_ecsm_air(opts));
    push("ECDAS".into(), &create_ecdas_air(opts));
    for label in 1..=L2G_LABELS {
        push(
            format!("L2G[{label}]"),
            &crate::continuation::l2g_memory_air(opts, label),
        );
    }
    let roots = [[0u8; 32]; crate::lfm::airs::NUM_LFM_CHIPS];
    let lfm = crate::lfm::airs::LfmAirs::new_with_hasher(
        &roots,
        opts,
        1,
        crate::hash_pin::BLOCK_HASHER,
        crate::lfm::airs::ChipSet::FULL,
    );
    for air in lfm.air_refs() {
        push(format!("LFM {}", air.name()), air);
    }
    out
}

/// One compiled program: every label that lowers to it, its lowering, and
/// the captured program it came from (the first label's).
pub(crate) struct Compiled {
    labels: Vec<String>,
    dev: DeviceProgram,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    program: Program,
}

/// The compiled set: one kernel per distinct worthwhile program, by key.
pub(crate) fn compiled_set() -> BTreeMap<u64, Compiled> {
    let mut set: BTreeMap<u64, Compiled> = BTreeMap::new();
    for blowup in [2, 4] {
        let opts = GoldilocksCubicProofOptions::with_blowup(blowup).expect("a valid blowup");
        for (label, program) in production_programs(&opts) {
            let dev = DeviceProgram::lower(&program);
            if !codegen::worth_compiling(&dev) {
                continue;
            }
            let entry = set
                .entry(codegen::structural_key(&dev))
                .or_insert_with(|| Compiled {
                    labels: Vec::new(),
                    dev,
                    program,
                });
            if !entry.labels.contains(&label) {
                entry.labels.push(label);
            }
        }
    }
    set
}

/// A label list short enough for a comment: runs of `L2G[k]` collapse.
fn labels(ls: &[String]) -> String {
    let l2g = ls.iter().filter(|l| l.starts_with("L2G[")).count();
    let mut out: Vec<String> = ls
        .iter()
        .filter(|l| !l.starts_with("L2G["))
        .cloned()
        .collect();
    if l2g > 0 {
        out.push(format!("L2G ({l2g} labels)"));
    }
    out.join(", ")
}

/// The generated CUDA source and key table for the current constraint IR.
pub(crate) fn generated_sources() -> (String, String) {
    let set = compiled_set();
    let mut cu = String::from(
        "// GENERATED by prover/src/tests/compiled_constraints.rs — do not edit.\n\
         // Straight-line twins of `constraint_composition_kernel` (constraint_interp.cu),\n\
         // one per production constraint program, keyed by its structure, and one mutant\n\
         // for a test (last). Regenerate with\n\
         // LAMBDA_VM_REGEN_COMPILED_CONSTRAINTS=1 cargo test -p lambda-vm-prover --lib \\\n\
         //     the_compiled_constraint_kernels_are_fresh\n\n",
    );
    cu.push_str(&codegen::prelude());
    let mut keys = String::from(
        "// GENERATED by prover/src/tests/compiled_constraints.rs — do not edit.\n\n\
         /// Compiled composition kernels in `constraint_compiled.cu`: (structural key,\n\
         /// kernel name), sorted by key.\n\
         pub const COMPILED_COMPOSITION_KERNELS: &[(u64, &str)] = &[\n",
    );
    for (key, c) in &set {
        let name = codegen::kernel_name(*key);
        let src =
            codegen::composition_kernel(&c.dev, &name, &labels(&c.labels)).unwrap_or_else(|e| {
                panic!(
                    "{}: the program cannot be emitted: {e:?}",
                    labels(&c.labels)
                )
            });
        cu.push('\n');
        cu.push_str(&src);
        keys.push_str(&format!("    ({key:#018x}, \"{name}\"),\n"));
    }
    keys.push_str("];\n");
    let key = mutant_key();
    let c = set.get(&key).expect("the mutant's program is compiled");
    cu.push_str(
        "\n// MUTANT, for the proof-bytes test's mutation control only: no key maps to it,\n\
         // so no program runs it.\n",
    );
    cu.push_str(
        &codegen::mutant_composition_kernel(
            &c.dev,
            &codegen::mutant_kernel_name(key),
            &labels(&c.labels),
        )
        .expect("the mutant's program emits"),
    );
    (cu, keys)
}

/// The committed source and key table are what the current IR generates.
#[test]
fn the_compiled_constraint_kernels_are_fresh() {
    let (cu, keys) = generated_sources();
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    if std::env::var(REGEN_ENV).is_ok_and(|v| v == "1") {
        std::fs::write(root.join(CU_PATH), &cu).expect("write the kernel source");
        std::fs::write(root.join(KEYS_PATH), &keys).expect("write the key table");
        return;
    }
    let read =
        |p: &str| std::fs::read_to_string(root.join(p)).unwrap_or_else(|e| panic!("{p}: {e}"));
    assert!(
        read(CU_PATH) == cu && read(KEYS_PATH) == keys,
        "the compiled constraint kernels are stale: regenerate with {REGEN_ENV}=1 cargo test \
         -p lambda-vm-prover --lib the_compiled_constraint_kernels_are_fresh"
    );
}

/// The compiled set covers the tables that carry the composition time, and
/// leaves the very large programs on the interpreter.
#[test]
fn the_compiled_set_covers_the_hot_tables() {
    let set = compiled_set();
    let covered: Vec<&String> = set.values().flat_map(|c| &c.labels).collect();
    for hot in [
        "CPU",
        "MEMW_R",
        "MEMW_A",
        "LT",
        "BITWISE",
        "LOAD",
        "STORE",
        "SHIFT",
        "MEMW",
        "DECODE",
        "L2G[1]",
        "LFM LFM_HASH",
        "LFM LFM_XALU",
        "LFM LFM_SELECT",
        "LFM LFM_LANES",
    ] {
        assert!(
            covered.iter().any(|l| l.as_str() == hot),
            "{hot} has no compiled kernel; covered: {covered:?}"
        );
    }
    for cold in ["KECCAK_RND", "ECDAS", "ECSM"] {
        assert!(
            !covered.iter().any(|l| l.as_str() == cold),
            "{cold} is too large to compile and must stay interpreted"
        );
    }
}

/// The key of the program that gets the mutant: BITWISE's, at the options the
/// proof-bytes tests prove at.
fn mutant_key() -> u64 {
    let air = create_bitwise_air(&bytes_options());
    codegen::structural_key(&DeviceProgram::lower(air.constraint_program()))
}

/// A program's traces, built once. An RV64 VM proof is not a function of its
/// program: six table builders deduplicate through a `HashMap` (`RandomState`)
/// and lay rows out in its iteration order, so two builds of one program's
/// traces differ, and so do their proofs. Proving copies of one build takes
/// that out; at grinding 0 (no nonce search) the proof is then a function of
/// the traces, the options and the prover.
pub(crate) struct FixedTraces {
    elf_bytes: Vec<u8>,
    program: Elf,
    traces: Traces,
}

impl FixedTraces {
    pub(crate) fn build(name: &str) -> Self {
        let elf_bytes = asm_elf_bytes(name);
        let program = Elf::load(&elf_bytes).expect("load the ELF");
        let run = Executor::new(&program, vec![])
            .expect("executor")
            .run()
            .expect("run");
        let traces = Traces::from_elf_and_logs(
            &program,
            &run.logs,
            &crate::tables::MaxRowsConfig::default(),
            &[],
            #[cfg(feature = "disk-spill")]
            stark::storage_mode::StorageMode::Ram,
        )
        .expect("build the traces");
        Self {
            elf_bytes,
            program,
            traces,
        }
    }

    /// The AIRs `prove_with_options` builds for these traces.
    fn airs(&self, opts: &ProofOptions) -> crate::VmAirs {
        crate::VmAirs::new(
            &self.program,
            opts,
            false,
            &self.traces.page_configs,
            &self.traces.table_counts(),
            None,
            true,
            None,
            None,
            None,
        )
    }

    /// The structural key of the named table's program, as these AIRs have it.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    fn program_key(&self, opts: &ProofOptions, table: &str) -> u64 {
        let airs = self.airs(opts);
        let air = airs
            .air_refs()
            .into_iter()
            .find(|a| a.name() == table)
            .unwrap_or_else(|| panic!("no {table} table"));
        codegen::structural_key(&DeviceProgram::lower(air.constraint_program()))
    }

    /// `prove_with_options`' prove step on a copy of the traces: the same
    /// AIRs, the same statement in the transcript, the same `multi_prove`.
    pub(crate) fn prove(&self, opts: &ProofOptions) -> Result<VmProof, ProvingError> {
        let mut traces = self.traces.clone();
        let table_counts = traces.table_counts();
        let airs = self.airs(opts);
        let runtime_page_ranges = traces.runtime_page_ranges();
        let num_private_input_pages = traces
            .page_configs
            .iter()
            .filter(|c| c.is_private_input)
            .count();
        let mut transcript = crate::hash_pin::block_transcript(&[]);
        crate::statement::absorb_statement(
            &mut transcript,
            crate::statement::StatementKind::Monolithic,
            &self.elf_bytes,
            &traces.public_output_bytes,
            &table_counts,
            num_private_input_pages,
            &runtime_page_ranges,
            opts.fri_final_poly_log_degree,
        );
        let proof = multi_prove_ram(airs.air_trace_pairs(&mut traces), &mut transcript)?;
        Ok(VmProof {
            proof,
            runtime_page_ranges,
            table_counts,
            public_output: traces.public_output_bytes.clone(),
            num_private_input_pages,
        })
    }

    /// Whether the monolithic verifier accepts the proof.
    pub(crate) fn verifies(&self, proof: &VmProof, opts: &ProofOptions) -> bool {
        matches!(
            crate::verify_with_options(proof, &self.elf_bytes, opts, None, None),
            Ok(true)
        )
    }
}

pub(crate) fn proof_bytes(proof: &VmProof) -> Vec<u8> {
    rkyv::to_bytes::<rkyv::rancor::Error>(proof)
        .expect("serialize")
        .to_vec()
}

/// Blowup 4 (the block's) and no grinding.
pub(crate) fn bytes_options() -> ProofOptions {
    let opts = GoldilocksCubicProofOptions::with_params(4, 128, 0).expect("options");
    assert_eq!(opts.grinding_factor, 0, "no nonce search");
    opts
}

/// ★ The proof-bytes control on its own, on the path this build proves on:
/// two proofs of one set of traces are the same bytes, and they verify. The
/// device test below starts with the same control.
#[test]
#[ignore = "proves a program twice at blowup 4, with the 2^20-row BITWISE table"]
fn one_set_of_traces_proves_the_same_bytes() {
    let fixed = FixedTraces::build(BYTES_PROGRAM);
    let opts = bytes_options();
    let first = proof_bytes(&fixed.prove(&opts).expect("prove"));
    let second = fixed.prove(&opts).expect("prove");
    assert!(
        first == proof_bytes(&second),
        "two proofs of one set of traces differ"
    );
    assert!(fixed.verifies(&second, &opts), "the proof does not verify");
    println!(
        "CCOMP BYTES: two proofs of one set of traces, {} bytes each, equal",
        first.len()
    );
}

/// Writes every compiled program as text for the host parity check
/// (`crypto/math-cuda/kernels/tools/ccomp_host_check.cpp`, which builds both
/// kernels as host C++ and compares them without a GPU), to the
/// directory `LAMBDA_VM_CCOMP_DUMP` names: one `P` line per program, then its
/// `N` nodes (`op a b res`), `R` roots, `C` base constants and `X` ext constants.
#[test]
#[ignore = "a dump for the host parity check, not a test"]
fn dump_compiled_programs() {
    use std::fmt::Write as _;
    let dir = std::env::var("LAMBDA_VM_CCOMP_DUMP").expect("LAMBDA_VM_CCOMP_DUMP=<dir>");
    let mut out = String::new();
    for (
        key,
        Compiled {
            labels: ls, dev, ..
        },
    ) in compiled_set()
    {
        let _ = writeln!(
            out,
            "P {key:016x} {} {} {} {} {} {} {} {}",
            codegen::kernel_name(key),
            dev.nodes.len(),
            dev.roots.len(),
            dev.num_base_slots,
            dev.num_ext_slots,
            dev.base_consts.len(),
            dev.ext_consts.len(),
            labels(&ls).replace(' ', "_"),
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
    }
    std::fs::write(std::path::Path::new(&dir).join("programs.txt"), out).expect("write the dump");
}

/// On the device: every compiled kernel against the interpreter, and a proof
/// proved both ways.
#[cfg(feature = "cuda")]
mod device {
    use super::*;
    use std::sync::Arc;

    use math::field::element::FieldElement;
    use math_cuda::device::backend;
    use math_cuda::lde::{GpuLdeBase, GpuLdeExt3};
    use stark::constraint_ir::device::{
        OP_ALPHA_POW, OP_RAP_CHALLENGE, OP_VAR, OPK_ALPHA, OPK_PAYLOAD_MASK, OPK_RAP, OPK_SHIFT,
        unpack_var,
    };
    use stark::constraint_ir::gpu_interp::{
        CompositionInputs, GPU_COMPOSITION_COMPILED_CALLS, GPU_COMPOSITION_SUBSTITUTE_CALLS,
        GpuComposition, override_compiled_constraints, substitute_compiled_kernel,
        try_eval_composition_gpu,
    };

    type Fp = FieldElement<GoldilocksField>;
    type Fp3 = FieldElement<GoldilocksExtension>;

    /// Tests that flip the process-wide override take this first.
    static OVERRIDE: std::sync::Mutex<()> = std::sync::Mutex::new(());

    struct SplitMix64(u64);
    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn fp(&mut self) -> Fp {
            Fp::from_raw(self.next_u64())
        }
        fn fp3(&mut self) -> Fp3 {
            Fp3::from_raw([self.fp(), self.fp(), self.fp()])
        }
    }

    /// What a program reads: main and aux columns, RAP and alpha table sizes,
    /// and the largest frame offset.
    fn footprint(dev: &DeviceProgram) -> (usize, usize, usize, usize, usize) {
        let (mut main, mut aux, mut rap, mut alpha, mut off) = (1, 1, 1, 1, 0);
        for n in &dev.nodes {
            if n.op == OP_VAR {
                let (is_main, offset, _, col) = unpack_var(n.a, n.b);
                if is_main {
                    main = main.max(col as usize + 1);
                } else {
                    aux = aux.max(col as usize + 1);
                }
                off = off.max(offset as usize);
                continue;
            }
            if n.op == OP_RAP_CHALLENGE {
                rap = rap.max(n.a as usize + 1);
            }
            if n.op == OP_ALPHA_POW {
                alpha = alpha.max(n.a as usize + 1);
            }
            for enc in [n.a, n.b] {
                let (kind, pay) = (enc >> OPK_SHIFT, (enc & OPK_PAYLOAD_MASK) as usize);
                if n.op >= stark::constraint_ir::device::OP_ADD && kind == OPK_RAP {
                    rap = rap.max(pay + 1);
                }
                if n.op >= stark::constraint_ir::device::OP_ADD && kind == OPK_ALPHA {
                    alpha = alpha.max(pay + 1);
                }
            }
        }
        (main, aux, rap, alpha, off)
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

    fn inputs(dev: &DeviceProgram, rows: usize, seed: u64) -> Inputs {
        let (main_cols, aux_cols, rap_len, alpha_len, _) = footprint(dev);
        let mut rng = SplitMix64(seed);
        let base: Vec<u64> = (0..main_cols * rows).map(|_| rng.next_u64()).collect();
        let ext: Vec<u64> = (0..aux_cols * 3 * rows).map(|_| rng.next_u64()).collect();
        let be = backend().expect("cuda backend");
        let stream = be.next_stream();
        let main_buf = stream.clone_htod(&base).expect("upload main");
        let aux_buf = stream.clone_htod(&ext).expect("upload aux");
        stream.synchronize().expect("sync");
        let main = GpuLdeBase {
            ready: None,
            buf: Arc::new(main_buf),
            m: main_cols,
            lde_size: rows,
            tree: None,
            trace_dev: None,
            trace_rows: 0,
        };
        let aux = GpuLdeExt3 {
            ready: None,
            buf: Arc::new(aux_buf),
            m: aux_cols,
            lde_size: rows,
            tree: None,
        };
        let b_col = vec![0, aux_cols - 1];
        Inputs {
            main,
            aux,
            rap: (0..rap_len).map(|_| rng.fp3()).collect(),
            alpha: (0..alpha_len).map(|_| rng.fp3()).collect(),
            offset: rng.fp3(),
            beta: (0..dev.roots.len()).map(|_| rng.fp3()).collect(),
            z_inv: (0..4).map(|_| rng.fp()).collect(),
            b_col,
            b_is_aux: vec![false, true],
            b_value: (0..2).map(|_| rng.fp3()).collect(),
            b_beta: (0..2).map(|_| rng.fp3()).collect(),
            b_z_inv: (0..2)
                .map(|_| Arc::new((0..rows).map(|_| rng.fp()).collect()))
                .collect(),
        }
    }

    /// One composition kept on the device, waited for: the kernel's time plus
    /// its small uploads, no D2H.
    fn composition_on_device(program: &Program, x: &Inputs, next_step: usize, rows: usize) {
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
            program, &x.main, &x.aux, &x.rap, &x.alpha, &x.offset, next_step, rows, &accum, true,
        ) {
            Some(GpuComposition::Dev(h)) => h.synchronize().expect("synchronize"),
            _ => panic!("the GPU composition path must engage"),
        }
    }

    /// `H` for one program under the current override, host-drained.
    fn composition(program: &Program, x: &Inputs, next_step: usize, rows: usize) -> Vec<u64> {
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
            program, &x.main, &x.aux, &x.rap, &x.alpha, &x.offset, next_step, rows, &accum, false,
        ) {
            Some(GpuComposition::Host(h)) => h,
            _ => panic!("the GPU composition path must engage"),
        }
    }

    /// ★ Every compiled kernel computes the interpreter's `H`, limb for limb,
    /// on random LDE columns (full-range, so non-canonical limbs too), random
    /// uniforms and random accumulation inputs, over a grid of many blocks.
    #[test]
    fn every_compiled_kernel_computes_the_interpreters_h() {
        let _guard = OVERRIDE.lock().unwrap_or_else(|e| e.into_inner());
        const ROWS: usize = 4096;
        let mut compared = 0;
        for (key, c) in compiled_set() {
            let (_, _, _, _, off) = footprint(&c.dev);
            let next_step = 4;
            assert!(
                off * next_step < ROWS,
                "{}: frame inside the domain",
                labels(&c.labels)
            );
            for seed in [1u64, 0xDEAD_BEEF] {
                let x = inputs(&c.dev, ROWS, seed ^ key);
                override_compiled_constraints(Some(false));
                let interp = composition(&c.program, &x, next_step, ROWS);
                override_compiled_constraints(Some(true));
                let before =
                    GPU_COMPOSITION_COMPILED_CALLS.load(std::sync::atomic::Ordering::SeqCst);
                let compiled = composition(&c.program, &x, next_step, ROWS);
                let after =
                    GPU_COMPOSITION_COMPILED_CALLS.load(std::sync::atomic::Ordering::SeqCst);
                override_compiled_constraints(None);
                assert!(
                    after > before,
                    "{}: the compiled kernel ran",
                    labels(&c.labels)
                );
                assert!(
                    interp == compiled,
                    "{} (key {key:016x}, seed {seed:#x}): the compiled H differs from the \
                     interpreter's",
                    labels(&c.labels)
                );
                compared += 1;
            }
        }
        assert!(compared >= 60, "only {compared} program runs compared");
    }

    /// ★ A proof made with the compiled kernels is byte for byte the proof the
    /// interpreter makes, from one set of traces ([`FixedTraces`]) at grinding
    /// 0. Two interpreter proofs are compared first: the control that makes
    /// the comparison mean something. Then the mutation control: with
    /// BITWISE's kernel swapped for its mutant (one node wrong), the proof
    /// must change and fail to verify, or the prover must refuse it.
    #[test]
    #[ignore = "requires a GPU: proves a program four times"]
    fn the_compiled_kernels_prove_the_same_bytes() {
        use std::sync::atomic::Ordering::SeqCst;
        let _guard = OVERRIDE.lock().unwrap_or_else(|e| e.into_inner());
        let fixed = FixedTraces::build(BYTES_PROGRAM);
        let opts = bytes_options();
        let prove = |on: bool| {
            override_compiled_constraints(Some(on));
            let proof = fixed.prove(&opts);
            override_compiled_constraints(None);
            proof
        };
        let control = proof_bytes(&prove(false).expect("prove"));
        let interp = proof_bytes(&prove(false).expect("prove"));
        assert!(
            control == interp,
            "the control: two interpreter proofs of one set of traces differ"
        );
        println!(
            "CCOMP BYTES: the control, two interpreter proofs of one set of traces: {} bytes each, equal",
            interp.len()
        );

        let before = GPU_COMPOSITION_COMPILED_CALLS.load(SeqCst);
        let compiled = prove(true).expect("prove");
        let ran = GPU_COMPOSITION_COMPILED_CALLS.load(SeqCst) - before;
        assert!(ran > 0, "no composition ran compiled");
        assert!(
            proof_bytes(&compiled) == interp,
            "the compiled kernels changed the proof"
        );
        assert!(
            fixed.verifies(&compiled, &opts),
            "the proof does not verify"
        );
        println!(
            "CCOMP BYTES: the compiled proof ({ran} compiled compositions) equals it and verifies"
        );

        let key = fixed.program_key(&opts, MUTANT_TABLE);
        assert_eq!(
            key,
            mutant_key(),
            "the proof's {MUTANT_TABLE} program is the one the mutant was generated for"
        );
        let kernel = math_cuda::constraint_interp::compiled_composition_kernel(key)
            .expect("the mutant's program has a kernel");
        let mutant: &'static str = Box::leak(codegen::mutant_kernel_name(key).into_boxed_str());
        let before = GPU_COMPOSITION_SUBSTITUTE_CALLS.load(SeqCst);
        substitute_compiled_kernel(Some((kernel, mutant)));
        let mutated = prove(true);
        substitute_compiled_kernel(None);
        let ran = GPU_COMPOSITION_SUBSTITUTE_CALLS.load(SeqCst) - before;
        assert!(ran > 0, "the mutant {mutant} never ran");
        match mutated {
            Err(e) => {
                println!("CCOMP BYTES: the mutant ({ran} compositions): the prover refused: {e:?}")
            }
            Ok(proof) => {
                assert!(
                    proof_bytes(&proof) != interp,
                    "the mutant {mutant} left the proof unchanged"
                );
                assert!(
                    !fixed.verifies(&proof, &opts),
                    "a proof made with the mutant {mutant} verifies"
                );
                println!(
                    "CCOMP BYTES: the mutant ({ran} compositions): the proof changed and does not verify"
                );
            }
        }
    }

    /// Kernel time per compiled program, interpreter against compiled, at one
    /// LDE size (the ratio is what the A/B's sizing uses).
    #[test]
    #[ignore = "a benchmark: run on the box"]
    fn bench_compiled_against_the_interpreter() {
        let _guard = OVERRIDE.lock().unwrap_or_else(|e| e.into_inner());
        const ROWS: usize = 1 << 20;
        println!("CCOMP BENCH: {ROWS} LDE rows, median of 5 (ms)");
        for (key, c) in compiled_set() {
            let x = inputs(&c.dev, ROWS, key);
            let time = |on: bool| {
                override_compiled_constraints(Some(on));
                let mut ms = Vec::new();
                for i in 0..6 {
                    let t = std::time::Instant::now();
                    composition_on_device(&c.program, &x, 4, ROWS);
                    if i > 0 {
                        ms.push(t.elapsed().as_secs_f64() * 1e3);
                    }
                }
                override_compiled_constraints(None);
                ms.sort_by(|a, b| a.total_cmp(b));
                ms[ms.len() / 2]
            };
            let (i, k) = (time(false), time(true));
            println!(
                "CCOMP BENCH {:<40} nodes {:5} interpreter {i:8.2} compiled {k:8.2} speedup {:5.2}x",
                labels(&c.labels),
                c.dev.nodes.len(),
                i / k
            );
        }
    }
}
