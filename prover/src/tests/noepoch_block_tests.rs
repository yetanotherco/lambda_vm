//! The no-epoch block prover ([`crate::block`]): the device recommit is
//! invisible to the proof, its root check refuses a trace that moved, and the
//! block itself proves and verifies under the monolithic verifier.
//!
//! Every test here proves a real VM program (BITWISE alone is a 2^20-row
//! preprocessed table at blowup 4), so all of them are ignored and run on the
//! GPU box. The equivalence tests prove copies of ONE build of the traces: six
//! builders lay rows out in `HashMap` order, so two builds of one program give
//! different proofs (see `compiled_constraints::FixedTraces`). At grinding 0 the
//! proof is then a function of the traces, the options and the prover.

use std::sync::atomic::Ordering;

use executor::elf::Elf;
use executor::vm::execution::Executor;
use stark::proof::options::{GoldilocksCubicProofOptions, ProofOptions};
use stark::residency_mode::{DEVICE_RECOMMITS, ResidencyMode};

use crate::VmProof;
use crate::block::{BlockTimes, prove_block, prove_block_traces};
use crate::tables::MaxRowsConfig;
use crate::tables::trace_builder::Traces;
use crate::test_utils::asm_elf_bytes;

/// The block's format (`ZfFormat::DEFAULT` over blowup 4) with no grinding, so
/// two proves of one set of traces can be compared byte for byte.
fn bytes_options() -> ProofOptions {
    let base = GoldilocksCubicProofOptions::with_params(4, 128, 0).expect("options");
    let opts = crate::zf_format::ZfFormat::DEFAULT.options(base);
    assert_eq!(opts.grinding_factor, 0, "no nonce search");
    opts
}

/// One build of a program's traces, proved as many times as a test needs.
struct OneBuild {
    elf_bytes: Vec<u8>,
    program: Elf,
    traces: Traces,
}

impl OneBuild {
    fn new(name: &str, max_rows: &MaxRowsConfig) -> Self {
        let elf_bytes = asm_elf_bytes(name);
        let program = Elf::load(&elf_bytes).expect("load the ELF");
        let run = Executor::new(&program, vec![])
            .expect("executor")
            .run()
            .expect("run");
        let traces = Traces::from_elf_and_logs(
            &program,
            &run.logs,
            max_rows,
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

    /// The block driver's prove step on a copy of the traces. `derive_decode`
    /// supplies DECODE's root derived beside the prove, as `prove_block` does;
    /// `false` lets the AIRs derive it themselves, as the monolithic prover does.
    fn prove(
        &self,
        opts: &ProofOptions,
        residency: ResidencyMode,
        derive_decode: bool,
    ) -> Result<VmProof, crate::Error> {
        let decode = derive_decode.then(|| {
            crate::tables::decode::commitment_from_elf_device_or_host(&self.program, opts)
                .expect("DECODE commitment")
        });
        prove_block_traces(
            &self.elf_bytes,
            &self.program,
            &mut self.traces.clone(),
            opts,
            decode,
            residency,
            &mut BlockTimes::default(),
        )
    }

    fn verifies(&self, proof: &VmProof, opts: &ProofOptions) -> bool {
        matches!(
            crate::block::verify_block(proof, &self.elf_bytes, opts),
            Ok(true)
        )
    }

    /// Whether the epoch-shape verifier (KECCAK_RND one table) accepts it.
    fn verifies_single_shape(&self, proof: &VmProof, opts: &ProofOptions) -> bool {
        matches!(
            crate::verify_with_options(proof, &self.elf_bytes, opts, None, None),
            Ok(true)
        )
    }

    /// The index of the named AIR in proof order.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    fn air_index(&self, opts: &ProofOptions, name: &str) -> usize {
        let airs = crate::VmAirs::new(
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
        );
        airs.air_refs()
            .iter()
            .position(|a| a.name() == name)
            .unwrap_or_else(|| panic!("no {name} table"))
    }
}

fn proof_bytes(proof: &VmProof) -> Vec<u8> {
    rkyv::to_bytes::<rkyv::rancor::Error>(proof)
        .expect("serialize")
        .to_vec()
}

/// `Retain` with DECODE's root derived by the AIRs (the monolithic prover's
/// prove step) against `RecomputeLdeDevice` with the root supplied (the block
/// driver's): the same bytes, both verifying, and under `cuda` the device
/// recommit must have run.
fn same_bytes_under_both_modes(name: &str, max_rows: &MaxRowsConfig) {
    let build = OneBuild::new(name, max_rows);
    let opts = bytes_options();
    let retained = build
        .prove(&opts, ResidencyMode::Retain, false)
        .expect("prove under Retain");
    let before = DEVICE_RECOMMITS.load(Ordering::SeqCst);
    let recommitted = build
        .prove(&opts, ResidencyMode::RecomputeLdeDevice, true)
        .expect("prove under RecomputeLdeDevice");
    let recommits = DEVICE_RECOMMITS.load(Ordering::SeqCst) - before;
    let sub_proofs = recommitted.proof.proofs.len();
    if cfg!(feature = "cuda") {
        assert!(
            recommits > 0,
            "{name}: no table was recommitted on the device"
        );
    }
    assert!(build.verifies(&retained, &opts), "{name}: Retain proof");
    assert!(
        build.verifies(&recommitted, &opts),
        "{name}: RecomputeLdeDevice proof"
    );
    let (a, b) = (proof_bytes(&retained), proof_bytes(&recommitted));
    assert!(
        a == b,
        "{name}: proof bytes moved between Retain and RecomputeLdeDevice"
    );
    println!(
        "NOEPOCH BYTES {name}: {sub_proofs} sub-proofs, {recommits} device recommit(s), \
         {} bytes, equal, both verify",
        a.len()
    );
}

/// (a) One chunk per table, BITWISE and DECODE preprocessed (the split-tree
/// device commit): add.elf at the production caps.
#[test]
#[ignore = "proves a VM program twice at blowup 4; GPU box gate (cuda)"]
fn noepoch_same_bytes_add() {
    same_bytes_under_both_modes("add", &MaxRowsConfig::default());
}

/// (b) Keccak and memory at 2^5-row caps: many chunks of every splittable table.
#[test]
#[ignore = "proves a VM program twice at blowup 4; GPU box gate (cuda)"]
fn noepoch_same_bytes_keccak_small_chunks() {
    same_bytes_under_both_modes("test_keccak_multi", &MaxRowsConfig::small());
}

/// (c) Every instruction class at 2^5-row caps.
#[test]
#[ignore = "proves a VM program twice at blowup 4; GPU box gate (cuda)"]
fn noepoch_same_bytes_all_instructions_small_chunks() {
    same_bytes_under_both_modes("all_instructions_64", &MaxRowsConfig::small());
}

/// ★ The recommit's root check on a VM proof: one cell of one table moved
/// between Round 1 and its fused task, for a plain table (CPU[0]) and a
/// preprocessed one (BITWISE, the split-tree path). The prover must refuse with
/// `RecomputedCommitmentMismatch`, and the hook must have fired (so the device
/// recommit ran for that table).
///
/// fib_iterative_160k, not add: add's CPU table has 8 rows, below the device
/// floor, so it commits on the host and is never recommitted (FAST2 210 read
/// exactly that: "CPU[0]: the perturbation never fired"). At 160k cycles CPU[0]
/// is 2^18 rows and commits on the device at the default thresholds.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "proves a VM program at blowup 4; GPU box gate (cuda)"]
fn noepoch_refuses_a_trace_that_moved() {
    let build = OneBuild::new("fib_iterative_160k", &MaxRowsConfig::default());
    let opts = bytes_options();
    for table in ["CPU[0]", "BITWISE"] {
        let idx = build.air_index(&opts, table);
        stark::residency_mode::test_hooks::perturb_before_recommit(idx);
        let out = build.prove(&opts, ResidencyMode::RecomputeLdeDevice, true);
        let fired =
            stark::residency_mode::test_hooks::PERTURB_BEFORE_RECOMMIT.load(Ordering::SeqCst) == 0;
        stark::residency_mode::test_hooks::PERTURB_BEFORE_RECOMMIT.store(0, Ordering::SeqCst);
        assert!(fired, "{table}: the perturbation never fired");
        match out {
            Err(crate::Error::Prover(msg)) => assert!(
                msg.contains("RecomputedCommitmentMismatch"),
                "{table}: wrong refusal: {msg}"
            ),
            Err(e) => panic!("{table}: wrong refusal: {e:?}"),
            Ok(_) => panic!("{table}: a trace that moved was proved"),
        }
        println!("NOEPOCH NEGATIVE {table} (index {idx}): refused, RecomputedCommitmentMismatch");
    }
}

/// Peak resident set of this process, from `/proc/self/status` (Linux).
fn vm_hwm_gib() -> Option<f64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let kb: f64 = status
        .lines()
        .find(|l| l.starts_with("VmHWM:"))?
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()?;
    Some(kb / (1024.0 * 1024.0))
}

/// ★ THE BLOCK: `prove_block` over `NOEPOCH_ELF` with `NOEPOCH_INPUT` at the
/// block's production format (`lfm::proof::block_base_options`, the process's
/// `ZfFormat`), then the monolithic verifier. Prints the census, the phase
/// walls, the peak RSS and one `NOEPOCH RESULT` line.
#[test]
#[ignore = "proves a whole block; GPU box only (NOEPOCH_ELF, NOEPOCH_INPUT)"]
fn noepoch_block_prove_and_verify() {
    let elf_path = std::env::var("NOEPOCH_ELF").expect("NOEPOCH_ELF=<guest ELF>");
    let input_path = std::env::var("NOEPOCH_INPUT").expect("NOEPOCH_INPUT=<private input>");
    let elf_bytes = std::fs::read(&elf_path).expect("read NOEPOCH_ELF");
    let input = std::fs::read(&input_path).expect("read NOEPOCH_INPUT");
    let opts = crate::lfm::proof::block_base_options();
    println!(
        "NOEPOCH BLOCK: {elf_path} ({} B), input {input_path} ({} B), blowup {} / {} queries / \
         grinding {}",
        elf_bytes.len(),
        input.len(),
        opts.blowup_factor,
        opts.fri_number_of_queries,
        opts.grinding_factor
    );

    let (proof, times) = prove_block(&elf_bytes, &input, &opts).expect("the block must prove");
    let prove_peak = vm_hwm_gib();
    let sub_proofs = proof.proof.proofs.len();
    let size = proof_bytes(&proof).len();

    let t = std::time::Instant::now();
    let verified = matches!(
        crate::block::verify_block(&proof, &elf_bytes, &opts),
        Ok(true)
    );
    let verify_secs = t.elapsed().as_secs_f64();
    println!(
        "NOEPOCH RESULT: verified={verified} · sub-proofs {sub_proofs} · base {:.2} s \
         (execute {:.2} · build {:.2} · setup {:.2} · prove {:.2}) · verify {verify_secs:.2} s · \
         proof {size} B · host peak {} · device recommits {}",
        times.total(),
        times.execute,
        times.build,
        times.setup,
        times.prove,
        prove_peak.map_or("unknown".to_string(), |g| format!("{g:.2} GiB")),
        DEVICE_RECOMMITS.load(Ordering::SeqCst),
    );
    assert!(verified, "the block proof does not verify");
}

/// The control arm for [`noepoch_block_prove_and_verify`]: #1009's base on the
/// same binary and the same inputs, `continuation::prove_continuation` at 2^21
/// epochs under the same options, timed the same way (execution to the last
/// epoch's proof, the global proof included; no recursion, no verify).
#[test]
#[ignore = "proves a whole block; GPU box only (NOEPOCH_ELF, NOEPOCH_INPUT)"]
fn noepoch_epoch_base_reference() {
    let elf_bytes = std::fs::read(std::env::var("NOEPOCH_ELF").expect("NOEPOCH_ELF"))
        .expect("read NOEPOCH_ELF");
    let input = std::fs::read(std::env::var("NOEPOCH_INPUT").expect("NOEPOCH_INPUT"))
        .expect("read NOEPOCH_INPUT");
    let opts = crate::lfm::proof::block_base_options();
    let t = std::time::Instant::now();
    let bundle = crate::continuation::prove_continuation(
        &elf_bytes,
        &input,
        crate::block::BLOCK_ROWS_LOG2,
        &opts,
    )
    .expect("the epoch base must prove");
    let base = t.elapsed().as_secs_f64();
    println!(
        "NOEPOCH REFERENCE: epoch base {base:.2} s · {} epochs · host peak {}",
        bundle.num_epochs(),
        vm_hwm_gib().map_or("unknown".to_string(), |g| format!("{g:.2} GiB")),
    );
}

/// The block shape's accelerator rule, host only: a chunked KECCAK_RND count is
/// accepted by the block shape and refused by the single shape every other
/// verifier uses; the other accelerators stay at one table under both.
#[test]
fn a_chunked_keccak_rnd_is_accepted_only_by_the_block_shape() {
    use crate::AcceleratorShape;
    let build_counts = || {
        let (elf, logs, _) = crate::test_utils::run_asm_elf("test_keccak");
        Traces::from_elf_and_logs_minimal(&elf, &logs, &Default::default(), &[])
            .unwrap()
            .table_counts()
    };
    let honest = build_counts();
    assert_eq!(honest.keccak_rnd, 1);
    let mut chunked = honest.clone();
    chunked.keccak_rnd = 4;
    assert!(
        chunked
            .validate_for(AcceleratorShape::KeccakRndChunked)
            .is_ok()
    );
    assert!(
        chunked.validate().is_err(),
        "the single shape refuses 4 KECCAK_RND tables"
    );
    let mut two_keccak = honest.clone();
    two_keccak.keccak = 2;
    assert!(
        two_keccak
            .validate_for(AcceleratorShape::KeccakRndChunked)
            .is_err(),
        "KECCAK (the permutation table) stays one table under the block shape"
    );
}

/// ★ KECCAK_RND chunked (S2): test_keccak_multi's three permutations at 24 rows
/// per chunk give three KECCAK_RND instances. The block verifier accepts the
/// proof; the single-shape verifier refuses the same bytes (the shape is the
/// verifier's constant, never read from the proof). Retain and
/// RecomputeLdeDevice agree byte for byte.
#[test]
#[ignore = "proves a VM program twice at blowup 4; GPU box gate (cuda)"]
fn noepoch_keccak_rnd_chunked_proves_and_verifies() {
    let max_rows = MaxRowsConfig {
        keccak_rnd: 24,
        ..MaxRowsConfig::default()
    };
    let build = OneBuild::new("test_keccak_multi", &max_rows);
    assert_eq!(
        build.traces.table_counts().keccak_rnd,
        3,
        "one permutation per chunk"
    );
    let opts = bytes_options();
    let retained = build
        .prove(&opts, ResidencyMode::Retain, false)
        .expect("prove under Retain");
    let recommitted = build
        .prove(&opts, ResidencyMode::RecomputeLdeDevice, true)
        .expect("prove under RecomputeLdeDevice");
    assert!(
        build.verifies(&recommitted, &opts),
        "the block verifier accepts it"
    );
    assert!(
        !build.verifies_single_shape(&recommitted, &opts),
        "the single-shape verifier must refuse 3 KECCAK_RND tables"
    );
    assert!(proof_bytes(&retained) == proof_bytes(&recommitted));
    println!(
        "NOEPOCH KECCAK_RND CHUNKED: 3 instances, block verifier accepts, single shape refuses, bytes equal"
    );
}
