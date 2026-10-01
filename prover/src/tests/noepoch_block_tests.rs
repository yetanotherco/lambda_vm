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
        self.prove_these(&self.traces, opts, residency, derive_decode)
    }

    /// [`Self::prove`] on a copy of `traces` (this build's, tampered).
    fn prove_these(
        &self,
        traces: &Traces,
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
            &mut traces.clone(),
            opts,
            decode,
            residency,
            Vec::new(),
            &mut BlockTimes::default(),
            &mut |_| {},
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
    // A refusal names its table and its check: the verifier's `error!` lines.
    let _ = env_logger::builder().is_test(true).try_init();
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
    assert!(chunked.validate_for(AcceleratorShape::BlockChunked).is_ok());
    assert!(
        chunked.validate().is_err(),
        "the single shape refuses 4 KECCAK_RND tables"
    );
    let mut two_keccak = honest.clone();
    two_keccak.keccak = 2;
    assert!(
        two_keccak
            .validate_for(AcceleratorShape::BlockChunked)
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

/// Every accelerator present, KECCAK_RND in two chunks and ECDAS in three: in
/// proof order (`VmAirs::air_refs`) the five fixed tables, COMMIT 5, KECCAK 6,
/// KECCAK_RND 7–8, ECSM 9, ECDAS 10–12, HINT 13, CPU 14–15, MEMW_R 16.
fn chunked_counts() -> crate::TableCounts {
    crate::TableCounts {
        cpu: 2,
        lt: 0,
        memw: 0,
        memw_aligned: 0,
        load: 0,
        mul: 0,
        dvrm: 0,
        shift: 0,
        branch: 0,
        memw_register: 1,
        eq: 0,
        bytewise: 0,
        store: 0,
        cpu32: 0,
        keccak: 1,
        keccak_rnd: 2,
        ecsm: 1,
        ecdas: 3,
        hint: 1,
        commit: 1,
        blake3: 0,
    }
}

/// The block shape's accelerator rule for ECDAS, host only: three ECDAS tables
/// pass the block shape and are refused by the single shape every other
/// verifier uses; ECSM (the call table) stays at one table under both.
#[test]
fn a_chunked_ecdas_is_accepted_only_by_the_block_shape() {
    use crate::AcceleratorShape;
    let counts = chunked_counts();
    assert!(counts.validate_for(AcceleratorShape::BlockChunked).is_ok());
    let mut ecdas_only = counts.clone();
    ecdas_only.keccak_rnd = 1;
    assert!(
        ecdas_only.validate().is_err(),
        "the single shape refuses 3 ECDAS tables"
    );
    let mut one_ecdas = ecdas_only.clone();
    one_ecdas.ecdas = 1;
    assert!(one_ecdas.validate().is_ok(), "the control: one of each");
    let mut two_ecsm = counts.clone();
    two_ecsm.ecsm = 2;
    assert!(
        two_ecsm
            .validate_for(AcceleratorShape::BlockChunked)
            .is_err(),
        "ECSM stays one table under the block shape"
    );
}

/// ★ The block shape caps the chunked tables' heights, and only theirs: an
/// ECDAS instance over [`crate::BLOCK_ECDAS_MAX_ROWS`] and a KECCAK_RND
/// instance over [`crate::BLOCK_KECCAK_RND_MAX_ROWS`] are refused by the block
/// shape and accepted by the single shape (which bounds no height); every
/// other table at a height far over both caps is accepted. The controls next
/// to each range (ECSM before ECDAS, HINT after it, KECCAK before KECCAK_RND)
/// are what an off-by-one in the instance ranges would cap instead.
#[test]
fn the_block_shape_caps_the_chunked_tables_heights() {
    use crate::AcceleratorShape::{BlockChunked, Single};
    let counts = chunked_counts();
    let n = counts.total().expect("fits") + crate::FIXED_TABLE_COUNT;
    assert_eq!(n, 17);
    let check = |shape, lengths: &[usize]| counts.check_heights_for(shape, |i| lengths[i]);
    let honest = vec![1usize << 10; n];
    assert!(check(BlockChunked, &honest).is_ok());

    let mut at_caps = honest.clone();
    at_caps[7..9].fill(crate::BLOCK_KECCAK_RND_MAX_ROWS);
    at_caps[10..13].fill(crate::BLOCK_ECDAS_MAX_ROWS);
    assert!(
        check(BlockChunked, &at_caps).is_ok(),
        "an instance at its cap passes"
    );

    for (what, idx, rows) in [
        ("ECDAS[2] over its cap", 12, crate::BLOCK_ECDAS_MAX_ROWS * 2),
        ("ECDAS[0] over its cap", 10, crate::BLOCK_ECDAS_MAX_ROWS * 2),
        (
            "KECCAK_RND[1] over its cap",
            8,
            crate::BLOCK_KECCAK_RND_MAX_ROWS * 2,
        ),
        (
            "KECCAK_RND[0] over its cap",
            7,
            crate::BLOCK_KECCAK_RND_MAX_ROWS * 2,
        ),
    ] {
        let mut lengths = honest.clone();
        lengths[idx] = rows;
        let refused = check(BlockChunked, &lengths);
        assert!(refused.is_err(), "the block shape must refuse {what}");
        assert!(
            check(Single, &lengths).is_ok(),
            "the single shape bounds no height ({what})"
        );
        println!("BLOCK SHAPE refuses {what}: {:?}", refused.err());
    }

    for (what, idx) in [
        ("COMMIT", 5),
        ("KECCAK", 6),
        ("ECSM", 9),
        ("HINT", 13),
        ("CPU[0]", 14),
        ("MEMW_R[0]", 16),
    ] {
        let mut lengths = honest.clone();
        lengths[idx] = 1 << 22;
        assert!(
            check(BlockChunked, &lengths).is_ok(),
            "{what} is not a chunked table: its height is not the shape's"
        );
    }
}

/// The instance ranges the height caps read are the chunked AIRs' positions in
/// [`crate::VmAirs::air_refs`] — the proof's order — and nothing next to them.
#[test]
fn block_chunked_ranges_name_the_chunked_airs() {
    let elf_bytes = asm_elf_bytes("poc_rodata_commit");
    let program = Elf::load(&elf_bytes).expect("load the ELF");
    let page_configs = Traces::page_configs_from_elf(&program);
    let counts = chunked_counts();
    let opts = bytes_options();
    let airs = crate::VmAirs::new(
        &program,
        &opts,
        false,
        &page_configs,
        &counts,
        None,
        true,
        None,
        None,
        None,
    );
    let names: Vec<String> = airs
        .air_refs()
        .iter()
        .map(|a| a.name().to_string())
        .collect();
    assert_eq!(
        names.len(),
        counts.total().unwrap() + crate::FIXED_TABLE_COUNT + page_configs.len()
    );
    for (table, range, _) in counts.block_chunked_ranges() {
        assert!(!range.is_empty());
        for (k, idx) in range.clone().enumerate() {
            assert_eq!(names[idx], format!("{table}[{k}]"));
        }
        let prefix = format!("{table}[");
        assert!(!names[range.start - 1].starts_with(&prefix));
        assert!(!names[range.end].starts_with(&prefix));
    }
}

/// The ECDAS chunk height of the box tests: test_ecsm_multi's three calls
/// (k = 1, 5, 0xABCDEF) make 0 + 3 + 39 = 42 double/add steps, so 16-row chunks
/// give three ECDAS instances (16, 16, 10 + padding) and the 0xABCDEF call runs
/// through all three.
const ECSM_MULTI_ECDAS_CHUNK: usize = 16;

fn ecsm_multi_chunked() -> OneBuild {
    let max_rows = MaxRowsConfig {
        ecdas: ECSM_MULTI_ECDAS_CHUNK,
        ..MaxRowsConfig::default()
    };
    let build = OneBuild::new("test_ecsm_multi", &max_rows);
    let heights: Vec<usize> = build.traces.ecdases.iter().map(|t| t.num_rows()).collect();
    assert_eq!(heights, [16, 16, 16], "three 16-row ECDAS chunks");
    build
}

/// ★ ECDAS chunked: test_ecsm_multi's 42 steps at 16 rows a chunk give three
/// ECDAS instances, one scalar multiplication running through all three. The
/// block verifier accepts the proof; the single-shape verifier refuses the
/// same bytes (the shape is the verifier's constant). Retain and
/// RecomputeLdeDevice agree byte for byte.
#[test]
#[ignore = "proves a VM program twice at blowup 4; GPU box gate (cuda)"]
fn noepoch_ecdas_chunked_proves_and_verifies() {
    let build = ecsm_multi_chunked();
    assert_eq!(build.traces.table_counts().ecdas, 3);
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
        "the single-shape verifier must refuse 3 ECDAS tables"
    );
    assert!(proof_bytes(&retained) == proof_bytes(&recommitted));
    println!(
        "NOEPOCH ECDAS CHUNKED: 3 instances, block verifier accepts, single shape refuses, bytes equal"
    );
}

/// Whether a prove of `traces` gives a proof the block verifier accepts (a
/// prover refusal counts as not accepted).
fn block_accepts(build: &OneBuild, traces: &Traces, opts: &ProofOptions) -> bool {
    match build.prove_these(traces, opts, ResidencyMode::Retain, false) {
        Ok(proof) => build.verifies(&proof, opts),
        Err(e) => {
            println!("    (the prover refused: {e:?})");
            false
        }
    }
}

/// ★ A call split across ECDAS chunks is held together by the Ecdas bus alone,
/// keyed by the call's timestamp and the step's `(round, op)`.
///
/// - Control: two whole rows swapped between chunk 0 and chunk 1 (a step of
///   the 5·G call and a step of the 0xABCDEF call) is the same multiset of
///   rows, and verifies: a chunk boundary carries nothing a bus does not.
/// - Negative: the same two rows swap only their timestamps, so each claims
///   the other's call. Their constraints still hold (no constraint reads the
///   timestamp), no range check reads it, and both rows have `NEXT_OP = 0`, so
///   their `Bit` sends are off: only the Ecdas tuples moved, and the block is
///   refused.
///
/// Mutation (box script): drop the timestamp from `ecsm::ecdas_tuple` and the
/// negative verifies — the call key is what refuses it.
#[test]
#[ignore = "proves a VM program three times at blowup 4; GPU box gate (cuda)"]
fn noepoch_ecdas_split_call_is_keyed_on_the_bus() {
    use crate::tables::ecdas::cols;
    let build = ecsm_multi_chunked();
    let opts = bytes_options();
    let ts = |t: &stark::trace::TraceTable<_, _>, r: usize| {
        (
            *t.main_table.get(r, cols::TIMESTAMP_0),
            *t.main_table.get(r, cols::TIMESTAMP_1),
        )
    };
    let next_op_off = |t: &stark::trace::TraceTable<_, _>, r: usize| {
        *t.main_table.get(r, cols::NEXT_OP) == crate::tables::types::FE::zero()
    };
    // Row a: chunk 0, the 5·G call (rows 0–2); row b: chunk 1, the 0xABCDEF
    // call. Both real rows with NEXT_OP = 0, from different calls.
    let (c0, c1) = (&build.traces.ecdases[0], &build.traces.ecdases[1]);
    let a = (0..3)
        .find(|&r| next_op_off(c0, r))
        .expect("a 5·G step with NEXT_OP = 0");
    let b = (0..16)
        .find(|&r| next_op_off(c1, r))
        .expect("a 0xABCDEF step with NEXT_OP = 0");
    assert_ne!(ts(c0, a), ts(c1, b), "two different calls");
    let one = crate::tables::types::FE::one();
    assert_eq!(*c0.main_table.get(a, cols::MU), one);
    assert_eq!(*c1.main_table.get(b, cols::MU), one);

    assert!(
        block_accepts(&build, &build.traces, &opts),
        "the honest chunked build verifies"
    );

    let mut swapped = build.traces.clone();
    for col in 0..cols::NUM_COLUMNS {
        let x = *swapped.ecdases[0].main_table.get(a, col);
        let y = *swapped.ecdases[1].main_table.get(b, col);
        swapped.ecdases[0].main_table.set(a, col, y);
        swapped.ecdases[1].main_table.set(b, col, x);
    }
    assert!(
        block_accepts(&build, &swapped, &opts),
        "whole rows swapped across chunks are the same steps: accepted"
    );
    println!("NOEPOCH ECDAS SPLIT CONTROL: whole rows swapped across chunks, accepted");

    let mut rekeyed = build.traces.clone();
    for col in [cols::TIMESTAMP_0, cols::TIMESTAMP_1] {
        let x = *rekeyed.ecdases[0].main_table.get(a, col);
        let y = *rekeyed.ecdases[1].main_table.get(b, col);
        rekeyed.ecdases[0].main_table.set(a, col, y);
        rekeyed.ecdases[1].main_table.set(b, col, x);
    }
    assert!(
        !block_accepts(&build, &rekeyed, &opts),
        "a continuation row reattached to another call must be refused"
    );
    println!(
        "NOEPOCH ECDAS SPLIT NEGATIVE: chunk 0 row {a} and chunk 1 row {b} swapped calls, refused"
    );
}

/// ★ The ECDAS height cap end to end: test_ecsm_multi's one ECDAS table padded
/// to 2^18 rows (a valid table: the extra rows are padding) proves; the block
/// verifier refuses it (an instance over [`crate::BLOCK_ECDAS_MAX_ROWS`]) and
/// the single-shape verifier, which bounds no height, accepts the same bytes.
///
/// Mutation (box script): `check_heights_for` returning `Ok(())` and the block
/// verifier accepts.
#[test]
#[ignore = "proves a VM program with a 2^18-row ECDAS at blowup 4; GPU box gate (cuda)"]
fn noepoch_ecdas_over_its_cap_is_refused() {
    let build = OneBuild::new("test_ecsm_multi", &MaxRowsConfig::default());
    let mut traces = build.traces.clone();
    assert_eq!(
        traces.ecdases.len(),
        1,
        "one ECDAS table at the default caps"
    );
    let table = &traces.ecdases[0];
    let (rows, width) = (table.num_rows(), table.main_table.width);
    assert!(rows < crate::BLOCK_ECDAS_MAX_ROWS);
    let padding = table.main_table.get_row(rows - 1).to_vec();
    assert_eq!(
        padding[crate::tables::ecdas::cols::MU],
        crate::tables::types::FE::zero(),
        "the last row is padding"
    );
    let tall = 2 * crate::BLOCK_ECDAS_MAX_ROWS;
    let mut data = Vec::with_capacity(tall * width);
    for r in 0..rows {
        data.extend_from_slice(table.main_table.get_row(r));
    }
    for _ in rows..tall {
        data.extend_from_slice(&padding);
    }
    traces.ecdases[0] = stark::trace::TraceTable::new_main(data, width, 1);
    let opts = bytes_options();
    let proof = build
        .prove_these(&traces, &opts, ResidencyMode::Retain, false)
        .expect("a padded ECDAS is a valid table");
    assert!(
        build.verifies_single_shape(&proof, &opts),
        "the single shape bounds no height: the proof itself is sound"
    );
    assert!(
        !build.verifies(&proof, &opts),
        "the block verifier must refuse an ECDAS instance over its cap"
    );
    println!("NOEPOCH ECDAS OVER CAP: 2^18 rows, single shape accepts, block verifier refuses");
}

/// The data-page record returns the root recorded for the same INIT column
/// (trailing zeros are the same column; the page base is not part of the root),
/// per layout.
#[test]
fn a_recorded_data_page_root_is_returned_for_the_same_columns_only() {
    use crate::tables::page::{self, PageConfig};
    use stark::leaf_layout::LeafLayout;
    let opts = bytes_options();
    // An INIT no real page has, so the process-wide record cannot reach another test.
    let init: Vec<u8> = b"noepoch record test: not a real page".to_vec();
    let mut padded = init.clone();
    padded.extend([0, 0, 0]);
    let config = PageConfig::with_data(0x7700_0000, padded);
    let root = [0xAB; 32];
    page::record_data_page_commitment(&config, &opts, LeafLayout::Row, root);
    let same = PageConfig::with_data(0x7780_0000, init);
    assert_eq!(
        page::data_page_commitment(&same, &opts, LeafLayout::Row),
        root
    );
    let other_layout_recorded = [0xCD; 32];
    page::record_data_page_commitment(&config, &opts, LeafLayout::RowPair, other_layout_recorded);
    assert_eq!(
        page::data_page_commitment(&config, &opts, LeafLayout::RowPair),
        other_layout_recorded,
        "each layout has its own record"
    );
}

/// ★ The device-derived data-page roots the block records are the host's: for
/// every ELF data page of all_instructions_64 and of the ethrex guest (when
/// `NOEPOCH_ELF` is set), both layouts, `commit_group_device_or_host_with` on a
/// page group equals `compute_precomputed_commitment_with`.
#[test]
#[ignore = "device commits; GPU box gate (cuda)"]
fn noepoch_device_page_roots_match_the_host() {
    use crate::tables::page;
    use stark::leaf_layout::LeafLayout;
    let opts = crate::lfm::proof::block_base_options();
    let mut elfs = vec![asm_elf_bytes("all_instructions_64")];
    if let Ok(path) = std::env::var("NOEPOCH_ELF") {
        elfs.push(std::fs::read(path).expect("read NOEPOCH_ELF"));
    }
    let mut checked = 0;
    for bytes in elfs {
        let program = Elf::load(&bytes).expect("load the ELF");
        for config in Traces::page_configs_from_elf(&program) {
            if config.init_values.is_none() {
                continue;
            }
            let group = page::preprocessed_group(&config);
            for layout in [LeafLayout::RowPair, LeafLayout::Row] {
                let device = crate::lfm::commit::commit_group_device_or_host_with(
                    "PAGE", &group, &opts, layout,
                );
                let host = page::compute_precomputed_commitment_with(&config, &opts, layout);
                assert_eq!(device, host, "page {:#x} {layout:?}", config.page_base);
                checked += 1;
            }
        }
    }
    assert!(checked > 0);
    println!("NOEPOCH PAGE ROOTS: {checked} device roots = host roots");
}

/// Each table's shape and main cells in order: the equality this test needs
/// (`TraceTable` has no `PartialEq`).
fn cells_of(
    tables: &[stark::trace::TraceTable<
        crate::tables::types::GoldilocksField,
        crate::tables::types::GoldilocksExtension,
    >],
) -> Vec<(usize, usize, Vec<u64>)> {
    tables
        .iter()
        .map(|t| {
            let m = &t.main_table;
            let cells = (0..m.height)
                .flat_map(|r| {
                    m.get_row(r)
                        .iter()
                        .map(|v| v.canonical())
                        .collect::<Vec<_>>()
                })
                .collect();
            (m.width, m.height, cells)
        })
        .collect()
}

/// Every row of `tables` (all chunks of one table type), as sortable limbs.
fn rows_of(
    tables: &[stark::trace::TraceTable<
        crate::tables::types::GoldilocksField,
        crate::tables::types::GoldilocksExtension,
    >],
) -> Vec<Vec<u64>> {
    let mut rows: Vec<Vec<u64>> = tables
        .iter()
        .flat_map(|t| {
            (0..t.main_table.height).map(move |r| {
                t.main_table
                    .get_row(r)
                    .iter()
                    .map(|v| v.canonical())
                    .collect()
            })
        })
        .collect();
    rows.sort();
    rows
}

/// The streamed phase A's traces are the serial build's under the block's own
/// caps (KECCAK_RND chunked, which the shared builder's tests do not cover):
/// windows of one CPU instance fed to [`WindowedTraceBuilder`], the last to
/// `finish`, the chunks put back with `insert_streamed`, equal
/// `Traces::from_elf_and_logs` — table by table and chunk by chunk for every
/// table laid out in op order, and row multiset by row multiset for the six
/// that lay rows out in `HashMap` order. Both of `build_streamed`'s schedules:
/// `push`, and the split builder (walk, then absorb, the jobs generated by the
/// caller). Host only, small programs.
#[test]
fn windowed_collection_builds_the_serial_traces() {
    use crate::tables::trace_builder::WindowedTraceBuilder;
    let chunked_keccak = MaxRowsConfig {
        keccak_rnd: 48,
        ..MaxRowsConfig::small()
    };
    for (name, max_rows) in [
        ("all_instructions_64", MaxRowsConfig::small()),
        ("test_keccak_multi", chunked_keccak),
        ("fib_iterative_160k", MaxRowsConfig::uniform(1 << 14)),
    ] {
        let elf_bytes = asm_elf_bytes(name);
        let program = Elf::load(&elf_bytes).expect("load the ELF");
        let run = Executor::new(&program, vec![])
            .expect("executor")
            .run()
            .expect("run");
        let serial = Traces::from_elf_and_logs(
            &program,
            &run.logs,
            &max_rows,
            &[],
            #[cfg(feature = "disk-spill")]
            stark::storage_mode::StorageMode::Ram,
        )
        .expect("serial build");

        for split in [false, true] {
            let name = format!("{name} (split: {split})");
            let mut builder = WindowedTraceBuilder::new(&program, &[], &max_rows).expect("builder");
            let windows: Vec<&[executor::vm::logs::Log]> = run.logs.chunks(max_rows.cpu).collect();
            let (last, rest) = windows.split_last().expect("a run has a window");
            let mut chunks = Vec::new();
            if split {
                let (mut walker, mut accumulator) = builder.split();
                for window in rest {
                    let walked = walker.walk(window).expect("walk");
                    chunks.extend(
                        accumulator
                            .absorb(walked)
                            .into_iter()
                            .map(|job| job.generate()),
                    );
                }
            } else {
                for window in rest {
                    chunks.extend(builder.push(window).expect("window"));
                }
            }
            let mut streamed = builder.finish(last).expect("finish");
            let n_streamed = chunks.len();
            streamed.insert_streamed(chunks).expect("insert");

            macro_rules! same {
            ($($field:ident),*) => {$(
                assert!(cells_of(&serial.$field) == cells_of(&streamed.$field), "{name}: {} differs", stringify!($field));
            )*};
        }
            same!(
                cpus,
                memws,
                memw_aligneds,
                loads,
                memw_registers,
                stores,
                shifts,
                keccaks,
                keccak_rnds,
                commits,
                cpu32s,
                ecsms,
                ecdases,
                hints,
                pages
            );
            macro_rules! same_one {
            ($($field:ident),*) => {$(
                assert!(
                    cells_of(std::slice::from_ref(&serial.$field)) == cells_of(std::slice::from_ref(&streamed.$field)),
                    "{name}: {} differs", stringify!($field)
                );
            )*};
        }
            same_one!(bitwise, decode, register, halt);
            macro_rules! same_rows {
            ($($field:ident),*) => {$(
                assert_eq!(serial.$field.len(), streamed.$field.len(), "{name}: {} chunks", stringify!($field));
                assert!(rows_of(&serial.$field) == rows_of(&streamed.$field), "{name}: {} rows differ", stringify!($field));
            )*};
        }
            same_rows!(lts, branches, muls, dvrms, eqs, bytewises);
            assert_eq!(serial.public_output_bytes, streamed.public_output_bytes);
            assert_eq!(serial.table_counts().cpu, streamed.table_counts().cpu);
            assert!(n_streamed > 0, "{name}: no chunk was streamed");
            println!(
                "{name}: windowed = serial ({} windows, {n_streamed} streamed chunks, {} KECCAK_RND)",
                windows.len(),
                streamed.keccak_rnds.len()
            );
        }
    }
}
