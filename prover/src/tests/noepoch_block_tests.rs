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

/// (d) A plain device-committed table (CPU[0] at 2^18 rows): the table the
/// kept top levels (`LAMBDA_VM_RECOMMIT_TOP_LEVELS`) apply to, so their
/// rebuilt openings must give the Retain proof's bytes.
#[test]
#[ignore = "proves a VM program twice at blowup 4; GPU box gate (cuda)"]
fn noepoch_same_bytes_fib_160k() {
    same_bytes_under_both_modes("fib_iterative_160k", &MaxRowsConfig::default());
}

// =========================================================================
// ZisK's Poseidon1 as the base hash (`p1/*` exploration, P2): run under
// `LAMBDA_VM_BASE_HASH=p1`, by the box gate. The tests above also run there
// unchanged, proving and verifying under P1.
// =========================================================================

fn require_p1_base() {
    assert_eq!(
        crate::hash_pin::base_hash(),
        crate::hash_pin::BaseHash::P1,
        "run under {}=p1",
        crate::hash_pin::BASE_HASH_ENV
    );
}

/// ★ A P1 block proof (grinding on, so the device grind runs) verifies under
/// its own configuration and is refused by the RPX one — the base hash is a
/// verifier constant and the two configurations do not agree by accident —
/// and a flipped byte in a main-trace path, an FRI path, the main root or the
/// nonce is refused.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "proves a VM program at blowup 4 under LAMBDA_VM_BASE_HASH=p1; GPU box gate (cuda)"]
fn noepoch_p1_proof_is_refused_by_rpx_and_when_tampered() {
    use crate::hash_pin::BaseHash;
    require_p1_base();
    let build = OneBuild::new("fib_iterative_160k", &MaxRowsConfig::default());
    let base = GoldilocksCubicProofOptions::with_params(4, 128, 12).expect("options");
    let opts = crate::zf_format::ZfFormat::DEFAULT.options(base);
    crate::hash_pin::warm_base_statics(&opts);
    crypto::grinding::reset_gpu_grind_calls();
    let proof = build
        .prove(&opts, ResidencyMode::RecomputeLdeDevice, true)
        .expect("P1 prove");
    assert!(
        crypto::grinding::gpu_grind_calls_p1() > 0,
        "no P1 device grind ran"
    );
    assert_eq!(
        crypto::grinding::gpu_grind_calls_rpx(),
        0,
        "an RPX grind ran"
    );
    let verify = |p: &VmProof, base: BaseHash| {
        matches!(
            crate::block::verify_block_under(p, &build.elf_bytes, &opts, base),
            Ok(true)
        )
    };
    assert!(verify(&proof, BaseHash::P1), "the P1 proof must verify");
    assert!(
        !verify(&proof, BaseHash::Rpx),
        "the RPX verifier accepted a P1 proof"
    );
    let idx = build.air_index(&opts, "CPU[0]");
    let tampers: [(&str, fn(&mut VmProof, usize)); 4] = [
        ("main-trace path node", |p, i| {
            p.proof.proofs[i].deep_poly_openings[0]
                .main_trace_polys
                .proof
                .merkle_path[0][0] ^= 1
        }),
        ("FRI path node", |p, i| {
            p.proof.proofs[i].query_list[0].layers_auth_paths[0].merkle_path[1][5] ^= 1
        }),
        ("main root", |p, i| {
            p.proof.proofs[i].lde_trace_main_merkle_root[31] ^= 1
        }),
        ("nonce", |p, i| {
            p.proof.proofs[i].nonce = p.proof.proofs[i].nonce.map(|n| n ^ 1)
        }),
    ];
    for (what, tamper) in tampers {
        let mut bad = proof.clone();
        tamper(&mut bad, idx);
        assert!(!verify(&bad, BaseHash::P1), "a tampered {what} verified");
        println!("NOEPOCH P1 NEGATIVE {what} (CPU[0], index {idx}): refused");
    }
    println!(
        "NOEPOCH P1: fib_iterative_160k verifies under P1, refused under RPX; {} P1 device grinds",
        crypto::grinding::gpu_grind_calls_p1()
    );
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
/// walls, the peak RSS, one `NOEPOCH DIGEST` and one `NOEPOCH RESULT` line.
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
         grinding {} · base hash {:?}",
        elf_bytes.len(),
        input.len(),
        opts.blowup_factor,
        opts.fri_number_of_queries,
        opts.grinding_factor,
        crate::hash_pin::base_hash(),
    );
    // Before the clock: under the P1 base hash the static preprocessed roots
    // are computed once here (the RPX arm reads them as constants).
    crate::hash_pin::warm_base_statics(&opts);
    #[cfg(feature = "cuda")]
    crypto::grinding::reset_gpu_grind_calls();

    let (proof, times) = prove_block(&elf_bytes, &input, &opts).expect("the block must prove");
    #[cfg(feature = "cuda")]
    println!(
        "NOEPOCH GRINDS: device rpx {} · p1 {}",
        crypto::grinding::gpu_grind_calls_rpx(),
        crypto::grinding::gpu_grind_calls_p1()
    );
    let prove_peak = vm_hwm_gib();
    let sub_proofs = proof.proof.proofs.len();
    let bytes = proof_bytes(&proof);
    let size = bytes.len();
    // Two processes prove the same bytes under LAMBDA_VM_FIXED_TRACE_HASH=1 and
    // LAMBDA_VM_DETERMINISTIC_GRIND: compare these lines.
    println!(
        "NOEPOCH DIGEST: {} (blake3 of the {size} proof bytes)",
        &blake3::hash(&bytes).to_hex()[..32]
    );
    drop(bytes);

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
/// verifier uses; HINT, COMMIT and BLAKE3 stay at one table under both.
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
    for (name, set) in [
        ("hint", (|c| c.hint = 2) as fn(&mut crate::TableCounts)),
        ("commit", |c| c.commit = 2),
        ("blake3", |c| c.blake3 = 2),
    ] {
        let mut two = honest.clone();
        set(&mut two);
        assert!(
            two.validate_for(AcceleratorShape::BlockChunked).is_err(),
            "{name} stays one table under the block shape"
        );
    }
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

/// Every accelerator present, KECCAK, KECCAK_RND and ECSM in two chunks and
/// ECDAS in three: in proof order (`VmAirs::air_refs`) the five fixed tables,
/// COMMIT 5, KECCAK 6–7, KECCAK_RND 8–9, ECSM 10–11, ECDAS 12–14, HINT 15,
/// CPU 16–17, MEMW_R 18.
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
        keccak: 2,
        keccak_rnd: 2,
        ecsm: 2,
        ecdas: 3,
        hint: 1,
        commit: 1,
        blake3: 0,
    }
}

/// The block shape's accelerator rule for each chunked table, host only: two
/// or three tables of KECCAK, KECCAK_RND, ECSM or ECDAS pass the block shape,
/// and each one alone over one table is refused by the single shape every
/// other verifier uses; HINT stays at one table under both.
#[test]
fn each_chunked_accelerator_is_accepted_only_by_the_block_shape() {
    use crate::AcceleratorShape;
    let counts = chunked_counts();
    assert!(counts.validate_for(AcceleratorShape::BlockChunked).is_ok());
    let mut one_of_each = counts.clone();
    (one_of_each.keccak, one_of_each.keccak_rnd) = (1, 1);
    (one_of_each.ecsm, one_of_each.ecdas) = (1, 1);
    assert!(one_of_each.validate().is_ok(), "the control: one of each");
    for (name, set) in [
        ("KECCAK", (|c| c.keccak = 2) as fn(&mut crate::TableCounts)),
        ("KECCAK_RND", |c| c.keccak_rnd = 2),
        ("ECSM", |c| c.ecsm = 2),
        ("ECDAS", |c| c.ecdas = 3),
    ] {
        let mut chunked = one_of_each.clone();
        set(&mut chunked);
        assert!(
            chunked.validate_for(AcceleratorShape::BlockChunked).is_ok(),
            "the block shape takes {name} chunked"
        );
        assert!(
            chunked.validate().is_err(),
            "the single shape refuses {name} chunked"
        );
    }
    let mut two_hint = counts.clone();
    two_hint.hint = 2;
    assert!(
        two_hint
            .validate_for(AcceleratorShape::BlockChunked)
            .is_err(),
        "HINT stays one table under the block shape"
    );
}

/// ★ The block shape caps the chunked tables' heights, and only theirs: a
/// KECCAK, KECCAK_RND, ECSM or ECDAS instance over its cap
/// ([`crate::BLOCK_KECCAK_MAX_ROWS`], [`crate::BLOCK_KECCAK_RND_MAX_ROWS`],
/// [`crate::BLOCK_ECSM_MAX_ROWS`], [`crate::BLOCK_ECDAS_MAX_ROWS`]) is refused
/// by the block shape and accepted by the single shape (which bounds no
/// height); every other table at a height far over every cap is accepted. The
/// controls next to the ranges (COMMIT before KECCAK, HINT after ECDAS) are
/// what an off-by-one in the instance ranges would cap instead
/// (`block_chunked_ranges_name_the_chunked_airs` pins the ranges themselves).
#[test]
fn the_block_shape_caps_the_chunked_tables_heights() {
    use crate::AcceleratorShape::{BlockChunked, Single};
    let counts = chunked_counts();
    let n = counts.total().expect("fits") + crate::FIXED_TABLE_COUNT;
    assert_eq!(n, 19);
    let check = |shape, lengths: &[usize]| counts.check_heights_for(shape, |i| lengths[i]);
    let honest = vec![1usize << 10; n];
    assert!(check(BlockChunked, &honest).is_ok());

    let mut at_caps = honest.clone();
    at_caps[6..8].fill(crate::BLOCK_KECCAK_MAX_ROWS);
    at_caps[8..10].fill(crate::BLOCK_KECCAK_RND_MAX_ROWS);
    at_caps[10..12].fill(crate::BLOCK_ECSM_MAX_ROWS);
    at_caps[12..15].fill(crate::BLOCK_ECDAS_MAX_ROWS);
    assert!(
        check(BlockChunked, &at_caps).is_ok(),
        "an instance at its cap passes"
    );

    for (what, idx, rows) in [
        ("ECDAS[2] over its cap", 14, crate::BLOCK_ECDAS_MAX_ROWS * 2),
        ("ECDAS[0] over its cap", 12, crate::BLOCK_ECDAS_MAX_ROWS * 2),
        ("ECSM[1] over its cap", 11, crate::BLOCK_ECSM_MAX_ROWS * 2),
        ("ECSM[0] over its cap", 10, crate::BLOCK_ECSM_MAX_ROWS * 2),
        (
            "KECCAK_RND[1] over its cap",
            9,
            crate::BLOCK_KECCAK_RND_MAX_ROWS * 2,
        ),
        (
            "KECCAK_RND[0] over its cap",
            8,
            crate::BLOCK_KECCAK_RND_MAX_ROWS * 2,
        ),
        (
            "KECCAK[1] over its cap",
            7,
            crate::BLOCK_KECCAK_MAX_ROWS * 2,
        ),
        (
            "KECCAK[0] over its cap",
            6,
            crate::BLOCK_KECCAK_MAX_ROWS * 2,
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
        ("HINT", 15),
        ("CPU[0]", 16),
        ("MEMW_R[0]", 18),
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

/// The DEEP batching phase's proven bits for a table of `rows` rows whose DEEP
/// batch has `l` terms, at rate `1 / blowup`: BCHKS25 Thm 4.2 in the Johnson
/// regime, the gap at its floor (`α = 0`), batching by powers of one challenge
/// — ZisK's security calculator (pil2-proofman `regimes.rs`,
/// `JohnsonBoundRegime::error_powers`), the accounting of record.
fn deep_batching_bits(rows: usize, l: usize, blowup: usize) -> f64 {
    let p = ((1u128 << 64) - (1u128 << 32) + 1) as f64;
    let field = p * p * p;
    let rate = 1.0 / blowup as f64;
    let sqrt_rate = rate.sqrt();
    let gap = 1.0 / 300.0;
    let ms = (sqrt_rate / (2.0 * gap)).ceil().max(3.0) + 0.5;
    let n = rows as f64 / rate;
    let first = (2.0 * ms.powi(5) + 3.0 * ms * gap * rate) * n / (3.0 * rate * sqrt_rate);
    let second = ms / sqrt_rate;
    -((first + second) / field * (l as f64 - 1.0)).log2()
}

/// ★ Every chunked table keeps its DEEP batching phase above the block's
/// minimum of record at its cap, whatever height a prover declares under it:
/// the batch size `L` (composition parts + main and aux columns + next-row
/// openings) is read from the AIR the verifier builds, and the bits at the cap
/// are pinned to the campaign calculator's (`danyblock/bits.out`: KECCAK
/// 129.213 at 2^18, KECCAK_RND 129.429 at 2^16, ECSM 129.488 at 2^17, ECDAS
/// 129.907 at 2^17). The minimum of record is the query phase (110 queries,
/// 20 bits of grinding: 128.946), which no table height moves; one doubling
/// past each cap falls under it, so each cap is the largest that keeps it.
#[test]
fn every_chunked_table_keeps_its_batching_bits_at_its_cap() {
    let opts = crate::lfm::proof::block_base_options();
    let blowup = opts.blowup_factor as usize;
    assert_eq!(
        (blowup, opts.fri_number_of_queries, opts.grinding_factor),
        (4, 110, 20),
        "the block's posture of record"
    );
    let gap = 1.0 / 300.0;
    let query_bits = opts.grinding_factor as f64
        + opts.fri_number_of_queries as f64 * -((1.0 / blowup as f64).sqrt() + gap).log2();
    assert!(
        (query_bits - 128.946).abs() < 1e-3,
        "query phase {query_bits}"
    );
    let airs: [(&str, crate::test_utils::VmAir, usize, usize, f64); 4] = [
        (
            "KECCAK",
            Box::new(crate::test_utils::create_keccak_air(&opts)),
            crate::BLOCK_KECCAK_MAX_ROWS,
            581,
            129.213,
        ),
        (
            "KECCAK_RND",
            Box::new(crate::test_utils::create_keccak_rnd_air(&opts)),
            crate::BLOCK_KECCAK_RND_MAX_ROWS,
            1999,
            129.429,
        ),
        (
            "ECSM",
            Box::new(crate::test_utils::create_ecsm_air(&opts)),
            crate::BLOCK_ECSM_MAX_ROWS,
            960,
            129.488,
        ),
        (
            "ECDAS",
            Box::new(crate::test_utils::create_ecdas_air(&opts)),
            crate::BLOCK_ECDAS_MAX_ROWS,
            718,
            129.907,
        ),
    ];
    for (name, air, cap, pinned_l, pinned_bits) in airs {
        let (main, aux) = air.trace_layout();
        let parts = air.composition_poly_degree_bound(cap) / cap;
        let l = parts + main + aux + air.trace_ood_next_row_columns().len();
        assert_eq!(l, pinned_l, "{name}: the DEEP batch size moved");
        let at_cap = deep_batching_bits(cap, l, blowup);
        let past_cap = deep_batching_bits(2 * cap, l, blowup);
        assert!(
            (at_cap - pinned_bits).abs() < 1e-3,
            "{name}: {at_cap:.3} bits at its cap, the calculator reads {pinned_bits}"
        );
        assert!(
            at_cap >= query_bits && at_cap >= 128.0,
            "{name}: {at_cap:.3} bits at its cap, under the minimum of record"
        );
        assert!(
            past_cap < query_bits,
            "{name}: {past_cap:.3} bits at twice its cap; a larger cap would do"
        );
        println!(
            "BLOCK CAP BITS {name}: L {l}, 2^{} rows {at_cap:.3} bits, 2^{} {past_cap:.3} \
             (minimum of record {query_bits:.3})",
            cap.trailing_zeros(),
            cap.trailing_zeros() + 1
        );
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

/// A chunked one-call-per-row accelerator of the box tests: KECCAK (one row
/// per permutation call) or ECSM (one row per scalar multiplication call).
#[derive(Clone, Copy)]
enum CallTable {
    Keccak,
    Ecsm,
}

impl CallTable {
    fn name(self) -> &'static str {
        match self {
            CallTable::Keccak => "KECCAK",
            CallTable::Ecsm => "ECSM",
        }
    }

    /// The guest that calls it three times: test_keccak_multi's three
    /// permutations, test_ecsm_multi's three scalar multiplications (k = 1, 5,
    /// 0xABCDEF).
    fn guest(self) -> &'static str {
        match self {
            CallTable::Keccak => "test_keccak_multi",
            CallTable::Ecsm => "test_ecsm_multi",
        }
    }

    /// The block's caps with this table cut every `calls` calls.
    fn chunked(self, calls: usize) -> MaxRowsConfig {
        match self {
            CallTable::Keccak => MaxRowsConfig {
                keccak: calls,
                ..MaxRowsConfig::default()
            },
            CallTable::Ecsm => MaxRowsConfig {
                ecsm: calls,
                ..MaxRowsConfig::default()
            },
        }
    }

    fn tables(self, traces: &mut Traces) -> &mut Vec<stark::trace::TraceTable<FieldF, FieldE>> {
        match self {
            CallTable::Keccak => &mut traces.keccaks,
            CallTable::Ecsm => &mut traces.ecsms,
        }
    }

    fn count(self, counts: &crate::TableCounts) -> usize {
        match self {
            CallTable::Keccak => counts.keccak,
            CallTable::Ecsm => counts.ecsm,
        }
    }

    /// The table's width and its MU and timestamp columns.
    fn columns(self) -> (usize, usize, [usize; 2]) {
        use crate::tables::{ecsm, keccak};
        match self {
            CallTable::Keccak => (
                keccak::cols::NUM_COLUMNS,
                keccak::cols::MU,
                [keccak::cols::TIMESTAMP_0, keccak::cols::TIMESTAMP_1],
            ),
            CallTable::Ecsm => (
                ecsm::cols::NUM_COLUMNS,
                ecsm::cols::MU,
                [ecsm::cols::TIMESTAMP_0, ecsm::cols::TIMESTAMP_1],
            ),
        }
    }

    /// The block format's cap on one instance.
    fn cap(self) -> usize {
        match self {
            CallTable::Keccak => crate::BLOCK_KECCAK_MAX_ROWS,
            CallTable::Ecsm => crate::BLOCK_ECSM_MAX_ROWS,
        }
    }
}

type FieldF = crate::tables::types::GoldilocksField;
type FieldE = crate::tables::types::GoldilocksExtension;

/// ★ KECCAK and ECSM chunked: three calls at one call a chunk give three
/// instances (and KECCAK at two calls a chunk two). The block verifier accepts
/// each proof; the single-shape verifier — every other verifier's — refuses
/// the same bytes, at a count of 2 as at 3 (the shape is the verifier's
/// constant, never read from the proof). Retain and RecomputeLdeDevice agree
/// byte for byte.
#[test]
#[ignore = "proves three VM programs twice each at blowup 4; GPU box gate (cuda)"]
fn noepoch_keccak_and_ecsm_chunked_prove_and_verify() {
    let opts = bytes_options();
    for (table, calls, instances) in [
        (CallTable::Keccak, 1, 3),
        (CallTable::Keccak, 2, 2),
        (CallTable::Ecsm, 1, 3),
    ] {
        let name = table.name();
        let build = OneBuild::new(table.guest(), &table.chunked(calls));
        assert_eq!(
            table.count(&build.traces.table_counts()),
            instances,
            "{name}: {calls} call(s) a chunk"
        );
        let retained = build
            .prove(&opts, ResidencyMode::Retain, false)
            .expect("prove under Retain");
        let recommitted = build
            .prove(&opts, ResidencyMode::RecomputeLdeDevice, true)
            .expect("prove under RecomputeLdeDevice");
        assert!(
            build.verifies(&recommitted, &opts),
            "{name}: the block verifier accepts {instances} instances"
        );
        assert!(
            !build.verifies_single_shape(&recommitted, &opts),
            "{name}: the single-shape verifier must refuse {instances} tables"
        );
        assert!(proof_bytes(&retained) == proof_bytes(&recommitted));
        println!(
            "NOEPOCH {name} CHUNKED: {instances} instances, block verifier accepts, single shape \
             refuses, bytes equal"
        );
    }
}

/// ★ A KECCAK or ECSM row is one whole call, so a chunk boundary falls between
/// calls, and a call reaches its rounds or steps, its scalar bits and its
/// memory only through buses keyed by its timestamp.
///
/// - Control: the first call of chunk 0 and the first call of chunk 1 swapped
///   as whole rows is the same multiset of rows, and verifies: the boundary
///   carries nothing a bus does not.
/// - Negative: the same two rows swap only their timestamps, so each claims
///   the other's call. No constraint of either table reads the timestamp, so
///   both rows still satisfy their constraints; only the bus tuples moved
///   (the Ecall receive still balances: both timestamps are still received
///   once), and the block is refused.
#[test]
#[ignore = "proves two VM programs three times each at blowup 4; GPU box gate (cuda)"]
fn noepoch_keccak_and_ecsm_rows_are_keyed_on_the_bus() {
    let opts = bytes_options();
    for table in [CallTable::Keccak, CallTable::Ecsm] {
        let name = table.name();
        let build = OneBuild::new(table.guest(), &table.chunked(1));
        let (width, mu, ts_cols) = table.columns();
        let one = crate::tables::types::FE::one();
        let mut traces = build.traces.clone();
        let chunks = table.tables(&mut traces);
        assert_eq!(chunks.len(), 3, "{name}: one call a chunk");
        let ts =
            |t: &stark::trace::TraceTable<FieldF, FieldE>| ts_cols.map(|c| *t.main_table.get(0, c));
        assert_eq!(*chunks[0].main_table.get(0, mu), one);
        assert_eq!(*chunks[1].main_table.get(0, mu), one);
        assert_ne!(
            ts(&chunks[0]),
            ts(&chunks[1]),
            "{name}: two different calls"
        );

        assert!(
            block_accepts(&build, &build.traces, &opts),
            "{name}: the honest chunked build verifies"
        );

        let swap = |traces: &mut Traces, cols: &[usize]| {
            let chunks = table.tables(traces);
            for &col in cols {
                let x = *chunks[0].main_table.get(0, col);
                let y = *chunks[1].main_table.get(0, col);
                chunks[0].main_table.set(0, col, y);
                chunks[1].main_table.set(0, col, x);
            }
        };
        let mut swapped = build.traces.clone();
        swap(&mut swapped, &(0..width).collect::<Vec<_>>());
        assert!(
            block_accepts(&build, &swapped, &opts),
            "{name}: whole calls swapped across chunks are the same calls: accepted"
        );
        println!("NOEPOCH {name} SPLIT CONTROL: whole rows swapped across chunks, accepted");

        let mut rekeyed = build.traces.clone();
        swap(&mut rekeyed, &ts_cols);
        assert!(
            !block_accepts(&build, &rekeyed, &opts),
            "{name}: a call reattached to another call's timestamp must be refused"
        );
        println!("NOEPOCH {name} SPLIT NEGATIVE: chunk 0 and chunk 1 swapped timestamps, refused");
    }
}

/// ★ The KECCAK and ECSM height caps end to end: the guest's one table padded
/// to twice its cap (a valid table: the extra rows are padding) proves; the
/// block verifier refuses it (an instance over [`crate::BLOCK_KECCAK_MAX_ROWS`]
/// / [`crate::BLOCK_ECSM_MAX_ROWS`]) and the single-shape verifier, which
/// bounds no height, accepts the same bytes.
///
/// Mutation (box script): `check_heights_for` returning `Ok(())` and the block
/// verifier accepts.
#[test]
#[ignore = "proves a 2^19-row KECCAK and a 2^18-row ECSM at blowup 4; GPU box gate (cuda)"]
fn noepoch_keccak_and_ecsm_over_their_caps_are_refused() {
    let opts = bytes_options();
    for table in [CallTable::Keccak, CallTable::Ecsm] {
        let name = table.name();
        let build = OneBuild::new(table.guest(), &MaxRowsConfig::default());
        let (_, mu, _) = table.columns();
        let mut traces = build.traces.clone();
        let tables = table.tables(&mut traces);
        assert_eq!(tables.len(), 1, "{name}: one table at the default caps");
        let (rows, width) = (tables[0].num_rows(), tables[0].main_table.width);
        assert!(rows < table.cap());
        let padding = tables[0].main_table.get_row(rows - 1).to_vec();
        assert_eq!(
            padding[mu],
            crate::tables::types::FE::zero(),
            "{name}: the last row is padding"
        );
        let tall = 2 * table.cap();
        let mut data = Vec::with_capacity(tall * width);
        for r in 0..rows {
            data.extend_from_slice(tables[0].main_table.get_row(r));
        }
        for _ in rows..tall {
            data.extend_from_slice(&padding);
        }
        tables[0] = stark::trace::TraceTable::new_main(data, width, 1);
        let proof = build
            .prove_these(&traces, &opts, ResidencyMode::Retain, false)
            .unwrap_or_else(|e| panic!("{name}: a padded table is a valid table: {e:?}"));
        assert!(
            build.verifies_single_shape(&proof, &opts),
            "{name}: the single shape bounds no height: the proof itself is sound"
        );
        assert!(
            !build.verifies(&proof, &opts),
            "{name}: the block verifier must refuse an instance over its cap"
        );
        println!(
            "NOEPOCH {name} OVER CAP: 2^{} rows, single shape accepts, block verifier refuses",
            tall.trailing_zeros()
        );
    }
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

/// The streamed phase A's traces are the serial build's under the block's own
/// caps (KECCAK, KECCAK_RND and ECSM chunked, which the shared builder's tests
/// do not cover):
/// windows of one CPU instance fed to [`WindowedTraceBuilder`], the last to
/// `finish`, the chunks put back with `insert_streamed`, equal
/// `Traces::from_elf_and_logs` table by table, chunk by chunk and row by row
/// (the six that lay rows out in `HashMap` order too: one hash state per
/// process, `tables::trace_hash`). Both of `build_streamed`'s schedules:
/// `push`, and the split builder (walk, then absorb, the jobs generated by the
/// caller). Host only, small programs.
#[test]
fn windowed_collection_builds_the_serial_traces() {
    use crate::tables::trace_builder::WindowedTraceBuilder;
    let chunked_keccak = MaxRowsConfig {
        keccak: 1,
        keccak_rnd: 48,
        ..MaxRowsConfig::small()
    };
    let chunked_ecsm = MaxRowsConfig {
        ecsm: 1,
        ..MaxRowsConfig::small()
    };
    for (name, max_rows) in [
        ("all_instructions_64", MaxRowsConfig::small()),
        ("test_keccak_multi", chunked_keccak),
        ("test_ecsm_multi", chunked_ecsm),
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
                pages,
                lts,
                branches,
                muls,
                dvrms,
                eqs,
                bytewises
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
            assert_eq!(serial.public_output_bytes, streamed.public_output_bytes);
            assert_eq!(serial.table_counts().cpu, streamed.table_counts().cpu);
            assert!(n_streamed > 0, "{name}: no chunk was streamed");
            println!(
                "{name}: windowed = serial ({} windows, {n_streamed} streamed chunks, {} KECCAK, {} \
                 KECCAK_RND, {} ECSM)",
                windows.len(),
                streamed.keccaks.len(),
                streamed.keccak_rnds.len(),
                streamed.ecsms.len()
            );
        }
    }
}
