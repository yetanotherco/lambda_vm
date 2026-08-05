//! End-to-end regression tests for two ways a prover could break the memory
//! argument's one-genesis-token-per-address invariant. Both were demonstrated as
//! working forgeries against `origin/main` (b082f9f6) and are now closed.
//!
//! **Route 1 — free `OFFSET` (arbitrary byte, arbitrary address).** A
//! private-input PAGE's `OFFSET` was a free main-trace column: `create_page_air`
//! builds PAGE with `EmptyConstraints`, no constraint references `cols::OFFSET`,
//! and `VmAirs::new` skipped `with_preprocessed` for `is_private_input` pages. The
//! Memory-bus address is `address_lo = page_base_lo + OFFSET`, so a row could be
//! pointed at any address sharing the page's high limb. Closed by preprocessing
//! `OFFSET` (only — `INIT` is the private input and stays main-trace).
//!
//! **Route 2 — duplicate page coverage (forces a chosen address to read `0`).**
//! Survived route 1's fix, and needs no private input at all. Nothing is forged at
//! the commitment layer: the prover declares a `runtime_page_ranges` entry over an
//! address the ELF already covers, and the injected zero-init page's `OFFSET`
//! *and* `INIT` match the shipped static zero-page commitment exactly. The address
//! then has two genesis tokens, and the two pages' rows swap which one each
//! consumes. Closed by rejecting duplicate page bases during the verifier's layout
//! reconstruction.
//!
//! The one-line distinction: preprocessing `OFFSET` restores "one row per address
//! *within* a page"; the duplicate-base check restores "one page per address".
//! Both are needed.
//!
//! The guest loads 8 bytes out of its own ELF `.data`, spills them to the stack
//! and commits them, so the proof's `public_output` is a direct function of the
//! ELF image — which the verifier binds via that data page's preprocessed
//! commitment. Each forgery's claim is that the proof verifies against the
//! *unmodified* ELF while reporting a different output.
//!
//! Run under **production** proof options, not `default_test_options()`, so none
//! of this can be written off as an artefact of a low-query configuration.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use stark::proof::options::ProofOptions;
use stark::prover::{IsStarkProver, Prover};

use crate::statement::{StatementKind, absorb_statement};
use crate::tables::bitwise::{cols as bw_cols, row_index as bw_row_index};
use crate::tables::page::cols as page_cols;
use crate::tables::trace_builder::Traces;
use crate::tables::types::{FE, VmTable};
use crate::test_utils::{E, asm_elf_bytes};
use crate::{MaxRowsConfig, VmAirs, VmProof};

use executor::elf::Elf;
use executor::vm::execution::Executor;

/// The 8 bytes the PoC guest keeps in `.data` (little-endian `.dword`).
const SECRET: [u8; 8] = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];

/// The byte we forge in its place.
const FORGED_BYTE: u8 = 0xEE;

/// The PRODUCTION options: exactly what the public `crate::verify` uses
/// (`GoldilocksCubicProofOptions::with_blowup(2)`, 128-bit security target).
/// Deliberately not `default_test_options()` — nobody should be able to write
/// this off as an artefact of a 3-query toy configuration.
fn opts() -> ProofOptions {
    crate::GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 is valid")
}

/// Raw-file offset of `SECRET` inside the ELF, plus the virtual address that
/// offset maps to (via the containing PT_LOAD program header).
fn locate_secret(elf_bytes: &[u8]) -> (usize, u64) {
    let file_off = elf_bytes
        .windows(SECRET.len())
        .position(|w| w == SECRET)
        .expect("SECRET pattern not found in ELF");

    let rd_u16 = |o: usize| u16::from_le_bytes(elf_bytes[o..o + 2].try_into().unwrap());
    let rd_u32 = |o: usize| u32::from_le_bytes(elf_bytes[o..o + 4].try_into().unwrap());
    let rd_u64 = |o: usize| u64::from_le_bytes(elf_bytes[o..o + 8].try_into().unwrap());

    let e_phoff = rd_u64(32) as usize;
    let e_phentsize = rd_u16(54) as usize;
    let e_phnum = rd_u16(56) as usize;
    const PT_LOAD: u32 = 1;

    for i in 0..e_phnum {
        let ph = e_phoff + i * e_phentsize;
        if rd_u32(ph) != PT_LOAD {
            continue;
        }
        let p_offset = rd_u64(ph + 8) as usize;
        let p_vaddr = rd_u64(ph + 16);
        let p_filesz = rd_u64(ph + 32) as usize;
        if file_off >= p_offset && file_off + SECRET.len() <= p_offset + p_filesz {
            return (file_off, p_vaddr + (file_off - p_offset) as u64);
        }
    }
    panic!("SECRET is not inside any PT_LOAD segment");
}

/// One repointed private-input PAGE row.
struct Forge {
    /// The address whose genesis byte we overwrite.
    target_addr: u64,
    /// The byte the forged init token carries.
    forged: u8,
    /// The byte the honest (preprocessed-bound) init token carries; the
    /// repointed row's PAGE-C4 consumes it so the bus still balances.
    real: u8,
}

/// How the malicious prover deviates from an honest trace.
enum Tamper {
    /// Repoint one private-input PAGE row (the hole under test).
    RepointPrivateRow(Forge),
    /// Overwrite the target byte's INIT directly on its own ELF-data PAGE.
    /// This is the "obvious" attack, and it is the CONTROL: that page IS
    /// preprocessed, so its INIT column is pinned by a per-page Merkle root
    /// recomputed by the verifier from the ELF. It must be rejected.
    DirectInitOnHonestPage { target_addr: u64, forged: u8 },
}

/// A malicious prover. Everything is the production pipeline; the only
/// deviations are (a) the execution logs may come from a different ELF than
/// the one whose identity/preprocessed roots are used, and (b) `forge`
/// rewrites one PAGE row.
fn craft_proof(
    honest_elf: &[u8],
    run_elf: &[u8],
    private_inputs: &[u8],
    forge: Option<Tamper>,
) -> Result<VmProof, stark::prover::ProvingError> {
    let options = opts();

    // Identity + all preprocessed roots come from the HONEST ELF.
    let program = Elf::load(honest_elf).expect("honest ELF load");

    // Execution logs come from whatever `run_elf` is.
    let run_program = Elf::load(run_elf).expect("run ELF load");
    let executor =
        Executor::new(&run_program, private_inputs.to_vec()).expect("executor construction");
    let result = executor.run().expect("run");

    let max_rows = MaxRowsConfig::default();
    let mut traces = Traces::from_elf_and_logs(
        &program,
        &result.logs,
        &max_rows,
        private_inputs,
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .expect("trace build");

    match forge {
        Some(Tamper::RepointPrivateRow(f)) => apply_forge(&mut traces, &f),
        Some(Tamper::DirectInitOnHonestPage {
            target_addr,
            forged,
        }) => apply_direct_init_tamper(&mut traces, target_addr, forged),
        None => {}
    }

    let table_counts = traces.table_counts();
    let airs = VmAirs::new(
        &program,
        &options,
        false,
        &traces.page_configs,
        &table_counts,
        None,
        true,
        None,
        None,
        None,
    );

    let runtime_page_ranges = traces.runtime_page_ranges();
    let num_private_input_pages = traces
        .page_configs
        .iter()
        .filter(|c| c.is_private_input)
        .count();

    let mut transcript = DefaultTranscript::<E>::new(&[]);
    absorb_statement(
        &mut transcript,
        StatementKind::Monolithic,
        honest_elf,
        &traces.public_output_bytes,
        &table_counts,
        num_private_input_pages,
        &runtime_page_ranges,
        options.fri_final_poly_log_degree,
    );

    let proof = Prover::multi_prove(
        airs.air_trace_pairs(&mut traces),
        &mut transcript,
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )?;

    Ok(VmProof {
        proof,
        runtime_page_ranges,
        table_counts,
        public_output: traces.public_output_bytes.clone(),
        num_private_input_pages,
    })
}

/// Repoint one unused private-input PAGE row at `f.target_addr` so that it
/// PROVIDES `(0, target, ts=0, forged)` on the Memory bus and CONSUMES the
/// honest `(0, target, ts=0, real)` token in its place.
fn apply_forge(traces: &mut Traces, f: &Forge) {
    let (page_idx, page_base) = traces
        .page_configs
        .iter()
        .enumerate()
        .find(|(_, c)| c.is_private_input)
        .map(|(i, c)| (i, c.page_base))
        .expect("a private-input page must exist");

    assert_eq!(
        page_base >> 32,
        f.target_addr >> 32,
        "address_hi is a constant per page, so the target must share it"
    );

    // Any private-input byte the guest never reads. Row 4096 is well past the
    // 4-byte length prefix and the (tiny) input payload.
    let row = 4096usize;
    {
        let page = &traces.pages[page_idx].main_table;
        assert_eq!(*page.get(row, page_cols::INIT), FE::zero());
        assert_eq!(*page.get(row, page_cols::FINI), FE::zero());
        assert_eq!(*page.get(row, page_cols::TIMESTAMP_LO), FE::zero());
        assert_eq!(*page.get(row, page_cols::TIMESTAMP_HI), FE::zero());
        assert_eq!(*page.get(row, page_cols::OFFSET), FE::from(row as u64));
    }

    let page = &mut traces.pages[page_idx].main_table;
    // address_lo = page_base_lo + OFFSET  ⇒  OFFSET = target - page_base (in F_p).
    page.set(
        row,
        page_cols::OFFSET,
        FE::from(f.target_addr) - FE::from(page_base),
    );
    page.set_byte(row, page_cols::INIT, f.forged);
    page.set_byte(row, page_cols::FINI, f.real);
    // TIMESTAMP stays 0: PAGE-C4 then consumes the honest genesis token, which
    // PAGE-C3 hardcodes at ts = 0.

    // The row's ARE_BYTES[init, fini] send moved from (0, 0) to (forged, real);
    // rebalance the BITWISE receiver multiplicities to match.
    move_are_bytes_multiplicity(traces, (0, 0), (f.forged, f.real));
}

/// Move one unit of `MU_ARE_BYTES` from the pair `from` to the pair `to`, so
/// the ARE_BYTES bus stays balanced after a PAGE row's `(init, fini)` changed.
fn move_are_bytes_multiplicity(traces: &mut Traces, from: (u8, u8), to: (u8, u8)) {
    let bw = &mut traces.bitwise.main_table;
    let dec = bw_row_index(from.0, from.1, 0);
    let inc = bw_row_index(to.0, to.1, 0);
    assert_ne!(dec, inc);
    let old_dec = *bw.get(dec, bw_cols::MU_ARE_BYTES);
    assert_ne!(old_dec, FE::zero(), "source pair must have multiplicity");
    bw.set(dec, bw_cols::MU_ARE_BYTES, old_dec - FE::one());
    let old_inc = *bw.get(inc, bw_cols::MU_ARE_BYTES);
    bw.set(inc, bw_cols::MU_ARE_BYTES, old_inc + FE::one());
}

/// CONTROL tamper: rewrite the target byte's INIT on its own (preprocessed)
/// ELF-data PAGE. The Memory bus balances perfectly afterwards — the page
/// simply provides the forged genesis token that MEMW consumes — so if this is
/// rejected, the rejection can only come from the preprocessed commitment.
fn apply_direct_init_tamper(traces: &mut Traces, target_addr: u64, forged: u8) {
    use crate::tables::page::{offset_in_page, page_base_for_address};

    let base = page_base_for_address(target_addr);
    let offset = offset_in_page(target_addr);
    let page_idx = traces
        .page_configs
        .iter()
        .position(|c| c.page_base == base)
        .expect("target page must exist");
    assert!(
        !traces.page_configs[page_idx].is_private_input,
        "the control must target an ELF-data page, not the private page"
    );
    assert!(
        traces.page_configs[page_idx].init_values.is_some(),
        "the control must target a page whose INIT is ELF-derived and committed"
    );

    let (old_init, fini) = {
        let page = &traces.pages[page_idx].main_table;
        let byte_at = |col: usize| -> u8 {
            u8::try_from(page.get(offset, col).to_raw()).expect("column holds a byte")
        };
        (byte_at(page_cols::INIT), byte_at(page_cols::FINI))
    };
    traces.pages[page_idx]
        .main_table
        .set_byte(offset, page_cols::INIT, forged);
    move_are_bytes_multiplicity(traces, (old_init, fini), (forged, fini));
}

/// Did the verifier accept this proof?
///
/// A rejection now arrives in two shapes: `Ok(false)` when a check inside the
/// STARK verification fails, and `Err(MalformedPageLayout)` when the page layout
/// is refused before any proof is checked at all. Both mean "not accepted", and
/// collapsing them here keeps the tests from having to care which fired.
fn verifier_accepts(proof: &VmProof, elf: &[u8]) -> bool {
    match crate::verify_with_options(proof, elf, &opts(), None, None) {
        Ok(accepted) => accepted,
        Err(crate::Error::MalformedPageLayout(_)) => false,
        Err(e) => panic!("verification failed for an unexpected reason: {e}"),
    }
}

// =============================================================================
// Tests
// =============================================================================

/// Sanity: the guest commits its own `.data` bytes, and the harness used
/// honestly produces a genuinely valid proof. Guards against a vacuous PoC.
#[test]
fn poc_control_honest_harness_verifies() {
    let elf = asm_elf_bytes("poc_rodata_commit");
    let proof = craft_proof(&elf, &elf, &[0u8], None).expect("honest proving must succeed");
    assert_eq!(
        proof.public_output,
        SECRET.to_vec(),
        "guest must commit its .data bytes"
    );
    assert!(
        verifier_accepts(&proof, &elf),
        "honest use of the harness must verify"
    );
    assert_eq!(
        proof.num_private_input_pages, 1,
        "one byte of private input must create exactly one private page"
    );
}

/// NEGATIVE CONTROL: run the patched program but do NOT repoint a PAGE row.
/// The genesis token the MEMW chain consumes at `secret` then has no provider
/// (the honest page provides the real byte), so the bus must not balance.
#[test]
fn poc_negative_control_forged_run_without_repointed_row_fails() {
    let honest = asm_elf_bytes("poc_rodata_commit");
    let (file_off, _addr) = locate_secret(&honest);
    let mut patched = honest.clone();
    patched[file_off] = FORGED_BYTE;

    let proof = craft_proof(&honest, &patched, &[0u8], None)
        .expect("the patched run still proves; the verifier must be the one to reject it");
    assert_eq!(
        proof.public_output[0], FORGED_BYTE,
        "the patched run must commit the forged byte"
    );
    assert!(
        !verifier_accepts(&proof, &honest),
        "without the repointed PAGE row this proof must be rejected"
    );
}

/// REGRESSION (route 1 — free `OFFSET`): repointing a private-input PAGE row at
/// an arbitrary address must not produce a verifying proof.
///
/// On `origin/main` this was ACCEPTED against the unmodified ELF while claiming a
/// `public_output` the program cannot produce. `VmAirs::new` now preprocesses
/// `OFFSET`, so the repointed column no longer matches the commitment.
///
/// The forgery can die at either of two layers and which one fires depends on
/// process state, so both are accepted. `commit_main_trace` caches precomputed
/// Merkle trees keyed by *the expected root* and skips the rebuild check on a hit
/// (`crypto/stark/src/prover.rs:1161-1170`): with a cold cache the prover itself
/// refuses, with a warm one it substitutes the correct cached tree and leaves the
/// verifier to reject. Asserting only one would make this pass or fail on test
/// ordering.
#[test]
fn forged_private_page_offset_is_rejected() {
    let honest = asm_elf_bytes("poc_rodata_commit");
    let (file_off, addr) = locate_secret(&honest);
    let mut patched = honest.clone();
    patched[file_off] = FORGED_BYTE;

    let crafted = craft_proof(
        &honest,
        &patched,
        &[0u8],
        Some(Tamper::RepointPrivateRow(Forge {
            target_addr: addr,
            forged: FORGED_BYTE,
            real: SECRET[0],
        })),
    );

    let proof = match crafted {
        Err(e) => {
            assert!(
                matches!(
                    e,
                    stark::prover::ProvingError::PrecomputedCommitmentMismatch
                ),
                "the repointed trace must be refused for the OFFSET commitment, not \
                 for some unrelated proving error: {e:?}"
            );
            return;
        }
        Ok(proof) => proof,
    };

    // Non-vacuity: the proof really does claim the forged byte.
    assert_eq!(
        proof.public_output[0], FORGED_BYTE,
        "forged proof must claim the forged byte"
    );
    assert_ne!(proof.public_output, SECRET.to_vec());

    assert!(
        !verifier_accepts(&proof, &honest),
        "SOUNDNESS REGRESSION: the verifier accepted a proof whose public output the \
         program cannot produce — a private-input PAGE row was repointed via its \
         OFFSET column. OFFSET must stay preprocessed (see `VmAirs::new`)."
    );
}

/// SECOND NEGATIVE CONTROL — isolates the defense being bypassed.
///
/// Same forged execution, but instead of repointing a private-input row we
/// overwrite INIT directly on the target byte's own ELF-data PAGE. The Memory
/// bus balances perfectly this way (that page simply provides the forged
/// genesis token MEMW consumes), so the ONLY thing that can reject it is that
/// page's preprocessed commitment, which the verifier recomputes from the ELF.
///
/// It is rejected — which is the point: the preprocessed commitment does its
/// job on ELF-data pages. The private-input page is the sole bypass, precisely
/// because `VmAirs::new` gives it no commitment at all.
#[test]
fn poc_negative_control_direct_init_tamper_on_preprocessed_page_fails() {
    let honest = asm_elf_bytes("poc_rodata_commit");
    let (file_off, addr) = locate_secret(&honest);
    let mut patched = honest.clone();
    patched[file_off] = FORGED_BYTE;

    let proof = craft_proof(
        &honest,
        &patched,
        &[0u8],
        Some(Tamper::DirectInitOnHonestPage {
            target_addr: addr,
            forged: FORGED_BYTE,
        }),
    )
    .expect("this tamper leaves OFFSET alone, so the prover still builds it");
    assert_eq!(proof.public_output[0], FORGED_BYTE);

    assert!(
        !verifier_accepts(&proof, &honest),
        "the preprocessed commitment must reject a direct INIT rewrite"
    );
}

/// REACHABILITY on the workload that matters.
///
/// The ethrex block guest reads its ENTIRE `ProgramInput` through
/// `get_private_input()` (`executor/programs/rust/ethrex/src/main.rs:8`), so
/// every real block proof carries private-input pages. This asserts it through
/// the production function itself — `private_input_page_count` is what the
/// trace builder uses to classify pages (`trace_builder.rs:2615`) and what the
/// verifier's `num_private_input_pages` is compared against.
///
/// Each such page contributes 2^18 = 262,144 rows whose `OFFSET` is free.
#[test]
fn poc_real_ethrex_inputs_produce_private_input_pages() {
    use crate::tables::page::{DEFAULT_PAGE_SIZE, private_input_page_count};

    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .to_path_buf();

    let mut checked = 0usize;
    for name in [
        "ethrex_empty_block",
        "ethrex_5_transfers",
        "ethrex_10_transfers",
        "ethrex_bench_4",
    ] {
        let path = root.join(format!("executor/tests/{name}.bin"));
        let Ok(bytes) = std::fs::read(&path) else {
            continue; // fixture not present in this checkout
        };
        let pages = private_input_page_count(&bytes);
        println!(
            "{name}: {} bytes -> {pages} private-input page(s) = {} free-OFFSET rows",
            bytes.len(),
            pages * DEFAULT_PAGE_SIZE
        );
        assert!(
            pages > 0,
            "{name} must produce at least one private-input page"
        );
        checked += 1;
    }
    assert!(checked > 0, "no ethrex fixture found to check");

    // Sanity on the classifier: page 0 of that span is classified private.
    assert!(crate::tables::page::is_private_input_page(
        executor::vm::memory::PRIVATE_INPUT_START_INDEX,
        1
    ));
}

// =============================================================================
// SECOND ROUTE: duplicate page coverage — survives the OFFSET fix
// =============================================================================
//
// Pinning OFFSET restores "one row per address WITHIN a page". It does not
// restore "one page per address". `page_configs_from_elf_and_runtime`
// (`trace_builder.rs:4149-4171`) builds a Vec, appends one zero-init config per
// entry of the prover-supplied `runtime_page_ranges`, sorts by page_base, and
// never dedupes; `verify_proof_parts` validates `table_counts` and
// `num_private_input_pages` and passes `runtime_page_ranges` through untouched.
// So a prover can declare a second, zero-init page over an address the ELF
// already covers. Nothing is forged at the commitment layer — the injected page
// is an ordinary zero page whose OFFSET *and* INIT match the shipped static
// zero-page commitment — yet the address now has two genesis tokens.

/// Inject a duplicate zero-init PAGE over `base`, which an ELF-data page
/// already covers. When `consume` is `Some((offset, real))`, that row is set to
/// consume the ELF page's genesis token `(base+offset, ts=0, real)`; otherwise
/// every row self-cancels.
fn inject_duplicate_zero_page(traces: &mut Traces, base: u64, consume: Option<(usize, u8)>) {
    use crate::tables::page::{DEFAULT_PAGE_SIZE, PageConfig, generate_page_trace_from_dense};

    // Insert directly after the ELF config for `base`, matching the verifier's
    // STABLE `sort_by_key(page_base)` — ELF configs are pushed before runtime
    // ones, so the ELF page wins the tie.
    let elf_idx = traces
        .page_configs
        .iter()
        .position(|c| c.page_base == base)
        .expect("an ELF page for this base must already exist");
    assert!(
        traces.page_configs[elf_idx].init_values.is_some(),
        "duplicate must shadow an ELF-data page"
    );

    let dup_cfg = PageConfig::zero_init(base);
    let mut dup_trace = generate_page_trace_from_dense(&dup_cfg, None, false);
    if let Some((offset, real)) = consume {
        dup_trace.main_table.set_byte(offset, page_cols::FINI, real);
    }
    traces.page_configs.insert(elf_idx + 1, dup_cfg);
    traces.pages.insert(elf_idx + 1, dup_trace);

    // The injected table sends ARE_BYTES[init, fini] on every row: (0,0)
    // throughout, except the one compensating row (0, real).
    let bw = &mut traces.bitwise.main_table;
    let mut bump = |x: u8, y: u8, n: u64| {
        let row = bw_row_index(x, y, 0);
        let cur = *bw.get(row, bw_cols::MU_ARE_BYTES);
        bw.set(row, bw_cols::MU_ARE_BYTES, cur + FE::from(n));
    };
    match consume {
        Some((_, real)) => {
            bump(0, 0, (DEFAULT_PAGE_SIZE - 1) as u64);
            bump(0, real, 1);
        }
        None => bump(0, 0, DEFAULT_PAGE_SIZE as u64),
    }
}

/// Like `craft_proof`, but injects a duplicate zero page over `dup_base` after
/// the traces are built. Production prove path otherwise.
fn craft_proof_with_duplicate_page(
    honest_elf: &[u8],
    run_elf: &[u8],
    dup_base: u64,
    consume: Option<(usize, u8)>,
) -> VmProof {
    let options = opts();
    let program = Elf::load(honest_elf).expect("honest ELF load");
    let run_program = Elf::load(run_elf).expect("run ELF load");
    let executor = Executor::new(&run_program, vec![]).expect("executor construction");
    let result = executor.run().expect("run");

    let max_rows = MaxRowsConfig::default();
    let mut traces = Traces::from_elf_and_logs(
        &program,
        &result.logs,
        &max_rows,
        &[],
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .expect("trace build");

    inject_duplicate_zero_page(&mut traces, dup_base, consume);

    let table_counts = traces.table_counts();
    let runtime_page_ranges = traces.runtime_page_ranges();
    let num_private_input_pages = traces
        .page_configs
        .iter()
        .filter(|c| c.is_private_input)
        .count();

    // The verifier rebuilds the layout from `runtime_page_ranges`. Before the
    // duplicate-page fix that rebuild reproduced our injected layout exactly,
    // which is what made the attack work; now it REJECTS it. Assert that
    // directly — it is the fix firing at the layer it should — and keep going so
    // the test still exercises the full prove → verify path end to end.
    match Traces::page_configs_from_elf_and_runtime(
        &program,
        &runtime_page_ranges,
        num_private_input_pages,
        usize::MAX,
    ) {
        Ok(rebuilt) => {
            let ours: Vec<u64> = traces.page_configs.iter().map(|c| c.page_base).collect();
            let theirs: Vec<u64> = rebuilt.iter().map(|c| c.page_base).collect();
            assert_eq!(ours, theirs, "prover/verifier page layouts must agree");
        }
        Err(crate::Error::MalformedPageLayout(msg)) => {
            assert!(
                msg.contains("exactly"),
                "the rebuild must fail on duplicate coverage specifically: {msg}"
            );
        }
        Err(e) => panic!("unexpected page-layout error: {e}"),
    }

    let airs = VmAirs::new(
        &program,
        &options,
        false,
        &traces.page_configs,
        &table_counts,
        None,
        true,
        None,
        None,
        None,
    );

    let mut transcript = DefaultTranscript::<E>::new(&[]);
    absorb_statement(
        &mut transcript,
        StatementKind::Monolithic,
        honest_elf,
        &traces.public_output_bytes,
        &table_counts,
        num_private_input_pages,
        &runtime_page_ranges,
        options.fri_final_poly_log_degree,
    );

    let proof = Prover::multi_prove(
        airs.air_trace_pairs(&mut traces),
        &mut transcript,
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .expect("multi_prove");

    VmProof {
        proof,
        runtime_page_ranges,
        table_counts,
        public_output: traces.public_output_bytes.clone(),
        num_private_input_pages,
    }
}

/// STRUCTURAL REGRESSION: one address range covered by TWO PAGE tables must be
/// refused, even when the execution is honest and every injected row
/// self-cancels.
///
/// This is the invariant, isolated from any forgery: "one page per address". It
/// passed on the pre-fix branch — the layout was simply unvalidated — and is the
/// test that flips to a failure if the duplicate-base check is ever removed. The
/// forgery test below needs a compensating row and so could in principle be
/// blocked by something else; this one cannot.
#[test]
fn dup_structural_duplicate_page_coverage_is_rejected() {
    let honest = asm_elf_bytes("poc_rodata_commit");
    let (_, addr) = locate_secret(&honest);
    let base = crate::tables::page::page_base_for_address(addr);

    let proof = craft_proof_with_duplicate_page(&honest, &honest, base, None);
    // The execution itself is honest, so the output is the real one; only the
    // page layout is malformed.
    assert_eq!(proof.public_output, SECRET.to_vec());
    assert!(
        !verifier_accepts(&proof, &honest),
        "SOUNDNESS REGRESSION: the verifier accepted a layout with two PAGE tables \
         over one address range. Each address must have exactly one genesis token, \
         or two rows can swap which token each consumes."
    );
}

/// NEGATIVE CONTROL for the second route: forged run (target byte reads 0),
/// duplicate page present but every row self-cancelling, so the forged genesis
/// token has no provider. Must be rejected.
#[test]
fn dup_negative_control_without_compensating_row_fails() {
    let honest = asm_elf_bytes("poc_rodata_commit");
    let (file_off, addr) = locate_secret(&honest);
    let base = crate::tables::page::page_base_for_address(addr);
    let mut patched = honest.clone();
    patched[file_off] = 0x00;

    let proof = craft_proof_with_duplicate_page(&honest, &patched, base, None);
    assert_eq!(proof.public_output[0], 0x00);
    assert!(
        !verifier_accepts(&proof, &honest),
        "without the compensating row this must be rejected"
    );
}

/// REGRESSION (route 2 — duplicate page): the end-to-end forgery must not verify.
///
/// Forged run plus the duplicate page's row for the target consuming the ELF
/// page's genesis token. On the pre-fix branch — including after the OFFSET fix —
/// ELF `.data` byte `0x11` was made to read as `0x00` and the proof was ACCEPTED
/// against the UNMODIFIED ELF.
///
/// Strictly weaker than the OFFSET break: the injected value is always 0, because
/// a zero-init page is the only kind a prover can conjure at a chosen base. But it
/// needs no private input and no free OFFSET, which is why the OFFSET fix alone
/// did not stop it.
#[test]
fn dup_duplicate_page_forgery_is_rejected() {
    let honest = asm_elf_bytes("poc_rodata_commit");
    let (file_off, addr) = locate_secret(&honest);
    let base = crate::tables::page::page_base_for_address(addr);
    let offset = crate::tables::page::offset_in_page(addr);
    let mut patched = honest.clone();
    patched[file_off] = 0x00;

    let proof = craft_proof_with_duplicate_page(&honest, &patched, base, Some((offset, SECRET[0])));

    // Non-vacuity: the proof really does report the zeroed byte.
    assert_eq!(proof.public_output[0], 0x00, "forged output");
    assert_ne!(proof.public_output, SECRET.to_vec());

    assert!(
        !verifier_accepts(&proof, &honest),
        "SOUNDNESS REGRESSION: an ELF .data byte was made to read as 0 and the proof \
         verified against the unmodified ELF, via a duplicate zero-init page over an \
         address the ELF already covers."
    );
}
