//! `verify_with_options(.., Some(decode), Some(pages))` does NOT bind the
//! supplied roots to `inner_elf`: a custom prover can absorb `elf_digest(X)`
//! into the Fiat-Shamir statement while the constrained instructions and every
//! preprocessed root are those of a DIFFERENT program Y, and verification with
//! `inner_elf = X` and Y's roots still returns `Ok(true)` (the "critical
//! soundness check" in `crypto/stark/src/verifier.rs` compares two
//! prover-controlled values here, so it is vacuous against a custom prover).
//!
//! The recursion guest does not rely on verify for that binding: it commits
//! `program_id(inner_elf, decode, pages)`, which folds the supplied roots into
//! the identity — so the same substitution yields an id that differs from the
//! honest `program_id(X)`, detectable by whoever recomputes it natively and
//! compares. These tests pin both facts: verify accepts, and the fold catches.

use std::collections::HashSet;
use std::path::PathBuf;

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use stark::prover::{IsStarkProver, Prover};

use crate::statement::{StatementKind, absorb_statement, elf_digest};
use crate::tables::trace_builder::Traces;
use crate::test_utils::E;
use crate::{Commitment, MaxRowsConfig, VmAirs, VmProof};

use executor::elf::Elf;
use executor::vm::execution::Executor;

/// Smallest inner proof (blowup=2, 1 query) — for speed; soundness of the
/// *scheme* is what's under test, not the FRI security level.
const MIN_PROOF_OPTIONS: stark::proof::options::ProofOptions =
    stark::proof::options::ProofOptions {
        blowup_factor: 2,
        fri_number_of_queries: 1,
        coset_offset: 3,
        grinding_factor: 1,
        fri_final_poly_log_degree: 7,
    };

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .to_path_buf()
}

fn read_guest_elf(name: &str) -> Vec<u8> {
    let path = workspace_root().join(format!("executor/program_artifacts/recursion/{name}.elf"));
    std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "failed to read {} — run `make compile-recursion-elfs`: {e}",
            path.display()
        )
    })
}

/// Precomputed DECODE + ELF-data-page commitments for `elf_bytes` under `opts`
/// — exactly what the recursion guest receives as private input.
fn precomputed_commitments(
    elf_bytes: &[u8],
    opts: &stark::proof::options::ProofOptions,
) -> (Commitment, Vec<(u64, Commitment)>) {
    let elf = Elf::load(elf_bytes).expect("ELF load failed");
    let decode_commitment =
        crate::tables::decode::commitment_from_elf(&elf, opts).expect("decode commitment failed");
    let page_commitments: Vec<(u64, Commitment)> = Traces::page_configs_from_elf(&elf)
        .iter()
        .filter(|c| c.init_values.is_some())
        .map(|c| {
            (
                c.page_base,
                crate::tables::page::compute_precomputed_commitment(c, opts),
            )
        })
        .collect();
    (decode_commitment, page_commitments)
}

/// The set of program-counter values fetched during a run of `elf_bytes`.
fn executed_pcs(elf_bytes: &[u8]) -> HashSet<u64> {
    let elf = Elf::load(elf_bytes).expect("ELF load failed");
    let executor = Executor::new(&elf, vec![]).expect("executor new");
    let result = executor.run().expect("run failed");
    result.logs.iter().map(|l| l.current_pc).collect()
}

/// Read a 4-byte word (LE) from an executable ELF segment at virtual address
/// `vaddr`, returning its raw-file byte offset and current value. Parses the
/// program headers directly so we can patch the raw bytes (and thus the
/// `elf_digest`) at exactly the right place.
fn exec_words(elf_bytes: &[u8]) -> Vec<(usize, u64, u32)> {
    let rd_u16 = |o: usize| u16::from_le_bytes(elf_bytes[o..o + 2].try_into().unwrap());
    let rd_u32 = |o: usize| u32::from_le_bytes(elf_bytes[o..o + 4].try_into().unwrap());
    let rd_u64 = |o: usize| u64::from_le_bytes(elf_bytes[o..o + 8].try_into().unwrap());

    let e_phoff = rd_u64(32) as usize;
    let e_phentsize = rd_u16(54) as usize;
    let e_phnum = rd_u16(56) as usize;

    const PT_LOAD: u32 = 1;
    const PF_X: u32 = 1;

    let mut out = Vec::new();
    for i in 0..e_phnum {
        let ph = e_phoff + i * e_phentsize;
        let p_type = rd_u32(ph);
        let p_flags = rd_u32(ph + 4);
        if p_type != PT_LOAD || (p_flags & PF_X) == 0 {
            continue;
        }
        let p_offset = rd_u64(ph + 8) as usize;
        let p_vaddr = rd_u64(ph + 16);
        let p_filesz = rd_u64(ph + 32) as usize;
        let mut off = 0usize;
        while off + 4 <= p_filesz {
            let file_off = p_offset + off;
            let vaddr = p_vaddr + off as u64;
            out.push((file_off, vaddr, rd_u32(file_off)));
            off += 4;
        }
    }
    out
}

/// Build program Y from program X (`= empty.elf`) by patching a single
/// executable-segment word at a PC that X never fetches, to a *different*
/// still-parseable instruction. Because the word is never fetched, Y halts
/// byte-identically to X, so their AIR structure (entry, segments, pages,
/// table counts, public output, runtime pages) is identical — they differ
/// ONLY in one instruction's bytes, hence different DECODE root, different
/// code-page root, and different `elf_digest`.
fn make_variant_program(x_bytes: &[u8]) -> Vec<u8> {
    let executed = executed_pcs(x_bytes);
    let words = exec_words(x_bytes);

    // A never-fetched slot we can rewrite to a valid, distinct instruction.
    // Candidates are canonical nops (`addi x0,x0,K`), which always parse.
    const NOP_0: u32 = 0x0000_0013; // addi x0, x0, 0
    const NOP_1: u32 = 0x0010_0013; // addi x0, x0, 1

    let (file_off, _vaddr, cur) = words
        .iter()
        .find(|(_, vaddr, _)| !executed.contains(vaddr))
        .copied()
        .expect("no never-executed executable word found to patch");

    let new_word = if cur == NOP_1 { NOP_0 } else { NOP_1 };

    let mut y = x_bytes.to_vec();
    y[file_off..file_off + 4].copy_from_slice(&new_word.to_le_bytes());
    assert_ne!(y, x_bytes.to_vec(), "variant must differ from base");
    y
}

/// Custom prover: prove `prove_elf`'s execution honestly, but absorb
/// `statement_elf`'s identity into the Fiat-Shamir transcript instead of
/// `prove_elf`'s. Every preprocessed root in the resulting proof is computed
/// from `prove_elf`. Mirrors `prove_with_options_and_inputs`, swapping only
/// the `elf_bytes` passed to `absorb_statement`.
fn custom_prove_with_statement_elf(
    prove_elf: &[u8],
    statement_elf: &[u8],
    opts: &stark::proof::options::ProofOptions,
) -> VmProof {
    let program = Elf::load(prove_elf).expect("prove ELF load failed");
    let executor = Executor::new(&program, vec![]).expect("executor new");
    let result = executor.run().expect("run failed");

    let max_rows = MaxRowsConfig::default();
    let mut traces = Traces::from_elf_and_logs(
        &program,
        &result.logs,
        &max_rows,
        &[],
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .expect("trace build failed");

    let table_counts = traces.table_counts();
    let airs = VmAirs::new(
        &program,
        opts,
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
        statement_elf, // <-- the substitution: X's identity, Y's everything else
        &traces.public_output_bytes,
        &table_counts,
        num_private_input_pages,
        &runtime_page_ranges,
        opts.fri_final_poly_log_degree,
    );

    let proof = Prover::multi_prove(
        airs.air_trace_pairs(&mut traces),
        &mut transcript,
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .expect("multi_prove failed");

    VmProof {
        proof,
        runtime_page_ranges,
        table_counts,
        public_output: traces.public_output_bytes.clone(),
        num_private_input_pages,
    }
}

/// Sanity: the custom prover, used honestly (statement == proven program),
/// produces genuinely valid proofs. Guards against a vacuous PoC.
#[test]
fn test_custom_prover_is_not_vacuous() {
    let x = read_guest_elf("empty");
    let proof = custom_prove_with_statement_elf(&x, &x, &MIN_PROOF_OPTIONS);
    let ok = crate::verify_with_options(&proof, &x, &MIN_PROOF_OPTIONS, None, None)
        .expect("verify errored");
    assert!(ok, "custom prover must produce valid proofs when honest");
}

/// `verify` accepts a proof whose Fiat-Shamir statement is X's but whose
/// constrained instructions and supplied roots are Y's — so verify is not the
/// binding. The `program_id` fold is: it commits an id that differs from the
/// honest id of X, making the substitution detectable downstream.
#[test]
fn test_supplied_decode_root_not_bound_to_inner_elf() {
    let x = read_guest_elf("empty");
    let y = make_variant_program(&x);

    // X and Y differ, and specifically in their preprocessed DECODE roots.
    assert_ne!(elf_digest(&x), elf_digest(&y), "elf_digest must differ");
    let (decode_x, pages_x) = precomputed_commitments(&x, &MIN_PROOF_OPTIONS);
    let (decode_y, pages_y) = precomputed_commitments(&y, &MIN_PROOF_OPTIONS);
    assert_ne!(decode_x, decode_y, "DECODE roots must differ (X vs Y)");

    // Craft a proof: constrain Y, but absorb X's identity into the statement.
    let proof = custom_prove_with_statement_elf(&y, &x, &MIN_PROOF_OPTIONS);

    // Negative control: the honest recompute path (None, None) rebuilds X's
    // roots and rejects — the proof is NOT coincidentally valid for X.
    let honest = crate::verify_with_options(&proof, &x, &MIN_PROOF_OPTIONS, None, None)
        .expect("verify errored");
    assert!(
        !honest,
        "honest recompute (None, None) must reject: proof carries Y's roots, X recompute differs"
    );

    // verify is NOT the binding: with Y's roots supplied (the guest's
    // private-input path), verification accepts for inner_elf = X.
    let accepted = crate::verify_with_options(
        &proof,
        &x,
        &MIN_PROOF_OPTIONS,
        Some(decode_y),
        Some(&pages_y),
    )
    .expect("verify errored");
    assert!(
        accepted,
        "verify unexpectedly rejected the mismatched-root proof"
    );

    // The fold IS the binding: folding Y's supplied roots into X's identity
    // yields an id that differs from the honest id of X.
    let forged_id = crate::statement::program_id_from_elf(&x, &decode_y, &pages_y).unwrap();
    let honest_id = crate::statement::program_id_from_elf(&x, &decode_x, &pages_x).unwrap();
    assert_ne!(
        forged_id, honest_id,
        "program_id fold must make the root substitution detectable"
    );
}
