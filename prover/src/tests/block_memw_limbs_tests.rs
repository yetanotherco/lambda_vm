//! MEMW_A's timestamp order by the step's limbs
//! ([`MemwAlignedOrder::Limbs`], the block's production form) on proved
//! blocks: small programs prove and verify under each form, each form's
//! verifier refuses the other's proof, and the limbs' block asks LT for fewer
//! comparisons. Box tier: each test proves.

use crate::block_whir::{
    BlockFormat, BlockWhirProof, Deviations, prove_block_whir_with, verify_block_whir,
};
use crate::tables::memw_aligned::MemwAlignedOrder;
use crate::test_utils::asm_elf_bytes;
use crate::tests::block_whir_p1_tests::{many_groups, options};
use multilinear::whir_chain::ArgueFormat;
use stark::proof::options::{BaseFormat, ProofOptions};

fn format(memw_aligned: MemwAlignedOrder) -> BlockFormat {
    BlockFormat {
        argue: ArgueFormat::BATCHED,
        memw_aligned,
        ..many_groups(BaseFormat::P1_WHIR)
    }
}

/// `elf` proved under `format` on the windowed build (the limbs' form needs
/// it), in windows of 8 cycles.
fn prove(elf: &[u8], format: &BlockFormat) -> BlockWhirProof {
    let mut o = options(0);
    o.window_log2 = Some(3);
    prove_block_whir_with(
        elf,
        &[],
        &ProofOptions::default_test_options(),
        format,
        &o,
        &Deviations::default(),
    )
    .expect("prove")
    .0
}

fn verify(proof: &BlockWhirProof, elf: &[u8], format: &BlockFormat) -> bool {
    verify_block_whir(proof, elf, &ProofOptions::default_test_options(), format).unwrap_or(false)
}

/// The rows LT's tables hold, padded: what the block's statement states for
/// the tables its frame names `LT[…]`.
fn lt_rows(proof: &BlockWhirProof, elf: &[u8], format: &BlockFormat) -> u64 {
    let frame = crate::block_whir::block_frame(
        proof.statement(),
        elf,
        &ProofOptions::default_test_options(),
        format,
    )
    .expect("the statement frames");
    frame
        .airs
        .air_refs()
        .iter()
        .zip(&proof.table_num_vars)
        .filter(|(air, _)| air.name().starts_with("LT["))
        .map(|(_, &n)| 1u64 << n)
        .sum()
}

/// ★ A block in the limbs' form proves and verifies (its BITWISE counts the
/// limbs, LT holds no MEMW_A comparison, and HALT's finalizations, steps up
/// to 2^32 − 2, fit the limbs, or its buses would not balance); LT's verifier
/// refuses it, and the limbs' verifier refuses the LT form's proof. Programs
/// with aligned memory accesses at every width, KECCAK's and many
/// instructions.
#[test]
#[ignore = "proves small blocks; box tier"]
fn each_memw_a_form_proves_and_refuses_the_others_block() {
    let (lt, limbs) = (
        format(MemwAlignedOrder::Lt),
        format(MemwAlignedOrder::Limbs),
    );
    for name in ["all_loadstore_32", "test_keccak", "all_instructions_64"] {
        let elf = asm_elf_bytes(name);
        let by_limbs = prove(&elf, &limbs);
        assert!(
            verify(&by_limbs, &elf, &limbs),
            "{name}: the limbs' verifier"
        );
        assert!(
            !verify(&by_limbs, &elf, &lt),
            "{name}: LT's verifier refuses it"
        );
        let by_lt = prove(&elf, &lt);
        assert!(
            verify(&by_lt, &elf, &lt),
            "{name}: LT's verifier, LT's proof"
        );
        assert!(
            !verify(&by_lt, &elf, &limbs),
            "{name}: the limbs' verifier refuses LT's"
        );
        assert_eq!(
            by_limbs.table_counts.memw_aligned, by_lt.table_counts.memw_aligned,
            "{name}: the same MEMW_A tables"
        );
        let rows = (lt_rows(&by_lt, &elf, &lt), lt_rows(&by_limbs, &elf, &limbs));
        assert!(rows.1 <= rows.0, "{name}: LT's rows do not grow: {rows:?}");
        eprintln!(
            "MEMW_A LIMBS BLOCK {name}: {} tables, {} groups · MEMW_A tables {} · LT tables {} → {}, \
             rows {} → {} · verified under its form, refused under the other, both ways",
            by_limbs.table_num_vars.len(),
            by_limbs.groups.len(),
            by_limbs.table_counts.memw_aligned,
            by_lt.table_counts.lt,
            by_limbs.table_counts.lt,
            rows.0,
            rows.1,
        );
    }
}
