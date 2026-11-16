//! The word memory format's proof field on blocks (`BlockWhirProof::
//! word_memory`): a split and a fused proof are the bytes before the field
//! then its seven zero counts; any nonzero count on a split or fused proof is
//! refused (and verifies with the refusal taken out, so the rule is what
//! refuses it); a split or fused proof does not verify under the word
//! format. The proving tests run where a block can be proved (the box's lib
//! run and CI). The word format's statement checks on hand-built statements
//! are `tables::word_memory_tests`'.

use crate::block_whir::{
    BlockFormat, BlockWhirProof, Deviations, WORD_COUNTS_UNCHECKED, prove_block_whir_with,
    verify_block_whir,
};
use crate::tables::MaxRowsConfig;
use crate::tables::fused_memory::MemoryFormat;
use crate::test_utils::asm_elf_bytes;
use stark::proof::options::ProofOptions;

fn format(memory: MemoryFormat) -> BlockFormat {
    BlockFormat::production().with_memory(memory)
}

/// `elf` proved as a streamed block under `format`.
fn prove(elf: &[u8], format: &BlockFormat) -> BlockWhirProof {
    let options = super::block_whir_tests::streamed(MaxRowsConfig::small(), 3, 9);
    prove_block_whir_with(
        elf,
        &[],
        &ProofOptions::default_test_options(),
        format,
        &options,
        &Deviations::default(),
    )
    .expect("prove")
    .0
}

fn verifies(proof: &BlockWhirProof, elf: &[u8], format: &BlockFormat) -> bool {
    verify_block_whir(proof, elf, &ProofOptions::default_test_options(), format).unwrap_or(false)
}

/// [`BlockWhirProof`] as it was before the word memory format: every field
/// but the last, in its order.
#[derive(rkyv::Archive, rkyv::Serialize)]
struct BeforeWords {
    proof: stark::multilinear_table::MultiProof<crate::test_utils::F, crate::test_utils::E>,
    table_num_vars: Vec<u8>,
    runtime_page_ranges: Vec<crate::RuntimePageRange>,
    table_counts: crate::TableCounts,
    public_output: Vec<u8>,
    num_private_input_pages: usize,
    groups: Vec<Vec<u32>>,
    prepared:
        Vec<multilinear::stacked_eval::StackedProof<crate::test_utils::F, crate::test_utils::E>>,
    argues: Vec<stark::multilinear_table::BatchedArgue<crate::test_utils::E>>,
    commit_ood: Vec<math::field::element::FieldElement<crate::test_utils::E>>,
    fused_memory: [u64; 2],
}

impl BeforeWords {
    fn of(proof: &BlockWhirProof) -> Self {
        let p = proof.clone();
        Self {
            proof: p.proof,
            table_num_vars: p.table_num_vars,
            runtime_page_ranges: p.runtime_page_ranges,
            table_counts: p.table_counts,
            public_output: p.public_output,
            num_private_input_pages: p.num_private_input_pages,
            groups: p.groups,
            prepared: p.prepared,
            argues: p.argues,
            commit_ood: p.commit_ood,
            fused_memory: p.fused_memory,
        }
    }
}

/// ★ A split and a fused proof's bytes are the format's before the word
/// counts, then the counts: the struct before them archives to exactly the
/// proof's bytes but its last 56, which are zero. So the bytes a binary
/// before the word format proves are this prefix, and a byte moved in any
/// earlier field moves the prefix (the check's mutant: one flipped byte of a
/// group index).
#[test]
fn a_split_and_a_fused_proof_are_the_bytes_before_the_word_counts_then_zeros() {
    let elf = asm_elf_bytes("fused_mem_cases");
    for memory in [MemoryFormat::Split, MemoryFormat::Fused] {
        let proof = prove(&elf, &format(memory));
        assert_eq!(proof.word_memory, [0; 7], "{memory:?}");
        let now = rkyv::to_bytes::<rkyv::rancor::Error>(&proof).expect("serialize");
        let before =
            rkyv::to_bytes::<rkyv::rancor::Error>(&BeforeWords::of(&proof)).expect("serialize");
        assert_eq!(now.len(), before.len() + 56, "{memory:?}: 56 bytes after");
        assert_eq!(
            &now[..before.len()],
            &before[..],
            "{memory:?}: the bytes before"
        );
        assert_eq!(
            &now[before.len()..],
            &[0u8; 56],
            "{memory:?}: the counts, zero"
        );
        let mut moved = proof.clone();
        moved.groups[0][0] ^= 1;
        let moved_before =
            rkyv::to_bytes::<rkyv::rancor::Error>(&BeforeWords::of(&moved)).expect("serialize");
        assert_ne!(
            &moved_before[..],
            &before[..],
            "a moved byte parts the prefix"
        );
    }
}

/// ★ A split or fused proof stating word tables is refused: the counts are
/// absorbed nowhere under those formats, so no value may ride in a proof. One
/// family at 1, and one at `2^32 − 1`, on an honest proof are each refused;
/// with the refusal taken out (a control mutant) each verifies, the counts
/// unbound.
#[test]
fn a_split_or_fused_proof_stating_word_tables_is_refused() {
    let elf = asm_elf_bytes("fused_mem_cases");
    for memory in [MemoryFormat::Split, MemoryFormat::Fused] {
        let f = format(memory);
        let honest = prove(&elf, &f);
        assert!(verifies(&honest, &elf, &f), "{memory:?}");
        for stated in [
            [1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, u64::from(u32::MAX)],
        ] {
            let mut forged = honest.clone();
            forged.word_memory = stated;
            assert!(
                !verifies(&forged, &elf, &f),
                "{memory:?} {stated:?} refused"
            );
            WORD_COUNTS_UNCHECKED.with(|c| c.set(true));
            let control = verifies(&forged, &elf, &f);
            WORD_COUNTS_UNCHECKED.with(|c| c.set(false));
            assert!(
                control,
                "{memory:?} {stated:?}: without the refusal it verifies"
            );
        }
    }
}

/// ★ A split or a fused proof does not verify under the word format: the
/// format is the verifier's constant, and the word format's statement (its
/// form word and counts, PAGE_W's height) is not theirs.
#[test]
fn a_split_or_fused_proof_is_refused_under_the_word_format() {
    let elf = asm_elf_bytes("fused_mem_cases");
    for memory in [MemoryFormat::Split, MemoryFormat::Fused] {
        let proof = prove(&elf, &format(memory));
        assert!(verifies(&proof, &elf, &format(memory)));
        assert!(
            !verifies(&proof, &elf, &format(MemoryFormat::Words)),
            "{memory:?}"
        );
    }
}
