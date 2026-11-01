//! The GKR slot packing on the block's base (`GkrPack::Chunks`): an opt-in
//! format value, `Off` the
//! production constant.
//!
//! Pinned here: its knob reads `off` or a price and nothing else; a packed
//! block proves and verifies on the host; today's verifier refuses it and the
//! packed verifier refuses today's proof; a chunk's output tampered is
//! refused; and the public block verifier's presets take the packing as a
//! named constant, `Off` unless the caller names another.

use crate::block_whir::{
    BlockFormat, BlockWhirProof, Deviations, prove_block_whir_with, verify_block_whir,
};
use crate::lfm::whir_block_tree::pack_setting;
use crate::test_utils::asm_elf_bytes;
use crate::tests::block_whir_p1_tests::{many_groups, options};
use math::field::element::FieldElement;
use multilinear::whir_chain::{ArgueFormat, GkrPack};
use stark::proof::options::{BaseFormat, ProofOptions};

type E = crate::tables::types::GoldilocksExtension;

const PACKED: ArgueFormat = ArgueFormat::Batched {
    bin_log_cells: 26,
    pack: GkrPack::Chunks { tree_log_cells: 8 },
};

fn format(argue: ArgueFormat) -> BlockFormat {
    BlockFormat {
        argue,
        ..many_groups(BaseFormat::P1_WHIR)
    }
}

fn prove(elf: &[u8], format: &BlockFormat) -> BlockWhirProof {
    prove_block_whir_with(
        elf,
        &[],
        &ProofOptions::default_test_options(),
        format,
        &options(0),
        &Deviations::default(),
    )
    .expect("prove")
    .0
}

fn verify(proof: &BlockWhirProof, elf: &[u8], format: &BlockFormat) -> bool {
    verify_block_whir(proof, elf, &ProofOptions::default_test_options(), format).unwrap_or(false)
}

/// `BLOCK_WHIR_ARGUE_PACK`: unset or `off` is today's, a count to 40 a price,
/// anything else refused.
#[test]
fn the_pack_knob_reads_off_or_a_price() {
    assert_eq!(pack_setting(None), Ok(GkrPack::Off));
    assert_eq!(pack_setting(Some("off")), Ok(GkrPack::Off));
    assert_eq!(
        pack_setting(Some(" 22 ")),
        Ok(GkrPack::Chunks { tree_log_cells: 22 })
    );
    assert_eq!(
        pack_setting(Some("0")),
        Ok(GkrPack::Chunks { tree_log_cells: 0 })
    );
    for bad in ["41", "-1", "on", "", "2^22"] {
        assert!(pack_setting(Some(bad)).is_err(), "{bad:?}");
    }
    assert_eq!(ArgueFormat::BATCHED, ArgueFormat::batched(26));
    assert!(matches!(
        ArgueFormat::BATCHED,
        ArgueFormat::Batched {
            pack: GkrPack::Off,
            ..
        }
    ));
}

/// ★ The production formats keep today's trees, and the block verifier's
/// presets take the packing only as a named constant: `Off` by default,
/// another only when the caller names it.
#[test]
fn the_verifiers_packing_is_a_named_constant() {
    assert!(matches!(
        BlockFormat::production().argue,
        ArgueFormat::Batched {
            pack: GkrPack::Off,
            ..
        }
    ));
    assert!(matches!(
        BlockFormat::pinned().argue,
        ArgueFormat::Batched {
            pack: GkrPack::Off,
            ..
        }
    ));
    let pack = GkrPack::Chunks { tree_log_cells: 22 };
    let named = BlockFormat::pinned().with_pack(pack);
    assert_eq!(
        named.argue,
        ArgueFormat::Batched {
            bin_log_cells: 26,
            pack
        }
    );
    assert_eq!(
        BlockFormat::pinned().with_pack(GkrPack::Off).argue,
        BlockFormat::pinned().argue
    );
    // A per-table format has no trees to pack.
    let per_table = BlockFormat {
        argue: ArgueFormat::PerTable,
        ..BlockFormat::pinned()
    };
    assert_eq!(per_table.with_pack(pack).argue, ArgueFormat::PerTable);
}

/// ★ A packed block proves and verifies on the host with more bus outputs
/// than tables (KECCAK_RND and CPU are cut); today's verifier refuses it, the
/// packed verifier refuses today's proof, and a group's first output tampered
/// is refused.
#[test]
fn a_packed_block_proves_and_verifies() {
    let elf = asm_elf_bytes("test_keccak");
    let packed = format(PACKED);
    let today = format(ArgueFormat::BATCHED);
    let proof = prove(&elf, &packed);
    let outputs: usize = proof.argues.iter().map(|a| a.bus_outputs.len()).sum();
    assert!(
        outputs > proof.table_num_vars.len(),
        "{outputs} bus outputs for {} tables: something is cut",
        proof.table_num_vars.len()
    );
    assert!(verify(&proof, &elf, &packed), "the packed verifier");
    assert!(!verify(&proof, &elf, &today), "today's verifier refuses it");
    let todays = prove(&elf, &today);
    assert!(verify(&todays, &elf, &today));
    assert!(
        !verify(&todays, &elf, &packed),
        "the packed verifier refuses today's"
    );
    for g in 0..proof.argues.len() {
        let mut tampered = proof.clone();
        tampered.argues[g].bus_outputs[0].0 += FieldElement::<E>::one();
        assert!(
            !verify(&tampered, &elf, &packed),
            "group {g}'s first output"
        );
    }
}
