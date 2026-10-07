//! D-WHIR-P1 S3: the block's base hash is a format field. A block format whose
//! base is [`BaseFormat::P1_WHIR`] proves and verifies under ZisK's Poseidon1
//! (4-ary trees capped at height 2, its field sponge and W8 grind), selected by
//! the format alone; RPX stays today's, byte for byte; the two bases do not
//! cross-verify; and the statement's tag names the base's geometry.

use crate::block_whir::{
    self, BlockFormat, BlockHash, BlockOptions, BlockWhirProof, Deviations, MAX_P1_CAP_HEIGHT,
    P1_TAG_OMITTED, block_statement_tag, checked_base, prove_block_whir_with, verify_block_whir,
};
use crate::statement::MULTILINEAR_BLOCK_TAG;
use crate::tables::MaxRowsConfig;
use crate::test_utils::asm_elf_bytes;
use crate::zf_format::ZfFormat;
use multilinear::whir_chain::{ArgueFormat, StackVars};
use stark::config::CommitmentHash;
use stark::proof::options::{BaseFormat, CapPolicy, ProofOptions};

/// A stack of 2^10 and two polynomials a group, so a small program spans
/// several groups, under `base`.
fn many_groups(base: BaseFormat) -> BlockFormat {
    let mut zf = ZfFormat::DEFAULT.with_base(base);
    zf.whir_stack = StackVars::new(10).expect("a valid stack");
    BlockFormat {
        zf,
        group_polys: 2,
        max_groups: block_whir::BLOCK_MAX_GROUPS,
        prepared: true,
        argue: ArgueFormat::PerTable,
    }
}

/// `block_whir_tests`' small-program options, `drop_levels` dropped from each
/// retired tree (binary levels; half as many 4-ary ones under P1).
fn options(drop_levels: usize) -> BlockOptions {
    BlockOptions {
        max_rows: MaxRowsConfig::small(),
        keccak_rnd_rows_log2: 16,
        ecdas_rows_log2: block_whir::BLOCK_ECDAS_ROWS_LOG2,
        keccak_rows_log2: block_whir::BLOCK_KECCAK_ROWS_LOG2,
        ecsm_rows_log2: block_whir::BLOCK_ECSM_ROWS_LOG2,
        drop_levels: multilinear::whir_commit::TreeDrop::uniform(drop_levels),
        window_log2: None,
        stream_keccak_rnd: false,
        stream_memw_lt: false,
        drop_streamed_ops: false,
        layout_workers: 0,
        layout_ahead: Some(2),
        pack_rest_as_laid_out: false,
        narrow: stark::multilinear_block::Narrowing::Card { min_cells: 0 },
        upload_ahead: true,
        memlog: false,
        finish_keccak_rnd_chunks: true,
        finish_cuts: true,
        finish_stream: true,
        rest_layout_bytes: Some(1 << 20),
        pack_finished: true,
        gpack: true,
        spill: block_whir::BlockSpillPolicy::Off,
        regen: None,
    }
}

fn prove(elf: &[u8], format: &BlockFormat, drop_levels: usize) -> BlockWhirProof {
    prove_block_whir_with(
        elf,
        &[],
        &ProofOptions::default_test_options(),
        format,
        &options(drop_levels),
        &Deviations::default(),
    )
    .expect("prove")
    .0
}

fn verify(proof: &BlockWhirProof, elf: &[u8], format: &BlockFormat) -> bool {
    verify_block_whir(proof, elf, &ProofOptions::default_test_options(), format).unwrap_or(false)
}

/// The tag names the base's effective geometry; RPX's is today's.
#[test]
fn the_statement_tag_names_the_bases_geometry() {
    assert_eq!(block_statement_tag(&BaseFormat::RPX), MULTILINEAR_BLOCK_TAG);
    let p1 = |cap| BaseFormat {
        hash: CommitmentHash::Poseidon1,
        arity4_cap: cap,
    };
    let tag = |cap| String::from_utf8(block_statement_tag(&p1(cap))).expect("ascii");
    let rpx = String::from_utf8(MULTILINEAR_BLOCK_TAG.to_vec()).expect("ascii");
    assert_eq!(tag(CapPolicy::Fixed(2)), format!("{rpx}/P1W16/C2"));
    assert_eq!(
        block_statement_tag(&BaseFormat::P1_WHIR),
        tag(CapPolicy::Fixed(2)).as_bytes()
    );
    assert_eq!(tag(CapPolicy::Off), format!("{rpx}/P1W16/C0"));
    assert_eq!(
        tag(CapPolicy::Fixed(0)),
        tag(CapPolicy::Off),
        "both uncapped"
    );
    assert_eq!(tag(CapPolicy::Auto), format!("{rpx}/P1W16/Cauto"));
    assert_ne!(tag(CapPolicy::Fixed(1)), tag(CapPolicy::Fixed(2)));
}

/// The base check: RPX and Poseidon1 at a cap its trees run, and nothing else.
#[test]
fn checked_base_refuses_what_no_block_proves() {
    assert_eq!(checked_base(&BaseFormat::RPX).ok(), Some(BlockHash::Rpx));
    assert_eq!(
        checked_base(&BaseFormat::P1_WHIR).ok(),
        Some(BlockHash::Poseidon1)
    );
    for cap in [
        CapPolicy::Off,
        CapPolicy::Auto,
        CapPolicy::Fixed(MAX_P1_CAP_HEIGHT as u8),
    ] {
        let base = BaseFormat {
            arity4_cap: cap,
            ..BaseFormat::P1_WHIR
        };
        assert_eq!(
            checked_base(&base).ok(),
            Some(BlockHash::Poseidon1),
            "{cap}"
        );
    }
    let tall = BaseFormat {
        arity4_cap: CapPolicy::Fixed(MAX_P1_CAP_HEIGHT as u8 + 1),
        ..BaseFormat::P1_WHIR
    };
    assert!(checked_base(&tall).is_err(), "a cap no tree runs at");
    for hash in [
        CommitmentHash::Keccak256,
        CommitmentHash::Blake3,
        CommitmentHash::Rpo256,
        CommitmentHash::Poseidon,
    ] {
        let base = BaseFormat {
            hash,
            arity4_cap: CapPolicy::Off,
        };
        assert!(checked_base(&base).is_err(), "{hash:?}");
    }
}

/// ★ One process proves and verifies both bases, each selected by the format
/// alone; neither verifies under the other's format; and the P1 proof is not
/// the RPX one. Groups retire and revive between the phases (drop 3).
#[test]
fn a_block_proves_under_both_bases_and_they_do_not_cross_verify() {
    let elf = asm_elf_bytes("all_instructions_64");
    let (rpx, p1) = (
        many_groups(BaseFormat::RPX),
        many_groups(BaseFormat::P1_WHIR),
    );
    let p1_proof = prove(&elf, &p1, 3);
    assert!(p1_proof.proof.columns.len() > 1, "several groups");
    assert!(
        verify(&p1_proof, &elf, &p1),
        "a P1 proof under the P1 format"
    );
    assert!(
        !verify(&p1_proof, &elf, &rpx),
        "a P1 proof under the RPX format"
    );
    let rpx_proof = prove(&elf, &rpx, 3);
    assert!(
        verify(&rpx_proof, &elf, &rpx),
        "an RPX proof under the RPX format"
    );
    assert!(
        !verify(&rpx_proof, &elf, &p1),
        "an RPX proof under the P1 format"
    );
    assert_ne!(p1_proof.proof.roots, rpx_proof.proof.roots);
    // The cap is the verifier's constant too.
    let other_cap = many_groups(BaseFormat {
        arity4_cap: CapPolicy::Fixed(1),
        ..BaseFormat::P1_WHIR
    });
    assert!(!verify(&p1_proof, &elf, &other_cap), "C2 read as C1");
}

/// ★ The tag is in the statement: a P1 proof is refused by a verifier that
/// absorbs RPX's tag in its place, and a proof made and verified with RPX's tag
/// in both places is accepted — the tag is the only thing the mutation moves.
#[test]
fn the_p1_statement_tag_is_bound() {
    let elf = asm_elf_bytes("sub");
    let p1 = many_groups(BaseFormat::P1_WHIR);
    let honest = prove(&elf, &p1, 0);
    assert!(verify(&honest, &elf, &p1));
    P1_TAG_OMITTED.with(|omitted| omitted.set(true));
    let refused = !verify(&honest, &elf, &p1);
    let untagged = prove(&elf, &p1, 0);
    let both_untagged = verify(&untagged, &elf, &p1);
    P1_TAG_OMITTED.with(|omitted| omitted.set(false));
    assert!(refused, "a P1 proof verified under RPX's tag");
    assert!(both_untagged, "the mutation is the tag alone");
    assert!(
        !verify(&untagged, &elf, &p1),
        "an untagged proof under the tag"
    );
}

/// ★ The statement the Poseidon1 leaf absorbs call by call
/// (`block_statement_calls`) is `absorb_block`'s own: the calls concatenate to
/// the RPX leaf's one run (`block_statement_bytes`), and `P1Transcript` fed
/// them lands where `absorb_block` leaves it.
#[test]
fn the_statement_calls_are_absorb_blocks_own() {
    use crypto::fiat_shamir::is_transcript::IsTranscript;
    use crypto::fiat_shamir::p1_transcript::P1Transcript;
    let elf = asm_elf_bytes("sub");
    let p1 = many_groups(BaseFormat::P1_WHIR);
    let proof = prove(&elf, &p1, 0);
    let opts = ProofOptions::default_test_options();
    let statement = proof.statement();
    let config = block_whir::block_frame(statement, &elf, &opts, &p1)
        .expect("the frame")
        .config;
    let tag = block_statement_tag(&p1.zf.base);
    let calls = block_whir::block_statement_calls(statement, &tag, &elf, &config)
        .expect("byte strings only");
    let bytes = block_whir::block_statement_bytes(
        statement,
        &tag,
        &crate::statement::elf_digest(&elf),
        &config,
    );
    assert!(calls.len() > 10, "{} calls", calls.len());
    assert_eq!(calls.concat(), bytes, "the calls are the run, cut");
    let mut direct = P1Transcript::new();
    block_whir::absorb_block(
        &mut direct,
        &tag,
        &elf,
        statement.public_output,
        statement.table_counts,
        statement.num_private_input_pages,
        statement.runtime_page_ranges,
        statement.table_num_vars,
        &config,
        statement.groups,
    );
    let mut replay = P1Transcript::new();
    for call in &calls {
        replay.append_bytes(call);
    }
    assert_eq!(replay.state(), direct.state(), "call for call");
    let mut merged = P1Transcript::new();
    merged.append_bytes(&bytes);
    assert_ne!(
        merged.state(),
        direct.state(),
        "one run is another stream under Poseidon1: the calls are load-bearing"
    );
}

/// ★ No environment read decides the base: the block's prove, verify and
/// plan sites dispatch through `with_block_hash!` on the format alone, whose
/// arms — RPX's and Poseidon1's — and base check read no environment and never
/// consult the WHIR hash knob, and the plan reads no knob either
/// (D-WHIR-P1 S7: `LAMBDA_VM_WHIR_HASH` left the block path).
#[test]
fn no_environment_read_decides_the_base() {
    let block = include_str!("../block_whir.rs");
    assert_eq!(
        block.matches("crate::with_whir_hash!").count(),
        0,
        "every block dispatch is on the format's base"
    );
    assert_eq!(
        block
            .matches("crate::with_block_hash!(format.zf.base")
            .count(),
        3
    );
    let check = &block[block.find("pub fn checked_base").expect("checked_base")..];
    let check = &check[..check.find("\n}\n").expect("its end")];
    let tag = &block[block.find("pub fn block_statement_tag").expect("the tag")..];
    let tag = &tag[..tag.find("\n}\n").expect("its end")];
    let knob = include_str!("../whir_hash_knob.rs");
    let arms = &knob[knob
        .find("macro_rules! with_block_hash")
        .expect("the macro")..];
    let arms = &arms[arms.find("match $crate::block_whir").expect("the match")..];
    let arms = &arms[..arms.find("\n}\n").expect("its end")];
    assert!(arms.contains("BlockHash::Rpx") && arms.contains("BlockHash::Poseidon1"));
    for (name, source) in [
        ("checked_base", check),
        ("the tag", tag),
        ("the macro's arms", arms),
    ] {
        for read in [
            "std::env",
            "env::var",
            "selected()",
            "whir_hash_knob",
            "with_whir_hash",
        ] {
            assert!(!source.contains(read), "{name} names {read}");
        }
    }
    // The plan reads other knobs (`LFM_WHIR_SHARE_INVERSE`), never the hash's.
    let plan = include_str!("../lfm/whir_block.rs");
    for read in ["whir_hash_knob", "with_whir_hash", "LAMBDA_VM_WHIR_HASH"] {
        assert!(!plan.contains(read), "the block plan names {read}");
    }
}
