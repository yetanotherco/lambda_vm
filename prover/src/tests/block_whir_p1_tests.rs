//! D-WHIR-P1 S3, D-P3B P3: the block's base is its proof family's
//! ([`crate::hash_pin::BlockFamily`]): [`BaseFormat::P1_WHIR`], ZisK's
//! Poseidon1 (4-ary trees capped at height 2, its field sponge and W8 grind).
//! The format names it, the family pins it: any other base — RPX's included —
//! is refused; a P1 proof verifies under no other base; and the statement's tag
//! names the base's geometry.

use crate::block_whir::{
    self, BlockFormat, BlockOptions, BlockWhirProof, Deviations, MAX_P1_CAP_HEIGHT, P1_TAG_OMITTED,
    block_statement_tag, checked_base, prove_block_whir_with, verify_block_whir,
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

/// The tag names the base's effective geometry under the family's hash: no
/// base, RPX's included, gets the untagged V1 tag.
#[test]
fn the_statement_tag_names_the_bases_geometry() {
    let p1 = |cap| BaseFormat {
        hash: CommitmentHash::Poseidon1,
        arity4_cap: cap,
    };
    let tag = |cap| String::from_utf8(block_statement_tag(&p1(cap))).expect("ascii");
    let v1 = String::from_utf8(MULTILINEAR_BLOCK_TAG.to_vec()).expect("ascii");
    assert_ne!(block_statement_tag(&BaseFormat::RPX), MULTILINEAR_BLOCK_TAG);
    assert_eq!(tag(CapPolicy::Fixed(2)), format!("{v1}/P1W16/C2"));
    assert_eq!(
        block_statement_tag(&BaseFormat::P1_WHIR),
        tag(CapPolicy::Fixed(2)).as_bytes()
    );
    assert_eq!(tag(CapPolicy::Off), format!("{v1}/P1W16/C0"));
    assert_eq!(
        tag(CapPolicy::Fixed(0)),
        tag(CapPolicy::Off),
        "both uncapped"
    );
    assert_eq!(tag(CapPolicy::Auto), format!("{v1}/P1W16/Cauto"));
    assert_ne!(tag(CapPolicy::Fixed(1)), tag(CapPolicy::Fixed(2)));
}

/// The base check: the block proof family's base and nothing else — not RPX,
/// not Poseidon1 at another cap, not another hash.
#[test]
fn checked_base_refuses_what_no_block_proves() {
    let family = crate::hash_pin::BLOCK_BASE;
    assert_eq!(family, BaseFormat::P1_WHIR);
    assert!(checked_base(&family).is_ok());
    assert!(
        checked_base(&BaseFormat::RPX).is_err(),
        "RPX left the block path"
    );
    for cap in [
        CapPolicy::Off,
        CapPolicy::Auto,
        CapPolicy::Fixed(1),
        CapPolicy::Fixed(MAX_P1_CAP_HEIGHT as u8),
        CapPolicy::Fixed(MAX_P1_CAP_HEIGHT as u8 + 1),
    ] {
        let base = BaseFormat {
            arity4_cap: cap,
            ..BaseFormat::P1_WHIR
        };
        assert!(checked_base(&base).is_err(), "{cap}");
    }
    for hash in [
        CommitmentHash::Keccak256,
        CommitmentHash::Blake3,
        CommitmentHash::Rpo256,
        CommitmentHash::Rpx256,
        CommitmentHash::Poseidon,
    ] {
        let base = BaseFormat {
            hash,
            ..BaseFormat::P1_WHIR
        };
        assert!(checked_base(&base).is_err(), "{hash:?}");
    }
}

/// ★ A block proves and verifies under the family's base; it verifies under no
/// other (RPX's, or Poseidon1 at another cap), and no other base proves: the
/// format names the base, the family decides it. Groups retire and revive
/// between the phases (drop 3).
#[test]
fn a_block_proves_under_its_family_base_and_no_other_verifies_it() {
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
    assert!(
        prove_block_whir_with(
            &elf,
            &[],
            &ProofOptions::default_test_options(),
            &rpx,
            &options(3),
            &Deviations::default(),
        )
        .is_err(),
        "an RPX base is refused, not proved"
    );
    // The cap is the verifier's constant too.
    let other_cap = many_groups(BaseFormat {
        arity4_cap: CapPolicy::Fixed(1),
        ..BaseFormat::P1_WHIR
    });
    assert!(!verify(&p1_proof, &elf, &other_cap), "C2 read as C1");
}

/// ★ The tag is in the statement: a P1 proof is refused by a verifier that
/// absorbs the untagged `…_V1` tag in its place, and a proof made and verified
/// with that tag in both places is accepted — the tag is the only thing the
/// mutation moves.
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
    assert!(refused, "a P1 proof verified under the untagged tag");
    assert!(both_untagged, "the mutation is the tag alone");
    assert!(
        !verify(&untagged, &elf, &p1),
        "an untagged proof under the tag"
    );
}

/// ★ The statement the Poseidon1 leaf absorbs call by call
/// (`block_statement_calls`) is `absorb_block`'s own: the calls concatenate to
/// the statement's one run (`block_statement_bytes`, the block id's preimage),
/// and `P1Transcript` fed them lands where `absorb_block` leaves it.
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

/// The WHIR configuration `base` dispatches to, by its type's name.
fn dispatched(base: BaseFormat) -> Result<&'static str, crate::Error> {
    Ok(crate::with_block_hash!(base, |H| {
        std::any::type_name::<H>()
    }))
}

/// ★ The dispatch itself, whatever the process's knob says: the family's base
/// binds the family's WHIR configuration (`P1Whir`), and every other base —
/// RPX's, keccak's — is refused, not bound to a configuration of its own.
#[test]
fn the_base_alone_binds_the_block_hash() {
    println!(
        "the process's WHIR hash knob: {:?}",
        std::env::var(crate::whir_hash_knob::ENV)
    );
    assert_eq!(
        dispatched(BaseFormat::P1_WHIR).expect("Poseidon1 dispatches"),
        std::any::type_name::<multilinear::whir_hash::P1Whir>()
    );
    assert!(
        dispatched(BaseFormat::RPX).is_err(),
        "an RPX base is refused"
    );
    let keccak = BaseFormat {
        hash: CommitmentHash::Keccak256,
        ..BaseFormat::RPX
    };
    assert!(dispatched(keccak).is_err(), "a keccak base is refused");
}

/// ★ No environment read decides the base: the block's prove, verify and
/// plan sites dispatch through `with_block_hash!` on the format alone, whose
/// body — the base check, then the family's configuration — reads no
/// environment and never consults the WHIR hash knob, and the plan reads no
/// knob either (D-WHIR-P1 S7: `LAMBDA_VM_WHIR_HASH` left the block path).
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
    let arms = &arms[..arms.find("\n}\n").expect("its end")];
    assert!(
        arms.contains("checked_base(&$base)?")
            && arms.contains("type $h = $crate::hash_pin::BlockWhirHash;"),
        "the macro checks the base and binds the family's configuration"
    );
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

/// ★ The block path names one proof family (D-P3B §2.1, review item 1): no
/// LFM prove, verify, harvest, shape or artifact site of the W3 tree is at the
/// legacy pin. The plan, the tree driver, the node and the block's dispatch
/// name neither the legacy family nor the pin's configuration, transcript,
/// prover, verifier or socket, nor the legacy-family entry points (which are
/// `LegacyFamily`'s instances); they reach the LFM through `BlockFamily`.
#[test]
fn no_block_path_site_names_the_legacy_pin() {
    let files = [
        ("block_whir.rs", include_str!("../block_whir.rs")),
        ("lfm/whir_block.rs", include_str!("../lfm/whir_block.rs")),
        (
            "lfm/whir_block_tree.rs",
            include_str!("../lfm/whir_block_tree.rs"),
        ),
        ("lfm/block_node.rs", include_str!("../lfm/block_node.rs")),
    ];
    let legacy = [
        "LegacyFamily",
        "BlockStarkHash",
        "BlockTranscript",
        "block_transcript(",
        "BlockProver",
        "BlockVerifier",
        "BLOCK_HASHER",
        "BLOCK_COMMITMENT_HASH",
        "WrapHash::production()",
        "verify_against_artifacts(",
        "harvest_child(",
        "DerivedChild::from_artifacts(",
        "build_artifacts_counted(",
        ".prove(&",
        "lfm_prove(",
    ];
    for (name, source) in files {
        // Only the code: the block path's own tests may name what they refuse.
        let code = source
            .split("#[cfg(test)]\nmod tests")
            .next()
            .unwrap_or(source);
        for word in legacy {
            assert!(!code.contains(word), "{name} names {word}");
        }
    }
    let plan = include_str!("../lfm/whir_block.rs");
    for word in [
        "build_artifacts_counted_under::<BlockStark>",
        "from_artifacts_under::<crate::hash_pin::BlockFamily>",
        "verify_against_artifacts_under::<crate::hash_pin::BlockFamily>",
        "node_program_under::<crate::hash_pin::BlockFamily>",
    ] {
        assert!(plan.contains(word), "the plan reaches the LFM by {word}");
    }
    let tree = include_str!("../lfm/whir_block_tree.rs");
    assert_eq!(
        tree.matches("prove_under::<BlockFamily>").count(),
        2,
        "leaves and nodes"
    );
    assert_eq!(
        tree.matches("harvest_child_under::<BlockFamily>").count(),
        2
    );
}
