//! What the block verifier holds as its own constants, from the prover crate
//! (whose tests CI runs): a WHIR chain's domain against the configured rate,
//! the query count every accepted format and statement derives, and BITWISE's
//! fixed height, which holds that count at the production one.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField as F;
use multilinear::mle::Mle;
use multilinear::whir::Domain;
use multilinear::whir_chain::{
    ArgueFormat, ChainConfig, ChainFormat, GrindBits, GroupFold, StackVars, commit, prove, verify,
};
use multilinear::whir_hash::KeccakWhir;
use stark::proof::options::ProofOptions;

use crate::block_whir::{self, BlockFormat, BlockOptions, Deviations, prove_block_whir_with};
use crate::multilinear_prove::chain_config_under;
use crate::tables::MaxRowsConfig;
use crate::test_utils::asm_elf_bytes;
use crate::zf_format::ZfFormat;

type FE = FieldElement<F>;

/// ★ A WHIR chain verified against a domain off its configuration's rate is
/// refused by name (`Error::DomainRateMismatch`), one size smaller or one
/// larger; against its own domain it verifies.
#[test]
fn a_chain_on_a_domain_off_its_rate_is_refused() {
    let cfg = ChainConfig {
        log_blowup: 1,
        log_folding: 2,
        num_queries: 3,
        grind: GrindBits::default(),
        format: ChainFormat::DEFAULT,
    };
    let num_vars = 6;
    let f = Mle::new(
        (0..1u64 << num_vars)
            .map(|i| FE::from(i.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 13))
            .collect(),
    )
    .expect("a cube");
    let z: Vec<FE> = (0..num_vars).map(|i| FE::from(101 + i as u64)).collect();
    let y = f.evaluate(&z).expect("a point of the cube");
    let transcript = || DefaultTranscript::<F>::new(b"a chain's domain");
    let (commitment, domain) = commit::<F, KeccakWhir>(&f, &cfg, true).expect("commits");
    let proof = prove::<F, F, _, KeccakWhir>(&f, &z, &commitment, &domain, &cfg, &mut transcript())
        .expect("proves");
    let rate = num_vars + cfg.log_blowup;
    let check = |domain: &Domain<F>| {
        verify::<F, F, _, KeccakWhir>(
            &proof,
            &commitment.root(),
            &z,
            y,
            domain,
            &cfg,
            &mut transcript(),
        )
    };
    for log_size in [rate - 1, rate + 1] {
        assert_eq!(
            check(&Domain::<F>::new(log_size).expect("a domain")),
            Err(multilinear::Error::DomainRateMismatch {
                expected: rate,
                got: log_size
            }),
            "a domain of 2^{log_size} points"
        );
    }
    assert_eq!(check(&domain), Ok(()), "its own domain");
}

/// ★ The query count from the code's own rule, at the default format, by
/// tallest stack: 229 from 23 variables, 228 from 15 to 22, fewer below. A
/// one-column table of `n` variables has a stack of `n`, so `[(1, n)]` makes
/// the tallest stack exactly `n`; BITWISE (21 columns of 2^20 rows) stacks to
/// 25.
#[test]
fn the_query_count_by_tallest_stack_at_the_default() {
    let q = |shapes: &[(usize, usize)]| chain_config_under(&ZfFormat::DEFAULT, shapes).num_queries;
    assert_eq!(q(&[(21, 20)]), 229);
    assert_eq!(q(&[(1, 23)]), 229);
    assert_eq!(q(&[(1, 22)]), 228);
    assert_eq!(q(&[(1, 15)]), 228);
    assert_eq!(q(&[(1, 14)]), 227);
    assert_eq!(q(&[(1, 7)]), 226);
    assert_eq!(q(&[(1, 6)]), 224);
    // BITWISE at 2^17: a stack of 22, the first height at 228.
    assert_eq!(q(&[(21, 18)]), 229);
    assert_eq!(q(&[(21, 17)]), 228);
}

/// ★ Every WHIR format `ZfFormat::from_lookup` accepts that moves the query
/// count — blowup, grind bits, grind placement, fold schedule and stack — at
/// BITWISE's stack (25) and at 27: the count is the (blowup, grind bits) pair's
/// alone, 229 / 225 / 114 / 112.
#[test]
fn every_accepted_format_runs_its_query_count() {
    for blowup in ["2", "4"] {
        for bits in ["18", "20"] {
            for grind in ["query", "all"] {
                for folds in ["uniform4", "first5", "first6"] {
                    for stack in ["25", "26", "27"] {
                        let pairs = [
                            (crate::zf_format::ENV_WHIR_BLOWUP, blowup),
                            (crate::zf_format::ENV_WHIR_GRIND_BITS, bits),
                            (crate::zf_format::ENV_WHIR_GRIND, grind),
                            (crate::zf_format::ENV_WHIR_FOLDS, folds),
                            (crate::zf_format::ENV_WHIR_STACK, stack),
                        ];
                        let f = ZfFormat::from_lookup(|k| {
                            pairs
                                .iter()
                                .find(|(n, _)| *n == k)
                                .map(|(_, v)| v.to_string())
                        })
                        .expect("an accepted format");
                        let at25 = chain_config_under(&f, &[(21, 20)]).num_queries;
                        let at27 = chain_config_under(&f, &[(21, 20), (64, 21)]).num_queries;
                        let want = match (blowup, bits) {
                            ("2", "18") => 229,
                            ("2", "20") => 225,
                            ("4", "18") => 114,
                            _ => 112,
                        };
                        assert_eq!(
                            (at25, at27),
                            (want, want),
                            "blowup {blowup} bits {bits} grind {grind} folds {folds} stack {stack}"
                        );
                    }
                }
            }
        }
    }
}

fn small_format() -> BlockFormat {
    let mut zf = ZfFormat::DEFAULT;
    zf.whir_stack = StackVars::new(10).expect("a valid stack");
    BlockFormat {
        zf,
        group_polys: block_whir::BLOCK_GROUP_POLYS,
        max_groups: block_whir::BLOCK_MAX_GROUPS,
        prepared: true,
        argue: ArgueFormat::PerTable,
        group: GroupFold::Indexed,
        leaf_scale: crate::lfm::whir_block::LeafScale::Fixed,
        leaf_query: crate::lfm::whir_chain::LeafQuery::V1,
    }
}

fn small_options() -> BlockOptions {
    BlockOptions {
        max_rows: MaxRowsConfig::default(),
        keccak_rnd_rows_log2: 16,
        ecdas_rows_log2: block_whir::BLOCK_ECDAS_ROWS_LOG2,
        keccak_rows_log2: block_whir::BLOCK_KECCAK_ROWS_LOG2,
        ecsm_rows_log2: block_whir::BLOCK_ECSM_ROWS_LOG2,
        copy_rows_log2: block_whir::BLOCK_COPY_ROWS_LOG2,
        drop_levels: multilinear::whir_commit::TreeDrop::uniform(3),
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
        rest_build_bytes: None,
        pack_finished: true,
        gpack: true,
        spill: block_whir::BlockSpillPolicy::Off,
        regen: None,
        phase_b_overlap: stark::multilinear_block::PhaseBOverlap::Off,
    }
}

/// ★ The frame holds BITWISE to its fixed 2^20 rows, so the query count stays
/// 229 by a rule of its own: the honest statement frames at 229; one with every
/// table at four rows, BITWISE included, or with BITWISE alone at 2^17 (which
/// would derive 226 and 228) is refused by name; and one with every other
/// table small but BITWISE at its height frames, still at 229.
#[test]
fn the_frame_refuses_a_bitwise_table_off_its_fixed_height() {
    let elf = asm_elf_bytes("sub");
    let format = small_format();
    let opts = ProofOptions::default_test_options();
    let proof = prove_block_whir_with(
        &elf,
        &[],
        &opts,
        &format,
        &small_options(),
        &Deviations::default(),
    )
    .expect("the block proves")
    .0;
    let honest = block_whir::block_frame(proof.statement(), &elf, &opts, &format).expect("frame");
    assert_eq!(
        honest.airs.air_refs()[block_whir::BITWISE_TABLE].name(),
        "BITWISE",
        "BITWISE is the first table"
    );
    assert_eq!(honest.shapes[block_whir::BITWISE_TABLE], (21, 20));
    assert_eq!(honest.config.num_queries, 229);

    let frame_at = |heights: &[u8]| {
        block_whir::block_frame(
            block_whir::BlockStatement {
                table_num_vars: heights,
                ..proof.statement()
            },
            &elf,
            &opts,
            &format,
        )
    };
    let refused_by_name = |heights: &[u8]| match frame_at(heights) {
        Err(crate::Error::InvalidTableCounts(why)) => why.starts_with("BITWISE states"),
        _ => false,
    };
    assert!(
        refused_by_name(&vec![2u8; proof.table_num_vars.len()]),
        "every table at four rows"
    );
    let small: Vec<u8> = proof.table_num_vars.iter().map(|&h| h.min(12)).collect();
    let mut bitwise_17 = small.clone();
    bitwise_17[block_whir::BITWISE_TABLE] = 17;
    assert!(refused_by_name(&bitwise_17), "BITWISE alone at 2^17");
    let mut bitwise_20 = small;
    bitwise_20[block_whir::BITWISE_TABLE] = 20;
    let frame = frame_at(&bitwise_20).expect("BITWISE at its height frames");
    assert_eq!(frame.config.num_queries, 229, "BITWISE holds the count");
}

/// Format knobs a verifier's process could carry, each off the default: the
/// WHIR blowup, grind bits and folds, the leaves' query check and scale, and
/// the leaves' share form.
const OFF_DEFAULT: [(&str, &str); 6] = [
    (crate::zf_format::ENV_WHIR_BLOWUP, "4"),
    (crate::zf_format::ENV_WHIR_GRIND_BITS, "20"),
    (crate::zf_format::ENV_WHIR_FOLDS, "uniform4"),
    (crate::lfm::whir_block::LEAF_QUERY_ENV, "v1"),
    (crate::lfm::whir_block::LEAF_DOUBLE_FROM_ENV, "off"),
    ("LFM_WHIR_SHARE_INVERSE", "0"),
];

/// ★ The block verifier reads no format from its environment. In a process of
/// its own whose environment sets every knob in [`OFF_DEFAULT`]
/// ([`the_pinned_presets_hold_under_an_environment_off_the_default`]), the
/// prover's format follows the environment while the verifier's presets stay
/// the pinned constants. So a verifier left with `LAMBDA_VM_ZF_WHIR_BLOWUP=4`
/// still checks a proof at the production format: it verifies a production
/// proof and refuses one made at blowup 4, whose statement binds the other
/// rate.
#[test]
fn the_block_verifier_reads_no_format_from_its_environment() {
    let name =
        "tests::verifier_pin_tests::the_pinned_presets_hold_under_an_environment_off_the_default";
    let out = std::process::Command::new(std::env::current_exe().expect("the test binary"))
        .args([
            "--exact",
            name,
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .envs(OFF_DEFAULT)
        .output()
        .expect("the test binary runs");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success() && stdout.contains("1 passed"),
        "the child under the knobs:\n{stdout}\n{stderr}"
    );
}

/// Run by [`the_block_verifier_reads_no_format_from_its_environment`] in a
/// process whose environment is [`OFF_DEFAULT`]; refuses to run anywhere else.
#[test]
#[ignore = "run in a process of its own by the_block_verifier_reads_no_format_from_its_environment"]
fn the_pinned_presets_hold_under_an_environment_off_the_default() {
    use crate::lfm::whir_block::{
        LeafScale, pinned_verifier_presets, share_inverse, verifier_presets_at,
    };
    use crate::lfm::whir_chain::LeafQuery;
    use crate::zf_format::WhirBlowup;
    use multilinear::whir_chain::LeafOrder;
    assert!(
        OFF_DEFAULT
            .iter()
            .all(|(k, v)| std::env::var(k).as_deref() == Ok(*v)),
        "run by its parent, under the knobs"
    );
    // The prover's format is the environment's.
    let production = BlockFormat::production();
    assert_eq!(production.zf.whir_blowup, 4);
    assert_eq!(production.zf.whir_grind_bits, 20);
    assert_ne!(production.zf.whir_folds, ZfFormat::DEFAULT.whir_folds);
    assert_eq!(production.leaf_query, LeafQuery::V1);
    assert_eq!(production.leaf_scale, LeafScale::Fixed);
    assert!(!share_inverse());

    // The verifier's is not.
    let debug = |o: &ProofOptions| format!("{o:?}");
    let pinned = pinned_verifier_presets(GroupFold::Indexed);
    assert_eq!(pinned.format.zf, ZfFormat::DEFAULT);
    assert_eq!(pinned.format.leaf_query, LeafQuery::Lean);
    assert_eq!(pinned.format.leaf_scale, LeafScale::DEFAULT);
    assert_eq!(pinned.format.group, GroupFold::Indexed);
    assert!(pinned.share_inverse);
    assert_eq!(
        debug(&pinned.base_options),
        debug(&crate::lfm::proof::block_base_options_under(
            &ZfFormat::DEFAULT
        ))
    );
    assert_eq!(
        debug(&pinned.wrap_options),
        debug(&crate::lfm::proof::aggregation_wrap_options_under(
            &ZfFormat::DEFAULT
        ))
    );

    // The named-choice form sets its four and pins the rest, folds and grind
    // included.
    let at = verifier_presets_at(
        GroupFold::Indexed,
        WhirBlowup::Four,
        LeafOrder::Natural,
        LeafScale::Fixed,
        LeafQuery::V1,
    )
    .expect("named choices");
    assert_eq!(
        at.format.zf,
        ZfFormat {
            whir_blowup: 4,
            whir_leaves: LeafOrder::Natural,
            ..ZfFormat::DEFAULT
        }
    );
    assert_eq!(
        (at.format.leaf_scale, at.format.leaf_query),
        (LeafScale::Fixed, LeafQuery::V1)
    );
    assert!(at.share_inverse);
}
