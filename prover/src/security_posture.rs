//! The security posture this build proves and verifies under: the production
//! one everywhere, a low-security TEST one in the prover crate's own unit-test
//! binary built with the `test-security` feature, and nowhere else.
//!
//! # What the test posture changes
//!
//! Only parameters whose job is soundness and whose price is time:
//!
//! - every WHIR chain (the base proof's and the W-LFM levels'): its query
//!   count is sized for [`TEST_CHAIN_SECURITY_BITS`] instead of
//!   [`crate::multilinear_prove::CHAIN_SECURITY_BITS`], and it grinds at most
//!   [`TEST_GRIND_BITS`] bits instead of the format's 18 (or 20). The W-LFM
//!   blowup-4 count is then derived by the same rule rather than pinned at
//!   [`crate::lfm::wlfm::WLFM_B4_QUERIES`].
//! - every STARK proof whose options come from
//!   [`crate::lfm::proof::aggregation_wrap_options_under`] (the LFM proofs) or
//!   [`crate::lfm::proof::block_base_options_under`] (the STARK block's base
//!   epochs): its query count is re-derived for [`TEST_CHAIN_SECURITY_BITS`]
//!   at its own blowup, and it grinds at most [`TEST_GRIND_BITS`] bits.
//!
//! Nothing else moves: the format (caps, folds, stacks, blowups, leaf order,
//! trees, hashes), the AIRs and the statements are the production ones. So a
//! byte identity between two arms of one build is as meaningful under either
//! posture, and the prover and every verifier of one build (host, in-guest,
//! levels) agree, because each derives its parameters from the same function.
//!
//! # How it is selected, and why nothing else can select it
//!
//! [`TEST_POSTURE`] is `cfg!(all(test, feature = "test-security"))`: true only
//! in the prover crate's own lib-test binary built with
//! `--features test-security`. Every non-test build (the release prover, the
//! verifier binaries, the integration tests, the guests) compiles the lib
//! without `cfg(test)`, so it proves and verifies at production even with the
//! feature on: the `const` assertion below makes that a compile error to
//! break, and `tests/security_posture.rs` checks it from a build with the
//! feature on. There is no environment knob: nothing a process reads at run
//! time selects it.
//!
//! The box jobs build two lib-test binaries: the timing one without the
//! feature (production), the gates' and the identity step's with it. The
//! timing harness (`rta_a_real_block_through_its_levels`) is the same test in
//! both, so it prints the build's posture (`RTA POSTURE: …`) and refuses to
//! run when `RTA_POSTURE` names another.
//!
//! # Production is unchanged
//!
//! Outside that one binary every function here is the identity (or returns
//! today's constant), so every production proof keeps its bytes; the pins in
//! the default lib-test binary (e.g. the production chain programs'
//! digests in `lfm::whir_chain_tests`) say so.

use stark::proof::options::ProofOptions;

/// ★ Whether this build proves under the TEST posture: the prover crate's
/// unit-test binary built with `--features test-security`, and nothing else.
pub const TEST_POSTURE: bool = cfg!(all(test, feature = "test-security"));

// A build of the library that is not its own unit-test binary can never prove
// or verify under the test posture, whatever features it was built with.
#[cfg(not(test))]
const _: () = assert!(!TEST_POSTURE);

/// The security target the test posture sizes every query count for.
pub const TEST_CHAIN_SECURITY_BITS: u8 = 32;

/// The most proof-of-work bits any grind spends under the test posture.
pub const TEST_GRIND_BITS: u8 = 8;

/// The target a WHIR chain's query count (and the round form's later counts)
/// is sized for in this build: [`crate::multilinear_prove::CHAIN_SECURITY_BITS`],
/// or [`TEST_CHAIN_SECURITY_BITS`] under the test posture.
pub const fn chain_security_bits() -> u8 {
    if TEST_POSTURE {
        TEST_CHAIN_SECURITY_BITS
    } else {
        crate::multilinear_prove::CHAIN_SECURITY_BITS
    }
}

/// The bits a WHIR chain grinds in this build when its format says `bits`:
/// `bits`, or at most [`TEST_GRIND_BITS`] under the test posture.
pub const fn whir_grind_bits(bits: u8) -> u8 {
    if TEST_POSTURE && bits > TEST_GRIND_BITS {
        TEST_GRIND_BITS
    } else {
        bits
    }
}

/// The query count a W-LFM chain at blowup 4 runs in this build, `derived`
/// being the count the rule gives it: the pinned
/// [`crate::lfm::wlfm::WLFM_B4_QUERIES`], or `derived` under the test posture.
#[cfg(feature = "parallel")]
pub const fn wlfm_b4_queries(derived: usize) -> usize {
    if TEST_POSTURE {
        derived
    } else {
        crate::lfm::wlfm::WLFM_B4_QUERIES
    }
}

/// `options` as this build proves and verifies them: unchanged, or under the
/// test posture with at most [`TEST_GRIND_BITS`] grind bits and the query
/// count the STARK rule gives [`TEST_CHAIN_SECURITY_BITS`] at their blowup.
pub fn stark_options(options: ProofOptions) -> ProofOptions {
    if !TEST_POSTURE {
        return options;
    }
    let grinding_factor = options.grinding_factor.min(TEST_GRIND_BITS);
    let derived = stark::proof::options::GoldilocksCubicProofOptions::with_params(
        options.blowup_factor,
        TEST_CHAIN_SECURITY_BITS,
        grinding_factor,
    )
    .expect("the test posture's target exceeds its grind at every blowup");
    ProofOptions {
        fri_number_of_queries: derived.fri_number_of_queries,
        grinding_factor,
        ..options
    }
}

/// The line a process proving under the test posture prints under the
/// format banner; `None` (nothing printed) at production.
pub fn banner() -> Option<String> {
    TEST_POSTURE.then(|| {
        format!(
            "SECURITY POSTURE: TEST (test-security build) · chains {TEST_CHAIN_SECURITY_BITS} bits · \
             grind <= {TEST_GRIND_BITS} bits · not a production proof"
        )
    })
}

/// The posture's name, as the timing harness prints it and `RTA_POSTURE`
/// spells it.
pub const fn name() -> &'static str {
    if TEST_POSTURE { "test" } else { "production" }
}

/// The box gate runs the same fixtures in both unit-test binaries. The first
/// writes the verified TEST proof, and the second reads it into aligned wire
/// storage for the production verifier.
#[cfg(test)]
pub(crate) fn exchange_test_proof(file: &str, proof: &[u8], check_foreign: impl FnOnce(&[u8])) {
    let dir = std::env::var_os("LAMBDA_VM_TEST_POSTURE_PROOFS")
        .expect("run both postures with make test-cross-posture");
    let path = std::path::PathBuf::from(dir).join(file);
    if TEST_POSTURE {
        std::fs::write(path, proof).expect("write the verified TEST proof");
    } else {
        let bytes = std::fs::read(path).expect("read the verified TEST proof");
        let mut aligned = rkyv::util::AlignedVec::<16>::with_capacity(bytes.len());
        aligned.extend_from_slice(&bytes);
        check_foreign(&aligned);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zf_format::ZfFormat;

    /// `ProofOptions` has no `PartialEq`: compare every field through `Debug`.
    fn debug(o: &ProofOptions) -> String {
        format!("{o:?}")
    }

    /// The production chain at the block's tallest stack: Q 229 at 18
    /// query-grind bits (the default format), whatever this binary's posture.
    fn production_q_at_25() -> (usize, u8) {
        let c = multilinear::whir_chain::ChainConfig::with_security_folds(
            ZfFormat::DEFAULT.whir_log_blowup(),
            crate::zf_format::PRODUCTION_WHIR_LOG_FOLDING,
            ZfFormat::DEFAULT.whir_folds,
            25,
            crate::multilinear_prove::CHAIN_SECURITY_BITS,
            ZfFormat::DEFAULT.whir_grind_bits(),
        );
        (c.num_queries, c.grind.query)
    }

    /// ★ Without the feature this binary is production: every function above
    /// is the identity or today's constant, and the production sites build
    /// today's parameters (Q 229 and 18 bits on the chains, Q 112 and 20 bits
    /// on the LFM proofs, Q 110 and 20 bits on the STARK base).
    #[cfg(not(feature = "test-security"))]
    #[test]
    fn without_the_feature_the_posture_is_production() {
        // `black_box`: the constant IS what is checked (which value this build compiled).
        assert!(!std::hint::black_box(TEST_POSTURE));
        assert_eq!(name(), "production");
        assert_eq!(banner(), None);
        assert_eq!(chain_security_bits(), 128);
        assert_eq!(whir_grind_bits(18), 18);
        assert_eq!(whir_grind_bits(20), 20);
        #[cfg(feature = "parallel")]
        assert_eq!(wlfm_b4_queries(7), 141);
        let preset = crate::recursion::Preset::Blowup4.options();
        assert_eq!(debug(&stark_options(preset.clone())), debug(&preset));

        assert_eq!(production_q_at_25(), (229, 18));
        let chain = crate::multilinear_prove::chain_config_under(&ZfFormat::DEFAULT, &[(1, 25)]);
        assert_eq!((chain.num_queries, chain.grind.query), (229, 18));
        let wrap = crate::lfm::proof::aggregation_wrap_options_under(&ZfFormat::DEFAULT);
        assert_eq!(
            (wrap.fri_number_of_queries, wrap.grinding_factor),
            (112, 20)
        );
        let base = crate::lfm::proof::block_base_options_under(&ZfFormat::DEFAULT);
        assert_eq!(
            (base.fri_number_of_queries, base.grinding_factor),
            (110, 20)
        );
    }

    /// ★ With the feature this binary is the test posture, and the production
    /// sites build its parameters: Q 54 and 8 bits on the chains at 25
    /// variables (against 229 and 18), Q 25 and 8 bits on the LFM proofs and
    /// the STARK base (against 112 / 110 and 20); every other option is the
    /// production one.
    #[cfg(feature = "test-security")]
    #[test]
    fn with_the_feature_the_posture_is_test() {
        assert!(std::hint::black_box(TEST_POSTURE));
        assert_eq!(name(), "test");
        assert!(banner().is_some_and(|b| b.starts_with("SECURITY POSTURE: TEST ")));
        assert_eq!(chain_security_bits(), TEST_CHAIN_SECURITY_BITS);
        assert_eq!(whir_grind_bits(18), 8);
        assert_eq!(whir_grind_bits(5), 5);
        #[cfg(feature = "parallel")]
        assert_eq!(wlfm_b4_queries(7), 7);

        assert_eq!(production_q_at_25(), (229, 18));
        let chain = crate::multilinear_prove::chain_config_under(&ZfFormat::DEFAULT, &[(1, 25)]);
        assert_eq!((chain.num_queries, chain.grind.query), (54, 8));
        assert_eq!(
            (chain.grind.folding, chain.grind.ood),
            (0, 0),
            "the query grind alone"
        );

        let production_wrap = {
            let mut o = stark::proof::options::GoldilocksCubicProofOptions::with_blowup(4)
                .expect("blowup 4");
            o.fri_number_of_queries = crate::lfm::proof::LFM_QUERIES;
            o.fri_final_poly_log_degree = 8;
            ZfFormat::DEFAULT.options(o)
        };
        let wrap = crate::lfm::proof::aggregation_wrap_options_under(&ZfFormat::DEFAULT);
        assert_eq!((wrap.fri_number_of_queries, wrap.grinding_factor), (25, 8));
        assert_eq!(
            debug(&ProofOptions {
                fri_number_of_queries: production_wrap.fri_number_of_queries,
                grinding_factor: production_wrap.grinding_factor,
                ..wrap.clone()
            }),
            debug(&production_wrap),
            "the wrap options move only in Q and the grind"
        );
        let base = crate::lfm::proof::block_base_options_under(&ZfFormat::DEFAULT);
        assert_eq!((base.fri_number_of_queries, base.grinding_factor), (25, 8));
        assert_eq!(base.blowup_factor, 4);
    }

    fn wrap_fixture() -> (
        crate::lfm::compiler::LfmProgram,
        Vec<Vec<crate::lfm::word::LfmWord>>,
    ) {
        use crate::lfm::builder::LfmBuilder;
        use crate::lfm::compiler::compile;
        use crate::lfm::word::base_word;
        use crate::tables::types::FE;

        let mut b = LfmBuilder::new();
        let arena = b.declare_arena(1);
        let h = b.hint_felt(arena, 0);
        let x = b.felt_const(FE::from(7u64));
        let s = b.add(x, h);
        let m = b.mul(s, x);
        b.public(m.as_cell());
        (compile(b.finish()), vec![vec![base_word(FE::from(3u64))]])
    }

    /// Both postures prove and verify a wrap, and refuse a tampered public word.
    #[test]
    #[cfg_attr(
        not(feature = "test-security"),
        ignore = "production posture smoke; box tier"
    )]
    fn an_lfm_proof_under_this_builds_wrap_options_verifies_and_refuses_a_tamper() {
        use crate::lfm::proof::{lfm_prove, verify_against_artifacts};
        use crate::lfm::registry::build_artifacts;
        use crate::lfm::word::base_word;
        use crate::tables::types::FE;

        let (program, arenas) = wrap_fixture();
        let opts = crate::lfm::proof::aggregation_wrap_options_under(&ZfFormat::DEFAULT);
        println!(
            "posture {}: LFM wrap Q {} grind {}",
            name(),
            opts.fri_number_of_queries,
            opts.grinding_factor
        );
        let artifacts = build_artifacts(&program, &opts);
        let proof = lfm_prove(&program, &artifacts, &arenas, &opts).expect("the program proves");
        assert!(
            verify_against_artifacts(&artifacts, &proof.proof, &proof.public_words, &opts),
            "the honest proof must verify, or the refusal below proves nothing"
        );
        assert_eq!(proof.public_words.len(), 1);
        assert_eq!(proof.public_words[0].1, base_word(FE::from(70u64)));
        let mut forged = proof.public_words.clone();
        forged[0].1 = base_word(FE::from(71u64));
        assert!(
            !verify_against_artifacts(&artifacts, &proof.proof, &forged, &opts),
            "a tampered public word must be refused"
        );
    }

    #[test]
    #[ignore = "requires both posture builds; run make test-cross-posture"]
    fn lfm_wrap_rejects_test_posture() {
        use crate::lfm::proof::{LfmStarkProof, lfm_prove, verify_against_artifacts};
        use crate::lfm::registry::build_artifacts;
        use crate::lfm::word::LfmWord;

        let (program, arenas) = wrap_fixture();
        let opts = crate::lfm::proof::aggregation_wrap_options_under(&ZfFormat::DEFAULT);
        let artifacts = build_artifacts(&program, &opts);
        let control = lfm_prove(&program, &artifacts, &arenas, &opts).expect("the wrap proves");
        let control = (control.proof, control.public_words);
        let verifies = |proof: &LfmStarkProof, public: &[(u32, LfmWord)]| {
            verify_against_artifacts(&artifacts, proof, public, &opts)
        };
        assert!(verifies(&control.0, &control.1));
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&control).expect("serialize the wrap");
        exchange_test_proof("wrap", &bytes, |foreign| {
            let (proof, public): (LfmStarkProof, Vec<(u32, LfmWord)>) =
                rkyv::from_bytes::<_, rkyv::rancor::Error>(foreign).expect("decode the TEST wrap");
            assert_eq!(public, control.1, "same public output");
            assert!(
                !verifies(&proof, &public),
                "production accepted the TEST wrap"
            );
        });
    }

    #[cfg(feature = "parallel")]
    fn prove_and_verify_shipped_block() -> (Vec<u8>, rkyv::util::AlignedVec) {
        use crate::block_whir::BlockFormat;
        use crate::lfm::compress::DEFAULT_PLAN;
        use crate::lfm::whir_block::BLOCK_FAN_IN;
        use crate::lfm::whir_block_tree::{
            DEFAULT_SHIPPED, StderrSink, WhirTreeConfig, prove_whir_block_tree, read_shipped,
            verify_whir_block_shipped_as,
        };

        let elf = crate::test_utils::asm_elf_bytes("basic_program");
        let format = BlockFormat::pinned();
        let cfg = WhirTreeConfig {
            leaves: None,
            fan_in: BLOCK_FAN_IN,
            argue: format.argue,
            format,
            compress: DEFAULT_PLAN.rate,
            compress_levers: DEFAULT_PLAN.levers,
            compress_wire: DEFAULT_PLAN.wire,
            compress2: DEFAULT_PLAN.c2,
            ..WhirTreeConfig::from_env().expect("valid tree options")
        };
        let run = prove_whir_block_tree(&elf, &[], &cfg, &StderrSink).expect("the block proves");
        let (shipped, bytes) = run
            .compressed
            .expect("default compression ran")
            .shipped(run.statement)
            .expect("encode the shipped proof");
        assert_eq!(shipped, DEFAULT_SHIPPED);
        let control = read_shipped(&bytes, shipped).expect("the shipped proof decodes");
        verify_whir_block_shipped_as(&elf, &control, DEFAULT_SHIPPED)
            .expect("the shipped proof verifies in its own posture");
        (elf, bytes)
    }

    #[cfg(all(feature = "parallel", feature = "test-security"))]
    #[test]
    fn the_shipped_compressed_block_verifies_at_test_security() {
        prove_and_verify_shipped_block();
    }

    /// The ignored cross-posture test generates these shipped bytes. Keep the
    /// ELF from that run alongside them because the proof binds its contents.
    #[cfg(all(feature = "parallel", feature = "test-security"))]
    #[test]
    fn the_shipped_test_posture_fixture_verifies() {
        use crate::lfm::whir_block_tree::{
            DEFAULT_SHIPPED, read_shipped, verify_whir_block_shipped_as,
        };

        let elf = include_bytes!("lfm/testdata/basic-program.elf");
        let bytes = include_bytes!("lfm/testdata/basic-program-test.wproof");
        let mut aligned = rkyv::util::AlignedVec::<16>::with_capacity(bytes.len());
        aligned.extend_from_slice(bytes);
        let proof = read_shipped(&aligned, DEFAULT_SHIPPED).expect("decode the shipped TEST proof");
        verify_whir_block_shipped_as(elf, &proof, DEFAULT_SHIPPED)
            .expect("verify the shipped TEST proof");
    }

    #[cfg(feature = "parallel")]
    #[test]
    #[ignore = "proves a shipped block in both builds; run make test-cross-posture"]
    fn compressed_block_rejects_test_posture() {
        use crate::lfm::whir_block_tree::{
            DEFAULT_SHIPPED, read_shipped, verify_whir_block_shipped_as,
        };

        let (elf, bytes) = prove_and_verify_shipped_block();
        let control = read_shipped(&bytes, DEFAULT_SHIPPED).expect("decode the control block");
        exchange_test_proof("compressed-block", &bytes, |foreign| {
            let proof = read_shipped(foreign, DEFAULT_SHIPPED).expect("decode the TEST block");
            let (test, production) = (proof.statement(), control.statement());
            assert_eq!(test.public_output, production.public_output);
            assert_eq!(
                crate::statement::table_count_values(test.table_counts),
                crate::statement::table_count_values(production.table_counts)
            );
            assert_eq!(test.table_num_vars, production.table_num_vars);
            let ranges = |s: crate::block_whir::BlockStatement<'_>| {
                s.runtime_page_ranges
                    .iter()
                    .map(|r| (r.base, r.count))
                    .collect::<Vec<_>>()
            };
            assert_eq!(ranges(test), ranges(production));
            assert_eq!(
                test.num_private_input_pages,
                production.num_private_input_pages
            );
            assert_eq!(test.groups, production.groups);
            assert_eq!(test.fused_memory, production.fused_memory);
            assert_eq!(test.word_memory, production.word_memory);
            assert!(
                verify_whir_block_shipped_as(&elf, &proof, DEFAULT_SHIPPED).is_err(),
                "production accepted the TEST compressed block"
            );
        });
    }

    /// A STARK option set that grinds less than the test posture's cap keeps
    /// its grind under it (and so its nonce layout); one at zero stays at zero.
    #[test]
    fn the_posture_never_raises_a_grind() {
        for g in [0u8, 1, 8, 20] {
            let o = ProofOptions {
                grinding_factor: g,
                ..crate::recursion::Preset::Blowup4.options()
            };
            let got = stark_options(o).grinding_factor;
            assert!(got <= g, "grind {g} became {got}");
            assert_eq!(
                got,
                if TEST_POSTURE {
                    g.min(TEST_GRIND_BITS)
                } else {
                    g
                }
            );
            assert!(whir_grind_bits(g) <= g);
        }
    }
}
