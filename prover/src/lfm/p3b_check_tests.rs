//! The block pin's checks, each with a test that fails without it (the P3b
//! review's laptop tier: emission, execution or no work at all; nothing
//! proves).
//!
//! - The in-guest Poseidon1 grind accepts exactly the host's nonces: over
//!   exhaustive windows at small factors, at the boundary bit (a nonce valid
//!   at `g − 1` bits and not at `g`) at small factors and at the block's own
//!   factor, over seeds with lanes near `p` and nonces at and past `p`, and as
//!   the leg runs it (state, check, nonce absorbed, the host's state after). A
//!   check one bit weaker turns these red; `the_grinding_check_is_p1s` does not
//!   see it.
//! - The configuration check refuses every mismatch of options and roots, and
//!   names the hash, not the cap.
//! - The block verifier and the tree's base harvest refuse a legacy hash
//!   before any work.
//! - The block's LFM tag is rendered by its cap alone, and no LFM tree at that
//!   cap is uncapped (so none reaches the odd-top padding case).
//! - Two public lanes whose bytes alias under the transcript's big-endian felt
//!   rule absorb alike, while the `LfmPublic` closure tells them apart.

use crypto::fiat_shamir::is_transcript::IsTranscript;
use crypto::grinding::is_valid_nonce;
use math::field::traits::IsPrimeField;
use stark::config::CommitmentHash;
use stark::proof::options::{BaseFormat, CapPolicy};
use stark::proof::stark::MultiProof;

use crate::hash_pin::{BLOCK_LFM, BLOCK_WRAP, Block, BlockHash, Legacy};
use crate::tables::types::{FE, FEE, GoldilocksField};

use super::algebraic_commit::{commitment_to_digest, digest_to_commitment};
use super::builder::LfmBuilder;
use super::compiler::{LfmProgram, compile};
use super::edsl::WrapDigest;
use super::executor::execute_serial;
use super::hash::HasherKind;
use super::p1_commit::{P1GrindDigest, P1Transcript};
use super::proof::LfmProof;
use super::transcript_replay::TranscriptReplay;
use super::word::{LfmWord, base_word};

const P: u64 = 0xFFFF_FFFF_0000_0001;

/// The guest's grind check at `factor` over a hinted seed word and nonce felt.
fn grind_program(factor: u8) -> LfmProgram {
    let mut b = LfmBuilder::new().with_wrap_hash(BLOCK_WRAP);
    let a_seed = b.declare_arena(1);
    let a_nonce = b.declare_arena(1);
    let s = WrapDigest::from_cell(b.hint_word(a_seed, 0));
    let n = b.hint_felt(a_nonce, 0);
    super::epoch::emit_grinding_check(&mut b, s, n, factor);
    let program = compile(b.finish());
    super::validator::validate(&program).expect("admissible");
    program
}

fn guest_accepts(program: &LfmProgram, lanes: [FE; 4], nonce: u64) -> bool {
    execute_serial(
        program,
        &[vec![lanes], vec![base_word(FE::from(nonce))]],
        &HasherKind::Poseidon1W16,
    )
    .is_ok()
}

fn host_accepts(lanes: &[FE; 4], nonce: u64, factor: u8) -> bool {
    let seed = super::algebraic_commit::digest_to_commitment(lanes);
    crypto::grinding::is_valid_nonce::<P1GrindDigest>(&seed, nonce, factor)
}

/// ★ The guest's P1 grind accepts exactly the host's nonce set: every nonce in
/// a window at small factors, and a boundary probe at 9 bits and at the
/// block's 20: the nonces whose lane 0 (under the factor-`g` inner hash) has
/// exactly `g − 1` leading zero bits must be refused, the ones with `g`
/// accepted. Seeds include lanes at and near `p`.
#[test]
fn the_guest_p1_grind_accepts_exactly_the_hosts_nonces() {
    let seeds: [[FE; 4]; 3] = [
        [
            FE::from(11u64),
            FE::from(22u64),
            FE::from(33u64),
            FE::from(44u64),
        ],
        [
            FE::from(P - 1),
            FE::from(P - 2),
            FE::from(1u64 << 63),
            FE::from(0xFFFF_FFFFu64),
        ],
        [
            FE::from(0xFFFF_FFFF_0000_0000u64),
            FE::zero(),
            FE::from(P - 1),
            FE::from(0x1_0000_0000u64),
        ],
    ];
    let block = block_lfm_grind();
    for (k, lanes) in seeds.iter().enumerate() {
        // Exhaustive windows at small factors.
        for factor in [1u8, 2, 5, 8] {
            let program = grind_program(factor);
            for nonce in 0..64u64 {
                assert_eq!(
                    guest_accepts(&program, *lanes, nonce),
                    host_accepts(lanes, nonce, factor),
                    "factor {factor} nonce {nonce}: guest and host disagree"
                );
            }
        }
        // The boundary probe, at a mid factor for every seed and at the
        // block's own factor for the first two: the nonces whose lane 0 (under
        // the factor-`g` inner hash) has exactly `g − 1` leading zeros. The
        // host refuses them; a guest that checked `g − 1` bits would not.
        let probes: &[u8] = if k < 2 { &[9, block] } else { &[9] };
        for &factor in probes {
            let program = grind_program(factor);
            let seed = super::algebraic_commit::digest_to_commitment(lanes);
            let inner = crypto::grinding::inner_hash_felts::<P1GrindDigest>(&seed, factor);
            let lane0 = |nonce: u64| -> u64 {
                let mut s = [FE::zero(); 8];
                for (x, v) in s.iter_mut().zip(inner) {
                    *x = FE::from(v);
                }
                s[4] = FE::from(nonce);
                GoldilocksField::canonical(crypto::hash::poseidon1_w8::permute(s)[0].value())
            };
            let (bound, edge_top) = (1u64 << (64 - factor), 1u64 << (65 - factor));
            let (mut hits, mut edge) = (0, 0);
            let mut nonce = 0u64;
            while (hits < 2 || edge < 2) && nonce < 1 << 26 {
                let v = lane0(nonce);
                if v < edge_top {
                    let valid = v < bound;
                    assert_eq!(
                        host_accepts(lanes, nonce, factor),
                        valid,
                        "factor {factor} nonce {nonce}: lane 0 is the host's rule"
                    );
                    assert_eq!(
                        guest_accepts(&program, *lanes, nonce),
                        valid,
                        "factor {factor} nonce {nonce}: lane 0 {v:#x}, bound {bound:#x}"
                    );
                    if valid {
                        hits += 1;
                    } else {
                        edge += 1;
                    }
                }
                nonce += 1;
            }
            assert!(
                hits >= 2 && edge >= 2,
                "factor {factor}: {hits} hits, {edge} boundary nonces found"
            );
        }
    }
}

/// The block's LFM grind, from the pin's options (20 at 828e1d8bc).
fn block_lfm_grind() -> u8 {
    let opts = super::proof::block_tree_options();
    assert_eq!(opts.format.base, BLOCK_LFM);
    opts.grinding_factor
}

/// `Σ 1/(z − (LfmPublic + index·α + Σ lane_l·α^{2+l}))`, as
/// `proof::expected_public_balance`.
fn public_balance(words: &[(u32, LfmWord)], z: FEE, alpha: FEE) -> FEE {
    let bus = FEE::from(crate::tables::types::BusId::LfmPublic as u64);
    let mut powers = [FEE::zero(); 5];
    powers[0] = alpha;
    for i in 1..5 {
        powers[i] = powers[i - 1] * alpha;
    }
    words.iter().fold(FEE::zero(), |acc, (index, word)| {
        let mut f = bus + FEE::from(*index as u64) * powers[0];
        for (l, lane) in word.iter().enumerate() {
            f += FEE::from(GoldilocksField::canonical(lane.value())) * powers[1 + l];
        }
        acc + (z - f).inv().expect("no collision")
    })
}

/// ★ Claim 8. Lanes `0x1_FFFFFFFF` and `0xFFFFFF00_00000000` are both
/// canonical; their little-endian bytes read big-endian are `0xFFFFFFFF01000000`
/// (≥ p, reduced to `0xFFFFFF`) and `0x0000000000FFFFFF`. Under the block's LFM
/// statement the two words absorb alike: the host's P1 transcript and the
/// node's in-guest replay reach one state. The `LfmPublic` closure takes the
/// lane VALUES, on the host and in-guest, so the two words' balances differ:
/// the alias moves no claim.
#[test]
fn aliased_public_lanes_share_the_transcript_not_the_closure() {
    let tag = Block::lfm_statement_tag(&super::proof::block_tree_options().format);
    let id = [0x5Au8; 32];
    let mk = |lane0: u64| -> Vec<(u32, LfmWord)> {
        vec![
            (
                0,
                [
                    FE::from(lane0),
                    FE::from(7u64),
                    FE::from(8u64),
                    FE::from(9u64),
                ],
            ),
            (
                1,
                [
                    FE::from(1u64),
                    FE::from(2u64),
                    FE::from(3u64),
                    FE::from(4u64),
                ],
            ),
        ]
    };
    let (w, w_alias) = (mk(0x1_FFFF_FFFF), mk(0xFFFF_FF00_0000_0000));
    assert_ne!(w, w_alias);

    // Host: one state.
    let host_state = |words: &[(u32, LfmWord)]| {
        let mut t = Block::transcript(&[]);
        super::statement::absorb_lfm_statement(&mut t, &tag, &id, words, 8);
        let s = Block::state_word(&t);
        let draw =
            IsTranscript::<crate::tables::types::GoldilocksExtension>::sample_field_element(&mut t);
        (s, draw)
    };
    assert_eq!(
        host_state(&w),
        host_state(&w_alias),
        "the aliased lanes absorb alike on the host"
    );

    // In-guest: the node's statement replay and its closure over the hinted
    // words, at a fixed (z, α).
    let (z, alpha) = (
        FEE::new([FE::from(123u64), FE::from(456u64), FE::from(789u64)]),
        FEE::new([FE::from(31u64), FE::from(41u64), FE::from(59u64)]),
    );
    let mut b = LfmBuilder::new().with_wrap_hash(BLOCK_WRAP);
    let arena = b.declare_arena((8 * w.len()) as u32);
    let publics = super::per_table_aggregator::hint_public_words(&mut b, arena, w.len());
    let mut t = super::transcript_replay::TranscriptReplay::new(&[]);
    super::per_table_aggregator::emit_lfm_statement(&mut t, &tag, &id, &publics, 8);
    let state = t.state(&mut b);
    b.public(state.cells()[0]);
    let (ez, ea) = (b.ext_const(&z), b.ext_const(&alpha));
    let balance = super::per_table_aggregator::emit_public_balance(&mut b, &publics, ez, ea);
    b.public(balance.as_cell());
    let program = compile(b.finish());
    super::validator::validate(&program).expect("admissible");
    let run = |words: &[(u32, LfmWord)]| -> (LfmWord, LfmWord) {
        let arena_words: Vec<LfmWord> = super::per_table_aggregator::publics_arena(
            &words.iter().map(|(_, w)| *w).collect::<Vec<_>>(),
        );
        let exec =
            execute_serial(&program, &[arena_words], &HasherKind::Poseidon1W16).expect("executes");
        (exec.public_words[0].1, exec.public_words[1].1)
    };
    let (gs, gb) = run(&w);
    let (gs_alias, gb_alias) = run(&w_alias);
    assert_eq!(gs, host_state(&w).0, "the guest's state is the host's");
    assert_eq!(gs, gs_alias, "the aliased lanes absorb alike in-guest");
    assert_eq!(gb, super::word::ext_word(&public_balance(&w, z, alpha)));
    assert_eq!(
        gb_alias,
        super::word::ext_word(&public_balance(&w_alias, z, alpha))
    );
    assert_ne!(gb, gb_alias, "the closure tells the aliased words apart");
}

/// ★ REV-P1-JUDGE's F2 and F4 on the LFM path. F2: the block's LFM tag reads
/// `…/P1W16/C2`, and no other cap policy the block path admits renders it
/// (`Off` and `Fixed(0)` share `C0`, one geometry; `Fixed(c > 8)`, which the
/// trees would clamp, is refused). F4: at the block's LFM cap no tree of any
/// depth is uncapped, so no LFM tree reaches the odd-top padding case: every
/// tree keeps `c ≥ 1`, and the walk below the cap is whole 4-ary levels.
#[test]
fn only_the_block_cap_renders_its_tag_and_no_lfm_tree_is_uncapped() {
    let block = super::proof::block_tree_options();
    let tag = Block::lfm_statement_tag(&block.format);
    assert_eq!(tag, b"LAMBDAVM_LFM_STATEMENT_V1/P1W16/C2".to_vec());
    let mut policies = vec![CapPolicy::Off, CapPolicy::Auto];
    policies.extend((0..=u8::MAX).map(CapPolicy::Fixed));
    let mut admitted = 0;
    for policy in policies {
        let mut format = block.format;
        format.base.arity4_cap = policy;
        let ok = crate::hash_pin::checked_base(&format).is_ok();
        if let CapPolicy::Fixed(c) = policy {
            assert_eq!(ok, c <= 8, "{policy:?}: admitted iff no tree clamps it");
        }
        if !ok {
            continue;
        }
        admitted += 1;
        let rendered = Block::lfm_statement_tag(&format);
        assert_eq!(
            rendered == tag,
            policy == CapPolicy::Fixed(2),
            "{policy:?} renders {}",
            String::from_utf8_lossy(&rendered)
        );
    }
    assert_eq!(admitted, 11, "Off, Auto and Fixed(0..=8)");

    let policy = stark::config::cap_policy_at_arity(&block.format, 4);
    assert_eq!(policy, CapPolicy::Fixed(2));
    for depth in 1..=40usize {
        let c = stark::merkle_caps::StarkCaps::tree_cap_height(
            policy,
            block.fri_number_of_queries,
            depth,
            4,
        );
        assert!(c >= 1, "depth {depth}: an uncapped LFM tree");
        let walked = depth - super::merkle_cap::mux_bits(depth, c, 4);
        assert_eq!(
            walked % 2,
            0,
            "depth {depth}: the odd top sits inside the cap"
        );
    }
}

// ============================ the grind, again ============================

fn felt(seed: u64, i: usize) -> FE {
    let mut z = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(i as u64 + 1);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    FE::from(z ^ (z >> 31))
}

fn run(b: LfmBuilder, arenas: &[Vec<LfmWord>]) -> Result<Vec<FE>, String> {
    let program = compile(b.finish());
    execute_serial(&program, arenas, &HasherKind::Poseidon1W16)
        .map(|e| e.public_words.iter().flat_map(|(_, w)| *w).collect())
        .map_err(|e| format!("{e:?}"))
}

/// The in-guest P1 grind (`epoch::emit_grinding_check`) over a hinted seed and
/// nonce: whether it executes.
fn guest_grind(lanes: [FE; 4], nonce: u64, g: u8) -> bool {
    let mut b = LfmBuilder::new().with_wrap_hash(BLOCK_WRAP);
    let a_seed = b.declare_arena(1);
    let a_nonce = b.declare_arena(1);
    let s = WrapDigest::from_cell(b.hint_word(a_seed, 0));
    let n = b.hint_felt(a_nonce, 0);
    super::epoch::emit_grinding_check(&mut b, s, n, g);
    run(b, &[vec![lanes], vec![base_word(FE::from(nonce))]]).is_ok()
}

/// ★ The in-guest Poseidon1 grind accepts exactly the host's nonces
/// (`is_valid_nonce::<P1GrindDigest>`): every nonce the host accepts at `g`
/// bits executes, and every nonce the host accepts at `g − 1` bits but not
/// at `g` (the boundary bit) is refused, at small factors, at the LFM
/// proofs' own factor (their options'), over seeds with lanes near `p` and nonces
/// at and past `p`. A grind weakened by one bit (`epoch.rs`
/// `bits[64 − factor..]` → `bits[65 − factor..]`) turns this red; the
/// existing `the_grinding_check_is_p1s` stays green under it.
#[test]
fn the_emitted_p1_grind_refuses_the_boundary_bit_and_nonces_past_p() {
    let seeds: [[FE; 4]; 3] = [
        core::array::from_fn(|i| felt(11, i)),
        [
            FE::from(P - 1),
            FE::from(P - 2),
            FE::from(0u64),
            FE::from(1u64),
        ],
        core::array::from_fn(|i| felt(97, i)),
    ];
    let mut boundary = 0usize;
    for lanes in seeds {
        let seed = digest_to_commitment(&lanes);
        for g in [1u8, 2, 3, 5, 8] {
            let (mut ok, mut edge, mut miss) = (0, 0, 0);
            let mut n = 0u64;
            while (ok < 6 || (g > 1 && edge < 4) || miss < 4) && n < 1 << 16 {
                let host = is_valid_nonce::<P1GrindDigest>(&seed, n, g);
                let at_edge = g > 1 && !host && is_valid_nonce::<P1GrindDigest>(&seed, n, g - 1);
                let wanted = if host {
                    ok < 6
                } else if at_edge {
                    edge < 4
                } else {
                    miss < 4
                };
                if wanted {
                    assert_eq!(
                        guest_grind(lanes, n, g),
                        host,
                        "g {g} nonce {n:#x}: the guest and the host disagree{}",
                        if at_edge { " at the boundary bit" } else { "" }
                    );
                    if host {
                        ok += 1;
                    } else if at_edge {
                        edge += 1;
                        boundary += 1;
                    } else {
                        miss += 1;
                    }
                }
                n += 1;
            }
            assert!(
                ok >= 6 && miss >= 4 && (g == 1 || edge >= 4),
                "g {g}: {ok}/{edge}/{miss}"
            );
            // Nonces at and past p: both sides read the nonce as a felt.
            for n in [P, P + 1, P + 3, u64::MAX] {
                assert_eq!(
                    guest_grind(lanes, n, g),
                    is_valid_nonce::<P1GrindDigest>(&seed, n, g),
                    "g {g} nonce {n:#x} past p"
                );
            }
        }
        // The block's own factor (its LFM options'): one accepted nonce and one
        // at the boundary.
        let g = block_lfm_grind();
        let good = (0u64..).find(|&n| is_valid_nonce::<P1GrindDigest>(&seed, n, g));
        let edge = (0u64..).find(|&n| {
            !is_valid_nonce::<P1GrindDigest>(&seed, n, g)
                && is_valid_nonce::<P1GrindDigest>(&seed, n, g - 1)
        });
        let (good, edge) = (good.expect("a nonce"), edge.expect("a boundary nonce"));
        assert!(
            guest_grind(lanes, good, g),
            "g {g}: the host's nonce {good:#x}"
        );
        assert!(
            !guest_grind(lanes, edge, g),
            "g {g}: nonce {edge:#x} has one zero bit too few"
        );
        boundary += 1;
    }
    assert!(boundary >= 40, "{boundary} boundary nonces tried");
}

/// The grind as the STARK leg runs it (`epoch::emit_table_challenges`): the
/// replayed transcript's state is the seed, the check, then the nonce
/// absorbed as a felt; the state after it is the host's after
/// `append_bytes(nonce.to_be_bytes())`, and the machine refuses every nonce
/// the host refuses.
#[test]
fn the_legs_grind_path_is_the_hosts() {
    let mut accepted = 0usize;
    let mut refused = 0usize;
    for case in 0..3u64 {
        let prefix = (case + 1).to_le_bytes();
        let ext: [FE; 3] = core::array::from_fn(|i| felt(case + 40, i));
        let mut host = P1Transcript::with_seed(b"p3b");
        host.append_bytes(&prefix);
        host.append_field_element(&crate::tables::types::FEE::new(ext));
        let seed = host.state();
        for g in [2u8, 5] {
            for n in (0..24u64).chain([P, P + 2, u64::MAX]) {
                let want = is_valid_nonce::<P1GrindDigest>(&seed, n, g);
                let mut b = LfmBuilder::new().with_wrap_hash(BLOCK_WRAP);
                let a = b.declare_arena(4);
                let ev: [_; 3] = core::array::from_fn(|i| b.hint_felt(a, i as u32));
                let nonce = b.hint_felt(a, 3);
                let mut t = TranscriptReplay::new(b"p3b");
                t.append_const_bytes(&prefix);
                t.append_ext(&mut b, ev);
                let s = t.state(&mut b);
                super::epoch::emit_grinding_check(&mut b, s, nonce, g);
                t.append_felt(&mut b, nonce);
                let after = t.state(&mut b);
                b.public(after.cells()[0]);
                let arena: Vec<LfmWord> = [ext[0], ext[1], ext[2], FE::from(n)]
                    .map(base_word)
                    .to_vec();
                let got = run(b, &[arena]);
                assert_eq!(got.is_ok(), want, "case {case} g {g} nonce {n:#x}: {got:?}");
                if want {
                    accepted += 1;
                    let mut h = host.clone();
                    h.append_bytes(&n.to_be_bytes());
                    assert_eq!(
                        got.expect("accepted"),
                        commitment_to_digest(&h.state()).to_vec(),
                        "case {case} g {g} nonce {n:#x}: the state after the nonce"
                    );
                } else {
                    refused += 1;
                }
            }
        }
    }
    assert!(accepted >= 10 && refused >= 10, "{accepted} / {refused}");
}

// ======================= the configuration check ========================

/// The configuration check (`proof::check_configuration`) admits a proof
/// only when the options' base and the artifacts' roots both name the
/// configuration's hash; it names the hash, not the cap (the cap is the
/// options', which the block verifier derives itself).
#[test]
fn the_configuration_check_refuses_every_mismatch() {
    use super::proof::check_configuration;
    let block = super::proof::block_tree_options();
    let legacy = super::proof::aggregation_wrap_options();
    let rpx = crate::hash_pin::LEGACY_COMMITMENT_HASH;
    let p1 = CommitmentHash::Poseidon1;
    assert_eq!(block.format.base, BLOCK_LFM);
    assert!(check_configuration::<Block>(p1, &block).is_ok());
    assert!(check_configuration::<Legacy>(rpx, &legacy).is_ok());
    for (what, ok) in [
        (
            "block: legacy roots",
            check_configuration::<Block>(rpx, &block).is_ok(),
        ),
        (
            "block: legacy options",
            check_configuration::<Block>(p1, &legacy).is_ok(),
        ),
        (
            "block: both legacy",
            check_configuration::<Block>(rpx, &legacy).is_ok(),
        ),
        (
            "legacy: P1 roots",
            check_configuration::<Legacy>(p1, &legacy).is_ok(),
        ),
        (
            "legacy: block options",
            check_configuration::<Legacy>(rpx, &block).is_ok(),
        ),
    ] {
        assert!(!ok, "{what} admitted");
    }
    let with = |hash, cap| {
        let mut o = block.clone();
        o.format.base = BaseFormat {
            hash,
            arity4_cap: cap,
        };
        o
    };
    assert!(
        check_configuration::<Block>(p1, &with(CommitmentHash::Keccak256, CapPolicy::Fixed(2)))
            .is_err(),
        "a hash the block path has no configuration for"
    );
    assert!(
        check_configuration::<Block>(p1, &with(p1, CapPolicy::Fixed(9))).is_err(),
        "a cap the 4-ary trees would clamp"
    );
    // The hash, not the cap: another cap passes here, and is the verifier's
    // own options' (its geometry and tag), never the proof's.
    assert!(check_configuration::<Block>(p1, &with(p1, CapPolicy::Fixed(1))).is_ok());
}

// ============== the block verifier's and harvest's pin checks ============

fn empty_counts() -> crate::TableCounts {
    crate::TableCounts {
        cpu: 0,
        lt: 0,
        memw: 0,
        memw_aligned: 0,
        load: 0,
        mul: 0,
        dvrm: 0,
        shift: 0,
        branch: 0,
        memw_register: 0,
        eq: 0,
        bytewise: 0,
        store: 0,
        cpu32: 0,
        keccak: 0,
        keccak_rnd: 0,
        ecsm: 0,
        ecdas: 0,
        hint: 0,
        commit: 0,
        blake3: 0,
    }
}

/// The block verifier refuses, before any work, presets whose base or LFM
/// hash is not the pin's: a legacy base under the pin's tree options, and the
/// pin's base under the legacy LFM options (`block_plan.rs` `verify_under`).
#[test]
fn the_block_verifier_refuses_a_legacy_hash_first() {
    use super::block_plan::{BlockShape, verify_block_tree_under};
    let shape = BlockShape {
        table_counts: empty_counts(),
        runtime_page_ranges: Vec::new(),
        num_private_input_pages: 0,
        public_output_len: 0,
        trace_lengths: Vec::new(),
    };
    let top = LfmProof {
        proof: MultiProof { proofs: Vec::new() },
        public_words: Vec::new(),
    };
    for (what, opts, wrap) in [
        (
            "a legacy base",
            super::proof::block_base_options_for(BaseFormat::RPX),
            super::proof::block_tree_options(),
        ),
        (
            "legacy LFM options",
            super::proof::block_tree_base_options(),
            super::proof::aggregation_wrap_options(),
        ),
    ] {
        let e =
            verify_block_tree_under(&[], &opts, &wrap, None, &shape, &[], &top).expect_err(what);
        assert!(e.contains("block pin's hash alone"), "{what}: {e}");
    }
}

/// The tree's base harvest admits only the pin's base: a legacy-base format is
/// refused before the proof is read (`block_tree.rs` `harvest_block_over`).
#[cfg(feature = "parallel")]
#[test]
fn the_tree_harvest_refuses_a_legacy_base_first() {
    let proof = crate::VmProof {
        proof: MultiProof { proofs: Vec::new() },
        runtime_page_ranges: Vec::new(),
        table_counts: empty_counts(),
        public_output: Vec::new(),
        num_private_input_pages: 0,
    };
    let opts = super::proof::block_base_options_for(BaseFormat::RPX);
    match super::block_tree::harvest_block_over(
        &opts,
        &[],
        &proof,
        true,
        None,
        &super::block_tree::StdoutSink,
    ) {
        Err(e) => assert!(e.contains("under the block pin alone"), "{e}"),
        Ok(_) => panic!("a legacy base was harvested"),
    }
}
