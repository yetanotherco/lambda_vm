//! The adversarial review REV-P1-A's prover tests (the subset REV-P1-JUDGE §7
//! keeps), flipped where the judge's fixes change their expectation: the
//! uncapped odd-depth top's padding is bound on the host as in-guest (F4), the
//! statement tag names the cap policy (F2), and a raw sample under Poseidon1 is
//! refused loudly.

use crate::tables::types::{FE, GoldilocksField};
use crypto::merkle_tree::cap::CappedRoot;
use crypto::merkle_tree::traits::IsMerkleTreeBackend;
use stark::config::Commitment;
use stark::proof::options::CapPolicy;

use super::algebraic_commit::{commitment_to_digest, digest_to_commitment};
use super::builder::LfmBuilder;
use super::compiler::compile;
use super::edsl::{WrapDigest, WrapHash};
use super::executor::execute_serial;
use super::hash::HasherKind;
use super::p1w16_emit::hint_order;
use super::word::{LfmWord, base_word};

type B = super::p1_commit::P1BatchBackend<GoldilocksField>;

fn d(seed: u64) -> Commitment {
    digest_to_commitment(&core::array::from_fn(|k| {
        FE::from(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (k as u64 + 1))
    }))
}

/// ★ Host and guest agree at an UNCAPPED odd-depth tree (`arity4_cap = Off`):
/// the top level's two padding siblings must be the padding node on the host
/// (REV-P1-JUDGE F4), as the guest's walk supplies them as constants. A root
/// over non-zero padding opens on neither, at every real index.
#[test]
fn rp1a_uncapped_odd_top_host_and_guest_refuse() {
    let depth = 3usize;
    let leaves: Vec<Commitment> = (0..8).map(|i| d(100 + i)).collect();
    let n0 = B::hash_four(&[leaves[0], leaves[1], leaves[2], leaves[3]]);
    let n1 = B::hash_four(&[leaves[4], leaves[5], leaves[6], leaves[7]]);
    let junk = d(7);
    let forged = B::hash_four(&[n0, n1, junk, junk]);

    for index in [0usize, 5] {
        let base = index / 4 * 4;
        let mut path: Vec<Commitment> = (base..base + 4)
            .filter(|&c| c != index)
            .map(|c| leaves[c])
            .collect();
        path.extend([if index < 4 { n1 } else { n0 }, junk, junk]);

        // Host (`stark::merkle_caps::TreeCheck` at c = 0 is this check).
        assert!(
            !CappedRoot::uncapped(&forged, depth).verify::<B>(&path, index, leaves[index]),
            "host refuses index {index}"
        );

        // Guest: the harvest's layout (`path_to_cap` → `hint_order`, which
        // drops the two padding siblings), walked and compared with the root.
        let triples: Vec<[Commitment; 3]> =
            path.chunks_exact(3).map(|t| [t[0], t[1], t[2]]).collect();
        let hints = hint_order(index, depth, &triples);
        let word = |c: &Commitment| -> LfmWord { commitment_to_digest(c) };
        let arenas: Vec<Vec<LfmWord>> = vec![
            vec![word(&leaves[index])],
            vec![base_word(FE::from(index as u64))],
            hints.iter().map(word).collect(),
            vec![word(&forged)],
        ];
        let mut b = LfmBuilder::new().with_wrap_hash(WrapHash::Poseidon1);
        let ids: Vec<_> = arenas
            .iter()
            .map(|a| b.declare_arena(a.len() as u32))
            .collect();
        let leaf = WrapDigest::from_cell(b.hint_word(ids[0], 0));
        let idx = b.hint_felt(ids[1], 0);
        let bits = b.bit_dec(idx, depth);
        let h: Vec<WrapDigest> = (0..arenas[2].len() as u32)
            .map(|i| WrapDigest::from_cell(b.hint_word(ids[2], i)))
            .collect();
        let root = b.hint_word(ids[3], 0);
        let root_lanes = [b.unpack(root)];
        let walked = super::edsl::wrap_merkle_walk(&mut b, leaf, &bits, &h);
        super::edsl::assert_digest_eq_lanes(&mut b, walked, &root_lanes);
        let program = compile(b.finish());
        super::validator::validate(&program).expect("admissible");
        assert!(
            execute_serial(&program, &arenas, &HasherKind::Poseidon1W16).is_err(),
            "guest refuses index {index}"
        );
    }
}

/// The P1 statement tag names the cap policy (REV-P1-JUDGE F2): `Auto`, which
/// caps every tree (height 3 at 110 openings), and `Off` read apart, and a
/// `Fixed(c)` past the arity-4 clamp, which would name a height no tree has, is
/// refused by the format check.
#[test]
fn rp1a_the_statement_tag_names_the_effective_cap() {
    use crate::hash_pin::{checked_base, p1_statement_tag};
    use stark::merkle_caps::StarkCaps;
    use stark::proof::options::{BaseFormat, ProofFormat};
    assert_ne!(
        p1_statement_tag(CapPolicy::Auto),
        p1_statement_tag(CapPolicy::Off)
    );
    assert_eq!(
        p1_statement_tag(CapPolicy::Off),
        p1_statement_tag(CapPolicy::Fixed(0))
    );
    let p1_at = |c: CapPolicy| ProofFormat {
        base: BaseFormat {
            arity4_cap: c,
            ..BaseFormat::P1
        },
        ..ProofFormat::LEGACY
    };
    assert!(checked_base(&p1_at(CapPolicy::Fixed(9))).is_err());
    assert!(checked_base(&p1_at(CapPolicy::Fixed(8))).is_ok());
    assert!(checked_base(&p1_at(CapPolicy::Auto)).is_ok());
    assert_eq!(
        StarkCaps::tree_cap_height(CapPolicy::Auto, 110, 22, 4),
        3,
        "Auto caps a P1 tree"
    );
    assert_eq!(StarkCaps::tree_cap_height(CapPolicy::Off, 110, 22, 4), 0);
    assert_eq!(
        p1_statement_tag(CapPolicy::Fixed(9)),
        b"LAMBDAVM_STARK_STATEMENT_V5/P1W16/C9".to_vec()
    );
    assert_eq!(
        StarkCaps::tree_cap_height(CapPolicy::Fixed(9), 110, 40, 4),
        8,
        "clamped to MAX_CAP_HEIGHT / 2"
    );
    assert_eq!(
        StarkCaps::tree_cap_height(CapPolicy::Fixed(8), 110, 40, 4),
        8,
        "Fixed(8) and Fixed(9) would be one geometry under two tags: Fixed(9) is refused"
    );
}

/// `TranscriptReplay::sample` (the keccak duplex's raw sample) has NO
/// Poseidon1 arm: under a Poseidon1 builder it falls to the RPX chain, which
/// would drain the pending segment away from ZisK's sponge. It is caught
/// LOUDLY, not silently: the chain's rows are twelve-felt `Hash` rows, and a
/// program that also has the socket's `Hash16` rows is refused by the
/// validator (`MixedHashWidths`).
#[test]
fn rp1a_a_raw_sample_under_poseidon1_is_refused_not_silent() {
    use super::transcript_replay::TranscriptReplay;
    let mut b = LfmBuilder::new().with_wrap_hash(WrapHash::Poseidon1);
    let a = b.declare_arena(1);
    let w = b.hint_word(a, 0);
    let mut t = TranscriptReplay::new(b"seed");
    let f = t.sample_felt(&mut b); // drives ZisK's sponge: Hash16 rows
    b.public(f.as_cell());
    t.append_word(&mut b, w); // lazily pending
    let s = t.sample(&mut b); // the missing arm: the RPX chain takes it
    b.public(s.cells()[0]);
    let f2 = t.sample_felt(&mut b);
    b.public(f2.as_cell());
    let program = compile(b.finish());
    let err = super::validator::validate(&program).expect_err("mixed widths");
    assert!(
        matches!(err, super::validator::LfmViolation::MixedHashWidths { .. }),
        "{err:?}"
    );
}
