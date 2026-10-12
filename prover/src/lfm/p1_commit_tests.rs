//! The Poseidon1 configuration's seams against the host oracle
//! (`crypto::hash::poseidon1_stark`, checked against ZisK's code).

use super::*;
use crypto::hash::poseidon1_stark::Merkle4;
use crypto::merkle_tree::merkle::MerkleTree;

type Base = P1BatchBackend<GoldilocksField>;
type Ext = P1BatchBackend<GoldilocksExtension>;

fn fe(i: u64) -> FE {
    FE::from(i.wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

fn fee(i: u64) -> FEE {
    FEE::new([fe(3 * i), fe(3 * i + 1), fe(3 * i + 2)])
}

#[test]
fn hash_bytes_agrees_with_hash_data() {
    for w in [1usize, 5, 12, 13, 40] {
        let row: Vec<FE> = (0..w as u64).map(fe).collect();
        let bytes: Vec<u8> = row
            .iter()
            .flat_map(|f| GoldilocksField::canonical(f.value()).to_be_bytes())
            .collect();
        assert_eq!(Base::hash_bytes(&bytes), Base::hash_data(&row), "width {w}");
        assert_eq!(Base::hash_data(&row), leaf(&row));
    }
}

#[test]
fn the_pair_and_batched_families_agree_on_a_two_element_leaf() {
    // `StarkHash`'s load-bearing invariant: the prover commits FRI layers with
    // `Pair` and the verifier authenticates them with `Batched`.
    for i in 0..8 {
        let (a, b) = (fee(2 * i), fee(2 * i + 1));
        assert_eq!(
            P1PairBackend::<GoldilocksExtension>::hash_data(&[a, b]),
            Ext::hash_data(&vec![a, b])
        );
    }
}

#[test]
fn a_tree_of_rows_is_the_oracles_tree() {
    for n in [1usize, 2, 8, 32, 128] {
        let rows: Vec<Vec<FE>> = (0..n as u64)
            .map(|r| (0..7).map(|c| fe(r * 7 + c)).collect())
            .collect();
        let tree = MerkleTree::<Base>::build(&rows).expect("non-empty");
        let leaves: Vec<_> = rows.iter().map(|r| linear_hash(r)).collect();
        let oracle = Merkle4::new(&leaves).expect("non-empty");
        assert_eq!(tree.root, digest_to_commitment(&oracle.root()), "{n} rows");
        let path = tree.get_proof_by_pos(n - 1).expect("in range");
        assert!(path.verify::<Base>(&tree.root, n - 1, &rows[n - 1]));
    }
}

#[test]
fn the_grinding_hash_is_one_width_8_permutation_per_nonce() {
    let seed = [7u8; 32];
    let factor = 8u8;
    let inner = crypto::grinding::inner_hash_felts::<P1GrindDigest>(&seed, factor);
    let nonce = crypto::grinding::generate_nonce_smallest::<P1GrindDigest>(&seed, factor)
        .expect("a nonce at 8 bits");
    assert!(crypto::grinding::is_valid_nonce::<P1GrindDigest>(
        &seed, nonce, factor
    ));
    // The per-nonce hash, by hand: [inner0..3, nonce, 0, 0, 0] through W8.
    let mut s = [FE::zero(); 8];
    for (lane, v) in s.iter_mut().zip(inner) {
        *lane = FE::from(v);
    }
    s[4] = FE::from(nonce);
    let lane0 = GoldilocksField::canonical(poseidon1_w8::permute(s)[0].value());
    assert!(lane0 < 1u64 << (64 - factor), "lane 0 meets the factor");
    if nonce > 0 {
        assert!(!crypto::grinding::is_valid_nonce::<P1GrindDigest>(
            &seed,
            nonce - 1,
            factor
        ));
    }
}

#[test]
fn the_transcript_binds_lengths_bytes_and_field_elements() {
    let first = |f: &dyn Fn(&mut P1Transcript)| {
        let mut t = P1Transcript::new();
        f(&mut t);
        <P1Transcript as IsTranscript<GoldilocksExtension>>::sample_field_element(&mut t)
    };
    let root = [9u8; 32];
    let a = first(&|t| t.append_bytes(&root));
    // A 32-byte root whose tail is zero, against the 8-byte integer it starts with.
    let mut short_root = [0u8; 32];
    short_root[..8].copy_from_slice(&root[..8]);
    assert_ne!(
        first(&|t| t.append_bytes(&short_root)),
        first(&|t| t.append_bytes(&root[..8]))
    );
    for i in 0..root.len() {
        let mut bad = root;
        bad[i] ^= 1;
        assert_ne!(first(&|t| t.append_bytes(&bad)), a, "byte {i}");
    }
    let x = fee(5);
    let b = first(&|t| t.append_field_element(&x));
    for k in 0..3 {
        let mut v = *x.value();
        v[k] += FE::one();
        assert_ne!(
            first(&|t| t.append_field_element(&FEE::new(v))),
            b,
            "coefficient {k}"
        );
    }
}

#[test]
fn reading_the_state_does_not_move_the_stream() {
    let mut a = P1Transcript::with_seed(b"seed");
    let mut b = P1Transcript::with_seed(b"seed");
    a.append_bytes(&[1, 2, 3]);
    b.append_bytes(&[1, 2, 3]);
    let s = <P1Transcript as IsTranscript<GoldilocksExtension>>::state(&a);
    assert_eq!(
        s,
        <P1Transcript as IsTranscript<GoldilocksExtension>>::state(&b)
    );
    for _ in 0..20 {
        assert_eq!(
            <P1Transcript as IsTranscript<GoldilocksExtension>>::sample_u64(&mut a, 1 << 20),
            <P1Transcript as IsTranscript<GoldilocksExtension>>::sample_u64(&mut b, 1 << 20)
        );
    }
    assert_ne!(
        <P1Transcript as IsTranscript<GoldilocksExtension>>::state(&a),
        s,
        "squeezing moves the state"
    );
}
