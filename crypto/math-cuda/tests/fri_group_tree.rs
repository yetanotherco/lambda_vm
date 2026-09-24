//! S3 group-leaf FRI layers on the device (FRI.md §5, lane I-FRI-D).
//!
//! - The group-leaf trees (`build_fri_group_tree_from_evals_ext3`, the kernels
//!   `FriCommitState::fold_and_commit_group` commits with) equal the host tree
//!   node for node: leaf `g` = the configuration's `Batched` leaf over the
//!   `2^d` consecutive values from `g·2^d`, parents the pair hash. Keccak and
//!   Blake3 against the host backends, d = 1..=6 at several sizes.
//! - Against the checked-in S3 vectors (`crypto/stark/tests/vectors/zf_fri`):
//!   (c) the first-leaf digest and the layer root of the KAT codeword for
//!   d = 1..=6 under Keccak, Blake3 AND RPX (the host RPX backend lives in the
//!   prover crate; the vector is its output); (b) the KAT codeword folded d
//!   times on the device with ζ, ζ², … equals the vector's `folded`.
//! - At d = 1 the group kernel IS the legacy pair-leaf kernel (the two-element
//!   invariant, on the device), under all three hashes.
//!
//! Needs a GPU.

use crypto::merkle_tree::merkle::MerkleTree;
use crypto::merkle_tree::traits::IsStreamingLeafBackend;
use math::fft::bit_reversing::in_place_bit_reverse_permute;
use math::fft::roots_of_unity::get_powers_of_primitive_root_coset;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;
use math_cuda::DeviceHash;
use math_cuda::fri::{FriCommitState, build_fri_group_tree_from_evals_ext3};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use stark::config::{Blake3StarkHash, KeccakStarkHash, StarkHash};

type F = GoldilocksField;
type E = Degree3GoldilocksExtensionField;
type Fp = FieldElement<F>;
type Fp3 = FieldElement<E>;

fn vectors_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../stark/tests/vectors/zf_fri")
}

fn read_vector(name: &str) -> String {
    std::fs::read_to_string(vectors_dir().join(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// Every decimal integer in `s`, in order (the vectors' ext limbs).
fn u64s(s: &str) -> Vec<u64> {
    let mut out = Vec::new();
    let mut cur: Option<u64> = None;
    for ch in s.chars() {
        match ch.to_digit(10) {
            Some(d) => cur = Some(cur.unwrap_or(0) * 10 + u64::from(d)),
            None => {
                if let Some(v) = cur.take() {
                    out.push(v);
                }
            }
        }
    }
    if let Some(v) = cur {
        out.push(v);
    }
    out
}

/// The value of `"key": [...]` on `line`, up to the bracket that closes it.
fn json_array<'a>(line: &'a str, key: &str) -> &'a str {
    let start = line
        .find(&format!("\"{key}\": ["))
        .unwrap_or_else(|| panic!("no {key}"))
        + key.len()
        + 4;
    let mut depth = 0i32;
    for (i, ch) in line[start..].char_indices() {
        match ch {
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth == 0 {
                    return &line[start..start + i + 1];
                }
            }
            _ => {}
        }
    }
    panic!("unterminated {key}")
}

fn hex32(s: &str) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, b) in out.iter_mut().enumerate() {
        *b = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).expect("hex");
    }
    out
}

/// The vector (b) KAT codeword as interleaved limbs, and per d its ζ and the
/// folded codeword.
#[allow(clippy::type_complexity)]
fn fold_vector() -> (Vec<u64>, Vec<(u32, [u64; 3], Vec<u64>)>) {
    let text = read_vector("b_group_folds.json");
    let codeword_line = text
        .lines()
        .find(|l| l.trim_start().starts_with("\"codeword\""))
        .expect("codeword line");
    let codeword = u64s(json_array(codeword_line, "codeword"));
    assert_eq!(codeword.len(), 3 * 128);
    let mut folds = Vec::new();
    for line in text.lines().filter(|l| l.contains("\"folded\"")) {
        let d = u64s(&line[..line.find("\"zeta\"").expect("zeta")])[0] as u32;
        let z = u64s(json_array(line, "zeta"));
        let folded = u64s(json_array(line, "folded"));
        assert_eq!(folded.len(), 3 * (128 >> d));
        folds.push((d, [z[0], z[1], z[2]], folded));
    }
    assert_eq!(folds.len(), 6);
    (codeword, folds)
}

/// The vector (c) digests for `hash`: per d, (first leaf, layer root).
fn leaf_vector(hash: &str) -> Vec<(u32, [u8; 32], [u8; 32])> {
    let text = read_vector(&format!("c_leaf_digests_{hash}.json"));
    let mut out = Vec::new();
    for line in text.lines().filter(|l| l.contains("\"first_leaf\"")) {
        let field = |key: &str| {
            let at = line.find(&format!("\"{key}\": \"")).expect(key) + key.len() + 5;
            hex32(&line[at..at + 64])
        };
        let d = u64s(&line[..line.find("\"first_leaf\"").expect("first_leaf")])[0] as u32;
        out.push((d, field("first_leaf"), field("layer_root")));
    }
    assert_eq!(out.len(), 6, "{hash}: d = 1..=6");
    out
}

fn limbs(v: &[Fp3]) -> Vec<u64> {
    v.iter()
        .flat_map(|e| {
            let c = e.value();
            [c[0].canonical(), c[1].canonical(), c[2].canonical()]
        })
        .collect()
}

fn from_limbs(v: &[u64]) -> Vec<Fp3> {
    v.chunks_exact(3)
        .map(|c| Fp3::new([Fp::from(c[0]), Fp::from(c[1]), Fp::from(c[2])]))
        .collect()
}

/// The host group tree: `H::Batched` leaves over consecutive groups, the pair
/// hash above (as `stark::fri::group_tree`).
fn host_group_nodes<H: StarkHash>(evals: &[Fp3], group: usize) -> Vec<[u8; 32]> {
    let leaves: Vec<[u8; 32]> = evals
        .chunks_exact(group)
        .map(|g| <H::Batched<E> as IsStreamingLeafBackend<E>>::hash_data_from_slices(g, &[]))
        .collect();
    MerkleTree::<H::Pair<E>>::build_from_hashed_leaves(leaves)
        .expect("tree")
        .nodes()
        .to_vec()
}

fn assert_nodes_eq(device: &[u8], host: &[[u8; 32]], what: &str) {
    assert_eq!(device.len(), host.len() * 32, "{what}: node count");
    for (i, h) in host.iter().enumerate() {
        assert_eq!(&device[i * 32..(i + 1) * 32], &h[..], "{what}: node {i}");
    }
}

fn random_evals(n: usize, seed: u64) -> Vec<Fp3> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            Fp3::new([
                Fp::from_raw(rng.r#gen::<u64>()),
                Fp::from_raw(rng.r#gen::<u64>()),
                Fp::from_raw(rng.r#gen::<u64>()),
            ])
        })
        .collect()
}

fn host_parity<H: StarkHash>(hash: DeviceHash, name: &str) {
    for d in 1..=6u32 {
        for extra in [1u32, 4, 9] {
            let n = 1usize << (d + extra);
            let evals = random_evals(n, 1000 + u64::from(d * 16 + extra));
            let raw: Vec<u64> = evals
                .iter()
                .flat_map(|e| {
                    let c = e.value();
                    [*c[0].value(), *c[1].value(), *c[2].value()]
                })
                .collect();
            let device = build_fri_group_tree_from_evals_ext3(&raw, d, hash).expect("device");
            assert_nodes_eq(
                &device,
                &host_group_nodes::<H>(&evals, 1 << d),
                &format!("{name} d={d} n=2^{}", d + extra),
            );
        }
    }
}

#[test]
fn group_tree_matches_host_keccak() {
    host_parity::<KeccakStarkHash>(DeviceHash::Keccak256, "keccak");
}

#[test]
fn group_tree_matches_host_blake3() {
    host_parity::<Blake3StarkHash>(DeviceHash::Blake3, "blake3");
}

#[test]
fn group_tree_matches_the_leaf_vectors() {
    let (codeword, _) = fold_vector();
    for (hash, name) in [
        (DeviceHash::Keccak256, "keccak"),
        (DeviceHash::Blake3, "blake3"),
        (DeviceHash::Rpx256, "rpx"),
    ] {
        for (d, first_leaf, root) in leaf_vector(name) {
            let nodes = build_fri_group_tree_from_evals_ext3(&codeword, d, hash).expect("device");
            let num_leaves = 128usize >> d;
            let leaf0 = (num_leaves - 1) * 32;
            assert_eq!(
                &nodes[leaf0..leaf0 + 32],
                &first_leaf,
                "{name} d={d}: first leaf"
            );
            assert_eq!(&nodes[..32], &root, "{name} d={d}: layer root");
        }
    }
}

#[test]
fn group_of_two_is_the_pair_leaf_kernel() {
    let evals = random_evals(1 << 12, 77);
    let raw = limbs(&evals);
    for (hash, pair) in [
        (
            DeviceHash::Keccak256,
            math_cuda::merkle::build_fri_layer_tree_from_evals_ext3(&raw).expect("keccak"),
        ),
        (
            DeviceHash::Blake3,
            math_cuda::blake3::build_fri_layer_tree_from_evals_ext3(&raw).expect("blake3"),
        ),
        (
            DeviceHash::Rpx256,
            math_cuda::rpx::build_fri_layer_tree_from_evals_ext3(&raw).expect("rpx"),
        ),
    ] {
        let group = build_fri_group_tree_from_evals_ext3(&raw, 1, hash).expect("group");
        assert_eq!(group, pair, "{hash:?}: d = 1 group tree != pair tree");
    }
}

/// `compute_coset_twiddles_inv`: the inverses of the coset points at the even
/// bit-reversed positions (`o·ω^i`, `i < n/2`, bit-reversed, inverted).
fn fold_twiddles(offset: u64, n: usize) -> Vec<u64> {
    let mut pts = get_powers_of_primitive_root_coset::<F>(
        n.trailing_zeros() as u64,
        n / 2,
        &Fp::from(offset),
    )
    .expect("roots");
    in_place_bit_reverse_permute(&mut pts);
    pts.iter()
        .map(|p| p.inv().expect("nonzero").canonical())
        .collect()
}

fn zeta_powers(z: [u64; 3], d: u32) -> Vec<[u64; 3]> {
    let mut zeta = from_limbs(&z)[0];
    let mut out = Vec::new();
    for level in 0..d {
        out.push(limbs(&[zeta]).try_into().expect("3 limbs"));
        if level + 1 < d {
            zeta = zeta.square();
        }
    }
    out
}

/// (b): the device folds (`fold_to_host`, the S3 terminal step) reproduce the
/// vector's `folded` for d = 1..=6; and `fold_and_commit_group` with no fold
/// commits the KAT codeword itself to the (c) root, with d − 1 folds then a
/// group of 2 to the root of the folded codeword's pair tree.
#[test]
fn device_folds_match_the_fold_vector() {
    let (codeword, folds) = fold_vector();
    let tw = fold_twiddles(3, 128);
    for (d, z, folded) in &folds {
        let mut st =
            FriCommitState::new(&codeword, &tw, 128, DeviceHash::Keccak256).expect("state");
        let out = st.fold_to_host(&zeta_powers(*z, *d)).expect("fold");
        assert_eq!(limbs(&from_limbs(&out)), *folded, "d={d}: folded codeword");
    }
    for (hash, name) in [
        (DeviceHash::Keccak256, "keccak"),
        (DeviceHash::Blake3, "blake3"),
        (DeviceHash::Rpx256, "rpx"),
    ] {
        for (d, _, root) in leaf_vector(name) {
            let mut st = FriCommitState::new(&codeword, &tw, 128, hash).expect("state");
            let (host, _, tree) = st.fold_and_commit_group(&[], d, true).expect("commit");
            assert_eq!(tree.root, root, "{name} d={d}: zero-fold group commit root");
            assert_eq!(tree.leaves_len, 128 >> d);
            assert_eq!(
                host.expect("drained"),
                codeword,
                "{name} d={d}: layer evals"
            );
        }
        // Folds then a commit: d folds, then groups of 2 over the result.
        for (d, z, folded) in folds.iter().filter(|(d, _, _)| *d <= 5) {
            let mut st = FriCommitState::new(&codeword, &tw, 128, hash).expect("state");
            let (host, _, tree) = st
                .fold_and_commit_group(&zeta_powers(*z, *d), 1, true)
                .expect("commit");
            let host = host.expect("drained");
            assert_eq!(
                limbs(&from_limbs(&host)),
                *folded,
                "{name} d={d}: layer evals"
            );
            let expect = build_fri_group_tree_from_evals_ext3(folded, 1, hash).expect("tree");
            assert_eq!(
                &tree.root[..],
                &expect[..32],
                "{name} d={d}: folded layer root"
            );
        }
    }
}
