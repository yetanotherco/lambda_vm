//! The WHIR base proof's 7-round BLAKE3 on the card (the opt-in BLAKE3 base): the module
//! loads at 7 rounds whatever the STARK cubin's count, its nonce search
//! returns the host predicate's smallest valid nonce, its coset-ordered base
//! tree is the host's, and — box tier, `#[ignore]` — the microbenchmark that
//! prices it against Poseidon1 at the base proof's shapes.
//!
//! Needs a GPU, like every test in this crate.

use crypto::grinding::{inner_hash_lanes, is_valid_nonce};
use crypto::hash::blake3::Blake3Std;
use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField as F;
use multilinear::mle::Mle;
use multilinear::whir::{self, Domain};
use multilinear::whir_commit::{CodewordCommitment, LeafOrder};
use multilinear::whir_hash::Blake3Whir;

type FE = FieldElement<F>;

/// The module's kernels are standard BLAKE3, and the STARK cubin keeps its own
/// count (6 under the default `blake3-6round`).
#[test]
fn the_whir_module_is_seven_rounds() {
    assert_eq!(
        math_cuda::blake3_whir::device_rounds().expect("needs a GPU"),
        7
    );
    let stark = math_cuda::blake3::device_rounds().expect("needs a GPU");
    println!("STARK BLAKE3 cubin rounds: {stark}");
}

/// At a factor whose smallest nonce is cheap to scan for, the card returns
/// it: valid under `is_valid_nonce::<Blake3Std>`, and no smaller one is.
#[test]
fn the_blake3_grind_returns_the_smallest_valid_nonce() {
    for (seed, factor) in [([14u8; 32], 14u8), ([3u8; 32], 12)] {
        let nonce = math_cuda::grinding::generate_nonce_blake3_gpu(
            &inner_hash_lanes::<Blake3Std>(&seed, factor),
            factor,
        )
        .expect("GPU grind (needs a GPU)");
        assert!(is_valid_nonce::<Blake3Std>(&seed, nonce, factor));
        assert!(
            (0..nonce).all(|n| !is_valid_nonce::<Blake3Std>(&seed, n, factor)),
            "nonce {nonce} is not the smallest (factor {factor})"
        );
    }
}

/// ★ The joint-leaf kernel's prefix midstate is the host's
/// (`blake3_joint::joint_leaf_midstate`) at every run width and the fold
/// widths a round 0 takes: the card restates the prefix's layout, so this is
/// what holds the two statements to one.
#[test]
fn the_joint_leaf_midstate_is_the_hosts() {
    use crypto::merkle_tree::backends::blake3_joint::joint_leaf_midstate;
    for blocks in 2u64..=4 {
        for log_block in [2u32, 4, 6] {
            assert_eq!(
                math_cuda::blake3_whir::joint_midstate(blocks, log_block).expect("needs a GPU"),
                joint_leaf_midstate(blocks, 1 << log_block, 8),
                "{blocks} blocks of 2^{log_block}"
            );
        }
    }
}

/// At the production grind (18 bits) the card's nonce is valid.
#[test]
fn the_blake3_grind_is_valid_at_the_production_factor() {
    let seed = [18u8; 32];
    let nonce = math_cuda::grinding::generate_nonce_blake3_gpu(
        &inner_hash_lanes::<Blake3Std>(&seed, 18),
        18,
    )
    .expect("GPU grind (needs a GPU)");
    assert!(is_valid_nonce::<Blake3Std>(&seed, nonce, 18));
}

/// The coset-ordered base tree (`LeafOrder::Coset`, block `j` at
/// `bitrev(j)`) the card builds under 7-round BLAKE3 is the host's.
#[test]
fn the_coset_ordered_blake3_tree_is_the_hosts() {
    for (num_vars, log_folding) in [(13usize, 6usize), (11, 4)] {
        let f = Mle::new(
            (0..1u64 << num_vars)
                .map(|i| FE::from(i.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 5))
                .collect(),
        )
        .expect("power of two");
        let raw: Vec<u64> = f.evals().iter().map(|v| *v.value()).collect();
        let (_, root) = math_cuda::whir::commit_codeword_parts_placed(
            &[(&raw, 0)],
            num_vars,
            1,
            log_folding,
            true,
            math_cuda::DeviceHash::Blake3,
            math_cuda::whir::LeafPlacement::Reversed,
        )
        .expect("device commit (needs a GPU)");
        let domain = Domain::<F>::new(num_vars + 1).expect("domain");
        let codeword = whir::encode::<F, F>(&whir::lift_coefficients(&f), &domain).expect("encode");
        let host = CodewordCommitment::<F, Blake3Whir>::from_codeword_on_host_in(
            codeword,
            log_folding,
            LeafOrder::Coset,
        )
        .expect("host commit");
        assert_eq!(root, host.root(), "2^{num_vars} at fold {log_folding}");
    }
}

/// ⛔ BOX TIER, a measurement: the card seconds of the WHIR leaf pass, the
/// inner levels and one 18-bit grind under Poseidon1 (the production base),
/// 7-round BLAKE3 and keccak, at the base proof's shapes — a round-0 base
/// codeword of 2^26 and 2^28 values at a fold of 6 (the median block's stack
/// of 2^27 at blowup 2 is 2^28), coset-ordered as production commits it, and
/// a later round's ext3 codeword of 2^22 values at a fold of 4.
///
/// `cargo test -p math-cuda --release --test blake3_whir -- --ignored --nocapture`
#[test]
#[ignore = "box tier: a measurement"]
fn whir_hash_microbench() {
    use math_cuda::DeviceHash::{Blake3, Keccak256, Poseidon1};
    use math_cuda::whir_hash_bench::{grind, tree};
    const REPS: usize = 5;
    let shapes: [(&str, bool, bool, usize, usize); 4] = [
        ("base 2^26 fold 6 coset-order", true, true, 26, 6),
        ("base 2^28 fold 6 coset-order", true, true, 28, 6),
        ("base 2^28 fold 6 natural", true, false, 28, 6),
        ("ext3 2^22 fold 4", false, false, 22, 4),
    ];
    for (label, base, reversed, log_elements, log_folding) in shapes {
        for (name, hash) in [("P1", Poseidon1), ("B3", Blake3), ("keccak", Keccak256)] {
            if reversed && hash == Keccak256 {
                continue;
            }
            let t = tree(hash, base, reversed, log_elements, log_folding, REPS)
                .unwrap_or_else(|e| panic!("{name} {label}: {e:?}"));
            let limbs = if base { 1.0 } else { 3.0 };
            let bytes = (1u64 << log_elements) as f64 * 8.0 * limbs;
            println!(
                "WHIR HASH BENCH: {label} · {name} · leaves {:.3} ms ({:.1} M leaves/s, {:.0} GB/s read) \
                 · inner {:.3} ms ({:.1} M parents/s) · tree {:.3} ms · {} leaves",
                t.leaves_secs * 1e3,
                t.num_leaves as f64 / t.leaves_secs / 1e6,
                bytes / t.leaves_secs / 1e9,
                t.inner_secs * 1e3,
                t.num_leaves as f64 / t.inner_secs / 1e6,
                (t.leaves_secs + t.inner_secs) * 1e3,
                t.num_leaves,
            );
        }
    }
    for (name, hash) in [("P1", Poseidon1), ("B3", Blake3), ("keccak", Keccak256)] {
        let secs = grind(hash, 18, 16).unwrap_or_else(|| panic!("{name} grind declined"));
        println!(
            "WHIR HASH BENCH: grind 18 bits · {name} · {:.3} ms a search (median of 16 seeds)",
            secs * 1e3
        );
    }
}
