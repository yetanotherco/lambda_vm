//! The codeword fold and the ext3 fold-block commit against the host.
//!
//! Runs on the merge-queue GPU box via `make test-math-cuda` — both need a real
//! device, like the other tests here.
//!
//! The references are the host arms themselves —
//! `whir::fold_codeword_k_on_host` and
//! `CodewordCommitment::from_codeword_on_host` — so the kernels cannot drift
//! from the protocol: what is compared is every folded value and the root over
//! the fold blocks. Naming them beats unsetting the kill switches, which are
//! read once per process and would leave the reference on the device.

use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;
use math::field::goldilocks::GoldilocksField as Gl;
use multilinear::mle::Mle;
use multilinear::whir::{self, Domain};
use multilinear::whir_commit::CodewordCommitment;
use multilinear::whir_hash::KeccakWhir;

type FE3 = FieldElement<Ext3>;
type FE = FieldElement<Gl>;

/// A codeword with no structure a kernel could accidentally satisfy: the
/// encoding of a pseudo-random multilinear, so it is a real codeword.
fn codeword(num_vars: usize, log_blowup: usize) -> (Vec<FE>, Domain<Gl>) {
    let evals: Vec<FE> = (0..(1u64 << num_vars))
        .map(|i| FE::from(i.wrapping_mul(6364136223846793005).wrapping_add(11) >> 9))
        .collect();
    let f = Mle::new(evals).expect("power of two");
    let domain = Domain::<Gl>::new(num_vars + log_blowup).expect("domain");
    let cw = whir::encode::<Gl, Gl>(&whir::lift_coefficients(&f), &domain).expect("encode");
    (cw, domain)
}

fn challenge(seed: u64) -> FE3 {
    FE3::new([
        FE::from(seed * 31 + 7),
        FE::from(seed * 17 + 5),
        FE::from(seed + 3),
    ])
}

/// `levels` folds against the host, then the same again on the ext3 codeword
/// the first group produced — the base entry point and the ext one.
#[test]
fn device_folds_match_the_host_fold() {
    let (cw, domain) = codeword(14, 2);
    let alphas: Vec<FE3> = (1..=4).map(challenge).collect();

    let (device, device_domain) =
        whir::fold_codeword_k::<Gl, Gl, Ext3>(&cw, &domain, &alphas).expect("device fold");
    let (host, host_domain) =
        whir::fold_codeword_k_on_host::<Gl, Gl, Ext3>(&cw, &domain, &alphas).expect("host fold");
    assert_eq!(device.len(), host.len());
    assert_eq!(device, host, "the base fold differs");
    assert_eq!(device_domain.log_size(), host_domain.log_size());

    let alphas: Vec<FE3> = (5..=7).map(challenge).collect();
    let (device_again, _) =
        whir::fold_codeword_k::<Gl, Ext3, Ext3>(&device, &device_domain, &alphas)
            .expect("device fold");
    let (host_again, _) =
        whir::fold_codeword_k_on_host::<Gl, Ext3, Ext3>(&host, &host_domain, &alphas)
            .expect("host fold");
    assert_eq!(device_again, host_again, "the extension fold differs");
}

/// The tree over an ext3 codeword's fold blocks must be the one the host
/// builds: same root, and openings that verify against it.
#[test]
fn the_device_ext3_commit_matches_the_host() {
    let (cw, domain) = codeword(14, 2);
    let alphas: Vec<FE3> = (1..=4).map(challenge).collect();
    let (folded, _) =
        whir::fold_codeword_k::<Gl, Gl, Ext3>(&cw, &domain, &alphas).expect("device fold");

    let device = CodewordCommitment::<_, KeccakWhir>::from_codeword(folded.clone(), 4)
        .expect("device commit");
    let host =
        CodewordCommitment::<_, KeccakWhir>::from_codeword_on_host(folded, 4).expect("host commit");
    assert_eq!(device.root(), host.root(), "roots differ");
    for index in [0, 1, device.num_leaves() / 3, device.num_leaves() - 1] {
        let opening = device.open(index).expect("open");
        assert!(
            multilinear::whir_commit::verify_opening::<_, KeccakWhir>(
                &device.root(),
                index,
                &opening
            ),
            "device opening at {index} does not verify"
        );
    }
}

/// ★ Six levels in one residency (W2 `first6`: the first round folds 6
/// variables of the base codeword), against the host arm.
///
/// ⚠ Called on the device ENTRY POINT, not through `whir::fold_codeword_k`:
/// that wrapper returns `None` below its size threshold or under
/// `LAMBDA_VM_NO_GPU_WHIR_FOLD` and falls back to the host silently, so a
/// comparison through it can be the host against itself. This one either runs
/// the kernels or fails.
#[test]
fn the_device_folds_six_levels_as_the_host_does() {
    for (num_vars, log_blowup) in [(14, 2), (12, 2), (7, 1)] {
        let (cw, domain) = codeword(num_vars, log_blowup);
        let alphas: Vec<FE3> = (1..=6).map(challenge).collect();
        let (host, host_domain) =
            whir::fold_codeword_k_on_host::<Gl, Gl, Ext3>(&cw, &domain, &alphas)
                .expect("host fold");

        // The arguments `multilinear::gpu::fold_codeword_k` builds.
        let two_inv = *FE::from(2u64).inv().expect("2 is invertible").value();
        let mut g_inv = domain.generator().inv().expect("a generator is invertible");
        let mut g_invs = Vec::with_capacity(alphas.len());
        for _ in 0..alphas.len() {
            g_invs.push(*g_inv.value());
            g_inv = g_inv.square();
        }
        let raw_alphas: Vec<u64> = alphas
            .iter()
            .flat_map(|a| a.value().iter().map(|c| *c.value()))
            .collect();
        let raw: Vec<u64> = cw.iter().map(|v| *v.value()).collect();
        let device = math_cuda::whir::fold_codeword_base(&raw, two_inv, &g_invs, &raw_alphas)
            .unwrap_or_else(|e| panic!("device fold at 2^{num_vars} (needs a GPU): {e:?}"));
        let device: Vec<FE3> = device
            .chunks_exact(3)
            .map(|c| FE3::new([FE::from_raw(c[0]), FE::from_raw(c[1]), FE::from_raw(c[2])]))
            .collect();

        assert_eq!(host.len(), cw.len() >> 6);
        assert_eq!(device.len(), host.len());
        assert_eq!(
            device, host,
            "the six-level base fold differs at 2^{num_vars}"
        );
        assert_eq!(host_domain.log_size(), num_vars + log_blowup - 6);
    }
}
