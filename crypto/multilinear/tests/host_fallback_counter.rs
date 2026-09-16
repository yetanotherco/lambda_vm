//! ★ Every commit is accounted for on exactly one side: it ran on the device,
//! or it fell back to the host. Neither is silent.
//!
//! No GPU needed, and that is the point — without a card every commit takes the
//! fallback arm, so the arm this file is about is the one that always runs here.
//!
//! ```text
//! cargo test -p multilinear --test host_fallback_counter
//! ```
//!
//! # Why this exists
//!
//! A device commit that declines does not report anything. `commit_calls()`
//! simply does not rise, and a count merely lower than expected says nothing
//! when the expected count is itself derived from the table census. That
//! silence hid a real regression: keeping a Merkle tree per commitment filled
//! the card, every commit after the ceiling encoded on the host instead, and
//! the only visible symptoms were host memory and a GPU-utilisation figure
//! that read like a scheduling problem. The arm was taken thousands of times
//! and no line of output named it.
//!
//! Falling back is not a slower route to the same place. `from_codeword`
//! retains the host codeword AND its node array for the rest of the proof, so
//! a card that fills partway through an epoch converts into gigabytes of host
//! memory that never comes back.
//!
//! # Why this assertion and not `host_fallbacks() == 3`
//!
//! Because that one would be true on this laptop and false on the box, and a
//! test whose truth depends on which machine ran it is not pinning anything.
//! The SUM is the invariant on both: three commits, three accounted for,
//! wherever they ran. It fails if the fallback arm forgets to count — the sum
//! reads 0 on a GPU-less host and short on a card, which is exactly the bug.
//!
//! Its own integration binary so the process-wide counters belong to it alone.
//! Read `crypto/math-cuda/tests/whir_tree_cache.rs` for what a global counter
//! costs when several tests share a binary.

use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField as F;
use multilinear::mle::Mle;
use multilinear::whir_chain::{ChainConfig, GrindBits, commit};
use multilinear::whir_hash::KeccakWhir;

fn poly(num_vars: usize, seed: u64) -> Mle<F> {
    Mle::new(
        (0..(1u64 << num_vars))
            .map(|i| FieldElement::from(i.wrapping_mul(6364136223846793005).wrapping_add(seed)))
            .collect(),
    )
    .expect("power of two")
}

#[test]
fn every_commit_is_counted_on_exactly_one_side() {
    let config = ChainConfig {
        log_blowup: 2,
        log_folding: 2,
        num_queries: 3,
        grind: GrindBits::default(),
    };

    multilinear::gpu::reset_call_counters();
    assert_eq!(
        multilinear::gpu::commit_calls() + multilinear::gpu::host_fallbacks(),
        0,
        "the reset must clear both counters, or the count below is someone else's"
    );

    const COMMITS: u64 = 3;
    for seed in 0..COMMITS {
        let f = poly(8, seed * 7 + 1);
        let _ = commit::<F, KeccakWhir>(&f, &config, false).expect("commit");
    }

    let on_device = multilinear::gpu::commit_calls();
    let on_host = multilinear::gpu::host_fallbacks();
    assert_eq!(
        on_device + on_host,
        COMMITS,
        "{COMMITS} commits were made; {on_device} on the device and {on_host} \
         on the host — an arm is silent"
    );
}
