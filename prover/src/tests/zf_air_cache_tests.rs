//! The AIR prototype cache (`test_utils::build_air`) must key the proof
//! FORMAT: an AIR built for one format and asked for under another is a
//! different verifier.
//!
//! Before this test the key was `(name, blowup, queries, coset, grinding,
//! final degree)` — every `ProofOptions` field except `format`, which the ZF
//! campaign added later. The first AIR built in a process then fixed the
//! format of every later AIR with the same name and parameters, whatever
//! format the caller asked for. Two gate reds on candidate-b were this and
//! nothing else:
//!
//! - `merkle_cap_vm` (`LAMBDA_VM_ZF_CAP=auto`): the capped prove cached capped
//!   AIRs, so `verify_with_options(.., &default, ..)` verified the capped proof
//!   with those capped AIRs and accepted it.
//! - `zf_vm_dp_tests`: in a fresh process the dp prove cached dp AIRs and the
//!   "default" verifier accepted the dp proof; in the lib suite an earlier test
//!   had cached default AIRs, so the dp prove proved at `pair` and the
//!   non-vacuity assertion fired.
//!
//! The options used here carry a query count no other test uses, so these
//! keys are this test's alone however the suite interleaves.

use stark::proof::options::{CapPolicy, FriMode, ProofFormat, ProofOptions};
use stark::traits::AIR;

use crate::test_utils::{create_cpu_air, create_halt_air};

/// A query count no other test builds AIRs with.
const PRIVATE_QUERIES: usize = 47;

fn base() -> ProofOptions {
    ProofOptions {
        fri_number_of_queries: PRIVATE_QUERIES,
        ..ProofOptions::default_test_options()
    }
}

fn formats() -> Vec<ProofFormat> {
    vec![
        ProofFormat {
            merkle_cap: CapPolicy::Auto,
            ..ProofFormat::DEFAULT
        },
        ProofFormat {
            fri_mode: FriMode::Dp,
            ..ProofFormat::DEFAULT
        },
        ProofFormat {
            merkle_cap: CapPolicy::Fixed(2),
            fri_mode: FriMode::Dp,
            ..ProofFormat::DEFAULT
        },
        ProofFormat::DEFAULT,
    ]
}

/// Every format asked for is the format handed back, in both build orders
/// (non-default first, then default; and the reverse through a second AIR),
/// and asking again (a cache hit) changes nothing.
#[test]
fn the_air_prototype_cache_keys_the_proof_format() {
    let options = |format: ProofFormat| ProofOptions { format, ..base() };
    // HALT: non-default formats first, the default last.
    for _round in 0..2 {
        for format in formats() {
            let air = create_halt_air(&options(format));
            assert_eq!(
                air.options().format,
                format,
                "HALT built for {format:?} carries another format"
            );
        }
    }
    // CPU (a constraint-bearing AIR): the default first, then the rest.
    for _round in 0..2 {
        for format in formats().into_iter().rev() {
            let air = create_cpu_air(&options(format));
            assert_eq!(
                air.options().format,
                format,
                "CPU built for {format:?} carries another format"
            );
        }
    }
}
