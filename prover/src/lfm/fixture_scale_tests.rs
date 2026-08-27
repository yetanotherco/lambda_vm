//! The SCALED fixture: the same FRI commitment-opening verifier at sizes where
//! `LFM_HASH` is a large share of the program, so that a hash swap is something
//! a prove can actually be MEASURED to feel.
//!
//! ## Why this exists
//!
//! Every algebraic-hash figure this campaign has is a PROXY — measured chip
//! cells and measured host permutation speed, extrapolated cells-linearly off
//! the one measured full-scale BLAKE3 prove. The fixture is the cheapest place
//! to test that extrapolation, because it already commits end to end under any
//! hasher. At its blessed size it cannot: `LFM_HASH` is 0.3-0.5% of ~15.9M
//! cells (`rpx_chip_tests::the_fixture_hash_share_is_too_small_to_measure_a_swap`),
//! swamped by fixed-height lookup tables that do not move with the workload.
//!
//! [`FixtureShape`] is what fixes that. The construction, the emitter and the
//! host prover are unchanged; only their bounds are read off a shape now, so
//! Merkle depth and query count can be dialled until the hash chip dominates.
//!
//! ## ⚠ What a number from here does and does not say
//!
//! This regime is NOT the aggregator's. Even scaled, the fixture carries a
//! fixed floor the aggregator does not have, and its hash share tops out well
//! below the aggregator's ~85%. What it can establish is one specific thing:
//! whether prove-time PEAK RSS moves LINEARLY with committed cells when the
//! only thing changed is which permutation fills the hash chip. That is the
//! assumption every algebraic projection in this campaign rests on, and it has
//! never been tested at all.
//!
//! It is also worth saying what this would NOT have caught. The campaign's two
//! model failures were a memory census that omitted terms (three OOM kills) and
//! an invocation count that was 42% low. Neither is a linearity question; a
//! linear model with a wrong coefficient is still linear. This tests the SHAPE
//! of the model, not its inputs.
//!
//! ## ★ What it measured (2026-08-27, M-series laptop, 11 rayon threads)
//!
//! 140 samples, ABBA-ordered, one arm per process; 23 distinct (hasher, shape)
//! points at blowup 2 spanning **17.4M to 254.8M committed cells** and hash
//! shares from 7.5% to 92.1%. Peak RSS reproduced to 0.2-2.9% per point; wall
//! clock to 1.3-11.6%, so RSS is by far the steadier instrument here.
//!
//! **Peak RSS is AFFINE in committed cells, and the line barely moves with the
//! hasher.** Per-hasher fits at blowup 2:
//!
//! ```text
//!   BLAKE3    1.308 GiB + 44.27 bytes/cell   R² 0.99984   (7 points)
//!   RPO       1.323 GiB + 42.96 bytes/cell   R² 0.99898   (6)
//!   RPX       1.241 GiB + 47.02 bytes/cell   R² 0.99269   (5)
//!   Poseidon  1.358 GiB + 40.51 bytes/cell   R² 0.99850   (5)
//! ```
//!
//! Four permutations whose `LFM_HASH` tables differ by 11x in width and 8x in
//! height agree on the slope to ±7.5% and on the intercept to ±4.5%. A fit
//! trained only on points up to 82M cells predicts the 254.8M-cell point — 3.1x
//! beyond its training range — to **−0.19%**. Committed cells is the sufficient
//! statistic; the hash table's aspect ratio is not.
//!
//! ⚠ **AFFINE, not PROPORTIONAL, and the difference is the whole point.** The
//! intercept is real: 1.24-1.36 GiB, hasher-independent, against a process
//! baseline of 0.02 GiB. So at this scale a through-the-origin cells model
//! over-credits a hash swap badly. Swapping BLAKE3 for RPO on the SAME program
//! cut cells 3.13x but peak RSS only 1.94x at 256 queries, and 5.20x versus
//! 3.57x at 1024 — the realised saving approaches the promised one only as the
//! fixed term stops mattering.
//!
//! ⚠ **The COEFFICIENT does not transfer between instruments, only the form.**
//! 43.9 bytes/cell applied to the aggregator's 12.2B cells predicts ~500 GiB,
//! where the aggregator measured 336.8. Anyone carrying a bytes/cell figure from
//! one program to another is wrong by about 1.5x; scaling a program's own
//! measured point by its own cell ratio is what the linearity result licenses.
//!
//! Blowup is a second axis and not a free one: at IDENTICAL committed cells,
//! moving blowup 2 → 4 cost +73% peak RSS under RPO and +62% under BLAKE3
//! (slope 43-47 → 59-71 bytes/cell, intercept 1.24-1.36 → 2.27-2.70 GiB).
//! Committed cells do not move with the blowup; the LDE holding them does.
//!
//! Wall clock is cells-affine too but a worse instrument: 39.7 ns/cell for
//! BLAKE3 against 47.9 for Poseidon, a 21% hasher spread where RSS showed 7.5%.

use std::time::Instant;

use super::airs::lfm_chip_census_with_hasher;
use super::compiler::LfmProgram;
use super::fixture::{
    FixtureShape, bump_lane0, fixture_prove_with_hasher, fixture_prove_with_shape, shape,
};
use super::hash::HasherKind;
use super::programs::{fri_toy_program, fri_toy_program_with_shape};
use super::proof::{LfmProveError, lfm_prove_with_hasher, verify_against_artifacts};
use super::registry::build_artifacts_with_hasher;
use crate::lfm::executor::LfmExecError;

use stark::proof::options::{GoldilocksCubicProofOptions, ProofOptions};

fn options() -> ProofOptions {
    GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 is valid")
}

/// The size the always-on correctness tests run at: big enough that every
/// generalised bound is a DIFFERENT number from the blessed shape's (paths 8
/// and 7 rather than 4 and 3, seven queries rather than four, a 26-word query
/// stride rather than 17), small enough to prove in seconds.
///
/// A shape that only changed the query count would leave every path-length
/// derivation untested, which is where the off-by-ones live.
const GREEN: FixtureShape = FixtureShape::new(9, 7);

// =========================================================================
// The derivation — checked against the constants it generalises
// =========================================================================

/// The default shape must reproduce the pinned constants EXACTLY. These are the
/// numbers the blessed `FriToyV0` program and its committed roots were built
/// from, so a derivation that drifted from them would silently re-shape a
/// blessed program rather than fail.
#[test]
fn the_default_shape_reproduces_the_pinned_constants() {
    let d = FixtureShape::default();
    assert_eq!(d.log_lde, shape::LOG_LDE);
    assert_eq!(d.num_queries, shape::NUM_QUERIES);
    assert_eq!(d.lde_size(), shape::LDE_SIZE);
    assert_eq!(d.query_bits(), shape::QUERY_BITS);
    assert_eq!(d.words_per_query(), shape::WORDS_PER_QUERY);
    // The two path lengths the old emitter spelled as literal 4 and 3.
    assert_eq!(d.main_path_len(), 4);
    assert_eq!(d.l1_path_len(), 3);
}

/// ★ The regression gate for the whole generalisation: emitting at the DEFAULT
/// shape must produce the blessed program bit-for-bit.
///
/// `program_id` is a digest over the compiled program, and the preprocessed
/// roots are commitments to its columns, so this is not "the same shape" — it
/// is the same instruction stream in the same order. Any reordering the
/// rewrite introduced, however semantically harmless, moves one of these.
#[test]
fn the_shaped_emitter_reproduces_the_blessed_program_at_the_default_size() {
    let opts = options();
    let blessed = fri_toy_program();
    let shaped = fri_toy_program_with_shape(FixtureShape::default());
    for kind in [HasherKind::Test, HasherKind::Rpo] {
        let a = build_artifacts_with_hasher(&blessed, &opts, kind);
        let b = build_artifacts_with_hasher(&shaped, &opts, kind);
        assert_eq!(
            a.program_id, b.program_id,
            "{kind:?}: program identity moved"
        );
        assert_eq!(a.roots, b.roots, "{kind:?}: a preprocessed root moved");
        assert_eq!(a.log_heights, b.log_heights);
    }
    // And the host prover at the default shape is the blessed prover.
    let x = fixture_prove_with_hasher(HasherKind::Rpo);
    let y = fixture_prove_with_shape(HasherKind::Rpo, FixtureShape::default());
    assert_eq!(x.commitments, y.commitments, "the fixture's roots moved");
    assert_eq!(x.openings, y.openings, "the fixture's openings moved");
}

/// The predicted `LFM_HASH` invocation count is the sizing instrument — every
/// choice of scale below is made from it rather than by trial proving — so it
/// is checked against the census the prover actually builds.
///
/// One invocation is one `LFM_HASH` row under EVERY hasher — the chip is a full
/// permutation per row and only its WIDTH moves — so `real_rows` is the count
/// directly. That invariance is the reason a hash swap is a pure width change
/// at fixed height, which is what makes the census's cells the only thing that
/// moves and so the only thing a measurement has to explain.
#[test]
fn the_predicted_hash_invocation_count_matches_the_census() {
    for sh in [FixtureShape::default(), GREEN, FixtureShape::new(12, 5)] {
        let program = fri_toy_program_with_shape(sh);
        for kind in [
            HasherKind::Rpo,
            HasherKind::Rpx,
            HasherKind::Poseidon,
            HasherKind::Blake3,
        ] {
            let census = lfm_chip_census_with_hasher(&program, kind);
            let h = census
                .iter()
                .find(|c| c.name == "LFM_HASH")
                .expect("the hash chip");
            assert_eq!(
                h.real_rows as usize,
                sh.hash_invocations(),
                "{sh:?} under {kind:?}: predicted {} invocations, census has {}",
                sh.hash_invocations(),
                h.real_rows,
            );
        }
    }
}

/// ★ The sizing constants, pinned: `LFM_HASH` cells per ROW, per hasher.
///
/// Every projected size in this lane is `rows · this`, so a silent width change
/// would re-scale the whole ladder while every ratio still looked plausible.
/// The convention is the census's own — main cells plus aux cells, an extension
/// aux column counted once — which is NOT the `cliff_cost` convention that
/// counts aux three times, and the two differ by six for RPO. Quoting one as
/// the other is how 439 becomes 445.
#[test]
fn the_hash_chip_cells_per_row_are_pinned_per_hasher() {
    let program = fri_toy_program_with_shape(GREEN);
    for (kind, expect) in [
        (HasherKind::Rpo, 439),
        (HasherKind::Rpx, 319),
        (HasherKind::Poseidon, 615),
        (HasherKind::Blake3, 3579),
    ] {
        let census = lfm_chip_census_with_hasher(&program, kind);
        let h = census
            .iter()
            .find(|c| c.name == "LFM_HASH")
            .expect("the hash chip");
        let per_row = (h.main_cells() + h.aux_cells()) / h.rows;
        assert_eq!(per_row, expect, "{kind:?}: LFM_HASH cells per row moved");
    }
}

/// The non-hash floor, pinned as the two constants it is: a fixed part that no
/// workload moves, and a per-query part.
///
/// This is what makes the fixture a DIFFERENT regime from the aggregator, and
/// the number that says how far it has to be scaled before a hash swap is
/// visible at all. It is asserted rather than only measured because the whole
/// sizing argument rests on the floor being flat.
#[test]
fn the_non_hash_floor_is_flat_in_the_query_count() {
    let per_query = 4_272u64;
    let fixed = 15_859_860u64;
    for q in [32usize, 64, 256, 1024] {
        let sh = FixtureShape::new(16, q);
        let program = fri_toy_program_with_shape(sh);
        let (total, hash, _) = census_totals(&program, HasherKind::Rpo);
        let predicted = fixed + per_query * q as u64;
        let non_hash = total - hash;
        assert!(
            non_hash.abs_diff(predicted) < 1_024,
            "q={q}: non-hash cells {non_hash}, predicted {predicted}"
        );
    }
}

// =========================================================================
// The scaled fixture, end to end — the same gates the blessed one carries
// =========================================================================

/// ★ A real prove-and-verify of a SCALED fixture, under every hasher the
/// campaign is choosing between.
///
/// This is the test that says the generalisation is a fixture and not a shape
/// calculator: the host proof at [`GREEN`] is authenticated by a machine
/// program emitted at [`GREEN`], every Merkle path is walked to a root the
/// transcript committed, and the machine proof itself verifies.
#[test]
fn the_machine_verifies_a_scaled_fixture_end_to_end() {
    let opts = options();
    let program = fri_toy_program_with_shape(GREEN);
    for kind in [
        HasherKind::Test,
        HasherKind::Rpo,
        HasherKind::Rpx,
        HasherKind::Poseidon,
    ] {
        let artifacts = build_artifacts_with_hasher(&program, &opts, kind);
        let inner = fixture_prove_with_shape(kind, GREEN);
        let proved = lfm_prove_with_hasher(
            &program,
            &artifacts,
            &[inner.commitments.clone(), inner.openings.clone()],
            &opts,
            kind,
        )
        .unwrap_or_else(|e| panic!("{kind:?}: the scaled fixture must prove, got {e:?}"));

        assert_eq!(proved.public_words[0].1, inner.commitments[0]);
        assert_eq!(proved.public_words[1].1, inner.commitments[1]);
        assert!(
            verify_against_artifacts(&artifacts, &proved.proof, &proved.public_words, &opts),
            "{kind:?}: the machine proof of a scaled fixture must verify"
        );
    }
}

/// Every tamper vector must make the scaled program UNPROVABLE. Without this
/// the scaled fixture would be a cost model wearing a verifier's clothes —
/// a program that hashes a great deal and checks nothing would measure exactly
/// the same and be worth nothing.
///
/// The vectors reach the parts the generalisation actually rewrote: a row under
/// the DEEPER main path, a sibling in the middle of that path, and an L1 leaf
/// in the LAST query — the one at the largest arena offset, which is where a
/// wrong `words_per_query` stride would show up.
#[test]
fn the_machine_rejects_tampered_scaled_proofs() {
    let opts = options();
    let program = fri_toy_program_with_shape(GREEN);
    let kind = HasherKind::Rpo;
    let artifacts = build_artifacts_with_hasher(&program, &opts, kind);
    let honest = fixture_prove_with_shape(kind, GREEN);
    let arenas = |p: &super::fixture::FriToyProof| vec![p.commitments.clone(), p.openings.clone()];

    let expect_reject = |a: Vec<Vec<super::LfmWord>>, what: &str| match lfm_prove_with_hasher(
        &program, &artifacts, &a, &opts, kind,
    ) {
        Err(LfmProveError::Exec(LfmExecError::DivByZero { .. })) => {}
        other => panic!(
            "{what}: expected a failed in-machine assert, got {:?}",
            other.map(|_| "accepted")
        ),
    };

    let stride = GREEN.words_per_query();
    let last = (GREEN.num_queries - 1) * stride;

    let mut t = arenas(&honest);
    t[1][0] = bump_lane0(&t[1][0]);
    expect_reject(t, "tampered opened row");

    // A sibling halfway up the deeper main path.
    let mut t = arenas(&honest);
    t[1][2 + GREEN.main_path_len() / 2] = bump_lane0(&t[1][2 + GREEN.main_path_len() / 2]);
    expect_reject(t, "tampered mid-path main sibling");

    // The LAST query's L1 leaf — the largest arena offset in the program.
    let mut t = arenas(&honest);
    let at = last + 4 + 2 * GREEN.main_path_len();
    t[1][at] = bump_lane0(&t[1][at]);
    expect_reject(t, "tampered L1 leaf of the last query");

    // A commitment, which breaks the transcript replay rather than a path.
    let mut t = arenas(&honest);
    t[0][0] = bump_lane0(&t[0][0]);
    expect_reject(t, "tampered commitment");
}

/// A scaled proof committed under one hash must not be provable under another —
/// the same property the blessed fixture carries, re-checked at a size where
/// every Merkle path is longer, since the paths are where it is enforced.
#[test]
fn a_scaled_proof_committed_under_one_hash_is_not_provable_under_another() {
    let opts = options();
    let program = fri_toy_program_with_shape(GREEN);
    let inner = fixture_prove_with_shape(HasherKind::Rpo, GREEN);
    let arenas = vec![inner.commitments.clone(), inner.openings.clone()];

    for other in [HasherKind::Rpx, HasherKind::Poseidon, HasherKind::Test] {
        let artifacts = build_artifacts_with_hasher(&program, &opts, other);
        match lfm_prove_with_hasher(&program, &artifacts, &arenas, &opts, other) {
            Err(LfmProveError::Exec(LfmExecError::DivByZero { .. })) => {}
            got => panic!(
                "an RPO-committed scaled proof must not prove under {other:?}, got {:?}",
                got.map(|_| "accepted")
            ),
        }
    }
}

// =========================================================================
// The census — sizing, and the hash share it buys
// =========================================================================

fn census_totals(program: &LfmProgram, kind: HasherKind) -> (u64, u64, u64) {
    let census = lfm_chip_census_with_hasher(program, kind);
    let total: u64 = census.iter().map(|c| c.main_cells() + c.aux_cells()).sum();
    let h = census
        .iter()
        .find(|c| c.name == "LFM_HASH")
        .expect("the hash chip");
    (total, h.main_cells() + h.aux_cells(), h.rows)
}

/// The sizing panel: hash share against shape, for every hasher.
///
/// ```text
/// cargo test --release -p lambda-vm-prover --lib \
///   lfm::fixture_scale_tests::the_scaled_fixture_census -- --ignored --exact --nocapture
/// ```
#[test]
#[ignore]
fn the_scaled_fixture_census() {
    let mut shapes = vec![FixtureShape::default(), GREEN];
    let mut q = 32;
    while q <= 4096 {
        shapes.push(FixtureShape::new(16, q));
        q *= 2;
    }
    println!(
        "{:>18}  {:>8}  {:>7}  {:>13}  {:>13}  {:>6}",
        "shape", "hashinv", "hasher", "total cells", "LFM_HASH", "share"
    );
    for sh in shapes {
        let program = fri_toy_program_with_shape(sh);
        for kind in [
            HasherKind::Rpo,
            HasherKind::Rpx,
            HasherKind::Poseidon,
            HasherKind::Blake3,
        ] {
            let (total, hash, rows) = census_totals(&program, kind);
            println!(
                "  log_lde {:>2} q {:>5}  {:>8}  {:>7}  {:>13}  {:>13}  {:>5.1}%  (rows {})",
                sh.log_lde,
                sh.num_queries,
                sh.hash_invocations(),
                format!("{kind:?}"),
                total,
                hash,
                100.0 * hash as f64 / total as f64,
                rows,
            );
        }
    }
}

// =========================================================================
// ★ THE MEASUREMENT — one arm per process
// =========================================================================

/// Peak RSS is a process HIGH-WATER MARK, so two arms in one process are one
/// measurement of the larger. This test therefore measures exactly ONE arm,
/// named by the environment, and the ABBA ordering is done by the caller
/// running it repeatedly.
///
/// ```text
/// LFM_FIXTURE_HASHER=rpo LFM_FIXTURE_LOG_LDE=16 LFM_FIXTURE_QUERIES=1024 \
/// cargo test --release -p lambda-vm-prover --lib \
///   lfm::fixture_scale_tests::the_scaled_fixture_measures_one_arm \
///   -- --ignored --exact --nocapture
/// ```
///
/// Reported at every stage boundary, because the high-water mark is monotone:
/// the INCREMENTS say which stage set the peak, and a run whose peak was set by
/// artifact building rather than by proving is not measuring the hash at all.
#[test]
#[ignore]
fn the_scaled_fixture_measures_one_arm() {
    let kind = match std::env::var("LFM_FIXTURE_HASHER")
        .unwrap_or_else(|_| "rpo".into())
        .as_str()
    {
        "rpo" => HasherKind::Rpo,
        "rpx" => HasherKind::Rpx,
        "poseidon" => HasherKind::Poseidon,
        "blake3" => HasherKind::Blake3,
        "test" => HasherKind::Test,
        other => panic!("unknown LFM_FIXTURE_HASHER {other}"),
    };
    let env_usize = |k: &str, d: usize| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let sh = FixtureShape::new(
        env_usize("LFM_FIXTURE_LOG_LDE", 16),
        env_usize("LFM_FIXTURE_QUERIES", 1024),
    );
    // The blowup is an axis in its own right: committed CELLS do not move with
    // it, but the LDE that holds them does, so a model that is cells-linear at
    // one blowup is not automatically the same line at another. Projecting the
    // laptop onto the box's posture needs to know which.
    let blowup = env_usize("LFM_FIXTURE_BLOWUP", 2) as u8;
    let opts = GoldilocksCubicProofOptions::with_blowup(blowup).expect("a valid blowup");
    let rss = || super::wrap_tests::peak_rss_gib().unwrap_or(f64::NAN);

    println!(
        "\n=== ARM {kind:?}  log_lde {} queries {} — {} hash invocations, \
         {} openings words, blowup {}, threads {} ===",
        sh.log_lde,
        sh.num_queries,
        sh.hash_invocations(),
        sh.num_queries * sh.words_per_query(),
        opts.blowup_factor,
        rayon_threads(),
    );

    let t = Instant::now();
    let program = fri_toy_program_with_shape(sh);
    println!(
        "emit+compile   {:>8.2}s   peak {:>6.3} GiB",
        t.elapsed().as_secs_f64(),
        rss()
    );

    let (total, hash_cells, hash_rows) = census_totals(&program, kind);
    println!(
        "census         total {total} cells, LFM_HASH {hash_cells} ({:.1}%) over {hash_rows} rows",
        100.0 * hash_cells as f64 / total as f64
    );

    let t = Instant::now();
    let inner = fixture_prove_with_shape(kind, sh);
    println!(
        "host fixture   {:>8.2}s   peak {:>6.3} GiB",
        t.elapsed().as_secs_f64(),
        rss()
    );

    let t = Instant::now();
    let artifacts = build_artifacts_with_hasher(&program, &opts, kind);
    println!(
        "artifacts      {:>8.2}s   peak {:>6.3} GiB",
        t.elapsed().as_secs_f64(),
        rss()
    );

    let t = Instant::now();
    let proved = lfm_prove_with_hasher(
        &program,
        &artifacts,
        &[inner.commitments.clone(), inner.openings.clone()],
        &opts,
        kind,
    )
    .expect("the scaled fixture must prove");
    let prove_secs = t.elapsed().as_secs_f64();
    let prove_peak = rss();
    println!("MACHINE PROVE  {prove_secs:>8.2}s   peak {prove_peak:>6.3} GiB");

    let t = Instant::now();
    assert!(
        verify_against_artifacts(&artifacts, &proved.proof, &proved.public_words, &opts),
        "the scaled machine proof must verify"
    );
    println!(
        "verify         {:>8.2}s   peak {:>6.3} GiB",
        t.elapsed().as_secs_f64(),
        rss()
    );

    // One machine-readable line per arm, for the driver to collect.
    println!(
        "RESULT hasher={kind:?} log_lde={} queries={} invocations={} cells={total} \
         hash_cells={hash_cells} sub_proofs={} prove_secs={prove_secs:.3} \
         peak_gib={prove_peak:.4} blowup={blowup} threads={}",
        sh.log_lde,
        sh.num_queries,
        sh.hash_invocations(),
        proved.proof.proofs.len(),
        rayon_threads(),
    );
}

fn rayon_threads() -> usize {
    #[cfg(feature = "parallel")]
    {
        rayon::current_num_threads()
    }
    #[cfg(not(feature = "parallel"))]
    {
        1
    }
}
