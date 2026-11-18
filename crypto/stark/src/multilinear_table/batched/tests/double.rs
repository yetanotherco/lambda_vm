//! The batched argue on the two-layer GKR ladder (`GkrLadder::Double`): round
//! trips across heights of both parities, bins and cut tables; every prover
//! fault the one-layer ladder refuses, refused here too; the departures only
//! a two-layer step admits; the verifier's checks made inert, each letting its
//! negative through; and neither ladder's verifier taking the other's proof.

use super::pack::{fan_airs, fan_cols, packed, packed_tables, plan_of};
use super::*;
use multilinear::gkr_lockstep::{Relation, StepFault, ladder_steps};
use multilinear::whir_chain::GkrLadder;

/// The batched format at `bin_log_cells` on the two-layer ladder.
fn double(bin_log_cells: u8) -> ChainConfig {
    config(ArgueFormat::batched(bin_log_cells).with_ladder(GkrLadder::Double))
}

/// `cfg` with its ladder replaced.
fn on(cfg: &ChainConfig, ladder: GkrLadder) -> ChainConfig {
    let mut cfg = *cfg;
    cfg.format.argue = cfg.format.argue.with_ladder(ladder);
    cfg
}

fn round_trip_double(a: usize, pad: usize, cap: u8, shifted: bool) -> Result<(), MlError> {
    let airs = airs();
    let cfg = double(cap);
    let committed = commit(tables(&airs, &columns(a, pad), shifted), &cfg);
    let proof = prove(&committed, &cfg, ProverFaults::default());
    // The proof has the schedule's shape, bin by bin.
    let plan = plan_of(&committed, &cfg);
    for (bin, ladder) in plan.bins.iter().zip(&proof.argue.gkr) {
        let heights: Vec<usize> = bin
            .iter()
            .flat_map(|&t| std::iter::repeat_n(plan.tree_vars(t), plan.chunks[t].count))
            .collect();
        let schedule = ladder_steps(&heights, GkrLadder::Double);
        assert_eq!(ladder.layers.len(), schedule.len());
        for (layer, step) in ladder.layers.iter().zip(&schedule) {
            assert_eq!(layer.sumcheck.rounds.len(), step.parent);
            assert!(
                layer
                    .sumcheck
                    .rounds
                    .iter()
                    .all(|r| r.evaluations.len() == step.degree)
            );
            assert_eq!(layer.halves.len(), step.halves());
        }
    }
    verify(&committed, &proof, &cfg)
}

// ─── round trips ─────────────────────────────────────────────────────────

/// ★ Three tables of heights of both parities (CPU `k = a + 2`, ADD
/// `k = a + pad`, MUL `k = a`), in one bin, in bins of one or two, and every
/// table alone; a shifted table too.
#[test]
fn tables_of_different_heights_round_trip_on_the_two_layer_ladder() {
    for (a, pad) in [(2, 0), (3, 1), (3, 2), (1, 3)] {
        for cap in [27u8, 5, 4, 3, 2, 0] {
            round_trip_double(a, pad, cap, false)
                .unwrap_or_else(|e| panic!("a={a} pad={pad} cap={cap}: {e:?}"));
        }
    }
    round_trip_double(2, 2, 3, true).expect("a shifted table across bins");
    round_trip_double(3, 1, 27, true).expect("a shifted table in one bin");
}

/// One-row tables (no ladder step of their own), a two-row CPU, each alone.
#[test]
fn edge_shapes_round_trip_on_the_two_layer_ladder() {
    round_trip_double(0, 0, 27, false).expect("two one-row tables and a two-row one");
    round_trip_double(0, 0, 0, false).expect("each alone");
    round_trip_double(0, 3, 27, false).expect("a one-row MUL beside taller tables");
}

/// ★ Cut tables on the two-layer ladder: a table's chunks share its height,
/// so they leave the ladder together at one point, as the chunk claims'
/// combination needs — in one bin, in small bins, and every table alone.
#[test]
fn cut_tables_round_trip_on_the_two_layer_ladder() {
    let airs = airs();
    let fans = fan_airs();
    let fan_cols = fan_cols();
    for cap in [27u8, 6, 4, 0] {
        let cfg = on(&packed(cap, 2), GkrLadder::Double);
        let committed = commit(packed_tables(&airs, &fans, &fan_cols), &cfg);
        let proof = prove(&committed, &cfg, ProverFaults::default());
        verify(&committed, &proof, &cfg).unwrap_or_else(|e| panic!("cap {cap}: {e:?}"));
        let single = on(&cfg, GkrLadder::Single);
        assert!(
            verify(&committed, &proof, &single).is_err(),
            "cap {cap}: the one-layer verifier refuses it"
        );
    }
}

/// The fixture the tamper tests share, on the two-layer ladder: CPU `k = 5`,
/// ADD `k = 4`, MUL `k = 3` in one bin. Step 0 (`ℓ = 0`): all three two
/// layers. Step 1 (`ℓ = 2`): CPU and ADD two layers, MUL one (it reaches its
/// input), degree 5. Step 2 (`ℓ = 4`): CPU one layer, degree 3.
fn fixture_double<'a>(
    airs: &'a Airs,
) -> (CommittedTables<'a, Fp, Ext, KeccakWhir>, Proof, ChainConfig) {
    let cfg = double(27);
    let committed = commit(tables(airs, &columns(3, 1), false), &cfg);
    let proof = prove(&committed, &cfg, ProverFaults::default());
    verify(&committed, &proof, &cfg).expect("the untampered proof verifies");
    (committed, proof, cfg)
}

fn tampered_double(tamper: impl FnOnce(&mut Proof)) -> Result<(), MlError> {
    let airs = airs();
    let (committed, mut proof, cfg) = fixture_double(&airs);
    tamper(&mut proof);
    verify(&committed, &proof, &cfg)
}

fn faulted_double(faults: ProverFaults) -> Result<(), MlError> {
    let airs = airs();
    let cfg = double(27);
    let committed = commit(tables(&airs, &columns(3, 1), false), &cfg);
    let proof = prove(&committed, &cfg, faults);
    verify(&committed, &proof, &cfg)
}

/// The fixture's ladder is the schedule's: three steps, the mixed one in the
/// middle.
#[test]
fn the_fixtures_two_layer_ladder_has_the_schedules_shape() {
    let airs = airs();
    let (_, proof, _) = fixture_double(&airs);
    let steps = ladder_steps(&[5, 4, 3], GkrLadder::Double);
    use Relation::{Double as D, Single as S};
    assert_eq!(
        steps
            .iter()
            .map(|s| (s.parent, s.degree, s.relations.clone()))
            .collect::<Vec<_>>(),
        vec![
            (0, 5, vec![(0, D), (1, D), (2, D)]),
            (2, 5, vec![(0, D), (1, D), (2, S)]),
            (4, 3, vec![(0, S)]),
        ]
    );
    let ladder = &proof.argue.gkr[0];
    assert_eq!(ladder.layers.len(), 3);
    assert_eq!(
        ladder
            .layers
            .iter()
            .map(|l| (l.sumcheck.rounds.len(), l.halves.len()))
            .collect::<Vec<_>>(),
        vec![(0, 6), (2, 5), (4, 1)]
    );
}

// ─── negatives: tampered proofs ──────────────────────────────────────────

#[test]
fn a_wrong_bus_output_is_refused_on_the_two_layer_ladder() {
    for t in 0..3 {
        assert!(
            tampered_double(|p| p.argue.bus_outputs[t].0 += ExtE::one()).is_err(),
            "table {t}'s numerator"
        );
        assert!(
            tampered_double(|p| p.argue.bus_outputs[t].1 += ExtE::one()).is_err(),
            "table {t}'s denominator"
        );
    }
}

/// ★ Each value of each half of the mixed step (two two-layer trees and the
/// one-layer MUL), one off: that step's residual refuses it and names the
/// step; a half missing or one too many is the wrong shape.
#[test]
fn each_value_of_the_mixed_step_one_off_is_refused() {
    for half in 0..5 {
        for value in 0..4 {
            assert_eq!(
                tampered_double(|p| {
                    let h = &mut p.argue.gkr[0].layers[1].halves[half];
                    *[&mut h.p_lo, &mut h.p_hi, &mut h.q_lo, &mut h.q_hi][value] += ExtE::one();
                }),
                Err(MlError::LayerRelationMismatch { layer: 1 }),
                "half {half}, value {value}"
            );
        }
    }
    let shape = Err(MlError::ArgueShapeMismatch {
        part: "ladder step",
    });
    assert_eq!(
        tampered_double(|p| {
            p.argue.gkr[0].layers[1].halves.pop();
        }),
        shape
    );
    assert_eq!(
        tampered_double(|p| {
            let extra = p.argue.gkr[0].layers[1].halves[4].clone();
            p.argue.gkr[0].layers[1].halves.push(extra);
        }),
        shape
    );
}

/// A degree-5 round value one off, a round of the wrong length, and a
/// constraint round value one off.
#[test]
fn a_wrong_round_is_refused_on_the_two_layer_ladder() {
    for value in 0..5 {
        assert_eq!(
            tampered_double(
                |p| p.argue.gkr[0].layers[1].sumcheck.rounds[1].evaluations[value] += ExtE::one()
            ),
            Err(MlError::LayerRelationMismatch { layer: 1 }),
            "value {value}"
        );
    }
    for len in [3usize, 6] {
        assert_eq!(
            tampered_double(|p| p.argue.gkr[0].layers[1].sumcheck.rounds[0]
                .evaluations
                .resize(len, ExtE::one())),
            Err(MlError::RoundDegreeMismatch {
                round: 0,
                expected: 5,
                got: len
            })
        );
    }
    assert_eq!(
        tampered_double(|p| p.argue.gkr[0].layers[2].sumcheck.rounds[0]
            .evaluations
            .resize(5, ExtE::one())),
        Err(MlError::RoundDegreeMismatch {
            round: 0,
            expected: 3,
            got: 5
        }),
        "a one-layer step's rounds stay degree 3"
    );
    assert_eq!(
        tampered_double(|p| p.argue.constraint.rounds[1].evaluations[2] += ExtE::one()),
        Err(MlError::BatchMismatch),
        "a constraint round"
    );
}

/// The fixture's last step (CPU alone, one layer, degree 3): each of its four
/// values one off is refused at that step; a round removed from or added to
/// the mixed step is refused by name.
#[test]
fn a_final_one_layer_step_and_a_steps_round_count_are_refused_on_the_argue() {
    for value in 0..4 {
        assert_eq!(
            tampered_double(|p| {
                let h = &mut p.argue.gkr[0].layers[2].halves[0];
                *[&mut h.p_lo, &mut h.p_hi, &mut h.q_lo, &mut h.q_hi][value] += ExtE::one();
            }),
            Err(MlError::LayerRelationMismatch { layer: 2 }),
            "value {value}"
        );
    }
    assert_eq!(
        tampered_double(|p| {
            p.argue.gkr[0].layers[1].sumcheck.rounds.pop();
        }),
        Err(MlError::RoundCountMismatch {
            expected: 2,
            got: 1
        })
    );
    assert_eq!(
        tampered_double(|p| {
            let extra = p.argue.gkr[0].layers[1].sumcheck.rounds[0].clone();
            p.argue.gkr[0].layers[1].sumcheck.rounds.push(extra);
        }),
        Err(MlError::RoundCountMismatch {
            expected: 2,
            got: 3
        })
    );
}

// ─── negatives: provers that break the format ────────────────────────────

/// ★ Every fault the one-layer ladder refuses, refused on the two-layer one:
/// padding by scaling, a suffix of `ξ`, a wrong weight or claim, another
/// bin partition.
#[test]
fn every_prover_fault_is_refused_on_the_two_layer_ladder() {
    assert_eq!(
        faulted_double(ProverFaults {
            scaled_padding: Some(2),
            ..Default::default()
        }),
        Err(MlError::BatchMismatch)
    );
    assert_eq!(
        faulted_double(ProverFaults {
            suffix_xi: Some(2),
            ..Default::default()
        }),
        Err(MlError::BatchMismatch)
    );
    for t in 0..3 {
        assert_eq!(
            faulted_double(ProverFaults {
                wrong_weight: Some(t),
                ..Default::default()
            }),
            Err(MlError::BatchMismatch),
            "weight of table {t}"
        );
        assert_eq!(
            faulted_double(ProverFaults {
                tamper_claim: Some(t),
                ..Default::default()
            }),
            Err(MlError::BatchMismatch),
            "claim of table {t}"
        );
    }
    for cap in [4u8, 0] {
        assert!(
            faulted_double(ProverFaults {
                bin_log_cells: Some(cap),
                ..Default::default()
            })
            .is_err(),
            "cap {cap}"
        );
    }
}

/// A forged balance (one output replaced by the value that balances the bus,
/// the ladder run on the true tree) is refused by step 0's residual alone,
/// one table at a time and on a cut table's first chunk.
#[test]
fn a_balancing_output_is_refused_by_the_residual_alone_on_the_two_layer_ladder() {
    let airs = airs();
    let checks = VerifierChecks {
        residual: false,
        ..VerifierChecks::ALL
    };
    // Uncut: MUL's multiplicity off unbalances the bus.
    let cfg = double(27);
    let mut cols = columns(3, 1);
    cols[2][3][1] += FE::one();
    let committed = commit(tables(&airs, &cols, false), &cfg);
    let honest = prove(&committed, &cfg, ProverFaults::default());
    assert_eq!(
        verify(&committed, &honest, &cfg),
        Err(MlError::BusImbalance),
        "premise: the trace's bus does not balance"
    );
    let forged = prove(
        &committed,
        &cfg,
        ProverFaults {
            balance_with: Some(1),
            ..ProverFaults::default()
        },
    );
    assert_eq!(
        verify(&committed, &forged, &cfg),
        Err(MlError::LayerRelationMismatch { layer: 0 })
    );
    assert_eq!(
        verify_with_checks(&committed, &forged, &cfg, checks),
        Ok(())
    );
    // Cut: FAN(7)'s odd sender reading a one.
    let fans = fan_airs();
    let mut fan_cols = fan_cols();
    fan_cols[1][1][2] = FE::one();
    let cfg = on(&packed(27, 2), GkrLadder::Double);
    let committed = commit(packed_tables(&airs, &fans, &fan_cols), &cfg);
    let forged = prove(
        &committed,
        &cfg,
        ProverFaults {
            balance_with: Some(4),
            ..ProverFaults::default()
        },
    );
    assert_eq!(
        verify(&committed, &forged, &cfg),
        Err(MlError::LayerRelationMismatch { layer: 0 })
    );
    assert_eq!(
        verify_with_checks(&committed, &forged, &cfg, checks),
        Ok(())
    );
}

/// The packing's faults on the two-layer ladder: another packing, two chunks
/// swapped, mass in a padding slot — each refused where the one-layer
/// ladder refuses it.
#[test]
fn the_packings_faults_are_refused_on_the_two_layer_ladder() {
    let airs = airs();
    let fans = fan_airs();
    let cfg = on(&packed(27, 2), GkrLadder::Double);
    let faulted = |fan_cols: &[Vec<Vec<FE>>], faults: ProverFaults| {
        let committed = commit(packed_tables(&airs, &fans, fan_cols), &cfg);
        let proof = prove(&committed, &cfg, faults);
        (
            verify(&committed, &proof, &cfg),
            verify_with_checks(
                &committed,
                &proof,
                &cfg,
                VerifierChecks {
                    constraint: false,
                    ..VerifierChecks::ALL
                },
            ),
        )
    };
    let cols = fan_cols();
    for pack in [
        GkrPack::Chunks { tree_log_cells: 0 },
        GkrPack::Chunks { tree_log_cells: 40 },
        GkrPack::Off,
    ] {
        assert!(
            faulted(
                &cols,
                ProverFaults {
                    pack: Some(pack),
                    ..ProverFaults::default()
                }
            )
            .0
            .is_err(),
            "{pack:?}"
        );
    }
    for t in 3..7 {
        assert_eq!(
            faulted(
                &cols,
                ProverFaults {
                    swap_chunks: Some(t),
                    ..ProverFaults::default()
                }
            )
            .0,
            Err(MlError::BatchMismatch),
            "table {t}'s first two chunks swapped"
        );
    }
    let mut unbalanced = fan_cols();
    unbalanced[1][1][2] = FE::one();
    let (verdict, without_constraint) = faulted(
        &unbalanced,
        ProverFaults {
            pad_mass: Some(3),
            ..ProverFaults::default()
        },
    );
    assert_eq!(verdict, Err(MlError::BatchMismatch));
    assert_eq!(
        without_constraint,
        Ok(()),
        "with the constraint check off the forgery passes: the bus balances and the ladder holds"
    );
}

/// ★ The departures only a two-layer step admits, by a prover that steps
/// honestly after them: a tree's halves sent `h₁, h₀`, each half's `lo` and
/// `hi` traded (both pass their step's residual, which is symmetric in
/// them), and the next step proved at `(c₂, c₁, ρ)`. Each refused at the next
/// step for a tree that goes on — every table of the fixture goes on after
/// step 0.
#[test]
fn a_departing_two_layer_step_is_refused() {
    for fault in [
        StepFault::SwapHalves,
        StepFault::SwapLoHi,
        StepFault::SwapPlane,
    ] {
        for t in 0..3 {
            assert_eq!(
                faulted_double(ProverFaults {
                    ladder: Some((t, fault)),
                    ..Default::default()
                }),
                Err(MlError::LayerRelationMismatch { layer: 1 }),
                "{fault:?} on table {t}"
            );
        }
    }
}

/// A tree that leaves the ladder at its swapped step: the ladder holds, and
/// its false input claim is refused where it meets the trace (the
/// constraint phase); with that check off it is accepted, so nothing else
/// objects. CPU `k = 2` and ADD `k = 2` leave at step 0.
#[test]
fn a_swapped_step_that_reaches_the_input_is_refused_by_the_constraint_phase() {
    let airs = airs();
    let cfg = double(27);
    let committed = commit(tables(&airs, &columns(0, 2), false), &cfg);
    let heights: Vec<usize> = committed
        .tables()
        .iter()
        .map(|t| ArgueShape::of(&t.statement()).input_vars)
        .collect();
    assert_eq!(
        heights,
        vec![2, 2, 0],
        "premise: CPU and ADD leave at step 0"
    );
    let checks = VerifierChecks {
        constraint: false,
        ..VerifierChecks::ALL
    };
    for fault in [StepFault::SwapHalves, StepFault::SwapLoHi] {
        for t in 0..2 {
            let proof = prove(
                &committed,
                &cfg,
                ProverFaults {
                    ladder: Some((t, fault)),
                    ..Default::default()
                },
            );
            assert_eq!(
                verify(&committed, &proof, &cfg),
                Err(MlError::BatchMismatch),
                "{fault:?} on table {t}"
            );
            assert_eq!(
                verify_with_checks(&committed, &proof, &cfg, checks),
                Ok(()),
                "{fault:?} on table {t}: only the constraint phase sees it"
            );
        }
    }
}

// ─── the ladder is the verifier's ────────────────────────────────────────

/// ★ A proof on one ladder is refused by a verifier on the other, both ways.
#[test]
fn a_proof_on_the_other_ladder_is_refused() {
    let airs = airs();
    for (a, pad, cap) in [(3usize, 1usize, 27u8), (2, 0, 3), (1, 3, 27)] {
        let cols = columns(a, pad);
        let single = batched(cap);
        let double = double(cap);
        let committed = commit(tables(&airs, &cols, false), &single);
        let one = prove(&committed, &single, ProverFaults::default());
        let two = prove(&committed, &double, ProverFaults::default());
        verify(&committed, &one, &single).expect("one layer a step");
        verify(&committed, &two, &double).expect("two layers a step");
        assert!(
            verify(&committed, &one, &double).is_err(),
            "a={a} pad={pad} cap={cap}: a one-layer proof under the two-layer verifier"
        );
        assert!(
            verify(&committed, &two, &single).is_err(),
            "a={a} pad={pad} cap={cap}: a two-layer proof under the one-layer verifier"
        );
    }
}

/// Two-layer proofs are deterministic, and differ from the one-layer ones.
#[test]
fn the_two_layer_ladder_is_deterministic() {
    use crypto::fiat_shamir::is_transcript::IsTranscript;
    let airs = airs();
    let run = |cfg: &ChainConfig| {
        let committed = commit(tables(&airs, &columns(3, 1), true), cfg);
        let mut transcript = DefaultTranscript::<Ext>::new(SEED);
        let proof = prove_batched_with(
            &committed,
            cfg,
            &mut transcript,
            None,
            ProverFaults::default(),
        )
        .expect("the prover runs");
        (
            bincode::serialize(&proof).expect("bytes"),
            transcript.state(),
        )
    };
    let (a, b) = (run(&double(27)), run(&double(27)));
    assert!(a == b, "two runs on the two-layer ladder differ");
    assert!(run(&batched(27)) != a, "the ladders give the same bytes");
}

/// ★ The device path (on the card where there is one; its host fallbacks
/// where there is none) proves the host reference's bytes on the two-layer
/// ladder: a card tree's two-layer steps run on the card's two-layer rounds
/// (and they did, where there is a card), its odd tables' last steps on
/// today's. Tables tall enough for the card's trees (CPU 2^13 rows) in one
/// bin and in bins of 2^14 cells, and small ones.
///
/// ```text
/// cargo test --release -p stark --features cuda,multilinear/cuda --lib -- \
///     multilinear_table::batched::tests::double::the_device_path_proves_the_host_references_bytes_on_the_two_layer_ladder --exact --nocapture
/// ```
#[test]
fn the_device_path_proves_the_host_references_bytes_on_the_two_layer_ladder() {
    use crypto::fiat_shamir::is_transcript::IsTranscript;
    let _switches = card_switches();
    let airs = airs();
    for (a, pad, cap) in [(12usize, 1usize, 27u8), (12, 1, 14), (3, 1, 27), (2, 2, 3)] {
        let cfg = double(cap);
        let cols = columns(a, pad);
        let run = |at: Where| {
            let committed = commit(tables(&airs, &cols, true), &cfg);
            let mut transcript = DefaultTranscript::<Ext>::new(SEED);
            let proof = prove_batched_on(
                &committed,
                &cfg,
                &mut transcript,
                None,
                ProverFaults::default(),
                at,
            )
            .unwrap_or_else(|e| panic!("{at:?}: {e:?}"));
            verify(&committed, &proof, &cfg).unwrap_or_else(|e| panic!("{at:?}: {e:?}"));
            (
                bincode::serialize(&proof).expect("bytes"),
                transcript.state(),
            )
        };
        let host = run(Where::Host);
        let before = multilinear::gkr_gruen::gruen2_layers();
        let device = run(Where::Device);
        let card_steps = multilinear::gkr_gruen::gruen2_layers() - before;
        eprintln!(
            "two-layer ladder a={a} pad={pad} cap={cap}: {} bytes, a device {}, {card_steps} card two-layer steps",
            host.0.len(),
            a_device()
        );
        if a_device() && a >= 12 {
            assert!(card_steps > 0, "a={a}: the card ran two-layer steps");
        }
        assert!(
            host == device,
            "a={a} pad={pad} cap={cap}: the bytes differ"
        );
    }
}

// ─── mutations: each check made inert lets its negative through ──────────

/// ★ The verifier's checks are as load-bearing on the two-layer ladder:
/// each made inert, its negative goes through.
#[test]
fn the_verifier_checks_are_load_bearing_on_the_two_layer_ladder() {
    let airs = airs();
    // The residual: a mid-step half off is no longer refused at its step.
    let (committed, mut proof, cfg) = fixture_double(&airs);
    proof.argue.gkr[0].layers[1].halves[2].p_hi += ExtE::one();
    assert_ne!(
        verify_with_checks(
            &committed,
            &proof,
            &cfg,
            VerifierChecks {
                residual: false,
                ..VerifierChecks::ALL
            }
        ),
        Err(MlError::LayerRelationMismatch { layer: 1 }),
        "made inert, the step no longer refuses its own tamper"
    );
    // The constraint equation: a violated constraint is accepted.
    let cfg = double(27);
    let mut cols = columns(3, 1);
    cols[1][2][2] += FE::one();
    cols[0][4][2] += FE::one();
    let committed = commit(tables(&airs, &cols, false), &cfg);
    let proof = prove(&committed, &cfg, ProverFaults::default());
    assert_eq!(
        verify(&committed, &proof, &cfg),
        Err(MlError::BatchMismatch)
    );
    assert_eq!(
        verify_with_checks(
            &committed,
            &proof,
            &cfg,
            VerifierChecks {
                constraint: false,
                ..VerifierChecks::ALL
            }
        ),
        Ok(())
    );
    // The bus balance: an unbalanced bus is accepted.
    let mut cols = columns(3, 1);
    cols[2][3][1] += FE::one();
    let committed = commit(tables(&airs, &cols, false), &cfg);
    let proof = prove(&committed, &cfg, ProverFaults::default());
    assert_eq!(verify(&committed, &proof, &cfg), Err(MlError::BusImbalance));
    assert_eq!(
        verify_with_checks(
            &committed,
            &proof,
            &cfg,
            VerifierChecks {
                balance: false,
                ..VerifierChecks::ALL
            }
        ),
        Ok(())
    );
    // The reduction's shape: a reduction the format has no place for.
    let committed = commit(tables(&airs, &columns(3, 1), true), &cfg);
    let mut proof = prove(&committed, &cfg, ProverFaults::default());
    proof.argue.reduces[1] = proof.argue.reduces[2].clone();
    assert_eq!(
        verify(&committed, &proof, &cfg),
        Err(MlError::ArgueShapeMismatch { part: "reduction" })
    );
    assert_eq!(
        verify_with_checks(
            &committed,
            &proof,
            &cfg,
            VerifierChecks {
                reduce_shape: false,
                ..VerifierChecks::ALL
            }
        ),
        Ok(())
    );
    // The preprocessed columns: a wrong one is accepted.
    let cols = columns(3, 1);
    let committed = commit(tables(&airs, &cols, false), &cfg);
    let proof = prove(&committed, &cfg, ProverFaults::default());
    let mut bad = cols[1][0].clone();
    bad[5] += FE::one();
    let wrong = vec![Mle::new(bad).unwrap()];
    let tables = committed.tables();
    let statements = vec![
        tables[0].statement(),
        tables[1].layout().statement_with_preprocessed(&wrong),
        tables[2].statement(),
    ];
    assert_eq!(
        verify_under(&committed, &statements, &proof, &cfg, VerifierChecks::ALL),
        Err(MlError::EvaluationMismatch)
    );
    assert_eq!(
        verify_under(
            &committed,
            &statements,
            &proof,
            &cfg,
            VerifierChecks {
                preprocessed: false,
                ..VerifierChecks::ALL
            }
        ),
        Ok(())
    );
}
