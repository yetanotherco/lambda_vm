//! Laptop gates for the last level's own format ([`super::last_level`]):
//! the knobs' words, today's levers when they are off, the chain config the
//! levers give under the WHIR round form, and the lean BLAKE3 fixture proved
//! and verified under each lever, with the refusals that show what binds it.

use multilinear::whir_chain::{FinalPoly, RoundForm, WhirFolds, check_round_form};
use stark::multilinear_table::batched::VerifierChecks;

use super::b3l_wlfm_tests::{honest, program};
use super::hash::HasherKind;
use super::last_level::{
    LAST_FINAL_MEASURED, LAST_FIRST_FOLD_MEASURED, LastLevel, evaluated_word, final_word,
    first_fold_word, last_level_setting, wire_word,
};
use super::levels::{LevelStop, MAX_LEVELS, is_last_level, level_levers, levers_for};
use super::wire::WireFormat;
use super::wlfm::{
    StackPlace, WlfmBuild, WlfmFamily, WlfmLevers, WlfmPrep, WlfmProof, WlfmRate,
    execute_and_fill_unmerged, first_fold_schedule, prepare_wlfm_rounds, prove_wlfm,
    prove_wlfm_traces, verify_wlfm, verify_wlfm_with, wlfm_config_levers, wlfm_config_rounds,
    wlfm_config_under,
};
use crate::tables::types::FE;

const HASHER: HasherKind = HasherKind::Poseidon1W16;

/// The levels' round form at the BLAKE3 base's default shrink.
const SHRINK: u8 = 3;

/// Every lever of the last level on, at the measured settings.
const ALL_ON: LastLevel = LastLevel {
    evaluated: true,
    first_fold: Some(LAST_FIRST_FOLD_MEASURED),
    final_poly: FinalPoly::Vars(LAST_FINAL_MEASURED),
};

/// Each lever alone, then all of them.
fn settings() -> Vec<(&'static str, LastLevel)> {
    vec![
        (
            "evaluated",
            LastLevel {
                evaluated: true,
                ..LastLevel::DEFAULT
            },
        ),
        (
            "first fold 4",
            LastLevel {
                first_fold: Some(LAST_FIRST_FOLD_MEASURED),
                ..LastLevel::DEFAULT
            },
        ),
        (
            "final 9",
            LastLevel {
                final_poly: FinalPoly::Vars(LAST_FINAL_MEASURED),
                ..LastLevel::DEFAULT
            },
        ),
        ("all", ALL_ON),
    ]
}

// =============================================================================
// The knobs
// =============================================================================

/// Each knob's word: unset is today's, its own on-words are taken, and
/// anything else is refused rather than read as off.
#[test]
fn the_last_level_knobs_read_their_words_and_refuse_the_rest() {
    assert_eq!(last_level_setting(None, None, None), Ok(LastLevel::DEFAULT));
    assert_eq!(evaluated_word(Some(" 1 ")), Ok(true));
    assert_eq!(evaluated_word(Some("0")), Ok(false));
    assert!(evaluated_word(Some("yes")).is_err());
    assert_eq!(first_fold_word(Some("4")), Ok(Some(4)));
    assert_eq!(first_fold_word(Some("OFF")), Ok(None));
    for bad in ["0", "7", "four", "-1"] {
        assert!(first_fold_word(Some(bad)).is_err(), "{bad}");
    }
    assert_eq!(final_word(Some("9")), Ok(FinalPoly::Vars(9)));
    assert_eq!(final_word(Some("0")), Ok(FinalPoly::Off));
    assert_eq!(final_word(Some("off")), Ok(FinalPoly::Off));
    for bad in ["11", "nine", "-3"] {
        assert!(final_word(Some(bad)).is_err(), "{bad}");
    }
    assert_eq!(wire_word(None), Ok(WireFormat::Rkyv));
    assert_eq!(wire_word(Some("Dedup")), Ok(WireFormat::Dedup));
    assert_eq!(wire_word(Some("compact")), Ok(WireFormat::Compact));
    assert!(wire_word(Some("zip")).is_err());
    assert_eq!(
        last_level_setting(Some("1"), Some("4"), Some("9")),
        Ok(ALL_ON)
    );
}

/// ★ DEFAULT-OFF: with the knobs off, the last level proves under every
/// level's levers, so no level's format moves; and a level that is not the
/// last proves under every level's levers whatever the knobs say.
#[test]
fn the_default_last_level_is_every_levels_levers() {
    for family in [WlfmFamily::P1, WlfmFamily::B3] {
        assert_eq!(LastLevel::DEFAULT.levers(family), level_levers(family));
        assert_eq!(
            levers_for(family, LastLevel::DEFAULT, true),
            level_levers(family)
        );
        for (_, last) in settings() {
            assert_eq!(levers_for(family, last, false), level_levers(family));
        }
        let on = ALL_ON.levers(family);
        assert_eq!(on.prep, WlfmPrep::Evaluated);
        assert_eq!(on.first_fold, Some(4));
        assert_eq!(on.final_poly, FinalPoly::Vars(9));
        assert_eq!((on.fold, on.grind, on.family), (None, None, family));
    }
    assert_eq!(level_levers(WlfmFamily::P1), WlfmLevers::DEFAULT);
}

/// Which level is the last: by its index under `Levels`, by its group count
/// under `Groups`, and at [`MAX_LEVELS`] at the latest.
#[test]
fn the_last_level_is_the_stops() {
    assert!(!is_last_level(LevelStop::Levels(2), 1, 9));
    assert!(is_last_level(LevelStop::Levels(2), 2, 9));
    assert!(is_last_level(LevelStop::Levels(3), 3, 1));
    assert!(!is_last_level(LevelStop::Levels(3), 2, 1));
    assert!(is_last_level(LevelStop::Levels(0), 1, 9), "a chain of one");
    assert!(!is_last_level(LevelStop::Groups(1), 1, 2));
    assert!(is_last_level(LevelStop::Groups(1), 2, 1));
    assert!(is_last_level(LevelStop::Groups(1), MAX_LEVELS, 5));
}

// =============================================================================
// The chain config
// =============================================================================

/// Shapes whose tallest stacked polynomial is 2^27, as a level's group:
/// one table of 16 columns at 2^23 (`one_stack` 23 + 4).
const PRODUCTION_SHAPES: [(usize, usize); 2] = [(16, 23), (8, 20)];

/// ★ The chain config under the last level's levers, against today's:
/// - off, the config is today's, byte for byte (the same fold word, count and
///   format), with or without the round form;
/// - a first fold of 4 is the uniform schedule; 5 is `First(5)`; 0 and 7 are
///   refused;
/// - under the round form, a first fold and a final polynomial are taken and
///   the host's checks hold the count to the rule at the union the stamped
///   schedule gives, while a stamped later fold stays refused, alone or with
///   a first fold;
/// - at the production height (2^27) the counts are the ones the size model
///   priced: round 0 keeps 229, the later rounds 114, 76, 57, 46 to the final
///   round 4 (7 variables left) under first fold 4, shrink 3, final 9.
#[test]
fn the_last_levels_levers_give_a_sized_round_form_config() {
    let shapes = PRODUCTION_SHAPES;
    let today = wlfm_config_rounds(WlfmRate::Two, FinalPoly::Off, None, None, None, &shapes)
        .expect("today's");
    let off = wlfm_config_levers(WlfmRate::Two, &WlfmLevers::DEFAULT, None, &shapes)
        .expect("the default levers");
    assert_eq!(off, today, "default-off is today's config");
    assert_eq!(
        off,
        wlfm_config_under(WlfmRate::Two, FinalPoly::Off, None, None, &shapes)
    );
    let today_w = wlfm_config_rounds(
        WlfmRate::Two,
        FinalPoly::Off,
        None,
        None,
        Some(SHRINK),
        &shapes,
    )
    .expect("today's round form");
    let off_w = wlfm_config_levers(WlfmRate::Two, &WlfmLevers::DEFAULT, Some(SHRINK), &shapes)
        .expect("the default levers under the round form");
    assert_eq!(off_w, today_w, "default-off is today's round-form config");
    assert_eq!(off_w.fold_word(), today_w.fold_word());

    assert_eq!(first_fold_schedule(4), Ok(WhirFolds::Uniform));
    assert!(matches!(first_fold_schedule(5), Ok(WhirFolds::First(k)) if k.get() == 5));
    assert!(first_fold_schedule(0).is_err());
    assert!(first_fold_schedule(7).is_err());

    let levers = ALL_ON.levers(WlfmFamily::B3);
    let on = wlfm_config_levers(WlfmRate::Two, &levers, Some(SHRINK), &shapes)
        .expect("the last level's config");
    check_round_form(&on).expect("the host's count check");
    let RoundForm::Whir(w) = on.format.rounds else {
        panic!("the round form")
    };
    let tallest = 27;
    let union = (1..=tallest).map(|m| on.rounds(m)).max().expect("rounds");
    assert_eq!(
        w.union_rounds(),
        union,
        "the union over the stamped schedule"
    );
    assert_eq!(on.schedule(tallest), vec![4, 4, 4, 4, 4, 4, 3]);
    assert_eq!(today_w.schedule(tallest), vec![6, 4, 4, 4, 4, 4, 1]);
    assert_eq!(on.final_round(tallest), Some((4, 7)));
    let queries = on.round_queries(tallest);
    println!(
        "LAST LEVEL COUNTS at 2^{tallest}: today {:?} · last level {queries:?} · union {union}",
        today_w.round_queries(tallest)
    );
    assert_eq!(
        on.num_queries, today_w.num_queries,
        "round 0 keeps its count"
    );
    assert_eq!(on.num_queries, 229);
    assert_eq!(&queries[..5], &[229, 114, 76, 57, 46]);
    assert_ne!(
        on.fold_word(),
        today_w.fold_word(),
        "the statement binds it"
    );

    // Each lever alone under the round form: taken and checked.
    for (name, last) in settings() {
        let c = wlfm_config_levers(
            WlfmRate::Two,
            &last.levers(WlfmFamily::B3),
            Some(SHRINK),
            &shapes,
        )
        .unwrap_or_else(|e| panic!("{name}: {e}"));
        check_round_form(&c).unwrap_or_else(|e| panic!("{name}: {e:?}"));
    }
    // A stamped later fold stays refused under the round form, and a first
    // fold beside a later one is refused under either form.
    let stamped = WlfmLevers {
        fold: Some(5),
        ..WlfmLevers::DEFAULT
    };
    let e = wlfm_config_levers(WlfmRate::Two, &stamped, Some(SHRINK), &shapes)
        .expect_err("a stamped fold");
    assert!(e.contains("stamped fold"), "{e}");
    let both = WlfmLevers {
        fold: Some(5),
        first_fold: Some(4),
        ..WlfmLevers::DEFAULT
    };
    for rounds in [None, Some(SHRINK)] {
        let e = wlfm_config_levers(WlfmRate::Two, &both, rounds, &shapes)
            .expect_err("two schedule levers");
        assert!(e.contains("one schedule lever"), "{e}");
    }
}

// =============================================================================
// The lean BLAKE3 fixture under the last level's levers
// =============================================================================

/// `program` built under the BLAKE3 family with `last`'s levers, its chains
/// under the round form, its stacks (if any) on the host.
fn build_last(program: &super::compiler::LfmProgram, last: LastLevel) -> WlfmBuild {
    prepare_wlfm_rounds(
        program,
        HASHER,
        WlfmRate::Two,
        last.levers(WlfmFamily::B3),
        Some(SHRINK),
    )
    .and_then(|pre| pre.commit(StackPlace::Host))
    .unwrap_or_else(|e| panic!("the build under {last:?}: {e}"))
}

fn prove_last(program: &super::compiler::LfmProgram, build: &WlfmBuild) -> WlfmProof {
    prove_wlfm(program, build, &[]).unwrap_or_else(|e| panic!("the prove: {e}"))
}

fn compact(proof: &WlfmProof) -> usize {
    super::wire::encode(proof).len()
}

/// ★ The lean fixture (BLAKE3 family, round form) proves and verifies under
/// each lever of the last level alone and under all of them; each is another
/// identity; all of them are smaller than today's; a proof under one setting
/// is refused against another's artifacts; and the census's parts sum to the
/// compact total.
#[test]
fn the_lean_fixture_proves_under_each_last_level_lever() {
    let program = honest();
    let today = build_last(&program, LastLevel::DEFAULT);
    let today_proof = prove_last(&program, &today);
    verify_wlfm(&today.artifacts, &today_proof, &today_proof.public_words)
        .expect("today's proof verifies");
    assert!(!today_proof.prepared.is_empty(), "today opens its stacks");
    let base_bytes = compact(&today_proof);
    let c = super::wire::census(&today_proof);
    assert_eq!(
        c.parts.iter().map(|(_, v)| v).sum::<usize>(),
        c.compact,
        "the census adds up"
    );
    println!("LAST LEVEL FIXTURE today: {base_bytes} B · {}", c.line());
    for (name, last) in settings() {
        let build = build_last(&program, last);
        let a = &build.artifacts;
        assert_ne!(a.program_id, today.artifacts.program_id, "{name}: identity");
        let proof = prove_last(&program, &build);
        verify_wlfm(a, &proof, &proof.public_words)
            .unwrap_or_else(|e| panic!("{name}: the proof verifies: {e}"));
        assert_eq!(proof.prepared.is_empty(), last.evaluated, "{name}");
        // Across settings, both ways.
        assert!(
            verify_wlfm(&today.artifacts, &proof, &proof.public_words).is_err(),
            "{name}: refused by today's artifacts"
        );
        assert!(
            verify_wlfm(a, &today_proof, &today_proof.public_words).is_err(),
            "{name}: today's proof refused by its artifacts"
        );
        let c = super::wire::census(&proof);
        assert_eq!(c.parts.iter().map(|(_, v)| v).sum::<usize>(), c.compact);
        println!(
            "LAST LEVEL FIXTURE {name}: {} B (today {base_bytes}) · dedup {} · rkyv {} · Q {:?}",
            c.compact, c.dedup, c.rkyv, c.queries
        );
        if name == "all" || name == "evaluated" {
            assert!(c.compact < base_bytes, "{name}: smaller");
        }
        // The compact files round-trip to the same proof, which verifies.
        for wire in [WireFormat::Compact, WireFormat::Dedup] {
            let bytes = super::last_level::last_level_file(&proof, wire).expect("encodes");
            let back = super::last_level::read_last_level_file(&bytes, wire).expect("decodes");
            assert_eq!(super::wire::encode(&back), super::wire::encode(&proof));
            verify_wlfm(a, &back, &back.public_words)
                .unwrap_or_else(|e| panic!("{name} {wire:?}: read back: {e}"));
        }
    }
}

/// The forger's prover: `program`'s execution proved against `build`, no
/// prefix check.
fn forge(program: &super::compiler::LfmProgram, build: &WlfmBuild) -> WlfmProof {
    let filled = execute_and_fill_unmerged(program, &[], HASHER).expect("the program executes");
    prove_wlfm_traces(build, filled, false).expect("the forger's prover")
}

/// ★ THE EVALUATED BINDING OF THE LEAN TABLES, under every lever of the last
/// level: a forged head instruction column (the twin program's flags,
/// proved against the honest program's artifacts) is refused, and only by
/// the verifier's own evaluation of the columns — with the preprocessed check
/// off (a mutation) the same proof verifies.
#[test]
fn a_forged_lean_head_column_is_refused_by_the_last_levels_evaluation() {
    let honest = honest();
    let forged = program(crypto::hash::blake3::chain::FLAGS_ONE_BLOCK as u8 ^ 0x10);
    let build = build_last(&honest, ALL_ON);
    let proof = forge(&forged, &build);
    assert!(proof.prepared.is_empty());
    let e = verify_wlfm(&build.artifacts, &proof, &proof.public_words)
        .expect_err("a forged flags column must be refused");
    println!("FORGED HEAD COLUMN: {e}");
    verify_wlfm_with(
        &build.artifacts,
        &proof,
        &proof.public_words,
        false,
        VerifierChecks {
            preprocessed: false,
            ..VerifierChecks::ALL
        },
    )
    .unwrap_or_else(|e| panic!("with the column check off the forgery verifies: {e}"));
}

/// ★ Every lean table's derived instruction columns (the round rows' and the
/// byte table's, the heads', the selects' and caps' when present) are bound
/// by the evaluation: one value of any one column changed in the verifier's
/// copy — the same check a prover with another column meets — refuses the
/// honest proof, and with the preprocessed check off (a mutation) it does
/// not. The identity is held as it was, so the check is all that moves.
#[test]
fn every_lean_tables_derived_columns_are_bound_by_the_evaluation() {
    let program = honest();
    let build = build_last(&program, ALL_ON);
    let proof = prove_last(&program, &build);
    let a = &build.artifacts;
    let airs = a.airs();
    let refs = airs.air_refs();
    let lean_from = refs.len() - airs.b3l.refs().len();
    let mut checked = Vec::new();
    for (t, air) in refs.iter().enumerate().skip(lean_from) {
        for c in 0..a.counts[t] {
            let mut tampered = a.clone();
            let mut values = tampered.evaluated[t][c].evals().to_vec();
            values[0] += FE::one();
            tampered.evaluated[t][c] = multilinear::mle::Mle::new(values).expect("a column");
            let name = air.name();
            assert!(
                verify_wlfm(&tampered, &proof, &proof.public_words).is_err(),
                "{name} column {c}: a changed value is refused"
            );
            verify_wlfm_with(
                &tampered,
                &proof,
                &proof.public_words,
                false,
                VerifierChecks {
                    preprocessed: false,
                    ..VerifierChecks::ALL
                },
            )
            .unwrap_or_else(|e| panic!("{name} column {c}: with the check off: {e}"));
            checked.push(format!("{name}:{c}"));
        }
    }
    println!("LEAN COLUMNS BOUND: {}", checked.join(" "));
    assert!(
        !checked.is_empty(),
        "the fixture has lean tables with prefixes"
    );
}

/// The final polynomial is bound: one value of it changed is refused.
#[test]
fn a_changed_final_polynomial_value_is_refused() {
    use multilinear::whir_chain::RoundOpenings;
    let program = honest();
    let build = build_last(&program, ALL_ON);
    let proof = prove_last(&program, &build);
    let mut tampered = proof.clone();
    let mut hit = false;
    'find: for stack in tampered.proof.columns.iter_mut() {
        for chain in stack.polys.iter_mut() {
            for round in chain.rounds.iter_mut() {
                if let RoundOpenings::Extension(p) = &mut round.openings
                    && let Some(block) = p.next.first_mut()
                    && block.proof.merkle_path.is_empty()
                    && !block.values.is_empty()
                {
                    block.values[0] += math::field::element::FieldElement::<
                        crate::tables::types::GoldilocksExtension,
                    >::one();
                    hit = true;
                    break 'find;
                }
            }
        }
    }
    assert!(hit, "a chain of the fixture reaches its final polynomial");
    assert!(verify_wlfm(&build.artifacts, &tampered, &tampered.public_words).is_err());
}

/// A W-leg refuses a child under any lever of the last level: only the host
/// verifies the last level.
#[test]
fn a_w_leg_refuses_a_child_under_the_last_levels_levers() {
    let program = honest();
    for (name, last) in settings() {
        let build = build_last(&program, last);
        let child = super::wlfm_leg::WlfmChild {
            artifacts: build.artifacts.clone(),
            num_public_words: 4,
            cap_mux: None,
        };
        let e = super::wlfm_leg::wlfm_leg_program(&child)
            .expect_err("a child the in-guest verifier cannot check");
        println!("W-LEG REFUSES {name}: {e}");
    }
}
