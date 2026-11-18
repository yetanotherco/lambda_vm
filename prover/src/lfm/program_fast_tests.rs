//! Level 1's program made the fast way (`LAMBDA_VM_LEVELS_PROGRAM_FAST`):
//! the parallel compile is the sequential one, instruction for instruction and
//! row for row, at every slicing; the admission validator's verdict beside the
//! execution is the validator's, and an inadmissible program is refused by it;
//! the knob is on unless its word is `0`. (The driver's level bytes with the knob on against the
//! levels proved with it off: the box-tier
//! `levels_tests::the_blake3_driver_with_its_program_fast_proves_the_same_level_bytes`.)

use super::compiler::{LfmProgram, compile, compile_parallel_sliced};
use super::instr::Instr;
use super::levels::{PROGRAM_FAST_ENV, joined_validation, spawn_validation};

fn verdict(program: &LfmProgram) -> Result<f64, String> {
    std::thread::scope(|scope| joined_validation(spawn_validation(scope, program)))
}

/// The beside verdict accepts what the validator accepts.
#[test]
fn an_admissible_program_passes_the_beside_validator() {
    for program in [super::wlfm_tests::honest(), super::b3l_wlfm_tests::honest()] {
        super::validator::validate(&program).expect("the fixture is admissible");
        verdict(&program).expect("and the beside verdict says so");
    }
}

/// A program whose multiplicity moved (as a compiler bug would) is refused by
/// the beside validator, as by the validator itself.
#[test]
fn an_inadmissible_program_is_refused_by_the_beside_validator() {
    let mut program = super::wlfm_tests::honest();
    let moved = program.instrs.iter_mut().find_map(|i| match i {
        Instr::BaseAlu { mult, .. } | Instr::Const { mult, .. } => {
            *mult += 1;
            Some(())
        }
        _ => None,
    });
    assert!(
        moved.is_some(),
        "the fixture has a base ALU or constant row"
    );
    assert!(
        super::validator::validate(&program).is_err(),
        "the validator refuses it"
    );
    let err = verdict(&program).expect_err("the beside verdict refuses it");
    assert!(err.contains("not admissible"), "{err}");
}

/// The knob's name; it is on by default, and only `0` turns it off.
#[test]
fn the_program_fast_knob_is_named_and_value_based() {
    use super::wlfm_knobs::knob_word_off;
    assert_eq!(PROGRAM_FAST_ENV, "LAMBDA_VM_LEVELS_PROGRAM_FAST");
    assert!(knob_word_off(Some("0")) && knob_word_off(Some(" 0 ")));
    for on in [None, Some(""), Some("1"), Some("yes"), Some("00")] {
        assert!(!knob_word_off(on), "{on:?} leaves it on");
    }
}

/// With no word in the environment the driver makes level 1's program the
/// fast way (unless a test overrides it).
#[test]
fn the_program_fast_knob_is_on_by_default() {
    if std::env::var_os(PROGRAM_FAST_ENV).is_none() {
        assert!(super::levels::program_fast());
    }
}

/// A program source, made afresh at each call.
type SourceFn = Box<dyn Fn() -> super::builder::LfmProgramSource>;

/// Program sources the compile is checked over: the trivial program, every
/// opcode, a keccak chain (its rows bind their ordinal), a BLAKE3 sponge, and
/// the BLAKE3 W-leg over a lean program's build (lean head rows, which bind
/// theirs, and muxes; ≈ 0.7 M instructions).
fn sources() -> Vec<(&'static str, SourceFn)> {
    use super::programs as p;
    use super::wlfm::{WlfmFamily, WlfmLevers, WlfmRate, build_wlfm_under};
    use super::wlfm_leg::{WlfmChild, emit_wlfm_leg, wlfm_leg_builder};
    let build = build_wlfm_under(
        &super::b3l_wlfm_tests::honest(),
        super::wlfm_tests::HASHER,
        WlfmRate::Two,
        WlfmLevers::with_family(WlfmFamily::B3),
        true,
    )
    .expect("builds");
    let child = WlfmChild {
        artifacts: build.artifacts.clone(),
        num_public_words: 8,
    };
    vec![
        ("trivial", Box::new(p::trivial_program_source)),
        (
            "permute coverage",
            Box::new(p::permute_coverage_program_source),
        ),
        ("keccak chain", Box::new(p::keccak_chain_program_source)),
        (
            "keccak sponge",
            Box::new(|| p::keccak_sponge_program_source(700)),
        ),
        (
            "blake3 sponge",
            Box::new(|| p::blake3_sponge_program_source(700)),
        ),
        (
            "blake3 w-leg",
            Box::new(move || {
                let mut b = wlfm_leg_builder(WlfmFamily::B3);
                let arena = b.declare_arena(0);
                let leg = emit_wlfm_leg(&mut b, &child, arena).expect("emits");
                for word in &leg.publics {
                    let packed =
                        b.pack_word([word.lanes[0], word.lanes[1], word.lanes[2], word.lanes[3]]);
                    b.public(packed);
                }
                b.finish()
            }),
        ),
    ]
}

/// ★ The parallel compile is the sequential one, whatever the slicing (one
/// instruction a slice, a few, and the production slice): every instruction,
/// every multiplicity, every group's rows and tags, and the program's shape.
#[test]
fn the_parallel_compile_is_the_compile_at_every_slicing() {
    for (name, source) in sources() {
        let want = format!("{:?}", compile(source()));
        for slice in [1usize, 3, 7, 4096, 1 << 20] {
            let got = format!("{:?}", compile_parallel_sliced(source(), slice));
            assert!(
                got == want,
                "{name}: slice {slice} compiles another program"
            );
        }
    }
}

/// A source with an address written twice compiles in parallel to a program
/// the validator refuses (where the sequential compile panics): the parallel
/// compile leaves its invariants to the validator, which the driver runs.
#[test]
fn a_twice_written_address_compiles_in_parallel_to_a_refused_program() {
    let mut source = super::programs::trivial_program_source();
    let first_out = source
        .instrs
        .iter()
        .find_map(|i| match i {
            Instr::Const { out, .. } => Some(*out),
            _ => None,
        })
        .expect("a constant");
    let twice = source
        .instrs
        .iter_mut()
        .filter_map(|i| match i {
            Instr::Const { out, .. } => Some(out),
            _ => None,
        })
        .nth(1)
        .expect("two constants");
    *twice = first_out;
    let program = compile_parallel_sliced(source, 2);
    assert!(super::validator::validate(&program).is_err());
    assert!(verdict(&program).is_err());
}

/// The beside-nice knob's name; only `1` turns it on, and it is off by
/// default.
#[test]
fn the_beside_nice_knob_is_named_and_off_by_default() {
    use super::levels::{BESIDE_NICE_ENV, beside_nice};
    use super::wlfm_knobs::knob_word_on;
    assert_eq!(BESIDE_NICE_ENV, "LAMBDA_VM_LEVELS_BESIDE_NICE");
    assert!(knob_word_on(Some("1")) && !knob_word_on(Some("0")) && !knob_word_on(None));
    if std::env::var_os(BESIDE_NICE_ENV).is_none() {
        assert!(!beside_nice());
    }
}

/// Level 1's stacks prepared on the beside pool are the prepare's on the
/// prover's pool: the same identity, the same prepared cells, the same chip
/// prefixes, committed.
#[test]
fn the_beside_pool_prepares_the_same_level() {
    use super::levels::{prepare_level, prepare_level_beside, set_beside_nice_for_test};
    use super::wlfm::{StackPlace, WlfmFamily};
    let program = super::b3l_wlfm_tests::honest();
    let plain = prepare_level(&program, WlfmFamily::B3).expect("prepares");
    let beside = {
        let _nice = set_beside_nice_for_test(true);
        prepare_level_beside(&program, WlfmFamily::B3).expect("prepares on the beside pool")
    };
    assert_eq!(plain.prepared_cells(), beside.prepared_cells());
    let plain = plain.commit(StackPlace::Host).expect("commits");
    let beside = beside.commit(StackPlace::Host).expect("commits");
    assert_eq!(beside.artifacts.program_id, plain.artifacts.program_id);
    assert!(beside.chip_prefixes() == plain.chip_prefixes());
}

/// The setup-fast knob's name; only `1` turns it on, and it is off by
/// default.
#[test]
fn the_setup_fast_knob_is_named_and_off_by_default() {
    use super::levels::{SETUP_FAST_ENV, setup_fast};
    assert_eq!(SETUP_FAST_ENV, "LAMBDA_VM_LEVELS_SETUP_FAST");
    if std::env::var_os(SETUP_FAST_ENV).is_none() {
        assert!(!setup_fast());
    }
}

/// The bucket-waves knob's name; it is on by default, and only `0` turns it
/// off.
#[test]
fn the_bucket_waves_knob_is_named_and_on_by_default() {
    use super::levels::{BUCKET_WAVES_ENV, bucket_waves};
    assert_eq!(BUCKET_WAVES_ENV, "LAMBDA_VM_LEVELS_BUCKET_WAVES");
    if std::env::var_os(BUCKET_WAVES_ENV).is_none() {
        assert!(bucket_waves());
    }
}
