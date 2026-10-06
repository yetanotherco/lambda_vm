//! The width-16 socket proved and verified, and the necessity of each of its
//! constraints (the socket review R-P1SOCKET's R1–R3):
//!
//! - **R1.** A `Hash16` program proved and verified end to end through the
//!   production LogUp path, and rows that satisfy every socket constraint but
//!   disagree with memory (or park a real witness on padding) refused by the
//!   verifier.
//! - **R2.** Each S-box constraint individually necessary: the other square
//!   root of `x³` violates exactly its `x³` constraint, and a moved `x⁷` with
//!   every later round recomputed violates exactly its `x⁷` constraint. The
//!   value-column perturbations in `p1w16_socket_tests` are caught by the NEXT
//!   round's constraints and cannot see either go missing. The bus side: a
//!   transposed coefficient matrix misreads the outputs.
//! - **R3.** The socket's `LFM_HASH` AIR lowered in-guest (the leg a level-1
//!   node runs for a `Poseidon1W16` child) agrees with the IR interpreter and the
//!   host verifier folder.

use stark::proof::options::{GoldilocksCubicProofOptions, ProofOptions};
use stark::trace::TraceTable;

use crate::tables::types::{FE, GoldilocksExtension, GoldilocksField, VmTable};
use crypto::hash::poseidon1_w16 as p1;

use super::builder::{Cell, LfmBuilder};
use super::compiler::{LfmProgram, compile};
use super::executor::execute;
use super::hash::HasherKind;
use super::p1w16_socket::{SOCKET_FORM, cols, fill_row, sboxed_lanes};
use super::proof::{prove_traces_with_hasher, verify_against};
use super::registry::build_artifacts_with_hasher;
use super::trace::build_traces_with_hasher;

type F = GoldilocksField;
type E = GoldilocksExtension;

const KIND: HasherKind = HasherKind::Poseidon1W16;

fn options() -> ProofOptions {
    GoldilocksCubicProofOptions::with_blowup(2).expect("options")
}

fn word(seed: u64) -> [FE; 4] {
    core::array::from_fn(|i| FE::from(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ i as u64))
}

/// Two chained permutations; the second reads the first's cell 0 twice and
/// every output of the second is published.
fn program() -> LfmProgram {
    let mut b = LfmBuilder::new();
    let ins: [Cell; 4] = core::array::from_fn(|k| b.digest_const(word(k as u64 + 1)).as_cell());
    let first = b.hash16(ins);
    let zero = b.digest_const([FE::zero(); 4]).as_cell();
    let second = b.hash16([first[0], zero, first[0], ins[3]]);
    for c in second {
        b.public(c);
    }
    compile(b.finish())
}

fn row_of(t: &TraceTable<F, E>, r: usize) -> Vec<FE> {
    (0..t.num_cols()).map(|c| *t.get_main(r, c)).collect()
}

fn write_row(t: &mut TraceTable<F, E>, r: usize, row: &[FE]) {
    for (c, v) in row.iter().enumerate().skip(cols::PREP_WIDTH) {
        t.main_table.set_fe(r, c, *v);
    }
}

/// `fill_row` without round constants: the witness a padding row
/// (`IS_REAL = 0`) satisfies for any `IN`.
fn fill_padding_row(input: [FE; 16], row: &mut [FE]) {
    row[cols::IN0..cols::IN0 + 16].copy_from_slice(&input);
    let mut s = input;
    for r in 0..p1::NUM_ROUNDS {
        let mut f = s;
        for (lane, a) in f.iter_mut().enumerate().take(sboxed_lanes(r)) {
            let x3 = *a * *a * *a;
            let x7 = x3 * x3 * *a;
            row[SOCKET_FORM.x3(r, lane)] = x3;
            row[SOCKET_FORM.x7(r, lane)] = x7;
            *a = x7;
        }
        s = p1::mds(&f);
    }
}

/// `Ok(true)` = accepted; `Ok(false)` = verifier rejection; `Err` = prover refusal.
fn round_trip(mutate: impl FnOnce(&mut TraceTable<F, E>)) -> Result<bool, String> {
    let opts = options();
    let program = program();
    super::validator::validate(&program).expect("admissible");
    let artifacts = build_artifacts_with_hasher(&program, &opts, KIND);
    let exec = execute(&program, &[], &KIND).expect("execute");
    let mut traces = build_traces_with_hasher(&program, &exec.records, KIND);
    mutate(&mut traces.hash);
    match prove_traces_with_hasher(
        &artifacts,
        &mut traces,
        &exec.public_words,
        &opts,
        KIND,
        stark::residency_mode::ResidencyMode::Retain,
    ) {
        Ok(proof) => Ok(verify_against(
            &artifacts.roots,
            &artifacts.program_id,
            artifacts.keccak_rnd_chunks,
            &proof,
            &exec.public_words,
            &opts,
            KIND,
            artifacts.chip_set,
        )),
        Err(e) => Err(format!("{e:?}")),
    }
}

/// Every socket constraint on `row` evaluates to zero (the row-local view).
fn satisfies_socket(row: &[FE]) -> bool {
    violated(row).is_empty()
}

#[test]
fn honest_hash16_program_proves_and_verifies() {
    assert_eq!(round_trip(|_| {}), Ok(true));
}

/// Row 0 recomputed from a different input: every constraint of the row holds,
/// its reads disagree with memory and its write disagrees with row 1's reads.
#[test]
fn a_consistent_row_over_a_wrong_input_is_not_accepted() {
    let r = round_trip(|t| {
        let mut row = row_of(t, 0);
        let mut input: [FE; 16] = core::array::from_fn(|i| row[cols::IN0 + i]);
        input[5] += FE::one();
        fill_row(SOCKET_FORM, input, &mut row);
        assert!(
            satisfies_socket(&row),
            "the forged row is constraint-consistent"
        );
        write_row(t, 0, &row);
    });
    assert_ne!(r, Ok(true), "{r:?}");
}

/// The published row recomputed from a different input: the outputs it sends
/// are a true permutation, but of the wrong cells.
#[test]
fn a_consistent_published_row_over_a_wrong_input_is_not_accepted() {
    let r = round_trip(|t| {
        let mut row = row_of(t, 1);
        let mut input: [FE; 16] = core::array::from_fn(|i| row[cols::IN0 + i]);
        input[15] += FE::one();
        fill_row(SOCKET_FORM, input, &mut row);
        assert!(satisfies_socket(&row));
        write_row(t, 1, &row);
    });
    assert_ne!(r, Ok(true), "{r:?}");
}

/// A padding row's `IN` is free and harmless: any input with the
/// constant-free witness is accepted, because the row sends and receives
/// nothing (IS_REAL = 0, MULT_k = 0 in the preprocessed group).
#[test]
fn garbage_on_a_padding_row_is_accepted_and_harmless() {
    let r = round_trip(|t| {
        assert!(t.num_rows() > 2, "a padding row exists");
        let last = t.num_rows() - 1;
        let mut row = row_of(t, last);
        assert_eq!(row[cols::IS_REAL], FE::zero());
        let input: [FE; 16] = core::array::from_fn(|i| FE::from(0xDEAD_0000 + i as u64));
        fill_padding_row(input, &mut row);
        assert!(satisfies_socket(&row));
        write_row(t, last, &row);
    });
    assert_eq!(r, Ok(true), "{r:?}");
}

/// The honest real-row witness placed on a padding row is NOT a valid padding
/// row (the constants do not vanish from it), so the prover cannot park a real
/// permutation there either.
#[test]
fn a_real_witness_on_a_padding_row_is_not_accepted() {
    let r = round_trip(|t| {
        let last = t.num_rows() - 1;
        let mut row = row_of(t, last);
        let input: [FE; 16] = core::array::from_fn(|i| FE::from(7 + i as u64));
        fill_row(SOCKET_FORM, input, &mut row);
        assert!(!satisfies_socket(&row));
        write_row(t, last, &row);
    });
    assert_ne!(r, Ok(true), "{r:?}");
}

/// The second square root: `x3 -> -x3` keeps `x7 = x3^2 * a`, so ONLY the
/// `x3 = a^3` constraint stands between the prover and a free S-box output.
/// Every single-column `+1` perturbation (the existing load-bearing test) is
/// caught by `x7`'s constraint instead, so it cannot see that constraint go.
#[test]
fn the_other_square_root_of_x3_is_caught_by_the_x3_constraint_alone() {
    let input: [FE; 16] = core::array::from_fn(|i| FE::from(0x1234_5678 + 17 * i as u64));
    let mut row = vec![FE::zero(); SOCKET_FORM.num_columns()];
    row[cols::IS_REAL] = FE::one();
    fill_row(SOCKET_FORM, input, &mut row);
    assert!(satisfies_socket(&row));
    let mut idx = 1 + 4;
    let mut checked = 0;
    for r in 0..p1::NUM_ROUNDS {
        for lane in 0..sboxed_lanes(r) {
            let mut bad = row.clone();
            let c = SOCKET_FORM.x3(r, lane);
            bad[c] = -bad[c];
            assert_eq!(violated(&bad), vec![idx], "round {r} lane {lane}");
            idx += 2;
            checked += 1;
        }
    }
    assert_eq!(checked, 150);
}

/// The production host evaluator of a bus value (`BusValue::combine_from`, the
/// i64 coefficient path) reads the OnBus output tokens as the permutation's
/// output cells.
#[test]
fn the_production_bus_evaluator_reads_the_outputs() {
    use math::field::element::FieldElement;
    let input: [FE; 16] = core::array::from_fn(|i| -FE::from(1 + i as u64));
    let mut row = vec![FE::zero(); SOCKET_FORM.num_columns()];
    row[cols::IS_REAL] = FE::one();
    let out = fill_row(SOCKET_FORM, input, &mut row);
    assert_eq!(out, p1::permute(input));
    let ix = super::p1w16_socket::bus_interactions(SOCKET_FORM);
    for (k, i) in ix.iter().enumerate().skip(4) {
        let cell = k - 4;
        let token: Vec<FieldElement<E>> = i
            .values
            .iter()
            .flat_map(|v| v.combine_from(|c| row[c].to_extension::<E>()))
            .collect();
        for j in 0..4 {
            assert_eq!(
                token[1 + j],
                out[4 * cell + j].to_extension::<E>(),
                "cell {cell}"
            );
        }
    }
}

/// The test-side mutation the review could not run in the chip
/// (`mds_entry(lane, i)` → `mds_entry(i, lane)` in `bus_interactions`): the
/// production interactions with their OnBus coefficient matrix transposed,
/// read by the production evaluator, no longer carry the permutation's
/// outputs — so the output check above is load-bearing for the coefficients'
/// orientation (Plonky3's circulant is not symmetric, so the transpose is a
/// different matrix).
#[test]
fn a_transposed_bus_coefficient_matrix_misreads_the_outputs() {
    use stark::lookup::{BusValue, LinearTerm};
    let input: [FE; 16] = core::array::from_fn(|i| FE::from(0x0BAD_C0DE + 29 * i as u64));
    let mut row = vec![FE::zero(); SOCKET_FORM.num_columns()];
    row[cols::IS_REAL] = FE::one();
    let out = fill_row(SOCKET_FORM, input, &mut row);
    let honest = super::p1w16_socket::bus_interactions(SOCKET_FORM);
    // The production matrix, read off the interactions: lane `o`'s term on
    // the last round's `x⁷(i)`.
    let last = p1::NUM_ROUNDS - 1;
    let mut m = [[0i64; 16]; 16];
    for (k, ix) in honest.iter().enumerate().skip(4) {
        for (j, v) in ix.values.iter().enumerate().skip(1) {
            let BusValue::Linear(terms) = v else {
                panic!("an OnBus output is a linear form")
            };
            for t in terms {
                let LinearTerm::Column {
                    coefficient,
                    column,
                } = t
                else {
                    panic!("an OnBus term is a column term")
                };
                let i = (0..16)
                    .find(|&i| SOCKET_FORM.x7(last, i) == *column)
                    .expect("a last-round x7 column");
                m[4 * (k - 4) + j - 1][i] = *coefficient;
            }
        }
    }
    assert!(
        (0..16).any(|o| (0..16).any(|i| m[o][i] != m[i][o])),
        "the MDS matrix is not symmetric, so its transpose is another matrix"
    );
    let mut mutant = honest.clone();
    for (k, ix) in mutant.iter_mut().enumerate().skip(4) {
        for (j, v) in ix.values.iter_mut().enumerate().skip(1) {
            let o = 4 * (k - 4) + j - 1;
            *v = BusValue::Linear(
                (0..16)
                    .map(|i| LinearTerm::Column {
                        coefficient: m[i][o],
                        column: SOCKET_FORM.x7(last, i),
                    })
                    .collect(),
            );
        }
    }
    let read = |ixs: &[stark::lookup::BusInteraction]| -> Vec<FE> {
        ixs.iter()
            .skip(4)
            .flat_map(|ix| {
                ix.values
                    .iter()
                    .skip(1)
                    .map(|v| v.combine_from(|c| row[c])[0])
            })
            .collect()
    };
    assert_eq!(
        read(&honest),
        out.to_vec(),
        "the production matrix reads P(IN)"
    );
    let misread = read(&mutant);
    assert_ne!(
        misread,
        out.to_vec(),
        "the transposed matrix reads something else"
    );
    assert!(
        misread.iter().zip(&out).filter(|(a, b)| a != b).count() > 1,
        "the transpose moves more than one output lane"
    );
}

fn violated(row: &[FE]) -> Vec<usize> {
    use math::field::element::FieldElement;
    use stark::constraints::builder::{ConstraintSet, ProverEvalFolder};
    use stark::frame::Frame;
    use stark::table::TableView;
    use stark::traits::TransitionEvaluationContext;
    let set = super::p1w16_socket::P1W16SocketConstraints { form: SOCKET_FORM };
    let n = ConstraintSet::<F, E>::meta(&set).len();
    let no_ch: Vec<FieldElement<E>> = vec![];
    let offset = FieldElement::<E>::zero();
    let frame = Frame::<F, E>::new(vec![TableView::new(vec![row.to_vec()], vec![vec![]])]);
    let ctx =
        TransitionEvaluationContext::new_prover(frame.as_row_frame(), &no_ch, &no_ch, &offset);
    let mut base_out = vec![FE::zero(); n];
    let mut ext_out = vec![FieldElement::<E>::zero(); n];
    let mut folder = ProverEvalFolder::new(&ctx, &mut base_out, &mut ext_out);
    set.eval(&mut folder);
    base_out
        .iter()
        .enumerate()
        .filter(|(_, v)| **v != FE::zero())
        .map(|(i, _)| i)
        .collect()
}

/// A real row whose S-box `(fr, flane)` output is moved by `delta` AFTER its
/// constraints are checked, every later round recomputed from the forged state.
fn forged_x7_row(input: [FE; 16], fr: usize, flane: usize, delta: FE) -> Vec<FE> {
    use super::p1w16_socket::cols;
    let mut row = vec![FE::zero(); SOCKET_FORM.num_columns()];
    row[cols::IS_REAL] = FE::one();
    row[cols::IN0..cols::IN0 + 16].copy_from_slice(&input);
    let mut s = input;
    for (r, rc) in p1::constants::ROUND_CONSTANTS.iter().enumerate() {
        let mut f: [FE; 16] = core::array::from_fn(|i| s[i] + FE::from(rc[i]));
        for (lane, a) in f.iter_mut().enumerate().take(sboxed_lanes(r)) {
            let x3 = *a * *a * *a;
            let mut x7 = x3 * x3 * *a;
            if (r, lane) == (fr, flane) {
                x7 += delta;
            }
            row[SOCKET_FORM.x3(r, lane)] = x3;
            row[SOCKET_FORM.x7(r, lane)] = x7;
            *a = x7;
        }
        s = p1::mds(&f);
    }
    row
}

/// Each `x7 = x3^2 * a` constraint is individually necessary: a row that
/// moves one S-box output and recomputes everything downstream violates
/// exactly that constraint and nothing else. (The existing single-column
/// perturbation tests are caught by the NEXT round's constraints for every
/// round but the last, so they would not notice one of these going missing.)
#[test]
fn every_x7_constraint_is_individually_necessary() {
    let input: [FE; 16] = core::array::from_fn(|i| FE::from(0xABCD_0000 + 3 * i as u64));
    let mut idx = 1 + 4;
    let mut checked = 0;
    for r in 0..p1::NUM_ROUNDS {
        for lane in 0..sboxed_lanes(r) {
            let row = forged_x7_row(input, r, lane, FE::one());
            assert_eq!(violated(&row), vec![idx + 1], "round {r} lane {lane}");
            idx += 2;
            checked += 1;
        }
    }
    assert_eq!(checked, 150);
}

/// R3, point (a) of the review on the VERIFIER paths: the machine's `LFM_HASH` AIR under
/// `Poseidon1W16` (socket constraints + LogUp with the OnBus i64 terms),
/// captured, serialized, lowered to LFM instructions (the in-guest leg) and
/// executed, agrees with the IR interpreter and with the host verifier folder
/// (`AIR::compute_transition`) on random all-extension OOD frames.
#[test]
fn the_socket_air_lowered_in_guest_matches_the_host_verifier() {
    use super::airs::{ChipSet, LfmAirs, NUM_LFM_CHIPS};
    use super::constraints::{
        OodOperands, analyze, emit_analyzed, hint_ood_frame, ood_frame_words,
    };
    use super::word::{ext_word, word_as_ext};
    use crate::tables::types::FEE;
    use stark::constraint_ir::{ConstraintArtifact, eval_program_verifier};
    use stark::frame::Frame;
    use stark::table::TableView;
    use stark::traits::TransitionEvaluationContext;

    let opts = options();
    let airs = LfmAirs::new_with_hasher(&[[0u8; 32]; NUM_LFM_CHIPS], &opts, 1, KIND, ChipSet::FULL);
    let refs = airs.air_refs();
    let air = refs
        .iter()
        .find(|a| a.name() == "LFM_HASH")
        .expect("LFM_HASH in the set");
    assert_eq!(
        air.constraints_meta().len(),
        SOCKET_FORM.num_constraints() + 4
    );

    let artifact = ConstraintArtifact::capture(&**air);
    let bytes = artifact.to_bytes().expect("serialize");
    let artifact = ConstraintArtifact::from_bytes(&bytes).expect("deserialize");
    let prog = artifact.program();
    let n = prog.roots.len();
    let an = analyze(&artifact);

    // The differential program (constraint_tests::differential_program).
    let mut b = LfmBuilder::new();
    let frame_arena = b.declare_arena(ood_frame_words(&artifact));
    let (steps, words) = hint_ood_frame(&mut b, &artifact, frame_arena, 0);
    assert_eq!(words, ood_frame_words(&artifact));
    let shape = &artifact.shape;
    let num_uniforms = 2 + (shape.max_bus_elements as usize + 2) + 1;
    let uniform_arena = b.declare_arena(num_uniforms as u32);
    let mut next = 0u32;
    let mut take = |b: &mut LfmBuilder| {
        let c = b.hint_word(uniform_arena, next).as_ext();
        next += 1;
        c
    };
    let rap_challenges = vec![take(&mut b), take(&mut b)];
    let alpha_powers: Vec<_> = (0..shape.max_bus_elements as usize + 2)
        .map(|_| take(&mut b))
        .collect();
    let table_offset = take(&mut b);
    let ood = OodOperands {
        steps,
        main_width: shape.main_width as usize,
        rap_challenges,
        alpha_powers,
        table_offset,
    };
    let evals = emit_analyzed(&mut b, &an, &ood);
    for e in &evals {
        b.public(e.as_cell());
    }
    let program = compile(b.finish());
    super::validator::validate(&program).expect("admissible");

    let main_width = shape.main_width as usize;
    let width = main_width + shape.aux_width as usize;
    let num_steps = shape.transition_offsets.len().max(1);
    let mut seed = 0x5EED_0001u64;
    let mut rnd = || {
        seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    for trial in 0..3 {
        let mut fp3 = || FEE::new([FE::from(rnd()), FE::from(rnd()), FE::from(rnd())]);
        let ostep: Vec<Vec<FEE>> = (0..num_steps)
            .map(|offset| {
                (0..width)
                    .map(|col| {
                        if offset == 0 || shape.next_row_columns.contains(&(col as u32)) {
                            fp3()
                        } else {
                            FEE::zero()
                        }
                    })
                    .collect()
            })
            .collect();
        let rap: Vec<FEE> = vec![fp3(), fp3()];
        let alphas: Vec<FEE> = (0..shape.max_bus_elements as usize + 2)
            .map(|_| fp3())
            .collect();
        let offset_v = fp3();

        let mut frame_words = Vec::new();
        for (offset, step) in ostep.iter().enumerate() {
            for (col, v) in step.iter().enumerate() {
                if offset == 0 || shape.next_row_columns.contains(&(col as u32)) {
                    frame_words.push(ext_word(v));
                }
            }
        }
        let uniforms: Vec<_> = rap
            .iter()
            .chain(&alphas)
            .chain(std::iter::once(&offset_v))
            .map(ext_word)
            .collect();
        let exec = execute(&program, &[frame_words, uniforms], &HasherKind::Rpx).expect("executes");

        let frame = Frame::<E, E>::new(
            ostep
                .iter()
                .map(|s| {
                    TableView::new(
                        vec![s[..main_width].to_vec()],
                        vec![s[main_width..].to_vec()],
                    )
                })
                .collect(),
        );
        let ctx =
            TransitionEvaluationContext::<F, E>::new_verifier(&frame, &rap, &alphas, &offset_v);
        let mut interp = vec![FEE::zero(); n];
        eval_program_verifier(&prog, &ctx, &mut interp);
        let folder = air.compute_transition(&ctx);
        assert!(
            interp.iter().all(|v| *v != FEE::zero()),
            "trial {trial}: a random frame leaves every constraint nonzero"
        );
        assert_eq!(folder.len(), n);
        for c in 0..n {
            let cell = exec.memory.get(evals[c].addr()).expect("written");
            let got = word_as_ext(&cell).expect("ext");
            assert_eq!(
                got, interp[c],
                "trial {trial}: constraint {c}: guest vs interpreter"
            );
            assert_eq!(
                folder[c], interp[c],
                "trial {trial}: constraint {c}: host folder vs interpreter"
            );
        }
    }
}

/// Point (a) of the review on the GPU aux path's encoding: the fingerprint descriptor the
/// device kernel consumes (its host mirror `eval_fingerprint`) gives every
/// socket interaction the fingerprint the host computes from the bus values.
#[cfg(feature = "cuda")]
#[test]
fn the_gpu_fingerprint_descriptor_matches_the_host() {
    use math::field::element::FieldElement;
    use stark::logup_gpu::{build_fingerprint_descriptor, eval_fingerprint};
    let input: [FE; 16] = core::array::from_fn(|i| FE::from(0xFFFF_FFFF_0000_0000 - i as u64));
    let mut row = vec![FE::zero(); SOCKET_FORM.num_columns()];
    row[cols::IS_REAL] = FE::one();
    for k in 0..4 {
        row[cols::IN_ADDR0 + k] = FE::from(10 + k as u64);
        row[cols::OUT_ADDR0 + k] = FE::from(20 + k as u64);
    }
    fill_row(SOCKET_FORM, input, &mut row);
    let ix = super::p1w16_socket::bus_interactions(SOCKET_FORM);
    let d = build_fingerprint_descriptor(&ix, 2);
    let z = FieldElement::<E>::new([FE::from(3u64), FE::from(5u64), FE::from(7u64)]);
    let alpha = FieldElement::<E>::new([FE::from(11u64), FE::from(13u64), FE::from(17u64)]);
    let powers: Vec<FieldElement<E>> = (0..d.alpha_powers_len.max(6))
        .scan(FieldElement::<E>::one(), |p, _| {
            let cur = p.clone();
            *p = &*p * &alpha;
            Some(cur)
        })
        .collect();
    for (k, i) in ix.iter().enumerate() {
        let mut lc = FieldElement::<E>::from(i.bus_id);
        for (j, v) in i.values.iter().enumerate() {
            let val = v.combine_from(|c| row[c].to_extension::<E>());
            lc += &val[0] * &powers[j + 1];
        }
        let want = &z - &lc;
        let got = eval_fingerprint(&d, k, |c| &row[c], &powers, &z);
        assert_eq!(got, want, "interaction {k}");
    }
    let max_coef = d.term_coef.iter().copied().max().unwrap();
    assert_eq!(
        max_coef, 101,
        "the OnBus coefficients reach the descriptor as the MDS entries"
    );
}
