//! Tests for the WHIR level-0 driver, which lives in
//! [`crate::lfm::whir_real_epoch`].
//!
//! They stayed here when the driver moved out, deliberately: they are the
//! control that the move changed nothing, and two of them reach
//! `#[cfg(test)]`-only counters (`decode_derivations`) that a production module
//! cannot call.

use crate::lfm::whir_real_epoch::*;
use crate::tables::local_to_global::epoch_label;
use crate::tables::register;
use executor::elf::Elf;
use multilinear::whir_hash::WhirHash;
use stark::config::Commitment;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multilinear_continuation;
    use crate::test_utils::asm_elf_bytes;
    use stark::proof::options::ProofOptions;

    /// A run whose epochs touch memory across the boundary, and which publishes
    /// output in its last epoch.
    fn a_run() -> (Vec<u8>, Vec<u8>) {
        let mut input: Vec<u8> = Vec::with_capacity(16);
        input.extend_from_slice(&16u32.to_le_bytes());
        input.extend_from_slice(&[0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]);
        input.extend_from_slice(&[0u8; 4]);
        (asm_elf_bytes("test_private_input_xpage"), input)
    }

    fn bundle() -> (
        Vec<u8>,
        ProofOptions,
        multilinear_continuation::ContinuationProof,
    ) {
        let (elf_bytes, input) = a_run();
        let opts = ProofOptions::default_test_options();
        let bundle = multilinear_continuation::prove_continuation(&elf_bytes, &input, 2, &opts)
            .expect("prove the continuation");
        (elf_bytes, opts, bundle)
    }

    /// ★ Every epoch of a real bundle harvests, including the LAST one — which
    /// the session path cannot build and which is the only epoch that publishes.
    #[test]
    fn every_epoch_of_a_bundle_harvests() {
        let (elf_bytes, opts, b) = bundle();
        assert!(b.epochs.len() >= 2, "a one-epoch run chains nothing");

        let mut published = 0usize;
        for index in 0..b.epochs.len() {
            let e = real_epoch_from_whir_continuation(&opts, &elf_bytes, &b, index, None)
                .unwrap_or_else(|e| panic!("epoch {index}: {e}"));
            assert_eq!(e.position.is_final, index + 1 == b.epochs.len());
            assert_eq!(e.position.label, epoch_label(index as u64));
            assert!(e.num_tables() > 0);
            assert_eq!(e.pc_start, Elf::load(&elf_bytes).expect("load").entry_point);
            assert_eq!(e.elf_digest, crate::statement::elf_digest(&elf_bytes));
            // `chain_config`'s constants, which `agrees_with` on the other
            // lineage also rests on. The query count is not asserted here: it
            // is discharged by the epoch VERIFYING under this config above.
            assert_eq!(e.config.log_blowup, 2);
            assert_eq!(e.config.log_folding, 4);
            assert!(e.config.num_queries > 0);
            published += usize::from(!e.public_output().is_empty());
        }
        assert!(
            published > 0,
            "no epoch published output, so the harvest never exercised the one \
             shape that differs"
        );
    }

    /// ★★ THE CHAIN IS THE DRIVER'S, NOT THE BUNDLE'S.
    ///
    /// Epoch 0 starts from the ELF's entry point and every later epoch from the
    /// PREVIOUS epoch's proved `reg_fini`. A driver that took the starting
    /// registers from the epoch being harvested would accept a bundle whose
    /// epochs are about unrelated register files, and nothing else here would
    /// notice.
    #[test]
    fn the_starting_registers_come_from_the_previous_epoch() {
        let (elf_bytes, _, b) = bundle();
        let elf = Elf::load(&elf_bytes).expect("load");
        assert!(b.epochs.len() >= 2);

        let first = whir_epoch_chain_position(&b, &elf, 0).expect("epoch 0");
        assert_eq!(
            first.register_init,
            register::register_init_from_entry_point(elf.entry_point),
            "epoch 0 must start from the ELF"
        );
        for index in 1..b.epochs.len() {
            let p = whir_epoch_chain_position(&b, &elf, index).expect("in range");
            assert_eq!(
                p.register_init,
                b.epochs[index - 1].reg_fini,
                "epoch {index} must start where epoch {} was proved to end",
                index - 1
            );
        }
        assert!(whir_epoch_chain_position(&b, &elf, b.epochs.len()).is_none());
    }

    /// ★★ A PROOF THAT DOES NOT VERIFY IS NOT HARVESTED.
    ///
    /// The tamper is the epoch's PUBLIC OUTPUT, which `absorb_epoch` binds into
    /// the statement before any challenge — so restating it moves every
    /// challenge and the proof cannot verify. Without the acceptance check the
    /// driver would hand the guest a wrap input built from it, and the failure
    /// would surface a whole wrap prove later, reading as an emitter bug.
    ///
    /// ⚠ THE OBVIOUS TAMPER — THE REGISTER CARRY — WAS NOT USABLE HERE WHEN
    /// THIS TEST WAS WRITTEN, AND THE REASON WAS A DEFECT.
    /// `VmAirs::new` gave REGISTER a preprocessed COMMITMENT and no columns
    /// closure, so the multilinear verifier's `check_preprocessed` walked an
    /// empty list in zero iterations and a restated `reg_fini` verified. That
    /// is fixed on this base: 5e3df0c0 binds INIT and FINI as columns, and the
    /// carry is refused by `verify_epoch` itself under
    /// `a_restated_register_carry_is_refused_by_the_epoch_it_lands_in` and
    /// `a_restated_register_fini_is_refused_by_the_epoch_that_states_it`
    /// (`multilinear_continuation_tests`), each beside its accept control.
    ///
    /// The PUBLIC OUTPUT remains this test's tamper by choice rather than by
    /// necessity: it is the field `absorb_epoch` binds before any challenge, so
    /// it exercises the driver's acceptance check through the statement the
    /// guest's replay will read, which is what this driver is for.
    #[test]
    fn an_epoch_that_does_not_verify_is_refused() {
        let (elf_bytes, opts, mut b) = bundle();
        assert!(b.epochs.len() >= 2);

        // The control first: the untouched bundle harvests. A refusal test
        // without it passes on a driver that refuses everything.
        real_epoch_from_whir_continuation(&opts, &elf_bytes, &b, 1, None)
            .expect("the untouched bundle must harvest");

        b.epochs[1].public_output.push(0xFF);
        let refused = real_epoch_from_whir_continuation(&opts, &elf_bytes, &b, 1, None);
        assert!(
            refused.is_err(),
            "a restated public output was harvested into a wrap input"
        );
    }

    /// An index past the end is an error, not a panic.
    #[test]
    fn an_index_past_the_end_is_refused() {
        let (elf_bytes, opts, b) = bundle();
        assert!(
            real_epoch_from_whir_continuation(&opts, &elf_bytes, &b, b.epochs.len(), None).is_err()
        );
    }

    /// ★★ THE WIDTHS ARE THE AIRS', AND THAT DISTINCTION IS NOT COSMETIC.
    ///
    /// `chain_config` takes the tallest STACK, and a stack's height is
    /// `one_stack(num_vars, width)` — so a config built from the heights alone,
    /// with widths stubbed at 1, is a DIFFERENT config with a different query
    /// count. The first draft of the driver did exactly that, because the proof
    /// states heights and not widths.
    ///
    /// This is what makes the distinction observable: the stubbed form must
    /// disagree with the harvested one. If it ever agrees, the assertion is
    /// vacuous at this shape and says so rather than passing quietly.
    #[test]
    fn a_config_built_without_the_widths_is_a_different_config() {
        let (elf_bytes, opts, b) = bundle();
        let harvested = real_epoch_from_whir_continuation(&opts, &elf_bytes, &b, 0, None)
            .expect("epoch 0 harvests");

        // ⚠ The assertion is on the TALLEST STACK and not on `num_queries`.
        // `num_queries` is a step function of the round count, so two different
        // stack heights can land on the same query count — measured: at this
        // fixture's shapes both forms give 112, and an inequality on the query
        // count would have been vacuous here while reading as a real check.
        // The stack height is where the widths actually enter.
        let tallest = |shapes: &[(usize, usize)]| {
            shapes
                .iter()
                .map(|&(width, num_vars)| {
                    multilinear::constraint_argument::one_stack(num_vars, width)
                })
                .max()
                .unwrap_or(1)
        };
        let stubbed: Vec<(usize, usize)> = harvested
            .proof
            .table_num_vars
            .iter()
            .map(|&n| (1usize, n as usize))
            .collect();
        let real: Vec<(usize, usize)> = harvested.shapes.to_vec();

        assert_ne!(
            tallest(&real),
            tallest(&stubbed),
            "the widths make no difference to the tallest stack at this shape, so \
             this test cannot see the defect it exists for"
        );
        assert_eq!(
            harvested.config,
            crate::multilinear_prove::chain_config(&real),
            "the harvested config is not the one its own shapes imply"
        );
    }

    /// ★ THE SUPPLIED COMMITMENT IS THE ONE CARRIED, not a rebuilt equal.
    ///
    /// `Some(c)` exists so a walk over every epoch takes DECODE's commitment
    /// once per bundle rather than once per epoch — lane P measured the STARK
    /// driver rebuilding it 38 times over 19 epochs. A driver that accepted the
    /// argument and rebuilt anyway would pass a value comparison against the
    /// real root, so the value handed in here is deliberately NOT the real one.
    /// It reaches only the AIR build's DECODE root, which the multilinear path
    /// does not compare, so the epoch still verifies.
    #[test]
    fn the_supplied_decode_commitment_is_the_one_carried() {
        let (elf_bytes, opts, b) = bundle();
        let supplied: Commitment = [0xABu8; 32];
        let e = real_epoch_from_whir_continuation(&opts, &elf_bytes, &b, 0, Some(supplied))
            .expect("epoch 0 harvests");
        assert_eq!(
            e.decode_commitment, supplied,
            "the driver ignored the commitment it was handed and built its own"
        );
    }

    /// A bundle whose epochs are proven under a hash NAMED HERE, not inherited
    /// from the process.
    ///
    /// ★★ AND THAT DISTINCTION IS THE WHOLE POINT OF THE HELPER. The cheap
    /// version of the refusal test proves with `prove_continuation` and harvests
    /// at `RpxWhir`; `prove_continuation` dispatches on the cached knob, which is
    /// keccak only because `LAMBDA_VM_WHIR_HASH` is usually unset. Run the suite
    /// with `LAMBDA_VM_WHIR_HASH=rpx` and that bundle is RPX, the two arms SILENTLY
    /// INVERT, and the failure reads as a broken hash agreement when it is a
    /// configuration mismatch. Re-proving the epochs under a literal `KeccakWhir`
    /// costs one more prove of the same run and makes the test mean the same thing
    /// in every process.
    ///
    /// The shell is a real bundle — a real cross-epoch proof, a real touched page
    /// set — because only the epochs are replaced. The driver reads `bundle.epochs`
    /// and nothing else, so the shell's own hash cannot reach this measurement.
    fn a_keccak_bundle() -> (
        Vec<u8>,
        ProofOptions,
        multilinear_continuation::ContinuationProof,
    ) {
        use multilinear::whir_hash::KeccakWhir;

        let (elf_bytes, input) = a_run();
        let opts = ProofOptions::default_test_options();
        let mut b = multilinear_continuation::prove_continuation(&elf_bytes, &input, 2, &opts)
            .expect("prove the continuation");

        let elf = Elf::load(&elf_bytes).expect("load");
        let artifacts = crate::tables::trace_builder::DecodeArtifacts::from_elf(&elf)
            .expect("decode artifacts");
        let prepared =
            multilinear_continuation::decode_prepared_for::<KeccakWhir>(&elf, &elf_bytes)
                .expect("DECODE's prepared commitment under keccak");

        let mut epochs = Vec::new();
        crate::continuation::for_each_epoch(&elf, &input, 2, &artifacts, |prepared_epoch, _| {
            let crate::continuation::PreparedEpoch {
                register_init,
                label,
                traces,
                boundary,
                is_final,
                ..
            } = prepared_epoch;
            epochs.push(multilinear_continuation::prove_epoch::<KeccakWhir>(
                &elf,
                &elf_bytes,
                &register_init,
                label,
                traces,
                is_final,
                &boundary,
                &opts,
                None,
                &prepared,
            )?);
            Ok(())
        })
        .expect("prove every epoch under a literal keccak");

        assert_eq!(
            epochs.len(),
            b.epochs.len(),
            "the explicit pass split the run differently from prove_continuation's"
        );
        assert!(epochs.len() >= 2, "a one-epoch run chains nothing");
        b.epochs = epochs;
        (elf_bytes, opts, b)
    }

    /// ★★ SEAM 1: THE HASH AGREEMENT, AT THE DRIVER'S OWN ENTRY POINT.
    ///
    /// The bundle carries no hash tag and cannot — a WHIR proof's bytes are
    /// hash-agnostic by design, which is the byte gate's own invariant — and the
    /// process knob describes the process, not the bundle. So the refusal is the
    /// verification: hand the driver an `H`, and a bundle proven under another
    /// hash fails because the transcript's sponge is part of the configuration.
    ///
    /// ⚠ BOTH ARMS, ON THE SAME BUNDLE. `verify_epoch_bookend` collapses every
    /// failure to `Ok(None)`, so the refusal carries no reason of its own and a
    /// driver that refused everything would pass the refusing half alone. The
    /// accept arm is what makes the refusal mean something.
    ///
    /// # What was mutated, and the ratio stated plainly
    ///
    /// ONE PROGRAM MUTATION: the verdict computed and discarded in
    /// [`real_epoch_from_whir_continuation_under`] — the mutated driver builds a
    /// `WhirRealEpoch` from a proof that failed verification, which the original
    /// never does. It turns THIS test red on the `HARVESTED under RPX` arm, and
    /// `an_epoch_that_does_not_verify_is_refused` and
    /// `a_restated_register_carry_is_refused_by_the_driver` red on theirs, while
    /// the six non-refusal tests stay green.
    ///
    /// ONE TEST-SIDE CONTROL, which is NOT a program mutation and is not counted
    /// as one: running the refusal arm at `KeccakWhir` too. That changes the
    /// TEST, not the driver — it is the accept arm read the other way — so what
    /// it demonstrates is this test's discriminating power, that the arm is
    /// measuring `H` and not something incidental about the bundle. It fails on
    /// the `Ok(_)` arm, as it must.
    #[test]
    fn an_epoch_proven_under_keccak_is_refused_when_harvested_under_rpx() {
        use multilinear::whir_hash::{KeccakWhir, RpxWhir};

        let (elf_bytes, opts, b) = a_keccak_bundle();

        // THE CONTROL FIRST, and on epoch 1 so the register carry is exercised
        // with it: the hash it was proven under must ACCEPT.
        let accepted = real_epoch_from_whir_continuation_under::<KeccakWhir>(
            &opts, &elf_bytes, &b, 1, None, None,
        )
        .expect("a keccak bundle must harvest under keccak");
        assert_eq!(
            accepted.position.label,
            epoch_label(1),
            "the control harvested some other epoch"
        );

        // A `match` rather than `expect_err`, which would want `WhirRealEpoch:
        // Debug` — a derive on a production type added for a test's
        // convenience, on a struct holding a whole proof.
        let refused = match real_epoch_from_whir_continuation_under::<RpxWhir>(
            &opts, &elf_bytes, &b, 1, None, None,
        ) {
            Err(reason) => reason,
            Ok(_) => panic!(
                "an epoch proven under keccak was HARVESTED under RPX; level 0's \
                 whole hash agreement rests on that being impossible"
            ),
        };

        // The reason is NAMED, and it names the hash the harvest RAN UNDER
        // rather than the process knob — which in this very test is whatever
        // the suite was started with, and is not RPX.
        assert!(
            refused.contains(<RpxWhir as WhirHash>::NAME),
            "the refusal must name the hash it verified under; it said: {refused}"
        );
        assert!(
            !refused.contains(<KeccakWhir as WhirHash>::NAME),
            "the refusal named the bundle's hash instead of the verifier's: {refused}"
        );
    }

    /// ★ THE POSTURE IS REPORTED AND DECIDES NOTHING.
    ///
    /// ⚠ THE ASSERTION IS A RELATION, NOT A VALUE. `whir_hash_knob::selected()`
    /// is a cached process-global, so `assert!(note.is_some())` would be an
    /// assertion about how the whole test binary was invoked — green under the
    /// default and red under `LAMBDA_VM_WHIR_HASH=rpx`, for no reason anybody
    /// reading the test would guess. What is true in every process is that the
    /// note fires exactly when the setting is not RPX.
    ///
    /// ★ AND IT FAILS IN BOTH DIRECTIONS, which a relation alone would not. The
    /// setting is read into a local ONCE and the note's text must contain THAT
    /// name, so a note naming a setting other than the one in force is red — not
    /// merely a note that fires at the wrong time. Measured: making the note
    /// report the opposite setting turns this test red on the naming assert
    /// while the `is_some()` relation still holds, so the two asserts catch
    /// different defects.
    #[test]
    fn the_process_hash_posture_is_reported_and_never_decides() {
        let setting = crate::whir_hash_knob::selected();
        let note = whir_process_posture_note();

        assert_eq!(
            note.is_some(),
            setting != crate::whir_hash_knob::Setting::Rpx,
            "the posture note disagreed with the process setting {setting:?}"
        );
        if let Some(text) = &note {
            assert!(
                text.contains(setting.name()),
                "the note must name the setting it reports: {text}"
            );
            assert!(
                text.contains(crate::whir_hash_knob::ENV),
                "the note must name the knob an operator would change: {text}"
            );
        }

        // AND IT DECIDES NOTHING. A bundle proven under this process's own hash
        // harvests whether the note fired or not — which is the half that would
        // break if anyone ever turned this report into a refusal.
        let (elf_bytes, opts, b) = bundle();
        real_epoch_from_whir_continuation(&opts, &elf_bytes, &b, 0, None)
            .expect("the posture note must not refuse anything");
    }

    /// ★ THE RESTATED REGISTER CARRY IS REFUSED — at the driver, on the base
    /// that fixed it.
    ///
    /// This file used to carry an `#[ignore]`d red flag recording the opposite:
    /// flipping one bit of `epochs[0].reg_fini` — the vector that IS epoch 1's
    /// `register_init` — and harvesting epoch 1 was ACCEPTED, because
    /// `VmAirs::new` gave REGISTER a preprocessed commitment and no columns
    /// closure and the multilinear verifier checks columns. 5e3df0c0 binds both
    /// ends as columns. The red flag is deleted; this is what replaces it, at
    /// the entry point level 0 actually calls.
    ///
    /// Both indices W1d established are flipped: index 1, and `X254_INDEX`, the
    /// synthetic commit index that rides in the same vector. The untouched
    /// bundle is harvested first, or a driver that refused everything would pass.
    #[test]
    fn a_restated_register_carry_is_refused_by_the_driver() {
        let (elf_bytes, opts, mut b) = bundle();
        assert!(b.epochs.len() >= 2, "a one-epoch run chains nothing");

        real_epoch_from_whir_continuation(&opts, &elf_bytes, &b, 1, None)
            .expect("the control: the untouched bundle must harvest epoch 1");

        let mut carry = b.epochs.clone();
        carry[0].reg_fini[1] ^= 1;
        let restated = std::mem::replace(&mut b.epochs, carry);
        assert!(
            real_epoch_from_whir_continuation(&opts, &elf_bytes, &b, 1, None).is_err(),
            "epoch 1 was harvested against a register file the chain never handed it"
        );

        let mut commit_index = restated;
        commit_index[0].reg_fini[crate::tables::register::X254_INDEX] ^= 1;
        b.epochs = commit_index;
        assert!(
            real_epoch_from_whir_continuation(&opts, &elf_bytes, &b, 1, None).is_err(),
            "epoch 1 was harvested against a restated commit index"
        );
    }
    /// ★ THE HANDED-IN PREPARED COMMITMENT IS THE ONE USED, not a rebuilt equal.
    ///
    /// `prepared` exists so a walk over every epoch derives DECODE's prepared
    /// commitment ONCE per bundle instead of once per epoch — fifteen
    /// derivations to one on the block. A driver that took the argument and
    /// derived its own anyway would be INDISTINGUISHABLE on the accept path, so
    /// the check is a prepared built from a DIFFERENT program: used, it must
    /// refuse; ignored, the epoch verifies exactly as it does today and this
    /// test fails. That asymmetry is the whole design — the refusal arm is what
    /// proves USE, and the accept arm is what stops a driver that refuses
    /// everything from passing it.
    ///
    /// ⚠ AND THE OTHER HALF IS UNREACHABLE RATHER THAN UNTESTED. A prepared
    /// built for the wrong HASH cannot be handed to this function at all:
    /// `DecodePrepared<H>` carries the hash in its type and the driver is
    /// `::<H>`, so the mismatch is a compile error. That is strictly better than
    /// a runtime refusal, and it is why no test for it exists — a test for a
    /// state the type system forbids is a check that cannot fail.
    ///
    /// ⚠ The refusal is NOT a shape guard. `DecodePrepared::agrees_with`
    /// compares only `log_blowup` and `log_folding`, which two programs at the
    /// same options share, so a wrong-program prepared sails past it and is
    /// caught by the derived roots block the transcript absorbs — the same
    /// cryptographic mechanism as the hash agreement. The reason is printed so
    /// a reader can see which path actually fired.
    #[test]
    fn the_prepared_commitment_handed_in_is_the_one_used() {
        use multilinear::whir_hash::KeccakWhir;

        let (elf_bytes, opts, b) = a_keccak_bundle();
        let elf = Elf::load(&elf_bytes).expect("load");

        // THE ACCEPT CONTROL: the bundle's own prepared commitment, handed in
        // rather than derived, must harvest.
        let own = multilinear_continuation::decode_prepared_for::<KeccakWhir>(&elf, &elf_bytes)
            .expect("the bundle's own prepared commitment");
        real_epoch_from_whir_continuation_under::<KeccakWhir>(
            &opts,
            &elf_bytes,
            &b,
            1,
            None,
            Some(&own),
        )
        .expect("the bundle's own prepared commitment must harvest");

        // A prepared built from a DIFFERENT program, at the same options.
        let other_bytes = asm_elf_bytes("sub");
        let other = Elf::load(&other_bytes).expect("load sub");
        let wrong =
            multilinear_continuation::decode_prepared_for::<KeccakWhir>(&other, &other_bytes)
                .expect("sub's prepared commitment");
        let refused = match real_epoch_from_whir_continuation_under::<KeccakWhir>(
            &opts,
            &elf_bytes,
            &b,
            1,
            None,
            Some(&wrong),
        ) {
            Err(reason) => reason,
            Ok(_) => panic!(
                "the driver harvested an epoch against a DECODE commitment prepared \
                 from a DIFFERENT program, so the `prepared` argument is accepted \
                 and ignored — a parameter that changes nothing is a display"
            ),
        };
        println!("PREPARED-REFUSAL  {refused}");
    }
    /// The ELF the bench knobs name, resolved WITHOUT panicking when it is
    /// absent.
    ///
    /// `multilinear_bench_tests::elf_bytes` panics on a missing program, which
    /// is right for a bench that must not silently measure the wrong thing and
    /// wrong for a test whose contract is to SKIP. Same two directories, same
    /// order, so the two cannot disagree about where a program lives.
    fn bench_elf_if_present(name: &str) -> Option<Vec<u8>> {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("workspace root")
            .join("executor/program_artifacts");
        for dir in ["rust", "asm"] {
            if let Ok(bytes) = std::fs::read(root.join(dir).join(format!("{name}.elf"))) {
                return Some(bytes);
            }
        }
        None
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        use sha2::Digest;
        let mut h = sha2::Sha256::new();
        h.update(bytes);
        h.finalize().iter().map(|b| format!("{b:02x}")).collect()
    }

    /// ★★ THE DRIVER AGAINST A REAL BLOCK BUNDLE: every epoch harvested, from
    /// ONE derivation of DECODE's prepared commitment.
    ///
    /// Everything the driver's other tests check runs on a three-epoch toy at
    /// 2^2. This runs the same walk a level-0 tree will run — fifteen epochs at
    /// 2^21 on the box — and is the only place the driver meets a card.
    ///
    /// # What it asserts, and what each assert would catch
    ///
    /// - EVERY epoch harvests. A driver that worked on epoch 0 and not on epoch
    ///   14 would be found here and nowhere else.
    /// - THE DERIVATION COUNT IS 1 AFTER THE WALK. The counter is the
    ///   production one `decode_residency_tests` uses, not a copy. The reset
    ///   sits AFTER the prove on purpose: `prove_continuation` derives its own,
    ///   and counting it would make this line describe the prove instead of the
    ///   walk.
    /// - THE CHAIN. Labels are `epoch_label(0..n)` in order, `is_final` is true
    ///   on the last epoch and only there, and epoch i+1's `register_init` IS
    ///   epoch i's proved `reg_fini`. That is the property
    ///   `the_starting_registers_come_from_the_previous_epoch` checks on three
    ///   epochs, over fifteen real ones.
    /// - THE POSTURE NOTE fires iff the knob is not RPX, so a keccak arm and an
    ///   RPX arm read differently and both are checked rather than one being
    ///   assumed.
    ///
    /// ⚠ NOT AN ASSERT, DELIBERATELY: comparing the harvested epochs' table
    /// counts to the bundle's. `WhirRealEpoch::num_tables()` returns
    /// `self.proof.table_num_vars.len()` and `proof` is a clone of the bundle's
    /// epoch, so that comparison is a value against itself and cannot fail on
    /// any input. What is checked instead is each epoch's AIR-derived
    /// `shapes.len()` against its stated height count — still only a
    /// restatement of the driver's own per-epoch guard, but one that runs
    /// fifteen times on real shapes rather than never.
    ///
    /// ⚠ AND THERE IS NO SHA GUARD THAT REFUSES. The transcript pin refuses a
    /// non-pinned ELF because it asserts exact COUNTS, which are a function of
    /// one program. Every assert here is structural and holds for any program,
    /// so refusing would reject valid runs and would also put this test out of
    /// reach of anyone without the block fixture. It prints the ELF's FULL
    /// 64-hex sha256 instead — full, never a prefix, for the reason
    /// `pin_skip_line` records: a diagnostic that can agree while the values
    /// differ is not a diagnostic.
    #[test]
    #[ignore = "the box runs it: a real block bundle under the process hash"]
    fn the_block_bundle_harvests_every_epoch_under_the_process_hash() {
        let name = std::env::var("LAMBDA_VM_BENCH_ELF").unwrap_or_else(|_| "ethrex".into());
        let input_name = std::env::var("LAMBDA_VM_BENCH_INPUT").unwrap_or_default();
        let epoch_size_log2: u32 = std::env::var("LAMBDA_VM_BENCH_EPOCH_LOG2")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(20);

        let Some(elf_bytes) = bench_elf_if_present(&name) else {
            println!(
                "L0-HARVEST SKIPPED - no ELF named {name} in executor/program_artifacts/\
                 {{rust,asm}}; set LAMBDA_VM_BENCH_ELF to a program that exists"
            );
            return;
        };
        let input = crate::tests::multilinear_bench_tests::input_bytes(&input_name);
        let opts = ProofOptions::default_test_options();

        println!(
            "L0-HARVEST fixture {name} sha {} ({} bytes)  input {} ({} bytes)  epoch 2^{}",
            sha256_hex(&elf_bytes),
            elf_bytes.len(),
            if input_name.is_empty() {
                "<none>"
            } else {
                &input_name
            },
            input.len(),
            epoch_size_log2,
        );

        crate::with_whir_hash!(|H| {
            let b = multilinear_continuation::prove_continuation(
                &elf_bytes,
                &input,
                epoch_size_log2,
                &opts,
            )
            .expect("prove the continuation under the process hash");

            let elf = Elf::load(&elf_bytes).expect("load");

            // ⚠ AFTER the prove. `prove_continuation` derives its own; counting
            // it would make the line below describe the prove, not the walk.
            multilinear_continuation::reset_decode_derivations();
            let prepared = multilinear_continuation::decode_prepared_for::<H>(&elf, &elf_bytes)
                .expect("DECODE's prepared commitment for the process hash");

            let started = std::time::Instant::now();
            let mut harvested = Vec::with_capacity(b.epochs.len());
            for index in 0..b.epochs.len() {
                let at = std::time::Instant::now();
                let e = real_epoch_from_whir_continuation_under::<H>(
                    &opts,
                    &elf_bytes,
                    &b,
                    index,
                    None,
                    Some(&prepared),
                )
                .unwrap_or_else(|e| panic!("epoch {index} of the block bundle: {e}"));
                let max_vars = e.shapes.iter().map(|&(_, v)| v).max().unwrap_or(0);
                println!(
                    "L0-HARVEST epoch {index}  tables {}  max_vars {max_vars}  harvest {:.3} s",
                    e.num_tables(),
                    at.elapsed().as_secs_f64(),
                );
                assert_eq!(
                    e.shapes.len(),
                    b.epochs[index].table_num_vars.len(),
                    "epoch {index}'s layout width and its stated height count disagree"
                );
                harvested.push(e);
            }

            let derivations = multilinear_continuation::decode_derivations();
            println!(
                "L0-HARVEST epochs {}  derivations {derivations}  total {:.3} s",
                harvested.len(),
                started.elapsed().as_secs_f64(),
            );

            assert!(
                !harvested.is_empty(),
                "the bundle carried no epochs, so nothing above was exercised"
            );
            assert_eq!(
                derivations,
                1,
                "DECODE's prepared commitment was derived {derivations} times over {} \
                 harvests; `prepared` is handed in once and must be reused",
                harvested.len()
            );

            // THE CHAIN, across every epoch of a real run.
            for (index, e) in harvested.iter().enumerate() {
                assert_eq!(
                    e.position.label,
                    epoch_label(index as u64),
                    "epoch {index} was labelled as some other epoch"
                );
                assert_eq!(
                    e.position.is_final,
                    index + 1 == harvested.len(),
                    "epoch {index}'s is_final is not its position"
                );
                if index > 0 {
                    assert_eq!(
                        e.position.register_init,
                        b.epochs[index - 1].reg_fini,
                        "epoch {index} did not start from epoch {}'s proved reg_fini",
                        index - 1
                    );
                }
            }

            assert_eq!(
                whir_process_posture_note().is_some(),
                crate::whir_hash_knob::selected() != crate::whir_hash_knob::Setting::Rpx,
                "the posture note disagreed with the process setting this bundle was \
                 proven and harvested under"
            );
        })
    }
    /// ★ THE AIR SET THE BUILDER WILL GET IS THE ONE THE VERIFIER ACCEPTED.
    ///
    /// V1's `whir_epoch_program` takes `EpochAirs<'_>` alongside the
    /// `WhirRealEpoch`, because `multi_verify`'s `TableStatement`s are built
    /// from the AIRs and a proof states heights, never widths.
    /// `epoch_airs_for` is what hands them over.
    ///
    /// ⚠ THIS IS NOT A COMPARISON OF TWO DERIVATIONS, AND THAT IS THE DESIGN.
    /// `verify_epoch_bookend` calls `epoch_airs_for` itself, so "the builder's
    /// AIRs are the verifier's AIRs" is true by construction and this test only
    /// has to check that what comes out describes the epoch it was asked for. A
    /// test that rebuilt the set independently and compared could pass with
    /// both halves wrong — the shape that let REGISTER's columns and its root
    /// describe different tables.
    ///
    /// The tamper is `is_final`, which decides whether HALT is in the set, so
    /// an inverted position yields a DIFFERENT NUMBER of AIRs. That is the
    /// check that the set is a function of the epoch's position rather than a
    /// constant the caller could have got anywhere.
    #[test]
    fn the_epoch_airs_describe_the_epoch_they_were_asked_for() {
        let (elf_bytes, opts, b) = bundle();
        let elf = Elf::load(&elf_bytes).expect("load");
        assert!(
            b.epochs.len() >= 2,
            "a one-epoch run has no non-final epoch"
        );

        let harvested = real_epoch_from_whir_continuation(&opts, &elf_bytes, &b, 0, None)
            .expect("epoch 0 harvests");
        let position = whir_epoch_chain_position(&b, &elf, 0).expect("epoch 0 has a position");

        let set = crate::multilinear_continuation::epoch_airs_for(
            &elf,
            &opts,
            &b.epochs[0],
            &position.register_init,
            position.is_final,
            position.label,
            Some(harvested.decode_commitment),
        );
        let refs = set.refs();

        // It reproduces the harvested shapes: the same count, and the same
        // WIDTH per table, which is the half a proof does not state and the
        // half a builder cannot guess.
        assert_eq!(
            refs.len(),
            harvested.shapes.len(),
            "the exposed AIR set and the harvested shapes disagree on the table count"
        );
        for (index, (air, &(width, _))) in refs.iter().zip(&harvested.shapes).enumerate() {
            assert_eq!(
                air.trace_layout().0,
                width,
                "table {index}'s width from the AIR set is not the harvested one"
            );
        }

        // THE TAMPER: the same epoch at the wrong position. `is_final` decides
        // whether HALT is in the set, so the count must move — if it does not,
        // this test cannot see a driver that ignored the position.
        let wrong = crate::multilinear_continuation::epoch_airs_for(
            &elf,
            &opts,
            &b.epochs[0],
            &position.register_init,
            !position.is_final,
            position.label,
            Some(harvested.decode_commitment),
        );
        assert_ne!(
            wrong.refs().len(),
            refs.len(),
            "inverting is_final left the AIR set identical, so the set does not \
             depend on the epoch's position and this test proves nothing"
        );

        // ★ AND THE SECOND TAMPER IS THE ONE THAT REACHES THE VERIFY. Restating
        // the epoch's `table_counts` changes how many AIRs the set has, so the
        // set no longer describes the proof in front of it — and because
        // `verify_epoch_bookend` builds ITS set through this same function, the
        // verification refuses rather than arguing against a layout the prover
        // never used. That is the by-construction claim reaching a verdict, not
        // just a count.
        let mut restated = b.clone();
        restated.epochs[0].table_counts.cpu += 1;
        let mismatched = crate::multilinear_continuation::epoch_airs_for(
            &elf,
            &opts,
            &restated.epochs[0],
            &position.register_init,
            position.is_final,
            position.label,
            Some(harvested.decode_commitment),
        );
        assert_ne!(
            mismatched.refs().len(),
            refs.len(),
            "a restated table count left the AIR set the same size, so the set is \
             not a function of the counts and the refusal below would prove nothing"
        );
        assert!(
            real_epoch_from_whir_continuation(&opts, &elf_bytes, &restated, 0, None).is_err(),
            "an epoch whose restated table counts no longer match its own AIR set \
             was harvested into a wrap input"
        );
    }
}
