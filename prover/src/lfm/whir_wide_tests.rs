//! Gates for the wide level-1 node (`whir_wide`).
//!
//! Every test here needs REAL epochs, so they run at the continuation fixture's
//! scale (`whir_epoch_program_tests::driver_bundle`: three epochs of
//! `test_private_input_xpage`, re-proved under a literal `RpxWhir`). They build
//! traces: box tier, never the laptop.
//!
//! The negative tests come in PAIRS. Each emits one broken wide node twice: with
//! the node's bindings, and with them left out (`WideTamper::skip_bindings`).
//! The first must be refused and the second must execute, which shows the
//! bindings are what refuse it and that every epoch leg still verifies on its
//! own. This session cannot run a deletion mutation, so the control stands in
//! for it.

use std::panic::AssertUnwindSafe;

use stark::traits::AIR;

use crate::lfm::whir_real_epoch::WhirRealEpoch;
use crate::multilinear_continuation::WhirEpochAirs;
use crate::tables::types::{FE, GoldilocksExtension, GoldilocksField};

use super::builder::LfmBuilder;
use super::compiler::compile;
use super::executor::execute;
use super::per_table_aggregator::{NodePublishSet, SchemaLayout, fold_l2g};
use super::whir_wide::{WideEpoch, WideTamper, wide_node_arena, wide_program};
use super::word::{LfmWord, base_word};

type AirRef<'a> =
    &'a dyn AIR<Field = GoldilocksField, FieldExtension = GoldilocksExtension, PublicInputs = ()>;

/// Every epoch of the fixture, harvested under RPX, with the AIR set the host's
/// verifier derives for it.
fn fixture_epochs() -> Vec<(WhirRealEpoch, WhirEpochAirs)> {
    let (elf_bytes, opts, bundle) = super::whir_epoch_program_tests::driver_bundle();
    let elf = executor::elf::Elf::load(&elf_bytes).expect("the inner ELF loads");
    (0..bundle.epochs.len())
        .map(|k| {
            let epoch = crate::lfm::whir_real_epoch::real_epoch_from_whir_continuation_under::<
                multilinear::whir_hash::RpxWhir,
            >(&opts, &elf_bytes, &bundle, k, None, None)
            .unwrap_or_else(|why| panic!("epoch {k} must harvest: {why}"));
            let airs = crate::multilinear_continuation::epoch_airs_for(
                &elf,
                &opts,
                &bundle.epochs[k],
                &epoch.position.register_init,
                epoch.position.is_final,
                epoch.position.label,
                Some(epoch.decode_commitment),
            );
            (epoch, airs)
        })
        .collect()
}

/// The tree position of epoch `k`, as the drivers assign it.
fn position(k: usize) -> u64 {
    crate::tables::local_to_global::epoch_label(k as u64)
}

/// The wide epochs `ks` of `held`, over the AIR references `refs` holds.
fn wide_of<'a>(
    held: &'a [(WhirRealEpoch, WhirEpochAirs)],
    refs: &'a [Vec<AirRef<'a>>],
    ks: &[usize],
) -> Vec<WideEpoch<'a>> {
    ks.iter()
        .map(|&k| WideEpoch {
            epoch: &held[k].0,
            airs: &refs[k][..],
        })
        .collect()
}

/// Emit, compile and execute a wide node over `ks` pinned at `labels`. `Ok`
/// carries the published words; `Err` says where it was refused.
fn run_wide(
    held: &[(WhirRealEpoch, WhirEpochAirs)],
    ks: &[usize],
    labels: &[u64],
    tamper: &WideTamper,
) -> Result<Vec<(u32, LfmWord)>, String> {
    let refs: Vec<Vec<AirRef<'_>>> = held.iter().map(|(_, airs)| airs.refs()).collect();
    let wide = wide_of(held, &refs, ks);
    let emitted = std::panic::catch_unwind(AssertUnwindSafe(|| {
        wide_program(&wide, labels, NodePublishSet::Aggregation, tamper)
    }));
    let program = emitted.map_err(|_| "refused at emission".to_string())?;
    let arenas = wide_node_arena(&wide);
    assert_eq!(
        program.arena_schema.lens,
        arenas.iter().map(|a| a.len() as u32).collect::<Vec<_>>(),
        "one arena per epoch, of exactly the words its filler writes, in chain order"
    );
    execute(&program, &arenas, &crate::hash_pin::BLOCK_HASHER)
        .map(|exec| exec.public_words)
        .map_err(|why| format!("refused at execution: {why:?}"))
}

fn word_at(words: &[(u32, LfmWord)], i: usize) -> LfmWord {
    words
        .iter()
        .find(|(at, _)| *at as usize == i)
        .unwrap_or_else(|| panic!("no word published at index {i}"))
        .1
}

/// The L2G digest an L1 node over these epochs' wraps publishes: `fold_l2g` over
/// each epoch's bookend root, in chain order, as four published lanes. Taken
/// from the host's copy of the roots, not from the wide node's wires.
fn expected_l2g_lanes(held: &[(WhirRealEpoch, WhirEpochAirs)], ks: &[usize]) -> Vec<LfmWord> {
    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let digests: Vec<super::edsl::WrapDigest> = ks
        .iter()
        .map(|&k| {
            let roots = held[k]
                .0
                .proof
                .l2g_roots(1)
                .expect("the bookend is one polynomial");
            let word = super::algebraic_commit::commitment_to_digest(&roots[0]);
            super::edsl::WrapDigest::from_cell(b.digest_const(word).as_cell())
        })
        .collect();
    let folded = fold_l2g(&mut b, &digests);
    for cell in folded.cells().to_vec() {
        for lane in b.unpack(cell) {
            b.public(lane.as_cell());
        }
    }
    let program = compile(b.finish());
    execute(&program, &[], &crate::hash_pin::BLOCK_HASHER)
        .expect("the fold of constants executes")
        .public_words
        .into_iter()
        .map(|(_, word)| word)
        .collect()
}

/// ★★ THE HONEST WIDE NODE: the machine executes every epoch the host accepted
/// in one program, and publishes what an L1 node over those epochs' wraps
/// publishes — the shared id, the FIRST epoch's INIT, the LAST epoch's FINI,
/// the position range, the last epoch's output and the folded L2G digest.
#[test]
fn a_wide_node_executes_its_epochs_and_publishes_the_node_schema() {
    let held = fixture_epochs();
    assert!(held.len() >= 3, "the fixture has three epochs");
    let ks = [0usize, 1, 2];
    let labels: Vec<u64> = ks.iter().map(|&k| position(k)).collect();
    let words = run_wide(&held, &ks, &labels, &WideTamper::default())
        .unwrap_or_else(|why| panic!("the honest wide node must execute: {why}"));

    let first = &held[ks[0]].0;
    let last = &held[ks[ks.len() - 1]].0;
    let layout = SchemaLayout::node(last.public_output().len().div_ceil(4));
    layout.assert_covers(words.len());
    // ⚠ ANTI-VACUITY: the chain must MOVE across the node, or "first INIT, last
    // FINI" could not tell the ends apart.
    assert_ne!(
        first.position.register_init, last.proof.reg_fini,
        "the fixture's first INIT and last FINI must differ for the ends to be checked"
    );

    let id = super::whir_epoch::byte_halves(&super::whir_epoch::epoch_program_id(first));
    for half in 0..2 {
        let want: LfmWord = [
            id[4 * half],
            id[4 * half + 1],
            id[4 * half + 2],
            id[4 * half + 3],
        ];
        assert_eq!(word_at(&words, layout.id(half)), want, "id word {half}");
    }
    for (r, value) in first.position.register_init.iter().enumerate() {
        assert_eq!(
            word_at(&words, layout.reg_init(r)),
            base_word(FE::from(u64::from(*value))),
            "INIT[{r}] must be the FIRST epoch's"
        );
    }
    for (r, value) in last.proof.reg_fini.iter().enumerate() {
        assert_eq!(
            word_at(&words, layout.reg_fini(r)),
            base_word(FE::from(u64::from(*value))),
            "FINI[{r}] must be the LAST epoch's"
        );
    }
    let ends = [labels[0], labels[labels.len() - 1]];
    for (i, label) in ends.iter().enumerate() {
        assert_eq!(
            word_at(&words, layout.label(2 * i)),
            base_word(FE::from(label & 0xFFFF_FFFF)),
            "label range end {i}, low half"
        );
        assert_eq!(
            word_at(&words, layout.label(2 * i + 1)),
            base_word(FE::from(label >> 32)),
            "label range end {i}, high half"
        );
    }
    for (i, half) in super::whir_epoch::byte_halves(last.public_output())
        .into_iter()
        .enumerate()
    {
        assert_eq!(
            word_at(&words, layout.out_half(i)),
            base_word(half),
            "output half {i} must be the LAST epoch's"
        );
    }
    let l2g = expected_l2g_lanes(&held, &ks);
    assert_eq!(l2g.len(), layout.l2g_words, "one published word per lane");
    for (w, want) in l2g.iter().enumerate() {
        assert_eq!(
            word_at(&words, layout.l2g_word(w)),
            *want,
            "L2G lane {w} must be the fold of the epochs' bookend roots"
        );
    }
}

/// A wide node over ONE epoch — the tree's leftover at an odd count — publishes
/// the node schema over that epoch alone.
#[test]
fn a_wide_node_of_one_epoch_publishes_the_node_schema() {
    let held = fixture_epochs();
    let k = held.len() - 1;
    let words = run_wide(&held, &[k], &[position(k)], &WideTamper::default())
        .unwrap_or_else(|why| panic!("a one-epoch wide node must execute: {why}"));
    let epoch = &held[k].0;
    let layout = SchemaLayout::node(epoch.public_output().len().div_ceil(4));
    layout.assert_covers(words.len());
    assert_eq!(
        word_at(&words, layout.reg_init(0)),
        base_word(FE::from(u64::from(epoch.position.register_init[0]))),
    );
    assert_eq!(
        word_at(&words, layout.label(0)),
        word_at(&words, layout.label(2)),
        "a one-epoch node's range starts and ends at its epoch"
    );
}

/// ★ THE REGISTER CHAIN: a wide node over epochs 0 and 2 — each a valid epoch,
/// each pinned to its own position — breaks FINI(0) = INIT(2), and is refused.
/// Its control, the same node without the bindings, executes.
#[test]
fn a_wide_node_over_a_broken_register_chain_is_refused() {
    let held = fixture_epochs();
    assert!(held.len() >= 3, "the fixture has three epochs");
    assert_ne!(
        held[0].0.proof.reg_fini, held[2].0.position.register_init,
        "ANTI-VACUITY: epoch 0's FINI must differ from epoch 2's INIT"
    );
    let ks = [0usize, 2];
    let labels = [position(0), position(2)];
    let refused = run_wide(&held, &ks, &labels, &WideTamper::default());
    assert!(
        refused.is_err(),
        "a wide node over a broken register chain must be refused"
    );
    println!("broken chain: {}", refused.unwrap_err());
    let control = WideTamper {
        skip_bindings: true,
        ..WideTamper::default()
    };
    run_wide(&held, &ks, &labels, &control).unwrap_or_else(|why| {
        panic!(
            "CONTROL: without the bindings the same node must execute, or the refusal \
             above is not the chain's: {why}"
        )
    });
}

/// ★ THE POSITIONS: a wide node over epochs 0 and 1 pinned to each other's
/// positions is refused. Its control, without the bindings, executes.
#[test]
fn a_wide_node_pinned_to_the_wrong_positions_is_refused() {
    let held = fixture_epochs();
    let ks = [0usize, 1];
    let swapped = [position(1), position(0)];
    let refused = run_wide(&held, &ks, &swapped, &WideTamper::default());
    assert!(
        refused.is_err(),
        "a wide node whose epochs carry other positions' labels must be refused"
    );
    println!("swapped positions: {}", refused.unwrap_err());
    let control = WideTamper {
        skip_bindings: true,
        ..WideTamper::default()
    };
    run_wide(&held, &ks, &swapped, &control).unwrap_or_else(|why| {
        panic!("CONTROL: without the bindings the same node must execute: {why}")
    });
}

/// ★ THE ATTESTATION ID: a wide node whose second epoch publishes another
/// program's id is refused. Its control, without the bindings, executes.
#[test]
fn a_wide_node_whose_epochs_disagree_on_the_id_is_refused() {
    let held = fixture_epochs();
    let ks = [0usize, 1];
    let labels = [position(0), position(1)];
    let other = [0x5au8; 32];
    assert_ne!(
        super::whir_epoch::epoch_program_id(&held[1].0),
        other,
        "ANTI-VACUITY: the substituted id must differ"
    );
    let tampered = WideTamper {
        id: Some((1, other)),
        ..WideTamper::default()
    };
    let refused = run_wide(&held, &ks, &labels, &tampered);
    assert!(
        refused.is_err(),
        "a wide node whose epochs disagree on the attestation id must be refused"
    );
    println!("disagreeing id: {}", refused.unwrap_err());
    let control = WideTamper {
        skip_bindings: true,
        ..tampered
    };
    run_wide(&held, &ks, &labels, &control).unwrap_or_else(|why| {
        panic!("CONTROL: without the bindings the same node must execute: {why}")
    });
}
