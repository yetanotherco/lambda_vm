//! The PARENT of `k` global SLICES: verify each slice proof, check the three
//! things that make their partial bus sums summands of ONE equation, and
//! republish the L2G prefix the block-artifact root reads.
//!
//! # What it is for
//!
//! [`super::global_split`] splits the global wrap's table set across `k`
//! programs. Each slice replays the SAME statement and the SAME Phase A over all
//! main roots, verifies only its own tables at their true index within the true
//! `num_tables`, and publishes `(z, α)`, the full L2G root prefix, and its
//! **partial** bus sum instead of closing against zero. Nothing in a slice
//! asserts the bus balances — that assert belongs here, and it is the whole
//! reason this program exists.
//!
//! # The three checks, and why each one is load-bearing
//!
//! 1. **Every slice published the same `(z, α)`.** Not a tidiness check. Slices
//!    on different transcripts are not summands of one quantity at all, so a sum
//!    over them would be adding numbers that do not belong to the same equation
//!    and a zero would mean nothing. ⇒ `(z, α)` agreement is what makes the sum
//!    a SUM; check 2 is what makes it BALANCE. Neither substitutes for the
//!    other, and dropping either leaves a check that cannot fail in the way it
//!    was meant to. ⛔ Anyone tempted to drop it as redundant — *"they replay the
//!    same Phase A, so of course they agree"* — is using a property of the honest
//!    prover to carry a check against a dishonest one.
//! 2. **The partials sum to ZERO**, through [`super::logup::emit_bus_closure`],
//!    which takes its target as a parameter — the unsliced wrap passes zero over
//!    tables, this passes zero over slices.
//! 3. **Every slice published the same L2G ROOTS**, and this program
//!    REPUBLISHES them. ★★ This is what LICENSES the republished prefix. A slice
//!    publishes the roots of tables it did NOT walk: the prefix comes from the
//!    Phase A absorb, which is outside the slice loop, so slice 0 publishes the
//!    root of an L2G table whose legs only slice 1 verified — **on its own that
//!    is a claim slice 0 cannot back**. The partition guarantees every table's
//!    legs are walked by exactly one slice; agreement guarantees the copy
//!    republished from slice 0 is the same root the walking slice verified.
//!    ⛔ Without it a slice could publish a root for a table it never verified
//!    and nothing would compare it against the slice that did.
//!
//!    ⚠ And it is **not** subsumed by check 1, though it looks it. `(z, α)` is
//!    *derived from* the roots, so agreement there implies agreement here — but
//!    only by collision resistance of the transcript, which this program does not
//!    verify. It verifies WORDS.
//!
//! # ⛔ The BOUNDS are pinned by PROGRAM IDENTITY, not by published constants
//!
//! A slice's bounds are emit-time constants of the slice program, so they are
//! baked into its compiled form and therefore into its `program_id` — and
//! [`super::per_table_aggregator::emit_leg`] absorbs each child's `program_id` as
//! an emit-time constant of THIS program. ⇒ **A slice with different bounds is a
//! DIFFERENT PROGRAM and its proof is rejected on identity.**
//!
//! ★ Which composes with the single-source rule rather than fighting it: the
//! parent and the slices build their [`SlicePartition`] from the same
//! `(num_tables, k)`, so the `program_id`s this program embeds are exactly the
//! ones the slices compiled to. There is no second copy of a bound anywhere — not
//! in a published word, not in a constant, not in a comment. And a
//! `SlicePartition` that does not tile **cannot be constructed**, so no pair of
//! programs that mis-partition can be compiled in the first place; the gap and
//! overlap arms live with that constructor, which is the thing they guard.
//!
//! ⛔ **Do NOT add a runtime re-check of the tiling here.** The fields are
//! private to `global_split` and the constructor is the only way in, so an
//! `assert_tiles()` in this module would be a check that cannot fail — which is
//! worse than no check, because it produces evidence.
//!
//! # Why the shared binding pass cannot be used
//!
//! [`super::per_table_aggregator::emit_chain_bindings`] asserts an attestation
//! id, a register run and a label pair on EVERY child. A slice has none of them:
//! it publishes `(z, α)`, the L2G roots and one partial. Handing a slice to that
//! pass would index past its published words. So this module writes the parent's
//! own binding pass, exactly as
//! [`super::block_root::assert_global_child_is_bound_only_by_l2g`] says the root
//! does for the global child.

use crate::tables::types::FEE;

use super::block_root::SliceLayout;
use super::builder::{Ext, LfmBuilder};
use super::global_split::SlicePartition;
use super::per_table_aggregator::{
    ChildShape, HintedPublicWord, LegArenas, LegCells, assert_words_equal, declare_leg_arenas,
    emit_leg,
};

/// Everything the parent verifies and republishes.
pub struct ParentInputs<'a> {
    /// The `k` slice proofs, in SLICE ORDER — slice `i` is `partition.slice(i)`.
    ///
    /// ⚠ Order is an obligation. Each shape carries the `program_id` this program
    /// embeds as a constant, and a slice program's id encodes its bounds; a
    /// permuted list would embed slice 1's id where slice 0's belongs and reject
    /// an honest prover.
    pub slices: &'a [ChildShape<'a>],
    /// The partition the SLICES were emitted against — the same object, not a
    /// second one built from the same numbers.
    pub partition: &'a SlicePartition,
    /// What each slice published: the wrap's set, then the partial.
    pub layout: &'a SliceLayout,
}

/// Emit the parent: verify every slice, run the three checks, republish the
/// prefix.
pub fn emit_global_parent(b: &mut LfmBuilder, inputs: &ParentInputs<'_>) {
    let ParentInputs {
        slices,
        partition,
        layout,
    } = *inputs;
    assert!(
        partition.k() >= 2,
        "a parent folds at least TWO slices: at k = 1 the global wrap closes its \
         own bus against zero and publishes no partial, so there is nothing to \
         sum and no agreement to check"
    );
    assert_eq!(
        slices.len(),
        partition.k(),
        "the partition splits the global proof into {} slices but {} slice proofs \
         were given; a parent over a different count would pin a different \
         partition than the one the slices were emitted against",
        partition.k(),
        slices.len(),
    );
    for (i, slice) in slices.iter().enumerate() {
        assert_eq!(
            slice.num_public_words,
            layout.total(),
            "slice {i} publishes {} words but the SLICE layout describes {} \
             ({} for the wrap's set, then the partial at index {}). Every index \
             this program reads — the shared pair, each L2G lane, the partial — \
             is shifted by a wrong layout, so it aborts here rather than compare \
             the wrong words",
            slice.num_public_words,
            layout.total(),
            layout.shared.total(),
            layout.partial_sum_word(),
        );
    }

    // ⚠ DECLARATION ORDER IS ABSORB ORDER. Every slice's arenas are declared
    // before any leg is emitted, exactly as `emit_node` does it, so the host
    // fills them as a plain per-child concatenation.
    let arenas: Vec<LegArenas> = slices.iter().map(|s| declare_leg_arenas(b, s)).collect();
    let legs: Vec<LegCells> = slices
        .iter()
        .zip(&arenas)
        .map(|(slice, a)| emit_leg(b, slice, a))
        .collect();

    emit_parent_checks_and_publishes(b, &legs, layout);
}

/// The parent's whole contribution over verified slice legs: the three checks,
/// then the republished prefix.
///
/// ⛔ Split out from [`emit_global_parent`] so a gate can drive it over a FIXTURE
/// of published words rather than over `k` real slice proofs — the packing of the
/// republished roots has to be verified by reading words back at each index, and
/// a gate that needed two proofs to do it would not be a gate anybody runs.
/// ⇒ It is the SAME code path, not a second spelling of it.
fn emit_parent_checks_and_publishes(b: &mut LfmBuilder, legs: &[LegCells], layout: &SliceLayout) {
    assert!(!legs.is_empty(), "a parent folds at least one slice leg");
    assert_every_slice_published_the_same_pair(b, legs, layout);
    assert_the_partials_sum_to_zero(b, legs, layout);
    assert_every_slice_published_the_same_roots(b, legs, layout);
    emit_parent_publishes(b, legs, layout);
}

/// CHECK 1 — every slice derived the same `(z, α)`.
///
/// Compared as WORDS, lane by lane, against slice 0. Not as extension elements:
/// a word comparison also pins the fourth lane, and the published pair is the
/// only evidence this program has that the `k` proofs were made under one
/// transcript.
fn assert_every_slice_published_the_same_pair(
    b: &mut LfmBuilder,
    legs: &[LegCells],
    layout: &SliceLayout,
) {
    let g = &layout.shared;
    for k in 1..legs.len() {
        for w in [g.z_word(), g.alpha_word()] {
            assert_words_equal(b, &legs[0].publics[w], &legs[k].publics[w]);
        }
    }
}

/// CHECK 2 — the partials sum to zero.
///
/// ★ Through [`super::logup::emit_bus_closure`] rather than a hand-written fold,
/// because that function already IS "sum a fixed-size list of contributions and
/// assert the total equals a target": the unsliced wrap passes zero over its
/// tables' contributions, and this passes zero over the slices' partials. Its
/// `num_contributing_tables` assert then pins the SLICE COUNT as program shape,
/// which is the property that matters — a parent that summed a number of
/// partials read off the proof would let the prover choose how many there were.
fn assert_the_partials_sum_to_zero(b: &mut LfmBuilder, legs: &[LegCells], layout: &SliceLayout) {
    let at = layout.partial_sum_word();
    let partials: Vec<Ext> = legs
        .iter()
        .map(|leg| published_ext(b, &leg.publics[at]))
        .collect();
    let shape = super::logup::LogUpShape {
        num_contributing_tables: legs.len(),
        num_output_bytes: 0,
    };
    let zero = b.ext_const(&FEE::zero());
    super::logup::emit_bus_closure(b, &shape, &partials, zero);
}

/// CHECK 3 — every slice published the same L2G roots.
///
/// See the module doc: this is what licenses republishing a prefix in which every
/// slice claims roots for tables it did not all walk. Compared against slice 0,
/// which is also the copy [`emit_parent_publishes`] republishes — so the words
/// compared and the words republished are literally the same cells.
fn assert_every_slice_published_the_same_roots(
    b: &mut LfmBuilder,
    legs: &[LegCells],
    layout: &SliceLayout,
) {
    let g = &layout.shared;
    for epoch in 0..g.num_epochs {
        for lane in 0..g.lanes_per_root {
            let w = g.l2g_word(epoch, lane);
            for k in 1..legs.len() {
                assert_words_equal(b, &legs[0].publics[w], &legs[k].publics[w]);
            }
        }
    }
}

/// What the parent publishes: `(z, α)` and the L2G roots, from slice 0, and
/// NOTHING for the sum — it asserted zero rather than publishing a partial.
///
/// ⛔ **THE WIDTH IS A COINCIDENCE OF ARITHMETIC, NOT A DESIGN.** That set is
/// `2 + num_epochs × lanes_per_root`, which is exactly
/// [`super::block_root::GlobalLayout::total`] — the same words the UNSLICED wrap
/// published. The type survives while its meaning changes completely: it stops
/// describing *what the global wrap published* and starts describing *what the
/// parent republished*. That is why `RootInputs`' field is named for the global
/// CHILD rather than for the wrap.
///
/// ⇒ ⛔ **And the packing is verified BY TEST, not by reading.**
/// [`super::block_root::emit_l2g_compare`] reads its global child's roots at
/// `2 + epoch × lanes + lane`. A parent that republished in a different order or
/// packing would still satisfy the length assert and the compare would read the
/// right COUNT of wrong words — silent, and downstream. See
/// [`tests::the_parent_republishes_every_root_at_the_index_the_root_reads`].
fn emit_parent_publishes(b: &mut LfmBuilder, legs: &[LegCells], layout: &SliceLayout) {
    let g = &layout.shared;
    let first = &legs[0];
    // ---- the shared pair, as the four-lane word the slice published. `z` and
    // `alpha` are EXTENSION values (three nonzero lanes), so they are repacked
    // whole rather than read as a base lane.
    for w in [g.z_word(), g.alpha_word()] {
        let lanes = &first.publics[w].lanes;
        let word = b.pack_word([lanes[0], lanes[1], lanes[2], lanes[3]]);
        b.public(word);
    }
    // ---- the L2G prefix, lane by lane, in the SAME shape a slice published it:
    // one BASE word per lane. The root reads `lanes[0]` of each of these words
    // (`GlobalLayout::epoch_digests`), which is right for a lane-per-word layout
    // and silently wrong for anything else.
    for epoch in 0..g.num_epochs {
        for lane in 0..g.lanes_per_root {
            let cell = first.publics[g.l2g_word(epoch, lane)].lanes[0].as_cell();
            b.public(cell);
        }
    }
}

/// One published word, read back as the extension element it was published as.
///
/// ⚠ `pack_word` then `as_ext`, NOT `pack_ext` over lanes 0..3. An extension
/// publish is `(a0, a1, a2, 0)`, and the executor's `read_ext` REFUSES a word
/// whose fourth lane is nonzero ([`super::word::word_as_ext`]). Packing all four
/// lanes therefore keeps that lane as a CHECK; dropping it would silently accept
/// a word that is not an extension element at all.
fn published_ext(b: &mut LfmBuilder, w: &HintedPublicWord) -> Ext {
    b.pack_word([w.lanes[0], w.lanes[1], w.lanes[2], w.lanes[3]])
        .as_ext()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::types::FE;
    use crate::tables::types::GoldilocksField;
    use math::field::traits::IsPrimeField;

    use super::super::block_root::GlobalLayout;
    use super::super::executor::{LfmExecError, LfmExecution, execute};
    use super::super::per_table_aggregator::hint_public_words;
    use super::super::word::{LfmWord, base_word, ext_word, word_as_base, word_as_ext};

    /// A root lane's fixture value, DISTINGUISHABLE at every `(epoch, lane)`.
    ///
    /// ⛔ The whole point of the packing gate: `epoch × lanes + lane` is the flat
    /// index, so this is injective and a transposition, an off-by-one or a
    /// re-grouping lands on a value that belongs somewhere else. A fixture of
    /// equal roots would pass under every one of those.
    fn root_lane(layout: &SliceLayout, epoch: usize, lane: usize) -> FE {
        FE::from(1 + (epoch * layout.shared.lanes_per_root + lane) as u64 * 1_000_003)
    }

    /// The `k` slices' published words: agreeing on `(z, α)` and on every root,
    /// with partials that sum to zero.
    ///
    /// ⛔ **WRITTEN POSITIONALLY, IN `global_slice_program`'s PUBLISH ORDER, AND
    /// NOT THROUGH THE LAYOUT — do not "tidy" this into indexed writes.** The
    /// parent READS through [`SliceLayout`]; a fixture that WROTE through it too
    /// would cancel any drift between the layout and the order a slice actually
    /// publishes in, and every arm below would pass while the parent read the
    /// wrong words off a real slice. Appending in the emitter's own order is the
    /// independent statement that makes the layout-indexed reads a check.
    fn honest_slice_publics(k: usize, layout: &SliceLayout) -> Vec<Vec<LfmWord>> {
        let g = &layout.shared;
        let z = FEE::new([FE::from(7), FE::from(8), FE::from(9)]);
        let alpha = FEE::new([FE::from(11), FE::from(12), FE::from(13)]);
        let mut partials: Vec<FEE> = (0..k - 1)
            .map(|i| FEE::from(977 * (i as u64 + 1)))
            .collect();
        let sum = partials.iter().fold(FEE::zero(), |acc, p| acc + *p);
        partials.push(FEE::zero() - sum);

        (0..k)
            .map(|i| {
                let mut words = Vec::with_capacity(layout.total());
                words.push(ext_word(&z));
                words.push(ext_word(&alpha));
                for epoch in 0..g.num_epochs {
                    for lane in 0..g.lanes_per_root {
                        words.push(base_word(root_lane(layout, epoch, lane)));
                    }
                }
                words.push(ext_word(&partials[i]));
                assert_eq!(words.len(), layout.total(), "the fixture IS the layout");
                words
            })
            .collect()
    }

    /// One slice's published words as the eight-halves-per-word arena
    /// `hint_public_words` reads — the serializer's own layout.
    fn publics_arena(words: &[LfmWord]) -> Vec<LfmWord> {
        let mut out = Vec::with_capacity(8 * words.len());
        for w in words {
            for lane in w {
                let v: u64 = GoldilocksField::canonical(lane.value());
                out.push(base_word(FE::from(v & 0xFFFF_FFFF)));
                out.push(base_word(FE::from(v >> 32)));
            }
        }
        out
    }

    /// Emit the parent's checks and republish over a FIXTURE of `k` slices'
    /// published words, then execute it.
    ///
    /// ⚠ The legs are hinted publics, not verified proofs: this drives
    /// [`emit_parent_checks_and_publishes`] — the parent's own contribution —
    /// and deliberately not `emit_leg`, which is already gated everywhere it is
    /// used. What is under test is what the parent DOES with words a slice
    /// published, and that is exactly what a fixture can hold.
    fn run_fixture(
        k: usize,
        num_epochs: usize,
        mutate: impl FnOnce(&mut Vec<Vec<LfmWord>>),
    ) -> (SliceLayout, Result<LfmExecution, LfmExecError>) {
        let layout = SliceLayout::over(GlobalLayout {
            num_epochs,
            lanes_per_root: super::super::proof_arena::lanes_per_root(),
        });
        let mut publics = honest_slice_publics(k, &layout);
        mutate(&mut publics);

        let mut b = LfmBuilder::new().with_wrap_hash(super::super::edsl::WrapHash::production());
        let ids: Vec<_> = (0..k)
            .map(|_| b.declare_arena((8 * layout.total()) as u32))
            .collect();
        let legs: Vec<LegCells> = ids
            .iter()
            .enumerate()
            .map(|(i, id)| {
                let publics = hint_public_words(&mut b, *id, layout.total());
                // ⛔ DISTINCT PER SLICE, ON PURPOSE. `LegCells::z_alpha` is the
                // PARENT's own per-child LFM pair — the challenges it derived to
                // verify that slice's proof — and it is NOT the global proof's
                // `(z, α)`, which the slice PUBLISHES. The parent must compare the
                // published words; if it ever compared `z_alpha` instead, these
                // deliberately unequal dummies make the HONEST arm fail rather
                // than let a fixture of equal values hide the substitution.
                let dummy = b.ext_const(&FEE::from(1_000 + i as u64));
                LegCells {
                    publics,
                    z_alpha: (dummy, dummy),
                }
            })
            .collect();
        emit_parent_checks_and_publishes(&mut b, &legs, &layout);
        let program = super::super::compiler::compile(b.finish());
        let arenas: Vec<Vec<LfmWord>> = publics.iter().map(|w| publics_arena(w)).collect();
        let exec = execute(&program, &arenas, &crate::hash_pin::BLOCK_HASHER);
        (layout, exec)
    }

    /// ★★ THE PACKING GATE — read back at EVERY index, not counted.
    ///
    /// `block_root::emit_l2g_compare` reads its global child's roots at
    /// `2 + epoch × lanes + lane` and refolds them against the interior's digest.
    /// If the parent republished in a different order or packing the LENGTH
    /// assert still passes and the compare reads the right COUNT of wrong
    /// words — silent, and downstream, where the failure is either an honest
    /// prover failing or a compare against something an adversary can steer.
    ///
    /// ⇒ Distinguishable roots, republished by the parent, read back through
    /// `GlobalLayout::l2g_word` — the accessor the root itself reads with, so
    /// there is ONE index derivation here rather than two compared against each
    /// other.
    #[test]
    fn the_parent_republishes_every_root_at_the_index_the_root_reads() {
        for k in [2usize, 3] {
            for num_epochs in [1usize, 2, 19] {
                let (layout, exec) = run_fixture(k, num_epochs, |_| {});
                let exec = exec.unwrap_or_else(|e| {
                    panic!("k={k}, {num_epochs} epochs: the honest parent must execute: {e:?}")
                });
                let g = &layout.shared;
                assert_eq!(
                    exec.public_words.len(),
                    g.total(),
                    "k={k}: the parent publishes the wrap's set and NOTHING for \
                     the sum — it asserted zero rather than publishing a partial"
                );
                let z = word_as_ext(&exec.public_words[g.z_word()].1).expect("z is an ext word");
                let alpha =
                    word_as_ext(&exec.public_words[g.alpha_word()].1).expect("alpha is an ext");
                assert_eq!(z, FEE::new([FE::from(7), FE::from(8), FE::from(9)]), "z");
                assert_eq!(
                    alpha,
                    FEE::new([FE::from(11), FE::from(12), FE::from(13)]),
                    "alpha"
                );
                for epoch in 0..num_epochs {
                    for lane in 0..g.lanes_per_root {
                        let got = word_as_base(&exec.public_words[g.l2g_word(epoch, lane)].1)
                            .unwrap_or_else(|| {
                                panic!(
                                    "epoch {epoch} lane {lane} was republished as something \
                                         that is not a BASE word; the root reads lanes[0] of it"
                                )
                            });
                        assert_eq!(
                            got,
                            root_lane(&layout, epoch, lane),
                            "k={k}, {num_epochs} epochs: epoch {epoch} lane {lane} was \
                             republished at index {} carrying another lane's value — the \
                             root's compare would refold the wrong roots and the length \
                             assert would not notice",
                            g.l2g_word(epoch, lane),
                        );
                    }
                }
            }
        }
    }

    /// ⛔ Slices that disagree on `(z, α)` are rejected.
    ///
    /// Without this the partials are not summands of one quantity and the
    /// zero-assert would be adding numbers from different equations.
    #[test]
    fn slices_that_disagree_on_z_alpha_are_rejected() {
        for w in [0usize, 1] {
            let (_, honest) = run_fixture(2, 3, |_| {});
            assert!(
                honest.is_ok(),
                "the honest control must execute, or the arm below proves nothing"
            );
            let (_, tampered) = run_fixture(2, 3, |p| {
                // One lane of slice 1's pair, moved. It still sums to zero and
                // every root still agrees; only the transcript claim differs.
                p[1][w][1] += FE::one();
            });
            assert!(
                tampered.is_err(),
                "slice 1 published a different {} and the parent accepted it",
                if w == 0 { "z" } else { "alpha" }
            );
        }
    }

    /// ⛔ Slices that disagree on an L2G ROOT are rejected.
    ///
    /// The check that licenses republishing a prefix each slice claims for tables
    /// it did not all walk. ⚠ Not subsumed by the `(z, α)` arm: `(z, α)` is
    /// derived FROM the roots, but only by collision resistance of the
    /// transcript, and this program does not verify that implication — which is
    /// why the fixture below leaves the pair agreeing and moves a root alone.
    #[test]
    fn slices_that_disagree_on_an_l2g_root_are_rejected() {
        let epochs = 3usize;
        let lanes = super::super::proof_arena::lanes_per_root();
        for epoch in 0..epochs {
            for lane in 0..lanes {
                let (layout, honest) = run_fixture(2, epochs, |_| {});
                assert!(honest.is_ok(), "the honest control must execute");
                let at = layout.shared.l2g_word(epoch, lane);
                let (_, tampered) = run_fixture(2, epochs, |p| {
                    p[1][at][0] += FE::one();
                });
                assert!(
                    tampered.is_err(),
                    "slice 1 published a different root at epoch {epoch} lane {lane} \
                     (word {at}) and the parent republished slice 0's copy anyway"
                );
            }
        }
    }

    /// ⛔ Partials that do not sum to zero are rejected — the bus check itself.
    #[test]
    fn slices_whose_partials_do_not_sum_to_zero_are_rejected() {
        for k in [2usize, 3] {
            let (_, honest) = run_fixture(k, 2, |_| {});
            assert!(honest.is_ok(), "the honest control must execute");
            for i in 0..k {
                let (layout, tampered) = run_fixture(k, 2, |p| {
                    let at = p[i].len() - 1;
                    p[i][at][0] += FE::one();
                });
                assert_eq!(
                    layout.partial_sum_word(),
                    layout.total() - 1,
                    "the fixture moved the LAST word, which must be the partial"
                );
                assert!(
                    tampered.is_err(),
                    "k={k}: slice {i}'s partial was moved and the bus still closed"
                );
            }
        }
    }

    /// ⛔ A partial that is not an EXTENSION word is rejected rather than
    /// truncated.
    ///
    /// The parent repacks all four lanes and reads the word as an extension
    /// element, so a fourth lane the slice never published is a refusal. A
    /// `pack_ext` over lanes 0..2 would drop it silently — and then the value the
    /// parent summed would not be the word the slice's statement absorbed.
    #[test]
    fn a_partial_whose_fourth_lane_is_nonzero_is_rejected() {
        let (_, honest) = run_fixture(2, 2, |_| {});
        assert!(honest.is_ok(), "the honest control must execute");
        let (_, tampered) = run_fixture(2, 2, |p| {
            let at = p[0].len() - 1;
            p[0][at][3] += FE::one();
        });
        assert!(
            tampered.is_err(),
            "a partial with a nonzero fourth lane is not an extension element and \
             must not be read as one"
        );
    }

    /// ⛔ A parent needs at least TWO slices, and exactly as many proofs as the
    /// partition has slices.
    #[test]
    #[should_panic(expected = "a parent folds at least TWO slices")]
    fn a_parent_over_a_single_slice_is_refused() {
        let mut b = LfmBuilder::new().with_wrap_hash(super::super::edsl::WrapHash::production());
        emit_global_parent(
            &mut b,
            &ParentInputs {
                slices: &[],
                partition: &SlicePartition::even(41, 1),
                layout: &SliceLayout::over(GlobalLayout {
                    num_epochs: 19,
                    lanes_per_root: super::super::proof_arena::lanes_per_root(),
                }),
            },
        );
    }
}
