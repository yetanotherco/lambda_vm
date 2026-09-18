//! Tests for the WHIR cross-epoch driver, which lives in
//! [`crate::lfm::whir_real_global`].
//!
//! They sit beside `whir_epoch_tests`, which does the same job for the level-0
//! driver, and they are deliberately about the values the harvest DERIVES
//! rather than the ones it copies: a field copied out of the bundle and
//! asserted against the bundle is one field read back through two names.

#[cfg(test)]
mod tests {
    use crate::lfm::proof_arena::lanes_per_root;
    use crate::lfm::whir_real_epoch::whir_process_posture_note;
    use crate::lfm::whir_real_global::real_global_from_whir_continuation;
    use crate::multilinear_continuation;
    use crate::tables::global_memory;
    use crate::tables::local_to_global;
    use crate::test_utils::asm_elf_bytes;
    use stark::proof::options::ProofOptions;

    /// A run whose epochs touch memory across the boundary — which is the only
    /// kind that has a cross-epoch proof worth harvesting.
    fn bundle() -> (
        Vec<u8>,
        ProofOptions,
        multilinear_continuation::ContinuationProof,
    ) {
        let mut input: Vec<u8> = Vec::with_capacity(16);
        input.extend_from_slice(&16u32.to_le_bytes());
        input.extend_from_slice(&[0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]);
        input.extend_from_slice(&[0u8; 4]);
        let elf_bytes = asm_elf_bytes("test_private_input_xpage");
        let opts = ProofOptions::default_test_options();
        let bundle = multilinear_continuation::prove_continuation(&elf_bytes, &input, 2, &opts)
            .expect("prove the continuation");
        (elf_bytes, opts, bundle)
    }

    /// ★ THE HARVEST, AND THE FOUR THINGS IT DERIVES RATHER THAN COPIES.
    ///
    /// The widths, because the proof states none; the group split, because it
    /// does not follow from the table count; the bookend root windows, because
    /// they are what the binding compares; and the published layout, because
    /// the cross-epoch wrap's set is a `GlobalLayout` and not a `SchemaLayout`.
    ///
    /// ⚠ WHAT THIS FIXTURE CANNOT SHOW. Its cross-epoch shape is 3 bookends and
    /// ONE page, so the page group is a singleton and is shape-identical to a
    /// bookend's: the sixteen-group split of a real block (fifteen singletons
    /// then one group of thirty-five) is NOT observable here, and the `sizes`
    /// assertion below pins a count rather than a split. Said here rather than
    /// left for a reader to assume otherwise.
    #[test]
    fn the_cross_epoch_proof_harvests_from_a_whir_continuation() {
        let (elf_bytes, opts, b) = bundle();
        if let Some(note) = whir_process_posture_note() {
            println!("{note}");
        }
        let g = real_global_from_whir_continuation(&opts, &elf_bytes, &b)
            .unwrap_or_else(|e| panic!("the cross-epoch proof does not harvest: {e}"));

        assert_eq!(g.num_epochs, b.epochs.len());
        assert_eq!(g.elf_digest, crate::statement::elf_digest(&elf_bytes));

        // ★ THE WIDTHS ARE THE AIRS' — the proof states heights only, so this
        // is the half a harvest cannot copy. The two families differ, which is
        // what makes their ORDER observable at all.
        assert_eq!(g.shapes.len(), g.num_tables());
        for (index, &(width, num_vars)) in g.shapes.iter().enumerate() {
            let expected = if index < g.num_epochs {
                local_to_global::cols::NUM_COLUMNS
            } else {
                global_memory::cols::NUM_COLUMNS
            };
            assert_eq!(
                width, expected,
                "table {index} is not the family the cross-epoch layout puts there"
            );
            assert_eq!(
                num_vars, b.global.table_num_vars[index] as usize,
                "table {index}'s height is not the one the proof states"
            );
        }

        // Every bookend alone, then the pages — the singletons are what let a
        // bookend's root be compared against the epoch that committed it.
        let pages = g.shapes.len() - g.num_epochs;
        let mut expected_sizes = vec![1usize; g.num_epochs];
        expected_sizes.push(pages);
        assert_eq!(
            g.sizes, expected_sizes,
            "the commitment split is not the one"
        );

        // ★ The windows are CONSECUTIVE and drawn from the proof's own root
        // list. ⚠ This does not independently re-derive how many polynomials
        // each bookend stacked into; what it can catch is a window that starts
        // in the wrong place or reads a root belonging to another group.
        assert_eq!(g.bookend_roots.len(), g.num_epochs);
        let mut start = 0usize;
        for (index, window) in g.bookend_roots.iter().enumerate() {
            assert!(
                !window.is_empty(),
                "epoch {index}'s bookend was committed in no polynomial at all"
            );
            assert_eq!(
                window.as_slice(),
                &b.global.proof.roots[start..start + window.len()],
                "epoch {index}'s bookend roots are not the window the proof carries there"
            );
            start += window.len();
        }
        assert!(
            start <= b.global.proof.roots.len(),
            "the bookend windows run past the roots block"
        );

        // The published set, as a type: `z`, `alpha`, then one root per epoch
        // at `lanes_per_root()` words. The indices are checked for what a
        // reader of the published vector needs — that no two land on one word.
        assert_eq!(g.published.num_epochs, g.num_epochs);
        assert_eq!(g.published.lanes_per_root, lanes_per_root());
        let total = g.published.total();
        assert_eq!(g.published.z_word(), 0);
        assert_eq!(g.published.alpha_word(), 1);
        let mut seen = std::collections::BTreeSet::new();
        for k in 0..g.num_epochs {
            for w in 0..lanes_per_root() {
                let word = g.published.l2g_word(k, w);
                assert!(word >= 2 && word < total, "l2g word {word} is out of range");
                assert!(seen.insert(word), "two roots share published word {word}");
            }
        }
        assert_eq!(
            seen.len() + 2,
            total,
            "the published set has unclaimed words"
        );

        // ⛔ The reserved field is EMPTY, and it stays empty until an INIT
        // opening exists. A harvest that filled it from
        // `recursion::precomputed_commitments`' per-page univariate roots would
        // be carrying objects no verifier on this path ever compares.
        assert!(g.prepared_roots.is_none());

        println!(
            "CROSS-EPOCH HARVEST: {} tables = {} bookends + {pages} pages, {} groups, \
             published {total} words ({} lanes per root)",
            g.num_tables(),
            g.num_epochs,
            g.sizes.len(),
            lanes_per_root(),
        );
    }

    /// ★ THE ACCEPTANCE CHECK IS REAL, and this is the arm that says so.
    ///
    /// The touched page set drives which cross-epoch tables exist, so a
    /// restated one must not harvest — and a driver that skipped its
    /// verification would hand back an input built from a proof nobody checked,
    /// pushing the failure into a guest an hour later.
    #[test]
    fn a_restated_page_set_does_not_harvest() {
        let (elf_bytes, opts, mut b) = bundle();
        assert!(
            !b.touched_page_bases.is_empty(),
            "the run touched no memory, so there is nothing to restate"
        );
        b.touched_page_bases.pop();
        let refused = real_global_from_whir_continuation(&opts, &elf_bytes, &b);
        let message = match refused {
            Ok(_) => panic!("a bundle with a restated page set was harvested"),
            Err(message) => message,
        };
        // The reason matters: this must be the verification refusing, not a
        // later arithmetic tripping over a short list.
        assert!(
            message.contains("does not verify") || message.contains("could not be verified"),
            "the refusal does not name the verification: {message}"
        );
    }

    /// A bundle with no epochs has no cross-epoch proof, and the driver says so
    /// rather than indexing into an empty list.
    #[test]
    fn a_bundle_with_no_epochs_is_refused_by_name() {
        let (elf_bytes, opts, mut b) = bundle();
        b.epochs.clear();
        let message = match real_global_from_whir_continuation(&opts, &elf_bytes, &b) {
            Ok(_) => panic!("a bundle with no epochs was harvested"),
            Err(message) => message,
        };
        assert!(
            message.contains("no epochs"),
            "the refusal does not name the empty bundle: {message}"
        );
    }
}
