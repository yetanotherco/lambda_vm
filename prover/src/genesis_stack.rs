//! Which cross-epoch genesis pages are carried by a PREPARED OPENING and which
//! by the sparse closed form — the routing half of the INIT hybrid.
//!
//! # Why there is a hybrid at all
//!
//! A cross-epoch GLOBAL_MEMORY page presents two preprocessed columns,
//! `[OFFSET, INIT]` ([`crate::tables::page::preprocessed_columns`]). INIT is the
//! page's genesis bytes, zero past `init_values.len()`, and the in-guest
//! verifier discharges it by the sparse form — one term per NONZERO entry, so
//! [`sparse_leg_rows`] rows. That is free for a zero-init page and cheap for a
//! page holding a few constants.
//!
//! It is not cheap for a page that is genuinely full. Measured on block
//! 25368371 (ELF sha256
//! `8f826601776d4085cbb6fbf0302fe8d8d5d1be7940ac1aaca24899c6244ec80a`,
//! 3,948,504 B; input `573004e62e3680a00d3cdbae19dc4897e2ec60d6ec0c1d05d9ef118cb8aef17f`,
//! 1,110,183 B; `LAMBDA_VM_MAX_ROWS_LOG2=21`, epoch 2^21), the run touches 35
//! pages — 30 genesis and 5 private — and the genesis bytes are CONCENTRATED:
//!
//! | page | nonzero of 262,144 | [`sparse_leg_rows`] |
//! |---|---|---|
//! | `0x0` | 116,692 | 2,100,474 |
//! | `0x40000` | 229,290 | 4,127,238 |
//! | `0x280000` | 223,380 | 4,020,858 |
//! | the other 27 | 0 | 18 each |
//!
//! Total 10,249,056 rows for INIT alone, against a 2–4 M band for the WHOLE
//! cross-epoch program. The sparse form cannot carry the block, and it is not
//! the pages that make it so — it is three of them.
//!
//! So the three go into a PREPARED OPENING: their INIT columns stacked into one
//! polynomial committed beside the proof, its root absorbed in the roots block,
//! each column settled at its own page table's reduced point. The other 27 keep
//! the sparse form, for which they cost the interned zero.
//!
//! # The threshold, and why it is this shape
//!
//! A page is carried by the opening exactly when its sparse leg alone would
//! cost more than the ENTIRE prepared leg. That is deliberately conservative:
//! every page that joins pays for the whole stack by itself, so the hybrid can
//! never be worse than the sparse form by more than one stack, and no page joins
//! on the strength of a marginal cost that depends on which other pages joined.
//!
//! ⚠ A SET-DEPENDENT RULE WAS THE ALTERNATIVE AND IT WAS REJECTED. Charging each
//! page the stack's MARGINAL cost and the set its FIXED cost is cheaper by a few
//! tens of thousands of rows and makes the answer depend on the order pages are
//! considered in. Prover, verifier and emitter must all reach the SAME set from
//! their own inputs or the transcript is dead at the first challenge, so a rule
//! that is a pure function of one page's own nonzero count is worth more than
//! the rows it gives up.
//!
//! ⛔ AND THE ANSWER IS INSENSITIVE TO THE CONSTANT AT THE BLOCK. The three
//! dense pages are 12x to 24x over [`PREPARED_LEG_ROWS`] and the 27 sparse ones
//! are four orders of magnitude under it. Any threshold between 19 and 2,100,474
//! rows selects the same three pages, which is the argument that this routing
//! decision is not a tuning knob.
//!
//! # ⛔ THE PRIVATE-INPUT PAGES ARE EXCLUDED FIRST, AND NOT AS AN OPTIMISATION
//!
//! The prover builds its page configs with
//! `global_memory_configs_from_init_page_data(..., include_private_genesis =
//! true)` and the verifier with [`crate::continuation::global_memory_configs`],
//! which passes `false`. A private-input page therefore carries
//! `init_values = Some(<the private bytes>)` on the prover and `Some(vec![])` on
//! the verifier. A threshold evaluated over every config would read a different
//! nonzero count for those pages on the two sides, select a different dense set,
//! commit a different stack, absorb a different root and diverge at `z` — a
//! failure with nothing in it that names its cause.
//!
//! So the filter is not "private pages are not worth stacking". It is that a
//! private page HAS no INIT preprocessed column to settle: its INIT is a
//! committed main column the verifier never recomputes. [`plan`] takes the
//! pages that present one, which is a property both sides derive from
//! `page_base` and `num_private_input_pages` — two values the cross-epoch
//! statement already binds.

use crate::tables::page::{self, PageConfig};
use crate::tables::types::FE;
use stark::multilinear_table::PreparedColumn;

/// INIT's index among a genesis page's preprocessed columns.
///
/// ⚠ TIED TO [`page::preprocessed_columns`], which returns
/// `vec![offset_column(), init_col]`. OFFSET is index 0 and keeps its closed
/// form — it is the identity ramp, whose extension is `sum_k 2^k * r_k` and
/// costs `num_vars - 1` rows, so stacking it would buy a closed form nothing
/// and cost a variable on the shared chain. This being 1 rather than 0 is why
/// a prepared opening names a column index instead of a prefix length.
pub const INIT_PREPROCESSED_COLUMN: usize = 1;

/// The rows the in-guest verifier spends discharging one page's INIT by the
/// sparse closed form: `MLE(r) = sum_{v_i != 0} v_i * eq(r, i)`, which is
/// `num_vars` complements hoisted plus `num_vars` rows per nonzero entry.
///
/// An all-zero page costs `num_vars` — the interned zero — and not nothing,
/// which is why the 27 zero pages of the block are 486 rows and not 0.
pub fn sparse_leg_rows(num_vars: usize, nonzero: usize) -> usize {
    num_vars + num_vars * nonzero
}

/// The rows a prepared opening of the genesis stack costs the in-guest
/// verifier, as the budget the threshold charges a page against.
///
/// ★ THE NUMBER IS V1i's MEASUREMENT, NOT AN ESTIMATE: 175,066 chain rows for
/// the stacked family polynomial at 24 variables, through the same `ChainShape`
/// and `chain_shape_rows` forms the epoch program's chains are sized by. The
/// block's stack is 3 columns of 2^18, so 20 variables and FEWER rows than that
/// — the budget is charged at the larger figure, which makes the threshold
/// conservative in the direction that matters: a page must be worth more than
/// the stack could possibly cost before it joins.
///
/// ⚠ CONFIGURATION IS PART OF THIS NUMBER. Chain rows are a function of the
/// `ChainConfig` — blowup, folding schedule, query count — so a posture change
/// moves it. It is a constant here rather than a call because this module sits
/// BELOW `crate::lfm`, where the cost forms live, and a routing rule the
/// protocol depends on must not depend on the machine's row accounting: prover,
/// verifier and emitter each derive the dense set from their own inputs, and a
/// set that moved with an emission constant would be a set they could disagree
/// about. The constant's provenance is asserted where it can be computed —
/// `crate::lfm` — rather than restated here.
pub const PREPARED_LEG_ROWS: usize = 175_066;

/// Whether a page's genesis is dense enough to be worth a prepared opening:
/// its sparse leg alone costs more than the whole stack could.
pub fn is_dense(num_vars: usize, nonzero: usize) -> bool {
    sparse_leg_rows(num_vars, nonzero) > PREPARED_LEG_ROWS
}

/// How many of a page's genesis bytes are nonzero — the only quantity the
/// threshold reads.
///
/// `init_values` is not padded to the page, so every offset at or past its
/// length is zero and costs nothing; a `None` page is zero to the last byte.
pub fn nonzero_entries(config: &PageConfig) -> usize {
    config
        .init_values
        .as_ref()
        .map(|values| values.iter().filter(|&&b| b != 0).count())
        .unwrap_or(0)
}

/// One page of the cross-epoch page family, as the routing decision sees it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PageRoute {
    /// Its index in the AIR set, bookends included — the index a
    /// [`PreparedColumn`] names.
    pub table: usize,
    pub page_base: u64,
    pub nonzero: usize,
    /// `false` for a private-input page, which presents no INIT column at all.
    pub has_init: bool,
    pub dense: bool,
}

/// Which genesis pages the prepared opening carries, and what it costs the
/// pages it leaves behind.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GenesisStackPlan {
    /// Every touched page, in the AIR set's own page order.
    pub routes: Vec<PageRoute>,
    /// Where each stacked column is settled, in stack order — the stack's
    /// column order IS page-base order, which is the order
    /// `global_memory_configs` hands the configs back in and the order
    /// `WhirGlobalAirs` keeps them.
    pub at: Vec<PreparedColumn>,
    /// The rows the pages NOT carried still cost by the sparse form.
    pub sparse_rows: usize,
}

impl GenesisStackPlan {
    /// Whether anything is stacked at all. A run whose genesis is entirely
    /// sparse carries no prepared opening, and the cross-epoch proof is then
    /// byte-for-byte the one it was before this route existed.
    pub fn is_empty(&self) -> bool {
        self.at.is_empty()
    }

    /// The page-family indices the opening carries, in stack order.
    pub fn dense_pages(&self) -> Vec<usize> {
        self.routes
            .iter()
            .enumerate()
            .filter(|(_, r)| r.dense)
            .map(|(page, _)| page)
            .collect()
    }
}

/// The routing decision, from the page configs and where the page family starts
/// in the AIR set.
///
/// `num_bookends` is how many local-to-global tables precede the pages — the
/// cross-epoch AIR set is every bookend then every page, and
/// `WhirGlobalAirs::refs` is the one place that order is written. The
/// [`PreparedColumn::table`] indices this produces are indices into THAT order,
/// because that is the order `multi_prove` and `multi_verify` match
/// positionally.
///
/// `num_vars` is the page tables' height in variables; every page is
/// `DEFAULT_PAGE_SIZE` rows, so one value covers them all and a page that did
/// not match would be a table nobody meant to build.
pub fn plan(configs: &[PageConfig], num_bookends: usize, num_vars: usize) -> GenesisStackPlan {
    let mut routes = Vec::with_capacity(configs.len());
    let mut at = Vec::new();
    let mut sparse_rows = 0usize;

    for (page, config) in configs.iter().enumerate() {
        let table = num_bookends + page;
        // ⛔ The private filter comes FIRST and is not the threshold's business:
        // a private page presents no INIT column, and its `init_values` differ
        // between prover and verifier by construction. See the module header.
        let has_init = !config.is_private_input;
        let nonzero = if has_init { nonzero_entries(config) } else { 0 };
        let dense = has_init && is_dense(num_vars, nonzero);
        if dense {
            at.push(PreparedColumn {
                table,
                column: INIT_PREPROCESSED_COLUMN,
            });
        } else if has_init {
            sparse_rows += sparse_leg_rows(num_vars, nonzero);
        }
        routes.push(PageRoute {
            table,
            page_base: config.page_base,
            nonzero,
            has_init,
            dense,
        });
    }

    GenesisStackPlan {
        routes,
        at,
        sparse_rows,
    }
}

/// The stacked columns themselves, in stack order: each dense page's INIT.
///
/// ⚠ TAKEN FROM [`page::preprocessed_columns`] AND NOT REBUILT. The column the
/// opening commits must be the column the page table's own argument claims, and
/// the page AIR's preprocessed columns come from that function; a second
/// spelling of "the genesis bytes as a column" is how two objects with the same
/// name come to hold different values.
pub fn stack_columns(configs: &[PageConfig], plan: &GenesisStackPlan) -> Vec<Vec<FE>> {
    plan.dense_pages()
        .into_iter()
        .map(|page| page::preprocessed_columns(&configs[page])[INIT_PREPROCESSED_COLUMN].clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The block's measured census, reproduced by the forms — which is an F1 on
    /// the census rather than a restatement of it.
    ///
    /// Read on the box at 60d790310 (`cens2`, one execution of the guest, no
    /// proving, no card): `GENESIS CENSUS: 35 pages = 30 genesis + 5 private;
    /// 569362 nonzero entries, worst page 229290; 10249056 leg rows`.
    const BLOCK_DENSE: [(u64, usize, usize); 3] = [
        (0x0, 116_692, 2_100_474),
        (0x40000, 229_290, 4_127_238),
        (0x280000, 223_380, 4_020_858),
    ];
    const BLOCK_ZERO_PAGES: usize = 27;
    const BLOCK_PAGE_VARS: usize = 18;
    const BLOCK_CENSUS_LEG_ROWS: usize = 10_249_056;

    #[test]
    fn the_sparse_form_reproduces_the_blocks_census_to_the_row() {
        let mut total = 0usize;
        for (base, nonzero, rows) in BLOCK_DENSE {
            let got = sparse_leg_rows(BLOCK_PAGE_VARS, nonzero);
            assert_eq!(got, rows, "page {base:#x}: {nonzero} nonzero entries");
            total += got;
        }
        // The 27 all-zero pages are not free: each costs the interned zero.
        total += BLOCK_ZERO_PAGES * sparse_leg_rows(BLOCK_PAGE_VARS, 0);
        assert_eq!(
            total, BLOCK_CENSUS_LEG_ROWS,
            "the closed form and the box census disagree about what INIT costs"
        );
    }

    /// ★ THE PRE-REGISTRATION: exactly the three pages, and nothing near the
    /// line.
    #[test]
    fn the_threshold_selects_exactly_the_blocks_three_dense_pages() {
        for (base, nonzero, _) in BLOCK_DENSE {
            assert!(
                is_dense(BLOCK_PAGE_VARS, nonzero),
                "page {base:#x} carries {nonzero} nonzero entries and must be stacked"
            );
        }
        assert!(
            !is_dense(BLOCK_PAGE_VARS, 0),
            "an all-zero page must never be stacked: the sparse form is free for it"
        );
        // The fixture guest `data_page_touch` reads 112 — far below, which is
        // the fixture gap this path has and states rather than hides.
        assert!(
            !is_dense(BLOCK_PAGE_VARS, 112),
            "the existing fixture's page must route sparse, or the gap is not the gap"
        );
    }

    /// The decision does not sit near the constant, which is what makes it a
    /// routing rule rather than a tuning knob.
    #[test]
    fn nothing_on_the_block_sits_near_the_threshold() {
        let least_dense = BLOCK_DENSE
            .iter()
            .map(|&(_, _, rows)| rows)
            .min()
            .expect("three pages");
        assert!(
            least_dense > 10 * PREPARED_LEG_ROWS,
            "the cheapest stacked page is {least_dense} rows against a {PREPARED_LEG_ROWS} \
             budget — closer than an order of magnitude makes the constant load-bearing"
        );
        let most_sparse = sparse_leg_rows(BLOCK_PAGE_VARS, 0);
        assert!(
            most_sparse * 1000 < PREPARED_LEG_ROWS,
            "the most expensive unstacked page is {most_sparse} rows against a \
             {PREPARED_LEG_ROWS} budget"
        );
    }

    /// The break-even in nonzero entries, stated so a fixture can be built to
    /// cross it and so a reader can size one.
    #[test]
    fn the_break_even_is_where_the_forms_say_it_is() {
        // The least S with `num_vars + num_vars*S > PREPARED_LEG_ROWS`, which is
        // floor + 1 and NOT `div_ceil`: were the division exact, `div_ceil`
        // would hand back an S whose leg EQUALS the budget and does not exceed
        // it. The two agree at 18 variables, which is exactly why the wrong one
        // would have gone unnoticed.
        let break_even = (PREPARED_LEG_ROWS - BLOCK_PAGE_VARS) / BLOCK_PAGE_VARS + 1;
        assert_eq!(
            break_even, 9_725,
            "the least nonzero count that earns a prepared opening at 18 variables"
        );
        assert!(!is_dense(BLOCK_PAGE_VARS, break_even - 1));
        assert!(is_dense(BLOCK_PAGE_VARS, break_even));
    }

    fn data_page(base: u64, bytes: Vec<u8>) -> PageConfig {
        PageConfig::with_data(base, bytes)
    }

    /// ⛔ THE PRIVATE FILTER, adversarially: a private-input page whose genesis
    /// bytes WOULD cross the threshold is still not stacked.
    ///
    /// This is the arm that would have caught the prover and the verifier
    /// selecting different dense sets. On the prover a private page carries the
    /// private input; on the verifier it carries an empty vec. If the threshold
    /// read those bytes, the two sides would disagree about the stack and the
    /// transcript would die at `z` with nothing naming why.
    #[test]
    fn a_private_page_dense_enough_to_qualify_is_still_not_stacked() {
        let dense_bytes = vec![0xABu8; 20_000];
        assert!(
            is_dense(BLOCK_PAGE_VARS, dense_bytes.len()),
            "the fixture bytes must qualify, or this test cannot fail"
        );

        let mut private = data_page(0xff000000, dense_bytes.clone());
        private.is_private_input = true;
        let public = data_page(0x40000, dense_bytes);

        let plan = plan(&[private, public], 3, BLOCK_PAGE_VARS);
        assert_eq!(
            plan.at,
            vec![PreparedColumn {
                table: 4,
                column: INIT_PREPROCESSED_COLUMN,
            }],
            "only the non-private page may be stacked, at its own AIR-set index"
        );
        assert!(!plan.routes[0].dense);
        assert!(!plan.routes[0].has_init);
        assert!(plan.routes[1].dense);
    }

    /// The same page set with the private page's bytes REMOVED — the verifier's
    /// view of it — plans identically. That is the property the two sides need
    /// and the reason the filter is on `is_private_input` and not on the bytes.
    #[test]
    fn the_prover_and_verifier_views_of_a_private_page_plan_alike() {
        let dense_bytes = vec![0xABu8; 20_000];
        let mut prover_side = data_page(0xff000000, dense_bytes.clone());
        prover_side.is_private_input = true;
        let mut verifier_side = data_page(0xff000000, Vec::new());
        verifier_side.is_private_input = true;
        let public = data_page(0x40000, dense_bytes);

        let from_prover = plan(&[prover_side, public.clone()], 3, BLOCK_PAGE_VARS);
        let from_verifier = plan(&[verifier_side, public], 3, BLOCK_PAGE_VARS);
        assert_eq!(
            from_prover.at, from_verifier.at,
            "prover and verifier must commit the same stack or the roots block diverges"
        );
        assert_eq!(from_prover.sparse_rows, from_verifier.sparse_rows);
    }

    /// The table index a stacked column names is the AIR set's, bookends
    /// included — not the page's index in its own family.
    #[test]
    fn the_stacked_column_names_its_index_in_the_whole_air_set() {
        let dense_bytes = vec![0x01u8; 20_000];
        let configs = vec![
            PageConfig::zero_init(0x0),
            data_page(0x40000, dense_bytes),
            PageConfig::zero_init(0x80000),
        ];
        let plan = plan(&configs, 15, BLOCK_PAGE_VARS);
        assert_eq!(
            plan.at,
            vec![PreparedColumn {
                table: 16,
                column: INIT_PREPROCESSED_COLUMN,
            }],
            "fifteen bookends precede the pages, so page 1 is table 16"
        );
        assert_eq!(plan.dense_pages(), vec![1]);
        // The two zero pages still cost the interned zero apiece.
        assert_eq!(plan.sparse_rows, 2 * sparse_leg_rows(BLOCK_PAGE_VARS, 0));
    }

    /// A run whose genesis is entirely sparse carries no opening at all, and
    /// the cross-epoch proof is then the one it was before this route existed.
    #[test]
    fn an_all_sparse_page_set_stacks_nothing() {
        let configs = vec![
            PageConfig::zero_init(0x0),
            data_page(0x40000, vec![1u8; 112]),
        ];
        let plan = plan(&configs, 3, BLOCK_PAGE_VARS);
        assert!(plan.is_empty());
        assert!(plan.dense_pages().is_empty());
    }
}
