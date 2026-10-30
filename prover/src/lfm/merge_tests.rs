//! The merged LFM tables ([`super::merge`], `LAMBDA_VM_ZF_LFM_MERGE`): the plan
//! takes equal heights only; the layout puts every prefix first; a part's body
//! reads its own main columns and nothing else; the buses carry over column by
//! column; the direct fill is the parts side by side; a merged program proves
//! and verifies, under its own artifacts only; a bad row inside a merged table
//! is refused, and the part's body is what refuses it (the mutant); a node's
//! leg verifies a merged child and refuses its tampers; and the format knob
//! parses, prints and refuses as its siblings do. With the flag off nothing
//! moves: the plan is empty, the digest is the unmerged one, the fill is the
//! unmerged fill.

use std::cell::RefCell;
use std::sync::{Mutex, MutexGuard};

use stark::constraints::builder::{ConstraintBuilder, ConstraintSet, Nil, RowDomain};
use stark::proof::options::{GoldilocksCubicProofOptions, OneRowMode, ProofOptions};
use stark::proof::view::MultiProofView;
use stark::trace::TraceTable;

use crate::hash_pin::{P1Family, ProofFamily};
use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};

use super::airs::{LfmAirs, NUM_LFM_CHIPS};
use super::builder::LfmBuilder;
use super::compiler::{LfmProgram, compile};
use super::executor::execute;
use super::merge::{
    LfmMerge, MERGE_CANDIDATES, MergeChip, MergedConstraints, MergedLayout, MergedTable, PartBody,
    PartShape, merged_interactions, plan_for_heights, plan_for_program, test_hooks,
};
use super::p1_node_tests::{P1Child, harvest, node, runs};
use super::proof::{LfmProof, lfm_prove, lfm_prove_under, prove_traces, verify_against_artifacts};
use super::registry::{LfmArtifacts, build_artifacts, build_artifacts_under};
use super::trace::{LfmTraces, build_traces_with_hasher, build_traces_with_merge};
use super::word::{LfmWord, base_word};

type F = GoldilocksField;
type E = GoldilocksExtension;

/// Every test here that builds a merged AIR holds this: the mutation hook
/// ([`test_hooks`]) is process-wide.
fn lock() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn fe(v: u64) -> FE {
    FE::from(v)
}

fn ext(a: u64, b: u64, c: u64) -> FEE {
    FEE::new([fe(a), fe(b), fe(c)])
}

fn options(merge: bool) -> ProofOptions {
    let mut o = GoldilocksCubicProofOptions::with_blowup(2).expect("options");
    o.format.merge_equal_height_tables = merge;
    o
}

/// A program with real rows in all four candidates — base ALU (add, sub, mul),
/// one select, an unpack and a pack, one hinted felt — each padded to the
/// four-row floor, so the four share a height and merge.
fn four_chip_program() -> (LfmProgram, Vec<Vec<LfmWord>>) {
    let mut b = LfmBuilder::new();
    let arena = b.declare_arena(1);
    let h0 = b.hint_felt(arena, 0);
    let x = b.felt_const(fe(7));
    let y = b.felt_const(fe(5));
    let s = b.add(x, y);
    let d = b.sub(s, h0);
    let m = b.mul(d, y);
    // An add nobody reads: its output's write multiplicity is zero, so its row
    // sends no memory token — only BALU's own body constrains its OUT
    // (`a_bad_balu_row_in_a_merged_table_is_refused_by_balus_own_body`).
    let _unread = b.add(y, x);
    let e0 = b.ext_const(&ext(1, 2, 3));
    let e1 = b.ext_const(&ext(4, 5, 6));
    let ep = b.emul(e0, e1);
    let bit = b.bit_const(true);
    let (l, r) = b.select(bit, x.as_cell(), y.as_cell());
    let da = b.digest_const([fe(1), fe(2), fe(3), fe(4)]);
    let db = b.digest_const([fe(5), fe(6), fe(7), fe(8)]);
    let dc = b.compress(da, db);
    let lanes = b.unpack(dc.as_cell());
    let packed = b.pack_word(lanes);
    b.public(packed);
    b.public(l);
    b.public(r);
    b.public(ep.as_cell());
    b.public(m.as_cell());
    (compile(b.finish()), vec![vec![base_word(fe(3))]])
}

fn verifies(artifacts: &LfmArtifacts, proof: &LfmProof, opts: &ProofOptions) -> bool {
    verify_against_artifacts(artifacts, &proof.proof, &proof.public_words, opts)
}

fn sub_proofs(proof: &LfmProof) -> usize {
    MultiProofView::Owned(&proof.proof.multi).len()
}

// ================================ the plan =================================

#[test]
fn the_plan_merges_equal_heights_only() {
    let mut h = [0u8; NUM_LFM_CHIPS];
    let set = |h: &mut [u8; NUM_LFM_CHIPS], b, s, l, i| {
        h[MergeChip::Balu.slot()] = b;
        h[MergeChip::Select.slot()] = s;
        h[MergeChip::Lanes.slot()] = l;
        h[MergeChip::Hint.slot()] = i;
    };
    // A leaf: all four at 2^21.
    set(&mut h, 21, 21, 21, 21);
    assert_eq!(plan_for_heights(&h, true), vec![MERGE_CANDIDATES.to_vec()]);
    assert!(plan_for_heights(&h, false).is_empty(), "off merges nothing");
    // A top: SELECT and HINT at 2^20, BALU and LANES at 2^19 — two merged AIRs,
    // ordered by their first chip.
    set(&mut h, 19, 20, 19, 20);
    assert_eq!(
        plan_for_heights(&h, true),
        vec![
            vec![MergeChip::Balu, MergeChip::Lanes],
            vec![MergeChip::Select, MergeChip::Hint]
        ]
    );
    // A chip alone at its height keeps its own AIR.
    set(&mut h, 18, 20, 20, 19);
    assert_eq!(
        plan_for_heights(&h, true),
        vec![vec![MergeChip::Select, MergeChip::Lanes]]
    );
    // Four heights: nothing to merge.
    set(&mut h, 17, 18, 19, 20);
    assert!(plan_for_heights(&h, true).is_empty());
    // Other chips' heights never matter.
    let mut other = [21u8; NUM_LFM_CHIPS];
    set(&mut other, 17, 18, 19, 20);
    assert!(plan_for_heights(&other, true).is_empty());
}

#[test]
fn the_layout_puts_every_prefix_first_and_maps_each_column_once() {
    let layout = MergedLayout::of(&MERGE_CANDIDATES);
    let shapes: Vec<PartShape> = MERGE_CANDIDATES.iter().map(|&c| PartShape::of(c)).collect();
    assert_eq!(
        layout.num_prep,
        shapes.iter().map(|s| s.prep).sum::<usize>()
    );
    assert_eq!(
        layout.num_columns,
        shapes.iter().map(|s| s.width).sum::<usize>()
    );
    // The census of the note: 32 preprocessed, 61 in all.
    assert_eq!((layout.num_prep, layout.num_columns), (32, 61));
    let mut hit = vec![0usize; layout.num_columns];
    for (part, shape) in layout.parts.iter().zip(&shapes) {
        assert_eq!(part.map.len(), shape.width);
        for (c, &m) in part.map.iter().enumerate() {
            hit[m] += 1;
            // A prefix column stays in the merged prefix; a value column after it.
            assert_eq!(
                c < shape.prep,
                m < layout.num_prep,
                "{:?} column {c}",
                part.chip
            );
        }
        // Each region keeps the part's order and is contiguous.
        assert!(
            part.map
                .windows(2)
                .all(|w| w[1] > w[0] || w[1] < layout.num_prep)
        );
    }
    assert!(
        hit.iter().all(|&n| n == 1),
        "every merged column is one part's"
    );
}

// ============================ the constraint body ==========================

/// A builder that records which leaves a body reads.
#[derive(Default)]
struct Recorder {
    mains: RefCell<Vec<(usize, usize)>>,
    others: RefCell<usize>,
    emitted: Vec<usize>,
}

impl ConstraintBuilder<F, E> for Recorder {
    type Expr = Nil;
    type ExprE = Nil;
    fn main(&self, offset: usize, col: usize) -> Nil {
        self.mains.borrow_mut().push((offset, col));
        Nil
    }
    fn aux(&self, _: usize, _: usize) -> Nil {
        *self.others.borrow_mut() += 1;
        Nil
    }
    fn challenge(&self, _: usize) -> Nil {
        *self.others.borrow_mut() += 1;
        Nil
    }
    fn alpha_pow(&self, _: usize) -> Nil {
        *self.others.borrow_mut() += 1;
        Nil
    }
    fn table_offset(&self) -> Nil {
        *self.others.borrow_mut() += 1;
        Nil
    }
    fn const_base(&self, _: u64) -> Nil {
        Nil
    }
    fn const_signed(&self, _: i64) -> Nil {
        Nil
    }
    fn emit_base_rows(&mut self, idx: usize, _: RowDomain, _: Nil) {
        self.emitted.push(idx);
    }
    fn emit_ext_rows(&mut self, _: usize, _: RowDomain, _: Nil) {
        *self.others.borrow_mut() += 1;
    }
}

/// ★ The premise [`super::merge::Remap`] rests on: a candidate's body reads
/// only its own main columns, at the current row, and emits base constraints
/// only — so remapping `main` and shifting the indices is the whole
/// translation. Then the merged body emits every part's constraints, densely,
/// each reading exactly its part's columns where the layout put them.
#[test]
fn a_part_body_reads_only_its_main_columns() {
    let _g = lock();
    for chip in MERGE_CANDIDATES {
        let shape = PartShape::of(chip);
        let mut r = Recorder::default();
        match shape.body {
            PartBody::Balu => super::chips::balu::BaluConstraints.eval(&mut r),
            PartBody::Select => super::chips::select::SelectConstraints.eval(&mut r),
            PartBody::Empty => {}
        }
        assert_eq!(*r.others.borrow(), 0, "{chip:?} reads no aux leaf");
        assert!(
            r.mains
                .borrow()
                .iter()
                .all(|&(o, c)| o == 0 && c < shape.width),
            "{chip:?} reads its own columns at the current row"
        );
        assert_eq!(r.emitted.len(), shape.num_base_constraints());
    }
    let layout = MergedLayout::of(&MERGE_CANDIDATES);
    let set = MergedConstraints::new(&layout).expect("the candidates pass the audit");
    let mut r = Recorder::default();
    set.eval(&mut r);
    let mut emitted = r.emitted.clone();
    emitted.sort_unstable();
    let total: usize = MERGE_CANDIDATES
        .iter()
        .map(|&c| PartShape::of(c).num_base_constraints())
        .sum();
    assert_eq!(emitted, (0..total).collect::<Vec<_>>(), "dense, each once");
    assert_eq!(ConstraintSet::<F, E>::max_degree(&set), 3, "BALU's degree");
    // BALU reads A, B, C, OUT and its five selectors; under the merge those are
    // its map's images.
    let balu = &layout.parts[0];
    let mut own = Recorder::default();
    super::chips::balu::BaluConstraints.eval(&mut own);
    let want: Vec<(usize, usize)> = own
        .mains
        .borrow()
        .iter()
        .map(|&(o, c)| (o, balu.map[c]))
        .collect();
    assert_eq!(r.mains.borrow()[..want.len()], want[..]);
}

/// ★ The buses carry over interaction by interaction: same bus, same side,
/// same element count, and every column read moved by its part's map — and
/// every packing stays contiguous (the premise of `remap_interaction`).
#[test]
fn every_merge_remaps_its_buses() {
    let layout = MergedLayout::of(&MERGE_CANDIDATES);
    let merged = merged_interactions(&layout).expect("contiguous");
    let mut at = 0;
    for part in &layout.parts {
        for original in PartShape::of(part.chip).interactions() {
            let m = &merged[at];
            at += 1;
            assert_eq!(m.bus_id, original.bus_id);
            assert_eq!(m.is_sender, original.is_sender);
            assert_eq!(m.num_bus_elements(), original.num_bus_elements());
            let mut want: Vec<usize> = original
                .columns_read()
                .iter()
                .map(|&c| part.map[c])
                .collect();
            want.sort_unstable();
            assert_eq!(m.columns_read(), want, "{:?}", part.chip);
            for v in &original.values {
                if let stark::lookup::BusValue::Packed {
                    start_column,
                    packing,
                } = v
                {
                    let first = part.map[*start_column];
                    assert!(
                        (0..packing.num_columns()).all(|k| part.map[start_column + k] == first + k)
                    );
                }
            }
        }
    }
    assert_eq!(at, merged.len());
    // 4 + 5 + 10 + 1 interactions: ten aux columns where the parts had eleven.
    assert_eq!(merged.len(), 20);
}

// ================================ the traces ===============================

fn main_cells(t: &TraceTable<F, E>) -> Vec<FE> {
    t.main_data_row_major().0.to_vec()
}

/// ★ The direct fill IS the parts side by side: the merged table built by the
/// fill equals the one laid out from the unmerged tables, cell for cell, and
/// splitting it back gives the unmerged tables. With an empty plan the fill is
/// the unmerged fill.
#[test]
fn the_merged_fill_is_the_parts_side_by_side() {
    let (program, arenas) = four_chip_program();
    let hasher = build_artifacts(&program, &options(false)).hasher;
    let exec = execute(&program, &arenas, &hasher).expect("executes");
    let plan = plan_for_program(&program, true);
    assert_eq!(
        plan,
        vec![MERGE_CANDIDATES.to_vec()],
        "the four share the floor"
    );

    let plain = build_traces_with_hasher(&program, &exec.records, hasher);
    let same = build_traces_with_merge(&program, &exec.records, hasher, &Vec::new());
    let chips = |t: &LfmTraces| -> Vec<Vec<FE>> {
        [&t.balu, &t.select, &t.lanes, &t.hint]
            .iter()
            .map(|x| main_cells(x))
            .collect()
    };
    assert_eq!(chips(&plain), chips(&same), "no plan, no change");
    assert!(same.merged.is_empty());

    let direct = build_traces_with_merge(&program, &exec.records, hasher, &plan);
    assert_eq!(direct.merged.len(), 1);
    assert!(direct.balu.num_rows() == 0 && direct.hint.num_rows() == 0);
    let mut copied = build_traces_with_hasher(&program, &exec.records, hasher);
    copied.merge_to(&plan).expect("merges");
    assert_eq!(main_cells(&direct.merged[0]), main_cells(&copied.merged[0]));
    // Cell by cell against the parts' own tables.
    let layout = MergedLayout::of(&plan[0]);
    let parts = chips(&plain);
    let merged = main_cells(&direct.merged[0]);
    for (part, cells) in layout.parts.iter().zip(&parts) {
        let w = part.map.len();
        for row in 0..direct.merged[0].num_rows() {
            for (c, &m) in part.map.iter().enumerate() {
                assert_eq!(merged[row * layout.num_columns + m], cells[row * w + c]);
            }
        }
    }
    copied.merge_to(&Vec::new()).expect("splits back");
    assert_eq!(chips(&copied), chips(&plain));
    assert!(copied.merged.is_empty());
}

// ============================ prove and verify =============================

/// ★ The round trip, and identity: a merged program has one AIR where its
/// four chips had four, its merged chips' own roots are the zero sentinel and
/// the merged root is bound into a different `program_id`; it proves and
/// verifies under its own artifacts and is refused under the unmerged ones,
/// and the reverse. With the flag off the artifacts carry no merge.
#[test]
fn a_merged_program_proves_and_verifies_under_its_own_artifacts_only() {
    let _g = lock();
    let (program, arenas) = four_chip_program();
    let (on, off) = (options(true), options(false));
    let merged = build_artifacts(&program, &on);
    let plain = build_artifacts(&program, &off);
    assert!(plain.merge.is_empty(), "off: no merge");
    assert_eq!(merged.merge.plan(), vec![MERGE_CANDIDATES.to_vec()]);
    assert_ne!(
        merged.program_id, plain.program_id,
        "the layout is identity"
    );
    for c in MERGE_CANDIDATES {
        assert_eq!(
            merged.roots[c.slot()],
            [0u8; 32],
            "{c:?}: no root of its own"
        );
        assert_ne!(plain.roots[c.slot()], [0u8; 32]);
    }
    let p_merged = lfm_prove(&program, &merged, &arenas, &on).expect("merged proves");
    let p_plain = lfm_prove(&program, &plain, &arenas, &off).expect("plain proves");
    assert_eq!(sub_proofs(&p_merged) + 3, sub_proofs(&p_plain));
    assert_eq!(p_merged.public_words, p_plain.public_words);
    assert!(
        verifies(&merged, &p_merged, &on),
        "the merged proof verifies"
    );
    assert!(verifies(&plain, &p_plain, &off), "the plain proof verifies");
    assert!(
        !verifies(&plain, &p_merged, &off),
        "a merged proof under plain artifacts"
    );
    assert!(
        !verifies(&merged, &p_plain, &on),
        "a plain proof under merged artifacts"
    );
}

/// The merged artifacts' AIR order: the merged AIR stands at BALU's place,
/// the other three chips leave, and the heights in AIR order follow
/// (`log_heights_in_air_order`, what a parent derives its child from).
#[test]
fn the_merged_air_stands_at_its_first_chip() {
    let _g = lock();
    let (program, _) = four_chip_program();
    let opts = options(true);
    let artifacts = build_artifacts(&program, &opts);
    let airs = LfmAirs::for_artifacts(&artifacts, &opts).expect("the AIR set");
    let names: Vec<String> = airs
        .air_refs()
        .iter()
        .map(|a| a.name().to_string())
        .collect();
    assert_eq!(names[1], "LFM_MERGED[BALU+SELECT+LANES+HINT]", "{names:?}");
    for gone in ["LFM_BALU", "LFM_SELECT", "LFM_LANES", "LFM_HINT"] {
        assert!(!names.iter().any(|n| n == gone), "{gone} left: {names:?}");
    }
    let heights = LfmAirs::log_heights_in_air_order(&artifacts).expect("no keccak");
    assert_eq!(heights.len(), names.len());
    let merged = &airs.air_refs()[1];
    assert_eq!(merged.num_precomputed_columns(), 32);
    assert_eq!(merged.context().trace_columns, 61 + 10, "61 main, 10 aux");
}

/// The traces with SELECT's padding row 2 given a non-boolean bit: no bus
/// token moves (the row's multiplicities are zero), only SELECT's booleanity
/// fails.
fn bad_select_traces(
    program: &LfmProgram,
    arenas: &[Vec<LfmWord>],
    artifacts: &LfmArtifacts,
) -> (LfmTraces, Vec<(u32, LfmWord)>) {
    let exec = execute(program, arenas, &artifacts.hasher).expect("executes");
    let plan = artifacts.merge.plan();
    let mut traces = build_traces_with_merge(program, &exec.records, artifacts.hasher, &plan);
    let layout = MergedLayout::of(&plan[0]);
    let select = layout
        .parts
        .iter()
        .find(|p| p.chip == MergeChip::Select)
        .expect("SELECT is merged");
    let row = 2;
    assert!(row >= program.groups.select.real_rows, "a padding row");
    traces.merged[0].set_main(row, select.map[super::chips::select::cols::BIT], fe(2));
    (traces, exec.public_words)
}

/// ★ Negative, then the mutant: a merged table with a non-boolean SELECT bit
/// on a padding row is refused by the verifier — and with the merged AIR's
/// SELECT body vacuous (its indices kept, every constraint zero; prover and
/// verifier both), the same table proves and VERIFIES. So the part's body
/// inside the merged AIR is what refuses it.
#[test]
fn a_bad_select_row_in_a_merged_table_is_refused_by_selects_own_body() {
    let _g = lock();
    let (program, arenas) = four_chip_program();
    let opts = options(true);
    let artifacts = build_artifacts(&program, &opts);

    let (mut traces, words) = bad_select_traces(&program, &arenas, &artifacts);
    let refused = match prove_traces(&artifacts, &mut traces, &words, &opts) {
        Ok(proof) => !verify_against_artifacts(&artifacts, &proof, &words, &opts),
        Err(_) => true,
    };
    assert!(refused, "the bad SELECT row must not verify");

    test_hooks::set_vacuous(Some(PartBody::Select));
    let mutant = std::panic::catch_unwind(|| {
        let (mut traces, words) = bad_select_traces(&program, &arenas, &artifacts);
        let proof =
            prove_traces(&artifacts, &mut traces, &words, &opts).expect("the mutant proves");
        verify_against_artifacts(&artifacts, &proof, &words, &opts)
    });
    test_hooks::set_vacuous(None);
    assert!(
        mutant.expect("the mutant runs"),
        "without SELECT's body the bad row verifies: the body is load-bearing"
    );
}

/// A wrong BALU output inside the merged table (a real row, so it also moves
/// the memory bus) is refused.
#[test]
fn a_wrong_balu_output_in_a_merged_table_is_refused() {
    let _g = lock();
    let (program, arenas) = four_chip_program();
    let opts = options(true);
    let artifacts = build_artifacts(&program, &opts);
    let exec = execute(&program, &arenas, &artifacts.hasher).expect("executes");
    let plan = artifacts.merge.plan();
    let mut traces = build_traces_with_merge(&program, &exec.records, artifacts.hasher, &plan);
    let layout = MergedLayout::of(&plan[0]);
    let col = layout.parts[0].map[super::chips::balu::cols::OUT];
    let v = *traces.merged[0].get_main(0, col);
    traces.merged[0].set_main(0, col, v + fe(1));
    let refused = match prove_traces(&artifacts, &mut traces, &exec.public_words, &opts) {
        Ok(proof) => !verify_against_artifacts(&artifacts, &proof, &exec.public_words, &opts),
        Err(_) => true,
    };
    assert!(refused);
}

// ============================== the node's leg =============================

/// ★ A node's leg verifies a MERGED Poseidon1 child in-guest: the leg is
/// derived from the merged artifacts (no leg-specific code), the honest child
/// executes, and tampers of the merged table's root, an opened value and an
/// OOD value are refused. The node over the merged child takes fewer socket
/// rows than over the plain one (printed for the census).
#[test]
fn a_node_verifies_a_merged_child_and_refuses_its_tampers() {
    let _g = lock();
    let program = super::programs::trivial_program();
    let child = |merge: bool| -> P1Child {
        let mut opts = super::p1_node_tests::options();
        opts.format.merge_equal_height_tables = merge;
        let hasher = build_artifacts(&program, &opts).hasher;
        let artifacts =
            build_artifacts_under::<<P1Family as ProofFamily>::Stark>(&program, &opts, hasher);
        let proof = lfm_prove_under::<P1Family>(
            &program,
            &artifacts,
            &super::p1_node_tests::arenas(),
            &opts,
        )
        .expect("prove");
        assert!(super::proof::verify_against_artifacts_under::<P1Family>(
            &artifacts,
            &proof.proof,
            &proof.public_words,
            &opts
        ));
        P1Child {
            artifacts,
            proof,
            opts,
        }
    };
    let c = child(true);
    assert!(!c.artifacts.merge.is_empty(), "the trivial program merges");
    let honest = harvest(&c);
    let node_program = node(&c, P1Family::STATEMENT_TAG);
    assert!(
        runs(&node_program, &honest),
        "the honest merged child executes"
    );

    let airs = LfmAirs::for_artifacts(&c.artifacts, &c.opts).expect("the AIR set");
    let t = airs
        .air_refs()
        .iter()
        .position(|a| a.name().starts_with("LFM_MERGED"))
        .expect("a merged AIR");
    let refused = |what: &str, edit: &dyn Fn(&mut super::harvest::HarvestedChild)| {
        let mut bad = harvest(&c);
        edit(&mut bad);
        assert!(!runs(&node_program, &bad), "{what}: the node must refuse");
    };
    refused("merged main root", &|h| h.tables[t].main_root[31] ^= 1);
    refused("merged opened value", &|h| {
        h.legs[t].openings[0][0].0[0][0] += fe(1)
    });
    refused("merged OOD value", &|h| {
        h.tables[t].ood_current[0] += FEE::from(1u64)
    });

    let plain = child(false);
    let plain_node = node(&plain, P1Family::STATEMENT_TAG);
    let (rows_merged, rows_plain) = (
        node_program.groups.hash.real_rows,
        plain_node.groups.hash.real_rows,
    );
    println!(
        "MERGE NODE CENSUS: socket rows for one child: merged {rows_merged} · plain {rows_plain} · AIRs {} vs {}",
        sub_proofs(&c.proof),
        sub_proofs(&plain.proof)
    );
    assert!(rows_merged < rows_plain, "fewer tables, fewer paths");
}

// ================================ the knob =================================

#[test]
fn the_merge_knob_parses_prints_and_refuses_like_its_siblings() {
    use crate::zf_format::{ENV_LFM_MERGE, ENV_ONE_ROW, ZfFormat};
    let parse = |pairs: &[(&str, &str)]| {
        ZfFormat::from_lookup(|name| {
            pairs
                .iter()
                .find(|(k, _)| *k == name)
                .map(|(_, v)| v.to_string())
        })
    };
    let default = parse(&[]).expect("default");
    assert!(!default.lfm_merge);
    assert!(
        !default.banner().contains("lfm_merge"),
        "the default banner is today's"
    );
    assert!(!default.proof_format().merge_equal_height_tables);
    let on = parse(&[(ENV_LFM_MERGE, "1")]).expect("on");
    assert!(on.lfm_merge && on.proof_format().merge_equal_height_tables);
    assert!(on.banner().ends_with(" lfm_merge=1"));
    assert!(!parse(&[(ENV_LFM_MERGE, "0")]).expect("off").lfm_merge);
    assert!(parse(&[(ENV_LFM_MERGE, "2")]).is_err(), "a typo aborts");
    assert!(
        parse(&[(ENV_LFM_MERGE, "1"), (ENV_ONE_ROW, "1")]).is_err(),
        "no merged one-row layout"
    );
    const { assert!(!ZfFormat::LEGACY.lfm_merge && !ZfFormat::DEFAULT.lfm_merge) };
    // The library treats the pair as off on both sides.
    let (program, _) = four_chip_program();
    let mut both = options(true);
    both.format.one_row = OneRowMode::On;
    assert!(super::registry::merge_plan_for(&program, &both).is_empty());
    assert_eq!(LfmMerge::none().airs_saved(), 0);
}

// ================================ the census ===============================

/// One tree program's AIRs under its artifacts: count, and main + aux cells
/// (an aux column is three base cells) over every AIR.
fn airs_and_cells(artifacts: &LfmArtifacts, wrap: &ProofOptions) -> (usize, u64) {
    let airs = LfmAirs::for_artifacts(artifacts, wrap).expect("the AIR set");
    let heights = LfmAirs::log_heights_in_air_order(artifacts).expect("no keccak");
    let refs = airs.air_refs();
    let cells = refs
        .iter()
        .zip(&heights)
        .map(|(air, h)| {
            let aux = air.context().trace_columns - air.trace_layout().0;
            (1u64 << h) * (air.trace_layout().0 + 3 * aux) as u64
        })
        .sum();
    (refs.len(), cells)
}

/// ★ The merge census (instrument, `--ignored --exact`): the median mix's
/// plan (the production format, the Poseidon1 base) under today's default
/// leaf scale, every leaf emitted once, heights only; per arm (plain, merged):
/// each leaf's AIRs and cells, then the top emitted over the leaves' shapes —
/// its socket rows, its chips' rows and its own AIRs — and the socket rows one
/// node spends on one leaf. No commit, no proof.
///
/// `CENSUS_ELF` names a real guest (default: the asm test guest).
#[test]
#[ignore = "instrument: run alone with --exact"]
fn merge_census() {
    use super::whir_block::WhirBlockPlan;
    use super::whir_chain::ChipRows;
    use crate::block_whir::BlockFormat;
    let elf_bytes = match std::env::var("CENSUS_ELF") {
        Ok(path) => std::fs::read(path).expect("read CENSUS_ELF"),
        Err(_) => crate::test_utils::asm_elf_bytes("poc_rodata_commit"),
    };
    let opts = super::proof::block_base_options();
    let production = BlockFormat::production();
    let format = BlockFormat {
        zf: production
            .zf
            .with_base(stark::proof::options::BaseFormat::P1_WHIR),
        ..production
    };
    let owned = super::whir_p1_census_tests::median_statement(&elf_bytes, &opts, &format);
    let plan = WhirBlockPlan::derive(&elf_bytes, &opts, &format, owned.view(), None)
        .expect("the plan derives");
    let words = plan.child_layout().total();
    println!(
        "MERGE CENSUS plan: {} groups · {} leaves · levels {:?}",
        plan.num_groups(),
        plan.partition().num_leaves(),
        plan.levels()
            .iter()
            .map(|l| l.arities.clone())
            .collect::<Vec<_>>()
    );
    for on in [false, true] {
        let label = if on { "merged" } else { "plain" };
        let mut wrap = super::proof::aggregation_wrap_options();
        wrap.format.merge_equal_height_tables = on;
        let mut kids = Vec::new();
        for k in 0..plan.partition().num_leaves() {
            let program = plan.leaf_program(k).expect("the leaf emits");
            let a = super::registry::census_artifacts_merged(
                &program,
                program.hasher(<P1Family as ProofFamily>::SOCKET),
                on,
            );
            let (n, cells) = airs_and_cells(&a, &wrap);
            println!(
                "MERGE CENSUS {label} leaf {k}: {n} AIRs · {:.1} M main+aux cells · merged {:?}",
                cells as f64 / 1e6,
                a.merge.plan()
            );
            kids.push(super::whir_block::derived_of(&a, &wrap, words).expect("derives"));
        }
        let one = plan
            .node_program(&[&kids[0]], false)
            .expect("one-child node");
        let one_rows = ChipRows::of(&one);
        println!(
            "MERGE CENSUS {label} one-child node over leaf 0: socket {} · {one_rows:?}",
            one_rows.hash
        );
        drop(one);
        // The first node above the leaves (the top when the tree has one
        // level): its rows, and its own AIRs under the arm.
        let level = plan.levels();
        let top_arity = level.first().map_or(1, |l| l.arities[0]);
        {
            let is_top = level.len() == 1;
            let refs: Vec<_> = kids.iter().take(top_arity).collect();
            let top = plan.node_program(&refs, is_top).expect("the node emits");
            let rows = ChipRows::of(&top);
            let a = super::registry::census_artifacts_merged(
                &top,
                top.hasher(<P1Family as ProofFamily>::SOCKET),
                on,
            );
            let (n, cells) = airs_and_cells(&a, &wrap);
            println!(
                "MERGE CENSUS {label} {} over {top_arity}: socket {} · {rows:?} · {n} AIRs · {:.1} M cells · merged {:?} · heights {:?}",
                if is_top { "top" } else { "first node" },
                rows.hash,
                cells as f64 / 1e6,
                a.merge.plan(),
                a.log_heights
            );
        }
    }
}

// ============================= review hardening ============================

/// ★ The merge's well-formedness is checked on EVERY path that
/// builds a merged AIR set — `LfmAirs::for_artifacts` (the leg's derivation and
/// the harvest), the host verifier and the prover — against the artifacts'
/// heights: a one-chip group, chips out of slot order, a chip in two merged
/// AIRs and a chip at another height than its merged AIR are each refused, and
/// nothing is built.
#[test]
fn a_malformed_merge_is_refused_on_every_path() {
    let _g = lock();
    let (program, arenas) = four_chip_program();
    let opts = options(true);
    let good = build_artifacts(&program, &opts);
    let proof = lfm_prove(&program, &good, &arenas, &opts).expect("proves");
    assert!(verifies(&good, &proof, &opts));
    let t = &good.merge.tables[0];
    let with = |tables: Vec<MergedTable>| LfmArtifacts {
        merge: LfmMerge { tables },
        ..good.clone()
    };
    let lone = MergedTable {
        chips: vec![MergeChip::Balu],
        ..t.clone()
    };
    let rest = MergedTable {
        chips: t.chips[1..].to_vec(),
        ..t.clone()
    };
    let swapped = MergedTable {
        chips: vec![
            MergeChip::Select,
            MergeChip::Balu,
            MergeChip::Lanes,
            MergeChip::Hint,
        ],
        ..t.clone()
    };
    let taller = MergedTable {
        log_height: t.log_height + 1,
        ..t.clone()
    };
    let cases = [
        ("a one-chip group", with(vec![lone, rest])),
        ("chips out of slot order", with(vec![swapped])),
        (
            "a chip in two merged AIRs",
            with(vec![t.clone(), t.clone()]),
        ),
        ("a chip at another height", with(vec![taller])),
    ];
    for (what, bad) in &cases {
        let err = LfmAirs::for_artifacts(bad, &opts).err();
        assert!(err.is_some(), "{what}: the AIR set must be refused");
        println!("MERGE REFUSED {what}: {}", err.unwrap_or_default());
        assert!(
            super::per_table_aggregator::DerivedChild::from_artifacts(bad, &opts, 5).is_err(),
            "{what}: the leg's derivation must refuse it"
        );
        assert!(!verifies(bad, &proof, &opts), "{what}: the host verifier");
        let exec = execute(&program, &arenas, &bad.hasher).expect("executes");
        let mut traces = build_traces_with_hasher(&program, &exec.records, bad.hasher);
        assert!(
            prove_traces(bad, &mut traces, &exec.public_words, &opts).is_err(),
            "{what}: the prover"
        );
    }
}

/// ★ The bus remap refuses, in every build, a packing that the
/// merged layout would split: a two-column packing over a part's last prefix
/// column and its first value column lands on non-adjacent merged columns.
#[test]
fn a_packing_split_by_the_layout_is_refused() {
    use stark::lookup::{BusInteraction, BusValue, Multiplicity, Packing};
    let layout = MergedLayout::of(&[MergeChip::Balu, MergeChip::Select]);
    let balu = &layout.parts[0];
    let prep = PartShape::of(MergeChip::Balu).prep;
    let straddle = BusInteraction::sender(
        7u64,
        Multiplicity::One,
        vec![BusValue::Packed {
            start_column: prep - 1,
            packing: Packing::Word2L,
        }],
    );
    assert!(super::merge::remap_interaction(&straddle, &balu.map).is_err());
    let inside = BusInteraction::sender(
        7u64,
        Multiplicity::One,
        vec![BusValue::Packed {
            start_column: prep,
            packing: Packing::Word2L,
        }],
    );
    let ok = super::merge::remap_interaction(&inside, &balu.map).expect("contiguous");
    assert_eq!(ok.columns_read(), vec![balu.map[prep], balu.map[prep] + 1]);
}

/// A body the remap could not translate faithfully.
struct ReadsAux;
impl ConstraintSet<F, E> for ReadsAux {
    fn eval<B: ConstraintBuilder<F, E>>(&self, b: &mut B) {
        let a = b.aux(0, 0);
        let m = b.main(0, 0);
        b.emit_ext(0, m * a);
    }
}
struct ReadsPastWidth;
impl ConstraintSet<F, E> for ReadsPastWidth {
    fn eval<B: ConstraintBuilder<F, E>>(&self, b: &mut B) {
        let m = b.main(0, 99);
        b.emit_base(0, m);
    }
}
struct Sparse;
impl ConstraintSet<F, E> for Sparse {
    fn eval<B: ConstraintBuilder<F, E>>(&self, b: &mut B) {
        let m = b.main(0, 0);
        b.emit_base(1, m);
    }
}

/// ★ The guard behind `Remap`'s pass-through leaves: the audit
/// every merged AIR build runs accepts the four candidates and refuses a body
/// that reads an aux column (or emits an extension constraint), one that reads
/// past its width, and one whose indices are not dense.
#[test]
fn the_body_audit_refuses_what_the_remap_cannot_translate() {
    for chip in MERGE_CANDIDATES {
        super::merge::audit_body(PartShape::of(chip)).expect("a candidate passes");
    }
    assert!(super::merge::audit_set(&ReadsAux, 4, "aux").is_err());
    assert!(super::merge::audit_set(&ReadsPastWidth, 4, "width").is_err());
    assert!(super::merge::audit_set(&Sparse, 4, "sparse").is_err());
}

/// The commitment-time answers a merged AIR must carry: every part's value
/// columns, counted from the parts' own shapes (not from the AIR the answer set
/// is computed from).
fn answers_of(chips: &[MergeChip]) -> usize {
    chips
        .iter()
        .map(|&c| {
            let s = PartShape::of(c);
            s.width - s.prep
        })
        .sum()
}

/// For every merged AIR of `artifacts` (in AIR order), its sub-proof index.
fn merged_positions(artifacts: &LfmArtifacts, opts: &ProofOptions) -> Vec<usize> {
    let airs = LfmAirs::for_artifacts(artifacts, opts).expect("the AIR set");
    airs.air_refs()
        .iter()
        .enumerate()
        .filter(|(_, a)| a.name().starts_with("LFM_MERGED"))
        .map(|(i, _)| i)
        .collect()
}

/// ★ A merged AIR answers, at the commitment-time point, exactly
/// its parts' value columns — 29 for the four candidates — and a false answer
/// at its last answered column is refused.
#[test]
fn a_merged_air_answers_every_parts_value_columns() {
    let _g = lock();
    let (program, arenas) = four_chip_program();
    let opts = options(true);
    let artifacts = build_artifacts(&program, &opts);
    let proof = lfm_prove(&program, &artifacts, &arenas, &opts).expect("proves");
    let at = merged_positions(&artifacts, &opts);
    assert_eq!(at.len(), 1);
    let answers = &proof.proof.commit_ood[at[0]];
    assert_eq!(answers.len(), answers_of(&MERGE_CANDIDATES));
    assert_eq!(answers.len(), 29);
    let mut bad = proof;
    let last = bad.proof.commit_ood[at[0]].len() - 1;
    bad.proof.commit_ood[at[0]][last] += FEE::from(1u64);
    assert!(!verifies(&artifacts, &bad, &opts), "a false last answer");
}

/// A program whose SELECT and HINT sit at eight rows and BALU and LANES at the
/// four-row floor: the two-merged-AIR shape of a node or a top.
fn two_group_program() -> (LfmProgram, Vec<Vec<LfmWord>>) {
    let mut b = LfmBuilder::new();
    let arena = b.declare_arena(6);
    let felts: Vec<_> = (0..6).map(|i| b.hint_felt(arena, i)).collect();
    let s = b.add(felts[0], felts[1]);
    b.public(s.as_cell());
    for i in 0..6 {
        let bit = b.bit_const(i % 2 == 0);
        let (l, r) = b.select(bit, felts[i].as_cell(), felts[(i + 1) % 6].as_cell());
        b.public(l);
        b.public(r);
    }
    let da = b.digest_const([fe(1), fe(2), fe(3), fe(4)]);
    let lanes = b.unpack(da.as_cell());
    let packed = b.pack_word(lanes);
    b.public(packed);
    let words = (0..6u64).map(|i| base_word(fe(100 + i))).collect();
    (compile(b.finish()), vec![words])
}

/// ★ Two merged AIRs in one proof — {BALU, LANES} and
/// {SELECT, HINT}, the node and top shape: the plan, the round trip, the
/// cross refusals, two AIRs fewer than plain, and each merged AIR's answers.
#[test]
fn two_merged_airs_in_one_proof_prove_and_verify() {
    let _g = lock();
    let (program, arenas) = two_group_program();
    let (on, off) = (options(true), options(false));
    let merged = build_artifacts(&program, &on);
    let plain = build_artifacts(&program, &off);
    assert_eq!(
        merged.merge.plan(),
        vec![
            vec![MergeChip::Balu, MergeChip::Lanes],
            vec![MergeChip::Select, MergeChip::Hint]
        ]
    );
    let p_merged = lfm_prove(&program, &merged, &arenas, &on).expect("merged proves");
    let p_plain = lfm_prove(&program, &plain, &arenas, &off).expect("plain proves");
    assert_eq!(sub_proofs(&p_merged) + 2, sub_proofs(&p_plain));
    assert!(verifies(&merged, &p_merged, &on));
    assert!(verifies(&plain, &p_plain, &off));
    assert!(!verifies(&plain, &p_merged, &off));
    assert!(!verifies(&merged, &p_plain, &on));
    let at = merged_positions(&merged, &on);
    assert_eq!(at.len(), 2);
    for (pos, table) in at.iter().zip(&merged.merge.tables) {
        assert_eq!(
            p_merged.proof.commit_ood[*pos].len(),
            answers_of(&table.chips),
            "{:?}",
            table.chips
        );
    }
    assert_eq!(
        (
            p_merged.proof.commit_ood[at[0]].len(),
            p_merged.proof.commit_ood[at[1]].len()
        ),
        (8, 21)
    );
}

/// ★ Negative, then the mutant, for BALU: an add nobody reads
/// (write multiplicity zero, so no memory token moves) given a wrong output
/// inside the merged table is refused — and with BALU's body vacuous in the
/// merged AIR, the same table proves and VERIFIES. BALU's body is what refuses
/// it.
#[test]
fn a_bad_balu_row_in_a_merged_table_is_refused_by_balus_own_body() {
    let _g = lock();
    let (program, arenas) = four_chip_program();
    let opts = options(true);
    let artifacts = build_artifacts(&program, &opts);
    let g = &program.groups.balu;
    let row = (0..g.real_rows)
        .find(|&r| {
            *g.at(r, super::layout::balu::MULT) == fe(0)
                && *g.at(r, super::layout::balu::SEL_ADD) == fe(1)
        })
        .expect("the unread add");
    let bad = || {
        let exec = execute(&program, &arenas, &artifacts.hasher).expect("executes");
        let plan = artifacts.merge.plan();
        let mut traces = build_traces_with_merge(&program, &exec.records, artifacts.hasher, &plan);
        let col = MergedLayout::of(&plan[0]).parts[0].map[super::chips::balu::cols::OUT];
        let v = *traces.merged[0].get_main(row, col);
        traces.merged[0].set_main(row, col, v + fe(1));
        (traces, exec.public_words)
    };
    let (mut traces, words) = bad();
    let refused = match prove_traces(&artifacts, &mut traces, &words, &opts) {
        Ok(proof) => !verify_against_artifacts(&artifacts, &proof, &words, &opts),
        Err(_) => true,
    };
    assert!(refused, "the bad BALU row must not verify");

    test_hooks::set_vacuous(Some(PartBody::Balu));
    let mutant = std::panic::catch_unwind(|| {
        let (mut traces, words) = bad();
        let proof =
            prove_traces(&artifacts, &mut traces, &words, &opts).expect("the mutant proves");
        verify_against_artifacts(&artifacts, &proof, &words, &opts)
    });
    test_hooks::set_vacuous(None);
    assert!(
        mutant.expect("the mutant runs"),
        "with BALU's body vacuous the bad row verifies: the body is load-bearing"
    );
}

/// ★ The merged root is the program's identity: two programs
/// that differ only in one merged chip's instruction columns (an add's
/// operands swapped, same heights) get different ids, and so does the same
/// program with its merged root moved.
#[test]
fn the_merged_root_is_folded_into_the_program_id() {
    let _g = lock();
    let opts = options(true);
    let build = |swap: bool| {
        let mut b = LfmBuilder::new();
        let arena = b.declare_arena(1);
        let h0 = b.hint_felt(arena, 0);
        let x = b.felt_const(fe(7));
        let y = b.felt_const(fe(5));
        let s = if swap { b.add(y, x) } else { b.add(x, y) };
        let d = b.sub(s, h0);
        let bit = b.bit_const(true);
        let (l, _) = b.select(bit, x.as_cell(), d.as_cell());
        let lanes = b.unpack(l);
        let packed = b.pack_word(lanes);
        b.public(packed);
        build_artifacts(&compile(b.finish()), &opts)
    };
    let (a, b) = (build(false), build(true));
    assert_eq!(a.log_heights, b.log_heights, "the same heights");
    assert_eq!(a.merge.plan(), b.merge.plan());
    assert_ne!(
        a.merge.tables[0].root, b.merge.tables[0].root,
        "BALU's columns differ"
    );
    for c in MERGE_CANDIDATES {
        assert_eq!(a.roots[c.slot()], b.roots[c.slot()], "both zero");
    }
    assert_ne!(a.program_id, b.program_id, "the merged root is identity");
    let mut moved = a.merge.clone();
    moved.tables[0].root[0] ^= 1;
    let id = |merge: &LfmMerge| {
        super::statement::lfm_program_id_merged_under(
            crate::hash_pin::BLOCK_COMMITMENT_HASH,
            &a.roots,
            &a.log_heights,
            a.keccak_rnd_chunks,
            a.hasher,
            a.chip_set,
            &a.blake3_chunk_roots,
            &a.blake3_chunk_log_heights,
            &a.hash_chunk_roots,
            &a.hash_chunk_log_heights,
            merge,
        )
    };
    assert_ne!(id(&a.merge), id(&moved));
}
