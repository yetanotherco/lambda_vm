//! A batched argue's GKR tree for interactions that read a public selector,
//! built on the card from a small store of the columns it reads and its
//! selectors (`multilinear::logup::resident_tree_with_public`,
//! `LAMBDA_VM_ARGUE_PUBLIC_COLUMNS`), is the tree the host builds from the
//! table's lifted factors: the same output and the same layers. The selectors
//! are the trace's public tables, which are `Selector::table` — the same
//! selector the verifier evaluates in closed form. Box tier: they need the
//! card, and refuse to run without one.

use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;
use math::field::goldilocks::GoldilocksField as Gl;
use multilinear::constraint_argument::{FactorKind, TraceData};
use multilinear::gkr::FractionTree;
use multilinear::logup::{Affine, Interaction, input_layer};
use multilinear::mle::Mle;
use multilinear::selector::Selector;

type EE = FieldElement<Ext3>;

fn next(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

/// A table of six columns at 2^12 rows, with a shifted factor and two
/// end-exemption selectors as its public factors, resident on the card; and
/// a chunk of interactions reading columns, the shifted factor and both
/// selectors.
fn table_and_chunk() -> (TraceData<Gl, Ext3>, Vec<Interaction<Ext3>>) {
    let num_vars = 12usize;
    let rows = 1usize << num_vars;
    let mut seed = 0x51EC_7012_u64;
    let columns: Vec<Mle<Gl>> = (0..6)
        .map(|_| {
            Mle::new(
                (0..rows)
                    .map(|_| FieldElement::<Gl>::from(next(&mut seed) % 0xFFFF_FFFF_0000_0001))
                    .collect(),
            )
            .unwrap()
        })
        .collect();
    let mut kinds: Vec<FactorKind> = (0..6).map(FactorKind::direct).collect();
    kinds.push(FactorKind::Public); // slot 6: all but the last row
    kinds.push(FactorKind::shifted(3, 1)); // slot 7
    kinds.push(FactorKind::Public); // slot 8: all but the last 3 rows
    let public = vec![
        Selector::except_last(1).table::<Ext3>(num_vars).unwrap(),
        Selector::except_last(3).table::<Ext3>(num_vars).unwrap(),
    ];
    let mut trace = TraceData::<Gl, Ext3>::new(columns.clone(), kinds, public).unwrap();
    let store = multilinear::gpu::upload_tables(&[multilinear::gpu::TableColumns::Wide(&columns)])
        .expect("the card takes the table's columns");
    trace.set_resident(std::sync::Arc::new(store), 0);
    let ext = |seed: &mut u64| EE::from(next(seed));
    let chunk = vec![
        Interaction::new(
            Affine::factor(6),
            Affine::new(
                vec![(0, ext(&mut seed)), (7, ext(&mut seed))],
                ext(&mut seed),
            ),
        ),
        Interaction::new(
            Affine::new(
                vec![(8, ext(&mut seed)), (2, ext(&mut seed))],
                ext(&mut seed),
            ),
            Affine::new(
                vec![(5, ext(&mut seed)), (6, ext(&mut seed))],
                ext(&mut seed),
            ),
        ),
        Interaction::new(
            Affine::factor(8),
            Affine::new(vec![(1, ext(&mut seed))], ext(&mut seed)),
        ),
    ];
    (trace, chunk)
}

/// Every layer the card holds (all but an input layer it handed back by
/// design) against the host reference's, and the outputs: how many layers
/// compared. Both trees' layers are read through `layer_copy`, never indexed:
/// a tree whose layers live on the card keeps none here.
fn compare(card: &FractionTree<Ext3>, host: &FractionTree<Ext3>) -> Result<usize, String> {
    if card.num_layers() != host.num_layers() {
        return Err(format!(
            "{} layers against {}",
            card.num_layers(),
            host.num_layers()
        ));
    }
    let mut compared = 0usize;
    for i in 0..host.num_layers() {
        let want = host
            .layer_copy(i)
            .ok_or_else(|| format!("the host reference has no layer {i}"))?;
        match card.layer_copy(i) {
            Some(layer) => {
                if layer.p.evals() != want.p.evals() || layer.q.evals() != want.q.evals() {
                    return Err(format!("layer {i} differs"));
                }
                compared += 1;
            }
            None if i + 1 == host.num_layers() && card.device().is_some() => {}
            None => return Err(format!("the card has no layer {i}")),
        }
    }
    if card.output() != host.output() {
        return Err("the outputs differ".into());
    }
    Ok(compared)
}

/// The host reference: the tree folded on the host whatever devices there
/// are, every layer kept here.
fn host_tree(trace: &TraceData<Gl, Ext3>, chunk: &[Interaction<Ext3>]) -> FractionTree<Ext3> {
    FractionTree::build_on_host(input_layer(chunk, &trace.factors().unwrap()).unwrap()).unwrap()
}

fn trees(shifted: bool) -> (FractionTree<Ext3>, FractionTree<Ext3>) {
    assert!(
        multilinear::gpu::reserve_budget() > 0,
        "this test needs the card (box tier)"
    );
    let (trace, chunk) = table_and_chunk();
    let host = host_tree(&trace, &chunk);
    multilinear::logup::fault_public_selector_shift(shifted);
    let card = multilinear::logup::resident_tree_with_public(&chunk, &trace);
    multilinear::logup::fault_public_selector_shift(false);
    let card = card.expect("the card builds the public-reading tree");
    assert!(card.device().is_some(), "the tree lives on the card");
    (card, host)
}

/// ★ The card's public-reading tree is the host's: the same output and every
/// layer the card holds.
#[test]
#[ignore = "box tier: needs the card"]
fn a_public_reading_tree_on_the_card_is_the_hosts() {
    let (card, host) = trees(false);
    let compared = compare(&card, &host).unwrap_or_else(|e| panic!("{e}"));
    assert!(
        compared + 1 >= host.num_layers(),
        "{compared} of {} layers compared",
        host.num_layers()
    );
    println!(
        "PUBLIC TREE IDENTITY: {compared} of {} layers equal, outputs equal",
        host.num_layers()
    );
}

/// The check sees a selector one row off, and only it: the honest card tree
/// compares equal in full (every layer read), and with a test's fault moving
/// each selector one row the comparison reports a differing layer or output.
#[test]
#[ignore = "box tier: needs the card"]
fn a_selector_one_row_off_builds_another_tree() {
    let (card, host) = trees(false);
    let compared = compare(&card, &host).unwrap_or_else(|e| panic!("the honest tree: {e}"));
    assert!(
        compared + 1 >= host.num_layers(),
        "the honest comparison read {compared} layers"
    );
    let (card, host) = trees(true);
    let err = compare(&card, &host).expect_err("a selector one row off went unseen");
    assert!(
        err.contains("differs"),
        "the shifted tree was refused for another reason: {err}"
    );
}

/// On the laptop: the comparison never indexes a tree's layers directly. A
/// stand-in for a card tree whose layers live elsewhere (its host `layers`
/// empty, no device) is refused with a reason, never a panic; a tree compares
/// equal to the same tree folded again, every layer read; a tree folded from
/// another chunk is told apart.
#[test]
fn the_comparison_reads_layers_without_indexing_them() {
    let num_vars = 6usize;
    let rows = 1usize << num_vars;
    let mut seed = 0xFEED_u64;
    let columns: Vec<Mle<Gl>> = (0..3)
        .map(|_| {
            Mle::new(
                (0..rows)
                    .map(|_| FieldElement::<Gl>::from(next(&mut seed) % 0xFFFF_FFFF_0000_0001))
                    .collect(),
            )
            .unwrap()
        })
        .collect();
    let mut kinds: Vec<FactorKind> = (0..3).map(FactorKind::direct).collect();
    kinds.push(FactorKind::Public);
    let public = vec![Selector::except_last(1).table::<Ext3>(num_vars).unwrap()];
    let trace = TraceData::<Gl, Ext3>::new(columns, kinds, public).unwrap();
    let chunk = vec![Interaction::new(
        Affine::factor(3),
        Affine::new(
            vec![(0, EE::from(5u64)), (2, EE::from(7u64))],
            EE::from(11u64),
        ),
    )];
    let host = host_tree(&trace, &chunk);
    assert_eq!(
        compare(&host_tree(&trace, &chunk), &host),
        Ok(host.num_layers())
    );
    let empty = host_tree(&trace, &chunk).without_host_layers();
    let err = compare(&empty, &host).expect_err("an empty stand-in compares nothing");
    assert!(err.contains("has no layer"), "{err}");
    // The reference side too: the bug a card run met was a reference built
    // on the card, its host layers empty, indexed directly.
    let empty = host_tree(&trace, &chunk).without_host_layers();
    let err = compare(&host, &empty).expect_err("an empty reference compares nothing");
    assert!(err.contains("the host reference has no layer"), "{err}");
    let other = vec![Interaction::new(
        Affine::factor(1),
        Affine::new(vec![(2, EE::from(3u64))], EE::from(9u64)),
    )];
    assert!(compare(&host_tree(&trace, &other), &host).is_err());
}
