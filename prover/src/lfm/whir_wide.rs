//! ★ The WIDE level-1 node: its epochs verified in-program, no wraps (D-WHIR
//! §7 L1; STRUCT §2's Design A on the W-LFM tree), behind
//! `LAMBDA_VM_LFM_WIDE=on`.
//!
//! A level-1 node of the WHIR tree verifies `k` wrap PROOFS, binds them and
//! publishes the node schema. A wide node verifies the `k` EPOCHS themselves:
//! each with the wrap's own verifier ([`emit_epoch_leg`]), on its own arena and
//! its own transcript, and then runs the node's own [`emit_chain_bindings`] and
//! [`emit_node_publishes`] over what each wrap WOULD have published
//! ([`epoch_would_publish`]). So it binds exactly what an L1 node binds between
//! its wraps — one attestation id, each register FINI against the next INIT,
//! each epoch label pinned to its tree position — in the same order, and
//! publishes the same schema. A parent binds it as it binds an L1 node, and with
//! `k` = the tree's fan-in the root's fold shape is the one it already computes.
//!
//! ⚠ NOTHING HERE RE-IMPLEMENTS A CHECK. The legs, the bindings and the
//! publishes are the wrap's and the node's functions; what is new is the loop
//! that puts them in one program, and the arena list that loop implies.
//!
//! ⛔ THE LABELS ARE THE TREE'S, NOT THE EPOCHS'. Each epoch's would-be label
//! word is its `position.label`, program text its statement interns; the pins
//! compare it against the position the DRIVER assigns, exactly as a node pins a
//! wrap child. Pinning an epoch's label against itself would be a check that
//! cannot fail.

use super::builder::LfmBuilder;
use super::compiler::LfmProgram;
use super::per_table_aggregator::{
    LegCells, NodePublishSet, NodePublishes, SchemaLayout, emit_chain_bindings, emit_node_publishes,
};
use super::whir_epoch::{EpochAirs, emit_epoch_leg, epoch_would_publish, whir_epoch_arena};
use super::whir_real_epoch::WhirRealEpoch;
use super::word::LfmWord;

/// One epoch of a wide node: the harvested epoch and the AIR set its verifier
/// is emitted against.
#[derive(Clone, Copy)]
pub struct WideEpoch<'a> {
    pub epoch: &'a WhirRealEpoch,
    pub airs: EpochAirs<'a>,
}

/// What a negative test changes about a wide node's emission; production emits
/// with [`WideTamper::default`], which changes nothing.
#[derive(Clone, Debug, Default)]
pub(crate) struct WideTamper {
    /// Emit WITHOUT the chain bindings: the paired control that shows the
    /// bindings are what refuse a broken chain.
    pub(crate) skip_bindings: bool,
    /// Publish this attestation id for epoch `.0` in place of its own.
    pub(crate) id: Option<(usize, [u8; 32])>,
}

/// ★ The wide node, emitted: every epoch's verifier leg, in chain order, then
/// the node's bindings and publishes over their would-be published words.
///
/// `labels[i]` is the tree position of `epochs[i]` — what the pins compare
/// against, and whose first and last the node publishes as its label range.
/// Returns the layout the node published against (a node's, whatever `k`).
pub fn emit_wide_node(
    b: &mut LfmBuilder,
    epochs: &[WideEpoch<'_>],
    labels: &[u64],
    publishes: NodePublishSet,
) -> SchemaLayout {
    emit_wide(b, epochs, labels, publishes, &WideTamper::default())
}

/// [`emit_wide_node`] with a test's [`WideTamper`] applied.
pub(crate) fn emit_wide(
    b: &mut LfmBuilder,
    epochs: &[WideEpoch<'_>],
    labels: &[u64],
    publishes: NodePublishSet,
    tamper: &WideTamper,
) -> SchemaLayout {
    assert!(
        !epochs.is_empty(),
        "a wide node verifies at least one epoch"
    );
    assert_eq!(
        labels.len(),
        epochs.len(),
        "one tree position per epoch of a wide node"
    );

    // ⚠ DECLARATION ORDER IS ARENA ORDER: each leg declares its epoch's one
    // arena first, so the host's arena list is the epochs' in chain order.
    let mut legs: Vec<LegCells> = Vec::with_capacity(epochs.len());
    let mut layouts: Vec<SchemaLayout> = Vec::with_capacity(epochs.len());
    for (i, e) in epochs.iter().enumerate() {
        let wires = emit_epoch_leg(b, e.epoch, e.airs);
        let mut published = wires.publishes(e.epoch);
        if let Some((at, id)) = tamper.id
            && at == i
        {
            published.text.program_id = id;
        }
        let (layout, publics) = epoch_would_publish(b, &published);
        legs.push(LegCells {
            publics,
            z_alpha: (wires.z, wires.alpha),
        });
        layouts.push(layout);
    }

    let positions: Vec<[u64; 1]> = labels.iter().map(|label| [*label]).collect();
    let position_refs: Vec<&[u64]> = positions.iter().map(|p| &p[..]).collect();
    if !tamper.skip_bindings {
        emit_chain_bindings(b, &legs, &layouts, &position_refs);
    }
    emit_node_publishes(
        b,
        &NodePublishes {
            legs: &legs,
            layouts: &layouts,
            label_range: (labels[0], labels[labels.len() - 1]),
        },
    );
    // The differential surface, after the schema, as `emit_node` places it.
    if publishes == NodePublishSet::Diagnostic {
        for leg in &legs {
            b.public(leg.z_alpha.0.as_cell());
            b.public(leg.z_alpha.1.as_cell());
        }
    }
    SchemaLayout::node(layouts.last().expect("nonempty").out_halves)
}

/// The wide node as one program, compiled, its published count pinned and
/// admitted.
pub fn wide_node_program(
    epochs: &[WideEpoch<'_>],
    labels: &[u64],
    publishes: NodePublishSet,
) -> LfmProgram {
    wide_program(epochs, labels, publishes, &WideTamper::default())
}

/// [`wide_node_program`] with a test's [`WideTamper`] applied.
pub(crate) fn wide_program(
    epochs: &[WideEpoch<'_>],
    labels: &[u64],
    publishes: NodePublishSet,
    tamper: &WideTamper,
) -> LfmProgram {
    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let layout = emit_wide(&mut b, epochs, labels, publishes, tamper);
    let program = super::compiler::compile(b.finish());
    if publishes == NodePublishSet::Aggregation {
        layout.assert_covers(program.public_len as usize);
    }
    super::validator::validate(&program).expect("a wide node must be admissible");
    program
}

/// The wide node's arenas: each epoch's ONE arena ([`whir_epoch_arena`]), in
/// chain order — the order [`emit_wide_node`] declares them in.
pub fn wide_node_arena(epochs: &[WideEpoch<'_>]) -> Vec<Vec<LfmWord>> {
    epochs
        .iter()
        .flat_map(|e| whir_epoch_arena(e.epoch, e.airs))
        .collect()
}
