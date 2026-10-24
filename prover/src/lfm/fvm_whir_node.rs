//! An LFM program that verifies a Field VM proof in the block WHIR format
//! ([`prove_whir_block`](crate::field_vm::whir::prove_whir_block)), through the
//! block leaf's own emitters: one leaf over all of the proof's groups, its bus
//! target the Field VM's public cells and its start/halt states.
//!
//! Field VM multilinear proofs bind their public endpoints through buses, so
//! the recursion AIRs erase the STARK public-input type and expose no boundary
//! constraints ([`UnitPi`]).

use crypto::fiat_shamir::is_transcript::IsStarkTranscript;
use math::field::element::FieldElement;
use stark::{
    config::Commitment,
    constraint_ir::ConstraintProgram,
    constraints::{boundary::BoundaryConstraints, builder::ConstraintMeta},
    context::AirContext,
    lookup::{BusInteraction, BusPublicInputs},
    proof::options::ProofOptions,
    trace::TraceTable,
    traits::{AIR, TransitionEvaluationContext},
};

use crate::block_whir::{
    BlockFormat, BlockFrame, FrameAirs, commit_group, group_stacks, validate_groups,
};
use crate::field_vm::cksum;
use crate::field_vm::decode;
use crate::field_vm::isa::Program;
use crate::field_vm::prove::{ExtraAir, MIN_ROWS, PublicCell, public_well_formed};
use crate::field_vm::whir::{
    AirBox, FvmWhirBlockProof, WhirAirs, block_statement_calls, block_statement_value_call,
    cksum_air, prepared_tables,
};
use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};

use super::block_node::BlockLayout;
use super::compiler::LfmProgram;
use super::per_table_aggregator::ChildShape;
use super::whir_block::{
    FvmBinding, FvmCell, LeafTarget, PlannedPrepared, WhirBlockPlan, plan_leaf_arena,
};
use super::word::LfmWord;

type F = GoldilocksField;
type E = GoldilocksExtension;

/// A Field VM AIR with its statement inputs erased. For the multilinear WHIR
/// path only: segment endpoints are closed by the FieldVmState bus and MEM's
/// address ramp is preprocessed.
pub(crate) struct UnitPi {
    inner: AirBox,
}

impl AIR for UnitPi {
    type Field = F;
    type FieldExtension = E;
    type PublicInputs = ();

    fn step_size(&self) -> usize {
        self.inner.step_size()
    }
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn new(_: &ProofOptions) -> Self {
        panic!("UnitPi wraps an existing AIR")
    }
    fn build_auxiliary_trace(
        &self,
        trace: &mut TraceTable<F, E>,
        challenges: &[FieldElement<E>],
    ) -> Option<BusPublicInputs<E>> {
        self.inner.build_auxiliary_trace(trace, challenges)
    }
    fn build_rap_challenges(
        &self,
        transcript: &mut dyn IsStarkTranscript<E, F>,
    ) -> Vec<FieldElement<E>> {
        self.inner.build_rap_challenges(transcript)
    }
    fn trace_layout(&self) -> (usize, usize) {
        self.inner.trace_layout()
    }
    fn has_trace_interaction(&self) -> bool {
        self.inner.has_trace_interaction()
    }
    fn max_bus_elements(&self) -> usize {
        self.inner.max_bus_elements()
    }
    fn bus_interactions(&self) -> &[BusInteraction] {
        self.inner.bus_interactions()
    }
    fn max_constraint_degree(&self) -> usize {
        self.inner.max_constraint_degree()
    }
    fn is_preprocessed(&self) -> bool {
        self.inner.is_preprocessed()
    }
    fn num_precomputed_columns(&self) -> usize {
        self.inner.num_precomputed_columns()
    }
    fn precomputed_commitment(&self) -> Commitment {
        self.inner.precomputed_commitment()
    }
    fn precomputed_commitment_for(
        &self,
        layout: stark::leaf_layout::LeafLayout,
    ) -> Option<Commitment> {
        self.inner.precomputed_commitment_for(layout)
    }
    fn precomputed_columns(&self) -> Vec<Vec<FieldElement<F>>> {
        self.inner.precomputed_columns()
    }
    fn num_auxiliary_rap_columns(&self) -> usize {
        self.inner.num_auxiliary_rap_columns()
    }
    fn trace_ood_next_row_columns(&self) -> Vec<usize> {
        self.inner.trace_ood_next_row_columns()
    }
    fn composition_poly_degree_bound(&self, n: usize) -> usize {
        self.inner.composition_poly_degree_bound(n)
    }
    fn compute_transition(&self, ctx: &TransitionEvaluationContext<F, E>) -> Vec<FieldElement<E>> {
        self.inner.compute_transition(ctx)
    }
    fn num_base_transition_constraints(&self) -> usize {
        self.inner.num_base_transition_constraints()
    }
    fn compute_transition_prover(
        &self,
        ctx: &TransitionEvaluationContext<F, E>,
        base: &mut [FieldElement<F>],
        ext: &mut [FieldElement<E>],
    ) {
        self.inner.compute_transition_prover(ctx, base, ext)
    }
    fn constraints_meta(&self) -> &[ConstraintMeta] {
        self.inner.constraints_meta()
    }
    fn constraint_program(&self) -> &ConstraintProgram<F, E> {
        self.inner.constraint_program()
    }
    fn precaptured_constraint_program(&self) -> Option<&ConstraintProgram<F, E>> {
        self.inner.precaptured_constraint_program()
    }
    fn boundary_constraints(
        &self,
        _: &(),
        _: &[FieldElement<E>],
        _: Option<&BusPublicInputs<E>>,
        _: usize,
    ) -> BoundaryConstraints<E> {
        BoundaryConstraints::from_constraints(Vec::new())
    }
    fn context(&self) -> &AirContext {
        self.inner.context()
    }
}

/// The one-leaf plan verifying `proof`: the frame from the Field VM's AIRs
/// and the proof's stated heights and groups, the prepared DECODE stacks
/// committed from the program, the block statement, the bus target.
pub(crate) fn fvm_whir_plan(
    program: &Program,
    program_id: &[Commitment],
    public: &[PublicCell],
    proof: &FvmWhirBlockProof,
    options: &ProofOptions,
) -> Result<WhirBlockPlan, String> {
    let cells: Vec<(u64, FvmCell)> = public
        .iter()
        .map(|(a, v)| (*a, FvmCell::Const(*v.value())))
        .collect();
    plan_with(
        program,
        program_id,
        &cells,
        proof,
        options,
        Vec::new(),
        None,
    )
}

/// [`fvm_whir_plan`] for a checksum-bound proof
/// ([`prove_whir_block_bound`](crate::field_vm::whir::prove_whir_block_bound)),
/// as `verify_whir_block_bound` checks it: the CKSUM table, `s` closed at
/// `CKSUM_ADDR` and `α` at `CKSUM_ALPHA_ADDR`. Nothing of the witness is in the
/// leaf: it reads `c_bin`, `s` and the cells at `inputs` from its arena
/// ([`bound_arena_prefix`]) and publishes them, and computes `α` from `c_bin`
/// and the proof's non-CKSUM roots. `public`'s other cells are the program's
/// constants; its values at `inputs` are not read.
pub(crate) fn fvm_whir_plan_bound(
    program: &Program,
    program_id: &[Commitment],
    public: &[PublicCell],
    inputs: &[u64],
    faddr: &[u64],
    proof: &FvmWhirBlockProof,
    options: &ProofOptions,
) -> Result<WhirBlockPlan, String> {
    let decode_count = decode::chunks(program, MIN_ROWS).len();
    if faddr.is_empty()
        || proof.groups.last().map(Vec::as_slice)
            != Some(&[proof.table_num_vars.len().saturating_sub(1) as u32][..])
        || proof.table_num_vars.len() != proof.segments.len() + decode_count + 2
        || public.iter().any(|(addr, _)| *addr >= cksum::CKSUM_ADDR)
        || inputs
            .iter()
            .any(|a| public.binary_search_by_key(a, |c| c.0).is_err())
    {
        return Err("not a checksum-bound proof of this program".into());
    }
    let root = cksum::commitment(faddr, options, 8);
    let air = cksum_air(root, faddr.to_vec(), options, 8);
    let mut cells: Vec<(u64, FvmCell)> = public
        .iter()
        .map(|(a, v)| {
            let cell = if inputs.contains(a) {
                FvmCell::Input
            } else {
                FvmCell::Const(*v.value())
            };
            (*a, cell)
        })
        .collect();
    cells.push((cksum::CKSUM_ADDR, FvmCell::Input));
    cells.push((cksum::CKSUM_ALPHA_ADDR, FvmCell::Alpha));
    plan_with(
        program,
        program_id,
        &cells,
        proof,
        options,
        vec![air],
        Some(FvmBinding { roots: 0 }),
    )
}

/// A bound v_fld leaf's arena prefix: `c_bin`, then the value of each public
/// cell the plan reads, from `public` (the proof's public cells with `s`).
pub(crate) fn bound_arena_prefix(
    plan: &WhirBlockPlan,
    c_bin: [u64; 4],
    public: &[PublicCell],
) -> Result<Vec<LfmWord>, String> {
    let LeafTarget::FieldVm { public: cells, .. } = &plan.target else {
        return Err("not a Field VM plan".into());
    };
    let mut words = vec![c_bin.map(FE::from)];
    for (addr, cell) in cells {
        if matches!(cell, FvmCell::Input) {
            let value = public
                .iter()
                .find(|c| c.0 == *addr)
                .ok_or_else(|| format!("no value for input cell {addr}"))?;
            words.push(super::word::ext_word(&value.1));
        }
    }
    Ok(words)
}

fn plan_with(
    program: &Program,
    program_id: &[Commitment],
    public: &[(u64, FvmCell)],
    proof: &FvmWhirBlockProof,
    options: &ProofOptions,
    extra: Vec<ExtraAir>,
    binding: Option<FvmBinding>,
) -> Result<WhirBlockPlan, String> {
    let stated: Vec<PublicCell> = public
        .iter()
        .map(|(a, cell)| match cell {
            FvmCell::Const(v) => (*a, FEE::new(*v)),
            FvmCell::Input | FvmCell::Alpha => (*a, FEE::zero()),
        })
        .collect();
    if !public_well_formed(&stated) {
        return Err("malformed public cells".into());
    }
    let runtime_calls: Vec<usize> = public
        .iter()
        .enumerate()
        .filter(|(_, (_, cell))| !matches!(cell, FvmCell::Const(_)))
        .flat_map(|(i, _)| (0..3).map(move |k| block_statement_value_call(program_id.len(), i, k)))
        .collect();
    let segments = proof.segments.len();
    let decode_count = decode::chunks(program, MIN_ROWS).len();
    if program_id.len() != decode_count || segments == 0 {
        return Err("the program id or the segments do not fit the program".into());
    }
    let mem_vars = *proof
        .table_num_vars
        .get(segments + decode_count)
        .ok_or("the proof states no MEM height")?;
    let airs = WhirAirs::new(
        program,
        program_id,
        segments,
        1usize << mem_vars,
        options,
        extra,
    );
    if airs.airs.len() != proof.table_num_vars.len() {
        return Err("the proof states another table count".into());
    }
    let shapes: Vec<(usize, usize)> = airs
        .airs
        .iter()
        .zip(&proof.table_num_vars)
        .map(|(a, &v)| (a.trace_layout().0, v as usize))
        .collect();
    let format = BlockFormat::production();
    let config = format.chain_config(&shapes);
    let order = validate_groups(&proof.groups, shapes.len()).map_err(|e| format!("{e:?}"))?;
    let sizes: Vec<usize> = proof.groups.iter().map(Vec::len).collect();
    let ordered: Vec<_> = order.iter().map(|&i| shapes[i]).collect();
    let (stack_layouts, domains) = crate::multilinear_prove::stacks(&ordered, &sizes, &config)
        .map_err(|e| format!("{e:?}"))?;

    let refs: Vec<_> = airs.airs.iter().map(|a| a.as_ref()).collect();
    let prepared_columns = prepared_tables(&refs, segments, decode_count);
    drop(refs);
    let stacks = group_stacks(prepared_columns, &proof.groups).map_err(|e| format!("{e:?}"))?;
    let groups: Vec<Vec<usize>> = proof
        .groups
        .iter()
        .map(|g| g.iter().map(|&t| t as usize).collect())
        .collect();
    let mut prepared = Vec::with_capacity(stacks.len());
    for stack in stacks {
        let group = stack.group;
        let start: usize = groups[..group].iter().map(Vec::len).sum();
        let tables = stack
            .tables
            .iter()
            .map(|&(position, air, n)| (position - start, air, n))
            .collect();
        let layout = stark::multilinear_table::global_layout(&stack.shapes(), config.format.stack)
            .map_err(|e| format!("prepared group {group}: {e:?}"))?;
        let domain = multilinear::whir::Domain::<F>::new(layout.n_stack() + config.log_blowup)
            .map_err(|e| format!("prepared group {group}: {e:?}"))?;
        let roots = commit_group::<crate::hash_pin::BlockWhirHash>(stack, &config)
            .map_err(|e| format!("prepared group {group}: {e:?}"))?
            .roots;
        prepared.push(PlannedPrepared {
            group,
            tables,
            roots,
            layout,
            domain,
        });
    }

    let statement_calls = block_statement_calls(
        program_id,
        &stated,
        &proof.segments,
        &proof.table_num_vars,
        &proof.groups,
        &config,
    );
    let target = LeafTarget::FieldVm {
        public: public.to_vec(),
        first: proof.segments[0].first.clone(),
        last: proof.segments[segments - 1].last.clone(),
        // The roots before CKSUM's group, the last: the binding's digest.
        binding: binding.map(|_| FvmBinding {
            roots: stack_layouts[..stack_layouts.len() - 1]
                .iter()
                .map(|l| l.num_polys())
                .sum(),
        }),
    };
    let frame = BlockFrame {
        page_configs: Vec::new(),
        airs: FrameAirs::Owned(
            airs.airs
                .into_iter()
                .map(|inner| Box::new(UnitPi { inner }) as Box<_>)
                .collect(),
        ),
        shapes,
        config,
        order,
        sizes,
        stack_layouts,
        domains,
    };
    Ok(WhirBlockPlan::for_proof(
        frame,
        statement_calls,
        runtime_calls,
        groups,
        prepared,
        target,
    ))
}

/// The verifier program: the plan's one leaf. It publishes the plan's id, the
/// state digest and, last, the bus sum less the target — zero on a valid proof.
pub(crate) fn emit_fvm_whir_verifier(plan: &WhirBlockPlan) -> Result<LfmProgram, String> {
    plan.leaf_program(0)
}

/// The verifier program's arena: `prefix` (a bound leaf's
/// [`bound_arena_prefix`], else empty), then the proof's roots, openings,
/// argues and prepared openings in the order the leaf hints them.
pub(crate) fn fvm_whir_arena(
    plan: &WhirBlockPlan,
    proof: &FvmWhirBlockProof,
    prefix: Vec<LfmWord>,
) -> Result<Vec<Vec<LfmWord>>, String> {
    let mut arena = plan_leaf_arena(plan, 0, &proof.proof, &proof.argues, &proof.prepared)?;
    arena[0].splice(0..0, prefix);
    Ok(arena)
}

/// The binding's words in the guest's output: `c_bin` (4), `α` (3), `s` (3),
/// each a `u64` of two published `u32` halves, at the output's end.
const BIND_HALVES: usize = 20;

/// Where the binding sits among each half's published words: the v_bin
/// leaf's (id 2, state, its output's halves, sum) and the bound v_fld leaf's
/// (id 2, state, its inputs, `c_bin`, `α`, `s`, sum).
#[derive(Clone, Copy, Debug)]
pub(crate) struct BindShape {
    pub(crate) vbin_words: usize,
    pub(crate) vbin_tail: usize,
    pub(crate) vfld_words: usize,
}

impl BindShape {
    /// The shape for a v_bin output of `output_len` bytes and a v_fld leaf of
    /// `inputs` inputs. The binding words must fall on whole halves.
    pub(crate) fn new(output_len: usize, inputs: usize) -> Result<Self, String> {
        if !output_len.is_multiple_of(4) || output_len < 4 * BIND_HALVES {
            return Err(format!(
                "a v_bin output of {output_len} bytes holds no binding"
            ));
        }
        let halves = output_len / 4;
        Ok(Self {
            vbin_words: 3 + halves + 1,
            vbin_tail: 3 + halves - BIND_HALVES,
            vfld_words: 7 + inputs,
        })
    }
}

/// Option B's binding, checked by the recursion: from the two halves'
/// published words (arena 0: the v_bin leaf's, arena 1: the bound v_fld
/// leaf's), `c_bin`, `α` and `s` of the guest's output equal the v_fld leaf's
/// — whose `α` that leaf recomputed from `c_bin` and its proof's roots — and
/// the v_fld bus sum is zero. Publishes both halves' ids. In a node these
/// words are the children's, bound by their verification.
pub(crate) fn emit_b_bind(shape: BindShape) -> Result<LfmProgram, String> {
    let mut b = super::builder::LfmBuilder::new();
    let bin = b.declare_arena(shape.vbin_words as u32);
    let fld = b.declare_arena(shape.vfld_words as u32);
    let two32 = b.felt_const(FE::from(1u64 << 32));
    let zero = b.felt_const(FE::zero());
    let claimed: Vec<_> = (0..BIND_HALVES / 2)
        .map(|i| {
            let at = (shape.vbin_tail + 2 * i) as u32;
            let lo = b.hint_word(bin, at);
            let lo = b.unpack(lo)[0];
            let hi = b.hint_word(bin, at + 1);
            let hi = b.unpack(hi)[0];
            b.mul_add(hi, two32, lo)
        })
        .collect();
    let base = shape.vfld_words - 4;
    for (word, (lanes, from)) in [(4usize, 0usize), (3, 4), (3, 7)].into_iter().enumerate() {
        let cell = b.hint_word(fld, (base + word) as u32);
        let got = b.unpack(cell);
        for k in 0..lanes {
            b.assert_eq(got[k], claimed[from + k]);
        }
    }
    let sum = b.hint_word(fld, (shape.vfld_words - 1) as u32);
    let sum = b.unpack(sum);
    for lane in sum {
        b.assert_eq(lane, zero);
    }
    for (arena, len) in [(bin, shape.vbin_words), (fld, shape.vfld_words)] {
        debug_assert!(len >= 2);
        for at in 0..2 {
            let id = b.hint_word(arena, at);
            b.public(id);
        }
    }
    let program = super::compiler::compile(b.finish());
    super::validator::validate(&program).map_err(|e| format!("not admissible: {e:?}"))?;
    Ok(program)
}

/// The native guest's output header before the leaf's words: its program
/// digest (32 bytes) and four `u32` region fields, in halves.
const HEADER_HALVES: usize = 12;

/// Where a B node takes each word the block leaf publishes from.
#[derive(Clone, Copy, Debug)]
pub(crate) enum LeafWord {
    /// The guest's `j`-th published word, from its output.
    Guest(usize),
    /// A constant of the pair's programs: the leaf's own (its id, its output
    /// halves).
    Const(LfmWord),
    /// The bound v_fld verifier's `i`-th published input (the leaf's bus sum).
    Field(usize),
}

/// What a B node is a function of: the v_bin half's block leaves (ordinary
/// block leaves over the guest's Lambda VM proof) and their layout, the bound
/// v_fld verifier, and where each word of the block leaf the pair stands for
/// comes from.
pub(crate) struct BNodeInputs<'a> {
    pub(crate) vbin: &'a [ChildShape<'a>],
    pub(crate) vbin_layout: BlockLayout,
    pub(crate) vfld: &'a ChildShape<'a>,
    pub(crate) leaf: &'a [LeafWord],
    /// The generic guest's schedule digest, the first words of its output;
    /// `None` for a per-leaf guest, whose program id names its program.
    pub(crate) vbin_digest: Option<[u64; 4]>,
}

impl BNodeInputs<'_> {
    /// Words each leaf word takes in the guest's output.
    const HALVES_PER_WORD: usize = 8;

    fn check(&self) -> Result<(), String> {
        let guest = self
            .leaf
            .iter()
            .filter(|w| matches!(w, LeafWord::Guest(_)))
            .count();
        let want = HEADER_HALVES + Self::HALVES_PER_WORD * guest + BIND_HALVES;
        if self.vbin.is_empty() || self.vbin_layout.out_halves != want {
            return Err(format!(
                "a v_bin output of {} halves does not hold a header, {guest} leaf words and the binding ({want})",
                self.vbin_layout.out_halves,
            ));
        }
        let inputs = self.vfld.num_public_words.saturating_sub(7);
        if self.vfld.num_public_words < 7
            || self
                .leaf
                .iter()
                .any(|w| matches!(w, LeafWord::Field(i) if *i >= inputs))
        {
            return Err("the bound v_fld verifier publishes no such input".into());
        }
        Ok(())
    }
}

/// Option B's pair node: verifies every v_bin block leaf and the bound v_fld
/// verifier; binds the v_bin leaves to one another and closes their bus (the
/// guest's proof is whole within the pair); checks the binding — `c_bin`, `α`
/// and `s` at the end of the guest's output equal the v_fld verifier's, whose
/// `α` it recomputed from `c_bin` and its proof's roots — and the v_fld bus
/// sum is zero; then publishes the block leaf's words from the guest's output,
/// in the block's child layout — the guest's from its output, the rest as the
/// constants the pair's programs fix — so the block's nodes take it as that
/// leaf.
///
/// Arenas: each v_bin child's, then the v_fld child's.
pub(crate) fn emit_b_node(inputs: &BNodeInputs<'_>) -> Result<LfmProgram, String> {
    use super::per_table_aggregator::{LegArenas, declare_leg_arenas, emit_leg};
    inputs.check()?;
    let mut b = super::builder::LfmBuilder::new()
        .with_wrap_hash(<crate::hash_pin::BlockFamily as crate::hash_pin::ProofFamily>::WRAP);
    let arenas: Vec<LegArenas> = inputs
        .vbin
        .iter()
        .chain(std::iter::once(inputs.vfld))
        .map(|child| declare_leg_arenas(&mut b, child))
        .collect();
    let mut legs: Vec<_> = inputs
        .vbin
        .iter()
        .chain(std::iter::once(inputs.vfld))
        .zip(&arenas)
        .map(|(child, a)| emit_leg(&mut b, child, a))
        .collect();
    let vfld = legs.pop().expect("the v_fld leg");

    let layout = inputs.vbin_layout;
    let sum = super::block_node::emit_block_bindings(
        &mut b,
        &legs,
        &layout,
        super::block_node::BlockBindings::ALL,
    );
    let zero_ext = b.ext_const(&FEE::zero());
    b.assert_eq_ext(sum, zero_ext);

    let two32 = b.felt_const(FE::from(1u64 << 32));
    let zero = b.felt_const(FE::zero());
    let first = &legs[0];
    let u64_at = |b: &mut super::builder::LfmBuilder, half: usize| {
        let lo = first.publics[layout.out_half(half)].lanes[0];
        let hi = first.publics[layout.out_half(half + 1)].lanes[0];
        b.mul_add(hi, two32, lo)
    };
    if let Some(digest) = inputs.vbin_digest {
        for (i, &x) in digest.iter().enumerate() {
            let got = u64_at(&mut b, 2 * i);
            let want = b.felt_const(FE::from(x));
            b.assert_eq(got, want);
        }
    }
    let tail = layout.out_halves - BIND_HALVES;
    let words = vfld.publics.len();
    for (word, (lanes, from)) in [(4usize, 0usize), (3, 4), (3, 7)].into_iter().enumerate() {
        let got = vfld.publics[words - 4 + word].lanes.clone();
        for (k, &g) in got.iter().enumerate().take(lanes) {
            let want = u64_at(&mut b, tail + 2 * (from + k));
            b.assert_eq(g, want);
        }
        if lanes == 3 {
            b.assert_eq(got[3], zero);
        }
    }
    for &lane in &vfld.publics[words - 1].lanes {
        b.assert_eq(lane, zero);
    }

    for word in inputs.leaf {
        let cell = match *word {
            LeafWord::Guest(w) => {
                let lanes: [_; 4] = core::array::from_fn(|j| {
                    u64_at(
                        &mut b,
                        HEADER_HALVES + BNodeInputs::HALVES_PER_WORD * w + 2 * j,
                    )
                });
                b.pack_word(lanes)
            }
            LeafWord::Const(value) => b.digest_const(value).as_cell(),
            LeafWord::Field(i) => {
                let lanes = &vfld.publics[3 + i].lanes;
                b.pack_word([lanes[0], lanes[1], lanes[2], lanes[3]])
            }
        };
        b.public(cell);
    }
    let program = super::compiler::compile(b.finish());
    super::validator::validate(&program).map_err(|e| format!("not admissible: {e:?}"))?;
    Ok(program)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field_vm::asm::Asm;
    use crate::field_vm::executor::{Memory, NoHints, execute as execute_fvm};
    use crate::field_vm::isa::{Arg, gpr};
    use crate::field_vm::prove::{default_options, generate_traces, program_id, public_cells};
    use crate::field_vm::whir::prove_whir_block;
    use crate::hash_pin::{BlockFamily, ProofFamily};
    use crate::lfm::executor::{LfmExecError, execute};
    use crate::lfm::whir_chain::onechain_steps::{self, Step};
    use crate::lfm::word::ext_word;
    use crate::tables::types::{FE, FEE};

    enum Verdict {
        Accept(Vec<LfmWord>),
        NonzeroSum(LfmWord),
        Refused(LfmExecError),
    }

    /// Runs the verifier over `arena`; on a refusal prints the instruction
    /// that failed, its step and its group's tables.
    fn run(plan: &WhirBlockPlan, arena: &[Vec<LfmWord>]) -> Verdict {
        let (program, marks) = onechain_steps::record(|| emit_fvm_whir_verifier(plan));
        let program = program.expect("the verifier emits");
        let hasher = program.hasher(<BlockFamily as ProofFamily>::SOCKET);
        match execute(&program, arena, &hasher) {
            Ok(exec) => {
                let sum = exec.public_words.last().expect("the sum is published").1;
                if sum == ext_word(&FEE::zero()) {
                    Verdict::Accept(exec.public_words.iter().map(|(_, w)| *w).collect())
                } else {
                    Verdict::NonzeroSum(sum)
                }
            }
            Err(error) => {
                report(plan, &program, &marks, &error);
                Verdict::Refused(error)
            }
        }
    }

    fn report(
        plan: &WhirBlockPlan,
        program: &LfmProgram,
        marks: &[(Step, usize)],
        error: &LfmExecError,
    ) {
        eprintln!("refused: {error:?} ({} instructions)", program.instrs.len());
        let LfmExecError::DivByZero { addr } = error else {
            return;
        };
        let Some(index) = program.instrs.iter().position(|i| {
            let mut out = Vec::new();
            i.writes_into(&mut out);
            out.iter().any(|a| a.0 == *addr)
        }) else {
            return;
        };
        eprintln!("  instruction {index}: {:?}", program.instrs[index]);
        if let Some((chain, step)) = onechain_steps::failed_at(program, marks, error) {
            eprintln!("  chain {chain:?}, step {step:?}");
        }
        // The k-th walk mark opens the k-th group; the group runs to the next.
        let walks: Vec<usize> = marks
            .iter()
            .filter(|(s, _)| *s == Step::Part("walk"))
            .map(|&(_, at)| at)
            .collect();
        let Some(g) = walks.iter().rposition(|&at| at <= index) else {
            eprintln!("  before the first group (the front)");
            return;
        };
        let end = walks.get(g + 1).copied().unwrap_or(program.instrs.len());
        let refs = plan.frame.airs.air_refs();
        let names: Vec<&str> = plan.groups[g].iter().map(|&t| refs[t].name()).collect();
        eprintln!(
            "  group {g}: instructions {}..{end}, tables {:?} {names:?}",
            walks[g], plan.groups[g]
        );
    }

    #[test]
    fn fvm_whir_verifier_accepts_and_rejects_a_changed_public_cell() {
        let (acc, count) = (gpr(0), gpr(1));
        let mut asm = Asm::new();
        asm.mov(Arg::reg(acc), Arg::imm(1))
            .mov(Arg::reg(count), Arg::imm(3));
        let top = asm.label();
        asm.bind(top);
        asm.mul(Arg::reg(acc), Arg::reg(acc), Arg::mem_abs(0))
            .sub(Arg::reg(count), Arg::reg(count), Arg::imm(1))
            .jump_not_zero(top)
            .store(Arg::mem_abs(1), Arg::reg(acc))
            .halt();
        let program = asm.finish().unwrap();
        let execution = execute_fvm(
            &program,
            Memory::with_init(2, &[FEE::from(5u64)]),
            &mut NoHints,
            1 << 12,
        )
        .unwrap();
        let options = default_options();
        let pid = program_id(&program, &options);
        let public = public_cells(&execution, &[0, 1]);
        let mut traces = generate_traces(&program, &execution, &[0, 1]);
        let proof = prove_whir_block(
            &pid,
            &program,
            &public,
            &mut traces,
            Vec::new(),
            &mut [],
            &options,
        )
        .unwrap();

        let plan = fvm_whir_plan(&program, &pid, &public, &proof, &options).unwrap();
        let arena = fvm_whir_arena(&plan, &proof, Vec::new()).unwrap();
        eprintln!(
            "groups {:?}, prepared {}, arena {} words",
            plan.groups,
            plan.prepared.len(),
            arena.iter().map(Vec::len).sum::<usize>()
        );
        assert!(
            matches!(run(&plan, &arena), Verdict::Accept(_)),
            "the proof is accepted"
        );

        let mut changed = public.clone();
        changed[1].1 += FEE::one();
        let changed_plan = fvm_whir_plan(&program, &pid, &changed, &proof, &options).unwrap();
        match run(&changed_plan, &arena) {
            Verdict::Accept(_) => panic!("a changed public cell is accepted"),
            Verdict::NonzeroSum(sum) => eprintln!("changed public cell: nonzero sum {sum:?}"),
            Verdict::Refused(e) => eprintln!("changed public cell: refused {e:?}"),
        }

        // The same statement with a changed bus target only: the transcript
        // replays honestly and the leaf's sum is left nonzero.
        let mut target_only = fvm_whir_plan(&program, &pid, &public, &proof, &options).unwrap();
        if let LeafTarget::FieldVm { public, .. } = &mut target_only.target
            && let FvmCell::Const(v) = &mut public[1].1
        {
            v[0] += FE::one();
        }
        match run(&target_only, &arena) {
            Verdict::NonzeroSum(sum) => eprintln!("changed bus target: nonzero sum {sum:?}"),
            Verdict::Accept(_) => panic!("a changed bus target is accepted"),
            Verdict::Refused(e) => {
                panic!("a changed bus target only should leave a sum, got {e:?}")
            }
        }
    }

    /// The small loop program and its run: `mem[1] = mem[0]^3`.
    fn small_run_from(x: u64) -> (Program, crate::field_vm::executor::Execution) {
        let (acc, count) = (gpr(0), gpr(1));
        let mut asm = Asm::new();
        asm.mov(Arg::reg(acc), Arg::imm(1))
            .mov(Arg::reg(count), Arg::imm(3));
        let top = asm.label();
        asm.bind(top);
        asm.mul(Arg::reg(acc), Arg::reg(acc), Arg::mem_abs(0))
            .sub(Arg::reg(count), Arg::reg(count), Arg::imm(1))
            .jump_not_zero(top)
            .store(Arg::mem_abs(1), Arg::reg(acc))
            .halt();
        let program = asm.finish().unwrap();
        let execution = execute_fvm(
            &program,
            Memory::with_init(2, &[FEE::from(x)]),
            &mut NoHints,
            1 << 12,
        )
        .unwrap();
        (program, execution)
    }

    fn coords(x: FEE) -> [u64; 3] {
        x.value().map(|c| c.canonical())
    }

    /// The guest's output for a binding: a header, then `c_bin ‖ α ‖ s`.
    fn guest_output(c_bin: [u64; 4], alpha: FEE, s: FEE) -> Vec<u8> {
        let mut out = vec![0x5a; 48];
        for x in c_bin.iter().chain(&coords(alpha)).chain(&coords(s)) {
            out.extend(x.to_le_bytes());
        }
        out
    }

    /// A v_bin leaf's published words over `output`: id, state, halves, sum.
    fn vbin_words(output: &[u8]) -> Vec<LfmWord> {
        let mut words = vec![
            [FE::from(1u64); 4],
            [FE::from(2u64); 4],
            [FE::from(3u64); 4],
        ];
        words.extend(
            super::super::whir_block::out_halves(output)
                .into_iter()
                .map(crate::lfm::word::base_word),
        );
        words.push(ext_word(&FEE::from(9u64)));
        words
    }

    /// A digest of a program's text: equal texts, equal program ids.
    fn text_digest(program: &LfmProgram) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        for instr in &program.instrs {
            format!("{instr:?}").hash(&mut h);
        }
        (
            program.num_addrs,
            program.public_len,
            &program.arena_schema.lens,
        )
            .hash(&mut h);
        h.finish()
    }

    /// One run of the small program as a bound v_fld proof: the plan (a
    /// function of the program alone), the honest arena, and the witness.
    struct BoundSmall {
        plan: WhirBlockPlan,
        proof: FvmWhirBlockProof,
        public: Vec<PublicCell>,
        c_bin: [u64; 4],
        alpha: FEE,
        s: FEE,
    }

    fn bound_small(x: u64) -> BoundSmall {
        use crate::field_vm::prove::generate_traces_reading;
        use crate::field_vm::whir::{prove_whir_block_bound, verify_whir_block_bound};
        let (program, execution) = small_run_from(x);
        let options = default_options();
        let pid = program_id(&program, &options);
        let public = public_cells(&execution, &[1]);
        let faddr = [0u64];
        let values = [execution.mem[0]];
        let c_bin = cksum::crossing_digest(&[coords(values[0])], &[]);
        let mut traces = generate_traces_reading(&program, &execution, &[1], None, &faddr);
        let (proof, alpha) = prove_whir_block_bound(
            &pid,
            &program,
            &public,
            &mut traces,
            &c_bin,
            &faddr,
            &values,
            &options,
        )
        .unwrap();
        let s = cksum::horner(&values, alpha);
        assert!(verify_whir_block_bound(
            &c_bin, &s, &pid, &program, &public, &faddr, &proof, &options
        ));
        let plan =
            fvm_whir_plan_bound(&program, &pid, &public, &[1], &faddr, &proof, &options).unwrap();
        BoundSmall {
            plan,
            proof,
            public,
            c_bin,
            alpha,
            s,
        }
    }

    impl BoundSmall {
        fn arena(&self, c_bin: [u64; 4], s: FEE, out: FEE) -> Vec<Vec<LfmWord>> {
            let mut public = self.public.clone();
            public[0].1 = out;
            public.push((cksum::CKSUM_ADDR, s));
            let prefix = bound_arena_prefix(&self.plan, c_bin, &public).unwrap();
            fvm_whir_arena(&self.plan, &self.proof, prefix).unwrap()
        }

        fn honest(&self) -> Vec<Vec<LfmWord>> {
            self.arena(self.c_bin, self.s, self.public[0].1)
        }
    }

    #[test]
    fn fvm_whir_bound_verifier_and_b_binding() {
        let small = bound_small(5);
        let (c_bin, alpha, s) = (small.c_bin, small.alpha, small.s);
        let arena = small.honest();
        let Verdict::Accept(vfld) = run(&small.plan, &arena) else {
            panic!("the bound proof is accepted");
        };
        assert_eq!(
            vfld.len(),
            8,
            "id, state, the output cell, c_bin, alpha, s, sum"
        );
        assert_eq!(vfld[3], ext_word(&small.public[0].1));
        assert_eq!(vfld[4], c_bin.map(FE::from));
        assert_eq!(vfld[5], ext_word(&alpha));
        assert_eq!(vfld[6], ext_word(&s));

        // CKSUM must be the last group, alone: α binds every root before it.
        {
            use crate::field_vm::whir::verify_whir_block_bound;
            let (program, execution) = small_run_from(5);
            let options = default_options();
            let pid = program_id(&program, &options);
            let mut moved = small.proof.clone();
            *moved.groups.last_mut().unwrap() = vec![0];
            assert!(
                fvm_whir_plan_bound(&program, &pid, &small.public, &[1], &[0], &moved, &options)
                    .is_err()
            );
            assert!(!verify_whir_block_bound(
                &c_bin,
                &s,
                &pid,
                &program,
                &public_cells(&execution, &[1]),
                &[0],
                &moved,
                &options
            ));
        }

        // Another c_bin, s or output cell in the arena: refused.
        let mut other = c_bin;
        other[0] += 1;
        let out = small.public[0].1;
        for (i, arena) in [
            small.arena(other, s, out),
            small.arena(c_bin, s + FEE::one(), out),
            small.arena(c_bin, s, out + FEE::one()),
        ]
        .iter()
        .enumerate()
        {
            assert!(
                !matches!(run(&small.plan, arena), Verdict::Accept(_)),
                "case {i}: a changed input is refused"
            );
        }

        // The binding across the halves: the guest's output against the leaf's
        // published words.
        let output = guest_output(c_bin, alpha, s);
        let shape = BindShape::new(output.len(), 1).unwrap();
        let bind = emit_b_bind(shape).unwrap();
        eprintln!("binding program: {} instructions", bind.instrs.len());
        let hasher = bind.hasher(<BlockFamily as ProofFamily>::SOCKET);
        let accept =
            |vbin: Vec<LfmWord>, vfld: Vec<LfmWord>| execute(&bind, &[vbin, vfld], &hasher).is_ok();
        assert!(accept(vbin_words(&output), vfld.clone()), "the halves bind");
        for (i, (c, a, s2)) in [
            (other, alpha, s),
            (c_bin, alpha + FEE::one(), s),
            (c_bin, alpha, s + FEE::one()),
        ]
        .into_iter()
        .enumerate()
        {
            assert!(
                !accept(vbin_words(&guest_output(c, a, s2)), vfld.clone()),
                "case {i}: another guest claim is refused"
            );
        }
        let mut unbalanced = vfld.clone();
        unbalanced[7] = ext_word(&FEE::one());
        assert!(
            !accept(vbin_words(&output), unbalanced),
            "a nonzero v_fld sum is refused"
        );
    }

    /// L102: the bound v_fld verifier is a function of the program, not of
    /// the run. Two runs on other inputs (other crossing cells, `c_bin`, `α`,
    /// `s` and output) give the same verifier text; each accepts its own
    /// arena and refuses the other's inputs.
    #[test]
    fn the_bound_verifier_does_not_depend_on_the_witness() {
        let (a, b) = (bound_small(5), bound_small(7));
        assert_ne!(a.alpha, b.alpha);
        assert_ne!(a.public[0].1, b.public[0].1);
        assert_eq!(a.plan.id(), b.plan.id(), "the plan's id");
        let (pa, pb) = (
            emit_fvm_whir_verifier(&a.plan).unwrap(),
            emit_fvm_whir_verifier(&b.plan).unwrap(),
        );
        assert_eq!(text_digest(&pa), text_digest(&pb), "the verifier's text");
        for (x, y) in [(&a, &b), (&b, &a)] {
            assert!(matches!(run(&x.plan, &x.honest()), Verdict::Accept(_)));
            let mixed = x.arena(y.c_bin, x.s, x.public[0].1);
            assert!(
                !matches!(run(&x.plan, &mixed), Verdict::Accept(_)),
                "the other run's c_bin is refused"
            );
        }
        let shape = |n| BindShape::new(n, 1).unwrap();
        let out = |x: &BoundSmall| guest_output(x.c_bin, x.alpha, x.s).len();
        assert_eq!(
            text_digest(&emit_b_bind(shape(out(&a))).unwrap()),
            text_digest(&emit_b_bind(shape(out(&b))).unwrap()),
            "the binding's text"
        );
    }

    /// `split.rs`'s region height for `cells` crossing cells (as
    /// `fvm_whir_leaf`'s).
    fn region_rows(cells: usize) -> usize {
        const CELL_BYTES: usize = 24;
        let page = crate::tables::page::DEFAULT_PAGE_SIZE;
        let gcd = |mut a: usize, mut b: usize| {
            while b != 0 {
                (a, b) = (b, a % b);
            }
            a
        };
        cells.next_power_of_two().max(page / gcd(CELL_BYTES, page))
    }

    /// One block leaf split as option B: its field half proved as a bound
    /// block WHIR v_fld proof and its LFM verifier, its hash half as the
    /// native binding guest with its input and expected output.
    struct BSides {
        leaf_words: Vec<LfmWord>,
        leaf_sources: Vec<LeafWord>,
        vplan: WhirBlockPlan,
        vprog: LfmProgram,
        varena: Vec<Vec<LfmWord>>,
        vfld_words: Vec<LfmWord>,
        elf: Vec<u8>,
        input: Vec<u8>,
        output: Vec<u8>,
        vbin_digest: Option<[u64; 4]>,
    }

    fn b_sides(plan: &WhirBlockPlan, base: &crate::block_whir::BlockWhirProof, k: usize) -> BSides {
        use crate::field_vm::prove::generate_traces_reading;
        use crate::field_vm::whir::{prove_whir_block_bound, verify_whir_block_bound};
        use crate::field_vm::{lfm_translate, native_vbin, split};
        use crate::lfm::hash::HasherKind;
        use std::time::Instant;

        let (leaf, scopes) = super::super::fn_scope::record(|| plan.leaf_program(k));
        let leaf = leaf.expect("the leaf emits");
        let arenas = super::super::whir_block::block_leaf_arena(plan, base, k).unwrap();
        let exec = execute(
            &leaf,
            &arenas,
            &leaf.hasher(<BlockFamily as ProofFamily>::SOCKET),
        )
        .expect("the leaf executes");
        let leaf_words: Vec<LfmWord> = exec.public_words.iter().map(|(_, w)| *w).collect();
        let cut = split::cut(&leaf, &exec, &arenas, true);
        let mut guest = 0;
        let mut leaf_sources = vec![None; leaf_words.len()];
        for (i, instr) in leaf.instrs.iter().enumerate() {
            if let crate::lfm::instr::Instr::Public { index, .. } = instr {
                leaf_sources[*index as usize] = Some(if cut.riscv[i] {
                    guest += 1;
                    LeafWord::Guest(guest - 1)
                } else {
                    LeafWord::Const(leaf_words[*index as usize])
                });
            }
        }
        let mut leaf_sources: Vec<LeafWord> = leaf_sources
            .into_iter()
            .map(|w| w.expect("every word published"))
            .collect();
        let from_guest: Vec<LfmWord> = leaf_sources
            .iter()
            .zip(&leaf_words)
            .filter(|(s, _)| matches!(s, LeafWord::Guest(_)))
            .map(|(_, w)| *w)
            .collect();
        assert_eq!(
            cut.publics, from_guest,
            "the guest outputs the leaf's computed words"
        );
        eprintln!("leaf {k} words: {leaf_sources:?}");
        let tr = lfm_translate::translate_outlined(
            &leaf,
            &exec,
            true,
            &cut.crossing(),
            &|i, _| cut.riscv[i] && !cut.both[i],
            &scopes,
        );
        let fvm_run = execute_fvm(
            &tr.program,
            Memory::with_init(tr.mem.len(), &tr.mem),
            &mut NoHints,
            1 << 27,
        )
        .expect("the field half executes");
        let (n_out, n_back) = (cut.out.len(), cut.back.len());
        let kept = &tr.kept;
        let faddr: Vec<u64> = kept[..n_out]
            .iter()
            .chain(&kept[n_out + n_back..])
            .chain(&kept[n_out..n_out + n_back])
            .copied()
            .collect();
        let in_cells = cut.in_cells();
        let back_cells: Vec<[u64; 3]> = cut
            .back
            .iter()
            .map(|a| {
                let w = cut.values[a];
                [w[0].canonical(), w[1].canonical(), w[2].canonical()]
            })
            .collect();
        let values: Vec<FEE> = faddr.iter().map(|&a| fvm_run.mem[a as usize]).collect();
        let mut public = tr.public.clone();
        public.sort_unstable();
        public.dedup();
        let opts = super::super::proof::aggregation_wrap_options();
        let pid = program_id(&tr.program, &opts);
        let cells = public_cells(&fvm_run, &public);
        // The cells the program computes, not its constants: the leaf's
        // inputs, and the words it publishes from them.
        let inputs: Vec<u64> = public
            .iter()
            .copied()
            .filter(|a| !tr.consts.iter().any(|(c, _)| c == a))
            .collect();
        for &(index, addr) in &tr.published {
            if let Some(i) = inputs.iter().position(|&a| a == addr) {
                assert_eq!(leaf_words[index as usize][3], FE::zero());
                leaf_sources[index as usize] = LeafWord::Field(i);
            }
        }
        eprintln!("leaf {k} inputs: {inputs:?}, words: {leaf_sources:?}");
        let c_bin = cksum::crossing_digest(&in_cells, &back_cells);
        let t = Instant::now();
        let mut traces = generate_traces_reading(&tr.program, &fvm_run, &public, None, &faddr);
        let (proof, alpha) = prove_whir_block_bound(
            &pid,
            &tr.program,
            &cells,
            &mut traces,
            &c_bin,
            &faddr,
            &values,
            &opts,
        )
        .expect("the bound v_fld proves");
        let prove_s = t.elapsed().as_secs_f64();
        let s = cksum::horner(&values, alpha);
        let t = Instant::now();
        assert!(verify_whir_block_bound(
            &c_bin,
            &s,
            &pid,
            &tr.program,
            &cells,
            &faddr,
            &proof,
            &opts
        ));
        eprintln!(
            "leaf {k}: {} LFM instructions; v_fld (block WHIR, bound): {} crossing, {} public, prove {prove_s:.1} s, host verify {:.2} s, tables {:?}, groups {:?}",
            leaf.instrs.len(),
            faddr.len(),
            cells.len(),
            t.elapsed().as_secs_f64(),
            proof.table_num_vars,
            proof.groups
        );

        let t = Instant::now();
        let vplan = fvm_whir_plan_bound(&tr.program, &pid, &cells, &inputs, &faddr, &proof, &opts)
            .expect("the v_fld plan");
        let (vprog, marks) = onechain_steps::record(|| emit_fvm_whir_verifier(&vplan));
        let vprog = vprog.expect("the v_fld verifier emits");
        let mut stated = cells.clone();
        stated.push((cksum::CKSUM_ADDR, s));
        let varena = fvm_whir_arena(
            &vplan,
            &proof,
            bound_arena_prefix(&vplan, c_bin, &stated).unwrap(),
        )
        .unwrap();
        let emit_s = t.elapsed().as_secs_f64();
        let census =
            super::super::airs::lfm_chip_census_with_hasher(&vprog, HasherKind::Poseidon1W16);
        let total: u64 = census.iter().map(|c| c.main_cells() + c.aux_cells()).sum();
        eprintln!(
            "  v_fld verifier: {} LFM instructions, arena {} words, emitted in {emit_s:.1} s, cells {total} ({:.1} M)",
            vprog.instrs.len(),
            varena.iter().map(Vec::len).sum::<usize>(),
            total as f64 / 1e6
        );
        for c in &census {
            eprintln!(
                "    chip {:<12} rows={:>9} main={:>4} aux={:>3} cells={:>11}",
                c.name,
                c.rows,
                c.main_cols,
                c.aux_cols,
                c.main_cells() + c.aux_cells()
            );
        }
        let mut parts: std::collections::BTreeMap<String, usize> = Default::default();
        for w in marks.windows(2) {
            let kind = format!("{:?}", w[0].0);
            *parts
                .entry(kind.split([' ', '{']).next().unwrap().to_string())
                .or_default() += w[1].1 - w[0].1;
        }
        eprintln!("    instructions by step kind: {parts:?}");
        let vfld_words = match run(&vplan, &varena) {
            Verdict::Accept(words) => words,
            _ => panic!("the v_fld verifier accepts"),
        };

        // `VBG_GENERIC`: the generic guest over the cut's schedule.
        let (elf, input, output, vbin_digest) = if std::env::var_os("VBG_GENERIC").is_some() {
            use crate::field_vm::vbin_generic as vg;
            let sched = vg::compile(&cut.words).expect("the cut compiles");
            let canon = |w: &LfmWord| -> [u64; 4] { core::array::from_fn(|l| w[l].canonical()) };
            let hints: Vec<[u64; 4]> = cut.hints.iter().map(canon).collect();
            let publics: Vec<[u64; 4]> = cut.publics.iter().map(canon).collect();
            let input = vg::input(&sched, &hints, &in_cells, coords(alpha));
            let output = vg::expected_output(&sched, &publics, c_bin, coords(alpha), coords(s));
            let digest = vg::schedule_digest(&sched.words);
            (vg::guest().to_vec(), input, output, Some(digest))
        } else {
            let g = native_vbin::build(&cut.words).expect("the cut's program decodes");
            let (input, layout) = cut.native_input_regions_bound(
                &in_cells,
                &cut.hints,
                region_rows(n_out + cut.input.len()),
                region_rows(n_back),
                coords(alpha),
            );
            let output = cut.expected_output_bound(&layout, c_bin, coords(alpha), coords(s));
            (g.elf, input, output, None)
        };
        BSides {
            leaf_words,
            leaf_sources,
            vplan,
            vprog,
            varena,
            vfld_words,
            elf,
            input,
            output,
            vbin_digest,
        }
    }

    /// The v_bin half proved on the Lambda VM under `format`, and its plan.
    fn prove_vbin(
        sides: &BSides,
        format: &crate::block_whir::BlockFormat,
        saved: Option<&str>,
    ) -> (crate::block_whir::BlockWhirProof, WhirBlockPlan) {
        use crate::block_whir::{BlockOptions, BlockWhirProof, prove_block_whir};
        use std::time::Instant;
        let bopts = super::super::proof::block_base_options();
        let vbin: BlockWhirProof = match saved.map(std::fs::read) {
            Some(Ok(bytes)) => rkyv::from_bytes::<BlockWhirProof, rkyv::rancor::Error>(&bytes)
                .expect("a saved v_bin proof"),
            _ => {
                let t = Instant::now();
                let (p, _) = prove_block_whir(
                    &sides.elf,
                    &sides.input,
                    &bopts,
                    format,
                    &BlockOptions::production(),
                )
                .expect("v_bin proves");
                eprintln!("  v_bin proved in {:.1} s", t.elapsed().as_secs_f64());
                assert_eq!(
                    p.public_output, sides.output,
                    "v_bin states the guest's output"
                );
                if let Some(path) = saved {
                    let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&p).expect("serialize");
                    std::fs::write(path, &bytes).expect("write the v_bin proof");
                }
                p
            }
        };
        assert_eq!(
            vbin.public_output, sides.output,
            "v_bin states the guest's output"
        );
        let t = Instant::now();
        let bplan = WhirBlockPlan::derive(&sides.elf, &bopts, format, vbin.statement(), None)
            .expect("the v_bin plan derives");
        eprintln!(
            "  v_bin plan: {} leaves, derived in {:.1} s",
            bplan.partition().num_leaves(),
            t.elapsed().as_secs_f64()
        );
        (vbin, bplan)
    }

    /// Option B on a real leaf, both halves verified by LFM programs and bound
    /// by [`emit_b_bind`]. The leaf from `fvm_whir_leaf::leaf_source`
    /// (`BLOCK_WHIR_ELF`, `FVM_LEAF_PROOF`, `FVM_LEAF`); with `FVM_BIND_VBIN=1`
    /// also the native binding guest proved on the Lambda VM
    /// (`FVM_VBIN_PROOF=path` keeps that proof) and its block leaves executed.
    #[test]
    #[ignore]
    fn mainnet_b_leaf_bound_verifiers() {
        let (plan, base) = super::super::fvm_whir_leaf::leaf_source();
        let k: usize = std::env::var("FVM_LEAF")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let sides = b_sides(&plan, &base, k);
        if std::env::var("FVM_BIND_VBIN").as_deref() != Ok("1") {
            return;
        }
        let saved = std::env::var("FVM_VBIN_PROOF").ok();
        let (vbin, bplan) = prove_vbin(
            &sides,
            &crate::block_whir::BlockFormat::production(),
            saved.as_deref(),
        );
        let mut sum = FEE::zero();
        let mut carrier = Vec::new();
        for leaf_k in 0..bplan.partition().num_leaves() {
            let t = std::time::Instant::now();
            let prog = bplan.leaf_program(leaf_k).expect("the v_bin leaf emits");
            let arena = super::super::whir_block::block_leaf_arena(&bplan, &vbin, leaf_k).unwrap();
            let ex = execute(
                &prog,
                &arena,
                &prog.hasher(<BlockFamily as ProofFamily>::SOCKET),
            )
            .expect("the v_bin leaf executes");
            let words: Vec<LfmWord> = ex.public_words.iter().map(|(_, w)| *w).collect();
            sum += crate::lfm::word::word_as_ext(words.last().unwrap()).unwrap();
            eprintln!(
                "  v_bin leaf {leaf_k}: {} instructions, executed in {:.1} s",
                prog.instrs.len(),
                t.elapsed().as_secs_f64()
            );
            if leaf_k == bplan.carrier() {
                carrier = words;
            }
        }
        eprintln!("v_bin leaves' bus shares close: {}", sum == FEE::zero());
        let shape = BindShape::new(sides.output.len(), sides.vfld_words.len() - 7).unwrap();
        let bind = emit_b_bind(shape).unwrap();
        let hasher = bind.hasher(<BlockFamily as ProofFamily>::SOCKET);
        assert!(
            execute(&bind, &[carrier.clone(), sides.vfld_words.clone()], &hasher).is_ok(),
            "the halves bind"
        );
        let mut forged = carrier;
        forged[shape.vbin_tail + 8][0] += FE::one();
        eprintln!(
            "binding: {} instructions; accepted, a forged alpha half refused: {}",
            bind.instrs.len(),
            execute(&bind, &[forged, sides.vfld_words], &hasher).is_err()
        );
    }

    /// One LFM program proved under the block family and harvested for its
    /// parent: the proof, the child's arenas, its derived shape.
    struct Proved {
        proof: super::super::proof::LfmProof,
        arenas: Vec<Vec<LfmWord>>,
        derived: super::super::per_table_aggregator::DerivedChild,
    }

    fn prove_lfm(
        label: &str,
        program: &LfmProgram,
        arenas: &[Vec<LfmWord>],
        words: usize,
    ) -> Proved {
        let opts = super::super::proof::aggregation_wrap_options();
        let t = std::time::Instant::now();
        let artifacts = super::super::whir_block::artifacts_of(program, &opts);
        let derived =
            super::super::whir_block::derived_of(&artifacts, &opts, words).expect("derived");
        let proof =
            super::super::proof::lfm_prove_under::<BlockFamily>(program, &artifacts, arenas, &opts)
                .unwrap_or_else(|e| panic!("{label} proves: {e:?}"));
        let census = super::super::airs::lfm_chip_census_with_hasher(
            program,
            program.hasher(<BlockFamily as ProofFamily>::SOCKET),
        );
        let cells: u64 = census.iter().map(|c| c.main_cells() + c.aux_cells()).sum();
        let (child, _) = super::super::harvest::harvest_child_verified_under::<BlockFamily>(
            artifacts, opts, &proof,
        )
        .unwrap_or_else(|e| panic!("{label} verifies: {e}"));
        eprintln!(
            "  {label}: {} instructions, {:.1} M cells, artifacts + prove + verify {:.1} s",
            program.instrs.len(),
            cells as f64 / 1e6,
            t.elapsed().as_secs_f64()
        );
        Proved {
            proof,
            arenas: super::super::per_table_aggregator_tests::child_arena_words(&child),
            derived,
        }
    }

    /// A B pair proved: the v_bin leaves' and the v_fld verifier's proofs.
    struct BPair {
        vbin: Vec<Proved>,
        vbin_layout: BlockLayout,
        vfld: Proved,
        leaf_words: Vec<LfmWord>,
        leaf_sources: Vec<LeafWord>,
        vbin_digest: Option<[u64; 4]>,
    }

    fn b_pair(plan: &WhirBlockPlan, base: &crate::block_whir::BlockWhirProof, k: usize) -> BPair {
        let sides = b_sides(plan, base, k);
        // A small block's leaf is a full-size verifier: its guest proves at the
        // production format. `FVM_B4_VBIN_DIR` keeps each pair's v_bin proof.
        let saved = std::env::var("FVM_B4_VBIN_DIR")
            .ok()
            .map(|d| format!("{d}/vbin_pair{k}.rkyv"));
        let (vbin, bplan) = prove_vbin(
            &sides,
            &crate::block_whir::BlockFormat::production(),
            saved.as_deref(),
        );
        let words = bplan.child_layout().total();
        let vbin_leaves = (0..bplan.partition().num_leaves())
            .map(|j| {
                let program = bplan.leaf_program(j).expect("the v_bin leaf emits");
                let arena = super::super::whir_block::block_leaf_arena(&bplan, &vbin, j).unwrap();
                prove_lfm(&format!("pair {k} v_bin leaf {j}"), &program, &arena, words)
            })
            .collect();
        let vfld = prove_lfm(
            &format!("pair {k} v_fld verifier"),
            &sides.vprog,
            &sides.varena,
            sides.vfld_words.len(),
        );
        let _ = &sides.vplan;
        BPair {
            vbin: vbin_leaves,
            vbin_layout: bplan.child_layout(),
            vfld,
            leaf_words: sides.leaf_words,
            leaf_sources: sides.leaf_sources,
            vbin_digest: sides.vbin_digest,
        }
    }

    /// The B node over `vbin`'s v_bin leaves and `vfld`'s verifier (another
    /// pair's, for the swap): its program and arenas.
    fn b_node(pair: &BPair, vfld: &Proved) -> (LfmProgram, Vec<Vec<LfmWord>>) {
        b_node_pinned(pair, vfld, pair.vbin_digest)
    }

    /// [`b_node`] pinning the guest's schedule digest to `vbin_digest`.
    fn b_node_pinned(
        pair: &BPair,
        vfld: &Proved,
        vbin_digest: Option<[u64; 4]>,
    ) -> (LfmProgram, Vec<Vec<LfmWord>>) {
        let shapes: Vec<_> = pair.vbin.iter().map(|p| p.derived.shape()).collect();
        let vfld_shape = vfld.derived.shape();
        let program = emit_b_node(&BNodeInputs {
            vbin: &shapes,
            vbin_layout: pair.vbin_layout,
            vfld: &vfld_shape,
            leaf: &pair.leaf_sources,
            vbin_digest,
        })
        .expect("the B node emits");
        let arenas = pair
            .vbin
            .iter()
            .chain(std::iter::once(vfld))
            .flat_map(|p| p.arenas.clone())
            .collect();
        (program, arenas)
    }

    /// ★ Option B through a block's recursion: the small block's two leaves,
    /// each a B pair (its v_bin block leaves and its bound v_fld verifier,
    /// proved as LFM proofs) under a B node that binds them and publishes the
    /// leaf's words; the block's own top over the two B nodes claims the block.
    /// A B node over one pair's v_bin and the other's v_fld is refused (the
    /// binding), and the top refuses a B node's changed publication.
    #[test]
    #[ignore = "proves two B pairs, their nodes and the top over a small block; box tier"]
    fn small_block_b_tree() {
        let format = super::super::whir_block_tests::small_format();
        let (elf, base) = super::super::whir_block_tests::small_block("test_commit_4", &format);
        // Proofs are not reproducible (grinding): the v_bin proofs kept in
        // `FVM_B4_VBIN_DIR` are of the block proof kept beside them.
        let base = match std::env::var("FVM_B4_VBIN_DIR") {
            Ok(dir) => {
                let path = format!("{dir}/base.rkyv");
                match std::fs::read(&path) {
                    Ok(bytes) => rkyv::from_bytes::<
                        crate::block_whir::BlockWhirProof,
                        rkyv::rancor::Error,
                    >(&bytes)
                    .expect("a saved block proof"),
                    Err(_) => {
                        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&base).unwrap();
                        std::fs::write(&path, &bytes).unwrap();
                        base
                    }
                }
            }
            Err(_) => base,
        };
        let plan = super::super::whir_block_tests::plan_of(&elf, &base, &format, Some(2));
        let layout = plan.child_layout();
        let hasher = |p: &LfmProgram| p.hasher(<BlockFamily as ProofFamily>::SOCKET);
        let mut pairs: Vec<BPair> = Vec::new();
        let mut nodes = Vec::new();
        for k in 0..plan.partition().num_leaves() {
            pairs.push(b_pair(&plan, &base, k));
            let pair = &pairs[k];
            let (program, arenas) = b_node(pair, &pair.vfld);
            let ex = execute(&program, &arenas, &hasher(&program)).expect("the B node accepts");
            let words: Vec<LfmWord> = ex.public_words.iter().map(|(_, w)| *w).collect();
            assert_eq!(
                words, pair.leaf_words,
                "the B node publishes leaf {k}'s words"
            );
            nodes.push(prove_lfm(
                &format!("B node {k}"),
                &program,
                &arenas,
                layout.total(),
            ));
        }

        // The halves of two pairs: every proof valid, the binding refuses.
        let (swapped, arenas) = b_node(&pairs[0], &pairs[1].vfld);
        assert!(
            execute(&swapped, &arenas, &hasher(&swapped)).is_err(),
            "pair 0's v_bin with pair 1's v_fld is refused"
        );
        // A generic guest's node is pinned to the leaf's schedule.
        if let Some(mut digest) = pairs[0].vbin_digest {
            digest[0] ^= 1;
            let (wrong, arenas) = b_node_pinned(&pairs[0], &pairs[0].vfld, Some(digest));
            assert!(
                execute(&wrong, &arenas, &hasher(&wrong)).is_err(),
                "a node pinned to another schedule refuses pair 0's guest"
            );
        }

        let derived: Vec<_> = nodes.iter().map(|n| &n.derived).collect();
        let top = plan.node_program(&derived, true).expect("the top emits");
        let arenas: Vec<Vec<LfmWord>> = nodes.iter().flat_map(|n| n.arenas.clone()).collect();
        let proved = prove_lfm("top", &top, &arenas, plan.top_layout().total());
        assert!(
            super::super::whir_block::top_claims(&plan, &proved.proof.public_words),
            "the top claims the block"
        );
        let mut tampered = arenas.clone();
        let second = nodes[0].arenas.len();
        tampered[second][2 * 8][0] += FE::one();
        assert!(
            execute(&top, &tampered, &hasher(&top)).is_err(),
            "the top refuses a B node's changed state"
        );
    }
}
