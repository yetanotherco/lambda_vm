//! Shape derivation helpers for Option A Field VM plus LFM hash-chip children.

use stark::config::Commitment;

use crate::field_vm::air::FvmPublicInputs;
use crate::field_vm::hash_side::HashSideId;
use crate::field_vm::prove::{Airs, ExtraAir};

/// AIR inputs needed to derive the verifier shape of a hybrid proof.
pub struct HybridArtifacts {
    pub pid: Vec<Commitment>,
    pub side: HashSideId,
    pub segments: usize,
    /// Whether MEM uses the LFM merged-memory boundary/AIR variant.
    pub mem_lfm: bool,
    /// Trace heights in proof order: FIELD_VM segments, DECODE, MEM, hash side.
    pub heights: Vec<u8>,
}

impl super::per_table_aggregator::DerivedChild {
    /// Derive a hybrid child. The public-cell count and segment boundary values
    /// are public statement shape/data supplied by the leaf plan.
    pub fn from_hybrid(
        artifacts: HybridArtifacts,
        opts: &crate::ProofOptions,
        num_public_cells: usize,
        segment_inputs: Vec<FvmPublicInputs>,
    ) -> Result<Self, String> {
        if artifacts.segments != segment_inputs.len() {
            return Err(format!(
                "{} segments but {} public segment inputs",
                artifacts.segments,
                segment_inputs.len()
            ));
        }
        let extra: Vec<ExtraAir> = crate::field_vm::hash_side::airs(&artifacts.side, opts);
        let airs = Airs::new(
            &artifacts.pid,
            opts,
            artifacts.segments,
            extra,
            artifacts.mem_lfm,
        );
        let refs = airs.refs();
        if refs.len() != artifacts.heights.len() {
            return Err(format!(
                "{} heights for {} AIRs",
                artifacts.heights.len(),
                refs.len()
            ));
        }
        let trees = stark::config::TreeGeometry::of::<
            <crate::hash_pin::BlockFamily as crate::hash_pin::ProofFamily>::Stark,
        >();
        let n = refs.len();
        let tables = refs
            .iter()
            .zip(&artifacts.heights)
            .enumerate()
            .map(|(idx, (air, h))| {
                let rows = 1usize
                    .checked_shl(*h as u32)
                    .ok_or_else(|| format!("table {idx}: trace height overflow"))?;
                if rows == 0 {
                    return Err(format!("table {idx}: invalid trace height"));
                }
                let (mut verify, analysis) =
                    super::epoch_verify::TableVerifyShape::derive_with(*air, rows, trees)?;
                // The quotient composition includes AIR boundary constraints
                // before the framework's LogUp accumulator boundary. FIELD_VM
                // and MEM have main-column boundaries in addition to that
                // accumulator pin; the public state values are already bound
                // by the child statement above.
                let mem_idx = artifacts.segments + artifacts.pid.len();
                if idx < artifacts.segments {
                    let pi = &segment_inputs[idx];
                    if pi.first.len() != crate::field_vm::air::STATE_WORDS
                        || pi.last.len() != crate::field_vm::air::STATE_WORDS
                        || pi.last_row >= rows
                    {
                        return Err(format!("table {idx}: invalid FIELD_VM boundary inputs"));
                    }
                    // AIR::boundary_constraints returns the main state pins
                    // first; AirWithBuses appends the accumulator pin after.
                    verify.quotient.boundary.clear();
                    let trace_generator =
                        <crate::tables::types::GoldilocksField as math::field::traits::IsFFTField>::get_primitive_root_of_unity(
                            *h as u64,
                        )
                        .map_err(|_| format!("table {idx}: no trace-domain generator"))?;
                    let columns = [crate::field_vm::air::cols::PC, crate::field_vm::air::cols::ZERO]
                        .into_iter()
                        .chain(
                            crate::field_vm::air::cols::REGS
                                ..crate::field_vm::air::cols::REGS
                                    + crate::field_vm::air::STATE_WORDS
                                    - 2,
                        )
                        .collect::<Vec<_>>();
                    for (step, words) in [(0usize, &pi.first), (pi.last_row, &pi.last)] {
                        verify.quotient.boundary.extend(
                            columns.iter().zip(words).map(|(&col, &word)| {
                                super::constraints::BoundaryTerm {
                                    col,
                                    point: trace_generator.pow(step as u64),
                                    value: crate::tables::types::FEE::from(word),
                                }
                            }),
                        );
                    }
                    verify.quotient.boundary.extend(
                        super::epoch_verify::boundary_terms(
                            air.has_aux_trace(),
                            verify.sub.deep.num_total_cols,
                        ),
                    );
                } else if idx == mem_idx {
                    // Both MEM AIR variants use MemBoundary: ADDR is pinned to
                    // zero at row 0. The LFM variant adds merged-memory
                    // constraints, but does not change this main-column pin.
                    verify.quotient.boundary.insert(
                        0,
                        super::constraints::BoundaryTerm {
                            col: crate::field_vm::mem::cols::ADDR,
                            point: crate::tables::types::FE::one(),
                            value: crate::tables::types::FEE::zero(),
                        },
                    );
                }
                let precomputed_root = if air.is_preprocessed() {
                    Some(
                        super::epoch_verify::layout_precomputed_commitment(*air, rows)
                            .ok_or_else(|| format!("table {idx}: no preprocessed root"))?,
                    )
                } else {
                    None
                };
                Ok(super::per_table_aggregator::DerivedTable {
                    challenge: super::epoch::TableChallengeShape::derive(*air, idx, n, rows),
                    verify,
                    analysis,
                    precomputed_root,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self {
            program_id: artifacts.pid.first().copied().unwrap_or([0; 32]),
            num_public_words: num_public_cells,
            fri_final_poly_log_degree: opts.fri_final_poly_log_degree,
            statement_tag: b"",
            kind: super::per_table_aggregator::OwnedChildKind::Hybrid {
                program_id_roots: artifacts.pid,
                segments: segment_inputs,
            },
            tables,
        })
    }
}

/// Leaf `k` of `plan` as a hybrid child, from the plan alone
/// ([`super::fvm_whir_leaf::Hybrid::shape`]).
pub(super) fn hybrid_leaf_shape(
    plan: &super::whir_block::WhirBlockPlan,
    k: usize,
    opts: &crate::ProofOptions,
) -> Result<super::per_table_aggregator::DerivedChild, String> {
    let (leaf, scopes) = super::fn_scope::record(|| plan.leaf_program(k));
    let (artifacts, cells, segments) = super::fvm_whir_leaf::Hybrid::shape(&leaf?, opts, &scopes)?;
    super::per_table_aggregator::DerivedChild::from_hybrid(artifacts, opts, cells, segments)
}

/// The top of `plan`'s tree with every leaf a hybrid proof, as a verifier
/// derives it: [`hybrid_leaf_shape`] under the plan's own nodes.
pub(super) fn derive_hybrid_top(
    plan: &super::whir_block::WhirBlockPlan,
    opts: &crate::ProofOptions,
    window: Option<usize>,
) -> Result<super::registry::LfmArtifacts, String> {
    plan.derive_top_with(opts, window, |k| hybrid_leaf_shape(plan, k, opts))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field_vm::air::FvmPublicInputs;
    use crate::field_vm::asm::Asm;
    use crate::field_vm::executor::{Memory, NoHints, execute as execute_fvm};
    use crate::field_vm::hash_side;
    use crate::field_vm::isa::{Arg, gpr};
    use crate::hash_pin::{BlockFamily, ProofFamily};
    use crate::lfm::instr::Addr;
    use crate::lfm::per_table_aggregator::DerivedChild;
    use crate::lfm::word::LfmWord;
    use crate::tables::types::{FE, FEE};
    use crypto::fiat_shamir::is_transcript::{IsStarkTranscript, IsTranscript};
    use math::field::traits::IsPrimeField;
    use stark::proof::view::MultiProofView;

    fn fvm_harvest_and_arenas(
        pid: &[Commitment],
        cells: &[(u64, FEE)],
        side: &hash_side::HashSideId,
        mem_lfm: bool,
        proof: &crate::field_vm::prove::FvmProof,
        opts: &crate::ProofOptions,
        segments: &[FvmPublicInputs],
    ) -> (
        super::super::per_table_aggregator::DerivedChild,
        Vec<Vec<LfmWord>>,
    ) {
        let extra = hash_side::airs(side, opts);
        let airs = crate::field_vm::prove::Airs::new(pid, opts, segments.len(), extra, mem_lfm);
        let refs = airs.refs();
        let heights: Vec<u8> = proof
            .proofs
            .iter()
            .map(|p| p.trace_length.trailing_zeros() as u8)
            .collect();
        assert_eq!(refs.len(), proof.proofs.len());
        assert_eq!(heights.len(), refs.len());

        let view = MultiProofView::Owned(proof);
        let mut transcript = crate::field_vm::prove::transcript(pid, cells, segments);
        for (i, air) in refs.iter().enumerate() {
            let v = view.get(i);
            if air.is_preprocessed() {
                transcript.append_bytes(
                    &super::super::harvest::layout_precomputed_commitment(*air, v.trace_length())
                        .expect("preprocessed FVM root"),
                );
            }
            transcript.append_bytes(v.lde_trace_main_merkle_root());
        }
        let lookup: Vec<FEE> = (0..stark::lookup::LOGUP_NUM_CHALLENGES)
            .map(|_| transcript.sample_field_element())
            .collect();
        let mut tables = Vec::with_capacity(refs.len());
        let mut legs = Vec::with_capacity(refs.len());
        for (idx, air) in refs.iter().enumerate() {
            let v = view.get(idx);
            let pi = v.public_inputs().expect("proof carries public inputs");
            let mut fork = transcript.clone();
            if refs.len() > 1 {
                fork.append_bytes(&(idx as u64).to_le_bytes());
            }
            if let Some(root) = v.lde_trace_aux_merkle_root() {
                fork.append_bytes(root);
            }
            if let Some(c) = v.bus_table_contribution() {
                fork.append_field_element(&c);
            }
            tables.push(
                super::super::harvest::host_table_forked_with_inputs::<BlockFamily, _>(
                    *air,
                    v,
                    &pi,
                    idx,
                    refs.len(),
                    &mut fork,
                    &lookup,
                )
                .expect("FVM table harvest"),
            );
            legs.push(
                super::super::harvest::build_table_legs_with_inputs(
                    *air,
                    v,
                    &pi,
                    &lookup,
                    stark::config::TreeGeometry::of::<<BlockFamily as ProofFamily>::Stark>(),
                    Some(&tables[idx].iotas),
                    BlockFamily::WRAP,
                )
                .expect("FVM table legs"),
            );
        }

        let mut public_words = Vec::with_capacity(cells.len());
        for (index, (addr, value)) in cells.iter().enumerate() {
            let lanes = value.value();
            public_words.push((
                index as u32,
                [
                    FE::from(*addr),
                    FE::from(lanes[0].canonical()),
                    FE::from(lanes[1].canonical()),
                    FE::from(lanes[2].canonical()),
                ],
            ));
        }
        let arenas = super::super::harvest::try_child_arenas_for::<FvmPublicInputs>(
            BlockFamily::WRAP,
            &public_words,
            &tables,
            &legs,
        )
        .expect("FVM arena serialization");
        let child = super::super::per_table_aggregator::DerivedChild::from_hybrid(
            HybridArtifacts {
                pid: pid.to_vec(),
                side: side.clone(),
                segments: segments.len(),
                mem_lfm,
                heights,
            },
            opts,
            cells.len(),
            segments.to_vec(),
        )
        .expect("hybrid child shape");
        (child, arenas)
    }

    fn empty_hash_side() -> hash_side::HashSideId {
        hash_side::HashSideId {
            bridge: None,
            hash: Vec::new(),
            lanes: Vec::new(),
            bitdec: None,
            select: Vec::new(),
            hint: Vec::new(),
            const_: None,
            balu: Vec::new(),
            xalu: Vec::new(),
            keccak: None,
            bitwise: None,
            hasher: crate::lfm::hash::HasherKind::Test,
            one_row: Vec::new(),
        }
    }

    fn one_child_program(
        child: &DerivedChild,
    ) -> (crate::lfm::compiler::LfmProgram, Vec<(usize, usize, usize)>) {
        let shape = child.shape();
        let mut b = crate::lfm::builder::LfmBuilder::new().with_wrap_hash(BlockFamily::WRAP);
        let declared = super::super::per_table_aggregator::declare_leg_arenas(&mut b, &shape);
        let (_, ranges) = super::super::per_table_aggregator::record_table_legs(|| {
            super::super::per_table_aggregator::emit_leg(&mut b, &shape, &declared)
        });
        (crate::lfm::compiler::compile(b.finish()), ranges)
    }

    /// Small, non-hash Field VM proof through the same one-child node path as
    /// the hybrid leaf. This isolates FIELD_VM, DECODE, and MEM from hash-side
    /// tables while keeping the BlockFamily STARK and wrap options.
    #[test]
    fn small_field_vm_proof_verifies_in_one_child_node() {
        let _ = crate::lfm::epoch::take_test_table_challenge_cells();
        let _ = crate::lfm::statement_replay::take_test_phase_a_challenge_cells();
        let opts = super::super::proof::aggregation_wrap_options();
        let (acc, count) = (gpr(0), gpr(1));
        let mut asm = Asm::new();
        asm.mov(Arg::reg(acc), Arg::imm(1))
            .mov(Arg::reg(count), Arg::imm(3));
        let loop_top = asm.label();
        asm.bind(loop_top);
        asm.mul(Arg::reg(acc), Arg::reg(acc), Arg::mem_abs(0))
            .sub(Arg::reg(count), Arg::reg(count), Arg::imm(1))
            .jump_not_zero(loop_top)
            .store(Arg::mem_abs(1), Arg::reg(acc))
            .halt();
        let program = asm.finish().expect("small FVM program");
        let execution = execute_fvm(
            &program,
            Memory::with_init(2, &[FEE::from(5u64)]),
            &mut NoHints,
            1 << 12,
        )
        .expect("FVM execution");
        let public = crate::field_vm::prove::public_cells(&execution, &[0, 1]);
        let pid = crate::field_vm::prove::program_id(&program, &opts);
        let mut traces = crate::field_vm::prove::generate_traces(&program, &execution, &[0, 1]);
        let segments = traces.segments.clone();
        let proof = crate::field_vm::prove::prove_traces(&pid, &public, &mut traces, &opts)
            .expect("Field VM proof");
        assert!(crate::field_vm::prove::verify(&pid, &public, &proof, &opts));

        // Independently replay the prover transcript from the proof's actual
        // Phase-A commitments, then table 0 through its OOD challenge.
        let mut host = crate::field_vm::prove::transcript(&pid, &public, &segments);
        for p in &proof.proofs {
            if let Some(root) = p.lde_trace_precomputed_merkle_root {
                host.append_bytes(&root);
            }
            host.append_bytes(&p.lde_trace_main_merkle_root);
        }
        let host_z = host.sample_field_element();
        let host_alpha = host.sample_field_element();
        let p0 = &proof.proofs[0];
        let mut table0 = host.clone();
        if proof.proofs.len() > 1 {
            table0.append_bytes(&0u64.to_le_bytes());
        }
        if let Some(root) = p0.lde_trace_aux_merkle_root {
            table0.append_bytes(&root);
        }
        if let Some(contribution) = p0.bus_public_inputs.as_ref().map(|x| x.table_contribution) {
            table0.append_field_element(&contribution);
        }
        let host_beta0 = table0.sample_field_element();
        table0.append_bytes(&p0.composition_poly_root);
        let host_ood0 = table0.sample_z_ood_with_domain_params(
            p0.trace_length,
            p0.trace_length * opts.blowup_factor as usize,
            &FE::from(opts.coset_offset),
        );
        let statement_len = crate::field_vm::prove::DOMAIN_TAG.len()
            + 8
            + pid.len() * 32
            + 8
            + public.len() * 32
            + 8
            + segments
                .iter()
                .map(|pi| 8 + (pi.first.len() + pi.last.len()) * 8)
                .sum::<usize>();
        eprintln!(
            "FVM transcript host: domain={:?} tables={} statement_bytes={statement_len} PhaseA LogUp(z,alpha)=({host_z:?},{host_alpha:?}) table0_beta={host_beta0:?} table0_ood={host_ood0:?}",
            crate::field_vm::prove::DOMAIN_TAG,
            proof.proofs.len(),
        );
        eprintln!("FVM statement fields: pid={pid:?} public={public:?} segments={segments:?}");
        eprintln!(
            "FVM Phase-A roots: table0 prep(proof)={:?} main={:?}; table0 aux={:?} composition={:?}",
            p0.lde_trace_precomputed_merkle_root,
            p0.lde_trace_main_merkle_root,
            p0.lde_trace_aux_merkle_root,
            p0.composition_poly_root
        );

        let side = empty_hash_side();
        let (child, arenas) =
            fvm_harvest_and_arenas(&pid, &public, &side, false, &proof, &opts, &segments);
        for (i, (table, p)) in child.tables.iter().zip(&proof.proofs).enumerate() {
            eprintln!(
                "FVM Phase-A table {i}: proof_prep={:?} node_prep={:?} main={:?}",
                p.lde_trace_precomputed_merkle_root,
                table.precomputed_root,
                p.lde_trace_main_merkle_root
            );
        }
        let (node, ranges) = one_child_program(&child);
        match super::super::executor::execute_with_memory_on_failure(
            &node,
            &arenas,
            &node.hasher(BlockFamily::SOCKET),
        ) {
            Ok(_) => {}
            Err((error, memory)) => {
                eprintln!("small FVM node failed: {error:?}");
                if let super::super::executor::LfmExecError::DivByZero { addr } = &error {
                    let mut buf = Vec::new();
                    if let Some(i) = node.instrs.iter().position(|ins| {
                        buf.clear();
                        ins.writes_into(&mut buf);
                        buf.iter().any(|a| a.0 == *addr)
                    }) {
                        eprintln!("  failing instruction {i} of {}", node.instrs.len());
                    }
                }
                if let Some((z_addr, alpha_addr)) =
                    crate::lfm::statement_replay::take_test_phase_a_challenge_cells()
                {
                    let z = memory
                        .get(Addr(z_addr))
                        .map(|w| FEE::new([w[0], w[1], w[2]]));
                    let alpha = memory
                        .get(Addr(alpha_addr))
                        .map(|w| FEE::new([w[0], w[1], w[2]]));
                    eprintln!(
                        "FVM transcript node Phase A: z_cell={z_addr} value={z:?} alpha_cell={alpha_addr} value={alpha:?}"
                    );
                    eprintln!(
                        "FVM Phase-A comparison: z_match={} alpha_match={}",
                        z == Some(host_z),
                        alpha == Some(host_alpha)
                    );
                }
                let cells = crate::lfm::epoch::take_test_table_challenge_cells();
                if let Some(&(beta_addr, z_addr)) = cells.first() {
                    let beta = memory
                        .get(Addr(beta_addr))
                        .map(|w| FEE::new([w[0], w[1], w[2]]));
                    let node_ood0 = memory
                        .get(Addr(z_addr))
                        .map(|w| FEE::new([w[0], w[1], w[2]]));
                    eprintln!(
                        "FVM transcript node: table0 beta_cell={beta_addr} value={beta:?} ood_cell={z_addr} value={node_ood0:?}; host_beta={host_beta0:?} beta_match={} host_ood={host_ood0:?} ood_match={}",
                        beta == Some(host_beta0),
                        node_ood0 == Some(host_ood0),
                    );
                } else {
                    eprintln!("FVM transcript node: no table challenge cells were recorded");
                }
                eprintln!(
                    "FIELD_VM[0] OOD verifier diagnostic requested; table ranges: {ranges:?}"
                );
                panic!("one-child node accepts the small Field VM proof");
            }
        }
    }

    /// End-to-end hybrid node gate: prove the small block's leaf 0 through the
    /// Field VM plus hash-chip split, harvest the proof's typed public inputs,
    /// and execute a one-child verifier. The statement's public cells are
    /// absorbed exactly as `emit_fvm_statement` reads the eight-half arena.
    #[test]
    #[ignore = "proves a small hybrid block leaf and executes its node; box tier"]
    fn one_child_hybrid_node_accepts_and_binds_public_cells() {
        let format = super::super::whir_block_tests::small_format();
        let (elf, block_proof) =
            super::super::whir_block_tests::small_block("test_commit_4", &format);
        let plan = super::super::whir_block_tests::plan_of(&elf, &block_proof, &format, Some(2));
        let (leaf, scopes) = super::super::fn_scope::record(|| plan.leaf_program(0));
        let leaf = leaf.expect("leaf program");
        let leaf_arenas =
            super::super::whir_block::block_leaf_arena(&plan, &block_proof, 0).expect("leaf arena");
        let lfm_exec =
            super::super::executor::execute(&leaf, &leaf_arenas, &leaf.hasher(BlockFamily::SOCKET))
                .expect("LFM leaf executes");
        let opts = super::super::proof::aggregation_wrap_options();
        let hybrid = super::super::fvm_whir_leaf::Hybrid::new(
            &leaf,
            &lfm_exec,
            &leaf_arenas,
            &opts,
            &scopes,
        );
        let (mut traces, mut extra) = hybrid.traces();
        let segments = traces.segments.clone();
        let (proof, _, _) = hybrid
            .prove(&mut traces, &mut extra, &opts)
            .expect("hybrid proof verifies");
        let (child, arenas) = fvm_harvest_and_arenas(
            &hybrid.pid,
            &hybrid.cells,
            &hybrid.id,
            true,
            &proof,
            &opts,
            &segments,
        );
        let (node, table_ranges) = one_child_program(&child);
        let airs = crate::field_vm::prove::Airs::new(
            &hybrid.pid,
            &opts,
            segments.len(),
            hash_side::airs(&hybrid.id, &opts),
            true,
        );
        let air_names: Vec<_> = airs.refs().iter().map(|air| air.name()).collect();
        if let Err(e) =
            super::super::executor::execute(&node, &arenas, &node.hasher(BlockFamily::SOCKET))
        {
            // Where the node refused: the instruction that wrote the failing
            // address, as a fraction of the program, and its neighbours.
            eprintln!("node refused: {e:?} ({} instructions)", node.instrs.len());
            if let super::super::executor::LfmExecError::DivByZero { addr } = &e {
                let mut buf = Vec::new();
                if let Some(i) = node.instrs.iter().position(|ins| {
                    buf.clear();
                    ins.writes_into(&mut buf);
                    buf.iter().any(|a| a.0 == *addr)
                }) {
                    if let Some((table_idx, start, end)) = table_ranges
                        .iter()
                        .find(|(_, start, end)| *start <= i && i < *end)
                    {
                        eprintln!(
                            "  failing instruction belongs to table {table_idx} ({}) [{start}..{end})",
                            air_names.get(*table_idx).copied().unwrap_or("unknown AIR")
                        );
                    }
                    eprintln!(
                        "  written by instruction {i} ({:.1}%)",
                        100.0 * i as f64 / node.instrs.len() as f64
                    );
                    for j in i.saturating_sub(6)..(i + 2).min(node.instrs.len()) {
                        eprintln!("    {j}: {:?}", node.instrs[j]);
                    }
                }
            }
            panic!("the one-child node accepts its hybrid proof");
        }

        let mut changed = arenas;
        assert!(!changed[0].is_empty(), "public cells are serialized first");
        changed[0][0][0] += FE::one();
        assert!(
            super::super::executor::execute(&node, &changed, &node.hasher(BlockFamily::SOCKET),)
                .is_err(),
            "changing a public cell makes the node reject"
        );
    }

    /// Two hybrid leaves pass through the block node's shared publication
    /// bindings. The negative case changes the second child's state lane 3,
    /// carried by its companion Field VM public cell.
    #[test]
    #[ignore = "proves two hybrid block leaves and executes their node; box tier"]
    fn two_hybrid_children_bind_the_complete_block_publications() {
        let format = super::super::whir_block_tests::small_format();
        let (elf, block_proof) =
            super::super::whir_block_tests::small_block("test_commit_4", &format);
        let plan = super::super::whir_block_tests::plan_of(&elf, &block_proof, &format, Some(2));
        let opts = super::super::proof::aggregation_wrap_options();
        let mut children = Vec::new();
        let mut child_arenas = Vec::new();
        for leaf_index in 0..2 {
            let (leaf, scopes) = super::super::fn_scope::record(|| plan.leaf_program(leaf_index));
            let leaf = leaf.expect("leaf program");
            let leaf_arenas =
                super::super::whir_block::block_leaf_arena(&plan, &block_proof, leaf_index)
                    .expect("leaf arena");
            let lfm_exec = super::super::executor::execute(
                &leaf,
                &leaf_arenas,
                &leaf.hasher(BlockFamily::SOCKET),
            )
            .expect("LFM leaf executes");
            let hybrid = super::super::fvm_whir_leaf::Hybrid::new(
                &leaf,
                &lfm_exec,
                &leaf_arenas,
                &opts,
                &scopes,
            );
            let (mut traces, mut extra) = hybrid.traces();
            let segments = traces.segments.clone();
            let (proof, _, _) = hybrid
                .prove(&mut traces, &mut extra, &opts)
                .expect("hybrid proof verifies");
            let (child, arenas) = fvm_harvest_and_arenas(
                &hybrid.pid,
                &hybrid.cells,
                &hybrid.id,
                true,
                &proof,
                &opts,
                &segments,
            );
            children.push(child);
            child_arenas.push(arenas);
        }

        let shapes: Vec<_> = children.iter().map(DerivedChild::shape).collect();
        let layout = plan.child_layout();
        let mut b = super::super::builder::LfmBuilder::new().with_wrap_hash(BlockFamily::WRAP);
        let (_, table_ranges) = super::super::per_table_aggregator::record_table_legs(|| {
            super::super::block_node::emit_block_node(
                &mut b,
                &super::super::block_node::BlockNodeInputs {
                    children: &shapes,
                    layout,
                    top: false,
                },
            )
        });
        let node = super::super::compiler::compile(b.finish());
        let first_arena_count = child_arenas[0].len();
        let mut arenas: Vec<Vec<LfmWord>> = child_arenas.into_iter().flatten().collect();
        // Test-only binding trace: a hybrid child's public arena contains
        // `[addr, c0, c1, c2]` primary tuples followed by companion tuples
        // `[addr + 2^39, c0, 0, 0]`. Print the addresses independently from
        // the values, then reconstruct each BlockLayout word as
        // `[primary.c0, primary.c1, primary.c2, companion.c0]`.
        fn tuple_lane(arena: &[LfmWord], word: usize, lane: usize) -> u64 {
            let offset = (word * 4 + lane) * 2;
            let lo = crate::tables::types::GoldilocksField::canonical(arena[offset][0].value());
            let hi = crate::tables::types::GoldilocksField::canonical(arena[offset + 1][0].value());
            lo | (hi << 32)
        }
        let logical_words = shapes[0].num_public_words / 2;
        let companion_start = logical_words;
        let word_name = |index: usize| {
            if index < 2 {
                format!("id{}", index)
            } else if index < 2 + layout.state_words {
                format!("state{}", index - 2)
            } else if index < 2 + layout.state_words + layout.out_halves {
                format!("out{}", index - 2 - layout.state_words)
            } else {
                "sum".to_owned()
            }
        };
        eprintln!("block binding comparisons (expected from child 0):");
        for child_index in 0..shapes.len() {
            let arena_index = if child_index == 0 {
                0
            } else {
                first_arena_count
            };
            let public_arena = &arenas[arena_index];
            for word in 0..layout.total() {
                let primary_addr = tuple_lane(public_arena, word, 0);
                let companion_addr = tuple_lane(public_arena, companion_start + word, 0);
                eprintln!(
                    "  word={} id/state/out/sum={} child={} addresses primary={} companion={}",
                    word,
                    word_name(word),
                    child_index,
                    primary_addr,
                    companion_addr,
                );
                for lane in 0..4 {
                    let (value, expected) = if lane == 3 {
                        (
                            tuple_lane(public_arena, companion_start + word, 1),
                            tuple_lane(&arenas[0], companion_start + word, 1),
                        )
                    } else {
                        (
                            tuple_lane(public_arena, word, lane + 1),
                            tuple_lane(&arenas[0], word, lane + 1),
                        )
                    };
                    eprintln!(
                        "  word={} id/state/out/sum={} child={} value_lane={} value={} expected={}",
                        word,
                        word_name(word),
                        child_index,
                        lane,
                        value,
                        expected,
                    );
                }
            }
        }
        if let Err(e) =
            super::super::executor::execute(&node, &arenas, &node.hasher(BlockFamily::SOCKET))
        {
            eprintln!(
                "two-child node refused: {e:?} ({} instructions)",
                node.instrs.len()
            );
            if let super::super::executor::LfmExecError::DivByZero { addr } = &e {
                let mut buf = Vec::new();
                if let Some(i) = node.instrs.iter().position(|ins| {
                    buf.clear();
                    ins.writes_into(&mut buf);
                    buf.iter().any(|a| a.0 == *addr)
                }) {
                    if let Some((table_idx, start, end)) = table_ranges
                        .iter()
                        .find(|(_, start, end)| *start <= i && i < *end)
                    {
                        eprintln!(
                            "  failing instruction belongs to table {table_idx} [{start}..{end})"
                        );
                    } else {
                        eprintln!("  failing instruction {i} belongs to the block binding leg");
                    }
                    eprintln!(
                        "  written by instruction {i} ({:.1}%)",
                        100.0 * i as f64 / node.instrs.len() as f64
                    );
                    for j in i.saturating_sub(6)..(i + 2).min(node.instrs.len()) {
                        eprintln!("    {j}: {:?}", node.instrs[j]);
                    }
                }
            }
            panic!("the two-child node executes successfully");
        }

        // Each public cell occupies eight half-words. The state is word 2;
        // its lane-3 value is c0 (lane 1) of the companion tuple in the second
        // sorted half, after that tuple's address lane.
        let public_cell_count = shapes[0].num_public_words / 2;
        let state_lane3_half = (public_cell_count / 2 + layout.state(0)) * 8 + 2;
        arenas[first_arena_count][state_lane3_half][0] += FE::one();
        assert!(
            super::super::executor::execute(&node, &arenas, &node.hasher(BlockFamily::SOCKET))
                .is_err()
        );
    }

    /// Rows and committed cells (trace length times opened width) of a proof.
    fn proof_cells<PI>(
        proof: &stark::proof::stark::MultiProof<
            crate::tables::types::GoldilocksField,
            crate::tables::types::GoldilocksExtension,
            PI,
        >,
    ) -> usize {
        proof
            .proofs
            .iter()
            .map(|p| p.trace_length * p.trace_ood_evaluations.width)
            .sum()
    }

    /// One LFM program of a tree proved under the block family: the proof,
    /// the arenas its parent reads, its derived shape, cells and seconds.
    fn prove_tree_lfm(
        program: &crate::lfm::compiler::LfmProgram,
        arenas: &[Vec<LfmWord>],
        words: usize,
    ) -> (
        super::super::proof::LfmProof,
        Vec<Vec<LfmWord>>,
        DerivedChild,
        usize,
        f64,
    ) {
        let opts = super::super::proof::aggregation_wrap_options();
        let t = std::time::Instant::now();
        let artifacts = super::super::whir_block::artifacts_of(program, &opts);
        let derived = super::super::whir_block::derived_of(&artifacts, &opts, words).unwrap();
        let proof =
            super::super::proof::lfm_prove_under::<BlockFamily>(program, &artifacts, arenas, &opts)
                .expect("the program proves");
        let secs = t.elapsed().as_secs_f64();
        let (child, _) = super::super::harvest::harvest_child_verified_under::<BlockFamily>(
            artifacts, opts, &proof,
        )
        .expect("the proof verifies");
        let cells = proof_cells(&proof.proof);
        (
            proof,
            super::super::per_table_aggregator_tests::child_arena_words(&child),
            derived,
            cells,
            secs,
        )
    }

    /// ★ Option A through the block's recursion: the small block's tree with
    /// every leaf a hybrid proof (the leaf's field half on the Field VM, its
    /// hash half on the LFM chips) and the plan's own top over them, which
    /// claims the block. The top refuses a leaf's changed state lane 3 (its
    /// companion public cell) and a changed id cell. The same block's LFM tree
    /// alongside, level by level: committed cells and prove seconds.
    #[test]
    #[ignore = "proves the small block's hybrid and LFM trees; box tier"]
    fn small_block_hybrid_tree_claims_the_block() {
        let format = super::super::whir_block_tests::small_format();
        let (elf, block_proof) =
            super::super::whir_block_tests::small_block("test_commit_4", &format);
        let plan = super::super::whir_block_tests::plan_of(&elf, &block_proof, &format, Some(2));
        let opts = super::super::proof::aggregation_wrap_options();
        let layout = plan.child_layout();
        let levels = plan.levels();
        assert_eq!(levels.len(), 1, "the small block's leaves close in the top");
        let leaves = plan.partition().num_leaves();

        let mut hybrid_children = Vec::new();
        let mut hybrid_arenas = Vec::new();
        let (mut a_cells, mut a_secs) = (0usize, 0.0f64);
        let mut lfm_children = Vec::new();
        let mut lfm_arenas = Vec::new();
        let (mut l_cells, mut l_secs) = (0usize, 0.0f64);
        for k in 0..leaves {
            let (leaf, scopes) = super::super::fn_scope::record(|| plan.leaf_program(k));
            let leaf = leaf.expect("leaf program");
            let leaf_arenas =
                super::super::whir_block::block_leaf_arena(&plan, &block_proof, k).unwrap();
            let exec = super::super::executor::execute(
                &leaf,
                &leaf_arenas,
                &leaf.hasher(BlockFamily::SOCKET),
            )
            .expect("LFM leaf executes");
            let hybrid = super::super::fvm_whir_leaf::Hybrid::new(
                &leaf,
                &exec,
                &leaf_arenas,
                &opts,
                &scopes,
            );
            let (mut traces, mut extra) = hybrid.traces();
            let segments = traces.segments.clone();
            let t = std::time::Instant::now();
            let (proof, _, _) = hybrid
                .prove(&mut traces, &mut extra, &opts)
                .expect("hybrid proof verifies");
            let secs = t.elapsed().as_secs_f64();
            let cells = proof_cells(&proof);
            let (child, arenas) = fvm_harvest_and_arenas(
                &hybrid.pid,
                &hybrid.cells,
                &hybrid.id,
                true,
                &proof,
                &opts,
                &segments,
            );
            eprintln!(
                "  A leaf {k}: {} tables, {:.1} M cells, prove+verify {secs:.1} s",
                proof.proofs.len(),
                cells as f64 / 1e6
            );
            a_cells += cells;
            a_secs += secs;
            hybrid_children.push(child);
            hybrid_arenas.push(arenas);

            let (_, arenas, derived, cells, secs) =
                prove_tree_lfm(&leaf, &leaf_arenas, layout.total());
            eprintln!(
                "  LFM leaf {k}: {} instructions, {:.1} M cells, artifacts+prove {secs:.1} s",
                leaf.instrs.len(),
                cells as f64 / 1e6
            );
            l_cells += cells;
            l_secs += secs;
            lfm_children.push(derived);
            lfm_arenas.push(arenas);
        }
        eprintln!(
            "LEVEL 0: A {:.1} M cells {a_secs:.1} s · LFM {:.1} M cells {l_secs:.1} s",
            a_cells as f64 / 1e6,
            l_cells as f64 / 1e6
        );

        let top_of = |children: &[DerivedChild]| {
            let refs: Vec<&DerivedChild> = children.iter().collect();
            plan.node_program(&refs, true).expect("the top emits")
        };
        let flat = |arenas: &[Vec<Vec<LfmWord>>]| -> Vec<Vec<LfmWord>> {
            arenas.iter().flatten().cloned().collect()
        };
        let a_top = top_of(&hybrid_children);
        let a_arenas = flat(&hybrid_arenas);
        let (a_proof, _, _, cells, secs) =
            prove_tree_lfm(&a_top, &a_arenas, plan.top_layout().total());
        assert!(
            super::super::whir_block::top_claims(&plan, &a_proof.public_words),
            "the hybrid tree's top claims the block"
        );
        eprintln!(
            "TOP: A {} instructions {:.1} M cells {secs:.1} s",
            a_top.instrs.len(),
            cells as f64 / 1e6
        );
        let l_top = top_of(&lfm_children);
        let (l_proof, _, _, cells, secs) =
            prove_tree_lfm(&l_top, &flat(&lfm_arenas), plan.top_layout().total());
        assert!(super::super::whir_block::top_claims(
            &plan,
            &l_proof.public_words
        ));
        assert_eq!(
            a_proof.public_words, l_proof.public_words,
            "both tops claim the same"
        );
        eprintln!(
            "TOP: LFM {} instructions {:.1} M cells {secs:.1} s",
            l_top.instrs.len(),
            cells as f64 / 1e6
        );

        // Tamper leaf 1's publics in the top's arenas: the state's lane 3 (a
        // companion cell's c0) and the first id cell's c0.
        let second = hybrid_arenas[0].len();
        let logical = hybrid_children[1].shape().num_public_words / 2;
        let state_lane3 = (logical + layout.state(0)) * 8 + 2;
        let id_c0 = layout.id(0) * 8 + 2;
        for (what, at) in [("state lane 3", state_lane3), ("id", id_c0)] {
            let mut changed = a_arenas.clone();
            changed[second][at][0] += FE::one();
            assert!(
                super::super::executor::execute(
                    &a_top,
                    &changed,
                    &a_top.hasher(BlockFamily::SOCKET)
                )
                .is_err(),
                "the top refuses leaf 1's changed {what}"
            );
        }
    }

    /// ★ Option A's tree derived as a verifier derives it: each leaf's shape
    /// from the plan alone, no proof of it. Each derived leaf has the proved
    /// leaf's heights and id, the derived top is the top over the proved
    /// leaves (its proof verifies against it and claims the block), and a
    /// leaf derived one table taller gives a top that proof is refused by.
    #[test]
    #[ignore = "proves the small block's hybrid leaves and top; box tier"]
    fn small_block_hybrid_tree_derives_its_top() {
        // SAFETY: an ignored test run alone; nothing else reads the
        // environment concurrently.
        unsafe { std::env::set_var("FVM_ONE_SEGMENT", "1") };
        let format = super::super::whir_block_tests::small_format();
        let (elf, block_proof) =
            super::super::whir_block_tests::small_block("test_commit_4", &format);
        let plan = super::super::whir_block_tests::plan_of(&elf, &block_proof, &format, Some(2));
        let opts = super::super::proof::aggregation_wrap_options();
        assert_eq!(
            plan.levels().len(),
            1,
            "the small block's leaves close in the top"
        );

        let mut children = Vec::new();
        let mut arenas = Vec::new();
        for k in 0..plan.partition().num_leaves() {
            let (leaf, scopes) = super::super::fn_scope::record(|| plan.leaf_program(k));
            let leaf = leaf.expect("leaf program");
            let leaf_arenas =
                super::super::whir_block::block_leaf_arena(&plan, &block_proof, k).unwrap();
            let exec = super::super::executor::execute(
                &leaf,
                &leaf_arenas,
                &leaf.hasher(BlockFamily::SOCKET),
            )
            .expect("LFM leaf executes");
            let hybrid = super::super::fvm_whir_leaf::Hybrid::new(
                &leaf,
                &exec,
                &leaf_arenas,
                &opts,
                &scopes,
            );
            let (mut traces, mut extra) = hybrid.traces();
            let segments = traces.segments.clone();
            let (proof, _, _) = hybrid
                .prove(&mut traces, &mut extra, &opts)
                .expect("hybrid proof verifies");
            let (child, a) = fvm_harvest_and_arenas(
                &hybrid.pid,
                &hybrid.cells,
                &hybrid.id,
                true,
                &proof,
                &opts,
                &segments,
            );
            let derived = super::hybrid_leaf_shape(&plan, k, &opts).expect("the leaf's shape");
            assert_eq!(
                derived.log2_trace_lengths(),
                child.log2_trace_lengths(),
                "leaf {k}: the derived heights are the proof's"
            );
            assert_eq!(
                derived.shape().program_id,
                child.shape().program_id,
                "leaf {k}: the derived Field VM program is the proved one"
            );
            eprintln!(
                "  leaf {k}: {} tables, {:.1} M cells, heights {:?}",
                proof.proofs.len(),
                proof_cells(&proof) as f64 / 1e6,
                derived.log2_trace_lengths()
            );
            children.push(child);
            arenas.push(a);
        }

        let refs: Vec<&DerivedChild> = children.iter().collect();
        let top = plan.node_program(&refs, true).expect("the top emits");
        let built = super::super::whir_block::artifacts_of(&top, &opts);
        let t = std::time::Instant::now();
        let derived = super::derive_hybrid_top(&plan, &opts, None).expect("the top derives");
        eprintln!(
            "  derived the top from the plan in {:.1} s",
            t.elapsed().as_secs_f64()
        );
        assert_eq!(
            derived.program_id, built.program_id,
            "the derived top is the top over the proved leaves"
        );
        let flat: Vec<Vec<LfmWord>> = arenas.iter().flatten().cloned().collect();
        let (proof, _, _, cells, secs) = prove_tree_lfm(&top, &flat, plan.top_layout().total());
        eprintln!(
            "TOP: {} instructions {:.1} M cells {secs:.1} s",
            top.instrs.len(),
            cells as f64 / 1e6
        );
        assert!(super::super::whir_block::top_claims(
            &plan,
            &proof.public_words
        ));
        super::super::harvest::harvest_child_verified_under::<BlockFamily>(
            derived,
            opts.clone(),
            &proof,
        )
        .expect("the top's proof verifies against the derived top");

        let taller = plan
            .derive_top_with(&opts, None, |k| {
                let (leaf, scopes) = super::super::fn_scope::record(|| plan.leaf_program(k));
                let (mut artifacts, cells, segments) =
                    super::super::fvm_whir_leaf::Hybrid::shape(&leaf?, &opts, &scopes)?;
                if k == 0 {
                    artifacts.heights[0] += 1;
                }
                DerivedChild::from_hybrid(artifacts, &opts, cells, segments)
            })
            .expect("the taller top derives");
        assert_ne!(taller.program_id, built.program_id);
        assert!(
            super::super::harvest::harvest_child_verified_under::<BlockFamily>(
                taller,
                opts.clone(),
                &proof,
            )
            .is_err(),
            "a top over a leaf one table taller refuses the proof"
        );
    }
}
