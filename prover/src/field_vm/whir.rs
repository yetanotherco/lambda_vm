//! WHIR/multilinear proof path for the Field VM.
//!
//! Unlike the STARK AIR boundary system, this path binds segment endpoints
//! through the signed `FieldVmState` LogUp bus. Program and public statement
//! data are absorbed before the table roots; the verifier then closes public
//! memory and state buses at the LogUp challenges.

/// Runs `$body` with `$h` the block path's WHIR hash (`hash_pin::BlockWhirHash`,
/// Poseidon1), so the Field VM proves under the same family as the block.
macro_rules! with_block_hash {
    (|$h:ident| $body:block) => {{
        #[allow(non_camel_case_types)]
        type $h = crate::hash_pin::BlockWhirHash;
        $body
    }};
}

use std::sync::Arc;

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::fiat_shamir::is_transcript::IsTranscript;
use math::field::element::FieldElement;
use multilinear::mle::Mle;
use stark::constraints::builder::EmptyConstraints;
use stark::lookup::{AirWithBuses, AuxiliaryTraceBuildData, NullBoundaryConstraintBuilder};
use stark::multilinear_air::Uniforms;
use stark::multilinear_block::{self, ArgueDeviation, BlockCommitted};
use stark::multilinear_table::{
    self, BatchedArgue, CommittedTable, CommittedTables, MultiProof, TableLayout, TableStatement,
};
use stark::proof::options::ProofOptions;
use stark::trace::TraceTable;
use stark::traits::AIR;

use super::air::{self, FieldVmConstraints, FvmPublicInputs};
use super::cksum::{self, CksumConstraints};
use super::decode;
use super::isa::Program;
use super::mem::{self, MemBoundary, MemConstraints};
use super::prove::{self, ExtraAir, FvmTraces, PublicCell};
use crate::tables::types::{BusId, FE, FEE, GoldilocksExtension, GoldilocksField};

type F = GoldilocksField;
type E = GoldilocksExtension;
pub(crate) type WhirAir = dyn AIR<Field = F, FieldExtension = E, PublicInputs = FvmPublicInputs>;
pub(crate) type AirBox = Box<WhirAir>;
type Multi = MultiProof<F, E>;

/// A Field VM multilinear proof and the public endpoint claims needed because
/// multilinear AIRs deliberately do not run STARK boundary constraints.
#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct FvmWhirProof {
    pub proof: Multi,
    /// Height in variables, in table order: segments, decode chunks, MEM,
    /// then any extra tables.
    pub table_num_vars: Vec<u8>,
    pub segments: Vec<FvmPublicInputs>,
}

/// A Field VM proof in the block recursion's grouped format. A recursion leaf
/// emitter can verify this once it has a frame derived from the Field VM AIRs:
/// AIR refs in table order, `(width, vars)` shapes, the groups and their table
/// order, per-group stacked layouts and domains, and the corresponding chain
/// configuration. It also needs the segment chain and public-cell statement
/// values to build the owed bus balance. DECODE's program-derived columns are
/// the prepared stack openings carried here.
#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct FvmWhirBlockProof {
    pub proof: Multi,
    pub table_num_vars: Vec<u8>,
    pub segments: Vec<FvmPublicInputs>,
    pub groups: Vec<Vec<u32>>,
    pub prepared: Vec<multilinear::stacked_eval::StackedProof<F, E>>,
    pub argues: Vec<BatchedArgue<E>>,
}

pub(crate) struct WhirAirs {
    pub(crate) airs: Vec<AirBox>,
}

impl WhirAirs {
    pub(crate) fn new(
        program: &Program,
        roots: &[stark::config::Commitment],
        segment_count: usize,
        mem_rows: usize,
        options: &ProofOptions,
        extra: Vec<ExtraAir>,
    ) -> Self {
        let aux = |interactions| AuxiliaryTraceBuildData { interactions };
        let mut airs: Vec<AirBox> = Vec::new();
        let fvm = AirWithBuses::<
            F,
            E,
            NullBoundaryConstraintBuilder,
            FvmPublicInputs,
            FieldVmConstraints,
        >::new(
            air::cols::NUM_COLUMNS,
            aux(air::bus_interactions()),
            options,
            1,
            FieldVmConstraints::new(),
        );
        airs.extend(
            (0..segment_count)
                .map(|k| Box::new(fvm.clone().with_name(&format!("FIELD_VM[{k}]"))) as AirBox),
        );

        let decode_preprocessed = decode::generate_traces(program, &[], prove::MIN_ROWS);
        assert_eq!(
            roots.len(),
            decode_preprocessed.len(),
            "one program root per decode chunk"
        );
        for (k, (root, prep)) in roots.iter().zip(decode_preprocessed).enumerate() {
            let columns = (0..decode::NUM_PRECOMPUTED_COLS)
                .map(|c| {
                    (0..prep.num_rows())
                        .map(|r| *prep.main_table.get(r, c))
                        .collect()
                })
                .collect::<Vec<Vec<FE>>>();
            let generator = Arc::new(move || columns.clone());
            let a = AirWithBuses::<
                F,
                E,
                NullBoundaryConstraintBuilder,
                FvmPublicInputs,
                EmptyConstraints,
            >::new(
                decode::NUM_COLUMNS,
                aux(decode::bus_interactions()),
                options,
                1,
                EmptyConstraints,
            )
            .with_name(&format!("FIELD_VM_DECODE[{k}]"))
            .with_preprocessed_columns(*root, decode::NUM_PRECOMPUTED_COLS, generator);
            airs.push(Box::new(a));
        }

        // ADDR is generated from the row index and checked as a preprocessed
        // column by WHIR. Its trace is the unique address ramp.
        let ramp = (0..mem_rows)
            .map(|r| FE::from(r as u64))
            .collect::<Vec<_>>();
        let mem_prep = Arc::new(move || vec![ramp.clone()]);
        let mem_a = AirWithBuses::<F, E, MemBoundary, FvmPublicInputs, MemConstraints>::new(
            mem::cols::NUM_COLUMNS,
            aux(mem::bus_interactions()),
            options,
            1,
            MemConstraints { lfm: false },
        )
        .with_name("FIELD_VM_MEM")
        .with_preprocessed_columns([0; 32], 1, mem_prep);
        airs.push(Box::new(mem_a));
        airs.extend(extra);
        Self { airs }
    }

    fn refs(&self) -> Vec<&WhirAir> {
        self.airs.iter().map(|a| a.as_ref()).collect()
    }
}

/// The tables whose preprocessed columns a block proof opens as prepared
/// stacks: the DECODE chunks and every extra that carries some (CKSUM's
/// addresses, too tall for a recursion leaf to evaluate as constants). MEM's
/// ramp has a closed form.
pub(crate) fn prepared_tables(
    refs: &[&WhirAir],
    segments: usize,
    decode_count: usize,
) -> Vec<(usize, Vec<Vec<FE>>)> {
    (segments..segments + decode_count)
        .chain(
            (segments + decode_count + 1..refs.len())
                .filter(|&i| refs[i].num_precomputed_columns() > 0),
        )
        .map(|i| (i, refs[i].precomputed_columns()))
        .collect()
}

fn layout_of<'a>(
    air: &'a WhirAir,
    width: usize,
    num_vars: usize,
) -> Result<TableLayout<'a, F, E>, String> {
    TableLayout::<F, E>::new(
        air.constraint_program(),
        air.constraints_meta(),
        air.bus_interactions(),
        width,
        num_vars,
        Uniforms::default(),
    )
    .map_err(|e| format!("{}: {e:?}", air.name()))
}

fn preprocessed(air: &WhirAir) -> Result<Vec<Mle<F>>, String> {
    air.precomputed_columns()
        .into_iter()
        .map(|v| Mle::new(v).map_err(|e| format!("{}: {e:?}", air.name())))
        .collect()
}

fn committed_tables<'a>(
    pairs: &[(&'a WhirAir, &'a TraceTable<F, E>)],
    shapes: &[(usize, usize)],
) -> Result<Vec<CommittedTable<'a, F, E>>, String> {
    if pairs.len() != shapes.len() {
        return Err("table and shape counts differ".into());
    }
    pairs
        .iter()
        .zip(shapes)
        .map(|(&(air, trace), &(width, vars))| {
            let layout = layout_of(air, width, vars)?;
            let columns = trace.columns_main();
            for (i, expected) in air.precomputed_columns().iter().enumerate() {
                if columns.get(i) != Some(expected) {
                    return Err(format!("{}: preprocessed column {i} mismatch", air.name()));
                }
            }
            CommittedTable::from_layout(layout, |c| columns[c as usize].clone())
                .map_err(|e| format!("{e:?}"))
        })
        .collect()
}

/// Build the optional CKSUM extra table with its crossing-cell columns
/// explicitly checked by WHIR.
pub fn cksum_air(
    root: stark::config::Commitment,
    faddr: Vec<u64>,
    options: &ProofOptions,
    min_rows: usize,
) -> ExtraAir {
    let columns = cksum::preprocessed_columns(&faddr, min_rows);
    let generator = Arc::new(move || columns.clone());
    let constraints = CksumConstraints;
    let air = AirWithBuses::<F, E, NullBoundaryConstraintBuilder, FvmPublicInputs, CksumConstraints>::new(
        cksum::cols::NUM_COLUMNS,
        AuxiliaryTraceBuildData { interactions: cksum::bus_interactions() },
        options,
        1,
        constraints,
    )
    .with_name("FIELD_VM_CKSUM")
    .with_preprocessed_columns(root, cksum::cols::NUM_PRECOMPUTED_COLS, generator);
    Box::new(air)
}

fn expected(
    public: &[PublicCell],
    segments: &[FvmPublicInputs],
    z: &FEE,
    alpha: &FEE,
) -> Option<FEE> {
    let pub_balance = prove::expected_public_balance(public, z, alpha)?;
    // Endpoint claims are checked for chain/start/halt before this function;
    // use the first/last stated tuples so any changed claim also changes the
    // balance closed by LogUp.
    fn fp(bus: BusId, words: &[u64], z: &FEE, alpha: &FEE) -> FEE {
        let mut power = *alpha;
        let mut token = FEE::from(bus as u64);
        for &word in words {
            token += FEE::from(word) * power;
            power *= alpha;
        }
        *z - token
    }
    let first = &segments.first()?.first;
    let last = &segments.last()?.last;
    let mut inverses = [
        fp(BusId::FieldVmState, first, z, alpha),
        fp(BusId::FieldVmState, last, z, alpha),
    ];
    FieldElement::inplace_batch_inverse(&mut inverses).ok()?;
    Some(pub_balance + inverses[0] - inverses[1])
}

fn well_formed(segments: &[FvmPublicInputs], heights: &[u8]) -> bool {
    if segments.is_empty() || segments.len() > heights.len() {
        return false;
    }
    segments.iter().enumerate().all(|(i, s)| {
        let h = 1usize.checked_shl(heights[i] as u32);
        s.last_row.checked_add(1) == h
            && s.first.len() == air::STATE_WORDS
            && s.last.len() == air::STATE_WORDS
    }) && segments.windows(2).all(|p| p[0].last == p[1].first)
        && segments
            .first()
            .is_some_and(|s| s.first == air::start_words())
        && segments.last().is_some_and(|s| s.last == air::halt_words())
}

/// Prove traces with WHIR. `program_id` is the same DECODE root list used by
/// the STARK prover; `extra_traces` follows `extra` after MEM.
pub fn prove_whir(
    program_id: &[stark::config::Commitment],
    program: &Program,
    public: &[PublicCell],
    traces: &mut FvmTraces,
    extra: Vec<ExtraAir>,
    extra_traces: &mut [TraceTable<F, E>],
    options: &ProofOptions,
) -> Result<FvmWhirProof, String> {
    if extra.len() != extra_traces.len() {
        return Err("one trace per extra AIR".into());
    }
    if program_id.len() != decode::chunks(program, prove::MIN_ROWS).len() {
        return Err("one program root per decode chunk".into());
    }
    if !prove::public_well_formed(public) {
        return Err("malformed public cells".into());
    }
    let airs = WhirAirs::new(
        program,
        program_id,
        traces.fvm.len(),
        traces.mem.num_rows(),
        options,
        extra,
    );
    let refs = airs.refs();
    let n = traces.fvm.len();
    let d = traces.decode.len();
    let mut pairs: Vec<(&WhirAir, &mut TraceTable<F, E>)> = Vec::new();
    for (a, t) in refs[..n].iter().zip(traces.fvm.iter_mut()) {
        pairs.push((*a, t));
    }
    for (a, t) in refs[n..n + d].iter().zip(traces.decode.iter_mut()) {
        pairs.push((*a, t));
    }
    pairs.push((refs[n + d], &mut traces.mem));
    for (a, t) in refs[n + d + 1..].iter().zip(extra_traces.iter_mut()) {
        pairs.push((*a, t));
    }
    let shapes: Vec<(usize, usize)> = pairs
        .iter()
        .map(|(a, t)| (a.trace_layout().0, t.num_rows().trailing_zeros() as usize))
        .collect();
    let config = crate::multilinear_prove::chain_config(&shapes);
    let mut committed = Vec::with_capacity(pairs.len());
    for ((air, trace), &(width, vars)) in pairs.iter_mut().zip(&shapes) {
        let layout = layout_of(*air, width, vars)?;
        let mut columns = trace.columns_main();
        for (i, expected) in air.precomputed_columns().iter().enumerate() {
            if columns.get(i) != Some(expected) {
                return Err(format!("{}: preprocessed column {i} mismatch", air.name()));
            }
        }
        committed.push(
            CommittedTable::from_layout(layout, |c| core::mem::take(&mut columns[c as usize]))
                .map_err(|e| format!("{e:?}"))?,
        );
    }
    let table_num_vars: Vec<u8> = shapes.iter().map(|s| s.1 as u8).collect();
    let proof = with_block_hash!(|H| {
        let mut transcript = DefaultTranscript::<
            E,
            <H as multilinear::whir_hash::WhirHash>::Transcript,
        >::new(b"lambda-vm/field-vm/whir/v1\0\0\0");
        // Rebuild statement bytes with the selected transcript hash type.
        absorb_statement(
            &mut transcript,
            program_id,
            public,
            &traces.segments,
            &table_num_vars,
        );
        let committed =
            CommittedTables::<_, _, H>::commit(committed, &config).map_err(|e| format!("{e:?}"))?;
        multilinear_table::multi_prove(&committed, &config, &mut transcript, None)
            .map_err(|e| format!("{e:?}"))
    })?;
    Ok(FvmWhirProof {
        proof,
        table_num_vars,
        segments: traces.segments.clone(),
    })
}

/// Prove in the same grouped, prepared-opening format used by block WHIR.
/// Groups are derived from the block's production stack cap and polynomial
/// budget; tables are committed by group and each group uses the configured
/// batched argue.
pub fn prove_whir_block(
    program_id: &[stark::config::Commitment],
    program: &Program,
    public: &[PublicCell],
    traces: &mut FvmTraces,
    extra: Vec<ExtraAir>,
    extra_traces: &mut [TraceTable<F, E>],
    options: &ProofOptions,
) -> Result<FvmWhirBlockProof, String> {
    if extra.len() != extra_traces.len() {
        return Err("one trace per extra AIR".into());
    }
    if program_id.len() != decode::chunks(program, prove::MIN_ROWS).len() {
        return Err("one program root per decode chunk".into());
    }
    if !prove::public_well_formed(public) {
        return Err("malformed public cells".into());
    }
    let airs = WhirAirs::new(
        program,
        program_id,
        traces.fvm.len(),
        traces.mem.num_rows(),
        options,
        extra,
    );
    let refs = airs.refs();
    let n = traces.fvm.len();
    let d = traces.decode.len();
    let mut pairs: Vec<(&WhirAir, &mut TraceTable<F, E>)> = Vec::new();
    for (a, t) in refs[..n].iter().zip(traces.fvm.iter_mut()) {
        pairs.push((*a, t));
    }
    for (a, t) in refs[n..n + d].iter().zip(traces.decode.iter_mut()) {
        pairs.push((*a, t));
    }
    pairs.push((refs[n + d], &mut traces.mem));
    for (a, t) in refs[n + d + 1..].iter().zip(extra_traces.iter_mut()) {
        pairs.push((*a, t));
    }
    let shapes: Vec<(usize, usize)> = pairs
        .iter()
        .map(|(a, t)| (a.trace_layout().0, t.num_rows().trailing_zeros() as usize))
        .collect();
    let format = crate::block_whir::BlockFormat::production();
    let config = format.chain_config(&shapes);
    let mut sizes =
        stark::multilinear_block::block_groups(&shapes, config.format.stack, format.group_polys)
            .map_err(|e| format!("{e:?}"))?;
    // A checksum-bound proof keeps its CKSUM table in a final singleton group,
    // so its challenge is bound to exactly the preceding groups' roots.
    if refs.last().is_some_and(|a| a.name() == "FIELD_VM_CKSUM") && !sizes.is_empty() {
        let last = sizes.pop().unwrap();
        if last > 1 {
            sizes.push(last - 1);
        }
        sizes.push(1);
    }
    if sizes.len() > format.max_groups {
        return Err(format!(
            "{} groups exceed block format limit {}",
            sizes.len(),
            format.max_groups
        ));
    }
    let groups: Vec<Vec<u32>> = sizes
        .iter()
        .scan(0usize, |start, &size| {
            let group = (*start..*start + size).map(|i| i as u32).collect();
            *start += size;
            Some(group)
        })
        .collect();
    let table_num_vars: Vec<u8> = shapes.iter().map(|s| s.1 as u8).collect();
    let mut tables = Vec::with_capacity(pairs.len());
    for ((air, trace), &(width, vars)) in pairs.iter_mut().zip(&shapes) {
        let layout = layout_of(*air, width, vars)?;
        let mut columns = trace.columns_main();
        for (i, expected) in air.precomputed_columns().iter().enumerate() {
            if columns.get(i) != Some(expected) {
                return Err(format!("{}: preprocessed column {i} mismatch", air.name()));
            }
        }
        tables.push(
            CommittedTable::from_layout(layout, |c| core::mem::take(&mut columns[c as usize]))
                .map_err(|e| format!("{e:?}"))?,
        );
    }
    let mut table_iter = tables.into_iter();
    let grouped_tables: Vec<Vec<_>> = sizes
        .iter()
        .map(|&size| table_iter.by_ref().take(size).collect())
        .collect();
    let committed = with_block_hash!(|H| {
        BlockCommitted::commit_groups::<H>(
            grouped_tables,
            &config,
            crate::block_whir::tree_drop_from_env(),
            multilinear_block::Narrowing::Wide,
            false,
        )
        .map_err(|e| format!("{e:?}"))
    })?;

    let prepared_columns = prepared_tables(&refs, n, d);
    let prepared_groups =
        crate::block_whir::group_stacks(prepared_columns, &groups).map_err(|e| format!("{e:?}"))?;
    let prepared = with_block_hash!(|H| {
        crate::block_whir::commit_groups::<H>(prepared_groups, &config)
            .map_err(|e| format!("{e:?}"))
    })?;
    let borrowed: Vec<Vec<&Mle<F>>> = prepared
        .iter()
        .map(|p| multilinear::stacking::borrow(&p.columns))
        .collect();
    let openings: Vec<multilinear_block::BlockPrepared<'_, F, _>> = prepared
        .iter()
        .zip(&borrowed)
        .map(|(p, columns)| multilinear_block::BlockPrepared {
            group: p.group,
            tables: p.tables.clone(),
            commitment: &p.commitment,
            columns,
        })
        .collect();
    let mut transcript =
        with_block_hash!(|H| { <H as multilinear::whir_hash::WhirHash>::sponge() });
    absorb_block_statement(
        &mut transcript,
        program_id,
        public,
        &traces.segments,
        &table_num_vars,
        &groups,
        &config,
    );
    let (proof, argues, prepared_openings, _) = with_block_hash!(|H| {
        multilinear_block::block_prove_on_forks::<F, E, _, H>(
            committed,
            &config,
            &mut transcript,
            &openings,
            &[],
            &|g| g,
            &ArgueDeviation::default(),
        )
        .map_err(|e| format!("{e:?}"))
    })?;
    Ok(FvmWhirBlockProof {
        proof,
        table_num_vars,
        segments: traces.segments.clone(),
        groups,
        prepared: prepared_openings,
        argues,
    })
}

/// Prove a checksum-bound block-format proof. The preliminary proof supplies
/// the ordered non-CKSUM roots for the binding challenge; the final proof puts
/// CKSUM in its own trailing group so those roots remain an exact prefix.
#[allow(clippy::too_many_arguments)]
pub fn prove_whir_block_bound(
    program_id: &[stark::config::Commitment],
    program: &Program,
    public: &[PublicCell],
    traces: &mut FvmTraces,
    c_bin: &[u64; 4],
    faddr: &[u64],
    values: &[FEE],
    options: &ProofOptions,
) -> Result<(FvmWhirBlockProof, FEE), String> {
    if !prove::public_well_formed(public)
        || faddr.is_empty()
        || faddr.len() != values.len()
        || public.iter().any(|(addr, _)| *addr >= cksum::CKSUM_ADDR)
    {
        return Err("malformed public cells or checksum inputs".into());
    }
    let preview_roots = preview_whir_block_roots(program_id, program, traces, options)?;
    let alpha = cksum::binding_alpha(c_bin, &cksum::field_roots_digest(&preview_roots));
    let (trace, checksum) = cksum::generate_trace(faddr, values, alpha, 8);
    let root = cksum::commitment(faddr, options, 8);
    let air = cksum_air(root, faddr.to_vec(), options, 8);
    let mut bound_public = public.to_vec();
    bound_public.push((cksum::CKSUM_ADDR, checksum));
    bound_public.push((cksum::CKSUM_ALPHA_ADDR, alpha));
    bound_public.sort_by_key(|cell| cell.0);
    let proof = prove_whir_block(
        program_id,
        program,
        &bound_public,
        traces,
        vec![air],
        &mut [trace],
        options,
    )?;
    let vars = *proof.table_num_vars.last().ok_or("missing CKSUM table")? as usize;
    let format = crate::block_whir::BlockFormat::production();
    let mut shapes: Vec<_> = proof
        .table_num_vars
        .iter()
        .take(proof.table_num_vars.len().saturating_sub(1))
        .enumerate()
        .map(|(i, &v)| {
            let width = if i < proof.segments.len() {
                air::cols::NUM_COLUMNS
            } else if i < proof.segments.len() + decode::chunks(program, prove::MIN_ROWS).len() {
                decode::NUM_COLUMNS
            } else {
                mem::cols::NUM_COLUMNS
            };
            (width, v as usize)
        })
        .collect();
    shapes.push((cksum::cols::NUM_COLUMNS, vars));
    let config = format.chain_config(&shapes);
    let checksum_polys =
        multilinear_table::global_layout(&[(cksum::cols::NUM_COLUMNS, vars)], config.format.stack)
            .map_err(|e| format!("{e:?}"))?
            .num_polys();
    let prefix_len = proof
        .proof
        .roots
        .len()
        .checked_sub(checksum_polys)
        .ok_or("missing checksum roots")?;
    if proof.proof.roots[..prefix_len] != preview_roots[..] {
        return Err("non-CKSUM group roots changed while adding CKSUM".into());
    }
    Ok((proof, alpha))
}

/// Phase A only, for the checksum binding preview. It deliberately shares the
/// same shapes, grouping and committed-table constructors as the final proof.
fn preview_whir_block_roots(
    program_id: &[stark::config::Commitment],
    program: &Program,
    traces: &mut FvmTraces,
    options: &ProofOptions,
) -> Result<Vec<multilinear::whir_commit::Commitment>, String> {
    let airs = WhirAirs::new(
        program,
        program_id,
        traces.fvm.len(),
        traces.mem.num_rows(),
        options,
        Vec::new(),
    );
    let refs = airs.refs();
    let n = traces.fvm.len();
    let d = traces.decode.len();
    let mut pairs: Vec<(&WhirAir, &mut TraceTable<F, E>)> = Vec::new();
    for (a, t) in refs[..n].iter().zip(traces.fvm.iter_mut()) {
        pairs.push((*a, t));
    }
    for (a, t) in refs[n..n + d].iter().zip(traces.decode.iter_mut()) {
        pairs.push((*a, t));
    }
    pairs.push((refs[n + d], &mut traces.mem));
    let shapes: Vec<_> = pairs
        .iter()
        .map(|(a, t)| (a.trace_layout().0, t.num_rows().trailing_zeros() as usize))
        .collect();
    let format = crate::block_whir::BlockFormat::production();
    let config = format.chain_config(&shapes);
    let sizes =
        stark::multilinear_block::block_groups(&shapes, config.format.stack, format.group_polys)
            .map_err(|e| format!("{e:?}"))?;
    let mut tables = Vec::with_capacity(pairs.len());
    for ((air, trace), &(width, vars)) in pairs.iter_mut().zip(&shapes) {
        let layout = layout_of(*air, width, vars)?;
        let mut columns = trace.columns_main();
        for (i, expected) in air.precomputed_columns().iter().enumerate() {
            if columns.get(i) != Some(expected) {
                return Err(format!("{}: preprocessed column {i} mismatch", air.name()));
            }
        }
        tables.push(
            CommittedTable::from_layout(layout, |c| core::mem::take(&mut columns[c as usize]))
                .map_err(|e| format!("{e:?}"))?,
        );
    }
    let mut iter = tables.into_iter();
    let grouped = sizes
        .iter()
        .map(|&size| iter.by_ref().take(size).collect())
        .collect::<Vec<Vec<_>>>();
    with_block_hash!(|H| {
        let committed = BlockCommitted::commit_groups::<H>(
            grouped,
            &config,
            crate::block_whir::tree_drop_from_env(),
            multilinear_block::Narrowing::Wide,
            false,
        )
        .map_err(|e| format!("{e:?}"))?;
        Ok(committed.roots().to_vec())
    })
}

/// WHIR proof with the crossing-cell checksum bound to the complete ordered
/// root list of the non-CKSUM group. The preview and final non-CKSUM commits
/// use the same shapes, chain config, and single-group layout.
#[allow(clippy::too_many_arguments)]
pub fn prove_whir_bound(
    program_id: &[stark::config::Commitment],
    program: &Program,
    public: &[PublicCell],
    traces: &mut FvmTraces,
    c_bin: &[u64; 4],
    faddr: &[u64],
    values: &[FEE],
    options: &ProofOptions,
) -> Result<(FvmWhirProof, FEE), String> {
    if !prove::public_well_formed(public) || faddr.is_empty() || faddr.len() != values.len() {
        return Err("malformed public cells or checksum inputs".into());
    }
    if program_id.len() != decode::chunks(program, prove::MIN_ROWS).len() {
        return Err("one program root per decode chunk".into());
    }
    let non_airs = WhirAirs::new(
        program,
        program_id,
        traces.fvm.len(),
        traces.mem.num_rows(),
        options,
        Vec::new(),
    );
    let non_refs = non_airs.refs();
    let mut non_pairs: Vec<(&WhirAir, &mut TraceTable<F, E>)> = Vec::new();
    let n = traces.fvm.len();
    let d = traces.decode.len();
    for (a, t) in non_refs[..n].iter().zip(traces.fvm.iter_mut()) {
        non_pairs.push((*a, t));
    }
    for (a, t) in non_refs[n..n + d].iter().zip(traces.decode.iter_mut()) {
        non_pairs.push((*a, t));
    }
    non_pairs.push((non_refs[n + d], &mut traces.mem));
    let non_shapes: Vec<(usize, usize)> = non_pairs
        .iter()
        .map(|(a, t)| (a.trace_layout().0, t.num_rows().trailing_zeros() as usize))
        .collect();
    let cksum_rows = faddr.len().next_power_of_two().max(8);
    let cksum_vars = cksum_rows.trailing_zeros() as usize;
    let cksum_root = cksum::commitment(faddr, options, 8);
    let all_shapes = non_shapes
        .iter()
        .copied()
        .chain([(cksum::cols::NUM_COLUMNS, cksum_vars)])
        .collect::<Vec<_>>();
    let config = crate::multilinear_prove::chain_config(&all_shapes);

    // Build the non-CKSUM tables once, then use identical table descriptions
    // for preview and final commits. `columns_main` owns a transpose copy, so
    // the execution traces remain available for the real proof.
    let preview_pairs: Vec<(&WhirAir, &TraceTable<F, E>)> = non_pairs
        .iter()
        .map(|(air, trace)| (*air, &**trace))
        .collect();
    let preview_roots = with_block_hash!(|H| {
        let preview_tables = committed_tables(&preview_pairs, &non_shapes)?;
        CommittedTables::<_, _, H>::commit_grouped(preview_tables, &[non_shapes.len()], &config)
            .map(|c| c.roots().to_vec())
            .map_err(|e| format!("{e:?}"))
    })?;
    drop(preview_pairs);
    let c_fld = cksum::field_roots_digest(&preview_roots);
    let alpha = cksum::binding_alpha(c_bin, &c_fld);
    let (mut cksum_trace, s) = cksum::generate_trace(faddr, values, alpha, 8);
    let cksum_air = cksum_air(cksum_root, faddr.to_vec(), options, 8);
    let mut bound_public = public.to_vec();
    if bound_public
        .iter()
        .any(|(addr, _)| *addr >= cksum::CKSUM_ADDR)
    {
        return Err("checksum address is already public".into());
    }
    bound_public.push((cksum::CKSUM_ADDR, s));
    bound_public.push((cksum::CKSUM_ALPHA_ADDR, alpha));
    bound_public.sort_by_key(|c| c.0);

    let mut all_pairs = non_pairs;
    all_pairs.push((&*cksum_air, &mut cksum_trace));
    let all_ref_pairs: Vec<(&WhirAir, &TraceTable<F, E>)> = all_pairs
        .iter()
        .map(|(air, trace)| (*air, &**trace))
        .collect();
    let committed = committed_tables(&all_ref_pairs, &all_shapes)?;
    let table_num_vars: Vec<u8> = all_shapes.iter().map(|s| s.1 as u8).collect();
    let proof = with_block_hash!(|H| {
        let mut transcript = DefaultTranscript::<
            E,
            <H as multilinear::whir_hash::WhirHash>::Transcript,
        >::new(b"lambda-vm/field-vm/whir/v1\0\0\0");
        absorb_statement(
            &mut transcript,
            program_id,
            &bound_public,
            &traces.segments,
            &table_num_vars,
        );
        let committed =
            CommittedTables::<_, _, H>::commit_grouped(committed, &[non_shapes.len(), 1], &config)
                .map_err(|e| format!("{e:?}"))?;
        assert_eq!(
            &committed.roots()[..preview_roots.len()],
            preview_roots.as_slice(),
            "non-CKSUM roots changed between preview and real commit"
        );
        multilinear_table::multi_prove(&committed, &config, &mut transcript, None)
            .map_err(|e| format!("{e:?}"))
    })?;
    Ok((
        FvmWhirProof {
            proof,
            table_num_vars,
            segments: traces.segments.clone(),
        },
        alpha,
    ))
}

/// The statement's `append_bytes` calls, in order: what a recursion leaf
/// replays call by call (the sponge binds each call's length).
fn statement_calls(
    roots: &[stark::config::Commitment],
    public: &[PublicCell],
    segments: &[FvmPublicInputs],
    table_num_vars: &[u8],
    big_endian: bool,
) -> Vec<Vec<u8>> {
    let mut calls: Vec<Vec<u8>> = Vec::new();
    calls.push((roots.len() as u64).to_le_bytes().to_vec());
    for r in roots {
        calls.push(r.to_vec());
    }
    calls.push((public.len() as u64).to_le_bytes().to_vec());
    for (addr, value) in public {
        calls.push(addr.to_le_bytes().to_vec());
        for c in value.value() {
            let c = c.canonical();
            calls.push(
                if big_endian {
                    c.to_be_bytes()
                } else {
                    c.to_le_bytes()
                }
                .to_vec(),
            );
        }
    }
    calls.push((segments.len() as u64).to_le_bytes().to_vec());
    for pi in segments {
        calls.push((pi.last_row as u64).to_le_bytes().to_vec());
        for w in pi.first.iter().chain(&pi.last) {
            calls.push(w.to_le_bytes().to_vec());
        }
    }
    calls.push((table_num_vars.len() as u64).to_le_bytes().to_vec());
    calls.push(table_num_vars.to_vec());
    calls
}

fn absorb_statement(
    t: &mut impl IsTranscript<E>,
    roots: &[stark::config::Commitment],
    public: &[PublicCell],
    segments: &[FvmPublicInputs],
    table_num_vars: &[u8],
) {
    for call in statement_calls(roots, public, segments, table_num_vars, false) {
        t.append_bytes(&call);
    }
}

/// [`statement_calls`] of the block format: its tag, then the groups and the
/// WHIR parameters.
pub(crate) fn block_statement_calls(
    roots: &[stark::config::Commitment],
    public: &[PublicCell],
    segments: &[FvmPublicInputs],
    table_num_vars: &[u8],
    groups: &[Vec<u32>],
    config: &multilinear::whir_chain::ChainConfig,
) -> Vec<Vec<u8>> {
    let mut calls = vec![b"lambda-vm/field-vm/whir-block/v2\0".to_vec()];
    calls.extend(statement_calls(
        roots,
        public,
        segments,
        table_num_vars,
        true,
    ));
    calls.push((groups.len() as u64).to_le_bytes().to_vec());
    for group in groups {
        calls.push((group.len() as u64).to_le_bytes().to_vec());
        for index in group {
            calls.push(u64::from(*index).to_le_bytes().to_vec());
        }
    }
    for value in [
        config.log_blowup as u64,
        config.fold_word(),
        config.num_queries as u64,
    ] {
        calls.push(value.to_le_bytes().to_vec());
    }
    for bits in [config.grind.folding, config.grind.ood, config.grind.query] {
        calls.push(vec![bits]);
    }
    calls
}

/// The index among [`block_statement_calls`] of coordinate `coord` of public
/// cell `cell`, under `roots` program-id roots.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn block_statement_value_call(roots: usize, cell: usize, coord: usize) -> usize {
    3 + roots + 4 * cell + 1 + coord
}

fn absorb_block_statement(
    t: &mut impl IsTranscript<E>,
    roots: &[stark::config::Commitment],
    public: &[PublicCell],
    segments: &[FvmPublicInputs],
    table_num_vars: &[u8],
    groups: &[Vec<u32>],
    config: &multilinear::whir_chain::ChainConfig,
) {
    for call in block_statement_calls(roots, public, segments, table_num_vars, groups, config) {
        t.append_bytes(&call);
    }
}

/// Verify a grouped Field VM WHIR proof using the production block format.
pub fn verify_whir_block(
    program_id: &[stark::config::Commitment],
    program: &Program,
    public: &[PublicCell],
    proof: &FvmWhirBlockProof,
    extra: Vec<ExtraAir>,
    options: &ProofOptions,
) -> bool {
    if !prove::public_well_formed(public) || !well_formed(&proof.segments, &proof.table_num_vars) {
        return false;
    }
    let decode_count = decode::chunks(program, prove::MIN_ROWS).len();
    if program_id.len() != decode_count {
        return false;
    }
    let mem_index = proof.segments.len() + decode_count;
    let Some(&mem_vars) = proof.table_num_vars.get(mem_index) else {
        return false;
    };
    let Some(mem_rows) = 1usize.checked_shl(mem_vars as u32) else {
        return false;
    };
    let airs = WhirAirs::new(
        program,
        program_id,
        proof.segments.len(),
        mem_rows,
        options,
        extra,
    );
    let refs = airs.refs();
    if refs.len() != proof.table_num_vars.len() {
        return false;
    }
    let shapes: Vec<_> = refs
        .iter()
        .zip(&proof.table_num_vars)
        .map(|(a, &v)| (a.trace_layout().0, v as usize))
        .collect();
    let format = crate::block_whir::BlockFormat::production();
    let config = format.chain_config(&shapes);
    let Ok(order) = crate::block_whir::validate_groups(&proof.groups, shapes.len()) else {
        return false;
    };
    let sizes: Vec<_> = proof.groups.iter().map(Vec::len).collect();
    if sizes.len() > format.max_groups || proof.groups.iter().any(Vec::is_empty) {
        return false;
    }
    let ordered_shapes: Vec<_> = order.iter().map(|&i| shapes[i]).collect();
    let Ok((stack_layouts, domains)) =
        crate::multilinear_prove::stacks(&ordered_shapes, &sizes, &config)
    else {
        return false;
    };
    if stack_layouts
        .iter()
        .zip(&sizes)
        .any(|(layout, &size)| layout.num_polys() > format.group_polys && size > 1)
    {
        return false;
    }
    let layouts: Vec<TableLayout<'_, F, E>> = match refs
        .iter()
        .zip(&shapes)
        .map(|(a, &(w, v))| layout_of(*a, w, v))
        .collect()
    {
        Ok(v) => v,
        Err(_) => return false,
    };
    let pre: Vec<Vec<Mle<F>>> = match refs.iter().map(|a| preprocessed(*a)).collect() {
        Ok(v) => v,
        Err(_) => return false,
    };
    let statements: Vec<TableStatement<'_, F, E>> = order
        .iter()
        .map(|&i| layouts[i].statement_with_preprocessed(&pre[i]))
        .collect();
    let prepared_columns = prepared_tables(&refs, proof.segments.len(), decode_count);
    let Ok(prepared_stacks) = crate::block_whir::group_stacks(prepared_columns, &proof.groups)
    else {
        return false;
    };
    let Ok(prepared): Result<Vec<crate::block_whir::GroupPrepared<_>>, _> =
        with_block_hash!(|H| { crate::block_whir::commit_groups::<H>(prepared_stacks, &config) })
    else {
        return false;
    };
    if proof.prepared.len() != prepared.len() {
        return false;
    }
    let checks: Vec<_> = prepared
        .iter()
        .map(|p| multilinear_block::BlockPreparedCheck {
            group: p.group,
            tables: p.tables.clone(),
            roots: &p.roots,
            layout: p.commitment.layout(),
            domain: p.commitment.domain(),
        })
        .collect();
    let derived: Vec<_> = checks
        .iter()
        .flat_map(|p| p.roots.iter().copied())
        .collect();
    with_block_hash!(|H| {
        let mut transcript = <H as multilinear::whir_hash::WhirHash>::sponge();
        absorb_block_statement(
            &mut transcript,
            program_id,
            public,
            &proof.segments,
            &proof.table_num_vars,
            &proof.groups,
            &config,
        );
        let mut probe = transcript.clone();
        multilinear_table::absorb_roots::<E, _>(&mut probe, &proof.proof.roots, &derived);
        let z: FEE = probe.sample_field_element();
        let alpha: FEE = probe.sample_field_element();
        let Some(owed) = expected(public, &proof.segments, &z, &alpha) else {
            return false;
        };
        multilinear_block::block_verify_with::<F, E, _, H>(
            &proof.proof,
            &proof.argues,
            &proof.prepared,
            &checks,
            &statements,
            &stack_layouts,
            &domains,
            &sizes,
            &owed,
            &config,
            &mut transcript,
            false,
            stark::multilinear_table::batched::VerifierChecks::ALL,
        )
        .is_ok()
    })
}

/// Verify the checksum-bound block proof, recomputing its binding challenge
/// from every root before the trailing CKSUM singleton group.
#[allow(clippy::too_many_arguments)]
pub fn verify_whir_block_bound(
    c_bin: &[u64; 4],
    s: &FEE,
    program_id: &[stark::config::Commitment],
    program: &Program,
    public: &[PublicCell],
    faddr: &[u64],
    proof: &FvmWhirBlockProof,
    options: &ProofOptions,
) -> bool {
    let decode_count = decode::chunks(program, prove::MIN_ROWS).len();
    if faddr.is_empty()
        || proof.groups.last().map(Vec::as_slice)
            != Some(&[proof.table_num_vars.len().saturating_sub(1) as u32][..])
        || program_id.len() != decode_count
        || proof.table_num_vars.len() != proof.segments.len() + decode_count + 2
        || !well_formed(&proof.segments, &proof.table_num_vars)
        || public.iter().any(|(addr, _)| *addr >= cksum::CKSUM_ADDR)
    {
        return false;
    }
    let root = cksum::commitment(faddr, options, 8);
    let dummy = cksum_air(root, faddr.to_vec(), options, 8);
    let airs = WhirAirs::new(
        program,
        program_id,
        proof.segments.len(),
        1usize
            .checked_shl(
                *proof
                    .table_num_vars
                    .get(proof.segments.len() + decode_count)
                    .unwrap_or(&0) as u32,
            )
            .unwrap_or(8),
        options,
        vec![dummy],
    );
    let refs = airs.refs();
    let shapes: Vec<_> = refs
        .iter()
        .zip(&proof.table_num_vars)
        .map(|(a, &v)| (a.trace_layout().0, v as usize))
        .collect();
    let format = crate::block_whir::BlockFormat::production();
    let config = format.chain_config(&shapes);
    let Some(&vars) = proof.table_num_vars.last() else {
        return false;
    };
    let Ok(layout) = multilinear_table::global_layout(
        &[(cksum::cols::NUM_COLUMNS, vars as usize)],
        config.format.stack,
    ) else {
        return false;
    };
    let Some(prefix_len) = proof.proof.roots.len().checked_sub(layout.num_polys()) else {
        return false;
    };
    let alpha = cksum::binding_alpha(
        c_bin,
        &cksum::field_roots_digest(&proof.proof.roots[..prefix_len]),
    );
    let air = cksum_air(root, faddr.to_vec(), options, 8);
    let mut bound_public = public.to_vec();
    bound_public.push((cksum::CKSUM_ADDR, *s));
    bound_public.push((cksum::CKSUM_ALPHA_ADDR, alpha));
    bound_public.sort_by_key(|cell| cell.0);
    verify_whir_block(
        program_id,
        program,
        &bound_public,
        proof,
        vec![air],
        options,
    )
}

/// Verify a WHIR proof against the program, public cells and supplied extras.
pub fn verify_whir(
    program_id: &[stark::config::Commitment],
    program: &Program,
    public: &[PublicCell],
    proof: &FvmWhirProof,
    extra: Vec<ExtraAir>,
    options: &ProofOptions,
) -> bool {
    if !prove::public_well_formed(public) || !well_formed(&proof.segments, &proof.table_num_vars) {
        return false;
    }
    let decode_count = decode::chunks(program, prove::MIN_ROWS).len();
    if program_id.len() != decode_count {
        return false;
    }
    let mem_vars_index = proof.segments.len() + decode_count;
    let Some(&mem_vars) = proof.table_num_vars.get(mem_vars_index) else {
        return false;
    };
    let Some(mem_rows) = 1usize.checked_shl(mem_vars as u32) else {
        return false;
    };
    let airs = WhirAirs::new(
        program,
        program_id,
        proof.segments.len(),
        mem_rows,
        options,
        extra,
    );
    let refs = airs.refs();
    if refs.len() != proof.table_num_vars.len() || refs.len() != proof.proof.tables.len() {
        return false;
    }
    let shapes: Vec<(usize, usize)> = refs
        .iter()
        .zip(&proof.table_num_vars)
        .map(|(a, &v)| (a.trace_layout().0, v as usize))
        .collect();
    let config = crate::multilinear_prove::chain_config(&shapes);
    let layouts: Vec<TableLayout<'_, F, E>> = match refs
        .iter()
        .zip(&shapes)
        .map(|(a, &(w, n))| layout_of(*a, w, n))
        .collect()
    {
        Ok(v) => v,
        Err(_) => return false,
    };
    let pre: Vec<Vec<Mle<F>>> = match refs.iter().map(|a| preprocessed(*a)).collect() {
        Ok(v) => v,
        Err(_) => return false,
    };
    let statements: Vec<TableStatement<'_, F, E>> = layouts
        .iter()
        .zip(&pre)
        .map(|(l, p)| l.statement_with_preprocessed(p))
        .collect();
    let sizes = [shapes.len()];
    let Ok((stacked, domains)) = crate::multilinear_prove::stacks(&shapes, &sizes, &config) else {
        return false;
    };
    with_block_hash!(|H| {
        let mut transcript = DefaultTranscript::<
            E,
            <H as multilinear::whir_hash::WhirHash>::Transcript,
        >::new(b"lambda-vm/field-vm/whir/v1\0\0\0");
        absorb_statement(
            &mut transcript,
            program_id,
            public,
            &proof.segments,
            &proof.table_num_vars,
        );
        let mut probe = transcript.clone();
        multilinear_table::absorb_roots::<E, _>(&mut probe, &proof.proof.roots, &[]);
        let z: FEE = probe.sample_field_element();
        let alpha: FEE = probe.sample_field_element();
        let Some(owed) = expected(public, &proof.segments, &z, &alpha) else {
            return false;
        };
        multilinear_table::multi_verify::<_, _, _, H>(
            &proof.proof,
            &statements,
            &stacked,
            &domains,
            &sizes,
            &owed,
            &config,
            &mut transcript,
            None,
        )
        .is_ok()
    })
}

/// Verify a checksum-bound WHIR proof. `s` is the checksum value expected by
/// the guest; the verifier also closes that same value at `CKSUM_ADDR` on the
/// public bus. The checksum AIR challenge is recomputed from the ordered roots
/// of the proof's non-CKSUM group.
#[allow(clippy::too_many_arguments)]
pub fn verify_whir_bound(
    c_bin: &[u64; 4],
    s: &FEE,
    program_id: &[stark::config::Commitment],
    program: &Program,
    public: &[PublicCell],
    faddr: &[u64],
    proof: &FvmWhirProof,
    options: &ProofOptions,
) -> bool {
    if !prove::public_well_formed(public)
        || !well_formed(&proof.segments, &proof.table_num_vars)
        || faddr.is_empty()
    {
        return false;
    }
    let decode_count = decode::chunks(program, prove::MIN_ROWS).len();
    if program_id.len() != decode_count || proof.proof.columns.len() != 2 {
        return false;
    }
    let non_root_count = proof.proof.columns[0].polys.len();
    let checksum_root_count = proof.proof.columns[1].polys.len();
    if non_root_count == 0 || proof.proof.roots.len() != non_root_count + checksum_root_count {
        return false;
    }
    let mem_vars_index = proof.segments.len() + decode_count;
    let Some(&mem_vars) = proof.table_num_vars.get(mem_vars_index) else {
        return false;
    };
    let Some(mem_rows) = 1usize.checked_shl(mem_vars as u32) else {
        return false;
    };
    let c_fld = cksum::field_roots_digest(&proof.proof.roots[..non_root_count]);
    let alpha = cksum::binding_alpha(c_bin, &c_fld);
    let non_airs = WhirAirs::new(
        program,
        program_id,
        proof.segments.len(),
        mem_rows,
        options,
        Vec::new(),
    );
    let root = cksum::commitment(faddr, options, 8);
    let checksum = cksum_air(root, faddr.to_vec(), options, 8);
    let airs = non_airs;
    let refs = airs.refs();
    let mut owned_refs = refs;
    owned_refs.push(checksum.as_ref());
    if owned_refs.len() != proof.table_num_vars.len()
        || owned_refs.len() != proof.proof.tables.len()
    {
        return false;
    }
    let shapes: Vec<(usize, usize)> = owned_refs
        .iter()
        .zip(&proof.table_num_vars)
        .map(|(a, &v)| (a.trace_layout().0, v as usize))
        .collect();
    let config = crate::multilinear_prove::chain_config(&shapes);
    let layouts: Vec<TableLayout<'_, F, E>> = match owned_refs
        .iter()
        .zip(&shapes)
        .map(|(a, &(w, n))| layout_of(*a, w, n))
        .collect()
    {
        Ok(v) => v,
        Err(_) => return false,
    };
    let pre: Vec<Vec<Mle<F>>> = match owned_refs.iter().map(|a| preprocessed(*a)).collect() {
        Ok(v) => v,
        Err(_) => return false,
    };
    let statements: Vec<TableStatement<'_, F, E>> = layouts
        .iter()
        .zip(&pre)
        .map(|(l, p)| l.statement_with_preprocessed(p))
        .collect();
    let sizes = [shapes.len() - 1, 1];
    let Ok((stacked, domains)) = crate::multilinear_prove::stacks(&shapes, &sizes, &config) else {
        return false;
    };
    let mut bound_public = public.to_vec();
    if bound_public
        .iter()
        .any(|(addr, _)| *addr >= cksum::CKSUM_ADDR)
    {
        return false;
    }
    bound_public.push((cksum::CKSUM_ADDR, *s));
    bound_public.push((cksum::CKSUM_ALPHA_ADDR, alpha));
    bound_public.sort_by_key(|c| c.0);
    with_block_hash!(|H| {
        let mut transcript = DefaultTranscript::<
            E,
            <H as multilinear::whir_hash::WhirHash>::Transcript,
        >::new(b"lambda-vm/field-vm/whir/v1\0\0\0");
        absorb_statement(
            &mut transcript,
            program_id,
            &bound_public,
            &proof.segments,
            &proof.table_num_vars,
        );
        let mut probe = transcript.clone();
        multilinear_table::absorb_roots::<E, _>(&mut probe, &proof.proof.roots, &[]);
        let z: FEE = probe.sample_field_element();
        let transcript_alpha: FEE = probe.sample_field_element();
        let Some(owed) = expected(&bound_public, &proof.segments, &z, &transcript_alpha) else {
            return false;
        };
        multilinear_table::multi_verify::<_, _, _, H>(
            &proof.proof,
            &statements,
            &stacked,
            &domains,
            &sizes,
            &owed,
            &config,
            &mut transcript,
            None,
        )
        .is_ok()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field_vm::asm::Asm;
    use crate::field_vm::executor::{Memory, NoHints, execute};
    use crate::field_vm::isa::{Arg, gpr};
    use crate::field_vm::prove::{self, default_options};

    fn run() -> (Program, super::super::executor::Execution) {
        let r = gpr(0);
        let mut asm = Asm::new();
        asm.mov(Arg::reg(r), Arg::imm(1));
        for _ in 0..10 {
            asm.add(Arg::reg(r), Arg::reg(r), Arg::imm(1));
        }
        asm.store(Arg::mem_abs(0), Arg::reg(r)).halt();
        let program = asm.finish().unwrap();
        let exec = execute(&program, Memory::new(1), &mut NoHints, 1024).unwrap();
        (program, exec)
    }

    #[test]
    fn field_vm_whir_statement_and_segment_chain() {
        let (program, exec) = run();
        let options = default_options();
        let roots = prove::program_id(&program, &options);
        let public = prove::public_cells(&exec, &[0]);

        let mut one_segment = prove::generate_traces(&program, &exec, &[0]);
        let one_proof = prove_whir(
            &roots,
            &program,
            &public,
            &mut one_segment,
            Vec::new(),
            &mut [],
            &options,
        )
        .unwrap();
        assert!(verify_whir(
            &roots,
            &program,
            &public,
            &one_proof,
            Vec::new(),
            &options
        ));

        let mut traces = prove::generate_traces_with(&program, &exec, &[0], Some(8));
        assert!(traces.fvm.len() > 1);
        let proof = prove_whir(
            &roots,
            &program,
            &public,
            &mut traces,
            Vec::new(),
            &mut [],
            &options,
        )
        .unwrap();
        assert!(verify_whir(
            &roots,
            &program,
            &public,
            &proof,
            Vec::new(),
            &options
        ));

        let mut changed_public = public.clone();
        changed_public[0].1 += FEE::one();
        assert!(!verify_whir(
            &roots,
            &program,
            &changed_public,
            &proof,
            Vec::new(),
            &options
        ));

        let mut changed_state = proof.clone();
        changed_state.segments.last_mut().unwrap().last[0] += 1;
        assert!(!verify_whir(
            &roots,
            &program,
            &public,
            &changed_state,
            Vec::new(),
            &options
        ));
    }

    #[test]
    fn field_vm_whir_block_statement_and_groups() {
        let (program, exec) = run();
        let options = default_options();
        let roots = prove::program_id(&program, &options);
        let public = prove::public_cells(&exec, &[0]);
        let mut traces = prove::generate_traces(&program, &exec, &[0]);
        let proof = prove_whir_block(
            &roots,
            &program,
            &public,
            &mut traces,
            Vec::new(),
            &mut [],
            &options,
        )
        .unwrap();
        assert!(verify_whir_block(
            &roots,
            &program,
            &public,
            &proof,
            Vec::new(),
            &options
        ));
        let mut changed_public = public;
        changed_public[0].1 += FEE::one();
        assert!(!verify_whir_block(
            &roots,
            &program,
            &changed_public,
            &proof,
            Vec::new(),
            &options
        ));
    }

    #[test]
    fn whir_checksum_binding_accepts_and_rejects_statement_changes() {
        let (program, exec) = run();
        let options = default_options();
        let roots = prove::program_id(&program, &options);
        let public_addrs = [0];
        let public = prove::public_cells(&exec, &public_addrs);
        let faddr = [0u64];
        let values = [exec.mem[0]];
        let c_bin = [7, 11, 13, 17];
        let mut traces =
            prove::generate_traces_reading(&program, &exec, &public_addrs, None, &faddr);
        let (proof, alpha) = prove_whir_bound(
            &roots,
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
        assert!(verify_whir_bound(
            &c_bin, &s, &roots, &program, &public, &faddr, &proof, &options
        ));

        assert!(!verify_whir_bound(
            &c_bin,
            &(s + FEE::one()),
            &roots,
            &program,
            &public,
            &faddr,
            &proof,
            &options,
        ));
        let mut changed_c_bin = c_bin;
        changed_c_bin[0] += 1;
        assert!(!verify_whir_bound(
            &changed_c_bin,
            &s,
            &roots,
            &program,
            &public,
            &faddr,
            &proof,
            &options,
        ));
    }
}
