//! ★ The in-row register format's soundness obligations (§6.2 of its design,
//! O1–O9), each as a forgery on the builders' traces, with no proof: a
//! malicious prover's edit of the honest in-row tables, then every inner bus's
//! net (`bus_net_tests`). A forgery is refused when a bus is left over that the
//! prover cannot cancel: a tuple no table holds, or an `IS_HALF` value of
//! 2^16 or more (BITWISE's values are preprocessed, only its multiplicities
//! are the prover's, so an in-range `IS_HALF` left over is a recount the
//! prover would make and is not read here). REGISTER's preprocessed files are
//! set to the verifier's before every net, so no forgery reaches them.
//!
//! Each forgery that a range check or a constraint refuses also runs with that
//! one check taken out of its AIR (the `over` interactions) and then balances
//! every bus: the forgery is whole, and the check is what refuses it. The
//! value-changing keep-alive balances every bus as it is; its constraint
//! refuses it. The proof-level forms of these forgeries are box tests
//! (`tests::regs_inrow_tests`), built from the same edits ([`Forgery`]).

use std::collections::BTreeMap;

use executor::elf::Elf;
use executor::vm::logs::Log;
use stark::lookup::{BusInteraction, Multiplicity};
use stark::trace::TraceTable;

use super::Traces;
use super::bus_net_tests::{Nets, named, nets_in};
use super::finish_tests::{run, windowed};
use crate::tables::MaxRowsConfig;
use crate::tables::cpu::{self, cols as cpu_cols, inrow_cols};
use crate::tables::register::{self, RegisterFormat, inrow};
use crate::tables::register_residual::{self as residual, cols as res};
use crate::tables::types::{BusId, FE, GoldilocksExtension, GoldilocksField};

type Table = TraceTable<GoldilocksField, GoldilocksExtension>;

const TWO_16: u64 = 1 << 16;
/// The field's order: `−k` reads as `P − k`.
const P: u64 = 0xFFFF_FFFF_0000_0001;

/// The in-row build of `program`'s run: windows of 1000 cycles, the small
/// chunks (every chunked table split), every table wide.
pub(crate) fn in_row_build(program: &Elf, logs: &[Log]) -> Traces {
    windowed(program, logs, &MaxRowsConfig::small(), 1000, &|b| {
        b.register_format(RegisterFormat::InRow)
            .expect("before any window")
    })
}

fn get(t: &Table, row: usize, col: usize) -> u64 {
    t.get_main(row, col).canonical_u64()
}

fn set(t: &mut Table, row: usize, col: usize, value: FE) {
    assert!(t.narrow_main().is_none(), "the forgeries edit wide tables");
    t.main_table.set(row, col, value);
}

fn set_u64(t: &mut Table, row: usize, col: usize, value: u64) {
    set(t, row, col, FE::from(value));
}

/// REGISTER's preprocessed columns (the register index, both files, the PC
/// flag) set to the verifier's: the program's initial file and the halted one.
pub(crate) fn with_verifier_files(program: &Elf, traces: &mut Traces) {
    let committed = u32::try_from(traces.public_output_bytes.len()).expect("a u32 count");
    let init = register::register_init_from_entry_point(program.entry_point);
    let fini = register::halted_fini(committed);
    let columns = inrow::preprocessed_columns(&init, &fini);
    let table = &mut traces.register;
    assert_eq!(
        table.num_rows(),
        inrow::NUM_ROWS,
        "REGISTER a register a row"
    );
    for (c, column) in columns.iter().enumerate() {
        for (row, value) in column.iter().enumerate() {
            set(table, row, c, *value);
        }
    }
}

/// BITWISE's `IS_HALF` counts recounted for `traces` as they stand, as a
/// prover of an edited build would recount them: every value under 2^16 that
/// the edits left sent and not counted (or counted and no longer sent) is
/// counted again, so the only `IS_HALF` values the build leaves over are of
/// 2^16 or more. Returns how many values were recounted.
pub(crate) fn recount_is_half(program: &Elf, traces: &mut Traces) -> usize {
    use crate::tables::bitwise;
    let halfword = u64::from(BusId::IsHalfword);
    let in_range = |nets: &Nets| -> Vec<(u64, i128)> {
        nets.left
            .get(&halfword)
            .into_iter()
            .flatten()
            .filter(|(t, _)| t[0] < TWO_16)
            .map(|(t, net)| (t[0], *net))
            .collect()
    };
    let stale = in_range(&nets_in(program, traces, &|_| None, RegisterFormat::InRow));
    let col = bitwise::cols::MU_IS_HALF;
    let table = &mut traces.bitwise;
    #[cfg(feature = "cuda")]
    table.clear_main_rowmajor_dev();
    for &(value, net) in &stale {
        let row = bitwise::row_index((value & 0xFF) as u8, (value >> 8) as u8, 0);
        let recount = |have: u64| {
            u64::try_from(i128::from(have) + net)
                .unwrap_or_else(|_| panic!("IS_HALF[{value}] counts {have}, net {net}"))
        };
        if let Some(packed) = table.narrow_main_mut() {
            let count = recount(packed.get(row, col));
            assert!(
                packed.set(row, col, count),
                "IS_HALF[{value}]: {count} fits its packed column"
            );
        } else {
            let count = recount(table.get_main(row, col).canonical_u64());
            table.main_table.set(row, col, FE::from(count));
        }
    }
    let left = in_range(&nets_in(program, traces, &|_| None, RegisterFormat::InRow));
    assert!(
        left.is_empty(),
        "no IS_HALF value under 2^16 is left after the recount: {left:?}"
    );
    stale.len()
}

// ---- the REG bus's nodes ----------------------------------------------------------------------------------------------

/// A node that consumes a register token on the REG bus.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Node {
    /// CPU table `t`'s row `r`, slot `k` (0: `rs1`, 1: `rs2`, 2: `rd`).
    Slot { t: usize, r: usize, k: usize },
    /// The residual table `t`'s row `r`.
    Residual { t: usize, r: usize },
    /// REGISTER's row `r`: a register's final token.
    Final { r: usize },
}

const SLOT_FLAG: [usize; 3] = [
    cpu_cols::READ_REGISTER1,
    cpu_cols::READ_REGISTER2,
    cpu_cols::WRITE_REGISTER,
];
const SLOT_REG: [usize; 3] = [cpu_cols::RS1, cpu_cols::RS2, cpu_cols::RD];
const SLOT_PREV: [usize; 3] = [
    inrow_cols::PREV_TS_1,
    inrow_cols::PREV_TS_2,
    inrow_cols::PREV_TS_D,
];

/// A CPU slot in use: its node, register, own timestamp and the previous one
/// it reads.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Slot {
    pub(crate) node: Node,
    pub(crate) reg: u64,
    pub(crate) ts: u64,
    pub(crate) prev: u64,
}

/// Every CPU slot in use, table by table, row by row.
pub(crate) fn slots(traces: &Traces) -> Vec<Slot> {
    let heights = crate::cpu_heights_from_traces(traces);
    let mut out = Vec::new();
    for (t, (table, base)) in traces
        .cpus
        .iter()
        .zip(cpu::timestamp_bases(&heights))
        .enumerate()
    {
        for r in 0..table.num_rows() {
            let ts = base + get(table, r, cpu_cols::TIMESTAMP);
            for k in 0..3 {
                if get(table, r, SLOT_FLAG[k]) == 1 {
                    out.push(Slot {
                        node: Node::Slot { t, r, k },
                        reg: get(table, r, SLOT_REG[k]),
                        ts: ts + k as u64,
                        prev: get(table, r, SLOT_PREV[k]),
                    });
                }
            }
        }
    }
    out
}

/// A residual row in use.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ResidualRow {
    pub(crate) node: Node,
    pub(crate) reg: u64,
    pub(crate) ts: u64,
    pub(crate) old_ts: u64,
    pub(crate) keep: bool,
    pub(crate) read: bool,
}

/// Every residual row in use (`μ = 1`).
pub(crate) fn residual_rows(traces: &Traces) -> Vec<ResidualRow> {
    let mut out = Vec::new();
    for (t, table) in traces.memw_registers.iter().enumerate() {
        for r in 0..table.num_rows() {
            let mu = get(table, r, res::MU_READ)
                + get(table, r, res::MU_WRITE)
                + get(table, r, res::MU_KEEP);
            if mu == 0 {
                continue;
            }
            let ts = get(table, r, res::TS_0) + (get(table, r, res::TS_1) << 32);
            let step = 1 + get(table, r, res::D0) + (get(table, r, res::D1) << 16);
            out.push(ResidualRow {
                node: Node::Residual { t, r },
                reg: get(table, r, res::REG),
                ts,
                old_ts: ts - step,
                keep: get(table, r, res::MU_KEEP) == 1,
                read: get(table, r, res::MU_READ) == 1,
            });
        }
    }
    out
}

/// The one node consuming register `reg`'s token at `ts` (not the PC's).
pub(crate) fn consumer_of(traces: &Traces, reg: u64, ts: u64) -> Node {
    let mut found: Vec<Node> = slots(traces)
        .into_iter()
        .filter(|s| s.reg == reg && s.prev == ts)
        .map(|s| s.node)
        .collect();
    found.extend(
        residual_rows(traces)
            .into_iter()
            .filter(|r| r.reg == reg && r.old_ts == ts)
            .map(|r| r.node),
    );
    let table = &traces.register;
    found.extend(
        (0..register::NUM_REGISTERS)
            .filter(|&r| {
                get(table, r, inrow::cols::IS_PC) == 0
                    && get(table, r, inrow::cols::OFFSET) == reg
                    && get(table, r, inrow::cols::TS) == ts
            })
            .map(|r| Node::Final { r }),
    );
    assert_eq!(found.len(), 1, "one consumer of ({reg}, {ts}): {found:?}");
    found[0]
}

/// `node` made to consume its register's token at `prev` instead.
pub(crate) fn repoint(traces: &mut Traces, node: Node, prev: u64) {
    match node {
        Node::Slot { t, r, k } => set_u64(&mut traces.cpus[t], r, SLOT_PREV[k], prev),
        Node::Residual { t, r } => {
            let table = &mut traces.memw_registers[t];
            let ts = get(table, r, res::TS_0) + (get(table, r, res::TS_1) << 32);
            let d = ts - prev - 1;
            assert!(d < 1 << 32, "a residual step fits its limbs");
            set_u64(table, r, res::D0, d & 0xFFFF);
            set_u64(table, r, res::D1, d >> 16);
        }
        Node::Final { r } => set_u64(&mut traces.register, r, inrow::cols::TS, prev),
    }
}

/// The step `node` takes when it consumes at `prev`, when it range-checks one
/// (a CPU slot): within its reach or not.
fn in_slot_reach(traces: &Traces, node: Node, prev: u64) -> bool {
    match node {
        Node::Slot { .. } => {
            let s = slots(traces)
                .into_iter()
                .find(|s| s.node == node)
                .expect("the slot");
            s.ts > prev && s.ts - prev <= TWO_16
        }
        Node::Residual { .. } | Node::Final { .. } => true,
    }
}

// ---- reading a forgery's nets -----------------------------------------------------------------------------------------

/// What the nets of a forged build leave that the prover cannot cancel.
#[derive(Debug, Default)]
pub(crate) struct Left {
    /// Every bus but IS_HALF with a tuple left over, named with its tuples.
    pub(crate) buses: BTreeMap<u64, Vec<(Vec<u64>, i128)>>,
    /// The IS_HALF values of 2^16 or more sent and not received, sorted.
    pub(crate) out_of_range: Vec<u64>,
    /// The Memory bus's register tokens left over.
    pub(crate) memory_registers: Vec<(Vec<u64>, i128)>,
}

impl Left {
    fn of(nets: Nets) -> Self {
        let mut left = Left {
            memory_registers: nets.memory_register_left,
            ..Left::default()
        };
        let halfword = u64::from(BusId::IsHalfword);
        for (bus, tuples) in nets.left {
            if bus == halfword {
                left.out_of_range.extend(
                    tuples
                        .iter()
                        .filter(|(t, net)| t[0] >= TWO_16 && *net > 0)
                        .map(|(t, _)| t[0]),
                );
            } else {
                left.buses.insert(bus, tuples);
            }
        }
        left.out_of_range.sort_unstable();
        left
    }

    /// Whether every bus balances, save what the prover recounts.
    pub(crate) fn accepted(&self) -> bool {
        self.buses.is_empty() && self.out_of_range.is_empty() && self.memory_registers.is_empty()
    }

    fn bus_names(&self) -> Vec<&'static str> {
        self.buses
            .keys()
            .map(|b| BusId::try_from(*b).map_or("?", |b| b.name()))
            .collect()
    }
}

impl std::fmt::Display for Left {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} · IS_HALF out of range {:?} · Memory register tokens {:?}",
            named(&self.buses),
            self.out_of_range,
            self.memory_registers
        )
    }
}

/// The nets of `traces` under the in-row AIRs (their interactions from `over`
/// where it names a table), REGISTER's files the verifier's.
pub(crate) fn left(
    program: &Elf,
    traces: &mut Traces,
    over: &dyn Fn(&str) -> Option<Vec<BusInteraction>>,
) -> Left {
    with_verifier_files(program, traces);
    Left::of(nets_in(program, traces, over, RegisterFormat::InRow))
}

/// The CPU tables' in-row interactions without slot `k`'s `IS_HALF` check: a
/// witness mutant.
type Over = Box<dyn Fn(&str) -> Option<Vec<BusInteraction>>>;

fn cpu_without_slot_range(traces: &Traces, k: usize) -> Over {
    let bases = cpu::timestamp_bases(&crate::cpu_heights_from_traces(traces));
    let halfword = u64::from(BusId::IsHalfword);
    Box::new(move |name: &str| {
        let i: usize = name.strip_prefix("CPU[")?.strip_suffix(']')?.parse().ok()?;
        let all = cpu::bus_interactions_inrow(bases[i]);
        let kept: Vec<BusInteraction> = all
            .iter()
            .filter(|bi| {
                !(bi.bus_id == halfword
                    && bi.is_sender
                    && matches!(bi.multiplicity, Multiplicity::Column(c) if c == SLOT_FLAG[k]))
            })
            .cloned()
            .collect();
        assert_eq!(kept.len() + 1, all.len(), "one IS_HALF check a slot");
        Some(kept)
    })
}

// ---- the forgeries ----------------------------------------------------------------------------------------------------

/// A malicious prover's edit of an honest in-row build.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Forgery {
    /// O1: a CPU slot consumes the token it produces itself (`PREV = ts`),
    /// and the slot's next reader takes the token the slot left.
    SlotSelfLoop,
    /// O1: `rs1` consumes `rs2`'s token of the same register and row
    /// (`PREV = ts + 1`), `rs2` `rs1`'s: a cycle; the next reader takes the
    /// token `rs1` left.
    SlotCycle,
    /// O1: a keep-alive dropped, its slot reading the token from before it
    /// (a step past 2^16).
    SlotPastReach,
    /// O1: a residual read consumes the token it produces itself.
    ResidualSelfLoop,
    /// O2: a keep-alive that changes its register's value, read by a write's
    /// old value.
    KeepAliveChangesValue,
    /// O4: a write's old value one off.
    WrongOldRd,
    /// O4: a keep-alive that consumes and produces a value no one produced.
    KeepAliveInventsValue,
    /// A keep-alive turned into a read no requester asks for, changing the
    /// value.
    ReadWithoutRequester,
}

/// The program each forgery is made on.
pub(crate) fn program_of(forgery: Forgery) -> &'static str {
    match forgery {
        Forgery::SlotSelfLoop | Forgery::WrongOldRd | Forgery::ResidualSelfLoop => {
            "all_instructions_64"
        }
        Forgery::SlotCycle
        | Forgery::SlotPastReach
        | Forgery::KeepAliveChangesValue
        | Forgery::KeepAliveInventsValue
        | Forgery::ReadWithoutRequester => "keep_alive_regs",
    }
}

/// Whether the residual row's old and new values are the same.
fn read_keeps_value(traces: &Traces, node: Node) -> bool {
    let Node::Residual { t, r } = node else {
        return false;
    };
    let table = &traces.memw_registers[t];
    get(table, r, res::OLD_0) == get(table, r, res::VAL_0)
        && get(table, r, res::OLD_1) == get(table, r, res::VAL_1)
}

/// A keep-alive whose token a write's old value reads, and that slot.
fn keep_alive_read_by_a_write(traces: &Traces) -> (ResidualRow, Node) {
    residual_rows(traces)
        .into_iter()
        .filter(|r| r.keep)
        .find_map(|r| {
            let consumer = consumer_of(traces, r.reg, r.ts);
            matches!(consumer, Node::Slot { k: 2, .. }).then_some((r, consumer))
        })
        .expect("a keep-alive read by a write")
}

fn residual_mut(traces: &mut Traces, node: Node) -> (&mut Table, usize) {
    let Node::Residual { t, r } = node else {
        panic!("a residual row")
    };
    (&mut traces.memw_registers[t], r)
}

fn slot_mut(traces: &mut Traces, node: Node) -> (&mut Table, usize, usize) {
    let Node::Slot { t, r, k } = node else {
        panic!("a CPU slot")
    };
    (&mut traces.cpus[t], r, k)
}

/// `forgery` applied to an honest in-row build. Returns what refuses it: the
/// out-of-range `IS_HALF` value it sends, and the witness interactions that
/// take that one check out (none where no range check refuses it).
pub(crate) fn forge(forgery: Forgery, traces: &mut Traces) -> (Option<u64>, Option<Over>) {
    match forgery {
        Forgery::SlotSelfLoop => {
            let (slot, consumer) = slots(traces)
                .into_iter()
                .filter(|s| matches!(s.node, Node::Slot { k: 0, .. }) && s.reg != 255)
                .find_map(|s| {
                    let c = consumer_of(traces, s.reg, s.ts);
                    in_slot_reach(traces, c, s.prev).then_some((s, c))
                })
                .expect("an rs1 slot whose reader can reach back");
            repoint(traces, slot.node, slot.ts);
            repoint(traces, consumer, slot.prev);
            (Some(P - 1), Some(cpu_without_slot_range(traces, 0)))
        }
        Forgery::SlotCycle => {
            let all = slots(traces);
            let (rs1, rs2) = all
                .iter()
                .filter(|s| matches!(s.node, Node::Slot { k: 0, .. }) && s.reg != 255)
                .find_map(|s| {
                    let Node::Slot { t, r, .. } = s.node else {
                        return None;
                    };
                    all.iter()
                        .find(|o| o.node == Node::Slot { t, r, k: 1 } && o.reg == s.reg)
                        .map(|o| (*s, *o))
                })
                .expect("a row reading one register twice");
            assert_eq!(rs2.prev, rs1.ts, "rs2 reads rs1's token");
            let consumer = consumer_of(traces, rs2.reg, rs2.ts);
            assert!(in_slot_reach(traces, consumer, rs1.prev));
            repoint(traces, rs1.node, rs2.ts);
            repoint(traces, consumer, rs1.prev);
            (Some(P - 2), Some(cpu_without_slot_range(traces, 0)))
        }
        Forgery::SlotPastReach => {
            let (keep, slot) = residual_rows(traces)
                .into_iter()
                .filter(|r| r.keep)
                .find_map(|r| {
                    let c = consumer_of(traces, r.reg, r.ts);
                    matches!(c, Node::Slot { .. }).then_some((r, c))
                })
                .expect("a keep-alive read by a slot");
            let (table, row) = residual_mut(traces, keep.node);
            set_u64(table, row, res::MU_KEEP, 0);
            repoint(traces, slot, keep.old_ts);
            let Node::Slot { k, .. } = slot else {
                unreachable!()
            };
            // The slot at `keep.ts + 1` reads `keep.old_ts`: its step less one.
            let value = keep.ts - keep.old_ts;
            assert!(
                value >= TWO_16,
                "the keep-alive's step was past the slot's reach"
            );
            (Some(value), Some(cpu_without_slot_range(traces, k)))
        }
        Forgery::ResidualSelfLoop => {
            let (row, consumer) = residual_rows(traces)
                .into_iter()
                .filter(|r| r.read && !r.keep && r.reg != 255 && read_keeps_value(traces, r.node))
                .find_map(|r| {
                    let c = consumer_of(traces, r.reg, r.ts);
                    in_slot_reach(traces, c, r.old_ts).then_some((r, c))
                })
                .expect("a residual read whose reader can reach back");
            // `old_ts = ts − 1 − D0 − 2^16·D1 = ts` at D0 = 2^16 − 1,
            // D1 = −1.
            let (table, r) = residual_mut(traces, row.node);
            set_u64(table, r, res::D0, TWO_16 - 1);
            set(table, r, res::D1, FE::zero() - FE::one());
            repoint(traces, consumer, row.old_ts);
            (
                Some(P - 1),
                Some(Box::new(|name: &str| {
                    name.starts_with("MEMW_R[").then(|| {
                        let all = residual::bus_interactions();
                        let kept: Vec<BusInteraction> = all
                            .iter()
                            .filter(|bi| {
                                !(bi.bus_id == u64::from(BusId::IsHalfword)
                                    && bi.columns_read().contains(&res::D1))
                            })
                            .cloned()
                            .collect();
                        assert_eq!(kept.len() + 1, all.len());
                        kept
                    })
                })),
            )
        }
        Forgery::KeepAliveChangesValue => {
            let (keep, slot) = keep_alive_read_by_a_write(traces);
            let (table, r) = residual_mut(traces, keep.node);
            let forged = get(table, r, res::VAL_0) + 1;
            set_u64(table, r, res::VAL_0, forged);
            let (table, r, _) = slot_mut(traces, slot);
            set_u64(table, r, inrow_cols::OLD_RD_0, forged);
            (None, None)
        }
        Forgery::WrongOldRd => {
            let slot = slots(traces)
                .into_iter()
                .find(|s| matches!(s.node, Node::Slot { k: 2, .. }))
                .expect("a write");
            let (table, r, _) = slot_mut(traces, slot.node);
            let old = get(table, r, inrow_cols::OLD_RD_0);
            set_u64(table, r, inrow_cols::OLD_RD_0, old ^ 1);
            (None, None)
        }
        Forgery::KeepAliveInventsValue => {
            let (keep, slot) = keep_alive_read_by_a_write(traces);
            let (table, r) = residual_mut(traces, keep.node);
            let forged = get(table, r, res::VAL_0) + 1;
            set_u64(table, r, res::VAL_0, forged);
            set_u64(table, r, res::OLD_0, forged);
            let (table, r, _) = slot_mut(traces, slot);
            set_u64(table, r, inrow_cols::OLD_RD_0, forged);
            (None, None)
        }
        Forgery::ReadWithoutRequester => {
            let (keep, slot) = keep_alive_read_by_a_write(traces);
            let (table, r) = residual_mut(traces, keep.node);
            set_u64(table, r, res::MU_KEEP, 0);
            set_u64(table, r, res::MU_READ, 1);
            let forged = get(table, r, res::VAL_0) + 1;
            set_u64(table, r, res::VAL_0, forged);
            let (table, r, _) = slot_mut(traces, slot);
            set_u64(table, r, inrow_cols::OLD_RD_0, forged);
            (None, None)
        }
    }
}

/// The residual constraints' values on row `r` of `table`.
pub(crate) fn residual_constraints(table: &Table, r: usize) -> Vec<FE> {
    let row: Vec<FE> = (0..res::NUM_COLUMNS)
        .map(|c| *table.get_main(r, c))
        .collect();
    eval_residual(&row)
}

fn eval_residual(row: &[FE]) -> Vec<FE> {
    use math::field::element::FieldElement;
    use stark::constraints::builder::{ConstraintSet, ProverEvalFolder};
    use stark::frame::Frame;
    use stark::table::TableView;
    use stark::traits::TransitionEvaluationContext;
    let set = residual::ResidualConstraints;
    let n = set.meta().len();
    let frame = Frame::<GoldilocksField, GoldilocksExtension>::new(vec![TableView::new(
        vec![row.to_vec()],
        vec![vec![]],
    )]);
    let no_e: Vec<FieldElement<GoldilocksExtension>> = vec![];
    let offset_e = FieldElement::<GoldilocksExtension>::zero();
    let ctx =
        TransitionEvaluationContext::new_prover(frame.as_row_frame(), &no_e, &no_e, &offset_e);
    let mut base = vec![FE::zero(); n];
    let mut ext = vec![FieldElement::<GoldilocksExtension>::zero(); n];
    let mut folder = ProverEvalFolder::new(&ctx, &mut base, &mut ext);
    set.eval(&mut folder);
    base
}

/// The program's honest in-row build: every inner bus balances (the control
/// each forgery's refusal is read against).
fn honest(name: &str) -> (Elf, Traces) {
    let (program, logs) = run(name);
    let mut traces = in_row_build(&program, &logs);
    let control = left(&program, &mut traces, &|_| None);
    assert!(control.accepted(), "{name}, honest: {control}");
    (program, traces)
}

// ---- the tests --------------------------------------------------------------------------------------------------------

/// The keep-alive rule at its edge: a step of 2^16 is in a slot's reach, a
/// step of 2^16 + 1 moves the slot to `ts − 1`.
#[test]
fn the_keep_alive_rule_moves_a_slot_only_past_its_reach() {
    use super::in_row_prev;
    let ts = 5 * TWO_16 + 7;
    assert_eq!(in_row_prev(ts, ts - 1), ts - 1);
    assert_eq!(in_row_prev(ts, ts - TWO_16), ts - TWO_16);
    assert_eq!(in_row_prev(ts, ts - TWO_16 - 1), ts - 1);
    assert_eq!(in_row_prev(ts, 1), ts - 1);
}

/// ★ Keep-alive rows map one to one onto the slots that need them: each
/// keep-alive at `t` is consumed by one CPU slot at `t + 1`, carries a step
/// past the slot's reach, and every slot's step is within it. Across the
/// programs, the keep-alive program's five among them.
#[test]
fn keep_alive_rows_map_one_to_one_onto_the_slots_that_read_ts_less_one() {
    let mut keeps_seen = 0;
    for name in [
        "keep_alive_regs",
        "all_instructions_64",
        "test_keccak_multi",
        "loop_4096",
    ] {
        let (_program, traces) = honest(name);
        let all = slots(&traces);
        for s in &all {
            assert!(
                s.ts > s.prev && s.ts - s.prev <= TWO_16,
                "{name}: a slot's step past its reach: {s:?}"
            );
        }
        for keep in residual_rows(&traces).into_iter().filter(|r| r.keep) {
            let readers: Vec<&Slot> = all
                .iter()
                .filter(|s| s.reg == keep.reg && s.prev == keep.ts)
                .collect();
            assert_eq!(readers.len(), 1, "{name}: {keep:?} read by {readers:?}");
            assert_eq!(readers[0].ts, keep.ts + 1, "{name}: at ts − 1");
            assert!(
                keep.ts + 1 - keep.old_ts > TWO_16,
                "{name}: a keep-alive not needed: {keep:?}"
            );
            keeps_seen += 1;
        }
    }
    assert_eq!(keeps_seen, 5, "the keep-alive program's five");
}

/// The forgeries a range check refuses: each leaves exactly its slot's (or
/// row's) out-of-range `IS_HALF` value and balances every other bus; with that
/// one check out of the AIR, it balances every bus.
fn refused_by_its_range_check(forgery: Forgery) {
    let name = program_of(forgery);
    let (program, mut traces) = honest(name);
    let (want, witness) = forge(forgery, &mut traces);
    let want = want.expect("a range value");
    let forged = left(&program, &mut traces, &|_| None);
    assert!(
        forged.buses.is_empty() && forged.memory_registers.is_empty(),
        "{forgery:?} on {name}: only the range check refuses it: {forged}"
    );
    assert_eq!(
        forged.out_of_range,
        vec![want],
        "{forgery:?} on {name}: {forged}"
    );
    let witness = witness.expect("a witness");
    let without = left(&program, &mut traces, &*witness);
    assert!(
        without.accepted(),
        "{forgery:?} on {name}: without its range check the forgery balances: {without}"
    );
}

/// ★ O1: a slot consuming its own token (`PREV = ts`) is refused by its
/// `IS_HALF[ts − PREV − 1]`, which reads −1.
#[test]
fn a_slot_consuming_its_own_token_is_refused_by_its_range_check() {
    refused_by_its_range_check(Forgery::SlotSelfLoop);
}

/// ★ O1: `rs1` consuming `rs2`'s token of the same row (`PREV = ts + 1`), a
/// two-node cycle off the register's path, is refused by `rs1`'s range check,
/// which reads −2.
#[test]
fn two_slots_consuming_each_others_tokens_are_refused_by_the_range_check() {
    refused_by_its_range_check(Forgery::SlotCycle);
}

/// ★ O1: a keep-alive dropped, its slot reading the token from more than
/// 2^16 before, is refused by the slot's range check (the step less one,
/// 2^16 or more).
#[test]
fn a_step_past_the_slots_reach_without_its_keep_alive_is_refused() {
    refused_by_its_range_check(Forgery::SlotPastReach);
}

/// ★ O1 (the residual table's limbs): a residual read consuming its own
/// token needs `D0 + 2^16·D1 = −1`, so `D1 = −1`, which its high limb's range
/// check refuses.
#[test]
fn a_residual_read_consuming_its_own_token_is_refused_by_its_high_limb() {
    refused_by_its_range_check(Forgery::ResidualSelfLoop);
}

/// ★ O2: a keep-alive that changes its register's value (read here by a
/// write's old value, which no other bus reads) balances every bus; its
/// constraint `MU_KEEP·(OLD − VAL) = 0` alone refuses it, and reads zero on
/// every honest row.
#[test]
fn a_keep_alive_that_changes_the_value_is_refused_by_its_constraint_alone() {
    let name = program_of(Forgery::KeepAliveChangesValue);
    let (program, mut traces) = honest(name);
    for table in &traces.memw_registers {
        for r in 0..table.num_rows() {
            assert!(
                residual_constraints(table, r)
                    .iter()
                    .all(|v| *v == FE::zero()),
                "{name}: row {r}"
            );
        }
    }
    let (keep, _) = keep_alive_read_by_a_write(&traces);
    forge(Forgery::KeepAliveChangesValue, &mut traces);
    let forged = left(&program, &mut traces, &|_| None);
    assert!(
        forged.accepted(),
        "the forgery balances every bus: {forged}"
    );
    let (table, r) = residual_mut(&mut traces, keep.node);
    let values = residual_constraints(table, r);
    let broken: Vec<usize> = (0..values.len())
        .filter(|&i| values[i] != FE::zero())
        .collect();
    assert_eq!(broken, vec![4], "MU_KEEP·(OLD_0 − VAL_0) alone: {values:?}");
}

/// ★ O4: a write's old value one off leaves the REG bus: the token it
/// consumes no one produced, and the one produced no one consumes.
#[test]
fn a_wrong_rd_old_value_is_refused_on_the_reg_bus() {
    let name = program_of(Forgery::WrongOldRd);
    let (program, mut traces) = honest(name);
    forge(Forgery::WrongOldRd, &mut traces);
    let forged = left(&program, &mut traces, &|_| None);
    assert_eq!(forged.bus_names(), vec!["Reg"], "{forged}");
    let reg = &forged.buses[&u64::from(BusId::Reg)];
    assert_eq!(reg.len(), 2, "{forged}");
    assert!(forged.out_of_range.is_empty(), "{forged}");
}

/// ★ O4: a keep-alive that keeps a value (its constraint holds) but one no
/// one produced leaves the REG bus.
#[test]
fn a_keep_alive_inventing_a_value_is_refused_on_the_reg_bus() {
    let name = program_of(Forgery::KeepAliveInventsValue);
    let (program, mut traces) = honest(name);
    let (keep, _) = keep_alive_read_by_a_write(&traces);
    forge(Forgery::KeepAliveInventsValue, &mut traces);
    let (table, r) = residual_mut(&mut traces, keep.node);
    assert!(
        residual_constraints(table, r)
            .iter()
            .all(|v| *v == FE::zero())
    );
    let forged = left(&program, &mut traces, &|_| None);
    assert_eq!(forged.bus_names(), vec!["Reg"], "{forged}");
    assert_eq!(forged.buses[&u64::from(BusId::Reg)].len(), 2, "{forged}");
}

/// ★ A residual read that changes the value with no requester (a keep-alive
/// turned into a read: every residual constraint holds, the REG bus
/// balances) is refused on the MEMW bus alone: its read receiver has no
/// sender.
#[test]
fn a_residual_read_with_no_requester_is_refused_on_the_memw_bus() {
    let name = program_of(Forgery::ReadWithoutRequester);
    let (program, mut traces) = honest(name);
    let (keep, _) = keep_alive_read_by_a_write(&traces);
    forge(Forgery::ReadWithoutRequester, &mut traces);
    let (table, r) = residual_mut(&mut traces, keep.node);
    assert!(
        residual_constraints(table, r)
            .iter()
            .all(|v| *v == FE::zero())
    );
    let forged = left(&program, &mut traces, &|_| None);
    assert_eq!(forged.bus_names(), vec!["Memw"], "{forged}");
    let memw = &forged.buses[&u64::from(BusId::Memw)];
    assert_eq!(memw.len(), 1, "{forged}");
    assert!(memw[0].1 < 0, "a receive no one sends: {forged}");
}

/// ★ The residual table's constraints on synthetic rows: zero on an honest
/// read, write, keep-alive and padding row; each flag off a bit, two flags
/// at once (`μ = 2`), and a keep-alive changing either word, each read
/// non-zero at its own constraint.
#[test]
fn the_residual_constraints_read_zero_on_honest_rows_and_not_on_violating_ones() {
    let row = |f: &dyn Fn(&mut [FE])| {
        let mut row = vec![FE::zero(); res::NUM_COLUMNS];
        row[res::REG] = FE::from(10u64);
        row[res::TS_0] = FE::from(400u64);
        row[res::VAL_0] = FE::from(7u64);
        row[res::OLD_0] = FE::from(7u64);
        f(&mut row);
        row
    };
    let zero = |v: &[FE]| v.iter().all(|x| *x == FE::zero());
    let broken = |v: &[FE]| {
        (0..v.len())
            .filter(|&i| v[i] != FE::zero())
            .collect::<Vec<_>>()
    };
    assert!(
        zero(&eval_residual(&[FE::zero(); res::NUM_COLUMNS])),
        "padding"
    );
    assert!(
        zero(&eval_residual(&row(&|r| r[res::MU_READ] = FE::one()))),
        "read"
    );
    assert!(
        zero(&eval_residual(&row(&|r| {
            r[res::MU_WRITE] = FE::one();
            r[res::VAL_0] = FE::from(9u64);
        }))),
        "a write changes the value"
    );
    assert!(
        zero(&eval_residual(&row(&|r| r[res::MU_KEEP] = FE::one()))),
        "keep-alive"
    );
    for (i, c) in [res::MU_READ, res::MU_WRITE, res::MU_KEEP]
        .into_iter()
        .enumerate()
    {
        let v = eval_residual(&row(&|r| r[c] = FE::from(2u64)));
        assert!(broken(&v).contains(&i), "flag {c} = 2: {v:?}");
    }
    let v = eval_residual(&row(&|r| {
        r[res::MU_READ] = FE::one();
        r[res::MU_WRITE] = FE::one();
    }));
    assert_eq!(broken(&v), vec![3], "μ = 2: {v:?}");
    for (word, idx) in [(res::VAL_0, 4), (res::VAL_1, 5)] {
        let v = eval_residual(&row(&|r| {
            r[res::MU_KEEP] = FE::one();
            r[word] = FE::from(8u64);
        }));
        assert_eq!(
            broken(&v),
            vec![idx],
            "a keep-alive changing word {word}: {v:?}"
        );
    }
}

/// ★ O9: every `IS_HALF` sender of the in-row AIRs weighs a bit: the CPU's
/// three slots their flags (bits by the CPU's own constraints), the residual
/// table's two limbs `μ` (a bit by its constraints), and no sender anywhere a
/// negated, difference or linear weight.
#[test]
fn every_is_half_sender_of_the_in_row_airs_weighs_a_bit() {
    let (program, traces) = honest("test_keccak_multi");
    let opts = stark::proof::options::ProofOptions::default_test_options();
    let inputs = crate::BlockAirInputs::new(
        &opts,
        &program,
        crate::cpu_heights_from_traces(&traces),
        &traces.public_output_bytes,
    )
    .expect("the inputs");
    let pages = traces.page_configs.clone();
    let airs = crate::block_airs(
        &program,
        &opts,
        &pages,
        &traces.table_counts(),
        &inputs,
        crate::block_whir::BlockFormat::production().keccak_rnd,
        crate::block_whir::BlockFormat::production().memw_aligned,
        RegisterFormat::InRow,
        crate::block_whir::BlockFormat::production().ecdas,
    );
    let halfword = u64::from(BusId::IsHalfword);
    let mut cpu_slots = 0;
    let mut residual_limbs = 0;
    for air in airs.air_refs() {
        for bi in air
            .bus_interactions()
            .iter()
            .filter(|bi| bi.bus_id == halfword && bi.is_sender)
        {
            match (&bi.multiplicity, air.name()) {
                (Multiplicity::Column(c), name)
                    if name.starts_with("CPU[") && SLOT_FLAG.contains(c) =>
                {
                    assert!(crate::constraints::cpu::BIT_FLAG_COLUMNS.contains(c));
                    cpu_slots += 1;
                }
                (Multiplicity::Sum3(a, b, c), name) if name.starts_with("MEMW_R[") => {
                    assert_eq!((*a, *b, *c), (res::MU_READ, res::MU_WRITE, res::MU_KEEP));
                    residual_limbs += 1;
                }
                (
                    Multiplicity::Negated(_) | Multiplicity::Diff(..) | Multiplicity::Linear(_),
                    name,
                ) => {
                    panic!("{name}: an IS_HALF sender weighs {:?}", bi.multiplicity)
                }
                _ => {}
            }
        }
    }
    assert_eq!(cpu_slots, 3 * traces.cpus.len());
    assert_eq!(residual_limbs, 2 * traces.memw_registers.len());
}

/// ★ O7: DECODE never flags x0: for every instruction of the programs, a
/// register slot reading or writing x0 has its flag off.
#[test]
fn decode_never_flags_a_read_or_write_of_x0() {
    for name in [
        "all_instructions_64",
        "test_keccak_multi",
        "keep_alive_regs",
    ] {
        let (program, _) = run(name);
        let artifacts = super::DecodeArtifacts::from_elf(&program).expect("decode");
        let mut seen = 0;
        for entry in artifacts.decode.entries() {
            let f = &entry.fields;
            assert!(!(f.read_register1 && f.rs1 == 0), "{name}: {f:?}");
            assert!(!(f.read_register2 && f.rs2 == 0), "{name}: {f:?}");
            assert!(!(f.write_register && f.rd == 0), "{name}: {f:?}");
            seen += 1;
        }
        assert!(seen > 0);
    }
}

/// ★ One path a register (§6.1's lemma, on the honest builds): on the REG
/// bus, each register's tokens run from REGISTER's initial one at timestamp 1
/// through every node that consumes and produces one (CPU slots, the PC
/// consume of every CPU row and padding row, residual rows, HALT), each
/// forward in time (HALT's PC bridge alone at one timestamp, its value going
/// to the padding `pc = 1`), to REGISTER's final one; and the PC's path ends on
/// the verifier's `pc = 1`.
#[test]
fn every_register_runs_one_path_from_its_initial_token_to_its_final_one() {
    for name in [
        "all_instructions_64",
        "keep_alive_regs",
        "test_keccak_multi",
        "test_commit_split",
    ] {
        let (program, mut traces) = honest(name);
        let paths = reg_paths(&program, &mut traces);
        assert_eq!(
            paths.len(),
            register::NUM_REGISTERS,
            "{name}: a path a register"
        );
        let pc = &paths[&255];
        let entry = [
            program.entry_point as u32,
            (program.entry_point >> 32) as u32,
        ];
        assert_eq!(
            pc.first(),
            Some(&(1, entry)),
            "{name}: the PC starts at the entry point"
        );
        assert_eq!(
            pc.last().map(|t| t.1),
            Some([1, 0]),
            "{name}: the PC ends on pc = 1"
        );
        let x0 = &paths[&0];
        assert!(x0.iter().all(|t| t.1 == [0, 0]), "{name}: x0 stays 0");
    }
}

/// Each register's path of tokens `(timestamp, value)` on the REG bus, from
/// its initial token to its final one; panics where the tokens do not make
/// one path, forward in time.
fn reg_paths(program: &Elf, traces: &mut Traces) -> BTreeMap<u64, Vec<(u64, [u32; 2])>> {
    use super::bus_net_tests::multiplicity;
    with_verifier_files(program, traces);
    let opts = stark::proof::options::ProofOptions::default_test_options();
    let inputs = crate::BlockAirInputs::new(
        &opts,
        program,
        crate::cpu_heights_from_traces(traces),
        &traces.public_output_bytes,
    )
    .expect("the inputs");
    let counts = traces.table_counts();
    let pages = traces.page_configs.clone();
    let airs = crate::block_airs(
        program,
        &opts,
        &pages,
        &counts,
        &inputs,
        crate::block_whir::BlockFormat::production().keccak_rnd,
        crate::block_whir::BlockFormat::production().memw_aligned,
        RegisterFormat::InRow,
        crate::block_whir::BlockFormat::production().ecdas,
    );
    let reg_bus = u64::from(BusId::Reg);
    type Token = (u64, [u32; 2]);
    // Per register: each consumed token to the token the same node produces;
    // the tokens produced alone (initial) and consumed alone (final).
    let mut edges: BTreeMap<u64, BTreeMap<Token, Token>> = BTreeMap::new();
    let mut starts: BTreeMap<u64, Vec<Token>> = BTreeMap::new();
    let mut ends: BTreeMap<u64, Vec<Token>> = BTreeMap::new();
    for (air, trace, _) in airs.air_trace_pairs(traces) {
        let reg: Vec<&BusInteraction> = air
            .bus_interactions()
            .iter()
            .filter(|bi| bi.bus_id == reg_bus)
            .collect();
        // REGISTER's padding rows give and take back the same token.
        let rows = if air.name() == "REGISTER" {
            register::NUM_REGISTERS
        } else {
            trace.num_rows()
        };
        for row in 0..rows {
            let g = |c: usize| *trace.get_main(row, c);
            let token = |bi: &BusInteraction| -> Option<(u64, Token)> {
                (multiplicity(&bi.multiplicity, &g) != FE::zero()).then(|| {
                    let v: Vec<u64> = bi
                        .values
                        .iter()
                        .flat_map(|v| v.combine_from(g))
                        .map(|x: FE| x.canonical_u64())
                        .collect();
                    (v[0], (v[1], [v[2] as u32, v[3] as u32]))
                })
            };
            let mut i = 0;
            while i < reg.len() {
                if i + 1 < reg.len() && reg[i].is_sender && !reg[i + 1].is_sender {
                    if let (Some((r, c)), Some((r2, p))) = (token(reg[i]), token(reg[i + 1])) {
                        assert_eq!(r, r2, "{}: a node keeps its register", air.name());
                        let old = edges.entry(r).or_default().insert(c, p);
                        assert!(
                            old.is_none(),
                            "{}: two nodes consume x{r}'s {c:?}",
                            air.name()
                        );
                    }
                    i += 2;
                } else {
                    if let Some((r, t)) = token(reg[i]) {
                        let side = if reg[i].is_sender {
                            &mut ends
                        } else {
                            &mut starts
                        };
                        side.entry(r).or_default().push(t);
                    }
                    i += 1;
                }
            }
        }
    }
    let mut paths = BTreeMap::new();
    for (reg, start) in starts {
        assert_eq!(start.len(), 1, "x{reg}: one initial token");
        let end = ends.remove(&reg).unwrap_or_default();
        assert_eq!(end.len(), 1, "x{reg}: one final token");
        let mut nodes = edges.remove(&reg).unwrap_or_default();
        let mut path = vec![start[0]];
        let mut at = start[0];
        while let Some(next) = nodes.remove(&at) {
            let halt_bridge = reg == 255 && next.0 == at.0 && next.1 == [1, 0];
            assert!(
                next.0 > at.0 || halt_bridge,
                "x{reg}: {at:?} → {next:?} goes back in time"
            );
            path.push(next);
            at = next;
        }
        assert_eq!(at, end[0], "x{reg}: the path ends at the final token");
        assert!(nodes.is_empty(), "x{reg}: nodes off the path: {nodes:?}");
        paths.insert(reg, path);
    }
    assert!(
        edges.is_empty(),
        "registers with no initial token: {:?}",
        edges.keys()
    );
    assert!(
        ends.is_empty(),
        "registers with no initial token: {:?}",
        ends.keys()
    );
    paths
}

/// ★ O5: a register access routed to MEMW_A under the in-row format (a
/// residual read taken out, the same MEMW message received by a MEMW_A row in
/// the build's form, its reader re-pointed) balances the MEMW and REG buses
/// (its timestamp comparison is an ALU request the LT tables answer once the
/// prover adds its row), but leaves the MEMW_A row's register tokens on the
/// Memory bus: no table produces a register token there any more (REGISTER
/// and the PC are on the REG bus), so the chain it consumes from has no
/// start.
#[test]
fn a_register_routed_to_memw_a_leaves_tokens_no_table_starts() {
    use crate::tables::memw_aligned::{self, AlignedRow, MemwAlignedOrder};
    let (program, mut traces) = honest("test_keccak_multi");
    let (row, consumer) = residual_rows(&traces)
        .into_iter()
        .filter(|r| r.read && !r.keep && r.reg != 255 && read_keeps_value(&traces, r.node))
        .map(|r| (r, consumer_of(&traces, r.reg, r.ts)))
        .find(|(r, c)| in_slot_reach(&traces, *c, r.old_ts) && r.ts < 1 << 32)
        .expect("a residual read");
    let (table, r) = residual_mut(&mut traces, row.node);
    let val = [
        get(table, r, res::VAL_0) as u32,
        get(table, r, res::VAL_1) as u32,
    ];
    let op = super::build_reg_fallback(2 * row.reg, val, val, row.ts, row.old_ts, true);
    // The build's (and the nets' AIRs') MEMW_A form.
    let aligned = AlignedRow::from_memw(&op, MemwAlignedOrder::Lt).expect("an aligned row");
    set_u64(table, r, res::MU_READ, 0);
    repoint(&mut traces, consumer, row.old_ts);
    traces
        .memw_aligneds
        .push(memw_aligned::generate_memw_aligned_trace(&[aligned]));
    let forged = left(&program, &mut traces, &|_| None);
    let lt_request = vec![(vec![row.old_ts, 0, row.ts, 0, 4, 1, 0], 1)];
    assert!(
        forged.bus_names().iter().all(|b| *b == "Alu")
            && forged
                .buses
                .get(&u64::from(BusId::Alu))
                .is_none_or(|t| *t == lt_request)
            && forged.out_of_range.is_empty(),
        "the MEMW and REG buses balance; ALU holds the row's true comparison alone: {forged}"
    );
    assert!(
        forged.memory_registers.iter().any(|(_, net)| *net > 0),
        "a register token consumed on the Memory bus that no table produced: {forged}"
    );
}

/// ★ O6: the PC chain with one 64-bit token cannot run past the exit call.
/// The guest calls `exit(1)`; the forgery keeps executing after it (zeroing
/// x10), with HALT's PC `(1, 0)` so that HALT's consume and emit cancel and
/// the real `next_pc` token goes on to the rows after the exit. REGISTER's
/// final PC is the verifier's `pc = 1` and no real row writes `next_pc = 1`,
/// so the chain is left on the REG bus at x255. Controls: the program with no
/// first exit proves its exit 0 honestly (every bus balances); running past
/// the exit without the HALT edit leaves the REG bus too.
#[test]
fn the_pc_chain_cannot_run_past_the_exit_call() {
    use crate::tables::halt::cols as halt_cols;
    use crate::test_utils::asm_elf_bytes;
    use crate::tests::halt_pc_forgery_poc::{
        FORGED_ROWS, GUEST, forged_logs, patched_elf, run_logs,
    };
    let patched = patched_elf(&asm_elf_bytes(GUEST));
    let program = Elf::load(&patched).expect("the ELF loads");
    let mut traces = in_row_build(&program, &run_logs(&patched));
    let control = left(&program, &mut traces, &|_| None);
    assert!(
        control.accepted(),
        "the patched program's honest exit 0: {control}"
    );

    let elf = asm_elf_bytes(GUEST);
    let program = Elf::load(&elf).expect("the ELF loads");
    let logs = forged_logs(&elf, FORGED_ROWS);
    let mut traces = in_row_build(&program, &logs);
    let plain = left(&program, &mut traces, &|_| None);
    assert_eq!(
        plain.bus_names(),
        vec!["Reg"],
        "past the exit, HALT honest: {plain}"
    );
    set_u64(&mut traces.halt, 0, halt_cols::PC_0, 1);
    set_u64(&mut traces.halt, 0, halt_cols::PC_1, 0);
    let forged = left(&program, &mut traces, &|_| None);
    assert_eq!(
        forged.bus_names(),
        vec!["Reg"],
        "past the exit, HALT's pc 1: {forged}"
    );
    let reg = &forged.buses[&u64::from(BusId::Reg)];
    assert!(
        reg.iter().all(|(t, _)| t[0] == 255),
        "the PC's tokens alone are left: {forged}"
    );
    assert!(
        reg.iter().any(|(t, _)| t[2..] == [1, 0]),
        "REGISTER's final pc = 1 is not reached: {forged}"
    );
}

/// ★ O6, the constants: HALT's PC emit is `(255, T + 1, 1, 0)` in its AIR
/// (no column), REGISTER's x255 row is the PC's (`IS_PC`) and its final file
/// holds `pc = 1` (the halted one the verifier rebuilds), and no CPU row of
/// the honest builds but a padding row carries `next_pc = 1`.
#[test]
fn the_pc_chains_end_is_the_verifiers_constants() {
    use crate::tables::halt;
    use stark::lookup::BusValue;
    let emit = halt::bus_interactions_inrow()
        .into_iter()
        .filter(|bi| bi.bus_id == u64::from(BusId::Reg) && !bi.is_sender)
        .collect::<Vec<_>>();
    assert_eq!(emit.len(), 1);
    assert!(matches!(emit[0].multiplicity, Multiplicity::One));
    let constant = |v: &BusValue, want: u64| {
        v.combine_from(|_| FE::from(12345u64)) == vec![FE::from(want)]
            && v.combine_from(|_| FE::from(54321u64)) == vec![FE::from(want)]
    };
    assert!(
        constant(&emit[0].values[0], 255)
            && constant(&emit[0].values[2], 1)
            && constant(&emit[0].values[3], 0)
    );
    let fini = register::halted_fini(7);
    let columns = inrow::preprocessed_columns(&fini, &fini);
    let pc_row = (0..register::NUM_REGISTERS)
        .find(|&r| inrow::register_of_row(r) == 255)
        .expect("x255's row");
    assert_eq!(columns[inrow::cols::IS_PC][pc_row], FE::one());
    assert_eq!(
        (
            columns[inrow::cols::FINI_0][pc_row],
            columns[inrow::cols::FINI_1][pc_row]
        ),
        (FE::one(), FE::zero())
    );
    for name in [
        "all_instructions_64",
        "keep_alive_regs",
        "test_keccak_multi",
    ] {
        let (_program, traces) = honest(name);
        for t in &traces.cpus {
            for r in 0..t.num_rows() {
                let padding = get(t, r, cpu_cols::PC_0) == 1 && get(t, r, cpu_cols::PC_1) == 0;
                let next_one =
                    get(t, r, cpu_cols::NEXT_PC_0) == 1 && get(t, r, cpu_cols::NEXT_PC_1) == 0;
                assert!(
                    !next_one || padding,
                    "{name}: a real row writes next_pc = 1"
                );
            }
        }
    }
}

/// ★ The in-row PC edge has no step check of its own: it consumes the PC token
/// at `ts − 3 + 3·pc_double_read`, and only `IS_BIT(pc_double_read)` (a CPU
/// constraint) keeps that at `ts − 3` or `ts`. This forged build sets
/// `pc_double_read = 4/3` on a padding row (its PC edge then consumes at
/// `ts + 1` the token it produces itself) and bridges the chain past it with
/// one keep-alive `(255, ts − 3) → (255, ts + 1)`: every bus balances (the
/// keep-alive's two `IS_HALF` limbs are in range, a recount the prover makes),
/// so the bit constraint is the only check that refuses it. A cycle of real
/// rows (a loop's body and its taken backward branch) detaches the same way,
/// which would let a prover run a loop iteration off the PC chain.
#[test]
fn the_pc_edge_rests_on_the_pc_double_read_bit_alone() {
    let (program, mut traces) = honest("all_instructions_64");

    // A padding row (pc = next_pc = 1) with a row after it.
    let heights = crate::cpu_heights_from_traces(&traces);
    let bases = cpu::timestamp_bases(&heights);
    let (t, r) = (0..traces.cpus.len())
        .flat_map(|t| (0..traces.cpus[t].num_rows()).map(move |r| (t, r)))
        .find(|&(t, r)| {
            let table = &traces.cpus[t];
            r + 1 < table.num_rows() && get(table, r, cpu_cols::PC_0) == cpu::CPU_PADDING_PC
        })
        .expect("a padding row with a row after it");
    let ts = bases[t] + get(&traces.cpus[t], r, cpu_cols::TIMESTAMP);
    let four_thirds = FE::from(4u64) * FE::from(3u64).inv().expect("3 is invertible");
    set(
        &mut traces.cpus[t],
        r,
        cpu_cols::PC_DOUBLE_READ,
        four_thirds,
    );

    // One residual padding row (μ = 0) made the bridging keep-alive.
    let (rt, rr) = (0..traces.memw_registers.len())
        .flat_map(|t| (0..traces.memw_registers[t].num_rows()).map(move |r| (t, r)))
        .find(|&(t, r)| {
            let table = &traces.memw_registers[t];
            [res::MU_READ, res::MU_WRITE, res::MU_KEEP]
                .iter()
                .all(|&c| get(table, r, c) == 0)
        })
        .expect("a residual padding row");
    for (c, v) in [
        (res::REG, 255u64),
        (res::TS_0, ts + 1),
        (res::TS_1, 0),
        (res::VAL_0, cpu::CPU_PADDING_PC),
        (res::VAL_1, 0),
        (res::OLD_0, cpu::CPU_PADDING_PC),
        (res::OLD_1, 0),
        (res::D0, 3),
        (res::D1, 0),
        (res::MU_KEEP, 1),
    ] {
        set_u64(&mut traces.memw_registers[rt], rr, c, v);
    }

    let forged = left(&program, &mut traces, &|_| None);
    println!("PC EDGE: pc_double_read = 4/3 at CPU[{t}] row {r} (ts {ts}): {forged}");
    assert!(
        forged.accepted(),
        "every bus balances with pc_double_read = 4/3 and one keep-alive: {forged}"
    );
    let pdr = *traces.cpus[t].get_main(r, cpu_cols::PC_DOUBLE_READ);
    assert_ne!(
        pdr * (pdr - FE::one()),
        FE::zero(),
        "only IS_BIT(pc_double_read) refuses it"
    );
    assert!(
        crate::constraints::cpu::BIT_FLAG_COLUMNS.contains(&cpu_cols::PC_DOUBLE_READ),
        "the bit constraint on pc_double_read is in the CPU's set"
    );
}

/// Every instruction kind at a spread of register fields, as `(kind, instruction)`.
fn every_instruction() -> Vec<(
    &'static str,
    executor::vm::instruction::decoding::Instruction,
)> {
    use executor::vm::instruction::decoding::{
        ArithOp, Comparison, CsrOp, Instruction, LoadStoreWidth,
    };
    let ops = [
        ArithOp::Add,
        ArithOp::Sub,
        ArithOp::Xor,
        ArithOp::Or,
        ArithOp::And,
        ArithOp::ShiftLeftLogical,
        ArithOp::ShiftRightLogical,
        ArithOp::ShiftRightArith,
        ArithOp::SetLessThan,
        ArithOp::SetLessThanU,
        ArithOp::Mul,
        ArithOp::MulHigh,
        ArithOp::MulHighSignedUnsigned,
        ArithOp::MulHighUnsigned,
        ArithOp::Div,
        ArithOp::DivUnsigned,
        ArithOp::Remainder,
        ArithOp::RemainderUnsigned,
    ];
    let widths = [
        LoadStoreWidth::Byte,
        LoadStoreWidth::Half,
        LoadStoreWidth::Word,
        LoadStoreWidth::DoubleWord,
        LoadStoreWidth::ByteUnsigned,
        LoadStoreWidth::HalfUnsigned,
        LoadStoreWidth::WordUnsigned,
    ];
    let conds = [
        Comparison::Equal,
        Comparison::NotEqual,
        Comparison::LessThan,
        Comparison::GreaterOrEqual,
        Comparison::LessThanUnsigned,
        Comparison::GreaterOrEqualUnsigned,
    ];
    let csrs = [
        CsrOp::CSRRW,
        CsrOp::CSRRS,
        CsrOp::CSRRC,
        CsrOp::CSRRWI,
        CsrOp::CSRRSI,
        CsrOp::CSRRCI,
    ];
    let mut all = Vec::new();
    for a in 0..32u32 {
        for b in [0u32, 1, 5, 17, 31] {
            for op in ops {
                all.push((
                    "Arith",
                    Instruction::Arith {
                        dst: a,
                        src1: b,
                        src2: a ^ 1,
                        op,
                    },
                ));
                all.push((
                    "Arith",
                    Instruction::Arith {
                        dst: b,
                        src1: a,
                        src2: b,
                        op,
                    },
                ));
                all.push((
                    "ArithImm",
                    Instruction::ArithImm {
                        dst: a,
                        src: b,
                        imm: -1,
                        op,
                    },
                ));
                all.push((
                    "ArithImm",
                    Instruction::ArithImm {
                        dst: b,
                        src: a,
                        imm: 7,
                        op,
                    },
                ));
                all.push((
                    "ArithW",
                    Instruction::ArithW {
                        dst: a,
                        src1: b,
                        src2: a,
                        op,
                    },
                ));
                all.push((
                    "ArithW",
                    Instruction::ArithW {
                        dst: b,
                        src1: a,
                        src2: b,
                        op,
                    },
                ));
                all.push((
                    "ArithImmW",
                    Instruction::ArithImmW {
                        dst: a,
                        src: b,
                        imm: 3,
                        op,
                    },
                ));
                all.push((
                    "ArithImmW",
                    Instruction::ArithImmW {
                        dst: b,
                        src: a,
                        imm: 3,
                        op,
                    },
                ));
            }
            all.push(("JAL", Instruction::JumpAndLink { dst: a, offset: 8 }));
            all.push((
                "JALR",
                Instruction::JumpAndLinkRegister {
                    base: b,
                    dst: a,
                    offset: 4,
                },
            ));
            all.push((
                "JALR",
                Instruction::JumpAndLinkRegister {
                    base: a,
                    dst: b,
                    offset: 4,
                },
            ));
            for width in widths {
                all.push((
                    "Store",
                    Instruction::Store {
                        src: a,
                        offset: 0,
                        base: b,
                        width,
                    },
                ));
                all.push((
                    "Store",
                    Instruction::Store {
                        src: b,
                        offset: 0,
                        base: a,
                        width,
                    },
                ));
                all.push((
                    "Load",
                    Instruction::Load {
                        dst: a,
                        offset: 0,
                        base: b,
                        width,
                    },
                ));
                all.push((
                    "Load",
                    Instruction::Load {
                        dst: b,
                        offset: 0,
                        base: a,
                        width,
                    },
                ));
            }
            for cond in conds {
                all.push((
                    "Branch",
                    Instruction::Branch {
                        src1: a,
                        src2: b,
                        cond,
                        offset: 8,
                    },
                ));
                all.push((
                    "Branch",
                    Instruction::Branch {
                        src1: b,
                        src2: a,
                        cond,
                        offset: 8,
                    },
                ));
            }
            for op in csrs {
                all.push((
                    "CSR",
                    Instruction::CSR {
                        csr: 0x300,
                        src: a,
                        dst: b,
                        op,
                    },
                ));
                all.push((
                    "CSR",
                    Instruction::CSR {
                        csr: 0x300,
                        src: b,
                        dst: a,
                        op,
                    },
                ));
            }
        }
        all.push((
            "LUI",
            Instruction::LoadUpperImm {
                dst: a,
                imm: 0x1000,
            },
        ));
        all.push((
            "AUIPC",
            Instruction::AddUpperImmToPc {
                dst: a,
                imm: 0x1000,
            },
        ));
    }
    all.push(("ECALL", Instruction::EcallEbreak));
    all.push(("FENCE", Instruction::Fence));
    all
}

/// ★ DECODE over every instruction kind and register field, independent of any
/// program (unlike [`decode_never_flags_a_read_or_write_of_x0`]): no flag on
/// x0; `rs2` and `rd` below 32; `rs1` is 255 (the PC) only on JAL and AUIPC,
/// where it is read and `pc_double_read` is set, and 17 on ECALL; no word
/// instruction names a register past x31 (CPU32 sends `2·rs`, `2·rd` as they
/// are).
#[test]
fn decode_flags_no_x0_and_names_the_pc_only_on_jal_and_auipc() {
    use crate::tables::types::ShrunkDecode;
    let mut kinds: BTreeMap<&str, usize> = BTreeMap::new();
    for (kind, instruction) in every_instruction() {
        let d = ShrunkDecode::from_instruction(instruction, 4);
        *kinds.entry(kind).or_default() += 1;
        assert!(
            !(d.read_register1 && d.rs1 == 0),
            "{kind}: {instruction:?} reads x0 on rs1"
        );
        assert!(
            !(d.read_register2 && d.rs2 == 0),
            "{kind}: {instruction:?} reads x0 on rs2"
        );
        assert!(
            !(d.write_register && d.rd == 0),
            "{kind}: {instruction:?} writes x0"
        );
        assert!(d.rs2 < 32 && d.rd < 32, "{kind}: {instruction:?}: {d:?}");
        match d.rs1 {
            255 => assert!(
                matches!(kind, "JAL" | "AUIPC") && d.read_register1,
                "{kind}: {instruction:?} names x255"
            ),
            r => assert!(r < 32, "{kind}: {instruction:?} names x{r}"),
        }
        if kind == "ECALL" {
            assert!(d.read_register1 && d.rs1 == 17 && d.ecall, "{d:?}");
        }
        // The CPU's pc_double_read is `read_register1 ∧ rs1 = 255` off word instructions.
        let pdr = !d.word_instr && d.read_register1 && d.rs1 == 255;
        assert_eq!(
            pdr,
            matches!(kind, "JAL" | "AUIPC"),
            "{kind}: {instruction:?}"
        );
        if d.word_instr {
            assert!(d.rs1 < 32 && d.rs2 < 32 && d.rd < 32, "{kind}: {d:?}");
        }
    }
    println!("DECODE KINDS CHECKED: {kinds:?}");
    assert_eq!(kinds.len(), 14, "every instruction kind: {kinds:?}");
}

// ---- registers routed to MEMW_A under the production pair (in-row registers, MEMW_A in limbs) ------------------------

/// The in-row build of `program`'s run with MEMW_A in the limbs form, as the
/// pinned block format builds it.
fn in_row_limbs_build(program: &Elf, logs: &[Log]) -> Traces {
    use crate::tables::memw_aligned::MemwAlignedOrder;
    windowed(program, logs, &MaxRowsConfig::small(), 1000, &|b| {
        b.register_format(RegisterFormat::InRow)
            .and_then(|b| b.memw_aligned_order(MemwAlignedOrder::Limbs))
            .expect("before any window")
    })
}

/// MEMW_A's interactions in the limbs form, for every MEMW_A table, with
/// `edit` applied (the nets' own AIRs take the per-table STARK's form).
fn memw_a_limbs(edit: fn(&mut Vec<BusInteraction>)) -> Over {
    use crate::tables::memw_aligned::{MemwAlignedOrder, bus_interactions_for};
    Box::new(move |name: &str| {
        name.starts_with("MEMW_A[").then(|| {
            let mut list = bus_interactions_for(MemwAlignedOrder::Limbs);
            edit(&mut list);
            list
        })
    })
}

/// The limbs' high-limb lookup, `IS_HALF[STEP_1]`, taken out: a witness mutant.
fn without_the_high_limb(list: &mut Vec<BusInteraction>) {
    use crate::tables::memw_aligned::cols as aligned;
    let before = list.len();
    list.retain(|i| {
        !(i.bus_id == u64::from(BusId::IsHalfword) && i.columns_read().contains(&aligned::STEP_1))
    });
    assert_eq!(list.len(), before - 1, "one IS_HALF[STEP_1] lookup");
}

/// A residual register read of the honest limbs build of `test_keccak_multi`
/// taken out and its MEMW message received by a new MEMW_A row in the limbs
/// form instead, the read's REG consumer re-pointed past it (so the REG and
/// MEMW buses balance). `old_ts_is_ts`: the row's old token is its own new one
/// (a self-loop), its limbs set to `STEP_0 + 2^16·STEP_1 = p − 1`.
fn register_routed_to_memw_a_limbs(old_ts_is_ts: bool) -> (Elf, Traces, Over) {
    use crate::tables::memw_aligned::{self, AlignedRow, MemwAlignedOrder, cols as aligned};
    let (program, logs) = run("test_keccak_multi");
    let mut traces = in_row_limbs_build(&program, &logs);
    let over = memw_a_limbs(|_| {});
    let control = left(&program, &mut traces, &*over);
    assert!(control.accepted(), "the honest limbs build: {control}");
    let (row, consumer) = residual_rows(&traces)
        .into_iter()
        .filter(|r| r.read && !r.keep && r.reg != 255 && read_keeps_value(&traces, r.node))
        .map(|r| (r, consumer_of(&traces, r.reg, r.ts)))
        .find(|(r, c)| in_slot_reach(&traces, *c, r.old_ts) && r.ts < 1 << 32)
        .expect("a residual read");
    let (table, r) = residual_mut(&mut traces, row.node);
    let val = [
        get(table, r, res::VAL_0) as u32,
        get(table, r, res::VAL_1) as u32,
    ];
    let op = super::build_reg_fallback(2 * row.reg, val, val, row.ts, row.old_ts, true);
    let aligned_row =
        AlignedRow::from_memw(&op, MemwAlignedOrder::Limbs).expect("an aligned row in limbs");
    assert!(aligned_row.limbs(), "the row is in the limbs form");
    set_u64(table, r, res::MU_READ, 0);
    repoint(&mut traces, consumer, row.old_ts);
    let mut memw_a = memw_aligned::generate_memw_aligned_trace(&[aligned_row]);
    if old_ts_is_ts {
        set_u64(&mut memw_a, 0, aligned::STEP_0, 0);
        set_u64(&mut memw_a, 0, aligned::STEP_1, (P - 1) >> 16);
    }
    traces.memw_aligneds.push(memw_a);
    (program, traces, over)
}

/// ★ A register access routed to MEMW_A in the limbs form under the in-row
/// register format: MEMW_A's `IS_REGISTER` column is free, so a row may take
/// a chip's register read. Its MEMW and REG buses balance, it asks no
/// comparison of the ALU bus (the limbs replace the LT lookup) and its limbs
/// are in range, but its old register token is left on the Memory bus: no
/// table produces a register token there under the in-row format, so the
/// chain it would consume from has no start.
#[test]
fn a_register_routed_to_memw_a_in_limbs_leaves_a_token_no_table_starts() {
    let (program, mut traces, over) = register_routed_to_memw_a_limbs(false);
    let forged = left(&program, &mut traces, &*over);
    println!("REGISTER ROUTED TO MEMW_A (limbs): {forged}");
    assert!(
        forged.buses.is_empty() && forged.out_of_range.is_empty(),
        "the MEMW, REG and ALU buses balance and the limbs are in range: {forged}"
    );
    assert!(
        forged.memory_registers.iter().any(|(_, net)| *net > 0),
        "a register token consumed on the Memory bus that no table produced: {forged}"
    );
}

/// ★ The closed loop of the same routing: the MEMW_A row consumes the very
/// register token it produces, so the Memory bus balances too. The limbs then
/// read `STEP_0 + 2^16·STEP_1 = p − 1`, and the high limb's `IS_HALF` lookup
/// alone refuses it; with that one lookup taken out every bus balances.
#[test]
fn a_register_self_loop_through_memw_a_in_limbs_is_refused_by_its_high_limb() {
    let (program, mut traces, _) = register_routed_to_memw_a_limbs(true);
    let over = memw_a_limbs(|_| {});
    let forged = left(&program, &mut traces, &*over);
    println!("REGISTER SELF-LOOP THROUGH MEMW_A (limbs): {forged}");
    assert!(
        forged.buses.is_empty() && forged.memory_registers.is_empty(),
        "only the range check refuses it: {forged}"
    );
    assert_eq!(
        forged.out_of_range,
        vec![(P - 1) >> 16],
        "the high limb is out of range: {forged}"
    );
    let without = left(&program, &mut traces, &*memw_a_limbs(without_the_high_limb));
    assert!(
        without.accepted(),
        "without the high limb's lookup the self-loop balances: {without}"
    );
}
