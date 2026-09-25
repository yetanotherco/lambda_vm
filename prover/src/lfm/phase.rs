//! Emission-phase labels for the recursion-size census (`LAMBDA_VM_GAP_PHASES=1`).
//!
//! Diagnostic only. An emitter opens a scope with [`enter`]; the builder tags
//! every instruction it emits with the path of the scopes open on this thread
//! (`"table/trace_open/merkle"`), and [`report`] prints one `GAPB PHASE` line
//! per path with the instruction count per chip. Off by default: with the
//! variable unset [`enter`] is a branch and nothing is recorded, and no
//! instruction, address or proof byte depends on it either way.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use super::instr::{BaseOp, ExtOp, HashMode, Instr};

/// Programs smaller than this are not reported (unit-test and helper programs).
const REPORT_MIN_INSTRS: usize = 50_000;

pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("LAMBDA_VM_GAP_PHASES").is_ok_and(|v| !v.is_empty()))
}

thread_local! {
    static STACK: RefCell<Vec<&'static str>> = const { RefCell::new(Vec::new()) };
    static CURRENT: Cell<u32> = const { Cell::new(0) };
}

fn names() -> &'static Mutex<(Vec<String>, HashMap<String, u32>)> {
    static NAMES: OnceLock<Mutex<(Vec<String>, HashMap<String, u32>)>> = OnceLock::new();
    NAMES.get_or_init(|| {
        let root = "-".to_string();
        Mutex::new((vec![root.clone()], HashMap::from([(root, 0)])))
    })
}

fn intern(path: String) -> u32 {
    let mut g = names().lock().expect("phase names");
    if let Some(&id) = g.1.get(&path) {
        return id;
    }
    let id = g.0.len() as u32;
    g.0.push(path.clone());
    g.1.insert(path, id);
    id
}

fn refresh() {
    let path = STACK.with(|s| s.borrow().join("/"));
    let id = if path.is_empty() { 0 } else { intern(path) };
    CURRENT.with(|c| c.set(id));
}

/// An open scope; closes on drop.
#[must_use = "a phase scope closes when the guard drops"]
pub struct Guard {
    pushed: bool,
}

/// Open a scope named `name`. A scope re-entered directly inside itself (a
/// walk calling a walk) is not repeated in the path.
pub fn enter(name: &'static str) -> Guard {
    if !enabled() {
        return Guard { pushed: false };
    }
    let pushed = STACK.with(|s| {
        let mut s = s.borrow_mut();
        if s.last() == Some(&name) {
            false
        } else {
            s.push(name);
            true
        }
    });
    if pushed {
        refresh();
    }
    Guard { pushed }
}

impl Drop for Guard {
    fn drop(&mut self) {
        if self.pushed {
            STACK.with(|s| s.borrow_mut().pop());
            refresh();
        }
    }
}

/// The id of the path open on this thread (0 = no scope).
pub fn current() -> u32 {
    CURRENT.with(|c| c.get())
}

const KINDS: [&str; 16] = [
    "const", "balu_lin", "balu_mul", "xalu_lin", "xalu_mul", "select", "bitdec", "hash_c",
    "hash_t", "hash_l", "hash_p", "hint", "pack", "unpack", "keccak", "other",
];

fn kind(i: &Instr) -> usize {
    match i {
        Instr::Const { .. } => 0,
        Instr::BaseAlu { op, .. } => match op {
            BaseOp::Add | BaseOp::Sub => 1,
            _ => 2,
        },
        Instr::ExtAlu { op, .. } => match op {
            ExtOp::Add | ExtOp::Sub => 3,
            _ => 4,
        },
        Instr::Select { .. } => 5,
        Instr::BitDec { .. } => 6,
        Instr::Hash { mode, .. } => match mode {
            HashMode::Compress => 7,
            HashMode::Transcript => 8,
            HashMode::Leaf => 9,
            HashMode::Permute => 10,
        },
        Instr::Hint { .. } => 11,
        Instr::Pack { .. } => 12,
        Instr::Unpack { .. } => 13,
        Instr::KeccakF(_) => 14,
        Instr::Blake3(_) | Instr::Public { .. } => 15,
    }
}

/// Print the per-phase instruction census of one program.
pub fn report(instrs: &[Instr], ids: &[u32]) {
    if instrs.len() < REPORT_MIN_INSTRS || ids.len() != instrs.len() {
        return;
    }
    static PROGRAM: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let prog = PROGRAM.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut by: std::collections::BTreeMap<u32, [u64; KINDS.len()]> = Default::default();
    for (i, id) in instrs.iter().zip(ids) {
        by.entry(*id).or_insert([0; KINDS.len()])[kind(i)] += 1;
    }
    let g = names().lock().expect("phase names");
    let mut lines: Vec<(String, [u64; KINDS.len()])> =
        by.into_iter().map(|(id, c)| (g.0[id as usize].clone(), c)).collect();
    drop(g);
    lines.sort_by(|a, b| a.0.cmp(&b.0));
    for (path, c) in lines {
        let fields: Vec<String> = KINDS
            .iter()
            .zip(c)
            .filter(|(_, n)| *n > 0)
            .map(|(k, n)| format!("{k}={n}"))
            .collect();
        eprintln!(
            "GAPB PHASE prog={prog} instrs={} phase={path} {}",
            instrs.len(),
            fields.join(" ")
        );
    }
}
