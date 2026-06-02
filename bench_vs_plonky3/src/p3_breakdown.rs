//! P3 phase breakdown collector.
//!
//! Wires a custom `tracing_subscriber::Layer` around `p3_uni_stark::prove`
//! (and `verify`), captures every span's wall-time + parent id, reconstructs
//! the span tree, and emits three complementary views as `BREAKDOWN` lines:
//!
//! ```text
//! kind=norm       normalized, prover-agnostic phases (norm_trace_lde,
//!                 norm_constraint_eval, norm_fri, ...). The Lambda emitter in
//!                 prove_bench.rs emits the SAME names from stark::instruments,
//!                 so the two breakdowns diff line-for-line (filter by prover).
//! kind=p3_native  full P3 span tree (prove subtree only), pre-order, with
//!                 depth=, ms= (self-time) and total_ms= (subtree wall).
//! kind=p3_agg     self-time aggregated by span name (e.g. every
//!                 build_merkle_tree instance summed) with a call count.
//! ```
//!
//! Two correctness notes:
//!
//! * prove vs verify: the subscriber stays installed across both, so the
//!   snapshot captures two roots (`prove_with_preprocessed`,
//!   `verify_with_preprocessed`). Every reported view is scoped to the PROVE
//!   root's subtree; verify spans are captured but not reported.
//! * self-time under rayon: P3 runs DFT/Merkle on worker threads, so a grouping
//!   span's children can run in parallel and their summed wall can exceed the
//!   parent's wall. `self_time = total - Σ child_total` then saturates to 0
//!   (correct: the parent did ~no work itself). The normalized phases therefore
//!   use the grouping span's TOTAL (wall) time, the robust comparable quantity.
//!
//! P3 prove-tree shape (from the rev pinned in `Cargo.toml`):
//!
//! ```text
//! prove_with_preprocessed
//! ├── commit to trace data            -> norm_trace_commit
//! │   ├── coset_lde_batch             -> norm_trace_lde
//! │   └── build_merkle_tree           -> norm_trace_merkle
//! ├── quotient_values                 -> norm_constraint_eval
//! ├── commit to quotient poly chunks  -> norm_quotient_commit
//! │   └── build_merkle_tree           -> norm_quotient_merkle
//! └── open                            -> norm_open
//!     └── FRI prover                  -> norm_fri   (norm_deep_ood = open - fri)
//! ```

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tracing::Subscriber;
use tracing::span::{Attributes, Id};
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::util::SubscriberInitExt;

// ----------------------------------------------------------------------------
// Capture layer
// ----------------------------------------------------------------------------

#[derive(Default)]
struct CollectorState {
    spans: Vec<RawSpan>,
    next_id: u64,
    open: HashMap<u64, u64>,
}

#[derive(Debug, Clone)]
struct RawSpan {
    our_id: u64,
    name: String,
    start: Instant,
    end: Option<Instant>,
    parent_our_id: Option<u64>,
    children: Vec<u64>,
}

pub struct Collector {
    state: Arc<Mutex<CollectorState>>,
}

struct CollectorLayer {
    state: Arc<Mutex<CollectorState>>,
}

impl Default for Collector {
    fn default() -> Self {
        Self::new()
    }
}

impl Collector {
    pub fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(CollectorState::default())),
        }
    }

    pub fn install(&self) -> Result<(), tracing_subscriber::util::TryInitError> {
        use tracing_subscriber::layer::SubscriberExt;
        let layer = CollectorLayer {
            state: self.state.clone(),
        };
        tracing_subscriber::registry()
            .with(layer)
            .with(tracing_subscriber::filter::LevelFilter::TRACE)
            .try_init()
    }

    /// Reconstruct the captured span tree into a [`Snapshot`].
    pub fn snapshot(&self) -> Snapshot {
        let spans = { self.state.lock().unwrap().spans.clone() };
        let now = Instant::now();

        // our_id -> position in `nodes`.
        let mut pos_of: HashMap<u64, usize> = HashMap::with_capacity(spans.len());
        let mut nodes: Vec<Node> = Vec::with_capacity(spans.len());
        for s in &spans {
            pos_of.insert(s.our_id, nodes.len());
            let end = s.end.unwrap_or(now);
            nodes.push(Node {
                name: s.name.clone(),
                self_time: Duration::ZERO,
                total_time: end.saturating_duration_since(s.start),
                depth: 0,
                parent: None,
                children: Vec::new(),
                root: 0,
            });
        }

        // Wire parent <-> children by position (preserving creation order).
        for s in &spans {
            let pos = pos_of[&s.our_id];
            if let Some(pid) = s.parent_our_id
                && let Some(&ppos) = pos_of.get(&pid)
            {
                nodes[pos].parent = Some(ppos);
                nodes[ppos].children.push(pos);
            }
        }

        // self_time = total - Σ child_total (saturating; see module note).
        for i in 0..nodes.len() {
            let child_total: Duration =
                nodes[i].children.iter().map(|&c| nodes[c].total_time).sum();
            nodes[i].self_time = nodes[i].total_time.saturating_sub(child_total);
        }

        let roots: Vec<usize> = (0..nodes.len())
            .filter(|&i| nodes[i].parent.is_none())
            .collect();

        // Assign depth + owning root over each subtree, and build a pre-order
        // (root-first) traversal so the emitted view reads top-down.
        let mut order: Vec<usize> = Vec::with_capacity(nodes.len());
        for &r in &roots {
            let mut stack = vec![(r, 0usize)];
            while let Some((n, d)) = stack.pop() {
                nodes[n].depth = d;
                nodes[n].root = r;
                let kids: Vec<usize> = nodes[n].children.clone();
                for c in kids {
                    stack.push((c, d + 1));
                }
            }
            preorder(r, &nodes, &mut order);
        }

        Snapshot {
            nodes,
            roots,
            order,
        }
    }
}

fn preorder(n: usize, nodes: &[Node], out: &mut Vec<usize>) {
    out.push(n);
    for &c in &nodes[n].children {
        preorder(c, nodes, out);
    }
}

impl<S> Layer<S> for CollectorLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, attrs: &Attributes, id: &Id, ctx: Context<S>) {
        let metadata = attrs.metadata();
        let tracing_id = id.clone().into_u64();
        let now = Instant::now();

        let mut st = self.state.lock().unwrap();
        st.next_id += 1;
        let our_id = st.next_id;

        let parent_our_id = ctx
            .lookup_current()
            .and_then(|cur| st.open.get(&cur.id().clone().into_u64()).copied());
        st.open.insert(tracing_id, our_id);

        if let Some(p) = parent_our_id
            && let Some(p_node) = st.spans.iter_mut().find(|n| n.our_id == p)
        {
            p_node.children.push(our_id);
        }

        st.spans.push(RawSpan {
            our_id,
            name: metadata.name().to_string(),
            start: now,
            end: None,
            parent_our_id,
            children: Vec::new(),
        });
    }

    fn on_close(&self, id: Id, _ctx: Context<S>) {
        let tracing_id = id.into_u64();
        let mut st = self.state.lock().unwrap();
        let our_id = match st.open.remove(&tracing_id) {
            Some(v) => v,
            None => return,
        };
        let now = Instant::now();
        if let Some(node) = st.spans.iter_mut().find(|n| n.our_id == our_id) {
            node.end = Some(now);
        }
    }
}

// ----------------------------------------------------------------------------
// Reconstructed tree + analysis
// ----------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Node {
    pub name: String,
    pub self_time: Duration,
    pub total_time: Duration,
    pub depth: usize,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    /// Position index of the root this node belongs to (prove vs verify).
    pub root: usize,
}

#[derive(Debug, Clone)]
pub struct Snapshot {
    pub nodes: Vec<Node>,
    pub roots: Vec<usize>,
    /// Pre-order (root-first) traversal of all roots' subtrees.
    pub order: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct NameAgg {
    pub name: String,
    pub self_time: Duration,
    pub total_time: Duration,
    pub calls: usize,
}

impl Snapshot {
    /// Position of the prove root (span name starts with "prove"). P3 wraps
    /// the real work in an outer `prove` span whose only child is
    /// `prove_with_preprocessed`; this returns the outermost one (used to scope
    /// the native subtree).
    pub fn prove_root(&self) -> Option<usize> {
        self.roots
            .iter()
            .copied()
            .find(|&r| self.nodes[r].name.to_ascii_lowercase().starts_with("prove"))
    }

    /// First node (pre-order) whose sanitized name equals `needle`, anywhere.
    fn find_named(&self, needle: &str) -> Option<usize> {
        self.order
            .iter()
            .copied()
            .find(|&i| sanitize_phase(&self.nodes[i].name) == needle)
    }

    /// First direct child of `parent` whose sanitized name equals `needle`.
    fn child_named(&self, parent: usize, needle: &str) -> Option<usize> {
        self.nodes[parent]
            .children
            .iter()
            .copied()
            .find(|&c| sanitize_phase(&self.nodes[c].name) == needle)
    }

    /// First descendant (DFS, excluding `root` itself) whose sanitized name
    /// equals `needle`.
    fn descendant_named(&self, root: usize, needle: &str) -> Option<usize> {
        let mut stack: Vec<usize> = self.nodes[root].children.clone();
        while let Some(n) = stack.pop() {
            if sanitize_phase(&self.nodes[n].name) == needle {
                return Some(n);
            }
            stack.extend(self.nodes[n].children.iter().copied());
        }
        None
    }

    fn total(&self, idx: usize) -> Duration {
        self.nodes[idx].total_time
    }

    /// Normalized, prover-agnostic phases derived from the P3 prove tree.
    /// Names mirror the Lambda `norm_*` phases so the two diff directly.
    pub fn canonical(&self) -> Vec<(&'static str, Duration)> {
        let mut out: Vec<(&'static str, Duration)> = Vec::new();
        // Navigate from the real uni-stark prove span (its direct children are
        // the round phases), falling back to the outer prove root for older P3.
        let Some(anchor) = self
            .find_named("prove_with_preprocessed")
            .or_else(|| self.prove_root())
        else {
            return out;
        };
        // norm_prove_total is the most inclusive prove span (the outer wrapper).
        let total_node = self.prove_root().unwrap_or(anchor);
        out.push(("norm_prove_total", self.total(total_node)));

        let pr = anchor;
        if let Some(tc) = self.child_named(pr, "commit_to_trace_data") {
            out.push(("norm_trace_commit", self.total(tc)));
            if let Some(lde) = self.child_named(tc, "coset_lde_batch") {
                out.push(("norm_trace_lde", self.total(lde)));
            }
            if let Some(mk) = self.child_named(tc, "build_merkle_tree") {
                out.push(("norm_trace_merkle", self.total(mk)));
            }
        }
        if let Some(q) = self.child_named(pr, "quotient_values") {
            out.push(("norm_constraint_eval", self.total(q)));
        }
        if let Some(qc) = self.child_named(pr, "commit_to_quotient_poly_chunks") {
            out.push(("norm_quotient_commit", self.total(qc)));
            if let Some(mk) = self.child_named(qc, "build_merkle_tree") {
                out.push(("norm_quotient_merkle", self.total(mk)));
            }
        }
        if let Some(op) = self.child_named(pr, "open") {
            let open_total = self.total(op);
            out.push(("norm_open", open_total));
            let fri = self
                .descendant_named(op, "fri_prover")
                .map(|f| self.total(f))
                .unwrap_or(Duration::ZERO);
            out.push(("norm_fri", fri));
            out.push(("norm_deep_ood", open_total.saturating_sub(fri)));
        }
        out
    }

    /// Self-time aggregated by sanitized span name over the given subtree,
    /// sorted by self-time descending.
    pub fn aggregate_subtree(&self, root: usize) -> Vec<NameAgg> {
        let mut map: HashMap<String, NameAgg> = HashMap::new();
        let mut stack = vec![root];
        while let Some(n) = stack.pop() {
            let key = sanitize_phase(&self.nodes[n].name);
            let e = map.entry(key.clone()).or_insert_with(|| NameAgg {
                name: key,
                self_time: Duration::ZERO,
                total_time: Duration::ZERO,
                calls: 0,
            });
            e.self_time += self.nodes[n].self_time;
            e.total_time += self.nodes[n].total_time;
            e.calls += 1;
            stack.extend(self.nodes[n].children.iter().copied());
        }
        let mut v: Vec<NameAgg> = map.into_values().collect();
        v.sort_by(|a, b| b.self_time.cmp(&a.self_time));
        v
    }
}

// ----------------------------------------------------------------------------
// Emission
// ----------------------------------------------------------------------------

fn to_ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// Emit all three breakdown views for the prove subtree as `BREAKDOWN<TAB>...`
/// lines, matching the schema the Lambda side emits.
pub fn print_breakdown(prover: &str, log_rows: u32, rows: usize, wall_ms: f64, snap: &Snapshot) {
    let emit = |kind: &str, phase: &str, extra: String| {
        println!(
            "BREAKDOWN\tworkload=fib_pair\tprover={prover}\tlog_rows={log_rows}\trows={rows}\tphase={phase}{extra}\tkind={kind}"
        );
    };

    emit("p3_total", "prove_total_wall", format!("\tms={wall_ms:.3}"));

    let Some(pr) = snap.prove_root() else {
        eprintln!(
            "warning: P3 snapshot has no prove_* root ({} roots)",
            snap.roots.len()
        );
        return;
    };
    let prove_total_ms = to_ms(snap.total(pr));
    emit(
        "p3_total",
        "prove_total_traced",
        format!("\tms={prove_total_ms:.3}"),
    );

    // (1) normalized cross-prover phases
    for (name, d) in snap.canonical() {
        let v = to_ms(d);
        let pct = if prove_total_ms > 0.0 {
            100.0 * v / prove_total_ms
        } else {
            0.0
        };
        emit("norm", name, format!("\tms={v:.3}\tpct={pct:.1}"));
    }

    // (2) native P3 hierarchy (prove subtree only), pre-order with depth
    for &i in &snap.order {
        let n = &snap.nodes[i];
        if n.root != pr {
            continue; // skip the verify subtree
        }
        let self_ms = to_ms(n.self_time);
        let total_ms = to_ms(n.total_time);
        if self_ms < 0.3 && total_ms < 1.0 {
            continue;
        }
        emit(
            "p3_native",
            &sanitize_phase(&n.name),
            format!(
                "\tdepth={}\tms={self_ms:.3}\ttotal_ms={total_ms:.3}",
                n.depth
            ),
        );
    }

    // (3) self-time aggregated by name (prove subtree)
    for a in snap.aggregate_subtree(pr) {
        let self_ms = to_ms(a.self_time);
        if self_ms < 0.3 {
            continue;
        }
        emit(
            "p3_agg",
            &a.name,
            format!(
                "\tms={self_ms:.3}\ttotal_ms={:.3}\tcalls={}",
                to_ms(a.total_time),
                a.calls
            ),
        );
    }
}

/// Lowercase, collapse non-alphanumeric runs to `_`, prefix a leading digit
/// with `p`. Used to turn human span names ("commit to trace data") into
/// machine-parseable phase tokens ("commit_to_trace_data").
pub fn sanitize_phase(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut last_was_sep = true;
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() {
            for c in ch.to_lowercase() {
                if c.is_ascii_alphanumeric() {
                    out.push(c);
                    last_was_sep = false;
                }
            }
        } else if !last_was_sep {
            out.push('_');
            last_was_sep = true;
        }
    }
    while out.ends_with('_') {
        out.pop();
    }
    if out.is_empty() {
        return "phase".to_string();
    }
    if out.chars().next().unwrap().is_ascii_digit() {
        out.insert(0, 'p');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leaf(name: &str, total_ms: u64) -> Node {
        Node {
            name: name.to_string(),
            self_time: Duration::from_millis(total_ms),
            total_time: Duration::from_millis(total_ms),
            depth: 0,
            parent: None,
            children: Vec::new(),
            root: 0,
        }
    }

    /// Hand-built tree mirroring the P3 prove shape:
    ///   0 prove_with_preprocessed (100) -> 1,4,5,7
    ///   1 commit to trace data    (36)  -> 2,3
    ///   2 coset_lde_batch         (11)
    ///   3 build_merkle_tree       (24)
    ///   4 quotient_values         (6)
    ///   5 commit to quotient poly chunks (26) -> 6
    ///   6 build_merkle_tree       (20)
    ///   7 open                    (40)  -> 8
    ///   8 FRI prover              (30)
    fn sample() -> Snapshot {
        let mut nodes = vec![
            leaf("prove_with_preprocessed", 100),
            leaf("commit to trace data", 36),
            leaf("coset_lde_batch", 11),
            leaf("build_merkle_tree", 24),
            leaf("quotient_values", 6),
            leaf("commit to quotient poly chunks", 26),
            leaf("build_merkle_tree", 20),
            leaf("open", 40),
            leaf("FRI prover", 30),
        ];
        nodes[0].children = vec![1, 4, 5, 7];
        nodes[1].children = vec![2, 3];
        nodes[5].children = vec![6];
        nodes[7].children = vec![8];
        let mut order = Vec::new();
        preorder(0, &nodes, &mut order);
        Snapshot {
            nodes,
            roots: vec![0],
            order,
        }
    }

    #[test]
    fn sanitize_basic() {
        assert_eq!(
            sanitize_phase("commit to trace data"),
            "commit_to_trace_data"
        );
        assert_eq!(sanitize_phase("FRI prover"), "fri_prover");
        assert_eq!(sanitize_phase("commit phase"), "commit_phase");
        assert_eq!(sanitize_phase("idft final poly"), "idft_final_poly");
        assert_eq!(sanitize_phase(""), "phase");
        assert_eq!(sanitize_phase("123abc"), "p123abc");
    }

    #[test]
    fn preorder_is_root_first() {
        let s = sample();
        // First entry is the root, not a leaf.
        assert_eq!(s.order[0], 0);
        assert_eq!(s.nodes[s.order[0]].name, "prove_with_preprocessed");
        assert_eq!(s.order, vec![0, 1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn canonical_maps_p3_tree() {
        let s = sample();
        let c: HashMap<&str, u128> = s
            .canonical()
            .into_iter()
            .map(|(n, d)| (n, d.as_millis()))
            .collect();
        assert_eq!(c["norm_prove_total"], 100);
        assert_eq!(c["norm_trace_commit"], 36);
        assert_eq!(c["norm_trace_lde"], 11);
        assert_eq!(c["norm_trace_merkle"], 24);
        assert_eq!(c["norm_constraint_eval"], 6);
        assert_eq!(c["norm_quotient_commit"], 26);
        assert_eq!(c["norm_quotient_merkle"], 20);
        assert_eq!(c["norm_open"], 40);
        assert_eq!(c["norm_fri"], 30);
        assert_eq!(c["norm_deep_ood"], 10); // open(40) - fri(30)
    }

    #[test]
    fn aggregate_sums_duplicate_names() {
        let s = sample();
        let agg = s.aggregate_subtree(0);
        let mk = agg.iter().find(|a| a.name == "build_merkle_tree").unwrap();
        assert_eq!(mk.calls, 2); // trace + quotient merkle
        assert_eq!(mk.total_time.as_millis(), 44); // 24 + 20
    }
}
