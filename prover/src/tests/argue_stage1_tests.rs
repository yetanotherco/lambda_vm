//! D-ARGUE's stage 1 on the tables the argue really proves — the VM's
//! (Q1a, `thoughts/zf/gap2/fix2/D-ARGUE.md` §4.1): the base-field
//! precondition, parity of the host reference (`multilinear::fused`) with
//! today's rounds, its negative controls, a census of what each technique
//! works on, and the op report that sizes the stage.
//!
//! Today's side of every comparison is the code the prover runs:
//! `batch::prove` over the rules `multilinear_table::prove` builds —
//! `IrShape::program` for the constraint, `logup::claim_statements` for the
//! bus. The laptop runs the reference on random columns, with the grid's
//! corners computed: random columns break every AIR. The corner skip needs a
//! trace that satisfies its AIR, and getting one means proving a program, so
//! that half is `#[ignore]`d and the box runs it
//! ([`the_stage1_reference_on_real_traces`]).
//!
//! ```text
//! cargo test -p lambda-vm-prover --lib tests::argue_stage1_tests
//! cargo test -p lambda-vm-prover --lib tests::argue_stage1_tests::the_stage1 -- --ignored --nocapture
//! ```

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::fiat_shamir::is_transcript::IsTranscript;
use math::field::element::FieldElement;
use multilinear::batch::{self, Rule};
use multilinear::claim_reduce;
use multilinear::constraint_argument::FactorKind;
use multilinear::eq::eq_mle;
use multilinear::fused::{
    self, Constraints, Counters, Mix, Options, Proved, Shape, Statement, Variant,
};
use multilinear::logup::{self, Interaction};
use multilinear::mle::Mle;
use multilinear::sumcheck::SumcheckProof;
use stark::constraint_ir::ir::Op as IrOp;
use stark::multilinear_air::{IrShape, Uniforms, beta_powers, live_nodes};
use stark::multilinear_table::{TableLayout, weight_slots};
use stark::traits::AIR;

use super::lean_program_census::{VmAir, vm_airs};
use crate::tables::types::{GoldilocksExtension, GoldilocksField};

type Gl = GoldilocksField;
type Ext3 = GoldilocksExtension;
type EE = FieldElement<Ext3>;
type FB = FieldElement<Gl>;
type DynAir<'a> = &'a dyn AIR<Field = Gl, FieldExtension = Ext3, PublicInputs = ()>;

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(0x9E37_79B9_7F4A_7C15 ^ seed.wrapping_mul(0x2545_F491_4F6C_DD1D) | 1)
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn base(&mut self) -> FB {
        FB::from(self.next())
    }

    fn ext(&mut self) -> EE {
        EE::new([self.base(), self.base(), self.base()])
    }

    fn exts(&mut self, n: usize) -> Vec<EE> {
        (0..n).map(|_| self.ext()).collect()
    }
}

/// Every table the argue proves, by name.
fn every_air<'a>(vm: &'a [(VmAir, &'static str)]) -> Vec<(DynAir<'a>, String)> {
    vm.iter()
        .map(|(air, label)| (&**air as DynAir<'a>, label.to_string()))
        .collect()
}

/// A table's argue, structure only: what `multilinear_table::prove` builds
/// before its batch — the shape, the factor kinds, the bus at stand-in
/// challenges `z` and `alpha`, the batching powers of a stand-in `beta` — and
/// the constraint part brought down to the base field.
struct Table {
    label: String,
    shape: IrShape<Gl, Ext3>,
    kinds: Vec<FactorKind>,
    num_columns: usize,
    constraints: Constraints<Gl>,
    interactions: Vec<Interaction<Ext3>>,
    betas: Vec<EE>,
}

impl Table {
    fn from_air(air: DynAir<'_>, label: &str, num_vars: usize, rng: &mut Rng) -> Self {
        let layout = TableLayout::<Gl, Ext3>::new(
            air.constraint_program(),
            air.constraints_meta(),
            air.bus_interactions(),
            air.trace_layout().0,
            num_vars,
            Uniforms::default(),
        )
        .unwrap_or_else(|e| panic!("{label}: {e:?}"));
        let slots = layout.slot_of().to_vec();
        let interactions = stark::multilinear_logup::interactions(
            air.bus_interactions(),
            slots.len(),
            &rng.ext(),
            &rng.ext(),
            |col| {
                slots
                    .get(col)
                    .copied()
                    .ok_or(multilinear::Error::UnknownPolynomial {
                        index: col,
                        len: slots.len(),
                    })
            },
        )
        .unwrap_or_else(|e| panic!("{label}: {e:?}"));
        let beta = rng.ext();
        Self::new(
            label,
            layout.shape().clone(),
            layout.kinds().to_vec(),
            layout.num_columns(),
            interactions,
            &beta,
        )
    }

    fn new(
        label: &str,
        shape: IrShape<Gl, Ext3>,
        kinds: Vec<FactorKind>,
        num_columns: usize,
        interactions: Vec<Interaction<Ext3>>,
        beta: &EE,
    ) -> Self {
        let constraints = Constraints::<Gl>::from_extension(
            &shape.steps_as_ops(),
            shape.root_steps(),
            shape.selector_of_root(),
            shape.degree(),
        )
        .unwrap_or_else(|e| panic!("{label}: the base-field precondition fails: {e:?}"));
        let betas = beta_powers(beta, shape.num_roots());
        Self {
            label: label.to_string(),
            shape,
            kinds,
            num_columns,
            constraints,
            interactions,
            betas,
        }
    }

    fn width(&self) -> usize {
        self.kinds.len()
    }

    /// The factors from committed columns: each view at its offset, and each
    /// public selector's table — all base-field, in slot order.
    fn factors_from(&self, columns: &[Mle<Gl>]) -> Vec<Mle<Gl>> {
        let num_vars = columns[0].num_vars();
        let mut selectors = self.shape.public_selectors().iter();
        self.kinds
            .iter()
            .map(|kind| match kind {
                FactorKind::Committed(source) => claim_reduce::materialize(columns, source)
                    .unwrap_or_else(|e| panic!("{}: {e:?}", self.label)),
                FactorKind::Public => selectors
                    .next()
                    .expect("a selector per public factor")
                    .table::<Gl>(num_vars)
                    .unwrap_or_else(|e| panic!("{}: {e:?}", self.label)),
            })
            .collect()
    }

    fn random_columns(&self, num_vars: usize, rng: &mut Rng) -> Vec<Mle<Gl>> {
        (0..self.num_columns)
            .map(|_| Mle::new((0..1usize << num_vars).map(|_| rng.base()).collect()).unwrap())
            .collect()
    }

    /// A session over `columns` at a random zerocheck point and GKR claim
    /// point.
    fn session(&self, columns: Vec<Mle<Gl>>, rng: &mut Rng) -> Session<'_> {
        let num_vars = columns[0].num_vars();
        let r = rng.exts(num_vars);
        let claim_point = rng.exts(logup::input_layer_vars(self.interactions.len(), num_vars));
        Session {
            table: self,
            factors: self.factors_from(&columns),
            columns,
            r,
            claim_point,
        }
    }
}

/// One table's batch over one set of columns and points.
struct Session<'t> {
    table: &'t Table,
    columns: Vec<Mle<Gl>>,
    factors: Vec<Mle<Gl>>,
    r: Vec<EE>,
    claim_point: Vec<EE>,
}

impl Session<'_> {
    fn num_vars(&self) -> usize {
        self.factors[0].num_vars()
    }

    fn statement(&self) -> Statement<'_, Gl, Ext3> {
        Statement {
            constraints: &self.table.constraints,
            betas: &self.table.betas,
            interactions: &self.table.interactions,
            claim_point: &self.claim_point,
            r: &self.r,
        }
    }

    fn transcript(&self) -> DefaultTranscript<Ext3> {
        DefaultTranscript::<Ext3>::new(self.table.label.as_bytes())
    }

    /// Today's three rules over today's factor list, as
    /// `multilinear_table::prove` builds them: the trace's factors lifted,
    /// `eq(r, ·)`, `eq(ρ, ·)`.
    fn todays(&self) -> (Vec<Mle<Ext3>>, Vec<Rule<'_, Ext3>>) {
        let table = self.table;
        let (weight_r, weight_z) = weight_slots(table.width());
        let zerocheck = Rule::compiled(
            table.shape.degree() + 1,
            table
                .shape
                .program(&table.betas, weight_r)
                .unwrap_or_else(|e| panic!("{}: {e:?}", table.label)),
        );
        let bus = logup::claim_statements(
            &table.interactions,
            &self.claim_point,
            self.num_vars(),
            weight_z,
        )
        .unwrap_or_else(|e| panic!("{}: {e:?}", table.label));
        let mut polys: Vec<Mle<Ext3>> = self
            .factors
            .iter()
            .map(|f| {
                Mle::new(f.evals().iter().map(|v| v.to_extension::<Ext3>()).collect()).unwrap()
            })
            .collect();
        polys.push(eq_mle(&self.r).unwrap());
        polys.push(eq_mle(&bus.row_point).unwrap());
        (polys, vec![zerocheck, bus.numerator, bus.denominator])
    }

    /// Each rule's sum over the cube: the claims an honest batch carries. On
    /// a trace that satisfies its AIR the first is zero.
    fn true_claims(&self) -> [EE; 3] {
        let (polys, rules) = self.todays();
        let mut scratch = Vec::new();
        core::array::from_fn(|k| {
            (0..polys[0].len()).fold(EE::zero(), |acc, i| {
                let values: Vec<EE> = polys.iter().map(|p| p.evals()[i]).collect();
                acc + rules[k].apply_in(&values, &mut scratch)
            })
        })
    }

    fn today(&self, claims: &[EE; 3]) -> (SumcheckProof<Ext3>, Vec<EE>, DefaultTranscript<Ext3>) {
        let (polys, rules) = self.todays();
        let mut t = self.transcript();
        let (proof, point) = batch::prove(polys, rules, claims, &mut t)
            .unwrap_or_else(|e| panic!("{}: today's batch: {e:?}", self.table.label));
        (proof, point, t)
    }

    fn fused(
        &self,
        claims: &[EE; 3],
        options: Options,
    ) -> (
        Result<Proved<Ext3>, multilinear::Error>,
        DefaultTranscript<Ext3>,
        Counters,
    ) {
        let mut t = self.transcript();
        let mut counters = Counters::default();
        let proved = fused::prove(
            &self.statement(),
            &self.factors,
            claims,
            options,
            &mut t,
            &mut counters,
        );
        (proved, t, counters)
    }

    /// The batching weights `fused::prove` draws after `claims`.
    fn lambdas(&self, claims: &[EE; 3]) -> Vec<EE> {
        let mut t = self.transcript();
        for claim in claims {
            t.append_field_element(claim);
        }
        multilinear::challenge_powers(&t.sample_field_element(), 3)
    }

    /// The first round `options` sends another message at than today's
    /// rounds, or `None` — then with the point, the transcript after the
    /// rounds, the next challenge, every factor's value at the point and the
    /// count against the model all checked too.
    fn parts_at(&self, claims: &[EE; 3], options: Options) -> Option<usize> {
        let label = &self.table.label;
        let n = self.num_vars();
        let (proof, point, mut t_today) = self.today(claims);
        let (proved, mut t, counters) = self.fused(claims, options);
        let (fused, fused_point, bound) =
            proved.unwrap_or_else(|e| panic!("{label} n={n} {options:?}: {e:?}"));
        if let Some(round) = proof
            .rounds
            .iter()
            .zip(&fused.rounds)
            .position(|(a, b)| a.evaluations != b.evaluations)
        {
            return Some(round);
        }
        assert_eq!(fused.rounds.len(), n, "{label} n={n} {options:?}");
        assert_eq!(fused_point, point, "{label} n={n} {options:?}: the point");
        assert_eq!(
            t.state(),
            t_today.state(),
            "{label} n={n} {options:?}: the transcript"
        );
        assert_eq!(
            t.sample_field_element(),
            t_today.sample_field_element(),
            "{label} n={n} {options:?}: the next challenge"
        );
        for (slot, (factor, value)) in self.factors.iter().zip(&bound).enumerate() {
            assert_eq!(
                *value,
                factor.evaluate_in(&point).unwrap(),
                "{label} n={n} {options:?}: factor {slot} at the point"
            );
        }
        let shape = Shape::of(
            &self.statement(),
            self.table.width(),
            &self.lambdas(claims),
            options.on_demand,
        )
        .unwrap_or_else(|e| panic!("{label}: {e:?}"));
        assert_eq!(
            fused::model(&shape, options),
            counters,
            "{label} n={n} {options:?}: the model is not what the reference counted"
        );
        None
    }
}

/// Every variant, corners computed: what random columns allow.
fn on_random_columns() -> Vec<Options> {
    let mut all = Vec::new();
    for variant in [Variant::Today, Variant::Bus] {
        for on_demand in [false, true] {
            all.push(Options {
                on_demand,
                ..Options::new(variant)
            });
        }
    }
    all.push(Options::new(Variant::Gruen));
    all.push(Options::new(Variant::Grid));
    all
}

fn skipped() -> Options {
    Options {
        skip_corners: true,
        ..Options::new(Variant::Grid)
    }
}

fn checked() -> Options {
    Options {
        check_corners: true,
        ..skipped()
    }
}

/// The tables Q1 is sized on: the widest precompile and the VM's tallest
/// table.
const THE_Q1_TABLES: [&str; 2] = ["KECCAK_RND", "CPU"];

// ── the base-field precondition ──────────────────────────────────────────────

/// What in a table's constraint part stops it running in the base field: a
/// live extension constant, a live uniform (a challenge, an alpha power, the
/// table offset), or a constant of the compiled DAG with an extension part.
fn not_base(air: DynAir<'_>) -> Vec<String> {
    let program = air.constraint_program();
    let roots = program.roots[..program.num_base].to_vec();
    let live = live_nodes(program, &roots);
    let mut found = Vec::new();
    for (id, op) in program.nodes.iter().enumerate() {
        if !live[id] {
            continue;
        }
        match op {
            IrOp::ConstExt(idx) => {
                let value = &program.ext_consts[*idx as usize];
                let limbs = (*value).to_subfield_vec::<Gl>();
                if limbs[1..].iter().any(|limb| *limb != FB::zero()) {
                    found.push(format!("node {id}: an extension constant"));
                }
            }
            IrOp::RapChallenge { .. } => found.push(format!("node {id}: a RAP challenge")),
            IrOp::AlphaPow { .. } => found.push(format!("node {id}: an alpha power")),
            IrOp::TableOffset => found.push(format!("node {id}: the table offset")),
            _ => {}
        }
    }
    found
}

/// ★ Every table's constraint part runs in the base field: no live extension
/// constant or uniform, and the compiled DAG comes down to the base field —
/// the precondition S1-5 rests on (D-ARGUE §2.6). An exception would list its
/// table here, and its roots would run in the extension.
#[test]
fn every_table_meets_the_base_field_precondition() {
    let vm = vm_airs();
    let mut exceptions = Vec::new();
    for (air, label) in every_air(&vm) {
        for reason in not_base(air) {
            exceptions.push(format!("{label}: {reason}"));
        }
        let layout = TableLayout::<Gl, Ext3>::new(
            air.constraint_program(),
            air.constraints_meta(),
            air.bus_interactions(),
            air.trace_layout().0,
            4,
            Uniforms::default(),
        )
        .unwrap_or_else(|e| panic!("{label}: {e:?}"));
        let shape = layout.shape();
        if let Err(e) = Constraints::<Gl>::from_extension(
            &shape.steps_as_ops(),
            shape.root_steps(),
            shape.selector_of_root(),
            shape.degree(),
        ) {
            exceptions.push(format!("{label}: {e}"));
        }
    }
    assert!(exceptions.is_empty(), "not base-field: {exceptions:#?}");
}

// ── parity on random columns ─────────────────────────────────────────────────

/// ★ (a) Every variant of the reference sends today's messages — rounds,
/// point, transcript, next challenge, every factor's value at the point — on
/// every VM table, over random columns with the corners
/// computed, at heights 1, 2, 3 and 5; and counts what the model says.
#[test]
fn stage1_sends_todays_messages_on_every_table() {
    let vm = vm_airs();
    let airs = every_air(&vm);
    assert_eq!(airs.len(), 26, "26 VM tables");
    for (at, (air, label)) in airs.iter().enumerate() {
        for n in [1usize, 2, 3, 5] {
            let mut rng = Rng::new((at * 100 + n) as u64);
            let table = Table::from_air(*air, label, n, &mut rng);
            let session = table.session(table.random_columns(n, &mut rng), &mut rng);
            let claims = session.true_claims();
            for options in on_random_columns() {
                assert_eq!(
                    session.parts_at(&claims, options),
                    None,
                    "{label} n={n}: {options:?}"
                );
            }
        }
    }
}

/// ★ (a) The same on the Q1 tables at 2^8 and 2^10 rows, and CPU at 2^12.
/// Heights up to 2^14 run on the box ([`stage1_sends_todays_messages_at_the_top_heights`]).
#[test]
fn stage1_sends_todays_messages_on_the_q1_tables() {
    run_the_q1_tables(&[
        ("KECCAK_RND", 8),
        ("KECCAK_RND", 10),
        ("CPU", 8),
        ("CPU", 10),
        ("CPU", 12),
    ]);
}

/// The Q1 tables at 2^12 and 2^14 rows — minutes on a laptop core, so the box
/// runs it (`--release`).
#[test]
#[ignore = "the Q1 tables at 2^12 and 2^14 rows: a box run"]
fn stage1_sends_todays_messages_at_the_top_heights() {
    run_the_q1_tables(&[("KECCAK_RND", 12), ("KECCAK_RND", 14), ("CPU", 14)]);
}

fn run_the_q1_tables(heights: &[(&str, usize)]) {
    let vm = vm_airs();
    let airs = every_air(&vm);
    for &(name, n) in heights {
        let (air, label) = airs
            .iter()
            .find(|(_, label)| label == name)
            .unwrap_or_else(|| panic!("no table named {name}"));
        let mut rng = Rng::new(0xA11CE + n as u64);
        let table = Table::from_air(*air, label, n, &mut rng);
        let session = table.session(table.random_columns(n, &mut rng), &mut rng);
        let claims = session.true_claims();
        for options in on_random_columns() {
            assert_eq!(
                session.parts_at(&claims, options),
                None,
                "{label} n={n}: {options:?}"
            );
        }
    }
}

// ── negative controls on random columns ──────────────────────────────────────

/// ⛔ (c) One coefficient of the bus column wrong — its first, its last — and
/// every variant that reads the column parts from today at round 0, on each of
/// the Q1 tables.
#[test]
fn a_wrong_bus_coefficient_parts_the_messages_on_the_q1_tables() {
    let vm = vm_airs();
    let airs = every_air(&vm);
    for name in THE_Q1_TABLES {
        let (air, label) = airs.iter().find(|(_, label)| label == name).unwrap();
        let mut rng = Rng::new(0xFA17);
        let n = 6;
        let table = Table::from_air(*air, label, n, &mut rng);
        let session = table.session(table.random_columns(n, &mut rng), &mut rng);
        let claims = session.true_claims();
        let terms = fused::BusColumn::new(
            &table.interactions,
            &session.claim_point[..session.claim_point.len() - n],
            &EE::one(),
            &EE::one(),
        )
        .unwrap()
        .terms()
        .len();
        for fault in [0, terms - 1] {
            for variant in [Variant::Bus, Variant::Gruen, Variant::Grid] {
                let options = Options {
                    bus_fault: Some(fault),
                    ..Options::new(variant)
                };
                assert_eq!(
                    session.parts_at(&claims, options),
                    Some(0),
                    "{label}: {options:?}"
                );
            }
        }
    }
}

/// ⛔ The corner skip on columns that break the AIR parts from today at round
/// 0 on each of the Q1 tables, and the corner check refuses at the first row.
#[test]
fn the_corner_skip_parts_on_random_columns_of_the_q1_tables() {
    let vm = vm_airs();
    let airs = every_air(&vm);
    for name in THE_Q1_TABLES {
        let (air, label) = airs.iter().find(|(_, label)| label == name).unwrap();
        let mut rng = Rng::new(0xC0DE);
        let n = 5;
        let table = Table::from_air(*air, label, n, &mut rng);
        let session = table.session(table.random_columns(n, &mut rng), &mut rng);
        let claims = session.true_claims();
        assert_eq!(session.parts_at(&claims, skipped()), Some(0), "{label}");
        assert_eq!(
            session.fused(&claims, checked()).0.unwrap_err(),
            multilinear::Error::ConstraintViolated { row: 0 },
            "{label}"
        );
    }
}

// ── the real-trace half (box) ────────────────────────────────────────────────

/// ★ (b) + (c), on traces that satisfy their AIRs: KECCAK_RND and CPU captured
/// from a WHIR proof of `test_keccak` (`multilinear_table::argue_capture`).
///
/// Each captured table, from a fresh transcript at its own GKR claim and
/// zerocheck point: today's claims `(0, p̃, q̃)` are the batch's true sums (the
/// trace satisfies its AIR and the GKR claim is the input layer's); every
/// variant sends today's messages, the corners skipped and then checked. Then
/// the controls: a cell the constraints read flipped — the corner check names
/// its row and the corner skip parts from today by round 1; and a wrong bus
/// coefficient, parting at round 0.
///
/// A proving test, so not on the laptop:
///
/// ```text
/// cargo test --release -p lambda-vm-prover --lib \
///     tests::argue_stage1_tests::the_stage1_reference_on_real_traces -- --ignored --exact --nocapture
/// ```
#[test]
#[ignore = "proves a program to capture real traces: a box run"]
fn the_stage1_reference_on_real_traces() {
    use crate::tables::MaxRowsConfig;
    use stark::multilinear_table::argue_capture::{self, Captured};
    use stark::proof::options::ProofOptions;

    let options = ProofOptions::default_test_options();
    let vm = vm_airs();
    let named: Vec<(DynAir<'_>, String)> = every_air(&vm)
        .into_iter()
        .filter(|(_, label)| THE_Q1_TABLES.contains(&label.as_str()))
        .collect();
    let width = |air: DynAir<'_>| {
        TableLayout::<Gl, Ext3>::new(
            air.constraint_program(),
            air.constraints_meta(),
            air.bus_interactions(),
            air.trace_layout().0,
            4,
            Uniforms::default(),
        )
        .unwrap()
        .num_columns()
    };
    let widths: Vec<usize> = named.iter().map(|(air, _)| width(*air)).collect();
    argue_capture::arm(&widths);
    let elf = crate::test_utils::asm_elf_bytes("test_keccak");
    crate::multilinear_prove::prove_with_options(&elf, &options, &MaxRowsConfig::default())
        .expect("test_keccak proves");
    argue_capture::arm(&[]);

    let mut seen: Vec<String> = Vec::new();
    for captured in argue_capture::take() {
        let captured = captured
            .downcast::<Captured<Gl, Ext3>>()
            .expect("a Goldilocks table");
        let num_vars = captured.columns[0].num_vars();
        // Which of the Q1 tables: the same layout, the same constraint DAG.
        let dag = Mix::of(&captured.shape.steps_as_ops());
        let Some((_, label)) = named.iter().find(|(air, label)| {
            let mut rng = Rng::new(1);
            let candidate = Table::from_air(*air, label, num_vars, &mut rng);
            candidate.kinds == captured.kinds
                && candidate.num_columns == captured.columns.len()
                && candidate.interactions.len() == captured.interactions.len()
                && Mix::of(&candidate.shape.steps_as_ops()) == dag
        }) else {
            continue;
        };
        let table = Table::new(
            label,
            captured.shape.clone(),
            captured.kinds.clone(),
            captured.columns.len(),
            captured.interactions.clone(),
            &captured.beta,
        );
        let session = Session {
            table: &table,
            factors: table.factors_from(&captured.columns),
            columns: captured.columns.clone(),
            r: captured.r.clone(),
            claim_point: captured.claim.point.clone(),
        };
        let claims = [EE::zero(), captured.claim.p, captured.claim.q];
        assert_eq!(
            session.true_claims(),
            claims,
            "{label} 2^{num_vars}: the captured claims are the batch's sums"
        );
        let mut every = on_random_columns();
        every.push(skipped());
        every.push(checked());
        for options in every {
            assert_eq!(
                session.parts_at(&claims, options),
                None,
                "{label} 2^{num_vars}: {options:?}"
            );
        }

        // ⛔ A flipped cell of a column the constraints read: the first that
        // breaks the AIR, which the corner check refuses.
        let mut flipped = None;
        'cells: for &slot in &table.constraints.reads() {
            let FactorKind::Committed(source) = table.kinds[slot] else {
                continue;
            };
            for row in [0usize, 1, 5, (1 << num_vars) / 2 + 3] {
                let mut columns = session.columns.clone();
                let mut values = columns[source.column].evals().to_vec();
                let row = row % values.len();
                values[row] += FB::one();
                columns[source.column] = Mle::new(values).unwrap();
                let broken = Session {
                    table: &table,
                    factors: table.factors_from(&columns),
                    columns,
                    r: session.r.clone(),
                    claim_point: session.claim_point.clone(),
                };
                if let Err(multilinear::Error::ConstraintViolated { row: at }) =
                    broken.fused(&claims, checked()).0
                {
                    flipped = Some((broken, source.column, at));
                    break 'cells;
                }
            }
        }
        let (broken, column, at) =
            flipped.unwrap_or_else(|| panic!("{label}: no flipped cell broke the AIR"));
        let mut claims_broken = broken.true_claims();
        claims_broken[0] = EE::zero();
        let parted = broken.parts_at(&claims_broken, skipped());
        assert!(
            matches!(parted, Some(0) | Some(1)),
            "{label}: a flipped cell of column {column} (the check refused row {at}) parted at {parted:?}"
        );
        eprintln!(
            "argue stage 1: {label} 2^{num_vars}: {} variants send today's messages; a flipped cell of \
             column {column} refused at row {at}, parted at round {}; a wrong bus coefficient parted at round 0",
            on_random_columns().len() + 2,
            parted.unwrap_or(usize::MAX)
        );

        // ⛔ A wrong bus coefficient.
        let faulty = Options {
            bus_fault: Some(0),
            ..skipped()
        };
        assert_eq!(session.parts_at(&claims, faulty), Some(0), "{label}");
        seen.push(label.clone());
    }
    seen.sort();
    seen.dedup();
    assert_eq!(
        seen,
        ["CPU", "KECCAK_RND"],
        "every one of the Q1 tables captured"
    );
}

// ── Q1b: the card (`multilinear::gpu_fused`) ─────────────────────────────────

/// ★ The blob the fused kernels walk is the constraint part: on every VM table,
/// the lowered program with its `ACC` steps, walked as the
/// grid kernel walks it, gives today's zerocheck rule (weight one) at random
/// rows. Host only — this is what makes the card's walk checkable without one.
#[test]
fn the_fused_blob_is_the_constraint_part_on_every_table() {
    use multilinear::gpu_fused::{lower_fused, run_lowered_host};
    let vm = vm_airs();
    for (at, (air, label)) in every_air(&vm).into_iter().enumerate() {
        let mut rng = Rng::new(0xB10B + at as u64);
        let table = Table::from_air(air, &label, 4, &mut rng);
        let lowered =
            lower_fused(&table.constraints).unwrap_or_else(|| panic!("{label}: no lowering"));
        let width = table.width();
        let rule = Rule::compiled(
            table.constraints.degree() + 1,
            table
                .constraints
                .zerocheck_program(&table.betas, width)
                .unwrap(),
        );
        for row in 0..3 {
            let values: Vec<FB> = (0..width).map(|_| rng.base()).collect();
            let mut lifted: Vec<EE> = values.iter().map(|v| v.to_extension::<Ext3>()).collect();
            lifted.push(EE::one());
            assert_eq!(
                run_lowered_host(&lowered, &table.betas, &values),
                rule.apply(&lifted),
                "{label}: row {row} ({} slots)",
                lowered.num_slots
            );
        }
    }
}

/// Card tests share process-global switches; this puts them back however a
/// test ends.
#[cfg(feature = "cuda")]
struct CardSwitches;

#[cfg(feature = "cuda")]
impl CardSwitches {
    fn set(faults: multilinear::gpu_fused::FusedFaults, xcheck: bool) -> Self {
        use multilinear::gpu_fused::*;
        // A mutation run (the box's) makes one check inert from outside, to
        // show the test that relies on it fails without it.
        let mut faults = faults;
        match std::env::var("LAMBDA_VM_ARGUE_FUSED_MUTATE").as_deref() {
            Ok("xcheck") => faults.xcheck_inert = true,
            Ok("corners") => faults.corners_inert = true,
            _ => {}
        }
        force_fused_faults(faults);
        force_argue_fused(Some(true));
        force_argue_fused_xcheck(Some(xcheck));
        Self
    }
}

#[cfg(feature = "cuda")]
impl Drop for CardSwitches {
    fn drop(&mut self) {
        use multilinear::gpu_fused::*;
        force_fused_faults(FusedFaults::default());
        force_argue_fused(None);
        force_argue_fused_xcheck(None);
        math_cuda::sumcheck::force_int_nodes(None);
    }
}

#[cfg(feature = "cuda")]
type OnCard = Result<(SumcheckProof<Ext3>, Vec<EE>, Vec<EE>), multilinear::Error>;

#[cfg(feature = "cuda")]
impl Session<'_> {
    /// The batch through the prover's entry (`batch::prove_resident_with`)
    /// with the factors on the card — the fused rounds, when they are on.
    fn on_the_card(&self, claims: &[EE; 3]) -> OnCard {
        use multilinear::batch::Weight;
        let table = self.table;
        let n = self.num_vars();
        let public: Vec<Mle<Ext3>> = table
            .shape
            .public_selectors()
            .iter()
            .map(|s| s.table::<Ext3>(n).unwrap())
            .collect();
        let device = multilinear::gpu::upload_factors_from_columns(
            &self.columns,
            &table.kinds,
            &public,
            None,
        )
        .unwrap_or_else(|| panic!("{}: the card takes the factors", table.label));
        let (polys, rules) = self.todays();
        let lifted: Vec<Mle<Ext3>> = polys[..table.width()].to_vec();
        let rho = self.claim_point[self.claim_point.len() - n..].to_vec();
        let mut t = self.transcript();
        batch::prove_resident_with(
            vec![Weight::Eq(self.r.clone()), Weight::Eq(rho)],
            Some(std::sync::Arc::new(device)),
            || Ok(lifted.clone()),
            rules,
            claims,
            Some(multilinear::gpu_fused::FusedInput {
                constraints: &table.constraints,
                betas: &table.betas,
                interactions: &table.interactions,
                claim_point: &self.claim_point,
                r: &self.r,
                columns: None,
                second: None,
            }),
            &mut t,
        )
    }

    /// The card's proof against today's host rounds: messages, point, and
    /// every trace factor's value at the point.
    fn assert_card_is_today(&self, claims: &[EE; 3], card: OnCard, what: &str) {
        let label = &self.table.label;
        let n = self.num_vars();
        let (proof, point, bound) = card.unwrap_or_else(|e| panic!("{label} n={n} {what}: {e:?}"));
        let (today, today_point, _) = self.today(claims);
        for (round, (a, b)) in today.rounds.iter().zip(&proof.rounds).enumerate() {
            assert_eq!(
                a.evaluations, b.evaluations,
                "{label} n={n} {what}: round {round}"
            );
        }
        assert_eq!(
            proof.rounds.len(),
            today.rounds.len(),
            "{label} n={n} {what}"
        );
        assert_eq!(point, today_point, "{label} n={n} {what}: the point");
        for (slot, factor) in self.factors.iter().enumerate() {
            assert_eq!(
                bound[slot],
                factor.evaluate_in(&point).unwrap(),
                "{label} n={n} {what}: factor {slot} at the point"
            );
        }
    }
}

/// ★ Q1b device parity (box): the fused rounds on the card send today's
/// messages — the Q1 tables at 2^10 and 2^12, every table at 2^8 or the height
/// the card's upload takes it at — over random
/// columns, the corners summed (random columns break every AIR), with the
/// cross-check on: today's device rounds replay the fused challenges and must
/// agree. Each run must have taken the card (`fused_sessions`) and been
/// confirmed (`fused_xchecks`).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "flips process-global argue switches: run alone, --ignored --test-threads=1"]
fn the_fused_rounds_on_the_card_send_todays_messages() {
    use multilinear::gpu_fused::*;
    let _switches = CardSwitches::set(
        FusedFaults {
            keep_corners: true,
            ..FusedFaults::default()
        },
        true,
    );
    let vm = vm_airs();
    // Each table at 2^8, or as far above as the card's upload takes it: it
    // declines a table under 4096 cells (`gpu::worth_the_device`), which the
    // prover then argues on the host — EQ's 12 factors need 2^9, a two-factor
    // chip 2^11.
    let mut runs: Vec<(String, usize)> = every_air(&vm)
        .iter()
        .map(|(air, label)| {
            let mut rng = Rng::new(0);
            let width = Table::from_air(*air, label, 8, &mut rng).width();
            let n = (8..).find(|n| width << n >= 4096).expect("a height");
            (label.clone(), n)
        })
        .collect();
    for label in THE_Q1_TABLES {
        runs.push((label.to_string(), 10));
        runs.push((label.to_string(), 12));
    }
    let airs = every_air(&vm);
    for (at, (label, n)) in runs.iter().enumerate() {
        let air = airs.iter().find(|(_, l)| l == label).unwrap().0;
        let mut rng = Rng::new(0xCA4D + at as u64);
        let table = Table::from_air(air, label, *n, &mut rng);
        let session = table.session(table.random_columns(*n, &mut rng), &mut rng);
        let claims = session.true_claims();
        let (sessions, checks) = (fused_sessions(), fused_xchecks());
        session.assert_card_is_today(&claims, session.on_the_card(&claims), "fused");
        assert_eq!(
            fused_sessions(),
            sessions + 1,
            "{label} n={n}: the card ran the fused rounds"
        );
        assert_eq!(
            fused_xchecks(),
            checks + 1,
            "{label} n={n}: today's rounds confirmed them"
        );
    }
    eprintln!(
        "argue Q1b: {} fused card runs sent today's messages",
        runs.len()
    );
}

/// ★ The bus column from rounds 0–1's rows (`argue_fused::bus_rows`, on by
/// default; `LAMBDA_VM_ZC_BUS_FROM_ROWS=0` is the folded factors' column) sends
/// today's messages: on every table at 2^8 (or the height the card takes it
/// at) and on the Q1 tables at 2^10 and 2^12, with it on and off, the card's
/// rounds at the crossover and then down to two cells, against today's host
/// rounds — the same rounds, point and bound factors. Then the rows folded by
/// one challenge only must part from today, so the comparison is load-bearing.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "flips process-global argue switches: run alone, --ignored --test-threads=1"]
fn the_bus_column_from_rows_sends_todays_messages() {
    use math_cuda::argue_fused::bus_rows;
    use multilinear::gpu_fused::*;
    struct Back;
    impl Drop for Back {
        fn drop(&mut self) {
            bus_rows::force(None);
            bus_rows::force_fault(false);
            force_zc_device_tail(None);
        }
    }
    let _back = Back;
    let _switches = CardSwitches::set(
        FusedFaults {
            keep_corners: true,
            ..FusedFaults::default()
        },
        false,
    );
    let vm = vm_airs();
    let airs = every_air(&vm);
    let mut runs: Vec<(String, usize)> = airs
        .iter()
        .map(|(air, label)| {
            let mut rng = Rng::new(0);
            let width = Table::from_air(*air, label, 8, &mut rng).width();
            let n = (8..).find(|n| width << n >= 4096).expect("a height");
            (label.clone(), n)
        })
        .collect();
    for label in THE_Q1_TABLES {
        runs.push((label.to_string(), 10));
        runs.push((label.to_string(), 12));
    }
    let mut checked = 0usize;
    for tail in [None, Some(Some(2usize))] {
        force_zc_device_tail(tail);
        for on in [true, false] {
            bus_rows::force(Some(on));
            for (at, (label, n)) in runs.iter().enumerate() {
                let air = airs.iter().find(|(_, l)| l == label).unwrap().0;
                let mut rng = Rng::new(0xB05 + at as u64);
                let table = Table::from_air(air, label, *n, &mut rng);
                let session = table.session(table.random_columns(*n, &mut rng), &mut rng);
                let claims = session.true_claims();
                let sessions = fused_sessions();
                let what = format!("bus from rows {on}, tail {tail:?}");
                session.assert_card_is_today(&claims, session.on_the_card(&claims), &what);
                assert_eq!(
                    fused_sessions(),
                    sessions + 1,
                    "{label} n={n} {what}: the card ran the fused rounds"
                );
                checked += 1;
            }
        }
    }
    bus_rows::force(Some(true));
    bus_rows::force_fault(true);
    for (at, label) in THE_Q1_TABLES.iter().enumerate() {
        let n = 10;
        let air = airs.iter().find(|(_, l)| l == label).unwrap().0;
        let mut rng = Rng::new(0xFA58 + at as u64);
        let table = Table::from_air(air, label, n, &mut rng);
        let session = table.session(table.random_columns(n, &mut rng), &mut rng);
        let claims = session.true_claims();
        let (today, today_point, _) = session.today(&claims);
        let parted = match session.on_the_card(&claims) {
            Err(_) => true,
            Ok((proof, point, _)) => {
                proof
                    .rounds
                    .iter()
                    .zip(&today.rounds)
                    .any(|(a, b)| a.evaluations != b.evaluations)
                    || point != today_point
            }
        };
        assert!(
            parted,
            "{label}: the faulted bus column must part from today's rounds"
        );
    }
    eprintln!("argue bus from rows: {checked} card runs sent today's messages; the fault parted");
}

/// ★ The fraction tree folded two levels a pass (`gkr::two_level_folds`, on by
/// default; `LAMBDA_VM_GKR_TWO_LEVEL_FOLDS=0` folds one level a pass) builds
/// the same levels: over random input layers of 2^1 to 2^12 fractions, stored
/// whole and padded (the first fold reading the padding as `0/1`), every level
/// above the input is bit for bit the one-level build's. Then the two-level
/// pass's upper cells off by one must part, so the comparison is load-bearing.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "flips process-global argue switches: run alone, --ignored --test-threads=1"]
fn the_two_level_tree_folds_build_todays_levels() {
    use math_cuda::gkr::{DeviceFractionTree, two_level_folds};
    struct Back;
    impl Drop for Back {
        fn drop(&mut self) {
            two_level_folds::force(None);
            two_level_folds::force_fault(false);
        }
    }
    let _back = Back;
    const MODULUS: u64 = 0xFFFF_FFFF_0000_0001;
    let mut rng = Rng::new(0x2F01D);
    let mut limbs = |n: usize| -> Vec<u64> { (0..n * 3).map(|_| rng.next() % MODULUS).collect() };
    let levels = |tree: &DeviceFractionTree| -> Vec<(Vec<u64>, Vec<u64>)> {
        (0..tree.num_layers() - 1)
            .map(|layer| tree.layer_to_host(layer).expect("a level above the input"))
            .collect()
    };
    let build = |p: &[u64], q: &[u64], real: usize, num_vars: usize| -> DeviceFractionTree {
        if real == 1usize << num_vars {
            return DeviceFractionTree::build(p, q).expect("the card builds the tree");
        }
        let be = math_cuda::device::backend().expect("a card");
        let stream = be.next_stream();
        let p = math_cuda::device::htod_or_trim(&stream, p).expect("upload");
        let q = math_cuda::device::htod_or_trim(&stream, q).expect("upload");
        DeviceFractionTree::from_padded_input(stream, p, q, real, num_vars)
            .expect("the card builds the padded tree")
    };
    let mut checked = 0;
    for num_vars in 1..=12usize {
        let full = 1usize << num_vars;
        for real in [full, full / 2 + 1, full - 1] {
            if real <= full / 2 || real == 0 {
                continue;
            }
            let (p, q) = (limbs(real), limbs(real));
            two_level_folds::force(Some(false));
            let today = levels(&build(&p, &q, real, num_vars));
            two_level_folds::force(Some(true));
            let two = levels(&build(&p, &q, real, num_vars));
            assert_eq!(
                two, today,
                "2^{num_vars} fractions, {real} stored: the levels"
            );
            if num_vars >= 3 {
                two_level_folds::force_fault(true);
                let faulted = levels(&build(&p, &q, real, num_vars));
                two_level_folds::force_fault(false);
                assert_ne!(
                    faulted, today,
                    "2^{num_vars}, {real} stored: the fault must part"
                );
            }
            checked += 1;
        }
    }
    eprintln!("gkr two-level folds: {checked} trees built today's levels; every fault parted");
}

/// ★ The input layer written with the two tree levels above it
/// (`gkr::fused_input`, on by default; `LAMBDA_VM_GKR_FUSED_INPUT=0` writes the
/// layer and folds it as before) makes the same layer and the same tree: over
/// random columns and random input plans of 4, 5 and 7 interactions at 8 and 64
/// rows, kept whole (the padding's `0/1` written, `from_device`) and truncated
/// (only the real cells, `from_truncated_input`), the input cells and every
/// level above them are bit for bit the plain writer's. Then the fused pass's
/// upper cells off by one must part, so the comparison is load-bearing.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "flips process-global argue switches: run alone, --ignored --test-threads=1"]
fn the_fused_input_writes_todays_layer_and_tree() {
    use math_cuda::gkr::{
        DeviceFractionTree, InputPlan, fused_input, input_fold2_from_columns, input_from_columns,
    };
    struct Back;
    impl Drop for Back {
        fn drop(&mut self) {
            fused_input::force(None);
            fused_input::force_fault(false);
        }
    }
    let _back = Back;
    const MODULUS: u64 = 0xFFFF_FFFF_0000_0001;
    let mut rng = Rng::new(0xF5ED);
    let be = math_cuda::device::backend().expect("a card");
    let levels = |tree: &DeviceFractionTree| -> Vec<(Vec<u64>, Vec<u64>)> {
        (0..tree.num_layers() - 1)
            .map(|layer| tree.layer_to_host(layer).expect("a level above the input"))
            .collect()
    };
    let mut checked = 0;
    for interactions in [4usize, 5, 7] {
        for rows in [8usize, 64] {
            let width = 6usize;
            let columns: Vec<u64> = (0..width * rows).map(|_| rng.next() % MODULUS).collect();
            // Each side: a constant and up to three terms on random columns and shifts.
            let mut side_start = vec![0u32];
            let (mut terms, mut coeffs, mut constants) = (Vec::new(), Vec::new(), Vec::new());
            for _ in 0..2 * interactions {
                for _ in 0..3 {
                    constants.push(rng.next() % MODULUS);
                }
                let count = (rng.next() % 4) as usize;
                for _ in 0..count {
                    terms.push((rng.next() % width as u64) * rows as u64);
                    terms.push(rng.next() % rows as u64);
                    for _ in 0..3 {
                        coeffs.push(rng.next() % MODULUS);
                    }
                }
                side_start.push(side_start.last().unwrap() + count as u32);
            }
            let num_vars = (interactions.next_power_of_two() * rows).trailing_zeros() as usize;
            let real = interactions * rows;
            for padding in [true, false] {
                if !padding && real == 1usize << num_vars {
                    continue;
                }
                let stream = be.next_stream();
                let column_buf =
                    math_cuda::device::htod_or_trim(&stream, &columns).expect("upload");
                let view = column_buf.slice(..);
                let plan = InputPlan::upload(&stream, &side_start, &terms, &coeffs, &constants)
                    .expect("the plan uploads");
                let tree_of = |p, q, top| {
                    if padding {
                        DeviceFractionTree::from_device_top(stream.clone(), p, q, top)
                    } else {
                        DeviceFractionTree::from_truncated_input_top(
                            stream.clone(),
                            p,
                            q,
                            real,
                            num_vars,
                            top,
                        )
                    }
                    .expect("the card builds the tree")
                };
                let (p, q) =
                    input_from_columns(&stream, &view, rows, &plan, padding).expect("write");
                let today_input = (
                    stream.clone_dtoh(&p).unwrap(),
                    stream.clone_dtoh(&q).unwrap(),
                );
                let today = levels(&tree_of(p, q, None));
                let ((p, q), top) = input_fold2_from_columns(&stream, &view, rows, &plan, padding)
                    .expect("fused write");
                let fused_input_cells = (
                    stream.clone_dtoh(&p).unwrap(),
                    stream.clone_dtoh(&q).unwrap(),
                );
                let fused = levels(&tree_of(p, q, Some(top)));
                let what = format!("{interactions} interactions, {rows} rows, padding {padding}");
                assert_eq!(fused_input_cells, today_input, "{what}: the input cells");
                assert_eq!(fused, today, "{what}: the levels");
                fused_input::force_fault(true);
                let ((p, q), top) = input_fold2_from_columns(&stream, &view, rows, &plan, padding)
                    .expect("fused write");
                let faulted = levels(&tree_of(p, q, Some(top)));
                fused_input::force_fault(false);
                assert_ne!(faulted, today, "{what}: the fault must part");
                checked += 1;
            }
        }
    }
    eprintln!("gkr fused input: {checked} layers and trees as today's; every fault parted");
}

/// ★ The production switches (fused on, the corners skipped, no
/// cross-check) on every table without roots: its constraint part is zero,
/// so its grid `{0,1}²` is all corners, the corner skip is exact on random
/// columns too, and no grid point is left to launch — the case FAST job
/// 273's B arm crashed on (a division by the empty row count). Against
/// today's host rounds; each run counted as a fused session.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "flips process-global argue switches: run alone, --ignored --test-threads=1"]
fn the_fused_rounds_skip_the_corners_of_every_table_without_roots() {
    use multilinear::gpu_fused::*;
    let _switches = CardSwitches::set(FusedFaults::default(), false);
    let vm = vm_airs();
    let mut runs = 0;
    for (at, (air, label)) in every_air(&vm).into_iter().enumerate() {
        let mut rng = Rng::new(0x5C1 + at as u64);
        let probe = Table::from_air(air, &label, 8, &mut rng);
        if probe.shape.num_roots() != 0 {
            continue;
        }
        let n = (8..)
            .find(|n| probe.width() << n >= 4096)
            .expect("a height");
        let table = Table::from_air(air, &label, n, &mut rng);
        let session = table.session(table.random_columns(n, &mut rng), &mut rng);
        let claims = session.true_claims();
        let sessions = fused_sessions();
        session.assert_card_is_today(&claims, session.on_the_card(&claims), "corners skipped");
        assert_eq!(
            fused_sessions(),
            sessions + 1,
            "{label} n={n}: the card ran the fused rounds"
        );
        runs += 1;
    }
    assert_eq!(runs, 12, "the twelve tables without roots");
    eprintln!(
        "argue Q1b: {runs} tables without roots sent today's messages with the corners skipped"
    );
}

/// ★ A proof made with the fused rounds verifies: `test_keccak` under the
/// production switches of the B arm (fused, integer nodes, no cross-check),
/// then with the cross-check, each proved and verified
/// (`multilinear_prove::verify_with_options`), each taking the card's fused
/// rounds on the tables tall enough (n ≥ 7) — among them tables without roots.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "flips process-global argue switches and proves a program: run alone, --ignored --test-threads=1"]
fn a_proof_made_with_the_fused_rounds_verifies() {
    use crate::tables::MaxRowsConfig;
    use multilinear::gpu_fused::*;
    use stark::proof::options::ProofOptions;
    let options = ProofOptions::default_test_options();
    let elf = crate::test_utils::asm_elf_bytes("test_keccak");
    for xcheck in [false, true] {
        let _switches = CardSwitches::set(FusedFaults::default(), xcheck);
        math_cuda::sumcheck::force_int_nodes(Some(true));
        let (sessions, checks) = (fused_sessions(), fused_xchecks());
        let proof =
            crate::multilinear_prove::prove_with_options(&elf, &options, &MaxRowsConfig::default())
                .unwrap_or_else(|e| panic!("xcheck {xcheck}: test_keccak proves: {e:?}"));
        let fused = fused_sessions() - sessions;
        assert!(fused > 0, "xcheck {xcheck}: the fused rounds ran");
        if xcheck {
            assert_eq!(
                fused_xchecks() - checks,
                fused,
                "every fused table confirmed"
            );
        }
        assert!(
            crate::multilinear_prove::verify_with_options(&proof, &elf, &options)
                .unwrap_or_else(|e| panic!("xcheck {xcheck}: verify: {e:?}")),
            "xcheck {xcheck}: the proof verifies"
        );
        eprintln!(
            "argue Q1b: test_keccak proved with {fused} fused tables (xcheck {xcheck}) and verified"
        );
    }
}

/// ★ S1-1 on today's kernel (`sumcheck::int_nodes`): the device rounds
/// with integer nodes send today's messages (the Q1 tables at 2^10, fused off).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "flips process-global argue switches: run alone, --ignored --test-threads=1"]
fn the_int_node_rounds_send_todays_messages() {
    let _switches = CardSwitches::set(Default::default(), false);
    multilinear::gpu_fused::force_argue_fused(Some(false));
    math_cuda::sumcheck::force_int_nodes(Some(true));
    let vm = vm_airs();
    for (at, (air, label)) in every_air(&vm)
        .into_iter()
        .filter(|(_, l)| THE_Q1_TABLES.contains(&l.as_str()))
        .enumerate()
    {
        let mut rng = Rng::new(0x1A7 + at as u64);
        let table = Table::from_air(air, &label, 10, &mut rng);
        let session = table.session(table.random_columns(10, &mut rng), &mut rng);
        let claims = session.true_claims();
        let before = multilinear::gpu::sumcheck_calls();
        session.assert_card_is_today(&claims, session.on_the_card(&claims), "int nodes");
        assert!(
            multilinear::gpu::sumcheck_calls() > before,
            "{label}: the card ran the rounds"
        );
    }
}

/// ⛔ A wrong bus coefficient on the card: the cross-check refuses the prove
/// (the Q1 tables at 2^10). Mutation `LAMBDA_VM_ARGUE_FUSED_MUTATE=xcheck` makes the
/// comparison inert, and this test must then FAIL.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "flips process-global argue switches: run alone, --ignored --test-threads=1"]
fn a_wrong_bus_coefficient_on_the_card_is_refused_by_the_cross_check() {
    use multilinear::gpu_fused::*;
    let _switches = CardSwitches::set(
        FusedFaults {
            bus: true,
            keep_corners: true,
            ..FusedFaults::default()
        },
        true,
    );
    let vm = vm_airs();
    for (at, (air, label)) in every_air(&vm)
        .into_iter()
        .filter(|(_, l)| THE_Q1_TABLES.contains(&l.as_str()))
        .enumerate()
    {
        let mut rng = Rng::new(0xBAD + at as u64);
        let table = Table::from_air(air, &label, 10, &mut rng);
        let session = table.session(table.random_columns(10, &mut rng), &mut rng);
        let claims = session.true_claims();
        match session.on_the_card(&claims) {
            Err(multilinear::Error::DeviceFailed {
                stage: "fused cross-check",
            }) => {}
            other => panic!(
                "{label}: a wrong bus coefficient was not refused: {:?}",
                other.map(|_| ())
            ),
        }
    }
}

/// ⛔ The corner check on the card refuses a trace that breaks its AIR —
/// random columns, the corners checked rather than summed (the Q1 tables at
/// 2^10). Mutation `LAMBDA_VM_ARGUE_FUSED_MUTATE=corners` makes it inert, and
/// this test must then FAIL.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "flips process-global argue switches: run alone, --ignored --test-threads=1"]
fn the_corner_check_on_the_card_refuses_a_broken_trace() {
    use multilinear::gpu_fused::*;
    let _switches = CardSwitches::set(FusedFaults::default(), true);
    let vm = vm_airs();
    for (at, (air, label)) in every_air(&vm)
        .into_iter()
        .filter(|(_, l)| THE_Q1_TABLES.contains(&l.as_str()))
        .enumerate()
    {
        let mut rng = Rng::new(0xC0 + at as u64);
        let table = Table::from_air(air, &label, 10, &mut rng);
        let session = table.session(table.random_columns(10, &mut rng), &mut rng);
        let claims = session.true_claims();
        match session.on_the_card(&claims) {
            Err(multilinear::Error::ConstraintViolated { .. }) => {}
            other => panic!(
                "{label}: a broken trace was not refused: {:?}",
                other.map(|_| ())
            ),
        }
    }
}

/// The device tail's rule (`multilinear::gpu_fused::zc_device_tail`: at least
/// [`multilinear::gpu_fused::DEVICE_TAIL_MIN_STEPS`] lowered steps) takes the
/// VM's long programs and none of the short ones: KECCAK_RND and ECDAS on,
/// CPU off. Prints every program it takes, so a box log shows which.
#[test]
fn the_device_tail_takes_the_long_vm_programs() {
    use multilinear::gpu_fused::{DEVICE_TAIL_MIN_STEPS, lower_fused};
    let mut vm = vm_airs();
    let opts = stark::proof::options::GoldilocksCubicProofOptions::with_blowup(2)
        .map(crate::security_posture::stark_options)
        .expect("blowup=2 valid");
    vm.push((
        Box::new(crate::test_utils::create_ecrec_air(&opts)),
        "ECREC",
    ));
    let mut long = Vec::new();
    for (at, (air, label)) in vm.iter().enumerate() {
        let mut rng = Rng::new(0x7A11_0000 + at as u64);
        let table = Table::from_air(&**air, label, 4, &mut rng);
        if table.constraints.roots().is_empty() {
            continue;
        }
        let steps = lower_fused(&table.constraints)
            .unwrap_or_else(|| panic!("{label}: lowers"))
            .nodes
            .len()
            / 2;
        if steps >= DEVICE_TAIL_MIN_STEPS {
            long.push(format!("{label} {steps}"));
        }
        match *label {
            "KECCAK_RND" | "ECDAS" => {
                assert!(steps >= DEVICE_TAIL_MIN_STEPS, "{label}: {steps} steps")
            }
            "CPU" => assert!(steps < DEVICE_TAIL_MIN_STEPS, "{label}: {steps} steps"),
            _ => {}
        }
    }
    eprintln!(
        "device tail takes {} programs: {}",
        long.len(),
        long.join(", ")
    );
    assert!(long.len() >= 2, "{long:?}");
}

/// The five rules' factors, rules and weight points.
#[cfg(feature = "cuda")]
type TwoBuses<'a> = (Vec<Mle<Ext3>>, Vec<Rule<'a, Ext3>>, Vec<Vec<EE>>);

/// A table's batch with a second bus at its own point (its local lookups:
/// rules four and five at `λ³` and `λ⁴`, `eq(ρ')` the third weight), on the
/// card through the fused session and on the host through today's rules, as
/// the batched argue hands them to `front_loaded::TableRounds`.
#[cfg(feature = "cuda")]
impl Session<'_> {
    /// The five rules over today's factors and the three weights, and the
    /// second bus's claim point (its interactions are the table's own, at
    /// another random point).
    fn two_buses(&self, second_point: &[EE]) -> TwoBuses<'_> {
        let table = self.table;
        let (mut polys, mut rules) = self.todays();
        let second = logup::claim_statements(
            &table.interactions,
            second_point,
            self.num_vars(),
            table.width() + 2,
        )
        .unwrap_or_else(|e| panic!("{}: {e:?}", table.label));
        let rho = self.claim_point[self.claim_point.len() - self.num_vars()..].to_vec();
        polys.push(eq_mle(&second.row_point).unwrap());
        rules.push(second.numerator);
        rules.push(second.denominator);
        (polys, rules, vec![self.r.clone(), rho, second.row_point])
    }

    /// The batch's five rules run through `TableRounds`: on the card when
    /// `card`, else on the host; the proof, the point and every factor's
    /// value at it.
    fn two_buses_proved(
        &self,
        second_point: &[EE],
        lambda: &EE,
        card: bool,
    ) -> (SumcheckProof<Ext3>, Vec<EE>, Vec<EE>) {
        use multilinear::front_loaded::{PaddingFault, RoundStepper, TableRounds};
        let table = self.table;
        let n = self.num_vars();
        let (polys, rules, weights) = self.two_buses(second_point);
        let lambdas = multilinear::challenge_powers(lambda, 5);
        // The table's own claim, `Σ_k λ^k·claim_k` over the cube.
        let mut scratch = Vec::new();
        let claim = (0..polys[0].len()).fold(EE::zero(), |acc, i| {
            let values: Vec<EE> = polys.iter().map(|p| p.evals()[i]).collect();
            rules.iter().zip(&lambdas).fold(acc, |acc, (rule, l)| {
                acc + l * rule.apply_in(&values, &mut scratch)
            })
        });
        let lifted: Vec<Mle<Ext3>> = polys[..table.width()].to_vec();
        let public: Vec<Mle<Ext3>> = table
            .shape
            .public_selectors()
            .iter()
            .map(|s| s.table::<Ext3>(n).unwrap())
            .collect();
        let device = card.then(|| {
            multilinear::gpu::upload_factors_from_columns(
                &self.columns,
                &table.kinds,
                &public,
                None,
            )
            .unwrap_or_else(|| panic!("{}: the card takes the factors", table.label))
        });
        let points: Vec<&[EE]> = weights.iter().map(Vec::as_slice).collect();
        let rho = &self.claim_point;
        let degree = rules.iter().map(Rule::degree).max().unwrap_or(2).max(2);
        let fused = card.then(|| multilinear::gpu_fused::FusedInput {
            constraints: &table.constraints,
            betas: &table.betas,
            interactions: &table.interactions,
            claim_point: rho,
            r: &self.r,
            columns: None,
            second: Some(multilinear::gpu_fused::SecondInput {
                interactions: &table.interactions,
                claim_point: second_point,
            }),
        });
        let mut rounds = vec![
            TableRounds::new(
                rules,
                lambdas,
                &claim,
                &points,
                n,
                device.as_ref(),
                None,
                fused,
                || Ok(lifted.clone()),
            )
            .unwrap_or_else(|e| panic!("{}: {e:?}", table.label)),
        ];
        assert_eq!(rounds[0].on_device(), card, "{}: where it ran", table.label);
        let mut t = self.transcript();
        let (proof, point) = multilinear::front_loaded::prove_with(
            &mut rounds,
            &[EE::one()],
            degree,
            &mut t,
            PaddingFault { scaled: None },
        )
        .unwrap_or_else(|e| panic!("{}: {e:?}", table.label));
        let _ = rounds[0].num_vars();
        let bound: Vec<EE> = rounds[0]
            .polys()
            .iter()
            .take(table.width())
            .map(|p| p.evals()[0])
            .collect();
        (proof, point, bound)
    }
}

/// The fused card session with a second bus (a table's local lookups at their
/// own point) sends the host's messages: KECCAK_RND and CPU at 2^10 and 2^12
/// on random columns, the corners summed, the card's rounds against the host's
/// five-rule batch — the same messages, point and factor values. With the
/// second bus's coefficient faulted the card parts from the host, so the
/// comparison is load-bearing.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "flips process-global argue switches: run alone, --ignored --test-threads=1"]
fn a_second_bus_on_the_card_sends_the_hosts_messages() {
    use multilinear::gpu_fused::*;
    let _switches = CardSwitches::set(
        FusedFaults {
            keep_corners: true,
            ..FusedFaults::default()
        },
        false,
    );
    let vm = vm_airs();
    let airs = every_air(&vm);
    let mut checked = 0usize;
    for label in THE_Q1_TABLES {
        for n in [10usize, 12] {
            let air = airs.iter().find(|(_, l)| l == label).unwrap().0;
            let mut rng = Rng::new(0x5EC0 + n as u64);
            let table = Table::from_air(air, label, n, &mut rng);
            let session = table.session(table.random_columns(n, &mut rng), &mut rng);
            let second_point = rng.exts(logup::input_layer_vars(table.interactions.len(), n));
            let lambda = rng.ext();
            let sessions = fused_sessions();
            let card = session.two_buses_proved(&second_point, &lambda, true);
            assert_eq!(
                fused_sessions(),
                sessions + 1,
                "{label} n={n}: the card ran the fused rounds"
            );
            let host = session.two_buses_proved(&second_point, &lambda, false);
            assert_eq!(card.0.rounds.len(), host.0.rounds.len(), "{label} n={n}");
            for (round, (a, b)) in host.0.rounds.iter().zip(&card.0.rounds).enumerate() {
                assert_eq!(a.evaluations, b.evaluations, "{label} n={n}: round {round}");
            }
            assert_eq!(card.1, host.1, "{label} n={n}: the point");
            assert_eq!(card.2, host.2, "{label} n={n}: the factors at the point");
            checked += 1;
        }
    }
    // The negative control: the first bus's first coefficient off by one on
    // the card (`FusedFaults::bus`) parts the card from the host.
    {
        let _faulted = CardSwitches::set(
            FusedFaults {
                keep_corners: true,
                bus: true,
                ..FusedFaults::default()
            },
            false,
        );
        let label = THE_Q1_TABLES[0];
        let air = airs.iter().find(|(_, l)| *l == label).unwrap().0;
        let mut rng = Rng::new(0x5EC0 + 10);
        let table = Table::from_air(air, label, 10, &mut rng);
        let session = table.session(table.random_columns(10, &mut rng), &mut rng);
        let second_point = rng.exts(logup::input_layer_vars(table.interactions.len(), 10));
        let lambda = rng.ext();
        let card = session.two_buses_proved(&second_point, &lambda, true);
        let host = session.two_buses_proved(&second_point, &lambda, false);
        let messages = |p: &SumcheckProof<Ext3>| -> Vec<Vec<EE>> {
            p.rounds.iter().map(|r| r.evaluations.clone()).collect()
        };
        assert_ne!(
            messages(&card.0),
            messages(&host.0),
            "a faulted bus parts the card"
        );
    }
    eprintln!("argue second bus: {checked} card runs sent the host's messages");
}

fn grid_production_airs() -> Vec<crate::test_utils::LabeledAir> {
    use crate::tables::{
        ecdas::EcdasForm, keccak_rnd::KeccakRndForm, memw_aligned::MemwAlignedOrder,
    };
    use crate::test_utils::*;

    let opts = stark::proof::options::GoldilocksCubicProofOptions::with_blowup(2)
        .map(crate::security_posture::stark_options)
        .expect("blowup=2 valid");
    let mut airs = production_airs(&opts);
    assert_eq!(airs.len(), NUM_PRODUCTION_AIRS);
    let variants: Vec<LabeledAir> = vec![
        ("CPU in-row", Box::new(create_cpu_air_inrow(&opts, 0))),
        (
            "REGISTER in-row",
            Box::new(create_register_air_inrow(&opts, 0)),
        ),
        ("HALT in-row", Box::new(create_halt_air_inrow(&opts))),
        (
            "MEMW_R residual",
            Box::new(create_register_residual_air(&opts)),
        ),
        (
            "MEMW_A limbs",
            Box::new(create_memw_aligned_air_for(&opts, MemwAlignedOrder::Limbs)),
        ),
        (
            "ECDAS halves",
            create_ecdas_air_for(&opts, EcdasForm::Halves, "ECDAS halves"),
        ),
        (
            "KECCAK_RND lean",
            Box::new(create_keccak_rnd_air_for(&opts, KeccakRndForm::Lean)),
        ),
        (
            "KECCAK_RND lean local",
            Box::new(create_keccak_rnd_air_for(&opts, KeccakRndForm::LeanLocal)),
        ),
        ("MEM_LOAD", Box::new(create_mem_load_air(&opts))),
        ("MEM_STORE", Box::new(create_mem_store_air(&opts))),
        ("MEM_LOAD8", Box::new(create_mem_load8_air(&opts))),
        ("MEM_STORE8", Box::new(create_mem_store8_air(&opts))),
        ("MEM_LOADW", Box::new(create_mem_loadw_air(&opts))),
        ("MEM_STOREW", Box::new(create_mem_storew_air(&opts))),
        ("MEM_LOAD2W", Box::new(create_mem_load2w_air(&opts))),
        ("MEM_STORE2W", Box::new(create_mem_store2w_air(&opts))),
        ("MEMW_W", Box::new(create_memw_w_air(&opts))),
        ("LOCAL_BYTES", Box::new(create_local_bytes_air(&opts))),
        ("KECCAK_W", Box::new(create_keccak_w_air(&opts))),
        ("ECREC_W", Box::new(create_ecrec_w_air(&opts))),
        ("COPY_W", Box::new(create_copy_w_air(&opts))),
    ];
    airs.extend(variants);
    airs
}

/// Select the same long programs as the production dispatch.
fn long_constraint_tables(n: usize, rng: &mut Rng) -> Vec<Table> {
    use multilinear::gpu_fused::{DEVICE_TAIL_MIN_STEPS, lower_fused};

    let tables: Vec<_> = grid_production_airs()
        .into_iter()
        .map(|(label, air)| Table::from_air(&*air, label, n, rng))
        .filter(|table| {
            lower_fused(&table.constraints)
                .expect(&table.label)
                .nodes
                .len()
                / 2
                >= DEVICE_TAIL_MIN_STEPS
        })
        .collect();
    for label in [
        "KECCAK_RND",
        "ECDAS",
        "ECREC",
        "ECSM",
        "KECCAK_W",
        "ECREC_W",
    ] {
        assert!(tables.iter().any(|table| table.label == label), "{label}");
    }
    tables
}

#[test]
fn production_constraint_chunks_fit_and_sum_to_the_whole_program() {
    use multilinear::fused_chunks::{CHUNKED_LEAF_CAP, decode, eval_chunked, lower_chunked, op};
    let mut rng = Rng::new(0xC16);
    for table in long_constraint_tables(8, &mut rng) {
        let label = &table.label;
        let chunked = lower_chunked(&table.constraints, CHUNKED_LEAF_CAP).expect(label);
        assert!(chunked.chunks() > 1, "{label}");
        for bounds in chunked.bounds.windows(2) {
            let reads = chunked.code[bounds[0] as usize..bounds[1] as usize]
                .iter()
                .map(|&step| decode(step))
                .filter_map(|(tag, column, _, _)| (tag == op::VAR).then_some(column))
                .collect::<std::collections::BTreeSet<_>>();
            assert!(reads.len() <= CHUNKED_LEAF_CAP, "{label}: {reads:?}");
        }
        let whole = table
            .constraints
            .zerocheck_program(&table.betas, table.width())
            .unwrap();
        let mut scratch = Vec::new();
        for _ in 0..3 {
            let mut values = rng.exts(table.width());
            values.push(EE::one());
            assert_eq!(
                eval_chunked(&chunked, &values, &table.betas),
                Some(whole.eval(&values, &mut scratch)),
                "{label}"
            );
        }
        eprintln!("{label}: {} chunks within the column cap", chunked.chunks());
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "one card, run alone"]
fn thin_constraint_rounds_in_chunks_match_the_whole_card_rounds() {
    use multilinear::gpu_fused::{FusedFaults, force_zc_chunked, take_tail};

    let _permit = crate::lfm::device_permit::serial_for_test();
    let _switches = CardSwitches::set(
        FusedFaults {
            keep_corners: true,
            ..FusedFaults::default()
        },
        false,
    );
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            force_zc_chunked(None);
        }
    }
    let _reset = Reset;
    let n = 8;
    let mut rng = Rng::new(0xC16);
    for table in long_constraint_tables(n, &mut rng) {
        let session = table.session(table.random_columns(n, &mut rng), &mut rng);
        let second_point = rng.exts(logup::input_layer_vars(table.interactions.len(), n));
        let lambda = rng.ext();
        let _ = take_tail();
        force_zc_chunked(Some(false));
        let whole = session.two_buses_proved(&second_point, &lambda, true);
        assert_eq!(take_tail().chunked_rounds, 0, "{}", table.label);
        force_zc_chunked(Some(true));
        let chunked = session.two_buses_proved(&second_point, &lambda, true);
        let rounds = take_tail().chunked_rounds;
        assert!(rounds > 0, "{}: no round used chunks", table.label);
        assert_eq!(
            chunked.0.rounds.len(),
            whole.0.rounds.len(),
            "{}",
            table.label
        );
        for (round, (chunked, whole)) in chunked.0.rounds.iter().zip(&whole.0.rounds).enumerate() {
            assert_eq!(
                chunked.evaluations, whole.evaluations,
                "{}: round {round}",
                table.label
            );
        }
        assert_eq!(chunked.1, whole.1, "{}: challenges", table.label);
        assert_eq!(chunked.2, whole.2, "{}: bound factors", table.label);
        eprintln!("{}: {rounds} chunked card rounds match", table.label);
    }
}

#[test]
fn dump_grid_root_degrees() {
    use multilinear::gpu::op;
    use multilinear::gpu_fused::{
        lower_fused,
        op::{ACC, NO_SELECTOR},
    };
    use std::collections::BTreeMap;

    for (label, air) in grid_production_airs() {
        let table = Table::from_air(&*air, label, 10, &mut Rng::new(1));
        if table.constraints.degree() != 3 {
            continue;
        }
        let program = lower_fused(&table.constraints).expect(label);
        let mut slots = vec![0; program.num_slots];
        let mut degree = Vec::new();
        let mut deps = Vec::<[Option<usize>; 2]>::new();
        let mut roots = BTreeMap::new();
        let mut live = [
            vec![false; program.nodes.len() / 2],
            vec![false; program.nodes.len() / 2],
        ];
        println!("GRID DUMP {label} begin");
        for (index, node) in program.nodes.chunks_exact(2).enumerate() {
            let (tag, a) = (node[0] as u32, (node[0] >> 32) as usize);
            let (b, dest) = (node[1] as u32, (node[1] >> 32) as usize);
            let (d, operands) = match tag {
                op::FIXED => (0, [None, None]),
                op::VAR => (1, [None, None]),
                op::NEG => (degree[slots[a]], [Some(slots[a]), None]),
                op::ADD | op::SUB | op::MUL => {
                    let (left, right) = (slots[a], slots[b as usize]);
                    let d = if tag == op::MUL {
                        degree[left] + degree[right]
                    } else {
                        usize::max(degree[left], degree[right])
                    };
                    (d, [Some(left), Some(right)])
                }
                ACC => (
                    degree[slots[a]] + usize::from(b != NO_SELECTOR),
                    [Some(slots[a]), None],
                ),
                _ => panic!("unknown opcode {tag}"),
            };
            println!("{index} op={tag} a={a} b={b} dest={dest} degree={d}");
            if tag == ACC {
                *roots.entry(d).or_insert(0) += 1;
                live[usize::from(d > 2)][index] = true;
            } else {
                slots[dest] = index;
            }
            degree.push(d);
            deps.push(operands);
        }
        for part in &mut live {
            for (i, operands) in deps.iter().enumerate().rev() {
                if part[i] {
                    for dep in operands.iter().flatten() {
                        part[*dep] = true;
                    }
                }
            }
        }
        let [low, high] = live.map(|part| part.into_iter().filter(|x| *x).count());
        println!(
            "GRID CENSUS {label}: roots {roots:?}; steps whole={} low={low} high={high}",
            deps.len()
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "one dedicated card, 2^21 rows"]
// Build and time both grid programs directly, independent of the dispatch knob.
fn bench_cpu_constraint_grid_split() {
    use math_cuda::argue_fused::{FusedBus, FusedZerocheck};
    use math_cuda::columns::DeviceColumns;
    use math_cuda::sumcheck::{ColumnFactors, ColumnSlot};
    use multilinear::gpu::{ext3_from_raw, ext3_raw};
    use multilinear::gpu_fused::{
        LOW_GRID_POINTS, add_low_grid, lower_fused, lower_fused_grid_split,
    };
    use std::time::Instant;

    let _permit = crate::lfm::device_permit::serial_for_test();
    let (_, air) = grid_production_airs()
        .into_iter()
        .find(|(label, _)| *label == "CPU in-row")
        .unwrap();
    let n = 21;
    let rows = 1 << n;
    let mut rng = Rng::new(0x519);
    let table = Table::from_air(&*air, "CPU in-row", n, &mut rng);
    let whole = lower_fused(&table.constraints).unwrap();
    let [low, high] = lower_fused_grid_split(&table.constraints).unwrap();
    let betas: Vec<u64> = table
        .betas
        .iter()
        .flat_map(|b| ext3_raw(b).unwrap())
        .collect();
    let factors: Vec<Vec<u64>> = (0..table.width())
        .map(|_| (0..rows).map(|_| rng.base().canonical_u64()).collect())
        .collect();
    let columns = DeviceColumns::upload(&factors.iter().map(Vec::as_slice).collect::<Vec<_>>())
        .expect("CPU-sized resident columns");
    drop(factors);
    let slots = (0..table.width())
        .map(ColumnSlot::Column)
        .collect::<Vec<_>>();
    let factors = ColumnFactors::new(&columns.view(0, table.width()), rows, &slots).unwrap();
    let weights: Vec<_> = rng
        .exts(n - 2)
        .iter()
        .flat_map(|r| ext3_raw(r).unwrap())
        .collect();
    let points: Vec<_> = (0..4)
        .flat_map(|a| (0..4).map(move |b| (a, b)))
        .filter(|&(a, b)| a >= 2 || b >= 2)
        .collect();
    let bus = FusedBus {
        slots: &[],
        coeffs: &[],
        constant: [0; 3],
    };
    let mut session = FusedZerocheck::new(
        factors.view(),
        whole.device_program(&betas),
        bus,
        &weights,
        &weights,
        points.len(),
        5,
        None,
        |be, bytes| be.try_reserve(bytes).ok(),
    )
    .unwrap()
    .expect("reserve a CPU grid session");
    assert!(
        session
            .prepare_grid_split([low.device_program(&betas), high.device_program(&betas)])
            .unwrap()
    );

    let mut run = |split: bool| {
        let tick = Instant::now();
        let (t, violation) = session
            .grid_constraints(&points, false, split.then_some(&LOW_GRID_POINTS[..]))
            .unwrap();
        assert!(violation.is_none());
        let mut grid = std::array::from_fn(|_| std::array::from_fn(|_| EE::zero()));
        for (&(a, b), limbs) in points.iter().zip(t.chunks_exact(3)) {
            grid[a as usize][b as usize] = ext3_from_raw(limbs);
        }
        if split {
            let low = std::array::from_fn(|i| {
                let at = (points.len() + i) * 3;
                ext3_from_raw(&t[at..at + 3])
            });
            add_low_grid(&mut grid, &low);
        }
        std::hint::black_box(grid);
        tick.elapsed().as_secs_f64() * 1000.0
    };
    for _ in 0..3 {
        run(false);
        run(true);
    }
    let mut whole_ms = Vec::new();
    let mut split_ms = Vec::new();
    for rep in 0..10 {
        let (whole, split) = if rep % 2 == 0 {
            (run(false), run(true))
        } else {
            let split = run(true);
            (run(false), split)
        };
        println!(
            "GRID BENCH rep={} whole_ms={whole:.6} split_ms={split:.6}",
            rep + 1
        );
        whole_ms.push(whole);
        split_ms.push(split);
    }
    whole_ms.sort_by(f64::total_cmp);
    split_ms.sort_by(f64::total_cmp);
    let whole_ms = (whole_ms[4] + whole_ms[5]) / 2.0;
    let split_ms = (split_ms[4] + split_ms[5]) / 2.0;
    println!(
        "GRID BENCH median whole_ms={whole_ms:.6} split_ms={split_ms:.6} ratio={:.6}",
        split_ms / whole_ms
    );
}

#[test]
fn the_lowered_grid_parts_sum_to_the_whole_on_random_rows() {
    use multilinear::gpu_fused::{lower_fused, lower_fused_grid_split, run_lowered_host};

    let mut rng = Rng::new(0x3533);
    for (label, air) in grid_production_airs() {
        let table = Table::from_air(&*air, label, 10, &mut rng);
        if table.constraints.degree() != 3 {
            continue;
        }
        let whole = lower_fused(&table.constraints).expect(label);
        let [low, high] = lower_fused_grid_split(&table.constraints).expect(label);
        for _ in 0..16 {
            let values: Vec<_> = (0..table.width()).map(|_| rng.base()).collect();
            let eval = |p| run_lowered_host(p, &table.betas, &values);
            assert_eq!(eval(&low) + eval(&high), eval(&whole), "{label}");
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "one card"]
fn the_split_grid_matches_the_whole_grid_on_every_cubic_program() {
    use math_cuda::argue_fused::{FusedBus, FusedZerocheck};
    use math_cuda::columns::DeviceColumns;
    use math_cuda::sumcheck::{ColumnFactors, ColumnSlot};
    use multilinear::gpu::{ext3_from_raw, ext3_raw};
    use multilinear::gpu_fused::{
        LOW_GRID_POINTS, add_low_grid, lower_fused, lower_fused_grid_split, run_lowered_host,
    };

    let _permit = crate::lfm::device_permit::serial_for_test();
    let n = 6;
    let rows = 1 << n;
    let quarter = rows / 4;
    let mut rng = Rng::new(0xc203);
    let points: Vec<_> = (0..4)
        .flat_map(|a| (0..4).map(move |b| (a, b)))
        .filter(|&(a, b)| a >= 2 || b >= 2)
        .collect();
    let mut tested = 0;
    for (label, air) in grid_production_airs() {
        let table = Table::from_air(&*air, label, n, &mut rng);
        if table.constraints.degree() != 3 {
            continue;
        }
        let whole = lower_fused(&table.constraints).expect(label);
        let [low, high] = lower_fused_grid_split(&table.constraints).expect(label);
        let betas: Vec<_> = table
            .betas
            .iter()
            .flat_map(|b| ext3_raw(b).unwrap())
            .collect();
        // Each corner varies in a distinct suffix bit. Choose its weight so
        // its weighted quadratic sum is zero while its rows can vary freely.
        // This exercises all the production root walks on nontrivial grids.
        let mut corner_rows = Vec::with_capacity(4);
        let mut weights = Vec::with_capacity(12);
        for _ in 0..4 {
            let pair =
                std::array::from_fn(|_| (0..table.width()).map(|_| rng.base()).collect::<Vec<_>>());
            let [l0, l1] = pair
                .each_ref()
                .map(|row| run_lowered_host(&low, &table.betas, row));
            let r = l0
                * (l0 - l1)
                    .inv()
                    .unwrap_or_else(|_| panic!("{label}: constant corner"));
            assert_eq!(l0 + r * (l1 - l0), EE::zero());
            corner_rows.push(pair);
            weights.extend(ext3_raw(&r).unwrap());
        }
        let factors: Vec<Vec<u64>> = (0..table.width())
            .map(|slot| {
                (0..rows)
                    .map(|row| {
                        let corner = row / quarter;
                        let variant = (row >> (3 - corner)) & 1;
                        corner_rows[corner][variant][slot].canonical_u64()
                    })
                    .collect()
            })
            .collect();
        let columns = DeviceColumns::upload(&factors.iter().map(Vec::as_slice).collect::<Vec<_>>())
            .expect(label);
        let slots = (0..table.width())
            .map(ColumnSlot::Column)
            .collect::<Vec<_>>();
        let factors = ColumnFactors::new(&columns.view(0, table.width()), rows, &slots).unwrap();
        let mut session = FusedZerocheck::new(
            factors.view(),
            whole.device_program(&betas),
            FusedBus {
                slots: &[],
                coeffs: &[],
                constant: [0; 3],
            },
            &weights,
            &weights,
            points.len(),
            5,
            None,
            |be, bytes| be.try_reserve(bytes).ok(),
        )
        .unwrap()
        .expect(label);
        assert!(
            session
                .prepare_grid_split([low.device_program(&betas), high.device_program(&betas)])
                .unwrap()
        );
        let (whole, _) = session.grid_constraints(&points, false, None).unwrap();
        let (split, _) = session
            .grid_constraints(&points, false, Some(&LOW_GRID_POINTS))
            .unwrap();
        let mut grid = [[EE::zero(); 4]; 4];
        for (&(a, b), limbs) in points.iter().zip(split.chunks_exact(3)) {
            grid[a as usize][b as usize] = ext3_from_raw(limbs);
        }
        let low_at_five = std::array::from_fn(|i| {
            let at = (points.len() + i) * 3;
            ext3_from_raw(&split[at..at + 3])
        });
        assert!(low_at_five.iter().any(|x| *x != EE::zero()), "{label}");
        add_low_grid(&mut grid, &low_at_five);
        for (&(a, b), limbs) in points.iter().zip(whole.chunks_exact(3)) {
            assert_eq!(
                grid[a as usize][b as usize],
                ext3_from_raw(limbs),
                "{label} ({a},{b})"
            );
        }
        eprintln!("{label}: split grid matches");
        tested += 1;
    }
    assert_eq!(tested, 21);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "one card, with the split grid enabled by default"]
fn checking_grid_corners_uses_the_whole_grid_and_counts_the_check() {
    check_grid_dispatch(true);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "one card, run alone with LAMBDA_VM_ZC_GRID_SPLIT=0"]
fn disabling_grid_split_uses_the_whole_grid() {
    check_grid_dispatch(false);
}

#[cfg(feature = "cuda")]
fn check_grid_dispatch(expect_split: bool) {
    use multilinear::front_loaded::TableRounds;
    use multilinear::gpu_fused::{FusedInput, fused_corner_checks, zc_grid_split};

    let _permit = crate::lfm::device_permit::serial_for_test();
    assert_eq!(
        zc_grid_split(),
        expect_split,
        "set LAMBDA_VM_ZC_GRID_SPLIT for this test"
    );
    let (_, air) = grid_production_airs()
        .into_iter()
        .find(|(label, _)| *label == "CPU in-row")
        .unwrap();
    let n = 10;
    let mut rng = Rng::new(0xa215);
    let table = Table::from_air(&*air, "CPU in-row", n, &mut rng);
    let zero = Mle::new(vec![FB::zero(); 1 << n]).unwrap();
    let session = table.session(vec![zero; table.num_columns], &mut rng);
    let claims = session.true_claims();
    assert_eq!(claims[0], EE::zero());
    let public: Vec<Mle<Ext3>> = table
        .shape
        .public_selectors()
        .iter()
        .map(|selector| selector.table::<Ext3>(n).unwrap())
        .collect();
    let device = multilinear::gpu::upload_factors_from_columns(
        &session.columns,
        &table.kinds,
        &public,
        None,
    )
    .expect("CPU factors on the card");
    let rho = &session.claim_point[session.claim_point.len() - n..];
    let lambdas = multilinear::challenge_powers(&rng.ext(), 3);
    let claim = claims
        .iter()
        .zip(&lambdas)
        .map(|(c, l)| c * l)
        .fold(EE::zero(), |a, b| a + b);
    for check in [false, true] {
        let _switches = CardSwitches::set(Default::default(), check);
        let before = fused_corner_checks();
        let (_, rules) = session.todays();
        let rounds = TableRounds::new(
            rules,
            lambdas.clone(),
            &claim,
            &[&session.r, rho],
            n,
            Some(&device),
            None,
            Some(FusedInput {
                constraints: &table.constraints,
                betas: &table.betas,
                interactions: &table.interactions,
                claim_point: &session.claim_point,
                r: &session.r,
                columns: None,
                second: None,
            }),
            || panic!("the grid must stay on the card"),
        )
        .unwrap();
        assert!(rounds.on_device());
        assert_eq!(rounds.grid_was_split(), expect_split && !check);
        assert_eq!(fused_corner_checks() - before, u64::from(check));
    }
}
