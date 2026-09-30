//! D-ARGUE's stage 1 on the tables the argue really proves — the VM's and the
//! W-LFM's (Q1a, `thoughts/zf/gap2/fix2/D-ARGUE.md` §4.1): the base-field
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

use super::lean_program_census::{VmAir, vm_airs, w_lfm_airs};
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

/// Every table the argue proves, by name: the VM's, then the W-LFM chips'.
fn every_air<'a>(
    vm: &'a [(VmAir, &'static str)],
    lfm: &'a crate::lfm::airs::LfmAirs,
) -> Vec<(DynAir<'a>, String)> {
    vm.iter()
        .map(|(air, label)| (&**air as DynAir<'a>, label.to_string()))
        .chain(
            lfm.air_refs()
                .into_iter()
                .map(|air| (air, air.name().to_string())),
        )
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

/// The three tables Q1 is sized on: the widest precompile, the VM's tallest
/// table, and the W-LFM's one big batch.
const THE_THREE: [&str; 3] = ["KECCAK_RND", "CPU", "LFM_HASH"];

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
    let lfm = w_lfm_airs();
    let mut exceptions = Vec::new();
    for (air, label) in every_air(&vm, &lfm) {
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
/// every VM table and every W-LFM chip, over random columns with the corners
/// computed, at heights 1, 2, 3 and 5; and counts what the model says.
#[test]
fn stage1_sends_todays_messages_on_every_table() {
    let vm = vm_airs();
    let lfm = w_lfm_airs();
    let airs = every_air(&vm, &lfm);
    assert_eq!(airs.len(), 36, "26 VM tables and 10 W-LFM chips");
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

/// ★ (a) The same on the three Q1 tables at 2^8 and 2^10 rows, and CPU at 2^12.
/// Heights up to 2^14 run on the box ([`stage1_sends_todays_messages_at_the_top_heights`]).
#[test]
fn stage1_sends_todays_messages_on_the_three() {
    run_the_three(&[
        ("KECCAK_RND", 8),
        ("KECCAK_RND", 10),
        ("CPU", 8),
        ("CPU", 10),
        ("CPU", 12),
        ("LFM_HASH", 8),
        ("LFM_HASH", 10),
    ]);
}

/// The three at 2^12 and 2^14 rows — minutes on a laptop core, so the box
/// runs it (`--release`).
#[test]
#[ignore = "the three Q1 tables at 2^12 and 2^14 rows: a box run"]
fn stage1_sends_todays_messages_at_the_top_heights() {
    run_the_three(&[
        ("KECCAK_RND", 12),
        ("KECCAK_RND", 14),
        ("CPU", 14),
        ("LFM_HASH", 12),
        ("LFM_HASH", 14),
    ]);
}

fn run_the_three(heights: &[(&str, usize)]) {
    let vm = vm_airs();
    let lfm = w_lfm_airs();
    let airs = every_air(&vm, &lfm);
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
/// the three.
#[test]
fn a_wrong_bus_coefficient_parts_the_messages_on_the_three() {
    let vm = vm_airs();
    let lfm = w_lfm_airs();
    let airs = every_air(&vm, &lfm);
    for name in THE_THREE {
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
/// 0 on each of the three, and the corner check refuses at the first row.
#[test]
fn the_corner_skip_parts_on_random_columns_of_the_three() {
    let vm = vm_airs();
    let lfm = w_lfm_airs();
    let airs = every_air(&vm, &lfm);
    for name in THE_THREE {
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
/// from a WHIR proof of `test_keccak`, LFM_HASH from a W-LFM proof of
/// `TrivialV0` under RPX (`multilinear_table::argue_capture`).
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
    use crate::lfm::hash::HasherKind;
    use crate::lfm::whir_proof::{build_whir_artifacts, lfm_prove_whir};
    use crate::tables::MaxRowsConfig;
    use crate::tables::types::FE;
    use stark::multilinear_table::argue_capture::{self, Captured};
    use stark::proof::options::ProofOptions;

    // The W-LFM program's own AIRs, as its build lays them out: its chip set
    // and hash chunks are the program's, which the census's stand-in set need
    // not match.
    let options = ProofOptions::default_test_options();
    let program = crate::lfm::programs::trivial_program();
    let build = build_whir_artifacts(&program, &options, HasherKind::Rpx).expect("W-LFM artifacts");
    let lfm = crate::lfm::whir_proof::airs_for(&build.artifacts, &options);
    let vm = vm_airs();
    let named: Vec<(DynAir<'_>, String)> = every_air(&vm, &lfm)
        .into_iter()
        .filter(|(_, label)| THE_THREE.contains(&label.as_str()))
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
    let arenas = vec![
        (0..4u64)
            .map(|i| core::array::from_fn(|j| FE::from(1_000 * (i + 1) + j as u64)))
            .collect(),
    ];
    lfm_prove_whir(&program, &build, &arenas, &options).expect("TrivialV0 proves under W-LFM");
    argue_capture::arm(&[]);

    let mut seen: Vec<String> = Vec::new();
    for captured in argue_capture::take() {
        let captured = captured
            .downcast::<Captured<Gl, Ext3>>()
            .expect("a Goldilocks table");
        let num_vars = captured.columns[0].num_vars();
        // Which of the three: the same layout, the same constraint DAG.
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
        ["CPU", "KECCAK_RND", "LFM_HASH"],
        "every one of the three captured"
    );
}

// ── Q1b: the card (`multilinear::gpu_fused`) ─────────────────────────────────

/// ★ The blob the fused kernels walk is the constraint part: on every VM table
/// and W-LFM chip, the lowered program with its `ACC` steps, walked as the
/// grid kernel walks it, gives today's zerocheck rule (weight one) at random
/// rows. Host only — this is what makes the card's walk checkable without one.
#[test]
fn the_fused_blob_is_the_constraint_part_on_every_table() {
    use multilinear::gpu_fused::{lower_fused, run_lowered_host};
    let vm = vm_airs();
    let lfm = w_lfm_airs();
    for (at, (air, label)) in every_air(&vm, &lfm).into_iter().enumerate() {
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

/// ★ Q1b device parity (FAST2): the fused rounds on the card send today's
/// messages — the three at 2^10 and 2^12, every table at 2^8 — over random
/// columns, the corners summed (random columns break every AIR), with the
/// cross-check on: today's device rounds replay the fused challenges and must
/// agree. Each run must have taken the card (`fused_sessions`) and been
/// confirmed (`fused_xchecks`).
#[cfg(feature = "cuda")]
#[test]
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
    let lfm = w_lfm_airs();
    let mut runs: Vec<(String, usize)> = every_air(&vm, &lfm)
        .iter()
        .map(|(_, label)| (label.clone(), 8))
        .collect();
    for label in THE_THREE {
        runs.push((label.to_string(), 10));
        runs.push((label.to_string(), 12));
    }
    let airs = every_air(&vm, &lfm);
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

/// ★ S1-1 on today's kernel (`LAMBDA_VM_ARGUE_INT_NODES`): the device rounds
/// with integer nodes send today's messages (the three at 2^10, fused off).
#[cfg(feature = "cuda")]
#[test]
fn the_int_node_rounds_send_todays_messages() {
    let _switches = CardSwitches::set(Default::default(), false);
    multilinear::gpu_fused::force_argue_fused(Some(false));
    math_cuda::sumcheck::force_int_nodes(Some(true));
    let vm = vm_airs();
    let lfm = w_lfm_airs();
    for (at, (air, label)) in every_air(&vm, &lfm)
        .into_iter()
        .filter(|(_, l)| THE_THREE.contains(&l.as_str()))
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
/// (the three at 2^10). Mutation `LAMBDA_VM_ARGUE_FUSED_MUTATE=xcheck` makes the
/// comparison inert, and this test must then FAIL.
#[cfg(feature = "cuda")]
#[test]
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
    let lfm = w_lfm_airs();
    for (at, (air, label)) in every_air(&vm, &lfm)
        .into_iter()
        .filter(|(_, l)| THE_THREE.contains(&l.as_str()))
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
/// random columns, the corners checked rather than summed (the three at
/// 2^10). Mutation `LAMBDA_VM_ARGUE_FUSED_MUTATE=corners` makes it inert, and
/// this test must then FAIL.
#[cfg(feature = "cuda")]
#[test]
fn the_corner_check_on_the_card_refuses_a_broken_trace() {
    use multilinear::gpu_fused::*;
    let _switches = CardSwitches::set(FusedFaults::default(), true);
    let vm = vm_airs();
    let lfm = w_lfm_airs();
    for (at, (air, label)) in every_air(&vm, &lfm)
        .into_iter()
        .filter(|(_, l)| THE_THREE.contains(&l.as_str()))
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

// ── census and op report (printing) ──────────────────────────────────────────

/// Stand-ins for a batch's challenges and points at `num_vars`, and today's
/// batch as the head walks it: on demand when it is big.
struct AtHeight {
    r: Vec<EE>,
    claim_point: Vec<EE>,
    lambdas: Vec<EE>,
    big: bool,
}

fn at_height(table: &Table, num_vars: usize, rng: &mut Rng) -> AtHeight {
    let r = rng.exts(num_vars);
    let claim_point = rng.exts(logup::input_layer_vars(table.interactions.len(), num_vars));
    let lambdas = multilinear::challenge_powers(&rng.ext(), 3);
    let statement = Statement {
        constraints: &table.constraints,
        betas: &table.betas,
        interactions: &table.interactions,
        claim_point: &claim_point,
        r: &r,
    };
    let batch = fused::batch_program(&statement, num_vars, table.width(), &lambdas).unwrap();
    let big = multilinear::gpu::is_big_batch(
        multilinear::gpu::lower(&batch)
            .expect("the batch lowers")
            .num_slots,
    );
    AtHeight {
        r,
        claim_point,
        lambdas,
        big,
    }
}

fn mix_line(mix: &Mix) -> String {
    format!(
        "{:6} steps (var {:5} fixed {:5} add/sub {:6} mul {:6} neg {:4})",
        mix.steps(),
        mix.var,
        mix.fixed,
        mix.add + mix.sub,
        mix.mul,
        mix.neg
    )
}

/// Census (ignored): per VM table and W-LFM chip, the batch split into its
/// constraint part and its bus part, each today and on demand; `d_C`, `D`,
/// the roots; the bus's sides, terms and distinct factors; the factors each
/// part reads; and the precondition.
///
/// `cargo test -p lambda-vm-prover --lib tests::argue_stage1_tests::the_stage1_census -- --ignored --nocapture`
#[test]
#[ignore = "a printing census for D-ARGUE's stage 1, not a gate"]
fn the_stage1_census() {
    const NUM_VARS: usize = 10;
    let vm = vm_airs();
    let lfm = w_lfm_airs();
    for (at, (air, label)) in every_air(&vm, &lfm).into_iter().enumerate() {
        let mut rng = Rng::new(at as u64);
        let table = Table::from_air(air, &label, NUM_VARS, &mut rng);
        let height = at_height(&table, NUM_VARS, &mut rng);
        let (weight_r, weight_z) = weight_slots(table.width());
        let constraint = table.shape.program(&table.betas, weight_r).unwrap();
        let bus =
            logup::claim_statements(&table.interactions, &height.claim_point, NUM_VARS, weight_z)
                .unwrap();
        let bus_part = multilinear::program::combine(
            &[
                bus.numerator.program().unwrap(),
                bus.denominator.program().unwrap(),
            ],
            &height.lambdas[1..],
        )
        .unwrap();
        let statement = Statement {
            constraints: &table.constraints,
            betas: &table.betas,
            interactions: &table.interactions,
            claim_point: &height.claim_point,
            r: &height.r,
        };
        let batch =
            fused::batch_program(&statement, NUM_VARS, table.width(), &height.lambdas).unwrap();
        let column = fused::column_program(&statement, table.width()).unwrap();
        let sides = 2 * table.interactions.len();
        let terms: usize = table
            .interactions
            .iter()
            .map(|i| i.numerator.terms().len() + i.denominator.terms().len())
            .sum();
        let l = fused::BusColumn::new(
            &table.interactions,
            &height.claim_point[..height.claim_point.len() - NUM_VARS],
            &height.lambdas[1],
            &height.lambdas[2],
        )
        .unwrap();
        let reads = table.constraints.reads();
        let bus_only = l
            .terms()
            .iter()
            .filter(|(slot, _)| reads.binary_search(slot).is_err())
            .count();
        let d = table.constraints.degree();
        println!(
            "{label:12} width {:4} · factors {:4} · roots {:4} ({:3} selected) · d_C {d} · D {} · {} batch · \
             bus: {sides:5} sides, {terms:5} terms, {:4} factors ({bus_only:4} only the bus's) · \
             constraint reads {:4} · precondition {}",
            air.trace_layout().0,
            table.width(),
            table.shape.num_roots(),
            table.constraints.selected(),
            (d + 1).max(2),
            if height.big { "BIG" } else { "small" },
            l.terms().len(),
            reads.len(),
            if not_base(air).is_empty() {
                "holds"
            } else {
                "FAILS"
            },
        );
        for (part, program) in [
            ("constraint", &constraint),
            ("bus", &bus_part),
            ("batch", &batch),
            ("S1-2 batch", &column),
        ] {
            println!(
                "{label:12}   {part:10} today {} · on demand {}",
                mix_line(&Mix::of(program.steps())),
                mix_line(&Mix::of(program.on_demand().steps())),
            );
        }
        println!(
            "{label:12}   {:10} {}",
            "base DAG",
            mix_line(&Mix::of(table.constraints.steps()))
        );
    }
}

/// The height each table is priced at: the head's, from job 241's epochs
/// (D-ARGUE §1.2) for the three, and a nominal 2^20 for the rest. The ratios
/// are per row, so a height moves them only through each round's fixed part.
fn head_height(label: &str) -> usize {
    match label {
        "KECCAK_RND" => 16,
        "CPU" => 21,
        "LFM_HASH" => 19,
        _ => 20,
    }
}

/// Op report (ignored): per table at its head height, today's modelled work
/// per row and each stage-1 variant's, in D-ARGUE §2.8's units, and the ratio
/// `R_model` of stage 1 to today — interpreted (stage 1's kernels as
/// interpreters) and compiled (stage 2). The three Q1 tables also print their
/// phases.
///
/// `cargo test -p lambda-vm-prover --lib tests::argue_stage1_tests::the_stage1_op_report -- --ignored --nocapture`
#[test]
#[ignore = "a printing op report for D-ARGUE's Q1a, not a gate"]
fn the_stage1_op_report() {
    let vm = vm_airs();
    let lfm = w_lfm_airs();
    println!(
        "{:12} {:>3} | {:>9} {:>9} | {:>9} {:>9} {:>9} {:>9} | {:>9} | {:>6} {:>6} | {:>7} {:>7}",
        "table",
        "n",
        "today",
        "+S1-1",
        "+bus",
        "+gruen",
        "+grid",
        "grid/int",
        "compiled",
        "R_int",
        "R_cmp",
        "B today",
        "B grid"
    );
    for (at, (air, label)) in every_air(&vm, &lfm).into_iter().enumerate() {
        let n = head_height(&label);
        let mut rng = Rng::new(0x0905 + at as u64);
        let table = Table::from_air(air, &label, n, &mut rng);
        let height = at_height(&table, n, &mut rng);
        let statement = Statement {
            constraints: &table.constraints,
            betas: &table.betas,
            interactions: &table.interactions,
            claim_point: &height.claim_point,
            r: &height.r,
        };
        let shape = Shape::of(&statement, table.width(), &height.lambdas, height.big).unwrap();
        let rows = (1u64 << n) as f64;
        let count = |variant: Variant| {
            fused::model(
                &shape,
                Options {
                    on_demand: height.big,
                    skip_corners: true,
                    ..Options::new(variant)
                },
            )
        };
        let (today, bus, gruen, grid) = (
            count(Variant::Today),
            count(Variant::Bus),
            count(Variant::Gruen),
            count(Variant::Grid),
        );
        let per_row = |c: &Counters, int_nodes: bool, interpreted: bool| {
            c.units(int_nodes, interpreted) / rows
        };
        let today_int = per_row(&today, false, true);
        let grid_int = per_row(&grid, true, true);
        let grid_cmp = per_row(&grid, true, false);
        println!(
            "{label:12} {n:>3} | {:>9.0} {:>9.0} | {:>9.0} {:>9.0} {:>9.0} {:>9.0} | {:>9.0} | {:>6.3} {:>6.3} | {:>7.0} {:>7.0}",
            today_int,
            per_row(&today, true, true),
            per_row(&bus, true, true),
            per_row(&gruen, true, true),
            grid_int,
            per_row(&grid, false, true),
            grid_cmp,
            grid_int / today_int,
            grid_cmp / today_int,
            today.total().bytes as f64 / rows,
            grid.total().bytes as f64 / rows,
        );
        if THE_THREE.contains(&label.as_str()) {
            for (name, counters) in [("today", &today), ("grid", &grid)] {
                for (phase, ops) in counters.phases() {
                    if *ops == fused::Ops::default() {
                        continue;
                    }
                    println!(
                        "    {label} {name:5} {phase:10} {:>10.1} u/row (interp) {:>10.1} u/row (compiled) \
                         · base mul {:>6.2} add {:>6.2} · ext mul {:>6.2} add {:>6.2} · mixed {:>6.2} · \
                         node reads {:>7.2} · steps {:>8.1} · {:>7.1} B · per row",
                        ops.units(name != "today", true) / rows,
                        ops.units(name != "today", false) / rows,
                        ops.base_mul as f64 / rows,
                        ops.base_add as f64 / rows,
                        ops.ext_mul as f64 / rows,
                        ops.ext_add as f64 / rows,
                        ops.mixed_mul as f64 / rows,
                        ops.node_reads as f64 / rows,
                        ops.steps as f64 / rows,
                        ops.bytes as f64 / rows,
                    );
                }
            }
        }
    }
}
