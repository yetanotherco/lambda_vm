//! Milestone A suite: the software layer round-trips — build → compile →
//! validate → execute — plus the negative paths (validator rejections,
//! executor runtime checks, compiler invariant panics).

use math::field::traits::IsPrimeField;

use crate::tables::types::{FE, FEE, GoldilocksField};

use super::builder::{LfmBuilder, LfmProgramSource};
use super::compiler::{LfmProgram, compile};
use super::executor::{LfmExecError, LfmExecution, execute};
use super::hash::{LfmHasher, TestPermutation};
use super::instr::{Addr, HashMode, Instr};
use super::layout;
use super::validator::{LfmViolation, validate};
use super::word::{LfmWord, base_word};

const GOLDILOCKS_P: u64 = 0xFFFF_FFFF_0000_0001;

fn fe(v: u64) -> FE {
    FE::from(v)
}

fn ext(a: u64, b: u64, c: u64) -> FEE {
    FEE::new([fe(a), fe(b), fe(c)])
}

fn run(program: &LfmProgram, arenas: &[Vec<LfmWord>]) -> LfmExecution {
    validate(program).expect("valid program");
    execute(program, arenas, &TestPermutation).expect("execution succeeds")
}

fn cell(exec: &LfmExecution, addr: Addr) -> LfmWord {
    exec.memory[addr.0 as usize].expect("cell written")
}

fn base_at(exec: &LfmExecution, addr: Addr) -> FE {
    super::word::word_as_base(&cell(exec, addr)).expect("base word")
}

fn ext_at(exec: &LfmExecution, addr: Addr) -> FEE {
    super::word::word_as_ext(&cell(exec, addr)).expect("ext word")
}

// ---- base ALU ----

#[test]
fn base_alu_round_trip() {
    let mut b = LfmBuilder::new();
    let x = b.felt_const(fe(7));
    let y = b.felt_const(fe(5));
    let s = b.add(x, y);
    let d = b.sub(x, y);
    let m = b.mul(x, y);
    let q = b.div(m, y);
    let h = b.mul_add(x, y, s); // 7·5 + 12 = 47
    let program = compile(b.finish());
    let exec = run(&program, &[]);
    assert_eq!(base_at(&exec, s.addr()), fe(12));
    assert_eq!(base_at(&exec, d.addr()), fe(2));
    assert_eq!(base_at(&exec, m.addr()), fe(35));
    assert_eq!(base_at(&exec, q.addr()), fe(7));
    assert_eq!(base_at(&exec, h.addr()), fe(47));
}

#[test]
fn div_zero_conventions() {
    // 0/0 = 1 (the assert mechanism's accepting case).
    let mut b = LfmBuilder::new();
    let z = b.felt_const(FE::zero());
    let q = b.div(z, z);
    let program = compile(b.finish());
    let exec = run(&program, &[]);
    assert_eq!(base_at(&exec, q.addr()), FE::one());

    // x/0 with x ≠ 0 errors.
    let mut b = LfmBuilder::new();
    let x = b.felt_const(fe(3));
    let z = b.felt_const(FE::zero());
    let _ = b.div(x, z);
    let program = compile(b.finish());
    validate(&program).expect("structurally valid");
    let err = execute(&program, &[], &TestPermutation).unwrap_err();
    assert!(matches!(err, LfmExecError::DivByZero { .. }));
}

#[test]
fn assert_lowering_pass_and_fail() {
    let mut b = LfmBuilder::new();
    let x = b.felt_const(fe(6));
    let y = b.felt_const(fe(2));
    let three = b.felt_const(fe(3));
    let q = b.mul(y, three);
    b.assert_eq(x, q);
    let program = compile(b.finish());
    run(&program, &[]); // passes

    let mut b = LfmBuilder::new();
    let x = b.felt_const(fe(6));
    let y = b.felt_const(fe(5));
    b.assert_eq(x, y);
    let program = compile(b.finish());
    let err = execute(&program, &[], &TestPermutation).unwrap_err();
    assert!(matches!(err, LfmExecError::DivByZero { .. }));
}

// ---- Fp3 ALU ----

#[test]
fn ext_alu_matches_field_reference() {
    let av = ext(3, 11, 2026);
    let bv = ext(9, 1, 77);
    let cv = ext(5, 4, 3);
    let f = fe(13);

    let mut b = LfmBuilder::new();
    let a = b.ext_const(&av);
    let bb = b.ext_const(&bv);
    let c = b.ext_const(&cv);
    let s = b.eadd(a, bb);
    let d = b.esub(a, bb);
    let p = b.emul(a, bb);
    let q = b.ediv(p, bb);
    let ma = b.emul_add(a, bb, c);
    let fl = b.felt_const(f);
    let mb = b.emul_base(a, fl);
    let program = compile(b.finish());
    let exec = run(&program, &[]);

    assert_eq!(ext_at(&exec, s.addr()), &av + &bv);
    assert_eq!(ext_at(&exec, d.addr()), &av - &bv);
    assert_eq!(ext_at(&exec, p.addr()), &av * &bv);
    assert_eq!(ext_at(&exec, q.addr()), av.clone());
    assert_eq!(ext_at(&exec, ma.addr()), &av * &bv + &cv);
    let [a0, a1, a2] = *av.value();
    assert_eq!(
        ext_at(&exec, mb.addr()),
        FEE::new([&a0 * &f, &a1 * &f, &a2 * &f])
    );
}

#[test]
fn ext_assert_and_horner() {
    // Horner: evaluate 5x² + 3x + 7 at x = (0,1,0) (i.e. w) via mul_add.
    let x = ext(0, 1, 0);
    let mut b = LfmBuilder::new();
    let xv = b.ext_const(&x);
    let c2 = b.ext_const(&ext(5, 0, 0));
    let c1 = b.ext_const(&ext(3, 0, 0));
    let c0 = b.ext_const(&ext(7, 0, 0));
    let acc = b.emul_add(c2, xv, c1); // 5x + 3
    let acc = b.emul_add(acc, xv, c0); // 5x² + 3x + 7
    let expected = &(&(&ext(5, 0, 0) * &x) + &ext(3, 0, 0)) * &x + &ext(7, 0, 0);
    let ex = b.ext_const(&expected);
    b.assert_eq_ext(acc, ex);
    let program = compile(b.finish());
    let exec = run(&program, &[]);
    assert_eq!(ext_at(&exec, acc.addr()), expected);
}

// ---- select / bitdec ----

#[test]
fn select_swaps_on_bit() {
    let mut b = LfmBuilder::new();
    let l = b.felt_const(fe(100));
    let r = b.felt_const(fe(200));
    let b0 = b.bit_const(false);
    let b1 = b.bit_const(true);
    let (l0, r0) = b.select(b0, l.as_cell(), r.as_cell());
    let (l1, r1) = b.select(b1, l.as_cell(), r.as_cell());
    let program = compile(b.finish());
    let exec = run(&program, &[]);
    assert_eq!(cell(&exec, l0.addr()), base_word(fe(100)));
    assert_eq!(cell(&exec, r0.addr()), base_word(fe(200)));
    assert_eq!(cell(&exec, l1.addr()), base_word(fe(200)));
    assert_eq!(cell(&exec, r1.addr()), base_word(fe(100)));
}

#[test]
fn non_boolean_select_bit_rejected() {
    let mut b = LfmBuilder::new();
    let l = b.felt_const(fe(1));
    let r = b.felt_const(fe(2));
    let two = b.felt_const(fe(2));
    let (_, _) = b.select(super::builder::Bit(two.addr()), l.as_cell(), r.as_cell());
    let program = compile(b.finish());
    let err = execute(&program, &[], &TestPermutation).unwrap_err();
    assert!(matches!(err, LfmExecError::NonBooleanBit(_)));
}

#[test]
fn bit_dec_edge_values() {
    // Canonical decomposition + the p-specific gadget witnesses at the edges.
    for v in [
        0u64,
        1,
        (1 << 32) - 1,
        GOLDILOCKS_P - 1,
        0x1234_5678_9ABC_DEF0,
    ] {
        let mut b = LfmBuilder::new();
        let x = b.felt_const(fe(v));
        let bits = b.bit_dec(x, 64);
        let program = compile(b.finish());
        let exec = run(&program, &[]);
        for (i, bit) in bits.iter().enumerate() {
            assert_eq!(
                base_at(&exec, bit.addr()),
                fe((v >> i) & 1),
                "bit {i} of {v:#x}"
            );
        }
        let row = &exec.records.bitdec[0];
        let top = (v >> 32) as u32;
        if top == u32::MAX {
            assert_eq!(row.z, FE::one(), "z for {v:#x}");
            assert_eq!(row.ginv, FE::zero());
        } else {
            assert_eq!(row.z, FE::zero(), "z for {v:#x}");
            let g = fe(0xFFFF_FFFFu64 - top as u64);
            assert_eq!(&row.ginv * &g, FE::one(), "ginv·g = 1 for {v:#x}");
        }
    }
}

#[test]
fn bit_dec_partial_width_allocates_only_requested_cells() {
    let mut b = LfmBuilder::new();
    let x = b.felt_const(fe(0b1011));
    let bits = b.bit_dec(x, 4);
    let program = compile(b.finish());
    let exec = run(&program, &[]);
    assert_eq!(bits.len(), 4);
    let vals: Vec<FE> = bits.iter().map(|bit| base_at(&exec, bit.addr())).collect();
    assert_eq!(vals, vec![fe(1), fe(1), fe(0), fe(1)]);
    // All 64 witness bits still recorded for the constraint columns.
    assert_eq!(
        exec.records.bitdec[0].bits[4..]
            .iter()
            .filter(|b| **b == FE::one())
            .count(),
        0
    );
}

// ---- hash ----

#[test]
fn hash_compress_and_permute_match_reference() {
    let hasher = TestPermutation;
    let a: LfmWord = core::array::from_fn(|i| fe(10 + i as u64));
    let c: LfmWord = core::array::from_fn(|i| fe(20 + i as u64));

    let mut b = LfmBuilder::new();
    let da = b.digest_const(a);
    let dc = b.digest_const(c);
    let d = b.compress(da, dc);
    let s0 = b.digest_const(core::array::from_fn(|i| fe(30 + i as u64)));
    let s1 = b.digest_const(core::array::from_fn(|i| fe(40 + i as u64)));
    let s2 = b.digest_const(core::array::from_fn(|i| fe(50 + i as u64)));
    let out = b.permute([s0.as_cell(), s1.as_cell(), s2.as_cell()]);
    let program = compile(b.finish());
    let exec = run(&program, &[]);

    assert_eq!(cell(&exec, d.addr()), hasher.compress(&a, &c));

    let mut state: [FE; 12] = core::array::from_fn(|_| FE::zero());
    for i in 0..4 {
        state[i] = fe(30 + i as u64);
        state[4 + i] = fe(40 + i as u64);
        state[8 + i] = fe(50 + i as u64);
    }
    let expected = hasher.permute(state);
    for (j, o) in out.iter().enumerate() {
        let w = cell(&exec, o.addr());
        for l in 0..4 {
            assert_eq!(w[l], expected[4 * j + l]);
        }
    }
}

// ---- hints / public ----

#[test]
fn hint_and_public_round_trip() {
    let w0: LfmWord = core::array::from_fn(|i| fe(100 + i as u64));
    let w1: LfmWord = core::array::from_fn(|i| fe(200 + i as u64));

    let mut b = LfmBuilder::new();
    let arena = b.declare_arena(2);
    let h0 = b.hint_word(arena, 0);
    let h1 = b.hint_word(arena, 1);
    b.public(h0);
    b.public(h1);
    let program = compile(b.finish());
    let exec = run(&program, &[vec![w0, w1]]);

    assert_eq!(cell(&exec, h0.addr()), w0);
    assert_eq!(exec.public_words, vec![(0, w0), (1, w1)]);
}

#[test]
fn arena_out_of_bounds_rejected_by_validator_and_executor() {
    let mut b = LfmBuilder::new();
    let arena = b.declare_arena(2);
    let h = b.hint_word(arena, 5); // past the declared length
    b.public(h);
    let program = compile(b.finish());
    assert_eq!(
        validate(&program).unwrap_err(),
        LfmViolation::ArenaOutOfBounds { arena: 0, index: 5 }
    );
    let err = execute(&program, &[vec![base_word(fe(1)); 2]], &TestPermutation).unwrap_err();
    assert!(matches!(err, LfmExecError::ArenaOutOfBounds { .. }));
}

// ---- validator negatives (via mutation of a valid program) ----

fn small_valid_program() -> LfmProgram {
    let mut b = LfmBuilder::new();
    let x = b.felt_const(fe(4));
    let y = b.felt_const(fe(9));
    let s = b.add(x, y);
    let p = b.mul(s, x);
    b.public(p.as_cell());
    compile(b.finish())
}

#[test]
fn validator_rejects_double_write() {
    let mut program = small_valid_program();
    // Point the mul's destination at the add's (already-written) cell.
    let add_out = program
        .instrs
        .iter()
        .find_map(|i| match i {
            Instr::BaseAlu {
                op: super::instr::BaseOp::Add,
                out,
                ..
            } => Some(*out),
            _ => None,
        })
        .unwrap();
    for i in &mut program.instrs {
        if let Instr::BaseAlu {
            op: super::instr::BaseOp::Mul,
            out,
            ..
        } = i
        {
            *out = add_out;
        }
    }
    assert_eq!(
        validate(&program).unwrap_err(),
        LfmViolation::DoubleWrite { addr: add_out.0 }
    );
    // The executor independently catches it.
    let err = execute(&program, &[], &TestPermutation).unwrap_err();
    assert_eq!(err, LfmExecError::DoubleWrite(add_out.0));
}

#[test]
fn validator_rejects_cycle() {
    let mut program = small_valid_program();
    for i in &mut program.instrs {
        if let Instr::BaseAlu {
            op: super::instr::BaseOp::Mul,
            out,
            a,
            ..
        } = i
        {
            *a = *out; // a := f(a) — balances for any value; must die here
        }
    }
    assert!(matches!(
        validate(&program).unwrap_err(),
        LfmViolation::CyclicRead { .. }
    ));
}

#[test]
fn validator_rejects_wrong_mult() {
    let mut program = small_valid_program();
    for i in &mut program.instrs {
        if let Instr::Const { mult, .. } = i {
            *mult += 1;
            break;
        }
    }
    assert!(matches!(
        validate(&program).unwrap_err(),
        LfmViolation::MultMismatch { .. }
    ));
}

#[test]
fn validator_rejects_non_one_hot_selector() {
    let mut program = small_valid_program();
    // Turn a second selector on in the first real BALU row.
    program.groups.balu.set(0, layout::balu::SEL_SUB, FE::one());
    assert_eq!(
        validate(&program).unwrap_err(),
        LfmViolation::NonOneHotSelector {
            chip: "LFM_BALU",
            row: 0
        }
    );
}

#[test]
fn validator_rejects_dirty_padding() {
    let mut program = small_valid_program();
    let row = program.groups.balu.real_rows; // first padding row
    program.groups.balu.set(row, layout::balu::MULT, fe(1));
    assert_eq!(
        validate(&program).unwrap_err(),
        LfmViolation::DirtyPadding {
            chip: "LFM_BALU",
            row
        }
    );
}

/// Check 9 at the level it operates: a multiplicity column of the COMMITTED
/// group, with the instruction list left honest.
///
/// `p − 1` is what `−1` looks like as a canonical field element, and a
/// negative send is the one stray multiplicity the LogUp count argument cannot
/// catch: it cancels an honest write instead of adding an unmatched token.
#[test]
fn validator_rejects_negative_multiplicity() {
    let mut program = small_valid_program();
    // Honest-path control: the program is otherwise admissible, so the
    // rejection below is about the multiplicity and nothing else.
    validate(&program).expect("the untampered program must pass admission");

    program
        .groups
        .const_
        .set(0, layout::const_::MULT, fe(GOLDILOCKS_P - 1));
    assert!(
        matches!(
            validate(&program).unwrap_err(),
            LfmViolation::MultOutOfRange {
                chip: "LFM_CONST",
                row: 0,
                col: layout::const_::MULT,
                ..
            }
        ),
        "a field-negative multiplicity must fail admission"
    );
}

/// The bound is a bound, not merely a sign test: a multiplicity far above any
/// read count the program can emit is rejected even though it is positive.
#[test]
fn validator_rejects_oversized_multiplicity() {
    let mut program = small_valid_program();
    validate(&program).expect("the untampered program must pass admission");

    program.groups.balu.set(0, layout::balu::MULT, fe(1 << 40));
    assert!(matches!(
        validate(&program).unwrap_err(),
        LfmViolation::MultOutOfRange {
            chip: "LFM_BALU",
            row: 0,
            ..
        }
    ));
}

/// The forgery this pair of checks exists for: a `Compress` row whose two
/// *spare* output slots carry a `−1` / `+1` pair aimed at one address.
///
/// `Instr::writes()` hides those slots for `Compress`, so checks 1 and 4 never
/// look at them — yet `emit_column_groups` copies them into the committed
/// group and `chips::hash` sends all three slots gated only by their own
/// `MULT`, with no mode factor. The negative send cancels the victim cell's
/// honest write and the positive one re-supplies it with the row's own
/// permutation output, so the token count is preserved exactly and the reader
/// observes a word nobody committed. This was executed end to end against the
/// real prover and verifier, and accepted, before these checks existed.
#[test]
fn validator_rejects_compress_ghost_slot_forgery() {
    let konst = |out: u64, v: u64| Instr::Const {
        out: Addr(out),
        value: [fe(v), FE::zero(), FE::zero(), FE::zero()],
        mult: 0,
    };
    let source = LfmProgramSource {
        instrs: vec![
            konst(0, 1),
            konst(1, 2),
            konst(2, 3), // the victim: a program constant the ghost pair replaces
            Instr::Hash {
                mode: HashMode::Compress,
                ins: [Addr(0), Addr(1), Addr(0)],
                outs: [Addr(3), Addr(2), Addr(2)],
                mults: [0, GOLDILOCKS_P - 1, 1],
            },
            Instr::Public {
                addr: Addr(2),
                index: 0,
            },
        ],
        num_addrs: 4,
        read_counts: vec![1, 1, 1, 0],
        arena_schema: Default::default(),
        public_len: 1,
    };
    let mut program = compile(source);

    // Leg 1: the point check on `instr.rs`'s placeholder convention.
    assert_eq!(
        validate(&program).unwrap_err(),
        LfmViolation::CompressSlotNotPlaceholder { instr: 3 }
    );

    // Leg 2: and check 9 denies it on its own. Repair the INSTRUCTION list —
    // the object checks 1–4 read — and leave the COMMITTED group hostile,
    // which is precisely the divergence that made the forgery admissible.
    let Instr::Hash { outs, mults, .. } = &mut program.instrs[3] else {
        panic!("instruction 3 is the hash row");
    };
    *outs = [Addr(3), Addr(0), Addr(0)];
    *mults = [0, 0, 0];
    assert!(matches!(
        validate(&program).unwrap_err(),
        LfmViolation::MultOutOfRange {
            chip: "LFM_HASH",
            row: 0,
            col: layout::hash::MULT1,
            ..
        }
    ));
}

#[test]
fn validator_rejects_read_of_unwritten() {
    let mut program = small_valid_program();
    let bogus = Addr(program.num_addrs - 1); // allocated range, but rewire below
    // Extend the address space by one and point an operand at the unwritten slot.
    program.num_addrs += 1;
    let unwritten = Addr(program.num_addrs - 1);
    for i in &mut program.instrs {
        if let Instr::BaseAlu {
            op: super::instr::BaseOp::Mul,
            b,
            ..
        } = i
        {
            *b = unwritten;
        }
    }
    let _ = bogus;
    assert_eq!(
        validate(&program).unwrap_err(),
        LfmViolation::ReadOfUnwritten { addr: unwritten.0 }
    );
}

// ---- compiler invariant panics (tripwires behind the validator) ----

#[test]
fn compiler_panics_on_double_assignment() {
    let source = LfmProgramSource {
        instrs: vec![
            Instr::Const {
                out: Addr(0),
                value: [FE::zero(), FE::zero(), FE::zero(), FE::zero()],
                mult: 0,
            },
            Instr::Const {
                out: Addr(0),
                value: [FE::one(), FE::zero(), FE::zero(), FE::zero()],
                mult: 0,
            },
        ],
        num_addrs: 1,
        read_counts: vec![0],
        arena_schema: Default::default(),
        public_len: 0,
    };
    let result = std::panic::catch_unwind(|| compile(source));
    assert!(result.is_err());
}

#[test]
fn compiler_panics_on_undrained_read_counts() {
    let mut read_counts = vec![0u64; 8];
    read_counts[7] = 1; // a read of an address nothing writes
    let source = LfmProgramSource {
        instrs: vec![Instr::Const {
            out: Addr(0),
            value: [FE::zero(), FE::zero(), FE::zero(), FE::zero()],
            mult: 0,
        }],
        num_addrs: 8,
        read_counts,
        arena_schema: Default::default(),
        public_len: 0,
    };
    let result = std::panic::catch_unwind(|| compile(source));
    assert!(result.is_err());
}

// ---- misc structural ----

#[test]
fn const_pool_interns_and_counts_reads() {
    let mut b = LfmBuilder::new();
    let x1 = b.felt_const(fe(42));
    let x2 = b.felt_const(fe(42)); // same cell
    assert_eq!(x1.addr(), x2.addr());
    let s = b.add(x1, x2); // two reads of the shared cell
    b.public(s.as_cell());
    let program = compile(b.finish());
    let mult = program
        .instrs
        .iter()
        .find_map(|i| match i {
            Instr::Const { out, mult, .. } if *out == x1.addr() => Some(*mult),
            _ => None,
        })
        .unwrap();
    assert_eq!(mult, 2);
    run(&program, &[]);
}

#[test]
fn canonical_helper_sanity() {
    // p ≡ 0: the canonical map the executor's BitDec relies on.
    assert_eq!(GoldilocksField::canonical(fe(GOLDILOCKS_P).value()), 0);
    assert_eq!(GoldilocksField::canonical(fe(5).value()), 5);
}

#[test]
fn digest_packing_round_trips() {
    let w: LfmWord = core::array::from_fn(|i| fe(0xDEAD_0000 + i as u64));
    let packed = super::word::pack_digest(&w);
    assert_eq!(super::word::unpack_digest(&packed), w);
}
