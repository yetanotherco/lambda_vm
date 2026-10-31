//! Tests for the ECREC table: every constraint holds on generated traces (computed
//! rows, bailed rows, padding), and isolated negative checks, one per check the
//! table carries: the parity of R's root, the scalar relations, the bounds, the
//! `xA ≠ xB` witness, the zero key on a bail, the scalar bits off a computed row.

use crate::tables::ecrec::{EcrecConstraints, EcrecOperation, NUM_CONSTRAINTS, cols};
use crate::tables::types::{FE, GoldilocksExtension, GoldilocksField};
use ecsm::{N_BYTES, P_BYTES, compute_ecrec_witness};
use math::field::element::FieldElement;
use stark::constraints::builder::{ConstraintSet, ProverEvalFolder};
use stark::frame::Frame;
use stark::table::TableView;
use stark::traits::TransitionEvaluationContext;

fn hex32(s: &str) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, b) in out.iter_mut().enumerate() {
        *b = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap();
    }
    out
}

/// A signature `r ‖ s`, its hash, the parity of R's y, and its key's x and y, all
/// big-endian.
type Vector = ([u8; 64], [u8; 32], u8, [u8; 32], [u8; 32]);

/// Two fixed signatures and their keys.
fn vectors() -> Vec<Vector> {
    let v = |r: &str, s: &str, z: &str, par: u8, qx: &str, qy: &str| {
        let mut sig = [0u8; 64];
        sig[..32].copy_from_slice(&hex32(r));
        sig[32..].copy_from_slice(&hex32(s));
        (sig, hex32(z), par, hex32(qx), hex32(qy))
    };
    vec![
        v(
            "3284bae4a1b693dfbe8ac67400e4891f3b53681c5159a315561051f3d2c57139",
            "97d4b2f2f728a54ed562a12237f6b11d32d626e7fcc24c3470b0729e095bd057",
            "c0ffee0011223344556677889900aabbccddeeff00112233445566778899aabb",
            0,
            "8733750fdf82edb525a2c9f9d201ad1b26e163b8093f3c8f6eb83377d157edb2",
            "ce023840e38085bfcf2b27bd66be6a877f0ec281b073ca0f9ab31cfc0733b62a",
        ),
        v(
            "7962d45b38e8bcf82fa8efa8432a01f20c9a53e24c7d3f11df197cb8e70926da",
            "ecec6811812f04794c1f92be72948f95a6cda7874445a39db40aab31c835f2cd",
            "0000000000000000000000000000000000000000000000000000000000000042",
            1,
            "76d2fdf1302d1fa9556f4df94ec84cefba6d482e54f47c6c2a238c1baa560f0e",
            "b754ac7e7a3e09c44184cb451a4f5fb557f32053eb015dffebb655b5cfd54d8a",
        ),
    ]
}

fn op(sig: &[u8; 64], msg: &[u8; 32], parity: u8) -> EcrecOperation {
    EcrecOperation {
        timestamp: 444,
        addr_out: 0x1000,
        addr_sig: 0x2000,
        addr_msg: 0x3000,
        witness: compute_ecrec_witness(sig, msg, parity),
    }
}

fn eval(main: Vec<FE>) -> Vec<FE> {
    let n = EcrecConstraints.meta().len();
    let frame = Frame::<GoldilocksField, GoldilocksExtension>::new(vec![TableView::new(
        vec![main],
        vec![vec![]],
    )]);
    let no_e: Vec<FieldElement<GoldilocksExtension>> = vec![];
    let offset_e = FieldElement::<GoldilocksExtension>::zero();
    let ctx =
        TransitionEvaluationContext::new_prover(frame.as_row_frame(), &no_e, &no_e, &offset_e);
    let mut base = vec![FE::zero(); n];
    let mut ext = vec![FieldElement::<GoldilocksExtension>::zero(); n];
    let mut folder = ProverEvalFolder::new(&ctx, &mut base, &mut ext);
    EcrecConstraints.eval(&mut folder);
    base
}

/// Row `row` of the table of `ops`, as values.
fn row_of(ops: &[EcrecOperation], row: usize) -> Vec<FE> {
    let trace = crate::tables::ecrec::generate_ecrec_trace(ops);
    (0..cols::NUM_COLUMNS)
        .map(|c| *trace.main_table.get(row, c))
        .collect()
}

fn failing(main: Vec<FE>) -> Vec<usize> {
    let v = eval(main);
    (0..v.len()).filter(|&i| v[i] != FE::zero()).collect()
}

/// A computed row of the first vector.
fn honest() -> Vec<FE> {
    let (sig, msg, par, ..) = vectors()[0];
    row_of(&[op(&sig, &msg, par)], 0)
}

fn set_bytes(main: &mut [FE], col: usize, bytes: &[u8]) {
    for (i, &b) in bytes.iter().enumerate() {
        main[col + i] = FE::from(b as u64);
    }
}

#[test]
fn constraint_count() {
    assert_eq!(EcrecConstraints.meta().len(), NUM_CONSTRAINTS);
    assert_eq!(NUM_CONSTRAINTS, 981);
}

/// Every constraint holds on a table of computed rows, bailed rows and padding,
/// and the computed rows' keys are the vectors'.
#[test]
fn constraints_hold_on_generated_trace() {
    let mut ops = Vec::new();
    for (sig, msg, par, qx, qy) in vectors() {
        let o = op(&sig, &msg, par);
        assert!(o.witness.ok);
        let (mut x, mut y) = (o.witness.x_q, o.witness.y_q);
        x.reverse();
        y.reverse();
        assert_eq!((x, y), (qx, qy), "the witness's key is the vector's");
        ops.push(o);
        // The other parity: a different R and key, still a computed row.
        ops.push(op(&sig, &msg, par ^ 1));
    }
    // Bailed rows: r = 0, s = N, z = 0.
    let (sig, msg, par, ..) = vectors()[0];
    let mut zero_r = sig;
    zero_r[..32].fill(0);
    let mut n_s = sig;
    let mut n_be = N_BYTES;
    n_be.reverse();
    n_s[32..].copy_from_slice(&n_be);
    for (s, m) in [(zero_r, msg), (n_s, msg), (sig, [0u8; 32])] {
        let o = op(&s, &m, par);
        assert!(!o.witness.ok);
        ops.push(o);
    }
    let trace = crate::tables::ecrec::generate_ecrec_trace(&ops);
    assert!(
        trace.num_rows() > ops.len(),
        "the table has padding rows too"
    );
    for row in 0..trace.num_rows() {
        let main: Vec<FE> = (0..cols::NUM_COLUMNS)
            .map(|c| *trace.main_table.get(row, c))
            .collect();
        let bad = failing(main);
        assert!(bad.is_empty(), "row {row}: constraints {bad:?}");
    }
}

/// The other root of R: no `H` makes `yR[0] = 2·H + PAR` hold for the opposite
/// parity, so R's sign is pinned by the x13 value.
#[test]
fn the_other_root_of_r_is_refused() {
    let base = honest();
    let par = base[cols::PAR];
    for h in 0..256u64 {
        let mut main = base.clone();
        main[cols::H] = FE::from(h);
        // Flip the parity bit the row claims; yR stays.
        main[cols::PAR] = FE::one() - par;
        assert!(failing(main).contains(&843), "H = {h}");
    }
    // The honest row's parity constraint holds.
    assert_eq!(eval(base)[843], FE::zero());
}

/// A wrong inverse, a wrong scalar bit of u1 or u2: the relation's carries no longer
/// close (some constraint of that relation fires).
#[test]
fn wrong_scalars_are_refused() {
    let base = honest();
    let mut w = base.clone();
    w[cols::W] += FE::one();
    assert!(failing(w).iter().any(|&i| (517..582).contains(&i)), "w");
    for (bits, range) in [(cols::U1, 582..647), (cols::U2, 647..712)] {
        let mut m = base.clone();
        m[bits + 9] = FE::one() - m[bits + 9];
        assert!(
            failing(m).iter().any(|i| range.contains(i)),
            "bits at {bits}"
        );
    }
}

/// `r ≥ N` on a computed row: the bound's overflow constraint fires.
#[test]
fn r_at_or_above_n_is_refused() {
    let mut main = honest();
    set_bytes(&mut main, cols::R, &N_BYTES);
    let bad = failing(main);
    // RLtN is the first bound: indices 844..852.
    assert!(bad.contains(&851), "{bad:?}");
}

/// `yR` and the key's coordinates must be below p: `p` itself is refused.
#[test]
fn coordinates_at_p_are_refused() {
    for (col, overflow) in [
        (cols::YR, 851 + 32),
        (cols::XQ, 851 + 56),
        (cols::YQ, 851 + 64),
    ] {
        let mut main = honest();
        set_bytes(&mut main, col, &P_BYTES);
        assert!(failing(main).contains(&overflow), "column {col}");
    }
}

/// `xA = xB` on a computed row (the chord undefined): no DINV satisfies the
/// non-degeneracy constraint.
#[test]
fn equal_chain_x_is_refused() {
    let mut main = honest();
    for i in 0..32 {
        main[cols::XB + i] = main[cols::XA + i];
    }
    for c in 0..8 {
        main[cols::DINV + c] = FE::from(12345u64 + c as u64);
    }
    assert!(failing(main).contains(&916));
}

/// A bail must write a zero key: OK = 0 with a key byte set fires.
#[test]
fn a_bail_writes_a_zero_key() {
    let (sig, msg, par, ..) = vectors()[0];
    let mut zero_r = sig;
    zero_r[..32].fill(0);
    let mut main = row_of(&[op(&zero_r, &msg, par)], 0);
    assert!(failing(main.clone()).is_empty(), "the honest bail holds");
    assert_eq!(main[cols::OK], FE::zero());
    main[cols::XQ + 5] = FE::from(7u64);
    assert!(failing(main).contains(&(917 + 5)));
}

/// Scalar bits on a row that computed nothing would fire phantom `Bit` receives.
#[test]
fn scalar_bits_off_a_computed_row_are_refused() {
    let (sig, msg, par, ..) = vectors()[0];
    let mut zero_r = sig;
    zero_r[..32].fill(0);
    let mut main = row_of(&[op(&zero_r, &msg, par)], 0);
    main[cols::U2 + 3] = FE::one();
    assert!(failing(main).contains(&516));
    // And OK on a padding row (MU = 0).
    let mut pad = vec![FE::zero(); cols::NUM_COLUMNS];
    pad[cols::OK] = FE::one();
    assert!(failing(pad).contains(&3));
}

/// The BITWISE lookups the builder counts for ECREC rows are exactly the ones the
/// table sends: every `AreBytes`, `IsHalfword` and `Zero` interaction evaluated on
/// every row (computed, bailed, padding), as a multiset.
#[test]
fn the_counted_lookups_are_the_tables_sends() {
    use crate::tables::bitwise::{BitwiseOperation, BitwiseOperationType};
    use crate::tables::types::BusId;
    use stark::lookup::Multiplicity;
    use std::collections::HashMap;

    let mut ops: Vec<EcrecOperation> = vectors()
        .iter()
        .flat_map(|(sig, msg, par, ..)| [op(sig, msg, *par), op(sig, msg, par ^ 1)])
        .collect();
    let (sig, msg, par, ..) = vectors()[1];
    let mut zero_r = sig;
    zero_r[..32].fill(0);
    ops.push(op(&zero_r, &msg, par));
    let trace = crate::tables::ecrec::generate_ecrec_trace(&ops);
    let mut sent: HashMap<BitwiseOperation, u64> = HashMap::new();
    for row in 0..trace.num_rows() {
        let get = |c: usize| *trace.main_table.get(row, c);
        for bi in crate::tables::ecrec::bus_interactions() {
            let kind = match bi.bus_id {
                id if id == BusId::AreBytes as u64 => 0,
                id if id == BusId::IsHalfword as u64 => 1,
                id if id == BusId::Zero as u64 => 2,
                _ => continue,
            };
            let Multiplicity::Column(mc) = bi.multiplicity else {
                panic!("a range check gated by a column");
            };
            let m = get(mc).canonical();
            if m == 0 {
                continue;
            }
            assert_eq!(m, 1);
            let vals: Vec<u64> = bi
                .values
                .iter()
                .flat_map(|v| v.combine_from(get))
                .map(|f| f.canonical())
                .collect();
            let lookup = match kind {
                0 => {
                    assert_eq!(vals[1], 0);
                    BitwiseOperation::byte_op(BitwiseOperationType::AreBytes, vals[0] as u8, 0)
                }
                1 => {
                    assert!(vals[0] < 1 << 16, "a halfword: {}", vals[0]);
                    BitwiseOperation::halfword(
                        BitwiseOperationType::IsHalf,
                        (vals[0] & 0xFF) as u8,
                        (vals[0] >> 8) as u8,
                    )
                }
                _ => {
                    assert_eq!(vals[1], 0, "a nonzero check");
                    BitwiseOperation::zero(vals[0] as u32)
                }
            };
            *sent.entry(lookup).or_default() += 1;
        }
    }
    let mut counted: HashMap<BitwiseOperation, u64> = HashMap::new();
    for l in crate::tables::trace_builder::collect_bitwise_from_ecrec(&ops) {
        *counted.entry(l).or_default() += 1;
    }
    assert!(!sent.is_empty());
    assert_eq!(sent, counted);
}

/// The swapped-chains forgery the block test refuses on the bus alone: its ECREC row
/// satisfies every ECREC constraint, and its chains' and addition's ECDAS rows every
/// ECDAS constraint — so the second chain's key is the only thing left to refuse it.
#[test]
fn the_swapped_chains_forgery_satisfies_every_constraint() {
    use crate::tables::ecdas::{EcdasConstraints, EcdasOperation, generate_ecdas_trace};
    let (sig, msg, par, ..) = vectors()[0];
    let mut forged = op(&sig, &msg, par);
    assert_eq!(
        forged.witness.r,
        crate::tables::trace_builder::FORGED_ECREC_R
    );
    let honest_key = (forged.witness.x_q, forged.witness.y_q);
    crate::tables::trace_builder::swap_ecrec_chains(&mut forged.witness);
    assert_ne!((forged.witness.x_q, forged.witness.y_q), honest_key);
    let row = row_of(std::slice::from_ref(&forged), 0);
    assert!(failing(row).is_empty(), "the forged ECREC row holds");

    let w = &forged.witness;
    let steps: Vec<EcdasOperation> = w
        .chain1
        .iter()
        .chain(&w.chain2)
        .chain(w.add.as_ref())
        .map(|step| EcdasOperation {
            timestamp: 444,
            step: step.clone(),
        })
        .collect();
    let trace = generate_ecdas_trace(&steps);
    let n = EcdasConstraints.meta().len();
    for r in 0..trace.num_rows() {
        let main: Vec<FE> = (0..crate::tables::ecdas::cols::NUM_COLUMNS)
            .map(|c| *trace.main_table.get(r, c))
            .collect();
        let frame = Frame::<GoldilocksField, GoldilocksExtension>::new(vec![TableView::new(
            vec![main],
            vec![vec![]],
        )]);
        let no_e: Vec<FieldElement<GoldilocksExtension>> = vec![];
        let offset_e = FieldElement::<GoldilocksExtension>::zero();
        let ctx =
            TransitionEvaluationContext::new_prover(frame.as_row_frame(), &no_e, &no_e, &offset_e);
        let mut base = vec![FE::zero(); n];
        let mut ext = vec![FieldElement::<GoldilocksExtension>::zero(); n];
        let mut folder = ProverEvalFolder::new(&ctx, &mut base, &mut ext);
        EcdasConstraints.eval(&mut folder);
        assert!(base.iter().all(|v| *v == FE::zero()), "ECDAS row {r}");
    }
}

/// The table's generator constants are G.
#[test]
fn the_generator_constants_are_g() {
    let (gx, gy) = ecsm::generator_le();
    let (sig, msg, par, ..) = vectors()[0];
    let w = op(&sig, &msg, par).witness;
    let first = &w.chain1[0];
    assert_eq!(
        (first.x_g, first.y_g),
        (gx, gy),
        "the first chain runs on G"
    );
    assert_eq!(
        crate::tables::ecrec::generator_le(),
        (gx, gy),
        "the table's seed and drain constants are G"
    );
}

// =============================================================================
// The `< N` bounds on s, u1 and u2 and the `< p` bounds on xA and xB: each is the
// only constraint refusing a row that is otherwise consistent (the relation's
// quotient and carries recomputed for the out-of-range value). Without u1 < N or
// u2 < N a prover takes the true scalar 2 (r = −z/2, or s = 2r), claims 2 + N, and
// the chain's last addition becomes G + G through ECDAS's chord branch, where λ
// is free; without s < N the signature (r, s + N), which Ethereum refuses,
// recovers the same key.
// =============================================================================

use crate::tables::ecrec::{Bound, CARRY_OFFSET_MODN};

/// Index of `bound`'s closing-carry constraint: the bounds are emitted in
/// `Bound::ALL` order from index 844, eight constraints each (seven carry bits,
/// then the carry out).
fn closing(bound: Bound) -> usize {
    let k = Bound::ALL
        .iter()
        .position(|b| std::mem::discriminant(b) == std::mem::discriminant(&bound))
        .expect("a bound");
    844 + 8 * k + 7
}

fn add_le(a: &[u8], b: &[u8], len: usize) -> Vec<u8> {
    let mut out = vec![0u8; len];
    let mut carry = 0u16;
    for (i, o) in out.iter_mut().enumerate() {
        let s = *a.get(i).unwrap_or(&0) as u16 + *b.get(i).unwrap_or(&0) as u16 + carry;
        *o = s as u8;
        carry = s >> 8;
    }
    assert_eq!(carry, 0, "add overflow");
    out
}

fn sub_le(a: &[u8], b: &[u8], len: usize) -> Vec<u8> {
    let mut out = vec![0u8; len];
    let mut borrow = 0i16;
    for (i, o) in out.iter_mut().enumerate() {
        let d = *a.get(i).unwrap_or(&0) as i16 - *b.get(i).unwrap_or(&0) as i16 - borrow;
        borrow = i16::from(d < 0);
        *o = d.rem_euclid(256) as u8;
    }
    assert_eq!(borrow, 0, "sub underflow");
    out
}

fn ge_le(a: &[u8], b: &[u8]) -> bool {
    let n = a.len().max(b.len());
    for i in (0..n).rev() {
        let (x, y) = (*a.get(i).unwrap_or(&0), *b.get(i).unwrap_or(&0));
        if x != y {
            return x > y;
        }
    }
    true
}

/// Big-endian of a little-endian 32-byte value.
fn be(le: &[u8]) -> [u8; 32] {
    std::array::from_fn(|i| le[31 - i])
}

/// `(2·x) mod N` for `x < N`, little-endian.
fn double_mod_n(x: &[u8; 32]) -> [u8; 32] {
    let d = add_le(x, x, 33);
    let r = if ge_le(&d, &N_BYTES) {
        sub_le(&d, &N_BYTES, 33)
    } else {
        d
    };
    r[..32].try_into().unwrap()
}

/// `N − x` for `0 < x < N`.
fn neg_mod_n(x: &[u8; 32]) -> [u8; 32] {
    sub_le(&N_BYTES, x, 32).try_into().unwrap()
}

/// Carries of `Σ_i S_i 256^i = 0` with `S_i = Σ a_j b_{i−j} − Σ q_j m_{i−j} + sign·c_i`.
fn carries(a: &[u8], b: &[u8], q: &[u8], m: &[u8], c: &[u8], sign: i128) -> [i64; 64] {
    let g = |v: &[u8], j: usize| *v.get(j).unwrap_or(&0) as i128;
    let mut out = [0i64; 64];
    let mut prev = 0i128;
    for (i, o) in out.iter_mut().enumerate() {
        let mut s = 0i128;
        for j in 0..=i {
            s += g(a, j) * g(b, i - j) - g(q, j) * g(m, i - j);
        }
        s += sign * g(c, i);
        let t = prev + s;
        assert_eq!(t.rem_euclid(256), 0, "relation does not hold at limb {i}");
        prev = t / 256;
        *o = prev as i64;
    }
    assert_eq!(out[63], 0, "relation does not close");
    out
}

fn fe_i64(c: i64) -> FE {
    if c >= 0 {
        FE::from(c as u64)
    } else {
        FE::zero() - FE::from((-c) as u64)
    }
}

/// A computed row of `sig`, `msg`, `parity` and its witness.
fn computed(sig: &[u8; 64], msg: &[u8; 32], parity: u8) -> (Vec<FE>, ecsm::EcrecWitness) {
    let o = op(sig, msg, parity);
    assert!(o.witness.ok, "the honest witness computes the key");
    let w = o.witness.clone();
    (row_of(&[o], 0), w)
}

fn set_bits(main: &mut [FE], col: usize, bytes: &[u8]) {
    for b in 0..256 {
        main[col + b] = FE::from(((bytes[b / 8] >> (b % 8)) & 1) as u64);
    }
}

fn set_carries(main: &mut [FE], col: usize, c: &[i64; 64]) {
    for (i, &v) in c.iter().enumerate() {
        if i < 63 {
            assert!(
                (0..1 << 16).contains(&(v + CARRY_OFFSET_MODN)),
                "carry {i} = {v} outside the IsHalfword window: the bus would refuse first"
            );
        }
        main[col + i] = fe_i64(v);
    }
}

fn set_halfwords(main: &mut [FE], col: usize, bytes: &[u8]) {
    for j in 0..16 {
        main[col + j] = FE::from(bytes[2 * j] as u64 + ((bytes[2 * j + 1] as u64) << 8));
    }
}

/// `r` and `s` of the first vector, big-endian.
fn r_s() -> ([u8; 32], [u8; 32]) {
    let (sig, ..) = vectors()[0];
    (sig[..32].try_into().unwrap(), sig[32..].try_into().unwrap())
}

fn sig_of(r_be: &[u8; 32], s_be: &[u8; 32]) -> [u8; 64] {
    let mut sig = [0u8; 64];
    sig[..32].copy_from_slice(r_be);
    sig[32..].copy_from_slice(s_be);
    sig
}

fn two() -> [u8; 32] {
    let mut two = [0u8; 32];
    two[0] = 2;
    two
}

const Z: &str = "c0ffee0011223344556677889900aabbccddeeff00112233445566778899aabb";

/// `u1' = u1 + N` with `u1 = 2` (`z = −2r mod N`), `qz' = qz + 1`: the u1 relation
/// still closes, every bit is a bit, every carry is in its window; only U1LtN's
/// closing carry refuses the row.
#[test]
fn u1_plus_n_is_refused_only_by_its_bound() {
    let (r_be, s_be) = r_s();
    let z_le = neg_mod_n(&double_mod_n(&be(&r_be)));
    let (mut main, w) = computed(&sig_of(&r_be, &s_be), &be(&z_le), 0);
    assert_eq!(w.u1, two(), "u1 = 2");
    assert!(failing(main.clone()).is_empty(), "the honest row holds");

    let u1p = add_le(&w.u1, &N_BYTES, 32);
    let qzp = add_le(&w.qz, &[1], 32);
    let cz = carries(&w.z, &w.w, &qzp, &N_BYTES, &u1p, 1);
    set_bits(&mut main, cols::U1, &u1p);
    set_bytes(&mut main, cols::QZ, &qzp);
    set_carries(&mut main, cols::CZ, &cz);
    // The best sub witness the prover has: (u1' − N) mod 2^256 = 2.
    set_halfwords(&mut main, cols::U1_SUB_N, &two());
    main[cols::LEN1] = FE::from(255u64);
    assert_eq!(failing(main), vec![closing(Bound::U1LtN)]);
}

/// `u2' = u2 + N` with `u2 = 2` (`s = 2r mod N`), `qu' = qu − 1`: only U2LtN refuses.
#[test]
fn u2_plus_n_is_refused_only_by_its_bound() {
    let (r_be, _) = r_s();
    let s_le = double_mod_n(&be(&r_be));
    let (mut main, w) = computed(&sig_of(&r_be, &be(&s_le)), &hex32(Z), 0);
    assert_eq!(w.u2, two(), "u2 = 2");
    assert!(failing(main.clone()).is_empty(), "the honest row holds");

    let u2p = add_le(&w.u2, &N_BYTES, 32);
    let qup = sub_le(&w.qu, &[1], 32);
    let cu = carries(&w.s, &w.w, &qup, &N_BYTES, &u2p, -1);
    set_bits(&mut main, cols::U2, &u2p);
    set_bytes(&mut main, cols::QU, &qup);
    set_carries(&mut main, cols::CU, &cu);
    set_halfwords(&mut main, cols::U2_SUB_N, &two());
    main[cols::LEN2] = FE::from(255u64);
    assert_eq!(failing(main), vec![closing(Bound::U2LtN)]);
}

/// `s' = s + N` with `s = 5` (Ethereum refuses `s ≥ N`; the same key would come
/// out: `u2` unchanged, `qu' = qu + w`): only SLtN refuses.
#[test]
fn s_plus_n_is_refused_only_by_its_bound() {
    let (r_be, _) = r_s();
    let mut s_le = [0u8; 32];
    s_le[0] = 5;
    let (mut main, w) = computed(&sig_of(&r_be, &be(&s_le)), &hex32(Z), 0);
    assert!(failing(main.clone()).is_empty(), "the honest row holds");

    let sp = add_le(&w.s, &N_BYTES, 32);
    let qup = add_le(&w.qu, &w.w, 32);
    let cu = carries(&sp, &w.w, &qup, &N_BYTES, &w.u2, -1);
    set_bytes(&mut main, cols::S, &sp);
    set_bytes(&mut main, cols::QU, &qup);
    set_carries(&mut main, cols::CU, &cu);
    set_halfwords(&mut main, cols::S_SUB_N, &s_le);
    assert_eq!(failing(main), vec![closing(Bound::SLtN)]);
}

/// `xA` or `xB` set to `p` (a non-canonical 0), the `xA ≠ xB` inverse re-solved:
/// only that coordinate's bound refuses.
#[test]
fn a_chain_x_at_p_is_refused_only_by_its_bound() {
    let (sig, msg, par, ..) = vectors()[0];
    for (col, sub_col, bound) in [
        (cols::XA, cols::XA_SUB_P, Bound::XaLtP),
        (cols::XB, cols::XB_SUB_P, Bound::XbLtP),
    ] {
        let (mut main, _) = computed(&sig, &msg, par);
        set_bytes(&mut main, col, &P_BYTES);
        set_halfwords(&mut main, sub_col, &[0u8; 32]);
        let word = |m: &[FE], base: usize, c: usize| -> i64 {
            (0..4)
                .map(|k| (m[base + 4 * c + k].canonical() as i64) << (8 * k))
                .sum()
        };
        for c in 0..8 {
            main[cols::DINV + c] = FE::zero();
        }
        let c = (0..8)
            .find(|&c| word(&main, cols::XB, c) != word(&main, cols::XA, c))
            .expect("a nonzero word difference");
        let d = word(&main, cols::XB, c) - word(&main, cols::XA, c);
        main[cols::DINV + c] = fe_i64(d).inv().unwrap();
        assert_eq!(failing(main), vec![closing(bound)], "{bound:?}");
    }
}
