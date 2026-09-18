//! SHA256 compression syscall (-1), following spec/sha256.typ.
//! One core row binds memory to the schedule, round chain, and feed-forward.
use super::{
    sha256_common::*,
    sha256_schedule,
    types::{BusId, GoldilocksExtension as E, GoldilocksField as F},
};
use crate::constraints::templates::{AddOperand, INV_SHIFT_32, emit_add_pair};
use stark::{
    constraints::builder::{ConstraintBuilder, ConstraintSet},
    lookup::{BusInteraction, BusValue},
    trace::TraceTable,
};
pub const PTR: usize = 4;
pub const H: usize = 60;
pub const M: usize = 92;
pub const OUT: usize = 156;
pub const LAST: usize = 188;
pub const CARRY: usize = 196;
pub const MU: usize = 204;
pub const WIDTH: usize = 205;
// 14 pointers: h[0..4], m[0..8], inclusive h end, inclusive m end.
#[derive(Clone, Debug)]
pub struct Operation {
    pub timestamp: u64,
    pub state_addr: u64,
    pub message_addr: u64,
    pub state: [u8; 32],
    pub message: [u8; 64],
}
impl Operation {
    pub fn state_words(&self) -> [u32; 8] {
        std::array::from_fn(|i| {
            u32::from_be_bytes(self.state[4 * i..4 * i + 4].try_into().unwrap())
        })
    }
    pub fn pointers(&self) -> [u64; 14] {
        std::array::from_fn(|i| match i {
            0..4 => self.state_addr + 8 * i as u64,
            4..12 => self.message_addr + 8 * (i - 4) as u64,
            12 => self.state_addr + 31,
            _ => self.message_addr + 63,
        })
    }
}
pub fn generate(ops: &[Operation]) -> TraceTable<F, E> {
    let mut rows = TraceRows::new(ops.len(), WIDTH);
    for op in ops {
        rows.push(|r| {
            r[0] = op.timestamp & 0xffffffff;
            r[1] = op.timestamp >> 32;
            r[2] = (op.timestamp + 1) & 0xffffffff;
            r[3] = (op.timestamp + 1) >> 32;
            for (i, p) in op.pointers().iter().enumerate() {
                for j in 0..4 {
                    r[PTR + 4 * i + j] = (p >> (16 * j)) & 65535;
                }
            }
            for i in 0..32 {
                r[H + i] = op.state[i] as u64;
            }
            for i in 0..64 {
                r[M + i] = op.message[i] as u64;
            }
            let mut out = op.state;
            executor::sha256::compress(&mut out, &op.message);
            for i in 0..32 {
                r[OUT + i] = out[i] as u64;
            }
            let w = executor::sha256::schedule(&op.message);
            let init = op.state_words();
            let mut s = init;
            for (&word, &constant) in w.iter().zip(&executor::sha256::K) {
                s = executor::sha256::round(s, word, constant);
            }
            for i in 0..8 {
                r[LAST + i] = s[i] as u64;
                r[CARRY + i] = (s[i] as u64 + init[i] as u64) >> 32;
            }
            r[MU] = 1;
        });
    }
    rows.finish()
}
fn mem(
    ptr: usize,
    old: Vec<BusValue>,
    new: Vec<BusValue>,
    ts: usize,
    reg: Option<u64>,
) -> BusInteraction {
    let mut v = old;
    v.push(BusValue::constant(reg.is_some() as u64));
    if let Some(reg) = reg {
        v.extend([BusValue::constant(2 * reg), BusValue::constant(0)]);
    } else {
        v.extend([half(ptr), half(ptr + 2)]);
    }
    v.extend(new);
    v.extend([
        col(ts),
        col(ts + 1),
        BusValue::constant(reg.is_some() as u64),
        BusValue::constant(0),
        BusValue::constant(reg.is_none() as u64),
    ]);
    send(BusId::Memw, MU, v)
}
pub fn bus_interactions() -> Vec<BusInteraction> {
    let mut v = vec![recv(
        BusId::Ecall,
        MU,
        vec![
            col(0),
            col(1),
            BusValue::constant(0xffffffff),
            BusValue::constant(0xffffffff),
        ],
    )];
    for (reg, p) in [(10, PTR), (11, PTR + 16)] {
        let mut bytes = vec![half(p), half(p + 2)];
        bytes.extend((0..6).map(|_| BusValue::constant(0)));
        v.push(mem(p, bytes.clone(), bytes, 0, Some(reg)));
    }
    for i in 0..8 {
        let bytes = (0..8).map(|j| col(M + 8 * i + j)).collect::<Vec<_>>();
        v.push(mem(PTR + 16 + 4 * i, bytes.clone(), bytes, 0, None));
    }
    for i in 0..4 {
        v.push(mem(
            PTR + 4 * i,
            (0..8).map(|j| col(H + 8 * i + j)).collect(),
            (0..8).map(|j| col(OUT + 8 * i + j)).collect(),
            2,
            None,
        ));
    }
    for i in 0..56 {
        v.push(send(BusId::IsHalfword, MU, vec![col(PTR + i)]));
    }
    for i in 0..16 {
        v.push(send(
            BusId::AreBytes,
            MU,
            vec![col(OUT + 2 * i), col(OUT + 2 * i + 1)],
        ));
    }
    for i in 0..16 {
        let mut interaction = recv(
            BusId::ShaM,
            MU,
            vec![col(0), col(1), BusValue::constant(i as u64), be(M + 4 * i)],
        );
        interaction.multiplicity =
            stark::lookup::Multiplicity::Linear(vec![stark::lookup::LinearTerm::Column {
                column: MU,
                coefficient: sha256_schedule::amount(i) as i64,
            }]);
        v.push(interaction);
    }
    let mut start = vec![col(0), col(1), BusValue::constant(0)];
    start.extend((0..8).map(|i| be(H + 4 * i)));
    v.push(send(BusId::ShaRound, MU, start));
    let mut end = vec![col(0), col(1), BusValue::constant(64)];
    end.extend((0..8).map(|i| col(LAST + i)));
    v.push(recv(BusId::ShaRound, MU, end));
    v
}
#[derive(Clone, Copy)]
pub struct Constraints;
impl ConstraintSet<F, E> for Constraints {
    fn max_degree(&self) -> usize {
        3
    }
    fn eval<B: ConstraintBuilder<F, E>>(&self, b: &mut B) {
        let mut id = 0;
        // The eight feed-forward carries, then MU on its own. MU is the
        // multiplicity of the ECALL receive, of all fourteen MEMW sends and of
        // both SHA256ROUND interactions, so its range check must not depend on
        // it happening to sit in the column right after the carries.
        check_bits(b, &mut id, CARRY, 8);
        check_bits(b, &mut id, MU, 1);
        emit_add_pair(
            b,
            id,
            &[MU],
            &AddOperand::dword(0),
            &AddOperand::constant(1),
            &AddOperand::dword(2),
        );
        id += 2;
        for i in 0..14 {
            let (base, offset) = match i {
                0..4 => (PTR, 8 * i),
                4..12 => (PTR + 16, 8 * (i - 4)),
                12 => (PTR, 31),
                _ => (PTR + 16, 63),
            };
            let p = PTR + 4 * i;
            emit_add_pair(
                b,
                id,
                &[MU],
                &AddOperand::from_dword_hl(base),
                &AddOperand::constant(offset as i64),
                &AddOperand::from_dword_hl(p),
            );
            id += 2;
            if i >= 12 {
                let lo = b.main(0, base) + b.const_base(65536) * b.main(0, base + 1);
                let hi = b.main(0, base + 2) + b.const_base(65536) * b.main(0, base + 3);
                let outlo = b.main(0, p) + b.const_base(65536) * b.main(0, p + 1);
                let outhi = b.main(0, p + 2) + b.const_base(65536) * b.main(0, p + 3);
                let carry = (lo + b.const_base(offset as u64) - outlo) * b.const_base(INV_SHIFT_32);
                b.emit_base(id, b.main(0, MU) * (hi + carry - outhi));
                id += 1;
            }
        }
        for i in 0..8 {
            let mut h = b.const_base(0);
            let mut out = b.const_base(0);
            for j in 0..4 {
                h = h + b.const_base(1 << (8 * (3 - j))) * b.main(0, H + 4 * i + j);
                out = out + b.const_base(1 << (8 * (3 - j))) * b.main(0, OUT + 4 * i + j);
            }
            b.emit_base(
                id,
                b.main(0, MU)
                    * (h + b.main(0, LAST + i)
                        - out
                        - b.const_base(1 << 32) * b.main(0, CARRY + i)),
            );
            id += 1;
        }
    }
}

pub fn rot_ops(ops: &[Operation]) -> Vec<(u32, usize)> {
    let mut v = vec![];
    for op in ops {
        let w = executor::sha256::schedule(&op.message);
        for i in 16..64 {
            v.push((w[i - 15], 0));
            v.push((w[i - 2], 1));
        }
        let mut s = op.state_words();
        for (&word, &constant) in w.iter().zip(&executor::sha256::K) {
            v.push((s[0], 2));
            v.push((s[4], 3));
            s = executor::sha256::round(s, word, constant);
        }
    }
    v
}
pub fn bitwise_ops(ops: &[Operation]) -> Vec<super::bitwise::BitwiseOperation> {
    use super::bitwise::{BitwiseOperation as Op, BitwiseOperationType as Ty};
    let mut v = vec![];
    for op in ops {
        for p in op.pointers() {
            for j in 0..4 {
                let h = (p >> (16 * j)) as u16;
                v.push(Op::halfword(Ty::IsHalf, h as u8, (h >> 8) as u8));
            }
        }
        let mut out = op.state;
        executor::sha256::compress(&mut out, &op.message);
        for i in 0..16 {
            v.push(Op::halfword(Ty::AreBytes, out[2 * i], out[2 * i + 1]));
        }
    }
    // ROTXOR resolves its shifts and XORs through BITWISE now, so its lookups
    // have to be counted here too or the bus does not balance.
    v.extend(super::sha256_rotxor::bitwise_ops(&rot_ops(ops)));
    v.extend(super::sha256_round::bitwise_ops(ops));
    v.extend(super::sha256_schedule::bitwise_ops(ops));
    v
}
