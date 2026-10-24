//! A minimal RV64IM assembler and ELF writer, enough to emit a guest program
//! from the prover without a toolchain (`native_vbin`).

pub type Reg = u32;

pub const ZERO: Reg = 0;
pub const RA: Reg = 1;
pub const SP: Reg = 2;
pub const GP: Reg = 3;
pub const TP: Reg = 4;
pub const T0: Reg = 5;
pub const T1: Reg = 6;
pub const T2: Reg = 7;
pub const S0: Reg = 8;
pub const S1: Reg = 9;
pub const A0: Reg = 10;
pub const A1: Reg = 11;
pub const A2: Reg = 12;
pub const A3: Reg = 13;
pub const A4: Reg = 14;
pub const A5: Reg = 15;
pub const A6: Reg = 16;
pub const A7: Reg = 17;
pub const S2: Reg = 18;
pub const T3: Reg = 28;
pub const T4: Reg = 29;
pub const T5: Reg = 30;
pub const T6: Reg = 31;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Label(usize);

#[derive(Clone, Copy)]
enum Fix {
    Branch,
    Jal,
}

#[derive(Default)]
pub struct Asm {
    pub code: Vec<u32>,
    labels: Vec<Option<usize>>,
    fixups: Vec<(usize, Label, Fix)>,
}

pub fn fits12(v: i64) -> bool {
    (-2048..2048).contains(&v)
}

fn r_type(f7: u32, rs2: Reg, rs1: Reg, f3: u32, rd: Reg, op: u32) -> u32 {
    f7 << 25 | rs2 << 20 | rs1 << 15 | f3 << 12 | rd << 7 | op
}

fn i_type(imm: i64, rs1: Reg, f3: u32, rd: Reg, op: u32) -> u32 {
    assert!(fits12(imm), "immediate {imm} out of range");
    ((imm as u32) & 0xfff) << 20 | rs1 << 15 | f3 << 12 | rd << 7 | op
}

fn s_type(imm: i64, rs2: Reg, rs1: Reg, f3: u32, op: u32) -> u32 {
    assert!(fits12(imm), "immediate {imm} out of range");
    let i = imm as u32;
    ((i >> 5) & 0x7f) << 25 | rs2 << 20 | rs1 << 15 | f3 << 12 | (i & 0x1f) << 7 | op
}

fn b_type(off: i64, rs2: Reg, rs1: Reg, f3: u32) -> u32 {
    assert!(
        (-4096..4096).contains(&off) && off % 2 == 0,
        "branch offset {off}"
    );
    let i = off as u32;
    ((i >> 12) & 1) << 31
        | ((i >> 5) & 0x3f) << 25
        | rs2 << 20
        | rs1 << 15
        | f3 << 12
        | ((i >> 1) & 0xf) << 8
        | ((i >> 11) & 1) << 7
        | 0x63
}

fn j_type(off: i64, rd: Reg) -> u32 {
    assert!(
        (-(1 << 20)..(1 << 20)).contains(&off) && off % 2 == 0,
        "jump offset {off}"
    );
    let i = off as u32;
    ((i >> 20) & 1) << 31
        | ((i >> 1) & 0x3ff) << 21
        | ((i >> 11) & 1) << 20
        | ((i >> 12) & 0xff) << 12
        | rd << 7
        | 0x6f
}

const OP: u32 = 0x33;
const OP_IMM: u32 = 0x13;
const LOAD: u32 = 0x03;
const STORE: u32 = 0x23;

impl Asm {
    pub fn pc(&self) -> usize {
        self.code.len()
    }

    fn emit(&mut self, w: u32) {
        self.code.push(w);
    }

    pub fn label(&mut self) -> Label {
        self.labels.push(None);
        Label(self.labels.len() - 1)
    }

    pub fn bind(&mut self, l: Label) {
        assert!(self.labels[l.0].is_none(), "label bound twice");
        self.labels[l.0] = Some(self.code.len());
    }

    pub fn here(&mut self) -> Label {
        let l = self.label();
        self.bind(l);
        l
    }

    /// Instructions between the current position and a bound label.
    pub fn distance(&self, l: Label) -> i64 {
        self.labels[l.0].expect("a bound label") as i64 - self.code.len() as i64
    }

    pub fn add(&mut self, rd: Reg, a: Reg, b: Reg) {
        self.emit(r_type(0, b, a, 0, rd, OP));
    }
    pub fn sub(&mut self, rd: Reg, a: Reg, b: Reg) {
        self.emit(r_type(0x20, b, a, 0, rd, OP));
    }
    pub fn sltu(&mut self, rd: Reg, a: Reg, b: Reg) {
        self.emit(r_type(0, b, a, 3, rd, OP));
    }
    pub fn xor(&mut self, rd: Reg, a: Reg, b: Reg) {
        self.emit(r_type(0, b, a, 4, rd, OP));
    }
    pub fn srl(&mut self, rd: Reg, a: Reg, b: Reg) {
        self.emit(r_type(0, b, a, 5, rd, OP));
    }
    pub fn sll(&mut self, rd: Reg, a: Reg, b: Reg) {
        self.emit(r_type(0, b, a, 1, rd, OP));
    }
    pub fn or(&mut self, rd: Reg, a: Reg, b: Reg) {
        self.emit(r_type(0, b, a, 6, rd, OP));
    }
    pub fn and(&mut self, rd: Reg, a: Reg, b: Reg) {
        self.emit(r_type(0, b, a, 7, rd, OP));
    }
    pub fn mul(&mut self, rd: Reg, a: Reg, b: Reg) {
        self.emit(r_type(1, b, a, 0, rd, OP));
    }
    pub fn mulhu(&mut self, rd: Reg, a: Reg, b: Reg) {
        self.emit(r_type(1, b, a, 3, rd, OP));
    }
    pub fn divu(&mut self, rd: Reg, a: Reg, b: Reg) {
        self.emit(r_type(1, b, a, 5, rd, OP));
    }
    pub fn addi(&mut self, rd: Reg, a: Reg, imm: i64) {
        self.emit(i_type(imm, a, 0, rd, OP_IMM));
    }
    pub fn sltiu(&mut self, rd: Reg, a: Reg, imm: i64) {
        self.emit(i_type(imm, a, 3, rd, OP_IMM));
    }
    pub fn xori(&mut self, rd: Reg, a: Reg, imm: i64) {
        self.emit(i_type(imm, a, 4, rd, OP_IMM));
    }
    pub fn ori(&mut self, rd: Reg, a: Reg, imm: i64) {
        self.emit(i_type(imm, a, 6, rd, OP_IMM));
    }
    pub fn andi(&mut self, rd: Reg, a: Reg, imm: i64) {
        self.emit(i_type(imm, a, 7, rd, OP_IMM));
    }
    pub fn slli(&mut self, rd: Reg, a: Reg, sh: u32) {
        assert!(sh < 64);
        self.emit(i_type(sh as i64, a, 1, rd, OP_IMM));
    }
    pub fn srli(&mut self, rd: Reg, a: Reg, sh: u32) {
        assert!(sh < 64);
        self.emit(i_type(sh as i64, a, 5, rd, OP_IMM));
    }
    pub fn mv(&mut self, rd: Reg, a: Reg) {
        self.addi(rd, a, 0);
    }
    /// `rd = −a`.
    pub fn neg(&mut self, rd: Reg, a: Reg) {
        self.sub(rd, ZERO, a);
    }
    pub fn lui(&mut self, rd: Reg, imm20: i64) {
        assert!((-(1 << 19)..(1 << 19)).contains(&imm20), "lui {imm20}");
        self.emit(((imm20 as u32) & 0xfffff) << 12 | rd << 7 | 0x37);
    }
    pub fn ld(&mut self, rd: Reg, base: Reg, imm: i64) {
        self.emit(i_type(imm, base, 3, rd, LOAD));
    }
    pub fn lwu(&mut self, rd: Reg, base: Reg, imm: i64) {
        self.emit(i_type(imm, base, 6, rd, LOAD));
    }
    pub fn sd(&mut self, rs: Reg, base: Reg, imm: i64) {
        self.emit(s_type(imm, rs, base, 3, STORE));
    }
    pub fn sw(&mut self, rs: Reg, base: Reg, imm: i64) {
        self.emit(s_type(imm, rs, base, 2, STORE));
    }
    pub fn ecall(&mut self) {
        self.emit(0x73);
    }
    pub fn jalr(&mut self, rd: Reg, base: Reg, imm: i64) {
        self.emit(i_type(imm, base, 0, rd, 0x67));
    }
    pub fn ret_via(&mut self, link: Reg) {
        self.jalr(ZERO, link, 0);
    }

    fn branch(&mut self, f3: u32, a: Reg, b: Reg, l: Label) {
        self.fixups.push((self.code.len(), l, Fix::Branch));
        self.emit(r_type(0, b, a, f3, 0, 0x63));
    }
    pub fn beq(&mut self, a: Reg, b: Reg, l: Label) {
        self.branch(0, a, b, l);
    }
    pub fn bne(&mut self, a: Reg, b: Reg, l: Label) {
        self.branch(1, a, b, l);
    }
    pub fn bltu(&mut self, a: Reg, b: Reg, l: Label) {
        self.branch(6, a, b, l);
    }
    /// Calls the bound label `l`, linking in `link`, over any distance.
    pub fn call(&mut self, link: Reg, l: Label) {
        let off = self.distance(l) * 4;
        if (-(1 << 20)..(1 << 20)).contains(&off) {
            self.jal(link, l);
        } else {
            let lo = (off << 52) >> 52;
            self.emit((((off - lo) >> 12) as u32 & 0xfffff) << 12 | link << 7 | 0x17);
            self.jalr(link, link, lo);
        }
    }

    pub fn jal(&mut self, rd: Reg, l: Label) {
        self.fixups.push((self.code.len(), l, Fix::Jal));
        self.emit(rd << 7 | 0x6f);
    }

    /// `rd = v`, in at most eight instructions.
    pub fn li(&mut self, rd: Reg, v: i64) {
        if fits12(v) {
            self.addi(rd, ZERO, v);
            return;
        }
        let lo = (v << 52) >> 52;
        let hi = v.wrapping_sub(lo);
        if hi == (hi as i32) as i64 && (hi >> 12) < (1 << 19) && (hi >> 12) >= -(1 << 19) {
            self.lui(rd, hi >> 12);
            if lo != 0 {
                self.addi(rd, rd, lo);
            }
            return;
        }
        let sh = hi.trailing_zeros().min(63);
        self.li(rd, hi >> sh);
        self.slli(rd, rd, sh);
        if lo != 0 {
            self.addi(rd, rd, lo);
        }
    }

    /// `rd = rs + v` (`rd` may be `rs`), through `tmp` when `v` is wide.
    pub fn add_imm(&mut self, rd: Reg, rs: Reg, v: i64, tmp: Reg) {
        if fits12(v) {
            if v != 0 || rd != rs {
                self.addi(rd, rs, v);
            }
        } else {
            self.li(tmp, v);
            self.add(rd, rs, tmp);
        }
    }

    /// Loops back to `top` while `r != 0`, over any distance.
    pub fn bnez_back(&mut self, r: Reg, top: Label) {
        if self.distance(top) * 4 >= -4096 {
            self.bne(r, ZERO, top);
        } else {
            let skip = self.label();
            self.beq(r, ZERO, skip);
            self.jal(ZERO, top);
            self.bind(skip);
        }
    }

    /// Resolves the labels: the instruction words.
    pub fn finish(mut self) -> Vec<u32> {
        for (at, l, fix) in std::mem::take(&mut self.fixups) {
            let target = self.labels[l.0].expect("an unbound label");
            let off = (target as i64 - at as i64) * 4;
            let w = self.code[at];
            self.code[at] = match fix {
                Fix::Branch => b_type(off, (w >> 20) & 31, (w >> 15) & 31, (w >> 12) & 7),
                Fix::Jal => j_type(off, (w >> 7) & 31),
            };
        }
        self.code
    }
}

/// An ELF64 RISC-V executable: `code` at `code_base` (read + execute), then
/// `data` at `data_base` (read only), entry at `code_base`.
pub fn elf(code_base: u64, code: &[u32], data_base: u64, data: &[u8]) -> Vec<u8> {
    let segments: Vec<(u64, Vec<u8>, u32)> = [
        (
            code_base,
            code.iter()
                .flat_map(|w| w.to_le_bytes())
                .collect::<Vec<u8>>(),
            5,
        ),
        (data_base, data.to_vec(), 4),
    ]
    .into_iter()
    .filter(|(_, b, _)| !b.is_empty())
    .collect();
    let mut out = vec![0u8; 64];
    out[..16].copy_from_slice(&[0x7f, b'E', b'L', b'F', 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    out[16..18].copy_from_slice(&2u16.to_le_bytes());
    out[18..20].copy_from_slice(&243u16.to_le_bytes());
    out[20..24].copy_from_slice(&1u32.to_le_bytes());
    out[24..32].copy_from_slice(&code_base.to_le_bytes());
    out[32..40].copy_from_slice(&64u64.to_le_bytes());
    out[52..54].copy_from_slice(&64u16.to_le_bytes());
    out[54..56].copy_from_slice(&56u16.to_le_bytes());
    out[56..58].copy_from_slice(&(segments.len() as u16).to_le_bytes());
    out[58..60].copy_from_slice(&64u16.to_le_bytes());
    let mut offset = (64 + 56 * segments.len()).next_multiple_of(16) as u64;
    let mut headers = Vec::new();
    for (base, bytes, flags) in &segments {
        let mut h = Vec::with_capacity(56);
        h.extend(1u32.to_le_bytes());
        h.extend(flags.to_le_bytes());
        h.extend(offset.to_le_bytes());
        h.extend(base.to_le_bytes());
        h.extend(base.to_le_bytes());
        h.extend((bytes.len() as u64).to_le_bytes());
        h.extend((bytes.len() as u64).to_le_bytes());
        h.extend(0x1000u64.to_le_bytes());
        headers.extend(h);
        offset += (bytes.len() as u64).next_multiple_of(16);
    }
    out.extend(headers);
    for (_, bytes, _) in &segments {
        out.resize(out.len().next_multiple_of(16), 0);
        out.extend(bytes);
    }
    out
}
