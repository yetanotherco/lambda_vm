use executor::vm::logs::Log;
use math::field::{
    element::FieldElement, fields::fft_friendly::babybear_u32::Babybear31PrimeField,
};

type FE = FieldElement<Babybear31PrimeField>;

pub const NUM_COLUMNS: usize = 15;

#[derive(Default)]
pub struct DecodeTableRow {
    pub pc: [FE; 2],
    pub rs1: FE,
    pub rs2: FE,
    pub rd: FE,
    pub write_register: FE,
    pub memory_2bytes: FE,
    pub memory_4bytes: FE,
    pub imm: [FE; 2],
    pub signed: FE,
    pub mp_selector: FE,
    pub muldiv_selector: FE,
    pub instruction: FE,
    pub multiplicity: FE,
}

impl DecodeTableRow {
    pub fn from_log(_log: &Log) -> Self {
        DecodeTableRow::default()
    }

    pub fn to_vec(self) -> Vec<FE> {
        let mut row = Vec::with_capacity(NUM_COLUMNS);

        // pc[2]
        row.extend_from_slice(&self.pc);
        row.push(self.rs1);
        row.push(self.rs2);
        row.push(self.rd);
        row.push(self.write_register);
        row.push(self.memory_2bytes);
        row.push(self.memory_4bytes);
        // imm[2]
        row.extend_from_slice(&self.imm);
        row.push(self.signed);
        row.push(self.mp_selector);
        row.push(self.muldiv_selector);

        row
    }
}
