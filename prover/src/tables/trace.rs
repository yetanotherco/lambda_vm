use crate::tables::{
    cpu::{self, CpuTableRow},
    decode::{self, DecodeTableRow},
};
use executor::vm::logs::Log;
use math::field::{
    element::FieldElement,
    fields::fft_friendly::{
        babybear_u32::Babybear31PrimeField, quartic_babybear_u32::Degree4BabyBearU32ExtensionField,
    },
};
use stark::trace::TraceTable;

type FE = FieldElement<Babybear31PrimeField>;

pub struct Trace {
    pub cpu_trace_table: TraceTable<Babybear31PrimeField, Degree4BabyBearU32ExtensionField>,
    pub decode_trace_table: TraceTable<Babybear31PrimeField, Degree4BabyBearU32ExtensionField>,
}

impl Trace {
    pub fn generate_trace_from_logs(logs: Vec<Log>) -> Trace {
        let mut cpu_table: Vec<FE> = Vec::new();
        let mut decode_table: Vec<FE> = Vec::new();

        for (i, log) in logs.iter().enumerate() {
            let timestamp = (i * 4) as u32;
            cpu_table.extend(CpuTableRow::from_log(log, timestamp).to_vec());
            decode_table.extend(DecodeTableRow::from_log(log).to_vec());
        }

        pad_to_next_power_of_two(&mut cpu_table, cpu::NUM_COLUMNS);
        pad_to_next_power_of_two(&mut decode_table, decode::NUM_COLUMNS);

        let cpu_trace_table = TraceTable::new_main(cpu_table, cpu::NUM_COLUMNS, 1);
        let decode_trace_table = TraceTable::new_main(decode_table, decode::NUM_COLUMNS, 1);

        Trace {
            cpu_trace_table,
            decode_trace_table,
        }
    }
}

pub fn pad_to_next_power_of_two(table: &mut Vec<FE>, num_columns: usize) {
    const MIN_ROWS: usize = 4;
    let num_rows = table.len() / num_columns;
    let target_rows = num_rows.max(MIN_ROWS).next_power_of_two();
    table.resize(target_rows * num_columns, FE::zero());
}
