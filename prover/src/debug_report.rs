//! Bus ID legend for debug output.
//!
//! The bus balance report runs inside the generic stark crate (which only knows
//! numeric IDs). This module prints the ID → name mapping so the output is
//! human-readable. Only compiled with the `debug-checks` feature.

use crate::tables::types::BusId;

/// Print a legend mapping numeric bus IDs to their names.
pub fn print_bus_legend() {
    eprintln!("=== BUS ID LEGEND ===");
    for id in 0u64..=32 {
        if let Ok(bus) = BusId::try_from(id) {
            eprintln!("  Bus {:2} = {}", id, bus.name());
        }
    }
    eprintln!("=====================");
}
