//! Bus balance debug report for the VM prover.
//!
//! Prints a global bus balance report after proving, aggregating per-bus sums
//! across all tables. Uses `BusId::name()` for human-readable output.
//!
//! This module is only compiled with the `debug-checks` feature.

use std::collections::HashMap;

use math::field::element::FieldElement;
use math::field::traits::IsField;
use stark::bus_debug::BUS_DEBUG_TRACKER;
use stark::proof::stark::MultiProof;

use crate::tables::types::BusId;

/// Print a global bus balance report from a completed multi-proof.
///
/// Iterates `proof.bus_public_inputs` (which contains per-bus sums computed
/// during round 1) and aggregates across all tables.
pub fn print_bus_balance_report<F, E, PI>(multi_proof: &MultiProof<F, E, PI>)
where
    F: math::field::traits::IsSubFieldOf<E> + math::field::traits::IsFFTField,
    E: IsField,
{
    let has_logup = multi_proof
        .proofs
        .iter()
        .any(|p| p.bus_public_inputs.is_some());
    if !has_logup {
        return;
    }

    let mut global_bus_sums: HashMap<u64, FieldElement<E>> = HashMap::new();
    let mut bus_senders: HashMap<u64, Vec<(String, FieldElement<E>)>> = HashMap::new();
    let mut bus_receivers: HashMap<u64, Vec<(String, FieldElement<E>)>> = HashMap::new();
    let mut global_sender_sums: HashMap<u64, FieldElement<E>> = HashMap::new();
    let mut global_receiver_sums: HashMap<u64, FieldElement<E>> = HashMap::new();

    for proof in &multi_proof.proofs {
        if let Some(bus_inputs) = &proof.bus_public_inputs {
            for (&bus_id, sum) in &bus_inputs.per_bus_sums {
                *global_bus_sums
                    .entry(bus_id)
                    .or_insert(FieldElement::zero()) += sum.clone();
            }
            for (&bus_id, sum) in &bus_inputs.per_bus_sender_sums {
                *global_sender_sums
                    .entry(bus_id)
                    .or_insert(FieldElement::zero()) += sum.clone();
                bus_senders
                    .entry(bus_id)
                    .or_default()
                    .push((bus_inputs.table_name.clone(), sum.clone()));
            }
            for (&bus_id, sum) in &bus_inputs.per_bus_receiver_sums {
                *global_receiver_sums
                    .entry(bus_id)
                    .or_insert(FieldElement::zero()) += sum.clone();
                bus_receivers
                    .entry(bus_id)
                    .or_default()
                    .push((bus_inputs.table_name.clone(), sum.clone()));
            }
        }
    }

    eprintln!("\n=== GLOBAL BUS BALANCE REPORT ===");
    let zero = FieldElement::<E>::zero();
    let mut bus_ids: Vec<_> = global_bus_sums.keys().copied().collect();
    bus_ids.sort();
    for bus_id in bus_ids {
        let total = &global_bus_sums[&bus_id];
        let name = bus_name(bus_id);
        if *total != zero {
            eprintln!("Bus {:2} ({:10}): IMBALANCED", bus_id, name);

            if let Some(senders) = bus_senders.get(&bus_id) {
                eprintln!("  SENDERS:");
                for (table_name, sum) in senders {
                    eprintln!("    [{:12}]: {:?}", table_name, sum);
                }
                if let Some(total_sent) = global_sender_sums.get(&bus_id) {
                    eprintln!("    → Total sent: {:?}", total_sent);
                }
            }

            if let Some(receivers) = bus_receivers.get(&bus_id) {
                eprintln!("  RECEIVERS:");
                for (table_name, sum) in receivers {
                    eprintln!("    [{:12}]: {:?}", table_name, sum);
                }
                if let Some(total_recv) = global_receiver_sums.get(&bus_id) {
                    eprintln!("    → Total received: {:?}", total_recv);
                }
            }

            eprintln!("  IMBALANCE: {:?}\n", total);
        } else {
            eprintln!("Bus {:2} ({:10}): BALANCED ✓", bus_id, name);
        }
    }
    eprintln!("=================================\n");

    // Run BusDebugTracker analysis if any interactions were logged
    let tracker = BUS_DEBUG_TRACKER.lock().unwrap_or_else(|e| {
        eprintln!("[BusDebugTracker] WARNING: mutex was poisoned, recovering data");
        e.into_inner()
    });
    if !tracker.is_empty() {
        if tracker.is_truncated() {
            eprintln!(
                "[BusDebugTracker] WARNING: Log truncated at {} entries — results may be incomplete",
                tracker.len()
            );
        }
        eprintln!(
            "[BusDebugTracker] Logged {} interactions, running analysis...",
            tracker.len()
        );
        let report = tracker.analyze_mismatches();
        report.print_summary();
    }
}

/// Map a raw bus ID to a human-readable name via `BusId`.
fn bus_name(bus_id: u64) -> &'static str {
    match BusId::try_from(bus_id) {
        Ok(id) => id.name(),
        Err(_) => "Unknown",
    }
}
