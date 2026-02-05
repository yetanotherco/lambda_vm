//! Bus Debug Tracker - Standalone LogUp bus debugging utility
//!
//! When a LogUp bus is imbalanced (sum ≠ 0), this tool logs every bus interaction
//! at fingerprint computation time, enabling post-hoc analysis to find exact mismatches.
//!
//! # Usage
//!
//! Enable tracking with environment variables:
//! ```bash
//! # Track all buses
//! DEBUG_BUS_TRACKER=1 cargo test --release test_name -- --nocapture
//!
//! # Track specific bus only
//! DEBUG_BUS_TRACKER=1 DEBUG_BUS_ID=14 cargo test ...
//! ```
//!
//! # Output
//!
//! - `bus_interactions.csv` - Raw data for custom analysis
//! - stderr - Mismatch report summary

use std::collections::HashMap;
use std::io::Write;
use std::sync::{LazyLock, Mutex};

/// Single bus interaction log entry
#[derive(Debug, Clone)]
pub struct BusInteractionLog {
    /// Table that produced this interaction
    pub table_name: String,
    /// Row index within the table
    pub row_idx: usize,
    /// Bus identifier (e.g., 14 for Memw)
    pub bus_id: u64,
    /// True for sender, false for receiver
    pub is_sender: bool,
    /// Multiplicity value (how many times this row participates)
    pub multiplicity: u64,
    /// Raw bus elements before alpha-combining: [bus_id, elem1, elem2, ...]
    /// Stored as hex strings for readability
    pub bus_elements: Vec<String>,
    /// Fingerprint = z - linear_combination(bus_elements, alpha)
    pub fingerprint: String,
}

/// Global debug tracker - enabled via `debug-checks` feature flag
pub static BUS_DEBUG_TRACKER: LazyLock<Mutex<BusDebugTracker>> =
    LazyLock::new(|| Mutex::new(BusDebugTracker::new()));

pub struct BusDebugTracker {
    bus_filter: Option<u64>,
    logs: Vec<BusInteractionLog>,
}

impl Default for BusDebugTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl BusDebugTracker {
    pub fn new() -> Self {
        let bus_filter = std::env::var("DEBUG_BUS_ID")
            .ok()
            .and_then(|s| s.parse().ok());

        Self {
            bus_filter,
            logs: Vec::new(),
        }
    }

    /// Log an interaction (optionally filtered by DEBUG_BUS_ID env var)
    pub fn log(&mut self, log: BusInteractionLog) {
        if let Some(filter) = self.bus_filter
            && log.bus_id != filter
        {
            return;
        }
        self.logs.push(log);
    }

    /// Get number of logged interactions
    pub fn len(&self) -> usize {
        self.logs.len()
    }

    /// Check if no interactions logged
    pub fn is_empty(&self) -> bool {
        self.logs.is_empty()
    }

    /// Generate mismatch report for imbalanced buses
    pub fn analyze_mismatches(&self) -> BusMismatchReport {
        let mut imbalanced = Vec::new();

        // Group logs by bus_id
        let mut by_bus: HashMap<u64, Vec<&BusInteractionLog>> = HashMap::new();
        for log in &self.logs {
            by_bus.entry(log.bus_id).or_default().push(log);
        }

        for (bus_id, logs) in by_bus {
            // Separate senders and receivers
            let senders: Vec<_> = logs.iter().filter(|l| l.is_sender).copied().collect();
            let receivers: Vec<_> = logs.iter().filter(|l| !l.is_sender).copied().collect();

            // Group by fingerprint and sum multiplicities
            let sender_fps = Self::group_by_fingerprint(&senders);
            let receiver_fps = Self::group_by_fingerprint(&receivers);

            // Find orphan senders (fingerprint in senders but not receivers)
            let orphan_senders: Vec<OrphanInteraction> = sender_fps
                .iter()
                .filter(|(fp, _)| !receiver_fps.contains_key(*fp))
                .flat_map(|(_, logs)| {
                    logs.iter().map(|log| OrphanInteraction {
                        table_name: log.table_name.clone(),
                        row_idx: log.row_idx,
                        multiplicity: log.multiplicity,
                        fingerprint: log.fingerprint.clone(),
                        bus_elements: log.bus_elements.clone(),
                    })
                })
                .collect();

            // Find orphan receivers (fingerprint in receivers but not senders)
            let orphan_receivers: Vec<OrphanInteraction> = receiver_fps
                .iter()
                .filter(|(fp, _)| !sender_fps.contains_key(*fp))
                .flat_map(|(_, logs)| {
                    logs.iter().map(|log| OrphanInteraction {
                        table_name: log.table_name.clone(),
                        row_idx: log.row_idx,
                        multiplicity: log.multiplicity,
                        fingerprint: log.fingerprint.clone(),
                        bus_elements: log.bus_elements.clone(),
                    })
                })
                .collect();

            // Find multiplicity mismatches (same fingerprint, different total counts)
            let multiplicity_mismatches: Vec<MultiplicityMismatch> = sender_fps
                .iter()
                .filter_map(|(fp, sender_logs)| {
                    receiver_fps.get(*fp).and_then(|receiver_logs| {
                        // Sum ALL sender multiplicities for this fingerprint
                        let total_sent: u64 = sender_logs.iter().map(|l| l.multiplicity).sum();
                        // Sum ALL receiver multiplicities for this fingerprint
                        let total_received: u64 =
                            receiver_logs.iter().map(|l| l.multiplicity).sum();

                        if total_sent != total_received {
                            Some(MultiplicityMismatch {
                                fingerprint: fp.to_string(),
                                bus_elements: sender_logs
                                    .first()
                                    .map(|l| l.bus_elements.clone())
                                    .unwrap_or_default(),
                                total_sent,
                                total_received,
                                senders: sender_logs
                                    .iter()
                                    .map(|l| (l.table_name.clone(), l.row_idx, l.multiplicity))
                                    .collect(),
                                receivers: receiver_logs
                                    .iter()
                                    .map(|l| (l.table_name.clone(), l.row_idx, l.multiplicity))
                                    .collect(),
                            })
                        } else {
                            None
                        }
                    })
                })
                .collect();

            if !orphan_senders.is_empty()
                || !orphan_receivers.is_empty()
                || !multiplicity_mismatches.is_empty()
            {
                imbalanced.push(BusImbalance {
                    bus_id,
                    orphan_senders,
                    orphan_receivers,
                    multiplicity_mismatches,
                });
            }
        }

        BusMismatchReport {
            imbalanced_buses: imbalanced,
        }
    }

    /// Group interactions by fingerprint
    fn group_by_fingerprint<'a>(
        logs: &[&'a BusInteractionLog],
    ) -> HashMap<&'a str, Vec<&'a BusInteractionLog>> {
        let mut grouped: HashMap<&str, Vec<&BusInteractionLog>> = HashMap::new();
        for log in logs {
            grouped.entry(&log.fingerprint).or_default().push(log);
        }
        grouped
    }

    /// Export to CSV
    pub fn to_csv(&self, path: &str) -> std::io::Result<()> {
        let mut file = std::fs::File::create(path)?;

        // Header
        writeln!(
            file,
            "table_name,row_idx,bus_id,is_sender,multiplicity,fingerprint,bus_elements"
        )?;

        // Data
        for log in &self.logs {
            let bus_elements_str = format!("{:?}", log.bus_elements);
            writeln!(
                file,
                "{},{},{},{},{},{},\"{}\"",
                log.table_name,
                log.row_idx,
                log.bus_id,
                log.is_sender,
                log.multiplicity,
                log.fingerprint,
                bus_elements_str.replace('"', "'")
            )?;
        }

        eprintln!("[BusDebugTracker] Wrote {} entries to {}", self.logs.len(), path);
        Ok(())
    }

    /// Clear all logs (useful for multiple test runs)
    pub fn clear(&mut self) {
        self.logs.clear();
    }
}

/// Analysis result showing where mismatches occur
#[derive(Debug)]
pub struct BusMismatchReport {
    pub imbalanced_buses: Vec<BusImbalance>,
}

impl BusMismatchReport {
    /// Print summary to stderr
    pub fn print_summary(&self) {
        if self.imbalanced_buses.is_empty() {
            eprintln!("\n=== BUS MISMATCH ANALYSIS ===");
            eprintln!("All tracked buses appear balanced (no orphans or multiplicity mismatches found)");
            eprintln!("=== END ANALYSIS ===\n");
            return;
        }

        eprintln!("\n=== BUS MISMATCH ANALYSIS ===");

        for imbalance in &self.imbalanced_buses {
            let issue_count = imbalance.orphan_senders.len()
                + imbalance.orphan_receivers.len()
                + imbalance.multiplicity_mismatches.len();

            eprintln!(
                "\nBus {}: {} issue(s) found",
                imbalance.bus_id, issue_count
            );

            // Orphan senders
            if !imbalance.orphan_senders.is_empty() {
                eprintln!("\n  ORPHAN SENDERS (sent but not received):");
                for (i, orphan) in imbalance.orphan_senders.iter().enumerate() {
                    if i >= 10 {
                        eprintln!(
                            "    ... and {} more",
                            imbalance.orphan_senders.len() - 10
                        );
                        break;
                    }
                    eprintln!(
                        "    {} row {}: mult={} fp={}",
                        orphan.table_name, orphan.row_idx, orphan.multiplicity, orphan.fingerprint
                    );
                    if orphan.bus_elements.len() <= 10 {
                        eprintln!("      bus_elements: {:?}", orphan.bus_elements);
                    } else {
                        eprintln!(
                            "      bus_elements: [{}, {}, ... ({} total)]",
                            orphan.bus_elements[0],
                            orphan.bus_elements[1],
                            orphan.bus_elements.len()
                        );
                    }
                }
            }

            // Orphan receivers
            if !imbalance.orphan_receivers.is_empty() {
                eprintln!("\n  ORPHAN RECEIVERS (expected but not sent):");
                for (i, orphan) in imbalance.orphan_receivers.iter().enumerate() {
                    if i >= 10 {
                        eprintln!(
                            "    ... and {} more",
                            imbalance.orphan_receivers.len() - 10
                        );
                        break;
                    }
                    eprintln!(
                        "    {} row {}: mult={} fp={}",
                        orphan.table_name, orphan.row_idx, orphan.multiplicity, orphan.fingerprint
                    );
                    if orphan.bus_elements.len() <= 10 {
                        eprintln!("      bus_elements: {:?}", orphan.bus_elements);
                    } else {
                        eprintln!(
                            "      bus_elements: [{}, {}, ... ({} total)]",
                            orphan.bus_elements[0],
                            orphan.bus_elements[1],
                            orphan.bus_elements.len()
                        );
                    }
                }
            }

            // Multiplicity mismatches
            if !imbalance.multiplicity_mismatches.is_empty() {
                eprintln!("\n  MULTIPLICITY MISMATCHES:");
                for (i, mismatch) in imbalance.multiplicity_mismatches.iter().enumerate() {
                    if i >= 10 {
                        eprintln!(
                            "    ... and {} more",
                            imbalance.multiplicity_mismatches.len() - 10
                        );
                        break;
                    }
                    eprintln!("    fingerprint={}", mismatch.fingerprint);
                    // Print concise sender summary
                    if mismatch.senders.len() <= 3 {
                        eprintln!(
                            "      Sent: {} ({:?})",
                            mismatch.total_sent, mismatch.senders
                        );
                    } else {
                        let first_3: Vec<_> = mismatch.senders.iter().take(3).collect();
                        eprintln!(
                            "      Sent: {} (from {} rows: {:?}...)",
                            mismatch.total_sent, mismatch.senders.len(), first_3
                        );
                    }
                    // Print concise receiver summary
                    if mismatch.receivers.len() <= 3 {
                        eprintln!(
                            "      Received: {} ({:?})",
                            mismatch.total_received, mismatch.receivers
                        );
                    } else {
                        let first_3: Vec<_> = mismatch.receivers.iter().take(3).collect();
                        eprintln!(
                            "      Received: {} (from {} rows: {:?}...)",
                            mismatch.total_received, mismatch.receivers.len(), first_3
                        );
                    }
                    let diff = mismatch.total_sent as i64 - mismatch.total_received as i64;
                    if diff > 0 {
                        eprintln!("      Deficit: {} lookup(s) not received!", diff);
                    } else {
                        eprintln!("      Excess: {} lookup(s) received without sending!", -diff);
                    }
                }
            }
        }

        eprintln!("\n=== END ANALYSIS ===\n");
    }

    /// Check if any mismatches were found
    pub fn has_mismatches(&self) -> bool {
        !self.imbalanced_buses.is_empty()
    }
}

#[derive(Debug)]
pub struct BusImbalance {
    pub bus_id: u64,
    /// Fingerprints that appear in senders but not receivers
    pub orphan_senders: Vec<OrphanInteraction>,
    /// Fingerprints that appear in receivers but not senders
    pub orphan_receivers: Vec<OrphanInteraction>,
    /// Fingerprints with mismatched multiplicities
    pub multiplicity_mismatches: Vec<MultiplicityMismatch>,
}

#[derive(Debug)]
pub struct OrphanInteraction {
    pub table_name: String,
    pub row_idx: usize,
    pub multiplicity: u64,
    pub fingerprint: String,
    pub bus_elements: Vec<String>,
}

#[derive(Debug)]
pub struct MultiplicityMismatch {
    pub fingerprint: String,
    pub bus_elements: Vec<String>,
    pub total_sent: u64,
    pub total_received: u64,
    pub senders: Vec<(String, usize, u64)>,   // (table, row, mult)
    pub receivers: Vec<(String, usize, u64)>, // (table, row, mult)
}

/// Helper to convert a field element's representative to u64 (if small enough)
/// Used for logging multiplicity values
pub fn field_to_u64_saturating<F>(val: &math::field::element::FieldElement<F>) -> u64
where
    F: math::field::traits::IsField,
{
    // Get the representative as a string and parse
    let repr = format!("{:?}", val);
    // Try to parse directly - if it fails or overflows, return u64::MAX
    repr.trim_start_matches("0x")
        .parse::<u64>()
        .unwrap_or_else(|_| {
            // Try hex parsing
            u64::from_str_radix(repr.trim_start_matches("0x"), 16).unwrap_or(u64::MAX)
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_tracker() {
        let tracker = BusDebugTracker {
            bus_filter: None,
            logs: Vec::new(),
        };
        let report = tracker.analyze_mismatches();
        assert!(report.imbalanced_buses.is_empty());
    }

    #[test]
    fn test_balanced_bus() {
        let tracker = BusDebugTracker {
            bus_filter: None,
            logs: vec![
                BusInteractionLog {
                    table_name: "CPU".to_string(),
                    row_idx: 0,
                    bus_id: 14,
                    is_sender: true,
                    multiplicity: 1,
                    bus_elements: vec!["14".to_string(), "0x1234".to_string()],
                    fingerprint: "0xABCD".to_string(),
                },
                BusInteractionLog {
                    table_name: "MEMW".to_string(),
                    row_idx: 0,
                    bus_id: 14,
                    is_sender: false,
                    multiplicity: 1,
                    bus_elements: vec!["14".to_string(), "0x1234".to_string()],
                    fingerprint: "0xABCD".to_string(),
                },
            ],
        };
        let report = tracker.analyze_mismatches();
        assert!(report.imbalanced_buses.is_empty());
    }

    #[test]
    fn test_orphan_sender() {
        let tracker = BusDebugTracker {
            bus_filter: None,
            logs: vec![
                BusInteractionLog {
                    table_name: "CPU".to_string(),
                    row_idx: 42,
                    bus_id: 14,
                    is_sender: true,
                    multiplicity: 1,
                    bus_elements: vec!["14".to_string(), "0x5678".to_string()],
                    fingerprint: "0x1111".to_string(),
                },
                // No receiver for this fingerprint
            ],
        };
        let report = tracker.analyze_mismatches();
        assert_eq!(report.imbalanced_buses.len(), 1);
        assert_eq!(report.imbalanced_buses[0].orphan_senders.len(), 1);
        assert_eq!(report.imbalanced_buses[0].orphan_senders[0].row_idx, 42);
    }

    #[test]
    fn test_multiplicity_mismatch() {
        let tracker = BusDebugTracker {
            bus_filter: None,
            logs: vec![
                BusInteractionLog {
                    table_name: "CPU".to_string(),
                    row_idx: 10,
                    bus_id: 14,
                    is_sender: true,
                    multiplicity: 2,
                    bus_elements: vec!["14".to_string()],
                    fingerprint: "0xAAAA".to_string(),
                },
                BusInteractionLog {
                    table_name: "LOAD".to_string(),
                    row_idx: 5,
                    bus_id: 14,
                    is_sender: true,
                    multiplicity: 1,
                    bus_elements: vec!["14".to_string()],
                    fingerprint: "0xAAAA".to_string(),
                },
                BusInteractionLog {
                    table_name: "MEMW".to_string(),
                    row_idx: 0,
                    bus_id: 14,
                    is_sender: false,
                    multiplicity: 2, // Should be 3!
                    bus_elements: vec!["14".to_string()],
                    fingerprint: "0xAAAA".to_string(),
                },
            ],
        };
        let report = tracker.analyze_mismatches();
        assert_eq!(report.imbalanced_buses.len(), 1);
        assert_eq!(report.imbalanced_buses[0].multiplicity_mismatches.len(), 1);
        let mismatch = &report.imbalanced_buses[0].multiplicity_mismatches[0];
        assert_eq!(mismatch.total_sent, 3);
        assert_eq!(mismatch.total_received, 2);
    }
}
