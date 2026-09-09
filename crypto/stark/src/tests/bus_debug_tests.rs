use crate::bus_debug::{BusDebugTracker, BusInteractionLog};

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
