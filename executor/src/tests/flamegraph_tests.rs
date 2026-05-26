//! Tests for symbol demangling in flamegraph generation.

use crate::flamegraph::demangle;

#[test]
fn test_demangle_simple() {
    assert_eq!(demangle("main"), "main");
    assert_eq!(demangle("_start"), "_start");
}
