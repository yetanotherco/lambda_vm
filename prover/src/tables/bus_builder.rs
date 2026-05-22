//! Builder for table `bus_interactions()` declarations.
//!
//! Each table's `bus_interactions()` function returns a `Vec<BusInteraction>`
//! describing every lookup the table sends or receives. The same idioms repeat
//! across tables: send a direct-packed column to a halfword range check, send
//! a (column, derived) pair to MSB16, range-check a virtual linear combination
//! via IS_B20, etc.
//!
//! Without a builder these are 5-7 line `BusInteraction::sender(...)` blocks
//! with a `vec![BusValue::Packed{...}]` argument, repeated dozens of times.
//! The builder reduces each interaction to a single intent-named call that
//! reads as a spec line ("send `col` to IS_HALFWORD with multiplicity mu").
//!
//! No macros: plain Rust methods, discoverable via `rust-analyzer`. Heterogeneous
//! interactions that do not match a named helper use the generic `send` / `recv`.

use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};

use super::types::BusId;

/// Accumulator for a table's `bus_interactions()` declaration.
pub struct BusInteractionsBuilder {
    inner: Vec<BusInteraction>,
}

impl BusInteractionsBuilder {
    /// Create a builder, pre-sizing for `n` interactions. Every `bus_interactions()`
    /// knows its interaction count up front, so there is no zero-arg constructor.
    pub fn with_capacity(n: usize) -> Self {
        Self {
            inner: Vec::with_capacity(n),
        }
    }

    pub fn into_vec(self) -> Vec<BusInteraction> {
        self.inner
    }

    // -------------------------------------------------------------------------
    // Sender helpers
    // -------------------------------------------------------------------------

    /// Send a single direct-packed column to a halfword range check.
    /// IS_HALFWORD[col]
    ///
    /// `mult` is taken by reference and cloned so callers can reuse the same
    /// `Multiplicity` value across many calls without an explicit `.clone()`
    /// at each site.
    pub fn send_halfword(&mut self, col: usize, mult: &Multiplicity) -> &mut Self {
        self.inner.push(BusInteraction::sender(
            BusId::IsHalfword,
            mult.clone(),
            vec![packed_direct(col)],
        ));
        self
    }

    /// Send a (input, output) pair to MSB16.
    /// MSB16[input_col] -> output_col
    pub fn send_msb16(
        &mut self,
        input_col: usize,
        output_col: usize,
        mult: &Multiplicity,
    ) -> &mut Self {
        self.inner.push(BusInteraction::sender(
            BusId::Msb16,
            mult.clone(),
            vec![packed_direct(input_col), packed_direct(output_col)],
        ));
        self
    }

    /// Send a B20 range check on a virtual linear-combination value (e.g. a carry).
    /// IS_B20[linear_terms]
    pub fn send_b20_linear(&mut self, mult: &Multiplicity, terms: Vec<LinearTerm>) -> &mut Self {
        self.inner.push(BusInteraction::sender(
            BusId::IsB20,
            mult.clone(),
            vec![BusValue::linear(terms)],
        ));
        self
    }

    /// Generic sender with caller-provided values. Takes ownership of `mult`;
    /// use when sending exactly one interaction with this multiplicity.
    pub fn send(&mut self, bus_id: BusId, mult: Multiplicity, values: Vec<BusValue>) -> &mut Self {
        self.inner
            .push(BusInteraction::sender(bus_id, mult, values));
        self
    }

    /// Send an XOR_BYTE lookup over three direct-packed columns (x, y, x ^ y).
    pub fn send_xor_byte(
        &mut self,
        x_col: usize,
        y_col: usize,
        result_col: usize,
        mult: &Multiplicity,
    ) -> &mut Self {
        self.inner.push(BusInteraction::sender(
            BusId::XorByte,
            mult.clone(),
            vec![
                packed_direct(x_col),
                packed_direct(y_col),
                packed_direct(result_col),
            ],
        ));
        self
    }

    // -------------------------------------------------------------------------
    // Receiver helpers
    // -------------------------------------------------------------------------

    /// Generic receiver with caller-provided values.
    pub fn recv(&mut self, bus_id: BusId, mult: Multiplicity, values: Vec<BusValue>) -> &mut Self {
        self.inner
            .push(BusInteraction::receiver(bus_id, mult, values));
        self
    }
}

/// Build a `BusValue::Packed` with `Packing::Direct` from a column index.
/// Exported because some callers compose a `vec![packed_direct(c), ...]` by hand.
pub fn packed_direct(col: usize) -> BusValue {
    BusValue::Packed {
        start_column: col,
        packing: Packing::Direct,
    }
}
