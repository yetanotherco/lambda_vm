# Assembly obligations — debts the epoch-verifier assembly must discharge

Started 2026-07-31. Each entry is a deferral whose safety argument is still
owed (standing-decisions method rule 5). Assembly (RESUME item 5) may not be
called done while any entry is OPEN. Add entries as legs flag them; close an
entry only with the verifying evidence named in it.

## OPEN

1. **`reg_fini` felt-width gap** (flagged by reg-tree, slice 1).
   Production's `reg_fini` is `Vec<u32>` — the TYPE is the entire
   enforcement. An LFM arena is untyped felts, so the machine's accepted set
   is wider than production's. Guard test
   `the_derivation_extends_a_non_u32_register_value_demonstrating_hazard`
   asserts the gap still exists. Assembly owes ONE of:
   - a 67-per-column range check on the register boundary columns, OR
   - the verified argument that no epoch proof can exist over a >u32
     register column (plausible via REG-C2's Memory-bus value word —
     currently UNVERIFIED; verifying it means a coherent-forgery analysis
     per method rule 4, not an assertion).
   Default is the range check: if assembly arrives and the argument is
   still unverified, emit the check.

## STANDING (from the RESUME, restated so this file is self-contained)

- Every per-epoch number so far is a COMPOSITION of per-AIR measurements,
  not a run. Assembly is what confirms or falsifies them.
- The arena-value join obligation (constraint/DEEP values = authenticated
  values) is DISCHARGED for the DEEP leg (deep-join slice 1, by shared
  cells + bound index). Any NEW leg that reads opened values (FRI, LogUp
  closure) inherits the same obligation and must join through the same
  cells, not parallel copies.
