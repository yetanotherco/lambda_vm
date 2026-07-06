use executor::{
    elf::{FunctionSymbol, SymbolTable},
    flamegraph::FlamegraphGenerator,
    vm::{
        execution::InstructionCache, instruction::decoding::Instruction, logs::Log,
        memory::U64HashMap,
    },
};

/// Helper to create a SymbolTable from a list of (name, address, size) tuples
fn make_symbol_table(symbols: Vec<(&str, u64, u64)>) -> SymbolTable {
    let mut functions: Vec<FunctionSymbol> = symbols
        .into_iter()
        .map(|(name, address, size)| FunctionSymbol {
            name: name.to_string(),
            address,
            size,
        })
        .collect();
    functions.sort_by_key(|f| f.address);
    SymbolTable::from_functions(functions)
}

/// Helper to create an instruction cache
fn make_instructions(instructions: Vec<(u64, Instruction)>) -> InstructionCache {
    let map: U64HashMap<Instruction> = instructions.into_iter().collect();
    InstructionCache::from_map(&map)
}

/// Helper to create a simple non-jump instruction (LoadUpperImm is simple)
fn nop_instruction() -> Instruction {
    Instruction::LoadUpperImm { dst: 0, imm: 0 }
}

/// Helper to build a `Log` for a plain PC transition (no register values needed
/// by any flamegraph test).
fn mk_log(current_pc: u64, next_pc: u64) -> Log {
    Log {
        current_pc,
        next_pc,
        src1_val: 0,
        src2_val: 0,
        dst_val: 0,
    }
}

// ============================================================================
// SymbolTable::lookup tests
// ============================================================================

#[test]
fn test_symbol_lookup_exact_match() {
    let table = make_symbol_table(vec![
        ("main", 0x1000, 100),
        ("foo", 0x1100, 50),
        ("bar", 0x1200, 80),
    ]);

    // Exact match on function start
    let sym = table.lookup(0x1000).unwrap();
    assert_eq!(sym.name, "main");

    let sym = table.lookup(0x1100).unwrap();
    assert_eq!(sym.name, "foo");
}

#[test]
fn test_symbol_lookup_within_function() {
    let table = make_symbol_table(vec![
        ("main", 0x1000, 100), // covers 0x1000-0x1063
        ("foo", 0x1100, 50),
    ]);

    // Address within function bounds
    let sym = table.lookup(0x1050).unwrap();
    assert_eq!(sym.name, "main");

    // Last byte of function (0x1000 + 99 = 0x1063)
    let sym = table.lookup(0x1063).unwrap();
    assert_eq!(sym.name, "main");

    // Just past the function boundary (should return None)
    assert!(table.lookup(0x1064).is_none());
}

#[test]
fn test_symbol_lookup_outside_bounds() {
    let table = make_symbol_table(vec![("main", 0x1000, 100), ("foo", 0x1200, 50)]);

    // Address before first function
    assert!(table.lookup(0x500).is_none());

    // Address between functions (after main ends, before foo starts)
    assert!(table.lookup(0x1100).is_none());

    // Address after all functions
    assert!(table.lookup(0x2000).is_none());
}

#[test]
fn test_symbol_lookup_zero_size() {
    // Zero-size symbols (common in ASM) should match any address >= start
    let table = make_symbol_table(vec![("asm_func", 0x1000, 0), ("next_func", 0x1100, 50)]);

    // Should match asm_func since size is 0
    let sym = table.lookup(0x1000).unwrap();
    assert_eq!(sym.name, "asm_func");

    let sym = table.lookup(0x1050).unwrap();
    assert_eq!(sym.name, "asm_func");

    // But once we hit next_func, that takes over
    let sym = table.lookup(0x1100).unwrap();
    assert_eq!(sym.name, "next_func");
}

#[test]
fn test_symbol_lookup_empty_table() {
    let table = SymbolTable::default();
    assert!(table.lookup(0x1000).is_none());
    assert!(table.is_empty());
    assert_eq!(table.len(), 0);
}

#[test]
fn test_symbol_lookup_overlapping_functions() {
    // In practice, functions shouldn't overlap, but test the behavior
    // Binary search finds the last function with address <= target
    let table = make_symbol_table(vec![
        ("outer", 0x1000, 200),
        ("inner", 0x1050, 50), // "nested" inside outer
    ]);

    // At inner's start, inner takes precedence
    let sym = table.lookup(0x1050).unwrap();
    assert_eq!(sym.name, "inner");

    // Before inner, outer is found
    let sym = table.lookup(0x1040).unwrap();
    assert_eq!(sym.name, "outer");
}

// ============================================================================
// FlamegraphGenerator tests
// ============================================================================

#[test]
fn test_flamegraph_simple_call_return() {
    let symbols = make_symbol_table(vec![("main", 0x1000, 100), ("foo", 0x2000, 50)]);
    let mut generator = FlamegraphGenerator::new(symbols, 0x1000);

    // Simulate: main calls foo, foo returns
    let instructions = make_instructions(vec![
        // main: some instructions
        (0x1000, nop_instruction()),
        (0x1004, nop_instruction()),
        // main: call foo (JAL with dst=ra, register 1)
        (0x1008, Instruction::JumpAndLink { dst: 1, offset: 0 }),
        // foo: some instructions
        (0x2000, nop_instruction()),
        (0x2004, nop_instruction()),
        // foo: return (JALR with base=ra, dst=zero)
        (
            0x2008,
            Instruction::JumpAndLinkRegister {
                dst: 0,
                base: 1,
                offset: 0,
            },
        ),
        // main: after return
        (0x100c, nop_instruction()),
    ]);

    let logs = vec![
        Log {
            current_pc: 0x1000,
            next_pc: 0x1004,
            src1_val: 0,
            src2_val: 0,
            dst_val: 0,
        },
        Log {
            current_pc: 0x1004,
            next_pc: 0x1008,
            src1_val: 0,
            src2_val: 0,
            dst_val: 0,
        },
        Log {
            current_pc: 0x1008,
            next_pc: 0x2000,
            src1_val: 0,
            src2_val: 0,
            dst_val: 0,
        }, // call
        Log {
            current_pc: 0x2000,
            next_pc: 0x2004,
            src1_val: 0,
            src2_val: 0,
            dst_val: 0,
        },
        Log {
            current_pc: 0x2004,
            next_pc: 0x2008,
            src1_val: 0,
            src2_val: 0,
            dst_val: 0,
        },
        Log {
            current_pc: 0x2008,
            next_pc: 0x100c,
            src1_val: 0,
            src2_val: 0,
            dst_val: 0,
        }, // return
        Log {
            current_pc: 0x100c,
            next_pc: 0x1010,
            src1_val: 0,
            src2_val: 0,
            dst_val: 0,
        },
    ];

    generator.process_logs(&logs, &instructions).unwrap();

    assert_eq!(generator.total_instructions(), 7);

    let mut output = Vec::new();
    generator.write_folded(&mut output).unwrap();
    let output_str = String::from_utf8(output).unwrap();

    // main should have some instructions, main;foo should have some
    assert!(output_str.contains("main "));
    assert!(output_str.contains("main;foo "));
}

#[test]
fn test_flamegraph_stack_underflow_protection() {
    // Test that returning from root doesn't cause issues
    let symbols = make_symbol_table(vec![("main", 0x1000, 100)]);
    let mut generator = FlamegraphGenerator::new(symbols, 0x1000);

    let instructions = make_instructions(vec![
        (0x1000, nop_instruction()),
        // Return instruction at root level
        (
            0x1004,
            Instruction::JumpAndLinkRegister {
                dst: 0,
                base: 1,
                offset: 0,
            },
        ),
        (0x1008, nop_instruction()),
    ]);

    let logs = vec![
        Log {
            current_pc: 0x1000,
            next_pc: 0x1004,
            src1_val: 0,
            src2_val: 0,
            dst_val: 0,
        },
        Log {
            current_pc: 0x1004,
            next_pc: 0x1008,
            src1_val: 0,
            src2_val: 0,
            dst_val: 0,
        }, // return at root
        Log {
            current_pc: 0x1008,
            next_pc: 0x100c,
            src1_val: 0,
            src2_val: 0,
            dst_val: 0,
        },
    ];

    // Should not panic
    generator.process_logs(&logs, &instructions).unwrap();

    assert_eq!(generator.total_instructions(), 3);

    let mut output = Vec::new();
    generator.write_folded(&mut output).unwrap();
    let output_str = String::from_utf8(output).unwrap();

    // All instructions should be under main (no <root> because we protect against underflow)
    assert!(output_str.contains("main 3"));
}

#[test]
fn test_flamegraph_tail_call() {
    let symbols = make_symbol_table(vec![
        ("main", 0x1000, 100),
        ("foo", 0x2000, 50),
        ("bar", 0x3000, 50),
    ]);
    let mut generator = FlamegraphGenerator::new(symbols, 0x1000);

    let instructions = make_instructions(vec![
        (0x1000, nop_instruction()),
        // main: call foo
        (0x1004, Instruction::JumpAndLink { dst: 1, offset: 0 }),
        (0x2000, nop_instruction()),
        // foo: tail call to bar (JAL with dst=0, doesn't save return address)
        (0x2004, Instruction::JumpAndLink { dst: 0, offset: 0 }),
        (0x3000, nop_instruction()),
        // bar: return
        (
            0x3004,
            Instruction::JumpAndLinkRegister {
                dst: 0,
                base: 1,
                offset: 0,
            },
        ),
        (0x1008, nop_instruction()),
    ]);

    let logs = vec![
        Log {
            current_pc: 0x1000,
            next_pc: 0x1004,
            src1_val: 0,
            src2_val: 0,
            dst_val: 0,
        },
        Log {
            current_pc: 0x1004,
            next_pc: 0x2000,
            src1_val: 0,
            src2_val: 0,
            dst_val: 0,
        }, // call foo
        Log {
            current_pc: 0x2000,
            next_pc: 0x2004,
            src1_val: 0,
            src2_val: 0,
            dst_val: 0,
        },
        Log {
            current_pc: 0x2004,
            next_pc: 0x3000,
            src1_val: 0,
            src2_val: 0,
            dst_val: 0,
        }, // tail call bar
        Log {
            current_pc: 0x3000,
            next_pc: 0x3004,
            src1_val: 0,
            src2_val: 0,
            dst_val: 0,
        },
        Log {
            current_pc: 0x3004,
            next_pc: 0x1008,
            src1_val: 0,
            src2_val: 0,
            dst_val: 0,
        }, // return
        Log {
            current_pc: 0x1008,
            next_pc: 0x100c,
            src1_val: 0,
            src2_val: 0,
            dst_val: 0,
        },
    ];

    generator.process_logs(&logs, &instructions).unwrap();

    let mut output = Vec::new();
    generator.write_folded(&mut output).unwrap();
    let output_str = String::from_utf8(output).unwrap();

    // After tail call, bar replaces foo on the stack
    // So we should see main;foo and main;bar but NOT main;foo;bar
    assert!(output_str.contains("main "));
    assert!(output_str.contains("main;foo "));
    assert!(output_str.contains("main;bar "));
    assert!(!output_str.contains("main;foo;bar"));
}

#[test]
fn test_flamegraph_indirect_call() {
    let symbols = make_symbol_table(vec![("main", 0x1000, 100), ("callback", 0x2000, 50)]);
    let mut generator = FlamegraphGenerator::new(symbols, 0x1000);

    let instructions = make_instructions(vec![
        (0x1000, nop_instruction()),
        // Indirect call via JALR with dst=ra (register 1)
        (
            0x1004,
            Instruction::JumpAndLinkRegister {
                dst: 1,
                base: 5,
                offset: 0,
            },
        ),
        (0x2000, nop_instruction()),
        (
            0x2004,
            Instruction::JumpAndLinkRegister {
                dst: 0,
                base: 1,
                offset: 0,
            },
        ),
        (0x1008, nop_instruction()),
    ]);

    let logs = vec![
        Log {
            current_pc: 0x1000,
            next_pc: 0x1004,
            src1_val: 0,
            src2_val: 0,
            dst_val: 0,
        },
        Log {
            current_pc: 0x1004,
            next_pc: 0x2000,
            src1_val: 0,
            src2_val: 0,
            dst_val: 0,
        }, // indirect call
        Log {
            current_pc: 0x2000,
            next_pc: 0x2004,
            src1_val: 0,
            src2_val: 0,
            dst_val: 0,
        },
        Log {
            current_pc: 0x2004,
            next_pc: 0x1008,
            src1_val: 0,
            src2_val: 0,
            dst_val: 0,
        }, // return
        Log {
            current_pc: 0x1008,
            next_pc: 0x100c,
            src1_val: 0,
            src2_val: 0,
            dst_val: 0,
        },
    ];

    generator.process_logs(&logs, &instructions).unwrap();

    let mut output = Vec::new();
    generator.write_folded(&mut output).unwrap();
    let output_str = String::from_utf8(output).unwrap();

    assert!(output_str.contains("main;callback "));
}

#[test]
fn test_flamegraph_unknown_symbols() {
    // Empty symbol table - should fall back to hex addresses
    let symbols = SymbolTable::default();
    let mut generator = FlamegraphGenerator::new(symbols, 0x1000);

    let instructions = make_instructions(vec![(0x1000, nop_instruction())]);

    let logs = vec![Log {
        current_pc: 0x1000,
        next_pc: 0x1004,
        src1_val: 0,
        src2_val: 0,
        dst_val: 0,
    }];

    generator.process_logs(&logs, &instructions).unwrap();

    let mut output = Vec::new();
    generator.write_folded(&mut output).unwrap();
    let output_str = String::from_utf8(output).unwrap();

    // Should contain hex address as fallback
    assert!(output_str.contains("0x1000 1"));
}

#[test]
fn test_flamegraph_instruction_not_found_error() {
    let symbols = make_symbol_table(vec![("main", 0x1000, 100)]);
    let mut generator = FlamegraphGenerator::new(symbols, 0x1000);

    // Empty instruction map
    let instructions = make_instructions(vec![]);

    let logs = vec![Log {
        current_pc: 0x1000,
        next_pc: 0x1004,
        src1_val: 0,
        src2_val: 0,
        dst_val: 0,
    }];

    let result = generator.process_logs(&logs, &instructions);
    assert!(result.is_err());
}

// ============================================================================
// Tail-call misdetection regression tests
// ============================================================================

#[test]
fn test_flamegraph_intra_function_jal_x0_does_not_alter_stack() {
    // `jal x0, <label>` inside `main` (a loop back-edge or if/else jump) must
    // not be treated as a tail call: current_pc and next_pc both resolve to
    // `main`.
    let symbols = make_symbol_table(vec![("main", 0x1000, 100)]);
    let mut generator = FlamegraphGenerator::new(symbols, 0x1000);

    let instructions = make_instructions(vec![
        (0x1000, nop_instruction()),
        // Ordinary jump within main (dst=0), still inside main's bounds.
        (0x1004, Instruction::JumpAndLink { dst: 0, offset: 0 }),
        (0x1020, nop_instruction()),
    ]);

    let logs = vec![
        mk_log(0x1000, 0x1004),
        mk_log(0x1004, 0x1020), // intra-function jump, not a call
        mk_log(0x1020, 0x1024),
    ];

    generator.process_logs(&logs, &instructions).unwrap();

    let mut output = Vec::new();
    generator.write_folded(&mut output).unwrap();
    let output_str = String::from_utf8(output).unwrap();

    // Everything stays under "main" — no spurious pop+push produced a
    // separate stack state.
    assert_eq!(output_str.trim(), "main 3");
}

#[test]
fn test_flamegraph_intra_function_jalr_x0_does_not_alter_stack() {
    // Same as above but via JALR (e.g. a jump-table dispatch): dst=0, base
    // != ra, landing back inside the same function.
    let symbols = make_symbol_table(vec![("main", 0x1000, 100)]);
    let mut generator = FlamegraphGenerator::new(symbols, 0x1000);

    let instructions = make_instructions(vec![
        (0x1000, nop_instruction()),
        (
            0x1004,
            Instruction::JumpAndLinkRegister {
                dst: 0,
                base: 5,
                offset: 0,
            },
        ),
        (0x1020, nop_instruction()),
    ]);

    let logs = vec![mk_log(0x1000, 0x1004), mk_log(0x1004, 0x1020)];

    generator.process_logs(&logs, &instructions).unwrap();

    let mut output = Vec::new();
    generator.write_folded(&mut output).unwrap();
    let output_str = String::from_utf8(output).unwrap();

    assert_eq!(output_str.trim(), "main 2");
}

#[test]
fn test_flamegraph_cross_function_tail_call_mutual() {
    // Mutual tail calls: main calls f (dst=1), f tail-calls g (dst=0, cross-
    // function), g tail-calls f back (dst=0, cross-function). Must keep
    // producing pop+push on every cross-function dst=0 jump, exactly as
    // before the fix.
    let symbols = make_symbol_table(vec![
        ("main", 0x1000, 100),
        ("f", 0x2000, 50),
        ("g", 0x3000, 50),
    ]);
    let mut generator = FlamegraphGenerator::new(symbols, 0x1000);

    let instructions = make_instructions(vec![
        (0x1000, Instruction::JumpAndLink { dst: 1, offset: 0 }), // call f
        (0x2000, Instruction::JumpAndLink { dst: 0, offset: 0 }), // f tail-calls g
        (0x3000, Instruction::JumpAndLink { dst: 0, offset: 0 }), // g tail-calls f
        (0x2004, nop_instruction()),
    ]);

    let logs = vec![
        mk_log(0x1000, 0x2000), // call f
        mk_log(0x2000, 0x3000), // tail call f -> g
        mk_log(0x3000, 0x2004), // tail call g -> f
        mk_log(0x2004, 0x2008),
    ];

    generator.process_logs(&logs, &instructions).unwrap();

    let mut output = Vec::new();
    generator.write_folded(&mut output).unwrap();
    let output_str = String::from_utf8(output).unwrap();

    // Each tail call replaces the top frame, never nests.
    assert!(output_str.contains("main;f "));
    assert!(output_str.contains("main;g "));
    assert!(output_str.contains("main;f 2")); // f entered twice (initial call, then g->f)
    assert!(!output_str.contains("main;f;g"));
    assert!(!output_str.contains("main;g;f"));
}

#[test]
fn test_flamegraph_self_tail_recursion_does_not_alter_stack() {
    // `f` jumps (dst=0) back to its own entry point — classic self-tail-
    // recursion / loop-as-tail-call codegen. Must fold into a single reused
    // frame, not push a new one per iteration.
    let symbols = make_symbol_table(vec![("main", 0x1000, 100), ("f", 0x2000, 50)]);
    let mut generator = FlamegraphGenerator::new(symbols, 0x1000);

    let instructions = make_instructions(vec![
        (0x1000, Instruction::JumpAndLink { dst: 1, offset: 0 }), // main calls f
        (0x2000, nop_instruction()),
        // f tail-calls itself (dst=0, lands back at its own entry point)
        (0x2004, Instruction::JumpAndLink { dst: 0, offset: 0 }),
        (
            0x2008,
            Instruction::JumpAndLinkRegister {
                dst: 0,
                base: 1,
                offset: 0,
            },
        ),
    ]);

    let logs = vec![
        mk_log(0x1000, 0x2000), // call f
        mk_log(0x2000, 0x2004),
        mk_log(0x2004, 0x2000), // self tail-recursion: f -> f
        mk_log(0x2000, 0x2004),
        mk_log(0x2004, 0x2000), // again
        mk_log(0x2000, 0x2008),
        mk_log(0x2008, 0x1004), // return
    ];

    generator.process_logs(&logs, &instructions).unwrap();

    let mut output = Vec::new();
    generator.write_folded(&mut output).unwrap();
    let output_str = String::from_utf8(output).unwrap();

    // All of f's iterations fold into the single "main;f" frame — no
    // "main;f;f;f..." nesting from the repeated self-tail-recursion.
    assert!(output_str.contains("main;f "));
    assert!(!output_str.contains("main;f;f"));
    assert_eq!(output_str.lines().count(), 2); // just "main" and "main;f"
}

#[test]
fn test_flamegraph_regular_recursion_dst1_still_pushes() {
    // `jal ra, f` from inside `f` (dst=1, self-call) must still push a new
    // frame per call, unaffected by the dst=0 tail-call fix — same-function
    // resolution must not leak into the dst=1 call path.
    let symbols = make_symbol_table(vec![("main", 0x1000, 100), ("f", 0x2000, 50)]);
    let mut generator = FlamegraphGenerator::new(symbols, 0x1000);

    let instructions = make_instructions(vec![
        (0x1000, Instruction::JumpAndLink { dst: 1, offset: 0 }), // main calls f
        (0x2000, nop_instruction()),
        // f recursively calls itself (dst=1, real call, saves return address)
        (0x2004, Instruction::JumpAndLink { dst: 1, offset: 0 }),
        (0x2008, nop_instruction()),
        (
            0x200c,
            Instruction::JumpAndLinkRegister {
                dst: 0,
                base: 1,
                offset: 0,
            },
        ), // return
    ]);

    let logs = vec![
        mk_log(0x1000, 0x2000), // call f (depth 1)
        mk_log(0x2000, 0x2004),
        mk_log(0x2004, 0x2000), // f calls f (depth 2)
        mk_log(0x2000, 0x2008),
        mk_log(0x2008, 0x200c),
        mk_log(0x200c, 0x2008), // return to depth 1
        mk_log(0x2008, 0x200c),
        mk_log(0x200c, 0x1004), // return to main
    ];

    generator.process_logs(&logs, &instructions).unwrap();

    let mut output = Vec::new();
    generator.write_folded(&mut output).unwrap();
    let output_str = String::from_utf8(output).unwrap();

    // Real recursion nests: main;f at depth 1, main;f;f at depth 2.
    assert!(output_str.contains("main;f "));
    assert!(output_str.contains("main;f;f "));
}

#[test]
fn test_flamegraph_dst0_jump_onto_zero_size_symbol_boundary() {
    // Known misattribution risk: a dst=0 jump landing exactly on a
    // zero-size (stripped/ASM) symbol's start
    // address is accepted by `SymbolTable::lookup` regardless of where the
    // jump came from, since zero-size symbols have no upper bound. Pin
    // current behavior (treated as a tail call, since it resolves to a
    // *different* symbol than the jump's origin) rather than silently
    // regressing further.
    let symbols = make_symbol_table(vec![("main", 0x1000, 100), ("asm_stub", 0x2000, 0)]);
    let mut generator = FlamegraphGenerator::new(symbols, 0x1000);

    let instructions = make_instructions(vec![
        (0x1004, Instruction::JumpAndLink { dst: 0, offset: 0 }),
        (0x2000, nop_instruction()),
    ]);

    // Two logs: the jump itself (counted under the pre-jump state, "main"),
    // then one instruction after landing (counted under whatever state the
    // jump produced) — this is what surfaces whether the mutation happened.
    let logs = vec![mk_log(0x1004, 0x2000), mk_log(0x2000, 0x2004)];

    generator.process_logs(&logs, &instructions).unwrap();

    let mut output = Vec::new();
    generator.write_folded(&mut output).unwrap();
    let output_str = String::from_utf8(output).unwrap();

    // main -> asm_stub is treated as a tail call (different symbols), pinning
    // today's behavior: the jump instruction itself is charged to "main"
    // (the state before the jump takes effect), and the following
    // instruction is charged to "main;asm_stub".
    let mut lines: Vec<&str> = output_str.lines().collect();
    lines.sort();
    assert_eq!(lines, vec!["main 1", "main;asm_stub 1"]);
}

#[test]
fn test_flamegraph_dst0_jump_with_unresolved_address_is_not_a_tail_call() {
    // A dst=0 jump where the current or next PC resolves to no symbol (asm
    // stubs, code in linker gaps) must be treated as an ordinary jump — no
    // pop+push — matching `maybe_tail_call`'s doc comment. Regression guard
    // against the `_ => false` bug that spuriously mutated the stack in
    // unsymbolized regions.
    let symbols = make_symbol_table(vec![("main", 0x1000, 100)]);
    let mut generator = FlamegraphGenerator::new(symbols, 0x1000);

    // The jump lands at 0x9000, which no symbol covers (unresolved next_pc).
    let instructions = make_instructions(vec![
        (0x1004, Instruction::JumpAndLink { dst: 0, offset: 0 }),
        (0x9000, nop_instruction()),
    ]);

    let logs = vec![mk_log(0x1004, 0x9000), mk_log(0x9000, 0x9004)];

    generator.process_logs(&logs, &instructions).unwrap();

    let mut output = Vec::new();
    generator.write_folded(&mut output).unwrap();
    let output_str = String::from_utf8(output).unwrap();

    // No stack mutation: both instructions stay charged to the pre-jump state
    // ("main"). A spurious tail call would have produced a second "main;<...>"
    // frame for the unresolved landing.
    let mut lines: Vec<&str> = output_str.lines().collect();
    lines.sort();
    assert_eq!(lines, vec!["main 2"]);
}

#[test]
fn test_flamegraph_cached_range_respects_nested_symbol_boundary() {
    // `inner` (0x1050, size 50) is nested inside `outer` (0x1000, size 200).
    // The tail-call fast-path caches outer's range, but that range must be
    // capped at inner's start (0x1050), not outer.address+outer.size — else a
    // dst=0 jump from outer into inner would be swallowed as "intra-outer" and
    // the genuine cross-function tail call would be lost. Regression guard for
    // the `lookup_range` effective-end cap.
    let symbols = make_symbol_table(vec![("outer", 0x1000, 200), ("inner", 0x1050, 50)]);
    let mut generator = FlamegraphGenerator::new(symbols, 0x1000);

    let instructions = make_instructions(vec![
        (0x1004, Instruction::JumpAndLink { dst: 0, offset: 0 }),
        (0x1010, Instruction::JumpAndLink { dst: 0, offset: 0 }),
        (0x1050, nop_instruction()),
    ]);

    // First jump stays inside outer (populates the cache); second jump crosses
    // into nested inner and must register as a tail call.
    let logs = vec![
        mk_log(0x1004, 0x1010),
        mk_log(0x1010, 0x1050),
        mk_log(0x1050, 0x1054),
    ];

    generator.process_logs(&logs, &instructions).unwrap();

    let mut output = Vec::new();
    generator.write_folded(&mut output).unwrap();
    let output_str = String::from_utf8(output).unwrap();

    let mut lines: Vec<&str> = output_str.lines().collect();
    lines.sort();
    // Two intra-outer instructions under "outer"; the instruction after the
    // tail call charged to "outer;inner" — proving the boundary jump mutated.
    assert_eq!(lines, vec!["outer 2", "outer;inner 1"]);
}

// ============================================================================
// Trie-fold correctness
// ============================================================================

#[test]
fn test_flamegraph_trie_fold_matches_hand_computed_counts() {
    // main (2 insns) -> calls f (1 insn) -> f self-tail-recurses twice more
    // (2 insns) -> returns -> main (1 insn).
    let symbols = make_symbol_table(vec![("main", 0x1000, 100), ("f", 0x2000, 50)]);
    let mut generator = FlamegraphGenerator::new(symbols, 0x1000);

    let instructions = make_instructions(vec![
        (0x1000, nop_instruction()),
        (0x1004, Instruction::JumpAndLink { dst: 1, offset: 0 }), // call f
        (0x2000, nop_instruction()),
        (0x2004, Instruction::JumpAndLink { dst: 0, offset: 0 }), // self tail-recursion
        (
            0x2008,
            Instruction::JumpAndLinkRegister {
                dst: 0,
                base: 1,
                offset: 0,
            },
        ), // return
        (0x1008, nop_instruction()),
    ]);

    let logs = vec![
        mk_log(0x1000, 0x1004), // main insn 1
        mk_log(0x1004, 0x2000), // call f
        mk_log(0x2000, 0x2004), // f insn 1
        mk_log(0x2004, 0x2000), // self tail-recursion
        mk_log(0x2000, 0x2004), // f insn 2
        mk_log(0x2004, 0x2000), // self tail-recursion again
        mk_log(0x2000, 0x2008), // f insn 3
        mk_log(0x2008, 0x1008), // return
        mk_log(0x1008, 0x100c), // main insn 2
    ];

    generator.process_logs(&logs, &instructions).unwrap();

    assert_eq!(generator.total_instructions(), 9);

    let mut output = Vec::new();
    generator.write_folded(&mut output).unwrap();
    let output_str = String::from_utf8(output).unwrap();

    // Hand-computed: main gets 2 instructions directly (insn 1, insn 2);
    // main;f gets all 3 of f's instructions folded into one state (self
    // tail-recursion never nests) plus... wait, the call/tail-call/return
    // instructions themselves are also counted under whichever stack state
    // is active *when they execute* (before the jump takes effect):
    //   0x1000 (main), 0x1004 (main, the call insn) -> 2 under main
    //   0x2000, 0x2004, 0x2000, 0x2004, 0x2000 (all under main;f) -> 5
    //   0x2008 (the return insn, under main;f) -> 1 -> total main;f = 6
    //   0x1008 (main, after return) -> 1 -> total main = 3
    let mut lines: Vec<&str> = output_str.lines().collect();
    lines.sort();
    assert_eq!(lines, vec!["main 3", "main;f 6"]);
}
