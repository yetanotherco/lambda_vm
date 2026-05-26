use executor::{
    elf::{FunctionSymbol, SymbolTable},
    flamegraph::FlamegraphGenerator,
    vm::{
        execution::InstructionCache,
        instruction::decoding::{DecodedInstruction, Instruction},
        logs::Log,
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
    let map: U64HashMap<DecodedInstruction> = instructions
        .into_iter()
        .map(|(addr, instr)| (addr, DecodedInstruction { instr, len: 4 }))
        .collect();
    InstructionCache::from_map(&map)
}

/// Helper to create a simple non-jump instruction (LoadUpperImm is simple)
fn nop_instruction() -> Instruction {
    Instruction::LoadUpperImm { dst: 0, imm: 0 }
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
