ASM_PROGRAMS_DIR=./programs/asm
ASM_ARTIFACTS_DIR=./program_artifacts/asm

RUST_PROGRAMS_DIR=./programs/rust
RUST_ARTIFACTS_DIR=./program_artifacts/rust


ASM_PROGRAMS = $(wildcard $(ASM_PROGRAMS_DIR)/*.s)
ARTIFACTS_ASM = $(patsubst $(ASM_PROGRAMS_DIR)/%.s, $(ASM_ARTIFACTS_DIR)/%.elf, $(ASM_PROGRAMS))

RUST_PROGRAM_DIRS := $(dir $(wildcard $(RUST_PROGRAMS_DIR)/*/Cargo.toml))
RUST_PROGRAMS := $(notdir $(basename $(RUST_PROGRAM_DIRS:%/=%)))
RUST_ARTIFACTS := $(addprefix $(RUST_ARTIFACTS_DIR)/, $(addsuffix .elf, $(RUST_PROGRAMS)))

compile-programs-asm: clean-asm $(ARTIFACTS_ASM)

compile-programs-rust: clean-rust $(RUST_ARTIFACTS)

compile-programs: compile-programs-asm compile-programs-rust

# Compile assembly .s -> .o
$(ASM_ARTIFACTS_DIR)/%.o: $(ASM_PROGRAMS_DIR)/%.s
	clang --target=riscv32 -c $< -o $@

# Link assembly .o -> .elf
$(ASM_ARTIFACTS_DIR)/%.elf: $(ASM_ARTIFACTS_DIR)/%.o
	riscv64-unknown-elf-ld -m elf32lriscv $< -o $@ -e main

# Compile rust
$(RUST_ARTIFACTS_DIR)/%.elf: $(RUST_PROGRAMS_DIR)/%/Cargo.toml
	cd $(RUST_PROGRAMS_DIR)/$* && \
		cargo +nightly rustc \
			--target riscv32im-unknown-none-elf \
			-Z build-std=core,compiler_builtins \
			-- --emit asm -C debuginfo=0 -C link-arg=-e -C link-arg=main
	cp $(RUST_PROGRAMS_DIR)/$*/target/riscv32im-unknown-none-elf/debug/$* $@
	rm -rf $(RUST_PROGRAMS_DIR)/$*/target

clean-asm:
	-rm -rf $(ASM_ARTIFACTS_DIR)
	mkdir -p $(ASM_ARTIFACTS_DIR)

clean-rust:
	-rm -rf $(RUST_ARTIFACTS_DIR)
	mkdir -p $(RUST_ARTIFACTS_DIR)

clean: clean-asm clean-rust

test: compile-programs test-no-compile

test-asm: compile-programs-asm test-asm-no-compile

test-asm-no-compile:
	cargo test --test asm

test-rust: compile-programs-rust
	cargo test --test rust

test-no-compile:
	cargo test
