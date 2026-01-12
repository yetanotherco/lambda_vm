UNAME := $(shell uname)

ASM_PROGRAMS_DIR=./programs/asm
ASM_ARTIFACTS_DIR=./program_artifacts/asm

RUST_PROGRAMS_DIR=./programs/rust
RUST_ARTIFACTS_DIR=./program_artifacts/rust

BENCH_PROGRAMS_DIR=./programs/bench
BENCH_ARTIFACTS_DIR=./program_artifacts/bench

ASM_PROGRAMS = $(wildcard $(ASM_PROGRAMS_DIR)/*.s)
ARTIFACTS_ASM = $(patsubst $(ASM_PROGRAMS_DIR)/%.s, $(ASM_ARTIFACTS_DIR)/%.elf, $(ASM_PROGRAMS))

RUST_PROGRAM_DIRS := $(dir $(wildcard $(RUST_PROGRAMS_DIR)/*/Cargo.toml))
RUST_PROGRAMS := $(notdir $(basename $(RUST_PROGRAM_DIRS:%/=%)))
RUST_ARTIFACTS := $(addprefix $(RUST_ARTIFACTS_DIR)/, $(addsuffix .elf, $(RUST_PROGRAMS)))

BENCH_PROGRAM_DIRS := $(dir $(wildcard $(BENCH_PROGRAMS_DIR)/*/Cargo.toml))
BENCH_PROGRAMS := $(notdir $(basename $(BENCH_PROGRAM_DIRS:%/=%)))
BENCH_ARTIFACTS := $(addprefix $(BENCH_ARTIFACTS_DIR)/, $(addsuffix .elf, $(BENCH_PROGRAMS)))

compile-programs-asm: clean-asm $(ARTIFACTS_ASM)

compile-programs-rust: clean-rust $(RUST_ARTIFACTS)

compile-bench: clean-bench $(BENCH_ARTIFACTS)

compile-programs: compile-programs-asm compile-programs-rust compile-bench

# Compile assembly .s -> .o
$(ASM_ARTIFACTS_DIR)/%.o: $(ASM_PROGRAMS_DIR)/%.s
	clang --target=riscv32 -c $< -o $@

# Link assembly .o -> .elf
$(ASM_ARTIFACTS_DIR)/%.elf: $(ASM_ARTIFACTS_DIR)/%.o
	riscv64-unknown-elf-ld -m elf32lriscv $< -o $@ -e main

# Compile rust
$(RUST_ARTIFACTS_DIR)/%.elf: $(RUST_PROGRAMS_DIR)/%/Cargo.toml
	cd $(RUST_PROGRAMS_DIR)/$* && \
		cargo build \
			--target riscv32im-lambda-vm-elf \
			-Z build-std=core,alloc,std,compiler_builtins,panic_abort \
			-Z build-std-features=compiler-builtins-mem 
	cp $(RUST_PROGRAMS_DIR)/$*/target/riscv32im-lambda-vm-elf/debug/$* $@

# Compile rust benches
$(BENCH_ARTIFACTS_DIR)/%.elf: $(BENCH_PROGRAMS_DIR)/%/Cargo.toml
	cd $(BENCH_PROGRAMS_DIR)/$* && \
		cargo build \
			--target riscv32im-lambda-vm-elf \
			-Z build-std=core,compiler_builtins \
			-Z build-std-features=compiler-builtins-mem 
	cp $(BENCH_PROGRAMS_DIR)/$*/target/riscv32im-lambda-vm-elf/debug/$* $@
	rm -rf $(BENCH_PROGRAMS_DIR)/$*/target

clean-asm:
	-rm -rf $(ASM_ARTIFACTS_DIR)
	mkdir -p $(ASM_ARTIFACTS_DIR)

clean-rust:
	-rm -rf $(RUST_ARTIFACTS_DIR)
	mkdir -p $(RUST_ARTIFACTS_DIR)

clean-bench:
	-rm -rf $(BENCH_ARTIFACTS_DIR)
	mkdir -p $(BENCH_ARTIFACTS_DIR)

clean: clean-asm clean-rust clean-bench

test: compile-programs test-no-compile

test-asm: compile-programs-asm test-asm-no-compile

test-asm-no-compile:
	cargo test --test asm

test-rust: compile-programs-rust
	cargo test --test rust

test-no-compile:
	cargo test

.PHONY: deps
deps:
ifeq ($(UNAME), Linux)
deps: deps-linux
endif
ifeq ($(UNAME), Darwin)
deps: deps-macos
endif

.PHONY: deps-linux
deps-linux:
	@# TODO
	@echo "not yet implemented"
	@exit 1

.PHONY: deps-macos
deps-macos:
	brew tap riscv-software-src/riscv
	brew install riscv-software-src/riscv/riscv-gnu-toolchain
