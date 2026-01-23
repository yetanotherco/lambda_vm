.PHONY: deps deps-linux deps-macos prepare-test-data compile-programs-asm compile-programs-rust compile-bench \
compile-programs clean-asm clean-rust clean-bench clean-shared clean test test-asm test-no-compile \
test-asm-no-compile test-rust test-rust-no-compile test-executor flamegraph-prover \
test-fast test-prover test-prover-all build check

UNAME := $(shell uname)

deps:
ifeq ($(UNAME), Linux)
deps: deps-linux
endif
ifeq ($(UNAME), Darwin)
deps: deps-macos
endif

deps-linux:
	@# TODO
	@echo "not yet implemented"
	@exit 1

deps-macos:
	brew tap riscv-software-src/riscv
	brew install riscv-software-src/riscv/riscv-gnu-toolchain

ASM_PROGRAMS_DIR=./executor/programs/asm
ASM_ARTIFACTS_DIR=./executor/program_artifacts/asm

RUST_PROGRAMS_DIR=./executor/programs/rust
RUST_ARTIFACTS_DIR=./executor/program_artifacts/rust

BENCH_PROGRAMS_DIR=./executor/programs/bench
BENCH_ARTIFACTS_DIR=./executor/program_artifacts/bench

SHARED_TARGET_DIR=./executor/shared_target

ASM_PROGRAMS = $(wildcard $(ASM_PROGRAMS_DIR)/*.s)
ARTIFACTS_ASM = $(patsubst $(ASM_PROGRAMS_DIR)/%.s, $(ASM_ARTIFACTS_DIR)/%.elf, $(ASM_PROGRAMS))

RUST_PROGRAM_DIRS := $(dir $(wildcard $(RUST_PROGRAMS_DIR)/*/Cargo.toml))
RUST_PROGRAMS := $(notdir $(basename $(RUST_PROGRAM_DIRS:%/=%)))
RUST_ARTIFACTS := $(addprefix $(RUST_ARTIFACTS_DIR)/, $(addsuffix .elf, $(RUST_PROGRAMS)))

BENCH_PROGRAM_DIRS := $(dir $(wildcard $(BENCH_PROGRAMS_DIR)/*/Cargo.toml))
BENCH_PROGRAMS := $(notdir $(basename $(BENCH_PROGRAM_DIRS:%/=%)))
BENCH_ARTIFACTS := $(addprefix $(BENCH_ARTIFACTS_DIR)/, $(addsuffix .elf, $(BENCH_PROGRAMS)))

ETHREX_FILE := executor/tests/ethrex_hoodi.bin
ETHREX_URL := https://lambda.alignedlayer.com/ethrex_hoodi.bin

SYSROOT_DIR := /opt/lambda-vm-sysroot
SYSROOT_TARBALL := /tmp/lambda-vm-sysroot-rv64im.tar.gz
SYSROOT_URL := https://lambda.alignedlayer.com/lambda-vm-sysroot-rv64im.tar.gz

# Custom RV64IM target spec location
RV64_TARGET_SPEC=$(CURDIR)/executor/programs/riscv64im-lambda-vm-elf.json

.PHONY: test prepare-test-data prepare-sysroot

prepare-test-data:
	@if [ ! -f "$(ETHREX_FILE)" ]; then \
		echo "Downloading ethrex_hoodi.bin..."; \
		curl -L "$(ETHREX_URL)" -o "$(ETHREX_FILE)"; \
	else \
		echo "ethrex_hoodi.bin already exists"; \
	fi

prepare-sysroot:
	@if [ ! -d "$(SYSROOT_DIR)" ]; then \
		echo "Downloading lambda-vm-sysroot-rv64im.tar.gz..."; \
		curl -L "$(SYSROOT_URL)" -o "$(SYSROOT_TARBALL)"; \
		echo "Extracting sysroot to $(SYSROOT_DIR)..."; \
		sudo mkdir -p /opt && sudo tar -xzf "$(SYSROOT_TARBALL)" -C /opt; \
		rm "$(SYSROOT_TARBALL)"; \
	else \
		echo "Sysroot already exists at $(SYSROOT_DIR)"; \
	fi

compile-programs-asm: $(ARTIFACTS_ASM)

compile-programs-rust: prepare-sysroot $(RUST_ARTIFACTS)

compile-bench: $(BENCH_ARTIFACTS)

compile-programs: compile-programs-asm compile-programs-rust compile-bench

# Compile and link assembly directly with clang (64-bit)
$(ASM_ARTIFACTS_DIR)/%.elf: $(ASM_PROGRAMS_DIR)/%.s
	@mkdir -p $(ASM_ARTIFACTS_DIR)
	clang --target=riscv64 -fuse-ld=lld -nostdlib -Wl,-e,main $< -o $@

# Compile rust (64-bit)
$(RUST_ARTIFACTS_DIR)/%.elf: $(RUST_PROGRAMS_DIR)/%/Cargo.toml
	@mkdir -p $(RUST_ARTIFACTS_DIR)
	cd $(RUST_PROGRAMS_DIR)/$* && \
		CARGO_TARGET_DIR=$(abspath $(SHARED_TARGET_DIR)) \
		rustup run nightly cargo build --release \
			--target $(RV64_TARGET_SPEC) \
			-Z build-std=core,alloc,std,compiler_builtins,panic_abort \
			-Z build-std-features=compiler-builtins-mem
	cp $(SHARED_TARGET_DIR)/riscv64im-lambda-vm-elf/release/$* $@

# Compile rust benches (64-bit)
$(BENCH_ARTIFACTS_DIR)/%.elf: $(BENCH_PROGRAMS_DIR)/%/Cargo.toml
	@mkdir -p $(BENCH_ARTIFACTS_DIR)
	cd $(BENCH_PROGRAMS_DIR)/$* && \
		CARGO_TARGET_DIR=$(abspath $(SHARED_TARGET_DIR)) \
		rustup run nightly cargo build --release \
			--target $(RV64_TARGET_SPEC) \
			-Z build-std=core,alloc,std,compiler_builtins,panic_abort \
			-Z build-std-features=compiler-builtins-mem
	cp $(SHARED_TARGET_DIR)/riscv64im-lambda-vm-elf/release/$* $@

clean-asm:
	-rm -rf $(ASM_ARTIFACTS_DIR)

clean-rust:
	-rm -rf $(RUST_ARTIFACTS_DIR)

clean-bench:
	-rm -rf $(BENCH_ARTIFACTS_DIR)

clean-shared:
	-rm -rf $(SHARED_TARGET_DIR)

clean: clean-asm clean-rust clean-bench clean-shared

test-executor: compile-programs test-no-compile

test-asm: compile-programs-asm test-asm-no-compile

test-asm-no-compile:
	cargo test -p executor --test asm

test-rust: compile-programs-rust prepare-test-data
	cargo test -p executor --test rust

test-rust-no-compile:
	cargo test -p executor --test rust

test-no-compile: prepare-test-data
	cargo test -p executor

test: compile-programs prepare-test-data
	cargo test

# === Quick test shortcuts ===

# Fast prover tests (skips ignored slow tests)
test-fast:
	cargo test -p prover -p stark -p executor -F stark/parallel

# Prover tests only (fast)
test-prover:
	cargo test -p prover -F stark/parallel

# Prover tests including slow ones
test-prover-all:
	cargo test -p prover -F stark/parallel -- --include-ignored

# Build all
build:
	cargo build --workspace

# Check (faster than build, no codegen)
check:
	cargo check --workspace

flamegraph-prover:
	cd crypto/stark && samply record cargo bench --bench profile_prover --features parallel
