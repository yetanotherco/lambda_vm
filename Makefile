.PHONY: deps deps-linux deps-macos prepare-test-data compile-programs-asm compile-programs-rust compile-bench \
compile-programs clean-asm clean-rust clean-bench clean-shared clean test test-asm test-no-compile \
test-asm-no-compile test-rust test-rust-no-compile test-executor flamegraph-prover \
test-fast test-prover test-prover-all test-disk-spill test-math-cuda test-cuda-integration bench-math-cuda bench-prover bench-prover-cuda build check clippy fmt lint

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

RUST_PROGRAM_DIRS := $(dir $(wildcard $(RUST_PROGRAMS_DIR)/*/Cargo.toml))
RUST_PROGRAMS := $(notdir $(basename $(RUST_PROGRAM_DIRS:%/=%)))
RUST_ARTIFACTS := $(addprefix $(RUST_ARTIFACTS_DIR)/, $(addsuffix .elf, $(RUST_PROGRAMS)))

BENCH_PROGRAM_DIRS := $(dir $(wildcard $(BENCH_PROGRAMS_DIR)/*/Cargo.toml))
BENCH_PROGRAMS := $(notdir $(basename $(BENCH_PROGRAM_DIRS:%/=%)))
BENCH_ARTIFACTS := $(addprefix $(BENCH_ARTIFACTS_DIR)/, $(addsuffix .elf, $(BENCH_PROGRAMS)))

ETHREX_FILE := executor/tests/ethrex_hoodi.bin
ETHREX_URL := https://lambda.alignedlayer.com/ethrex_hoodi.bin

# Override with: make ... SYSROOT_DIR=$HOME/.lambda-vm-sysroot
# to install the sysroot in a user-writable location and avoid sudo.
SYSROOT_DIR ?= /opt/lambda-vm-sysroot
# Fixed, global path: prepare-sysroot assumes a single writer at a time. The recipe
# `rm -f`s this before downloading, so a stale tarball can't be extracted — but two
# concurrent `make prepare-sysroot` on one host would race on it. The current CI runs
# no concurrent jobs sharing a SYSROOT_DIR; revisit (e.g. mktemp/flock) if that changes.
SYSROOT_TARBALL := /tmp/lambda-vm-sysroot-rv64im.tar.gz
SYSROOT_URL := https://lambda.alignedlayer.com/lambda-vm-sysroot-rv64im.tar.gz
# CFLAGS for ckzg / ethrex guest programs: overrides the hardcoded `/opt/lambda-vm-sysroot`
# in their .cargo/config.toml so cargo picks up our $(SYSROOT_DIR) instead.
# $(abspath ...) because the build rule cd's into the program dir before invoking cargo.
SYSROOT_CFLAGS := --target=riscv64 -march=rv64im -mabi=lp64 --sysroot=$(abspath $(SYSROOT_DIR))

CLANG ?= clang
ASM_CFLAGS ?= --target=riscv64 -march=rv64im -mabi=lp64
ASM_LDFLAGS ?= -fuse-ld=lld -nostdlib -Wl,-e,main

# Custom RV64IM target spec location
RV64_TARGET_SPEC=$(CURDIR)/executor/programs/riscv64im-lambda-vm-elf.json

.PHONY: test prepare-test-data prepare-sysroot

prepare-test-data:
	@set -e; \
	if [ ! -f "$(ETHREX_FILE)" ]; then \
		echo "Downloading ethrex_hoodi.bin..."; \
		curl -fL --proto '=https' "$(ETHREX_URL)" -o "$(ETHREX_FILE)" \
			|| { rm -f "$(ETHREX_FILE)"; exit 1; }; \
	else \
		echo "ethrex_hoodi.bin already exists"; \
	fi

# The guard checks for include/stdlib.h (not just the include/ dir) so that a PARTIAL
# sysroot — directories present but missing the C standard library headers — is detected
# as incomplete and re-provisioned, instead of being mistaken for a complete one. When it
# re-provisions, it first removes any existing $(SYSROOT_DIR) and re-extracts from scratch,
# so a partial/stale/corrupt sysroot self-heals without manual intervention on the runner.
# A basename allowlist guards the rm -rf: SYSROOT_DIR must end in lambda-vm-sysroot or
# .lambda-vm-sysroot, so an accidental override (e.g. SYSROOT_DIR=/opt) can't be wiped,
# especially via the sudo fallback. This is typo/misconfig prevention, NOT a security
# boundary — a caller that controls SYSROOT_DIR can still point it at any */lambda-vm-sysroot.
prepare-sysroot:
	@set -e; \
	if [ -f "$(SYSROOT_DIR)/include/stdlib.h" ] && [ -d "$(SYSROOT_DIR)/lib" ]; then \
		echo "Sysroot already exists at $(SYSROOT_DIR)"; \
	else \
		case "$$(basename "$(SYSROOT_DIR)")" in \
			lambda-vm-sysroot|.lambda-vm-sysroot) : ;; \
			*) echo "prepare-sysroot: refusing to (sudo) rm -rf SYSROOT_DIR=$(SYSROOT_DIR) - expected a path ending in lambda-vm-sysroot or .lambda-vm-sysroot"; exit 1 ;; \
		esac; \
		echo "Provisioning sysroot at $(SYSROOT_DIR) (downloading lambda-vm-sysroot-rv64im.tar.gz)..."; \
		rm -f "$(SYSROOT_TARBALL)"; \
		curl -fL --proto '=https' "$(SYSROOT_URL)" -o "$(SYSROOT_TARBALL)" \
			|| { rm -f "$(SYSROOT_TARBALL)"; exit 1; }; \
		echo "Extracting sysroot to $(SYSROOT_DIR)..."; \
		if mkdir -p "$(SYSROOT_DIR)" 2>/dev/null && [ -w "$(SYSROOT_DIR)" ]; then \
			rm -rf "$(SYSROOT_DIR)" && mkdir -p "$(SYSROOT_DIR)" \
				&& tar -xzf "$(SYSROOT_TARBALL)" -C "$(SYSROOT_DIR)" --strip-components=1 --no-same-owner \
				|| { rm -rf "$(SYSROOT_DIR)" "$(SYSROOT_TARBALL)"; exit 1; }; \
		else \
			echo "$(SYSROOT_DIR) is not writable; using sudo."; \
			echo "Tip: re-run with SYSROOT_DIR=\$$HOME/.lambda-vm-sysroot to avoid sudo."; \
			sudo rm -rf "$(SYSROOT_DIR)" && sudo mkdir -p "$(SYSROOT_DIR)" \
				&& sudo tar -xzf "$(SYSROOT_TARBALL)" -C "$(SYSROOT_DIR)" --strip-components=1 --no-same-owner \
				|| { sudo rm -rf "$(SYSROOT_DIR)"; rm -f "$(SYSROOT_TARBALL)"; exit 1; }; \
		fi; \
		rm -f "$(SYSROOT_TARBALL)"; \
	fi

compile-programs-asm:
	@mkdir -p $(ASM_ARTIFACTS_DIR)
	@set -e; for src in $(ASM_PROGRAMS); do \
		echo "$(CLANG) $(ASM_CFLAGS) $(ASM_LDFLAGS) $$src -o $(ASM_ARTIFACTS_DIR)/$$(basename $$src .s).elf"; \
		$(CLANG) $(ASM_CFLAGS) $(ASM_LDFLAGS) $$src -o $(ASM_ARTIFACTS_DIR)/$$(basename $$src .s).elf; \
	done

compile-programs-rust: prepare-sysroot $(RUST_ARTIFACTS)

compile-bench: prepare-sysroot $(BENCH_ARTIFACTS)

compile-programs: compile-programs-asm compile-programs-rust compile-bench


# Compile rust (64-bit)
# Order-only `| prepare-sysroot` so a direct `make .../foo.elf` provisions the sysroot
# first (the aggregate compile-programs-rust/compile-bench targets already do, but a
# bare pattern-rule invocation like `make -B .../ethrex.elf` would otherwise skip it
# and fail to compile C deps such as c-kzg). Order-only because prepare-sysroot is
# .PHONY — a normal prereq would force a rebuild every time; its recipe is idempotent.
$(RUST_ARTIFACTS_DIR)/%.elf: $(RUST_PROGRAMS_DIR)/%/Cargo.toml | prepare-sysroot
	@mkdir -p $(RUST_ARTIFACTS_DIR)
	cd $(RUST_PROGRAMS_DIR)/$* && \
		CARGO_TARGET_DIR=$(abspath $(SHARED_TARGET_DIR)) \
		CFLAGS_riscv64im_lambda_vm_elf="$(SYSROOT_CFLAGS)" \
		rustup run nightly-2026-02-01 cargo build --release \
			--target $(RV64_TARGET_SPEC) \
			-Z build-std=core,alloc,std,compiler_builtins,panic_abort \
			-Z build-std-features=compiler-builtins-mem \
			-Z json-target-spec
	cp $(SHARED_TARGET_DIR)/riscv64im-lambda-vm-elf/release/$* $@

# Compile rust benches (64-bit)
$(BENCH_ARTIFACTS_DIR)/%.elf: $(BENCH_PROGRAMS_DIR)/%/Cargo.toml | prepare-sysroot
	@mkdir -p $(BENCH_ARTIFACTS_DIR)
	cd $(BENCH_PROGRAMS_DIR)/$* && \
		CARGO_TARGET_DIR=$(abspath $(SHARED_TARGET_DIR)) \
		CFLAGS_riscv64im_lambda_vm_elf="$(SYSROOT_CFLAGS)" \
		rustup run nightly-2026-02-01 cargo build --release \
			--target $(RV64_TARGET_SPEC) \
			-Z build-std=core,alloc,std,compiler_builtins,panic_abort \
			-Z build-std-features=compiler-builtins-mem \
			-Z json-target-spec
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

test-flamegraph:
	cargo test -p executor --test flamegraph

test: compile-programs prepare-test-data
	cargo test

# === Quick test shortcuts ===

# Fast prover tests (skips ignored slow tests)
test-fast:
	cargo test -p lambda-vm-prover -p stark -p executor -F stark/parallel

# Prover tests only
test-prover:
	cargo test -p lambda-vm-prover

# Prover tests including slow ones
test-prover-all:
	cargo test -p lambda-vm-prover -- --include-ignored

# Prover tests with debug-checks (shows bus balance report)
test-prover-debug:
	cargo test -p lambda-vm-prover --features debug-checks -- --nocapture

# Disk-spill tests (stark + prover). FORCE_DISK_SPILL is required by the prover tests.
test-disk-spill:
	cargo test --release -p stark --features disk-spill disk_spill
	FORCE_DISK_SPILL=1 cargo test --release -p lambda-vm-prover --features disk-spill -- disk_spill count_table_lengths

# math-cuda parity tests (requires NVIDIA GPU + nvcc)
test-math-cuda:
	cargo test -p math-cuda --release

# End-to-end cuda dispatch coverage (requires NVIDIA GPU + nvcc).
# Asserts every R1/R2/R3 GPU counter fired on a real prove.
test-cuda-integration:
	cargo test -p lambda-vm-prover --release --features cuda \
	    --test cuda_path_integration -- --ignored --nocapture

# math-cuda quick microbench (median of 10 runs)
bench-math-cuda:
	cargo test -p math-cuda --release --test bench_quick -- --ignored --nocapture

# Single-prove wall-time bench (warm-up + profiled run of fib_iterative_1M).
bench-prover:
	cargo test -p lambda-vm-prover --release --test bench_single -- --ignored --nocapture

# Single-prove wall-time bench with the GPU LDE path enabled.
# Needs an NVIDIA GPU + CUDA toolkit/driver.
bench-prover-cuda:
	cargo test -p lambda-vm-prover --release --features cuda --test bench_single -- --ignored --nocapture

# Build all
build:
	cargo build --workspace

# Check (faster than build, no codegen)
check:
	cargo check --workspace

# === Linting ===
# op_ref: We pass big integers (U256/U384) and field elements by reference since operator
# impls delegate to &self internally, avoiding unnecessary 32-48 byte copies.

clippy:
	cargo clippy --workspace --all-targets -- -D warnings -A clippy::op_ref
	cargo clippy --workspace --all-targets --no-default-features --features lambda-vm-prover/debug-checks -- -D warnings -A clippy::op_ref
	cargo clippy --workspace --all-targets --features lambda-vm-prover/disk-spill -- -D warnings -A clippy::op_ref

fmt:
	cargo fmt --all

# Run clippy + fmt check (used by CI)
lint:
	cargo fmt --check --all
	cargo clippy --workspace --all-targets -- -D warnings -A clippy::op_ref
	cargo clippy --workspace --all-targets --no-default-features --features lambda-vm-prover/debug-checks -- -D warnings -A clippy::op_ref
	cargo clippy --workspace --all-targets --features lambda-vm-prover/disk-spill -- -D warnings -A clippy::op_ref

flamegraph-prover:
	cd crypto/stark && samply record cargo bench --bench profile_prover --features parallel
