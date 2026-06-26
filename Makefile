.PHONY: deps deps-linux deps-macos compile-programs-asm compile-programs-rust compile-bench \
compile-programs compile-recursion-elfs clean-asm clean-rust clean-bench clean-shared \
clean-recursion-elfs clean test test-asm \
test-rust test-executor test-flamegraph flamegraph-prover test-profile-recursion test-profile-recursion-single test-profile-recursion-multi \
test-fast test-prover test-prover-all test-disk-spill test-math-cuda test-cuda-integration \
bench-math-cuda bench-prover bench-prover-cuda build check clippy fmt lint regen-ethrex-fixtures \
update-ethrex-fixture-checksums check-ethrex-fixture-checksums

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

ASM_PROGRAMS := $(wildcard $(ASM_PROGRAMS_DIR)/*.s)
ASM_ARTIFACTS := $(patsubst $(ASM_PROGRAMS_DIR)/%.s,$(ASM_ARTIFACTS_DIR)/%.elf,$(ASM_PROGRAMS))

RUST_PROGRAM_DIRS := $(dir $(wildcard $(RUST_PROGRAMS_DIR)/*/Cargo.toml))
RUST_PROGRAMS := $(notdir $(basename $(RUST_PROGRAM_DIRS:%/=%)))
RUST_ARTIFACTS := $(addprefix $(RUST_ARTIFACTS_DIR)/, $(addsuffix .elf, $(RUST_PROGRAMS)))

BENCH_PROGRAM_DIRS := $(dir $(wildcard $(BENCH_PROGRAMS_DIR)/*/Cargo.toml))
BENCH_PROGRAMS := $(notdir $(basename $(BENCH_PROGRAM_DIRS:%/=%)))
BENCH_ARTIFACTS := $(addprefix $(BENCH_ARTIFACTS_DIR)/, $(addsuffix .elf, $(BENCH_PROGRAMS)))

# Recursion smoke-test guests, in bench_vs/lambda/ (shared with bench_vs/run.sh)
# rather than executor/programs/. The recursion guest is the in-VM STARK verifier.
RECURSION_GUESTS_DIR=./bench_vs/lambda
RECURSION_ARTIFACTS_DIR=./executor/program_artifacts/recursion
RECURSION_GUESTS := empty fibonacci recursion
RECURSION_ARTIFACTS := $(addprefix $(RECURSION_ARTIFACTS_DIR)/, $(addsuffix .elf, $(RECURSION_GUESTS)))

# Override with: make ... SYSROOT_DIR=$HOME/.lambda-vm-sysroot
# to install the sysroot in a user-writable location and avoid sudo.
SYSROOT_DIR ?= /opt/lambda-vm-sysroot
SYSROOT_URL := https://lambda.alignedlayer.com/lambda-vm-sysroot-rv64im.tar.gz
SYSROOT_SHA256 := 420e394a096f3859235e3a8121a8d5a10f995ac48e636e8d700f17d50803a0e7
# CFLAGS for guest programs with C dependencies: overrides the hardcoded `/opt/lambda-vm-sysroot`
# in their .cargo/config.toml so cargo picks up our $(SYSROOT_DIR) instead.
# $(abspath ...) because the build rule cd's into the program dir before invoking cargo.
SYSROOT_CFLAGS := --target=riscv64 -march=rv64im -mabi=lp64 --sysroot=$(abspath $(SYSROOT_DIR))

CLANG ?= clang
ASM_CFLAGS ?= --target=riscv64 -march=rv64im -mabi=lp64
ASM_LDFLAGS ?= -fuse-ld=lld -nostdlib -Wl,-e,main

# Custom RV64IM target spec location
RV64_TARGET_SPEC=$(CURDIR)/executor/programs/riscv64im-lambda-vm-elf.json

.PHONY: test prepare-sysroot

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
		tmp_dir=""; \
		cleanup() { if [ -n "$$tmp_dir" ]; then rm -rf "$$tmp_dir"; fi; }; \
		trap 'cleanup' EXIT; \
		trap 'cleanup; exit 130' INT; \
		trap 'cleanup; exit 143' TERM; \
		tmp_dir="$$(mktemp -d /tmp/lambda-vm-sysroot.XXXXXX)"; \
		tarball="$$tmp_dir/lambda-vm-sysroot-rv64im.tar.gz"; \
		echo "Provisioning sysroot at $(SYSROOT_DIR) (downloading lambda-vm-sysroot-rv64im.tar.gz)..."; \
		curl -fL --proto '=https' "$(SYSROOT_URL)" -o "$$tarball"; \
		echo "Verifying sysroot checksum..."; \
		checksum_ok=false; \
		if command -v sha256sum >/dev/null 2>&1; then \
			printf '%s  %s\n' "$(SYSROOT_SHA256)" "$$tarball" | sha256sum -c - >/dev/null && checksum_ok=true; \
		elif command -v shasum >/dev/null 2>&1; then \
			actual="$$(shasum -a 256 "$$tarball" | awk '{print $$1}')"; \
			[ "$$actual" = "$(SYSROOT_SHA256)" ] && checksum_ok=true; \
		else \
			echo "prepare-sysroot: missing sha256sum or shasum for checksum verification" >&2; \
			exit 1; \
		fi; \
		if [ "$$checksum_ok" != true ]; then \
			echo "prepare-sysroot: checksum mismatch for $(SYSROOT_URL)" >&2; \
			exit 1; \
		fi; \
		echo "Extracting sysroot to $(SYSROOT_DIR)..."; \
		if mkdir -p "$(SYSROOT_DIR)" 2>/dev/null && [ -w "$(SYSROOT_DIR)" ]; then \
			rm -rf "$(SYSROOT_DIR)" && mkdir -p "$(SYSROOT_DIR)" \
				&& tar -xzf "$$tarball" -C "$(SYSROOT_DIR)" --strip-components=1 --no-same-owner \
				|| { rm -rf "$(SYSROOT_DIR)"; exit 1; }; \
		else \
			echo "$(SYSROOT_DIR) is not writable; using sudo."; \
			echo "Tip: re-run with SYSROOT_DIR=\$$HOME/.lambda-vm-sysroot to avoid sudo."; \
			sudo rm -rf "$(SYSROOT_DIR)" && sudo mkdir -p "$(SYSROOT_DIR)" \
				&& sudo tar -xzf "$$tarball" -C "$(SYSROOT_DIR)" --strip-components=1 --no-same-owner \
				|| { sudo rm -rf "$(SYSROOT_DIR)"; exit 1; }; \
		fi; \
	fi

compile-programs-asm: $(ASM_ARTIFACTS)

$(ASM_ARTIFACTS_DIR):
	mkdir -p $@

$(ASM_ARTIFACTS_DIR)/%.elf: $(ASM_PROGRAMS_DIR)/%.s | $(ASM_ARTIFACTS_DIR)
	$(CLANG) $(ASM_CFLAGS) $(ASM_LDFLAGS) $< -o $@

compile-programs-rust: prepare-sysroot $(RUST_ARTIFACTS)

compile-bench: prepare-sysroot $(BENCH_ARTIFACTS)

# NOTE: the recursion smoke tests are #[ignore]d (not run by `make test` /
# `test-executor`) because they're too slow for CI today; only `test-prover-all`
# runs them. We still compile their guest ELFs on every build so they keep
# compiling until the tests are fast enough to run in CI.
compile-programs: compile-programs-asm compile-programs-rust compile-bench compile-recursion-elfs

compile-recursion-elfs: prepare-sysroot $(RECURSION_ARTIFACTS)

$(RECURSION_ARTIFACTS_DIR):
	mkdir -p $@


$(RUST_ARTIFACTS_DIR):
	mkdir -p $@

$(BENCH_ARTIFACTS_DIR):
	mkdir -p $@

# The guest .elf rules depend on FORCE so their recipe always runs: cargo already
# tracks the full dependency graph, so we let it decide what to rebuild (a fast
# no-op when nothing changed) rather than re-encode that in Make prereqs.
.PHONY: FORCE
FORCE:

# The guest .elf rules all share one canned recipe: the cargo build invocation is
# identical across the rust, bench, and recursion guests. They differ only in the
# source directory ($(1)) and the built-binary name suffix ($(2): empty when the
# binary == crate name, `-bench` for the recursion suite, whose crates are named
# <name>-bench). cargo owns the dep graph (see FORCE above), so the recipe always
# runs and lets cargo decide what to actually rebuild.
define build_guest_elf
cd $(1)/$* && \
	CARGO_TARGET_DIR=$(abspath $(SHARED_TARGET_DIR)) \
	CFLAGS_riscv64im_lambda_vm_elf="$(SYSROOT_CFLAGS)" \
	rustup run nightly-2026-02-01 cargo build --release \
		--target $(RV64_TARGET_SPEC) \
		-Z build-std=core,alloc,std,compiler_builtins,panic_abort \
		-Z build-std-features=compiler-builtins-mem \
		-Z json-target-spec
cp $(SHARED_TARGET_DIR)/riscv64im-lambda-vm-elf/release/$*$(2) $@
endef

# Compile rust (64-bit)
# Order-only `| prepare-sysroot` so a direct `make .../foo.elf` provisions the sysroot
# first (the aggregate compile-programs-rust/compile-bench targets already do, but a
# bare pattern-rule invocation like `make -B .../ethrex.elf` would otherwise skip it
# and fail to compile guest C dependencies). Order-only because prepare-sysroot is
# .PHONY — a normal prereq would force a rebuild every time; its recipe is idempotent.
$(RUST_ARTIFACTS_DIR)/%.elf: FORCE | prepare-sysroot $(RUST_ARTIFACTS_DIR)
	$(call build_guest_elf,$(RUST_PROGRAMS_DIR),)

# Compile rust benches (64-bit)
$(BENCH_ARTIFACTS_DIR)/%.elf: FORCE | prepare-sysroot $(BENCH_ARTIFACTS_DIR)
	$(call build_guest_elf,$(BENCH_PROGRAMS_DIR),)

# Recursion-suite guests (bench_vs/lambda/): the crate's binary is <name>-bench, so
# copy <name>-bench -> <name>.elf. std-inclusive build-std covers both the no_std
# inner guests and the std recursion verifier. Prover tests read these prebuilt
# artifacts like every other program (see prover/src/tests/recursion_smoke_test.rs).
$(RECURSION_ARTIFACTS_DIR)/%.elf: FORCE | prepare-sysroot $(RECURSION_ARTIFACTS_DIR)
	$(call build_guest_elf,$(RECURSION_GUESTS_DIR),-bench)

clean-asm:
	-rm -rf $(ASM_ARTIFACTS_DIR)

clean-rust:
	-rm -rf $(RUST_ARTIFACTS_DIR)

clean-bench:
	-rm -rf $(BENCH_ARTIFACTS_DIR)

clean-shared:
	-rm -rf $(SHARED_TARGET_DIR)

clean-recursion-elfs:
	-rm -rf $(RECURSION_ARTIFACTS_DIR)

clean: clean-asm clean-rust clean-bench clean-shared clean-recursion-elfs

test-executor: compile-programs
	cargo test -p executor

test-asm: compile-programs-asm
	cargo test -p executor --test asm

test-rust: compile-programs-rust
	cargo test -p executor --test rust

test-flamegraph:
	cargo test -p executor --test flamegraph

test-profile-recursion: test-profile-recursion-single test-profile-recursion-multi

test-profile-recursion-single: compile-programs-rust
	cargo test --package lambda-vm-prover --lib test_recursion_pc_histogram_1query -- --ignored --nocapture

test-profile-recursion-multi: compile-programs-rust
	cargo test --package lambda-vm-prover --lib test_recursion_pc_histogram_multiquery -- --ignored --nocapture

# Regenerate the committed ethrex block fixtures (see tooling/ethrex-fixtures).
# Run after bumping the ethrex rev; README checksums are refreshed automatically.
regen-ethrex-fixtures:
	cd tooling/ethrex-fixtures && \
		cargo run --release -- 0  ../../executor/tests/ethrex_empty_block.bin && \
		cargo run --release -- 1  ../../executor/tests/ethrex_simple_tx.bin && \
		cargo run --release -- 10 ../../executor/tests/ethrex_10_transfers.bin
	$(MAKE) update-ethrex-fixture-checksums

update-ethrex-fixture-checksums:
	python3 tooling/ethrex-fixtures/update_readme_checksums.py

check-ethrex-fixture-checksums:
	python3 tooling/ethrex-fixtures/update_readme_checksums.py --check

test: compile-programs
	cargo test

# === Quick test shortcuts ===

# Fast prover tests (skips ignored slow tests)
test-fast:
	cargo test -p lambda-vm-prover -p stark -p executor -F stark/parallel

# Prover tests only
test-prover:
	cargo test -p lambda-vm-prover

# Prover tests including slow ones. The recursion smoke tests (#[ignore]d) read
# prebuilt guest ELFs from executor/program_artifacts/recursion/, so build them first.
test-prover-all: compile-recursion-elfs
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
