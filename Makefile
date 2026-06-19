.PHONY: deps deps-linux deps-macos prepare-test-data compile-programs-asm compile-programs-rust compile-bench \
compile-programs clean-asm clean-rust clean-bench clean-shared clean test test-asm test-no-compile \
test-asm-no-compile test-rust test-rust-no-compile test-executor flamegraph-prover \
test-fast test-prover test-prover-all test-disk-spill test-math-cuda test-cuda-integration bench-math-cuda bench-prover bench-prover-cuda build check clippy fmt lint \
proofs proofs-charon proofs-aeneas proofs-hax proofs-check \
proofs-lean-build proofs-lean-build-hax proofs-lean-build-aeneas

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
SYSROOT_TARBALL := /tmp/lambda-vm-sysroot-rv64im.tar.gz
SYSROOT_URL := https://lambda.alignedlayer.com/lambda-vm-sysroot-rv64im.tar.gz
# CFLAGS for ckzg / ethrex guest programs: overrides the hardcoded `/opt/lambda-vm-sysroot`
# in their .cargo/config.toml so cargo picks up our $(SYSROOT_DIR) instead.
# $(abspath ...) because the build rule cd's into the program dir before invoking cargo.
SYSROOT_CFLAGS := --target=riscv64 -march=rv64im -mabi=lp64 --sysroot=$(abspath $(SYSROOT_DIR))

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
	@if [ -d "$(SYSROOT_DIR)/include" ] && [ -d "$(SYSROOT_DIR)/lib" ]; then \
		echo "Sysroot already exists at $(SYSROOT_DIR)"; \
	else \
		echo "Downloading lambda-vm-sysroot-rv64im.tar.gz..."; \
		curl -L "$(SYSROOT_URL)" -o "$(SYSROOT_TARBALL)"; \
		echo "Extracting sysroot to $(SYSROOT_DIR)..."; \
		if mkdir -p "$(SYSROOT_DIR)" 2>/dev/null && [ -w "$(SYSROOT_DIR)" ]; then \
			tar -xzf "$(SYSROOT_TARBALL)" -C "$(SYSROOT_DIR)" --strip-components=1 \
				|| { rm -rf "$(SYSROOT_DIR)" "$(SYSROOT_TARBALL)"; exit 1; }; \
		else \
			echo "$(SYSROOT_DIR) is not writable; using sudo."; \
			echo "Tip: re-run with SYSROOT_DIR=\$$HOME/.lambda-vm-sysroot to avoid sudo."; \
			sudo mkdir -p "$(SYSROOT_DIR)" \
				&& sudo tar -xzf "$(SYSROOT_TARBALL)" -C "$(SYSROOT_DIR)" --strip-components=1 \
				|| { sudo rm -rf "$(SYSROOT_DIR)"; rm -f "$(SYSROOT_TARBALL)"; exit 1; }; \
		fi; \
		rm "$(SYSROOT_TARBALL)"; \
	fi
# Note: the tarball rm above only runs on success — each error handler
# cleans up the tarball itself before `exit 1`.

compile-programs-asm:
	@mkdir -p $(ASM_ARTIFACTS_DIR)
	@set -e; for src in $(ASM_PROGRAMS); do \
		echo "clang --target=riscv64 -fuse-ld=lld -nostdlib -Wl,-e,main $$src -o $(ASM_ARTIFACTS_DIR)/$$(basename $$src .s).elf"; \
		clang --target=riscv64 -fuse-ld=lld -nostdlib -Wl,-e,main $$src -o $(ASM_ARTIFACTS_DIR)/$$(basename $$src .s).elf; \
	done

compile-programs-rust: prepare-sysroot $(RUST_ARTIFACTS)

compile-bench: prepare-sysroot $(BENCH_ARTIFACTS)

compile-programs: compile-programs-asm compile-programs-rust compile-bench


# Compile rust (64-bit)
$(RUST_ARTIFACTS_DIR)/%.elf: $(RUST_PROGRAMS_DIR)/%/Cargo.toml
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
$(BENCH_ARTIFACTS_DIR)/%.elf: $(BENCH_PROGRAMS_DIR)/%/Cargo.toml
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

# =============================================================================
# Formal proofs — generated Lean code from Rust sources
#
# Layout:
#   proofs/charon/   — LLBC files (charon output, one per crate)
#   proofs/aeneas/   — Lean extraction (aeneas output, split-files per crate)
#   proofs/hax/      — Lean extraction (hax output, per crate)
#
# Manually maintained (never overwritten by generation):
#   proofs/aeneas/lakefile.toml, proofs/aeneas/lean-toolchain
#   proofs/aeneas/{Math,Crypto,Executor}.lean   (static entry-point, 1 line each)
#   proofs/hax/lakefile.toml, proofs/hax/lean-toolchain  (one lean_lib per crate)
#
# Targets: proofs (all), proofs-charon, proofs-aeneas, proofs-hax, proofs-check (CI).
# =============================================================================

PROOFS_DIR    := proofs
CHARON_DIR    := $(PROOFS_DIR)/charon
AENEAS_DIR    := $(PROOFS_DIR)/aeneas
HAX_DIR       := $(PROOFS_DIR)/hax

CHARON_MATH     := $(CHARON_DIR)/math.llbc
CHARON_CRYPTO   := $(CHARON_DIR)/crypto.llbc
CHARON_EXECUTOR := $(CHARON_DIR)/executor.llbc

# Dedicated cargo target dir for charon. Charon only emits an .llbc when the crate
# actually (re)compiles; sharing ./target with `cargo build` means a cache hit skips
# the charon-driver and produces no output. An isolated target dir guarantees a clean
# compile, hence deterministic emission (mirrors how hax uses ./target/hax).
CHARON_TARGET_DIR := $(CURDIR)/target/charon

proofs: proofs-charon proofs-aeneas proofs-hax

# --- charon: Rust → LLBC ---

$(CHARON_DIR):
	mkdir -p $(CHARON_DIR)

# `math`'s default features include `parallel` (rayon), whose `par_chunks_mut`
# paths are not extractable (hax #420). Extract with `--no-default-features
# --features alloc` to drop only the parallelism — the crate ships a real
# sequential path for each, so this matches the hax invocation below and keeps
# all three backends extracting the same code. `crypto`'s default (`asm`,`std`,
# and `std` now pulls `alloc`) does NOT enable rayon (`crypto/parallel` is
# opt-in), so it extracts under defaults with no flag change; `executor` has no
# features to gate.
$(CHARON_MATH): $(CHARON_DIR)
	CARGO_TARGET_DIR=$(CHARON_TARGET_DIR) \
		charon cargo --preset=aeneas --dest-file=$(CURDIR)/$(CHARON_MATH) -- -p math --lib --no-default-features --features alloc

$(CHARON_CRYPTO): $(CHARON_DIR)
	CARGO_TARGET_DIR=$(CHARON_TARGET_DIR) \
		charon cargo --preset=aeneas --dest-file=$(CURDIR)/$(CHARON_CRYPTO) -- -p crypto --lib

$(CHARON_EXECUTOR): $(CHARON_DIR)
	CARGO_TARGET_DIR=$(CHARON_TARGET_DIR) \
		charon cargo --preset=aeneas --dest-file=$(CURDIR)/$(CHARON_EXECUTOR) -- -p executor --lib

proofs-charon: $(CHARON_MATH) $(CHARON_CRYPTO) $(CHARON_EXECUTOR)

# --- aeneas: LLBC → Lean (split-files, one subdir per crate) ---
# Entry-point files (Math.lean, Crypto.lean, Executor.lean) are static and not regenerated.
# The _Template files are generated fresh each time; rename to FunsExternal.lean /
# TypesExternal.lean once and fill them in — they will not be overwritten.

$(AENEAS_DIR)/Math/Funs.lean: $(CHARON_MATH)
	aeneas -backend lean $(CHARON_MATH) \
		-dest $(AENEAS_DIR) -subdir Math -split-files; \
	test -f $@
	# Patch aeneas codegen defects in the generated Math/Types.lean (duplicate
	# struct fields; mutually-recursive rand Rng/Fill -> opaque axioms) so the
	# Lean compiles. Idempotent; fails loudly if aeneas output changes shape.
	python3 $(CURDIR)/proofs/scripts/patch_aeneas_math_types.py $(AENEAS_DIR)/Math/Types.lean

$(AENEAS_DIR)/Crypto/Funs.lean: $(CHARON_CRYPTO)
	aeneas -backend lean $(CHARON_CRYPTO) \
		-dest $(AENEAS_DIR) -subdir Crypto -split-files; \
	test -f $@
	# Patch the duplicate-field codegen defects in Crypto/{Types,Funs}.lean
	# (same class as Math; see the script), then carve the self-contained,
	# zero-maintenance single-leaf Merkle `Proof::verify` subset
	# (Crypto/MerkleVerify.lean) by dependency closure — the full Crypto/Funs.lean
	# does not compile (out-of-scope upstream-blocked code), but the carved subset
	# does.
	python3 $(CURDIR)/proofs/scripts/patch_aeneas_crypto_types.py $(AENEAS_DIR)/Crypto
	python3 $(CURDIR)/proofs/scripts/carve_merkle_verify.py $(AENEAS_DIR)/Crypto

$(AENEAS_DIR)/Executor/Funs.lean: $(CHARON_EXECUTOR)
	aeneas -backend lean $(CHARON_EXECUTOR) \
		-dest $(AENEAS_DIR) -subdir Executor -split-files; \
	test -f $@

proofs-aeneas: $(AENEAS_DIR)/Math/Funs.lean $(AENEAS_DIR)/Crypto/Funs.lean $(AENEAS_DIR)/Executor/Funs.lean

# UPSTREAM-BLOCKED (#4): associated-type EQUALITY constraints.
# `IsUnsignedInteger: Shr<usize, Output = Self> + BitAnd<Output = Self> + ...`
# (crypto/math/src/unsigned_integer/traits.rs:6) puts `Output = Self` equalities
# in supertrait bounds; `IsField`/`IsPrimeField` depend on it transitively
# (field/traits.rs:232,254,269). hax (#1921) cannot emit these to Lean (the root
# of the `IsField.AssociatedTypes`/`IsUnsignedInteger.AssociatedTypes` "unknown
# identifier" failures in math.lean) and aeneas warns "Found an associated type
# in a trait declaration ... Aeneas cannot handle such types today". Aeneas has
# an internal `parameterize_trait_types` flag built for exactly this, but it is
# NOT exposed as a CLI option, so it can't be enabled without rebuilding aeneas
# from OCaml source. These traits are kept opaque/external; no production change.

# --- hax: Rust → Lean (annotated items only) ---
# hax writes one <crate>.lean per crate directly into HAX_DIR, all covered by the
# single hand-maintained $(HAX_DIR)/lakefile.toml (one lean_lib per crate).
# Note: only the math crate is currently targeted. math.lean emits but does NOT
# yet fully compile: remaining blockers are #4 assoc-type equality (HAX0001, see
# above), the sequential-FFT `&mut input[range]` slice borrows in bowers_fft.rs
# (HAX0003/HAX0010, hax #420), and missing `core_models.*` stdlib models in the
# hax Lean proof-lib. crypto/executor are not yet extractable.
#
# We extract with `--no-default-features --features alloc` (i.e. WITHOUT `parallel`).
# The rayon `par_chunks_mut` paths (FFT, batch-inverse) are gated behind the
# `parallel` feature and are NOT extractable (hax issue #420). Disabling the
# feature removes them from extraction; the crate ships a real, separately
# compiled sequential path for each (the same code used by no-parallel/wasm
# builds), so this excludes only parallelism, not the verified computation —
# e.g. `inplace_batch_inverse` extracts via its sequential form, which is the
# exact code the verifier runs at its (sub-threshold) slice sizes.

proofs-hax:
	mkdir -p $(HAX_DIR)
	cargo hax -C '-p' 'math' '--no-default-features' '--features' 'alloc' ';' into \
		--output-dir $(CURDIR)/$(HAX_DIR) lean; \
	test -f $(HAX_DIR)/math.lean
	# Inject `import CoreModelsSupplement` so math.lean sees our opaque stubs for
	# core_models.* stdlib models missing from the pinned Hax proof-lib.
	python3 $(CURDIR)/proofs/scripts/patch_hax_math.py $(HAX_DIR)/math.lean
	# crypto: extract ONLY the in-scope Merkle items via hax's native item
	# selection `-i` (glob + `+` transitive-dependency closure), rather than
	# gating the source with cfg(hax). `+...::verify` pulls in Proof::verify and
	# its dependency closure (the Proof struct + IsMerkleTreeBackend trait);
	# everything else (poseidon, transcript, batch, concrete backends) is left
	# out. See proofs/hax/HAX_INCLUDE below.
	cargo hax -C '-p' 'crypto' ';' into \
		-i '-** +crypto::merkle_tree::proof::**::verify' \
		--output-dir $(CURDIR)/$(HAX_DIR) lean; \
	test -f $(HAX_DIR)/crypto.lean
	python3 $(CURDIR)/proofs/scripts/patch_hax_math.py $(HAX_DIR)/crypto.lean

# --- CI check: regenerate into a temp dir and diff against committed output ---

# Regenerates everything into a throwaway dir and diffs against the committed Lean.
# _Template files are excluded: they are advisory starting points for the manually
# maintained *External.lean files, and aeneas may skip them on partial extraction.
# MerkleVerifyProofs.lean is excluded too: it is the hand-written proof file (not
# produced by extraction), so it must not count as "stale generated output".
proofs-check:
	$(eval TMPDIR := $(shell mktemp -d))
	@trap 'rm -rf $(TMPDIR)' EXIT; \
	mkdir -p $(TMPDIR)/charon $(TMPDIR)/aeneas $(TMPDIR)/hax; \
	CARGO_TARGET_DIR=$(TMPDIR)/target charon cargo --preset=aeneas --dest-file=$(TMPDIR)/charon/math.llbc -- -p math --lib --no-default-features --features alloc; \
	CARGO_TARGET_DIR=$(TMPDIR)/target charon cargo --preset=aeneas --dest-file=$(TMPDIR)/charon/crypto.llbc -- -p crypto --lib; \
	CARGO_TARGET_DIR=$(TMPDIR)/target charon cargo --preset=aeneas --dest-file=$(TMPDIR)/charon/executor.llbc -- -p executor --lib; \
	aeneas -backend lean $(TMPDIR)/charon/math.llbc -dest $(TMPDIR)/aeneas -subdir Math -split-files; true; \
	python3 $(CURDIR)/proofs/scripts/patch_aeneas_math_types.py $(TMPDIR)/aeneas/Math/Types.lean; \
	aeneas -backend lean $(TMPDIR)/charon/crypto.llbc -dest $(TMPDIR)/aeneas -subdir Crypto -split-files; true; \
	python3 $(CURDIR)/proofs/scripts/patch_aeneas_crypto_types.py $(TMPDIR)/aeneas/Crypto; \
	python3 $(CURDIR)/proofs/scripts/carve_merkle_verify.py $(TMPDIR)/aeneas/Crypto; \
	aeneas -backend lean $(TMPDIR)/charon/executor.llbc -dest $(TMPDIR)/aeneas -subdir Executor -split-files; true; \
	cargo hax -C '-p' 'math' '--no-default-features' '--features' 'alloc' ';' into --output-dir $(TMPDIR)/hax lean; \
	python3 $(CURDIR)/proofs/scripts/patch_hax_math.py $(TMPDIR)/hax/math.lean; \
	diff -r --brief --exclude="*_Template.lean" --exclude="*External.lean" \
		$(TMPDIR)/aeneas/Math     $(AENEAS_DIR)/Math     || { echo "FAIL: Math aeneas output is stale"; exit 1; }; \
	diff -r --brief --exclude="*_Template.lean" --exclude="*External.lean" --exclude="MerkleVerifyProofs.lean" \
		$(TMPDIR)/aeneas/Crypto   $(AENEAS_DIR)/Crypto   || { echo "FAIL: Crypto aeneas output is stale"; exit 1; }; \
	diff -r --brief --exclude="*_Template.lean" \
		$(TMPDIR)/aeneas/Executor $(AENEAS_DIR)/Executor || { echo "FAIL: Executor aeneas output is stale"; exit 1; }; \
	diff --brief \
		$(TMPDIR)/hax/math.lean   $(HAX_DIR)/math.lean   || { echo "FAIL: hax math output is stale"; exit 1; }; \
	echo "proofs-check: all generated files are up to date"
	# Compile the buildable extracted Lean (aeneas Math). Catches a regression
	# where the committed Lean is up-to-date but no longer well-formed. hax is
	# excluded here — it is #4-blocked (run `make proofs-lean-build-hax`).
	$(MAKE) proofs-lean-build

# --- lake build: compile the extracted Lean ---
# Extraction emitting a file is NOT proof the Lean is well-formed; a partial or
# garbage extraction can still write output. `lake build` is the real check that
# the generated definitions and the hand-maintained *External.lean stubs compile
# and that the opaque (#4 trait-bound / #6 aeneas) externals line up.
# Each subdir is an independent lake project (its own lakefile.toml +
# lean-toolchain + manifest, pulling Hax / aeneas backends from git).
#
# `proofs-lean-build` aggregates ONLY the targets that currently compile, so it
# is a green gate suitable for `proofs-check`/CI. The hax build is BLOCKED on the
# upstream assoc-type-equality limit (#4; see the comment above proofs-aeneas)
# and is kept as a separately-runnable target, NOT in the aggregate, so a known
# upstream gap doesn't turn the whole proofs flow red.
proofs-lean-build: proofs-lean-build-aeneas

# Runnable but expected to FAIL on #4 (IsField/IsUnsignedInteger assoc-type
# equality). All non-#4 errors are resolved (asm, core_models, iterators).
proofs-lean-build-hax:
	cd $(HAX_DIR) && lake build

proofs-lean-build-aeneas:
	# `Math`: full crate. `MerkleVerify`: the carved single-leaf Merkle
	# `Proof::verify` subset (the rest of `Crypto`, and `Executor`, are not
	# buildable — out-of-scope upstream-blocked code: assoc-type equality #4,
	# poseidon/transcript/batch codegen limits). `MerkleVerifyProofs`: the
	# hand-written panic-freedom, index-algebra and completeness proofs about the
	# carved defs — built here so a proof regression turns the green gate red.
	cd $(AENEAS_DIR) && lake build Math MerkleVerify MerkleVerifyProofs
