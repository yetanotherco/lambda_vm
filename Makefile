.PHONY: deps deps-linux deps-macos compile-programs-asm compile-programs-rust compile-bench \
compile-programs compile-recursion-elfs clean-asm clean-rust clean-bench clean-shared \
clean-recursion-elfs clean test test-asm \
test-rust test-ethrex test-ethrex-offline test-executor test-syscalls test-flamegraph flamegraph-prover test-profile-recursion test-profile-recursion-single test-profile-recursion-multi \
test-profile-recursion-block recursion-profile-block-input \
test-fast test-prover test-prover-all test-prover-debug test-disk-spill test-math-cuda test-cuda-integration test-cuda-fallback \
test-prover-cuda test-prover-comprehensive-cuda \
bench-math-cuda bench-prover bench-prover-cuda build check clippy fmt lint regen-ethrex-fixtures \
update-ethrex-fixture-checksums check-ethrex-fixture-checksums ethrex-real-block-fixture \
ethrex-real-block-cache ethrex-real-block-converter-cache print-real-block-fixture \
print-real-block-fixture-url \
test-ethrex-real-block-converter regen-real-block-fixture

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
RECURSION_GUESTS := empty fibonacci
RECURSION_ARTIFACTS := $(addprefix $(RECURSION_ARTIFACTS_DIR)/, $(addsuffix .elf, $(RECURSION_GUESTS)))

# The recursion verifier itself (bench_vs/lambda/recursion) requires picking
# exactly one of its preset Cargo features at build time (fixes the inner
# ProofOptions — see main.rs). Each preset builds its own distinctly named
# [[bin]] (recursion-<preset>-bench) to its own artifact, via the
# define/foreach/eval below rather than the generic %.elf pattern rule.
# `required-features` is a subset match, so e.g. `--features "continuation min"`
# also satisfies plain `recursion-min-bench`'s `required-features = ["min"]`,
# racing a concurrent `make -j` build of `recursion-min.elf` for the same
# shared-target-dir path. `--bin $(2)` in build_guest_elf pins each invocation
# to its one target bin.
RECURSION_VERIFIER_PRESETS := min blowup2 blowup4 blowup8
# `continuation` feature: verify a multi-epoch ContinuationProof bundle instead
# of a monolithic VmProof. Only the presets the benchmarks actually measure.
RECURSION_CONT_PRESETS := min blowup2 blowup4
RECURSION_VERIFIER_ARTIFACTS := $(addprefix $(RECURSION_ARTIFACTS_DIR)/recursion-, $(addsuffix .elf, $(RECURSION_VERIFIER_PRESETS))) \
	$(addprefix $(RECURSION_ARTIFACTS_DIR)/recursion-cont-, $(addsuffix .elf, $(RECURSION_CONT_PRESETS)))

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

.PHONY: test test-syscalls test-ethrex-crypto prepare-sysroot

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

# NOTE: the recursion smoke tests read these prebuilt guest ELFs. The fast ones
# run on every `cargo test` (so `make test`, which depends on this target, needs
# them); the slow ones stay #[ignore]d (only `test-prover-all` runs them). We
# compile the guest ELFs on every build so the tests always have them ready.
compile-programs: compile-programs-asm compile-programs-rust compile-bench compile-recursion-elfs

compile-recursion-elfs: prepare-sysroot $(RECURSION_ARTIFACTS) $(RECURSION_VERIFIER_ARTIFACTS)

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
# identical across the rust, bench, and recursion guests. They differ in the
# crate directory ($(1), the full path — callers interpolate $* themselves, so
# a target's stem needn't match its crate dir name, e.g. the recursion-verifier
# presets below), the built binary's filename ($(2)), and optional extra cargo
# args ($(3), e.g. `--features min`). cargo owns the dep graph (see FORCE
# above), so the recipe always runs and lets cargo decide what to rebuild.
define build_guest_elf
cd $(1) && \
	CARGO_TARGET_DIR=$(abspath $(SHARED_TARGET_DIR)) \
	CFLAGS_riscv64im_lambda_vm_elf="$(SYSROOT_CFLAGS)" \
	rustup run nightly-2026-02-01 cargo build --release \
		--target $(RV64_TARGET_SPEC) \
		-Z build-std=core,alloc,std,compiler_builtins,panic_abort \
		-Z build-std-features=compiler-builtins-mem \
		-Z json-target-spec \
		--bin $(2) \
		$(3)
cp $(SHARED_TARGET_DIR)/riscv64im-lambda-vm-elf/release/$(2) $@
endef

# Compile rust (64-bit)
# Order-only `| prepare-sysroot` so a direct `make .../foo.elf` provisions the sysroot
# first (the aggregate compile-programs-rust/compile-bench targets already do, but a
# bare pattern-rule invocation like `make -B .../ethrex.elf` would otherwise skip it
# and fail to compile guest C dependencies). Order-only because prepare-sysroot is
# .PHONY — a normal prereq would force a rebuild every time; its recipe is idempotent.
$(RUST_ARTIFACTS_DIR)/%.elf: FORCE | prepare-sysroot $(RUST_ARTIFACTS_DIR)
	$(call build_guest_elf,$(RUST_PROGRAMS_DIR)/$*,$*)

# Compile rust benches (64-bit)
$(BENCH_ARTIFACTS_DIR)/%.elf: FORCE | prepare-sysroot $(BENCH_ARTIFACTS_DIR)
	$(call build_guest_elf,$(BENCH_PROGRAMS_DIR)/$*,$*)

# Recursion-suite guests (bench_vs/lambda/): the crate's binary is <name>-bench, so
# copy <name>-bench -> <name>.elf. std-inclusive build-std covers both the no_std
# inner guests and the std recursion verifier. Prover tests read these prebuilt
# artifacts like every other program (see prover/src/tests/recursion_smoke_test.rs).
$(RECURSION_ARTIFACTS_DIR)/%.elf: FORCE | prepare-sysroot $(RECURSION_ARTIFACTS_DIR)
	$(call build_guest_elf,$(RECURSION_GUESTS_DIR)/$*,$*-bench)

# One differently named [[bin]] per preset (recursion-<preset>-bench, gated on
# that preset's Cargo feature) -> a differently named artifact. define/foreach/
# eval rather than a pattern rule (stem "recursion-min" wouldn't match crate
# dir "recursion") or copy-paste (presets list is the single source of truth).
# $(1) is the preset; the recipe uses $$ so `$$(call build_guest_elf,...)`
# expands at recipe-run time (where $@ is defined).
define recursion_verifier_rule
$(RECURSION_ARTIFACTS_DIR)/recursion-$(1).elf: FORCE | prepare-sysroot $(RECURSION_ARTIFACTS_DIR)
	$$(call build_guest_elf,$$(RECURSION_GUESTS_DIR)/recursion,recursion-$(1)-bench,--features $(1))
endef
$(foreach preset,$(RECURSION_VERIFIER_PRESETS),$(eval $(call recursion_verifier_rule,$(preset))))

# Continuation variants: same crate, `continuation` feature on top of the preset
# feature -> recursion-cont-<preset>-bench -> recursion-cont-<preset>.elf.
define recursion_cont_verifier_rule
$(RECURSION_ARTIFACTS_DIR)/recursion-cont-$(1).elf: FORCE | prepare-sysroot $(RECURSION_ARTIFACTS_DIR)
	$$(call build_guest_elf,$$(RECURSION_GUESTS_DIR)/recursion,recursion-cont-$(1)-bench,--features "continuation $(1)")
endef
$(foreach preset,$(RECURSION_CONT_PRESETS),$(eval $(call recursion_cont_verifier_rule,$(preset))))

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

# ===== Real block: the benchmark workload =====
#
# A genuine Ethereum block, as opposed to the synthetic N-plain-transfer blocks
# from tooling/ethrex-fixtures. Two artifacts, both gitignored and both FETCHED
# rather than built:
#
#   the fixture   the rkyv ProgramInput the benchmarks prove (~1 MB)
#   the cache     the ethrex-replay JSON it was converted from (~2 MB), read only
#                 by `regen-real-block-fixture`. The converter's TESTS read a
#                 different, upstream-pinned cache — see below.
#
# Fetching a verified binary is the same contract as prepare-sysroot above, and it
# keeps the converter, the ~335-package ethrex host dependency tree and an
# ethrex-replay `rev` pin off the path of everyone who just wants to run a
# benchmark. It also decouples the block from what upstream happens to host:
# ethrex-replay publishes a cache for Hoodi and nothing else, so any mainnet block
# is unreachable by the convert-locally route (its cache takes ~4 minutes and ~700
# calls against an archive RPC to produce) and trivial by this one.
#
# The converter still exists and is still tested — see "Real-block converter"
# below. It is a regeneration tool for ethrex rev bumps, not a build step.
#
# ---- Repointing to a different block ----
# These SIX lines and nothing else. Every path below derives from them, the
# benchmark scripts and CI resolve the fixture through
# `make -s print-real-block-fixture`, and no workflow, script or env var anywhere
# names a block. Outside this file the repoint touches only REAL_BLOCK_FIXTURE in
# tooling/ethrex-tests, which points the usability screen at the block actually
# being proven. The converter's own pins do NOT move — see below.
# tooling/ethrex-block-converter/README.md carries the procedure and each candidate's
# measured cost.
ETHREX_REAL_BLOCK_NETWORK := mainnet
ETHREX_REAL_BLOCK := 25368371
ETHREX_REAL_BLOCK_FIXTURE_URL := https://github.com/yetanotherco/lambda_vm/releases/download/bench-fixtures-v1/ethrex_mainnet_25368371.bin
ETHREX_REAL_BLOCK_FIXTURE_SHA256 := 61eba49b6b254f4a05def5a47b08a21ae3eee56f0d37bcd7b3a24b0cc1e4a300
# The block's source cache, hosted in the same release. Only `regen-real-block-fixture`
# reads it — the converter's TESTS use a different, upstream-pinned cache (below).
ETHREX_REAL_BLOCK_CACHE_URL := https://github.com/yetanotherco/lambda_vm/releases/download/bench-fixtures-v1/cache_mainnet_25368371.json
ETHREX_REAL_BLOCK_CACHE_SHA256 := 7aa88a5f7c5755b7575870f95e6c5c26186947f5e9e0d52199148c74e2a2736b

ETHREX_REAL_BLOCK_ID := $(ETHREX_REAL_BLOCK_NETWORK)_$(ETHREX_REAL_BLOCK)
ETHREX_REAL_BLOCK_FIXTURE := executor/tests/ethrex_$(ETHREX_REAL_BLOCK_ID).bin
ETHREX_REAL_BLOCK_CACHE := tooling/ethrex-block-converter/caches/cache_$(ETHREX_REAL_BLOCK_ID).json

# $(call ensure_verified,url,sha256,dest,label,url-var-name)
#
# Guard-then-fetch, the same shape as prepare-sysroot above: the digest of whatever
# is already on disk is checked on EVERY invocation, so this catches the three ways
# a wrong artifact gets there — a stale copy from before a re-upload under the same
# block number, a corrupted file, and a hand-placed one — not just an absent file.
# That is why these are phony targets rather than file rules: a file rule does not
# run when its target exists, which is exactly the case that needs checking.
# (It replaces the ethrex-replay rev stamp, which guarded the same class of
# staleness for the one input that used to be pinned by rev.)
#
# On a miss it downloads to a temp file and verifies BEFORE moving into place, so an
# interrupted or corrupted transfer cannot leave a file that later reads as valid and
# silently changes what every benchmark measures. The "neither sha256sum nor shasum"
# case is a hard error, as in prepare-sysroot — a skipped check would defeat the
# point of fetching a binary at all.
define ensure_verified
	@set -e; \
	dest="$(3)"; want="$(2)"; \
	if command -v sha256sum >/dev/null 2>&1; then shacmd="sha256sum"; \
	elif command -v shasum >/dev/null 2>&1; then shacmd="shasum -a 256"; \
	else echo "$(4): missing sha256sum or shasum for checksum verification" >&2; exit 1; fi; \
	sha_of() { $$shacmd "$$1" | awk '{print $$1}'; }; \
	if [ -f "$$dest" ] && [ "$$(sha_of "$$dest")" = "$$want" ]; then \
		exit 0; \
	fi; \
	if [ -f "$$dest" ]; then \
		echo "$(4) $$dest does not match $$want - refetching."; \
	fi; \
	if [ -z "$(1)" ]; then \
		echo "$(4): $(5) is unset." >&2; \
		echo "  The $(ETHREX_REAL_BLOCK_ID) $(4) is fetched, not built. Set $(5) in the" >&2; \
		echo "  Makefile to wherever the artifact is hosted; see" >&2; \
		echo "  tooling/ethrex-block-converter/README.md for how to produce and host one." >&2; \
		exit 1; \
	fi; \
	mkdir -p $(dir $(3)); \
	tmp="$$dest.tmp"; \
	cleanup() { rm -f "$$tmp"; }; \
	trap 'cleanup' EXIT; \
	trap 'cleanup; exit 130' INT; \
	trap 'cleanup; exit 143' TERM; \
	echo "Downloading $(4) $(ETHREX_REAL_BLOCK_ID)..."; \
	curl -fL --proto '=https' --retry 3 --retry-delay 2 --retry-all-errors "$(1)" -o "$$tmp"; \
	echo "Verifying $(4) checksum..."; \
	if [ "$$(sha_of "$$tmp")" != "$$want" ]; then \
		echo "$(4): checksum mismatch for $(1)" >&2; \
		exit 1; \
	fi; \
	mv "$$tmp" "$$dest"; \
	trap - EXIT
endef

ethrex-real-block-fixture:
	$(call ensure_verified,$(ETHREX_REAL_BLOCK_FIXTURE_URL),$(ETHREX_REAL_BLOCK_FIXTURE_SHA256),$(ETHREX_REAL_BLOCK_FIXTURE),fixture,ETHREX_REAL_BLOCK_FIXTURE_URL)

ethrex-real-block-cache:
	$(call ensure_verified,$(ETHREX_REAL_BLOCK_CACHE_URL),$(ETHREX_REAL_BLOCK_CACHE_SHA256),$(ETHREX_REAL_BLOCK_CACHE),cache,ETHREX_REAL_BLOCK_CACHE_URL)

# Single source of truth for the benchmark tooling. scripts/bench_verify.sh,
# scripts/bench_abba.sh, scripts/perf_diff.sh and
# .github/workflows/benchmark-pr.yml read the fixture path from here instead of
# hardcoding it, so repointing the block above moves every benchmark at once.
# `-s` on the caller's side keeps the output clean.
print-real-block-fixture:
	@echo $(ETHREX_REAL_BLOCK_FIXTURE)

# Lets CI ask "is the fixture hosted yet?" without parsing the Makefile. Prints
# nothing while the URL is unset, which is the condition callers branch on.
print-real-block-fixture-url:
	@echo $(ETHREX_REAL_BLOCK_FIXTURE_URL)

# ===== Real-block converter (regeneration tool, off the build path) =====
#
# Only needed when the guest's ethrex `rev` moves and the fixture has to be
# rebuilt, or when validating a candidate block. Nothing in the benchmark or test
# path builds this crate.
#
# Its TEST input is pinned to Hoodi 1265656, independently of whichever block the
# benchmarks currently prove, and stays there across a repoint. What these tests
# exercise is the CONVERSION — cache JSON in, correctly-laid-out rkyv out — which
# any real block demonstrates equally well. Hoodi's is the one cache ethrex-replay
# publishes, so pinning there costs us no hosting, cannot drift, and leaves the
# benchmark block free to change without touching this crate.
#
# Pinned by immutable `rev`, as the guest pins ethrex itself: a branch ref would let
# the converter's reproducibility digest drift under a fixed input.
ETHREX_REPLAY_REV := 2693e0182a8734117151d8ea2891eda5afc60383
ETHREX_CONVERTER_TEST_BLOCK := hoodi_1265656
ETHREX_CONVERTER_CACHE := tooling/ethrex-block-converter/caches/cache_$(ETHREX_CONVERTER_TEST_BLOCK).json
# The cache filename is keyed on the block only, and its download rule has no other
# prerequisite, so make would treat an already-present cache as up to date across an
# `ETHREX_REPLAY_REV` bump and silently keep reading the old input. Depending on a
# rev-stamped marker makes a re-pin discard the stale cache; without it the mismatch
# only surfaces downstream as a `conversion_is_reproducible` digest failure, which
# reads as "regenerate the fixture" and points at the wrong thing.
ETHREX_REPLAY_REV_STAMP := tooling/ethrex-block-converter/caches/.replay-rev-$(ETHREX_REPLAY_REV)

$(ETHREX_REPLAY_REV_STAMP):
	mkdir -p $(dir $@)
	rm -f $(ETHREX_CONVERTER_CACHE) $(dir $@).replay-rev-*
	touch $@

$(ETHREX_CONVERTER_CACHE): $(ETHREX_REPLAY_REV_STAMP)
	mkdir -p $(dir $@)
	curl -fsSL --retry 3 --retry-delay 2 --retry-all-errors -o $@.tmp \
		https://raw.githubusercontent.com/lambdaclass/ethrex-replay/$(ETHREX_REPLAY_REV)/caches/cache_$(ETHREX_CONVERTER_TEST_BLOCK).json
	mv $@.tmp $@

ethrex-real-block-converter-cache: $(ETHREX_CONVERTER_CACHE)

# Converter correctness: host-side parity through the guest's own Crypto impl, the
# network-rejection guard, and the reproducibility digest. Runs on changes to the
# converter (see .github/workflows/ethrex-block-converter.yml), not on every PR.
test-ethrex-real-block-converter: $(ETHREX_CONVERTER_CACHE)
	cd tooling/ethrex-block-converter && cargo test --release

# Manual regeneration of the BENCHMARK fixture (not the converter's test block):
# fetches that block's own cache and re-converts it, overwriting the fixture in
# place so you can hash the result and upload it. That upload, plus SHA256/URL at
# the top, is how the fixture is actually replaced.
regen-real-block-fixture: ethrex-real-block-cache
	cd tooling/ethrex-block-converter && \
		cargo run --release -- ../../$(ETHREX_REAL_BLOCK_CACHE) ../../$(ETHREX_REAL_BLOCK_FIXTURE)

# ethrex host-reference tests live in the detached `tooling/ethrex-tests`
# workspace (ethrex pins rkyv's `unaligned` feature; isolated Cargo.lock).
# Needs the real-block fixture, so it needs the fixture URL to be set. This is a
# local convenience target: no workflow invokes it. The PR gate spells out the
# `-offline` variant below inline (pr_main.yaml), and ethrex-block-converter.yml's
# block-usability job fetches the fixture and runs
# `test_ethrex_real_block_native` on its own.
test-ethrex: compile-programs-rust ethrex-real-block-fixture
	cd tooling/ethrex-tests && cargo test --release -- --include-ignored --skip test_ethrex_real_block_vm

# Offline variant: no network, and what the PR gate runs. `--skip
# test_ethrex_real_block` is a substring match, so it drops both real-block tests —
# the `_vm` one and the `_native` one, which reads the fetched fixture and would
# otherwise fail on a clean checkout. The committed synthetic fixtures and
# `no_kzg_backend_linked` still run.
test-ethrex-offline: compile-programs-rust
	cd tooling/ethrex-tests && cargo test --release -- --include-ignored --skip test_ethrex_real_block

test-flamegraph:
	cargo test -p executor --test flamegraph

test-profile-recursion: test-profile-recursion-single test-profile-recursion-multi

test-profile-recursion-single: compile-recursion-elfs
	cargo test --package lambda-vm-prover --lib test_recursion_profile_1query -- --ignored --nocapture

test-profile-recursion-multi: compile-recursion-elfs
	cargo test --package lambda-vm-prover --lib test_recursion_profile_multiquery -- --ignored --nocapture

# Pre-proved continuation input for test_recursion_profile_blowup4_block: proving
# a real ethrex block is real prover work, not the verifier-guest cost the test
# profiles, so it's built ONCE here rather than re-proven on every test run.
# Epoch=2^21 matches scripts/bench_recursion_scaling.sh's default.
RECURSION_PROFILE_BLOCK_INPUT := $(RECURSION_ARTIFACTS_DIR)/recursion-cont-blowup4-block4.bin

recursion-profile-block-input: $(RECURSION_PROFILE_BLOCK_INPUT)

$(RECURSION_PROFILE_BLOCK_INPUT): $(RUST_ARTIFACTS_DIR)/ethrex.elf executor/tests/ethrex_bench_4.bin | $(RECURSION_ARTIFACTS_DIR)
	rm -f /tmp/recursion_input.bin /tmp/recursion_input.bin.expected
	RECURSION_DUMP_PRESET=blowup4 RECURSION_DUMP_EPOCH_LOG2=21 \
		RECURSION_DUMP_INNER_ELF=$(CURDIR)/$(RUST_ARTIFACTS_DIR)/ethrex.elf \
		RECURSION_DUMP_INNER_INPUT=$(CURDIR)/executor/tests/ethrex_bench_4.bin \
		cargo test --release -p lambda-vm-prover --lib test_dump_recursion_input -- --ignored --nocapture
	mv /tmp/recursion_input.bin $@
	mv /tmp/recursion_input.bin.expected $@.expected

# Real-block profile (ethrex, blowup=4/4 transfers), via the `continuation` guest.
test-profile-recursion-block: compile-recursion-elfs $(RECURSION_PROFILE_BLOCK_INPUT)
	cargo test --package lambda-vm-prover --lib --release test_recursion_profile_blowup4_block -- --ignored --nocapture

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

# The syscalls crate is excluded from the workspace (riscv-only bare-metal
# entrypoints/allocator that assemble only for the guest target — see the root
# Cargo.toml exclude), so the root `cargo test` never reaches its host
# differential tests (the keccak sponge vs sha3 reference). Run them explicitly
# in the crate dir; wired into `test` below and run as a dedicated step
# in CI's cli-test job (pr_main.yaml).
test-syscalls:
	cd syscalls && cargo test

# ethrex-crypto is a detached workspace (excluded from the root members), so a
# root `cargo test` never runs it. Run it explicitly, like test-syscalls.
test-ethrex-crypto:
	cd crypto/ethrex-crypto && cargo test

test: compile-programs test-syscalls test-ethrex-crypto
	cargo test

# === Quick test shortcuts ===

# Fast prover tests (skips ignored slow tests). Recursion smoke/PoC tests read
# prebuilt guest ELFs, so build them first.
test-fast: compile-recursion-elfs
	cargo test -p lambda-vm-prover -p stark -p executor -F stark/parallel

# Prover tests only
test-prover: compile-recursion-elfs
	cargo test -p lambda-vm-prover

# Prover tests including slow ones. The recursion smoke tests read prebuilt
# guest ELFs from executor/program_artifacts/recursion/ — the fast ones on every
# run, the slow ones (still #[ignore]d) only under --include-ignored — so build
# them first.
test-prover-all: compile-recursion-elfs
	cargo test -p lambda-vm-prover -- --include-ignored

# Prover tests with debug-checks (shows bus balance report). Also unfiltered, so
# it runs the non-ignored recursion tests that read prebuilt guest ELFs.
test-prover-debug: compile-recursion-elfs
	cargo test -p lambda-vm-prover --features debug-checks -- --nocapture

# Disk-spill tests (stark + prover). FORCE_DISK_SPILL is required by the prover tests.
test-disk-spill:
	cargo test --release -p stark --features disk-spill disk_spill
	FORCE_DISK_SPILL=1 cargo test --release -p lambda-vm-prover --features disk-spill -- disk_spill count_table_lengths

# math-cuda parity tests (requires NVIDIA GPU + nvcc)
test-math-cuda:
	cargo test -p math-cuda --release

# End-to-end cuda dispatch coverage (requires NVIDIA GPU + nvcc).
# Asserts the R1-R4 GPU dispatch counters fired on a real prove.
# --test-threads=1: these tests reset and assert on process-global GPU call
# counters, so they must run serially or one test's reset races another's read.
test-cuda-integration:
	cargo test -p lambda-vm-prover --release --features cuda \
	    --test cuda_path_integration -- --ignored --nocapture --test-threads=1

# GPU error-path coverage (requires NVIDIA GPU + nvcc).
# Forces cuda dispatch errors and asserts the CPU fallback still produces a verifying proof.
test-cuda-fallback:
	cargo test -p lambda-vm-prover --release --features test-cuda-faults \
	    --test cuda_fallback_tests -- --ignored --nocapture --test-threads=1

# The prover/stark/crypto/ecsm test suite with the GPU (cuda) path enabled (requires NVIDIA
# GPU + nvcc). The GPU CI counterpart of CPU CI's sharded prover tests. Single-threaded: the
# GPU serializes proves and the dispatch counters are process-global. cuda on prover cascades
# to stark; crypto/ecsm build without it (they have no GPU path).
# compile-recursion-elfs: this unfiltered run executes the non-ignored recursion
# smoke tests, which read prebuilt guest ELFs; scripts/gpu_test.sh otherwise never builds them.
test-prover-cuda: compile-recursion-elfs
	cargo test --release -p lambda-vm-prover -p stark -p crypto -p ecsm \
	    --features lambda-vm-prover/cuda -- --test-threads=1

# The comprehensive all-instructions prove (ignored by default) on the GPU path (requires
# NVIDIA GPU + nvcc). GPU counterpart of the all-instructions half of CPU CI's merge-queue-only
# comprehensive job (the CPU job also runs test_recursion_execute; recursion has no GPU leg yet).
test-prover-comprehensive-cuda:
	cargo test --release -p lambda-vm-prover --features cuda \
	    test_prove_elfs_all_instructions_64_full -- --ignored --test-threads=1 --nocapture

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
	# The cuda feature gates whole modules + cuda-only integration tests. build.rs emits empty
	# cubin stubs when nvcc is absent, so this checks on a GPU-less host (CI lint runner, dev laptop)
	# too — no GPU required. Catches cuda-gated breakage that the non-cuda passes above miss.
	cargo clippy --workspace --all-targets --features lambda-vm-prover/cuda -- -D warnings -A clippy::op_ref

flamegraph-prover:
	cd crypto/stark && samply record cargo bench --bench profile_prover --features parallel
