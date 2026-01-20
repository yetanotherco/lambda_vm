.PHONY: test test_all test_prover test_prover_all build check clean

# Fast tests for prover (skips ignored slow tests)
test:
	cargo test -p prover -p stark -p executor -F stark/parallel

# All tests including slow VM prover tests (~17 min)
test_all:
	cargo test -p prover -p stark -p executor -F stark/parallel -- --include-ignored

# Prover tests only (fast)
test_prover:
	cargo test -p prover -F stark/parallel

# Prover tests including slow ones
test_prover_all:
	cargo test -p prover -F stark/parallel -- --include-ignored

# Build all
build:
	cargo build --workspace

# Check (faster than build, no codegen)
check:
	cargo check --workspace

# Clean build artifacts
clean:
	cargo clean
