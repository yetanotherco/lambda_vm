# CLI
A basic cli to run and prove compiled riscv programs. As this is still a work in progress, it can currently only run the programs and not prove them.

## Usage

```bash
cargo run -p cli --release -- <PROGRAM.elf>
```

You can supply any compiled elf file, you can find some at `executor/program_artifacts`.

## Guest Program Flamegraphs

Generate flamegraphs showing where the guest RISC-V program spends its execution time (by instruction count).

### Generate Folded Stacks

```bash
cargo run -p cli --release -- <PROGRAM.elf> --flamegraph folded.txt
```

### Convert to SVG

Requires [inferno](https://github.com/jonhoo/inferno) or [flamegraph.pl](https://github.com/brendangregg/FlameGraph):

```bash
# Install inferno (one-time)
cargo install inferno

# Generate SVG
cat folded.txt | inferno-flamegraph > flamegraph.svg
```

### Example

```bash
# Generate flamegraph for quicksort benchmark
cargo run -p cli --release -- executor/program_artifacts/bench/quicksort.elf --flamegraph /tmp/quicksort.txt
cat /tmp/quicksort.txt | inferno-flamegraph --title "quicksort" > quicksort_flamegraph.svg
```

### Notes

- The flamegraph shows **instruction count** per function, not wall-clock time
- Function names are demangled from Rust symbols
- Inlined functions won't appear (they're merged into their caller)
- Syscalls using `ecall` are not tracked as separate function calls
