# sail-riscv-lean

The official RISC-V spec ([riscv/sail-riscv](https://github.com/riscv/sail-riscv)), translated to Lean
by the Lean backend of the Sail compiler ([rems-project/sail](https://github.com/rems-project/sail)).
`formal-riscv` uses it to state what each RISC-V instruction does.

Nothing here is written by hand. `LeanRV64D/`, `LeanRV64D.lean`, `lakefile.toml` and `lean-toolchain`
are exactly what the Sail compiler outputs for the commits in [`PINS`](PINS), and `LICENCE` is
sail-riscv's (BSD-2-Clause), which covers the generated model. CI
(`.github/workflows/sail-riscv-lean.yml`) checks it: it regenerates the model from those commits and
fails if the result differs from what is committed.

## Update it

1. Change the commits in `PINS`.
2. Run `scripts/generate.sh` (needs opam, git, cmake, gmp, z3 and pkg-config; it installs Sail in its own
   opam switch). Generating takes about a minute once Sail is built.
3. Run `lake update` here and `lake update Lean_RV64D` in `formal-riscv`, then build `formal-riscv`.
   Lake may set `formal-riscv/lean-toolchain` to the model's version; put it back to the one in
   `formal/lean-toolchain`, which `formal-riscv` must match.
   This folder's `lean-toolchain` is the compiler's output and stays as it is: Lake builds the
   model with `formal-riscv`'s toolchain, since it is only used as a dependency.
4. Commit the result.
