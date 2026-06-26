# Agent Rules

- Before pushing code or conflict-resolution commits to a PR branch, run `make lint` from the repository root. `cargo fmt --check` is not a substitute because CI's `Lint` job runs the full `make lint` target.
- If `make lint` cannot complete, do not push unless the user explicitly accepts that risk, and report the incomplete lint result.
