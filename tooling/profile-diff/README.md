# profile-diff

Diff two Lambda VM guest profiles and report what moved. Use it to check whether
an optimization actually shifted cost, and where.

It consumes the **folded-stack** format emitted by the CLI flamegraph
(`cli execute --flamegraph <file>` or `cli prove --flamegraph <file>`), including
the syscall-aware `ecall:*` leaf frames. Each line is `frame;frame;frame <count>`.

## Usage

The script has no dependencies and a PEP-723 header, so `uv` runs it directly:

```sh
# regression table on stdout (biggest absolute movers first)
uv run tooling/profile-diff/profile_diff.py base.folded new.folded

# only frames that moved by >= 1000, and write differential folded stacks
uv run tooling/profile-diff/profile_diff.py base.folded new.folded \
    --min-delta 1000 --folded-out diff.folded

# render the diff as a flamegraph (requires inferno)
cat diff.folded | inferno-flamegraph > diff.svg
```

`base` is the baseline; `new` is the run you are comparing against it. A positive
delta means the frame got **more** expensive in `new`.

## Flags

| Flag | Description |
|---|---|
| `--min-delta <N>` | Hide frames whose `|delta|` is below `N` (default: 1). |
| `--top <N>` | Show only the `N` biggest movers. |
| `--folded-out <FILE>` | Also write differential folded stacks (counts are `|delta|`, leaf tagged `[+]`/`[-]`) for a diff flamegraph. |

## A typical loop

```sh
cli execute prog.elf --flamegraph base.folded     # before a change
# ... make the optimization ...
cli execute prog.elf --flamegraph new.folded      # after
uv run tooling/profile-diff/profile_diff.py base.folded new.folded
```
