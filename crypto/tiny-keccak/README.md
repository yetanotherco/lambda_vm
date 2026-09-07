# tiny-keccak 2.0.2 — vendored + Lambda VM patch

Vendored copy of upstream [`tiny-keccak`](https://github.com/debris/tiny-keccak)
2.0.2 (CC0-1.0, see `LICENSE`), with one Lambda VM patch:

- `src/keccakf.rs`: on `riscv64` targets, `keccakf` routes the Keccak-f[1600]
  permutation to the Lambda VM keccak chip via the `keccak_permute` ecall
  (`lambda-vm-syscalls`, target-gated dep) instead of running in software.
  Host builds keep the original software implementation.

Used via `[patch.crates-io]` in guest workspaces (e.g.
`executor/programs/rust/ethrex/Cargo.toml`) so that every guest keccak —
including ethrex's free-fn `keccak_hash` path (trie/RLP/tx hashing), which does
not go through the `Crypto` trait — is proven by the chip instead of executed
as guest instructions. Same pattern as `sp1-patches/tiny-keccak`.

Measured (20-tx blocks): −270k cycles on transfers, −399k on ERC20, −367k on
mixed; software `keccakf` drops to 0 in the guest flamegraph.

Upstream updates: re-vendor from the crates-io 2.0.2 sources and re-apply the
two hunks (`keccakf` dispatch in `src/keccakf.rs`, target-gated dep in
`Cargo.toml`).
