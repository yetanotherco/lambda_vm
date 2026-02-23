# Benchmark Server Setup

## As admin user (or user with sudo permissions)

```bash
ssh admin@<server-ip>
```

Install dependencies:

```bash
sudo apt-get install -y time
sudo apt update
sudo apt install lsb-release wget software-properties-common gnupg
```

Install and set up LLVM 18 toolchain:

```bash
wget -qO- https://apt.llvm.org/llvm.sh | sudo bash -s -- 18
sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-18 100
sudo update-alternatives --install /usr/bin/lld lld /usr/bin/lld-18 100
clang --version # must be clang 18
```

## As app user

```bash
ssh app@<server-ip>
```

### 1. Generate an SSH key

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
```

Add the printed public key as a deploy key in the repository settings:
**GitHub repo > Settings > Deploy keys > Add deploy key**.

### 2. Clone the repository

```bash
git clone git@github.com:yetanotherco/lambda_vm.git
cd lambda_vm
```

### 3. Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
rustup toolchain install nightly-2026-02-01-x86_64-unknown-linux-gnu
```

### 4. Compile programs

```bash
make compile-programs-asm
```

The server is now ready to run benchmarks. See [BENCHMARKS.md](BENCHMARKS.md) for usage.
