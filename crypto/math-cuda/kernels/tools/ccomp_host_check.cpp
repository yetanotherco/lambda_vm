// Host parity check of the compiled composition kernels against the interpreter.
//
// Builds BOTH `constraint_composition_kernel` (constraint_interp.cu) and every
// generated `ccomp_*` kernel (constraint_compiled.cu) as plain host C++ (one
// "thread" walks every row), feeds them the same random LDE columns, uniforms
// and accumulation inputs, and compares `H` limb for limb. No GPU needed: the
// device field arithmetic in goldilocks.cuh / ext3.cuh is ordinary C++.
//
//   1. dump the programs: LAMBDA_VM_CCOMP_DUMP=<dir> cargo test -p lambda-vm-prover --lib \
//        -- --ignored --exact tests::compiled_constraints::dump_compiled_programs
//   2. make the kernel table:
//        awk '/^P /{print "{\"" $3 "\", &" $3 "},"}' <dir>/programs.txt > <dir>/ccomp_table.inc
//   3. build and run (any C++17 compiler with unsigned __int128):
//        c++ -std=c++17 -O1 -I<dir> -Icrypto/math-cuda/kernels \
//          crypto/math-cuda/kernels/tools/ccomp_host_check.cpp -o <dir>/check
//        <dir>/check <dir>/programs.txt
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#define __device__
#define __global__
#define __host__
#define __forceinline__ inline
#define __restrict__ __restrict
#define __launch_bounds__(...)
struct HostDim3 {
    unsigned x, y, z;
};
static HostDim3 blockIdx{0, 0, 0}, threadIdx{0, 0, 0}, blockDim{1, 1, 1}, gridDim{1, 1, 1};
static inline unsigned long long __umul64hi(unsigned long long a, unsigned long long b) {
    return (unsigned long long)(((unsigned __int128)a * b) >> 64);
}

#include "constraint_interp.cu"
#include "constraint_compiled.cu"

typedef void (*Kernel)(Fe3 *, const uint64_t *, uint64_t, const uint64_t *, const Fe3 *,
                       const uint64_t *, uint64_t, const Fe3 *, const Fe3 *, const Fe3 *,
                       const uint64_t *, uint64_t, const uint64_t *, uint64_t, uint64_t, uint64_t,
                       const Fe3 *, const uint64_t *, uint64_t, uint64_t, const uint64_t *,
                       const uint64_t *, const Fe3 *, const Fe3 *, const uint64_t *, uint64_t *,
                       uint64_t *);
struct Entry {
    const char *name;
    Kernel fn;
};
static const Entry TABLE[] = {
#include "ccomp_table.inc"
};

static uint64_t rng_state;
static uint64_t next_u64() {
    rng_state += 0x9E3779B97F4A7C15ull;
    uint64_t z = rng_state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

struct Program {
    std::string key, name, labels;
    std::vector<uint64_t> nodes;  // 2 per node, as `pack_nodes`
    std::vector<uint64_t> roots, base_consts, ext_consts;
    uint64_t num_nodes = 0, num_base_slots = 0, num_ext_slots = 0;
};

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: check <programs.txt>\n");
        return 2;
    }
    std::ifstream in(argv[1]);
    std::vector<Program> progs;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        std::string tag;
        ss >> tag;
        if (tag == "P") {
            Program p;
            uint64_t roots, nb, ne;
            ss >> p.key >> p.name >> p.num_nodes >> roots >> p.num_base_slots >> p.num_ext_slots >> nb >>
                ne >> p.labels;
            progs.push_back(p);
        } else if (tag == "N") {
            uint64_t op, a, b, res;
            ss >> op >> a >> b >> res;
            progs.back().nodes.push_back(op | (a << 32));
            progs.back().nodes.push_back(b | (res << 32));
        } else if (tag == "R" || tag == "C" || tag == "X") {
            std::vector<uint64_t> &v = tag == "R"   ? progs.back().roots
                                       : tag == "C" ? progs.back().base_consts
                                                    : progs.back().ext_consts;
            uint64_t x;
            while (ss >> x) v.push_back(x);
        }
    }
    const uint64_t N = 64, NEXT_STEP = 2, Z_LEN = 4;
    int failures = 0, checked = 0;
    for (const Program &p : progs) {
        Kernel fn = nullptr;
        for (const Entry &e : TABLE)
            if (p.name == e.name) fn = e.fn;
        if (!fn) {
            std::printf("MISSING %s (%s)\n", p.name.c_str(), p.labels.c_str());
            failures++;
            continue;
        }
        // The footprint the program reads.
        uint64_t main_cols = 1, aux_cols = 1, rap_len = 1, alpha_len = 1;
        for (uint64_t i = 0; i < p.num_nodes; i++) {
            uint32_t op = (uint32_t)p.nodes[2 * i], a = (uint32_t)(p.nodes[2 * i] >> 32),
                     b = (uint32_t)p.nodes[2 * i + 1];
            if (op == OP_VAR) {
                uint64_t col = a & 0xFFFFu;
                if ((b >> 16) & 1u) main_cols = std::max(main_cols, col + 1);
                else aux_cols = std::max(aux_cols, col + 1);
                continue;
            }
            if (op == OP_RAP_CHALLENGE) rap_len = std::max<uint64_t>(rap_len, a + 1);
            if (op == OP_ALPHA_POW) alpha_len = std::max<uint64_t>(alpha_len, a + 1);
            for (uint32_t enc : {a, b}) {
                uint32_t kind = enc >> OPK_SHIFT, pay = enc & OPK_PAYLOAD_MASK;
                if (op < OP_ADD) break;
                if (kind == OPK_RAP) rap_len = std::max<uint64_t>(rap_len, pay + 1);
                if (kind == OPK_ALPHA) alpha_len = std::max<uint64_t>(alpha_len, pay + 1);
            }
        }
        for (uint64_t seed : {1ull, 0xDEADBEEFull}) {
            rng_state = seed ^ std::hash<std::string>{}(p.key);
            auto rnd = [](size_t n) {
                std::vector<uint64_t> v(n);
                for (auto &x : v) x = next_u64();
                return v;
            };
            std::vector<uint64_t> main = rnd(main_cols * N), aux = rnd(aux_cols * 3 * N),
                                  rap = rnd(rap_len * 3), alpha = rnd(alpha_len * 3), off = rnd(3),
                                  beta = rnd(p.roots.size() * 3 + 3), z_inv = rnd(Z_LEN);
            uint64_t nbd = 2;
            std::vector<uint64_t> b_col = {0, aux_cols - 1}, b_is_aux = {0, 1}, b_value = rnd(6),
                                  b_beta = rnd(6), b_z_inv = rnd(nbd * N);
            std::vector<uint64_t> h_interp(N * 3), h_comp(N * 3);
            std::vector<uint64_t> vb(p.num_base_slots + 1), ve(3 * p.num_ext_slots + 3);
            const Fe3 *ext_consts = (const Fe3 *)(p.ext_consts.empty() ? nullptr : p.ext_consts.data());
            constraint_composition_kernel(
                (Fe3 *)h_interp.data(), p.nodes.data(), p.num_nodes, p.base_consts.data(), ext_consts,
                p.roots.data(), p.roots.size(), (const Fe3 *)rap.data(), (const Fe3 *)alpha.data(),
                (const Fe3 *)off.data(), main.data(), N, aux.data(), N, NEXT_STEP, N,
                (const Fe3 *)beta.data(), z_inv.data(), Z_LEN, nbd, b_col.data(), b_is_aux.data(),
                (const Fe3 *)b_value.data(), (const Fe3 *)b_beta.data(), b_z_inv.data(), vb.data(),
                ve.data());
            fn((Fe3 *)h_comp.data(), p.nodes.data(), p.num_nodes, p.base_consts.data(), ext_consts,
               p.roots.data(), p.roots.size(), (const Fe3 *)rap.data(), (const Fe3 *)alpha.data(),
               (const Fe3 *)off.data(), main.data(), N, aux.data(), N, NEXT_STEP, N,
               (const Fe3 *)beta.data(), z_inv.data(), Z_LEN, nbd, b_col.data(), b_is_aux.data(),
               (const Fe3 *)b_value.data(), (const Fe3 *)b_beta.data(), b_z_inv.data(), nullptr,
               nullptr);
            checked++;
            for (uint64_t i = 0; i < N * 3; i++) {
                if (h_interp[i] != h_comp[i]) {
                    std::printf("MISMATCH %s (%s) seed %llx: row %llu limb %llu interp %016llx compiled %016llx\n",
                                p.name.c_str(), p.labels.c_str(), (unsigned long long)seed,
                                (unsigned long long)(i / 3), (unsigned long long)(i % 3),
                                (unsigned long long)h_interp[i], (unsigned long long)h_comp[i]);
                    failures++;
                    break;
                }
            }
        }
    }
    std::printf("HOST PARITY: %d program-seed runs, %zu programs, %d failures\n", checked, progs.size(), failures);
    return failures == 0 ? 0 : 1;
}
