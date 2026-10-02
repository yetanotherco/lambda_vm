// Host parity check of the bounded-slot interpreter (constraint_si.cu) against
// the composition interpreter (constraint_interp.cu).
//
// Builds `constraint_composition_kernel` and every `si_*` kernel as plain host
// C++, feeds them the same random LDE columns (full-range limbs, so
// non-canonical ones too), uniforms and accumulation inputs, and compares `H`
// limb for limb. The interpreter runs as one thread; each `si_*` kernel runs
// as a grid of several blocks of several threads, each thread emulated in turn
// (the threads share nothing, so the order is immaterial), which exercises the
// slot layout and the multi-row tiles. No GPU needed.
//
//   1. dump the programs (every production program at 24, 48 and 128 words):
//        LAMBDA_VM_SI_DUMP=<dir> cargo test -p lambda-vm-prover --lib \
//          -- --ignored --exact tests::budgeted_constraints::dump_budgeted_programs
//   2. build and run (any C++17 compiler with unsigned __int128):
//        c++ -std=c++17 -O1 -Icrypto/math-cuda/kernels \
//          crypto/math-cuda/kernels/tools/si_host_check.cpp -o <dir>/si_check
//        <dir>/si_check <dir>/si_programs.txt
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

#define __device__
#define __global__
#define __host__
#define __shared__
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
struct uint4 {
    uint32_t x, y, z, w;
};
template <class T> static inline T __ldg(const T *p) { return *p; }
// The kernels' dynamic shared memory; each emulated block reuses it.
uint64_t si_smem_words[1 << 22];

#include "constraint_interp.cu"
#include "constraint_si.cu"

typedef void (*SiKernel)(Fe3 *, const uint4 *, uint32_t, const uint64_t *, const Fe3 *, const Fe3 *,
                         const uint64_t *, uint64_t, const uint64_t *, uint64_t, uint64_t, uint64_t,
                         const uint64_t *, uint64_t, uint64_t, const uint64_t *, const uint64_t *,
                         const Fe3 *, const Fe3 *, const uint64_t *);
struct Variant {
    const char *name;
    SiKernel fn;
    uint32_t max_words; // 0: shared memory (any width)
    uint32_t block, grid;
};
static const Variant VARIANTS[] = {
    {"si_smem_r1", si_smem_r1, 0, 8, 4},   {"si_smem_r2", si_smem_r2, 0, 8, 3},
    {"si_smem_r1/1x1", si_smem_r1, 0, 1, 1}, {"si_local_w32", si_local_w32, 32, 4, 5},
    {"si_local_w48", si_local_w48, 48, 4, 5}, {"si_local_w64", si_local_w64, 64, 4, 5},
    {"si_local_w128", si_local_w128, 128, 4, 5},
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
    std::string label;
    uint64_t num_nodes = 0, num_base_slots = 0, num_ext_slots = 0;
    uint64_t num_steps = 0, num_words = 0, num_rap = 0, num_alpha = 0, main_cols = 1, aux_cols = 1,
             budget = 0;
    std::vector<uint64_t> nodes; // 2 per node, as `pack_nodes`
    std::vector<uint64_t> roots, base_consts, ext_consts;
    std::vector<uint4> steps;
};

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: si_check <si_programs.txt>\n");
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
            uint64_t roots;
            ss >> p.label >> p.num_nodes >> roots >> p.num_base_slots >> p.num_ext_slots >> p.num_steps >>
                p.num_words >> p.num_rap >> p.num_alpha >> p.main_cols >> p.aux_cols >> p.budget;
            progs.push_back(p);
        } else if (tag == "N") {
            uint64_t op, a, b, res;
            ss >> op >> a >> b >> res;
            progs.back().nodes.push_back(op | (a << 32));
            progs.back().nodes.push_back(b | (res << 32));
        } else if (tag == "S") {
            uint64_t op, a, b, d;
            ss >> op >> a >> b >> d;
            progs.back().steps.push_back(uint4{(uint32_t)op, (uint32_t)a, (uint32_t)b, (uint32_t)d});
        } else if (tag == "R" || tag == "C" || tag == "X") {
            std::vector<uint64_t> &v = tag == "R"   ? progs.back().roots
                                       : tag == "C" ? progs.back().base_consts
                                                    : progs.back().ext_consts;
            uint64_t x;
            while (ss >> x) v.push_back(x);
        }
    }
    const uint64_t N = 128, NEXT_STEP = 4, Z_LEN = 4;
    int failures = 0, checked = 0;
    for (Program &p : progs) {
        if (p.steps.size() != p.num_steps) {
            std::printf("BAD DUMP %s: %zu steps, header %llu\n", p.label.c_str(), p.steps.size(),
                        (unsigned long long)p.num_steps);
            failures++;
            continue;
        }
        p.steps.push_back(uint4{0, 0, 0, 0}); // the prefetch pad
        uint64_t rap_len = std::max<uint64_t>(p.num_rap, 1), alpha_len = std::max<uint64_t>(p.num_alpha, 1);
        for (uint64_t seed : {1ull, 0xDEADBEEFull}) {
            rng_state = seed ^ std::hash<std::string>{}(p.label) ^ p.budget;
            auto rnd = [](size_t n) {
                std::vector<uint64_t> v(n);
                for (auto &x : v) x = next_u64();
                return v;
            };
            std::vector<uint64_t> main = rnd(p.main_cols * N), aux = rnd(p.aux_cols * 3 * N),
                                  rap = rnd(rap_len * 3), alpha = rnd(alpha_len * 3), off = rnd(3),
                                  beta = rnd(p.roots.size() * 3 + 3), z_inv = rnd(Z_LEN);
            uint64_t nbd = 2;
            std::vector<uint64_t> b_col = {0, p.aux_cols - 1}, b_is_aux = {0, 1}, b_value = rnd(6),
                                  b_beta = rnd(6), b_z_inv = rnd(nbd * N);
            // The reference: the interpreter, one thread.
            std::vector<uint64_t> h_ref(N * 3);
            std::vector<uint64_t> vb(p.num_base_slots + 1), ve(3 * p.num_ext_slots + 3);
            const Fe3 *ext_consts = (const Fe3 *)(p.ext_consts.empty() ? nullptr : p.ext_consts.data());
            blockIdx = {0, 0, 0};
            threadIdx = {0, 0, 0};
            blockDim = {1, 1, 1};
            gridDim = {1, 1, 1};
            constraint_composition_kernel(
                (Fe3 *)h_ref.data(), p.nodes.data(), p.num_nodes, p.base_consts.data(), ext_consts,
                p.roots.data(), p.roots.size(), (const Fe3 *)rap.data(), (const Fe3 *)alpha.data(),
                (const Fe3 *)off.data(), main.data(), N, aux.data(), N, NEXT_STEP, N,
                (const Fe3 *)beta.data(), z_inv.data(), Z_LEN, nbd, b_col.data(), b_is_aux.data(),
                (const Fe3 *)b_value.data(), (const Fe3 *)b_beta.data(), b_z_inv.data(), vb.data(),
                ve.data());
            // The ext uniform table: ext constants, RAP challenges, alpha powers, offset.
            std::vector<uint64_t> uni(p.ext_consts);
            uni.insert(uni.end(), rap.begin(), rap.begin() + 3 * p.num_rap);
            uni.insert(uni.end(), alpha.begin(), alpha.begin() + 3 * p.num_alpha);
            uni.insert(uni.end(), off.begin(), off.end());
            for (const Variant &v : VARIANTS) {
                if (v.max_words != 0 && p.num_words > v.max_words) continue;
                std::vector<uint64_t> h(N * 3, 0x5A5A5A5A5A5A5A5Aull);
                blockDim = {v.block, 1, 1};
                gridDim = {v.grid, 1, 1};
                for (unsigned bx = 0; bx < v.grid; bx++) {
                    for (unsigned tx = 0; tx < v.block; tx++) {
                        blockIdx = {bx, 0, 0};
                        threadIdx = {tx, 0, 0};
                        v.fn((Fe3 *)h.data(), p.steps.data(), (uint32_t)p.num_steps, p.base_consts.data(),
                             (const Fe3 *)uni.data(), (const Fe3 *)beta.data(), main.data(), N, aux.data(),
                             N, NEXT_STEP, N, z_inv.data(), Z_LEN, nbd, b_col.data(), b_is_aux.data(),
                             (const Fe3 *)b_value.data(), (const Fe3 *)b_beta.data(), b_z_inv.data());
                    }
                }
                checked++;
                for (uint64_t i = 0; i < N * 3; i++) {
                    if (h[i] != h_ref[i]) {
                        std::printf("MISMATCH %s at %llu words, %s, seed %llx: row %llu limb %llu "
                                    "interp %016llx si %016llx\n",
                                    p.label.c_str(), (unsigned long long)p.budget, v.name,
                                    (unsigned long long)seed, (unsigned long long)(i / 3),
                                    (unsigned long long)(i % 3), (unsigned long long)h_ref[i],
                                    (unsigned long long)h[i]);
                        failures++;
                        break;
                    }
                }
            }
        }
    }
    // The control: the same check must fail on a program with one
    // accumulation's coefficient index moved.
    int control = -1;
    for (Program &p : progs) {
        size_t k = 0;
        while (k < p.num_steps && p.steps[k].x != SI_ACC_E && p.steps[k].x != SI_ACC_B) k++;
        if (k == p.num_steps || p.roots.size() < 2) continue;
        rng_state = 7;
        auto rnd = [](size_t n) {
            std::vector<uint64_t> v(n);
            for (auto &x : v) x = next_u64();
            return v;
        };
        uint64_t rap_len = std::max<uint64_t>(p.num_rap, 1), alpha_len = std::max<uint64_t>(p.num_alpha, 1);
        std::vector<uint64_t> main = rnd(p.main_cols * N), aux = rnd(p.aux_cols * 3 * N),
                              rap = rnd(rap_len * 3), alpha = rnd(alpha_len * 3), off = rnd(3),
                              beta = rnd(p.roots.size() * 3 + 3), z_inv = rnd(Z_LEN), b_z_inv = rnd(2 * N),
                              b_value = rnd(6), b_beta = rnd(6);
        std::vector<uint64_t> b_col = {0, p.aux_cols - 1}, b_is_aux = {0, 1};
        std::vector<uint64_t> uni(p.ext_consts);
        uni.insert(uni.end(), rap.begin(), rap.begin() + 3 * p.num_rap);
        uni.insert(uni.end(), alpha.begin(), alpha.begin() + 3 * p.num_alpha);
        uni.insert(uni.end(), off.begin(), off.end());
        std::vector<uint64_t> h0(N * 3), h1(N * 3);
        blockDim = {1, 1, 1};
        gridDim = {1, 1, 1};
        blockIdx = {0, 0, 0};
        threadIdx = {0, 0, 0};
        for (int m = 0; m < 2; m++) {
            if (m == 1) p.steps[k].z = (p.steps[k].z + 1) % p.roots.size();
            si_smem_r1((Fe3 *)(m ? h1 : h0).data(), p.steps.data(), (uint32_t)p.num_steps,
                       p.base_consts.data(), (const Fe3 *)uni.data(), (const Fe3 *)beta.data(), main.data(), N,
                       aux.data(), N, NEXT_STEP, N, z_inv.data(), Z_LEN, 2, b_col.data(), b_is_aux.data(),
                       (const Fe3 *)b_value.data(), (const Fe3 *)b_beta.data(), b_z_inv.data());
        }
        control = h0 != h1 ? 1 : 0;
        std::printf("SI HOST CONTROL: %s with accumulation %zu's coefficient moved: H %s\n", p.label.c_str(), k,
                    control ? "changed (the check can fail)" : "UNCHANGED");
        break;
    }
    std::printf("SI HOST PARITY: %d variant-program-seed runs, %zu lowerings, %d failures\n", checked,
                progs.size(), failures);
    return failures == 0 && checked > 0 && control == 1 ? 0 : 1;
}
