// Known-answer tests for the round kernels whose sums leave through a slot (`kernels/sumcheck.cu`:
// `slot_sums` and the `_slot` kernels; `LAMBDA_VM_ARGUE_ROUND_SLOT`), run on the host through
// `cuda_host_simt_shim.h` — every thread of a block a real thread, blocks one after another — so the
// last block's reduction, its barriers and the ticket run as written. No GPU, no nvcc.
//
// WHAT IT COVERS, for each slot kernel (`gkr_round_gruen_slot`, `gkr_round_gruen_real_slot`)
// against today's kernel followed by `sum_partials_ext3`, over grids of 1, 2, 3, 5, 37 and 2,048
// blocks and 2 and 3 rows:
//   1. the slot's sums RAW-identical to today's second launch's (the proof reduces them by
//      `ext3_from_raw`, but the kernel promises the same limbs);
//   2. every block's partials, and everything the kernel writes in place, raw-identical;
//   3. the ticket back at 0 after the launch, and a second launch over the same ticket finding its
//      last block again (a new `seq` in the slot);
//   4. `seq` written AFTER the system fence and the sums BEFORE it (the shim's fence takes a snapshot
//      of the slot).
//
// WHAT IT DOES NOT COVER, and the card tests still own: concurrent blocks racing for the ticket (the
// shim runs blocks in order), the card's memory model, mapped host memory over PCIe, nvcc.
//
// Build and run with `make test-round-slot-host-kat`.

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "cuda_host_simt_shim.h"

#define __shared__
#define __launch_bounds__(...)
extern "C" {
uint64_t shared[3 * 1024 + 8];
}

// The system fence, observed: the slot as it stands when the last block fences.
static uint64_t *kat_slot = nullptr;
static std::vector<uint64_t> kat_snapshot;
static int kat_fences = 0;
static void kat_fence_system() {
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    if (kat_slot) kat_snapshot.assign(kat_slot, kat_slot + 128);
    ++kat_fences;
}
#define __threadfence_system() kat_fence_system()

#include "sumcheck.cu"

namespace {

int failures = 0;
int checks = 0;

void check(bool ok, const std::string &what) {
    ++checks;
    if (!ok) {
        printf("FAIL: %s\n", what.c_str());
        ++failures;
    }
}

const uint64_t P = 0xFFFFFFFF00000001ull;
uint64_t state = 0x7A3C5E1F9B2D4680ull;
uint64_t next_u64() {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}

// A raw limb: every fourth one non-canonical, in [p, 2^64) — so additions carry, and an order of
// summation that differs shows in the limbs.
uint64_t raw_limb() {
    uint64_t v = next_u64() % P;
    if ((next_u64() & 3) == 0 && v < 0xFFFFFFFFull) v += P;
    return v;
}

std::vector<uint64_t> random_words(size_t n) {
    std::vector<uint64_t> v(n);
    for (auto &x : v) x = raw_limb();
    return v;
}

const uint64_t SENTINEL = 0x5E5E5E5E5E5E5E5Eull;

struct Slot {
    std::vector<uint64_t> words = std::vector<uint64_t>(128, 0);
    unsigned int ticket = 0;
    Slot() { words[96] = SENTINEL; }
};

// Today's second launch: `sum_partials_ext3` over `rows × blocks` partials.
std::vector<uint64_t> todays_sums(const std::vector<uint64_t> &partials, unsigned rows,
                                  unsigned blocks) {
    std::vector<uint64_t> out(rows * 3, 0);
    uint64_t b = blocks;
    simt_launch(rows, 256, [&] { sum_partials_ext3(partials.data(), b, out.data()); });
    return out;
}

// The checks every slot launch answers: the sums, the ticket, the fence's order. `run` launches
// the slot kernel once with the given `seq`.
template <typename Run>
void check_slot(const std::string &what, Slot &slot, const std::vector<uint64_t> &want,
                unsigned rows, Run run) {
    kat_slot = slot.words.data();
    kat_fences = 0;
    kat_snapshot.clear();
    run(uint64_t(41));
    bool sums = memcmp(slot.words.data(), want.data(), rows * 3 * 8) == 0;
    check(sums, what + ": the slot's sums = today's second launch's, raw");
    check(slot.words[96] == 41, what + ": seq in the slot");
    check(slot.ticket == 0, what + ": the ticket back at 0");
    check(kat_fences == 1, what + ": one system fence, by the last block");
    bool ordered = kat_snapshot.size() == 128 && kat_snapshot[96] == SENTINEL &&
                   memcmp(kat_snapshot.data(), want.data(), rows * 3 * 8) == 0;
    check(ordered, what + ": the sums out before the fence, seq after it");
    // A second launch over the same ticket finds its last block again.
    kat_fences = 0;
    run(uint64_t(42));
    check(slot.words[96] == 42 && kat_fences == 1 && slot.ticket == 0,
          what + ": a second launch over the same ticket writes its seq");
    kat_slot = nullptr;
}

unsigned launches = 0;

// ── the GKR layer's round ─────────────────────────────────────────────────────────────────────

void gkr_case(unsigned m, unsigned grid, unsigned block, bool fold, bool want_h0, bool real) {
    uint64_t half = 1ull << m, quarter = fold ? half / 4 : half / 2;
    uint64_t stored = real ? half / 2 + 1 : half;  // a kept layer's run, or a whole one
    std::vector<uint64_t> p = random_words(2 * half * 3), q = random_words(2 * half * 3);
    for (uint64_t i = stored; i < half; ++i) {
        // what a kept layer's padding reads as: the whole layer holds it as raw 0/1
        p[(half + i) * 3] = p[(half + i) * 3 + 1] = p[(half + i) * 3 + 2] = 0;
        q[(half + i) * 3] = 1;
        q[(half + i) * 3 + 1] = q[(half + i) * 3 + 2] = 0;
    }
    std::vector<uint64_t> e_hi = random_words(half * 3), e_lo = random_words(4 * 3);
    uint32_t low = 2;
    std::vector<uint64_t> s = random_words(3), l = random_words(3);
    unsigned rows = want_h0 ? 3 : 2;
    std::vector<uint64_t> pa = p, qa = q, pb = p, qb = q;
    std::vector<uint64_t> partials_a(3 * grid * 3, 0), partials_b(3 * grid * 3, 0);
    std::string what = "gkr m " + std::to_string(m) + " grid " + std::to_string(grid) + " block " +
                       std::to_string(block) + (fold ? " fold" : "") + (want_h0 ? " h0" : "") +
                       (real ? " kept" : "");
    simt_launch(grid, block, [&] {
        if (real)
            gkr_round_gruen_real(pa.data(), qa.data(), half, quarter, fold, s[0], s[1], s[2], l[0],
                                 l[1], l[2], e_hi.data(), e_lo.data(), low, want_h0, stored,
                                 partials_a.data());
        else
            gkr_round_gruen(pa.data(), qa.data(), half, quarter, fold, s[0], s[1], s[2], l[0], l[1],
                            l[2], e_hi.data(), e_lo.data(), low, want_h0, partials_a.data());
    });
    std::vector<uint64_t> want = todays_sums(partials_a, rows, grid);
    Slot slot;
    bool first = true;
    check_slot(what, slot, want, rows, [&](uint64_t seq) {
        // The second launch starts from the same layer, so its sums are the first's.
        std::vector<uint64_t> pc = p, qc = q;
        uint64_t *pp = first ? pb.data() : pc.data(), *qq = first ? qb.data() : qc.data();
        simt_launch(grid, block, [&] {
            if (real)
                gkr_round_gruen_real_slot(pp, qq, half, quarter, fold, s[0], s[1], s[2], l[0], l[1],
                                          l[2], e_hi.data(), e_lo.data(), low, want_h0, stored,
                                          partials_b.data(), &slot.ticket, slot.words.data(), seq);
            else
                gkr_round_gruen_slot(pp, qq, half, quarter, fold, s[0], s[1], s[2], l[0], l[1],
                                     l[2], e_hi.data(), e_lo.data(), low, want_h0,
                                     partials_b.data(), &slot.ticket, slot.words.data(), seq);
        });
        if (first) {
            check(partials_a == partials_b, what + ": every block's partials, raw");
            check(pa == pb && qa == qb, what + ": the halves folded in place, raw");
        }
        first = false;
        ++launches;
    });
}

}  // namespace

int main() {
    // The GKR round: today's and the kept layer's, fold and not, H(0) and not, over 1 / 2 / 37
    // blocks (a grid wider than the work leaves zero partials, which the reduction still sums) and
    // 2,048 blocks every one of which has work (every lane of the reduction, and lanes that wrap).
    for (unsigned grid : {1u, 2u, 37u}) {
        gkr_case(10, grid, 64, false, false, false);
        gkr_case(10, grid, 64, true, true, true);
    }
    gkr_case(9, 3, 256, true, false, false);
    gkr_case(12, 5, 128, false, true, true);
    gkr_case(17, 2048, 32, true, true, true);
    printf("round slot host KAT: %u slot launches, %d checks: the slot's sums = today's second launch's (raw); "
           "partials and in-place writes raw; ticket reset; seq after the fence\n",
           launches, checks);
    if (failures) {
        printf("round slot host KAT: %d FAILED\n", failures);
        return 1;
    }
    printf("round slot host KAT: all passed\n");
    return 0;
}
