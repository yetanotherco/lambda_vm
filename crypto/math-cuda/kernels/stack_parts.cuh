#pragma once

// Part covering e, or the last part preceding it. The table holds
// (offset, len, address) in ascending order, with disjoint intervals.
__device__ __forceinline__ uint32_t stack_part_index(const uint64_t *parts, uint32_t lo,
                                                     uint32_t hi, uint64_t e) {
    while (hi - lo > 1) {
        const uint32_t mid = lo + (hi - lo) / 2;
        if (parts[3 * mid] <= e) lo = mid;
        else hi = mid;
    }
    return lo;
}

__device__ __forceinline__ uint64_t stack_part_word(const uint64_t *parts, uint32_t p0,
                                                    uint32_t p1, uint64_t e) {
    if (p0 == p1) return 0;
    const uint32_t p = stack_part_index(parts, p0, p1, e);
    const uint64_t offset = parts[3 * p], len = parts[3 * p + 1];
    if (e < offset || e >= offset + len) return 0;
    return reinterpret_cast<const uint64_t *>(parts[3 * p + 2])[e - offset];
}
