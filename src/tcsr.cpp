#pragma once

#include "tcsr.hpp"

TCSRMatrix::TCSRMatrix(int M, int N, float *matrix)
    : m_(M), n_(N) {

    int num_of_u32 = M * N / 32;
    std::vector<uint32_t> bitmaps(num_of_u32, 0);
    
    int bit_index = 0;
    int value_index = 0;
    blk_idx_.push_back(value_index);

    // we will iter by block (column major)
    for (int block_x = 0; block_x < N; block_x += 32) {
        for (int block_y = 0; block_y < M; block_y += 32) {
            // for each block, arrange bitmap by thread ID order
            for (int i = 0; i < 32; i++) {
                for (int j = 0; j < 32; j++) {
                    int word = bit_index / 32;
                    int offset = bit_index % 32;
                    int ax = block_x + i;
                    int ay = block_y + j;
                    float val = matrix[ay * N + ax];
                    if (val != 0.0f) {
                        nz_values.push_back(val);
                        value_index += 1;
                        bitmaps[word] |= (1u << offset);
                    }
                    bit_index += 1;
                }
            }
            blk_idx_.push_back(value_index);
        }
    }
    bitmaps_ = bitmaps;
}

auto TCSRMatrix::GetBlkIdx() -> int * {
    return blk_idx_.data();
}

auto TCSRMatrix::GetBitmaps() -> uint32_t * {
    return bitmaps_.data();
}

auto TCSRMatrix::GetValues() -> float * {
    return nz_values.data();
}

auto TCSRMatrix::BlkIdxSize() -> int {
    return blk_idx_.size();
}

auto TCSRMatrix::BitmapsSize() -> int {
    return bitmaps_.size();
}

auto TCSRMatrix::ValuesSize() -> int {
    return nz_values.size();
}


