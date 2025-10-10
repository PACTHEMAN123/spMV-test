#include "awsp_ref.hpp"
#include <iostream>

AWSPRefMatrix::AWSPRefMatrix(int M, int N, float *matrix) {
    std::vector<uint32_t> bitmaps(M * N / 32, 0);
    std::vector<std::vector<float>> nz_val_warps;
    std::vector<int> max_nz_warps(4, 0);
    int bit_index = 0;
    
    for (int bn = 0; bn < N; bn += 32) {
        for (int warp_id = 0; warp_id < 4; warp_id += 1) {
            std::vector<float> nz_val_warp;
            for (int r = 0; r < M/4; r++) {
                for (int i = 0; i < 32; i++) {
                    int word_idx = bit_index / 32;
                    int offset = bit_index % 32;
                    float val = matrix[(M*warp_id/4 + r)*N + (bn+i)];
                    if (val != 0.0f) {
                        nz_val_warp.push_back(val);
                        bitmaps[word_idx] |= (1u << offset);
                    }
                    bit_index += 1;
                }
            }
            int cur_nz_val_warp = nz_val_warp.size();
            max_nz_warps[warp_id] = std::max(max_nz_warps[warp_id], cur_nz_val_warp);
            nz_val_warps.push_back(nz_val_warp);
        }
    }

    bitmaps_ = bitmaps;

    int sum_max_nz_warp = 0;
    std::vector<int> warp_nz_offset(4, 0);
    for (int i = 0; i < 4; i++) {
        sum_max_nz_warp += max_nz_warps[i];
        warp_nz_offset[i] = sum_max_nz_warp;
        // std::cout << "warp " << i << " max nz: " << warp_nz_offset[i] << std::endl;
    }
    warp_nz_offset_ = warp_nz_offset; // !!!

    // std::cout << "nz_warp shape: " << nz_val_warps.size() / 4 << std::endl;

    int nz_values_size = N / 32 * sum_max_nz_warp;
    std::vector<float> nz_values(nz_values_size, 0);
    int idx = 0;
    for (int i = 0; i < N/32; i++) {
        for (int warp_id = 0; warp_id < 4; warp_id += 1) {
            auto nz_val_warp = nz_val_warps[idx++];
            int offset = warp_id == 0 ? 0 : warp_nz_offset[warp_id-1];
            for (int j = 0; j < nz_val_warp.size(); j++) {
                nz_values[i*sum_max_nz_warp + offset + j] = nz_val_warp[j];
            }
                
        }
    }
    nz_values_ = nz_values;
}

auto AWSPRefMatrix::GetBitmaps() -> uint32_t * {
    return bitmaps_.data();
}

auto AWSPRefMatrix::GetValues() -> float * {
    return nz_values_.data();
}

auto AWSPRefMatrix::BitmapsSize() -> int {
    return bitmaps_.size();
}

auto AWSPRefMatrix::ValuesSize() -> int {
    return nz_values_.size();
}

auto AWSPRefMatrix::GetWarpNZOffset() -> int * {
    return warp_nz_offset_.data();
}