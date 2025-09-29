#include "awsp.hpp"

AWSPMatrix::AWSPMatrix(int M, int N, float *matrix) {

    std::vector<uint32_t> bitmaps(M * N / 32, 0);
    std::vector<std::vector<float>> nz_val_bks;
    int bit_index = 0;
    nz_bk_max_ = 0;

    for (int bn = 0; bn < N; bn += 32) {
        for (int bm = 0; bm < M; bm += 32) {
            // collect values in one block
            std::vector<float> nz_val_bk;

            for (int i = 0; i < 32; i++) {
                for (int j = 0; j < 32; j++) {
                    int word_idx = bit_index / 32;
                    int offset = bit_index % 32;
                    float val = matrix[(bm + i) * N + (bn + j)];
                    if (val != 0.0f) {
                        nz_val_bk.push_back(val);
                        bitmaps[word_idx] |= (1u << offset);
                    }
                    bit_index++;
                }
            }

            int cur_bk_size = nz_val_bk.size();
            nz_bk_max_ = std::max(nz_bk_max_, cur_bk_size);
            nz_val_bks.push_back(nz_val_bk);
        }
    }

    bitmaps_ = bitmaps;

    // serialize the nz values
    // do the zero-padding block-wise
    int nz_values_size = M*N / (32*32) * nz_bk_max_;
    std::vector<float> nz_values(nz_values_size, 0);

    for (int i = 0; i < M*N / (32*32); i++) {
        auto nz_vals_bk = nz_val_bks[i];
        for (int j = 0; j < nz_vals_bk.size(); j++) {
            nz_values[i * nz_bk_max_ + j] = nz_vals_bk[j];
        }
    }

    nz_values_ = nz_values;
}

auto AWSPMatrix::GetBitmaps() -> uint32_t * {
    return bitmaps_.data();
}

auto AWSPMatrix::GetValues() -> float * {
    return nz_values_.data();
}

auto AWSPMatrix::BitmapsSize() -> int {
    return bitmaps_.size();
}

auto AWSPMatrix::ValuesSize() -> int {
    return nz_values_.size();
}