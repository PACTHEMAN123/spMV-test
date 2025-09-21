#include "wsp.hpp"

WSPMatrix::WSPMatrix(int M, int N, float *matrix) {
    int num_of_u32 = M * N / 32;
    std::vector<uint32_t> bitmaps(num_of_u32, 0);
    std::vector<std::vector<float>> nz_values_rows;
    nz_max_n = N;
    nz_max_m = 0;

    int bit_index = 0;
    for (int i = 0; i < N; i++) {
        std::vector<float> nz_values_row;
        for (int j = 0; j < M; j++) {
            int word_idx = bit_index / 32;
            int offset = bit_index % 32;
            float val = matrix[j * N + i];
            if (val != 0.0f) {
                nz_values_row.push_back(val);
                bitmaps[word_idx] |= (1u << offset);
            }
            bit_index++;
        }
        int cur_size = nz_values_row.size();
        nz_max_m = std::max(nz_max_m, cur_size);
        nz_values_rows.push_back(nz_values_row);
    }

    bitmaps_ = bitmaps;

    // serialize the nz values
    std::vector<float> nz_values(nz_max_m * nz_max_n, 0);
    for (int i = 0; i < nz_max_n; i++) {
        auto nz_values_row = nz_values_rows[i];
        for (int j = 0; j < nz_values_row.size(); j++) {
            nz_values[i * nz_max_m + j] = nz_values_row[j];
        }
    }

    nz_values_ = nz_values;
}

auto WSPMatrix::GetBitmaps() -> uint32_t * {
    return bitmaps_.data();
}

auto WSPMatrix::GetValues() -> float * {
    return nz_values_.data();
}

auto WSPMatrix::BitmapsSize() -> int {
    return bitmaps_.size();
}

auto WSPMatrix::ValuesSize() -> int {
    return nz_values_.size();
}