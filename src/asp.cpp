#include "asp.hpp"

ASPMatrix::ASPMatrix(int M, int N, float *matrix) {

    for (int bn = 0; bn < N; bn += 32) {
        for (int bm = 0; bm < M; bm += 32) {
            for (int i = 0; i < 32; i++) {
                for (int j = 0; j < 32; j++) {
                    values_.push_back(matrix[(bm + i) * N + bn + j]);
                }
            }
        }
    }
}

auto ASPMatrix::GetValues() -> float * {
    return values_.data();
}

auto ASPMatrix::ValuesSize() -> int {
    return values_.size();
}