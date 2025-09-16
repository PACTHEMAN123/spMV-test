#pragma once

#include "matrix_csr.hpp"

CSRMatrix::CSRMatrix(int m, int n, float *matrix)
    : m_(m), n_(n) {
    // convert the normal matrix into CSR format
    // CSR will be column-major
    int cur_row_pointer = 0;
    for (int i = 0; i < n; i++) {
        row_pointers.push_back(cur_row_pointer);
        int num_nz = 0;
        for (int j = 0; j < m; j++) {
            float value = matrix[j * n + i];
            if (value != 0.0f) {
                nz_values.push_back(value);
                col_indices.push_back(j);
                num_nz += 1;
            }
        }
        cur_row_pointer += num_nz;
    }
}

void CSRMatrix::PrintCSR() {
    for (auto val: nz_values)
        std::cout << val << " ";
    std::cout << std::endl;

    for (auto row_ptr: row_pointers)
        std::cout << row_ptr << " ";
    std::cout << std::endl;

    for (auto col_idx: col_indices)
        std::cout << col_idx << " ";
    std::cout << std::endl;
}

auto CSRMatrix::GetRowPtrs() -> int * {
    return row_pointers.data();
}

auto CSRMatrix::GetColIdxs() -> int * {
    return col_indices.data();
}

auto CSRMatrix::GetValues() -> float * {
    return nz_values.data();
}

auto CSRMatrix::RowPtrsSize() -> int {
    return row_pointers.size();
}

auto CSRMatrix::ColIdxsSize() -> int {
    return col_indices.size();
}

auto CSRMatrix::ValuesSize() -> int {
    return nz_values.size();
}


