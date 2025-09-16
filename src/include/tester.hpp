#pragma once

#include <iostream>
#include <assert.h>
#include <stdlib.h>
#include <random>
#include <vector>
#include <iomanip> // for std::setw, std::setprecision

class SparseSgemvTester {
public:
    // constructor
    SparseSgemvTester(int m, int n);
    // deconstructor
    ~SparseSgemvTester() {
        if (A_host != nullptr)
            free(A_host);
        if (X_host != nullptr)
            free(X_host);
        if (A_host_compressed != nullptr)
            free(A_host_compressed);
        if (Y_cpu_host != nullptr)
            free(Y_cpu_host);
        for (auto host: Y_gpu_hosts)
            free(host);
        if (bitmap != nullptr)
            free(bitmap);
    }
    auto RunTest() -> void;

private:
    int m_, n_;
    float *A_host = nullptr;
    float *X_host = nullptr;
    float *A_host_compressed = nullptr;
    int A_host_compressed_size_ = 0;
    float *Y_cpu_host = nullptr;
    std::vector<float *> Y_gpu_hosts;

    // bitmap
    uint32_t *bitmap = nullptr;
    auto GenerateBitMap() -> void;

    // helpers functions
    auto GetRandomMatrix() -> void;
    auto GetCompressedMatrix() -> void;
    auto GetRandomVector() -> void;
    auto Print() -> void;
    

    // reference for correctness
    auto SgemvCPU() -> void;
    auto PrintCPU() -> void;

    auto SgemvGPU() -> void;

    auto CompareY() -> void;
};