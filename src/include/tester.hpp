#pragma once

#include <iostream>
#include <assert.h>
#include <stdlib.h>
#include <random>
#include <iomanip> // for std::setw, std::setprecision

class SparseSgemvTester {
public:
    SparseSgemvTester(int m, int n);
    auto RunTest() -> void;

private:
    int m_, n_;
    float *A_host = nullptr;
    float *X_host = nullptr;
    float *A_host_compressed = nullptr;
    int A_host_compressed_size_ = 0;
    float *Y_cpu_host = nullptr;
    float *Y_gpu_host = nullptr;

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