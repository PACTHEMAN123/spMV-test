#pragma once
#include <cuda_runtime.h>
#include <cstdio>

// GPU kernel launcher
void spmv_gpu(int M, int N, float *A, float *X, float *Y);


// cuda check
#define CUDA_CHECK(call) \
{ \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}