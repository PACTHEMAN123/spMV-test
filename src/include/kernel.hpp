#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>
#include <cublas_v2.h>

// GPU kernel launcher
void tiling_gemv_gpu(int M, int N, float *A_host, float *X_host, float *Y_host);
void cublas_gemv_gpu(int M, int N, float *A, float *X, float *Y);
void naive_gemv_gpu(int M, int N, float *A_host, float *X_host, float *Y_host);
void csr_naive_gemv_gpu(int M, int N, float *A_host, float *X_host, float *Y_host);
void csr_tiling_gemv_gpu(int M, int N, float *A_host, float *X_host, float *Y_host);
void wsp_gemv_gpu(int M, int N, float *A_host, float *X_host, float *Y_host, int version);
void asp_gemv_gpu(int M, int N, float *A_host, float *X_host, float *Y_host, int version);

// cuda check
#define CUDA_CHECK(call) \
{ \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// cuda kernel run time
#define TIME_KERNEL(kernel_call)                                             \
    {                                                                     \
        cudaEvent_t start, stop;                                             \
        CUDA_CHECK(cudaEventCreate(&start));                                 \
        CUDA_CHECK(cudaEventCreate(&stop));                                  \
                                                                             \
        CUDA_CHECK(cudaEventRecord(start));                                  \
        (kernel_call);                                                         \
        CUDA_CHECK(cudaEventRecord(stop));                                   \
        CUDA_CHECK(cudaEventSynchronize(stop));                              \
                                                                             \
        float ms = 0.0f;                                                     \
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));                  \
        std::cout << #kernel_call << " took " << ms << " ms" << std::endl;   \
                                                                             \
        cudaEventDestroy(start);                                             \
        cudaEventDestroy(stop);                                              \
    }