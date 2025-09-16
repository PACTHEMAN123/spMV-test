#include "kernel.hpp"

// version 0: naive
__global__ void naive_kernel(int M, int N, float *A, float *X, float *Y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 0.0f;
    for (int i = 0; i < M; i++) {
        acc += X[i] * A[i * N + idx];
    }
    Y[idx] = acc;
}

void naive_gemv_gpu(int M, int N, float *A_host, float *X_host, float *Y_host) {
    dim3 block(32, 1, 1);
    dim3 grid(N / 32);

    const size_t size_A = M * N * sizeof(float);
    const size_t size_X = M * sizeof(float);
    const size_t size_Y = N * sizeof(float);

    float *A_device, *X_device, *Y_device;
    CUDA_CHECK(cudaMalloc((void **)&A_device, size_A));
    CUDA_CHECK(cudaMalloc((void **)&X_device, size_X));
    CUDA_CHECK(cudaMalloc((void **)&Y_device, size_Y));

    CUDA_CHECK(cudaMemcpy(A_device, A_host, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(X_device, X_host, size_X, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaDeviceSynchronize());

    // call the kernel
    TIME_KERNEL((naive_kernel<<<grid, block>>>(M, N, A_device, X_device, Y_device)));
    
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(Y_host, Y_device, size_Y, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(A_device));
    CUDA_CHECK(cudaFree(X_device));
    CUDA_CHECK(cudaFree(Y_device));

    return;
}