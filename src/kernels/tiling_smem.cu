#include "kernel.hpp"

// version 1: tiling + share memory
__global__ void tiling_kernel(int M, int N, float *A, float *X, float *Y) {

    __shared__ float ldg_x_buffer[32];
    __shared__ float ldg_a_buffer[32][32];
    float acc = 0.0f;

    #pragma unroll
    for (int block_ks = 0; block_ks < M; block_ks += 32) {
        // load the x
        ldg_x_buffer[threadIdx.x] = X[block_ks + threadIdx.x];

        // load the A
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            int ay = block_ks + i;
            int ax = blockIdx.x * blockDim.x + threadIdx.x;
            ldg_a_buffer[i][threadIdx.x] = A[ay * N + ax];
        }

        __syncthreads();

        // compute
        for (int i = 0; i < 32; i++) {
            acc += ldg_x_buffer[i] * ldg_a_buffer[i][threadIdx.x];
        }
    }

    Y[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

void tiling_gemv_gpu(int M, int N, float *A_host, float *X_host, float *Y_host) {
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
    TIME_KERNEL((tiling_kernel<<<grid, block>>>(M, N, A_device, X_device, Y_device)));
    
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(Y_host, Y_device, size_Y, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(A_device));
    CUDA_CHECK(cudaFree(X_device));
    CUDA_CHECK(cudaFree(Y_device));
    return;
}