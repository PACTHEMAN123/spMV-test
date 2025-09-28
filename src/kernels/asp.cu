#include "kernel.hpp"
#include "asp.hpp"

#include <iostream>

__global__ void asp_kernel_v0(
    int M, int N,
    float *A, float *X, float *Y
) {
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    float sum = 0.0f;
    float *A_ptr = A + blockIdx.x * (M * 32) + (32 * M/4) * warp_id + lane_id;
    float *X_ptr = X + M/4 * warp_id + lane_id;

    for (int bk = 0; bk < M / 4; bk += 32) {
        // load x
        float x = *X_ptr;

        for (int i = 0; i < 32; i++) {
            // shuffle current x
            float cur_x = __shfl_sync(0xffffffff, x, i);
            if (cur_x != 0.0f)
                sum += *A_ptr * cur_x;
            A_ptr += 32;
        }
        X_ptr += 32;
    }

    __shared__ float reduce_sum[3][32];
    if (warp_id != 0)
        reduce_sum[warp_id - 1][lane_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        for (int i = 0;i < 3; i++){
            sum += reduce_sum[i][lane_id];
        }
        Y[blockIdx.x * 32 + lane_id] = sum;
    }
}

// version1: using pipeline
__global__ void asp_kernel_v1(
    int M, int N,
    float *A, float *X, float *Y
) {
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    float sum = 0.0f;
    float *A_ptr = A + blockIdx.x * (M * 32) + (32 * M/4) * warp_id + lane_id;
    float *X_ptr = X + M/4 * warp_id + lane_id;
    float X_buf[3] = {0};
    float A_buf[32] = {0};

    // before pipeline 
    // head stage 0: load x0
    X_buf[0] = *X_ptr; X_ptr += 32;
    // head stage 1: load x1
    X_buf[1] = X_buf[0]; X_buf[0] = *X_ptr; X_ptr += 32;
    // head stage 2: load A
    for (int i = 0; i < 32; i++) {
        float cur_x = __shfl_sync(0xffffffff, X_buf[1], i);
        if (cur_x != 0.0f) A_buf[i] = *A_ptr;
        A_ptr += 32;
    }

    // main pipeline
    for (int bk = 0; bk < (M/4 - 32*2); bk += 32) {
        // load x
        X_buf[2] = X_buf[1]; X_buf[1] = X_buf[0]; X_buf[0] = *X_ptr;
        // load and compute A
        for (int i = 0; i < 32; i++) {
            // consume buffered A
            float x_calc = __shfl_sync(0xffffffff, X_buf[2], i);
            if (x_calc != 0.0f) sum += A_buf[i] * x_calc;
            // load new A
            float x_load = __shfl_sync(0xffffffff, X_buf[1], i);
            if (x_load != 0.0f) A_buf[i] = *A_ptr;
            A_ptr += 32;
        }
        X_ptr += 32;
    }

    // after pipeline 
    // tail stage 0 (no need to load new X)
    X_buf[2] = X_buf[1]; X_buf[1] = X_buf[0];
    for (int i = 0; i < 32; i++) {
        float x_calc = __shfl_sync(0xffffffff, X_buf[2], i);
        if (x_calc != 0.0f) sum += A_buf[i] * x_calc;
        float x_load = __shfl_sync(0xffffffff, X_buf[1], i);
        if (x_load != 0.0f) A_buf[i] = *A_ptr;
        A_ptr += 32;
    }
    // tail stage 1 (no need to load new A)
    X_buf[2] = X_buf[1];
    for (int i = 0; i < 32; i++) {
        float x_calc = __shfl_sync(0xffffffff, X_buf[2], i);
        if (x_calc != 0.0f) sum += A_buf[i] * x_calc;
    }

    __shared__ float reduce_sum[3][32];
    if (warp_id != 0)
        reduce_sum[warp_id - 1][lane_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        for (int i = 0;i < 3; i++){
            sum += reduce_sum[i][lane_id];
        }
        Y[blockIdx.x * 32 + lane_id] = sum;
    }
}

// version2: using double pipeline
__global__ void asp_kernel_v2(
    int M, int N,
    float *A, float *X, float *Y
) {
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    float sum = 0.0f;
    float *A_ptr = A + blockIdx.x * (M * 32) + (32 * M/4) * warp_id + lane_id;
    float *X_ptr = X + M/4 * warp_id + lane_id;
    float X_buf[2][3] = {0};
    float A_buf[2][32] = {0};

    // before pipeline 
    // head stage 0: load x0
    X_buf[0][0] = *X_ptr; X_ptr += 32;
    X_buf[1][0] = *X_ptr; X_ptr += 32;
    // head stage 1: load x1
    X_buf[0][1] = X_buf[0][0]; X_buf[0][0] = *X_ptr; X_ptr += 32;
    X_buf[1][1] = X_buf[1][0]; X_buf[1][0] = *X_ptr; X_ptr += 32;
    // head stage 2: load A
    for (int i = 0; i < 32; i++) {
        float x_load = __shfl_sync(0xffffffff, X_buf[0][1], i);
        if (x_load != 0.0f) A_buf[0][i] = *A_ptr;
        x_load = __shfl_sync(0xffffffff, X_buf[1][1], i);
        if (x_load != 0.0f) A_buf[1][i] = *(A_ptr + 32 * 32);
        A_ptr += 32;
    }
    A_ptr += 32 * 32;

    // main pipeline
    for (int bk = 0; bk < (M/4 - 64*2); bk += 64) {
        // load x
        X_buf[0][2] = X_buf[0][1]; X_buf[0][1] = X_buf[0][0]; X_buf[0][0] = *X_ptr; 
        X_ptr += 32;
        X_buf[1][2] = X_buf[1][1]; X_buf[1][1] = X_buf[1][0]; X_buf[1][0] = *X_ptr; 
        X_ptr += 32;
        // load and compute A
        for (int i = 0; i < 32; i++) {
            // consume buffered A
            float x_calc = __shfl_sync(0xffffffff, X_buf[0][2], i);
            if (x_calc != 0.0f) sum += A_buf[0][i] * x_calc;
            x_calc = __shfl_sync(0xffffffff, X_buf[1][2], i);
            if (x_calc != 0.0f) sum += A_buf[1][i] * x_calc;

            // load new A
            float x_load = __shfl_sync(0xffffffff, X_buf[0][1], i);
            if (x_load != 0.0f) A_buf[0][i] = *A_ptr;
            x_load = __shfl_sync(0xffffffff, X_buf[1][1], i);
            if (x_load != 0.0f) A_buf[1][i] = *(A_ptr + 32 * 32);
            A_ptr += 32;
        }
        A_ptr += 32 * 32;
    }

    // after pipeline 
    // tail stage 0 (no need to load new X)
    X_buf[0][2] = X_buf[0][1]; X_buf[0][1] = X_buf[0][0];
    X_buf[1][2] = X_buf[1][1]; X_buf[1][1] = X_buf[1][0];
    for (int i = 0; i < 32; i++) {
        // consume buffered A
        float x_calc = __shfl_sync(0xffffffff, X_buf[0][2], i);
        if (x_calc != 0.0f) sum += A_buf[0][i] * x_calc;
        x_calc = __shfl_sync(0xffffffff, X_buf[1][2], i);
        if (x_calc != 0.0f) sum += A_buf[1][i] * x_calc;

        // load new A
        float x_load = __shfl_sync(0xffffffff, X_buf[0][1], i);
        if (x_load != 0.0f) A_buf[0][i] = *A_ptr;
        x_load = __shfl_sync(0xffffffff, X_buf[1][1], i);
        if (x_load != 0.0f) A_buf[1][i] = *(A_ptr + 32 * 32);
        A_ptr += 32;
    }
    A_ptr += 32 * 32;

    // tail stage 1 (no need to load new A)
    X_buf[0][2] = X_buf[0][1];
    X_buf[1][2] = X_buf[1][1];
    for (int i = 0; i < 32; i++) {
        float x_calc = __shfl_sync(0xffffffff, X_buf[0][2], i);
        if (x_calc != 0.0f) sum += A_buf[0][i] * x_calc;
        x_calc = __shfl_sync(0xffffffff, X_buf[1][2], i);
        if (x_calc != 0.0f) sum += A_buf[1][i] * x_calc;
    }

    __shared__ float reduce_sum[3][32];
    if (warp_id != 0)
        reduce_sum[warp_id - 1][lane_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        for (int i = 0;i < 3; i++){
            sum += reduce_sum[i][lane_id];
        }
        Y[blockIdx.x * 32 + lane_id] = sum;
    }
}

void asp_gemv_gpu(int M, int N, float *A_host, float *X_host, float *Y_host, int version) {
    dim3 block(128, 1, 1);
    dim3 grid(N / 32, 1, 1);

    ASPMatrix asp(M, N, A_host);

    const size_t size_A = asp.ValuesSize() * sizeof(float);
    float *A_device;
    CUDA_CHECK(cudaMalloc((void **)&A_device, size_A));

    const size_t size_X = M * sizeof(float);
    const size_t size_Y = N * sizeof(float);

    float *X_device, *Y_device;
    CUDA_CHECK(cudaMalloc((void **)&X_device, size_X));
    CUDA_CHECK(cudaMalloc((void **)&Y_device, size_Y));

    CUDA_CHECK(cudaMemcpy(A_device, asp.GetValues(), size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(X_device, X_host, size_X, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaDeviceSynchronize());
    switch (version)
    {
        case 0:
            TIME_KERNEL((asp_kernel_v0<<<grid, block>>>(
                M, N, 
                A_device, X_device, Y_device
            )));
            break;
        case 1:
            TIME_KERNEL((asp_kernel_v1<<<grid, block>>>(
                M, N, 
                A_device, X_device, Y_device
            )));
            break;
        case 2:
            TIME_KERNEL((asp_kernel_v2<<<grid, block>>>(
                M, N, 
                A_device, X_device, Y_device
            )));
            break;
        default:
            break;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(Y_host, Y_device, size_Y, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(A_device));
    CUDA_CHECK(cudaFree(X_device));
    CUDA_CHECK(cudaFree(Y_device));
    return;
}