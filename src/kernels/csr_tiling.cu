#include "kernel.hpp"
#include "matrix_csr.hpp"
#include <stdio.h>


// using CSR format, naive
__global__ void csr_tiling_kernel(
    int M, int N, 
    float *A_vals, int A_vals_size, 
    int *A_row_ptrs, int A_row_ptrs_size,
    uint32_t *A_bmp,
    float *X, float *Y
) {
    // block size: 32 * 32 threads

    // buffers
    __shared__ float ldg_x_buffer[32];
    __shared__ float ldg_a_buffer[32][32];
    float acc = 0.0f;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int begin = A_row_ptrs[idx];
    // printf("[%d] global (%d, %d)\n", threadIdx.x, begin, end);

    int cur_begin = begin; 
    #pragma unroll
    for (int block_ks = 0; block_ks < M; block_ks += 32) {
        // load the x
        ldg_x_buffer[threadIdx.x] = X[block_ks + threadIdx.x];

        // load the column-major bitmap
        int ax = blockIdx.x * blockDim.x + threadIdx.x;
        int bmp_idx = ax * (M / 32) + block_ks / 32;
        uint32_t bmp = A_bmp[bmp_idx];

        // reset the A
        #pragma unroll 32
        for (int i = 0; i < 32; i++) {
            ldg_a_buffer[i][threadIdx.x] = 0.0f;
        }

        // count how many elemens in this tile
        int num_nz = __popc(bmp);
        int cur_end = cur_begin + num_nz;

        // printf("[%d] bmp %x, cur (%d, %d)\n", threadIdx.x, bmp,cur_begin, cur_end);

        // selectively load the A from global to smem
        int cnt = 0;
        #pragma unroll 32
        for (int offset = 0; offset < 32; offset++) {
            uint32_t mask = 1u << offset;
            if (mask & bmp) {
                ldg_a_buffer[offset][threadIdx.x] = A_vals[cur_begin + cnt];
                cnt += 1;
            }
        }

        // printf("[%d] cnt: %d, pop_cnt: %d\n", threadIdx.x, cnt, num_nz);

        __syncthreads();

        // compute
        #pragma unroll 32
        for (int i = 0; i < 32; i++) {
            acc += ldg_x_buffer[i] * ldg_a_buffer[i][threadIdx.x];
        }

        // update iters
        cur_begin += num_nz;
    }

    Y[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

std::vector<uint32_t> gen_csr_bitmap(int M, int N, float *matrix) {
    int num_of_u32 = M * N / 32;
    std::vector<uint32_t> bitmap(num_of_u32, 0);

    int bit_index = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            int word = bit_index / 32;
            int offset = bit_index % 32;
            if (matrix[j * N + i] != 0.0f) {
                bitmap[word] |= (1u << offset);
            }
            bit_index += 1;
        }
    }
    return bitmap;
}


void csr_tiling_gemv_gpu(int M, int N, float *A_host, float *X_host, float *Y_host) {
    dim3 block(32, 1, 1);
    dim3 grid(N / 32);

    CSRMatrix csr(M, N, A_host);
    auto csr_bmp = gen_csr_bitmap(M, N, A_host);

    const size_t size_A_row_ptrs = csr.RowPtrsSize() * sizeof(int);
    const size_t size_A_values = csr.ValuesSize() * sizeof(float);
    const size_t size_A_bitmap = csr_bmp.size() * sizeof(uint32_t);

    float *A_values_device;
    int *A_row_ptrs_device;
    uint32_t *A_bmp_device;
    CUDA_CHECK(cudaMalloc((void **)&A_values_device, size_A_values));
    CUDA_CHECK(cudaMalloc((void **)&A_row_ptrs_device, size_A_row_ptrs));
    CUDA_CHECK(cudaMalloc((void **)&A_bmp_device, size_A_bitmap));

    const size_t size_X = M * sizeof(float);
    const size_t size_Y = N * sizeof(float);

    float *X_device, *Y_device;
    CUDA_CHECK(cudaMalloc((void **)&X_device, size_X));
    CUDA_CHECK(cudaMalloc((void **)&Y_device, size_Y));

    CUDA_CHECK(cudaMemcpy(A_values_device, csr.GetValues(), size_A_values, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(A_row_ptrs_device, csr.GetRowPtrs(), size_A_row_ptrs, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(A_bmp_device, csr_bmp.data(), size_A_bitmap, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(X_device, X_host, size_X, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaDeviceSynchronize());

    // call the kernel
    TIME_KERNEL((csr_tiling_kernel<<<grid, block>>>(
        M, N, 
        A_values_device, csr.ValuesSize(),
        A_row_ptrs_device, csr.RowPtrsSize(),
        A_bmp_device,
        X_device, Y_device
    )));
    
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(Y_host, Y_device, size_Y, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(A_values_device));
    CUDA_CHECK(cudaFree(A_row_ptrs_device));
    CUDA_CHECK(cudaFree(X_device));
    CUDA_CHECK(cudaFree(Y_device));

    return;
}