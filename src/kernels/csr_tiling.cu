#include "kernel.hpp"
#include "matrix_csr.hpp"
#include "tcsr.hpp"
#include <stdio.h>

// small tricks to accelerate :)
__device__ int warp_prefix_sum(int val) {
    unsigned mask = 0xffffffff; // full warp, 32 threads
    for (int offset = 1; offset < 32; offset <<= 1) {
        int n = __shfl_up_sync(mask, val, offset);
        if (threadIdx.x % 32 >= offset) val += n;
    }
    return val;
}

void __device__ print_smem(float *smem) {
    if (threadIdx.x == 0) {
        printf("smem [%f, %f, %f, %f, %f, %f, %f, %f]", smem[0], smem[1], smem[2], smem[3], smem[4], smem[5], smem[6], smem[7]);
    }
}


// using CSR format and tiling method
__global__ void csr_tiling_kernel(
    int M, int N, 
    float *A_vals, int A_vals_size, 
    int *A_blk_idxs, int A_blkidx_size,
    uint32_t *A_bmp,
    float *X, float *Y
) {
    // IDs
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32; 

    // buffers
    __shared__ float ldg_x_buffer[32];
    __shared__ float ldg_a_buffer[32][32];
    __shared__ uint32_t ldg_bmp_buffer[32];
    // count how many value in this tile
    __shared__ int num_nz[32];
    // count how many value before this tile
    __shared__ int sum_nz[32];
    float acc = 0.0f;

    // printf("[%d] global (%d, %d)\n", threadIdx.x, begin, end);
    #pragma unroll
    for (int block_ks = 0; block_ks < M; block_ks += 32) {

        if (warp_id == 0) {
            // load the x
            ldg_x_buffer[lane_id] = X[block_ks + lane_id];
        }
        
        if (warp_id == 1) {
            // load the block-wise column-major bitmap
            int ax = blockIdx.x * blockDim.x + lane_id;
            int bmp_idx = blockIdx.x * M + block_ks + lane_id;
            ldg_bmp_buffer[lane_id] = A_bmp[bmp_idx];
            num_nz[lane_id] = __popc(ldg_bmp_buffer[lane_id]);
            sum_nz[lane_id] = warp_prefix_sum(num_nz[lane_id]);
        }

        if (warp_id == 2) {
            // reset the A
            #pragma unroll 32
            for (int i = 0; i < 32; i++) {
                ldg_a_buffer[i][lane_id] = 0.0f;
            }
        }
        
        __syncthreads();

        // load the A from global to share memory
        uint32_t one_bit_mask = 1u << lane_id;
        uint32_t length_mask = (lane_id == 0) ? 0 : ((1u << lane_id) - 1);
        // the begin idx in values of this block
        int block_begin_idx = A_blk_idxs[blockIdx.x * (M / 32) + block_ks / 32];
        // if (threadIdx.x == 0) {
        //     printf("[%d] block_idx %d\n", blockIdx.x, block_begin_idx);
        // }
        for (int i = 0; i < 4; i++) {
            int tile_id = 4 * warp_id + i;
            int begin = sum_nz[tile_id] - num_nz[tile_id];
            uint32_t bitmap = ldg_bmp_buffer[tile_id];
            int cnt = __popc(bitmap & length_mask);
            if (one_bit_mask & bitmap) {
                ldg_a_buffer[lane_id][tile_id] = A_vals[block_begin_idx + begin + cnt];
            }
        }

        __syncthreads();

        // compute
        // todo: use parrelled compute
        if (warp_id == 0) {
            #pragma unroll 32
            for (int i = 0; i < 32; i++) {
                acc += ldg_x_buffer[i] * ldg_a_buffer[i][lane_id];
                // if (blockIdx.x == 1 && threadIdx.x == 0) {
                //     printf("thread [%2d][%2d] in block %d, (%f)*(%f), cur_acc: %f\n", warp_id, lane_id, blockIdx.x, ldg_x_buffer[i], ldg_a_buffer[i][lane_id], acc);
                // }
            }
        }

        __syncthreads();
    }

    __syncthreads();

    if (warp_id == 0) {
        // printf("Y[%d] acc %f\n", blockIdx.x * blockDim.x + lane_id, acc);
        Y[blockIdx.x * 32 + lane_id] = acc;
    }
}

void csr_tiling_gemv_gpu(int M, int N, float *A_host, float *X_host, float *Y_host) {
    dim3 block(256, 1, 1);
    dim3 grid(N / 32);

    TCSRMatrix tcsr(M, N, A_host);

    const size_t size_A_blkidx = tcsr.BlkIdxSize() * sizeof(int);
    const size_t size_A_values = tcsr.ValuesSize() * sizeof(float);
    const size_t size_A_bitmap = tcsr.BitmapsSize() * sizeof(uint32_t);

    float *A_values_device;
    int *A_blkidx_device;
    uint32_t *A_bmp_device;
    CUDA_CHECK(cudaMalloc((void **)&A_values_device, size_A_values));
    CUDA_CHECK(cudaMalloc((void **)&A_blkidx_device, size_A_blkidx));
    CUDA_CHECK(cudaMalloc((void **)&A_bmp_device, size_A_bitmap));

    const size_t size_X = M * sizeof(float);
    const size_t size_Y = N * sizeof(float);

    float *X_device, *Y_device;
    CUDA_CHECK(cudaMalloc((void **)&X_device, size_X));
    CUDA_CHECK(cudaMalloc((void **)&Y_device, size_Y));

    CUDA_CHECK(cudaMemcpy(A_values_device, tcsr.GetValues(), size_A_values, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(A_blkidx_device, tcsr.GetBlkIdx(), size_A_blkidx, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(A_bmp_device, tcsr.GetBitmaps(), size_A_bitmap, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(X_device, X_host, size_X, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaDeviceSynchronize());

    // call the kernel
    TIME_KERNEL((csr_tiling_kernel<<<grid, block>>>(
        M, N, 
        A_values_device, tcsr.ValuesSize(),
        A_blkidx_device, tcsr.BlkIdxSize(),
        A_bmp_device,
        X_device, Y_device
    )));
    
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(Y_host, Y_device, size_Y, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(A_values_device));
    CUDA_CHECK(cudaFree(A_blkidx_device));
    CUDA_CHECK(cudaFree(X_device));
    CUDA_CHECK(cudaFree(Y_device));

    return;
}