#include "awsp.hpp"
#include "kernel.hpp"

__global__ void awsp_kernel_v0(
    int M, int N,
    int nz_bk_max,
    uint32_t *bitmaps,
    float* A_vals,
    float *X, float *Y
) {
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    uint32_t curr_mask = (1u << lane_id);
    uint32_t prev_mask = (lane_id == 0) ? 0 : ((1u << lane_id) - 1);

    float sum = 0;
    // X pointer: warp offset
    float *X_ptr = X + M/4*warp_id + lane_id;
    // A pointer: block offset
    float *A_ptr = A_vals + (blockIdx.x*M/32 + M/32*warp_id/4) * nz_bk_max;
    // bitmap pointer: same
    uint32_t *B_ptr = bitmaps + (blockIdx.x*M/32 + M/32*warp_id/4) * 32 + lane_id;

    for (int out_bk = 0; out_bk < M / 4; out_bk += 32) {
        uint32_t bitmap = *B_ptr;
        float x = *X_ptr;
        int A_val_bk_offset = 0;

        #pragma unroll
        for (int i = 0; i < 32; i++) {
            float cur_x = __shfl_sync(0xffffffff, x, i);
            uint32_t cur_bitmap = __shfl_sync(0xffffffff, bitmap, i);
            if (cur_x != 0.0f) {
                if (cur_bitmap & curr_mask) {
                    int A_val_row_offset = __popc(cur_bitmap & prev_mask);
                    float a = A_ptr[A_val_bk_offset + A_val_row_offset];
                    sum += cur_x * a;
                }
            }
            A_val_bk_offset += __popc(cur_bitmap);
        }
        A_ptr += nz_bk_max;
        B_ptr += 32;
        X_ptr += 32;
    }

    // reduction
    __shared__ float reduce_sum[3][32];
    if (warp_id != 0)
        reduce_sum[warp_id - 1][lane_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        for (int i = 0; i < 3; i++) {
            sum += reduce_sum[i][lane_id];
        }
        Y[blockIdx.x * 32 + lane_id] = sum;
    }
}

void awsp_gemv_gpu(int M, int N, float *A_host, float *X_host, float *Y_host, int version) {
    dim3 block(128, 1, 1);
    dim3 grid(N / 32, 1, 1);

    AWSPMatrix awsp(M, N, A_host);

    const size_t size_A_val = awsp.ValuesSize() * sizeof(float);
    const size_t size_A_bmp = awsp.BitmapsSize() * sizeof(uint32_t);

    float *A_val_dev;
    uint32_t *A_bmp_dev;
    CUDA_CHECK(cudaMalloc((void **)&A_val_dev, size_A_val));
    CUDA_CHECK(cudaMalloc((void **)&A_bmp_dev, size_A_bmp));

    const size_t size_X = M * sizeof(float);
    const size_t size_Y = N * sizeof(float);

    float *X_device, *Y_device;
    CUDA_CHECK(cudaMalloc((void **)&X_device, size_X));
    CUDA_CHECK(cudaMalloc((void **)&Y_device, size_Y));

    CUDA_CHECK(cudaMemcpy(A_val_dev, awsp.GetValues(), size_A_val, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(A_bmp_dev, awsp.GetBitmaps(), size_A_bmp, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(X_device, X_host, size_X, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaDeviceSynchronize());

    switch (version)
    {
        case 0:
            TIME_KERNEL((awsp_kernel_v0<<<grid, block>>>(
                M, N, 
                awsp.nz_bk_max_,
                A_bmp_dev,
                A_val_dev,
                X_device, Y_device
            )));
            break;
        default:
            break;
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(Y_host, Y_device, size_Y, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(A_val_dev));
    CUDA_CHECK(cudaFree(A_bmp_dev));
    CUDA_CHECK(cudaFree(X_device));
    CUDA_CHECK(cudaFree(Y_device));
    return;
}